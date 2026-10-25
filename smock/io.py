"""Save and load a galaxy's spectra and photometry, and its images, as .npz files.

An .npz is numpy's zip of named arrays; np.load opens it and reads each array by name.
Neither file needs pickle, or smock, to read.

The spectrum file holds three arrays:

  spectrum    one row per wavelength (structured array). Wavelength and frequency in the
              rest and observed frames, then for each component C (e.g. "total",
              "stellar_total", "reprocessed", "AGN_total", "total_dust_emission"):
                C_Lnu_rest_erg_s_Hz   rest-frame luminosity density
                C_nuLnu_rest_erg_s    nu_rest * L_nu (the old pipeline used nu_obs)
                C_fnu_obs_nJy         observed flux density, after the IGM
  photometry  one row per filter: filter code, observed pivot wavelength, then for each
              component C_fnu_nJy and C_mag_AB (from the real filter curves)
  meta        a JSON string: what was observed and how (simulation, galaxy, redshift,
              crop, dust, grid, ...) plus the synthesizer version and smock commit

    save_spectrum(path, seds, photometry, meta)
    spectrum, photometry, meta = load_spectrum(path)

The images file holds:
  filters     the filter codes, in the order of each image stack's axis 0
  <stage>     one (n_filters, n, n) float array per stage [nJy per pixel], e.g.
              "total_raw", "total_psf", "total_noisy", "stellar_psf", "AGN_psf"; indexed (filter, x, y), so
              display one with plt.imshow(img.T, origin="lower")
  segmap      (n, n) int32 segmentation map (0 = background, 1, 2, ... = sources)
  meta        a JSON string, as for the spectrum

    save_images(path, filter_codes, {stage: stack}, segmap, meta)
    images, meta = load_images(path)       # images: {"filters": ..., stage: ..., "segmap": ...}

Column names carry their units. Values are read from synthesizer with .to(...), never
in-place conversions (CLAUDE.md).
"""
import datetime
import json
import subprocess
from pathlib import Path

import numpy as np

AB_ZERO_POINT_NJY = 3631e9


def ab_mag(fnu_njy):
    """Flux density [nJy] -> AB magnitude; NaN where the flux is not positive."""
    f = np.asarray(fnu_njy, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(f > 0, -2.5 * np.log10(f / AB_ZERO_POINT_NJY), np.nan)


def spectrum_table(seds):
    """Structured array of the spectra in `seds` ({component: Sed}, all on one wavelength
    grid, with observed fluxes already computed)."""
    first = next(iter(seds.values()))
    columns = {
        "lam_rest_AA": first.lam.to("angstrom").value,
        "lam_obs_AA": first.obslam.to("angstrom").value,
        "nu_rest_Hz": first.nu.to("Hz").value,
        "nu_obs_Hz": first.obsnu.to("Hz").value,
    }
    for label, sed in seds.items():
        if not np.array_equal(sed.lam.to("angstrom").value, columns["lam_rest_AA"]):
            raise ValueError(f"{label} is on a different wavelength grid")
        lnu = sed.lnu.to("erg/s/Hz").value
        columns[f"{label}_Lnu_rest_erg_s_Hz"] = lnu
        columns[f"{label}_nuLnu_rest_erg_s"] = columns["nu_rest_Hz"] * lnu
        columns[f"{label}_fnu_obs_nJy"] = sed.fnu.to("nJy").value
    return _structured(columns)


def photometry_table(photometry):
    """Structured array of band photometry ({component: PhotometryCollection}, one filter
    set for all)."""
    first = next(iter(photometry.values()))
    filters = list(first.filters)
    columns = {
        "filter": np.array([f.filter_code for f in filters]),
        "pivot_obs_AA": np.array([f.pivwv().to("angstrom").value for f in filters]),
    }
    for label, phot in photometry.items():
        fnu = np.array([phot[f.filter_code].to("nJy").value for f in filters])
        columns[f"{label}_fnu_nJy"] = fnu
        columns[f"{label}_mag_AB"] = ab_mag(fnu)
    return _structured(columns)


def _structured(columns):
    dtype = [(name, values.dtype) for name, values in columns.items()]
    out = np.empty(len(next(iter(columns.values()))), dtype=dtype)
    for name, values in columns.items():
        out[name] = values
    return out


def _git_commit():
    """The smock commit these results were made with, or None outside a git checkout."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _meta_json(meta):
    """meta plus provenance (time, synthesizer version, smock commit), as a JSON string array."""
    import synthesizer       # only for its version; imported here so importing smock stays light

    meta = {**meta, "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "synthesizer_version": synthesizer.__version__, "smock_commit": _git_commit()}
    return np.array(json.dumps(meta, default=_json_default))


def save_spectrum(path, seds, photometry, meta):
    """Write seds ({component: Sed}) and photometry ({component: PhotometryCollection}) to
    `path` (.npz), with `meta` (a JSON-serialisable dict) plus provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, spectrum=spectrum_table(seds), photometry=photometry_table(photometry),
             meta=_meta_json(meta))
    return path


def _json_default(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f"meta value of type {type(x).__name__} is not JSON-serialisable")


def load_spectrum(path):
    """(spectrum, photometry, meta) from a file written by save_spectrum."""
    with np.load(path, allow_pickle=False) as f:
        return f["spectrum"], f["photometry"], json.loads(str(f["meta"]))


def image_stack(images, filter_codes, unit="nJy"):
    """(n_filters, n, n) array of a synthesizer ImageCollection's images, in `unit` per pixel."""
    return np.stack([(images[fc].arr * images[fc].units).to(unit).value for fc in filter_codes])


def save_images(path, filter_codes, stages, segmap, meta):
    """Write image stacks to `path` (.npz).

    filter_codes: the filters, in axis-0 order
    stages:       {name: (n_filters, n, n) array [nJy per pixel]}, e.g. from image_stack
    segmap:       (n, n) segmentation labels, or None
    meta:         a JSON-serialisable dict (pixel scale, PSFs, noise, ...); provenance is added
    """
    arrays = {"filters": np.array(filter_codes)}
    for name, stack in stages.items():
        stack = np.asarray(stack, dtype=np.float64)
        if stack.ndim != 3 or stack.shape[0] != len(filter_codes):
            raise ValueError(f"{name}: expected ({len(filter_codes)}, n, n), got {stack.shape}")
        arrays[name] = stack
    if segmap is not None:
        arrays["segmap"] = np.asarray(segmap, dtype=np.int32)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays, meta=_meta_json(meta))
    return path


def load_images(path):
    """(images, meta) from a file written by save_images; images maps each array's name to it."""
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files if k != "meta"}, json.loads(str(f["meta"]))
