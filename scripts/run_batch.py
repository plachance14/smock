"""Mock observations of many galaxies in one job: the large-galaxy notebook's steps, in a loop.

    python scripts/run_batch.py --snap 47 --sidx-file my_galaxies.txt
    python scripts/run_batch.py --snap 47 --sidx-file my_galaxies.txt --task 3 --ntasks 8   # one share of the list

Each galaxy goes through the same steps as notebooks/walkthrough_large_galaxy.ipynb, with the
same settings (the block below uses the notebooks' names and defaults; edit it, or copy this file,
to change them). Its stars run in chunks sized to the memory this job has, as one chunk when the
galaxy fits, so small and large galaxies take the same route. The results match the large-galaxy
notebook's exactly and the walkthrough's to rounding (tests/test_batch.py checks both).

What a batch does differently from the notebooks, because no one is watching each galaxy:
  - the grids, emission model, filters, PSFs and instrument are made once per job, not per galaxy;
  - no figures, unless QUICK_LOOK is on (then one PNG per galaxy: the image beside the spectrum);
  - a galaxy that fails is logged with its traceback and the run goes on to the next one;
  - galaxies whose three files already exist are skipped, so a job that ran out of time can
    simply be submitted again (--redo makes them again);
  - galaxies with fewer than MIN_STARS stars in the crop are skipped, as in the old pipeline;
  - each task adds a line per galaxy to catalogue_task<k>.jsonl (its main measurements, or why
    it was skipped or failed, how long it took, the memory it used, and its files' names);
    scripts/merge_catalogues.py turns all the tasks' lines into one table, catalogue.csv.

The galaxy list (--sidx-file) is a text file of subhalo indices, one per line, or a .npy array
(e.g. from notebooks/make_subhalo_list.ipynb). With --ntasks N it is shuffled (with a fixed seed)
and dealt out, so each of the N tasks gets a similar mix of large and small galaxies. In a SLURM
array, leave out --task and --ntasks: they are taken from SLURM_ARRAY_TASK_ID and
SLURM_ARRAY_TASK_COUNT. Ask SLURM for plenty of memory: the chunk size follows the allocation,
and a galaxy that runs out of memory is lost. For example (on Vera; set your own account):

    #!/bin/bash
    #SBATCH --partition=RM
    #SBATCH --array=0-27              # 28 tasks, each working through its share of the list
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=1
    #SBATCH --mem=32G                 # per task, generous: the z = 6 test galaxies (5-21k stars) peak at 7-18 GB
    #SBATCH --time=12:00:00
    python scripts/run_batch.py --snap 47 --sidx-file my_galaxies.txt

Galaxies with many stars still run, in chunks, but slowly and with the data cubes' memory on top
(about 1 GB each for 3" images: see the large-galaxy notebook's step 7), so give a list of large
galaxies fewer tasks with more memory each (e.g. --mem=120G). Galaxies are skipped when their files
already exist, so a task that ran out of time can be submitted again.
"""
import argparse
import json
import os
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
from unyt import K, Myr, angstrom, arcsec, km, kpc, nJy, s, unyt_array

from synthesizer.emission_models import (AttenuatedEmission, BlackHoleEmissionModel, DustEmission,
                                         EmissionModel, LOSStellarEmission)
from synthesizer.emission_models.attenuation import Inoue14
from synthesizer.emission_models.generators.dust import Greybody
from synthesizer.emission_models.transformers import CoveringFraction, EscapingFraction, PowerLaw
from synthesizer.emissions import Sed
from synthesizer.grid import Grid
from synthesizer.imaging import Image, ImageCollection, SpectralCube
from synthesizer.instruments import FilterCollection, PhotometricImager
from synthesizer.kernel_functions import Kernel
from synthesizer.particle import BlackHoles, Galaxy, Gas, Stars
from synthesizer.pipeline.pipeline_utils import accumulate_pipeline_results_from_child

from smock import agn, analysis, config, dust, imaging, memory
from smock.data import code_mass_to_msun, crop, kappa_code_to_physical, load_galaxy, open_snapshot, subhalo_row
from smock.dust import WindowedPowerLaw
from smock.io import ab_mag, image_stack, save_images, save_spectrum

# ======================================================================================
# Settings: the notebooks' names and defaults (see the notebooks for what each one does)
# ======================================================================================
# 1. paths and the images
PIG_DIR = config.EXTERNAL["astrid_pig_dir"]     # the full ASTRID snapshots on Vera; config.TEST_CUTOUT_PIG_DIR for the test data
SIM = "ASTRID"
OBSERVED_REDSHIFT = None
GRID_DIR = config.GRID_DIR / "sps/photoionised/bpass"
GRID_NAME = "bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps"
INCLUDE_AGN = True
AGN_GRID_DIR = config.GRID_DIR / "agn/photoionised/agn-cloudy"
AGN_GRID_NAME = "agn-cloudy_T20_aox-1.40_auv-0.50_cloudy-standard_blr_core"
OUT_DIR = config.OUTPUT_DIR / "batch"           # or --out-dir
PARTICLES = "group"
PIXEL = 0.03                    # arcsec
FOV = 3.0                       # arcsec
BUFFER_PIX = 5
IMAGE_FOV = FOV + 2 * BUFFER_PIX * PIXEL
# 2. the crop
CROP_SHAPE = "prism"
CROP_UNIT = "arcsec"
KERNEL_MARGIN_PIX = 10
CROP_SIZE = IMAGE_FOV / 2 + KERNEL_MARGIN_PIX * PIXEL
GAS_MARGIN_CKPCH = 5.0
MIN_STARS = 50                  # batch only: skip galaxies with fewer stars in the crop (the old pipeline's 50)
# 3. stars, gas and black holes
STAR_MASS = "initial_masses"
DUST_MODEL = "code_units"
EFFICIENCY = 0.1
DISC_TEMPERATURE = "synthesizer"
DISC_RADIUS = 3.0
DISC_RADIUS_UNIT = "r_isco"
R_ISCO = 6.0
AGN_GRID_POINT = dict(ionisation_parameter=10**-1.5, hydrogen_density=1e10, column_density=23.0)
AGN_U_FROM = "grid_point"
COVERING_FRACTION = 1.0
COVERING_MODE = "random"
COVERING_SEED = "bh_id_and_snapshot"
# 4. line-of-sight dust (KAPPA_CODE depends on the snapshot: see kappa_for)
KAPPA_CODE_LOW, KAPPA_CODE_HIGH, KAPPA_CODE_JOIN = 10**2.9, 10**4.1, (2.0, 4.0)
KAPPA_DUST = 0.36
KAPPA_BC = 1.0
Z_BC = 0.01
SPH_KERNEL = "quintic"
# 5. the emission model
YOUNG_AGE = 10 * Myr
DUST_SLOPE_BC = -1.0
DUST_SLOPE_ISM = -1.0           # or -0.7 (Charlot & Fall 2000; see the notebooks' step 5)
AGN_DUST_SLOPE = DUST_SLOPE_ISM
AGN_DUST_WINDOW = (20 * angstrom, 1e6 * angstrom)
DUST_TEMPERATURE = 100 * K
DUST_BETA = 2.0
# 6. the instrument
IGM = Inoue14
FILTER_CODES = [f"JWST/NIRCam.{f}" for f in imaging.NIRCAM_WIDE]
PSF_SOURCE = "stpsf"
PSF_PRODUCT = "DET_SAMP"
MOFFAT_FWHM = imaging.DECALS_DESIGN_SEEING_ARCSEC
MOFFAT_BETA = imaging.MOFFAT_BETA
NOISE_PRESET = "JADES-Medium"
NOISE_MODE = imaging.SURVEY_DEPTH_KINDS[NOISE_PRESET]
SIGMA_PIX = 1.0
R_AP = imaging.SURVEY_APERTURE_RADIUS_ARCSEC
# 7. chunks
CHUNK_STARS = "auto"
MEMORY_LIMIT_GB = None
MEMORY_SAFETY = 0.5
TRIAL_STARS = 500
APERTURE_SPECTRUM = "cube"      # or "two_pass", "preset"
PRESET_APERTURE_ARCSEC = 0.5
# 8. noise
SEED = 1
# 10. segmentation
DETECT_FILTER = "JWST/NIRCam.F444W"
DETECT_NSIGMA = 1.5
DETECT_NPIXELS = 10
DETECT_CONNECTIVITY = 4
DEBLEND_NLEVELS = 32
DEBLEND_CONTRAST = 0.001
IMAGE_CENTRE = "catalogue"
DETECT_SMOOTH = None
# 11. morphology
RUN_STATMORPH = True
# 12. measurements in the source
APERTURE = "segment"
APERTURE_R_HALF = 2.0
SFR_AGE = 100 * Myr
COMPACTNESS_RADII = (0.1, 0.2)
COMPACTNESS_METHOD = "exact"
IMAGE_PHOTOMETRY = ()
# 15. LRD selection (into the catalogue)
LRD_SELECTION = "ambra"         # or "kokorev24", or None
LRD_MAX_OFFSET_CKPCH = 10.0
LRD_PHOTOMETRY = "particles"
# batch only
QUICK_LOOK = False              # one PNG per galaxy: the image beside the spectrum


def kappa_for(snap):
    """The ISM opacity for this snapshot's DUST_MODEL [pc^2 per Msun of metals, or of dust], and KAPPA_CODE."""
    kappa_code = dust.kappa_code_for_redshift(snap.redshift, low=KAPPA_CODE_LOW, high=KAPPA_CODE_HIGH, join=KAPPA_CODE_JOIN)
    kappa = kappa_code_to_physical(kappa_code, snap.a, snap.h) if DUST_MODEL == "code_units" else KAPPA_DUST
    return kappa, kappa_code


# ======================================================================================
# Made once per job (steps 5-7 of the notebooks)
# ======================================================================================
class Job:
    """The grids, emission model, filters, PSFs, instrument and kernel, shared by every galaxy."""

    def __init__(self):
        self.grid = Grid(GRID_NAME, grid_dir=str(GRID_DIR))
        grid = self.grid
        # ---- stars ----
        self.stellar = LOSStellarEmission(grid, young_upper_limit=YOUNG_AGE, birth_cloud_slope=DUST_SLOPE_BC,
                                          ism_slope=DUST_SLOPE_ISM)
        self.stellar_dust = DustEmission(
            dust_emission_model=Greybody(DUST_TEMPERATURE, emissivity=DUST_BETA),
            dust_lum_intrinsic=self.stellar["reprocessed"],
            dust_lum_attenuated=self.stellar,
            emitter="stellar",
            label="stellar_dust_emission",
        )
        per_particle = [self.stellar, self.stellar_dust]
        totals, dust_parts, reprocessed_parts = [self.stellar], [self.stellar_dust], [self.stellar["reprocessed"]]
        # ---- AGN (built once; run only for galaxies with black holes) ----
        self.agn_dust = None
        if INCLUDE_AGN:
            agn_grid = Grid(AGN_GRID_NAME, grid_dir=str(AGN_GRID_DIR), new_lam=grid.lam)
            los, f_attr = ("agn_los_covered",), ("agn_covering_fraction",)
            disc = BlackHoleEmissionModel("AGN_incident", grid=agn_grid, extract="incident")
            through_clouds = BlackHoleEmissionModel("AGN_transmitted_full", grid=agn_grid, extract="transmitted")
            clouds = BlackHoleEmissionModel("AGN_nebular_full", grid=agn_grid, extract="nebular")
            agn_reprocessed = BlackHoleEmissionModel("AGN_reprocessed", combine=(
                BlackHoleEmissionModel("AGN_escaped", apply_to=disc, transformer=EscapingFraction(covering_attrs=los)),
                BlackHoleEmissionModel("AGN_transmitted", apply_to=through_clouds, transformer=CoveringFraction(covering_attrs=los)),
                BlackHoleEmissionModel("AGN_nebular", apply_to=clouds, transformer=CoveringFraction(covering_attrs=f_attr)),
            ))
            agn_dust_curve = (PowerLaw(slope=AGN_DUST_SLOPE) if AGN_DUST_WINDOW is None
                              else WindowedPowerLaw(AGN_DUST_SLOPE, *AGN_DUST_WINDOW))
            agn_total = AttenuatedEmission(dust_curve=agn_dust_curve, apply_to=agn_reprocessed,
                                           emitter="blackhole", tau_v="tau_v", label="AGN_total")
            self.agn_dust = DustEmission(
                dust_emission_model=Greybody(DUST_TEMPERATURE, emissivity=DUST_BETA),
                dust_lum_intrinsic=agn_reprocessed,
                dust_lum_attenuated=agn_total,
                emitter="blackhole",
                label="AGN_dust_emission",
            )
            per_particle += [agn_total, self.agn_dust]
            totals.append(agn_total)
            dust_parts.append(self.agn_dust)
            reprocessed_parts.append(agn_reprocessed)
        # ---- the galaxy ----
        total_dust = EmissionModel("total_dust_emission", combine=dust_parts, emitter="galaxy")
        reprocessed_total = EmissionModel("reprocessed_total", combine=reprocessed_parts, emitter="galaxy")
        self.model = EmissionModel("total", combine=(*totals, total_dust), emitter="galaxy",
                                   related_models=(reprocessed_total,))
        for m in per_particle:
            m.set_per_particle(True)
        self.model.set_save(False, set_all=True)
        for label in self.saved(INCLUDE_AGN):
            self.model[label].set_save(True)

        # ---- filters, PSFs, noise and the instrument (step 6) ----
        self.filters = FilterCollection(filter_codes=FILTER_CODES)
        self.psfs, self.depths = {}, {}
        for fc in FILTER_CODES:
            f = fc.split(".")[-1]
            if PSF_SOURCE == "stpsf":
                self.psfs[fc], header = imaging.load_stpsf_psf(config.PSF_DIR, f, PSF_PRODUCT)
            elif PSF_SOURCE == "gaussian":
                self.psfs[fc] = imaging.gaussian_psf(imaging.NIRCAM_FWHM_ARCSEC[f], PIXEL)
            else:
                fwhm = MOFFAT_FWHM[f] if isinstance(MOFFAT_FWHM, dict) else MOFFAT_FWHM
                self.psfs[fc] = imaging.moffat_psf(fwhm, PIXEL, MOFFAT_BETA)
            if NOISE_MODE == "fixed_sigma":
                sigma_pix = SIGMA_PIX
            else:
                depth_njy = imaging.ab_mag_to_njy(imaging.SURVEY_DEPTHS_5SIGMA[NOISE_PRESET][f])
                if NOISE_MODE == "aperture":
                    ee = header["EE_R0P1"] if PSF_SOURCE == "stpsf" else imaging.encircled_energy(self.psfs[fc], PIXEL, R_AP)
                    sigma_pix = imaging.sigma_pix_from_depth(depth_njy, 5, R_AP, PIXEL, ee)
                elif NOISE_MODE == "psf_weighted":
                    sigma_pix = imaging.sigma_pix_from_psf_depth(depth_njy, 5, self.psfs[fc])
                else:
                    sigma_pix = depth_njy / 5
            self.depths[fc] = 5 * sigma_pix * nJy
        self.instrument = PhotometricImager(label="NIRCam_0.03", filters=self.filters, resolution=PIXEL * arcsec,
                                            psfs=self.psfs, depth=self.depths, snrs=5.0)
        # chunk images are added up, and the noise goes on the sum once: a copy without noise
        self.imager = PhotometricImager(label=self.instrument.label, filters=self.filters,
                                        resolution=self.instrument.resolution, psfs=self.psfs)
        self.kernel = Kernel(SPH_KERNEL)

    @staticmethod
    def saved(agn_on):
        return ["total", "reprocessed_total", "total_dust_emission", "stellar_total", "reprocessed",
                "stellar_dust_emission"] + (["AGN_total", "AGN_reprocessed", "AGN_dust_emission"] if agn_on else [])


# ======================================================================================
# One galaxy (steps 2-15 of the large-galaxy notebook)
# ======================================================================================
class Skip(Exception):
    """A galaxy left out on purpose (too few stars), not a failure."""


def build_galaxy(snap, snap_number, sidx, z_obs):
    """Steps 2-3: load and crop the particles, then synthesizer's Stars, Gas and BlackHoles."""
    row = subhalo_row(snap, sidx)
    gal = load_galaxy(snap, sidx=row, particles=PARTICLES, observed_z=z_obs)
    m_star_subfind = float(code_mass_to_msun(snap.file["SubGroups/SubhaloMassType"][row:row + 1][0, 4], snap.h))
    gal = crop(gal, CROP_SIZE, CROP_UNIT, shape=CROP_SHAPE, gas_margin_ckpch=GAS_MARGIN_CKPCH)
    z = gal.observed_redshift
    S, G, B = gal.stars, gal.gas, gal.bhs
    if len(S["ages"]) < MIN_STARS:
        raise Skip(f"{len(S['ages'])} stars in the crop (< MIN_STARS = {MIN_STARS})")
    if DUST_MODEL == "code_units":
        dust_to_metal = 1.0
    else:
        dtm_z, dtm_age = dust.dtm_inputs(G, S)
        dust_to_metal = float(dust.dtm_vijayan19(dtm_z, dtm_age))
    stars = Stars(initial_masses=S[STAR_MASS], current_masses=S["current_masses"], ages=S["ages"],
                  metallicities=S["metallicities"], coordinates=S["coordinates"],
                  smoothing_lengths=S["smoothing_lengths"], redshift=z)
    gas = Gas(masses=G["masses"], metallicities=G["metallicities"], coordinates=G["coordinates"],
              smoothing_lengths=G["smoothing_lengths"], dust_to_metal_ratio=dust_to_metal,
              star_formation_rates=G["star_formation_rates"], redshift=z)
    n_bh = len(B["ids"])
    agn_on = INCLUDE_AGN and n_bh > 0
    black_holes, ids, los_covered, t_disc, bh_grid_point = None, [], np.zeros(0), None, dict(AGN_GRID_POINT)
    if n_bh > 0:
        mass, mdot, ids = B["masses"], B["accretion_rates"], B["ids"]
        if DISC_TEMPERATURE == "synthesizer":
            t_disc = agn.synthesizer_blue_bump_temperature(mass, mdot)
        elif DISC_TEMPERATURE == "T_max":
            t_disc = agn.disc_max_temperature(mass, mdot, r_isco=R_ISCO)
        elif DISC_TEMPERATURE == "radius":
            t_disc = agn.disc_temperature_at(mass, mdot, DISC_RADIUS, unit=DISC_RADIUS_UNIT, r_isco=R_ISCO)
        else:
            t_disc = np.full(n_bh, float(DISC_TEMPERATURE)) * K
        if AGN_U_FROM == "old_pipeline":
            incident_grid = config.GRID_DIR / "agn/incident/agn-cloudy" / (AGN_GRID_NAME.split("_cloudy-")[0] + ".hdf5")
            bh_grid_point["ionisation_parameter"] = agn.old_pipeline_ionisation_parameter(
                agn.ionising_photons_per_lbol(incident_grid, t_disc))
        if COVERING_MODE == "weighted_combination":
            los_covered = np.full(n_bh, COVERING_FRACTION)
        else:
            seeds = [None if COVERING_SEED is None else [int(i), snap_number] for i in ids]
            los_covered = np.array([float(np.random.default_rng(sd).random() < COVERING_FRACTION) for sd in seeds])
        black_holes = BlackHoles(
            masses=mass, accretion_rates=mdot, epsilons=EFFICIENCY, coordinates=B["coordinates"], ids=ids,
            smoothing_lengths=unyt_array(np.full(n_bh, 1e-3), "kpc"),
            blue_bump_temperature=t_disc.to("K").value,
            **{name: value * np.ones(n_bh) for name, value in bh_grid_point.items()},
            agn_covering_fraction=np.full(n_bh, COVERING_FRACTION),
            agn_los_covered=los_covered,
            redshift=z,
        )
    galaxy = Galaxy(stars=stars, gas=gas, black_holes=black_holes, redshift=z,
                    centre=unyt_array([0.0, 0.0, 0.0], "kpc"))
    grid_flags = 0
    if agn_on:
        report = agn.grid_check(AGN_GRID_DIR / f"{AGN_GRID_NAME}.hdf5", blue_bump_temperature=t_disc,
                                **{k: v * np.ones(n_bh) for k, v in bh_grid_point.items()})
        grid_flags = sum(bool(r["outside"] or r["flags"]) for r in report)
    return dict(gal=gal, galaxy=galaxy, z=z, agn_on=agn_on, ids=ids, los_covered=los_covered, n_bh=n_bh,
                m_star_subfind=m_star_subfind, dust_to_metal=dust_to_metal, grid_flags=grid_flags)


def observe(job, snap, snap_number, sidx, out_dir):
    """One galaxy from start to finish; returns its catalogue row (a dict)."""
    t_start = time.time()
    g = build_galaxy(snap, snap_number, sidx, OBSERVED_REDSHIFT)
    gal, galaxy, z, agn_on = g["gal"], g["galaxy"], g["z"], g["agn_on"]
    cosmo, kernel, filters, grid, model = snap.cosmo, job.kernel, job.filters, job.grid, job.model
    SAVED = job.saved(agn_on)

    # ---- step 4: line-of-sight dust ----
    kappa, kappa_code = kappa_for(snap)
    galaxy.get_stellar_los_tau_v(kappa=kappa, kernel=kernel)
    if agn_on:
        galaxy.get_black_hole_los_tau_v(kappa=kappa, kernel=kernel)
    galaxy.stars.young_tau_v = KAPPA_BC * galaxy.stars.metallicities / Z_BC

    # ---- step 7: how many stars at a time ----
    STAR_LABELS = [label for label in SAVED if model[label].emitter == "stellar"]
    BH_LABELS = [label for label in SAVED if model[label].emitter == "blackhole"]
    STAR_IMAGE_LABELS = ["stellar_total", "stellar_dust_emission"]
    BH_IMAGE_LABELS = ["AGN_total", "AGN_dust_emission"] if agn_on else []
    image_kw = dict(fov=IMAGE_FOV * arcsec, instrument=job.imager, kernel=kernel.get_kernel(), cosmo=cosmo)
    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z).to("kpc/arcsec").value
    cube_kw = dict(resolution=PIXEL * kpc_per_arcsec * kpc, fov=IMAGE_FOV * kpc_per_arcsec * kpc)
    cube_shape = (*SpectralCube(lam=grid.lam, **cube_kw).npix, len(grid.lam))
    cube_gb = np.prod(cube_shape) * 8 / 1e9
    cube_labels = STAR_LABELS if APERTURE_SPECTRUM == "cube" else []
    cube_units = {}
    reserved_gb = (len(cube_labels) + 1) * cube_gb if cube_labels else 0.0

    def run_chunk(chunk, weights=None, cubes=None):
        with np.errstate(over="ignore"):
            chunk.get_spectra(job.stellar_dust)
        chunk.get_observed_spectra(cosmo=cosmo, igm=IGM)
        chunk.get_photo_fnu(filters)
        chunk.stars.get_particle_photo_fnu(filters)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"\s*Coordinates must be centered")
            chunk.get_images_flux(*STAR_IMAGE_LABELS, **image_kw)
        for p in cubes or {}:
            cube = SpectralCube(lam=grid.lam, **cube_kw)
            cube.generate_data_cube_smoothed(chunk.stars.particle_spectra[p], coordinates=chunk.stars.centered_coordinates,
                                             smoothing_lengths=chunk.stars.smoothing_lengths,
                                             kernel=kernel.get_kernel(), kernel_threshold=1)
            cubes[p] += cube.arr
            cube_units[p] = cube.units
            del cube
        if weights is None:
            return None
        return {p: weights @ chunk.stars.particle_spectra[p].lnu.to("erg/s/Hz").value for p in STAR_LABELS}

    stars_only = Galaxy(stars=galaxy.stars, redshift=z, centre=galaxy.centre)
    n_star = galaxy.stars.nparticles
    free = memory.free_gb(MEMORY_LIMIT_GB)
    if free is not None:
        free -= reserved_gb
        if free <= 0:
            raise MemoryError(f"the data cubes need {reserved_gb:.1f} GB, more than the memory left")
    if CHUNK_STARS == "auto" and free is None:
        raise RuntimeError("can't tell how much memory is free here: set MEMORY_LIMIT_GB, or CHUNK_STARS to a number")
    if CHUNK_STARS == "auto" and n_star <= TRIAL_STARS:
        chunk_stars = n_star
    elif CHUNK_STARS == "auto":
        trial = stars_only.split(TRIAL_STARS)[0]
        n_trial = trial.stars.nparticles
        _, rise = memory.peak_rise_gb(run_chunk, trial, np.ones(n_trial))
        if rise is None:
            rise = memory.spectra_per_star_estimate(job.stellar_dust) * len(grid.lam) * 8 / 1e9 * n_trial
        gb_per_star = rise / n_trial
        if cube_labels:
            gb_per_star += len(grid.lam) * 8 / 1e9
        del trial
        chunk_stars = min(n_star, memory.stars_per_chunk(free, gb_per_star, MEMORY_SAFETY))
    else:
        chunk_stars = n_star if CHUNK_STARS is None else int(CHUNK_STARS)
    n_chunks = -(-n_star // chunk_stars)
    star_xy = galaxy.stars.centered_coordinates[:, :2].to("kpc").value
    in_preset = np.hypot(*star_xy.T) <= PRESET_APERTURE_ARCSEC * kpc_per_arcsec

    # ---- step 8: spectra, photometry and images, chunk by chunk ----
    preset_lnu = {p: 0.0 for p in STAR_LABELS}
    star_cubes = {p: np.zeros(cube_shape) for p in cube_labels}
    chunks = stars_only.split(chunk_stars)
    start, peaks = 0, []
    for k in range(n_chunks):
        chunk = chunks.pop(0)
        n_c = chunk.stars.nparticles
        weights = in_preset[start:start + n_c].astype(float) if APERTURE_SPECTRUM == "preset" else None
        before = memory.rss_gb()
        sums, rise = memory.peak_rise_gb(run_chunk, chunk, weights, star_cubes)
        peaks.append(None if rise is None else before + rise)
        if sums is not None:
            for p in STAR_LABELS:
                preset_lnu[p] = preset_lnu[p] + sums[p]
        if chunk is not stars_only:
            accumulate_pipeline_results_from_child(stars_only, chunk)
        start += n_c
        del chunk, sums
    if agn_on:
        bhs_only = Galaxy(black_holes=galaxy.black_holes, redshift=z, centre=galaxy.centre)
        with np.errstate(over="ignore"):
            bhs_only.get_spectra(job.agn_dust)
        bhs_only.get_observed_spectra(cosmo=cosmo, igm=IGM)
        bhs_only.get_photo_fnu(filters)
        galaxy.black_holes.get_particle_photo_fnu(filters)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"\s*Coordinates must be centered")
            bhs_only.get_images_flux(*BH_IMAGE_LABELS, **image_kw)
    owner = {label: galaxy if model[label].emitter == "galaxy" else
             galaxy.stars if model[label].emitter == "stellar" else galaxy.black_holes for label in SAVED}
    parts = {label: [label] for label in SAVED if model[label].emitter != "galaxy"}
    parts["total_dust_emission"] = [p for p in ("stellar_dust_emission", "AGN_dust_emission") if p in parts]
    parts["reprocessed_total"] = [p for p in ("reprocessed", "AGN_reprocessed") if p in parts]
    parts["total"] = [p for p in ("stellar_total", "AGN_total") if p in parts] + parts["total_dust_emission"]
    for label in SAVED:
        if model[label].emitter == "galaxy":
            galaxy.spectra[label] = sum((owner[p].spectra[p] for p in parts[label][1:]), owner[parts[label][0]].spectra[parts[label][0]])
            galaxy.spectra[label].get_fnu(cosmo=cosmo, z=z, igm=IGM)
            galaxy.photo_fnu[label] = galaxy.spectra[label].get_photo_fnu(filters)
    inst = job.instrument.label
    raw = sum((owner[p].images_fnu[inst][p] for p in parts["total"][1:]), owner["stellar_total"].images_fnu[inst]["stellar_total"])
    psf = sum((owner[p].images_psf_fnu[inst][p] for p in parts["total"][1:]), owner["stellar_total"].images_psf_fnu[inst]["stellar_total"])
    np.random.seed([SEED, snap_number, sidx])
    noisy = ImageCollection(resolution=psf.resolution, fov=psf.fov,
                            imgs={fc: job.instrument.apply_noise(psf[fc], fc, aperture_radius=job.instrument.depth_app_radius)
                                  for fc in FILTER_CODES})
    IMAGE_LABELS = ["total", "stellar_total"] + (["AGN_total"] if agn_on else [])
    alone = {label: (owner[label].images_fnu[inst][label], owner[label].images_psf_fnu[inst][label])
             for label in IMAGE_LABELS[1:]}

    # ---- step 9: save the spectrum and photometry ----
    settings = {
        "simulation": SIM, "snapshot": snap_number, "redshift": snap.redshift, "observed_z": z, "particles": PARTICLES,
        "centre": gal.centre_method, "star_mass": STAR_MASS, "dust_model": DUST_MODEL, "sph_kernel": SPH_KERNEL,
        **({"kappa_code": kappa_code} if DUST_MODEL == "code_units" else {"kappa_dust": KAPPA_DUST, "dtm_fit": "Vijayan+19 Eq. 15"}),
        "kappa_pc2_per_Msun": kappa, "kappa_BC": KAPPA_BC, "Z_BC": Z_BC,
        "young_age_Myr": YOUNG_AGE.to("Myr").value, "dust_slope_BC": DUST_SLOPE_BC, "dust_slope_ISM": DUST_SLOPE_ISM,
        "dust_temperature_K": DUST_TEMPERATURE.to("K").value, "dust_beta": DUST_BETA,
        "grid": GRID_NAME,
        **({"agn_grid": AGN_GRID_NAME, "agn_grid_point": {k: float(np.asarray(getattr(v, "value", v)))
                                                         for k, v in AGN_GRID_POINT.items()},
            "agn_u_from": AGN_U_FROM, "efficiency": EFFICIENCY, "disc_temperature": str(DISC_TEMPERATURE), "disc_radius": DISC_RADIUS, "disc_radius_unit": DISC_RADIUS_UNIT,
            "r_isco_rg": R_ISCO, "covering_fraction": COVERING_FRACTION, "covering_mode": COVERING_MODE,
            "covering_seed": COVERING_SEED, "agn_dust_slope": AGN_DUST_SLOPE,
            "agn_dust_window_A": None if AGN_DUST_WINDOW is None else [w.to("angstrom").value for w in AGN_DUST_WINDOW]}
           if INCLUDE_AGN else {}),
        "igm": IGM.__name__, "cosmology": {"H0": cosmo.H0.value, "Om0": cosmo.Om0},
    }
    galaxy_meta = {
        "sidx": sidx, "M_star_subfind": g["m_star_subfind"], "dust_to_metal": g["dust_to_metal"],
        "centre_ckpch": gal.centre_ckpch, "crop": gal.crop, "agn": agn_on,
        **({"bh_ids": [int(i) for i in g["ids"]], "los_covered": g["los_covered"].tolist()} if agn_on else {}),
    }
    meta = {**settings, **galaxy_meta}
    stem = f"{SIM}_{snap_number:03d}_sidx{sidx}"
    save_spectrum(out_dir / f"{stem}_spectrum.npz", {c: owner[c].spectra[c] for c in SAVED},
                  {c: owner[c].photo_fnu[c] for c in SAVED}, meta)

    # ---- step 10: segmentation ----
    detect = noisy[DETECT_FILTER].arr
    sigma_detect = (job.depths[DETECT_FILTER] / 5).to("nJy").value
    segm = analysis.segment(detect, sigma_detect, nsigma=DETECT_NSIGMA, npixels=DETECT_NPIXELS,
                            connectivity=DETECT_CONNECTIVITY, nlevels=DEBLEND_NLEVELS, contrast=DEBLEND_CONTRAST,
                            smooth_kernel=job.psfs[DETECT_FILTER] if DETECT_SMOOTH == "psf" else None)
    segmentation_ok = segm is not None
    if segmentation_ok:
        label, centroid, offset_px = analysis.pick_source(detect, segm)
        source = segm.data == label
    else:
        label, centroid, offset_px = 0, analysis.image_centre(detect.shape), 0.0
        source = np.ones(detect.shape, dtype=bool)
    pixel_kpc = PIXEL * kpc_per_arcsec
    n_out = round(FOV / PIXEL)
    target = centroid if IMAGE_CENTRE == "light" else analysis.image_centre(detect.shape)
    window = analysis.centred_window(n_out, detect.shape[0], target)

    # ---- step 11: morphology ----
    morph, morphology_ok = None, False
    if RUN_STATMORPH and segmentation_ok:
        import statmorph
        from astropy.utils.exceptions import AstropyDeprecationWarning
        others = segm.data.astype(bool) & ~source
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyDeprecationWarning)
            morph = statmorph.SourceMorphology(detect, segm.data, label, mask=others,
                                               weightmap=np.full(detect.shape, sigma_detect),
                                               psf=job.psfs[DETECT_FILTER])
        morphology_ok = morph.flag <= 1
    morphology = {f"morph_{k}": float(getattr(morph, k)) if morph is not None else np.nan
                  for k in ("concentration", "asymmetry", "smoothness", "gini", "m20", "sersic_n")}
    morphology |= {f"morph_{k}_arcsec": float(getattr(morph, k)) * PIXEL if morph is not None else np.nan
                   for k in ("r20", "r50", "r80", "sersic_rhalf")}
    morphology |= {f"morph_{k}": int(getattr(morph, k)) if morph is not None else -1 for k in ("flag", "flag_sersic")}

    # ---- step 12: measurements in the source ----
    grid_kw = dict(resolution=pixel_kpc * kpc, fov=IMAGE_FOV * kpc_per_arcsec * kpc)
    smooth_kw = dict(img_type="smoothed", kernel=kernel.get_kernel(), kernel_threshold=1)
    maps = {
        "stellar_mass": galaxy.get_map_stellar_mass(**grid_kw, **smooth_kw).arr,
        "gas_mass": galaxy.get_map_gas_mass(**grid_kw, **smooth_kw).arr,
        "sfr_young": galaxy.get_map_sfr(**grid_kw, **smooth_kw, age_bin=SFR_AGE).arr,
    }
    sfr_map = Image(**grid_kw)
    sfr_map.generate_img_smoothed(signal=galaxy.gas.star_formation_rates.to("Msun/yr").value,
                                  coordinates=galaxy.gas.centered_coordinates,
                                  smoothing_lengths=galaxy.gas.smoothing_lengths,
                                  kernel=kernel.get_kernel(), kernel_threshold=1)
    maps["sfr_gas"] = sfr_map.arr
    r_half_px = analysis.half_mass_radius(maps["stellar_mass"], source, centroid)
    if APERTURE == "segment":
        aperture = source
    else:
        aperture = source & (analysis.pixel_distances(source.shape, centroid) <= APERTURE_R_HALF * r_half_px)
    star_in = analysis.in_mask(galaxy.stars.centered_coordinates[:, :2].to("kpc").value, aperture, pixel_kpc)
    bh = galaxy.black_holes
    bh_xy = bh.centered_coordinates[:, :2].to("kpc").value if bh is not None else np.zeros((0, 2))
    bh_in = analysis.in_mask(bh_xy, aperture, pixel_kpc) if bh is not None else np.zeros(0, bool)
    props = {
        "r_half_arcsec": r_half_px * PIXEL,
        "r_half_pkpc": r_half_px * pixel_kpc,
        "aperture_pixels": int(aperture.sum()),
        **{f"{name}_aperture": float(m[aperture].sum()) for name, m in maps.items()},
        "M_star_particles": float(galaxy.stars.current_masses[star_in].sum().to("Msun").value),
        "n_stars": int(star_in.sum()),
        "bh_ids": [int(i) for i in bh.ids[bh_in]] if bh is not None else [],
        "M_bh": float(bh.masses[bh_in].sum().to("Msun").value) if bh is not None else 0.0,
        "Mdot_bh": float(bh.accretion_rates[bh_in].sum().to("Msun/yr").value) if bh is not None else 0.0,
    }
    if APERTURE_SPECTRUM == "cube":
        assert star_cubes[STAR_LABELS[0]].shape[:2] == aperture.shape
        part_lnu = {p: (star_cubes[p][aperture].sum(axis=0) * cube_units[p]).to("erg/s/Hz").value for p in STAR_LABELS}
        props["aperture_spectrum"] = "stars spread over their kernels (smoothed data cube)"
    elif APERTURE_SPECTRUM == "two_pass":
        part_lnu = {p: 0.0 for p in STAR_LABELS}
        chunks = stars_only.split(chunk_stars)
        start = 0
        while chunks:
            chunk = chunks.pop(0)
            n_c = chunk.stars.nparticles
            weights = star_in[start:start + n_c].astype(float)
            start += n_c
            if not weights.any():
                continue
            with np.errstate(over="ignore"):
                chunk.get_spectra(job.stellar_dust)
            for p in STAR_LABELS:
                part_lnu[p] = part_lnu[p] + weights @ chunk.stars.particle_spectra[p].lnu.to("erg/s/Hz").value
            del chunk
        props["aperture_spectrum"] = "stars whose positions are inside (two passes)"
    else:
        part_lnu = dict(preset_lnu)
        props["aperture_spectrum"] = f"preset circle, r = {PRESET_APERTURE_ARCSEC} arcsec about the catalogue position"
    del star_cubes
    for p in BH_LABELS:
        part_lnu[p] = bh_in.astype(float) @ galaxy.black_holes.particle_spectra[p].lnu.to("erg/s/Hz").value
    aperture_seds, aperture_phot = {}, {}
    for c in SAVED:
        lnu = sum(part_lnu[p] for p in parts[c])
        aperture_seds[c] = Sed(grid.lam, lnu=lnu * unyt_array(1.0, "erg/s/Hz"))
        aperture_seds[c].get_fnu(cosmo, z, igm=IGM)
        aperture_phot[c] = aperture_seds[c].get_photo_fnu(filters)
    stage_image = {"psf": psf, "noisy": noisy}
    image_phot = {kind: {fc: float(stage_image[kind][fc].arr[aperture].sum()) for fc in FILTER_CODES}
                  for kind in IMAGE_PHOTOMETRY}
    bright_bh = None
    if bh_in.any():
        k = np.argmax(np.where(bh_in, bh.accretion_rates.value, -np.inf))
        bright_bh = analysis.pixel_indices(bh_xy[k:k + 1], source.shape, pixel_kpc)[0]
    centres = analysis.compactness_centres(centroid, bright_bh)
    props["compactness"], best_centre, _ = analysis.compactness(psf[DETECT_FILTER].arr, source, centres, PIXEL,
                                                                COMPACTNESS_RADII, COMPACTNESS_METHOD)

    # ---- step 13: save the aperture spectrum and the images ----
    source_settings = {
        "detect_filter": DETECT_FILTER, "detect_nsigma": DETECT_NSIGMA, "detect_npixels": DETECT_NPIXELS,
        "detect_connectivity": DETECT_CONNECTIVITY, "deblend_nlevels": DEBLEND_NLEVELS,
        "deblend_contrast": DEBLEND_CONTRAST, "aperture_r_half": APERTURE_R_HALF, "sfr_young_age_Myr": SFR_AGE.to("Myr").value,
        "compactness_radii_arcsec": COMPACTNESS_RADII, "compactness_method": COMPACTNESS_METHOD,
        "detect_smooth": DETECT_SMOOTH, "image_photometry": list(IMAGE_PHOTOMETRY),
    }
    source_values = {
        "segmentation_ok": segmentation_ok, "source_label": label,
        "aperture": APERTURE if segmentation_ok else "whole_image",
        "centroid_offset_xy_arcsec": (centroid - analysis.image_centre(detect.shape)) * PIXEL,
        "offset_px": offset_px, "morphology_ok": morphology_ok, **morphology, **props,
        "image_aperture_fnu_nJy": {kind: {fc.split(".")[-1]: f for fc, f in phot.items()} for kind, phot in image_phot.items()},
    }
    source_meta = {**source_settings, **source_values}
    save_spectrum(out_dir / f"{stem}_aperture_spectrum.npz", aperture_seds, aperture_phot, {**meta, **source_meta})
    image_settings = {
        "pixel_arcsec": PIXEL, "fov_arcsec": FOV, "units": "nJy per pixel",
        "image_centre": IMAGE_CENTRE, "buffer_pix": BUFFER_PIX,
        "axes": "filter, x, y", "psf_source": PSF_SOURCE, "psf_product": PSF_PRODUCT, "noise_preset": NOISE_PRESET,
        "kernel_margin_pix": KERNEL_MARGIN_PIX,
        "sigma_pix_nJy": {fc.split(".")[-1]: (job.depths[fc] / 5).to("nJy").value for fc in FILTER_CODES},
    }
    image_values = {"pixel_pkpc": pixel_kpc, "window_start_ij_px": window, "centroid_ij_px": centroid - window,
                    "noise_seed": [SEED, snap_number, sidx]}
    images_meta = {**meta, **source_meta, **image_settings, **image_values}
    stage_images = {"total_raw": raw, "total_psf": psf, "total_noisy": noisy}
    for lab, prefix in (("stellar_total", "stellar"), ("AGN_total", "AGN")):
        if lab in alone:
            stage_images[f"{prefix}_raw"], stage_images[f"{prefix}_psf"] = alone[lab]
    stages = {name: analysis.cut_window(image_stack(images, FILTER_CODES), window, n_out)
              for name, images in stage_images.items()}
    segmap = analysis.cut_window(segm.data, window, n_out) if segmentation_ok else np.zeros((n_out, n_out), int)
    save_images(out_dir / f"{stem}_images.npz", FILTER_CODES, stages, segmap, images_meta)

    # ---- step 14 (optional): a quick look ----
    if QUICK_LOOK:
        quick_look(out_dir, stem, filters, z)

    # ---- step 15: LRD selection; then the catalogue row ----
    total = {fc.split(".")[-1]: float(galaxy.photo_fnu["total"][fc].to("nJy").value) for fc in FILTER_CODES}
    in_source = {fc.split(".")[-1]: float(aperture_phot["total"][fc].to("nJy").value) for fc in FILTER_CODES}
    row = {"sidx": sidx, "status": "ok", "z": z, "n_stars_crop": n_star, "n_gas_crop": len(gal.gas["masses"]),
           "n_bh_crop": g["n_bh"], "agn": agn_on, "agn_grid_flags": g["grid_flags"], "chunks": n_chunks,
           "M_star_subfind": g["m_star_subfind"], "dust_to_metal": g["dust_to_metal"],
           "segmentation_ok": segmentation_ok, "morphology_ok": morphology_ok,
           "offset_arcsec": offset_px * PIXEL,
           **{k: v for k, v in props.items() if k not in ("bh_ids", "aperture_spectrum")},
           **{k: v for k, v in morphology.items()},
           **{f"total_{f}_mag": float(ab_mag(v)) for f, v in total.items()},
           **{f"aperture_{f}_mag": float(ab_mag(v)) for f, v in in_source.items()}}
    if LRD_SELECTION is not None:
        fluxes = ({fc: aperture_phot["total"][fc].to("nJy").value for fc in FILTER_CODES} if LRD_PHOTOMETRY == "particles"
                  else image_phot[LRD_PHOTOMETRY])
        mags = {fc.split(".")[-1]: float(ab_mag(f)) for fc, f in fluxes.items()}
        select = {"ambra": analysis.lrd_selection_ambra, "kokorev24": analysis.lrd_selection_kokorev24}[LRD_SELECTION]
        colours = select(mags, props["compactness"])
        checks = analysis.lrd_source_quality(segmentation_ok, props["M_bh"], props["stellar_mass_aperture"], g["m_star_subfind"],
                                             offset_px * pixel_kpc * (1 + snap.redshift) * snap.h, max_offset_ckpch=LRD_MAX_OFFSET_CKPCH)
        row |= {"lrd_colours": bool(colours["lrd"]), "lrd_checks_ok": bool(checks["ok"]),
                "lrd": bool(colours["lrd"] and checks["ok"])}
    known = [p for p in peaks if p is not None]
    row |= {"seconds": round(time.time() - t_start, 1), "peak_memory_gb": round(max(known), 2) if known else np.nan,
            "files": stem}
    return row


def quick_look(out_dir, stem, filters, z):
    """The image beside the spectrum (step 14's last figure), as a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from smock import plotting
    from smock.io import load_images, load_spectrum
    images, image_meta = load_images(out_dir / f"{stem}_images.npz")
    spectrum, _, spectrum_meta = load_spectrum(out_dir / f"{stem}_spectrum.npz")
    plotting.plot_image_and_spectrum(images, image_meta, spectrum, spectrum_meta,
                                     rgb_kwargs=dict(label=f"z = {z:.1f}", display_arcsec=2.0, centre="centroid", scale_bar="kpc"),
                                     spectrum_kwargs=dict(filters=filters))
    plt.savefig(out_dir / f"{stem}_quick_look.png", dpi=80, bbox_inches="tight")
    plt.close("all")


# ======================================================================================
# The loop
# ======================================================================================
def galaxy_list(path, task, ntasks):
    """This task's share of the subhalo list: shuffled with a fixed seed, then dealt out."""
    path = Path(path)
    sidxs = np.load(path) if path.suffix == ".npy" else np.loadtxt(path, dtype=np.int64, ndmin=1)
    sidxs = np.asarray(sidxs, dtype=np.int64)
    if ntasks > 1:
        sidxs = np.random.default_rng(0).permutation(sidxs)
    return [int(i) for i in sidxs[task::ntasks]]


def files_exist(out_dir, snap_number, sidx):
    stem = f"{SIM}_{snap_number:03d}_sidx{sidx}"
    return all((out_dir / f"{stem}_{kind}.npz").exists() for kind in ("spectrum", "aperture_spectrum", "images"))


def append_row(path, row):
    """Add one galaxy's line to the task's catalogue (JSON lines: one dict per line, so rows of
    failed or skipped galaxies, with fewer entries, can sit beside full ones)."""
    with open(path, "a") as fh:
        fh.write(json.dumps(row, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--snap", type=int, required=True, help="snapshot number (PIG_<snap>_subfind in PIG_DIR)")
    ap.add_argument("--sidx-file", required=True, help="subhalo indices: a text file, one per line, or a .npy")
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--ntasks", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    ap.add_argument("--out-dir", default=None, help=f"default: {OUT_DIR}/<SIM>_<snap>")
    ap.add_argument("--pig-dir", default=None, help="snapshots folder (default PIG_DIR in the settings)")
    ap.add_argument("--chunk-stars", default=None, help='overrides CHUNK_STARS: "auto", a number, or "none" (all at once)')
    ap.add_argument("--redo", action="store_true", help="make galaxies again even if their files exist")
    args = ap.parse_args()
    global CHUNK_STARS
    if args.chunk_stars is not None:
        CHUNK_STARS = None if args.chunk_stars.lower() == "none" else (
            "auto" if args.chunk_stars == "auto" else int(args.chunk_stars))
    snap = open_snapshot(args.pig_dir or PIG_DIR, args.snap)
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR / f"{SIM}_{args.snap:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    mine = galaxy_list(args.sidx_file, args.task, args.ntasks)
    catalogue = out_dir / f"catalogue_task{args.task}.jsonl"
    t0 = time.time()
    print(f"task {args.task}/{args.ntasks}: {len(mine)} galaxies at z = {snap.redshift:.3f}, into {out_dir}", flush=True)
    job = Job()
    used, limit = memory.rss_gb(), memory.memory_limit_gb()
    print(f"set-up: {time.time() - t0:.0f} s; memory in use "
          f"{'unknown' if used is None else f'{used:.1f} GB'}, limit {'unknown' if limit is None else f'{limit:.0f} GB'}", flush=True)
    counts = dict(ok=0, done_before=0, skipped=0, failed=0)
    for i, sidx in enumerate(mine):
        if not args.redo and files_exist(out_dir, args.snap, sidx):
            counts["done_before"] += 1
            continue
        t = time.time()
        try:
            row = observe(job, snap, args.snap, sidx, out_dir)
            counts["ok"] += 1
            print(f"[{i + 1}/{len(mine)}] sidx {sidx}: ok, {row['n_stars_crop']} stars in {row['chunks']} chunk(s), "
                  f"{row['seconds']:.0f} s, peak {row['peak_memory_gb']} GB", flush=True)
        except Skip as e:
            counts["skipped"] += 1
            row = {"sidx": sidx, "status": f"skipped: {e}", "seconds": round(time.time() - t, 1)}
            print(f"[{i + 1}/{len(mine)}] sidx {sidx}: skipped ({e})", flush=True)
        except Exception as e:                       # one galaxy's failure doesn't stop the run
            counts["failed"] += 1
            row = {"sidx": sidx, "status": f"failed: {type(e).__name__}: {e}", "seconds": round(time.time() - t, 1)}
            print(f"[{i + 1}/{len(mine)}] sidx {sidx}: FAILED\n{traceback.format_exc()}", flush=True)
        append_row(catalogue, row)
    print(f"task {args.task} finished in {(time.time() - t0) / 60:.1f} min: " +
          ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in counts.items()), flush=True)


if __name__ == "__main__":
    main()
