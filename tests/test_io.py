"""Known-answer tests for smock.io: AB magnitudes, the observed flux, rest-frame nu L_nu,
band photometry and the .npz round trip."""
import numpy as np
import pytest
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from unyt import Hz, angstrom, erg, s

from synthesizer.emissions import Sed
from synthesizer.instruments import Filter, FilterCollection

from smock.io import (ab_mag, image_stack, load_images, load_spectrum, photometry_table, save_images,
                      save_spectrum, spectrum_table)

Z = 6.0
COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)
LNU = 1e28                                    # erg/s/Hz, flat in L_nu


@pytest.fixture(scope="module")
def seds():
    lam = np.logspace(2, 6, 2000) * angstrom
    out = {}
    for label, scale in (("total", 1.0), ("reprocessed", 2.0)):
        sed = Sed(lam, lnu=np.full(lam.size, scale * LNU) * erg / s / Hz)
        sed.get_fnu(COSMO, Z)                 # no IGM: a flat spectrum stays flat
        out[label] = sed
    return out


@pytest.fixture(scope="module")
def photometry(seds):
    # top-hat filters (no SVO needed), on the observed wavelength grid
    obs_lam = seds["total"].obslam
    filters = FilterCollection(filters=[
        Filter("generic/filter.K", lam_min=20000 * angstrom, lam_max=25000 * angstrom, new_lam=obs_lam),
        Filter("generic/filter.L", lam_min=30000 * angstrom, lam_max=40000 * angstrom, new_lam=obs_lam)])
    for sed in seds.values():
        sed.get_photo_fnu(filters)
    return {label: sed.photo_fnu for label, sed in seds.items()}


def test_ab_mag():
    assert ab_mag(3631e9) == pytest.approx(0.0)                 # 3631 Jy = 0 AB
    assert ab_mag(1.0) == pytest.approx(31.4, abs=1e-4)         # 1 nJy = 31.4 AB
    assert np.isnan(ab_mag(0.0)) and np.isnan(ab_mag(-1.0))


def test_observed_flux_and_rest_nuLnu(seds):
    spec = spectrum_table(seds)
    # f_nu = (1 + z) L_nu / (4 pi d_L^2): luminosity -> flux, independently with astropy
    d_l = COSMO.luminosity_distance(Z).to(u.cm).value
    fnu_njy = (1 + Z) * LNU / (4 * np.pi * d_l**2) / 1e-32
    assert spec["total_fnu_obs_nJy"] == pytest.approx(fnu_njy, rel=1e-6)
    assert spec["reprocessed_fnu_obs_nJy"] == pytest.approx(2 * fnu_njy, rel=1e-6)
    # and back again: the round trip recovers L_nu
    l_back = spec["total_fnu_obs_nJy"] * 1e-32 * 4 * np.pi * d_l**2 / (1 + Z)
    assert l_back == pytest.approx(spec["total_Lnu_rest_erg_s_Hz"], rel=1e-6)
    # nu L_nu uses the rest-frame nu, and the frames differ by 1 + z
    c_aa = 2.99792458e18
    assert spec["total_nuLnu_rest_erg_s"] == pytest.approx(c_aa / spec["lam_rest_AA"] * LNU, rel=1e-9)
    assert spec["lam_obs_AA"] == pytest.approx((1 + Z) * spec["lam_rest_AA"])
    assert spec["nu_obs_Hz"] == pytest.approx(spec["nu_rest_Hz"] / (1 + Z))


def test_band_photometry_of_flat_spectrum(seds, photometry):
    # a flat f_nu spectrum has that f_nu in any band
    phot = photometry_table(photometry)
    fnu = spectrum_table(seds)["total_fnu_obs_nJy"][0]
    assert list(phot["filter"]) == ["generic/filter.K", "generic/filter.L"]
    assert phot["total_fnu_nJy"] == pytest.approx([fnu, fnu], rel=1e-6)
    assert phot["total_mag_AB"] == pytest.approx(ab_mag([fnu, fnu]))
    assert 20000 < phot["pivot_obs_AA"][0] < 25000


def test_round_trip(tmp_path, seds, photometry):
    meta = {"sidx": np.int64(303604), "redshift": Z, "crop": {"size": 1.5, "unit": "arcsec"}}
    path = save_spectrum(tmp_path / "gal.npz", seds, photometry, meta)
    spec, phot, meta_back = load_spectrum(path)                 # allow_pickle=False inside
    np.testing.assert_array_equal(spec, spectrum_table(seds))
    np.testing.assert_array_equal(phot, photometry_table(photometry))
    assert meta_back["sidx"] == 303604 and meta_back["crop"]["unit"] == "arcsec"
    assert {"created", "synthesizer_version", "smock_commit"} <= set(meta_back)


def test_mismatched_grids_are_refused(seds):
    other = Sed(np.logspace(2, 6, 100) * angstrom, lnu=np.full(100, LNU) * erg / s / Hz)
    other.get_fnu(COSMO, Z)
    with pytest.raises(ValueError):
        spectrum_table({"total": seds["total"], "other": other})


def test_images_round_trip(tmp_path):
    # image_stack converts each image to nJy per pixel; the file keeps axis order (filter, x, y)
    from synthesizer.imaging import Image
    from unyt import arcsec, unyt_array

    rng = np.random.default_rng(0)
    codes = ["JWST/NIRCam.F200W", "JWST/NIRCam.F444W"]
    arrs = {fc: rng.normal(size=(10, 10)) for fc in codes}
    images = {fc: Image(resolution=0.03 * arcsec, fov=0.3 * arcsec, img=unyt_array(a, "uJy"))
              for fc, a in arrs.items()}
    stack = image_stack(images, codes)
    assert stack.shape == (2, 10, 10) and np.allclose(stack[1], 1000 * arrs[codes[1]])

    segmap = np.zeros((10, 10), int)
    segmap[4:6, 4:6] = 1
    path = save_images(tmp_path / "x_images.npz", codes, {"stellar_psf": stack}, segmap, {"pixel_arcsec": 0.03})
    back, meta = load_images(path)
    assert sorted(back) == ["filters", "segmap", "stellar_psf"]
    assert back["filters"].tolist() == codes and back["segmap"].dtype == np.int32
    assert np.array_equal(back["stellar_psf"], stack) and np.array_equal(back["segmap"], segmap)
    assert meta["pixel_arcsec"] == 0.03 and "synthesizer_version" in meta
    with pytest.raises(ValueError):
        save_images(tmp_path / "bad.npz", codes, {"stellar_psf": stack[:1]}, None, {})
