"""Known-answer tests for the observer-style options: the mosaic-like PSF, compactness with the sky kept,
S/N in a circle, PSF-matched circle photometry, Kokorev et al. 2024's S/N cut, and the continuum slopes."""
import numpy as np
import pytest

from smock import config
from smock.analysis import (aperture_flux, aperture_snr, compactness, continuum_slope, lrd_selection_kokorev24,
                            matched_circle_fluxes, observer_mask, point_source_fraction, segment, spectral_shape,
                            v_shaped)
from smock.imaging import gaussian_psf, load_stpsf_psf, mosaic_psf, nircam_channel

HAVE_PSFS = config.PSF_DIR is not None and config.PSF_DIR.exists()


def c_of(psf, pixel=0.03):
    mid = np.array([(psf.shape[0] - 1) / 2.0] * 2)
    return aperture_flux(psf, mid, 0.2 / pixel) / aperture_flux(psf, mid, 0.1 / pixel)


def test_nircam_channel():
    assert nircam_channel("F090W") == "SW" and nircam_channel("F200W") == "SW"
    assert nircam_channel("F277W") == "LW" and nircam_channel("F444W") == "LW"


def test_mosaic_psf_without_detector_effects_is_det_samp():
    """With the squares and the Gaussian reduced to nothing, the result is the oversampled PSF summed over
    each output pixel, which is what STPSF's DET_SAMP is."""
    rng = np.random.default_rng(3)
    over = rng.random((55, 55))
    over /= over.sum()
    import smock.imaging as im
    saved = dict(im.NIRCAM_DETECTOR_PIXEL_ARCSEC), dict(im.NIRCAM_CHARGE_DIFFUSION_ARCSEC)
    try:
        im.NIRCAM_DETECTOR_PIXEL_ARCSEC["LW"], im.NIRCAM_CHARGE_DIFFUSION_ARCSEC["LW"] = 0.0, 0.0
        out = mosaic_psf(over, 0.006, 5, "F444W")
    finally:
        im.NIRCAM_DETECTOR_PIXEL_ARCSEC.update(saved[0]), im.NIRCAM_CHARGE_DIFFUSION_ARCSEC.update(saved[1])
    assert np.allclose(out, over.reshape(11, 5, 11, 5).sum(axis=(1, 3)), atol=1e-12)


def test_mosaic_psf_refuses_even_arrays():
    with pytest.raises(ValueError):
        mosaic_psf(np.ones((50, 50)), 0.006, 5, "F444W")


@pytest.mark.skipif(not HAVE_PSFS, reason="STPSF PSFs not downloaded")
def test_mosaic_psf_f444w_point_source():
    """Our F444W point source: 1.29 with the telescope's PSF, 1.48 mosaic-like (1.45 with a 0.75 drop),
    as checked against STPSF's own detector model (1.49) and the DJA mosaic PSFs (1.50 at 0.04")."""
    det, _ = load_stpsf_psf(config.PSF_DIR, "F444W")
    over, header = load_stpsf_psf(config.PSF_DIR, "F444W", "OVERSAMP")
    fine = header["SMOCKPIX"] / header["SMOCKOVR"]
    full = mosaic_psf(over, fine, header["SMOCKOVR"], "F444W")
    assert full.shape == det.shape and full.sum() == pytest.approx(1.0)
    assert c_of(det) == pytest.approx(1.286, abs=0.003)
    assert c_of(full) == pytest.approx(1.478, abs=0.005)
    assert c_of(mosaic_psf(over, fine, header["SMOCKOVR"], "F444W", pixfrac=0.75)) == pytest.approx(1.450, abs=0.005)


def test_observer_mask_keeps_sky_and_drops_neighbours():
    seg = np.array([[0, 1, 2], [0, 1, 0]])
    assert observer_mask(seg, 1).tolist() == [[True, True, False], [True, True, True]]


def test_sky_kept_compactness_of_a_faint_point_source():
    """A faint point source's segment is smaller than the 0.2" aperture, so zeroing the pixels outside it
    lowers C; with the sky kept (noiseless image) C is the PSF's own."""
    psf = gaussian_psf(0.145, 0.03)
    n = 101
    img = np.zeros((n, n))
    img[50, 50] = 1.0
    from scipy.signal import fftconvolve
    clean = fftconvolve(img, psf, mode="same")
    sigma = clean.max() / 8.0                     # faint: a small segment
    rng = np.random.default_rng(0)
    segm = segment(clean + rng.normal(0, sigma, clean.shape), sigma)
    label = int(segm.data[50, 50])
    seg_c, _, _ = compactness(clean, segm.data == label, [np.array([50.0, 50.0])], 0.03)
    sky_c, _, _ = compactness(clean, observer_mask(segm.data, label), [np.array([50.0, 50.0])], 0.03)
    assert sky_c == pytest.approx(c_of(fftconvolve(np.pad(np.ones((1, 1)), 50), psf, mode="same")), rel=1e-6)
    assert seg_c < sky_c - 0.05


def test_aperture_snr_white_noise():
    img = np.zeros((41, 41))
    img[20, 20] = 100.0
    assert aperture_snr(img, (20.0, 20.0), 5.0, 2.0) == pytest.approx(100.0 / (2.0 * np.sqrt(np.pi * 25.0)))


def test_matched_circle_point_source_total():
    """A point source in every band: the circle holds the PSF's share of it, so dividing by
    point_source_fraction gives the total, whatever the band."""
    psf = gaussian_psf(0.145, 0.03, size=61)
    raw = {}
    for band, flux in (("F150W", 3.0), ("F444W", 7.0)):
        r = np.zeros((81, 81))
        r[40, 40] = flux
        raw[band] = r
    f = matched_circle_fluxes(raw, psf, np.array([40.0, 40.0]), 6.0)
    frac = point_source_fraction(psf, 6.0)
    assert 0.5 < frac < 1.0
    assert f["F150W"] / frac == pytest.approx(3.0, rel=1e-6)
    assert f["F444W"] / frac == pytest.approx(7.0, rel=1e-6)


def test_kokorev24_snr_cut():
    m = {"F115W": 27.6, "F150W": 27.5, "F200W": 27.5, "F277W": 26.7, "F356W": 26.0, "F444W": 26.0}
    assert lrd_selection_kokorev24(m, 1.5)["lrd"]                     # no S/N given: not applied
    assert lrd_selection_kokorev24(m, 1.5, snr=20.0)["lrd"]
    s = lrd_selection_kokorev24(m, 1.5, snr=10.0)
    assert not s["detected"] and not s["lrd"]


def broken_power_law(lam, beta_uv, beta_opt, split=3645.0):
    flam = np.where(lam < split, (lam / split) ** beta_uv, (lam / split) ** beta_opt)
    return flam * lam**2 / 2.99792458e18          # back to L_nu


def test_continuum_slopes_of_a_broken_power_law_with_lines():
    lam = np.geomspace(1000, 10000, 3000)
    lnu = broken_power_law(lam, -1.5, 0.8)
    with_lines = lnu.copy()
    for l0 in (4861, 5007, 6563, 2100):            # 2100 A is not in the mask list: the clipping removes it
        with_lines[np.abs(lam - l0) < 3] *= 30
    s = spectral_shape(lam, with_lines)
    assert s["beta_uv"] == pytest.approx(-1.5, abs=0.01)
    assert s["beta_opt"] == pytest.approx(0.8, abs=0.01)
    assert bool(v_shaped(s["beta_uv"], s["beta_opt"]))
    assert continuum_slope(lam, lnu, 3000, 7000) > -1.5            # a window across the split mixes both


def test_v_shaped_thresholds():
    assert v_shaped(-1.0, 0.5) and not v_shaped(-0.1, 0.5) and not v_shaped(-1.0, -0.1)
    assert not v_shaped(-0.3, 0.1)                                  # difference 0.4 < 0.5
    assert v_shaped(-0.5, 0.1, max_beta_uv=-0.37, min_difference=0.0)    # Kocevski et al. 2024's version
