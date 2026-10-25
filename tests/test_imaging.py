"""Known-answer tests for smock.imaging: AB fluxes, the Gaussian PSF, encircled energy and
the depth -> per-pixel noise conversion."""
import numpy as np
import pytest

from smock import config
from smock.imaging import (NIRCAM_WIDE, SURVEY_DEPTHS_5SIGMA, ab_mag_to_njy, encircled_energy, gaussian_psf,
                           load_stpsf_psf, scale_depth, sigma_pix_from_depth)


def test_ab_mag_to_njy():
    assert ab_mag_to_njy(0.0) == pytest.approx(3631e9)
    assert ab_mag_to_njy(23.9) == pytest.approx(1000.0, rel=1e-4)     # 23.9 AB = 1 uJy
    assert ab_mag_to_njy(31.4) == pytest.approx(1.0, rel=1e-4)        # 31.4 AB = 1 nJy


def test_gaussian_psf_width():
    psf = gaussian_psf(fwhm_arcsec=0.15, pixel_arcsec=0.03)            # sigma = 2.123 px
    assert psf.sum() == pytest.approx(1.0)
    n = psf.shape[0]
    assert n % 2 == 1 and np.unravel_index(psf.argmax(), psf.shape) == (n // 2, n // 2)
    x = np.arange(n) - n // 2
    var_x = (psf.sum(axis=0) * x**2).sum()
    assert np.sqrt(var_x) == pytest.approx(0.15 / 0.03 / 2.3548, rel=1e-3)


def test_encircled_energy_single_pixel():
    # all the flux in the central pixel, spread over its area
    psf = np.zeros((5, 5))
    psf[2, 2] = 1.0
    assert encircled_energy(psf, 1.0, 0.5, subsample=200) == pytest.approx(np.pi / 4, rel=1e-3)
    assert encircled_energy(psf, 1.0, 0.75) == pytest.approx(1.0)       # beyond the pixel corners


def test_encircled_energy_of_gaussian():
    # A pixelated Gaussian is the Gaussian blurred by one pixel: variance sigma^2 + 1/12 px^2,
    # then EE(r) = 1 - exp(-r^2 / 2 sigma_eff^2).
    psf = gaussian_psf(fwhm_arcsec=0.15, pixel_arcsec=0.03, size=41)
    sigma_eff = np.sqrt((0.15 / 2.3548) ** 2 + 0.03**2 / 12)
    for r in (0.05, 0.1, 0.2):
        assert encircled_energy(psf, 0.03, r, subsample=20) == pytest.approx(
            1 - np.exp(-r**2 / (2 * sigma_eff**2)), abs=3e-3)
    assert encircled_energy(psf, 0.03, 10.0) == pytest.approx(1.0)


def test_sigma_pix_from_depth():
    # 10 nJy at 5 sigma, all flux in the aperture, r = 0.1" on 0.03" pixels: N = 34.907
    assert sigma_pix_from_depth(10.0, 5, 0.1, 0.03, ee=1.0) == pytest.approx(2.0 / np.sqrt(np.pi * (0.1 / 0.03) ** 2))
    # the aperture loses (1 - ee) of a point source, so the same depth means less noise
    assert sigma_pix_from_depth(10.0, 5, 0.1, 0.03, ee=0.6) == pytest.approx(
        0.6 * sigma_pix_from_depth(10.0, 5, 0.1, 0.03, ee=1.0))


def test_scale_depth():
    # 4x the exposure: 2x the S/N, 0.753 mag deeper
    assert scale_depth(28.0, 1000.0, 4000.0) == pytest.approx(28.0 + 2.5 * np.log10(2.0))


def test_survey_presets():
    jades, ceers = SURVEY_DEPTHS_5SIGMA["JADES-Medium"], SURVEY_DEPTHS_5SIGMA["CEERS"]
    assert jades["F200W"] == pytest.approx(29.02 + 0.7526, abs=1e-4)   # 10 sigma -> 5 sigma
    # JADES Medium F200W (7.6 ks) scaled to CEERS's 2835 s matches CEERS ...
    assert scale_depth(jades["F200W"], 7600, 2835) == pytest.approx(ceers["F200W"], abs=0.1)
    # ... which is how CEERS's missing F090W is filled: JADES F090W (9.2 ks) at 2835 s
    assert ceers["F090W"] == pytest.approx(28.85 + 0.7526 - 1.25 * np.log10(9200 / 2835), abs=1e-4)
    assert set(ceers) == set(jades)



HAVE_PSFS = config.PSF_DIR is not None and config.PSF_DIR.exists()


@pytest.mark.skipif(not HAVE_PSFS, reason="STPSF PSFs not downloaded")
@pytest.mark.parametrize("filt", NIRCAM_WIDE)
def test_stpsf_psfs(filt):
    det, header = load_stpsf_psf(config.PSF_DIR, filt)
    over, _ = load_stpsf_psf(config.PSF_DIR, filt, "OVERSAMP")
    for psf in (det, over):
        n = psf.shape[0]
        assert psf.sum() == pytest.approx(1.0, rel=1e-5)
        assert np.unravel_index(psf.argmax(), psf.shape) == (n // 2, n // 2)
    assert over.shape[0] == 5 * det.shape[0]
    # the header's EE (from the 0.006" grid) is what ours gives on that grid; on the 0.03"
    # array r = 0.1" is only 3.3 pixels, so ours is up to ~0.02 low there: use the header's
    assert encircled_energy(over, 0.006, 0.1) == pytest.approx(header["EE_R0P1"], abs=0.003)
    assert encircled_energy(det, 0.03, 0.1) == pytest.approx(header["EE_R0P1"], abs=0.025)
    assert header["FILTER"] == filt and header["SMOCKPIX"] == 0.03


def _bin5(a):
    m = a.shape[0] // 5
    return a[:m * 5, :m * 5].reshape(m, 5, m, 5).sum(axis=(1, 3))


@pytest.mark.skipif(not HAVE_PSFS, reason="STPSF PSFs not downloaded")
@pytest.mark.parametrize("filt", ["F090W", "F444W"])
def test_det_samp_matches_fine_grid_convolution(filt):
    """On a 0.03" image (light summed over pixels, as synthesizer draws it), convolving
    with DET_SAMP matches the exact answer (draw on the 0.006" grid, convolve with OVERSAMP,
    sum into 0.03" pixels): exactly for a point at a pixel centre, within 2% for a compact clump."""
    from scipy.signal import fftconvolve
    det, _ = load_stpsf_psf(config.PSF_DIR, filt)
    over, _ = load_stpsf_psf(config.PSF_DIR, filt, "OVERSAMP")
    m = 61 * 5
    yy, xx = (np.indices((m, m)) - (m - 1) / 2) * 0.006
    point = np.zeros((m, m))
    point[m // 2, m // 2] = 1.0
    clump = np.exp(-(xx**2 + yy**2) / (2 * (0.06 / 2.3548) ** 2))
    for src, tol in ((point, 1e-6), (clump / clump.sum(), 0.02)):
        exact = _bin5(fftconvolve(src, over, mode="same"))
        ours = fftconvolve(_bin5(src), det, mode="same")
        assert np.abs(ours - exact).max() <= tol * exact.max()
