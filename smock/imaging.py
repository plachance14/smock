"""Small helpers for the imaging step: a Gaussian PSF fallback, encircled energy, and
survey depths -> per-pixel noise.

The synthesizer instrument itself (filters, PSFs, noise) is built in the notebook; these
only prepare the numbers it takes. Fluxes are in nJy and angles in arcsec throughout.

Why the noise conversion is ours: surveys quote point-source depths
in small apertures, corrected to total flux. synthesizer 1.3.1's aperture-depth branch
divides the aperture noise by N_pix instead of sqrt(N_pix), so we convert the depth to a
per-pixel sigma here and give synthesizer that as a per-pixel ("point-source") depth.
"""
from pathlib import Path

import numpy as np

AB_ZERO_POINT_NJY = 3631e9          # 0 AB mag = 3631 Jy
FIVE_SIGMA_FROM_TEN = 2.5 * np.log10(2.0)   # mag to add to a 10-sigma depth for 5 sigma

NIRCAM_WIDE = ("F090W", "F115W", "F150W", "F200W", "F277W", "F356W", "F444W")

# PSF FWHM [arcsec] for the Gaussian fallback: JWST's documented NIRCam values, the same
# ones the old pipeline used (native-pixel FWHMs x 0.031" or 0.063").
NIRCAM_FWHM_ARCSEC = {"F090W": 0.033, "F115W": 0.040, "F150W": 0.050, "F200W": 0.066,
                      "F277W": 0.092, "F356W": 0.116, "F444W": 0.145}

def scale_depth(mag, t_from, t_to):
    """Depth [AB mag] for exposure time t_to, from one at t_from (background-limited: S/N ~ sqrt(t))."""
    return mag + 1.25 * np.log10(t_to / t_from)


# JADES Medium tier (7-12 ks per filter): Eisenstein et al. 2023 (arXiv:2306.02465),
# Table 4, quoted as 10-sigma in d = 0.2" apertures; F090W's exposure time [s] from the same table.
_JADES_MEDIUM_10SIGMA = {"F090W": 28.85, "F115W": 28.98, "F150W": 28.93, "F200W": 29.02,
                         "F277W": 29.32, "F356W": 29.32, "F444W": 29.03}
_JADES_MEDIUM_F090W_S = 9200.0
CEERS_EXPOSURE_S = 2835.0       # per filter and pointing (F115W twice that)

# 5-sigma point-source depths [AB mag]. How each was measured is in SURVEY_DEPTH_KINDS: the
# JWST ones in r = 0.1" apertures, corrected to total flux; the DECaLS one from a PSF fit.
SURVEY_DEPTHS_5SIGMA = {
    # CEERS final survey: Finkelstein et al. 2025 (arXiv:2501.04085), Table 5, measured in
    # d = 0.2" apertures. CEERS has no F090W, so it is derived: the JADES Medium F090W depth
    # scaled to CEERS's exposure time. The same scaling turns JADES
    # Medium F200W into 29.24, against CEERS's measured 29.3.
    "CEERS": {"F090W": float(scale_depth(_JADES_MEDIUM_10SIGMA["F090W"] + FIVE_SIGMA_FROM_TEN,
                                         _JADES_MEDIUM_F090W_S, CEERS_EXPOSURE_S)),
              "F115W": 29.2, "F150W": 29.1, "F200W": 29.3, "F277W": 29.5, "F356W": 29.4,
              "F444W": 29.0},
    "JADES-Medium": {f: m10 + FIVE_SIGMA_FROM_TEN for f, m10 in _JADES_MEDIUM_10SIGMA.items()},
    # DECaLS (DESI Legacy Imaging Surveys, DECam): the survey's design depths for two
    # exposures in 1.5" seeing (legacysurvey.org/dr10/description). Real depths vary across
    # the sky: the Legacy Surveys' psfdepth_<band> values give them (psfdepth_to_njy).
    "DECaLS-like": {"g": 24.7, "r": 23.9, "z": 23.0},
}
# "aperture": a point source in an r = SURVEY_APERTURE_RADIUS_ARCSEC aperture, corrected to
# total flux (sigma_pix_from_depth); "psf_weighted": a PSF fit, as the Legacy Surveys'
# psfdepth (sigma_pix_from_psf_depth)
SURVEY_DEPTH_KINDS = {"CEERS": "aperture", "JADES-Medium": "aperture", "DECaLS-like": "psf_weighted"}

DECAM_PIXEL_ARCSEC = 0.262          # Legacy Surveys coadds
DECALS_DESIGN_SEEING_ARCSEC = 1.5   # the seeing the DECaLS design depths assume (FWHM)
NANOMAGGY_NJY = 3631.0              # 1 nanomaggy = AB 22.5 = 3.631 microJy
# Moffat slope for ground-based seeing. Pure atmospheric turbulence gives about 4.765
# (Trujillo et al. 2001); real seeing PSFs usually have broader wings (lower beta).
MOFFAT_BETA = 3.5
SURVEY_APERTURE_RADIUS_ARCSEC = 0.1


def ab_mag_to_njy(mag):
    """AB magnitude -> flux density [nJy]."""
    return AB_ZERO_POINT_NJY * 10.0 ** (-0.4 * np.asarray(mag, dtype=np.float64))


def gaussian_psf(fwhm_arcsec, pixel_arcsec, size=None):
    """Circular Gaussian PSF sampled at pixel centres, normalised to sum 1.

    size: odd number of pixels per side (default: covers +-4 FWHM). The peak is on the
    central pixel.
    """
    sigma_pix = fwhm_arcsec / pixel_arcsec / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    if size is None:
        size = 2 * int(np.ceil(4.0 * fwhm_arcsec / pixel_arcsec)) + 1
    y, x = np.mgrid[:size, :size] - (size - 1) / 2.0
    psf = np.exp(-(x**2 + y**2) / (2.0 * sigma_pix**2))
    return psf / psf.sum()


def moffat_psf(fwhm_arcsec, pixel_arcsec, beta=MOFFAT_BETA, size=None):
    """Circular Moffat PSF sampled at pixel centres, normalised to sum 1: a model of
    ground-based seeing, whose wings are broader than a Gaussian's.

    I(r) is proportional to (1 + (r / alpha)^2)^-beta, with alpha = FWHM / (2 sqrt(2^(1/beta) - 1)).
    size: odd number of pixels per side (default: covers +-6 FWHM, which holds all but
    ~0.02% of the light for beta = 3.5). The peak is on the central pixel.
    """
    alpha_pix = fwhm_arcsec / pixel_arcsec / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    if size is None:
        size = 2 * int(np.ceil(6.0 * fwhm_arcsec / pixel_arcsec)) + 1
    y, x = np.mgrid[:size, :size] - (size - 1) / 2.0
    psf = (1.0 + (x**2 + y**2) / alpha_pix**2) ** -beta
    return psf / psf.sum()


def encircled_energy(psf, pixel_arcsec, radius_arcsec, subsample=10):
    """Fraction of a PSF's flux inside a circle of radius_arcsec about the array centre.

    Each pixel is split into subsample^2 sub-pixels, so partly covered pixels count by
    the fraction of their area inside the circle.
    """
    n = psf.shape[0]
    sub = (np.arange(n * subsample) + 0.5) / subsample - n / 2.0     # sub-pixel centres [px]
    yy, xx = np.meshgrid(sub, sub, indexing="ij")
    inside = (xx**2 + yy**2) < (radius_arcsec / pixel_arcsec) ** 2
    covered = inside.reshape(n, subsample, n, subsample).mean(axis=(1, 3))
    return float((psf * covered).sum() / psf.sum())


def sigma_pix_from_depth(depth_njy, nsigma, aperture_radius_arcsec, pixel_arcsec, ee):
    """Per-pixel white-noise sigma [nJy] that reproduces a survey point-source depth.

    A point source at the depth (total flux depth_njy) puts ee * depth_njy in the
    aperture, which is nsigma times the aperture noise. With independent pixels the
    aperture noise is sigma_pix * sqrt(N_pix), with N_pix = pi r^2 / pixel^2.

    Convention: an aperture with exact partial-pixel weights w has noise
    sigma_pix * sqrt(sum w^2), about 5% below sqrt(N_pix) at r = 3.3 px. The surveys
    don't say which they used, so depths are uncertain at the ~0.05 mag level (fine for
    now; worth revisiting if the noise level becomes critical).
    """
    n_pix = np.pi * (aperture_radius_arcsec / pixel_arcsec) ** 2
    return ee * depth_njy / nsigma / np.sqrt(n_pix)


def sigma_pix_from_psf_depth(depth_njy, nsigma, psf):
    """Per-pixel white-noise sigma [nJy] that reproduces a PSF-weighted point-source depth.

    A PSF fit to a point source on white noise measures its flux with
    sigma_flux = sigma_pix / sqrt(sum P^2), where P is the PSF on the image's pixels,
    normalised to sum 1. A depth of depth_njy at nsigma means sigma_flux = depth_njy / nsigma,
    so sigma_pix = sigma_flux * sqrt(sum P^2). This is how the Legacy Surveys define psfdepth.
    """
    p = np.asarray(psf, dtype=np.float64)
    p = p / p.sum()
    return depth_njy / nsigma * np.sqrt((p**2).sum())


def psfdepth_to_njy(psfdepth, nsigma=5.0):
    """Flux [nJy] of an nsigma point source from a Legacy Surveys psfdepth_<band> value
    [1/nanomaggy^2]: nsigma / sqrt(psfdepth) nanomaggies."""
    return nsigma / np.sqrt(psfdepth) * NANOMAGGY_NJY


def load_stpsf_psf(psf_dir, filt, kind="DET_SAMP", pixel_arcsec=0.03):
    """One of our STPSF PSFs (made by smock-recipes/psfs): (array, header).

    kind: "DET_SAMP" (STPSF's detector-sampled PSF on the 0.03" grid: use it directly, with
        psf_resample_factor = 1, which is within 2% of the exact fine-grid answer and better than
        synthesizer's own supersampling) or "OVERSAMP" (5x finer). Each sums to 1.
        The header has EE_R0P1, the encircled energy in r = 0.1", and STPSF's settings.
    """
    from astropy.io import fits     # imported here so importing smock stays light
    with fits.open(Path(psf_dir) / f"NIRCam_{filt}_{pixel_arcsec}arcsec.fits") as hdul:
        return hdul[kind].data.astype(np.float64), hdul[0].header
