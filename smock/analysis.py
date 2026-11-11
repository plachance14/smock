"""Segmentation, source choice and the measurements made inside the source.

These steps work on plain arrays, so they live here. The synthesizer steps (images,
property maps) stay in the notebook.

Pixel convention: synthesizer images are indexed [i, j] with i along x (axis 0). Pixel
i covers [-fov/2 + i res, -fov/2 + (i + 1) res], so the galaxy centre sits at array
index (n - 1) / 2 on both axes. Positions here are array indices (i, j) [px], never
photutils' (x, y) = (column, row) order.

    segm = segment(image, sigma_pix)                 # detect + deblend
    label, centre, dist = pick_source(image, segm)   # nearest the image centre
    r_half = half_mass_radius(mass_map, segm.data == label, centre)
    aperture = (segm.data == label) & (pixel_distances(shape, centre) <= 2 * r_half)
    inside = in_mask(positions, aperture, pixel_size)   # which particles count
    c, best, all_c = compactness(image, segm.data == label, centres)
    start = centred_window(n, image.shape[0], centre)   # re-centre by cropping
    saved = cut_window(stack, start, n)
"""
import numpy as np
from scipy.signal import fftconvolve
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.segmentation import SourceCatalog, deblend_sources, detect_sources, detect_threshold

# the old pipeline's settings (AB_final_common.py L582-584, AB_data_gen_final.py L594-599)
NSIGMA = 1.5
NPIXELS = 10
CONNECTIVITY = 4          # photutils 3 defaults to 8; the old pipeline used 4
NLEVELS = 32
CONTRAST = 0.001

# compactness apertures (radii), as published: f(r < 0.2") / f(r < 0.1")
COMPACTNESS_RADII_ARCSEC = (0.1, 0.2)


def image_centre(shape):
    """Array index (i, j) [px] of the galaxy centre: the middle of the image."""
    return np.array([(shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0])


def pixel_distances(shape, centre):
    """Distance [px] of every pixel centre from centre (i, j) [px]."""
    i, j = np.indices(shape, dtype=np.float64)
    return np.hypot(i - centre[0], j - centre[1])


def segment(image, sigma_pix=None, nsigma=NSIGMA, npixels=NPIXELS, connectivity=CONNECTIVITY,
            nlevels=NLEVELS, contrast=CONTRAST, smooth_kernel=None):
    """Detect and deblend sources in a background-subtracted image.

    image:     2D array [any flux unit per pixel]
    sigma_pix: the noise's per-pixel sigma, in the image's units. The threshold is then
               nsigma * sigma_pix (exact for our white depth noise). None lets
               photutils estimate the background and sigma from the image itself, as the old
               pipeline did; that estimate reads high when a bright galaxy fills much of the
               frame.
    smooth_kernel: None, or a 2D kernel with an odd number of pixels on each side (e.g. the
               filter's PSF). Detection and deblending then run on a copy of the image
               convolved with it (normalised to sum to 1), which smooths the segments' edges,
               as in statmorph's tutorial. The threshold still comes from the unsmoothed
               image, since it says whether a source stands out from the real noise; faint
               sources near the limit can be lost. Measurements should use the unsmoothed
               image.
    Returns a photutils SegmentationImage (0 = background, 1, 2, ... = sources), or None
    if nothing passes the threshold.
    """
    if sigma_pix is None:
        threshold = detect_threshold(image, nsigma)
    else:
        threshold = nsigma * sigma_pix
    if smooth_kernel is not None:
        kernel = np.asarray(smooth_kernel, float)
        if kernel.ndim != 2 or not all(n % 2 for n in kernel.shape):
            raise ValueError(f"smooth_kernel must be 2D with odd sides, so it stays centred; got {kernel.shape}")
        image = fftconvolve(image, kernel / kernel.sum(), mode="same")
    segm = detect_sources(image, threshold, npixels, connectivity=connectivity)
    if segm is None:
        return None
    # contrast_method is pinned because photutils warns its default may change.
    return deblend_sources(image, segm, npixels, n_levels=nlevels, contrast=contrast,
                           contrast_method="basin", connectivity=connectivity)


def source_centroids(image, segm):
    """Labels and flux-weighted centroids (i, j) [px] of every source in segm."""
    cat = SourceCatalog(image, segm)
    # photutils' x is the column (axis 1) and y the row (axis 0)
    return np.asarray(cat.label), np.column_stack([cat.y_centroid, cat.x_centroid])


def pick_source(image, segm, target=None):
    """The source whose centroid is nearest target (i, j) [px], default the image centre.

    Returns (label, centroid (i, j) [px], distance [px]). The distance is a true one
    (the old pipeline used the squared distance).
    """
    if target is None:
        target = image_centre(image.shape)
    labels, centroids = source_centroids(image, segm)
    dist = np.hypot(*(centroids - target).T)
    k = np.argmin(dist)
    return int(labels[k]), centroids[k], float(dist[k])


def half_mass_radius(mass_map, mask, centre):
    """Projected half-mass radius [px] of mass_map within mask, about centre (i, j) [px].

    Pixels in the mask are sorted by the distance of their centres from centre, and the
    cumulative mass is interpolated to half the masked total, starting from zero mass at
    r = 0. This fixes the old pipeline's two r_half bugs: the cumulative sum now runs over the
    masked pixels only (it ran over the whole map, biasing r_half low), and a dominant
    central pixel gives a radius inside that pixel (the index became -1, i.e. the
    farthest pixel in the image).
    """
    r = pixel_distances(mass_map.shape, centre)[mask]
    m = np.asarray(mass_map)[mask]
    order = np.argsort(r, kind="stable")
    cum = np.concatenate([[0.0], np.cumsum(m[order])])
    radii = np.concatenate([[0.0], r[order]])
    return float(np.interp(0.5 * cum[-1], cum, radii))


def pixel_indices(positions, shape, pixel_size):
    """Array index (i, j) of the pixel holding each point.

    positions:  (N, 2) offsets from the galaxy centre, in the units of pixel_size
                (e.g. kpc and kpc per pixel). Use the same projection as the image.
    Returns an (N, 2) int array; points outside the image get -1 in both columns.
    """
    shape = np.asarray(shape)
    idx = np.floor(np.asarray(positions) / pixel_size + shape / 2.0).astype(int)
    outside = np.any((idx < 0) | (idx >= shape), axis=1)
    idx[outside] = -1
    return idx


def in_mask(positions, mask, pixel_size):
    """Boolean per point: does its pixel lie in mask? (point positions, as in the old pipeline)."""
    idx = pixel_indices(positions, mask.shape, pixel_size)
    inside = idx[:, 0] >= 0
    out = np.zeros(len(idx), dtype=bool)
    out[inside] = mask[idx[inside, 0], idx[inside, 1]]
    return out


def centred_window(n, n_full, centre):
    """Start index (i0, j0) of the n x n window of an n_full x n_full image that puts
    centre (i, j) [px] within half a pixel of the window's middle, (n - 1) / 2.

    This re-centres images by cropping rather than re-imaging:
    make the images a few pixels wider than wanted, then cut the window out. Half a pixel
    is enough to put the source in the central pixels for inspection. The window is kept
    inside the image, so a centre further off than the buffer is matched only as closely
    as the buffer allows.
    """
    start = np.round(np.asarray(centre, float) - (n - 1) / 2.0).astype(int)
    return np.clip(start, 0, n_full - n)


def cut_window(arr, start, n):
    """The n x n window at start (i0, j0) of the last two axes of arr."""
    return arr[..., start[0]:start[0] + n, start[1]:start[1] + n]


def compactness_centres(centroid, bh=None):
    """The centres compactness is tried at (as published): the source centroid and
    the brightest BH's pixel (i, j) [px], each also shifted by +0.5 px on both axes."""
    centres = [np.asarray(centroid, float), np.asarray(centroid, float) + 0.5]
    if bh is not None:
        centres += [np.asarray(bh, float), np.asarray(bh, float) + 0.5]
    return centres


def aperture_flux(image, centre, radius, method="exact"):
    """Flux inside a circle of radius [px] about centre (i, j) [px].

    method="exact": the exact area of each pixel inside the circle (photutils).
    method="taper": the old pipeline's weights, 1 inside the radius falling linearly to 0 over the
    next half pixel (by pixel-centre distance). It acts like a circle about a quarter
    pixel larger.
    """
    if method == "exact":
        aperture = CircularAperture((centre[1], centre[0]), radius)    # photutils wants (x, y)
        return float(aperture_photometry(image, aperture, method="exact")["aperture_sum"][0])
    if method == "taper":
        d = pixel_distances(image.shape, centre)
        weight = np.clip((radius + 0.5 - d) / 0.5, 0.0, 1.0)
        return float(np.sum(image * weight))
    raise ValueError(f"method must be 'exact' or 'taper', not {method!r}")


def compactness(image, mask, centres, pixel_arcsec, radii_arcsec=COMPACTNESS_RADII_ARCSEC,
                method="exact"):
    """f(r < outer) / f(r < inner) of image within mask, minimised over centres.

    image: the noiseless PSF-convolved one (as published), or the noisy one, as observers measure.
    mask:  the pixels that count; everything else is set to zero. As published, the source's
           segment; observers instead keep the sky and set only other sources' pixels to zero
           (observer_mask). centres: a list of (i, j) [px], e.g. compactness_centres(...) or just
           [centroid]. Returns (minimum, the centre giving it, the ratio at every centre).
    """
    masked = np.where(mask, image, 0.0)
    inner, outer = (r / pixel_arcsec for r in radii_arcsec)
    ratios = np.array([aperture_flux(masked, c, outer, method) / aperture_flux(masked, c, inner, method)
                       for c in centres])
    k = int(np.argmin(ratios))
    return float(ratios[k]), centres[k], ratios


def observer_mask(segm_data, label):
    """The pixels an observer's aperture keeps: everything except other sources' segments.

    Setting only the source's own segment (as published) drops the faint light outside it, which
    lowers the compactness of faint sources (a point source at S/N ~ 14 reads ~1.06 instead of 1.29).
    """
    return ~((segm_data > 0) & (segm_data != label))


def aperture_snr(image, centre, radius_px, sigma_pix):
    """S/N of the flux in a circle of radius_px [px] about centre (i, j), for white noise of sigma_pix
    per pixel (same units as image): flux / (sigma_pix sqrt(pi r^2))."""
    return aperture_flux(image, centre, radius_px) / (sigma_pix * np.sqrt(np.pi * radius_px**2))


def matched_circle_fluxes(raw_images, psf, centre, radius_px, noise=None):
    """Fluxes in a circle on PSF-matched images, as photometric catalogues measure colours.

    raw_images: {band: 2D unconvolved image} on one pixel grid; psf: the broadest band's PSF (F444W's),
    centred, odd-sized; centre (i, j) [px], e.g. the F444W centroid; radius_px [px] (0.18" / pixel for
    Kokorev et al. 2024's 0.36" diameter); noise: None, or {band: 2D noise image} added after the
    convolution. Every band is convolved with the same PSF, so the circle sees the same part of the
    galaxy in each. Returns {band: flux in the circle} (image units). For total fluxes, divide every band
    by the same point_source_fraction(psf, radius_px), as the catalogues do for compact sources: the bands
    share one PSF, so the colours stay those of the circle (correcting one band alone would change them).
    """
    out = {}
    for band, img in raw_images.items():
        matched = fftconvolve(img, psf / psf.sum(), mode="same")
        if noise is not None:
            matched = matched + noise[band]
        out[band] = aperture_flux(matched, centre, radius_px)
    return out


def point_source_fraction(psf, radius_px):
    """Share of a point source's light inside a circle of radius_px [px] about the PSF's middle."""
    mid = np.array([(psf.shape[0] - 1) / 2.0, (psf.shape[1] - 1) / 2.0])
    return aperture_flux(psf / psf.sum(), mid, radius_px)


# ---- the shape of the spectrum -------------------------------------------------------------
# Continuum slopes beta (f_lambda ~ lambda^beta, so a flat f_nu is beta = -2) on the rest-frame
# spectrum, with the lines masked. The spectroscopic LRD papers (Hviding et al. 2025, de Graaff et al.
# 2025) split the UV and optical slopes at the Balmer limit, 3645 A, and fit 1200-3645 and 3645-7000 A.
# split_aa moves the split, e.g. to 3000 A so that the optical slope spans the Balmer break.

# rest wavelengths [A] masked by +-LINE_MASK_AA: the papers' optical lines plus the strong UV ones
CONTINUUM_LINES_AA = (1216, 1240, 1302, 1335, 1397, 1549, 1640, 1663, 1750, 1909, 2326, 2424, 2470, 2798,
                      3346, 3426, 3727, 3869, 3889, 3970, 4026, 4069, 4102, 4340, 4363, 4471, 4686, 4861,
                      4959, 5007, 5199, 5755, 5876, 6300, 6364, 6548, 6563, 6584, 6678, 6717, 6731, 7065,
                      7136, 7325, 7751)
LINE_MASK_AA = 50.0
C_AA_PER_S = 2.99792458e18


def continuum_slope(lam_aa, lnu, lo_aa, hi_aa, lines_aa=CONTINUUM_LINES_AA, mask_aa=LINE_MASK_AA):
    """beta of f_lambda ~ lambda^beta between lo_aa and hi_aa [rest A], from L_nu (any units).

    A straight-line fit of log f_lambda against log lambda with +-mask_aa about each line removed,
    then points more than 3 sigma above the fit (lines not in the list) dropped and the fit repeated
    (up to three times). NaN with fewer than 5 points.
    """
    lam = np.asarray(lam_aa, float)
    flam = np.asarray(lnu, float) * C_AA_PER_S / lam**2
    keep = (lam >= lo_aa) & (lam <= hi_aa) & (flam > 0)
    for l0 in lines_aa:
        keep &= np.abs(lam - l0) > mask_aa
    if keep.sum() < 5:
        return np.nan
    x, y = np.log10(lam[keep]), np.log10(flam[keep])
    for _ in range(3):
        fit = np.polyfit(x, y, 1)
        resid = y - np.polyval(fit, x)
        ok = resid < 3 * np.std(resid) + 1e-12
        if ok.all():
            break
        x, y = x[ok], y[ok]
    return float(np.polyfit(x, y, 1)[0])


def spectral_shape(lam_aa, lnu, split_aa=3645.0, uv_from_aa=1200.0, opt_to_aa=7000.0, agn_lnu=None):
    """Continuum shape of a rest-frame spectrum (lam_aa [A], lnu any L_nu units).

    Returns a dict: beta_uv (uv_from_aa to split_aa), beta_opt (split_aa to opt_to_aa), the Balmer break
    as the mean f_nu in 4000-4100 A over that in 3620-3720 A (de Graaff et al. 2025's windows), the turnover
    wavelength and the slopes either side of it (turnover_wavelength), and, when agn_lnu (the AGN's own L_nu)
    is given, the AGN's share of L_nu at 1500, 3000 and 5000 A.
    """
    lam, lnu = np.asarray(lam_aa, float), np.asarray(lnu, float)
    red, blue = (lam >= 4000) & (lam <= 4100), (lam >= 3620) & (lam <= 3720)
    out = {"beta_uv": continuum_slope(lam, lnu, uv_from_aa, split_aa),
           "beta_opt": continuum_slope(lam, lnu, split_aa, opt_to_aa),
           "balmer_break": float(lnu[red].mean() / lnu[blue].mean()) if red.any() and blue.any() else np.nan,
           "split_aa": float(split_aa), **turnover_wavelength(lam, lnu)}
    if agn_lnu is not None:
        for l0 in (1500, 3000, 5000):
            k = int(np.argmin(np.abs(lam - l0)))
            out[f"agn_share_{l0}"] = float(agn_lnu[k] / lnu[k]) if lnu[k] > 0 else np.nan
    return out


def turnover_wavelength(lam_aa, lnu, lo_aa=1500.0, hi_aa=6000.0, break_aa=(2500.0, 5000.0), step_aa=10.0,
                        lines_aa=CONTINUUM_LINES_AA, mask_aa=LINE_MASK_AA):
    """Where the UV and optical slopes meet: a broken power law in f_nu fitted between lo_aa and hi_aa [rest A],
    the break free between break_aa (Setton et al. 2024, sec. 3.2: 0.15-0.6 um, break between 0.25 and 0.5 um).

    lam_aa [A], lnu (any L_nu units); lines masked as in continuum_slope. The two power laws meet at the break;
    each trial break is a straight-line fit of log f_nu against log lambda on each side, and the break with the
    smallest squared residual wins. Returns a dict: turnover_aa, and the slopes alpha_uv and alpha_opt of
    f_nu ~ lambda^alpha (beta = alpha - 2). NaN with fewer than 5 points on a side.
    """
    lam, lnu = np.asarray(lam_aa, float), np.asarray(lnu, float)
    keep = (lam >= lo_aa) & (lam <= hi_aa) & (lnu > 0)
    for l0 in lines_aa:
        keep &= np.abs(lam - l0) > mask_aa
    x, y = np.log10(lam[keep]), np.log10(lnu[keep])
    best = (np.inf, np.nan, np.nan, np.nan)
    for lb in np.arange(break_aa[0], break_aa[1] + step_aa / 2, step_aa):
        xb = np.log10(lb)
        blue, red = x < xb, x >= xb
        if blue.sum() < 5 or red.sum() < 5:
            continue
        # one intercept at the break, two slopes: y = a + s1 min(x - xb, 0) + s2 max(x - xb, 0)
        design = np.column_stack([np.ones_like(x), np.minimum(x - xb, 0), np.maximum(x - xb, 0)])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = float(np.sum((y - design @ coef) ** 2))
        if resid < best[0]:
            best = (resid, lb, coef[1], coef[2])
    return {"turnover_aa": float(best[1]), "alpha_uv": float(best[2]), "alpha_opt": float(best[3])}


# ---- emission lines --------------------------------------------------------------------------
# Measured on the rest-frame spectrum, before any broadening, as the excess over a straight-line continuum.
# The grids are sampled every ~1000 km/s, so a narrow line sits in one bin and its neighbours ([N II] next to
# H-alpha, for example) blend with it, much as at NIRSpec PRISM's resolution.

LINES_AA = {"Halpha": 6562.80, "Hbeta": 4861.32, "OIII_5007": 5006.84, "OI_8446": 8446.4,
            "Pabeta": 12818.1, "Paalpha": 18751.0}
C_KMS = 2.99792458e5
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def line_luminosity(lam_aa, lnu_erg_s_hz, line_aa, half_width_kms=1500.0, side_kms=(3000.0, 6000.0)):
    """Luminosity [erg/s] of the line at line_aa [rest A] in a rest-frame spectrum (lam_aa [A], L_nu [erg/s/Hz]).

    The excess of L_lambda over a straight line fitted to the continuum between side_kms on both sides, summed
    over the bins whose centres lie within half_width_kms of the line, each bin weighted by its width. NaN if
    a side has fewer than 2 bins.
    """
    lam = np.asarray(lam_aa, float)
    llam = np.asarray(lnu_erg_s_hz, float) * C_AA_PER_S / lam**2            # erg/s/A
    dv = (lam - line_aa) / line_aa * C_KMS
    edges = np.concatenate([[lam[0]], 0.5 * (lam[1:] + lam[:-1]), [lam[-1]]])
    width = np.diff(edges)                                                  # A
    side = (np.abs(dv) >= side_kms[0]) & (np.abs(dv) <= side_kms[1])
    if (side & (dv < 0)).sum() < 2 or (side & (dv > 0)).sum() < 2:
        return np.nan
    fit = np.polyfit(lam[side], llam[side], 1)
    inside = np.abs(dv) <= half_width_kms
    return float(np.sum((llam[inside] - np.polyval(fit, lam[inside])) * width[inside]))


def continuum_luminosity_5100(lam_aa, lnu_erg_s_hz, lo_aa=5050.0, hi_aa=5150.0):
    """lambda L_lambda [erg/s] at 5100 A rest: the median L_lambda between lo_aa and hi_aa (no strong lines
    there) times 5100 A."""
    lam = np.asarray(lam_aa, float)
    llam = np.asarray(lnu_erg_s_hz, float) * C_AA_PER_S / lam**2
    k = (lam >= lo_aa) & (lam <= hi_aa)
    return float(5100.0 * np.median(llam[k])) if k.any() else np.nan


def line_checks(lam_aa, lnu_erg_s_hz):
    """The line measurements observers compare LRDs on, from a rest-frame spectrum (lam_aa [A], L_nu [erg/s/Hz]).

    Returns a dict: L_<line> [erg/s] for LINES_AA (NaN outside the spectrum); L_5100 [erg/s]; EW_Halpha [A]
    (rest); Halpha_Hbeta (the Balmer decrement; LRDs: mean and median ~8.7, de Graaff et al. 2025 sec. 4.2);
    and the ratios to de Graaff et al. 2025's LRD relations, Halpha_vs_dG25 = L_Halpha / [10^43.24 (L_5100 /
    10^44)^0.91] (their Eq. 2, scatter 0.23 dex) and OI_vs_dG25 = L_OI_8446 / [10^41.13 (L_Halpha / 10^43)^1.27]
    (Eq. 6), so 1 is on the relation.
    """
    out = {f"L_{name}": line_luminosity(lam_aa, lnu_erg_s_hz, l0) for name, l0 in LINES_AA.items()}
    l5100 = continuum_luminosity_5100(lam_aa, lnu_erg_s_hz)
    ha, hb, oi = out["L_Halpha"], out["L_Hbeta"], out["L_OI_8446"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out |= {"L_5100": l5100,
                "EW_Halpha": float(ha / (l5100 / 5100.0)) if l5100 > 0 else np.nan,   # rough: continuum at 5100 A
                "Halpha_Hbeta": float(ha / hb) if hb > 0 else np.nan,
                "Halpha_vs_dG25": float(ha / (10**43.24 * (l5100 / 1e44) ** 0.91)) if l5100 > 0 else np.nan,
                "OI_vs_dG25": float(oi / (10**41.13 * (ha / 1e43) ** 1.27)) if ha > 0 else np.nan}
    return out


def virial_fwhm_kms(m_bh_msun, l_halpha_erg_s, epsilon=1.075):
    """The broad H-alpha FWHM [km/s] that observers' virial mass estimate would give for this black hole: Reines &
    Volonteri 2015 Eq. 1 (Reines et al. 2013 Eq. 5, with Bentz et al. 2013's radius-luminosity relation) solved
    for the width, log M_BH = log epsilon + 6.57 + 0.47 log(L_Halpha / 10^42 erg/s) + 2.06 log(FWHM / 10^3 km/s),
    epsilon = 1.075. m_bh_msun [Msun], l_halpha_erg_s [erg/s] (works on arrays). NaN where L <= 0."""
    m, lha = np.asarray(m_bh_msun, float), np.asarray(l_halpha_erg_s, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_fwhm = (np.log10(m) - np.log10(epsilon) - 6.57 - 0.47 * np.log10(lha / 1e42)) / 2.06 + 3.0
        return np.where((lha > 0) & (m > 0), 10**log_fwhm, np.nan)


def v_shaped(beta_uv, beta_opt, max_beta_uv=-0.2, min_beta_opt=0.0, min_difference=0.5):
    """The V-shaped continuum of Hviding et al. 2025 and de Graaff et al. 2025 (without their
    statistical margins): beta_uv < max_beta_uv, beta_opt > min_beta_opt and beta_opt - beta_uv >
    min_difference. Kocevski et al. 2024's photometric version is max_beta_uv = -0.37,
    min_difference = 0 (and beta_uv > -2.8)."""
    with np.errstate(invalid="ignore"):
        return (np.asarray(beta_uv) < max_beta_uv) & (np.asarray(beta_opt) > min_beta_opt) & \
               (np.asarray(beta_opt) - np.asarray(beta_uv) > min_difference)


# ---- is it a little red dot? ---------------------------------------------------------------
# Selections on the source's AB magnitudes (a dict, band -> magnitude, e.g. {"F444W": 26.1, ...};
# NaN where the flux is not positive, which fails every cut) and its compactness. They work on
# numbers or on arrays (one entry per galaxy). The walkthrough's optional step 15 applies one
# (LRD_SELECTION, "ambra" by default) and only reports the answer; nothing else depends on it.

# Each selection has two halves that can be mixed: its colours (with the paper's other photometric cuts) and its
# compactness. lrd_compact holds the compactness rules; each lrd_selection_* applies its own when given a value and
# skips it when given None, so lrd_selection_ambra(mags, None) is the AMBRA colours alone.
LRD_COMPACTNESS_RULES = {
    "ambra": 'f(<0.2") / f(<0.1") in F444W <= 1.7 (and >= 1e-4)',
    "kokorev24": 'the same as "ambra" (Kokorev et al. 2024 write < 1.7)',
    "vshape": 'the same as "ambra" (de Graaff et al. 2025 use f(r<0.2") / f(r<0.1") < 1.7)',
    "rinaldi25": 'f(d<0.5") / f(d<0.25") in F444W < 1.7 (RINALDI25_RADII_ARCSEC)',
    "rinaldi26": "fitted F444W effective radius <= 0.06 arcsec",
}


def lrd_compact(rule, value):
    """Whether a source is compact under one of LRD_COMPACTNESS_RULES; value is that rule's measurement (the
    flux ratio, or the effective radius in arcsec for "rinaldi26"); NaN fails. Works on arrays."""
    v = np.asarray(value, dtype=float)
    with np.errstate(invalid="ignore"):
        if rule in ("ambra", "kokorev24", "vshape"):
            return (v >= 1e-4) & (v <= 1.7)
        if rule == "rinaldi25":
            return v < 1.7
        if rule == "rinaldi26":
            return v <= 0.06
    raise ValueError(f"compactness rule {rule!r}: use one of {sorted(LRD_COMPACTNESS_RULES)}")


def lrd_selection_vshape(beta_uv, beta_opt):
    """A spectroscopic selection: the V-shaped continuum of de Graaff et al. 2025 (sec. 2.1; Hviding et al. 2025 use
    the same slope cuts), beta_opt > 0, beta_uv < -0.2 and beta_opt - beta_uv > 0.5, with the slopes measured on the
    rest-frame spectrum (spectral_shape: 1200-3645 and 3645-7000 A, lines masked; they fit the two power laws
    joined at the Balmer limit, here they are fitted separately, and they ask for each cut in 95% of their fit's
    samples). Their compactness is applied separately ("vshape" in lrd_compact). Returns a dict: v_shaped, lrd."""
    v = v_shaped(beta_uv, beta_opt)
    return dict(v_shaped=v, lrd=v)


def lrd_selection_ambra(mags, compactness):
    """The selection of LaChance et al. 2026 (arXiv:2609.26999; LRDs in AMBRA), adapted from UNCOVER's.

    lenient (the paper's LRDs): F444W < 28, compactness within [1e-4, 1.7], and red in one of
        red1: F115W - F150W < 0.8 and F200W - F356W > 1.0
        red2: F150W - F200W < 0.8 and F277W - F444W > 1.0
    strict: as lenient, but red1 also needs F200W - F277W > 0.7 and red2 F277W - F356W > 0.7.
    Returns a dict of booleans: red1, red2, strict_red1, strict_red2, compact, lenient, strict,
    and lrd (= lenient). The paper also applies lrd_source_quality.
    """
    m = {band: np.asarray(value, dtype=float) for band, value in mags.items()}
    c = np.asarray(compactness, dtype=float)
    with np.errstate(invalid="ignore"):             # NaN compares False
        bright = m["F444W"] < 28.0
        red1 = (m["F115W"] - m["F150W"] < 0.8) & (m["F200W"] - m["F356W"] > 1.0)
        red2 = (m["F150W"] - m["F200W"] < 0.8) & (m["F277W"] - m["F444W"] > 1.0)
        strict_red1 = red1 & (m["F200W"] - m["F277W"] > 0.7)
        strict_red2 = red2 & (m["F277W"] - m["F356W"] > 0.7)
        compact = lrd_compact("ambra", c) if compactness is not None else np.ones_like(bright)
    lenient = bright & (red1 | red2) & compact
    return dict(red1=red1, red2=red2, strict_red1=strict_red1, strict_red2=strict_red2, compact=compact,
                lenient=lenient, strict=bright & (strict_red1 | strict_red2) & compact, lrd=lenient)


def lrd_selection_kokorev24(mags, compactness, snr=None, min_snr=14.0):
    """Kokorev et al. 2024 (arXiv:2401.09981, sec. 3.1), adapted from Labbe et al. 2023:

        red1: F115W - F150W < 0.8 and F200W - F277W > 0.7 and F200W - F356W > 1.0
        red2: F150W - F200W < 0.8 and F277W - F356W > 0.6 and F277W - F444W > 0.7
        compact: f(r < 0.2") / f(r < 0.1") in F444W < 1.7 (treated as <= 1.7, the same rule as "ambra")
        not a brown dwarf: F115W - F200W > -0.5
        F444W < 27.7
        detected: S/N > min_snr (14) in F444W, when snr is given (measured on the noisy image, e.g.
                  aperture_snr in a 0.32" diameter circle, as UNCOVER's)

    Their other cuts (a PSF-dominated profile, photometric redshifts) are left out. Returns a dict
    of booleans: red1, red2, compact, not_brown_dwarf, detected, lrd.
    """
    m = {band: np.asarray(value, dtype=float) for band, value in mags.items()}
    c = np.asarray(compactness, dtype=float)
    with np.errstate(invalid="ignore"):
        red1 = (m["F115W"] - m["F150W"] < 0.8) & (m["F200W"] - m["F277W"] > 0.7) & (m["F200W"] - m["F356W"] > 1.0)
        red2 = (m["F150W"] - m["F200W"] < 0.8) & (m["F277W"] - m["F356W"] > 0.6) & (m["F277W"] - m["F444W"] > 0.7)
        not_brown_dwarf = m["F115W"] - m["F200W"] > -0.5
        bright = m["F444W"] < 27.7
        compact = lrd_compact("kokorev24", c) if compactness is not None else np.ones_like(bright)
        detected = np.ones_like(bright) if snr is None else np.asarray(snr, dtype=float) > min_snr
    return dict(red1=red1, red2=red2, compact=compact, not_brown_dwarf=not_brown_dwarf, detected=detected,
                lrd=(red1 | red2) & compact & not_brown_dwarf & bright & detected)


RINALDI25_RADII_ARCSEC = (0.125, 0.25)      # their compactness: f(d < 0.5") / f(d < 0.25"), radii in arcsec


def lrd_selection_rinaldi25(mags, compactness, redshift=None):
    """Rinaldi et al. 2025 (arXiv:2411.14383, sec. 2.2), LRDs in JADES GOODS-N and GOODS-S:

        blue slope: F150W - F200W < 0.8
        red slope: F277W - F444W > 0.7
        not a brown dwarf: F115W - F200W > -0.5
        compact: f(d < 0.5") / f(d < 0.25") in F444W < 1.7 (RINALDI25_RADII_ARCSEC; a different
                 ratio from the other selections' f(< 0.2") / f(< 0.1"), so measure it at these radii)
        redshift between 4 and 8, when redshift is given (theirs are photometric)

    They measured colours in 0.5"-diameter circles. Their visual check of each source and its SED,
    which kept 99 of about 350 candidates, is left out, so this selection lets in more than theirs.
    Returns a dict of booleans: blue, red, not_brown_dwarf, compact, in_redshift, lrd.
    """
    m = {band: np.asarray(value, dtype=float) for band, value in mags.items()}
    c = np.asarray(compactness, dtype=float)
    with np.errstate(invalid="ignore"):
        blue = m["F150W"] - m["F200W"] < 0.8
        red = m["F277W"] - m["F444W"] > 0.7
        not_brown_dwarf = m["F115W"] - m["F200W"] > -0.5
        compact = lrd_compact("rinaldi25", c) if compactness is not None else np.ones_like(blue)
        in_redshift = (np.ones_like(compact) if redshift is None
                       else (np.asarray(redshift, dtype=float) >= 4.0) & (np.asarray(redshift, dtype=float) <= 8.0))
    return dict(blue=blue, red=red, not_brown_dwarf=not_brown_dwarf, compact=compact, in_redshift=in_redshift,
                lrd=blue & red & not_brown_dwarf & compact & in_redshift)


def lrd_selection_rinaldi26(mags, r_eff_arcsec, snr=None, max_r_eff_arcsec=0.06, min_snr=3.0):
    """Rinaldi et al. 2026 (arXiv:2604.07138, sec. 3.2.4), LRDs in JADES DR5; Rinaldi et al. 2025's colours relaxed:

        blue slope: F150W - F200W < 1.0
        red slope: F277W - F444W > 0.5
        compact: a fitted size instead of a flux ratio, the F444W effective radius r_eff_arcsec [arcsec]
                 <= max_r_eff_arcsec (0.06", "consistent with being unresolved"); they fit a single Sersic
                 profile with the PSF (pysersic). NaN (no good fit) fails
        not a brown dwarf: F115W - F200W > -0.5 (Greene et al. 2024), and outside Hainline et al. 2025's
                 brown-dwarf locus: F277W - F444W < 0.4 (F115W - F277W) + 1.9 and F115W - F277W > -0.5
        detected: S/N > min_snr (3) in F444W, when snr is given

    Their visual check of each candidate (which kept 321 of 816) is left out, so this selection lets in
    more than theirs. Returns a dict of booleans: blue, red, compact, not_brown_dwarf, detected, lrd.
    """
    m = {band: np.asarray(value, dtype=float) for band, value in mags.items()}
    r = np.asarray(r_eff_arcsec, dtype=float)
    with np.errstate(invalid="ignore"):
        blue = m["F150W"] - m["F200W"] < 1.0
        red = m["F277W"] - m["F444W"] > 0.5
        compact = (r <= max_r_eff_arcsec) if r_eff_arcsec is not None else np.ones_like(blue)
        uv_ir = m["F115W"] - m["F277W"]
        not_brown_dwarf = ((m["F115W"] - m["F200W"] > -0.5) & (m["F277W"] - m["F444W"] < 0.4 * uv_ir + 1.9)
                           & (uv_ir > -0.5))
        detected = np.ones_like(blue) if snr is None else np.asarray(snr, dtype=float) > min_snr
    return dict(blue=blue, red=red, compact=compact, not_brown_dwarf=not_brown_dwarf, detected=detected,
                lrd=blue & red & compact & not_brown_dwarf & detected)


def lrd_source_quality(segmentation_ok, bh_mass, stellar_mass, subfind_stellar_mass, offset_ckpch,
                       min_bh_mass=10.0, min_stellar_mass=1e5, max_mass_ratio=5.0, max_offset_ckpch=10.0):
    """LaChance et al. 2026's checks that the source is a well-measured galaxy with a black hole.

    bh_mass [Msun]: of the black holes inside the aperture (>= min_bh_mass: there is one);
    stellar_mass [Msun]: inside the aperture (>= min_stellar_mass);
    subfind_stellar_mass [Msun]: the subhalo's, from the catalogue. stellar_mass / it must be
        <= max_mass_ratio, so that the source isn't mostly other galaxies' stars;
    offset_ckpch [comoving kpc/h]: from the catalogue position to the source's centroid
        (<= max_offset_ckpch), so that the source found is the subhalo we asked for.
    The old pipeline's offset cut, 40, was applied to a squared distance (no square root),
    which made it about 7 pixels, 5.6-6.0 ckpc/h at z = 5-8; the default here is a little
    looser, as the old cut was meant to be.
    Returns a dict of booleans, one per check, and ok (all of them).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.asarray(stellar_mass, float) / np.asarray(subfind_stellar_mass, float)
        checks = dict(segmented=np.asarray(segmentation_ok, bool),
                      has_bh=np.asarray(bh_mass, float) >= min_bh_mass,
                      has_stars=np.asarray(stellar_mass, float) >= min_stellar_mass,
                      mass_ratio=ratio <= max_mass_ratio,
                      centred=np.asarray(offset_ckpch, float) <= max_offset_ckpch)
    checks["ok"] = np.logical_and.reduce(list(checks.values()))
    return checks
