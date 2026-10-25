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

    image is the noiseless PSF-convolved one (as published), mask the source's
    segment, centres a list of (i, j) [px]. Returns (minimum, the centre giving it,
    the ratio at every centre).
    """
    masked = np.where(mask, image, 0.0)
    inner, outer = (r / pixel_arcsec for r in radii_arcsec)
    ratios = np.array([aperture_flux(masked, c, outer, method) / aperture_flux(masked, c, inner, method)
                       for c in centres])
    k = int(np.argmin(ratios))
    return float(ratios[k]), centres[k], ratios


# ---- is it a little red dot? ---------------------------------------------------------------
# Selections on the source's AB magnitudes (a dict, band -> magnitude, e.g. {"F444W": 26.1, ...};
# NaN where the flux is not positive, which fails every cut) and its compactness. They work on
# numbers or on arrays (one entry per galaxy). Nothing in the walkthrough applies them by default.

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
        compact = (c >= 1e-4) & (c <= 1.7)
    lenient = bright & (red1 | red2) & compact
    return dict(red1=red1, red2=red2, strict_red1=strict_red1, strict_red2=strict_red2, compact=compact,
                lenient=lenient, strict=bright & (strict_red1 | strict_red2) & compact, lrd=lenient)


def lrd_selection_kokorev24(mags, compactness):
    """Kokorev et al. 2024 (arXiv:2401.09981, sec. 3.1), adapted from Labbe et al. 2023:

        red1: F115W - F150W < 0.8 and F200W - F277W > 0.7 and F200W - F356W > 1.0
        red2: F150W - F200W < 0.8 and F277W - F356W > 0.6 and F277W - F444W > 0.7
        compact: f(r < 0.2") / f(r < 0.1") in F444W < 1.7
        not a brown dwarf: F115W - F200W > -0.5
        F444W < 27.7

    Their other cuts need a real observation (> 14 sigma in F444W, a PSF-dominated profile,
    photometric redshifts) and are left out. Returns a dict of booleans: red1, red2, compact,
    not_brown_dwarf, lrd.
    """
    m = {band: np.asarray(value, dtype=float) for band, value in mags.items()}
    c = np.asarray(compactness, dtype=float)
    with np.errstate(invalid="ignore"):
        red1 = (m["F115W"] - m["F150W"] < 0.8) & (m["F200W"] - m["F277W"] > 0.7) & (m["F200W"] - m["F356W"] > 1.0)
        red2 = (m["F150W"] - m["F200W"] < 0.8) & (m["F277W"] - m["F356W"] > 0.6) & (m["F277W"] - m["F444W"] > 0.7)
        compact = c < 1.7
        not_brown_dwarf = m["F115W"] - m["F200W"] > -0.5
        bright = m["F444W"] < 27.7
    return dict(red1=red1, red2=red2, compact=compact, not_brown_dwarf=not_brown_dwarf,
                lrd=(red1 | red2) & compact & not_brown_dwarf & bright)


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
