"""Known-answer tests for smock.analysis: segmentation and source choice, r_half, the
particle -> pixel lookup (against synthesizer's own), and compactness apertures."""
import numpy as np
import pytest
from unyt import kpc

from smock.analysis import (aperture_flux, centred_window, compactness, compactness_centres, cut_window,
                            half_mass_radius, image_centre, in_mask, pick_source, pixel_distances,
                            lrd_selection_ambra, lrd_selection_kokorev24, lrd_source_quality,
                            pixel_indices, segment)


def gaussian_blob(shape, centre, sigma, total=1.0):
    """Circular Gaussian sampled at pixel centres, centre (i, j) [px], summing to ~total."""
    i, j = np.indices(shape, dtype=float)
    g = np.exp(-((i - centre[0]) ** 2 + (j - centre[1]) ** 2) / (2 * sigma**2))
    return total * g / (2 * np.pi * sigma**2)


def test_image_centre_and_distances():
    assert np.allclose(image_centre((100, 100)), [49.5, 49.5])
    d = pixel_distances((3, 4), (1.0, 1.0))
    assert d[1, 1] == 0.0 and d[1, 3] == 2.0 and d[0, 0] == pytest.approx(np.sqrt(2))


def test_segment_two_sources_and_pick_nearest():
    rng = np.random.default_rng(1)
    shape, sigma_pix = (100, 100), 1.0
    true = {"near": (52.3, 47.6), "far": (20.0, 75.0)}
    image = (gaussian_blob(shape, true["near"], 3.0, 2000.0) + gaussian_blob(shape, true["far"], 2.0, 800.0)
             + rng.normal(0.0, sigma_pix, shape))
    segm = segment(image, sigma_pix)
    assert segm.n_labels == 2
    label, centroid, dist = pick_source(image, segm)
    # the centroid of a bright Gaussian, over its segment, lands within a tenth of a pixel
    assert np.allclose(centroid, true["near"], atol=0.1)
    assert dist == pytest.approx(np.hypot(52.3 - 49.5, 47.6 - 49.5), abs=0.1)     # a true distance
    assert segm.data[52, 48] == label
    # another target picks the other source
    label_far, centroid_far, _ = pick_source(image, segm, target=(20, 75))
    assert label_far != label and np.allclose(centroid_far, true["far"], atol=0.15)


def test_deblending_splits_touching_sources():
    shape = (60, 60)
    image = gaussian_blob(shape, (30, 25), 2.5, 1000.0) + gaussian_blob(shape, (30, 34), 2.5, 1000.0)
    one_threshold_island = segment(image, 0.1, nlevels=1, contrast=1.0)     # deblending effectively off
    assert one_threshold_island.n_labels == 1
    assert segment(image, 0.1).n_labels == 2


@pytest.mark.filterwarnings("ignore::photutils.utils.exceptions.NoDetectionsWarning")
def test_nothing_detected():
    rng = np.random.default_rng(2)
    assert segment(rng.normal(0.0, 1.0, (50, 50)), 1.0, nsigma=10.0) is None



def test_smoothing_with_a_one_pixel_kernel_changes_nothing():
    rng = np.random.default_rng(3)
    shape = (80, 80)
    image = gaussian_blob(shape, (40, 38), 3.0, 1500.0) + rng.normal(0.0, 1.0, shape)
    plain, smoothed = segment(image, 1.0), segment(image, 1.0, smooth_kernel=np.ones((1, 1)))
    assert np.array_equal(plain.data, smoothed.data)


@pytest.mark.filterwarnings("ignore::photutils.utils.exceptions.NoDetectionsWarning")
def test_smoothing_keeps_the_unsmoothed_threshold():
    # a lone 5 sigma pixel passes 1.5 sigma; smoothed with a sigma = 1.5 px Gaussian it peaks at
    # 5 x 0.07 = 0.35 sigma, so it is lost. (Rescaling the threshold to the smoothed noise,
    # 1.5 sigma x sqrt(sum k^2) = 0.28 sigma, would have kept it.)
    image = np.zeros((41, 41))
    image[20, 20] = 5.0
    assert segment(image, 1.0, npixels=1).n_labels == 1
    kernel = gaussian_blob((15, 15), (7, 7), 1.5)
    assert segment(image, 1.0, npixels=1, smooth_kernel=kernel) is None


def test_smoothing_kernel_must_have_odd_sides():
    with pytest.raises(ValueError):
        segment(np.zeros((20, 20)), 1.0, smooth_kernel=np.ones((4, 5)))

def test_half_mass_radius_of_gaussian():
    # a 2D Gaussian holds half its mass inside sigma * sqrt(2 ln 2)
    shape, sigma = (201, 201), 8.0
    centre = (100.0, 100.0)
    mass = gaussian_blob(shape, centre, sigma)
    mask = np.ones(shape, bool)
    assert half_mass_radius(mass, mask, centre) == pytest.approx(sigma * np.sqrt(2 * np.log(2)), rel=0.01)


def test_half_mass_radius_ignores_mass_outside_mask():
    # the old pipeline summed the whole map, so a neighbour inside the radius pulled r_half down
    shape, sigma = (201, 201), 8.0
    centre = (100.0, 100.0)
    mass = gaussian_blob(shape, centre, sigma)
    neighbour = gaussian_blob(shape, (100.0, 108.0), 2.0, 0.5)
    mask = pixel_distances(shape, (100.0, 108.0)) > 6.0               # the neighbour's segment is cut out
    alone = half_mass_radius(np.where(mask, mass, 0.0), mask, centre)
    assert half_mass_radius(mass + neighbour, mask, centre) == pytest.approx(alone, rel=1e-3)


def test_half_mass_radius_dominant_central_pixel():
    # one pixel holds 1.0 of 1.1: half the total (0.55) is reached 0.55 of the way to that
    # pixel's centre, interpolating from zero mass at r = 0 (the old pipeline gave the farthest pixel)
    mass = np.zeros((21, 21))
    mass[10, 10] = 1.0
    mass[0, 0] = 0.1                                                  # a far, small pixel
    centre = (10.3, 10.4)
    r_first = np.hypot(0.3, 0.4)
    assert half_mass_radius(mass, np.ones_like(mass, bool), centre) == pytest.approx(0.55 * r_first)


def test_pixel_indices_match_synthesizer_histogram():
    # synthesizer's own histogram image decides which pixel each point lands in
    from synthesizer.imaging import Image

    rng = np.random.default_rng(3)
    res, fov = 0.7, 10 * 0.7                                          # 10 x 10 pixels
    pos = rng.uniform(-fov / 2 + 1e-6, fov / 2 - 1e-6, (200, 3))
    pos[:, 2] = 0.0
    idx = pixel_indices(pos[:, :2], (10, 10), res)
    for k in range(len(pos)):
        # a zero-weight mirror point keeps the coordinates centred, as synthesizer expects
        img = Image(resolution=res * kpc, fov=fov * kpc)
        img.generate_img_hist(signal=np.array([1.0, 0.0]), coordinates=np.vstack([pos[k], -pos[k]]) * kpc)
        assert tuple(np.argwhere(img.arr == 1.0)[0]) == tuple(idx[k])


def test_pixel_indices_outside_and_in_mask():
    idx = pixel_indices(np.array([[0.0, 0.0], [-5.1, 0.0], [4.9, 4.9]]), (10, 10), 1.0)
    assert idx.tolist() == [[5, 5], [-1, -1], [9, 9]]
    mask = np.zeros((10, 10), bool)
    mask[5, 5] = True
    assert in_mask(np.array([[0.2, 0.7], [-0.2, 0.0], [9.0, 0.0]]), mask, 1.0).tolist() == [True, False, False]


def test_exact_aperture_on_flat_image():
    flat = np.ones((40, 40))
    for r in (10 / 3, 20 / 3):
        assert aperture_flux(flat, (19.5, 19.5), r) == pytest.approx(np.pi * r**2, rel=1e-9)


def test_taper_aperture_is_a_quarter_pixel_larger():
    # the old pipeline's taper has area pi r^2 + int_r^{r+1/2} 2 pi d (r + 1/2 - d) / (1/2) dd
    # = pi (r^2 + r/2 + 1/12): about a circle of radius r + 1/4. Pixel-centre sampling adds
    # ~1% lattice noise for one centre, so average over random sub-pixel centres.
    rng = np.random.default_rng(4)
    flat = np.ones((40, 40))
    r = 20 / 3
    areas = [aperture_flux(flat, 19.5 + rng.uniform(-0.5, 0.5, 2), r, "taper") for _ in range(200)]
    assert np.mean(areas) == pytest.approx(np.pi * (r**2 + r / 2 + 1 / 12), rel=2e-3)


def test_compactness_known_answers():
    shape, pix = (100, 100), 0.03
    mask = np.ones(shape, bool)
    centre = (49.5, 49.5)
    # a flat image: ratio of the aperture areas, (0.2 / 0.1)^2
    c, _, _ = compactness(np.ones(shape), mask, [centre], pix)
    assert c == pytest.approx(4.0, rel=1e-9)
    # a Gaussian: (1 - exp(-r2^2 / 2 s^2)) / (1 - exp(-r1^2 / 2 s^2))
    s = 5.0
    r1, r2 = 0.1 / pix, 0.2 / pix
    expected = (1 - np.exp(-r2**2 / (2 * s**2))) / (1 - np.exp(-r1**2 / (2 * s**2)))
    c, _, _ = compactness(gaussian_blob(shape, centre, s), mask, [centre], pix)
    assert c == pytest.approx(expected, rel=2e-3)


def test_compactness_takes_the_minimum_centre():
    shape, pix = (100, 100), 0.03
    image = gaussian_blob(shape, (49.5, 49.5), 3.0)
    centres = compactness_centres((49.5, 49.5), bh=(55.0, 55.0))
    assert len(centres) == 4
    c, best, ratios = compactness(image, np.ones(shape, bool), centres, pix)
    assert np.allclose(best, (49.5, 49.5)) and c == ratios.min() and ratios[2] > c   # off-centre is less compact


def test_centred_window_recentres_by_cropping():
    n, buffer = 100, 5
    n_full = n + 2 * buffer
    # the image centre gives the central window
    start = centred_window(n, n_full, image_centre((n_full, n_full)))
    assert start.tolist() == [buffer, buffer]
    # an off-centre point lands within half a pixel of the window's middle
    for centre in [(56.26, 54.0), (50.1, 58.49), (59.4, 50.6)]:
        start = centred_window(n, n_full, centre)
        assert np.all(np.abs(np.asarray(centre) - start - (n - 1) / 2) <= 0.5)
    # further off than the buffer: the window stops at the image edge
    assert centred_window(n, n_full, (80.0, 30.0)).tolist() == [2 * buffer, 0]


def test_cut_window_keeps_pixels():
    stack = np.arange(3 * 12 * 12).reshape(3, 12, 12)
    cut = cut_window(stack, np.array([1, 2]), 10)
    assert cut.shape == (3, 10, 10) and cut[2, 0, 0] == stack[2, 1, 2] and cut[0, 9, 9] == stack[0, 10, 11]


# ---- LRD selections ------------------------------------------------------------------------

def _mags(**colours):
    """Magnitudes from F444W = 26 and the colours given (band minus F444W); flat otherwise."""
    bands = ("F090W", "F115W", "F150W", "F200W", "F277W", "F356W", "F444W")
    return {b: 26.0 + colours.get(b, 0.0) for b in bands}


def test_lrd_ambra_red1_lenient_not_strict():
    # blue in F115W-F150W (0.2), red in F200W-F356W (1.5), but F200W-F277W only 0.5
    m = _mags(F115W=1.7, F150W=1.5, F200W=1.5, F277W=1.0, F356W=0.0)
    s = lrd_selection_ambra(m, 1.3)
    assert s["red1"] and not s["red2"] and s["lenient"] and s["lrd"] and not s["strict"]


def test_lrd_ambra_red2_strict_and_cuts():
    # F150W-F200W 0.3, F277W-F444W 1.6, F277W-F356W 0.8
    m = _mags(F090W=2.0, F115W=2.0, F150W=1.9, F200W=1.6, F277W=1.6, F356W=0.8)
    assert lrd_selection_ambra(m, 1.3)["strict"]
    assert not lrd_selection_ambra(m, 1.8)["lrd"]                       # not compact
    assert not lrd_selection_ambra({**m, "F444W": 28.5}, 1.3)["lrd"]    # too faint
    assert not lrd_selection_ambra({**m, "F150W": np.nan}, 1.3)["red2"]  # no flux: fails


def test_lrd_kokorev24():
    m = _mags(F115W=1.6, F150W=1.5, F200W=1.5, F277W=0.7, F356W=0.0)   # red1: 0.1, 0.8, 1.5
    assert lrd_selection_kokorev24(m, 1.5)["lrd"]
    assert not lrd_selection_kokorev24(m, 1.7)["lrd"]                   # strictly below 1.7
    bd = {**m, "F115W": 26.0 + 0.9}                                     # F115W-F200W = -0.6
    s = lrd_selection_kokorev24(bd, 1.5)
    assert not s["not_brown_dwarf"] and not s["lrd"]


def test_lrd_selection_on_arrays():
    m = {b: np.array([v, v + 5.0]) for b, v in _mags(F115W=1.7, F150W=1.5, F200W=1.5, F277W=1.0).items()}
    assert lrd_selection_ambra(m, np.array([1.3, 1.3]))["lrd"].tolist() == [True, False]   # 2nd: F444W 31


def test_lrd_source_quality():
    good = dict(segmentation_ok=True, bh_mass=1e6, stellar_mass=1e8, subfind_stellar_mass=1e8, offset_ckpch=1.0)
    assert lrd_source_quality(**good)["ok"]
    assert lrd_source_quality(**{**good, "offset_ckpch": 9.5})["ok"]               # default limit: 10 ckpc/h
    assert not lrd_source_quality(**{**good, "offset_ckpch": 10.5})["centred"]
    assert not lrd_source_quality(**{**good, "offset_ckpch": 5.5}, max_offset_ckpch=5.0)["ok"]
    assert not lrd_source_quality(**{**good, "stellar_mass": 6e8})["mass_ratio"]
    assert not lrd_source_quality(**{**good, "bh_mass": 0.0})["has_bh"]
    assert not lrd_source_quality(**{**good, "segmentation_ok": False})["ok"]
