"""Known-answer tests for smock.plotting: filters chosen by name, the stretch's vmax from the
shown channels only, image orientation and display-window coordinates, pixel sizes in each
unit, and smoke tests of the figures on synthetic files."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from smock.plotting import (apply_norm, asinh_norm, channel_stack, display_window, pixel_size,  # noqa: E402
                            plot_2x2, plot_image_and_spectrum, plot_rgb, plot_spectrum,
                            rgb_filters_for, show_image)

FILTERS = np.array([f"JWST/NIRCam.{f}" for f in ("F090W", "F115W", "F150W", "F200W", "F277W", "F356W", "F444W")])
META = {"redshift": 6.0, "cosmology": {"H0": 67.74, "Om0": 0.3089}, "pixel_arcsec": 0.03,
        "pixel_pkpc": 0.175, "centroid_ij_px": [52.0, 47.0]}


def synthetic_images(n=100):
    """A (filter, x, y) stack where filter k holds the constant k + 1, plus a segmap."""
    stack = np.stack([np.full((n, n), k + 1.0) for k in range(len(FILTERS))])
    return {"filters": FILTERS, "total_noisy": stack, "segmap": np.zeros((n, n), np.int32)}


def test_rgb_filters_by_redshift():
    assert rgb_filters_for(6.5) == ("F444W", "F200W", "F115W")
    assert rgb_filters_for(6.6) == ("F444W", "F277W", "F150W")


def test_channel_stack_picks_by_name():
    stack = synthetic_images()["total_noisy"]
    ch = channel_stack(stack, FILTERS, ("F444W", "JWST/NIRCam.F200W", "F115W"))
    assert [c[0, 0] for c in ch] == [7.0, 4.0, 2.0]


def test_stretch_vmax_from_shown_channels_only():
    stack = synthetic_images()["total_noisy"].copy()
    stack[0] = 1e6                       # F090W, not shown, is far brighter
    channels = channel_stack(stack, FILTERS, ("F444W", "F200W", "F115W"))
    assert asinh_norm(channels).vmax == 7.0
    assert apply_norm(channels, asinh_norm).max() == pytest.approx(1.0)


def test_apply_norm_accepts_a_normalize_and_masks_to_zero():
    channels = np.array([[[0.0, 1.0], [10.0, 100.0]]])
    scaled = apply_norm(channels, LogNorm(vmin=1.0, vmax=100.0))
    assert np.allclose(scaled, [[[0.0, 0.0], [0.5, 1.0]]])


def test_orientation_and_display_coordinates():
    image = np.zeros((10, 8))
    image[7, 2] = 1.0                    # pixel i = 7 (x), j = 2 (y)
    fig, ax = plt.subplots()
    show_image(ax, image[4:, 1:], start=(4, 1))
    shown = ax.images[0]
    assert shown.origin == "lower"
    assert shown.get_array()[2 - 1, 7 - 4] == 1.0          # row = y, column = x
    assert list(shown.get_extent()) == [3.5, 9.5, 0.5, 7.5]      # axes in saved-image indices
    plt.close(fig)


def test_display_window_centres_and_stays_inside():
    start, size = display_window(100)
    assert size == 100 and list(start) == [0, 0]
    start, size = display_window(100, 66.7)
    assert size == 67 and list(start) == [16, 16]
    start, _ = display_window(100, 20, centre=(95.0, 3.0))
    assert list(start) == [80, 0]


def test_pixel_size_units():
    assert pixel_size(META, "arcsec") == 0.03
    assert pixel_size(META, "kpc") == 0.175
    assert pixel_size(META, "ckpc/h") == pytest.approx(0.175 * 7.0 * 0.6774)
    with pytest.raises(ValueError):
        pixel_size(META, "pc")


def test_figures_run_on_synthetic_files():
    images = synthetic_images()
    lam_obs = np.geomspace(5e3, 6e4, 50)
    spectrum = np.zeros(50, dtype=[("lam_obs_AA", float), ("total_nuLnu_rest_erg_s", float),
                                   ("stellar_total_nuLnu_rest_erg_s", float)])
    spectrum["lam_obs_AA"] = lam_obs
    spectrum["total_nuLnu_rest_erg_s"] = 1e44
    spectrum["stellar_total_nuLnu_rest_erg_s"] = 5e43
    ax = plot_rgb(images, META, display_arcsec=2.0, centre="centroid", scale_bar="kpc", label="test")
    assert ax.images[0].get_array().shape == (67, 67, 3)
    plot_2x2(images, META, norm=LogNorm(vmin=1.0, vmax=7.0))
    ax = plot_spectrum(spectrum, META)
    assert [line.get_label() for line in ax.lines] == ["total", "stars, attenuated"]
    assert ax.get_ylim() == pytest.approx((1e43, 1e44 * 10**1.5))
    plot_image_and_spectrum(images, META, spectrum, META)
    plt.close("all")


class TopHat:
    """Stand-in for a synthesizer Filter: a top-hat from lo to hi [um]."""

    def __init__(self, name, lo, hi):
        from unyt import um
        self.filter_code = f"JWST/NIRCam.{name}"
        self.lam = np.linspace(lo - 0.1, hi + 0.1, 200) * um
        self.t = ((self.lam.value >= lo) & (self.lam.value <= hi)).astype(float)


def test_bands_coloured_short_to_long_and_legend_optional():
    spectrum = np.zeros(50, dtype=[("lam_obs_AA", float), ("total_nuLnu_rest_erg_s", float)])
    spectrum["lam_obs_AA"] = np.geomspace(5e3, 6e4, 50)
    spectrum["total_nuLnu_rest_erg_s"] = 1e44
    filters = [TopHat("F444W", 3.9, 5.0), TopHat("F090W", 0.8, 1.0), TopHat("F200W", 1.75, 2.25)]
    ax = plot_spectrum(spectrum, META, filters=filters, legend=None)
    spans = sorted(ax.patches, key=lambda p: p.get_x())
    shortest, longest = spans[0].get_facecolor(), spans[-1].get_facecolor()
    assert shortest[2] > shortest[0] and longest[0] > longest[2]      # blue end, then red end
    assert ax.get_legend() is None
    plt.close("all")
