"""Figures of the saved images and spectra (Phase 4): RGB composites, a 2x2 of the RGB and its
three channels, the spectrum, and the image next to the spectrum.

Everything reads the files written by smock.io (load_images, load_spectrum), so a figure can
be remade without re-running synthesizer. Matplotlib only; no global style changes.

Pixel convention (as in smock.analysis): image stacks are (filter, x, y), so pixel (i, j)
has i along x. Panels show image.T with origin="lower", and their axes are in the saved
image's pixel indices (i, j), even after a display crop. Positions from the meta, such as
"centroid_ij_px", can therefore be drawn directly (add_circle).

The colour normalisation is the user's choice: pass any matplotlib Normalize (AsinhNorm,
LogNorm, PowerNorm, ...), or a function that takes the three displayed channels and returns
one, as asinh_norm does. It is applied to the three channels together, so their relative
brightness is kept.

    images, meta = load_images(path)
    ax = plot_rgb(images, meta, display_arcsec=2.0, scale_bar="kpc", label="z = 6")
    add_circle(ax, meta["centroid_ij_px"], 0.2 / meta["pixel_arcsec"])
    fig = plot_2x2(images, meta)
    spectrum, photometry, meta = load_spectrum(path)
    ax = plot_spectrum(spectrum, meta, filters=filters)
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import AsinhNorm, Normalize
from matplotlib.ticker import NullFormatter, ScalarFormatter

# R, G, B by redshift, as in ambra_lrd's rgb_image: F444W always red; green and blue move
# redder above z = 6.5
RGB_FILTERS_LOW_Z = ("F444W", "F200W", "F115W")
RGB_FILTERS_HIGH_Z = ("F444W", "F277W", "F150W")
RGB_SWITCH_Z = 6.5

# text drawn on images (size label, free-text label, filter names) [pt]; always bold
IMAGE_FONTSIZE = 18

# spectrum components drawn by default: (label, colour, linestyle); missing ones are skipped
SPECTRUM_STYLES = {
    "total": ("total", "0.3", "-"),
    "stellar_total": ("stars, attenuated", "tab:green", "-"),
    "reprocessed": ("stars, intrinsic", "tab:green", "--"),
    "AGN_total": ("AGN, attenuated", "tab:blue", "-"),
    "AGN_reprocessed": ("AGN, intrinsic", "tab:blue", "--"),
}

# spectrum legend: matplotlib legend arguments; outside the axes on the right, with no frame,
# so it never covers the spectrum (the figure grows to fit it). Inside: e.g. dict(loc=3, ncol=2)
SPECTRUM_LEGEND = dict(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)


def rgb_filters_for(z):
    """Default (R, G, B) filter names for redshift z (ambra_lrd's choice)."""
    return RGB_FILTERS_LOW_Z if z <= RGB_SWITCH_Z else RGB_FILTERS_HIGH_Z


def asinh_norm(channels, linear_frac=0.03):
    """ambra_lrd's default stretch: asinh with vmin = 0, vmax = the brightest pixel of the
    displayed channels, and a linear width of linear_frac * vmax (the image's units)."""
    vmax = float(np.max(channels))
    return AsinhNorm(linear_width=linear_frac * vmax, vmin=0.0, vmax=vmax, clip=True)


def channel_stack(stack, filters, names):
    """The images of the named filters, (len(names), n, n), from a (filter, x, y) stack.

    filters are the stack's filter codes (e.g. "JWST/NIRCam.F444W"); names are short names
    ("F444W") or full codes. Chosen by name, never by position, since filter order differs
    between pipelines.
    """
    short = [str(f).split(".")[-1] for f in filters]
    return np.stack([stack[short.index(name.split(".")[-1])] for name in names])


def apply_norm(channels, norm):
    """channels scaled to [0, 1] by norm (a Normalize, or a function of channels returning one).

    Values the norm masks (e.g. LogNorm below vmin) become 0.
    """
    if not isinstance(norm, Normalize):
        norm = norm(channels)
    # scale-based norms (AsinhNorm, LogNorm) only take 1-D input
    scaled = np.ma.filled(norm(channels.ravel()), 0.0).reshape(channels.shape)
    return np.clip(scaled, 0.0, 1.0)


def rgb_array(stack, filters, rgb=("F444W", "F200W", "F115W"), norm=asinh_norm):
    """(n, n, 3) RGB array in [0, 1], indexed (x, y, colour), from a (filter, x, y) stack."""
    return np.moveaxis(apply_norm(channel_stack(stack, filters, rgb), norm), 0, -1)


def display_window(n, display_px=None, centre=None):
    """Square display window on an n x n image: start (i, j) [px] and its size [px].

    display_px None shows the whole image. centre (i, j) [px] defaults to the image centre;
    the window is shifted to stay inside the image.
    """
    size = n if display_px is None else min(int(round(display_px)), n)
    centre = np.full(2, (n - 1) / 2.0) if centre is None else np.asarray(centre, dtype=float)
    start = np.clip(np.round(centre - (size - 1) / 2.0).astype(int), 0, n - size)
    return start, size


def show_image(ax, image, start=(0, 0), **imshow_kwargs):
    """Draw an (x, y[, 3]) image whose pixel (0, 0) is pixel `start` of the saved image.

    Axes are in saved-image pixel indices (i along x, j along y). Ticks are off.
    """
    n_i, n_j = image.shape[:2]
    extent = (start[0] - 0.5, start[0] + n_i - 0.5, start[1] - 0.5, start[1] + n_j - 0.5)
    ax.imshow(np.swapaxes(image, 0, 1), origin="lower", extent=extent, **imshow_kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def pixel_size(meta, unit):
    """Size of one pixel in unit: "arcsec", "kpc" (physical) or "ckpc/h" (comoving)."""
    if unit == "arcsec":
        return meta["pixel_arcsec"]
    if unit == "kpc":
        return meta["pixel_pkpc"]
    if unit == "ckpc/h":
        return meta["pixel_pkpc"] * (1 + meta["redshift"]) * meta["cosmology"]["H0"] / 100.0
    raise ValueError(f"unit must be 'arcsec', 'kpc' or 'ckpc/h', not {unit!r}")


def add_scale_bar(ax, width, unit, y=0.94, color="white", fontsize=IMAGE_FONTSIZE):
    """A double-headed arrow across the panel's full width, labelled with that width.

    width: the panel's width in unit (e.g. n_pixels * pixel_size(meta, unit)).
    y: height in axes fraction.
    """
    ax.annotate("", xy=(0.0, y), xytext=(1.0, y), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="<->", color=color, lw=2))
    label = f'{width:.1f}"' if unit == "arcsec" else f"{width:.1f} {unit}"
    ax.text(0.5, y - 0.02, label, transform=ax.transAxes, ha="center", va="top", color=color,
            fontsize=fontsize, fontweight="bold")


def add_label(ax, text, loc="lower left", color="white", fontsize=IMAGE_FONTSIZE):
    """Free text in a corner of the panel: loc is "upper/lower left/right"."""
    vert, horiz = loc.split()
    x, ha = (0.03, "left") if horiz == "left" else (0.97, "right")
    y, va = (0.03, "bottom") if vert == "lower" else (0.97, "top")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, color=color, fontsize=fontsize,
            fontweight="bold")


def add_circle(ax, centre, radius_px, color="cyan", linestyle="--", label=None, **kwargs):
    """A circle of radius_px [px] at centre (i, j) [px, saved-image indices], e.g. an aperture.

    For a radius in arcsec, divide by meta["pixel_arcsec"]. label, if given, goes in the legend.
    """
    circle = plt.Circle(tuple(centre), radius_px, fill=False, color=color, linestyle=linestyle,
                        label=label, **kwargs)
    ax.add_patch(circle)
    return circle


def _display(images, meta, stage, display_arcsec, centre):
    """The stage's stack and its display window (start, size) [px]."""
    stack = images[stage]
    display_px = None if display_arcsec is None else display_arcsec / meta["pixel_arcsec"]
    centre = meta["centroid_ij_px"] if centre == "centroid" else None if centre == "image" else centre
    start, size = display_window(stack.shape[1], display_px, centre)
    return stack[:, start[0]:start[0] + size, start[1]:start[1] + size], start, size


def plot_rgb(images, meta, stage="total_noisy", rgb=None, norm=asinh_norm, display_arcsec=None,
             centre="image", scale_bar=None, label=None, ax=None, fontsize=IMAGE_FONTSIZE):
    """One RGB panel of an images file (smock.io.load_images).

    stage:          which stack, e.g. "total_noisy", "total_psf", "stellar_psf"
    rgb:            (R, G, B) filter names; None picks them from the redshift (rgb_filters_for)
    norm:           a matplotlib Normalize, or a function of the three channels returning one
    display_arcsec: width shown [arcsec]; None shows the whole saved image
    centre:         "image", "centroid" (meta["centroid_ij_px"]) or (i, j) [px]
    scale_bar:      None, or the unit of the size arrow: "arcsec", "kpc" or "ckpc/h"
    label:          free text in the lower-left corner
    fontsize:       of the text on the image [pt]; it is bold
    Returns the axes; their coordinates are saved-image pixel indices.
    """
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    rgb = rgb or rgb_filters_for(meta["redshift"])
    stack, start, size = _display(images, meta, stage, display_arcsec, centre)
    show_image(ax, rgb_array(stack, images["filters"], rgb, norm), start)
    if scale_bar:
        add_scale_bar(ax, size * pixel_size(meta, scale_bar), scale_bar, fontsize=fontsize)
    if label:
        add_label(ax, label, fontsize=fontsize)
    return ax


def plot_2x2(images, meta, stage="total_noisy", rgb=None, norm=asinh_norm, display_arcsec=None,
             centre="image", scale_bar=None, label=None, cmap="Greys", fontsize=IMAGE_FONTSIZE):
    """The RGB (upper left) and its three channels in grey (the thesis-talk layout).

    Each channel is shown with the RGB's normalisation, so the panels compare directly; the
    default "Greys" shows bright as black. Other arguments as plot_rgb. Returns the figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), gridspec_kw=dict(wspace=0, hspace=0))
    rgb = rgb or rgb_filters_for(meta["redshift"])
    plot_rgb(images, meta, stage, rgb, norm, display_arcsec, centre, scale_bar, label, axes[0, 0], fontsize)
    stack, start, _ = _display(images, meta, stage, display_arcsec, centre)
    channels = apply_norm(channel_stack(stack, images["filters"], rgb), norm)
    for ax, name, channel in zip(axes.flat[1:], rgb, channels):
        show_image(ax, channel, start, cmap=cmap, vmin=0.0, vmax=1.0)
        add_label(ax, name, loc="lower right", color="black" if cmap == "Greys" else "white",
                  fontsize=fontsize)
    return fig


def plot_spectrum(spectrum, meta, components=None, filters=None, lam_range_um=(0.8, 5.5),
                  y_range_dex=(1.0, 1.5), legend=SPECTRUM_LEGEND, band_cmap="rainbow", band_alpha=0.25,
                  ax=None, fontsize=14):
    """Rest-frame nu L_nu [erg/s] against observed wavelength [um], from a spectrum file
    (smock.io.load_spectrum).

    components:   {component: (legend label, colour, linestyle)}; default SPECTRUM_STYLES,
                  skipping components the file doesn't hold
    filters:      a synthesizer FilterCollection; if given, each band is shaded where its
                  transmission is above half its peak, and named
    band_cmap:    colormap for the bands, sampled from short to long wavelength
                  ("rainbow": purple to red); band_alpha is their opacity
    lam_range_um: observed wavelength range shown
    y_range_dex:  (below, above): the y axis spans this many dex below and above the total's
                  median in that range (a median, so strong lines don't set the scale)
    legend:       matplotlib legend arguments (default SPECTRUM_LEGEND, outside the axes), or
                  None for no legend
    """
    ax = ax or plt.subplots(figsize=(9, 5))[1]
    lam_um = spectrum["lam_obs_AA"] / 1e4
    shown = (lam_um >= lam_range_um[0]) & (lam_um <= lam_range_um[1])
    for name, (line_label, colour, style) in (components or SPECTRUM_STYLES).items():
        column = f"{name}_nuLnu_rest_erg_s"
        if column in spectrum.dtype.names and np.any(spectrum[column][shown] > 0):
            ax.plot(lam_um[shown], spectrum[column][shown], color=colour, ls=style, lw=1.2, label=line_label,
                    zorder=3 if name == "total" else 2)
    if filters is not None:
        bands = sorted(((f.lam.to("um").value[f.t > 0.5 * f.t.max()], f) for f in filters),
                       key=lambda b: b[0].min())
        colours = plt.get_cmap(band_cmap)(np.linspace(0.0, 1.0, len(bands)))
        for (band, f), colour in zip(bands, colours):
            ax.axvspan(band.min(), band.max(), color=colour, alpha=band_alpha, lw=0, zorder=0)
            ax.text(np.sqrt(band.min() * band.max()), 0.97, f.filter_code.split(".")[-1],
                    transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=fontsize - 3)
    middle = np.median(spectrum["total_nuLnu_rest_erg_s"][shown])
    ax.set(xscale="log", yscale="log", xlim=lam_range_um,
           ylim=(middle / 10 ** y_range_dex[0], middle * 10 ** y_range_dex[1]))
    ax.set_xticks([t for t in (0.8, 1, 1.5, 2, 3, 4, 5) if lam_range_um[0] <= t <= lam_range_um[1]])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"observed wavelength [$\mu$m]", fontsize=fontsize)
    ax.set_ylabel(r"rest-frame $\nu L_\nu$ [erg s$^{-1}$]", fontsize=fontsize)
    if legend is not None:
        ax.legend(**{"fontsize": fontsize - 2, **legend})
    return ax


def plot_image_and_spectrum(images, image_meta, spectrum, spectrum_meta, rgb_kwargs=None,
                            spectrum_kwargs=None):
    """The RGB panel beside the spectrum (ambra_lrd's layout, widths 0.35 : 0.65).

    rgb_kwargs and spectrum_kwargs go to plot_rgb and plot_spectrum. Returns the figure.
    """
    fig, (ax_img, ax_spec) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw=dict(width_ratios=(0.35, 0.65)))
    plot_rgb(images, image_meta, ax=ax_img, **(rgb_kwargs or {}))
    plot_spectrum(spectrum, spectrum_meta, ax=ax_spec, **(spectrum_kwargs or {}))
    return fig
