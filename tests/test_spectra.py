"""Known-answer tests for the spectrum measurements: the turnover wavelength, line luminosities, the virial line
width and the PRISM-like blurring. Synthetic spectra on a grid sampled like the stellar and AGN grids (every
~1000 km/s)."""
import numpy as np

from smock.analysis import (C_AA_PER_S, continuum_luminosity_5100, degrade_resolution, line_checks, line_luminosity,
                            turnover_wavelength, virial_fwhm_kms)

LAM = np.exp(np.arange(np.log(900.0), np.log(30000.0), 1000.0 / 2.99792458e5))     # A, every 1000 km/s


def _lnu_from_llam(llam):
    return llam * LAM**2 / C_AA_PER_S


def _add_line(llam, l0, lum):
    """Put a line of luminosity lum [erg/s] into the bin nearest l0 [A]."""
    k = int(np.argmin(np.abs(LAM - l0)))
    edges = np.concatenate([[LAM[0]], 0.5 * (LAM[1:] + LAM[:-1]), [LAM[-1]]])
    out = llam.copy()
    out[k] += lum / (edges[k + 1] - edges[k])
    return out


def test_turnover_of_a_broken_power_law():
    lnu = np.where(LAM < 3645.0, (LAM / 3645.0) ** 0.5, (LAM / 3645.0) ** 2.0)       # f_nu ~ lambda^alpha
    t = turnover_wavelength(LAM, lnu)
    assert abs(t["turnover_aa"] - 3645.0) <= 20.0
    assert abs(t["alpha_uv"] - 0.5) < 0.02 and abs(t["alpha_opt"] - 2.0) < 0.02


def test_line_luminosity_of_a_line_on_a_sloped_continuum():
    llam = 1e40 * (LAM / 5000.0) ** -1.3                     # erg/s/A
    lum = 3e42
    got = line_luminosity(LAM, _lnu_from_llam(_add_line(llam, 6562.8, lum)), 6562.8)
    assert abs(got / lum - 1) < 0.01
    assert abs(line_luminosity(LAM, _lnu_from_llam(llam), 6562.8)) < 1e-3 * lum    # no line: nothing


def test_line_checks_on_de_graaff_relation():
    llam = 1e40 * (LAM / 5100.0) ** 0.5
    l5100 = 5100.0 * 1e40
    l_ha = 10**43.24 * (l5100 / 1e44) ** 0.91
    spec = _add_line(_add_line(llam, 6562.8, l_ha), 4861.32, l_ha / 8.0)
    c = line_checks(LAM, _lnu_from_llam(spec))
    assert abs(c["L_5100"] / l5100 - 1) < 0.01
    assert abs(c["Halpha_vs_dG25"] - 1) < 0.02
    assert abs(c["Halpha_Hbeta"] - 8.0) < 0.1
    assert abs(continuum_luminosity_5100(LAM, _lnu_from_llam(llam)) / l5100 - 1) < 0.01


def test_virial_fwhm_inverts_reines_volonteri():
    fwhm, lha = 2000.0, 1e42
    m = 10 ** (np.log10(1.075) + 6.57 + 0.47 * np.log10(lha / 1e42) + 2.06 * np.log10(fwhm / 1e3))
    assert abs(virial_fwhm_kms(m, lha) / fwhm - 1) < 1e-10
    assert np.isnan(virial_fwhm_kms(m, 0.0))


def test_degrade_resolution_keeps_flux_and_sets_the_width():
    lam = np.linspace(1.0, 5.0, 40001)                       # um
    flam = np.zeros_like(lam)
    flam[20000] = 1.0 / (lam[1] - lam[0])                    # a narrow line at 3 um, total 1
    out = degrade_resolution(lam, flam, lambda l: np.full_like(l, 100.0))
    assert abs(np.trapezoid(out, lam) - 1) < 1e-3
    half = out >= out.max() / 2
    fwhm = lam[half][-1] - lam[half][0]
    assert abs(fwhm / (3.0 / 100.0) - 1) < 0.05               # FWHM = lambda / R


def _toy_table():
    """A shutter table whose share is 1 inside a 0.2" x 0.46" box and falls off outside, the same at two wavelengths
    except for a factor (the centred share), which the weights must divide out."""
    off = np.round(np.arange(-1.0, 1.0001, 0.02), 4)
    dy, dx = np.meshgrid(off, off, indexing="ij")
    box = np.exp(-np.maximum(np.abs(dx) - 0.1, 0) / 0.05 - np.maximum(np.abs(dy) - 0.23, 0) / 0.05)
    return {"lam_um": np.array([1.0, 4.0]), "offset_arcsec": off, "share": np.array([0.8 * box, 0.5 * box])}


def test_shutter_weights_and_spectrum():
    from smock.analysis import shutter_offsets, shutter_spectrum, shutter_weight_maps, table_wavelength_coefficients
    table = _toy_table()
    dx, dy = shutter_offsets(*np.indices((61, 61)), (30, 30), 0.03)
    w = shutter_weight_maps(table, dx, dy)
    assert w.shape == (2, 61, 61) and np.allclose(w[:, 30, 30], 1.0)          # the centre keeps everything
    assert w[0, 30, 0] < 1e-3                                                  # 0.9" across the shutter: almost nothing
    assert abs(w[0, 30 + 7, 30] - 1.0) < 1e-6                                  # 0.21" along the length: still inside
    coef = table_wavelength_coefficients(table["lam_um"], np.array([0.5, 1.0, 2.5, 4.0, 6.0]))
    assert np.allclose(coef.sum(axis=0), 1.0) and np.allclose(coef[:, 2], [0.5, 0.5])
    cube = np.zeros((61, 61, 5))
    cube[30, 30] = 1.0                                                         # a point source at the centre
    cube[30, 0] = 1.0                                                          # and one far off across the shutter
    assert np.allclose(shutter_spectrum(cube, w, coef), 1.0, atol=1e-3)
    dx90, dy90 = shutter_offsets(*np.indices((61, 61)), (30, 30), 0.03, pa_deg=90.0)        # turned: the length now runs along j
    assert abs(shutter_weight_maps(table, dx90, dy90)[0, 30, 30 + 7] - 1.0) < 1e-6
