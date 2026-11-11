"""Known-answer tests for the spectrum measurements: the turnover wavelength, line luminosities and the virial
line width. Synthetic spectra on a grid sampled like the stellar and AGN grids (every
~1000 km/s)."""
import numpy as np

from smock.analysis import (C_AA_PER_S, continuum_luminosity_5100, line_checks, line_luminosity,
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

