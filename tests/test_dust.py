"""Known-answer tests for smock.dust.WindowedPowerLaw."""
import numpy as np
import pytest
from unyt import angstrom, um

from synthesizer.emission_models.transformers import PowerLaw

from smock.dust import WindowedPowerLaw


def test_matches_powerlaw_inside_and_is_zero_outside():
    lam = np.array([5.0, 19.9, 20.0, 912.0, 5500.0, 2e4, 1e6, 2e6]) * angstrom
    windowed = WindowedPowerLaw(slope=-1.0).get_tau(lam)
    plain = PowerLaw(slope=-1.0).get_tau(lam)
    inside = (lam.value >= 20.0) & (lam.value <= 1e6)
    assert windowed[inside] == pytest.approx(plain[inside], rel=1e-12)
    assert np.all(windowed[~inside] == 0.0)
    assert windowed[4] == pytest.approx(1.0)          # normalised at 5500 A


def test_slope_and_custom_window():
    curve = WindowedPowerLaw(slope=-2.0, lam_min=0.1 * um, lam_max=1.0 * um)
    tau = curve.get_tau(np.array([500.0, 1100.0, 2750.0, 2e4]) * angstrom)
    assert tau[0] == 0.0 and tau[3] == 0.0
    assert tau[2] == pytest.approx((2750 / 5500) ** -2.0, rel=1e-12)


def test_transmission_is_one_outside_the_window():
    lam = np.array([10.0, 5500.0]) * angstrom
    t = WindowedPowerLaw().get_transmission(np.array([0.5, 2.0]), lam)
    assert t[:, 0] == pytest.approx([1.0, 1.0])
    assert t[:, 1] == pytest.approx(np.exp(-np.array([0.5, 2.0])))


# ---- ISM opacity and dust-to-metal ratio ----------------------------------------------

from unyt import Msun, Myr, unyt_array, yr  # noqa: E402

from smock.dust import VIJAYAN19, dtm_inputs, dtm_vijayan19, kappa_code_for_redshift  # noqa: E402


def test_kappa_code_schedule():
    for z in (0.0, 1.0, 2.0):
        assert kappa_code_for_redshift(z) == pytest.approx(10**2.9, rel=1e-12)
    for z in (4.0, 6.0, 12.0):
        assert kappa_code_for_redshift(z) == pytest.approx(10**4.1, rel=1e-12)
    assert kappa_code_for_redshift(3.0) == pytest.approx(10**3.5, rel=1e-12)   # halfway in log10
    assert kappa_code_for_redshift(2.5) == pytest.approx(10**3.2, rel=1e-12)
    zs = np.linspace(0, 8, 81)
    assert np.all(np.diff([kappa_code_for_redshift(z) for z in zs]) >= 0)
    # a constant schedule
    assert kappa_code_for_redshift(0.5, low=10**4.1) == pytest.approx(10**4.1, rel=1e-12)


def test_dtm_limits():
    p = VIJAYAN19
    assert dtm_vijayan19(0.02, 0.0) == pytest.approx(p["D0"], rel=1e-12)       # no time for grain growth
    assert dtm_vijayan19(0.02, 1e10) == pytest.approx(p["D1"], rel=1e-9)       # old and metal-rich: saturated
    assert dtm_vijayan19(0.0, 1e10) == pytest.approx(p["D0"], rel=1e-12)       # no metals: no growth
    ages = np.logspace(6, 10, 9)
    assert np.all(np.diff(dtm_vijayan19(0.01, ages)) >= 0)                     # grows with age
    assert np.all(np.diff(dtm_vijayan19(np.logspace(-4, -1.5, 9), 3e8)) >= 0)  # and with metallicity


def test_dtm_by_hand():
    # Z = 0.01, Age = 300 Myr: tau = 5e4 / (0.008 * 0.01) = 6.25e8 yr, x = 0.017 * 0.01^-1.337 * 0.48^2.122
    x = 0.017 * 0.01**-1.337 * (3e8 / 6.25e8) ** 2.122
    assert dtm_vijayan19(0.01, 3e8) == pytest.approx(0.008 + 0.321 * (1 - np.exp(-x)), rel=1e-12)


def test_dtm_inputs_weighting():
    gas = {"masses": unyt_array([1.0, 3.0, 4.0], Msun), "metallicities": np.array([0.01, 0.03, 0.5]),
           "star_formation_rates": unyt_array([1.0, 1.0, 0.0], "Msun/yr")}
    stars = {"initial_masses": unyt_array([1.0, 3.0], Msun), "ages": unyt_array([100.0, 500.0], Myr)}
    z, age = dtm_inputs(gas, stars)
    assert z == pytest.approx((0.01 + 3 * 0.03) / 4)          # star-forming gas only
    assert age == pytest.approx((100 + 3 * 500) / 4 * 1e6)    # birth-mass weighted, in yr
    gas["star_formation_rates"] = unyt_array([0.0, 0.0, 0.0], "Msun/yr")
    z, _ = dtm_inputs(gas, stars)
    assert z == pytest.approx((0.01 + 0.09 + 2.0) / 8)        # none star-forming: all the gas
