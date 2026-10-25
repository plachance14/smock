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
