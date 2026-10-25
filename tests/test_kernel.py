"""Known-answer test: synthesizer's quintic kernel is MP-Gadget's, with SmoothingLength = support radius.

MP-Gadget (libgadget/densitykernel.c) stores the support radius H as `SmoothingLength` and evaluates
W(r) = 1/(120 pi) (3/H)^3 w(3 r/H), with w(q) = (3-q)^5 - 6(2-q)^5 + 15(1-q)^5 (terms dropped beyond
their support). synthesizer's Kernel("quintic") is tabulated on r/h in [0, 1), i.e. h is also the
support radius, so gas smoothing lengths go to synthesizer unchanged (after the unit conversion).
"""
import numpy as np
import pytest
from scipy.integrate import quad

from synthesizer.kernel_functions import quintic


def mp_gadget_quintic(u):
    """MP-Gadget quintic spline at u = r/H (H = support radius = SmoothingLength), H = 1."""
    q = 3.0 * np.asarray(u, dtype=float)
    w = np.where(q < 3, (3 - q) ** 5, 0.0) - np.where(q < 2, 6 * (2 - q) ** 5, 0.0) \
        + np.where(q < 1, 15 * (1 - q) ** 5, 0.0)
    return 27.0 / (120.0 * np.pi) * w


def test_quintic_matches_mp_gadget():
    u = np.linspace(0.0, 0.999, 1000)
    syn = np.array([quintic(x) for x in u])
    # synthesizer's coefficients are rounded to ~8 digits: agreement is ~4e-6, not machine precision
    np.testing.assert_allclose(syn, mp_gadget_quintic(u), rtol=1e-5, atol=1e-6 * mp_gadget_quintic(0.0))
    assert quintic(1.0) == 0.0


def test_quintic_normalised_in_3d():
    norm = quad(lambda r: 4 * np.pi * r * r * quintic(r), 0, 1, points=[1 / 3, 2 / 3])[0]
    assert norm == pytest.approx(1.0, rel=1e-5)
