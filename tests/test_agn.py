"""Known-answer tests for smock.agn: the thin-disc temperatures and the grid check."""
import numpy as np
import pytest
from unyt import unyt_array

from smock.agn import (GRID_FLAGS, disc_max_temperature, disc_temperature_at, grid_check,
                       ionising_photons_per_lbol, old_pipeline_ionisation_parameter)

# SI constants, independent of unyt (CODATA 2018; IAU 2015 nominal solar mass). unyt's values
# differ in the fifth digit, so comparisons with these are good to ~1e-4.
G_SI, C_SI, SIGMA_SI = 6.67430e-11, 2.99792458e8, 5.670374419e-8
MSUN_KG, YR_S = 1.98841e30, 3.15576e7          # Julian year, as unyt uses


def thin_disc_scale_temperature(m_msun, mdot_msun_yr, r_isco_rg=6.0):
    """T* = [3 G M Mdot / (8 pi sigma r_isco^3)]^(1/4) [K], in plain SI."""
    m, mdot = m_msun * MSUN_KG, mdot_msun_yr * MSUN_KG / YR_S
    r = r_isco_rg * G_SI * m / C_SI**2
    return (3 * G_SI * m * mdot / (8 * np.pi * SIGMA_SI * r**3)) ** 0.25


def test_t_max_is_the_peak_of_the_thin_disc_profile():
    # T(r)^4 ∝ r^-3 (1 - sqrt(r_in/r)) peaks at r = (49/36) r_in, where T = T* (36/49)^(3/4) (1/7)^(1/4)
    m, mdot = 1e7, 0.1
    expected = thin_disc_scale_temperature(m, mdot) * (36 / 49) ** 0.75 * (1 / 7) ** 0.25
    assert disc_max_temperature(m, mdot).to("K").value == pytest.approx(expected, rel=1e-4)
    radii = np.linspace(1.01, 3.0, 400)
    profile = [disc_temperature_at(m, mdot, x).value for x in radii]
    assert radii[np.argmax(profile)] == pytest.approx(49 / 36, abs=0.01)


def test_t_max_matches_synthesizers_scale_temperature():
    # synthesizer 1.3.1's bb_temperature = 2.24e9 (Mdot/Msun yr^-1)^(1/4) (M/Msun)^(-1/2) K is
    # T* for r_isco = 6 r_g; T_max = 0.4878 T*. 2.24e9 is rounded.
    m, mdot = 10**6.5, 0.02
    t_star_synthesizer = 2.24e9 * mdot**0.25 * m**-0.5
    ratio = disc_max_temperature(m, mdot).value / t_star_synthesizer
    assert ratio == pytest.approx((36 / 49) ** 0.75 * (1 / 7) ** 0.25, rel=2e-3)


def test_matches_the_old_formula_with_the_exact_sigma():
    # the old pipeline's T_r at rmax = (49/36) isco, with SB_const 5.670e-8 (the old code had 5.680e-8)
    m, mdot = 3e6, 0.005
    m_kg, mdot_si = m * MSUN_KG, mdot * MSUN_KG / YR_S
    isco = 6.0 * G_SI * m_kg / C_SI**2
    rmax = (49.0 / 36.0) * isco
    t_old = ((3 * G_SI * m_kg * mdot_si) / (8 * np.pi * SIGMA_SI * rmax**3)) ** 0.25 \
        * (1 - np.sqrt(isco / rmax)) ** 0.25
    assert disc_max_temperature(m, mdot).value == pytest.approx(t_old, rel=1e-4)


def test_radius_units_agree():
    m, mdot = 1e8, 1.0
    t = [disc_temperature_at(m, mdot, 2.0, "r_isco").value,
         disc_temperature_at(m, mdot, 12.0, "r_g").value,
         disc_temperature_at(m, mdot, 6.0, "R_s").value]
    assert t == pytest.approx([t[0]] * 3, rel=1e-12)
    assert disc_temperature_at(m, mdot, 49 / 36).value == pytest.approx(disc_max_temperature(m, mdot).value,
                                                                         rel=1e-12)


def test_r_isco_option_moves_the_disc_inwards():
    # T_max ∝ r_isco^(-3/4) at fixed M and Mdot
    m, mdot = 1e7, 0.1
    hotter = disc_max_temperature(m, mdot, r_isco=3.0).value / disc_max_temperature(m, mdot).value
    assert hotter == pytest.approx(2 ** 0.75, rel=1e-12)


def test_scalings_and_zero_accretion():
    t0 = disc_max_temperature(1e6, 0.01).value
    assert disc_max_temperature(4e6, 0.01).value == pytest.approx(t0 / 2, rel=1e-12)      # M^-1/2
    assert disc_max_temperature(1e6, 0.16).value == pytest.approx(t0 * 2, rel=1e-12)      # Mdot^1/4
    assert disc_max_temperature(np.array([1e6, 1e6]), np.array([0.0, 0.01])).value == \
        pytest.approx([0.0, t0], rel=1e-12)


def test_unyt_and_plain_inputs_agree():
    plain = disc_max_temperature(1e6, 0.01).value
    with_units = disc_max_temperature(unyt_array(1e6 * 1.98841e33, "g"), unyt_array(10.0, "Msun/kyr"))
    assert with_units.value == pytest.approx(plain, rel=1e-4)


def test_bad_radius_or_unit_raises():
    with pytest.raises(ValueError):
        disc_temperature_at(1e6, 0.01, 0.5)              # inside r_isco
    with pytest.raises(ValueError):
        disc_temperature_at(1e6, 0.01, 10.0, "pc")


# ---- grid_check ----------------------------------------------------------------------

def _toy_grid(path):
    """A 3 x 2 grid: T (log_on_read, K) x column (linear), with a limit cycle at (2, 1)."""
    import h5py
    with h5py.File(path, "w") as f:
        f.attrs["axes"] = ["big_bump_temperature", "column_densities"]
        f["axes/big_bump_temperature"] = [1e4, 1e5, 1e6]
        f["axes/big_bump_temperature"].attrs["log_on_read"] = True
        f["axes/column_densities"] = [23.0, 24.0]
        f["axes/column_densities"].attrs["log_on_read"] = False
        flags = np.zeros((3, 2), dtype=np.int64)
        flags[2, 1] = GRID_FLAGS["limit_cycle"]
        flags[0, 0] = GRID_FLAGS["rerun"]          # not a warning flag
        f["smock_flags"] = flags
    return path


def test_grid_check_flags_and_range(tmp_path):
    path = _toy_grid(tmp_path / "toy.hdf5")
    report = grid_check(path, big_bump_temperature=np.array([10**5.5, 1e5, 10**3.5, 1e6]),
                        column_densities=np.array([23.5, 24.0, 23.0, 25.0]))
    assert report[0] == {"outside": [], "flags": ["limit_cycle"]}     # between T 1e5-1e6, N_H 23-24
    assert report[1] == {"outside": [], "flags": []}                  # exactly on (1e5, 24): one point
    assert report[2] == {"outside": ["big_bump_temperature"], "flags": []}   # clamped to 1e4; rerun ignored
    assert report[3] == {"outside": ["column_densities"], "flags": ["limit_cycle"]}  # clamped to (1e6, 24)


def test_grid_check_needs_every_axis(tmp_path):
    path = _toy_grid(tmp_path / "toy.hdf5")
    with pytest.raises(ValueError):
        grid_check(path, big_bump_temperature=np.array([1e5]))


def test_grid_check_takes_bh_attribute_names_and_units(tmp_path):
    # singular names (as BlackHoles attributes are), unyt values converted, scalars broadcast
    path = _toy_grid(tmp_path / "toy.hdf5")
    report = grid_check(path, big_bump_temperature=unyt_array([1e-2, 10**5.5 / 1e3], "MK") * 1e3,
                        column_density=23.5, accretion_rates=np.zeros(2))
    assert [r["flags"] for r in report] == [[], ["limit_cycle"]]


def test_old_pipeline_ionisation_parameter_matches_its_cloudy_runs():
    # Q(H) per erg of L_bol at log T = 4.2, 5.0, 6.0 (incident grid, alpha_ox -1.4, alpha_uv -0.5)
    # against the log U CLOUDY printed for the old runs (turbulence/24_5_NH_9_5_hden, temp_*.out)
    log_q = np.array([8.131, 9.497, 9.966])
    log_u = np.log10(old_pipeline_ionisation_parameter(log_q))
    # independently: log U = log Q + 43.5 - log(4 pi) - 2 x 16.7 - 9.5 - log c [cm/s]
    expected = log_q + 43.5 - np.log10(4 * np.pi) - 33.4 - 9.5 - np.log10(C_SI * 100)
    assert log_u == pytest.approx(expected, abs=1e-4)
    assert log_u == pytest.approx([-2.84, -1.48, -1.01], abs=0.01)


def test_ionising_photons_per_lbol_interpolates_in_log_t(tmp_path):
    import h5py
    path = tmp_path / "incident.hdf5"
    with h5py.File(path, "w") as f:
        f["axes/big_bump_temperature"] = [1e4, 1e5]
        f["log10_specific_ionising_luminosity/HI"] = [8.0, 9.0]
    assert ionising_photons_per_lbol(path, [1e4, 10**4.5, 1e5]) == pytest.approx([8.0, 8.5, 9.0])

