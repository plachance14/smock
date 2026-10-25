"""Black-hole quantities for the AGN grid: the disc temperature, and a check against the grid.

The AGN grids are built on CLOUDY's `agn` continuum, whose big-bump temperature T is an axis
of the grid. These functions give each BH its T from the simulation's mass and accretion
rate. (Its L_bol, the grid's weight, synthesizer's BlackHoles computes itself from the
accretion rate: epsilon * Mdot c^2.) They are thin-disc formulae (Shakura & Sunyaev 1973):

    T(r)^4 = 3 G M Mdot / (8 pi sigma r^3) * (1 - sqrt(r_in / r)),   r_in = r_isco

which peaks at r = (49/36) r_isco. That peak, T_max, is the default, as in the
author's old pipeline. The old code's Stefan-Boltzmann constant had a typo (5.680e-8);
the exact value is used here.

r_isco is in units of r_g = GM/c^2 and defaults to 6 (a non-spinning BH). That is not
consistent with a radiative efficiency of 0.1 (ASTRID's and AMBRA's), which implies r_isco ~ 3.5 r_g
(spin ~ 0.7); the mismatch is kept from the old pipeline and documented, not fixed.

Inputs may be unyt quantities or plain numbers in the stated units. Nothing is read on import.

`old_pipeline_ionisation_parameter` is for reproducing the old pipeline (testing only). Its
CLOUDY runs fixed the source's luminosity and the gas's distance (`luminosity 43.5 total`,
`radius 16.7`, n_H = 10^9.5) rather than U, so their U rose with the disc temperature: log U
-2.84 at log T 4.2, -1.48 at 5.0, -1.01 at 6.0 (from the runs' own output). The AGN grids fix
U instead. Q(H) per unit L_bol at each T comes from the incident grid
(`ionising_photons_per_lbol`).

`grid_check` reports, per BH, whether it falls outside the AGN grid (synthesizer clamps such
values to the edge without a warning) and which flagged grid points its interpolation uses.
The flags are smock's own `smock_flags` dataset, which synthesizer does not read.
"""
import h5py
import numpy as np
from unyt import G, c, unyt_array
from unyt.physical_constants import stefan_boltzmann_constant as sigma_sb

R_ISCO_RG = 6.0                    # innermost stable orbit, in GM/c^2 (non-spinning)
T_MAX_RADIUS = 49.0 / 36.0         # the thin disc's hottest radius, in units of r_in

# radius units for disc_temperature_at: lengths in GM/c^2 for each name
_RADIUS_UNITS_RG = {"r_g": 1.0, "R_s": 2.0}


def _values(x, unit):
    """Plain float array of x in `unit` (x may be a unyt quantity or a number in `unit`)."""
    if hasattr(x, "units"):
        return np.asarray(x.to(unit).value, dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def _thin_disc_temperature(mass, accretion_rate, r_rg, r_isco_rg):
    """Thin-disc temperature [K] at radius r_rg (in GM/c^2) around a disc ending at r_isco_rg."""
    if np.any(np.asarray(r_rg) < r_isco_rg):
        raise ValueError(f"radius {r_rg} r_g is inside the disc's inner edge ({r_isco_rg} r_g)")
    m = unyt_array(_values(mass, "Msun"), "Msun")
    mdot = unyt_array(_values(accretion_rate, "Msun/yr"), "Msun/yr")
    r = r_rg * G * m / c**2
    flux_term = 3 * G * m * mdot / (8 * np.pi * sigma_sb * r**3)
    t4 = flux_term * (1.0 - np.sqrt(r_isco_rg / r_rg))
    return (t4.to("K**4")) ** 0.25


def disc_max_temperature(mass, accretion_rate, r_isco=R_ISCO_RG):
    """The disc's peak temperature T_max [K], at (49/36) r_isco (the default).

    mass [Msun], accretion_rate [Msun/yr], r_isco in GM/c^2. Zero accretion gives 0 K.
    """
    return _thin_disc_temperature(mass, accretion_rate, T_MAX_RADIUS * r_isco, r_isco)


def disc_temperature_at(mass, accretion_rate, radius, unit="r_isco", r_isco=R_ISCO_RG):
    """Thin-disc temperature [K] at a chosen radius (an option).

    radius is a number in `unit`: "r_isco" (multiples of r_isco), "r_g" (GM/c^2) or "R_s"
    (Schwarzschild radii, 2GM/c^2). For example, Inayoshi & Maiolino's "10 r_g" means 10 R_s,
    about 3.3 r_isco, which is 0.17 dex cooler than T_max. It must not be inside r_isco.
    mass [Msun], accretion_rate [Msun/yr], r_isco in GM/c^2.
    """
    if unit == "r_isco":
        r_rg = radius * r_isco
    elif unit in _RADIUS_UNITS_RG:
        r_rg = radius * _RADIUS_UNITS_RG[unit]
    else:
        raise ValueError(f"unit must be 'r_isco', 'r_g' or 'R_s', not {unit!r}")
    return _thin_disc_temperature(mass, accretion_rate, r_rg, r_isco)


# Bits of the AGN grids' smock_flags dataset (written by smock-recipes/agn_grid)
# the old pipeline's CLOUDY setup, which fixed L and r rather than U
OLD_PIPELINE_LUMINOSITY = 10**43.5         # erg/s, total (`luminosity 43.5 total`)
OLD_PIPELINE_RADIUS = 10**16.7             # cm (`radius 16.7`)
OLD_PIPELINE_HYDROGEN_DENSITY = 10**9.5    # cm^-3 (`hden 9.5`)


def ionising_photons_per_lbol(incident_grid_path, temperature):
    """log10 Q(H) per unit L_bol [photons/s per erg/s] of the AGN's incident SED at each disc
    temperature [K], linear in log T between the incident grid's temperatures.

    Read from the file itself: synthesizer rescales this grid to per Lsun when it loads it.
    """
    with h5py.File(incident_grid_path, "r") as f:
        log_t = np.log10(f["axes/big_bump_temperature"][:])
        log_q = f["log10_specific_ionising_luminosity/HI"][:]
    return np.interp(np.log10(_values(temperature, "K")), log_t, log_q)


def old_pipeline_ionisation_parameter(log10_q_per_lbol, luminosity=OLD_PIPELINE_LUMINOSITY,
                                      radius=OLD_PIPELINE_RADIUS, hydrogen_density=OLD_PIPELINE_HYDROGEN_DENSITY):
    """Ionisation parameter U = Q(H) / (4 pi r^2 n_H c) of gas at radius [cm] with density
    hydrogen_density [cm^-3], lit by a source of luminosity [erg/s] that emits
    10^log10_q_per_lbol ionising photons per erg (ionising_photons_per_lbol). The defaults
    are the old pipeline's CLOUDY setup, which this reproduces (testing only).
    """
    q = 10.0 ** np.asarray(log10_q_per_lbol, dtype=np.float64) * luminosity
    return q / (4.0 * np.pi * radius**2 * hydrogen_density * c.to("cm/s").value)


GRID_FLAGS = {"rerun": 1, "nudged": 2, "thick_es": 4, "hole": 8, "masers_off": 16, "limit_cycle": 32,
              "filled": 64}
# the flags worth a warning: the spectrum there is an estimate (cycle half-amplitude or fill error)
WARN_FLAGS = ("hole", "limit_cycle", "filled")


def _cic_corners(x, v):
    """Indices of the grid points along one axis that CIC interpolation at v uses.

    Mirrors synthesizer's weights (extensions/weights.h): v is clamped to the axis, and a
    point with zero weight is not used. x is the axis (ascending), already in log if the
    axis is log_on_read.
    """
    if len(x) == 1:
        return [0]
    v = np.clip(v, x[0], x[-1])
    j = min(np.searchsorted(x, v, "right") - 1, len(x) - 2)
    t = (v - x[j]) / (x[j + 1] - x[j])
    return [k for k, w in ((j, 1 - t), (j + 1, t)) if w > 0]


def _axis_names(axis):
    """The names a BH attribute may have for a grid axis: as stored, or singular."""
    names = {axis}
    if axis.endswith("ies"):
        names.add(axis[:-3] + "y")          # hydrogen_densities -> hydrogen_density
    elif axis.endswith("s"):
        names.add(axis[:-1])                # turbulences -> turbulence
    return names


def grid_check(grid_path, **values):
    """Per BH: the grid axes it lies outside, and the flags of the grid points it uses.

    grid_path: the AGN grid's HDF5 file. values: one array per grid axis, keyed by the
    axis's name in the file or its singular (as synthesizer's BlackHoles attributes are,
    e.g. hydrogen_density for hydrogen_densities). Plain values are in the axis's units
    (K, cm^-3; column_densities as log10 N_H); unyt values are converted. Extra keys are
    ignored. Returns a list, one dict per BH, with "outside" (axis names) and "flags"
    (names from WARN_FLAGS).
    """
    with h5py.File(grid_path, "r") as f:
        axes = list(f.attrs["axes"])
        grid_axes = []
        for a in axes:
            key = next((k for k in values if k in _axis_names(a)), None)
            if key is None:
                raise ValueError(f"no value for the grid axis {a!r} (axes: {axes})")
            v = values[key]
            unit = str(f[f"axes/{a}"].attrs.get("Units", "dimensionless"))
            if hasattr(v, "units") and unit != "dimensionless":
                v = v.to(unit)
            v = np.asarray(getattr(v, "value", v), dtype=np.float64)
            x = f[f"axes/{a}"][:]
            log = bool(f[f"axes/{a}"].attrs.get("log_on_read", False))
            grid_axes.append((a, np.log10(x) if log else x, log, v))
        flags = f["smock_flags"][:]
    n = max(np.atleast_1d(v).size for *_, v in grid_axes)
    report = []
    for i in range(n):
        outside, corners = [], [()]
        for a, x, log, vals in grid_axes:
            vals = np.atleast_1d(vals)
            v = float(vals[i] if vals.size > 1 else vals[0])
            v = np.log10(v) if log else v
            if v < x[0] - 1e-9 or v > x[-1] + 1e-9:
                outside.append(a)
            corners = [c + (k,) for c in corners for k in _cic_corners(x, v)]
        touched = np.bitwise_or.reduce([int(flags[c]) for c in corners])
        report.append({"outside": outside,
                       "flags": [name for name in WARN_FLAGS if touched & GRID_FLAGS[name]]})
    return report
