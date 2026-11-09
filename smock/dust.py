"""Dust helpers: the ISM dust opacity and dust-to-metal ratio, and a windowed power-law curve.

**The ISM opacity** (walkthrough steps 3-4, DUST_MODEL). tau_V = kappa x the dust column in
front of each star, which synthesizer adds up through the gas's SPH kernels. Two models:

- "code_units" (default): kappa per unit *metal* mass (dust-to-metal ratio 1), fixed in
  MP-Gadget's comoving units, (ckpc/h)^2 per 1e10 Msun/h, as the old pipeline did.
  `kappa_code_for_redshift` picks the value from the snapshot's redshift: 10^2.9 up to z = 2
  (tuned at z = 0 for arXiv:2605.13843 and used there to z = 1.5), 10^4.1 from z = 4 (the old
  pipeline's, tuned at z ~ 4), with log10 kappa linear in z between. Neither value fits the
  bright end of the UV luminosity function at z = 2-3 well (10^2.9 gives 3-4x too many bright
  galaxies there, 10^4.1 2-3x too few), so the join is only a bridge between the two.
- "physical_dtm": kappa per unit *dust* mass in physical units (`KAPPA_DUST`, pc^2/Msun), times
  each galaxy's dust-to-metal ratio from Vijayan et al. (2019, MNRAS 489, 4072; arXiv:1904.02196),
  `dtm_vijayan19`, which grows with the galaxy's metallicity and age. One value works at every
  redshift: in our check on ASTRID (stars only, light within 10 pkpc, slope -1), 0.27 matched the
  observed UV luminosity functions at z = 2.5-4 and stays close at z = 2-8, where the dust
  matters; the rest-optical ones at z = 0.5-1.5 come out within the spread of the surveys. The
  default, 0.3, sits between that and 0.36, which reproduces 10^2.9 at z = 0 (old metal-rich
  galaxies saturate at DTM = 0.329, and 0.36 x 0.329 = 0.117 pc^2/Msun of metals). With a
  shallower dust curve (slope -0.7, Charlot & Fall 2000) the same UV needs about 0.39.

**A power-law dust curve that acts only inside a wavelength window.**

synthesizer 1.3.1 applies every dust curve at every wavelength of a spectrum, and its dust
transformer refuses a wavelength mask ("currently not supported"). A power law with slope
-1 then gives tau = 275 tau_V at 2 nm, so dust absorbs an AGN's X-rays even at tau_V ~ 0.1,
and the energy balance turns them into dust emission: about half the AGN's absorbed energy
in the test galaxy 303604.
Real dust absorbs X-rays far more weakly (they mostly pass, or are absorbed by gas).

WindowedPowerLaw is synthesizer's PowerLaw with tau = 0 outside [lam_min, lam_max]. The
default window, 2 nm - 100 um, is the one the author's old pipeline used for its AGN. The
walkthrough uses it for the AGN only: the stars emit almost nothing outside it.
"""
import numpy as np
from unyt import Myr, angstrom

from synthesizer.emission_models.transformers import PowerLaw
from synthesizer.units import accepts

LAM_MIN = 20.0 * angstrom          # 2 nm (0.6 keV)
LAM_MAX = 1e6 * angstrom           # 100 um


class WindowedPowerLaw(PowerLaw):
    """tau(lam) / tau_V = (lam / 5500 A)^slope inside [lam_min, lam_max], and 0 outside."""

    def __init__(self, slope=-1.0, lam_min=LAM_MIN, lam_max=LAM_MAX):
        PowerLaw.__init__(self, slope=slope)
        self.lam_min = lam_min.to("angstrom").value
        self.lam_max = lam_max.to("angstrom").value
        self.description = f"power law dust curve, only between {self.lam_min:g} and {self.lam_max:g} A"

    def __repr__(self):
        return f"WindowedPowerLaw(slope={self.slope}, lam_min={self.lam_min:g} A, lam_max={self.lam_max:g} A)"

    @accepts(lam=angstrom)
    def get_tau_at_lam(self, lam):
        """V-band-normalised optical depth at lam: the power law in the window, else 0."""
        tau = PowerLaw.get_tau_at_lam(self, lam)
        x = np.asarray(lam.ndview)
        inside = (x >= self.lam_min) & (x <= self.lam_max)
        return np.where(inside, tau, 0.0).astype(np.asarray(tau).dtype, copy=False)


# ---- the ISM dust opacity -----------------------------------------------------------

KAPPA_CODE_LOW_Z = 10**2.9      # (ckpc/h)^2 / (1e10 Msun/h) of metals; tuned at z = 0 (arXiv:2605.13843)
KAPPA_CODE_HIGH_Z = 10**4.1     # the same units; the old pipeline's, tuned at z ~ 4
KAPPA_CODE_JOIN = (2.0, 4.0)    # low-z value up to the first redshift, high-z value from the second
KAPPA_DUST = 0.3                # pc^2 / Msun of dust, for "physical_dtm" (UV LFs at z = 2-4; ~0.39 with slope -0.7)

# Vijayan et al. (2019) Eq. 15 and their canonical grain-growth timescale (Section 2)
VIJAYAN19 = dict(D0=0.008, D1=0.329, alpha=0.017, beta=-1.337, gamma=2.122, tau_acc0_yr=5e4)


def kappa_code_for_redshift(z, low=KAPPA_CODE_LOW_Z, high=KAPPA_CODE_HIGH_Z, join=KAPPA_CODE_JOIN):
    """ISM opacity in code units [(ckpc/h)^2 / (1e10 Msun/h) of metals] for a snapshot at redshift z.

    `low` up to z = join[0], `high` from z = join[1], and log10 kappa linear in z between.
    """
    z0, z1 = join
    f = np.clip((float(z) - z0) / (z1 - z0), 0.0, 1.0)
    return float(10 ** ((1 - f) * np.log10(low) + f * np.log10(high)))


def dtm_vijayan19(z_metal, age_yr, p=VIJAYAN19):
    """Dust-to-metal mass ratio from Vijayan et al. (2019) Eq. 15.

    DTM = D0 + (D1 - D0) [1 - exp(-alpha Z^beta (Age / tau)^gamma)], tau = tau_acc,0 / (D0 Z):
    D0 is the supernova-made dust a young galaxy starts with, D1 the ratio once grain growth has
    saturated. z_metal is the ISM metallicity (mass fraction, not solar units) and age_yr the
    galaxy's mass-weighted stellar age [yr]; scalars or arrays.
    """
    z = np.maximum(np.asarray(z_metal, dtype=np.float64), 0.0)
    age = np.asarray(age_yr, dtype=np.float64)
    # alpha Z^beta (Age D0 Z / tau_acc,0)^gamma, with the two powers of Z combined (beta + gamma > 0),
    # so Z = 0 gives D0 rather than dividing by zero
    x = p["alpha"] * z ** (p["beta"] + p["gamma"]) * (age * p["D0"] / p["tau_acc0_yr"]) ** p["gamma"]
    return p["D0"] + (p["D1"] - p["D0"]) * (1.0 - np.exp(-x))


def dtm_inputs(gas, stars):
    """(ISM metallicity, mass-weighted stellar age [yr]) of a galaxy, as Vijayan et al. (2019) use them.

    gas and stars are the dictionaries from smock.data.load_galaxy. The metallicity is the
    mass-weighted Z of the star-forming gas (all the gas if none is forming stars); the age is
    weighted by the stars' birth masses. NaN without gas or stars.
    """
    m, zg = np.asarray(gas["masses"].to("Msun").value), np.asarray(gas["metallicities"])
    sf = np.asarray(gas["star_formation_rates"].to("Msun/yr").value) > 0
    use = sf if sf.any() else np.ones_like(sf)
    z = float(np.average(zg[use], weights=m[use])) if use.any() else np.nan
    mb = np.asarray(stars["initial_masses"].to("Msun").value)
    age = (float(np.average(stars["ages"].to(Myr).value, weights=mb)) * 1e6 if mb.size else np.nan)
    return z, age
