"""A power-law dust curve that acts only inside a wavelength window.

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
from unyt import angstrom

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
