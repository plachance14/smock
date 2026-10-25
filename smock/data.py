"""Read galaxies from MP-Gadget PIG snapshots (ASTRID, AMBRA) in physical units.

Three entry points, each with whole-FoF-group (default) or subhalo-only particles::

    snap = open_snapshot(pig_dir, 47)
    load_galaxy(snap, sidx=303604)                      # its whole FoF group; centred on SubhaloPos
    load_galaxy(snap, sidx=303604, particles="subhalo") # that subhalo's particles only, same centre
    load_galaxy(snap, gidx=9732)                        # FoF group; centred on MassCenterPosition
    load_galaxy(snap, bhid=220344825797)                # BH; centred on the BH
    load_galaxy(snap, gidx=9732, centre="stellar_shrinking_sphere")   # or recentre on the stars
    crop(gal, 1.95, "arcsec")                           # keep a square prism, half-width 1.95", along z

Particle ranges come from SubhaloOffsetType/SubhaloLenType and FoF OffsetByType/
LengthByType. A subhalo's or BH's FoF group is the group whose particle range contains
it. Nothing is read on import.

Units: MP-Gadget code units in, physical units out (checked on ASTRID 047):

    Position, SmoothingLength   ckpc/h                      -> kpc (physical), relative to the centre
    Mass, BlackholeMass         1e10 Msun/h                 -> Msun
    BlackholeAccretionRate      1e10 Msun/h per (kpc/h)/(km/s) = 977.8 Myr/h  -> Msun/yr
    StarFormationRate (gas)     Msun/yr                     -> Msun/yr (already physical)
    StarFormationTime           scale factor at formation   -> age at the snapshot, Myr
    Metallicity                 metal mass fraction         -> unchanged
"""
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path

import bigfile
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.spatial import cKDTree
from unyt import unyt_array, unyt_quantity

# ---- unit conversions (each has a known-answer test in tests/test_data.py) ----------

MASS_UNIT_MSUN_H = 1e10            # code mass unit, in Msun/h
# MP-Gadget's internal units: length 1 kpc/h, mass 1e10 Msun/h, velocity 1 km/s; the time
# unit is length/velocity = 977.8 Myr/h, and rates are mass per that time unit. The old
# pipeline rounded it to 980 Myr/h.
CODE_TIME_UNIT_YR_H = float(unyt_quantity(1.0, "kpc*s/km").to("yr").value)
MDOT_TO_MSUN_PER_YR = MASS_UNIT_MSUN_H / CODE_TIME_UNIT_YR_H     # h cancels
MIN_STAR_AGE_MYR = 0.001           # old pipeline: stars with age <= 0 are set to 1 kyr
STAR_SMOOTHING_K = 60              # star smoothing length: distance to the 60th nearest star (incl. itself);
                                   # the old pipeline used 32


def ckpch_to_kpc(x, a, h):
    """Comoving kpc/h -> physical kpc."""
    return np.asarray(x) * a / h


def code_mass_to_msun(m, h):
    """Code mass (1e10 Msun/h) -> Msun."""
    return np.asarray(m) * MASS_UNIT_MSUN_H / h


def code_mdot_to_msun_per_yr(mdot):
    """Code accretion rate -> Msun/yr (h cancels)."""
    return np.asarray(mdot) * MDOT_TO_MSUN_PER_YR


def stellar_ages_myr(a_form, a_snap, cosmo):
    """Age at the snapshot [Myr] of stars formed at scale factors a_form; ages <= 0 -> 1 kyr."""
    z_form = 1.0 / np.asarray(a_form, dtype=np.float64) - 1.0
    ages = (cosmo.age(1.0 / a_snap - 1.0) - cosmo.age(z_form)).to("Myr").value
    return np.where(ages <= 0.0, MIN_STAR_AGE_MYR, ages)


def kappa_code_to_physical(kappa_code, a, h):
    """Dust opacity in the original calibration units -> pc^2/Msun.

    The calibration (sunset.py, used by the old pipeline) is per metal mass in MP-Gadget units,
    (ckpc/h)^2 / (1e10 Msun/h). synthesizer's LOS optical depth takes pc^2/Msun with
    physical columns. The optical depth is the same either way; the a^2 makes visible
    the evolution that the comoving calibration hid.
    """
    return kappa_code * a**2 / (1e4 * h)


def wrap_relative(pos, centre, box):
    """Positions relative to centre with periodic minimum-image wrapping (input units).

    `box` is always the whole simulation volume (header BoxSize), never a crop: only the
    simulation is periodic.
    """
    return (np.asarray(pos, dtype=np.float64) - centre + box / 2) % box - box / 2


# ---- snapshots and lookups ---------------------------------------------------------

@dataclass
class Snapshot:
    """An open PIG BigFile and the header values the unit conversions need."""
    path: Path
    file: bigfile.File = field(repr=False)
    a: float
    h: float
    box_ckpch: float
    cosmo: FlatLambdaCDM

    @property
    def redshift(self):
        return 1.0 / self.a - 1.0

    @cached_property
    def bh_ids(self):
        """5/ID for every BH (read once, on first BH-ID lookup)."""
        return np.asarray(self.file["5/ID"][:])


def open_snapshot(pig_dir, snap):
    """Open <pig_dir>/PIG_<snap>_subfind; cosmology and box come from its header."""
    path = Path(pig_dir) / f"PIG_{int(snap):03d}_subfind"
    f = bigfile.File(str(path))
    hdr = f["Header"].attrs
    h = float(hdr["HubbleParam"][0])
    # As the old pipeline: flat LambdaCDM from H0 and Omega0 only (no radiation term).
    cosmo = FlatLambdaCDM(H0=100.0 * h, Om0=float(hdr["Omega0"][0]))
    return Snapshot(path=path, file=f, a=float(hdr["Time"][0]), h=h,
                    box_ckpch=float(hdr["BoxSize"][0]), cosmo=cosmo)


def _row(snap, block, i):
    return np.asarray(snap.file[block][i:i + 1])[0]


def subhalo_row(snap, sidx):
    """Row in this snapshot of the subhalo with full-PIG index sidx.

    The same index for a full PIG. A cutout (the test cutout, made by
    smock-recipes/test_cutout) renumbers its subhalos and keeps the full-PIG index
    in SubGroups/OriginalIndex.
    """
    if "SubGroups/OriginalIndex" not in snap.file.blocks:
        return int(sidx)
    rows = np.flatnonzero(np.asarray(snap.file["SubGroups/OriginalIndex"][:]) == sidx)
    if len(rows) == 0:
        raise KeyError(f"subhalo {sidx} is not in {snap.path}")
    return int(rows[0])


def subhalo_range(snap, sidx):
    """(start, length) of subhalo sidx's particles, per type (arrays of 6)."""
    return (_row(snap, "SubGroups/SubhaloOffsetType", sidx).astype(np.int64),
            _row(snap, "SubGroups/SubhaloLenType", sidx).astype(np.int64))


def group_range(snap, gidx):
    """(start, length) of FoF group gidx's particles, per type (arrays of 6)."""
    return (_row(snap, "FOFGroups/OffsetByType", gidx).astype(np.int64),
            _row(snap, "FOFGroups/LengthByType", gidx).astype(np.int64))


def _last_row_starting_at_or_before(snap, offset_block, ptype, index):
    """Binary search, with point reads, for the last row whose type-`ptype` offset is
    <= index. Offsets never decrease along the rows (checked on ASTRID 047 and AMBRA 053
    for all types), so this is the only row whose range can contain particle `index`."""
    lo, hi = 0, snap.file[offset_block].size      # answer is in [lo, hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _row(snap, offset_block, mid)[ptype] <= index:
            lo = mid
        else:
            hi = mid
    return lo


def _contains(start, length, ptype, index):
    return start[ptype] <= index < start[ptype] + length[ptype]


def group_of_particle(snap, ptype, index):
    """FoF group holding particle `index` of type ptype. PIGs hold group particles only,
    so every particle is in a group."""
    g = _last_row_starting_at_or_before(snap, "FOFGroups/OffsetByType", ptype, index)
    if not _contains(*group_range(snap, g), ptype, index):
        raise LookupError(f"type-{ptype} particle {index} is not inside group {g}'s range")
    return g


def subhalo_of_particle(snap, ptype, index):
    """Subhalo holding particle `index` of type ptype; LookupError for FoF fuzz."""
    s = _last_row_starting_at_or_before(snap, "SubGroups/SubhaloOffsetType", ptype, index)
    if not _contains(*subhalo_range(snap, s), ptype, index):
        raise LookupError(f"type-{ptype} particle {index} is in no subhalo (FoF fuzz)")
    return s


def group_of_subhalo(snap, sidx):
    """FoF group whose particle ranges contain all of subhalo sidx's particles."""
    start, length = subhalo_range(snap, sidx)
    ptype = int(np.argmax(length))                 # any non-empty type will do
    g = group_of_particle(snap, ptype, start[ptype])
    g_start, g_len = group_range(snap, g)
    inside = (length == 0) | ((start >= g_start) & (start + length <= g_start + g_len))
    if not inside.all():
        raise LookupError(f"subhalo {sidx} is not wholly inside group {g}")
    return g


def bh_index(snap, bhid):
    """Index of BH `bhid` in the 5/ blocks."""
    match = np.nonzero(snap.bh_ids == np.uint64(bhid))[0]
    if match.size == 0:
        raise LookupError(f"BH {bhid} is not in {snap.path.name}")
    return int(match[0])


# ---- particles ---------------------------------------------------------------------

def _read(snap, ptype, fields, start, length):
    """Raw blocks ptype/<field> for particles [start, start + length)."""
    s0, s1 = int(start[ptype]), int(start[ptype] + length[ptype])
    return {k: np.asarray(snap.file[f"{ptype}/{k}"][s0:s1]) for k in fields}


def star_smoothing_lengths(coords, k=STAR_SMOOTHING_K):
    """Distance to the k-th nearest star, counting the star itself (the old pipeline's
    smooth()). Same units as coords. Fewer than k stars: the farthest one is used."""
    n = len(coords)
    if n < 2:
        return np.zeros(n)
    d, _ = cKDTree(coords).query(coords, k=min(k, n))
    return d[:, -1].copy()


def shrinking_sphere_centre(pos, mass, shrink=0.975, min_particles=100):
    """Shrinking-sphere centre (Power et al. 2003): start from the centre of mass of all
    particles, then repeatedly shrink the sphere by `shrink` about the centre of mass of
    the particles inside it, until fewer than min_particles remain. pos must already be
    unwrapped (no periodic jumps). Same units as pos."""
    centre = np.average(pos, axis=0, weights=mass)
    radius = np.sqrt(((pos - centre) ** 2).sum(axis=1)).max()
    inside = np.ones(len(pos), dtype=bool)
    while inside.sum() >= min_particles:
        radius *= shrink
        inside = ((pos - centre) ** 2).sum(axis=1) < radius**2
        if inside.sum() < min_particles:
            break
        centre = np.average(pos[inside], axis=0, weights=mass[inside])
    return centre


@dataclass
class Galaxy:
    """One galaxy's particles in physical units, with coordinates relative to the centre.

    stars: coordinates [kpc], initial_masses and current_masses [Msun], ages [Myr],
           metallicities, smoothing_lengths [kpc]
    gas:   coordinates [kpc], masses [Msun], metallicities, smoothing_lengths [kpc],
           star_formation_rates [Msun/yr]
    bhs:   ids, coordinates [kpc], masses [Msun] (BlackholeMass), accretion_rates [Msun/yr]
    """
    redshift: float
    h: float                   # from the snapshot header
    cosmo: FlatLambdaCDM = field(repr=False)
    entry: str                 # "subhalo", "group" or "bh"
    particles: str             # "subhalo" or "group"
    gidx: int                  # FoF group (always known)
    sidx: int | None           # subhalo, if the entry or the particle set has one
    bhid: int | None
    centre_ckpch: np.ndarray   # in simulation coordinates (comoving kpc/h)
    centre_method: str         # "catalogue", "stellar_shrinking_sphere" or "explicit"
    stars: dict
    gas: dict
    bhs: dict
    crop: dict | None = None   # set by crop(): what was kept


def _stars(snap, start, length, centre, smoothing_k=STAR_SMOOTHING_K):
    raw = _read(snap, 4, ["Position", "Mass", "TotalMassReturned", "Metallicity", "StarFormationTime"],
                start, length)
    rel = wrap_relative(raw["Position"], centre, snap.box_ckpch)
    coords = ckpch_to_kpc(rel, snap.a, snap.h)
    current = code_mass_to_msun(raw["Mass"].astype(np.float64), snap.h)
    returned = code_mass_to_msun(raw["TotalMassReturned"].astype(np.float64), snap.h)
    return {
        "coordinates": unyt_array(coords, "kpc"),
        # Birth mass = Mass + TotalMassReturned: MP-Gadget (metal_return.c) moves returned
        # mass from Mass into TotalMassReturned, and in ASTRID 047 the returned fraction is
        # 0 below 5 Myr and rises with age. The walkthrough gives synthesizer the birth mass by
        # default (STAR_MASS); the old pipeline used Mass.
        "initial_masses": unyt_array(current + returned, "Msun"),
        "current_masses": unyt_array(current, "Msun"),
        "ages": unyt_array(stellar_ages_myr(raw["StarFormationTime"], snap.a, snap.cosmo), "Myr"),
        "metallicities": raw["Metallicity"].astype(np.float64),
        # From wrapped positions, so groups across the box edge get the right neighbours.
        "smoothing_lengths": unyt_array(star_smoothing_lengths(coords, smoothing_k), "kpc"),
    }


def _gas(snap, start, length, centre):
    raw = _read(snap, 0, ["Position", "Mass", "Metallicity", "SmoothingLength", "StarFormationRate"],
                start, length)
    rel = wrap_relative(raw["Position"], centre, snap.box_ckpch)
    return {
        "coordinates": unyt_array(ckpch_to_kpc(rel, snap.a, snap.h), "kpc"),
        "masses": unyt_array(code_mass_to_msun(raw["Mass"].astype(np.float64), snap.h), "Msun"),
        "metallicities": raw["Metallicity"].astype(np.float64),
        "smoothing_lengths": unyt_array(ckpch_to_kpc(raw["SmoothingLength"].astype(np.float64),
                                                     snap.a, snap.h), "kpc"),
        "star_formation_rates": unyt_array(raw["StarFormationRate"].astype(np.float64), "Msun/yr"),
    }


def _bhs(snap, start, length, centre):
    raw = _read(snap, 5, ["ID", "Position", "BlackholeMass", "BlackholeAccretionRate"], start, length)
    rel = wrap_relative(raw["Position"], centre, snap.box_ckpch)
    return {
        "ids": raw["ID"],
        "coordinates": unyt_array(ckpch_to_kpc(rel, snap.a, snap.h), "kpc"),
        "masses": unyt_array(code_mass_to_msun(raw["BlackholeMass"].astype(np.float64), snap.h), "Msun"),
        "accretion_rates": unyt_array(code_mdot_to_msun_per_yr(
            raw["BlackholeAccretionRate"].astype(np.float64)), "Msun/yr"),
    }


def stellar_shrinking_sphere_centre(snap, start, length):
    """Shrinking-sphere centre of the stars in a particle range, in simulation
    coordinates (comoving kpc/h)."""
    raw = _read(snap, 4, ["Position", "Mass"], start, length)
    if len(raw["Mass"]) == 0:
        raise ValueError("no stars to centre on; use centre='catalogue' or explicit coordinates")
    anchor = raw["Position"][0].astype(np.float64)        # unwrap about any member
    rel = wrap_relative(raw["Position"], anchor, snap.box_ckpch)
    return (anchor + shrinking_sphere_centre(rel, raw["Mass"].astype(np.float64))) % snap.box_ckpch


def load_galaxy(snap, *, sidx=None, gidx=None, bhid=None, particles="group", centre="catalogue",
                star_smoothing_k=STAR_SMOOTHING_K):
    """Particles of one galaxy, chosen by exactly one of sidx, gidx or bhid.

    particles: "group" (default: the whole FoF group, fuzz included, so neighbours in the
        field are imaged too and a satellite isn't shown as if it were alone) or "subhalo"
        (that subhalo's particles only). A gidx entry always uses the group.
    centre: what the coordinates are relative to.
        "catalogue" (default): the entry's own position. That is SubhaloPos for sidx,
            FOFGroups/MassCenterPosition (centre of mass of all particle types, dark
            matter included) for gidx, and the BH's position for bhid.
        "stellar_shrinking_sphere": the shrinking-sphere centre of the loaded stars.
        or explicit simulation coordinates (comoving kpc/h, length 3).
        Image-based recentring (e.g. for compactness) comes later, in the analysis step.
    star_smoothing_k: star smoothing length = distance to the k-th nearest loaded star,
        itself included (default 60; the old pipeline used 32).
    """
    if sum(x is not None for x in (sidx, gidx, bhid)) != 1:
        raise ValueError("give exactly one of sidx, gidx, bhid")
    if particles not in ("subhalo", "group"):
        raise ValueError("particles must be 'subhalo' or 'group'")

    if sidx is not None:
        entry = "subhalo"
        gidx = group_of_subhalo(snap, sidx)
        catalogue_centre = _row(snap, "SubGroups/SubhaloPos", sidx)
    elif gidx is not None:
        entry, particles = "group", "group"
        catalogue_centre = _row(snap, "FOFGroups/MassCenterPosition", gidx)
    else:
        entry = "bh"
        i = bh_index(snap, bhid)
        gidx = group_of_particle(snap, 5, i)
        if particles == "subhalo":
            sidx = subhalo_of_particle(snap, 5, i)
        catalogue_centre = _row(snap, "5/Position", i)

    start, length = subhalo_range(snap, sidx) if particles == "subhalo" else group_range(snap, gidx)
    if isinstance(centre, str):
        if centre == "catalogue":
            centre_ckpch = np.asarray(catalogue_centre, dtype=np.float64)
        elif centre == "stellar_shrinking_sphere":
            centre_ckpch = stellar_shrinking_sphere_centre(snap, start, length)
        else:
            raise ValueError("centre must be 'catalogue', 'stellar_shrinking_sphere' or coordinates")
        method = centre
    else:
        centre_ckpch, method = np.asarray(centre, dtype=np.float64).reshape(3), "explicit"

    return Galaxy(redshift=snap.redshift, h=snap.h, cosmo=snap.cosmo,
                  entry=entry, particles=particles, gidx=int(gidx),
                  sidx=None if sidx is None else int(sidx), bhid=None if bhid is None else int(bhid),
                  centre_ckpch=centre_ckpch, centre_method=method,
                  stars=_stars(snap, start, length, centre_ckpch, star_smoothing_k),
                  gas=_gas(snap, start, length, centre_ckpch),
                  bhs=_bhs(snap, start, length, centre_ckpch))


# ---- cropping (line of sight along z) ------------------------------------------------

CROP_UNITS = ("kpc", "ckpc/h", "arcsec", "r_half")
CROP_SHAPES = ("cylinder", "prism", "cube")
OLD_GAS_MARGIN_CKPCH = 5.0   # the old pipeline kept gas 5 ckpc/h beyond the stellar crop


def projected_half_mass_radius(coords, masses):
    """Radius in the x-y plane, about the origin, that encloses half the mass [coords units].

    Returns the first particle radius at which the enclosed mass reaches half. (The old
    code's `- 1` index returned the farthest particle when the innermost one held half
    the mass.)
    """
    r = np.hypot(coords[:, 0], coords[:, 1])
    order = np.argsort(r)
    enclosed = np.cumsum(masses[order])
    return r[order][np.searchsorted(enclosed, 0.5 * enclosed[-1])]


def crop_length_kpc(galaxy, size, unit):
    """A crop size given in `unit` -> physical kpc, at the galaxy's redshift.

    unit: "kpc" (physical), "ckpc/h", "arcsec", or "r_half": multiples of the projected
        half-mass radius of the loaded stars, from their current masses (as the old
        pipeline's "stellar_r_half").
    """
    a = 1.0 / (1.0 + galaxy.redshift)
    if unit == "kpc":
        return float(size)
    if unit == "ckpc/h":
        return float(ckpch_to_kpc(size, a, galaxy.h))
    if unit == "arcsec":
        return float(size / galaxy.cosmo.arcsec_per_kpc_proper(galaxy.redshift).value)
    if unit == "r_half":
        s = galaxy.stars
        return float(size * projected_half_mass_radius(s["coordinates"].to("kpc").value,
                                                       s["current_masses"].to("Msun").value))
    raise ValueError(f"unit must be one of {CROP_UNITS}")


def _inside(coords, half_kpc, shape):
    c = coords.to("kpc").value
    if shape == "cylinder":
        return np.hypot(c[:, 0], c[:, 1]) < half_kpc
    if shape == "prism":
        return (np.abs(c[:, :2]) < half_kpc).all(axis=1)
    return (np.abs(c) < half_kpc).all(axis=1)          # cube


def crop(galaxy, size, unit="arcsec", shape="prism", gas_margin_ckpch=OLD_GAS_MARGIN_CKPCH):
    """A copy of `galaxy` keeping only the particles near its centre; the line of sight is z.

    shape: "prism" (default): |x| < size and |y| < size, any z. It covers a square image,
               corners included;
           "cylinder" (the old pipeline): x^2 + y^2 < size^2, any z. A cylinder as wide as
               the image leaves its corners empty;
           "cube":  |x|, |y| and |z| < size.
    size, unit: radius or half-width, in one of CROP_UNITS (see crop_length_kpc).
    gas_margin_ckpch: gas is kept out to size + this margin [comoving kpc/h], so gas just
        outside the crop still dims stars near its edge. Stars and BHs are cut at size.
    """
    if shape not in CROP_SHAPES:
        raise ValueError(f"shape must be one of {CROP_SHAPES}")
    size_kpc = crop_length_kpc(galaxy, size, unit)
    margin_kpc = float(ckpch_to_kpc(gas_margin_ckpch, 1.0 / (1.0 + galaxy.redshift), galaxy.h))

    def keep(part, half_kpc):
        mask = _inside(part["coordinates"], half_kpc, shape)
        return {k: v[mask] for k, v in part.items()}

    return replace(galaxy, stars=keep(galaxy.stars, size_kpc), gas=keep(galaxy.gas, size_kpc + margin_kpc),
                   bhs=keep(galaxy.bhs, size_kpc),
                   crop={"size": size, "unit": unit, "shape": shape,
                         "size_kpc": size_kpc, "gas_margin_kpc": margin_kpc})
