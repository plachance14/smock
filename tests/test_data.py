"""Known-answer tests for smock.data: unit conversions, lookups and the three entry points."""
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from smock import config
from smock.data import (Galaxy, ckpch_to_kpc, code_mass_to_msun, code_mdot_to_msun_per_yr, crop,
                        crop_length_kpc, kappa_code_to_physical, load_galaxy, open_snapshot,
                        projected_half_mass_radius, subhalo_row,
                        shrinking_sphere_centre, star_smoothing_lengths, stellar_ages_myr,
                        wrap_relative)
from unyt import unyt_array

from .conftest import A, H

MPC_KM = 3.0856775814913673e19
MYR_S = 3.15576e13                 # Julian Myr, as astropy and unyt use
# code time unit (kpc/h)/(km/s), in yr/h; Mdot unit = 1e10 Msun/h per that
MDOT_UNIT_MSUN_PER_YR = 1e10 / (MPC_KM / 1e3 / (MYR_S / 1e6))    # = 10.2271...


# ---- unit conversions ----------------------------------------------------------------

def test_ckpch_to_kpc():
    assert ckpch_to_kpc(100.0, a=0.25, h=0.5) == pytest.approx(50.0)


def test_code_mass_to_msun():
    assert code_mass_to_msun(1.0, h=0.5) == pytest.approx(2e10)


def test_code_mdot_to_msun_per_yr():
    # 1e10 Msun/h per (kpc/h)/(km/s) = 977.79 Myr/h (the old pipeline used 980)
    assert MDOT_UNIT_MSUN_PER_YR == pytest.approx(10.22712, rel=1e-6)
    assert code_mdot_to_msun_per_yr(1.0) == pytest.approx(MDOT_UNIT_MSUN_PER_YR, rel=1e-9)


def test_stellar_ages_einstein_de_sitter():
    # Om0 = 1 (no radiation): t(a) = 2 / (3 H0) a^1.5 exactly
    cosmo = FlatLambdaCDM(H0=70.0, Om0=1.0)
    hubble_time_myr = MPC_KM / 70.0 / MYR_S
    t = lambda a: 2.0 / 3.0 * hubble_time_myr * a**1.5
    ages = stellar_ages_myr([0.1, 0.2], a_snap=0.25, cosmo=cosmo)
    assert ages == pytest.approx([t(0.25) - t(0.1), t(0.25) - t(0.2)], rel=1e-6)


def test_stellar_ages_clamped_at_formation():
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    assert stellar_ages_myr([0.25], a_snap=0.25, cosmo=cosmo)[0] == pytest.approx(0.001)


def test_kappa_code_to_physical():
    # about 0.052 pc^2/Msun at z = 5 for h = 0.6774
    assert kappa_code_to_physical(10**4.1, a=1 / 6, h=0.6774) == pytest.approx(0.051624, rel=1e-4)
    # the optical depth is unchanged: one metal mass over one area, in either unit system
    a, h = 0.2, 0.7
    m_code, area_code = 3e-4, 2.0                 # 1e10 Msun/h, (ckpc/h)^2
    m_phys = m_code * 1e10 / h                    # Msun
    area_phys = area_code * (1e3 * a / h) ** 2    # pc^2
    assert kappa_code_to_physical(5.0, a, h) * m_phys / area_phys == pytest.approx(5.0 * m_code / area_code)


def test_wrap_relative_minimum_image():
    rel = wrap_relative([[999.0, 0, 0], [1.0, 0, 0]], np.array([1.0, 0, 0]), 1000.0)
    assert rel[:, 0] == pytest.approx([-2.0, 0.0])


def test_star_smoothing_lengths_kth_neighbour():
    # k = 2 counts the star itself, so this is the nearest-neighbour distance
    line = np.array([[0.0, 0, 0], [1, 0, 0], [3, 0, 0], [6, 0, 0]])
    assert star_smoothing_lengths(line, k=2) == pytest.approx([1, 1, 2, 3])


def test_shrinking_sphere_finds_dense_clump():
    rng = np.random.default_rng(1)
    clump = rng.normal([5.0, 5, 5], 0.2, size=(300, 3))
    halo = rng.uniform(-50, 50, size=(300, 3))
    pos = np.vstack([clump, halo])
    centre = shrinking_sphere_centre(pos, np.ones(len(pos)))
    assert np.linalg.norm(centre - 5.0) < 0.1


# ---- entry points on the fake PIG ------------------------------------------------------

@pytest.fixture(scope="module")
def snap(fake_pig_dir):
    return open_snapshot(fake_pig_dir, 7)


def star_mass_codes(gal):
    """Recover the fixture's star labels (mass = label * 1e-5 code units)."""
    return np.rint(gal.stars["current_masses"].to("Msun").value / code_mass_to_msun(1e-5, H)).astype(int)


def test_header(snap):
    assert (snap.a, snap.h, snap.box_ckpch) == (A, H, 1000.0)
    assert snap.redshift == pytest.approx(3.0)


def test_group_particles_by_default(snap):
    assert load_galaxy(snap, sidx=1).particles == "group"
    assert load_galaxy(snap, bhid=102).particles == "group"


def test_subhalo_only(snap):
    gal = load_galaxy(snap, sidx=1, particles="subhalo")
    assert (gal.entry, gal.particles, gal.gidx, gal.sidx) == ("subhalo", "subhalo", 0, 1)
    assert sorted(star_mass_codes(gal)) == [4, 5]
    assert gal.bhs["ids"].tolist() == [102]
    assert len(gal.gas["masses"]) == 1
    # centred on SubhaloPos (110, 100, 100): the stars sit at x = 0 and 1 ckpc/h
    assert gal.stars["coordinates"][:, 0].to("kpc").value == pytest.approx([0.0, A / H])


def test_subhalo_with_group_particles(snap):
    gal = load_galaxy(snap, sidx=1, particles="group")
    assert sorted(star_mass_codes(gal)) == [1, 2, 3, 4, 5, 6]     # fuzz star 6 included
    assert gal.bhs["ids"].tolist() == [101, 102, 103]
    assert gal.centre_ckpch == pytest.approx([110, 100, 100])


def test_group_entry_across_box_edge(snap):
    gal = load_galaxy(snap, gidx=2)
    assert (gal.entry, gal.particles, gal.sidx, gal.centre_method) == ("group", "group", None, "catalogue")
    assert sorted(star_mass_codes(gal)) == [7, 8]
    assert len(gal.bhs["ids"]) == 0
    # centred on MassCenterPosition (0, 500, 500): stars at x = 999 and 1 sit at -1 and +1
    x = gal.stars["coordinates"][:, 0].to("kpc").value
    assert sorted(x) == pytest.approx([-A / H, A / H])


def test_stellar_shrinking_sphere_centre_option(snap):
    # two stars (masses 7:8), fewer than the shrinking sphere's minimum: their centre of mass
    gal = load_galaxy(snap, gidx=2, centre="stellar_shrinking_sphere")
    x = gal.stars["coordinates"][:, 0].to("kpc").value
    assert gal.centre_method == "stellar_shrinking_sphere"
    assert np.ptp(x) == pytest.approx(2 * A / H)                # not split by the box edge
    assert np.average(x, weights=gal.stars["current_masses"].value) == pytest.approx(0.0, abs=1e-6)


def test_explicit_centre(snap):
    gal = load_galaxy(snap, sidx=0, particles="subhalo", centre=[100.0, 100.0, 100.0])
    assert gal.centre_method == "explicit"
    assert gal.bhs["coordinates"].to("kpc").value == pytest.approx(np.zeros((1, 3)))
    with pytest.raises(ValueError):
        load_galaxy(snap, sidx=0, centre="middle")


def test_bh_entry(snap):
    gal = load_galaxy(snap, bhid=102, particles="subhalo")
    assert (gal.entry, gal.particles, gal.gidx, gal.sidx) == ("bh", "subhalo", 0, 1)
    assert gal.bhs["coordinates"].to("kpc").value == pytest.approx(np.zeros((1, 3)))
    gal = load_galaxy(snap, bhid=102, particles="group")
    assert gal.bhs["ids"].tolist() == [101, 102, 103]


def test_fuzz_bh(snap):
    gal = load_galaxy(snap, bhid=103, particles="group")
    assert gal.gidx == 0
    with pytest.raises(LookupError, match="fuzz"):
        load_galaxy(snap, bhid=103, particles="subhalo")


def test_unknown_bh_and_bad_arguments(snap):
    with pytest.raises(LookupError):
        load_galaxy(snap, bhid=999)
    with pytest.raises(ValueError):
        load_galaxy(snap)
    with pytest.raises(ValueError):
        load_galaxy(snap, sidx=0, gidx=0)


def test_physical_units(snap):
    gal = load_galaxy(snap, sidx=0, particles="subhalo")
    s, g, b = gal.stars, gal.gas, gal.bhs
    # star 1: Mass 1e-5 + returned 0.5e-5 code units
    i = int(np.argmin(s["current_masses"]))
    assert s["current_masses"][i].to("Msun").value == pytest.approx(1e-5 * 1e10 / H, rel=1e-6)
    assert s["initial_masses"][i].to("Msun").value == pytest.approx(1.5e-5 * 1e10 / H, rel=1e-6)
    assert s["ages"].to("Myr").value.min() == pytest.approx(0.001)           # star formed at a_snap
    assert g["smoothing_lengths"].to("kpc").value == pytest.approx(2.0 * A / H)
    assert sorted(g["star_formation_rates"].to("Msun/yr").value) == pytest.approx([0.0, 0.5])
    assert b["masses"].to("Msun").value == pytest.approx([1e-4 * 1e10 / H], rel=1e-6)
    assert b["accretion_rates"].to("Msun/yr").value == pytest.approx([1e-3 * MDOT_UNIT_MSUN_PER_YR], rel=1e-6)


# ---- cropping ------------------------------------------------------------------------

def toy_galaxy(star_xyz, gas_xyz, z=3.0, h=0.5):
    """A Galaxy record holding only what crop() reads: coordinates [kpc] and star masses."""
    stars = {"coordinates": unyt_array(np.asarray(star_xyz, float), "kpc"),
             "current_masses": unyt_array(np.ones(len(star_xyz)), "Msun")}
    gas = {"coordinates": unyt_array(np.asarray(gas_xyz, float), "kpc")}
    bhs = {"coordinates": unyt_array(np.zeros((0, 3)), "kpc"), "ids": np.zeros(0, int)}
    return Galaxy(redshift=z, h=h, cosmo=FlatLambdaCDM(H0=100.0 * h, Om0=1.0), entry="subhalo",
                  particles="subhalo", gidx=0, sidx=0, bhid=None, centre_ckpch=np.zeros(3),
                  centre_method="catalogue", stars=stars, gas=gas, bhs=bhs)


def test_projected_half_mass_radius():
    # equal masses at projected r = 1..4 (z ignored): half is reached at the 2nd particle
    xyz = np.array([[1.0, 0, 9], [0, 2.0, -9], [3.0, 0, 0], [0, 4.0, 0]])
    assert projected_half_mass_radius(xyz, np.ones(4)) == pytest.approx(2.0)
    # the innermost particle holds more than half: its radius, not the farthest (the old pipeline's -1 index)
    assert projected_half_mass_radius(xyz[:2], np.array([3.0, 1.0])) == pytest.approx(1.0)


def test_crop_length_units():
    gal = toy_galaxy([[1.0, 0, 0], [0, 2.0, 0], [3.0, 0, 0], [0, 4.0, 0]], [[0.0, 0, 0]])
    assert crop_length_kpc(gal, 2.0, "kpc") == 2.0
    assert crop_length_kpc(gal, 100.0, "ckpc/h") == pytest.approx(100.0 * 0.25 / 0.5)   # a = 1/4
    # Einstein-de Sitter: D_A = (2c/H0) (1 - 1/sqrt(1+z)) / (1+z) = c/(4 H0) at z = 3
    d_a_kpc = 299792.458 / 50.0 / 4.0 * 1e3
    assert crop_length_kpc(gal, 1.0, "arcsec") == pytest.approx(d_a_kpc * np.pi / 180 / 3600, rel=1e-6)
    assert crop_length_kpc(gal, 1.5, "r_half") == pytest.approx(3.0)
    with pytest.raises(ValueError):
        crop_length_kpc(gal, 1.0, "pc")


@pytest.mark.parametrize("shape, kept", [("cylinder", [0, 2]), ("prism", [0, 1, 2]), ("cube", [1])])
def test_crop_shapes(shape, kept):
    stars = [[0.9, 0, 5.0],      # inside the circle, far along the line of sight
             [0.8, 0.8, 0.0],    # inside the square, outside the circle
             [0.5, 0.5, 2.0]]    # near the axis, beyond the cube's z
    gal = toy_galaxy(stars, [[0.0, 0, 0]])
    out = crop(gal, 1.0, "kpc", shape=shape, gas_margin_ckpch=0.0)
    assert out.stars["coordinates"].to("kpc").value.tolist() == [stars[i] for i in kept]
    assert out.crop["shape"] == shape and out.crop["size_kpc"] == 1.0
    assert len(gal.stars["coordinates"]) == 3                     # the input is not modified


def test_crop_gas_margin():
    gal = toy_galaxy([[0.0, 0, 0]], [[1.2, 0, 0], [1.6, 0, 0]])
    # 1 ckpc/h = 0.5 kpc at a = 1/4, h = 0.5: gas kept out to 1.5 kpc
    out = crop(gal, 1.0, "kpc", gas_margin_ckpch=1.0)
    assert out.gas["coordinates"].to("kpc").value[:, 0].tolist() == [1.2]
    assert out.crop["gas_margin_kpc"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        crop(gal, 1.0, "kpc", shape="sphere")


# ---- the real test cutout, when it is present -------------------------------------------

CUTOUT = config.TEST_CUTOUT_PIG_DIR / "PIG_047_subfind"


def test_subhalo_row_is_identity_for_a_full_pig(snap):
    assert subhalo_row(snap, 2) == 2


@pytest.mark.skipif(not CUTOUT.exists(), reason="test cutout not downloaded")
def test_subhalo_row_in_cutout():
    snap = open_snapshot(config.TEST_CUTOUT_PIG_DIR, 47)
    assert subhalo_row(snap, 303604) == 179 and subhalo_row(snap, 4283857) == 192
    with pytest.raises(KeyError):
        subhalo_row(snap, 1)


@pytest.mark.skipif(not CUTOUT.exists(), reason="test cutout not downloaded")
@pytest.mark.parametrize("sidx_row", range(5))
def test_cutout_subhalos_match_catalogue(sidx_row):
    snap = open_snapshot(config.TEST_CUTOUT_PIG_DIR, 47)
    f = snap.file
    # the chosen subhalos are the first rows of their groups in the cutout
    sidx = int(np.asarray(f["FOFGroups/GroupFirstSub"][sidx_row:sidx_row + 1])[0])
    gal = load_galaxy(snap, sidx=sidx, particles="subhalo")
    mtype = np.asarray(f["SubGroups/SubhaloMassType"][sidx:sidx + 1])[0].astype(np.float64)
    assert gal.gidx == sidx_row
    assert gal.stars["current_masses"].sum().to("Msun").value == pytest.approx(
        mtype[4] * 1e10 / snap.h, rel=1e-5)
    assert gal.gas["masses"].sum().to("Msun").value == pytest.approx(mtype[0] * 1e10 / snap.h, rel=1e-5)
    if len(gal.bhs["ids"]):
        main_bh = gal.bhs["ids"][np.argmax(gal.bhs["masses"])]
        by_bh = load_galaxy(snap, bhid=main_bh, particles="subhalo")
        assert (by_bh.gidx, by_bh.sidx) == (gal.gidx, sidx)


@pytest.mark.skipif(not CUTOUT.exists(), reason="test cutout not downloaded")
@pytest.mark.parametrize("column, ptype, particle_block", [
    ("SubhaloBHMdot", 5, "5/BlackholeAccretionRate"),   # the BH L_bol cut in make_subhalo_list.ipynb
    ("SubhaloSFR", 0, "0/StarFormationRate"),           # its example SFR cut
])
def test_catalogue_rates_are_sums_over_particles(column, ptype, particle_block):
    f = open_snapshot(config.TEST_CUTOUT_PIG_DIR, 47).file
    off = f["SubGroups/SubhaloOffsetType"][:][:, ptype]
    n = f["SubGroups/SubhaloLenType"][:][:, ptype]
    rate = f[particle_block][:]
    summed = np.array([rate[o:o + k].sum() for o, k in zip(off, n)])
    assert np.any(n > 1)
    np.testing.assert_allclose(f[f"SubGroups/{column}"][:], summed, rtol=1e-6)
