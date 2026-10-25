"""Synthetic fixtures: a tiny PIG BigFile with the real block names and dtypes."""
import bigfile
import numpy as np
import pytest

A, H, BOX = 0.25, 0.5, 1000.0      # scale factor, h, box [ckpc/h]


def _write(f, block, data):
    f.create_from_array(block, np.asarray(data), Nfile=1)


def make_fake_pig(path):
    """Three FoF groups, written as PIG_<snap>_subfind would be.

    group 0: subhalo 0 (3 stars, 2 gas, BH 101), subhalo 1 (2 stars, 1 gas, BH 102),
             then fuzz (1 star, 1 gas, BH 103), near x = 100
    group 1: 1 gas particle only, no subhalo (zero-length star and BH ranges)
    group 2: subhalo 2 (2 stars, 1 gas, no BH), straddling the box edge at x = 0
    """
    # particles per group, by type (gas, DM, -, -, stars, BHs); DM is not stored
    g_len = np.array([[4, 0, 0, 0, 6, 3], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 2, 0]], dtype="u4")
    g_off = np.zeros((3, 6), dtype="u8")
    g_off[1:] = np.cumsum(g_len, axis=0)[:-1]
    s_len = np.array([[2, 0, 0, 0, 3, 1], [1, 0, 0, 0, 2, 1], [1, 0, 0, 0, 2, 0]], dtype="i4")
    s_off = np.array([[0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 3, 1], [5, 0, 0, 0, 6, 3]], dtype="u8")

    star_pos = np.array([[100, 100, 100], [101, 100, 100], [100, 101, 100],     # subhalo 0
                         [110, 100, 100], [111, 100, 100],                      # subhalo 1
                         [120, 100, 100],                                       # fuzz
                         [999, 500, 500], [1, 500, 500]], dtype="f8")           # group 2 (wraps)
    gas_pos = np.array([[100, 100, 101], [100, 100, 99], [110, 100, 101], [120, 100, 101],
                        [300, 300, 300], [999.5, 500, 500]], dtype="f8")
    bh_pos = np.array([[100, 100, 100], [110, 100, 100], [120, 100, 100]], dtype="f8")

    with bigfile.File(str(path), create=True) as f:
        with f.create("Header") as hd:
            hd.attrs["Time"] = np.array([A])
            hd.attrs["HubbleParam"] = np.array([H])
            hd.attrs["BoxSize"] = np.array([BOX])
            hd.attrs["Omega0"] = np.array([0.3])
        _write(f, "FOFGroups/LengthByType", g_len)
        _write(f, "FOFGroups/OffsetByType", g_off)
        _write(f, "FOFGroups/MassCenterPosition",
               np.array([[105, 100, 100], [300, 300, 300], [0, 500, 500]], dtype="f8"))
        _write(f, "SubGroups/SubhaloLenType", s_len)
        _write(f, "SubGroups/SubhaloOffsetType", s_off)
        _write(f, "SubGroups/SubhaloPos", np.array([[100, 100, 100], [110, 100, 100], [0, 500, 500]], dtype="f4"))
        _write(f, "4/Position", star_pos)
        _write(f, "4/Mass", np.arange(1, 9, dtype="f4") * 1e-5)            # star i has mass i * 1e-5
        _write(f, "4/TotalMassReturned", np.full(8, 0.5e-5, dtype="f4"))
        _write(f, "4/Metallicity", np.full(8, 0.02, dtype="f4"))
        _write(f, "4/StarFormationTime", np.array([0.1, 0.2, 0.25, 0.1, 0.1, 0.1, 0.2, 0.2], dtype="f4"))
        _write(f, "0/Position", gas_pos)
        _write(f, "0/Mass", np.full(6, 1e-4, dtype="f4"))
        _write(f, "0/Metallicity", np.full(6, 0.01, dtype="f4"))
        _write(f, "0/SmoothingLength", np.full(6, 2.0, dtype="f4"))
        _write(f, "0/StarFormationRate", np.array([0.5, 0, 1.5, 0, 0, 0], dtype="f4"))
        _write(f, "5/ID", np.array([101, 102, 103], dtype="u8"))
        _write(f, "5/Position", bh_pos)
        _write(f, "5/BlackholeMass", np.array([1e-4, 2e-4, 3e-4], dtype="f4"))
        _write(f, "5/BlackholeAccretionRate", np.array([1e-3, 0, 2e-3], dtype="f4"))


@pytest.fixture(scope="session")
def fake_pig_dir(tmp_path_factory):
    """Directory holding PIG_007_subfind, the fake snapshot."""
    d = tmp_path_factory.mktemp("pig")
    make_fake_pig(d / "PIG_007_subfind")
    return d
