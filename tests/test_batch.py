"""The batch script against the notebooks, on one test galaxy (303604): run only when asked,
because it takes 15-30 minutes and needs the test cutout, the grids and the PSFs.

    SMOCK_SLOW_TESTS=1 python -m pytest tests/test_batch.py -v

Run it from smock_env, the environment SETUP.txt makes: the notebooks run on its kernel
(scripts/run_notebook.py's default), so the batch script must use the same packages. From another
environment, small version differences (e.g. in statmorph's Sersic fit) break the exact comparison.

It runs, with the same settings, scripts/run_batch.py, the large-galaxy notebook and the
walkthrough (each from a copy with SIDX and the chunk size set, via scripts/run_notebook.py), and
compares every array the three save. The batch script follows the large-galaxy notebook step by
step, so their files must be identical (stars in chunks of 1000, so the chunks are really
tested); the walkthrough adds the stars up in a different order, so it must agree to rounding.
Run it after changing the notebooks or the batch script: it is what keeps them in step.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from smock import config

REPO = Path(__file__).parents[1]
SIDX, SNAP, CHUNK = 303604, 47, 1000
STEM = f"ASTRID_{SNAP:03d}_sidx{SIDX}"
IGNORE = {"created", "smock_commit"}           # when and with which code version each file was made
pytestmark = pytest.mark.skipif(os.environ.get("SMOCK_SLOW_TESTS") != "1",
                                reason="slow: set SMOCK_SLOW_TESTS=1 to run")


def notebook_copy(name, tmp, changes):
    """A copy of notebooks/<name>.ipynb with the lines starting with each key replaced."""
    nb = json.loads((REPO / "notebooks" / f"{name}.ipynb").read_text())
    for cell in nb["cells"]:
        lines = "".join(cell["source"]).splitlines(keepends=True)
        for i, line in enumerate(lines):
            for start, new in changes.items():
                if line.startswith(start):
                    lines[i] = new + "\n"
        cell["source"] = lines
    path = tmp / f"{name}.ipynb"
    path.write_text(json.dumps(nb))
    return path


def saved_arrays(path):
    """Every array in a saved .npz (structured arrays column by column), and its metadata."""
    x = np.load(path)
    out = {}
    for k in x.files:
        if k != "meta":
            out.update({f"{k}.{c}": x[k][c] for c in x[k].dtype.names} if x[k].dtype.names else {k: x[k]})
    meta = {k: v for k, v in json.loads(str(x["meta"])).items() if k not in IGNORE}
    return out, meta


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("batch")
    env = {**os.environ, "SMOCK_OUTPUT_DIR": str(tmp)}
    pig = str(config.TEST_CUTOUT_PIG_DIR)
    (tmp / "list.txt").write_text(f"{SIDX}\n")
    subprocess.run([sys.executable, str(REPO / "scripts/run_batch.py"), "--snap", str(SNAP), "--sidx-file",
                    str(tmp / "list.txt"), "--pig-dir", pig, "--chunk-stars", str(CHUNK), "--out-dir",
                    str(tmp / "batch")], check=True, env=env)
    common = {"SIDX =": f"SIDX = {SIDX}", "SNAP =": f"SNAP = {SNAP}", "PIG_DIR =": "PIG_DIR = config.TEST_CUTOUT_PIG_DIR"}
    for name, extra in (("walkthrough_large_galaxy", {"CHUNK_STARS =": f"CHUNK_STARS = {CHUNK}"}), ("walkthrough", {})):
        subprocess.run([sys.executable, str(REPO / "scripts/run_notebook.py"),
                        str(notebook_copy(name, tmp, {**common, **extra}))], check=True, env=env)
    return {"batch": tmp / "batch", "large": tmp / "walkthrough_large_galaxy", "walkthrough": tmp / "walkthrough"}


@pytest.mark.parametrize("kind", ["spectrum", "aperture_spectrum", "images"])
def test_batch_matches_the_large_galaxy_notebook_exactly(outputs, kind):
    a, ma = saved_arrays(outputs["batch"] / f"{STEM}_{kind}.npz")
    b, mb = saved_arrays(outputs["large"] / f"{STEM}_{kind}.npz")
    assert set(a) == set(b)
    for k in a:     # NaN counts as equal: e.g. dust-emission magnitudes in the bluest filters
        assert np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f"), k
    assert ma == mb


@pytest.mark.parametrize("kind", ["spectrum", "aperture_spectrum", "images"])
def test_batch_matches_the_walkthrough_to_rounding(outputs, kind):
    a, _ = saved_arrays(outputs["batch"] / f"{STEM}_{kind}.npz")
    b, _ = saved_arrays(outputs["walkthrough"] / f"{STEM}_{kind}.npz")
    assert set(a) == set(b)
    for k in a:
        if a[k].dtype.kind == "f":
            np.testing.assert_allclose(a[k], b[k], rtol=1e-10, atol=1e-10 * np.nanmax(np.abs(b[k])), err_msg=k)
        else:
            assert np.array_equal(a[k], b[k]), k
