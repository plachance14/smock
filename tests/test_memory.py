"""Known-answer tests for smock.memory."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from smock import memory

ON_LINUX = sys.platform.startswith("linux")


def test_limit_from_slurm_per_node():
    assert memory.memory_limit_gb({"SLURM_MEM_PER_NODE": "16384"}, cgroup=False) == pytest.approx(16384 * 1024**2 / 1e9)


def test_limit_from_slurm_per_cpu():
    env = {"SLURM_MEM_PER_CPU": "4500", "SLURM_CPUS_ON_NODE": "4"}
    assert memory.memory_limit_gb(env, cgroup=False) == pytest.approx(4 * 4500 * 1024**2 / 1e9)


@pytest.mark.skipif(not ON_LINUX, reason="reads /proc")
def test_free_is_limit_minus_use():
    used = memory.rss_gb()
    assert 0 < used < 100
    assert memory.free_gb(limit_gb=used + 1.0) == pytest.approx(1.0, abs=0.05)


@pytest.mark.skipif(not ON_LINUX, reason="reads /proc")
def test_peak_rise_of_an_allocation():
    def allocate():
        a = np.ones(50_000_000)        # 0.4 GB, touched so it is resident
        return float(a[-1])
    result, rise = memory.peak_rise_gb(allocate)
    assert result == 1.0
    assert rise == pytest.approx(0.4, abs=0.05)


def test_spectra_per_star_estimate():
    models = {"top": SimpleNamespace(per_particle=True, emitter="stellar"),
              "a": SimpleNamespace(per_particle=True, emitter="stellar"),
              "b": SimpleNamespace(per_particle=False, emitter="stellar"),
              "agn": SimpleNamespace(per_particle=True, emitter="blackhole")}
    model = type("M", (), {"saved_labels": list(models), "__getitem__": lambda self, k: models[k]})()
    assert memory.spectra_per_star_estimate(model) == 7                     # 2 x 2 + 1 = 5 < in flight
    assert memory.spectra_per_star_estimate(model, in_flight=3) == 5


def test_stars_per_chunk():
    assert memory.stars_per_chunk(10.0, 1e-3) == 5000                       # half of 10 GB at 1 MB per star
    assert memory.stars_per_chunk(10.0, 1e-3, safety=0.25) == 2500
    assert memory.stars_per_chunk(0.01, 1e-3) == 100                        # never below the minimum


def _fake_cgroups(tmp_path, v2=False):
    """A cgroup tree like a SLURM step's: job limit 32 GB, step limit 8 GB (or no step limit)."""
    root = tmp_path / "cgroup"
    base = root if v2 else root / "memory"
    step = base / "slurm/uid_1/job_2/step_0"
    step.mkdir(parents=True)
    name = "memory.max" if v2 else "memory.limit_in_bytes"
    (base / name).write_text("max\n" if v2 else "9223372036854771712\n")         # the top: no limit
    (base / "slurm/uid_1/job_2" / name).write_text(f"{32 * 10**9}\n")
    proc = tmp_path / "proc_cgroup"
    proc.write_text("0::/slurm/uid_1/job_2/step_0\n" if v2 else
                    "8:cpu,cpuacct:/slurm/uid_1/job_2/step_0\n7:memory:/slurm/uid_1/job_2/step_0\n")
    return root, proc, step / name


@pytest.mark.parametrize("v2", [False, True])
def test_cgroup_limit_takes_the_smallest_enforced_limit(tmp_path, v2):
    root, proc, step_file = _fake_cgroups(tmp_path, v2)
    assert memory.cgroup_limit_gb(proc, root) == pytest.approx(32.0)      # only the job is limited
    step_file.write_text(f"{8 * 10**9}\n")
    assert memory.cgroup_limit_gb(proc, root) == pytest.approx(8.0)       # a smaller step limit wins


def test_cgroup_limit_none_without_a_limit(tmp_path):
    proc = tmp_path / "proc_cgroup"
    proc.write_text("7:memory:/user.slice\n")
    (tmp_path / "cgroup/memory/user.slice").mkdir(parents=True)
    (tmp_path / "cgroup/memory/user.slice/memory.limit_in_bytes").write_text("9223372036854771712\n")
    assert memory.cgroup_limit_gb(proc, tmp_path / "cgroup") is None
    assert memory.cgroup_limit_gb(tmp_path / "missing", tmp_path / "cgroup") is None

