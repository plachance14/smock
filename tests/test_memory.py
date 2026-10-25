"""Known-answer tests for smock.memory."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from smock import memory

ON_LINUX = sys.platform.startswith("linux")


def test_limit_from_slurm_per_node():
    assert memory.memory_limit_gb({"SLURM_MEM_PER_NODE": "16384"}) == pytest.approx(16384 * 1024**2 / 1e9)


def test_limit_from_slurm_per_cpu():
    env = {"SLURM_MEM_PER_CPU": "4500", "SLURM_CPUS_ON_NODE": "4"}
    assert memory.memory_limit_gb(env) == pytest.approx(4 * 4500 * 1024**2 / 1e9)


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
