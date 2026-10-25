"""How much memory a job has, how much it is using, and how many stars fit in a chunk.

Used by the large-galaxy notebook to split a galaxy's stars into chunks that fit. Each star's
spectra cost the same whatever its neighbours, so memory grows linearly with the stars in a
chunk; measuring one small trial chunk with your own emission model gives the cost per star.

    free = memory.free_gb()                                   # what is left for this job
    _, rise = memory.peak_rise_gb(run_chunk, trial_stars)     # a trial chunk's peak [GB]
    n = memory.stars_per_chunk(free, rise / n_trial)          # a cautious chunk size

The measurements read /proc, so they work on Linux (any cluster); elsewhere they return None
and the notebook falls back on spectra_per_star_estimate. Sizes are in GB (1e9 bytes).
Nothing is read on import.
"""
import os
from pathlib import Path

BYTES_PER_GB = 1e9


def _proc_status_gb(key):
    """A memory line of /proc/self/status (VmRSS, VmHWM) [GB]; None where /proc is missing."""
    status = Path("/proc/self/status")
    if not status.exists():
        return None
    for line in status.read_text().splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1]) * 1024 / BYTES_PER_GB     # the kernel's "kB" are kiB
    return None


def rss_gb():
    """Memory this process is using now [GB] (resident set size); None off Linux."""
    return _proc_status_gb("VmRSS")


def memory_limit_gb(environ=None):
    """Memory this job may use [GB].

    Inside a SLURM job, its allocation: SLURM_MEM_PER_NODE, or SLURM_MEM_PER_CPU times the
    CPUs on the node (both in MiB). Otherwise the machine's total memory (shared, on a login
    node: see free_gb). None if neither is known.
    """
    env = os.environ if environ is None else environ
    mib = 1024**2 / BYTES_PER_GB
    if env.get("SLURM_MEM_PER_NODE"):
        return int(env["SLURM_MEM_PER_NODE"]) * mib
    if env.get("SLURM_MEM_PER_CPU"):
        cpus = int(env.get("SLURM_CPUS_ON_NODE") or env.get("SLURM_NTASKS") or 1)
        return int(env["SLURM_MEM_PER_CPU"]) * cpus * mib
    return _meminfo_gb("MemTotal")


def _meminfo_gb(key):
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text().splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1]) * 1024 / BYTES_PER_GB       # kiB
    return None


def free_gb(limit_gb=None):
    """Memory left for this job [GB]: its limit (memory_limit_gb, or limit_gb) minus what it
    uses now, and never more than the machine has available (other users share a login
    node). None if unknown."""
    limit = memory_limit_gb() if limit_gb is None else limit_gb
    used, available = rss_gb(), _meminfo_gb("MemAvailable")
    if limit is None or used is None:
        return available
    left = limit - used
    return left if available is None else min(left, available)


def peak_rise_gb(func, *args, **kwargs):
    """Run func(*args, **kwargs); return (its result, how far this process's peak memory rose
    above its memory just before [GB]). The rise is None off Linux.

    Resets the kernel's peak counter (VmHWM) first, by writing to /proc/self/clear_refs.
    """
    try:
        Path("/proc/self/clear_refs").write_text("5")
        before = rss_gb()
    except OSError:
        before = None
    result = func(*args, **kwargs)
    if before is None:
        return result, None
    return result, _proc_status_gb("VmHWM") - before


def spectra_per_star_estimate(model, emitter="stellar", in_flight=7):
    """Rough number of spectra held per particle while an emission model runs: each saved
    per-particle spectrum is kept twice (rest frame and observed), but while the model runs
    about `in_flight` intermediate ones exist at once, saved or not (7 for the walkthrough's
    stellar model, measured). Multiply by
    the number of wavelengths x 8 bytes for the memory per particle. Prefer a measured trial
    chunk (peak_rise_gb): this is the fallback.
    """
    n_saved = sum(1 for label in model.saved_labels
                  if model[label].per_particle and model[label].emitter == emitter)
    return max(in_flight, 2 * n_saved + 1)


def stars_per_chunk(free, gb_per_star, safety=0.5, minimum=100):
    """How many stars to put in a chunk: `safety` x the free memory [GB] / the cost per star
    [GB], rounded down, and at least `minimum`.

    The default safety factor (half the free memory) is deliberately cautious: a few extra
    chunks cost a little time, while a chunk slightly too big stops the run with an
    out-of-memory error.
    """
    return max(minimum, int(safety * free / gb_per_star))
