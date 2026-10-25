"""Per-machine paths: the one place to edit when data move.

Three kinds of data are kept apart:

- simulation input: the PIG snapshots (listed per machine in ``EXTERNAL``) and the public
  test cutout (``TEST_DATA_DIR``, in the repo, git-ignored, downloaded separately);
- synthesizer data: SPS and AGN grids, our 0.03" NIRCam PSFs and noise templates, in
  synthesizer's own data folder (``SYNTHESIZER_DIR``: set it with the environment variable
  of the same name, which synthesizer reads too): grids in grids/, PSFs and noise in
  instrument_cache/. synthesizer also keeps its copies of the filter curves there
  (svo_filter_cache/): the download includes the NIRCam ones, and others are fetched on first use;
- produced data: spectra, images and plots (``OUTPUT_DIR``, in the repo, git-ignored).

Library code takes paths as arguments and gets its defaults from here; it never spells
out a cluster path itself.

Override without editing this file::

    export SMOCK_ROOT=/path/to/your/copy/of/smock              # repo root
    export SMOCK_MACHINE=vera                                  # force the machine (default: detected)
    export SYNTHESIZER_DIR=/path/to/synthesizer_data           # holds grids/, instrument_cache/ (synthesizer reads it too)
    export SMOCK_OUTPUT_DIR=/path/for/outputs

Importing this module only reads environment variables; it opens no files.
"""
import os
from pathlib import Path

from platformdirs import user_data_dir    # comes with synthesizer


def detect_machine():
    """'vera' (PSC, /hildafs present) or 'local' (anything else)."""
    forced = os.environ.get("SMOCK_MACHINE")
    if forced:
        return forced
    return "vera" if Path("/hildafs").is_dir() else "local"


MACHINE = detect_machine()
ROOT = Path(os.environ.get("SMOCK_ROOT", Path(__file__).resolve().parents[1]))

# The public ASTRID test cutout: a BigFile laid out like a PIG
# (made by smock-recipes/test_cutout).
TEST_DATA_DIR = ROOT / "test_data"
TEST_CUTOUT_PIG_DIR = TEST_DATA_DIR / "ASTRID"   # holds PIG_<snap>_subfind/, like astrid_pig_dir
# Spectra, images and plots made by the notebooks and scripts.
OUTPUT_DIR = Path(os.environ.get("SMOCK_OUTPUT_DIR", ROOT / "outputs"))

# Simulation snapshots outside the repo. None = not available on that machine.
_EXTERNAL = {
    "vera": {
        # Read-only shared data, owned by others: never write here.
        "astrid_pig_dir": Path("/hildafs/datasets/Asterix/PIG2"),       # PIG_<snap>_subfind/
        "ambra_pig_dir": Path("/hildafs/datasets/Asterix/AMBRA/PIG"),   # PIG_<snap>_subfind/
    },
    "local": {
        "astrid_pig_dir": None,
        "ambra_pig_dir": None,
    },
}
EXTERNAL = _EXTERNAL.get(MACHINE, _EXTERNAL["local"])

# synthesizer data: the same folders synthesizer itself uses, found the same way (synthesizer
# 1.3.1, data/initialise.py), so one setting serves both. SYNTHESIZER_DIR if set, else
# synthesizer's default per-user folder. Inside it, grids/ holds the grids and
# instrument_cache/ the instruments (filters, PSFs, noise; as synthesizer's data index, syndex,
# sorts them); either can be moved on its own with SYNTHESIZER_GRID_DIR or
# SYNTHESIZER_INSTRUMENT_CACHE.
SYNTHESIZER_DIR = Path(os.environ.get("SYNTHESIZER_DIR") or user_data_dir("Synthesizer"))
GRID_DIR = Path(os.environ.get("SYNTHESIZER_GRID_DIR") or SYNTHESIZER_DIR / "grids")   # sps/..., agn/...
INSTRUMENT_DIR = Path(os.environ.get("SYNTHESIZER_INSTRUMENT_CACHE") or SYNTHESIZER_DIR / "instrument_cache")
PSF_DIR = INSTRUMENT_DIR / "psfs"        # our STPSF NIRCam PSFs (made by smock-recipes/psfs)
NOISE_DIR = INSTRUMENT_DIR / "noise"     # blank-sky noise templates (optional, to come)
