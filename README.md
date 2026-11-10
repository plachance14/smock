# smock

Mock JWST/NIRCam observations of galaxies from the ASTRID and AMBRA simulations,
made with [synthesizer](https://synthesizer-project.github.io/synthesizer/).

Pick a galaxy by its subhalo index, FoF group or black hole ID, and smock reads its stars, gas
and black holes from the snapshot. These values will be converted to physical units. The walkthrough notebook then goes step-by-step through the process of producing a mock observation using synthesizer. During this process we:

- Produce the spectrum of the stars and, optionally, the AGN, with line-of-sight dust and dust emission;
- Produce NIRCam photometry and images in seven filters (F090W to F444W). We produce images raw (no instrumentation), w/ PSF, and w/PSF and noise;
- Find the galaxy in the image (segmentation), measure the galaxy's properties, and perform morphological fitting;
- Output sample figures of the images and the spectrum.

The notebook is meant to be read and adapted: we try to call out each choice with a named variable at the top of
its cell, and the synthesizer calls are written out, so you can take pieces into your own code.

## Getting started
Follow [SETUP.txt](SETUP.txt): get smock and the [data folder from Google Drive](https://drive.google.com/drive/folders/1aruCXByevDyer_LeufUUbo_knz311sDV?usp=sharing), make a Python
environment, and set the data's path. Then open [notebooks/walkthrough.ipynb](notebooks/walkthrough.ipynb)
and step through it cell by cell.

If you are interested in the differences / updates from the AMBRA LRDs pipeline, see
[CHANGES_FROM_OLD_PIPELINE.md](CHANGES_FROM_OLD_PIPELINE.md) for what changed and how to switch
back.

How the downloaded data (the test cutout, the PSFs and the AGN grids) were made is in
[smock-recipes](https://github.com/plachance14/smock-recipes), referred to in the code as
`smock-recipes/<product>`.

## What's here
- `smock/`: the library. `config.py` holds the data paths; `data.py` reads snapshots and
  converts units; `agn.py`, `dust.py` and `imaging.py` are small helpers for the notebook;
  `analysis.py` does segmentation and the measurements in the source; `io.py` saves spectra and
  images; `plotting.py` makes the figures; `memory.py` sizes the large-galaxy notebook's chunks.
- `notebooks/walkthrough.ipynb`: one galaxy, from snapshot to figures.
- `notebooks/walkthrough_large_galaxy.ipynb`: Shares many of the same steps as the walkthrough notebook, but is used on a very large galaxy. This allows us to demonstrate the chunking of star particles to allow the job to fit in memory.
- `notebooks/make_subhalo_list.ipynb`: Creates a list of subhalo indices with specific properties, to allow for easier access when looking to produce mock observations of specific types of galaxies. The example cuts are on stellar mass and black-hole luminosity, but you should customize this to your own needs.
- `scripts/run_notebook.py`: runs a notebook from top to bottom without opening it.
- `scripts/run_batch.py`: mock observations of many galaxies in one job (e.g. a SLURM array over a subhalo list from `make_subhalo_list.ipynb`), with the large-galaxy notebook's steps and settings; each galaxy gets the notebooks' three files, and each task a catalogue that `scripts/merge_catalogues.py` joins into one table. Its docstring says how to run it.
- `tests/`: checks against known answers (`python -m pytest`). `tests/test_batch.py` checks that the batch script gives the notebooks' results; it takes 15-30 minutes, so it runs only with `SMOCK_SLOW_TESTS=1`.

## How to cite
You don't need to cite smock itself. If you publish something made with it, please cite
Synthesizer, which does the modelling:
- Lovell et al. 2025, OJAp, 8, [doi:10.33232/001c.145766](https://doi.org/10.33232/001c.145766)
- Roper et al. 2026, JOSS, 11, 9436, [doi:10.21105/joss.09436](https://doi.org/10.21105/joss.09436)
- Vijayan et al. 2026, [arXiv:2607.27467](https://arxiv.org/abs/2607.27467) (Synthesizer's grids)

Beyond that, it depends on which parts you use. For reference:
- the pipeline as a whole: LaChance et al. 2026, [arXiv:2609.26999](https://arxiv.org/abs/2609.26999)
  (LRDs in AMBRA), and LaChance et al. 2025, OJAp, 8,
  [doi:10.33232/001c.129991](https://doi.org/10.33232/001c.129991) (galaxy morphologies in ASTRID)
- BPASS, the stellar population models: Stanway & Eldridge 2018, MNRAS, 479, 75
- CLOUDY, which made the AGN grids: Gunasekera et al. 2025, [arXiv:2508.01102](https://arxiv.org/abs/2508.01102)
- ASTRID: Bird et al. 2022, MNRAS, 512, 3703; Ni et al. 2022, MNRAS, 513, 670
- AMBRA: Zhou et al. 2026, [arXiv:2604.01123](https://arxiv.org/abs/2604.01123)
- the default dust-to-metal ratio: Vijayan et al. 2019, MNRAS, 489, 4072
- statmorph, for the morphology: Rodriguez-Gomez et al. 2019, MNRAS, 483, 4140
- the old dust normalisation (`DUST_MODEL = "code_units"`): Bird et al. 2022 at high redshift,
  Hafezianzadeh et al. 2026, [arXiv:2605.13843](https://arxiv.org/abs/2605.13843) at low redshift

## Questions and problems
Open an [issue on GitHub](https://github.com/plachance14/smock/issues), or email Patrick LaChance, p.lachance@soton.ac.uk.

## Licence
MIT; see [LICENSE](LICENSE).
