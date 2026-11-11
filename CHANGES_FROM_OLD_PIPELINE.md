# Changes from the old pipeline

smock is a rewrite of Patrick LaChance's AMBRA LRDs pipeline, the synthesizer-based code that
made mock catalogues, spectra, images and segmentation maps of AMBRA galaxies at z = 5–8. If
you compare smock's outputs with that pipeline's products, or with results built on them,
expect differences: some come from bugs fixed, some from better defaults, and some from newer
synthesizer versions. This note lists the ones that change numbers or how you work. Effects
quoted were measured on the test galaxies, not on a population.

## Switching back to the old pipeline's choices

Where a default changed and the old choice is still reasonable, it is one setting away in
`notebooks/walkthrough.ipynb`:

| What | smock default | Old pipeline | To switch back |
|---|---|---|---|
| Stellar mass given to synthesizer | birth mass (`Mass + TotalMassReturned`) | current mass, so stars were ~0.15 dex too faint | `STAR_MASS = "current_masses"` |
| Birth-cloud dust, τ_V at Z = 0.01 | 1 (FLARES) | 3 | `KAPPA_BC = 3` |
| ISM dust | an opacity per unit dust mass (`KAPPA_DUST = 0.3`) × each galaxy's dust-to-metal ratio from its metallicity and age (Vijayan et al. 2019), tuned on ASTRID's UV luminosity functions at z = 2–4 | dust = metals, with a fixed opacity in code units, 10^4.1 at every redshift (tuned at z ≈ 4) | `DUST_MODEL = "code_units"` and `KAPPA_CODE = 10**4.1` (with `"code_units"` alone, 10^2.9 up to z = 2, rising to 10^4.1 at z = 4) |
| Particle crop (stars, gas and black holes) | square prism, half the image width + 10 px | 1.5″ cylinder, depth ±100 ckpc/h | `CROP_SHAPE = "cylinder"`, `CROP_SIZE = 1.5` (no depth cut, which only matters for distant particles in whole-group mode) |
| Star smoothing lengths | 60th-neighbour distance | 32nd | `load_galaxy(..., star_smoothing_k=32)` |
| AGN gas | standard BLR grid (n_H = 10^10 cm⁻³, N_H = 10^23 cm⁻²) | gas-enshrouded (n_H = 10^9.5, N_H = 10^24.5, v_turb = 400 km/s, 0.1 Z☉) | the gas-enshrouded grid and point given next to `AGN_GRID_POINT` |
| AGN dust slope | −1, the stars' slope | −2 | `AGN_DUST_SLOPE = -2` |
| Aperture for spectra and catalogue values | the whole segment | segment pixels within 2 r½ | `APERTURE = "segment_and_r_half"` |
| Stars in the aperture spectrum | each star's light spread over its smoothing kernel, as in the images (a synthesizer data cube added up over the aperture), so a star near the edge counts in part | the stars whose positions fall inside, each with all its light | `APERTURE_SPECTRUM = "points"` (all three: walkthrough, large-galaxy notebook, batch script) |
| Compactness apertures | exact pixel overlaps | edges tapered over half a pixel, which biased compactness ~9% low | `COMPACTNESS_METHOD = "taper"` |

The AGN covering fraction is new. Its default, f = 1, gives the old behaviour: all the disc's
light passes through the gas.

The default dust model is new: an opacity per unit dust mass in physical units, with each
galaxy's dust-to-metal ratio from its metallicity and age (Vijayan et al. 2019). One value then
works at every redshift. At z ≥ 6 it gives small galaxies 2–3× less dust per metal mass than the
old pipeline's 10^4.1, and massive galaxies at z ≈ 5 about twice as much. In a test on the old
pipeline's AMBRA LRDs (with its other settings), about 3%, 13% and 23% of those still selected
with 10^4.1 drop out at z = 5, 6 and 7.

**If you select little red dots (LRDs),** three other changes in the table matter far more than the
dust model: birth masses, the weaker birth-cloud dust (1 instead of 3) and the exact compactness
apertures. The first two add starlight around the AGN, and the third removes the old ~9% low bias
in compactness. Sources therefore come out less compact, a little bluer and brighter. In a test on
the old pipeline's AMBRA LRDs at z = 5–7 (all with the gas-enshrouded AGN), smock with the old
pipeline's settings found 74–97% of them again. With smock's defaults it found only 3–8%, and most
of the rest failed the compactness cut. Switched to smock's default one at a time:

| Setting | Share of the old LRDs still found |
|---|---|
| Birth-cloud dust | 21–51% |
| Birth masses | 50–90% |
| Exact compactness apertures | 28–69% |

The AGN dust slope, the disc temperature and the aperture spectrum hardly changed the selection. To
compare with the old pipeline's LRD results, switch those three back (table above). I'm still looking
into which settings are right for comparing with observed LRDs. Observers measure compactness a little
differently: on the noisy image, at the source's centroid, keeping the faint light around the source, and
in mosaics whose stars look slightly broader than our PSF. smock can do the same (`COMPACTNESS_IMAGE =
"noisy"`, `COMPACTNESS_MASK = "others"`, `COMPACTNESS_CENTRES = "centroid"`, `PSF_SOURCE = "stpsf_mosaic"`),
and step 15 can measure colours as photometric catalogues do (`LRD_PHOTOMETRY = "matched_circle"`).
Step 15 also offers other published selections besides the old pipeline's (`LRD_SELECTION`: Kokorev et
al. 2024, Rinaldi et al. 2025 and Rinaldi et al. 2026). For comparing with spectra, step 12 can broaden the AGN's
lines (`AGN_LINE_WIDTH`), and step 15 can measure the lines (`LINE_CHECKS`) and select LRDs on the spectrum's
continuum slopes (`LRD_COLOURS = "vshape"`, de Graaff et al. 2025); its colour and compactness cuts can be
mixed (`LRD_COLOURS`, `LRD_COMPACTNESS`).

## Changes with no switch back

**Stars and dust**
- All quantities are converted to physical units before reaching synthesizer. With
  `DUST_MODEL = "code_units"` the dust opacity κ is still entered in the old calibration units
  and converted, so τ is unchanged; the default model's opacity is in physical units already.
- The emission model is synthesizer's `LOSStellarEmission`. Stars older than 10 Myr now
  get nebular emission too, which adds 0.004–0.02 dex in NIRCam at z = 6.
- Absorbed starlight is re-emitted as an energy-balanced greybody (100 K, β = 2), and so is
  the AGN's. Both are part of the total spectrum; the old pipeline had no dust emission. At
  100 K it peaks in the far-infrared.
- Line-of-sight dust still integrates each gas particle's projected SPH kernel, but newer
  synthesizer (≥ 1.1) changes one detail. When a star or black hole sits inside a gas
  particle's kernel along the line of sight, only the part of the kernel in front of it
  counts; before, that particle counted fully or not at all, depending on whether its centre
  was in front. No systematic shift is expected on average, but individual τ values can
  differ, mostly for black holes embedded in dense gas.
- Star smoothing lengths are computed after wrapping the periodic box, which matters only
  for groups that straddle the box edge.

**Simulation data**
- A subhalo's or black hole's FoF group is the one whose particle ranges contain its
  particles. The old pipeline took the nearest group centre among the first 3 million groups.
- The redshift and cosmology are read from the snapshot header. Snapshot numbers map to
  different redshifts on different machines: the old z = 8 products came from snapshot 028,
  which exists only on Frontera.
- Black-hole accretion rates use MP-Gadget's exact time unit (977.79 Myr/h, not 980): +0.23%
  in Ṁ and L_bol.

**AGN**
- The AGN spectrum comes from new CLOUDY grids, read and interpolated by synthesizer. The
  ionisation parameter is fixed at each grid point (log U = −1.5 by default), and the
  spectra are normalised per unit bolometric luminosity. The old grids fixed the luminosity
  and radius instead, so U varied with disc temperature and density. For the old pipeline's
  LRDs this makes the AGN up to ~1.8× brighter in F150W and F200W at z = 5–6 (rest-UV) and
  5–20% brighter in the rest-optical. The walkthrough's testing option
  `AGN_U_FROM = "old_pipeline"` gives each black hole the U it had in the old runs.
- The new CLOUDY runs use GalacticConcordance abundances with every element switched on.
  The old runs had Ca, Na, Al, Ni and several other elements off, so they lacked those
  elements' lines and opacity.
- The AGN now gets IGM absorption. Its dust still acts only at 2 nm–100 µm, as before.
- The disc temperature (the CLOUDY `agn` continuum's blue-bump temperature) is now the one
  synthesizer itself gives a black hole: a thin disc's temperature at 10 Schwarzschild radii
  (Inayoshi & Maiolino 2025). The old pipeline used the disc's hottest point, at (49/36) of
  the innermost stable orbit, which is 0.17 dex hotter; `DISC_TEMPERATURE = "T_max"` gives
  that back.
- Spectra are interpolated linearly in flux along disc temperature (before: in log flux).
- The grids' disc temperatures reach 10^6.5 K. The old tables stopped at 10^6 K, and hotter
  discs were held at 10^6 K without a warning. At the cool end the grids start at about
  10^4.2 K; synthesizer holds cooler discs at that edge, and the walkthrough reports any black
  hole outside the grid.

**Images**
- The PSFs are STPSF (formerly WebbPSF) models on the 0.03″ grid. The old pipeline used
  Gaussians.
- Noise is white, matched to survey depths (JADES-Medium-like by default, or CEERS-like, or
  DECaLS-like for DECam).
  The old pipeline drew Poisson noise for 10 ks exposures; there is no Poisson option yet.
- Images use synthesizer's orientation; the old images were transposed relative to it.
- In the RGB figures, the brightest value of the colour stretch comes from the three filters
  shown; the old figures took it from all seven.
- All photometry, including the LRD colours, uses the real filter curves. The old colours
  came from top-hat band averages.

**Segmentation and morphology**
- The detection threshold is 1.5 × the known noise per pixel. The old pipeline estimated the
  noise from the image itself, which reads high when a bright galaxy fills the frame (13%
  high for one test galaxy).
- Two bugs in the half-mass radius are fixed: an index error when the central pixel held
  half the mass, and a cumulative sum taken over the whole map instead of the segment.
- The LRD source check measures how far the source's centre is from the catalogue position
  as a true distance. The old pipeline left out a square root, so its limit of 40 worked out
  at about 7 pixels (5.6–6.0 ckpc/h at z = 5–8); smock's default is 10 ckpc/h
  (`max_offset_ckpch`), a little looser, as the old limit was meant to be.
- statmorph gets the known noise as its weight map and the same PSF as the images.
- Failures are flagged separately (`segmentation_ok`, `morphology_ok`). In the old pipeline
  a morphology failure still left `segmentation_passed` True.
- Two new options, off by default, so they change nothing unless switched on: finding the
  sources on a PSF-smoothed copy of the image, for smoother segment edges (`DETECT_SMOOTH`), and
  measuring the in-source photometry on the PSF-convolved or noisy images as well as from the
  particles (`IMAGE_PHOTOMETRY`; `LRD_PHOTOMETRY` picks which one the LRD colours use).

**Outputs**
- Spectra are saved as `.npz` files with explicit rest- and observed-frame columns, not
  pickled DataFrames.
- The old "νL_ν" column was ν_obs · L_ν,rest, i.e. νL_ν/(1 + z); smock's `*_nuLnu_rest`
  column is the true rest-frame value. Magnitudes and fluxes were never affected.

**Not carried over:** Bluetides, the time-series and black-hole-history tools (a black-hole
ID can still select a galaxy), IFU and spectral instrument modelling, and the old voxel
dust method.
