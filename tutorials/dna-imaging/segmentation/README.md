# DNA simulation and segmentation

`DNA_simulation_and_segmentation.ipynb` simulates closed DNA molecules (plasmids
and minicircles) on a substrate as AFM height images, then trains or loads a
two-headed U-Net that segments the DNA and its crossings in simulated and real
AFM data. Real `.spm` scans are analysed with TopoStats using the U-Net.

## Contents

| Path | What it is |
|---|---|
| `DNA_simulation_and_segmentation.ipynb` | the notebook (Part 1: dataset generation, Part 2: segmentation) |
| `dna_simulation_helpers.py` | every function used in Part 1 and the benchmark memory tracker |
| `unet_segmentation_helpers.py` | every function used in Part 2 |
| `setup_topostats.py` | builds the verified TopoStats contour profile for Python 3.10-3.12 |
| `TOPOSTATS_PYTHON312.md` | dependency audit, exact compatibility changes and validation |
| `checkpoints/` | trained networks (see below) |
| `configs/topostats_{relaxed,supercoiled,minicircle}.yaml` | TopoStats settings for each kind of DNA |
| `requirements.txt` | Python packages (see *Setting up a new machine*) |
| `../common-assets/noise/` | blank `.spm` scan and PSD noise model used to add realistic noise |
| `../common-assets/example-images/` | real example molecules shown in the notebook |
| `../common-assets/example-spm/` | a real relaxed-plasmid scan for the TopoStats demonstration |
| `../common-assets/real-test-data/` | real height maps (`.npy`, nm, 0.5 nm/pixel) with DNA masks |

Class definitions (dataset, loss, metrics, networks, TopoStats predictor) are in
the notebook itself.

## Setting up a new machine

The notebook needs this folder **and** `../common-assets/` next to it (it reads
the noise files, example images, example scan and real test data from there).

```bash
conda create -n dna-afm python=3.12.13
conda activate dna-afm
# PyTorch with the CUDA build for your GPU, see https://pytorch.org (CPU build also works)
pip install torch
pip install -r requirements.txt
python setup_topostats.py
jupyter lab
```

Then open `DNA_simulation_and_segmentation.ipynb` from this folder. OpenMM picks
CUDA, OpenCL or CPU automatically; on a machine without a GPU the MD pathway
still works (slower), and Part 2 runs on the CPU.

## Running on Colab or Kaggle

Use the *Open in Colab* badge and run section 0 in a fresh Python 3.12.13
runtime. It downloads the helpers, setup script, four checkpoints, three
fine-tuned presets, shared noise assets and example `.spm` scan. On Kaggle,
enable Internet first. Set `INSTALL_TOPOSTATS = False` if raw `.spm` analysis
is not needed. The full real test dataset is not downloaded automatically.

The setup builds `topostats==2.4.2+asap312.1` from the small, checksum-verified
official PyPI wheel. This **ASAP contour profile** supports Python 3.12 for
the notebook's scan-processing workflow; it deliberately omits the unused
Jones-polynomial knot/link classifier and its `topoly` dependency. Filtering,
grain detection, U-Net segmentation, crossing resolution, ordered traces,
splines, contour lengths and circularity use the upstream code. This is a
modified build, not an official TopoStats Python 3.12 release. See
[the compatibility audit](TOPOSTATS_PYTHON312.md).

There is no full Git clone, blanket warning suppression or
`--ignore-requires-python`. If installation changes numerical libraries that
were already imported, restart the session once and rerun setup.

Section 15 defaults to `TOPOSTATS_CONFIG_SOURCE = "preset"`: the experiment's
fine-tuned relaxed/supercoiled/minicircle settings, with editable common
overrides. Set it to `"custom"` to use a complete TopoStats 2.4.2 YAML instead;
enter its path in `TOPOSTATS_CUSTOM_YAML` or at the prompt. The custom nested
scientific settings are retained and validated. The notebook controls input
and output paths, the selected checkpoint, one worker and plot saving.

## Running

Run the notebook from this folder (paths are relative to it).

| Goal | Cells |
|---|---|
| generate a dataset | Part 1, sections 1-8 |
| segment only | the imports cell of section 1.1, then Part 2, sections 9-15 |
| everything | top to bottom |

Main dependencies: NumPy, SciPy, Matplotlib, scikit-image, pandas, PyTorch,
Lightning, Deeplay, DeepTrack, joblib, psutil; OpenMM for the MD pathway;
TopoStats 2.4 for `.spm` analysis. Dataset generation takes about 15 s per
molecule with MD (GPU) and about 2 s without.

## Two simulation pathways

* **MD** (`USE_MD = True`): a 3-D persistent random-walk ring is relaxed onto the
  substrate with an annealed coarse-grained OpenMM force field (bonds,
  `K_ANGLE` bending, surface attraction, height confinement, WCA repulsion),
  then rendered with the width/taper AFM renderer.
* **Non-MD** (`USE_MD = False`): a calibrated 2-D persistent ring drawn
  directly on the substrate, rendered with the calibrated non-MD settings.

The contour length is set directly (`CONTOUR_LENGTH_NM`, and
`CONTOUR_LENGTH_RANGE_NM` for datasets): the bead count is the contour length
divided by the bond length, rounded, and the bond is adjusted so that
beads × bond equals the request. Measured contours match exactly for non-MD and
within about 0.3 % for MD (thermal bond fluctuations).

All defaults are the optimised values used to generate the training data of the
provided checkpoints (MD: `DNA_simulated_1000_v2` settings; non-MD: run 8 fit
plus the crossing-height calibration of 12 October 2026).

## Checkpoints

All four were trained with the same recipe: 800 simulated molecules (25 %
115.26 nm minicircles, 75 % 768.4 nm plasmids), already-filtered TopoStats
grain crops resized to 512 px, heights normalised over −1…5 nm,
DNA loss = 0.4 BCE + 0.6 Dice + 0.5 clDice, crossing focal loss,
Adam 3 × 10⁻⁴, batch 4, 50 epochs, seed 3, best validation loss kept.

| File | Architecture | Training data | Selected epoch | Val. loss | Val. Dice | Val. IoU |
|---|---|---|---|---|---|---|
| `md_res_unet.pt` | Res-U-Net (7.6 M) | MD | 41 | 0.309 | 0.889 | 0.800 |
| `md_simple_unet.pt` | simple U-Net (2.1 M) | MD | 34 | 0.297 | 0.885 | 0.795 |
| `non_md_res_unet.pt` | Res-U-Net (7.6 M) | non-MD | 7 | 0.284 | 0.902 | 0.821 |
| `non_md_simple_unet.pt` | simple U-Net (2.1 M) | non-MD | 36 | 0.290 | 0.884 | 0.792 |

Each file holds `state_dict`, `metadata` (architecture, preprocessing, training
and loss settings, validation metrics) and the per-epoch `history`. Validation
scores are on each pathway's own validation crops, so they are not directly
comparable across pathways.
