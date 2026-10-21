# DNA imaging tutorials

Tutorials on atomic force microscopy (AFM) images of DNA. Each tutorial lives
in its own folder with its notebook, helper scripts and trained networks;
data shared by several tutorials (noise files, real example images and scans,
real test data) is in [`common-assets/`](common-assets).

| Folder | Notebook | Content |
|---|---|---|
| [`segmentation/`](segmentation) | [`DNA_simulation_and_segmentation.ipynb`](segmentation/DNA_simulation_and_segmentation.ipynb) | Simulate closed DNA molecules on a substrate as AFM height images, then train or load a U-Net that segments the DNA and its crossings in simulated and real data, including real `.spm` scans analysed with TopoStats. |

---

# Simulation and segmentation of DNA imaged by AFM

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Prakhar-Dutta/ASAP/blob/main/tutorials/dna-imaging/segmentation/DNA_simulation_and_segmentation.ipynb)

The notebook has two parts:

1. **Dataset generation.** Closed DNA molecules (plasmids and minicircles) are
   simulated, relaxed onto a substrate and rendered as AFM height maps with
   realistic noise. Every sample has a height map (`.npy`, heights in nm), a
   binary **DNA mask** and a soft **crossing mask** that highlights where the
   molecule crosses itself, plus its metadata, all indexed in `manifest.csv`.
2. **Segmentation.** A two-headed U-Net (DNA and crossings), built with
   [Deeplay](https://github.com/DeepTrackAI/deeplay) and fed by a lazy
   [DeepTrack](https://github.com/DeepTrackAI/DeepTrack2) pipeline, is trained
   on the simulated data, or one of the four provided networks is loaded. It
   is evaluated on simulated data and then applied to real AFM data; real
   `.spm` scans are analysed with [TopoStats](https://github.com/AFM-SPM/TopoStats)
   using the U-Net for segmentation.

```text
contour length ─► chain (MD or non-MD) ─► AFM rendering + noise ─► image, DNA mask, crossing mask
                                                                          │
                                       manifest.csv ◄────────────────────┘
                                            │
            DeepTrack source ─► lazy loading ─► U-Net (train or load) ─► simulated / real data / TopoStats
```

The **contour length** is the main control: the length of the DNA backbone in
nm (1 bp ≈ 0.34 nm, so the default 768.4 nm is a 2.26 kbp plasmid and
115.26 nm a 339 bp minicircle). The notebook turns it into a chain of beads
(`round(contour / BOND_LENGTH)` beads, with the bond adjusted so that
beads × bond equals the request). Measured contour lengths match exactly for
the non-MD pathway and within about 0.3 % for the MD pathway.

Every function used by the notebook is in a helper script next to it
(`dna_simulation_helpers.py` for Part 1, `unet_segmentation_helpers.py` for
Part 2), documented with parameters, return values and examples. The classes
(dataset, loss, metrics, networks, TopoStats predictor) are written out in the
notebook.

## Running the notebook

**On your own machine.** Set up the environment as described in
[`segmentation/README.md`](segmentation/README.md) (`pip install -r
requirements.txt` after PyTorch), then open the notebook from the
`segmentation/` folder; all paths are relative to it, and `../common-assets/`
must be next to it.

**On Colab or Kaggle.** Open the notebook with the badge above (or upload it
to Kaggle) and run **section 0** first. On Colab and Kaggle it installs
Deeplay, DeepTrack and OpenMM and downloads the helper scripts,
`checkpoints/` and `configs/` from this repository (on Kaggle, switch on
*Internet* in the notebook settings). You can also download
`dna_simulation_helpers.py` and `unet_segmentation_helpers.py` yourself and
put them next to the notebook. `common-assets/` is not downloaded, so the
simulated images get no AFM noise and the real-data examples need your own
files; everything else runs.

**Which cells to run.**

| I want to… | Run |
|---|---|
| run on Colab or Kaggle | section 0 first, then as below |
| generate a dataset only | Part 1, sections 1 → 8 |
| segment only (provided networks, an earlier dataset, or my own data) | the imports cell of section 1.1, then Part 2, sections 9 → 15 |
| both | the whole notebook, top to bottom |

Sections 2-7 are optional checks; dataset generation (section 8) only needs
sections 1.1-1.3.

## Part 1: dataset generation

| Section | What it does |
|---|---|
| 1.1 Imports | Imports Part 1 and chooses the pathway with `USE_MD`. |
| 1.2 Variables | All simulation settings (below). |
| 1.3 Noise | Loads the noise source: a blank `.spm` scan, or the PSD noise model. |
| 2 Chain creation | Builds and plots one initial ring. |
| 3 MD relaxation | Explains the force field and tests the OpenMM installation. |
| 4 Relaxed chain | Generates one molecule; for MD, an animation of the relaxation. |
| 5 Real DNA | Real AFM images of a relaxed plasmid, a minicircle and a supercoiled plasmid, with their TopoStats contour lengths. |
| 6 Sanity check | Four molecules with their DNA and crossing masks, optionally with decoys. |
| 7 Benchmarking | Times every stage, tracks peak memory and projects the time for `N_SAMPLES`. |
| 8 Dataset generation | Writes the dataset and `manifest.csv`. |

### Two pathways

* **MD (`USE_MD = True`).** A closed 3-D persistent random walk is lifted
  `BASE_Z` above the substrate and relaxed onto it with a coarse-grained
  Langevin simulation in [OpenMM](https://openmm.org): harmonic bonds, bending
  stiffness (`K_ANGLE`), surface attraction, height confinement, a hard wall
  at z = 0 and a short-range WCA repulsion that stops the chain passing
  through itself. Confinement and repulsion are annealed: they start soft, so
  the ring can lose unnecessary crossings, and stiffen over the first 85 % of
  the run. About 15 s per molecule on a GPU; OpenMM uses CUDA, OpenCL or the
  CPU.
* **Non-MD (`USE_MD = False`).** A calibrated 2-D persistent ring drawn
  directly on the substrate and scaled to the exact contour length. About 2 s
  per molecule, no OpenMM needed, less realistic crossings.

Both pathways are rendered by the same AFM renderer, with optimised settings
for each pathway. These are the settings that produced the training data of the
provided networks.

### Settings (section 1.2)

| Group | Variables |
|---|---|
| Dataset | `OUT_DIR`, `N_SAMPLES` (1000), `BASE_SEED` |
| Image | `NX`, `NY` (512 px), `NM_PER_PIXEL` (1.0), `CONTOUR_LENGTH_NM` (768.4), `CHAIN_RESOLUTION_NM` (internal render grid, 0.5 nm), `TIP_RADIUS_NM` |
| Chain, both pathways | `BOND_LENGTH`, `PERSISTENCE_BONDS`, `K_ANGLE` |
| MD pathway | `BASE_Z` (55 nm), `N_FRAMES` (350), `STEPS_PER_FRAME` (300) |
| Non-MD pathway | `ANGLE_STIFFNESS_MULT`, `NON_MD_ANGLE_LIMIT_RAD`, `NON_MD_Z_STD_NM`, `NON_MD_BASE_Z_NM` |
| Rendering and masks | `DNA_HEIGHT_NM`, `DNA_MASK_DIAMETER_NM` (4.47 nm), `AFM_KW` (renderer: width/taper, blur, grain, crossing-height boost) |
| Crossing mask | `CROSS_MIN_SEP_BEADS`, `CROSS_SIGMA_CENTER_DIAM_MULT`, `CROSS_SIGMA_PERP_DIAM_MULT`, `CROSS_CHAIN_EXTENT`, `CROSS_CENTER_WEIGHT`, `CROSS_CHAIN_WEIGHT`, `CROSS_CLIP_TO_DNA_MASK` |
| Decoys | `DECOY_EVERY_N_SAMPLES` (every 50th sample), `DECOY_N_BEADS_RANGE_POOL` |

`BOND_LENGTH`, `PERSISTENCE_BONDS` and `K_ANGLE` act per bond, so changing
them changes the stiffness of the chain in nm and hence its shape; the
requested contour length is kept. Molecules are centred and never cropped: a
molecule that does not fit in the field of view is rejected and the next seed
is used. Re-run the cell after changing a variable.

**Decoys** are short fragments placed near a corner of the image. They appear
in the AFM image but not in the masks, so the network learns to ignore debris
next to the molecule.

### Noise (section 1.3)

* **Blank `.spm` scan** (`BLANK_SPM_PATH`): a scan of the substrate without
  DNA. The plane tilt and line offsets are removed, as TopoStats would do, and
  the rest is used as a noise texture, so the data looks as if it had been
  imaged on that set-up. Use a blank from your own AFM to match your data.
* **PSD noise model** (`MODEL_PATH`): used when no blank is available; built
  from an ensemble of blanks on nickel substrates. `PSD_NOISE_METHOD` is
  `"mean_full2d"` (default), `"lognormal_full2d"` or `"empirical_radial"`.

`TARGET_NOISE_RMS_NM` sets the noise amplitude of single images. Without
either file, no noise is added and a message says so.

### Dataset generation (section 8)

| Option | Meaning |
|---|---|
| `SAMPLE_TIMEOUT_S` | a seed is skipped if one sample takes longer than this (60 s) |
| `CONTOUR_LENGTH_RANGE_NM` | contour lengths (default `[115.26, 768.4, 768.4, 768.4]`: 25 % minicircles, 75 % plasmids) |
| `DNA_DIAMETER_RANGE_NM` | DNA diameter as seen by AFM, i.e. the rendered DNA **height** |
| `TIP_RADIUS_RANGE_NM` | AFM tip radius |
| `NOISE_RMS_RANGE_NM` | noise amplitude (RMS height) |
| `EXPORT_PREVIEW_PNGS`, `PREVIEW_PNG_INDICES` | PNG previews for these sample indices |

Each range accepts a **list** (one value picked per sample; repeat a value to
pick it more often), a **tuple** `(low, high)` (uniform draw) or a **single
number**. Every sample draws its values with a generator seeded by its own
seed, so a dataset is reproducible.

Samples are generated one at a time in a worker process. Seeds that fail
(time-out, rejected MD run, molecule too large for the field of view) are
skipped until `N_SAMPLES` samples exist.

## Part 2: segmentation

| Section | What it does |
|---|---|
| 9 Imports and settings | Dataset, network choice, preprocessing, training and loss settings. |
| 10 Data | Manifest → DeepTrack sources → lazy loading pipeline → data loaders. |
| 11 Loss and metrics | The two-task loss and the logged DNA IoU and Dice. |
| 12 The U-Net | Res-U-Net or simple U-Net, wrapped in a Deeplay `Regressor`. |
| 13 Training and validation | Loads a provided network (default) or trains a new one; training curves. |
| 14 Validation metrics | Loss, DNA Dice and IoU, crossing error, crossing-head check and previews. |
| 15 Real data | `.npy` or image files with masks, single files, or `.spm` scans through TopoStats. |

### Settings (section 9)

| Variable | Default | Meaning |
|---|---|---|
| `DATASET_DIR` | the dataset from Part 1 | folder with `manifest.csv` |
| `ARCHITECTURE` | `"residual"` | `"residual"` (Res-U-Net) or `"simple"` (plain U-Net) |
| `TRAIN_MODEL` | `False` | `False` loads `CHECKPOINT_PATH`; `True` trains on `DATASET_DIR` |
| `CHECKPOINT_PATH` | provided network for `USE_MD` and `ARCHITECTURE` | network to load |
| `TARGET_SIZE` | 512 | network input size (pixels) |
| `NORMALIZATION_BOUNDS_NM` | (-1, 5) | heights are clipped to this range and scaled to 0-1 |
| `CROP_PADDING_PX` | (20, 60) | training crops: molecule plus a random 20-60 px margin |
| `VAL_CROP_PADDING_PX` | 40 | validation and evaluation margin (`None` = full image) |
| `MODEL_CFG`, `TRAIN_CFG`, `LOSS_CFG` | as used for the provided networks | network, training (batch 4, 50 epochs, Adam 3 × 10⁻⁴) and loss settings |

### Data (section 10)

The manifest is read into rows of file paths and split into training and
validation sets (its `split` column if present, otherwise a random 80 : 20
split). Each set becomes a DeepTrack source, and the pipeline
`dt.Value(row) >> dt.Lambda(load_sample)` reads the arrays only when the
network asks for a sample:

1. load the height map, the DNA mask and the crossing mask (all zero if
   missing);
2. crop to the molecule (DNA-mask bounding box plus a margin), like the grain
   crops TopoStats gives the network on real data;
3. apply random flips and 90° rotations (training only);
4. normalise the heights over −1…5 nm;
5. resize to 512 × 512 keeping the aspect ratio, padding with the background
   value.

Images are `(1, 512, 512)`; targets are `(2, 512, 512)` with DNA in channel 0
and crossings in channel 1. Your own data works too: a folder with height maps
in nm, DNA masks and optional crossing masks, plus a `manifest.csv` with the
columns `image_npy`, `dna_mask_npy` and (optionally) `cross_mask_npy`, and an
optional `split` column.

### Loss and metrics (section 11)

* **DNA loss** = 0.4 × weighted BCE (DNA pixels weighted 1.5) + 0.6 × Dice
  loss + 0.5 × clDice loss. clDice
  ([Shit et al., CVPR 2021](https://arxiv.org/abs/2003.07311)) compares soft
  skeletons and rewards a connected centre line, which matters for tracing.
* **Crossing loss** = focal BCE with crossing pixels weighted by 1 + 20 × the
  target, plus penalties on crossing probability on ordinary DNA (weight 2.0)
  and on background (weight 0.14).
* **Total** = 3.0 × DNA loss + 0.02 × crossing loss.
* **Metrics**: DNA IoU and Dice at a 0.5 threshold, logged every epoch.

### Networks and checkpoints (sections 12-13)

* **Res-U-Net** (`ResUNetTwoHeads`, 7.6 M parameters): residual blocks with
  GroupNorm at four levels, a bottleneck, and separate DNA and crossing heads.
* **Simple U-Net** (`SimpleUNet`, 2.1 M parameters): one convolution per
  level and a two-channel output, layer for layer the same as Deeplay's
  `UNet2d(1, [32, 64, 128, 256], 2)`.

Four trained networks are provided in `segmentation/checkpoints/`:
`md_res_unet.pt`, `md_simple_unet.pt`, `non_md_res_unet.pt` and
`non_md_simple_unet.pt`. Each was trained on 800 simulated molecules of its
pathway for 50 epochs, keeping the epoch with the lowest validation loss
(validation Dice 0.88-0.90); see
[`segmentation/README.md`](segmentation/README.md#checkpoints) for the
details. A newly trained network is saved to
`checkpoints/my_<pathway>_<architecture>.pt`; the provided files are never
overwritten.

### Real data (section 15)

| `REAL_INPUT` | `REAL_DATA_TYPE` | What happens |
|---|---|---|
| `"dataset"` | `"npy"` or `"image"` | every image in `REAL_DATASET_DIR` is segmented; images with a matching `mask_<name>` are cropped to the molecule and scored (Dice, IoU, precision, recall, predicted vs true DNA length) |
| `"single"` | `"npy"` or `"image"` | `REAL_SINGLE_FILE` is segmented and its mask measured |
| `"single"` / `"dataset"` | `"spm"` | raw scans are analysed with TopoStats, using the U-Net for segmentation |

`.npy` files must contain heights in nm; image files are mapped from grey
values onto `IMAGE_HEIGHT_RANGE_NM`. `REAL_NM_PER_PIXEL` converts lengths to nm
(`.spm` files carry their own pixel size). The defaults analyse the example
scan in `common-assets/example-spm/`; `common-assets/real-test-data/` holds 11
real height maps with DNA masks (0.5 nm per pixel).

**TopoStats** filters the scan, finds grains above `grain_threshold_nm`,
re-segments every grain crop with the U-Net (pixels above `confidence` form the
mask), then skeletonises, resolves crossings and fits splines. The result is
the **contour length** of each molecule, whether it is circular, and the
crossings in each grain. Three presets in `segmentation/configs/` are tuned
for `relaxed` plasmids, `supercoiled` plasmids and `minicircle`s; the
notebook shows their most important settings so they can be changed. The
U-Net is connected to TopoStats for the current session only; nothing in the
installed TopoStats is changed. This needs **TopoStats 2.4.2**
(`pip install topostats==2.4.2`).

## Files created by the notebook

Next to the notebook:

```text
<OUT_DIR>/                         # dna_dataset_md or dna_dataset_non_md
├── images/img_XXXX.npy            # height maps (nm)
├── dna_masks/dna_XXXX.npy         # binary DNA masks
├── cross_masks/cross_XXXX.npy     # soft crossing masks
├── meta/meta_XXXX.npz             # coordinates and parameters of each sample
├── preview_pngs/                  # optional previews
└── manifest.csv                   # links them, with the settings of each sample

logs_unet/<pathway>_<architecture>/version_<N>/metrics.csv   # training logs (TRAIN_MODEL = True)
checkpoints/my_<pathway>_<architecture>.pt                    # newly trained network
topostats_output/                                             # TopoStats output for .spm scans
```

## Directory structure

```text
tutorials/dna-imaging/
├── README.md
├── segmentation/
│   ├── DNA_simulation_and_segmentation.ipynb
│   ├── dna_simulation_helpers.py        # every function used in Part 1
│   ├── unet_segmentation_helpers.py     # every function used in Part 2
│   ├── checkpoints/                     # MD / non-MD × Res-U-Net / simple U-Net
│   ├── configs/                         # TopoStats presets
│   ├── requirements.txt
│   └── README.md                        # set-up and checkpoint details
├── common-assets/                       # shared by the DNA tutorials
│   ├── noise/                           # blank .spm scan and PSD noise model
│   ├── example-images/                  # real molecules shown in section 5
│   ├── example-spm/                     # real relaxed-plasmid scan for TopoStats
│   └── real-test-data/                  # image_N / mask_N (.npy in nm, and .png)
└── simulated-data/                      # a few example simulated height maps
```

## Dependencies

NumPy, SciPy, Matplotlib, scikit-image, pandas, Pillow, PyYAML, psutil and
joblib; PyTorch, Lightning, torchmetrics, Deeplay 0.1.4 and DeepTrack 2.0.1
for Part 2; OpenMM for the MD pathway; TopoStats 2.4.2 for `.spm` scans. The
exact versions tested are listed in
[`segmentation/requirements.txt`](segmentation/requirements.txt).

## Tips

* **Your own molecules.** Change `CONTOUR_LENGTH_RANGE_NM` (and
  `CONTOUR_LENGTH_NM` for the single-molecule sections); widen the field of
  view with `NX`, `NY` or `NM_PER_PIXEL` if long molecules are rejected.
* **Your own instrument.** Point `BLANK_SPM_PATH` to a blank scan of your
  substrate and set `NOISE_RMS_RANGE_NM`, `TIP_RADIUS_RANGE_NM` and
  `DNA_DIAMETER_RANGE_NM` to what you measure.
* **Slow machines.** If the benchmark (section 7) shows more than about
  `SAMPLE_TIMEOUT_S` per molecule, for example with MD on a CPU or a busy
  GPU, raise `SAMPLE_TIMEOUT_S`; otherwise every seed is skipped.
* **Crossings.** If the crossing head lights up ordinary DNA, raise
  `cross_false_positive_weight`; if it predicts no crossings, raise
  `cross_pos_weight` or `cross_loss_weight` (in `LOSS_CFG`).
* **Same preprocessing everywhere.** The TopoStats `lower_norm_bound` and
  `upper_norm_bound` must match `NORMALIZATION_BOUNDS_NM` used in training.
* **Windows paths.** TopoStats writes files under `topostats_output/` with long
  names; keep the repository in a short path (well under 260 characters in
  total) or enable long paths in Windows.
