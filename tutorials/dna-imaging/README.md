# DNA Chain AFM Simulation and U-Net Segmentation

The **Simulation of DNA chain with U-Net** notebook provides a complete
pipeline for generating synthetic atomic force microscopy (AFM) images of DNA
chains, creating DNA and crossing masks, training a U-Net segmentation model,
and evaluating the model on both simulated validation data and real test data.

The notebook is designed to be customizable and easy to navigate. Users can
adjust the physical simulation parameters, AFM rendering settings, mask
parameters, U-Net architecture, training configuration, and evaluation inputs
from clearly defined configuration cells.

---

## Overview

The tutorial consists of 7 main stages:

**Simulate chains** -> DNA chains are generated as closed bead-spring rings,
either with molecular-dynamics relaxation or with a faster non-MD persistent
walk.

**Render AFM images** -> Relaxed chain coordinates are converted into
AFM-like height maps using a height-based rendering pipeline with configurable
tip, blur, taper, ridge, and crossing-boost parameters.

**Create masks** -> DNA segmentation masks and crossing masks are generated
from the same simulated chain geometry.

**Generate a dataset** -> Images, masks, crossing masks, metadata, and a
manifest file are written to disk. Slow or failing seeds are skipped
automatically using a cross-platform timeout.

**Connect to DeepTrack sources** -> The manifest is converted into
DeepTrack-backed source records. Images and masks are loaded lazily from disk
through a DeepTrack pipeline.

**Train a Deeplay U-Net** -> A two-channel U-Net predicts DNA segmentation and
crossing heatmaps. The DNA channel is trained with BCE-Dice loss, while the
crossing channel is trained with weighted MSE.

**Evaluate and visualize** -> The trained model is evaluated on the validation
set and optionally on a separate real test dataset. Prediction previews show the
input image, DNA target, DNA prediction, crossing target, and crossing
prediction.

**Outputs**:

- `OUT_DIR/images/` — simulated AFM height maps as `.npy` files
- `OUT_DIR/dna_masks/` — DNA segmentation masks as `.npy` files
- `OUT_DIR/cross_masks/` — crossing masks or heatmaps as `.npy` files
- `OUT_DIR/meta/` — per-sample metadata as compressed `.npz` files
- `OUT_DIR/manifest.csv` — index of generated images, masks, metadata, and
  simulation settings
- `logs/train_<N>_epochs/` — Deeplay/Lightning CSV logs for each training run

---

## Notebook sections

### 1. Optional molecular dynamics and imports

The notebook can run with or without molecular dynamics. When molecular
dynamics is enabled, OpenMM is used to relax the DNA chain on a substrate.
When molecular dynamics is disabled, a faster 2-D persistent random-walk chain
is used instead.

Users can control this behavior with:

| Parameter | Description |
|---|---|
| `USE_MD` | Enables or disables OpenMM molecular-dynamics relaxation |

When running on machines without CUDA, OpenMM should be installed in a
platform-independent way and allowed to fall back to OpenCL or CPU.

### 2. Global simulation configuration

This section defines the main parameters controlling dataset size, chain
geometry, AFM image size, rendering behavior, DNA masks, and crossing masks.
These are the primary knobs for changing the generated dataset.

| Group | Key parameters |
|---|---|
| Dataset | `OUT_DIR`, `N_SAMPLES`, `BASE_SEED` |
| Image size | `NX`, `NY` |
| Chain geometry | `N_BEADS`, `BEAD_COUNTS`, `BOND_LENGTH`, `PERSISTENCE_BONDS` |
| MD relaxation | `K_ANGLE`, `BASE_Z`, `N_FRAMES`, `STEPS_PER_FRAME` |
| Non-MD generation | `ANGLE_STIFNESS_MULT` |
| AFM rendering | `tip_radius`, `nm_per_px`, `AFM_KW` |
| DNA mask | `DNA_MASK_DILATE_PX` |
| Crossing mask | `CROSS_MIN_SEP_BEADS`, `CROSS_SIGMA_CENTER_PX`, `CROSS_SIGMA_PERP_PX`, `CROSS_CHAIN_EXTENT`, `CROSS_CENTER_WEIGHT`, `CROSS_CHAIN_WEIGHT`, `CROSS_CLIP_TO_DNA_MASK` |

Changing these parameters changes the appearance and the training data.
For example, increasing chain length or adding decoys can change what the model learns 
about the DNA and un-connected fragments, while increasing DNA-mask dilation can make masks thicker
and applicable to bigger tip sizes. 

### 3. Noise configuration

The notebook supports adding realistic AFM-like background noise. Noise can be
sampled from a stored noise assets which has been made using an enseble of blank measurements
or can be uploaded by the users themselves if they have access to blank measurements of the 
substrate without the DNA sample, thus incorporating the noise from the instrunment into their 
generated data.

This allows users to move from idealized synthetic images toward images that
more closely resemble experimental AFM data.

### 4. Chain creation

The function `make_tangled_ring_initial` creates a closed 3-D polymer ring
using a persistent random-walk tangent model. It controls the initial topology
of the DNA chain before relaxation or rendering.

Important parameters include:

- bead count
- bond length
- persistence length in bead units
- initial height above the substrate
- random seed

### 5. Molecular-dynamics relaxation

When `USE_MD = True`, `run_md_relaxation` uses OpenMM to relax the initial
chain. This can produce more physically realistic chain conformations, but it
is slower and may fail for some seeds.

The dataset-generation cell uses a cross-platform process timeout so that seeds
that take too long or cause OpenMM errors are skipped automatically.

### 6. Non-MD chain generation

When `USE_MD = False`, the notebook uses `make_ring_2d_persistent_initial` to
create a faster 2-D chain. This mode is useful for rapid testing or generating
larger datasets without the computational cost of molecular dynamics.

### 7. MD animation

The function `show_md_animation` visualizes molecular-dynamics frames in the
notebook. This section is useful for checking whether relaxation behaves as
expected before generating a full dataset.

### 8. AFM rendering

The height-based AFM rendering section converts chain coordinates into an
AFM-like height map. It includes geometry helpers, tip-convolution-inspired
rendering, crossing-height boosting, optional tapering, center-ridge addition,
and final smoothing.

Key AFM parameters include:

| Parameter | Effect |
|---|---|
| `dna_diameter_nm` | Controls the physical DNA diameter used by the renderer |
| `tip_radius_nm` | Controls apparent broadening from the AFM tip |
| `target_nm_per_px` | Controls physical sampling before resizing |
| `max_height_nm` | Controls height scaling and clipping |
| `final_blur_sigma_px` | Controls final image smoothness |
| `add_center_ridge` | Adds a ridge along the DNA centerline |
| `ridge_sigma_nm` | Controls ridge width |
| `ridge_amp_nm` | Controls ridge height |
| `enable_crossing_boost` | Increases height near detected crossings |
| `guaranteed_offset_nm` | Controls crossing-height boost strength |

### 9. Noise functions

Noise helper functions parse and sample AFM-like background noise. These
functions allow simulated images to include experimentally motivated background
texture rather than only clean rendered chains.

### 10. DNA ground-truth mask

The DNA mask section converts chain coordinates into a binary segmentation
mask. The mask is used as channel 0 of the U-Net target.

The main user-facing control is:

| Parameter | Effect |
|---|---|
| `DNA_MASK_DILATE_PX` | Expands the DNA mask by a fixed number of pixels |

Larger dilation can improve training stability but may produce masks that are
less precise around the DNA boundary.

### 11. Crossing detection and crossing mask

A crossing occurs when 2 DNA segments of the chain overlap with each other.
Crossings are detected from polyline intersections between non-neighboring DNA
segments. Crossing masks are then generated as localized heatmaps around those
intersections.

The crossing target is used as channel 1 of the U-Net target.

Important crossing controls include:

| Parameter | Effect |
|---|---|
| `CROSS_MIN_SEP_BEADS` | Minimum bead separation for valid self-crossings |
| `CROSS_SIGMA_CENTER_PX` | Width of the central crossing peak |
| `CROSS_SIGMA_PERP_PX` | Width perpendicular to the crossing direction |
| `CROSS_CHAIN_EXTENT` | Length of the chain-aligned crossing mask support |
| `CROSS_CENTER_WEIGHT` | Weight of the crossing center |
| `CROSS_CHAIN_WEIGHT` | Weight of the chain-aligned part of the crossing mask |
| `CROSS_CLIP_TO_DNA_MASK` | Restricts crossing masks to DNA-positive pixels |

### 12. Decoy chain functions

The notebook can add short decoy DNA-like fragments to the AFM image. These
fragments are composited into the image but excluded from the DNA and crossing
masks. This teaches the model not to blindly label every DNA-like object as
part of the main chain.

### 13. Chain generation pipeline

The functions `generate_chain_with_beads`, `render_chain_and_masks`, and
`generate_one_sample_multilength` form the complete single-sample simulation
pipeline.

For each sample, the pipeline:

1. Generates a chain with a chosen seed and bead count.
2. Optionally relaxes it with molecular dynamics.
3. Detects crossings.
4. Renders the AFM image.
5. Creates the DNA mask.
6. Creates the crossing mask.
7. Adds optional decoy and noise components.
8. Returns a sample dictionary containing image, masks, metadata, and crossing
   count.

### 14. Sanity check

The sanity-check section generates a small set of samples and visualizes the
resulting AFM images, DNA masks, and crossing masks. This should be run before
large dataset generation to confirm that the simulation parameters are
reasonable.

### 15. Benchmarking

The benchmarking section measures runtime, memory usage, and system details for
a small number of generated samples. This helps estimate the time needed to
produce a full dataset, especially when molecular dynamics is enabled.

### 16. Dataset generation

The dataset-generation cell writes the final simulated dataset to disk. It
creates a `manifest.csv` file that stores paths to each image, DNA mask,
crossing mask, metadata file, seed, bead count, crossing count, and AFM
parameters.

The cell uses a cross-platform timeout so that problematic seeds are skipped.
This avoids relying on Linux-only `SIGALRM` behavior and makes the notebook
usable on Windows and macOS.

Key outputs are:

- `images/img_XXXX.npy`
- `dna_masks/dna_XXXX.npy`
- `cross_masks/cross_XXXX.npy`
- `meta/meta_XXXX.npz`
- `manifest.csv`

### 17. DeepTrack source connection

The U-Net section begins by reading `manifest.csv` and converting its rows into
DeepTrack source records. The source stores file paths, not loaded arrays. This
keeps memory use low and allows the dataset to be loaded lazily.

The manifest columns used are:

| Column | Purpose |
|---|---|
| `image_npy` | Path to the AFM image array |
| `dna_mask_npy` | Path to the DNA segmentation mask |
| `cross_mask_npy` | Path to the crossing mask |

A train/validation split is created using `VAL_FRACTION` and `SEED`.

### 18. Lazy loading and preprocessing pipeline

The function `build_pipeline` uses `dt.Value(source.samples)` and
`dt.Lambda(...)` to load each sample from disk. The preprocessing pipeline:

1. Loads the image and masks.
2. Normalizes the AFM image using robust percentiles.
3. Converts the DNA mask to binary form.
4. Loads the crossing mask.
5. Optionally applies geometric and intensity augmentation.
6. Resizes and pads all arrays to `TARGET_SIZE`.
7. Stacks DNA and crossing targets into a two-channel target array.

The final dataset returns:

- image tensor: `(1, TARGET_SIZE, TARGET_SIZE)`
- target tensor: `(2, TARGET_SIZE, TARGET_SIZE)`

where target channel 0 is DNA and target channel 1 is crossings.

### 19. Deeplay U-Net model

The segmentation model is built with Deeplay using `dl.UNet2d`. The model has
one input channel and two output channels.

| Channel | Meaning |
|---|---|
| input channel 0 | normalized AFM image |
| output channel 0 | DNA segmentation logits |
| output channel 1 | crossing logits |

The default model configuration is:

```python
MODEL_CFG = dict(
    in_channels=1,
    channels=[16, 32, 64, 128],
    out_channels=2,
)
```

Increasing `channels` gives the U-Net more capacity but increases memory use
and training time.

### 20. Multitask loss and DNA Dice metric

The notebook uses a custom `DNACrossingLoss`:

- DNA segmentation is trained with BCE-Dice loss.
- Crossings are trained with weighted MSE after sigmoid activation.
- `cross_pos_weight` increases the importance of rare positive crossing pixels.
- `cross_loss_weight` controls how much the crossing task contributes to the
  total loss.

The DNA Dice metric is implemented as a TorchMetrics metric and logged during
training.

Important training and loss parameters are:

| Parameter | Effect |
|---|---|
| `batch_size` | Number of samples per training batch |
| `max_epochs` | Number of training epochs |
| `lr` | Optimizer learning rate |
| `dna_pos_weight` | Increases penalty for missing DNA pixels |
| `dna_loss_weight` | Global weight on DNA loss |
| `cross_loss_weight` | Global weight on crossing loss |
| `cross_pos_weight` | Local positive-pixel weight for crossing targets |
| `dice_smooth` | Stabilizes Dice loss and Dice metric |

### 21. Training and logging

Training is performed with a Deeplay/Lightning trainer. The notebook selects
the accelerator automatically and prints the detected platform.

The trainer uses:

- CUDA with mixed precision when a CUDA GPU is available
- Apple Silicon MPS or CPU with full precision otherwise
Logging:
- CSV logging to `logs/train_<max_epochs>_epochs/version_<N>/metrics.csv`

The training-curve plotting function automatically reads the metrics file from
the most recent trainer logger.

### 22. Validation metrics

The notebook evaluates the trained model on the validation set using:

| Metric | Description |
|---|---|
| `loss` | Full multitask validation loss |
| `dna_dice` | Hard-threshold DNA Dice score |
| `dna_iou` | Hard-threshold DNA intersection-over-union |
| `cross_mse` | Crossing heatmap mean-squared error |

DNA Dice and DNA IoU are the most important metrics if the main goal is DNA
segmentation.

### 23. Prediction preview

The prediction preview shows:

1. input AFM image
2. DNA target
3. DNA prediction
4. crossing target
5. crossing prediction

All panels are plotted with fixed `vmin=0` and `vmax=1` so that weak crossing
predictions are not visually exaggerated by automatic contrast scaling.

### 24. Real-data evaluation

The final section evaluates the trained model on a separate real test dataset.
The user can either provide an existing test-data root or a zip file. If the
root does not exist and the zip file does, the notebook extracts the zip file
and builds the test root.

The real-data loader recursively searches for supported image and mask files:

- `.png`
- `.jpg`
- `.jpeg`
- `.tif`
- `.tiff`
- `.npy`

Images and masks are matched by normalized file stem. Crossing masks are
optional; if no crossing mask is found, a zero crossing target is used. In that
case, real-test DNA Dice and DNA IoU are meaningful, while crossing MSE should
be interpreted carefully.

---

## Directory structure

The directory is typically structured like this, with 

```text
ASAP/
└── tutorials/
    └── Simulation of DNA chains 
        └── Notebooks
        |   └── Simulation_of_DNA_chain_with_U_Net.ipynb
        |   └── logs/
        |       └── train_<max_epochs>_epochs/
        |           └── version_<N>/
        |               └── metrics.csv
        |
        └── Noise assets
        |   └── psd_noise_model.npz
        |   └── 20240411_blank_water.0_00000.spm
        |
        └── Simulated data
        |      dna_dataset_100_4lengths_MD/
        |      ├── images/
        |      |   ├── img_0000.npy
        |      |   └── ...
        |      ├── dna_masks/
        |      |   ├── dna_0000.npy
        |      |   └── ...
        |      ├── cross_masks/
        |      |   ├── cross_0000.npy
        |      |   └── ...
        |      ├── meta/
        |      |   ├── meta_0000.npz
        |      |   └── ...
        |      └── manifest.csv
        |
        └── Test data
        |   ├── images/
        |   |   ├── img_00.npy
        |   |   └── ...
        |   └── masks/
        |       ├── mask_00.npy
        |       └── ...
        |   
        └── DNA_simulation_with_UNet_README.md
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy`, `scipy` | Numerical simulation and image processing |
| `matplotlib` | Static plotting and prediction previews |
| `pillow` | Loading real image files such as PNG and TIFF |
| `pandas` | Reading the manifest and CSV metrics |
| `torch` | Tensor operations and model backend |
| `lightning` | Training loop and logging backend |
| `deeplay` | U-Net model and training application wrapper |
| `deeptrack` | Source and lazy pipeline connection |
| `torchmetrics` | DNA Dice metric logging |
| `joblib` | Cross-platform timeout for sample generation |
| `openmm` | Optional molecular-dynamics relaxation |
| `psutil` | Benchmarking and memory reporting |

Install the common dependencies with:

```bash
pip install deeptrack deeplay lightning pandas pillow torchmetrics joblib
```

Install OpenMM only if molecular dynamics is needed:

```bash
pip install openmm
```

---

## Quick start

1. Choose whether to use molecular dynamics by setting `USE_MD`.
2. Adjust the simulation parameters in the global settings cell, especially
   `N_SAMPLES`, `BEAD_COUNTS`, `AFM_KW`, and the crossing-mask parameters.
3. Run the sanity-check section and inspect the generated images and masks.
4. Run the benchmarking section if using molecular dynamics or large datasets.
5. Run the dataset-generation cell to write `.npy` files and `manifest.csv`.
6. Run the DeepTrack source section to build train and validation sources.
7. Train the Deeplay U-Net.
8. Plot training metrics and evaluate on the validation set.
9. Optionally set `TEST_ROOT` or `TEST_ZIP_PATH` and evaluate on real test data.

---

## Suggested tuning workflow

For DNA segmentation quality, start with:

1. `BG_Q` and `HIGH_Q` in `normalize_afm`
2. `dna_pos_weight`
3. `TARGET_SIZE`
4. U-Net `channels`
5. number of generated samples

For crossing prediction quality, tune:

1. `CROSS_SIGMA_CENTER_PX`
2. `CROSS_SIGMA_PERP_PX`
3. `CROSS_CHAIN_EXTENT`
4. `cross_pos_weight`
5. `cross_loss_weight`

If crossing predictions light up ordinary DNA segments, reduce
`cross_pos_weight` or add additional penalties for false crossing predictions
on non-crossing DNA. If crossing predictions collapse to zero, increase
`cross_pos_weight` or `cross_loss_weight`.

---

## Notes

- Keep the same preprocessing for training, validation, and testing.
- Use fixed visualization scaling for prediction panels to avoid misleading
  crossing heatmaps.
- If real test data has no crossing masks, focus primarily on DNA Dice and DNA
  IoU rather than crossing MSE.
