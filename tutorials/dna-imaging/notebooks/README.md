# DNA Chain AFM Simulation and ResU-Net Workflow

This repository contains a Jupyter notebook for generating simulated AFM height maps of DNA chains and training a two-output ResU-Net model to segment DNA and identify crossing regions.

Notebook: `Simulation_of_DNA_chains_and_Res_U_Net.ipynb`

The workflow has two main parts:

1. **DNA chain AFM simulation**: creates synthetic AFM images, DNA masks, crossing masks, and metadata.
2. **ResU-Net training/evaluation**: loads generated samples, trains a two-channel segmentation model, validates it, and can evaluate on real AFM data.

The model output channels are:

| Channel | Target |
|---:|---|
| 0 | DNA segmentation |
| 1 | DNA crossing / overlap regions |

---

## Current simulation design

The notebook separates the physical simulation/rendering concepts:

| Concept | Current variable | Meaning |
|---|---|---|
| Final image size | `NX`, `NY` | Controls the output array shape. |
| Internal AFM render grid | `chain_resolution_nm` | Controls the internal nm/pixel sampling used to rasterize and render the chain. |
| DNA physical width | `AFM_KW["dna_diameter_nm"]` | Controls rendered DNA diameter and physical mask widths. |
| Tip size | `tip_radius_nm`, `AFM_KW["tip_radius_nm"]` | Physical AFM tip radius in nm. Tip convolution is always applied near the end of rendering. |
| DNA target mask width | `DNA_MASK_DIAMETER_NM` | Physical nm width used for DNA supervision masks. |
| Crossing target width | `CROSS_SIGMA_CENTER_DIAM_MULT`, `CROSS_SIGMA_PERP_DIAM_MULT` | Crossing mask width is derived from DNA diameter at final output resolution. |

This means that changing `chain_resolution_nm` should change only the internal rendering fidelity and runtime, not the final target mask size. Changing `NX`/`NY` changes the final array size. Changing `dna_diameter_nm` changes the physical DNA width used by the renderer and masks.

---

## Important current parameter values

### Dataset generation

| Parameter | Current value | Meaning |
|---|---:|---|
| `OUT_DIR` | `dna_dataset_unnormalized_crossing_fixed_1000_4lengths_MD` | Dataset output directory. |
| `N_SAMPLES` | `1000` | Number of valid generated samples. |
| `BASE_SEED` | `1` | First random seed. |
| `SAMPLE_TIMEOUT_S` | `10.0` | Maximum time allowed for one sample before skipping the seed. |
| `EXPORT_PREVIEW_PNGS` | `False` | Whether PNG previews are exported during dataset generation. |

### Image size and render resolution

| Parameter | Current value | Meaning |
|---|---:|---|
| `NX` | `512` | Final image width in pixels. |
| `NY` | `512` | Final image height in pixels. |
| `TARGET_SIZE` | `int(NX)` | U-Net input/target size. |
| `chain_resolution_nm` | `0.5` | Internal AFM render-grid sampling in nm/pixel. |

### Chain geometry and relaxation

| Parameter | Current value | Meaning |
|---|---:|---|
| `N_BEADS` | `90` | Default bead count for visual tests. |
| `BEAD_COUNTS` | `[70, 80, 90, 100]` | Bead counts cycled through dataset generation. |
| `BOND_LENGTH` | `1.0` | Distance between neighbouring beads in the chain model. |
| `PERSISTENCE_BONDS` | `23.0` | Persistence length measured in bead bonds. |
| `K_ANGLE` | `12.0` | Angular stiffness during MD relaxation. |
| `BASE_Z` | `5.0` | Initial chain height above substrate. |
| `ANGLE_STIFNESS_MULT` | `0.4` | Angle stiffness multiplier in non-MD mode. |
| `N_FRAMES` | `200` | Number of MD frames recorded. |
| `STEPS_PER_FRAME` | `200` | MD steps per recorded frame. |

---

## AFM rendering parameters

The main AFM rendering dictionary is `AFM_KW`.

| Parameter | Current value | Meaning |
|---|---:|---|
| `AFM_KW["dna_diameter_nm"]` | `1.5` | Physical DNA diameter in nm. |
| `AFM_KW["tip_radius_nm"]` | `0.5` | Physical AFM tip radius in nm. |
| `AFM_KW["max_height_nm"]` | `6.0` | Height clipping limit. |
| `AFM_KW["chain_resolution_nm"]` | `0.5` | Internal render-grid resolution. |
| `AFM_KW["max_radius_px"]` | `128` | Max structuring-element radius on the internal render grid. |
| `AFM_KW["final_blur_sigma_px"]` | `0.32` | Gaussian blur before tip convolution. |
| `AFM_KW["apply_edge_taper"]` | `True` | Applies DNA edge taper. |
| `AFM_KW["taper_sigma_nm"]` | `0.35` | Physical width of edge taper. |
| `AFM_KW["taper_floor"]` | `0.10` | Minimum edge taper multiplier. |
| `AFM_KW["add_center_ridge"]` | `False` | Whether to add a center ridge along DNA. |
| `AFM_KW["ridge_sigma_nm"]` | `0.16` | Ridge width, if enabled. |
| `AFM_KW["ridge_amp_nm"]` | `0.04` | Ridge amplitude, if enabled. |
| `AFM_KW["grain_nm"]` | `0.0` | Extra DNA-local grain amplitude. |
| `AFM_KW["grain_sigma_px"]` | `0.6` | DNA grain smoothing scale. |

### Edge taper logic at crossings

The edge taper rule is:

```text
Apply edge taper to DNA normally, including the bottom crossing strand,
but skip tapering where the top-strand crossing footprint is present.
```

In practice:

```python
edge_taper_region = dna_body_region & ~top_cross_region
```

This prevents plus-shaped crossing artifact while still giving tapered DNA edges on the bottom strand outside the top-strand overlap.

### Tip convolution

Tip convolution is always applied near the end of `create_z_based_afm`. It broadens the apparent AFM footprint according to `tip_radius_nm`. The DNA target mask is not based on this apparent tip-broadened region.

---

## Crossing rendering and mask parameters

| Parameter | Current value | Meaning |
|---|---:|---|
| `AFM_KW["enable_crossing_boost"]` | `True` | Makes top crossing regions taller. |
| `AFM_KW["min_separation_beads"]` | `12` | Minimum bead separation for image-level crossing boost. |
| `AFM_KW["boost_window_beads"]` | `1` | Number of beads around crossing center boosted. |
| `AFM_KW["guaranteed_offset_nm"]` | `0.25` | Minimum height offset at top crossing. |
| `AFM_KW["boost_method"]` | `"absolute"` | Crossing boost rule. |
| `AFM_KW["boost_profile"]` | `"gaussian"` | Crossing boost taper profile. |
| `AFM_KW["boost_sigma_beads"]` | `0.4` | Width of boost profile in beads. |
| `AFM_KW["far_clip_nm"]` | `2.5` | Optional non-crossing height clip. |
| `AFM_KW["far_clip_window_beads"]` | `3` | Window excluded from far clipping. |
| `CROSS_MIN_SEP_BEADS` | `8` | Minimum bead separation for crossing mask generation. |
| `CROSS_SIGMA_CENTER_DIAM_MULT` | `0.5` | Crossing-center sigma as multiple of DNA mask diameter. |
| `CROSS_SIGMA_PERP_DIAM_MULT` | `0.25` | Perpendicular sigma as multiple of DNA mask diameter. |
| `CROSS_CHAIN_EXTENT` | `2.0` | Extent of chain-aligned crossing mask. |
| `CROSS_CENTER_WEIGHT` | `0.75` | Weight of central crossing peak. |
| `CROSS_CHAIN_WEIGHT` | `0.5` | Weight of chain-aligned crossing target. |
| `CROSS_CLIP_TO_DNA_MASK` | `False` | Whether crossing masks are clipped to DNA masks. |

`CROSS_SIGMA_CENTER_PX` and `CROSS_SIGMA_PERP_PX` remain as fallback values, but the active workflow computes crossing-mask sigma from DNA diameter and final output pixel size.

---

## DNA mask parameters

| Parameter | Current value | Meaning |
|---|---:|---|
| `DNA_MASK_DIAMETER_NM` | `0.5 * AFM_KW["dna_diameter_nm"]` | Physical width used to generate the DNA supervision mask. |

The DNA target mask is generated from bead coordinates and the chosen physical mask diameter at final output resolution. It is intentionally independent of tip radius and internal render resolution.

---

## Noise parameters

| Parameter | Current value | Meaning |
|---|---:|---|
| `TARGET_NOISE_RMS_NM` | notebook setting | Target RMS amplitude for added background noise. |
| `USE_BLANK_SPM_NOISE` | notebook setting | Use a blank SPM scan when available. |
| `USE_PSD_NOISE_FALLBACK` | notebook setting | Use PSD-based synthetic noise if blank SPM noise is unavailable. |
| `PSD_NOISE_METHOD` | notebook setting | PSD noise generation mode. |

Noise is added after the rendered AFM image is produced. This keeps image rendering and background noise generation separate.

---

## Generated dataset structure

A generated dataset contains:

```text
OUT_DIR/
  images/       # AFM height maps, .npy
  dna_masks/    # DNA target masks, .npy
  cross_masks/  # crossing target masks, .npy
  meta/         # per-sample metadata, .npz
  manifest.csv  # dataset index and paths
```

Each generated sample returns and/or stores:

| Key | Meaning |
|---|---|
| `afm_img` | Final AFM height map. |
| `dna_mask` | DNA segmentation target. |
| `cross_mask` | Crossing target. |
| `extent` | Physical field of view `(xmin, xmax, ymin, ymax)`. |
| `n_crossings` | Number of crossing targets. |
| `decoy_coords` | Optional decoy fragment coordinates. |
| `chain_resolution_nm` | Effective internal render resolution. |
| `output_nm_per_px` | Effective final output physical scale. |
| `tip_radius_nm` | Physical tip radius. |
| `dna_diameter_nm` | Physical DNA diameter. |

---

## Training configuration

| Parameter | Current value | Meaning |
|---|---:|---|
| `SEED` | `5` | Training/data split seed. |
| `VAL_FRACTION` | `0.20` | Validation fraction. |
| `BG_Q` | `25` | Background percentile for AFM normalization. |
| `HIGH_Q` | `99.98` | Upper percentile for AFM normalization. |
| `MODEL_CFG["channels"]` | `[32, 64, 128, 256]` | ResU-Net channel widths. |
| `TRAIN_CFG["batch_size"]` | `4` | Batch size. |
| `TRAIN_CFG["num_workers"]` | `2` | DataLoader workers. |
| `TRAIN_CFG["max_epochs"]` | `50` | Maximum training epochs. |
| `TRAIN_CFG["lr"]` | `3e-4` | Learning rate. |
| `TRAIN_CFG["dna_pos_weight"]` | `1.6` | DNA positive-pixel BCE weight. |
| `TRAIN_CFG["cross_loss_weight"]` | `0.25` | Overall crossing-head loss weight. |
| `TRAIN_CFG["cross_pos_weight"]` | `50.0` | Crossing positive-pixel weight. |
| `TRAIN_CFG["cross_false_positive_weight"]` | `1.0` | Extra penalty for crossing false positives. |
| `TRAIN_CFG["background_false_positive_weight"]` | `0.02` | Extra penalty for predicting on background. |

---

## Recommended tuning order

For improving real-data transfer, tune in this order:

1. AFM height statistics: `dna_diameter_nm`, `tip_radius_nm`, `final_blur_sigma_px`, `TARGET_NOISE_RMS_NM`.
2. Mask geometry: `DNA_MASK_DIAMETER_NM`, `CROSS_SIGMA_CENTER_DIAM_MULT`, `CROSS_SIGMA_PERP_DIAM_MULT`.
3. Crossing visibility: `guaranteed_offset_nm`, `boost_window_beads`, `boost_sigma_beads`.
4. Normalization: `BG_Q`, `HIGH_Q`.
5. Loss weights: `cross_loss_weight`, `cross_pos_weight`, `dna_pos_weight`.
6. Model capacity: `MODEL_CFG["channels"]`.
7. Dataset size and bead-count distribution: `N_SAMPLES`, `BEAD_COUNTS`.

---

## Beads, base pairs, and calibration

The notebook simulates DNA as a bead chain. A bead count is a coarse-grained contour discretization and is not automatically equal to a base-pair count.

A useful physical approximation for B-form dsDNA is:

```text
1 bp ≈ 0.34 nm contour length
1 helical turn ≈ 10.5 bp
canonical dsDNA persistence length ≈ 50 nm
```

If real labelled masks with known base-pair counts are available, use their skeleton length to estimate an empirical pixel-to-bp or bead-to-bp mapping. This calibration is only valid for masks generated at the same image resolution, magnification, and preprocessing pipeline.


---

## Notes on portability

- `USE_MD=True` requires OpenMM and can be slower or platform-specific.
- If OpenMM is unavailable, use the non-MD chain generator.
- `TRAIN_CFG["num_workers"] = 0` is safest for Windows/macOS notebooks; the current notebook uses `2` for faster loading where supported.
- Real test-data discovery depends on file and folder naming. Update the image/mask keyword tuples if no samples are found.
