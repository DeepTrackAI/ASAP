# AFM Simulation and Regression Model

AFM Simulation and Regression Model notebook provides a complete pipeline for simulating Atomic Force Microscopy (AFM) force-distance curves, training a deep regression model to extract mechanical properties from those curves, and validating the model on real experimental data. The tutorial is designed to be customazible, and easy to navigate.

---

## Overview

The tutorial consists of 5 stages: 

**Simulate** -> Synthetic AFM force maps are generated from configurable cantilever, contact-mechanics, and sample parameters

**Preprocess** -> Curves are normalised, randomly cropped, and resampled to a fixed length to form the training set

**Train** -> A two-head MLP predicts contact point and Young's modulus jointly via PyTorch Lightning

**Evaluate (synthetic)** -> The best checkpoint is tested on a held-out simulated map with scatter-plot and interactive map visualisations

**Evaluate (real)** -> The model is applied to the 3T3 fibroblast dataset using iterative contact-point patching, with Hertz-fit RMSE as the quality metric



**Outputs**:
- `results/` — example map images, force-curve plots and experimental data analysis
- `logs/regressor/` — Lightning CSVLogger metrics and a trained model (one versioned directory per run)

---

## Notebook sections

### 1. Optional installations
Installs `deeplay`, `plotly`, `ipywidgets`, and `ipympl`. Uncomment the pip line when running on Colab or Kaggle.

### 2. Imports and configuration
Imports core libraries and defines the three configuration dataclasses that control the entire pipeline. **These are the primary knobs the user needs to adjust.**

| Dataclass | Key fields |
|---|---|
| `SimulationConfig` | Grid size `(nx, ny)`, pixel size, z-length, z-speed, sampling frequency, setpoint force range, tip shape, noise type, curve mode (`approach` / `approach_retract`), artefact flag |
| `DatasetConfig` | Number of training samples, topography types, substrate/feature stiffness ranges, height range |
| `ModelConfig` | Batch size, learning rate, max epochs, contact-point loss weight |

### 3. Cantilever model with PSD — `AFMCantilever`
Models the physical cantilever (spring constant, resonance frequency, quality factor, temperature). Generates thermal noise via one of three methods:
- **`rms`** — Equipartition theorem; requires only $k$ and $T$
- **`theoretical`** — Damped harmonic oscillator PSD; requires $k$, $f_0$, $Q$, $T$
- **`experimental`** — Sampled from a `.tnd` calibration file recorded during the actual experiment

### 4. Contact mechanics — `HertzContact`
Converts indentation depth to contact force for **spherical** and **pyramidal** tip geometries. Supports an optional **bi-layer** mode where a stiffer substrate is reached beyond a configurable transition depth, enabling artefact simulation.

### 5. Synthetic sample-map generator — `SampleGenerator`
Generates correlated 2D maps of topography, stiffness, and contact point. Two topography modes are available:
- **`gaussian_features`** — Random Gaussian bumps on a flat substrate
- **`hemisphere`** — A single hemispherical feature centred on the scan area

Contact-point values are derived from the topography: taller pixels correspond to an earlier contact (smaller z-value at first touch).

### 6. AFM simulator — `AFMSimulator`
Combines cantilever, contact model, and sample generator into a vectorised per-pixel simulator. For each pixel it:
1. Computes the approach force curve using the Hertz model
2. Terminates the approach at the setpoint force
3. (Optionally) appends a retraction segment
4. Adds thermal noise drawn from the PSD

### 7. Plotting utilities
Three helper functions used throughout the notebook:
- `plot_maps` — side-by-side topography / stiffness / contact-point maps
- `plot_force_curve` — single force-distance curve with approach and retract segments
- `plot_psd` — log-log displacement PSD with resonance frequency marker

### 8. Instantiate the simulator — example sample
Instantiates the cantilever, contact model, sample generator, and simulator from the config objects. Generates one example sample, plots the three maps, an example force curve, and the noise PSD.

### 9. Curve generation and preprocessing
Loops over `n_samples` synthetic samples. For each curve, 10 augmented variants are created by randomly cropping the pre-contact and post-contact portions of the displacement axis, then resampling to a fixed length of 1 500 points. The input feature vector is the horizontal stack `[displacement (µm), force (nN)]` of length 3 000. The target vector is `[normalised contact point, normalised stiffness]`. A 70/30 train/validation split is created and wrapped in `DataLoader`s.

### 10. MLP training

**Model — `MLP`**  
A two-head network implemented as a `LightningModule`:
- Shared encoder: `Linear(3000→1024) → ReLU → Linear(1024→512) → ReLU`
- Contact-point head: `Linear(512→256) → ReLU → Linear(256→1)`
- Stiffness head: `Linear(513→256) → ReLU → Linear(256→1)` — receives the encoder output concatenated with the contact-point prediction, enforcing sequential prediction

**Trainer**  
PyTorch Lightning `Trainer` (via `deeplay`) with:
- `ModelCheckpoint` saving the top-3 checkpoints by `val_loss`
- `CSVLogger` logging `train_loss` (per step) and `val_loss` (per epoch) to `logs/regressor/version_<N>/metrics.csv`
- Up to 1 000 epochs, auto device selection

A loss curve plot is generated from the CSV log after training.

### 11. Evaluate on the generated map
Loads the best Lightning checkpoint and runs inference on the full example simulation map. Results are shown as:
- Scatter plots of predicted vs. true contact point and stiffness
- An **interactive clickable map** (requires `plotly` + `ipywidgets`) showing the predicted stiffness map alongside the ground truth; clicking any pixel overlays the measured curve with the ML-predicted and ground-truth Hertz fits

### 12. Evaluate on the real data
Loads the [3T3 fibroblast dataset](https://github.com/DeepTrackAI/3t3_cell_dataset) (approach + retraction curves, Young's modulus labels, quality labels).

**Iterative contact-point patching**: the model first predicts a coarse contact point, then the leading pre-contact portion of each curve is progressively trimmed at eight cut fractions (0 %, 10 %, …, 70 %) and predictions are averaged. This makes the stiffness estimate robust to pre-contact noise.

Evaluation outputs:
- Contact-point scatter plot (good-quality curves, `label == 1`)
- Stiffness scatter plot (filtered to < 5 kPa)
- RMSE on contact point and stiffness 
- **Interactive Plotly slider** showing raw experimental curves sorted by ML reconstruction loss, overlaid with ML and regular Hertz fit computed with MSE fitting
- Histogram + scatter comparison of per-curve RMSE for ML-predicted parameters vs. reference (data-processing) parameters

---

## Directory structure

```
ASAP/
├── tutorials/    
│   └── simu-inference
│       └── notebooks/
│           ├── Part_1_AFM_simulation_final.ipynb   ← this notebook                    
│           ├── thermal-noise-data_vDeflection_*.tnd
│           ├── plotting.py
|           ├── results/                     # saved figures
|           └── README.md
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy`, `scipy` | Numerical simulation and interpolation |
| `torch`, `lightning` | Model definition and training |
| `deeplay` | Lightning `Trainer` wrapper |
| `scikit-learn` | Train/test split and RMSE metrics |
| `matplotlib` | Static plots |
| `plotly`, `ipywidgets`, `anywidget` | Interactive map and curve viewers |
| `pandas` | CSV metrics parsing |

Install all at once:
```bash
pip install deeplay plotly ipywidgets anywidget ipympl scikit-learn
```

---

## Quick start

1. Set your experimental parameters in `SimulationConfig` (cell 10) — at minimum `tip_shape`, `noise_type`, and `experimental_noise_path`.
2. Set the stiffness and height ranges expected from your sample in `DatasetConfig`.
3. Run all cells top to bottom.
4. After the real-data analysis, save the output and if the user wants to do a quality control or anomaly detection advance to section 2.

To load a pretrained model instead of training from scratch, uncomment the `pretraiend_model_path` block in Section 11.

