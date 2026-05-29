<h3 align="center">ASAP - AFM Simulation and Analysis Notebooks</h3>
<p align="center">
  <a href="/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="license">
  </a>
  <a href="https://github.com/DeepTrackAI/ASAP">
    <img src="https://img.shields.io/badge/GitHub-DeepTrackAI%2FASAP-blue?logo=github" alt="GitHub repository">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.x-blue" alt="Python version">
  </a>
  <a href="https://jupyter.org/">
    <img src="https://img.shields.io/badge/Jupyter-notebooks-orange?logo=jupyter" alt="Jupyter notebooks">
  </a>
</p>

ASAP is a repository of interactive notebooks for simulating, processing,
and analyzing atomic force microscopy (AFM) data. The notebooks combine
physics-inspired simulation, image and signal processing, and deep learning
workflows to support reproducible AFM experiments and model development.

The repository is organized around executable Jupyter notebooks rather than a
single installable Python package. Each notebook is intended to be readable,
modifiable, and runnable as a complete workflow.

# Core Philosophy

ASAP is designed around notebook-based scientific workflows. The main goals
are:

* **Reproducibility:** Keep simulation settings, preprocessing steps, model
  parameters, and outputs visible in a single executable document.
* **Practical AFM workflows:** Provide complete examples that connect AFM data
  generation or loading to downstream analysis, visualization, or quality
  control.
* **DeepTrackAI integration:** Use tools from the DeepTrackAI ecosystem where
  they are helpful for data handling, model construction, and training.
* **Extensibility:** Make notebooks easy to adapt to new AFM datasets,
  microscope settings, simulation parameters, and learning objectives.

# ASAP and DeepTrack

DeepTrack provides a flexible framework for building image-generation and
image-analysis pipelines. ASAP focuses on complete AFM-centered workflows that
can use DeepTrack-style ideas, but the emphasis is on applied notebooks for
specific AFM tasks such as DNA-chain simulation, force-curve simulation, and
force-spectroscopy quality control.

# ASAP and Deeplay

Deeplay provides reusable deep-learning components built on PyTorch. ASAP can
use Deeplay in notebooks that train neural networks, but ASAP itself is not a
model library. Instead, it provides end-to-end examples that may include data
simulation, preprocessing, training, evaluation, and visualization.

# Installation

ASAP currently does not define a single package installation workflow. Each
notebook highlights the dependencies needed for that workflow and, where
appropriate, includes installation cells for those packages. Users should follow
the dependency notes in the notebook they want to run.

Common packages used across the tutorials include NumPy, SciPy, Matplotlib,
Pillow, Pandas, DeepTrack, Deeplay, PyTorch, and Lightning. Notebooks that use
molecular dynamics may additionally require OpenMM.

# Tutorials

The tutorials are grouped into three main folders under `ASAP/tutorials/`.

## Simulation of DNA chains

The DNA-chain tutorial provides a complete pipeline for generating synthetic
AFM images of DNA chains, creating DNA and crossing masks, training a U-Net
segmentation model, and evaluating the model on both simulated validation data
and real test data.

The workflow includes:

* closed DNA-chain generation with optional molecular-dynamics relaxation;
* AFM-like height-map rendering with configurable tip, blur, ridge, taper, and
  crossing-height parameters;
* DNA segmentation masks and crossing heatmaps generated from the simulated
  chain geometry;
* dataset export to `.npy` files plus `manifest.csv` metadata;
* DeepTrack source records and lazy loading through a DeepTrack pipeline;
* Deeplay U-Net training for DNA segmentation and crossing prediction;
* validation and optional real-test evaluation.

Main notebook:

* [**Simulation of DNA chain with U-Net**](tutorials/Simulation%20of%20DNA%20chains/Notebooks/Simulation_of_DNA_chain_with_U_Net.ipynb)

## AFM data quality control

The AFM data quality-control tutorial trains a conditional variational
autoencoder on AFM force spectroscopy curves. It uses paired approach and
retraction curves from the 3T3 cell dataset and builds a latent representation
for identifying good, bad, and unknown-quality force curves.

Main notebook:

* [**Quality Control of AFM Force Spectroscopy Data with Conditional
  Variational Autoencoders**](tutorials/quality-control/QC.ipynb)

Supporting files include the CVAE implementation, pretrained or saved model
artifacts, and the 3T3 cell dataset folder.

## AFM simulation and regression model

The AFM simulation and regression tutorial provides a complete workflow for
simulating AFM force-distance curves, training a deep regression model to
extract mechanical properties, and validating the model on real experimental
data.

The workflow includes:

* synthetic force-map generation from configurable cantilever, contact-mechanics,
  and sample parameters;
* preprocessing by normalization, cropping, and resampling;
* a two-head regression model that predicts contact point and Young's modulus;
* synthetic-map evaluation with scatter plots and interactive maps;
* real-data evaluation on the 3T3 fibroblast dataset using iterative
  contact-point patching and Hertz-fit RMSE comparisons.

Main notebook:

* **AFM Simulation and Regression Model**

# Repository Structure

The tutorial directories are organized as follows:

```text
ASAP/
├── tutorials/
│   ├── Simulation of DNA chains/
│   │   ├── Notebooks/
│   │   │   └── Simulation_of_DNA_chain_with_U_Net.ipynb
│   │   ├── Noise assets/
│   │   │   ├── psd_noise_model.npz
│   │   │   └── 20240411_blank_water.0_00000.spm
│   │   ├── Simulated data/
│   │   │   └── dna_dataset_100_4lengths_MD/
│   │   │       ├── images/
│   │   │       │   ├── img_0000.npy
│   │   │       │   └── ...
│   │   │       ├── dna_masks/
│   │   │       │   ├── dna_0000.npy
│   │   │       │   └── ...
│   │   │       ├── cross_masks/
│   │   │       │   ├── cross_0000.npy
│   │   │       │   └── ...
│   │   │       ├── meta/
│   │   │       │   ├── meta_0000.npz
│   │   │       │   └── ...
│   │   │       └── manifest.csv
│   │   ├── Test data/
│   │   │   ├── images/
│   │   │   │   ├── img_00.npy
│   │   │   │   └── ...
│   │   │   └── masks/
│   │   │       ├── mask_00.npy
│   │   │       └── ...
│   │   └── DNA_simulation_with_UNet_README.md
│   │
│   ├── quality-control/
│   │   ├── 3t3_cell_dataset/
│   │   ├── models/
│   │   ├── QC.ipynb
│   │   └── cvae.py
│   │
│   └── simu-inference/
│       ├── Part_1_AFM_simulation.ipynb
│       ├── plotting.py
│       ├── results/
│       ├── logs/
│       │   └── regressor/
│       └── README.md
├── LICENSE
├── README.md
└── requirements.txt
```

# Dependencies

The notebooks install or import the packages they need. Typical dependencies
include:

| Package | Purpose |
|---|---|
| `numpy`, `scipy` | Numerical simulation and image/signal processing |
| `matplotlib` | Static visualization and prediction previews |
| `pandas` | Manifest handling and CSV metric parsing |
| `pillow` | Loading real image files such as PNG and TIFF |
| `torch`, `lightning` | Model backend and training loop |
| `deeplay` | Reusable deep-learning models and applications |
| `deeptrack` | Source and lazy pipeline handling |
| `torchmetrics` | Metric logging for segmentation models |
| `joblib` | Cross-platform timeout handling |
| `openmm` | Optional molecular-dynamics relaxation |
| `plotly`, `ipywidgets` | Interactive map and curve viewers |
| `scikit-learn` | Train/test splitting and regression metrics |

Install the common deep-learning and notebook dependencies with:

```bash
pip install deeptrack deeplay lightning pandas pillow torchmetrics joblib
```

Install OpenMM only if molecular dynamics is needed:

```bash
pip install openmm[cuda12]
```

# Contributions

Contributions are welcome. Useful contributions include:

* Improving notebook documentation and cell organization.
* Adding clearer installation instructions for specific platforms.
* Adding new AFM simulation or analysis notebooks.
* Improving dataset-loading and export utilities.
* Adding tests or small validation datasets where appropriate.

# License

ASAP is distributed under the MIT License. See the `LICENSE` file for details.

# Funding

This project is funded by the MSCA SPM 4.0 Doctoral Network.

Additionally, this project is part of the DeepTrackAI ecosystem. If you use
ASAP together with DeepTrack tools, please also consult the relevant
DeepTrackAI project pages for citation and funding information.
