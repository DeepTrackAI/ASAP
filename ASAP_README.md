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

ASAP is a repository of tutorial notebooks for simulating, processing,
and analyzing atomic force microscopy (AFM) data. The notebooks combine
physics-inspired simulations, image and signal processing, and deep learning
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
generation or acquisition to downstream analysis, visualization, or quality
control.
* **Extensibility:** Make notebooks easy to adapt to new AFM datasets,
microscope settings, simulation parameters, and learning objectives.

# What ASAP provides

ASAP provides a flexible framework for building AFM-imaging and
AFM-data analysis pipelines. ASAP focuses on complete AFM-centered workflows that
can use DeepTrack-style ideas, but the emphasis is on applied notebooks for
specific AFM tasks such as DNA-chain and sample simulation and force-curve quality
control.

# ASAP compatibility with Deeplay

Deeplay provides reusable deep-learning components built on PyTorch and Lightning.

ASAP uses Deeplay in notebooks that train neural networks, but ASAP itself is not a
model library. Instead, it provides end-to-end examples that include data
simulation, preprocessing, training, evaluation, and visualization.

# Installation

ASAP currently does not require package installation. The user is

prompted to install the dependencies needed by the notebooks you want to run.

Some notebooks install their own dependencies inside notebook cells. Common
packages used across the tutorials include NumPy, SciPy, Matplotlib, Pillow,
DeepTrack, Deeplay, and PyTorch/Lightning. Notebooks that use molecular
dynamics may additionally require OpenMM.

# Notebooks

The following notebooks are available in the repository.

## Tutorials

* [**Simulation of DNA chains imaged via AFM**](tutorials/Simulation_of_DNA_chains_imaged_via_AFM.ipynb) <a href="https://colab.research.google.com/github/DeepTrackAI/ASAP/blob/main/tutorials/Simulation\_of\_DNA\_chains\_imaged\_via\_AFM.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

  Generates synthetic AFM-like height images of DNA chains relaxed on a
substrate. The notebook writes simulated images, DNA segmentation masks,
crossing masks, and metadata to disk. It includes configurable chain lengths,
AFM rendering parameters, mask generation, optional noise models, sanity
checks, benchmarking, and full dataset export.

* [**Simulation of DNA chains imaged via AFM - draft for Giovanni**](tutorials/Simulation_of_DNA_chains_imaged_via_AFM_draft_for_Giovanni%20%281%29.ipynb) <a href="https://colab.research.google.com/github/DeepTrackAI/ASAP/blob/main/tutorials/Simulation\_of\_DNA\_chains\_imaged\_via\_AFM\_draft\_for\_Giovanni%20(1).ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

  Draft variant of the DNA-chain AFM simulation workflow. It explores related
simulation settings, including molecular-dynamics-based relaxation,
AFM-rendering controls, noise options, and dataset generation for DNA masks
and crossing masks. Because it is marked as a draft, users should prefer the
main DNA simulation notebook unless they specifically need the experimental
settings in this version.

  ## Quality Control

* [**Quality Control of AFM Force Spectroscopy Data with Conditional Variational Autoencoders**](tutorials/quality-control/QC.ipynb) <a href="https://colab.research.google.com/github/DeepTrackAI/ASAP/blob/main/tutorials/quality-control/QC.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

  Trains a conditional variational autoencoder on AFM force spectroscopy
curves in a self-supervised manner. The notebook loads paired approach and
retraction curves from the 3T3 cell dataset, builds a latent representation
of force curves, and uses that representation to support quality control and
identification of good, bad, and unknown-quality curves.

  # Repository Structure

  ```text
ASAP/
├── assets/
├── tutorials/
│   ├── quality-control/
│   │   ├── QC.ipynb
│   │   └── cvae.py
│   ├── Simulation\_of\_DNA\_chains\_imaged\_via\_AFM.ipynb
│   ├── Simulation\_of\_DNA\_chains\_imaged\_via\_AFM\_draft\_for\_Giovanni (1).ipynb
│   └── psd\_noise\_model.npz
├── LICENSE
├── README.md
└── requirements.txt
```

  # Contributing

  Contributions are welcome. Useful contributions include:

* improving notebook documentation and cell organization,
* adding clearer installation instructions for specific platforms,
* adding new AFM simulation or analysis notebooks,
* improving dataset-loading and export utilities,
* adding tests or small validation datasets where appropriate.

  # License

  ASAP is distributed under the MIT License. See the `LICENSE` file for details.

  # Funding

  This project is funded by the MSCA SPM 4.0 Doctoral Network.

  Additionally, this project is part of the DeepTrackAI ecosystem.

  If you use ASAP togetherwith Deep TrackAI tools, please also consult the

  relevant DeepTrackAI project pages for citation and funding information.

