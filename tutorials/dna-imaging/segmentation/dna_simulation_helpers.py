"""Functions for Part 1 of DNA_simulation_and_segmentation.ipynb.

Part 1 of the notebook simulates closed DNA molecules on a substrate and
renders them as AFM height maps with matching DNA and crossing masks.
This module holds every function used by Part 1 and the benchmark memory
tracker class. The learning-model classes are defined in the notebook.

The notebook defines its variables in ordinary cells and hands them to
this module with ``configure(globals())``. Only the names listed in
``SETTING_NAMES`` are read. The module-level values are the defaults
that apply before ``configure`` is called; they are the optimised MD
settings.

Key Features
------------
- **Two simulation pathways**

    With molecular dynamics (annealed coarse-grained OpenMM force field)
    or without (calibrated 2-D persistent ring). The optimised settings
    of both pathways produced the training data of the provided
    networks.

- **Contour length as the main control**

    A requested contour length is turned into a bead count and a bond
    length so that the chain has exactly (non-MD) or very nearly (MD)
    the requested length.

- **Realistic AFM rendering**

    Height-based rendering with a DNA cross-section, edge taper,
    raised crossings, blur, tip convolution, grain and noise from a
    blank scan or a power-spectral-density model.

- **Training targets**

    A DNA mask of fixed width and a soft crossing mask for every image;
    decoy fragments appear in the image but not in the masks.

- **Dataset generation**

    Values drawn per sample from numbers, lists or ranges, a time-out
    per sample in a worker process, PNG previews and a ``manifest.csv``.

Module Structure
----------------
Functions:

- `configure(namespace, **overrides) -> dict[str, Any]`

    Copy notebook variables into this module.

- `current_settings() -> dict[str, Any]`

    Return a copy of the active settings.

- `contour_to_beads(contour_length_nm, ...) -> tuple[int, float]`

    Convert a contour length into a bead count and bond length.

- `closed_contour_nm(coords, use_z) -> float`

    Contour length of a closed chain.

- `closed_lengths(coords) -> dict[str, float]`

    Measured 3-D and projected contour and bond statistics.

- `min_nonbonded_distance(coords) -> float`

    Closest approach of non-neighbouring beads.

- `make_tangled_ring_initial_safe(seed, n_beads, ...) -> np.ndarray`

    Initial 3-D ring for the MD pathway.

- `plot_initial_chain(coords, title) -> None`

    3-D schematic of a chain above the substrate.

- `test_openmm_installation() -> None`

    Run OpenMM's installation test.

- `run_md_relaxation(coords0, seed, ...) -> list[np.ndarray]`

    Relax a ring onto the substrate with OpenMM.

- `make_ring_2d_persistent_initial(seed, n_beads, ...) -> list`

    Closed 2-D persistent ring.

- `make_non_md_ring(seed, contour_length_nm) -> np.ndarray`

    Non-MD ring with exactly the requested contour.

- `plot_relaxed_chain(coords, title) -> None`

    A chain on the substrate, in 3-D and from above.

- `show_md_animation(frames, interval_ms) -> None`

    HTML animation of MD frames.

- `create_z_based_afm(coords, ...) -> tuple[np.ndarray, dict]`

    Render an AFM height image of a chain (with the rendering helpers
    `dna_shape`, `AFM_tip`, `boost_crossings_intersections`,
    `collect_intersections`, `merge_crossing_clusters`,
    `calculate_crossing_taper_weight`, `compute_internal_render_grid`,
    `resize_image_to_shape`, `upsample_nn` and `disk_footprint`).

- `load_blank_spm_noise(blank_path, ...) -> tuple`

    Extract a noise texture from a blank scan (with the helpers
    `parse_blocks`, `read_block_candidates`,
    `choose_best_height_image`, `robust_range`,
    `looks_like_afm_height`, `maybe_to_nm`, `plane_remove`,
    `line_flatten_median`, `feature_mask`, `inpaint_simple`,
    `bandpass_noise` and `extract_topostats_like_noise`).

- `load_model(path) -> dict[str, np.ndarray]`

    Load the PSD noise model.

- `generate_noise(seed, ...) -> np.ndarray`

    Synthesise noise from the PSD model.

- `sample_noise_image(noise_source, noise_asset, ...) -> np.ndarray`

    Noise image for one sample (with `random_transform_noise_texture`
    and `tile_or_crop`).

- `load_noise_asset(verbose) -> tuple[str, Any, dict]`

    Load the noise source chosen in the notebook.

- `make_dna_mask_from_beads(coords, extent, nx, ny, ...) -> np.ndarray`

    DNA mask of fixed width (with `nm_to_px` and
    `rasterize_closed_polyline_mask`).

- `find_polyline_crossings(coords_xy, ...) -> list[dict]`

    Crossings of a chain seen from above (with `canon_axis`).

- `make_crossing_mask(coords, extent, nx, ny, ...) -> tuple`

    Soft crossing mask (with `add_crossing_patch`).

- `make_decoy_chain(seed, ...) -> np.ndarray`

    Short decoy fragment.

- `place_decoy_in_corner(decoy_coords, global_extent, ...) -> np.ndarray`

    Move a decoy into an image corner.

- `generate_chain(seed, contour_length_nm, ...) -> dict[str, Any]`

    One relaxed ring of the requested contour length.

- `render_chain_and_masks(chain, ...) -> dict[str, Any]`

    AFM image, DNA mask and crossing mask of a chain.

- `generate_one_sample(seed, contour_length_nm, ...) -> dict[str, Any]`

    Chain, image and masks in one call.

- `generate_sanity_samples(seeds, contour_lengths_nm, ...) -> list`

    Samples for the sanity check.

- `plot_sanity_samples(samples) -> None`

    Sanity-check figure with scale bars (with `add_scale_bar` and
    `nice_scale_bar_nm`).

- `bench_stages(seed, contour_length_nm) -> dict[str, float]`

    Time the stages of one sample.

- `benchmark_pipeline(seed, contour_length_nm, ...) -> dict[str, Any]`

    Benchmark and run-time projection.

- `draw_value(spec, rng) -> float`

    Draw a value from a number, list or range (with `describe_spec`).

- `generate_dataset(contour_length_range_nm, ...) -> str`

    Generate the dataset and its manifest (with
    `get_sample_executor`, `reset_sample_executor`,
    `generate_sample_worker`, `generate_sample_with_timeout` and
    `save_preview_png`).

Attributes:

- `SETTING_NAMES`: tuple[str, ...]

    Names that `configure` copies from the notebook.

- `AFM_KW`: dict[str, Any]

    Keyword arguments of `create_z_based_afm` used for every sample.

- `USE_MD`, `NX`, `NY`, `NM_PER_PIXEL`, `CONTOUR_LENGTH_NM`, ...

    The notebook variables (see the notebook for their meaning).

Examples
--------
>>> import dna_simulation_helpers as sim

Generate one non-MD minicircle without noise:

>>> _ = sim.configure(USE_MD=False, BOND_LENGTH=7.362098159496696)
>>> sample = sim.generate_one_sample(1, 115.26)
>>> sample["afm_img"].shape, round(sample["projected_contour_nm"], 2)
((512, 512), 115.26)
>>> _ = sim.configure(USE_MD=True, BOND_LENGTH=4.5)

"""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import statistics
import threading
import subprocess
import sys
import time
from collections.abc import Iterable
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import psutil
from IPython.display import HTML, display
from matplotlib import animation
from scipy.ndimage import (
    affine_transform,
    binary_dilation,
    distance_transform_edt,
    gaussian_filter,
    grey_dilation,
    median_filter,
    zoom,
)

# A setting given as one number, a list of discrete values or a
# (low, high) range; see draw_value().
RangeSpec = float | list[float] | tuple[float, float]

try:  # OpenMM is only needed for the MD pathway.
    import openmm
    import openmm.unit as unit
except ImportError:  # pragma: no cover - depends on the environment
    openmm = None
    unit = None


# ────────────────────────────────────────────────────────────────────────────
# Default settings (overwritten by configure())
# ────────────────────────────────────────────────────────────────────────────

USE_MD = True

# Dataset
OUT_DIR = "dna_dataset"
N_SAMPLES = 1000
BASE_SEED = 1

# Image
NX = 512
NY = 512
NM_PER_PIXEL = 1.0
CONTOUR_LENGTH_NM = 768.4
CHAIN_RESOLUTION_NM = 0.5
TIP_RADIUS_NM = 7.0

# Chain (both pathways)
BOND_LENGTH = 4.5
PERSISTENCE_BONDS = 40.0
K_ANGLE = 15

# MD pathway
BASE_Z = 55.0
N_FRAMES = 350
STEPS_PER_FRAME = 300

# Non-MD pathway
ANGLE_STIFFNESS_MULT = 1.3329571644033593
NON_MD_ANGLE_LIMIT_RAD = 2.2072170954704875
NON_MD_Z_STD_NM = 0.0
NON_MD_BASE_Z_NM = 0.2

# Minimum bead counts used when a contour length is converted to beads
MD_MIN_BEADS = 8
NON_MD_MIN_BEADS = 12

# Rendering (optimised MD width/taper renderer)
DNA_HEIGHT_NM = 2.3
AFM_KW = dict(
    nx=NX,
    ny=NY,
    dna_diameter_nm=4.47,
    dna_height_nm=DNA_HEIGHT_NM,
    taper_width_nm=2.2,
    tip_radius_nm=TIP_RADIUS_NM,
    max_height_nm=5.0,
    chain_resolution_nm=CHAIN_RESOLUTION_NM,
    max_radius_px=128,
    radius_shrink_px=0.0,
    final_blur_sigma_px=0.22,
    apply_edge_taper=True,
    taper_sigma_nm=2.7,
    taper_floor=0.05,
    add_center_ridge=False,
    ridge_sigma_nm=0.16,
    ridge_amp_nm=0.04,
    grain_nm=0.0,
    grain_sigma_px=0.6,
    enable_crossing_boost=True,
    min_separation_beads=12,
    boost_window_beads=1,
    guaranteed_offset_nm=0.25,
    boost_method="absolute",
    boost_profile="gaussian",
    boost_sigma_beads=0.4,
    far_clip_nm=2.5,
    far_clip_window_beads=3,
    return_crossing_info=True,
    return_masks=True,
)

# Masks
DNA_MASK_DIAMETER_NM = 4.47
CROSS_MIN_SEP_BEADS = 3
CROSS_SIGMA_CENTER_PX = 2.0
CROSS_SIGMA_PERP_PX = 1.4
CROSS_SIGMA_CENTER_DIAM_MULT = 1.0
CROSS_SIGMA_PERP_DIAM_MULT = 0.5
CROSS_CHAIN_EXTENT = 1.0
CROSS_CENTER_WEIGHT = 0.75
CROSS_CHAIN_WEIGHT = 0.5
CROSS_CLIP_TO_DNA_MASK = False

# Decoy fragments
DECOY_EVERY_N_SAMPLES = 50
DECOY_N_BEADS_RANGE_POOL = [(2, 3), (3, 5), (4, 6)]

# Noise
TARGET_NOISE_RMS_NM = 0.10
USE_BLANK_SPM_NOISE = True
USE_PLANE_REMOVE = True
USE_LINE_FLATTEN = True
USE_BANDPASS_FILTER = True
BLANK_SPM_PATH = "../common-assets/noise/20240411_blank_water.0_00000.spm"
USE_PSD_NOISE_FALLBACK = True
MODEL_PATH = "../common-assets/noise/psd_noise_model.npz"
PSD_NOISE_METHOD = "mean_full2d"
PSD_NOISE_STD_SCALE = 2.0
NOISE_NATIVE_NM_PER_PIXEL = 0.5
TARGET_SHAPE = (512, 512)  # default shape for generate_noise()
EPS = 1e-12

# Dataset generation
SAMPLE_TIMEOUT_S = 60.0

SETTING_NAMES = (
    "USE_MD",
    "OUT_DIR",
    "N_SAMPLES",
    "BASE_SEED",
    "NX",
    "NY",
    "NM_PER_PIXEL",
    "CONTOUR_LENGTH_NM",
    "CHAIN_RESOLUTION_NM",
    "TIP_RADIUS_NM",
    "BOND_LENGTH",
    "PERSISTENCE_BONDS",
    "K_ANGLE",
    "BASE_Z",
    "N_FRAMES",
    "STEPS_PER_FRAME",
    "ANGLE_STIFFNESS_MULT",
    "NON_MD_ANGLE_LIMIT_RAD",
    "NON_MD_Z_STD_NM",
    "NON_MD_BASE_Z_NM",
    "MD_MIN_BEADS",
    "NON_MD_MIN_BEADS",
    "DNA_HEIGHT_NM",
    "AFM_KW",
    "DNA_MASK_DIAMETER_NM",
    "CROSS_MIN_SEP_BEADS",
    "CROSS_SIGMA_CENTER_PX",
    "CROSS_SIGMA_PERP_PX",
    "CROSS_SIGMA_CENTER_DIAM_MULT",
    "CROSS_SIGMA_PERP_DIAM_MULT",
    "CROSS_CHAIN_EXTENT",
    "CROSS_CENTER_WEIGHT",
    "CROSS_CHAIN_WEIGHT",
    "CROSS_CLIP_TO_DNA_MASK",
    "DECOY_EVERY_N_SAMPLES",
    "DECOY_N_BEADS_RANGE_POOL",
    "TARGET_NOISE_RMS_NM",
    "USE_BLANK_SPM_NOISE",
    "USE_PLANE_REMOVE",
    "USE_LINE_FLATTEN",
    "USE_BANDPASS_FILTER",
    "BLANK_SPM_PATH",
    "USE_PSD_NOISE_FALLBACK",
    "MODEL_PATH",
    "PSD_NOISE_METHOD",
    "PSD_NOISE_STD_SCALE",
    "NOISE_NATIVE_NM_PER_PIXEL",
    "SAMPLE_TIMEOUT_S",
)


def configure(
    namespace: dict[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Copy notebook variables into this module.

    The notebook calls ``configure(globals())`` after its variable cells.
    Only names in ``SETTING_NAMES`` are taken; dictionaries and lists are
    copied so later edits in the notebook do not change the module behind
    its back (call ``configure`` again after changing a variable).
    ``AFM_KW["nx"]`` and ``AFM_KW["ny"]`` are always set to ``NX`` and
    ``NY``.

    Parameters
    ----------
    namespace: dict[str, Any] | None, optional
        Usually ``globals()`` of the notebook. Defaults to `None`.
    **overrides: Any
        Individual settings, applied after ``namespace``.

    Returns
    -------
    dict[str, Any]
        The settings that were applied.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    Apply two settings and ignore a name the module does not know:

    >>> applied = sim.configure({"NX": 256, "NY": 256, "SOMETHING": 1})
    >>> sorted(applied)
    ['NX', 'NY']
    >>> sim.AFM_KW["nx"], sim.AFM_KW["ny"]
    (256, 256)
    >>> _ = sim.configure(NX=512, NY=512)

    """

    module = globals()
    source = dict(namespace or {})
    source.update(overrides)
    applied = {}
    for name in SETTING_NAMES:
        if name not in source:
            continue
        value = source[name]
        if isinstance(value, dict):
            value = dict(value)
        elif isinstance(value, list):
            value = list(value)
        module[name] = value
        applied[name] = value
    # The renderer always draws at the configured image size.
    module["AFM_KW"] = dict(
        module["AFM_KW"], nx=int(module["NX"]), ny=int(module["NY"])
    )
    return applied


def current_settings() -> dict[str, Any]:
    """Return a copy of the active settings.

    The dictionary is picklable and is sent to the worker process that
    generates the dataset, which applies it with :func:`configure`.

    Returns
    -------
    dict[str, Any]
        Setting name to value, for every name in ``SETTING_NAMES``.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> settings = sim.current_settings()
    >>> settings["NX"], "AFM_KW" in settings
    (512, True)

    """

    module = globals()
    return {name: module[name] for name in SETTING_NAMES}


# ────────────────────────────────────────────────────────────────────────────
# Contour length → beads
# ────────────────────────────────────────────────────────────────────────────


def contour_to_beads(
    contour_length_nm: float,
    bond_length_nm: float | None = None,
    min_beads: int | None = None,
) -> tuple[int, float]:
    """Convert a contour length into a bead count and bond length.

    The chain is a closed ring of ``n`` beads joined by ``n`` bonds, so its
    contour length is ``n * bond``. The bead count is the requested length
    divided by the preferred bond length, rounded to the nearest integer,
    and the bond is then stretched or shrunk slightly so that ``n * bond``
    equals the requested length exactly. Asking for 1 nm more therefore
    gives a chain that is 1 nm longer, while the coarse-graining stays
    close to the preferred bond length.

    Parameters
    ----------
    contour_length_nm: float
        Requested contour length in nm.
    bond_length_nm: float | None, optional
        Preferred bond length in nm. Defaults to `None`, which uses
        ``BOND_LENGTH``.
    min_beads: int | None, optional
        Smallest allowed bead count. Defaults to `None`, which uses
        ``MD_MIN_BEADS`` for the MD pathway and ``NON_MD_MIN_BEADS``
        otherwise.

    Returns
    -------
    tuple[int, float]
        The bead count and the actual bond length in nm.

    Raises
    ------
    ValueError
        If the contour length or the bond length is not finite and
        positive.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    A 440 nm ring with a preferred 4.5 nm bond:

    >>> sim.contour_to_beads(440.0, 4.5, 8)
    (98, 4.489795918367347)

    One nanometre more keeps the bead count and lengthens the bond:

    >>> n, bond = sim.contour_to_beads(441.0, 4.5, 8)
    >>> n, round(n * bond, 6)
    (98, 441.0)

    """

    contour = float(contour_length_nm)
    bond = float(BOND_LENGTH if bond_length_nm is None else bond_length_nm)
    if not (
        np.isfinite(contour)
        and contour > 0
        and np.isfinite(bond)
        and bond > 0
    ):
        raise ValueError(
            "contour_length_nm and bond_length_nm must be finite and positive"
        )
    if min_beads is None:
        min_beads = MD_MIN_BEADS if USE_MD else NON_MD_MIN_BEADS
    n_beads = max(int(min_beads), int(round(contour / bond)))
    return n_beads, contour / n_beads


def closed_contour_nm(coords: np.ndarray, use_z: bool = False) -> float:
    """Return the contour length of a closed chain.

    The length includes the closing bond from the last bead back to the
    first one.

    Parameters
    ----------
    coords: np.ndarray
        Bead coordinates with shape ``(N, 2)`` or ``(N, 3)`` in nm.
    use_z: bool, optional
        Include the z coordinate (3-D length). Defaults to `False`, which
        measures the projected (x, y) length seen from above.

    Returns
    -------
    float
        Contour length in nm.

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    A unit square has a closed contour of 4:

    >>> square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> sim.closed_contour_nm(square)
    4.0

    """

    xyz = np.asarray(coords, dtype=np.float64)
    xyz = xyz[:, :3] if use_z else xyz[:, :2]
    return float(np.linalg.norm(np.roll(xyz, -1, axis=0) - xyz, axis=1).sum())


def closed_lengths(coords: np.ndarray) -> dict[str, float]:
    """Measure the 3-D and projected contour of a closed chain.

    These measured values are recorded separately from the nominal
    contour (number of beads times bond length), together with bond
    statistics.

    Parameters
    ----------
    coords: np.ndarray
        Bead coordinates with shape ``(N, 3)`` in nm, ``N >= 3``.

    Returns
    -------
    dict[str, float]
        ``actual_3d_contour_nm``, ``projected_contour_nm``,
        ``min_bond_nm``, ``max_bond_nm``, ``mean_bond_nm`` and
        ``std_bond_nm``.

    Raises
    ------
    ValueError
        If ``coords`` is not an ``(N, 3)`` array with ``N >= 3`` or contains
        non-finite values.

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    A square of side 1 lifted along z by 0, 1, 0, 1:

    >>> coords = np.array(
    ...     [[0, 0, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=float
    ... )
    >>> lengths = sim.closed_lengths(coords)
    >>> round(lengths["projected_contour_nm"], 3)
    4.0
    >>> round(lengths["actual_3d_contour_nm"], 3)
    5.657

    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) < 3:
        raise ValueError("coords must be an (N,3) array with N>=3")
    if not np.all(np.isfinite(coords)):
        raise ValueError("non-finite coordinates")
    increments = np.roll(coords, -1, axis=0) - coords
    bonds = np.linalg.norm(increments, axis=1)
    return dict(
        actual_3d_contour_nm=float(bonds.sum()),
        projected_contour_nm=float(
            np.linalg.norm(increments[:, :2], axis=1).sum()
        ),
        min_bond_nm=float(bonds.min()),
        max_bond_nm=float(bonds.max()),
        mean_bond_nm=float(bonds.mean()),
        std_bond_nm=float(bonds.std()),
    )


def min_nonbonded_distance(coords: np.ndarray) -> float:
    """Return the smallest distance between non-neighbouring beads.

    Beads that are directly bonded (circular index separation below 2)
    are ignored, so the value measures how close the chain comes to
    itself.

    Parameters
    ----------
    coords: np.ndarray
        Bead coordinates with shape ``(N, 3)`` in nm.

    Returns
    -------
    float
        Minimum distance in nm between beads at least two bonds apart.

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    >>> square = np.array(
    ...     [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float
    ... )
    >>> round(sim.min_nonbonded_distance(square), 4)
    1.4142

    """

    coords = np.asarray(coords, dtype=np.float64)
    n = len(coords)
    distances = np.linalg.norm(
        coords[:, None, :] - coords[None, :, :], axis=2
    )
    indices = np.arange(n)
    separation = np.abs(indices[:, None] - indices[None, :])
    separation = np.minimum(separation, n - separation)
    distances[separation < 2] = np.inf
    return float(distances.min())


def _resample_closed(coords: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed polyline at ``n`` equal arc-length steps.

    Parameters
    ----------
    coords: np.ndarray
        Vertices of the closed curve with shape ``(M, D)``.
    n: int
        Number of output points.

    Returns
    -------
    np.ndarray
        ``n`` points with shape ``(n, D)``, equally spaced along the closed
        curve and starting at the first vertex.

    Raises
    ------
    ValueError
        If a segment of the curve has zero or non-finite length.

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    Resample a unit square to eight points:

    >>> square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    >>> sim._resample_closed(square, 8)[:3]
    array([[0. , 0. ],
           [0.5, 0. ],
           [1. , 0. ]])

    """

    closed = np.vstack([coords, coords[0]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 1e-12):
        raise ValueError("degenerate periodic bridge")
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    distances = np.linspace(0.0, cumulative[-1], n, endpoint=False)
    segments = np.searchsorted(cumulative, distances, side="right") - 1
    fractions = (distances - cumulative[segments]) / lengths[segments]
    return closed[segments] + fractions[:, None] * (
        closed[segments + 1] - closed[segments]
    )


# ────────────────────────────────────────────────────────────────────────────
# Chain creation (3-D persistent random walk, MD pathway)
# ────────────────────────────────────────────────────────────────────────────


def make_tangled_ring_initial_safe(
    seed: int,
    n_beads: int,
    bond_length: float,
    persistence_bonds: float = 40.0,
    base_z: float = 55.0,
    *,
    max_attempts: int = 16,
    min_bond_fraction: float = 0.5,
    min_nonbonded_fraction: float = 0.5,
) -> np.ndarray:
    """Build the initial 3-D ring for the MD pathway.

    A persistent random walk of ``n_beads`` steps of ``bond_length`` is
    drawn: every step turns the direction by a small random angle with
    variance ``1 / persistence_bonds`` about a random axis. The step
    vectors are made to sum to zero (a periodic bridge), so the walk closes
    on itself, the beads are respaced evenly along the closed curve, the
    ring is scaled to the nominal contour ``n_beads * bond_length`` and
    lifted to ``base_z`` above the substrate.

    A draw with a very short or very long bond, or two non-neighbouring
    beads closer than ``min_nonbonded_fraction * bond_length``, is
    rejected and redrawn (up to ``max_attempts`` times). The MD force
    field itself is not changed.

    Parameters
    ----------
    seed: int
        Non-negative random seed.
    n_beads: int
        Number of beads, at least 8.
    bond_length: float
        Bond length in nm.
    persistence_bonds: float, optional
        Directional memory of the walk, in bonds. Defaults to 40.0.
    base_z: float, optional
        Height of the ring's centre above the substrate in nm. Defaults to
        55.0.
    max_attempts: int, optional
        Number of draws before giving up. Defaults to 16.
    min_bond_fraction: float, optional
        Shortest accepted bond as a fraction of ``bond_length``. Defaults
        to 0.5.
    min_nonbonded_fraction: float, optional
        Closest accepted approach of non-neighbouring beads as a fraction
        of ``bond_length``. Defaults to 0.5.

    Returns
    -------
    np.ndarray
        Bead coordinates with shape ``(n_beads, 3)`` in nm (float32).

    Raises
    ------
    ValueError
        If an argument is invalid or no valid ring is found within
        ``max_attempts`` draws.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    A 440 nm ring of 98 beads lifted to 55 nm:

    >>> coords = sim.make_tangled_ring_initial_safe(1, 98, 440 / 98)
    >>> coords.shape
    (98, 3)
    >>> round(sim.closed_contour_nm(coords, use_z=True), 1)
    440.0

    """
    n = int(n_beads)
    if n != n_beads or n < 8:
        raise ValueError("n_beads must be an integer of at least 8")
    if int(seed) != seed or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    b = float(bond_length)
    p = float(persistence_bonds)
    if not (
        np.isfinite(b)
        and b > 0
        and np.isfinite(p)
        and p > 0
        and np.isfinite(base_z)
    ):
        raise ValueError(
            "bond length/persistence must be positive; base_z must be finite"
        )
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    for attempt in range(int(max_attempts)):
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), attempt])
        )
        increments = np.empty((n, 3), dtype=np.float64)
        vec = np.array([1.0, 0.0, 0.0])
        for k in range(n):
            axis = rng.normal(size=3)
            axis -= axis.dot(vec) * vec
            if np.linalg.norm(axis) < 1e-12:
                basis = np.eye(3)[int(np.argmin(np.abs(vec)))]
                axis = basis - basis.dot(vec) * vec
            axis /= np.linalg.norm(axis)
            angle = rng.normal(scale=np.sqrt(1 / p))
            vec = vec * np.cos(angle) + np.cross(axis, vec) * np.sin(angle)
            vec /= np.linalg.norm(vec)
            increments[k] = b * vec
        increments -= increments.mean(axis=0)
        coords = np.vstack(
            [np.zeros((1, 3)), np.cumsum(increments[:-1], axis=0)]
        )
        try:
            coords = _resample_closed(coords, n)
        except ValueError:
            continue
        coords -= coords.mean(axis=0)
        length = closed_lengths(coords)["actual_3d_contour_nm"]
        if length <= 1e-12:
            continue
        coords *= n * b / length
        coords[:, 2] += float(base_z)
        result = coords.astype(np.float32)
        measures = closed_lengths(result)
        if (
            measures["min_bond_nm"] >= min_bond_fraction * b
            and measures["max_bond_nm"] <= 1.5 * b
            and min_nonbonded_distance(result) >= min_nonbonded_fraction * b
        ):
            return result
    raise ValueError(
        f"No valid initial ring for seed={seed} after {max_attempts} attempts"
    )


def plot_initial_chain(
    coords: np.ndarray,
    title: str | None = None,
) -> None:
    """Show a 3-D schematic of a chain above the substrate.

    The chain is drawn as connected beads, bead 0 is marked in red and the
    substrate (z = 0) is drawn as a transparent plane.

    Parameters
    ----------
    coords: np.ndarray
        Bead coordinates with shape ``(N, 3)`` in nm.
    title: str | None, optional
        Figure title. Defaults to `None`, which shows the bead count.

    Returns
    -------
    None
        The figure is shown in the notebook.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> coords = sim.make_tangled_ring_initial_safe(43, 171, 4.4936)
    >>> sim.plot_initial_chain(coords, "Initial ring")  # doctest: +SKIP

    """

    cc = np.vstack([coords, coords[0]])  # close the ring for plotting

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(cc[:, 0], cc[:, 1], cc[:, 2], "-o", markersize=3, linewidth=1.2)
    ax.scatter(
        cc[0, 0],
        cc[0, 1],
        cc[0, 2],
        color="red",
        s=60,
        zorder=5,
        label="bead 0",
    )

    # Draw substrate plane
    xs = np.linspace(cc[:, 0].min(), cc[:, 0].max(), 2)
    ys = np.linspace(cc[:, 1].min(), cc[:, 1].max(), 2)
    X, Y = np.meshgrid(xs, ys)
    ax.plot_surface(X, Y, np.zeros_like(X), alpha=0.15, color="cyan")

    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_zlabel("z (nm)")
    ax.set_title(title or f"Initial ring, {len(coords)} beads")
    ax.legend()
    plt.tight_layout()
    plt.show()


def test_openmm_installation() -> None:
    """Run OpenMM's own installation test.

    The test lists the available platforms (Reference, CPU, CUDA, OpenCL)
    and checks that they compute the same forces. A short message is
    printed instead when OpenMM is not installed, which is expected when
    only the non-MD pathway is used.

    Returns
    -------
    None
        The test report is printed.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> sim.test_openmm_installation()  # doctest: +SKIP

    """

    if openmm is None:
        print(
            "OpenMM is not installed. This is fine for the non-MD pathway; "
            "install it (e.g. `pip install openmm`) to use USE_MD = True."
        )
        return
    from openmm import testInstallation

    try:
        testInstallation.main()
    except Exception as e:  # pragma: no cover - platform dependent
        print(f"Test failed: {e}")


# ────────────────────────────────────────────────────────────────────────────
# Molecular dynamics relaxation (annealed OpenMM force field)
# ────────────────────────────────────────────────────────────────────────────


def run_md_relaxation(
    coords0: np.ndarray,
    seed: int,
    n_frames: int = N_FRAMES,
    steps_per_frame: int = STEPS_PER_FRAME,
    anneal_fraction: float = 0.85,
    kz_soft: float = 0.0005,
    khigh_soft: float = 0.0005,
    fads_soft: float = 0.005,
    wca_sigma_soft_factor: float = 2.0,
    wca_sigma_final_factor: float = 0.80,
    bond_length: float | None = None,
) -> list[np.ndarray]:
    """Relax a ring onto the substrate with Langevin dynamics in OpenMM.

    The initial ring from :func:`make_tangled_ring_initial_safe` is
    simulated as a coarse-grained bead chain with:

    - harmonic bonds and bending stiffness (``K_ANGLE``) along the ring,
    - a surface attraction and a quadratic plus quartic height
      confinement that pull the chain onto the substrate and flatten
      crossings,
    - a stiff wall at z = 0,
    - a Weeks-Chandler-Andersen (WCA) repulsion between non-bonded beads.

    The confinement is annealed from soft to stiff over the first
    ``anneal_fraction`` of the run, so early frames can rearrange the ring
    and remove crossings by strand passage, and later frames compress the
    adsorbed chain to realistic crossing heights. The WCA diameter is
    annealed in the opposite direction, from large to small. A variable
    Langevin integrator advances ``steps_per_frame`` steps per frame.

    To change how the chain relaxes, edit the force constants in this
    function (``F_ADS``, ``K_Z``, ``K_HIGH``, ``Z_SOFT``, ``WALL_K``,
    ``WCA_EPS_KT``, temperature, collision rate) or the annealing
    arguments. The current values are the optimised values used for the
    MD training data. The platform is chosen in the order CUDA, OpenCL,
    CPU.

    Parameters
    ----------
    coords0: np.ndarray
        Initial bead coordinates with shape ``(N, 3)`` in nm.
    seed: int
        Random seed of the Langevin integrator.
    n_frames: int, optional
        Number of recorded frames. Defaults to ``N_FRAMES``.
    steps_per_frame: int, optional
        Integration steps between frames. Defaults to ``STEPS_PER_FRAME``.
    anneal_fraction: float, optional
        Fraction of all steps over which the confinement and the WCA
        diameter are ramped to their final values. Defaults to 0.85.
    kz_soft: float, optional
        Starting value of the quadratic confinement ``K_Z`` in nm^-2.
        Defaults to 0.0005.
    khigh_soft: float, optional
        Starting value of the quartic confinement ``K_HIGH`` in nm^-4.
        Defaults to 0.0005.
    fads_soft: float, optional
        Starting value of the adsorption force ``F_ADS``. Defaults to
        0.005.
    wca_sigma_soft_factor: float, optional
        Starting WCA diameter as a multiple of the bond length. Defaults
        to 2.0.
    wca_sigma_final_factor: float, optional
        Final WCA diameter as a multiple of the bond length. Defaults to
        0.80.
    bond_length: float | None, optional
        Equilibrium bond length in nm (the actual bond from
        :func:`contour_to_beads`). Defaults to `None`, which uses
        ``BOND_LENGTH``.

    Returns
    -------
    list[np.ndarray]
        ``n_frames + 1`` coordinate arrays of shape ``(N, 3)`` in nm; frame
        0 is the geometry before energy minimisation.

    Raises
    ------
    RuntimeError
        If no OpenMM platform (CUDA, OpenCL or CPU) can be initialised.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    Relax a 115 nm minicircle for ten frames:

    >>> n, bond = sim.contour_to_beads(115.26, 4.5, 8)
    >>> coords0 = sim.make_tangled_ring_initial_safe(1, n, bond)
    >>> frames = sim.run_md_relaxation(
    ...     coords0, seed=1, n_frames=10, bond_length=bond
    ... )  # doctest: +SKIP
    >>> len(frames), frames[-1].shape  # doctest: +SKIP
    (11, (26, 3))

    """

    coords0 = np.asarray(coords0, dtype=np.float32)
    n = int(coords0.shape[0])
    bond_length = float(BOND_LENGTH if bond_length is None else bond_length)

    # how big the bounding box for the molecular dynamics needs to be
    extent = (coords0.max(axis=0) - coords0.min(axis=0)).max()
    box = float(extent + 50.0)

    # All lengths in nm, energies in kJ/mol.
    BOLTZ_kJmol = 0.00831445  # kJ / (mol · K)
    temperature = 100.0  # K  (sets energy scale kT ≈ 0.0083 kJ/mol)
    kT = BOLTZ_kJmol * temperature  # kJ/mol
    conlen = bond_length  # nm
    mass_amu = 100.0  # Da per bead
    collision_rate = 0.6  # ps⁻¹
    error_tol = 0.005

    # ── Build system ────────────────────────────────────────────────────────
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box, 0, 0),
        openmm.Vec3(0, box, 0),
        openmm.Vec3(0, 0, box),
    )
    for _ in range(n):
        system.addParticle(mass_amu)

    # ── 1. Harmonic bonds ───────────────────────────────────────────────────

    # OpenMM HarmonicBondForce: E = (k/2)·(r − r0)²  →  k = kT / wiggle²
    bond_L = bond_length  # nm
    wiggle = 0.1  # nm  (bondWiggleDistance)
    k_bond = kT / (wiggle**2)  # kJ/mol/nm²

    bond_force = openmm.HarmonicBondForce()
    bond_force.setUsesPeriodicBoundaryConditions(False)
    for i in range(n):
        bond_force.addBond(i, (i + 1) % n, bond_L, k_bond)
    system.addForce(bond_force)

    # ── 2. Bending (angle) stiffness ────────────────────────────────────────
    # E = kT · k_angle · (1 − cos(θ − π))  — equilibrium at θ = π (straight)
    k_angle = K_ANGLE
    angle_force = openmm.CustomAngleForce(
        "kT * kangle * (1 - cos(theta - 3.141592653589793))"
    )
    angle_force.addGlobalParameter("kT", kT)
    angle_force.addGlobalParameter("kangle", k_angle)
    for i in range(n):
        angle_force.addAngle((i - 1) % n, i, (i + 1) % n, [])
    system.addForce(angle_force)

    # ── 3. Surface attraction and nonlinear height confinement ──────────────
    #
    # For z >= 0:
    #
    # U(z) = kT * [
    #     Fads * z
    #     + 0.5 * Kz * z^2
    #     + 0.25 * Khigh * max(z - zsoft, 0)^4
    # ]
    #
    # The adsorption term keeps ordinary beads near the surface.
    # The quadratic term increasingly pushes beads down with height.
    # The quartic term strongly suppresses unusually tall crossings while
    # having little effect on beads below zsoft.
    #
    # Fads, Kz, and Khigh are annealed during the run (see the frame loop
    # below): they start at the *_soft values passed in, and ramp linearly
    # up to F_ADS/K_Z/K_HIGH over `anneal_fraction` of the total steps.
    #
    # For z < 0, a stiff harmonic wall prevents penetration into the
    # substrate.

    F_ADS = 18.0  # dimensionless constant downward force (final value)
    K_Z = 5.0  # nm^-2; linearly increasing downward force (final value)
    K_HIGH = 27.0  # nm^-4; strong high-z confinement (final value)
    Z_SOFT = 2.000  # nm; quartic confinement begins above this height
    WALL_K = 500.0  # nm^-2; substrate penetration penalty

    wall_expr = (
        "kT*("
        "Fads*z*step(z)"
        "+ 0.5*Kz*z^2*step(z)"
        "+ 0.25*Khigh*(z-zsoft)^4*step(z-zsoft)"
        "+ 0.5*Kwall*z^2*step(-z)"
        ")"
    )

    wall_force = openmm.CustomExternalForce(wall_expr)
    wall_force.addGlobalParameter("kT", kT)
    # Start the confinement at its soft values; the frame loop below ramps
    # these up toward F_ADS/K_Z/K_HIGH as the run progresses.
    wall_force.addGlobalParameter("Fads", fads_soft)
    wall_force.addGlobalParameter("Kz", kz_soft)
    wall_force.addGlobalParameter("Khigh", khigh_soft)
    wall_force.addGlobalParameter("zsoft", Z_SOFT)
    wall_force.addGlobalParameter("Kwall", WALL_K)

    for i in range(n):
        wall_force.addParticle(i, [])

    system.addForce(wall_force)

    # ── 4. WCA excluded-volume repulsion ────────────────────────────────────
    #
    # sigma is annealed over the run, but in the *opposite* direction to
    # Fads/Kz/Khigh: it starts large (wca_sigma_soft_factor * conlen) so beads
    # push apart strongly at long range and nonphysical crossings/overlaps get
    # resolved early, then shrinks to wca_sigma_final_factor * conlen
    # (near the physical DNA diameter) as the run progresses, letting the
    # chain settle
    # closer together by the end. Follows the same anneal_fraction schedule.

    WCA_SIGMA_SOFT_NM = wca_sigma_soft_factor * conlen
    WCA_SIGMA_FINAL_NM = wca_sigma_final_factor * conlen
    WCA_EPS_KT = 28.0

    rep_sigma_soft = float(WCA_SIGMA_SOFT_NM)
    # Static neighbor-list cutoff must bound the *largest* possible rc across
    # the whole anneal (i.e. based on the soft/larger sigma), since sig itself
    # is now a runtime-varying global parameter and setCutoffDistance() cannot
    # be updated per-frame.
    rep_cutoff = (2.0 ** (1.0 / 6.0)) * rep_sigma_soft

    rep_expr = (
        "4*eps*((sig/r)^12-(sig/r)^6+0.25)"
        "*step(rc-r);"
        "rc=2^(1.0/6.0)*sig"
    )

    rep_force = openmm.CustomNonbondedForce(rep_expr)
    rep_force.setNonbondedMethod(
        openmm.CustomNonbondedForce.CutoffNonPeriodic
    )
    rep_force.setCutoffDistance(rep_cutoff)

    rep_force.addGlobalParameter("eps", WCA_EPS_KT * kT)
    # Start sig at its soft (large) value; the frame loop below ramps it down
    # toward WCA_SIGMA_FINAL_NM as the run progresses.
    rep_force.addGlobalParameter("sig", rep_sigma_soft)

    for i in range(n):
        rep_force.addParticle([])

    # Exclude directly bonded neighbours.
    for i in range(n):
        rep_force.addExclusion(i, (i + 1) % n)

    system.addForce(rep_force)

    # ── Integrator ──────────────────────────────────────────────────────────
    integrator = openmm.VariableLangevinIntegrator(
        temperature, collision_rate, error_tol
    )
    integrator.setRandomNumberSeed(int(seed))

    # ── Platform selection: OpenCL → CPU fallback ───────────────────────────
    context = None
    # OpenCL enables Mac users to be able to use GPU acceleration as well
    for platform_name in ("CUDA", "OpenCL", "CPU"):
        try:
            platform = openmm.Platform.getPlatformByName(platform_name)
            context = openmm.Context(system, integrator, platform)
            print(f"Using {platform_name} platform.")
            break
        except openmm.OpenMMException as e:
            print(f"  {platform_name} unavailable: {e}")
        except Exception as e:
            print(f"  {platform_name} failed unexpectedly: {e}")

    if context is None:
        raise RuntimeError("No OpenMM platform could be initialised.")

    # ── Set initial positions and box ───────────────────────────────────────
    positions = [
        openmm.Vec3(
            float(coords0[i, 0]), float(coords0[i, 1]), float(coords0[i, 2])
        )
        for i in range(n)
    ]
    context.setPositions(positions)
    context.setPeriodicBoxVectors(
        openmm.Vec3(box, 0, 0),
        openmm.Vec3(0, box, 0),
        openmm.Vec3(0, 0, box),
    )

    # Capture the true initial geometry before any minimisation
    # (just in case the chain reaches substrate even before the 1st frame)
    frames = []
    state = context.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    frames.append(pos.copy())

    # Cap iterations
    openmm.LocalEnergyMinimizer.minimize(
        context, tolerance=50, maxIterations=200
    )

    # ── Record frames, annealing the confinement strength ───────────────────
    #
    # Fads, Kz, and Khigh are ramped linearly from their soft starting values
    # up to their final (F_ADS/K_Z/K_HIGH) values over the first
    # `anneal_fraction` of the total integration steps, then held constant.
    # This gives the ring room to reduce its apparent crossing count via
    # genuine strand-passage while confinement is still weak, then compresses
    # the resulting lower-crossing conformation down to the target height
    # once confinement reaches its final, stiff value.
    n_total_steps = int(n_frames) * int(steps_per_frame)
    anneal_steps = max(float(anneal_fraction) * n_total_steps, 1.0)
    step_count = 0

    for _ in range(int(n_frames)):
        progress = min(step_count / anneal_steps, 1.0)
        context.setParameter(
            "Fads", fads_soft + progress * (F_ADS - fads_soft)
        )
        context.setParameter("Kz", kz_soft + progress * (K_Z - kz_soft))
        context.setParameter(
            "Khigh", khigh_soft + progress * (K_HIGH - khigh_soft)
        )
        # sigma ramps DOWN (soft/large -> final/small), opposite direction
        # from the confinement terms above.
        context.setParameter(
            "sig",
            WCA_SIGMA_SOFT_NM
            + progress * (WCA_SIGMA_FINAL_NM - WCA_SIGMA_SOFT_NM),
        )

        integrator.step(int(steps_per_frame))
        step_count += int(steps_per_frame)

        state = context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        frames.append(pos.copy())

    return frames


# ────────────────────────────────────────────────────────────────────────────
# Non-MD chain generation (2-D persistent ring)
# ────────────────────────────────────────────────────────────────────────────


def make_ring_2d_persistent_initial(
    seed: int,
    n_beads: int,
    bond_length: float = BOND_LENGTH,
    persistence_bonds: float = PERSISTENCE_BONDS,
    angle_stiffness_mult: float = ANGLE_STIFFNESS_MULT,
    z_std: float = 0.08,
    base_z: float = 0.2,
    angle_limit: float = np.pi / 2,
) -> list[np.ndarray]:
    """Generate a closed 2-D persistent ring on the substrate.

    A periodic sequence of turning angles is drawn with standard deviation
    ``sqrt(2 / persistence_bonds) / angle_stiffness_mult``, clipped to
    ``angle_limit``, smoothed to suppress sharp corners and corrected so
    that it turns by exactly 2 pi, which closes the ring. The curve is then
    resampled by arc length so all bonds are ``bond_length`` long, centred
    at the origin, and given heights ``base_z`` with Gaussian scatter
    ``z_std``.

    Parameters
    ----------
    seed: int
        Random seed.
    n_beads: int
        Number of beads, at least 6.
    bond_length: float, optional
        Distance between neighbouring beads in nm. Defaults to
        ``BOND_LENGTH``.
    persistence_bonds: float, optional
        Directional memory in bonds; larger values give a smoother ring.
        Defaults to ``PERSISTENCE_BONDS``.
    angle_stiffness_mult: float, optional
        Divides the turning-angle spread; larger values give a smoother
        ring. Defaults to ``ANGLE_STIFFNESS_MULT``.
    z_std: float, optional
        Standard deviation of the bead heights in nm. Defaults to 0.08.
    base_z: float, optional
        Mean bead height above the substrate in nm. Defaults to 0.2.
    angle_limit: float, optional
        Largest turning angle per bond in radians. Defaults to pi / 2.

    Returns
    -------
    list[np.ndarray]
        A single coordinate array of shape ``(n_beads, 3)`` in nm, wrapped
        in a list so the output looks like the frames of
        :func:`run_md_relaxation`.

    Raises
    ------
    ValueError
        If ``n_beads`` is below 6 or the generated ring has zero length.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> frames = sim.make_ring_2d_persistent_initial(42, n_beads=60)
    >>> len(frames), frames[0].shape
    (1, (60, 3))

    """

    rng = np.random.default_rng(int(seed))
    n = int(n_beads)
    if n < 6:
        raise ValueError(
            "n_beads must be at least"
            "6 for a stable closed persistent ring."
        )

    bond_length = float(bond_length)

    # ------------------------------------------------------------------
    # 1) Build a periodic turning-angle process on the ring
    # ------------------------------------------------------------------

    # larger persistence_bonds / angle_stiffness_mult => less angular noise
    sigma_dtheta = np.sqrt(2.0 / max(float(persistence_bonds), 1.0)) / max(
        float(angle_stiffness_mult), 1e-6
    )

    # Raw local turning increments
    dtheta = rng.normal(0.0, sigma_dtheta, size=n).astype(np.float64)
    dtheta = np.clip(dtheta, -float(angle_limit), float(angle_limit))

    # Circular smoothing of turning increments to suppress sharp local
    # corners.
    # Stronger persistence / stiffness => broader smoothing.
    smooth_width = int(
        np.clip(np.round(max(float(persistence_bonds), 1.0) / 6.0), 1, 8)
    )
    if smooth_width > 0:
        offsets = np.arange(-smooth_width, smooth_width + 1, dtype=np.float64)
        kernel_sigma = max(smooth_width / 2.0, 1e-6)
        kernel = np.exp(-0.5 * (offsets / kernel_sigma) ** 2)
        kernel /= kernel.sum()

        dtheta_smooth = np.zeros_like(dtheta)
        for shift, w in zip(offsets.astype(int), kernel):
            dtheta_smooth += w * np.roll(dtheta, shift)
        dtheta = dtheta_smooth

    # Enforce exact total turning of 2*pi for a closed planar ring.
    # This avoids the nonphysical post hoc closure drift used before.
    total_turn = dtheta.sum()
    dtheta += (2.0 * np.pi - total_turn) / n

    # Re-clip gently after correction, then re-enforce total turning once
    # more.
    dtheta = np.clip(dtheta, -float(angle_limit), float(angle_limit))
    dtheta += (2.0 * np.pi - dtheta.sum()) / n

    # ---------------------------------------------------------------------
    # 2) Build periodic tangent angles and integrate to a dense closed path
    # ---------------------------------------------------------------------
    theta0 = rng.uniform(0.0, 2.0 * np.pi)
    thetas = theta0 + np.cumsum(dtheta)

    # Use a dense piecewise-linear curve first; later resample uniformly.
    tangents = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)

    # Integrate unit-speed steps to get a provisional loop
    xy = np.zeros((n, 2), dtype=np.float64)
    xy[1:] = np.cumsum(tangents[:-1] * bond_length, axis=0)
    end_error = (xy[-1] + tangents[-1] * bond_length) - xy[0]

    # Distribute closure correction through the bond vectors
    bond_corr = end_error / n
    bonds = tangents * bond_length - bond_corr[None, :]

    # Renormalize bond lengths back toward bond_length while keeping closure.
    bond_norms = np.linalg.norm(bonds, axis=1)
    good = bond_norms > 1e-12
    bonds[good] *= (bond_length / bond_norms[good])[:, None]

    # Re-enforce closure after renormalization
    bonds -= bonds.sum(axis=0, keepdims=True) / n

    # Reconstruct coordinates from corrected bonds
    xy = np.zeros((n, 2), dtype=np.float64)
    xy[1:] = np.cumsum(bonds[:-1], axis=0)

    # -----------------------------------------------------------------
    # 3) Arc-length resample so bead spacing stays uniform and physical
    # -----------------------------------------------------------------
    closed_xy = np.vstack([xy, xy[0]])
    seg = np.linalg.norm(np.diff(closed_xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total_len = s[-1]

    if total_len <= 1e-12:
        raise ValueError(
            "Degenerate ring generated;" "total contour length is zero."
        )

    target_s = np.linspace(0.0, total_len, n + 1)[:-1]

    x_resamp = np.interp(target_s, s, closed_xy[:, 0])
    y_resamp = np.interp(target_s, s, closed_xy[:, 1])
    xy = np.stack([x_resamp, y_resamp], axis=1)

    # Final contour-length normalization
    # so mean bond length matches bond_length
    final_bonds = np.roll(xy, -1, axis=0) - xy
    mean_bond = np.mean(np.linalg.norm(final_bonds, axis=1))
    if mean_bond > 1e-12:
        xy *= bond_length / mean_bond

    # Center at origin
    xy -= xy.mean(axis=0, keepdims=True)

    # ------------------------------------------------------------------
    # 4) Assembly with preserved Gaussian z-noise logic
    # ------------------------------------------------------------------
    coords = np.zeros((n, 3), dtype=np.float32)
    coords[:, 0] = xy[:, 0].astype(np.float32)
    coords[:, 1] = xy[:, 1].astype(np.float32)
    coords[:, 2] = rng.normal(loc=base_z, scale=z_std, size=n).astype(
        np.float32
    )

    # Return as a list containing one frame to mimic run_md_relaxation output
    return [coords]


def make_non_md_ring(
    seed: int,
    contour_length_nm: float | None = None,
) -> np.ndarray:
    """Generate a non-MD ring with exactly the requested contour.

    This is the calibrated non-MD generator. The requested contour length
    is converted to beads with :func:`contour_to_beads`, the closed 2-D
    ring is drawn with :func:`make_ring_2d_persistent_initial` using the
    non-MD settings (``PERSISTENCE_BONDS``, ``ANGLE_STIFFNESS_MULT``,
    ``NON_MD_ANGLE_LIMIT_RAD``, ``NON_MD_Z_STD_NM``,
    ``NON_MD_BASE_Z_NM``), and the (x, y) coordinates are rescaled in
    float64 so that the closed contour, including the last-to-first bond,
    equals the request to numerical precision.

    Parameters
    ----------
    seed: int
        Random seed.
    contour_length_nm: float | None, optional
        Requested contour length in nm. Defaults to `None`, which uses
        ``CONTOUR_LENGTH_NM``.

    Returns
    -------
    np.ndarray
        Bead coordinates with shape ``(N, 3)`` in nm (float32).

    Raises
    ------
    ArithmeticError
        If the rescaled contour differs from the request by more than
        1e-6 nm.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> coords = sim.make_non_md_ring(1, 115.26)
    >>> round(sim.closed_contour_nm(coords), 3)
    115.26

    """

    length = float(
        CONTOUR_LENGTH_NM if contour_length_nm is None else contour_length_nm
    )
    n_beads, bond = contour_to_beads(length, BOND_LENGTH, NON_MD_MIN_BEADS)
    coords = make_ring_2d_persistent_initial(
        int(seed),
        n_beads=n_beads,
        bond_length=bond,
        persistence_bonds=PERSISTENCE_BONDS,
        angle_stiffness_mult=ANGLE_STIFFNESS_MULT,
        z_std=NON_MD_Z_STD_NM,
        base_z=NON_MD_BASE_Z_NM,
        angle_limit=NON_MD_ANGLE_LIMIT_RAD,
    )[0].astype(np.float64)
    # Include the closing bond and correct in float64 after the float32
    # generator.
    for _ in range(2):
        coords[:, :2] *= length / closed_contour_nm(coords)
    if abs(closed_contour_nm(coords) - length) > 1e-6:
        raise ArithmeticError("Closed contour calibration failed")
    return coords.astype(np.float32)


def plot_relaxed_chain(
    coords: np.ndarray,
    title: str | None = None,
) -> None:
    """Show a chain lying on the substrate, in 3-D and from above.

    Used for the non-MD pathway, where the generated ring is already on
    the substrate and there is no relaxation to animate. The right-hand
    panel gives the contour length and the bead count.

    Parameters
    ----------
    coords: np.ndarray
        Bead coordinates with shape ``(N, 3)`` in nm.
    title: str | None, optional
        Figure title. Defaults to `None` (no title).

    Returns
    -------
    None
        The figure is shown in the notebook.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> coords = sim.make_non_md_ring(1, 768.4)
    >>> sim.plot_relaxed_chain(coords, "Non-MD ring")  # doctest: +SKIP

    """

    c = np.asarray(coords, dtype=np.float32)
    cc = np.vstack([c, c[0]])
    pad = 10.0
    x_min, x_max = c[:, 0].min() - pad, c[:, 0].max() + pad
    y_min, y_max = c[:, 1].min() - pad, c[:, 1].max() + pad

    fig = plt.figure(figsize=(12, 5.5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    X, Y = np.meshgrid(
        np.linspace(x_min, x_max, 2), np.linspace(y_min, y_max, 2)
    )
    ax3d.plot_surface(X, Y, np.zeros_like(X), alpha=0.2, color="cyan")
    ax3d.plot(cc[:, 0], cc[:, 1], cc[:, 2], "-", linewidth=1)
    ax3d.scatter(c[:, 0], c[:, 1], c[:, 2], s=5)
    ax3d.set_xlim(x_min, x_max)
    ax3d.set_ylim(y_min, y_max)
    ax3d.set_zlim(-1.0, max(5.0, float(c[:, 2].max()) + 3.0))
    ax3d.set_xlabel("x (nm)")
    ax3d.set_ylabel("y (nm)")
    ax3d.set_zlabel("z (nm)")
    ax3d.set_title("On the substrate (3-D)")

    ax2d = fig.add_subplot(1, 2, 2)
    ax2d.plot(cc[:, 0], cc[:, 1], "-", linewidth=1)
    ax2d.scatter(c[:, 0], c[:, 1], s=6)
    ax2d.set_aspect("equal")
    ax2d.set_xlabel("x (nm)")
    ax2d.set_ylabel("y (nm)")
    ax2d.set_title(
        f"Seen from above: contour {closed_contour_nm(c):.1f} nm, "
        f"{len(c)} beads"
    )

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# ────────────────────────────────────────────────────────────────────────────
# MD animation
# ────────────────────────────────────────────────────────────────────────────


def show_md_animation(
    frames: list[np.ndarray],
    interval_ms: int = 150,
) -> None:
    """Show an HTML animation of MD frames in the notebook.

    Every frame is drawn in 3-D with the substrate as a transparent plane;
    the title gives the frame number and the height range of the beads.
    The x and y limits come from the first frame so that drift does not
    change the scale. The input arrays are not modified.

    Parameters
    ----------
    frames: list[np.ndarray]
        Coordinate arrays of shape ``(N, 3)`` in nm, for example from
        :func:`run_md_relaxation`.
    interval_ms: int, optional
        Time between frames in milliseconds. Defaults to 150.

    Returns
    -------
    None
        The animation is displayed with ``IPython.display``.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> chain = sim.generate_chain(
    ...     1, 768.4, return_frames=True
    ... )  # doctest: +SKIP
    >>> sim.show_md_animation(chain["frames"][::10])  # doctest: +SKIP

    """

    frames = [np.asarray(f, dtype=np.float32) for f in frames]

    # Use only the first frame for x/y limits
    # — avoids COM drift blowing the scale
    first = frames[0]
    xy_pad = 5.0
    x_min, x_max = first[:, 0].min() - xy_pad, first[:, 0].max() + xy_pad
    y_min, y_max = first[:, 1].min() - xy_pad, first[:, 1].max() + xy_pad

    # Clamp z: floor at -1 (just below substrate),
    # ceiling from initial max + padding
    z_min = -1.0
    z_max = float(first[:, 2].max()) + 3.0

    x_plane = np.linspace(x_min, x_max, 2)
    y_plane = np.linspace(y_min, y_max, 2)
    X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
    Z_plane = np.zeros_like(X_plane)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    def _set_axes():
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z (nm)")

    def init():
        _set_axes()
        return []

    def update(i):
        ax.cla()
        c = frames[i]
        # Clip beads to visible z range for plotting so outliers don't distort
        visible = (c[:, 2] >= z_min) & (c[:, 2] <= z_max)
        cc = np.vstack([c, c[0]])
        ax.plot(cc[:, 0], cc[:, 1], cc[:, 2], "-", linewidth=1)
        ax.scatter(c[visible, 0], c[visible, 1], c[visible, 2], s=5)
        ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.2, color="cyan")
        _set_axes()
        ax.set_title(
            f"Frame {i}/{len(frames)-1}  |  "
            f"z ∈ [{c[:,2].min():.1f}, {c[:,2].max():.1f}] nm"
        )
        return []

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        init_func=init,
        interval=int(interval_ms),
        blit=False,
    )
    plt.close(fig)
    display(HTML(ani.to_jshtml()))


# ────────────────────────────────────────────────────────────────────────────
# Height-based AFM rendering
# ────────────────────────────────────────────────────────────────────────────


def _seg_intersect_2d(
    p1: np.ndarray,
    p2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    eps: float = 1e-12,
) -> tuple[bool, np.ndarray | None, float | None, float | None]:
    """Intersect two 2-D line segments.

    The intersection point satisfies
    ``p1 + t * (p2 - p1) = q1 + u * (q2 - q1)`` with ``0 <= t, u <= 1``.
    Parallel segments are reported as not intersecting.

    Parameters
    ----------
    p1: np.ndarray
        Start of the first segment, shape ``(2,)``.
    p2: np.ndarray
        End of the first segment, shape ``(2,)``.
    q1: np.ndarray
        Start of the second segment, shape ``(2,)``.
    q2: np.ndarray
        End of the second segment, shape ``(2,)``.
    eps: float, optional
        Tolerance for parallel segments and for the end points. Defaults
        to 1e-12.

    Returns
    -------
    tuple[bool, np.ndarray | None, float | None, float | None]
        Whether the segments intersect, the intersection point and the
        parameters ``t`` and ``u`` (all `None` without an intersection).

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    Two diagonals of the unit square cross at its centre:

    >>> hit, point, t, u = sim._seg_intersect_2d(
    ...     np.array([0.0, 0.0]), np.array([1.0, 1.0]),
    ...     np.array([0.0, 1.0]), np.array([1.0, 0.0]),
    ... )
    >>> hit, point.tolist(), float(t), float(u)
    (True, [0.5, 0.5], 0.5, 0.5)

    """

    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    r = p2 - p1
    s = q2 - q1
    rxs = r[0] * s[1] - r[1] * s[0]
    if abs(rxs) < eps:
        return False, None, None, None
    qp = q1 - p1
    t = (qp[0] * s[1] - qp[1] * s[0]) / rxs
    u = (qp[0] * r[1] - qp[1] * r[0]) / rxs
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return True, p1 + t * r, t, u
    return False, None, None, None


def _circ_sep(
    n: int,
    i: int,
    j: int,
) -> int:
    """Return the circular distance between two bead indices.

    On a ring of ``n`` beads, beads ``i`` and ``j`` are separated by the
    shorter way round the ring. This distinguishes genuine crossings from
    neighbouring segments.

    Parameters
    ----------
    n: int
        Number of beads in the ring.
    i: int
        Index of the first bead.
    j: int
        Index of the second bead.

    Returns
    -------
    int
        The number of bonds between the two beads along the shorter path.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    On a ring of 10 beads, beads 1 and 9 are two bonds apart:

    >>> sim._circ_sep(10, 1, 9)
    2

    """

    d = abs(int(i) - int(j))
    return min(d, n - d)


def disk_footprint(
    r_px: float,
) -> np.ndarray:
    """Return a boolean disk of radius ``r_px`` pixels.

    The disk is used as a structuring element to widen one-pixel lines
    into masks of a given width.

    Parameters
    ----------
    r_px: float
        Radius in pixels.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(2R + 1, 2R + 1)`` with ``R = ceil(r_px)``,
        `True` inside the disk.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> sim.disk_footprint(1).astype(int)
    array([[0, 1, 0],
           [1, 1, 1],
           [0, 1, 0]])

    """

    if r_px < 0.5:
        return np.ones((1, 1), dtype=bool)
    r = int(np.ceil(r_px))
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y) <= r * r


def upsample_nn(
    a: np.ndarray,
    out_shape: tuple[int, int],
) -> np.ndarray:
    """Upsample an array by repeating pixels (nearest neighbour).

    The array is enlarged by integer factors that cover ``out_shape`` and
    then cropped to exactly ``out_shape``.

    Parameters
    ----------
    a: np.ndarray
        2-D array to upsample.
    out_shape: tuple[int, int]
        Target shape ``(rows, columns)``.

    Returns
    -------
    np.ndarray
        Array of shape ``out_shape``.

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    >>> sim.upsample_nn(np.array([[1, 2], [3, 4]]), (4, 4))
    array([[1, 1, 2, 2],
           [1, 1, 2, 2],
           [3, 3, 4, 4],
           [3, 3, 4, 4]])

    """

    ny_out, nx_out = out_shape
    ny_in, nx_in = a.shape
    if (ny_out, nx_out) == (ny_in, nx_in):
        return a
    sy = max(ny_out // ny_in, 1)
    sx = max(nx_out // nx_in, 1)
    a_up = a.repeat(sy, axis=0).repeat(sx, axis=1)
    return a_up[:ny_out, :nx_out]


def compute_internal_render_grid(
    extent: tuple[float, float, float, float],
    chain_resolution_nm: float,
    min_size: int = 16,
) -> tuple[int, int, float]:
    """Size the internal rendering grid for a physical extent.

    The grid covers ``extent`` at ``chain_resolution_nm`` per pixel; the
    effective resolution is adjusted slightly so that the grid has a whole
    number of pixels.

    Parameters
    ----------
    extent: tuple[float, float, float, float]
        Physical extent ``(xmin, xmax, ymin, ymax)`` in nm.
    chain_resolution_nm: float
        Requested internal resolution in nm per pixel.
    min_size: int, optional
        Smallest number of pixels along each axis. Defaults to 16.

    Returns
    -------
    tuple[int, int, float]
        Grid width, grid height and effective nm per pixel.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    A 100 nm square at 0.5 nm per pixel:

    >>> nx, ny, p_eff = sim.compute_internal_render_grid(
    ...     (0.0, 100.0, 0.0, 100.0), 0.5
    ... )
    >>> nx, ny, round(p_eff, 4)
    (201, 201, 0.5)

    """
    xmin, xmax, ymin, ymax = extent
    width_nm = max(float(xmax - xmin), 1e-6)
    height_nm = max(float(ymax - ymin), 1e-6)
    target_p = max(float(chain_resolution_nm), 1e-6)
    nx = max(int(min_size), int(np.ceil(width_nm / target_p)) + 1)
    ny = max(int(min_size), int(np.ceil(height_nm / target_p)) + 1)
    px_eff = width_nm / max(nx - 1, 1)
    py_eff = height_nm / max(ny - 1, 1)
    p_eff = 0.5 * (px_eff + py_eff)
    return nx, ny, float(p_eff)


def resize_image_to_shape(
    a: np.ndarray,
    out_shape: tuple[int, int],
    order: int = 1,
) -> np.ndarray:
    """Resize a 2-D array to an exact shape.

    The array is scaled with ``scipy.ndimage.zoom`` and then padded or
    cropped by at most a pixel so the output matches ``out_shape``
    exactly.

    Parameters
    ----------
    a: np.ndarray
        2-D array to resize.
    out_shape: tuple[int, int]
        Target shape ``(rows, columns)``.
    order: int, optional
        Spline order of the interpolation (0 = nearest, 1 = linear).
        Defaults to 1.

    Returns
    -------
    np.ndarray
        Array of shape ``out_shape``.

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    >>> sim.resize_image_to_shape(np.ones((10, 10)), (25, 30)).shape
    (25, 30)

    """

    a = np.asarray(a)
    out_shape = tuple(int(v) for v in out_shape)
    if a.shape == out_shape:
        return a.copy()
    zoom_factors = (out_shape[0] / a.shape[0], out_shape[1] / a.shape[1])
    out = zoom(
        a, zoom_factors, order=order, mode="nearest", prefilter=(order > 1)
    )
    out = out[: out_shape[0], : out_shape[1]]
    if out.shape != out_shape:
        pad_y = max(0, out_shape[0] - out.shape[0])
        pad_x = max(0, out_shape[1] - out.shape[1])
        out = np.pad(out, ((0, pad_y), (0, pad_x)), mode="edge")
        out = out[: out_shape[0], : out_shape[1]]
    return out


def dna_shape(
    radius_nm: float,
    chain_resolution_nm: float,
    max_radius_px: int = 96,
    height_nm: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the structuring element that gives DNA its cross-section.

    The element is a spherical cap of lateral radius ``radius_nm`` whose
    peak is ``height_nm``; grey-dilating the chain's centre line with it
    gives the rendered DNA strand.

    Parameters
    ----------
    radius_nm: float
        Lateral radius of the strand in nm.
    chain_resolution_nm: float
        Pixel size of the rendering grid in nm.
    max_radius_px: int, optional
        Largest allowed radius in pixels. Defaults to 96.
    height_nm: float | None, optional
        Peak height in nm. Defaults to `None`, which uses
        ``2 * radius_nm``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The boolean footprint and the height structure (``-1e9`` outside
        the footprint), both of shape ``(2R + 1, 2R + 1)``.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    A 4.47 nm wide, 2.3 nm high strand on a 0.5 nm grid:

    >>> footprint, cap = sim.dna_shape(4.47 / 2, 0.5, height_nm=2.3)
    >>> footprint.shape, round(float(cap.max()), 2)
    ((11, 11), 2.3)

    """
    radius_nm = max(float(radius_nm), 1e-12)
    chain_resolution_nm = max(float(chain_resolution_nm), 1e-12)
    if height_nm is None:
        height_nm = 2.0 * radius_nm
    height_nm = max(float(height_nm), 1e-12)
    r_px = radius_nm / chain_resolution_nm
    radius_px = int(np.clip(np.ceil(r_px), 1, max_radius_px))
    y, x = np.ogrid[
        -radius_px : radius_px + 1,
        -radius_px : radius_px + 1,
    ]
    d2 = x * x + y * y
    inside = d2 <= r_px * r_px
    d_nm = np.sqrt(d2).astype(np.float32)
    d_nm *= np.float32(chain_resolution_nm)
    structure = np.full(
        (2 * radius_px + 1, 2 * radius_px + 1),
        -1e9,
        dtype=np.float32,
    )
    cap = np.sqrt(np.maximum(radius_nm**2 - d_nm[inside] ** 2, 0.0))
    cap /= radius_nm
    structure[inside] = (height_nm * cap).astype(np.float32)
    return inside.astype(bool), structure


def AFM_tip(
    tip_radius_nm: float,
    chain_resolution_nm: float,
    max_radius_px: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the structuring element of a spherical AFM tip.

    Grey-dilating a height map with this element mimics the broadening by
    a tip of radius ``tip_radius_nm``. A zero or tiny radius gives a
    one-pixel element that leaves the image unchanged.

    Parameters
    ----------
    tip_radius_nm: float
        Tip radius in nm.
    chain_resolution_nm: float
        Pixel size of the rendering grid in nm.
    max_radius_px: int, optional
        Largest allowed radius in pixels. Defaults to 128.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The boolean footprint and the tip profile (0 at the centre,
        negative towards the edge, ``-1e9`` outside the footprint).

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> footprint, tip = sim.AFM_tip(7.0, 0.5)
    >>> footprint.shape, float(tip.max())
    ((29, 29), 0.0)
    >>> sim.AFM_tip(0.0, 0.5)[0].shape
    (1, 1)

    """
    tip_radius_nm = max(float(tip_radius_nm), 0.0)
    chain_resolution_nm = max(float(chain_resolution_nm), 1e-12)
    if tip_radius_nm <= 1e-12:
        inside = np.ones((1, 1), dtype=bool)
        structure = np.zeros((1, 1), dtype=np.float32)
        return inside, structure
    r_px = tip_radius_nm / chain_resolution_nm
    radius_px = int(np.clip(np.ceil(r_px), 1, max_radius_px))
    y, x = np.ogrid[
        -radius_px : radius_px + 1,
        -radius_px : radius_px + 1,
    ]
    d2 = x * x + y * y
    inside = d2 <= r_px * r_px
    d_nm = np.sqrt(d2).astype(np.float32)
    d_nm *= np.float32(chain_resolution_nm)
    structure = np.full(
        (2 * radius_px + 1, 2 * radius_px + 1),
        -1e9,
        dtype=np.float32,
    )
    structure[inside] = (
        np.sqrt(np.maximum(tip_radius_nm**2 - d_nm[inside] ** 2, 0.0))
        - tip_radius_nm
    ).astype(np.float32)
    return inside.astype(bool), structure


def calculate_crossing_taper_weight(
    idx_off: int,
    w: int,
    profile: Literal["gaussian", "hemisphere", "linear"] = "gaussian",
    sigma: float | None = None,
) -> float:
    """Return the weight that fades a crossing-height boost.

    Beads within ``w`` beads of a crossing are raised; the weight is 1 at
    the crossing and falls off with the distance ``idx_off``, so the
    raised strand joins the rest of the chain smoothly. The Gaussian
    profile reaches 0 at ``w`` beads; the linear and hemisphere profiles
    reach 0 at ``w + 1`` beads.

    Parameters
    ----------
    idx_off: int
        Distance in beads from the crossing.
    w: int
        Half-width of the boosted window in beads.
    profile: "gaussian" | "hemisphere" | "linear", optional
        Shape of the fade. Defaults to "gaussian".
    sigma: float | None, optional
        Width of the Gaussian profile in beads. Defaults to `None`, which
        uses ``0.5 * (w + 0.5)``.

    Returns
    -------
    float
        Weight between 0.0 and 1.0.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> sim.calculate_crossing_taper_weight(0, 3)
    1.0
    >>> sim.calculate_crossing_taper_weight(3, 3, profile="linear")
    0.25
    >>> sim.calculate_crossing_taper_weight(3, 3)
    0.0

    """

    if w <= 0:
        return 1.0
    a = abs(idx_off)
    if profile == "linear":
        return 1.0 - (a / (w + 1.0))
    if profile == "hemisphere":
        x = a / (w + 1.0)
        return float(np.sqrt(max(0.0, 1.0 - x * x)))
    # gaussian (default)
    if sigma is None:
        sigma = max(1e-6, 0.5 * (w + 0.5))
    g = float(np.exp(-(idx_off**2) / (2.0 * sigma**2)))
    g_end = float(np.exp(-(w**2) / (2.0 * sigma**2)))
    if g_end < 0.999:
        g = float(np.clip((g - g_end) / (1.0 - g_end), 0.0, 1.0))
    return g


def collect_intersections(
    xy_nm: np.ndarray,
    min_separation_beads: int = 12,
    merge_radius_nm: float = 0.5,
) -> list[dict]:
    """Find the self-crossings of a closed polyline.

    Every pair of segments is tested with :func:`_seg_intersect_2d`;
    neighbouring segments and segments closer than
    ``min_separation_beads`` along the ring are skipped. Hits within
    ``merge_radius_nm`` of each other are merged with
    :func:`merge_crossing_clusters`.

    Parameters
    ----------
    xy_nm: np.ndarray
        Bead positions with shape ``(N, 2)`` in nm.
    min_separation_beads: int, optional
        Smallest separation along the ring, in beads, for two segments to
        count as crossing. Defaults to 12.
    merge_radius_nm: float, optional
        Hits closer than this are merged. Defaults to 0.5.

    Returns
    -------
    list[dict]
        One dictionary per crossing with ``pt_nm`` (position), ``seg_i``
        and ``seg_j`` (segment indices), ``t`` and ``u`` (positions along
        the segments) and ``n_hits`` (number of merged hits).

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    A figure-eight crosses itself once:

    >>> s = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    >>> eight = np.stack([np.sin(s), np.sin(s) * np.cos(s)], axis=1) * 50
    >>> crossings = sim.collect_intersections(eight, min_separation_beads=5)
    >>> len(crossings)
    1

    """

    xy = np.asarray(xy_nm, dtype=float)
    n = int(xy.shape[0])
    raw = []
    for i in range(n):
        i2 = (i + 1) % n
        p1, p2 = xy[i], xy[i2]
        for j in range(i + 1, n):
            j2 = (j + 1) % n
            if i == j or i2 == j or j2 == i:
                continue
            if _circ_sep(n, i, j) < int(min_separation_beads):
                continue
            q1, q2 = xy[j], xy[j2]
            hit, pt, t, u = _seg_intersect_2d(p1, p2, q1, q2)
            if hit:
                raw.append(
                    dict(
                        pt=np.array(pt, float),
                        seg_i=int(i),
                        seg_j=int(j),
                        t=float(t),
                        u=float(u),
                    )
                )

    clusters = merge_crossing_clusters(raw, merge_radius_nm=merge_radius_nm)
    return clusters


def merge_crossing_clusters(
    raw_list: list[dict],
    merge_radius_nm: float = 0.5,
) -> list[dict]:
    """Merge nearby intersection hits into single crossings.

    Hits are taken in order; each one joins the first cluster whose
    running centroid is within ``merge_radius_nm``, otherwise it starts a
    new cluster. The merged crossing is placed at the cluster centroid and
    keeps the segment information of the cluster's first hit.

    Parameters
    ----------
    raw_list: list[dict]
        Raw hits, each with a ``"pt"`` entry (array of shape ``(2,)``)
        and the segment information ``seg_i``, ``seg_j``, ``t``, ``u``.
    merge_radius_nm: float, optional
        Largest distance to a cluster centroid for a hit to join it.
        Defaults to 0.5.

    Returns
    -------
    list[dict]
        One dictionary per cluster with ``pt_nm``, ``seg_i``, ``seg_j``,
        ``t``, ``u`` and ``n_hits``.

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    Two hits 0.1 nm apart become one crossing:

    >>> hits = [
    ...     dict(pt=np.array([0.0, 0.0]), seg_i=1, seg_j=9, t=0.5, u=0.5),
    ...     dict(pt=np.array([0.1, 0.0]), seg_i=2, seg_j=9, t=0.0, u=0.6),
    ... ]
    >>> merged = sim.merge_crossing_clusters(hits)
    >>> len(merged), merged[0]["n_hits"]
    (1, 2)

    """

    clusters = []
    for r in raw_list:
        pt = r["pt"].astype(float)
        placed = False
        for c in clusters:
            if np.linalg.norm(pt - c["pt_sum"] / c["n"]) <= float(
                merge_radius_nm
            ):
                c["pt_sum"] += pt
                c["n"] += 1
                c["items"].append(r)
                placed = True
                break
        if not placed:
            clusters.append({"pt_sum": pt.copy(), "n": 1, "items": [r]})

    out = []
    for c in clusters:
        pt = (c["pt_sum"] / c["n"]).astype(np.float32)
        rep = c["items"][0]
        out.append(
            dict(
                pt_nm=pt,
                seg_i=int(rep["seg_i"]),
                seg_j=int(rep["seg_j"]),
                t=float(rep["t"]),
                u=float(rep["u"]),
                n_hits=int(c["n"]),
            )
        )
    return out


def boost_crossings_intersections(
    x_nm: np.ndarray,
    y_nm: np.ndarray,
    zc_nm: np.ndarray,
    crossings: list[dict] | None = None,
    min_separation_beads: int = 12,
    boost_window_beads: int = 2,
    guaranteed_offset_nm: float = 1.0,
    boost_method: Literal["additive", "absolute"] = "additive",
    boost_profile: Literal["gaussian", "hemisphere", "linear"] = "gaussian",
    boost_sigma_beads: float | None = None,
    far_clip_nm: float | None = 2.5,
    far_clip_window_beads: int = 12,
    merge_radius_nm: float = 0.5,
) -> tuple[np.ndarray, list[dict]]:
    """Raise the upper strand at every crossing.

    For each crossing, the strand that is higher (larger z) is taken as
    the upper one, and its beads within ``boost_window_beads`` of the
    crossing are raised to at least the highest bead of the lower strand
    in that window plus ``guaranteed_offset_nm``
    (``boost_method="absolute"``), or to the higher of the two strands
    plus the offset (``"additive"``), faded with
    :func:`calculate_crossing_taper_weight`. Optionally, beads far from
    any crossing are capped at ``far_clip_nm``.

    Parameters
    ----------
    x_nm: np.ndarray
        x coordinates of the beads in nm.
    y_nm: np.ndarray
        y coordinates of the beads in nm.
    zc_nm: np.ndarray
        Heights of the beads in nm.
    crossings: list[dict] | None, optional
        Precomputed crossings. Defaults to `None`, which detects them with
        :func:`collect_intersections`.
    min_separation_beads: int, optional
        Smallest separation along the ring for a crossing. Defaults to 12.
    boost_window_beads: int, optional
        Half-width of the raised window in beads. Defaults to 2.
    guaranteed_offset_nm: float, optional
        Height of the upper strand above the lower one in nm. Defaults to
        1.0.
    boost_method: "additive" | "absolute", optional
        How the offset is applied. Defaults to "additive".
    boost_profile: "gaussian" | "hemisphere" | "linear", optional
        Fade profile along the raised window. Defaults to "gaussian".
    boost_sigma_beads: float | None, optional
        Width of the Gaussian fade in beads. Defaults to `None`.
    far_clip_nm: float | None, optional
        Cap for beads far from crossings in nm; `None` disables the cap.
        Defaults to 2.5.
    far_clip_window_beads: int, optional
        Beads within this distance of a crossing are not capped. Defaults
        to 12.
    merge_radius_nm: float, optional
        Merge radius used when detecting crossings. Defaults to 0.5.

    Returns
    -------
    tuple[np.ndarray, list[dict]]
        The new heights and one record per crossing (with the indices of
        the upper and lower centre beads).

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    Raise the upper strand of a figure-eight:

    >>> s = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    >>> x, y = np.sin(s) * 50, np.sin(s) * np.cos(s) * 50
    >>> z = np.where(np.arange(40) < 20, 0.1, 0.0).astype(np.float32)
    >>> z_new, info = sim.boost_crossings_intersections(
    ...     x, y, z, min_separation_beads=5, guaranteed_offset_nm=1.0,
    ...     boost_method="absolute",
    ... )
    >>> len(info), round(float(z_new.max()), 2)
    (1, 1.1)

    """

    x = np.asarray(x_nm, dtype=np.float32)
    y = np.asarray(y_nm, dtype=np.float32)
    z = np.asarray(zc_nm, dtype=np.float32).copy()
    n = int(z.shape[0])
    if n < 5:
        return z.astype(np.float32), []

    if crossings is None:
        xy = np.stack([x, y], axis=1)
        crossings = collect_intersections(
            xy,
            min_separation_beads=int(min_separation_beads),
            merge_radius_nm=float(merge_radius_nm),
        )

    def get_segment_indices(
        center: int,
        w: int,
    ) -> np.ndarray:
        """Calculate indices in a window around a center bead.

        Parameters
        ----------
        center : int
            The index of the center bead.
        w : int
            The number of beads on either side of the center
            to include.

        Returns
        -------
        np.ndarray

        """

        return np.array(
            [(center + k) % n for k in range(-w, w + 1)], dtype=np.int32
        )

    crossing_info = []
    crossing_centers = []
    w = int(boost_window_beads)

    for c in crossings:
        i = int(c["seg_i"])
        j = int(c["seg_j"])
        i2 = (i + 1) % n
        j2 = (j + 1) % n
        ti = float(c.get("t", 0.5))
        uj = float(c.get("u", 0.5))

        bead_i = i if ti < 0.5 else i2
        bead_j = j if uj < 0.5 else j2
        zi = float((1.0 - ti) * z[i] + ti * z[i2])
        zj = float((1.0 - uj) * z[j] + uj * z[j2])

        top_center, bottom_center = (
            (bead_i, bead_j) if zi >= zj else (bead_j, bead_i)
        )
        top_seg = get_segment_indices(top_center, w)
        bottom_seg = get_segment_indices(bottom_center, w)

        bl_max = float(z[bottom_seg].max())
        tl_max = float(z[top_seg].max())

        if boost_method == "absolute":
            target_height = bl_max + float(guaranteed_offset_nm)
        else:
            target_height = max(tl_max, bl_max) + float(guaranteed_offset_nm)

        for off in range(-w, w + 1):
            k = (top_center + off) % n
            wt = float(
                calculate_crossing_taper_weight(
                    off, w, profile=boost_profile, sigma=boost_sigma_beads
                )
            )
            z[k] = float((1.0 - wt) * z[k] + wt * max(z[k], target_height))

        crossing_centers.extend([int(top_center), int(bottom_center)])
        crossing_info.append(
            dict(
                pt_nm=np.asarray(c["pt_nm"], dtype=np.float32),
                seg_i=int(i),
                seg_j=int(j),
                top_center=int(top_center),
                bottom_center=int(bottom_center),
            )
        )

    # Clip beads far from any crossing
    if crossing_centers and far_clip_nm is not None:
        centers = np.array(crossing_centers, dtype=np.int32)
        r = int(far_clip_window_beads)
        for k in range(n):
            d = int(
                np.min(
                    np.minimum(np.abs(k - centers), n - np.abs(k - centers))
                )
            )
            if d > r:
                z[k] = min(z[k], float(far_clip_nm))

    return z.astype(np.float32), crossing_info


def create_z_based_afm(
    coords: np.ndarray,
    nx: int = 800,
    ny: int = 800,
    dna_diameter_nm: float = 2.0,
    tip_radius_nm: float = 1.0,
    max_height_nm: float = 6.0,
    chain_resolution_nm: float = 0.1,
    max_radius_px: int = 128,
    radius_shrink_px: float = 0.0,
    final_blur_sigma_px: float = 0.2,
    apply_edge_taper: bool = True,
    taper_sigma_nm: float = 0.45,
    taper_floor: float = 0.1,
    add_center_ridge: bool = True,
    ridge_sigma_nm: float = 0.25,
    ridge_amp_nm: float = 0.25,
    grain_nm: float = 0.0,
    grain_sigma_px: float = 0.6,
    grain_seed: int = 1,
    enable_crossing_boost: bool = True,
    min_separation_beads: int = 12,
    boost_window_beads: int = 2,
    guaranteed_offset_nm: float = 1.0,
    boost_method: Literal["absolute", "additive"] = "absolute",
    boost_profile: Literal["gaussian", "hemisphere", "linear"] = "gaussian",
    boost_sigma_beads: float | None = None,
    crossings_precomputed: list[dict] | None = None,
    far_clip_nm: float | None = 2.5,
    far_clip_window_beads: int = 12,
    return_crossing_info: bool = False,
    return_masks: bool = True,
    extent: tuple[float, float, float, float] | None = None,
    dna_height_nm: float | None = None,
    taper_width_nm: float | None = None,
) -> tuple[np.ndarray, dict | tuple]:
    """Render an AFM height image of a chain.

    The bead coordinates are projected onto a fine internal grid
    (``chain_resolution_nm``), the centre line is drawn with the bead
    heights (crossings raised with
    :func:`boost_crossings_intersections`), widened into a strand with
    :func:`dna_shape`, attenuated towards its edges (edge taper), blurred,
    convolved with a spherical tip (:func:`AFM_tip`), given multiplicative
    grain and finally resampled to ``nx`` by ``ny`` pixels covering
    ``extent``.

    The lateral width (``dna_diameter_nm``) is an effective image width
    and the height (``dna_height_nm``) is set independently; neither
    changes the MD beads or forces.

    Parameters
    ----------
    coords: np.ndarray
        Bead coordinates with shape ``(N, 3)`` in nm.
    nx: int, optional
        Output width in pixels. Defaults to 800.
    ny: int, optional
        Output height in pixels. Defaults to 800.
    dna_diameter_nm: float, optional
        Effective lateral DNA width in nm. Defaults to 2.0.
    tip_radius_nm: float, optional
        Tip radius in nm. Defaults to 1.0.
    max_height_nm: float, optional
        Heights are clipped to this value in nm. Defaults to 6.0.
    chain_resolution_nm: float, optional
        Pixel size of the internal grid in nm. Defaults to 0.1.
    max_radius_px: int, optional
        Largest structuring-element radius in pixels. Defaults to 128.
    radius_shrink_px: float, optional
        Kept for compatibility; not used. Defaults to 0.0.
    final_blur_sigma_px: float, optional
        Gaussian blur before tip convolution, in output pixels. Defaults
        to 0.2.
    apply_edge_taper: bool, optional
        Lower the strand towards its edges. Defaults to `True`.
    taper_sigma_nm: float, optional
        Width of the edge taper in nm. Defaults to 0.45.
    taper_floor: float, optional
        Smallest taper factor (0 to 1). Defaults to 0.1.
    add_center_ridge: bool, optional
        Add a narrow ridge along the centre line. Defaults to `True`.
    ridge_sigma_nm: float, optional
        Width of the ridge in nm. Defaults to 0.25.
    ridge_amp_nm: float, optional
        Height of the ridge in nm. Defaults to 0.25.
    grain_nm: float, optional
        Relative amplitude of the multiplicative grain. Defaults to 0.0.
    grain_sigma_px: float, optional
        Smoothing of the grain in pixels. Defaults to 0.6.
    grain_seed: int, optional
        Random seed of the grain. Defaults to 1.
    enable_crossing_boost: bool, optional
        Raise the upper strand at crossings. Defaults to `True`.
    min_separation_beads: int, optional
        Smallest separation along the ring for a crossing. Defaults to 12.
    boost_window_beads: int, optional
        Half-width of the raised window in beads. Defaults to 2.
    guaranteed_offset_nm: float, optional
        Height of the upper strand above the lower one in nm. Defaults to
        1.0.
    boost_method: "absolute" | "additive", optional
        How the crossing offset is applied. Defaults to "absolute".
    boost_profile: "gaussian" | "hemisphere" | "linear", optional
        Fade profile of the raised window. Defaults to "gaussian".
    boost_sigma_beads: float | None, optional
        Width of the Gaussian fade in beads. Defaults to `None`.
    crossings_precomputed: list[dict] | None, optional
        Crossings to boost. Defaults to `None`, which detects them.
    far_clip_nm: float | None, optional
        Cap for beads far from crossings in nm. Defaults to 2.5.
    far_clip_window_beads: int, optional
        Beads within this distance of a crossing are not capped. Defaults
        to 12.
    return_crossing_info: bool, optional
        Add the crossing records to the returned dictionary. Defaults to
        `False`.
    return_masks: bool, optional
        Return a dictionary of masks and metadata instead of the extent.
        Defaults to `True`.
    extent: tuple[float, float, float, float] | None, optional
        Output extent ``(xmin, xmax, ymin, ymax)`` in nm. Defaults to
        `None`, which uses the chain's bounding box plus 2 nm.
    dna_height_nm: float | None, optional
        Height of the strand in nm. Defaults to `None`, which uses
        ``dna_diameter_nm``.
    taper_width_nm: float | None, optional
        Width of the outer band attenuated by the edge taper in nm, at
        most half the width. Defaults to `None`, which tapers from the
        centre line (non-MD renderer).

    Returns
    -------
    tuple[np.ndarray, dict | tuple]
        The height image (``ny`` by ``nx``, nm) and either a dictionary
        with masks and metadata (``return_masks=True``) or the extent.

    Raises
    ------
    ValueError
        If the width, height or taper settings are not valid.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    Render a 115 nm non-MD ring at 1 nm per pixel:

    >>> coords = sim.make_non_md_ring(1, 115.26)
    >>> image, info = sim.create_z_based_afm(
    ...     coords, nx=128, ny=128, dna_diameter_nm=4.47, dna_height_nm=2.3,
    ...     tip_radius_nm=7.0, chain_resolution_nm=0.5,
    ...     extent=(-63.5, 63.5, -63.5, 63.5),
    ... )
    >>> image.shape, round(info["output_nm_per_px"], 3)
    ((128, 128), 1.0)

    """

    dna_height_nm = float(
        dna_diameter_nm if dna_height_nm is None else dna_height_nm
    )
    if not np.isfinite(dna_height_nm) or dna_height_nm <= 0:
        raise ValueError("dna_height_nm must be finite and positive")
    if not np.isfinite(dna_diameter_nm) or dna_diameter_nm <= 0:
        raise ValueError("dna_diameter_nm must be finite and positive")
    if (
        taper_width_nm is not None
        and not 0 <= taper_width_nm <= dna_diameter_nm / 2
    ):
        raise ValueError(
            "taper_width_nm must be between zero and the tube radius"
        )
    if not 0 <= taper_floor <= 1 or taper_sigma_nm <= 0:
        raise ValueError("Invalid taper floor/sigma")
    x, y, z = coords.T
    nx_out = int(nx)
    ny_out = int(ny)
    chain_resolution_nm = max(float(chain_resolution_nm), 1e-06)
    if extent is None:
        output_xmin = float(x.min() - 2.0)
        output_xmax = float(x.max() + 2.0)
        output_ymin = float(y.min() - 2.0)
        output_ymax = float(y.max() + 2.0)
    else:
        output_xmin, output_xmax, output_ymin, output_ymax = extent
    output_extent = (
        float(output_xmin),
        float(output_xmax),
        float(output_ymin),
        float(output_ymax),
    )
    output_px_eff_x = (output_xmax - output_xmin) / max(nx_out - 1, 1)
    output_px_eff_y = (output_ymax - output_ymin) / max(ny_out - 1, 1)
    output_p_eff = 0.5 * (output_px_eff_x + output_px_eff_y)
    render_pad_nm = max(
        2.0 * float(dna_diameter_nm),
        2.0 * float(tip_radius_nm),
        4.0 * chain_resolution_nm,
    )
    render_extent = (
        float(output_xmin - render_pad_nm),
        float(output_xmax + render_pad_nm),
        float(output_ymin - render_pad_nm),
        float(output_ymax + render_pad_nm),
    )
    nx_eff, ny_eff, p_eff = compute_internal_render_grid(
        render_extent, chain_resolution_nm, min_size=16
    )
    xmin, xmax, ymin, ymax = render_extent
    ix = np.clip(
        ((x - xmin) / (xmax - xmin) * (nx_eff - 1)).astype(np.int32),
        0,
        nx_eff - 1,
    )
    iy = np.clip(
        ((y - ymin) / (ymax - ymin) * (ny_eff - 1)).astype(np.int32),
        0,
        ny_eff - 1,
    )
    zc = (z - z.min()).astype(np.float32)
    crossing_info = []
    if enable_crossing_boost:
        zc, crossing_info = boost_crossings_intersections(
            x.astype(np.float32),
            y.astype(np.float32),
            zc,
            crossings=crossings_precomputed,
            min_separation_beads=int(min_separation_beads),
            boost_window_beads=int(boost_window_beads),
            guaranteed_offset_nm=float(guaranteed_offset_nm),
            boost_method=str(boost_method),
            boost_profile=str(boost_profile),
            boost_sigma_beads=boost_sigma_beads,
            far_clip_nm=far_clip_nm,
            far_clip_window_beads=int(far_clip_window_beads),
        )
    z_line = np.zeros((ny_eff, nx_eff), dtype=np.float32)
    line_mask = np.zeros((ny_eff, nx_eff), dtype=bool)
    npts = len(ix)
    for k in range(npts):
        k2 = (k + 1) % npts
        x0, y0, z0 = (int(ix[k]), int(iy[k]), float(zc[k]))
        x1, y1, z1 = (int(ix[k2]), int(iy[k2]), float(zc[k2]))
        n_seg = int(max(abs(x1 - x0), abs(y1 - y0)) + 1)
        if n_seg <= 1:
            z_line[y0, x0] = max(z_line[y0, x0], z0)
            line_mask[y0, x0] = True
            continue
        xs = np.linspace(x0, x1, n_seg).astype(np.int32)
        ys = np.linspace(y0, y1, n_seg).astype(np.int32)
        zs = np.linspace(z0, z1, n_seg).astype(np.float32)
        if xs.size > 1:
            keep = np.ones(xs.shape[0], dtype=bool)
            keep[1:] = (xs[1:] != xs[:-1]) | (ys[1:] != ys[:-1])
            xs = xs[keep]
            ys = ys[keep]
            zs = zs[keep]
        z_line[ys, xs] = np.maximum(z_line[ys, xs], zs)
        line_mask[ys, xs] = True
    top_cross_line_mask = np.zeros_like(line_mask)
    bottom_cross_line_mask = np.zeros_like(line_mask)
    if crossing_info:
        crossing_window = int(boost_window_beads)
        for crossing in crossing_info:
            if "top_center" not in crossing:
                continue
            if "bottom_center" not in crossing:
                continue
            top_center = int(crossing["top_center"])
            bottom_center = int(crossing["bottom_center"])
            for off in range(-crossing_window, crossing_window + 1):
                for center, mask in (
                    (top_center, top_cross_line_mask),
                    (bottom_center, bottom_cross_line_mask),
                ):
                    k0 = (center + off) % npts
                    k1 = (k0 + 1) % npts
                    x0, y0 = (int(ix[k0]), int(iy[k0]))
                    x1, y1 = (int(ix[k1]), int(iy[k1]))
                    n_seg = int(max(abs(x1 - x0), abs(y1 - y0)) + 1)
                    if n_seg <= 1:
                        if 0 <= x0 < nx_eff and 0 <= y0 < ny_eff:
                            mask[y0, x0] = True
                        continue
                    xs = np.linspace(x0, x1, n_seg).astype(np.int32)
                    ys = np.linspace(y0, y1, n_seg).astype(np.int32)
                    valid = (
                        (xs >= 0) & (xs < nx_eff) & (ys >= 0) & (ys < ny_eff)
                    )
                    xs = xs[valid]
                    ys = ys[valid]
                    if xs.size:
                        mask[ys, xs] = True
    r_eff = max(0.5 * float(dna_diameter_nm), 0.2)
    fp_dna, cap = dna_shape(
        radius_nm=r_eff,
        chain_resolution_nm=p_eff,
        max_radius_px=max_radius_px,
        height_nm=dna_height_nm,
    )
    surface = grey_dilation(z_line, footprint=fp_dna, structure=cap).astype(
        np.float32
    )
    dna_body_region = (
        grey_dilation(line_mask.astype(np.uint8), footprint=fp_dna) > 0
    )
    surface[~dna_body_region] = 0.0
    top_cross_region = (
        grey_dilation(top_cross_line_mask.astype(np.uint8), footprint=fp_dna)
        > 0
    )
    bottom_cross_region = (
        grey_dilation(
            bottom_cross_line_mask.astype(np.uint8), footprint=fp_dna
        )
        > 0
    )
    if apply_edge_taper and np.any(line_mask):
        taper_dist_px = distance_transform_edt(~line_mask).astype(np.float32)
        taper_dist_nm = taper_dist_px * np.float32(p_eff)
        sig = max(float(taper_sigma_nm), 1e-06)
        # width=None reproduces the old Gaussian attenuation from the
        # centre line.
        # Otherwise taper only the outer radial band of width taper_width_nm.
        band = r_eff if taper_width_nm is None else float(taper_width_nm)
        taper_start_nm = r_eff - band
        edge_distance_nm = np.maximum(taper_dist_nm - taper_start_nm, 0.0)
        factor = taper_floor + (1.0 - taper_floor) * np.exp(
            -(edge_distance_nm**2) / (2.0 * sig**2)
        )
        intersection_zone = bottom_cross_region & top_cross_region
        blur_sigma_px = max(1.0, 0.35 * float(dna_diameter_nm) / p_eff)
        weight = gaussian_filter(
            intersection_zone.astype(np.float32), sigma=blur_sigma_px
        )
        smooth_factor = factor * (1.0 - weight) + 1.0 * weight
        surface[dna_body_region] *= smooth_factor[dna_body_region]
    if add_center_ridge and ridge_amp_nm > 0 and np.any(line_mask):
        ridge_dist_px = distance_transform_edt(~line_mask).astype(np.float32)
        ridge_dist_nm = ridge_dist_px * np.float32(p_eff)
        rs = max(float(ridge_sigma_nm), 1e-06)
        ridge = np.exp(-(ridge_dist_nm**2) / (2.0 * rs**2))
        surface[dna_body_region] += ridge_amp_nm * ridge[dna_body_region]
    np.clip(surface, 0, max_height_nm, out=surface)
    if final_blur_sigma_px > 0:
        surface = gaussian_filter(
            surface, sigma=float(final_blur_sigma_px)
        ).astype(np.float32)
    fp_tip, tip_struct = AFM_tip(
        tip_radius_nm=float(tip_radius_nm),
        chain_resolution_nm=p_eff,
        max_radius_px=max_radius_px,
    )
    surface = grey_dilation(
        surface, footprint=fp_tip, structure=tip_struct
    ).astype(np.float32)
    apparent_region = (
        grey_dilation(dna_body_region.astype(np.uint8), footprint=fp_tip) > 0
    )
    surface[~apparent_region] = 0.0
    np.clip(surface, 0, max_height_nm, out=surface)
    if grain_nm and grain_nm > 0:
        rng = np.random.default_rng(int(grain_seed))
        noise = rng.normal(0.0, 1.0, size=surface.shape).astype(np.float32)
        if grain_sigma_px and grain_sigma_px > 0:
            noise = gaussian_filter(
                noise, sigma=float(grain_sigma_px)
            ).astype(np.float32)
        noise_std = noise[apparent_region].std()
        std = (
            float(noise_std)
            if np.any(apparent_region)
            else float(noise.std())
        )
        if std > 1e-09:
            noise /= np.float32(std)
        surface[apparent_region] *= (
            1.0 + np.float32(grain_nm) * noise[apparent_region]
        )
        np.clip(surface, 0, max_height_nm, out=surface)
    x_scale = (nx_eff - 1) / max(xmax - xmin, 1e-12)
    y_scale = (ny_eff - 1) / max(ymax - ymin, 1e-12)
    crop_x0 = int(np.floor((output_xmin - xmin) * x_scale))
    crop_x1 = int(np.ceil((output_xmax - xmin) * x_scale)) + 1
    crop_y0 = int(np.floor((output_ymin - ymin) * y_scale))
    crop_y1 = int(np.ceil((output_ymax - ymin) * y_scale)) + 1
    crop_x0 = max(0, min(nx_eff - 1, crop_x0))
    crop_x1 = max(crop_x0 + 1, min(nx_eff, crop_x1))
    crop_y0 = max(0, min(ny_eff - 1, crop_y0))
    crop_y1 = max(crop_y0 + 1, min(ny_eff, crop_y1))
    crop_slice = np.s_[crop_y0:crop_y1, crop_x0:crop_x1]
    out_shape = (ny_out, nx_out)
    surface = resize_image_to_shape(
        surface[crop_slice], out_shape=out_shape, order=1
    ).astype(np.float32)
    line_mask = (
        resize_image_to_shape(
            line_mask[crop_slice].astype(np.float32),
            out_shape=out_shape,
            order=0,
        )
        > 0.5
    )
    top_cross_line_mask = (
        resize_image_to_shape(
            top_cross_line_mask[crop_slice].astype(np.float32),
            out_shape=out_shape,
            order=0,
        )
        > 0.5
    )
    bottom_cross_line_mask = (
        resize_image_to_shape(
            bottom_cross_line_mask[crop_slice].astype(np.float32),
            out_shape=out_shape,
            order=0,
        )
        > 0.5
    )
    top_cross_region = (
        resize_image_to_shape(
            top_cross_region[crop_slice].astype(np.float32),
            out_shape=out_shape,
            order=1,
        )
        > 0.25
    )
    apparent_region = (
        resize_image_to_shape(
            apparent_region[crop_slice].astype(np.float32),
            out_shape=out_shape,
            order=1,
        )
        > 0.25
    )
    mask_radius_px = max(
        1,
        int(np.ceil(0.5 * float(dna_diameter_nm) / max(output_p_eff, 1e-12))),
    )
    yy, xx = np.ogrid[
        -mask_radius_px : mask_radius_px + 1,
        -mask_radius_px : mask_radius_px + 1,
    ]
    dna_mask_fp = xx * xx + yy * yy <= mask_radius_px * mask_radius_px
    dna_body_region = (
        make_dna_mask_from_beads(
            coords,
            output_extent,
            nx_out,
            ny_out,
            dna_diameter_nm=dna_diameter_nm,
        )
        > 0
    )
    top_cross_region_out = (
        grey_dilation(
            top_cross_line_mask.astype(np.uint8), footprint=dna_mask_fp
        )
        > 0
    )
    bottom_cross_region_out = (
        grey_dilation(
            bottom_cross_line_mask.astype(np.uint8), footprint=dna_mask_fp
        )
        > 0
    )
    crossing_region = top_cross_region_out | bottom_cross_region_out
    if return_masks:
        debug = {
            "line_mask": line_mask,
            "top_cross_line_mask": top_cross_line_mask,
            "bottom_cross_line_mask": bottom_cross_line_mask,
            "dna_region": dna_body_region,
            "dna_body_region": dna_body_region,
            "apparent_region": apparent_region,
            "top_cross_region": top_cross_region_out,
            "bottom_cross_region": bottom_cross_region_out,
            "crossing_region": crossing_region,
            "extent": output_extent,
            "output_extent": output_extent,
            "render_extent": render_extent,
            "p_nm_per_px": float(output_p_eff),
            "output_nm_per_px": float(output_p_eff),
            "chain_resolution_nm": float(p_eff),
            "requested_chain_resolution_nm": float(chain_resolution_nm),
            "tip_radius_nm": float(tip_radius_nm),
            "dna_diameter_nm": float(dna_diameter_nm),
            "dna_height_nm": dna_height_nm,
            "taper_width_nm": taper_width_nm,
            "taper_sigma_nm": float(taper_sigma_nm),
            "mask_width_multiplier": 1.0,
        }
        if return_crossing_info:
            debug["crossings"] = crossing_info
        return (surface, debug)
    return (surface, output_extent)


# ────────────────────────────────────────────────────────────────────────────
# Noise
# ────────────────────────────────────────────────────────────────────────────


def robust_range(a: np.ndarray | list) -> tuple[float, float, float]:
    """Return the 1st and 99th percentiles of the finite values.

    Ignoring the extreme 1 % on each side and non-finite values (NaN,
    infinity) gives a range that is not dominated by outliers.

    Parameters
    ----------
    a: np.ndarray | list
        Input values.

    Returns
    -------
    tuple[float, float, float]
        The 1st percentile, the 99th percentile and their difference.

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    >>> low, high, spread = sim.robust_range(np.arange(101.0))
    >>> low, high, spread
    (1.0, 99.0, 98.0)

    """

    a = np.asarray(a, dtype=np.float32)
    p1, p99 = np.percentile(a[np.isfinite(a)], [1, 99])
    return float(p1), float(p99), float(p99 - p1)


def looks_like_afm_height(img: np.ndarray) -> bool:
    """Check whether an array looks like an AFM height map.

    The robust range (:func:`robust_range`) must be finite, positive and
    below 1e9; flat, all-NaN or wildly scaled arrays are rejected.

    Parameters
    ----------
    img: np.ndarray
        Candidate height map.

    Returns
    -------
    bool
        `True` if the array has a finite, positive and sane range.

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    >>> noise = np.random.default_rng(0).normal(size=(8, 8))
    >>> sim.looks_like_afm_height(noise)
    True
    >>> sim.looks_like_afm_height(np.zeros((8, 8)))
    False

    """

    p1, p99, rng = robust_range(img)
    return np.isfinite(rng) and 0 < rng <= 1e9


def maybe_to_nm(img: np.ndarray) -> tuple[np.ndarray, str]:
    """Convert heights from metres to nm when they look like metres.

    Heights whose robust range is below 1e-6 are assumed to be in metres
    and multiplied by 1e9.

    Parameters
    ----------
    img: np.ndarray
        Height map.

    Returns
    -------
    tuple[np.ndarray, str]
        The (possibly converted) height map and a note on the units
        (``"meters->nm (auto)"`` or ``"as-is (nm or counts)"``).

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    >>> heights_m = np.linspace(0, 2e-9, 100).reshape(10, 10)
    >>> heights_nm, note = sim.maybe_to_nm(heights_m)
    >>> round(float(heights_nm.max()), 3), note
    (2.0, 'meters->nm (auto)')

    """

    p1, p99, rng = robust_range(img)
    mx = max(abs(p1), abs(p99))
    if mx < 1e-3:
        return img.astype(np.float32) * 1e9, "meters->nm (auto)"
    return img.astype(np.float32), "as-is (nm or counts)"


def parse_blocks(
    filename: str | Path,
    max_blocks: int = 128,
) -> tuple[bytes, str, list[dict]]:
    """Read a Bruker ``.spm`` file and parse its data-block headers.

    The ASCII header at the start of the file is decoded, and every image
    block is described by its data offset, length, bytes per pixel, size
    in pixels, Z scale and Z offset. Duplicate blocks are removed.

    Parameters
    ----------
    filename: str | Path
        Path of the ``.spm`` file.
    max_blocks: int, optional
        Largest number of blocks to parse. Defaults to 128.

    Returns
    -------
    tuple[bytes, str, list[dict]]
        The raw file content, the decoded header and one dictionary per
        image block.

    Raises
    ------
    RuntimeError
        If the header contains no ``Data offset`` entry or no block can be
        parsed.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> raw, header, blocks = sim.parse_blocks(
    ...     "../common-assets/noise/20240411_blank_water.0_00000.spm"
    ... )  # doctest: +SKIP
    >>> len(blocks) > 0  # doctest: +SKIP
    True

    """

    raw = open(filename, "rb").read()
    i_end = raw.find(b"\x1a")
    if i_end == -1:
        i_end = min(len(raw), 400_000)
    header = raw[:i_end].decode("latin1", errors="ignore")

    matches = [m.start() for m in re.finditer(r"Data offset", header)]
    if not matches:
        raise RuntimeError("No 'Data offset' found in header.")

    blocks = []
    for k, pos in enumerate(matches[:max_blocks]):
        win = header[max(0, pos - 2500) : min(len(header), pos + 2500)]

        def grab_int(key):
            m = re.search(rf"{re.escape(key)}\s*:\s*([0-9]+)", win)
            return int(m.group(1)) if m else None

        def grab_float(key):
            pattern = (
                rf"{re.escape(key)}\s*:\s*"
                r"([+-]?[0-9]*\.?[0-9]+(?:[Ee][+-]?[0-9]+)?)"
            )
            m = re.search(pattern, win)
            return float(m.group(1)) if m else None

        name_candidate = re.findall(
            r"@[^:\n\r]{0,40}:\s*([^\n\r]{1,80})", win
        )
        name = name_candidate[-1].strip() if name_candidate else None

        off = grab_int("Data offset")
        length = grab_int("Data length")
        bpp = grab_int("Bytes/pixel")
        nx_b = grab_int("Samps/line")
        ny_b = grab_int("Number of lines")
        zscale = grab_float("Z scale")
        zoff = grab_float("Z offset")

        if None in (off, length, bpp, nx_b, ny_b):
            continue
        blocks.append(
            dict(
                name=name or f"block_{k}",
                off=off,
                length=length,
                bpp=bpp,
                nx=nx_b,
                ny=ny_b,
                zscale=zscale,
                zoff=zoff,
            )
        )

    uniq, seen = [], set()
    for b in blocks:
        if b["off"] not in seen:
            uniq.append(b)
            seen.add(b["off"])
    if not uniq:
        raise RuntimeError(
            "Found 'Data offset' strings" " but no parseable blocks."
        )
    return raw, header, uniq


def read_block_candidates(
    raw: bytes,
    b: dict,
) -> list[tuple[str, np.ndarray]]:
    """Decode one data block with every plausible integer type.

    The block is decoded as signed and unsigned integers of its bytes per
    pixel, little- and big-endian, reshaped to its size and scaled with
    its Z scale and Z offset when they are known.

    Parameters
    ----------
    raw: bytes
        Raw content of the ``.spm`` file.
    b: dict
        One block description from :func:`parse_blocks`.

    Returns
    -------
    list[tuple[str, np.ndarray]]
        One ``(data type, image)`` pair per decoding.

    Raises
    ------
    RuntimeError
        If the block extends past the end of the file.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> raw, header, blocks = sim.parse_blocks(
    ...     "../common-assets/noise/20240411_blank_water.0_00000.spm"
    ... )  # doctest: +SKIP
    >>> candidates = sim.read_block_candidates(
    ...     raw, blocks[0]
    ... )  # doctest: +SKIP
    >>> candidates[0][1].shape  # doctest: +SKIP
    (512, 512)

    """

    off, length, bpp = b["off"], b["length"], b["bpp"]
    nx_b, ny_b = b["nx"], b["ny"]
    blob = raw[off : off + length]
    if len(blob) < length:
        raise RuntimeError("Truncated block.")

    n = nx_b * ny_b
    cands = []
    dtype_map = {
        2: ["<i2", ">i2", "<u2", ">u2"],
        4: ["<i4", ">i4", "<f4", ">f4"],
        1: ["u1"],
    }
    for dt in dtype_map.get(bpp, []):
        arr = np.frombuffer(blob, dtype=np.dtype(dt), count=n)
        if arr.size != n:
            continue
        cands.append((dt, arr.reshape(ny_b, nx_b).astype(np.float32)))

    out = []
    for dt, img in cands:
        z = img
        if b.get("zscale") is not None:
            z = z * np.float32(b["zscale"])
            if b.get("zoff") is not None:
                z = z + np.float32(b["zoff"])
        out.append((dt, z))
    return out


def choose_best_height_image(
    raw: bytes,
    blocks: list[dict],
    verbose: bool = True,
) -> tuple[dict, np.ndarray]:
    """Pick the decoded block that most plausibly is the height map.

    All blocks and all decodings from :func:`read_block_candidates` are
    tested with :func:`looks_like_afm_height`; among the plausible ones,
    the decoding with the largest log-variance (most topographic detail)
    is chosen.

    Parameters
    ----------
    raw: bytes
        Raw content of the ``.spm`` file.
    blocks: list[dict]
        Block descriptions from :func:`parse_blocks`.
    verbose: bool, optional
        Print the chosen block and its range. Defaults to `True`.

    Returns
    -------
    tuple[dict, np.ndarray]
        The chosen block description and its decoded image.

    Raises
    ------
    RuntimeError
        If no block decodes to a plausible height map.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> raw, header, blocks = sim.parse_blocks(
    ...     "../common-assets/noise/20240411_blank_water.0_00000.spm"
    ... )  # doctest: +SKIP
    >>> block, image = sim.choose_best_height_image(
    ...     raw, blocks
    ... )  # doctest: +SKIP
    >>> image.shape  # doctest: +SKIP
    (512, 512)

    """

    best, best_score = None, -np.inf
    for b in blocks:
        try:
            for dt, img in read_block_candidates(raw, b):
                if not looks_like_afm_height(img):
                    continue
                p1, p99, rng = robust_range(img)
                var = float(np.var(img))
                score = np.log10(var + 1e-9) - 0.001 * max(rng - 1e6, 0)
                if score > best_score:
                    best_score = score
                    best = (b, dt, img, (p1, p99, rng, var))
        except Exception:
            continue

    if best is None:
        raise RuntimeError(
            "Could not find a" "plausible AFM height image decode."
        )
    b, dt, img, stats = best
    if verbose:
        p1, p99, rng, var = stats
        print(
            f"Chosen block: {b['name']!r} {b['ny']}x{b['nx']} "
            f"bpp={b['bpp']} decode={dt}"
        )
        print(
            f"Robust p1={p1:.3g}, p99={p99:.3g}, "
            f"range={rng:.3g}, var={var:.3g}"
        )
    return b, img


def plane_remove(
    z: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Fit and subtract a tilted plane from a height map.

    The plane ``z = a x + b y + c`` is fitted by least squares, using only
    the pixels where ``mask`` is `True` if a mask is given (for example the
    bare substrate).

    Parameters
    ----------
    z: np.ndarray
        2-D height map.
    mask: np.ndarray | None, optional
        Boolean array of the same shape selecting the pixels used for the
        fit. Defaults to `None` (all pixels).

    Returns
    -------
    np.ndarray
        The height map with the plane subtracted.

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    A pure tilt is removed completely:

    >>> yy, xx = np.mgrid[0:16, 0:16]
    >>> flat = sim.plane_remove(0.1 * xx + 0.2 * yy + 3.0)
    >>> bool(np.allclose(flat, 0.0, atol=1e-5))
    True

    """

    ny, nx = z.shape
    Y, X = np.mgrid[0:ny, 0:nx]
    A = np.c_[X.ravel(), Y.ravel(), np.ones(X.size)]
    b = z.ravel()
    if mask is not None:
        m = mask.ravel().astype(bool)
        A, b = A[m], b[m]
    C, *_ = np.linalg.lstsq(A, b, rcond=None)
    plane = (C[0] * X + C[1] * Y + C[2]).astype(np.float32)
    return (z - plane).astype(np.float32)


def line_flatten_median(
    z: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Subtract the median of every scan line.

    AFM images often show horizontal stripes from drift between scan
    lines; shifting every row to zero median removes them. With a mask,
    only the masked pixels of each row are used for its median.

    Parameters
    ----------
    z: np.ndarray
        2-D height map.
    mask: np.ndarray | None, optional
        Boolean array of the same shape selecting background pixels.
        Defaults to `None` (all pixels).

    Returns
    -------
    np.ndarray
        The flattened height map.

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    >>> striped = np.repeat(np.arange(4.0)[:, None], 5, axis=1)
    >>> float(sim.line_flatten_median(striped).max())
    0.0

    """

    out = z.astype(np.float32, copy=True)
    for i in range(out.shape[0]):
        if mask is None or not np.any(mask[i]):
            med = np.median(out[i])
        else:
            med = np.median(out[i, mask[i]])
        out[i] -= np.float32(med)
    return out


def feature_mask(
    z: np.ndarray,
    smooth_sigma_px: float = 3.0,
    thresh_sigma: float = 3.0,
    dilate_px: int = 4,
) -> np.ndarray:
    """Mark pixels that stand out from the local background.

    The image is smoothed with a Gaussian of ``smooth_sigma_px``; pixels
    whose smoothed height exceeds the median by more than ``thresh_sigma``
    robust standard deviations (from the median absolute deviation) are
    marked and the mask is dilated by ``dilate_px``.

    Parameters
    ----------
    z: np.ndarray
        2-D height map.
    smooth_sigma_px: float, optional
        Gaussian smoothing in pixels. Defaults to 3.0.
    thresh_sigma: float, optional
        Threshold in robust standard deviations. Defaults to 3.0.
    dilate_px: int, optional
        Dilation radius in pixels. Defaults to 4.

    Returns
    -------
    np.ndarray
        Boolean mask, `True` on features.

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    A bump in noise is detected:

    >>> z = np.random.default_rng(0).normal(0, 0.05, (64, 64))
    >>> z[30:34, 30:34] += 3.0
    >>> mask = sim.feature_mask(z)
    >>> bool(mask[32, 32]), bool(mask[5, 5])
    (True, False)

    """

    z_s = gaussian_filter(z, sigma=float(smooth_sigma_px)).astype(np.float32)
    resid = (z - z_s).astype(np.float32)
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    sig = 1.4826 * mad if mad > 1e-12 else float(np.std(resid) + 1e-12)
    feat = np.abs(resid - med) > (float(thresh_sigma) * sig)
    if dilate_px and dilate_px > 0:
        r = int(dilate_px)
        yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
        fp = (xx * xx + yy * yy) <= r * r
        feat = grey_dilation(feat.astype(np.uint8), footprint=fp) > 0
    return feat


def inpaint_simple(
    z: np.ndarray,
    mask: np.ndarray,
    smooth_sigma_px: float = 8.0,
) -> np.ndarray:
    """Replace masked pixels by a smooth estimate of the background.

    Parameters
    ----------
    z: np.ndarray
        2-D height map.
    mask: np.ndarray
        Boolean mask, `True` where pixels are replaced.
    smooth_sigma_px: float, optional
        Gaussian smoothing of the background estimate in pixels; larger
        values give a smoother fill. Defaults to 8.0.

    Returns
    -------
    np.ndarray
        The height map with the masked pixels filled.

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    >>> z = np.zeros((32, 32))
    >>> z[10:12, 10:12] = 5.0
    >>> filled = sim.inpaint_simple(z, z > 1)
    >>> float(filled[10, 10]) < 5.0
    True

    """

    bg = gaussian_filter(z, sigma=float(smooth_sigma_px)).astype(np.float32)
    out = z.copy()
    out[mask] = bg[mask]
    return out


def bandpass_noise(
    z: np.ndarray,
    low_sigma_px: float = 1.0,
    high_sigma_px: float = 30.0,
) -> np.ndarray:
    """Band-pass filter a height map (difference of Gaussians).

    The image blurred with ``low_sigma_px`` (removes pixel noise) minus
    the image blurred with ``high_sigma_px`` (removes slow trends such as
    tilt or bowing).

    Parameters
    ----------
    z: np.ndarray
        2-D height map.
    low_sigma_px: float, optional
        Small Gaussian width in pixels. Defaults to 1.0.
    high_sigma_px: float, optional
        Large Gaussian width in pixels. Defaults to 30.0.

    Returns
    -------
    np.ndarray
        The band-passed height map.

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    A constant offset is removed:

    >>> out = sim.bandpass_noise(np.full((64, 64), 7.0))
    >>> bool(np.allclose(out, 0.0))
    True

    """

    low = gaussian_filter(z, sigma=float(low_sigma_px)).astype(np.float32)
    high = gaussian_filter(z, sigma=float(high_sigma_px)).astype(np.float32)
    return (low - high).astype(np.float32)


def extract_topostats_like_noise(
    z_in: np.ndarray,
    median_size: int = 3,
    bg_quantile: float = 40.0,
    feature_sigma_px: float = 3.0,
    feature_thresh_sigma: float = 3.0,
    feature_dilate_px: int = 4,
    inpaint_sigma_px: float = 10.0,
    band_low_sigma_px: float = 0.8,
    band_high_sigma_px: float = 35.0,
    target_rms_nm: float | None = None,
    apply_plane_remove: bool = True,
    apply_line_flatten: bool = True,
    apply_bandpass: bool = True,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Extract the substrate noise texture from a height map.

    The image is median filtered, levelled (plane and line flattening on
    a rough background estimate), cleared of features
    (:func:`feature_mask`, :func:`inpaint_simple`) and band-pass filtered
    (:func:`bandpass_noise`). The resulting zero-mean texture can be
    rescaled to ``target_rms_nm``.

    Parameters
    ----------
    z_in: np.ndarray
        2-D height map in nm (usually a blank scan).
    median_size: int, optional
        Median filter size in pixels. Defaults to 3.
    bg_quantile: float, optional
        Percentile below which pixels count as background for the first
        levelling. Defaults to 40.0.
    feature_sigma_px: float, optional
        Smoothing used to detect features. Defaults to 3.0.
    feature_thresh_sigma: float, optional
        Feature threshold in robust standard deviations. Defaults to 3.0.
    feature_dilate_px: int, optional
        Dilation of the feature mask in pixels. Defaults to 4.
    inpaint_sigma_px: float, optional
        Smoothing used to fill features. Defaults to 10.0.
    band_low_sigma_px: float, optional
        Small band-pass width in pixels. Defaults to 0.8.
    band_high_sigma_px: float, optional
        Large band-pass width in pixels. Defaults to 35.0.
    target_rms_nm: float | None, optional
        Rescale the texture to this RMS in nm. Defaults to `None`.
    apply_plane_remove: bool, optional
        Remove a tilted plane. Defaults to `True`.
    apply_line_flatten: bool, optional
        Subtract the median of every line. Defaults to `True`.
    apply_bandpass: bool, optional
        Apply the band-pass filter. Defaults to `True`.

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray, np.ndarray]
        The texture RMS in nm, the zero-mean texture, the levelled image
        and the feature mask.

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    >>> z = np.random.default_rng(1).normal(0, 0.3, (128, 128))
    >>> rms, texture, flat, features = sim.extract_topostats_like_noise(
    ...     z, target_rms_nm=0.1
    ... )
    >>> round(rms, 3), texture.shape
    (0.1, (128, 128))

    """

    z = z_in.astype(np.float32)
    if median_size and median_size > 1:
        z = median_filter(z, size=int(median_size)).astype(np.float32)

    q = np.percentile(z, float(bg_quantile))
    bg0 = z <= q
    z_flat = z.copy()
    if apply_plane_remove:
        z_flat = plane_remove(z_flat, mask=bg0)

    if apply_line_flatten:
        z_flat = line_flatten_median(z_flat, mask=bg0)

    feat = feature_mask(
        z_flat,
        smooth_sigma_px=feature_sigma_px,
        thresh_sigma=feature_thresh_sigma,
        dilate_px=feature_dilate_px,
    )
    z_bg = inpaint_simple(
        z_flat,
        feat,
        smooth_sigma_px=inpaint_sigma_px,
    )
    if apply_bandpass:
        tex = bandpass_noise(
            z_bg,
            low_sigma_px=band_low_sigma_px,
            high_sigma_px=band_high_sigma_px,
        )
    else:
        tex = z_bg.astype(np.float32)

    bg = ~feat
    rms = float(np.std(tex[bg])) if np.any(bg) else float(np.std(tex))
    if target_rms_nm is not None:
        target = float(target_rms_nm)
        if rms > 1e-12:
            tex *= target / rms
            rms = target
    return (
        rms,
        tex.astype(np.float32),
        z_flat.astype(np.float32),
        feat.astype(bool),
    )


def tile_or_crop(
    tex: np.ndarray,
    out_shape: tuple[int, int],
    seed: int | None = 0,
) -> np.ndarray:
    """Cut a crop of a given shape from a texture, tiling if needed.

    A texture smaller than ``out_shape`` is repeated until it is large
    enough; the crop position is drawn with ``seed`` so different samples
    get different crops.

    Parameters
    ----------
    tex: np.ndarray
        2-D texture.
    out_shape: tuple[int, int]
        Shape of the crop ``(rows, columns)``.
    seed: int | None, optional
        Random seed of the crop position. Defaults to 0.

    Returns
    -------
    np.ndarray
        Crop of shape ``out_shape``.

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    >>> sim.tile_or_crop(np.ones((10, 10)), (25, 30), seed=1).shape
    (25, 30)

    """

    ty, tx = tex.shape
    oy, ox = out_shape
    if (ty, tx) == (oy, ox):
        return tex
    tiled = np.tile(tex, (int(np.ceil(oy / ty)), int(np.ceil(ox / tx))))
    rng = np.random.default_rng(seed)
    y0 = int(rng.integers(0, tiled.shape[0] - oy + 1))
    x0 = int(rng.integers(0, tiled.shape[1] - ox + 1))
    return tiled[y0 : y0 + oy, x0 : x0 + ox].astype(np.float32)


def load_blank_spm_noise(
    blank_path: str | Path,
    target_rms_nm: float = 0.06,
    verbose: bool = True,
    apply_plane_remove: bool = USE_PLANE_REMOVE,
    apply_line_flatten: bool = USE_LINE_FLATTEN,
    apply_bandpass: bool = USE_BANDPASS_FILTER,
) -> tuple[float, np.ndarray, dict]:
    """Load a blank ``.spm`` scan and extract its noise texture.

    The file is parsed (:func:`parse_blocks`), the height channel is
    chosen (:func:`choose_best_height_image`), converted to nm
    (:func:`maybe_to_nm`) and the substrate noise is extracted with
    :func:`extract_topostats_like_noise`.

    Parameters
    ----------
    blank_path: str | Path
        Path of the blank ``.spm`` scan.
    target_rms_nm: float, optional
        RMS of the returned texture in nm. Defaults to 0.06.
    verbose: bool, optional
        Print the chosen channel and statistics. Defaults to `True`.
    apply_plane_remove: bool, optional
        Remove a tilted plane. Defaults to ``USE_PLANE_REMOVE``.
    apply_line_flatten: bool, optional
        Subtract line medians. Defaults to ``USE_LINE_FLATTEN``.
    apply_bandpass: bool, optional
        Band-pass filter the texture. Defaults to ``USE_BANDPASS_FILTER``.

    Returns
    -------
    tuple[float, np.ndarray, dict]
        The texture RMS in nm, the zero-mean texture and a dictionary with
        diagnostics (``flattened`` image, ``feature_mask``, ``units``,
        ``block``).

    Raises
    ------
    RuntimeError
        If the file cannot be parsed or has no plausible height channel.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> rms, texture, diag = sim.load_blank_spm_noise(
    ...     "../common-assets/noise/20240411_blank_water.0_00000.spm",
    ...     target_rms_nm=0.1, verbose=False,
    ... )  # doctest: +SKIP
    >>> round(rms, 3), texture.shape  # doctest: +SKIP
    (0.1, (512, 512))

    """

    raw, header, blocks = parse_blocks(blank_path)
    b, img0 = choose_best_height_image(raw, blocks, verbose=verbose)
    img_nm, unit_note = maybe_to_nm(img0)
    if verbose:
        print("Loaded via parser:", img0.shape, "| units:", unit_note)

    rms_nm, noise_tex, z_flat, feat = extract_topostats_like_noise(
        img_nm,
        median_size=3,
        bg_quantile=45,
        feature_sigma_px=3.0,
        feature_thresh_sigma=3.0,
        feature_dilate_px=6,
        inpaint_sigma_px=10.0,
        band_low_sigma_px=0.7,
        band_high_sigma_px=40.0,
        target_rms_nm=target_rms_nm,
    )
    diag = {
        "flattened": z_flat,
        "feature_mask": feat,
        "height_nm": img_nm,
        "unit_note": unit_note,
    }
    if verbose:
        print(
            f"Extracted noise RMS (bg): {rms_nm:.4f} nm | "
            f"tex shape: {noise_tex.shape}"
        )
    return rms_nm, noise_tex, diag


def load_model(
    path: str | Path = MODEL_PATH,
) -> dict[str, np.ndarray]:
    """Load the power-spectral-density (PSD) noise model.

    The model describes the substrate roughness of an ensemble of blank
    scans in the frequency domain, which lets :func:`generate_noise`
    synthesise noise with the same texture.

    Parameters
    ----------
    path: str | Path, optional
        Path of the ``.npz`` model. Defaults to ``MODEL_PATH``.

    Returns
    -------
    dict[str, np.ndarray]
        The arrays stored in the model (mean and standard deviation of the
        log-PSD, frequencies, RMS values of the blanks and related
        metadata).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> model = sim.load_model(
    ...     "../common-assets/noise/psd_noise_model.npz"
    ... )  # doctest: +SKIP
    >>> "log_psd_mean" in model  # doctest: +SKIP
    True

    """

    data = np.load(path)
    return {
        "log_psd_mean": data["log_psd_mean"].astype(np.float32),
        "log_psd_std": data["log_psd_std"].astype(np.float32),
        "radial_freq": data["radial_freq"].astype(np.float32),
        "rms_values_nm": data["rms_values_nm"].astype(np.float32),
    }


def generate_noise(
    seed: int,
    target_shape: tuple[int, int] = TARGET_SHAPE,
    target_rms_nm: float | None = TARGET_NOISE_RMS_NM,
    method: Literal[
        "mean_full2d", "lognormal_full2d", "empirical_radial"
    ] = "mean_full2d",
    std_scale: float = 1.0,
    model: dict | None = None,
    model_path: str | Path | None = MODEL_PATH,
) -> np.ndarray:
    """Synthesise one AFM-like noise image from the PSD model.

    Random Fourier phases are combined with the model's power spectrum
    (Fourier synthesis), so the noise has the spatial frequencies of real
    blank scans; the result is scaled to ``target_rms_nm``.

    Parameters
    ----------
    seed: int
        Random seed.
    target_shape: tuple[int, int], optional
        Output shape ``(rows, columns)``; the spectrum is resized if
        needed. Defaults to ``TARGET_SHAPE``.
    target_rms_nm: float | None, optional
        RMS of the noise in nm; `None` draws one from the RMS values of the
        blanks in the model. Defaults to ``TARGET_NOISE_RMS_NM``.
    method: "mean_full2d" | "lognormal_full2d" | "empirical_radial", optional
        ``"mean_full2d"`` uses the mean 2-D spectrum (variation only from
        the random phases, recommended); ``"lognormal_full2d"`` also draws
        the spectrum from its log-normal spread; ``"empirical_radial"``
        synthesises isotropic noise from the radial spectrum. Defaults to
        "mean_full2d".
    std_scale: float, optional
        Scales the spectral spread for ``"lognormal_full2d"``. Defaults to
        1.0.
    model: dict | None, optional
        Preloaded model from :func:`load_model`. Defaults to `None`.
    model_path: str | Path | None, optional
        Model to load when ``model`` is `None`. Defaults to
        ``MODEL_PATH``.

    Returns
    -------
    np.ndarray
        Noise image of shape ``target_shape`` in nm.

    Raises
    ------
    ValueError
        If ``method`` is not one of the three methods.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> model = sim.load_model(
    ...     "../common-assets/noise/psd_noise_model.npz"
    ... )  # doctest: +SKIP
    >>> noise = sim.generate_noise(
    ...     1, (256, 256), 0.1, model=model
    ... )  # doctest: +SKIP
    >>> noise.shape, round(float(noise.std()), 2)  # doctest: +SKIP
    ((256, 256), 0.1)

    """

    if model is None:
        model = load_model(model_path)

    rng = np.random.default_rng(int(seed))
    out_shape = tuple(target_shape)

    if method == "mean_full2d":
        psd_sample = np.exp(model["log_psd_mean"])

    elif method == "lognormal_full2d":
        mu = model["log_psd_mean"]
        noise = rng.standard_normal(mu.shape).astype(np.float32)
        psd_sample = np.exp(mu + std_scale * mu * noise)

    elif method == "empirical_radial":
        mu = model["log_psd_mean"]
        radial_psd = np.exp(mu.mean(axis=1))
        radial_freq_for_interp = np.abs(np.fft.fftfreq(mu.shape[0], d=1.0))

        fy = np.fft.fftfreq(out_shape[0], d=1.0)
        fx = np.fft.rfftfreq(out_shape[1], d=1.0)
        fy2d, fx2d = np.meshgrid(fy, fx, indexing="ij")
        kr = np.sqrt(fy2d**2 + fx2d**2)

        psd_sample = (
            np.interp(
                kr.ravel(),
                radial_freq_for_interp,
                radial_psd,
                left=float(radial_psd[0]),
                right=float(radial_psd[-1]),
            )
            .reshape(kr.shape)
            .astype(np.float32)
        )

    else:
        raise ValueError(
            f"Unknown method {method!r}. "
            "Choose 'mean_full2d', 'lognormal_full2d', or 'empirical_radial'."
        )

    psd_ny, psd_nx = psd_sample.shape
    out_ny, out_nx = out_shape
    rfft_nx = out_nx // 2 + 1

    if (psd_ny, psd_nx) != (out_ny, rfft_nx):
        from scipy.ndimage import zoom

        psd_sample = zoom(
            psd_sample, (out_ny / psd_ny, rfft_nx / psd_nx), order=1
        )

    psd_sample = psd_sample.astype(np.float32)

    phase = rng.uniform(0.0, 2.0 * np.pi, size=psd_sample.shape)
    spec = np.sqrt(np.maximum(psd_sample, EPS)) * np.exp(1j * phase)

    z = np.fft.irfft2(spec, s=out_shape).real.astype(np.float32)
    z -= np.float32(np.mean(z))
    rmsv = model["rms_values_nm"]
    trmsf = float(target_rms_nm)
    target = float(rng.choice(rmsv)) if target_rms_nm is None else trmsf
    cur = float(np.std(z))
    if cur > EPS:
        z *= target / cur

    return z


def random_transform_noise_texture(
    tex: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Apply a reproducible random affine transform to a noise texture.

    Rotation, anisotropic scaling, translation, optional flips and a small
    amplitude jitter turn one blank scan into many different-looking
    noise realisations with the same statistics.

    Parameters
    ----------
    tex: np.ndarray
        2-D noise texture.
    seed: int
        Random seed of the transform.

    Returns
    -------
    np.ndarray
        Transformed texture of the same shape.

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    >>> tex = np.random.default_rng(0).normal(size=(64, 64))
    >>> out = sim.random_transform_noise_texture(tex, seed=3)
    >>> out.shape == tex.shape
    True

    """

    rng = np.random.default_rng(int(seed))
    tex = np.asarray(tex, dtype=np.float32)

    theta = np.deg2rad(float(rng.uniform(0.0, 360.0)))
    base_scale = float(rng.uniform(0.85, 1.15))
    anis = float(rng.uniform(0.85, 1.15))
    sx, sy = base_scale * anis, base_scale / anis

    c, s = float(np.cos(theta)), float(np.sin(theta))
    A = np.array([[c * sx, -s * sy], [s * sx, c * sy]], dtype=np.float32)
    M = np.linalg.inv(A).astype(np.float32)

    h, w = tex.shape
    center = np.array([(h - 1) / 2.0, (w - 1) / 2.0], dtype=np.float32)
    t = np.array(
        [
            float(rng.uniform(-0.15 * h, 0.15 * h)),
            float(rng.uniform(-0.15 * w, 0.15 * w)),
        ],
        dtype=np.float32,
    )
    offset = center - M @ (center + t)

    out = affine_transform(
        tex,
        matrix=M,
        offset=offset,
        output_shape=tex.shape,
        order=1,
        mode="wrap",
        prefilter=False,
    ).astype(np.float32)

    if bool(rng.integers(0, 2)):
        out = out[::-1, :]
    if bool(rng.integers(0, 2)):
        out = out[:, ::-1]
    out *= float(rng.uniform(0.90, 1.10))
    return out.astype(np.float32)


def sample_noise_image(
    noise_source: Literal["blank_spm", "psd_model", "none"],
    noise_asset: dict | np.ndarray | None,
    seed: int,
    out_shape: tuple[int, int],
    target_rms_nm: float | None,
    nm_per_pixel: float = 1.0,
    noise_native_nm_px: float = 0.5,
) -> np.ndarray | None:
    """Return the noise image added to one sample.

    The noise is generated on the native grid of the noise asset
    (``noise_native_nm_px``) over the same physical field as the image and
    then resampled to ``out_shape``, so it keeps its physical length scale
    whatever the pixel size. It is shifted to zero mean and scaled to
    ``target_rms_nm``.

    Parameters
    ----------
    noise_source: "blank_spm" | "psd_model" | "none"
        Where the noise comes from (see :func:`load_noise_asset`).
    noise_asset: dict | np.ndarray | None
        The blank texture (``"blank_spm"``) or the PSD model
        (``"psd_model"``).
    seed: int
        Random seed.
    out_shape: tuple[int, int]
        Shape of the image ``(rows, columns)``.
    target_rms_nm: float | None
        RMS of the noise in nm; `None` keeps the asset's own scale.
    nm_per_pixel: float, optional
        Pixel size of the image in nm. Defaults to 1.0.
    noise_native_nm_px: float, optional
        Pixel size of the noise asset in nm. Defaults to 0.5.

    Returns
    -------
    np.ndarray | None
        Noise image in nm, or `None` when there is no noise asset.

    Raises
    ------
    ValueError
        If the noise asset has zero variance.

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    Blank-scan noise from a random texture, scaled to 0.1 nm RMS:

    >>> texture = np.random.default_rng(0).normal(size=(512, 512))
    >>> noise = sim.sample_noise_image(
    ...     "blank_spm", texture, 1, (128, 128), 0.1, nm_per_pixel=1.0
    ... )
    >>> noise.shape, round(float(np.sqrt(np.mean(noise ** 2))), 3)
    ((128, 128), 0.1)
    >>> sim.sample_noise_image("none", None, 1, (8, 8), 0.1) is None
    True

    """

    if noise_asset is None:
        return None
    req_ny = max(16, int(out_shape[0] * nm_per_pixel / noise_native_nm_px))
    req_nx = max(16, int(out_shape[1] * nm_per_pixel / noise_native_nm_px))
    if noise_source == "blank_spm":
        noise_tex = random_transform_noise_texture(noise_asset, seed=seed)
        raw_noise = tile_or_crop(
            noise_tex, (req_ny, req_nx), seed=seed
        ).astype(np.float32)
    elif noise_source == "psd_model":
        raw_noise = generate_noise(
            seed=seed,
            target_shape=(req_ny, req_nx),
            target_rms_nm=target_rms_nm,
            method=PSD_NOISE_METHOD,
            std_scale=PSD_NOISE_STD_SCALE,
            model=noise_asset,
        ).astype(np.float32)
    else:
        return None
    if (req_ny, req_nx) != out_shape:
        zoom_y = out_shape[0] / req_ny
        zoom_x = out_shape[1] / req_nx
        raw_noise = zoom(raw_noise, (zoom_y, zoom_x), order=1)
    raw_noise = raw_noise.astype(np.float32)
    raw_noise -= float(raw_noise.mean())
    if target_rms_nm is not None:
        measured_rms = float(
            np.sqrt(np.mean(raw_noise.astype(np.float64) ** 2))
        )
        if measured_rms <= 1e-12 and float(target_rms_nm) > 0:
            raise ValueError("Noise asset has zero variance")
        raw_noise *= float(target_rms_nm) / max(measured_rms, 1e-12)
    return raw_noise


# ────────────────────────────────────────────────────────────────────────────
# DNA ground-truth mask
# ────────────────────────────────────────────────────────────────────────────


def nm_to_px(
    coords_xy_nm: np.ndarray,
    extent: tuple[float, float, float, float],
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert (x, y) positions in nm to integer pixel indices.

    Positions are mapped linearly from ``extent`` onto an ``nx`` by
    ``ny`` grid and clipped to the valid index range.

    Parameters
    ----------
    coords_xy_nm: np.ndarray
        Positions with shape ``(N, 2)`` in nm.
    extent: tuple[float, float, float, float]
        Physical extent ``(xmin, xmax, ymin, ymax)`` in nm.
    nx: int
        Number of pixels along x.
    ny: int
        Number of pixels along y.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Column indices and row indices (int32).

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    >>> ix, iy = sim.nm_to_px(
    ...     np.array([[0.0, 0.0], [10.0, 5.0]]), (0, 10, 0, 10), 11, 11
    ... )
    >>> ix.tolist(), iy.tolist()
    ([0, 10], [0, 5])

    """

    x = coords_xy_nm[:, 0]
    y = coords_xy_nm[:, 1]
    xmin, xmax, ymin, ymax = extent
    ix = np.clip(
        ((x - xmin) / (xmax - xmin) * (nx - 1)).astype(np.int32), 0, nx - 1
    )
    iy = np.clip(
        ((y - ymin) / (ymax - ymin) * (ny - 1)).astype(np.int32), 0, ny - 1
    )
    return ix, iy


def rasterize_closed_polyline_mask(
    coords: np.ndarray,
    extent: tuple[float, float, float, float],
    nx: int,
    ny: int,
) -> np.ndarray:
    """Draw a closed chain as a one-pixel-wide line.

    Every bond, including the one from the last bead back to the first, is
    drawn by dense linear interpolation between its pixel end points.

    Parameters
    ----------
    coords: np.ndarray
        Bead positions with shape ``(N, 2)`` or ``(N, 3)`` in nm; only x
        and y are used.
    extent: tuple[float, float, float, float]
        Physical extent ``(xmin, xmax, ymin, ymax)`` in nm.
    nx: int
        Number of pixels along x.
    ny: int
        Number of pixels along y.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(ny, nx)``, `True` on the line.

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    A square ring drawn on a 21 by 21 grid:

    >>> square = np.array([[-5, -5], [5, -5], [5, 5], [-5, 5]], float)
    >>> line = sim.rasterize_closed_polyline_mask(
    ...     square, (-10, 10, -10, 10), 21, 21
    ... )
    >>> int(line.sum())
    40

    """

    coords = np.asarray(coords, dtype=np.float32)
    ix, iy = nm_to_px(coords[:, :2], extent, nx, ny)
    out = np.zeros((ny, nx), dtype=bool)
    npts = len(ix)
    for k in range(npts):
        k2 = (k + 1) % npts
        x0, y0 = int(ix[k]), int(iy[k])
        x1, y1 = int(ix[k2]), int(iy[k2])
        n_seg = int(max(abs(x1 - x0), abs(y1 - y0)) + 1)
        if n_seg <= 1:
            out[y0, x0] = True
            continue
        xs = np.linspace(x0, x1, n_seg).astype(np.int32)
        ys = np.linspace(y0, y1, n_seg).astype(np.int32)
        if xs.size > 1:
            keep = np.ones(xs.shape[0], dtype=bool)
            keep[1:] = (xs[1:] != xs[:-1]) | (ys[1:] != ys[:-1])
            xs, ys = xs[keep], ys[keep]
        out[ys, xs] = True
    return out


def make_dna_mask_from_beads(
    coords: np.ndarray,
    extent: tuple[float, float, float, float],
    nx: int,
    ny: int,
    dna_diameter_nm: float | None = None,
    dilate_px: int | None = None,
) -> np.ndarray:
    """Rasterise the DNA mask as a tube of constant width.

    Every pixel whose centre lies within ``dna_diameter_nm / 2`` of a bond
    (a line segment between consecutive beads, including the closing bond)
    is part of the mask. The width is used exactly, without rounding the
    radius to whole pixels. Without a diameter, the one-pixel centre line
    (optionally dilated by ``dilate_px``) is returned.

    Parameters
    ----------
    coords: np.ndarray
        Bead positions with shape ``(N, 2)`` or ``(N, 3)`` in nm.
    extent: tuple[float, float, float, float]
        Physical extent ``(xmin, xmax, ymin, ymax)`` in nm.
    nx: int
        Number of pixels along x.
    ny: int
        Number of pixels along y.
    dna_diameter_nm: float | None, optional
        Width of the tube in nm. Defaults to `None`.
    dilate_px: int | None, optional
        Dilation of the centre line when no diameter is given. Defaults to
        `None`.

    Returns
    -------
    np.ndarray
        Mask of shape ``(ny, nx)`` (uint8, 1 on DNA).

    Raises
    ------
    ValueError
        If ``dna_diameter_nm`` is not finite and positive.

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    A 4.47 nm wide straight segment pair on a 1 nm grid:

    >>> beads = np.array([[-20.0, 0.0], [20.0, 0.0]])
    >>> mask = sim.make_dna_mask_from_beads(
    ...     beads, (-30, 30, -30, 30), 61, 61, dna_diameter_nm=4.47
    ... )
    >>> int(mask[:, 30].sum())
    5

    """
    if dna_diameter_nm is None:
        line = rasterize_closed_polyline_mask(coords, extent, nx, ny)
        if dilate_px and dilate_px > 0:
            line = binary_dilation(
                line, structure=disk_footprint(float(dilate_px))
            )
        return line.astype(np.uint8)
    radius = float(dna_diameter_nm) / 2
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("DNA/mask width must be finite and positive")
    xmin, xmax, ymin, ymax = map(float, extent)
    px = (xmax - xmin) / (nx - 1)
    py = (ymax - ymin) / (ny - 1)
    mask = np.zeros((ny, nx), dtype=bool)
    xy = np.asarray(coords, dtype=float)[:, :2]
    for a, b in zip(xy, np.roll(xy, -1, axis=0)):
        c0 = max(0, int(np.floor((min(a[0], b[0]) - radius - xmin) / px)))
        c1 = min(nx, int(np.ceil((max(a[0], b[0]) + radius - xmin) / px)) + 1)
        r0 = max(0, int(np.floor((min(a[1], b[1]) - radius - ymin) / py)))
        r1 = min(ny, int(np.ceil((max(a[1], b[1]) + radius - ymin) / py)) + 1)
        if r1 <= r0 or c1 <= c0:
            continue
        yy, xx = np.mgrid[r0:r1, c0:c1]
        dx = xmin + xx * px - a[0]
        dy = ymin + yy * py - a[1]
        segment = b - a
        length2 = float(segment @ segment)
        t = np.clip(
            (dx * segment[0] + dy * segment[1]) / max(length2, 1e-20), 0, 1
        )
        distance2 = (dx - t * segment[0]) ** 2 + (dy - t * segment[1]) ** 2
        mask[r0:r1, c0:c1] |= distance2 <= radius**2 + 1e-12
    return mask.astype(np.uint8)


# ────────────────────────────────────────────────────────────────────────────
# Crossing detection and crossing mask
# ────────────────────────────────────────────────────────────────────────────


def canon_axis(u: np.ndarray) -> np.ndarray | None:
    """Normalise a 2-D direction and fix its sign.

    ``v`` and ``-v`` describe the same axis; the sign is chosen so that
    the first non-zero component is positive, which makes directions
    comparable.

    Parameters
    ----------
    u: np.ndarray
        2-D vector.

    Returns
    -------
    np.ndarray | None
        Unit vector with a canonical sign, or `None` for a (near) zero
        vector.

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    >>> sim.canon_axis(np.array([-3.0, -4.0]))
    array([0.6, 0.8])
    >>> sim.canon_axis(np.zeros(2)) is None
    True

    """

    u = np.asarray(u, dtype=float)
    nrm = float(np.linalg.norm(u))
    if nrm < 1e-12:
        return None
    u = u / nrm
    if (u[0] < 0) or (abs(u[0]) < 1e-12 and u[1] < 0):
        u = -u
    return u


def find_polyline_crossings(
    coords_xy: np.ndarray,
    min_separation_beads: int = CROSS_MIN_SEP_BEADS,
    merge_radius_nm: float = 0.5,
) -> list[dict]:
    """Detect the crossings of a closed chain seen from above.

    Every pair of bonds is tested; neighbouring bonds and bonds closer than
    ``min_separation_beads`` along the ring are skipped. Hits within
    ``merge_radius_nm`` of each other become one crossing, and the strand
    directions at each crossing are stored (directions within about 18
    degrees count as the same strand).

    Parameters
    ----------
    coords_xy: np.ndarray
        Bead positions with shape ``(N, 2)`` in nm.
    min_separation_beads: int, optional
        Smallest separation along the ring, in beads, for a crossing.
        Defaults to ``CROSS_MIN_SEP_BEADS``.
    merge_radius_nm: float, optional
        Hits closer than this are merged. Defaults to 0.5.

    Returns
    -------
    list[dict]
        One dictionary per crossing with ``pt_nm`` (position),
        ``axes_nm`` (unit strand directions), ``seg_i``, ``seg_j``, ``t``,
        ``u`` and ``n_hits``.

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    A figure-eight crosses itself once:

    >>> s = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    >>> eight = np.stack([np.sin(s), np.sin(s) * np.cos(s)], axis=1) * 50
    >>> crossings = sim.find_polyline_crossings(eight)
    >>> point = [abs(round(float(v), 3)) for v in crossings[0]["pt_nm"]]
    >>> len(crossings), point
    (1, [0.0, 0.0])

    """

    xy = np.asarray(coords_xy, dtype=float)
    n = int(xy.shape[0])
    raw = []
    for i in range(n):
        i2 = (i + 1) % n
        p1, p2 = xy[i], xy[i2]
        for j in range(i + 1, n):
            j2 = (j + 1) % n
            if i == j or i2 == j or j2 == i:
                continue
            if _circ_sep(n, i, j) < int(min_separation_beads):
                continue
            q1, q2 = xy[j], xy[j2]
            hit, pt, t, u = _seg_intersect_2d(p1, p2, q1, q2)
            if not hit:
                continue
            ua = canon_axis(p2 - p1)
            ub = canon_axis(q2 - q1)
            if ua is None or ub is None:
                continue
            raw.append(
                dict(
                    pt=np.array(pt, float),
                    axes=[ua, ub],
                    seg_i=int(i),
                    seg_j=int(j),
                    t=float(t),
                    u=float(u),
                )
            )
    clusters = []
    for r in raw:
        pt = r["pt"]
        placed = False
        for c in clusters:
            fmc = float(merge_radius_nm)
            centre = c["pt_sum"] / c["n"]
            if np.linalg.norm(pt - centre) <= fmc:
                c["pt_sum"] += pt
                c["n"] += 1
                c["axes"].extend(r["axes"])
                c["items"].append(r)
                placed = True
                break
        if not placed:
            clusters.append(
                {
                    "pt_sum": pt.copy(),
                    "n": 1,
                    "axes": list(r["axes"]),
                    "items": [r],
                }
            )
    out = []
    for c in clusters:
        pt_mean = c["pt_sum"] / c["n"]
        axes = []
        for uvec in c["axes"]:
            uvec = canon_axis(uvec)
            if uvec is None:
                continue
            if any(abs(np.dot(uvec, v)) > 0.95 for v in axes):
                continue
            axes.append(uvec)
        rep = c["items"][0]
        out.append(
            dict(
                pt_nm=np.asarray(pt_mean, dtype=np.float32),
                axes_nm=(
                    [np.asarray(v, dtype=np.float32) for v in axes]
                    if axes
                    else [np.array([1.0, 0.0], np.float32)]
                ),
                seg_i=int(rep["seg_i"]),
                seg_j=int(rep["seg_j"]),
                t=float(rep["t"]),
                u=float(rep["u"]),
                n_hits=int(c["n"]),
            )
        )
    return out


def _indices_around_center(
    center: int, npts: int, half_window: int
) -> list[int]:
    """Return the bead indices within ``half_window`` of a centre bead.

    Indices wrap around the ring.

    Parameters
    ----------
    center: int
        Index of the centre bead.
    npts: int
        Number of beads in the ring.
    half_window: int
        Number of beads on each side.

    Returns
    -------
    list[int]
        ``2 * half_window + 1`` bead indices.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> sim._indices_around_center(0, 10, 2)
    [8, 9, 0, 1, 2]

    """
    return [
        int((center + off) % npts)
        for off in range(-int(half_window), int(half_window) + 1)
    ]


def _crossing_centers_from_record(
    crossing: dict,
) -> tuple[int | None, int | None]:
    """Return the two centre beads of a crossing record.

    Records from the renderer give the upper and lower centre beads; plain
    crossing records give the two segment indices.

    Parameters
    ----------
    crossing: dict
        A crossing record.

    Returns
    -------
    tuple[int | None, int | None]
        The two centre indices, or ``(None, None)`` if the record has
        neither form.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> sim._crossing_centers_from_record({"seg_i": 3, "seg_j": 40})
    (3, 40)
    >>> sim._crossing_centers_from_record({})
    (None, None)

    """
    if "top_center" in crossing and "bottom_center" in crossing:
        return int(crossing["top_center"]), int(crossing["bottom_center"])
    if "seg_i" in crossing and "seg_j" in crossing:
        return int(crossing["seg_i"]), int(crossing["seg_j"])
    return None, None


def _rasterize_local_polyline_mask(
    coords: np.ndarray,
    indices: list[int],
    extent: tuple[float, float, float, float],
    nx: int,
    ny: int,
    radius_px: int,
) -> np.ndarray:
    """Draw part of a chain as a widened line.

    The bonds between consecutive ``indices`` are drawn on the pixel grid
    and dilated with a disk of ``radius_px``.

    Parameters
    ----------
    coords: np.ndarray
        Bead positions with shape ``(N, 2)`` or ``(N, 3)`` in nm.
    indices: list[int]
        Consecutive bead indices to draw.
    extent: tuple[float, float, float, float]
        Physical extent ``(xmin, xmax, ymin, ymax)`` in nm.
    nx: int
        Number of pixels along x.
    ny: int
        Number of pixels along y.
    radius_px: int
        Dilation radius in pixels.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(ny, nx)``.

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    >>> beads = np.array([[-5.0, 0.0], [0.0, 0.0], [5.0, 0.0]])
    >>> mask = sim._rasterize_local_polyline_mask(
    ...     beads, [0, 1, 2], (-10, 10, -10, 10), 21, 21, 1
    ... )
    >>> int(mask.sum())
    35

    """
    coords = np.asarray(coords, dtype=np.float32)
    npts = int(coords.shape[0])
    xmin, xmax, ymin, ymax = extent
    width = max(float(xmax - xmin), 1e-12)
    height = max(float(ymax - ymin), 1e-12)
    ix = np.clip(
        ((coords[:, 0] - xmin) / width * (int(nx) - 1)).astype(np.int32),
        0,
        int(nx) - 1,
    )
    iy = np.clip(
        ((coords[:, 1] - ymin) / height * (int(ny) - 1)).astype(np.int32),
        0,
        int(ny) - 1,
    )
    line = np.zeros((int(ny), int(nx)), dtype=bool)
    if len(indices) == 1:
        k = int(indices[0]) % npts
        line[iy[k], ix[k]] = True
    else:
        for a, b in zip(indices[:-1], indices[1:]):
            a = int(a) % npts
            b = int(b) % npts
            x0, y0 = int(ix[a]), int(iy[a])
            x1, y1 = int(ix[b]), int(iy[b])
            n_seg = int(max(abs(x1 - x0), abs(y1 - y0)) + 1)
            if n_seg <= 1:
                line[y0, x0] = True
                continue
            xs = np.linspace(x0, x1, n_seg).astype(np.int32)
            ys = np.linspace(y0, y1, n_seg).astype(np.int32)
            valid = (xs >= 0) & (xs < int(nx)) & (ys >= 0) & (ys < int(ny))
            if np.any(valid):
                line[ys[valid], xs[valid]] = True
    radius_px = max(1, int(radius_px))
    yy, xx = np.ogrid[
        -radius_px : radius_px + 1,
        -radius_px : radius_px + 1,
    ]
    fp = (xx * xx + yy * yy) <= radius_px * radius_px
    return grey_dilation(line.astype(np.uint8), footprint=fp) > 0


def _soften_binary_target(binary: np.ndarray, sigma_px: float) -> np.ndarray:
    """Turn a binary mask into a soft target that decays outside it.

    Pixels inside the mask are 1; outside, the value falls as a Gaussian
    of the distance to the mask with width ``sigma_px``.

    Parameters
    ----------
    binary: np.ndarray
        Boolean mask.
    sigma_px: float
        Decay width in pixels.

    Returns
    -------
    np.ndarray
        Soft target (float32) with values between 0 and 1.

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    >>> binary = np.zeros((5, 5), bool)
    >>> binary[2, 2] = True
    >>> soft = sim._soften_binary_target(binary, 1.0)
    >>> float(soft[2, 2]), round(float(soft[2, 3]), 3)
    (1.0, 0.607)

    """
    binary = binary.astype(bool)
    if not np.any(binary):
        return np.zeros(binary.shape, dtype=np.float32)
    sigma_px = max(float(sigma_px), 1e-6)
    dist = distance_transform_edt(~binary).astype(np.float32)
    target = np.exp(-(dist**2) / (2.0 * sigma_px**2)).astype(np.float32)
    target[binary] = 1.0
    return target


def add_crossing_patch(
    mask: np.ndarray,
    pt_px: tuple[float, float],
    axes_px: list[np.ndarray],
    sigma_center: float = 2.0,
    chain_extent: float = 5.0,
    sigma_perp: float = 1.8,
    center_weight: float = 1.0,
    chain_weight: float = 0.8,
) -> None:
    """Paint a round Gaussian patch into a crossing mask (in place).

    Used when the two strands of a crossing do not overlap on the pixel
    grid; the patch is combined with ``mask`` by taking the maximum.

    Parameters
    ----------
    mask: np.ndarray
        2-D float mask, modified in place.
    pt_px: tuple[float, float]
        Patch centre ``(x, y)`` in pixels.
    axes_px: list[np.ndarray]
        Strand directions; kept for compatibility, not used.
    sigma_center: float, optional
        Width of the patch in pixels. Defaults to 2.0.
    chain_extent: float, optional
        Kept for compatibility, not used. Defaults to 5.0.
    sigma_perp: float, optional
        Alternative width; the larger of the two widths is used. Defaults
        to 1.8.
    center_weight: float, optional
        Peak value of the patch. Defaults to 1.0.
    chain_weight: float, optional
        Kept for compatibility, not used. Defaults to 0.8.

    Returns
    -------
    None
        ``mask`` is updated in place.

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    >>> mask = np.zeros((21, 21), np.float32)
    >>> sim.add_crossing_patch(mask, (10.0, 10.0), [], sigma_center=2.0)
    >>> float(mask[10, 10]), float(mask[0, 0])
    (1.0, 0.0)

    """
    h, w = mask.shape
    cx, cy = float(pt_px[0]), float(pt_px[1])
    sigma = max(float(sigma_center), float(sigma_perp), 1e-6)
    radius = int(np.ceil(3.0 * sigma))
    x0 = max(0, int(np.floor(cx - radius)))
    x1 = min(w - 1, int(np.ceil(cx + radius)))
    y0 = max(0, int(np.floor(cy - radius)))
    y1 = min(h - 1, int(np.ceil(cy + radius)))
    xs = np.arange(x0, x1 + 1, dtype=float)
    ys = np.arange(y0, y1 + 1, dtype=float)
    xg, yg = np.meshgrid(xs, ys)
    dx = xg - cx
    dy = yg - cy
    patch = np.exp(-0.5 * (dx * dx + dy * dy) / (sigma * sigma)).astype(
        np.float32
    )
    patch *= float(center_weight)
    mask[y0 : y1 + 1, x0 : x1 + 1] = np.maximum(
        mask[y0 : y1 + 1, x0 : x1 + 1],
        patch,
    )


def make_crossing_mask(
    coords: np.ndarray,
    extent: tuple[float, float, float, float],
    nx: int,
    ny: int,
    sigma_center_px: float = CROSS_SIGMA_CENTER_PX,
    sigma_perp_px: float = CROSS_SIGMA_PERP_PX,
    chain_extent: float = CROSS_CHAIN_EXTENT,
    center_weight: float = CROSS_CENTER_WEIGHT,
    chain_weight: float = CROSS_CHAIN_WEIGHT,
    min_separation_beads: int = CROSS_MIN_SEP_BEADS,
    crossings_precomputed: list[dict] | None = None,
    merge_radius_nm: float = 0.5,
) -> tuple[np.ndarray, list[dict]]:
    """Build the soft crossing mask used as the second training target.

    For every crossing, the two strand pieces within ``chain_extent``
    beads of it are drawn (:func:`_rasterize_local_polyline_mask`); their
    overlap marks the crossing and their union the strands next to it.
    Both are softened (:func:`_soften_binary_target`), weighted with
    ``center_weight`` and ``chain_weight``, combined by maximum and scaled
    so the mask peaks at 1.

    Parameters
    ----------
    coords: np.ndarray
        Bead positions with shape ``(N, 3)`` in nm.
    extent: tuple[float, float, float, float]
        Physical extent ``(xmin, xmax, ymin, ymax)`` in nm.
    nx: int
        Number of pixels along x.
    ny: int
        Number of pixels along y.
    sigma_center_px: float, optional
        Softening width of the overlap in pixels. Defaults to
        ``CROSS_SIGMA_CENTER_PX``.
    sigma_perp_px: float, optional
        Half-width of the drawn strands in pixels. Defaults to
        ``CROSS_SIGMA_PERP_PX``.
    chain_extent: float, optional
        Beads on each side of the crossing that are drawn. Defaults to
        ``CROSS_CHAIN_EXTENT``.
    center_weight: float, optional
        Weight of the overlap. Defaults to ``CROSS_CENTER_WEIGHT``.
    chain_weight: float, optional
        Weight of the strands. Defaults to ``CROSS_CHAIN_WEIGHT``.
    min_separation_beads: int, optional
        Smallest separation along the ring for a crossing. Defaults to
        ``CROSS_MIN_SEP_BEADS``.
    crossings_precomputed: list[dict] | None, optional
        Crossings to draw. Defaults to `None`, which detects them with
        :func:`find_polyline_crossings`.
    merge_radius_nm: float, optional
        Merge radius used when detecting crossings. Defaults to 0.5.

    Returns
    -------
    tuple[np.ndarray, list[dict]]
        The crossing mask (float32, 0 to 1, shape ``(ny, nx)``) and the
        crossings used.

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    >>> s = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    >>> eight = np.stack(
    ...     [np.sin(s), np.sin(s) * np.cos(s), np.zeros_like(s)], axis=1
    ... ) * 50
    >>> mask, crossings = sim.make_crossing_mask(
    ...     eight, (-60, 60, -60, 60), 121, 121
    ... )
    >>> len(crossings), float(mask.max())
    (1, 1.0)

    """
    coords = np.asarray(coords, dtype=np.float32)
    xy_nm = coords[:, :2].astype(float)
    npts = int(coords.shape[0])
    if crossings_precomputed is None:
        crossings = find_polyline_crossings(
            xy_nm,
            min_separation_beads=min_separation_beads,
            merge_radius_nm=float(merge_radius_nm),
        )
    else:
        crossings = crossings_precomputed
    mask = np.zeros((int(ny), int(nx)), dtype=np.float32)
    if not crossings:
        return mask.astype(np.float32), crossings
    strand_radius_px = max(
        1,
        int(np.ceil(max(float(sigma_perp_px), 0.5))),
    )
    soft_sigma_px = max(0.5, float(sigma_center_px))
    half_window = max(1, int(round(float(chain_extent))))
    overlap_binary = np.zeros((int(ny), int(nx)), dtype=bool)
    union_binary = np.zeros((int(ny), int(nx)), dtype=bool)
    for crossing in crossings:
        centre_a, centre_b = _crossing_centers_from_record(crossing)
        if centre_a is None or centre_b is None:
            continue
        indices_a = _indices_around_center(
            centre_a,
            npts=npts,
            half_window=half_window,
        )
        indices_b = _indices_around_center(
            centre_b,
            npts=npts,
            half_window=half_window,
        )
        mask_a = _rasterize_local_polyline_mask(
            coords=coords,
            indices=indices_a,
            extent=extent,
            nx=int(nx),
            ny=int(ny),
            radius_px=strand_radius_px,
        )
        mask_b = _rasterize_local_polyline_mask(
            coords=coords,
            indices=indices_b,
            extent=extent,
            nx=int(nx),
            ny=int(ny),
            radius_px=strand_radius_px,
        )
        local_overlap = mask_a & mask_b
        local_union = mask_a | mask_b
        if not np.any(local_overlap):
            xmin, xmax, ymin, ymax = extent
            sx = (int(nx) - 1) / max(float(xmax - xmin), 1e-12)
            sy = (int(ny) - 1) / max(float(ymax - ymin), 1e-12)
            x_nm, y_nm = crossing["pt_nm"]
            px = (float(x_nm) - float(xmin)) * sx
            py = (float(y_nm) - float(ymin)) * sy
            tmp = np.zeros_like(mask, dtype=np.float32)
            add_crossing_patch(
                tmp,
                pt_px=(px, py),
                axes_px=[],
                sigma_center=soft_sigma_px,
                chain_extent=chain_extent,
                sigma_perp=strand_radius_px,
                center_weight=1.0,
                chain_weight=0.0,
            )
            local_overlap = tmp > 0.25
        overlap_binary |= local_overlap
        union_binary |= local_union
    overlap_soft = _soften_binary_target(
        overlap_binary,
        sigma_px=soft_sigma_px,
    )
    union_soft = _soften_binary_target(
        union_binary,
        sigma_px=max(0.5, 0.5 * soft_sigma_px),
    )
    mask = np.maximum(
        float(center_weight) * overlap_soft,
        float(chain_weight) * union_soft,
    )
    m = float(mask.max())
    if m > 1e-12:
        mask /= m
    return mask.astype(np.float32), crossings


# ────────────────────────────────────────────────────────────────────────────
# Decoy fragments
# ────────────────────────────────────────────────────────────────────────────


def make_decoy_chain(
    seed: int,
    n_beads_range: tuple[int, int] = (2, 4),
    bond_length: float = BOND_LENGTH,
) -> np.ndarray:
    """Make a short, slightly curved DNA fragment (a decoy).

    The fragment has between ``n_beads_range[0]`` and ``n_beads_range[1]``
    beads, a random direction, small random bends and heights between 2.5
    and 4 nm; it is not relaxed.

    Parameters
    ----------
    seed: int
        Random seed.
    n_beads_range: tuple[int, int], optional
        Smallest and largest number of beads. Defaults to ``(2, 4)``.
    bond_length: float, optional
        Distance between beads in nm. Defaults to ``BOND_LENGTH``.

    Returns
    -------
    np.ndarray
        Bead positions with shape ``(N, 3)`` in nm, starting at the
        origin.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> decoy = sim.make_decoy_chain(42, (3, 3), bond_length=4.5)
    >>> decoy.shape
    (3, 3)

    """

    rng = np.random.default_rng(int(seed))
    n_b = rng.integers(n_beads_range[0], n_beads_range[1] + 1)
    coords = np.zeros((n_b, 3), dtype=np.float32)
    angle = rng.uniform(0, 2 * np.pi)
    dirn = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32)

    for i in range(1, n_b):
        da = rng.normal(0, 0.3)
        c, s = np.cos(da), np.sin(da)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        dirn = R @ dirn
        dirn = dirn / np.linalg.norm(dirn)
        coords[i] = coords[i - 1] + bond_length * dirn

    coords[:, 2] = rng.uniform(2.5, 4.0, n_b)
    return coords


def place_decoy_in_corner(
    decoy_coords: np.ndarray,
    global_extent: tuple[float, float, float, float],
    corner_margin_nm: float = 2.0,
    seed: int | None = None,
) -> np.ndarray:
    """Move a decoy fragment into a random corner of the image.

    The fragment's centroid is moved to ``corner_margin_nm`` inside one of
    the four corners of ``global_extent``, chosen with ``seed``.

    Parameters
    ----------
    decoy_coords: np.ndarray
        Bead positions with shape ``(N, 3)`` in nm.
    global_extent: tuple[float, float, float, float]
        Image extent ``(xmin, xmax, ymin, ymax)`` in nm.
    corner_margin_nm: float, optional
        Distance of the centroid from the corner in nm. Defaults to 2.0.
    seed: int | None, optional
        Random seed of the corner. Defaults to `None`, which derives a
        seed from the coordinates.

    Returns
    -------
    np.ndarray
        The moved bead positions.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> decoy = sim.make_decoy_chain(1, (3, 3), bond_length=4.5)
    >>> moved = sim.place_decoy_in_corner(
    ...     decoy, (0, 100, 0, 100), corner_margin_nm=8.0, seed=0
    ... )
    >>> centre = moved[:, :2].mean(axis=0)
    >>> sorted(round(float(c)) for c in centre) in ([8, 8], [8, 92], [92, 92])
    True

    """

    if seed is None:
        seed = int(np.sum(decoy_coords))
    rng = np.random.default_rng(int(seed))
    xmin, xmax, ymin, ymax = global_extent
    corner = rng.integers(0, 4)
    targets = [
        (xmin + corner_margin_nm, ymax - corner_margin_nm),  # top-left
        (xmax - corner_margin_nm, ymax - corner_margin_nm),  # top-right
        (xmin + corner_margin_nm, ymin + corner_margin_nm),  # bottom-left
        (xmax - corner_margin_nm, ymin + corner_margin_nm),  # bottom-right
    ]
    tx, ty = targets[corner]
    centroid = decoy_coords.mean(axis=0)
    offset = np.array([tx, ty, 0.0]) - centroid
    return decoy_coords + offset


# ────────────────────────────────────────────────────────────────────────────
# Sample pipeline
# ────────────────────────────────────────────────────────────────────────────


def load_noise_asset(verbose: bool = True) -> tuple[str, Any, dict]:
    """Load the noise source chosen in the noise cell.

    A blank ``.spm`` scan (``BLANK_SPM_PATH``) is preferred. If it is
    missing or cannot be parsed, the PSD noise model (``MODEL_PATH``) is
    used. If neither is available, noise injection is disabled and the
    images are rendered without noise.

    Parameters
    ----------
    verbose: bool, optional
        Print which noise source was loaded. Defaults to `True`.

    Returns
    -------
    tuple[str, Any, dict]
        The noise source (``"blank_spm"``, ``"psd_model"`` or ``"none"``),
        the noise asset (texture, model or `None`) and diagnostics.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> source, asset, diag = sim.load_noise_asset(
    ...     verbose=False
    ... )  # doctest: +SKIP
    >>> source in ("blank_spm", "psd_model", "none")  # doctest: +SKIP
    True

    """

    noise_source = "none"
    noise_asset = None
    noise_diag: dict[str, Any] = {"source": "none"}

    if USE_BLANK_SPM_NOISE:
        blank_path = str(BLANK_SPM_PATH) if BLANK_SPM_PATH else ""
        if blank_path and os.path.isfile(blank_path):
            try:
                noise_rms_nm, noise_asset, noise_diag = load_blank_spm_noise(
                    blank_path,
                    target_rms_nm=TARGET_NOISE_RMS_NM,
                    verbose=verbose,
                )
                noise_source = "blank_spm"
                if verbose:
                    print(
                        "Loaded blank .spm noise texture  "
                        f"RMS ≈ {noise_rms_nm:.3f} nm"
                    )
            except Exception as e:
                print(
                    "blank .spm noise failed; trying PSD fallback. Error:", e
                )
        elif verbose:
            print(
                f"No blank .spm file found at {blank_path!r}; "
                "trying PSD fallback noise model."
            )

    if noise_source == "none" and USE_PSD_NOISE_FALLBACK:
        try:
            noise_asset = load_model(MODEL_PATH)
            noise_source = "psd_model"
            noise_diag = {
                "source": "psd_model",
                "model_path": str(MODEL_PATH),
                "method": PSD_NOISE_METHOD,
                "target_rms_nm": float(TARGET_NOISE_RMS_NM),
            }
            if verbose:
                print(f"Loaded PSD fallback noise model from: {MODEL_PATH}")
        except Exception as e:
            print(
                "PSD fallback noise model failed; proceeding without "
                "noise. Error:",
                e,
            )
            noise_asset = None

    if noise_source == "none" and verbose:
        print("Noise injection disabled.")
    return noise_source, noise_asset, noise_diag


def generate_chain(
    seed: int,
    contour_length_nm: float | None = None,
    add_decoy: bool = False,
    decoy_n_beads_range: tuple[int, int] = (2, 4),
    return_frames: bool = False,
) -> dict[str, Any]:
    """Build one relaxed ring of the requested contour length.

    MD pathway (``USE_MD = True``): the contour length is converted to
    beads (:func:`contour_to_beads`), the initial ring is lifted to
    ``BASE_Z`` (:func:`make_tangled_ring_initial_safe`) and relaxed onto
    the substrate with the annealed OpenMM force field
    (:func:`run_md_relaxation`). Implausible results (final contour
    outside 75-125 % of the request, or degenerate bonds) raise
    ``ValueError`` so the caller can try the next seed.

    Non-MD pathway: :func:`make_non_md_ring` draws a 2-D ring whose
    contour equals the request exactly.

    The crossings of the final chain seen from above are detected and an
    optional decoy fragment is generated.

    Parameters
    ----------
    seed: int
        Random seed (initial ring and Langevin integrator).
    contour_length_nm: float | None, optional
        Requested contour length in nm. Defaults to `None`, which uses
        ``CONTOUR_LENGTH_NM``.
    add_decoy: bool, optional
        Add a short decoy fragment near a corner. Defaults to `False`.
    decoy_n_beads_range: tuple[int, int], optional
        Bead-count range of the decoy. Defaults to ``(2, 4)``.
    return_frames: bool, optional
        Also return every MD frame (used for the animation). Defaults to
        `False`.

    Returns
    -------
    dict[str, Any]
        ``seed``, ``pathway``, ``requested_contour_nm``, ``n_beads``,
        ``bond_length_nm``, ``coords`` (``(N, 3)``, nm), ``crossings``,
        ``decoy_coords``, ``projected_contour_nm``, ``contour_3d_nm`` and,
        if requested, ``frames``.

    Raises
    ------
    ImportError
        If the MD pathway is selected but OpenMM is not installed.
    ValueError
        If the MD result is not plausible.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    A non-MD minicircle:

    >>> _ = sim.configure(USE_MD=False, BOND_LENGTH=7.362098159496696)
    >>> chain = sim.generate_chain(3, 115.26)
    >>> chain["pathway"], round(chain["projected_contour_nm"], 2)
    ('non_md', 115.26)
    >>> _ = sim.configure(USE_MD=True, BOND_LENGTH=4.5)

    """

    seed = int(seed)
    length = float(
        CONTOUR_LENGTH_NM if contour_length_nm is None else contour_length_nm
    )
    frames = None

    if USE_MD:
        if openmm is None:
            raise ImportError(
                "USE_MD = True needs OpenMM; install it or set "
                "USE_MD = False."
            )
        n_beads, bond = contour_to_beads(length, BOND_LENGTH, MD_MIN_BEADS)
        coords0 = make_tangled_ring_initial_safe(
            seed,
            n_beads,
            bond,
            persistence_bonds=PERSISTENCE_BONDS,
            base_z=BASE_Z,
        )
        frames = run_md_relaxation(
            coords0,
            seed=seed,
            n_frames=N_FRAMES,
            steps_per_frame=STEPS_PER_FRAME,
            bond_length=bond,
        )
        coords = np.asarray(frames[-1], dtype=np.float32)
        if not np.all(np.isfinite(coords)):
            raise ValueError("MD produced non-finite coordinates")
        lengths = closed_lengths(coords)
        ratio = lengths["actual_3d_contour_nm"] / length
        if not 0.75 <= ratio <= 1.25:
            raise ValueError(f"Implausible final contour ratio {ratio:.3f}")
        if lengths["min_bond_nm"] < 0.1 or lengths["max_bond_nm"] > 2 * bond:
            raise ValueError("Degenerate or grossly stretched final bond")
    else:
        coords = make_non_md_ring(seed, length)
        n_beads, bond = len(coords), length / len(coords)
        frames = [coords]

    crossings = find_polyline_crossings(
        coords[:, :2],
        min_separation_beads=CROSS_MIN_SEP_BEADS,
        merge_radius_nm=0.5,
    )

    decoy_coords = None
    if add_decoy:
        xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
        ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
        decoy_raw = make_decoy_chain(
            seed + 9999,
            n_beads_range=decoy_n_beads_range,
            bond_length=BOND_LENGTH,
        )
        decoy_coords = place_decoy_in_corner(
            decoy_raw,
            (xmin, xmax, ymin, ymax),
            corner_margin_nm=15.0,
            seed=seed,
        )

    chain = dict(
        seed=seed,
        pathway="md" if USE_MD else "non_md",
        requested_contour_nm=length,
        n_beads=int(n_beads),
        bond_length_nm=float(bond),
        coords=coords,
        crossings=crossings,
        decoy_coords=decoy_coords,
        projected_contour_nm=closed_contour_nm(coords),
        contour_3d_nm=closed_contour_nm(coords, use_z=True),
    )
    if return_frames:
        chain["frames"] = frames
    return chain


def render_chain_and_masks(
    chain: dict,
    noise_source: Literal["blank_spm", "psd_model", "none"] = "none",
    noise_asset: dict | np.ndarray | None = None,
    dna_height_nm: float | None = None,
    tip_radius_nm: float | None = None,
    target_noise_rms_nm: float | None = None,
) -> dict[str, Any]:
    """Render the AFM image, DNA mask and crossing mask of a chain.

    The field of view is ``NX`` by ``NY`` pixels of ``NM_PER_PIXEL`` nm,
    centred on the chain. A chain that does not fit (with a margin of the
    DNA width plus the tip radius) raises ``ValueError`` instead of being
    cropped, so dataset generation moves on to the next seed.

    The image is rendered with :func:`create_z_based_afm` using
    ``AFM_KW``; ``dna_height_nm`` and ``tip_radius_nm`` override the
    values in ``AFM_KW`` for this sample. Decoys appear in the image but
    not in the masks. Noise is added after rendering. Both masks use
    ``DNA_MASK_DIAMETER_NM`` as the DNA width, independently of the
    rendered width. Crossing heights are boosted at the crossings stored
    in ``chain`` (MD pathway) or at the crossings the non-MD renderer was
    calibrated with (detected at ``AFM_KW["min_separation_beads"]``).

    Parameters
    ----------
    chain: dict[str, Any]
        Output of :func:`generate_chain`.
    noise_source: "blank_spm" | "psd_model" | "none", optional
        Noise source from :func:`load_noise_asset`. Defaults to "none".
    noise_asset: dict | np.ndarray | None, optional
        Noise asset from :func:`load_noise_asset`. Defaults to `None`.
    dna_height_nm: float | None, optional
        Rendered DNA height in nm. Defaults to `None`, which uses
        ``AFM_KW["dna_height_nm"]``.
    tip_radius_nm: float | None, optional
        Tip radius in nm. Defaults to `None`, which uses
        ``AFM_KW["tip_radius_nm"]``.
    target_noise_rms_nm: float | None, optional
        Noise RMS in nm. Defaults to `None`, which uses
        ``TARGET_NOISE_RMS_NM``.

    Returns
    -------
    dict[str, Any]
        ``afm_img`` (noisy height map, nm), ``afm_img_clean``,
        ``dna_mask`` (uint8), ``cross_mask`` (float32, 0 to 1), ``extent``
        (nm), ``n_crossings``, ``decoy_coords`` and the rendering
        parameters used.

    Raises
    ------
    ValueError
        If the noise RMS is negative or the chain does not fit in the
        field of view.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    Render a non-MD minicircle without noise:

    >>> _ = sim.configure(USE_MD=False, BOND_LENGTH=7.362098159496696)
    >>> sample = sim.render_chain_and_masks(sim.generate_chain(3, 115.26))
    >>> sample["afm_img"].shape, int(sample["dna_mask"].max())
    ((512, 512), 1)
    >>> _ = sim.configure(USE_MD=True, BOND_LENGTH=4.5)

    """

    seed = int(chain["seed"])
    coords = chain["coords"]
    decoy_coords = chain.get("decoy_coords", None)

    afm_kw = dict(AFM_KW)
    for key in (
        "target_nm_per_px",
        "render_nm_per_px",
        "output_nm_per_px",
        "extent",
    ):
        afm_kw.pop(key, None)
    if dna_height_nm is not None:
        afm_kw["dna_height_nm"] = float(dna_height_nm)
    if tip_radius_nm is not None:
        afm_kw["tip_radius_nm"] = float(tip_radius_nm)
    noise_rms = float(
        TARGET_NOISE_RMS_NM
        if target_noise_rms_nm is None
        else target_noise_rms_nm
    )
    if not np.isfinite(noise_rms) or noise_rms < 0:
        raise ValueError(
            "target_noise_rms_nm must be finite and non-negative"
        )

    # ── Field of view centred on the chain ──────────────────────────────────
    pixel_nm = float(NM_PER_PIXEL)
    cx = (coords[:, 0].min() + coords[:, 0].max()) / 2.0
    cy = (coords[:, 1].min() + coords[:, 1].max()) / 2.0
    width_nm = (NX - 1) * pixel_nm
    height_nm = (NY - 1) * pixel_nm
    extent = (
        float(cx - width_nm / 2.0),
        float(cx + width_nm / 2.0),
        float(cy - height_nm / 2.0),
        float(cy + height_nm / 2.0),
    )
    margin_nm = float(afm_kw["dna_diameter_nm"]) + float(
        afm_kw["tip_radius_nm"]
    )
    if (
        coords[:, 0].min() < extent[0] + margin_nm
        or coords[:, 0].max() > extent[1] - margin_nm
        or coords[:, 1].min() < extent[2] + margin_nm
        or coords[:, 1].max() > extent[3] - margin_nm
    ):
        raise ValueError(
            "Chain does not fit in the field of view; it is never cropped. "
            "Use a larger NM_PER_PIXEL / NX / NY or a shorter contour."
        )

    if USE_MD:
        render_crossings = chain["crossings"]
    else:
        render_crossings = collect_intersections(
            coords[:, :2],
            min_separation_beads=int(afm_kw["min_separation_beads"]),
        )

    afm_kw.update(
        nx=NX,
        ny=NY,
        grain_seed=seed,
        crossings_precomputed=render_crossings,
        return_masks=True,
        return_crossing_info=True,
    )
    afm_img, dbg = create_z_based_afm(coords, extent=extent, **afm_kw)
    extent = tuple(dbg["extent"])

    # ── Decoy fragment (image only, excluded from the masks) ────────────────
    if decoy_coords is not None:
        decoy_coords = place_decoy_in_corner(
            decoy_coords, extent, corner_margin_nm=8.0, seed=seed
        )
        decoy_kw = dict(afm_kw)
        decoy_kw.update(
            apply_edge_taper=False,
            enable_crossing_boost=False,
            add_center_ridge=False,
            crossings_precomputed=None,
            return_crossing_info=False,
        )
        decoy_afm, _ = create_z_based_afm(
            decoy_coords, extent=extent, **decoy_kw
        )
        afm_img = np.maximum(afm_img, decoy_afm)

    # ── Noise ───────────────────────────────────────────────────────────────
    afm_img_clean = afm_img.copy()
    noise_img = sample_noise_image(
        noise_source=noise_source,
        noise_asset=noise_asset,
        seed=seed,
        out_shape=afm_img.shape,
        target_rms_nm=noise_rms,
        nm_per_pixel=pixel_nm,
        noise_native_nm_px=NOISE_NATIVE_NM_PER_PIXEL,
    )
    if noise_img is not None:
        afm_img = np.minimum(
            afm_img + noise_img.astype(np.float32), afm_kw["max_height_nm"]
        ).astype(np.float32)

    # ── Masks (annotation width DNA_MASK_DIAMETER_NM) ───────────────────────
    mask_width = float(DNA_MASK_DIAMETER_NM)
    dna_mask = make_dna_mask_from_beads(
        coords, extent, NX, NY, dna_diameter_nm=mask_width
    ).astype(np.uint8)
    xmin, xmax, ymin, ymax = extent
    output_nm_per_px = 0.5 * (
        (xmax - xmin) / max(NX - 1, 1) + (ymax - ymin) / max(NY - 1, 1)
    )
    sigma_center_px = max(
        0.5,
        CROSS_SIGMA_CENTER_DIAM_MULT
        * mask_width
        / max(output_nm_per_px, 1e-12),
    )
    sigma_perp_px = max(
        0.5,
        CROSS_SIGMA_PERP_DIAM_MULT
        * mask_width
        / max(output_nm_per_px, 1e-12),
    )
    cross_mask, crossings_out = make_crossing_mask(
        coords,
        extent,
        NX,
        NY,
        sigma_center_px=sigma_center_px,
        sigma_perp_px=sigma_perp_px,
        chain_extent=CROSS_CHAIN_EXTENT,
        center_weight=CROSS_CENTER_WEIGHT,
        chain_weight=CROSS_CHAIN_WEIGHT,
        min_separation_beads=CROSS_MIN_SEP_BEADS,
        crossings_precomputed=chain["crossings"],
        merge_radius_nm=0.5,
    )
    if CROSS_CLIP_TO_DNA_MASK:
        cross_mask = cross_mask * dna_mask.astype(np.float32)

    return dict(
        seed=seed,
        afm_img=afm_img.astype(np.float32),
        afm_img_clean=afm_img_clean.astype(np.float32),
        dna_mask=dna_mask,
        cross_mask=cross_mask.astype(np.float32),
        extent=np.array(extent, dtype=np.float32),
        n_crossings=int(len(crossings_out)),
        decoy_coords=decoy_coords,
        nm_per_pixel=float(output_nm_per_px),
        dna_height_nm=float(afm_kw["dna_height_nm"]),
        tip_radius_nm=float(afm_kw["tip_radius_nm"]),
        noise_rms_nm=noise_rms if noise_img is not None else 0.0,
        mask_width_nm=mask_width,
        crossing_sigma_center_px=float(sigma_center_px),
        crossing_sigma_perp_px=float(sigma_perp_px),
    )


def generate_one_sample(
    seed: int,
    contour_length_nm: float | None = None,
    noise_source: Literal["blank_spm", "psd_model", "none"] = "none",
    noise_asset: dict | np.ndarray | None = None,
    add_decoy: bool = False,
    decoy_n_beads_range: tuple[int, int] = (2, 4),
    dna_height_nm: float | None = None,
    tip_radius_nm: float | None = None,
    target_noise_rms_nm: float | None = None,
) -> dict[str, Any]:
    """Generate one complete sample: chain, AFM image and both masks.

    Calls :func:`generate_chain` and :func:`render_chain_and_masks` and
    keeps the chain description (coordinates, requested and measured
    contour lengths, bead count, bond length) with the rendered sample.

    Parameters
    ----------
    seed: int
        Random seed.
    contour_length_nm: float | None, optional
        Requested contour length in nm. Defaults to `None`, which uses
        ``CONTOUR_LENGTH_NM``.
    noise_source: "blank_spm" | "psd_model" | "none", optional
        Noise source from :func:`load_noise_asset`. Defaults to "none".
    noise_asset: dict | np.ndarray | None, optional
        Noise asset from :func:`load_noise_asset`. Defaults to `None`.
    add_decoy: bool, optional
        Add a decoy fragment. Defaults to `False`.
    decoy_n_beads_range: tuple[int, int], optional
        Bead-count range of the decoy. Defaults to ``(2, 4)``.
    dna_height_nm: float | None, optional
        Rendered DNA height in nm. Defaults to `None`.
    tip_radius_nm: float | None, optional
        Tip radius in nm. Defaults to `None`.
    target_noise_rms_nm: float | None, optional
        Noise RMS in nm. Defaults to `None`.

    Returns
    -------
    dict[str, Any]
        The rendered sample plus ``pathway``, ``requested_contour_nm``,
        ``n_beads``, ``bond_length_nm``, ``coords``,
        ``projected_contour_nm``, ``contour_3d_nm`` and ``has_decoy``.

    Raises
    ------
    ValueError
        If the chain is rejected or does not fit in the field of view.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> _ = sim.configure(USE_MD=False, BOND_LENGTH=7.362098159496696)
    >>> sample = sim.generate_one_sample(1, 115.26, add_decoy=True)
    >>> sample["has_decoy"], sample["afm_img"].shape
    (True, (512, 512))
    >>> _ = sim.configure(USE_MD=True, BOND_LENGTH=4.5)

    """

    chain = generate_chain(
        seed,
        contour_length_nm,
        add_decoy=add_decoy,
        decoy_n_beads_range=decoy_n_beads_range,
    )
    sample = render_chain_and_masks(
        chain,
        noise_source=noise_source,
        noise_asset=noise_asset,
        dna_height_nm=dna_height_nm,
        tip_radius_nm=tip_radius_nm,
        target_noise_rms_nm=target_noise_rms_nm,
    )
    for key in (
        "pathway",
        "requested_contour_nm",
        "n_beads",
        "bond_length_nm",
        "coords",
        "projected_contour_nm",
        "contour_3d_nm",
    ):
        sample[key] = chain[key]
    sample["has_decoy"] = chain["decoy_coords"] is not None
    return sample


# ────────────────────────────────────────────────────────────────────────────
# Sanity check
# ────────────────────────────────────────────────────────────────────────────


def nice_scale_bar_nm(field_nm: float) -> float:
    """Pick a round scale-bar length of about a fifth of the field.

    Parameters
    ----------
    field_nm: float
        Width of the field of view in nm.

    Returns
    -------
    float
        The largest of 500, 200, 100, 50, 20, 10, 5, 2 and 1 nm that is at
        most a fifth of the field (at least 1 nm).

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> sim.nice_scale_bar_nm(511.0), sim.nice_scale_bar_nm(60.0)
    (100.0, 10.0)

    """

    target = field_nm / 5.0
    for value in (500, 200, 100, 50, 20, 10, 5, 2, 1):
        if value <= target:
            return float(value)
    return 1.0


def add_scale_bar(
    ax: plt.Axes,
    extent: tuple[float, float, float, float],
    length_nm: float | None = None,
    color: str = "white",
) -> None:
    """Draw a labelled scale bar in the lower-left corner of an image.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        Axis showing an image drawn with ``extent=extent`` (nm) and
        ``origin="lower"``.
    extent: tuple[float, float, float, float]
        Image extent ``(xmin, xmax, ymin, ymax)`` in nm.
    length_nm: float | None, optional
        Bar length in nm. Defaults to `None`, which uses
        :func:`nice_scale_bar_nm`.
    color: str, optional
        Colour of the bar and its label. Defaults to "white".

    Returns
    -------
    None
        The bar is drawn on ``ax``.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    >>> fig, ax = plt.subplots()
    >>> _ = ax.imshow(np.zeros((10, 10)), extent=(0, 511, 0, 511))
    >>> sim.add_scale_bar(ax, (0, 511, 0, 511))
    >>> ax.texts[0].get_text()
    '100 nm'
    >>> plt.close(fig)

    """

    xmin, xmax, ymin, ymax = [float(v) for v in extent]
    length = length_nm or nice_scale_bar_nm(xmax - xmin)
    x0 = xmin + 0.05 * (xmax - xmin)
    y0 = ymin + 0.06 * (ymax - ymin)
    ax.plot(
        [x0, x0 + length],
        [y0, y0],
        color=color,
        linewidth=3,
        solid_capstyle="butt",
    )
    ax.text(
        x0 + length / 2,
        y0 + 0.025 * (ymax - ymin),
        f"{length:g} nm",
        color=color,
        ha="center",
        va="bottom",
        fontsize=9,
    )


def generate_sanity_samples(
    seeds: list[int],
    contour_lengths_nm: list[float],
    add_decoy: bool,
    noise_source: Literal["blank_spm", "psd_model", "none"],
    noise_asset: Any,
    max_extra_seeds: int = 20,
) -> list[dict[str, Any]]:
    """Generate one sample per (seed, contour length) pair.

    If a seed fails (for example the molecule does not fit in the field of
    view or the MD run is rejected), the following seeds are tried, up to
    ``max_extra_seeds``, and a note is printed.

    Parameters
    ----------
    seeds: list[int]
        Starting seed of each sample.
    contour_lengths_nm: list[float]
        Requested contour length of each sample in nm.
    add_decoy: bool
        Add a decoy fragment to every sample.
    noise_source: "blank_spm" | "psd_model" | "none"
        Noise source from :func:`load_noise_asset`.
    noise_asset: Any
        Noise asset from :func:`load_noise_asset`.
    max_extra_seeds: int, optional
        Number of following seeds to try when one fails. Defaults to 20.

    Returns
    -------
    list[dict[str, Any]]
        One sample (see :func:`generate_one_sample`) per pair.

    Raises
    ------
    RuntimeError
        If no seed in the allowed range gives a valid sample.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> _ = sim.configure(USE_MD=False, BOND_LENGTH=7.362098159496696)
    >>> samples = sim.generate_sanity_samples(
    ...     [1, 2], [115.26, 768.4], False, "none", None
    ... )
    >>> [round(s["projected_contour_nm"], 1) for s in samples]
    [115.3, 768.4]
    >>> _ = sim.configure(USE_MD=True, BOND_LENGTH=4.5)

    """

    samples = []
    for seed, length in zip(seeds, contour_lengths_nm):
        last_error = None
        for trial in range(max_extra_seeds + 1):
            try:
                samples.append(
                    generate_one_sample(
                        seed + trial,
                        length,
                        noise_source=noise_source,
                        noise_asset=noise_asset,
                        add_decoy=add_decoy,
                    )
                )
                if trial:
                    print(
                        f"Seed {seed} skipped ({last_error}); "
                        f"used seed {seed + trial} instead."
                    )
                break
            except Exception as exc:  # try the next seed
                last_error = f"{type(exc).__name__}: {exc}"
        else:
            raise RuntimeError(
                f"No valid sample for seeds {seed}-"
                f"{seed + max_extra_seeds}: {last_error}"
            )
    return samples


def plot_sanity_samples(samples: list[dict[str, Any]]) -> None:
    """Show image, DNA mask and crossing mask of each sample.

    Each row is one molecule. The title above the image gives the seed,
    the measured contour length (closed centre line seen from above) and
    the number of crossings. Every panel has a scale bar. Decoys appear in
    the image only.

    Parameters
    ----------
    samples: list[dict[str, Any]]
        Output of :func:`generate_sanity_samples`.

    Returns
    -------
    None
        The figure is shown in the notebook.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> samples = sim.generate_sanity_samples(
    ...     [1], [115.26], True, "none", None
    ... )  # doctest: +SKIP
    >>> sim.plot_sanity_samples(samples)  # doctest: +SKIP

    """

    fig, axes = plt.subplots(
        len(samples), 3, figsize=(15, 5 * len(samples)), squeeze=False
    )
    for r, s in enumerate(samples):
        ext = [float(v) for v in s["extent"]]
        ax1, ax2, ax3 = axes[r]
        ax1.imshow(
            s["afm_img"],
            cmap="afmhot",
            origin="lower",
            extent=ext,
            vmin=-0.5,
            vmax=4.0,
        )
        decoy = " · with decoy" if s.get("has_decoy") else ""
        crossings = f"{s['n_crossings']} crossing" + (
            "" if s["n_crossings"] == 1 else "s"
        )
        ax1.set_title(
            f"Seed {s['seed']} · contour ≈ {s['projected_contour_nm']:.1f} nm"
            f" · {crossings}{decoy}"
        )
        ax2.imshow(s["dna_mask"], cmap="gray", origin="lower", extent=ext)
        ax2.set_title("DNA mask (decoy excluded)")
        ax3.imshow(s["cross_mask"], cmap="magma", origin="lower", extent=ext)
        ax3.set_title("Crossing mask")
        for ax in (ax1, ax2, ax3):
            add_scale_bar(ax, ext)
            ax.axis("off")
    plt.tight_layout()
    plt.show()


# ────────────────────────────────────────────────────────────────────────────
# Benchmarking
# ────────────────────────────────────────────────────────────────────────────


class PeakMemTracker:
    """Track the peak resident memory of this process in a thread.

    Between `start()` and `stop()`, a background thread reads the resident
    set size of the process every `poll_interval_s` seconds and keeps the
    largest value. The benchmark in section 7 uses it to report the peak
    memory of the pipeline.

    Parameters
    ----------
    poll_interval_s: float, optional
        Time between two readings in seconds. Defaults to 0.05.

    Attributes
    ----------
    peak_mb: float
        Largest resident set size seen while tracking, in MB.

    Methods
    -------
    `start() -> None`
        Start reading the memory in a background thread.
    `stop() -> None`
        Stop reading and wait for the thread to finish.

    Examples
    --------
    >>> import time

    >>> tracker = PeakMemTracker()
    >>> tracker.start()
    >>> time.sleep(0.2)
    >>> tracker.stop()
    >>> tracker.peak_mb > 0
    True

    """

    peak_mb: float
    _process: psutil.Process
    _poll_interval_s: float
    _stop_event: threading.Event
    _thread: threading.Thread | None

    def __init__(
        self: PeakMemTracker,
        poll_interval_s: float = 0.05,
    ) -> None:
        """Initialize the tracker without starting it.

        Parameters
        ----------
        poll_interval_s: float, optional
            Time between two readings in seconds. Defaults to 0.05.

        """

        self._process = psutil.Process(os.getpid())
        self._poll_interval_s = float(poll_interval_s)
        self._stop_event = threading.Event()
        self._thread = None
        self.peak_mb = 0.0

    def _run(self: PeakMemTracker) -> None:
        """Read the memory until the stop event is set.

        Returns
        -------
        None
            `peak_mb` is updated in place.

        """

        while not self._stop_event.is_set():
            try:
                rss_mb = self._process.memory_info().rss / 1e6
                self.peak_mb = max(self.peak_mb, rss_mb)
            except Exception:
                # The process information can be briefly unavailable.
                pass
            self._stop_event.wait(self._poll_interval_s)

    def start(self: PeakMemTracker) -> None:
        """Start reading the memory in a background thread.

        Returns
        -------
        None
            The thread runs until `stop()` is called.

        Examples
        --------
        >>> tracker = PeakMemTracker()
        >>> tracker.start()
        >>> tracker.stop()

        """

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self: PeakMemTracker) -> None:
        """Stop reading the memory and wait for the thread to finish.

        Returns
        -------
        None
            `peak_mb` holds the peak seen since `start()`.

        Examples
        --------
        >>> tracker = PeakMemTracker()
        >>> tracker.start()
        >>> tracker.stop()
        >>> isinstance(tracker.peak_mb, float)
        True

        """

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()



def bench_stages(
    seed: int,
    contour_length_nm: float,
) -> dict[str, float]:
    """Time the main stages of one sample.

    Parameters
    ----------
    seed: int
        Random seed.
    contour_length_nm: float
        Requested contour length in nm.

    Returns
    -------
    dict[str, float]
        Seconds spent in ``"1_chain_init"``, ``"2_md_relax"`` (the non-MD
        ring for the non-MD pathway), ``"3_crossing_detect"`` and
        ``"4_afm_render_masks"``.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> _ = sim.configure(USE_MD=False, BOND_LENGTH=7.362098159496696)
    >>> timings = sim.bench_stages(1, 115.26)
    >>> sorted(timings)
    ['1_chain_init', '2_md_relax', '3_crossing_detect', '4_afm_render_masks']
    >>> _ = sim.configure(USE_MD=True, BOND_LENGTH=4.5)

    """

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    if USE_MD:
        n_beads, bond = contour_to_beads(
            contour_length_nm, BOND_LENGTH, MD_MIN_BEADS
        )
        coords0 = make_tangled_ring_initial_safe(
            int(seed),
            n_beads,
            bond,
            persistence_bonds=PERSISTENCE_BONDS,
            base_z=BASE_Z,
        )
        timings["1_chain_init"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        frames = run_md_relaxation(
            coords0,
            seed=int(seed),
            n_frames=N_FRAMES,
            steps_per_frame=STEPS_PER_FRAME,
            bond_length=bond,
        )
        coords = np.asarray(frames[-1], dtype=np.float32)
        timings["2_md_relax"] = time.perf_counter() - t0
    else:
        timings["1_chain_init"] = 0.0
        coords = make_non_md_ring(int(seed), contour_length_nm)
        timings["2_md_relax"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    crossings = find_polyline_crossings(
        coords[:, :2],
        min_separation_beads=CROSS_MIN_SEP_BEADS,
        merge_radius_nm=0.5,
    )
    timings["3_crossing_detect"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    chain = dict(
        seed=int(seed), coords=coords, crossings=crossings, decoy_coords=None
    )
    try:
        render_chain_and_masks(chain)
    except ValueError:
        # A molecule that does not fit is still a valid timing of the
        # render set-up.
        pass
    timings["4_afm_render_masks"] = time.perf_counter() - t0
    return timings


def benchmark_pipeline(
    seed: int,
    contour_length_nm: float,
    n_reps: int = 3,
    total_samples: int | None = None,
    tracker_cls: type | None = None,
    print_report: bool = True,
) -> dict[str, Any]:
    """Benchmark sample generation and project the dataset run time.

    Prints a system profile, generates ``n_reps`` timed samples (seeds
    ``seed`` to ``seed + n_reps - 1``), records the peak memory with
    ``tracker_cls`` (normally this module's ``PeakMemTracker`` class) and
    projects the total time for ``total_samples`` samples.

    Parameters
    ----------
    seed: int
        First random seed.
    contour_length_nm: float
        Contour length of the benchmarked molecules in nm.
    n_reps: int, optional
        Number of repetitions. Defaults to 3.
    total_samples: int | None, optional
        Dataset size for the projection. Defaults to `None` (no
        projection).
    tracker_cls: type | None, optional
        Class with ``start()``, ``stop()`` and ``peak_mb``. Defaults to
        `None` (no memory tracking).
    print_report: bool, optional
        Print the report. Defaults to `True`.

    Returns
    -------
    dict[str, Any]
        Wall times, stage timings, medians, peak memory and the projected
        total hours.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> _ = sim.configure(USE_MD=False, BOND_LENGTH=7.362098159496696)
    >>> result = sim.benchmark_pipeline(
    ...     1, 115.26, n_reps=1, print_report=False
    ... )
    >>> result["mode"], len(result["wall_times_s"])
    ('non_md', 1)
    >>> _ = sim.configure(USE_MD=True, BOND_LENGTH=4.5)

    """

    import psutil

    cpu_logical = psutil.cpu_count(logical=True)
    available_ram_gb = psutil.virtual_memory().available / 1e9
    platforms = None
    if USE_MD and openmm is not None:
        platforms = [
            openmm.Platform.getPlatform(i).getName()
            for i in range(openmm.Platform.getNumPlatforms())
        ]

    if print_report:
        print("=" * 62)
        print("  SYSTEM PROFILE")
        print("=" * 62)
        print(f"  OS              : {platform.system()} {platform.release()}")
        print(f"  CPU (logical)   : {cpu_logical} cores")
        print(f"  RAM available   : {available_ram_gb:.1f} GB")
        if platforms is not None:
            print(f"  OpenMM Platforms: {platforms}")
        print(
            "  Mode            : "
            f"{'MD (OpenMM)' if USE_MD else 'Non-MD (2-D ring)'}"
        )
        print(f"  Contour length  : {contour_length_nm:g} nm")
        print()
        print("=" * 62)
        print("  RUNNING BENCHMARK")
        print("=" * 62)

    tracker = tracker_cls() if tracker_cls is not None else None
    if tracker is not None:
        tracker.start()
    wall_times, stage_timings = [], []
    for rep in range(int(n_reps)):
        t0 = time.perf_counter()
        stage_timings.append(bench_stages(int(seed) + rep, contour_length_nm))
        wall_times.append(time.perf_counter() - t0)
        if print_report:
            print(f"  Rep {rep + 1}/{n_reps}: {wall_times[-1]:.2f}s")
    if tracker is not None:
        tracker.stop()

    median_wall = statistics.median(wall_times)
    median_stages = {
        key: statistics.median(t[key] for t in stage_timings)
        for key in stage_timings[0]
    }
    projected_hours = (
        median_wall * int(total_samples) / 3600 if total_samples else None
    )
    peak_mb = getattr(tracker, "peak_mb", None)

    if print_report:
        print("=" * 62)
        print(f"  Median Wall Time: {median_wall:.2f} s")
        for key, value in median_stages.items():
            print(f"    {key:<20s}: {value:.2f} s")
        if peak_mb is not None:
            print(f"  Peak RAM (RSS)  : {peak_mb:.1f} MB")
        if projected_hours is not None:
            print(
                f"  Projected Total : {projected_hours:.1f} hours for "
                f"{int(total_samples)} samples"
            )
        print("=" * 62)

    return dict(
        seed=int(seed),
        contour_length_nm=float(contour_length_nm),
        n_reps=int(n_reps),
        mode="md" if USE_MD else "non_md",
        cpu_logical=cpu_logical,
        available_ram_gb=available_ram_gb,
        openmm_platforms=platforms,
        wall_times_s=wall_times,
        stage_timings_s=stage_timings,
        median_wall_time_s=median_wall,
        median_stage_times_s=median_stages,
        peak_rss_mb=peak_mb,
        total_samples=total_samples,
        projected_total_hours=projected_hours,
    )


# ────────────────────────────────────────────────────────────────────────────
# Dataset generation
# ────────────────────────────────────────────────────────────────────────────


def draw_value(
    spec: float | list[float] | tuple[float, float],
    rng: np.random.Generator,
) -> float:
    """Draw one value from a setting given as a number, list or range.

    - a single number: always that number;
    - a list ``[a, b, c]``: one of the listed values, each equally likely;
    - a tuple ``(low, high)``: uniformly distributed between ``low`` and
      ``high``.

    Parameters
    ----------
    spec: float | list[float] | tuple[float, float]
        The setting as written in the notebook.
    rng: np.random.Generator
        Random generator seeded per sample.

    Returns
    -------
    float
        The drawn value.

    Raises
    ------
    ValueError
        For an empty list or a tuple that is not ``(low, high)`` with
        ``low <= high``.

    Examples
    --------
    >>> import numpy as np
    >>> import dna_simulation_helpers as sim

    >>> rng = np.random.default_rng(0)
    >>> sim.draw_value(2.3, rng)
    2.3
    >>> sim.draw_value([115.26, 768.4], rng) in (115.26, 768.4)
    True
    >>> 6.0 <= sim.draw_value((6.0, 8.0), rng) <= 8.0
    True

    """

    if isinstance(spec, tuple):
        if len(spec) != 2 or float(spec[0]) > float(spec[1]):
            raise ValueError(
                f"A range must be (low, high) with low <= high, got {spec}"
            )
        return float(rng.uniform(float(spec[0]), float(spec[1])))
    if isinstance(spec, list):
        if not spec:
            raise ValueError("A list of discrete values must not be empty")
        return float(spec[int(rng.integers(0, len(spec)))])
    return float(spec)


def describe_spec(
    spec: float | list[float] | tuple[float, float],
) -> str:
    """Describe a number, list or range setting in words.

    Parameters
    ----------
    spec: float | list[float] | tuple[float, float]
        The setting as written in the notebook.

    Returns
    -------
    str
        For example ``"one of 1.8, 2.3, 2.8"``.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> sim.describe_spec([1.8, 2.3])
    'one of 1.8, 2.3'
    >>> sim.describe_spec((6.0, 8.0))
    'uniform between 6 and 8'
    >>> sim.describe_spec(0.1) == "fixed at 0.1"
    True

    """

    if isinstance(spec, tuple):
        return f"uniform between {spec[0]:g} and {spec[1]:g}"
    if isinstance(spec, list):
        return "one of " + ", ".join(f"{v:g}" for v in spec)
    return f"fixed at {float(spec):g}"


_SAMPLE_EXECUTOR = None


def get_sample_executor() -> Any:
    """Return the single-worker process pool used for dataset generation.

    Each sample runs in a separate Python process, so a time-out works on
    every operating system. The worker is created when first needed and
    then reused, which avoids starting a new process for every seed.
    ``joblib`` is installed if it is missing.

    Returns
    -------
    Any
        A reusable ``loky`` executor running one task at a time.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> executor = sim.get_sample_executor()  # doctest: +SKIP
    >>> executor.submit(sum, [1, 2]).result()  # doctest: +SKIP
    3
    >>> sim.reset_sample_executor()  # doctest: +SKIP

    """

    global _SAMPLE_EXECUTOR
    if _SAMPLE_EXECUTOR is None:
        try:
            from joblib.externals.loky import get_reusable_executor
        except ImportError:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "joblib"]
            )
            from joblib.externals.loky import get_reusable_executor
        _SAMPLE_EXECUTOR = get_reusable_executor(max_workers=1)
    return _SAMPLE_EXECUTOR


def reset_sample_executor() -> None:
    """Shut down and kill the dataset worker.

    Called after a time-out or an error, so that an OpenMM run does not
    continue in the background after its seed has been skipped.

    Returns
    -------
    None
        The next :func:`get_sample_executor` call starts a new worker.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> sim.reset_sample_executor()

    """

    global _SAMPLE_EXECUTOR
    if _SAMPLE_EXECUTOR is not None:
        _SAMPLE_EXECUTOR.shutdown(wait=False, kill_workers=True)
    _SAMPLE_EXECUTOR = None


_WORKER_STATE: dict[str, Any] = {}


def generate_sample_worker(
    settings: dict[str, Any],
    noise_source: Literal["blank_spm", "psd_model", "none"],
    noise_asset: Any,
    seed: int,
    contour_length_nm: float,
    add_decoy: bool,
    decoy_n_beads_range: tuple[int, int],
    dna_height_nm: float,
    tip_radius_nm: float,
    noise_rms_nm: float,
) -> dict[str, Any]:
    """Generate one sample inside the worker process.

    The worker is a fresh Python process, so the notebook settings are
    passed along and applied with :func:`configure` before the sample is
    generated with :func:`generate_one_sample`.

    Parameters
    ----------
    settings: dict[str, Any]
        Settings from :func:`current_settings`.
    noise_source: "blank_spm" | "psd_model" | "none"
        Noise source from :func:`load_noise_asset`.
    noise_asset: Any
        Noise asset from :func:`load_noise_asset`.
    seed: int
        Random seed.
    contour_length_nm: float
        Requested contour length in nm.
    add_decoy: bool
        Add a decoy fragment.
    decoy_n_beads_range: tuple[int, int]
        Bead-count range of the decoy.
    dna_height_nm: float
        Rendered DNA height in nm.
    tip_radius_nm: float
        Tip radius in nm.
    noise_rms_nm: float
        Noise RMS in nm.

    Returns
    -------
    dict[str, Any]
        The sample from :func:`generate_one_sample`.

    Raises
    ------
    ValueError
        If the chain is rejected or does not fit in the field of view.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> settings = dict(sim.current_settings(), USE_MD=False,
    ...                 BOND_LENGTH=7.362098159496696)
    >>> sample = sim.generate_sample_worker(
    ...     settings, "none", None, 1, 115.26, False, (2, 4), 2.7, 8.7, 0.0
    ... )
    >>> sample["pathway"]
    'non_md'
    >>> _ = sim.configure(USE_MD=True, BOND_LENGTH=4.5)

    """

    configure(settings)
    return generate_one_sample(
        int(seed),
        float(contour_length_nm),
        noise_source=noise_source,
        noise_asset=noise_asset,
        add_decoy=bool(add_decoy),
        decoy_n_beads_range=tuple(int(x) for x in decoy_n_beads_range),
        dna_height_nm=float(dna_height_nm),
        tip_radius_nm=float(tip_radius_nm),
        target_noise_rms_nm=float(noise_rms_nm),
    )


def generate_sample_with_timeout(
    noise_source: Literal["blank_spm", "psd_model", "none"],
    noise_asset: Any,
    seed: int,
    contour_length_nm: float,
    add_decoy: bool,
    decoy_n_beads_range: tuple[int, int],
    dna_height_nm: float,
    tip_radius_nm: float,
    noise_rms_nm: float,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Generate and check one sample, giving up after ``timeout_s``.

    The sample is generated in the worker process
    (:func:`generate_sample_worker`). If it takes longer than the time-out
    the worker is killed; errors from OpenMM or the simulator are passed
    on. In both cases the caller skips the seed.

    Parameters
    ----------
    noise_source: "blank_spm" | "psd_model" | "none"
        Noise source from :func:`load_noise_asset`.
    noise_asset: Any
        Noise asset from :func:`load_noise_asset`.
    seed: int
        Random seed.
    contour_length_nm: float
        Requested contour length in nm.
    add_decoy: bool
        Add a decoy fragment.
    decoy_n_beads_range: tuple[int, int]
        Bead-count range of the decoy.
    dna_height_nm: float
        Rendered DNA height in nm.
    tip_radius_nm: float
        Tip radius in nm.
    noise_rms_nm: float
        Noise RMS in nm.
    timeout_s: float | None, optional
        Time limit in seconds. Defaults to `None`, which uses
        ``SAMPLE_TIMEOUT_S``.

    Returns
    -------
    dict[str, Any]
        The checked sample.

    Raises
    ------
    TimeoutError
        If generation takes longer than the time limit.
    ValueError
        If an output contains non-finite values or the simulator rejects
        the seed.

    Examples
    --------
    >>> import dna_simulation_helpers as sim

    >>> sample = sim.generate_sample_with_timeout(
    ...     "none", None, 1, 768.4, False, (2, 4), 2.3, 7.0, 0.0,
    ...     timeout_s=60,
    ... )  # doctest: +SKIP
    >>> sample["afm_img"].shape  # doctest: +SKIP
    (512, 512)

    """

    timeout = float(SAMPLE_TIMEOUT_S if timeout_s is None else timeout_s)
    executor = get_sample_executor()
    start = time.perf_counter()
    future = executor.submit(
        generate_sample_worker,
        current_settings(),
        noise_source,
        noise_asset,
        int(seed),
        float(contour_length_nm),
        bool(add_decoy),
        tuple(int(x) for x in decoy_n_beads_range),
        float(dna_height_nm),
        float(tip_radius_nm),
        float(noise_rms_nm),
    )
    try:
        sample = future.result(timeout=timeout)
    except FutureTimeoutError as exc:
        future.cancel()
        reset_sample_executor()
        raise TimeoutError("Simulation took too long!") from exc
    except Exception:
        future.cancel()
        reset_sample_executor()
        raise
    if time.perf_counter() - start > timeout:
        reset_sample_executor()
        raise TimeoutError("Simulation took too long!")
    for key in ("afm_img", "dna_mask", "cross_mask"):
        if not np.all(np.isfinite(np.asarray(sample[key]))):
            raise ValueError(f"Non-finite values found in {key}")
    return sample


def save_preview_png(
    sample: dict[str, Any],
    idx: int,
    preview_dir: str,
) -> str:
    """Save a PNG preview of a generated image (no masks).

    The image is shown with the AFM colour map, a scale bar and a title
    with the index, seed and contour length.

    Parameters
    ----------
    sample: dict[str, Any]
        Sample with ``afm_img``, ``extent``, ``seed`` and
        ``projected_contour_nm``.
    idx: int
        Dataset index used in the file name.
    preview_dir: str
        Output folder.

    Returns
    -------
    str
        Path of the written ``preview_XXXX.png``.

    Examples
    --------
    >>> import tempfile
    >>> import dna_simulation_helpers as sim

    >>> _ = sim.configure(USE_MD=False, BOND_LENGTH=7.362098159496696)
    >>> sample = sim.generate_one_sample(1, 115.26)
    >>> path = sim.save_preview_png(sample, 0, tempfile.mkdtemp())
    >>> path.endswith("preview_0000.png")
    True
    >>> _ = sim.configure(USE_MD=True, BOND_LENGTH=4.5)

    """

    preview_path = os.path.join(preview_dir, f"preview_{idx:04d}.png")
    ext = [float(v) for v in sample["extent"]]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(
        sample["afm_img"],
        cmap="afmhot",
        origin="lower",
        extent=ext,
        vmin=-0.5,
        vmax=4.0,
    )
    add_scale_bar(ax, ext)
    ax.set_title(
        f"#{idx} · seed {sample['seed']} · "
        f"{sample['projected_contour_nm']:.1f} nm",
        fontsize=9,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(preview_path, dpi=150)
    plt.close(fig)
    return preview_path


MANIFEST_HEADER = [
    "index",
    "seed",
    "pathway",
    "requested_contour_nm",
    "projected_contour_nm",
    "contour_3d_nm",
    "n_beads",
    "bond_length_nm",
    "persistence_bonds",
    "dna_height_nm",
    "tip_radius_nm",
    "noise_rms_nm",
    "nm_per_pixel",
    "image_npy",
    "dna_mask_npy",
    "cross_mask_npy",
    "meta_npz",
    "n_crossings",
    "has_decoy",
    "afm_params_json",
]


def generate_dataset(
    contour_length_range_nm: RangeSpec,
    dna_diameter_range_nm: RangeSpec,
    tip_radius_range_nm: RangeSpec,
    noise_rms_range_nm: RangeSpec,
    noise_source: Literal["blank_spm", "psd_model", "none"],
    noise_asset: Any,
    export_preview_pngs: bool = False,
    preview_png_indices: Iterable[int] = (),
    out_dir: str | None = None,
    n_samples: int | None = None,
    base_seed: int | None = None,
    timeout_s: float | None = None,
) -> str:
    """Generate the dataset and write ``manifest.csv``.

    For every sample, a random generator seeded with the current seed
    draws the contour length, DNA height ("diameter"), tip radius and
    noise RMS with :func:`draw_value`. Every
    ``DECOY_EVERY_N_SAMPLES``-th sample gets a decoy fragment. Seeds that
    fail (time-out, rejected MD run, molecule too large for the field of
    view) are skipped and the next seed is used, so exactly ``n_samples``
    samples are written.

    Files written under ``out_dir``: ``images/img_XXXX.npy`` (height in
    nm), ``dna_masks/dna_XXXX.npy``, ``cross_masks/cross_XXXX.npy``,
    ``meta/meta_XXXX.npz`` (coordinates and parameters), optional
    ``preview_pngs/preview_XXXX.png``, and ``manifest.csv`` linking them.

    Parameters
    ----------
    contour_length_range_nm: float | list[float] | tuple[float, float]
        Contour lengths in nm (number, list or range).
    dna_diameter_range_nm: float | list[float] | tuple[float, float]
        Rendered DNA heights in nm.
    tip_radius_range_nm: float | list[float] | tuple[float, float]
        Tip radii in nm.
    noise_rms_range_nm: float | list[float] | tuple[float, float]
        Noise RMS values in nm.
    noise_source: "blank_spm" | "psd_model" | "none"
        Noise source from :func:`load_noise_asset`.
    noise_asset: Any
        Noise asset from :func:`load_noise_asset`.
    export_preview_pngs: bool, optional
        Write PNG previews. Defaults to `False`.
    preview_png_indices: Iterable[int], optional
        Dataset indices to preview. Defaults to ``()``.
    out_dir: str | None, optional
        Output folder. Defaults to `None`, which uses ``OUT_DIR``.
    n_samples: int | None, optional
        Number of samples. Defaults to `None`, which uses ``N_SAMPLES``.
    base_seed: int | None, optional
        First seed. Defaults to `None`, which uses ``BASE_SEED``.
    timeout_s: float | None, optional
        Time limit per sample. Defaults to `None`, which uses
        ``SAMPLE_TIMEOUT_S``.

    Returns
    -------
    str
        Path of ``manifest.csv``.

    Examples
    --------
    >>> import tempfile
    >>> import dna_simulation_helpers as sim

    Three non-MD samples without noise:

    >>> _ = sim.configure(USE_MD=False, BOND_LENGTH=7.362098159496696)
    >>> manifest = sim.generate_dataset(
    ...     [115.26, 768.4], 2.7, 8.7, 0.0, "none", None,
    ...     out_dir=tempfile.mkdtemp(), n_samples=3,
    ... )  # doctest: +SKIP
    >>> manifest.endswith("manifest.csv")  # doctest: +SKIP
    True

    """

    out_dir = str(OUT_DIR if out_dir is None else out_dir)
    n_samples = int(N_SAMPLES if n_samples is None else n_samples)
    seed = int(BASE_SEED if base_seed is None else base_seed)
    preview_indices = set(int(i) for i in preview_png_indices)
    preview_dir = os.path.join(out_dir, "preview_pngs")
    for sub in ("images", "dna_masks", "cross_masks", "meta"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    if export_preview_pngs:
        os.makedirs(preview_dir, exist_ok=True)

    manifest_path = os.path.join(out_dir, "manifest.csv")
    afm_params = json.dumps(
        dict(AFM_KW, mask_width_nm=DNA_MASK_DIAMETER_NM), default=str
    )
    skipped: dict[str, int] = {}
    started = time.perf_counter()
    print(
        f"Starting dataset generation: {n_samples} samples "
        f"({'MD' if USE_MD else 'non-MD'} pathway) → "
        f"{os.path.abspath(out_dir)}"
    )

    with open(manifest_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(MANIFEST_HEADER)
        idx = 0
        while idx < n_samples:
            rng = np.random.default_rng(seed)
            contour = draw_value(contour_length_range_nm, rng)
            height = draw_value(dna_diameter_range_nm, rng)
            tip = draw_value(tip_radius_range_nm, rng)
            noise_rms = draw_value(noise_rms_range_nm, rng)
            add_decoy = (
                bool(DECOY_EVERY_N_SAMPLES)
                and idx % int(DECOY_EVERY_N_SAMPLES) == 0
            )
            decoy_range = tuple(
                DECOY_N_BEADS_RANGE_POOL[
                    int(rng.integers(0, len(DECOY_N_BEADS_RANGE_POOL)))
                ]
            )
            try:
                sample = generate_sample_with_timeout(
                    noise_source,
                    noise_asset,
                    seed,
                    contour,
                    add_decoy,
                    decoy_range,
                    height,
                    tip,
                    noise_rms,
                    timeout_s=timeout_s,
                )
            except Exception as exc:
                reason = type(exc).__name__
                skipped[reason] = skipped.get(reason, 0) + 1
                print(f"Skipping seed {seed} due to {reason}: {exc}")
                seed += 1
                continue

            paths = dict(
                image_npy=os.path.join("images", f"img_{idx:04d}.npy"),
                dna_mask_npy=os.path.join("dna_masks", f"dna_{idx:04d}.npy"),
                cross_mask_npy=os.path.join(
                    "cross_masks", f"cross_{idx:04d}.npy"
                ),
                meta_npz=os.path.join("meta", f"meta_{idx:04d}.npz"),
            )
            np.save(
                os.path.join(out_dir, paths["image_npy"]), sample["afm_img"]
            )
            np.save(
                os.path.join(out_dir, paths["dna_mask_npy"]),
                sample["dna_mask"],
            )
            np.save(
                os.path.join(out_dir, paths["cross_mask_npy"]),
                sample["cross_mask"],
            )
            np.savez_compressed(
                os.path.join(out_dir, paths["meta_npz"]),
                seed=sample["seed"],
                coords=sample["coords"],
                extent=sample["extent"],
                requested_contour_nm=sample["requested_contour_nm"],
                projected_contour_nm=sample["projected_contour_nm"],
                n_beads=sample["n_beads"],
                bond_length_nm=sample["bond_length_nm"],
                n_crossings=sample["n_crossings"],
                has_decoy=sample["has_decoy"],
                dna_height_nm=sample["dna_height_nm"],
                tip_radius_nm=sample["tip_radius_nm"],
                noise_rms_nm=sample["noise_rms_nm"],
                nm_per_pixel=sample["nm_per_pixel"],
            )
            writer.writerow(
                [
                    idx,
                    sample["seed"],
                    sample["pathway"],
                    f"{sample['requested_contour_nm']:.4f}",
                    f"{sample['projected_contour_nm']:.4f}",
                    f"{sample['contour_3d_nm']:.4f}",
                    sample["n_beads"],
                    f"{sample['bond_length_nm']:.6f}",
                    PERSISTENCE_BONDS,
                    f"{sample['dna_height_nm']:.4f}",
                    f"{sample['tip_radius_nm']:.4f}",
                    f"{sample['noise_rms_nm']:.4f}",
                    f"{sample['nm_per_pixel']:.6f}",
                    paths["image_npy"],
                    paths["dna_mask_npy"],
                    paths["cross_mask_npy"],
                    paths["meta_npz"],
                    sample["n_crossings"],
                    sample["has_decoy"],
                    afm_params,
                ]
            )
            handle.flush()
            if export_preview_pngs and idx in preview_indices:
                save_preview_png(sample, idx, preview_dir)
            if (idx + 1) % 10 == 0 or idx + 1 == n_samples:
                elapsed = time.perf_counter() - started
                print(
                    f"Progress: {idx + 1}/{n_samples}  "
                    f"({elapsed / 60:.1f} min elapsed)"
                )
            idx += 1
            seed += 1

    reset_sample_executor()
    print("\nDone. Dataset written to:", os.path.abspath(out_dir))
    print("Manifest:", os.path.abspath(manifest_path))
    if skipped:
        print("Skipped seeds by reason:", skipped)
    return manifest_path
