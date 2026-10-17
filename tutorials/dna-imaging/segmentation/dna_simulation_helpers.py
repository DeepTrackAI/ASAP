"""Helper functions for Part 1 of ``DNA_simulation_and_segmentation.ipynb``.

This module holds every function used by the dataset-generation part of the
notebook: chain creation (with or without molecular dynamics), AFM height
rendering, noise injection, DNA and crossing masks, decoy fragments, the
sample pipeline, sanity-check plots, benchmarking and dataset generation.
Class definitions stay in the notebook.

The notebook defines its variables in ordinary cells and then hands them to
this module with ``configure(globals())``. Only the names listed in
``SETTING_NAMES`` are read; everything else in the notebook namespace is
ignored. The module-level values below are the defaults that apply before
``configure`` is called; they are the optimised MD settings.

Where the optimised values come from
------------------------------------
* MD pathway: the annealed OpenMM force field, the periodic-bridge initial
  ring and the width/taper renderer used to build the MD training data
  (``DNA_simulated_1000_v2``) behind the MD checkpoints.
* Non-MD pathway: the calibrated 2-D persistent-ring generator (run 8 fit plus
  the crossing-height calibration of 12 October 2026) used for the non-MD
  checkpoints.

Lengths are in nanometres throughout unless a name says otherwise.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import traceback
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
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

try:  # OpenMM is only needed for the MD pathway.
    import openmm
    import openmm.unit as unit
except ImportError:  # pragma: no cover - depends on the environment
    openmm = None
    unit = None


# ─────────────────────────────────────────────────────────────────────────────
# Default settings (overwritten by configure())
# ─────────────────────────────────────────────────────────────────────────────

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
    "USE_MD", "OUT_DIR", "N_SAMPLES", "BASE_SEED",
    "NX", "NY", "NM_PER_PIXEL", "CONTOUR_LENGTH_NM", "CHAIN_RESOLUTION_NM", "TIP_RADIUS_NM",
    "BOND_LENGTH", "PERSISTENCE_BONDS", "K_ANGLE",
    "BASE_Z", "N_FRAMES", "STEPS_PER_FRAME",
    "ANGLE_STIFFNESS_MULT", "NON_MD_ANGLE_LIMIT_RAD", "NON_MD_Z_STD_NM", "NON_MD_BASE_Z_NM",
    "MD_MIN_BEADS", "NON_MD_MIN_BEADS",
    "DNA_HEIGHT_NM", "AFM_KW", "DNA_MASK_DIAMETER_NM",
    "CROSS_MIN_SEP_BEADS", "CROSS_SIGMA_CENTER_PX", "CROSS_SIGMA_PERP_PX",
    "CROSS_SIGMA_CENTER_DIAM_MULT", "CROSS_SIGMA_PERP_DIAM_MULT", "CROSS_CHAIN_EXTENT",
    "CROSS_CENTER_WEIGHT", "CROSS_CHAIN_WEIGHT", "CROSS_CLIP_TO_DNA_MASK",
    "DECOY_EVERY_N_SAMPLES", "DECOY_N_BEADS_RANGE_POOL",
    "TARGET_NOISE_RMS_NM", "USE_BLANK_SPM_NOISE", "USE_PLANE_REMOVE", "USE_LINE_FLATTEN",
    "USE_BANDPASS_FILTER", "BLANK_SPM_PATH", "USE_PSD_NOISE_FALLBACK", "MODEL_PATH",
    "PSD_NOISE_METHOD", "PSD_NOISE_STD_SCALE", "NOISE_NATIVE_NM_PER_PIXEL",
    "SAMPLE_TIMEOUT_S",
)


def configure(namespace: dict | None = None, **overrides: Any) -> dict[str, Any]:
    """Copy notebook variables into this module.

    The notebook calls ``configure(globals())`` after its variable cells. Only
    names in ``SETTING_NAMES`` are taken; dictionaries and lists are copied so
    later edits in the notebook do not change the module behind its back
    (call ``configure`` again after changing a variable).

    Parameters
    ----------
    namespace : dict, optional
        Usually ``globals()`` of the notebook.
    **overrides
        Individual settings, applied after ``namespace``.

    Returns
    -------
    dict[str, Any]
        The settings that were applied.

    Examples
    --------
    >>> applied = configure({"NX": 256, "NY": 256, "SOMETHING_ELSE": 1})
    >>> sorted(applied)
    ['NX', 'NY']
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
    module["AFM_KW"] = dict(module["AFM_KW"], nx=int(module["NX"]), ny=int(module["NY"]))
    return applied


def current_settings() -> dict[str, Any]:
    """Return a picklable copy of the active settings (sent to worker processes)."""

    module = globals()
    return {name: module[name] for name in SETTING_NAMES}


# ─────────────────────────────────────────────────────────────────────────────
# Contour length → beads
# ─────────────────────────────────────────────────────────────────────────────


def contour_to_beads(
    contour_length_nm: float,
    bond_length_nm: float | None = None,
    min_beads: int | None = None,
) -> tuple[int, float]:
    """Convert a requested contour length into a bead count and bond length.

    The chain is a closed ring of ``n`` beads joined by ``n`` bonds, so its
    contour length is ``n * bond``. The bead count is the requested length
    divided by the preferred bond length, rounded to the nearest integer, and
    the bond is then stretched or shrunk slightly so that ``n * bond`` equals
    the requested length exactly. Asking for 1 nm more therefore gives a chain
    that is 1 nm longer, while the coarse-graining stays close to the
    preferred bond length (within half a bond spread over all beads).

    Parameters
    ----------
    contour_length_nm : float
        Requested contour length in nm.
    bond_length_nm : float, optional
        Preferred bond length in nm. Defaults to ``BOND_LENGTH``.
    min_beads : int, optional
        Smallest allowed bead count. Defaults to ``MD_MIN_BEADS`` for the MD
        pathway and ``NON_MD_MIN_BEADS`` otherwise.

    Returns
    -------
    tuple[int, float]
        ``(n_beads, actual_bond_length_nm)``.

    Raises
    ------
    ValueError
        If the contour or bond length is not finite and positive.

    Examples
    --------
    >>> contour_to_beads(440.0, 4.5, 8)
    (98, 4.489795918367347)
    """

    contour = float(contour_length_nm)
    bond = float(BOND_LENGTH if bond_length_nm is None else bond_length_nm)
    if not (np.isfinite(contour) and contour > 0 and np.isfinite(bond) and bond > 0):
        raise ValueError("contour_length_nm and bond_length_nm must be finite and positive")
    if min_beads is None:
        min_beads = MD_MIN_BEADS if USE_MD else NON_MD_MIN_BEADS
    n_beads = max(int(min_beads), int(round(contour / bond)))
    return n_beads, contour / n_beads


def closed_contour_nm(coords: np.ndarray, use_z: bool = False) -> float:
    """Return the contour length of a closed chain, including the closing bond.

    Parameters
    ----------
    coords : np.ndarray
        Bead coordinates with shape ``(N, 2)`` or ``(N, 3)`` in nm.
    use_z : bool, optional
        Include the z coordinate (3-D length). Defaults to ``False``, which
        measures the projected (x, y) length seen from above.

    Returns
    -------
    float
        Contour length in nm.
    """

    xyz = np.asarray(coords, dtype=np.float64)
    xyz = xyz[:, :3] if use_z else xyz[:, :2]
    return float(np.linalg.norm(np.roll(xyz, -1, axis=0) - xyz, axis=1).sum())


def closed_lengths(coords):
    """Record measured 3-D and projected contour separately from nominal N*b."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) < 3:
        raise ValueError("coords must be an (N,3) array with N>=3")
    if not np.all(np.isfinite(coords)):
        raise ValueError("non-finite coordinates")
    increments = np.roll(coords, -1, axis=0) - coords
    bonds = np.linalg.norm(increments, axis=1)
    return dict(actual_3d_contour_nm=float(bonds.sum()),
                projected_contour_nm=float(np.linalg.norm(increments[:, :2], axis=1).sum()),
                min_bond_nm=float(bonds.min()), max_bond_nm=float(bonds.max()),
                mean_bond_nm=float(bonds.mean()), std_bond_nm=float(bonds.std()))


def min_nonbonded_distance(coords):
    coords = np.asarray(coords, dtype=np.float64)
    n = len(coords)
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    indices = np.arange(n)
    separation = np.abs(indices[:, None] - indices[None, :])
    separation = np.minimum(separation, n - separation)
    distances[separation < 2] = np.inf
    return float(distances.min())


def _resample_closed(coords, n):
    closed = np.vstack([coords, coords[0]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 1e-12):
        raise ValueError("degenerate periodic bridge")
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    distances = np.linspace(0.0, cumulative[-1], n, endpoint=False)
    segments = np.searchsorted(cumulative, distances, side="right") - 1
    fractions = (distances - cumulative[segments]) / lengths[segments]
    return closed[segments] + fractions[:, None] * (closed[segments+1] - closed[segments])


# ─────────────────────────────────────────────────────────────────────────────
# Chain creation (3-D persistent random walk, MD pathway)
# ─────────────────────────────────────────────────────────────────────────────


def make_tangled_ring_initial_safe(seed, n_beads, bond_length,
                                   persistence_bonds=40.0, base_z=55.0,
                                   *, max_attempts=16, min_bond_fraction=0.5,
                                   min_nonbonded_fraction=0.5):
    """Return finite distinct initial beads without changing the MD force field.

    The arc-length interpolation follows the correlated random bridge. Scaling
    sets its nominal total length; individual bond lengths need not be identical
    before energy minimization. Bad initial geometry raises after bounded
    deterministic attempts. It never silently substitutes a circle or changes
    K_ANGLE, persistence_bonds, or the target contour. These geometric guards do
    not prove MD stability; retain finite-energy/coordinate checks during MD.
    """
    n = int(n_beads)
    if n != n_beads or n < 8:
        raise ValueError("n_beads must be an integer of at least 8")
    if int(seed) != seed or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    b = float(bond_length)
    p = float(persistence_bonds)
    if not (np.isfinite(b) and b > 0 and np.isfinite(p) and p > 0 and np.isfinite(base_z)):
        raise ValueError("bond length/persistence must be positive; base_z must be finite")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    for attempt in range(int(max_attempts)):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), attempt]))
        increments = np.empty((n, 3), dtype=np.float64)
        vec = np.array([1.0, 0.0, 0.0])
        for k in range(n):
            axis = rng.normal(size=3)
            axis -= axis.dot(vec)*vec
            if np.linalg.norm(axis) < 1e-12:
                basis = np.eye(3)[int(np.argmin(np.abs(vec)))]
                axis = basis - basis.dot(vec)*vec
            axis /= np.linalg.norm(axis)
            angle = rng.normal(scale=np.sqrt(1/p))
            vec = vec*np.cos(angle) + np.cross(axis, vec)*np.sin(angle)
            vec /= np.linalg.norm(vec)
            increments[k] = b*vec
        increments -= increments.mean(axis=0)
        coords = np.vstack([np.zeros((1, 3)), np.cumsum(increments[:-1], axis=0)])
        try:
            coords = _resample_closed(coords, n)
        except ValueError:
            continue
        coords -= coords.mean(axis=0)
        length = closed_lengths(coords)["actual_3d_contour_nm"]
        if length <= 1e-12:
            continue
        coords *= n*b/length
        coords[:, 2] += float(base_z)
        result = coords.astype(np.float32)
        measures = closed_lengths(result)
        if (measures["min_bond_nm"] >= min_bond_fraction*b
                and measures["max_bond_nm"] <= 1.5*b
                and min_nonbonded_distance(result) >= min_nonbonded_fraction*b):
            return result
    raise ValueError(f"No valid initial ring for seed={seed} after {max_attempts} attempts")


def plot_initial_chain(
    coords: np.ndarray,
    title: str | None = None,
) -> None:
    """Show a 3-D schematic of a chain above the substrate.

    The chain is drawn as connected beads, bead 0 is marked in red and the
    substrate (z = 0) is drawn as a transparent plane.

    Parameters
    ----------
    coords : np.ndarray
        Bead coordinates with shape ``(N, 3)`` in nm.
    title : str, optional
        Figure title. Defaults to a title with the bead count.
    """

    cc = np.vstack([coords, coords[0]])  # close the ring for plotting

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(cc[:, 0], cc[:, 1], cc[:, 2], "-o", markersize=3, linewidth=1.2)
    ax.scatter(cc[0, 0], cc[0, 1], cc[0, 2], color="red", s=60, zorder=5, label="bead 0")

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
    """Run OpenMM's own installation test (lists platforms and checks forces).

    Prints a short message instead when OpenMM is not installed, which is
    expected when only the non-MD pathway is used.
    """

    if openmm is None:
        print("OpenMM is not installed. This is fine for the non-MD pathway; "
              "install it (e.g. `pip install openmm`) to use USE_MD = True.")
        return
    from openmm import testInstallation

    try:
        testInstallation.main()
    except Exception as e:  # pragma: no cover - platform dependent
        print(f"Test failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Molecular dynamics relaxation (annealed OpenMM force field)
# ─────────────────────────────────────────────────────────────────────────────


def run_md_relaxation(
    coords0: np.ndarray,
    seed: int,
    n_frames: int = N_FRAMES,
    steps_per_frame: int = STEPS_PER_FRAME,
    anneal_fraction: float = 0.85,   # fraction of the run spent ramping soft -> stiff
    kz_soft: float = 0.0005,           # nm^-2; starting (soft) value of Kz
    khigh_soft: float = 0.0005,        # nm^-4; starting (soft) value of Khigh
    fads_soft: float = 0.005,         # dimensionless; starting (soft) value of Fads
    wca_sigma_soft_factor: float = 2.0,   # x conlen; starting (soft/large) WCA sigma
    wca_sigma_final_factor: float = 0.80,  # x conlen; final (stiff/small) WCA sigma
    bond_length: float | None = None,      # nm; defaults to BOND_LENGTH
) -> list[np.ndarray]:
    """Takes initial bead coordinates and runs Langevin dynamics using OpenMM

    This function uses the coordinates generated using the
    `make_tangled_ring_initial_safe` function to run a molecular dynamics
    simulation. There are certain forces that are defined and applied to the
    bead which are then resolved using OpenMM.
    The forces applied are:
    - Harmonic bonds + angle stiffness (``K_ANGLE``) along the ring
    - Surface attraction (linear in z) + hard wall at z = 0
    - Short-range Weeks–Chandler–Andersen (WCA)
      repulsion between non-bonded beads
    - A nonlinear height-confinement term (quadratic + quartic above a soft
      threshold) whose strength is annealed over the course of the run: it
      starts soft, so early frames can relax the ring's apparent crossing
      count via genuine strand-passage, and ramps up to its final value so
      later frames compress the adsorbed structure down to the target
      crossing height.
    The variable langevin integrator from OpenMM is used and the forces are
    resolved for 'steps_per_frame' steps for every frame in n_frames.

    To change how the chain relaxes, edit the force constants in this
    function (F_ADS, K_Z, K_HIGH, Z_SOFT, WALL_K, WCA_EPS_KT, temperature,
    collision rate) or the annealing arguments below. These are the
    optimised values used to generate the MD training data.

    Parameters
    ----------
    coords0 : np.ndarray
        The starting [x, y, z] coordinates of the polymer beads.
    seed : int
        Random seed for the Langevin integrator.
    n_frames : int, optional
        Number of frames to record during the simulation. Defaults to N_FRAMES.
    steps_per_frame : int, optional
        Number of integration steps to perform between recorded frames.
        Defaults to STEPS_PER_FRAME.
    anneal_fraction : float, optional
        Fraction (0-1) of the total integration steps over which Fads, Kz,
        and Khigh are linearly ramped from their soft starting values up to
        their final (module-level) values. After this fraction of the run,
        the confinement stays at its final, stiff value. Defaults to 0.85.
    kz_soft, khigh_soft, fads_soft : float, optional
        Starting ("soft") values for Kz, Khigh, and Fads at frame 0. These
        should be small enough that the ring can still relax laterally and
        pass strands over one another during the soft phase.
    wca_sigma_soft_factor, wca_sigma_final_factor : float, optional
        Starting ("soft") and final WCA excluded-volume diameter, each
        expressed as a multiple of the bond length (conlen). Unlike the
        confinement terms, this one ramps *down* over the run: it starts
        large so beads repel each other strongly at long range and resolve
        nonphysical crossings/overlaps early, then shrinks toward the
        physical DNA diameter so the chain can settle closer as it
        approaches its final, adsorbed conformation. Defaults to 2.0 and
        0.80 respectively.
    bond_length : float, optional
        Equilibrium bond length in nm (the actual bond from
        :func:`contour_to_beads`). Defaults to ``BOND_LENGTH``.

    Returns
    -------
    A list of coordinate arrays of shape (num_beads, 3). The list contains
    num_frames + 1 entries, where frame 0 is the pre-minimization geometry.

    Raises
    ------
    RuntimeError
        If no OpenMM platform (CUDA, OpenCL, or CPU) could be initialised.

    Examples
    --------
    >>> frames = run_md_relaxation(coords0, seed=1, n_frames=10,
    ...                            steps_per_frame=STEPS_PER_FRAME)  # doctest: +SKIP
    >>> len(frames), frames[0].shape  # doctest: +SKIP
    (11, (171, 3))

    """

    coords0 = np.asarray(coords0, dtype=np.float32)
    n       = int(coords0.shape[0])
    bond_length = float(BOND_LENGTH if bond_length is None else bond_length)

    # how big the bounding box for the molecular dynamics needs to be
    extent = (coords0.max(axis=0) - coords0.min(axis=0)).max()
    box    = float(extent + 50.0)


    # All lengths in nm, energies in kJ/mol.
    BOLTZ_kJmol    = 0.00831445 # kJ / (mol · K)
    temperature    = 100.0 # K  (sets energy scale kT ≈ 0.0083 kJ/mol)
    kT             = BOLTZ_kJmol * temperature # kJ/mol
    conlen         = bond_length # nm
    mass_amu       = 100.0 # Da per bead
    collision_rate = 0.6 # ps⁻¹
    error_tol      = 0.005

    # ── Build system ─────────────────────────────────────────────────────────
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box, 0, 0),
        openmm.Vec3(0, box, 0),
        openmm.Vec3(0, 0, box),
    )
    for _ in range(n):
        system.addParticle(mass_amu)

   # ── 1. Harmonic bonds ────────────────────────────────────────────────────

    # OpenMM HarmonicBondForce: E = (k/2)·(r − r0)²  →  k = kT / wiggle²
    bond_L = bond_length          # nm
    wiggle = 0.1                  # nm  (bondWiggleDistance)
    k_bond = kT / (wiggle ** 2)   # kJ/mol/nm²

    bond_force = openmm.HarmonicBondForce()
    bond_force.setUsesPeriodicBoundaryConditions(False)
    for i in range(n):
        bond_force.addBond(i, (i + 1) % n, bond_L, k_bond)
    system.addForce(bond_force)

    # ── 2. Bending (angle) stiffness ─────────────────────────────────────────
    # E = kT · k_angle · (1 − cos(θ − π))  — equilibrium at θ = π (straight)
    k_angle     = K_ANGLE
    angle_force = openmm.CustomAngleForce(
        "kT * kangle * (1 - cos(theta - 3.141592653589793))"
    )
    angle_force.addGlobalParameter("kT",     kT)
    angle_force.addGlobalParameter("kangle", k_angle)
    for i in range(n):
        angle_force.addAngle((i - 1) % n, i, (i + 1) % n, [])
    system.addForce(angle_force)

    # ── 3. Surface attraction and nonlinear height confinement ──────────────────
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
    # The quartic term strongly suppresses unusually tall crossings while having
    # little effect on beads below zsoft.
    #
    # Fads, Kz, and Khigh are annealed during the run (see the frame loop
    # below): they start at the *_soft values passed in, and ramp linearly
    # up to F_ADS/K_Z/K_HIGH over `anneal_fraction` of the total steps.
    #
    # For z < 0, a stiff harmonic wall prevents penetration into the substrate.

    F_ADS = 18.0           # dimensionless constant downward force (final value)
    K_Z = 5.0              # nm^-2; linearly increasing downward force (final value)
    K_HIGH = 27.0          # nm^-4; strong high-z confinement (final value)
    Z_SOFT = 2.000         # nm; quartic confinement begins above this height
    WALL_K = 500.0         # nm^-2; substrate penetration penalty

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


    # ── 4. WCA excluded-volume repulsion ─────────────────────────────────────────
    #
    # sigma is annealed over the run, but in the *opposite* direction to
    # Fads/Kz/Khigh: it starts large (wca_sigma_soft_factor * conlen) so beads
    # push apart strongly at long range and nonphysical crossings/overlaps get
    # resolved early, then shrinks to wca_sigma_final_factor * conlen (near the
    # physical DNA diameter) as the run progresses, letting the chain settle
    # closer together by the end. Follows the same anneal_fraction schedule.

    WCA_SIGMA_SOFT_NM  = wca_sigma_soft_factor * conlen
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

    # ── Integrator ───────────────────────────────────────────────────────────
    integrator = openmm.VariableLangevinIntegrator(temperature,
                                                   collision_rate, error_tol)
    integrator.setRandomNumberSeed(int(seed))

    # ── Platform selection: OpenCL → CPU fallback ────────────────────────────
    context = None
    # OpenCL enables Mac users to be able to use GPU acceleration as well
    for platform_name in ("CUDA","OpenCL","CPU"):
        try:
            platform = openmm.Platform.getPlatformByName(platform_name)
            context  = openmm.Context(system, integrator, platform)
            print(f"Using {platform_name} platform.")
            break
        except openmm.OpenMMException as e:
            print(f"  {platform_name} unavailable: {e}")
        except Exception as e:
            print(f"  {platform_name} failed unexpectedly: {e}")

    if context is None:
        raise RuntimeError("No OpenMM platform could be initialised.")

    # ── Set initial positions and box ────────────────────────────────────────
    positions = [
        openmm.Vec3(float(coords0[i, 0]),
                    float(coords0[i, 1]), float(coords0[i, 2]))
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
    pos   = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    frames.append(pos.copy())

    # Cap iterations
    openmm.LocalEnergyMinimizer.minimize(context,
                                         tolerance=50, maxIterations=200)

    # ── Record frames, annealing the confinement strength ───────────────────
    #
    # Fads, Kz, and Khigh are ramped linearly from their soft starting values
    # up to their final (F_ADS/K_Z/K_HIGH) values over the first
    # `anneal_fraction` of the total integration steps, then held constant.
    # This gives the ring room to reduce its apparent crossing count via
    # genuine strand-passage while confinement is still weak, then compresses
    # the resulting lower-crossing conformation down to the target height
    # once confinement reaches its final, stiff value.
    n_total_steps  = int(n_frames) * int(steps_per_frame)
    anneal_steps   = max(float(anneal_fraction) * n_total_steps, 1.0)
    step_count     = 0

    for _ in range(int(n_frames)):
        progress = min(step_count / anneal_steps, 1.0)
        context.setParameter("Fads",  fads_soft  + progress * (F_ADS  - fads_soft))
        context.setParameter("Kz",    kz_soft    + progress * (K_Z    - kz_soft))
        context.setParameter("Khigh", khigh_soft + progress * (K_HIGH - khigh_soft))
        # sigma ramps DOWN (soft/large -> final/small), opposite direction
        # from the confinement terms above.
        context.setParameter("sig",   WCA_SIGMA_SOFT_NM
                                       + progress * (WCA_SIGMA_FINAL_NM - WCA_SIGMA_SOFT_NM))

        integrator.step(int(steps_per_frame))
        step_count += int(steps_per_frame)

        state = context.getState(getPositions=True)
        pos   = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        frames.append(pos.copy())

    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Non-MD chain generation (2-D persistent ring)
# ─────────────────────────────────────────────────────────────────────────────


def make_ring_2d_persistent_initial(
    seed: int,
    n_beads: int,
    bond_length: float = BOND_LENGTH,                  # From Cell 1
    persistence_bonds: float = PERSISTENCE_BONDS,      # From Cell 1
    angle_stiffness_mult: float = ANGLE_STIFFNESS_MULT, # From Cell 1
    # Small Gaussian noise for Z
    z_std: float = 0.08,
    # Low altitude for "deposited" look
    base_z: float = 0.2,
    # Acts as a soft per-step cap
    angle_limit: float = np.pi/2,
) -> list[np.ndarray]:
    """Generate a closed 2D persistent ring and return it as [coords].

    The function builds a periodic sequence of turning angles, smooths the
    sequence to suppress sharp corners, and enforces total turning to ensure
    the tangent field is compatible with a closed loop. The resulting curve is
    resampled by arc length to maintain uniform bead spacing.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_beads : int
        Number of beads in the polymer ring.
    bond_length : float, optional
        The desired distance between consecutive beads.
        Defaults to BOND_LENGTH.
    persistence_bonds : float, optional
        Stiffness parameter; larger values result in a smoother ring with
        longer directional memory. Defaults to PERSISTENCE_BONDS.
    angle_stiffness_mult : float, optional
        Multiplier for angular fluctuations; larger values result in
        smaller fluctuations. Defaults to ANGLE_STIFFNESS_MULT.
    z_std : float, optional
        Standard deviation for the Gaussian noise applied to the z-axis.
        Defaults to 0.08.
    base_z : float, optional
        The baseline altitude for the deposited coordinates. Defaults to 0.2.
    angle_limit : float, optional
        A soft cap on local turning increments (radians). Defaults to np.pi/2.

    Returns
    -------
    coords : np.ndarray
        Array of shape (N, 3) containing the [x, y, z] coordinates

    Raises
    ------
    ValueError
        If num_beads is less than 6 or if the generated ring has zero
        contour length.

    Examples
    --------
    >>> coords = make_ring_2d_persistent_initial(seed=42)
    >>> print(coords.shape)
    (90, 3)

    """

    rng = np.random.default_rng(int(seed))
    n = int(n_beads)
    if n < 6:
        raise ValueError("n_beads must be at least"
                         "6 for a stable closed persistent ring.")

    bond_length = float(bond_length)

    # ------------------------------------------------------------------
    # 1) Build a periodic turning-angle process on the ring
    # ------------------------------------------------------------------

    # larger persistence_bonds / angle_stiffness_mult => less angular noise
    sigma_dtheta = (
        np.sqrt(2.0 / max(float(persistence_bonds), 1.0))
        / max(float(angle_stiffness_mult), 1e-6)
    )

    # Raw local turning increments
    dtheta = rng.normal(0.0, sigma_dtheta, size=n).astype(np.float64)
    dtheta = np.clip(dtheta, -float(angle_limit), float(angle_limit))

    # Circular smoothing of turning increments to suppress sharp local corners.
    # Stronger persistence / stiffness => broader smoothing.
    smooth_width = int(np.clip(np.round(max(float(persistence_bonds), 1.0)
                       / 6.0), 1, 8))
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

    # Re-clip gently after correction, then re-enforce total turning once more.
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
        raise ValueError("Degenerate ring generated;"
                         "total contour length is zero.")

    target_s = np.linspace(0.0, total_len, n + 1)[:-1]

    x_resamp = np.interp(target_s, s, closed_xy[:, 0])
    y_resamp = np.interp(target_s, s, closed_xy[:, 1])
    xy = np.stack([x_resamp, y_resamp], axis=1)

    # Final contour-length normalization
    # so mean bond length matches bond_length
    final_bonds = np.roll(xy, -1, axis=0) - xy
    mean_bond = np.mean(np.linalg.norm(final_bonds, axis=1))
    if mean_bond > 1e-12:
        xy *= (bond_length / mean_bond)

    # Center at origin
    xy -= xy.mean(axis=0, keepdims=True)

    # ------------------------------------------------------------------
    # 4) Assembly with preserved Gaussian z-noise logic
    # ------------------------------------------------------------------
    coords = np.zeros((n, 3), dtype=np.float32)
    coords[:, 0] = xy[:, 0].astype(np.float32)
    coords[:, 1] = xy[:, 1].astype(np.float32)
    coords[:, 2] = rng.normal(loc=base_z,
                              scale=z_std, size=n).astype(np.float32)

    # Return as a list containing one frame to mimic run_md_relaxation output
    return [coords]


def make_non_md_ring(
    seed: int,
    contour_length_nm: float | None = None,
) -> np.ndarray:
    """Generate a non-MD ring whose projected contour equals the request exactly.

    This is the calibrated non-MD generator: the requested contour length is
    converted to beads with :func:`contour_to_beads`, the closed 2-D
    persistent ring is drawn with :func:`make_ring_2d_persistent_initial`
    using the non-MD settings (``PERSISTENCE_BONDS``, ``ANGLE_STIFFNESS_MULT``,
    ``NON_MD_ANGLE_LIMIT_RAD``, ``NON_MD_Z_STD_NM``, ``NON_MD_BASE_Z_NM``), and
    the (x, y) coordinates are finally rescaled in float64 so that the closed
    contour, including the last-to-first bond, matches the request to
    numerical precision.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    contour_length_nm : float, optional
        Requested contour length in nm. Defaults to ``CONTOUR_LENGTH_NM``.

    Returns
    -------
    np.ndarray
        Bead coordinates with shape ``(N, 3)`` in nm (float32).

    Raises
    ------
    ArithmeticError
        If the rescaled contour differs from the request by more than 1e-6 nm.

    Examples
    --------
    >>> coords = make_non_md_ring(seed=1, contour_length_nm=115.26)
    >>> round(closed_contour_nm(coords), 3)
    115.26
    """

    length = float(CONTOUR_LENGTH_NM if contour_length_nm is None else contour_length_nm)
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
    # Include the closing bond and correct in float64 after the float32 generator.
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

    Used for the non-MD pathway, where the generated ring is already
    "deposited" on the substrate (there is no relaxation to animate).

    Parameters
    ----------
    coords : np.ndarray
        Bead coordinates with shape ``(N, 3)`` in nm.
    title : str, optional
        Figure title.
    """

    c = np.asarray(coords, dtype=np.float32)
    cc = np.vstack([c, c[0]])
    pad = 10.0
    x_min, x_max = c[:, 0].min() - pad, c[:, 0].max() + pad
    y_min, y_max = c[:, 1].min() - pad, c[:, 1].max() + pad

    fig = plt.figure(figsize=(12, 5.5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    X, Y = np.meshgrid(np.linspace(x_min, x_max, 2), np.linspace(y_min, y_max, 2))
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
    ax2d.set_title(f"Seen from above: contour {closed_contour_nm(c):.1f} nm, {len(c)} beads")

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MD animation
# ─────────────────────────────────────────────────────────────────────────────


def show_md_animation(
    frames: list[np.ndarray],
    interval_ms: int = 150,
) -> None:
    """Renders an in-notebook HTML5 animation of MD frames.

    This function is used to visualize the molecular dynamics simulation
    generated using the `run_md_relaxation` function.
    A plane has been added for visualization purposes and the range of
    absolute z values is also displayed.
    This function is purely for visualisation and does not mutate any arrays.

    Parameters
    ----------
    frames : list
        A list of coordinate arrays of shape (num_beads, 3).
    interval_ms : int, optional
        The time interval between frames in milliseconds. Defaults to 150.

    Returns
    -------
    None
        The function calls `IPython.display.display` to render the
        animation object directly in the notebook.

    Examples
    --------
    >>> show_md_animation(frames)

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
    ax  = fig.add_subplot(111, projection="3d")

    def _set_axes():
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z (nm)")

    def init():
        _set_axes()
        return []

    def update(i):
        ax.cla()
        c  = frames[i]
        # Clip beads to visible z range for plotting so outliers don't distort
        visible = (c[:, 2] >= z_min) & (c[:, 2] <= z_max)
        cc = np.vstack([c, c[0]])
        ax.plot(cc[:, 0], cc[:, 1], cc[:, 2], "-", linewidth=1)
        ax.scatter(c[visible, 0], c[visible, 1], c[visible, 2], s=5)
        ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.2, color="cyan")
        _set_axes()
        ax.set_title(f"Frame {i}/{len(frames)-1}  |  "
                     f"z ∈ [{c[:,2].min():.1f}, {c[:,2].max():.1f}] nm")
        return []

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), init_func=init,
        interval=int(interval_ms), blit=False
    )
    plt.close(fig)
    display(HTML(ani.to_jshtml()))


# ─────────────────────────────────────────────────────────────────────────────
# Height-based AFM rendering
# ─────────────────────────────────────────────────────────────────────────────


def _seg_intersect_2d(
    p1: np.ndarray,
    p2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    eps: float = 1e-12,
) -> tuple[bool, np.ndarray | None, float | None, float | None]:
    """Calculates the strict 2-D intersection of 2 line segments.

    This function is one of the functions used for the creation of the
    chain from the bead coordinates.
    The intersection point is defined by the parameters t and u such that:
    point = p1 + t * (p2 - p1) = q1 + u * (q2 - q1)

    Parameters
    ----------
    p1, p2 : np.ndarray
        Start and end points of the first line segment.
    q1, q2 : np.ndarray
        Start and end points of the second line segment.
    eps : float, optional
        A small positive value for numerical stability. Defaults to 1e-12.

    Returns
    -------
    is_intersecting : bool
        True if the segments intersect, False otherwise.
    intersection_point : np.ndarray or None
        The [x, y] coordinates of the intersection point, or None if no
        intersection exists.
    parameter_t : float or None
        The interpolation parameter along the first segment, or None.
    parameter_u : float or None
        The interpolation parameter along the second segment, or None.

    Examples
    --------
    >>> hit, pt, t, u = _seg_intersect_2d(p1, p2, q1, q2)
    >>> print(hit, pt, t, u)
    False None None None

    """

    p1 = np.asarray(p1, dtype=float); p2 = np.asarray(p2, dtype=float)
    q1 = np.asarray(q1, dtype=float); q2 = np.asarray(q2, dtype=float)
    r   = p2 - p1;  s = q2 - q1
    rxs = r[0]*s[1] - r[1]*s[0]
    if abs(rxs) < eps:
        return False, None, None, None
    qp = q1 - p1
    t  = (qp[0]*s[1] - qp[1]*s[0]) / rxs
    u  = (qp[0]*r[1] - qp[1]*r[0]) / rxs
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return True, p1 + t*r, t, u
    return False, None, None, None


def _circ_sep(
    n: int,
    i: int,
    j: int,
) -> int:
    """Circular distance between bead indices on a ring of n beads.

    This helper function determines the minimal separation between two beads
    along the perimeter of the ring, accounting for the periodic boundary.
    It is used to distinguish between beads that form physical crossings
    and those that are simply sequential neighbors. This function is used
    downstream for crossing calculations

    Parameters
    ----------
    n : int
        Number of beads in the ring.
    i, j : int
        Indices of the two beads for which to calculate the separation.

    Returns
    -------
    int
        The minimal distance between the two beads.

    Examples
    --------
    >>> d = _circ_sep(n, i, j)
    >>> d.type
    int

    """

    d = abs(int(i) - int(j))
    return min(d, n - d)


def disk_footprint(
    r_px: float,
) -> np.ndarray:
    """Boolean circular footprint of radius r_px pixels.

    This function creates a 2D boolean mask representing a disk of a
    specified radius. It is used to convert discrete bead coordinates into a
    continuous chain structure by dilating each bead coordinate into a
    disk-like structure. This is what builds the DNA chain radius.

    Parameters
    ----------
    r_px : float
        Radius of the disk in pixels.

    Returns
    -------
    np.ndarray
        A boolean array with True values inside the disk and false outside.

    Examples
    --------
    >>> footprint = disk_footprint(r_px)
    >>> print(footprint.shape, footprint.dtype)
    (5, 5) bool

    """

    if r_px < 0.5:
        return np.ones((1, 1), dtype=bool)
    r = int(np.ceil(r_px))
    y, x = np.ogrid[-r:r+1, -r:r+1]
    return (x*x + y*y) <= r*r


def upsample_nn(
    a: np.ndarray,
    out_shape: tuple[int, int],
) -> np.ndarray:
    """Nearest-neighbour upsample array a to out_shape.

    This function increases the resolution of an input array (such as an AFM
    height map or a boolean mask) by repeating elements. If the target
    dimensions are not exact multiples of the input dimensions, the resulting
    array is clipped to the specified target shape.

    Parameters
    ----------
    a : np.ndarray
        Input array to be upsampled.
    out_shape : tuple
        Target shape of the upsampled array.

    Returns
    -------
    np.ndarray
        Upsampled array of shape out_shape

    Examples
    --------
    >>> upsampled_array = upsample_nn(input_array, target_shape)
    >>> print(upsampled_array.shape)
    (target_shape[0], target_shape[1])

    """

    ny_out, nx_out = out_shape
    ny_in,  nx_in  = a.shape
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
    """Build the internal render grid from physical extent and chain resolution.

    Parameters
    ----------
    extent : tuple[float, float, float, float]
        Physical render extent as ``(xmin, xmax, ymin, ymax)`` in nm.
    chain_resolution_nm : float
        Internal chain/render resolution in nm per pixel.
    min_size : int
        Minimum number of pixels in each render-grid dimension.

    Returns
    -------
    tuple[int, int, float]
        Render-grid width, render-grid height, and effective nm per pixel.

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
    order: int =1,
) -> np.ndarray:
    """Resize 2D array a to out_shape using scipy.ndimage.zoom.

    This function utilizes `scipy.ndimage.zoom` to scale the input array. It
    includes safety checks for edge-case padding and cropping to ensure the
    output array matches the requested dimensions exactly.

    Parameters
    ----------
    a : np.ndarray
        Input array to be resized.
    out_shape : tuple
        Target shape of the resized
    order : int, optional
        Order of the interpolation. Defaults to 1.

    Returns
    -------
    np.ndarray
        Resized array of shape out_shape

    Examples
    --------
    >>> resized_array = resize_image_to_shape(input_array, target_shape)
    >>> print(resized_array.shape)
    (target_shape[0], target_shape[1])

    """

    a = np.asarray(a)
    out_shape = tuple(int(v) for v in out_shape)
    if a.shape == out_shape:
        return a.copy()
    zoom_factors = (out_shape[0] / a.shape[0], out_shape[1] / a.shape[1])
    out = zoom(a, zoom_factors, order=order, mode="nearest",
               prefilter=(order > 1))
    out = out[:out_shape[0], :out_shape[1]]
    if out.shape != out_shape:
        pad_y = max(0, out_shape[0] - out.shape[0])
        pad_x = max(0, out_shape[1] - out.shape[1])
        out = np.pad(out, ((0, pad_y), (0, pad_x)), mode="edge")
        out = out[:out_shape[0], :out_shape[1]]
    return out


def dna_shape(
    radius_nm: float,
    chain_resolution_nm: float,
    max_radius_px: int = 96,
    height_nm: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a spherical-cap structuring element for DNA rendering.

    The lateral DNA radius is set by ``radius_nm``. The peak height is set by
    ``height_nm`` and should usually be the physical DNA diameter.

    """
    radius_nm = max(float(radius_nm), 1e-12)
    chain_resolution_nm = max(float(chain_resolution_nm), 1e-12)
    if height_nm is None:
        height_nm = 2.0 * radius_nm
    height_nm = max(float(height_nm), 1e-12)
    r_px = radius_nm / chain_resolution_nm
    radius_px = int(np.clip(np.ceil(r_px), 1, max_radius_px))
    y, x = np.ogrid[
        -radius_px:radius_px + 1,
        -radius_px:radius_px + 1,
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
    cap = np.sqrt(np.maximum(radius_nm**2 - d_nm[inside]**2, 0.0))
    cap /= radius_nm
    structure[inside] = (height_nm * cap).astype(np.float32)
    return inside.astype(bool), structure


def AFM_tip(
    tip_radius_nm: float,
    chain_resolution_nm: float,
    max_radius_px: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a spherical-cap structuring element for AFM tip convolution.

    Tip convolution is always applied by ``create_z_based_afm``. A zero or very
    small tip radius produces a no-op one-pixel footprint.

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
        -radius_px:radius_px + 1,
        -radius_px:radius_px + 1,
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
        np.sqrt(np.maximum(tip_radius_nm**2 - d_nm[inside]**2, 0.0))
        - tip_radius_nm
    ).astype(np.float32)
    return inside.astype(bool), structure


def calculate_crossing_taper_weight(
    idx_off: int,
    w: int,
    profile: str = "gaussian",
    sigma: float | None = None,
) -> float:
    """Smooth weight in [0, 1] used to fade the crossing height boost.

    This function generates a weighting factor between 0 and 1 used to
    gradually decrease the height offset applied to DNA segments near a
    detected crossing. It ensures that the transition between the boosted
    crossing region and the standard polymer chain is smooth.
    There are 3 types of profiles possible to be applied
    for smoothing a crossing, guassian, hemisphere, and linear.

    Parameters
    ----------
    idx_off : int
        The distance in bead indices from the center of the crossing.
    w : int
        The number of beads on either side of the center over which the weight
        should taper to zero.
    profile : str, optional
        The mathematical shape of the taper. Options are:
        - 'gaussian': A smooth bell-shaped decay (recommended).
        - 'hemisphere': A circular arc taper.
        - 'linear': A simple linear ramp.
        Defaults to 'gaussian'.
    sigma : float, optional
        The standard deviation for the Gaussian profile. If None, it is
        automatically determined based on the window width.

    Returns
    -------
    float
        A weighting factor in the range [0.0, 1.0].

    Examples
    --------
    >>> wt = calculate_crossing_taper_weight(idx_off, w)
    >>> print(wt.dtype)
    float32

    """

    if w <= 0:
        return 1.0
    a = abs(idx_off)
    if profile == "linear":
        return 1.0 - (a / (w + 1.0))
    if profile == "hemisphere":
        x = a / (w + 1.0)
        return float(np.sqrt(max(0.0, 1.0 - x*x)))
    # gaussian (default)
    if sigma is None:
        sigma = max(1e-6, 0.5 * (w + 0.5))
    g     = float(np.exp(-(idx_off**2) / (2.0 * sigma**2)))
    g_end = float(np.exp(-(w**2)       / (2.0 * sigma**2)))
    if g_end < 0.999:
        g = float(np.clip((g - g_end) / (1.0 - g_end), 0.0, 1.0))
    return g


def collect_intersections(
    xy_nm: np.ndarray,
    min_separation_beads: int = 12,
    merge_radius_nm: float = 0.5,
)-> list[dict]:
    """Identify all self-intersections in a closed ring polyline.

    This function performs a brute-force search across all segment pairs of
    the polyline to find crossings. It ignores adjacent segments, shared
    vertices, and segments that are close to each other along the contour of
    the ring. Detected intersections are subsequently clustered to handle
    multi-segment vertex hits.

    Parameters
    ----------

    xy_nm : np.ndarray
        Array of shape (n, 2)
    min_separation_beads : int, optional
        Minimum separation between segments in beads. Defaults to 12.
    merge_radius_nm : float, optional
        Maximum separation between segments in nanometers. Defaults to 0.5.

    Returns
    -------
    list of dicts:
      {pt_nm, seg_i, seg_j, t, u, n_hits}
    list[dict]
        A list of dictionaries, where each dictionary represents a unique
        crossing and contains:
        - "pt_nm": The [x, y] coordinates of the intersection.
        - "seg_i": The index of the first segment involved.
        - "seg_j": The index of the second segment involved.
        - "t": Interpolation parameter along segment A.
        - "u": Interpolation parameter along segment B.
        - "n_hits": The number of raw intersections merged into this cluster.

    Examples
    --------
    >>> clusters = collect_intersections(xy_nm,
                      min_separation_beads=int(min_separation_beads),
                      merge_radius_nm=float(merge_radius_nm)
    >>> print(clusters.type)
    list[dicts]

    """

    xy = np.asarray(xy_nm, dtype=float)
    n  = int(xy.shape[0])
    raw = []
    for i in range(n):
        i2   = (i + 1) % n
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
                raw.append(dict(pt=np.array(pt, float),
                                seg_i=int(i), seg_j=int(j),
                                t=float(t), u=float(u)))

    clusters = merge_crossing_clusters(raw, merge_radius_nm=merge_radius_nm)
    return clusters


def merge_crossing_clusters(
    raw_list: list[dict],
    merge_radius_nm: float = 0.5,
) -> list[dict]:
    """Group nearby raw intersection points into single averaged clusters.

    This function uses a greedy proximity-based algorithm to cluster
    intersections that occur close to each other (e.g., where multiple
    segments meet at a single vertex). It calculates the centroid for each
    cluster and retains metadata from the first intersection encountered in
    each group.

    Parameters
    ----------
    raw_list : list[dict]
      A list of dictionaries containing raw intersection data. Each
      dictionary must include the key "point" (a 2D numpy array).
    merge_radius_nm : float, optional
      The maximum Euclidean distance allowed between an intersection point
      and a cluster's current centroid for it to be merged. Defaults to 0.5.

    Returns
    -------
    list[dict]

    Examples
    --------
    >>> clusters = merge_crossing_clusters(raw_list,
                                        merge_radius_nm=float(merge_radius_nm))
    >>> print(clusters.type)
    list[dict]

    """

    clusters = []
    for r in raw_list:
        pt = r["pt"].astype(float)
        placed = False
        for c in clusters:
            if np.linalg.norm(pt -
                              c["pt_sum"] / c["n"]) <= float(merge_radius_nm):
                c["pt_sum"] += pt
                c["n"]      += 1
                c["items"].append(r)
                placed = True
                break
        if not placed:
            clusters.append({"pt_sum": pt.copy(), "n": 1, "items": [r]})

    out = []
    for c in clusters:
        pt  = (c["pt_sum"] / c["n"]).astype(np.float32)
        rep = c["items"][0]
        out.append(dict(
            pt_nm  = pt,
            seg_i  = int(rep["seg_i"]),
            seg_j  = int(rep["seg_j"]),
            t      = float(rep["t"]),
            u      = float(rep["u"]),
            n_hits = int(c["n"]),
        ))
    return out


def boost_crossings_intersections(
    x_nm: np.ndarray,
    y_nm: np.ndarray,
    zc_nm: np.ndarray,
    crossings: list[dict] | None = None,
    min_separation_beads: int = 12,
    boost_window_beads: int = 2,
    guaranteed_offset_nm: float = 1.0,
    boost_method: str = "additive",
    boost_profile: str = "gaussian",
    boost_sigma_beads: float | None = None,
    far_clip_nm: float | None =2.5,
    far_clip_window_beads: int = 12,
    merge_radius_nm: float = 0.5,
) -> tuple[np.ndarray, list[dict]]:
    """Boost z-height at strand crossings so they are visually distinct.

    This function identifies self-intersections in a 2D projection and
    physically separates the strands in 3D. For each crossing, it determines
    which segment is 'on top' and raises its vertical profile using a tapered
    window to ensure a realistic and visually distinct crossover point.
    For each crossing it:
    - Determines which strand is on top (higher z)
    - Raises top-strand beads in a tapered window to at least
      (bottom_max + guaranteed_offset_nm)
    - Optionally clips non-crossing beads to far_clip_nm

    Parameters
    ----------
    x_nm : np.ndarray
      The x-coordinates of the polymer chain.
    y_nm : np.ndarray
      The y-coordinates of the polymer chain.
    zc_nm : np.ndarray
      The initial z-coordinates of the polymer chain.
    crossings : list[dict]
      Pre-computed crossing data. If None, crossings are detected
        automatically using `collect_polyline_intersections`.
    min_separation_beads : int, optional
      Minimum separation between segments in beads. Defaults to 12.
    boost_window_beads : int, optional
      The number of beads to boost. Defaults to 2.
    guaranteed_offset_nm : float, optional
      The minimum boost height in nanometers. Defaults to 1.0.
    boost_method : str
      The logic for height calculation
    boost_profile : str
      The shape of the tapering profile
    boost_sigma_beads : float
      The standard deviation for gaussian tapering
    far_clip_nm : float
      If applied caps the height of beads far from any crossing to this value.
    far_clip_window_beads : int
      The number of beads to clip.
    merge_radius_nm : float
      The maximum Euclidean distance for nearby raw intersections.

    Returns
    -------
    z : np.ndarray
        The updated z-coordinates with crossings boosted.
    crossing_info : list[dict]
        Information regarding the identified and modified crossings.

    Examples
    --------
    >>> z, crossing_info = boost_crossings_intersections(x, y, z)
    >>> print(z.shape, type(crossing_info))
    (N,3) <class 'list'>

    """

    x = np.asarray(x_nm, dtype=np.float32)
    y = np.asarray(y_nm, dtype=np.float32)
    z = np.asarray(zc_nm, dtype=np.float32).copy()
    n = int(z.shape[0])
    if n < 5:
        return z.astype(np.float32), []

    if crossings is None:
        xy       = np.stack([x, y], axis=1)
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

        return np.array([(center + k) % n for k in range(-w, w+1)],
                        dtype=np.int32)

    crossing_info    = []
    crossing_centers = []
    w = int(boost_window_beads)

    for c in crossings:
        i  = int(c["seg_i"]); j  = int(c["seg_j"])
        i2 = (i + 1) % n;     j2 = (j + 1) % n
        ti = float(c.get("t", 0.5));  uj = float(c.get("u", 0.5))

        bead_i = i  if ti < 0.5 else i2
        bead_j = j  if uj < 0.5 else j2
        zi     = float((1.0 - ti) * z[i]  + ti * z[i2])
        zj     = float((1.0 - uj) * z[j]  + uj * z[j2])

        top_center, bottom_center = (bead_i, bead_j) if zi >= zj else (bead_j,
                                                                       bead_i)
        top_seg    = get_segment_indices(top_center,    w)
        bottom_seg = get_segment_indices(bottom_center, w)

        bl_max = float(z[bottom_seg].max())
        tl_max    = float(z[top_seg].max())

        if boost_method == "absolute":
            target_height = bl_max + float(guaranteed_offset_nm)
        else:
            target_height = max(tl_max, bl_max) + float(guaranteed_offset_nm)

        for off in range(-w, w+1):
            k   = (top_center + off) % n
            wt  = float(calculate_crossing_taper_weight(off, w,
                                      profile=boost_profile,
                                      sigma=boost_sigma_beads))
            z[k] = float((1.0 - wt) * z[k] + wt * max(z[k], target_height))

        crossing_centers.extend([int(top_center), int(bottom_center)])
        crossing_info.append(dict(
            pt_nm        = np.asarray(c["pt_nm"], dtype=np.float32),
            seg_i        = int(i),
            seg_j        = int(j),
            top_center   = int(top_center),
            bottom_center= int(bottom_center),
        ))

    # Clip beads far from any crossing
    if crossing_centers and far_clip_nm is not None:
        centers = np.array(crossing_centers, dtype=np.int32)
        r       = int(far_clip_window_beads)
        for k in range(n):
            d = int(np.min(np.minimum(np.abs(k - centers),
                                      n - np.abs(k - centers))))
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
    boost_method: str = "absolute",
    boost_profile: str = "gaussian",
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
    """Render an AFM-like height image from 3D DNA coordinates.

    This function projects 3D bead coordinates onto a 2D grid, manages
    crossings, rasterizes the polymer backbone, applies morphological
    dilations for DNA shape and AFM tip convolution, and adds realistic optical
    and physical artifacts (taper, ridge, grain).

    The lateral width (``dna_diameter_nm``) is an effective image-rendering
    width and the height (``dna_height_nm``) is set independently; neither
    changes the MD bead size or forces.

    Parameters
    ----------
    coords : np.ndarray
      The 3D coordinates of the polymer chain.
    nx : int, optional
      The number of pixels in the x-direction. Defaults to 800.
    ny : int, optional
      The number of pixels in the y-direction. Defaults to 800.
    dna_diameter_nm : float, optional
      Effective lateral DNA width in nanometers (before tip convolution).
      Defaults to 2.0.
    dna_height_nm : float, optional
      Height of the DNA cap in nanometers. Defaults to ``dna_diameter_nm``.
    taper_width_nm : float, optional
      Width of the outer radial band that the edge taper attenuates. ``None``
      reproduces the original Gaussian attenuation measured from the centre
      line (used by the non-MD pathway). Must be at most half the width.
    tip_radius_nm : float, optional
      The radius of the tip in nanometers. Defaults to 1.0.
    max_height_nm : float, optional
      The maximum height of the output image in nanometers. Defaults to 6.0.
    chain_resolution_nm : float, optional
      The resolution of the polymer chain in nanometers. Defaults to 0.10.
    max_radius_px : int
        Maximum radius in pixels.
    radius_shrink_px : float
        Radius shrinking factor.
    final_blur_sigma_px : float
        Final blur sigma in pixels.
    apply_edge_taper : bool
        Apply edge taper.
    taper_sigma_nm : float
        Taper sigma in nanometers.
    taper_floor : float
        Taper floor.
    add_center_ridge : bool
        Add center ridge.
    ridge_sigma_nm : float
        Ridge sigma in nanometers.
    ridge_amp_nm : float
        Ridge amplitude in nanometers.
    grain_nm : float
        Grain size in nanometers.
    grain_sigma_px : float
        Grain sigma in pixels.
    grain_seed : int
        Grain seed.
    enable_crossing_boost : bool
        Enable crossing boost.
    min_separation_beads : int
        Minimum separation between segments in beads.
    boost_window_beads : int
        The number of beads to boost.
    guaranteed_offset_nm : float
        The minimum boost height in nanometers.
    boost_method : str
        The logic for height calculation
    boost_profile : str
        The shape of the tapering profile
    boost_sigma_beads : float
        The standard deviation for gaussian tapering
    crossings_precomputed : list[dict]
        Pre-computed crossing data.
    far_clip_nm : float
        If applied caps the height of beads
        far from any crossing to this value.
    far_clip_window_beads : int
        The number of beads to clip.
    return_crossing_info : bool
        Return crossing info.
    return_masks : bool
        Return masks.
    extent : tuple[float, float, float, float]
        Extent of the output image.

    Returns
    -------
    afm_image : np.ndarray
        The simulated 2D AFM height map.
    debug_dict : dict or tuple
        A dictionary containing masks and metadata (if return_masks=True),
        or a tuple containing the physical (xmin, xmax, ymin, ymax) extent.

    Examples
    --------
    >>> afm_image, debug_dict = create_z_based_afm(coords, nx=800, ny=800,
                                                  return_masks=True)
    >>> print(afm_image.shape, type(debug_dict))
    (800, 800) <class 'dict'>
    """

    dna_height_nm = float(dna_diameter_nm if dna_height_nm is None else dna_height_nm)
    if not np.isfinite(dna_height_nm) or dna_height_nm <= 0:
        raise ValueError("dna_height_nm must be finite and positive")
    if not np.isfinite(dna_diameter_nm) or dna_diameter_nm <= 0:
        raise ValueError("dna_diameter_nm must be finite and positive")
    if taper_width_nm is not None and not 0 <= taper_width_nm <= dna_diameter_nm / 2:
        raise ValueError("taper_width_nm must be between zero and the tube radius")
    if not 0 <= taper_floor <= 1 or taper_sigma_nm <= 0:
        raise ValueError("Invalid taper floor/sigma")
    (x, y, z) = coords.T
    nx_out = int(nx)
    ny_out = int(ny)
    chain_resolution_nm = max(float(chain_resolution_nm), 1e-06)
    if extent is None:
        output_xmin = float(x.min() - 2.0)
        output_xmax = float(x.max() + 2.0)
        output_ymin = float(y.min() - 2.0)
        output_ymax = float(y.max() + 2.0)
    else:
        (output_xmin, output_xmax, output_ymin, output_ymax) = extent
    output_extent = (float(output_xmin), float(output_xmax), float(output_ymin), float(output_ymax))
    output_px_eff_x = (output_xmax - output_xmin) / max(nx_out - 1, 1)
    output_px_eff_y = (output_ymax - output_ymin) / max(ny_out - 1, 1)
    output_p_eff = 0.5 * (output_px_eff_x + output_px_eff_y)
    render_pad_nm = max(2.0 * float(dna_diameter_nm), 2.0 * float(tip_radius_nm), 4.0 * chain_resolution_nm)
    render_extent = (float(output_xmin - render_pad_nm), float(output_xmax + render_pad_nm), float(output_ymin - render_pad_nm), float(output_ymax + render_pad_nm))
    (nx_eff, ny_eff, p_eff) = compute_internal_render_grid(render_extent, chain_resolution_nm, min_size=16)
    (xmin, xmax, ymin, ymax) = render_extent
    ix = np.clip(((x - xmin) / (xmax - xmin) * (nx_eff - 1)).astype(np.int32), 0, nx_eff - 1)
    iy = np.clip(((y - ymin) / (ymax - ymin) * (ny_eff - 1)).astype(np.int32), 0, ny_eff - 1)
    zc = (z - z.min()).astype(np.float32)
    crossing_info = []
    if enable_crossing_boost:
        (zc, crossing_info) = boost_crossings_intersections(x.astype(np.float32), y.astype(np.float32), zc, crossings=crossings_precomputed, min_separation_beads=int(min_separation_beads), boost_window_beads=int(boost_window_beads), guaranteed_offset_nm=float(guaranteed_offset_nm), boost_method=str(boost_method), boost_profile=str(boost_profile), boost_sigma_beads=boost_sigma_beads, far_clip_nm=far_clip_nm, far_clip_window_beads=int(far_clip_window_beads))
    z_line = np.zeros((ny_eff, nx_eff), dtype=np.float32)
    line_mask = np.zeros((ny_eff, nx_eff), dtype=bool)
    npts = len(ix)
    for k in range(npts):
        k2 = (k + 1) % npts
        (x0, y0, z0) = (int(ix[k]), int(iy[k]), float(zc[k]))
        (x1, y1, z1) = (int(ix[k2]), int(iy[k2]), float(zc[k2]))
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
            if 'top_center' not in crossing:
                continue
            if 'bottom_center' not in crossing:
                continue
            top_center = int(crossing['top_center'])
            bottom_center = int(crossing['bottom_center'])
            for off in range(-crossing_window, crossing_window + 1):
                for (center, mask) in ((top_center, top_cross_line_mask), (bottom_center, bottom_cross_line_mask)):
                    k0 = (center + off) % npts
                    k1 = (k0 + 1) % npts
                    (x0, y0) = (int(ix[k0]), int(iy[k0]))
                    (x1, y1) = (int(ix[k1]), int(iy[k1]))
                    n_seg = int(max(abs(x1 - x0), abs(y1 - y0)) + 1)
                    if n_seg <= 1:
                        if 0 <= x0 < nx_eff and 0 <= y0 < ny_eff:
                            mask[y0, x0] = True
                        continue
                    xs = np.linspace(x0, x1, n_seg).astype(np.int32)
                    ys = np.linspace(y0, y1, n_seg).astype(np.int32)
                    valid = (xs >= 0) & (xs < nx_eff) & (ys >= 0) & (ys < ny_eff)
                    xs = xs[valid]
                    ys = ys[valid]
                    if xs.size:
                        mask[ys, xs] = True
    r_eff = max(0.5 * float(dna_diameter_nm), 0.2)
    (fp_dna, cap) = dna_shape(radius_nm=r_eff, chain_resolution_nm=p_eff, max_radius_px=max_radius_px, height_nm=dna_height_nm)
    surface = grey_dilation(z_line, footprint=fp_dna, structure=cap).astype(np.float32)
    dna_body_region = grey_dilation(line_mask.astype(np.uint8), footprint=fp_dna) > 0
    surface[~dna_body_region] = 0.0
    top_cross_region = grey_dilation(top_cross_line_mask.astype(np.uint8), footprint=fp_dna) > 0
    bottom_cross_region = grey_dilation(bottom_cross_line_mask.astype(np.uint8), footprint=fp_dna) > 0
    if apply_edge_taper and np.any(line_mask):
        taper_dist_px = distance_transform_edt(~line_mask).astype(np.float32)
        taper_dist_nm = taper_dist_px * np.float32(p_eff)
        sig = max(float(taper_sigma_nm), 1e-06)
        # width=None reproduces the old Gaussian attenuation from the centerline.
        # Otherwise taper only the outer radial band of width taper_width_nm.
        band = r_eff if taper_width_nm is None else float(taper_width_nm)
        taper_start_nm = r_eff - band
        edge_distance_nm = np.maximum(taper_dist_nm - taper_start_nm, 0.0)
        factor = taper_floor + (1.0 - taper_floor) * np.exp(-edge_distance_nm ** 2 / (2.0 * sig ** 2))
        intersection_zone = bottom_cross_region & top_cross_region
        blur_sigma_px = max(1.0, 0.35 * float(dna_diameter_nm) / p_eff)
        weight = gaussian_filter(intersection_zone.astype(np.float32), sigma=blur_sigma_px)
        smooth_factor = factor * (1.0 - weight) + 1.0 * weight
        surface[dna_body_region] *= smooth_factor[dna_body_region]
    if add_center_ridge and ridge_amp_nm > 0 and np.any(line_mask):
        ridge_dist_px = distance_transform_edt(~line_mask).astype(np.float32)
        ridge_dist_nm = ridge_dist_px * np.float32(p_eff)
        rs = max(float(ridge_sigma_nm), 1e-06)
        ridge = np.exp(-ridge_dist_nm ** 2 / (2.0 * rs ** 2))
        surface[dna_body_region] += ridge_amp_nm * ridge[dna_body_region]
    np.clip(surface, 0, max_height_nm, out=surface)
    if final_blur_sigma_px > 0:
        surface = gaussian_filter(surface, sigma=float(final_blur_sigma_px)).astype(np.float32)
    (fp_tip, tip_struct) = AFM_tip(tip_radius_nm=float(tip_radius_nm), chain_resolution_nm=p_eff, max_radius_px=max_radius_px)
    surface = grey_dilation(surface, footprint=fp_tip, structure=tip_struct).astype(np.float32)
    apparent_region = grey_dilation(dna_body_region.astype(np.uint8), footprint=fp_tip) > 0
    surface[~apparent_region] = 0.0
    np.clip(surface, 0, max_height_nm, out=surface)
    if grain_nm and grain_nm > 0:
        rng = np.random.default_rng(int(grain_seed))
        noise = rng.normal(0.0, 1.0, size=surface.shape).astype(np.float32)
        if grain_sigma_px and grain_sigma_px > 0:
            noise = gaussian_filter(noise, sigma=float(grain_sigma_px)).astype(np.float32)
        noise_std = noise[apparent_region].std()
        std = float(noise_std) if np.any(apparent_region) else float(noise.std())
        if std > 1e-09:
            noise /= np.float32(std)
        surface[apparent_region] *= 1.0 + np.float32(grain_nm) * noise[apparent_region]
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
    surface = resize_image_to_shape(surface[crop_slice], out_shape=out_shape, order=1).astype(np.float32)
    line_mask = resize_image_to_shape(line_mask[crop_slice].astype(np.float32), out_shape=out_shape, order=0) > 0.5
    top_cross_line_mask = resize_image_to_shape(top_cross_line_mask[crop_slice].astype(np.float32), out_shape=out_shape, order=0) > 0.5
    bottom_cross_line_mask = resize_image_to_shape(bottom_cross_line_mask[crop_slice].astype(np.float32), out_shape=out_shape, order=0) > 0.5
    top_cross_region = resize_image_to_shape(top_cross_region[crop_slice].astype(np.float32), out_shape=out_shape, order=1) > 0.25
    apparent_region = resize_image_to_shape(apparent_region[crop_slice].astype(np.float32), out_shape=out_shape, order=1) > 0.25
    mask_radius_px = max(1, int(np.ceil(0.5 * float(dna_diameter_nm) / max(output_p_eff, 1e-12))))
    (yy, xx) = np.ogrid[-mask_radius_px:mask_radius_px + 1, -mask_radius_px:mask_radius_px + 1]
    dna_mask_fp = xx * xx + yy * yy <= mask_radius_px * mask_radius_px
    dna_body_region = make_dna_mask_from_beads(coords, output_extent, nx_out, ny_out, dna_diameter_nm=dna_diameter_nm) > 0
    top_cross_region_out = grey_dilation(top_cross_line_mask.astype(np.uint8), footprint=dna_mask_fp) > 0
    bottom_cross_region_out = grey_dilation(bottom_cross_line_mask.astype(np.uint8), footprint=dna_mask_fp) > 0
    crossing_region = top_cross_region_out | bottom_cross_region_out
    if return_masks:
        debug = {'line_mask': line_mask, 'top_cross_line_mask': top_cross_line_mask, 'bottom_cross_line_mask': bottom_cross_line_mask, 'dna_region': dna_body_region, 'dna_body_region': dna_body_region, 'apparent_region': apparent_region, 'top_cross_region': top_cross_region_out, 'bottom_cross_region': bottom_cross_region_out, 'crossing_region': crossing_region, 'extent': output_extent, 'output_extent': output_extent, 'render_extent': render_extent, 'p_nm_per_px': float(output_p_eff), 'output_nm_per_px': float(output_p_eff), 'chain_resolution_nm': float(p_eff), 'requested_chain_resolution_nm': float(chain_resolution_nm), 'tip_radius_nm': float(tip_radius_nm), 'dna_diameter_nm': float(dna_diameter_nm), 'dna_height_nm': dna_height_nm, 'taper_width_nm': taper_width_nm, 'taper_sigma_nm': float(taper_sigma_nm), 'mask_width_multiplier': 1.0}
        if return_crossing_info:
            debug['crossings'] = crossing_info
        return (surface, debug)
    return (surface, output_extent)


# ─────────────────────────────────────────────────────────────────────────────
# Noise
# ─────────────────────────────────────────────────────────────────────────────


def robust_range(a:np.ndarray | list) -> tuple[float, float, float]:
    """Calculate the 1st and 99th percentiles of finite values in an array.

    This function provides a robust measure of the data range by ignoring
    extreme outliers (the top and bottom 1%) as well as non-finite values
    (NaNs and infinities).

    Parameters
    ----------
    a : np.ndarray | list
        The input data array to be evaluated.

    Returns
    -------
    tuple[float, float, float]
        A tuple containing:
        - percentile_1: The 1st percentile of the finite data.
        - percentile_99: The 99th percentile of the finite data.
        - robust_spread: The difference between the 99th and 1st percentiles.

    Examples
    --------
    >>> a,b,c = robust_range(img)
    >>> print(a,b,c)
    262 1023 761

    """

    a = np.asarray(a, dtype=np.float32)
    p1, p99 = np.percentile(a[np.isfinite(a)], [1, 99])
    return float(p1), float(p99), float(p99 - p1)


def looks_like_afm_height(img: np.ndarray) -> bool:
    """Determine if an image array represents a valid AFM height map.

    This function checks if the dynamic range of the image (excluding
    extreme outliers) is finite, strictly positive, and within a physically
    sane upper bound.

    Parameters
    ----------
    img : np.ndarray
        The 2D array representing the simulated or real AFM height map.

    Returns
    -------
    bool
        True if the image has a valid, finite, and positive dynamic range;
        False otherwise (e.g., if the array is flat, infinite, or all NaNs).

    Examples
    --------
    >>> Flag = looks_like_afm_height(img)
    >>> print(Flag)
    True

    """

    p1, p99, rng = robust_range(img)
    return np.isfinite(rng) and 0 < rng <= 1e9


def maybe_to_nm(img: np.ndarray) -> tuple[np.ndarray, str]:
    """Auto-convert from metres to nm if values look metre-scale.

    This function is used to convert the height map from metres to nm if the
    measured height is in metres.

    Parameters
    ----------
    img : np.ndarray
        The 2D array representing the AFM height map.

    Returns
    -------
    A tuple consisting of:
      - np.ndarray
          The image after the converion of the height map if needed.
      - str
          either as is or meters to nanometers depending on passed values.

    Examples
    --------
    >>> img = maybe_to_nm(img)
    >>> print(img.shape)
    (800, 800)

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
    """Read a .spm file and parse binary data-block metadata.

    This function extracts the ASCII header from a Scanning Probe Microscopy
    (SPM) file, locates the metadata blocks for the image data, and parses the
    structural parameters required to read the raw binary data.

    Parameters
    ----------
    filename: str | Path
      The path to the .spm file to be parsed.
    max_blocks: int
      The maximum number of blocks to parse.

    Returns
    -------
    tuple[bytes, str, list[dict]]
      A tuple containing:
      - raw: The raw binary data from the .spm file.
      - header: The ASCII header from the .spm file.
      - uniq: A list of dictionaries containing the parsed block metadata.

    Raises
    ------
    RuntimeError
      If no 'Data offset' strings are found in the header.

    Examples
    --------
    >>> raw, header, blocks = parse_blocks(filename)
    >>> print(type(raw), type(header), type(blocks))
    <class 'bytes'> <class 'str'> <class 'list'>

    """

    raw    = open(filename, "rb").read()
    i_end  = raw.find(b"\x1a")
    if i_end == -1:
        i_end = min(len(raw), 400_000)
    header = raw[:i_end].decode("latin1", errors="ignore")

    matches = [m.start() for m in re.finditer(r"Data offset", header)]
    if not matches:
        raise RuntimeError("No 'Data offset' found in header.")

    blocks = []
    for k, pos in enumerate(matches[:max_blocks]):
        win = header[max(0, pos-2500): min(len(header), pos+2500)]

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

        name_candidate = re.findall(r"@[^:\n\r]{0,40}:\s*([^\n\r]{1,80})", win)
        name = name_candidate[-1].strip() if name_candidate else None

        off    = grab_int("Data offset")
        length = grab_int("Data length")
        bpp    = grab_int("Bytes/pixel")
        nx_b   = grab_int("Samps/line")
        ny_b   = grab_int("Number of lines")
        zscale = grab_float("Z scale")
        zoff   = grab_float("Z offset")

        if None in (off, length, bpp, nx_b, ny_b):
            continue
        blocks.append(dict(name=name or f"block_{k}",
                           off=off, length=length, bpp=bpp,
                           nx=nx_b, ny=ny_b,
                           zscale=zscale, zoff=zoff))

    uniq, seen = [], set()
    for b in blocks:
        if b["off"] not in seen:
            uniq.append(b); seen.add(b["off"])
    if not uniq:
        raise RuntimeError("Found 'Data offset' strings"
                           " but no parseable blocks.")
    return raw, header, uniq


def read_block_candidates(
    raw: bytes,
    b: dict,
) -> list[tuple[str, np.ndarray]]:
    """Decode a binary data block using all plausible NumPy data types.

    This function attempts to parse the binary blob using all logical dtypes
    for the given bytes-per-pixel and applies physical Z-scaling to the
    results.

    Parameters
    ----------
    raw : bytes
      The raw binary data from the .spm file.
    b : dict
      The parsed block metadata.

    Returns
    -------
    list[tuple[str, np.ndarray]]
      A list of candidate tuples. Each tuple contains:
      - dtype_str: The string representation of the data type.
      - image_array: The decoded image array.

    Raises
    ------
    Runtime error
    - Truncated block.

    Examples
    --------
    >>> candidates = read_block_candidates(raw, b)
    >>> print(type(candidates))
    <class 'list'>

    """

    off, length, bpp = b["off"], b["length"], b["bpp"]
    nx_b, ny_b       = b["nx"], b["ny"]
    blob = raw[off:off+length]
    if len(blob) < length:
        raise RuntimeError("Truncated block.")

    n = nx_b * ny_b
    cands = []
    dtype_map = {2: ["<i2", ">i2", "<u2", ">u2"],
                 4: ["<i4", ">i4", "<f4", ">f4"],
                 1: ["u1"]}
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
)-> tuple[dict, np.ndarray]:
    """Evaluate all data blocks to find the most plausible AFM height map.

    This function iterate all blocks and dtype candidates and then picks the
    one that:
    - passes looks_like_afm_height
    - has the highest log-variance score
    (which is done to favor candidates with more topographical detail)

    Parameters
    ----------
    raw : bytes
      The raw binary data from the .spm file.
    blocks : list[dict]
      The parsed block metadata.
    verbose : bool
      Whether to print diagnostics.

    Returns
    -------
    tuple[dict, np.ndarray]

    Raises
    ------
    RuntimeError
      If no suitable block is found.

    Examples
    --------
    >>> b, img = choose_best_height_image(raw, blocks)
    >>> print(type(b), type(img))
    <class 'dict'> <class 'numpy.ndarray'>

    """

    best, best_score = None, -np.inf
    for b in blocks:
        try:
            for dt, img in read_block_candidates(raw, b):
                if not looks_like_afm_height(img):
                    continue
                p1, p99, rng = robust_range(img)
                var   = float(np.var(img))
                score = np.log10(var + 1e-9) - 0.001 * max(rng - 1e6, 0)
                if score > best_score:
                    best_score = score
                    best = (b, dt, img, (p1, p99, rng, var))
        except Exception:
            continue

    if best is None:
        raise RuntimeError("Could not find a"
                           "plausible AFM height image decode.")
    b, dt, img, stats = best
    if verbose:
        p1, p99, rng, var = stats
        print(f"Chosen block: {b['name']!r} {b['ny']}x{b['nx']} "
              f"bpp={b['bpp']} decode={dt}")
        print(f"Robust p1={p1:.3g}, p99={p99:.3g}, "
              f"range={rng:.3g}, var={var:.3g}")
    return b, img


def plane_remove(
    z: np.ndarray,
    mask: np.ndarray | None = None,
)-> np.ndarray:
    """Fit and subtract a first-order tilted plane from a 2D height map.

    This function performs background leveling by fitting a plane defined by
    the equation z = ax + by + c to the image. If a mask is provided, the fit
    is calculated only using the pixels where the mask is True (typically
    representing the flat substrate).

    Parameters
    ----------
    z : np.ndarray
      The 2D height map.
    mask : np.ndarray | None, optional
      A boolean mask of the same shape as z. If provided, the fit is
      calculated only using the pixels where the mask is True.
      This is useful for excluding features like DNA from the plane fit.
      Defaults to None.

    Returns
    -------
    np.ndarray
      The flattened height map.

    Examples
    --------
    >>> z_flat = plane_remove(z)
    >>> print(z_flat.shape)
    (800, 800)

    """

    ny, nx = z.shape
    Y, X   = np.mgrid[0:ny, 0:nx]
    A = np.c_[X.ravel(), Y.ravel(), np.ones(X.size)]
    b = z.ravel()
    if mask is not None:
        m = mask.ravel().astype(bool)
        A, b = A[m], b[m]
    C, *_ = np.linalg.lstsq(A, b, rcond=None)
    plane  = (C[0]*X + C[1]*Y + C[2]).astype(np.float32)
    return (z - plane).astype(np.float32)


def line_flatten_median(
    z: np.ndarray,
    mask: np.ndarray | None = None,
)-> np.ndarray:
    """Subtract per-row median (background pixels only if mask given).

    AFM images often exhibit horizontal stripes due to low-frequency noise or
    z-axis drift during scanning. This function levels each scan line
    independently by shifting its median value to zero.

    Parameters
    ----------
    z: np.ndarray
      The 2D height map
    mask: np.ndarray | None, optional
      A boolean mask of the same shape as z. If provided,
      the median is calculated only using the pixels where the mask is True.
      This prevents features (like DNA) from biasing the
      flattening of the background. Defaults to None.

    Returns
    -------
    np.ndarray
      The flattened height map.

    Examples
    --------
    >>> z_flat = line_flatten_median(z)
    >>> print(z_flat.shape)
    (800, 800)

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
)-> np.ndarray:
    """Detects topographical features as pixels deviating by a threshold.

    This function identifies pixels that deviate significantly from a smoothed
    local background. It uses the Median Absolute Deviation (MAD) to estimate
    the noise floor, making it robust against the features it is trying to
    detect.

    Parameters
    ----------
    z : np.ndarray
      The 2D height map.
    smooth_sigma_px : float, optional
      The smoothing width in pixels. Defaults to 3.0.
    thresh_sigma : float, optional
      The number of standard deviations above the median to consider a pixel
      as a feature. Defaults to 3.0.
    dilate_px : int, optional
      The radius of circular footprint to dilate the feature mask.
      Defaults to 4.

    Returns
    -------
    np.ndarray
      A boolean mask where true indicates a detected feature

    Examples
    --------
    >>> feat = feature_mask(z)
    >>> print(feat.shape, feat.dtype)
    (800, 800) bool

    """

    z_s   = gaussian_filter(z, sigma=float(smooth_sigma_px)).astype(np.float32)
    resid = (z - z_s).astype(np.float32)
    med   = float(np.median(resid))
    mad   = float(np.median(np.abs(resid - med)))
    sig   = 1.4826 * mad if mad > 1e-12 else float(np.std(resid) + 1e-12)
    feat  = np.abs(resid - med) > (float(thresh_sigma) * sig)
    if dilate_px and dilate_px > 0:
        r     = int(dilate_px)
        yy, xx = np.ogrid[-r:r+1, -r:r+1]
        fp    = (xx*xx + yy*yy) <= r*r
        feat  = grey_dilation(feat.astype(np.uint8), footprint=fp) > 0
    return feat


def inpaint_simple(
    z: np.ndarray,
    mask: np.ndarray,
    smooth_sigma_px: float = 8.0,
) -> np.ndarray:
    """Replace masked regions with a smooth estimate of the local background.

    This function calculates a heavily blurred version of the original image to
    serve as a background proxy and patches the masked areas with this
    estimate.

    Parameters
    ----------
    z : np.ndarray
      The 2D height map.
    mask : np.ndarray
      A boolean mask where True indicates pixels to be inpainted (features)
    smooth_sigma_px : float, optional
      The sigma value for the gaussian filter.
      Higher value resutls in smoother inpainted images. Defaults to 8.0.

    Returns
    -------
    np.ndarray
      The inpainted height

    Examples
    --------
    >>> z_bg = inpaint_simple(z, feat)
    >>> print(z_bg.shape)
    (800, 800)

    """

    bg  = gaussian_filter(z, sigma=float(smooth_sigma_px)).astype(np.float32)
    out = z.copy()
    out[mask] = bg[mask]
    return out


def bandpass_noise(
    z: np.ndarray,
    low_sigma_px: float = 1.0,
    high_sigma_px: float = 30.0,
) -> np.ndarray:
    """Performs bandpass filtering on the noise image.

    This function performs a bandpass filter by calculating the difference
    between two Gaussian-blurred versions of the same image. It effectively
    removes:
    1.  High-frequency noise: Suppressed by the `low_sigma` blur.
    2.  Low-frequency trends: (like tilt or bowing) Removed by subtracting
        the `high_sigma` blur.

    Parameters
    ----------
    z : np.ndarray
      The 2D height
    low_sigma_px : float, optional
      The sigma value for gaussian filter defining the high frequency cutoff.
      Defaults to 1.0.
    high_sigma_px : float, optional
      The sigma value for gaussian filter defining the low frequency cutoff.
      Defaults to 30.0.

    Returns
    -------
    np.ndarray
      The bandpassed height map.

    Examples
    --------
    >>> tex = bandpass_noise(z)
    >>> print(tex.shape)
    (800, 800)

    """

    low  = gaussian_filter(z, sigma=float(low_sigma_px)).astype(np.float32)
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
    """Execute the full noise-extraction pipeline on an AFM height map.

    This function isolates the pure substrate noise texture by leveling the
    image (plane flatten and line flatten), masking out topographical features
    (detect features), inpainting those regions, and applying a bandpass
    filter.
    The resulting image can be rescaled to a target RMS value.
    The resulting texture can be used to augment synthetic datasets.

    Parameters
    ----------
    z_in : np.ndarray
      The 2D height map.
    median_size : int, optional
      The size of the median filter to apply. Defaults to 3.
    bg_quantile : float, optional
      The quantile to used to generate a rough initial background guess for the
      preliminary plane and line leveling. Defaults to 40.0.
    feature_sigma_px : float, optional
      The gaussian sigma for background estimation for the feature mask.
      Defaults to 3.0.
    feature_thresh_sigma : float, optional
      The number of standard deviations above the median to consider a pixel as
      a feature. Defaults to 3.0.
    feature_dilate_px : int, optional
      The radius of curcular footprint to dilate the feature mask. Defaults to
      4.
    inpaint_sigma_px : float, optional
      The sigma value for the gaussian filter. Higher value results in smoother
      inpainted image. Defaults to 10.0.
    band_low_sigma_px : float, optional
      The sigma value for gaussian filter defining the high frequency cutoff.
      Defaults to 0.8.
    band_high_sigma_px : float, optional
      The sigma value for gaussian filter defining the low frequency cutoff.
      Defaults to 35.0.
    target_rms_nm: float | None, optional
      If provided, rescales the extracted noise textureto match this target RMS
      roughness. Defaults to None.
    apply_plane_remove : bool, optional
      Whether to apply the plane remove step. Defaults to True.
    apply_line_flatten : bool, optional
      Whether to apply the line flatten step. Defaults to True.
    apply_bandpass : bool, optional
      Whether to apply the bandpass step. Defaults to True.

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray, np.ndarray]
      A tuple containing:
      - rms_nm: The RMS roughness of the extracted noise texture.
      - noise_texture: The extracted, zero mean noise texture.
      - z_flat: The flattened height map.
      - feature_mask: A boolean mask where true indicates a detected feature.

    Examples
    --------
    >>> rms_nm, noise_tex, z_flat, feat = extract_topostats_like_noise(z)
    >>> print(rms_nm)
    0.06
    >>> print(noise_tex.shape)
    (800, 800)
    >>> print(z_flat.shape)
    (800, 800)
    >>> print(feat.shape)
    (800, 800)

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
)-> np.ndarray:
    """Tile a texture array and extract a deterministic random crop.

    This function ensures that an extracted background noise texture can be
    resized to fit any target simulated image size. If the texture is smaller
    than the target, it is tiled (repeated) to cover the area. A random crop is
    then taken to prevent obvious repeating patterns at the origin.

    Parameters
    ----------
    tex : np.ndarray
      The 2D height map
    out_shape : tuple[int, int]
      The target output shape.
    seed : int | None, optional
      The random seed. Defaults to 0.

    Returns
    -------
    np.ndarray
      The resized and cropped noise texture.

    Examples
    --------
    >>> tex = tile_or_crop(tex, (ny, nx))
    >>> print(tex.shape)
    (800, 800)

    """

    ty, tx = tex.shape
    oy, ox = out_shape
    if (ty, tx) == (oy, ox):
        return tex
    tiled = np.tile(tex, (int(np.ceil(oy / ty)), int(np.ceil(ox / tx))))
    rng   = np.random.default_rng(seed)
    y0    = int(rng.integers(0, tiled.shape[0] - oy + 1))
    x0    = int(rng.integers(0, tiled.shape[1] - ox + 1))
    return tiled[y0:y0+oy, x0:x0+ox].astype(np.float32)


def load_blank_spm_noise(
    blank_path: str | Path,
    target_rms_nm: float = 0.06,
    verbose: bool = True,
    apply_plane_remove: bool = USE_PLANE_REMOVE,
    apply_line_flatten: bool = USE_LINE_FLATTEN,
    apply_bandpass: bool = USE_BANDPASS_FILTER,
) -> tuple[float, np.ndarray, dict]:
    """Load an SPM file and extract a calibrated background noise texture.

    This high-level function handles the full lifecycle of noise extraction:
    1. Parsing the Bruker .spm binary and metadata.
    2. Heuristically selecting the best height channel.
    3. Calibrating raw digital units to physical nanometers.
    4. Leveling, masking, and bandpass-filtering the substrate noise.

    Parameters
    ----------
    blank_path : str | Path
      The path to the blank .spm file.
    target_rms_nm : float, optional
      The target RMS roughness of the extracted noise texture.
      Defaults to 0.06.
    verbose : bool, optional
      Whether to print progress messages and diagnostic statistics.
      Defaults to True.
    apply_plane_remove: bool, optional
      Whether or not to perform plane removal on the image. Defaults to
      USE_PLANE_REMOVE.

    Returns
    -------
    tuple[float, np.ndarray, dict]
      A tuple containing:
      - rms_nm: The RMS roughness of the extracted noise texture.
      - noise_texture: The extracted, zero mean noise texture.
      - diagnostics_dict: A dictionary containing diagnostic information.

    Examples
    --------
    >>> rms_nm, noise_tex, diag = load_blank_spm_noise(blank_path)
    >>> print(rms_nm)
    0.06
    >>> print(noise_tex.shape)
    (800, 800)
    >>> print(diag['flattened'].shape)
    (800, 800)
    >>> print(diag['feature_mask'].shape)
    (800, 800)

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
    diag = {"flattened": z_flat, "feature_mask": feat,
            "height_nm": img_nm, "unit_note": unit_note}
    if verbose:
        print(f"Extracted noise RMS (bg): {rms_nm:.4f} nm | "
              f"tex shape: {noise_tex.shape}")
    return rms_nm, noise_tex, diag


def load_model(
    path: str | Path = MODEL_PATH,
) -> dict[str, np.ndarray]:
    """Load the Power Spectral Density noise model from a compressed archive.

    The model typically contains the statistical distribution of substrate
    roughness in the frequency domain, allowing for the generation of
    synthetic noise that matches the 'color' and texture of the substrate.

    Parameters
    ----------
    path : str | Path, optional
      The path to the compressed archive containing the PSD model.

    Returns
    -------
    dict[str, np.ndarray]
      A dictionary containing the PSD model components namely:
        - "log_psd_mean": The mean of the log-transformed radial PSD.
        - "log_psd_std": The standard deviation of the
                         log-transformed radial PSD.
        - "radial_freq": The spatial frequency coordinates.
        - "rms_values_nm": RMS values associated with the training set.

    Examples
    --------
    >>> model = load_model()
    >>> print(type(model))
    <class 'dict'>

    """

    data = np.load(path)
    return {
        "log_psd_mean":  data["log_psd_mean"].astype(np.float32),
        "log_psd_std":   data["log_psd_std"].astype(np.float32),
        "radial_freq":   data["radial_freq"].astype(np.float32),
        "rms_values_nm": data["rms_values_nm"].astype(np.float32),
    }


def generate_noise(
    seed: int,
    target_shape: tuple[int, int] = TARGET_SHAPE,
    target_rms_nm: float | None = TARGET_NOISE_RMS_NM,
    method: str = "mean_full2d",
    std_scale: float = 1.0,
    model: dict | None = None,
    model_path: str | Path | None = MODEL_PATH,
) -> np.ndarray:
    """Generate a single random AFM-like background noise image from the PSD.

    This function performs Fourier synthesis to create textures that match
    the spatial frequency characteristics of the PSD model provided.

    Parameters
    ----------
    seed : int
        RNG seed for reproducibility.
    target_shape : tuple[int, int]
        Output image shape (rows, cols).
        The PSD is resized automatically if it differs
        from the shape used when building the model. Defaults to TARGET_SHAPE.
    target_rms_nm : float | None
        RMS amplitude in nm.
        If None a value is randomly taken from the blank-file ensemble.
        Defaults to TARGET_NOISE_RMS_NM.
    method : str
        'mean_full2d'      -- mean log-PSD;
        variance comes only from random phase (recommended).
        'lognormal_full2d' -- Samples a new PSD
         from the log-normal distribution. (more sample to sample variation)
        'empirical_radial' -- isotropic synthesis from the mean radial PSD.
    std_scale : float
        Multiplier for the PSD on log_psd_std
        for lognormal method (>1 -> more variation).
    model : dict | None
        Pre-loaded model dict. If None, loaded from model_path on each call.
    model_path : str | Path | None
        Path to the .npz file (used only when model is None).

    Returns
    -------
    z: np.ndarray

    Raises
    ------
    ValueError
        If an unknown method is provided.

    Examples
    --------
    >>> z = generate_noise(seed=42)
    >>> print(z.shape)
    (800, 800)

    """

    if model is None:
        model = load_model(model_path)

    rng = np.random.default_rng(int(seed))
    out_shape = tuple(target_shape)

    if method == "mean_full2d":
        psd_sample = np.exp(model["log_psd_mean"])

    elif method == "lognormal_full2d":
        mu= model["log_psd_mean"]
        noise = rng.standard_normal(mu.shape).astype(np.float32)
        psd_sample = np.exp(mu + std_scale * mu * noise)

    elif method == "empirical_radial":
        mu = model["log_psd_mean"]
        radial_psd = np.exp(mu.mean(axis=1))
        radial_freq_for_interp = np.abs(np.fft.fftfreq(mu.shape[0], d=1.0))

        fy = np.fft.fftfreq(out_shape[0], d=1.0)
        fx = np.fft.rfftfreq(out_shape[1], d=1.0)
        fy2d, fx2d = np.meshgrid(fy, fx, indexing="ij")
        kr = np.sqrt(fy2d ** 2 + fx2d ** 2)

        psd_sample = np.interp(
            kr.ravel(),
            radial_freq_for_interp,
            radial_psd,
            left=float(radial_psd[0]),
            right=float(radial_psd[-1]),
        ).reshape(kr.shape).astype(np.float32)

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
        psd_sample = zoom(psd_sample, (out_ny / psd_ny, rfft_nx / psd_nx),
                          order=1)

    psd_sample = psd_sample.astype(np.float32)

    phase = rng.uniform(0.0, 2.0 * np.pi, size=psd_sample.shape)
    spec  = np.sqrt(np.maximum(psd_sample, EPS)) * np.exp(1j * phase)

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
    """Apply a deterministic random affine transform to a noise texture.

    This function generates a reproducible random transformation of an input
    texture using the provided seed. The transformation includes rotation,
    anisotropic scaling, translation, optional vertical and horizontal flips,
    and a final amplitude jitter.
    It is intended to produce visually distinct noise realizations from a
    single acquired blank SPM texture, while preserving the overall texture
    statistics.

    Parameters
    ----------
    tex: np.ndarray
      Input noise texture of shape (H, W).
    seed: int
      Random seed controlling the random transformation.

    Returns
    -------
    out: np.ndarray
      Transformed noise texture of shape (H, W).

    Examples
    --------
    >>> transformed = random_transform_noise_texture(tex, seed=42)
    transformed
    >>> transformed.shape == tex.shape
    True

    """

    rng  = np.random.default_rng(int(seed))
    tex  = np.asarray(tex, dtype=np.float32)

    theta      = np.deg2rad(float(rng.uniform(0.0, 360.0)))
    base_scale = float(rng.uniform(0.85, 1.15))
    anis       = float(rng.uniform(0.85, 1.15))
    sx, sy     = base_scale * anis, base_scale / anis

    c, s = float(np.cos(theta)), float(np.sin(theta))
    A    = np.array([[c*sx, -s*sy], [s*sx, c*sy]], dtype=np.float32)
    M    = np.linalg.inv(A).astype(np.float32)

    h, w   = tex.shape
    center = np.array([(h-1)/2.0, (w-1)/2.0], dtype=np.float32)
    t      = np.array([float(rng.uniform(-0.15*h, 0.15*h)),
                       float(rng.uniform(-0.15*w, 0.15*w))], dtype=np.float32)
    offset = center - M @ (center + t)

    out = affine_transform(tex, matrix=M, offset=offset,
                           output_shape=tex.shape, order=1,
                           mode="wrap", prefilter=False).astype(np.float32)

    if bool(rng.integers(0, 2)):
        out = out[::-1, :]
    if bool(rng.integers(0, 2)):
        out = out[:, ::-1]
    out *= float(rng.uniform(0.90, 1.10))
    return out.astype(np.float32)


def sample_noise_image(
    noise_source: str,
    noise_asset: dict | np.ndarray | None,
    seed: int,
    out_shape: tuple[int, int],
    target_rms_nm: float | None,
    nm_per_pixel: float = 1.0,
    noise_native_nm_px: float = 0.5,
) -> np.ndarray | None:
    """Return a per-sample noise image from either blank.spm or PSD fallback.

    The noise is generated on the native grid of the noise asset
    (``noise_native_nm_px``), covering the same physical field as the image,
    and then resampled to ``out_shape``, so the noise texture keeps its
    physical length scale whatever ``nm_per_pixel`` is. The result is shifted
    to zero mean and scaled to ``target_rms_nm``.

    Parameters
    ----------
    noise_source : str
      'blank_spm' or 'psd_model'
    noise_asset : dict | np.ndarray | None
      If 'blank_spm', this is the 2D texture array.
      If 'psd_model', this is the dictionary containing PSD statistics.
    seed : int
      RNG seed for reproducibility.
    out_shape : tuple[int, int]
      The (rows, cols) of the desired height map.
    target_rms_nm : float | None
      If provided, rescales the extracted noise texture
      to match this target RMS roughness.
    nm_per_pixel : float, optional
      Pixel size of the output image in nm. Defaults to 1.0.
    noise_native_nm_px : float, optional
      Pixel size the noise asset is sampled at, in nm. Defaults to 0.5.

    Returns
    -------
    np.ndarray | None
      The per-sample noise image or None if disabled.

    Examples
    --------
    >>> noise_tex = sample_noise_image(noise_source, noise_asset,
    ...                                seed, (512, 512), 0.1)  # doctest: +SKIP
    >>> noise_tex.shape  # doctest: +SKIP
    (512, 512)
    """

    if noise_asset is None:
        return None
    req_ny = max(16, int(out_shape[0] * nm_per_pixel / noise_native_nm_px))
    req_nx = max(16, int(out_shape[1] * nm_per_pixel / noise_native_nm_px))
    if noise_source == 'blank_spm':
        noise_tex = random_transform_noise_texture(noise_asset, seed=seed)
        raw_noise = tile_or_crop(noise_tex, (req_ny, req_nx), seed=seed).astype(np.float32)
    elif noise_source == 'psd_model':
        raw_noise = generate_noise(seed=seed, target_shape=(req_ny, req_nx), target_rms_nm=target_rms_nm, method=PSD_NOISE_METHOD, std_scale=PSD_NOISE_STD_SCALE, model=noise_asset).astype(np.float32)
    else:
        return None
    if (req_ny, req_nx) != out_shape:
        zoom_y = out_shape[0] / req_ny
        zoom_x = out_shape[1] / req_nx
        raw_noise = zoom(raw_noise, (zoom_y, zoom_x), order=1)
    raw_noise = raw_noise.astype(np.float32)
    raw_noise -= float(raw_noise.mean())
    if target_rms_nm is not None:
        measured_rms = float(np.sqrt(np.mean(raw_noise.astype(np.float64) ** 2)))
        if measured_rms <= 1e-12 and float(target_rms_nm) > 0:
            raise ValueError('Noise asset has zero variance')
        raw_noise *= float(target_rms_nm) / max(measured_rms, 1e-12)
    return raw_noise


# ─────────────────────────────────────────────────────────────────────────────
# DNA ground-truth mask
# ─────────────────────────────────────────────────────────────────────────────


def nm_to_px(
    coords_xy_nm: np.ndarray,
    extent: tuple[float, float, float, float],
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map (x, y) coordinates in nanometres to integer pixel indices (ix, iy).

    Linearly maps each coordinate from the physical extent to the discrete
    pixel grid, then clips to valid index bounds.

    Parameters
    ----------
    coords_xy_nm : np.ndarray
      The (x, y) coordinates in nanometres.
    extent : tuple[float, float, float, float]
      The physical extent (xmin, xmax, ymin, ymax) in nanometres.¨
    nx : int
      The number of pixels in the x-direction.
    ny : int
      The number of pixels in the y-direction.

    Returns
    -------
    ix, iy : tuple[np.ndarray, np.ndarray]
      The integer pixel indices which are:
        - ix: The x-coordinate indices.
        - iy: The y-coordinate indices.

    Examples
    --------
    >>> ix, iy = nm_to_px(coords_xy_nm, extent, nx, ny)
    >>> print(ix.dtype, iy.dtype)
    int32 int32

    """

    x = coords_xy_nm[:, 0];  y = coords_xy_nm[:, 1]
    xmin, xmax, ymin, ymax = extent
    ix = np.clip(((x - xmin) / (xmax - xmin) * (nx - 1)).astype(np.int32),
                 0, nx - 1)
    iy = np.clip(((y - ymin) / (ymax - ymin) * (ny - 1)).astype(np.int32),
                 0, ny - 1)
    return ix, iy


def rasterize_closed_polyline_mask(
    coords: np.ndarray,
    extent: tuple[float, float, float, float],
    nx: int,
    ny: int,
) -> np.ndarray:
    """Rasterize a closed bead polyline into a 1-pixel-wide binary mask.

    Each consecutive bead pair (including the wrap-around from the last bead
    back to the first) is drawn onto the grid using dense linear interpolation
    along the segment. Duplicate pixels that would arise at segment junctions
    are de-duplicated before writing.

    Parameters
    ----------
    coords : np.ndarray
      Array of shape (N, 3) or (N, 2) containing bead positions in
        nanometres. Only the first two columns (x, y) are used.
    extent : tuple[float, float, float, float]
      The physical bounding box (xmin, xmax, ymin, ymax) in nanometres.
    nx : int
      The number of pixels in the x-axis.
    ny : int
      The number of pixels in the y-axis.

    Returns
    -------
    np.ndarray
      A binary mask of shape (ny, nx),
      with True at every pixel where the polyline is present
      and False everywhere else.

    Examples
    --------
    >>> mask = rasterize_closed_polyline_mask(coords, extent, nx=512, ny=512)
    >>> print(mask.shape, mask.dtype)
    (512, 512) bool

    """

    coords = np.asarray(coords, dtype=np.float32)
    ix, iy = nm_to_px(coords[:, :2], extent, nx, ny)
    out    = np.zeros((ny, nx), dtype=bool)
    npts   = len(ix)
    for k in range(npts):
        k2 = (k + 1) % npts
        x0, y0 = int(ix[k]),  int(iy[k])
        x1, y1 = int(ix[k2]), int(iy[k2])
        n_seg  = int(max(abs(x1 - x0), abs(y1 - y0)) + 1)
        if n_seg <= 1:
            out[y0, x0] = True
            continue
        xs = np.linspace(x0, x1, n_seg).astype(np.int32)
        ys = np.linspace(y0, y1, n_seg).astype(np.int32)
        if xs.size > 1:
            keep     = np.ones(xs.shape[0], dtype=bool)
            keep[1:] = (xs[1:] != xs[:-1]) | (ys[1:] != ys[:-1])
            xs, ys   = xs[keep], ys[keep]
        out[ys, xs] = True
    return out


def make_dna_mask_from_beads(coords, extent, nx, ny, dna_diameter_nm=None, dilate_px=None):
    """Sample a continuous-width tube at native pixel centers; no radius rounding.

    DNA width and mask width use exactly the same diameter. Rasterization still
    has normal pixel-grid uncertainty, but there is no 1.5 multiplier or ceiled
    integer-radius dilation. This does not encode the AFM tip's wider footprint.
    """
    if dna_diameter_nm is None:
        line = rasterize_closed_polyline_mask(coords, extent, nx, ny)
        if dilate_px and dilate_px > 0:
            line = binary_dilation(line, structure=disk_footprint(float(dilate_px)))
        return line.astype(np.uint8)
    radius = float(dna_diameter_nm) / 2
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("DNA/mask width must be finite and positive")
    xmin,xmax,ymin,ymax = map(float,extent)
    px=(xmax-xmin)/(nx-1); py=(ymax-ymin)/(ny-1)
    mask=np.zeros((ny,nx),dtype=bool)
    xy=np.asarray(coords,dtype=float)[:,:2]
    for a,b in zip(xy,np.roll(xy,-1,axis=0)):
        c0=max(0,int(np.floor((min(a[0],b[0])-radius-xmin)/px)))
        c1=min(nx,int(np.ceil((max(a[0],b[0])+radius-xmin)/px))+1)
        r0=max(0,int(np.floor((min(a[1],b[1])-radius-ymin)/py)))
        r1=min(ny,int(np.ceil((max(a[1],b[1])+radius-ymin)/py))+1)
        if r1<=r0 or c1<=c0:
            continue
        yy,xx=np.mgrid[r0:r1,c0:c1]
        dx=xmin+xx*px-a[0];dy=ymin+yy*py-a[1]
        segment=b-a; length2=float(segment@segment)
        t=np.clip((dx*segment[0]+dy*segment[1])/max(length2,1e-20),0,1)
        distance2=(dx-t*segment[0])**2+(dy-t*segment[1])**2
        mask[r0:r1,c0:c1] |= distance2<=radius**2+1e-12
    return mask.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Crossing detection and crossing mask
# ─────────────────────────────────────────────────────────────────────────────


def canon_axis(
    u: np.ndarray) -> np.ndarray | None:
    """Normalize a 2-D vector and canonicalise its sign.

    Ensures that +v and -v are treated as the same axis direction by flipping
    the sign so the first nonzero component is always positive. This makes axis
    comparisons order-independent.

    Parameters
    ----------
    u: np.ndarray
      A 2-D vector to normalize

    Returns
    -------
    np.ndarray | None
      Unit vector with canonicalized sign, or None if u is zero or if input has
      a norm below 1e-12.

    Examples
    --------
    >>> u = np.array([1.0, 2.0])
    >>> v = canon_axis(u)
    >>> print(v)
    [0.4472136 0.8944272]

    """

    u=np.asarray(u,dtype=float)
    nrm=float(np.linalg.norm(u))
    if nrm<1e-12:
        return None
    u=u/nrm
    if (u[0]<0) or (abs(u[0])<1e-12 and u[1]<0):
        u=-u
    return u


def find_polyline_crossings(
    coords_xy: np.ndarray,
    min_separation_beads: int = CROSS_MIN_SEP_BEADS,
    merge_radius_nm: float = 0.5,
) -> list[dict]:
    """Detect all strand crossings of a closed ring polyline in XY.

    Performs a brute-force search over all segment pairs, skipping adjacent
    segments, shared-vertex pairs, and pairs whose circular contour separation
    is less than ``minimum_separation_beads``. Raw intersection hits that land
    within ``merge_radius_nm`` of each other are clustered into a single
    crossing, and duplicate chain-axis directions (within ~18°) are
    deduplicated per cluster.

    Parameters
    ----------
    coords_xy : np.ndarray
      Array of shape (N,2) containing the [x,y] bead positions in nanometers
    min_separation_beads: int, optional
      Minimum circular contour separation (in beads) required between 2
      segments before their intersection is considered. Defaults to
      CROSS_MIN_SEP_BEADS.
    merge_radius_nm: float, optional
      Maximum distance in nanometers between 2 raw intersections that are
      clustered into a single crossing. Defaults to 0.5 nm.

    Returns
    -------
    list[dict]
      Each dict represents one crossing and contains:
        - ``pt_nm``   : np.ndarray, shape (2,) — position in nanometres.
        - ``axes_nm`` : list of np.ndarray — deduplicated unit chain
                        direction vectors at the crossing.
        - ``seg_i``   : int — index of the first segment.
        - ``seg_j``   : int — index of the second segment.
        - ``t``       : float — parametric position along segment i.
        - ``u``       : float — parametric position along segment j.
        - ``n_hits``  : int — number of raw hits merged into this cluster.

    Examples
    --------
    >>> crossings = find_polyline_crossings(coords[:,:2])
    >>> print(type(crossings))
    <class 'list'>

    """

    xy=np.asarray(coords_xy,dtype=float)
    n=int(xy.shape[0])
    raw=[]
    for i in range(n):
        i2=(i+1)%n
        p1,p2=xy[i],xy[i2]
        for j in range(i+1,n):
            j2=(j+1)%n
            if i==j or i2==j or j2==i:
                continue
            if _circ_sep(n,i,j)<int(min_separation_beads):
                continue
            q1,q2=xy[j],xy[j2]
            hit,pt,t,u=_seg_intersect_2d(p1,p2,q1,q2)
            if not hit:
                continue
            ua=canon_axis(p2-p1)
            ub=canon_axis(q2-q1)
            if ua is None or ub is None:
                continue
            raw.append(
                dict(
                    pt=np.array(pt,float),
                    axes=[ua,ub],
                    seg_i=int(i),
                    seg_j=int(j),
                    t=float(t),
                    u=float(u),
                )
            )
    clusters=[]
    for r in raw:
        pt=r["pt"]
        placed=False
        for c in clusters:
            fmc=float(merge_radius_nm)
            centre=c["pt_sum"]/c["n"]
            if np.linalg.norm(pt-centre)<=fmc:
                c["pt_sum"]+=pt
                c["n"]+=1
                c["axes"].extend(r["axes"])
                c["items"].append(r)
                placed=True
                break
        if not placed:
            clusters.append(
                {
                    "pt_sum":pt.copy(),
                    "n":1,
                    "axes":list(r["axes"]),
                    "items":[r],
                }
            )
    out=[]
    for c in clusters:
        pt_mean=c["pt_sum"]/c["n"]
        axes=[]
        for uvec in c["axes"]:
            uvec=canon_axis(uvec)
            if uvec is None:
                continue
            if any(abs(np.dot(uvec,v))>0.95 for v in axes):
                continue
            axes.append(uvec)
        rep=c["items"][0]
        out.append(
            dict(
                pt_nm=np.asarray(pt_mean,dtype=np.float32),
                axes_nm=[np.asarray(v,dtype=np.float32) for v in axes]
                if axes else [np.array([1.0,0.0],np.float32)],
                seg_i=int(rep["seg_i"]),
                seg_j=int(rep["seg_j"]),
                t=float(rep["t"]),
                u=float(rep["u"]),
                n_hits=int(c["n"]),
            )
        )
    return out


def _indices_around_center(center:int,npts:int,half_window:int)->list[int]:
    """Return circular bead indices around a centre bead/segment."""
    return [
        int((center+off)%npts)
        for off in range(-int(half_window),int(half_window)+1)
    ]


def _crossing_centers_from_record(crossing:dict)->tuple[int|None,int|None]:
    """Return top/bottom or segment centres from a crossing record."""
    if "top_center" in crossing and "bottom_center" in crossing:
        return int(crossing["top_center"]),int(crossing["bottom_center"])
    if "seg_i" in crossing and "seg_j" in crossing:
        return int(crossing["seg_i"]),int(crossing["seg_j"])
    return None,None


def _rasterize_local_polyline_mask(
    coords:np.ndarray,
    indices:list[int],
    extent:tuple[float,float,float,float],
    nx:int,
    ny:int,
    radius_px:int,
)->np.ndarray:
    """Rasterize a local circular polyline window into a binary mask."""
    coords=np.asarray(coords,dtype=np.float32)
    npts=int(coords.shape[0])
    xmin,xmax,ymin,ymax=extent
    width=max(float(xmax-xmin),1e-12)
    height=max(float(ymax-ymin),1e-12)
    ix=np.clip(
        ((coords[:,0]-xmin)/width*(int(nx)-1)).astype(np.int32),
        0,
        int(nx)-1,
    )
    iy=np.clip(
        ((coords[:,1]-ymin)/height*(int(ny)-1)).astype(np.int32),
        0,
        int(ny)-1,
    )
    line=np.zeros((int(ny),int(nx)),dtype=bool)
    if len(indices)==1:
        k=int(indices[0])%npts
        line[iy[k],ix[k]]=True
    else:
        for a,b in zip(indices[:-1],indices[1:]):
            a=int(a)%npts
            b=int(b)%npts
            x0,y0=int(ix[a]),int(iy[a])
            x1,y1=int(ix[b]),int(iy[b])
            n_seg=int(max(abs(x1-x0),abs(y1-y0))+1)
            if n_seg<=1:
                line[y0,x0]=True
                continue
            xs=np.linspace(x0,x1,n_seg).astype(np.int32)
            ys=np.linspace(y0,y1,n_seg).astype(np.int32)
            valid=(xs>=0)&(xs<int(nx))&(ys>=0)&(ys<int(ny))
            if np.any(valid):
                line[ys[valid],xs[valid]]=True
    radius_px=max(1,int(radius_px))
    yy,xx=np.ogrid[
        -radius_px:radius_px+1,
        -radius_px:radius_px+1,
    ]
    fp=(xx*xx+yy*yy)<=radius_px*radius_px
    return grey_dilation(line.astype(np.uint8),footprint=fp)>0


def _soften_binary_target(binary:np.ndarray,sigma_px:float)->np.ndarray:
    """Convert a binary target into a smooth distance-transform target."""
    binary=binary.astype(bool)
    if not np.any(binary):
        return np.zeros(binary.shape,dtype=np.float32)
    sigma_px=max(float(sigma_px),1e-6)
    dist=distance_transform_edt(~binary).astype(np.float32)
    target=np.exp(-(dist**2)/(2.0*sigma_px**2)).astype(np.float32)
    target[binary]=1.0
    return target


def add_crossing_patch(
    mask:np.ndarray,
    pt_px:tuple[float,float],
    axes_px:list[np.ndarray],
    sigma_center:float=2.0,
    chain_extent:float=5.0,
    sigma_perp:float=1.8,
    center_weight:float=1.0,
    chain_weight:float=0.8,
)->None:
    """Paint a compact isotropic crossing patch.

    Kept for backwards compatibility with older notebook cells.
    """
    h,w=mask.shape
    cx,cy=float(pt_px[0]),float(pt_px[1])
    sigma=max(float(sigma_center),float(sigma_perp),1e-6)
    radius=int(np.ceil(3.0*sigma))
    x0=max(0,int(np.floor(cx-radius)))
    x1=min(w-1,int(np.ceil(cx+radius)))
    y0=max(0,int(np.floor(cy-radius)))
    y1=min(h-1,int(np.ceil(cy+radius)))
    xs=np.arange(x0,x1+1,dtype=float)
    ys=np.arange(y0,y1+1,dtype=float)
    xg,yg=np.meshgrid(xs,ys)
    dx=xg-cx
    dy=yg-cy
    patch=np.exp(-0.5*(dx*dx+dy*dy)/(sigma*sigma)).astype(np.float32)
    patch*=float(center_weight)
    mask[y0:y1+1,x0:x1+1]=np.maximum(
        mask[y0:y1+1,x0:x1+1],
        patch,
    )


def make_crossing_mask(
    coords:np.ndarray,
    extent:tuple[float,float,float,float],
    nx:int,
    ny:int,
    sigma_center_px:float=CROSS_SIGMA_CENTER_PX,
    sigma_perp_px:float=CROSS_SIGMA_PERP_PX,
    chain_extent:float=CROSS_CHAIN_EXTENT,
    center_weight:float=CROSS_CENTER_WEIGHT,
    chain_weight:float=CROSS_CHAIN_WEIGHT,
    min_separation_beads:int=CROSS_MIN_SEP_BEADS,
    crossings_precomputed:list[dict]|None=None,
    merge_radius_nm:float=0.5,
)->tuple[np.ndarray,list[dict]]:
    """Build the crossing ground-truth mask as a soft footprint target."""
    coords=np.asarray(coords,dtype=np.float32)
    xy_nm=coords[:,:2].astype(float)
    npts=int(coords.shape[0])
    if crossings_precomputed is None:
        crossings=find_polyline_crossings(
            xy_nm,
            min_separation_beads=min_separation_beads,
            merge_radius_nm=float(merge_radius_nm),
        )
    else:
        crossings=crossings_precomputed
    mask=np.zeros((int(ny),int(nx)),dtype=np.float32)
    if not crossings:
        return mask.astype(np.float32),crossings
    strand_radius_px=max(
        1,
        int(np.ceil(max(float(sigma_perp_px),0.5))),
    )
    soft_sigma_px=max(0.5,float(sigma_center_px))
    half_window=max(1,int(round(float(chain_extent))))
    overlap_binary=np.zeros((int(ny),int(nx)),dtype=bool)
    union_binary=np.zeros((int(ny),int(nx)),dtype=bool)
    for crossing in crossings:
        centre_a,centre_b=_crossing_centers_from_record(crossing)
        if centre_a is None or centre_b is None:
            continue
        indices_a=_indices_around_center(
            centre_a,
            npts=npts,
            half_window=half_window,
        )
        indices_b=_indices_around_center(
            centre_b,
            npts=npts,
            half_window=half_window,
        )
        mask_a=_rasterize_local_polyline_mask(
            coords=coords,
            indices=indices_a,
            extent=extent,
            nx=int(nx),
            ny=int(ny),
            radius_px=strand_radius_px,
        )
        mask_b=_rasterize_local_polyline_mask(
            coords=coords,
            indices=indices_b,
            extent=extent,
            nx=int(nx),
            ny=int(ny),
            radius_px=strand_radius_px,
        )
        local_overlap=mask_a&mask_b
        local_union=mask_a|mask_b
        if not np.any(local_overlap):
            xmin,xmax,ymin,ymax=extent
            sx=(int(nx)-1)/max(float(xmax-xmin),1e-12)
            sy=(int(ny)-1)/max(float(ymax-ymin),1e-12)
            x_nm,y_nm=crossing["pt_nm"]
            px=(float(x_nm)-float(xmin))*sx
            py=(float(y_nm)-float(ymin))*sy
            tmp=np.zeros_like(mask,dtype=np.float32)
            add_crossing_patch(
                tmp,
                pt_px=(px,py),
                axes_px=[],
                sigma_center=soft_sigma_px,
                chain_extent=chain_extent,
                sigma_perp=strand_radius_px,
                center_weight=1.0,
                chain_weight=0.0,
            )
            local_overlap=tmp>0.25
        overlap_binary|=local_overlap
        union_binary|=local_union
    overlap_soft=_soften_binary_target(
        overlap_binary,
        sigma_px=soft_sigma_px,
    )
    union_soft=_soften_binary_target(
        union_binary,
        sigma_px=max(0.5,0.5*soft_sigma_px),
    )
    mask=np.maximum(
        float(center_weight)*overlap_soft,
        float(chain_weight)*union_soft,
    )
    m=float(mask.max())
    if m>1e-12:
        mask/=m
    return mask.astype(np.float32),crossings


# ─────────────────────────────────────────────────────────────────────────────
# Decoy fragments
# ─────────────────────────────────────────────────────────────────────────────


def make_decoy_chain(
    seed: int,
    n_beads_range: tuple[int, int] = (2, 4),
    bond_length: float = BOND_LENGTH,
) -> np.ndarray:
    """Generate a small (2–4 bead) slightly curved chain without MD relaxation.

    This function is used to generate a small decoy chain segment with the
    number of beads given by ``n_beads_range``.

    Parameters
    ----------
    seed: int
      Random seed for number of beads within range, ``n_beads_range``.
    n_beads_range: tuple[int, int], optional
      Range of number of beads to generate the decoy. Defaults to (2,4)
    bond_length: float, optional
      Bond length in nanometers. Defaults to BOND_LENGTH.

    Returns
    -------
    coords: np.ndarray
      Array of shape (N,3) containing bead positions in nanometers.

    Examples
    --------
    >>> coords = make_decoy_chain(seed=42)
    >>> print(coords.shape)
    (3, 3)

    """

    rng    = np.random.default_rng(int(seed))
    n_b    = rng.integers(n_beads_range[0], n_beads_range[1] + 1)
    coords = np.zeros((n_b, 3), dtype=np.float32)
    angle  = rng.uniform(0, 2 * np.pi)
    dirn   = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32)

    for i in range(1, n_b):
        da = rng.normal(0, 0.3)
        c, s = np.cos(da), np.sin(da)
        R    = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        dirn = R @ dirn
        dirn = dirn / np.linalg.norm(dirn)
        coords[i] = coords[i-1] + bond_length * dirn

    coords[:, 2] = rng.uniform(2.5, 4.0, n_b)
    return coords


def place_decoy_in_corner(
    decoy_coords: np.ndarray,
    global_extent: tuple[float, float, float, float],
    corner_margin_nm: float = 2.0,
    seed: int | None = None,
) -> np.ndarray:
    """Translate decoy_coords so its centroid lands in a random corner.

    This function takes the coordinates of the decoy and places it at a
    distance of corner_margin_nm away from one of the corners of the image.

    Parameters
    ----------
    decoy_coords: np.ndarray
      Array of shape (N,3) containing bead positions in nanometers.
    global_extent: tuple[float float float float]
      The physical bounding box (xmin, xmax, ymin, ymax) in nanometers.
    corner_margin_nm: float, optional
      Distance in nanometers to place the decoy inside the frame.
      Defaults to 2.0.
    seed: int | None, optional
      Random seed for corner selection. Defaults to None.

    Returns
    -------
    decoy_coords: np.ndarray

    Examples
    --------
    >>> decoy_coords = place_decoy_in_corner(decoy_coords, global_extent)
    >>> print(decoy_coords.shape)
    (3, 3)

    """

    if seed is None:
        seed = int(np.sum(decoy_coords))
    rng    = np.random.default_rng(int(seed))
    xmin, xmax, ymin, ymax = global_extent
    corner = rng.integers(0, 4)
    targets = [
        (xmin + corner_margin_nm, ymax - corner_margin_nm),   # top-left
        (xmax - corner_margin_nm, ymax - corner_margin_nm),   # top-right
        (xmin + corner_margin_nm, ymin + corner_margin_nm),   # bottom-left
        (xmax - corner_margin_nm, ymin + corner_margin_nm),   # bottom-right
    ]
    tx, ty = targets[corner]
    centroid = decoy_coords.mean(axis=0)
    offset   = np.array([tx, ty, 0.0]) - centroid
    return decoy_coords + offset


# ─────────────────────────────────────────────────────────────────────────────
# Sample pipeline
# ─────────────────────────────────────────────────────────────────────────────


def load_noise_asset(verbose: bool = True) -> tuple[str, Any, dict]:
    """Load the noise source chosen in the noise cell.

    A blank ``.spm`` scan (``BLANK_SPM_PATH``) is preferred. If it is missing or
    cannot be parsed, the PSD noise model (``MODEL_PATH``) is used. If neither
    is available, noise injection is disabled.

    Parameters
    ----------
    verbose : bool, optional
        Print which noise source was loaded. Defaults to ``True``.

    Returns
    -------
    tuple[str, Any, dict]
        ``(noise_source, noise_asset, diagnostics)`` where ``noise_source`` is
        ``"blank_spm"``, ``"psd_model"`` or ``"none"``.
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
                    print(f"Loaded blank .spm noise texture  RMS ≈ {noise_rms_nm:.3f} nm")
            except Exception as e:
                print("blank .spm noise failed; trying PSD fallback. Error:", e)
        elif verbose:
            print(f"No blank .spm file found at {blank_path!r}; trying PSD fallback noise model.")

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
            print("PSD fallback noise model failed; proceeding without noise. Error:", e)
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

    MD pathway (``USE_MD = True``): the contour length is converted to beads
    (:func:`contour_to_beads`), the periodic-bridge initial ring is lifted to
    ``BASE_Z`` (:func:`make_tangled_ring_initial_safe`) and relaxed onto the
    substrate with the annealed OpenMM force field
    (:func:`run_md_relaxation`). Implausible results (final contour outside
    75-125 % of the request, or degenerate bonds) raise ``ValueError`` so the
    caller can try the next seed.

    Non-MD pathway: :func:`make_non_md_ring` draws a 2-D persistent ring whose
    contour equals the request exactly.

    Self-crossings of the final (x, y) projection are then detected and an
    optional decoy fragment is generated.

    Parameters
    ----------
    seed : int
        Random seed (initial ring and Langevin integrator).
    contour_length_nm : float, optional
        Requested contour length in nm. Defaults to ``CONTOUR_LENGTH_NM``.
    add_decoy : bool, optional
        Add a short decoy fragment near a corner. Defaults to ``False``.
    decoy_n_beads_range : tuple[int, int], optional
        Bead-count range of the decoy fragment. Defaults to ``(2, 4)``.
    return_frames : bool, optional
        Also return every MD frame (used for the animation). Defaults to
        ``False``.

    Returns
    -------
    dict[str, Any]
        Keys ``seed``, ``pathway``, ``requested_contour_nm``, ``n_beads``,
        ``bond_length_nm``, ``coords`` (``(N, 3)``, nm), ``crossings``,
        ``decoy_coords``, ``projected_contour_nm``, ``contour_3d_nm`` and, if
        requested, ``frames``.

    Examples
    --------
    >>> chain = generate_chain(seed=3, contour_length_nm=115.26)
    >>> chain["n_beads"], round(chain["projected_contour_nm"], 1)  # doctest: +SKIP
    (26, 115.1)
    """

    seed = int(seed)
    length = float(CONTOUR_LENGTH_NM if contour_length_nm is None else contour_length_nm)
    frames = None

    if USE_MD:
        if openmm is None:
            raise ImportError("USE_MD = True needs OpenMM; install it or set USE_MD = False.")
        n_beads, bond = contour_to_beads(length, BOND_LENGTH, MD_MIN_BEADS)
        coords0 = make_tangled_ring_initial_safe(
            seed, n_beads, bond, persistence_bonds=PERSISTENCE_BONDS, base_z=BASE_Z,
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
        decoy_raw = make_decoy_chain(seed + 9999, n_beads_range=decoy_n_beads_range,
                                     bond_length=BOND_LENGTH)
        decoy_coords = place_decoy_in_corner(
            decoy_raw, (xmin, xmax, ymin, ymax), corner_margin_nm=15.0, seed=seed)

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
    noise_source: str = "none",
    noise_asset: dict | np.ndarray | None = None,
    dna_height_nm: float | None = None,
    tip_radius_nm: float | None = None,
    target_noise_rms_nm: float | None = None,
) -> dict[str, Any]:
    """Render the AFM image, DNA mask and crossing mask of a chain.

    The field of view is ``NX x NY`` pixels of ``NM_PER_PIXEL`` nm, centred on
    the chain. A chain that does not fit inside it (with a margin of the DNA
    width plus the tip radius) raises ``ValueError`` instead of being cropped,
    so dataset generation simply moves on to the next seed.

    The image is rendered with :func:`create_z_based_afm` using ``AFM_KW``.
    ``dna_height_nm`` and ``tip_radius_nm`` override the values in
    ``AFM_KW`` for this sample (the dataset draws them from ranges). Decoys
    are composited into the image but excluded from both masks. Noise is
    added after rendering. Both masks use ``DNA_MASK_DIAMETER_NM`` as the DNA
    width, independently of the rendered width.

    Crossing-height boosting uses the crossings stored in ``chain`` for the MD
    pathway. The non-MD renderer was calibrated with crossings detected at
    ``AFM_KW["min_separation_beads"]``, so those are recomputed here.

    Parameters
    ----------
    chain : dict
        Output of :func:`generate_chain`.
    noise_source : str, optional
        ``"blank_spm"``, ``"psd_model"`` or ``"none"``. Defaults to ``"none"``.
    noise_asset : dict | np.ndarray | None, optional
        Asset returned by :func:`load_noise_asset`.
    dna_height_nm : float, optional
        Rendered DNA height in nm. Defaults to ``AFM_KW["dna_height_nm"]``.
    tip_radius_nm : float, optional
        Tip radius in nm. Defaults to ``AFM_KW["tip_radius_nm"]``.
    target_noise_rms_nm : float, optional
        Noise RMS in nm. Defaults to ``TARGET_NOISE_RMS_NM``.

    Returns
    -------
    dict[str, Any]
        ``afm_img`` (noisy height map, nm), ``afm_img_clean``, ``dna_mask``
        (uint8), ``cross_mask`` (float32, 0-1), ``extent`` (nm),
        ``n_crossings``, ``decoy_coords`` and the rendering parameters used.

    Raises
    ------
    ValueError
        If the chain does not fit in the field of view.
    """

    seed = int(chain["seed"])
    coords = chain["coords"]
    decoy_coords = chain.get("decoy_coords", None)

    afm_kw = dict(AFM_KW)
    for key in ("target_nm_per_px", "render_nm_per_px", "output_nm_per_px", "extent"):
        afm_kw.pop(key, None)
    if dna_height_nm is not None:
        afm_kw["dna_height_nm"] = float(dna_height_nm)
    if tip_radius_nm is not None:
        afm_kw["tip_radius_nm"] = float(tip_radius_nm)
    noise_rms = float(TARGET_NOISE_RMS_NM if target_noise_rms_nm is None else target_noise_rms_nm)
    if not np.isfinite(noise_rms) or noise_rms < 0:
        raise ValueError("target_noise_rms_nm must be finite and non-negative")

    # ── Field of view centred on the chain ───────────────────────────────────
    pixel_nm = float(NM_PER_PIXEL)
    cx = (coords[:, 0].min() + coords[:, 0].max()) / 2.0
    cy = (coords[:, 1].min() + coords[:, 1].max()) / 2.0
    width_nm = (NX - 1) * pixel_nm
    height_nm = (NY - 1) * pixel_nm
    extent = (float(cx - width_nm / 2.0), float(cx + width_nm / 2.0),
              float(cy - height_nm / 2.0), float(cy + height_nm / 2.0))
    margin_nm = float(afm_kw["dna_diameter_nm"]) + float(afm_kw["tip_radius_nm"])
    if (coords[:, 0].min() < extent[0] + margin_nm or coords[:, 0].max() > extent[1] - margin_nm
            or coords[:, 1].min() < extent[2] + margin_nm or coords[:, 1].max() > extent[3] - margin_nm):
        raise ValueError("Chain does not fit in the field of view; it is never cropped. "
                         "Use a larger NM_PER_PIXEL / NX / NY or a shorter contour.")

    if USE_MD:
        render_crossings = chain["crossings"]
    else:
        render_crossings = collect_intersections(
            coords[:, :2], min_separation_beads=int(afm_kw["min_separation_beads"]))

    afm_kw.update(nx=NX, ny=NY, grain_seed=seed, crossings_precomputed=render_crossings,
                  return_masks=True, return_crossing_info=True)
    afm_img, dbg = create_z_based_afm(coords, extent=extent, **afm_kw)
    extent = tuple(dbg["extent"])

    # ── Decoy fragment (image only, excluded from the masks) ─────────────────
    if decoy_coords is not None:
        decoy_coords = place_decoy_in_corner(decoy_coords, extent, corner_margin_nm=8.0, seed=seed)
        decoy_kw = dict(afm_kw)
        decoy_kw.update(apply_edge_taper=False, enable_crossing_boost=False, add_center_ridge=False,
                        crossings_precomputed=None, return_crossing_info=False)
        decoy_afm, _ = create_z_based_afm(decoy_coords, extent=extent, **decoy_kw)
        afm_img = np.maximum(afm_img, decoy_afm)

    # ── Noise ────────────────────────────────────────────────────────────────
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
        afm_img = np.minimum(afm_img + noise_img.astype(np.float32),
                             afm_kw["max_height_nm"]).astype(np.float32)

    # ── Masks (annotation width DNA_MASK_DIAMETER_NM) ────────────────────────
    mask_width = float(DNA_MASK_DIAMETER_NM)
    dna_mask = make_dna_mask_from_beads(coords, extent, NX, NY, dna_diameter_nm=mask_width).astype(np.uint8)
    xmin, xmax, ymin, ymax = extent
    output_nm_per_px = 0.5 * ((xmax - xmin) / max(NX - 1, 1) + (ymax - ymin) / max(NY - 1, 1))
    sigma_center_px = max(0.5, CROSS_SIGMA_CENTER_DIAM_MULT * mask_width / max(output_nm_per_px, 1e-12))
    sigma_perp_px = max(0.5, CROSS_SIGMA_PERP_DIAM_MULT * mask_width / max(output_nm_per_px, 1e-12))
    cross_mask, crossings_out = make_crossing_mask(
        coords, extent, NX, NY,
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
    noise_source: str = "none",
    noise_asset: dict | np.ndarray | None = None,
    add_decoy: bool = False,
    decoy_n_beads_range: tuple[int, int] = (2, 4),
    dna_height_nm: float | None = None,
    tip_radius_nm: float | None = None,
    target_noise_rms_nm: float | None = None,
) -> dict[str, Any]:
    """Generate one complete sample: relaxed chain, AFM image and both masks.

    Calls :func:`generate_chain` and :func:`render_chain_and_masks` and merges
    their outputs (chain coordinates, requested and measured contour lengths,
    bead count and bond length are kept for the metadata).

    Parameters
    ----------
    seed : int
        Random seed.
    contour_length_nm : float, optional
        Requested contour length in nm. Defaults to ``CONTOUR_LENGTH_NM``.
    noise_source, noise_asset
        Output of :func:`load_noise_asset`.
    add_decoy : bool, optional
        Add a decoy fragment. Defaults to ``False``.
    decoy_n_beads_range : tuple[int, int], optional
        Decoy bead-count range.
    dna_height_nm, tip_radius_nm, target_noise_rms_nm : float, optional
        Per-sample rendering overrides (see :func:`render_chain_and_masks`).

    Returns
    -------
    dict[str, Any]
        The rendered sample plus the chain description.

    Examples
    --------
    >>> sample = generate_one_sample(seed=1, contour_length_nm=115.26)  # doctest: +SKIP
    >>> sample["afm_img"].shape == (NY, NX)  # doctest: +SKIP
    True
    """

    chain = generate_chain(seed, contour_length_nm, add_decoy=add_decoy,
                           decoy_n_beads_range=decoy_n_beads_range)
    sample = render_chain_and_masks(
        chain,
        noise_source=noise_source,
        noise_asset=noise_asset,
        dna_height_nm=dna_height_nm,
        tip_radius_nm=tip_radius_nm,
        target_noise_rms_nm=target_noise_rms_nm,
    )
    for key in ("pathway", "requested_contour_nm", "n_beads", "bond_length_nm", "coords",
                "projected_contour_nm", "contour_3d_nm"):
        sample[key] = chain[key]
    sample["has_decoy"] = chain["decoy_coords"] is not None
    return sample


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────


def nice_scale_bar_nm(field_nm: float) -> float:
    """Pick a round scale-bar length of roughly a fifth of the field of view."""

    target = field_nm / 5.0
    for value in (500, 200, 100, 50, 20, 10, 5, 2, 1):
        if value <= target:
            return float(value)
    return 1.0


def add_scale_bar(
    ax,
    extent: tuple[float, float, float, float],
    length_nm: float | None = None,
    color: str = "white",
) -> None:
    """Draw a labelled scale bar in the lower-left corner of an image axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis showing an image drawn with ``extent=extent`` (nm) and
        ``origin="lower"``.
    extent : tuple
        Image extent ``(xmin, xmax, ymin, ymax)`` in nm.
    length_nm : float, optional
        Bar length. Defaults to a round value of about a fifth of the field.
    color : str, optional
        Bar and label colour. Defaults to white.
    """

    xmin, xmax, ymin, ymax = [float(v) for v in extent]
    length = length_nm or nice_scale_bar_nm(xmax - xmin)
    x0 = xmin + 0.05 * (xmax - xmin)
    y0 = ymin + 0.06 * (ymax - ymin)
    ax.plot([x0, x0 + length], [y0, y0], color=color, linewidth=3, solid_capstyle="butt")
    ax.text(x0 + length / 2, y0 + 0.025 * (ymax - ymin), f"{length:g} nm",
            color=color, ha="center", va="bottom", fontsize=9)


def generate_sanity_samples(
    seeds: list[int],
    contour_lengths_nm: list[float],
    add_decoy: bool,
    noise_source: str,
    noise_asset: Any,
    max_extra_seeds: int = 20,
) -> list[dict[str, Any]]:
    """Generate one sample per (seed, contour length) pair for the sanity check.

    If a seed fails (e.g. the molecule does not fit in the field of view or the
    MD run is rejected), the next seed is tried, up to ``max_extra_seeds``
    times, and a note is printed.

    Parameters
    ----------
    seeds : list[int]
        Starting seed for each sample.
    contour_lengths_nm : list[float]
        Requested contour length for each sample.
    add_decoy : bool
        Add a decoy fragment to every sample.
    noise_source, noise_asset
        Output of :func:`load_noise_asset`.
    max_extra_seeds : int, optional
        How many following seeds to try when one fails. Defaults to 20.

    Returns
    -------
    list[dict[str, Any]]
        One sample dictionary per requested pair.
    """

    samples = []
    for seed, length in zip(seeds, contour_lengths_nm):
        last_error = None
        for trial in range(max_extra_seeds + 1):
            try:
                samples.append(generate_one_sample(
                    seed + trial, length, noise_source=noise_source, noise_asset=noise_asset,
                    add_decoy=add_decoy))
                if trial:
                    print(f"Seed {seed} skipped ({last_error}); used seed {seed + trial} instead.")
                break
            except Exception as exc:  # try the next seed
                last_error = f"{type(exc).__name__}: {exc}"
        else:
            raise RuntimeError(f"No valid sample for seeds {seed}-{seed + max_extra_seeds}: {last_error}")
    return samples


def plot_sanity_samples(samples: list[dict[str, Any]]) -> None:
    """Show image, DNA mask and crossing mask of each sample, with scale bars.

    Each row is one molecule. The title above the image lists the seed, the
    measured contour length (projected, closed centre line) and the number of
    crossings. Decoys appear in the image only.

    Parameters
    ----------
    samples : list[dict[str, Any]]
        Output of :func:`generate_sanity_samples`.
    """

    fig, axes = plt.subplots(len(samples), 3, figsize=(15, 5 * len(samples)), squeeze=False)
    for r, s in enumerate(samples):
        ext = [float(v) for v in s["extent"]]
        ax1, ax2, ax3 = axes[r]
        ax1.imshow(s["afm_img"], cmap="afmhot", origin="lower", extent=ext, vmin=-0.5, vmax=4.0)
        decoy = " · with decoy" if s.get("has_decoy") else ""
        crossings = f"{s['n_crossings']} crossing" + ("" if s["n_crossings"] == 1 else "s")
        ax1.set_title(f"Seed {s['seed']} · contour ≈ {s['projected_contour_nm']:.1f} nm"
                      f" · {crossings}{decoy}")
        ax2.imshow(s["dna_mask"], cmap="gray", origin="lower", extent=ext)
        ax2.set_title("DNA mask (decoy excluded)")
        ax3.imshow(s["cross_mask"], cmap="magma", origin="lower", extent=ext)
        ax3.set_title("Crossing mask")
        for ax in (ax1, ax2, ax3):
            add_scale_bar(ax, ext)
            ax.axis("off")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Benchmarking
# ─────────────────────────────────────────────────────────────────────────────


def bench_stages(
    seed: int,
    contour_length_nm: float,
) -> dict[str, float]:
    """Time the main stages of one sample.

    Parameters
    ----------
    seed : int
        Random seed for the benchmarked sample.
    contour_length_nm : float
        Requested contour length in nm.

    Returns
    -------
    dict[str, float]
        Seconds spent in ``"1_chain_init"``, ``"2_md_relax"`` (the non-MD ring
        for the non-MD pathway), ``"3_crossing_detect"`` and
        ``"4_afm_render_masks"``.
    """

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    if USE_MD:
        n_beads, bond = contour_to_beads(contour_length_nm, BOND_LENGTH, MD_MIN_BEADS)
        coords0 = make_tangled_ring_initial_safe(int(seed), n_beads, bond,
                                                 persistence_bonds=PERSISTENCE_BONDS, base_z=BASE_Z)
        timings["1_chain_init"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        frames = run_md_relaxation(coords0, seed=int(seed), n_frames=N_FRAMES,
                                   steps_per_frame=STEPS_PER_FRAME, bond_length=bond)
        coords = np.asarray(frames[-1], dtype=np.float32)
        timings["2_md_relax"] = time.perf_counter() - t0
    else:
        timings["1_chain_init"] = 0.0
        coords = make_non_md_ring(int(seed), contour_length_nm)
        timings["2_md_relax"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    crossings = find_polyline_crossings(coords[:, :2], min_separation_beads=CROSS_MIN_SEP_BEADS,
                                        merge_radius_nm=0.5)
    timings["3_crossing_detect"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    chain = dict(seed=int(seed), coords=coords, crossings=crossings, decoy_coords=None)
    try:
        render_chain_and_masks(chain)
    except ValueError:
        pass  # a molecule that does not fit is still a valid timing of the render setup
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
    """Benchmark sample generation and estimate the time for a full dataset.

    Prints a system profile, runs ``n_reps`` timed samples, records the peak
    memory with ``tracker_cls`` (the ``_PeakMemTracker`` class defined in the
    notebook) and projects the total runtime.

    Parameters
    ----------
    seed : int
        Base random seed; repetition ``k`` uses ``seed + k``.
    contour_length_nm : float
        Contour length of the benchmarked molecules (nm).
    n_reps : int, optional
        Number of repetitions. Defaults to 3.
    total_samples : int, optional
        Dataset size used for the runtime projection.
    tracker_cls : type, optional
        Class with ``start()``, ``stop()`` and ``peak_mb``.
    print_report : bool, optional
        Print the report. Defaults to ``True``.

    Returns
    -------
    dict[str, Any]
        Wall times, stage timings, medians, peak memory and the projection.
    """

    import psutil

    cpu_logical = psutil.cpu_count(logical=True)
    available_ram_gb = psutil.virtual_memory().available / 1e9
    platforms = None
    if USE_MD and openmm is not None:
        platforms = [openmm.Platform.getPlatform(i).getName()
                     for i in range(openmm.Platform.getNumPlatforms())]

    if print_report:
        print("=" * 62)
        print("  SYSTEM PROFILE")
        print("=" * 62)
        print(f"  OS              : {platform.system()} {platform.release()}")
        print(f"  CPU (logical)   : {cpu_logical} cores")
        print(f"  RAM available   : {available_ram_gb:.1f} GB")
        if platforms is not None:
            print(f"  OpenMM Platforms: {platforms}")
        print(f"  Mode            : {'MD (OpenMM)' if USE_MD else 'Non-MD (2-D ring)'}")
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
    median_stages = {key: statistics.median(t[key] for t in stage_timings) for key in stage_timings[0]}
    projected_hours = median_wall * int(total_samples) / 3600 if total_samples else None
    peak_mb = getattr(tracker, "peak_mb", None)

    if print_report:
        print("=" * 62)
        print(f"  Median Wall Time: {median_wall:.2f} s")
        for key, value in median_stages.items():
            print(f"    {key:<20s}: {value:.2f} s")
        if peak_mb is not None:
            print(f"  Peak RAM (RSS)  : {peak_mb:.1f} MB")
        if projected_hours is not None:
            print(f"  Projected Total : {projected_hours:.1f} hours for {int(total_samples)} samples")
        print("=" * 62)

    return dict(seed=int(seed), contour_length_nm=float(contour_length_nm), n_reps=int(n_reps),
                mode="md" if USE_MD else "non_md", cpu_logical=cpu_logical,
                available_ram_gb=available_ram_gb, openmm_platforms=platforms,
                wall_times_s=wall_times, stage_timings_s=stage_timings,
                median_wall_time_s=median_wall, median_stage_times_s=median_stages,
                peak_rss_mb=peak_mb, total_samples=total_samples,
                projected_total_hours=projected_hours)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset generation
# ─────────────────────────────────────────────────────────────────────────────


def draw_value(spec: Any, rng: np.random.Generator) -> float:
    """Draw one value from a range or a list of discrete values.

    * a single number → always that number;
    * a list ``[a, b, c]`` → one of the listed values, each equally likely;
    * a tuple ``(low, high)`` → uniformly distributed between ``low`` and ``high``.

    Parameters
    ----------
    spec : float | list | tuple
        The setting as written in the notebook.
    rng : np.random.Generator
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
    >>> rng = np.random.default_rng(0)
    >>> draw_value(2.3, rng)
    2.3
    >>> draw_value([115.26, 768.4], rng) in (115.26, 768.4)
    True
    >>> 6.0 <= draw_value((6.0, 8.0), rng) <= 8.0
    True
    """

    if isinstance(spec, tuple):
        if len(spec) != 2 or float(spec[0]) > float(spec[1]):
            raise ValueError(f"A range must be (low, high) with low <= high, got {spec}")
        return float(rng.uniform(float(spec[0]), float(spec[1])))
    if isinstance(spec, list):
        if not spec:
            raise ValueError("A list of discrete values must not be empty")
        return float(spec[int(rng.integers(0, len(spec)))])
    return float(spec)


def describe_spec(spec: Any) -> str:
    """Describe a range/list setting in words (for printing)."""

    if isinstance(spec, tuple):
        return f"uniform between {spec[0]:g} and {spec[1]:g}"
    if isinstance(spec, list):
        return "one of " + ", ".join(f"{v:g}" for v in spec)
    return f"fixed at {float(spec):g}"


_SAMPLE_EXECUTOR = None


def get_sample_executor():
    """Return the reusable single-worker process pool used for sample generation.

    Each sample runs in a separate Python process, which makes the time-out
    independent of operating-system signal handling (works on Linux, macOS and
    Windows). The worker is created when first needed and then reused, which
    avoids starting a new process per seed.

    Returns
    -------
    loky.reusable_executor._ReusablePoolExecutor
        Executor running one simulation at a time.
    """

    global _SAMPLE_EXECUTOR
    if _SAMPLE_EXECUTOR is None:
        try:
            from joblib.externals.loky import get_reusable_executor
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])
            from joblib.externals.loky import get_reusable_executor
        _SAMPLE_EXECUTOR = get_reusable_executor(max_workers=1)
    return _SAMPLE_EXECUTOR


def reset_sample_executor() -> None:
    """Shut down (and kill) the sample worker after a time-out or error.

    Killing the worker stops an OpenMM run that would otherwise continue in
    the background after its seed has been skipped.
    """

    global _SAMPLE_EXECUTOR
    if _SAMPLE_EXECUTOR is not None:
        _SAMPLE_EXECUTOR.shutdown(wait=False, kill_workers=True)
    _SAMPLE_EXECUTOR = None


_WORKER_STATE: dict[str, Any] = {}


def generate_sample_worker(
    settings: dict[str, Any],
    noise_source: str,
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

    The worker is a fresh Python process, so the notebook settings are passed
    along and applied with :func:`configure` before the sample is generated.

    Returns
    -------
    dict[str, Any]
        Output of :func:`generate_one_sample`.
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
    noise_source: str,
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
    """Generate and validate one sample, giving up after ``timeout_s`` seconds.

    The sample is generated in the worker process. If it takes longer than
    the time-out, the worker is killed so the caller can skip the seed.
    Errors raised by OpenMM or the simulation are passed on for the same
    reason.

    Returns
    -------
    dict[str, Any]
        The validated sample.

    Raises
    ------
    TimeoutError
        If generation exceeds the time-out.
    ValueError
        If an output array contains non-finite values (or the simulator
        rejected the seed).
    """

    timeout = float(SAMPLE_TIMEOUT_S if timeout_s is None else timeout_s)
    executor = get_sample_executor()
    start = time.perf_counter()
    future = executor.submit(
        generate_sample_worker, current_settings(), noise_source, noise_asset,
        int(seed), float(contour_length_nm), bool(add_decoy),
        tuple(int(x) for x in decoy_n_beads_range),
        float(dna_height_nm), float(tip_radius_nm), float(noise_rms_nm),
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
    """Save a PNG preview of a generated AFM image (image only, no masks).

    Parameters
    ----------
    sample : dict
        Generated sample containing ``afm_img`` and ``extent``.
    idx : int
        Dataset index used in the file name.
    preview_dir : str
        Output directory.

    Returns
    -------
    str
        Path of the written PNG.
    """

    preview_path = os.path.join(preview_dir, f"preview_{idx:04d}.png")
    ext = [float(v) for v in sample["extent"]]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(sample["afm_img"], cmap="afmhot", origin="lower", extent=ext, vmin=-0.5, vmax=4.0)
    add_scale_bar(ax, ext)
    ax.set_title(f"#{idx} · seed {sample['seed']} · {sample['projected_contour_nm']:.1f} nm", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(preview_path, dpi=150)
    plt.close(fig)
    return preview_path


MANIFEST_HEADER = [
    "index", "seed", "pathway", "requested_contour_nm", "projected_contour_nm", "contour_3d_nm",
    "n_beads", "bond_length_nm", "persistence_bonds", "dna_height_nm", "tip_radius_nm",
    "noise_rms_nm", "nm_per_pixel", "image_npy", "dna_mask_npy", "cross_mask_npy", "meta_npz",
    "n_crossings", "has_decoy", "afm_params_json",
]


def generate_dataset(
    contour_length_range_nm: Any,
    dna_diameter_range_nm: Any,
    tip_radius_range_nm: Any,
    noise_rms_range_nm: Any,
    noise_source: str,
    noise_asset: Any,
    export_preview_pngs: bool = False,
    preview_png_indices: Any = (),
    out_dir: str | None = None,
    n_samples: int | None = None,
    base_seed: int | None = None,
    timeout_s: float | None = None,
) -> str:
    """Generate the dataset and write ``manifest.csv``.

    For every sample, a random generator seeded with the current seed draws
    the contour length, DNA height ("diameter"), tip radius and noise RMS
    with :func:`draw_value`. Every ``DECOY_EVERY_N_SAMPLES``-th sample gets a
    decoy fragment. Seeds that fail (time-out, rejected MD run, molecule too
    large for the field of view) are skipped and the next seed is used, so
    exactly ``n_samples`` samples are written.

    Files written under ``out_dir``: ``images/img_XXXX.npy`` (height in nm),
    ``dna_masks/dna_XXXX.npy``, ``cross_masks/cross_XXXX.npy``,
    ``meta/meta_XXXX.npz`` (coordinates and parameters), optional
    ``preview_pngs/preview_XXXX.png`` and ``manifest.csv`` linking them.

    Parameters
    ----------
    contour_length_range_nm, dna_diameter_range_nm, tip_radius_range_nm, noise_rms_range_nm
        Number, list (discrete values) or ``(low, high)`` tuple.
    noise_source, noise_asset
        Output of :func:`load_noise_asset`.
    export_preview_pngs : bool, optional
        Write PNG previews. Defaults to ``False``.
    preview_png_indices : iterable of int, optional
        Dataset indices to preview.
    out_dir, n_samples, base_seed, timeout_s : optional
        Default to ``OUT_DIR``, ``N_SAMPLES``, ``BASE_SEED`` and
        ``SAMPLE_TIMEOUT_S``.

    Returns
    -------
    str
        Path of the manifest.
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
    afm_params = json.dumps(dict(AFM_KW, mask_width_nm=DNA_MASK_DIAMETER_NM), default=str)
    skipped: dict[str, int] = {}
    started = time.perf_counter()
    print(f"Starting dataset generation: {n_samples} samples "
          f"({'MD' if USE_MD else 'non-MD'} pathway) → {os.path.abspath(out_dir)}")

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
            add_decoy = bool(DECOY_EVERY_N_SAMPLES) and idx % int(DECOY_EVERY_N_SAMPLES) == 0
            decoy_range = tuple(DECOY_N_BEADS_RANGE_POOL[int(rng.integers(0, len(DECOY_N_BEADS_RANGE_POOL)))])
            try:
                sample = generate_sample_with_timeout(
                    noise_source, noise_asset, seed, contour, add_decoy, decoy_range,
                    height, tip, noise_rms, timeout_s=timeout_s)
            except Exception as exc:
                reason = type(exc).__name__
                skipped[reason] = skipped.get(reason, 0) + 1
                print(f"Skipping seed {seed} due to {reason}: {exc}")
                seed += 1
                continue

            paths = dict(
                image_npy=os.path.join("images", f"img_{idx:04d}.npy"),
                dna_mask_npy=os.path.join("dna_masks", f"dna_{idx:04d}.npy"),
                cross_mask_npy=os.path.join("cross_masks", f"cross_{idx:04d}.npy"),
                meta_npz=os.path.join("meta", f"meta_{idx:04d}.npz"),
            )
            np.save(os.path.join(out_dir, paths["image_npy"]), sample["afm_img"])
            np.save(os.path.join(out_dir, paths["dna_mask_npy"]), sample["dna_mask"])
            np.save(os.path.join(out_dir, paths["cross_mask_npy"]), sample["cross_mask"])
            np.savez_compressed(
                os.path.join(out_dir, paths["meta_npz"]),
                seed=sample["seed"], coords=sample["coords"], extent=sample["extent"],
                requested_contour_nm=sample["requested_contour_nm"],
                projected_contour_nm=sample["projected_contour_nm"],
                n_beads=sample["n_beads"], bond_length_nm=sample["bond_length_nm"],
                n_crossings=sample["n_crossings"], has_decoy=sample["has_decoy"],
                dna_height_nm=sample["dna_height_nm"], tip_radius_nm=sample["tip_radius_nm"],
                noise_rms_nm=sample["noise_rms_nm"], nm_per_pixel=sample["nm_per_pixel"],
            )
            writer.writerow([
                idx, sample["seed"], sample["pathway"], f"{sample['requested_contour_nm']:.4f}",
                f"{sample['projected_contour_nm']:.4f}", f"{sample['contour_3d_nm']:.4f}",
                sample["n_beads"], f"{sample['bond_length_nm']:.6f}", PERSISTENCE_BONDS,
                f"{sample['dna_height_nm']:.4f}", f"{sample['tip_radius_nm']:.4f}",
                f"{sample['noise_rms_nm']:.4f}", f"{sample['nm_per_pixel']:.6f}",
                paths["image_npy"], paths["dna_mask_npy"], paths["cross_mask_npy"], paths["meta_npz"],
                sample["n_crossings"], sample["has_decoy"], afm_params,
            ])
            handle.flush()
            if export_preview_pngs and idx in preview_indices:
                save_preview_png(sample, idx, preview_dir)
            if (idx + 1) % 10 == 0 or idx + 1 == n_samples:
                elapsed = time.perf_counter() - started
                print(f"Progress: {idx + 1}/{n_samples}  ({elapsed / 60:.1f} min elapsed)")
            idx += 1
            seed += 1

    reset_sample_executor()
    print("\nDone. Dataset written to:", os.path.abspath(out_dir))
    print("Manifest:", os.path.abspath(manifest_path))
    if skipped:
        print("Skipped seeds by reason:", skipped)
    return manifest_path
