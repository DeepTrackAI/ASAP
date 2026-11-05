from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from afm_simulation import AFMCantilever


def plot_maps(
    topography_m: np.ndarray,
    stiffness_map_pa: np.ndarray,
    contact_point_map_m: np.ndarray,
) -> None:
    """Plot topography, stiffness, and contact-point maps side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    im0 = axes[0].imshow(topography_m * 1e6, cmap="viridis", aspect="equal")
    axes[0].set_title("Topography [µm]")
    axes[0].set_xlabel("X [pixel]")
    axes[0].set_ylabel("Y [pixel]")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(stiffness_map_pa / 1e3, cmap="magma", aspect="equal")
    axes[1].set_title("Stiffness [kPa]")
    axes[1].set_xlabel("X [pixel]")
    axes[1].set_ylabel("Y [pixel]")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(contact_point_map_m * 1e6, cmap="cividis", 
                         aspect="equal")
    axes[2].set_title("Contact point [µm]")
    axes[2].set_xlabel("X [pixel]")
    axes[2].set_ylabel("Y [pixel]")
    fig.colorbar(im2, ax=axes[2], shrink=0.8)

    fig.tight_layout()
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(results_dir / "example_gaus.png")
    plt.show()


def plot_force_curve(
    displacement_curve_m: np.ndarray,
    force_curve_n: np.ndarray,
    retract_start_index: int | None = None,
    valid_length: int | None = None,
    title: str = "Force curve",
) -> None:
    """Plot approach and retract segments of one force curve."""
    if valid_length is None:
        valid_length = len(displacement_curve_m)
    if retract_start_index is None:
        retract_start_index = valid_length // 2

    displacement_curve_m = np.asarray(
        displacement_curve_m[:valid_length], dtype=np.float64
    )
    force_curve_n = np.asarray(force_curve_n[:valid_length], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        displacement_curve_m[:retract_start_index] * 1e6,
        force_curve_n[:retract_start_index] * 1e9,
        label="Approach",
    )
    ax.plot(
        displacement_curve_m[retract_start_index:] * 1e6,
        force_curve_n[retract_start_index:] * 1e9,
        label="Retract",
    )
    ax.set_xlabel("Displacement [µm]")
    ax.set_ylabel("Force [nN]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(results_dir / "fdc_ex.png")
    plt.show()


def plot_psd(
    frequency_hz: np.ndarray,
    spectral_density_m2_per_hz: np.ndarray,
    cantilever: AFMCantilever,
    sampling_frequency: float,
    title: str = "Displacement PSD",
) -> None:
    """Plot a square-root displacement PSD on log-log axes."""
    fig, ax = plt.subplots(figsize=(7, 4))
    in_band = frequency_hz <= (sampling_frequency / 2.0)
    ax.loglog(
        frequency_hz / 1e3,
        np.sqrt(spectral_density_m2_per_hz),
        linewidth=2.0,
        label="Full PSD",
    )
    ax.loglog(
        frequency_hz[in_band] / 1e3,
        np.sqrt(spectral_density_m2_per_hz[in_band]),
        linewidth=2.0,
        label="Within Nyquist",
    )
    ax.axvline(
        cantilever.f0 / 1e3,
        color="red",
        linestyle="--",
        label=f"f0 = {cantilever.f0 / 1e3:.1f} kHz",
    )
    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel("Sqrt(PSD) [m / sqrt(Hz)]")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(results_dir / "psd_ex.png")
    plt.show()
