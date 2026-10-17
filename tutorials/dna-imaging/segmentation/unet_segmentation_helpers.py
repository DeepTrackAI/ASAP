"""Helper functions for Part 2 of ``DNA_simulation_and_segmentation.ipynb``.

Part 2 trains (or loads) a two-headed U-Net that segments DNA (channel 0) and
highlights DNA crossings (channel 1) in AFM height maps, evaluates it on the
simulated data, and applies it to real data, optionally inside TopoStats.

This module holds every function used by Part 2: reading the dataset
manifest into DeepTrack sources, the lazy loading pipeline, the building
blocks of the loss, checkpoint input/output, training curves, validation
metrics, prediction previews, real-data loading and evaluation, and running
TopoStats with the trained network. Class definitions (dataset, loss, metric,
networks, TopoStats predictor) stay in the notebook.

Preprocessing matches the shipped checkpoints: heights in nm are clipped to
``NORMALIZATION_BOUNDS_NM`` and scaled to 0-1, then resized (aspect ratio
preserved, bilinear) and centre-padded to ``TARGET_SIZE`` pixels. Network
outputs are restored to the native pixels before thresholding.
"""

from __future__ import annotations

import logging
import random
import sys
import types
from importlib import resources
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from scipy import ndimage as ndi

TARGET_SIZE = 512
NORMALIZATION_BOUNDS_NM = (-1.0, 5.0)

CHECKPOINT_FILES = {
    ("md", "residual"): "md_res_unet.pt",
    ("md", "simple"): "md_simple_unet.pt",
    ("non_md", "residual"): "non_md_res_unet.pt",
    ("non_md", "simple"): "non_md_simple_unet.pt",
}


# ─────────────────────────────────────────────────────────────────────────────
# Manifest → DeepTrack sources
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_COLUMNS = ("image_npy", "image_path", "img_path", "afm_path")
DNA_COLUMNS = ("dna_mask_npy", "dna_path", "mask_path", "label_path")
CROSS_COLUMNS = ("cross_mask_npy", "cross_path", "crossing_path")


def find_column(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
    required: bool = True,
) -> str | None:
    """Return the first matching column in a manifest.

    Iterates through ``candidates`` in order and returns the name of the
    first one present in ``frame``. This allows several naming conventions
    for manifests made outside this notebook.

    Parameters
    ----------
    frame : pd.DataFrame
        Manifest whose columns are searched.
    candidates : tuple[str, ...]
        Ordered column names to try (``IMAGE_COLUMNS``, ``DNA_COLUMNS`` or
        ``CROSS_COLUMNS``).
    required : bool, optional
        Raise ``KeyError`` when nothing matches. Defaults to ``True``.

    Returns
    -------
    str | None
        The matching column, or ``None`` if optional and missing.

    Raises
    ------
    KeyError
        If ``required`` is ``True`` and no candidate is present.
    """

    for column in candidates:
        if column in frame.columns:
            return column
    if required:
        raise KeyError(f"Manifest is missing one of: {', '.join(candidates)}")
    return None


def resolve_data_path(value: Any, root: Path) -> Path:
    """Resolve an absolute path, or a path relative to the dataset folder.

    Parameters
    ----------
    value : Any
        Path as read from the manifest.
    root : Path
        Folder that relative paths are relative to (the manifest's folder).

    Returns
    -------
    Path
        The resolved path.
    """

    path = Path(str(value))
    return path if path.is_absolute() else Path(root) / path


def manifest_to_records(
    frame: pd.DataFrame,
    root: Path,
) -> list[dict[str, str]]:
    """Convert a dataset manifest into DeepTrack source rows.

    Finds the image, DNA mask and (optional) crossing mask columns, resolves
    the paths relative to ``root`` and checks that the files exist. A
    ``split`` column (``train`` / ``validation``), if present, is kept.

    Parameters
    ----------
    frame : pd.DataFrame
        The manifest.
    root : Path
        The dataset folder.

    Returns
    -------
    list[dict[str, str]]
        One row per sample with keys ``id``, ``image_path``, ``dna_path``,
        ``cross_path`` (empty when no crossing mask exists) and ``split``.

    Raises
    ------
    FileNotFoundError
        If required files are missing (the first ten are listed).
    RuntimeError
        If no valid sample was found.
    """

    image_col = find_column(frame, IMAGE_COLUMNS)
    dna_col = find_column(frame, DNA_COLUMNS)
    cross_col = find_column(frame, CROSS_COLUMNS, required=False)
    records, missing = [], []
    for row_index, row in frame.iterrows():
        image_path = resolve_data_path(row[image_col], root)
        dna_path = resolve_data_path(row[dna_col], root)
        cross_path = ""
        if cross_col is not None and not pd.isna(row[cross_col]):
            cross_path = str(resolve_data_path(row[cross_col], root))
        if not image_path.exists():
            missing.append(str(image_path))
            continue
        if not dna_path.exists():
            missing.append(str(dna_path))
            continue
        if cross_path and not Path(cross_path).exists():
            missing.append(cross_path)
            cross_path = ""
        records.append(dict(
            id=str(row.get("index", row_index)),
            image_path=str(image_path),
            dna_path=str(dna_path),
            cross_path=cross_path,
            split=str(row.get("split", "")) if "split" in frame.columns else "",
        ))
    if missing:
        raise FileNotFoundError("Some manifest files could not be found. First missing files:\n"
                                + "\n".join(missing[:10]))
    if not records:
        raise RuntimeError("No valid samples were found in the manifest.")
    return records


def split_records(
    records: list[dict[str, str]],
    val_fraction: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split records into training and validation lists.

    If the manifest already assigns ``train`` / ``validation`` in a ``split``
    column, that assignment is used. Otherwise a random split with
    ``val_fraction`` of the samples in validation is drawn with ``seed``.

    Parameters
    ----------
    records : list[dict[str, str]]
        Output of :func:`manifest_to_records`.
    val_fraction : float
        Fraction of samples used for validation (random split only).
    seed : int
        Random seed for the split.

    Returns
    -------
    tuple[list, list]
        ``(train_records, val_records)``.
    """

    if records and all(r.get("split") in ("train", "validation", "test") for r in records):
        return ([r for r in records if r["split"] == "train"],
                [r for r in records if r["split"] == "validation"])
    if len(records) < 2:
        return records, records
    order = np.random.default_rng(seed).permutation(len(records))
    n_val = min(max(1, int(round(len(records) * val_fraction))), len(records) - 1)
    val_index = set(int(i) for i in order[:n_val])
    train_rows = [r for i, r in enumerate(records) if i not in val_index]
    val_rows = [r for i, r in enumerate(records) if i in val_index]
    return train_rows, val_rows


# ─────────────────────────────────────────────────────────────────────────────
# Lazy loading pipeline
# ─────────────────────────────────────────────────────────────────────────────


def load_array(path: str | Path) -> np.ndarray:
    """Load a ``.npy`` array or an image file as a 2-D float32 array.

    Colour images are reduced to one channel by averaging RGB.

    Parameters
    ----------
    path : str | Path
        File to load.

    Returns
    -------
    np.ndarray
        2-D float32 array.
    """

    path = Path(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    else:
        with Image.open(path) as image:
            array = np.asarray(image)
    array = np.asarray(array)
    if array.ndim == 3:
        array = array[..., :3].mean(axis=-1)
    return np.asarray(np.squeeze(array), dtype=np.float32)


def normalize_height(
    image: np.ndarray,
    lower: float = NORMALIZATION_BOUNDS_NM[0],
    upper: float = NORMALIZATION_BOUNDS_NM[1],
) -> np.ndarray:
    """Clip a height map (nm) to ``[lower, upper]`` and scale it to 0-1.

    Fixed physical bounds (rather than per-image percentiles) keep a DNA
    strand of a given height at the same input value in every image.

    Parameters
    ----------
    image : np.ndarray
        Height map in nm.
    lower, upper : float, optional
        Normalisation bounds in nm. Default to ``NORMALIZATION_BOUNDS_NM``.

    Returns
    -------
    np.ndarray
        Normalised float32 array.
    """

    image = np.nan_to_num(np.asarray(image, dtype=np.float32))
    if upper <= lower:
        raise ValueError("Invalid normalization bounds.")
    return ((np.clip(image, lower, upper) - lower) / (upper - lower)).astype(np.float32)


def resize_and_pad_np(
    array: np.ndarray,
    target_size: int,
    mode: str,
    fill: float = 0.0,
) -> np.ndarray:
    """Resize an array isotropically and centre it on a square canvas.

    The longer side is scaled to ``target_size`` (aspect ratio preserved) and
    the rest is padded with ``fill``.

    Parameters
    ----------
    array : np.ndarray
        2-D array.
    target_size : int
        Side of the output.
    mode : str
        ``"bilinear"`` for images/soft targets, ``"nearest"`` for binary masks.
    fill : float, optional
        Padding value. Defaults to 0.

    Returns
    -------
    np.ndarray
        Array of shape ``(target_size, target_size)``.
    """

    array = np.ascontiguousarray(np.nan_to_num(np.squeeze(np.asarray(array, dtype=np.float32))))
    height, width = array.shape[-2:]
    scale = min(target_size / height, target_size / width)
    new_h, new_w = max(1, int(round(height * scale))), max(1, int(round(width * scale)))
    kwargs: dict[str, Any] = dict(size=(new_h, new_w), mode=mode)
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    resized = F.interpolate(torch.from_numpy(array)[None, None], **kwargs)[0, 0].numpy()
    canvas = np.full((target_size, target_size), fill, dtype=np.float32)
    top, left = (target_size - new_h) // 2, (target_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return np.ascontiguousarray(canvas)


def prepare_height(
    image: np.ndarray,
    target_size: int = TARGET_SIZE,
    lower: float = NORMALIZATION_BOUNDS_NM[0],
    upper: float = NORMALIZATION_BOUNDS_NM[1],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Normalise, resize and pad one height map for the network.

    Returns the ``(1, 1, target_size, target_size)`` input tensor and the
    geometry needed by :func:`restore_probability` to map outputs back to the
    native pixels.

    Parameters
    ----------
    image : np.ndarray
        Height map in nm.
    target_size : int, optional
        Network input size. Defaults to ``TARGET_SIZE``.
    lower, upper : float, optional
        Normalisation bounds in nm.

    Returns
    -------
    tuple[torch.Tensor, dict]
        Input tensor and geometry dictionary.
    """

    image = normalize_height(image, lower, upper)
    height, width = image.shape
    scale = min(target_size / height, target_size / width)
    new_h, new_w = max(1, round(height * scale)), max(1, round(width * scale))
    top, left = (target_size - new_h) // 2, (target_size - new_w) // 2
    resized = F.interpolate(torch.from_numpy(np.ascontiguousarray(image))[None, None],
                            size=(new_h, new_w), mode="bilinear", align_corners=False)
    background = float(np.clip(-lower / (upper - lower), 0, 1))
    canvas = torch.full((1, 1, target_size, target_size), background, dtype=torch.float32)
    canvas[..., top:top + new_h, left:left + new_w] = resized
    return canvas, dict(original_shape=(height, width), top=top, left=left,
                        resized_shape=(new_h, new_w), target_size=target_size)


def restore_probability(probability: np.ndarray, geometry: dict[str, Any]) -> np.ndarray:
    """Map a network probability map back to the native image pixels.

    Removes the padding added by :func:`prepare_height` and resizes the
    continuous probabilities (before any thresholding) to the original shape.
    """

    top, left = geometry["top"], geometry["left"]
    height, width = geometry["resized_shape"]
    unpadded = np.asarray(probability, dtype=np.float32)[top:top + height, left:left + width]
    restored = F.interpolate(torch.from_numpy(np.ascontiguousarray(unpadded))[None, None],
                             size=geometry["original_shape"], mode="bilinear", align_corners=False)
    return restored[0, 0].numpy().clip(0, 1)


def crop_to_molecule(
    arrays: list[np.ndarray],
    reference_mask: np.ndarray,
    padding_px: int,
) -> list[np.ndarray]:
    """Crop arrays to the bounding box of ``reference_mask`` plus padding.

    This mimics a TopoStats grain crop: the network then sees the molecule at
    roughly the same scale as the shipped checkpoints were trained on.

    Parameters
    ----------
    arrays : list[np.ndarray]
        Arrays with the same shape as ``reference_mask``.
    reference_mask : np.ndarray
        Mask whose non-zero pixels define the molecule.
    padding_px : int
        Margin added on every side (clipped at the image border).

    Returns
    -------
    list[np.ndarray]
        Cropped arrays (unchanged if the mask is empty).
    """

    rows, cols = np.nonzero(reference_mask > 0.5)
    if rows.size == 0:
        return arrays
    r0 = max(0, int(rows.min()) - padding_px)
    r1 = min(reference_mask.shape[0], int(rows.max()) + padding_px + 1)
    c0 = max(0, int(cols.min()) - padding_px)
    c1 = min(reference_mask.shape[1], int(cols.max()) + padding_px + 1)
    return [np.ascontiguousarray(a[r0:r1, c0:c1]) for a in arrays]


def augment_triplet(
    image: np.ndarray,
    dna: np.ndarray,
    cross: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the same random flips and 90° rotation to image and both targets."""

    if random.random() < 0.5:
        image, dna, cross = np.fliplr(image), np.fliplr(dna), np.fliplr(cross)
    if random.random() < 0.5:
        image, dna, cross = np.flipud(image), np.flipud(dna), np.flipud(cross)
    k = random.randint(0, 3)
    if k:
        image, dna, cross = np.rot90(image, k), np.rot90(dna, k), np.rot90(cross, k)
    return np.ascontiguousarray(image), np.ascontiguousarray(dna), np.ascontiguousarray(cross)


def load_sample_factory(
    target_size: int = TARGET_SIZE,
    augment: bool = False,
    bounds: tuple[float, float] = NORMALIZATION_BOUNDS_NM,
    crop_padding_px: int | tuple[int, int] | None = None,
):
    """Return the function that loads one manifest row (used by DeepTrack).

    The returned ``load_sample(sample)`` reads the image and masks from the
    paths in the row, optionally crops them to the molecule
    (:func:`crop_to_molecule`; a ``(low, high)`` padding is drawn at random
    per sample), optionally augments them, normalises the image and resizes
    everything to ``target_size``.

    Parameters
    ----------
    target_size : int, optional
        Network input size.
    augment : bool, optional
        Random flips/rotations. Defaults to ``False``.
    bounds : tuple[float, float], optional
        Height normalisation bounds in nm.
    crop_padding_px : int | tuple[int, int] | None, optional
        Crop padding in pixels; ``None`` keeps the full image.

    Returns
    -------
    callable
        ``load_sample(sample: dict) -> (image (1, H, W), target (2, H, W))``.
    """

    lower, upper = bounds
    background = float(np.clip(-lower / (upper - lower), 0, 1))

    def load_sample(sample: dict[str, str]):
        image = np.nan_to_num(load_array(sample["image_path"])).astype(np.float32)
        dna = (load_array(sample["dna_path"]) > 0.5).astype(np.float32)
        if sample.get("cross_path"):
            cross = np.nan_to_num(load_array(sample["cross_path"]))
            if float(cross.max()) > 1.0:
                cross = cross / float(cross.max())
            cross = np.clip(cross, 0.0, 1.0).astype(np.float32)
        else:
            cross = np.zeros_like(dna, dtype=np.float32)

        if crop_padding_px is not None:
            if isinstance(crop_padding_px, tuple):
                padding = random.randint(int(crop_padding_px[0]), int(crop_padding_px[1]))
            else:
                padding = int(crop_padding_px)
            image, dna, cross = crop_to_molecule([image, dna, cross], dna, padding)
        if augment:
            image, dna, cross = augment_triplet(image, dna, cross)

        image = resize_and_pad_np(normalize_height(image, lower, upper), target_size, "bilinear",
                                  fill=background)
        dna = (resize_and_pad_np(dna, target_size, "nearest") > 0.5).astype(np.float32)
        cross = np.clip(resize_and_pad_np(cross, target_size, "bilinear"), 0.0, 1.0)
        return (np.ascontiguousarray(image[None]).astype(np.float32),
                np.ascontiguousarray(np.stack([dna, cross]).astype(np.float32)))

    return load_sample


def build_pipeline(
    source,
    target_size: int,
    augment: bool,
    bounds: tuple[float, float] = NORMALIZATION_BOUNDS_NM,
    crop_padding_px: int | tuple[int, int] | None = None,
):
    """Create the lazy DeepTrack pipeline ``Value(row) >> Lambda(load_sample)``.

    The source only stores file paths; arrays are read when an item is
    requested, so the dataset never has to fit in memory.

    Parameters
    ----------
    source : deeptrack.sources.Source
        Source whose ``.samples`` are manifest rows.
    target_size : int
        Network input size.
    augment : bool
        Random flips/rotations.
    bounds : tuple[float, float], optional
        Height normalisation bounds in nm.
    crop_padding_px : int | tuple[int, int] | None, optional
        See :func:`load_sample_factory`.

    Returns
    -------
    deeptrack.Feature
        The lazy pipeline.
    """

    import deeptrack as dt

    sample_value = dt.Value(source.samples)
    loader = dt.Lambda(
        function=load_sample_factory,
        target_size=target_size,
        augment=augment,
        bounds=bounds,
        crop_padding_px=crop_padding_px,
    )
    return sample_value >> loader


# ─────────────────────────────────────────────────────────────────────────────
# Loss building blocks and scores
# ─────────────────────────────────────────────────────────────────────────────


def soft_erode(x: torch.Tensor) -> torch.Tensor:
    """Differentiable erosion (min-pooling along rows and columns)."""

    return torch.minimum(-F.max_pool2d(-x, (3, 1), (1, 1), (1, 0)),
                         -F.max_pool2d(-x, (1, 3), (1, 1), (0, 1)))


def soft_open(x: torch.Tensor) -> torch.Tensor:
    """Differentiable morphological opening (erosion followed by dilation)."""

    return F.max_pool2d(soft_erode(x), 3, 1, 1)


def soft_skeletonize(x: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    """Differentiable skeleton of a probability map (used by clDice).

    Repeatedly erodes the map and keeps what opening removes, which leaves a
    thin centre line that still carries gradients.
    """

    skeleton = F.relu(x - soft_open(x))
    for _ in range(iterations):
        x = soft_erode(x)
        delta = F.relu(x - soft_open(x))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def mean_per_sample(x: torch.Tensor) -> torch.Tensor:
    """Mean over all but the batch dimension."""

    return x.flatten(1).mean(1)


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sample mean of ``value`` over the pixels where ``mask`` is 1."""

    return (value * mask).flatten(1).sum(1) / (mask.flatten(1).sum(1) + 1e-6)


def dice_score_from_prob(
    prob: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean hard Dice score of thresholded probabilities against a binary mask.

    Dice = 2|P ∩ T| / (|P| + |T|).
    """

    pred = (prob >= threshold).float()
    dims = tuple(range(1, pred.ndim))
    inter = torch.sum(pred * target, dim=dims)
    denom = torch.sum(pred, dim=dims) + torch.sum(target, dim=dims)
    return ((2.0 * inter + eps) / (denom + eps)).mean()


def iou_score_from_prob(
    prob: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean hard IoU of thresholded probabilities against a binary mask.

    IoU = |P ∩ T| / |P ∪ T|.
    """

    pred = (prob >= threshold).float()
    dims = tuple(range(1, pred.ndim))
    inter = torch.sum(pred * target, dim=dims)
    union = torch.sum((pred + target) > 0, dim=dims)
    return ((inter + eps) / (union + eps)).mean()


def binary_scores(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Dice, IoU, precision and recall of two binary masks (NumPy)."""

    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    inter = float(np.logical_and(pred, truth).sum())
    p, t = float(pred.sum()), float(truth.sum())
    union = float(np.logical_or(pred, truth).sum())
    return dict(dice=(2 * inter + 1e-6) / (p + t + 1e-6), iou=(inter + 1e-6) / (union + 1e-6),
                precision=inter / max(p, 1.0), recall=inter / max(t, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Networks: small helpers used by the U-Net classes in the notebook
# ─────────────────────────────────────────────────────────────────────────────


def valid_group_count(num_channels: int, max_groups: int = 8) -> int:
    """Largest GroupNorm group count (≤ ``max_groups``) that divides ``num_channels``."""

    for groups in (int(max_groups), 8, 4, 2, 1):
        if groups <= num_channels and num_channels % groups == 0:
            return groups
    return 1


def count_parameters(model: torch.nn.Module) -> int:
    """Number of trainable parameters of a network."""

    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoints and training curves
# ─────────────────────────────────────────────────────────────────────────────


def checkpoint_for(
    use_md: bool,
    architecture: str,
    checkpoint_dir: str | Path = "checkpoints",
) -> Path:
    """Return the shipped checkpoint for a pathway and architecture.

    Parameters
    ----------
    use_md : bool
        ``True`` for the MD-trained models, ``False`` for non-MD.
    architecture : str
        ``"residual"`` (Res-U-Net) or ``"simple"`` (simple U-Net).
    checkpoint_dir : str | Path, optional
        Folder holding the checkpoints. Defaults to ``"checkpoints"``.

    Returns
    -------
    Path
        Path to the checkpoint file.
    """

    key = ("md" if use_md else "non_md", architecture)
    if key not in CHECKPOINT_FILES:
        raise ValueError(f"architecture must be 'residual' or 'simple', got {architecture!r}")
    return Path(checkpoint_dir) / CHECKPOINT_FILES[key]


def load_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load weights into ``model`` and return the checkpoint's metadata/history.

    Accepts the checkpoints shipped with this tutorial
    (``{"state_dict", "metadata", "history"}``), plain state dicts, and
    state dicts saved from a Deeplay/Lightning application (keys prefixed
    with ``model.``).

    Parameters
    ----------
    model : torch.nn.Module
        Network with the matching architecture.
    path : str | Path
        Checkpoint file.
    device : str | torch.device, optional
        Device to move the model to. Defaults to ``"cpu"``.

    Returns
    -------
    dict[str, Any]
        The checkpoint payload without the weights.
    """

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if any(key.startswith("model.") for key in state):
        state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return {k: v for k, v in payload.items() if k != "state_dict"} if isinstance(payload, dict) else {}


def save_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> Path:
    """Save a network in the same format as the shipped checkpoints."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(dict(state_dict=state, metadata=metadata or {}, history=history or []), path)
    return path


def pick_precision() -> tuple[str, str]:
    """Return the accelerator name and a suitable Lightning precision setting.

    Mixed precision (``"16-mixed"``) on CUDA GPUs, full precision otherwise.
    """

    if torch.cuda.is_available():
        return f"CUDA GPU: {torch.cuda.get_device_name(0)}", "16-mixed"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "Apple Silicon MPS", "32-true"
    return "CPU", "32-true"


def get_latest_metrics_path(trainer) -> Path:
    """Return the ``metrics.csv`` written by the trainer's CSV logger.

    Raises
    ------
    RuntimeError
        If the trainer has no logger with a log directory.
    FileNotFoundError
        If the CSV does not exist.
    """

    if trainer.logger is None or getattr(trainer.logger, "log_dir", None) is None:
        raise RuntimeError("The trainer does not have a CSV logger.")
    metrics_path = Path(trainer.logger.log_dir) / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Could not find the metrics CSV: {metrics_path}")
    return metrics_path


def history_from_lightning_csv(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a Lightning ``metrics.csv`` into the checkpoint history format."""

    metrics = metrics.sort_values("epoch").groupby("epoch", as_index=False).last()
    rows = []
    for _, r in metrics.iterrows():
        def pick(prefix, name):
            for column in (f"{prefix}_{name}_epoch", f"{prefix}{name}_epoch", f"{prefix}_{name}"):
                if column in metrics.columns and not pd.isna(r[column]):
                    return float(r[column])
            return float("nan")
        iou_cols = [c for c in metrics.columns if "iou" in c.lower()]
        rows.append(dict(
            epoch=int(r["epoch"]) + 1,
            train=dict(loss=pick("train", "loss"),
                       iou=next((float(r[c]) for c in iou_cols if c.startswith("train") and not pd.isna(r[c])), float("nan"))),
            validation=dict(loss=pick("val", "loss"),
                            iou=next((float(r[c]) for c in iou_cols if c.startswith("val") and not pd.isna(r[c])), float("nan"))),
        ))
    return rows


def plot_training_history(history: list[dict[str, Any]], title: str = "") -> None:
    """Plot training/validation loss and DNA IoU (and Dice if logged) per epoch.

    Parameters
    ----------
    history : list[dict]
        Rows with ``epoch``, ``train`` and ``validation`` dictionaries
        (checkpoint history or :func:`history_from_lightning_csv`).
    title : str, optional
        Figure title.
    """

    if not history:
        print("No training history available.")
        return
    epochs = [r["epoch"] for r in history]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for split, style in (("train", "-"), ("validation", "--")):
        axs[0].plot(epochs, [r[split].get("loss", np.nan) for r in history], style, label=split)
        axs[1].plot(epochs, [r[split].get("iou", np.nan) for r in history], style, label=f"{split} IoU")
        if any("dice" in r[split] for r in history):
            axs[1].plot(epochs, [r[split].get("dice", np.nan) for r in history], style, alpha=0.6,
                        label=f"{split} Dice")
    best = [r["epoch"] for r in history if r.get("selected_best")]
    for ax in axs:
        if best:
            ax.axvline(best[-1], color="gray", linewidth=0.8, linestyle=":", label="selected epoch")
        ax.set_xlabel("Epoch")
        ax.legend()
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Loss")
    axs[1].set_ylabel("Score")
    axs[1].set_title("DNA overlap scores")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    show_figure(plt.gcf())


# ─────────────────────────────────────────────────────────────────────────────
# Validation metrics and previews
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_loader(
    model: torch.nn.Module,
    loader,
    loss_fn,
    stage: str = "validation",
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate loss, DNA Dice, DNA IoU and crossing MSE over a data loader.

    Parameters
    ----------
    model : torch.nn.Module
        Network returning DNA (channel 0) and crossing (channel 1) logits.
    loader : DataLoader
        Yields ``(image, target)`` batches.
    loss_fn : callable
        The training loss.
    stage : str, optional
        Label used when printing.
    threshold : float, optional
        DNA probability threshold. Defaults to 0.5.

    Returns
    -------
    dict[str, float]
        Batch-size-weighted means of ``loss``, ``dna_dice``, ``dna_iou`` and
        ``cross_mse``.
    """

    model.eval()
    device = next(model.parameters()).device
    totals = dict(loss=0.0, dna_dice=0.0, dna_iou=0.0, cross_mse=0.0)
    count = 0
    with torch.no_grad():
        for image, target in loader:
            image, target = image.to(device), target.to(device)
            pred = model(image)
            dna_prob = torch.sigmoid(pred[:, 0:1])
            cross_prob = torch.sigmoid(pred[:, 1:2])
            n = int(image.shape[0])
            totals["loss"] += float(loss_fn(pred, target)) * n
            totals["dna_dice"] += float(dice_score_from_prob(dna_prob, target[:, 0:1], threshold)) * n
            totals["dna_iou"] += float(iou_score_from_prob(dna_prob, target[:, 0:1], threshold)) * n
            totals["cross_mse"] += float(F.mse_loss(cross_prob, target[:, 1:2])) * n
            count += n
    metrics = {key: value / max(count, 1) for key, value in totals.items()}
    print(stage, {k: round(v, 4) for k, v in metrics.items()})
    return metrics


def inspect_crossing_batch(batch, model: torch.nn.Module) -> None:
    """Print how strongly the crossing head responds at crossings vs elsewhere."""

    image, target = batch
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        cross_prob = torch.sigmoid(model(image.to(device))[:, 1:2]).cpu()
    target_cross, target_dna = target[:, 1:2].cpu(), target[:, 0:1].cpu()
    positive = target_cross > 0.05
    dna_region = target_dna > 0.6
    print("Crossing target max:", round(float(target_cross.max()), 3))
    print("Crossing prediction max:", round(float(cross_prob.max()), 3))
    print("Crossing pixels in batch:", int(positive.sum()), "| DNA pixels:", int(dna_region.sum()))
    if int(positive.sum()):
        print("Mean prediction at crossings:", round(float(cross_prob[positive].mean()), 3))
    if int((dna_region & ~positive).sum()):
        print("Mean prediction on non-crossing DNA:", round(float(cross_prob[dna_region & ~positive].mean()), 3))
    if int((~dna_region).sum()):
        print("Mean prediction on background:", round(float(cross_prob[~dna_region].mean()), 3))


def show_prediction(batch, model: torch.nn.Module, index: int = 0) -> None:
    """Show image, DNA target/prediction and crossing target/prediction for one sample."""

    image, target = batch
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        pred = model(image.to(device))
        dna_prob = torch.sigmoid(pred[:, 0:1]).cpu()
        cross_prob = torch.sigmoid(pred[:, 1:2]).cpu()
    panels = [image[index, 0].cpu(), target[index, 0].cpu(), dna_prob[index, 0],
              target[index, 1].cpu(), cross_prob[index, 0]]
    titles = ["input (normalised)", "DNA target", "DNA prediction", "crossing target", "crossing prediction"]
    fig, axes = plt.subplots(1, len(panels), figsize=(17, 3.6))
    for ax, panel, title in zip(axes, panels, titles):
        if title.startswith("input"):
            ax.imshow(panel, cmap="afmhot", vmin=float(torch.quantile(panel.float(), 0.01)),
                      vmax=float(torch.quantile(panel.float(), 0.995)))
        else:
            ax.imshow(panel, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    show_figure(plt.gcf())


# ─────────────────────────────────────────────────────────────────────────────
# Real data
# ─────────────────────────────────────────────────────────────────────────────

DATA_TYPES = {
    "npy": (".npy",),
    "image": (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    "spm": (".spm",),
}
MASK_WORDS = ("mask", "label", "gt", "annotation", "seg")


def normalized_stem(path: str | Path) -> str:
    """Return a file stem with image/mask prefixes and suffixes removed.

    ``image_12.npy`` and ``mask_12.npy`` both give ``"12"``, so images and
    their masks can be paired by name.
    """

    stem = Path(path).stem.lower()
    for word in ("image", "img", "height") + MASK_WORDS:
        stem = stem.replace(f"{word}_", "").replace(f"_{word}", "").replace(word, "")
    return stem.strip("_- ")


def collect_real_pairs(root: str | Path, data_type: str) -> list[dict[str, Any]]:
    """Find images (and matching masks, if any) of one type in a folder.

    Files whose name contains a mask word (``mask``, ``label``, ``gt``, ...)
    are treated as masks and paired with the image of the same
    :func:`normalized_stem`.

    Parameters
    ----------
    root : str | Path
        Folder to search (recursively).
    data_type : str
        ``"npy"``, ``"image"`` or ``"spm"``.

    Returns
    -------
    list[dict[str, Any]]
        Rows with ``id``, ``image_path`` and ``mask_path`` (``None`` if absent).
    """

    if data_type not in DATA_TYPES:
        raise ValueError(f"data_type must be one of {list(DATA_TYPES)}")
    files = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in DATA_TYPES[data_type])
    masks = {normalized_stem(p): p for p in files if any(w in p.stem.lower() for w in MASK_WORDS)}
    images = [p for p in files if not any(w in p.stem.lower() for w in MASK_WORDS)]
    return [dict(id=normalized_stem(p), image_path=p, mask_path=masks.get(normalized_stem(p)))
            for p in images]


def load_real_height(
    path: str | Path,
    data_type: str,
    image_height_range_nm: tuple[float, float] = (0.0, 4.0),
) -> np.ndarray:
    """Load a real height map in nm.

    ``.npy`` files are assumed to contain heights in nm. Image files contain
    grey values, which are mapped linearly from their 0-255 (or 0-65535)
    range onto ``image_height_range_nm``.

    Parameters
    ----------
    path : str | Path
        File to load.
    data_type : str
        ``"npy"`` or ``"image"`` (use :func:`run_topostats` for ``.spm``).
    image_height_range_nm : tuple[float, float], optional
        Height range represented by black and white in image files.

    Returns
    -------
    np.ndarray
        Height map in nm (float32).
    """

    array = load_array(path)
    if data_type == "image":
        top = 65535.0 if array.max() > 255 else 255.0
        low, high = image_height_range_nm
        array = low + (array / top) * (high - low)
    return np.nan_to_num(array).astype(np.float32)


def predict_native(
    model: torch.nn.Module,
    height_nm: np.ndarray,
    bounds: tuple[float, float] = NORMALIZATION_BOUNDS_NM,
    target_size: int = TARGET_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the network on one height map and return native-size probabilities.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        DNA and crossing probability maps with the shape of ``height_nm``.
    """

    device = next(model.parameters()).device
    tensor, geometry = prepare_height(height_nm, target_size, *bounds)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor.to(device)))[0].cpu().numpy()
    return restore_probability(prob[0], geometry), restore_probability(prob[1], geometry)


def measure_mask(mask: np.ndarray, nm_per_pixel: float | None = None) -> dict[str, float]:
    """Basic measurements of a binary DNA mask.

    The skeleton length is the number of skeleton pixels, counting diagonal
    steps as √2; it is a quick estimate of the total DNA length in the mask,
    not a TopoStats contour length.

    Returns
    -------
    dict[str, float]
        ``components``, ``area_px``, ``skeleton_length_px`` and, if the pixel
        size is known, ``skeleton_length_nm``.
    """

    from skimage.morphology import skeletonize

    mask = np.asarray(mask, bool)
    _, components = ndi.label(mask, structure=np.ones((3, 3)))
    skeleton = skeletonize(mask)
    straight = (np.logical_and(skeleton[:, 1:], skeleton[:, :-1]).sum()
                + np.logical_and(skeleton[1:, :], skeleton[:-1, :]).sum())
    diagonal = (np.logical_and(skeleton[1:, 1:], skeleton[:-1, :-1]).sum()
                + np.logical_and(skeleton[1:, :-1], skeleton[:-1, 1:]).sum())
    length_px = float(straight + np.sqrt(2.0) * diagonal)
    result = dict(components=int(components), area_px=int(mask.sum()), skeleton_length_px=length_px)
    if nm_per_pixel:
        result["skeleton_length_nm"] = length_px * float(nm_per_pixel)
    return result


def evaluate_real_dataset(
    model: torch.nn.Module,
    pairs: list[dict[str, Any]],
    data_type: str,
    threshold: float = 0.5,
    nm_per_pixel: float | None = None,
    crop_padding_px: int | None = 40,
    image_height_range_nm: tuple[float, float] = (0.0, 4.0),
) -> pd.DataFrame:
    """Segment every image of a real dataset and score it against its mask.

    Images with a mask are cropped to the molecule (mask bounding box plus
    ``crop_padding_px``, like a TopoStats grain crop) before segmentation;
    use ``crop_padding_px=None`` to segment the full image. Images without a
    mask are segmented and measured but not scored.

    Returns
    -------
    pd.DataFrame
        One row per image with Dice, IoU, precision, recall and the
        predicted/true skeleton lengths.
    """

    rows = []
    for pair in pairs:
        height = load_real_height(pair["image_path"], data_type, image_height_range_nm)
        truth = None
        if pair.get("mask_path") is not None:
            truth = load_array(pair["mask_path"]) > 0.5
            if crop_padding_px is not None:
                height, truth_f = crop_to_molecule([height, truth.astype(np.float32)], truth, crop_padding_px)
                truth = truth_f > 0.5
        dna_prob, _ = predict_native(model, height)
        pred = dna_prob > threshold
        row = dict(id=pair["id"], file=Path(pair["image_path"]).name, shape=f"{height.shape[0]}x{height.shape[1]}")
        measured = measure_mask(pred, nm_per_pixel)
        row["pred_components"] = measured["components"]
        row["pred_length"] = measured.get("skeleton_length_nm", measured["skeleton_length_px"])
        if truth is not None:
            row.update(binary_scores(pred, truth))
            true_measure = measure_mask(truth, nm_per_pixel)
            row["true_length"] = true_measure.get("skeleton_length_nm", true_measure["skeleton_length_px"])
        rows.append(row)
    return pd.DataFrame(rows)


def show_real_dataset_results(
    results: pd.DataFrame,
    model: torch.nn.Module,
    pairs: list[dict[str, Any]],
    data_type: str,
    threshold: float = 0.5,
    crop_padding_px: int | None = 40,
    n_examples: int = 4,
    length_unit: str = "nm",
    image_height_range_nm: tuple[float, float] = (0.0, 4.0),
) -> None:
    """Print summary metrics and show example segmentations of a real dataset."""

    scored = results.dropna(subset=["dice"]) if "dice" in results else results.iloc[0:0]
    if len(scored):
        summary = scored[["dice", "iou", "precision", "recall"]].agg(["mean", "std", "min", "max"])
        print(f"Scored images: {len(scored)} of {len(results)}")
        display_df(summary.round(3))
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
        scored[["dice", "iou", "precision", "recall"]].plot.box(ax=axes[0])
        axes[0].set_title("Per-image scores")
        axes[0].set_ylim(0, 1.02)
        axes[1].scatter(scored["true_length"], scored["pred_length"])
        lim = [0, float(max(scored["true_length"].max(), scored["pred_length"].max()) * 1.1)]
        axes[1].plot(lim, lim, "k--", linewidth=0.8)
        axes[1].set_xlabel(f"Skeleton length of the true mask ({length_unit})")
        axes[1].set_ylabel(f"Skeleton length of the prediction ({length_unit})")
        axes[1].set_title("Predicted vs true DNA length")
        plt.tight_layout()
        show_figure(plt.gcf())
    display_df(results.round(3))

    examples = pairs[:n_examples]
    fig, axes = plt.subplots(len(examples), 3, figsize=(12, 4 * len(examples)), squeeze=False)
    for row_axes, pair in zip(axes, examples):
        height = load_real_height(pair["image_path"], data_type, image_height_range_nm)
        truth = load_array(pair["mask_path"]) > 0.5 if pair.get("mask_path") is not None else None
        if truth is not None and crop_padding_px is not None:
            height, truth_f = crop_to_molecule([height, truth.astype(np.float32)], truth, crop_padding_px)
            truth = truth_f > 0.5
        dna_prob, cross_prob = predict_native(model, height)
        row_axes[0].imshow(height, cmap="afmhot", vmin=-0.5, vmax=4.0)
        row_axes[0].set_title(f"{Path(pair['image_path']).name}")
        row_axes[1].imshow(height, cmap="gray", vmin=-0.5, vmax=4.0)
        row_axes[1].imshow(np.ma.masked_where(dna_prob <= threshold, dna_prob), cmap="viridis", alpha=0.6, vmin=0, vmax=1)
        if truth is not None:
            row_axes[1].contour(truth, levels=[0.5], colors="red", linewidths=0.6)
        row_axes[1].set_title("prediction (colour) / true mask (red)")
        row_axes[2].imshow(cross_prob, cmap="magma", vmin=0, vmax=1)
        row_axes[2].set_title("crossing probability")
        for ax in row_axes:
            ax.axis("off")
    plt.tight_layout()
    show_figure(plt.gcf())


def analyse_single_image(
    model: torch.nn.Module,
    path: str | Path,
    data_type: str,
    threshold: float = 0.5,
    nm_per_pixel: float | None = None,
    image_height_range_nm: tuple[float, float] = (0.0, 4.0),
) -> dict[str, Any]:
    """Segment one ``.npy`` or image file and measure the predicted mask."""

    height = load_real_height(path, data_type, image_height_range_nm)
    dna_prob, cross_prob = predict_native(model, height)
    mask = dna_prob > threshold
    return dict(path=Path(path), height=height, dna_prob=dna_prob, cross_prob=cross_prob, mask=mask,
                nm_per_pixel=nm_per_pixel, measurements=measure_mask(mask, nm_per_pixel))


def show_single_image_result(result: dict[str, Any]) -> None:
    """Show the image, the DNA segmentation and the crossing map of one file."""

    m = result["measurements"]
    unit = "nm" if "skeleton_length_nm" in m else "px"
    length = m.get("skeleton_length_nm", m["skeleton_length_px"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(result["height"], cmap="afmhot", vmin=-0.5, vmax=4.0)
    axes[0].set_title(result["path"].name)
    axes[1].imshow(result["height"], cmap="gray", vmin=-0.5, vmax=4.0)
    axes[1].imshow(np.ma.masked_where(~result["mask"], result["dna_prob"]), cmap="viridis", alpha=0.6, vmin=0, vmax=1)
    axes[1].set_title(f"DNA mask: {m['components']} pieces, skeleton ≈ {length:.0f} {unit}")
    axes[2].imshow(result["cross_prob"], cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("crossing probability")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    show_figure(plt.gcf())


def show_figure(fig) -> None:
    """Display a Matplotlib figure in the notebook and close it.

    Displaying the figure object explicitly works whatever Matplotlib backend
    is active (some libraries, e.g. TopoStats' plotting set-up, can leave a
    non-interactive backend behind, after which ``plt.show()`` shows nothing).
    """

    try:
        from IPython import get_ipython
        from IPython.display import display

        if get_ipython() is not None:
            display(fig)
            plt.close(fig)
            return
    except ImportError:  # pragma: no cover
        pass
    plt.show()


def display_df(frame: pd.DataFrame) -> None:
    """Display a DataFrame as a table in Jupyter (falls back to print)."""

    try:
        from IPython.display import display
        display(frame)
    except ImportError:  # pragma: no cover
        print(frame.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# TopoStats with the trained U-Net
# ─────────────────────────────────────────────────────────────────────────────

_ORIGINAL_PREDICT_UNET = None


def install_topostats_unet_patch(predictor_cls: type) -> None:
    """Let TopoStats use a PyTorch U-Net from this notebook (this session only).

    TopoStats' grain detection loads ``grains.unet_config.model_path`` with
    ``topostats.torch_unet.TorchUNetPredictor`` when the path ends in ``.pt``
    and then calls ``predict_unet`` on every grain crop. This replaces both,
    in memory only, so that

    * ``predictor_cls`` (defined in the notebook) builds the right network
      (simple U-Net or Res-U-Net) from the checkpoint, and
    * each grain crop is preprocessed exactly as during training and the
      probability map is restored to the crop's pixels before the
      ``confidence`` threshold is applied.

    No installed file is changed; restarting the kernel undoes the patch.

    Parameters
    ----------
    predictor_cls : type
        Class with ``__init__(model_path)`` and
        ``native_probability(image, lower_norm_bound, upper_norm_bound)``.
    """

    global _ORIGINAL_PREDICT_UNET
    import topostats
    import topostats.grains as grains
    import topostats.unet_masking as masking

    try:
        import topostats.torch_unet as adapter
    except ImportError:  # older TopoStats without the PyTorch adapter module
        adapter = types.ModuleType("topostats.torch_unet")
        sys.modules["topostats.torch_unet"] = adapter
        topostats.torch_unet = adapter
    adapter.TorchUNetPredictor = predictor_cls

    if _ORIGINAL_PREDICT_UNET is None:
        _ORIGINAL_PREDICT_UNET = masking.predict_unet

    def predict_unet(image, model, confidence, model_input_shape, upper_norm_bound, lower_norm_bound):
        if not isinstance(model, predictor_cls):
            return _ORIGINAL_PREDICT_UNET(image, model, confidence, model_input_shape,
                                          upper_norm_bound, lower_norm_bound)
        probability = model.native_probability(image, lower_norm_bound=lower_norm_bound,
                                                upper_norm_bound=upper_norm_bound)
        foreground = probability > confidence
        return np.stack([~foreground, foreground], axis=-1)

    grains.predict_unet = predict_unet
    masking.predict_unet = predict_unet


def quiet_topostats() -> None:
    """Show only warnings and errors from TopoStats and its file readers."""

    for name in ("topostats", "AFMReader"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        from loguru import logger as _loguru
        _loguru.remove()
        _loguru.add(sys.stderr, level="WARNING")
    except ImportError:
        pass


def load_topostats_config(yaml_path: str | Path) -> dict[str, Any]:
    """Read a TopoStats YAML configuration file into a dictionary."""

    with open(yaml_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def update_nested(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively update a nested configuration dictionary (returns ``config``)."""

    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            update_nested(config[key], value)
        else:
            config[key] = value
    return config


TOPOSTATS_EXPOSED_SETTINGS = {
    # notebook name: path inside the TopoStats configuration
    "confidence": ("grains", "unet_config", "confidence"),
    "lower_norm_bound": ("grains", "unet_config", "lower_norm_bound"),
    "upper_norm_bound": ("grains", "unet_config", "upper_norm_bound"),
    "remove_disconnected_grains": ("grains", "unet_config", "remove_disconnected_grains"),
    "filter_threshold_method": ("filter", "threshold_method"),
    "filter_gaussian_size": ("filter", "gaussian_size"),
    "grain_threshold_nm": ("grains", "threshold_absolute"),
    "grain_area_thresholds_nm2": ("grains", "area_thresholds"),
    "grain_crop_padding": ("grains", "grain_crop_padding"),
    "remove_edge_intersecting_grains": ("grains", "remove_edge_intersecting_grains"),
    "pruning_max_length_nm": ("disordered_tracing", "pruning_params", "max_length"),
    "node_joining_length_nm": ("nodestats", "node_joining_length"),
    "node_extend_dist_nm": ("nodestats", "node_extend_dist"),
}


def exposed_topostats_settings(yaml_path: str | Path) -> dict[str, Any]:
    """Read the settings exposed in the notebook from a TopoStats preset file.

    Returns
    -------
    dict[str, Any]
        Notebook name → value, for every entry of ``TOPOSTATS_EXPOSED_SETTINGS``.
    """

    config = load_topostats_config(yaml_path)
    values = {}
    for name, keys in TOPOSTATS_EXPOSED_SETTINGS.items():
        node = config
        for key in keys:
            node = node[key]
        values[name] = node
    return values


def build_topostats_config(
    yaml_path: str | Path,
    settings: dict[str, Any],
    spm_path: str | Path,
    model_path: str | Path | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Load a preset, apply the notebook's exposed settings and complete it.

    Parameters
    ----------
    yaml_path : str | Path
        TopoStats preset (``configs/topostats_*.yaml``).
    settings : dict
        Notebook name → value (see ``TOPOSTATS_EXPOSED_SETTINGS``).
    spm_path, model_path, output_dir
        See :func:`prepare_topostats_config`.

    Returns
    -------
    dict
        Configuration ready for :func:`run_topostats`.
    """

    config = load_topostats_config(yaml_path)
    for name, value in settings.items():
        keys = TOPOSTATS_EXPOSED_SETTINGS[name]
        node = config
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
    return prepare_topostats_config(config, spm_path, model_path, output_dir)


def prepare_topostats_config(
    config: dict[str, Any],
    spm_path: str | Path,
    model_path: str | Path | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Complete a TopoStats configuration for an in-notebook run.

    Sets the input folder, output folder and U-Net checkpoint, switches off
    TopoStats' own plot files (the notebook draws its own figures), runs on
    one core, validates the result against TopoStats' schema and adds the
    plotting dictionary that the command-line tool normally adds.

    Parameters
    ----------
    config : dict
        Configuration from :func:`load_topostats_config` (and overrides).
    spm_path : str | Path
        The scan to process.
    model_path : str | Path | None
        U-Net checkpoint; ``None`` uses TopoStats' height thresholding only.
    output_dir : str | Path
        Folder for TopoStats outputs.

    Returns
    -------
    dict
        The validated configuration.
    """

    from topostats.config import update_plotting_config
    from topostats.validation import DEFAULT_CONFIG_SCHEMA, validate_config

    quiet_topostats()
    config = dict(config)
    config["base_dir"] = Path(spm_path).resolve().parent
    config["output_dir"] = Path(output_dir).resolve()
    config["file_ext"] = Path(spm_path).suffix.lower()
    config["cores"] = 1
    config["grains"]["unet_config"]["model_path"] = None if model_path is None else str(Path(model_path).resolve())
    config["plotting"]["run"] = False
    validate_config(config, schema=DEFAULT_CONFIG_SCHEMA, config_type="YAML configuration file")
    config["output_dir"].mkdir(parents=True, exist_ok=True)
    config["plotting"]["plot_dict"] = yaml.safe_load(
        (resources.files("topostats") / "plotting_dictionary.yaml").read_text())
    config["plotting"] = update_plotting_config(config["plotting"])
    # Some TopoStats stages save diagnostic plots even when plotting is off; the
    # notebook draws its own figures, so no plot files are written at all.
    for options in config["plotting"]["plot_dict"].values():
        options["save"] = False
        options["core_set"] = False
    return config


def run_topostats(spm_path: str | Path, config: dict[str, Any], quiet: bool = True) -> dict[str, Any]:
    """Run the full TopoStats pipeline on one scan inside the notebook.

    Filtering → grain detection (height threshold, then the U-Net on each
    grain crop) → grain statistics → disordered tracing → node statistics
    (crossings) → ordered tracing → splining → curvature.

    Parameters
    ----------
    spm_path : str | Path
        The ``.spm`` scan.
    config : dict
        Output of :func:`prepare_topostats_config`.
    quiet : bool, optional
        Hide TopoStats' INFO log messages. Defaults to ``True``.

    Returns
    -------
    dict[str, Any]
        ``topostats`` (the processed TopoStats object), ``molecules``,
        ``grains``, ``image_stats``, ``disordered`` and ``branches`` data
        frames, and the ``pixel_nm`` size.
    """

    from topostats.io import LoadScans
    from topostats.processing import process_scan

    if quiet:
        quiet_topostats()

    # An absolute path keeps TopoStats' processed files inside config["output_dir"].
    scans = LoadScans([Path(spm_path).resolve()], config=config)
    scans.get_data()
    scan = next(iter(scans.img_dict.values()))
    filename, grain_df, topo, image_df, disordered_df, branch_df, molecule_df = process_scan(
        scan,
        base_dir=config["base_dir"],
        filter_config=config["filter"],
        grains_config=config["grains"],
        grainstats_config=config["grainstats"],
        disordered_tracing_config=config["disordered_tracing"],
        nodestats_config=config["nodestats"],
        ordered_tracing_config=config["ordered_tracing"],
        splining_config=config["splining"],
        curvature_config=config["curvature"],
        plotting_config=config["plotting"],
        output_dir=config["output_dir"],
    )
    return dict(filename=filename, topostats=topo, molecules=molecule_df, grains=grain_df,
                image_stats=image_df, disordered=disordered_df, branches=branch_df,
                pixel_nm=float(topo.pixel_to_nm_scaling))


def molecule_table(result: dict[str, Any]) -> pd.DataFrame:
    """Per-molecule contour lengths (nm), circularity and grain crossing counts."""

    topo = result["topostats"]
    rows = []
    for grain_number, crop in topo.grain_crops.items():
        trace = getattr(crop, "_ordered_trace", None) or getattr(crop, "ordered_trace", None)
        molecules = getattr(trace, "molecule_data", None) or {}
        nodes = getattr(crop, "_nodes", None) or getattr(crop, "nodes", None) or {}
        for molecule_number, molecule in molecules.items():
            length = getattr(molecule, "contour_length", None)
            rows.append(dict(grain=int(grain_number), molecule=int(molecule_number),
                             contour_length_nm=None if length is None else float(length) * 1e9,
                             circular=getattr(molecule, "circular", None),
                             crossings_in_grain=len(nodes)))
    return pd.DataFrame(rows)


def show_topostats_results(result: dict[str, Any], title: str = "") -> pd.DataFrame:
    """Show the filtered scan, the U-Net grain mask and the ordered traces.

    Ordered traces are drawn as their splines over the filtered image and
    labelled with grain number and contour length.

    Returns
    -------
    pd.DataFrame
        The per-molecule table (also displayed).
    """

    topo = result["topostats"]
    pixel_nm = result["pixel_nm"]
    image = np.asarray(topo.image)
    ny, nx = image.shape
    extent = (0, nx * pixel_nm, ny * pixel_nm, 0)
    mask = np.asarray(topo.full_mask_tensor)[..., 1] if getattr(topo, "full_mask_tensor", None) is not None else None

    # TopoStats applies its own Matplotlib style globally; draw this figure in the default style.
    with plt.style.context("default"):
        return _draw_topostats_results(result, topo, pixel_nm, image, extent, mask, title)


def _draw_topostats_results(result, topo, pixel_nm, image, extent, mask, title) -> pd.DataFrame:
    """Draw the three panels of :func:`show_topostats_results` and show the table."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    axes[0].imshow(image, cmap="afmhot", vmin=-1, vmax=4, extent=extent)
    axes[0].set_title("Filtered image (TopoStats)")
    axes[1].imshow(image, cmap="gray", vmin=-1, vmax=4, extent=extent)
    if mask is not None:
        axes[1].imshow(np.ma.masked_where(~mask, mask), cmap="autumn", alpha=0.55, extent=extent)
    axes[1].set_title("Grain mask from the U-Net")
    axes[2].imshow(image, cmap="gray", vmin=-1, vmax=4, extent=extent)
    colours = plt.cm.tab10.colors
    table = molecule_table(result)
    for k, (grain_number, crop) in enumerate(topo.grain_crops.items()):
        trace = getattr(crop, "_ordered_trace", None) or getattr(crop, "ordered_trace", None)
        molecules = getattr(trace, "molecule_data", None) or {}
        bbox = getattr(crop, "_bbox", None) or getattr(crop, "bbox", None)
        for molecule in molecules.values():
            coords = getattr(molecule, "splined_coords", None)
            if coords is None or len(coords) == 0:
                coords = getattr(molecule, "ordered_coords", None)
            if coords is None or len(coords) == 0 or bbox is None:
                continue
            rows = (np.asarray(coords)[:, 0] + bbox[0]) * pixel_nm
            cols = (np.asarray(coords)[:, 1] + bbox[1]) * pixel_nm
            colour = colours[k % len(colours)]
            axes[2].plot(cols, rows, "-", color=colour, linewidth=1.4)
            length = getattr(molecule, "contour_length", None)
            label = f"G{grain_number}" + (f": {float(length) * 1e9:.0f} nm" if length else "")
            axes[2].text(cols.mean(), max(rows.min() - 4 * pixel_nm, 0), label, color="white",
                         fontsize=9, ha="center", va="bottom", clip_on=True,
                         bbox=dict(facecolor=colour, alpha=0.75, pad=1.5, edgecolor="none"))
    axes[2].set_title("Ordered traces (splines)")
    for ax in axes:
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    show_figure(plt.gcf())
    display_df(table.round(1))
    return table
