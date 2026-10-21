"""Functions for Part 2 of DNA_simulation_and_segmentation.ipynb.

Part 2 trains, or loads, a two-headed U-Net that segments DNA (output
channel 0) and highlights DNA crossings (channel 1) in AFM height maps,
evaluates it on simulated data and applies it to real data, optionally
inside TopoStats. This module holds every function used by Part 2; the
classes (dataset, loss, metrics, networks, TopoStats predictor) are
defined in the notebook.

Preprocessing matches the provided checkpoints: heights in nm are
clipped to ``NORMALIZATION_BOUNDS_NM`` and scaled to 0-1, then resized
(aspect ratio kept, bilinear) and centre-padded to ``TARGET_SIZE``
pixels. Network outputs are restored to the native pixels before they
are thresholded.

Key Features
------------
- **Lazy DeepTrack data pipeline**

    Manifest rows become DeepTrack sources; arrays are read, cropped to
    the molecule, augmented, normalised and resized only when a sample
    is requested.

- **Loss building blocks and scores**

    Differentiable skeletons for the clDice loss, per-sample means, and
    Dice, IoU, precision and recall.

- **Checkpoints and training curves**

    The provided networks, loading and saving in the same format, and
    plots of the recorded training history.

- **Real data**

    Height maps (``.npy``), images and ``.spm`` scans; whole datasets
    are scored against their masks, single files are segmented and
    measured.

- **TopoStats with the trained U-Net**

    TopoStats 2.4.2 is connected to the PyTorch network for the current
    session only, and its filtered image, U-Net mask and ordered traces
    are shown.

Module Structure
----------------
Functions:

- `find_column(frame, candidates, required) -> str | None`

    First matching manifest column.

- `resolve_data_path(value, root) -> Path`

    Path relative to the dataset folder.

- `manifest_to_records(frame, root) -> list[dict[str, str]]`

    Manifest rows to DeepTrack source rows.

- `split_records(records, val_fraction, seed) -> tuple[list, list]`

    Training and validation split.

- `load_array(path) -> np.ndarray`

    Load a ``.npy`` file or an image.

- `normalize_height(image, lower, upper) -> np.ndarray`

    Clip and scale heights to 0-1.

- `resize_and_pad_np(array, target_size, mode, fill) -> np.ndarray`

    Aspect-preserving resize onto a square canvas.

- `prepare_height(image, target_size, lower, upper) -> tuple`

    Network input tensor and the geometry to undo it.

- `restore_probability(probability, geometry) -> np.ndarray`

    Network output back to the native pixels.

- `crop_to_molecule(arrays, reference_mask, padding_px) -> list`

    Crop arrays to a molecule, like a TopoStats grain crop.

- `augment_triplet(image, dna, cross) -> tuple`

    The same random flips and rotation for image and targets.

- `load_sample_factory(target_size, augment, ...) -> Callable`

    Loader of one manifest row.

- `build_pipeline(source, target_size, augment, ...) -> Any`

    Lazy DeepTrack pipeline.

- `soft_erode(x)`, `soft_open(x)`, `soft_skeletonize(x, iterations)`

    Differentiable morphology for the clDice loss.

- `mean_per_sample(x)`, `masked_mean(value, mask)`

    Per-sample reductions used by the loss.

- `dice_score_from_prob(prob, target, ...)`,
  `iou_score_from_prob(prob, target, ...)`,
  `binary_scores(pred, truth)`

    Overlap scores.

- `valid_group_count(num_channels, max_groups) -> int`,
  `count_parameters(model) -> int`

    Small helpers for the network classes.

- `checkpoint_for(use_md, architecture, checkpoint_dir) -> Path`

    The provided checkpoint for a pathway and architecture.

- `load_checkpoint(model, path, device) -> dict[str, Any]`,
  `save_checkpoint(model, path, metadata, history) -> Path`

    Checkpoint input and output.

- `pick_precision() -> tuple[str, str]`,
  `get_latest_metrics_path(trainer) -> Path`,
  `history_from_lightning_csv(metrics) -> list[dict[str, Any]]`,
  `plot_training_history(history, title) -> None`

    Training set-up and training curves.

- `evaluate_loader(model, loader, loss_fn, ...) -> dict[str, float]`,
  `inspect_crossing_batch(batch, model) -> None`,
  `show_prediction(batch, model, index) -> None`

    Validation metrics and previews.

- `normalized_stem(path)`, `collect_real_pairs(root, data_type)`,
  `load_real_height(path, data_type, ...)`,
  `predict_native(model, height_nm, ...)`, `measure_mask(mask, ...)`,
  `evaluate_real_dataset(model, pairs, ...)`,
  `show_real_dataset_results(results, model, pairs, ...)`,
  `analyse_single_image(model, path, ...)`,
  `show_single_image_result(result)`, `show_figure(fig)`,
  `display_df(frame)`

    Real-data evaluation and display.

- `check_topostats_version()`, `install_topostats_unet_patch(cls)`,
  `quiet_topostats()`, `load_topostats_config(yaml_path)`,
  `update_nested(config, overrides)`,
  `exposed_topostats_settings(yaml_path)`,
  `build_topostats_config(yaml_path, settings, ...)`,
  `prepare_topostats_config(config, spm_path, ...)`,
  `run_topostats(spm_path, config, quiet)`, `molecule_table(result)`,
  `show_topostats_results(result, title)`

    TopoStats with the trained U-Net.

Attributes:

- `TARGET_SIZE`: int

    Network input size in pixels (512).

- `NORMALIZATION_BOUNDS_NM`: tuple[float, float]

    Height range mapped to 0-1 (-1 to 5 nm).

- `CHECKPOINT_FILES`: dict[tuple[str, str], str]

    File name of the provided checkpoint per pathway and architecture.

- `TOPOSTATS_EXPOSED_SETTINGS`: dict[str, tuple[str, ...]]

    TopoStats settings shown in the notebook and where they live in the
    configuration.

Examples
--------
>>> import torch
>>> import unet_segmentation_helpers as seg

Prepare a height map for the network and map the output back:

>>> height = torch.rand(300, 200).numpy() * 3
>>> tensor, geometry = seg.prepare_height(height)
>>> tuple(tensor.shape)
(1, 1, 512, 512)
>>> seg.restore_probability(tensor[0, 0].numpy(), geometry).shape
(300, 200)

"""

from __future__ import annotations

import logging
import random
import sys
import types
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Any, Literal

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


# ────────────────────────────────────────────────────────────────────────────
# Manifest → DeepTrack sources
# ────────────────────────────────────────────────────────────────────────────

IMAGE_COLUMNS = ("image_npy", "image_path", "img_path", "afm_path")
DNA_COLUMNS = ("dna_mask_npy", "dna_path", "mask_path", "label_path")
CROSS_COLUMNS = ("cross_mask_npy", "cross_path", "crossing_path")


def find_column(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
    required: bool = True,
) -> str | None:
    """Return the first of several column names present in a manifest.

    This lets manifests made outside the notebook use other common column
    names.

    Parameters
    ----------
    frame: pd.DataFrame
        The manifest.
    candidates: tuple[str, ...]
        Column names to try, in order (``IMAGE_COLUMNS``, ``DNA_COLUMNS``
        or ``CROSS_COLUMNS``).
    required: bool, optional
        Raise an error when no candidate is present. Defaults to `True`.

    Returns
    -------
    str | None
        The first column found, or `None` if none is found and the column
        is optional.

    Raises
    ------
    KeyError
        If ``required`` is `True` and no candidate is present.

    Examples
    --------
    >>> import pandas as pd
    >>> import unet_segmentation_helpers as seg

    >>> frame = pd.DataFrame(columns=["img_path", "mask_path"])
    >>> seg.find_column(frame, seg.IMAGE_COLUMNS)
    'img_path'
    >>> seg.find_column(frame, seg.CROSS_COLUMNS, required=False) is None
    True

    """

    for column in candidates:
        if column in frame.columns:
            return column
    if required:
        raise KeyError(f"Manifest is missing one of: {', '.join(candidates)}")
    return None


def resolve_data_path(value: Any, root: Path) -> Path:
    """Resolve a manifest path relative to the dataset folder.

    Absolute paths are returned unchanged.

    Parameters
    ----------
    value: Any
        Path as read from the manifest.
    root: Path
        Folder that relative paths start from (the manifest's folder).

    Returns
    -------
    Path
        The resolved path.

    Examples
    --------
    >>> from pathlib import Path
    >>> import unet_segmentation_helpers as seg

    >>> seg.resolve_data_path("images/img_0000.npy", Path("data")).as_posix()
    'data/images/img_0000.npy'

    """

    path = Path(str(value))
    return path if path.is_absolute() else Path(root) / path


def manifest_to_records(
    frame: pd.DataFrame,
    root: Path,
) -> list[dict[str, str]]:
    """Turn a dataset manifest into DeepTrack source rows.

    The image, DNA mask and optional crossing mask columns are found with
    :func:`find_column`, paths are resolved relative to ``root`` and
    checked. A ``split`` column (``train`` / ``validation``), if present,
    is kept.

    Parameters
    ----------
    frame: pd.DataFrame
        The manifest.
    root: Path
        The dataset folder.

    Returns
    -------
    list[dict[str, str]]
        One row per sample with ``id``, ``image_path``, ``dna_path``,
        ``cross_path`` (empty without a crossing mask) and ``split``.

    Raises
    ------
    FileNotFoundError
        If files listed in the manifest are missing (the first ten are
        listed).
    RuntimeError
        If the manifest contains no valid sample.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> import numpy as np
    >>> import pandas as pd
    >>> import unet_segmentation_helpers as seg

    >>> root = Path(tempfile.mkdtemp())
    >>> np.save(root / "img.npy", np.zeros((4, 4)))
    >>> np.save(root / "dna.npy", np.zeros((4, 4)))
    >>> frame = pd.DataFrame(
    ...     {"image_npy": ["img.npy"], "dna_mask_npy": ["dna.npy"]}
    ... )
    >>> records = seg.manifest_to_records(frame, root)
    >>> records[0]["cross_path"], Path(records[0]["image_path"]).name
    ('', 'img.npy')

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
        records.append(
            dict(
                id=str(row.get("index", row_index)),
                image_path=str(image_path),
                dna_path=str(dna_path),
                cross_path=cross_path,
                split=(
                    str(row.get("split", ""))
                    if "split" in frame.columns
                    else ""
                ),
            )
        )
    if missing:
        raise FileNotFoundError(
            "Some manifest files could not be found. First missing files:\n"
            + "\n".join(missing[:10])
        )
    if not records:
        raise RuntimeError("No valid samples were found in the manifest.")
    return records


def split_records(
    records: list[dict[str, str]],
    val_fraction: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split records into training and validation lists.

    If every record has a ``split`` of ``train``, ``validation`` or
    ``test``, that assignment is used (``test`` records are left out).
    Otherwise ``val_fraction`` of the records, drawn with ``seed``, go to
    validation.

    Parameters
    ----------
    records: list[dict[str, str]]
        Output of :func:`manifest_to_records`.
    val_fraction: float
        Fraction of records used for validation in a random split.
    seed: int
        Random seed of the split.

    Returns
    -------
    tuple[list[dict[str, str]], list[dict[str, str]]]
        Training records and validation records.

    Examples
    --------
    >>> import unet_segmentation_helpers as seg

    >>> records = [{"id": str(i), "split": ""} for i in range(10)]
    >>> train, val = seg.split_records(records, 0.2, seed=3)
    >>> len(train), len(val)
    (8, 2)

    """

    if records and all(
        r.get("split") in ("train", "validation", "test") for r in records
    ):
        return (
            [r for r in records if r["split"] == "train"],
            [r for r in records if r["split"] == "validation"],
        )
    if len(records) < 2:
        return records, records
    order = np.random.default_rng(seed).permutation(len(records))
    n_val = min(
        max(1, int(round(len(records) * val_fraction))), len(records) - 1
    )
    val_index = set(int(i) for i in order[:n_val])
    train_rows = [r for i, r in enumerate(records) if i not in val_index]
    val_rows = [r for i, r in enumerate(records) if i in val_index]
    return train_rows, val_rows


# ────────────────────────────────────────────────────────────────────────────
# Lazy loading pipeline
# ────────────────────────────────────────────────────────────────────────────


def load_array(path: str | Path) -> np.ndarray:
    """Load a ``.npy`` array or an image as a 2-D float32 array.

    Colour images are converted to one channel by averaging red, green and
    blue.

    Parameters
    ----------
    path: str | Path
        File to load.

    Returns
    -------
    np.ndarray
        2-D float32 array.

    Examples
    --------
    >>> import unet_segmentation_helpers as seg

    >>> image = seg.load_array("../common-assets/real-test-data/image_5.npy")
    >>> image.shape, image.dtype
    ((512, 512), dtype('float32'))

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
    """Clip a height map to ``[lower, upper]`` nm and scale it to 0-1.

    Fixed physical bounds (rather than per-image percentiles) give a DNA
    strand of a given height the same input value in every image.

    Parameters
    ----------
    image: np.ndarray
        Height map in nm.
    lower: float, optional
        Height mapped to 0 in nm. Defaults to -1.0.
    upper: float, optional
        Height mapped to 1 in nm. Defaults to 5.0.

    Returns
    -------
    np.ndarray
        Normalised float32 array.

    Raises
    ------
    ValueError
        If ``upper`` is not larger than ``lower``.

    Examples
    --------
    >>> import numpy as np
    >>> import unet_segmentation_helpers as seg

    >>> seg.normalize_height(np.array([-2.0, -1.0, 2.0, 5.0, 9.0]))
    array([0. , 0. , 0.5, 1. , 1. ], dtype=float32)

    """

    image = np.nan_to_num(np.asarray(image, dtype=np.float32))
    if upper <= lower:
        raise ValueError("Invalid normalization bounds.")
    return ((np.clip(image, lower, upper) - lower) / (upper - lower)).astype(
        np.float32
    )


def resize_and_pad_np(
    array: np.ndarray,
    target_size: int,
    mode: Literal["bilinear", "nearest", "bicubic"],
    fill: float = 0.0,
) -> np.ndarray:
    """Resize an array keeping its aspect ratio and pad it to a square.

    The longer side becomes ``target_size``; the array is centred on a
    square canvas filled with ``fill``.

    Parameters
    ----------
    array: np.ndarray
        2-D array.
    target_size: int
        Side of the output in pixels.
    mode: "bilinear" | "nearest" | "bicubic"
        Interpolation: "bilinear" for images and soft targets, "nearest"
        for binary masks.
    fill: float, optional
        Value of the padding. Defaults to 0.0.

    Returns
    -------
    np.ndarray
        Array of shape ``(target_size, target_size)``.

    Examples
    --------
    >>> import numpy as np
    >>> import unet_segmentation_helpers as seg

    >>> out = seg.resize_and_pad_np(np.ones((100, 50)), 64, "nearest")
    >>> out.shape, int(out.sum())
    ((64, 64), 2048)

    """

    array = np.ascontiguousarray(
        np.nan_to_num(np.squeeze(np.asarray(array, dtype=np.float32)))
    )
    height, width = array.shape[-2:]
    scale = min(target_size / height, target_size / width)
    new_h, new_w = max(1, int(round(height * scale))), max(
        1, int(round(width * scale))
    )
    kwargs: dict[str, Any] = dict(size=(new_h, new_w), mode=mode)
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    resized = F.interpolate(torch.from_numpy(array)[None, None], **kwargs)[
        0, 0
    ].numpy()
    canvas = np.full((target_size, target_size), fill, dtype=np.float32)
    top, left = (target_size - new_h) // 2, (target_size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return np.ascontiguousarray(canvas)


def prepare_height(
    image: np.ndarray,
    target_size: int = TARGET_SIZE,
    lower: float = NORMALIZATION_BOUNDS_NM[0],
    upper: float = NORMALIZATION_BOUNDS_NM[1],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Prepare one height map as network input.

    The heights are normalised with :func:`normalize_height`, resized so
    the longer side is ``target_size`` (bilinear, aspect ratio kept) and
    centre-padded with the normalised value of 0 nm. The geometry needed
    by :func:`restore_probability` is returned as well.

    Parameters
    ----------
    image: np.ndarray
        Height map in nm.
    target_size: int, optional
        Network input size in pixels. Defaults to 512.
    lower: float, optional
        Lower normalisation bound in nm. Defaults to -1.0.
    upper: float, optional
        Upper normalisation bound in nm. Defaults to 5.0.

    Returns
    -------
    tuple[torch.Tensor, dict[str, Any]]
        Input tensor of shape ``(1, 1, target_size, target_size)`` and the
        geometry (original shape, resized shape and offsets).

    Examples
    --------
    >>> import numpy as np
    >>> import unet_segmentation_helpers as seg

    >>> tensor, geometry = seg.prepare_height(np.zeros((300, 200)))
    >>> tuple(tensor.shape), geometry["resized_shape"]
    ((1, 1, 512, 512), (512, 341))

    """

    image = normalize_height(image, lower, upper)
    height, width = image.shape
    scale = min(target_size / height, target_size / width)
    new_h, new_w = max(1, round(height * scale)), max(1, round(width * scale))
    top, left = (target_size - new_h) // 2, (target_size - new_w) // 2
    resized = F.interpolate(
        torch.from_numpy(np.ascontiguousarray(image))[None, None],
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    )
    background = float(np.clip(-lower / (upper - lower), 0, 1))
    canvas = torch.full(
        (1, 1, target_size, target_size), background, dtype=torch.float32
    )
    canvas[..., top : top + new_h, left : left + new_w] = resized
    return canvas, dict(
        original_shape=(height, width),
        top=top,
        left=left,
        resized_shape=(new_h, new_w),
        target_size=target_size,
    )


def restore_probability(
    probability: np.ndarray, geometry: dict[str, Any]
) -> np.ndarray:
    """Map a network probability map back to the native pixels.

    The padding added by :func:`prepare_height` is removed and the
    continuous probabilities are resized to the original shape before any
    threshold is applied.

    Parameters
    ----------
    probability: np.ndarray
        Probability map of shape ``(target_size, target_size)``.
    geometry: dict[str, Any]
        Geometry returned by :func:`prepare_height`.

    Returns
    -------
    np.ndarray
        Probabilities with the original image shape, clipped to 0-1.

    Examples
    --------
    >>> import numpy as np
    >>> import unet_segmentation_helpers as seg

    >>> _, geometry = seg.prepare_height(np.zeros((300, 200)))
    >>> seg.restore_probability(np.full((512, 512), 0.7), geometry).shape
    (300, 200)

    """

    top, left = geometry["top"], geometry["left"]
    height, width = geometry["resized_shape"]
    unpadded = np.asarray(probability, dtype=np.float32)[
        top : top + height, left : left + width
    ]
    restored = F.interpolate(
        torch.from_numpy(np.ascontiguousarray(unpadded))[None, None],
        size=geometry["original_shape"],
        mode="bilinear",
        align_corners=False,
    )
    return restored[0, 0].numpy().clip(0, 1)


def crop_to_molecule(
    arrays: list[np.ndarray],
    reference_mask: np.ndarray,
    padding_px: int,
) -> list[np.ndarray]:
    """Crop arrays to the bounding box of a mask plus a margin.

    This mimics a TopoStats grain crop, so the network sees the molecule
    at about the scale the provided checkpoints were trained on. The
    margin is clipped at the image border; an empty mask leaves the
    arrays unchanged.

    Parameters
    ----------
    arrays: list[np.ndarray]
        Arrays with the same shape as ``reference_mask``.
    reference_mask: np.ndarray
        Mask whose non-zero pixels define the molecule.
    padding_px: int
        Margin on every side in pixels.

    Returns
    -------
    list[np.ndarray]
        The cropped arrays.

    Examples
    --------
    >>> import numpy as np
    >>> import unet_segmentation_helpers as seg

    >>> mask = np.zeros((100, 100))
    >>> mask[40:50, 30:60] = 1
    >>> image, = seg.crop_to_molecule([np.ones((100, 100))], mask, 5)
    >>> image.shape
    (20, 40)

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
    """Apply the same random flips and 90 degree rotation to three arrays.

    Used for training augmentation of an image and its two targets, which
    stay aligned.

    Parameters
    ----------
    image: np.ndarray
        Image.
    dna: np.ndarray
        DNA mask.
    cross: np.ndarray
        Crossing mask.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The transformed (contiguous) arrays.

    Examples
    --------
    >>> import random
    >>> import numpy as np
    >>> import unet_segmentation_helpers as seg

    >>> random.seed(0)
    >>> image = np.arange(6.0).reshape(2, 3)
    >>> a, b, c = seg.augment_triplet(image, image > 2, image * 0)
    >>> bool(np.array_equal(a > 2, b))
    True

    """

    if random.random() < 0.5:
        image, dna, cross = np.fliplr(image), np.fliplr(dna), np.fliplr(cross)
    if random.random() < 0.5:
        image, dna, cross = np.flipud(image), np.flipud(dna), np.flipud(cross)
    k = random.randint(0, 3)
    if k:
        image, dna, cross = (
            np.rot90(image, k),
            np.rot90(dna, k),
            np.rot90(cross, k),
        )
    return (
        np.ascontiguousarray(image),
        np.ascontiguousarray(dna),
        np.ascontiguousarray(cross),
    )


def load_sample_factory(
    target_size: int = TARGET_SIZE,
    augment: bool = False,
    bounds: tuple[float, float] = NORMALIZATION_BOUNDS_NM,
    crop_padding_px: int | tuple[int, int] | None = None,
) -> Callable[[dict[str, str]], tuple[np.ndarray, np.ndarray]]:
    """Return the function that loads one manifest row.

    The returned ``load_sample(sample)`` reads the image and masks from
    the paths in the row, optionally crops them to the molecule
    (:func:`crop_to_molecule`; a ``(low, high)`` padding is drawn at random
    per sample), optionally augments them (:func:`augment_triplet`),
    normalises the image and resizes everything to ``target_size``. It is
    wrapped in a ``deeptrack.Lambda`` by :func:`build_pipeline`.

    Parameters
    ----------
    target_size: int, optional
        Network input size in pixels. Defaults to 512.
    augment: bool, optional
        Random flips and rotations. Defaults to `False`.
    bounds: tuple[float, float], optional
        Height normalisation bounds in nm. Defaults to ``(-1.0, 5.0)``.
    crop_padding_px: int | tuple[int, int] | None, optional
        Crop margin in pixels; `None` keeps the full image. Defaults to
        `None`.

    Returns
    -------
    Callable[[dict[str, str]], tuple[np.ndarray, np.ndarray]]
        ``load_sample(row)``, returning the image ``(1, H, W)`` and the
        targets ``(2, H, W)`` (DNA, crossings).

    Examples
    --------
    >>> import unet_segmentation_helpers as seg

    >>> load = seg.load_sample_factory(target_size=256, crop_padding_px=40)
    >>> row = {
    ...     "image_path": "../common-assets/real-test-data/image_5.npy",
    ...     "dna_path": "../common-assets/real-test-data/mask_5.npy",
    ... }
    >>> image, target = load(row)
    >>> image.shape, target.shape
    ((1, 256, 256), (2, 256, 256))

    """

    lower, upper = bounds
    background = float(np.clip(-lower / (upper - lower), 0, 1))

    def load_sample(
        sample: dict[str, str],
    ) -> tuple[np.ndarray, np.ndarray]:
        image = np.nan_to_num(load_array(sample["image_path"])).astype(
            np.float32
        )
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
                padding = random.randint(
                    int(crop_padding_px[0]), int(crop_padding_px[1])
                )
            else:
                padding = int(crop_padding_px)
            image, dna, cross = crop_to_molecule(
                [image, dna, cross], dna, padding
            )
        if augment:
            image, dna, cross = augment_triplet(image, dna, cross)

        image = resize_and_pad_np(
            normalize_height(image, lower, upper),
            target_size,
            "bilinear",
            fill=background,
        )
        dna = (resize_and_pad_np(dna, target_size, "nearest") > 0.5).astype(
            np.float32
        )
        cross = np.clip(
            resize_and_pad_np(cross, target_size, "bilinear"), 0.0, 1.0
        )
        return (
            np.ascontiguousarray(image[None]).astype(np.float32),
            np.ascontiguousarray(np.stack([dna, cross]).astype(np.float32)),
        )

    return load_sample


def build_pipeline(
    source: Any,
    target_size: int,
    augment: bool,
    bounds: tuple[float, float] = NORMALIZATION_BOUNDS_NM,
    crop_padding_px: int | tuple[int, int] | None = None,
) -> Any:
    """Create the lazy DeepTrack pipeline for a source of manifest rows.

    The pipeline is ``dt.Value(row) >> dt.Lambda(load_sample)``; the
    source only stores file paths, and the arrays are read when a sample
    is requested, so the dataset never has to fit in memory.

    Parameters
    ----------
    source: Any
        ``deeptrack.sources.Source`` whose samples are manifest rows.
    target_size: int
        Network input size in pixels.
    augment: bool
        Random flips and rotations.
    bounds: tuple[float, float], optional
        Height normalisation bounds in nm. Defaults to ``(-1.0, 5.0)``.
    crop_padding_px: int | tuple[int, int] | None, optional
        See :func:`load_sample_factory`. Defaults to `None`.

    Returns
    -------
    Any
        The DeepTrack pipeline; calling it on a row returns the image and
        the targets.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import unet_segmentation_helpers as seg

    >>> rows = [{
    ...     "image_path": "../common-assets/real-test-data/image_5.npy",
    ...     "dna_path": "../common-assets/real-test-data/mask_5.npy",
    ... }]
    >>> source = dt.sources.Source(samples=rows)
    >>> pipeline = seg.build_pipeline(source, 128, augment=False)
    >>> image, target = pipeline(source[0])
    >>> image.shape, target.shape
    ((1, 128, 128), (2, 128, 128))

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


# ────────────────────────────────────────────────────────────────────────────
# Loss building blocks and scores
# ────────────────────────────────────────────────────────────────────────────


def soft_erode(x: torch.Tensor) -> torch.Tensor:
    """Differentiable erosion of a probability map.

    The minimum over 3-pixel windows along rows and along columns, used
    by :func:`soft_skeletonize`.

    Parameters
    ----------
    x: torch.Tensor
        Probabilities of shape ``(B, C, H, W)``.

    Returns
    -------
    torch.Tensor
        The eroded map, same shape.

    Examples
    --------
    >>> import torch
    >>> import unet_segmentation_helpers as seg

    >>> x = torch.zeros(1, 1, 5, 5)
    >>> x[..., 1:4, 1:4] = 1
    >>> int(seg.soft_erode(x).sum())
    1

    """

    return torch.minimum(
        -F.max_pool2d(-x, (3, 1), (1, 1), (1, 0)),
        -F.max_pool2d(-x, (1, 3), (1, 1), (0, 1)),
    )


def soft_open(x: torch.Tensor) -> torch.Tensor:
    """Differentiable opening (erosion followed by dilation).

    Parameters
    ----------
    x: torch.Tensor
        Probabilities of shape ``(B, C, H, W)``.

    Returns
    -------
    torch.Tensor
        The opened map, same shape.

    Examples
    --------
    >>> import torch
    >>> import unet_segmentation_helpers as seg

    A single bright pixel is removed by opening:

    >>> x = torch.zeros(1, 1, 5, 5)
    >>> x[..., 2, 2] = 1
    >>> float(seg.soft_open(x).max())
    0.0

    """

    return F.max_pool2d(soft_erode(x), 3, 1, 1)


def soft_skeletonize(x: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    """Differentiable skeleton of a probability map (for clDice).

    The map is eroded repeatedly and, at every step, what an opening
    removes is added to the skeleton; the result is a thin centre line
    that still passes gradients to the network.

    Parameters
    ----------
    x: torch.Tensor
        Probabilities of shape ``(B, C, H, W)``.
    iterations: int, optional
        Number of erosions; should exceed half the strand width in pixels.
        Defaults to 10.

    Returns
    -------
    torch.Tensor
        The soft skeleton, same shape.

    Examples
    --------
    >>> import torch
    >>> import unet_segmentation_helpers as seg

    The skeleton of a 3-pixel-wide bar is its centre line:

    >>> x = torch.zeros(1, 1, 9, 9)
    >>> x[..., 3:6, 1:8] = 1
    >>> skeleton = seg.soft_skeletonize(x)
    >>> int(skeleton[0, 0, 4].sum()), int(skeleton[0, 0, 3].sum())
    (5, 0)

    """

    skeleton = F.relu(x - soft_open(x))
    for _ in range(iterations):
        x = soft_erode(x)
        delta = F.relu(x - soft_open(x))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def mean_per_sample(x: torch.Tensor) -> torch.Tensor:
    """Average over all dimensions except the first (batch) one.

    Parameters
    ----------
    x: torch.Tensor
        Tensor of shape ``(B, ...)``.

    Returns
    -------
    torch.Tensor
        One value per sample, shape ``(B,)``.

    Examples
    --------
    >>> import torch
    >>> import unet_segmentation_helpers as seg

    >>> seg.mean_per_sample(torch.ones(2, 1, 4, 4))
    tensor([1., 1.])

    """

    return x.flatten(1).mean(1)


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average ``value`` over the pixels where ``mask`` is 1, per sample.

    Parameters
    ----------
    value: torch.Tensor
        Values of shape ``(B, ...)``.
    mask: torch.Tensor
        Weights (usually 0 or 1) of the same shape.

    Returns
    -------
    torch.Tensor
        One average per sample, shape ``(B,)``; samples without any masked
        pixel give 0.

    Examples
    --------
    >>> import torch
    >>> import unet_segmentation_helpers as seg

    >>> value = torch.tensor([[1.0, 3.0, 5.0]])
    >>> mask = torch.tensor([[1.0, 1.0, 0.0]])
    >>> round(float(seg.masked_mean(value, mask)), 4)
    2.0

    """

    return (value * mask).flatten(1).sum(1) / (mask.flatten(1).sum(1) + 1e-6)


def dice_score_from_prob(
    prob: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean hard Dice score of thresholded probabilities.

    Dice = 2 |P and T| / (|P| + |T|) for the predicted pixels P
    (``prob >= threshold``) and the true pixels T, averaged over the batch.

    Parameters
    ----------
    prob: torch.Tensor
        Probabilities of shape ``(B, ...)``.
    target: torch.Tensor
        Binary targets of the same shape.
    threshold: float, optional
        Probability threshold. Defaults to 0.5.
    eps: float, optional
        Smoothing that avoids division by zero. Defaults to 1e-6.

    Returns
    -------
    torch.Tensor
        The mean Dice score (scalar).

    Examples
    --------
    >>> import torch
    >>> import unet_segmentation_helpers as seg

    >>> prob = torch.tensor([[0.9, 0.8, 0.1, 0.1]])
    >>> target = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    >>> round(float(seg.dice_score_from_prob(prob, target)), 3)
    0.5

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
    """Mean hard IoU of thresholded probabilities.

    IoU = |P and T| / |P or T| for the predicted pixels P
    (``prob >= threshold``) and the true pixels T, averaged over the batch.

    Parameters
    ----------
    prob: torch.Tensor
        Probabilities of shape ``(B, ...)``.
    target: torch.Tensor
        Binary targets of the same shape.
    threshold: float, optional
        Probability threshold. Defaults to 0.5.
    eps: float, optional
        Smoothing that avoids division by zero. Defaults to 1e-6.

    Returns
    -------
    torch.Tensor
        The mean IoU (scalar).

    Examples
    --------
    >>> import torch
    >>> import unet_segmentation_helpers as seg

    >>> prob = torch.tensor([[0.9, 0.8, 0.1, 0.1]])
    >>> target = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    >>> round(float(seg.iou_score_from_prob(prob, target)), 3)
    0.333

    """

    pred = (prob >= threshold).float()
    dims = tuple(range(1, pred.ndim))
    inter = torch.sum(pred * target, dim=dims)
    union = torch.sum((pred + target) > 0, dim=dims)
    return ((inter + eps) / (union + eps)).mean()


def binary_scores(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Dice, IoU, precision and recall of two binary masks.

    Parameters
    ----------
    pred: np.ndarray
        Predicted mask.
    truth: np.ndarray
        True mask of the same shape.

    Returns
    -------
    dict[str, float]
        ``dice``, ``iou``, ``precision`` (fraction of the prediction that
        is DNA) and ``recall`` (fraction of the DNA that was found).

    Examples
    --------
    >>> import numpy as np
    >>> import unet_segmentation_helpers as seg

    >>> scores = seg.binary_scores(
    ...     np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0])
    ... )
    >>> {k: round(v, 3) for k, v in scores.items()}
    {'dice': 0.5, 'iou': 0.333, 'precision': 0.5, 'recall': 0.5}

    """

    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    inter = float(np.logical_and(pred, truth).sum())
    p, t = float(pred.sum()), float(truth.sum())
    union = float(np.logical_or(pred, truth).sum())
    return dict(
        dice=(2 * inter + 1e-6) / (p + t + 1e-6),
        iou=(inter + 1e-6) / (union + 1e-6),
        precision=inter / max(p, 1.0),
        recall=inter / max(t, 1.0),
    )


# ────────────────────────────────────────────────────────────────────────────
# Networks: small helpers used by the U-Net classes in the notebook
# ────────────────────────────────────────────────────────────────────────────


def valid_group_count(num_channels: int, max_groups: int = 8) -> int:
    """Largest GroupNorm group count that divides the channel count.

    Tries ``max_groups``, 8, 4, 2 and 1 in this order.

    Parameters
    ----------
    num_channels: int
        Number of channels to normalise.
    max_groups: int, optional
        Preferred number of groups. Defaults to 8.

    Returns
    -------
    int
        A group count that divides ``num_channels``.

    Examples
    --------
    >>> import unet_segmentation_helpers as seg

    >>> seg.valid_group_count(32), seg.valid_group_count(6)
    (8, 2)

    """

    for groups in (int(max_groups), 8, 4, 2, 1):
        if groups <= num_channels and num_channels % groups == 0:
            return groups
    return 1


def count_parameters(model: torch.nn.Module) -> int:
    """Number of trainable parameters of a network.

    Parameters
    ----------
    model: torch.nn.Module
        The network.

    Returns
    -------
    int
        Number of trainable parameters.

    Examples
    --------
    >>> import torch
    >>> import unet_segmentation_helpers as seg

    >>> seg.count_parameters(torch.nn.Linear(10, 2))
    22

    """

    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ────────────────────────────────────────────────────────────────────────────
# Checkpoints and training curves
# ────────────────────────────────────────────────────────────────────────────


def checkpoint_for(
    use_md: bool,
    architecture: Literal["residual", "simple"],
    checkpoint_dir: str | Path = "checkpoints",
) -> Path:
    """Return the provided checkpoint for a pathway and architecture.

    Parameters
    ----------
    use_md: bool
        `True` for the networks trained on MD data, `False` for non-MD.
    architecture: "residual" | "simple"
        Res-U-Net or plain U-Net.
    checkpoint_dir: str | Path, optional
        Folder with the checkpoints. Defaults to "checkpoints".

    Returns
    -------
    Path
        Path of the checkpoint file.

    Raises
    ------
    ValueError
        If ``architecture`` is neither "residual" nor "simple".

    Examples
    --------
    >>> import unet_segmentation_helpers as seg

    >>> seg.checkpoint_for(True, "residual").as_posix()
    'checkpoints/md_res_unet.pt'
    >>> seg.checkpoint_for(False, "simple").name
    'non_md_simple_unet.pt'

    """

    key = ("md" if use_md else "non_md", architecture)
    if key not in CHECKPOINT_FILES:
        raise ValueError(
            "architecture must be 'residual' or 'simple', "
            f"got {architecture!r}"
        )
    return Path(checkpoint_dir) / CHECKPOINT_FILES[key]


def load_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load weights into a network and return the checkpoint's records.

    Accepts the checkpoints provided with this tutorial (``state_dict``,
    ``metadata``, ``history``), plain state dictionaries, and state
    dictionaries saved from a Deeplay or Lightning application (keys
    prefixed with ``model.``). The network is moved to ``device`` and set
    to evaluation mode.

    Parameters
    ----------
    model: torch.nn.Module
        Network with the matching architecture.
    path: str | Path
        Checkpoint file.
    device: str | torch.device, optional
        Device for the network. Defaults to "cpu".

    Returns
    -------
    dict[str, Any]
        Everything in the checkpoint except the weights (for example
        ``metadata`` and ``history``).

    Raises
    ------
    RuntimeError
        If the weights do not fit the network's architecture.

    Examples
    --------
    >>> import tempfile
    >>> import torch
    >>> import unet_segmentation_helpers as seg

    >>> path = tempfile.mktemp(suffix=".pt")
    >>> _ = seg.save_checkpoint(torch.nn.Linear(3, 1), path, {"note": "x"})
    >>> info = seg.load_checkpoint(torch.nn.Linear(3, 1), path)
    >>> info["metadata"]
    {'note': 'x'}

    """

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    state = (
        payload.get("state_dict", payload)
        if isinstance(payload, dict)
        else payload
    )
    if any(key.startswith("model.") for key in state):
        state = {
            k.removeprefix("model."): v
            for k, v in state.items()
            if k.startswith("model.")
        }
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return (
        {k: v for k, v in payload.items() if k != "state_dict"}
        if isinstance(payload, dict)
        else {}
    )


def save_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> Path:
    """Save a network in the format of the provided checkpoints.

    Parameters
    ----------
    model: torch.nn.Module
        The network.
    path: str | Path
        Output file; missing folders are created.
    metadata: dict[str, Any] | None, optional
        Description of the network and its training. Defaults to `None`.
    history: list[dict[str, Any]] | None, optional
        Per-epoch training history. Defaults to `None`.

    Returns
    -------
    Path
        Path of the written file.

    Examples
    --------
    >>> import tempfile
    >>> import torch
    >>> import unet_segmentation_helpers as seg

    >>> path = seg.save_checkpoint(
    ...     torch.nn.Linear(3, 1), tempfile.mktemp(suffix=".pt")
    ... )
    >>> sorted(torch.load(path, weights_only=True))
    ['history', 'metadata', 'state_dict']

    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(
        dict(
            state_dict=state, metadata=metadata or {}, history=history or []
        ),
        path,
    )
    return path


def pick_precision() -> tuple[str, str]:
    """Return the accelerator name and a suitable Lightning precision.

    Mixed precision (``"16-mixed"``) on CUDA GPUs, full precision on Apple
    Silicon and on the CPU.

    Returns
    -------
    tuple[str, str]
        A description of the accelerator and the precision setting.

    Examples
    --------
    >>> import unet_segmentation_helpers as seg

    >>> name, precision = seg.pick_precision()
    >>> precision in ("16-mixed", "32-true")
    True

    """

    if torch.cuda.is_available():
        return f"CUDA GPU: {torch.cuda.get_device_name(0)}", "16-mixed"
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return "Apple Silicon MPS", "32-true"
    return "CPU", "32-true"


def get_latest_metrics_path(trainer: Any) -> Path:
    """Return the ``metrics.csv`` written by a trainer's CSV logger.

    Parameters
    ----------
    trainer: Any
        The Deeplay or Lightning trainer used for training.

    Returns
    -------
    Path
        Path of ``metrics.csv`` of the latest run.

    Raises
    ------
    RuntimeError
        If the trainer has no logger with a log folder.
    FileNotFoundError
        If the CSV file does not exist.

    Examples
    --------
    >>> import unet_segmentation_helpers as seg

    >>> path = seg.get_latest_metrics_path(trainer)  # doctest: +SKIP
    >>> path.name  # doctest: +SKIP
    'metrics.csv'

    """

    if (
        trainer.logger is None
        or getattr(trainer.logger, "log_dir", None) is None
    ):
        raise RuntimeError("The trainer does not have a CSV logger.")
    metrics_path = Path(trainer.logger.log_dir) / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Could not find the metrics CSV: {metrics_path}"
        )
    return metrics_path


def history_from_lightning_csv(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a Lightning ``metrics.csv`` into the checkpoint history.

    Each epoch becomes ``{"epoch", "train": {...}, "validation": {...}}``
    with the loss and the logged DNA IoU, the format stored in the
    provided checkpoints and plotted by :func:`plot_training_history`.

    Parameters
    ----------
    metrics: pd.DataFrame
        Contents of ``metrics.csv``.

    Returns
    -------
    list[dict[str, Any]]
        One entry per epoch (epochs counted from 1).

    Examples
    --------
    >>> import pandas as pd
    >>> import unet_segmentation_helpers as seg

    >>> metrics = pd.DataFrame({
    ...     "epoch": [0, 0],
    ...     "train_loss_epoch": [0.9, None],
    ...     "val_loss_epoch": [None, 0.8],
    ...     "val_DNAIouMetric": [None, 0.7],
    ... })
    >>> row = seg.history_from_lightning_csv(metrics)[0]
    >>> row["epoch"], row["train"]["loss"], row["validation"]["iou"]
    (1, 0.9, 0.7)

    """

    metrics = (
        metrics.sort_values("epoch").groupby("epoch", as_index=False).last()
    )
    rows = []
    for _, r in metrics.iterrows():

        def pick(prefix: str, name: str) -> float:
            for column in (
                f"{prefix}_{name}_epoch",
                f"{prefix}{name}_epoch",
                f"{prefix}_{name}",
            ):
                if column in metrics.columns and not pd.isna(r[column]):
                    return float(r[column])
            return float("nan")

        iou_cols = [c for c in metrics.columns if "iou" in c.lower()]
        rows.append(
            dict(
                epoch=int(r["epoch"]) + 1,
                train=dict(
                    loss=pick("train", "loss"),
                    iou=next(
                        (
                            float(r[c])
                            for c in iou_cols
                            if c.startswith("train") and not pd.isna(r[c])
                        ),
                        float("nan"),
                    ),
                ),
                validation=dict(
                    loss=pick("val", "loss"),
                    iou=next(
                        (
                            float(r[c])
                            for c in iou_cols
                            if c.startswith("val") and not pd.isna(r[c])
                        ),
                        float("nan"),
                    ),
                ),
            )
        )
    return rows


def plot_training_history(
    history: list[dict[str, Any]], title: str = ""
) -> None:
    """Plot loss and DNA overlap scores per epoch.

    The left panel shows the training and validation loss, the right one
    the DNA IoU (and Dice if recorded); a dotted line marks the epoch that
    was kept.

    Parameters
    ----------
    history: list[dict[str, Any]]
        Checkpoint history or :func:`history_from_lightning_csv` output.
    title: str, optional
        Figure title. Defaults to "".

    Returns
    -------
    None
        The figure is shown in the notebook.

    Examples
    --------
    >>> import unet_segmentation_helpers as seg

    >>> history = [
    ...     {"epoch": 1, "train": {"loss": 0.9, "iou": 0.6},
    ...      "validation": {"loss": 0.8, "iou": 0.7}, "selected_best": True},
    ... ]
    >>> seg.plot_training_history(history, "example")  # doctest: +SKIP

    """

    if not history:
        print("No training history available.")
        return
    epochs = [r["epoch"] for r in history]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for split, style in (("train", "-"), ("validation", "--")):
        axs[0].plot(
            epochs,
            [r[split].get("loss", np.nan) for r in history],
            style,
            label=split,
        )
        axs[1].plot(
            epochs,
            [r[split].get("iou", np.nan) for r in history],
            style,
            label=f"{split} IoU",
        )
        if any("dice" in r[split] for r in history):
            axs[1].plot(
                epochs,
                [r[split].get("dice", np.nan) for r in history],
                style,
                alpha=0.6,
                label=f"{split} Dice",
            )
    best = [r["epoch"] for r in history if r.get("selected_best")]
    for ax in axs:
        if best:
            ax.axvline(
                best[-1],
                color="gray",
                linewidth=0.8,
                linestyle=":",
                label="selected epoch",
            )
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


# ────────────────────────────────────────────────────────────────────────────
# Validation metrics and previews
# ────────────────────────────────────────────────────────────────────────────


def evaluate_loader(
    model: torch.nn.Module,
    loader: Any,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    stage: str = "validation",
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate the loss, DNA Dice, DNA IoU and crossing MSE.

    Runs the network over all batches without gradients and averages the
    metrics, weighted by batch size; the result is also printed.

    Parameters
    ----------
    model: torch.nn.Module
        Network returning DNA (channel 0) and crossing (channel 1) logits.
    loader: Any
        Data loader yielding ``(image, target)`` batches.
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        The training loss.
    stage: str, optional
        Label used when printing. Defaults to "validation".
    threshold: float, optional
        DNA probability threshold. Defaults to 0.5.

    Returns
    -------
    dict[str, float]
        ``loss``, ``dna_dice``, ``dna_iou`` and ``cross_mse``.

    Examples
    --------
    >>> import torch
    >>> import unet_segmentation_helpers as seg

    A 1x1 convolution standing in for the U-Net:

    >>> model = torch.nn.Conv2d(1, 2, 1)
    >>> batch = (torch.rand(2, 1, 16, 16), torch.zeros(2, 2, 16, 16))
    >>> loss = torch.nn.functional.binary_cross_entropy_with_logits
    >>> metrics = seg.evaluate_loader(model, [batch], loss, "demo")
    demo ...
    >>> sorted(metrics)
    ['cross_mse', 'dna_dice', 'dna_iou', 'loss']

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
            totals["dna_dice"] += (
                float(
                    dice_score_from_prob(dna_prob, target[:, 0:1], threshold)
                )
                * n
            )
            totals["dna_iou"] += (
                float(
                    iou_score_from_prob(dna_prob, target[:, 0:1], threshold)
                )
                * n
            )
            totals["cross_mse"] += (
                float(F.mse_loss(cross_prob, target[:, 1:2])) * n
            )
            count += n
    metrics = {key: value / max(count, 1) for key, value in totals.items()}
    print(stage, {k: round(v, 4) for k, v in metrics.items()})
    return metrics


def inspect_crossing_batch(
    batch: tuple[torch.Tensor, torch.Tensor],
    model: torch.nn.Module,
) -> None:
    """Print how the crossing head responds at and away from crossings.

    For one batch, prints the largest crossing target and prediction, the
    number of crossing and DNA pixels, and the mean crossing probability
    at crossings, on ordinary DNA and on background. A good crossing head
    is high at crossings and low elsewhere.

    Parameters
    ----------
    batch: tuple[torch.Tensor, torch.Tensor]
        Images ``(B, 1, H, W)`` and targets ``(B, 2, H, W)``.
    model: torch.nn.Module
        The network.

    Returns
    -------
    None
        The statistics are printed.

    Examples
    --------
    >>> import torch
    >>> import unet_segmentation_helpers as seg

    >>> target = torch.zeros(1, 2, 8, 8)
    >>> target[0, 0, 2:6, 2:6] = 1
    >>> target[0, 1, 3:5, 3:5] = 1
    >>> seg.inspect_crossing_batch(
    ...     (torch.rand(1, 1, 8, 8), target), torch.nn.Conv2d(1, 2, 1)
    ... )  # doctest: +ELLIPSIS
    Crossing target max: 1.0
    ...

    """

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
    print(
        "Crossing pixels in batch:",
        int(positive.sum()),
        "| DNA pixels:",
        int(dna_region.sum()),
    )
    if int(positive.sum()):
        print(
            "Mean prediction at crossings:",
            round(float(cross_prob[positive].mean()), 3),
        )
    if int((dna_region & ~positive).sum()):
        print(
            "Mean prediction on non-crossing DNA:",
            round(float(cross_prob[dna_region & ~positive].mean()), 3),
        )
    if int((~dna_region).sum()):
        print(
            "Mean prediction on background:",
            round(float(cross_prob[~dna_region].mean()), 3),
        )


def show_prediction(
    batch: tuple[torch.Tensor, torch.Tensor],
    model: torch.nn.Module,
    index: int = 0,
) -> None:
    """Show one sample's input, targets and predictions.

    Five panels: the normalised input, the DNA target and prediction, and
    the crossing target and prediction (probabilities from 0 = black to
    1 = white).

    Parameters
    ----------
    batch: tuple[torch.Tensor, torch.Tensor]
        Images ``(B, 1, H, W)`` and targets ``(B, 2, H, W)``.
    model: torch.nn.Module
        The network.
    index: int, optional
        Sample of the batch to show. Defaults to 0.

    Returns
    -------
    None
        The figure is shown in the notebook.

    Examples
    --------
    >>> import torch
    >>> import unet_segmentation_helpers as seg

    >>> batch = (torch.rand(1, 1, 16, 16), torch.zeros(1, 2, 16, 16))
    >>> seg.show_prediction(batch, torch.nn.Conv2d(1, 2, 1))  # doctest: +SKIP

    """

    image, target = batch
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        pred = model(image.to(device))
        dna_prob = torch.sigmoid(pred[:, 0:1]).cpu()
        cross_prob = torch.sigmoid(pred[:, 1:2]).cpu()
    panels = [
        image[index, 0].cpu(),
        target[index, 0].cpu(),
        dna_prob[index, 0],
        target[index, 1].cpu(),
        cross_prob[index, 0],
    ]
    titles = [
        "input (normalised)",
        "DNA target",
        "DNA prediction",
        "crossing target",
        "crossing prediction",
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(17, 3.6))
    for ax, panel, title in zip(axes, panels, titles):
        if title.startswith("input"):
            ax.imshow(
                panel,
                cmap="afmhot",
                vmin=float(torch.quantile(panel.float(), 0.01)),
                vmax=float(torch.quantile(panel.float(), 0.995)),
            )
        else:
            ax.imshow(panel, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    show_figure(plt.gcf())


# ────────────────────────────────────────────────────────────────────────────
# Real data
# ────────────────────────────────────────────────────────────────────────────

DATA_TYPES = {
    "npy": (".npy",),
    "image": (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    "spm": (".spm",),
}
MASK_WORDS = ("mask", "label", "gt", "annotation", "seg")


def normalized_stem(path: str | Path) -> str:
    """Return a file name stem without image or mask words.

    ``image_12.npy`` and ``mask_12.npy`` both give ``"12"``, so images and
    their masks can be paired by name.

    Parameters
    ----------
    path: str | Path
        File path.

    Returns
    -------
    str
        The stem without "image", "img", "height", "mask", "label", "gt",
        "annotation" or "seg" and surrounding separators.

    Examples
    --------
    >>> import unet_segmentation_helpers as seg

    >>> seg.normalized_stem("image_12.npy")
    '12'
    >>> seg.normalized_stem("mask_12.npy")
    '12'

    """

    stem = Path(path).stem.lower()
    for word in ("image", "img", "height") + MASK_WORDS:
        stem = (
            stem.replace(f"{word}_", "")
            .replace(f"_{word}", "")
            .replace(word, "")
        )
    return stem.strip("_- ")


def collect_real_pairs(
    root: str | Path,
    data_type: Literal["npy", "image", "spm"],
) -> list[dict[str, Any]]:
    """Find the images of one type in a folder and pair them with masks.

    Files whose name contains a mask word ("mask", "label", "gt", ...)
    are treated as masks and paired with the image of the same
    :func:`normalized_stem`.

    Parameters
    ----------
    root: str | Path
        Folder to search (recursively).
    data_type: "npy" | "image" | "spm"
        Type of file to collect.

    Returns
    -------
    list[dict[str, Any]]
        One entry per image with ``id``, ``image_path`` and ``mask_path``
        (`None` without a mask).

    Raises
    ------
    ValueError
        If ``data_type`` is not "npy", "image" or "spm".

    Examples
    --------
    >>> import unet_segmentation_helpers as seg

    >>> pairs = seg.collect_real_pairs(
    ...     "../common-assets/real-test-data", "npy"
    ... )
    >>> len(pairs), all(p["mask_path"] is not None for p in pairs)
    (11, True)

    """

    if data_type not in DATA_TYPES:
        raise ValueError(f"data_type must be one of {list(DATA_TYPES)}")
    files = sorted(
        p
        for p in Path(root).rglob("*")
        if p.suffix.lower() in DATA_TYPES[data_type]
    )
    masks = {
        normalized_stem(p): p
        for p in files
        if any(w in p.stem.lower() for w in MASK_WORDS)
    }
    images = [
        p for p in files if not any(w in p.stem.lower() for w in MASK_WORDS)
    ]
    return [
        dict(
            id=normalized_stem(p),
            image_path=p,
            mask_path=masks.get(normalized_stem(p)),
        )
        for p in images
    ]


def load_real_height(
    path: str | Path,
    data_type: Literal["npy", "image"],
    image_height_range_nm: tuple[float, float] = (0.0, 4.0),
) -> np.ndarray:
    """Load a real height map in nm.

    ``.npy`` files are assumed to contain heights in nm. Image files
    contain grey values, which are mapped linearly from 0-255 (or 0-65535)
    onto ``image_height_range_nm``.

    Parameters
    ----------
    path: str | Path
        File to load.
    data_type: "npy" | "image"
        Type of the file (``.spm`` scans go through :func:`run_topostats`).
    image_height_range_nm: tuple[float, float], optional
        Heights represented by black and white in image files. Defaults to
        ``(0.0, 4.0)``.

    Returns
    -------
    np.ndarray
        Height map in nm (float32).

    Examples
    --------
    >>> import unet_segmentation_helpers as seg

    >>> height = seg.load_real_height(
    ...     "../common-assets/real-test-data/image_5.png", "image"
    ... )
    >>> float(height.min()) >= 0.0 and float(height.max()) <= 4.0
    True

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
    """Run the network on one height map at its native size.

    The map is prepared with :func:`prepare_height`, passed through the
    network, and both probability maps are restored to the input shape
    with :func:`restore_probability`.

    Parameters
    ----------
    model: torch.nn.Module
        Network returning DNA and crossing logits.
    height_nm: np.ndarray
        Height map in nm.
    bounds: tuple[float, float], optional
        Normalisation bounds in nm. Defaults to ``(-1.0, 5.0)``.
    target_size: int, optional
        Network input size in pixels. Defaults to 512.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        DNA and crossing probabilities with the shape of ``height_nm``.

    Examples
    --------
    >>> import numpy as np
    >>> import torch
    >>> import unet_segmentation_helpers as seg

    A 1x1 convolution standing in for the U-Net:

    >>> dna, cross = seg.predict_native(
    ...     torch.nn.Conv2d(1, 2, 1), np.zeros((300, 200)), target_size=64
    ... )
    >>> dna.shape, cross.shape
    ((300, 200), (300, 200))

    """

    device = next(model.parameters()).device
    tensor, geometry = prepare_height(height_nm, target_size, *bounds)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor.to(device)))[0].cpu().numpy()
    return restore_probability(prob[0], geometry), restore_probability(
        prob[1], geometry
    )


def measure_mask(
    mask: np.ndarray, nm_per_pixel: float | None = None
) -> dict[str, float]:
    """Basic measurements of a binary DNA mask.

    The skeleton length counts skeleton pixel steps, with diagonal steps
    counted as sqrt(2). It is a quick estimate of the total DNA length in
    the mask, not a TopoStats contour length.

    Parameters
    ----------
    mask: np.ndarray
        Binary mask.
    nm_per_pixel: float | None, optional
        Pixel size in nm, for a length in nm. Defaults to `None`.

    Returns
    -------
    dict[str, float]
        ``components``, ``area_px``, ``skeleton_length_px`` and, with a
        pixel size, ``skeleton_length_nm``.

    Examples
    --------
    >>> import numpy as np
    >>> import unet_segmentation_helpers as seg

    A straight line of 21 pixels has 20 steps:

    >>> mask = np.zeros((11, 31), bool)
    >>> mask[5, 5:26] = True
    >>> m = seg.measure_mask(mask, nm_per_pixel=0.5)
    >>> m["components"], m["area_px"], m["skeleton_length_nm"]
    (1, 21, 10.0)

    """

    from skimage.morphology import skeletonize

    mask = np.asarray(mask, bool)
    _, components = ndi.label(mask, structure=np.ones((3, 3)))
    skeleton = skeletonize(mask)
    straight = (
        np.logical_and(skeleton[:, 1:], skeleton[:, :-1]).sum()
        + np.logical_and(skeleton[1:, :], skeleton[:-1, :]).sum()
    )
    diagonal = (
        np.logical_and(skeleton[1:, 1:], skeleton[:-1, :-1]).sum()
        + np.logical_and(skeleton[1:, :-1], skeleton[:-1, 1:]).sum()
    )
    length_px = float(straight + np.sqrt(2.0) * diagonal)
    result = dict(
        components=int(components),
        area_px=int(mask.sum()),
        skeleton_length_px=length_px,
    )
    if nm_per_pixel:
        result["skeleton_length_nm"] = length_px * float(nm_per_pixel)
    return result


def evaluate_real_dataset(
    model: torch.nn.Module,
    pairs: list[dict[str, Any]],
    data_type: Literal["npy", "image"],
    threshold: float = 0.5,
    nm_per_pixel: float | None = None,
    crop_padding_px: int | None = 40,
    image_height_range_nm: tuple[float, float] = (0.0, 4.0),
) -> pd.DataFrame:
    """Segment every image of a real dataset and score it.

    Images with a mask are cropped to the molecule (mask bounding box plus
    ``crop_padding_px``, like a TopoStats grain crop) before segmentation;
    ``crop_padding_px=None`` segments the full images. Images without a
    mask are segmented and measured but not scored.

    Parameters
    ----------
    model: torch.nn.Module
        The network.
    pairs: list[dict[str, Any]]
        Output of :func:`collect_real_pairs`.
    data_type: "npy" | "image"
        Type of the files.
    threshold: float, optional
        DNA probability threshold. Defaults to 0.5.
    nm_per_pixel: float | None, optional
        Pixel size in nm for lengths in nm. Defaults to `None` (lengths in
        pixels).
    crop_padding_px: int | None, optional
        Crop margin in pixels. Defaults to 40.
    image_height_range_nm: tuple[float, float], optional
        Heights of black and white in image files. Defaults to
        ``(0.0, 4.0)``.

    Returns
    -------
    pd.DataFrame
        One row per image with Dice, IoU, precision, recall, the number of
        predicted pieces and the predicted and true skeleton lengths.

    Examples
    --------
    >>> import torch
    >>> import unet_segmentation_helpers as seg

    A 1x1 convolution standing in for the U-Net:

    >>> pairs = seg.collect_real_pairs(
    ...     "../common-assets/real-test-data", "npy"
    ... )
    >>> results = seg.evaluate_real_dataset(
    ...     torch.nn.Conv2d(1, 2, 1), pairs[:2], "npy", nm_per_pixel=0.5
    ... )
    >>> len(results), "dice" in results
    (2, True)

    """

    rows = []
    for pair in pairs:
        height = load_real_height(
            pair["image_path"], data_type, image_height_range_nm
        )
        truth = None
        if pair.get("mask_path") is not None:
            truth = load_array(pair["mask_path"]) > 0.5
            if crop_padding_px is not None:
                height, truth_f = crop_to_molecule(
                    [height, truth.astype(np.float32)], truth, crop_padding_px
                )
                truth = truth_f > 0.5
        dna_prob, _ = predict_native(model, height)
        pred = dna_prob > threshold
        row = dict(
            id=pair["id"],
            file=Path(pair["image_path"]).name,
            shape=f"{height.shape[0]}x{height.shape[1]}",
        )
        measured = measure_mask(pred, nm_per_pixel)
        row["pred_components"] = measured["components"]
        row["pred_length"] = measured.get(
            "skeleton_length_nm", measured["skeleton_length_px"]
        )
        if truth is not None:
            row.update(binary_scores(pred, truth))
            true_measure = measure_mask(truth, nm_per_pixel)
            row["true_length"] = true_measure.get(
                "skeleton_length_nm", true_measure["skeleton_length_px"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def show_real_dataset_results(
    results: pd.DataFrame,
    model: torch.nn.Module,
    pairs: list[dict[str, Any]],
    data_type: Literal["npy", "image"],
    threshold: float = 0.5,
    crop_padding_px: int | None = 40,
    n_examples: int = 4,
    length_unit: str = "nm",
    image_height_range_nm: tuple[float, float] = (0.0, 4.0),
) -> None:
    """Show the scores and example segmentations of a real dataset.

    Prints summary statistics, shows box plots of the scores, a plot of
    predicted against true DNA length, the full table, and overlays of the
    prediction (colour) and the true mask (red outline) for the first
    ``n_examples`` images.

    Parameters
    ----------
    results: pd.DataFrame
        Output of :func:`evaluate_real_dataset`.
    model: torch.nn.Module
        The network (used for the example overlays).
    pairs: list[dict[str, Any]]
        Output of :func:`collect_real_pairs`.
    data_type: "npy" | "image"
        Type of the files.
    threshold: float, optional
        DNA probability threshold. Defaults to 0.5.
    crop_padding_px: int | None, optional
        Crop margin in pixels. Defaults to 40.
    n_examples: int, optional
        Number of example overlays. Defaults to 4.
    length_unit: str, optional
        Unit shown on the length axes. Defaults to "nm".
    image_height_range_nm: tuple[float, float], optional
        Heights of black and white in image files. Defaults to
        ``(0.0, 4.0)``.

    Returns
    -------
    None
        Tables and figures are shown in the notebook.

    Examples
    --------
    >>> import torch
    >>> import unet_segmentation_helpers as seg

    >>> model = torch.nn.Conv2d(1, 2, 1)
    >>> pairs = seg.collect_real_pairs(
    ...     "../common-assets/real-test-data", "npy"
    ... )
    >>> results = seg.evaluate_real_dataset(model, pairs, "npy")
    >>> seg.show_real_dataset_results(
    ...     results, model, pairs, "npy"
    ... )  # doctest: +SKIP

    """

    scored = (
        results.dropna(subset=["dice"])
        if "dice" in results
        else results.iloc[0:0]
    )
    if len(scored):
        summary = scored[["dice", "iou", "precision", "recall"]].agg(
            ["mean", "std", "min", "max"]
        )
        print(f"Scored images: {len(scored)} of {len(results)}")
        display_df(summary.round(3))
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
        scored[["dice", "iou", "precision", "recall"]].plot.box(ax=axes[0])
        axes[0].set_title("Per-image scores")
        axes[0].set_ylim(0, 1.02)
        axes[1].scatter(scored["true_length"], scored["pred_length"])
        lim = [
            0,
            float(
                max(scored["true_length"].max(), scored["pred_length"].max())
                * 1.1
            ),
        ]
        axes[1].plot(lim, lim, "k--", linewidth=0.8)
        axes[1].set_xlabel(
            f"Skeleton length of the true mask ({length_unit})"
        )
        axes[1].set_ylabel(
            f"Skeleton length of the prediction ({length_unit})"
        )
        axes[1].set_title("Predicted vs true DNA length")
        plt.tight_layout()
        show_figure(plt.gcf())
    display_df(results.round(3))

    examples = pairs[:n_examples]
    fig, axes = plt.subplots(
        len(examples), 3, figsize=(12, 4 * len(examples)), squeeze=False
    )
    for row_axes, pair in zip(axes, examples):
        height = load_real_height(
            pair["image_path"], data_type, image_height_range_nm
        )
        truth = (
            load_array(pair["mask_path"]) > 0.5
            if pair.get("mask_path") is not None
            else None
        )
        if truth is not None and crop_padding_px is not None:
            height, truth_f = crop_to_molecule(
                [height, truth.astype(np.float32)], truth, crop_padding_px
            )
            truth = truth_f > 0.5
        dna_prob, cross_prob = predict_native(model, height)
        row_axes[0].imshow(height, cmap="afmhot", vmin=-0.5, vmax=4.0)
        row_axes[0].set_title(f"{Path(pair['image_path']).name}")
        row_axes[1].imshow(height, cmap="gray", vmin=-0.5, vmax=4.0)
        row_axes[1].imshow(
            np.ma.masked_where(dna_prob <= threshold, dna_prob),
            cmap="viridis",
            alpha=0.6,
            vmin=0,
            vmax=1,
        )
        if truth is not None:
            row_axes[1].contour(
                truth, levels=[0.5], colors="red", linewidths=0.6
            )
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
    data_type: Literal["npy", "image"],
    threshold: float = 0.5,
    nm_per_pixel: float | None = None,
    image_height_range_nm: tuple[float, float] = (0.0, 4.0),
) -> dict[str, Any]:
    """Segment one ``.npy`` or image file and measure the mask.

    Parameters
    ----------
    model: torch.nn.Module
        The network.
    path: str | Path
        The file.
    data_type: "npy" | "image"
        Type of the file.
    threshold: float, optional
        DNA probability threshold. Defaults to 0.5.
    nm_per_pixel: float | None, optional
        Pixel size in nm. Defaults to `None`.
    image_height_range_nm: tuple[float, float], optional
        Heights of black and white in image files. Defaults to
        ``(0.0, 4.0)``.

    Returns
    -------
    dict[str, Any]
        ``path``, ``height``, ``dna_prob``, ``cross_prob``, ``mask``,
        ``nm_per_pixel`` and ``measurements`` (see :func:`measure_mask`).

    Examples
    --------
    >>> import torch
    >>> import unet_segmentation_helpers as seg

    >>> result = seg.analyse_single_image(
    ...     torch.nn.Conv2d(1, 2, 1),
    ...     "../common-assets/real-test-data/image_5.npy", "npy",
    ... )
    >>> result["mask"].shape
    (512, 512)

    """

    height = load_real_height(path, data_type, image_height_range_nm)
    dna_prob, cross_prob = predict_native(model, height)
    mask = dna_prob > threshold
    return dict(
        path=Path(path),
        height=height,
        dna_prob=dna_prob,
        cross_prob=cross_prob,
        mask=mask,
        nm_per_pixel=nm_per_pixel,
        measurements=measure_mask(mask, nm_per_pixel),
    )


def show_single_image_result(result: dict[str, Any]) -> None:
    """Show the image, DNA segmentation and crossing map of one file.

    The DNA panel title gives the number of pieces and the skeleton
    length.

    Parameters
    ----------
    result: dict[str, Any]
        Output of :func:`analyse_single_image`.

    Returns
    -------
    None
        The figure is shown in the notebook.

    Examples
    --------
    >>> import torch
    >>> import unet_segmentation_helpers as seg

    >>> result = seg.analyse_single_image(
    ...     torch.nn.Conv2d(1, 2, 1),
    ...     "../common-assets/real-test-data/image_5.npy", "npy",
    ... )
    >>> seg.show_single_image_result(result)  # doctest: +SKIP

    """

    m = result["measurements"]
    unit = "nm" if "skeleton_length_nm" in m else "px"
    length = m.get("skeleton_length_nm", m["skeleton_length_px"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(result["height"], cmap="afmhot", vmin=-0.5, vmax=4.0)
    axes[0].set_title(result["path"].name)
    axes[1].imshow(result["height"], cmap="gray", vmin=-0.5, vmax=4.0)
    axes[1].imshow(
        np.ma.masked_where(~result["mask"], result["dna_prob"]),
        cmap="viridis",
        alpha=0.6,
        vmin=0,
        vmax=1,
    )
    axes[1].set_title(
        f"DNA mask: {m['components']} pieces, skeleton ≈ {length:.0f} {unit}"
    )
    axes[2].imshow(result["cross_prob"], cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("crossing probability")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    show_figure(plt.gcf())


def show_figure(fig: plt.Figure) -> None:
    """Display a Matplotlib figure in the notebook and close it.

    Displaying the figure object explicitly works whatever Matplotlib
    backend is active; some libraries (such as TopoStats' plotting
    set-up) can leave a backend behind after which ``plt.show()`` shows
    nothing. Outside a notebook, ``plt.show()`` is used.

    Parameters
    ----------
    fig: plt.Figure
        The figure.

    Returns
    -------
    None
        The figure is displayed.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import unet_segmentation_helpers as seg

    >>> fig, ax = plt.subplots()
    >>> seg.show_figure(fig)  # doctest: +SKIP

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
    """Display a table in the notebook (printed outside a notebook).

    Parameters
    ----------
    frame: pd.DataFrame
        The table.

    Returns
    -------
    None
        The table is displayed.

    Examples
    --------
    >>> import pandas as pd
    >>> import unet_segmentation_helpers as seg

    >>> seg.display_df(pd.DataFrame({"a": [1]}))  # doctest: +SKIP

    """

    try:
        from IPython.display import display

        display(frame)
    except ImportError:  # pragma: no cover
        print(frame.to_string())


# ────────────────────────────────────────────────────────────────────────────
# TopoStats with the trained U-Net
# ────────────────────────────────────────────────────────────────────────────

_ORIGINAL_PREDICT_UNET = None
_REAL_KERAS = None
_PATCH_INSTALLED = False
REQUIRED_TOPOSTATS = (2, 4, 2)
PYTORCH_SUFFIXES = (".pt", ".pth", ".ckpt")


def check_topostats_version() -> str:
    """Make sure TopoStats 2.4.2 (or a later 2.4.x) is installed.

    Older versions use a different configuration format and internals, so
    the presets in ``configs/`` and the U-Net connection would fail with
    confusing errors.

    Returns
    -------
    str
        The installed TopoStats version.

    Raises
    ------
    RuntimeError
        If TopoStats is missing or has another version; the message says
        how to install 2.4.2.

    Examples
    --------
    >>> import unet_segmentation_helpers as seg

    >>> seg.check_topostats_version()  # doctest: +SKIP
    '2.4.2'

    """

    import importlib.metadata
    import re

    try:
        version = importlib.metadata.version("topostats")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "TopoStats is not installed. Install it with: "
            "pip install topostats==2.4.2"
        ) from exc
    numbers = tuple(int(n) for n in re.findall(r"\d+", version)[:3])
    if numbers < REQUIRED_TOPOSTATS or numbers[:2] != REQUIRED_TOPOSTATS[:2]:
        raise RuntimeError(
            f"Section 15 needs TopoStats 2.4.2 but {version} is "
            "installed. Install it with: pip install topostats==2.4.2  "
            "(then restart the kernel)"
        )
    return version


def install_topostats_unet_patch(predictor_cls: type) -> None:
    """Let TopoStats use a PyTorch U-Net from the notebook.

    TopoStats' grain detection loads the network given by
    ``grains.unet_config.model_path`` (with Keras in the official release)
    and calls ``predict_unet`` on every grain crop. Both are intercepted
    in memory only:

    - a model path ending in ``.pt``, ``.pth`` or ``.ckpt`` is loaded with
      ``predictor_cls`` (defined in the notebook), which builds the right
      network from the checkpoint; other paths still go to Keras;
    - each grain crop is preprocessed exactly as during training, and its
      probability map is restored to the crop's pixels before the
      ``confidence`` threshold is applied.

    No installed file is changed; restarting the kernel undoes the patch.
    Installing it twice is harmless.

    Parameters
    ----------
    predictor_cls: type
        Class with ``__init__(model_path)``, ``input_shape`` and
        ``native_probability(image, lower_norm_bound, upper_norm_bound)``.

    Returns
    -------
    None
        TopoStats is patched for this session.

    Raises
    ------
    RuntimeError
        If TopoStats 2.4.2 is not installed (see
        :func:`check_topostats_version`).

    Examples
    --------
    >>> import unet_segmentation_helpers as seg

    >>> seg.install_topostats_unet_patch(TorchUNetPredictor)  # doctest: +SKIP

    """

    global _ORIGINAL_PREDICT_UNET, _REAL_KERAS, _PATCH_INSTALLED
    check_topostats_version()
    import topostats
    import topostats.grains as grains
    import topostats.unet_masking as masking

    # Official TopoStats: grains.improve_grain_segmentation_unet calls
    # keras.models.load_model.
    if _REAL_KERAS is None:
        _REAL_KERAS = grains.keras
    real_keras = _REAL_KERAS

    def load_model(path: str | Path, *args: Any, **kwargs: Any) -> Any:
        if str(path).lower().endswith(PYTORCH_SUFFIXES):
            return predictor_cls(path)
        return real_keras.models.load_model(path, *args, **kwargs)

    keras_models = types.ModuleType("keras.models")
    keras_models.__getattr__ = lambda name: getattr(real_keras.models, name)
    keras_models.load_model = load_model
    keras_proxy = types.ModuleType("keras")
    keras_proxy.__getattr__ = lambda name: getattr(real_keras, name)
    keras_proxy.models = keras_models
    grains.keras = keras_proxy

    # Builds whose grains.py imports a PyTorch adapter module instead.
    try:
        import topostats.torch_unet as adapter
    except ImportError:
        adapter = types.ModuleType("topostats.torch_unet")
        sys.modules["topostats.torch_unet"] = adapter
        topostats.torch_unet = adapter
    adapter.TorchUNetPredictor = predictor_cls

    if _ORIGINAL_PREDICT_UNET is None:
        _ORIGINAL_PREDICT_UNET = masking.predict_unet

    def predict_unet(
        image: np.ndarray,
        model: Any,
        confidence: float,
        model_input_shape: tuple[int | None, ...],
        upper_norm_bound: float,
        lower_norm_bound: float,
    ) -> np.ndarray:
        if not isinstance(model, predictor_cls):
            return _ORIGINAL_PREDICT_UNET(
                image,
                model,
                confidence,
                model_input_shape,
                upper_norm_bound,
                lower_norm_bound,
            )
        probability = model.native_probability(
            image,
            lower_norm_bound=lower_norm_bound,
            upper_norm_bound=upper_norm_bound,
        )
        foreground = probability > confidence
        return np.stack([~foreground, foreground], axis=-1)

    grains.predict_unet = predict_unet
    masking.predict_unet = predict_unet
    _PATCH_INSTALLED = True


def quiet_topostats() -> None:
    """Show only warnings and errors from TopoStats and its readers.

    Returns
    -------
    None
        The log levels are changed.

    Examples
    --------
    >>> import unet_segmentation_helpers as seg

    >>> seg.quiet_topostats()  # doctest: +SKIP

    """

    for name in ("topostats", "AFMReader"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        from loguru import logger as _loguru

        _loguru.remove()
        _loguru.add(sys.stderr, level="WARNING")
    except ImportError:
        pass


def load_topostats_config(yaml_path: str | Path) -> dict[str, Any]:
    """Read a TopoStats YAML configuration file.

    Parameters
    ----------
    yaml_path: str | Path
        The YAML file.

    Returns
    -------
    dict[str, Any]
        The configuration.

    Examples
    --------
    >>> import unet_segmentation_helpers as seg

    >>> config = seg.load_topostats_config("configs/topostats_relaxed.yaml")
    >>> config["grains"]["unet_config"]["confidence"]
    0.3

    """

    with open(yaml_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def update_nested(
    config: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Recursively update a nested configuration dictionary in place.

    Parameters
    ----------
    config: dict[str, Any]
        Configuration to update.
    overrides: dict[str, Any]
        New values; nested dictionaries update nested sections.

    Returns
    -------
    dict[str, Any]
        ``config`` (updated in place).

    Examples
    --------
    >>> import unet_segmentation_helpers as seg

    >>> config = {"grains": {"threshold": 0.7, "padding": 10}}
    >>> seg.update_nested(config, {"grains": {"padding": 30}})
    {'grains': {'threshold': 0.7, 'padding': 30}}

    """

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
    "remove_disconnected_grains": (
        "grains",
        "unet_config",
        "remove_disconnected_grains",
    ),
    "filter_threshold_method": ("filter", "threshold_method"),
    "filter_gaussian_size": ("filter", "gaussian_size"),
    "grain_threshold_nm": ("grains", "threshold_absolute"),
    "grain_area_thresholds_nm2": ("grains", "area_thresholds"),
    "grain_crop_padding": ("grains", "grain_crop_padding"),
    "remove_edge_intersecting_grains": (
        "grains",
        "remove_edge_intersecting_grains",
    ),
    "pruning_max_length_nm": (
        "disordered_tracing",
        "pruning_params",
        "max_length",
    ),
    "node_joining_length_nm": ("nodestats", "node_joining_length"),
    "node_extend_dist_nm": ("nodestats", "node_extend_dist"),
}


def exposed_topostats_settings(yaml_path: str | Path) -> dict[str, Any]:
    """Read the settings shown in the notebook from a TopoStats preset.

    Parameters
    ----------
    yaml_path: str | Path
        The preset (``configs/topostats_*.yaml``).

    Returns
    -------
    dict[str, Any]
        Notebook name to value, for every entry of
        ``TOPOSTATS_EXPOSED_SETTINGS``.

    Examples
    --------
    >>> import unet_segmentation_helpers as seg

    >>> settings = seg.exposed_topostats_settings(
    ...     "configs/topostats_minicircle.yaml"
    ... )
    >>> settings["confidence"], settings["grain_threshold_nm"]
    (0.5, [1.0])

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
    """Load a preset, apply the notebook's settings and complete it.

    Parameters
    ----------
    yaml_path: str | Path
        The preset (``configs/topostats_*.yaml``).
    settings: dict[str, Any]
        Notebook name to value (see ``TOPOSTATS_EXPOSED_SETTINGS``).
    spm_path: str | Path
        The scan to process.
    model_path: str | Path | None
        U-Net checkpoint; `None` uses TopoStats' height thresholding only.
    output_dir: str | Path
        Folder for TopoStats' output files.

    Returns
    -------
    dict[str, Any]
        Configuration ready for :func:`run_topostats`.

    Raises
    ------
    KeyError
        If ``settings`` contains a name that is not in
        ``TOPOSTATS_EXPOSED_SETTINGS``.
    RuntimeError
        If TopoStats 2.4.2 is not installed.

    Examples
    --------
    >>> import unet_segmentation_helpers as seg

    >>> preset = "configs/topostats_relaxed.yaml"
    >>> config = seg.build_topostats_config(
    ...     preset, seg.exposed_topostats_settings(preset),
    ...     "../common-assets/example-spm/scan.spm",
    ...     "checkpoints/md_res_unet.pt", "topostats_output",
    ... )  # doctest: +SKIP

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
    """Complete a TopoStats configuration for a run inside the notebook.

    Sets the input and output folders and the U-Net checkpoint, runs on
    one core, switches off TopoStats' own plot files (the notebook draws
    its own figures), validates the configuration against TopoStats'
    schema and adds the plotting dictionary that the command-line tool
    normally adds.

    Parameters
    ----------
    config: dict[str, Any]
        Configuration from :func:`load_topostats_config`.
    spm_path: str | Path
        The scan to process.
    model_path: str | Path | None
        U-Net checkpoint; `None` uses height thresholding only.
    output_dir: str | Path
        Folder for TopoStats' output files.

    Returns
    -------
    dict[str, Any]
        The validated configuration.

    Raises
    ------
    RuntimeError
        If TopoStats 2.4.2 is not installed.
    schema.SchemaError
        If the configuration does not fit TopoStats' schema.

    Examples
    --------
    >>> import unet_segmentation_helpers as seg

    >>> config = seg.prepare_topostats_config(
    ...     seg.load_topostats_config("configs/topostats_relaxed.yaml"),
    ...     "../common-assets/example-spm/scan.spm",
    ...     "checkpoints/md_res_unet.pt", "topostats_output",
    ... )  # doctest: +SKIP

    """

    from topostats.config import update_plotting_config
    from topostats.validation import DEFAULT_CONFIG_SCHEMA, validate_config

    check_topostats_version()
    quiet_topostats()
    config = dict(config)
    config["base_dir"] = Path(spm_path).resolve().parent
    config["output_dir"] = Path(output_dir).resolve()
    config["file_ext"] = Path(spm_path).suffix.lower()
    config["cores"] = 1
    config["grains"]["unet_config"]["model_path"] = (
        None if model_path is None else str(Path(model_path).resolve())
    )
    config["plotting"]["run"] = False
    validate_config(
        config,
        schema=DEFAULT_CONFIG_SCHEMA,
        config_type="YAML configuration file",
    )
    config["output_dir"].mkdir(parents=True, exist_ok=True)
    config["plotting"]["plot_dict"] = yaml.safe_load(
        (
            resources.files("topostats") / "plotting_dictionary.yaml"
        ).read_text()
    )
    config["plotting"] = update_plotting_config(config["plotting"])
    # Some TopoStats stages save diagnostic plots even when plotting is
    # off; the notebook draws its own figures, so no plot files are
    # written at all.
    for options in config["plotting"]["plot_dict"].values():
        options["save"] = False
        options["core_set"] = False
    return config


def run_topostats(
    spm_path: str | Path, config: dict[str, Any], quiet: bool = True
) -> dict[str, Any]:
    """Run the full TopoStats pipeline on one scan in the notebook.

    Filtering, grain detection (height threshold, then the U-Net on every
    grain crop), grain statistics, disordered tracing, node statistics
    (crossings), ordered tracing, splining and curvature.

    Parameters
    ----------
    spm_path: str | Path
        The ``.spm`` scan.
    config: dict[str, Any]
        Output of :func:`prepare_topostats_config` or
        :func:`build_topostats_config`.
    quiet: bool, optional
        Hide TopoStats' information messages. Defaults to `True`.

    Returns
    -------
    dict[str, Any]
        ``filename``, ``topostats`` (the processed TopoStats object), the
        data frames ``molecules``, ``grains``, ``image_stats``,
        ``disordered`` and ``branches``, and ``pixel_nm``.

    Raises
    ------
    RuntimeError
        If the configuration names a PyTorch checkpoint but
        :func:`install_topostats_unet_patch` has not been called.

    Examples
    --------
    >>> import unet_segmentation_helpers as seg

    >>> result = seg.run_topostats(spm_path, config)  # doctest: +SKIP
    >>> seg.molecule_table(result)  # doctest: +SKIP

    """

    from topostats.io import LoadScans
    from topostats.processing import process_scan

    model_path = config["grains"]["unet_config"].get("model_path")
    if (
        model_path
        and str(model_path).lower().endswith(PYTORCH_SUFFIXES)
        and not _PATCH_INSTALLED
    ):
        raise RuntimeError(
            "TopoStats cannot load a PyTorch checkpoint by itself. Run "
            "seg.install_topostats_unet_patch(TorchUNetPredictor) first "
            "(the section 15 cell does this)."
        )
    if quiet:
        quiet_topostats()

    # An absolute path keeps TopoStats' processed files inside
    # config["output_dir"].
    scans = LoadScans([Path(spm_path).resolve()], config=config)
    scans.get_data()
    scan = next(iter(scans.img_dict.values()))
    (
        filename,
        grain_df,
        topo,
        image_df,
        disordered_df,
        branch_df,
        molecule_df,
    ) = process_scan(
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
    if not getattr(topo, "grain_crops", None):
        print(
            f"TopoStats found no molecules in {Path(spm_path).name}. "
            "If an error is shown above, fix that first; otherwise try a "
            "lower grain_threshold_nm or smaller "
            "grain_area_thresholds_nm2, or another preset."
        )
    return dict(
        filename=filename,
        topostats=topo,
        molecules=molecule_df,
        grains=grain_df,
        image_stats=image_df,
        disordered=disordered_df,
        branches=branch_df,
        pixel_nm=float(topo.pixel_to_nm_scaling),
    )


MOLECULE_COLUMNS = [
    "grain",
    "molecule",
    "contour_length_nm",
    "circular",
    "crossings_in_grain",
]


def molecule_table(result: dict[str, Any]) -> pd.DataFrame:
    """Per-molecule contour lengths, circularity and crossings.

    Parameters
    ----------
    result: dict[str, Any]
        Output of :func:`run_topostats`.

    Returns
    -------
    pd.DataFrame
        One row per traced molecule with ``grain``, ``molecule``,
        ``contour_length_nm``, ``circular`` and ``crossings_in_grain``;
        empty (same columns) when nothing was traced.

    Examples
    --------
    >>> import types
    >>> import unet_segmentation_helpers as seg

    >>> empty = {"topostats": types.SimpleNamespace(grain_crops=None)}
    >>> table = seg.molecule_table(empty)
    >>> len(table), list(table.columns[:3])
    (0, ['grain', 'molecule', 'contour_length_nm'])

    """

    topo = result["topostats"]
    rows = []
    for grain_number, crop in (
        getattr(topo, "grain_crops", None) or {}
    ).items():
        trace = getattr(crop, "_ordered_trace", None) or getattr(
            crop, "ordered_trace", None
        )
        molecules = getattr(trace, "molecule_data", None) or {}
        nodes = (
            getattr(crop, "_nodes", None)
            or getattr(crop, "nodes", None)
            or {}
        )
        for molecule_number, molecule in molecules.items():
            length = getattr(molecule, "contour_length", None)
            rows.append(
                dict(
                    grain=int(grain_number),
                    molecule=int(molecule_number),
                    contour_length_nm=(
                        None if length is None else float(length) * 1e9
                    ),
                    circular=getattr(molecule, "circular", None),
                    crossings_in_grain=len(nodes),
                )
            )
    return pd.DataFrame(rows, columns=MOLECULE_COLUMNS)


def show_topostats_results(
    result: dict[str, Any], title: str = ""
) -> pd.DataFrame:
    """Show the filtered scan, the U-Net grain mask and the traces.

    Three panels: TopoStats' filtered image, the grain mask produced by
    the U-Net, and the ordered traces (splines) labelled with grain number
    and contour length; the molecule table is displayed below. The figure
    is drawn in Matplotlib's default style, because TopoStats changes the
    global style.

    Parameters
    ----------
    result: dict[str, Any]
        Output of :func:`run_topostats`.
    title: str, optional
        Figure title. Defaults to "".

    Returns
    -------
    pd.DataFrame
        The per-molecule table (see :func:`molecule_table`).

    Examples
    --------
    >>> import unet_segmentation_helpers as seg

    >>> table = seg.show_topostats_results(result, "scan")  # doctest: +SKIP

    """

    topo = result["topostats"]
    pixel_nm = result["pixel_nm"]
    image = np.asarray(topo.image)
    ny, nx = image.shape
    extent = (0, nx * pixel_nm, ny * pixel_nm, 0)
    mask = (
        np.asarray(topo.full_mask_tensor)[..., 1]
        if getattr(topo, "full_mask_tensor", None) is not None
        else None
    )

    # TopoStats applies its own Matplotlib style globally; draw this
    # figure in the default style.
    with plt.style.context("default"):
        return _draw_topostats_results(
            result, topo, pixel_nm, image, extent, mask, title
        )


def _draw_topostats_results(
    result: dict[str, Any],
    topo: Any,
    pixel_nm: float,
    image: np.ndarray,
    extent: tuple[float, float, float, float],
    mask: np.ndarray | None,
    title: str,
) -> pd.DataFrame:
    """Draw the three panels of :func:`show_topostats_results`.

    Parameters
    ----------
    result: dict[str, Any]
        Output of :func:`run_topostats`.
    topo: Any
        The processed TopoStats object.
    pixel_nm: float
        Pixel size in nm.
    image: np.ndarray
        Filtered height map.
    extent: tuple[float, float, float, float]
        Image extent in nm for ``imshow``.
    mask: np.ndarray | None
        Grain mask, or `None`.
    title: str
        Figure title.

    Returns
    -------
    pd.DataFrame
        The per-molecule table.

    Examples
    --------
    >>> import unet_segmentation_helpers as seg

    >>> seg._draw_topostats_results(
    ...     result, topo, 1.0, image, extent, mask, ""
    ... )  # doctest: +SKIP

    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    axes[0].imshow(image, cmap="afmhot", vmin=-1, vmax=4, extent=extent)
    axes[0].set_title("Filtered image (TopoStats)")
    axes[1].imshow(image, cmap="gray", vmin=-1, vmax=4, extent=extent)
    if mask is not None:
        axes[1].imshow(
            np.ma.masked_where(~mask, mask),
            cmap="autumn",
            alpha=0.55,
            extent=extent,
        )
    axes[1].set_title("Grain mask from the U-Net")
    axes[2].imshow(image, cmap="gray", vmin=-1, vmax=4, extent=extent)
    colours = plt.cm.tab10.colors
    table = molecule_table(result)
    for k, (grain_number, crop) in enumerate(
        (getattr(topo, "grain_crops", None) or {}).items()
    ):
        trace = getattr(crop, "_ordered_trace", None) or getattr(
            crop, "ordered_trace", None
        )
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
            label = f"G{grain_number}" + (
                f": {float(length) * 1e9:.0f} nm" if length else ""
            )
            axes[2].text(
                cols.mean(),
                max(rows.min() - 4 * pixel_nm, 0),
                label,
                color="white",
                fontsize=9,
                ha="center",
                va="bottom",
                clip_on=True,
                bbox=dict(
                    facecolor=colour, alpha=0.75, pad=1.5, edgecolor="none"
                ),
            )
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
