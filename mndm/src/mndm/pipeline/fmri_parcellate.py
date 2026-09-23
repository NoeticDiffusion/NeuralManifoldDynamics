"""Label-mean parcellation of 4D BOLD onto a 3D atlas (P1.2).

Live ds007216 gate (Schaefer-200, ``91×109×91×293``): the historical
per-label boolean mask + ``nanmean`` is faster than finite-weighted
``np.add.at`` (~1.8 s vs ~17.5 s) with ``max_abs_diff == 0`` on that
volume. Production therefore keeps the loop. Scatter-add remains as a
tested equivalent, not the hot path.

Semantics: background (label <= 0) is ignored, non-finite voxels do not
contribute to ``nanmean``, and an all-NaN ROI stays NaN.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


def parcellate_label_means_loop(
    bold_data: np.ndarray,
    atlas_data: np.ndarray,
    labels: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reference implementation: one boolean mask per label."""
    if bold_data.ndim != 4:
        raise ValueError(f"Expected 4D BOLD, got shape {bold_data.shape}")
    if atlas_data.shape != bold_data.shape[:3]:
        raise ValueError(
            f"Atlas shape {atlas_data.shape} does not match BOLD spatial "
            f"shape {bold_data.shape[:3]}"
        )
    n_times = int(bold_data.shape[3])
    if labels is None:
        uniq = np.unique(atlas_data)
        labels = uniq[uniq > 0].astype(int, copy=False)
    region_ts: List[np.ndarray] = []
    kept: List[int] = []
    for lab in labels:
        lab_int = int(lab)
        mask = atlas_data == lab_int
        if not np.any(mask):
            continue
        voxels = bold_data[mask, :]
        if voxels.ndim != 2 or voxels.shape[1] != n_times:
            voxels = voxels.reshape(-1, n_times)
        with np.errstate(all="ignore"):
            ts = np.nanmean(voxels, axis=0)
        region_ts.append(np.asarray(ts, dtype=np.float32))
        kept.append(lab_int)
    if not region_ts:
        return np.zeros((0,), dtype=np.int32), np.zeros((0, n_times), dtype=np.float32)
    return np.asarray(kept, dtype=np.int32), np.stack(region_ts, axis=0)


def parcellate_label_means(
    bold_data: np.ndarray,
    atlas_data: np.ndarray,
    labels: Optional[Iterable[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Production label means: boolean-mask ``nanmean`` (live-gate winner)."""
    return parcellate_label_means_loop(bold_data, atlas_data, labels=labels)


def parcellate_label_means_scatter(
    bold_data: np.ndarray,
    atlas_data: np.ndarray,
    labels: Optional[Iterable[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Finite-weighted scatter-add; equivalent totals, slower on Schaefer-scale 4D."""
    if bold_data.ndim != 4:
        raise ValueError(f"Expected 4D BOLD, got shape {bold_data.shape}")
    if atlas_data.shape != bold_data.shape[:3]:
        raise ValueError(
            f"Atlas shape {atlas_data.shape} does not match BOLD spatial "
            f"shape {bold_data.shape[:3]}"
        )
    n_times = int(bold_data.shape[3])
    atlas_int = np.maximum(np.asarray(atlas_data, dtype=np.int32).ravel(), 0)
    bold_flat = np.asarray(bold_data, dtype=np.float64).reshape(atlas_int.size, n_times)
    if labels is None:
        uniq = np.unique(atlas_int)
        label_ids = uniq[uniq > 0]
    else:
        label_ids = np.asarray(list(labels), dtype=np.int32)
        label_ids = label_ids[label_ids > 0]
    if label_ids.size == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0, n_times), dtype=np.float32)

    max_lab = int(max(int(atlas_int.max()) if atlas_int.size else 0, int(label_ids.max())))
    minlength = max_lab + 1
    sums = np.zeros((minlength, n_times), dtype=np.float64)
    counts = np.zeros((minlength, n_times), dtype=np.float64)
    finite = np.isfinite(bold_flat)
    weighted = np.where(finite, bold_flat, 0.0)
    finite_w = finite.astype(np.float64, copy=False)
    # Equivalent to per-volume bincount; one scatter-add over voxels × time.
    np.add.at(sums, atlas_int, weighted)
    np.add.at(counts, atlas_int, finite_w)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts
    means[counts <= 0] = np.nan

    present = label_ids[np.isin(label_ids, atlas_int)]
    if present.size == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0, n_times), dtype=np.float32)
    ts = np.asarray(means[present], dtype=np.float32)
    return np.asarray(present, dtype=np.int32), ts
