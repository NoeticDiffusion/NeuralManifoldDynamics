"""
fmri_dynamic.py
Sliding-window connectivity metrics for fMRI."""

from __future__ import annotations

from typing import Dict, List, Mapping, Tuple

import numpy as np


def compute_dynamic_fc_features(
    roi_ts: np.ndarray,
    sfreq: float,
    config: Mapping[str, object] | None,
) -> Tuple[Dict[str, float], np.ndarray | None]:
    """Compute variance/entropy over sliding-window FC."""

    if roi_ts.ndim != 2:
        raise ValueError("roi_ts must be 2-D (n_rois, n_times)")
    if sfreq <= 0:
        raise ValueError("sfreq must be positive")

    cfg = config or {}
    win_len_sec = float(cfg.get("window_length_sec", 30.0))
    step_sec = float(cfg.get("step_sec", 2.0))
    summary_cfg = cfg.get("summary", {"variance": True, "entropy": True})
    retain = bool(cfg.get("retain_matrices", False))

    win_len = max(int(round(win_len_sec * sfreq)), 1)
    step = max(int(round(step_sec * sfreq)), 1)
    windows = _sliding_windows(roi_ts.shape[1], win_len, step)
    if not windows:
        return {}, None

    fc_values: List[float] = []
    fc_mats: List[np.ndarray] = []
    for window in windows:
        segment = roi_ts[:, window]
        if segment.shape[1] < 2:
            continue
        fc = np.corrcoef(segment)
        triu = fc[np.triu_indices_from(fc, k=1)]
        if triu.size == 0:
            continue
        fc_values.append(float(np.nanmean(triu)))
        if retain:
            fc_mats.append(fc.astype(np.float32))

    if not fc_values:
        return {}, None

    arr = np.asarray(fc_values, dtype=np.float32)
    features: Dict[str, float] = {}
    if summary_cfg.get("variance", True):
        features["fmri_dFC_variance"] = float(np.nanvar(arr))
    if summary_cfg.get("std", False):
        features["fmri_dFC_std"] = float(np.nanstd(arr))
    if summary_cfg.get("mean", False):
        features["fmri_dFC_mean"] = float(np.nanmean(arr))
    if summary_cfg.get("entropy", True):
        features["fmri_dFC_entropy"] = float(_approx_entropy(arr))

    stack = np.stack(fc_mats, axis=0) if (retain and fc_mats) else None
    return features, stack


def _sliding_windows(n_samples: int, win_len: int, step: int) -> List[slice]:
    windows: List[slice] = []
    start = 0
    while start + win_len <= n_samples:
        windows.append(slice(start, start + win_len))
        start += step
    return windows


def _approx_entropy(series: np.ndarray) -> float:
    hist, _ = np.histogram(series[~np.isnan(series)], bins=20, density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return float("nan")
    return float(-np.sum(hist * np.log(hist)))

