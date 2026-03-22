"""
eeg_dfc.py
Dynamic functional connectivity metrics for EEG (sliding-window Pearson)."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


def compute_eeg_dynamic_fc_features(
    eeg_data: np.ndarray,
    sfreq: float,
    config: Mapping[str, object] | None,
) -> Tuple[Dict[str, float], np.ndarray | None]:
    """Return summary stats for sliding-window FC plus optional FC matrices."""

    if eeg_data.ndim != 2:
        raise ValueError("eeg_data must be 2-D (n_channels, n_samples)")
    if sfreq <= 0:
        raise ValueError("sfreq must be positive")

    cfg = config or {}
    win_len_sec = float(cfg.get("window_length_sec", 4.0))
    step_sec = float(cfg.get("step_sec", 0.5))
    max_windows = cfg.get("max_windows", None)
    summary_stats = cfg.get("summary", {"variance": True})
    retain_mats = bool(cfg.get("retain_matrices", False))

    win_len = max(int(round(win_len_sec * sfreq)), 1)
    step = max(int(round(step_sec * sfreq)), 1)
    windows = _sliding_windows(eeg_data.shape[1], win_len, step)
    if windows and isinstance(max_windows, (int, float)) and int(max_windows) > 0:
        windows = _subsample_windows(windows, int(max_windows))
    if not windows:
        return {}, None

    fc_means: List[float] = []
    fc_mats: List[np.ndarray] = []

    for window in windows:
        segment = eeg_data[:, window]
        if segment.shape[1] < 2:
            continue
        if retain_mats:
            fc = np.corrcoef(segment)
            # keep upper triangle without diagonal
            triu = fc[np.triu_indices_from(fc, k=1)]
            if triu.size == 0:
                continue
            fc_means.append(float(np.nanmean(triu)))
            fc_mats.append(fc.astype(np.float32))
        else:
            # Fast path: we only need the mean correlation (upper triangle mean),
            # so avoid building the full [C,C] matrix per window.
            mean_corr = _mean_upper_triangle_corr_fast(segment)
            if np.isfinite(mean_corr):
                fc_means.append(float(mean_corr))

    if not fc_means:
        return {}, None

    fc_means_arr = np.asarray(fc_means, dtype=np.float32)
    features: Dict[str, float] = {}
    if summary_stats.get("variance", True):
        features["eeg_dfc_variance"] = float(np.nanvar(fc_means_arr))
    if summary_stats.get("mean", False):
        features["eeg_dfc_mean"] = float(np.nanmean(fc_means_arr))
    if summary_stats.get("std", False):
        features["eeg_dfc_std"] = float(np.nanstd(fc_means_arr))
    if summary_stats.get("entropy", False):
        features["eeg_dfc_entropy"] = float(_approx_entropy(fc_means_arr))

    fc_stack = np.stack(fc_mats, axis=0) if (retain_mats and fc_mats) else None
    return features, fc_stack


def _sliding_windows(n_samples: int, win_len: int, step: int) -> List[slice]:
    """Internal helper: sliding windows."""
    windows: List[slice] = []
    start = 0
    while start + win_len <= n_samples:
        windows.append(slice(start, start + win_len))
        start += step
    return windows


def _subsample_windows(windows: List[slice], max_windows: int) -> List[slice]:
    """Evenly subsample a list of windows to cap runtime while preserving coverage."""
    if max_windows <= 0 or len(windows) <= max_windows:
        return windows
    idx = np.linspace(0, len(windows) - 1, num=max_windows, dtype=int)
    # ensure strictly increasing / unique
    idx = np.unique(idx)
    return [windows[int(i)] for i in idx.tolist()]


def _mean_upper_triangle_corr_fast(segment: np.ndarray) -> float:
    """Compute mean of the upper triangle (k=1) of corrcoef(segment) without forming the matrix.

    This matches:
        np.nanmean(np.corrcoef(segment)[np.triu_indices(C, k=1)])
    for typical EEG segments, but is much faster when retain_matrices is False.

    Notes:
    - Channels with zero/invalid std yield NaN correlations in np.corrcoef. We mimic
      np.nanmean behavior by excluding invalid channels from the computation.
    """
    if segment.ndim != 2:
        return float("nan")
    C, T = int(segment.shape[0]), int(segment.shape[1])
    if C < 2 or T < 2:
        return float("nan")

    # Match np.corrcoef normalization (ddof=1)
    mean = np.nanmean(segment, axis=1, keepdims=True)
    x = segment - mean
    std = np.nanstd(x, axis=1, ddof=1, keepdims=True)
    valid = np.isfinite(std[:, 0]) & (std[:, 0] > 0)
    if int(np.sum(valid)) < 2:
        return float("nan")
    x = x[valid, :]
    std = std[valid, :]
    C2 = int(x.shape[0])

    z = x / std
    # corr = (z @ z.T) / (T-1)
    # sum_all = sum_{i,j} corr_ij = sum_t (sum_i z_i,t)^2 / (T-1)
    s = np.sum(z, axis=0)
    sum_all = float(np.sum(s * s) / float(T - 1))
    # mean of off-diagonals: (sum_all - C) / (C*(C-1))
    # Here diagonal is ~1 per valid channel.
    denom = float(C2 * (C2 - 1))
    if denom <= 0:
        return float("nan")
    return (sum_all - float(C2)) / denom


def _approx_entropy(series: np.ndarray) -> float:
    # Simple discretised entropy estimate
    """Internal helper: approx entropy."""
    if series.size == 0:
        return float("nan")
    hist, _ = np.histogram(series[~np.isnan(series)], bins=20, density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return float("nan")
    return float(-np.sum(hist * np.log(hist)))

