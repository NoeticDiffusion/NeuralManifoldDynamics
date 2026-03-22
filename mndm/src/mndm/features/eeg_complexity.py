"""EEG complexity metrics (entropy/fractal measures) for NDT ingest v1.2.

The module operates on continuous multi-channel EEG arrays. It aggregates
channels into a global montage (median across channels) before computing
windowed complexity metrics such as Sample Entropy, Permutation Entropy,
Lempel–Ziv Complexity, Multiscale Entropy, and Higuchi Fractal Dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

import numpy as np

try:
    import antropy as ant  # type: ignore
except Exception:  # pragma: no cover
    ant = None


@dataclass
class ComplexityConfig:
    metrics: Mapping[str, bool]
    window_length: int
    window_step: int
    max_windows: int | None
    summary_stats: Sequence[str]
    mse_scales: int


def compute_eeg_complexity_features(
    eeg_data: np.ndarray,
    sfreq: float,
    config: Mapping[str, object] | None,
) -> Dict[str, float]:
    """Compute windowed complexity metrics for EEG (global montage)."""

    if eeg_data.ndim != 2:
        raise ValueError("eeg_data must be 2-D (n_channels, n_samples)")
    if sfreq <= 0:
        raise ValueError("sfreq must be positive")

    cfg = _parse_config(config, sfreq)
    windows = _sliding_windows(eeg_data.shape[1], cfg.window_length, cfg.window_step)
    if windows and cfg.max_windows is not None and cfg.max_windows > 0:
        windows = _subsample_windows(windows, cfg.max_windows)
    if not windows:
        return {}

    # Global montage: median across channels to stay robust to drop-outs.
    global_trace = np.median(eeg_data, axis=0)

    per_metric_values: Dict[str, List[float]] = {}
    for window in windows:
        segment = global_trace[window]
        if segment.size < 16:
            continue

        if cfg.metrics.get("sample_entropy", False):
            per_metric_values.setdefault("sample_entropy", []).append(_sample_entropy(segment))

        if cfg.metrics.get("permutation_entropy", False):
            per_metric_values.setdefault("permutation_entropy", []).append(_permutation_entropy(segment))

        if cfg.metrics.get("lz_complexity", False):
            per_metric_values.setdefault("lz_complexity", []).append(_lz_complexity(segment))

        if cfg.metrics.get("multiscale_entropy", False):
            mse = _multiscale_entropy(segment, cfg.mse_scales)
            per_metric_values.setdefault("multiscale_entropy", []).append(mse)

        if cfg.metrics.get("higuchi_fd", False):
            per_metric_values.setdefault("higuchi_fd", []).append(_higuchi_fd(segment))

    features: Dict[str, float] = {}
    for metric_name, values in per_metric_values.items():
        series = np.asarray(values, dtype=np.float32)
        for stat in cfg.summary_stats:
            val = _reduce_stat(series, stat)
            if val is not None:
                features[f"eeg_complexity_{metric_name}_{stat}"] = val
    return features


def _parse_config(config: Mapping[str, object] | None, sfreq: float) -> ComplexityConfig:
    """Internal helper: parse config."""
    cfg = config or {}
    metrics = cfg.get("metrics", {}) if isinstance(cfg, Mapping) else {}
    windows_cfg = cfg.get("windows", {}) if isinstance(cfg, Mapping) else {}
    win_len_sec = float(windows_cfg.get("length_sec", 5.0))
    win_step_sec = float(windows_cfg.get("step_sec", 1.0))
    max_windows_raw = windows_cfg.get("max_windows", None) if isinstance(windows_cfg, Mapping) else None
    summary_cfg = cfg.get("summary_stats", ["mean", "std"])
    mse_scales = int(cfg.get("mse_scales", 5))

    window_length = max(int(round(win_len_sec * sfreq)), 1)
    window_step = max(int(round(win_step_sec * sfreq)), 1)
    if window_length < 2:
        raise ValueError("complexity window length too small")

    summary_stats = [str(s).lower() for s in summary_cfg] if summary_cfg else ["mean", "std"]
    max_windows = int(max_windows_raw) if isinstance(max_windows_raw, (int, float)) and int(max_windows_raw) > 0 else None
    return ComplexityConfig(
        metrics=metrics,
        window_length=window_length,
        window_step=window_step,
        max_windows=max_windows,
        summary_stats=summary_stats,
        mse_scales=max(mse_scales, 1),
    )


def _sliding_windows(n_samples: int, win_len: int, step: int) -> List[slice]:
    """Internal helper: sliding windows."""
    windows: List[slice] = []
    start = 0
    while start + win_len <= n_samples:
        windows.append(slice(start, start + win_len))
        start += step
    return windows


def _subsample_windows(windows: List[slice], max_windows: int) -> List[slice]:
    """Evenly subsample windows to cap runtime while preserving coverage."""
    if max_windows <= 0 or len(windows) <= max_windows:
        return windows
    idx = np.linspace(0, len(windows) - 1, num=max_windows, dtype=int)
    idx = np.unique(idx)
    return [windows[int(i)] for i in idx.tolist()]


def _sample_entropy(data: np.ndarray) -> float:
    """Internal helper: sample entropy."""
    if ant is not None:
        try:
            return float(ant.sample_entropy(data, order=2))
        except Exception:
            pass
    return float(np.nan)


def _permutation_entropy(data: np.ndarray) -> float:
    """Internal helper: permutation entropy."""
    if ant is not None:
        try:
            return float(ant.perm_entropy(data, order=3, normalize=True))
        except Exception:
            pass
    return float(np.nan)


def _lz_complexity(data: np.ndarray) -> float:
    """Internal helper: lz complexity."""
    if ant is not None:
        try:
            return float(ant.lziv_complexity(data, normalize=True))
        except Exception:
            pass
    return float(np.nan)


def _multiscale_entropy(data: np.ndarray, scales: int) -> float:
    """Internal helper: multiscale entropy."""
    if ant is not None:
        try:
            mse = ant.multiscale_entropy(data, scale=scales, maxscale=scales)
            return float(np.nanmean(mse))
        except Exception:
            pass
    return float(np.nan)


def _higuchi_fd(data: np.ndarray) -> float:
    """Internal helper: higuchi fd."""
    if ant is not None:
        try:
            return float(ant.higuchi_fd(data))
        except Exception:
            pass
    return float(np.nan)


def _reduce_stat(series: np.ndarray, stat: str) -> float | None:
    """Robust reduction helper that avoids NaN-only slices.

    Returns None when there are no finite values for the requested
    statistic so callers can skip emitting that feature instead of
    triggering NumPy RuntimeWarnings (e.g. mean of empty slice).
    """

    if series.size == 0:
        return None

    finite = np.isfinite(series)
    if not finite.any():
        return None

    vals = series[finite]

    if stat == "mean":
        return float(np.mean(vals))
    if stat == "std":
        # std over a single value is well-defined (=0.0) but if upstream
        # code ever changes ddof handling, keep this guard explicit.
        return float(np.std(vals))
    if stat == "median":
        return float(np.median(vals))
    if stat == "max":
        return float(np.max(vals))
    if stat == "min":
        return float(np.min(vals))
    return None

