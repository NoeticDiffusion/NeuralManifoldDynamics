"""EEG synchrony features (coherence/PLV/PPC/PLI/wPLI/dPLI).

Operates on continuous preprocessed EEG arrays and produces summary statistics
per band and ROI pair as configured in the v1.2 ingest spec.

Typical inputs are ``eeg_data`` (channels × samples), ``sfreq``, channel
labels, and a ``config`` mapping with ``bands``, ``windows``, ``metrics``,
``roi_pairs``, and ``outputs``.

Returns a dict mapping feature names to floats (for example
``eeg_sync_alpha_FP_plv_mean``). Time-resolved tensors are not persisted here;
the ingest pipeline stores per-window tensors elsewhere.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
from scipy.signal import butter, coherence, filtfilt, hilbert

logger = logging.getLogger(__name__)

DEFAULT_BANDS = [
    {"name": "delta", "f_low": 1.0, "f_high": 4.0},
    {"name": "theta", "f_low": 4.0, "f_high": 8.0},
    {"name": "alpha", "f_low": 8.0, "f_high": 12.0},
    {"name": "beta", "f_low": 13.0, "f_high": 30.0},
    {"name": "gamma", "f_low": 30.0, "f_high": 80.0},
]


@dataclass
class PairDef:
    name: str
    idx1: int
    idx2: int


def compute_eeg_synchrony_features(
    eeg_data: np.ndarray,
    sfreq: float,
    channel_names: Sequence[str] | None,
    config: Mapping[str, object] | None,
) -> Dict[str, float]:
    """Return summary features for configured bands and ROI pairs."""

    if eeg_data.ndim != 2:
        raise ValueError("eeg_data must be 2-D (n_channels, n_samples)")
    if sfreq <= 0:
        raise ValueError("sfreq must be positive")

    cfg = config or {}
    bands = _parse_bands(cfg.get("bands"))
    windows_cfg = cfg.get("windows", {}) if isinstance(cfg, Mapping) else {}
    win_len_sec = float(windows_cfg.get("length_sec", 2.0))
    win_step_sec = float(windows_cfg.get("step_sec", 0.1))
    max_windows = windows_cfg.get("max_windows", None)
    metrics_cfg = cfg.get("metrics", {}) if isinstance(cfg, Mapping) else {}
    roi_pairs_cfg = cfg.get("roi_pairs", []) if isinstance(cfg, Mapping) else []
    outputs_cfg = cfg.get("outputs", {}) if isinstance(cfg, Mapping) else {}
    summary_stats = _parse_summary_stats(outputs_cfg.get("summary_stats"))

    if not roi_pairs_cfg:
        return {}
    if channel_names is None:
        raise ValueError("channel_names required to resolve synchrony ROI pairs")

    pairs = _resolve_pairs(roi_pairs_cfg, channel_names)
    if not pairs:
        return {}

    win_len = max(int(round(win_len_sec * sfreq)), 1)
    win_step = max(int(round(win_step_sec * sfreq)), 1)
    windows = _sliding_windows(eeg_data.shape[1], win_len, win_step)
    if windows and isinstance(max_windows, (int, float)) and int(max_windows) > 0:
        windows = _subsample_windows(windows, int(max_windows))
    if not windows:
        return {}

    features: Dict[str, float] = {}
    for band in bands:
        filtered = _bandpass(eeg_data, sfreq, band.f_low, band.f_high)
        phases = np.angle(hilbert(filtered, axis=1))

        for pair in pairs:
            values = _compute_metrics_for_pair(
                filtered[pair.idx1],
                filtered[pair.idx2],
                phases[pair.idx1],
                phases[pair.idx2],
                sfreq=sfreq,
                windows=windows,
                metrics=metrics_cfg,
            )
            for metric_name, series in values.items():
                key_prefix = f"eeg_sync_{band.name}_{pair.name}_{metric_name}"
                for stat in summary_stats:
                    val = _reduce_stat(series, stat)
                    if val is not None:
                        features[f"{key_prefix}_{stat}"] = val

    return features


@dataclass
class BandDef:
    name: str
    f_low: float
    f_high: float


def _parse_bands(bands_cfg) -> List[BandDef]:
    """Internal helper: parse bands."""
    bands_input = bands_cfg if isinstance(bands_cfg, Iterable) and bands_cfg else DEFAULT_BANDS
    bands: List[BandDef] = []
    for item in bands_input:
        try:
            name = str(item["name"])
            f_low = float(item["f_low"])
            f_high = float(item["f_high"])
        except Exception as exc:  # pragma: no cover - config errors
            raise ValueError(f"Invalid band definition: {item}") from exc
        if not (0.0 < f_low < f_high):
            raise ValueError(f"Band {name} has invalid limits ({f_low}, {f_high})")
        bands.append(BandDef(name=name, f_low=f_low, f_high=f_high))
    return bands


def _parse_summary_stats(stats_cfg) -> List[str]:
    """Internal helper: parse summary stats."""
    if isinstance(stats_cfg, Sequence) and stats_cfg:
        return [str(s).lower() for s in stats_cfg]
    return ["mean", "std"]


def _resolve_pairs(roi_pairs_cfg: Iterable[Mapping[str, object]], channel_names: Sequence[str]) -> List[PairDef]:
    """Internal helper: resolve pairs."""
    name_to_idx = {str(ch).upper(): idx for idx, ch in enumerate(channel_names)}
    pairs: List[PairDef] = []
    for entry in roi_pairs_cfg:
        try:
            name = str(entry["name"])
            chan1, chan2 = entry["channels"]
        except Exception as exc:
            raise ValueError(f"Invalid roi_pairs entry: {entry}") from exc
        idx1 = name_to_idx.get(str(chan1).upper())
        idx2 = name_to_idx.get(str(chan2).upper())
        if idx1 is None or idx2 is None:
            logger.warning("Skipping synchrony pair %s (missing %s/%s in montage)", name, chan1, chan2)
            continue
        pairs.append(PairDef(name=name, idx1=idx1, idx2=idx2))
    return pairs


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


def _bandpass(data: np.ndarray, sfreq: float, f_low: float, f_high: float, order: int = 4) -> np.ndarray:
    """Internal helper: bandpass."""
    nyq = sfreq / 2.0
    if not (0 < f_low < f_high < nyq):
        raise ValueError(f"Invalid bandpass limits ({f_low}, {f_high}) for sfreq={sfreq}")
    b, a = butter(order, [f_low / nyq, f_high / nyq], btype="band")
    return filtfilt(b, a, data, axis=1)


def _compute_metrics_for_pair(
    x_band: np.ndarray,
    y_band: np.ndarray,
    phase_x: np.ndarray,
    phase_y: np.ndarray,
    sfreq: float,
    windows: Sequence[slice],
    metrics: Mapping[str, object],
) -> Dict[str, np.ndarray]:
    """Internal helper: compute metrics for pair."""
    out: Dict[str, np.ndarray] = {}
    if not windows:
        return out

    # Assume regular windows from _sliding_windows: constant length, constant step
    win_len = int(windows[0].stop - windows[0].start)
    if win_len < 3:
        return out
    starts = np.asarray([int(w.start) for w in windows], dtype=np.int64)
    ends = starts + win_len
    # Guard against any pathological windows
    valid_w = (starts >= 0) & (ends <= int(x_band.shape[0]))
    if not bool(np.all(valid_w)):
        starts = starts[valid_w]
        ends = ends[valid_w]
        if starts.size == 0:
            return out

    # ---- Coherence (kept as-is; expensive but definition-preserving) ----
    if metrics.get("coherence", False):
        coh_vals: List[float] = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            x = x_band[s:e]
            y = y_band[s:e]
            coh_vals.append(_mean_coherence(x, y, sfreq))
        out["coh"] = np.asarray(coh_vals, dtype=np.float32)

    # ---- Phase-based metrics (vectorized via prefix sums) ----
    need_phase = bool(
        metrics.get("plv", False)
        or metrics.get("ppc", False)
        or metrics.get("pli", False)
        or metrics.get("wpli", False)
        or metrics.get("dpli", False)
    )
    if not need_phase:
        return out

    phase_diff = phase_x - phase_y

    def _window_sum(arr: np.ndarray) -> np.ndarray:
        # prefix sum trick: sum(arr[s:e]) = csum[e] - csum[s]
        """Internal helper: window sum."""
        csum = np.concatenate([np.asarray([0], dtype=arr.dtype), np.cumsum(arr, dtype=arr.dtype)])
        return csum[ends] - csum[starts]

    if metrics.get("plv", False) or metrics.get("ppc", False):
        u = np.exp(1j * phase_diff).astype(np.complex64, copy=False)
        sum_u = _window_sum(u)
        if metrics.get("plv", False):
            out["plv"] = (np.abs(sum_u) / float(win_len)).astype(np.float32)
        if metrics.get("ppc", False):
            # PPC = (|sum exp(i*phi)|^2 - N) / (N*(N-1))
            N = float(win_len)
            denom = (N * (N - 1.0)) + 1e-12
            ppc = ((np.abs(sum_u) ** 2) - N) / denom
            out["ppc"] = np.asarray(ppc, dtype=np.float32)

    if metrics.get("pli", False):
        sgn = np.sign(np.sin(phase_diff)).astype(np.float32, copy=False)
        mean_sgn = _window_sum(sgn) / float(win_len)
        out["pli"] = np.abs(mean_sgn).astype(np.float32)

    if metrics.get("wpli", False):
        im = np.sin(phase_diff).astype(np.float32, copy=False)
        num = np.abs(_window_sum(im) / float(win_len))
        denom = (_window_sum(np.abs(im)) / float(win_len)) + 1e-12
        out["wpli"] = (num / denom).astype(np.float32)

    if metrics.get("dpli", False):
        pos = (phase_diff > 0).astype(np.float32, copy=False)
        out["dpli"] = (_window_sum(pos) / float(win_len)).astype(np.float32)

    return out


def _mean_coherence(x: np.ndarray, y: np.ndarray, sfreq: float) -> float:
    """Internal helper: mean coherence."""
    nperseg = min(len(x), 256)
    if nperseg < 8:
        return float("nan")
    f, coh = coherence(x, y, fs=sfreq, nperseg=nperseg)
    return float(np.nanmean(coh))


def _ppc_unbiased(phase_diff: np.ndarray) -> float:
    # PPC formula using pairwise cosines:
    # PPC = (|sum exp(i*phi)|^2 - N) / (N*(N-1))
    """Internal helper: ppc unbiased."""
    complex_sum = np.sum(np.exp(1j * phase_diff))
    N = phase_diff.size
    if N < 2:
        return float("nan")
    numerator = np.abs(complex_sum) ** 2 - N
    denominator = N * (N - 1)
    return float(numerator / (denominator + 1e-12))


def _reduce_stat(series: np.ndarray, stat: str) -> float | None:
    """Internal helper: reduce stat."""
    if series.size == 0:
        return None
    if stat == "mean":
        return float(np.nanmean(series))
    if stat == "std":
        return float(np.nanstd(series))
    if stat == "median":
        return float(np.nanmedian(series))
    if stat == "max":
        return float(np.nanmax(series))
    if stat == "min":
        return float(np.nanmin(series))
    return None

