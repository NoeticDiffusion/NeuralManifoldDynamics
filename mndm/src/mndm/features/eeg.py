"""EEG feature extraction (bandpower, ratios, entropy, optional ensembles).

Consumes a preprocessed ``signals`` mapping (EEG arrays and metadata) and ingest
``config`` (epoching and ``features.eeg``); returns a per-epoch DataFrame
(``epoch_id``, timing, spectral features, etc.).
"""

from __future__ import annotations

import math
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
import time

from core import ensembles
from . import epoch_selection
from . import eeg_complexity
from . import eeg_dfc
from . import eeg_sync
from core.metrics import graph as graph_metrics
from .. import nwb_intervals
from ..reproducibility import resolve_component_seed

try:
    from mne.time_frequency import psd_array_multitaper
except Exception:  # pragma: no cover
    psd_array_multitaper = None  # type: ignore

logger = logging.getLogger(__name__)
_ENTROPY_FALLBACK_WARNED = False
_MULTITAPER_NONCONVERGENCE_WARNING_RE = r"Iterative multi-taper PSD computation did not converge.*"


def _run_multitaper_psd_safely(data: np.ndarray, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Run multitaper PSD while suppressing known non-convergence noise for EEG/iEEG.

    This warning can be frequent for clinical EEG/iEEG and does not always indicate
    a hard failure. Keep the behavior scoped to EEG feature extraction only so that
    non-EEG pipelines (e.g., fMRI) retain their own warning visibility.
    """

    if psd_array_multitaper is None:  # pragma: no cover - guarded by caller
        raise RuntimeError("psd_array_multitaper is unavailable")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=_MULTITAPER_NONCONVERGENCE_WARNING_RE,
            category=RuntimeWarning,
        )
        return psd_array_multitaper(data, **kwargs)


def _run_welch_psd_batched(
    data: np.ndarray,
    *,
    sfreq: float,
    fmin: float,
    fmax: float,
    nperseg: int,
    noverlap: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Welch PSD over all leading epoch/signal dimensions at once."""
    freqs, psd = signal.welch(
        data,
        fs=sfreq,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        axis=-1,
    )
    in_band = (freqs >= fmin) & (freqs <= fmax)
    return psd[..., in_band], freqs[in_band]


def _integrated_bandpower(psd_1d: np.ndarray, freqs_1d: np.ndarray, f_lo: float, f_hi: float) -> float:
    """Integrated power over [f_lo, f_hi] using trapezoidal integration."""
    mask = (freqs_1d >= float(f_lo)) & (freqs_1d <= float(f_hi))
    if not np.any(mask):
        return np.nan
    try:
        return float(np.trapezoid(psd_1d[mask], freqs_1d[mask]))
    except AttributeError:
        return float(np.trapz(psd_1d[mask], freqs_1d[mask]))


def _deep_merge_mapping(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge two config mappings."""
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge_mapping(
                dict(merged.get(key, {})),
                dict(value),
            )
        else:
            merged[key] = value
    return merged


def _resolve_conventional_eeg_cfg(config: Mapping[str, Any], dataset_id: Optional[str]) -> Dict[str, Any]:
    """Resolve conventional EEG config with optional dataset overrides."""
    root = config.get("conventional_eeg", {}) if isinstance(config, Mapping) else {}
    if not isinstance(root, Mapping):
        return {}
    merged: Dict[str, Any] = {k: root[k] for k in root if k != "datasets"}
    ds_map = root.get("datasets", {})
    if dataset_id and isinstance(ds_map, Mapping):
        ds_cfg = ds_map.get(dataset_id)
        if isinstance(ds_cfg, Mapping):
            merged = _deep_merge_mapping(merged, dict(ds_cfg))
    return merged


def _resolve_conventional_packs(conventional_cfg: Mapping[str, Any]) -> set[str]:
    """Return normalized conventional EEG pack names."""
    packs_raw = conventional_cfg.get("packs", ["tier1"]) if isinstance(conventional_cfg, Mapping) else ["tier1"]
    if isinstance(packs_raw, (str, bytes)):
        return {str(packs_raw).strip().lower()}
    if isinstance(packs_raw, list):
        return {str(v).strip().lower() for v in packs_raw if str(v).strip()}
    return {"tier1"}


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return a finite ratio or NaN when the denominator is not usable."""
    try:
        num = float(numerator)
        den = float(denominator)
    except Exception:
        return np.nan
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0:
        return np.nan
    return float(num / den)


def _cumulative_power_edge_frequency(psd_1d: np.ndarray, freqs_1d: np.ndarray, fraction: float) -> float:
    """Return the frequency where cumulative PSD reaches the requested fraction."""
    psd = np.asarray(psd_1d, dtype=float)
    freqs = np.asarray(freqs_1d, dtype=float)
    if psd.size == 0 or freqs.size == 0 or psd.size != freqs.size:
        return np.nan
    finite = np.isfinite(psd) & np.isfinite(freqs)
    psd = psd[finite]
    freqs = freqs[finite]
    if psd.size == 0 or freqs.size == 0:
        return np.nan
    if psd.size == 1:
        return float(freqs[0])
    psd = np.clip(psd, a_min=0.0, a_max=None)
    increments = 0.5 * (psd[1:] + psd[:-1]) * np.diff(freqs)
    cumulative = np.concatenate([[0.0], np.cumsum(increments)])
    total = float(cumulative[-1]) if cumulative.size else float("nan")
    if not np.isfinite(total) or total <= 0:
        return np.nan
    target = float(np.clip(fraction, 0.0, 1.0)) * total
    return float(np.interp(target, cumulative, freqs))


def _peak_frequency_in_band(
    psd_1d: np.ndarray,
    freqs_1d: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    """Return the dominant frequency within a target band."""
    psd = np.asarray(psd_1d, dtype=float)
    freqs = np.asarray(freqs_1d, dtype=float)
    if psd.size == 0 or freqs.size == 0 or psd.size != freqs.size:
        return np.nan
    mask = (
        np.isfinite(psd)
        & np.isfinite(freqs)
        & (freqs >= float(low_hz))
        & (freqs <= float(high_hz))
    )
    if not np.any(mask):
        return np.nan
    band_psd = psd[mask]
    band_freqs = freqs[mask]
    if band_psd.size == 0:
        return np.nan
    idx = int(np.argmax(band_psd))
    return float(band_freqs[idx])


def _compute_conventional_tier1_features(
    *,
    psd_1d: np.ndarray,
    freqs_1d: np.ndarray,
    bandpowers: Mapping[str, float],
    eeg_bands: Mapping[str, Any],
    conventional_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compute config-driven Tier 1 conventional EEG comparator features."""
    export_cfg = conventional_cfg.get("export", {}) if isinstance(conventional_cfg, Mapping) else {}
    per_epoch_columns = bool(export_cfg.get("per_epoch_columns", True))
    summaries = bool(export_cfg.get("summaries", True))
    if not (per_epoch_columns or summaries):
        return {}

    enabled = bool(conventional_cfg.get("enabled", False))
    packs = _resolve_conventional_packs(conventional_cfg)
    if not enabled or "tier1" not in packs:
        return {}

    tier1_cfg = conventional_cfg.get("tier1", {}) if isinstance(conventional_cfg, Mapping) else {}
    if not isinstance(tier1_cfg, Mapping):
        tier1_cfg = {}

    out: Dict[str, Any] = {}

    if bool(tier1_cfg.get("relative_bandpower", False)):
        finite_bandpowers = [
            float(value)
            for value in bandpowers.values()
            if np.isfinite(float(value))
        ]
        total_bandpower = float(np.sum(finite_bandpowers)) if finite_bandpowers else float("nan")
        for band_name, bandpower in bandpowers.items():
            out[f"eeg_conventional_relative_{band_name}"] = _safe_ratio(
                float(bandpower),
                total_bandpower,
            )

    ratio_specs: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
        "theta_alpha": (("theta",), ("alpha",)),
        "delta_alpha": (("delta",), ("alpha",)),
        "alpha_theta": (("alpha",), ("theta",)),
        "beta_alpha": (("beta",), ("alpha",)),
        "slowing_index": (("delta", "theta"), ("alpha", "beta")),
    }
    ratios_raw = tier1_cfg.get("ratios", [])
    ratio_names = [str(v).strip().lower() for v in ratios_raw] if isinstance(ratios_raw, list) else []
    for ratio_name in ratio_names:
        spec = ratio_specs.get(ratio_name)
        if spec is None:
            continue
        numerator = float(np.sum([float(bandpowers.get(name, np.nan)) for name in spec[0]]))
        denominator = float(np.sum([float(bandpowers.get(name, np.nan)) for name in spec[1]]))
        out[f"eeg_conventional_ratio_{ratio_name}"] = _safe_ratio(numerator, denominator)

    peak_cfg = tier1_cfg.get("peak_frequency", {})
    if not isinstance(peak_cfg, Mapping):
        peak_cfg = {}
    alpha_band = eeg_bands.get("alpha", [8, 12]) if isinstance(eeg_bands, Mapping) else [8, 12]
    alpha_low = float(alpha_band[0]) if isinstance(alpha_band, (list, tuple)) and len(alpha_band) >= 2 else 8.0
    alpha_high = float(alpha_band[1]) if isinstance(alpha_band, (list, tuple)) and len(alpha_band) >= 2 else 12.0

    if bool(peak_cfg.get("alpha_peak_frequency", False)):
        out["eeg_conventional_peak_alpha_frequency"] = _peak_frequency_in_band(
            psd_1d,
            freqs_1d,
            alpha_low,
            alpha_high,
        )
    if bool(peak_cfg.get("median_frequency", False)):
        out["eeg_conventional_peak_median_frequency"] = _cumulative_power_edge_frequency(
            psd_1d,
            freqs_1d,
            0.50,
        )
    if bool(peak_cfg.get("spectral_edge_95", False)):
        out["eeg_conventional_peak_spectral_edge_95"] = _cumulative_power_edge_frequency(
            psd_1d,
            freqs_1d,
            0.95,
        )
    return out


def _compute_conventional_complexity_features(
    *,
    epoch_signal: np.ndarray,
    sfreq: float,
    order: int,
    delay: int,
    normalize: bool,
    conventional_cfg: Mapping[str, Any],
    precomputed_hjorth: Optional[Tuple[float, float]] = None,
    precomputed_permutation_entropy: Optional[float] = None,
    permutation_entropy_degraded: bool = False,
) -> Dict[str, Any]:
    """Compute config-driven conventional EEG complexity comparator features."""
    export_cfg = conventional_cfg.get("export", {}) if isinstance(conventional_cfg, Mapping) else {}
    per_epoch_columns = bool(export_cfg.get("per_epoch_columns", True))
    summaries = bool(export_cfg.get("summaries", True))
    if not (per_epoch_columns or summaries):
        return {}

    enabled = bool(conventional_cfg.get("enabled", False))
    packs = _resolve_conventional_packs(conventional_cfg)
    if not enabled or "complexity" not in packs:
        return {}

    complexity_cfg = conventional_cfg.get("complexity", {}) if isinstance(conventional_cfg, Mapping) else {}
    if not isinstance(complexity_cfg, Mapping):
        complexity_cfg = {}

    out: Dict[str, Any] = {}
    if bool(complexity_cfg.get("spectral_entropy", False)):
        out["eeg_conventional_complexity_spectral_entropy"] = _compute_spectral_entropy(
            epoch_signal,
            sfreq=float(sfreq),
        )
    if bool(complexity_cfg.get("permutation_entropy", False)):
        if precomputed_permutation_entropy is not None and not permutation_entropy_degraded:
            out["eeg_conventional_complexity_permutation_entropy"] = precomputed_permutation_entropy
        else:
            out["eeg_conventional_complexity_permutation_entropy"] = _compute_permutation_entropy(
                epoch_signal,
                order=int(order),
                delay=int(delay),
                normalize=bool(normalize),
            )
    if precomputed_hjorth is None:
        hjorth_mobility, hjorth_complexity = _compute_hjorth_metrics(epoch_signal)
    else:
        hjorth_mobility, hjorth_complexity = precomputed_hjorth
    if bool(complexity_cfg.get("hjorth_complexity", False)):
        out["eeg_conventional_complexity_hjorth_complexity"] = hjorth_complexity
    if bool(complexity_cfg.get("hjorth_mobility", False)):
        out["eeg_conventional_complexity_hjorth_mobility"] = hjorth_mobility
    return out


def _compute_conventional_connectivity_features(
    *,
    eeg_data: np.ndarray,
    sfreq: float,
    eeg_channels: Optional[Sequence[str]],
    conventional_cfg: Mapping[str, Any],
) -> Dict[str, float]:
    """Compute config-driven conventional EEG connectivity comparator features."""
    export_cfg = conventional_cfg.get("export", {}) if isinstance(conventional_cfg, Mapping) else {}
    per_epoch_columns = bool(export_cfg.get("per_epoch_columns", True))
    summaries = bool(export_cfg.get("summaries", True))
    if not (per_epoch_columns or summaries):
        return {}

    enabled = bool(conventional_cfg.get("enabled", False))
    packs = _resolve_conventional_packs(conventional_cfg)
    if not enabled or "connectivity" not in packs:
        return {}
    if eeg_channels is None:
        return {}

    connectivity_cfg = conventional_cfg.get("connectivity", {}) if isinstance(conventional_cfg, Mapping) else {}
    if not isinstance(connectivity_cfg, Mapping):
        return {}

    sync_features = eeg_sync.compute_eeg_synchrony_features(
        eeg_data,
        sfreq=float(sfreq),
        channel_names=eeg_channels,
        config=connectivity_cfg,
    )
    out: Dict[str, float] = {}
    for key, value in sync_features.items():
        raw_key = str(key)
        if raw_key.startswith("eeg_sync_"):
            suffix = raw_key[len("eeg_sync_") :]
        else:
            suffix = raw_key
        out[f"eeg_conventional_connectivity_{suffix}"] = float(value)
    return out


def _moving_average_1d(values: np.ndarray, window: int) -> np.ndarray:
    """Return a same-length moving average for a 1-D array."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    w = max(1, int(window))
    if w <= 1:
        return arr.copy()
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(arr, kernel, mode="same")


def _compute_conventional_coma_features(
    *,
    epoch_signal: np.ndarray,
    sfreq: float,
    bandpowers: Mapping[str, float],
    conventional_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compute ICU/coma-oriented proxy features from a single EEG epoch."""
    export_cfg = conventional_cfg.get("export", {}) if isinstance(conventional_cfg, Mapping) else {}
    per_epoch_columns = bool(export_cfg.get("per_epoch_columns", True))
    summaries = bool(export_cfg.get("summaries", True))
    if not (per_epoch_columns or summaries):
        return {}

    enabled = bool(conventional_cfg.get("enabled", False))
    packs = _resolve_conventional_packs(conventional_cfg)
    if not enabled or "coma" not in packs:
        return {}

    coma_cfg = conventional_cfg.get("coma", {}) if isinstance(conventional_cfg, Mapping) else {}
    if not isinstance(coma_cfg, Mapping):
        coma_cfg = {}
    suppression_cfg = coma_cfg.get("suppression", {})
    if not isinstance(suppression_cfg, Mapping):
        suppression_cfg = {}

    def _float_or_default(value: Any, default: float) -> float:
        try:
            out = float(value)
        except Exception:
            out = float(default)
        return out if np.isfinite(out) else float(default)

    amplitude_floor = _float_or_default(suppression_cfg.get("amplitude_floor", 0.01), 0.01)
    relative_to_rms = _float_or_default(suppression_cfg.get("relative_to_rms", 0.20), 0.20)
    smoothing_sec = max(0.0, _float_or_default(suppression_cfg.get("envelope_smoothing_sec", 0.25), 0.25))
    smooth_window = int(round(float(sfreq) * smoothing_sec)) if np.isfinite(sfreq) and sfreq > 0 else 1
    smooth_window = max(1, smooth_window)

    arr = np.asarray(epoch_signal, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return {
            "eeg_conventional_coma_suppression_ratio": np.nan,
            "eeg_conventional_coma_continuity_proxy": np.nan,
            "eeg_conventional_coma_burst_suppression_proxy": np.nan,
            "eeg_conventional_coma_alpha_delta_ratio": np.nan,
        }

    x = arr[finite]
    rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else np.nan
    threshold = max(float(amplitude_floor), float(relative_to_rms) * float(rms)) if np.isfinite(rms) else float(amplitude_floor)

    envelope = _moving_average_1d(np.abs(x), smooth_window)
    suppression_mask = envelope <= threshold
    suppression_ratio = float(np.mean(suppression_mask)) if suppression_mask.size else np.nan

    out: Dict[str, Any] = {}
    if bool(coma_cfg.get("suppression_ratio", True)):
        out["eeg_conventional_coma_suppression_ratio"] = suppression_ratio
    if bool(coma_cfg.get("continuity_proxy", True)):
        out["eeg_conventional_coma_continuity_proxy"] = (
            1.0 - suppression_ratio if np.isfinite(suppression_ratio) else np.nan
        )
    if bool(coma_cfg.get("burst_suppression_proxy", True)):
        if suppression_mask.size <= 1 or not np.isfinite(suppression_ratio):
            burst_proxy = np.nan
        else:
            transitions = float(np.mean(suppression_mask[1:] != suppression_mask[:-1]))
            mix = float(4.0 * suppression_ratio * (1.0 - suppression_ratio))
            burst_proxy = float(np.clip(mix * transitions, 0.0, 1.0))
        out["eeg_conventional_coma_burst_suppression_proxy"] = burst_proxy
    if bool(coma_cfg.get("alpha_delta_ratio", True)):
        out["eeg_conventional_coma_alpha_delta_ratio"] = _safe_ratio(
            float(bandpowers.get("alpha", np.nan)),
            float(bandpowers.get("delta", np.nan)),
        )
    return out


def _compute_conventional_coma_reactivity_proxy(
    *,
    frame: pd.DataFrame,
    conventional_cfg: Mapping[str, Any],
) -> Optional[np.ndarray]:
    """Compute a deterministic epoch-to-epoch reactivity proxy series."""
    if frame.empty:
        return None

    export_cfg = conventional_cfg.get("export", {}) if isinstance(conventional_cfg, Mapping) else {}
    per_epoch_columns = bool(export_cfg.get("per_epoch_columns", True))
    summaries = bool(export_cfg.get("summaries", True))
    if not (per_epoch_columns or summaries):
        return None

    enabled = bool(conventional_cfg.get("enabled", False))
    packs = _resolve_conventional_packs(conventional_cfg)
    if not enabled or "coma" not in packs:
        return None

    coma_cfg = conventional_cfg.get("coma", {}) if isinstance(conventional_cfg, Mapping) else {}
    if not isinstance(coma_cfg, Mapping):
        coma_cfg = {}
    reactivity_cfg_raw = coma_cfg.get("reactivity_proxy", {})
    if isinstance(reactivity_cfg_raw, bool):
        if not reactivity_cfg_raw:
            return None
        reactivity_cfg: Mapping[str, Any] = {}
    elif isinstance(reactivity_cfg_raw, Mapping):
        reactivity_cfg = reactivity_cfg_raw
        if not bool(reactivity_cfg.get("enabled", True)):
            return None
    else:
        reactivity_cfg = {}

    default_features = [
        "eeg_conventional_coma_alpha_delta_ratio",
        "eeg_conventional_coma_continuity_proxy",
        "eeg_permutation_entropy",
        "eeg_highfreq_power_30_45",
    ]
    feature_cols_raw = reactivity_cfg.get("diff_features", default_features)
    if isinstance(feature_cols_raw, (str, bytes)):
        candidate_cols = [str(feature_cols_raw)]
    elif isinstance(feature_cols_raw, list):
        candidate_cols = [str(col).strip() for col in feature_cols_raw if str(col).strip()]
    else:
        candidate_cols = list(default_features)

    try:
        eps = float(reactivity_cfg.get("eps", 1.0e-6))
    except Exception:
        eps = 1.0e-6
    if not np.isfinite(eps) or eps <= 0:
        eps = 1.0e-6
    try:
        clip_at = float(reactivity_cfg.get("clip_at", 5.0))
    except Exception:
        clip_at = 5.0

    delta_series: list[np.ndarray] = []
    for col in candidate_cols:
        if col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        if int(np.sum(finite)) < 2:
            continue
        med = float(np.nanmedian(values))
        mad = float(np.nanmedian(np.abs(values - med)))
        scale = float(1.4826 * mad) if np.isfinite(mad) else np.nan
        if not np.isfinite(scale) or scale <= eps:
            std = float(np.nanstd(values))
            scale = std if np.isfinite(std) and std > eps else eps
        z = (values - med) / scale
        delta = np.abs(np.diff(z, prepend=z[0]))
        delta_series.append(delta)

    if not delta_series:
        return None

    stacked = np.vstack(delta_series)
    proxy = np.nanmean(stacked, axis=0)
    if np.isfinite(clip_at) and clip_at > 0:
        proxy = np.clip(proxy, 0.0, clip_at) / clip_at
    return proxy.astype(np.float32)


def _compute_hjorth_metrics(data: np.ndarray) -> Tuple[float, float]:
    """Compute Hjorth mobility and complexity for a 1D epoch signal."""
    x = np.asarray(data, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return np.nan, np.nan
    dx = np.diff(x)
    ddx = np.diff(dx)
    var_x = float(np.var(x))
    var_dx = float(np.var(dx)) if dx.size else np.nan
    var_ddx = float(np.var(ddx)) if ddx.size else np.nan
    if not np.isfinite(var_x) or var_x <= 0:
        return np.nan, np.nan
    if not np.isfinite(var_dx) or var_dx <= 0:
        return np.nan, np.nan
    mobility = float(np.sqrt(var_dx / var_x))
    if not np.isfinite(var_ddx) or var_ddx <= 0 or mobility <= 0:
        return mobility, np.nan
    complexity = float((np.sqrt(var_ddx / var_dx)) / mobility)
    return mobility, complexity


def _default_stage_map() -> Dict[str, int]:
    """Internal helper: default stage map."""
    return {"W": 0, "Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "R": 4, "L": -1}

def _resolve_epoching_sampling_cfg(config: Mapping[str, Any], dataset_id: Optional[str]) -> Dict[str, Any]:
    """Resolve epoch sampling config with optional per-dataset overrides."""
    epoching = config.get("epoching", {}) if isinstance(config, Mapping) else {}
    merged: Dict[str, Any] = {}
    if isinstance(epoching, Mapping):
        sampling = epoching.get("sampling", {})
        if isinstance(sampling, Mapping):
            merged.update(dict(sampling))
        if dataset_id:
            ds_map = epoching.get("datasets", {})
            if isinstance(ds_map, Mapping):
                ds_cfg = ds_map.get(dataset_id, {})
                if isinstance(ds_cfg, Mapping):
                    ds_sampling = ds_cfg.get("sampling", {})
                    if isinstance(ds_sampling, Mapping):
                        merged.update(dict(ds_sampling))
    repro_cfg = config.get("reproducibility", {}) if isinstance(config, Mapping) else {}
    if isinstance(repro_cfg, Mapping) and repro_cfg:
        merged["reproducibility"] = dict(repro_cfg)
    return merged


def _find_events_tsv_for_raw(file_path: str) -> Optional[Path]:
    """Best-effort locate BIDS *_events.tsv next to a raw file path."""
    if not file_path:
        return None
    try:
        p = Path(file_path)
    except Exception:
        return None
    parent = p.parent
    stem = p.stem
    base_core = stem
    if stem.endswith("_eeg"):
        base_core = stem[:-4]
    elif stem.endswith("_ieeg"):
        base_core = stem[:-5]
    candidate = parent / f"{base_core}_events.tsv"
    if candidate.exists():
        return candidate
    legacy = parent / "events.tsv"
    if legacy.exists():
        return legacy
    return None


def _resolve_sleep_annotation_cfg(config: Mapping[str, Any], dataset_id: Optional[str]) -> Dict[str, Any]:
    """Resolve optional sleep-annotation config with dataset overrides."""
    root = config.get("sleep_annotations", {}) if isinstance(config, Mapping) else {}
    if not isinstance(root, Mapping):
        return {}
    merged: Dict[str, Any] = {k: root[k] for k in root if k != "datasets"}
    ds_map = root.get("datasets", {})
    if dataset_id and isinstance(ds_map, Mapping):
        ds_cfg = ds_map.get(dataset_id)
        if isinstance(ds_cfg, Mapping):
            merged.update(dict(ds_cfg))
    return merged


def _find_sleep_annotation_for_raw(file_path: str, annotation_cfg: Mapping[str, Any]) -> Optional[Path]:
    """Best-effort locate a text sleep annotation sidecar near the raw file."""
    try:
        raw_path = Path(file_path)
    except Exception:
        return None

    if not raw_path.exists():
        return None

    if "annotation_file" in annotation_cfg and annotation_cfg.get("annotation_file"):
        fixed = raw_path.parent / str(annotation_cfg.get("annotation_file"))
        if fixed.exists():
            return fixed

    globs = annotation_cfg.get("file_globs", ["*.txt", "*.ann"])
    if not isinstance(globs, list) or not globs:
        globs = ["*.txt", "*.ann"]

    parent_levels = int(annotation_cfg.get("search_parent_levels", 1) or 1)
    parent_levels = max(0, min(parent_levels, 4))

    candidates: List[Path] = []
    dirs: List[Path] = [raw_path.parent]
    for i in range(1, parent_levels + 1):
        if i < len(raw_path.parents):
            dirs.append(raw_path.parents[i])
    # Preserve order while removing duplicates.
    dirs = list(dict.fromkeys(dirs))
    for d in dirs:
        for pattern in globs:
            try:
                candidates.extend([p for p in d.glob(str(pattern)) if p.is_file()])
            except Exception:
                continue

    if not candidates:
        return None

    # Stable de-duplication preserving lexical order.
    uniq = sorted({str(p.resolve()): p for p in candidates}.values(), key=lambda p: str(p))

    preferred = annotation_cfg.get("preferred_files", [])
    if isinstance(preferred, list):
        preferred_norm = {str(x).lower() for x in preferred}
        for p in uniq:
            if p.name.lower() in preferred_norm:
                return p

    raw_stem = raw_path.stem.lower()
    for p in uniq:
        if raw_stem and raw_stem in p.stem.lower():
            return p

    subj_hint = raw_path.parent.name.lower()
    for p in uniq:
        if subj_hint and subj_hint in p.stem.lower():
            return p

    return uniq[0]


def _label_epochs_from_sleep_annotation(
    epoch_meta: List[tuple[int, int, int]],
    sfreq: float,
    annotation_path: Path,
    stage_map: Mapping[str, int],
) -> Optional[np.ndarray]:
    """Label epochs using annotation text rows: <label> <start_sec> <duration_sec>."""
    rows: List[tuple[float, float, int]] = []
    try:
        with annotation_path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = str(raw).strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                label = str(parts[0]).strip()
                try:
                    onset = float(parts[1])
                    duration = float(parts[2])
                except Exception:
                    continue
                code = int(stage_map.get(label, stage_map.get(label.upper(), -1)))
                rows.append((onset, duration, code))
    except Exception:
        logger.exception("Failed reading sleep annotation file: %s", annotation_path)
        return None

    if not rows:
        return None

    t_mid = np.asarray([(s + e) * 0.5 / sfreq for (_, s, e) in epoch_meta], dtype=float)
    out = np.full((t_mid.shape[0],), -1, dtype=np.int16)
    for onset, duration, code in rows:
        end = onset + (duration if duration > 0 else 0.0)
        if end <= onset:
            mask = np.isclose(t_mid, onset, atol=1e-3)
        else:
            mask = (t_mid >= onset) & (t_mid < end)
        if mask.any():
            out[mask] = int(code)
    return out


def _label_epochs_with_stages(
    epoch_meta: List[tuple[int, int, int]],
    sfreq: float,
    events_df: pd.DataFrame,
    stage_columns: List[str],
    onset_column: str = "onset",
    duration_column: str = "duration",
    stage_map: Optional[Mapping[str, int]] = None,
) -> Optional[np.ndarray]:
    """Return per-epoch stage codes using BIDS events onset/duration + stage columns."""
    if events_df is None or events_df.empty:
        return None
    if onset_column not in events_df.columns:
        return None
    stage_col = None
    for c in stage_columns:
        if c in events_df.columns:
            stage_col = c
            break
    if stage_col is None:
        return None

    onset = pd.to_numeric(events_df[onset_column], errors="coerce").to_numpy(dtype=float)
    duration = (
        pd.to_numeric(events_df[duration_column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if duration_column in events_df.columns
        else np.zeros((len(events_df),), dtype=float)
    )
    stage_series = events_df[stage_col]
    stage_num = pd.to_numeric(stage_series, errors="coerce")
    if stage_map is None:
        stage_map = _default_stage_map()
    # Fill non-numeric stage values via explicit mapping (e.g. "N2", "REM", "Wake").
    if stage_num.isna().any():
        mapped = stage_series.astype(str).str.strip().map(
            lambda s: stage_map.get(s, stage_map.get(s.upper(), np.nan))
        )
        stage_num = stage_num.fillna(pd.to_numeric(mapped, errors="coerce"))
    stage = stage_num.to_numpy(dtype=float)
    valid = np.isfinite(onset) & np.isfinite(duration) & np.isfinite(stage)
    onset = onset[valid]
    duration = duration[valid]
    stage = stage[valid].astype(int, copy=False)
    if onset.size == 0:
        return None

    # Epoch midpoints (seconds)
    t_mid = np.asarray([(s + e) * 0.5 / sfreq for (_, s, e) in epoch_meta], dtype=float)
    out = np.full((t_mid.shape[0],), -1, dtype=np.int16)
    for o, d, s in zip(onset, duration, stage):
        end = o + (d if d > 0 else 0.0)
        if end <= o:
            mask = np.isclose(t_mid, o, atol=1e-3)
        else:
            mask = (t_mid >= o) & (t_mid < end)
        if mask.any():
            out[mask] = int(s)
    return out.astype(np.int16, copy=False)


def _parse_ds003838_task_state(label_text: str) -> tuple[Optional[str], Optional[str], Optional[float]]:
    """Infer ds003838 task/load labels from one event-row text blob."""
    text = str(label_text or "").strip().lower()
    if not text:
        return None, None, None
    if "rest" in text:
        return "rest", "rest", 0.0

    load = None
    for candidate in (13, 9, 5):
        if f"{candidate}" in text:
            load = float(candidate)
            break

    if any(token in text for token in ("listen", "encoding", "passive")):
        return "listen", (f"load_{int(load)}" if load is not None else "listen"), load
    if any(token in text for token in ("mem", "memory", "maintenance", "recall", "probe", "response")):
        if load is not None:
            return f"mem{int(load)}", f"load_{int(load)}", load
        return "memory", "memory", np.nan
    if load is not None:
        return f"mem{int(load)}", f"load_{int(load)}", load
    return None, None, None


def _label_epochs_for_ds003838_task(
    epoch_meta: List[tuple[int, int, int]],
    sfreq: float,
    events_df: pd.DataFrame,
    raw_file_path: Optional[str],
) -> Dict[str, np.ndarray]:
    """Infer ds003838 within-run task/load labels from BIDS events."""
    if events_df is None or events_df.empty or "onset" not in events_df.columns:
        return {}
    onset = pd.to_numeric(events_df["onset"], errors="coerce").to_numpy(dtype=float)
    duration = (
        pd.to_numeric(events_df["duration"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if "duration" in events_df.columns
        else np.zeros((len(events_df),), dtype=float)
    )
    text_cols = [col for col in events_df.columns if str(col).lower() not in {"onset", "duration", "sample"}]
    row_text = events_df[text_cols].astype(str).agg(" ".join, axis=1).tolist() if text_cols else [""] * len(events_df)
    midpoints = np.asarray([(start + end) * 0.5 / sfreq for (_, start, end) in epoch_meta], dtype=float)
    state = np.asarray([""] * len(epoch_meta), dtype=object)
    load_label = np.asarray([""] * len(epoch_meta), dtype=object)
    load_n = np.full((len(epoch_meta),), np.nan, dtype=np.float32)

    for idx, text in enumerate(row_text):
        state_label, load_label_value, load_numeric = _parse_ds003838_task_state(text)
        if state_label is None:
            continue
        start_sec = float(onset[idx]) if idx < len(onset) and np.isfinite(onset[idx]) else np.nan
        if not np.isfinite(start_sec):
            continue
        dur_sec = float(duration[idx]) if idx < len(duration) and np.isfinite(duration[idx]) else 0.0
        if dur_sec > 0:
            end_sec = start_sec + dur_sec
        elif idx + 1 < len(onset) and np.isfinite(onset[idx + 1]):
            end_sec = float(onset[idx + 1])
        else:
            end_sec = np.inf
        mask = (midpoints >= start_sec) & (midpoints < end_sec)
        if not np.any(mask):
            continue
        state[mask] = str(state_label)
        if load_label_value is not None:
            load_label[mask] = str(load_label_value)
        if load_numeric is not None:
            load_n[mask] = float(load_numeric)

    if raw_file_path and "task-rest" in str(raw_file_path).lower() and not np.any(state != ""):
        state[:] = "rest"
        load_label[:] = "rest"
        load_n[:] = 0.0

    out: Dict[str, np.ndarray] = {}
    if np.any(state != ""):
        out["task_state_label"] = state
    if np.any(load_label != ""):
        out["task_load_label"] = load_label
    if np.isfinite(load_n).any():
        out["task_load_n"] = load_n.astype(np.float32, copy=False)
    return out


def _contiguous_runs(indices: np.ndarray) -> List[tuple[int, int]]:
    """Return inclusive (start_idx, end_idx) runs over sorted integer indices."""
    if indices.size == 0:
        return []
    idx = np.sort(indices.astype(int))
    runs: List[tuple[int, int]] = []
    start = int(idx[0])
    prev = int(idx[0])
    for v in idx[1:]:
        v = int(v)
        if v == prev + 1:
            prev = v
            continue
        runs.append((start, prev))
        start = v
        prev = v
    runs.append((start, prev))
    return runs


def _select_stage_stratified_blocks(
    stage_per_epoch: np.ndarray,
    epoch_step_sec: float,
    sampling_cfg: Mapping[str, Any],
) -> Optional[np.ndarray]:
    """Select epoch indices via stage-stratified contiguous blocks."""
    if stage_per_epoch is None or stage_per_epoch.size == 0:
        return None

    target_minutes = sampling_cfg.get("target_minutes", {})
    if not isinstance(target_minutes, Mapping) or not target_minutes:
        return None
    block_minutes = float(sampling_cfg.get("block_minutes", 5) or 5.0)
    block_epochs = max(1, int(round((block_minutes * 60.0) / max(epoch_step_sec, 1e-6))))
    seed, _ = resolve_component_seed(
        {"reproducibility": sampling_cfg.get("reproducibility", {})} if isinstance(sampling_cfg, Mapping) else None,
        fallback_seed=sampling_cfg.get("seed"),
        fallback_source="epoching.sampling.seed",
    )
    rng = np.random.default_rng(seed)

    # Default stage mapping (ds005555 AASM codes)
    stage_code_map = {"Wake": 0, "W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}

    chosen: List[int] = []
    for stage_key, minutes in target_minutes.items():
        code = stage_code_map.get(str(stage_key), None)
        if code is None:
            try:
                code = int(stage_key)  # type: ignore[arg-type]
            except Exception:
                continue
        minutes_f = float(minutes or 0.0)
        if minutes_f <= 0:
            continue
        target_epochs = max(1, int(round((minutes_f * 60.0) / max(epoch_step_sec, 1e-6))))

        idxs = np.where(stage_per_epoch.astype(int) == int(code))[0]
        if idxs.size == 0:
            continue
        runs = _contiguous_runs(idxs)
        runs = sorted(runs, key=lambda r: (r[1] - r[0] + 1), reverse=True)

        need = target_epochs
        for (a, b) in runs:
            run_len = (b - a + 1)
            if run_len < block_epochs:
                continue
            max_blocks = max(1, need // block_epochs)
            max_blocks = min(max_blocks, (run_len // block_epochs))
            if max_blocks <= 1:
                start = a + (run_len - block_epochs) // 2
                chosen.extend(list(range(start, start + block_epochs)))
                need -= block_epochs
            else:
                span = run_len - block_epochs
                qs = np.linspace(0.0, 1.0, num=max_blocks, endpoint=True)
                starts = [a + int(round(q * span)) for q in qs]
                jitter = rng.integers(-1, 2, size=len(starts))
                for s0, j in zip(starts, jitter):
                    s1 = int(np.clip(int(s0) + int(j), a, b - block_epochs + 1))
                    chosen.extend(list(range(s1, s1 + block_epochs)))
                    need -= block_epochs
                    if need <= 0:
                        break
            if need <= 0:
                break

    if not chosen:
        return None
    return np.unique(np.asarray(chosen, dtype=int))



def compute_eeg_features(signals: Mapping[str, Any], config: Mapping[str, Any]) -> pd.DataFrame:
    """Compute per-epoch EEG features (with optional channel-shift ensembles).

    Args:
        signals: PreprocessedSignals or dict with ``signals`` and ``sfreq`` keys.
        config: Configuration with epoching and feature settings.

    Returns:
        Per-epoch features. Columns include global montage bands
        (``eeg_delta`` … ``eeg_gamma``), ratios, and entropy; optional per-group
        columns when ``robustness.ensembles`` is enabled, using column name
        patterns such as ``eeg_delta__g_frontal`` or ``eeg_alpha_theta__g_frontal``.
    """
    eeg_signals = signals.get("signals", {})
    if "eeg" not in eeg_signals:
        return pd.DataFrame()

    eeg_data = eeg_signals["eeg"]
    sfreq = signals.get("sfreq", 250)

    # Dataset / channel metadata (may be absent in tests or legacy callers)
    dataset_id = signals.get("dataset_id")
    raw_file_path = signals.get("file_path")
    events_df = None
    channels_map = signals.get("channels") if isinstance(signals, Mapping) else None
    eeg_channels = None
    if isinstance(channels_map, Mapping):
        eeg_channels = channels_map.get("eeg")

    # Resolve ensemble config (if any)
    robustness_cfg = config.get("robustness", {}) if isinstance(config, dict) else {}
    ensembles_cfg = robustness_cfg.get("ensembles", {}) if isinstance(robustness_cfg, dict) else {}
    group_defs: List[ensembles.EnsembleGroupDef] = []
    payload_meta = signals.get("meta", {}) if isinstance(signals, Mapping) else {}
    lfp_selection = payload_meta.get("lfp_channel_selection", {}) if isinstance(payload_meta, Mapping) else {}
    lfp_groups = lfp_selection.get("ensemble_groups", {}) if isinstance(lfp_selection, Mapping) else {}
    if isinstance(lfp_groups, Mapping) and lfp_groups and eeg_channels:
        try:
            group_defs = ensembles.realize_ensemble_groups({"groups": lfp_groups}, dataset_id, eeg_channels)
        except Exception:
            logger.exception("Failed to resolve LFP depth ensembles; continuing without ensembles")
    elif isinstance(ensembles_cfg, Mapping) and ensembles_cfg.get("enabled", False) and eeg_channels:
        try:
            group_defs = ensembles.realize_ensemble_groups(ensembles_cfg, dataset_id, eeg_channels)
        except Exception:
            # Fail-safe: ensembles are optional; never break core feature extraction.
            logger.exception("Failed to resolve ensemble groups; continuing without ensembles")
            group_defs = []

    # Get epoching parameters
    epoching = config.get("epoching", {}) if isinstance(config, dict) else {}
    length_s = epoching.get("length_s", 8.0)
    step_s = epoching.get("step_s", 4.0)
    # Optional per-dataset overrides: epoching.datasets.<dataset_id>.{length_s,step_s}
    if dataset_id and isinstance(epoching, Mapping):
        ds_map = epoching.get("datasets", {})
        if isinstance(ds_map, Mapping):
            ds_cfg = ds_map.get(dataset_id, {})
            if isinstance(ds_cfg, Mapping):
                if "length_s" in ds_cfg:
                    length_s = float(ds_cfg.get("length_s", length_s) or length_s)
                if "step_s" in ds_cfg:
                    step_s = float(ds_cfg.get("step_s", step_s) or step_s)

    # Get feature bands
    features_cfg = config.get("features", {}) if isinstance(config, dict) else {}
    eeg_bands = features_cfg.get(
        "eeg_bands",
        {"delta": [1, 4], "theta": [4, 8], "alpha": [8, 12], "beta": [13, 30], "gamma": [30, 45]},
    )
    conventional_cfg = _resolve_conventional_eeg_cfg(config, dataset_id)

    # PSD method config
    psd_cfg = features_cfg.get("eeg_psd", {}) if isinstance(features_cfg, dict) else {}
    psd_method = str(psd_cfg.get("method", "multitaper")).lower()
    mt_bandwidth = psd_cfg.get("bandwidth", None)
    psd_fmin = float(psd_cfg.get("fmin", 0.0) or 0.0)
    psd_fmax = float(psd_cfg.get("fmax", 50.0) or 50.0)
    nyquist = float(sfreq) * 0.5
    psd_fmax = min(psd_fmax, nyquist)
    if psd_fmax <= psd_fmin:
        psd_fmin = 0.0
        psd_fmax = nyquist

    # Optional PSD multiverse (secondary method for stability checks)
    multiverse_cfg = robustness_cfg.get("multiverse", {}) if isinstance(robustness_cfg, dict) else {}
    psd_mv_cfg = multiverse_cfg.get("psd", {}) if isinstance(multiverse_cfg, dict) else {}
    psd_mv_enabled = bool(psd_mv_cfg.get("enabled", False))
    psd_secondary_method = str(psd_mv_cfg.get("secondary_method", "welch")).lower()

    # Epoch the data
    epoch_length_samples = int(length_s * sfreq)
    epoch_step_samples = int(step_s * sfreq)
    n_samples = eeg_data.shape[1]
    n_epochs = (n_samples - epoch_length_samples) // epoch_step_samples + 1

    # Build epoch index metadata first (cheap), then optionally sample epochs
    # before slicing data (expensive).
    meta_all: List[tuple[int, int, int]] = []
    for epoch_idx in range(max(n_epochs, 0)):
        start_idx = epoch_idx * epoch_step_samples
        end_idx = start_idx + epoch_length_samples
        if end_idx > n_samples:
            break
        meta_all.append((epoch_idx, start_idx, end_idx))

    if not meta_all:
        return pd.DataFrame()

    # BDF BAD_ annotations are represented as NaNs in the preprocessing
    # payload. Reject whole windows touching them instead of concatenating
    # clean samples around an artifact gap into a synthetic epoch.
    bdf_meta = payload_meta.get("bdf_adapter", {}) if isinstance(payload_meta, Mapping) else {}
    bad_segments_meta = bdf_meta.get("bad_segments", {}) if isinstance(bdf_meta, Mapping) else {}
    if isinstance(bad_segments_meta, Mapping) and bad_segments_meta.get("enabled") and bad_segments_meta.get("available"):
        valid_samples = np.isfinite(eeg_data).all(axis=0)
        invalid_prefix = np.concatenate(([0], np.cumsum((~valid_samples).astype(np.int64))))
        original_count = len(meta_all)
        meta_all = [
            item
            for item in meta_all
            if int(invalid_prefix[item[2]] - invalid_prefix[item[1]]) == 0
        ]
        dropped = original_count - len(meta_all)
        if dropped:
            logger.info(
                "Rejected %d/%d EEG epochs touching BDF BAD_ segments for %s",
                dropped,
                original_count,
                Path(str(raw_file_path)).name if raw_file_path else "recording",
            )

    sampling_cfg = epoch_selection.resolve_epoching_sampling_cfg(config, dataset_id)
    chosen_meta: List[tuple[int, int, int]] = meta_all
    stage_per_epoch: Optional[np.ndarray] = None
    nwb_epoch_labels = nwb_intervals.NwbEpochLabels()
    if raw_file_path and str(raw_file_path).lower().endswith(".nwb"):
        try:
            nwb_epoch_labels = nwb_intervals.align_nwb_intervals_to_epochs(
                file_path=Path(str(raw_file_path)),
                config=config,
                dataset_id=dataset_id,
                epoch_meta=meta_all,
                sfreq=float(sfreq),
            )
            if nwb_epoch_labels.stage is not None:
                stage_per_epoch = nwb_epoch_labels.stage
        except Exception:
            logger.exception("NWB interval state-label extraction failed for %s", raw_file_path)
    if (
        isinstance(sampling_cfg, Mapping)
        and bool(sampling_cfg.get("enabled", False))
        and str(sampling_cfg.get("method", "")).lower() == "stage_stratified_blocks"
        and raw_file_path
    ):
        try:
            stage_map = epoch_selection.default_stage_map()
            cfg_stage_map = sampling_cfg.get("stage_map", {}) if isinstance(sampling_cfg, Mapping) else {}
            if isinstance(cfg_stage_map, Mapping):
                for k, v in cfg_stage_map.items():
                    try:
                        stage_map[str(k)] = int(v)
                    except Exception:
                        continue
            events_path = epoch_selection.find_events_tsv_for_raw(str(raw_file_path))
            if events_path is not None:
                events_df = pd.read_csv(events_path, sep="\t")
                stage_cols = sampling_cfg.get("stage_columns", ["stage_hum", "stage_ai", "stage"])
                stage_cols_list = (
                    [str(c) for c in stage_cols]
                    if isinstance(stage_cols, list)
                    else ["stage_hum", "stage_ai", "stage"]
                )
                stage_per_epoch = epoch_selection.label_epochs_with_stages(
                    epoch_meta=meta_all,
                    sfreq=float(sfreq),
                    events_df=events_df,
                    stage_columns=stage_cols_list,
                    onset_column=str(sampling_cfg.get("onset_column", "onset")),
                    duration_column=str(sampling_cfg.get("duration_column", "duration")),
                    stage_map=stage_map,
                    sampling_cfg=sampling_cfg if isinstance(sampling_cfg, Mapping) else None,
                )

            # Fallback for non-BIDS sleep datasets (e.g. ANPHY text annotations).
            if stage_per_epoch is None:
                ann_cfg = epoch_selection.resolve_sleep_annotation_cfg(config, dataset_id)
                if bool(ann_cfg.get("enabled", False)):
                    cfg_map = ann_cfg.get("stage_map", {})
                    if isinstance(cfg_map, Mapping):
                        for k, v in cfg_map.items():
                            try:
                                stage_map[str(k)] = int(v)
                            except Exception:
                                continue
                    ann_path = epoch_selection.find_sleep_annotation_for_raw(str(raw_file_path), ann_cfg)
                    if ann_path is not None:
                        stage_per_epoch = epoch_selection.label_epochs_from_sleep_annotation(
                            epoch_meta=meta_all,
                            sfreq=float(sfreq),
                            annotation_path=ann_path,
                            stage_map=stage_map,
                        )

            chosen_idx = epoch_selection.select_stage_stratified_blocks(
                stage_per_epoch=stage_per_epoch if stage_per_epoch is not None else np.asarray([], dtype=int),
                epoch_step_sec=float(step_s),
                sampling_cfg=sampling_cfg,
            )
            if chosen_idx is not None and chosen_idx.size:
                chosen_set = set(int(x) for x in chosen_idx.tolist())
                chosen_meta = [m for i, m in enumerate(meta_all) if i in chosen_set]
                logger.info(
                    "Stage-stratified sampling kept %d/%d epochs for %s",
                    len(chosen_meta),
                    len(meta_all),
                    Path(str(raw_file_path)).name,
                )
        except Exception:
            logger.exception("Stage-stratified epoch sampling failed; continuing with full recording")

    # If no stage labels were loaded during sampling, try once more so the
    # output can still expose a `stage` column for summarize.
    if stage_per_epoch is None and raw_file_path:
        try:
            stage_map = epoch_selection.default_stage_map()
            cfg_stage_map = sampling_cfg.get("stage_map", {}) if isinstance(sampling_cfg, Mapping) else {}
            if isinstance(cfg_stage_map, Mapping):
                for k, v in cfg_stage_map.items():
                    try:
                        stage_map[str(k)] = int(v)
                    except Exception:
                        continue
            # 1) BIDS events
            events_path = epoch_selection.find_events_tsv_for_raw(str(raw_file_path))
            if events_path is not None:
                events_df = pd.read_csv(events_path, sep="\t")
                stage_cols = (
                    sampling_cfg.get("stage_columns", ["stage_hum", "stage_ai", "stage", "sleep_stage"])
                    if isinstance(sampling_cfg, Mapping)
                    else ["stage_hum", "stage_ai", "stage", "sleep_stage"]
                )
                stage_cols_list = [str(c) for c in stage_cols] if isinstance(stage_cols, list) else ["stage_hum", "stage_ai", "stage", "sleep_stage"]
                stage_per_epoch = epoch_selection.label_epochs_with_stages(
                    epoch_meta=meta_all,
                    sfreq=float(sfreq),
                    events_df=events_df,
                    stage_columns=stage_cols_list,
                    onset_column=str(sampling_cfg.get("onset_column", "onset")) if isinstance(sampling_cfg, Mapping) else "onset",
                    duration_column=str(sampling_cfg.get("duration_column", "duration")) if isinstance(sampling_cfg, Mapping) else "duration",
                    stage_map=stage_map,
                    sampling_cfg=sampling_cfg if isinstance(sampling_cfg, Mapping) else None,
                )

            # 2) External text annotations
            if stage_per_epoch is None:
                ann_cfg = epoch_selection.resolve_sleep_annotation_cfg(config, dataset_id)
                if bool(ann_cfg.get("enabled", False)):
                    cfg_map = ann_cfg.get("stage_map", {})
                    if isinstance(cfg_map, Mapping):
                        for k, v in cfg_map.items():
                            try:
                                stage_map[str(k)] = int(v)
                            except Exception:
                                continue
                    ann_path = epoch_selection.find_sleep_annotation_for_raw(str(raw_file_path), ann_cfg)
                    if ann_path is not None:
                        stage_per_epoch = epoch_selection.label_epochs_from_sleep_annotation(
                            epoch_meta=meta_all,
                            sfreq=float(sfreq),
                            annotation_path=ann_path,
                            stage_map=stage_map,
                        )
        except Exception:
            logger.exception("Failed to infer per-epoch stages for %s", raw_file_path)

    # Precompute aggregated epochs for the selected meta:
    # - global montage: median across all EEG channels
    # - optional per-group montages for channel-shift ensembles
    epochs_agg: List[np.ndarray] = []
    meta: List[tuple[int, int, int]] = []
    for (epoch_idx, start_idx, end_idx) in chosen_meta:
        epoch_data = eeg_data[:, start_idx:end_idx]

        # Global montage
        epoch_series = [np.median(epoch_data, axis=0)]

        # Per-group montages (if any)
        for g in group_defs:
            try:
                group_epoch = epoch_data[g.indices, :]
                if group_epoch.size == 0:
                    continue
                epoch_series.append(np.median(group_epoch, axis=0))
            except Exception:
                logger.debug("Skipping ensemble group '%s' for epoch %s", g.name, epoch_idx)

        epochs_agg.append(np.stack(epoch_series, axis=0))  # [1 + G, T]
        meta.append((epoch_idx, start_idx, end_idx))

    if not epochs_agg:
        return pd.DataFrame()

    epochs_agg_arr = np.stack(epochs_agg, axis=0)  # [E, 1 + G, T]
    n_epochs_real, n_signals, _ = epochs_agg_arr.shape

    stage_by_epoch_id: Dict[int, int] = {}
    if stage_per_epoch is not None:
        try:
            for i_meta, (epoch_id_all, _, _) in enumerate(meta_all):
                if i_meta < len(stage_per_epoch):
                    stage_by_epoch_id[int(epoch_id_all)] = int(stage_per_epoch[i_meta])
        except Exception:
            stage_by_epoch_id = {}
    task_labels_by_epoch_id: Dict[str, Dict[int, Any]] = {}
    if str(dataset_id or "").strip().lower() == "ds003838" and events_df is not None:
        try:
            ds003838_labels = _label_epochs_for_ds003838_task(
                meta_all,
                float(sfreq),
                events_df,
                raw_file_path,
            )
            for label_name, values in ds003838_labels.items():
                arr = np.asarray(values, dtype=object if values.dtype.kind in {"U", "O"} else values.dtype)
                value_map: Dict[int, Any] = {}
                for i_meta, (epoch_id_all, _, _) in enumerate(meta_all):
                    if i_meta >= len(arr):
                        continue
                    raw_value = arr[i_meta]
                    if raw_value is None:
                        continue
                    if isinstance(raw_value, str) and raw_value == "":
                        continue
                    value_map[int(epoch_id_all)] = raw_value
                if value_map:
                    task_labels_by_epoch_id[str(label_name)] = value_map
        except Exception:
            logger.exception("Failed to infer ds003838 task/load labels for %s", raw_file_path)
    nwb_labels_by_epoch_id: Dict[str, Dict[int, Any]] = {}
    if nwb_epoch_labels.labels:
        for label_name, values in nwb_epoch_labels.labels.items():
            try:
                arr = np.asarray(values, dtype=object)
                value_map: Dict[int, Any] = {}
                for i_meta, (epoch_id_all, _, _) in enumerate(meta_all):
                    if i_meta >= len(arr):
                        continue
                    raw_value = arr[i_meta]
                    if raw_value is None:
                        continue
                    value = str(raw_value).strip()
                    if value:
                        value_map[int(epoch_id_all)] = value
                if value_map:
                    nwb_labels_by_epoch_id[str(label_name)] = value_map
            except Exception:
                logger.exception("Failed to align NWB label column '%s'", label_name)

    # Compute PSD in batch (primary method)
    t_psd0 = time.perf_counter()
    if psd_method == "multitaper" and psd_array_multitaper is not None:
        # psd: [E, 1 + G, F]
        psd, freqs = _run_multitaper_psd_safely(
            epochs_agg_arr,
            sfreq=sfreq,
            fmin=psd_fmin,
            fmax=psd_fmax,
            bandwidth=mt_bandwidth,
            adaptive=True,
            normalization="full",
            verbose=False,
        )
    else:
        # Fallback to Welch over all epochs and montages in one call.
        nperseg = min(epoch_length_samples, 512)
        noverlap = nperseg // 2
        psd, freqs = _run_welch_psd_batched(
            epochs_agg_arr,
            sfreq=float(sfreq),
            fmin=float(psd_fmin),
            fmax=float(psd_fmax),
            nperseg=nperseg,
            noverlap=noverlap,
        )

    if freqs is None:
        # Should not happen, but guard against edge cases
        return pd.DataFrame()
    t_psd1 = time.perf_counter()

    # Optional secondary PSD for multiverse (global montage only)
    psd_alt = None
    freqs_alt = None
    if psd_mv_enabled:
        epochs_global = epochs_agg_arr[:, 0, :]  # [E, T]
        if psd_secondary_method == "multitaper" and psd_array_multitaper is not None:
            try:
                psd_alt, freqs_alt = _run_multitaper_psd_safely(
                    epochs_global,
                    sfreq=sfreq,
                    fmin=psd_fmin,
                    fmax=psd_fmax,
                    bandwidth=mt_bandwidth,
                    adaptive=True,
                    normalization="full",
                    verbose=False,
                )
            except Exception:
                logger.exception("Secondary PSD (multitaper) failed; disabling PSD multiverse for this run")
                psd_alt = None
                freqs_alt = None
        elif psd_secondary_method == "welch":
            try:
                nperseg_alt = min(epoch_length_samples, 512)
                noverlap_alt = nperseg_alt // 2
                psd_alt, freqs_alt = _run_welch_psd_batched(
                    epochs_global,
                    sfreq=float(sfreq),
                    fmin=float(psd_fmin),
                    fmax=float(psd_fmax),
                    nperseg=nperseg_alt,
                    noverlap=noverlap_alt,
                )
            except Exception:
                logger.exception("Secondary PSD (welch) failed; disabling PSD multiverse for this run")
                psd_alt = None
                freqs_alt = None
        else:
            logger.warning("Unsupported secondary PSD method '%s'; disabling PSD multiverse", psd_secondary_method)
            psd_alt = None
            freqs_alt = None

    # Build feature rows
    pe_cfg = features_cfg.get("permutation_entropy", {}) if isinstance(features_cfg, Mapping) else {}
    pe_order = int(pe_cfg.get("order", 5))
    pe_delay = int(pe_cfg.get("delay", 1))
    pe_normalize = bool(pe_cfg.get("normalize", True))

    records: List[Dict[str, Any]] = []
    ratios = features_cfg.get("ratios", {})

    for i, (epoch_id, start_idx, end_idx) in enumerate(meta):
        features_dict: Dict[str, Any] = {
            "epoch_id": epoch_id,
            "t_start": start_idx / sfreq,
            "t_end": end_idx / sfreq,
        }
        if stage_by_epoch_id:
            features_dict["stage"] = int(stage_by_epoch_id.get(int(epoch_id), -1))
        for label_name, value_map in nwb_labels_by_epoch_id.items():
            value = value_map.get(int(epoch_id))
            if value is not None:
                features_dict[label_name] = value
        for label_name, value_map in task_labels_by_epoch_id.items():
            value = value_map.get(int(epoch_id))
            if value is not None:
                features_dict[label_name] = value

        # --- Global montage (index 0) ---
        global_bandpowers: Dict[str, float] = {}
        for band_name, (low, high) in eeg_bands.items():
            lo = max(float(low), float(psd_fmin))
            hi = min(float(high), float(psd_fmax))
            bandpower = _integrated_bandpower(psd[i, 0, :], freqs, lo, hi)
            features_dict[f"eeg_{band_name}"] = bandpower
            global_bandpowers[str(band_name)] = bandpower
        # Explicit high-frequency power proxy used by embodied fallback policy.
        features_dict["eeg_highfreq_power_30_45"] = _integrated_bandpower(
            psd[i, 0, :], freqs, max(30.0, float(psd_fmin)), min(45.0, float(psd_fmax))
        )

        if "alpha_theta" in ratios:
            alpha = features_dict.get("eeg_alpha", np.nan)
            theta = features_dict.get("eeg_theta", np.nan)
            features_dict["eeg_alpha_theta"] = (
                alpha / theta if (theta is not None and isinstance(theta, (int, float)) and theta > 0) else np.nan
            )
        if "beta_alpha" in ratios:
            beta = features_dict.get("eeg_beta", np.nan)
            alpha = features_dict.get("eeg_alpha", np.nan)
            features_dict["eeg_beta_alpha"] = (
                beta / alpha if (alpha is not None and isinstance(alpha, (int, float)) and alpha > 0) else np.nan
            )
        hj_mob, hj_comp = _compute_hjorth_metrics(epochs_agg_arr[i, 0])
        features_dict["eeg_hjorth_mobility"] = hj_mob
        features_dict["eeg_hjorth_complexity"] = hj_comp

        # Energetic complexity metric for global montage:
        # permutation entropy (primary) with deterministic spectral fallback.
        entropy_value, entropy_meta = _compute_entropy_feature(
            epochs_agg_arr[i, 0],
            sfreq=float(sfreq),
            order=pe_order,
            delay=pe_delay,
            normalize=pe_normalize,
        )
        features_dict["eeg_permutation_entropy"] = entropy_value
        # Backward-compatible alias used by legacy direct weights.
        features_dict["eeg_sample_entropy"] = entropy_value
        features_dict["eeg_entropy_construct"] = str(entropy_meta.get("construct", "energetic_complexity"))
        features_dict["eeg_entropy_metric"] = str(entropy_meta.get("metric", "permutation_entropy"))
        features_dict["eeg_entropy_backend"] = str(entropy_meta.get("backend", "numpy"))
        features_dict["eeg_entropy_degraded_mode"] = bool(entropy_meta.get("degraded_mode", False))
        features_dict["eeg_entropy_reason"] = entropy_meta.get("reason")

        conventional_features = _compute_conventional_tier1_features(
            psd_1d=psd[i, 0, :],
            freqs_1d=freqs,
            bandpowers=global_bandpowers,
            eeg_bands=eeg_bands if isinstance(eeg_bands, Mapping) else {},
            conventional_cfg=conventional_cfg,
        )
        conventional_features.update(
            _compute_conventional_complexity_features(
                epoch_signal=epochs_agg_arr[i, 0],
                sfreq=float(sfreq),
                order=pe_order,
                delay=pe_delay,
                normalize=pe_normalize,
                conventional_cfg=conventional_cfg,
                precomputed_hjorth=(hj_mob, hj_comp),
                precomputed_permutation_entropy=entropy_value,
                permutation_entropy_degraded=bool(entropy_meta.get("degraded_mode", False)),
            )
        )
        conventional_features.update(
            _compute_conventional_coma_features(
                epoch_signal=epochs_agg_arr[i, 0],
                sfreq=float(sfreq),
                bandpowers=global_bandpowers,
                conventional_cfg=conventional_cfg,
            )
        )
        if conventional_features:
            features_dict.update(conventional_features)

        # QC flag for EEG epoch (no NaNs in core EEG bands)
        core_cols = ["eeg_delta", "eeg_theta", "eeg_alpha", "eeg_beta", "eeg_gamma"]
        features_dict["qc_ok_eeg"] = bool(all(not np.isnan(features_dict.get(c, np.nan)) for c in core_cols))

        # --- Optional secondary PSD features for multiverse (global montage only) ---
        if psd_mv_enabled and psd_alt is not None and freqs_alt is not None:
            band_alt: Dict[str, float] = {}
            for band_name, (low, high) in eeg_bands.items():
                lo = max(float(low), float(psd_fmin))
                hi = min(float(high), float(psd_fmax))
                bandpower_alt = _integrated_bandpower(psd_alt[i, :], freqs_alt, lo, hi)
                band_alt[band_name] = bandpower_alt
                features_dict[f"eeg_{band_name}__psd_alt"] = bandpower_alt

            if "alpha_theta" in ratios:
                alpha_alt = band_alt.get("alpha", np.nan)
                theta_alt = band_alt.get("theta", np.nan)
                features_dict["eeg_alpha_theta__psd_alt"] = (
                    alpha_alt / theta_alt
                    if (theta_alt is not None and isinstance(theta_alt, (int, float)) and theta_alt > 0)
                    else np.nan
                )
            if "beta_alpha" in ratios:
                beta_alt = band_alt.get("beta", np.nan)
                alpha_alt = band_alt.get("alpha", np.nan)
                features_dict["eeg_beta_alpha__psd_alt"] = (
                    beta_alt / alpha_alt
                    if (alpha_alt is not None and isinstance(alpha_alt, (int, float)) and alpha_alt > 0)
                    else np.nan
                )

        # --- Per-group montages (if any) ---
        # psd index 1..n_signals-1 correspond to group_defs in order
        for g_idx, group in enumerate(group_defs, start=1):
            suffix = f"__g_{group.safe_name}"
            for band_name, (low, high) in eeg_bands.items():
                lo = max(float(low), float(psd_fmin))
                hi = min(float(high), float(psd_fmax))
                bandpower = _integrated_bandpower(psd[i, g_idx, :], freqs, lo, hi)
                features_dict[f"eeg_{band_name}{suffix}"] = bandpower
            features_dict[f"eeg_highfreq_power_30_45{suffix}"] = _integrated_bandpower(
                psd[i, g_idx, :], freqs, max(30.0, float(psd_fmin)), min(45.0, float(psd_fmax))
            )

            if "alpha_theta" in ratios:
                alpha_g = features_dict.get(f"eeg_alpha{suffix}", np.nan)
                theta_g = features_dict.get(f"eeg_theta{suffix}", np.nan)
                features_dict[f"eeg_alpha_theta{suffix}"] = (
                    alpha_g / theta_g
                    if (theta_g is not None and isinstance(theta_g, (int, float)) and theta_g > 0)
                    else np.nan
                )
            if "beta_alpha" in ratios:
                beta_g = features_dict.get(f"eeg_beta{suffix}", np.nan)
                alpha_g = features_dict.get(f"eeg_alpha{suffix}", np.nan)
                features_dict[f"eeg_beta_alpha{suffix}"] = (
                    beta_g / alpha_g
                    if (alpha_g is not None and isinstance(alpha_g, (int, float)) and alpha_g > 0)
                    else np.nan
                )
            hj_mob_g, hj_comp_g = _compute_hjorth_metrics(epochs_agg_arr[i, g_idx])
            features_dict[f"eeg_hjorth_mobility{suffix}"] = hj_mob_g
            features_dict[f"eeg_hjorth_complexity{suffix}"] = hj_comp_g

            # Entropy per group (value-only; provenance is tracked globally).
            group_entropy, _ = _compute_entropy_feature(
                epochs_agg_arr[i, g_idx],
                sfreq=float(sfreq),
                order=pe_order,
                delay=pe_delay,
                normalize=pe_normalize,
            )
            features_dict[f"eeg_permutation_entropy{suffix}"] = group_entropy
            features_dict[f"eeg_sample_entropy{suffix}"] = group_entropy

        records.append(features_dict)

    df = pd.DataFrame(records)
    t_epoch1 = time.perf_counter()

    try:
        coma_reactivity_proxy = _compute_conventional_coma_reactivity_proxy(
            frame=df,
            conventional_cfg=conventional_cfg,
        )
        if coma_reactivity_proxy is not None and len(coma_reactivity_proxy) == len(df):
            df["eeg_conventional_coma_reactivity_proxy"] = coma_reactivity_proxy
    except Exception:
        logger.exception("Conventional EEG coma reactivity proxy failed; continuing without reactivity proxy")

    # ------------------------------------------------------------------
    # v1.2 modular extensions (synchrony / complexity / dFC / graph)
    # ------------------------------------------------------------------
    advanced_features: Dict[str, float] = {}
    fc_stack: Optional[np.ndarray] = None

    ndt_cfg = config.get("ndt_ingest", {}) if isinstance(config, Mapping) else {}
    modalities_cfg = ndt_cfg.get("modalities", {}) if isinstance(ndt_cfg, Mapping) else {}
    eeg_ndt_cfg = modalities_cfg.get("eeg", {}) if isinstance(modalities_cfg, Mapping) else {}

    # Allow per-dataset overrides under ndt_ingest.modalities.eeg.datasets.<dataset_id>.
    if dataset_id and isinstance(eeg_ndt_cfg, Mapping):
        ds_overrides = eeg_ndt_cfg.get("datasets", {})
        if isinstance(ds_overrides, Mapping):
            ds_cfg = ds_overrides.get(dataset_id)
            if isinstance(ds_cfg, Mapping):
                merged_cfg = dict(eeg_ndt_cfg)
                for k, v in ds_cfg.items():
                    if isinstance(v, Mapping) and isinstance(merged_cfg.get(k), Mapping):
                        tmp = dict(merged_cfg.get(k, {}))
                        tmp.update(dict(v))
                        merged_cfg[k] = tmp
                    else:
                        merged_cfg[k] = v
                eeg_ndt_cfg = merged_cfg

    if isinstance(eeg_ndt_cfg, Mapping) and eeg_ndt_cfg.get("enabled", False):
        # Synchrony (requires channel names)
        try:
            sync_cfg = eeg_ndt_cfg.get("synchrony", {})
            if sync_cfg and sync_cfg.get("enabled", False) and eeg_channels:
                t0 = time.perf_counter()
                sync_feats = eeg_sync.compute_eeg_synchrony_features(
                    eeg_data,
                    sfreq=float(sfreq),
                    channel_names=eeg_channels,
                    config=sync_cfg,
                )
                advanced_features.update(sync_feats)
                logger.info("EEG synchrony time: %.2fs", time.perf_counter() - t0)
        except Exception:
            logger.exception("EEG synchrony computation failed; continuing without synchrony features")

        # Complexity (global montage)
        try:
            comp_cfg = eeg_ndt_cfg.get("complexity", {})
            if comp_cfg and comp_cfg.get("enabled", False):
                t0 = time.perf_counter()
                comp_feats = eeg_complexity.compute_eeg_complexity_features(
                    eeg_data,
                    sfreq=float(sfreq),
                    config=comp_cfg,
                )
                advanced_features.update(comp_feats)
                logger.info("EEG complexity time: %.2fs", time.perf_counter() - t0)
        except Exception:
            logger.exception("EEG complexity computation failed; continuing without complexity features")

        # Dynamic FC (sliding-window Pearson)
        try:
            dfc_cfg = eeg_ndt_cfg.get("dynamic_fc", {})
            if dfc_cfg and dfc_cfg.get("enabled", False):
                t0 = time.perf_counter()
                dfc_feats, fc_stack = eeg_dfc.compute_eeg_dynamic_fc_features(
                    eeg_data,
                    sfreq=float(sfreq),
                    config=dfc_cfg,
                )
                advanced_features.update(dfc_feats)
                logger.info("EEG dynamic_fc time: %.2fs", time.perf_counter() - t0)
        except Exception:
            logger.exception("EEG dynamic FC computation failed; continuing without dFC features")

        # Graph metrics (operate on average FC matrix)
        try:
            graph_cfg = eeg_ndt_cfg.get("graph_metrics", {})
            if graph_cfg and graph_cfg.get("enabled", False):
                t0 = time.perf_counter()
                fc_matrix = None
                if fc_stack is not None and fc_stack.size:
                    fc_matrix = np.nanmean(fc_stack, axis=0)
                elif eeg_data.shape[0] >= 2:
                    fc_matrix = np.corrcoef(eeg_data)
                if fc_matrix is not None and np.all(np.isfinite(fc_matrix)):
                    graph_feats = graph_metrics.compute_graph_metrics(fc_matrix, graph_cfg)
                    advanced_features.update({f"eeg_{k}": v for k, v in graph_feats.items()})
                logger.info("EEG graph_metrics time: %.2fs", time.perf_counter() - t0)
        except Exception:
            logger.exception("EEG graph metrics computation failed; continuing without graph features")

    try:
        conventional_connectivity = _compute_conventional_connectivity_features(
            eeg_data=eeg_data,
            sfreq=float(sfreq),
            eeg_channels=eeg_channels,
            conventional_cfg=conventional_cfg,
        )
        if conventional_connectivity:
            advanced_features.update(conventional_connectivity)
    except Exception:
        logger.exception("Conventional EEG connectivity computation failed; continuing without connectivity features")

    if advanced_features and not df.empty:
        adv_cols = {
            key: np.full(len(df), value, dtype=np.float32)
            for key, value in advanced_features.items()
        }
        adv_df = pd.DataFrame(adv_cols, index=df.index)
        df = pd.concat([df, adv_df], axis=1)

    logger.info(
        "Computed %d EEG epochs (groups=%d). Timings: epoch+PSD=%.2fs (PSD=%.2fs)",
        len(df),
        len(group_defs),
        (t_epoch1 - t_psd0),
        (t_psd1 - t_psd0),
    )
    return df


def _compute_spectral_entropy(data: np.ndarray, sfreq: float) -> float:
    """Deterministic entropy fallback based on normalized Welch PSD."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0:
        return np.nan
    finite = np.isfinite(arr)
    if not finite.any():
        return np.nan
    x = arr[finite]
    if x.size == 0:
        return np.nan
    nperseg = int(min(256, x.size))
    if nperseg < 8 or not np.isfinite(sfreq) or sfreq <= 0:
        return np.nan
    freqs, psd = signal.welch(x, fs=float(sfreq), nperseg=nperseg)
    _ = freqs  # Keep explicit for readability/debugging.
    psd = np.abs(np.asarray(psd, dtype=np.float64))
    total = float(np.sum(psd))
    if not np.isfinite(total):
        return np.nan
    if total <= 0:
        return 0.0
    psd_norm = psd / total
    psd_norm = np.clip(psd_norm, 1e-12, None)
    psd_norm /= float(np.sum(psd_norm))
    return float(-np.sum(psd_norm * np.log(psd_norm)))


def _compute_permutation_entropy(
    data: np.ndarray,
    order: int = 5,
    delay: int = 1,
    normalize: bool = True,
) -> float:
    """Compute permutation entropy deterministically with NumPy only."""
    x = np.asarray(data, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    order_i = int(order)
    delay_i = int(delay)
    if order_i < 2 or delay_i < 1:
        return np.nan
    if n < (order_i - 1) * delay_i + 2:
        return np.nan

    y = x[::delay_i]
    if y.size < order_i + 1:
        return np.nan
    emb = sliding_window_view(y, window_shape=order_i)
    if emb.size == 0:
        return np.nan

    patterns = np.argsort(emb, axis=1, kind="mergesort")
    pat_view = np.ascontiguousarray(patterns).view(
        np.dtype((np.void, patterns.dtype.itemsize * patterns.shape[1]))
    )
    _, counts = np.unique(pat_view, return_counts=True)
    if counts.size == 0:
        return np.nan

    p = counts.astype(np.float64) / float(np.sum(counts))
    h_val = -np.sum(p * np.log(p + 1e-32))
    if normalize:
        denom = math.log(math.factorial(order_i))
        if denom <= 0:
            return np.nan
        h_val /= denom
    return float(h_val)


def _compute_entropy_feature(
    data: np.ndarray,
    sfreq: float,
    order: int = 5,
    delay: int = 1,
    normalize: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    """Compute energetic_complexity with permutation entropy primary."""
    base_meta: Dict[str, Any] = {
        "construct": "energetic_complexity",
        "metric": "permutation_entropy",
        "backend": "numpy",
        "degraded_mode": False,
        "reason": None,
    }
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).any():
        out = dict(base_meta)
        out["degraded_mode"] = True
        out["reason"] = "empty_or_non_finite_signal"
        return np.nan, out
    try:
        value = _compute_permutation_entropy(arr, order=int(order), delay=int(delay), normalize=bool(normalize))
        if np.isfinite(value):
            return float(value), base_meta
        raise ValueError("permutation_entropy returned non-finite value")
    except Exception:
        global _ENTROPY_FALLBACK_WARNED
        if not _ENTROPY_FALLBACK_WARNED:
            logger.warning(
                "permutation entropy failed; using spectral entropy fallback for energetic complexity"
            )
            _ENTROPY_FALLBACK_WARNED = True
        try:
            fallback = _compute_spectral_entropy(arr, sfreq=float(sfreq))
            out = dict(base_meta)
            out["metric"] = "spectral_entropy"
            out["backend"] = "scipy_welch"
            out["degraded_mode"] = True
            out["reason"] = "permutation_entropy_failed"
            return fallback, out
        except Exception:
            out = dict(base_meta)
            out["degraded_mode"] = True
            out["reason"] = "entropy_fallback_failed"
            return np.nan, out


def _compute_sample_entropy(
    data: np.ndarray,
    m: int = 2,
    r: float = 0.2,
    sfreq: float = 250.0,
) -> float:
    """Backward-compatible alias; returns permutation-entropy based value."""
    _ = (m, r)  # retained for API compatibility
    value, _ = _compute_entropy_feature(data, sfreq=float(sfreq))
    return float(value)


