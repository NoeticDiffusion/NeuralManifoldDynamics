"""EEG feature extraction (bandpower, ratios, entropy, optional ensembles).

Consumes a preprocessed ``signals`` mapping (EEG arrays and metadata) and ingest
``config`` (epoching and ``features.eeg``); returns a per-epoch DataFrame
(``epoch_id``, timing, spectral features, etc.).
"""

from __future__ import annotations

import math
import logging
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import signal
import time

from core import ensembles
from .eeg_entropy import compute_permutation_entropy as _compute_permutation_entropy
from . import epoch_selection
from . import eeg_complexity
from . import eeg_dfc
from . import eeg_sync
from core.metrics import graph as graph_metrics
from .. import nwb_intervals
from ..reproducibility import resolve_component_seed
from ..signal_support_provenance import map_preprocessed_window_to_source_samples

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
    w = min(w, int(arr.size))
    kernel = np.ones(w, dtype=float)
    sums = np.convolve(arr, kernel, mode="same")
    counts = np.convolve(np.ones(arr.size, dtype=float), kernel, mode="same")
    return sums / np.maximum(counts, 1.0)


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open intervals for contiguous True values."""
    values = np.asarray(mask, dtype=bool).reshape(-1)
    if not values.size or not values.any():
        return []
    edges = np.diff(np.r_[False, values, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _compute_technical_eeg_qc(
    *,
    eeg_window: np.ndarray,
    payload_meta: Mapping[str, Any],
    channel_names: Any,
) -> Dict[str, Any]:
    """Describe signal-level technical flags without masking EEG physiology."""
    data = np.asarray(eeg_window, dtype=float)
    if data.ndim != 2:
        data = np.empty((0, 0), dtype=float)
    n_channels, n_samples = data.shape
    finite = np.isfinite(data)
    finite_fraction = float(finite.mean()) if finite.size else np.nan
    sample_valid = finite.all(axis=0) if n_channels else np.zeros(n_samples, dtype=bool)
    gap_runs = _true_runs(~sample_valid)
    flatline_count = 0
    for channel in data:
        channel_finite = channel[np.isfinite(channel)]
        if channel_finite.size > 1 and float(np.max(channel_finite)) == float(np.min(channel_finite)):
            flatline_count += 1

    names = channel_names if isinstance(channel_names, (list, tuple)) else None
    if n_channels == 0:
        channel_status, channel_reason = "unsupported", "no_eeg_channels"
    elif names is None:
        channel_status, channel_reason = "not_assessed", "eeg_channel_identity_unavailable"
    elif len(names) != n_channels:
        channel_status, channel_reason = "unsupported", "eeg_channel_metadata_count_mismatch"
    else:
        channel_status, channel_reason = "computed", None

    reasons = []
    if n_channels == 0 or n_samples == 0:
        technical_status = "not_assessed"
        reasons.append("empty_eeg_window")
    else:
        technical_status = "valid"
        if finite_fraction < 1.0:
            technical_status = "invalid"
            reasons.append("non_finite_input_samples")
        if flatline_count > 0:
            if technical_status == "valid":
                technical_status = "suspect"
            reasons.append(
                "all_eeg_channels_exactly_flatlined"
                if flatline_count == n_channels
                else "one_or_more_eeg_channels_exactly_flatlined"
            )
        if channel_status == "unsupported":
            technical_status = "invalid"
            reasons.append(channel_reason)

    rails = payload_meta.get("eeg_adc_rails", {}) if isinstance(payload_meta, Mapping) else {}
    clip_status, clip_reason, clip_fraction = "not_assessed", "verified_adc_rails_unavailable", np.nan
    if isinstance(rails, Mapping) and rails.get("status") == "verified":
        try:
            rail_min = float(rails["min"])
            rail_max = float(rails["max"])
            if math.isfinite(rail_min) and math.isfinite(rail_max) and rail_max > rail_min and finite.any():
                clip_fraction = float(np.mean((data[finite] <= rail_min) | (data[finite] >= rail_max)))
                clip_status, clip_reason = "computed", None
                if clip_fraction > 0:
                    if technical_status == "valid":
                        technical_status = "suspect"
                    reasons.append("verified_adc_rail_hits_observed")
            else:
                clip_status, clip_reason = "not_assessed", "verified_adc_rails_invalid_or_no_finite_samples"
        except (KeyError, TypeError, ValueError, OverflowError):
            clip_status, clip_reason = "not_assessed", "verified_adc_rails_invalid_or_no_finite_samples"

    return {
        "technical_eeg_status": technical_status,
        "technical_eeg_reason": ";".join(dict.fromkeys(reasons)) if reasons else None,
        "technical_eeg_channel_status": channel_status,
        "technical_eeg_channel_reason": channel_reason,
        "technical_eeg_channel_count": int(n_channels),
        "technical_eeg_finite_fraction": finite_fraction,
        "technical_eeg_gap_count": int(len(gap_runs)),
        "technical_eeg_flatline_channel_count": int(flatline_count),
        "technical_eeg_clipping_status": clip_status,
        "technical_eeg_clipping_reason": clip_reason,
        "technical_eeg_adc_rail_fraction": clip_fraction,
    }


def _compute_conventional_coma_features(
    *,
    epoch_signal: np.ndarray,
    sfreq: float,
    bandpowers: Mapping[str, float],
    conventional_cfg: Mapping[str, Any],
    amplitude_unit: Optional[str] = None,
    amplitude_unit_status: str = "unknown",
    reference_mode: Optional[str] = None,
    reference_status: str = "unknown",
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
    finite_runs = _true_runs(finite)
    gap_runs = _true_runs(~finite)
    finite_fraction = float(finite.mean()) if finite.size else 0.0
    if not np.any(finite):
        return {
            "eeg_conventional_coma_suppression_ratio": np.nan,
            "eeg_conventional_coma_continuity_proxy": np.nan,
            "eeg_conventional_coma_burst_suppression_proxy": np.nan,
            "eeg_conventional_coma_alpha_delta_ratio": np.nan,
            "eeg_conventional_coma_amplitude_rms": np.nan,
            "coma_proxy_suppression_status": "not_assessed",
            "coma_proxy_suppression_reason": "no_finite_input_samples",
            "coma_proxy_amplitude_status": "not_assessed",
            "coma_proxy_amplitude_reason": "no_finite_input_samples",
            "coma_proxy_unit_status": str(amplitude_unit_status or "unknown"),
            "coma_proxy_amplitude_unit": amplitude_unit or "unknown",
            "coma_proxy_reference_status": str(reference_status or "unknown"),
            "coma_proxy_reference_mode": reference_mode or "unknown",
            "coma_proxy_valid_sample_fraction": finite_fraction,
            "coma_proxy_gap_count": int(len(gap_runs)),
            "coma_proxy_segment_count": 0,
            "coma_proxy_threshold_value": np.nan,
        }

    x = arr[finite]
    rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else np.nan
    threshold = max(float(amplitude_floor), float(relative_to_rms) * float(rms)) if np.isfinite(rms) else float(amplitude_floor)
    suppressed_count = 0
    transition_count = 0
    adjacency_count = 0
    for start, end in finite_runs:
        envelope = _moving_average_1d(np.abs(arr[start:end]), smooth_window)
        segment_mask = envelope <= threshold
        suppressed_count += int(segment_mask.sum())
        if segment_mask.size > 1:
            transition_count += int(np.count_nonzero(segment_mask[1:] != segment_mask[:-1]))
            adjacency_count += int(segment_mask.size - 1)
    suppression_ratio = float(suppressed_count / x.size) if x.size else np.nan
    unit_verified = str(amplitude_unit_status).strip().lower() == "verified" and bool(amplitude_unit)
    reference_verified = str(reference_status).strip().lower() == "verified" and bool(reference_mode)
    amplitude_qualified = unit_verified and reference_verified
    proxy_status = "computed" if not gap_runs else "computed_segmented_with_gaps"
    proxy_reason_parts = []
    if not unit_verified:
        proxy_reason_parts.append("amplitude_unit_unverified")
    if not reference_verified:
        proxy_reason_parts.append("amplitude_reference_unverified")
    proxy_reason = ";".join(proxy_reason_parts) if proxy_reason_parts else None
    if not amplitude_qualified:
        # The absolute floor has physical units. Without a verified scale the
        # thresholded outputs are not interpretable, so retain statuses and
        # coverage but do not export a numeric physiological proxy.
        suppression_ratio = np.nan

    out: Dict[str, Any] = {}
    if bool(coma_cfg.get("suppression_ratio", True)):
        out["eeg_conventional_coma_suppression_ratio"] = suppression_ratio
    if bool(coma_cfg.get("continuity_proxy", True)):
        out["eeg_conventional_coma_continuity_proxy"] = (
            1.0 - suppression_ratio if np.isfinite(suppression_ratio) else np.nan
        )
    if bool(coma_cfg.get("burst_suppression_proxy", True)):
        if adjacency_count <= 0 or not np.isfinite(suppression_ratio):
            burst_proxy = np.nan
        else:
            transitions = float(transition_count / adjacency_count)
            mix = float(4.0 * suppression_ratio * (1.0 - suppression_ratio))
            burst_proxy = float(np.clip(mix * transitions, 0.0, 1.0))
        out["eeg_conventional_coma_burst_suppression_proxy"] = burst_proxy
    if bool(coma_cfg.get("alpha_delta_ratio", True)):
        out["eeg_conventional_coma_alpha_delta_ratio"] = _safe_ratio(
            float(bandpowers.get("alpha", np.nan)),
            float(bandpowers.get("delta", np.nan)),
        )
    out["eeg_conventional_coma_amplitude_rms"] = rms if amplitude_qualified else np.nan
    out.update({
        "coma_proxy_suppression_status": proxy_status if amplitude_qualified else "not_testable",
        "coma_proxy_suppression_reason": proxy_reason,
        "coma_proxy_unit_status": str(amplitude_unit_status or "unknown"),
        "coma_proxy_amplitude_unit": amplitude_unit or "unknown",
        "coma_proxy_reference_status": str(reference_status or "unknown"),
        "coma_proxy_reference_mode": reference_mode or "unknown",
        "coma_proxy_valid_sample_fraction": finite_fraction,
        "coma_proxy_gap_count": int(len(gap_runs)),
        "coma_proxy_segment_count": int(len(finite_runs)),
        "coma_proxy_threshold_value": float(threshold) if amplitude_qualified else np.nan,
        "coma_proxy_amplitude_status": "computed" if amplitude_qualified else "not_testable",
        "coma_proxy_amplitude_reason": proxy_reason,
    })
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

    # Explicitly masked BAD_ annotations (including opt-in source quality
    # intervals) are represented as NaNs in the preprocessing payload. Reject
    # whole windows touching them rather than constructing epochs across gaps.
    bdf_meta = payload_meta.get("bdf_adapter", {}) if isinstance(payload_meta, Mapping) else {}
    bad_segments_meta = bdf_meta.get("bad_segments", {}) if isinstance(bdf_meta, Mapping) else {}
    quality_policy_meta = payload_meta.get("quality_interval_policy", {}) if isinstance(payload_meta, Mapping) else {}
    reject_masked_annotation_windows = bool(
        isinstance(bad_segments_meta, Mapping)
        and bad_segments_meta.get("enabled")
        and bad_segments_meta.get("available")
    ) or bool(
        isinstance(quality_policy_meta, Mapping)
        and quality_policy_meta.get("enabled")
        and quality_policy_meta.get("mask_bad_annotations")
    )
    if reject_masked_annotation_windows:
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
                "Rejected %d/%d EEG epochs touching masked BAD_ intervals for %s",
                dropped,
                original_count,
                Path(str(raw_file_path)).name if raw_file_path else "recording",
            )

    sampling_cfg = epoch_selection.resolve_epoching_sampling_cfg(config, dataset_id)
    # Some analyses deliberately freeze their labels from the filename or a
    # separate design table and must not inspect BIDS event rows. Preserve the
    # historical default while allowing a dataset overlay to opt out of event
    # reads for both stage-stratified sampling and the stage-column fallback.
    read_bids_events = bool(sampling_cfg.get("read_bids_events", True))
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
            events_path = (
                epoch_selection.find_events_tsv_for_raw(str(raw_file_path))
                if read_bids_events
                else None
            )
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
            events_path = (
                epoch_selection.find_events_tsv_for_raw(str(raw_file_path))
                if read_bids_events
                else None
            )
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
    psd_method_id = "mne.time_frequency.psd_array_multitaper"
    psd_fallback_reason = "none"
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
        psd_method_id = "scipy.signal.welch"
        if psd_method == "multitaper":
            psd_fallback_reason = "multitaper_unavailable"
        elif psd_method != "welch":
            psd_fallback_reason = f"unsupported_configured_method:{psd_method}"
        else:
            psd_fallback_reason = "configured_welch"
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
    channels_map = signals.get("channels", {}) if isinstance(signals, Mapping) else {}
    eeg_channel_names = channels_map.get("eeg") if isinstance(channels_map, Mapping) else None
    amplitude_unit = payload_meta.get("eeg_amplitude_unit") if isinstance(payload_meta, Mapping) else None
    amplitude_unit_status = (
        str(payload_meta.get("eeg_amplitude_unit_status", "unknown"))
        if isinstance(payload_meta, Mapping)
        else "unknown"
    )
    reference_mode = payload_meta.get("eeg_reference_mode") if isinstance(payload_meta, Mapping) else None
    reference_status = (
        str(payload_meta.get("eeg_reference_status", "unknown"))
        if isinstance(payload_meta, Mapping)
        else "unknown"
    )
    support_provenance = payload_meta.get("signal_support_provenance", {}) if isinstance(payload_meta, Mapping) else {}
    # MNPS 3.4 candidates are computed on the preprocessed source segment
    # before per-window row assembly. They are opt-in and live beside, rather
    # than inside, the legacy v2 projection feature map.
    v3_4_candidates_by_epoch: Dict[int, Dict[str, Any]] = {}
    try:
        from ..mnps.v3_4 import compute_eeg_candidates, resolve_v3_4_config

        v3_4_cfg = resolve_v3_4_config(config)
        if v3_4_cfg and bool(v3_4_cfg.get("enabled", False)):
            eeglab_import_meta = payload_meta.get("eeglab_import", {}) if isinstance(payload_meta, Mapping) else {}
            flattened_epochs = (
                isinstance(eeglab_import_meta, Mapping)
                and str(eeglab_import_meta.get("source", "")).strip().lower() == "eeglab_epoched_concat"
            )
            candidate_records = compute_eeg_candidates(
                np.asarray(eeg_data),
                sfreq=float(sfreq),
                windows=meta,
                channel_labels=list(eeg_channel_names or []),
                config=config,
                dataset_id=str(dataset_id) if dataset_id is not None else None,
                reference=str(reference_mode) if reference_mode else None,
                reference_set_hash=(
                    str(payload_meta.get("reference_set_hash"))
                    if isinstance(payload_meta, Mapping) and payload_meta.get("reference_set_hash")
                    else None
                ),
                source_file=str(raw_file_path) if raw_file_path is not None else None,
                preprocessed_segmented_epochs=flattened_epochs,
            )
            v3_4_candidates_by_epoch = {
                int(window_meta[0]): record
                for window_meta, record in zip(meta, candidate_records)
            }
    except Exception:
        # Explicit v3.4 activation is a measurement request: configuration and
        # provenance errors must fail extraction rather than silently omit it.
        if isinstance(config, Mapping):
            requested_v3_4 = config.get("mnps_v3_4", {})
            requested_v3_4 = requested_v3_4 if isinstance(requested_v3_4, Mapping) else config
            if bool(requested_v3_4.get("enabled", False)) or config.get("schema") == "mnps_flat_coordinates":
                raise
        logger.exception("Optional MNPS 3.4 candidate extraction failed")
    v31_cfg = config.get("mnps_v3_1_measurements", {}) if isinstance(config, Mapping) else {}
    v31_cfg = v31_cfg if isinstance(v31_cfg, Mapping) else {}
    v31_enabled = bool(v31_cfg.get("enabled", False))
    v31_measurement_windows: List[Dict[str, Any]] = []
    v31_temporal_realizations: List[Dict[str, Any]] = []
    v31_recording_level_experimental: Dict[str, Any] | None = None
    v31_module = None
    if v31_enabled:
        try:
            from ..mnps.v3_1 import pipeline_measurements as v31_module
        except Exception:
            logger.exception("MNPS v3.1 measurement module unavailable; continuing with ordinary EEG features")
            v31_enabled = False
    source_id = "unknown_source"
    source_identity_metadata: Dict[str, Any] = {
        "identity_basis": "unknown",
        "source_sha256": None,
        "dataset_id": str(dataset_id or "unknown_dataset"),
    }
    if isinstance(support_provenance, Mapping):
        source_record = support_provenance.get("source", {})
        if isinstance(source_record, Mapping):
            source_sha256 = source_record.get("sha256") or source_record.get("source_sha256")
            if source_sha256:
                source_id = str(source_sha256)
                source_identity_metadata.update({
                    "identity_basis": "source_file_sha256",
                    "source_sha256": str(source_sha256),
                    "hash_stable": source_record.get("hash_stable"),
                })
    if source_id == "unknown_source":
        source_path = str(payload_meta.get("file", raw_file_path or "")) if isinstance(payload_meta, Mapping) else str(raw_file_path or "")
        source_name = Path(source_path).name if source_path else "unknown_file"
        identity_material = "|".join((str(dataset_id or "unknown_dataset"), source_path or source_name))
        source_id = hashlib.sha256(identity_material.encode("utf-8")).hexdigest()
        source_identity_metadata.update({
            "identity_basis": "dataset_and_source_path_sha256" if source_path else "dataset_and_unknown_file_sha256",
            "source_path_sha256": hashlib.sha256(source_path.encode("utf-8")).hexdigest() if source_path else None,
            "source_name": source_name,
        })
    source_identity_metadata["source_identity_id"] = source_id
    operations = support_provenance.get("operations", []) if isinstance(support_provenance, Mapping) else []
    configured_filter_band = None
    for operation in operations if isinstance(operations, list) else []:
        if not isinstance(operation, Mapping) or str(operation.get("name", "")) != "eeg_bandpass":
            continue
        parameters = operation.get("parameters", {})
        if isinstance(parameters, Mapping) and isinstance(parameters.get("bandpass_hz"), (list, tuple)):
            configured_filter_band = list(parameters["bandpass_hz"])
            break
    input_branch = {
        "branch_id": "mndm.preprocessed_native_eeg_window.v1",
        "reference_mode": reference_mode or "unknown",
        "reference_set_sha256": payload_meta.get("reference_set_hash") if isinstance(payload_meta, Mapping) else None,
        "filter_band_hz": configured_filter_band,
        "amplitude_unit": amplitude_unit or "unknown",
        "reference_status": reference_status,
    }
    eeg_acquisition_qc = payload_meta.get("eeg_acquisition_qc", {}) if isinstance(payload_meta, Mapping) else {}
    prefilter_qc = eeg_acquisition_qc.get("prefilter", {}) if isinstance(eeg_acquisition_qc, Mapping) else {}
    postfilter_qc = eeg_acquisition_qc.get("postfilter", {}) if isinstance(eeg_acquisition_qc, Mapping) else {}
    eeg_channel_count = len(eeg_channel_names) if isinstance(eeg_channel_names, (list, tuple)) else 0
    acquisition_qc = {
        "line_noise_48_52_ratio": prefilter_qc.get("line_noise_48_52_ratio") if isinstance(prefilter_qc, Mapping) else None,
        "line_noise_48_52_ratio_status": prefilter_qc.get("line_noise_48_52_ratio_status", "unknown") if isinstance(prefilter_qc, Mapping) else "unknown",
        "postfilter_line_noise_residual": postfilter_qc.get("line_noise_48_52_ratio") if isinstance(postfilter_qc, Mapping) else None,
        "postfilter_line_noise_residual_status": postfilter_qc.get("line_noise_48_52_ratio_status", "unknown") if isinstance(postfilter_qc, Mapping) else "unknown",
        "broadband_noise_proxy": prefilter_qc.get("broadband_noise_proxy") if isinstance(prefilter_qc, Mapping) else None,
        "broadband_noise_proxy_status": prefilter_qc.get("broadband_noise_proxy_status", "unknown") if isinstance(prefilter_qc, Mapping) else "unknown",
        "channel_type_profile": payload_meta.get("channel_type_profile") if isinstance(payload_meta, Mapping) else None,
        "eeg_channel_count": payload_meta.get("eeg_channel_count", eeg_channel_count) if isinstance(payload_meta, Mapping) else eeg_channel_count,
        "misc_channel_count": payload_meta.get("misc_channel_count") if isinstance(payload_meta, Mapping) else None,
        "reference_set_hash": payload_meta.get("reference_set_hash") if isinstance(payload_meta, Mapping) else None,
        "channel_set_hash": payload_meta.get("channel_set_hash") if isinstance(payload_meta, Mapping) else None,
        "measurement_adjustment_applied": False,
    }

    for i, (epoch_id, start_idx, end_idx) in enumerate(meta):
        support_provenance = payload_meta.get("signal_support_provenance", {}) if isinstance(payload_meta, Mapping) else {}
        eeglab_import = payload_meta.get("eeglab_import", {}) if isinstance(payload_meta, Mapping) else {}
        is_flattened_eeglab_epochs = (
            isinstance(eeglab_import, Mapping)
            and str(eeglab_import.get("source", "")).strip().lower() == "eeglab_epoched_concat"
        )
        source_grid_supported = not (
            isinstance(support_provenance, Mapping)
            and support_provenance.get("source_sample_grid_supported") is False
        )
        source_mapping_reason = (
            str(support_provenance.get("source_sample_grid_reason") or "source_clock_mapping_not_supported")
            if isinstance(support_provenance, Mapping)
            else "source_clock_mapping_not_supported"
        )
        allow_source_grid = not is_flattened_eeglab_epochs and source_grid_supported
        refusal_reason = (
            "flattened_eeglab_epochs_have_no_continuous_source_sample_map"
            if is_flattened_eeglab_epochs
            else source_mapping_reason
        )
        source_sample_map = map_preprocessed_window_to_source_samples(
            int(start_idx),
            int(end_idx),
            preprocessed_sfreq_hz=float(sfreq),
            provenance=support_provenance if isinstance(support_provenance, Mapping) else {},
            allow_source_grid=allow_source_grid,
            refusal_reason=refusal_reason,
        )
        input_window = np.asarray(eeg_data[:, start_idx:end_idx])
        finite_fraction = float(np.isfinite(input_window).mean()) if input_window.size else np.nan
        input_finite_status = "all_finite" if finite_fraction == 1.0 else "contains_nonfinite" if np.isfinite(finite_fraction) else "not_assessed"
        technical_qc = _compute_technical_eeg_qc(
            eeg_window=input_window,
            payload_meta=payload_meta if isinstance(payload_meta, Mapping) else {},
            channel_names=eeg_channel_names,
        )
        sample_valid = np.isfinite(input_window).all(axis=0) if input_window.shape[0] else np.zeros(input_window.shape[1], dtype=bool)
        gap_spans = []
        for gap_start, gap_end in _true_runs(~sample_valid):
            gap_source = map_preprocessed_window_to_source_samples(
                int(start_idx + gap_start),
                int(start_idx + gap_end),
                preprocessed_sfreq_hz=float(sfreq),
                provenance=support_provenance if isinstance(support_provenance, Mapping) else {},
                allow_source_grid=allow_source_grid,
                refusal_reason=refusal_reason,
            )
            gap_spans.append({
                "preprocessed_sample_start": int(start_idx + gap_start),
                "preprocessed_sample_end_exclusive": int(start_idx + gap_end),
                "source_sample_mapping_status": gap_source.get("source_sample_mapping_status", "unknown"),
                "source_sample_mapping_reason": gap_source.get("source_sample_mapping_reason"),
                "nominal_source_sample_start": gap_source.get("nominal_source_sample_start"),
                "nominal_source_sample_end_exclusive": gap_source.get("nominal_source_sample_end_exclusive"),
                "nominal_source_time_start_sec": gap_source.get("nominal_source_time_start_sec"),
                "nominal_source_time_end_sec": gap_source.get("nominal_source_time_end_sec"),
            })
        features_dict: Dict[str, Any] = {
            "epoch_id": epoch_id,
            "t_start": start_idx / sfreq,
            "t_end": end_idx / sfreq,
            **source_sample_map,
            "input_sample_finite_status": input_finite_status,
            "input_sample_finite_fraction": finite_fraction,
            "input_sample_gap_count": int(len(gap_spans)),
            "input_sample_gap_spans": json.dumps(gap_spans, separators=(",", ":")),
            "input_sample_invalid_reason": None if input_finite_status == "all_finite" else (
                "non_finite_samples_in_nominal_epoch" if input_finite_status == "contains_nonfinite" else "finite_sample_check_unavailable"
            ),
            **technical_qc,
        }
        v3_4_record = v3_4_candidates_by_epoch.get(int(epoch_id))
        if v3_4_record is not None:
            for key in (
                "alpha_centroid_hz",
                "alpha_envelope_cv",
                "spatial_logvariance_dispersion",
                "alpha_centroid_status",
                "alpha_envelope_status",
                "spatial_logvariance_status",
                "alpha_centroid_channel_count",
                "alpha_envelope_channel_count",
                "spatial_logvariance_channel_count",
                "channel_labels_json",
                "channel_set_hash",
                "channel_compatibility_set_hash",
                "channel_order_hash",
                "reference",
                "reference_set_hash",
                "reference_status",
                "dataset_id",
                "source_file",
                "source_surface_hash",
                "alpha_centroid_hz_per_channel_json",
                "alpha_envelope_cv_per_channel_json",
                "spatial_logvariance_per_channel_json",
                "alpha_envelope_edge_guard_sec",
                "implementation_hashes_json",
                "source_component_hashes_json",
            ):
                if key not in v3_4_record:
                    continue
                value = v3_4_record[key]
                col_name = f"mnps_v3_4_{key}"
                if key == "alpha_centroid_hz":
                    col_name = "mnps_v3_4_alpha_centroid_hz"
                elif key == "alpha_envelope_cv":
                    col_name = "mnps_v3_4_alpha_envelope_cv"
                elif key == "spatial_logvariance_dispersion":
                    col_name = "mnps_v3_4_spatial_logvariance_dispersion"
                features_dict[col_name] = value
            features_dict["mnps_v3_4_channel_count"] = int(len(eeg_channel_names or []))
            features_dict["mnps_v3_4_source_file_sha256"] = source_identity_metadata.get("source_sha256")
            features_dict["mnps_v3_4_source_file_sha256_component"] = (
                Path(str(raw_file_path)).name if raw_file_path is not None else "unknown_source_component"
            )
            features_dict["mnps_v3_4_source_identity_basis"] = source_identity_metadata.get("identity_basis")
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
        if bool(features_cfg.get("hjorth_activity", False)):
            epoch_1d = np.asarray(epochs_agg_arr[i, 0], dtype=np.float64)
            epoch_1d = epoch_1d[np.isfinite(epoch_1d)]
            features_dict["eeg_hjorth_activity"] = (
                float(np.var(epoch_1d)) if epoch_1d.size >= 4 else np.nan
            )

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
                amplitude_unit=str(amplitude_unit) if amplitude_unit else None,
                amplitude_unit_status=amplitude_unit_status,
                reference_mode=str(reference_mode) if reference_mode else None,
                reference_status=reference_status,
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
    # Keep the actual runtime backend with every feature row. The config and
    # implementation hash alone do not reveal whether the multitaper fallback
    # branch was selected in this process.
    df["eeg_psd_method_id"] = psd_method_id
    df["eeg_psd_fallback_reason"] = psd_fallback_reason
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

    if v31_enabled and v31_module is not None:
        labels_for_measurement = tuple(str(name) for name in eeg_channel_names) if eeg_channel_names is not None else ()
        segment_id = Path(str(payload_meta.get("file", raw_file_path or "recording"))).name if isinstance(payload_meta, Mapping) else "recording"
        try:
            requested_window_sec = float(v31_cfg.get("window_length_sec", 20.0) or 20.0)
            requested_step_sec = float(v31_cfg.get("step_sec", requested_window_sec) or requested_window_sec)
            requested_window_samples = int(round(requested_window_sec * float(sfreq)))
            step_samples = int(round(requested_step_sec * float(sfreq)))
        except (TypeError, ValueError, OverflowError):
            requested_window_samples = 0
            step_samples = 0
            requested_window_sec = 20.0
        if requested_window_samples > 0 and step_samples > 0 and eeg_data.shape[1] >= 2:
            if eeg_data.shape[1] < requested_window_samples:
                measurement_window_samples = int(eeg_data.shape[1])
                window_starts = [0]
            else:
                measurement_window_samples = requested_window_samples
                window_starts = list(range(0, int(eeg_data.shape[1]) - measurement_window_samples + 1, step_samples))
            eeglab_import = payload_meta.get("eeglab_import", {}) if isinstance(payload_meta, Mapping) else {}
            flattened_epochs = (
                isinstance(eeglab_import, Mapping)
                and str(eeglab_import.get("source", "")).strip().lower() == "eeglab_epoched_concat"
            )
            source_grid_supported = not (
                isinstance(support_provenance, Mapping)
                and support_provenance.get("source_sample_grid_supported") is False
            )
            for measurement_window_index, start_idx in enumerate(window_starts):
                end_idx = int(start_idx + measurement_window_samples)
                input_window = np.asarray(eeg_data[:, start_idx:end_idx])
                finite = bool(input_window.size and np.isfinite(input_window).all())
                source_map = map_preprocessed_window_to_source_samples(
                    int(start_idx),
                    int(end_idx),
                    preprocessed_sfreq_hz=float(sfreq),
                    provenance=support_provenance if isinstance(support_provenance, Mapping) else {},
                    allow_source_grid=not flattened_epochs and source_grid_supported,
                    refusal_reason=(
                        "flattened_eeglab_epochs_have_no_continuous_source_sample_map"
                        if flattened_epochs
                        else str(
                            support_provenance.get("source_sample_grid_reason") or "source_clock_mapping_not_supported"
                            if isinstance(support_provenance, Mapping)
                            else "source_clock_mapping_not_supported"
                        )
                    ),
                )
                source_is_continuous = False if flattened_epochs else None
                explicit_grid_support = (
                    support_provenance.get("source_sample_grid_supported")
                    if isinstance(support_provenance, Mapping)
                    else None
                )
                source_mapping_verified = source_map.get("source_sample_mapping_status") == "nominal_bounds_mapped"
                timestamps_uniform = (
                    True if explicit_grid_support is True and source_mapping_verified
                    else None
                )
                effective_filter_support_status = source_map.get("actual_sample_contributors_status")
                support_qc = {
                    "source_is_continuous": source_is_continuous,
                    "timestamps_uniform": timestamps_uniform,
                    "input_samples_finite": finite,
                    "gap_free": None,
                    "bad_interval_free": None,
                    "within_recording_bounds": bool(0 <= int(start_idx) < end_idx <= int(eeg_data.shape[1])),
                    "effective_filter_support_valid": (
                        True if effective_filter_support_status == "finite_causal_conservative_bound" else None
                    ),
                    "effective_filter_support_status": effective_filter_support_status,
                }
                try:
                    v31_result = v31_module.compute_pipeline_window_measurements(
                        input_window,
                        labels_for_measurement,
                        sample_rate_hz=float(sfreq),
                        window_start_sec=float(start_idx) / float(sfreq),
                        window_end_sec=float(end_idx) / float(sfreq),
                        segment_id=segment_id,
                        source_identity=source_id,
                        input_branch=input_branch,
                        support_qc=support_qc,
                        acquisition_qc=acquisition_qc,
                        sequence_complexity_reference_count=int(v31_cfg.get("sequence_complexity_reference_count", 32) or 32),
                        activity_block_sec=float(v31_cfg.get("activity_block_sec", 2.0) or 2.0),
                    )
                except Exception as exc:
                    logger.exception(
                        "MNPS v3.1 candidates failed on independent measurement window %s; preserving legacy EEG features",
                        measurement_window_index,
                    )
                    v31_result = v31_module.failed_pipeline_window_measurements(
                        channel_names=labels_for_measurement,
                        sample_rate_hz=float(sfreq),
                        window_start_sec=float(start_idx) / float(sfreq),
                        window_end_sec=float(end_idx) / float(sfreq),
                        segment_id=segment_id,
                        input_branch=input_branch,
                        support_qc=support_qc,
                        error=exc,
                    )
                v31_measurement_windows.append({
                    "window_index": int(measurement_window_index),
                    "requested_window_length_sec": float(requested_window_sec),
                    "window_start_sec": float(start_idx) / float(sfreq),
                    "window_end_sec": float(end_idx) / float(sfreq),
                    "source_sample_mapping": source_map,
                    "source_identity": dict(source_identity_metadata),
                    **v31_result,
                })

        temporal_realizations_cfg = v31_cfg.get("temporal_realizations", {})
        temporal_realizations_cfg = temporal_realizations_cfg if isinstance(temporal_realizations_cfg, Mapping) else {}
        m2_win4_cfg = temporal_realizations_cfg.get("temporal_persistence_win4s_v1", {})
        m2_win4_cfg = m2_win4_cfg if isinstance(m2_win4_cfg, Mapping) else {}
        if bool(m2_win4_cfg.get("enabled", True)):
            realization_id = "temporal_persistence.win4s.v1"
            support_window_sec = 4.0
            step_sec = 4.0
            try:
                realization_sfreq = float(sfreq)
                if not math.isfinite(realization_sfreq) or realization_sfreq <= 0.0:
                    raise ValueError("invalid sampling frequency")
                requested_window_samples = int(round(support_window_sec * realization_sfreq))
                step_samples = int(round(step_sec * realization_sfreq))
            except (TypeError, ValueError, OverflowError):
                realization_sfreq = math.nan
                requested_window_samples = 0
                step_samples = 0
            realization_windows: List[Dict[str, Any]] = []
            realization_reason = None
            if requested_window_samples <= 1 or step_samples <= 0:
                realization_status = "unsupported"
                realization_reason = "invalid_sample_rate_or_temporal_realization_parameters"
            elif eeg_data.shape[1] < requested_window_samples:
                realization_status = "unsupported"
                realization_reason = "recording_shorter_than_requested_support"
            else:
                window_starts = list(range(0, int(eeg_data.shape[1]) - requested_window_samples + 1, step_samples))
                eeglab_import = payload_meta.get("eeglab_import", {}) if isinstance(payload_meta, Mapping) else {}
                flattened_epochs = (
                    isinstance(eeglab_import, Mapping)
                    and str(eeglab_import.get("source", "")).strip().lower() == "eeglab_epoched_concat"
                )
                source_grid_supported = not (
                    isinstance(support_provenance, Mapping)
                    and support_provenance.get("source_sample_grid_supported") is False
                )
                for window_index, start_idx in enumerate(window_starts):
                    end_idx = int(start_idx + requested_window_samples)
                    input_window = np.asarray(eeg_data[:, start_idx:end_idx])
                    finite = bool(input_window.size and np.isfinite(input_window).all())
                    source_map = map_preprocessed_window_to_source_samples(
                        int(start_idx),
                        int(end_idx),
                        preprocessed_sfreq_hz=float(sfreq),
                        provenance=support_provenance if isinstance(support_provenance, Mapping) else {},
                        allow_source_grid=not flattened_epochs and source_grid_supported,
                        refusal_reason=(
                            "flattened_eeglab_epochs_have_no_continuous_source_sample_map"
                            if flattened_epochs
                            else str(
                                support_provenance.get("source_sample_grid_reason") or "source_clock_mapping_not_supported"
                                if isinstance(support_provenance, Mapping)
                                else "source_clock_mapping_not_supported"
                            )
                        ),
                    )
                    source_mapping_verified = source_map.get("source_sample_mapping_status") == "nominal_bounds_mapped"
                    explicit_grid_support = (
                        support_provenance.get("source_sample_grid_supported")
                        if isinstance(support_provenance, Mapping)
                        else None
                    )
                    effective_filter_support_status = source_map.get("actual_sample_contributors_status")
                    support_qc = {
                        "source_is_continuous": False if flattened_epochs else None,
                        "timestamps_uniform": (
                            True if explicit_grid_support is True and source_mapping_verified else None
                        ),
                        "input_samples_finite": finite,
                        "gap_free": None,
                        "bad_interval_free": None,
                        "within_recording_bounds": bool(0 <= int(start_idx) < end_idx <= int(eeg_data.shape[1])),
                        "effective_filter_support_valid": (
                            True if effective_filter_support_status == "finite_causal_conservative_bound" else None
                        ),
                        "effective_filter_support_status": effective_filter_support_status,
                    }
                    try:
                        measurement = v31_module.compute_pipeline_m2_temporal_measurement(
                            input_window,
                            labels_for_measurement,
                            sample_rate_hz=float(sfreq),
                        window_start_sec=float(start_idx) / float(sfreq),
                        window_end_sec=float(end_idx) / realization_sfreq,
                            segment_id=segment_id,
                            input_branch=input_branch,
                            support_qc=support_qc,
                            acquisition_qc=acquisition_qc,
                        )
                    except Exception as exc:
                        logger.exception("MNPS v3.1 M2 4-second temporal realization failed on window %s", window_index)
                        measurement = {
                            "candidate_id": "M2",
                            "estimator_id": "TP.pipeline_native_window_lag1_decorrelation.v1",
                            "value": None,
                            "units": "1",
                            "computed": False,
                            "valid": False,
                            "qualified": False,
                            "status": "unsupported",
                            "reason": "candidate_estimator_exception",
                            "details": {"exception_type": type(exc).__name__},
                        }
                    realization_windows.append({
                        "window_index": int(window_index),
                        "realization_id": realization_id,
                        "support_window_sec_requested": support_window_sec,
                        "step_sec_requested": step_sec,
                        "window_start_sec": float(start_idx) / float(sfreq),
                        "window_end_sec": float(end_idx) / float(sfreq),
                        "source_sample_mapping": source_map,
                        "source_identity": dict(source_identity_metadata),
                        "measurements": [measurement],
                    })
                realization_status = "computed" if realization_windows else "no_windows"
                if not realization_windows:
                    realization_reason = "no_full_support_window_on_requested_grid"
            v31_temporal_realizations.append({
                "realization_id": realization_id,
                "candidate_id": "M2",
                "estimator_id": "TP.pipeline_native_window_lag1_decorrelation.v1",
                "support_window_sec_requested": support_window_sec,
                "step_sec_requested": step_sec,
                "grid_alignment": "preprocessed_array_sample_grid",
                "status": realization_status,
                "reason": realization_reason,
                "qualified": False,
                "coordinate_assignment": None,
                "windows": realization_windows,
            })

        recording_cfg = v31_cfg.get("recording_level_experimental", {})
        recording_cfg = recording_cfg if isinstance(recording_cfg, Mapping) else {}
        d2_recording_cfg = recording_cfg.get("d2_scg", {})
        d2_recording_cfg = d2_recording_cfg if isinstance(d2_recording_cfg, Mapping) else {}
        if bool(d2_recording_cfg.get("enabled", True)):
            recording_values = np.asarray(eeg_data)
            recording_sample_count = int(recording_values.shape[1]) if recording_values.ndim == 2 else 0
            recording_duration = recording_sample_count / float(sfreq) if float(sfreq) > 0.0 else 0.0
            recording_is_finite = bool(recording_values.size and np.isfinite(recording_values).all())
            quality_record = (
                support_provenance.get("source_quality_intervals", {})
                if isinstance(support_provenance, Mapping) else {}
            )
            quality_record = quality_record if isinstance(quality_record, Mapping) else {}
            quality_status = str(quality_record.get("status", "unknown"))
            quality_intervals = quality_record.get("intervals", [])
            crop_record = support_provenance.get("crop", {}) if isinstance(support_provenance, Mapping) else {}
            crop_record = crop_record if isinstance(crop_record, Mapping) else {}
            try:
                recording_source_start = float(crop_record.get("actual_tmin_from_source_start_sec", 0.0) or 0.0)
            except (TypeError, ValueError, OverflowError):
                recording_source_start = None
            bad_interval_free: bool | None = None
            if quality_status == "none_observed":
                bad_interval_free = True
            elif quality_status == "observed" and isinstance(quality_intervals, list) and recording_source_start is not None:
                recording_source_end = recording_source_start + recording_duration
                parsed_intervals: list[tuple[float, float]] = []
                intervals_valid = True
                for interval in quality_intervals:
                    if not isinstance(interval, Mapping):
                        intervals_valid = False
                        break
                    try:
                        interval_start = float(interval["start_sec"])
                        interval_end = interval_start + float(interval["duration_sec"])
                    except (KeyError, TypeError, ValueError, OverflowError):
                        intervals_valid = False
                        break
                    if not math.isfinite(interval_start) or not math.isfinite(interval_end):
                        intervals_valid = False
                        break
                    parsed_intervals.append((interval_start, interval_end))
                if intervals_valid:
                    bad_interval_free = not any(
                        interval_start < recording_source_end and interval_end > recording_source_start
                        for interval_start, interval_end in parsed_intervals
                    )

            eeglab_import = payload_meta.get("eeglab_import", {}) if isinstance(payload_meta, Mapping) else {}
            is_flattened_eeglab_epochs = (
                isinstance(eeglab_import, Mapping)
                and str(eeglab_import.get("source", "")).strip().lower() == "eeglab_epoched_concat"
            )
            raw_gap_samples = (
                support_provenance.get("time_gap_sample_indices")
                if isinstance(support_provenance, Mapping) else None
            )
            source_grid_reason = str(
                support_provenance.get("source_sample_grid_reason", "")
                if isinstance(support_provenance, Mapping) else ""
            ).casefold()
            known_discontinuity = any(
                token in source_grid_reason
                for token in ("gap", "segment", "discontinu", "boundary")
            )
            if is_flattened_eeglab_epochs or known_discontinuity:
                source_is_continuous: bool | None = False
                gap_free: bool | None = False
            else:
                # This is the regular sample grid returned by the continuous
                # preprocessing path. Exact mapping to the source clock is a
                # separate provenance field and is retained below as QC.
                source_is_continuous = True
                gap_free = (
                    bool(len(raw_gap_samples) == 0)
                    if isinstance(raw_gap_samples, (list, tuple))
                    else None
                )
            timestamps_uniform = bool(math.isfinite(float(sfreq)) and float(sfreq) > 0.0)
            within_recording_bounds = bool(recording_sample_count > 0)
            temporal_support = (
                support_provenance.get("temporal_support", {})
                if isinstance(support_provenance, Mapping) else {}
            )
            temporal_support = temporal_support if isinstance(temporal_support, Mapping) else {}
            temporal_support_status = str(temporal_support.get("status", "unknown"))
            estimator_filter_support_valid = (
                True if temporal_support_status in {"finite_causal", "finite_causal_conservative_bound"}
                else None
            )
            support_warnings: list[str] = []
            if not recording_is_finite:
                support_warnings.append("recording_contains_nonfinite_samples")
            if isinstance(support_provenance, Mapping) and support_provenance.get("source_sample_grid_supported") is not True:
                support_warnings.append("exact_source_sample_grid_mapping_not_verified")
            if quality_status not in {"none_observed", "observed"}:
                support_warnings.append("source_bad_interval_inventory_incomplete")
            if is_flattened_eeglab_epochs:
                support_warnings.append("flattened_epoched_source_has_no_continuous_recording_support")
            if not isinstance(raw_gap_samples, (list, tuple)):
                support_warnings.append("source_gap_inventory_not_available")
            if estimator_filter_support_valid is None:
                support_warnings.append("upstream_filter_edge_support_not_verified")
            support_qc = {
                "source_is_continuous": source_is_continuous,
                "timestamps_uniform": timestamps_uniform,
                "input_samples_finite": recording_is_finite,
                "gap_free": gap_free,
                "bad_interval_free": bad_interval_free,
                "within_recording_bounds": within_recording_bounds,
                "effective_filter_support_valid": estimator_filter_support_valid,
                "effective_filter_support_basis": "upstream_temporal_support_provenance; unknown_when_no_finite_causal_support_record",
                "source_sample_grid_supported": (
                    support_provenance.get("source_sample_grid_supported")
                    if isinstance(support_provenance, Mapping) else None
                ),
                "source_sample_grid_reason": (
                    support_provenance.get("source_sample_grid_reason")
                    if isinstance(support_provenance, Mapping) else None
                ),
                "continuity_assessment_basis": (
                    "flattened_epoch_concat_or_source_grid_reason_declares_discontinuity"
                    if is_flattened_eeglab_epochs or known_discontinuity
                    else "preprocessed_sample_array; exact_source_clock_mapping_reported_separately"
                ),
                "source_gap_inventory_count": len(raw_gap_samples) if isinstance(raw_gap_samples, (list, tuple)) else None,
                "source_bad_interval_status": quality_status,
                "source_bad_interval_count": len(quality_intervals) if isinstance(quality_intervals, list) else None,
                "warnings": sorted(set(support_warnings)),
            }
            processing_operations = (
                support_provenance.get("operations", [])
                if isinstance(support_provenance, Mapping) else []
            )
            preprocessing_metadata = {
                "signal_support_provenance_schema": (
                    support_provenance.get("schema") if isinstance(support_provenance, Mapping) else None
                ),
                "temporal_support": (
                    support_provenance.get("temporal_support") if isinstance(support_provenance, Mapping) else None
                ),
                "effective_time_support": (
                    support_provenance.get("effective_time_support") if isinstance(support_provenance, Mapping) else None
                ),
                "operations": [
                    {
                        "name": operation.get("name"),
                        "status": operation.get("status"),
                        "parameters": operation.get("parameters"),
                        "actual_kwargs": operation.get("actual_kwargs"),
                    }
                    for operation in processing_operations if isinstance(operation, Mapping)
                ] if isinstance(processing_operations, list) else [],
                "recording_duration_sec": float(recording_duration),
                "sample_rate_hz": float(sfreq),
                "upstream_filter_edge_support_status": (
                    (support_provenance.get("temporal_support") or {}).get("status", "unknown")
                    if isinstance(support_provenance, Mapping)
                    and isinstance(support_provenance.get("temporal_support", {}), Mapping)
                    else "unknown"
                ),
            }
            recording_input_branch = {
                **input_branch,
                "branch_id": "mndm.preprocessed_native_eeg_recording.v1",
                "aggregation_level": "recording",
            }
            try:
                v31_recording_level_experimental = (
                    v31_module.compute_pipeline_recording_level_experimental_measurements(
                        recording_values,
                        labels_for_measurement,
                        sample_rate_hz=float(sfreq),
                        support_qc=support_qc,
                        source_identity=source_id,
                        dataset_id=str(dataset_id) if dataset_id is not None else None,
                        input_branch=recording_input_branch,
                        preprocessing_metadata=preprocessing_metadata,
                        signal_units=str(amplitude_unit or "unknown"),
                        include_core19=bool(d2_recording_cfg.get("include_core19", True)),
                        include_ds004148_core60=bool(d2_recording_cfg.get("include_ds004148_core60", True)),
                    )
                )
            except Exception as exc:
                logger.exception("MNPS v3.1 recording-level D2/SCG failed; preserving window measurements")
                v31_recording_level_experimental = {
                    "schema": "mndm.mnps_v3_1.recording_level_experimental.v1",
                    "status": "candidate_pipeline_exception",
                    "qualified": False,
                    "coordinate_assignment": None,
                    "historical_slot": "D2",
                    "construct_id": "spatial_coupling_gradient",
                    "short_name": "SCG",
                    "aggregation_level": "recording",
                    "support_window_sec": float(recording_duration),
                    "step_sec": None,
                    "source_identity": source_id,
                    "support_qc": support_qc,
                    "reason": "candidate_pipeline_exception",
                    "exception_type": type(exc).__name__,
                }

    logger.info(
        "Computed %d EEG epochs (groups=%d). Timings: epoch+PSD=%.2fs (PSD=%.2fs)",
        len(df),
        len(group_defs),
        (t_epoch1 - t_psd0),
        (t_psd1 - t_psd0),
    )
    if v31_enabled:
        df.attrs["mnps_v3_1_measurements"] = {
            "schema": "mndm.mnps_v3_1.pipeline_measurements.v1",
            "status": "computed" if (
                v31_measurement_windows
                or any(realization.get("windows") for realization in v31_temporal_realizations)
                or v31_recording_level_experimental
            ) else "no_windows",
            "registry_version": "mndm.mnps_v3_1.construct_registry.v1",
            "standard_candidate_ids": ["M1", "M2", "M3", "D1", "D3", "E1", "E2", "E3"],
            "experimental_candidate_ids": ["D2"],
            "window_grid": {
                "window_length_sec_requested": float(v31_cfg.get("window_length_sec", 20.0) or 20.0),
                "step_sec_requested": float(v31_cfg.get("step_sec", v31_cfg.get("window_length_sec", 20.0)) or 20.0),
                "alignment": "preprocessed_array_sample_grid",
                "grid_is_separate_from_legacy_feature_epoch_grid": True,
            },
            "qualified": False,
            "coordinate_assignment": None,
            "source_identity": dict(source_identity_metadata),
            "windows": v31_measurement_windows,
            "temporal_realizations": v31_temporal_realizations,
            "recording_level_experimental_measurements": v31_recording_level_experimental,
            "acquisition_qc": acquisition_qc,
        }
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


