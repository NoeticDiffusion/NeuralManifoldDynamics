"""Experimental MEG candidates for the separate MNPS 3.4 surface.

Inherited features deliberately use the legacy median-waveform estimator. The
two alpha candidates are channel-wise estimators and retain their per-window
sensor support. MAG and GRAD are accepted only from explicit preprocessing
outputs; this module never infers an unknown sensor family.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import signal


MEG_V34_SCHEMA = "mndm.meg_mnps_v3_4.experimental.v1"
MEG_V34_DEFINITION_VERSION = "3.4-meg-exp.1"

MEG_V34_VARIANTS: dict[str, tuple[str, ...]] = {
    "existing_coordinate_set": (
        "delta_theta_level",
        "alpha_level",
        "beta_alpha_ratio",
        "gamma_level",
        "hjorth_mobility",
        "alpha_theta_ratio",
        "permutation_entropy",
        "hjorth_complexity",
        "sample_entropy",
    ),
    "removal_only": (
        "delta_theta_level",
        "alpha_level",
        "beta_alpha_ratio",
        "gamma_level",
        "hjorth_mobility",
        "alpha_theta_ratio",
        "permutation_entropy",
    ),
    "alpha_replacement": (
        "delta_theta_level",
        "alpha_level",
        "beta_alpha_ratio",
        "gamma_level",
        "hjorth_mobility",
        "alpha_theta_ratio",
        "permutation_entropy",
        "alpha_centroid_hz",
        "alpha_envelope_cv",
    ),
}


def resolve_meg_v3_4_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the explicit MEG 3.4 overlay, if present."""
    value = config.get("mnps_v3_4_meg") if isinstance(config, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _stable_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _json_text(values: Sequence[Any]) -> str:
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _channel_hashes(labels: Sequence[str]) -> tuple[str, str]:
    ordered = [str(value) for value in labels]
    return _stable_hash(sorted(ordered)), _stable_hash(ordered)


def _sample_entropy_reason(values: np.ndarray, *, m: int, r_scale: float, max_n: int) -> tuple[float, str]:
    """Return native SampEn and an explicit failure reason without substitution."""
    from ...features.meg import _sample_entropy

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0 or not np.isfinite(arr).all():
        return float("nan"), "nonfinite_or_empty_median_waveform"
    if arr.size < 2 * (m + 1):
        return float("nan"), "too_few_samples"
    sampled = arr
    if len(sampled) > max_n:
        idx = np.round(np.linspace(0, len(sampled) - 1, max_n)).astype(int)
        sampled = sampled[idx]
    scale = float(np.std(sampled, ddof=1)) if len(sampled) > 1 else 0.0
    if not np.isfinite(scale) or scale <= 0.0:
        return float("nan"), "zero_variance"
    value = float(_sample_entropy(arr, m=m, r_scale=r_scale, max_n=max_n))
    if not np.isfinite(value):
        return float("nan"), "no_template_matches_or_zero_extension_probability"
    return value, "computed"


def _safe_log10(value: Any) -> tuple[float, str]:
    try:
        number = float(value)
    except Exception:
        return float("nan"), "non_numeric_input"
    if not np.isfinite(number):
        return float("nan"), "nonfinite_input"
    if number <= 0.0:
        return float("nan"), "nonpositive_power"
    return float(np.log10(number)), "computed"


def _resolve_inherited_psd_identity(config: Mapping[str, Any], sfreq: float) -> tuple[str, str]:
    from ...features.eeg import psd_array_multitaper
    from ...features.meg import _resolve_meg_psd_cfg

    features_cfg = config.get("features", {}) if isinstance(config, Mapping) else {}
    method, _, _, _ = _resolve_meg_psd_cfg(
        features_cfg if isinstance(features_cfg, Mapping) else {}, sfreq
    )
    if method == "multitaper" and psd_array_multitaper is not None:
        return "mne.time_frequency.psd_array_multitaper", "none"
    if method == "multitaper":
        return "scipy.signal.welch", "multitaper_unavailable"
    if method == "welch":
        return "scipy.signal.welch", "configured_welch"
    return "scipy.signal.welch", f"unsupported_configured_method:{method}"


def _validate_config(config: Mapping[str, Any]) -> tuple[list[str], Mapping[str, Any], Mapping[str, Any]]:
    cfg = resolve_meg_v3_4_config(config)
    if not cfg or not bool(cfg.get("enabled", False)):
        return [], {}, {}
    version = str(cfg.get("definition_version", MEG_V34_DEFINITION_VERSION))
    if version != MEG_V34_DEFINITION_VERSION:
        raise ValueError(f"Unsupported MEG 3.4 definition_version: {version}")
    selected = cfg.get("variants", list(MEG_V34_VARIANTS))
    if not isinstance(selected, Sequence) or isinstance(selected, (str, bytes)):
        raise ValueError("mnps_v3_4_meg.variants must be a list of variant ids")
    variants = [str(value) for value in selected]
    if not variants:
        raise ValueError("mnps_v3_4_meg.variants must select at least one variant")
    unknown = sorted(set(variants) - set(MEG_V34_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown MEG 3.4 variants: {unknown}")
    if len(set(variants)) != len(variants):
        raise ValueError("mnps_v3_4_meg.variants contains duplicate ids")
    if str(cfg.get("sensor_family_policy", "explicit_mag_grad")) != "explicit_mag_grad":
        raise ValueError("MEG 3.4 requires sensor_family_policy=explicit_mag_grad")
    scaling = cfg.get("scaling", {})
    if not isinstance(scaling, Mapping) or str(scaling.get("mode", "raw_only")) != "raw_only":
        raise ValueError("This MEG 3.4 implementation exports raw features only; scaling.mode must be raw_only")
    if scaling.get("anchor_manifest") or scaling.get("anchor_manifest_by_condition"):
        raise ValueError("EEG or other anchors cannot be applied to the raw MEG 3.4 surface")
    estimators = cfg.get("estimators", {})
    if not isinstance(estimators, Mapping):
        raise ValueError("mnps_v3_4_meg.estimators must be a mapping")
    centroid = estimators.get("alpha_centroid_hz", {})
    envelope = estimators.get("alpha_envelope_cv", {})
    if not isinstance(centroid, Mapping) or not isinstance(envelope, Mapping):
        raise ValueError("MEG alpha estimator settings must be mappings")
    return variants, centroid, envelope


def _alpha_centroid(
    data: np.ndarray,
    labels: Sequence[str],
    start: int,
    stop: int,
    sfreq: float,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    band = cfg.get("frequency_range_hz", (7.0, 13.0))
    low, high = float(band[0]), float(band[1])
    psd_cfg = cfg.get("psd", {})
    psd_cfg = psd_cfg if isinstance(psd_cfg, Mapping) else {}
    method = str(psd_cfg.get("method", "welch")).strip().lower()
    if method != "welch":
        raise ValueError("MEG alpha_centroid_hz currently requires the frozen Welch estimator")
    segment_sec = float(psd_cfg.get("segment_sec", 4.0))
    overlap_fraction = float(psd_cfg.get("overlap_fraction", 0.5))
    if not 0.0 <= overlap_fraction < 1.0 or segment_sec <= 0.0:
        raise ValueError("Invalid alpha-centroid Welch segment/overlap settings")
    window_data = np.asarray(data[:, start:stop], dtype=np.float64)
    full_labels = [str(value) for value in labels]
    used_labels: list[str] = []
    result = {
        "value": float("nan"),
        "status": "unsupported",
        "reason": "unknown",
        "channel_labels": used_labels,
        "channel_count": 0,
        "psd_method_id": "scipy.signal.welch",
    }
    if window_data.ndim != 2 or window_data.shape[0] != len(full_labels):
        result.update(status="failed", reason="channel_label_data_shape_mismatch")
        return result
    if start < 0 or stop > data.shape[1] or stop <= start:
        result.update(status="failed", reason="window_outside_source_support")
        return result
    valid = np.isfinite(window_data).all(axis=1)
    if not valid.any():
        result.update(status="unsupported", reason="no_finite_channel_support")
        return result
    channel_data = window_data[valid]
    used_labels = [label for label, keep in zip(full_labels, valid.tolist()) if keep]
    nperseg = min(max(4, int(round(segment_sec * sfreq))), channel_data.shape[1])
    noverlap = min(int(round(nperseg * overlap_fraction)), nperseg - 1)
    try:
        freqs, psd = signal.welch(
            channel_data,
            fs=sfreq,
            window=str(psd_cfg.get("window", "hann")),
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=str(psd_cfg.get("detrend", "constant")),
            scaling=str(psd_cfg.get("scaling", "density")),
            axis=1,
        )
    except Exception as exc:
        result.update(
            status="failed",
            reason=f"welch_error:{type(exc).__name__}",
            channel_labels=used_labels,
            channel_count=len(used_labels),
        )
        return result
    in_band = (freqs >= low) & (freqs <= high)
    if not in_band.any():
        result.update(
            status="unsupported",
            reason="alpha_band_outside_psd_grid",
            channel_labels=used_labels,
            channel_count=len(used_labels),
        )
        return result
    median_psd = np.median(psd[:, in_band], axis=0)
    denominator = float(np.sum(median_psd))
    if not np.isfinite(denominator) or denominator <= 0.0:
        result.update(
            status="unsupported",
            reason="nonpositive_psd_denominator",
            channel_labels=used_labels,
            channel_count=len(used_labels),
        )
        return result
    centroid = float(np.sum(freqs[in_band] * median_psd) / denominator)
    status = "computed" if len(used_labels) == len(full_labels) else "computed_partial_channel_support"
    result.update(
        value=centroid,
        status=status,
        reason="none" if status == "computed" else "nonfinite_channels_excluded_and_listed",
        channel_labels=used_labels,
        channel_count=len(used_labels),
        nperseg_samples=int(nperseg),
        noverlap_samples=int(noverlap),
    )
    return result


def _alpha_envelope_by_channel(
    data: np.ndarray,
    sfreq: float,
    cfg: Mapping[str, Any],
) -> tuple[np.ndarray, list[list[tuple[int, int]]], str]:
    band = cfg.get("band_hz", (8.0, 12.0))
    low, high = float(band[0]), float(band[1])
    filter_cfg = cfg.get("bandpass", {})
    filter_cfg = filter_cfg if isinstance(filter_cfg, Mapping) else {}
    order = int(filter_cfg.get("order", 4))
    envelopes = np.full(data.shape, np.nan, dtype=np.float64)
    runs_by_channel: list[list[tuple[int, int]]] = []
    try:
        sos = signal.butter(order, [low, high], btype="bandpass", fs=sfreq, output="sos")
    except Exception as exc:
        return envelopes, [[] for _ in range(data.shape[0])], f"filter_configuration_error:{type(exc).__name__}"
    padlen = 3 * (2 * len(sos) + 1)
    for channel_index, channel in enumerate(data):
        channel_runs: list[tuple[int, int]] = []
        for run_start, run_end in _runs(np.isfinite(channel)):
            segment = np.asarray(channel[run_start:run_end], dtype=np.float64)
            if len(segment) <= padlen:
                continue
            try:
                filtered = signal.sosfiltfilt(sos, segment)
                envelope = np.abs(signal.hilbert(filtered))
            except Exception:
                continue
            envelopes[channel_index, run_start:run_end] = envelope
            channel_runs.append((run_start, run_end))
        runs_by_channel.append(channel_runs)
    return envelopes, runs_by_channel, "none"


def _alpha_envelope_cv(
    data: np.ndarray,
    labels: Sequence[str],
    envelopes: np.ndarray,
    runs_by_channel: Sequence[Sequence[tuple[int, int]]],
    start: int,
    stop: int,
    sfreq: float,
    cfg: Mapping[str, Any],
    filter_status: str,
) -> dict[str, Any]:
    full_labels = [str(value) for value in labels]
    result: dict[str, Any] = {
        "value": float("nan"),
        "status": "unsupported",
        "reason": filter_status,
        "channel_labels": [],
        "channel_count": 0,
        "filter_method_id": "scipy.signal.sosfiltfilt+scipy.signal.hilbert",
    }
    if filter_status != "none":
        result["status"] = "failed"
        return result
    if start < 0 or stop > data.shape[1] or stop <= start:
        result.update(status="failed", reason="window_outside_source_support")
        return result
    guard = max(0, int(np.ceil(float(cfg.get("edge_support_guard_sec", 10.0)) * sfreq)))
    per_channel_cv: list[float] = []
    used_labels: list[str] = []
    had_containing_run = False
    had_run_without_guard = False
    for idx, channel_runs in enumerate(runs_by_channel):
        supporting = next(((a, b) for a, b in channel_runs if a <= start and stop <= b), None)
        if supporting is None:
            continue
        had_containing_run = True
        run_start, run_end = supporting
        if start - run_start < guard or run_end - stop < guard:
            had_run_without_guard = True
            continue
        values = envelopes[idx, start:stop]
        if not values.size or not np.isfinite(values).all():
            continue
        mean = float(np.mean(values))
        if not np.isfinite(mean) or mean <= 0.0:
            continue
        cv = float(np.std(values, ddof=0) / mean)
        if np.isfinite(cv):
            per_channel_cv.append(cv)
            used_labels.append(full_labels[idx])
    if not per_channel_cv:
        result.update(
            status="unsupported",
            reason=(
                "within_declared_segment_edge_guard"
                if had_run_without_guard
                else "no_continuous_channel_support"
            ),
        )
        return result
    total = len(full_labels)
    status = "computed" if len(used_labels) == total else "computed_partial_channel_support"
    result.update(
        value=float(np.median(np.asarray(per_channel_cv, dtype=np.float64))),
        status=status,
        reason="none" if status == "computed" else "channels_without_continuous_guarded_support_excluded_and_listed",
        channel_labels=used_labels,
        channel_count=len(used_labels),
        edge_support_guard_sec=float(cfg.get("edge_support_guard_sec", 10.0)),
    )
    return result


def _native_sample_entropy(
    data: np.ndarray,
    start: int,
    stop: int,
    *,
    m: int,
    r_scale: float,
    max_n: int,
) -> tuple[float, str]:
    window = np.asarray(data[:, start:stop], dtype=np.float64)
    if start < 0 or stop > data.shape[1] or stop <= start:
        return float("nan"), "window_outside_source_support"
    if window.size == 0 or not np.isfinite(window).all():
        return float("nan"), "nonfinite_required_channel_support"
    median_waveform = np.median(window, axis=0)
    return _sample_entropy_reason(median_waveform, m=m, r_scale=r_scale, max_n=max_n)


def compute_meg_v3_4_features(
    signals: Mapping[str, Any],
    inherited_features: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Add experimental MEG 3.4 measurements beside, not into, legacy fields."""
    variants, centroid_cfg, envelope_cfg = _validate_config(config)
    if not variants:
        return pd.DataFrame(index=inherited_features.index)
    signal_map = signals.get("signals", {}) if isinstance(signals, Mapping) else {}
    channels_map = signals.get("channels", {}) if isinstance(signals, Mapping) else {}
    sfreq = float(signals.get("sfreq", 250.0) or 250.0)
    features_cfg = config.get("features", {}) if isinstance(config, Mapping) else {}
    pe_cfg = features_cfg.get("permutation_entropy", {}) if isinstance(features_cfg, Mapping) else {}
    pe_cfg = pe_cfg if isinstance(pe_cfg, Mapping) else {}
    se_cfg = (resolve_meg_v3_4_config(config).get("estimators", {}) or {}).get("sample_entropy", {})
    se_cfg = se_cfg if isinstance(se_cfg, Mapping) else {}
    se_m = int(se_cfg.get("m", 2))
    se_r = float(se_cfg.get("r_scale", 0.2))
    se_max_n = int(se_cfg.get("max_n", 300))
    psd_method_id, psd_fallback_reason = _resolve_inherited_psd_identity(config, sfreq)
    records = pd.DataFrame(index=inherited_features.index)

    explicit_families: list[str] = []
    for family in ("mag", "grad"):
        key = f"meg_{family}"
        data = signal_map.get(key) if isinstance(signal_map, Mapping) else None
        if isinstance(data, np.ndarray) and data.ndim == 2 and data.size:
            explicit_families.append(family)

    if not explicit_families:
        combined = signal_map.get("meg") if isinstance(signal_map, Mapping) else None
        status = "unknown_sensor_type" if isinstance(combined, np.ndarray) and combined.size else "no_explicit_meg_family"
        records["mnps_v3_4_meg_sensor_family_status"] = status
        records["mnps_v3_4_meg_sensor_family_reason"] = (
            "explicit_meg_mag_or_meg_grad_required" if status == "unknown_sensor_type" else "no_family_data"
        )
        return records

    starts = pd.to_numeric(inherited_features.get("t_start", pd.Series(np.zeros(len(records)))), errors="coerce").to_numpy(dtype=float)
    ends = pd.to_numeric(inherited_features.get("t_end", pd.Series(np.zeros(len(records)))), errors="coerce").to_numpy(dtype=float)

    for family in explicit_families:
        key = f"meg_{family}"
        data = np.asarray(signal_map[key], dtype=np.float64)
        labels_raw = channels_map.get(key, []) if isinstance(channels_map, Mapping) else []
        labels = [str(value) for value in labels_raw] if isinstance(labels_raw, Sequence) and not isinstance(labels_raw, (str, bytes)) else []
        prefix = f"mnps_v3_4_meg_{family}"
        family_status = "explicit_sensor_family"
        if len(labels) != data.shape[0] or not labels:
            family_status = "missing_or_mismatched_channel_labels"
            labels = []
        set_hash, order_hash = _channel_hashes(labels)
        records[f"{prefix}_sensor_family_status"] = family_status
        records[f"{prefix}_channel_labels_json"] = _json_text(labels)
        records[f"{prefix}_channel_set_hash"] = set_hash
        records[f"{prefix}_channel_order_hash"] = order_hash
        records[f"{prefix}_channel_count"] = len(labels)
        records[f"{prefix}_inherited_psd_method_id"] = psd_method_id
        records[f"{prefix}_inherited_psd_fallback_reason"] = psd_fallback_reason

        family_features: dict[str, np.ndarray] = {}
        family_reasons: dict[str, list[str]] = {}
        log_features = {
            "delta_theta_level": ("delta", "theta"),
            "alpha_level": ("alpha",),
            "gamma_level": ("gamma",),
        }
        for feature_name, input_names in log_features.items():
            vals = np.full(len(records), np.nan, dtype=np.float64)
            reasons: list[str] = []
            for row_idx in range(len(records)):
                logs = []
                failure = "computed"
                for input_name in input_names:
                    col = f"meg_{family}_{input_name}"
                    value = inherited_features.iloc[row_idx].get(col, np.nan)
                    log_value, why = _safe_log10(value)
                    if not np.isfinite(log_value):
                        failure = why
                        break
                    logs.append(log_value)
                if failure == "computed":
                    vals[row_idx] = float(np.mean(logs))
                reasons.append("none" if failure == "computed" else failure)
            family_features[feature_name] = vals
            family_reasons[feature_name] = reasons

        direct_map = {
            "beta_alpha_ratio": "beta_alpha",
            "hjorth_mobility": "hjorth_mobility",
            "alpha_theta_ratio": "alpha_theta",
            "hjorth_complexity": "hjorth_complexity",
        }
        for coord_name, feature_name in direct_map.items():
            col = f"meg_{family}_{feature_name}"
            values = pd.to_numeric(
                inherited_features[col] if col in inherited_features else pd.Series(np.full(len(records), np.nan)),
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            family_features[coord_name] = values
            family_reasons[coord_name] = ["none" if np.isfinite(v) else "nonfinite_inherited_feature" for v in values]

        entropy_col = f"meg_{family}_permutation_entropy"
        metric_col = f"meg_{family}_entropy_metric"
        degraded_col = f"meg_{family}_entropy_degraded_mode"
        entropy_values = pd.to_numeric(
            inherited_features[entropy_col] if entropy_col in inherited_features else pd.Series(np.full(len(records), np.nan)),
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        entropy_metrics = inherited_features[metric_col].astype(str).tolist() if metric_col in inherited_features else ["unknown"] * len(records)
        entropy_degraded = inherited_features[degraded_col].fillna(False).astype(bool).tolist() if degraded_col in inherited_features else [False] * len(records)
        entropy_reasons = inherited_features.get(
            f"meg_{family}_entropy_reason", pd.Series([""] * len(records))
        ).fillna("").astype(str).tolist()
        corrected_entropy = np.full(len(records), np.nan, dtype=np.float64)
        corrected_entropy_reasons: list[str] = []
        for i, value in enumerate(entropy_values):
            if entropy_metrics[i] != "permutation_entropy":
                corrected_entropy_reasons.append(f"unexpected_metric:{entropy_metrics[i]}")
            elif entropy_degraded[i]:
                corrected_entropy_reasons.append("permutation_entropy_degraded_mode")
            elif not np.isfinite(value):
                corrected_entropy_reasons.append(entropy_reasons[i] or "permutation_entropy_nonfinite")
            else:
                corrected_entropy[i] = value
                corrected_entropy_reasons.append("none")
        family_features["permutation_entropy"] = corrected_entropy
        family_reasons["permutation_entropy"] = corrected_entropy_reasons

        sample_entropy_values = np.full(len(records), np.nan, dtype=np.float64)
        sample_entropy_reasons: list[str] = []
        if family_status == "explicit_sensor_family":
            for row_idx, (start_sec, end_sec) in enumerate(zip(starts, ends)):
                start = int(round(start_sec * sfreq))
                stop = int(round(end_sec * sfreq))
                value, reason = _native_sample_entropy(
                    data, start, stop, m=se_m, r_scale=se_r, max_n=se_max_n
                )
                sample_entropy_values[row_idx] = value
                sample_entropy_reasons.append(reason)
        else:
            sample_entropy_reasons = ["missing_or_mismatched_channel_labels"] * len(records)
        family_features["sample_entropy"] = sample_entropy_values
        family_reasons["sample_entropy"] = sample_entropy_reasons
        for coord_name, values in family_features.items():
            records[f"{prefix}_{coord_name}"] = values
            records[f"{prefix}_{coord_name}_status"] = [
                "computed" if np.isfinite(value) else "unsupported" for value in values
            ]
            records[f"{prefix}_{coord_name}_reason"] = family_reasons[coord_name]

        if family_status == "explicit_sensor_family":
            envelopes, runs_by_channel, filter_status = _alpha_envelope_by_channel(data, sfreq, envelope_cfg)
            for candidate_name in ("alpha_centroid_hz", "alpha_envelope_cv"):
                candidate_values = np.full(len(records), np.nan, dtype=np.float64)
                statuses: list[str] = []
                reasons: list[str] = []
                counts: list[int] = []
                labels_json: list[str] = []
                set_hashes: list[str] = []
                order_hashes: list[str] = []
                method_ids: list[str] = []
                for row_idx, (start_sec, end_sec) in enumerate(zip(starts, ends)):
                    start = int(round(start_sec * sfreq))
                    stop = int(round(end_sec * sfreq))
                    result = (
                        _alpha_centroid(data, labels, start, stop, sfreq, centroid_cfg)
                        if candidate_name == "alpha_centroid_hz"
                        else _alpha_envelope_cv(
                            data, labels, envelopes, runs_by_channel, start, stop,
                            sfreq, envelope_cfg, filter_status,
                        )
                    )
                    candidate_values[row_idx] = float(result["value"])
                    statuses.append(str(result["status"]))
                    reasons.append(str(result["reason"]))
                    used = list(result["channel_labels"])
                    counts.append(int(result["channel_count"]))
                    labels_json.append(_json_text(used))
                    used_set_hash, used_order_hash = _channel_hashes(used)
                    set_hashes.append(used_set_hash)
                    order_hashes.append(used_order_hash)
                    method_ids.append(str(result.get("psd_method_id", result.get("filter_method_id", "unknown"))))
                records[f"{prefix}_{candidate_name}"] = candidate_values
                records[f"{prefix}_{candidate_name}_status"] = statuses
                records[f"{prefix}_{candidate_name}_reason"] = reasons
                records[f"{prefix}_{candidate_name}_channel_count"] = np.asarray(counts, dtype=np.int32)
                records[f"{prefix}_{candidate_name}_channel_labels_json"] = labels_json
                records[f"{prefix}_{candidate_name}_channel_set_hash"] = set_hashes
                records[f"{prefix}_{candidate_name}_channel_order_hash"] = order_hashes
                if candidate_name == "alpha_centroid_hz":
                    records[f"{prefix}_{candidate_name}_estimator_id"] = method_ids
                else:
                    records[f"{prefix}_{candidate_name}_estimator_id"] = method_ids
        else:
            for candidate_name in ("alpha_centroid_hz", "alpha_envelope_cv"):
                records[f"{prefix}_{candidate_name}"] = np.full(len(records), np.nan)
                records[f"{prefix}_{candidate_name}_status"] = "unsupported"
                records[f"{prefix}_{candidate_name}_reason"] = "missing_or_mismatched_channel_labels"
                records[f"{prefix}_{candidate_name}_channel_count"] = 0
                records[f"{prefix}_{candidate_name}_channel_labels_json"] = "[]"
                records[f"{prefix}_{candidate_name}_channel_set_hash"] = _stable_hash([])
                records[f"{prefix}_{candidate_name}_channel_order_hash"] = _stable_hash([])
                records[f"{prefix}_{candidate_name}_estimator_id"] = "unsupported_missing_explicit_sensor_family"

    records["mnps_v3_4_meg_definition_version"] = MEG_V34_DEFINITION_VERSION
    records["mnps_v3_4_meg_selected_variants_json"] = _json_text(variants)
    return records


__all__ = [
    "MEG_V34_DEFINITION_VERSION",
    "MEG_V34_SCHEMA",
    "MEG_V34_VARIANTS",
    "compute_meg_v3_4_features",
    "resolve_meg_v3_4_config",
]
