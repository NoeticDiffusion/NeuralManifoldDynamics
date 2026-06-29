"""Helpers for building additive AnchorState exports."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_ANCHOR_INDEX_SPECS: Dict[str, Dict[str, Sequence[str]]] = {
    "sympathetic_index": {
        "positive": (("ecg_hrv_hr_mean_bpm", "ecg_hr_bpm"), "ppg_rate_bpm", "pupil_dilation_velocity", "eog_blink_rate"),
        "negative": (("ecg_hrv_rmssd_ms", "ecg_rmssd"),),
    },
    "vagal_index": {
        "positive": (("ecg_hrv_rmssd_ms", "ecg_rmssd"), ("ecg_hrv_sdnn_ms", "ecg_sdnn")),
        "negative": (("ecg_hrv_hr_mean_bpm", "ecg_hr_bpm"),),
    },
    "vascular_index": {
        "positive": ("ppg_amplitude_mean",),
        "negative": ("ppg_amplitude_cv",),
    },
    "pupil_arousal_index": {
        "positive": ("pupil_dilation_velocity", "pupil_diameter_std", "pupil_diameter_mean"),
        "negative": (),
    },
}


def _resolve_anchor_state_config(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Resolve anchor-state config with optional defaults."""
    raw = config.get("anchor_state", {}) if isinstance(config, Mapping) else {}
    cfg = dict(raw) if isinstance(raw, Mapping) else {}
    cfg.setdefault("enabled", False)
    cfg.setdefault("quality_columns", {})
    cfg.setdefault("index_specs", DEFAULT_ANCHOR_INDEX_SPECS)
    return cfg


def _feature_series(
    robust_z_values: np.ndarray,
    robust_z_names: Sequence[str],
    candidates: Sequence[Any],
) -> tuple[np.ndarray | None, list[str]]:
    """Return the mean robust-z series across available candidate columns."""
    if robust_z_values.size == 0 or not robust_z_names:
        return None, []
    name_to_idx = {str(name): idx for idx, name in enumerate(robust_z_names)}
    resolved: list[str] = []
    for candidate in candidates:
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
            for name in candidate:
                name_str = str(name)
                if name_str in name_to_idx:
                    resolved.append(name_str)
                    break
            continue
        name = str(candidate)
        if name in name_to_idx:
            resolved.append(name)
    used = list(dict.fromkeys(resolved))
    if not used:
        return None, []
    cols = [name_to_idx[name] for name in used]
    values = np.asarray(robust_z_values[:, cols], dtype=np.float32)
    counts = np.sum(np.isfinite(values), axis=1)
    summed = np.nansum(values, axis=1)
    series = np.divide(
        summed,
        np.maximum(counts, 1),
        out=np.full(values.shape[0], np.nan, dtype=np.float32),
        where=counts > 0,
    )
    return series.astype(np.float32, copy=False), used


def _combine_signed_series(
    robust_z_values: np.ndarray,
    robust_z_names: Sequence[str],
    *,
    positive: Sequence[Any],
    negative: Sequence[Any],
) -> tuple[np.ndarray, dict[str, list[str]]]:
    """Combine positive and negative candidate series into one signed index."""
    pos_series, pos_used = _feature_series(robust_z_values, robust_z_names, positive)
    neg_series, neg_used = _feature_series(robust_z_values, robust_z_names, negative)
    base = np.full(robust_z_values.shape[0], np.nan, dtype=np.float32)
    contributions: list[np.ndarray] = []
    if pos_series is not None:
        contributions.append(pos_series.astype(np.float32, copy=False))
    if neg_series is not None:
        contributions.append((-neg_series).astype(np.float32, copy=False))
    if contributions:
        stacked = np.vstack(contributions)
        counts = np.sum(np.isfinite(stacked), axis=0)
        summed = np.nansum(stacked, axis=0)
        base = np.divide(
            summed,
            np.maximum(counts, 1),
            out=np.full(stacked.shape[1], np.nan, dtype=np.float32),
            where=counts > 0,
        )
    return base.astype(np.float32, copy=False), {"positive": pos_used, "negative": neg_used}


def _extract_quality_series(features_df: pd.DataFrame, candidates: Sequence[str]) -> np.ndarray | None:
    """Extract the first available quality-like column from the feature frame."""
    for candidate in candidates:
        if candidate not in features_df.columns:
            continue
        arr = pd.to_numeric(features_df[candidate], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        return arr.astype(np.float32, copy=False)
    return None


def _normalize_candidate_spec(values: Sequence[Any]) -> list[Any]:
    """Normalize candidate specs while preserving fallback tiers."""
    normalized: list[Any] = []
    for value in values:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            normalized.append([str(v) for v in value])
        else:
            normalized.append(str(value))
    return normalized


def _finite_gradient(series: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Compute a stable gradient while preserving missing support."""
    out = np.full_like(series, np.nan, dtype=np.float32)
    finite = np.isfinite(series) & np.isfinite(time)
    if np.sum(finite) < 2:
        return out
    x = np.asarray(time[finite], dtype=np.float64)
    y = np.asarray(series[finite], dtype=np.float64)
    grad = np.gradient(y, x)
    out[finite] = np.asarray(grad, dtype=np.float32)
    return out


def build_anchor_state_exports(
    *,
    features_df: pd.DataFrame,
    robust_z_values: np.ndarray | None,
    robust_z_names: Sequence[str] | None,
    time: np.ndarray,
    config: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build `anchor_state`, `anchor_state_dot`, and `anchor_quality` export groups."""
    cfg = _resolve_anchor_state_config(config)
    if not bool(cfg.get("enabled", False)):
        return {}, {}, {}, {}
    if robust_z_values is None or robust_z_names is None:
        return {}, {}, {}, {}
    matrix = np.asarray(robust_z_values, dtype=np.float32)
    names = [str(v) for v in robust_z_names]
    if matrix.ndim != 2 or matrix.shape[0] != len(time):
        return {}, {}, {}, {}

    index_specs = cfg.get("index_specs", DEFAULT_ANCHOR_INDEX_SPECS)
    if not isinstance(index_specs, Mapping):
        index_specs = DEFAULT_ANCHOR_INDEX_SPECS

    anchor_names: list[str] = []
    anchor_cols: list[np.ndarray] = []
    source_manifest: Dict[str, Any] = {}
    for index_name, spec in index_specs.items():
        if not isinstance(spec, Mapping):
            continue
        series, used = _combine_signed_series(
            matrix,
            names,
            positive=_normalize_candidate_spec(spec.get("positive", []) or []),
            negative=_normalize_candidate_spec(spec.get("negative", []) or []),
        )
        if not np.isfinite(series).any():
            continue
        anchor_names.append(str(index_name))
        anchor_cols.append(series.astype(np.float32, copy=False))
        source_manifest[str(index_name)] = used

    if not anchor_cols:
        return {}, {}, {}, {}

    anchor_matrix = np.column_stack(anchor_cols).astype(np.float32, copy=False)
    counts = np.sum(np.isfinite(anchor_matrix), axis=1)
    summed = np.nansum(anchor_matrix, axis=1)
    anchor_index = np.divide(
        summed,
        np.maximum(counts, 1),
        out=np.full(anchor_matrix.shape[0], np.nan, dtype=np.float32),
        where=counts > 0,
    ).astype(np.float32, copy=False)
    anchor_names.append("anchor_index")
    anchor_matrix = np.column_stack([anchor_matrix, anchor_index]).astype(np.float32, copy=False)

    dot_matrix = np.column_stack(
        [_finite_gradient(anchor_matrix[:, idx], np.asarray(time, dtype=np.float64)) for idx in range(anchor_matrix.shape[1])]
    ).astype(np.float32, copy=False)

    quality_candidates = {
        "ecg_quality": ("ecg_hrv_quality_score", "ecg_quality_score", "qc_ok_ecg_hrv", "qc_ok_ecg"),
        "ppg_quality": ("ppg_quality_score", "qc_ok_ppg"),
        "pupil_quality": ("pupil_quality_score", "qc_ok_pupil"),
    }
    quality_names: list[str] = []
    quality_cols: list[np.ndarray] = []
    for quality_name, candidates in quality_candidates.items():
        series = _extract_quality_series(features_df, candidates)
        if series is None:
            continue
        quality_names.append(str(quality_name))
        quality_cols.append(series.astype(np.float32, copy=False))
    support_fraction = (np.sum(np.isfinite(anchor_matrix[:, :-1]), axis=1) / max(anchor_matrix.shape[1] - 1, 1)).astype(np.float32)
    quality_names.append("anchor_support_fraction")
    quality_cols.append(support_fraction)
    quality_matrix = np.column_stack(quality_cols).astype(np.float32, copy=False)

    attrs = {
        "contract": "mndm.anchor_state.v1",
        "source": "features_robust_z",
        "source_features_json": json.dumps(source_manifest, ensure_ascii=False, separators=(",", ":")),
    }
    anchor_state = {"values": anchor_matrix, "names": anchor_names, "attrs": attrs}
    anchor_state_dot = {"values": dot_matrix, "names": list(anchor_names), "attrs": {"source": "anchor_state"}}
    anchor_quality = {
        "values": quality_matrix,
        "names": quality_names,
        "attrs": {"contract": "mndm.anchor_quality.v1"},
    }
    diagnostics = {
        "enabled": True,
        "names": list(anchor_names),
        "quality_names": list(quality_names),
        "source_features": source_manifest,
        "available_feature_count": int(len(names)),
    }
    return anchor_state, anchor_state_dot, anchor_quality, diagnostics
