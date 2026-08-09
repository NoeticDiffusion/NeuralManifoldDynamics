"""Helpers for building additive AnchorState exports.

AnchorState v0.1 (contract: mndm.anchor_state.v1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Four component indices from cardiac and pupillometry features:
    sympathetic_index, vagal_index, vascular_index, pupil_arousal_index
Composite: anchor_index (nanmean of available components).

AnchorState v0.2 (contract: mndm.anchor_state.v2) — MNDM 2.5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Extends v0.1 with three optional new component families:
    resp_anchor_index       – respiratory regularity + depth, inversely rated
    ocular_stability_index  – eye stillness (low blink + saccade rate)
    cardioresp_anchor_index – composite of cardioresp_coupling_index, PLV, and spectral coherence between cardiac and respiratory rhythms

When enabled via ``anchor_state.v2.enabled: true`` in config, these three
additional indices are emitted alongside the v0.1 set.  The v0.2 composite
``anchor_index_v02`` is the nanmean across ALL active component families
(v0.1 + v0.2).  The legacy ``anchor_index`` (v0.1 nanmean only) is
preserved verbatim for backward compatibility.

Principle (science lead note)
------------------------------
> Do not hide respiration inside one global anchor score.  Report respiration,
> HRV, cardiorespiratory coupling, pupil, and ocular features separately.

Column presence rules
---------------------
* v0.1 columns are always emitted when ``anchor_state.enabled: true``.
* v0.2 extension columns are appended only when the required source feature
  columns (resp_*, eog_*, cardioresp_*) are actually present in the feature
  table.  A config flag can also disable them explicitly.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd


ANCHOR_GUARD_POLICY_VERSION = "mndm.anchor_guard.v1"
DEFAULT_ANCHOR_ABS_MAX = 1e4


DEFAULT_ANCHOR_INDEX_SPECS: Dict[str, Dict[str, Any]] = {
    "sympathetic_index": {
        "positive": (("ecg_hrv_hr_mean_bpm", "ecg_hr_bpm"), "ppg_rate_bpm", "pupil_dilation_velocity"),
        "negative": (("ecg_hrv_rmssd_ms", "ecg_rmssd"),),
        # EOG remains an ocular signal, not standalone evidence of sympathetic
        # physiology. At least one ECG/PPG/pupillometry source must be present
        # in the raw feature surface on each row.
        "eligibility": {
            "require_any_of": (
                ("ecg_hrv_hr_mean_bpm", "ecg_hr_bpm"),
                "ppg_rate_bpm",
                "pupil_dilation_velocity",
            ),
        },
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

# AnchorState v0.2 extension specs.
# These are appended to the v0.1 set when the required source columns exist.
ANCHOR_INDEX_SPECS_V2_EXTENSIONS: Dict[str, Dict[str, Sequence[str]]] = {
    "resp_anchor_index": {
        # Higher resp_regular_index and resp_depth_index → more anchored.
        # Higher resp_rate_bpm → less anchored (rapid breathing = arousal/anxiety).
        # resp_slowing_index = −rate_bpm/20, so it is ALREADY inversely coded:
        # high slowing_index = slow breathing = anchored → should be POSITIVE.
        # BUG FIXED 2026-07-09 (diary 184): was incorrectly placed in "negative",
        # which double-negated it and made the rate component reward fast breathing.
        "positive": ("resp_regular_index", "resp_depth_index", "resp_slowing_index"),
        "negative": (),
    },
    "ocular_stability_index": {
        # eog_eye_stability_index is kept as a standalone feature but not
        # used here: when most epochs are zero-clipped its MAD collapses to
        # zero, causing the robust-z fallback to produce gigantic values.
        # blink_rate and saccade_rate (both negative) carry the same signal
        # and have stable non-zero distributions.
        "positive": (),
        "negative": ("eog_blink_rate", "eog_heog_saccade_rate"),
    },
    "cardioresp_anchor_index": {
        "positive": ("cardioresp_coupling_index", "cardioresp_rpeak_resp_plv", "cardioresp_coherence"),
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
    cfg.setdefault("guard_policy_version", ANCHOR_GUARD_POLICY_VERSION)
    guards_raw = cfg.get("guards", {})
    guards: Dict[str, Any] = dict(guards_raw) if isinstance(guards_raw, Mapping) else {}
    guards.setdefault("abs_max", DEFAULT_ANCHOR_ABS_MAX)
    cfg["guards"] = guards
    # AnchorState v0.2 extension
    v2_raw = cfg.get("v2", {})
    v2_cfg: Dict[str, Any] = dict(v2_raw) if isinstance(v2_raw, Mapping) else {}
    v2_cfg.setdefault("enabled", False)
    cfg["v2"] = v2_cfg
    return cfg


def _resolve_candidate_names(candidates: Sequence[Any], names: Sequence[str]) -> list[str]:
    """Resolve candidate/fallback feature names against an available name list."""
    available = {str(name) for name in names}
    resolved: list[str] = []
    for candidate in candidates:
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
            for name in candidate:
                name_str = str(name)
                if name_str in available:
                    resolved.append(name_str)
                    break
            continue
        name = str(candidate)
        if name in available:
            resolved.append(name)
    return list(dict.fromkeys(resolved))


def _feature_provenance_valid_mask(
    feature_metadata: Mapping[str, Any] | None,
    names: Sequence[str],
) -> np.ndarray:
    """Return feature-level validity from strict robust-z export provenance."""
    valid = np.ones(len(names), dtype=bool)
    if not isinstance(feature_metadata, Mapping):
        return valid

    exported_valid = feature_metadata.get("robust_z_valid")
    if exported_valid is not None:
        arr = np.asarray(exported_valid).reshape(-1)
        if arr.size == len(names):
            valid &= np.asarray(arr, dtype=np.float64) > 0

    invalid_reason = feature_metadata.get("robust_z_invalid_reason")
    if invalid_reason is not None:
        arr = np.asarray(invalid_reason, dtype=object).reshape(-1)
        if arr.size == len(names):
            valid &= np.asarray(
                [str(reason or "").strip() == "" for reason in arr],
                dtype=bool,
            )
    return valid


def _feature_series(
    robust_z_values: np.ndarray,
    robust_z_names: Sequence[str],
    candidates: Sequence[Any],
    *,
    feature_provenance_valid: np.ndarray,
    abs_max: float,
) -> tuple[np.ndarray | None, list[str], np.ndarray]:
    """Return the mean robust-z series across available candidate columns."""
    n_rows = int(robust_z_values.shape[0]) if robust_z_values.ndim == 2 else 0
    empty_valid = np.zeros(n_rows, dtype=bool)
    if robust_z_values.size == 0 or not robust_z_names:
        return None, [], empty_valid
    name_to_idx = {str(name): idx for idx, name in enumerate(robust_z_names)}
    used = _resolve_candidate_names(candidates, robust_z_names)
    if not used:
        return None, [], empty_valid
    cols = [name_to_idx[name] for name in used]
    values = np.asarray(robust_z_values[:, cols], dtype=np.float32)
    finite_and_valid = (
        np.isfinite(values)
        & feature_provenance_valid[np.asarray(cols, dtype=int)][None, :]
        & (np.abs(values) <= float(abs_max))
    )
    counts = np.sum(finite_and_valid, axis=1)
    summed = np.sum(np.where(finite_and_valid, values, 0.0), axis=1)
    series = np.divide(
        summed,
        np.maximum(counts, 1),
        out=np.full(values.shape[0], np.nan, dtype=np.float32),
        where=counts > 0,
    )
    return series.astype(np.float32, copy=False), used, counts > 0


def _combine_signed_series(
    robust_z_values: np.ndarray,
    robust_z_names: Sequence[str],
    *,
    positive: Sequence[Any],
    negative: Sequence[Any],
    feature_provenance_valid: np.ndarray,
    abs_max: float,
) -> tuple[np.ndarray, dict[str, list[str]], np.ndarray]:
    """Combine positive and negative candidate series into one signed index."""
    pos_series, pos_used, pos_valid = _feature_series(
        robust_z_values,
        robust_z_names,
        positive,
        feature_provenance_valid=feature_provenance_valid,
        abs_max=abs_max,
    )
    neg_series, neg_used, neg_valid = _feature_series(
        robust_z_values,
        robust_z_names,
        negative,
        feature_provenance_valid=feature_provenance_valid,
        abs_max=abs_max,
    )
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
    return (
        base.astype(np.float32, copy=False),
        {"positive": pos_used, "negative": neg_used},
        np.asarray(pos_valid | neg_valid, dtype=bool),
    )


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


def _raw_candidate_support_mask(features_df: pd.DataFrame, candidate: Any) -> np.ndarray:
    """Return rows with finite raw support for one candidate/fallback group."""
    support = np.zeros(len(features_df), dtype=bool)
    candidate_names = (
        list(candidate)
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes))
        else [candidate]
    )
    for name in candidate_names:
        name_str = str(name)
        if name_str not in features_df.columns:
            continue
        values = pd.to_numeric(features_df[name_str], errors="coerce").to_numpy(
            dtype=np.float32,
            copy=False,
        )
        support |= np.isfinite(values)
    return support


def _index_eligibility_mask(
    features_df: pd.DataFrame,
    spec: Mapping[str, Any],
) -> np.ndarray:
    """Apply optional row-level raw-physiology eligibility rules for one index."""
    eligible = np.ones(len(features_df), dtype=bool)
    raw = spec.get("eligibility", {})
    eligibility = dict(raw) if isinstance(raw, Mapping) else {}

    require_any = eligibility.get("require_any_of", []) or []
    if require_any:
        support = np.zeros(len(features_df), dtype=bool)
        for candidate in require_any:
            support |= _raw_candidate_support_mask(features_df, candidate)
        eligible &= support

    require_all = eligibility.get("require_all_of", []) or []
    for candidate in require_all:
        eligible &= _raw_candidate_support_mask(features_df, candidate)

    return eligible


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


def _build_index_series(
    matrix: np.ndarray,
    names: list[str],
    index_specs: Mapping[str, Any],
    *,
    features_df: pd.DataFrame,
    feature_provenance_valid: np.ndarray,
    abs_max: float,
    emit_unresolved: bool,
) -> tuple[list[str], list[np.ndarray], Dict[str, Any], list[np.ndarray], list[np.ndarray]]:
    """Build guarded per-index arrays from a robust-z matrix."""
    anchor_names: list[str] = []
    anchor_cols: list[np.ndarray] = []
    source_manifest: Dict[str, Any] = {}
    eligible_masks: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    for index_name, spec in index_specs.items():
        if not isinstance(spec, Mapping):
            continue
        series, used, source_valid = _combine_signed_series(
            matrix,
            names,
            positive=_normalize_candidate_spec(spec.get("positive", []) or []),
            negative=_normalize_candidate_spec(spec.get("negative", []) or []),
            feature_provenance_valid=feature_provenance_valid,
            abs_max=abs_max,
        )
        if not emit_unresolved and not (used["positive"] or used["negative"]):
            continue
        eligible = _index_eligibility_mask(features_df, spec)
        valid = eligible & source_valid & np.isfinite(series)
        guarded_series = np.asarray(series, dtype=np.float32).copy()
        guarded_series[~valid] = np.nan
        anchor_names.append(str(index_name))
        anchor_cols.append(guarded_series)
        source_manifest[str(index_name)] = used
        eligible_masks.append(eligible)
        valid_masks.append(valid)
    return anchor_names, anchor_cols, source_manifest, eligible_masks, valid_masks


def _nanmean_valid_composite(
    cols: list[np.ndarray],
    valid_masks: list[np.ndarray],
) -> np.ndarray:
    """Compute a row-wise composite only over validated component values."""
    if not cols:
        return np.array([], dtype=np.float32)
    stacked = np.vstack([c.astype(np.float32) for c in cols])
    valid = np.vstack([mask.astype(bool) for mask in valid_masks]) & np.isfinite(stacked)
    counts = np.sum(valid, axis=0)
    summed = np.sum(np.where(valid, stacked, 0.0), axis=0)
    return np.divide(
        summed,
        np.maximum(counts, 1),
        out=np.full(stacked.shape[1], np.nan, dtype=np.float32),
        where=counts > 0,
    ).astype(np.float32, copy=False)


def resolve_anchor_validation_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Resolve the non-blocking-by-default anchor export validation policy."""
    cfg = _resolve_anchor_state_config(config)
    guards = cfg.get("guards", {}) if isinstance(cfg.get("guards"), Mapping) else {}
    raw = cfg.get("validation", {})
    validation = dict(raw) if isinstance(raw, Mapping) else {}
    abs_max = float(validation.get("abs_max", guards.get("abs_max", DEFAULT_ANCHOR_ABS_MAX)))
    max_over_iqr = float(validation.get("max_over_iqr", 1e3))
    if not np.isfinite(abs_max) or abs_max <= 0:
        raise ValueError("anchor_state validation abs_max must be finite and > 0")
    if not np.isfinite(max_over_iqr) or max_over_iqr <= 0:
        raise ValueError("anchor_state validation max_over_iqr must be finite and > 0")
    return {
        "enabled": bool(validation.get("enabled", True)),
        "blocking": bool(validation.get("blocking", False)),
        "abs_max": abs_max,
        "max_over_iqr": max_over_iqr,
        "guard_policy_version": str(cfg["guard_policy_version"]),
    }


def validate_anchor_state_exports(
    anchor_state: Mapping[str, Any] | None,
    *,
    policy: Mapping[str, Any],
) -> Dict[str, Any]:
    """Summarize finite support and scale warnings for an AnchorState export."""
    resolved_policy = dict(policy)
    enabled = bool(resolved_policy.get("enabled", True))
    blocking = bool(resolved_policy.get("blocking", False))
    abs_max = float(resolved_policy.get("abs_max", DEFAULT_ANCHOR_ABS_MAX))
    max_over_iqr_threshold = float(resolved_policy.get("max_over_iqr", 1e3))
    guard_policy_version = str(
        resolved_policy.get("guard_policy_version", ANCHOR_GUARD_POLICY_VERSION)
    )
    if not enabled:
        return {
            "enabled": False,
            "blocking": blocking,
            "guard_policy_version": guard_policy_version,
            "thresholds": {"abs_max": abs_max, "max_over_iqr": max_over_iqr_threshold},
            "status": "disabled",
            "components": {},
            "n_warnings": 0,
        }
    components: Dict[str, Any] = {}
    statuses: list[str] = []

    values_raw = anchor_state.get("values") if isinstance(anchor_state, Mapping) else None
    names_raw = anchor_state.get("names") if isinstance(anchor_state, Mapping) else None
    values = np.asarray(values_raw, dtype=np.float32) if values_raw is not None else np.empty((0, 0))
    names = [str(name) for name in (list(names_raw) if names_raw is not None else [])]
    if values.ndim != 2 or values.shape[1] != len(names):
        return {
            "enabled": enabled,
            "blocking": blocking,
            "guard_policy_version": guard_policy_version,
            "thresholds": {"abs_max": abs_max, "max_over_iqr": max_over_iqr_threshold},
            "status": "not_available",
            "components": {},
            "n_warnings": 0,
        }

    for idx, name in enumerate(names):
        series = np.asarray(values[:, idx], dtype=np.float64)
        finite = series[np.isfinite(series)]
        finite_count = int(finite.size)
        nan_count = int(series.size - finite_count)
        if finite_count:
            q25, q75 = np.percentile(finite, [25.0, 75.0])
            iqr = float(q75 - q25)
            max_abs = float(np.max(np.abs(finite)))
        else:
            iqr = float("nan")
            max_abs = float("nan")
        if finite_count == 0:
            max_over_iqr = float("nan")
        elif iqr > 0:
            max_over_iqr = float(max_abs / iqr)
        elif max_abs == 0:
            max_over_iqr = 0.0
        else:
            max_over_iqr = float("inf")

        invalid_count = int(np.sum(np.isfinite(series) & (np.abs(series) > abs_max)))
        warnings: list[str] = []
        status = "ok"
        if invalid_count:
            warnings.append(f"{invalid_count} finite row(s) exceed abs_max={abs_max:g}")
            status = "fail"
        if finite_count and iqr > 0 and max_over_iqr > max_over_iqr_threshold:
            warnings.append(
                f"max_over_iqr={max_over_iqr:.6g} exceeds threshold={max_over_iqr_threshold:g}"
            )
            if status == "ok":
                status = "warning"
        elif finite_count and iqr == 0 and max_abs > 0:
            warnings.append("nonzero constant series has zero IQR")
            if status == "ok":
                status = "warning"

        components[name] = {
            "finite_count": finite_count,
            "nan_count": nan_count,
            "finite_fraction": float(finite_count / series.size) if series.size else float("nan"),
            "iqr": iqr,
            "max_abs": max_abs,
            "max_over_iqr": max_over_iqr,
            "invalid_count": invalid_count,
            "warnings": warnings,
            "status": status,
        }
        statuses.append(status)

    if "fail" in statuses:
        status = "fail"
    elif "warning" in statuses:
        status = "warning"
    else:
        status = "ok"
    return {
        "enabled": enabled,
        "blocking": blocking,
        "guard_policy_version": guard_policy_version,
        "thresholds": {"abs_max": abs_max, "max_over_iqr": max_over_iqr_threshold},
        "status": status,
        "components": components,
        "n_warnings": int(sum(bool(item["warnings"]) for item in components.values())),
    }


def build_anchor_state_exports(
    *,
    features_df: pd.DataFrame,
    robust_z_values: np.ndarray | None,
    robust_z_names: Sequence[str] | None,
    time: np.ndarray,
    config: Mapping[str, Any] | None = None,
    feature_metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build ``anchor_state``, ``anchor_state_dot``, and ``anchor_quality`` exports.

    Emits AnchorState v0.1 columns by default.  When
    ``anchor_state.v2.enabled: true`` is set in config and the matching
    source features are present, also emits AnchorState v0.2 extension
    columns and the ``anchor_index_v02`` composite.

    The legacy ``anchor_index`` (v0.1 nanmean) is always preserved verbatim.
    """
    cfg = _resolve_anchor_state_config(config)
    if not bool(cfg.get("enabled", False)):
        return {}, {}, {}, {}
    if robust_z_values is None or robust_z_names is None:
        return {}, {}, {}, {}
    matrix = np.asarray(robust_z_values, dtype=np.float32)
    names = [str(v) for v in robust_z_names]
    if matrix.ndim != 2 or matrix.shape[0] != len(time) or matrix.shape[1] != len(names):
        return {}, {}, {}, {}
    guards = cfg.get("guards", {}) if isinstance(cfg.get("guards"), Mapping) else {}
    abs_max = float(guards.get("abs_max", DEFAULT_ANCHOR_ABS_MAX))
    if not np.isfinite(abs_max) or abs_max <= 0:
        raise ValueError("anchor_state.guards.abs_max must be finite and > 0")
    feature_provenance_valid = _feature_provenance_valid_mask(feature_metadata, names)

    # --- v0.1 component indices ---
    index_specs = cfg.get("index_specs", DEFAULT_ANCHOR_INDEX_SPECS)
    if not isinstance(index_specs, Mapping):
        index_specs = DEFAULT_ANCHOR_INDEX_SPECS

    (
        v1_names,
        v1_cols,
        source_manifest,
        v1_eligible,
        v1_valid,
    ) = _build_index_series(
        matrix,
        names,
        index_specs,
        features_df=features_df,
        feature_provenance_valid=feature_provenance_valid,
        abs_max=abs_max,
        emit_unresolved=True,
    )
    if not v1_cols:
        return {}, {}, {}, {}

    # v0.1 composite: anchor_index = nanmean(v0.1 components)
    anchor_index_v01 = _nanmean_valid_composite(v1_cols, v1_valid)
    anchor_index_v01_eligible = (
        np.any(np.vstack(v1_eligible), axis=0)
        if v1_eligible
        else np.zeros(matrix.shape[0], dtype=bool)
    )
    anchor_index_v01_valid = np.isfinite(anchor_index_v01)

    # --- v0.2 extension indices (optional) ---
    v2_cfg = cfg.get("v2", {}) if isinstance(cfg.get("v2"), Mapping) else {}
    v2_enabled = bool(v2_cfg.get("enabled", False))

    v2_names: list[str] = []
    v2_cols: list[np.ndarray] = []
    v2_eligible: list[np.ndarray] = []
    v2_valid: list[np.ndarray] = []

    if v2_enabled:
        (
            ext_names,
            ext_cols,
            ext_manifest,
            ext_eligible,
            ext_valid,
        ) = _build_index_series(
            matrix,
            names,
            ANCHOR_INDEX_SPECS_V2_EXTENSIONS,
            features_df=features_df,
            feature_provenance_valid=feature_provenance_valid,
            abs_max=abs_max,
            emit_unresolved=False,
        )
        v2_names = ext_names
        v2_cols = ext_cols
        v2_eligible = ext_eligible
        v2_valid = ext_valid
        source_manifest.update(ext_manifest)

    # --- Assemble the full output matrix ---
    # Order: v0.1 components, anchor_index (v0.1), v0.2 extensions, anchor_index_v02
    all_names: list[str] = list(v1_names) + ["anchor_index"]
    all_cols: list[np.ndarray] = list(v1_cols) + [anchor_index_v01]
    all_eligible: list[np.ndarray] = list(v1_eligible) + [anchor_index_v01_eligible]
    all_valid: list[np.ndarray] = list(v1_valid) + [anchor_index_v01_valid]

    if v2_names:
        all_names.extend(v2_names)
        all_cols.extend(v2_cols)
        all_eligible.extend(v2_eligible)
        all_valid.extend(v2_valid)
        # v0.2 composite: nanmean across ALL components (v0.1 + v0.2)
        anchor_index_v02 = _nanmean_valid_composite(
            list(v1_cols) + list(v2_cols),
            list(v1_valid) + list(v2_valid),
        )
        anchor_index_v02_eligible = np.any(
            np.vstack(list(v1_eligible) + list(v2_eligible)),
            axis=0,
        )
        anchor_index_v02_valid = np.isfinite(anchor_index_v02)
        all_names.append("anchor_index_v02")
        all_cols.append(anchor_index_v02)
        all_eligible.append(anchor_index_v02_eligible)
        all_valid.append(anchor_index_v02_valid)

    anchor_matrix = np.column_stack(all_cols).astype(np.float32, copy=False)

    dot_matrix = np.column_stack(
        [
            _finite_gradient(anchor_matrix[:, idx], np.asarray(time, dtype=np.float64))
            for idx in range(anchor_matrix.shape[1])
        ]
    ).astype(np.float32, copy=False)

    # --- Quality surface ---
    quality_candidates = {
        "ecg_quality": ("ecg_hrv_quality_score", "ecg_quality_score", "qc_ok_ecg_hrv", "qc_ok_ecg"),
        "ppg_quality": ("ppg_quality_score", "qc_ok_ppg"),
        "pupil_quality": ("pupil_quality_score", "qc_ok_pupil"),
        "resp_quality": ("resp_signal_quality", "qc_ok_resp"),
        "eog_quality": ("qc_ok_eog",),
        "cardioresp_quality": ("qc_ok_cardioresp",),
    }
    quality_names: list[str] = []
    quality_cols_list: list[np.ndarray] = []
    for quality_name, candidates in quality_candidates.items():
        series = _extract_quality_series(features_df, candidates)
        if series is None:
            continue
        quality_names.append(str(quality_name))
        quality_cols_list.append(series.astype(np.float32, copy=False))

    # n_component_cols = all columns except composite(s) at end
    n_component_cols = len(v1_names)
    support_fraction = (
        np.sum(np.isfinite(anchor_matrix[:, :n_component_cols]), axis=1)
        / max(n_component_cols, 1)
    ).astype(np.float32)
    quality_names.append("anchor_support_fraction")
    quality_cols_list.append(support_fraction)
    anchor_valid_fraction = (
        np.sum(np.vstack(v1_valid), axis=0) / max(n_component_cols, 1)
    ).astype(np.float32)
    quality_names.append("anchor_valid_fraction")
    quality_cols_list.append(anchor_valid_fraction)
    for anchor_name, eligible, valid in zip(all_names, all_eligible, all_valid):
        quality_names.append(f"{anchor_name}_eligible")
        quality_cols_list.append(np.asarray(eligible, dtype=np.float32))
        quality_names.append(f"{anchor_name}_valid")
        quality_cols_list.append(np.asarray(valid, dtype=np.float32))
    quality_matrix = np.column_stack(quality_cols_list).astype(np.float32, copy=False)

    contract_version = "mndm.anchor_state.v2" if v2_names else "mndm.anchor_state.v1"
    attrs = {
        "contract": contract_version,
        "source": "features_robust_z",
        "source_features_json": json.dumps(source_manifest, ensure_ascii=False, separators=(",", ":")),
        "guard_policy_version": str(cfg["guard_policy_version"]),
        "guard_abs_max": float(abs_max),
    }
    anchor_state = {"values": anchor_matrix, "names": all_names, "attrs": attrs}
    anchor_state_dot = {"values": dot_matrix, "names": list(all_names), "attrs": {"source": "anchor_state"}}
    anchor_quality = {
        "values": quality_matrix,
        "names": quality_names,
        "attrs": {
            "contract": "mndm.anchor_quality.v1",
            "quality_surface": "v2",
            "guard_policy_version": str(cfg["guard_policy_version"]),
        },
    }
    diagnostics = {
        "enabled": True,
        "contract": contract_version,
        "names": list(all_names),
        "quality_names": list(quality_names),
        "source_features": source_manifest,
        "available_feature_count": int(len(names)),
        "v2_enabled": v2_enabled,
        "v2_names": v2_names,
        "guard_policy_version": str(cfg["guard_policy_version"]),
        "guard_abs_max": float(abs_max),
    }
    return anchor_state, anchor_state_dot, anchor_quality, diagnostics
