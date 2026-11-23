"""HDF5-ready exports for the separate experimental MNPS 3.4 MEG surface.

All coordinates in this export are unscaled raw measurements. Anchors from
EEG or another MEG realization are rejected; a future MEG anchor must have its
own declared calibration fit and manifest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .meg_candidates import (
    MEG_V34_DEFINITION_VERSION,
    MEG_V34_SCHEMA,
    MEG_V34_VARIANTS,
    resolve_meg_v3_4_config,
)


_RAW_FEATURES = (
    "delta",
    "theta",
    "alpha",
    "beta",
    "gamma",
    "beta_alpha",
    "hjorth_mobility",
    "alpha_theta",
    "permutation_entropy",
    "hjorth_complexity",
    "sample_entropy",
    "alpha_centroid_hz",
    "alpha_envelope_cv",
)

_COORDINATE_INPUTS = {
    "delta_theta_level": ("delta_theta_level",),
    "alpha_level": ("alpha_level",),
    "beta_alpha_ratio": ("beta_alpha_ratio",),
    "gamma_level": ("gamma_level",),
    "hjorth_mobility": ("hjorth_mobility",),
    "alpha_theta_ratio": ("alpha_theta_ratio",),
    "permutation_entropy": ("permutation_entropy",),
    "hjorth_complexity": ("hjorth_complexity",),
    "sample_entropy": ("sample_entropy",),
    "alpha_centroid_hz": ("alpha_centroid_hz",),
    "alpha_envelope_cv": ("alpha_envelope_cv",),
}

_COORDINATE_UNITS = {
    "delta_theta_level": "log10_native_power",
    "alpha_level": "log10_native_power",
    "beta_alpha_ratio": "dimensionless_ratio",
    "gamma_level": "log10_native_power",
    "hjorth_mobility": "inherited_hjorth_units",
    "alpha_theta_ratio": "dimensionless_ratio",
    "permutation_entropy": "dimensionless_entropy",
    "hjorth_complexity": "inherited_hjorth_units",
    "sample_entropy": "dimensionless_entropy",
    "alpha_centroid_hz": "Hz",
    "alpha_envelope_cv": "dimensionless",
}


def _hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _decode_json_list(values: Sequence[Any]) -> list[str]:
    if len(values) == 0:
        return []
    value = values[0]
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return []
    try:
        parsed = json.loads(str(value))
    except Exception:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _float_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame:
        return np.full(len(frame), np.nan, dtype=np.float64)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)


def _string_column(frame: pd.DataFrame, name: str, default: str = "unknown") -> np.ndarray:
    if name not in frame:
        return np.full(len(frame), default, dtype=object)
    return frame[name].fillna(default).astype(str).to_numpy(dtype=object)


def _family_payload(frame: pd.DataFrame, family: str, variants: Sequence[str]) -> dict[str, Any]:
    prefix = f"mnps_v3_4_meg_{family}"
    coordinate_values: dict[str, np.ndarray] = {}
    coordinate_reasons: dict[str, np.ndarray] = {}
    for name in _COORDINATE_INPUTS:
        col = f"{prefix}_{name}"
        coordinate_values[name] = _float_column(frame, col)
        coordinate_reasons[name] = _string_column(frame, f"{col}_reason", "feature_not_computed")

    raw_names = [f"meg_{family}_{name}" for name in _RAW_FEATURES[:-3]] + [
        f"{prefix}_sample_entropy",
        f"{prefix}_alpha_centroid_hz",
        f"{prefix}_alpha_envelope_cv",
    ]
    raw_values = np.column_stack([_float_column(frame, name) for name in raw_names]) if raw_names else np.empty((len(frame), 0))
    raw_support = np.isfinite(raw_values).astype(np.uint8)

    variant_payloads: dict[str, Any] = {}
    for variant_id in variants:
        names = list(MEG_V34_VARIANTS[variant_id])
        values = np.column_stack([coordinate_values[name] for name in names])
        support = np.isfinite(values).astype(np.uint8)
        variant_payloads[variant_id] = {
            "purpose": {
                "existing_coordinate_set": "Corrected experimental baseline; native SampEn and no metric substitution.",
                "removal_only": "Seven inherited coordinates with e_s and e_m omitted.",
                "alpha_replacement": "Seven inherited coordinates plus the two MEG-specific alpha candidates.",
            }[variant_id],
            "coordinate_names": names,
            "coordinate_units": [_COORDINATE_UNITS[name] for name in names],
            "raw_values": values,
            "coordinate_support": support,
            "complete_window_mask": support.all(axis=1).astype(np.uint8),
            "failure_reasons": {name: coordinate_reasons[name] for name in names},
        }

    channel_labels_by_window = _string_column(frame, f"{prefix}_channel_labels_json", "[]")
    channel_labels = _decode_json_list(channel_labels_by_window)
    return {
        "sensor_family": family,
        "aggregation": "inherited_features_median_waveform; alpha_candidates_channelwise_then_median",
        "raw_feature_names": raw_names,
        "raw_features": {"values": raw_values, "support": raw_support},
        "variants": variant_payloads,
        "support": {
            "channel_labels_ordered": channel_labels,
            "channel_labels_by_window_json": channel_labels_by_window,
            "channel_count": pd.to_numeric(
                frame.get(f"{prefix}_channel_count", pd.Series(np.zeros(len(frame)))), errors="coerce"
            ).fillna(0).to_numpy(dtype=np.int32),
            "channel_set_hash": _string_column(frame, f"{prefix}_channel_set_hash"),
            "channel_order_hash": _string_column(frame, f"{prefix}_channel_order_hash"),
            "sensor_family_status": _string_column(frame, f"{prefix}_sensor_family_status"),
            "inherited_psd_method_id": _string_column(frame, f"{prefix}_inherited_psd_method_id"),
            "inherited_psd_fallback_reason": _string_column(
                frame, f"{prefix}_inherited_psd_fallback_reason"
            ),
            "sample_entropy_status": _string_column(frame, f"{prefix}_sample_entropy_status"),
            "sample_entropy_reason": _string_column(frame, f"{prefix}_sample_entropy_reason"),
            "permutation_entropy_status": _string_column(
                frame, f"{prefix}_permutation_entropy_status"
            ),
            "permutation_entropy_reason": _string_column(
                frame, f"{prefix}_permutation_entropy_reason"
            ),
            "alpha_centroid_hz_status": _string_column(frame, f"{prefix}_alpha_centroid_hz_status"),
            "alpha_centroid_hz_reason": _string_column(frame, f"{prefix}_alpha_centroid_hz_reason"),
            "alpha_centroid_hz_estimator_id": _string_column(
                frame, f"{prefix}_alpha_centroid_hz_estimator_id"
            ),
            "alpha_centroid_hz_channel_count": pd.to_numeric(
                frame.get(f"{prefix}_alpha_centroid_hz_channel_count", pd.Series(np.zeros(len(frame)))),
                errors="coerce",
            ).fillna(0).to_numpy(dtype=np.int32),
            "alpha_centroid_hz_channel_labels_by_window_json": _string_column(
                frame, f"{prefix}_alpha_centroid_hz_channel_labels_json", "[]"
            ),
            "alpha_centroid_hz_channel_set_hash": _string_column(
                frame, f"{prefix}_alpha_centroid_hz_channel_set_hash"
            ),
            "alpha_centroid_hz_channel_order_hash": _string_column(
                frame, f"{prefix}_alpha_centroid_hz_channel_order_hash"
            ),
            "alpha_envelope_cv_status": _string_column(frame, f"{prefix}_alpha_envelope_cv_status"),
            "alpha_envelope_cv_reason": _string_column(frame, f"{prefix}_alpha_envelope_cv_reason"),
            "alpha_envelope_cv_estimator_id": _string_column(
                frame, f"{prefix}_alpha_envelope_cv_estimator_id"
            ),
            "alpha_envelope_cv_channel_count": pd.to_numeric(
                frame.get(f"{prefix}_alpha_envelope_cv_channel_count", pd.Series(np.zeros(len(frame)))),
                errors="coerce",
            ).fillna(0).to_numpy(dtype=np.int32),
            "alpha_envelope_cv_channel_labels_by_window_json": _string_column(
                frame, f"{prefix}_alpha_envelope_cv_channel_labels_json", "[]"
            ),
            "alpha_envelope_cv_channel_set_hash": _string_column(
                frame, f"{prefix}_alpha_envelope_cv_channel_set_hash"
            ),
            "alpha_envelope_cv_channel_order_hash": _string_column(
                frame, f"{prefix}_alpha_envelope_cv_channel_order_hash"
            ),
        },
    }


def build_meg_measurement_export(
    features: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Assemble raw unscaled coordinate variants and complete support metadata."""
    cfg = resolve_meg_v3_4_config(config)
    if not cfg or not bool(cfg.get("enabled", False)):
        return None
    if not isinstance(features, pd.DataFrame) or features.empty:
        return None
    variants = [str(value) for value in cfg.get("variants", list(MEG_V34_VARIANTS))]
    unknown = sorted(set(variants) - set(MEG_V34_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown MEG 3.4 variants in export: {unknown}")
    scaling = cfg.get("scaling", {})
    if not isinstance(scaling, Mapping) or str(scaling.get("mode", "raw_only")) != "raw_only":
        raise ValueError("MEG 3.4 export currently supports raw_only scaling mode")
    if scaling.get("anchor_manifest") or scaling.get("anchor_manifest_by_condition"):
        raise ValueError("MEG 3.4 export refuses external anchors until a MEG calibration fit is implemented")

    families: dict[str, Any] = {}
    for family in ("mag", "grad"):
        prefix = f"mnps_v3_4_meg_{family}"
        if f"{prefix}_sensor_family_status" in features.columns:
            families[family] = _family_payload(features, family, variants)
    unknown_family = None
    if "mnps_v3_4_meg_sensor_family_status" in features:
        unknown_family = {
            "status": _string_column(features, "mnps_v3_4_meg_sensor_family_status"),
            "reason": _string_column(features, "mnps_v3_4_meg_sensor_family_reason"),
        }

    for variant_id in variants:
        dimensions = {len(family["variants"][variant_id]["coordinate_names"]) for family in families.values()}
        expected = {7} if variant_id == "removal_only" else {9}
        if dimensions and dimensions != expected:
            raise ValueError(f"MEG variant {variant_id} has unexpected dimensions: {dimensions}")

    start = pd.to_numeric(features.get("t_start", pd.Series(np.full(len(features), np.nan))), errors="coerce").to_numpy(dtype=np.float64)
    end = pd.to_numeric(features.get("t_end", pd.Series(np.full(len(features), np.nan))), errors="coerce").to_numpy(dtype=np.float64)
    epoch = pd.to_numeric(features.get("epoch_id", pd.Series(np.arange(len(features)))), errors="coerce").to_numpy(dtype=np.int64)
    implementation_paths = [
        Path(__file__).with_name("meg_candidates.py"),
        Path(__file__),
        Path(__file__).parents[2] / "features" / "meg.py",
    ]
    implementation_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in implementation_paths
        if path.is_file()
    }
    estimator_cfg = cfg.get("estimators", {})
    return {
        "schema": MEG_V34_SCHEMA,
        "definition_version": str(cfg.get("definition_version", MEG_V34_DEFINITION_VERSION)),
        "status": "experimental_recorded_raw_unscaled",
        "qualified": False,
        "scale_status": "not_scaled_raw_feature_units",
        "anchor_application_status": "not_applied",
        "variants": variants,
        "variant_definitions": {
            key: {"coordinate_names": list(MEG_V34_VARIANTS[key]), "coordinate_count": len(MEG_V34_VARIANTS[key])}
            for key in variants
        },
        "coordinate_units": dict(_COORDINATE_UNITS),
        "window_axis": {
            "epoch_id": epoch,
            "window_start_sec": start,
            "window_end_sec": end,
        },
        "sensor_families": families,
        "unknown_sensor_family": unknown_family,
        "provenance": {
            "sensor_family_policy": "explicit_mag_grad; unknown remains unknown",
            "historical_replay_preserved": True,
            "corrected_experimental_baseline": "existing_coordinate_set",
            "inherited_aggregation": "median waveform across sensors, then inherited estimator",
            "candidate_aggregation": "per-channel estimator, then median across sensors of one family",
            "candidate_estimators": dict(estimator_cfg) if isinstance(estimator_cfg, Mapping) else {},
            "selected_variants": variants,
            "configuration_hash": _hash(cfg),
            "implementation_hashes": implementation_hashes,
            "scale_policy": "raw_only; MEG-specific calibration manifest required before scaling",
        },
    }


__all__ = ["build_meg_measurement_export"]
