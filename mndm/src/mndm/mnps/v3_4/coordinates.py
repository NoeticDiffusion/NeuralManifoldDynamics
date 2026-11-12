"""Frozen-anchor coordinate assembly and HDF5 measurement payloads for MNPS 3.4."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .candidates import resolve_v3_4_config, runtime_implementation_hashes


COORDINATE_ORDER = (
    "delta_theta_level",
    "alpha_level",
    "beta_alpha_ratio",
    "gamma_level",
    "hjorth_mobility",
    "alpha_theta_ratio",
    "permutation_entropy",
    "alpha_centroid_hz",
    "alpha_envelope_cv",
)

COORDINATE_INPUTS: dict[str, tuple[tuple[str, str, float], ...]] = {
    "delta_theta_level": (("eeg_delta", "log10", 0.5), ("eeg_theta", "log10", 0.5)),
    "alpha_level": (("eeg_alpha", "log10", 1.0),),
    "beta_alpha_ratio": (("eeg_beta_alpha", "identity", 1.0),),
    "gamma_level": (("eeg_gamma", "log10", 1.0),),
    "hjorth_mobility": (("eeg_hjorth_mobility", "identity", 1.0),),
    "alpha_theta_ratio": (("eeg_alpha_theta", "identity", 1.0),),
    "permutation_entropy": (("eeg_permutation_entropy", "identity", 1.0),),
    "alpha_centroid_hz": (("mnps_v3_4_alpha_centroid_hz", "identity", 1.0),),
    "alpha_envelope_cv": (("mnps_v3_4_alpha_envelope_cv", "identity", 1.0),),
}

FEATURE_ORDER = tuple(dict.fromkeys(item[0] for inputs in COORDINATE_INPUTS.values() for item in inputs))
IDENTITY_COLUMNS = (
    "mnps_v3_4_source_surface_hash",
    "mnps_v3_4_channel_compatibility_set_hash",
    "mnps_v3_4_reference",
    "mnps_v3_4_reference_set_hash",
)


def _canonical_hash(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _truthy_series(series: pd.Series) -> np.ndarray:
    """Parse stored booleans without treating nonempty strings like ``False`` as true."""
    def parse(value: Any) -> bool:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return False
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(np.isfinite(value) and float(value) != 0.0)
        return str(value).strip().lower() in {"true", "1", "yes", "y"}

    return series.map(parse).to_numpy(dtype=bool)


def _verify_anchor_hash(manifest: Mapping[str, Any]) -> None:
    stored = str(manifest.get("anchor_hash", ""))
    unhashed = dict(manifest)
    unhashed.pop("anchor_hash", None)
    computed = _canonical_hash(unhashed)
    if not stored or stored != computed:
        raise ValueError("MNPS 3.4 anchor manifest hash is missing or does not match its contents")


def _participant_ids(frame: pd.DataFrame) -> np.ndarray:
    if "subject" in frame.columns:
        ids = frame["subject"].fillna("").astype(str).to_numpy(dtype=object)
        if all(str(value).strip() for value in ids):
            return ids
    if "subject_id" in frame.columns:
        ids = frame["subject_id"].fillna("").astype(str).to_numpy(dtype=object)
        if all(str(value).strip() for value in ids):
            return ids
    if "file" in frame.columns:
        ids = []
        for file_name in frame["file"].fillna("").astype(str):
            match = re.search(r"(?i)(sub-[^/\\_]+)", file_name)
            ids.append(match.group(1) if match else "")
        if all(ids):
            return np.asarray(ids, dtype=object)
    raise ValueError("MNPS 3.4 anchors require a participant identifier or BIDS subject in file paths")


def _normalize_condition(value: Any) -> str | None:
    compact = re.sub(r"[^a-z]", "", str(value).lower())
    if "eyesclosed" in compact:
        return "EyesClosed"
    if "eyesopen" in compact:
        return "EyesOpen"
    return None


def _recording_condition(frame: pd.DataFrame) -> str | None:
    """Return one unambiguous eyes-open/closed condition for a recording frame."""
    observed: set[str] = set()
    for column in ("task", "condition", "file"):
        if column not in frame:
            continue
        for value in frame[column].dropna().astype(str).unique().tolist():
            condition = _normalize_condition(value)
            if condition:
                observed.add(condition)
    return next(iter(observed)) if len(observed) == 1 else None


def _transform(values: np.ndarray, transform: str) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(x)
    if transform == "log10":
        valid = finite & (x > 0)
        out[valid] = np.log10(x[valid])
    elif transform == "identity":
        out[finite] = x[finite]
    else:
        raise ValueError(f"Unsupported MNPS 3.4 input transform: {transform}")
    return out


def _input_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    values = np.full((len(frame), len(FEATURE_ORDER)), np.nan, dtype=np.float64)
    index = {name: i for i, name in enumerate(FEATURE_ORDER)}
    for coordinate in COORDINATE_ORDER:
        for feature_name, transform_name, _ in COORDINATE_INPUTS[coordinate]:
            if feature_name not in frame:
                continue
            transformed = _transform(pd.to_numeric(frame[feature_name], errors="coerce").to_numpy(), transform_name)
            values[:, index[feature_name]] = transformed
            if feature_name == "eeg_permutation_entropy" and "eeg_entropy_degraded_mode" in frame:
                degraded = _truthy_series(frame["eeg_entropy_degraded_mode"])
                values[degraded, index[feature_name]] = np.nan
    return values, list(FEATURE_ORDER)


def _raw_feature_matrix(frame: pd.DataFrame, feature_names: Sequence[str]) -> np.ndarray:
    values = np.full((len(frame), len(feature_names)), np.nan, dtype=np.float64)
    for j, name in enumerate(feature_names):
        if name in frame:
            values[:, j] = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
    return values


def _feature_scaled_values(
    transformed_features: np.ndarray,
    feature_names: Sequence[str],
    anchors: Mapping[str, Any],
) -> np.ndarray:
    output = np.full_like(transformed_features, np.nan, dtype=np.float64)
    for j, name in enumerate(feature_names):
        anchor = anchors.get(name, {}) if isinstance(anchors, Mapping) else {}
        if not isinstance(anchor, Mapping) or anchor.get("status") != "computed":
            continue
        center = float(anchor["center"])
        scale = float(anchor["scale"])
        if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0:
            continue
        finite = np.isfinite(transformed_features[:, j])
        output[finite, j] = (transformed_features[finite, j] - center) / scale
    return output


def _assemble_coordinates(feature_values: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    index = {str(name): i for i, name in enumerate(feature_names)}
    coordinates = np.full((feature_values.shape[0], len(COORDINATE_ORDER)), np.nan, dtype=np.float64)
    for j, coordinate in enumerate(COORDINATE_ORDER):
        contributions: list[np.ndarray] = []
        supported = np.ones(feature_values.shape[0], dtype=bool)
        for feature_name, _, coefficient in COORDINATE_INPUTS[coordinate]:
            col = feature_values[:, index[feature_name]]
            supported &= np.isfinite(col)
            contributions.append(float(coefficient) * col)
        if contributions:
            total = np.sum(np.column_stack(contributions), axis=1)
            coordinates[supported, j] = total[supported]
    return coordinates


def _fit_robust_anchor(
    values: np.ndarray,
    participants: np.ndarray,
    *,
    minimum_participants: int,
) -> dict[str, Any]:
    """Fit robust location/scale over windows with equal total participant weight.

    Each participant contributes total weight one to the finite observations for
    this measurement. Thus a participant with more valid windows does not gain
    more influence, while within-participant window variation remains part of
    the reference distribution.
    """
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    ids = np.asarray(participants, dtype=object).reshape(-1)
    if len(x) != len(ids):
        raise ValueError("MNPS 3.4 anchor values and participant identifiers must have equal lengths")
    finite = np.isfinite(x)
    x = x[finite]
    ids = ids[finite]
    if not x.size:
        return {
            "center": None,
            "scale": None,
            "participant_count": 0,
            "window_count": 0,
            "status": "unsupported_no_finite_reference_values",
        }

    counts: dict[str, int] = {}
    for participant in ids:
        key = str(participant)
        counts[key] = counts.get(key, 0) + 1
    participant_count = len(counts)
    if participant_count < int(minimum_participants):
        return {
            "center": None,
            "scale": None,
            "participant_count": participant_count,
            "window_count": int(x.size),
            "status": "unsupported_insufficient_reference_participants",
            "minimum_participants": int(minimum_participants),
        }
    weights = np.asarray([1.0 / counts[str(participant)] for participant in ids], dtype=np.float64)
    weight_sum = float(weights.sum())
    normalized_weights = weights / weight_sum

    def weighted_quantile(sample: np.ndarray, q: float) -> float:
        order = np.argsort(sample, kind="mergesort")
        sorted_sample = sample[order]
        cumulative = np.cumsum(normalized_weights[order])
        index = min(int(np.searchsorted(cumulative, float(q), side="left")), len(sorted_sample) - 1)
        return float(sorted_sample[index])

    center = weighted_quantile(x, 0.5)
    abs_deviation = np.abs(x - center)
    mad = weighted_quantile(abs_deviation, 0.5)
    q1 = weighted_quantile(x, 0.25)
    q3 = weighted_quantile(x, 0.75)
    mean = float(np.sum(normalized_weights * x))
    centered = x - mean
    variance = float(np.sum(normalized_weights * centered**2))
    std = float(np.sqrt(max(variance, 0.0)))
    skewness = float(np.sum(normalized_weights * centered**3) / (std**3)) if std > 0 else None
    kurtosis_excess = float(np.sum(normalized_weights * centered**4) / (variance**2) - 3.0) if variance > 0 else None
    scale = 1.4826 * mad
    status = "computed" if np.isfinite(scale) and scale > 0 else "unsupported_degenerate_reference_scale"
    return {
        "center": center,
        "scale": scale if np.isfinite(scale) else None,
        "scale_definition": "1.4826 * equal_participant_weighted_median_absolute_deviation_of_finite_window_values",
        "window_weighting": "each participant total weight 1; equal weights within participant over finite windows",
        "quantile_definition": "inverse_cdf_left_continuous; first value whose cumulative normalized weight is >= q",
        "participant_count": int(participant_count),
        "window_count": int(x.size),
        "mean": mean,
        "std_population": std,
        "median": center,
        "mad": mad,
        "mad_sigma": 1.4826 * mad,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "minimum": float(np.min(x)),
        "maximum": float(np.max(x)),
        "skewness": skewness,
        "kurtosis_excess": kurtosis_excess,
        "status": status,
    }


def _require_single_surface(frame: pd.DataFrame) -> dict[str, str]:
    missing = [key for key in IDENTITY_COLUMNS if key not in frame.columns]
    if missing:
        raise ValueError(f"MNPS 3.4 feature table is missing surface identity fields: {missing}")
    surface: dict[str, str] = {}
    for key in IDENTITY_COLUMNS:
        values = frame[key].dropna().astype(str).unique().tolist()
        if len(values) != 1 or not values[0]:
            raise ValueError(f"MNPS 3.4 anchor fitting/application requires one nonempty {key}; got {values[:5]}")
        surface[key] = values[0]
    return surface


def fit_anchor_manifest(
    features: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    anchor_id: str,
    minimum_participants: int = 3,
    reference_condition: str | None = None,
) -> dict[str, Any]:
    """Fit participant-balanced feature and coordinate anchors from a declared cohort."""
    cfg = resolve_v3_4_config(config)
    if not cfg or not bool(cfg.get("enabled", False)):
        raise ValueError("MNPS 3.4 anchors can only be fitted from an explicitly enabled v3.4 config")
    if features.empty:
        raise ValueError("Cannot fit MNPS 3.4 anchors from an empty features table")
    participants = _participant_ids(features)
    participant_names = sorted(set(str(value) for value in participants))
    if len(participant_names) < int(minimum_participants):
        raise ValueError(
            f"MNPS 3.4 anchor fit needs at least {minimum_participants} participants; found {len(participant_names)}"
        )
    scaling_cfg = cfg.get("scaling", {}) if isinstance(cfg.get("scaling"), Mapping) else {}
    configured_anchor_id = scaling_cfg.get("anchor_id")
    configured_by_condition = scaling_cfg.get("anchor_manifests_by_condition", {})
    if reference_condition and isinstance(configured_by_condition, Mapping):
        selected_condition_entry = next(
            (
                entry for key, entry in configured_by_condition.items()
                if _normalize_condition(key) == _normalize_condition(reference_condition)
            ),
            None,
        )
        if isinstance(selected_condition_entry, Mapping) and selected_condition_entry.get("anchor_id"):
            configured_anchor_id = selected_condition_entry.get("anchor_id")
    if configured_anchor_id and str(configured_anchor_id) != str(anchor_id):
        raise ValueError(
            f"MNPS 3.4 anchor id {anchor_id!r} does not match configured anchor id {configured_anchor_id!r}"
        )
    surface = _require_single_surface(features)
    transformed, feature_names = _input_matrix(features)
    feature_anchors = {
        name: _fit_robust_anchor(
            transformed[:, j], participants, minimum_participants=int(minimum_participants)
        )
        for j, name in enumerate(feature_names)
    }
    feature_scaled = _feature_scaled_values(transformed, feature_names, feature_anchors)
    assembled = _assemble_coordinates(feature_scaled, feature_names)
    coordinate_anchors = {
        name: _fit_robust_anchor(
            assembled[:, j], participants, minimum_participants=int(minimum_participants)
        )
        for j, name in enumerate(COORDINATE_ORDER)
    }
    dataset_versions = cfg.get("measurement_surface", {}).get("dataset_overlay", {})
    reference = cfg.get("scaling", {}).get("anchor_scope", "frozen_reference_cohort_or_training_partition")
    manifest: dict[str, Any] = {
        "schema": "mndm.mnps_v3_4.anchor_manifest.v1",
        "anchor_id": str(anchor_id),
        "definition_version": str(cfg.get("definition_version", "3.4")),
        "status": "experimental_frozen_reference",
        "qualified": False,
        "surface": surface,
        "reference": {
            "scope": str(reference),
            "dataset_overlay": dataset_versions,
            "participants": participant_names,
            "participant_count": len(participant_names),
            "participant_weighting": "equal_participant_total_weight_per_measurement; equal_weights_over_that_participant's_finite_selected_windows",
            "fit_and_apply_overlap": True,
            "evaluation_status": "transductive_pilot_calibration_not_held_out_validation",
            "condition": str(reference_condition) if reference_condition else "caller_selected_rows",
            "conditions_used_for_anchor_fit": str(reference_condition) if reference_condition else "all_rows_supplied_to_fit",
            "session_policy": "caller_selected_rows; cohort protocol should select first-session pre-task recordings",
            "artifact_qc_status": "not_established_by_anchor_fit; preserve the per-window qc_ok_eeg value and interpret -1 as not assessed",
            "eligibility_rule": "finite measurement-specific estimator inputs and declared channel/reference surface; technical row filtering is recorded by the caller",
        },
        "feature_anchors": feature_anchors,
        "coordinate_anchors": coordinate_anchors,
        "feature_order": feature_names,
        "coordinate_order": list(COORDINATE_ORDER),
        "coordinate_weights": {name: 1.0 for name in COORDINATE_ORDER},
        "transforms": {
            name: [{"feature": feature, "transform": transform, "coefficient": coefficient}
                   for feature, transform, coefficient in inputs]
            for name, inputs in COORDINATE_INPUTS.items()
        },
        "clipping": "none",
        "nonpositive_log_input_policy": "invalid_no_floor",
        "degenerate_scale_policy": "coordinate_or_feature_invalid_no_epsilon_substitution",
    }
    manifest["anchor_hash"] = _canonical_hash(manifest)
    manifest["anchor_fit_source"] = {
        "features_table": None,
        "features_table_sha256": None,
    }
    # anchor_fit_source is finalized by the CLI when it knows the exact input file.
    manifest.pop("anchor_hash", None)
    manifest["anchor_hash"] = _canonical_hash(manifest)
    return manifest


def _read_channel_labels_by_window(frame: pd.DataFrame) -> list[list[str]]:
    if "mnps_v3_4_channel_labels_json" not in frame:
        return [[] for _ in range(len(frame))]
    labels_by_window: list[list[str]] = []
    for raw in frame["mnps_v3_4_channel_labels_json"].fillna("[]").astype(str):
        try:
            decoded = json.loads(raw)
        except Exception:
            decoded = []
        labels_by_window.append([str(value) for value in decoded] if isinstance(decoded, list) else [])
    return labels_by_window


def _json_rows(frame: pd.DataFrame, column: str, n_channels: int) -> np.ndarray:
    rows = []
    raw_values = frame[column].fillna("[]").astype(str).tolist() if column in frame else ["[]"] * len(frame)
    for raw in raw_values:
        try:
            decoded = json.loads(raw)
        except Exception:
            decoded = []
        values = np.full(n_channels, np.nan, dtype=np.float64)
        if isinstance(decoded, list):
            for i, value in enumerate(decoded[:n_channels]):
                if value is not None:
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        continue
                    values[i] = number if np.isfinite(number) else np.nan
        rows.append(values)
    return np.vstack(rows) if rows else np.empty((0, n_channels), dtype=np.float64)


def build_measurement_export(
    features: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    artifact_qc_applied: bool | None = None,
) -> dict[str, Any] | None:
    """Build the per-recording measurement export consumed by ``/measurements/v3_4``."""
    cfg = resolve_v3_4_config(config)
    if not cfg or not bool(cfg.get("enabled", False)):
        return None
    scaling = cfg.get("scaling", {}) if isinstance(cfg.get("scaling"), Mapping) else {}
    anchor_paths_by_condition = scaling.get("anchor_manifests_by_condition", {})
    has_condition_anchors = isinstance(anchor_paths_by_condition, Mapping) and bool(anchor_paths_by_condition)
    recording_condition = _recording_condition(features)
    selected_anchor_entry: Any = None
    if has_condition_anchors and recording_condition is not None:
        selected_anchor_entry = next(
            (
                entry for key, entry in anchor_paths_by_condition.items()
                if _normalize_condition(key) == recording_condition
            ),
            None,
        )
    if isinstance(selected_anchor_entry, Mapping):
        anchor_path_value = selected_anchor_entry.get("path", selected_anchor_entry.get("anchor_manifest"))
        selected_anchor_id = selected_anchor_entry.get("anchor_id")
    elif selected_anchor_entry is not None:
        anchor_path_value = selected_anchor_entry
        selected_anchor_id = None
    else:
        anchor_path_value = scaling.get("anchor_manifest") if not has_condition_anchors else None
        selected_anchor_id = None
    anchor_path: Path | None = None
    manifest: dict[str, Any] | None = None
    anchor_status = "computed"
    anchor_failure_reason: str | None = None
    anchor_manifest_sha256: str | None = None
    surface: dict[str, str] | None = None
    implementation_hash_values = (
        features["mnps_v3_4_implementation_hashes_json"].dropna().astype(str).unique().tolist()
        if "mnps_v3_4_implementation_hashes_json" in features
        else []
    )
    implementation_hash_manifests: list[dict[str, Any]] = []
    for value in implementation_hash_values:
        try:
            parsed = json.loads(value)
        except Exception:
            parsed = {"invalid_manifest": value}
        implementation_hash_manifests.append(parsed if isinstance(parsed, dict) else {"invalid_manifest": value})
    current_implementation_hashes = runtime_implementation_hashes()
    try:
        if has_condition_anchors and recording_condition is None:
            raise ValueError("condition_specific_anchor_requires_one_unambiguous_eyes_condition")
        if has_condition_anchors and selected_anchor_entry is None:
            raise ValueError(f"no_condition_specific_anchor_configured:{recording_condition}")
        if len(implementation_hash_manifests) != 1 or implementation_hash_manifests[0] != current_implementation_hashes:
            raise ValueError("feature_to_summary_runtime_implementation_hash_mismatch")
        if not anchor_path_value:
            raise ValueError("scaling.anchor_manifest_not_configured")
        anchor_path = Path(str(anchor_path_value))
        if not anchor_path.is_absolute():
            anchor_path = Path.cwd() / anchor_path
        if not anchor_path.is_file():
            raise ValueError("anchor_manifest_not_found")
        anchor_bytes_for_check = anchor_path.read_bytes()
        anchor_manifest_sha256 = hashlib.sha256(anchor_bytes_for_check).hexdigest()
        loaded = json.loads(anchor_bytes_for_check.decode("utf-8"))
        if not isinstance(loaded, dict) or loaded.get("schema") != "mndm.mnps_v3_4.anchor_manifest.v1":
            raise ValueError("anchor_manifest_schema_mismatch")
        _verify_anchor_hash(loaded)
        expected_anchor_id = selected_anchor_id or scaling.get("anchor_id")
        if expected_anchor_id and str(loaded.get("anchor_id")) != str(expected_anchor_id):
            raise ValueError("anchor_id_mismatch")
        manifest_condition = (loaded.get("reference", {}) or {}).get("condition")
        if has_condition_anchors and _normalize_condition(manifest_condition) != recording_condition:
            raise ValueError("anchor_condition_mismatch")
        try:
            surface = _require_single_surface(features)
        except ValueError as exc:
            raise ValueError(f"mixed_measurement_surface:{exc}") from exc
        if dict(loaded.get("surface", {})) != surface:
            raise ValueError("anchor_surface_mismatch")
        manifest = loaded
    except Exception as exc:
        anchor_status = "unsupported_anchor_application"
        anchor_failure_reason = str(exc)
    transformed, feature_names = _input_matrix(features)
    raw_features = _raw_feature_matrix(features, feature_names)
    feature_scaled = (
        _feature_scaled_values(transformed, feature_names, manifest.get("feature_anchors", {}))
        if manifest is not None
        else np.full_like(transformed, np.nan, dtype=np.float64)
    )
    coordinate_pre_reference = _assemble_coordinates(feature_scaled, feature_names)
    coordinate_values = np.full_like(coordinate_pre_reference, np.nan, dtype=np.float64)
    coordinate_anchors = manifest.get("coordinate_anchors", {}) if manifest is not None else {}
    for j, name in enumerate(COORDINATE_ORDER):
        anchor = coordinate_anchors.get(name, {}) if isinstance(coordinate_anchors, Mapping) else {}
        if not isinstance(anchor, Mapping) or anchor.get("status") != "computed":
            continue
        center = float(anchor["center"])
        scale = float(anchor["scale"])
        if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0:
            continue
        finite = np.isfinite(coordinate_pre_reference[:, j])
        coordinate_values[finite, j] = (coordinate_pre_reference[finite, j] - center) / scale

    raw_coordinate_values = np.full((len(features), len(COORDINATE_ORDER)), np.nan, dtype=np.float64)
    raw_coordinate_support = np.zeros_like(raw_coordinate_values, dtype=bool)
    raw_index = {name: j for j, name in enumerate(feature_names)}
    for j, coordinate in enumerate(COORDINATE_ORDER):
        pieces = []
        valid = np.ones(len(features), dtype=bool)
        for feature_name, transform_name, coefficient in COORDINATE_INPUTS[coordinate]:
            if feature_name not in raw_index:
                valid[:] = False
                break
            if feature_name not in features:
                valid[:] = False
                break
            raw_values = pd.to_numeric(features[feature_name], errors="coerce").to_numpy(dtype=np.float64)
            current = _transform(raw_values, transform_name)
            if feature_name == "eeg_permutation_entropy" and "eeg_entropy_degraded_mode" in features:
                degraded = _truthy_series(features["eeg_entropy_degraded_mode"])
                current[degraded] = np.nan
            valid &= np.isfinite(current)
            pieces.append(float(coefficient) * current)
        if pieces:
            total = np.sum(np.column_stack(pieces), axis=1)
            raw_coordinate_values[valid, j] = total[valid]
        raw_coordinate_support[:, j] = valid

    channel_labels_by_window = _read_channel_labels_by_window(features)
    label_variants = {tuple(labels) for labels in channel_labels_by_window}
    channel_labels = list(next(iter(label_variants))) if len(label_variants) == 1 else []
    n_channels = max((len(labels) for labels in channel_labels_by_window), default=0)
    per_channel = {
        "alpha_centroid_hz": _json_rows(features, "mnps_v3_4_alpha_centroid_hz_per_channel_json", n_channels),
        "alpha_envelope_cv": _json_rows(features, "mnps_v3_4_alpha_envelope_cv_per_channel_json", n_channels),
        "spatial_logvariance_log10_variance": _json_rows(features, "mnps_v3_4_spatial_logvariance_per_channel_json", n_channels),
    }
    def _column(name: str, default: Any = "unknown") -> np.ndarray:
        if name not in features:
            return np.full(len(features), default, dtype=object)
        return features[name].fillna(default).to_numpy(dtype=object)

    qc_columns = [column for column in features.columns if str(column).startswith("qc_ok_")]
    qc = {str(column): pd.to_numeric(features[column], errors="coerce").fillna(-1).to_numpy(dtype=np.int8)
          for column in qc_columns}
    if "qc_ok_eeg" in qc and artifact_qc_applied is not True:
        qc["qc_ok_eeg"] = np.full(len(features), -1, dtype=np.int8)
    entropy_degraded = (
        _truthy_series(features["eeg_entropy_degraded_mode"])
        if "eeg_entropy_degraded_mode" in features
        else np.zeros(len(features), dtype=np.bool_)
    )
    validity = np.isfinite(coordinate_values)
    complete = validity.all(axis=1)
    start = pd.to_numeric(features.get("t_start", pd.Series(np.full(len(features), np.nan))), errors="coerce").to_numpy(dtype=np.float64)
    end = pd.to_numeric(features.get("t_end", pd.Series(np.full(len(features), np.nan))), errors="coerce").to_numpy(dtype=np.float64)
    return {
        "schema": "mndm.mnps_v3_4.experimental_flat_9d.v1",
        "definition_version": str(cfg.get("definition_version", "3.4")),
        "status": "experimental_recorded",
        "qualified": False,
        "coordinate_order": list(COORDINATE_ORDER),
        "coordinate_names": list(COORDINATE_ORDER),
        "raw_features": {
            "values": raw_features,
            "names": feature_names,
            "support": np.isfinite(raw_features).astype(np.uint8),
        },
        "raw_coordinates": {"values": raw_coordinate_values, "support": raw_coordinate_support},
        "feature_scaled_values": {"values": feature_scaled, "names": feature_names},
        "coordinate_pre_reference_values": coordinate_pre_reference,
        "coordinates": {"values": coordinate_values, "names": list(COORDINATE_ORDER)},
        "coordinate_validity_mask": validity.astype(np.uint8),
        "complete_9d_mask": complete.astype(np.uint8),
        "auxiliary": {
            "spatial_logvariance_dispersion": pd.to_numeric(
                features.get("mnps_v3_4_spatial_logvariance_dispersion", pd.Series(np.full(len(features), np.nan))),
                errors="coerce",
            ).to_numpy(dtype=np.float64),
            "spatial_logvariance_status": _column("mnps_v3_4_spatial_logvariance_status"),
        },
        "per_channel_candidates": {
            "channel_labels": channel_labels,
            "channel_labels_by_window_json": np.asarray(
                [json.dumps(labels, ensure_ascii=False, separators=(",", ":")) for labels in channel_labels_by_window],
                dtype=object,
            ),
            **per_channel,
        },
        "window_start_sec": start,
        "window_end_sec": end,
        "support": {
            "source_surface_hash": _column("mnps_v3_4_source_surface_hash"),
            "source_file_sha256": _column("mnps_v3_4_source_file_sha256", default=""),
            "source_file_sha256_component": _column("mnps_v3_4_source_file_sha256_component", default="unknown_source_component"),
            "source_component_hashes_json": _column("mnps_v3_4_source_component_hashes_json", default="{}"),
            "source_identity_basis": _column("mnps_v3_4_source_identity_basis", default="unknown"),
            "channel_set_hash": _column("mnps_v3_4_channel_set_hash"),
            "channel_compatibility_set_hash": _column("mnps_v3_4_channel_compatibility_set_hash"),
            "channel_order_hash": _column("mnps_v3_4_channel_order_hash"),
            "channel_count": pd.to_numeric(
                features.get("mnps_v3_4_channel_count", pd.Series(np.zeros(len(features)))),
                errors="coerce",
            ).fillna(0).to_numpy(dtype=np.int32),
            "channel_labels": channel_labels,
            "reference": _column("mnps_v3_4_reference"),
            "reference_set_hash": _column("mnps_v3_4_reference_set_hash"),
            "reference_status": _column("mnps_v3_4_reference_status"),
            "implementation_hashes_by_window_json": _column("mnps_v3_4_implementation_hashes_json", default="{}"),
            "alpha_centroid_status": _column("mnps_v3_4_alpha_centroid_status"),
            "alpha_envelope_status": _column("mnps_v3_4_alpha_envelope_status"),
            "alpha_centroid_channel_count": pd.to_numeric(
                features.get("mnps_v3_4_alpha_centroid_channel_count", pd.Series(np.zeros(len(features)))),
                errors="coerce",
            ).fillna(0).to_numpy(dtype=np.int32),
            "alpha_envelope_channel_count": pd.to_numeric(
                features.get("mnps_v3_4_alpha_envelope_channel_count", pd.Series(np.zeros(len(features)))),
                errors="coerce",
            ).fillna(0).to_numpy(dtype=np.int32),
            "eeg_entropy_degraded_mode": entropy_degraded.astype(np.uint8),
            "qc": qc,
        },
        "provenance": {
            "anchor_id": manifest.get("anchor_id") if manifest is not None else None,
            "anchor_condition": recording_condition,
            "anchor_hash": manifest.get("anchor_hash") if manifest is not None else None,
            "anchor_manifest_path": str(anchor_path) if anchor_path else None,
            "anchor_manifest_sha256": anchor_manifest_sha256,
            "anchor_application_status": anchor_status,
            "anchor_application_failure_reason": anchor_failure_reason,
            "anchor_scope": (manifest.get("reference", {}) or {}).get("scope") if manifest is not None else None,
            "anchor_reference_participants": (manifest.get("reference", {}) or {}).get("participants", []) if manifest is not None else [],
            "fit_and_apply_overlap": (manifest.get("reference", {}) or {}).get("fit_and_apply_overlap") if manifest is not None else None,
            "evaluation_status": (manifest.get("reference", {}) or {}).get("evaluation_status") if manifest is not None else "not_scaled_no_compatible_anchor",
            "implementation_hashes_recorded_by_features": implementation_hash_manifests,
            "implementation_hashes_at_summarize": current_implementation_hashes,
            "legacy_sign_correspondence_before_coordinate_scaling": {
                "delta_theta_level": "negative v2 m_a under matching source and unclipped feature transforms",
                "alpha_level": "negative v2 m_e under matching source and unclipped feature transforms",
            },
            "clipping": "none",
            "source_file_names": sorted(set(Path(str(value)).name for value in features.get("file", pd.Series([], dtype=str)).dropna())),
            "source_run_manifest_ref": "../run_manifest.json",
        },
        "coordinate_weights": {name: 1.0 for name in COORDINATE_ORDER},
        "weighting_status": "neutral_design_default_pending_validation",
    }


__all__ = ["COORDINATE_ORDER", "COORDINATE_INPUTS", "FEATURE_ORDER", "build_measurement_export", "fit_anchor_manifest"]
