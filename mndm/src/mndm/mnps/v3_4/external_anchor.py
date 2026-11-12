"""Selectable, immutable MNPS 3.4 reference-anchor application.

This module deliberately keeps preset resolution and compatibility policy out of
the feature-estimator modules whose hashes identify extracted realizations.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import coordinates
from .candidates import resolve_v3_4_config


PRESET_UNITS = {
    "eeg_delta": "microvolt^2",
    "eeg_theta": "microvolt^2",
    "eeg_alpha": "microvolt^2",
    "eeg_beta_alpha": "dimensionless_ratio",
    "eeg_gamma": "microvolt^2",
    "eeg_hjorth_mobility": "dimensionless_per_sample",
    "eeg_alpha_theta": "dimensionless_ratio",
    "eeg_permutation_entropy": "dimensionless",
    "mnps_v3_4_alpha_centroid_hz": "Hz",
    "mnps_v3_4_alpha_envelope_cv": "dimensionless",
}

_REPO_ROOT = Path(__file__).resolve().parents[5]
_PRESET_DIR = _REPO_ROOT / "mndm" / "config" / "v.3.4" / "anchors"
PRESETS: dict[str, dict[str, str]] = {
    "ds005385_ec_v1": {
        "path": "mnps_v3_4_ds005385_ses1_pretask_EyesClosed_20261111_v1.json",
        "anchor_id": "ds005385_ses1_pretask_EyesClosed_20261111_v1",
        "sha256": "79da8c208a6c9240e84cefee7abc9a26c3121d59c1c86bc3324d33172193d1ab",
        "reference_condition": "EyesClosed",
        "reference_channel_order_hashes": "2bb38cc8424949a8d2ab47ae40a3ebac6e9af28fe188b27e630667b7c41fe9ae,4e0ce192491c7f5c8d79f5e87ecf0d09011977a1c7a9d597e8475cf2a059fb92",
    },
    "ds005385_eo_v1": {
        "path": "mnps_v3_4_ds005385_ses1_pretask_EyesOpen_20261111_v1.json",
        "anchor_id": "ds005385_ses1_pretask_EyesOpen_20261111_v1",
        "sha256": "f3bb94310778e91b046fbf77f0540ce860af6a65d4607e1e8593c01f800705b1",
        "reference_condition": "EyesOpen",
        "reference_channel_order_hashes": "2bb38cc8424949a8d2ab47ae40a3ebac6e9af28fe188b27e630667b7c41fe9ae,4e0ce192491c7f5c8d79f5e87ecf0d09011977a1c7a9d597e8475cf2a059fb92",
    },
}


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalized_condition(value: Any) -> str | None:
    compact = re.sub(r"[^a-z]", "", str(value).lower())
    if "eyesclosed" in compact:
        return "EyesClosed"
    if "eyesopen" in compact:
        return "EyesOpen"
    return None


def _scaling_config(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    v34 = resolve_v3_4_config(config)
    scaling = v34.get("scaling", {}) if isinstance(v34, Mapping) else {}
    return v34, scaling if isinstance(scaling, Mapping) else {}


def resolve_anchor_selection(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Resolve explicit v3.4 preset/custom selection; ``None`` keeps legacy behavior."""
    _, scaling = _scaling_config(config)
    selection = scaling.get("external_anchor", {})
    if not isinstance(selection, Mapping):
        raise ValueError("mnps_v3_4.scaling.external_anchor must be a mapping")
    preset = str(selection.get("preset") or "").strip() or None
    custom_path = str(selection.get("path") or "").strip() or None
    mode = str(selection.get("compatibility", "strict")).strip().lower() or "strict"
    reason = str(selection.get("transfer_reason") or "").strip() or None
    if preset and custom_path:
        raise ValueError("Select one MNPS 3.4 anchor source: preset or custom JSON path, not both")
    if mode not in {"strict", "exploratory_transfer"}:
        raise ValueError("MNPS 3.4 anchor compatibility must be 'strict' or 'exploratory_transfer'")
    if not preset and not custom_path:
        if mode != "strict" or reason:
            raise ValueError("MNPS 3.4 compatibility options require an explicit preset or custom anchor JSON")
        return None
    if preset and preset not in PRESETS:
        raise ValueError(f"Unknown MNPS 3.4 anchor preset {preset!r}; choices: {sorted(PRESETS)}")
    if mode == "exploratory_transfer" and not reason:
        raise ValueError("Exploratory MNPS 3.4 anchor transfer requires a nonempty transfer_reason")
    if mode == "strict" and reason:
        raise ValueError("transfer_reason is only valid with compatibility='exploratory_transfer'")
    return {
        "preset": preset,
        "path": custom_path,
        "compatibility": mode,
        "transfer_reason": reason,
        "reference_feature_units": selection.get("reference_feature_units"),
        "expected_anchor_id": selection.get("anchor_id"),
        "expected_sha256": selection.get("sha256"),
    }


def _resolve_anchor_path(selection: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[Path, str]:
    preset = selection.get("preset")
    if preset:
        spec = PRESETS[str(preset)]
        path = _PRESET_DIR / spec["path"]
        return path, f"preset:{preset}"
    raw_path = str(selection.get("path") or "").strip()
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        base = config.get("config_dir")
        if base:
            path = Path(str(base)).expanduser() / path
        else:
            path = Path.cwd() / path
    return path.resolve(), raw_path


def _resolve_legacy_anchor_paths(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve old config paths relative to their YAML file while preserving defaults."""
    result = copy.deepcopy(dict(config))
    v34 = resolve_v3_4_config(result)
    scaling = v34.get("scaling", {}) if isinstance(v34, Mapping) else {}
    if not isinstance(scaling, dict):
        return result
    config_dir = result.get("config_dir")

    def resolve(raw: Any) -> Any:
        if not raw:
            return raw
        path = Path(str(raw)).expanduser()
        if path.is_absolute():
            return str(path)
        candidates = []
        if config_dir:
            candidates.append(Path(str(config_dir)) / path)
        candidates.append(Path.cwd() / path)
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate.resolve())
        return str(candidates[0].resolve()) if candidates else str(path.resolve())

    if scaling.get("anchor_manifest"):
        scaling["anchor_manifest"] = resolve(scaling["anchor_manifest"])
    condition_entries = scaling.get("anchor_manifests_by_condition", {})
    if isinstance(condition_entries, dict):
        for name, entry in list(condition_entries.items()):
            if isinstance(entry, dict):
                key = "path" if "path" in entry else "anchor_manifest"
                if entry.get(key):
                    entry[key] = resolve(entry[key])
            elif entry:
                # Old scalar entries are file paths.
                condition_entries[name] = resolve(entry)
    return result


def _build_unanchored_payload(
    features: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    artifact_qc_applied: bool | None,
) -> dict[str, Any] | None:
    """Build raw v3.4 measurements without applying a legacy or external anchor."""
    from .coordinates import build_measurement_export

    base_config = copy.deepcopy(dict(config))
    base_v34 = resolve_v3_4_config(base_config)
    if not base_v34:
        return None
    base_scaling = dict(base_v34.get("scaling", {}) or {})
    base_scaling["anchor_manifest"] = None
    base_scaling["anchor_manifests_by_condition"] = {}
    base_v34["scaling"] = base_scaling
    payload = build_measurement_export(
        features, base_config, artifact_qc_applied=artifact_qc_applied
    )
    return payload if isinstance(payload, dict) else None


def _unresolved_selection(config: Mapping[str, Any]) -> dict[str, Any]:
    try:
        selection = resolve_anchor_selection(config)
        if selection is not None:
            resolved = dict(selection)
        else:
            resolved = {}
    except ValueError:
        _, scaling = _scaling_config(config)
        raw = scaling.get("external_anchor", {})
        resolved = dict(raw) if isinstance(raw, Mapping) else {"selection_value": str(raw)}
    if resolved.get("preset") or resolved.get("path"):
        try:
            path, display_path = _resolve_anchor_path(resolved, config)
            resolved["_runtime_failure_path"] = display_path
            resolved["_runtime_failure_resolved_path"] = str(path)
            if path.is_file():
                resolved["_runtime_failure_file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        except (OSError, ValueError, KeyError):
            pass
    return resolved


def load_selected_anchor(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Read and verify the selected anchor bytes and their embedded contract."""
    selection = resolve_anchor_selection(config)
    if selection is None:
        return None
    path, display_path = _resolve_anchor_path(selection, config)
    if not path.is_file():
        raise FileNotFoundError(
            f"MNPS 3.4 anchor file not found: {path}. Check the preset assets or custom JSON path."
        )
    raw = path.read_bytes()
    file_sha256 = hashlib.sha256(raw).hexdigest()
    preset = selection.get("preset")
    preset_spec = PRESETS.get(str(preset)) if preset else None
    if preset_spec and selection.get("expected_sha256") and str(selection["expected_sha256"]) != preset_spec["sha256"]:
        raise ValueError("A registered MNPS 3.4 preset SHA-256 cannot be overridden by config")
    expected_sha = preset_spec["sha256"] if preset_spec else selection.get("expected_sha256")
    if expected_sha and file_sha256 != str(expected_sha):
        raise ValueError(
            f"MNPS 3.4 anchor SHA-256 mismatch for {display_path}: "
            f"expected {expected_sha}, found {file_sha256}"
        )
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"MNPS 3.4 anchor JSON could not be decoded: {display_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"MNPS 3.4 anchor manifest must be a JSON object: {display_path}")
    if manifest.get("schema") != "mndm.mnps_v3_4.anchor_manifest.v1":
        raise ValueError(f"Unsupported MNPS 3.4 anchor schema: {manifest.get('schema')!r}")
    if manifest.get("status") != "experimental_frozen_reference" or bool(manifest.get("qualified", False)):
        raise ValueError("MNPS 3.4 external anchors must retain experimental_frozen_reference and qualified=false")
    unhashed = dict(manifest)
    embedded_hash = unhashed.pop("anchor_hash", None)
    if not embedded_hash or str(embedded_hash) != _canonical_hash(unhashed):
        raise ValueError("MNPS 3.4 anchor embedded anchor_hash is corrupt or does not match its contents")
    if preset_spec and selection.get("expected_anchor_id") and str(selection["expected_anchor_id"]) != preset_spec["anchor_id"]:
        raise ValueError("A registered MNPS 3.4 preset anchor ID cannot be overridden by config")
    expected_id = preset_spec["anchor_id"] if preset_spec else selection.get("expected_anchor_id")
    if expected_id and str(manifest.get("anchor_id")) != str(expected_id):
        raise ValueError(
            f"MNPS 3.4 anchor ID mismatch: expected {expected_id!r}, found {manifest.get('anchor_id')!r}"
        )
    manifest["_runtime_file_sha256"] = file_sha256
    manifest["_runtime_display_path"] = display_path
    manifest["_runtime_resolved_path"] = str(path)
    manifest["_runtime_reference_feature_units"] = (
        selection.get("reference_feature_units")
        or (PRESET_UNITS if preset_spec else manifest.get("feature_units"))
    )
    manifest["_runtime_reference_channel_order_hashes"] = (
        [value for value in str(preset_spec.get("reference_channel_order_hashes", "")).split(",") if value]
        if preset_spec else manifest.get("reference_channel_order_hashes")
    )
    return manifest, selection


def _finite_positive_anchor(anchor: Any, *, label: str) -> tuple[float, float]:
    if not isinstance(anchor, Mapping) or anchor.get("status") != "computed":
        raise ValueError(f"MNPS 3.4 anchor is missing a computed parameter for {label}")
    try:
        center = float(anchor["center"])
        scale = float(anchor["scale"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"MNPS 3.4 anchor center/scale is missing or invalid for {label}") from exc
    if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"MNPS 3.4 anchor center/scale must be finite with scale > 0 for {label}")
    return center, scale


def _validate_anchor_contract(manifest: Mapping[str, Any], v34: Mapping[str, Any]) -> None:
    if str(manifest.get("definition_version")) != str(v34.get("definition_version", "3.4")):
        raise ValueError(
            f"MNPS 3.4 anchor definition_version {manifest.get('definition_version')!r} "
            f"does not match target {v34.get('definition_version', '3.4')!r}"
        )
    if list(manifest.get("feature_order", []) or []) != list(coordinates.FEATURE_ORDER):
        raise ValueError("MNPS 3.4 anchor feature_order is altered or unsupported")
    if list(manifest.get("coordinate_order", []) or []) != list(coordinates.COORDINATE_ORDER):
        raise ValueError("MNPS 3.4 anchor coordinate_order is altered or unsupported")
    expected_transforms = {
        name: [
            {"feature": feature, "transform": transform, "coefficient": coefficient}
            for feature, transform, coefficient in inputs
        ]
        for name, inputs in coordinates.COORDINATE_INPUTS.items()
    }
    if manifest.get("transforms") != expected_transforms:
        raise ValueError("MNPS 3.4 anchor transformations or coordinate coefficients are altered")
    expected_weights = {name: 1.0 for name in coordinates.COORDINATE_ORDER}
    if manifest.get("coordinate_weights") != expected_weights:
        raise ValueError("MNPS 3.4 anchor coordinate weights are altered or unsupported")
    feature_anchors = manifest.get("feature_anchors")
    coordinate_anchors = manifest.get("coordinate_anchors")
    if not isinstance(feature_anchors, Mapping) or not isinstance(coordinate_anchors, Mapping):
        raise ValueError("MNPS 3.4 anchor feature_anchors and coordinate_anchors must be objects")
    for name in coordinates.FEATURE_ORDER:
        _finite_positive_anchor(feature_anchors.get(name), label=f"feature {name}")
    for name in coordinates.COORDINATE_ORDER:
        _finite_positive_anchor(coordinate_anchors.get(name), label=f"coordinate {name}")


def _one_identity_value(frame: pd.DataFrame, column: str) -> tuple[str | None, str | None]:
    if column not in frame:
        return None, "missing_column"
    values = [str(v) for v in frame[column].dropna().astype(str).unique().tolist() if str(v).strip()]
    if len(values) > 1:
        raise ValueError(f"MNPS 3.4 application requires one target realization per recording; {column} has {values[:5]}")
    if not values:
        return None, "missing_value"
    return values[0], None


def _target_identity(frame: pd.DataFrame, config: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    identity: dict[str, Any] = {}
    missing: list[str] = []
    for column in coordinates.IDENTITY_COLUMNS:
        value, missing_reason = _one_identity_value(frame, column)
        identity[column] = value
        if missing_reason:
            missing.append(column)
    channel_labels_by_window = coordinates._read_channel_labels_by_window(frame)
    normalized_sets = {tuple(sorted(label.strip().casefold() for label in labels)) for labels in channel_labels_by_window if labels}
    if len(normalized_sets) > 1:
        raise ValueError("MNPS 3.4 application found multiple channel sets within one target recording")
    order_variants = sorted({tuple(labels) for labels in channel_labels_by_window if labels})
    if not order_variants:
        missing.append("mnps_v3_4_channel_labels_json")
    identity["ordered_channel_labels_by_window"] = [list(labels) for labels in order_variants]
    identity["channel_order_variation_qc"] = len(order_variants) > 1
    identity["channel_order_hashes"] = sorted(
        str(value) for value in frame.get("mnps_v3_4_channel_order_hash", pd.Series(dtype=object)).dropna().unique()
        if str(value).strip()
    )
    identity["channel_order_hash_metadata_missing"] = not bool(identity["channel_order_hashes"])
    identity["channel_count"] = max((len(labels) for labels in channel_labels_by_window), default=0)
    v34 = resolve_v3_4_config(config)
    surface = v34.get("measurement_surface", {}) if isinstance(v34, Mapping) else {}
    overlay = surface.get("dataset_overlay", {}) if isinstance(surface, Mapping) else {}
    overlay = overlay if isinstance(overlay, Mapping) else {}
    identity["dataset_id"] = overlay.get("dataset_id") or config.get("dataset_id")
    identity["modality"] = surface.get("modality") if isinstance(surface, Mapping) else None
    identity["target_condition"] = coordinates._recording_condition(frame)
    identity["feature_units"] = surface.get("feature_units") if isinstance(surface, Mapping) else None
    return identity, missing


_OVERLAY_COMPATIBILITY_FIELDS = (
    "dataset_id",
    "dataset_version",
    "sampling_rate_hz",
    "realized_sampling_rate_hz",
    "resampling_method",
    "filter_specification",
    "line_frequency_hz",
    "reference",
    "channel_selection",
    "bad_channel_policy",
    "channel_label_set_compatibility",
    "continuous_segment_boundaries",
    "edge_support_policy",
    "estimator_parameter_manifest",
)


def _compatibility_report(
    manifest: Mapping[str, Any],
    target_identity: Mapping[str, Any],
    missing_identity: list[str],
    config: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    reference = manifest.get("reference", {})
    reference = reference if isinstance(reference, Mapping) else {}
    reference_overlay = reference.get("dataset_overlay", {})
    reference_overlay = reference_overlay if isinstance(reference_overlay, Mapping) else {}
    v34 = resolve_v3_4_config(config)
    target_surface = v34.get("measurement_surface", {}) if isinstance(v34, Mapping) else {}
    target_surface = target_surface if isinstance(target_surface, Mapping) else {}
    target_overlay = target_surface.get("dataset_overlay", {})
    target_overlay = target_overlay if isinstance(target_overlay, Mapping) else {}
    anchor_surface = manifest.get("surface", {})
    anchor_surface = anchor_surface if isinstance(anchor_surface, Mapping) else {}

    mismatches: list[dict[str, Any]] = []
    missing: list[str] = list(missing_identity)
    checked: list[str] = []
    for field in coordinates.IDENTITY_COLUMNS:
        expected = anchor_surface.get(field)
        observed = target_identity.get(field)
        checked.append(field)
        if expected in (None, "") or observed in (None, ""):
            if field not in missing:
                missing.append(field)
        elif str(expected) != str(observed):
            mismatches.append({"field": field, "reference": expected, "target": observed})

    for field in _OVERLAY_COMPATIBILITY_FIELDS:
        expected = reference_overlay.get(field)
        observed = target_overlay.get(field)
        checked.append(f"measurement_surface.dataset_overlay.{field}")
        if expected is None or observed is None:
            missing.append(f"measurement_surface.dataset_overlay.{field}")
        elif expected != observed:
            mismatches.append({
                "field": f"measurement_surface.dataset_overlay.{field}",
                "reference": expected,
                "target": observed,
            })

    reference_units = manifest.get("_runtime_reference_feature_units")
    target_units = target_identity.get("feature_units")
    reference_surface_hash = anchor_surface.get("mnps_v3_4_source_surface_hash")
    target_surface_hash = target_identity.get("mnps_v3_4_source_surface_hash")
    unit_resolution = "explicit"
    if not isinstance(reference_units, Mapping) or not reference_units:
        if reference_surface_hash and target_surface_hash and reference_surface_hash == target_surface_hash and isinstance(target_units, Mapping):
            reference_units = target_units
            unit_resolution = "inferred_from_exact_source_surface_identity"
            missing.append("anchor_manifest.explicit_feature_units")
        else:
            reference_units = {}
            unit_resolution = "unresolved"
            missing.append("anchor_manifest.explicit_feature_units")
    if not isinstance(target_units, Mapping) or not target_units:
        target_units = {}
        missing.append("measurement_surface.feature_units")
    unit_blockers: list[str] = []
    for name in coordinates.FEATURE_ORDER:
        reference_unit = reference_units.get(name)
        target_unit = target_units.get(name)
        if not reference_unit or not target_unit:
            unit_blockers.append(name)
            missing.append(f"measurement_surface.feature_units.{name}")
            continue
        if str(reference_unit) != str(target_unit):
            unit_blockers.append(name)
            mismatches.append({
                "field": f"feature_units.{name}",
                "reference": reference_unit,
                "target": target_unit,
            })
    return {
        "mode": selection["compatibility"],
        "status": "pending_policy",
        "compatible": not mismatches and not missing,
        "checked_fields": checked,
        "known_mismatches": mismatches,
        "missing_compatibility_metadata": sorted(set(missing)),
        "feature_units_resolution": unit_resolution,
        "feature_unit_application_blockers": sorted(set(unit_blockers)),
        "unresolved_missing_compatibility_metadata": sorted(
            field for field in set(missing)
            if not (field == "anchor_manifest.explicit_feature_units" and unit_resolution == "inferred_from_exact_source_surface_identity")
        ),
        "channel_order_differences_are_qc": bool(target_identity.get("channel_order_variation_qc")),
        "channel_order_qc": {
            "reference_hashes": manifest.get("_runtime_reference_channel_order_hashes"),
            "target_hashes": target_identity.get("channel_order_hashes", []),
            "differences": (
                sorted(manifest.get("_runtime_reference_channel_order_hashes", []))
                != sorted(target_identity.get("channel_order_hashes", []))
                if manifest.get("_runtime_reference_channel_order_hashes") is not None
                else None
            ),
            "difference_is_compatibility_blocker": False,
        },
        "transfer_reason": selection.get("transfer_reason"),
    }


def _target_participants(frame: pd.DataFrame) -> tuple[set[str], str]:
    try:
        raw_participants = coordinates._participant_ids(frame)
    except ValueError:
        return set(), "unknown"
    participants: set[str] = set()
    for value in raw_participants:
        label = str(value).strip()
        if not label:
            return set(), "unknown"
        # Treat common BIDS numeric forms as the same participant. The anchors
        # store IDs like ``sub-001``; input tables sometimes carry ``1`` or
        # ``sub-1`` instead. Keep nonnumeric labels intact apart from case and
        # surrounding whitespace so arbitrary BIDS subject names still match.
        numeric = re.fullmatch(r"(?i)(?:sub-)?(\d+)", label)
        if numeric:
            participants.add(f"sub-{int(numeric.group(1)):03d}")
        else:
            participants.add(label.casefold())
    return participants, "known" if participants else "unknown"


def _apply_anchor_values(
    payload: dict[str, Any], frame: pd.DataFrame, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    transformed, feature_names = coordinates._input_matrix(frame)
    feature_scaled = coordinates._feature_scaled_values(
        transformed, feature_names, manifest["feature_anchors"]
    )
    coordinate_pre_reference = coordinates._assemble_coordinates(feature_scaled, feature_names)
    coordinate_values = np.full_like(coordinate_pre_reference, np.nan, dtype=np.float64)
    for j, name in enumerate(coordinates.COORDINATE_ORDER):
        center, scale = _finite_positive_anchor(manifest["coordinate_anchors"].get(name), label=f"coordinate {name}")
        supported = np.isfinite(coordinate_pre_reference[:, j])
        coordinate_values[supported, j] = (coordinate_pre_reference[supported, j] - center) / scale
    payload["feature_scaled_values"] = {
        "values": feature_scaled,
        "names": feature_names,
        "support": np.isfinite(feature_scaled).astype(np.uint8),
    }
    payload["coordinate_pre_reference_values"] = coordinate_pre_reference
    payload["coordinates"] = {"values": coordinate_values, "names": list(coordinates.COORDINATE_ORDER)}
    payload["coordinate_validity_mask"] = np.isfinite(coordinate_values).astype(np.uint8)
    payload["complete_9d_mask"] = np.isfinite(coordinate_values).all(axis=1).astype(np.uint8)
    return payload


def _mark_external_anchor_unsupported(
    payload: dict[str, Any],
    manifest: Mapping[str, Any] | None,
    selection: Mapping[str, Any],
    target_identity: Mapping[str, Any],
    compatibility: Mapping[str, Any] | None,
    reason: str,
) -> dict[str, Any]:
    """Preserve computable unanchored measurements when anchor use is unsupported."""
    manifest = manifest if isinstance(manifest, Mapping) else {}
    reference = manifest.get("reference", {})
    reference = reference if isinstance(reference, Mapping) else {}
    reference_overlay = reference.get("dataset_overlay", {})
    reference_overlay = reference_overlay if isinstance(reference_overlay, Mapping) else {}
    provenance = dict(payload.get("provenance", {}) or {})
    provenance.update({
        "anchor_id": manifest.get("anchor_id") or selection.get("expected_anchor_id") or (
            PRESETS.get(str(selection.get("preset")), {}).get("anchor_id")
        ),
        "anchor_reference_condition": reference.get("condition") or (
            PRESETS.get(str(selection.get("preset")), {}).get("reference_condition")
        ),
        "target_condition": target_identity.get("target_condition"),
        "anchor_hash": manifest.get("anchor_hash"),
        "anchor_manifest_sha256": manifest.get("_runtime_file_sha256") or selection.get("_runtime_failure_file_sha256"),
        "anchor_manifest_path": manifest.get("_runtime_display_path") or selection.get("_runtime_failure_path") or selection.get("path") or (f"preset:{selection.get('preset')}" if selection.get("preset") else None),
        "anchor_schema": manifest.get("schema"),
        "anchor_definition_version": manifest.get("definition_version"),
        "anchor_status": manifest.get("status", "unavailable_or_invalid"),
        "anchor_qualified": bool(manifest.get("qualified", False)),
        "anchor_reference_identity": {
            "dataset_id": reference_overlay.get("dataset_id") or (
                "ds005385" if selection.get("preset") in PRESETS else None
            ),
            "dataset_version": reference_overlay.get("dataset_version"),
            "scope": reference.get("scope"),
            "cohort_manifest_sha256": (manifest.get("anchor_fit_source", {}) or {}).get("cohort_manifest_sha256"),
            "feature_table_sha256": (manifest.get("anchor_fit_source", {}) or {}).get("features_table_sha256"),
            "reference_participant_count": reference.get("participant_count"),
            "reference_channel_order_hashes": manifest.get("_runtime_reference_channel_order_hashes"),
        },
        "anchor_fit_scope": reference.get("evaluation_status"),
        "reference_fit_and_apply_overlap": reference.get("fit_and_apply_overlap"),
        "reference_evaluation_status": reference.get("evaluation_status"),
        "application_scope": "anchor_application_unsupported",
        "target_reference_participant_overlap_status": "unknown",
        "participant_overlap_identity_scope": "not_evaluated_anchor_unsupported",
        "target_reference_overlap_participant_count": None,
        "selected_compatibility_mode": selection.get("compatibility", "strict"),
        "measurement_compatibility": dict(compatibility or {}),
        "target_measurement_identity": dict(target_identity),
        "anchor_application_status": "unsupported_anchor_application",
        "anchor_application_failure_reason": reason,
        "qualification_status": "unqualified",
        "qualification_reasons": [
            "MNPS 3.4 anchor is experimental and not a qualified population norm",
            "the selected anchor could not be applied to this feature realization",
        ],
        "transfer_reason": selection.get("transfer_reason"),
    })
    payload["provenance"] = provenance
    payload["status"] = "experimental_anchor_unsupported"
    payload["qualified"] = False
    return payload


def _validate_feature_implementation_hashes(features: pd.DataFrame) -> None:
    """Require every source window to identify the exact feature realization."""
    column = "mnps_v3_4_implementation_hashes_json"
    if column not in features or features.empty:
        raise ValueError("MNPS 3.4 external anchor application requires per-window feature implementation hashes")
    values = features[column]
    if values.isna().any() or values.astype(str).str.strip().eq("").any():
        raise ValueError("MNPS 3.4 external anchor application rejects missing per-window feature implementation hashes")
    expected = coordinates.runtime_implementation_hashes()
    for row_index, value in values.items():
        try:
            parsed = json.loads(str(value))
        except Exception as exc:
            raise ValueError(
                f"MNPS 3.4 feature implementation hash manifest is invalid at row {row_index}"
            ) from exc
        if not isinstance(parsed, Mapping) or dict(parsed) != expected:
            raise ValueError(
                f"MNPS 3.4 feature implementation hash mismatch at row {row_index}"
            )


def build_measurement_export_with_external_anchor(
    features: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    artifact_qc_applied: bool | None = None,
) -> dict[str, Any] | None:
    """Build the normal payload and, only when explicitly selected, apply a frozen anchor."""
    from .coordinates import build_measurement_export

    try:
        loaded = load_selected_anchor(config)
    except (OSError, ValueError) as exc:
        v34 = resolve_v3_4_config(config)
        if not v34 or not bool(v34.get("enabled", False)):
            raise
        payload = _build_unanchored_payload(
            features, config, artifact_qc_applied=artifact_qc_applied
        )
        if payload is None:
            return None
        selection = _unresolved_selection(config)
        try:
            target_identity, _ = _target_identity(features, config)
        except ValueError:
            target_identity = {}
        return _mark_external_anchor_unsupported(
            payload, None, selection, target_identity, None,
            f"anchor_manifest_unavailable_or_invalid: {exc}",
        )
    if loaded is None:
        return build_measurement_export(
            features,
            _resolve_legacy_anchor_paths(config),
            artifact_qc_applied=artifact_qc_applied,
        )
    manifest, selection = loaded
    v34 = resolve_v3_4_config(config)
    if not v34 or not bool(v34.get("enabled", False)):
        raise ValueError("MNPS 3.4 external anchors require an explicitly enabled v3.4 configuration")
    # Ask the established exporter for its unanchored/raw-feature/QC payload.
    # This is retained even if reference application is unsupported, so an
    # anchor compatibility problem cannot suppress otherwise computable output.
    payload = _build_unanchored_payload(
        features, config, artifact_qc_applied=artifact_qc_applied
    )
    if not isinstance(payload, dict):
        raise ValueError("MNPS 3.4 base measurement export could not be created")

    try:
        target_identity, missing_identity = _target_identity(features, config)
    except ValueError as exc:
        return _mark_external_anchor_unsupported(
            payload, manifest, selection, {}, None,
            f"target_measurement_identity_unresolved: {exc}",
        )
    compatibility = _compatibility_report(manifest, target_identity, missing_identity, config, selection)
    try:
        _validate_anchor_contract(manifest, v34)
    except ValueError as exc:
        return _mark_external_anchor_unsupported(
            payload, manifest, selection, target_identity, compatibility,
            f"anchor_contract_invalid: {exc}",
        )

    strict_incompatibility = (
        selection["compatibility"] == "strict"
        and (compatibility["known_mismatches"] or compatibility["unresolved_missing_compatibility_metadata"])
    )
    if strict_incompatibility:
        compatibility["status"] = "strict_incompatible"
        compatibility["compatible"] = False
        reason = "strict_anchor_compatibility_failed: " + json.dumps({
            "known_mismatches": compatibility["known_mismatches"],
            "missing_compatibility_metadata": compatibility["unresolved_missing_compatibility_metadata"],
        }, ensure_ascii=False, sort_keys=True)
        return _mark_external_anchor_unsupported(
            payload, manifest, selection, target_identity, compatibility, reason
        )
    if compatibility["feature_unit_application_blockers"]:
        return _mark_external_anchor_unsupported(
            payload, manifest, selection, target_identity, compatibility,
            "feature_units_unresolved_or_incompatible: "
            + ", ".join(compatibility["feature_unit_application_blockers"]),
        )
    if selection["compatibility"] == "strict":
        compatibility["status"] = "strict_compatible"
        compatibility["compatible"] = True
    else:
        compatibility["status"] = "exploratory_transfer_measurement_incompatible"
        compatibility["compatible"] = False

    try:
        _validate_feature_implementation_hashes(features)
    except ValueError as exc:
        return _mark_external_anchor_unsupported(
            payload, manifest, selection, target_identity, compatibility, str(exc)
        )
    base_provenance = payload.get("provenance", {})
    recorded_hashes = (
        base_provenance.get("implementation_hashes_recorded_by_features", [])
        if isinstance(base_provenance, Mapping) else []
    )
    summarize_hashes = (
        base_provenance.get("implementation_hashes_at_summarize")
        if isinstance(base_provenance, Mapping) else None
    )
    if len(recorded_hashes) != 1 or recorded_hashes[0] != summarize_hashes:
        return _mark_external_anchor_unsupported(
            payload, manifest, selection, target_identity, compatibility,
            "MNPS 3.4 external anchor application requires exactly one feature implementation-hash "
            "manifest matching the summarize runtime"
        )
    payload = _apply_anchor_values(payload, features, manifest)

    provenance = dict(payload.get("provenance", {}) or {})
    reference = manifest.get("reference", {})
    reference = reference if isinstance(reference, Mapping) else {}
    fit_participants, fit_participant_status = _target_participants(
        pd.DataFrame({"subject": list(reference.get("participants", []) or [])})
    ) if reference.get("participants") else (set(), "unknown")
    target_participants, target_participant_status = _target_participants(features)
    reference_overlay = reference.get("dataset_overlay", {})
    reference_overlay = reference_overlay if isinstance(reference_overlay, Mapping) else {}
    reference_dataset_id = reference_overlay.get("dataset_id")
    target_dataset_id = target_identity.get("dataset_id")
    reference_dataset_key = str(reference_dataset_id).strip().casefold() if reference_dataset_id is not None else ""
    target_dataset_key = str(target_dataset_id).strip().casefold() if target_dataset_id is not None else ""
    same_participant_namespace = (
        bool(reference_dataset_key)
        and bool(target_dataset_key)
        and reference_dataset_key == target_dataset_key
    )
    overlap = (
        target_participants & fit_participants
        if same_participant_namespace
        and target_participant_status == "known"
        and fit_participant_status == "known"
        else set()
    )
    if not same_participant_namespace or target_participant_status == "unknown" or fit_participant_status == "unknown":
        overlap_status = "unknown"
        application_scope = "participant_overlap_unknown"
    elif overlap:
        overlap_status = "overlap_detected"
        application_scope = "reference_cohort_overlap"
    else:
        overlap_status = "no_overlap_detected"
        application_scope = "external_application"
    reference_identity = {
        "dataset_id": reference_overlay.get("dataset_id"),
        "dataset_version": reference_overlay.get("dataset_version"),
        "scope": reference.get("scope"),
        "cohort_manifest_sha256": (manifest.get("anchor_fit_source", {}) or {}).get("cohort_manifest_sha256"),
        "feature_table_sha256": (manifest.get("anchor_fit_source", {}) or {}).get("features_table_sha256"),
        "reference_participant_count": reference.get("participant_count"),
        "reference_channel_order_hashes": manifest.get("_runtime_reference_channel_order_hashes"),
    }
    provenance.update({
        "anchor_id": manifest.get("anchor_id"),
        "anchor_reference_condition": reference.get("condition"),
        "target_condition": target_identity.get("target_condition"),
        "anchor_hash": manifest.get("anchor_hash"),
        "anchor_manifest_sha256": manifest.get("_runtime_file_sha256"),
        "anchor_manifest_path": manifest.get("_runtime_display_path"),
        "anchor_schema": manifest.get("schema"),
        "anchor_definition_version": manifest.get("definition_version"),
        "anchor_status": manifest.get("status"),
        "anchor_qualified": bool(manifest.get("qualified", False)),
        "anchor_reference_identity": reference_identity,
        "anchor_fit_scope": reference.get("evaluation_status"),
        "reference_fit_and_apply_overlap": reference.get("fit_and_apply_overlap"),
        "reference_evaluation_status": reference.get("evaluation_status"),
        "application_scope": application_scope,
        "target_reference_participant_overlap_status": overlap_status,
        "participant_overlap_identity_scope": (
            "same_dataset_id" if same_participant_namespace
            else "dataset_identity_missing" if not reference_dataset_key or not target_dataset_key
            else "different_dataset_ids"
        ),
        "target_reference_overlap_participant_count": len(overlap),
        "selected_compatibility_mode": selection["compatibility"],
        "measurement_compatibility": compatibility,
        "target_measurement_identity": target_identity,
        "anchor_application_status": (
            "computed" if selection["compatibility"] == "strict" else "computed_exploratory_transfer"
        ),
        "anchor_application_failure_reason": None,
        "qualification_status": "unqualified",
        "qualification_reasons": [
            "MNPS 3.4 cohort anchor is experimental and not a qualified population norm",
            "reference cohort artifact QC was not established during anchor fitting",
        ] + (["measurement surface differs from reference anchor"] if selection["compatibility"] == "exploratory_transfer" else []),
        "transfer_reason": selection.get("transfer_reason"),
    })
    payload["provenance"] = provenance
    payload["status"] = "experimental_exploratory_transfer" if selection["compatibility"] == "exploratory_transfer" else "experimental_external_anchor_applied"
    payload["qualified"] = False
    return payload


def anchor_run_fingerprint(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the run-level identity that prevents an explicit anchor mix on resume."""
    try:
        loaded = load_selected_anchor(config)
    except (OSError, ValueError) as exc:
        selection = _unresolved_selection(config)
        if not selection:
            return None
        v34 = resolve_v3_4_config(config)
        measurement_surface = v34.get("measurement_surface", {}) if isinstance(v34, Mapping) else {}
        measurement_surface = measurement_surface if isinstance(measurement_surface, Mapping) else {}
        stable_selection = {
            key: value for key, value in selection.items()
            if not str(key).startswith("_runtime_")
        }
        preset_spec = PRESETS.get(str(selection.get("preset"))) if selection.get("preset") else None
        return {
            "anchor_id": selection.get("expected_anchor_id") or (preset_spec.get("anchor_id") if preset_spec else None),
            "anchor_manifest_sha256": selection.get("_runtime_failure_file_sha256") or selection.get("expected_sha256") or (preset_spec.get("sha256") if preset_spec else None),
            "anchor_manifest_path": selection.get("_runtime_failure_path") or selection.get("path") or (f"preset:{selection.get('preset')}" if selection.get("preset") else None),
            "selection_status": "unsupported_manifest",
            "selection_failure_reason": str(exc),
            "selection_config_sha256": _canonical_hash(stable_selection),
            "compatibility_mode": selection.get("compatibility", "strict"),
            "transfer_reason": selection.get("transfer_reason"),
            "reference_condition": preset_spec.get("reference_condition") if preset_spec else None,
            "reference_feature_units_sha256": _canonical_hash(selection.get("reference_feature_units")),
            "target_measurement_surface_sha256": _canonical_hash(measurement_surface),
        }
    if loaded is None:
        return None
    manifest, selection = loaded
    v34 = resolve_v3_4_config(config)
    measurement_surface = v34.get("measurement_surface", {}) if isinstance(v34, Mapping) else {}
    measurement_surface = measurement_surface if isinstance(measurement_surface, Mapping) else {}
    return {
        "anchor_id": manifest.get("anchor_id"),
        "anchor_manifest_sha256": manifest.get("_runtime_file_sha256"),
        "compatibility_mode": selection["compatibility"],
        "transfer_reason": selection.get("transfer_reason"),
        "reference_condition": (manifest.get("reference", {}) or {}).get("condition"),
        "reference_feature_units_sha256": _canonical_hash(
            manifest.get("_runtime_reference_feature_units")
        ),
        "target_measurement_surface_sha256": _canonical_hash(measurement_surface),
    }


def guard_run_anchor_selection(
    run_dir: Path,
    config: Mapping[str, Any],
    *,
    resume: bool,
) -> dict[str, Any] | None:
    """Persist and enforce one explicit external anchor identity per summarize run."""
    fingerprint = anchor_run_fingerprint(config)
    path = Path(run_dir) / "mnps_v3_4_anchor_selection.json"
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Cannot validate existing MNPS 3.4 run anchor selection at {path}: {exc}") from exc
        if fingerprint is None:
            raise ValueError(
                "This summarize run is pinned to an external MNPS 3.4 anchor. "
                "Resume it with the same explicit anchor selection or start a new run directory."
            )
        if existing != fingerprint:
            raise ValueError(
                "Refusing to mix or overwrite MNPS 3.4 anchor realizations in one summarize run: "
                f"existing={existing}, requested={fingerprint}. Start a new run directory."
            )
        return fingerprint
    if fingerprint is None:
        return None
    existing_h5 = list(Path(run_dir).rglob("*.h5"))
    if existing_h5:
        raise ValueError(
            "Cannot safely apply a selected MNPS 3.4 anchor while resuming a run without its anchor "
            f"selection record; found {len(existing_h5)} existing H5 files. Start a new run directory."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fingerprint, indent=2, ensure_ascii=False), encoding="utf-8")
    return fingerprint


__all__ = [
    "PRESETS",
    "PRESET_UNITS",
    "anchor_run_fingerprint",
    "build_measurement_export_with_external_anchor",
    "guard_run_anchor_selection",
    "load_selected_anchor",
    "resolve_anchor_selection",
]
