"""Opt-in drift, diffusion and one-step exports on MNPS 3.4 coordinates."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections.abc import Mapping
from pathlib import Path
import platform
from typing import Any

import numpy as np

from ...pipeline.dynamical_families_export import build_dynamical_families_export
from .coordinates import COORDINATE_ORDER
from .jacobian_adapter import _ANCHOR_APPLIED

SCHEMA = "mndm.mnps_v3_4.dynamics.v1"
DEFAULT_WINDOW_STEP_TOLERANCE_SEC = 1e-9
_LAYER_NAMES = {"raw": "v3_4_raw", "anchored": "v3_4_anchored"}


def attach_v3_4_dynamics(
    payload: Mapping[str, Any] | None,
    config: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Attach opt-in dynamical families without changing legacy 3D outputs."""
    if not isinstance(payload, Mapping) or not payload:
        return None if payload is None else dict(payload)
    out = dict(payload)
    block = config.get("v3_4_dynamics") if isinstance(config, Mapping) else None
    if not isinstance(block, Mapping) or not bool(block.get("enabled", False)):
        return out
    out["dynamics"] = _build_dynamics(out, block, config or {})
    return out


def attach_meg_v3_4_dynamics(
    payload: Mapping[str, Any] | None,
    config: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Attach dynamics independently to each raw MAG/GRAD coordinate variant.

    MEG has no fitted external anchor in the current experimental contract.
    Drift and diffusion therefore run on the explicitly raw mixed-unit chart;
    anchored-only one-step estimation remains unavailable with its reason.
    """
    if not isinstance(payload, Mapping) or not payload:
        return None if payload is None else dict(payload)
    out = dict(payload)
    block = config.get("v3_4_dynamics") if isinstance(config, Mapping) else None
    if not isinstance(block, Mapping) or not bool(block.get("enabled", False)):
        return out
    families = out.get("sensor_families")
    if not isinstance(families, Mapping):
        return out
    axis = out.get("window_axis")
    axis = axis if isinstance(axis, Mapping) else {}
    start = np.asarray(axis.get("window_start_sec", []), dtype=np.float64).reshape(-1)
    end = np.asarray(axis.get("window_end_sec", []), dtype=np.float64).reshape(-1)
    provenance = out.get("provenance")
    provenance = dict(provenance) if isinstance(provenance, Mapping) else {}
    provenance["anchor_application_status"] = str(
        out.get("anchor_application_status") or "not_applied"
    )
    updated_families: dict[str, Any] = {}
    for family_name, family_payload in families.items():
        if not isinstance(family_payload, Mapping):
            updated_families[str(family_name)] = family_payload
            continue
        family_out = dict(family_payload)
        variants = family_out.get("variants")
        if not isinstance(variants, Mapping):
            updated_families[str(family_name)] = family_out
            continue
        updated_variants: dict[str, Any] = {}
        for variant_id, variant_payload in variants.items():
            if not isinstance(variant_payload, Mapping):
                updated_variants[str(variant_id)] = variant_payload
                continue
            variant_out = dict(variant_payload)
            values = np.asarray(variant_out.get("raw_values", []), dtype=np.float64)
            support = np.asarray(variant_out.get("coordinate_support", []), dtype=bool)
            complete = np.asarray(variant_out.get("complete_window_mask", []), dtype=bool)
            adapter_payload = {
                "coordinate_order": list(variant_out.get("coordinate_names") or []),
                "raw_coordinates": {"values": values, "support": support},
                "coordinates": {"values": np.full_like(values, np.nan) if values.ndim == 2 else None},
                "complete_coordinate_mask": complete,
                "window_start_sec": start,
                "window_end_sec": end,
                "provenance": provenance,
            }
            dynamics = _build_dynamics(
                adapter_payload,
                block,
                config or {},
                allow_noncanonical_coordinates=True,
                complete_mask_key="complete_coordinate_mask",
            )
            dynamics["sensor_family"] = str(family_name)
            dynamics["coordinate_variant"] = str(variant_id)
            dynamics["coordinate_units"] = list(variant_out.get("coordinate_units") or [])
            dynamics["coordinate_dimension"] = int(values.shape[1]) if values.ndim == 2 else 0
            one_step_cfg = block.get("one_step")
            one_step_cfg = one_step_cfg if isinstance(one_step_cfg, Mapping) else {}
            if bool(one_step_cfg.get("enabled", False)):
                dynamics["one_step_status"] = {
                    "computation_status": "not_testable",
                    "failure_reason": "one_step_requires_anchored_coordinates; MEG_anchor_application_status_not_applied",
                }
            variant_out["dynamics"] = dynamics
            updated_variants[str(variant_id)] = variant_out
        family_out["variants"] = updated_variants
        updated_families[str(family_name)] = family_out
    out["sensor_families"] = updated_families
    return out


def _build_dynamics(
    payload: Mapping[str, Any],
    block: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    allow_noncanonical_coordinates: bool = False,
    complete_mask_key: str = "complete_9d_mask",
) -> dict[str, Any]:
    names = [str(name) for name in list(payload.get("coordinate_order") or [])]
    start = _window_vector(payload, "window_start_sec")
    end = _window_vector(payload, "window_end_sec")
    if start.size != end.size:
        return _unavailable("window_time_shape_mismatch", names)
    center = (start + end) / 2.0
    n_windows = center.size
    complete = _complete_mask(payload, n_windows, key=complete_mask_key)
    provenance = payload.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    anchor_status = _as_text(provenance.get("anchor_application_status"))
    anchor_identity = {
        key: _as_text(provenance.get(source))
        for key, source in (
            ("anchor_id", "anchor_id"),
            ("anchor_hash", "anchor_hash"),
            ("anchor_manifest_sha256", "anchor_manifest_sha256"),
        )
    }

    if not names or (not allow_noncanonical_coordinates and names != list(COORDINATE_ORDER)):
        return _unavailable("v3_4_coordinate_order_mismatch", names)
    expected_step = _expected_step(block, config)
    step_tolerance = _window_step_tolerance(block)
    if not np.isfinite(expected_step) or expected_step <= 0:
        return _unavailable("positive_window_step_sec_required", names)
    if not np.isfinite(step_tolerance) or step_tolerance < 0:
        return _unavailable("nonnegative_window_step_tolerance_sec_required", names)

    raw_group = payload.get("raw_coordinates")
    raw_group = raw_group if isinstance(raw_group, Mapping) else {}
    raw_values = _coordinate_values(raw_group, n_windows, len(names))
    raw_support = _coordinate_support(raw_group, raw_values)
    raw_valid = raw_support
    raw_state = _masked_state(raw_values, raw_valid)

    anchored_group = payload.get("coordinates")
    anchored_group = anchored_group if isinstance(anchored_group, Mapping) else {}
    anchored_values = _coordinate_values(anchored_group, n_windows, len(names))
    anchored_valid = complete & np.isfinite(anchored_values).all(axis=1)
    anchor_applied = anchor_status in _ANCHOR_APPLIED
    if not anchor_applied:
        anchored_valid[:] = False
    anchored_state = _masked_state(anchored_values, anchored_valid)

    layer_cfg = block.get("coordinate_layers")
    layer_cfg = layer_cfg if isinstance(layer_cfg, Mapping) else {}
    results: dict[str, Any] = {}
    for layer_key, values, valid in (
        ("raw", raw_state, raw_valid),
        ("anchored", anchored_state, anchored_valid),
    ):
        layer_name = _LAYER_NAMES[layer_key]
        cfg = layer_cfg.get(layer_key, {})
        cfg = cfg if isinstance(cfg, Mapping) else {}
        enabled = bool(cfg.get("enabled", True))
        if not enabled:
            continue
        if layer_key == "anchored" and not anchor_applied:
            results[layer_name] = _unavailable(
                "external_anchor_not_applied", names, layer=layer_name,
                coordinate_unit_note="dimensionless_reference_scores",
            )
            continue
        if layer_key == "anchored" and not all(anchor_identity.values()):
            results[layer_name] = _unavailable(
                "anchor_identity_incomplete", names, layer=layer_name,
                coordinate_unit_note="dimensionless_reference_scores",
            )
            continue

        family_config = _family_config(block, include_one_step=layer_key == "anchored")
        if not any(bool(family_config.get(key, {}).get("enabled", False)) for key in ("drift", "diffusion", "one_step")):
            continue
        segment = _continuous_valid_segments(valid, center, expected_step, step_tolerance)
        v3_4_config = _json_safe(block)
        family_output = build_dynamical_families_export(
            config={"dynamical_families": family_config},
            state=values,
            time=center,
            stage=None,
            segment_id=segment,
            coordinate_layer=layer_name,
            coordinate_names=names,
        )
        results[layer_name] = {
            "computation_status": _family_status(family_output),
            "coordinate_layer": layer_name,
            "coordinate_names": names,
            "coordinate_unit_note": (
                "mixed_source_units" if layer_key == "raw" else "dimensionless_reference_scores"
            ),
            "window_center_sec": center,
            "window_index": np.arange(n_windows, dtype=np.int32),
            ("complete_coordinate_mask" if allow_noncanonical_coordinates else "complete_9d_mask"): valid.astype(np.int8),
            "segment_id": segment,
            "families": family_output,
            "provenance": {
                "expected_window_step_sec": float(expected_step),
                "window_step_tolerance_sec": float(step_tolerance),
                "time_semantics": "overlapping_window_features_at_declared_window_step",
                "neighbor_metric": "euclidean_on_declared_coordinates",
                "raw_window_support_independence": "not_established",
                "input_coordinates_sha256": _array_hash(values),
                "window_center_sha256": _array_hash(center),
                "complete_window_mask_sha256": _array_hash(valid.astype(np.float64)),
                "anchor_application_status": anchor_status,
                **(anchor_identity if layer_key == "anchored" else {}),
                "v3_4_dynamics_config": v3_4_config,
                "v3_4_dynamics_config_sha256": _json_hash(v3_4_config),
                "expected_window_step_sec": float(expected_step),
                "effective_family_config": family_config,
                "effective_family_config_sha256": _json_hash(family_config),
                "implementation_hashes_at_summarize": _json_safe(
                    provenance.get("implementation_hashes_at_summarize", {})
                ),
                "dynamics_implementation_hashes": _implementation_hashes(),
                "software_versions": _software_versions(),
            },
        }
    layer_statuses = [
        str(value.get("computation_status", ""))
        for value in results.values()
        if isinstance(value, Mapping)
    ]
    if "computed" in layer_statuses:
        computation_status = "computed"
    elif "insufficient_support" in layer_statuses:
        computation_status = "insufficient_support"
    elif "invalid" in layer_statuses:
        computation_status = "invalid"
    else:
        computation_status = "not_testable"
    return {
        "schema": SCHEMA,
        "computation_status": computation_status,
        "coordinate_order": names,
        "n_windows": int(n_windows),
        "coordinate_realizations": results,
        "provenance": {
            "definition_version": "mnps_v3_4_dynamics_v1",
            "anchor_application_status": anchor_status,
            **anchor_identity,
        },
    }


def _family_config(block: Mapping[str, Any], *, include_one_step: bool) -> dict[str, Any]:
    out = {"enabled": True}
    for key in ("drift", "diffusion", "one_step"):
        value = block.get(key, {})
        out[key] = dict(value) if isinstance(value, Mapping) else {}
    if not include_one_step:
        out["one_step"]["enabled"] = False
    for family_name in ("drift", "diffusion"):
        family = out[family_name]
        if not bool(family.get("enabled", False)):
            continue
        neighborhood = family.get("neighborhood")
        neighborhood = dict(neighborhood) if isinstance(neighborhood, Mapping) else {}
        neighborhood.setdefault("k", 32)
        family["neighborhood"] = neighborhood
        family.setdefault("min_neighborhood_samples", 28)
        family.setdefault("tie_break_neighbors_by_index", True)
        if family_name == "drift":
            family.setdefault("weight_mode", "uniform")
    one_step = out["one_step"]
    if bool(one_step.get("enabled", False)):
        one_step.setdefault("declared_lags", [1])
    return out


def _expected_step(block: Mapping[str, Any], config: Mapping[str, Any]) -> float:
    declared = block.get("nominal_dt_sec")
    if declared is None:
        surface = config.get("measurement_surface")
        surface = surface if isinstance(surface, Mapping) else {}
        declared = surface.get("window_step_sec")
    try:
        return float(declared)
    except (TypeError, ValueError):
        return float("nan")


def _window_step_tolerance(block: Mapping[str, Any]) -> float:
    try:
        return float(block.get("window_step_tolerance_sec", DEFAULT_WINDOW_STEP_TOLERANCE_SEC))
    except (TypeError, ValueError):
        return float("nan")


def _coordinate_values(group: Mapping[str, Any], n: int, dimension: int) -> np.ndarray:
    values = group.get("values")
    if values is None:
        return np.full((n, dimension), np.nan, dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (n, dimension):
        return np.full((n, dimension), np.nan, dtype=np.float64)
    return array


def _coordinate_support(group: Mapping[str, Any], values: np.ndarray) -> np.ndarray:
    support = group.get("support")
    if support is None:
        return np.isfinite(values).all(axis=1)
    mask = np.asarray(support, dtype=bool)
    if mask.shape != values.shape:
        return np.zeros(values.shape[0], dtype=bool)
    return mask.all(axis=1) & np.isfinite(values).all(axis=1)


def _complete_mask(payload: Mapping[str, Any], n: int, *, key: str = "complete_9d_mask") -> np.ndarray:
    complete = payload.get(key)
    if complete is None:
        return np.zeros(n, dtype=bool)
    mask = np.asarray(complete, dtype=bool).reshape(-1)
    return mask if mask.size == n else np.zeros(n, dtype=bool)


def _window_vector(payload: Mapping[str, Any], name: str) -> np.ndarray:
    value = payload.get(name)
    if value is None:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _masked_state(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    state = np.array(values, dtype=np.float64, copy=True)
    state[~valid] = np.nan
    return state


def _continuous_valid_segments(
    valid: np.ndarray,
    time: np.ndarray,
    step: float,
    tolerance: float,
) -> np.ndarray:
    segment = np.zeros(valid.size, dtype=np.int32)
    current = 0
    for idx in range(1, valid.size):
        contiguous = (
            bool(valid[idx - 1])
            and bool(valid[idx])
            and np.isfinite(time[idx - 1])
            and np.isfinite(time[idx])
            and bool(
                np.isclose(
                    time[idx] - time[idx - 1],
                    step,
                    rtol=0.0,
                    atol=tolerance,
                )
            )
        )
        if not contiguous:
            current += 1
        segment[idx] = current
    return segment


def _family_status(families: Mapping[str, Any]) -> str:
    statuses = [
        str(value.get("computation_status", ""))
        for value in families.values()
        if isinstance(value, Mapping)
    ]
    if any(status == "computed" for status in statuses):
        return "computed"
    return statuses[0] if statuses else "not_testable"


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _implementation_hashes() -> dict[str, str]:
    mndm_root = Path(__file__).resolve().parents[2]
    paths = {
        "v3_4_dynamics_adapter": Path(__file__),
        "dynamical_families_export": mndm_root / "pipeline" / "dynamical_families_export.py",
        "finite_lag_drift": mndm_root / "dynamical_families" / "finite_lag_drift.py",
        "diffusion_geometry": mndm_root / "dynamical_families" / "diffusion_geometry.py",
        "one_step_operator": mndm_root / "dynamical_families" / "one_step_operator.py",
        "transition_support": mndm_root / "dynamical_families" / "transition_support.py",
        "validity": mndm_root / "dynamical_families" / "validity.py",
    }
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
        if path.is_file()
    }


def _software_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("numpy", "scipy", "h5py"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def _json_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if value is None:
        return ""
    return str(value)


def _unavailable(
    reason: str,
    names: list[str],
    *,
    layer: str = "unknown",
    coordinate_unit_note: str = "unknown",
) -> dict[str, Any]:
    return {
        "computation_status": "not_testable",
        "failure_reason": reason,
        "coordinate_layer": layer,
        "coordinate_names": names,
        "coordinate_unit_note": coordinate_unit_note,
    }
