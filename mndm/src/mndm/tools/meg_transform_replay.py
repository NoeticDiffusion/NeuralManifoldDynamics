"""Deterministic validation of exported MEG feature transforms and coordinates.

This is an audit tool: it never changes an H5 or redefines MEG as an EEG-like
surface.  Grouped helmet features must follow their base feature's transform.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd

from .. import projection
from .projection_replay import resolve_effective_mapping


class MEGTransformReplayError(ValueError):
    """Raised for a missing or inconsistent MEG transform contract."""


def _names(values: np.ndarray) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def _base_feature(name: str) -> str:
    return name.split("__g_", 1)[0]


def _is_power_feature(name: str) -> bool:
    base = _base_feature(name).lower()
    return any(token in base for token in ("power", "delta", "theta", "alpha", "beta", "gamma"))


def validate_meg_transform_contract(
    raw_values: np.ndarray,
    names: list[str],
    projection_z: np.ndarray,
    config: Mapping[str, Any],
    *,
    tolerance: float = 2e-5,
) -> dict[str, Any]:
    """Replay every ``meg_*`` column and verify its configured export surface."""
    if raw_values.shape != projection_z.shape or raw_values.shape[1] != len(names):
        raise MEGTransformReplayError("features_raw and features_projection_z must have matching shapes/names")
    standardization = config.get("mnps_projection", {}).get("feature_standardization", {})
    clip = float(config.get("mnps_projection", {}).get("clip_threshold", 6.0))
    meg_indices = [i for i, name in enumerate(names) if name.startswith("meg_")]
    if not meg_indices:
        raise MEGTransformReplayError("No meg_* columns found in /features_raw")
    failures: list[str] = []
    inherited: list[str] = []
    max_abs_diff = 0.0
    for idx in meg_indices:
        name = names[idx]
        base = _base_feature(name)
        steps = projection._resolve_feature_pipeline(base, standardization)
        replay = projection._apply_column_pipeline(raw_values[:, idx], steps, clip)
        exported = projection_z[:, idx]
        finite = np.isfinite(replay) & np.isfinite(exported)
        diff = float(np.max(np.abs(replay[finite] - exported[finite]))) if finite.any() else 0.0
        max_abs_diff = max(max_abs_diff, diff)
        if diff > tolerance or not np.array_equal(np.isfinite(replay), np.isfinite(exported)):
            failures.append(f"{name}: exported projection-z differs (max_abs_diff={diff:.3g})")
        if "__g_" in name:
            base_steps = projection._resolve_feature_pipeline(base, standardization)
            if list(steps) != list(base_steps):
                failures.append(f"{name}: grouped feature does not inherit {base}'s transform")
            inherited.append(name)
        has_log10 = any(str(step).lower() == "log10" for step in steps)
        if _is_power_feature(name) and not has_log10:
            failures.append(f"{name}: MEG power feature lacks required log10 transform")
        if any(token in _base_feature(name).lower() for token in ("entropy", "hjorth", "ratio")) and has_log10:
            failures.append(f"{name}: ratio/entropy/Hjorth feature must not receive log10")
    return {
        "status": "ok" if not failures else "failed",
        "n_meg_features": len(meg_indices),
        "n_grouped_features": len(inherited),
        "max_abs_diff": max_abs_diff,
        "failures": failures,
    }


def validate_meg_h5(
    h5_path: Path,
    config: Mapping[str, Any],
    *,
    dataset_id: str,
    tolerance: float = 2e-5,
) -> dict[str, Any]:
    """Validate transform parity and, when present, replay exported MNPS coordinates."""
    path = Path(h5_path)
    with h5py.File(path, "r") as h5:
        required = ("features_raw/values", "features_raw/names", "features_projection_z/values", "features_projection_z/names")
        missing = [key for key in required if key not in h5]
        if missing:
            raise MEGTransformReplayError(f"{path}: missing required surfaces: {missing}")
        raw = np.asarray(h5["features_raw/values"], dtype=np.float32)
        raw_names = _names(h5["features_raw/names"][:])
        projection_z = np.asarray(h5["features_projection_z/values"], dtype=np.float32)
        projection_names = _names(h5["features_projection_z/names"][:])
        if raw_names != projection_names:
            raise MEGTransformReplayError(f"{path}: raw/projection-z names differ")
        report = validate_meg_transform_contract(raw, raw_names, projection_z, config, tolerance=tolerance)
        report["h5_path"] = str(path)
        report["coordinate_replay"] = {"status": "not_available"}
        if "coords_9d/values" in h5 and "coords_9d/names" in h5:
            frame = pd.DataFrame(raw, columns=raw_names)
            effective = resolve_effective_mapping(config, dataset_id)
            coords, names, _ = projection.project_features_v2(
                frame, effective["subcoords_spec"], normalize=effective["normalize_mode"],
                feature_standardization=effective["feature_standardization"], clip_threshold=effective["clip_threshold"],
            )
            stored_names = _names(h5["coords_9d/names"][:])
            stored = np.asarray(h5["coords_9d/values"], dtype=np.float32)
            finite = np.isfinite(coords) & np.isfinite(stored)
            coord_diff = float(np.max(np.abs(coords[finite] - stored[finite]))) if finite.any() else 0.0
            report["coordinate_replay"] = {
                "status": "ok" if names == stored_names and coord_diff <= tolerance else "failed",
                "names_match": names == stored_names,
                "max_abs_diff": coord_diff,
            }
            if report["coordinate_replay"]["status"] != "ok":
                report["failures"].append("coords_9d replay differs from stored projection")
        report["status"] = "ok" if not report["failures"] and report["coordinate_replay"]["status"] != "failed" else "failed"
        return report
