"""Bounded Family B export and audit for an existing MNDM HDF5 file.

Family B in this tool is the continuous-time Jacobian metrics surface already
defined by ``mndm.dynamics.jacobian_metrics``.  The tool deliberately starts
from a saved Jacobian and its saved fit diagnostics.  It does not estimate a
new Jacobian, change the 0.9 fidelity gate, or promote the experimental
discrete one-step map to a production family.

The HDF5 output is a small, canonical writer output containing the source
trajectory, selected coordinate layer, Jacobian, diagnostics, and
``/jacobian[/_9D]/derived_metrics/v1``.  A separate JSON file contains an
audit-only chronological blocked holdout comparison of a global one-step map
against a train-target-mean and persistence baseline.  The audit is
explicitly nominal when raw temporal support is not certified by the source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from core.io import h5_writer

from ..dynamics.jacobian_metrics import compute_jacobian_metrics
from ..measurement_certificate import VALID_COMPUTATION_STATUSES, attach_certificate
from ..schema import MNPSPayload


FAMILY_B_EXPORT_SCHEMA_VERSION = "mndm.family_b_export.v1"
FAMILY_B_FIDELITY_THRESHOLD = 0.9
CANONICAL_JACOBIAN_METRICS_SCHEMA = "mndm.jacobian_metrics.v1"
DEFAULT_HORIZON = 1
DEFAULT_TEST_FRACTION = 0.2
DEFAULT_GAP_RELATIVE_TOLERANCE = 0.05


def _decode(value: Any) -> Any:
    """Decode HDF5 scalar strings while leaving numeric values unchanged."""
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode(value[()])
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_safe(value: Any) -> Any:
    """Convert HDF5/numpy values into JSON-safe metadata."""
    value = _decode(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _read_scalar(group: h5py.Group, key: str, default: Any = None) -> Any:
    if key not in group:
        return default
    return _decode(group[key][()])


def _read_names(dataset: h5py.Dataset) -> list[str]:
    values = np.asarray(dataset[:]).reshape(-1)
    return [str(_decode(value)) for value in values]


def _read_mapping(group: h5py.Group) -> dict[str, Any]:
    """Read a small nested HDF5 mapping, retaining arrays and scalar values."""
    out: dict[str, Any] = {}
    for key, value in group.items():
        if isinstance(value, h5py.Group):
            out[str(key)] = _read_mapping(value)
        else:
            raw = value[()]
            out[str(key)] = _decode(raw)
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_optional(h5: h5py.File, path: str, *, dtype: Any | None = None) -> np.ndarray | None:
    if path not in h5:
        return None
    value = np.asarray(h5[path][:])
    return value.astype(dtype, copy=False) if dtype is not None else value


def _resolve_layer_name(h5: h5py.File, requested: str | None) -> str:
    if requested:
        name = str(requested).strip().replace("/", "_").replace("\\", "_")
        if name not in h5:
            raise ValueError(f"coordinate layer is absent: {name}")
        return name
    attr_name = _decode(h5.attrs.get("primary_coordinate_layer"))
    if attr_name and str(attr_name) in h5:
        return str(attr_name)
    for candidate in ("coords_3d_subject_anchored", "coords_3d_cohort_anchored", "coords_9d_subject_anchored"):
        if candidate in h5:
            return candidate
    raise ValueError("no coordinate layer is available")


@dataclass
class FamilyBSource:
    path: Path
    sha256: str
    dataset_id: str
    coordinate_layer: str
    coordinate_contract: str | None
    coordinate_names: list[str]
    coordinate_values: np.ndarray
    coordinate_attrs: dict[str, Any]
    time: np.ndarray
    mnps: np.ndarray
    mnps_dot: np.ndarray
    window_start: np.ndarray | None
    window_end: np.ndarray | None
    epoch_id: np.ndarray | None
    jacobian_group: str
    jacobian: np.ndarray
    jacobian_dot: np.ndarray | None
    centers: np.ndarray
    affine_reference: np.ndarray | None
    affine_intercept: np.ndarray | None
    diagnostics: dict[str, Any]
    metric_status: str | None
    metric_failure_reason: str | None
    signal_support_status: str | None
    signal_support_reason: str | None
    source_attrs: dict[str, Any]
    signal_support_metadata: dict[str, Any]
    raw_files: np.ndarray | None
    metric_fidelity_threshold: float | None
    source_metric_summary: dict[str, Any]
    source_metric_certificate: dict[str, Any]


def load_family_b_source(path: str | Path, coordinate_layer: str | None = None) -> FamilyBSource:
    """Read and validate the minimum source contract for a Family B export."""
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_hash = _sha256(source_path)
    with h5py.File(source_path, "r") as h5:
        primary_layer = _decode(h5.attrs.get("primary_coordinate_layer"))
        if coordinate_layer is not None and primary_layer and str(coordinate_layer) != str(primary_layer):
            raise ValueError(
                "requested coordinate layer does not match the source primary layer; "
                "an alternate layer requires its matching Jacobian export"
            )
        layer_name = _resolve_layer_name(h5, coordinate_layer)
        layer = h5[layer_name]
        if not isinstance(layer, h5py.Group) or "values" not in layer:
            raise ValueError(f"coordinate layer lacks values: {layer_name}")
        coords = np.asarray(layer["values"][:], dtype=np.float32)
        if coords.ndim != 2 or coords.shape[0] < 2:
            raise ValueError("coordinate layer values must be a non-empty 2-D trajectory")
        if coords.shape[1] not in (3, 9):
            raise ValueError("Family B coordinates must have exactly 3 or 9 dimensions")
        if layer_name.startswith("coords_3d") and coords.shape[1] != 3:
            raise ValueError("coords_3d layer must have exactly 3 dimensions")
        if layer_name.startswith("coords_9d") and coords.shape[1] != 9:
            raise ValueError("coords_9d layer must have exactly 9 dimensions")
        names = _read_names(layer["names"]) if "names" in layer else [f"dim_{i}" for i in range(coords.shape[1])]
        if len(names) != coords.shape[1]:
            raise ValueError("coordinate layer names do not match values")

        time = np.asarray(h5["time"][:], dtype=np.float64) if "time" in h5 else None
        mnps = np.asarray(h5["mnps_3d"][:], dtype=np.float32) if "mnps_3d" in h5 else None
        mnps_dot = np.asarray(h5["mnps_3d_dot"][:], dtype=np.float32) if "mnps_3d_dot" in h5 else None
        if time is None or mnps is None or mnps_dot is None:
            raise ValueError("source must contain time, mnps_3d, and mnps_3d_dot")
        if time.ndim != 1 or time.shape[0] != coords.shape[0]:
            raise ValueError("time and coordinate layer must share a time axis")
        if mnps.shape != (time.shape[0], 3) or mnps_dot.shape != mnps.shape:
            raise ValueError("canonical MNPS arrays must have shape [T, 3]")
        if not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
            raise ValueError("time must be finite and strictly increasing")
        window_start = _read_optional(h5, "window_start", dtype=np.float32)
        window_end = _read_optional(h5, "window_end", dtype=np.float32)
        if (window_start is None) != (window_end is None):
            raise ValueError("window_start and window_end must be provided together")
        if window_start is not None and window_end is not None:
            if window_start.shape != time.shape or window_end.shape != time.shape:
                raise ValueError("window bounds must align with the time axis")
            if not np.isfinite(window_start).all() or not np.isfinite(window_end).all():
                raise ValueError("window bounds must be finite")
            if np.any(window_end <= window_start) or np.any(np.diff(window_start) < 0):
                raise ValueError("window bounds must be positive and ordered")
        epoch_id = _read_optional(h5, "epoch_id", dtype=np.int64)
        if epoch_id is not None and epoch_id.shape != time.shape:
            raise ValueError("epoch_id must align with the time axis")

        is_9d = layer_name.startswith("coords_9d")
        jacobian_group = "jacobian_9D" if is_9d else "jacobian"
        if f"{jacobian_group}/J_hat" not in h5:
            raise ValueError(f"source Jacobian is absent: /{jacobian_group}/J_hat")
        jacobian = np.asarray(h5[f"{jacobian_group}/J_hat"][:], dtype=np.float32)
        if jacobian.ndim != 3 or jacobian.shape[1] != jacobian.shape[2] or jacobian.shape[1] != coords.shape[1]:
            raise ValueError("Jacobian shape must be [W, D, D] and match the selected coordinates")
        if f"{jacobian_group}/centers" not in h5:
            raise ValueError(f"source Jacobian centers are absent: /{jacobian_group}/centers")
        raw_centers = np.asarray(h5[f"{jacobian_group}/centers"][:]).reshape(-1)
        try:
            centers_float = np.asarray(raw_centers, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Jacobian centers must be numeric integer indices") from exc
        if (
            not np.isfinite(centers_float).all()
            or np.any(centers_float != np.floor(centers_float))
            or np.any(centers_float < np.iinfo(np.int32).min)
            or np.any(centers_float > np.iinfo(np.int32).max)
        ):
            raise ValueError("Jacobian centers must be finite integer-valued indices")
        centers = centers_float.astype(np.int32)
        if (
            centers.shape[0] != jacobian.shape[0]
            or np.any(centers < 0)
            or np.any(centers >= time.shape[0])
            or np.any(np.diff(centers) <= 0)
        ):
            raise ValueError("Jacobian centers must align with J_hat and index the time axis")

        diagnostics: dict[str, Any] = {}
        diag_path = f"{jacobian_group}/diagnostics"
        if diag_path in h5:
            diag_group = h5[diag_path]
            for key, value in diag_group.items():
                if isinstance(value, h5py.Dataset):
                    diagnostics[str(key)] = np.asarray(value[:]) if value.shape else _decode(value[()])
            for key, value in diag_group.attrs.items():
                diagnostics[str(key)] = _decode(value)

        metrics_path = f"{jacobian_group}/derived_metrics/v1"
        metric_status = None
        metric_failure = None
        metric_threshold = None
        source_metric_summary: dict[str, Any] = {}
        source_metric_certificate: dict[str, Any] = {}
        if metrics_path in h5:
            metrics_group = h5[metrics_path]
            metric_schema = _read_scalar(metrics_group, "schema_version", None)
            if metric_schema is None and "_schema_version" in metrics_group.attrs:
                metric_schema = _decode(metrics_group.attrs["_schema_version"])
            if metric_schema is not None and str(metric_schema) != CANONICAL_JACOBIAN_METRICS_SCHEMA:
                raise ValueError(
                    "source Jacobian metrics schema is not canonical: "
                    f"{metric_schema!r}"
                )
            raw_metric_status = _read_scalar(metrics_group, "computation_status", None)
            metric_status = str(raw_metric_status).strip() if raw_metric_status is not None else None
            if metric_status == "":
                metric_status = None
            if metric_status is None:
                raise ValueError("source Jacobian metrics computation status is required")
            if metric_status is not None and metric_status not in VALID_COMPUTATION_STATUSES:
                raise ValueError(f"unknown source Jacobian computation status: {metric_status!r}")
            metric_failure = str(_read_scalar(metrics_group, "failure_reason", "")) or None
            if "summary" in metrics_group:
                source_metric_summary = _read_mapping(metrics_group["summary"])
            for key in ("computation_status", "measurement_validity", "claim_status", "failure_reason"):
                if key in metrics_group:
                    source_metric_certificate[key] = _read_scalar(metrics_group, key)
            if "provenance" in metrics_group and "fit_fidelity_threshold" in metrics_group["provenance"]:
                try:
                    metric_threshold = float(_read_scalar(metrics_group["provenance"], "fit_fidelity_threshold"))
                except (TypeError, ValueError):
                    metric_threshold = None
            # Retain the source window series when a diagnostics group is absent.
            if "rel_mse_baseline_windows" not in diagnostics and "series" in metrics_group:
                series = metrics_group["series"]
                if "rel_mse_baseline" in series:
                    diagnostics["rel_mse_baseline_windows"] = np.asarray(series["rel_mse_baseline"][:])

        support_group_path = "provenance/signal_support_provenance"
        support_status = None
        support_reason = None
        support_metadata: dict[str, Any] = {}
        if support_group_path in h5:
            support = h5[support_group_path]
            support_status = str(_read_scalar(support, "temporal_support_status", "")) or None
            support_reason = str(_read_scalar(support, "temporal_support_reason", "")) or None
            for key, value in support.items():
                if isinstance(value, h5py.Dataset) and value.shape == ():
                    support_metadata[str(key)] = _decode(value[()])

        source_attr_keys = (
            "dataset_id",
            "subject_id",
            "dataset",
            "session",
            "run",
            "acq",
            "task",
            "condition",
            "geometry_contract_status",
            "window_sec",
            "window_step_sec",
            "sample_period_sec",
            "mndm_version",
            "schema_version",
        )
        source_attrs = {
            str(key): _decode(h5.attrs[key])
            for key in source_attr_keys
            if key in h5.attrs
        }
        raw_files = None
        if "row_source/raw_file" in h5:
            raw_files = np.asarray(
                [str(_decode(value)) for value in h5["row_source/raw_file"][:]],
                dtype=object,
            )
            if raw_files.shape != time.shape:
                raise ValueError("row_source/raw_file must align with the time axis")

        def optional(group: str, name: str, dtype: Any = np.float32) -> np.ndarray | None:
            key = f"{group}/{name}"
            return np.asarray(h5[key][:], dtype=dtype) if key in h5 else None

        return FamilyBSource(
            path=source_path,
            sha256=source_hash,
            dataset_id=str(_decode(h5.attrs.get("dataset_id", source_path.stem))),
            coordinate_layer=layer_name,
            coordinate_contract=(str(_decode(h5.attrs.get("primary_coordinate_contract")))
                                 if h5.attrs.get("primary_coordinate_contract") is not None
                                 else str(layer.attrs.get("coordinate_contract")) if layer.attrs.get("coordinate_contract") is not None else None),
            coordinate_names=names,
            coordinate_values=coords,
            coordinate_attrs={str(key): _decode(value) for key, value in layer.attrs.items()},
            time=time,
            mnps=mnps,
            mnps_dot=mnps_dot,
            window_start=window_start,
            window_end=window_end,
            epoch_id=epoch_id,
            jacobian_group=jacobian_group,
            jacobian=jacobian,
            jacobian_dot=optional(jacobian_group, "J_dot"),
            centers=centers,
            affine_reference=optional(jacobian_group, "affine_reference"),
            affine_intercept=optional(jacobian_group, "affine_intercept"),
            diagnostics=diagnostics,
            metric_status=metric_status,
            metric_failure_reason=metric_failure,
            signal_support_status=support_status,
            signal_support_reason=support_reason,
            source_attrs=source_attrs,
            signal_support_metadata=support_metadata,
            raw_files=raw_files,
            metric_fidelity_threshold=metric_threshold,
            source_metric_summary=source_metric_summary,
            source_metric_certificate=source_metric_certificate,
        )


def _finite_gate_inputs(source: FamilyBSource) -> tuple[np.ndarray | None, float]:
    windows = source.diagnostics.get("rel_mse_baseline_windows")
    if windows is not None:
        candidate = np.asarray(windows, dtype=float).reshape(-1)
        # A malformed or all-missing per-window diagnostic must fail closed;
        # do not silently fall back to a scalar median from another window
        # population.
        if candidate.shape[0] != source.jacobian.shape[0]:
            return None, float("nan")
        return candidate, float("nan")
    window_values: np.ndarray | None = None
    scalar = source.diagnostics.get("rel_mse_baseline_median")
    if scalar is None:
        scalar = float("nan")
    try:
        scalar_value = float(scalar)
    except (TypeError, ValueError):
        scalar_value = float("nan")
    if window_values is None and not np.isfinite(scalar_value):
        # An existing Family B gate is mandatory for this export tool; absent
        # diagnostics fail closed rather than reopening legacy computation.
        scalar_value = float("nan")
    return window_values, scalar_value


def _invalidate_metric_series(metrics: dict[str, Any]) -> None:
    series = metrics.get("series")
    if not isinstance(series, Mapping):
        return
    for key, value in series.items():
        if key == "rel_mse_baseline":
            # Keep the diagnostic that explains why the scientific series
            # were withheld.
            continue
        array = np.asarray(value)
        if array.dtype.kind in "iu":
            series[key] = np.full(array.shape, -1, dtype=array.dtype)
        elif array.dtype.kind == "b":
            series[key] = np.zeros(array.shape, dtype=array.dtype)
        else:
            series[key] = np.full(array.shape, np.nan, dtype=np.float32)


def _recompute_metrics(source: FamilyBSource) -> dict[str, Any]:
    windows, scalar = _finite_gate_inputs(source)
    threshold = FAMILY_B_FIDELITY_THRESHOLD
    if source.metric_fidelity_threshold is not None and np.isfinite(source.metric_fidelity_threshold):
        threshold = min(threshold, float(source.metric_fidelity_threshold))
    metrics = compute_jacobian_metrics(
        source.jacobian,
        rel_mse_baseline_windows=windows,
        rel_mse_baseline_median=scalar,
        fit_fidelity_threshold=threshold,
        nominal_dt_sec=float(np.median(np.diff(source.time))),
        estimator_diagnostics=source.diagnostics,
    )
    # A source refusal is part of the measurement provenance.  Reconstructing
    # the metric from J_hat must not turn an earlier refusal into a computed
    # Family B object merely because a diagnostic was omitted or changed.  A
    # contradictory source certificate (status="computed" with an explicit
    # fit_identified=False/unknown) is also fail-closed.
    source_fit_contradiction = (
        source.metric_status == "computed"
        and source.source_metric_summary.get("fit_identified") is not True
    )
    source_refusal = bool(source.metric_status and source.metric_status != "computed") or source_fit_contradiction
    if source_refusal:
        metrics["computation_status"] = (
            source.metric_status
            if source.metric_status and source.metric_status != "computed"
            else "insufficient_support"
        )
        metrics["failure_reason"] = (
            "source_fit_identified_false"
            if source_fit_contradiction
            else source.metric_failure_reason or "source_family_b_refusal"
        )
        metrics["summary"]["fit_identified"] = False
        _invalidate_metric_series(metrics)
        metrics["summary"]["n_windows_metrics_valid"] = 0
        metrics["summary"]["n_windows_stable_reactive"] = 0
        metrics["summary"]["stable_reactive_fraction"] = float("nan")
        metrics = attach_certificate(metrics)

    provenance = metrics.setdefault("provenance", {})
    provenance.update(
        {
            "family_id": "B",
            "family_object": "continuous_time_jacobian_metrics",
            "export_schema_version": FAMILY_B_EXPORT_SCHEMA_VERSION,
            "source_h5_sha256": source.sha256,
            "source_jacobian_path": f"/{source.jacobian_group}/J_hat",
            "source_metrics_path": f"/{source.jacobian_group}/derived_metrics/v1",
            "primary_coordinate_layer": source.coordinate_layer,
            "primary_coordinate_contract": source.coordinate_contract,
            "fidelity_threshold_fixed": FAMILY_B_FIDELITY_THRESHOLD,
            "effective_fidelity_threshold": threshold,
            "source_metric_summary": source.source_metric_summary,
            "source_metric_certificate": source.source_metric_certificate,
            "source_jacobian_diagnostics": source.diagnostics,
        }
    )
    return metrics


def _mse(predicted: np.ndarray, observed: np.ndarray) -> float:
    if predicted.shape != observed.shape or predicted.size == 0:
        return float("nan")
    errors = np.asarray(predicted, dtype=float) - np.asarray(observed, dtype=float)
    finite = np.isfinite(errors).all(axis=1)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(np.sum(errors[finite] ** 2, axis=1)))


def _relative(value: float, denominator: float) -> float | None:
    if not np.isfinite(value) or not np.isfinite(denominator) or denominator <= 0:
        return None
    return float(value / denominator)


def blocked_one_step_audit(
    source: FamilyBSource,
    *,
    horizon: int = DEFAULT_HORIZON,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    gap_relative_tolerance: float = DEFAULT_GAP_RELATIVE_TOLERANCE,
) -> dict[str, Any]:
    """Compare global Phi, train mean, and persistence on a blocked holdout.

    This is a diagnostic only.  It has no effect on the Family B HDF5
    computation status and is not a qualification claim.
    """
    if int(horizon) != 1:
        raise ValueError("the bounded Family B audit supports horizon=1 only")
    horizon = 1
    t = source.time
    x = np.asarray(source.coordinate_values, dtype=float)
    n = int(x.shape[0])
    nominal_dt = float(np.median(np.diff(t))) if n > 1 else float("nan")
    expected_dt = float(horizon * nominal_dt)
    src = np.arange(max(0, n - horizon), dtype=np.int32)
    tgt = src + horizon
    pair_dt = t[tgt] - t[src] if src.size else np.zeros((0,), dtype=float)
    clock_ok = np.isfinite(pair_dt) & np.isfinite(expected_dt) & (
        np.abs(pair_dt - expected_dt) <= max(abs(expected_dt), np.finfo(float).eps) * float(gap_relative_tolerance)
    )
    finite = (
        np.isfinite(x[src]).all(axis=1) & np.isfinite(x[tgt]).all(axis=1)
        if src.size
        else np.zeros((0,), dtype=bool)
    )
    epoch_ok = np.ones(src.shape, dtype=bool)
    if source.epoch_id is not None and source.epoch_id.shape == t.shape:
        epoch = np.asarray(source.epoch_id)
        epoch_ok = epoch[tgt] - epoch[src] == horizon
    file_ok = np.ones(src.shape, dtype=bool)
    if source.raw_files is not None and source.raw_files.shape == t.shape:
        file_ok = source.raw_files[src] == source.raw_files[tgt]
    valid = clock_ok & finite & epoch_ok & file_ok

    split = int(np.floor((n - horizon) * (1.0 - float(test_fraction))))
    split = min(max(split, 0), max(0, n - horizon))
    overlap_steps = 0
    if source.window_start is not None and source.window_end is not None:
        ws = np.asarray(source.window_start, dtype=float)
        we = np.asarray(source.window_end, dtype=float)
        if ws.shape == t.shape and we.shape == t.shape:
            widths = we - ws
            positive_steps = np.diff(ws)
            positive_steps = positive_steps[np.isfinite(positive_steps) & (positive_steps > 0)]
            width = float(np.nanmedian(widths[np.isfinite(widths) & (widths > 0)])) if np.any(np.isfinite(widths) & (widths > 0)) else float("nan")
            step = float(np.nanmedian(positive_steps)) if positive_steps.size else float("nan")
            if np.isfinite(width) and np.isfinite(step) and step > 0:
                overlap_steps = max(0, int(np.ceil(width / step)) - 1)

    train_mask = valid & (src < split)
    test_mask = valid & (src >= split + overlap_steps)
    if source.window_start is not None and source.window_end is not None:
        test_starts = np.asarray(source.window_start, dtype=float)[src[test_mask]]
        test_targets = tgt[test_mask]
        window_start = np.asarray(source.window_start, dtype=float)
        window_end = np.asarray(source.window_end, dtype=float)
        # Purge against the earliest interval start in the held-out pairs and
        # the full interval occupied by each training pair.  A median window
        # width is useful for choosing the coarse block, but cannot establish
        # disjointness when source windows have irregular widths.
        test_interval_starts = np.concatenate(
            [window_start[src[test_mask]], window_start[test_targets]]
        ) if test_targets.size else np.zeros((0,), dtype=float)
        test_start = float(np.nanmin(test_interval_starts)) if test_interval_starts.size else float("nan")
        train_interval_ends = np.maximum(window_end[src], window_end[tgt])
        if np.isfinite(test_start):
            train_mask &= train_interval_ends <= test_start
    else:
        train_mask &= src < split - overlap_steps - horizon
    excluded_by_block = int(np.sum(valid & ~train_mask & ~test_mask))
    train_src = src[train_mask]
    test_src = src[test_mask]
    train_tgt = train_src + horizon
    test_tgt = test_src + horizon
    result: dict[str, Any] = {
        "schema_version": "mndm.family_b_blocked_one_step_audit.v1",
        "status": "insufficient_support",
        "qualification_status": "not_assessed",
        "claim_status": "no_biological_claim",
        "source_h5_sha256": source.sha256,
        "coordinate_layer": source.coordinate_layer,
        "coordinate_contract": source.coordinate_contract,
        "horizon_steps": horizon,
        "nominal_dt_sec": nominal_dt,
        "n_timepoints": n,
        "n_valid_pairs": int(np.sum(valid)),
        "n_train_pairs": int(train_src.size),
        "n_test_pairs": int(test_src.size),
        "split_source_index": split,
        "overlap_exclusion_steps": overlap_steps,
        "n_pairs_excluded_by_block_gap": excluded_by_block,
        "n_pairs_rejected_clock_gap": int(np.sum(~clock_ok)),
        "n_pairs_rejected_nonfinite": int(np.sum(clock_ok & ~finite)),
        "n_pairs_rejected_epoch_boundary": int(np.sum(clock_ok & finite & ~epoch_ok)),
        "n_pairs_rejected_source_file_boundary": int(np.sum(clock_ok & finite & ~file_ok)),
        "support_scope": "nominal_windows_only" if source.signal_support_status in (None, "unknown") else "source_recorded",
        "support_status": source.signal_support_status or "unknown",
        "support_reason": source.signal_support_reason or "raw temporal support was not certified",
        "design_rank": None,
        "design_full_rank": None,
        "finite_predictions": None,
        "models": {},
    }
    if source.window_start is None or source.window_end is None:
        result["support_scope"] = "nominal_bounds_unavailable"
        result["failure_reason"] = "nominal_bounds_unavailable"
        return result
    dim = int(x.shape[1])
    if train_src.size < max(dim + 2, 2) or test_src.size < 1:
        result["failure_reason"] = "blocked_holdout_insufficient_support"
        return result

    x_train = x[train_src]
    y_train = x[train_tgt]
    x_test = x[test_src]
    y_test = x[test_tgt]
    finite_train = np.isfinite(x_train).all(axis=1) & np.isfinite(y_train).all(axis=1)
    finite_test = np.isfinite(x_test).all(axis=1) & np.isfinite(y_test).all(axis=1)
    x_train, y_train = x_train[finite_train], y_train[finite_train]
    x_test, y_test = x_test[finite_test], y_test[finite_test]
    if x_train.shape[0] < max(dim + 2, 2) or x_test.shape[0] < 1:
        result["failure_reason"] = "blocked_holdout_nonfinite_support"
        return result

    centered = x_train - np.mean(x_train, axis=0, keepdims=True)
    design = np.column_stack([centered, np.ones((centered.shape[0], 1))])
    coefficients, _, _, _ = np.linalg.lstsq(design, y_train, rcond=None)
    design_rank = int(np.linalg.matrix_rank(design))
    result["design_rank"] = design_rank
    result["design_full_rank"] = bool(design_rank >= dim + 1)
    if not result["design_full_rank"]:
        result["failure_reason"] = "blocked_holdout_rank_deficient_design"
        return result
    phi = coefficients[:-1, :]
    intercept = coefficients[-1, :]
    pred_phi = (x_test - np.mean(x_train, axis=0, keepdims=True)) @ phi + intercept
    train_target_mean = np.mean(y_train, axis=0, keepdims=True)
    pred_mean = np.repeat(train_target_mean, x_test.shape[0], axis=0)
    pred_persistence = x_test
    result["finite_predictions"] = bool(
        np.isfinite(pred_phi).all()
        and np.isfinite(pred_mean).all()
        and np.isfinite(pred_persistence).all()
    )
    if not result["finite_predictions"]:
        result["failure_reason"] = "blocked_holdout_nonfinite_predictions"
        return result
    mean_test = np.mean(y_test, axis=0, keepdims=True)
    mean_baseline_mse = _mse(np.repeat(mean_test, y_test.shape[0], axis=0), y_test)
    phi_mse = _mse(pred_phi, y_test)
    mean_mse = _mse(pred_mean, y_test)
    persistence_mse = _mse(pred_persistence, y_test)
    if not np.isfinite([phi_mse, mean_mse, persistence_mse, mean_baseline_mse]).all():
        result["failure_reason"] = "blocked_holdout_nonfinite_scores"
        return result
    result["status"] = "computed"
    result["models"] = {
        "phi": {
            "mse": phi_mse,
            "relative_to_test_mean": _relative(phi_mse, mean_baseline_mse),
        },
        "train_target_mean": {
            "mse": mean_mse,
            "relative_to_test_mean": _relative(mean_mse, mean_baseline_mse),
        },
        "persistence": {
            "mse": persistence_mse,
            "relative_to_test_mean": _relative(persistence_mse, mean_baseline_mse),
        },
    }
    return result


def _write_export_h5(
    source: FamilyBSource,
    output_path: Path,
    metrics: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> None:
    """Write the bounded subset through the established HDF5 writer."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    attrs = {
        "family_b_export_schema_version": FAMILY_B_EXPORT_SCHEMA_VERSION,
        "family_b_fidelity_threshold": FAMILY_B_FIDELITY_THRESHOLD,
        "source_h5_sha256": source.sha256,
        "source_dataset_id": source.dataset_id,
        "source_jacobian_path": f"/{source.jacobian_group}/J_hat",
        "primary_coordinate_layer": source.coordinate_layer,
        "primary_coordinate_contract": source.coordinate_contract,
        "source_metric_status": source.metric_status,
        "source_metric_failure_reason": source.metric_failure_reason,
        "audit_schema_version": audit.get("schema_version"),
    }
    # Preserve source identity and the measurement-boundary fields needed to
    # interpret this subset.  Prefixing avoids accidentally changing the
    # output file's own canonical attrs.
    for key, value in source.source_attrs.items():
        if key == "dataset_id":
            continue
        attrs[f"source_{key}"] = value
    if "mndm_version" in source.source_attrs:
        # This subset carries the source's canonical measurements unchanged;
        # retain the source writer version instead of letting the compatibility
        # writer default silently to its own older version.
        attrs["mndm_version"] = source.source_attrs["mndm_version"]
    provenance = {
        "family_b_export": {
            "schema_version": FAMILY_B_EXPORT_SCHEMA_VERSION,
            "source_h5_sha256": source.sha256,
            "source_path_basename": source.path.name,
            "source_dataset_id": source.dataset_id,
            "source_jacobian_path": f"/{source.jacobian_group}/J_hat",
            "source_metrics_path": f"/{source.jacobian_group}/derived_metrics/v1",
            "source_metric_status": source.metric_status,
            "source_metric_failure_reason": source.metric_failure_reason,
            "fidelity_threshold_fixed": FAMILY_B_FIDELITY_THRESHOLD,
            "audit_json_schema_version": audit.get("schema_version"),
            "audit_support_scope": audit.get("support_scope"),
            "source_attrs": source.source_attrs,
            "source_signal_support_metadata": source.signal_support_metadata,
        },
        "signal_support_provenance": {
            "temporal_support_status": source.signal_support_status or "unknown",
            "temporal_support_reason": source.signal_support_reason or "raw temporal support was not certified",
        },
    }
    layer = {
        "values": source.coordinate_values,
        "names": source.coordinate_names,
        "attrs": source.coordinate_attrs,
    }
    payload_kwargs: dict[str, Any] = {
        "time": source.time,
        "x": source.mnps,
        "x_dot": source.mnps_dot,
        "window_start": source.window_start,
        "window_end": source.window_end,
        "epoch_id": source.epoch_id,
        "coordinate_layers": {source.coordinate_layer: layer},
        "attrs": attrs,
        "provenance": provenance,
    }
    if source.jacobian_group == "jacobian":
        payload_kwargs.update(
            {
                "jacobian": source.jacobian,
                "jacobian_dot": source.jacobian_dot,
                "jacobian_centers": source.centers,
                "jacobian_affine_reference": source.affine_reference,
                "jacobian_affine_intercept": source.affine_intercept,
                "jacobian_derived_metrics": dict(metrics),
            }
        )
    else:
        payload_kwargs.update(
            {
                "jacobian_9D": source.jacobian,
                "jacobian_9D_dot": source.jacobian_dot,
                "jacobian_9D_centers": source.centers,
                "jacobian_9D_affine_reference": source.affine_reference,
                "jacobian_9D_affine_intercept": source.affine_intercept,
                "jacobian_9D_derived_metrics": dict(metrics),
            }
        )
    payload = MNPSPayload(**payload_kwargs)
    manifest = {
        "export_schema_version": FAMILY_B_EXPORT_SCHEMA_VERSION,
        "dataset_id": source.dataset_id,
        "source_h5_sha256": source.sha256,
        "source_jacobian_path": f"/{source.jacobian_group}/J_hat",
        "coordinate_layer": source.coordinate_layer,
        "coordinate_contract": source.coordinate_contract,
        "computation_status": metrics.get("computation_status"),
        "failure_reason": metrics.get("failure_reason"),
        "audit": dict(audit),
    }
    # The existing writer's diagnostic slot is attached to the primary group.
    # For a 9D export, relocate that slot under the matching 9D Jacobian group
    # after writing.  This keeps diagnostics beside the J_hat they describe;
    # the canonical 9D metric path remains authoritative.
    h5_writer.write_h5(
        output_path,
        source.dataset_id,
        payload,
        manifest=manifest,
        jacobian_diagnostics=source.diagnostics,
    )
    if source.jacobian_group == "jacobian_9D":
        with h5py.File(output_path, "a") as h5:
            if "jacobian/diagnostics" in h5:
                h5.move("jacobian/diagnostics", "jacobian_9D/diagnostics")
                if "jacobian" in h5 and not h5["jacobian"]:
                    del h5["jacobian"]


def build_family_b_export(
    source_path: str | Path,
    output_path: str | Path,
    *,
    audit_json_path: str | Path | None = None,
    coordinate_layer: str | None = None,
    horizon: int = DEFAULT_HORIZON,
) -> dict[str, Any]:
    """Build a generic Family B subset export and an audit-only JSON sidecar."""
    source_candidate = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    audit_path = (
        Path(audit_json_path).expanduser().resolve()
        if audit_json_path is not None
        else output.with_name(f"{output.stem}_audit.json")
    )
    if output == source_candidate:
        raise ValueError("output_h5 must be separate from source_h5")
    if audit_path in {source_candidate, output}:
        raise ValueError("audit_json must be separate from source_h5 and output_h5")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    if audit_path.exists():
        raise FileExistsError(f"refusing to overwrite audit JSON: {audit_path}")
    source = load_family_b_source(source_candidate, coordinate_layer=coordinate_layer)
    metrics = _recompute_metrics(source)
    audit = blocked_one_step_audit(source, horizon=horizon)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(_json_safe(audit), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_export_h5(source, output, metrics, audit)
    return {
        "schema_version": FAMILY_B_EXPORT_SCHEMA_VERSION,
        "source_h5": str(source.path),
        "source_h5_sha256": source.sha256,
        "output_h5": str(output),
        "audit_json": str(audit_path),
        "coordinate_layer": source.coordinate_layer,
        "coordinate_contract": source.coordinate_contract,
        "jacobian_group": source.jacobian_group,
        "computation_status": metrics.get("computation_status"),
        "failure_reason": metrics.get("failure_reason"),
        "audit_status": audit.get("status"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_h5", type=Path)
    parser.add_argument("output_h5", type=Path)
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument("--coordinate-layer", default=None)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    args = parser.parse_args(argv)
    result = build_family_b_export(
        args.source_h5,
        args.output_h5,
        audit_json_path=args.audit_json,
        coordinate_layer=args.coordinate_layer,
        horizon=args.horizon,
    )
    print(json.dumps(_json_safe(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
