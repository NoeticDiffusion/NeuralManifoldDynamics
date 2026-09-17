"""Realized successive-increment turning (SL-LEV-MES-001 rotation L0).

``turning_angle_level0`` is the Euclidean angle between consecutive
lag-1 displacements. ``turning_rate_level0`` is that angle divided by
the observed first-step ``dt``. Insufficient displacement is undefined
(NaN), not zero. This is not ``operator_rotation_rate_level2`` and not
``generator_rotation_norm_level3``. Cloud-volume expansion remains
withheld.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import TURNING_SCHEMA_VERSION, build_provenance, unavailable_result
from .measurement_register import (
    MEASUREMENT_ID_TURNING_ANGLE,
    MEASUREMENT_ID_TURNING_RATE,
    QUALIFICATION_TURNING,
    stamp_register_fields,
)
from .transition_support import (
    build_transition_support,
    support_series_fields,
    support_settings_fields,
    support_summary_fields,
)
from .validity import validate_trajectory
from ..inferential_grain import attach_grain_for_schema
from ..measurement_certificate import attach_certificate

ESTIMATOR_NAME = "successive_increment_turning"
DEFAULT_MIN_DISPLACEMENT = 1e-12
REASON_SUBJECT_ANCHORED_3D = "turning_subject_anchored_3d_only"
REASON_MIN_DISPLACEMENT = "turning_min_displacement_invalid"
REASON_PAIRS = "insufficient_turning_pairs"
REASON_DIRECTION = "turning_direction_undefined"
_SUBJECT_ANCHORED_LAYERS = frozenset({"coords_3d_subject_anchored", "subject_anchored"})


def _finalize(result: Mapping[str, Any]) -> dict[str, Any]:
    return attach_grain_for_schema(attach_certificate(result), TURNING_SCHEMA_VERSION)


def _stamp_qualification(result: dict[str, Any], token: str) -> dict[str, Any]:
    out = dict(result)
    out["qualification_status"] = token
    summary = dict(out.get("summary") or {})
    summary["qualification_status"] = token
    out["summary"] = summary
    return out


def _unavailable(
    *,
    status: str,
    failure_reason: str,
    coordinate_layer: str,
    coordinate_names: list[str],
    extra_summary: Mapping[str, Any] | None = None,
    extra_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = unavailable_result(
        TURNING_SCHEMA_VERSION,
        status=status,
        failure_reason=failure_reason,
        coordinate_layer=coordinate_layer,
        coordinate_names=coordinate_names,
    )
    summary = dict(result.get("summary") or {})
    if extra_summary:
        summary.update(dict(extra_summary))
    result["summary"] = summary
    provenance = dict(result.get("provenance") or {})
    settings = dict(provenance.get("settings") or {})
    settings["estimator"] = ESTIMATOR_NAME
    settings["not_operator_rotation"] = True
    settings["not_generator_rotation"] = True
    settings["undefined_not_zero"] = True
    if extra_settings:
        settings.update(dict(extra_settings))
    provenance["settings"] = settings
    provenance["estimator"] = ESTIMATOR_NAME
    result["provenance"] = provenance
    parent = stamp_register_fields(result, MEASUREMENT_ID_TURNING_RATE)
    parent = _stamp_qualification(parent, QUALIFICATION_TURNING)
    parent = _finalize(parent)
    rate_leaf = dict(parent)
    angle_leaf = stamp_register_fields(dict(parent), MEASUREMENT_ID_TURNING_ANGLE)
    angle_leaf = _stamp_qualification(angle_leaf, QUALIFICATION_TURNING)
    angle_leaf = _finalize(angle_leaf)
    parent[MEASUREMENT_ID_TURNING_RATE] = rate_leaf
    parent[MEASUREMENT_ID_TURNING_ANGLE] = angle_leaf
    return parent


def unavailable_turning_family(
    *,
    status: str,
    failure_reason: str,
    coordinate_layer: str,
    coordinate_names: list[str],
    extra_summary: Mapping[str, Any] | None = None,
    extra_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Schema-complete unavailable turning payload with both L0 leaves."""
    return _unavailable(
        status=status,
        failure_reason=failure_reason,
        coordinate_layer=coordinate_layer,
        coordinate_names=coordinate_names,
        extra_summary=extra_summary,
        extra_settings=extra_settings,
    )


def _angle_between(v0: np.ndarray, v1: np.ndarray, min_displacement: float) -> float:
    n0 = float(np.linalg.norm(v0))
    n1 = float(np.linalg.norm(v1))
    if n0 < min_displacement or n1 < min_displacement:
        return float("nan")
    cosine = float(np.dot(v0, v1) / (n0 * n1))
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def estimate_turning_rate(
    state: np.ndarray,
    time: np.ndarray,
    *,
    segment_id: np.ndarray | None = None,
    coordinate_layer: str = "coords_3d_subject_anchored",
    coordinate_names: Sequence[str] | None = None,
    min_samples: int = 30,
    min_pairs: int = 20,
    max_gap_sec: float | None = None,
    max_dt_relative_deviation: float = 0.05,
    min_displacement: float = DEFAULT_MIN_DISPLACEMENT,
) -> dict[str, Any]:
    """Window-level realized turning from consecutive lag-1 displacements."""
    names = list(coordinate_names or ["m", "d", "e"])
    layer = str(coordinate_layer or "").strip()
    x_in = np.asarray(state)
    if layer not in _SUBJECT_ANCHORED_LAYERS or x_in.ndim != 2 or x_in.shape[1] != 3:
        return _unavailable(
            status="not_testable",
            failure_reason=REASON_SUBJECT_ANCHORED_3D,
            coordinate_layer=layer or "unknown",
            coordinate_names=names,
        )
    if not np.isfinite(min_displacement) or float(min_displacement) <= 0.0:
        return _unavailable(
            status="not_testable",
            failure_reason=REASON_MIN_DISPLACEMENT,
            coordinate_layer=layer,
            coordinate_names=names,
            extra_summary={"requested_min_displacement": float(min_displacement)},
        )
    floor = float(min_displacement)
    x, t, segments, _finite, reason = validate_trajectory(
        state, time, min_samples=min_samples, segment_id=segment_id
    )
    if reason is not None or x is None or t is None or segments is None:
        return _unavailable(
            status="insufficient_support" if reason == "insufficient_samples" else "invalid",
            failure_reason=str(reason or "invalid_trajectory"),
            coordinate_layer=layer,
            coordinate_names=names,
        )
    support = build_transition_support(
        x,
        t,
        segments,
        lag=1,
        max_gap_sec=max_gap_sec,
        max_dt_relative_deviation=float(max_dt_relative_deviation),
    )
    shared_settings = {
        "estimator": ESTIMATOR_NAME,
        "min_pairs": int(min_pairs),
        "min_displacement": floor,
        "undefined_not_zero": True,
        "not_operator_rotation": True,
        "not_generator_rotation": True,
        "not_cloud_volume": True,
        **support_settings_fields(support),
    }
    if support.failure_reason == "non_positive_nominal_dt":
        return _unavailable(
            status="invalid",
            failure_reason="non_positive_nominal_dt",
            coordinate_layer=layer,
            coordinate_names=names,
            extra_settings=shared_settings,
        )
    if support.failure_reason == "materially_irregular_increment_timestep":
        return _unavailable(
            status="not_testable",
            failure_reason="materially_irregular_increment_timestep",
            coordinate_layer=layer,
            coordinate_names=names,
            extra_settings=shared_settings,
        )
    source_idx = np.asarray(support.source_idx, dtype=np.int32)
    increments = np.asarray(support.increments, dtype=float)
    dts = np.asarray(support.dts, dtype=float)
    index_by_source = {int(src): int(i) for i, src in enumerate(source_idx.tolist())}
    keep = np.array([int(src) + 1 in index_by_source for src in source_idx], dtype=bool)
    if int(np.sum(keep)) < int(min_pairs):
        return _unavailable(
            status="insufficient_support",
            failure_reason=REASON_PAIRS,
            coordinate_layer=layer,
            coordinate_names=names,
            extra_settings=shared_settings,
            extra_summary={"n_turning_pairs": int(np.sum(keep)), "n_lag1_pairs": int(source_idx.size)},
        )
    pair_support = support.subset(keep)
    pair_source = np.asarray(pair_support.source_idx, dtype=np.int32)
    shared_settings = {
        **shared_settings,
        **support_settings_fields(pair_support),
        "lag1_transition_support_id": support.support_id,
        "n_lag1_pairs": int(source_idx.size),
        "n_turning_pairs": int(pair_source.size),
    }
    n_time = int(x.shape[0])
    angle = np.full(n_time, np.nan, dtype=np.float32)
    rate = np.full(n_time, np.nan, dtype=np.float32)
    n_defined = 0
    n_undefined = 0
    for src in pair_source.tolist():
        i0 = index_by_source[int(src)]
        i1 = index_by_source[int(src) + 1]
        theta = _angle_between(increments[i0], increments[i1], floor)
        dt = float(dts[i0])
        if not np.isfinite(theta) or not np.isfinite(dt) or dt <= 0.0:
            n_undefined += 1
            continue
        n_defined += 1
        angle[int(src)] = np.float32(theta)
        rate[int(src)] = np.float32(theta / dt)
    pair_summary = {
        "n_lag1_pairs": int(source_idx.size),
        "n_turning_pairs": int(pair_source.size),
        "n_defined_turning": n_defined,
        "n_undefined_turning": n_undefined,
    }
    if n_defined < int(min_pairs) // 2:
        return _unavailable(
            status="insufficient_support",
            failure_reason=REASON_DIRECTION,
            coordinate_layer=layer,
            coordinate_names=names,
            extra_settings=shared_settings,
            extra_summary=pair_summary,
        )
    finite_rate = rate[np.isfinite(rate)]
    summary = {
        **support_summary_fields(pair_support),
        **pair_summary,
        "lag1_transition_support_id": support.support_id,
        "turning_rate_median": float(np.median(finite_rate)),
        "turning_angle_median": float(np.median(angle[np.isfinite(angle)])),
        "min_displacement": floor,
        "undefined_not_zero": True,
        "not_operator_rotation": True,
        "not_generator_rotation": True,
        "declared_lag_steps": 1,
    }
    series = {
        **support_series_fields(pair_support),
        "turning_rate": rate,
        "turning_angle": angle,
    }
    result = {
        "schema_version": TURNING_SCHEMA_VERSION,
        "computation_status": "computed",
        "failure_reason": None,
        "series": series,
        "summary": summary,
        "provenance": build_provenance(
            coordinate_layer=layer,
            coordinate_names=names,
            time_semantics="within_segment_successive_increment_turning",
            estimator=ESTIMATOR_NAME,
            settings=shared_settings,
        ),
    }
    rate_leaf = stamp_register_fields(result, MEASUREMENT_ID_TURNING_RATE)
    rate_leaf = _stamp_qualification(rate_leaf, QUALIFICATION_TURNING)
    rate_leaf = _finalize(rate_leaf)
    angle_leaf = stamp_register_fields(dict(rate_leaf), MEASUREMENT_ID_TURNING_ANGLE)
    angle_leaf = _stamp_qualification(angle_leaf, QUALIFICATION_TURNING)
    angle_leaf = _finalize(angle_leaf)
    parent = dict(rate_leaf)
    parent[MEASUREMENT_ID_TURNING_RATE] = rate_leaf
    parent[MEASUREMENT_ID_TURNING_ANGLE] = angle_leaf
    return _finalize(parent)
