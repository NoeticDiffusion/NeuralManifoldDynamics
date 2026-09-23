"""Chart-drift measurements under ``mndm.chart_drift.v1`` (SL-LEV-MES-003).

Written nested names are measurement identities, not the older parallel
Level 0–3 evidence classes. Cross-fit is a level-1 variant. Multi-lag
consistency is diagnostics, not ``ito_drift_level3``. ``/mnps_3d_dot`` is
the Savitzky–Golay sibling and is not consumed here. None of these fields
is a ``drift_source`` for diffusion ``A_bD``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .contracts import CHART_DRIFT_SCHEMA_VERSION, build_provenance, unavailable_result
from .measurement_register import (
    ESTIMAND_MEAN_INCREMENT_NOMINAL_DT,
    ESTIMAND_PER_STEP_DX_DT,
    MEASUREMENT_ID_CONDITIONAL_MEAN_RATE,
    MEASUREMENT_ID_REALIZED_VELOCITY,
    QUALIFICATION_ITO_NOT_QUALIFIED,
    VARIANT_BLOCKED_CROSSFIT,
    VARIANT_LAG_DIAGNOSTICS,
    VARIANT_POOLED,
    stamp_register_fields,
)
from ..measurement_certificate import attach_certificate
from ..inferential_grain import attach_grain_for_schema
from .transition_support import (
    TransitionSupport,
    embargo_claim_fields,
    build_transition_support,
    support_series_fields,
    support_settings_fields,
    support_summary_fields,
)
from .validity import (
    chunked_nearest_neighbors,
    increment_pairs_at_lag,
    validate_trajectory,
)

ESTIMATOR_NAME = "local_conditional_increment_mean"
WEIGHT_INVERSE_DISTANCE = "inverse_distance"
WEIGHT_UNIFORM = "uniform"
ALLOWED_WEIGHT_MODES = frozenset({WEIGHT_INVERSE_DISTANCE, WEIGHT_UNIFORM})
REASON_INSUFFICIENT_LOCAL = "insufficient_local_support"
REASON_LAG_INCONSISTENT = "lag_inconsistent"
REASON_SUBJECT_ANCHORED_3D = "chart_drift_subject_anchored_3d_only"
MULTI_LAG_CONSISTENT = "multi_lag_consistent"
DEFAULT_ITO_LAGS = (1, 2, 4)
_DISTANCE_FLOOR = 1e-12
_SUBJECT_ANCHORED_LAYERS = frozenset(
    {"coords_3d_subject_anchored", "subject_anchored"}
)


def _finalize(result: Mapping[str, Any]) -> dict[str, Any]:
    return attach_grain_for_schema(attach_certificate(result))


def _unavailable(
    *,
    status: str,
    failure_reason: str,
    coordinate_layer: str,
    coordinate_names: list[str],
    measurement_id: str | None = None,
    variant_id: str | None = None,
    extra_summary: Mapping[str, Any] | None = None,
    extra_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = unavailable_result(
        CHART_DRIFT_SCHEMA_VERSION,
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
    settings["not_sde_drift"] = True
    settings["estimator"] = ESTIMATOR_NAME
    if extra_settings:
        settings.update(dict(extra_settings))
    provenance["settings"] = settings
    provenance["estimator"] = ESTIMATOR_NAME
    result["provenance"] = provenance
    if measurement_id is not None:
        result = stamp_register_fields(result, measurement_id, variant_id=variant_id)
    return result


def _layer_refusal(
    *,
    state: np.ndarray,
    coordinate_layer: str,
    coordinate_names: list[str] | None,
) -> dict[str, Any] | None:
    names = coordinate_names or [
        f"dim_{idx}"
        for idx in range(np.asarray(state).shape[1] if np.asarray(state).ndim == 2 else 0)
    ]
    x = np.asarray(state)
    layer = str(coordinate_layer or "").strip()
    if layer not in _SUBJECT_ANCHORED_LAYERS:
        return _unavailable(
            status="not_testable",
            failure_reason=REASON_SUBJECT_ANCHORED_3D,
            coordinate_layer=layer or "unknown",
            coordinate_names=names,
        )
    if x.ndim != 2 or x.shape[1] != 3:
        return _unavailable(
            status="not_testable",
            failure_reason=REASON_SUBJECT_ANCHORED_3D,
            coordinate_layer=layer,
            coordinate_names=names,
        )
    if len(names) != 3:
        return _unavailable(
            status="invalid",
            failure_reason="coordinate_name_dimension_mismatch",
            coordinate_layer=layer,
            coordinate_names=names,
        )
    return None


def _weight_vector(
    squared_distances: np.ndarray,
    *,
    weight_mode: str,
    epsilon: float,
) -> np.ndarray:
    n_neighbors = int(squared_distances.size)
    if n_neighbors == 0:
        return np.zeros(0, dtype=float)
    if weight_mode == WEIGHT_UNIFORM:
        return np.ones(n_neighbors, dtype=float)
    distances = np.sqrt(np.maximum(np.asarray(squared_distances, dtype=float), 0.0))
    positive = distances[distances > 0.0]
    floor = float(np.median(positive)) * 1e-3 if positive.size else _DISTANCE_FLOOR
    floor = max(floor, _DISTANCE_FLOOR)
    weights = 1.0 / (np.maximum(distances, floor) + float(epsilon))
    if not np.all(np.isfinite(weights)) or float(np.sum(weights)) <= 0.0:
        return np.ones(n_neighbors, dtype=float)
    return weights


def _effective_count(weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0
    sq = float(np.sum(np.asarray(weights, dtype=float) ** 2))
    if sq <= 0.0:
        return 0.0
    return float(total * total / sq)


def _prepare_trajectory(
    state: np.ndarray,
    time: np.ndarray,
    *,
    segment_id: np.ndarray | None,
    coordinate_layer: str,
    coordinate_names: list[str] | None,
    min_samples: int,
    neighborhood_k: int,
    min_neighborhood_samples: int,
    measurement_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], int, dict[str, Any] | None]:
    refusal = _layer_refusal(
        state=state,
        coordinate_layer=coordinate_layer,
        coordinate_names=coordinate_names,
    )
    names = coordinate_names or ["m", "d", "e"]
    if refusal is not None:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=bool), names, 0, stamp_register_fields(
            refusal, measurement_id
        )
    x, t, segments, finite_state, failure = validate_trajectory(
        state, time, min_samples=min_samples, segment_id=segment_id
    )
    if failure is not None or x is None or t is None or segments is None or finite_state is None:
        return x if x is not None else np.zeros((0, 3)), t if t is not None else np.zeros(0), segments if segments is not None else np.zeros(0, dtype=np.int32), finite_state if finite_state is not None else np.zeros(0, dtype=bool), names, 0, _unavailable(
            status="insufficient_support" if failure and "insufficient" in failure else "invalid",
            failure_reason=failure or "invalid_trajectory",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
            measurement_id=measurement_id,
        )
    minimum_support = max(int(min_neighborhood_samples), 3 * x.shape[1] + 1)
    if int(neighborhood_k) < minimum_support:
        return x, t, segments, finite_state, names, minimum_support, _unavailable(
            status="invalid",
            failure_reason="neighborhood_k_below_dimension_aware_minimum_support",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
            measurement_id=measurement_id,
        )
    return x, t, segments, finite_state, names, minimum_support, None


def _regular_increments(
    x: np.ndarray,
    t: np.ndarray,
    segments: np.ndarray,
    *,
    lag: int,
    max_gap_sec: float | None,
    max_dt_relative_deviation: float,
    min_neighborhood_samples: int,
    neighborhood_k: int,
    coordinate_layer: str,
    coordinate_names: list[str],
    measurement_id: str,
    variant_id: str | None = None,
) -> tuple[TransitionSupport, dict[str, Any] | None]:
    support = build_transition_support(
        x,
        t,
        segments,
        lag=int(lag),
        max_gap_sec=max_gap_sec,
        max_dt_relative_deviation=float(max_dt_relative_deviation),
    )
    n_pairs = int(support.source_idx.size)
    if n_pairs < int(min_neighborhood_samples):
        return support, _unavailable(
            status="insufficient_support",
            failure_reason=REASON_INSUFFICIENT_LOCAL,
            coordinate_layer=coordinate_layer,
            coordinate_names=coordinate_names,
            measurement_id=measurement_id,
            variant_id=variant_id,
            extra_summary={"n_increment_pairs": n_pairs, "lag": int(lag)},
        )
    if int(neighborhood_k) >= n_pairs:
        return support, _unavailable(
            status="insufficient_support",
            failure_reason=REASON_INSUFFICIENT_LOCAL,
            coordinate_layer=coordinate_layer,
            coordinate_names=coordinate_names,
            measurement_id=measurement_id,
            variant_id=variant_id,
            extra_summary={
                "n_increment_pairs": n_pairs,
                "lag": int(lag),
                "failure_detail": "neighborhood_k_not_strictly_local",
            },
        )
    if support.failure_reason == "non_positive_nominal_dt":
        return support, _unavailable(
            status="invalid",
            failure_reason="non_positive_nominal_dt",
            coordinate_layer=coordinate_layer,
            coordinate_names=coordinate_names,
            measurement_id=measurement_id,
            variant_id=variant_id,
        )
    if support.failure_reason == "materially_irregular_increment_timestep":
        return support, _unavailable(
            status="not_testable",
            failure_reason="materially_irregular_increment_timestep",
            coordinate_layer=coordinate_layer,
            coordinate_names=coordinate_names,
            measurement_id=measurement_id,
            variant_id=variant_id,
        )
    return support, None


def _fill_conditional_mean(
    x: np.ndarray,
    t: np.ndarray,
    finite_idx: np.ndarray,
    *,
    ref_source_idx: np.ndarray,
    ref_increments: np.ndarray,
    neighborhood_k: int,
    minimum_support: int,
    max_neighborhood_radius: float | None,
    min_temporal_span_sec: float | None,
    weight_mode: str,
    nominal_dt: float,
    epsilon: float,
    store_neighbor_source_idx: bool,
) -> dict[str, np.ndarray]:
    n_time, dimension = x.shape
    b_hat = np.full((n_time, dimension), np.nan, dtype=np.float32)
    n_effective = np.zeros(n_time, dtype=np.float32)
    radius = np.full(n_time, np.nan, dtype=np.float32)
    valid = np.zeros(n_time, dtype=np.int8)
    se_hat = np.full((n_time, dimension), np.nan, dtype=np.float32)
    neighbor_source_idx = None
    if store_neighbor_source_idx:
        neighbor_source_idx = np.full((n_time, int(neighborhood_k)), -1, dtype=np.int32)
    n_refs = int(ref_source_idx.size)
    if n_refs < int(minimum_support) or int(neighborhood_k) >= n_refs or finite_idx.size == 0:
        return {
            "b_hat": b_hat,
            "n_effective": n_effective,
            "radius": radius,
            "valid": valid,
            "se_hat": se_hat,
            "neighbor_source_idx": neighbor_source_idx,
        }
    k_use = min(int(neighborhood_k), n_refs)
    increment_state = x[ref_source_idx]
    nearest_indices, nearest_distances = chunked_nearest_neighbors(
        x[finite_idx],
        increment_state,
        k_use,
    )
    ref_times = t[ref_source_idx]
    for row, center in enumerate(finite_idx):
        nearest = np.asarray(nearest_indices[row], dtype=np.intp)
        distances = np.asarray(nearest_distances[row], dtype=float)
        source_ids = ref_source_idx[nearest]
        keep = source_ids != int(center)
        nearest = nearest[keep]
        distances = distances[keep]
        source_ids = source_ids[keep]
        if nearest.size < int(minimum_support):
            continue
        if max_neighborhood_radius is not None and float(np.sqrt(np.max(distances))) > float(
            max_neighborhood_radius
        ):
            continue
        local_dx = ref_increments[nearest]
        finite_rows = np.all(np.isfinite(local_dx), axis=1)
        nearest = nearest[finite_rows]
        distances = distances[finite_rows]
        source_ids = source_ids[finite_rows]
        local_dx = local_dx[finite_rows]
        if local_dx.shape[0] < int(minimum_support):
            continue
        neighbor_times = ref_times[nearest]
        if int(np.unique(source_ids).size) < int(minimum_support):
            continue
        if min_temporal_span_sec is not None:
            span = float(np.max(neighbor_times) - np.min(neighbor_times))
            if not np.isfinite(span) or span < float(min_temporal_span_sec):
                continue
        weights = _weight_vector(distances, weight_mode=weight_mode, epsilon=epsilon)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            continue
        n_eff = _effective_count(weights)
        if n_eff < float(minimum_support):
            continue
        mean_dx = (weights[:, None] * local_dx).sum(axis=0) / weight_sum
        b_hat[center] = (mean_dx / float(nominal_dt)).astype(np.float32)
        n_effective[center] = np.float32(n_eff)
        radius[center] = np.float32(np.sqrt(np.max(distances)))
        std_dx = np.std(local_dx, axis=0, ddof=1) if local_dx.shape[0] > 1 else np.full(dimension, np.nan)
        se_hat[center] = (std_dx / (np.sqrt(n_eff) * float(nominal_dt))).astype(np.float32)
        valid[center] = 1
        if neighbor_source_idx is not None:
            n_keep = min(int(source_ids.size), neighbor_source_idx.shape[1])
            neighbor_source_idx[center, :n_keep] = source_ids[:n_keep]
    return {
        "b_hat": b_hat,
        "n_effective": n_effective,
        "radius": radius,
        "valid": valid,
        "se_hat": se_hat,
        "neighbor_source_idx": neighbor_source_idx,
    }


def _computed_result(
    *,
    measurement_id: str,
    variant_id: str | None,
    coordinate_layer: str,
    coordinate_names: list[str],
    series: dict[str, Any],
    summary: dict[str, Any],
    settings: dict[str, Any],
    failure_reason: str | None = None,
    computation_status: str = "computed",
    time_semantics: str,
    estimator: str = ESTIMATOR_NAME,
) -> dict[str, Any]:
    if computation_status == "ito_qualified":
        raise ValueError("ito_qualified is not a legal computation_status")
    payload = {
        "schema_version": CHART_DRIFT_SCHEMA_VERSION,
        "computation_status": computation_status,
        "failure_reason": failure_reason,
        "series": series,
        "summary": dict(summary),
        "provenance": build_provenance(
            coordinate_layer=coordinate_layer,
            coordinate_names=coordinate_names,
            time_semantics=time_semantics,
            estimator=estimator,
            settings={
                "not_sde_drift": True,
                **settings,
            },
        ),
    }
    return _finalize(stamp_register_fields(payload, measurement_id, variant_id=variant_id))


def estimate_realized_velocity_level0(
    state: np.ndarray,
    time: np.ndarray,
    *,
    segment_id: np.ndarray | None = None,
    coordinate_layer: str = "coords_3d_subject_anchored",
    coordinate_names: list[str] | None = None,
    min_samples: int = 30,
    max_gap_sec: float | None = None,
) -> dict[str, Any]:
    """Per-step forward difference ``(x_{t+1}-x_t)/Δt``. Not ``/mnps_3d_dot``."""
    names = list(coordinate_names or ["m", "d", "e"])
    refusal = _layer_refusal(
        state=state,
        coordinate_layer=coordinate_layer,
        coordinate_names=names,
    )
    if refusal is not None:
        return stamp_register_fields(refusal, MEASUREMENT_ID_REALIZED_VELOCITY)
    x, t, segments, finite_state, failure = validate_trajectory(
        state, time, min_samples=min_samples, segment_id=segment_id
    )
    if failure is not None or x is None or t is None or segments is None:
        return _unavailable(
            status="insufficient_support" if failure and "insufficient" in failure else "invalid",
            failure_reason=failure or "invalid_trajectory",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
            measurement_id=MEASUREMENT_ID_REALIZED_VELOCITY,
        )
    source_idx, increments, dts = increment_pairs_at_lag(
        x, t, segments, lag=1, max_gap_sec=max_gap_sec
    )
    n_time, dimension = x.shape
    dx_dt = np.full((n_time, dimension), np.nan, dtype=np.float32)
    observed_dt = np.full(n_time, np.nan, dtype=np.float32)
    valid = np.zeros(n_time, dtype=np.int8)
    for row, source in enumerate(source_idx):
        step = float(dts[row])
        if not np.isfinite(step) or step <= 0.0:
            continue
        dx_dt[int(source)] = (increments[row] / step).astype(np.float32)
        observed_dt[int(source)] = np.float32(step)
        valid[int(source)] = 1
    if int(np.sum(valid)) < 1:
        return _unavailable(
            status="insufficient_support",
            failure_reason=REASON_INSUFFICIENT_LOCAL,
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
            measurement_id=MEASUREMENT_ID_REALIZED_VELOCITY,
            extra_summary={"n_increment_pairs": int(increments.shape[0])},
            extra_settings={"not_alias_of_mnps_3d_dot": True},
        )
    return _computed_result(
        measurement_id=MEASUREMENT_ID_REALIZED_VELOCITY,
        variant_id=None,
        coordinate_layer=coordinate_layer,
        coordinate_names=names,
        series={
            "dx_dt": dx_dt,
            "observed_dt": observed_dt,
            "valid": valid,
        },
        summary={
            "n_timepoints": n_time,
            "n_valid_timepoints": int(np.sum(valid)),
            "n_increment_pairs": int(increments.shape[0]),
            "not_sde_drift": True,
            "not_alias_of_mnps_3d_dot": True,
        },
        settings={
            "min_samples": int(min_samples),
            "max_gap_sec": max_gap_sec,
            "not_alias_of_mnps_3d_dot": True,
            "lag": 1,
        },
        time_semantics="within_segment_forward_difference_over_observed_dt",
        estimator="within_segment_forward_difference",
    )


def estimate_finite_lag_drift(
    state: np.ndarray,
    time: np.ndarray,
    *,
    segment_id: np.ndarray | None = None,
    coordinate_layer: str = "coords_3d_subject_anchored",
    coordinate_names: list[str] | None = None,
    neighborhood_k: int = 20,
    min_samples: int = 30,
    min_neighborhood_samples: int = 10,
    max_gap_sec: float | None = None,
    max_dt_relative_deviation: float = 0.05,
    max_neighborhood_radius: float | None = None,
    min_temporal_span_sec: float | None = None,
    min_valid_fraction: float = 0.1,
    weight_mode: str = WEIGHT_INVERSE_DISTANCE,
    lag: int = 1,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Pooled level-1 conditional mean rate: mean increment over nominal ``dt``."""
    names = list(coordinate_names or ["m", "d", "e"])
    mode = str(weight_mode or WEIGHT_INVERSE_DISTANCE).strip() or WEIGHT_INVERSE_DISTANCE
    if mode not in ALLOWED_WEIGHT_MODES:
        return _unavailable(
            status="invalid",
            failure_reason="unsupported_weight_mode",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
            measurement_id=MEASUREMENT_ID_CONDITIONAL_MEAN_RATE,
            variant_id=VARIANT_POOLED,
        )
    x, t, segments, finite_state, names, minimum_support, failure = _prepare_trajectory(
        state,
        time,
        segment_id=segment_id,
        coordinate_layer=coordinate_layer,
        coordinate_names=names,
        min_samples=min_samples,
        neighborhood_k=neighborhood_k,
        min_neighborhood_samples=min_neighborhood_samples,
        measurement_id=MEASUREMENT_ID_CONDITIONAL_MEAN_RATE,
    )
    if failure is not None:
        return stamp_register_fields(
            failure, MEASUREMENT_ID_CONDITIONAL_MEAN_RATE, variant_id=VARIANT_POOLED
        )
    support, inc_failure = _regular_increments(
        x,
        t,
        segments,
        lag=lag,
        max_gap_sec=max_gap_sec,
        max_dt_relative_deviation=max_dt_relative_deviation,
        min_neighborhood_samples=minimum_support,
        neighborhood_k=neighborhood_k,
        coordinate_layer=coordinate_layer,
        coordinate_names=names,
        measurement_id=MEASUREMENT_ID_CONDITIONAL_MEAN_RATE,
        variant_id=VARIANT_POOLED,
    )
    if inc_failure is not None:
        return inc_failure
    source_idx = support.source_idx
    increments = support.increments
    nominal_dt = support.nominal_dt_sec
    relative_deviation = support.observed_dt_relative_deviation
    filled = _fill_conditional_mean(
        x,
        t,
        np.flatnonzero(finite_state),
        ref_source_idx=source_idx,
        ref_increments=increments,
        neighborhood_k=int(neighborhood_k),
        minimum_support=minimum_support,
        max_neighborhood_radius=max_neighborhood_radius,
        min_temporal_span_sec=min_temporal_span_sec,
        weight_mode=mode,
        nominal_dt=nominal_dt,
        epsilon=float(epsilon),
        store_neighbor_source_idx=False,
    )
    n_time = int(x.shape[0])
    if int(np.sum(filled["valid"])) < max(1, int(np.ceil(float(min_valid_fraction) * n_time))):
        return _unavailable(
            status="insufficient_support",
            failure_reason=REASON_INSUFFICIENT_LOCAL,
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
            measurement_id=MEASUREMENT_ID_CONDITIONAL_MEAN_RATE,
            variant_id=VARIANT_POOLED,
            extra_summary={
                "n_valid_timepoints": int(np.sum(filled["valid"])),
                "n_timepoints": n_time,
                "n_increment_pairs": int(increments.shape[0]),
            },
        )
    return _computed_result(
        measurement_id=MEASUREMENT_ID_CONDITIONAL_MEAN_RATE,
        variant_id=VARIANT_POOLED,
        coordinate_layer=coordinate_layer,
        coordinate_names=names,
        series={
            "b_hat": filled["b_hat"],
            "n_effective": filled["n_effective"],
            "radius": filled["radius"],
            "valid": filled["valid"],
            "se_hat": filled["se_hat"],
            **support_series_fields(support),
        },
        summary={
            "n_timepoints": n_time,
            "n_valid_timepoints": int(np.sum(filled["valid"])),
            "n_increment_pairs": int(increments.shape[0]),
            "nominal_dt_sec": float(nominal_dt),
            "lag": int(lag),
            "lag_sec": float(nominal_dt),
            "max_dt_relative_deviation": float(relative_deviation),
            "not_sde_drift": True,
            **support_summary_fields(support),
        },
        settings={
            "neighborhood_k": int(neighborhood_k),
            "min_samples": int(min_samples),
            "min_neighborhood_samples": int(min_neighborhood_samples),
            "dimension_aware_minimum_support": int(minimum_support),
            "max_gap_sec": max_gap_sec,
            "max_dt_relative_deviation": float(max_dt_relative_deviation),
            "max_neighborhood_radius": max_neighborhood_radius,
            "min_temporal_span_sec": min_temporal_span_sec,
            "min_valid_fraction": float(min_valid_fraction),
            "weight_mode": mode,
            "lag": int(lag),
            "lag_sec": float(nominal_dt),
            **support_settings_fields(support),
        },
        time_semantics="within_segment_finite_lag_conditional_increment_mean_divided_by_nominal_dt",
    )


def estimate_crossfit_drift(
    state: np.ndarray,
    time: np.ndarray,
    *,
    segment_id: np.ndarray | None = None,
    coordinate_layer: str = "coords_3d_subject_anchored",
    coordinate_names: list[str] | None = None,
    neighborhood_k: int = 20,
    min_samples: int = 30,
    min_neighborhood_samples: int = 10,
    max_gap_sec: float | None = None,
    max_dt_relative_deviation: float = 0.05,
    max_neighborhood_radius: float | None = None,
    min_temporal_span_sec: float | None = None,
    min_valid_fraction: float = 0.1,
    weight_mode: str = WEIGHT_INVERSE_DISTANCE,
    n_blocks: int = 2,
    embargo_steps: int = 4,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Blocked-crossfit variant of the same level-1 conditional mean. Not a level upgrade."""
    names = list(coordinate_names or ["m", "d", "e"])
    if int(n_blocks) != 2:
        return _unavailable(
            status="invalid",
            failure_reason="crossfit_requires_two_temporal_blocks",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
            measurement_id=MEASUREMENT_ID_CONDITIONAL_MEAN_RATE,
            variant_id=VARIANT_BLOCKED_CROSSFIT,
        )
    mode = str(weight_mode or WEIGHT_INVERSE_DISTANCE).strip() or WEIGHT_INVERSE_DISTANCE
    if mode not in ALLOWED_WEIGHT_MODES:
        return _unavailable(
            status="invalid",
            failure_reason="unsupported_weight_mode",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
            measurement_id=MEASUREMENT_ID_CONDITIONAL_MEAN_RATE,
            variant_id=VARIANT_BLOCKED_CROSSFIT,
        )
    x, t, segments, finite_state, names, minimum_support, failure = _prepare_trajectory(
        state,
        time,
        segment_id=segment_id,
        coordinate_layer=coordinate_layer,
        coordinate_names=names,
        min_samples=min_samples,
        neighborhood_k=neighborhood_k,
        min_neighborhood_samples=min_neighborhood_samples,
        measurement_id=MEASUREMENT_ID_CONDITIONAL_MEAN_RATE,
    )
    if failure is not None:
        return stamp_register_fields(
            failure, MEASUREMENT_ID_CONDITIONAL_MEAN_RATE, variant_id=VARIANT_BLOCKED_CROSSFIT
        )
    support, inc_failure = _regular_increments(
        x,
        t,
        segments,
        lag=1,
        max_gap_sec=max_gap_sec,
        max_dt_relative_deviation=max_dt_relative_deviation,
        min_neighborhood_samples=minimum_support,
        neighborhood_k=neighborhood_k,
        coordinate_layer=coordinate_layer,
        coordinate_names=names,
        measurement_id=MEASUREMENT_ID_CONDITIONAL_MEAN_RATE,
        variant_id=VARIANT_BLOCKED_CROSSFIT,
    )
    if inc_failure is not None:
        return inc_failure
    source_idx = support.source_idx
    increments = support.increments
    nominal_dt = support.nominal_dt_sec
    relative_deviation = support.observed_dt_relative_deviation
    split_index = int(np.median(source_idx))
    embargo = int(embargo_steps)
    fold1_mask = source_idx < (split_index - embargo)
    fold2_mask = source_idx > (split_index + embargo)
    fold1_idx = source_idx[fold1_mask]
    fold2_idx = source_idx[fold2_mask]
    fold1_dx = increments[fold1_mask]
    fold2_dx = increments[fold2_mask]
    crossfit_support = support.subset(fold1_mask | fold2_mask)
    if (
        fold1_idx.size < int(minimum_support)
        or fold2_idx.size < int(minimum_support)
        or int(neighborhood_k) >= int(fold1_idx.size)
        or int(neighborhood_k) >= int(fold2_idx.size)
    ):
        return _unavailable(
            status="insufficient_support",
            failure_reason=REASON_INSUFFICIENT_LOCAL,
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
            measurement_id=MEASUREMENT_ID_CONDITIONAL_MEAN_RATE,
            variant_id=VARIANT_BLOCKED_CROSSFIT,
            extra_summary={
                "split_index": split_index,
                "embargo_steps": embargo,
                **embargo_claim_fields(),
                "fold1_n_increments": int(fold1_idx.size),
                "fold2_n_increments": int(fold2_idx.size),
            },
        )
    finite_idx = np.flatnonzero(finite_state)
    query1 = finite_idx[finite_idx < (split_index - embargo)]
    query2 = finite_idx[finite_idx > (split_index + embargo)]
    filled1 = _fill_conditional_mean(
        x,
        t,
        query1,
        ref_source_idx=fold2_idx,
        ref_increments=fold2_dx,
        neighborhood_k=int(neighborhood_k),
        minimum_support=minimum_support,
        max_neighborhood_radius=max_neighborhood_radius,
        min_temporal_span_sec=min_temporal_span_sec,
        weight_mode=mode,
        nominal_dt=nominal_dt,
        epsilon=float(epsilon),
        store_neighbor_source_idx=True,
    )
    filled2 = _fill_conditional_mean(
        x,
        t,
        query2,
        ref_source_idx=fold1_idx,
        ref_increments=fold1_dx,
        neighborhood_k=int(neighborhood_k),
        minimum_support=minimum_support,
        max_neighborhood_radius=max_neighborhood_radius,
        min_temporal_span_sec=min_temporal_span_sec,
        weight_mode=mode,
        nominal_dt=nominal_dt,
        epsilon=float(epsilon),
        store_neighbor_source_idx=True,
    )
    n_time, dimension = x.shape
    b_hat = np.full((n_time, dimension), np.nan, dtype=np.float32)
    n_effective = np.zeros(n_time, dtype=np.float32)
    radius = np.full(n_time, np.nan, dtype=np.float32)
    valid = np.zeros(n_time, dtype=np.int8)
    se_hat = np.full((n_time, dimension), np.nan, dtype=np.float32)
    neighbor_source_idx = np.full((n_time, int(neighborhood_k)), -1, dtype=np.int32)
    fold_id = np.zeros(n_time, dtype=np.int8)
    fold_id[query1] = 1
    fold_id[query2] = 2
    for filled in (filled1, filled2):
        mask = filled["valid"] == 1
        b_hat[mask] = filled["b_hat"][mask]
        n_effective[mask] = filled["n_effective"][mask]
        radius[mask] = filled["radius"][mask]
        valid[mask] = 1
        se_hat[mask] = filled["se_hat"][mask]
        neighbor_source_idx[mask] = filled["neighbor_source_idx"][mask]
    if int(np.sum(valid)) < max(1, int(np.ceil(float(min_valid_fraction) * n_time))):
        return _unavailable(
            status="insufficient_support",
            failure_reason=REASON_INSUFFICIENT_LOCAL,
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
            measurement_id=MEASUREMENT_ID_CONDITIONAL_MEAN_RATE,
            variant_id=VARIANT_BLOCKED_CROSSFIT,
            extra_summary={
                "n_valid_timepoints": int(np.sum(valid)),
                "n_timepoints": n_time,
                "fold1_n_increments": int(fold1_idx.size),
                "fold2_n_increments": int(fold2_idx.size),
            },
        )
    return _computed_result(
        measurement_id=MEASUREMENT_ID_CONDITIONAL_MEAN_RATE,
        variant_id=VARIANT_BLOCKED_CROSSFIT,
        coordinate_layer=coordinate_layer,
        coordinate_names=names,
        series={
            "b_hat": b_hat,
            "n_effective": n_effective,
            "radius": radius,
            "valid": valid,
            "se_hat": se_hat,
            "fold_id": fold_id,
            "neighbor_source_idx": neighbor_source_idx,
            **support_series_fields(crossfit_support),
        },
        summary={
            "n_timepoints": n_time,
            "n_valid_timepoints": int(np.sum(valid)),
            "n_increment_pairs": int(increments.shape[0]),
            "nominal_dt_sec": float(nominal_dt),
            "max_dt_relative_deviation": float(relative_deviation),
            "split_index": split_index,
            "embargo_steps": embargo,
            "n_blocks": 2,
            **embargo_claim_fields(),
            "fold1_n_increments": int(fold1_idx.size),
            "fold2_n_increments": int(fold2_idx.size),
            "fold1_source_idx": fold1_idx.astype(np.int32),
            "fold2_source_idx": fold2_idx.astype(np.int32),
            "not_sde_drift": True,
            **support_summary_fields(crossfit_support),
        },
        settings={
            "neighborhood_k": int(neighborhood_k),
            "min_samples": int(min_samples),
            "min_neighborhood_samples": int(min_neighborhood_samples),
            "dimension_aware_minimum_support": int(minimum_support),
            "max_gap_sec": max_gap_sec,
            "max_dt_relative_deviation": float(max_dt_relative_deviation),
            "max_neighborhood_radius": max_neighborhood_radius,
            "min_temporal_span_sec": min_temporal_span_sec,
            "min_valid_fraction": float(min_valid_fraction),
            "weight_mode": mode,
            "n_blocks": 2,
            "embargo_steps": embargo,
            **embargo_claim_fields(),
            **support_settings_fields(crossfit_support),
        },
        time_semantics="blocked_temporal_crossfit_conditional_increment_mean",
    )


def _relative_field_disagreement(
    left: np.ndarray,
    right: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    delta = np.linalg.norm(left - right, axis=1)
    return delta / (0.5 * (left_norm + right_norm) + float(epsilon))


def estimate_lag_diagnostics(
    state: np.ndarray,
    time: np.ndarray,
    *,
    segment_id: np.ndarray | None = None,
    coordinate_layer: str = "coords_3d_subject_anchored",
    coordinate_names: list[str] | None = None,
    neighborhood_k: int = 20,
    min_samples: int = 30,
    min_neighborhood_samples: int = 10,
    max_gap_sec: float | None = None,
    max_dt_relative_deviation: float = 0.05,
    max_neighborhood_radius: float | None = None,
    min_temporal_span_sec: float | None = None,
    min_valid_fraction: float = 0.1,
    weight_mode: str = WEIGHT_INVERSE_DISTANCE,
    lags: tuple[int, ...] | list[int] = DEFAULT_ITO_LAGS,
    consistency_rel_tol: float = 0.5,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Multi-lag consistency diagnostics on level-1 fields. Not ``ito_drift_level3``."""
    names = list(coordinate_names or ["m", "d", "e"])
    lag_values = tuple(int(value) for value in lags)
    if lag_values != DEFAULT_ITO_LAGS:
        return _unavailable(
            status="invalid",
            failure_reason="lag_diagnostics_requires_lags_1_2_4",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
            measurement_id=MEASUREMENT_ID_CONDITIONAL_MEAN_RATE,
            variant_id=VARIANT_LAG_DIAGNOSTICS,
        )
    fields: dict[int, dict[str, Any]] = {}
    for lag in lag_values:
        fields[lag] = estimate_finite_lag_drift(
            state,
            time,
            segment_id=segment_id,
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
            neighborhood_k=neighborhood_k,
            min_samples=min_samples,
            min_neighborhood_samples=min_neighborhood_samples,
            max_gap_sec=max_gap_sec,
            max_dt_relative_deviation=max_dt_relative_deviation,
            max_neighborhood_radius=max_neighborhood_radius,
            min_temporal_span_sec=min_temporal_span_sec,
            min_valid_fraction=min_valid_fraction,
            weight_mode=weight_mode,
            lag=lag,
            epsilon=epsilon,
        )
        if fields[lag].get("computation_status") != "computed":
            status = str(fields[lag].get("computation_status") or "insufficient_support")
            reason = str(fields[lag].get("failure_reason") or REASON_INSUFFICIENT_LOCAL)
            return _unavailable(
                status=status if status != "computed" else "insufficient_support",
                failure_reason=reason,
                coordinate_layer=coordinate_layer,
                coordinate_names=names,
                measurement_id=MEASUREMENT_ID_CONDITIONAL_MEAN_RATE,
                variant_id=VARIANT_LAG_DIAGNOSTICS,
                extra_summary={"failed_lag": int(lag)},
            )
    b1 = np.asarray(fields[1]["series"]["b_hat"], dtype=float)
    b2 = np.asarray(fields[2]["series"]["b_hat"], dtype=float)
    b4 = np.asarray(fields[4]["series"]["b_hat"], dtype=float)
    valid = (
        (np.asarray(fields[1]["series"]["valid"]) == 1)
        & (np.asarray(fields[2]["series"]["valid"]) == 1)
        & (np.asarray(fields[4]["series"]["valid"]) == 1)
        & np.all(np.isfinite(b1), axis=1)
        & np.all(np.isfinite(b2), axis=1)
        & np.all(np.isfinite(b4), axis=1)
    )
    n_time = int(b1.shape[0])
    if int(np.sum(valid)) < max(1, int(np.ceil(float(min_valid_fraction) * n_time))):
        return _unavailable(
            status="insufficient_support",
            failure_reason=REASON_INSUFFICIENT_LOCAL,
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
            measurement_id=MEASUREMENT_ID_CONDITIONAL_MEAN_RATE,
            variant_id=VARIANT_LAG_DIAGNOSTICS,
            extra_summary={"n_jointly_valid_timepoints": int(np.sum(valid))},
        )
    rel12 = _relative_field_disagreement(b1, b2, epsilon=epsilon)
    rel14 = _relative_field_disagreement(b1, b4, epsilon=epsilon)
    rel24 = _relative_field_disagreement(b2, b4, epsilon=epsilon)
    max_rel = np.maximum(np.maximum(rel12, rel14), rel24)
    median_rel = float(np.median(max_rel[valid]))
    lag_inconsistent = bool(median_rel > float(consistency_rel_tol))
    dt1 = float(fields[1]["summary"]["nominal_dt_sec"])
    dt2 = float(fields[2]["summary"]["nominal_dt_sec"])
    dt4 = float(fields[4]["summary"]["nominal_dt_sec"])
    design = np.column_stack(
        [np.ones(3, dtype=float), np.array([dt1, dt2, dt4], dtype=float)]
    )
    b0_hat = np.full_like(b1, np.nan, dtype=np.float32)
    for index in np.flatnonzero(valid):
        observations = np.vstack([b1[index], b2[index], b4[index]])
        intercept = np.empty(observations.shape[1], dtype=float)
        for dim in range(observations.shape[1]):
            fit, *_ = np.linalg.lstsq(design, observations[:, dim], rcond=None)
            intercept[dim] = fit[0]
        b0_hat[index] = intercept.astype(np.float32)
    jointly_valid = valid.astype(np.int8)
    return _computed_result(
        measurement_id=MEASUREMENT_ID_CONDITIONAL_MEAN_RATE,
        variant_id=VARIANT_LAG_DIAGNOSTICS,
        coordinate_layer=coordinate_layer,
        coordinate_names=names,
        series={
            "b_hat": b1.astype(np.float32),
            "b_hat_lag1": b1.astype(np.float32),
            "b_hat_lag2": b2.astype(np.float32),
            "b_hat_lag4": b4.astype(np.float32),
            "b0_hat": b0_hat,
            "n_effective": np.asarray(fields[1]["series"]["n_effective"]),
            "radius": np.asarray(fields[1]["series"]["radius"]),
            "valid": jointly_valid,
            "se_hat": np.asarray(fields[1]["series"]["se_hat"]),
            "max_relative_lag_disagreement": max_rel.astype(np.float32),
        },
        summary={
            "n_timepoints": n_time,
            "n_valid_timepoints": int(np.sum(jointly_valid)),
            "lags": [1, 2, 4],
            "consistency_rel_tol": float(consistency_rel_tol),
            "median_max_relative_lag_disagreement": median_rel,
            "multi_lag_consistency": (
                REASON_LAG_INCONSISTENT if lag_inconsistent else MULTI_LAG_CONSISTENT
            ),
            "qualification_status": QUALIFICATION_ITO_NOT_QUALIFIED,
            "not_sde_drift": True,
            "b0_hat_is_diagnostic_only": True,
            "not_ito_drift_level3": True,
            "transition_support_id_lag1": (fields[1].get("summary") or {}).get(
                "transition_support_id"
            ),
            "transition_support_id_lag2": (fields[2].get("summary") or {}).get(
                "transition_support_id"
            ),
            "transition_support_id_lag4": (fields[4].get("summary") or {}).get(
                "transition_support_id"
            ),
            **embargo_claim_fields(),
        },
        settings={
            "neighborhood_k": int(neighborhood_k),
            "lags": [1, 2, 4],
            "consistency_rel_tol": float(consistency_rel_tol),
            "never_auto_ito_qualified": True,
            "not_ito_drift_level3": True,
        },
        failure_reason=None,
        computation_status="computed",
        time_semantics="multi_lag_conditional_mean_rate_consistency_diagnostic",
    )


def estimate_chart_drift_family(
    state: np.ndarray,
    time: np.ndarray,
    *,
    segment_id: np.ndarray | None = None,
    coordinate_layer: str = "coords_3d_subject_anchored",
    coordinate_names: list[str] | None = None,
    neighborhood_k: int = 20,
    min_samples: int = 30,
    min_neighborhood_samples: int = 10,
    max_gap_sec: float | None = None,
    max_dt_relative_deviation: float = 0.05,
    max_neighborhood_radius: float | None = None,
    min_temporal_span_sec: float | None = None,
    min_valid_fraction: float = 0.1,
    weight_mode: str = WEIGHT_INVERSE_DISTANCE,
    realized_velocity_enabled: bool = True,
    pooled_enabled: bool = True,
    blocked_crossfit_enabled: bool = True,
    lag_diagnostics_enabled: bool = True,
    n_blocks: int = 2,
    embargo_steps: int = 4,
    ito_lags: tuple[int, ...] | list[int] = DEFAULT_ITO_LAGS,
    consistency_rel_tol: float = 0.5,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Orchestrate SL-LEV-MES-003 chart-drift measurements under one family."""
    names = list(coordinate_names or ["m", "d", "e"])
    refusal = _layer_refusal(
        state=state,
        coordinate_layer=coordinate_layer,
        coordinate_names=names,
    )
    if refusal is not None:
        return refusal
    if not (
        realized_velocity_enabled
        or pooled_enabled
        or blocked_crossfit_enabled
        or lag_diagnostics_enabled
    ):
        return _unavailable(
            status="not_requested",
            failure_reason="no_enabled_chart_drift_measurement",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    shared = dict(
        segment_id=segment_id,
        coordinate_layer=coordinate_layer,
        coordinate_names=names,
        neighborhood_k=neighborhood_k,
        min_samples=min_samples,
        min_neighborhood_samples=min_neighborhood_samples,
        max_gap_sec=max_gap_sec,
        max_dt_relative_deviation=max_dt_relative_deviation,
        max_neighborhood_radius=max_neighborhood_radius,
        min_temporal_span_sec=min_temporal_span_sec,
        min_valid_fraction=min_valid_fraction,
        weight_mode=weight_mode,
        epsilon=epsilon,
    )
    realized = None
    if realized_velocity_enabled:
        realized = estimate_realized_velocity_level0(
            state,
            time,
            segment_id=segment_id,
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
            min_samples=min_samples,
            max_gap_sec=max_gap_sec,
        )
    level1: dict[str, Any] = {
        "measurement_id": MEASUREMENT_ID_CONDITIONAL_MEAN_RATE,
        "interpretation_level": 1,
    }
    if pooled_enabled:
        level1[VARIANT_POOLED] = estimate_finite_lag_drift(state, time, **shared)
    if blocked_crossfit_enabled:
        level1[VARIANT_BLOCKED_CROSSFIT] = estimate_crossfit_drift(
            state,
            time,
            n_blocks=n_blocks,
            embargo_steps=embargo_steps,
            **shared,
        )
    if lag_diagnostics_enabled:
        level1[VARIANT_LAG_DIAGNOSTICS] = estimate_lag_diagnostics(
            state,
            time,
            lags=ito_lags,
            consistency_rel_tol=consistency_rel_tol,
            **shared,
        )
    computed_leaves: list[dict[str, Any]] = []
    if realized is not None:
        computed_leaves.append(realized)
    for key in (VARIANT_POOLED, VARIANT_BLOCKED_CROSSFIT, VARIANT_LAG_DIAGNOSTICS):
        item = level1.get(key)
        if isinstance(item, Mapping):
            computed_leaves.append(item)
    statuses = [str(item.get("computation_status") or "") for item in computed_leaves]
    parent_status = "not_testable"
    parent_reason: str | None = "no_computed_chart_drift_measurement"
    if any(status == "computed" for status in statuses):
        parent_status = "computed"
        parent_reason = None
    elif any(status == "invalid" for status in statuses):
        parent_status = "invalid"
        parent_reason = next(
            str(item.get("failure_reason") or "invalid")
            for item in computed_leaves
            if item.get("computation_status") == "invalid"
        )
    elif any(status == "insufficient_support" for status in statuses):
        parent_status = "insufficient_support"
        parent_reason = REASON_INSUFFICIENT_LOCAL
    elif statuses:
        parent_status = statuses[0] if statuses[0] else "not_testable"
        parent_reason = next(
            (str(item.get("failure_reason")) for item in computed_leaves if item.get("failure_reason")),
            parent_reason,
        )
    diagnostics = level1.get(VARIANT_LAG_DIAGNOSTICS) or {}
    summary = {
        "not_sde_drift": True,
        "independent_drift_for_A_bD": False,
        "realized_velocity_status": None if realized is None else realized.get("computation_status"),
        "pooled_status": (level1.get(VARIANT_POOLED) or {}).get("computation_status"),
        "blocked_crossfit_status": (level1.get(VARIANT_BLOCKED_CROSSFIT) or {}).get(
            "computation_status"
        ),
        "lag_diagnostics_status": diagnostics.get("computation_status"),
        "qualification_status": QUALIFICATION_ITO_NOT_QUALIFIED,
        "ito_drift_level3_written": False,
    }
    if diagnostics:
        summary["multi_lag_consistency"] = (diagnostics.get("summary") or {}).get(
            "multi_lag_consistency"
        )
    payload: dict[str, Any] = {
        "schema_version": CHART_DRIFT_SCHEMA_VERSION,
        "computation_status": parent_status,
        "failure_reason": parent_reason,
        "series": {},
        "summary": summary,
        "provenance": build_provenance(
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
            time_semantics="chart_drift_realized_velocity_and_conditional_mean_rate",
            estimator=ESTIMATOR_NAME,
            settings={
                "not_sde_drift": True,
                "independent_drift_for_A_bD": False,
                "never_auto_ito_qualified": True,
                "ito_drift_level3_written": False,
                "realized_velocity_enabled": bool(realized_velocity_enabled),
                "pooled_enabled": bool(pooled_enabled),
                "blocked_crossfit_enabled": bool(blocked_crossfit_enabled),
                "lag_diagnostics_enabled": bool(lag_diagnostics_enabled),
            },
        ),
        MEASUREMENT_ID_CONDITIONAL_MEAN_RATE: level1,
    }
    if realized is not None:
        payload[MEASUREMENT_ID_REALIZED_VELOCITY] = realized
    return _finalize(payload)


def estimate_conditional_mean_rate_level1(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Public name for the pooled finite-lag conditional mean."""
    return estimate_finite_lag_drift(*args, **kwargs)


def estimate_blocked_crossfit_mean_rate(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Public name for the blocked-crossfit level-1 variant."""
    return estimate_crossfit_drift(*args, **kwargs)
