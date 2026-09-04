"""Local diffusion-geometry estimator (``mndm.diffusion_geometry.v1``).

The estimator targets conditional increment covariance ``a(x)`` in chart
space.  ``contract_status=standard`` names the schema contract, not an
empirical or NDT license.  The object remains chart-dependent and is not a
latent Itô tensor.  Jacobian residual covariance is never accepted as
diffusion.  MNPS ``x_dot`` is not an independently qualified SDE drift.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .chart_drift import (
    A_SEMANTICS_RAW,
    FORBIDDEN_SOURCE_REASONS,
    MODE_ALIGNMENT_ONLY,
    MODE_NOT_SUPPLIED,
    RATIO_SEMANTICS_C1,
    RATIO_SEMANTICS_NA,
    REASON_C2_CLOSED,
    REASON_NOT_SUPPLIED,
    RESIDUALIZATION_NONE,
    SOURCE_NOT_SUPPLIED,
    SOURCE_TRUTH_KNOWN,
)
from .contracts import DIFFUSION_GEOMETRY_SCHEMA_VERSION, build_provenance, unavailable_result
from ..measurement_certificate import attach_certificate
from ..inferential_grain import attach_grain_for_schema
from .validity import (
    chunked_nearest_neighbors,
    increment_pairs,
    project_to_psd,
    validate_trajectory,
)


def _tensor_metrics(tensor: np.ndarray, epsilon: float) -> dict[str, float]:
    values = np.linalg.eigvalsh(tensor)
    trace = float(np.trace(tensor))
    squared_trace = float(np.sum(values**2))
    return {
        "D_total": trace,
        "d_diff": float(trace**2 / squared_trace) if squared_trace > epsilon else float("nan"),
        "c_diff": float(values[-1] / (trace + epsilon)) if trace >= 0 else float("nan"),
    }


def _metrics(tensor: np.ndarray, drift: np.ndarray, dt_sec: float, epsilon: float) -> dict[str, float]:
    out = _tensor_metrics(tensor, epsilon)
    drift_norm_sq = float(np.dot(drift, drift))
    trace = out["D_total"]
    out["A_bD"] = (
        float(drift @ tensor @ drift / (drift_norm_sq * trace + epsilon))
        if drift_norm_sq > 0
        else float("nan")
    )
    out["R_b_over_a"] = float(dt_sec * drift_norm_sq / (trace + epsilon))
    return out


def estimate_local_diffusion_geometry(
    state: np.ndarray,
    time: np.ndarray,
    *,
    drift: np.ndarray | None = None,
    residualize_increments: bool = False,
    drift_source: str | None = None,
    segment_id: np.ndarray | None = None,
    coordinate_layer: str = "unknown",
    coordinate_names: list[str] | None = None,
    neighborhood_k: int = 20,
    min_samples: int = 30,
    min_neighborhood_samples: int = 10,
    max_gap_sec: float | None = None,
    max_dt_relative_deviation: float = 0.05,
    max_neighborhood_radius: float | None = None,
    min_valid_fraction: float = 0.1,
    psd_floor: float = 1e-8,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Estimate local diffusion tensors from within-segment state increments.

    Chart-space increment covariance under ``mndm.diffusion_geometry.v1``
    (``contract_status=standard``).  It requires approximately regular
    sampling because raw increment covariance scales with the time step;
    irregular time grids are refused rather than silently normalized.
    Without an independently supplied drift, ``A_bD`` and ``R_b_over_a``
    are ``not_testable`` (``independent_drift_not_supplied``), not silent
    zeros.

    C1 (authorized for synthetic qualification): ``drift`` is an alignment
    field only.  ``residualize_increments`` defaults to False so ``a_hat``
    stays raw increment covariance.  C2 residualization is **not
    authorized**: a True flag is ``invalid`` /
    ``c2_residualize_increments_not_authorized`` rather than a silent
    change to ``a_hat``.
    """
    x, t, segments, finite_state, failure = validate_trajectory(
        state, time, min_samples=min_samples, segment_id=segment_id
    )
    names = coordinate_names or [f"dim_{idx}" for idx in range(np.asarray(state).shape[1] if np.asarray(state).ndim == 2 else 0)]
    if failure is not None or x is None or t is None or segments is None or finite_state is None:
        return unavailable_result(
            DIFFUSION_GEOMETRY_SCHEMA_VERSION,
            status="insufficient_support" if failure and "insufficient" in failure else "invalid",
            failure_reason=failure or "invalid_trajectory",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    if len(names) != x.shape[1]:
        return unavailable_result(
            DIFFUSION_GEOMETRY_SCHEMA_VERSION,
            status="invalid",
            failure_reason="coordinate_name_dimension_mismatch",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    minimum_dimension_support = max(int(min_neighborhood_samples), 3 * x.shape[1] + 1)
    if neighborhood_k < minimum_dimension_support:
        return unavailable_result(
            DIFFUSION_GEOMETRY_SCHEMA_VERSION,
            status="invalid",
            failure_reason="neighborhood_k_below_dimension_aware_minimum_support",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    source_token = (
        str(drift_source).strip()
        if drift_source
        else (SOURCE_TRUTH_KNOWN if drift is not None else SOURCE_NOT_SUPPLIED)
    )
    alignment_failure = None
    if source_token in FORBIDDEN_SOURCE_REASONS:
        drift = None
        residualize_increments = False
        alignment_failure = FORBIDDEN_SOURCE_REASONS[source_token]
    if residualize_increments:
        return unavailable_result(
            DIFFUSION_GEOMETRY_SCHEMA_VERSION,
            status="invalid",
            failure_reason=REASON_C2_CLOSED,
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    source_idx, increments, dts = increment_pairs(x, t, segments, max_gap_sec=max_gap_sec)
    if increments.shape[0] < int(min_neighborhood_samples):
        return unavailable_result(
            DIFFUSION_GEOMETRY_SCHEMA_VERSION,
            status="insufficient_support",
            failure_reason="insufficient_valid_increment_pairs",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    nominal_dt = float(np.median(dts))
    relative_deviation = float(np.max(np.abs(dts - nominal_dt)) / nominal_dt)
    if relative_deviation > float(max_dt_relative_deviation):
        return unavailable_result(
            DIFFUSION_GEOMETRY_SCHEMA_VERSION,
            status="not_testable",
            failure_reason="materially_irregular_increment_timestep",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )

    drift_available = drift is not None
    if drift is None:
        drift_values = None
        residuals = increments
        drift_mode = MODE_NOT_SUPPLIED
        residualization_token = RESIDUALIZATION_NONE
        a_semantics = A_SEMANTICS_RAW
        ratio_semantics = RATIO_SEMANTICS_NA
    else:
        drift_array = np.asarray(drift, dtype=float)
        if drift_array.shape != x.shape:
            return unavailable_result(
                DIFFUSION_GEOMETRY_SCHEMA_VERSION,
                status="invalid",
                failure_reason="drift_shape_mismatch",
                coordinate_layer=coordinate_layer,
                coordinate_names=names,
            )
        drift_values = drift_array[source_idx]
        residuals = increments
        drift_mode = MODE_ALIGNMENT_ONLY
        residualization_token = RESIDUALIZATION_NONE
        a_semantics = A_SEMANTICS_RAW
        ratio_semantics = RATIO_SEMANTICS_C1

    n_time, dimension = x.shape
    tensors = np.full((n_time, dimension, dimension), np.nan, dtype=np.float32)
    valid = np.zeros(n_time, dtype=np.int8)
    metrics = {name: np.full(n_time, np.nan, dtype=np.float32) for name in ("D_total", "d_diff", "c_diff", "A_bD", "R_b_over_a")}
    support_count = np.zeros(n_time, dtype=np.int32)
    increment_state = x[source_idx]
    k = min(int(neighborhood_k), increment_state.shape[0])
    psd_floor_applied = np.zeros(n_time, dtype=np.int8)
    raw_min_eigenvalue = np.full(n_time, np.nan, dtype=np.float32)
    finite_idx = np.flatnonzero(finite_state)
    nearest_indices, nearest_distances = chunked_nearest_neighbors(
        x[finite_idx],
        increment_state,
        k,
    )
    for row, center in enumerate(finite_idx):
        nearest = nearest_indices[row]
        distances = nearest_distances[row]
        if max_neighborhood_radius is not None and float(np.sqrt(np.max(distances))) > float(max_neighborhood_radius):
            continue
        local_residual = residuals[nearest]
        finite_rows = np.all(np.isfinite(local_residual), axis=1)
        local_residual = local_residual[finite_rows]
        support_count[center] = int(local_residual.shape[0])
        if local_residual.shape[0] < minimum_dimension_support:
            continue
        covariance = np.atleast_2d(np.cov(local_residual, rowvar=False, ddof=1)) / nominal_dt
        try:
            tensor, psd_diagnostics = project_to_psd(covariance, eigenvalue_floor=psd_floor)
        except (ValueError, np.linalg.LinAlgError):
            continue
        psd_floor_applied[center] = int(psd_diagnostics["psd_floor_applied"])
        raw_min_eigenvalue[center] = float(psd_diagnostics["raw_min_eigenvalue"])
        tensors[center] = tensor.astype(np.float32)
        if drift_available and drift_values is not None:
            mean_drift = np.mean(drift_values[nearest][finite_rows], axis=0)
            values = _metrics(tensor, mean_drift, nominal_dt, epsilon)
        else:
            # Do not call _metrics with a zero drift: that would set
            # R_b_over_a to 0. Alignment scalars stay NaN until an
            # independent drift is supplied.
            values = {
                **_tensor_metrics(tensor, epsilon),
                "A_bD": float("nan"),
                "R_b_over_a": float("nan"),
            }
        for name, value in values.items():
            metrics[name][center] = value
        valid[center] = 1

    if int(np.sum(valid)) < max(1, int(np.ceil(float(min_valid_fraction) * n_time))):
        return unavailable_result(
            DIFFUSION_GEOMETRY_SCHEMA_VERSION,
            status="insufficient_support",
            failure_reason="insufficient_valid_neighborhood_coverage",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    return attach_grain_for_schema(attach_certificate({
        "schema_version": DIFFUSION_GEOMETRY_SCHEMA_VERSION,
        "computation_status": "computed",
        "failure_reason": None,
        "series": {
            "a_hat": tensors,
                # Standard public contract names; legacy aliases above remain
                # during the v1 migration period.
                "diffusion_tensor": tensors,
            "valid": valid,
                "validity_status": valid,
            "support_count": support_count,
                "support_n": support_count,
            "psd_floor_applied": psd_floor_applied,
            "raw_min_eigenvalue": raw_min_eigenvalue,
            **metrics,
                "diffusion_total": metrics["D_total"],
                "diffusion_effective_dimension": metrics["d_diff"],
                "diffusion_concentration": metrics["c_diff"],
                "drift_diffusion_alignment": metrics["A_bD"],
        },
        "summary": {
            "n_timepoints": int(n_time),
            "n_valid_timepoints": int(np.sum(valid)),
            "n_increment_pairs": int(increments.shape[0]),
            "nominal_dt_sec": nominal_dt,
            "max_dt_relative_deviation": relative_deviation,
            "A_bD_computation_status": (
                "computed" if drift_available else "not_testable"
            ),
            "R_b_over_a_computation_status": (
                "computed" if drift_available else "not_testable"
            ),
            "drift_alignment_failure_reason": (
                None if drift_available else (alignment_failure or REASON_NOT_SUPPLIED)
            ),
            "a_semantics": a_semantics,
            "ratio_semantics": ratio_semantics,
        },
        "provenance": build_provenance(
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
            time_semantics="within_segment_one_step_increment_covariance_divided_by_nominal_dt",
            estimator="local_increment_covariance",
            settings={
                "neighborhood_k": int(neighborhood_k),
                "min_samples": int(min_samples),
                "min_neighborhood_samples": int(min_neighborhood_samples),
                "dimension_aware_minimum_support": minimum_dimension_support,
                "max_gap_sec": max_gap_sec,
                "max_dt_relative_deviation": float(max_dt_relative_deviation),
                "max_neighborhood_radius": max_neighborhood_radius,
                "min_valid_fraction": float(min_valid_fraction),
                "psd_floor": float(psd_floor),
                "drift_source": source_token,
                "drift_mode": drift_mode,
                "drift_residualization": residualization_token,
                "a_semantics": a_semantics,
                "ratio_semantics": ratio_semantics,
            },
        ),
    }))
