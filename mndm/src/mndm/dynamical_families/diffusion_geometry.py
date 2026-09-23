"""Local diffusion-geometry estimator (``mndm.diffusion_geometry.v1``).

The estimator targets conditional increment covariance ``a(x)`` in chart
space.  Register identity: ``conditional_covariance_rate_level1`` documents
existing ``a_hat`` (centered increment covariance over nominal ``dt``), not
a rename and not ``ito_diffusion_tensor_level3``.  ``contract_status=standard``
names the schema contract, not an empirical or NDT license.  The object
remains chart-dependent and is not a latent Itô tensor.  Jacobian residual
covariance is never accepted as diffusion.  MNPS ``x_dot`` is not an
independently qualified SDE drift.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .chart_drift import (
    A_SEMANTICS_RAW,
    MODE_ALIGNMENT_ONLY,
    MODE_NOT_SUPPLIED,
    RATIO_SEMANTICS_C1,
    RATIO_SEMANTICS_NA,
    REASON_C2_CLOSED,
    REASON_NOT_SUPPLIED,
    RESIDUALIZATION_NONE,
    resolve_chart_drift,
    source_is_requested,
)
from .contracts import DIFFUSION_GEOMETRY_SCHEMA_VERSION, build_provenance, unavailable_result
from .measurement_register import (
    MEASUREMENT_ID_CONDITIONAL_COVARIANCE,
    MEASUREMENT_ID_INCREMENT_COVARIANCE,
    QUALIFICATION_INCREMENT_COV,
    stamp_register_fields,
)
from ..measurement_certificate import attach_certificate
from ..inferential_grain import attach_grain, attach_grain_for_schema
from .validity import (
    chunked_nearest_neighbors,
    project_to_psd,
    validate_trajectory,
)
from .transition_support import (
    build_transition_support,
    support_series_fields,
    support_settings_fields,
    support_summary_fields,
)


def _stamp_covariance_identity(result: dict[str, Any]) -> dict[str, Any]:
    """Document ``a_hat`` as level-1 conditional covariance without renaming it."""
    return stamp_register_fields(result, MEASUREMENT_ID_CONDITIONAL_COVARIANCE)


def _stamp_qualification(result: dict[str, Any], token: str) -> dict[str, Any]:
    out = dict(result)
    out["qualification_status"] = token
    summary = dict(out.get("summary") or {})
    summary["qualification_status"] = token
    out["summary"] = summary
    return out


def _increment_covariance_leaf(
    *,
    increments: np.ndarray | None,
    source_idx: np.ndarray | None,
    n_time: int,
    dimension: int,
    coordinate_layer: str,
    coordinate_names: list[str],
    extra_summary: Mapping[str, Any] | None = None,
    extra_settings: Mapping[str, Any] | None = None,
    status: str | None = None,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    """Recording-level unconditional increment covariance, not a_hat and not /dt."""
    series_cov = np.full((n_time, dimension, dimension), np.nan, dtype=np.float32)
    cov = None
    computed = False
    reason = failure_reason
    if increments is not None:
        finite = np.all(np.isfinite(increments), axis=1)
        rows = np.asarray(increments, dtype=float)[finite]
        if rows.shape[0] >= 2 and rows.shape[1] == dimension:
            cov_raw = np.atleast_2d(np.cov(rows, rowvar=False, ddof=1))
            if cov_raw.shape == (dimension, dimension) and np.all(np.isfinite(cov_raw)):
                cov = cov_raw.astype(np.float32)
                computed = True
                if source_idx is not None:
                    for center in np.asarray(source_idx, dtype=np.int32):
                        t_idx = int(center)
                        if 0 <= t_idx < n_time:
                            series_cov[t_idx] = cov
    if not computed and reason is None:
        reason = "insufficient_increment_pairs"
    result = unavailable_result(
        DIFFUSION_GEOMETRY_SCHEMA_VERSION,
        status="not_testable",
        failure_reason=reason or "not_requested",
        coordinate_layer=coordinate_layer,
        coordinate_names=coordinate_names,
    )
    if computed:
        result["computation_status"] = "computed"
        result["failure_reason"] = None
    else:
        result["computation_status"] = status or "insufficient_support"
        result["failure_reason"] = reason
    result["series"] = {"increment_covariance": series_cov}
    summary = dict(result.get("summary") or {})
    if extra_summary:
        summary.update(dict(extra_summary))
    summary["not_divided_by_dt"] = True
    summary["not_local_knn"] = True
    summary["not_ito_diffusion_tensor"] = True
    if cov is not None:
        summary["increment_covariance"] = cov
        n_pairs = int(np.sum(np.all(np.isfinite(increments), axis=1))) if increments is not None else 0
        summary["n_increment_pairs"] = n_pairs
    result["summary"] = summary
    provenance = dict(result.get("provenance") or {})
    settings = dict(provenance.get("settings") or {})
    settings["not_divided_by_dt"] = True
    settings["not_local_knn"] = True
    settings["not_ito_diffusion_tensor"] = True
    if extra_settings:
        settings.update(dict(extra_settings))
    provenance["settings"] = settings
    provenance["estimator"] = "unconditional_increment_covariance"
    provenance["time_semantics"] = (
        "within_segment_unconditional_increment_covariance_not_divided_by_dt"
    )
    result["provenance"] = provenance
    leaf = stamp_register_fields(result, MEASUREMENT_ID_INCREMENT_COVARIANCE)
    leaf = _stamp_qualification(leaf, QUALIFICATION_INCREMENT_COV)
    leaf = attach_certificate(leaf)
    return attach_grain(
        leaf,
        native="recording",
        parent="recording",
        repeated_measure="false",
    )


def _support_increment_kwargs(support: Any, x: np.ndarray) -> dict[str, Any]:
    return {
        "increments": support.increments,
        "source_idx": support.source_idx,
        "n_time": int(x.shape[0]),
        "dimension": int(x.shape[1]),
    }


def _attach_increment(parent: dict[str, Any], leaf: dict[str, Any]) -> dict[str, Any]:
    out = dict(parent)
    out[MEASUREMENT_ID_INCREMENT_COVARIANCE] = leaf
    return out


def _unavailable_covariance(**kwargs: Any) -> dict[str, Any]:
    increments = kwargs.pop("increments", None)
    source_idx = kwargs.pop("source_idx", None)
    n_time = int(kwargs.pop("n_time", 0) or 0)
    dimension = kwargs.pop("dimension", None)
    parent = attach_grain_for_schema(
        attach_certificate(
            _stamp_covariance_identity(
                unavailable_result(DIFFUSION_GEOMETRY_SCHEMA_VERSION, **kwargs)
            )
        )
    )
    names = list(kwargs.get("coordinate_names") or ["m", "d", "e"])
    leaf = _increment_covariance_leaf(
        increments=increments,
        source_idx=source_idx,
        n_time=n_time,
        dimension=int(dimension or max(len(names), 1)),
        coordinate_layer=str(kwargs.get("coordinate_layer") or "unknown"),
        coordinate_names=names,
        status=str(kwargs.get("status") or "insufficient_support"),
        failure_reason=str(kwargs.get("failure_reason") or "invalid_trajectory"),
    )
    return _attach_increment(parent, leaf)


def unavailable_diffusion_geometry(
    *,
    status: str,
    failure_reason: str,
    coordinate_layer: str,
    coordinate_names: list[str],
) -> dict[str, Any]:
    """Schema-complete unavailable diffusion payload with nested increment L0."""
    names = list(coordinate_names or ["m", "d", "e"])
    return _unavailable_covariance(
        status=status,
        failure_reason=failure_reason,
        coordinate_layer=coordinate_layer,
        coordinate_names=names,
        n_time=0,
        dimension=max(len(names), 1),
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
    zeros. Chart-drift family outputs are not that independent ``b``.
    A drift vector without ``drift_source=truth_known_chart_b`` is not
    auto-promoted.

    C1 (authorized for synthetic qualification): ``drift`` is an alignment
    field only when ``drift_source`` is ``truth_known_chart_b``.
    ``residualize_increments`` defaults to False so ``a_hat``
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
        return _unavailable_covariance(
            status="insufficient_support" if failure and "insufficient" in failure else "invalid",
            failure_reason=failure or "invalid_trajectory",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    if len(names) != x.shape[1]:
        return _unavailable_covariance(
            status="invalid",
            failure_reason="coordinate_name_dimension_mismatch",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    minimum_dimension_support = max(int(min_neighborhood_samples), 3 * x.shape[1] + 1)
    if residualize_increments:
        # C2 refuses a_hat residualization before any drift-source interpretation.
        # Nested increment_covariance_level0 still writes.
        support = build_transition_support(
            x,
            t,
            segments,
            lag=1,
            max_gap_sec=max_gap_sec,
            max_dt_relative_deviation=float(max_dt_relative_deviation),
        )
        return _unavailable_covariance(
            status="invalid",
            failure_reason=REASON_C2_CLOSED,
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
            **_support_increment_kwargs(support, x),
        )
    resolved = resolve_chart_drift(
        source=drift_source,
        field=drift,
        mode=MODE_ALIGNMENT_ONLY,
        enabled=source_is_requested(drift_source),
    )
    alignment_failure = resolved.failure_reason
    source_token = resolved.source
    drift = resolved.field
    support = build_transition_support(
        x,
        t,
        segments,
        lag=1,
        max_gap_sec=max_gap_sec,
        max_dt_relative_deviation=float(max_dt_relative_deviation),
    )
    source_idx = support.source_idx
    increments = support.increments
    increment_ctx = _support_increment_kwargs(support, x)
    if neighborhood_k < minimum_dimension_support:
        return _unavailable_covariance(
            status="invalid",
            failure_reason="neighborhood_k_below_dimension_aware_minimum_support",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
            **increment_ctx,
        )
    if increments.shape[0] < int(min_neighborhood_samples):
        return _unavailable_covariance(
            status="insufficient_support",
            failure_reason="insufficient_valid_increment_pairs",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
            **increment_ctx,
        )
    if support.failure_reason == "non_positive_nominal_dt":
        return _unavailable_covariance(
            status="invalid",
            failure_reason="non_positive_nominal_dt",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
            **increment_ctx,
        )
    if support.failure_reason == "materially_irregular_increment_timestep":
        return _unavailable_covariance(
            status="not_testable",
            failure_reason="materially_irregular_increment_timestep",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
            **increment_ctx,
        )
    nominal_dt = float(support.nominal_dt_sec)
    relative_deviation = float(support.observed_dt_relative_deviation)

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
            return _unavailable_covariance(
                status="invalid",
                failure_reason="drift_shape_mismatch",
                coordinate_layer=coordinate_layer,
                coordinate_names=names,
                **increment_ctx,
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
        return _unavailable_covariance(
            status="insufficient_support",
            failure_reason="insufficient_valid_neighborhood_coverage",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
            increments=increments,
            source_idx=source_idx,
            n_time=n_time,
            dimension=dimension,
        )
    parent = attach_grain_for_schema(attach_certificate(_stamp_covariance_identity({
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
            **support_series_fields(support),
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
            "independent_b_for_A_bD": bool(drift_available),
            "drift_alignment_failure_reason": (
                None if drift_available else (alignment_failure or REASON_NOT_SUPPLIED)
            ),
            "a_semantics": a_semantics,
            "ratio_semantics": ratio_semantics,
            **support_summary_fields(support),
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
                **support_settings_fields(support),
            },
        ),
    })))
    increment_leaf = _increment_covariance_leaf(
        increments=increments,
        source_idx=source_idx,
        n_time=n_time,
        dimension=dimension,
        coordinate_layer=coordinate_layer,
        coordinate_names=names,
        extra_summary=support_summary_fields(support),
        extra_settings=support_settings_fields(support),
    )
    return _attach_increment(parent, increment_leaf)
