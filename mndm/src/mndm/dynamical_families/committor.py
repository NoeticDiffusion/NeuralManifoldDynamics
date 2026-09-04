"""Standard data-driven committor estimator with explicit A/B labels."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .contracts import COMMITTOR_SCHEMA_VERSION, build_provenance, unavailable_result
from ..measurement_certificate import attach_certificate
from ..inferential_grain import attach_grain_for_schema
from .validity import chunked_nearest_neighbors, increment_pairs, validate_trajectory


def _next_hit_outcomes(labels: np.ndarray, segments: np.ndarray, set_a: set[int], set_b: set[int]) -> np.ndarray:
    """Encode the first future A/B hit per row: 0=A, 1=B, NaN=unresolved."""
    out = np.full(labels.shape[0], np.nan, dtype=float)
    if labels.size == 0:
        return out
    starts = np.concatenate(
        [
            np.asarray([0], dtype=np.int32),
            np.flatnonzero(segments[1:] != segments[:-1]).astype(np.int32) + 1,
        ]
    )
    ends = np.concatenate([starts[1:], np.asarray([labels.size], dtype=np.int32)])
    for start, end in zip(starts, ends):
        next_outcome = float("nan")
        for index in range(int(end) - 1, int(start) - 1, -1):
            value = int(labels[index])
            if value in set_a:
                next_outcome = 0.0
            elif value in set_b:
                next_outcome = 1.0
            out[index] = next_outcome
    return out


def estimate_committor(
    state: np.ndarray,
    time: np.ndarray,
    regime_labels: np.ndarray,
    *,
    set_A: Sequence[int],
    set_B: Sequence[int],
    segment_id: np.ndarray | None = None,
    coordinate_layer: str = "unknown",
    coordinate_names: list[str] | None = None,
    neighborhood_k: int = 30,
    min_support: int = 30,
    min_transition_segments: int = 5,
    max_neighborhood_radius: float | None = None,
    min_valid_fraction: float = 0.1,
) -> dict[str, Any]:
    """Estimate q(A→B) by local averaging of observed first-hit outcomes.

    This is not a substitute for simulator Monte-Carlo truth or a generator
    solution.  It is deliberately unavailable when labels or future A/B hits
    do not support the stated first-hit semantics.  It serializes ``q_A_to_B``
    from observed hits; it does not emit ``V_1/2`` or ``|grad q|``.
    """
    x, t, segments, finite_state, failure = validate_trajectory(
        state, time, min_samples=min_support, segment_id=segment_id
    )
    names = coordinate_names or [f"dim_{idx}" for idx in range(np.asarray(state).shape[1] if np.asarray(state).ndim == 2 else 0)]
    if failure is not None or x is None or t is None or segments is None or finite_state is None:
        return unavailable_result(
            COMMITTOR_SCHEMA_VERSION,
            status="insufficient_support" if failure and "insufficient" in failure else "invalid",
            failure_reason=failure or "invalid_trajectory",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    labels = np.asarray(regime_labels).reshape(-1)
    if labels.size != x.shape[0]:
        return unavailable_result(
            COMMITTOR_SCHEMA_VERSION,
            status="invalid",
            failure_reason="regime_label_shape_mismatch",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    try:
        numeric_labels = labels.astype(float, copy=False)
    except (TypeError, ValueError):
        numeric_labels = np.array([np.nan])
    if not np.all(np.isfinite(numeric_labels)) or not np.all(numeric_labels == np.floor(numeric_labels)):
        return unavailable_result(
            COMMITTOR_SCHEMA_VERSION,
            status="invalid",
            failure_reason="regime_labels_must_be_finite_integers",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    if len(names) != x.shape[1]:
        return unavailable_result(
            COMMITTOR_SCHEMA_VERSION,
            status="invalid",
            failure_reason="coordinate_name_dimension_mismatch",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    if neighborhood_k < min_support:
        return unavailable_result(
            COMMITTOR_SCHEMA_VERSION,
            status="invalid",
            failure_reason="neighborhood_k_below_minimum_support",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    set_a, set_b = {int(v) for v in set_A}, {int(v) for v in set_B}
    if not set_a or not set_b or set_a & set_b:
        return unavailable_result(
            COMMITTOR_SCHEMA_VERSION,
            status="invalid",
            failure_reason="invalid_or_overlapping_regime_sets",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    labels = numeric_labels.astype(int, copy=False)
    segment_has_both = np.array(
        [bool(np.any(np.isin(labels[segments == segment], list(set_a))) and np.any(np.isin(labels[segments == segment], list(set_b)))) for segment in np.unique(segments)]
    )
    if int(np.sum(segment_has_both)) < int(min_transition_segments):
        return unavailable_result(
            COMMITTOR_SCHEMA_VERSION,
            status="not_testable",
            failure_reason="insufficient_independent_A_B_transition_segments",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    outcomes = _next_hit_outcomes(labels, segments, set_a, set_b)
    observed = np.isfinite(outcomes) & finite_state
    if int(np.sum(observed)) < int(min_support):
        return unavailable_result(
            COMMITTOR_SCHEMA_VERSION,
            status="insufficient_support",
            failure_reason="insufficient_resolved_first_hit_outcomes",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )

    result = np.full(x.shape[0], np.nan, dtype=np.float32)
    support = np.zeros(x.shape[0], dtype=np.int32)
    observed_idx = np.flatnonzero(observed)
    observed_state = x[observed]
    k = min(int(neighborhood_k), observed_idx.size)
    finite_idx = np.flatnonzero(finite_state)
    nearest_indices, nearest_distances = chunked_nearest_neighbors(
        x[finite_idx],
        observed_state,
        k,
    )
    for row, index in enumerate(finite_idx):
        nearest = nearest_indices[row]
        distances = nearest_distances[row]
        if max_neighborhood_radius is not None and float(np.sqrt(np.max(distances))) > float(max_neighborhood_radius):
            continue
        local = outcomes[observed_idx[nearest]]
        support[index] = int(local.size)
        if local.size >= int(min_support):
            result[index] = float(np.mean(local))
    result[finite_state & np.isin(labels, list(set_a))] = 0.0
    result[finite_state & np.isin(labels, list(set_b))] = 1.0

    valid = np.isfinite(result)
    if int(np.sum(valid)) < max(1, int(np.ceil(float(min_valid_fraction) * x.shape[0]))):
        return unavailable_result(
            COMMITTOR_SCHEMA_VERSION,
            status="insufficient_support",
            failure_reason="no_committor_neighborhoods_with_sufficient_support",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    interface = (result > 0.25) & (result < 0.75)
    return attach_grain_for_schema(attach_certificate({
        "schema_version": COMMITTOR_SCHEMA_VERSION,
        "computation_status": "computed",
        "failure_reason": None,
        "series": {
            "q_A_to_B": result,
            "resolved_first_hit_outcome": outcomes.astype(np.float32),
            "support_count": support,
            "valid": valid.astype(np.int8),
        },
        "summary": {
            "n_valid_timepoints": int(np.sum(valid)),
            "n_resolved_first_hit_outcomes": int(np.sum(observed)),
            "n_independent_A_B_transition_segments": int(np.sum(segment_has_both)),
            "transition_interface_fraction_0_25_0_75": float(np.mean(interface[valid])),
        },
        "provenance": build_provenance(
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
            time_semantics="within_segment_first_future_hit_of_explicit_A_or_B",
            estimator="local_first_hit_outcome_average",
            settings={
                "set_A": sorted(set_a),
                "set_B": sorted(set_b),
                "neighborhood_k": int(neighborhood_k),
                "min_support": int(min_support),
                "min_transition_segments": int(min_transition_segments),
                "max_neighborhood_radius": max_neighborhood_radius,
                "min_valid_fraction": float(min_valid_fraction),
            },
        ),
    }))


def estimate_committor_local_law_dense_grid_o2b(
    state: np.ndarray,
    time: np.ndarray,
    reaction_coordinate: np.ndarray,
    regime_labels: np.ndarray,
    *,
    set_A: Sequence[int],
    set_B: Sequence[int],
    grid_min: float,
    grid_max: float,
    diffusion_coefficient: float,
    segment_id: np.ndarray | None = None,
    coordinate_layer: str = "unknown",
    coordinate_names: list[str] | None = None,
    reaction_coordinate_name: str = "reaction_coordinate",
    grid_resolution: int = 65,
    min_samples: int = 30,
    min_support_per_grid: int = 30,
    min_transition_segments: int = 5,
    max_dt_relative_deviation: float = 0.05,
    min_valid_fraction: float = 0.1,
) -> dict[str, Any]:
    """Estimate a 1-D local-law committor on the frozen O2b grid contract.

    The reaction coordinate is explicit and scalar. One-step increments are
    binned onto a uniformly spaced query grid, local drift is estimated from
    those increments, and the 1-D constant-diffusion quadrature is applied.
    The estimator refuses coarse grids or under-supported query points rather
    than silently falling back to the legacy sparse-neighborhood estimator.
    An internal 1-D potential is used only for the quadrature that produces
    ``q``; neither ``V_1/2`` nor ``|grad q|`` is serialized.
    """
    x, t, segments, finite_state, failure = validate_trajectory(
        state,
        time,
        min_samples=min_samples,
        segment_id=segment_id,
    )
    names = coordinate_names or [
        f"dim_{idx}"
        for idx in range(np.asarray(state).shape[1] if np.asarray(state).ndim == 2 else 0)
    ]
    if failure is not None or x is None or t is None or segments is None or finite_state is None:
        return unavailable_result(
            COMMITTOR_SCHEMA_VERSION,
            status="insufficient_support" if failure and "insufficient" in failure else "invalid",
            failure_reason=failure or "invalid_trajectory",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    if len(names) != x.shape[1]:
        return unavailable_result(
            COMMITTOR_SCHEMA_VERSION,
            status="invalid",
            failure_reason="coordinate_name_dimension_mismatch",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )

    reaction = np.asarray(reaction_coordinate, dtype=float).reshape(-1)
    labels = np.asarray(regime_labels).reshape(-1)
    if reaction.size != x.shape[0]:
        return unavailable_result(
            COMMITTOR_SCHEMA_VERSION,
            status="invalid",
            failure_reason="reaction_coordinate_shape_mismatch",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    if labels.size != x.shape[0]:
        return unavailable_result(
            COMMITTOR_SCHEMA_VERSION,
            status="invalid",
            failure_reason="regime_label_shape_mismatch",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    if not np.all(np.isfinite(reaction)):
        return unavailable_result(
            COMMITTOR_SCHEMA_VERSION,
            status="invalid",
            failure_reason="reaction_coordinate_must_be_finite",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    try:
        numeric_labels = labels.astype(float, copy=False)
    except (TypeError, ValueError):
        numeric_labels = np.full(labels.size, np.nan, dtype=float)
    if not np.all(np.isfinite(numeric_labels)) or not np.all(numeric_labels == np.floor(numeric_labels)):
        return unavailable_result(
            COMMITTOR_SCHEMA_VERSION,
            status="invalid",
            failure_reason="regime_labels_must_be_finite_integers",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )

    set_a, set_b = {int(v) for v in set_A}, {int(v) for v in set_B}
    if not set_a or not set_b or set_a & set_b:
        return unavailable_result(
            COMMITTOR_SCHEMA_VERSION,
            status="invalid",
            failure_reason="invalid_or_overlapping_regime_sets",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    if not np.isfinite(grid_min) or not np.isfinite(grid_max) or grid_min >= grid_max:
        return unavailable_result(
            COMMITTOR_SCHEMA_VERSION,
            status="invalid",
            failure_reason="invalid_reaction_coordinate_boundaries",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    if int(grid_resolution) < 65:
        return unavailable_result(
            COMMITTOR_SCHEMA_VERSION,
            status="not_testable",
            failure_reason="o2b_grid_resolution_below_minimum",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    if int(min_support_per_grid) < 2:
        return unavailable_result(
            COMMITTOR_SCHEMA_VERSION,
            status="invalid",
            failure_reason="min_support_per_grid_below_two",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    if not np.isfinite(diffusion_coefficient) or float(diffusion_coefficient) <= 0:
        return unavailable_result(
            COMMITTOR_SCHEMA_VERSION,
            status="invalid",
            failure_reason="diffusion_coefficient_must_be_positive",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )

    labels = numeric_labels.astype(int, copy=False)
    segment_starts = np.concatenate(
        [
            np.asarray([0], dtype=np.int32),
            np.flatnonzero(segments[1:] != segments[:-1]).astype(np.int32) + 1,
        ]
    )
    segment_values = segments[segment_starts]
    segment_has_a = np.logical_or.reduceat(
        np.isin(labels, list(set_a)),
        segment_starts,
    )
    segment_has_b = np.logical_or.reduceat(
        np.isin(labels, list(set_b)),
        segment_starts,
    )
    segment_has_first_hit = segment_has_a | segment_has_b
    n_transition_segments = int(np.sum(segment_has_first_hit))
    if (
        n_transition_segments < int(min_transition_segments)
        or not bool(np.any(segment_has_a))
        or not bool(np.any(segment_has_b))
    ):
        return unavailable_result(
            COMMITTOR_SCHEMA_VERSION,
            status="not_testable",
            failure_reason="insufficient_independent_A_B_transition_segments",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )

    source_idx, z_increments, dts = increment_pairs(
        reaction[:, None],
        t,
        segments,
        max_gap_sec=None,
    )
    if z_increments.shape[0] < int(min_support_per_grid):
        return unavailable_result(
            COMMITTOR_SCHEMA_VERSION,
            status="insufficient_support",
            failure_reason="insufficient_reaction_coordinate_increment_pairs",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    nominal_dt = float(np.median(dts))
    if not np.isfinite(nominal_dt) or nominal_dt <= 0:
        return unavailable_result(
            COMMITTOR_SCHEMA_VERSION,
            status="invalid",
            failure_reason="invalid_reaction_coordinate_timestep",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    relative_deviation = float(np.max(np.abs(dts - nominal_dt)) / nominal_dt)
    if relative_deviation > float(max_dt_relative_deviation):
        return unavailable_result(
            COMMITTOR_SCHEMA_VERSION,
            status="not_testable",
            failure_reason="materially_irregular_increment_timestep",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )

    query_grid = np.linspace(float(grid_min), float(grid_max), int(grid_resolution))
    source_reaction = reaction[source_idx]
    in_bounds = (source_reaction >= float(grid_min)) & (source_reaction <= float(grid_max))
    if not np.any(in_bounds):
        return unavailable_result(
            COMMITTOR_SCHEMA_VERSION,
            status="insufficient_support",
            failure_reason="no_reaction_coordinate_support_inside_boundaries",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    source_reaction = source_reaction[in_bounds]
    increments = z_increments[in_bounds, 0]
    increment_dt = dts[in_bounds]
    nearest_grid = np.argmin(np.abs(source_reaction[:, None] - query_grid[None, :]), axis=1)
    support = np.bincount(nearest_grid, minlength=query_grid.size).astype(np.int32)
    if np.any(support < int(min_support_per_grid)):
        return unavailable_result(
            COMMITTOR_SCHEMA_VERSION,
            status="insufficient_support",
            failure_reason="dense_query_grid_has_under_supported_points",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )

    drift_values = increments / increment_dt
    drift_sums = np.bincount(
        nearest_grid,
        weights=drift_values,
        minlength=query_grid.size,
    )
    drift_estimate = drift_sums / support

    potential_estimate = np.concatenate(
        [
            [0.0],
            np.cumsum(
                -0.5
                * (drift_estimate[1:] + drift_estimate[:-1])
                * np.diff(query_grid)
            ),
        ]
    )
    exponent = 2.0 * (
        potential_estimate - float(np.max(potential_estimate))
    ) / float(diffusion_coefficient)
    integrand = np.exp(exponent)
    cumulative = np.concatenate(
        [
            [0.0],
            np.cumsum(
                0.5
                * (integrand[1:] + integrand[:-1])
                * np.diff(query_grid)
            ),
        ]
    )
    total = float(cumulative[-1])
    if not np.isfinite(total) or total <= 0:
        return unavailable_result(
            COMMITTOR_SCHEMA_VERSION,
            status="invalid",
            failure_reason="nonfinite_committor_quadrature",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    q_grid = cumulative / total
    q_series = np.interp(reaction, query_grid, q_grid, left=0.0, right=1.0).astype(np.float32)
    finite_query = finite_state & np.isfinite(reaction)
    time_grid_index = np.argmin(
        np.abs(reaction[:, None] - query_grid[None, :]),
        axis=1,
    )
    time_support = np.where(
        (reaction >= float(grid_min)) & (reaction <= float(grid_max)),
        support[time_grid_index],
        0,
    ).astype(np.int32)
    q_series[finite_query & np.isin(labels, list(set_a))] = 0.0
    q_series[finite_query & np.isin(labels, list(set_b))] = 1.0
    outcomes = _next_hit_outcomes(labels, segments, set_a, set_b)
    resolved = np.isfinite(outcomes) & finite_query
    valid = np.isfinite(q_series) & finite_query
    if int(np.sum(valid)) < max(1, int(np.ceil(float(min_valid_fraction) * x.shape[0]))):
        return unavailable_result(
            COMMITTOR_SCHEMA_VERSION,
            status="insufficient_support",
            failure_reason="insufficient_valid_committor_support",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )

    return attach_grain_for_schema(attach_certificate({
        "schema_version": COMMITTOR_SCHEMA_VERSION,
        "computation_status": "computed",
        "failure_reason": None,
        "series": {
            "q_A_to_B": q_series,
            "q_hat": q_series,
            "q_grid": q_grid.astype(np.float32),
            "query_grid": query_grid.astype(np.float32),
            "reaction_coordinate": reaction.astype(np.float32),
            "resolved_first_hit_outcome": outcomes.astype(np.float32),
            "support_count": time_support,
            "grid_support_count": support,
            "valid": valid.astype(np.int8),
            "drift_estimate_grid": drift_estimate.astype(np.float32),
        },
        "summary": {
            "n_valid_timepoints": int(np.sum(valid)),
            "n_resolved_first_hit_outcomes": int(np.sum(resolved)),
            "n_independent_A_B_transition_segments": n_transition_segments,
            "n_segments_with_A_first_hit": int(np.sum(segment_has_a)),
            "n_segments_with_B_first_hit": int(np.sum(segment_has_b)),
            "grid_resolution": int(query_grid.size),
            "min_support_per_grid": int(np.min(support)),
            "nominal_dt_sec": nominal_dt,
            "max_dt_relative_deviation": relative_deviation,
            "grid_min": float(grid_min),
            "grid_max": float(grid_max),
            "diffusion_coefficient": float(diffusion_coefficient),
        },
        "provenance": build_provenance(
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
            time_semantics="within_segment_one_step_local_law_dense_grid_quadrature",
            estimator="local_law_dense_grid_o2b",
            settings={
                "set_A": sorted(set_a),
                "set_B": sorted(set_b),
                "reaction_coordinate_name": str(reaction_coordinate_name),
                "grid_resolution": int(query_grid.size),
                "grid_min": float(grid_min),
                "grid_max": float(grid_max),
                "diffusion_coefficient": float(diffusion_coefficient),
                "min_samples": int(min_samples),
                "min_support_per_grid": int(min_support_per_grid),
                "min_transition_segments": int(min_transition_segments),
                "first_hit_label_semantics": "one_terminal_A_or_B_label_per_independent_segment",
                "segment_id_supplied": segment_id is not None,
                "n_segments": int(segment_values.size),
                "gap_policy": "segment_id_breaks_increment_pairs",
                "max_dt_relative_deviation": float(max_dt_relative_deviation),
            },
        ),
    }))
