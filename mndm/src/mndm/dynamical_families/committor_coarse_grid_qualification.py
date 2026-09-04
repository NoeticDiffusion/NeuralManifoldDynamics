"""NMD-local truth-known coarse-grid committor qualification.

This module is validation infrastructure, not a production estimator. It
provides two independent truth systems, deterministic trajectories, and an
audit-only O2b replay that permits coarse grids without changing the
production ``grid_resolution >= 65`` guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.integrate import quad

from .contracts import COMMITTOR_SCHEMA_VERSION, build_provenance
from .validity import increment_pairs, validate_trajectory

TRUTH_A = -0.8
TRUTH_B = 0.8
TRUTH_X0 = 0.0
CANDIDATE_GRIDS = (9, 17, 33, 65)
SUPPORT_FLOOR = 64
MIN_TRANSITION_SEGMENTS = 20
MAX_DT_RELATIVE_DEVIATION = 0.05
DEFAULT_DT = 1e-3
DEFAULT_N_LOCAL_PER_GRID_POINT = 128
DEFAULT_N_FIRST_PASSAGE = 128
DEFAULT_T_MAX_SEC = 60.0


@dataclass(frozen=True)
class TruthSystem:
    """A scalar absorbing diffusion with an independent scale-function truth."""

    system_id: str
    sigma: float
    drift_kind: str
    mu: float = 0.0

    @property
    def diffusion_coefficient(self) -> float:
        return float(self.sigma**2)

    def drift(self, x: np.ndarray | float) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        if self.drift_kind == "constant":
            return np.full_like(values, self.mu, dtype=float)
        if self.drift_kind == "curved":
            return (
                0.55 * values
                - 0.10
                + 0.20 * np.sin(np.pi * values / 1.6)
            )
        raise ValueError(f"unknown_truth_drift:{self.drift_kind}")

    def drift_primitive(self, x: np.ndarray | float) -> np.ndarray:
        """Return integral_a^x b(z) dz for the scale-function truth."""
        values = np.asarray(x, dtype=float)
        if self.drift_kind == "constant":
            return self.mu * (values - TRUTH_A)
        if self.drift_kind == "curved":
            return (
                0.275 * (values**2 - TRUTH_A**2)
                - 0.10 * (values - TRUTH_A)
                + 0.20
                * (1.6 / np.pi)
                * (
                    np.cos(np.pi * TRUTH_A / 1.6)
                    - np.cos(np.pi * values / 1.6)
                )
            )
        raise ValueError(f"unknown_truth_drift:{self.drift_kind}")

    def exact_committor(self, query: Sequence[float]) -> np.ndarray:
        """Evaluate the analytic/scale-function reference independently."""
        values = np.asarray(query, dtype=float).reshape(-1)
        if self.drift_kind == "constant" and abs(self.mu) > 0:
            numerator = 1.0 - np.exp(
                -2.0 * self.mu * (values - TRUTH_A) / self.diffusion_coefficient
            )
            denominator = 1.0 - np.exp(
                -2.0 * self.mu * (TRUTH_B - TRUTH_A)
                / self.diffusion_coefficient
            )
            return numerator / denominator
        if self.drift_kind == "constant":
            return (values - TRUTH_A) / (TRUTH_B - TRUTH_A)

        def weight(point: float) -> float:
            return float(
                np.exp(
                    -2.0
                    * float(self.drift_primitive(point))
                    / self.diffusion_coefficient
                )
            )

        denominator, denominator_error = quad(
            weight,
            TRUTH_A,
            TRUTH_B,
            epsabs=1e-12,
            epsrel=1e-12,
            limit=200,
        )
        if (
            not np.isfinite(denominator)
            or denominator <= 0
            or denominator_error > 1e-10
        ):
            raise ValueError("truth_quadrature_not_converged")
        result = np.empty_like(values)
        for index, value in enumerate(values):
            if value <= TRUTH_A:
                result[index] = 0.0
            elif value >= TRUTH_B:
                result[index] = 1.0
            else:
                numerator, error = quad(
                    weight,
                    TRUTH_A,
                    float(value),
                    epsabs=1e-12,
                    epsrel=1e-12,
                    limit=200,
                )
                if error > 1e-10 or not np.isfinite(numerator):
                    raise ValueError("truth_quadrature_not_converged")
                result[index] = numerator / denominator
        return result


TRUTH_SYSTEMS: tuple[TruthSystem, ...] = (
    TruthSystem(
        system_id="TQ_const_nonzero_drift",
        sigma=0.50,
        drift_kind="constant",
        mu=0.18,
    ),
    TruthSystem(
        system_id="TQ_curved_scale_function",
        sigma=0.45,
        drift_kind="curved",
    ),
)


def _rng(
    seed: int,
    system_index: int,
    grid_resolution: int,
    stream_id: int,
    *parts: int,
) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence(
            [
                int(seed),
                int(system_index),
                int(grid_resolution),
                int(stream_id),
                *[int(part) for part in parts],
            ]
        )
    )


def _append_local_probe_segments(
    *,
    system: TruthSystem,
    grid_resolution: int,
    seed: int,
    n_local_per_grid_point: int,
    dt: float,
    state_parts: list[np.ndarray],
    time_parts: list[np.ndarray],
    label_parts: list[np.ndarray],
    segment_parts: list[np.ndarray],
    next_segment: int,
) -> int:
    query_grid = np.linspace(TRUTH_A, TRUTH_B, int(grid_resolution))
    for grid_index, source_value in enumerate(query_grid):
        rng = _rng(seed, TRUTH_SYSTEMS.index(system), grid_resolution, 10, grid_index)
        increments = (
            float(system.drift(source_value)) * dt
            + system.sigma * np.sqrt(dt) * rng.normal(size=n_local_per_grid_point)
        )
        source = np.full(n_local_per_grid_point, source_value, dtype=float)
        target = source + increments
        state_parts.append(np.column_stack([source, target]).reshape(-1, 1))
        time_parts.append(np.tile(np.asarray([0.0, dt], dtype=float), n_local_per_grid_point))
        label_parts.append(np.full(2 * n_local_per_grid_point, -1, dtype=np.int8))
        segment_parts.append(
            np.repeat(
                np.arange(
                    next_segment,
                    next_segment + n_local_per_grid_point,
                    dtype=np.int32,
                ),
                2,
            )
        )
        next_segment += n_local_per_grid_point
    return next_segment


def _append_first_passage_segments(
    *,
    system: TruthSystem,
    grid_resolution: int,
    seed: int,
    n_first_passage: int,
    dt: float,
    t_max_sec: float,
    state_parts: list[np.ndarray],
    time_parts: list[np.ndarray],
    label_parts: list[np.ndarray],
    segment_parts: list[np.ndarray],
    next_segment: int,
) -> tuple[int, dict[str, int]]:
    """Simulate paths while retaining only in-bounds rows and terminal carriers."""
    rng = _rng(seed, TRUTH_SYSTEMS.index(system), grid_resolution, 20)
    max_steps = int(round(float(t_max_sec) / float(dt)))
    paths = [[float(TRUTH_X0)] for _ in range(int(n_first_passage))]
    times = [[0.0] for _ in range(int(n_first_passage))]
    labels = [[-1] for _ in range(int(n_first_passage))]
    current = np.full(int(n_first_passage), float(TRUTH_X0), dtype=float)
    active = np.ones(int(n_first_passage), dtype=bool)
    for step in range(max_steps):
        active_indices = np.flatnonzero(active)
        if active_indices.size == 0:
            break
        next_values = current[active_indices] + (
            system.drift(current[active_indices]) * dt
            + system.sigma * np.sqrt(dt) * rng.normal(size=active_indices.size)
        )
        for local_index, path_index in enumerate(active_indices):
            value = float(next_values[local_index])
            if value <= TRUTH_A or value >= TRUTH_B:
                labels[int(path_index)][-1] = 0 if value <= TRUTH_A else 1
                active[int(path_index)] = False
                continue
            current[int(path_index)] = value
            paths[int(path_index)].append(value)
            times[int(path_index)].append(float(step + 1) * dt)
            labels[int(path_index)].append(-1)
    for path_index in range(int(n_first_passage)):
        state_parts.append(np.asarray(paths[path_index], dtype=float)[:, None])
        time_parts.append(np.asarray(times[path_index], dtype=float))
        label_parts.append(np.asarray(labels[path_index], dtype=np.int8))
        segment_parts.append(
            np.full(
                len(paths[path_index]),
                next_segment,
                dtype=np.int32,
            )
        )
        next_segment += 1
    return next_segment, {
        "n_paths": int(n_first_passage),
        "n_a": int(sum(labels[index][-1] == 0 for index in range(len(labels)))),
        "n_b": int(sum(labels[index][-1] == 1 for index in range(len(labels)))),
        "n_censored": int(sum(labels[index][-1] == -1 for index in range(len(labels)))),
        "appended_absorbing_boundary_rows": 0,
    }


def build_truth_trajectories(
    system: TruthSystem,
    *,
    seed: int,
    grid_resolution: int,
    n_local_per_grid_point: int = DEFAULT_N_LOCAL_PER_GRID_POINT,
    n_first_passage: int = DEFAULT_N_FIRST_PASSAGE,
    dt: float = DEFAULT_DT,
    t_max_sec: float = DEFAULT_T_MAX_SEC,
) -> dict[str, Any]:
    """Build deterministic audit-only local probes and first-passage paths."""
    state_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    segment_parts: list[np.ndarray] = []
    next_segment = 0
    next_segment = _append_local_probe_segments(
        system=system,
        grid_resolution=grid_resolution,
        seed=seed,
        n_local_per_grid_point=n_local_per_grid_point,
        dt=dt,
        state_parts=state_parts,
        time_parts=time_parts,
        label_parts=label_parts,
        segment_parts=segment_parts,
        next_segment=next_segment,
    )
    next_segment, first_passage_summary = _append_first_passage_segments(
        system=system,
        grid_resolution=grid_resolution,
        seed=seed,
        n_first_passage=n_first_passage,
        dt=dt,
        t_max_sec=t_max_sec,
        state_parts=state_parts,
        time_parts=time_parts,
        label_parts=label_parts,
        segment_parts=segment_parts,
        next_segment=next_segment,
    )
    return {
        "state": np.concatenate(state_parts, axis=0),
        "time": np.concatenate(time_parts),
        "regime_labels": np.concatenate(label_parts),
        "segment_id": np.concatenate(segment_parts),
        "truth_system_id": system.system_id,
        "seed": int(seed),
        "grid_resolution": int(grid_resolution),
        "diffusion_coefficient": system.diffusion_coefficient,
        "first_passage_summary": first_passage_summary,
        "appended_absorbing_boundary_rows": 0,
    }


def estimate_audit_only_coarse_grid(
    state: np.ndarray,
    time: np.ndarray,
    regime_labels: np.ndarray,
    *,
    set_A: Sequence[int],
    set_B: Sequence[int],
    grid_min: float,
    grid_max: float,
    diffusion_coefficient: float,
    segment_id: np.ndarray,
    grid_resolution: int,
    min_support_per_grid: int = SUPPORT_FLOOR,
    min_transition_segments: int = MIN_TRANSITION_SEGMENTS,
    max_dt_relative_deviation: float = MAX_DT_RELATIVE_DEVIATION,
) -> dict[str, Any]:
    """Run the q-producing audit path without the production G>=65 guard."""
    x, t, segments, finite_state, failure = validate_trajectory(
        state,
        time,
        min_samples=30,
        segment_id=segment_id,
    )
    if failure or x is None or t is None or segments is None or finite_state is None:
        return {
            "computation_status": "not_testable",
            "failure_reason": failure or "invalid_trajectory",
        }
    if not bool(np.all(finite_state)):
        return {
            "computation_status": "not_testable",
            "failure_reason": "nonfinite_state_or_time_support",
        }
    labels = np.asarray(regime_labels).reshape(-1)
    if labels.size != x.shape[0]:
        return {
            "computation_status": "invalid",
            "failure_reason": "regime_label_shape_mismatch",
        }
    if not np.all(np.isfinite(labels)):
        return {
            "computation_status": "invalid",
            "failure_reason": "regime_labels_must_be_finite",
        }
    if (
        not np.isfinite(grid_min)
        or not np.isfinite(grid_max)
        or grid_min >= grid_max
        or int(grid_resolution) not in CANDIDATE_GRIDS
    ):
        return {
            "computation_status": "invalid",
            "failure_reason": "invalid_coarse_grid_contract",
        }
    if not np.isfinite(diffusion_coefficient) or diffusion_coefficient <= 0:
        return {
            "computation_status": "invalid",
            "failure_reason": "diffusion_coefficient_must_be_positive",
        }
    set_a, set_b = {int(value) for value in set_A}, {int(value) for value in set_B}
    if not set_a or not set_b or set_a & set_b:
        return {
            "computation_status": "invalid",
            "failure_reason": "invalid_or_overlapping_regime_sets",
        }
    segment_starts = np.concatenate(
        [
            np.asarray([0], dtype=np.int32),
            np.flatnonzero(segments[1:] != segments[:-1]).astype(np.int32) + 1,
        ]
    )
    segment_has_a = np.logical_or.reduceat(
        np.isin(labels, list(set_a)),
        segment_starts,
    )
    segment_has_b = np.logical_or.reduceat(
        np.isin(labels, list(set_b)),
        segment_starts,
    )
    n_transition_segments = int(np.sum(segment_has_a | segment_has_b))
    if (
        n_transition_segments < int(min_transition_segments)
        or not bool(np.any(segment_has_a))
        or not bool(np.any(segment_has_b))
    ):
        return {
            "computation_status": "not_testable",
            "failure_reason": "insufficient_independent_A_B_transition_segments",
        }
    source_idx, increments, dts = increment_pairs(
        x,
        t,
        segments,
        max_gap_sec=None,
    )
    if increments.shape[0] < int(min_support_per_grid):
        return {
            "computation_status": "not_testable",
            "failure_reason": "insufficient_reaction_coordinate_increment_pairs",
        }
    nominal_dt = float(np.median(dts))
    relative_deviation = float(
        np.max(np.abs(dts - nominal_dt)) / nominal_dt
    )
    if relative_deviation > float(max_dt_relative_deviation):
        return {
            "computation_status": "not_testable",
            "failure_reason": "materially_irregular_increment_timestep",
        }
    query_grid = np.linspace(float(grid_min), float(grid_max), int(grid_resolution))
    source_reaction = x[source_idx, 0]
    in_bounds = (
        (source_reaction >= float(grid_min))
        & (source_reaction <= float(grid_max))
    )
    if not np.any(in_bounds):
        return {
            "computation_status": "not_testable",
            "failure_reason": "no_reaction_coordinate_support_inside_boundaries",
        }
    source_reaction = source_reaction[in_bounds]
    local_increments = increments[in_bounds, 0]
    local_dts = dts[in_bounds]
    nearest_grid = np.argmin(
        np.abs(source_reaction[:, None] - query_grid[None, :]),
        axis=1,
    )
    support = np.bincount(
        nearest_grid,
        minlength=query_grid.size,
    ).astype(np.int32)
    if np.any(support < int(min_support_per_grid)):
        return {
            "computation_status": "not_testable",
            "failure_reason": "dense_query_grid_has_under_supported_points",
            "query_grid": query_grid,
            "grid_support_count": support,
            "summary": {
                "grid_resolution": int(grid_resolution),
                "min_support_per_grid": int(np.min(support)),
                "n_transition_segments": n_transition_segments,
                "nominal_dt_sec": nominal_dt,
                "max_dt_relative_deviation": relative_deviation,
            },
        }
    drift_estimate = np.asarray(
        [
            np.mean(
                local_increments[nearest_grid == grid_index]
                / local_dts[nearest_grid == grid_index]
            )
            for grid_index in range(query_grid.size)
        ],
        dtype=float,
    )
    potential = np.concatenate(
        [
            [0.0],
            np.cumsum(
                -0.5
                * (drift_estimate[1:] + drift_estimate[:-1])
                * np.diff(query_grid)
            ),
        ]
    )
    exponent = 2.0 * (potential - np.max(potential)) / float(diffusion_coefficient)
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
        return {
            "computation_status": "invalid",
            "failure_reason": "nonfinite_committor_quadrature",
        }
    q_grid = cumulative / total
    if not np.all(np.isfinite(q_grid)):
        return {
            "computation_status": "invalid",
            "failure_reason": "nonfinite_q_grid",
        }
    return {
        "schema_version": COMMITTOR_SCHEMA_VERSION,
        "computation_status": "computed",
        "failure_reason": None,
        "series": {
            "q_grid": q_grid.astype(np.float64),
            "query_grid": query_grid.astype(np.float64),
            "grid_support_count": support,
            "drift_estimate_grid": drift_estimate.astype(np.float64),
        },
        "summary": {
            "grid_resolution": int(grid_resolution),
            "min_support_per_grid": int(np.min(support)),
            "n_transition_segments": n_transition_segments,
            "nominal_dt_sec": nominal_dt,
            "max_dt_relative_deviation": relative_deviation,
            "diffusion_coefficient": float(diffusion_coefficient),
            "appended_absorbing_boundary_rows": 0,
        },
        "provenance": build_provenance(
            coordinate_layer="truth_known_scalar",
            coordinate_names=["truth_known_x"],
            time_semantics="audit_only_within_segment_local_law",
            estimator="audit_only_local_law_dense_grid_o2b",
            settings={
                "grid_resolution": int(grid_resolution),
                "min_support_per_grid": int(min_support_per_grid),
                "min_transition_segments": int(min_transition_segments),
                "max_dt_relative_deviation": float(max_dt_relative_deviation),
                "set_A": sorted(set_a),
                "set_B": sorted(set_b),
                "audit_only": True,
                "production_guard_unchanged": True,
            },
        ),
    }


def score_truth_grid(
    estimate: dict[str, Any],
    system: TruthSystem,
) -> dict[str, float | bool]:
    """Compare one candidate q-grid with its independent truth."""
    if estimate.get("computation_status") != "computed":
        return {
            "truth_status": "not_scored",
            "truth_valid": False,
        }
    q_grid = np.asarray(estimate["series"]["q_grid"], dtype=float)
    query_grid = np.asarray(estimate["series"]["query_grid"], dtype=float)
    try:
        truth = system.exact_committor(query_grid)
    except (FloatingPointError, ValueError, OverflowError) as error:
        return {
            "truth_status": "not_testable",
            "truth_valid": False,
            "truth_failure_reason": str(error),
        }
    error = q_grid - truth
    return {
        "truth_status": "computed",
        "truth_valid": bool(np.all(np.isfinite(truth))),
        "rmse_q": float(np.sqrt(np.mean(error**2))),
        "mae_q": float(np.mean(np.abs(error))),
        "e_max": float(np.max(np.abs(error))),
        "endpoint_abs_error": float(
            max(abs(float(q_grid[0])), abs(float(q_grid[-1]) - 1.0))
        ),
        "q_range_min": float(np.min(q_grid)),
        "q_range_max": float(np.max(q_grid)),
        "min_q_difference": float(np.min(np.diff(q_grid))),
    }
