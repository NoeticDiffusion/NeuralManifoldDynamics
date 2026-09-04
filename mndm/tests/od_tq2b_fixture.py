"""Truth-known OD-TQ2b trajectories and references.

This module is validation infrastructure only. It does not define a production
measurement or alter the MNPS contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


X_A = -0.8
X_B = 0.8
X0 = 0.0
DT = 1e-3
GRID_RESOLUTION = 65


@dataclass(frozen=True)
class TruthSystem:
    system_id: str
    k: float
    sigma: float
    tilt: float = 0.0
    bump_center: float | None = None
    bump_half_width: float | None = None
    bump_amplitude: float = 0.0

    @property
    def diffusion_coefficient(self) -> float:
        return self.sigma**2

    def potential(self, x: np.ndarray | float) -> np.ndarray:
        scalar_input = np.asarray(x).ndim == 0
        values = np.atleast_1d(np.asarray(x, dtype=float))
        result = self.k * (np.square(values) - 1.0) ** 2 + self.tilt * values
        if self.bump_center is not None and self.bump_half_width is not None:
            xi = (values - self.bump_center) / self.bump_half_width
            mask = np.abs(xi) < 1.0
            bump = np.zeros_like(values)
            xi_active = xi[mask]
            bump[mask] = self.bump_amplitude * np.exp(
                1.0 - 1.0 / (1.0 - np.square(xi_active))
            )
            result = result + bump
        return result[0] if scalar_input else result

    def drift(self, x: np.ndarray | float) -> np.ndarray:
        scalar_input = np.asarray(x).ndim == 0
        values = np.atleast_1d(np.asarray(x, dtype=float))
        result = 4.0 * self.k * values * (1.0 - np.square(values)) - self.tilt
        if self.bump_center is not None and self.bump_half_width is not None:
            xi = (values - self.bump_center) / self.bump_half_width
            mask = np.abs(xi) < 1.0
            xi_active = xi[mask]
            bump = self.bump_amplitude * np.exp(
                1.0 - 1.0 / (1.0 - np.square(xi_active))
            )
            bump_derivative = (
                -2.0
                * xi_active
                * bump
                / np.square(1.0 - np.square(xi_active))
                / self.bump_half_width
            )
            result = result.copy()
            result[mask] -= bump_derivative
        return result[0] if scalar_input else result


TRUTH_SYSTEMS: tuple[TruthSystem, ...] = (
    TruthSystem(system_id="T0_symmetric", k=0.5, sigma=0.5),
    TruthSystem(
        system_id="T1_remote_barrier",
        k=0.5,
        sigma=0.5,
        bump_center=0.55,
        bump_half_width=0.30,
        bump_amplitude=0.40,
    ),
    TruthSystem(system_id="T2_tilted", k=0.5, sigma=0.5, tilt=0.12),
    TruthSystem(system_id="T3_low_diffusion", k=0.5, sigma=0.35),
)


def exact_committor(system: TruthSystem, query: np.ndarray) -> np.ndarray:
    """Compute the 1-D gradient-system committor by high-resolution quadrature."""
    fine_grid = np.linspace(X_A, X_B, 40_001, dtype=float)
    potential = system.potential(fine_grid)
    weights = np.exp(2.0 * (potential - np.max(potential)) / system.diffusion_coefficient)
    cumulative = np.concatenate(
        [
            [0.0],
            np.cumsum(
                0.5
                * (weights[1:] + weights[:-1])
                * np.diff(fine_grid)
            ),
        ]
    )
    truth_grid = cumulative / cumulative[-1]
    return np.interp(np.asarray(query, dtype=float), fine_grid, truth_grid, left=0.0, right=1.0)


def monte_carlo_committor(
    system: TruthSystem,
    *,
    start: float = X0,
    seed: int,
    n_paths: int = 256,
    max_steps: int = 60_000,
) -> dict[str, float | int]:
    """Independent first-passage reference for one starting coordinate."""
    rng = _rng(seed, TRUTH_SYSTEMS.index(system), 30)
    current = np.full(int(n_paths), float(start), dtype=float)
    active = np.ones(int(n_paths), dtype=bool)
    outcomes = np.full(int(n_paths), -1, dtype=np.int8)
    for _ in range(int(max_steps)):
        if not np.any(active):
            break
        active_indices = np.flatnonzero(active)
        current[active_indices] += (
            system.drift(current[active_indices]) * DT
            + system.sigma * np.sqrt(DT) * rng.normal(size=active_indices.size)
        )
        hit_a = current[active_indices] <= X_A
        hit_b = current[active_indices] >= X_B
        if np.any(hit_a):
            outcomes[active_indices[hit_a]] = 0
            active[active_indices[hit_a]] = False
        if np.any(hit_b):
            outcomes[active_indices[hit_b]] = 1
            active[active_indices[hit_b]] = False
    resolved = outcomes >= 0
    n_resolved = int(np.sum(resolved))
    n_b = int(np.sum(outcomes[resolved] == 1))
    q_mc = float(n_b / n_resolved) if n_resolved else float("nan")
    standard_error = (
        float(np.sqrt(q_mc * (1.0 - q_mc) / n_resolved))
        if n_resolved and np.isfinite(q_mc)
        else float("nan")
    )
    return {
        "q_mc": q_mc,
        "q_mc_se": standard_error,
        "n_resolved": n_resolved,
        "n_censored": int(np.sum(~resolved)),
    }


def local_match_audit() -> dict[str, float]:
    """Return the protected T0/T1 local-law differences at x0."""
    t0, t1 = TRUTH_SYSTEMS[:2]
    h = 1e-5
    return {
        "drift_abs_diff": float(abs(t0.drift(X0) - t1.drift(X0))),
        "jacobian_abs_diff": float(
            abs(
                (t0.drift(X0 + h) - t0.drift(X0 - h))
                - (t1.drift(X0 + h) - t1.drift(X0 - h))
            )
            / (2.0 * h)
        ),
        "diffusion_abs_diff": float(
            abs(t0.diffusion_coefficient - t1.diffusion_coefficient)
        ),
    }


def _rng(seed: int, system_index: int, stream: int, *parts: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), int(system_index), int(stream), *map(int, parts)])
    )


def build_truth_trajectories(
    system: TruthSystem,
    *,
    seed: int,
    support_per_grid: int = 2_048,
    n_first_passage: int = 256,
    max_steps: int = 60_000,
) -> dict[str, np.ndarray | str | float | int]:
    """Build explicit local-law and first-passage segments for one truth system.

    Local-law probe segments have no terminal label and supply dense support.
    First-passage segments have one terminal A/B label and supply independent
    transition outcomes. Both are real Euler-Maruyama trajectory segments with
    disjoint segment IDs.
    """
    query_grid = np.linspace(X_A, X_B, GRID_RESOLUTION, dtype=float)
    state_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    segment_parts: list[np.ndarray] = []
    next_segment = 0

    for grid_index, value in enumerate(query_grid):
        rng = _rng(seed, TRUTH_SYSTEMS.index(system), 10, grid_index)
        increments = (
            float(system.drift(value)) * DT
            + system.sigma * np.sqrt(DT) * rng.normal(size=support_per_grid)
        )
        source = np.full(support_per_grid, value, dtype=float)
        target = source + increments
        pair_values = np.column_stack([source, target]).reshape(-1)
        state_parts.append(
            np.column_stack(
                [
                    pair_values,
                    np.zeros(2 * support_per_grid, dtype=float),
                    np.zeros(2 * support_per_grid, dtype=float),
                ]
            )
        )
        time_parts.append(
            np.tile(np.asarray([0.0, DT], dtype=float), support_per_grid)
        )
        label_parts.append(np.full(2 * support_per_grid, -1, dtype=np.int8))
        segment_parts.append(
            np.repeat(
                np.arange(next_segment, next_segment + support_per_grid, dtype=np.int32),
                2,
            )
        )
        next_segment += support_per_grid

    first_passage_rng = _rng(seed, TRUTH_SYSTEMS.index(system), 20)
    path_values = np.full(
        (n_first_passage, max_steps + 1),
        np.nan,
        dtype=float,
    )
    path_values[:, 0] = X0
    current = np.full(n_first_passage, X0, dtype=float)
    active = np.ones(n_first_passage, dtype=bool)
    path_lengths = np.full(n_first_passage, max_steps + 1, dtype=np.int32)
    terminal_labels = np.full(n_first_passage, -1, dtype=np.int8)
    for step in range(1, max_steps + 1):
        active_indices = np.flatnonzero(active)
        if active_indices.size == 0:
            break
        current[active_indices] += (
            system.drift(current[active_indices]) * DT
            + system.sigma
            * np.sqrt(DT)
            * first_passage_rng.normal(size=active_indices.size)
        )
        hit_a = current[active_indices] <= X_A
        hit_b = current[active_indices] >= X_B
        current[active_indices[hit_a]] = X_A
        current[active_indices[hit_b]] = X_B
        path_values[active_indices, step] = current[active_indices]
        finished = hit_a | hit_b
        if np.any(finished):
            finished_indices = active_indices[finished]
            path_lengths[finished_indices] = step + 1
            terminal_labels[finished_indices[hit_a[finished]]] = 0
            terminal_labels[finished_indices[hit_b[finished]]] = 1
            active[finished_indices] = False

    for path_index in range(n_first_passage):
        length = int(path_lengths[path_index])
        values = path_values[path_index, :length]
        labels = np.full(length, -1, dtype=np.int8)
        if terminal_labels[path_index] >= 0:
            labels[-1] = terminal_labels[path_index]
        state_parts.append(
            np.column_stack(
                [
                    values,
                    np.zeros(len(values), dtype=float),
                    np.zeros(len(values), dtype=float),
                ]
            )
        )
        time_parts.append(np.arange(length, dtype=float) * DT)
        label_parts.append(labels)
        segment_parts.append(
            np.full(len(values), next_segment, dtype=np.int32)
        )
        next_segment += 1

    return {
        "state": np.concatenate(state_parts, axis=0),
        "time": np.concatenate(time_parts, axis=0),
        "reaction_coordinate": np.concatenate(
            [part[:, 0] for part in state_parts], axis=0
        ),
        "regime_labels": np.concatenate(label_parts, axis=0),
        "segment_id": np.concatenate(segment_parts, axis=0),
        "system_id": system.system_id,
        "diffusion_coefficient": system.diffusion_coefficient,
        "n_segments": next_segment,
    }


def truth_metrics(
    system: TruthSystem,
    q_grid: np.ndarray,
) -> dict[str, float]:
    """Score an adapter grid against the independent quadrature truth."""
    query_grid = np.linspace(X_A, X_B, GRID_RESOLUTION, dtype=float)
    truth = exact_committor(system, query_grid)
    estimate = np.asarray(q_grid, dtype=float)
    error = estimate - truth
    return {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "max_abs_error": float(np.max(np.abs(error))),
        "q_x0_truth": float(exact_committor(system, np.asarray([X0]))[0]),
        "q_x0_estimate": float(estimate[GRID_RESOLUTION // 2]),
    }
