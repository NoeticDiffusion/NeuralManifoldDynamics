"""Synthetic fail-closed tests for OD-SLP-002A-NMD-TQ."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.dynamical_families.committor import (  # noqa: E402
    estimate_committor_local_law_dense_grid_o2b,
)
from mndm.dynamical_families.committor_coarse_grid_qualification import (  # noqa: E402
    CANDIDATE_GRIDS,
    SUPPORT_FLOOR,
    TRUTH_SYSTEMS,
    build_truth_trajectories,
    estimate_audit_only_coarse_grid,
    score_truth_grid,
)


def _fixture(grid_resolution: int = 9) -> dict[str, object]:
    return build_truth_trajectories(
        TRUTH_SYSTEMS[0],
        seed=15000,
        grid_resolution=grid_resolution,
        n_local_per_grid_point=64,
        n_first_passage=48,
        t_max_sec=10.0,
    )


def test_truth_systems_are_independent_and_monotone() -> None:
    query = np.linspace(-0.8, 0.8, 33)
    first = TRUTH_SYSTEMS[0].exact_committor(query)
    second = TRUTH_SYSTEMS[1].exact_committor(query)
    assert np.isclose(first[0], 0.0)
    assert np.isclose(first[-1], 1.0)
    assert np.isclose(second[0], 0.0)
    assert np.isclose(second[-1], 1.0)
    assert np.all(np.diff(first) >= 0)
    assert np.all(np.diff(second) >= 0)
    assert not np.allclose(first, second)


def test_audit_path_computes_all_candidate_grid_shapes() -> None:
    for grid in CANDIDATE_GRIDS:
        fixture = _fixture(grid)
        result = estimate_audit_only_coarse_grid(
            fixture["state"],
            fixture["time"],
            fixture["regime_labels"],
            set_A=[0],
            set_B=[1],
            grid_min=-0.8,
            grid_max=0.8,
            diffusion_coefficient=TRUTH_SYSTEMS[0].diffusion_coefficient,
            segment_id=fixture["segment_id"],
            grid_resolution=grid,
        )
        assert result["computation_status"] == "computed"
        assert result["series"]["q_grid"].shape == (grid,)
        assert result["series"]["grid_support_count"].shape == (grid,)
        assert np.min(result["series"]["grid_support_count"]) >= SUPPORT_FLOOR
        metrics = score_truth_grid(result, TRUTH_SYSTEMS[0])
        assert metrics["truth_valid"]


def test_terminal_carrier_does_not_append_absorbing_boundary() -> None:
    fixture = _fixture(9)
    assert fixture["appended_absorbing_boundary_rows"] == 0
    assert fixture["first_passage_summary"]["appended_absorbing_boundary_rows"] == 0
    result = estimate_audit_only_coarse_grid(
        fixture["state"],
        fixture["time"],
        fixture["regime_labels"],
        set_A=[0],
        set_B=[1],
        grid_min=-0.8,
        grid_max=0.8,
        diffusion_coefficient=TRUTH_SYSTEMS[0].diffusion_coefficient,
        segment_id=fixture["segment_id"],
        grid_resolution=9,
    )
    assert result["summary"]["appended_absorbing_boundary_rows"] == 0


def test_under_supported_point_fails_closed() -> None:
    fixture = _fixture(9)
    state = np.asarray(fixture["state"], dtype=float).copy()
    segment_id = np.asarray(fixture["segment_id"])
    time = np.asarray(fixture["time"], dtype=float)
    same_segment = segment_id[1:] == segment_id[:-1]
    positive_dt = np.diff(time) > 0
    source_values = state[:-1, 0]
    query_grid = np.linspace(-0.8, 0.8, 9)
    nearest = np.argmin(
        np.abs(source_values[:, None] - query_grid[None, :]),
        axis=1,
    )
    source_indices = np.flatnonzero(same_segment & positive_dt & (nearest == 0))
    # Remove every source assigned to the first Voronoi bin, including
    # first-passage sources; deleting only the local probes is insufficient.
    state[source_indices, 0] = -2.0
    result = estimate_audit_only_coarse_grid(
        state,
        fixture["time"],
        fixture["regime_labels"],
        set_A=[0],
        set_B=[1],
        grid_min=-0.8,
        grid_max=0.8,
        diffusion_coefficient=TRUTH_SYSTEMS[0].diffusion_coefficient,
        segment_id=segment_id,
        grid_resolution=9,
    )
    assert result["computation_status"] == "not_testable"
    assert result["failure_reason"] == "dense_query_grid_has_under_supported_points"
    assert result["grid_support_count"].shape == (9,)
    assert np.min(result["grid_support_count"]) < SUPPORT_FLOOR


def test_irregular_time_and_nonfinite_state_refuse() -> None:
    fixture = _fixture(9)
    irregular_time = np.asarray(fixture["time"], dtype=float).copy()
    irregular_time[1] += 0.001
    irregular = estimate_audit_only_coarse_grid(
        fixture["state"],
        irregular_time,
        fixture["regime_labels"],
        set_A=[0],
        set_B=[1],
        grid_min=-0.8,
        grid_max=0.8,
        diffusion_coefficient=TRUTH_SYSTEMS[0].diffusion_coefficient,
        segment_id=fixture["segment_id"],
        grid_resolution=9,
    )
    assert irregular["computation_status"] == "not_testable"
    assert irregular["failure_reason"] == "materially_irregular_increment_timestep"

    nonfinite_state = np.asarray(fixture["state"], dtype=float).copy()
    nonfinite_state[0, 0] = np.nan
    invalid = estimate_audit_only_coarse_grid(
        nonfinite_state,
        fixture["time"],
        fixture["regime_labels"],
        set_A=[0],
        set_B=[1],
        grid_min=-0.8,
        grid_max=0.8,
        diffusion_coefficient=TRUTH_SYSTEMS[0].diffusion_coefficient,
        segment_id=fixture["segment_id"],
        grid_resolution=9,
    )
    assert invalid["computation_status"] == "not_testable"


def test_production_guard_stays_at_65() -> None:
    fixture = _fixture(9)
    production = estimate_committor_local_law_dense_grid_o2b(
        fixture["state"],
        fixture["time"],
        fixture["state"][:, 0],
        fixture["regime_labels"],
        set_A=[0],
        set_B=[1],
        grid_min=-0.8,
        grid_max=0.8,
        diffusion_coefficient=TRUTH_SYSTEMS[0].diffusion_coefficient,
        segment_id=fixture["segment_id"],
        grid_resolution=33,
        min_support_per_grid=SUPPORT_FLOOR,
    )
    assert production["computation_status"] == "not_testable"
    assert production["failure_reason"] == "o2b_grid_resolution_below_minimum"


def test_fixture_replay_is_deterministic() -> None:
    first = _fixture(17)
    second = _fixture(17)
    for key in ("state", "time", "regime_labels", "segment_id"):
        assert np.array_equal(first[key], second[key])
