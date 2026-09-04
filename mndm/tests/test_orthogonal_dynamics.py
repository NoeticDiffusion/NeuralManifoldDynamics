"""Synthetic contract tests for dynamical-family estimators."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.dynamical_families.committor import estimate_committor
from mndm.dynamical_families.diffusion_geometry import estimate_local_diffusion_geometry
from mndm.dynamical_families.resilience import summarize_finite_amplitude_resilience
from mndm.dynamical_families import validity as validity_module
from mndm.dynamical_families.validity import chunked_nearest_neighbors


def test_mnps_namespace_preserves_legacy_chart_exports() -> None:
    from mndm.mnps import MNPS_AXIS_NAMES, mnps_9d_CANONICAL_ORDER

    assert MNPS_AXIS_NAMES == ("m", "d", "e")
    assert len(mnps_9d_CANONICAL_ORDER) == 9


def test_chunked_nearest_neighbors_matches_serial_argpartition_and_ties() -> None:
    references = np.asarray([[0.0], [0.0], [1.0], [2.0], [3.0]])
    queries = np.asarray([[0.0], [0.5], [2.2], [3.0]])
    k = 3

    expected_indices = np.vstack(
        [
            np.argpartition(
                np.sum((references - query) ** 2, axis=1),
                k - 1,
            )[:k]
            for query in queries
        ]
    )
    expected_distances = np.vstack(
        [
            np.sum((references - query) ** 2, axis=1)[indices]
            for query, indices in zip(queries, expected_indices)
        ]
    )

    for chunk_size in (1, 2, 4):
        indices, distances = chunked_nearest_neighbors(
            queries,
            references,
            k,
            chunk_size=chunk_size,
        )
        assert np.array_equal(indices, expected_indices)
        assert np.array_equal(distances, expected_distances)


def test_chunked_nearest_neighbors_memory_limiter_preserves_9d_identity(
    monkeypatch,
) -> None:
    rng = np.random.default_rng(20260820)
    queries = rng.normal(size=(17, 9))
    references = rng.normal(size=(101, 9))
    default_indices, default_distances = chunked_nearest_neighbors(
        queries,
        references,
        7,
        chunk_size=256,
    )

    one_query_budget = references.shape[0] * (references.shape[1] + 1) * references.itemsize
    monkeypatch.setattr(
        validity_module,
        "_KNN_DISTANCE_MEMORY_BUDGET_BYTES",
        one_query_budget,
    )
    limited_indices, limited_distances = chunked_nearest_neighbors(
        queries,
        references,
        7,
        chunk_size=256,
    )

    assert np.array_equal(limited_indices, default_indices)
    assert np.array_equal(limited_distances, default_distances)


def test_chunked_committor_radius_gate_preserves_labeled_overrides() -> None:
    n_segments, per_segment = 5, 30
    state = np.tile(np.linspace(-1.0, 1.0, per_segment), n_segments)[:, None]
    labels = np.full(n_segments * per_segment, -1, dtype=np.int8)
    labels[::per_segment] = 0
    labels[per_segment - 1 :: per_segment] = 1
    segment_id = np.repeat(np.arange(n_segments), per_segment)

    result = estimate_committor(
        state,
        np.arange(n_segments * per_segment, dtype=float),
        labels,
        set_A=[0],
        set_B=[1],
        segment_id=segment_id,
        neighborhood_k=30,
        min_support=20,
        min_transition_segments=2,
        max_neighborhood_radius=0.0,
        min_valid_fraction=0.01,
    )

    assert result["computation_status"] == "computed"
    assert np.all(result["series"]["support_count"] == 0)
    assert np.all(result["series"]["q_A_to_B"][labels == 0] == 0.0)
    assert np.all(result["series"]["q_A_to_B"][labels == 1] == 1.0)
    assert np.isnan(result["series"]["q_A_to_B"][labels == -1]).all()


def test_chunked_diffusion_radius_gate_marks_far_center_invalid() -> None:
    state = np.zeros((50, 1), dtype=float)
    state[-1, 0] = 100.0
    result = estimate_local_diffusion_geometry(
        state,
        np.arange(state.shape[0], dtype=float),
        neighborhood_k=10,
        min_samples=30,
        min_neighborhood_samples=10,
        max_neighborhood_radius=0.0,
        min_valid_fraction=0.1,
    )

    assert result["computation_status"] == "computed"
    assert result["series"]["valid"][-1] == 0
    assert result["series"]["support_count"][-1] == 0
    assert np.isnan(result["series"]["diffusion_tensor"][-1]).all()
    assert int(np.sum(result["series"]["valid"])) >= 49


def test_diffusion_geometry_recovers_isotropic_increment_covariance() -> None:
    rng = np.random.default_rng(7)
    n, dt, sigma = 800, 0.01, 0.4
    increments = rng.normal(scale=sigma * np.sqrt(dt), size=(n - 1, 2))
    state = np.vstack([np.zeros((1, 2)), np.cumsum(increments, axis=0)])
    result = estimate_local_diffusion_geometry(
        state,
        np.arange(n) * dt,
        neighborhood_k=n - 1,
        min_samples=100,
        min_neighborhood_samples=100,
    )

    assert result["computation_status"] == "computed"
    valid = result["series"]["valid"].astype(bool)
    estimated_trace = float(np.nanmean(result["series"]["D_total"][valid]))
    assert np.isclose(estimated_trace, 2 * sigma**2, rtol=0.2)
    assert np.isclose(float(np.nanmean(result["series"]["d_diff"][valid])), 2.0, rtol=0.15)
    assert result["summary"]["A_bD_computation_status"] == "not_testable"
    assert result["summary"]["R_b_over_a_computation_status"] == "not_testable"
    assert result["summary"]["drift_alignment_failure_reason"] == "independent_drift_not_supplied"
    assert result["summary"]["a_semantics"] == "raw_increment_covariance"
    assert result["summary"]["ratio_semantics"] == "not_applicable"
    assert result["provenance"]["settings"]["drift_mode"] == "not_supplied"
    assert result["provenance"]["settings"]["drift_residualization"] == "none"
    assert np.all(np.isnan(result["series"]["A_bD"]))
    assert np.all(np.isnan(result["series"]["R_b_over_a"]))


def test_diffusion_geometry_rejects_irregular_time_steps() -> None:
    state = np.arange(80, dtype=float)[:, None]
    time = np.arange(80, dtype=float)
    time[40:] += 1.0
    result = estimate_local_diffusion_geometry(
        state,
        time,
        neighborhood_k=30,
        min_samples=30,
        min_neighborhood_samples=10,
    )
    assert result["computation_status"] == "not_testable"
    assert result["failure_reason"] == "materially_irregular_increment_timestep"


def test_diffusion_geometry_recovers_anisotropy_and_drift_alignment() -> None:
    rng = np.random.default_rng(11)
    n, dt = 1000, 0.01
    drift = np.tile(np.array([1.0, 0.0]), (n, 1))
    aligned_noise = rng.normal(size=(n - 1, 2)) * np.sqrt(dt) * np.array([0.5, 0.1])
    aligned_state = np.vstack([np.zeros((1, 2)), np.cumsum(drift[:-1] * dt + aligned_noise, axis=0)])
    orthogonal_noise = rng.normal(size=(n - 1, 2)) * np.sqrt(dt) * np.array([0.1, 0.5])
    orthogonal_state = np.vstack([np.zeros((1, 2)), np.cumsum(drift[:-1] * dt + orthogonal_noise, axis=0)])
    common = dict(
        drift=drift,
        residualize_increments=False,
        drift_source="truth_known_chart_b",
        neighborhood_k=n - 1,
        min_samples=100,
        min_neighborhood_samples=100,
    )
    aligned = estimate_local_diffusion_geometry(aligned_state, np.arange(n) * dt, **common)
    orthogonal = estimate_local_diffusion_geometry(orthogonal_state, np.arange(n) * dt, **common)

    assert aligned["computation_status"] == orthogonal["computation_status"] == "computed"
    assert aligned["summary"]["A_bD_computation_status"] == "computed"
    assert aligned["summary"]["R_b_over_a_computation_status"] == "computed"
    assert aligned["summary"]["drift_alignment_failure_reason"] is None
    assert aligned["summary"]["a_semantics"] == "raw_increment_covariance"
    assert aligned["summary"]["ratio_semantics"] == "chart_velocity_to_increment_spread"
    assert aligned["provenance"]["settings"]["drift_mode"] == "alignment_only"
    assert aligned["provenance"]["settings"]["drift_residualization"] == "none"
    assert np.nanmean(aligned["series"]["c_diff"]) > 0.8
    assert np.nanmean(aligned["series"]["A_bD"]) > np.nanmean(orthogonal["series"]["A_bD"])


def test_committor_requires_and_uses_explicit_first_hit_labels() -> None:
    n_segments, per_segment = 5, 30
    state = np.tile(np.linspace(-1.0, 1.0, per_segment), n_segments)[:, None]
    labels = np.full(n_segments * per_segment, -1, dtype=np.int8)
    labels[::per_segment] = 0
    labels[per_segment - 1 :: per_segment] = 1
    segment_id = np.repeat(np.arange(n_segments), per_segment)
    result = estimate_committor(
        state,
        np.arange(n_segments * per_segment, dtype=float),
        labels,
        set_A=[0],
        set_B=[1],
        segment_id=segment_id,
        neighborhood_k=n_segments * per_segment,
        min_support=30,
    )
    assert result["computation_status"] == "computed"
    assert np.nanmean(result["series"]["q_A_to_B"]) > 0.9


def test_committor_refuses_single_labeled_transit() -> None:
    state = np.linspace(-1.0, 1.0, 60)[:, None]
    labels = np.full(60, -1, dtype=np.int8)
    labels[0], labels[-1] = 0, 1
    result = estimate_committor(
        state,
        np.arange(60, dtype=float),
        labels,
        set_A=[0],
        set_B=[1],
        neighborhood_k=30,
        min_support=20,
        min_transition_segments=2,
    )
    assert result["computation_status"] == "not_testable"
    assert result["failure_reason"] == "insufficient_independent_A_B_transition_segments"


def test_resilience_uses_observed_perturbation_outcomes() -> None:
    amplitudes = np.repeat(np.array([0.0, 1.0, 2.0]), 30)
    returned = np.concatenate([np.ones(30), np.ones(18), np.zeros(12), np.ones(9), np.zeros(21)])
    result = summarize_finite_amplitude_resilience(
        amplitudes,
        returned,
        min_trials_per_amplitude=20,
    )
    assert result["computation_status"] == "computed"
    assert np.isclose(result["amplitude_curve"][0]["basin_return_probability"], 1.0)
    assert result["summary"]["r50_discrete_first_bin_at_or_below_half"] == 2.0


def test_resilience_rejects_negative_recovery_times() -> None:
    result = summarize_finite_amplitude_resilience(
        np.zeros(20),
        np.ones(20),
        recovery_time_sec=-np.ones(20),
        min_trials_per_amplitude=20,
    )
    assert result["computation_status"] == "invalid"
    assert result["failure_reason"] == "negative_recovery_time"
