"""Shared lag-1 transition support for pooled drift and diffusion."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))

from mndm.dynamical_families.diffusion_geometry import estimate_local_diffusion_geometry
from mndm.dynamical_families.finite_lag_drift import (
    estimate_blocked_crossfit_mean_rate,
    estimate_conditional_mean_rate_level1,
    estimate_lag_diagnostics,
    _fill_conditional_mean,
)
from mndm.dynamical_families.measurement_register import (
    FORBIDDEN_EMBARGO_CLAIM_KEYS,
    MEASUREMENT_ID_CONDITIONAL_MEAN_RATE,
    PHYSICAL_CROSSFIT_SOURCE_IDX,
    PHYSICAL_DIFFUSION_SOURCE_IDX,
    PHYSICAL_POOLED_SOURCE_IDX,
    SUPPORT_OBJECT_LAG1,
    VARIANT_BLOCKED_CROSSFIT,
    VARIANT_POOLED,
    support_object_entry,
)
from mndm.dynamical_families.transition_support import (
    EMBARGO_SEMANTICS_INDEX_STEPS,
    RAW_WINDOW_SUPPORT_INDEPENDENCE_NOT_ESTABLISHED,
    build_transition_support,
)
from mndm.dynamical_families.validity import increment_pairs_at_lag, validate_trajectory
from mndm.pipeline.dynamical_families_export import build_dynamical_families_export

_B_TRUE = np.array([0.50, -0.25, 0.10], dtype=float)


def _linear_sde(
    *,
    seed: int = 7,
    n: int = 1800,
    dt: float = 0.01,
    sigma: float = 0.05,
    b: np.ndarray = _B_TRUE,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=sigma * np.sqrt(dt), size=(n - 1, 3))
    state = np.zeros((n, 3), dtype=float)
    for i in range(n - 1):
        state[i + 1] = state[i] + b * dt + noise[i]
    time = np.arange(n, dtype=float) * dt
    return state, time


def _estimator_kwargs() -> dict:
    return dict(
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
        neighborhood_k=20,
        min_samples=30,
        min_neighborhood_samples=10,
        max_dt_relative_deviation=0.05,
        min_valid_fraction=0.1,
        weight_mode="inverse_distance",
    )


def _diffusion_config(*, drift_family_enabled: bool) -> dict:
    return {
        "dynamical_families": {
            "enabled": True,
            "coordinate_layer": "subject_anchored",
            "diffusion": {
                "enabled": True,
                "neighborhood": {"k": 20},
                "min_samples": 30,
                "min_neighborhood_samples": 10,
            },
            "drift": {
                "enabled": drift_family_enabled,
                "neighborhood": {"k": 20},
                "min_samples": 30,
                "min_neighborhood_samples": 10,
                "weight_mode": "inverse_distance",
                "realized_velocity_level0": {"enabled": True},
                "conditional_mean_rate_level1": {
                    "pooled": {"enabled": True},
                    "blocked_crossfit": {"enabled": True, "n_blocks": 2, "embargo_steps": 4},
                    "lag_diagnostics": {
                        "enabled": True,
                        "lags": [1, 2, 4],
                        "consistency_rel_tol": 0.5,
                    },
                },
            },
        }
    }


def test_pooled_and_diffusion_share_lag1_source_idx_and_support_id() -> None:
    state, time = _linear_sde()
    segments = np.zeros(time.size, dtype=np.int32)
    expected_idx, _, _ = increment_pairs_at_lag(
        state, time, segments, lag=1, max_gap_sec=None
    )
    pooled = estimate_conditional_mean_rate_level1(state, time, **_estimator_kwargs())
    diffusion = estimate_local_diffusion_geometry(
        state,
        time,
        neighborhood_k=20,
        min_samples=30,
        min_neighborhood_samples=10,
        max_dt_relative_deviation=0.05,
    )
    assert pooled["computation_status"] == diffusion["computation_status"] == "computed"
    assert np.array_equal(pooled["series"]["source_idx"], expected_idx)
    assert np.array_equal(diffusion["series"]["source_idx"], expected_idx)
    assert pooled["summary"]["transition_support_id"] == diffusion["summary"]["transition_support_id"]
    assert pooled["summary"]["embargo_semantics"] == EMBARGO_SEMANTICS_INDEX_STEPS
    assert diffusion["summary"]["embargo_semantics"] == EMBARGO_SEMANTICS_INDEX_STEPS
    assert (
        pooled["summary"]["raw_window_support_independence"]
        == RAW_WINDOW_SUPPORT_INDEPENDENCE_NOT_ESTABLISHED
    )
    assert (
        diffusion["summary"]["raw_window_support_independence"]
        == RAW_WINDOW_SUPPORT_INDEPENDENCE_NOT_ESTABLISHED
    )
    support = build_transition_support(state, time, segments, lag=1)
    assert support.support_id == pooled["summary"]["transition_support_id"]
    assert support.failure_reason is None


def test_interior_nan_is_excluded_from_both_families() -> None:
    state, time = _linear_sde(n=400)
    state = state.copy()
    state[80] = np.nan
    segments = np.zeros(time.size, dtype=np.int32)
    expected_idx, _, _ = increment_pairs_at_lag(
        state, time, segments, lag=1, max_gap_sec=None
    )
    assert 79 not in set(expected_idx.tolist())
    assert 80 not in set(expected_idx.tolist())
    assert 78 in set(expected_idx.tolist())
    assert 81 in set(expected_idx.tolist())
    pooled = estimate_conditional_mean_rate_level1(state, time, **_estimator_kwargs())
    diffusion = estimate_local_diffusion_geometry(
        state,
        time,
        neighborhood_k=20,
        min_samples=30,
        min_neighborhood_samples=10,
    )
    assert pooled["computation_status"] == "computed"
    assert diffusion["computation_status"] == "computed"
    assert np.array_equal(pooled["series"]["source_idx"], expected_idx)
    assert np.array_equal(diffusion["series"]["source_idx"], expected_idx)


def test_crossfit_support_id_is_embargoed_subset() -> None:
    state, time = _linear_sde()
    pooled = estimate_conditional_mean_rate_level1(state, time, **_estimator_kwargs())
    crossfit = estimate_blocked_crossfit_mean_rate(
        state, time, embargo_steps=4, **_estimator_kwargs()
    )
    assert pooled["computation_status"] == crossfit["computation_status"] == "computed"
    assert crossfit["variant_id"] == VARIANT_BLOCKED_CROSSFIT
    assert crossfit["summary"]["embargo_semantics"] == EMBARGO_SEMANTICS_INDEX_STEPS
    assert crossfit["provenance"]["settings"]["embargo_semantics"] == EMBARGO_SEMANTICS_INDEX_STEPS
    assert (
        crossfit["summary"]["raw_window_support_independence"]
        == RAW_WINDOW_SUPPORT_INDEPENDENCE_NOT_ESTABLISHED
    )
    assert (
        crossfit["provenance"]["settings"]["raw_window_support_independence"]
        == RAW_WINDOW_SUPPORT_INDEPENDENCE_NOT_ESTABLISHED
    )
    pooled_idx = pooled["series"]["source_idx"]
    xfit_idx = crossfit["series"]["source_idx"]
    assert xfit_idx.size < pooled_idx.size
    assert set(xfit_idx.tolist()).issubset(set(pooled_idx.tolist()))
    union = np.sort(
        np.concatenate(
            [
                crossfit["summary"]["fold1_source_idx"],
                crossfit["summary"]["fold2_source_idx"],
            ]
        )
    )
    assert np.array_equal(xfit_idx, union)
    assert crossfit["summary"]["transition_support_id"] != pooled["summary"]["transition_support_id"]


def test_support_id_ignores_neighborhood_k() -> None:
    state, time = _linear_sde(n=600)
    kwargs = _estimator_kwargs()
    k20 = estimate_conditional_mean_rate_level1(state, time, **{**kwargs, "neighborhood_k": 20})
    k25 = estimate_conditional_mean_rate_level1(state, time, **{**kwargs, "neighborhood_k": 25})
    assert k20["computation_status"] == k25["computation_status"] == "computed"
    assert k20["summary"]["transition_support_id"] == k25["summary"]["transition_support_id"]
    assert np.array_equal(k20["series"]["source_idx"], k25["series"]["source_idx"])


def test_estimator_arrays_match_independent_pair_selection() -> None:
    state, time = _linear_sde()
    kwargs = _estimator_kwargs()
    pooled = estimate_conditional_mean_rate_level1(state, time, **kwargs)
    diffusion = estimate_local_diffusion_geometry(
        state,
        time,
        neighborhood_k=20,
        min_samples=30,
        min_neighborhood_samples=10,
        max_dt_relative_deviation=0.05,
    )
    x, t, segments, finite_state, _ = validate_trajectory(
        state, time, min_samples=30, segment_id=None
    )
    assert x is not None and t is not None and segments is not None and finite_state is not None
    source_idx, increments, dts = increment_pairs_at_lag(
        x, t, segments, lag=1, max_gap_sec=None
    )
    nominal_dt = float(np.median(dts))
    filled = _fill_conditional_mean(
        x,
        t,
        np.flatnonzero(finite_state),
        ref_source_idx=source_idx,
        ref_increments=increments,
        neighborhood_k=20,
        minimum_support=max(10, 3 * x.shape[1] + 1),
        max_neighborhood_radius=None,
        min_temporal_span_sec=None,
        weight_mode="inverse_distance",
        nominal_dt=nominal_dt,
        epsilon=1e-12,
        store_neighbor_source_idx=False,
    )
    assert np.array_equal(pooled["series"]["b_hat"], filled["b_hat"], equal_nan=True)
    again = estimate_local_diffusion_geometry(
        state,
        time,
        neighborhood_k=20,
        min_samples=30,
        min_neighborhood_samples=10,
        max_dt_relative_deviation=0.05,
    )
    assert np.array_equal(
        diffusion["series"]["a_hat"], again["series"]["a_hat"], equal_nan=True
    )
    assert "source_idx" in diffusion["series"]
    assert diffusion["summary"]["A_bD_computation_status"] == "not_testable"


def test_lag_diagnostics_support_ids_differ_per_lag() -> None:
    state, time = _linear_sde()
    pooled = estimate_conditional_mean_rate_level1(state, time, lag=1, **_estimator_kwargs())
    diagnostics = estimate_lag_diagnostics(state, time, **_estimator_kwargs())
    assert diagnostics["computation_status"] == "computed"
    lag1 = diagnostics["summary"]["transition_support_id_lag1"]
    lag2 = diagnostics["summary"]["transition_support_id_lag2"]
    lag4 = diagnostics["summary"]["transition_support_id_lag4"]
    assert lag1 == pooled["summary"]["transition_support_id"]
    assert lag2 != lag1
    assert lag4 != lag2
    assert lag4 != lag1
    assert diagnostics["summary"]["embargo_semantics"] == EMBARGO_SEMANTICS_INDEX_STEPS
    assert (
        diagnostics["summary"]["raw_window_support_independence"]
        == RAW_WINDOW_SUPPORT_INDEPENDENCE_NOT_ESTABLISHED
    )


def test_export_stamps_overlap_only_when_both_computed() -> None:
    state, time = _linear_sde(n=500)
    kwargs = dict(
        state=state,
        time=time,
        stage=None,
        segment_id=np.zeros(time.size, dtype=np.int32),
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
    )
    both = build_dynamical_families_export(
        config=_diffusion_config(drift_family_enabled=True), **kwargs
    )
    diffusion_only = build_dynamical_families_export(
        config=_diffusion_config(drift_family_enabled=False), **kwargs
    )
    pooled = both["drift"][MEASUREMENT_ID_CONDITIONAL_MEAN_RATE][VARIANT_POOLED]
    diffusion = both["diffusion"]
    assert pooled["summary"]["lag1_support_ids_match"] is True
    assert diffusion["summary"]["lag1_support_ids_match"] is True
    n_common = int(pooled["summary"]["n_common_source_idx"])
    assert n_common == int(pooled["series"]["source_idx"].size)
    assert n_common == int(diffusion["series"]["source_idx"].size)
    assert "lag1_support_ids_match" not in (diffusion_only["diffusion"].get("summary") or {})
    assert diffusion["summary"]["A_bD_computation_status"] == "not_testable"


def test_support_object_is_not_a_measurement_level() -> None:
    entry = support_object_entry(SUPPORT_OBJECT_LAG1)
    assert entry["not_a_measurement_level"] is True
    assert entry["interpretation_level"] is None
    assert entry["object_kind"] == "transition_support"
    assert entry["embargo_semantics"] == EMBARGO_SEMANTICS_INDEX_STEPS
    assert entry["raw_window_support_independence"] == RAW_WINDOW_SUPPORT_INDEPENDENCE_NOT_ESTABLISHED
    assert entry["physical_path"] == PHYSICAL_POOLED_SOURCE_IDX
    assert PHYSICAL_DIFFUSION_SOURCE_IDX in entry["also_written_at"]
    assert PHYSICAL_CROSSFIT_SOURCE_IDX in entry["variant_subset_paths"]
    assert PHYSICAL_CROSSFIT_SOURCE_IDX not in entry["also_written_at"]


def test_index_embargo_does_not_separate_overlapping_raw_windows() -> None:
    """Disjoint window indices can still share raw samples (003 §7.1)."""
    window_length = 30
    hop = 1
    embargo_steps = 4
    n_windows = 40
    split = n_windows // 2

    def raw_support(window_i: int) -> set[int]:
        start = window_i * hop
        return set(range(start, start + window_length))

    fold1 = [i for i in range(n_windows) if i < split - embargo_steps]
    fold2 = [i for i in range(n_windows) if i > split + embargo_steps]
    assert fold1 and fold2
    assert min(fold2) - max(fold1) - 1 == 2 * embargo_steps + 1
    shared: set[int] = set()
    for left in fold1:
        for right in fold2:
            shared |= raw_support(left) & raw_support(right)
    assert shared


def test_savgol_filter_support_exceeds_frozen_index_embargo() -> None:
    """A typical Savitzky-Golay window is wider than embargo_steps=4."""
    savgol_window = 11
    half_support = savgol_window // 2
    embargo_steps = 4
    assert half_support > embargo_steps


@pytest.mark.parametrize("claim_key", FORBIDDEN_EMBARGO_CLAIM_KEYS)
def test_yaml_refuses_raw_window_embargo_claims(claim_key: str) -> None:
    state, time = _linear_sde(n=80)
    kwargs = dict(
        state=state,
        time=time,
        stage=None,
        segment_id=np.zeros(time.size, dtype=np.int32),
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
    )
    with pytest.raises(ValueError, match=claim_key):
        build_dynamical_families_export(
            config={
                "dynamical_families": {
                    "enabled": True,
                    claim_key: True,
                    "diffusion": {"enabled": True},
                }
            },
            **kwargs,
        )
    with pytest.raises(ValueError, match=claim_key):
        build_dynamical_families_export(
            config={
                "dynamical_families": {
                    "enabled": True,
                    "one_step": {"enabled": True, claim_key: True},
                }
            },
            **kwargs,
        )
