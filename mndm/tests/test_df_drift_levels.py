"""SL-LEV-MES-003: chart-drift identities, not a parallel Level 0–3 ladder."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))

from mndm.dynamical_families import (
    CHART_DRIFT_SCHEMA_VERSION,
    family_forbids,
    get_family,
)
from mndm.dynamical_families.chart_drift import SOURCE_CROSSFIT, resolve_ingest_chart_drift
from mndm.dynamical_families.finite_lag_drift import (
    REASON_INSUFFICIENT_LOCAL,
    REASON_LAG_INCONSISTENT,
    estimate_blocked_crossfit_mean_rate,
    estimate_chart_drift_family,
    estimate_conditional_mean_rate_level1,
    estimate_lag_diagnostics,
    estimate_realized_velocity_level0,
)
from mndm.dynamical_families.measurement_register import (
    MEASUREMENT_ID_CONDITIONAL_COVARIANCE,
    MEASUREMENT_ID_CONDITIONAL_MEAN_RATE,
    MEASUREMENT_ID_INCREMENT_COVARIANCE,
    MEASUREMENT_ID_INNOVATION_COVARIANCE,
    MEASUREMENT_ID_ITO_DIFFUSION,
    MEASUREMENT_ID_ITO_DRIFT,
    MEASUREMENT_ID_REALIZED_VELOCITY,
    MEASUREMENT_ID_SMOOTHED_VELOCITY,
    PHYSICAL_DIFFUSION_A_HAT,
    PHYSICAL_MNPS_DOT,
    QUALIFICATION_ITO_NOT_QUALIFIED,
    SUPPORT_SUFFICIENT,
    VARIANT_BLOCKED_CROSSFIT,
    VARIANT_LAG_DIAGNOSTICS,
    VARIANT_POOLED,
    register_entry,
)
from mndm.dynamical_families.validity import increment_pairs, increment_pairs_at_lag
from mndm.pipeline.dynamical_families_export import build_dynamical_families_export
from mndm.projection import estimate_derivatives
from mndm.schema import MNPSPayload

ROOT = Path(__file__).resolve().parents[2]

_B_TRUE = np.array([0.50, -0.25, 0.10], dtype=float)
_LEVEL1_MEAN_TOL = 0.15
_BROWNIAN_MEAN_TOL = 0.12
_HELD_OUT_MEAN_TOL = 0.20


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


def _brownian(*, seed: int = 11, n: int = 1600, dt: float = 0.01, sigma: float = 0.05):
    return _linear_sde(seed=seed, n=n, dt=dt, sigma=sigma, b=np.zeros(3))


def _period2_oscillation(*, n: int = 800, dt: float = 0.01, amplitude: float = 5.0):
    state = np.zeros((n, 3), dtype=float)
    state[1::2] = amplitude
    time = np.arange(n, dtype=float) * dt
    return state, time


def _assert_interleaved_holdout_recovers_b(state: np.ndarray, time: np.ndarray) -> None:
    """Held-out neighborhood conditional mean on interleaved increment sources."""
    from mndm.dynamical_families.finite_lag_drift import _fill_conditional_mean

    segments = np.zeros(time.size, dtype=np.int32)
    source_idx, increments, dts = increment_pairs(
        np.asarray(state, dtype=float),
        np.asarray(time, dtype=float),
        segments,
        max_gap_sec=None,
    )
    odd = np.zeros(source_idx.size, dtype=bool)
    odd[1::2] = True
    even = ~odd
    nominal_dt = float(np.median(dts))
    filled = _fill_conditional_mean(
        np.asarray(state, dtype=float),
        np.asarray(time, dtype=float),
        source_idx[even],
        ref_source_idx=source_idx[odd],
        ref_increments=increments[odd],
        neighborhood_k=20,
        minimum_support=10,
        max_neighborhood_radius=None,
        min_temporal_span_sec=None,
        weight_mode="inverse_distance",
        nominal_dt=nominal_dt,
        epsilon=1e-12,
        store_neighbor_source_idx=False,
    )
    valid = filled["valid"][source_idx[even]] == 1
    assert int(np.sum(valid)) > 50
    holdout_mean = np.mean(filled["b_hat"][source_idx[even]][valid], axis=0)
    assert np.all(np.sign(holdout_mean) == np.sign(_B_TRUE))
    assert np.all(np.abs(holdout_mean - _B_TRUE) < _HELD_OUT_MEAN_TOL)


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


def _walk_status_tokens(payload: object) -> list[str]:
    tokens: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {
                "computation_status",
                "qualification_status",
                "ito_qualification_status",
                "failure_reason",
                "measurement_id",
                "variant_id",
            }:
                tokens.append(str(value))
            tokens.extend(_walk_status_tokens(value))
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            tokens.extend(_walk_status_tokens(item))
    return tokens


def test_increment_pairs_lag1_matches_legacy_helper() -> None:
    state, time = _linear_sde(n=40)
    segments = np.zeros(time.size, dtype=np.int32)
    left = increment_pairs(state, time, segments, max_gap_sec=None)
    right = increment_pairs_at_lag(state, time, segments, lag=1, max_gap_sec=None)
    for a, b in zip(left, right):
        np.testing.assert_array_equal(a, b)


def test_pooled_level1_recovers_constant_sde_drift() -> None:
    state, time = _linear_sde()
    result = estimate_conditional_mean_rate_level1(state, time, **_estimator_kwargs())
    assert result["computation_status"] == "computed"
    assert result["measurement_id"] == MEASUREMENT_ID_CONDITIONAL_MEAN_RATE
    assert result["interpretation_level"] == 1
    assert result["variant_id"] == VARIANT_POOLED
    assert result["summary"]["estimand"] == "mean_increment_over_nominal_dt"
    assert result["support_status"] == SUPPORT_SUFFICIENT
    assert result["qualification_status"] == QUALIFICATION_ITO_NOT_QUALIFIED
    assert result["support_status"] != result["qualification_status"]
    assert result["provenance"]["settings"]["not_sde_drift"] is True
    valid = np.asarray(result["series"]["valid"]) == 1
    estimate = np.asarray(result["series"]["b_hat"], dtype=float)[valid]
    mean_b = np.mean(estimate, axis=0)
    assert np.all(np.sign(mean_b) == np.sign(_B_TRUE))
    assert np.all(np.abs(mean_b - _B_TRUE) < _LEVEL1_MEAN_TOL)
    _assert_interleaved_holdout_recovers_b(state, time)


def test_pure_brownian_is_near_zero_and_not_ito_qualified() -> None:
    state, time = _brownian()
    result = estimate_conditional_mean_rate_level1(state, time, **_estimator_kwargs())
    assert result["computation_status"] == "computed"
    valid = np.asarray(result["series"]["valid"]) == 1
    mean_b = np.mean(np.asarray(result["series"]["b_hat"], dtype=float)[valid], axis=0)
    assert np.all(np.abs(mean_b) < _BROWNIAN_MEAN_TOL)
    diagnostics = estimate_lag_diagnostics(
        state, time, consistency_rel_tol=0.5, **_estimator_kwargs()
    )
    assert diagnostics["qualification_status"] == QUALIFICATION_ITO_NOT_QUALIFIED
    assert "ito_qualified" not in _walk_status_tokens(diagnostics)


def test_gaps_and_short_series_are_explicit_not_zero_filled() -> None:
    short_state, short_time = _linear_sde(n=12)
    short = estimate_conditional_mean_rate_level1(short_state, short_time, **_estimator_kwargs())
    assert short["computation_status"] == "insufficient_support"
    assert short["failure_reason"] in {REASON_INSUFFICIENT_LOCAL, "insufficient_samples"}
    assert short["series"] == {}

    state, time = _linear_sde(n=400)
    state = state.copy()
    state[180] = np.nan
    gapped = estimate_conditional_mean_rate_level1(state, time, **_estimator_kwargs())
    assert gapped["computation_status"] == "computed"
    assert int(gapped["series"]["valid"][180]) == 0
    assert np.all(np.isnan(gapped["series"]["b_hat"][180]))
    assert not np.any(gapped["series"]["b_hat"][180] == 0.0)
    segments = np.zeros(time.size, dtype=np.int32)
    lag2_idx, _, _ = increment_pairs_at_lag(
        np.asarray(state, dtype=float),
        np.asarray(time, dtype=float),
        segments,
        lag=2,
        max_gap_sec=None,
    )
    assert 179 not in set(lag2_idx.tolist())
    assert 180 not in set(lag2_idx.tolist())


def test_blocked_crossfit_is_level1_variant_with_index_step_embargo() -> None:
    state, time = _linear_sde(n=1200)
    result = estimate_blocked_crossfit_mean_rate(state, time, embargo_steps=4, **_estimator_kwargs())
    assert result["computation_status"] == "computed"
    assert result["measurement_id"] == MEASUREMENT_ID_CONDITIONAL_MEAN_RATE
    assert result["interpretation_level"] == 1
    assert result["variant_id"] == VARIANT_BLOCKED_CROSSFIT
    assert result["summary"]["embargo_semantics"] == "index_steps"
    fold1 = set(np.asarray(result["summary"]["fold1_source_idx"]).tolist())
    fold2 = set(np.asarray(result["summary"]["fold2_source_idx"]).tolist())
    assert fold1.isdisjoint(fold2)
    split = int(result["summary"]["split_index"])
    embargo = int(result["summary"]["embargo_steps"])
    assert embargo == 4
    assert not any(abs(int(idx) - split) <= embargo for idx in fold1 | fold2)
    neighbors = np.asarray(result["series"]["neighbor_source_idx"])
    fold_id = np.asarray(result["series"]["fold_id"])
    valid = np.asarray(result["series"]["valid"]) == 1
    for center in np.flatnonzero(valid):
        used = neighbors[center]
        used = used[used >= 0]
        if int(fold_id[center]) == 1:
            assert set(used.tolist()).issubset(fold2)
        elif int(fold_id[center]) == 2:
            assert set(used.tolist()).issubset(fold1)
        else:
            raise AssertionError("valid crossfit window must belong to a fold")


def test_lag_diagnostics_constant_b_is_consistent_not_level3() -> None:
    state, time = _linear_sde()
    result = estimate_lag_diagnostics(state, time, consistency_rel_tol=0.5, **_estimator_kwargs())
    assert result["computation_status"] == "computed"
    assert result["measurement_id"] == MEASUREMENT_ID_CONDITIONAL_MEAN_RATE
    assert result["interpretation_level"] == 1
    assert result["variant_id"] == VARIANT_LAG_DIAGNOSTICS
    assert result["summary"]["multi_lag_consistency"] == "multi_lag_consistent"
    assert result["qualification_status"] == QUALIFICATION_ITO_NOT_QUALIFIED
    assert result["summary"]["not_ito_drift_level3"] is True
    assert result["computation_status"] != "ito_qualified"
    assert "ito_qualified" not in _walk_status_tokens(result)


def test_lag_inconsistent_does_not_erase_pooled_level1() -> None:
    state, time = _period2_oscillation()
    diagnostics = estimate_lag_diagnostics(
        state, time, consistency_rel_tol=0.5, **_estimator_kwargs()
    )
    assert diagnostics["computation_status"] == "computed"
    assert diagnostics["summary"]["multi_lag_consistency"] == REASON_LAG_INCONSISTENT
    assert diagnostics["support_status"] == SUPPORT_SUFFICIENT
    assert diagnostics["qualification_status"] == QUALIFICATION_ITO_NOT_QUALIFIED
    assert diagnostics["support_status"] != diagnostics["qualification_status"]
    family = estimate_chart_drift_family(state, time, **_estimator_kwargs())
    pooled = family[MEASUREMENT_ID_CONDITIONAL_MEAN_RATE][VARIANT_POOLED]
    nested = family[MEASUREMENT_ID_CONDITIONAL_MEAN_RATE][VARIANT_LAG_DIAGNOSTICS]
    assert pooled["computation_status"] == "computed"
    assert pooled["support_status"] == SUPPORT_SUFFICIENT
    assert pooled["qualification_status"] == QUALIFICATION_ITO_NOT_QUALIFIED
    assert nested["computation_status"] == "computed"
    assert nested["support_status"] == SUPPORT_SUFFICIENT
    assert nested["qualification_status"] == QUALIFICATION_ITO_NOT_QUALIFIED
    assert nested["summary"]["multi_lag_consistency"] == REASON_LAG_INCONSISTENT
    assert MEASUREMENT_ID_ITO_DRIFT not in family
    assert "ito_drift_level3" not in _walk_status_tokens(family)
    assert "ito_qualified" not in _walk_status_tokens(family)


def test_realized_velocity_is_not_savgol_mnps_dot() -> None:
    state, time = _linear_sde(n=400, dt=0.01)
    realized = estimate_realized_velocity_level0(
        state,
        time,
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
        min_samples=30,
    )
    assert realized["computation_status"] == "computed"
    assert realized["measurement_id"] == MEASUREMENT_ID_REALIZED_VELOCITY
    assert realized["interpretation_level"] == 0
    assert realized["summary"]["estimand"] == "per_step_forward_difference_over_observed_dt"
    assert realized["summary"]["not_alias_of_mnps_3d_dot"] is True
    dx_dt = np.asarray(realized["series"]["dx_dt"], dtype=float)
    valid = np.asarray(realized["series"]["valid"]) == 1
    assert np.all(np.isnan(dx_dt[-1]))
    savgol = estimate_derivatives(state, dt=0.01, method="sav_gol", window=7, polyorder=3)
    assert np.all(np.isfinite(savgol[-1]))
    assert not np.allclose(dx_dt[valid], savgol[valid], atol=1e-5)
    smoothed = register_entry(MEASUREMENT_ID_SMOOTHED_VELOCITY)
    assert smoothed["physical_path"] == PHYSICAL_MNPS_DOT
    assert smoothed["written_by_drift_family"] is False


def _diffusion_config(*, drift_family_enabled: bool, xdot_source: bool = False) -> dict:
    return {
        "dynamical_families": {
            "enabled": True,
            "coordinate_layer": "subject_anchored",
            "diffusion": {
                "enabled": True,
                "neighborhood": {"k": 20},
                "min_samples": 30,
                "min_neighborhood_samples": 10,
                "drift": {
                    "enabled": bool(xdot_source),
                    "mode": "alignment_only",
                    "source": "mnps_xdot" if xdot_source else "not_supplied",
                },
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


def test_export_with_drift_keeps_diffusion_a_hat_and_A_bD_not_testable() -> None:
    state, time = _linear_sde(n=500)
    segment_id = np.zeros(time.size, dtype=np.int32)
    kwargs = dict(
        state=state,
        time=time,
        stage=None,
        segment_id=segment_id,
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
    )
    off = build_dynamical_families_export(config=_diffusion_config(drift_family_enabled=False), **kwargs)
    on = build_dynamical_families_export(config=_diffusion_config(drift_family_enabled=True), **kwargs)
    assert "drift" not in off
    assert off["diffusion"]["computation_status"] == on["diffusion"]["computation_status"] == "computed"
    for key in ("a_hat", "D_total", "d_diff", "c_diff", "valid", "A_bD", "R_b_over_a"):
        np.testing.assert_array_equal(
            off["diffusion"]["series"][key],
            on["diffusion"]["series"][key],
            err_msg=key,
        )
    assert on["diffusion"]["summary"]["A_bD_computation_status"] == "not_testable"
    assert on["diffusion"]["summary"]["R_b_over_a_computation_status"] == "not_testable"
    assert on["diffusion"]["summary"]["drift_alignment_failure_reason"] == "independent_drift_not_supplied"
    assert on["diffusion"]["provenance"]["settings"]["drift_source"] == "not_supplied"
    drift = on["drift"]
    assert drift["schema_version"] == CHART_DRIFT_SCHEMA_VERSION
    assert drift["computation_status"] == "computed"
    assert drift["measurement_validity"] == "not_assessed"
    assert drift["grain"]["native"] == "window"
    assert drift["grain"]["parent"] == "recording"
    assert drift["summary"]["independent_drift_for_A_bD"] is False
    assert drift["summary"]["ito_drift_level3_written"] is False
    level1 = drift[MEASUREMENT_ID_CONDITIONAL_MEAN_RATE]
    pooled = level1[VARIANT_POOLED]
    crossfit = level1[VARIANT_BLOCKED_CROSSFIT]
    diagnostics = level1[VARIANT_LAG_DIAGNOSTICS]
    assert pooled["computation_status"] == "computed"
    assert crossfit["computation_status"] == "computed"
    assert pooled["measurement_id"] == crossfit["measurement_id"] == MEASUREMENT_ID_CONDITIONAL_MEAN_RATE
    assert pooled["interpretation_level"] == crossfit["interpretation_level"] == 1
    assert diagnostics["qualification_status"] == QUALIFICATION_ITO_NOT_QUALIFIED
    assert MEASUREMENT_ID_ITO_DRIFT not in drift
    assert "finite_lag" not in drift
    assert "crossfit" not in drift
    assert "ito_candidate" not in drift
    assert "ito_qualified" not in _walk_status_tokens(drift)
    covariance = register_entry(MEASUREMENT_ID_CONDITIONAL_COVARIANCE)
    assert covariance["physical_path"] == PHYSICAL_DIFFUSION_A_HAT
    assert covariance["identity_relation"] == "documented_identity_of_existing_a_hat_not_a_rename"
    assert covariance["estimand"] == "centered_increment_covariance_over_nominal_dt"
    assert covariance["diffusion_convention"] == "a_not_D_over_2"
    assert "conditional_covariance_rate_level1" not in on["diffusion"]
    diffusion = on["diffusion"]
    assert diffusion["measurement_id"] == MEASUREMENT_ID_CONDITIONAL_COVARIANCE
    assert diffusion["interpretation_level"] == 1
    assert diffusion["qualification_status"] == QUALIFICATION_ITO_NOT_QUALIFIED
    assert diffusion["summary"]["estimand"] == "centered_increment_covariance_over_nominal_dt"
    assert diffusion["summary"]["not_ito_diffusion_tensor"] is True
    assert "a_hat" in diffusion["series"]
    assert MEASUREMENT_ID_ITO_DIFFUSION not in diffusion
    assert MEASUREMENT_ID_INNOVATION_COVARIANCE not in diffusion
    increment = diffusion[MEASUREMENT_ID_INCREMENT_COVARIANCE]
    assert increment["computation_status"] == "computed"
    assert increment["summary"]["not_divided_by_dt"] is True
    assert increment["summary"]["not_local_knn"] is True
    assert increment["qualification_status"] == "unconditional_increment_covariance_not_a_hat"
    assert increment["grain"]["native"] == "recording"
    assert increment["grain"]["repeated_measure"] == "false"
    cov0 = np.asarray(increment["summary"]["increment_covariance"], dtype=float)
    a_hat = np.asarray(diffusion["series"]["a_hat"], dtype=float)
    finite_a = a_hat[np.isfinite(a_hat).all(axis=(1, 2))]
    assert cov0.shape == (3, 3)
    assert np.all(np.isfinite(cov0))
    dx = np.diff(np.asarray(state, dtype=float), axis=0)
    expected = np.cov(dx, rowvar=False, ddof=1)
    np.testing.assert_allclose(cov0, expected, rtol=1e-5, atol=1e-6)
    mean_a = np.mean(finite_a, axis=0)
    assert not np.allclose(cov0, mean_a, rtol=1e-3, atol=1e-3)
    assert increment["provenance"]["estimator"] == "unconditional_increment_covariance"


def test_mnps_xdot_source_stays_closed_with_drift_family_enabled() -> None:
    state, time = _linear_sde(n=400)
    export = build_dynamical_families_export(
        config=_diffusion_config(drift_family_enabled=True, xdot_source=True),
        state=state,
        time=time,
        stage=None,
        segment_id=np.zeros(time.size, dtype=np.int32),
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
    )
    diffusion = export["diffusion"]
    assert diffusion["summary"]["A_bD_computation_status"] == "not_testable"
    assert diffusion["summary"]["drift_alignment_failure_reason"] == "mnps_xdot_as_sde_drift"
    assert np.all(np.isnan(diffusion["series"]["A_bD"]))
    resolved = resolve_ingest_chart_drift(
        {"drift": {"enabled": True, "source": "mnps_xdot", "mode": "alignment_only"}}
    )
    assert resolved.field is None
    assert resolved.source == "mnps_xdot"
    crossfit_closed = resolve_ingest_chart_drift(
        {"drift": {"enabled": True, "source": SOURCE_CROSSFIT, "mode": "alignment_only"}}
    )
    assert crossfit_closed.field is None
    assert crossfit_closed.failure_reason == "crossfit_not_authorized_before_m3"


def test_old_nested_yaml_keys_are_refused() -> None:
    state, time = _linear_sde(n=80)
    kwargs = dict(
        state=state,
        time=time,
        stage=None,
        segment_id=np.zeros(time.size, dtype=np.int32),
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
    )
    for key in ("finite_lag", "crossfit", "ito_candidate", "crossfit_local_chart_b"):
        config = {
            "dynamical_families": {
                "enabled": True,
                "drift": {"enabled": False, key: {"enabled": True}},
            }
        }
        with pytest.raises(ValueError, match=key):
            build_dynamical_families_export(config=config, **kwargs)
    with pytest.raises(ValueError, match="increment_covariance_level0"):
        build_dynamical_families_export(
            config={
                "dynamical_families": {
                    "enabled": True,
                    "diffusion": {"enabled": True, "increment_covariance_level0": True},
                }
            },
            **kwargs,
        )
    with pytest.raises(ValueError, match="increment_covariance_level0"):
        build_dynamical_families_export(
            config={
                "dynamical_families": {
                    "enabled": False,
                    "diffusion": {"enabled": False, "increment_covariance_level0": True},
                }
            },
            **kwargs,
        )


def test_increment_l0_computes_when_local_a_hat_exits_early() -> None:
    from mndm.dynamical_families.diffusion_geometry import estimate_local_diffusion_geometry

    state, time = _linear_sde(n=40)
    low_pairs = estimate_local_diffusion_geometry(
        state,
        time,
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
        neighborhood_k=80,
        min_samples=30,
        min_neighborhood_samples=80,
    )
    assert low_pairs["computation_status"] == "insufficient_support"
    assert low_pairs["failure_reason"] == "insufficient_valid_increment_pairs"
    increment = low_pairs[MEASUREMENT_ID_INCREMENT_COVARIANCE]
    assert increment["computation_status"] == "computed"
    cov0 = np.asarray(increment["summary"]["increment_covariance"], dtype=float)
    expected = np.cov(np.diff(np.asarray(state, dtype=float), axis=0), rowvar=False, ddof=1)
    np.testing.assert_allclose(cov0, expected, rtol=1e-5, atol=1e-6)

    irregular_time = np.arange(80, dtype=float)
    irregular_time[40:] += 1.0
    irregular_state = np.zeros((80, 3), dtype=float)
    irregular_state[:, 0] = np.arange(80, dtype=float)
    irregular = estimate_local_diffusion_geometry(
        irregular_state,
        irregular_time,
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
        neighborhood_k=20,
        min_samples=30,
        min_neighborhood_samples=10,
    )
    assert irregular["computation_status"] == "not_testable"
    assert irregular["failure_reason"] == "materially_irregular_increment_timestep"
    inc_irreg = irregular[MEASUREMENT_ID_INCREMENT_COVARIANCE]
    assert inc_irreg["computation_status"] == "computed"
    assert inc_irreg["summary"]["not_divided_by_dt"] is True
    assert inc_irreg["grain"]["native"] == "recording"


def test_registry_and_payload_write_new_nested_names(tmp_path: Path) -> None:
    assert family_forbids("drift", "mnps_xdot_as_sde_drift")
    assert family_forbids("drift", "ito_qualified_auto_promotion")
    assert get_family("drift")["schema"] == CHART_DRIFT_SCHEMA_VERSION
    assert get_family("drift")["namespace"] == "/dynamical_families/drift/v1"
    state, time = _linear_sde(n=360)
    export = build_dynamical_families_export(
        config=_diffusion_config(drift_family_enabled=True),
        state=state,
        time=time,
        stage=None,
        segment_id=np.zeros(time.size, dtype=np.int32),
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
    )
    from core.io.h5_writer import write_h5
    import h5py

    payload = MNPSPayload(
        time=time,
        x=state.astype(np.float32),
        x_dot=np.zeros_like(state, dtype=np.float32),
        dynamical_families=export,
    )
    output = write_h5(tmp_path / "drift_levels.h5", "drift_levels", payload)
    with h5py.File(output, "r") as handle:
        root = "/dynamical_families/drift/v1"
        assert f"{root}/realized_velocity_level0" in handle
        assert f"{root}/conditional_mean_rate_level1/pooled" in handle
        assert f"{root}/conditional_mean_rate_level1/blocked_crossfit" in handle
        assert f"{root}/conditional_mean_rate_level1/lag_diagnostics" in handle
        assert f"{root}/finite_lag" not in handle
        assert f"{root}/crossfit" not in handle
        assert f"{root}/ito_candidate" not in handle
        assert f"{root}/ito_drift_level3" not in handle
        assert "/dynamical_families/diffusion/v1" in handle
        assert "/dynamical_families/diffusion/v1/ito_diffusion_tensor_level3" not in handle
        assert "/dynamical_families/diffusion/v1/innovation_covariance_level2" not in handle
        assert "/dynamical_families/diffusion/v1/increment_covariance_level0" in handle
        increment = handle["/dynamical_families/diffusion/v1/increment_covariance_level0"]
        assert increment["qualification_status"][()].decode() == "unconditional_increment_covariance_not_a_hat"
        grain = increment["grain"]
        assert grain["native"][()].decode() == "recording"
        assert grain["repeated_measure"][()].decode() == "false"
        assert "/dynamical_families/diffusion/v1/conditional_covariance_rate_level1" not in handle
        diffusion = handle["/dynamical_families/diffusion/v1"]
        assert "series/a_hat" in diffusion
        assert diffusion["measurement_id"][()].decode() == MEASUREMENT_ID_CONDITIONAL_COVARIANCE
        assert int(diffusion["interpretation_level"][()]) == 1
        assert diffusion["summary"]["estimand"][()].decode() == "centered_increment_covariance_over_nominal_dt"
        assert diffusion["summary"]["diffusion_convention"][()].decode() == "a_not_D_over_2"
        assert diffusion["summary"]["qualification_status"][()].decode() == QUALIFICATION_ITO_NOT_QUALIFIED
        assert diffusion["summary"]["A_bD_computation_status"][()].decode() == "not_testable"
        pooled = handle[f"{root}/conditional_mean_rate_level1/pooled"]
        assert pooled["computation_status"][()].decode() == "computed"
        assert pooled["measurement_validity"][()].decode() == "not_assessed"
        assert pooled["measurement_id"][()].decode() == MEASUREMENT_ID_CONDITIONAL_MEAN_RATE
        assert int(pooled["interpretation_level"][()]) == 1
        assert "series/source_idx" in pooled
        assert "transition_support_id" in pooled["summary"]
        diagnostics = handle[f"{root}/conditional_mean_rate_level1/lag_diagnostics"]
        assert diagnostics["summary"]["qualification_status"][()].decode() == QUALIFICATION_ITO_NOT_QUALIFIED
        assert "ito_qualified" not in {
            diagnostics["computation_status"][()].decode(),
            diagnostics["summary"]["qualification_status"][()].decode(),
        }
        assert "series/source_idx" in diffusion
        assert "transition_support_id" in diffusion["summary"]
        assert bool(diffusion["summary"]["lag1_support_ids_match"][()]) is True
        assert bool(pooled["summary"]["lag1_support_ids_match"][()]) is True
        crossfit = handle[f"{root}/conditional_mean_rate_level1/blocked_crossfit"]
        assert crossfit["summary"]["embargo_semantics"][()].decode() == "index_steps"
        assert "series/source_idx" in crossfit


def test_cohort_or_9d_layer_is_not_testable() -> None:
    state, time = _linear_sde(n=200)
    result = estimate_chart_drift_family(
        state,
        time,
        coordinate_layer="coords_3d_cohort_anchored",
        coordinate_names=["m", "d", "e"],
    )
    assert result["computation_status"] == "not_testable"
    assert result["failure_reason"] == "chart_drift_subject_anchored_3d_only"
    wide = np.hstack([state, state])
    nine = estimate_conditional_mean_rate_level1(
        wide,
        time,
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"] * 3,
        neighborhood_k=20,
        min_samples=30,
        min_neighborhood_samples=10,
    )
    assert nine["computation_status"] == "not_testable"
    assert nine["failure_reason"] == "chart_drift_subject_anchored_3d_only"


def test_withheld_ito_and_covariance_siblings_are_register_only() -> None:
    withheld_drift = register_entry(MEASUREMENT_ID_ITO_DRIFT)
    assert withheld_drift["physical_path"] is None
    assert withheld_drift["written_by_drift_family"] is False
    assert withheld_drift["interpretation_level"] == 3
    increment0 = register_entry(MEASUREMENT_ID_INCREMENT_COVARIANCE)
    assert increment0["physical_path"] == (
        "/dynamical_families/diffusion/v1/increment_covariance_level0"
    )
    assert increment0["written_by_diffusion_family"] is True
    assert family_forbids("diffusion", "increment_covariance_level0")
    innovation = register_entry(MEASUREMENT_ID_INNOVATION_COVARIANCE)
    assert innovation["physical_path"] == (
        "/dynamical_families/one_step/v1/innovation_covariance_level2"
    )
    assert innovation["written_by_diffusion_family"] is False
    assert innovation["written_by_one_step_family"] is True
    assert innovation["interpretation_level"] == 2
    ito_a = register_entry(MEASUREMENT_ID_ITO_DIFFUSION)
    assert ito_a["physical_path"] is None
    assert ito_a["written_by_diffusion_family"] is False
    assert ito_a["interpretation_level"] == 3
