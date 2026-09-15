import numpy as np
import pytest

from mndm.tools.wq_horizon_diagnostic import (
    diagnostic_from_arrays,
    load_gate_e_arrays,
    ordinary_recurrence,
    scaled_recurrence,
)


def test_stable_system_scaled_matches_ordinary_within_range():
    phi = np.repeat((0.8 * np.eye(2))[None], 20, axis=0)
    q = np.eye(2) * 0.1
    result = diagnostic_from_arrays(phi, q, horizons=(1, 2, 4, 10))
    assert result["predeclared_horizons"] == [1, 2, 4, 10]
    assert all(r["ordinary_finite"] and r["scaled_representable"] for r in result["results"])
    assert max(r["relative_error_when_representable"] for r in result["results"]) < 1e-12
    assert result["q_psd"] and result["q_symmetric"] and result["q_logdet"] is not None
    assert result["per_start_support"]["4"] == 17


def test_noncommuting_chronological_order_is_preserved():
    phi = np.asarray([[[1.0, 1.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]])
    q = np.diag([1.0, 2.0])
    forward = ordinary_recurrence(phi, q, 2)["covariance"]
    reverse = ordinary_recurrence(phi[::-1], q, 2)["covariance"]
    assert not np.allclose(forward, reverse)
    expected = phi[1] @ q @ phi[1].T + q
    assert np.allclose(forward, expected)


def test_frozen_horizon_protocol_is_covered_exactly_for_long_prefix():
    phi = np.repeat((0.9 * np.eye(1))[None], 1024, axis=0)
    result = diagnostic_from_arrays(phi, np.ones((1, 1)), horizons=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024))
    assert result["predeclared_horizons"] == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    assert all(row["horizon_covered"] for row in result["results"])


def test_unstable_system_separates_finite_math_from_float_representability():
    phi = np.repeat((1.0e200 * np.eye(1))[None], 4, axis=0)
    q = np.ones((1, 1))
    scaled = scaled_recurrence(phi, q, 4)
    assert scaled["scaled_finite"]
    assert not scaled["representable"]
    assert scaled["first_range_exceed_step"] is not None


def test_zero_phi_and_q_remain_absorbing_zero():
    scaled = scaled_recurrence(np.zeros((4, 2, 2)), np.zeros((2, 2)), 4)
    assert scaled["scaled_finite"] and scaled["representable"]
    assert np.all(scaled["normalized_covariance"] == 0)


def test_invalid_and_gap_inputs_fail_without_bridging(tmp_path):
    import h5py
    p = tmp_path / "gap.h5"
    with h5py.File(p, "w") as h:
        h.create_dataset("transition_residuals/v1/primary/series/phi_one_step", data=np.eye(1)[None])
        h.create_dataset("transition_residuals/v1/primary/series/source_window_id", data=[0])
        h.create_dataset("transition_residuals/v1/primary/series/target_window_id", data=[2])
        h.create_dataset("transition_residuals/v1/primary/series/dt_sec", data=[30.0])
        for name, value in {
            "schema_version": "mndm.transition_residuals.v1",
            "computation_status": "computed",
            "measurement_validity": "not_assessed",
            "claim_status": "no_biological_claim",
        }.items():
            h.create_dataset(f"transition_residuals/v1/primary/{name}", data=np.bytes_(value))
        h.create_dataset("transition_residual_covariance_proxy/v1/primary/covariance", data=np.eye(1))
        for name, value in {
            "schema_version": "mndm.transition_residual_covariance_proxy.v1",
            "computation_status": "computed",
            "q_time_semantics": "one_step_transition_covariance",
            "q_units": "state_squared",
            "conversion_model": "not_applicable",
            "q_scope": "recording",
            "q_semantics": "transition_residual_covariance_proxy",
            "measurement_validity": "not_assessed",
            "claim_status": "no_biological_claim",
        }.items():
            h.create_dataset(f"transition_residual_covariance_proxy/v1/primary/{name}", data=np.bytes_(value))
        h.create_dataset("transition_residual_covariance_proxy/v1/primary/q_dt_sec", data=30.0)
        h.create_dataset("transition_residual_covariance_proxy/v1/primary/q_max_dt_deviation_sec", data=0.0)
    with pytest.raises(ValueError, match="not adjacent"):
        load_gate_e_arrays(p)


def test_irregular_positive_dt_fails_without_bridging(tmp_path):
    import h5py
    p = tmp_path / "irregular.h5"
    with h5py.File(p, "w") as h:
        base = "transition_residuals/v1/primary/series"
        h.create_dataset(f"{base}/phi_one_step", data=np.repeat(np.eye(1)[None], 2, axis=0))
        h.create_dataset(f"{base}/source_window_id", data=[0, 1])
        h.create_dataset(f"{base}/target_window_id", data=[1, 2])
        h.create_dataset(f"{base}/dt_sec", data=[30.0, 60.0])
        qbase = "transition_residual_covariance_proxy/v1/primary"
        h.create_dataset(f"{qbase}/covariance", data=np.eye(1))
        for name, value in {
            "schema_version": "mndm.transition_residual_covariance_proxy.v1",
            "computation_status": "computed",
            "q_time_semantics": "one_step_transition_covariance",
            "q_units": "state_squared",
            "conversion_model": "not_applicable",
            "q_scope": "recording",
            "q_semantics": "transition_residual_covariance_proxy",
            "measurement_validity": "not_assessed",
            "claim_status": "no_biological_claim",
        }.items():
            h.create_dataset(f"{qbase}/{name}", data=np.bytes_(value))
        for name, value in {
            "schema_version": "mndm.transition_residuals.v1",
            "computation_status": "computed",
            "measurement_validity": "not_assessed",
            "claim_status": "no_biological_claim",
        }.items():
            h.create_dataset(f"transition_residuals/v1/primary/{name}", data=np.bytes_(value))
        h.create_dataset(f"{qbase}/q_dt_sec", data=30.0)
        h.create_dataset(f"{qbase}/q_max_dt_deviation_sec", data=0.0)
    with pytest.raises(ValueError, match="fixed expected Q dt tolerance"):
        load_gate_e_arrays(p)
