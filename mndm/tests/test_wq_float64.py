"""Precision and scale-stability checks for stochastic reachability."""

import warnings
import inspect
from pathlib import Path

import h5py

import numpy as np

from mndm.dynamics.stochastic_reachability import (
    _stable_logdet,
    _stable_trace_ratio,
    compute_stochastic_reachability,
    estimate_transition_residual_covariance_proxy,
)
import mndm.dynamics.stochastic_reachability as reachability_module
from mndm.schema import MNPSPayload
from core.io.h5_writer import write_h5


def _q_contract(covariance, *, precision="float64"):
    return {
        "schema_version": "mndm.transition_residual_covariance_proxy.v1",
        "computation_status": "computed",
        "q_time_semantics": "one_step_transition_covariance",
        "q_units": "state_squared",
        "conversion_model": "not_applicable",
        "q_dt_sec": 30.0,
        "covariance": np.asarray(covariance),
        "numerical_precision": precision,
    }


def _transitions(n=20):
    residual = np.tile(np.asarray([[1.0, 0.0], [0.0, 2.0]]), (n // 2, 1))
    return {
        "computation_status": "computed",
        "series": {
            "transition_residual": residual,
            "dt_sec": np.full(n, 30.0),
        },
        "provenance": {"crossfit_status": "cross_fitted"},
    }


def test_transition_q_float64_is_opt_in_and_preserved():
    default = estimate_transition_residual_covariance_proxy(_transitions())
    high = estimate_transition_residual_covariance_proxy(_transitions(), precision="float64")
    assert default["covariance"].dtype == np.dtype("float32")
    assert high["covariance"].dtype == np.dtype("float64")
    assert high["residual_mean"].dtype == np.dtype("float64")
    assert high["numerical_precision"] == "float64"


def test_effective_rank_avoids_log_warnings_for_zero_eigenvalues():
    transitions = _transitions()
    transitions["series"]["transition_residual"] = np.ones((20, 2))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = estimate_transition_residual_covariance_proxy(
            transitions, min_eigenvalue=0.0, precision="float64"
        )
    assert not [warning for warning in caught if issubclass(warning.category, RuntimeWarning)]
    assert np.isnan(result["effective_rank"])


def test_float64_reachability_summaries_are_scale_stable():
    q = np.eye(2, dtype=np.float64) * 1.0e300
    result = compute_stochastic_reachability(
        [np.eye(2, dtype=np.float64) * 1.0e150],
        q,
        q_contract=_q_contract(q),
        precision="float64",
    )
    assert result["computation_status"] == "computed"
    assert result["w_q"].dtype == np.dtype("float64")
    assert np.isfinite(result["v_norm"])
    assert np.isfinite(result["d_eff"])
    assert np.isfinite(result["c_1_q"])
    assert np.isfinite(result["a_q"])
    assert result["numerical_precision"] == "float64"


def test_scale_normalized_trace_summary_handles_finite_near_max_entries():
    q = np.eye(8, dtype=np.float64) * 1.0e308
    result = compute_stochastic_reachability(
        [np.eye(8, dtype=np.float64)],
        q,
        q_contract=_q_contract(q),
        precision="float64",
    )
    assert result["computation_status"] == "computed"
    assert np.isfinite(result["a_q"])
    assert np.isfinite(result["v_norm"])
    assert np.isfinite(result["d_eff"])


def test_trace_ratio_uses_diagonal_trace_and_log_scale_subtraction():
    numerator = np.asarray([[1.0e308, 4.0e307], [2.0e307, 1.0e308]])
    denominator = np.asarray([[1.0e-308, 3.0e-309], [2.0e-309, 1.0e-308]])
    expected = (np.log(2.0) + np.log(1.0e308)) - (np.log(2.0) + np.log(1.0e-308))
    assert np.isclose(_stable_trace_ratio(numerator, denominator), expected, rtol=1e-14)


def test_regularized_logdet_keeps_rank_deficient_modes():
    epsilon = 0.1
    eigenvalues = np.asarray([1.0, 0.0]) + epsilon
    expected = np.log(1.1) + np.log(0.1)
    assert np.isclose(_stable_logdet(eigenvalues), expected, rtol=1e-14)


def test_logdet_extreme_spread_does_not_underflow_normalized_modes():
    assert np.isclose(_stable_logdet(np.asarray([1.0e300, 1.0e-300])), 0.0, atol=1e-12)


def test_reachability_preserves_complete_post_projection_qc():
    q = np.eye(2, dtype=np.float64)
    result = compute_stochastic_reachability(
        [np.eye(2)], q, q_contract=_q_contract(q), precision="float64"
    )
    qc = result["w_q_projection_qc"]
    assert qc["q_output_dtype"] == "float64"
    assert "q_psd_tolerance" in qc
    assert qc["q_floor_met_within_tolerance_post_dtype"]
    assert qc["q_psd_material_failure"] is False


def test_reachability_fails_closed_on_material_post_dtype_psd_failure(monkeypatch):
    def bad_projection(matrix, *, min_eigenvalue, precision):
        return np.eye(1, dtype=np.float64), {
            "q_psd_post_dtype": False,
            "q_requested_min_eigenvalue": min_eigenvalue,
            "q_min_eigenvalue": -1.0,
            "q_psd_tolerance": 1e-12,
        }

    monkeypatch.setattr(reachability_module, "project_to_psd", bad_projection)
    q = np.eye(1, dtype=np.float64)
    result = reachability_module.compute_stochastic_reachability(
        [np.eye(1)], q, q_contract=_q_contract(q), precision="float64"
    )
    assert result["computation_status"] == "invalid"
    assert result["failure_reason"] == "post_dtype_psd_failure"


def test_float64_w_recurrence_overflow_remains_invalid():
    q = np.eye(1, dtype=np.float64) * 1.0e100
    result = compute_stochastic_reachability(
        [np.asarray([[1.0e150]]), np.asarray([[1.0e150]])],
        q,
        q_contract=_q_contract(q),
        precision="float64",
    )
    assert result["computation_status"] == "invalid"
    assert result["failure_reason"] == "reachability_numerical_overflow"


def test_default_recurrence_keeps_legacy_float64_arithmetic_before_float32_output():
    phi = [np.eye(2) * (1.0 + 1.0e-7) for _ in range(100)]
    q = np.eye(2) * 0.123456789
    default = compute_stochastic_reachability(phi, q, q_contract=_q_contract(q))
    explicit = compute_stochastic_reachability(
        phi, q, q_contract=_q_contract(q), precision="float64"
    )
    np.testing.assert_array_equal(default["w_q"], explicit["w_q"].astype(np.float32))


def test_subject_summary_wires_float64_to_q_and_reachability_helpers():
    from mndm.pipeline import summary as summary_module

    source = inspect.getsource(summary_module.SubjectSummaryRunner.run)
    assert source.count('precision="float64"') == 4


def test_float64_reachability_roundtrips_as_float64_in_actual_h5_writer(tmp_path: Path):
    q = np.eye(2, dtype=np.float64) * 0.25
    result = compute_stochastic_reachability(
        [np.eye(2, dtype=np.float64)], q, q_contract=_q_contract(q), precision="float64"
    )
    payload = MNPSPayload(
        time=np.asarray([0.0, 1.0]),
        x=np.zeros((2, 3)),
        x_dot=np.zeros((2, 3)),
        stochastic_reachability=result,
    )
    output = write_h5(tmp_path / "float64_reachability.h5", "test", payload)
    with h5py.File(output, "r") as handle:
        assert handle["/stochastic_reachability/v1/primary/w_q"].dtype == np.dtype("float64")
