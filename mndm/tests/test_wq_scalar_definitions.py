"""Closed-form scalar oracles for stochastic reachability summaries."""

import numpy as np

from mndm.dynamics import stochastic_reachability as reachability


def _contract(covariance: np.ndarray) -> dict[str, object]:
    return {
        "computation_status": "computed",
        "q_time_semantics": "one_step_transition_covariance",
        "q_units": "state_squared",
        "conversion_model": "not_applicable",
        "q_dt_sec": 1.0,
        "covariance": covariance,
        "schema_version": "test.q.v1",
    }


def test_non_diagonal_two_edge_oracle_uses_matrix_trace_and_logdet() -> None:
    q = np.array([[2.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    # First edge contributes Q; the final edge has Phi=diag(2,1), giving
    # W = Q + Phi Q Phi.T = [[10, 3], [3, 2]].
    propagators = [np.zeros((2, 2)), np.diag([2.0, 1.0])]

    result = reachability.compute_stochastic_reachability(
        propagators, q, q_contract=_contract(q), epsilon=1e-8, precision="float64"
    )

    w = np.array([[10.0, 3.0], [3.0, 2.0]])
    eigenvalues = np.linalg.eigvalsh(w)
    assert result["computation_status"] == "computed"
    assert np.allclose(result["w_q"], w)
    assert np.isclose(result["v_norm"], np.log(np.linalg.det(w + 1e-8 * np.eye(2))) / 4.0)
    assert np.isclose(result["d_eff"], np.sum(eigenvalues) ** 2 / np.sum(eigenvalues**2))
    assert np.isclose(result["c_1_q"], eigenvalues[-1] / np.sum(eigenvalues))
    assert np.isclose(result["a_q"], np.log(2.0))


def test_rank_deficient_q_uses_regularized_determinant_after_w_floor() -> None:
    q = np.diag([1.0, 0.0])

    result = reachability.compute_stochastic_reachability(
        [np.eye(2)], q, q_contract=_contract(q), epsilon=0.1, precision="float64"
    )

    assert result["computation_status"] == "computed"
    assert np.allclose(result["w_q"], np.diag([1.0, 0.1]))
    assert np.isclose(result["v_norm"], np.log(1.1 * 0.2) / 4.0)
    assert np.isclose(result["d_eff"], 1.1**2 / (1.0**2 + 0.1**2))
    assert np.isclose(result["c_1_q"], 1.0 / 1.1)


def test_tiny_q_is_floor_regularized_and_aq_uses_epsilon_denominator() -> None:
    q = np.eye(2, dtype=np.float64) * 1e-300

    result = reachability.compute_stochastic_reachability(
        [np.eye(2)], q, q_contract=_contract(q), epsilon=1e-8, precision="float64"
    )

    assert result["computation_status"] == "computed"
    assert np.allclose(result["w_q"], np.eye(2) * 1e-8)
    assert np.isclose(result["a_q"], np.log(2.0))


def test_stable_trace_ratio_handles_distinct_finite_scales() -> None:
    numerator = np.diag([1e100, 2e100])
    denominator = np.diag([1e-100, 2e-100])

    observed = reachability._stable_trace_ratio(numerator, denominator)

    assert np.isfinite(observed)
    assert np.isclose(observed, np.log(1e200))
