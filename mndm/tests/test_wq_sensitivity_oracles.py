"""Independent sensitivity oracles using the public W_Q reachability API."""

import numpy as np
import pytest

from mndm.dynamics.stochastic_reachability import compute_stochastic_reachability
from mndm.tools import wq_sensitivity


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


def _run(phi: list[np.ndarray], q: np.ndarray, epsilon: float = 1e-8) -> dict[str, object]:
    return compute_stochastic_reachability(
        phi, q, q_contract=_contract(q), epsilon=epsilon, precision="float64"
    )


def test_zero_perturbation_is_exact_identity_for_public_metrics() -> None:
    q = np.array([[2.0, 1.0], [1.0, 1.5]], dtype=np.float64)
    phi = [np.array([[1.1, 0.2], [0.0, 0.9]], dtype=np.float64)]
    perturbation = np.eye(2)
    q_perturbed = perturbation @ q @ perturbation.T

    base = _run(phi, q)
    zero = _run(phi, q_perturbed)

    assert base["computation_status"] == zero["computation_status"] == "computed"
    assert np.array_equal(base["w_q"], zero["w_q"])
    for key in ("v_norm", "d_eff", "c_1_q", "a_q"):
        assert np.array_equal(base[key], zero[key])


def test_scalar_congruence_has_closed_form_q_scaling() -> None:
    q = np.array([[2.0]], dtype=np.float64)
    r = np.array([[1.5]], dtype=np.float64)
    q_perturbed = r @ q @ r.T

    result = _run([np.eye(1)], q_perturbed)

    assert result["computation_status"] == "computed"
    assert np.isclose(q_perturbed[0, 0], 4.5)
    assert np.isclose(result["w_q"][0, 0], 4.5)
    assert np.isclose(result["v_norm"], np.log(4.5 + 1e-8) / 2.0)
    assert np.isclose(result["d_eff"], 1.0)
    assert np.isclose(result["c_1_q"], 1.0)
    assert np.isclose(result["a_q"], 0.0)


def test_nondiagonal_metrics_are_finite_and_report_psd_qc() -> None:
    q = np.array([[2.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    phi = [np.zeros((2, 2)), np.diag([2.0, 1.0])]

    result = _run(phi, q)

    assert result["computation_status"] == "computed"
    assert np.all(np.isfinite(result["w_q"]))
    assert all(np.isfinite(result[name]) for name in ("v_norm", "d_eff", "c_1_q", "a_q"))
    assert result["w_q_psd_correction"] is False
    assert result["w_q_min_eigenvalue"] > 0.0


def test_sensitivity_orchestrator_freezes_production_epsilon_and_perturbs_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    phi = np.stack(
        [
            np.array([[1.0 + 0.01 * i, 0.02], [0.0, 0.98 - 0.005 * i]])
            for i in range(8)
        ],
        axis=0,
    )
    q = np.array([[2.0, 0.4], [0.4, 1.0]], dtype=np.float64)
    captured: list[tuple[np.ndarray, np.ndarray, float]] = []
    original = wq_sensitivity.compute_stochastic_reachability

    def capture(propagators, covariance, *, q_contract, epsilon, precision):
        captured.append((np.asarray(propagators, dtype=np.float64).copy(), np.asarray(covariance).copy(), float(epsilon)))
        return original(
            propagators, covariance, q_contract=q_contract, epsilon=epsilon, precision=precision
        )

    monkeypatch.setattr(wq_sensitivity, "compute_stochastic_reachability", capture)
    result = wq_sensitivity.sensitivity_from_arrays(
        phi,
        q,
        _contract(q),
        horizons=(1,),
        n_starts=1,
        epsilons=(0.0, 1e-6),
        seeds=(0,),
    )

    assert captured
    assert {epsilon for _, _, epsilon in captured} == {1e-8}
    assert any(not np.array_equal(window, phi[:1]) for window, _, _ in captured)
    zero_records = [
        row for row in result["records"]
        if float(row.get("perturbation_epsilon", row.get("epsilon", np.nan))) == 0.0
    ]
    assert zero_records
    for branch in ("phi_only", "q_only"):
        branch_rows = [row for row in result["records"] if row["branch"] == branch]
        baseline_metrics = [
            tuple(row["baseline"].get(name) for name in ("w_q_maxabs", "v_norm", "d_eff", "c_1_q", "a_q"))
            for row in branch_rows
        ]
        assert len(set(baseline_metrics)) == 1
    phi_rows = [row for row in result["records"] if row["branch"] == "phi_only"]
    assert any(row.get("relative_phi_distortion", 0.0) == 0.0 for row in phi_rows)
    assert any(row.get("relative_phi_distortion", 0.0) > 0.0 for row in phi_rows)
    for row in zero_records:
        if row.get("status") == "computed":
            assert row.get("delta_w_maxabs_normalized", 0.0) == 0.0


def test_phi_perturbation_reuses_per_edge_directions_across_horizons(monkeypatch: pytest.MonkeyPatch) -> None:
    phi = np.stack(
        [
            np.array([[1.0 + 0.1 * i, 0.03 * (i + 1)], [0.0, 0.7 + 0.2 * i]])
            for i in range(6)
        ],
        axis=0,
    )
    q = np.eye(2, dtype=np.float64)
    captured: list[np.ndarray] = []
    original = wq_sensitivity.compute_stochastic_reachability

    def capture(propagators, covariance, *, q_contract, epsilon, precision):
        captured.append(np.asarray(propagators, dtype=np.float64).copy())
        return original(
            propagators, covariance, q_contract=q_contract, epsilon=epsilon, precision=precision
        )

    monkeypatch.setattr(wq_sensitivity, "compute_stochastic_reachability", capture)
    wq_sensitivity.sensitivity_from_arrays(
        phi,
        q,
        _contract(q),
        horizons=(1, 2, 4),
        n_starts=1,
        epsilons=(1e-6,),
        seeds=(0,),
    )

    variants: dict[int, np.ndarray] = {}
    for window in captured:
        horizon = int(window.shape[0])
        if horizon in (1, 2, 4) and not np.array_equal(window, phi[:horizon]):
            variants[horizon] = window
    assert set(variants) == {1, 2, 4}
    reference_delta = variants[4][0] - phi[0]
    for horizon, window in variants.items():
        delta = window - phi[:horizon]
        assert np.allclose(delta[0], reference_delta, rtol=1e-12, atol=1e-15)
        for edge in range(horizon):
            relative_norm = np.linalg.norm(delta[edge]) / np.linalg.norm(phi[edge])
            assert np.isclose(relative_norm, 1e-6, rtol=1e-6, atol=1e-12)
