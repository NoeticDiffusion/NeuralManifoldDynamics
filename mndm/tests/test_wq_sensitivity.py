"""Synthetic contracts for bounded W_Q sensitivity diagnostics."""

import json

import numpy as np
import pytest

from mndm.tools.wq_sensitivity import (
    _perturb_phi,
    _perturb_q,
    sensitivity_from_arrays,
)


def _contract(q):
    return {
        "schema_version": "mndm.transition_residual_covariance_proxy.v1",
        "computation_status": "computed",
        "q_time_semantics": "one_step_transition_covariance",
        "q_units": "state_squared",
        "conversion_model": "not_applicable",
        "q_scope": "recording",
        "q_semantics": "transition_residual_covariance_proxy",
        "q_dt_sec": 30.0,
        "covariance": q,
    }


def test_identity_perturbation_exercises_actual_metrics():
    phi = np.repeat((0.9 * np.eye(2))[None], 12, axis=0)
    q = np.eye(2) * 0.1
    result = sensitivity_from_arrays(
        phi, q, _contract(q), horizons=(1, 2), n_starts=2,
        epsilons=(0.0,), seeds=(0,)
    )
    assert result["protocol"]["q_perturbation"] == "PSD_congruence"
    assert result["records"]
    for row in result["records"]:
        assert row["status"] == "computed"
        assert row["delta_w_maxabs_normalized"] == 0.0
        assert row["delta_v_norm_abs"] == 0.0


def test_sensitivity_is_reproducible_and_start_order_is_frozen():
    phi = np.repeat((0.8 * np.eye(2))[None], 12, axis=0)
    q = np.eye(2) * 0.2
    kwargs = dict(horizons=(1, 4), n_starts=3, epsilons=(1e-8,), seeds=(2,))
    first = sensitivity_from_arrays(phi, q, _contract(q), **kwargs)
    second = sensitivity_from_arrays(phi, q, _contract(q), **kwargs)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["protocol"]["start_indices"] == sorted(first["protocol"]["start_indices"])


def test_perturbation_epsilon_controls_realized_input_distortion():
    phi = np.repeat((0.8 * np.eye(2))[None], 12, axis=0)
    q = np.eye(2) * 0.2
    result = sensitivity_from_arrays(
        phi, q, _contract(q), horizons=(2,), n_starts=1,
        epsilons=(1e-10, 1e-8, 1e-6), seeds=(0,)
    )
    phi_rows = [row for row in result["records"] if row["branch"] == "phi_only"]
    assert [row["perturbation_epsilon"] for row in phi_rows] == [1e-10, 1e-8, 1e-6]
    assert all(row["relative_phi_distortion"] > 0 for row in phi_rows)
    assert result["protocol"]["production_psd_floor"] == 1e-8


def test_baseline_uses_one_fixed_production_floor_across_perturbation_epsilons():
    phi = np.repeat((0.8 * np.eye(2))[None], 12, axis=0)
    q = np.eye(2) * 0.2
    result = sensitivity_from_arrays(
        phi, q, _contract(q), horizons=(2,), n_starts=1,
        epsilons=(1e-10, 1e-8, 1e-6), seeds=(0,)
    )
    baselines = [
        row["baseline"] for row in result["records"] if row["branch"] == "phi_only"
    ]
    assert len({json.dumps(item, sort_keys=True) for item in baselines}) == 1
    by_eps = {row["perturbation_epsilon"]: row["baseline"] for row in result["records"] if row["branch"] == "phi_only"}
    assert by_eps[1e-10] == by_eps[1e-8] == by_eps[1e-6]


def test_w_delta_uses_shared_base_and_variant_scale_and_preserves_gate_f_metadata():
    phi = np.repeat((0.8 * np.eye(2))[None], 12, axis=0)
    q = np.eye(2) * 0.2
    metadata = _contract(q)
    metadata["reachability_metadata"] = {
        "computation_status": "invalid",
        "failure_reason": "reachability_numerical_overflow",
    }
    result = sensitivity_from_arrays(
        phi, q, metadata, horizons=(2,), n_starts=1, epsilons=(1e-6,), seeds=(0,)
    )
    assert result["source"]["reachability_metadata"]["failure_reason"] == "reachability_numerical_overflow"
    for row in result["records"]:
        if row["status"] != "computed":
            continue
        base = np.asarray(row["base_w_q"], dtype=float)
        variant = np.asarray(row["variant_w_q"], dtype=float)
        expected = np.max(np.abs(variant - base)) / max(
            np.max(np.abs(base)), np.max(np.abs(variant)), np.finfo(float).tiny
        )
        assert np.isclose(row["delta_w_maxabs_normalized"], expected)


def test_q_perturbation_is_psd_congruence_and_phi_changes_are_seeded():
    q = np.diag([1.0, 0.2])
    perturbed_q = _perturb_q(q, 0.05, np.random.default_rng(4))
    assert np.min(np.linalg.eigvalsh(perturbed_q)) >= -1e-12
    phi = np.eye(2)[None]
    assert not np.array_equal(phi, _perturb_phi(phi, 0.05, np.random.default_rng(4)))


def test_phi_edge_perturbations_share_exact_prefixes_across_horizons():
    phi = np.repeat((0.9 * np.eye(2))[None], 4, axis=0)
    short = _perturb_phi(phi[:1], 1e-6, np.random.default_rng(7))
    long = _perturb_phi(phi[:4], 1e-6, np.random.default_rng(7))
    np.testing.assert_array_equal(short[0], long[0])


def test_bad_inputs_fail_closed():
    q = np.eye(2)
    with pytest.raises(ValueError, match="finite square"):
        sensitivity_from_arrays(np.full((2, 2, 2), np.nan), q, _contract(q))
    with pytest.raises(ValueError, match="positive"):
        sensitivity_from_arrays(np.eye(2)[None], q, _contract(q), horizons=(0,))
