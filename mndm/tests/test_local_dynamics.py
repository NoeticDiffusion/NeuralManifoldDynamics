"""Analytic contract tests for versioned local-dynamics measurements."""

from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))

from mndm.dynamics.finite_time_response import compute_finite_time_response
from mndm.dynamics.jacobian_metrics import compute_jacobian_metrics
from mndm.dynamics.stochastic_reachability import (
    compute_stochastic_reachability,
    compute_stochastic_reachability_from_gate_e,
    estimate_one_step_transition_covariance,
    estimate_transition_residual_covariance_proxy,
    make_derivative_residual_covariance_proxy,
)
from mndm.dynamics.transition_residuals import (
    affine_one_step_predict,
    affine_one_step_state_matrix,
    compute_transition_residuals,
)
from mndm.dynamical_families import family_forbids
from mndm.schema import MNPSPayload
from core.io.h5_writer import write_h5


def test_jacobian_metrics_detect_stable_non_normal_reactivity() -> None:
    # Eigenvalues are -1, but the symmetric part has a positive eigenvalue.
    j = np.array([[[-1.0, 3.0], [0.0, -1.0]]])
    result = compute_jacobian_metrics(j)
    series = result["series"]
    assert series["spectral_abscissa"][0] < 0
    assert series["numerical_abscissa"][0] > 0
    assert series["stable_reactive_flag"][0] == 1
    assert result["summary"]["stable_reactive_fraction"] == 1.0


def test_jacobian_metrics_preserve_invalid_windows_and_support_counts() -> None:
    j = np.array([[[-1.0, 0.0], [0.0, -2.0]], [[np.nan, 0.0], [0.0, 1.0]]])
    result = compute_jacobian_metrics(j)
    assert result["summary"]["n_windows_total"] == 2
    assert result["summary"]["n_windows_metrics_valid"] == 1
    assert np.isnan(result["series"]["spectral_abscissa"][1])
    assert result["series"]["stable_reactive_flag"][1] == -1


def test_finite_time_response_uses_ordered_actual_steps() -> None:
    j = np.repeat(np.array([[[-1.0]]]), 3, axis=0)
    result = compute_finite_time_response(
        j,
        horizon_steps=[2],
        time=np.array([0.0, 0.25, 1.0]),
        propagator_mode="time_ordered_expm",
    )
    horizon = result["horizons"]["h2"]
    assert result["computation_status"] == "computed"
    assert horizon["n_sequences_valid"] == 1
    assert np.isclose(horizon["series"]["actual_horizon_sec"][0], 1.0)
    assert np.isclose(horizon["series"]["g_max"][0], np.exp(-1.0))


def test_finite_time_response_caches_ordered_step_exponentials(monkeypatch) -> None:
    from mndm.dynamics import finite_time_response as response_module

    j = np.repeat(np.array([[[-0.1, 0.02], [0.0, -0.2]]]), 4, axis=0)
    calls = []
    original_expm = response_module.expm

    def _counting_expm(matrix):
        calls.append(np.asarray(matrix).copy())
        return original_expm(matrix)

    monkeypatch.setattr(response_module, "expm", _counting_expm)

    result = compute_finite_time_response(
        j,
        horizon_steps=[1, 2, 3],
        time=np.arange(4, dtype=float),
        propagator_mode="time_ordered_expm",
    )

    assert result["horizons"]["h1"]["n_sequences_valid"] == 3
    assert result["horizons"]["h2"]["n_sequences_valid"] == 2
    assert result["horizons"]["h3"]["n_sequences_valid"] == 1
    # Three distinct observed steps are reused across all horizon/start windows.
    assert len(calls) == 3


def test_finite_time_response_preserves_noncommuting_step_order() -> None:
    from mndm.dynamics import finite_time_response as response_module

    j = np.array(
        [
            [[0.0, 1.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, -1.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=float,
    )
    result = compute_finite_time_response(
        j,
        horizon_steps=[2],
        time=np.array([0.0, 1.0, 2.0]),
        propagator_mode="time_ordered_expm",
    )

    expected_phi = response_module.expm(j[1]) @ response_module.expm(j[0])
    expected = response_module._response_metrics(expected_phi, 2.0)
    series = result["horizons"]["h2"]["series"]
    for name in ("g_max", "log_g_max", "gamma_1", "d_resp", "c_1"):
        assert np.isclose(series[name][0], expected[name])


def test_finite_time_response_breaks_at_removed_jacobian_centers() -> None:
    j = np.repeat(np.array([[[-1.0]]]), 3, axis=0)
    result = compute_finite_time_response(
        j,
        horizon_steps=[2],
        time=np.array([0.0, 1.0, 2.0]),
        centers=np.array([10, 11, 15]),
        validation_level="heldout_predictive",
    )
    assert result["horizons"]["h2"]["n_sequences_valid"] == 0
    assert result["validation_level"] == "heldout_predictive"


def test_finite_time_response_combines_center_and_time_breaks() -> None:
    j = np.repeat(np.array([[[-1.0]]]), 4, axis=0)
    result = compute_finite_time_response(
        j,
        horizon_steps=[2],
        centers=np.array([0, 10, 11, 12]),
        time=np.array([0.0, 1.0, 2.0, 10.0]),
    )
    assert result["horizons"]["h2"]["n_sequences_valid"] == 0


def test_residual_contract_rejects_derivative_proxy_for_reachability() -> None:
    residuals = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    proxy = make_derivative_residual_covariance_proxy(residuals, q_dt_sec=1.0)
    result = compute_stochastic_reachability([np.eye(2)], proxy["covariance"], q_contract=proxy)
    assert result["computation_status"] == "unavailable"
    assert result["failure_reason"] == "q_contract_not_admissible"


def test_reachability_distinguishes_admissible_transition_covariance() -> None:
    state = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    predicted = np.zeros_like(state)
    proxy = estimate_one_step_transition_covariance(state, predicted, q_dt_sec=1.0)
    result = compute_stochastic_reachability(
        [2.0 * np.eye(2), 2.0 * np.eye(2)],
        proxy["covariance"],
        q_contract=proxy,
    )
    assert result["computation_status"] == "computed"
    assert result["a_q"] > 0
    assert result["d_eff"] > 0


def test_reachability_rejects_empty_transitions_and_invalid_q_contract() -> None:
    state = np.array([[1.0], [2.0], [3.0]])
    proxy = estimate_one_step_transition_covariance(state, np.zeros_like(state), q_dt_sec=0.5)
    empty = compute_stochastic_reachability([], proxy["covariance"], q_contract=proxy)
    assert empty["failure_reason"] == "empty_one_step_transitions"
    proxy["computation_status"] = "invalid"
    rejected = compute_stochastic_reachability([np.eye(1)], proxy["covariance"], q_contract=proxy)
    assert rejected["failure_reason"] == "q_contract_not_admissible"


def test_writer_exports_versioned_jacobian_metrics_path(tmp_path: Path) -> None:
    payload = MNPSPayload(
        time=np.array([0.0, 1.0]),
        x=np.zeros((2, 3)),
        x_dot=np.zeros((2, 3)),
        jacobian=np.repeat(np.eye(3)[None], 1, axis=0),
        jacobian_derived_metrics=compute_jacobian_metrics(np.repeat(np.eye(3)[None], 1, axis=0)),
    )
    out = write_h5(tmp_path / "local-dynamics.h5", "test", payload)
    with h5py.File(out, "r") as handle:
        assert "/jacobian/derived_metrics/v1/series/spectral_abscissa" in handle
        assert handle["/jacobian/derived_metrics"].attrs["_schema_version"] == "mndm.jacobian_metrics.v1"


def test_writer_exports_transition_residual_contract(tmp_path: Path) -> None:
    payload = MNPSPayload(
        time=np.array([0.0, 1.0]),
        x=np.zeros((2, 3)),
        x_dot=np.zeros((2, 3)),
        transition_residuals={
            "schema_version": "mndm.transition_residuals.v1",
            "computation_status": "computed",
            "series": {"dt_sec": np.array([1.0]), "transition_residual": np.zeros((1, 3))},
        },
    )
    out = write_h5(tmp_path / "transition-residuals.h5", "test", payload)
    with h5py.File(out, "r") as handle:
        assert "/transition_residuals/v1/primary/series/transition_residual" in handle


def test_jacobian_estimate_exports_affine_fit_parameters() -> None:
    from mndm.jacobian import estimate_local_jacobians

    x = np.arange(12, dtype=np.float32).reshape(6, 2)
    xdot = 0.5 * x + 1.0
    nn = np.tile(np.arange(6, dtype=np.int32), (6, 1))
    result = estimate_local_jacobians(x, xdot, nn, super_window=1, ridge_alpha=1e-6)
    assert result.affine_reference.shape == (6, 2)
    assert result.affine_intercept.shape == (6, 2)


def test_affine_prediction_handles_singular_jacobian_exactly() -> None:
    # dx/dt = b has x(t + dt) = x(t) + b * dt even when J is singular.
    prediction = affine_one_step_predict(
        np.zeros((1, 1)),
        np.array([2.0]),
        np.array([5.0]),
        np.array([3.0]),
        0.25,
    )
    assert np.allclose(prediction, [3.5])


def test_transition_residuals_are_crossfit_and_do_not_bridge_centers() -> None:
    x = np.arange(8, dtype=float)[:, None]
    xdot = np.ones_like(x)
    nn = np.tile(np.arange(8, dtype=np.int32), (8, 1))
    result = compute_transition_residuals(
        x,
        xdot,
        np.arange(8, dtype=float),
        np.array([2, 3, 5], dtype=np.int32),
        nn,
        super_window=1,
        ridge_alpha=1e-6,
        distance_weighted=False,
        crossfit_embargo_steps=0,
        coordinate_contract="subject_anchored",
        coordinate_layer="coords_3d_subject_anchored",
    )
    assert result["computation_status"] == "computed"
    assert result["summary"]["n_transitions_valid"] == 1
    assert result["provenance"]["crossfit_status"] == "leave_one_transition_out"
    phi = np.asarray(result["series"]["phi_one_step"])
    assert phi.shape == (1, 1, 1)
    assert np.isfinite(phi).all()


def test_transition_covariance_rejects_irregular_steps() -> None:
    transitions = {
        "computation_status": "computed",
        "series": {
            "transition_residual": np.array([[1.0], [2.0], [3.0]]),
            "dt_sec": np.array([0.25, 0.25, 0.5]),
        },
        "provenance": {"crossfit_status": "leave_one_transition_out"},
    }
    proxy = estimate_transition_residual_covariance_proxy(transitions, max_dt_deviation_sec=1e-6)
    assert proxy["computation_status"] == "unavailable"
    assert proxy["failure_reason"] == "materially_irregular_dt"
    reachability = compute_stochastic_reachability_from_gate_e(
        {
            **transitions,
            "series": {
                **transitions["series"],
                "phi_one_step": np.ones((3, 1, 1), dtype=np.float32),
            },
        },
        proxy,
    )
    assert reachability["computation_status"] == "unavailable"
    assert reachability["failure_reason"] == "materially_irregular_dt"
    assert reachability["measurement_validity"] == "not_applicable"
    assert reachability["claim_status"] == "no_biological_claim"
    assert reachability["grain"]["native"] == "recording_horizon"


def test_gate_e_identity_phi_yields_nq_reachability() -> None:
    residuals = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.5, -0.5]])
    n_steps = int(residuals.shape[0])
    transitions = {
        "computation_status": "computed",
        "series": {
            "transition_residual": residuals,
            "dt_sec": np.ones(residuals.shape[0], dtype=np.float32),
            "phi_one_step": np.repeat(np.eye(2, dtype=np.float32)[None], n_steps, axis=0),
        },
        "provenance": {
            "crossfit_status": "leave_one_transition_out",
            "prediction_fit_policy": "affine_expm",
        },
    }
    proxy = estimate_transition_residual_covariance_proxy(transitions)
    result = compute_stochastic_reachability_from_gate_e(transitions, proxy)
    assert result["computation_status"] == "computed"
    assert result["measurement_validity"] == "not_assessed"
    assert result["claim_status"] == "no_biological_claim"
    assert result["grain"]["native"] == "recording_horizon"
    assert result["provenance"]["q_source"] == "transition_residual_covariance_proxy"
    assert result["provenance"]["propagator_source"] == "gate_e_crossfit_expm_J_dt"
    assert result["n_propagator_steps"] == n_steps
    expected = n_steps * np.asarray(proxy["covariance"], dtype=float)
    assert np.allclose(result["w_q"], expected, rtol=1e-5, atol=1e-6)
    assert result["q_units"] == "state_squared"
    assert result["conversion_model"] == "not_applicable"


def test_gate_e_phi_follows_q_validity_mask() -> None:
    residuals = np.array(
        [[1.0, 0.0], [-1.0, 0.0], [np.nan, 0.0], [0.0, 1.0], [0.0, -1.0], [0.5, -0.5]]
    )
    n_rows = int(residuals.shape[0])
    transitions = {
        "computation_status": "computed",
        "series": {
            "transition_residual": residuals,
            "dt_sec": np.ones(n_rows, dtype=np.float32),
            "phi_one_step": np.repeat(np.eye(2, dtype=np.float32)[None], n_rows, axis=0),
        },
        "provenance": {"crossfit_status": "leave_one_transition_out"},
    }
    proxy = estimate_transition_residual_covariance_proxy(transitions)
    result = compute_stochastic_reachability_from_gate_e(transitions, proxy)
    assert proxy["computation_status"] == "computed"
    assert result["computation_status"] == "computed"
    assert result["n_propagator_steps"] == int(proxy["q_n_samples"])
    assert result["n_propagator_steps"] == n_rows - 1
    expected = result["n_propagator_steps"] * np.asarray(proxy["covariance"], dtype=float)
    assert np.allclose(result["w_q"], expected, rtol=1e-5, atol=1e-6)


def test_gate_e_adapter_refuses_library_one_step_q_schema() -> None:
    transitions = {
        "computation_status": "computed",
        "series": {
            "transition_residual": np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
            "dt_sec": np.ones(4, dtype=np.float32),
            "phi_one_step": np.repeat(np.eye(2, dtype=np.float32)[None], 4, axis=0),
        },
    }
    library_q = estimate_one_step_transition_covariance(
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]),
        np.zeros((4, 2)),
        q_dt_sec=1.0,
    )
    result = compute_stochastic_reachability_from_gate_e(transitions, library_q)
    assert result["computation_status"] == "unavailable"
    assert result["failure_reason"] == "q_contract_not_admissible"


def test_zero_jacobian_state_matrix_is_identity() -> None:
    phi = affine_one_step_state_matrix(np.zeros((2, 2)), 0.25)
    assert np.allclose(phi, np.eye(2))


def test_missing_phi_makes_reachability_unavailable() -> None:
    transitions = {
        "computation_status": "computed",
        "series": {
            "transition_residual": np.ones((6, 2), dtype=np.float32),
            "dt_sec": np.ones(6, dtype=np.float32),
        },
    }
    proxy = estimate_transition_residual_covariance_proxy(transitions)
    result = compute_stochastic_reachability_from_gate_e(transitions, proxy)
    assert result["computation_status"] == "unavailable"
    assert result["failure_reason"] == "missing_one_step_propagators"


def test_writer_exports_stochastic_reachability_outside_families(tmp_path: Path) -> None:
    n_steps = 3
    proxy = estimate_one_step_transition_covariance(
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]),
        np.zeros((4, 2)),
        q_dt_sec=1.0,
    )
    reachability = compute_stochastic_reachability(
        [np.eye(2)] * n_steps,
        proxy["covariance"],
        q_contract=proxy,
    )
    payload = MNPSPayload(
        time=np.array([0.0, 1.0]),
        x=np.zeros((2, 3)),
        x_dot=np.zeros((2, 3)),
        stochastic_reachability=reachability,
    )
    out = write_h5(tmp_path / "reachability.h5", "test", payload)
    with h5py.File(out, "r") as handle:
        assert "/stochastic_reachability/v1/primary/w_q" in handle
        assert "/dynamical_families/spread" not in handle
        assert "spread" not in handle.get("dynamical_families", {})


def test_family_forbids_gate_e_proxy_as_process_noise() -> None:
    assert family_forbids("spread", "gate_e_proxy_as_process_noise")
    assert family_forbids("spread", "derivative_residual_covariance")
    assert family_forbids("diffusion", "jacobian_derivative_residual_as_diffusion")
    assert family_forbids("diffusion", "mnps_xdot_as_sde_drift")
    assert family_forbids("diffusion", "jacobian_intercept_as_sde_drift")
    assert family_forbids("diffusion", "local_increment_mean_as_sde_drift")


def test_common_profiles_do_not_enable_reachability_or_ftr() -> None:
    import yaml

    config_root = Path(__file__).resolve().parents[1] / "config"
    for name in (
        "config_ingest_common_eeg.yaml",
        "config_ingest_common_fmri.yaml",
        "config_ingest_common_ephys.yaml",
        "config_ingest_common_nwb.yaml",
        "config_ingest_common_nwb_mouse_eeg.yaml",
        "config_ingest_common_nwb_rodent.yaml",
    ):
        data = yaml.safe_load((config_root / name).read_text(encoding="utf-8")) or {}
        local = data.get("local_dynamics") or {}
        assert not bool((local.get("stochastic_reachability") or {}).get("enabled", False)), name
        assert not bool((local.get("finite_time_response") or {}).get("enabled", False)), name
        assert not bool((local.get("transition_residuals") or {}).get("enabled", False)), name
    ds004100 = yaml.safe_load(
        (config_root / "sources" / "openneuro" / "config_ingest_ds004100.yaml").read_text(encoding="utf-8")
    ) or {}
    ds_local = ds004100.get("local_dynamics") or {}
    assert not bool((ds_local.get("stochastic_reachability") or {}).get("enabled", False))
