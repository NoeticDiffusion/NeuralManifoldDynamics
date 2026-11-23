"""MNPS 3.4 flat-nine Jacobian adapter."""

from __future__ import annotations

import numpy as np

from mndm.mnps.v3_4.jacobian_adapter import attach_meg_v3_4_jacobians, attach_v3_4_jacobian


def _trajectory(n: int = 48, dt: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(n, dtype=np.float64) * dt
    phases = np.linspace(0.15, 0.9, 9)
    values = np.column_stack([np.sin(phase * t + 0.2 * i) for i, phase in enumerate(phases)])
    start = t
    end = t + 8.0
    return values, start, end


def _payload(values: np.ndarray, start: np.ndarray, end: np.ndarray, *, anchor_status: str, complete: np.ndarray | None = None) -> dict:
    n = int(values.shape[0])
    support = np.isfinite(values)
    if complete is None:
        complete = np.zeros((n,), dtype=bool)
    return {
        "coordinate_order": [
            "delta_theta_level",
            "alpha_level",
            "beta_alpha_ratio",
            "gamma_level",
            "hjorth_mobility",
            "alpha_theta_ratio",
            "permutation_entropy",
            "alpha_centroid_hz",
            "alpha_envelope_cv",
        ],
        "raw_coordinates": {"values": values, "support": support},
        "coordinates": {"values": np.full_like(values, np.nan)},
        "complete_9d_mask": complete,
        "window_start_sec": start,
        "window_end_sec": end,
        "provenance": {"anchor_application_status": anchor_status},
    }


def _config(**jacobian) -> dict:
    block = {"enabled": True, "whiten": False}
    block.update(jacobian)
    return {
        "schema": "mnps_flat_coordinates",
        "definition_version": "3.4",
        "enabled": True,
        "mnps": {
            "knn_k": 12,
            "super_window": 3,
            "ridge_alpha": 1.0,
            "knn_metric": "euclidean",
            "derivative": {"method": "sav_gol", "window": 7, "polyorder": 3},
            "derivative_robust": {"enabled": True, "min_seg": 9},
        },
        "v3_4_jacobian": block,
    }


def test_raw_layer_emits_finite_nine_by_nine_jacobian():
    values, start, end = _trajectory()
    values = values.copy()
    values[5, 8] = np.nan
    payload = _payload(values, start, end, anchor_status="unsupported_anchor_application")
    out = attach_v3_4_jacobian(payload, _config())
    jac = out["jacobian"]
    assert jac["computation_status"] == "computed"
    assert jac["coordinate_layer"] == "v3_4_raw"
    assert jac["J_hat"].shape[1:] == (9, 9)
    assert jac["J_hat"].shape[0] == jac["window_index"].shape[0]
    assert np.isfinite(jac["J_hat"]).all()
    assert np.any(np.abs(jac["J_hat"]) > 1e-6)
    assert 5 not in set(int(i) for i in jac["window_index"])
    abscissa = np.asarray(jac["derived_metrics"]["series"]["spectral_abscissa"], dtype=float)
    assert abscissa.shape[0] == jac["J_hat"].shape[0]
    assert np.isfinite(abscissa).all()
    assert jac["derived_metrics"]["computation_status"] == "computed"


def test_anchored_layer_is_used_only_after_the_anchor_applied():
    values, start, end = _trajectory()
    anchored = values * 0.1 + 0.3
    payload = _payload(
        values,
        start,
        end,
        anchor_status="computed",
        complete=np.ones((values.shape[0],), dtype=bool),
    )
    payload["coordinates"]["values"] = anchored
    out = attach_v3_4_jacobian(payload, _config())
    assert out["jacobian"]["coordinate_layer"] == "v3_4_anchored"
    assert out["jacobian"]["computation_status"] == "computed"


def test_disabled_flag_leaves_the_coordinate_payload_without_a_jacobian():
    values, start, end = _trajectory()
    payload = _payload(values, start, end, anchor_status="unsupported_anchor_application")
    out = attach_v3_4_jacobian(payload, _config(enabled=False))
    assert "jacobian" not in out


def test_short_series_does_not_emit_a_zero_jacobian():
    values, start, end = _trajectory(n=4)
    payload = _payload(values, start, end, anchor_status="unsupported_anchor_application")
    out = attach_v3_4_jacobian(payload, _config())
    jac = out["jacobian"]
    assert jac["computation_status"] in {"insufficient_support", "not_testable"}
    assert "J_hat" not in jac


def test_missing_flag_defaults_to_on():
    values, start, end = _trajectory()
    payload = _payload(values, start, end, anchor_status="unsupported_anchor_application")
    config = _config()
    del config["v3_4_jacobian"]
    out = attach_v3_4_jacobian(payload, config)
    assert out["jacobian"]["computation_status"] == "computed"


def test_meg_jacobians_keep_sensor_families_and_coordinate_variants_separate():
    values, start, end = _trajectory()
    family_payload = {
        "variants": {
            "existing_coordinate_set": {
                "coordinate_names": [f"mag_{idx}" for idx in range(9)],
                "coordinate_units": ["native"] * 9,
                "raw_values": values,
                "coordinate_support": np.ones_like(values, dtype=np.uint8),
                "complete_window_mask": np.ones(values.shape[0], dtype=np.uint8),
            },
            "removal_only": {
                "coordinate_names": [f"mag_{idx}" for idx in range(7)],
                "coordinate_units": ["native"] * 7,
                "raw_values": values[:, :7],
                "coordinate_support": np.ones(values[:, :7].shape, dtype=np.uint8),
                "complete_window_mask": np.ones(values.shape[0], dtype=np.uint8),
            },
        }
    }
    payload = {
        "window_axis": {"window_start_sec": start, "window_end_sec": end},
        "anchor_application_status": "not_applied",
        "sensor_families": {"mag": family_payload, "grad": family_payload},
    }
    config = _config()
    inherited_eeg_only = attach_meg_v3_4_jacobians(payload, config)
    assert "jacobian" not in inherited_eeg_only["sensor_families"]["mag"]["variants"]["existing_coordinate_set"]
    config["v3_4_meg_jacobian"] = config.pop("v3_4_jacobian")
    out = attach_meg_v3_4_jacobians(payload, config)

    mag_9d = out["sensor_families"]["mag"]["variants"]["existing_coordinate_set"]["jacobian"]
    grad_9d = out["sensor_families"]["grad"]["variants"]["existing_coordinate_set"]["jacobian"]
    mag_7d = out["sensor_families"]["mag"]["variants"]["removal_only"]["jacobian"]
    assert mag_9d["computation_status"] == "computed"
    assert grad_9d["computation_status"] == "computed"
    assert mag_9d["sensor_family"] == "mag"
    assert grad_9d["sensor_family"] == "grad"
    assert mag_9d["coordinate_variant"] == "existing_coordinate_set"
    assert mag_7d["computation_status"] == "not_testable"
    assert mag_7d["failure_reason"] == "v3_4_jacobian_requires_flat_nine"
