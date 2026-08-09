"""Tests for AnchorState export helpers."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def test_build_anchor_state_exports_creates_values_dot_and_quality():
    """AnchorState helper should build aligned state, derivative, and quality groups."""
    from mndm.pipeline.anchor_state import build_anchor_state_exports

    features_df = pd.DataFrame(
        {
            "ecg_quality_score": [0.9, 0.8, 0.7],
            "ppg_quality_score": [0.6, 0.7, 0.8],
            "pupil_quality_score": [1.0, 0.5, 0.0],
        }
    )
    names = [
        "ecg_hr_bpm",
        "ecg_rmssd",
        "ppg_rate_bpm",
        "ppg_amplitude_mean",
        "ppg_amplitude_cv",
        "pupil_dilation_velocity",
        "pupil_diameter_std",
        "pupil_diameter_mean",
    ]
    robust_z = np.asarray(
        [
            [0.5, 1.0, 0.2, 0.3, -0.1, 0.4, 0.1, 0.2],
            [0.6, 0.8, 0.1, 0.2, -0.2, 0.5, 0.2, 0.3],
            [0.7, 0.6, 0.0, 0.1, -0.3, 0.6, 0.3, 0.4],
        ],
        dtype=np.float32,
    )
    anchor_state, anchor_state_dot, anchor_quality, diagnostics = build_anchor_state_exports(
        features_df=features_df,
        robust_z_values=robust_z,
        robust_z_names=names,
        time=np.asarray([0.0, 2.0, 4.0], dtype=np.float64),
        config={"anchor_state": {"enabled": True}},
    )

    assert anchor_state["values"].shape == (3, 5)
    assert anchor_state["names"][-1] == "anchor_index"
    assert anchor_state_dot["values"].shape == (3, 5)
    assert anchor_quality["values"].shape[0] == 3
    assert "anchor_support_fraction" in anchor_quality["names"]
    assert diagnostics["enabled"] is True


def test_build_anchor_state_exports_prefers_hrv_superwindow_features_when_present():
    """AnchorState should use HRV v0.1 columns before legacy short-window ECG fallbacks."""
    from mndm.pipeline.anchor_state import build_anchor_state_exports

    features_df = pd.DataFrame(
        {
            "ecg_hrv_quality_score": [0.9, 0.8, 0.7],
            "ecg_quality_score": [0.1, 0.1, 0.1],
        }
    )
    names = [
        "ecg_hr_bpm",
        "ecg_rmssd",
        "ecg_sdnn",
        "ecg_hrv_hr_mean_bpm",
        "ecg_hrv_rmssd_ms",
        "ecg_hrv_sdnn_ms",
        "ppg_rate_bpm",
        "ppg_amplitude_mean",
        "ppg_amplitude_cv",
        "pupil_dilation_velocity",
        "pupil_diameter_std",
        "pupil_diameter_mean",
    ]
    robust_z = np.asarray(
        [
            [0.1, 0.1, 0.1, 0.8, 1.2, 0.9, 0.2, 0.3, -0.1, 0.4, 0.1, 0.2],
            [0.2, 0.2, 0.2, 0.7, 1.0, 0.8, 0.1, 0.2, -0.2, 0.5, 0.2, 0.3],
            [0.3, 0.3, 0.3, 0.6, 0.8, 0.7, 0.0, 0.1, -0.3, 0.6, 0.3, 0.4],
        ],
        dtype=np.float32,
    )

    _, _, anchor_quality, diagnostics = build_anchor_state_exports(
        features_df=features_df,
        robust_z_values=robust_z,
        robust_z_names=names,
        time=np.asarray([0.0, 2.0, 4.0], dtype=np.float64),
        config={"anchor_state": {"enabled": True}},
    )

    assert diagnostics["source_features"]["vagal_index"]["positive"] == ["ecg_hrv_rmssd_ms", "ecg_hrv_sdnn_ms"]
    assert diagnostics["source_features"]["vagal_index"]["negative"] == ["ecg_hrv_hr_mean_bpm"]
    ecg_quality_idx = anchor_quality["names"].index("ecg_quality")
    assert np.allclose(anchor_quality["values"][:, ecg_quality_idx], np.array([0.9, 0.8, 0.7], dtype=np.float32))


def test_anchor_state_rejects_eog_only_sympathetic_fallback():
    """EOG alone is ocular evidence and cannot create a physiological anchor."""
    from mndm.pipeline.anchor_state import build_anchor_state_exports

    features_df = pd.DataFrame({"eog_blink_rate": [0.0, 0.0, 0.0, 0.875]})
    anchor_state, _, anchor_quality, _ = build_anchor_state_exports(
        features_df=features_df,
        robust_z_values=np.asarray([[0.0], [0.0], [0.0], [8.75e8]], dtype=np.float32),
        robust_z_names=["eog_blink_rate"],
        time=np.arange(4, dtype=np.float64),
        config={"anchor_state": {"enabled": True}},
    )

    sympathetic_idx = anchor_state["names"].index("sympathetic_index")
    anchor_idx = anchor_state["names"].index("anchor_index")
    eligible_idx = anchor_quality["names"].index("sympathetic_index_eligible")
    valid_idx = anchor_quality["names"].index("sympathetic_index_valid")
    assert np.isnan(anchor_state["values"][:, sympathetic_idx]).all()
    assert np.isnan(anchor_state["values"][:, anchor_idx]).all()
    assert np.all(anchor_quality["values"][:, eligible_idx] == 0.0)
    assert np.all(anchor_quality["values"][:, valid_idx] == 0.0)


def test_anchor_state_invalidates_degenerate_provenance_and_empty_composite():
    """A finite legacy-scale input is rejected when robust-z provenance says invalid."""
    from mndm.pipeline.anchor_state import build_anchor_state_exports

    features_df = pd.DataFrame({"ecg_hr_bpm": [60.0, 60.0, 60.0, 65.0]})
    anchor_state, _, anchor_quality, _ = build_anchor_state_exports(
        features_df=features_df,
        robust_z_values=np.asarray([[0.0], [0.0], [0.0], [5.0e8]], dtype=np.float32),
        robust_z_names=["ecg_hr_bpm"],
        time=np.arange(4, dtype=np.float64),
        feature_metadata={
            "robust_z_valid": np.asarray([1], dtype=np.int8),
            "robust_z_invalid_reason": np.asarray(["degenerate_scale"], dtype=object),
        },
        config={"anchor_state": {"enabled": True}},
    )

    sympathetic_idx = anchor_state["names"].index("sympathetic_index")
    anchor_idx = anchor_state["names"].index("anchor_index")
    valid_idx = anchor_quality["names"].index("anchor_index_valid")
    assert np.isnan(anchor_state["values"][:, sympathetic_idx]).all()
    assert np.isnan(anchor_state["values"][:, anchor_idx]).all()
    assert np.all(anchor_quality["values"][:, valid_idx] == 0.0)


def test_anchor_state_preserves_well_supported_multimodal_values():
    """Guardrails leave a valid ECG/PPG/pupil anchor calculation unchanged."""
    from mndm.pipeline.anchor_state import build_anchor_state_exports

    names = ["ecg_hr_bpm", "ecg_rmssd", "ppg_rate_bpm", "pupil_dilation_velocity"]
    robust_z = np.asarray(
        [
            [0.2, 0.1, 0.4, 0.8],
            [0.4, 0.2, 0.2, 0.6],
            [0.6, 0.3, 0.0, 0.4],
        ],
        dtype=np.float32,
    )
    features_df = pd.DataFrame(
        {
            "ecg_hr_bpm": [60.0, 62.0, 64.0],
            "ecg_rmssd": [30.0, 29.0, 28.0],
            "ppg_rate_bpm": [61.0, 63.0, 65.0],
            "pupil_dilation_velocity": [0.1, 0.2, 0.3],
        }
    )
    anchor_state, _, anchor_quality, _ = build_anchor_state_exports(
        features_df=features_df,
        robust_z_values=robust_z,
        robust_z_names=names,
        time=np.arange(3, dtype=np.float64),
        config={"anchor_state": {"enabled": True}},
    )

    sympathetic_idx = anchor_state["names"].index("sympathetic_index")
    expected_positive = robust_z[:, [0, 2, 3]].mean(axis=1)
    expected = np.vstack([expected_positive, -robust_z[:, 1]]).mean(axis=0)
    valid_idx = anchor_quality["names"].index("sympathetic_index_valid")
    assert np.allclose(anchor_state["values"][:, sympathetic_idx], expected)
    assert np.all(anchor_quality["values"][:, valid_idx] == 1.0)
    assert anchor_quality["attrs"]["quality_surface"] == "v2"


def test_anchor_state_scale_validator_reports_gross_finite_values():
    """Validator exposes an auditable failure without mutating the export."""
    from mndm.pipeline.anchor_state import validate_anchor_state_exports

    result = validate_anchor_state_exports(
        {
            "values": np.asarray([[0.0, 0.1], [0.0, 2.0e4], [0.0, 0.2]], dtype=np.float32),
            "names": ["anchor_index", "sympathetic_index"],
        },
        policy={
            "enabled": True,
            "blocking": False,
            "abs_max": 1.0e4,
            "max_over_iqr": 100.0,
            "guard_policy_version": "mndm.anchor_guard.v1",
        },
    )

    sympathetic = result["components"]["sympathetic_index"]
    assert result["status"] == "fail"
    assert sympathetic["finite_count"] == 3
    assert sympathetic["invalid_count"] == 1
    assert sympathetic["max_abs"] == 2.0e4
    assert sympathetic["status"] == "fail"
