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
