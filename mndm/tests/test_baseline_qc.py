"""Tests for reviewer-facing baseline and null QA helpers."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def test_compute_feature_baseline_comparisons_exports_expected_families():
    from mndm.pipeline.baseline_qc import compute_feature_baseline_comparisons

    t = np.linspace(0.0, 6.0 * np.pi, 64, dtype=float)
    x = np.stack(
        [
            np.sin(t),
            np.cos(t),
            np.sin(0.5 * t + 0.3),
        ],
        axis=1,
    ).astype(np.float32)
    sub_frame = pd.DataFrame(
        {
            "eeg_permutation_entropy": np.sin(t) + 0.1,
            "eeg_sample_entropy": np.cos(t) - 0.2,
            "eeg_alpha": np.sin(0.25 * t) + 2.0,
            "eeg_beta": np.cos(0.25 * t) + 1.5,
            "eeg_dfc_variance": np.sin(0.1 * t) + 0.5,
            "eeg_entropy_degraded_mode": np.zeros_like(t),
        }
    )

    out = compute_feature_baseline_comparisons(
        sub_frame=sub_frame,
        x=x,
        dt_sec=4.0,
        review_qc_cfg={
            "baseline_comparisons": {
                "enabled": True,
                "smoothing_window_epochs": 5,
                "max_series_per_family": 4,
            }
        },
    )

    assert out is not None
    assert set(out["nmd_axes"].keys()) == {"m", "d", "e"}
    assert "entropy_raw" in out["families"]
    assert "entropy_smoothed" in out["families"]
    assert "variance_bandpower" in out["families"]
    assert "sliding_window_fc" in out["families"]
    assert "eeg_entropy_degraded_mode" not in out["families"]["entropy_raw"]


def test_compute_null_sanity_tests_returns_three_surrogates():
    from mndm.pipeline.baseline_qc import compute_null_sanity_tests

    t = np.linspace(0.0, 8.0 * np.pi, 96, dtype=float)
    x = np.stack(
        [
            np.sin(t),
            np.cos(t),
            np.sin(0.4 * t + 0.2),
        ],
        axis=1,
    ).astype(np.float32)
    file_labels = np.array(["file_a"] * 48 + ["file_b"] * 48, dtype=object)

    out = compute_null_sanity_tests(
        x=x,
        dt_sec=2.0,
        derivative_cfg={"method": "central", "window": 5, "polyorder": 2},
        derivative_robust_cfg={"enabled": True, "max_jump": 5.0, "min_seg": 9},
        file_labels=file_labels,
        knn_k=8,
        knn_metric="euclidean",
        whiten=True,
        super_window=3,
        ridge_alpha=1e-3,
        distance_weighted=True,
        review_qc_cfg={"null_sanity_tests": {"enabled": True, "seed": 7}},
    )

    assert out is not None
    assert out["status"] == "ok"
    assert out["source_level"] == "mnps_3d"
    assert set(out["surrogates"].keys()) == {"shuffled_time", "phase_randomized", "white_noise"}
    assert set(out["comparisons_to_original"].keys()) == {"shuffled_time", "phase_randomized", "white_noise"}
    for name, summary in out["surrogates"].items():
        assert "jacobian" in summary
        assert "tau_summary" in summary
        assert summary["jacobian"]["windows"] >= 0
        assert name in out["comparisons_to_original"]
