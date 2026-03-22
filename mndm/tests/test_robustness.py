"""Tests for robustness summary computation."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def test_compute_robust_summary_empty():
    """Test that empty data returns valid summary."""
    from mndm.robustness import compute_robust_summary
    
    rdm_df = pd.DataFrame()
    transient_mask = np.array([])
    config = {"robustness": {"summary": "median", "seed": 42}}
    
    out = compute_robust_summary(rdm_df, transient_mask, config)
    
    assert isinstance(out, dict)
    assert "incl" in out
    assert "excl" in out
    assert "coverage_ok" in out
    assert out["coverage_ok"] is False


def test_compute_robust_summary_basic():
    """Test basic summary computation."""
    from mndm.robustness import compute_robust_summary
    
    # Create synthetic R/D/M data
    rdm_df = pd.DataFrame({
        "epoch_id": range(10),
        "R": np.random.randn(10),
        "D": np.random.randn(10),
        "M": np.random.randn(10)
    })
    
    transient_mask = np.zeros(10, dtype=bool)
    config = {
        "robustness": {
            "summary": "median",
            "bootstrap_n": 100,
            "seed": 42,
            "coverage": {"min_epochs": 5}
        }
    }
    
    out = compute_robust_summary(rdm_df, transient_mask, config)
    
    assert isinstance(out, dict)
    assert "incl" in out
    assert "excl" in out
    assert "ci95" in out
    assert "stability" in out
    assert "coverage_ok" in out
    assert "transient_frac" in out
    
    # Legacy R/D/M input should be normalized to canonical m/d/e output.
    assert "m" in out["incl"]
    assert "d" in out["incl"]
    assert "e" in out["incl"]


def test_bootstrap_reproducibility():
    """Test that bootstrap with fixed seed is reproducible."""
    from mndm.robustness import compute_robust_summary
    
    rdm_df = pd.DataFrame({
        "epoch_id": range(20),
        "R": np.random.randn(20),
        "D": np.random.randn(20),
        "M": np.random.randn(20)
    })
    
    transient_mask = np.zeros(20, dtype=bool)
    config = {
        "robustness": {
            "summary": "median",
            "bootstrap_n": 100,
            "seed": 42,
            "coverage": {"min_epochs": 5}
        }
    }
    
    # Run twice with same seed
    out1 = compute_robust_summary(rdm_df, transient_mask, config)
    out2 = compute_robust_summary(rdm_df, transient_mask, config)
    
    # CI95 should be identical
    assert out1["ci95"]["m"] == out2["ci95"]["m"]
    assert out1["ci95"]["d"] == out2["ci95"]["d"]
    assert out1["ci95"]["e"] == out2["ci95"]["e"]


def test_compute_robust_summary_handles_mask_length_mismatch():
    """Mismatched transient masks should not break positional filtering."""
    from mndm.robustness import compute_robust_summary

    rdm_df = pd.DataFrame(
        {
            "R": [1.0, 2.0, np.nan, 4.0],
            "D": [1.0, 2.0, 3.0, 4.0],
            "M": [1.0, 2.0, 3.0, 4.0],
        }
    )
    out = compute_robust_summary(
        rdm_df,
        transient_mask=np.array([True, False], dtype=bool),
        cfg={"robustness": {"summary": "mean", "bootstrap_n": 16, "seed": 7}},
    )
    assert set(out["incl"].keys()) == {"m", "d", "e"}
    assert np.isfinite(out["incl"]["d"])
    assert np.isfinite(out["excl"]["e"])


def test_summarize_array_and_split_half():
    """Test new robustness helpers for subcoordinates and reliability."""
    from mndm.robustness import summarize_array, split_half_reliability

    rng = np.random.default_rng(123)
    # Create synthetic data: 100 epochs, 3 subcoordinates
    base = rng.normal(size=(100, 1))
    values = np.hstack([base, base * 2.0, rng.normal(size=(100, 1))])
    names = ["s1", "s2", "noise"]

    cfg = {"robustness": {"summary": "median", "trim_pct": 0.2, "bootstrap_n": 100, "seed": 7}}

    summary = summarize_array(values, names, cfg)
    assert set(summary.keys()) == set(names)
    for item in summary.values():
        assert "point" in item and "ci_low" in item and "ci_high" in item

    rel = split_half_reliability(values, names)
    assert set(rel.keys()) == set(names)
    # Reliability for first two should be higher than for pure noise
    assert rel["s1"] > rel["noise"]
    assert rel["s2"] > rel["noise"]


def test_entropy_sanity_checks_flags_degenerate():
    """Entropy sanity checks should flag flat/low-variance entropy axes as provisional."""
    from mndm.robustness import entropy_sanity_checks

    # Create coords_9d with columns [e_e, e_s, e_m] nearly constant
    T = 50
    coords = np.hstack([
        0.5 * np.ones((T, 1)),
        0.2 * np.ones((T, 1)),
        0.8 * np.ones((T, 1)),
    ])
    names = ["e_e", "e_s", "e_m"]

    out = entropy_sanity_checks(coords, names)
    assert set(out.keys()) == set(names)
    for info in out.values():
        assert info["n_unique"] < 3 or info["var"] < 1e-4
        assert "nan_frac" in info
        assert info["provisional"] is True


def test_compute_tau_summary_marks_nan_when_series_has_missing():
    """Tau should be undefined if a column has NaNs (strict policy)."""
    from mndm.pipeline.robustness_helpers import compute_tau_summary

    values = np.array(
        [
            [1.0, 0.1],
            [2.0, np.nan],
            [3.0, 0.3],
            [4.0, 0.4],
            [5.0, 0.5],
            [6.0, 0.6],
            [7.0, 0.7],
            [8.0, 0.8],
        ],
        dtype=float,
    )
    out = compute_tau_summary(values, ["a", "b"], dt_sec=1.0)
    assert np.isfinite(out["a"]["tau_sec"])
    assert np.isclose(out["a"]["nan_frac"], 0.0)
    assert np.isnan(out["b"]["tau_sec"])
    assert np.isclose(out["b"]["nan_frac"], 1.0 / 8.0)


def test_compute_tau_summary_interpolates_when_configured():
    """Tau should be computable with interpolation policy for sparse NaNs."""
    from mndm.pipeline.robustness_helpers import compute_tau_summary

    values = np.array(
        [
            [1.0, 0.1],
            [2.0, np.nan],
            [3.0, 0.3],
            [4.0, 0.4],
            [5.0, 0.5],
            [6.0, 0.6],
            [7.0, 0.7],
            [8.0, 0.8],
        ],
        dtype=float,
    )
    out = compute_tau_summary(values, ["a", "b"], dt_sec=1.0, nan_policy="interpolate")
    assert np.isfinite(out["a"]["tau_sec"])
    assert np.isfinite(out["b"]["tau_sec"])
    assert out["b"]["nan_policy"] == "interpolate"


def test_compute_tier2_jacobian_metrics_reports_subsampling():
    """Condition-number stats should report when window subsampling is used."""
    from mndm.pipeline.robustness_helpers import compute_tier2_jacobian_metrics

    W = 6000
    J = np.repeat(np.eye(3, dtype=float)[None, :, :], W, axis=0)
    out = compute_tier2_jacobian_metrics(J, max_windows_for_condition_number=5000)
    cond = out["jacobian_condition_number"]
    assert cond["subsampled"] is True
    assert cond["estimated_on_windows"] == 5000
    assert cond["total_finite_windows"] == W


def test_compute_tier2_jacobian_metrics_includes_rel_mse_baseline():
    """Tier-2 summary should include rel_mse_baseline descriptives when available."""
    from mndm.pipeline.robustness_helpers import compute_tier2_jacobian_metrics

    J = np.repeat(np.eye(3, dtype=float)[None, :, :], 4, axis=0)
    jac_diag = {
        "rel_mse_baseline_windows": np.array([0.6, 0.8, 1.1, 0.9], dtype=float),
    }
    out = compute_tier2_jacobian_metrics(J, jacobian_diagnostics=jac_diag, max_windows_for_condition_number=10)
    rel = out["rel_mse_baseline"]
    assert rel["n"] == 4
    assert np.isclose(rel["median"], 0.85)


def test_compute_emmi_metrics_uses_safe_ratio_near_zero_denominator():
    """E/M ratio should be NaN when denominator median is near zero."""
    from mndm.pipeline.robustness_helpers import compute_emmi_metrics

    x = np.array(
        [
            [1e-8, 2.0, 3.0],
            [-1e-8, 2.0, 3.0],
            [2e-8, 2.0, 3.0],
            [-2e-8, 2.0, 3.0],
        ],
        dtype=float,
    )
    x_dot = np.ones_like(x, dtype=float)
    out = compute_emmi_metrics(x=x, x_dot=x_dot)
    assert np.isnan(out["emmi_e_over_m_median"])
    assert np.isfinite(out["mv_abs_median"])


def test_build_qc_summary_reports_stem_collisions(tmp_path):
    """QC summary should expose ambiguous stem mappings."""
    from mndm.pipeline.robustness_helpers import build_qc_summary

    qc_dir = tmp_path / "qc_artifacts"
    qc_dir.mkdir(parents=True, exist_ok=True)
    (qc_dir / "run01_qc_artifacts.json").write_text(
        '{"artifact":{"method":"asr","bad_eeg_channels":["Fz"]}}',
        encoding="utf-8",
    )

    sub_frame = pd.DataFrame(
        {
            "file": [
                "dir_a/run01.edf",
                "dir_b/run01.set",
            ]
        }
    )
    qc = build_qc_summary(
        dataset_label="dsX",
        ds_path=tmp_path,
        sub_id="sub-01",
        ses_id="ses-01",
        sub_frame=sub_frame,
        dt=1.0,
        ensemble_summary=None,
        robust_summary=None,
        dist_summary=None,
        entropy_qc=None,
    )
    collisions = qc["artifacts"]["stem_collisions"]
    assert "run01" in collisions
    assert sorted(collisions["run01"]) == ["run01.edf", "run01.set"]


