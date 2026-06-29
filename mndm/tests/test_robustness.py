"""Tests for robustness summary computation."""

import json
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


def test_compute_window_time_audit_uses_realized_bounds():
    """Time-grid audit should recover realized dt and window length from bounds."""
    from mndm.pipeline.robustness_helpers import compute_window_time_audit

    time = np.array([2.0, 5.0, 8.0, 11.0], dtype=float)
    window_start = np.array([0.0, 3.0, 6.0, 9.0], dtype=float)
    window_end = np.array([4.0, 7.0, 10.0, 13.0], dtype=float)

    out = compute_window_time_audit(
        time=time,
        window_start=window_start,
        window_end=window_end,
        dt_sec_runtime=3.0,
        dt_sec_config=3.0,
        window_sec_config=4.0,
    )

    assert out["status"] == "ok"
    assert out["dt_source"] == "window_start"
    assert out["dt_matches_runtime"] is True
    assert out["dt_matches_config_formula"] is True
    assert out["window_len_matches_config"] is True
    assert np.isclose(out["dt_intervals_sec"]["median"], 3.0)
    assert np.isclose(out["window_lengths_sec"]["median"], 4.0)


def test_compute_derivative_self_consistency_flags_large_mismatch():
    """Derivative QA should warn when `mnps_3d_dot` disagrees with finite differences."""
    from mndm.pipeline.robustness_helpers import compute_derivative_self_consistency

    x = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    x_dot = np.full_like(x, 10.0, dtype=float)
    time = np.arange(len(x), dtype=float)

    out = compute_derivative_self_consistency(
        x=x,
        x_dot=x_dot,
        time=time,
        dt_sec=1.0,
        derivative_cfg={"window": 5},
        finite_interval_warn_frac=0.95,
        rel_error_warn=1.0,
        speed_ratio_warn=5.0,
        warn_bad_frac=0.05,
    )

    assert out["status"] == "warning"
    assert out["intervals_compared"] == 4
    assert out["bad_interval_fraction"] > 0.05
    assert out["vector_rel_error"]["p95"] > 1.0


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


def test_compute_mnps_mnj_sanity_respects_disabled_config():
    """MNPS/MNJ sanity helper should be opt-in."""
    from mndm.pipeline.robustness_helpers import compute_mnps_mnj_sanity

    out = compute_mnps_mnj_sanity(
        x=np.ones((8, 3), dtype=float),
        x_dot=np.ones((8, 3), dtype=float),
        time=np.arange(8, dtype=float),
        dt_sec=1.0,
        coords_9d=np.ones((8, 9), dtype=float),
        coords_9d_names=["m_a", "m_e", "m_o", "d_n", "d_l", "d_s", "e_e", "e_s", "e_m"],
        jacobian=np.repeat(np.eye(3, dtype=float)[None, :, :], 4, axis=0),
        jacobian_diagnostics={},
        review_qc_cfg={},
        projection_contract={},
        file_labels=None,
        derivative_cfg={"window": 5},
    )
    assert out is None


def test_compute_standard_geometry_contract_marks_primary_from_v2_rows_invalid():
    """Primary geometry should drop non-finite v2 rows when 3D depends on v2."""
    from mndm.pipeline.robustness_helpers import compute_standard_geometry_contract

    x = np.array(
        [
            [0.0, 0.1, 0.2],
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5],
        ],
        dtype=float,
    )
    coords_9d = np.tile(np.linspace(0.1, 0.9, 9, dtype=float), (4, 1))
    coords_9d[2, 6] = np.nan
    out = compute_standard_geometry_contract(
        x=x,
        coords_9d=coords_9d,
        coords_9d_names=["m_a", "m_e", "m_o", "d_n", "d_l", "d_s", "e_e", "e_s", "e_m"],
        primary_requires_coords_9d=True,
    )

    assert out["status"] == "adjusted"
    assert out["shared_time_grid"]["epochs_before_policy"] == 4
    assert out["shared_time_grid"]["epochs_dropped"] == 1
    assert out["coords_9d"]["nonfinite_row_count"] == 1


def test_apply_standard_jacobian_window_policy_filters_near_singular_windows():
    """Standard Jacobian policy should remove near-singular windows from canonical output."""
    from mndm.jacobian import JacobianResult
    from mndm.pipeline.robustness_helpers import apply_standard_jacobian_window_policy

    j_hat = np.array(
        [
            np.eye(3, dtype=np.float32),
            np.diag([1.0, 1e-12, 1e-12]).astype(np.float32),
            (2.0 * np.eye(3, dtype=np.float32)),
        ]
    )
    j_dot = np.gradient(j_hat, axis=0).astype(np.float32)
    jac = JacobianResult(
        j_hat=j_hat,
        j_dot=j_dot,
        centers=np.array([1, 2, 3], dtype=np.int32),
        diagnostics={"condition_number_windows": np.array([1.0, 1e12, 2.0], dtype=np.float32)},
    )

    filtered, summary = apply_standard_jacobian_window_policy(jac, condition_number_max=1e10)

    assert filtered is not None
    assert filtered.j_hat.shape[0] == 2
    assert filtered.centers.tolist() == [1, 3]
    assert summary["invalid_windows"] == 1
    assert summary["status"] == "adjusted"
    assert filtered.diagnostics["hard_invalid_centers"].tolist() == [2]


def test_compute_mnps_mnj_sanity_flags_9d_and_mnj_instability():
    """Sanity helper should flag broken 9D support and ill-conditioned MNJ fits."""
    from mndm.pipeline.robustness_helpers import compute_mnps_mnj_sanity

    x = np.array(
        [
            [0.0, 0.1, 0.2],
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5],
            [0.4, 0.5, 0.6],
            [0.5, 0.6, 0.7],
        ],
        dtype=float,
    )
    coords_9d = np.column_stack(
        [
            np.linspace(0.1, 0.6, 6),
            np.linspace(0.2, 0.7, 6),
            np.linspace(0.3, 0.8, 6),
            np.linspace(0.4, 0.9, 6),
            np.linspace(0.5, 1.0, 6),
            np.linspace(0.6, 1.1, 6),
            np.full(6, np.nan),
            np.full(6, 0.25),
            np.linspace(0.8, 1.3, 6),
        ]
    )
    names = ["m_a", "m_e", "m_o", "d_n", "d_l", "d_s", "e_e", "e_s", "e_m"]
    jacobian = np.repeat(np.diag([1.0, 1e-8, 1e-8])[None, :, :], 4, axis=0)
    jac_diag = {"rel_mse_baseline_windows": np.array([1.2, 1.3, 1.1, 1.4], dtype=float)}
    x_dot = np.gradient(x, axis=0)

    out = compute_mnps_mnj_sanity(
        x=x,
        x_dot=x_dot,
        time=np.arange(len(x), dtype=float),
        dt_sec=1.0,
        coords_9d=coords_9d,
        coords_9d_names=names,
        jacobian=jacobian,
        jacobian_diagnostics=jac_diag,
        review_qc_cfg={
            "mnps_mnj_sanity": {
                "enabled": True,
                "cond_warn": 100.0,
                "rel_mse_warn": 1.0,
                "robustified_variant": {
                    "enabled": True,
                    "winsorize_quantiles": [0.05, 0.95],
                    "jacobian_ridge_floor": 1e-3,
                },
            }
        },
        projection_contract={
            "mode_requested": "from_v2",
            "mode_effective": "from_v2",
            "x_definition": "derived_mde_from_v2_fixed_weighted_projection",
            "from_v2": {"fallback_reason": None},
        },
        file_labels=None,
        derivative_cfg={"window": 5},
    )

    assert out is not None
    assert out["status"] == "warning"
    assert out["degeneracy_flags"]["coords_9d_has_degenerate_subcoord"] is True
    assert out["degeneracy_flags"]["coords_9d_has_all_nan_subcoord"] is True
    assert out["degeneracy_flags"]["coords_9d_entropy_family_degenerate"] is True
    assert out["degeneracy_flags"]["mnj_high_condition_number"] is True
    assert out["degeneracy_flags"]["mnj_rel_mse_degraded"] is True
    assert out["robustified_comparison"]["enabled"] is True
    assert out["subcoordinate_degeneracy"]["e_e"]["all_nan"] is True
    assert out["subcoordinate_degeneracy"]["e_s"]["degenerate"] is True


def test_write_qc_files_includes_mnps_mnj_sanity(tmp_path):
    """QC writer should persist the MNPS/MNJ sanity block."""
    from mndm.pipeline.summary_qc import write_qc_files

    target_dir = tmp_path / "subject_out"
    target_dir.mkdir(parents=True, exist_ok=True)
    sub_frame = pd.DataFrame({"file": ["sub-001_task-rest_eeg.set"]})

    write_qc_files(
        target_dir=target_dir,
        dataset_label="dsX:sub-001",
        ds_path=tmp_path,
        sub_id="sub-001",
        ses_id="ses-01",
        sub_frame=sub_frame,
        dt=1.0,
        ensemble_summary=None,
        robust_summary=None,
        dist_summary=None,
        baseline_comparisons=None,
        tau_summary=None,
        tier2_jacobian=None,
        tier2_emmi=None,
        null_sanity_tests=None,
        entropy_qc=None,
        geometry_contract={"status": "adjusted", "shared_time_grid": {"epochs_dropped": 1}},
        mnps_mnj_sanity={"status": "warning", "degeneracy_flags": {"mnj_high_condition_number": True}},
    )

    qc_payload = json.loads((target_dir / "qc_summary.json").read_text(encoding="utf-8"))
    assert qc_payload["geometry_contract"]["status"] == "adjusted"
    assert qc_payload["mnps_mnj_sanity"]["status"] == "warning"
    assert qc_payload["mnps_mnj_sanity"]["degeneracy_flags"]["mnj_high_condition_number"] is True


