"""Tests for fMRI feature extraction."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("scipy")


def test_compute_fmri_features_empty():
    """Empty signals dict should yield empty DataFrame."""
    from mndm.features.fmri import compute_fmri_features

    signals = {"signals": {}, "sfreq": 1.0}
    config: dict = {}
    out = compute_fmri_features(signals, config)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 0


def test_compute_fmri_features_basic_shape():
    """Synthetic fMRI data should produce per-window features with expected columns."""
    from mndm.features.fmri import compute_fmri_features

    n_regions = 5
    n_times = 100
    sfreq = 1.0  # 1 Hz (TR = 1s) for simplicity
    rng = np.random.default_rng(0)
    fmri_data = rng.standard_normal((n_regions, n_times)).astype(float)

    signals = {"signals": {"fmri": fmri_data}, "sfreq": sfreq}
    config = {
        "features": {"fmri": {"window_sec": 10.0, "step_sec": 5.0}},
        "epoching": {"length_s": 10.0, "step_s": 5.0},
    }

    out = compute_fmri_features(signals, config)
    assert isinstance(out, pd.DataFrame)
    assert len(out) > 0
    for col in (
        "epoch_id",
        "t_start",
        "t_end",
        "fmri_entropy_global",
        "fmri_lf_power",
        "fmri_variance_global",
        "fmri_FC_mean",
        "fmri_kuramoto_global",
        "fmri_modularity",
        "fmri_dFC_variance",
        "fmri_slow4_slow5_ratio",
        "fmri_ar1_coefficient",
        "fmri_gradient_ratio",
        "fmri_lf_power_delta_valid",
    ):
        assert col in out.columns

    # Basic sanity checks on values
    assert np.all(np.isfinite(out["fmri_lf_power"]))
    assert np.all(np.isfinite(out["fmri_variance_global"]))
    assert int(out["fmri_lf_power_delta_valid"].iloc[0]) == 0
    assert bool(np.isnan(out["fmri_lf_power_delta"].iloc[0]))


def test_compute_fmri_features_v25_stage2_opt_in_metrics():
    """NMD-fMRI-v2.5 Stage 2 candidate features (entropy/connectivity/
    temporal-persistence/ALFF) must be opt-in only and, when enabled, must
    produce finite values distinct from the legacy alias columns."""
    from mndm.features.fmri import compute_fmri_features

    n_regions = 12
    n_times = 400
    sfreq = 0.5  # TR = 2s, like ds007216
    rng = np.random.default_rng(42)
    fmri_data = rng.standard_normal((n_regions, n_times)).astype(float)
    roi_names = [f"7Networks_LH_{net}_{i}" for i, net in enumerate(
        ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"] * 2
    )][:n_regions]

    signals = {
        "signals": {"fmri": fmri_data},
        "sfreq": sfreq,
        "channels": {"fmri": roi_names},
    }
    config = {
        "features": {"fmri": {"window_sec": 45.0, "step_sec": 20.0}},
        "epoching": {"length_s": 45.0, "step_s": 20.0},
        "metrics": {
            "compute_spectral_entropy": True,
            "compute_permutation_entropy_fmri": True,
            "compute_sample_entropy_fmri": True,
            "compute_network_fc": True,
            "compute_ar2": True,
            "compute_temporal_smoothness": True,
            "compute_hurst": True,
            "compute_alff": True,
            "compute_falff": True,
            "compute_dynamic_deltas": True,
        },
    }

    out = compute_fmri_features(signals, config)
    assert isinstance(out, pd.DataFrame)
    assert len(out) >= 2  # need >=2 epochs to exercise delta features

    new_cols = (
        "fmri_spectral_entropy",
        "fmri_permutation_entropy",
        "fmri_sample_entropy",
        "fmri_within_network_fc",
        "fmri_between_network_fc",
        "fmri_network_segregation_index",
        "fmri_participation_coefficient",
        "fmri_ar2_coefficient",
        "fmri_temporal_smoothness",
        "fmri_hurst_exponent",
        "fmri_ALFF",
        "fmri_fALFF",
        "fmri_modularity_delta",
        "fmri_FC_mean_change_rate",
    )
    for col in new_cols:
        assert col in out.columns, f"missing expected v2.5 Stage 2 column {col}"
        assert np.any(np.isfinite(out[col])), f"column {col} is entirely non-finite"

    # Contract repair: these must remain literal aliases (unchanged behavior).
    assert np.allclose(out["fmri_entropy_global"], out["fmri_variance_global"], equal_nan=True)
    assert np.allclose(out["fmri_region_var_mean"], out["fmri_variance_global"], equal_nan=True)
    assert np.allclose(out["fmri_lf_power"], out["fmri_signal_power"], equal_nan=True)

    # Spectral entropy must NOT equal the alias columns (real, independent metric).
    assert not np.allclose(out["fmri_spectral_entropy"], out["fmri_variance_global"], equal_nan=True)

    # First epoch's dynamic deltas must be NaN (no previous epoch yet).
    assert bool(np.isnan(out["fmri_modularity_delta"].iloc[0]))
    assert bool(np.isnan(out["fmri_FC_mean_change_rate"].iloc[0]))


def test_compute_fmri_features_v25_stage2_metrics_default_off():
    """When not explicitly enabled, none of the new Stage 2 columns should
    appear -- existing dataset configs must see zero schema/behavior change."""
    from mndm.features.fmri import compute_fmri_features

    n_regions = 6
    n_times = 120
    sfreq = 1.0
    rng = np.random.default_rng(7)
    fmri_data = rng.standard_normal((n_regions, n_times)).astype(float)

    signals = {"signals": {"fmri": fmri_data}, "sfreq": sfreq}
    config = {
        "features": {"fmri": {"window_sec": 10.0, "step_sec": 5.0}},
        "epoching": {"length_s": 10.0, "step_s": 5.0},
    }
    out = compute_fmri_features(signals, config)
    for col in (
        "fmri_spectral_entropy",
        "fmri_permutation_entropy",
        "fmri_sample_entropy",
        "fmri_within_network_fc",
        "fmri_ar2_coefficient",
        "fmri_temporal_smoothness",
        "fmri_hurst_exponent",
        "fmri_ALFF",
        "fmri_fALFF",
        "fmri_modularity_delta",
        "fmri_FC_mean_change_rate",
    ):
        assert col not in out.columns


def test_compute_fmri_features_invalid_bandpass_raises():
    """Invalid bandpass for the given sfreq should hard-fail."""
    from mndm.features.fmri import compute_fmri_features

    n_regions = 4
    n_times = 64
    sfreq = 1.0  # nyquist = 0.5 Hz
    rng = np.random.default_rng(1)
    fmri_data = rng.standard_normal((n_regions, n_times)).astype(float)

    signals = {"signals": {"fmri": fmri_data}, "sfreq": sfreq}
    config = {
        "features": {"fmri": {"window_sec": 10.0, "step_sec": 5.0}},
        "epoching": {"length_s": 10.0, "step_s": 5.0},
        "preprocessing": {"bandpass": [0.01, 0.6]},  # invalid: f_high > nyquist
    }

    with pytest.raises(ValueError, match="Invalid bandpass parameters"):
        _ = compute_fmri_features(signals, config)


def test_dvars_scrubbed_epoch_projects_to_nan_not_zero():
    """P0.1: DVARS-scrubbed feature rows remain missing after robust_z projection."""
    from mndm.features.fmri import compute_fmri_features
    from mndm.projection import project_features_with_coverage

    n_regions = 8
    n_times = 30
    sfreq = 1.0
    fmri_data = np.ones((n_regions, n_times), dtype=float)
    fmri_data[:, 10:20] = np.linspace(0.0, 400.0, 10, dtype=float)

    out = compute_fmri_features(
        {"signals": {"fmri": fmri_data}, "sfreq": sfreq},
        {
            "features": {"fmri": {"window_sec": 10.0, "step_sec": 10.0}},
            "epoching": {"length_s": 10.0, "step_s": 10.0},
            "metrics": {"dvars_threshold": 5.0},
            "preprocessing": {"bandpass": [0.01, 0.1]},
        },
    )
    assert len(out) >= 2
    assert float(out.loc[1, "fmri_dvars"]) > 5.0
    assert bool(np.isnan(out.loc[1, "fmri_variance_global"]))
    assert np.isfinite(out.loc[0, "fmri_variance_global"])

    x, coverage, _ = project_features_with_coverage(
        out,
        {
            "m": {"fmri_variance_global": 1.0},
            "d": {"fmri_variance_global": 1.0},
            "e": {"fmri_variance_global": 1.0},
        },
        normalize="robust_z",
        feature_standardization={"fmri_variance_global": ["robust_z", "clip"]},
    )
    assert np.isfinite(x[0]).all()
    assert np.all(np.isnan(x[1]))
    assert np.allclose(coverage[0], 1.0)
    assert np.allclose(coverage[1], 0.0)


def test_compute_fmri_features_copies_resolved_tr_from_meta():
    """P0.3: feature rows carry tr_sec / tr_source from preprocess meta."""
    from mndm.features.fmri import compute_fmri_features

    n_regions = 4
    n_times = 40
    rng = np.random.default_rng(1)
    signals = {
        "signals": {"fmri": rng.standard_normal((n_regions, n_times))},
        "sfreq": 0.5,
        "meta": {"tr_sec": 2.0, "sfreq": 0.5, "tr_source": "bids_json"},
    }
    out = compute_fmri_features(
        signals,
        {"features": {"fmri": {"window_sec": 10.0, "step_sec": 10.0}}},
    )
    assert len(out) > 0
    assert np.allclose(out["fmri_tr_sec"], 2.0)
    assert np.allclose(out["fmri_sfreq"], 0.5)
    assert set(out["fmri_tr_source"].astype(str)) == {"bids_json"}


def test_compute_fmri_features_copies_nuisance_status_from_meta():
    """P0.5: feature rows carry nuisance_status from preprocess meta."""
    from mndm.features.fmri import compute_fmri_features

    n_regions = 4
    n_times = 40
    rng = np.random.default_rng(1)
    signals = {
        "signals": {"fmri": rng.standard_normal((n_regions, n_times))},
        "sfreq": 0.5,
        "meta": {"tr_sec": 2.0, "sfreq": 0.5, "tr_source": "bids_json", "nuisance_status": "applied"},
    }
    out = compute_fmri_features(
        signals,
        {"features": {"fmri": {"window_sec": 10.0, "step_sec": 10.0}}},
    )
    assert len(out) > 0
    assert set(out["fmri_nuisance_status"].astype(str)) == {"applied"}


def test_compute_fmri_features_copies_atlas_space_from_meta():
    """P0.6: feature rows carry atlas vs BOLD space provenance from preprocess meta."""
    from mndm.features.fmri import compute_fmri_features

    n_regions = 4
    n_times = 40
    rng = np.random.default_rng(1)
    signals = {
        "signals": {"fmri": rng.standard_normal((n_regions, n_times))},
        "sfreq": 0.5,
        "meta": {
            "tr_sec": 2.0,
            "sfreq": 0.5,
            "atlas_space_status": "matched",
            "atlas_ornt": "RAS",
            "bold_ornt": "RAS",
            "atlas_affine_match": 1,
        },
    }
    out = compute_fmri_features(
        signals,
        {"features": {"fmri": {"window_sec": 10.0, "step_sec": 10.0}}},
    )
    assert len(out) > 0
    assert set(out["fmri_atlas_space_status"].astype(str)) == {"matched"}
    assert set(out["fmri_atlas_ornt"].astype(str)) == {"RAS"}
    assert np.allclose(out["fmri_atlas_affine_match"], 1)


def test_compute_fmri_features_missing_sfreq_is_not_testable():
    """P0.3: feature extraction must not guess TR=1.0 when sfreq is absent."""
    from mndm.features.fmri import compute_fmri_features

    signals = {"signals": {"fmri": np.ones((3, 20), dtype=float)}}
    with pytest.raises(ValueError, match="NOT_TESTABLE"):
        compute_fmri_features(
            signals,
            {"features": {"fmri": {"window_sec": 10.0, "step_sec": 10.0}}},
        )


def _ar_lag_corr_loop(epoch: np.ndarray, lag: int) -> float:
    """Historical per-ROI corrcoef loop (P2.1 identity reference)."""
    vals: list[float] = []
    for i in range(epoch.shape[0]):
        x = np.asarray(epoch[i], dtype=float)
        if x.size < lag + 2:
            continue
        x0, x1 = x[:-lag], x[lag:]
        if np.nanstd(x0) <= 0 or np.nanstd(x1) <= 0:
            continue
        corr = np.corrcoef(x0, x1)[0, 1]
        if np.isfinite(corr):
            vals.append(float(corr))
    return float(np.mean(vals)) if vals else float("nan")


def test_ar1_ar2_match_per_roi_corrcoef_loop():
    """P2.1: vectorized lag corr equals the historical ROI loop, including NaNs."""
    from mndm.features.fmri_epoch_metrics import _compute_ar1_mean, _compute_ar2_mean

    rng = np.random.default_rng(0)
    epoch = rng.standard_normal((40, 48))
    epoch[3, 5] = np.nan
    epoch[7, :] = 1.5
    epoch[11, :] = np.nan
    ar1 = _compute_ar1_mean(epoch)
    ar2 = _compute_ar2_mean(epoch)
    assert ar1 == pytest.approx(_ar_lag_corr_loop(epoch, 1), rel=0, abs=1e-12)
    assert ar2 == pytest.approx(_ar_lag_corr_loop(epoch, 2), rel=0, abs=1e-12)
    assert not np.isfinite(_compute_ar1_mean(epoch[:2, :2]))


def test_dfc_fallback_reuses_epoch_fc_matrix():
    """P2.2: too few subwindows → nanvar of the provided epoch FC, no second corrcoef."""
    from mndm.features.fmri_epoch_metrics import _compute_dfc_variance, fc_triu_mean

    rng = np.random.default_rng(1)
    epoch = rng.standard_normal((8, 16))
    fc = np.corrcoef(epoch)
    triu = fc[np.triu_indices_from(fc, k=1)]
    expected = float(np.nanvar(triu[np.isfinite(triu)]))
    got = _compute_dfc_variance(epoch, sfreq=1.0, min_timepoints_fc=10, fc_epoch=fc)
    assert got == pytest.approx(expected, rel=0, abs=1e-12)
    assert got != pytest.approx(fc_triu_mean(fc))


def _participation_loop(fc: np.ndarray, roi_names: list[str]) -> float:
    from mndm.features.fmri_epoch_metrics import _infer_network_label

    groups: dict[str, list[int]] = {}
    for idx, name in enumerate(roi_names):
        groups.setdefault(_infer_network_label(str(name)), []).append(idx)
    idx_to_group = {idx: g for g, idxs in groups.items() for idx in idxs}
    absfc = np.abs(np.nan_to_num(fc, nan=0.0))
    np.fill_diagonal(absfc, 0.0)
    n = fc.shape[0]
    pc_vals: list[float] = []
    for i in range(n):
        total = float(np.sum(absfc[i, :]))
        if total <= 1e-12:
            continue
        group_sums: dict[str, float] = {}
        for j in range(n):
            if j == i:
                continue
            g = idx_to_group.get(j)
            if g is None:
                continue
            group_sums[g] = group_sums.get(g, 0.0) + float(absfc[i, j])
        frac_sq_sum = sum((s / total) ** 2 for s in group_sums.values())
        pc_vals.append(1.0 - frac_sq_sum)
    return float(np.mean(pc_vals)) if pc_vals else float("nan")


def test_participation_coefficient_matches_node_loop():
    """P2.2: group-indicator participation equals the n×n Python loop."""
    from mndm.features.fmri_epoch_metrics import _compute_network_fc_metrics

    rng = np.random.default_rng(2)
    n = 14
    x = rng.standard_normal((n, 40))
    fc = np.corrcoef(x)
    names = [
        f"7Networks_LH_{net}_{i}"
        for i, net in enumerate(["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Default"] * 3)
    ][:n]
    out = _compute_network_fc_metrics(fc, names)
    assert out["fmri_participation_coefficient"] == pytest.approx(
        _participation_loop(fc, names), rel=0, abs=1e-12
    )


def test_participation_includes_singleton_networks():
    """Participation is defined on the full partition; Chan segregation is not."""
    from mndm.features.fmri_epoch_metrics import _compute_network_fc_metrics

    rng = np.random.default_rng(4)
    n = 6
    fc = np.corrcoef(rng.standard_normal((n, 30)))
    names = [
        "7Networks_LH_Vis_1",
        "7Networks_LH_Vis_2",
        "UniqueA",
        "UniqueB",
        "UniqueC",
        "UniqueD",
    ]
    out = _compute_network_fc_metrics(fc, names)
    assert not np.isfinite(out["fmri_within_network_fc"])
    assert not np.isfinite(out["fmri_network_segregation_index"])
    assert out["fmri_participation_coefficient"] == pytest.approx(
        _participation_loop(fc, names), rel=0, abs=1e-12
    )
    assert np.isfinite(out["fmri_participation_coefficient"])


def test_epoch_fd_range_max_matches_boolean_mask():
    """P2.4: searchsorted epoch max equals the historical bool-mask max."""
    from mndm.pipeline.summary import epoch_fd_range_max

    rng = np.random.default_rng(3)
    fd = rng.random(80)
    fd[10] = np.nan
    tr = 2.0
    frame_t = np.arange(fd.size, dtype=float) * tr
    t_start = np.array([0.0, 14.0, 30.0, 90.0])
    t_end = np.array([14.0, 30.0, 50.0, 200.0])
    got = epoch_fd_range_max(fd, frame_t, t_start, t_end)
    expected = np.full(t_start.shape, np.nan)
    for i, (s, e) in enumerate(zip(t_start, t_end)):
        vals = fd[(frame_t >= s) & (frame_t < e)]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            expected[i] = float(np.max(vals))
    np.testing.assert_allclose(got, expected, equal_nan=True, atol=0, rtol=0)


