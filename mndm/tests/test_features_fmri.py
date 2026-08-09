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


