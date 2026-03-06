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


