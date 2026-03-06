"""Tests for EEG feature extraction."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("scipy")


def test_compute_eeg_features_empty():
    """Test that empty signals return empty DataFrame."""
    from mndm.features.eeg import compute_eeg_features
    
    signals = {"signals": {}, "sfreq": 250}
    config = {"epoching": {"length_s": 8.0, "step_s": 4.0}}
    
    out = compute_eeg_features(signals, config)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 0


def test_compute_eeg_features_shape():
    """Test that EEG features have expected shape and columns."""
    from mndm.features.eeg import compute_eeg_features
    
    # Create synthetic EEG data
    n_channels = 10
    n_samples = 250 * 8  # 8 seconds at 250 Hz
    eeg_data = np.random.randn(n_channels, n_samples)
    
    signals = {"signals": {"eeg": eeg_data}, "sfreq": 250}
    config = {
        "epoching": {"length_s": 8.0, "step_s": 4.0},
        "features": {
            "eeg_bands": {
                "delta": [1, 4], "theta": [4, 8], "alpha": [8, 12],
                "beta": [13, 30], "gamma": [30, 45]
            }
        }
    }
    
    out = compute_eeg_features(signals, config)
    assert isinstance(out, pd.DataFrame)
    assert len(out) > 0
    assert "epoch_id" in out.columns
    assert "eeg_delta" in out.columns
    assert "eeg_alpha" in out.columns
    assert "eeg_hjorth_mobility" in out.columns
    assert "eeg_hjorth_complexity" in out.columns
    assert "eeg_highfreq_power_30_45" in out.columns


def test_compute_eeg_features_values():
    """Test that feature values are reasonable."""
    from mndm.features.eeg import compute_eeg_features
    
    # Create synthetic EEG data
    n_channels = 10
    n_samples = 250 * 8
    eeg_data = np.random.randn(n_channels, n_samples)
    
    signals = {"signals": {"eeg": eeg_data}, "sfreq": 250}
    config = {
        "epoching": {"length_s": 8.0, "step_s": 4.0},
        "features": {"eeg_bands": {"alpha": [8, 12]}}
    }
    
    out = compute_eeg_features(signals, config)
    
    # Values should be finite and non-negative for power
    assert np.all(np.isfinite(out["eeg_alpha"]))
    assert np.all(out["eeg_alpha"] >= 0)
    assert np.all(np.isfinite(out["eeg_hjorth_mobility"]))
    assert np.all(np.isfinite(out["eeg_hjorth_complexity"]))
    assert np.all(np.isfinite(out["eeg_highfreq_power_30_45"]))
    assert np.all(out["eeg_highfreq_power_30_45"] >= 0)


def test_compute_eeg_features_permutation_entropy_metadata():
    """Permutation entropy is primary and exposed via stable provenance columns."""
    from mndm.features.eeg import compute_eeg_features

    rng = np.random.default_rng(42)
    eeg_data = rng.normal(size=(8, 250 * 8)).astype(np.float32)
    signals = {"signals": {"eeg": eeg_data}, "sfreq": 250}
    config = {
        "epoching": {"length_s": 8.0, "step_s": 4.0},
        "features": {"eeg_bands": {"alpha": [8, 12]}},
    }
    out = compute_eeg_features(signals, config)
    assert "eeg_sample_entropy" in out.columns
    assert "eeg_permutation_entropy" in out.columns
    assert "eeg_entropy_metric" in out.columns
    assert "eeg_entropy_backend" in out.columns
    assert "eeg_entropy_degraded_mode" in out.columns
    assert "eeg_entropy_reason" in out.columns
    assert set(out["eeg_entropy_metric"].astype(str).unique()) == {"permutation_entropy"}
    assert set(out["eeg_entropy_backend"].astype(str).unique()) == {"numpy"}
    assert not bool(out["eeg_entropy_degraded_mode"].astype(bool).any())
    assert np.allclose(
        out["eeg_sample_entropy"].to_numpy(dtype=float),
        out["eeg_permutation_entropy"].to_numpy(dtype=float),
        equal_nan=True,
    )


