"""Tests for EEG feature extraction."""

from pathlib import Path
import sys
import warnings

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


def test_compute_eeg_features_suppresses_multitaper_nonconvergence_warning(monkeypatch):
    """The known multitaper non-convergence warning is suppressed for EEG/iEEG."""
    from mndm.features import eeg as eeg_mod

    def _fake_psd(data, **kwargs):
        warnings.warn(
            "Iterative multi-taper PSD computation did not converge.",
            RuntimeWarning,
            stacklevel=1,
        )
        arr = np.asarray(data)
        freqs = np.linspace(1.0, 45.0, 8)
        if arr.ndim == 3:
            return np.ones((arr.shape[0], arr.shape[1], freqs.size), dtype=float), freqs
        if arr.ndim == 2:
            return np.ones((arr.shape[0], freqs.size), dtype=float), freqs
        raise AssertionError("Unexpected PSD input rank")

    monkeypatch.setattr(eeg_mod, "psd_array_multitaper", _fake_psd)

    rng = np.random.default_rng(7)
    eeg_data = rng.normal(size=(8, 250 * 8)).astype(np.float32)
    signals = {"signals": {"eeg": eeg_data}, "sfreq": 250}
    config = {
        "epoching": {"length_s": 8.0, "step_s": 4.0},
        "features": {
            "eeg_psd": {"method": "multitaper", "fmin": 1.0, "fmax": 45.0},
            "eeg_bands": {"delta": [1, 4], "theta": [4, 8], "alpha": [8, 12], "beta": [13, 30], "gamma": [30, 45]},
        },
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = eeg_mod.compute_eeg_features(signals, config)

    assert not out.empty
    mt_warnings = [
        rec
        for rec in caught
        if "Iterative multi-taper PSD computation did not converge" in str(rec.message)
    ]
    assert mt_warnings == []


def test_compute_eeg_features_keeps_other_runtime_warnings(monkeypatch):
    """Only the known non-convergence warning is suppressed."""
    from mndm.features import eeg as eeg_mod

    def _fake_psd(data, **kwargs):
        warnings.warn("Synthetic multitaper runtime warning", RuntimeWarning, stacklevel=1)
        arr = np.asarray(data)
        freqs = np.linspace(1.0, 45.0, 8)
        if arr.ndim == 3:
            return np.ones((arr.shape[0], arr.shape[1], freqs.size), dtype=float), freqs
        if arr.ndim == 2:
            return np.ones((arr.shape[0], freqs.size), dtype=float), freqs
        raise AssertionError("Unexpected PSD input rank")

    monkeypatch.setattr(eeg_mod, "psd_array_multitaper", _fake_psd)

    rng = np.random.default_rng(11)
    eeg_data = rng.normal(size=(6, 250 * 8)).astype(np.float32)
    signals = {"signals": {"eeg": eeg_data}, "sfreq": 250}
    config = {
        "epoching": {"length_s": 8.0, "step_s": 4.0},
        "features": {
            "eeg_psd": {"method": "multitaper", "fmin": 1.0, "fmax": 45.0},
            "eeg_bands": {"delta": [1, 4], "theta": [4, 8], "alpha": [8, 12], "beta": [13, 30], "gamma": [30, 45]},
        },
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = eeg_mod.compute_eeg_features(signals, config)

    assert not out.empty
    synthetic = [rec for rec in caught if "Synthetic multitaper runtime warning" in str(rec.message)]
    assert synthetic, "Expected non-target RuntimeWarning to remain visible"


