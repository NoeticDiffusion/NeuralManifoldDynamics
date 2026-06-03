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


def test_compute_eeg_features_conventional_tier1_outputs(monkeypatch):
    """Tier-1 conventional EEG config should add relative/ratio/peak metrics."""
    from mndm.features import eeg as eeg_mod

    freqs = np.arange(1.0, 46.0, dtype=float)
    psd_template = np.ones_like(freqs, dtype=float)
    alpha_mask = (freqs >= 8.0) & (freqs <= 12.0)
    psd_template[alpha_mask] = np.asarray([1.0, 3.0, 9.0, 3.0, 1.0], dtype=float)
    beta_mask = (freqs >= 13.0) & (freqs <= 30.0)
    psd_template[beta_mask] = 0.5
    gamma_mask = freqs >= 30.0
    psd_template[gamma_mask] = 0.25

    def _fake_psd(data, **kwargs):
        arr = np.asarray(data)
        if arr.ndim != 3:
            raise AssertionError("Expected batched EEG PSD input")
        tiled = np.tile(psd_template, (arr.shape[0], arr.shape[1], 1))
        return tiled.astype(float), freqs

    monkeypatch.setattr(eeg_mod, "psd_array_multitaper", _fake_psd)

    rng = np.random.default_rng(21)
    eeg_data = rng.normal(size=(6, 250 * 8)).astype(np.float32)
    signals = {"signals": {"eeg": eeg_data}, "sfreq": 250}
    config = {
        "epoching": {"length_s": 8.0, "step_s": 4.0},
        "features": {
            "eeg_psd": {"method": "multitaper", "fmin": 1.0, "fmax": 45.0},
            "eeg_bands": {
                "delta": [1, 4],
                "theta": [4, 8],
                "alpha": [8, 12],
                "beta": [13, 30],
                "gamma": [30, 45],
            },
        },
        "conventional_eeg": {
            "enabled": True,
            "packs": ["tier1"],
            "export": {"per_epoch_columns": True, "summaries": True},
            "tier1": {
                "relative_bandpower": True,
                "ratios": ["theta_alpha", "delta_alpha", "alpha_theta", "beta_alpha", "slowing_index"],
                "peak_frequency": {
                    "alpha_peak_frequency": True,
                    "median_frequency": True,
                    "spectral_edge_95": True,
                },
            },
        },
    }

    out = eeg_mod.compute_eeg_features(signals, config)
    assert not out.empty
    row = out.iloc[0]

    relative_cols = [
        "eeg_conventional_relative_delta",
        "eeg_conventional_relative_theta",
        "eeg_conventional_relative_alpha",
        "eeg_conventional_relative_beta",
        "eeg_conventional_relative_gamma",
    ]
    assert all(col in out.columns for col in relative_cols)
    assert row[relative_cols].sum() == pytest.approx(1.0, abs=1e-6)
    assert row["eeg_conventional_ratio_theta_alpha"] == pytest.approx(
        row["eeg_theta"] / row["eeg_alpha"]
    )
    assert row["eeg_conventional_ratio_delta_alpha"] == pytest.approx(
        row["eeg_delta"] / row["eeg_alpha"]
    )
    assert row["eeg_conventional_ratio_alpha_theta"] == pytest.approx(
        row["eeg_alpha"] / row["eeg_theta"]
    )
    assert row["eeg_conventional_ratio_beta_alpha"] == pytest.approx(
        row["eeg_beta"] / row["eeg_alpha"]
    )
    assert row["eeg_conventional_ratio_slowing_index"] == pytest.approx(
        (row["eeg_delta"] + row["eeg_theta"]) / (row["eeg_alpha"] + row["eeg_beta"])
    )
    assert row["eeg_conventional_peak_alpha_frequency"] == pytest.approx(10.0)
    assert np.isfinite(row["eeg_conventional_peak_median_frequency"])
    assert np.isfinite(row["eeg_conventional_peak_spectral_edge_95"])
    assert row["eeg_conventional_peak_median_frequency"] < row["eeg_conventional_peak_spectral_edge_95"]


def test_compute_eeg_features_conventional_complexity_outputs():
    """Complexity pack should emit conventional complexity columns per epoch."""
    from mndm.features.eeg import compute_eeg_features

    rng = np.random.default_rng(123)
    eeg_data = rng.normal(size=(8, 250 * 8)).astype(np.float32)
    signals = {"signals": {"eeg": eeg_data}, "sfreq": 250}
    config = {
        "epoching": {"length_s": 8.0, "step_s": 4.0},
        "features": {
            "eeg_psd": {"method": "welch", "fmin": 1.0, "fmax": 45.0},
            "permutation_entropy": {"order": 5, "delay": 1, "normalize": True},
        },
        "conventional_eeg": {
            "enabled": True,
            "packs": ["complexity"],
            "export": {"per_epoch_columns": True, "summaries": True},
            "complexity": {
                "spectral_entropy": True,
                "permutation_entropy": True,
                "hjorth_complexity": True,
                "hjorth_mobility": True,
            },
        },
    }

    out = compute_eeg_features(signals, config)
    assert not out.empty
    expected_cols = [
        "eeg_conventional_complexity_spectral_entropy",
        "eeg_conventional_complexity_permutation_entropy",
        "eeg_conventional_complexity_hjorth_complexity",
        "eeg_conventional_complexity_hjorth_mobility",
    ]
    assert all(col in out.columns for col in expected_cols)
    assert np.all(np.isfinite(out["eeg_conventional_complexity_spectral_entropy"]))
    assert np.allclose(
        out["eeg_conventional_complexity_permutation_entropy"].to_numpy(dtype=float),
        out["eeg_permutation_entropy"].to_numpy(dtype=float),
        equal_nan=True,
    )
    assert np.allclose(
        out["eeg_conventional_complexity_hjorth_complexity"].to_numpy(dtype=float),
        out["eeg_hjorth_complexity"].to_numpy(dtype=float),
        equal_nan=True,
    )
    assert np.allclose(
        out["eeg_conventional_complexity_hjorth_mobility"].to_numpy(dtype=float),
        out["eeg_hjorth_mobility"].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_compute_eeg_features_conventional_connectivity_outputs(monkeypatch):
    """Connectivity pack should emit conventional synchrony columns."""
    from mndm.features import eeg as eeg_mod

    def _fake_sync_features(eeg_data, sfreq, channel_names, config):
        assert list(channel_names) == ["F3", "P3", "Fz", "POz", "O1", "O2"]
        assert float(sfreq) == 250.0
        return {
            "eeg_sync_alpha_FP_plv_mean": 0.61,
            "eeg_sync_alpha_FB_coh_mean": 0.42,
        }

    monkeypatch.setattr(eeg_mod.eeg_sync, "compute_eeg_synchrony_features", _fake_sync_features)

    rng = np.random.default_rng(456)
    eeg_data = rng.normal(size=(6, 250 * 8)).astype(np.float32)
    signals = {
        "signals": {"eeg": eeg_data},
        "sfreq": 250,
        "channels": {"eeg": ["F3", "P3", "Fz", "POz", "O1", "O2"]},
    }
    config = {
        "epoching": {"length_s": 8.0, "step_s": 4.0},
        "features": {"eeg_psd": {"method": "welch", "fmin": 1.0, "fmax": 45.0}},
        "conventional_eeg": {
            "enabled": True,
            "packs": ["connectivity"],
            "export": {"per_epoch_columns": True, "summaries": True},
            "connectivity": {
                "roi_pairs": [
                    {"name": "FP", "channels": ["F3", "P3"]},
                    {"name": "FB", "channels": ["Fz", "POz"]},
                ],
                "metrics": {"plv": True, "coherence": True},
                "outputs": {"summary_stats": ["mean"]},
            },
        },
    }

    out = eeg_mod.compute_eeg_features(signals, config)
    assert not out.empty
    assert "eeg_conventional_connectivity_alpha_FP_plv_mean" in out.columns
    assert "eeg_conventional_connectivity_alpha_FB_coh_mean" in out.columns
    assert np.allclose(
        out["eeg_conventional_connectivity_alpha_FP_plv_mean"].to_numpy(dtype=float),
        0.61,
    )
    assert np.allclose(
        out["eeg_conventional_connectivity_alpha_FB_coh_mean"].to_numpy(dtype=float),
        0.42,
    )


