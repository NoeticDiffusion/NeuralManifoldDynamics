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


def test_batched_welch_matches_serial_epoch_and_montage_psd():
    """Batched Welch must preserve serial PSD values and frequency bins."""
    from mndm.features import eeg as eeg_mod

    rng = np.random.default_rng(20260820)
    for shape in ((3, 2, 2000), (4, 1, 17)):
        data = rng.normal(size=shape)
        nperseg = min(shape[-1], 512)
        noverlap = nperseg // 2
        batched_psd, batched_freqs = eeg_mod._run_welch_psd_batched(
            data,
            sfreq=250.0,
            fmin=1.0,
            fmax=45.0,
            nperseg=nperseg,
            noverlap=noverlap,
        )

        serial_rows = []
        serial_freqs = None
        for epoch in data:
            serial_signals = []
            for signal_1d in epoch:
                freqs, psd = eeg_mod.signal.welch(
                    signal_1d,
                    fs=250.0,
                    window="hann",
                    nperseg=nperseg,
                    noverlap=noverlap,
                    detrend="constant",
                    scaling="density",
                )
                in_band = (freqs >= 1.0) & (freqs <= 45.0)
                serial_freqs = freqs[in_band]
                serial_signals.append(psd[in_band])
            serial_rows.append(np.stack(serial_signals, axis=0))
        serial_psd = np.stack(serial_rows, axis=0)

        assert np.array_equal(batched_freqs, serial_freqs)
        assert np.array_equal(batched_psd, serial_psd)


def test_batched_welch_preserves_nonfinite_rows_and_empty_band():
    """Welch batching keeps non-finite isolation and empty frequency masks."""
    from mndm.features import eeg as eeg_mod

    rng = np.random.default_rng(20260822)
    data = rng.normal(size=(2, 2, 512))
    data[0, 0, 0] = np.nan
    data[1, 1, 0] = np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        batched_psd, batched_freqs = eeg_mod._run_welch_psd_batched(
            data,
            sfreq=250.0,
            fmin=1.0,
            fmax=45.0,
            nperseg=512,
            noverlap=256,
        )
        serial_rows = []
        serial_freqs = None
        for epoch in data:
            serial_signals = []
            for signal_1d in epoch:
                freqs, psd = eeg_mod.signal.welch(
                    signal_1d,
                    fs=250.0,
                    window="hann",
                    nperseg=512,
                    noverlap=256,
                    detrend="constant",
                    scaling="density",
                )
                in_band = (freqs >= 1.0) & (freqs <= 45.0)
                serial_freqs = freqs[in_band]
                serial_signals.append(psd[in_band])
            serial_rows.append(np.stack(serial_signals, axis=0))
        serial_psd = np.stack(serial_rows, axis=0)

        empty_psd, empty_freqs = eeg_mod._run_welch_psd_batched(
            data,
            sfreq=250.0,
            fmin=1000.0,
            fmax=1001.0,
            nperseg=512,
            noverlap=256,
        )

    assert np.array_equal(batched_freqs, serial_freqs)
    assert np.array_equal(batched_psd, serial_psd, equal_nan=True)
    assert np.isnan(batched_psd[0, 0]).all()
    assert np.isnan(batched_psd[1, 1]).all()
    assert np.isfinite(batched_psd[0, 1]).all()
    assert np.isfinite(batched_psd[1, 0]).all()
    assert empty_psd.shape == (2, 2, 0)
    assert empty_freqs.shape == (0,)


def test_compute_eeg_features_calls_welch_once_for_all_epochs(monkeypatch):
    """The primary Welch fallback should receive the complete PSD batch."""
    from mndm.features import eeg as eeg_mod

    calls = []
    original_welch = eeg_mod.signal.welch

    def _counting_welch(data, *args, **kwargs):
        calls.append(np.asarray(data).shape)
        return original_welch(data, *args, **kwargs)

    monkeypatch.setattr(eeg_mod.signal, "welch", _counting_welch)
    rng = np.random.default_rng(20260821)
    signals = {
        "signals": {"eeg": rng.normal(size=(6, 250 * 16)).astype(np.float32)},
        "sfreq": 250,
    }
    config = {
        "epoching": {"length_s": 8.0, "step_s": 4.0},
        "features": {
            "eeg_psd": {"method": "welch", "fmin": 1.0, "fmax": 45.0},
        },
    }

    out = eeg_mod.compute_eeg_features(signals, config)

    assert not out.empty
    assert calls == [(len(out), 1, 2000)]


def test_compute_eeg_features_batches_secondary_welch(monkeypatch):
    """Secondary multiverse Welch should use one global epoch batch."""
    from mndm.features import eeg as eeg_mod

    calls = []
    original_welch = eeg_mod.signal.welch

    def _counting_welch(data, *args, **kwargs):
        calls.append(np.asarray(data).shape)
        return original_welch(data, *args, **kwargs)

    monkeypatch.setattr(eeg_mod.signal, "welch", _counting_welch)
    rng = np.random.default_rng(20260823)
    signals = {
        "signals": {"eeg": rng.normal(size=(6, 250 * 16)).astype(np.float32)},
        "sfreq": 250,
    }
    config = {
        "epoching": {"length_s": 8.0, "step_s": 4.0},
        "features": {
            "eeg_psd": {"method": "welch", "fmin": 1.0, "fmax": 45.0},
            "eeg_bands": {"alpha": [8.0, 12.0]},
        },
        "robustness": {
            "multiverse": {
                "psd": {"enabled": True, "secondary_method": "welch"},
            },
        },
    }

    out = eeg_mod.compute_eeg_features(signals, config)

    assert not out.empty
    assert calls == [(len(out), 1, 2000), (len(out), 2000)]
    assert "eeg_alpha__psd_alt" in out.columns
    assert np.all(np.isfinite(out["eeg_alpha__psd_alt"]))


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


def test_compute_eeg_features_reuses_global_complexity_metrics(monkeypatch):
    """The conventional complexity pack should not recompute global metrics."""
    from mndm.features import eeg as eeg_mod

    calls = {"hjorth": 0, "permutation_entropy": 0}
    original_hjorth = eeg_mod._compute_hjorth_metrics
    original_permutation_entropy = eeg_mod._compute_permutation_entropy

    def _counting_hjorth(data):
        calls["hjorth"] += 1
        return original_hjorth(data)

    def _counting_permutation_entropy(*args, **kwargs):
        calls["permutation_entropy"] += 1
        return original_permutation_entropy(*args, **kwargs)

    monkeypatch.setattr(eeg_mod, "_compute_hjorth_metrics", _counting_hjorth)
    monkeypatch.setattr(eeg_mod, "_compute_permutation_entropy", _counting_permutation_entropy)

    rng = np.random.default_rng(124)
    eeg_data = rng.normal(size=(8, 250 * 16)).astype(np.float32)
    signals = {"signals": {"eeg": eeg_data}, "sfreq": 250}
    config = {
        "epoching": {"length_s": 8.0, "step_s": 4.0},
        "features": {
            "eeg_psd": {"method": "welch", "fmin": 1.0, "fmax": 45.0},
            "permutation_entropy": {"order": 5, "delay": 1, "normalize": True},
        },
        "robustness": {"ensembles": {"enabled": False}},
        "conventional_eeg": {
            "enabled": True,
            "packs": ["complexity"],
            "export": {"per_epoch_columns": True, "summaries": True},
            "complexity": {
                "permutation_entropy": True,
                "hjorth_complexity": True,
                "hjorth_mobility": True,
            },
        },
    }

    out = eeg_mod.compute_eeg_features(signals, config)

    assert len(out) == 3
    assert calls == {"hjorth": len(out), "permutation_entropy": len(out)}


def test_conventional_complexity_does_not_copy_degraded_entropy(monkeypatch):
    """Degraded primary entropy must not become the conventional PE column."""
    from mndm.features import eeg as eeg_mod

    direct_value = 0.125
    monkeypatch.setattr(
        eeg_mod,
        "_compute_permutation_entropy",
        lambda *_args, **_kwargs: direct_value,
    )
    conventional_cfg = {
        "enabled": True,
        "packs": ["complexity"],
        "export": {"per_epoch_columns": True, "summaries": True},
        "complexity": {
            "permutation_entropy": True,
            "hjorth_complexity": True,
            "hjorth_mobility": True,
        },
    }

    normal = eeg_mod._compute_conventional_complexity_features(
        epoch_signal=np.arange(32, dtype=float),
        sfreq=250.0,
        order=5,
        delay=1,
        normalize=True,
        conventional_cfg=conventional_cfg,
        precomputed_hjorth=(0.25, 0.5),
        precomputed_permutation_entropy=0.875,
        permutation_entropy_degraded=False,
    )
    degraded = eeg_mod._compute_conventional_complexity_features(
        epoch_signal=np.arange(32, dtype=float),
        sfreq=250.0,
        order=5,
        delay=1,
        normalize=True,
        conventional_cfg=conventional_cfg,
        precomputed_hjorth=(0.25, 0.5),
        precomputed_permutation_entropy=0.875,
        permutation_entropy_degraded=True,
    )

    assert normal["eeg_conventional_complexity_permutation_entropy"] == 0.875
    assert degraded["eeg_conventional_complexity_permutation_entropy"] == direct_value
    assert degraded["eeg_conventional_complexity_hjorth_mobility"] == 0.25
    assert degraded["eeg_conventional_complexity_hjorth_complexity"] == 0.5


def test_compute_eeg_features_preserves_degraded_entropy_semantics(monkeypatch):
    """End-to-end degraded entropy keeps conventional PE distinct from fallback."""
    from mndm.features import eeg as eeg_mod

    monkeypatch.setattr(
        eeg_mod,
        "_compute_permutation_entropy",
        lambda *_args, **_kwargs: np.nan,
    )
    rng = np.random.default_rng(125)
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
            "complexity": {"permutation_entropy": True},
        },
    }

    out = eeg_mod.compute_eeg_features(signals, config)

    assert set(out["eeg_entropy_metric"].astype(str).unique()) == {"spectral_entropy"}
    assert out["eeg_entropy_degraded_mode"].astype(bool).all()
    assert np.all(np.isfinite(out["eeg_permutation_entropy"].to_numpy(dtype=float)))
    assert np.all(np.isnan(out["eeg_conventional_complexity_permutation_entropy"].to_numpy(dtype=float)))


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


def test_synchrony_connectivity_uses_clean_segments_around_bad_mask():
    """A BAD_ NaN gap must not poison recording-level synchrony summaries."""
    from mndm.features.eeg_sync import compute_eeg_synchrony_features

    sfreq = 128.0
    data = np.random.default_rng(17).normal(size=(2, int(20 * sfreq)))
    data[:, int(8 * sfreq) : int(9 * sfreq)] = np.nan
    config = {
        "bands": [{"name": "infant_theta", "f_low": 3.0, "f_high": 6.0}],
        "windows": {"length_sec": 2.0, "step_sec": 1.0},
        "roi_pairs": [{"name": "F4_P4", "channels": ["F4", "P4"]}],
        "metrics": {"coherence": True},
        "outputs": {"summary_stats": ["mean", "std"]},
    }

    result = compute_eeg_synchrony_features(data, sfreq, ["F4", "P4"], config)

    assert np.isfinite(result["eeg_sync_infant_theta_F4_P4_coh_mean"])
    assert np.isfinite(result["eeg_sync_infant_theta_F4_P4_coh_std"])


def test_synchrony_reuses_filter_and_hilbert_per_band_segment(monkeypatch):
    """Filtering and Hilbert transforms are shared across ROI pairs."""
    from mndm.features import eeg_sync

    sfreq = 128.0
    rng = np.random.default_rng(18)
    data = rng.normal(size=(4, int(12 * sfreq)))
    data[:, int(4 * sfreq) : int(5 * sfreq)] = np.nan
    config = {
        "bands": [
            {"name": "low", "f_low": 3.0, "f_high": 6.0},
            {"name": "high", "f_low": 8.0, "f_high": 12.0},
        ],
        "windows": {"length_sec": 2.0, "step_sec": 1.0},
        "roi_pairs": [
            {"name": "pair_01", "channels": ["C0", "C1"]},
            {"name": "pair_23", "channels": ["C2", "C3"]},
        ],
        "metrics": {"plv": True},
        "outputs": {"summary_stats": ["mean"]},
    }
    bandpass_calls = []
    hilbert_calls = []
    original_bandpass = eeg_sync._bandpass
    original_hilbert = eeg_sync.hilbert

    def _counting_bandpass(*args, **kwargs):
        bandpass_calls.append(True)
        return original_bandpass(*args, **kwargs)

    def _counting_hilbert(*args, **kwargs):
        hilbert_calls.append(True)
        return original_hilbert(*args, **kwargs)

    monkeypatch.setattr(eeg_sync, "_bandpass", _counting_bandpass)
    monkeypatch.setattr(eeg_sync, "hilbert", _counting_hilbert)

    result = eeg_sync.compute_eeg_synchrony_features(
        data,
        sfreq,
        ["C0", "C1", "C2", "C3"],
        config,
    )

    # Two bands × two clean segments; neither operation depends on the pair.
    assert len(bandpass_calls) == 4
    assert len(hilbert_calls) == 4
    assert set(result) == {
        "eeg_sync_low_pair_01_plv_mean",
        "eeg_sync_low_pair_23_plv_mean",
        "eeg_sync_high_pair_01_plv_mean",
        "eeg_sync_high_pair_23_plv_mean",
    }
    assert all(np.isfinite(value) for value in result.values())


def test_compute_eeg_features_conventional_coma_outputs(monkeypatch):
    """Coma pack should emit suppression/continuity and reactivity proxy columns."""
    from mndm.features import eeg as eeg_mod

    freqs = np.arange(1.0, 46.0, dtype=float)
    psd_template = np.ones_like(freqs, dtype=float)
    psd_template[(freqs >= 1.0) & (freqs <= 4.0)] = 2.0
    psd_template[(freqs >= 8.0) & (freqs <= 12.0)] = 1.0

    def _fake_psd(data, **kwargs):
        arr = np.asarray(data)
        if arr.ndim != 3:
            raise AssertionError("Expected batched EEG PSD input")
        tiled = np.tile(psd_template, (arr.shape[0], arr.shape[1], 1))
        return tiled.astype(float), freqs

    monkeypatch.setattr(eeg_mod, "psd_array_multitaper", _fake_psd)

    rng = np.random.default_rng(2026)
    sfreq = 250
    n_channels = 6
    n_samples = sfreq * 20
    t = np.arange(n_samples, dtype=float) / float(sfreq)
    amp = np.where((np.floor(t / 2.0) % 2.0) == 0.0, 0.02, 1.0)
    eeg_data = (rng.normal(size=(n_channels, n_samples)) * amp[np.newaxis, :]).astype(np.float32)
    signals = {"signals": {"eeg": eeg_data}, "sfreq": sfreq}
    config = {
        "epoching": {"length_s": 4.0, "step_s": 2.0},
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
            "packs": ["coma"],
            "export": {"per_epoch_columns": True, "summaries": True},
            "coma": {
                "suppression_ratio": True,
                "burst_suppression_proxy": True,
                "continuity_proxy": True,
                "alpha_delta_ratio": True,
                "reactivity_proxy": {"enabled": True, "clip_at": 5.0},
            },
        },
    }

    out = eeg_mod.compute_eeg_features(signals, config)
    assert not out.empty
    expected_cols = [
        "eeg_conventional_coma_suppression_ratio",
        "eeg_conventional_coma_continuity_proxy",
        "eeg_conventional_coma_burst_suppression_proxy",
        "eeg_conventional_coma_alpha_delta_ratio",
        "eeg_conventional_coma_reactivity_proxy",
    ]
    assert all(col in out.columns for col in expected_cols)

    suppression = out["eeg_conventional_coma_suppression_ratio"].to_numpy(dtype=float)
    continuity = out["eeg_conventional_coma_continuity_proxy"].to_numpy(dtype=float)
    burst_proxy = out["eeg_conventional_coma_burst_suppression_proxy"].to_numpy(dtype=float)
    reactivity = out["eeg_conventional_coma_reactivity_proxy"].to_numpy(dtype=float)
    ratio = out["eeg_conventional_coma_alpha_delta_ratio"].to_numpy(dtype=float)

    assert np.all(np.isfinite(suppression))
    assert np.all(np.isfinite(continuity))
    assert np.all(np.isfinite(burst_proxy))
    assert np.all(np.isfinite(reactivity))
    assert np.all(np.isfinite(ratio))
    assert np.all((suppression >= 0.0) & (suppression <= 1.0))
    assert np.all((continuity >= 0.0) & (continuity <= 1.0))
    assert np.all((burst_proxy >= 0.0) & (burst_proxy <= 1.0))
    assert np.all((reactivity >= 0.0) & (reactivity <= 1.0))
    assert np.allclose(continuity, 1.0 - suppression, atol=1e-6, rtol=0.0)
    assert np.allclose(
        ratio,
        out["eeg_alpha"].to_numpy(dtype=float) / out["eeg_delta"].to_numpy(dtype=float),
        equal_nan=True,
    )
    assert float(np.nanmax(reactivity)) > 0.0


def test_compute_eeg_features_conventional_coma_reactivity_can_be_disabled(monkeypatch):
    """Disabling coma reactivity proxy should remove the reactivity column."""
    from mndm.features import eeg as eeg_mod

    freqs = np.arange(1.0, 46.0, dtype=float)

    def _fake_psd(data, **kwargs):
        arr = np.asarray(data)
        if arr.ndim != 3:
            raise AssertionError("Expected batched EEG PSD input")
        tiled = np.ones((arr.shape[0], arr.shape[1], freqs.size), dtype=float)
        return tiled, freqs

    monkeypatch.setattr(eeg_mod, "psd_array_multitaper", _fake_psd)

    rng = np.random.default_rng(77)
    eeg_data = rng.normal(size=(4, 250 * 12)).astype(np.float32)
    signals = {"signals": {"eeg": eeg_data}, "sfreq": 250}
    config = {
        "epoching": {"length_s": 4.0, "step_s": 2.0},
        "features": {"eeg_psd": {"method": "multitaper", "fmin": 1.0, "fmax": 45.0}},
        "conventional_eeg": {
            "enabled": True,
            "packs": ["coma"],
            "coma": {
                "suppression_ratio": True,
                "reactivity_proxy": {"enabled": False},
            },
        },
    }

    out = eeg_mod.compute_eeg_features(signals, config)
    assert "eeg_conventional_coma_suppression_ratio" in out.columns
    assert "eeg_conventional_coma_reactivity_proxy" not in out.columns

