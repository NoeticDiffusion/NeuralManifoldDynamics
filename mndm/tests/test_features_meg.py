"""Tests for MEG shadow feature extraction."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("scipy")


def test_compute_meg_features_exports_diagnostic_and_combined_surfaces():
    """MAG/GRAD diagnostics and combined `meg_*` columns should be emitted together."""
    from mndm.features.meg import compute_meg_features

    sfreq = 100.0
    times = np.arange(0.0, 16.0, 1.0 / sfreq, dtype=np.float32)
    mag_1 = np.sin(2 * np.pi * 10.0 * times)
    mag_2 = 0.8 * np.sin(2 * np.pi * 6.0 * times)
    grad_1 = 0.6 * np.sin(2 * np.pi * 12.0 * times)
    grad_2 = 0.4 * np.sin(2 * np.pi * 20.0 * times)
    signals = {
        "signals": {
            "meg": np.vstack([mag_1, grad_1, mag_2, grad_2]),
            "meg_mag": np.vstack([mag_1, mag_2]),
            "meg_grad": np.vstack([grad_1, grad_2]),
        },
        "channels": {
            "meg": ["MEG0111", "MEG0112", "MEG0121", "MEG0122"],
            "meg_mag": ["MEG0111", "MEG0121"],
            "meg_grad": ["MEG0112", "MEG0122"],
        },
        "sfreq": sfreq,
        "dataset_id": "ds003645",
    }
    config = {
        "epoching": {"length_s": 8.0, "step_s": 4.0},
        "features": {},
    }

    out = compute_meg_features(signals, config)

    assert isinstance(out, pd.DataFrame)
    assert len(out) == 3
    for col in (
        "meg_mag_alpha",
        "meg_grad_alpha",
        "meg_alpha",
        "meg_beta_alpha",
        "meg_hjorth_mobility",
        "meg_hjorth_complexity",
        "meg_permutation_entropy",
        "meg_highfreq_power_30_45",
        "meg_entropy_metric",
        "qc_ok_meg",
    ):
        assert col in out.columns
    assert np.all(np.isfinite(out["meg_alpha"].to_numpy(dtype=float)))
    assert np.all(out["meg_alpha"].to_numpy(dtype=float) > 0.0)
    assert np.all(np.isfinite(out["meg_highfreq_power_30_45"].to_numpy(dtype=float)))
    assert set(out["meg_entropy_metric"].astype(str).unique()) == {"permutation_entropy"}
    assert set(out["meg_entropy_backend"].astype(str).unique()) == {"numpy"}


def test_compute_meg_features_can_split_families_from_combined_meg_names_only():
    """Fallback family inference should still work when preprocess only exposed `meg`."""
    from mndm.features.meg import compute_meg_features

    sfreq = 100.0
    times = np.arange(0.0, 12.0, 1.0 / sfreq, dtype=np.float32)
    signals = {
        "signals": {
            "meg": np.vstack(
                [
                    np.sin(2 * np.pi * 8.0 * times),
                    np.sin(2 * np.pi * 10.0 * times),
                    np.sin(2 * np.pi * 12.0 * times),
                    np.sin(2 * np.pi * 18.0 * times),
                ]
            ),
        },
        "channels": {
            "meg": ["MEG0111", "MEG0112", "MEG0121", "MEG0122"],
        },
        "sfreq": sfreq,
    }
    out = compute_meg_features(signals, {"epoching": {"length_s": 6.0, "step_s": 3.0}, "features": {}})
    assert not out.empty
    assert "meg_mag_alpha" in out.columns
    assert "meg_grad_alpha" in out.columns
    assert "meg_alpha" in out.columns


def test_compute_meg_features_exports_mag_helmet_groups_without_changing_global_surface():
    """Configured MAG helmet groups should emit suffixed features only."""
    from mndm.features.meg import compute_meg_features

    sfreq = 100.0
    times = np.arange(0.0, 12.0, 1.0 / sfreq, dtype=np.float32)
    names = ["MLF11", "MLF12", "MRF11", "MRF12", "MLO11", "MLO12"]
    signals = {
        "signals": {
            "meg_mag": np.vstack(
                [
                    np.sin(2 * np.pi * freq * times)
                    for freq in (6.0, 8.0, 10.0, 12.0, 14.0, 16.0)
                ]
            ),
        },
        "channels": {"meg_mag": names},
        "sfreq": sfreq,
        "dataset_id": "ds003568",
    }
    config = {
        "epoching": {"length_s": 6.0, "step_s": 3.0},
        "features": {},
        "robustness": {
            "meg_ensembles": {
                "enabled": True,
                "sensor_family": "mag",
                "min_channels": 3,
                "groups": {
                    "anterior": names[:3],
                    "posterior": names[3:],
                    "too_small": names[:2],
                },
            },
        },
    }

    out = compute_meg_features(signals, config)

    assert "meg_alpha" in out.columns
    assert "meg_alpha__g_anterior" in out.columns
    assert "meg_alpha__g_posterior" in out.columns
    assert "meg_alpha__g_too_small" not in out.columns
    assert np.all(np.isfinite(out["meg_alpha__g_anterior"].to_numpy(dtype=float)))
