"""Tests for spike-rate population feature extraction."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def test_compute_ephys_features_empty() -> None:
    from mndm.features.ephys import compute_ephys_features

    assert compute_ephys_features({"signals": {}, "sfreq": 20.0}, {}).empty


def test_compute_ephys_features_window_contract() -> None:
    from mndm.features.ephys import compute_ephys_features

    rng = np.random.default_rng(42)
    rates = rng.poisson(4.0, size=(8, 200)).astype(float)
    result = compute_ephys_features(
        {"signals": {"ephys": rates}, "sfreq": 20.0, "dataset_id": "synthetic"},
        {"epoching": {"length_s": 2.0, "step_s": 1.0}},
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 9
    assert {
        "epoch_id",
        "t_start",
        "t_end",
        "ephys_mean_rate_hz",
        "ephys_participation_ratio",
        "ephys_pairwise_corr_mean",
        "ephys_rate_entropy",
        "ephys_ar1",
    }.issubset(result.columns)
    assert np.isfinite(result["ephys_mean_rate_hz"]).all()
    assert (result["ephys_participation_ratio"] > 0).all()
