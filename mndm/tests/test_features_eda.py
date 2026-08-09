"""Tests for EDA (electrodermal activity) feature extraction."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.features.eda import (
    _decompose_session_scipy,
    _downsample_1d,
    _resolve_eda_config,
    compute_eda_features,
)


# ---------------------------------------------------------------------------
# Synthetic signal helpers
# ---------------------------------------------------------------------------

def _synthetic_eda(
    duration_s: float = 60.0,
    sfreq: float = 250.0,
    tonic_start: float = 2.0,
    tonic_end: float = 3.0,
    scr_times_s: tuple[float, ...] = (5.0, 15.0, 25.0, 35.0, 45.0),
    scr_amplitude: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Build a plausible EDA signal: linear tonic drift + Gaussian SCR bumps."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * sfreq)
    t = np.arange(n) / sfreq
    tonic = np.linspace(tonic_start, tonic_end, n)
    phasic = np.zeros(n)
    for t0 in scr_times_s:
        phasic += scr_amplitude * np.exp(-0.5 * ((t - t0) / 0.75) ** 2)
    noise = rng.normal(scale=0.005, size=n)
    return tonic + phasic + noise


def _signal_payload(eda_1d: np.ndarray, sfreq: float) -> dict:
    return {
        "signals": {"eda": eda_1d[None, :]},
        "sfreq": sfreq,
        "channels": {"eda": ["EDA"]},
        "dataset_id": None,
        "file_path": None,
    }


# ---------------------------------------------------------------------------
# _resolve_eda_config
# ---------------------------------------------------------------------------

def test_resolve_eda_config_defaults():
    cfg = _resolve_eda_config({})
    assert cfg["enabled"] is True
    assert cfg["target_sfreq_hz"] == 50.0
    assert cfg["decomposition_method"] == "neurokit"
    assert cfg["min_signal_range_uS"] == 0.01


def test_resolve_eda_config_override():
    cfg = _resolve_eda_config({"features": {"eda": {"enabled": False, "target_sfreq_hz": 20.0}}})
    assert cfg["enabled"] is False
    assert cfg["target_sfreq_hz"] == 20.0
    # Untouched defaults remain present.
    assert cfg["decomposition_method"] == "neurokit"


# ---------------------------------------------------------------------------
# _downsample_1d
# ---------------------------------------------------------------------------

def test_downsample_1d_reduces_length_and_sfreq():
    x = np.sin(2 * np.pi * 0.1 * np.arange(4000) / 250.0)
    ds, new_sfreq = _downsample_1d(x, sfreq=250.0, target_sfreq=50.0)
    assert new_sfreq == pytest.approx(50.0, rel=0.01)
    assert len(ds) == pytest.approx(len(x) * 50.0 / 250.0, rel=0.05)


def test_downsample_1d_noop_when_target_above_native():
    x = np.arange(100, dtype=float)
    ds, new_sfreq = _downsample_1d(x, sfreq=50.0, target_sfreq=250.0)
    assert new_sfreq == 50.0
    assert np.array_equal(ds, x)


def test_downsample_1d_noop_on_empty_array():
    x = np.array([], dtype=float)
    ds, new_sfreq = _downsample_1d(x, sfreq=250.0, target_sfreq=50.0)
    assert new_sfreq == 250.0
    assert ds.size == 0


# ---------------------------------------------------------------------------
# _decompose_session_scipy
# ---------------------------------------------------------------------------

def test_decompose_session_scipy_recovers_tonic_trend():
    eda = _synthetic_eda(tonic_start=2.0, tonic_end=4.0, scr_times_s=())
    result = _decompose_session_scipy(eda, sfreq=250.0, scr_min_distance_s=1.0, scr_prominence_mult=5.0)
    tonic = result["tonic"]
    # Tonic estimate should track the overall rising trend (start < end).
    assert np.nanmean(tonic[: len(tonic) // 10]) < np.nanmean(tonic[-len(tonic) // 10 :])


def test_decompose_session_scipy_detects_scr_peaks():
    eda = _synthetic_eda(scr_times_s=(5.0, 15.0, 25.0))
    result = _decompose_session_scipy(eda, sfreq=250.0, scr_min_distance_s=1.0, scr_prominence_mult=5.0)
    # Expect roughly one detected peak per injected SCR bump (allow slack).
    assert 2 <= len(result["scr_peaks"]) <= 6


# ---------------------------------------------------------------------------
# compute_eda_features — end to end
# ---------------------------------------------------------------------------

def test_compute_eda_features_no_channel_returns_empty():
    signals = {"signals": {}, "sfreq": 250.0}
    df = compute_eda_features(signals, {})
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_compute_eda_features_disabled_returns_empty():
    eda = _synthetic_eda()
    signals = _signal_payload(eda, sfreq=250.0)
    df = compute_eda_features(signals, {"features": {"eda": {"enabled": False}}})
    assert df.empty


def test_compute_eda_features_basic_schema():
    eda = _synthetic_eda(duration_s=40.0)
    signals = _signal_payload(eda, sfreq=250.0)
    df = compute_eda_features(signals, {})
    assert not df.empty
    expected_cols = {
        "eda_tonic_scl",
        "eda_tonic_slope",
        "eda_phasic_scr_rate",
        "eda_phasic_scr_amp",
        "eda_phasic_scr_count",
        "eda_phasic_auc",
        "eda_arousal_index",
        "eda_source_channel",
        "qc_ok_eda",
        "epoch_id",
        "t_start",
        "t_end",
    }
    assert expected_cols.issubset(set(df.columns))
    # Default epoching (8s/4s) over 40s should yield multiple epochs.
    assert len(df) >= 5
    assert (df["eda_source_channel"] == "EDA").all()


def test_compute_eda_features_qc_ok_for_good_signal():
    eda = _synthetic_eda(duration_s=40.0)
    signals = _signal_payload(eda, sfreq=250.0)
    df = compute_eda_features(signals, {})
    # A clean, non-flat synthetic signal should pass epoch-level QC.
    assert df["qc_ok_eda"].mean() > 0.5


def test_compute_eda_features_flat_signal_fails_session_qc():
    eda = np.full(int(40.0 * 250.0), 2.0)
    signals = _signal_payload(eda, sfreq=250.0)
    df = compute_eda_features(signals, {})
    assert not df.empty
    # Flat/constant signal has zero range -> session-level QC gate rejects it.
    assert (~df["qc_ok_eda"]).all()
    assert df["eda_tonic_scl"].isna().all()


def test_compute_eda_features_tonic_slope_sign_matches_trend():
    eda = _synthetic_eda(duration_s=60.0, tonic_start=1.0, tonic_end=5.0, scr_times_s=())
    signals = _signal_payload(eda, sfreq=250.0)
    df = compute_eda_features(signals, {})
    ok = df[df["qc_ok_eda"]]
    assert not ok.empty
    # Rising tonic drift should yield a positive median slope.
    assert ok["eda_tonic_slope"].median() > 0


def test_compute_eda_features_respects_dataset_epoch_override():
    eda = _synthetic_eda(duration_s=40.0)
    signals = _signal_payload(eda, sfreq=250.0)
    signals["dataset_id"] = "dsTEST"
    config = {"epoching": {"length_s": 8.0, "step_s": 4.0, "datasets": {"dsTEST": {"length_s": 4.0, "step_s": 4.0}}}}
    df = compute_eda_features(signals, config)
    # Shorter epochs -> more of them over the same recording.
    assert len(df) >= 9
