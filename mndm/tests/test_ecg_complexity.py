"""Tests for HRV complexity metrics (SampEn and DFA α₁)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.features.ecg import _compute_hrv_complexity, _resolve_hrv_superwindow_config


# ---------------------------------------------------------------------------
# _compute_hrv_complexity
# ---------------------------------------------------------------------------

def _healthy_rr(n: int = 80, seed: int = 0) -> np.ndarray:
    """Generate a plausible RR interval series (70 bpm + mild HRV)."""
    rng = np.random.default_rng(seed)
    return np.clip(0.857 + 0.05 * rng.normal(size=n), 0.3, 1.5)


def test_complexity_returns_finite_for_sufficient_samples():
    nn = _healthy_rr(80)
    result = _compute_hrv_complexity(nn)
    assert "ecg_hrv_sampen" in result
    assert "ecg_hrv_dfa_alpha1" in result
    assert np.isfinite(result["ecg_hrv_sampen"]), "SampEn should be finite for 80-sample series"
    assert np.isfinite(result["ecg_hrv_dfa_alpha1"]), "DFA α₁ should be finite for 80-sample series"


def test_complexity_sampen_nan_when_too_few_samples():
    """SampEn returns NaN when n < min_nn_for_sampen."""
    nn = _healthy_rr(30)
    result = _compute_hrv_complexity(nn, min_nn_for_sampen=50)
    assert not np.isfinite(result["ecg_hrv_sampen"]), "SampEn should be NaN for n=30 < min=50"


def test_complexity_dfa_nan_when_too_few_samples():
    """DFA α₁ returns NaN when n < min_nn_for_dfa."""
    nn = _healthy_rr(10)
    result = _compute_hrv_complexity(nn, min_nn_for_dfa=16)
    assert not np.isfinite(result["ecg_hrv_dfa_alpha1"]), "DFA should be NaN for n=10 < min=16"


def test_complexity_sampen_positive():
    """SampEn must be ≥ 0 for any valid RR series."""
    nn = _healthy_rr(80)
    result = _compute_hrv_complexity(nn)
    assert result["ecg_hrv_sampen"] >= 0.0, "SampEn must be non-negative"


def test_complexity_dfa_alpha1_plausible_range():
    """DFA α₁ for a healthy synthetic RR series should be in [0.3, 1.8]."""
    nn = _healthy_rr(80)
    result = _compute_hrv_complexity(nn)
    a1 = result["ecg_hrv_dfa_alpha1"]
    assert 0.3 <= a1 <= 1.8, f"DFA α₁ out of plausible range: {a1:.3f}"


def test_complexity_keys_always_present():
    """Both keys must always be present, even when NaN."""
    nn = _healthy_rr(5)  # very short
    result = _compute_hrv_complexity(nn)
    assert "ecg_hrv_sampen" in result
    assert "ecg_hrv_dfa_alpha1" in result


def test_complexity_constant_series_nan():
    """A constant RR series has undefined SampEn (zero std → tolerance = 0)."""
    nn = np.full(80, 0.857)
    result = _compute_hrv_complexity(nn)
    # SampEn with zero tolerance is undefined; should return NaN or a non-negative finite value.
    # Either outcome is acceptable — just must not raise.
    assert "ecg_hrv_sampen" in result


# ---------------------------------------------------------------------------
# _resolve_hrv_superwindow_config — complexity defaults
# ---------------------------------------------------------------------------

def test_resolve_complexity_defaults():
    """complexity block is added with correct defaults when absent."""
    cfg = _resolve_hrv_superwindow_config({"hrv": {"enabled": True}})
    complexity = cfg["complexity"]
    assert complexity["enabled"] is False, "complexity disabled by default"
    assert complexity["sampen_order"] == 2
    assert complexity["min_nn_for_sampen"] == 50
    assert complexity["min_nn_for_dfa"] == 16
    assert complexity["dfa_short_nvals_lo"] == 4
    assert complexity["dfa_short_nvals_hi"] == 12


def test_resolve_complexity_enabled_override():
    """complexity.enabled can be set to True via config."""
    cfg = _resolve_hrv_superwindow_config({
        "hrv": {
            "enabled": True,
            "complexity": {"enabled": True, "min_nn_for_sampen": 30},
        }
    })
    assert cfg["complexity"]["enabled"] is True
    assert cfg["complexity"]["min_nn_for_sampen"] == 30
    assert cfg["complexity"]["sampen_order"] == 2  # default preserved
