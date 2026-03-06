"""Tests for extended MNPS / NDT coordinate utilities."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def test_compute_kappa_E_1d_basic():
    from mndm.extensions import compute_kappa_E_1d

    dt = 0.1
    t = np.arange(0, 2, dt)
    # Smooth quadratic energy curve
    E = (t - 1.0) ** 2

    kappa = compute_kappa_E_1d(E, dt=dt)
    assert kappa.shape == E.shape
    # Boundary values should be NaN
    assert np.isnan(kappa[0])
    assert np.isnan(kappa[-1])
    # Interior values should be finite
    assert np.all(np.isfinite(kappa[1:-1]))


def test_compute_OKoh_on_simple_graph():
    from mndm.extensions import compute_OKoh

    # Simple 3-node line graph with stronger edges on 0-1 and 1-2
    C = np.array(
        [
            [0.0, 0.9, 0.1],
            [0.9, 0.0, 0.8],
            [0.1, 0.8, 0.0],
        ],
        dtype=float,
    )

    res = compute_OKoh(C)
    assert "thresholds" in res
    assert "beta0" in res
    assert "beta1" in res
    assert "OKoh0" in res
    assert "OKoh1" in res
    assert res["thresholds"].shape == res["beta0"].shape == res["beta1"].shape


def test_compute_TIG_autocorr_reasonable_values():
    from mndm.extensions import compute_TIG_autocorr

    dt = 0.1
    t = np.arange(0, 20, dt)
    # Slowly varying 1D trajectory with clear temporal structure
    s = np.sin(2 * np.pi * 0.1 * t)
    max_lag_sec = 5.0

    res = compute_TIG_autocorr(s, dt=dt, max_lag_sec=max_lag_sec, n_lags=20)
    assert "lags_sec" in res
    assert "autocorr" in res
    assert res["lags_sec"].shape == res["autocorr"].shape
    # tau and TIG should be finite and TIG clipped to [0, 1]
    assert np.isfinite(res["tau"])
    assert np.isfinite(res["TIG"])
    assert 0.0 <= res["TIG"] <= 1.0
    # Smooth signal should not trigger provisional flag in typical settings
    assert res.get("provisional") in (False, 0)


def test_compute_TIG_autocorr_saturates_and_flags_provisional():
    from mndm.extensions import compute_TIG_autocorr

    dt = 0.1
    t = np.arange(0, 20, dt)
    # Nearly constant trajectory -> autocorr ~ 1, ill-conditioned exponential fit
    s = np.ones_like(t)
    max_lag_sec = 5.0

    res = compute_TIG_autocorr(s, dt=dt, max_lag_sec=max_lag_sec, n_lags=20)
    assert "TIG" in res
    assert "tau" in res
    # Should be saturated to TIG=1.0 with provisional=True
    assert np.isfinite(res["tau"])
    assert res["TIG"] == pytest.approx(1.0)
    assert res.get("provisional") in (True, 1)


@pytest.mark.parametrize("n_channels, n_times", [(3, 256)])
def test_compute_rfm_shapes(n_channels, n_times):
    pytest.importorskip("scipy")
    from mndm.extensions import compute_rfm

    sfreq = 64.0
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n_channels, n_times)).astype(float)

    # Use small windows in seconds on this synthetic data
    window_sec = 2.0
    step_sec = 1.0

    res = compute_rfm(
        x,
        sfreq=sfreq,
        window_sec=window_sec,
        step_sec=step_sec,
        n_modes=2,
        band=None,
    )

    assert res.eigvals.shape[0] == res.times.shape[0]
    assert res.eigvecs.shape[0] == res.times.shape[0]
    assert res.eigvals.shape[1] == n_channels
    assert res.eigvecs.shape[1] == 2  # n_modes
    assert res.eigvecs.shape[2] == n_channels
    assert res.dominance.shape == res.times.shape
    # Dominance is a normalized ratio in [0, 1]
    assert np.all(res.dominance >= 0.0) and np.all(res.dominance <= 1.0 + 1e-6)


def test_compute_rfm_handles_n_modes_gt_channels():
    """When n_modes > n_channels, compute_rfm should not crash.

    It should return eigvecs with the requested n_modes, padding the extra
    modes with NaN.
    """
    pytest.importorskip("scipy")
    from mndm.extensions import compute_rfm

    sfreq = 64.0
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 256)).astype(float)  # 2 channels only

    res = compute_rfm(
        x,
        sfreq=sfreq,
        window_sec=2.0,
        step_sec=1.0,
        n_modes=3,
        band=None,
    )

    assert res.eigvecs.shape[1] == 3
    assert res.eigvecs.shape[2] == 2
    # Third mode cannot exist for 2-channel data -> should be NaN-padded.
    assert np.all(np.isnan(res.eigvecs[:, 2, :]))

