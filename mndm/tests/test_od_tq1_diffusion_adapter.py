"""OD-TQ1: truth-known diffusion replay at the MNDM adapter boundary."""

from __future__ import annotations

import numpy as np

from mndm.dynamical_families.diffusion_geometry import estimate_local_diffusion_geometry


def test_od_tq1_recovers_and_exposes_standard_diffusion_aliases() -> None:
    rng = np.random.default_rng(103)
    n, dt, sigma = 900, 0.01, 0.3
    state = np.vstack([np.zeros((1, 2)), np.cumsum(rng.normal(0, sigma * np.sqrt(dt), (n - 1, 2)), axis=0)])
    result = estimate_local_diffusion_geometry(
        state, np.arange(n) * dt, neighborhood_k=n - 1, min_samples=100, min_neighborhood_samples=100
    )
    assert result["computation_status"] == "computed"
    series = result["series"]
    assert np.shares_memory(series["diffusion_tensor"], series["a_hat"])
    assert np.shares_memory(series["diffusion_total"], series["D_total"])
    assert np.isclose(np.nanmean(series["diffusion_total"]), 2 * sigma**2, rtol=0.2)


def test_od_tq1_rejects_gap_as_not_testable() -> None:
    state = np.arange(100, dtype=float)[:, None]
    time = np.arange(100, dtype=float)
    time[50:] += 1.0
    result = estimate_local_diffusion_geometry(
        state, time, neighborhood_k=30, min_samples=30, min_neighborhood_samples=10
    )
    assert result["computation_status"] == "not_testable"
