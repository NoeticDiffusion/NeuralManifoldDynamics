"""Tests for population-rate binning and event alignment."""

from __future__ import annotations

import numpy as np

from neuropixel_ingest.binning import bin_spike_times, build_event_aligned_tensor


def test_bin_spike_times_returns_hz_matrix() -> None:
    rates, times = bin_spike_times(
        [np.asarray([0.05, 0.15, 0.85]), np.asarray([0.55])],
        bin_sec=0.5,
        start_sec=0.0,
        stop_sec=1.0,
    )
    assert rates.shape == (2, 2)
    assert np.allclose(rates, [[4.0, 2.0], [0.0, 2.0]])
    assert np.allclose(times, [0.25, 0.75])


def test_event_alignment_preserves_event_and_unit_axes() -> None:
    rates = np.asarray([[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 0.0, 1.0]])
    times = np.asarray([0.25, 0.75, 1.25, 1.75])
    tensor, offsets = build_event_aligned_tensor(
        rates, times, [1.25], pre_sec=0.5, post_sec=0.5
    )
    assert tensor.shape == (1, 2, 2)
    assert np.allclose(offsets, [-0.5, 0.0])
    assert np.allclose(tensor[0], [[2.0, 1.0], [3.0, 0.0]])
