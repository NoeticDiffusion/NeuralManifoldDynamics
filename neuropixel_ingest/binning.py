"""Spike-time binning and event-aligned population-rate tensors."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def infer_time_bounds(spike_times: Iterable[np.ndarray]) -> tuple[float, float]:
    """Infer inclusive session bounds from finite spike times."""
    nonempty = [np.asarray(values, dtype=float).reshape(-1) for values in spike_times]
    finite = [values[np.isfinite(values)] for values in nonempty if np.isfinite(values).any()]
    if not finite:
        raise ValueError("Cannot infer time bounds from empty spike trains.")
    return float(min(values.min() for values in finite)), float(max(values.max() for values in finite))


def bin_spike_times(
    spike_times: Sequence[np.ndarray],
    *,
    bin_sec: float,
    start_sec: float | None = None,
    stop_sec: float | None = None,
    smoothing_sigma_sec: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert spike trains to a unit-by-bin firing-rate matrix in Hz.

    ``stop_sec`` is exclusive.  Empty units are supported as long as explicit
    bounds are supplied, which is essential for recordings with silent units.
    """
    if not np.isfinite(bin_sec) or bin_sec <= 0:
        raise ValueError("bin_sec must be a positive finite number.")
    if start_sec is None or stop_sec is None:
        inferred_start, inferred_stop = infer_time_bounds(spike_times)
        start_sec = inferred_start if start_sec is None else start_sec
        stop_sec = inferred_stop + bin_sec if stop_sec is None else stop_sec
    if not np.isfinite(start_sec) or not np.isfinite(stop_sec) or stop_sec <= start_sec:
        raise ValueError("Require finite bounds with stop_sec > start_sec.")

    n_bins = int(np.ceil((stop_sec - start_sec) / bin_sec))
    edges = start_sec + np.arange(n_bins + 1, dtype=float) * bin_sec
    edges[-1] = max(edges[-1], float(stop_sec))
    rates = np.zeros((len(spike_times), n_bins), dtype=np.float32)
    for unit_idx, values in enumerate(spike_times):
        times = np.asarray(values, dtype=float).reshape(-1)
        valid = times[np.isfinite(times) & (times >= start_sec) & (times < stop_sec)]
        rates[unit_idx] = np.histogram(valid, bins=edges)[0].astype(np.float32) / float(bin_sec)

    if smoothing_sigma_sec is not None and smoothing_sigma_sec > 0:
        try:
            from scipy.ndimage import gaussian_filter1d
        except ImportError as exc:
            raise RuntimeError("scipy is required for smoothed spike-rate matrices.") from exc
        sigma_bins = float(smoothing_sigma_sec) / float(bin_sec)
        rates = gaussian_filter1d(rates, sigma=sigma_bins, axis=1, mode="nearest").astype(np.float32)
    return rates, (edges[:-1] + edges[1:]) / 2.0


def build_event_aligned_tensor(
    rates: np.ndarray,
    rate_times_sec: np.ndarray,
    event_times_sec: Sequence[float],
    *,
    pre_sec: float,
    post_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract fixed event-aligned windows as ``[events, bins, units]``.

    Events whose complete window is unavailable are filled with NaN, retaining
    event order and making missing coverage explicit.
    """
    rate_matrix = np.asarray(rates, dtype=float)
    times = np.asarray(rate_times_sec, dtype=float).reshape(-1)
    if rate_matrix.ndim != 2 or rate_matrix.shape[1] != len(times):
        raise ValueError("rates must have shape [n_units, n_bins] matching rate_times_sec.")
    if len(times) < 2:
        raise ValueError("At least two rate bins are required for event alignment.")
    bin_sec = float(np.median(np.diff(times)))
    offsets = np.arange(-pre_sec, post_sec, bin_sec, dtype=float)
    output = np.full((len(event_times_sec), len(offsets), rate_matrix.shape[0]), np.nan, dtype=float)
    for event_idx, event_time in enumerate(event_times_sec):
        if not np.isfinite(event_time):
            continue
        wanted = float(event_time) + offsets
        indices = np.rint((wanted - times[0]) / bin_sec).astype(int)
        valid = (indices >= 0) & (indices < len(times))
        close = np.zeros_like(valid)
        close[valid] = np.abs(times[indices[valid]] - wanted[valid]) <= bin_sec * 0.51
        valid &= close
        output[event_idx, valid, :] = rate_matrix[:, indices[valid]].T
    return output, offsets
