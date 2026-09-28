"""Small helpers for processing finite temporal segments independently."""

from __future__ import annotations

from typing import Callable

import numpy as np


def finite_slices(values: np.ndarray) -> list[slice]:
    """Return maximal half-open runs of finite values in a one-dimensional array."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError("finite_slices expects a one-dimensional array")
    finite = np.isfinite(arr)
    if not finite.any():
        return []
    edges = np.diff(np.r_[False, finite, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [slice(int(start), int(stop)) for start, stop in zip(starts, stops)]


def apply_to_finite_segments(
    values: np.ndarray,
    operation: Callable[[np.ndarray], np.ndarray],
    *,
    fallback: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, list[slice], list[slice]]:
    """Apply an operation independently to each finite run, preserving gaps.

    No output sample in one finite run can depend on samples from another run.
    If ``operation`` fails for a run, an optional fallback is applied to that
    run alone. The returned tuple is ``(output, runs, fallback_runs)``.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("apply_to_finite_segments expects a one-dimensional array")
    out = np.full(arr.shape, np.nan, dtype=float)
    runs = finite_slices(arr)
    fallback_runs: list[slice] = []
    for run in runs:
        segment = arr[run]
        try:
            transformed = np.asarray(operation(segment), dtype=float)
        except Exception:
            if fallback is None:
                raise
            transformed = np.asarray(fallback(segment), dtype=float)
            fallback_runs.append(run)
        if transformed.shape != segment.shape:
            raise ValueError("segment operation must preserve the input shape")
        out[run] = transformed
    return out, runs, fallback_runs
