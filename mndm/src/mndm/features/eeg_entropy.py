"""Small NumPy-only entropy helpers shared by EEG feature paths."""

from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def compute_permutation_entropy(
    data: np.ndarray,
    order: int = 5,
    delay: int = 1,
    normalize: bool = True,
) -> float:
    """Compute permutation entropy deterministically with NumPy only."""
    x = np.asarray(data, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    order_i = int(order)
    delay_i = int(delay)
    if order_i < 2 or delay_i < 1:
        return np.nan
    if n < (order_i - 1) * delay_i + 2:
        return np.nan

    y = x[::delay_i]
    if y.size < order_i + 1:
        return np.nan
    emb = sliding_window_view(y, window_shape=order_i)
    if emb.size == 0:
        return np.nan

    patterns = np.argsort(emb, axis=1, kind="mergesort")
    pat_view = np.ascontiguousarray(patterns).view(
        np.dtype((np.void, patterns.dtype.itemsize * patterns.shape[1]))
    )
    _, counts = np.unique(pat_view, return_counts=True)
    if counts.size == 0:
        return np.nan

    p = counts.astype(np.float64) / float(np.sum(counts))
    h_val = -np.sum(p * np.log(p + 1e-32))
    if normalize:
        denom = math.log(math.factorial(order_i))
        if denom <= 0:
            return np.nan
        h_val /= denom
    return float(h_val)
