"""T
transients.py
ransient detection for R time-series (z-spike/CUSUM tagging).

Inputs
------
- r_series: array of per-epoch R values.
- z_thresh: z-score threshold for spike detection.
- pad_epochs: number of epochs to pad around detected transients.

Outputs
-------
- Boolean mask marking transient regions (True = transient).
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


def detect_transients(r_series: Iterable[float], z_thresh: float = 3.0, pad_epochs: int = 1) -> np.ndarray:
    """Detect transients in a 1-D series using robust z-score thresholding.
    
    Parameters
    ----------
    r_series
        Array of per-epoch R values.
    z_thresh
        Z-score threshold for spike detection.
    pad_epochs
        Number of epochs to pad around detected transients.
    
    Returns
    -------
    Boolean array (True = transient epoch).
    """
    r_array = np.asarray(list(r_series), dtype=float)
    if r_array.size == 0:
        return np.zeros((0,), dtype=bool)

    finite_mask = np.isfinite(r_array)
    if int(finite_mask.sum()) < 4:
        return np.zeros(r_array.shape[0], dtype=bool)

    # Robust center/scale to avoid outliers inflating sigma and hiding transients.
    finite_vals = r_array[finite_mask]
    med = float(np.median(finite_vals))
    abs_dev = np.abs(finite_vals - med)
    mad = float(np.median(abs_dev))
    if np.isfinite(mad) and mad > 0:
        sigma = 1.4826 * mad
    else:
        # If MAD collapses (e.g. highly quantized/flat signals), estimate scale
        # from the central bulk and avoid letting a single large spike set sigma.
        q = float(np.quantile(abs_dev, 0.8)) if abs_dev.size > 0 else 0.0
        core = finite_vals[abs_dev <= q] if q > 0 else finite_vals[abs_dev == 0]
        sigma = float(np.std(core, ddof=0)) if core.size > 1 else 0.0
        if not np.isfinite(sigma) or sigma <= 0:
            pos_dev = abs_dev[abs_dev > 0]
            if pos_dev.size > 0:
                sigma = float(np.min(pos_dev))
            else:
                sigma = float(np.std(finite_vals, ddof=0))
    if not np.isfinite(sigma) or sigma <= 0:
        return np.zeros(r_array.shape[0], dtype=bool)

    z_scores = np.full(r_array.shape[0], np.nan, dtype=float)
    z_scores[finite_mask] = np.abs((r_array[finite_mask] - med) / sigma)

    transient_mask = np.zeros(r_array.shape[0], dtype=bool)
    transient_mask[finite_mask] = z_scores[finite_mask] > float(z_thresh)

    # Vectorized padding (binary dilation via 1-D convolution), O(T).
    pad = max(0, int(pad_epochs))
    if pad > 0 and transient_mask.any():
        kernel = np.ones((2 * pad + 1,), dtype=int)
        transient_mask = np.convolve(transient_mask.astype(int), kernel, mode="same") > 0

    logger.debug(
        "Detected %d transient epochs (%.1f%%)",
        int(np.sum(transient_mask)),
        float(100.0 * np.mean(transient_mask)),
    )
    return transient_mask


