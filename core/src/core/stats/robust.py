"""Robust summaries and lightweight confidence intervals.

SciPy-free helpers for medians, trimmed means, order-statistic CIs for the
median, normal CIs for the mean, and column-wise summaries of 2-D arrays.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Any, Dict, Sequence

import numpy as np


def robust_1d(values: np.ndarray, summary: str = "median", trim_pct: float = 0.0) -> float:
    """Compute a robust 1D summary (median or trimmed mean).

    Args:
        values: 1-D or flattenable numeric array; non-finite values are dropped.
        summary: ``median``, ``mean``, or ``mean`` with ``trim_pct`` in ``(0,1)``
            for a trimmed mean.
        trim_pct: Fraction trimmed from each tail when using trimmed mean.

    Returns:
        Scalar summary, or NaN if no finite values.
    """

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")

    summary = str(summary).lower()
    if summary == "median":
        return float(np.median(arr))

    if summary == "mean" and 0.0 < trim_pct < 1.0:
        frac_each = trim_pct / 2.0
        n = arr.size
        k = int(frac_each * n)
        if k * 2 >= n:
            return float(np.mean(arr))
        arr_sorted = np.sort(arr)
        trimmed = arr_sorted[k : n - k]
        return float(np.mean(trimmed))

    return float(np.mean(arr))


def ci_median_orderstat(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Approximate :math:`(1-\\alpha)` CI for the median using order-statistic ranks.

    Args:
        values: 1-D numeric array.
        alpha: Two-sided error rate (default 0.05 for a nominal 95% interval).

    Returns:
        ``(low, high)`` bounds, or ``(nan, nan)`` if too few samples.
    """

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n < 3:
        return float("nan"), float("nan")

    z = float(NormalDist().inv_cdf(1.0 - float(alpha) / 2.0))
    m = 0.5 * (n - 1)
    delta = 0.5 * z * np.sqrt(float(n))
    lo = int(max(0, np.floor(m - delta)))
    hi = int(min(n - 1, np.ceil(m + delta)))

    lo_val = float(np.partition(arr, lo)[lo])
    hi_val = float(np.partition(arr, hi)[hi])
    if lo_val > hi_val:
        lo_val, hi_val = hi_val, lo_val
    return lo_val, hi_val


def ci_mean_normal(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Normal-approximation CI for the mean (SciPy-free).

    Args:
        values: 1-D numeric array.
        alpha: Two-sided error rate.

    Returns:
        ``(low, high)``, or ``(nan, nan)`` if fewer than two finite samples.
    """

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n < 2:
        return float("nan"), float("nan")
    z = float(NormalDist().inv_cdf(1.0 - float(alpha) / 2.0))
    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    se = sd / float(np.sqrt(n))
    return mu - z * se, mu + z * se


def summarize_array(
    values: np.ndarray,
    axis_names: Sequence[str],
    cfg: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """Summarize columns of a 2-D array with robust point estimates and CIs.

    Args:
        values: Array shaped ``(n_rows, len(axis_names))``.
        axis_names: Names for each column (MNPS axis or feature names).
        cfg: Must include ``robustness`` settings (summary type, CI method,
            bootstrap count, seed, etc.).

    Returns:
        Mapping from axis name to ``{"point", "ci_low", "ci_high"}`` floats.

    Raises:
        ValueError: If shape does not match ``axis_names``.
    """

    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(axis_names):
        raise ValueError("values must be 2-D with second dimension matching axis_names")

    robustness = cfg.get("robustness", {}) if isinstance(cfg, dict) else {}
    summary_type = robustness.get("summary", "median")
    trim_pct = float(robustness.get("trim_pct", 0.0) or 0.0)
    bootstrap_n = int(robustness.get("bootstrap_n", 0) or 0)
    seed = int(robustness.get("seed", 42))
    ci_method = str(robustness.get("ci_method", "auto")).lower()
    ci_level = float(robustness.get("ci_level", 0.95) or 0.95)
    alpha = float(max(min(1.0 - ci_level, 1.0), 0.0))
    if alpha <= 0.0:
        alpha = 0.05

    rng = np.random.default_rng(seed)
    out: Dict[str, Dict[str, float]] = {}

    for col_idx, name in enumerate(axis_names):
        col = values[:, col_idx]
        mask = np.isfinite(col)
        finite = col[mask]

        point = robust_1d(finite, summary=summary_type, trim_pct=trim_pct)

        if finite.size == 0:
            ci_low = float("nan")
            ci_high = float("nan")
        elif ci_method in {"auto", "orderstat"} and str(summary_type).lower() == "median":
            ci_low, ci_high = ci_median_orderstat(finite, alpha=alpha)
        elif ci_method in {"auto", "normal"} and str(summary_type).lower() == "mean" and not (0.0 < trim_pct < 1.0):
            ci_low, ci_high = ci_mean_normal(finite, alpha=alpha)
        elif bootstrap_n <= 0:
            ci_low = float("nan")
            ci_high = float("nan")
        else:
            samples = []
            n = finite.size
            for _ in range(bootstrap_n):
                idx = rng.integers(0, n, size=n)
                sample = finite[idx]
                samples.append(robust_1d(sample, summary=summary_type, trim_pct=trim_pct))
            arr_samples = np.asarray(samples, dtype=float)
            ci_low = float(np.percentile(arr_samples, 2.5))
            ci_high = float(np.percentile(arr_samples, 97.5))

        out[str(name)] = {
            "point": float(point),
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

    return out

