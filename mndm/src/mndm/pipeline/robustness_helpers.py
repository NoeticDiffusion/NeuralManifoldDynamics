"""
robustness_helpers.py
Robustness, reliability, and QC summary utilities for MNPS."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .. import ensembles, projection, robustness
from ..jacobian import JacobianResult

logger = logging.getLogger(__name__)

STANDARD_GEOMETRY_POLICY_VERSION = "standard_invalidity_v1"
STANDARD_GEOMETRY_CONSTANT_VAR_TOL = 1e-8
STANDARD_GEOMETRY_MIN_UNIQUE_VALUES = 3
STANDARD_GEOMETRY_MAX_JACOBIAN_CONDITION = 1e10

def _distributional_descriptives(
    values: np.ndarray,
    names: list[str],
) -> Dict[str, Dict[str, Any]]:
    """Compute neutral distributional descriptives for each column.

    This is intentionally analysis-agnostic: no hypothesis tests, no group comparisons,
    just basic functionals of the per-epoch distribution.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(names):
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for j, name in enumerate(names):
        col = arr[:, j]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            out[str(name)] = {
                "n": 0,
                "nan_frac": 1.0,
                "mean": float("nan"),
                "median": float("nan"),
                "std": float("nan"),
                "iqr": float("nan"),
                "delta_mean_median": float("nan"),
            }
            continue

        mean = float(np.mean(finite))
        median = float(np.median(finite))
        std = float(np.std(finite, ddof=0))
        # MAD: median(|x - median(x)|); robust scale. Also export scaled MAD_sigma
        # for approximate comparability to std under normality.
        try:
            mad = float(np.median(np.abs(finite - median)))
        except Exception:
            mad = float("nan")
        mad_sigma = float(1.4826 * mad) if np.isfinite(mad) else float("nan")
        # Shape descriptors use central moments (outlier-sensitive by design).
        skewness = float("nan")
        kurtosis_excess = float("nan")
        if np.isfinite(std) and std > 0:
            try:
                z = (finite - mean) / std
                skewness = float(np.mean(z ** 3))
                kurtosis_excess = float(np.mean(z ** 4) - 3.0)
            except Exception:
                skewness = float("nan")
                kurtosis_excess = float("nan")
        try:
            q25 = float(np.percentile(finite, 25))
            q75 = float(np.percentile(finite, 75))
            iqr = float(q75 - q25)
        except Exception:
            iqr = float("nan")
        out[str(name)] = {
            "n": int(finite.size),
            "nan_frac": float(1.0 - (finite.size / col.size if col.size else 1.0)),
            "mean": mean,
            "median": median,
            "std": std,
            "iqr": iqr,
            "mad": mad,
            "mad_sigma": mad_sigma,
            "skewness": skewness,
            "kurtosis_excess": kurtosis_excess,
            "delta_mean_median": float(mean - median),
        }
    return out


def compute_dist_summary(
    x: np.ndarray,
    coords_9d: Optional[np.ndarray],
    coords_9d_names: list[str],
) -> Dict[str, Any]:
    """Compute distributional summaries for MNPS axes and optional v2 subcoords."""
    result: Dict[str, Any] = {}
    try:
        result["axes"] = _distributional_descriptives(np.asarray(x, dtype=float), ["m", "d", "e"])
    except Exception:
        logger.exception("Failed to compute distributional descriptives for MNPS axes")
        result["axes"] = {}

    if coords_9d is not None and coords_9d_names:
        try:
            result["subcoords"] = _distributional_descriptives(
                np.asarray(coords_9d, dtype=float),
                list(coords_9d_names),
            )
        except Exception:
            logger.exception("Failed to compute distributional descriptives for Stratified MNPS subcoords")
            result["subcoords"] = {}
    return result


def compute_tau_summary(
    values: np.ndarray,
    names: list[str],
    dt_sec: float,
    max_lag_sec: float = 60.0,
    threshold: float = 1.0 / np.e,
    nan_policy: str = "strict",
) -> Dict[str, Dict[str, Any]]:
    """Compute an autocorrelation length (tau) per column.

    Definition used here (simple, reproducible): the first lag where ACF falls below `threshold`
    (default: 1/e). Reported in seconds.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(names):
        return {}
    if not np.isfinite(dt_sec) or dt_sec <= 0:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    max_lag = int(max(1, round(max_lag_sec / dt_sec)))
    policy = str(nan_policy).strip().lower() or "strict"
    if policy not in {"strict", "interpolate"}:
        policy = "strict"

    def _acf_tau_1d(col: np.ndarray) -> float:
        """Internal helper: acf tau 1d."""
        x = np.asarray(col, dtype=float)
        finite_mask = np.isfinite(x)
        if not np.all(finite_mask):
            if policy == "strict":
                return float("nan")
            # Interpolate in index space without changing sample positions.
            if not np.any(finite_mask):
                return float("nan")
            idx = np.arange(x.size, dtype=float)
            x = np.interp(idx, idx[finite_mask], x[finite_mask])
        if x.size < 8:
            return float("nan")
        x = x - float(np.mean(x))
        var = float(np.mean(x * x))
        if not np.isfinite(var) or var <= 0:
            return float("nan")
        n = int(x.size)
        L = min(max_lag, n - 1)
        if L < 1:
            return float("nan")
        # FFT-based autocorrelation (biased) for speed and numerical stability
        nfft = 1
        while nfft < 2 * n:
            nfft *= 2
        fx = np.fft.rfft(x, n=nfft)
        ac = np.fft.irfft(fx * np.conj(fx), n=nfft)[: n]
        acf = ac / (ac[0] if ac[0] != 0 else (var * n))
        acf = np.asarray(acf, dtype=float)
        # find first lag where ACF < threshold
        below = np.where(acf[1 : L + 1] < float(threshold))[0]
        if below.size == 0:
            return float(L * dt_sec)
        lag = int(below[0] + 1)
        return float(lag * dt_sec)

    for j, name in enumerate(names):
        col = arr[:, j]
        tau = _acf_tau_1d(col)
        out[str(name)] = {
            "tau_sec": float(tau),
            "dt_sec": float(dt_sec),
            "max_lag_sec": float(max_lag_sec),
            "threshold": float(threshold),
            "nan_frac": float(1.0 - np.mean(np.isfinite(col))) if col.size else 1.0,
            "nan_policy": policy,
        }
    return out


def compute_tier2_jacobian_metrics(
    jacobian: Optional[np.ndarray],
    jacobian_diagnostics: Optional[Mapping[str, Any]] = None,
    max_windows_for_condition_number: int = 5000,
) -> Dict[str, Any]:
    """Tier-2 MNJ-adjacent metrics from the primary (typically 3×3) Jacobian."""
    if jacobian is None:
        return {}
    J = np.asarray(jacobian, dtype=float)
    if J.ndim != 3 or J.shape[0] == 0 or J.shape[1] != J.shape[2]:
        return {}
    W, D, _ = J.shape

    # Trace series (divergence)
    trace = np.trace(J, axis1=1, axis2=2)
    trace_f = trace[np.isfinite(trace)]
    signed_div: Dict[str, float] = {
        "n": int(trace_f.size),
        "frac_pos": float(np.mean(trace_f > 0)) if trace_f.size else float("nan"),
        "frac_neg": float(np.mean(trace_f < 0)) if trace_f.size else float("nan"),
        "mean_pos": float(np.mean(trace_f[trace_f > 0])) if np.any(trace_f > 0) else float("nan"),
        "mean_neg": float(np.mean(trace_f[trace_f < 0])) if np.any(trace_f < 0) else float("nan"),
        "mean_abs": float(np.mean(np.abs(trace_f))) if trace_f.size else float("nan"),
    }

    # Condition number per window: kappa(J)=sigma_max/sigma_min
    cond = np.full((W,), np.nan, dtype=float)
    finite_rows = np.all(np.isfinite(J), axis=(1, 2))
    total_finite_windows = int(np.sum(finite_rows))
    cond_windows = np.flatnonzero(finite_rows)
    max_w = int(max_windows_for_condition_number) if max_windows_for_condition_number is not None else 0
    subsampled = False
    if max_w > 0 and cond_windows.size > max_w:
        pick = np.linspace(0, cond_windows.size - 1, num=max_w, dtype=int)
        cond_windows = cond_windows[pick]
        subsampled = True

    if cond_windows.size > 0:
        try:
            svals = np.linalg.svd(J[cond_windows], compute_uv=False)  # [Ws, D]
            smin = np.min(svals, axis=1)
            smax = np.max(svals, axis=1)
            ok = np.isfinite(smin) & np.isfinite(smax) & (smin > 0)
            cond_vals = np.full(smin.shape, np.nan, dtype=float)
            cond_vals[ok] = smax[ok] / smin[ok]
            cond[cond_windows] = cond_vals
        except Exception:
            logger.exception("Failed vectorized SVD for Jacobian condition number")
    cond_desc = _distributional_descriptives(cond.reshape(-1, 1), ["kappa"])
    cond_desc = cond_desc.get("kappa", {}) if isinstance(cond_desc, dict) else {}
    if isinstance(cond_desc, dict):
        cond_desc["estimated_on_windows"] = int(cond_windows.size)
        cond_desc["total_finite_windows"] = total_finite_windows
        cond_desc["subsampled"] = bool(subsampled)

    # Rotation coherence (3D only): axis stability of antisymmetric component
    rot: Dict[str, Any] = {}
    if D == 3:
        omega = 0.5 * (J - np.transpose(J, (0, 2, 1)))
        wvec = np.stack([omega[:, 2, 1], omega[:, 0, 2], omega[:, 1, 0]], axis=1)  # [W,3]
        norms = np.linalg.norm(wvec, axis=1)
        mask = np.isfinite(norms) & (norms > 0)
        if np.any(mask):
            axes = (wvec[mask] / norms[mask, None]).astype(float)
            mean_axis = np.mean(axes, axis=0)
            mnorm = float(np.linalg.norm(mean_axis))
            rot = {
                "n": int(axes.shape[0]),
                "mean_resultant_length": mnorm,  # 0..1
                "mean_axis": [float(x) for x in (mean_axis / (mnorm if mnorm > 0 else 1.0))],
            }
        else:
            rot = {"n": 0, "mean_resultant_length": float("nan"), "mean_axis": [float("nan")] * 3}

    rel_mse_summary: Dict[str, Any] = {}
    if isinstance(jacobian_diagnostics, Mapping):
        rel_raw = jacobian_diagnostics.get("rel_mse_baseline_windows", None)
        if rel_raw is not None:
            rel_arr = np.asarray(rel_raw, dtype=float).reshape(-1)
            rel_desc = _distributional_descriptives(rel_arr.reshape(-1, 1), ["rel"])
            rel_desc = rel_desc.get("rel", {}) if isinstance(rel_desc, dict) else {}
            if isinstance(rel_desc, dict):
                rel_desc["note"] = "Relative local-fit MSE vs intercept-only baseline (<1 better than baseline)."
                rel_mse_summary = rel_desc
        elif "rel_mse_baseline_median" in jacobian_diagnostics:
            rel_mse_summary = {
                "median": float(jacobian_diagnostics.get("rel_mse_baseline_median")),
                "note": "Relative local-fit MSE vs intercept-only baseline (<1 better than baseline).",
            }

    return {
        "signed_divergence_balance": signed_div,
        "jacobian_condition_number": cond_desc,
        "rotation_coherence": rot,
        "rel_mse_baseline": rel_mse_summary,
    }


def _winsorize_matrix(
    values: np.ndarray,
    quantiles: tuple[float, float],
) -> np.ndarray:
    """Winsorize finite values column-wise without changing NaN support."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return np.asarray(arr, dtype=float)
    q_low, q_high = quantiles
    if not (0.0 <= q_low <= q_high <= 1.0):
        return np.asarray(arr, dtype=float)
    out = np.asarray(arr, dtype=float).copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        finite_mask = np.isfinite(col)
        if not np.any(finite_mask):
            continue
        finite = col[finite_mask]
        try:
            lo = float(np.quantile(finite, q_low))
            hi = float(np.quantile(finite, q_high))
        except Exception:
            continue
        col_out = col.copy()
        col_out[finite_mask] = np.clip(finite, lo, hi)
        out[:, j] = col_out
    return out


def _axis_degeneracy_summary(
    values: np.ndarray,
    names: Sequence[str],
    *,
    constant_var_tol: float,
    min_unique_values: int,
    rounding_decimals: int = 6,
) -> Dict[str, Dict[str, Any]]:
    """Summarize per-axis finite support and simple degeneracy checks."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(names):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for j, name in enumerate(names):
        col = arr[:, j]
        finite = col[np.isfinite(col)]
        nan_frac = float(1.0 - (finite.size / col.size if col.size else 1.0))
        if finite.size == 0:
            out[str(name)] = {
                "n": 0,
                "nan_frac": nan_frac,
                "var": float("nan"),
                "n_unique": 0,
                "all_nan": True,
                "constant": False,
                "degenerate": True,
            }
            continue
        var = float(np.var(finite))
        rounded = np.round(finite, int(max(0, rounding_decimals)))
        n_unique = int(np.unique(rounded).size)
        constant = bool(var <= float(constant_var_tol) or n_unique < int(max(1, min_unique_values)))
        out[str(name)] = {
            "n": int(finite.size),
            "nan_frac": nan_frac,
            "var": var,
            "n_unique": n_unique,
            "all_nan": False,
            "constant": constant,
            "degenerate": bool(constant),
        }
    return out


def _coordinate_space_validity(
    values: Optional[np.ndarray],
    names: Sequence[str],
    *,
    label: str,
    expected_dim: int,
    finite_row_warn_frac: float,
    constant_var_tol: float,
    min_unique_values: int,
) -> Dict[str, Any]:
    """Summarize shape, finite support, and per-axis degeneracy for one coordinate space."""
    if values is None:
        return {
            "label": label,
            "status": "not_available",
            "shape": None,
            "expected_dim": int(expected_dim),
            "warnings": [],
            "per_axis": {},
            "degenerate_axes": [],
            "all_nan_axes": [],
        }
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        return {
            "label": label,
            "status": "invalid_shape",
            "shape": list(arr.shape),
            "expected_dim": int(expected_dim),
            "warnings": [f"{label}: expected 2-D array, got shape {list(arr.shape)}"],
            "per_axis": {},
            "degenerate_axes": [],
            "all_nan_axes": [],
        }
    finite_rows = np.all(np.isfinite(arr), axis=1) if arr.shape[0] else np.zeros((0,), dtype=bool)
    finite_row_fraction = float(np.mean(finite_rows)) if finite_rows.size else 0.0
    per_axis = _axis_degeneracy_summary(
        arr,
        list(names),
        constant_var_tol=float(constant_var_tol),
        min_unique_values=int(min_unique_values),
    )
    degenerate_axes = sorted([name for name, info in per_axis.items() if bool(info.get("degenerate", False))])
    all_nan_axes = sorted([name for name, info in per_axis.items() if bool(info.get("all_nan", False))])
    warnings: list[str] = []
    if arr.shape[1] != int(expected_dim):
        warnings.append(f"{label}: expected dimension {expected_dim}, got {arr.shape[1]}")
    if len(names) != arr.shape[1]:
        warnings.append(f"{label}: coordinate names length does not match value columns")
    if finite_row_fraction < float(finite_row_warn_frac):
        warnings.append(
            f"{label}: finite row fraction {finite_row_fraction:.3f} below warn threshold {float(finite_row_warn_frac):.3f}"
        )
    if all_nan_axes:
        warnings.append(f"{label}: all-NaN axes present ({', '.join(all_nan_axes)})")
    elif degenerate_axes:
        warnings.append(f"{label}: degenerate axes present ({', '.join(degenerate_axes)})")
    return {
        "label": label,
        "status": "warning" if warnings else "ok",
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "expected_dim": int(expected_dim),
        "finite_row_fraction": finite_row_fraction,
        "names_match_shape": bool(len(names) == arr.shape[1]),
        "per_axis": per_axis,
        "degenerate_axes": degenerate_axes,
        "all_nan_axes": all_nan_axes,
        "warnings": warnings,
    }


def _jacobian_condition_distribution(
    jacobian: Optional[np.ndarray],
    *,
    ridge_floor: float = 0.0,
    max_windows_for_condition_number: int = 5000,
) -> Dict[str, Any]:
    """Compute distributional condition-number stats for a Jacobian stack."""
    if jacobian is None:
        return {}
    J = np.asarray(jacobian, dtype=float)
    if J.ndim != 3 or J.shape[0] == 0 or J.shape[1] != J.shape[2]:
        return {}
    finite_rows = np.all(np.isfinite(J), axis=(1, 2))
    cond_windows = np.flatnonzero(finite_rows)
    total_finite_windows = int(cond_windows.size)
    max_w = int(max_windows_for_condition_number) if max_windows_for_condition_number is not None else 0
    subsampled = False
    if max_w > 0 and cond_windows.size > max_w:
        pick = np.linspace(0, cond_windows.size - 1, num=max_w, dtype=int)
        cond_windows = cond_windows[pick]
        subsampled = True
    cond_vals = np.asarray([], dtype=float)
    if cond_windows.size > 0:
        try:
            svals = np.linalg.svd(J[cond_windows], compute_uv=False)
            smin = np.min(svals, axis=1)
            smax = np.max(svals, axis=1)
            if float(ridge_floor) > 0:
                denom = np.maximum(smin, float(ridge_floor))
                ok = np.isfinite(denom) & np.isfinite(smax) & (denom > 0)
                cond_vals = np.asarray(smax[ok] / denom[ok], dtype=float)
            else:
                ok = np.isfinite(smin) & np.isfinite(smax) & (smin > 0)
                cond_vals = np.asarray(smax[ok] / smin[ok], dtype=float)
        except Exception:
            logger.exception("Failed Jacobian SVD in MNPS/MNJ sanity helper")
            cond_vals = np.asarray([], dtype=float)
    out = {
        "n": int(cond_vals.size),
        "estimated_on_windows": int(cond_windows.size),
        "total_finite_windows": total_finite_windows,
        "subsampled": bool(subsampled),
        "ridge_floor": float(ridge_floor),
        "mean": float(np.mean(cond_vals)) if cond_vals.size else float("nan"),
        "median": float(np.median(cond_vals)) if cond_vals.size else float("nan"),
        "p95": float(np.percentile(cond_vals, 95)) if cond_vals.size else float("nan"),
        "max": float(np.max(cond_vals)) if cond_vals.size else float("nan"),
    }
    return out


def _per_window_jacobian_condition_numbers(jacobian: Optional[np.ndarray]) -> np.ndarray:
    """Compute per-window Jacobian condition numbers."""
    if jacobian is None:
        return np.zeros((0,), dtype=np.float64)
    J = np.asarray(jacobian, dtype=float)
    if J.ndim != 3 or J.shape[0] == 0 or J.shape[1] != J.shape[2]:
        return np.zeros((0,), dtype=np.float64)
    cond = np.full((J.shape[0],), np.nan, dtype=np.float64)
    finite_windows = np.all(np.isfinite(J), axis=(1, 2))
    if not np.any(finite_windows):
        return cond
    try:
        svals = np.linalg.svd(J[finite_windows], compute_uv=False)
        smin = np.min(svals, axis=1)
        smax = np.max(svals, axis=1)
        ok = np.isfinite(smin) & np.isfinite(smax) & (smin > 0)
        finite_cond = np.full((int(np.sum(finite_windows)),), np.nan, dtype=np.float64)
        finite_cond[ok] = np.asarray(smax[ok] / smin[ok], dtype=np.float64)
        cond[np.flatnonzero(finite_windows)] = finite_cond
    except Exception:
        logger.exception("Failed vectorized SVD for per-window Jacobian condition numbers")
    return cond


def _finite_vector_summary(values: np.ndarray) -> Dict[str, Any]:
    """Summarize a 1-D numeric vector over finite entries only."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "iqr": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    q25 = float(np.percentile(finite, 25))
    q75 = float(np.percentile(finite, 75))
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "iqr": float(q75 - q25),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def compute_window_time_audit(
    *,
    time: np.ndarray,
    window_start: np.ndarray,
    window_end: np.ndarray,
    dt_sec_runtime: float,
    dt_sec_config: float,
    window_sec_config: float,
) -> Dict[str, Any]:
    """Audit realized MNPS step size and window length from the exported time bounds."""
    time_arr = np.asarray(time, dtype=float).reshape(-1)
    start_arr = np.asarray(window_start, dtype=float).reshape(-1)
    end_arr = np.asarray(window_end, dtype=float).reshape(-1)
    n_time = int(time_arr.size)
    same_length = bool(start_arr.size == n_time and end_arr.size == n_time)

    finite_bounds_mask = (
        np.isfinite(start_arr) & np.isfinite(end_arr)
        if same_length
        else np.zeros((0,), dtype=bool)
    )
    finite_bounds_fraction = (
        float(np.mean(finite_bounds_mask))
        if finite_bounds_mask.size
        else 0.0
    )

    window_lengths = (
        end_arr[finite_bounds_mask] - start_arr[finite_bounds_mask]
        if finite_bounds_mask.size
        else np.asarray([], dtype=float)
    )
    nonpositive_window_len_count = int(np.sum(np.isfinite(window_lengths) & (window_lengths <= 0)))
    positive_window_lengths = window_lengths[np.isfinite(window_lengths) & (window_lengths > 0)]

    dt_source = "window_start"
    dt_intervals = np.asarray([], dtype=float)
    if same_length and start_arr.size > 1:
        start_diff = np.diff(start_arr)
        start_pair_mask = np.isfinite(start_arr[:-1]) & np.isfinite(start_arr[1:])
        dt_intervals = np.asarray(start_diff[start_pair_mask], dtype=float)
    if dt_intervals.size == 0 and time_arr.size > 1:
        dt_source = "time"
        time_diff = np.diff(time_arr)
        time_pair_mask = np.isfinite(time_arr[:-1]) & np.isfinite(time_arr[1:])
        dt_intervals = np.asarray(time_diff[time_pair_mask], dtype=float)

    nonpositive_dt_count = int(np.sum(np.isfinite(dt_intervals) & (dt_intervals <= 0)))
    positive_dt_intervals = dt_intervals[np.isfinite(dt_intervals) & (dt_intervals > 0)]

    dt_summary = _finite_vector_summary(positive_dt_intervals)
    window_len_summary = _finite_vector_summary(positive_window_lengths)
    dt_median = float(dt_summary.get("median", np.nan))
    window_len_median = float(window_len_summary.get("median", np.nan))

    runtime_tol = max(1e-6, 0.01 * abs(float(dt_sec_runtime))) if np.isfinite(dt_sec_runtime) else 1e-6
    config_dt_tol = max(1e-6, 0.01 * abs(float(dt_sec_config))) if np.isfinite(dt_sec_config) else 1e-6
    window_tol = max(1e-6, 0.01 * abs(float(window_sec_config))) if np.isfinite(window_sec_config) else 1e-6

    dt_matches_runtime = bool(
        np.isfinite(dt_median)
        and np.isfinite(dt_sec_runtime)
        and abs(dt_median - float(dt_sec_runtime)) <= runtime_tol
    )
    dt_matches_config = bool(
        np.isfinite(dt_median)
        and np.isfinite(dt_sec_config)
        and abs(dt_median - float(dt_sec_config)) <= config_dt_tol
    )
    window_len_matches_config = bool(
        np.isfinite(window_len_median)
        and np.isfinite(window_sec_config)
        and abs(window_len_median - float(window_sec_config)) <= window_tol
    )

    warnings: list[str] = []
    if not same_length:
        warnings.append("Time grid: time/window arrays have inconsistent lengths")
    if finite_bounds_mask.size and finite_bounds_fraction < 1.0:
        warnings.append(
            f"Time grid: finite window-bound fraction {finite_bounds_fraction:.3f} below 1.000"
        )
    if nonpositive_window_len_count > 0:
        warnings.append(
            f"Time grid: {nonpositive_window_len_count} non-positive window lengths detected"
        )
    if n_time > 1 and positive_dt_intervals.size == 0:
        warnings.append("Time grid: unable to recover positive inter-window dt values")
    elif n_time > 1 and not dt_matches_runtime:
        warnings.append(
            "Time grid: recovered dt disagrees with runtime dt used for derivatives/Jacobians"
        )
    if positive_window_lengths.size > 0 and not window_len_matches_config:
        warnings.append("Time grid: recovered window length disagrees with configured window_sec")

    return {
        "status": "warning" if warnings else "ok",
        "warnings": warnings,
        "window_count": int(n_time),
        "finite_window_bounds_fraction": finite_bounds_fraction,
        "dt_source": dt_source,
        "dt_sec_runtime": float(dt_sec_runtime) if np.isfinite(dt_sec_runtime) else float("nan"),
        "dt_sec_config": float(dt_sec_config) if np.isfinite(dt_sec_config) else float("nan"),
        "window_sec_config": float(window_sec_config) if np.isfinite(window_sec_config) else float("nan"),
        "dt_matches_runtime": dt_matches_runtime,
        "dt_matches_config_formula": dt_matches_config,
        "window_len_matches_config": window_len_matches_config,
        "nonpositive_dt_count": nonpositive_dt_count,
        "nonpositive_window_len_count": nonpositive_window_len_count,
        "dt_intervals_sec": dt_summary,
        "window_lengths_sec": window_len_summary,
    }


def _build_segment_edge_mask(
    n_samples: int,
    *,
    file_labels: Optional[Sequence[Any]] = None,
    edge_half_width: int = 1,
) -> np.ndarray:
    """Mark samples near segment boundaries where finite-difference checks are less stable."""
    edge_mask = np.zeros((int(max(0, n_samples)),), dtype=bool)
    if n_samples <= 0:
        return edge_mask
    width = max(1, int(edge_half_width))
    labels_arr = None
    if file_labels is not None:
        labels_arr = np.asarray(list(file_labels), dtype=object).reshape(-1)
        if labels_arr.size != n_samples:
            labels_arr = None

    segment_starts = [0]
    segment_ends: list[int] = []
    if labels_arr is not None and labels_arr.size == n_samples:
        for idx in range(1, n_samples):
            if labels_arr[idx] != labels_arr[idx - 1]:
                segment_ends.append(idx)
                segment_starts.append(idx)
        segment_ends.append(n_samples)
    else:
        segment_ends = [n_samples]

    for start, end in zip(segment_starts, segment_ends):
        if end <= start:
            continue
        local_width = min(width, end - start)
        edge_mask[start : start + local_width] = True
        edge_mask[end - local_width : end] = True
    return edge_mask


def compute_derivative_self_consistency(
    *,
    x: np.ndarray,
    x_dot: Optional[np.ndarray],
    time: Optional[np.ndarray],
    dt_sec: Optional[float],
    file_labels: Optional[Sequence[Any]] = None,
    derivative_cfg: Optional[Mapping[str, Any]] = None,
    finite_interval_warn_frac: float = 0.95,
    rel_error_warn: float = 1.0,
    speed_ratio_warn: float = 5.0,
    warn_bad_frac: float = 0.05,
) -> Dict[str, Any]:
    """Compare finite-difference MNPS velocity to the exported `mnps_3d_dot` field."""
    x_arr = np.asarray(x, dtype=float)
    x_dot_arr = np.asarray(x_dot, dtype=float) if x_dot is not None else None
    if (
        x_dot_arr is None
        or x_arr.ndim != 2
        or x_dot_arr.ndim != 2
        or x_arr.shape != x_dot_arr.shape
        or x_arr.shape[0] < 2
        or x_arr.shape[1] != 3
    ):
        return {
            "status": "not_available",
            "warnings": [],
            "intervals_total": max(0, int(x_arr.shape[0]) - 1) if x_arr.ndim == 2 else 0,
            "intervals_compared": 0,
        }

    n_samples = int(x_arr.shape[0])
    if time is not None:
        time_arr = np.asarray(time, dtype=float).reshape(-1)
    elif dt_sec is not None and np.isfinite(float(dt_sec)) and float(dt_sec) > 0:
        time_arr = np.arange(n_samples, dtype=float) * float(dt_sec)
    else:
        time_arr = np.asarray([], dtype=float)
    if time_arr.size != n_samples:
        if dt_sec is not None and np.isfinite(float(dt_sec)) and float(dt_sec) > 0:
            time_arr = np.arange(n_samples, dtype=float) * float(dt_sec)
        else:
            return {
                "status": "not_available",
                "warnings": ["Derivative QA: missing valid time base for finite-difference comparison"],
                "intervals_total": max(0, n_samples - 1),
                "intervals_compared": 0,
            }

    file_arr = None
    if file_labels is not None:
        file_arr = np.asarray(list(file_labels), dtype=object).reshape(-1)
        if file_arr.size != n_samples:
            file_arr = None
    same_file_mask = (
        np.asarray(file_arr[:-1] == file_arr[1:], dtype=bool)
        if file_arr is not None and n_samples > 1
        else np.ones((n_samples - 1,), dtype=bool)
    )

    dt_interval = np.diff(time_arr)
    dx = np.diff(x_arr, axis=0)
    fd_vec = np.full_like(dx, np.nan, dtype=float)
    valid_dt_mask = np.isfinite(dt_interval) & (dt_interval > 0)
    fd_valid_mask = valid_dt_mask & same_file_mask & np.all(np.isfinite(dx), axis=1)
    if np.any(fd_valid_mask):
        fd_vec[fd_valid_mask] = dx[fd_valid_mask] / dt_interval[fd_valid_mask, None]

    dot_mid = 0.5 * (x_dot_arr[:-1] + x_dot_arr[1:])
    compared_mask = fd_valid_mask & np.all(np.isfinite(dot_mid), axis=1)
    intervals_total = int(n_samples - 1)
    intervals_compared = int(np.sum(compared_mask))
    finite_interval_fraction = float(intervals_compared / max(1, intervals_total))
    if intervals_compared == 0:
        return {
            "status": "warning",
            "warnings": ["Derivative QA: no valid intervals available for self-consistency comparison"],
            "intervals_total": intervals_total,
            "intervals_compared": 0,
            "finite_interval_fraction": finite_interval_fraction,
        }

    fd_compared = fd_vec[compared_mask]
    dot_compared = dot_mid[compared_mask]
    fd_speed = np.linalg.norm(fd_compared, axis=1)
    dot_speed = np.linalg.norm(dot_compared, axis=1)
    diff_speed = np.linalg.norm(dot_compared - fd_compared, axis=1)
    rel_error = diff_speed / np.maximum(fd_speed, 1e-8)
    denom = dot_speed * fd_speed
    cosine = np.full((intervals_compared,), np.nan, dtype=float)
    cosine_ok = np.isfinite(denom) & (denom > 0)
    if np.any(cosine_ok):
        cosine[cosine_ok] = np.sum(dot_compared[cosine_ok] * fd_compared[cosine_ok], axis=1) / denom[cosine_ok]
    speed_ratio = np.ones((intervals_compared,), dtype=float)
    both_zero = (fd_speed <= 1e-8) & (dot_speed <= 1e-8)
    nonzero = ~both_zero
    if np.any(nonzero):
        speed_ratio[nonzero] = np.maximum(fd_speed[nonzero], dot_speed[nonzero]) / np.maximum(
            np.minimum(fd_speed[nonzero], dot_speed[nonzero]),
            1e-8,
        )

    deriv_window = 1
    if isinstance(derivative_cfg, Mapping):
        deriv_window = max(1, int(derivative_cfg.get("window", 3) or 3) // 2)
    edge_sample_mask = _build_segment_edge_mask(
        n_samples,
        file_labels=file_labels,
        edge_half_width=deriv_window,
    )
    edge_interval_mask = edge_sample_mask[:-1] | edge_sample_mask[1:]
    compared_edge_mask = edge_interval_mask[compared_mask]
    compared_interior_mask = ~compared_edge_mask
    bad_interval_mask = (rel_error > float(rel_error_warn)) | (speed_ratio > float(speed_ratio_warn))
    bad_fraction = float(np.mean(bad_interval_mask)) if bad_interval_mask.size else 0.0

    warnings: list[str] = []
    if finite_interval_fraction < float(finite_interval_warn_frac):
        warnings.append(
            f"Derivative QA: finite interval fraction {finite_interval_fraction:.3f} below warn threshold {float(finite_interval_warn_frac):.3f}"
        )
    rel_error_p95 = float(np.percentile(rel_error, 95)) if rel_error.size else float("nan")
    if np.isfinite(rel_error_p95) and rel_error_p95 > float(rel_error_warn):
        warnings.append(
            f"Derivative QA: vector relative error p95 {rel_error_p95:.3g} exceeded warn threshold {float(rel_error_warn):.3g}"
        )
    speed_ratio_p95 = float(np.percentile(speed_ratio, 95)) if speed_ratio.size else float("nan")
    if np.isfinite(speed_ratio_p95) and speed_ratio_p95 > float(speed_ratio_warn):
        warnings.append(
            f"Derivative QA: symmetric speed ratio p95 {speed_ratio_p95:.3g} exceeded warn threshold {float(speed_ratio_warn):.3g}"
        )
    if bad_fraction > float(warn_bad_frac):
        warnings.append(
            f"Derivative QA: bad interval fraction {bad_fraction:.3f} exceeded warn threshold {float(warn_bad_frac):.3f}"
        )

    return {
        "status": "warning" if warnings else "ok",
        "warnings": warnings,
        "intervals_total": intervals_total,
        "intervals_compared": intervals_compared,
        "finite_interval_fraction": finite_interval_fraction,
        "dt_sec_runtime": float(dt_sec) if dt_sec is not None and np.isfinite(float(dt_sec)) else float("nan"),
        "interval_dt_sec": _finite_vector_summary(dt_interval[compared_mask]),
        "fd_speed": _finite_vector_summary(fd_speed),
        "mnps_3d_dot_speed": _finite_vector_summary(dot_speed),
        "vector_rel_error": _finite_vector_summary(rel_error),
        "speed_ratio_symmetric": _finite_vector_summary(speed_ratio),
        "cosine_similarity": _finite_vector_summary(cosine),
        "bad_interval_fraction": bad_fraction,
        "bad_interval_count": int(np.sum(bad_interval_mask)),
        "edge_interval_fraction": float(np.mean(compared_edge_mask)) if compared_edge_mask.size else 0.0,
        "edge_vector_rel_error": _finite_vector_summary(rel_error[compared_edge_mask]),
        "interior_vector_rel_error": _finite_vector_summary(rel_error[compared_interior_mask]),
        "same_file_interval_fraction": float(np.mean(same_file_mask)) if same_file_mask.size else 1.0,
        "edge_half_width_samples": int(deriv_window),
    }


def compute_standard_geometry_contract(
    *,
    x: np.ndarray,
    coords_9d: Optional[np.ndarray],
    coords_9d_names: Sequence[str],
    primary_requires_coords_9d: bool = False,
) -> Dict[str, Any]:
    """Evaluate the always-on mathematical invalidity contract for exported geometry."""
    x_arr = np.asarray(x, dtype=float)
    n_rows = int(x_arr.shape[0]) if x_arr.ndim == 2 else 0
    x_names = ["m", "d", "e"]
    mnps_validity = _coordinate_space_validity(
        x_arr,
        x_names,
        label="mnps_3d",
        expected_dim=3,
        finite_row_warn_frac=0.0,
        constant_var_tol=STANDARD_GEOMETRY_CONSTANT_VAR_TOL,
        min_unique_values=STANDARD_GEOMETRY_MIN_UNIQUE_VALUES,
    )
    coords_validity = _coordinate_space_validity(
        coords_9d,
        list(coords_9d_names),
        label="coords_9d",
        expected_dim=9,
        finite_row_warn_frac=0.0,
        constant_var_tol=STANDARD_GEOMETRY_CONSTANT_VAR_TOL,
        min_unique_values=STANDARD_GEOMETRY_MIN_UNIQUE_VALUES,
    )

    keep_mask = np.ones((n_rows,), dtype=bool)
    mnps_nonfinite_mask = np.zeros((n_rows,), dtype=bool)
    coords_nonfinite_mask = np.zeros((n_rows,), dtype=bool)
    coords_shape_matches_time = False

    if x_arr.ndim == 2 and x_arr.shape[0] == n_rows:
        mnps_nonfinite_mask = ~np.all(np.isfinite(x_arr), axis=1)
        keep_mask &= ~mnps_nonfinite_mask

    if coords_9d is not None:
        coords_arr = np.asarray(coords_9d, dtype=float)
        if coords_arr.ndim == 2 and coords_arr.shape[0] == n_rows:
            coords_shape_matches_time = True
            coords_nonfinite_mask = ~np.all(np.isfinite(coords_arr), axis=1)
            if bool(primary_requires_coords_9d):
                keep_mask &= ~coords_nonfinite_mask

    dropped_mask = ~keep_mask
    drop_reason_counts = {
        "mnps_3d_nonfinite_rows": int(np.sum(mnps_nonfinite_mask)),
        "coords_9d_nonfinite_rows_affecting_primary": int(np.sum(coords_nonfinite_mask)) if bool(primary_requires_coords_9d) else 0,
    }
    coords_nonfinite_retained = int(np.sum(coords_nonfinite_mask & keep_mask)) if coords_shape_matches_time else 0
    issues_present = bool(
        int(np.sum(dropped_mask)) > 0
        or bool(mnps_validity.get("degenerate_axes"))
        or bool(coords_validity.get("degenerate_axes"))
        or coords_nonfinite_retained > 0
    )

    return {
        "policy_version": STANDARD_GEOMETRY_POLICY_VERSION,
        "status": "adjusted" if issues_present else "ok",
        "primary_requires_coords_9d": bool(primary_requires_coords_9d),
        "shared_time_grid": {
            "epochs_before_policy": int(n_rows),
            "epochs_retained": int(np.sum(keep_mask)),
            "epochs_dropped": int(np.sum(dropped_mask)),
            "drop_fraction": float(np.mean(dropped_mask)) if dropped_mask.size else 0.0,
            "dropped_epoch_indices_preview": [int(idx) for idx in np.flatnonzero(dropped_mask)[:32]],
            "drop_reason_counts": drop_reason_counts,
        },
        "mnps_3d": {
            "finite_row_fraction_before": float(np.mean(~mnps_nonfinite_mask)) if mnps_nonfinite_mask.size else 0.0,
            "nonfinite_row_count": int(np.sum(mnps_nonfinite_mask)),
            "degenerate_axes": list(mnps_validity.get("degenerate_axes", [])),
            "all_nan_axes": list(mnps_validity.get("all_nan_axes", [])),
        },
        "coords_9d": {
            "available": bool(coords_9d is not None),
            "shape_matches_time_grid": bool(coords_shape_matches_time),
            "finite_row_fraction_before": (
                float(np.mean(~coords_nonfinite_mask))
                if coords_shape_matches_time and coords_nonfinite_mask.size
                else float("nan")
            ),
            "nonfinite_row_count": int(np.sum(coords_nonfinite_mask)) if coords_shape_matches_time else 0,
            "nonfinite_rows_retained_on_shared_grid": int(coords_nonfinite_retained),
            "degenerate_axes": list(coords_validity.get("degenerate_axes", [])),
            "all_nan_axes": list(coords_validity.get("all_nan_axes", [])),
        },
        "_row_keep_mask": keep_mask,
    }


def apply_standard_jacobian_window_policy(
    jacobian_result: Optional[JacobianResult],
    *,
    condition_number_max: float = STANDARD_GEOMETRY_MAX_JACOBIAN_CONDITION,
) -> tuple[Optional[JacobianResult], Dict[str, Any]]:
    """Filter mathematically invalid Jacobian windows from canonical exports."""
    if jacobian_result is None:
        return None, {
            "policy_version": STANDARD_GEOMETRY_POLICY_VERSION,
            "status": "not_available",
            "windows_raw": 0,
            "windows_retained": 0,
            "invalid_windows": 0,
            "condition_number_threshold": float(condition_number_max),
            "invalid_reason_counts": {},
        }

    j_hat = np.asarray(jacobian_result.j_hat, dtype=np.float32)
    centers = np.asarray(jacobian_result.centers, dtype=np.int32)
    diagnostics = dict(jacobian_result.diagnostics or {})
    if j_hat.ndim != 3 or j_hat.shape[0] == 0:
        diagnostics.setdefault("condition_number_windows", np.zeros((0,), dtype=np.float64))
        return JacobianResult(
            j_hat=j_hat,
            j_dot=np.asarray(jacobian_result.j_dot, dtype=np.float32),
            centers=centers,
            diagnostics=diagnostics,
        ), {
            "policy_version": STANDARD_GEOMETRY_POLICY_VERSION,
            "status": "not_available",
            "windows_raw": int(j_hat.shape[0]) if j_hat.ndim == 3 else 0,
            "windows_retained": int(j_hat.shape[0]) if j_hat.ndim == 3 else 0,
            "invalid_windows": 0,
            "condition_number_threshold": float(condition_number_max),
            "invalid_reason_counts": {},
        }

    cond_raw = diagnostics.get("condition_number_windows")
    cond_arr = np.asarray(cond_raw, dtype=np.float64).reshape(-1) if cond_raw is not None else np.zeros((0,), dtype=np.float64)
    if cond_arr.shape[0] != j_hat.shape[0]:
        cond_arr = _per_window_jacobian_condition_numbers(j_hat)

    nonfinite_mask = ~np.all(np.isfinite(j_hat), axis=(1, 2))
    cond_nonfinite_mask = ~np.isfinite(cond_arr)
    cond_too_high_mask = np.isfinite(cond_arr) & (cond_arr >= float(condition_number_max))
    invalid_mask = nonfinite_mask | cond_nonfinite_mask | cond_too_high_mask
    valid_mask = ~invalid_mask

    filtered_diagnostics: Dict[str, Any] = {}
    for key, value in diagnostics.items():
        if isinstance(value, np.ndarray):
            arr = np.asarray(value)
            filtered_diagnostics[key] = arr[valid_mask] if arr.ndim >= 1 and arr.shape[0] == valid_mask.size else arr
            continue
        if isinstance(value, (list, tuple)):
            arr = np.asarray(value)
            filtered_diagnostics[key] = arr[valid_mask] if arr.ndim >= 1 and arr.shape[0] == valid_mask.size else arr
            continue
        filtered_diagnostics[key] = value

    filtered_j_hat = j_hat[valid_mask]
    spacing = float(filtered_diagnostics.get("j_dot_dt", diagnostics.get("j_dot_dt", 1.0)) or 1.0)
    if filtered_j_hat.shape[0] > 1:
        filtered_j_dot = np.gradient(filtered_j_hat, spacing, axis=0).astype(np.float32)
    else:
        filtered_j_dot = np.zeros_like(filtered_j_hat, dtype=np.float32)

    filtered_diagnostics["windows_raw"] = float(j_hat.shape[0])
    filtered_diagnostics["windows"] = float(filtered_j_hat.shape[0])
    filtered_diagnostics["condition_number_windows"] = cond_arr[valid_mask].astype(np.float64)
    filtered_diagnostics["hard_invalid_window_mask"] = invalid_mask.astype(np.int8)
    filtered_diagnostics["hard_invalid_centers"] = centers[invalid_mask].astype(np.int32)
    filtered_diagnostics["hard_invalid_windows"] = float(np.sum(invalid_mask))
    filtered_diagnostics["hard_invalid_nonfinite_windows"] = float(np.sum(nonfinite_mask))
    filtered_diagnostics["hard_invalid_condition_number_windows"] = float(np.sum(cond_too_high_mask | cond_nonfinite_mask))
    filtered_diagnostics["hard_invalid_condition_number_threshold"] = float(condition_number_max)

    filtered_result = JacobianResult(
        j_hat=filtered_j_hat.astype(np.float32),
        j_dot=filtered_j_dot,
        centers=centers[valid_mask].astype(np.int32),
        diagnostics=filtered_diagnostics,
    )
    invalid_window_count = int(np.sum(invalid_mask))
    invalid_centers = centers[invalid_mask]
    return filtered_result, {
        "policy_version": STANDARD_GEOMETRY_POLICY_VERSION,
        "status": "adjusted" if invalid_window_count > 0 else "ok",
        "windows_raw": int(j_hat.shape[0]),
        "windows_retained": int(filtered_j_hat.shape[0]),
        "invalid_windows": invalid_window_count,
        "invalid_window_fraction": float(np.mean(invalid_mask)) if invalid_mask.size else 0.0,
        "condition_number_threshold": float(condition_number_max),
        "invalid_reason_counts": {
            "nonfinite_windows": int(np.sum(nonfinite_mask)),
            "condition_number_windows": int(np.sum(cond_too_high_mask | cond_nonfinite_mask)),
        },
        "invalid_centers_preview": [int(idx) for idx in invalid_centers[:32]],
    }


def apply_anchor_coupling_window_policy(
    coupling_export: Optional[Mapping[str, Any]],
    *,
    condition_number_max: float = STANDARD_GEOMETRY_MAX_JACOBIAN_CONDITION,
    min_windows: int = 3,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Filter additive anchor-coupling windows using the standard condition policy."""
    if not isinstance(coupling_export, Mapping) or not coupling_export:
        return {}, {
            "policy_version": STANDARD_GEOMETRY_POLICY_VERSION,
            "status": "not_available",
            "windows_raw": 0,
            "windows_retained": 0,
            "invalid_windows": 0,
            "condition_number_threshold": float(condition_number_max),
            "min_windows_required": int(min_windows),
        }

    j_z = np.asarray(coupling_export.get("J_z"), dtype=np.float32)
    centers = np.asarray(coupling_export.get("centers", []), dtype=np.int32)
    diagnostics = dict(coupling_export.get("diagnostics", {}) or {})
    if j_z.ndim != 3 or j_z.shape[0] == 0:
        diagnostics.setdefault("condition_number_windows", np.zeros((0,), dtype=np.float64))
        filtered = dict(coupling_export)
        filtered["diagnostics"] = diagnostics
        return filtered, {
            "policy_version": STANDARD_GEOMETRY_POLICY_VERSION,
            "status": "not_available",
            "windows_raw": int(j_z.shape[0]) if j_z.ndim == 3 else 0,
            "windows_retained": int(j_z.shape[0]) if j_z.ndim == 3 else 0,
            "invalid_windows": 0,
            "condition_number_threshold": float(condition_number_max),
            "min_windows_required": int(min_windows),
        }

    cond_raw = diagnostics.get("condition_number_windows")
    cond_arr = np.asarray(cond_raw, dtype=np.float64).reshape(-1) if cond_raw is not None else np.zeros((0,), dtype=np.float64)
    if cond_arr.shape[0] != j_z.shape[0]:
        cond_arr = _per_window_jacobian_condition_numbers(j_z)

    nonfinite_mask = ~np.all(np.isfinite(j_z), axis=(1, 2))
    cond_nonfinite_mask = ~np.isfinite(cond_arr)
    cond_too_high_mask = np.isfinite(cond_arr) & (cond_arr >= float(condition_number_max))
    invalid_mask = nonfinite_mask | cond_nonfinite_mask | cond_too_high_mask
    valid_mask = ~invalid_mask

    filtered: Dict[str, Any] = {}
    for key, value in coupling_export.items():
        if key == "diagnostics":
            continue
        arr = None
        try:
            arr = np.asarray(value)
        except Exception:
            arr = None
        if arr is not None and arr.ndim >= 1 and arr.shape[0] == valid_mask.size:
            filtered[key] = arr[valid_mask]
        else:
            filtered[key] = value

    filtered_diagnostics: Dict[str, Any] = {}
    for key, value in diagnostics.items():
        arr = None
        if isinstance(value, (np.ndarray, list, tuple)):
            try:
                arr = np.asarray(value)
            except Exception:
                arr = None
        if arr is not None and arr.ndim >= 1 and arr.shape[0] == valid_mask.size:
            filtered_diagnostics[key] = arr[valid_mask]
        else:
            filtered_diagnostics[key] = value
    filtered_diagnostics["windows_raw"] = float(j_z.shape[0])
    filtered_diagnostics["windows"] = float(np.sum(valid_mask))
    filtered_diagnostics["condition_number_windows"] = cond_arr[valid_mask].astype(np.float64)
    filtered_diagnostics["hard_invalid_window_mask"] = invalid_mask.astype(np.int8)
    filtered_diagnostics["hard_invalid_centers"] = centers[invalid_mask].astype(np.int32)
    filtered_diagnostics["hard_invalid_condition_number_threshold"] = float(condition_number_max)
    filtered_diagnostics["min_windows_required"] = int(min_windows)
    filtered["diagnostics"] = filtered_diagnostics

    retained = int(np.sum(valid_mask))
    status = "adjusted" if int(np.sum(invalid_mask)) > 0 else "ok"
    if retained < int(min_windows):
        filtered = {}
        status = "insufficient_windows"
    return filtered, {
        "policy_version": STANDARD_GEOMETRY_POLICY_VERSION,
        "status": status,
        "windows_raw": int(j_z.shape[0]),
        "windows_retained": retained,
        "invalid_windows": int(np.sum(invalid_mask)),
        "invalid_window_fraction": float(np.mean(invalid_mask)) if invalid_mask.size else 0.0,
        "condition_number_threshold": float(condition_number_max),
        "min_windows_required": int(min_windows),
    }


def compute_mnps_mnj_sanity(
    *,
    x: np.ndarray,
    x_dot: Optional[np.ndarray],
    time: Optional[np.ndarray],
    dt_sec: Optional[float],
    coords_9d: Optional[np.ndarray],
    coords_9d_names: Sequence[str],
    jacobian: Optional[np.ndarray],
    jacobian_diagnostics: Optional[Mapping[str, Any]],
    review_qc_cfg: Optional[Mapping[str, Any]] = None,
    projection_contract: Optional[Mapping[str, Any]] = None,
    file_labels: Optional[Sequence[Any]] = None,
    derivative_cfg: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Compute reviewer-facing sanity checks for MNPS 3D/9D and MNJ."""
    cfg = review_qc_cfg or {}
    sanity_cfg = cfg.get("mnps_mnj_sanity", {}) if isinstance(cfg, Mapping) else {}
    if not isinstance(sanity_cfg, Mapping) or not sanity_cfg.get("enabled", False):
        return None

    coord_spaces = sanity_cfg.get("coord_spaces", ["mnps_3d", "coords_9d"])
    selected_spaces = {str(name).strip().lower() for name in coord_spaces if str(name).strip()}
    if not selected_spaces:
        selected_spaces = {"mnps_3d", "coords_9d"}
    finite_row_warn_frac = float(sanity_cfg.get("finite_row_warn_frac", 0.95) or 0.95)
    constant_var_tol = float(sanity_cfg.get("constant_var_tol", 1e-8) or 1e-8)
    min_unique_values = int(sanity_cfg.get("min_unique_values", 3) or 3)
    cond_warn = float(sanity_cfg.get("cond_warn", 1e6) or 1e6)
    rel_mse_warn = float(sanity_cfg.get("rel_mse_warn", 1.0) or 1.0)
    derivative_rel_error_warn = float(sanity_cfg.get("derivative_rel_error_warn", 1.0) or 1.0)
    derivative_speed_ratio_warn = float(sanity_cfg.get("derivative_speed_ratio_warn", 5.0) or 5.0)
    derivative_bad_frac_warn = float(sanity_cfg.get("derivative_bad_frac_warn", 0.05) or 0.05)
    robust_cfg = sanity_cfg.get("robustified_variant", {}) if isinstance(sanity_cfg, Mapping) else {}
    robust_cfg = robust_cfg if isinstance(robust_cfg, Mapping) else {}
    robust_enabled = bool(robust_cfg.get("enabled", True))
    winsorize_quantiles = tuple(robust_cfg.get("winsorize_quantiles", [0.01, 0.99]) or [0.01, 0.99])
    if len(winsorize_quantiles) != 2:
        winsorize_quantiles = (0.01, 0.99)
    winsorize_quantiles = (float(winsorize_quantiles[0]), float(winsorize_quantiles[1]))
    jacobian_ridge_floor = float(robust_cfg.get("jacobian_ridge_floor", 1e-6) or 1e-6)

    x_names = ["m", "d", "e"]
    coordinate_validity: Dict[str, Any] = {}
    if "mnps_3d" in selected_spaces:
        coordinate_validity["mnps_3d"] = _coordinate_space_validity(
            x,
            x_names,
            label="mnps_3d",
            expected_dim=3,
            finite_row_warn_frac=finite_row_warn_frac,
            constant_var_tol=constant_var_tol,
            min_unique_values=min_unique_values,
        )
    if "coords_9d" in selected_spaces:
        coordinate_validity["coords_9d"] = _coordinate_space_validity(
            coords_9d,
            list(coords_9d_names),
            label="coords_9d",
            expected_dim=9,
            finite_row_warn_frac=finite_row_warn_frac,
            constant_var_tol=constant_var_tol,
            min_unique_values=min_unique_values,
        )

    subcoordinate_degeneracy = (
        (coordinate_validity.get("coords_9d") or {}).get("per_axis", {}) if "coords_9d" in coordinate_validity else {}
    )
    entropy_axes = {"e_e", "e_s", "e_m"}
    entropy_family_degenerate = bool(
        any(bool((subcoordinate_degeneracy.get(name) or {}).get("degenerate", False)) for name in entropy_axes)
    )

    jacobian_arr = np.asarray(jacobian, dtype=float) if jacobian is not None else None
    jacobian_validity: Dict[str, Any]
    if jacobian_arr is None or jacobian_arr.ndim != 3 or jacobian_arr.shape[0] == 0:
        jacobian_validity = {
            "status": "not_available",
            "warnings": [],
            "finite_window_fraction": float("nan"),
            "failed_windows": float("nan"),
            "condition_number": {},
            "rel_mse_baseline": {},
        }
    else:
        finite_window_mask = np.all(np.isfinite(jacobian_arr), axis=(1, 2))
        finite_window_fraction = float(np.mean(finite_window_mask)) if finite_window_mask.size else 0.0
        condition_number = _jacobian_condition_distribution(jacobian_arr, ridge_floor=0.0)
        rel_summary = (compute_tier2_jacobian_metrics(jacobian_arr, jacobian_diagnostics) or {}).get(
            "rel_mse_baseline", {}
        )
        failed_windows = float((jacobian_diagnostics or {}).get("failed", np.nan)) if isinstance(jacobian_diagnostics, Mapping) else float("nan")
        jac_warnings: list[str] = []
        if finite_window_fraction < finite_row_warn_frac:
            jac_warnings.append(
                f"MNJ: finite window fraction {finite_window_fraction:.3f} below warn threshold {finite_row_warn_frac:.3f}"
            )
        cond_p95 = float(condition_number.get("p95", np.nan))
        cond_max = float(condition_number.get("max", np.nan))
        if (np.isfinite(cond_p95) and cond_p95 >= cond_warn) or (np.isfinite(cond_max) and cond_max >= cond_warn):
            jac_warnings.append(
                f"MNJ: condition number exceeded warn threshold {cond_warn:.3g} (p95={cond_p95:.3g}, max={cond_max:.3g})"
            )
        rel_mse_median = float((rel_summary or {}).get("median", np.nan))
        if np.isfinite(rel_mse_median) and rel_mse_median > rel_mse_warn:
            jac_warnings.append(
                f"MNJ: rel_mse_baseline median {rel_mse_median:.3g} exceeded warn threshold {rel_mse_warn:.3g}"
            )
        jacobian_validity = {
            "status": "warning" if jac_warnings else "ok",
            "warnings": jac_warnings,
            "finite_window_fraction": finite_window_fraction,
            "failed_windows": failed_windows,
            "condition_number": condition_number,
            "rel_mse_baseline": rel_summary,
        }

    derivative_self_consistency = compute_derivative_self_consistency(
        x=x,
        x_dot=x_dot,
        time=time,
        dt_sec=dt_sec,
        file_labels=file_labels,
        derivative_cfg=derivative_cfg,
        finite_interval_warn_frac=finite_row_warn_frac,
        rel_error_warn=derivative_rel_error_warn,
        speed_ratio_warn=derivative_speed_ratio_warn,
        warn_bad_frac=derivative_bad_frac_warn,
    )

    projection_info = dict(projection_contract or {}) if isinstance(projection_contract, Mapping) else {}
    from_v2 = projection_info.get("from_v2", {}) if isinstance(projection_info.get("from_v2", {}), Mapping) else {}
    coords_9d_name_list = [str(name) for name in coords_9d_names]
    family_counts = {
        "m": int(sum(str(name).startswith("m_") for name in coords_9d_name_list)),
        "d": int(sum(str(name).startswith("d_") for name in coords_9d_name_list)),
        "e": int(sum(str(name).startswith("e_") for name in coords_9d_name_list)),
    }
    family_layout_ok = bool(
        not coords_9d_name_list or (len(coords_9d_name_list) == 9 and family_counts == {"m": 3, "d": 3, "e": 3})
    )
    projection_warnings: list[str] = []
    if coords_9d_name_list and len(set(coords_9d_name_list)) != len(coords_9d_name_list):
        projection_warnings.append("Projection contract: duplicate coords_9d names detected")
    if coords_9d_name_list and not family_layout_ok:
        projection_warnings.append("Projection contract: coords_9d family layout deviates from expected 3/3/3 split")
    fallback_reason = from_v2.get("fallback_reason")
    if isinstance(fallback_reason, str) and fallback_reason.strip():
        projection_warnings.append(f"Projection contract: mnps_3d from_v2 fallback reason = {fallback_reason}")
    projection_contract_summary = {
        "status": "warning" if projection_warnings else "ok",
        "mode_requested": projection_info.get("mode_requested"),
        "mode_effective": projection_info.get("mode_effective"),
        "x_definition": projection_info.get("x_definition"),
        "from_v2": from_v2,
        "coords_9d_names_unique": bool(len(set(coords_9d_name_list)) == len(coords_9d_name_list)),
        "coords_9d_family_counts": family_counts,
        "coords_9d_expected_family_layout_ok": family_layout_ok,
        "warnings": projection_warnings,
    }

    degeneracy_flags = {
        "mnps_3d_low_finite_fraction": bool(
            float((coordinate_validity.get("mnps_3d") or {}).get("finite_row_fraction", 1.0)) < finite_row_warn_frac
        )
        if "mnps_3d" in coordinate_validity
        else False,
        "mnps_3d_has_degenerate_axis": bool((coordinate_validity.get("mnps_3d") or {}).get("degenerate_axes"))
        if "mnps_3d" in coordinate_validity
        else False,
        "coords_9d_low_finite_fraction": bool(
            float((coordinate_validity.get("coords_9d") or {}).get("finite_row_fraction", 1.0)) < finite_row_warn_frac
        )
        if "coords_9d" in coordinate_validity
        else False,
        "coords_9d_has_degenerate_subcoord": bool((coordinate_validity.get("coords_9d") or {}).get("degenerate_axes"))
        if "coords_9d" in coordinate_validity
        else False,
        "coords_9d_has_all_nan_subcoord": bool((coordinate_validity.get("coords_9d") or {}).get("all_nan_axes"))
        if "coords_9d" in coordinate_validity
        else False,
        "coords_9d_entropy_family_degenerate": entropy_family_degenerate,
        "mnj_low_finite_window_fraction": bool(
            np.isfinite(jacobian_validity.get("finite_window_fraction", np.nan))
            and float(jacobian_validity.get("finite_window_fraction", np.nan)) < finite_row_warn_frac
        ),
        "mnj_high_condition_number": bool(
            (
                np.isfinite(float((jacobian_validity.get("condition_number") or {}).get("p95", np.nan)))
                and float((jacobian_validity.get("condition_number") or {}).get("p95", np.nan)) >= cond_warn
            )
            or (
                np.isfinite(float((jacobian_validity.get("condition_number") or {}).get("max", np.nan)))
                and float((jacobian_validity.get("condition_number") or {}).get("max", np.nan)) >= cond_warn
            )
        ),
        "mnj_rel_mse_degraded": bool(
            np.isfinite(float((jacobian_validity.get("rel_mse_baseline") or {}).get("median", np.nan)))
            and float((jacobian_validity.get("rel_mse_baseline") or {}).get("median", np.nan)) > rel_mse_warn
        ),
        "mnps_3d_dot_inconsistent": bool(derivative_self_consistency.get("status") == "warning"),
        "projection_contract_warning": bool(projection_warnings),
    }
    degeneracy_flags["combined_geometry_instability"] = bool(
        degeneracy_flags["coords_9d_has_degenerate_subcoord"]
        and (degeneracy_flags["mnj_high_condition_number"] or degeneracy_flags["mnj_rel_mse_degraded"])
    )

    robustified_comparison: Dict[str, Any] = {"enabled": bool(robust_enabled)}
    if robust_enabled:
        robust_spaces: Dict[str, Any] = {}
        if "mnps_3d" in coordinate_validity:
            x_wins = _winsorize_matrix(np.asarray(x, dtype=float), winsorize_quantiles)
            robust_spaces["mnps_3d"] = {
                "winsorize_quantiles": [float(winsorize_quantiles[0]), float(winsorize_quantiles[1])],
                "distribution": _distributional_descriptives(x_wins, x_names),
                "per_axis": _axis_degeneracy_summary(
                    x_wins,
                    x_names,
                    constant_var_tol=float(constant_var_tol),
                    min_unique_values=int(min_unique_values),
                ),
            }
        if "coords_9d" in coordinate_validity and coords_9d is not None and coords_9d_names:
            coords_wins = _winsorize_matrix(np.asarray(coords_9d, dtype=float), winsorize_quantiles)
            robust_spaces["coords_9d"] = {
                "winsorize_quantiles": [float(winsorize_quantiles[0]), float(winsorize_quantiles[1])],
                "distribution": _distributional_descriptives(coords_wins, list(coords_9d_names)),
                "per_axis": _axis_degeneracy_summary(
                    coords_wins,
                    list(coords_9d_names),
                    constant_var_tol=float(constant_var_tol),
                    min_unique_values=int(min_unique_values),
                ),
            }
        robust_cond = _jacobian_condition_distribution(
            jacobian_arr,
            ridge_floor=float(jacobian_ridge_floor),
        )
        robustified_comparison.update(
            {
                "coordinate_spaces": robust_spaces,
                "jacobian_condition_number": robust_cond,
                "warning_deltas": {
                    "coords_9d_degenerate_raw_count": int(len((coordinate_validity.get("coords_9d") or {}).get("degenerate_axes", [])))
                    if "coords_9d" in coordinate_validity
                    else 0,
                    "coords_9d_degenerate_robust_count": int(
                        sum(
                            bool(info.get("degenerate", False))
                            for info in (((robust_spaces.get("coords_9d") or {}).get("per_axis", {})).values())
                        )
                    )
                    if "coords_9d" in robust_spaces
                    else 0,
                    "mnps_3d_degenerate_raw_count": int(len((coordinate_validity.get("mnps_3d") or {}).get("degenerate_axes", [])))
                    if "mnps_3d" in coordinate_validity
                    else 0,
                    "mnps_3d_degenerate_robust_count": int(
                        sum(
                            bool(info.get("degenerate", False))
                            for info in (((robust_spaces.get("mnps_3d") or {}).get("per_axis", {})).values())
                        )
                    )
                    if "mnps_3d" in robust_spaces
                    else 0,
                    "mnj_condition_warning_raw": bool(degeneracy_flags["mnj_high_condition_number"]),
                    "mnj_condition_warning_robust": bool(
                        (
                            np.isfinite(float((robust_cond or {}).get("p95", np.nan)))
                            and float((robust_cond or {}).get("p95", np.nan)) >= cond_warn
                        )
                        or (
                            np.isfinite(float((robust_cond or {}).get("max", np.nan)))
                            and float((robust_cond or {}).get("max", np.nan)) >= cond_warn
                        )
                    ),
                },
            }
        )

    warnings: list[str] = []
    for block in coordinate_validity.values():
        warnings.extend(list(block.get("warnings", [])))
    warnings.extend(list(jacobian_validity.get("warnings", [])))
    warnings.extend(list(derivative_self_consistency.get("warnings", [])))
    warnings.extend(projection_warnings)
    warnings = sorted(dict.fromkeys([str(w) for w in warnings if str(w).strip()]))
    status = "warning" if warnings else "ok"
    return {
        "status": status,
        "warnings": warnings,
        "projection_contract": projection_contract_summary,
        "coordinate_validity": coordinate_validity,
        "jacobian_validity": jacobian_validity,
        "derivative_self_consistency": derivative_self_consistency,
        "subcoordinate_degeneracy": subcoordinate_degeneracy,
        "degeneracy_flags": degeneracy_flags,
        "robustified_comparison": robustified_comparison,
    }


def compute_emmi_metrics(x: np.ndarray, x_dot: np.ndarray) -> Dict[str, float]:
    """Tier-2 derived indices (control/sensitivity proxies) from MNPS + speed."""
    X = np.asarray(x, dtype=float)
    Xd = np.asarray(x_dot, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3 or Xd.ndim != 2 or Xd.shape[1] != 3:
        return {}

    m = X[:, 0]
    d = X[:, 1]
    e = X[:, 2]
    speed = np.linalg.norm(Xd, axis=1)

    def _finite_median(arr: np.ndarray) -> float:
        """Internal helper: finite median."""
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        return float(np.median(a)) if a.size else float("nan")

    def _finite_mean(arr: np.ndarray) -> float:
        """Internal helper: finite mean."""
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        return float(np.mean(a)) if a.size else float("nan")

    m_med = _finite_median(m)
    d_med = _finite_median(d)
    e_med = _finite_median(e)
    sp_med = _finite_median(speed)
    sp_mean = _finite_mean(speed)

    mv_med = float("nan")
    mv_abs_med = float("nan")
    if np.isfinite(m_med) and np.isfinite(d_med) and np.isfinite(e_med):
        mv_med = float(m_med * d_med * e_med)
        mv_abs_med = float(abs(m_med) * abs(d_med) * abs(e_med))

    def _safe_ratio(num: float, den: float, abs_den_floor: float = 1e-6) -> float:
        """Internal helper: safe ratio."""
        if not (np.isfinite(num) and np.isfinite(den)):
            return float("nan")
        if abs(float(den)) < float(abs_den_floor):
            return float("nan")
        return float(num / den)

    emmi_e_over_m = _safe_ratio(e_med, m_med, abs_den_floor=1e-6)
    mv_over_speed = _safe_ratio(mv_med, sp_med, abs_den_floor=1e-6)

    return {
        "speed_mean": sp_mean,
        "speed_median": sp_med,
        "mv_median": mv_med,
        "mv_abs_median": mv_abs_med,
        "emmi_e_over_m_median": emmi_e_over_m,
        "mv_over_speed_median": mv_over_speed,
    }


def compute_ensemble_summary_for_subject(
    config: Mapping[str, Any],
    dataset_id: str,
    sub_frame: pd.DataFrame,
    coords_9d_names: list[str],
    subcoords_spec: Mapping[str, Mapping[str, float]],
    normalize_mode: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Compute ensemble-mean and variance over Stratified MNPS subcoordinates."""

    if not coords_9d_names or not subcoords_spec:
        return None

    robustness_cfg = config.get("robustness", {}) if isinstance(config, Mapping) else {}
    ensembles_cfg = robustness_cfg.get("ensembles", {}) if isinstance(robustness_cfg, Mapping) else {}
    if not isinstance(ensembles_cfg, Mapping) or not ensembles_cfg.get("enabled", False):
        return None

    config_groups = ensembles.resolve_config_groups(ensembles_cfg, dataset_id)
    if not config_groups:
        return None

    used_features: set[str] = set()
    for _, weight_map in subcoords_spec.items():
        if isinstance(weight_map, Mapping):
            for feat_name in weight_map.keys():
                used_features.add(str(feat_name))
    if not used_features:
        return None

    group_summaries: list[np.ndarray] = []
    realised_groups: list[str] = []

    for group_name in config_groups.keys():
        safe_name = ensembles.sanitize_group_name(group_name)
        suffix = f"__g_{safe_name}"

        needed_cols = {c for c in used_features if c in sub_frame.columns}
        needed_cols.update({f"{feat}{suffix}" for feat in used_features if f"{feat}{suffix}" in sub_frame.columns})
        df_group = sub_frame.loc[:, sorted(needed_cols)].copy() if needed_cols else pd.DataFrame(index=sub_frame.index)
        for feat in used_features:
            group_col = f"{feat}{suffix}"
            if group_col in df_group.columns:
                df_group[feat] = df_group[group_col]

        try:
            coords_g, names_g, _ = projection.project_features_v2(
                df_group,
                subcoords_spec,
                normalize=normalize_mode,
            )
        except Exception as exc:
            logger.warning("Failed to project v2 features for ensemble group %s in %s: %s", group_name, dataset_id, exc)
            continue

        if coords_g.size == 0:
            continue

        if coords_9d_names and list(names_g) != list(coords_9d_names):
            logger.warning(
                "coords_9d name mismatch for ensemble group %s in %s; skipping group", group_name, dataset_id
            )
            continue

        summary_g = np.nanmedian(coords_g, axis=0)
        if not np.all(np.isfinite(summary_g)):
            if np.all(np.isnan(summary_g)):
                continue
            summary_g = np.where(np.isfinite(summary_g), summary_g, np.nan)

        group_summaries.append(summary_g.astype(np.float32))
        realised_groups.append(str(group_name))

    if not group_summaries:
        return None

    stack = np.stack(group_summaries, axis=0)
    ensemble_mean = np.nanmean(stack, axis=0)
    ensemble_var = np.nanvar(stack, axis=0)

    return {
        "groups_config": config_groups,
        "groups_realised": realised_groups,
        "subcoord_names": list(coords_9d_names),
        "mean": {name: float(ensemble_mean[i]) for i, name in enumerate(coords_9d_names)},
        "var": {name: float(ensemble_var[i]) for i, name in enumerate(coords_9d_names)},
    }


def compute_robust_and_reliability_summaries(
    config: Mapping[str, Any],
    mnps_cfg: Mapping[str, Any],
    x: np.ndarray,
    coords_9d: Optional[np.ndarray],
    coords_9d_names: list[str],
) -> Dict[str, Any]:
    """Compute robust summaries and split-half reliability for MNPS coordinates."""

    result: Dict[str, Any] = {}

    try:
        axes_names = ["m", "d", "e"]
        axes_summary = robustness.summarize_array(x, axes_names, config)
        axes_reliability = robustness.split_half_reliability(x, axes_names)
        result["axes"] = {
            "summary": axes_summary,
            "reliability": axes_reliability,
        }
    except Exception:
        logger.exception("Failed to compute robustness summaries for MNPS axes")

    if coords_9d is not None and coords_9d_names:
        try:
            sub_summary = robustness.summarize_array(coords_9d, coords_9d_names, config)
            sub_reliability = robustness.split_half_reliability(coords_9d, coords_9d_names)
            result["subcoords"] = {
                "summary": sub_summary,
                "reliability": sub_reliability,
            }
        except Exception:
            logger.exception("Failed to compute robustness summaries for Stratified MNPS subcoordinates")

    return result


def compute_psd_multiverse_stability(
    config: Mapping[str, Any],
    ds_id: str,
    sub_frame: pd.DataFrame,
    coords_9d: Optional[np.ndarray],
    coords_9d_names: list[str],
    subcoords_spec: Mapping[str, Mapping[str, float]],
    normalize_mode: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Compute PSD multiverse stability indices for Stratified subcoordinates."""

    if coords_9d is None or not coords_9d_names or not subcoords_spec:
        return None

    robustness_cfg = config.get("robustness", {}) if isinstance(config, Mapping) else {}
    multiverse_cfg = robustness_cfg.get("multiverse", {}) if isinstance(robustness_cfg, Mapping) else {}
    psd_mv_cfg = multiverse_cfg.get("psd", {}) if isinstance(multiverse_cfg, Mapping) else {}
    if not isinstance(psd_mv_cfg, Mapping) or not psd_mv_cfg.get("enabled", False):
        return None

    used_features: set[str] = set()
    for _, weight_map in subcoords_spec.items():
        if isinstance(weight_map, Mapping):
            for feat_name in weight_map.keys():
                used_features.add(str(feat_name))

    alt_candidates = {feat for feat in used_features if f"{feat}__psd_alt" in sub_frame.columns}
    if not alt_candidates:
        return None

    needed_cols = set(used_features)
    needed_cols.update({f"{feat}__psd_alt" for feat in alt_candidates})
    needed_cols = {c for c in needed_cols if c in sub_frame.columns}
    df_alt = sub_frame.loc[:, sorted(needed_cols)].copy() if needed_cols else pd.DataFrame(index=sub_frame.index)
    for feat in alt_candidates:
        alt_col = f"{feat}__psd_alt"
        df_alt[feat] = df_alt[alt_col]

    try:
        coords_alt, names_alt, _ = projection.project_features_v2(
            df_alt,
            subcoords_spec,
            normalize=normalize_mode,
        )
    except Exception as exc:
        logger.warning("Failed to project v2 features for PSD multiverse in %s: %s", ds_id, exc)
        return None

    if coords_alt.size == 0 or list(names_alt) != list(coords_9d_names):
        return None

    T, K = coords_9d.shape
    if T != coords_alt.shape[0] or K != coords_alt.shape[1]:
        return None

    stability: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(coords_9d_names):
        primary = coords_9d[:, idx]
        alt = coords_alt[:, idx]
        mask = np.isfinite(primary) & np.isfinite(alt)
        primary = primary[mask]
        alt = alt[mask]
        if primary.size < 2:
            stability[str(name)] = {"corr": float("nan"), "mean_abs_diff": float("nan")}
            continue
        p_mean = float(primary.mean())
        a_mean = float(alt.mean())
        num = float(np.sum((primary - p_mean) * (alt - a_mean)))
        den = float(np.sqrt(np.sum((primary - p_mean) ** 2) * np.sum((alt - a_mean) ** 2)))
        corr = float("nan") if den == 0 else float(num / den)
        mad = float(np.mean(np.abs(primary - alt)))
        stability[str(name)] = {"corr": corr, "mean_abs_diff": mad}

    primary_method = str(config.get("features", {}).get("eeg_psd", {}).get("method", "multitaper")).lower()
    secondary_method = str(psd_mv_cfg.get("secondary_method", "welch")).lower()

    return {
        "methods": {
            "primary": primary_method,
            "secondary": secondary_method,
        },
        "stability": stability,
    }


def build_qc_summary(
    dataset_label: str,
    ds_path: Path,
    sub_id: str,
    ses_id: Optional[str],
    sub_frame: pd.DataFrame,
    dt: float,
    ensemble_summary: Optional[Dict[str, Any]],
    robust_summary: Dict[str, Any] | None,
    dist_summary: Dict[str, Any] | None,
    entropy_qc: Dict[str, Any] | None,
    geometry_contract: Dict[str, Any] | None = None,
    mnps_mnj_sanity: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Aggregate per-subject robustness into a compact QC summary."""

    epochs = int(len(sub_frame))
    seconds = float(epochs * dt)

    artifact_summary: Dict[str, Any] = {}
    try:
        qc_dir = ds_path / "qc_artifacts"
        methods: set[str] = set()
        bad_channels: set[str] = set()
        if qc_dir.exists() and "file" in sub_frame.columns:
            file_values = [str(f) for f in sub_frame["file"].dropna().astype(str).unique()]
            stem_to_names: Dict[str, set[str]] = {}
            for file_name in file_values:
                stem_to_names.setdefault(Path(file_name).stem, set()).add(Path(file_name).name)
            stem_collisions = {s: sorted(list(v)) for s, v in stem_to_names.items() if len(v) > 1}
            if stem_collisions:
                logger.warning(
                    "QC artifact stem collisions detected for %s: %s",
                    dataset_label,
                    stem_collisions,
                )
            file_stems = set(stem_to_names.keys())
            import json  # local import to avoid top-level dependency

            for stem in file_stems:
                qc_path = qc_dir / f"{stem}_qc_artifacts.json"
                if not qc_path.exists():
                    continue
                try:
                    with qc_path.open("r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    continue
                art = meta.get("artifact", {}) if isinstance(meta, dict) else {}
                method = art.get("method")
                if isinstance(method, str):
                    methods.add(method)
                for ch in art.get("bad_eeg_channels", []) or []:
                    bad_channels.add(str(ch))
        artifact_summary = {
            "methods": sorted(methods) if methods else [],
            "bad_eeg_channels": sorted(bad_channels) if bad_channels else [],
            "n_bad_eeg_channels": len(bad_channels),
            "stem_collisions": stem_collisions if "stem_collisions" in locals() else {},
        }
    except Exception:
        logger.exception("Failed to aggregate artifact metrics for %s", dataset_label)
        artifact_summary = {}

    ensemble_metrics: Dict[str, Any] = {}
    if ensemble_summary is not None:
        var_map = ensemble_summary.get("var", {}) or {}
        try:
            vals = np.asarray(list(var_map.values()), dtype=float)
            ensemble_metrics = {
                "var_by_subcoord": var_map,
                "var_mean": float(np.nanmean(vals)) if vals.size else float("nan"),
                "var_max": float(np.nanmax(vals)) if vals.size else float("nan"),
            }
        except Exception:
            ensemble_metrics = {"var_by_subcoord": var_map}

    reliability_axes = {}
    reliability_subcoords = {}
    if robust_summary:
        reliability_axes = (robust_summary.get("axes") or {}).get("reliability", {}) or {}
        reliability_subcoords = (robust_summary.get("subcoords") or {}).get("reliability", {}) or {}

    provisional_axes: Dict[str, Any] = {}
    if entropy_qc:
        for name, info in entropy_qc.items():
            if bool(info.get("provisional", False)):
                provisional_axes[name] = info

    return {
        "dataset_id": dataset_label,
        "subject": sub_id,
        "session": ses_id,
        "coverage": {
            "epochs": epochs,
            "seconds": seconds,
        },
        "artifacts": artifact_summary,
        "ensemble": ensemble_metrics,
        "reliability": {
            "axes": reliability_axes,
            "subcoords": reliability_subcoords,
        },
        # Neutral distributional descriptives (mean/median/std/iqr + delta).
        "dist_summary": dist_summary or {},
        "entropy_provisional": provisional_axes,
        "geometry_contract": geometry_contract or {},
        "mnps_mnj_sanity": mnps_mnj_sanity or {},
    }

