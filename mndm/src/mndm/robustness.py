"""Robust summary statistics, outlier down-weighting, and bootstrap CIs.

Inputs
------
- coords_df: DataFrame with per-epoch m/d/e values.
- transient_mask: boolean array marking transient epochs.
- config: robustness settings (summary type, trim, bootstrap_n, seed, coverage).

Outputs
-------
- RobustSummary with incl/excl summaries, CI95, stability, coverage flags.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

from core.stats import robust as robust_stats
from .reproducibility import resolve_base_seed

logger = logging.getLogger(__name__)


def _cfg_with_resolved_seed(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg) if isinstance(cfg, dict) else {}
    robustness_cfg = out.get("robustness", {})
    robustness_cfg = dict(robustness_cfg) if isinstance(robustness_cfg, dict) else {}
    seed, source = resolve_base_seed(out)
    robustness_cfg["seed"] = int(seed)
    out["robustness"] = robustness_cfg
    repro_cfg = dict(out.get("reproducibility", {}) or {}) if isinstance(out.get("reproducibility", {}), dict) else {}
    repro_cfg.setdefault("seed", int(seed))
    repro_cfg.setdefault("seed_source", str(source))
    out["reproducibility"] = repro_cfg
    return out


@dataclass
class RobustSummary:
    """Container for robust MNPS summaries."""
    incl: Dict[str, float]
    excl: Dict[str, float]
    ci95: Dict[str, Tuple[float, float]]
    stability: Dict[str, float]
    coverage_ok: bool
    transient_frac: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "incl": self.incl,
            "excl": self.excl,
            "ci95": self.ci95,
            "stability": self.stability,
            "coverage_ok": self.coverage_ok,
            "transient_frac": self.transient_frac,
        }


def compute_robust_summary(coords_df: pd.DataFrame, transient_mask: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Compute robust summaries for MNPS coordinates.
    
    Parameters
    ----------
    coords_df
        DataFrame with columns: epoch_id, m, d, e
    transient_mask
        Boolean array marking transient epochs.
    cfg
        Configuration dict with robustness settings.
    
    Returns
    -------
    RobustSummary with incl/excl stats, CI95, stability, coverage.
    """
    if len(coords_df) == 0:
        return {
            "incl": {},
            "excl": {},
            "ci95": {},
            "stability": {},
            "coverage_ok": False,
            "transient_frac": 0.0,
        }
    # Canonical axis names are m/d/e. Accept legacy R/D/M as fallback input.
    axis_aliases = {"m": ["m", "R"], "d": ["d", "D"], "e": ["e", "M"]}
    selected_cols: Dict[str, str] = {}
    for axis, candidates in axis_aliases.items():
        for col in candidates:
            if col in coords_df.columns:
                selected_cols[axis] = col
                break
        if axis not in selected_cols:
            raise ValueError(
                f"coords_df missing required axis '{axis}' "
                f"(accepted aliases: {candidates})"
            )
    required = ["m", "d", "e"]

    cfg_resolved = _cfg_with_resolved_seed(cfg if isinstance(cfg, dict) else {})
    robustness = cfg_resolved.get("robustness", {}) if isinstance(cfg_resolved, dict) else {}
    summary_type = str(robustness.get("summary", "median")).lower()
    trim_pct = float(robustness.get("trim_pct", 0.0) or 0.0)
    bootstrap_n = int(robustness.get("bootstrap_n", 1000) or 0)
    seed = int(robustness.get("seed", 42))
    coverage = robustness.get("coverage", {})
    rng = np.random.default_rng(seed)

    mask_arr = np.asarray(transient_mask, dtype=bool) if transient_mask is not None else np.zeros((0,), dtype=bool)
    if mask_arr.size != len(coords_df):
        logger.warning(
            "Transient mask length mismatch (mask=%s, rows=%s); defaulting to all-non-transient",
            mask_arr.size,
            len(coords_df),
        )
        mask_arr = np.zeros((len(coords_df),), dtype=bool)

    # Transient fraction
    transient_frac = float(np.sum(mask_arr) / len(mask_arr)) if mask_arr.size > 0 else 0.0

    # Check coverage
    min_epochs = int(coverage.get("min_epochs", 20) or 20)
    coverage_ok = (len(coords_df) >= min_epochs) and transient_frac < 0.5

    def _point(arr: np.ndarray) -> float:
        return robust_stats.robust_1d(arr, summary=summary_type, trim_pct=trim_pct)

    def _finite(arr: pd.Series) -> np.ndarray:
        vals = np.asarray(arr.to_numpy(dtype=float), dtype=float)
        return vals[np.isfinite(vals)]

    # Compute summaries: incl (all epochs)
    incl_stats = {
        ax: float(_point(_finite(coords_df[selected_cols[ax]])))
        for ax in required
    }

    # Compute summaries: excl (no transients)
    coords_excl = coords_df.iloc[~mask_arr] if mask_arr.size > 0 else coords_df
    if len(coords_excl) > 0:
        excl_stats = {
            ax: float(_point(_finite(coords_excl[selected_cols[ax]])))
            for ax in required
        }
    else:
        excl_stats = incl_stats.copy()

    # Bootstrap CI95
    ci95 = {}
    for axis in required:
        finite = (
            _finite(coords_excl[selected_cols[axis]])
            if len(coords_excl) > 0
            else np.asarray([], dtype=float)
        )
        if finite.size > 0 and bootstrap_n > 0:
            n = finite.size
            idx = rng.integers(0, n, size=(bootstrap_n, n))
            samples = finite[idx]
            if summary_type == "median":
                bootstrap_samples = np.median(samples, axis=1)
            elif summary_type == "mean" and 0.0 < trim_pct < 1.0:
                # Keep trimmed-mean semantics for bootstrap draws.
                bootstrap_samples = np.array(
                    [robust_stats.robust_1d(s, summary="mean", trim_pct=trim_pct) for s in samples],
                    dtype=float,
                )
            else:
                bootstrap_samples = np.mean(samples, axis=1)
            ci95[axis] = (float(np.percentile(bootstrap_samples, 2.5)), float(np.percentile(bootstrap_samples, 97.5)))
        else:
            val = excl_stats[axis] if np.isfinite(excl_stats[axis]) else incl_stats[axis]
            ci95[axis] = (float(val), float(val))

    # Stability: robust dispersion ratio mapped to [0,1]
    stability = {}
    for axis in required:
        finite = (
            _finite(coords_excl[selected_cols[axis]])
            if len(coords_excl) > 0
            else np.asarray([], dtype=float)
        )
        if finite.size > 0:
            std_val = float(np.std(finite, ddof=0))
            med_val = float(np.median(finite))
            mad = float(np.median(np.abs(finite - med_val)))
            mad_sigma = float(1.4826 * mad) if np.isfinite(mad) else float("nan")
            denom = mad_sigma if np.isfinite(mad_sigma) and mad_sigma > 1e-9 else 1e-9
            ratio = float(std_val / denom)
            stability[axis] = float(1.0 / (1.0 + ratio))
        else:
            stability[axis] = float("nan")

    logger.debug("Robust summary computed (transient_frac=%.3f, coverage_ok=%s)", transient_frac, coverage_ok)

    return {
        "incl": incl_stats,
        "excl": excl_stats,
        "ci95": ci95,
        "stability": stability,
        "coverage_ok": coverage_ok,
        "transient_frac": transient_frac,
    }


def summarize_array(
    values: np.ndarray,
    axis_names: Sequence[str],
    cfg: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """Summarize columns of a 2-D array with robust point estimates and CI95.

    Parameters
    ----------
    values
        Array of shape [T, K] with per-epoch values.
    axis_names
        Names for the K columns, e.g. ['m_a', 'm_e', ...].
    cfg
        Config dict with 'robustness' section (summary, trim_pct,
        bootstrap_n, seed).

    Returns
    -------
    dict
        Mapping name -> {'point', 'ci_low', 'ci_high'}.
    """
    cfg_resolved = _cfg_with_resolved_seed(cfg if isinstance(cfg, dict) else {})
    return robust_stats.summarize_array(values, axis_names, cfg_resolved)


def split_half_reliability(
    values: np.ndarray,
    axis_names: Sequence[str],
    split_mode: str = "odd_even",
) -> Dict[str, float]:
    """Compute split-half temporal consistency (correlation) for each axis.

    Parameters
    ----------
    values
        Array of shape [T, K] with per-epoch values.
    axis_names
        Names for the K columns.

    Returns
    -------
    dict
        Mapping name -> Pearson correlation between split traces.
        NaN when not enough data.
    """

    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(axis_names):
        raise ValueError("values must be 2-D with second dimension matching axis_names")

    T = values.shape[0]
    if T < 4:
        # Not enough epochs for a meaningful split-half estimate
        return {str(name): float("nan") for name in axis_names}

    mode = str(split_mode).strip().lower()
    if mode == "first_second":
        half = T // 2
        a = values[:half, :]
        b = values[-half:, :]
    else:
        a = values[0::2, :]
        b = values[1::2, :]
        half = min(a.shape[0], b.shape[0])
        a = a[:half, :]
        b = b[:half, :]

    out: Dict[str, float] = {}
    for col_idx, name in enumerate(axis_names):
        va = a[:, col_idx]
        vb = b[:, col_idx]
        mask = np.isfinite(va) & np.isfinite(vb)
        va = va[mask]
        vb = vb[mask]
        if va.size < 2:
            out[str(name)] = float("nan")
            continue
        # Pearson correlation
        va_mean = va.mean()
        vb_mean = vb.mean()
        num = np.sum((va - va_mean) * (vb - vb_mean))
        den = np.sqrt(np.sum((va - va_mean) ** 2) * np.sum((vb - vb_mean) ** 2))
        if den == 0:
            out[str(name)] = float("nan")
        else:
            out[str(name)] = float(num / den)

    return out


def entropy_sanity_checks(
    coords_9d: np.ndarray,
    coords_9d_names: Sequence[str],
    target_axes: Sequence[str] | None = None,
    var_threshold: float = 1e-4,
    min_unique: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """Check for degenerate entropy/efficiency subcoordinates.

    Parameters
    ----------
    coords_9d
        Array of shape [T, K] with Stratified MNPS subcoordinates in [0, 1].
    coords_9d_names
        Names for the subcoordinates.
    target_axes
        Subset of names to check (default: ['e_e', 'e_s', 'e_m']).
    var_threshold
        Variance threshold below which a subcoordinate is considered degenerate.
    min_unique
        Minimum number of unique values (after rounding) required to avoid
        being marked as degenerate.

    Returns
    -------
    dict
        Mapping axis -> {'var', 'n_unique', 'provisional'}.
    """

    coords_9d = np.asarray(coords_9d, dtype=float)
    names = list(coords_9d_names)
    if coords_9d.ndim != 2 or coords_9d.shape[1] != len(names):
        raise ValueError("coords_9d must be 2-D with second dimension matching coords_9d_names")

    if target_axes is None:
        target_axes = ("e_e", "e_s", "e_m")

    idx_map = {name: i for i, name in enumerate(names)}
    out: Dict[str, Dict[str, Any]] = {}
    for axis in target_axes:
        idx = idx_map.get(str(axis))
        if idx is None:
            continue
        series = coords_9d[:, idx]
        finite = series[np.isfinite(series)]
        nan_frac = float(1.0 - (finite.size / series.size if series.size else 1.0))
        if finite.size == 0:
            out[str(axis)] = {"var": float("nan"), "n_unique": 0, "nan_frac": nan_frac, "provisional": True}
            continue
        var = float(np.var(finite))
        # Round to reduce spurious uniqueness from floating noise
        unique_vals = np.unique(np.round(finite, 6))
        n_unique = int(unique_vals.size)
        provisional = bool((var < var_threshold) or (n_unique < min_unique))
        out[str(axis)] = {"var": var, "n_unique": n_unique, "nan_frac": nan_frac, "provisional": provisional}

    return out


