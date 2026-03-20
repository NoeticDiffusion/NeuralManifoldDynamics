"""Reviewer-oriented baseline and null QA summaries for MNPS outputs."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .. import jacobian, projection
from ..reproducibility import resolve_component_seed
from .robustness_helpers import compute_tau_summary, compute_tier2_jacobian_metrics

AXIS_NAMES = ["m", "d", "e"]
SERIES_METADATA_EXCLUDE = ("construct", "metric", "backend", "reason", "degraded_mode")
EEG_BANDPOWER_CANDIDATES = [
    "eeg_delta",
    "eeg_theta",
    "eeg_alpha",
    "eeg_beta",
    "eeg_gamma",
    "eeg_highfreq_power_30_45",
]
FMRI_VARIANCE_POWER_CANDIDATES = [
    "fmri_variance_global",
    "fmri_signal_power",
    "fmri_slow4_slow5_ratio",
    "fmri_ar1_coefficient",
]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _corr_1d(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    aa = np.asarray(a[mask], dtype=float)
    bb = np.asarray(b[mask], dtype=float)
    if aa.size < 2:
        return float("nan")
    aa = aa - float(np.mean(aa))
    bb = bb - float(np.mean(bb))
    denom = float(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(aa * bb) / denom)


def _approx_entropy_1d(series: np.ndarray, bins: int = 20) -> float:
    finite = np.asarray(series, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 4:
        return float("nan")
    hist, _ = np.histogram(finite, bins=int(max(2, bins)), density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return float("nan")
    return float(-np.sum(hist * np.log(hist)))


def _series_summary(series: np.ndarray, dt_sec: float) -> Dict[str, Any]:
    arr = np.asarray(series, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    out: Dict[str, Any] = {
        "n": int(finite.size),
        "nan_frac": float(1.0 - (finite.size / arr.size if arr.size else 1.0)),
        "mean": float("nan"),
        "median": float("nan"),
        "std": float("nan"),
        "approx_entropy": float("nan"),
        "tau_sec": float("nan"),
        "lag1_autocorr": float("nan"),
    }
    if finite.size == 0:
        return out
    out["mean"] = float(np.mean(finite))
    out["median"] = float(np.median(finite))
    out["std"] = float(np.std(finite, ddof=0))
    out["approx_entropy"] = _approx_entropy_1d(finite)
    tau = compute_tau_summary(arr[:, None], ["value"], dt_sec=float(dt_sec), nan_policy="interpolate")
    out["tau_sec"] = _safe_float((tau.get("value") or {}).get("tau_sec"))
    if finite.size >= 2:
        out["lag1_autocorr"] = _corr_1d(finite[:-1], finite[1:])
    return out


def _summarize_series_against_axes(series: np.ndarray, x: np.ndarray, dt_sec: float) -> Dict[str, Any]:
    summary = _series_summary(series, dt_sec=dt_sec)
    axis_corrs = {
        axis: _corr_1d(np.asarray(series, dtype=float).reshape(-1), x[:, idx])
        for idx, axis in enumerate(AXIS_NAMES)
    }
    best_axis = None
    best_abs = float("nan")
    finite_pairs = [(axis, abs(val)) for axis, val in axis_corrs.items() if np.isfinite(val)]
    if finite_pairs:
        best_axis, best_abs = max(finite_pairs, key=lambda item: item[1])
    summary["corr_to_mnps"] = axis_corrs
    summary["best_axis"] = best_axis
    summary["best_axis_abs_corr"] = float(best_abs) if np.isfinite(best_abs) else float("nan")
    return summary


def _global_numeric_columns(sub_frame: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in sub_frame.columns:
        if "__g_" in str(col):
            continue
        if not pd.api.types.is_numeric_dtype(sub_frame[col]):
            continue
        cols.append(str(col))
    return cols


def _order_candidates(candidates: Sequence[str], preferred: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in preferred:
        if name in candidates and name not in seen:
            out.append(str(name))
            seen.add(str(name))
    for name in sorted([str(c) for c in candidates]):
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


def _select_baseline_families(sub_frame: pd.DataFrame, max_series_per_family: int) -> Dict[str, list[str]]:
    numeric_cols = _global_numeric_columns(sub_frame)
    entropy_cols = [
        c
        for c in numeric_cols
        if "entropy" in c.lower() and not any(token in c.lower() for token in SERIES_METADATA_EXCLUDE)
    ]
    variance_bandpower_cols = [
        c
        for c in numeric_cols
        if (
            c in EEG_BANDPOWER_CANDIDATES
            or c in FMRI_VARIANCE_POWER_CANDIDATES
            or ("variance" in c.lower() and "dfc" not in c.lower())
        )
    ]
    sliding_fc_cols = [c for c in numeric_cols if "dfc" in c.lower()]
    families = {
        "entropy_raw": _order_candidates(
            entropy_cols,
            preferred=["eeg_permutation_entropy", "eeg_sample_entropy", "fmri_entropy_global"],
        ),
        "variance_bandpower": _order_candidates(
            variance_bandpower_cols,
            preferred=EEG_BANDPOWER_CANDIDATES + FMRI_VARIANCE_POWER_CANDIDATES,
        ),
        "sliding_window_fc": _order_candidates(
            sliding_fc_cols,
            preferred=["eeg_dfc_variance", "eeg_dfc_entropy", "fmri_dFC_variance", "fmri_dFC_entropy"],
        ),
    }
    return {
        family: cols[: int(max(0, max_series_per_family))]
        for family, cols in families.items()
        if cols
    }


def compute_feature_baseline_comparisons(
    *,
    sub_frame: pd.DataFrame,
    x: np.ndarray,
    dt_sec: float,
    review_qc_cfg: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Summarize simple baseline feature families against MNPS axes."""
    cfg = review_qc_cfg or {}
    baseline_cfg = cfg.get("baseline_comparisons", {}) if isinstance(cfg, Mapping) else {}
    if not isinstance(baseline_cfg, Mapping) or not baseline_cfg.get("enabled", False):
        return None
    if x.ndim != 2 or x.shape[1] != 3 or len(sub_frame) != x.shape[0]:
        return None

    max_series = int(baseline_cfg.get("max_series_per_family", 6) or 6)
    smooth_window = int(baseline_cfg.get("smoothing_window_epochs", 5) or 5)
    selected = _select_baseline_families(sub_frame, max_series_per_family=max_series)

    nmd_axes = {
        axis: _series_summary(x[:, idx], dt_sec=dt_sec)
        for idx, axis in enumerate(AXIS_NAMES)
    }

    families: Dict[str, Dict[str, Any]] = {}
    for family_name, columns in selected.items():
        family_payload: Dict[str, Any] = {}
        for col in columns:
            series = pd.to_numeric(sub_frame[col], errors="coerce").to_numpy(dtype=float)
            family_payload[col] = _summarize_series_against_axes(series, x=x, dt_sec=dt_sec)
        if family_payload:
            families[family_name] = family_payload

    entropy_smoothed: Dict[str, Any] = {}
    for col in selected.get("entropy_raw", []):
        series = pd.to_numeric(sub_frame[col], errors="coerce")
        smoothed = series.rolling(window=max(1, smooth_window), center=True, min_periods=1).mean().to_numpy(dtype=float)
        entropy_smoothed[f"{col}__smoothed"] = _summarize_series_against_axes(smoothed, x=x, dt_sec=dt_sec)
    if entropy_smoothed:
        families["entropy_smoothed"] = entropy_smoothed

    return {
        "config": {
            "max_series_per_family": max_series,
            "smoothing_window_epochs": smooth_window,
        },
        "nmd_axes": nmd_axes,
        "families": families,
        "note": (
            "Neutral reviewer-facing summaries for simple baseline feature families. "
            "These are comparison surfaces only, not superiority claims."
        ),
    }


def _filtered_x_and_files(
    x: np.ndarray,
    file_labels: Optional[Sequence[Any]],
) -> tuple[np.ndarray, Optional[np.ndarray], float]:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return np.zeros((0, 3), dtype=np.float32), None, 0.0
    finite_mask = np.isfinite(arr).all(axis=1)
    filtered = arr[finite_mask]
    frac_kept = float(np.mean(finite_mask)) if finite_mask.size else 0.0
    files_out = None
    if file_labels is not None:
        files_arr = np.asarray(file_labels)
        if files_arr.shape[0] == arr.shape[0]:
            files_out = files_arr[finite_mask]
    return filtered, files_out, frac_kept


def _compute_dot(
    x: np.ndarray,
    *,
    dt_sec: float,
    derivative_cfg: Mapping[str, Any],
    derivative_robust_cfg: Optional[Mapping[str, Any]],
    file_labels: Optional[Sequence[Any]],
) -> np.ndarray:
    robust_cfg = derivative_robust_cfg or {}
    use_segmented = bool(robust_cfg.get("enabled", True))
    method = str(derivative_cfg.get("method", "sav_gol"))
    window = int(derivative_cfg.get("window", 7))
    polyorder = int(derivative_cfg.get("polyorder", 3))
    if file_labels is not None and len(file_labels) == len(x) and len(np.unique(file_labels)) > 1:
        dot = np.zeros_like(x, dtype=np.float32)
        file_arr = np.asarray(file_labels)
        for value in np.unique(file_arr):
            mask = file_arr == value
            sub_x = x[mask]
            if sub_x.size == 0:
                continue
            if use_segmented:
                dot[mask] = projection.estimate_derivatives_segmented(
                    sub_x,
                    dt_sec,
                    method=method,
                    max_jump=float(robust_cfg.get("max_jump", 5.0)),
                    min_seg=int(robust_cfg.get("min_seg", 9)),
                    savgol_window=window,
                    polyorder=polyorder,
                )
            else:
                dot[mask] = projection.estimate_derivatives(
                    sub_x,
                    dt_sec,
                    method=method,
                    window=window,
                    polyorder=polyorder,
                )
        return dot
    if use_segmented:
        return projection.estimate_derivatives_segmented(
            x,
            dt_sec,
            method=method,
            max_jump=float(robust_cfg.get("max_jump", 5.0)),
            min_seg=int(robust_cfg.get("min_seg", 9)),
            savgol_window=window,
            polyorder=polyorder,
        )
    return projection.estimate_derivatives(
        x,
        dt_sec,
        method=method,
        window=window,
        polyorder=polyorder,
    )


def _path_metrics(x: np.ndarray, dt_sec: float) -> Dict[str, Any]:
    if x.shape[0] < 2:
        return {"path_total": float("nan"), "step_mean": float("nan"), "step_median": float("nan")}
    steps = np.linalg.norm(np.diff(x, axis=0), axis=1)
    return {
        "path_total": float(np.sum(steps)),
        "step_mean": float(np.mean(steps)),
        "step_median": float(np.median(steps)),
        "dt_sec": float(dt_sec),
    }


def _summarize_surrogate(
    x: np.ndarray,
    *,
    dt_sec: float,
    derivative_cfg: Mapping[str, Any],
    derivative_robust_cfg: Optional[Mapping[str, Any]],
    file_labels: Optional[Sequence[Any]],
    knn_k: int,
    knn_metric: str,
    whiten: bool,
    super_window: int,
    ridge_alpha: float,
    distance_weighted: bool,
) -> Dict[str, Any]:
    x_dot = _compute_dot(
        x,
        dt_sec=dt_sec,
        derivative_cfg=derivative_cfg,
        derivative_robust_cfg=derivative_robust_cfg,
        file_labels=file_labels,
    )
    nn_idx = projection.build_knn_indices(
        x,
        k=int(knn_k),
        metric=str(knn_metric),
        whiten=bool(whiten),
    )
    jac_res = jacobian.estimate_local_jacobians(
        x,
        x_dot,
        nn_idx,
        super_window=int(super_window),
        ridge_alpha=float(ridge_alpha),
        distance_weighted=bool(distance_weighted),
        j_dot_dt=float(dt_sec),
    )
    tau = compute_tau_summary(x, AXIS_NAMES, dt_sec=float(dt_sec), nan_policy="interpolate")
    tier2 = compute_tier2_jacobian_metrics(jac_res.j_hat, jacobian_diagnostics=jac_res.diagnostics)
    axes = {
        axis: _series_summary(x[:, idx], dt_sec=dt_sec)
        for idx, axis in enumerate(AXIS_NAMES)
    }
    return {
        "axes": axes,
        "tau_summary": tau,
        "path_metrics": _path_metrics(x, dt_sec=dt_sec),
        "jacobian": {
            "windows": int(jac_res.j_hat.shape[0]),
            "failed_windows": _safe_float(jac_res.diagnostics.get("failed")),
            "rel_mse_baseline_median": _safe_float(jac_res.diagnostics.get("rel_mse_baseline_median")),
            "tier2": tier2,
        },
    }


def _mean_axis_metric(summary: Mapping[str, Any], key: str) -> float:
    vals = []
    axes = summary.get("axes", {}) if isinstance(summary, Mapping) else {}
    for axis in AXIS_NAMES:
        val = _safe_float((axes.get(axis) or {}).get(key))
        if np.isfinite(val):
            vals.append(val)
    return float(np.mean(vals)) if vals else float("nan")


def _mean_axis_tau(summary: Mapping[str, Any]) -> float:
    vals = []
    tau = summary.get("tau_summary", {}) if isinstance(summary, Mapping) else {}
    for axis in AXIS_NAMES:
        val = _safe_float((tau.get(axis) or {}).get("tau_sec"))
        if np.isfinite(val):
            vals.append(val)
    return float(np.mean(vals)) if vals else float("nan")


def _comparison_to_original(original: Mapping[str, Any], surrogate: Mapping[str, Any]) -> Dict[str, Any]:
    orig_std = _mean_axis_metric(original, "std")
    surr_std = _mean_axis_metric(surrogate, "std")
    orig_tau = _mean_axis_tau(original)
    surr_tau = _mean_axis_tau(surrogate)
    orig_path = _safe_float(((original.get("path_metrics") or {}).get("path_total")))
    surr_path = _safe_float(((surrogate.get("path_metrics") or {}).get("path_total")))
    orig_rel = _safe_float((((original.get("jacobian") or {}).get("tier2") or {}).get("rel_mse_baseline") or {}).get("median"))
    surr_rel = _safe_float((((surrogate.get("jacobian") or {}).get("tier2") or {}).get("rel_mse_baseline") or {}).get("median"))
    orig_rot = _safe_float((((original.get("jacobian") or {}).get("tier2") or {}).get("rotation_coherence") or {}).get("mean_resultant_length"))
    surr_rot = _safe_float((((surrogate.get("jacobian") or {}).get("tier2") or {}).get("rotation_coherence") or {}).get("mean_resultant_length"))

    def _ratio(num: float, den: float) -> float:
        if not (np.isfinite(num) and np.isfinite(den)) or abs(den) < 1e-8:
            return float("nan")
        return float(num / den)

    return {
        "mean_axis_std_ratio": _ratio(surr_std, orig_std),
        "mean_axis_tau_ratio": _ratio(surr_tau, orig_tau),
        "path_total_ratio": _ratio(surr_path, orig_path),
        "jacobian_rel_mse_baseline_ratio": _ratio(surr_rel, orig_rel),
        "rotation_coherence_delta": float(surr_rot - orig_rot) if np.isfinite(surr_rot) and np.isfinite(orig_rot) else float("nan"),
    }


def compute_null_sanity_tests(
    *,
    x: np.ndarray,
    dt_sec: float,
    derivative_cfg: Mapping[str, Any],
    derivative_robust_cfg: Optional[Mapping[str, Any]],
    file_labels: Optional[Sequence[Any]],
    knn_k: int,
    knn_metric: str,
    whiten: bool,
    super_window: int,
    ridge_alpha: float,
    distance_weighted: bool,
    review_qc_cfg: Optional[Mapping[str, Any]] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Run minimal surrogate/null tests on the exported MNPS trajectory."""
    cfg = review_qc_cfg or {}
    null_cfg = cfg.get("null_sanity_tests", {}) if isinstance(cfg, Mapping) else {}
    if not isinstance(null_cfg, Mapping) or not null_cfg.get("enabled", False):
        return None

    x_finite, file_finite, frac_kept = _filtered_x_and_files(x, file_labels=file_labels)
    if x_finite.shape[0] < max(12, int(super_window) + 2):
        return {
            "status": "skipped_insufficient_finite_rows",
            "finite_rows_used": int(x_finite.shape[0]),
            "finite_row_fraction": float(frac_kept),
        }

    seed, seed_source = resolve_component_seed(
        config,
        fallback_seed=null_cfg.get("seed"),
        fallback_source="robustness.review_qc.null_sanity_tests.seed",
    )
    rng = np.random.default_rng(seed)

    original = _summarize_surrogate(
        x_finite,
        dt_sec=dt_sec,
        derivative_cfg=derivative_cfg,
        derivative_robust_cfg=derivative_robust_cfg,
        file_labels=file_finite,
        knn_k=knn_k,
        knn_metric=knn_metric,
        whiten=whiten,
        super_window=super_window,
        ridge_alpha=ridge_alpha,
        distance_weighted=distance_weighted,
    )

    shuffle_idx = rng.permutation(x_finite.shape[0])
    shuffled = x_finite[shuffle_idx]
    phase_randomized = jacobian.phase_randomise(x_finite, seed=seed + 1)
    mean = np.mean(x_finite, axis=0, keepdims=True)
    std = np.std(x_finite, axis=0, keepdims=True)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    white_noise = rng.normal(loc=mean, scale=std, size=x_finite.shape).astype(np.float32)

    surrogates = {
        "shuffled_time": shuffled,
        "phase_randomized": phase_randomized,
        "white_noise": white_noise,
    }

    surrogate_summaries: Dict[str, Any] = {}
    comparisons: Dict[str, Any] = {}
    for name, x_surr in surrogates.items():
        summary = _summarize_surrogate(
            np.asarray(x_surr, dtype=np.float32),
            dt_sec=dt_sec,
            derivative_cfg=derivative_cfg,
            derivative_robust_cfg=derivative_robust_cfg,
            file_labels=file_finite,
            knn_k=knn_k,
            knn_metric=knn_metric,
            whiten=whiten,
            super_window=super_window,
            ridge_alpha=ridge_alpha,
            distance_weighted=distance_weighted,
        )
        surrogate_summaries[name] = summary
        comparisons[name] = _comparison_to_original(original, summary)

    return {
        "status": "ok",
        "source_level": "mnps_3d",
        "finite_rows_used": int(x_finite.shape[0]),
        "finite_row_fraction": float(frac_kept),
        "seed": seed,
        "seed_source": seed_source,
        "note": (
            "These nulls operate on the exported MNPS 3D trajectory, not on raw acquisition signals. "
            "They are intended as reviewer-facing sanity checks on temporal structure and Jacobian stability."
        ),
        "original": original,
        "surrogates": surrogate_summaries,
        "comparisons_to_original": comparisons,
    }
