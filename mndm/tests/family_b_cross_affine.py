"""Test-only 3D→3D vs 9D→3D cross-affine maps (Family B sl_001).

Not a production measurement module. Kept next to the tests that use it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from mndm.jacobian import (
    OOS_HOLDOUT_STRIDE,
    SUPPORT_MODE_KNN,
    SUPPORT_MODE_TIME_LOCAL,
    _affine_rel_mse,
    _fit_ridge,
    _gather_support_indices,
    _holdout_split,
    _normalize_support_mode,
    infer_knn_k,
    summarize_neighborhood_sample_counts,
    summarize_oos_rel_mse,
)


@dataclass
class CrossAffineResult:
    map_hat: np.ndarray
    centers: np.ndarray
    diagnostics: Dict[str, Any]
    affine_reference: Optional[np.ndarray] = None
    affine_intercept: Optional[np.ndarray] = None


def n_cross_affine_parameters(state_dim: int, target_dim: int) -> int:
    """Parameters in ``ẏ ≈ B(x − x̄) + b``."""
    return int(target_dim) * (int(state_dim) + 1)


def cross_affine_config_fields(
    *,
    state_dim: int,
    target_dim: int,
    knn_k: int,
    super_window: int,
    ridge_alpha: float,
    distance_weighted: bool,
    support_mode: str = SUPPORT_MODE_KNN,
) -> Dict[str, Any]:
    n_params = n_cross_affine_parameters(state_dim, target_dim)
    min_samples = int(state_dim) + 1
    if str(support_mode) == SUPPORT_MODE_TIME_LOCAL:
        min_samples = max(min_samples, n_params)
    return {
        "support_mode": str(support_mode),
        "knn_k": int(knn_k),
        "super_window": int(super_window),
        "ridge_alpha": float(ridge_alpha),
        "distance_weighted": bool(distance_weighted),
        "min_samples": int(min_samples),
        "n_affine_parameters": int(n_params),
        "state_dim": int(state_dim),
        "target_dim": int(target_dim),
        "estimator_family": "cross_affine_state_to_target_dot",
    }


def _fit_cross_affine(
    x_state: np.ndarray,
    y_dot: np.ndarray,
    src: np.ndarray,
    center: int,
    *,
    x_metric: np.ndarray,
    ridge_alpha: float,
    distance_weighted: bool,
    min_samples: int,
) -> Optional[Dict[str, Any]]:
    if src.size < int(min_samples):
        return None
    x_samples = x_state[src]
    y_samples = y_dot[src]
    finite = np.isfinite(x_samples).all(axis=1) & np.isfinite(y_samples).all(axis=1)
    x_samples = x_samples[finite]
    y_samples = y_samples[finite]
    src = src[finite]
    if x_samples.shape[0] < int(min_samples):
        return None
    x_mean = np.mean(x_samples, axis=0, keepdims=True)
    design = x_samples - x_mean
    col_scale = np.std(design, axis=0, ddof=0)
    col_scale = np.where(np.isfinite(col_scale) & (col_scale > 1e-8), col_scale, 1.0).astype(np.float32)
    design_std = design / col_scale[None, :]
    design_aug = np.hstack([design_std, np.ones((design.shape[0], 1), dtype=np.float32)])
    weights = None
    if distance_weighted:
        center_vec = np.asarray(x_metric[int(center)], dtype=np.float32).reshape(-1)
        metric_samples = np.asarray(x_metric[src], dtype=np.float32)
        if center_vec.size != metric_samples.shape[1] or not np.isfinite(center_vec).all():
            return None
        finite_m = np.isfinite(metric_samples).all(axis=1)
        if int(finite_m.sum()) < int(min_samples):
            return None
        d = np.linalg.norm(metric_samples - center_vec[None, :], axis=1)
        if not np.isfinite(d).all():
            return None
        d_pos = d[d > 0]
        sigma = float(np.median(d_pos)) if d_pos.size > 0 else float(np.median(d))
        sigma = sigma if np.isfinite(sigma) and sigma > 1e-6 else 1.0
        weights = np.exp(-0.5 * (d / sigma) ** 2).astype(np.float32)
        weights = weights / (float(np.mean(weights)) + 1e-8)
        if not np.isfinite(weights).all():
            return None
    a, b = _fit_ridge(design_aug, y_samples, ridge_alpha, sample_weights=weights)
    mapped = (a / col_scale[None, :]).astype(np.float32)
    if not np.isfinite(mapped).all():
        return None
    rel = _affine_rel_mse(x_samples, y_samples, mapped, b, x_mean.reshape(-1))
    return {
        "map": mapped,
        "affine_reference": x_mean.reshape(-1).astype(np.float32),
        "affine_intercept": b.astype(np.float32),
        "rel_mse_baseline": rel,
        "support_indices": src,
        "n_fit_samples": int(x_samples.shape[0]),
    }


def _holdout_cross_rel_mse(
    x_state: np.ndarray,
    y_dot: np.ndarray,
    src: np.ndarray,
    center: int,
    *,
    x_metric: np.ndarray,
    ridge_alpha: float,
    distance_weighted: bool,
    min_samples: int,
) -> Tuple[float, int]:
    train_src, hold_src = _holdout_split(src, OOS_HOLDOUT_STRIDE)
    if train_src.size < int(min_samples) or hold_src.size < 1:
        return float("nan"), 0
    oos_fit = _fit_cross_affine(
        x_state,
        y_dot,
        train_src,
        center,
        x_metric=x_metric,
        ridge_alpha=ridge_alpha,
        distance_weighted=distance_weighted,
        min_samples=min_samples,
    )
    if oos_fit is None:
        return float("nan"), 0
    hold_x = x_state[hold_src]
    hold_y = y_dot[hold_src]
    finite = np.isfinite(hold_x).all(axis=1) & np.isfinite(hold_y).all(axis=1)
    if int(finite.sum()) < 1:
        return float("nan"), 0
    rel = _affine_rel_mse(
        hold_x[finite],
        hold_y[finite],
        oos_fit["map"],
        oos_fit["affine_intercept"],
        oos_fit["affine_reference"],
    )
    return float(rel), int(finite.sum())


def estimate_local_cross_affine(
    x_state: np.ndarray,
    y_dot: np.ndarray,
    nn_idx: np.ndarray,
    *,
    super_window: int = 3,
    ridge_alpha: float = 1.0,
    distance_weighted: bool = False,
    knn_k: Optional[int] = None,
    support_mode: str = SUPPORT_MODE_KNN,
    x_metric: Optional[np.ndarray] = None,
) -> CrossAffineResult:
    """Fit ``ẏ ≈ B(x_state − x̄) + b`` on Jacobian support indices."""
    if x_state.ndim != 2 or y_dot.ndim != 2:
        raise ValueError("estimate_local_cross_affine expects 2D state and target arrays")
    if x_state.shape[0] != y_dot.shape[0]:
        raise ValueError("x_state and y_dot must share the time axis")
    state_dim = int(x_state.shape[1])
    target_dim = int(y_dot.shape[1])
    total = int(x_state.shape[0])
    super_window = max(1, int(super_window))
    if super_window % 2 == 0:
        super_window += 1
    support_mode = _normalize_support_mode(support_mode)
    effective_knn_k = 0 if support_mode == SUPPORT_MODE_TIME_LOCAL else infer_knn_k(nn_idx, knn_k)
    cfg = cross_affine_config_fields(
        state_dim=state_dim,
        target_dim=target_dim,
        knn_k=effective_knn_k,
        super_window=super_window,
        ridge_alpha=ridge_alpha,
        distance_weighted=distance_weighted,
        support_mode=support_mode,
    )
    min_samples = int(cfg["min_samples"])
    metric = x_state if x_metric is None else np.asarray(x_metric, dtype=np.float32)
    if metric.ndim != 2 or metric.shape[0] != total:
        raise ValueError("x_metric must be [T, D_metric] aligned with x_state")

    empty = CrossAffineResult(
        map_hat=np.zeros((0, target_dim, state_dim), dtype=np.float32),
        centers=np.zeros((0,), dtype=np.int32),
        diagnostics={
            "windows": 0,
            "failed": 0,
            "failed_insufficient_neighbours": 0,
            "failed_nonfinite_center": 0,
            **cfg,
            **summarize_neighborhood_sample_counts([]),
            **summarize_oos_rel_mse([], [], stride=OOS_HOLDOUT_STRIDE),
        },
        affine_reference=np.zeros((0, state_dim), dtype=np.float32),
        affine_intercept=np.zeros((0, target_dim), dtype=np.float32),
    )
    if x_state.size == 0 or y_dot.size == 0 or total < min_samples:
        return empty

    half = super_window // 2
    centers = np.arange(half, total - half, dtype=np.int32)
    maps = []
    centers_ok: list[int] = []
    failures = 0
    failed_insufficient = 0
    failed_nonfinite_center = 0
    rel_mse: list[float] = []
    rel_mse_oos: list[float] = []
    n_samples: list[int] = []
    n_holdout: list[int] = []
    refs: list[np.ndarray] = []
    intercepts: list[np.ndarray] = []

    for center in centers:
        gathered = _gather_support_indices(int(center), nn_idx, super_window, total, support_mode)
        if gathered.size < min_samples:
            failures += 1
            failed_insufficient += 1
            continue
        if distance_weighted and not np.isfinite(metric[int(center)]).all():
            failures += 1
            failed_nonfinite_center += 1
            continue
        fit = _fit_cross_affine(
            x_state,
            y_dot,
            gathered,
            int(center),
            x_metric=metric,
            ridge_alpha=ridge_alpha,
            distance_weighted=distance_weighted,
            min_samples=min_samples,
        )
        if fit is None:
            failures += 1
            failed_insufficient += 1
            continue
        maps.append(np.asarray(fit["map"], dtype=np.float32))
        centers_ok.append(int(center))
        rel_mse.append(float(fit["rel_mse_baseline"]))
        n_samples.append(int(fit["n_fit_samples"]))
        oos_rel, n_hold = _holdout_cross_rel_mse(
            x_state,
            y_dot,
            np.asarray(fit["support_indices"], dtype=np.int32),
            int(center),
            x_metric=metric,
            ridge_alpha=ridge_alpha,
            distance_weighted=distance_weighted,
            min_samples=min_samples,
        )
        rel_mse_oos.append(float(oos_rel))
        n_holdout.append(int(n_hold))
        refs.append(np.asarray(fit["affine_reference"], dtype=np.float32))
        intercepts.append(np.asarray(fit["affine_intercept"], dtype=np.float32))

    if not maps:
        empty.diagnostics.update(
            {
                "failed": float(failures),
                "failed_insufficient_neighbours": float(failed_insufficient),
                "failed_nonfinite_center": float(failed_nonfinite_center),
            }
        )
        return empty

    map_hat = np.stack(maps, axis=0)
    finite_rel = np.asarray(rel_mse, dtype=np.float64)
    finite_rel = finite_rel[np.isfinite(finite_rel)]
    diagnostics = {
        "windows": float(map_hat.shape[0]),
        "failed": float(failures),
        "failed_insufficient_neighbours": float(failed_insufficient),
        "failed_nonfinite_center": float(failed_nonfinite_center),
        "rel_mse_baseline_median": float(np.median(finite_rel)) if finite_rel.size else float("nan"),
        "rel_mse_baseline_windows": np.asarray(rel_mse, dtype=np.float32),
        **cfg,
        **summarize_neighborhood_sample_counts(n_samples),
        **summarize_oos_rel_mse(rel_mse_oos, n_holdout, stride=OOS_HOLDOUT_STRIDE),
    }
    return CrossAffineResult(
        map_hat=map_hat,
        centers=np.asarray(centers_ok, dtype=np.int32),
        diagnostics=diagnostics,
        affine_reference=np.stack(refs, axis=0),
        affine_intercept=np.stack(intercepts, axis=0),
    )
