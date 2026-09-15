"""Experimental discrete one-step maps on MNPS trajectories.

This is a different estimator family from ``estimate_local_jacobians``: it
fits ``x_{t+h} ≈ Φ(x_t - x̄) + b`` from coordinate pairs and does not use
``x_dot``. Each support index ``i`` is paired with ``i + horizon``, not
with the center's next state. In-sample next-state ``rel_mse`` is a
different residual from the production ẋ-affine gate; do not treat a
value below 0.9 as Family B ``computed``.

The estimator is for synthetic identification tests and bounded replays.
It is not the production Jacobian and is not wired into summarize.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .jacobian import (
    OOS_HOLDOUT_STRIDE,
    SUPPORT_MODE_KNN,
    SUPPORT_MODE_TIME_LOCAL,
    _affine_rel_mse,
    _fit_ridge,
    _gather_support_indices,
    _holdout_split,
    _normalize_support_mode,
    infer_knn_k,
    neighborhood_config_fields,
    summarize_neighborhood_sample_counts,
    summarize_oos_rel_mse,
)


@dataclass
class DiscreteMapResult:
    phi_hat: np.ndarray
    euler_j: np.ndarray
    centers: np.ndarray
    diagnostics: Dict[str, Any]
    affine_reference: Optional[np.ndarray] = None
    affine_intercept: Optional[np.ndarray] = None


def _pair_indices(indices: np.ndarray, horizon: int, total: int) -> Tuple[np.ndarray, np.ndarray]:
    src = np.asarray(indices, dtype=np.int32).ravel()
    src = src[(src >= 0) & (src + int(horizon) < int(total))]
    return src, (src + int(horizon)).astype(np.int32)


def _fit_discrete_pairs(
    x: np.ndarray,
    src: np.ndarray,
    tgt: np.ndarray,
    center: int,
    *,
    ridge_alpha: float,
    distance_weighted: bool,
    min_samples: int,
) -> Optional[Dict[str, Any]]:
    if src.size < int(min_samples):
        return None
    x_src = x[src]
    x_tgt = x[tgt]
    finite = np.isfinite(x_src).all(axis=1) & np.isfinite(x_tgt).all(axis=1)
    x_src = x_src[finite]
    x_tgt = x_tgt[finite]
    src = src[finite]
    if x_src.shape[0] < int(min_samples):
        return None
    x_mean = np.mean(x_src, axis=0, keepdims=True)
    design = x_src - x_mean
    col_scale = np.std(design, axis=0, ddof=0)
    col_scale = np.where(np.isfinite(col_scale) & (col_scale > 1e-8), col_scale, 1.0).astype(np.float32)
    design_std = design / col_scale[None, :]
    design_aug = np.hstack([design_std, np.ones((design.shape[0], 1), dtype=np.float32)])
    weights = None
    if distance_weighted:
        center_vec = np.asarray(x[int(center)], dtype=np.float32).reshape(-1)
        if center_vec.size != x_src.shape[1] or not np.isfinite(center_vec).all():
            return None
        d = np.linalg.norm(x_src - center_vec[None, :], axis=1)
        d_pos = d[d > 0]
        sigma = float(np.median(d_pos)) if d_pos.size > 0 else float(np.median(d))
        sigma = sigma if np.isfinite(sigma) and sigma > 1e-6 else 1.0
        weights = np.exp(-0.5 * (d / sigma) ** 2).astype(np.float32)
        weights = weights / (float(np.mean(weights)) + 1e-8)
        if not np.isfinite(weights).all():
            return None
    a, b = _fit_ridge(design_aug, x_tgt, ridge_alpha, sample_weights=weights)
    phi = (a / col_scale[None, :]).astype(np.float32)
    rel = _affine_rel_mse(x_src, x_tgt, phi, b, x_mean.reshape(-1))
    return {
        "phi": phi,
        "affine_reference": x_mean.reshape(-1).astype(np.float32),
        "affine_intercept": b.astype(np.float32),
        "rel_mse_baseline": rel,
        "support_indices": src,
        "n_fit_samples": int(x_src.shape[0]),
    }


def _holdout_discrete_rel_mse(
    x: np.ndarray,
    src: np.ndarray,
    horizon: int,
    center: int,
    *,
    ridge_alpha: float,
    distance_weighted: bool,
    min_samples: int,
) -> Tuple[float, int]:
    train_src, hold_src = _holdout_split(src, OOS_HOLDOUT_STRIDE)
    if train_src.size < int(min_samples) or hold_src.size < 1:
        return float("nan"), 0
    train_tgt = train_src + int(horizon)
    hold_tgt = hold_src + int(horizon)
    oos_fit = _fit_discrete_pairs(
        x,
        train_src,
        train_tgt,
        center,
        ridge_alpha=ridge_alpha,
        distance_weighted=distance_weighted,
        min_samples=min_samples,
    )
    if oos_fit is None:
        return float("nan"), 0
    hold_x = x[hold_src]
    hold_y = x[hold_tgt]
    finite = np.isfinite(hold_x).all(axis=1) & np.isfinite(hold_y).all(axis=1)
    if int(finite.sum()) < 1:
        return float("nan"), 0
    rel = _affine_rel_mse(
        hold_x[finite],
        hold_y[finite],
        oos_fit["phi"],
        oos_fit["affine_intercept"],
        oos_fit["affine_reference"],
    )
    return float(rel), int(finite.sum())


def estimate_local_discrete_maps(
    x: np.ndarray,
    nn_idx: np.ndarray,
    *,
    super_window: int = 3,
    ridge_alpha: float = 1.0,
    distance_weighted: bool = False,
    knn_k: Optional[int] = None,
    support_mode: str = SUPPORT_MODE_KNN,
    horizon: int = 1,
    dt_sec: float = 1.0,
) -> DiscreteMapResult:
    """Fit local affine one-step maps ``x_{t+h} ≈ Φ(x_t - x̄) + b``.

    ``euler_j = (Φ - I) / (h Δt)`` is a discrete-to-generator conversion for
    comparison only. It is not a production continuous-time Jacobian.
    """
    if x.ndim != 2:
        raise ValueError("estimate_local_discrete_maps expects a 2D state array")
    dim = int(x.shape[1])
    total = int(x.shape[0])
    horizon = max(1, int(horizon))
    dt = float(dt_sec) if np.isfinite(dt_sec) and float(dt_sec) > 0 else 1.0
    super_window = max(1, int(super_window))
    if super_window % 2 == 0:
        super_window += 1
    support_mode = _normalize_support_mode(support_mode)
    effective_knn_k = 0 if support_mode == SUPPORT_MODE_TIME_LOCAL else infer_knn_k(nn_idx, knn_k)
    neighborhood_cfg = neighborhood_config_fields(
        knn_k=effective_knn_k,
        super_window=super_window,
        ridge_alpha=ridge_alpha,
        distance_weighted=distance_weighted,
        dim=dim,
        support_mode=support_mode,
    )
    min_samples = int(neighborhood_cfg["min_samples"])
    family_cfg = {
        **neighborhood_cfg,
        "estimator_family": "discrete_one_step_map",
        "horizon": int(horizon),
        "dt_sec": float(dt),
    }

    empty = DiscreteMapResult(
        phi_hat=np.zeros((0, dim, dim), dtype=np.float32),
        euler_j=np.zeros((0, dim, dim), dtype=np.float32),
        centers=np.zeros((0,), dtype=np.int32),
        diagnostics={
            "windows": 0,
            "failed": 0,
            "failed_insufficient_pairs": 0,
            "failed_nonfinite_center": 0,
            **family_cfg,
            **summarize_neighborhood_sample_counts([]),
            **summarize_oos_rel_mse([], [], stride=OOS_HOLDOUT_STRIDE),
        },
        affine_reference=np.zeros((0, dim), dtype=np.float32),
        affine_intercept=np.zeros((0, dim), dtype=np.float32),
    )
    if x.size == 0 or total < horizon + min_samples:
        return empty

    half = super_window // 2
    centers = np.arange(half, total - half, dtype=np.int32)
    phi_list = []
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
        src, tgt = _pair_indices(gathered, horizon, total)
        if src.size < min_samples:
            failures += 1
            failed_insufficient += 1
            continue
        if distance_weighted and not np.isfinite(x[int(center)]).all():
            failures += 1
            failed_nonfinite_center += 1
            continue
        fit = _fit_discrete_pairs(
            x,
            src,
            tgt,
            int(center),
            ridge_alpha=ridge_alpha,
            distance_weighted=distance_weighted,
            min_samples=min_samples,
        )
        if fit is None or not np.isfinite(np.asarray(fit["phi"])).all():
            failures += 1
            failed_insufficient += 1
            continue
        phi_list.append(np.asarray(fit["phi"], dtype=np.float32))
        centers_ok.append(int(center))
        rel_mse.append(float(fit["rel_mse_baseline"]))
        n_samples.append(int(fit["n_fit_samples"]))
        oos_rel, n_hold = _holdout_discrete_rel_mse(
            x,
            np.asarray(fit["support_indices"], dtype=np.int32),
            horizon,
            int(center),
            ridge_alpha=ridge_alpha,
            distance_weighted=distance_weighted,
            min_samples=min_samples,
        )
        rel_mse_oos.append(float(oos_rel))
        n_holdout.append(int(n_hold))
        refs.append(np.asarray(fit["affine_reference"], dtype=np.float32))
        intercepts.append(np.asarray(fit["affine_intercept"], dtype=np.float32))

    if not phi_list:
        empty.diagnostics.update(
            {
                "failed": float(failures),
                "failed_insufficient_pairs": float(failed_insufficient),
                "failed_nonfinite_center": float(failed_nonfinite_center),
            }
        )
        return empty

    phi_hat = np.stack(phi_list, axis=0)
    eye = np.eye(dim, dtype=np.float32)
    euler_j = ((phi_hat - eye[None, :, :]) / float(horizon * dt)).astype(np.float32)
    finite_rel = np.asarray(rel_mse, dtype=np.float64)
    finite_rel = finite_rel[np.isfinite(finite_rel)]
    diagnostics = {
        "windows": float(phi_hat.shape[0]),
        "failed": float(failures),
        "failed_insufficient_pairs": float(failed_insufficient),
        "failed_nonfinite_center": float(failed_nonfinite_center),
        "rel_mse_baseline_median": float(np.median(finite_rel)) if finite_rel.size else float("nan"),
        "rel_mse_baseline_windows": np.asarray(rel_mse, dtype=np.float32),
        **family_cfg,
        **summarize_neighborhood_sample_counts(n_samples),
        **summarize_oos_rel_mse(rel_mse_oos, n_holdout, stride=OOS_HOLDOUT_STRIDE),
    }
    return DiscreteMapResult(
        phi_hat=phi_hat,
        euler_j=euler_j,
        centers=np.asarray(centers_ok, dtype=np.int32),
        diagnostics=diagnostics,
        affine_reference=np.stack(refs, axis=0),
        affine_intercept=np.stack(intercepts, axis=0),
    )
