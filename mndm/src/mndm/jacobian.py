"""Local Jacobian estimation for MNPS trajectories."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class JacobianResult:
    j_hat: np.ndarray
    j_dot: np.ndarray
    centers: np.ndarray
    diagnostics: Dict[str, Any]


def _gather_indices(center: int, nn_idx: np.ndarray, super_window: int, total: int) -> np.ndarray:
    half = super_window // 2
    candidates: list[np.ndarray] = []
    for offset in range(-half, half + 1):
        idx = center + offset
        if idx < 0 or idx >= total:
            continue
        candidates.append(np.array([idx], dtype=np.int32))
        if nn_idx.size > 0:
            candidates.append(np.asarray(nn_idx[idx], dtype=np.int32).ravel())
    if not candidates:
        return np.zeros((0,), dtype=np.int32)
    return np.unique(np.concatenate(candidates, axis=0))


def _fit_ridge(design: np.ndarray, target: np.ndarray, alpha: float, sample_weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    if sample_weights is not None:
        # Apply sqrt weights to rows: W^{1/2} X, W^{1/2} y
        w = np.sqrt(sample_weights).reshape(-1, 1).astype(np.float32)
        Xw = design * w
        yw = target * w
        xtx = Xw.T @ Xw
        xty = Xw.T @ yw
    else:
        xtx = design.T @ design
        xty = design.T @ target

    ridge = np.eye(xtx.shape[0], dtype=np.float32) * alpha
    ridge[-1, -1] = 0.0  # do not regularize intercept term

    try:
        coeff = np.linalg.solve(xtx + ridge, xty)
    except np.linalg.LinAlgError:
        coeff, *_ = np.linalg.lstsq(
            xtx + ridge + 1e-6 * np.eye(xtx.shape[0], dtype=np.float32),
            xty,
            rcond=None,
        )

    # General case: last row = intercept, remaining rows = linear terms
    a = coeff[:-1].T.astype(np.float32)
    b = coeff[-1].astype(np.float32)
    return a, b


def estimate_local_jacobians(
    x: np.ndarray,
    x_dot: np.ndarray,
    nn_idx: np.ndarray,
    super_window: int = 3,
    ridge_alpha: float = 1.0,
    distance_weighted: bool = False,
    j_dot_dt: Optional[float] = None,
) -> JacobianResult:
    """Estimate windowed Jacobians from MNPS trajectories."""
    if x.ndim != 2 or x_dot.ndim != 2:
        raise ValueError("estimate_local_jacobians expects 2D arrays for x and x_dot")
    if x.shape != x_dot.shape:
        raise ValueError("x and x_dot must have the same shape")

    dim = x.shape[1]

    if x.size == 0 or x_dot.size == 0:
        return JacobianResult(
            j_hat=np.zeros((0, dim, dim), dtype=np.float32),
            j_dot=np.zeros((0, dim, dim), dtype=np.float32),
            centers=np.zeros((0,), dtype=np.int32),
            diagnostics={"windows": 0, "failed": 0},
        )

    super_window = max(1, super_window)
    if super_window % 2 == 0:
        super_window += 1

    half = super_window // 2
    centers = np.arange(half, x.shape[0] - half, dtype=np.int32)
    j_list = []
    centers_ok: list[int] = []
    failures = 0
    local_fit_mse: list[float] = []
    local_fit_mse_baseline: list[float] = []
    rel_mse_baseline: list[float] = []

    for center in centers:
        neighbour_idx = _gather_indices(center, nn_idx, super_window, x.shape[0])
        if neighbour_idx.size < dim + 1:  # minimum to solve dim params + intercept
            failures += 1
            continue

        x_samples = x[neighbour_idx]
        xdot_samples = x_dot[neighbour_idx]
        # Critical for NaN-aware pipelines: only fit on rows where both x and x_dot are finite.
        finite_mask = np.isfinite(x_samples).all(axis=1) & np.isfinite(xdot_samples).all(axis=1)
        x_samples = x_samples[finite_mask]
        xdot_samples = xdot_samples[finite_mask]
        if x_samples.shape[0] < dim + 1:
            failures += 1
            continue

        x_mean = np.mean(x_samples, axis=0, keepdims=True)
        design = x_samples - x_mean
        # Standardize local design to keep ridge strength comparable across scale-shifted runs.
        # We later map coefficients back to the original units.
        col_scale = np.std(design, axis=0, ddof=0)
        col_scale = np.where(np.isfinite(col_scale) & (col_scale > 1e-8), col_scale, 1.0).astype(np.float32)
        design_std = design / col_scale[None, :]
        ones = np.ones((design.shape[0], 1), dtype=np.float32)
        design_aug = np.hstack([design_std, ones])

        # Optional distance weights (closer points weigh more)
        weights = None
        if distance_weighted:
            center_vec = x[center]
            d = np.linalg.norm(x_samples - center_vec[None, :], axis=1)
            # Use a smooth RBF kernel instead of inverse distance:
            # inverse distance over-emphasizes d=0 (the center sample) and can collapse the fit.
            d_pos = d[d > 0]
            sigma = float(np.median(d_pos)) if d_pos.size > 0 else float(np.median(d))
            if not np.isfinite(sigma) or sigma <= 1e-6:
                sigma = 1.0
            weights = np.exp(-0.5 * (d / sigma) ** 2).astype(np.float32)
            # Keep effective sample size sane by normalizing around mean weight.
            weights = weights / (float(np.mean(weights)) + 1e-8)

        a, b = _fit_ridge(design_aug, xdot_samples, ridge_alpha, sample_weights=weights)
        # Local fit residual diagnostics for auditability:
        # compare fitted local linear model against an intercept-only baseline.
        y_hat = design_aug @ np.vstack([a.T, b[None, :]])
        residual = xdot_samples - y_hat
        mse_model = float(np.mean(residual ** 2))
        y0 = np.mean(xdot_samples, axis=0, keepdims=True)
        residual_baseline = xdot_samples - y0
        mse_baseline = float(np.mean(residual_baseline ** 2))
        rel_mse = float("nan")
        if np.isfinite(mse_baseline) and mse_baseline > 1e-12:
            rel_mse = float(mse_model / mse_baseline)
        a_orig = (a / col_scale[None, :]).astype(np.float32)
        j_list.append(a_orig)
        centers_ok.append(int(center))
        local_fit_mse.append(mse_model)
        local_fit_mse_baseline.append(mse_baseline)
        rel_mse_baseline.append(rel_mse)

    if not j_list:
        return JacobianResult(
            j_hat=np.zeros((0, dim, dim), dtype=np.float32),
            j_dot=np.zeros((0, dim, dim), dtype=np.float32),
            centers=np.zeros((0,), dtype=np.int32),
            diagnostics={"windows": 0, "failed": float(failures)},
        )

    j_hat = np.stack(j_list, axis=0)
    # j_dot is computed as a centered finite-difference gradient along window order.
    # If j_dot_dt is provided (>0), it is interpreted as seconds per Jacobian step.
    if j_hat.shape[0] > 1:
        spacing = float(j_dot_dt) if (j_dot_dt is not None and np.isfinite(j_dot_dt) and j_dot_dt > 0) else 1.0
        j_dot = np.gradient(j_hat, spacing, axis=0).astype(np.float32)
    else:
        j_dot = np.zeros_like(j_hat, dtype=np.float32)

    diagnostics = {
        "windows": float(j_hat.shape[0]),
        "failed": float(failures),
        "j_dot_mode": "gradient",
        "j_dot_dt": float(j_dot_dt) if (j_dot_dt is not None and np.isfinite(j_dot_dt) and j_dot_dt > 0) else 1.0,
        "local_fit_mse_median": float(np.nanmedian(np.asarray(local_fit_mse, dtype=np.float64))) if local_fit_mse else float("nan"),
        "local_fit_mse_baseline_median": float(np.nanmedian(np.asarray(local_fit_mse_baseline, dtype=np.float64))) if local_fit_mse_baseline else float("nan"),
        "rel_mse_baseline_median": float(np.nanmedian(np.asarray(rel_mse_baseline, dtype=np.float64))) if rel_mse_baseline else float("nan"),
        "local_fit_mse_windows": np.asarray(local_fit_mse, dtype=np.float32),
        "local_fit_mse_baseline_windows": np.asarray(local_fit_mse_baseline, dtype=np.float32),
        "rel_mse_baseline_windows": np.asarray(rel_mse_baseline, dtype=np.float32),
    }

    return JacobianResult(
        j_hat=j_hat,
        j_dot=j_dot,
        centers=np.asarray(centers_ok, dtype=np.int32),
        diagnostics=diagnostics,
    )


def phase_randomise(x: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fft = np.fft.rfft(x, axis=0)
    phases = rng.uniform(0, 2 * np.pi, size=fft.shape)
    # Preserve DC (and Nyquist for even-length signals) as real-valued bins.
    phases[0] = 0.0
    if x.shape[0] % 2 == 0:
        phases[-1] = 0.0
    fft_random = np.abs(fft) * np.exp(1j * phases)
    # Keep original DC sign/magnitude to preserve channel-wise means.
    fft_random[0] = fft[0]
    if x.shape[0] % 2 == 0:
        fft_random[-1] = fft[-1]
    return np.fft.irfft(fft_random, n=x.shape[0], axis=0).astype(x.dtype)


def window_shuffle(x: np.ndarray, window: int, seed: Optional[int] = None) -> np.ndarray:
    if window <= 1:
        return x.copy()
    rng = np.random.default_rng(seed)
    num_windows = x.shape[0] // window
    reshaped = x[: num_windows * window].reshape(num_windows, window, -1)
    rng.shuffle(reshaped, axis=0)
    shuffled = reshaped.reshape(-1, x.shape[1])
    remainder = x[num_windows * window :]
    return np.vstack([shuffled, remainder]) if remainder.size else shuffled


