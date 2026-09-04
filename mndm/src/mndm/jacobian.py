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
    affine_reference: Optional[np.ndarray] = None
    affine_intercept: Optional[np.ndarray] = None


def _gather_indices(center: int, nn_idx: np.ndarray, super_window: int, total: int) -> np.ndarray:
    """Internal helper: gather indices."""
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
    """Internal helper: fit ridge."""
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


def fit_local_affine_at_center(
    x: np.ndarray,
    x_dot: np.ndarray,
    nn_idx: np.ndarray,
    center: int,
    *,
    super_window: int,
    ridge_alpha: float,
    distance_weighted: bool,
    exclude_indices: Optional[Sequence[int]] = None,
    neighbour_indices: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    """Fit the local affine derivative model at one center.

    ``exclude_indices`` is used only by transition-residual cross-fitting. It
    never changes the canonical Jacobian estimate returned by
    :func:`estimate_local_jacobians`.
    """
    dim = int(x.shape[1])
    if neighbour_indices is None:
        neighbours = _gather_indices(int(center), nn_idx, int(super_window), x.shape[0])
    else:
        neighbours = np.asarray(neighbour_indices, dtype=np.int32).ravel()
    if exclude_indices is not None and len(exclude_indices) > 0:
        excluded = np.asarray(list(exclude_indices), dtype=np.int32)
        neighbours = neighbours[~np.isin(neighbours, excluded)]
    if neighbours.size < dim + 1:
        return None
    x_samples = x[neighbours]
    xdot_samples = x_dot[neighbours]
    finite_mask = np.isfinite(x_samples).all(axis=1) & np.isfinite(xdot_samples).all(axis=1)
    x_samples = x_samples[finite_mask]
    xdot_samples = xdot_samples[finite_mask]
    if x_samples.shape[0] < dim + 1:
        return None

    x_mean = np.mean(x_samples, axis=0, keepdims=True)
    design = x_samples - x_mean
    col_scale = np.std(design, axis=0, ddof=0)
    col_scale = np.where(np.isfinite(col_scale) & (col_scale > 1e-8), col_scale, 1.0).astype(np.float32)
    design_std = design / col_scale[None, :]
    design_aug = np.hstack([design_std, np.ones((design.shape[0], 1), dtype=np.float32)])
    weights = None
    if distance_weighted:
        center_vec = x[int(center)]
        d = np.linalg.norm(x_samples - center_vec[None, :], axis=1)
        d_pos = d[d > 0]
        sigma = float(np.median(d_pos)) if d_pos.size > 0 else float(np.median(d))
        sigma = sigma if np.isfinite(sigma) and sigma > 1e-6 else 1.0
        weights = np.exp(-0.5 * (d / sigma) ** 2).astype(np.float32)
        weights = weights / (float(np.mean(weights)) + 1e-8)
    a, b = _fit_ridge(design_aug, xdot_samples, ridge_alpha, sample_weights=weights)
    y_hat = design_aug @ np.vstack([a.T, b[None, :]])
    residual = xdot_samples - y_hat
    mse_model = float(np.mean(residual**2))
    baseline = xdot_samples - np.mean(xdot_samples, axis=0, keepdims=True)
    mse_baseline = float(np.mean(baseline**2))
    return {
        "jacobian": (a / col_scale[None, :]).astype(np.float32),
        "affine_reference": x_mean.reshape(-1).astype(np.float32),
        "affine_intercept": b.astype(np.float32),
        "mse_model": mse_model,
        "mse_baseline": mse_baseline,
        "rel_mse_baseline": float(mse_model / mse_baseline) if np.isfinite(mse_baseline) and mse_baseline > 1e-12 else float("nan"),
        "support_indices": neighbours,
    }


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
    failed_centers: list[int] = []
    failed_insufficient_neighbours = 0
    failed_nonfinite_samples = 0
    local_fit_mse: list[float] = []
    local_fit_mse_baseline: list[float] = []
    rel_mse_baseline: list[float] = []
    affine_reference_list: list[np.ndarray] = []
    affine_intercept_list: list[np.ndarray] = []

    for center in centers:
        neighbour_idx = _gather_indices(center, nn_idx, super_window, x.shape[0])
        if neighbour_idx.size < dim + 1:  # minimum to solve dim params + intercept
            failures += 1
            failed_centers.append(int(center))
            failed_insufficient_neighbours += 1
            continue
        fit = fit_local_affine_at_center(
            x,
            x_dot,
            nn_idx,
            int(center),
            super_window=super_window,
            ridge_alpha=ridge_alpha,
            distance_weighted=distance_weighted,
            neighbour_indices=neighbour_idx,
        )
        if fit is None:
            failures += 1
            failed_centers.append(int(center))
            failed_nonfinite_samples += 1
            continue
        j_list.append(np.asarray(fit["jacobian"], dtype=np.float32))
        centers_ok.append(int(center))
        local_fit_mse.append(float(fit["mse_model"]))
        local_fit_mse_baseline.append(float(fit["mse_baseline"]))
        rel_mse_baseline.append(float(fit["rel_mse_baseline"]))
        affine_reference_list.append(np.asarray(fit["affine_reference"], dtype=np.float32))
        affine_intercept_list.append(np.asarray(fit["affine_intercept"], dtype=np.float32))

    if not j_list:
        return JacobianResult(
            j_hat=np.zeros((0, dim, dim), dtype=np.float32),
            j_dot=np.zeros((0, dim, dim), dtype=np.float32),
            centers=np.zeros((0,), dtype=np.int32),
            diagnostics={
                "windows": 0,
                "failed": float(failures),
                "attempted_centers": np.asarray(centers, dtype=np.int32),
                "failed_centers": np.asarray(failed_centers, dtype=np.int32),
                "failed_insufficient_neighbours": float(failed_insufficient_neighbours),
                "failed_nonfinite_samples": float(failed_nonfinite_samples),
                "condition_number_windows": np.zeros((0,), dtype=np.float64),
            },
        affine_reference=np.zeros((0, dim), dtype=np.float32),
        affine_intercept=np.zeros((0, dim), dtype=np.float32),
        )

    j_hat = np.stack(j_list, axis=0)
    condition_number = np.full((j_hat.shape[0],), np.nan, dtype=np.float64)
    try:
        svals = np.linalg.svd(j_hat.astype(np.float64), compute_uv=False)
        smin = np.min(svals, axis=1)
        smax = np.max(svals, axis=1)
        ok = np.isfinite(smin) & np.isfinite(smax) & (smin > 0)
        condition_number[ok] = np.asarray(smax[ok] / smin[ok], dtype=np.float64)
    except Exception:
        logger.exception("Failed vectorized SVD for per-window Jacobian condition numbers")

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
        "rel_mse_baseline_median": (
            float(np.median(np.asarray(rel_mse_baseline, dtype=np.float64)[np.isfinite(np.asarray(rel_mse_baseline, dtype=np.float64))]))
            if np.any(np.isfinite(np.asarray(rel_mse_baseline, dtype=np.float64)))
            else float("nan")
        ),
        "local_fit_mse_windows": np.asarray(local_fit_mse, dtype=np.float32),
        "local_fit_mse_baseline_windows": np.asarray(local_fit_mse_baseline, dtype=np.float32),
        "rel_mse_baseline_windows": np.asarray(rel_mse_baseline, dtype=np.float32),
        "condition_number_windows": condition_number,
        "affine_reference_windows": np.stack(affine_reference_list, axis=0).astype(np.float32),
        "affine_intercept_windows": np.stack(affine_intercept_list, axis=0).astype(np.float32),
        "attempted_centers": np.asarray(centers, dtype=np.int32),
        "failed_centers": np.asarray(failed_centers, dtype=np.int32),
        "failed_insufficient_neighbours": float(failed_insufficient_neighbours),
        "failed_nonfinite_samples": float(failed_nonfinite_samples),
    }

    return JacobianResult(
        j_hat=j_hat,
        j_dot=j_dot,
        centers=np.asarray(centers_ok, dtype=np.int32),
        diagnostics=diagnostics,
        affine_reference=np.stack(affine_reference_list, axis=0),
        affine_intercept=np.stack(affine_intercept_list, axis=0),
    )


def estimate_anchor_coupling(
    x: np.ndarray,
    x_dot: np.ndarray,
    anchor_state: np.ndarray,
    anchor_state_dot: np.ndarray,
    nn_idx: np.ndarray,
    *,
    super_window: int = 3,
    ridge_alpha: float = 1.0,
    distance_weighted: bool = False,
    j_dot_dt: Optional[float] = None,
) -> Dict[str, Any]:
    """Estimate additive body-brain coupling blocks from a joint local model."""
    x_arr = np.asarray(x, dtype=np.float32)
    xdot_arr = np.asarray(x_dot, dtype=np.float32)
    a_arr = np.asarray(anchor_state, dtype=np.float32)
    adot_arr = np.asarray(anchor_state_dot, dtype=np.float32)
    if x_arr.ndim != 2 or xdot_arr.ndim != 2 or a_arr.ndim != 2 or adot_arr.ndim != 2:
        return {}
    if x_arr.shape != xdot_arr.shape or a_arr.shape != adot_arr.shape:
        return {}
    if x_arr.shape[0] != a_arr.shape[0] or x_arr.shape[0] == 0:
        return {}

    z = np.concatenate([x_arr, a_arr], axis=1).astype(np.float32, copy=False)
    z_dot = np.concatenate([xdot_arr, adot_arr], axis=1).astype(np.float32, copy=False)
    result = estimate_local_jacobians(
        z,
        z_dot,
        nn_idx,
        super_window=super_window,
        ridge_alpha=ridge_alpha,
        distance_weighted=distance_weighted,
        j_dot_dt=j_dot_dt,
    )
    if result.j_hat.size == 0:
        return {}

    x_dim = int(x_arr.shape[1])
    a_dim = int(a_arr.shape[1])
    j_hat = np.asarray(result.j_hat, dtype=np.float32)
    j_dot = np.asarray(result.j_dot, dtype=np.float32)
    j_xa = j_hat[:, :x_dim, x_dim:]
    j_ax = j_hat[:, x_dim:, :x_dim]
    j_xa_dot = j_dot[:, :x_dim, x_dim:]
    j_ax_dot = j_dot[:, x_dim:, :x_dim]

    forward_drive = np.linalg.norm(j_xa.astype(np.float64), axis=(1, 2))
    reverse_drive = np.linalg.norm(j_ax.astype(np.float64), axis=(1, 2))
    denom = forward_drive + reverse_drive
    asymmetry = np.full_like(forward_drive, np.nan, dtype=np.float64)
    valid = np.isfinite(denom) & (denom > 1e-8)
    asymmetry[valid] = (forward_drive[valid] - reverse_drive[valid]) / denom[valid]
    rotational_exchange = np.linalg.norm(
        j_xa.astype(np.float64) - np.swapaxes(j_ax.astype(np.float64), 1, 2),
        axis=(1, 2),
    )
    metrics = np.stack(
        [
            forward_drive.astype(np.float32),
            reverse_drive.astype(np.float32),
            asymmetry.astype(np.float32),
            rotational_exchange.astype(np.float32),
        ],
        axis=1,
    )
    metric_names = [
        "forward_drive_fro",
        "reverse_drive_fro",
        "directional_asymmetry",
        "rotational_exchange",
    ]
    diagnostics = dict(result.diagnostics)
    diagnostics.update(
        {
            "schema": "mndm.anchor_coupling.v1",
            "x_dim": x_dim,
            "anchor_dim": a_dim,
            "joint_dim": int(x_dim + a_dim),
            "metric_names": list(metric_names),
            "forward_drive_median": float(np.nanmedian(forward_drive)) if forward_drive.size else float("nan"),
            "reverse_drive_median": float(np.nanmedian(reverse_drive)) if reverse_drive.size else float("nan"),
            "directional_asymmetry_median": float(np.nanmedian(asymmetry)) if asymmetry.size else float("nan"),
            "rotational_exchange_median": (
                float(np.nanmedian(rotational_exchange)) if rotational_exchange.size else float("nan")
            ),
        }
    )
    return {
        "J_z": j_hat,
        "J_z_dot": j_dot,
        "J_xa": j_xa.astype(np.float32),
        "J_ax": j_ax.astype(np.float32),
        "J_xa_dot": j_xa_dot.astype(np.float32),
        "J_ax_dot": j_ax_dot.astype(np.float32),
        "centers": np.asarray(result.centers, dtype=np.int32),
        "metrics": metrics.astype(np.float32),
        "metric_names": metric_names,
        "diagnostics": diagnostics,
    }


def phase_randomise(x: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """Handle phase randomise."""
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
    """Handle window shuffle."""
    if window <= 1:
        return x.copy()
    rng = np.random.default_rng(seed)
    num_windows = x.shape[0] // window
    reshaped = x[: num_windows * window].reshape(num_windows, window, -1)
    rng.shuffle(reshaped, axis=0)
    shuffled = reshaped.reshape(-1, x.shape[1])
    remainder = x[num_windows * window :]
    return np.vstack([shuffled, remainder]) if remainder.size else shuffled


