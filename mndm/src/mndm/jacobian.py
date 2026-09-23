"""Local Jacobian estimation for MNPS trajectories."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

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


SUPPORT_MODE_KNN = "knn"
SUPPORT_MODE_TIME_LOCAL = "time_local"
VALID_SUPPORT_MODES: Tuple[str, ...] = (SUPPORT_MODE_KNN, SUPPORT_MODE_TIME_LOCAL)

NEIGHBORHOOD_PROVENANCE_KEYS: Tuple[str, ...] = (
    "support_mode",
    "knn_k",
    "super_window",
    "ridge_alpha",
    "distance_weighted",
    "min_samples",
    "n_neighborhood_samples_median",
    "n_neighborhood_samples_min",
    "n_affine_parameters",
    "super_window_requested",
    "require_determined_support",
)

OOS_HOLDOUT_STRIDE = 4
OOS_PROVENANCE_KEYS: Tuple[str, ...] = (
    "rel_mse_baseline_oos_median",
    "oos_holdout_stride",
)


def infer_knn_k(nn_idx: np.ndarray, knn_k: Optional[int] = None) -> int:
    """Return the chart-neighborhood width actually present in ``nn_idx``.

    ``build_knn_indices`` may cap requested ``k`` to ``T-1``. Provenance must
    record that effective width, not the possibly larger requested ``knn_k``.
    """
    arr = np.asarray(nn_idx)
    if arr.ndim == 2:
        return int(arr.shape[1])
    if knn_k is not None:
        return int(knn_k)
    return 0


def neighborhood_config_fields(
    *,
    knn_k: int,
    super_window: int,
    ridge_alpha: float,
    distance_weighted: bool,
    dim: int,
    support_mode: str = SUPPORT_MODE_KNN,
) -> Dict[str, Any]:
    """Estimator settings that determine Jacobian neighborhood support."""
    n_params = int(dim) * (int(dim) + 1)
    min_samples = int(dim) + 1
    if str(support_mode) == SUPPORT_MODE_TIME_LOCAL:
        # In-sample rel_mse on fewer rows than affine parameters is an overfit
        # trap (3D: 7 samples vs 12 parameters can beat baseline on noise).
        min_samples = max(min_samples, n_params)
    return {
        "support_mode": str(support_mode),
        "knn_k": int(knn_k),
        "super_window": int(super_window),
        "ridge_alpha": float(ridge_alpha),
        "distance_weighted": bool(distance_weighted),
        "min_samples": int(min_samples),
        "n_affine_parameters": n_params,
    }


def summarize_neighborhood_sample_counts(counts: Sequence[int]) -> Dict[str, Any]:
    """Summarize per-window unique samples actually used in the affine fit."""
    arr = np.asarray(list(counts), dtype=np.int32)
    if arr.size == 0:
        return {
            "n_neighborhood_samples": arr,
            "n_neighborhood_samples_median": float("nan"),
            "n_neighborhood_samples_min": float("nan"),
        }
    values = arr.astype(np.float64)
    return {
        "n_neighborhood_samples": arr,
        "n_neighborhood_samples_median": float(np.median(values)),
        "n_neighborhood_samples_min": float(np.min(values)),
    }


def neighborhood_support_provenance(diagnostics: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Copy neighborhood estimator settings into Jacobian-metrics provenance."""
    if not isinstance(diagnostics, Mapping):
        return {}
    out: Dict[str, Any] = {}
    for key in NEIGHBORHOOD_PROVENANCE_KEYS + OOS_PROVENANCE_KEYS:
        if key not in diagnostics:
            continue
        value = diagnostics[key]
        if isinstance(value, np.bool_):
            out[key] = bool(value)
        elif isinstance(value, np.floating):
            out[key] = float(value)
        elif isinstance(value, np.integer):
            out[key] = int(value)
        else:
            out[key] = value
    return out


def _gather_indices(center: int, nn_idx: np.ndarray, super_window: int, total: int) -> np.ndarray:
    """Gather unique chart-kNN indices from ``super_window`` time neighbors."""
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


def _gather_time_local_indices(center: int, super_window: int, total: int) -> np.ndarray:
    """Gather a contiguous time window around ``center`` (no chart kNN).

    ``super_window`` is the same odd length used by the kNN estimator: the
    gathered set is ``[center - half, center + half]`` clipped to ``[0, total)``.
    This is an experimental support mode for mixing-vs-identification tests.
    It is not the production Jacobian neighborhood.
    """
    half = max(0, int(super_window) // 2)
    lo = max(0, int(center) - half)
    hi = min(int(total), int(center) + half + 1)
    if hi <= lo:
        return np.zeros((0,), dtype=np.int32)
    return np.arange(lo, hi, dtype=np.int32)


def _normalize_support_mode(support_mode: Optional[str]) -> str:
    mode = str(support_mode or SUPPORT_MODE_KNN).strip().lower()
    if mode not in VALID_SUPPORT_MODES:
        raise ValueError(
            f"Unsupported Jacobian support_mode={support_mode!r}; "
            f"expected one of {VALID_SUPPORT_MODES}"
        )
    return mode


def summarize_oos_rel_mse(
    values: Sequence[float],
    holdout_counts: Sequence[int],
    *,
    stride: int = OOS_HOLDOUT_STRIDE,
) -> Dict[str, Any]:
    """Summarize per-window holdout relative MSE (does not alter J_hat)."""
    arr = np.asarray(list(values), dtype=np.float32)
    counts = np.asarray(list(holdout_counts), dtype=np.int32)
    finite = arr.astype(np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        "rel_mse_baseline_oos_windows": arr,
        "rel_mse_baseline_oos_median": float(np.median(finite)) if finite.size else float("nan"),
        "n_holdout_samples": counts,
        "oos_holdout_stride": int(stride),
    }


def _holdout_split(indices: np.ndarray, stride: int = OOS_HOLDOUT_STRIDE) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic train/holdout split of sorted unique support indices."""
    order = np.unique(np.asarray(indices, dtype=np.int32))
    if order.size == 0:
        empty = np.zeros((0,), dtype=np.int32)
        return empty, empty
    ranks = np.arange(order.size)
    holdout_mask = (ranks % max(int(stride), 2)) == (max(int(stride), 2) - 1)
    return order[~holdout_mask], order[holdout_mask]


def _affine_rel_mse(
    x_eval: np.ndarray,
    xdot_eval: np.ndarray,
    jacobian: np.ndarray,
    intercept: np.ndarray,
    reference: np.ndarray,
) -> float:
    """Relative MSE of an affine map vs mean-ẋ on an evaluation set.

    The baseline is the mean of ``xdot_eval`` (the scored points), matching
    the in-sample ``rel_mse_baseline`` definition. This is test-set
    mean-normalized scoring, not a train-only intercept baseline.
    """
    if x_eval.shape[0] == 0:
        return float("nan")
    pred = (x_eval - np.asarray(reference, dtype=np.float32).reshape(1, -1)) @ np.asarray(
        jacobian, dtype=np.float32
    ).T + np.asarray(intercept, dtype=np.float32).reshape(1, -1)
    residual = xdot_eval - pred
    mse_model = float(np.mean(residual**2))
    baseline = xdot_eval - np.mean(xdot_eval, axis=0, keepdims=True)
    mse_baseline = float(np.mean(baseline**2))
    if not np.isfinite(mse_baseline) or mse_baseline <= 1e-12:
        return float("nan")
    return float(mse_model / mse_baseline)


def restrict_indices_to_segment(
    indices: np.ndarray,
    center: int,
    segment_id: np.ndarray,
) -> np.ndarray:
    """Drop support indices that do not share ``center``'s time segment.

    Chart kNN may point across a coverage gap. Those indices are not samples
    of the local vector field on either side of the gap.
    """
    idx = np.asarray(indices, dtype=np.int32).ravel()
    seg = np.asarray(segment_id)
    if idx.size == 0:
        return idx
    center_i = int(center)
    if center_i < 0 or center_i >= seg.shape[0]:
        return np.zeros((0,), dtype=np.int32)
    in_range = (idx >= 0) & (idx < seg.shape[0])
    idx = idx[in_range]
    if idx.size == 0:
        return idx
    return idx[seg[idx] == seg[center_i]]


def jacobian_segment_ids(
    n: int,
    t_start: Optional[np.ndarray],
    dt: float,
    file_ids: Optional[np.ndarray] = None,
    gap_tol: float = 0.25,
) -> np.ndarray:
    """Integer segment ids from file boundaries and ``Δt_start`` gaps.

    The gap rule matches derivative segmentation: an interval larger than
    ``dt * (1 + gap_tol)``, or a non-finite interval, starts a new segment.
    """
    from .projection import time_gap_slices

    count = int(n)
    ids = np.zeros((count,), dtype=np.int32)
    if count <= 0:
        return ids
    if t_start is None:
        slices = [slice(0, count)]
    else:
        slices = time_gap_slices(np.asarray(t_start, dtype=float), float(dt), tol=float(gap_tol))
    files = None if file_ids is None else np.asarray(file_ids).reshape(-1)
    seg = 0
    for sl in slices:
        if files is None or files.size != count:
            ids[sl] = seg
            seg += 1
            continue
        block = files[sl]
        start = 0 if sl.start is None else int(sl.start)
        run = 0
        for i in range(1, int(block.shape[0])):
            if block[i] != block[i - 1]:
                ids[start + run : start + i] = seg
                seg += 1
                run = i
        ids[start + run : start + int(block.shape[0])] = seg
        seg += 1
    return ids


def _gather_support_indices(
    center: int,
    nn_idx: np.ndarray,
    super_window: int,
    total: int,
    support_mode: str,
) -> np.ndarray:
    if support_mode == SUPPORT_MODE_TIME_LOCAL:
        return _gather_time_local_indices(center, super_window, total)
    return _gather_indices(center, nn_idx, super_window, total)


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
    min_samples: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Fit the local affine derivative model at one center.

    ``exclude_indices`` is used only by transition-residual cross-fitting. It
    never changes the canonical Jacobian estimate returned by
    :func:`estimate_local_jacobians`.
    """
    dim = int(x.shape[1])
    required = int(min_samples) if min_samples is not None else dim + 1
    if neighbour_indices is None:
        neighbours = _gather_indices(int(center), nn_idx, int(super_window), x.shape[0])
    else:
        neighbours = np.asarray(neighbour_indices, dtype=np.int32).ravel()
    if exclude_indices is not None and len(exclude_indices) > 0:
        excluded = np.asarray(list(exclude_indices), dtype=np.int32)
        neighbours = neighbours[~np.isin(neighbours, excluded)]
    if neighbours.size < required:
        return None
    x_samples = x[neighbours]
    xdot_samples = x_dot[neighbours]
    finite_mask = np.isfinite(x_samples).all(axis=1) & np.isfinite(xdot_samples).all(axis=1)
    x_samples = x_samples[finite_mask]
    xdot_samples = xdot_samples[finite_mask]
    if x_samples.shape[0] < required:
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
        "n_fit_samples": int(x_samples.shape[0]),
    }


def _holdout_rel_mse_at_center(
    x: np.ndarray,
    x_dot: np.ndarray,
    nn_idx: np.ndarray,
    center: int,
    *,
    neighbour_idx: np.ndarray,
    super_window: int,
    ridge_alpha: float,
    distance_weighted: bool,
    min_samples: int,
) -> Tuple[float, int]:
    """Fit on a train split of the neighborhood; score relative MSE on holdout.

    Canonical ``J_hat`` is unchanged: this is an auxiliary diagnostic.
    """
    train_idx, holdout_idx = _holdout_split(neighbour_idx, OOS_HOLDOUT_STRIDE)
    if train_idx.size < int(min_samples) or holdout_idx.size < 1:
        return float("nan"), 0
    oos_fit = fit_local_affine_at_center(
        x,
        x_dot,
        nn_idx,
        int(center),
        super_window=super_window,
        ridge_alpha=ridge_alpha,
        distance_weighted=distance_weighted,
        neighbour_indices=train_idx,
        min_samples=min_samples,
    )
    if oos_fit is None:
        return float("nan"), 0
    hold_x = x[holdout_idx]
    hold_xd = x_dot[holdout_idx]
    finite = np.isfinite(hold_x).all(axis=1) & np.isfinite(hold_xd).all(axis=1)
    if int(finite.sum()) < 1:
        return float("nan"), 0
    rel = _affine_rel_mse(
        hold_x[finite],
        hold_xd[finite],
        oos_fit["jacobian"],
        oos_fit["affine_intercept"],
        oos_fit["affine_reference"],
    )
    return float(rel), int(finite.sum())


def estimate_local_jacobians(
    x: np.ndarray,
    x_dot: np.ndarray,
    nn_idx: np.ndarray,
    super_window: int = 3,
    ridge_alpha: float = 1.0,
    distance_weighted: bool = False,
    j_dot_dt: Optional[float] = None,
    knn_k: Optional[int] = None,
    support_mode: str = SUPPORT_MODE_KNN,
    require_determined_support: bool = False,
    segment_id: Optional[np.ndarray] = None,
    forbid_cross_gap: bool = False,
) -> JacobianResult:
    """Estimate windowed Jacobians from MNPS trajectories.

    Each center is one MNPS window. The production ``support_mode="knn"`` fit
    is a local affine map on unique indices gathered from ``super_window``
    time neighbors, each contributing ``knn_k`` chart neighbors. The 8 s / 4 s
    grid is the center lattice, not the sample size of the fit.

    ``support_mode="time_local"`` is experimental: the same affine fit, but
    only on a contiguous time window. It is for synthetic mixing tests and
    bounded replays. It does not change the default summarize estimator.

    An even ``super_window`` is realized as the next odd length. Provenance
    keeps both ``super_window_requested`` and the realized ``super_window``.

    ``require_determined_support`` refuses a window unless the unique
    neighborhood has at least ``dim * (dim + 1)`` samples, the affine
    parameter count. Without that flag the ridge fit may still return a
    finite ``J_hat`` and sets ``ridge_underdetermined``. Default remains
    false so existing EEG neighborhoods are unchanged.

    ``segment_id`` drops chart neighbors that sit in another time segment.
    ``forbid_cross_gap`` with no ``segment_id`` is ``not_testable`` rather
    than a silent return to the historical cross-gap neighborhood.
    Omit both to keep that historical neighborhood.
    """
    if x.ndim != 2 or x_dot.ndim != 2:
        raise ValueError("estimate_local_jacobians expects 2D arrays for x and x_dot")
    if x.shape != x_dot.shape:
        raise ValueError("x and x_dot must have the same shape")

    dim = x.shape[1]
    super_window_requested = max(1, int(super_window))
    super_window = super_window_requested if super_window_requested % 2 else super_window_requested + 1
    support_mode = _normalize_support_mode(support_mode)
    effective_knn_k = (
        0 if support_mode == SUPPORT_MODE_TIME_LOCAL else infer_knn_k(nn_idx, knn_k)
    )
    neighborhood_cfg = neighborhood_config_fields(
        knn_k=effective_knn_k,
        super_window=super_window,
        ridge_alpha=ridge_alpha,
        distance_weighted=distance_weighted,
        dim=dim,
        support_mode=support_mode,
    )
    neighborhood_cfg["super_window_requested"] = int(super_window_requested)
    neighborhood_cfg["require_determined_support"] = bool(require_determined_support)
    if require_determined_support:
        neighborhood_cfg["min_samples"] = max(
            int(neighborhood_cfg["min_samples"]),
            int(neighborhood_cfg["n_affine_parameters"]),
        )
    min_samples = int(neighborhood_cfg["min_samples"])
    segment_arr = None
    if segment_id is not None:
        segment_arr = np.asarray(segment_id)
        if segment_arr.shape[0] != x.shape[0]:
            raise ValueError(
                f"segment_id length {segment_arr.shape[0]} does not match trajectory length {x.shape[0]}"
            )

    if forbid_cross_gap and segment_arr is None:
        return JacobianResult(
            j_hat=np.zeros((0, dim, dim), dtype=np.float32),
            j_dot=np.zeros((0, dim, dim), dtype=np.float32),
            centers=np.zeros((0,), dtype=np.int32),
            diagnostics={
                "windows": 0,
                "failed": 0,
                **neighborhood_cfg,
                **summarize_neighborhood_sample_counts([]),
                **summarize_oos_rel_mse([], [], stride=OOS_HOLDOUT_STRIDE),
                "computation_status": "not_testable",
                "computation_status_reason": "gap_boundaries_unresolved",
                "ridge_underdetermined": False,
                "forbid_cross_gap": True,
            },
        )

    if x.size == 0 or x_dot.size == 0:
        return JacobianResult(
            j_hat=np.zeros((0, dim, dim), dtype=np.float32),
            j_dot=np.zeros((0, dim, dim), dtype=np.float32),
            centers=np.zeros((0,), dtype=np.int32),
            diagnostics={
                "windows": 0,
                "failed": 0,
                **neighborhood_cfg,
                **summarize_neighborhood_sample_counts([]),
                **summarize_oos_rel_mse([], [], stride=OOS_HOLDOUT_STRIDE),
                "computation_status": "not_testable",
                "computation_status_reason": "empty_trajectory",
                "ridge_underdetermined": False,
            },
        )

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
    rel_mse_baseline_oos: list[float] = []
    n_holdout_samples: list[int] = []
    n_neighborhood_samples: list[int] = []
    affine_reference_list: list[np.ndarray] = []
    affine_intercept_list: list[np.ndarray] = []

    for center in centers:
        neighbour_idx = _gather_support_indices(
            center, nn_idx, super_window, x.shape[0], support_mode
        )
        if segment_arr is not None:
            neighbour_idx = restrict_indices_to_segment(neighbour_idx, int(center), segment_arr)
        if neighbour_idx.size < min_samples:
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
            min_samples=min_samples,
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
        n_neighborhood_samples.append(int(fit["n_fit_samples"]))
        oos_rel, n_hold = _holdout_rel_mse_at_center(
            x,
            x_dot,
            nn_idx,
            int(center),
            neighbour_idx=np.asarray(fit["support_indices"], dtype=np.int32),
            super_window=super_window,
            ridge_alpha=ridge_alpha,
            distance_weighted=distance_weighted,
            min_samples=min_samples,
        )
        rel_mse_baseline_oos.append(float(oos_rel))
        n_holdout_samples.append(int(n_hold))
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
                **neighborhood_cfg,
                **summarize_neighborhood_sample_counts([]),
                **summarize_oos_rel_mse([], [], stride=OOS_HOLDOUT_STRIDE),
                "computation_status": "not_testable",
                "computation_status_reason": "insufficient_neighborhood_support",
                "ridge_underdetermined": bool(require_determined_support),
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
        **neighborhood_cfg,
        **summarize_neighborhood_sample_counts(n_neighborhood_samples),
        **summarize_oos_rel_mse(
            rel_mse_baseline_oos, n_holdout_samples, stride=OOS_HOLDOUT_STRIDE
        ),
    }
    sample_summary = summarize_neighborhood_sample_counts(n_neighborhood_samples)
    n_min = float(sample_summary["n_neighborhood_samples_min"])
    n_params = int(neighborhood_cfg["n_affine_parameters"])
    diagnostics["ridge_underdetermined"] = bool(np.isfinite(n_min) and n_min < n_params)
    diagnostics["computation_status"] = "ok" if j_hat.shape[0] else "not_testable"
    if diagnostics["computation_status"] != "ok":
        diagnostics["computation_status_reason"] = "insufficient_neighborhood_support"

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


