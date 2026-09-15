"""Shared validity helpers for local-dynamics products."""

from __future__ import annotations

from typing import Any

import numpy as np


def build_trajectory_segments(
    centers: np.ndarray,
    *,
    time: np.ndarray | None = None,
    max_gap_sec: float | None = None,
) -> dict[str, Any]:
    """Create contiguous segment IDs for ordered Jacobian windows.

    A non-increasing center/time index or a gap larger than ``max_gap_sec`` is
    a hard boundary.  The caller additionally combines this with finite
    operator checks for each requested horizon.
    """
    c = np.asarray(centers, dtype=np.int64).reshape(-1)
    n = int(c.size)
    ids = np.full(n, -1, dtype=np.int32)
    if not n:
        return {"trajectory_segment_id": ids, "max_gap_sec": max_gap_sec, "timebase_policy": "fixed"}
    if time is None:
        steps = np.diff(c.astype(float))
        policy = "fixed"
    else:
        t = np.asarray(time, dtype=float).reshape(-1)
        if t.shape[0] != n:
            raise ValueError("time must align with Jacobian centers")
        steps = np.diff(t)
        policy = "observed"
    positive = steps[np.isfinite(steps) & (steps > 0)]
    nominal = float(np.median(positive)) if positive.size else float("nan")
    gap_limit = float(max_gap_sec) if max_gap_sec is not None else (
        1.5 * nominal if np.isfinite(nominal) else float("inf")
    )
    segment = 0
    ids[0] = segment
    for idx, step in enumerate(steps, start=1):
        if not np.isfinite(step) or step <= 0 or step > gap_limit:
            segment += 1
        ids[idx] = segment
    return {
        "trajectory_segment_id": ids,
        "max_gap_sec": gap_limit,
        "nominal_dt_sec": nominal,
        "timebase_policy": policy,
        "max_step_deviation_sec": (
            float(np.max(np.abs(positive - nominal))) if positive.size and np.isfinite(nominal) else float("nan")
        ),
    }


def project_to_psd(
    covariance: np.ndarray,
    *,
    min_eigenvalue: float = 1e-8,
    output_dtype: Any = np.float32,
    precision: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Symmetrize and eigenvalue-floor a covariance-like matrix.

    ``float32`` remains the default for compatibility.  Callers carrying
    large stochastic covariances can opt into ``output_dtype=np.float64``.
    The eigendecomposition is performed after finite max-absolute scaling so
    that forming ``q + q.T`` does not overflow merely because both terms are
    large.  The requested floor is absolute in the input matrix's units; the
    returned QC is measured again after the requested output-dtype cast.
    """
    # ``precision`` is retained as a keyword bridge for the stochastic
    # reachability callers while ``output_dtype`` remains the direct API.
    if precision is not None:
        output_dtype = precision
    dtype = np.dtype(output_dtype)
    if dtype.kind != "f":
        raise ValueError("output_dtype must be a real floating dtype")
    floor = float(min_eigenvalue)
    if not np.isfinite(floor) or floor < 0.0:
        raise ValueError("min_eigenvalue must be finite and non-negative")

    q = np.asarray(covariance, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != q.shape[1] or not np.all(np.isfinite(q)):
        raise ValueError("covariance must be finite and square")

    # Work in a bounded coordinate system.  Scaling the eigenspectrum back
    # only occurs after the matrix product, and every potentially overflowing
    # result is checked explicitly below.
    scale = float(np.max(np.abs(q))) if q.size else 0.0
    if not np.isfinite(scale):
        raise ValueError("covariance scale is non-finite")
    if scale == 0.0:
        vals = np.zeros(q.shape[0], dtype=np.float64)
        clipped_normalized = np.full(q.shape[0], floor, dtype=np.float64)
        corrected = np.eye(q.shape[0], dtype=np.float64) * floor
        rank = 0
    else:
        normalized = q / scale
        normalized_sym = 0.5 * (normalized + normalized.T)
        vals, vecs = np.linalg.eigh(normalized_sym)
        floor_normalized = floor / scale
        clipped_normalized = np.maximum(vals, floor_normalized)
        corrected_normalized = (vecs * clipped_normalized[None, :]) @ vecs.T
        corrected = scale * corrected_normalized
        rank = int(np.sum(clipped_normalized > floor_normalized))
    if not np.all(np.isfinite(corrected)):
        raise ValueError("PSD reconstruction is non-finite")

    result = np.asarray(corrected, dtype=dtype)
    if not np.all(np.isfinite(result)):
        raise ValueError("PSD reconstruction is non-finite after output dtype cast")

    # Measure the actual serialized/output representation.  Use the same
    # bounded eigensolver pattern so even a large float64 result is auditable.
    result64 = np.asarray(result, dtype=np.float64)
    result_scale = float(np.max(np.abs(result64))) if result64.size else 0.0
    if result_scale == 0.0:
        post_min = 0.0
    else:
        post_normalized = result64 / result_scale
        post_sym = 0.5 * (post_normalized + post_normalized.T)
        post_vals = np.linalg.eigvalsh(post_sym)
        post_min = float(post_vals[0] * result_scale)
        if not np.isfinite(post_min):
            raise ValueError("post-dtype PSD eigenvalue is non-finite")
    psd_tolerance = max(
        float(np.finfo(dtype).eps) * max(1.0, result_scale),
        float(np.finfo(dtype).tiny),
    )
    if not np.isfinite(psd_tolerance):
        raise ValueError("post-dtype PSD tolerance is non-finite")

    return result, {
        "q_psd_correction": bool(np.any(vals < (floor / scale if scale else floor))),
        "q_min_eigenvalue": post_min,
        "q_requested_min_eigenvalue": floor,
        "q_psd_tolerance": psd_tolerance,
        "q_psd_post_dtype": bool(post_min >= -psd_tolerance),
        "q_floor_met_post_dtype": bool(post_min >= floor - psd_tolerance),
        "q_rank": rank,
        "q_output_dtype": dtype.name,
        "q_input_scale": scale,
        "q_output_scale": result_scale,
        "q_regularization": "eigenvalue_floor",
        "q_shrinkage": 0.0,
    }
