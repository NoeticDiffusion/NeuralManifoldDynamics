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
) -> tuple[np.ndarray, dict[str, Any]]:
    """Symmetrize and eigenvalue-floor a covariance-like matrix."""
    q = np.asarray(covariance, dtype=float)
    if q.ndim != 2 or q.shape[0] != q.shape[1] or not np.all(np.isfinite(q)):
        raise ValueError("covariance must be finite and square")
    sym = 0.5 * (q + q.T)
    vals, vecs = np.linalg.eigh(sym)
    floor = max(float(min_eigenvalue), 0.0)
    clipped = np.maximum(vals, floor)
    corrected = (vecs * clipped[None, :]) @ vecs.T
    return corrected.astype(np.float32), {
        "q_psd_correction": bool(np.any(vals < floor)),
        "q_min_eigenvalue": float(np.min(clipped)),
        "q_rank": int(np.sum(clipped > floor)),
        "q_regularization": "eigenvalue_floor",
        "q_shrinkage": 0.0,
    }
