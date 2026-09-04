"""Validity helpers for standard orthogonal-dynamics estimators."""

from __future__ import annotations

import numpy as np


_KNN_DISTANCE_MEMORY_BUDGET_BYTES = 64 * 1024 * 1024


def validate_trajectory(
    state: np.ndarray,
    time: np.ndarray,
    *,
    min_samples: int,
    segment_id: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, str | None]:
    """Validate aligned trajectory inputs and return normalized arrays.

    The function does not silently drop rows.  Callers use the returned
    validity mask to expose finite support in their result contract.
    """
    x = np.asarray(state, dtype=float)
    t = np.asarray(time, dtype=float).reshape(-1)
    if x.ndim != 2 or t.ndim != 1 or x.shape[0] != t.size:
        return None, None, None, None, "state_time_shape_mismatch"
    if x.shape[0] < int(min_samples):
        return None, None, None, None, "insufficient_samples"
    if segment_id is None:
        segments = np.zeros(t.size, dtype=np.int32)
    else:
        segments = np.asarray(segment_id).reshape(-1)
        if segments.size != t.size:
            return None, None, None, None, "segment_id_shape_mismatch"
    finite = np.isfinite(t) & np.all(np.isfinite(x), axis=1)
    if int(np.sum(finite)) < int(min_samples):
        return None, None, None, finite, "insufficient_finite_support"
    adjacent_same_segment = segments[1:] == segments[:-1]
    adjacent_finite = finite[1:] & finite[:-1]
    if np.any(
        adjacent_same_segment
        & adjacent_finite
        & ((t[1:] - t[:-1]) <= 0)
    ):
        return None, None, None, finite, "non_monotone_time_within_segment"
    return x, t, segments, finite, None


def increment_pairs(
    state: np.ndarray,
    time: np.ndarray,
    segment_id: np.ndarray,
    *,
    max_gap_sec: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return source rows, increments, and positive within-segment time steps."""
    dx = state[1:] - state[:-1]
    dt = time[1:] - time[:-1]
    same_segment = segment_id[1:] == segment_id[:-1]
    valid = same_segment & np.isfinite(dt) & (dt > 0) & np.all(np.isfinite(dx), axis=1)
    if max_gap_sec is not None:
        valid &= dt <= float(max_gap_sec)
    return np.flatnonzero(valid).astype(np.int32), dx[valid], dt[valid]


def chunked_nearest_neighbors(
    queries: np.ndarray,
    references: np.ndarray,
    k: int,
    *,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact kNN indices and squared distances in query chunks.

    Distances are vectorized over each query chunk, while ``argpartition`` is
    still applied independently to each row. This preserves the serial
    estimator's tie behavior and avoids changing neighbor identity.
    """
    query_array = np.asarray(queries, dtype=float)
    reference_array = np.asarray(references, dtype=float)
    if query_array.ndim != 2 or reference_array.ndim != 2:
        raise ValueError("queries and references must be two-dimensional")
    if query_array.shape[1] != reference_array.shape[1]:
        raise ValueError("queries and references must have matching dimensions")
    if reference_array.shape[0] == 0:
        raise ValueError("references must not be empty")
    k_int = int(k)
    if k_int < 1 or k_int > reference_array.shape[0]:
        raise ValueError("k must be between one and the number of references")
    chunk_int = int(chunk_size)
    if chunk_int < 1:
        raise ValueError("chunk_size must be positive")

    nearest_indices = np.empty((query_array.shape[0], k_int), dtype=np.intp)
    nearest_distances = np.empty((query_array.shape[0], k_int), dtype=float)
    bytes_per_query = reference_array.shape[0] * (
        reference_array.shape[1] + 1
    ) * reference_array.itemsize
    memory_limited_chunk = max(
        1,
        _KNN_DISTANCE_MEMORY_BUDGET_BYTES // max(1, bytes_per_query),
    )
    effective_chunk = min(chunk_int, int(memory_limited_chunk))
    for start in range(0, query_array.shape[0], effective_chunk):
        stop = min(start + effective_chunk, query_array.shape[0])
        differences = (
            reference_array[None, :, :]
            - query_array[start:stop, None, :]
        )
        np.square(differences, out=differences)
        distance_block = np.sum(differences, axis=2)
        for offset, distances in enumerate(distance_block):
            indices = np.argpartition(distances, k_int - 1)[:k_int]
            row = start + offset
            nearest_indices[row] = indices
            nearest_distances[row] = distances[indices]
    return nearest_indices, nearest_distances


def project_to_psd(matrix: np.ndarray, *, eigenvalue_floor: float) -> tuple[np.ndarray, dict[str, float | int]]:
    """Symmetrize and floor a covariance estimate with explicit diagnostics."""
    sym = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    if not np.all(np.isfinite(sym)):
        raise ValueError("nonfinite_covariance")
    values, vectors = np.linalg.eigh(sym)
    floored = np.maximum(values, float(eigenvalue_floor))
    projected = (vectors * floored) @ vectors.T
    return projected, {
        "raw_min_eigenvalue": float(values[0]),
        "psd_floor_applied": int(np.any(values < float(eigenvalue_floor))),
        "psd_eigenvalue_floor": float(eigenvalue_floor),
    }
