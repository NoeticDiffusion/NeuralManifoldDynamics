"""Cross-fitted affine one-step transition residuals (Gate E1)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy.linalg import expm

from ..jacobian import fit_local_affine_at_center
from ..measurement_certificate import attach_certificate
from ..inferential_grain import attach_grain_for_schema


TRANSITION_RESIDUALS_SCHEMA_VERSION = "mndm.transition_residuals.v1"


def affine_one_step_state_matrix(jacobian: np.ndarray, dt_sec: float) -> np.ndarray:
    """Linear one-step map of the affine generator: ``expm(J dt)``.

    The intercept shifts the mean prediction and does not enter covariance
    propagation for ``W_Q``.
    """
    if not np.isfinite(dt_sec) or dt_sec <= 0:
        raise ValueError("dt_sec must be finite and positive")
    return expm(np.asarray(jacobian, dtype=float) * float(dt_sec)).astype(np.float32)


def affine_one_step_predict(
    jacobian: np.ndarray,
    affine_intercept: np.ndarray,
    affine_reference: np.ndarray,
    state: np.ndarray,
    dt_sec: float,
) -> np.ndarray:
    """Propagate a frozen affine continuous-time model by an observed step."""
    J = np.asarray(jacobian, dtype=float)
    b = np.asarray(affine_intercept, dtype=float).reshape(-1)
    x_ref = np.asarray(affine_reference, dtype=float).reshape(-1)
    x = np.asarray(state, dtype=float).reshape(-1)
    dim = int(x.shape[0])
    if J.shape != (dim, dim) or b.shape != (dim,) or x_ref.shape != (dim,):
        raise ValueError("affine parameters must match state dimension")
    if not np.isfinite(dt_sec) or dt_sec <= 0 or not all(np.all(np.isfinite(v)) for v in (J, b, x_ref, x)):
        raise ValueError("affine prediction inputs must be finite and dt_sec positive")
    augmented = np.zeros((dim + 1, dim + 1), dtype=float)
    augmented[:dim, :dim] = J
    augmented[:dim, dim] = b
    z_next = expm(augmented * float(dt_sec)) @ np.concatenate([x - x_ref, [1.0]])
    return (z_next[:dim] + x_ref).astype(np.float32)


def _empty_result(n_windows: int, *, reason: str) -> dict[str, Any]:
    result = {
        "schema_version": TRANSITION_RESIDUALS_SCHEMA_VERSION,
        "computation_status": reason,
        "failure_reason": None if reason == "computed" else reason,
        "series": {
            "source_window_id": np.zeros(0, dtype=np.int32),
            "target_window_id": np.zeros(0, dtype=np.int32),
            "source_center": np.zeros(0, dtype=np.int32),
            "target_center": np.zeros(0, dtype=np.int32),
            "dt_sec": np.zeros(0, dtype=np.float32),
            "x_pred_next": np.zeros((0, 0), dtype=np.float32),
            "transition_residual": np.zeros((0, 0), dtype=np.float32),
            "phi_one_step": np.zeros((0, 0, 0), dtype=np.float32),
        },
        "summary": {"n_windows": int(n_windows), "n_transitions_candidate": 0, "n_transitions_valid": 0},
        "provenance": {},
    }
    return attach_grain_for_schema(attach_certificate(result))


def compute_transition_residuals(
    state: np.ndarray,
    state_dot: np.ndarray,
    time: np.ndarray,
    centers: np.ndarray,
    nn_indices: np.ndarray,
    *,
    super_window: int,
    ridge_alpha: float,
    distance_weighted: bool,
    crossfit_embargo_steps: int,
    coordinate_contract: str,
    coordinate_layer: str,
    segment_id: np.ndarray | None = None,
    coordinate_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Construct cross-fitted residuals for consecutive retained raw centers.

    A source fit excludes an embargo around both endpoints. This is deliberately
    separate from canonical ``J_hat`` estimation, whose semantics remain fixed.
    """
    x = np.asarray(state, dtype=float)
    xdot = np.asarray(state_dot, dtype=float)
    t = np.asarray(time, dtype=float).reshape(-1)
    c = np.asarray(centers, dtype=np.int32).reshape(-1)
    if x.ndim != 2 or x.shape != xdot.shape or t.shape[0] != x.shape[0]:
        return _empty_result(c.size, reason="invalid_state_alignment")
    if c.size < 2:
        return _empty_result(c.size, reason="insufficient_windows")
    if np.any(c < 0) or np.any(c >= x.shape[0]):
        return _empty_result(c.size, reason="invalid_centers")
    segments = np.zeros(x.shape[0], dtype=np.int32) if segment_id is None else np.asarray(segment_id, dtype=np.int32)
    if segments.shape != (x.shape[0],):
        return _empty_result(c.size, reason="invalid_segment_alignment")
    dim = int(x.shape[1])
    embargo = max(0, int(crossfit_embargo_steps))
    records: list[dict[str, Any]] = []
    candidate = int(c.size - 1)
    for source_window in range(candidate):
        target_window = source_window + 1
        source_center, target_center = int(c[source_window]), int(c[target_window])
        # Never bridge filtered/failed Jacobian centers, raw segments, or time gaps.
        if target_center != source_center + 1 or segments[source_center] != segments[target_center]:
            continue
        dt_sec = float(t[target_center] - t[source_center])
        if not np.isfinite(dt_sec) or dt_sec <= 0:
            continue
        excluded = np.concatenate(
            [
                np.arange(max(0, source_center - embargo), min(x.shape[0], source_center + embargo + 1)),
                np.arange(max(0, target_center - embargo), min(x.shape[0], target_center + embargo + 1)),
            ]
        )
        fit = fit_local_affine_at_center(
            x,
            xdot,
            nn_indices,
            source_center,
            super_window=super_window,
            ridge_alpha=ridge_alpha,
            distance_weighted=distance_weighted,
            exclude_indices=np.unique(excluded),
        )
        if fit is None:
            continue
        try:
            predicted = affine_one_step_predict(
                fit["jacobian"],
                fit["affine_intercept"],
                fit["affine_reference"],
                x[source_center],
                dt_sec,
            )
            phi = affine_one_step_state_matrix(fit["jacobian"], dt_sec)
        except ValueError:
            continue
        records.append(
            {
                "source_window_id": source_window,
                "target_window_id": target_window,
                "source_center": source_center,
                "target_center": target_center,
                "dt_sec": dt_sec,
                "x_pred_next": predicted,
                "transition_residual": x[target_center].astype(np.float32) - predicted,
                "phi_one_step": phi,
            }
        )
    if not records:
        result = _empty_result(c.size, reason="no_crossfit_valid_transitions")
    else:
        result = _empty_result(c.size, reason="computed")
        result["series"] = {
            name: np.asarray([record[name] for record in records], dtype=dtype)
            for name, dtype in (
                ("source_window_id", np.int32),
                ("target_window_id", np.int32),
                ("source_center", np.int32),
                ("target_center", np.int32),
                ("dt_sec", np.float32),
                ("x_pred_next", np.float32),
                ("transition_residual", np.float32),
                ("phi_one_step", np.float32),
            )
        }
    result["summary"] = {
        "n_windows": int(c.size),
        "n_transitions_candidate": candidate,
        "n_transitions_valid": int(len(records)),
        "residual_mean": (
            np.mean(result["series"]["transition_residual"], axis=0).astype(np.float32)
            if records
            else np.full(dim, np.nan, dtype=np.float32)
        ),
    }
    result["provenance"] = {
        "propagator_semantics": "frozen_affine_continuous_time_generator_exact_expm",
        "prediction_fit_policy": "leave_one_transition_out_with_temporal_embargo",
        "crossfit_status": "leave_one_transition_out",
        "crossfit_embargo_steps": embargo,
        "coordinate_contract": str(coordinate_contract),
        "coordinate_layer": str(coordinate_layer),
        "coordinate_names": list(coordinate_names or []),
        "dimension": dim,
    }
    return attach_grain_for_schema(attach_certificate(result))
