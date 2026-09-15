"""Residual-covariance and stochastic-reachability measurement contracts."""

from __future__ import annotations

from functools import wraps
from typing import Any, Mapping, Sequence

import numpy as np

from .validity import project_to_psd
from ..measurement_certificate import attach_certificate
from ..inferential_grain import attach_grain_for_schema


RESIDUAL_COVARIANCE_SCHEMA_VERSION = "mndm.residual_covariance_proxy.v1"
TRANSITION_RESIDUAL_COVARIANCE_SCHEMA_VERSION = "mndm.transition_residual_covariance_proxy.v1"
STOCHASTIC_REACHABILITY_SCHEMA_VERSION = "mndm.stochastic_reachability.v1"


def _resolve_precision(precision: str) -> tuple[str, np.dtype]:
    """Resolve the opt-in numerical dtype without changing generic defaults."""
    value = str(precision).lower()
    if value not in {"float32", "float64"}:
        raise ValueError("precision must be 'float32' or 'float64'")
    return value, np.dtype(value)


def _stable_logdet(eigenvalues: np.ndarray) -> float:
    """Compute log(abs(det)) from a complete eigenvalue vector."""
    values = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return float("nan")
    absolute = np.abs(values)
    if np.any(absolute == 0):
        return float("-inf")
    # Summing logs avoids underflow in a finite extreme-spread spectrum such
    # as [1e300, 1e-300], while preserving slogdet's log-absolute-determinant
    # definition over every shifted mode.
    return float(np.sum(np.log(absolute)))


def _stable_trace_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    denominator_floor: float = 0.0,
) -> float:
    """Return log(trace(numerator) / trace(denominator)) without overflow."""
    num_arr = np.asarray(numerator, dtype=np.float64)
    den_arr = np.asarray(denominator, dtype=np.float64)
    if num_arr.ndim != 2 or den_arr.ndim != 2:
        return float("nan")
    num_diag = np.diag(num_arr)
    den_diag = np.diag(den_arr)
    num_scale = float(np.max(np.abs(num_diag))) if num_diag.size else 0.0
    den_scale = float(np.max(np.abs(den_diag))) if den_diag.size else 0.0
    if num_scale <= 0:
        return float("nan")
    num = float(np.sum(num_diag / num_scale))
    den = float(np.sum(den_diag / den_scale)) if den_scale > 0 else 0.0
    if not np.isfinite(num) or not np.isfinite(den) or num <= 0:
        return float("nan")
    floor = float(denominator_floor)
    if not np.isfinite(floor) or floor < 0:
        return float("nan")
    denominator_log = (
        (np.log(den_scale) + np.log(den))
        if den_scale > 0 and den > 0
        else (np.log(floor) if floor > 0 else float("nan"))
    )
    if floor > 0:
        denominator_log = max(denominator_log, np.log(floor))
    if not np.isfinite(denominator_log):
        return float("nan")
    return float(np.log(num_scale) + np.log(num) - denominator_log)


def _stable_log_trace(matrix: np.ndarray) -> float:
    """Compute log(trace(matrix)) using diagonal scaling."""
    diagonal = np.diag(np.asarray(matrix, dtype=np.float64))
    scale = float(np.max(np.abs(diagonal))) if diagonal.size else 0.0
    if scale <= 0:
        return float("-inf")
    normalized_trace = float(np.sum(diagonal / scale))
    if not np.isfinite(normalized_trace) or normalized_trace <= 0:
        return float("nan")
    return float(np.log(scale) + np.log(normalized_trace))


def _safe_symmetric(matrix: np.ndarray) -> np.ndarray:
    """Symmetrize finite near-limit entries without doubling them first."""
    arr = np.asarray(matrix, dtype=np.float64)
    scale = float(np.max(np.abs(arr))) if arr.size else 0.0
    if scale == 0.0:
        return 0.5 * (arr + arr.T)
    return 0.5 * (arr / scale + arr.T / scale) * scale


def _certify_result(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        result = fn(*args, **kwargs)
        if isinstance(result, Mapping) and "computation_status" in result:
            return attach_grain_for_schema(attach_certificate(result))
        return result

    return wrapped


def phi_q_alignment(
    phi: np.ndarray,
    reachability: np.ndarray,
    *,
    eigengap_threshold: float = 1e-6,
    subspace_rank: int = 2,
) -> dict[str, Any]:
    """Return a gated leading-direction or stable subspace alignment diagnostic."""
    p = np.asarray(phi, dtype=float)
    w = np.asarray(reachability, dtype=float)
    if p.ndim != 2 or w.shape != p.shape or not np.all(np.isfinite(p)) or not np.all(np.isfinite(w)):
        return {"alignment_status": "invalid_input"}
    _, _, vt = np.linalg.svd(p, full_matrices=False)
    values_w, vectors_w = np.linalg.eigh(0.5 * (w + w.T))
    values_w = values_w[::-1]
    vectors_w = vectors_w[:, ::-1]
    phi_values = np.linalg.svd(p, compute_uv=False)
    phi_gap = float(phi_values[0] - phi_values[1]) if phi_values.size > 1 else float("inf")
    q_gap = float(values_w[0] - values_w[1]) if values_w.size > 1 else float("inf")
    if phi_gap > eigengap_threshold and q_gap > eigengap_threshold:
        return {
            "alignment_status": "top1",
            "phi_q_alignment_top1": float(abs(np.dot(vt[0], vectors_w[:, 0])) ** 2),
            "phi_eigengap": phi_gap,
            "q_eigengap": q_gap,
        }
    k = min(int(subspace_rank), p.shape[0])
    return {
        "alignment_status": "subspace",
        "phi_q_alignment_subspace_k": float(np.linalg.norm(vt[:k] @ vectors_w[:, :k], ord="fro") ** 2 / k),
        "subspace_rank": k,
        "phi_eigengap": phi_gap,
        "q_eigengap": q_gap,
    }


@_certify_result
def estimate_one_step_transition_covariance(
    state: np.ndarray,
    predicted_next_state: np.ndarray,
    *,
    min_eigenvalue: float = 1e-8,
    ddof: int = 1,
    q_dt_sec: float | None = None,
    precision: str = "float32",
) -> dict[str, Any]:
    """Estimate a PSD one-step transition-residual covariance proxy.

    This is admissible as a discrete ``W_Q`` input because the residual has
    state units, unlike a derivative-residual covariance.
    """
    precision_name, precision_dtype = _resolve_precision(precision)
    if q_dt_sec is None or not np.isfinite(q_dt_sec) or q_dt_sec <= 0:
        return {
            "schema_version": RESIDUAL_COVARIANCE_SCHEMA_VERSION,
            "computation_status": "invalid",
            "failure_reason": "missing_or_invalid_q_dt_sec",
        }
    x = np.asarray(state, dtype=float)
    xhat = np.asarray(predicted_next_state, dtype=float)
    if x.ndim != 2 or x.shape != xhat.shape or x.shape[0] < 2:
        return {"schema_version": RESIDUAL_COVARIANCE_SCHEMA_VERSION, "computation_status": "invalid"}
    residual = x - xhat
    valid = np.all(np.isfinite(residual), axis=1)
    residual = residual[valid]
    if residual.shape[0] <= int(ddof):
        return {
            "schema_version": RESIDUAL_COVARIANCE_SCHEMA_VERSION,
            "computation_status": "invalid",
            "q_time_semantics": "one_step_transition_covariance",
            "failure_reason": "insufficient_samples",
            "q_n_samples": int(residual.shape[0]),
        }
    raw = np.atleast_2d(np.cov(residual, rowvar=False, ddof=int(ddof)))
    covariance, qc = project_to_psd(raw, min_eigenvalue=min_eigenvalue, precision=precision_name)
    return {
        "schema_version": RESIDUAL_COVARIANCE_SCHEMA_VERSION,
        "computation_status": "computed",
        "q_time_semantics": "one_step_transition_covariance",
        "q_semantics": "residual_innovation_proxy",
        "q_n_samples": int(residual.shape[0]),
        "q_units": "state_squared",
        "q_dt_sec": float(q_dt_sec),
        "degrees_of_freedom_policy": f"covariance_ddof_{int(ddof)}",
        "conversion_model": "not_applicable",
        "out_of_sample_status": "not_available",
        "covariance": np.asarray(covariance, dtype=precision_dtype),
        "numerical_precision": precision_name,
        **qc,
    }


@_certify_result
def make_derivative_residual_covariance_proxy(
    derivative_residuals: np.ndarray,
    *,
    min_eigenvalue: float = 1e-8,
    ddof: int = 1,
    q_dt_sec: float | None = None,
    precision: str = "float32",
) -> dict[str, Any]:
    """Represent derivative residual covariance without claiming W_Q validity."""
    precision_name, precision_dtype = _resolve_precision(precision)
    if q_dt_sec is None or not np.isfinite(q_dt_sec) or q_dt_sec <= 0:
        return {
            "schema_version": RESIDUAL_COVARIANCE_SCHEMA_VERSION,
            "computation_status": "invalid",
            "failure_reason": "missing_or_invalid_q_dt_sec",
        }
    residual = np.asarray(derivative_residuals, dtype=float)
    if residual.ndim != 2 or residual.shape[0] <= int(ddof):
        return {"schema_version": RESIDUAL_COVARIANCE_SCHEMA_VERSION, "computation_status": "invalid"}
    residual = residual[np.all(np.isfinite(residual), axis=1)]
    if residual.shape[0] <= int(ddof):
        return {"schema_version": RESIDUAL_COVARIANCE_SCHEMA_VERSION, "computation_status": "invalid"}
    covariance, qc = project_to_psd(
        np.atleast_2d(np.cov(residual, rowvar=False, ddof=int(ddof))),
        min_eigenvalue=min_eigenvalue,
        precision=precision_name,
    )
    return {
        "schema_version": RESIDUAL_COVARIANCE_SCHEMA_VERSION,
        "computation_status": "computed",
        "q_time_semantics": "derivative_residual_covariance_proxy",
        "q_semantics": "residual_innovation_proxy",
        "q_n_samples": int(residual.shape[0]),
        "q_units": "state_squared_per_second_squared",
        "q_dt_sec": float(q_dt_sec),
        "conversion_model": "required_before_stochastic_reachability",
        "covariance": np.asarray(covariance, dtype=precision_dtype),
        "numerical_precision": precision_name,
        **qc,
    }


def _valid_transition_rows(residual: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Finite residual rows with strictly positive observed steps."""
    residual_arr = np.asarray(residual, dtype=float)
    dt_arr = np.asarray(dt, dtype=float).reshape(-1)
    if residual_arr.ndim != 2 or residual_arr.shape[0] != dt_arr.size:
        return np.zeros(0, dtype=bool)
    return np.all(np.isfinite(residual_arr), axis=1) & np.isfinite(dt_arr) & (dt_arr > 0)


@_certify_result
def estimate_transition_residual_covariance_proxy(
    transition_residuals: Mapping[str, Any],
    *,
    max_dt_deviation_sec: float = 1e-6,
    min_eigenvalue: float = 1e-8,
    ddof: int = 1,
    precision: str = "float32",
) -> dict[str, Any]:
    """Pool cross-fitted transition residuals into a recording-level Q proxy."""
    precision_name, precision_dtype = _resolve_precision(precision)
    series = transition_residuals.get("series", {}) if isinstance(transition_residuals, Mapping) else {}
    provenance = transition_residuals.get("provenance", {}) if isinstance(transition_residuals, Mapping) else {}
    residual = np.asarray(series.get("transition_residual", []), dtype=float)
    dt = np.asarray(series.get("dt_sec", []), dtype=float).reshape(-1)
    if transition_residuals.get("computation_status") != "computed":
        return {
            "schema_version": TRANSITION_RESIDUAL_COVARIANCE_SCHEMA_VERSION,
            "computation_status": "unavailable",
            "failure_reason": "transition_residuals_not_computed",
        }
    if residual.ndim != 2 or residual.shape[0] != dt.size:
        return {
            "schema_version": TRANSITION_RESIDUAL_COVARIANCE_SCHEMA_VERSION,
            "computation_status": "invalid",
            "failure_reason": "invalid_transition_residual_alignment",
        }
    valid = _valid_transition_rows(residual, dt)
    residual, dt = residual[valid], dt[valid]
    if residual.shape[0] <= max(int(ddof), residual.shape[1]):
        return {
            "schema_version": TRANSITION_RESIDUAL_COVARIANCE_SCHEMA_VERSION,
            "computation_status": "unavailable",
            "failure_reason": "insufficient_transition_support",
            "q_n_samples": int(residual.shape[0]),
        }
    q_dt_sec = float(np.median(dt))
    if float(np.max(np.abs(dt - q_dt_sec))) > float(max_dt_deviation_sec):
        return {
            "schema_version": TRANSITION_RESIDUAL_COVARIANCE_SCHEMA_VERSION,
            "computation_status": "unavailable",
            "failure_reason": "materially_irregular_dt",
            "q_n_samples": int(residual.shape[0]),
            "q_dt_sec": q_dt_sec,
            "q_max_dt_deviation_sec": float(np.max(np.abs(dt - q_dt_sec))),
        }
    raw = np.atleast_2d(np.cov(residual, rowvar=False, ddof=int(ddof)))
    covariance, qc = project_to_psd(raw, min_eigenvalue=min_eigenvalue, precision=precision_name)
    covariance = np.asarray(covariance, dtype=precision_dtype)
    eigenvalues = np.linalg.eigvalsh(_safe_symmetric(covariance))
    trace = float(np.sum(eigenvalues))
    if trace > 0 and np.isfinite(trace):
        normalized = eigenvalues[eigenvalues > 0] / trace
        entropy = -float(np.sum(normalized * np.log(normalized))) if normalized.size else float("nan")
        effective_rank = float(np.exp(entropy))
    else:
        effective_rank = float("nan")
    return {
        "schema_version": TRANSITION_RESIDUAL_COVARIANCE_SCHEMA_VERSION,
        "computation_status": "computed",
        "q_time_semantics": "one_step_transition_covariance",
        "q_semantics": "transition_residual_covariance_proxy",
        "q_scope": "recording",
        "q_n_samples": int(residual.shape[0]),
        "q_dt_sec": q_dt_sec,
        "q_max_dt_deviation_sec": float(np.max(np.abs(dt - q_dt_sec))),
        "q_units": "state_squared",
        "degrees_of_freedom_policy": f"covariance_ddof_{int(ddof)}",
        "conversion_model": "not_applicable",
        "crossfit_status": provenance.get("crossfit_status"),
        "prediction_fit_policy": provenance.get("prediction_fit_policy"),
        "coordinate_contract": provenance.get("coordinate_contract"),
        "coordinate_layer": provenance.get("coordinate_layer"),
        "residual_mean": np.mean(residual, axis=0).astype(precision_dtype),
        "residual_mean_norm": float(np.linalg.norm(np.mean(residual, axis=0))),
        "effective_rank": effective_rank,
        "numerical_precision": precision_name,
        "covariance": covariance,
        **qc,
    }


@_certify_result
def compute_stochastic_reachability(
    propagators: Sequence[np.ndarray],
    covariance: np.ndarray,
    *,
    q_contract: dict[str, Any],
    epsilon: float = 1e-8,
    precision: str = "float32",
) -> dict[str, Any]:
    """Propagate a discrete one-step residual covariance through transitions.

    The implementation refuses derivative-residual covariance unless a caller
    has frozen and recorded a conversion model outside this function.
    """
    precision_name, precision_dtype = _resolve_precision(precision)
    semantics = str(q_contract.get("q_time_semantics", ""))
    if (
        q_contract.get("computation_status") != "computed"
        or semantics != "one_step_transition_covariance"
        or q_contract.get("q_units") != "state_squared"
        or q_contract.get("conversion_model") != "not_applicable"
        or not np.isfinite(q_contract.get("q_dt_sec", np.nan))
        or float(q_contract.get("q_dt_sec", np.nan)) <= 0.0
    ):
        return {
            "schema_version": STOCHASTIC_REACHABILITY_SCHEMA_VERSION,
            "computation_status": "unavailable",
            "failure_reason": "q_contract_not_admissible",
            "q_time_semantics": semantics,
        }
    contract_covariance = q_contract.get("covariance")
    if contract_covariance is None:
        return {
            "schema_version": STOCHASTIC_REACHABILITY_SCHEMA_VERSION,
            "computation_status": "invalid",
            "failure_reason": "q_contract_missing_covariance",
        }
    # Preserve the historical float64 recurrence arithmetic.  ``precision``
    # controls covariance/Q output precision and PSD serialization; it does
    # not move the existing W recursion into float32.
    q = np.asarray(contract_covariance, dtype=np.float64)
    supplied_covariance = np.asarray(covariance, dtype=np.float64)
    if supplied_covariance.shape != q.shape or not np.allclose(supplied_covariance, q, equal_nan=True):
        return {
            "schema_version": STOCHASTIC_REACHABILITY_SCHEMA_VERSION,
            "computation_status": "invalid",
            "failure_reason": "covariance_does_not_match_q_contract",
        }
    if q.ndim != 2 or q.shape[0] != q.shape[1] or not np.all(np.isfinite(q)):
        return {
            "schema_version": STOCHASTIC_REACHABILITY_SCHEMA_VERSION,
            "computation_status": "invalid",
            "failure_reason": "invalid_covariance",
        }
    if not propagators:
        return {
            "schema_version": STOCHASTIC_REACHABILITY_SCHEMA_VERSION,
            "computation_status": "invalid",
            "failure_reason": "empty_one_step_transitions",
        }
    W = np.zeros_like(q, dtype=np.float64)
    # Keep the baseline as one Q term and account for the number of steps in
    # log space; summing N large Q matrices can overflow even when the actual
    # W recurrence remains finite.
    baseline = q.copy()
    try:
        with np.errstate(over="raise"):
            for phi in propagators:
                p = np.asarray(phi, dtype=np.float64)
                if p.shape != q.shape or not np.all(np.isfinite(p)):
                    return {
                        "schema_version": STOCHASTIC_REACHABILITY_SCHEMA_VERSION,
                        "computation_status": "invalid",
                        "failure_reason": "invalid_propagator",
                    }
                W = p @ W @ p.T + q
                # The recording baseline is N * Q conceptually.  Do not form
                # that matrix in the production path.
        W, w_qc = project_to_psd(W, min_eigenvalue=float(epsilon), precision=precision_name)
        W = np.asarray(W, dtype=precision_dtype)
        if w_qc.get("q_psd_post_dtype") is False:
            return {
                "schema_version": STOCHASTIC_REACHABILITY_SCHEMA_VERSION,
                "computation_status": "invalid",
                "failure_reason": "post_dtype_psd_failure",
                "numerical_precision": precision_name,
                "w_q_projection_qc": dict(w_qc),
            }
        eigvals = np.linalg.eigvalsh(_safe_symmetric(W))[::-1]
        scale = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
        normalized = eigvals / scale if scale > 0 else np.asarray([], dtype=np.float64)
        total_normalized = float(np.sum(normalized)) if normalized.size else 0.0
        sq_total_normalized = float(np.sum(normalized**2)) if normalized.size else 0.0
        regularized_eigvals = eigvals + float(epsilon)
        logdet = _stable_logdet(regularized_eigvals)
        log_trace_w = _stable_log_trace(W)
        log_trace_q = _stable_log_trace(baseline)
        denominator_log = max(
            np.log(len(propagators)) + log_trace_q,
            np.log(float(epsilon)),
        )
        a_q = float(log_trace_w - denominator_log) if np.isfinite(log_trace_w) else float("nan")
    except (OverflowError, FloatingPointError, ValueError):
        return {
            "schema_version": STOCHASTIC_REACHABILITY_SCHEMA_VERSION,
            "computation_status": "invalid",
            "failure_reason": "reachability_numerical_overflow",
        }
    return {
        "schema_version": STOCHASTIC_REACHABILITY_SCHEMA_VERSION,
        "computation_status": "computed",
        "q_time_semantics": semantics,
        "q_schema_version": q_contract.get("schema_version"),
        "q_dt_sec": float(q_contract["q_dt_sec"]),
        "q_units": q_contract.get("q_units"),
        "conversion_model": q_contract.get("conversion_model"),
        "w_q": W,
        "numerical_precision": precision_name,
        "v_norm": float(logdet / (2.0 * W.shape[0])) if np.isfinite(logdet) else float("nan"),
        "d_eff": float(total_normalized**2 / sq_total_normalized) if sq_total_normalized > 0 else float("nan"),
        "c_1_q": float(normalized[0] / total_normalized) if total_normalized > 0 else float("nan"),
        "a_q": a_q,
        "w_q_psd_correction": w_qc.get("q_psd_correction"),
        "w_q_min_eigenvalue": w_qc.get("q_min_eigenvalue"),
        "w_q_projection_qc": {
            **dict(w_qc),
            "q_floor_met_exact_post_dtype": bool(
                w_qc.get("q_min_eigenvalue", float("nan")) >= w_qc.get("q_requested_min_eigenvalue", float("inf"))
            ),
            "q_floor_met_within_tolerance_post_dtype": bool(w_qc.get("q_floor_met_post_dtype", False)),
            "q_psd_material_failure": bool(not w_qc.get("q_psd_post_dtype", False)),
        },
        "n_propagator_steps": int(len(propagators)),
    }


def _unavailable_reachability(
    *,
    reason: str,
    q_contract: Mapping[str, Any] | None = None,
    extra_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = q_contract if isinstance(q_contract, Mapping) else {}
    provenance = {
        "q_source": "transition_residual_covariance_proxy",
        "propagator_source": "gate_e_crossfit_expm_J_dt",
        "gate": "F",
    }
    if isinstance(extra_provenance, Mapping):
        provenance.update(extra_provenance)
    return attach_grain_for_schema(
        attach_certificate(
            {
                "schema_version": STOCHASTIC_REACHABILITY_SCHEMA_VERSION,
                "computation_status": "unavailable",
                "failure_reason": str(reason),
                "q_time_semantics": str(contract.get("q_time_semantics") or ""),
                "q_schema_version": contract.get("schema_version"),
                "provenance": provenance,
            }
        )
    )


def compute_stochastic_reachability_from_gate_e(
    transition_residuals: Mapping[str, Any] | None,
    q_proxy: Mapping[str, Any] | None,
    *,
    precision: str = "float32",
    jacobian_fit_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Gate F ingest adapter: W_Q from Gate E Φ and recording-level Q only.

    ``jacobian_fit_gate`` is the same recording-level local-fit fidelity gate
    computed for ``/jacobian/derived_metrics/v1`` (see
    ``mndm.dynamics.jacobian_metrics.compute_jacobian_metrics``:
    ``summary['fit_identified']`` / ``summary['rel_mse_baseline_median']``).
    The one-step propagators consumed here (``phi_one_step``,
    ``gate_e_crossfit_expm_J_dt``) come from the same local-affine estimator
    family. When that recording's local fit does not beat the no-dynamics
    baseline, ``J`` (and hence Φ = expm(J·Δt)) is not an identified operator;
    attempting the ``W <- ΦWΦ^T + Q`` recursion on it is expected to overflow
    (see ``project/mnps_v3/tests/ingest_jacobian_fidelity_handover.md``, S2).
    Fail closed instead of running the recursion to the numerical-overflow
    failure mode.
    """
    residuals = transition_residuals if isinstance(transition_residuals, Mapping) else {}
    proxy = q_proxy if isinstance(q_proxy, Mapping) else {}
    gate = jacobian_fit_gate if isinstance(jacobian_fit_gate, Mapping) else {}
    if gate.get("fit_identified") is False:
        return _unavailable_reachability(
            reason="upstream_jacobian_local_fit_not_identified",
            q_contract=proxy,
            extra_provenance={
                "upstream_rel_mse_baseline_median": gate.get("rel_mse_baseline_median"),
            },
        )
    if residuals.get("computation_status") != "computed":
        return _unavailable_reachability(
            reason=str(residuals.get("failure_reason") or "transition_residuals_not_computed"),
            q_contract=proxy,
        )
    if proxy.get("computation_status") != "computed":
        return _unavailable_reachability(
            reason=str(proxy.get("failure_reason") or "q_contract_not_admissible"),
            q_contract=proxy,
        )
    if proxy.get("schema_version") != TRANSITION_RESIDUAL_COVARIANCE_SCHEMA_VERSION:
        return _unavailable_reachability(reason="q_contract_not_admissible", q_contract=proxy)
    series = residuals.get("series", {}) if isinstance(residuals.get("series"), Mapping) else {}
    residual = np.asarray(series.get("transition_residual", []), dtype=float)
    dt = np.asarray(series.get("dt_sec", []), dtype=float).reshape(-1)
    phi = np.asarray(series.get("phi_one_step", []), dtype=float)
    if (
        phi.ndim != 3
        or residual.ndim != 2
        or phi.shape[0] != residual.shape[0]
        or residual.shape[0] != dt.size
        or phi.shape[1] != phi.shape[2]
        or phi.shape[1] != residual.shape[1]
    ):
        return _unavailable_reachability(reason="missing_one_step_propagators", q_contract=proxy)
    phi = phi[_valid_transition_rows(residual, dt)]
    if phi.shape[0] == 0 or int(proxy.get("q_n_samples", -1)) != int(phi.shape[0]):
        return _unavailable_reachability(reason="phi_q_row_mismatch", q_contract=proxy)
    result = compute_stochastic_reachability(
        [step for step in phi],
        proxy["covariance"],
        q_contract=dict(proxy),
        precision=precision,
    )
    provenance = dict(result.get("provenance") or {}) if isinstance(result.get("provenance"), Mapping) else {}
    provenance.update(
        {
            "q_source": "transition_residual_covariance_proxy",
            "propagator_source": "gate_e_crossfit_expm_J_dt",
            "gate": "F",
        }
    )
    result["provenance"] = provenance
    return result
