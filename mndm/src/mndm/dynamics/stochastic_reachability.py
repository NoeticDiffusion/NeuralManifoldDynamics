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
) -> dict[str, Any]:
    """Estimate a PSD one-step transition-residual covariance proxy.

    This is admissible as a discrete ``W_Q`` input because the residual has
    state units, unlike a derivative-residual covariance.
    """
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
    covariance, qc = project_to_psd(raw, min_eigenvalue=min_eigenvalue)
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
        "covariance": covariance,
        **qc,
    }


@_certify_result
def make_derivative_residual_covariance_proxy(
    derivative_residuals: np.ndarray,
    *,
    min_eigenvalue: float = 1e-8,
    ddof: int = 1,
    q_dt_sec: float | None = None,
) -> dict[str, Any]:
    """Represent derivative residual covariance without claiming W_Q validity."""
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
        "covariance": covariance,
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
) -> dict[str, Any]:
    """Pool cross-fitted transition residuals into a recording-level Q proxy."""
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
    covariance, qc = project_to_psd(raw, min_eigenvalue=min_eigenvalue)
    eigenvalues = np.linalg.eigvalsh(covariance.astype(float))
    trace = float(np.sum(eigenvalues))
    effective_rank = float(np.exp(-np.sum(np.where(eigenvalues > 0, (eigenvalues / trace) * np.log(eigenvalues / trace), 0.0)))) if trace > 0 else float("nan")
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
        "residual_mean": np.mean(residual, axis=0).astype(np.float32),
        "residual_mean_norm": float(np.linalg.norm(np.mean(residual, axis=0))),
        "effective_rank": effective_rank,
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
) -> dict[str, Any]:
    """Propagate a discrete one-step residual covariance through transitions.

    The implementation refuses derivative-residual covariance unless a caller
    has frozen and recorded a conversion model outside this function.
    """
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
    q = np.asarray(contract_covariance, dtype=float)
    supplied_covariance = np.asarray(covariance, dtype=float)
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
    W = np.zeros_like(q)
    baseline = np.zeros_like(q)
    try:
        with np.errstate(over="raise"):
            for phi in propagators:
                p = np.asarray(phi, dtype=float)
                if p.shape != q.shape or not np.all(np.isfinite(p)):
                    return {
                        "schema_version": STOCHASTIC_REACHABILITY_SCHEMA_VERSION,
                        "computation_status": "invalid",
                        "failure_reason": "invalid_propagator",
                    }
                W = p @ W @ p.T + q
                baseline = baseline + q
        W, _ = project_to_psd(W, min_eigenvalue=float(epsilon))
        eigvals = np.linalg.eigvalsh(W.astype(float))[::-1]
        total = float(np.sum(eigvals))
        sq_total = float(np.sum(eigvals**2))
        logdet = float(np.linalg.slogdet(W + float(epsilon) * np.eye(W.shape[0]))[1])
        baseline_trace = float(np.trace(baseline))
        trace = float(np.trace(W))
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
        "v_norm": float(logdet / (2.0 * W.shape[0])),
        "d_eff": float(total**2 / sq_total) if sq_total > 0 else float("nan"),
        "c_1_q": float(eigvals[0] / total) if total > 0 else float("nan"),
        "a_q": float(np.log(trace / max(baseline_trace, float(epsilon)))),
        "n_propagator_steps": int(len(propagators)),
    }


def _unavailable_reachability(*, reason: str, q_contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    contract = q_contract if isinstance(q_contract, Mapping) else {}
    return attach_grain_for_schema(
        attach_certificate(
            {
                "schema_version": STOCHASTIC_REACHABILITY_SCHEMA_VERSION,
                "computation_status": "unavailable",
                "failure_reason": str(reason),
                "q_time_semantics": str(contract.get("q_time_semantics") or ""),
                "q_schema_version": contract.get("schema_version"),
                "provenance": {
                    "q_source": "transition_residual_covariance_proxy",
                    "propagator_source": "gate_e_crossfit_expm_J_dt",
                    "gate": "F",
                },
            }
        )
    )


def compute_stochastic_reachability_from_gate_e(
    transition_residuals: Mapping[str, Any] | None,
    q_proxy: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Gate F ingest adapter: W_Q from Gate E Φ and recording-level Q only."""
    residuals = transition_residuals if isinstance(transition_residuals, Mapping) else {}
    proxy = q_proxy if isinstance(q_proxy, Mapping) else {}
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
