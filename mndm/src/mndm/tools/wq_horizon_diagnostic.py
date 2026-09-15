"""Bounded offline diagnostics for Gate F ``W_Q`` horizon propagation.

This module does not alter the ingest estimator or Gate F configuration.  It
compares ordinary covariance recurrence with a scale tracked representation so
overflow/underflow is visible without clipping a matrix or silently changing
the recurrence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .wq_diagnostic_input import load_gate_e_arrays

PREDECLARED_HORIZONS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
FULL_RECORDING_HORIZON = "full_recording"


def _array(value: Any, *, ndim: int | None = None) -> np.ndarray:
    out = np.asarray(value, dtype=float)
    if ndim is not None and out.ndim != ndim:
        raise ValueError(f"expected array ndim={ndim}, got shape {out.shape}")
    return out


def _matrix_contract(phi: Any, q: Any) -> tuple[np.ndarray, np.ndarray]:
    p = _array(phi, ndim=3)
    cov = _array(q, ndim=2)
    if p.shape[1] != p.shape[2] or cov.shape != p.shape[1:]:
        raise ValueError(f"matrix shape mismatch phi={p.shape}, q={cov.shape}")
    if not np.isfinite(p).all() or not np.isfinite(cov).all():
        raise ValueError("nonfinite propagator or covariance")
    return p, cov


def ordinary_recurrence(phi: Any, q: Any, horizon: int) -> dict[str, Any]:
    """Run ``W <- Phi W Phi.T + Q`` and report first nonfinite step."""
    p, cov = _matrix_contract(phi, q)
    if horizon < 0:
        raise ValueError("horizon must be nonnegative")
    if horizon > len(p):
        return {"horizon_steps": int(horizon), "completed_steps": 0, "covariance": None,
                "finite": None, "support_sufficient": False, "first_nonfinite_step": None}
    dim = p.shape[1]
    w = np.zeros((dim, dim), dtype=float)
    first_nonfinite = None
    support_sufficient = horizon <= len(p)
    for step in range(1, min(horizon, len(p)) + 1):
        with np.errstate(over="ignore", invalid="ignore"):
            w = p[step - 1] @ w @ p[step - 1].T + cov
        if first_nonfinite is None and not np.isfinite(w).all():
            first_nonfinite = step
            break
    return {
        "horizon_steps": int(horizon),
        "completed_steps": int(horizon if first_nonfinite is None else first_nonfinite),
        "covariance": w,
        "finite": bool(np.isfinite(w).all()),
        "support_sufficient": support_sufficient,
        "first_nonfinite_step": first_nonfinite,
    }


def scaled_recurrence(phi: Any, q: Any, horizon: int) -> dict[str, Any]:
    """Run the same recurrence with a tracked positive scalar log scale.

    The returned ``normalized_covariance`` and ``log_scale`` satisfy
    ``W = exp(log_scale) * normalized_covariance`` while representable.  No
    eigenvalues, matrices, or covariances are clipped.
    """
    p, cov = _matrix_contract(phi, q)
    if horizon < 0:
        raise ValueError("horizon must be nonnegative")
    if horizon > len(p):
        return {"horizon_steps": int(horizon), "completed_steps": 0,
                "normalized_covariance": None, "log_scale": None,
                "scaled_finite": None, "representable": None,
                "support_sufficient": False, "first_range_exceed_step": None,
                "first_unrepresentable_step": None, "underflow_loss": False,
                "covariance": None}
    dim = p.shape[1]
    wn = np.zeros((dim, dim), dtype=float)
    log_scale = 0.0
    first_unrepresentable = None
    first_range_exceed = None
    underflow_loss = False
    has_state = False
    max_log = math.log(np.finfo(float).max)
    for step in range(1, min(horizon, len(p)) + 1):
        # Normalize Phi before multiplication so a large but finite operator
        # cannot overflow merely because its square is outside float range.
        if not has_state:
            candidate = cov.copy()
            predicted_log_scale = 0.0
        else:
            phi_norm = float(np.max(np.abs(p[step - 1])))
            q_norm = float(np.max(np.abs(cov)))
            if not np.isfinite(phi_norm) or not np.isfinite(q_norm):
                first_unrepresentable = step
                break
            pred_log = log_scale + 2.0 * math.log(phi_norm) if phi_norm > 0 else -math.inf
            q_log = math.log(q_norm) if q_norm > 0 else -math.inf
            base_log = max(pred_log, q_log)
            if not np.isfinite(base_log):
                # A zero propagator and zero Q keep the absorbing zero state.
                wn = np.zeros_like(wn)
                log_scale = -math.inf
                has_state = True
                continue
            p_term = np.zeros_like(wn)
            q_term = np.zeros_like(cov)
            if np.isfinite(pred_log):
                p_norm = p[step - 1] / phi_norm
                p_factor = math.exp(pred_log - base_log)
                p_term = p_factor * (p_norm @ wn @ p_norm.T)
                underflow_loss = underflow_loss or (p_factor == 0.0)
            if np.isfinite(q_log):
                q_factor = math.exp(q_log - base_log)
                q_term = q_factor * (cov / q_norm)
                underflow_loss = underflow_loss or (q_factor == 0.0)
            candidate = p_term + q_term
            predicted_log_scale = base_log
        norm = float(np.max(np.abs(candidate)))
        if not np.isfinite(norm):
            first_unrepresentable = step
            wn = candidate
            break
        if norm == 0:
            wn = np.zeros_like(candidate)
            log_scale = -math.inf
            has_state = True
            continue
        log_scale = predicted_log_scale + math.log(norm)
        wn = candidate / norm
        has_state = True
        if first_range_exceed is None and log_scale > max_log:
            first_range_exceed = step
    reconstructable = bool(first_unrepresentable is None and (log_scale == -math.inf or log_scale <= max_log))
    covariance = np.exp(log_scale) * wn if reconstructable else None
    return {
        "horizon_steps": int(horizon),
        "horizon_covered": bool(horizon <= len(np.asarray(phi))),
        "completed_steps": int(horizon if first_unrepresentable is None else first_unrepresentable),
        "normalized_covariance": wn,
        "log_scale": float(log_scale),
        "scaled_finite": bool(np.isfinite(wn).all()),
        "representable": reconstructable,
        "support_sufficient": bool(horizon <= len(p)),
        "first_range_exceed_step": first_range_exceed,
        "underflow_loss": underflow_loss,
        "first_unrepresentable_step": first_unrepresentable,
        "covariance": covariance,
    }


def compare_horizon(phi: Any, q: Any, horizon: int) -> dict[str, Any]:
    ordinary = ordinary_recurrence(phi, q, horizon)
    scaled = scaled_recurrence(phi, q, horizon)
    relative_error = None
    if ordinary["finite"] and scaled["representable"] and scaled["covariance"] is not None:
        # Frobenius norms can overflow even when every matrix entry is finite.
        # Normalize both matrices by a shared max-abs entry before measuring
        # their difference; this preserves the comparison without changing the
        # recurrence or hiding an unrepresentable endpoint.
        lhs = ordinary["covariance"]
        rhs = scaled["covariance"]
        shared = max(float(np.max(np.abs(lhs))), float(np.max(np.abs(rhs))))
        if np.isfinite(shared) and shared > 0:
            relative_error = float(np.linalg.norm((lhs - rhs) / shared, ord="fro") /
                                   max(float(np.linalg.norm(lhs / shared, ord="fro")), np.finfo(float).tiny))
        elif shared == 0:
            relative_error = 0.0
    support_sufficient = bool(ordinary["support_sufficient"] and scaled["support_sufficient"])
    return {
        "horizon_steps": int(horizon),
        "horizon_covered": bool(horizon <= len(np.asarray(phi))),
        "support_sufficient": support_sufficient,
        "ordinary_finite": ordinary["finite"] if support_sufficient else None,
        "ordinary_first_nonfinite_step": ordinary["first_nonfinite_step"],
        "scaled_finite": scaled["scaled_finite"],
        "scaled_representable": scaled["representable"] if support_sufficient else None,
        "scaled_log": scaled["log_scale"],
        "scaled_first_unrepresentable_step": scaled["first_unrepresentable_step"],
        "scaled_first_range_exceed_step": scaled["first_range_exceed_step"],
        "scaled_underflow_loss": scaled["underflow_loss"],
        "scaled_endpoint_maxabs": (float(np.max(np.abs(scaled["normalized_covariance"]))) if scaled["normalized_covariance"] is not None else None),
        "scaled_endpoint_trace": (float(np.trace(scaled["normalized_covariance"])) if scaled["normalized_covariance"] is not None else None),
        "relative_error_when_representable": relative_error,
        "qualification_status": "not_assessed",
        "claim_status": "no_biological_claim",
    }


def diagnostic_from_arrays(phi: Any, q: Any, horizons: tuple[int, ...] = PREDECLARED_HORIZONS) -> dict[str, Any]:
    p, cov = _matrix_contract(phi, q)
    selected = tuple(sorted({int(h) for h in horizons if int(h) > 0}))
    full = len(p)
    all_horizons = selected + ((full,) if full and full not in selected else ())
    symmetric = bool(np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12))
    eig = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    q_scale = float(np.trace(cov))
    sign, logdet = np.linalg.slogdet(cov)
    per_start_support = {str(h): max(0, int(len(p) - h + 1)) for h in selected}
    per_start_details = {
        str(h): {"n_supported_starts": count, "first_start": 0 if count else None,
                 "last_start": count - 1 if count else None}
        for h, count in ((int(k), v) for k, v in per_start_support.items())
    }
    per_start_results = {
        str(h): [
            {"start": start, **compare_horizon(p[start : start + h], cov, h)}
            for start in range(max(0, len(p) - h + 1))
        ]
        for h in selected
    }
    return {
        "predeclared_horizons": list(selected),
        "full_recording_steps": int(full),
        "matrix_dimension": int(p.shape[1]),
        "q_scope": "recording_level",
        "independent_local_diffusion": False,
        "q_symmetric": symmetric,
        "q_antisymmetry_fro": float(np.linalg.norm(cov - cov.T, ord="fro")),
        "q_min_eigenvalue": float(np.min(eig)),
        "q_psd": bool(np.min(eig) >= -1e-10),
        "q_trace": q_scale,
        "q_logdet": float(logdet) if sign > 0 else None,
        "q_scaling_reference": "trace_and_logdet_of_serialized_recording_covariance",
        "prefix_support": {str(h): bool(h <= full) for h in selected},
        "per_start_support": per_start_support,
        "per_start_details": per_start_details,
        "per_start_results": per_start_results,
        "protocol_omissions": [
            "per_step_eigenvalue_trace_not_recorded",
            "per_step_trace_logdet_not_recorded",
            "per_start_covariance_endpoints_not_hashed",
        ],
        "protocol_sha256": hashlib.sha256(json.dumps({"horizons": list(selected), "full_recording": True}, sort_keys=True).encode("utf-8")).hexdigest(),
        "results": [compare_horizon(p, cov, h) for h in all_horizons],
    }


def run_h5_diagnostic(h5_path: Path, out_dir: Path, branch: str = "primary") -> Path:
    phi, q, source = load_gate_e_arrays(h5_path, branch=branch)
    result = {"source": source, **diagnostic_from_arrays(phi, q)}
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{Path(h5_path).stem}_{branch}_wq_horizon_diagnostic.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--branch", default="primary")
    args = parser.parse_args(argv)
    print(run_h5_diagnostic(args.h5, args.out_dir, args.branch))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
