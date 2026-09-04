"""Derived local-Jacobian measurements.

The functions in this module interpret an already estimated continuous-time
Jacobian field.  They do not alter estimator semantics or infer biology.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..measurement_certificate import attach_certificate
from ..inferential_grain import attach_grain_for_schema


JACOBIAN_METRICS_SCHEMA_VERSION = "mndm.jacobian_metrics.v1"
_REGIME_CODES = {
    "invalid": -1,
    "marginal": 0,
    "stable_nonreactive": 1,
    "stable_reactive": 2,
    "unstable": 3,
}


def _empty_result(n_windows: int = 0) -> dict[str, Any]:
    """Return a schema-stable empty result."""
    result = {
        "schema_version": JACOBIAN_METRICS_SCHEMA_VERSION,
        "series": {
            "spectral_abscissa": np.full(n_windows, np.nan, dtype=np.float32),
            "numerical_abscissa": np.full(n_windows, np.nan, dtype=np.float32),
            "symmetric_rate_min": np.full(n_windows, np.nan, dtype=np.float32),
            "symmetric_rate_max": np.full(n_windows, np.nan, dtype=np.float32),
            "reactivity_gap": np.full(n_windows, np.nan, dtype=np.float32),
            "stable_reactive_flag": np.full(n_windows, -1, dtype=np.int8),
            "dynamical_regime": np.full(n_windows, _REGIME_CODES["invalid"], dtype=np.int8),
            "spectral_radius": np.full(n_windows, np.nan, dtype=np.float32),
            "frobenius_norm": np.full(n_windows, np.nan, dtype=np.float32),
            "trace": np.full(n_windows, np.nan, dtype=np.float32),
            "rotation_norm": np.full(n_windows, np.nan, dtype=np.float32),
            "henrici_departure": np.full(n_windows, np.nan, dtype=np.float32),
        },
        "summary": {
            "n_windows_total": int(n_windows),
            "n_windows_jacobian_valid": 0,
            "n_windows_metrics_valid": 0,
            "n_windows_stable_reactive": 0,
            "stable_reactive_fraction": float("nan"),
        },
        "provenance": {
            "stability_zero_tolerance": 1e-8,
            "reactivity_zero_tolerance": 1e-8,
            "metric_norm": "euclidean",
        },
        "computation_status": "insufficient_support",
    }
    return attach_grain_for_schema(attach_certificate(result))


def _finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def compute_jacobian_metrics(
    jacobian: np.ndarray | None,
    *,
    stability_zero_tolerance: float = 1e-8,
    reactivity_zero_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Compute per-window and recording-level Jacobian Metrics v1.

    ``numerical_abscissa`` is metric-dependent and is therefore explicitly
    computed in the Euclidean coordinate metric of the release-fixed chart.
    Invalid windows remain represented as NaN/-1 rather than being imputed.
    """
    if jacobian is None:
        return _empty_result()
    J = np.asarray(jacobian, dtype=float)
    if J.ndim != 3 or J.shape[1] != J.shape[2]:
        return _empty_result()

    result = _empty_result(int(J.shape[0]))
    series = result["series"]
    valid = np.all(np.isfinite(J), axis=(1, 2))
    n_valid = int(np.sum(valid))
    result["summary"]["n_windows_jacobian_valid"] = n_valid
    result["provenance"].update(
        {
            "stability_zero_tolerance": float(stability_zero_tolerance),
            "reactivity_zero_tolerance": float(reactivity_zero_tolerance),
            "dimension": int(J.shape[1]),
        }
    )
    if not n_valid:
        return result

    Jv = J[valid]
    try:
        eigvals = np.linalg.eigvals(Jv)
        spectral_abscissa = np.max(np.real(eigvals), axis=1)
        spectral_radius = np.max(np.abs(eigvals), axis=1)
        symmetric = 0.5 * (Jv + np.swapaxes(Jv, 1, 2))
        symmetric_eigs = np.linalg.eigvalsh(symmetric)
        numerical_abscissa = symmetric_eigs[:, -1]
        symmetric_rate_min = symmetric_eigs[:, 0]
        reactivity_gap = numerical_abscissa - spectral_abscissa
        frobenius = np.linalg.norm(Jv, axis=(1, 2))
        trace = np.trace(Jv, axis1=1, axis2=2)
        skew = 0.5 * (Jv - np.swapaxes(Jv, 1, 2))
        rotation = np.linalg.norm(skew, axis=(1, 2))
        eig_energy = np.sum(np.abs(eigvals) ** 2, axis=1)
        henrici_radicand = np.maximum(frobenius**2 - eig_energy, 0.0)
        henrici = np.sqrt(henrici_radicand) / np.maximum(frobenius, np.finfo(float).eps)
    except np.linalg.LinAlgError:
        return result

    stable_reactive = (
        (spectral_abscissa < -float(stability_zero_tolerance))
        & (numerical_abscissa > float(reactivity_zero_tolerance))
    )
    regime = np.full(Jv.shape[0], _REGIME_CODES["marginal"], dtype=np.int8)
    regime[spectral_abscissa > float(stability_zero_tolerance)] = _REGIME_CODES["unstable"]
    stable = spectral_abscissa < -float(stability_zero_tolerance)
    regime[stable & ~(numerical_abscissa > float(reactivity_zero_tolerance))] = _REGIME_CODES[
        "stable_nonreactive"
    ]
    regime[stable_reactive] = _REGIME_CODES["stable_reactive"]

    values = {
        "spectral_abscissa": spectral_abscissa,
        "numerical_abscissa": numerical_abscissa,
        "symmetric_rate_min": symmetric_rate_min,
        "symmetric_rate_max": numerical_abscissa,
        "reactivity_gap": reactivity_gap,
        "spectral_radius": spectral_radius,
        "frobenius_norm": frobenius,
        "trace": trace,
        "rotation_norm": rotation,
        "henrici_departure": henrici,
    }
    for name, value in values.items():
        series[name][valid] = np.asarray(value, dtype=np.float32)
    series["stable_reactive_flag"][valid] = stable_reactive.astype(np.int8)
    series["dynamical_regime"][valid] = regime

    n_metrics_valid = int(
        np.sum(
            np.isfinite(spectral_abscissa)
            & np.isfinite(numerical_abscissa)
            & np.isfinite(symmetric_rate_min)
        )
    )
    n_stable_reactive = int(np.sum(stable_reactive))
    result["summary"].update(
        {
            "n_windows_metrics_valid": n_metrics_valid,
            "n_windows_stable_reactive": n_stable_reactive,
            "stable_reactive_fraction": (
                float(n_stable_reactive / n_metrics_valid) if n_metrics_valid else float("nan")
            ),
            **{f"mean_{name}": _finite_mean(value) for name, value in values.items()},
        }
    )
    n_metrics_valid = int(result["summary"]["n_windows_metrics_valid"])
    result["computation_status"] = (
        "computed" if n_metrics_valid > 0 else "insufficient_support"
    )
    return attach_grain_for_schema(attach_certificate(result))


def flatten_metric_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return a writer-friendly mapping with series and scalar summaries split."""
    out = {
        "series": dict(result.get("series", {})),
        "summary": dict(result.get("summary", {})),
        "provenance": dict(result.get("provenance", {})),
        "schema_version": str(result.get("schema_version", JACOBIAN_METRICS_SCHEMA_VERSION)),
    }
    for key in ("computation_status", "measurement_validity", "claim_status"):
        if key in result:
            out[key] = result[key]
    if isinstance(result.get("grain"), Mapping):
        out["grain"] = dict(result["grain"])
    return out
