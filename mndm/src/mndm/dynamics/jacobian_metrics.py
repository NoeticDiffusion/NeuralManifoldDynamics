"""Derived local-Jacobian measurements.

The functions in this module interpret an already estimated continuous-time
Jacobian field.  They do not alter estimator semantics or infer biology.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..measurement_certificate import attach_certificate
from ..inferential_grain import attach_grain_for_schema
from ..jacobian import neighborhood_support_provenance


JACOBIAN_METRICS_SCHEMA_VERSION = "mndm.jacobian_metrics.v1"
_REGIME_CODES = {
    "invalid": -1,
    "marginal": 0,
    "stable_nonreactive": 1,
    "stable_reactive": 2,
    "unstable": 3,
}


FIT_FIDELITY_GATE_VERSION = "provisional_icare_20260911"
_DEFAULT_FIT_FIDELITY_THRESHOLD = 0.9


def resolve_fit_fidelity_gate_input(
    raw_value: float | None,
    *,
    gate_enabled: bool,
) -> float | None:
    """Resolve a caller-side fit-fidelity diagnostic for the gate call site.

    :func:`compute_jacobian_metrics` distinguishes "gate not requested"
    (argument is ``None``) from "gate requested but unresolvable" (argument
    is a non-None, non-finite value): the former computes unconditionally
    (legacy behavior), the latter fails **closed**
    (``computation_status="insufficient_support"``,
    ``failure_reason="fit_fidelity_unknown"``).

    When a dataset overlay enables the gate
    (``local_dynamics.jacobian_metrics.fit_fidelity_gate.enabled: true``) but
    the estimator's own diagnostics mapping is missing the expected key
    (``raw_value is None``, e.g. an upstream diagnostics-population bug or an
    unexpected code path), that MUST still resolve to "requested" -- a
    missing diagnostic must not silently reopen a gate the config explicitly
    asked to be enabled. This substitutes NaN (requested-but-unresolved)
    rather than forwarding ``None`` (not-requested) in that case.

    When the gate is not enabled at all, always returns ``None`` regardless
    of ``raw_value``, preserving byte-for-byte legacy behavior for datasets
    that never opted in.
    """
    if not gate_enabled:
        return None
    return raw_value if raw_value is not None else float("nan")


def _empty_result(n_windows: int = 0, *, failure_reason: str | None = None) -> dict[str, Any]:
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
            "rel_mse_baseline": np.full(n_windows, np.nan, dtype=np.float32),
        },
        "summary": {
            "n_windows_total": int(n_windows),
            "n_windows_jacobian_valid": 0,
            "n_windows_metrics_valid": 0,
            "n_windows_fit_unidentified": 0,
            "n_windows_stable_reactive": 0,
            "stable_reactive_fraction": float("nan"),
            "rel_mse_baseline_median": float("nan"),
            "fit_identified": None,
        },
        "provenance": {
            "stability_zero_tolerance": 1e-8,
            "reactivity_zero_tolerance": 1e-8,
            "metric_norm": "euclidean",
            "operator_semantics": "continuous_time_generator",
            "abscissa_units": "1/second",
            "nominal_dt_sec": float("nan"),
            "fit_fidelity_gate": "not_evaluated",
            "fit_fidelity_gate_version": FIT_FIDELITY_GATE_VERSION,
            "fit_fidelity_threshold": _DEFAULT_FIT_FIDELITY_THRESHOLD,
            "knn_k": None,
            "super_window": None,
            "ridge_alpha": None,
            "distance_weighted": None,
            "min_samples": None,
            "n_neighborhood_samples_median": float("nan"),
            "n_neighborhood_samples_min": float("nan"),
        },
        "computation_status": "insufficient_support",
        "failure_reason": failure_reason,
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
    rel_mse_baseline_median: float | None = None,
    rel_mse_baseline_windows: np.ndarray | None = None,
    fit_fidelity_threshold: float = _DEFAULT_FIT_FIDELITY_THRESHOLD,
    nominal_dt_sec: float | None = None,
    estimator_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute per-window and recording-level Jacobian Metrics v1.

    ``numerical_abscissa`` is metric-dependent and is therefore explicitly
    computed in the Euclidean coordinate metric of the release-fixed chart.
    Invalid windows remain represented as NaN/-1 rather than being imputed.

    Fit-fidelity gate
    ------------------
    ``alpha``/``omega`` (``spectral_abscissa``/``numerical_abscissa``) and the
    ``dynamical_regime``/``stable_reactive_flag`` classification derived from
    them are only scientifically meaningful when the local affine fit that
    produced ``J_hat`` beats the no-dynamics (mean) baseline. When the
    estimator's own diagnostics are supplied via ``rel_mse_baseline_median``
    and/or ``rel_mse_baseline_windows`` (see
    ``JacobianResult.diagnostics['rel_mse_baseline_median']`` /
    ``['rel_mse_baseline_windows']`` from :mod:`mndm.jacobian`), this recording
    is fail-closed: a recording whose ``rel_mse_baseline_median`` is at or
    above ``fit_fidelity_threshold`` (near-null local fit) is reported as
    ``computation_status='insufficient_support'`` with
    ``failure_reason='local_linear_fit_not_better_than_baseline'`` and the
    regime/flag series are withheld (left at their invalid/not-testable fill
    values) instead of classifying an unidentified operator as stable,
    reactive, or unstable.

    ``fit_fidelity_threshold`` defaults to ``0.9``, the provisional value from
    the I-CARE CPC1-vs-CPC5 audit (see
    ``project/mnps_v3/tests/ingest_jacobian_fidelity_handover.md``). It is not
    yet frozen against a non-clinical qualification set (Type C measurement
    contract change per ``AGENTS.md``); callers may override it explicitly.
    When neither ``rel_mse_baseline_median`` nor ``rel_mse_baseline_windows``
    is supplied at all, the gate is **not evaluated** and behavior is
    unchanged from prior releases (``provenance.fit_fidelity_gate ==
    'not_evaluated'``, ``summary.fit_identified is None``).

    Gate evaluation is by *argument presence*, not by resolved finiteness: if
    a caller supplies either argument but the resulting fit-quality value is
    unresolvable (all-NaN windows, or a ``rel_mse_baseline_windows`` whose
    length does not match ``jacobian.shape[0]`` with no finite scalar
    fallback), the gate fails **closed** rather than silently falling back to
    unconditional computation (``provenance.fit_fidelity_gate ==
    'rel_mse_baseline_unknown'``, ``summary.fit_identified is False``,
    ``failure_reason='fit_fidelity_unknown'``). "Unknown fidelity" is not the
    same as "not evaluated".

    When both a scalar ``rel_mse_baseline_median`` and an aligned
    ``rel_mse_baseline_windows`` are supplied, the window array wins: it is
    aligned to ``jacobian`` by construction
    (:func:`mndm.jacobian.estimate_local_jacobians` appends per-window
    diagnostics in lockstep with ``J_hat``), whereas a caller-supplied scalar
    may describe a different (e.g. pre-filter) window population than the
    ``jacobian`` actually passed in.

    When ``estimator_diagnostics`` is supplied (typically
    ``JacobianResult.diagnostics``), neighborhood support is copied into
    ``provenance`` next to ``fit_fidelity_threshold``:             "knn_k",
            ``super_window``, ``ridge_alpha``, ``distance_weighted``, ``min_samples``,
            ``support_mode``, ``n_affine_parameters``, ``rel_mse_baseline_oos_median``,
    ``oos_holdout_stride``, and per-recording
            ``n_neighborhood_samples_median`` / ``min``. The
    per-window ``n_neighborhood_samples`` vector stays on
    ``/jacobian/diagnostics`` and is not duplicated here.
    """
    def _attach_neighborhood(payload: dict[str, Any]) -> dict[str, Any]:
        payload["provenance"].update(neighborhood_support_provenance(estimator_diagnostics))
        return payload

    if jacobian is None:
        return _attach_neighborhood(_empty_result(failure_reason="jacobian_unavailable"))
    J = np.asarray(jacobian, dtype=float)
    if J.ndim != 3 or J.shape[1] != J.shape[2]:
        return _attach_neighborhood(_empty_result(failure_reason="invalid_jacobian_shape"))

    result = _empty_result(int(J.shape[0]))
    series = result["series"]
    valid = np.all(np.isfinite(J), axis=(1, 2))
    n_valid = int(np.sum(valid))
    result["summary"]["n_windows_jacobian_valid"] = n_valid

    # Gate inputs are resolved by *argument presence*, not by whether the
    # resolved value happens to be finite. A caller that explicitly attempted
    # to supply fit-fidelity diagnostics (e.g. a degenerate window whose
    # rel_mse_baseline could not be computed, or a misaligned window array)
    # must not silently fall back to the legacy unconditional-compute path;
    # an unresolvable gate fails closed instead of failing open.
    gate_requested = rel_mse_baseline_median is not None or rel_mse_baseline_windows is not None
    window_rel_mse: np.ndarray | None = None
    if rel_mse_baseline_windows is not None:
        candidate = np.asarray(rel_mse_baseline_windows, dtype=float).reshape(-1)
        if candidate.shape[0] == J.shape[0]:
            window_rel_mse = candidate
    # Per-window diagnostics are aligned to J_hat by construction
    # (mndm.jacobian.estimate_local_jacobians appends them together), so a
    # correctly shaped window array is preferred over a scalar median, which
    # a caller may have failed to recompute after independently filtering
    # windows out of J_hat (see apply_standard_jacobian_window_policy).
    if window_rel_mse is not None and np.any(np.isfinite(window_rel_mse)):
        median_rel_mse = float(np.nanmedian(window_rel_mse))
    elif rel_mse_baseline_median is not None and np.isfinite(rel_mse_baseline_median):
        median_rel_mse = float(rel_mse_baseline_median)
    else:
        median_rel_mse = float("nan")
    median_known = bool(np.isfinite(median_rel_mse))
    if not gate_requested:
        fit_identified: bool | None = None
    elif median_known:
        fit_identified = median_rel_mse < float(fit_fidelity_threshold)
    else:
        # Diagnostics were requested but could not be resolved to a usable
        # value (e.g. all-NaN windows, or a shape-mismatched window array
        # with no usable scalar fallback): unknown fidelity is not the same
        # as an identified fit and must not be treated as "not evaluated".
        fit_identified = False

    if not gate_requested:
        fit_fidelity_gate_state = "not_evaluated"
    elif median_known:
        fit_fidelity_gate_state = "rel_mse_baseline_median"
    else:
        fit_fidelity_gate_state = "rel_mse_baseline_unknown"

    dt_value = (
        float(nominal_dt_sec)
        if nominal_dt_sec is not None and np.isfinite(nominal_dt_sec) and nominal_dt_sec > 0
        else float("nan")
    )
    result["provenance"].update(
        {
            "stability_zero_tolerance": float(stability_zero_tolerance),
            "reactivity_zero_tolerance": float(reactivity_zero_tolerance),
            "dimension": int(J.shape[1]),
            "nominal_dt_sec": dt_value,
            "fit_fidelity_gate": fit_fidelity_gate_state,
            "fit_fidelity_gate_version": FIT_FIDELITY_GATE_VERSION,
            "fit_fidelity_threshold": float(fit_fidelity_threshold),
            "rel_mse_baseline_median": median_rel_mse,
        }
    )
    result["provenance"].update(neighborhood_support_provenance(estimator_diagnostics))
    result["summary"]["rel_mse_baseline_median"] = median_rel_mse
    result["summary"]["fit_identified"] = fit_identified
    if not n_valid:
        result["failure_reason"] = "no_finite_jacobian_windows"
        return result

    Jv = J[valid]
    if window_rel_mse is not None:
        series["rel_mse_baseline"][valid] = window_rel_mse[valid].astype(np.float32)

    if gate_requested and not fit_identified:
        # J is finite (a fit was produced) but this recording's local affine
        # model either did not beat the no-dynamics baseline, or its fit
        # quality could not be determined at all. Do not derive alpha/omega
        # regime claims from a near-null or unidentified operator: leave the
        # regime/flag series at their invalid/not-testable fill values and
        # withhold the family rather than serializing a plausible-looking
        # but unidentified stability classification.
        result["computation_status"] = "insufficient_support"
        result["failure_reason"] = (
            "local_linear_fit_not_better_than_baseline"
            if median_known
            else "fit_fidelity_unknown"
        )
        return attach_grain_for_schema(attach_certificate(result))

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
        result["failure_reason"] = "eigendecomposition_failed"
        return result

    # Sub-gate individual windows even when the *recording-level* median
    # passes: the median is a simplification (per the handover's suggested
    # recording-level rule), but a per-window rel_mse_baseline array lets us
    # withhold specific windows whose own local fit did not beat the
    # baseline, instead of letting a passing recording-level median license
    # an alpha/omega claim on every window regardless of its individual fit
    # quality.
    window_fit_unidentified = np.zeros(Jv.shape[0], dtype=bool)
    if window_rel_mse is not None:
        window_rel_mse_valid = window_rel_mse[valid]
        window_fit_unidentified = ~(
            np.isfinite(window_rel_mse_valid) & (window_rel_mse_valid < float(fit_fidelity_threshold))
        )
    if np.any(window_fit_unidentified):
        for arr in (
            spectral_abscissa,
            numerical_abscissa,
            symmetric_rate_min,
            reactivity_gap,
            spectral_radius,
            frobenius,
            trace,
            rotation,
            henrici,
        ):
            arr[window_fit_unidentified] = np.nan

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
    regime[window_fit_unidentified] = _REGIME_CODES["invalid"]

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
    stable_reactive_flag = stable_reactive.astype(np.int8)
    stable_reactive_flag[window_fit_unidentified] = -1
    series["stable_reactive_flag"][valid] = stable_reactive_flag
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
            "n_windows_fit_unidentified": int(np.sum(window_fit_unidentified)),
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
    if result["computation_status"] != "computed":
        result["failure_reason"] = "no_finite_metric_windows"
    return attach_grain_for_schema(attach_certificate(result))


def flatten_metric_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return a writer-friendly mapping with series and scalar summaries split."""
    out = {
        "series": dict(result.get("series", {})),
        "summary": dict(result.get("summary", {})),
        "provenance": dict(result.get("provenance", {})),
        "schema_version": str(result.get("schema_version", JACOBIAN_METRICS_SCHEMA_VERSION)),
    }
    for key in ("computation_status", "failure_reason", "measurement_validity", "claim_status"):
        if key in result:
            out[key] = result[key]
    if isinstance(result.get("grain"), Mapping):
        out["grain"] = dict(result["grain"])
    return out
