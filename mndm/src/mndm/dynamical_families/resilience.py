"""Finite-amplitude resilience summaries for perturbation-result experiments."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .contracts import FINITE_AMPLITUDE_RESILIENCE_SCHEMA_VERSION, build_provenance, unavailable_result
from ..measurement_certificate import attach_certificate
from ..inferential_grain import attach_grain_for_schema


def summarize_finite_amplitude_resilience(
    amplitudes: np.ndarray,
    returned_to_reference: np.ndarray,
    *,
    survived: np.ndarray | None = None,
    recovery_time_sec: np.ndarray | None = None,
    coordinate_layer: str = "simulator_truth",
    coordinate_names: list[str] | None = None,
    min_trials_per_amplitude: int = 20,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize explicitly observed perturbation outcomes by amplitude.

    The input is intentionally outcome-level.  This function does not infer
    basin stability from a Jacobian, spontaneous trajectory, or tangent gain.
    """
    radius = np.asarray(amplitudes, dtype=float).reshape(-1)
    returned = np.asarray(returned_to_reference).reshape(-1)
    names = coordinate_names or []
    if radius.size == 0 or returned.size != radius.size:
        return unavailable_result(
            FINITE_AMPLITUDE_RESILIENCE_SCHEMA_VERSION,
            status="invalid",
            failure_reason="amplitude_return_shape_mismatch",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    valid = np.isfinite(radius) & np.isin(returned, [0, 1, False, True])
    survival_values = None
    if survived is not None:
        survival_values = np.asarray(survived).reshape(-1)
        if survival_values.size != radius.size:
            return unavailable_result(
                FINITE_AMPLITUDE_RESILIENCE_SCHEMA_VERSION,
                status="invalid",
                failure_reason="survival_shape_mismatch",
                coordinate_layer=coordinate_layer,
                coordinate_names=names,
            )
        valid &= np.isin(survival_values, [0, 1, False, True])
    recovery_values = None
    if recovery_time_sec is not None:
        recovery_values = np.asarray(recovery_time_sec, dtype=float).reshape(-1)
        if recovery_values.size != radius.size:
            return unavailable_result(
                FINITE_AMPLITUDE_RESILIENCE_SCHEMA_VERSION,
                status="invalid",
                failure_reason="recovery_time_shape_mismatch",
                coordinate_layer=coordinate_layer,
                coordinate_names=names,
            )
        if np.any(np.isfinite(recovery_values) & (recovery_values < 0)):
            return unavailable_result(
                FINITE_AMPLITUDE_RESILIENCE_SCHEMA_VERSION,
                status="invalid",
                failure_reason="negative_recovery_time",
                coordinate_layer=coordinate_layer,
                coordinate_names=names,
            )
    if int(np.sum(valid)) < int(min_trials_per_amplitude):
        return unavailable_result(
            FINITE_AMPLITUDE_RESILIENCE_SCHEMA_VERSION,
            status="insufficient_support",
            failure_reason="insufficient_valid_perturbation_trials",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )

    rows: list[dict[str, float | int]] = []
    for amplitude in np.unique(radius[valid]):
        member = valid & (radius == amplitude)
        n_trials = int(np.sum(member))
        if n_trials < int(min_trials_per_amplitude):
            continue
        row: dict[str, float | int] = {
            "amplitude": float(amplitude),
            "n_trials": n_trials,
            "basin_return_probability": float(np.mean(returned[member].astype(float))),
        }
        row["return_fraction"] = row["basin_return_probability"]
        if bool((protocol or {}).get("non_return_is_escape", False)):
            row["escape_fraction"] = 1.0 - float(row["basin_return_probability"])
        if survival_values is not None:
            row["survivability"] = float(np.mean(survival_values[member].astype(float)))
        if recovery_values is not None:
            finite_recovery = recovery_values[member & (returned.astype(bool))]
            row["mean_recovery_time_sec"] = float(np.mean(finite_recovery[np.isfinite(finite_recovery)])) if np.any(np.isfinite(finite_recovery)) else float("nan")
        rows.append(row)
    if not rows:
        return unavailable_result(
            FINITE_AMPLITUDE_RESILIENCE_SCHEMA_VERSION,
            status="insufficient_support",
            failure_reason="no_amplitudes_meet_minimum_trial_count",
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
        )
    ordered_rows = sorted(rows, key=lambda row: float(row["amplitude"]))
    probabilities = np.asarray([float(row["basin_return_probability"]) for row in ordered_rows])
    monotone_nonincreasing = bool(np.all(np.diff(probabilities) <= 1e-12))
    r50 = next(
        (float(row["amplitude"]) for row in ordered_rows if float(row["basin_return_probability"]) <= 0.5),
        float("nan"),
    )
    return attach_grain_for_schema(attach_certificate({
        "schema_version": FINITE_AMPLITUDE_RESILIENCE_SCHEMA_VERSION,
        "computation_status": "computed",
        "failure_reason": None,
        "series": {},
        "amplitude_curve": ordered_rows,
        "summary": {
            "r50_discrete_first_bin_at_or_below_half": r50,
            "n_amplitudes_valid": len(ordered_rows),
            "curve_monotone_nonincreasing": monotone_nonincreasing,
        },
        "provenance": build_provenance(
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
            time_semantics="finite_amplitude_perturbation_outcome",
            estimator="observed_perturbation_outcome_summary",
                settings={
                    "min_trials_per_amplitude": int(min_trials_per_amplitude),
                    "protocol": dict(protocol or {}),
                },
        ),
    }))
