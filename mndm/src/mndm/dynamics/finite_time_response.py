"""Finite-time responses of a time-varying local Jacobian field.

The serialized peak-gain analogue of theory ``G_peak`` is
``g_peak_over_horizons`` on this schema, not a ``jacobian_metrics`` field.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.linalg import expm

from .validity import build_trajectory_segments
from ..measurement_certificate import attach_certificate
from ..inferential_grain import attach_grain_for_schema


FINITE_TIME_RESPONSE_SCHEMA_VERSION = "mndm.finite_time_response.v1"


def _empty_result(n_windows: int, horizons: Sequence[int], *, reason: str = "not_requested") -> dict[str, Any]:
    result = {
        "schema_version": FINITE_TIME_RESPONSE_SCHEMA_VERSION,
        "computation_status": reason,
        "validation_level": "not_applicable" if reason != "computed" else "model_derived",
        "failure_reason": None if reason == "computed" else reason,
        "horizons": {},
        "provenance": {},
        "n_windows": int(n_windows),
        "horizon_steps_requested": [int(h) for h in horizons],
    }
    return attach_grain_for_schema(attach_certificate(result))


def _step_durations(time: np.ndarray | None, n: int, nominal_dt_sec: float) -> tuple[np.ndarray, dict[str, Any]]:
    if time is None:
        return np.full(max(0, n - 1), float(nominal_dt_sec)), {
            "timebase_policy": "fixed",
            "nominal_dt_sec": float(nominal_dt_sec),
            "max_step_deviation_sec": 0.0,
        }
    t = np.asarray(time, dtype=float).reshape(-1)
    if t.shape[0] != n:
        raise ValueError("Jacobian time must align with windows")
    steps = np.diff(t)
    finite = steps[np.isfinite(steps) & (steps > 0)]
    nominal = float(np.median(finite)) if finite.size else float(nominal_dt_sec)
    return steps, {
        "timebase_policy": "observed",
        "nominal_dt_sec": nominal,
        "max_step_deviation_sec": float(np.max(np.abs(finite - nominal))) if finite.size else float("nan"),
    }


def _response_metrics(phi: np.ndarray, duration: float) -> dict[str, float]:
    singular = np.linalg.svd(phi, compute_uv=False)
    energy = singular**2
    energy_sum = float(np.sum(energy))
    log_gain = float(np.log(singular[0])) if singular.size and singular[0] > 0 else float("-inf")
    g_max = float(np.exp(log_gain)) if log_gain < np.log(np.finfo(float).max) else float("inf")
    return {
        "log_g_max": log_gain,
        "g_max": g_max,
        "gamma_1": float(log_gain / duration) if duration > 0 else float("nan"),
        "d_resp": float(energy_sum**2 / np.sum(energy**2)) if np.sum(energy**2) > 0 else float("nan"),
        "c_1": float(energy[0] / energy_sum) if energy_sum > 0 else float("nan"),
    }


def _family_transfer(phi: np.ndarray, family_indices: dict[str, Sequence[int]], threshold: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    dim = int(phi.shape[0])
    normalized_indices = {name: np.asarray(indices, dtype=int) for name, indices in family_indices.items()}
    if any(indices.ndim != 1 or indices.size == 0 or np.any(indices < 0) or np.any(indices >= dim) for indices in normalized_indices.values()):
        return {"transfer_status": "invalid_family_indices", "transfer_self": float("nan"), "transfer_cross": float("nan")}
    self_values: list[float] = []
    for source, src in normalized_indices.items():
        source_energy = float(np.linalg.norm(phi[:, src], ord="fro") ** 2)
        if source_energy < threshold:
            out[f"transfer_status_{source}"] = "degenerate_source"
            for target in normalized_indices:
                out[f"transfer_{target}_from_{source}"] = float("nan")
            continue
        out[f"transfer_status_{source}"] = "ok"
        for target, tgt in normalized_indices.items():
            value = float(np.linalg.norm(phi[np.ix_(tgt, src)], ord="fro") ** 2 / source_energy)
            out[f"transfer_{target}_from_{source}"] = value
            if target == source:
                self_values.append(value)
    out["transfer_self"] = float(np.mean(self_values)) if len(self_values) == len(normalized_indices) else float("nan")
    out["transfer_cross"] = float(1.0 - out["transfer_self"]) if np.isfinite(out["transfer_self"]) else float("nan")
    return out


def compute_finite_time_response(
    jacobian: np.ndarray | None,
    *,
    horizon_steps: Sequence[int],
    time: np.ndarray | None = None,
    centers: np.ndarray | None = None,
    nominal_dt_sec: float = 1.0,
    propagator_mode: str = "time_ordered_expm",
    max_gap_sec: float | None = None,
    family_indices: dict[str, Sequence[int]] | None = None,
    source_energy_threshold: float = 1e-12,
    validation_level: str = "model_derived",
) -> dict[str, Any]:
    """Compute finite-time response summaries for valid contiguous sequences.

    The primary ``time_ordered_expm`` mode uses each observed local generator
    and its actual interval.  ``frozen_j_expm`` is deliberately a comparator.
    """
    horizons = sorted({int(h) for h in horizon_steps if int(h) > 0})
    if jacobian is None:
        return _empty_result(0, horizons, reason="unavailable")
    J = np.asarray(jacobian, dtype=float)
    if J.ndim != 3 or J.shape[1] != J.shape[2] or not horizons:
        return _empty_result(int(J.shape[0]) if J.ndim else 0, horizons, reason="invalid")
    if propagator_mode not in {"time_ordered_expm", "frozen_j_expm", "discrete_transition_product"}:
        raise ValueError(f"Unsupported propagator_mode: {propagator_mode}")
    if validation_level not in {"model_derived", "heldout_predictive", "perturbational", "not_applicable"}:
        raise ValueError(f"Unsupported validation_level: {validation_level}")
    if propagator_mode == "discrete_transition_product":
        return _empty_result(J.shape[0], horizons, reason="unavailable")

    n, dim, _ = J.shape
    centers_array = np.arange(n, dtype=np.int32) if centers is None else np.asarray(centers, dtype=np.int32).reshape(-1)
    if centers_array.shape[0] != n:
        raise ValueError("centers must align with Jacobian windows")
    durations, timebase = _step_durations(time, n, nominal_dt_sec)
    # Original center gaps survive post-estimation window filtering and are
    # hard discontinuities even when remaining timestamps appear regular.
    center_segments = build_trajectory_segments(centers_array, time=None, max_gap_sec=None)
    segments = center_segments
    if timebase["timebase_policy"] == "observed":
        time_segments = build_trajectory_segments(
            centers_array,
            time=time,
            max_gap_sec=max_gap_sec,
        )
        center_breaks = np.diff(center_segments["trajectory_segment_id"]) != 0
        time_breaks = np.diff(time_segments["trajectory_segment_id"]) != 0
        # Segment IDs are arbitrary labels, so their maximum is not a union of
        # discontinuities. A boundary in either source is a hard boundary.
        segments = dict(time_segments)
        segments["trajectory_segment_id"] = np.concatenate(
            [np.array([0], dtype=np.int32), np.cumsum(center_breaks | time_breaks, dtype=np.int32)]
        )
        segments["max_gap_sec"] = time_segments["max_gap_sec"]
    finite = np.all(np.isfinite(J), axis=(1, 2))
    result = _empty_result(n, horizons, reason="computed")
    result["validation_level"] = validation_level
    result["provenance"] = {
        "operator_semantics": "continuous_time_generator",
        "propagator_mode": propagator_mode,
        "source_energy_threshold": float(source_energy_threshold),
        "peak_tie_policy": "earliest",
        **timebase,
        "max_gap_sec": segments["max_gap_sec"],
        "center_gap_policy": "hard_break_when_center_step_exceeds_1.5x_nominal",
    }
    ordered_step_propagators: list[np.ndarray | None] | None = None
    if propagator_mode == "time_ordered_expm":
        ordered_step_propagators = [None] * max(0, n - 1)

    for h in horizons:
        candidate = max(0, n - h)
        records: list[dict[str, Any]] = []
        for start in range(candidate):
            stop = start + h
            window_valid = bool(
                np.all(finite[start:stop])
                and np.all(
                    segments["trajectory_segment_id"][start : stop + 1]
                    == segments["trajectory_segment_id"][start]
                )
            )
            duration = float(np.sum(durations[start:stop])) if stop <= durations.size else float("nan")
            if not window_valid or not np.isfinite(duration) or duration <= 0:
                continue
            if propagator_mode == "frozen_j_expm":
                phi = expm(J[start] * duration)
            else:
                phi = np.eye(dim)
                for index in range(start, stop):
                    step_propagator = ordered_step_propagators[index]
                    if step_propagator is None:
                        step_propagator = expm(J[index] * durations[index])
                        ordered_step_propagators[index] = step_propagator
                    phi = step_propagator @ phi
            metric = _response_metrics(phi, duration)
            metric.update({"start_window": start, "actual_horizon_sec": duration})
            if family_indices is not None:
                metric.update(_family_transfer(phi, family_indices, float(source_energy_threshold)))
            records.append(metric)
        summary: dict[str, Any] = {
            "horizon_steps": h,
            "n_sequences_candidate": candidate,
            "n_sequences_valid": len(records),
            "requested_horizon_sec": float(h * timebase["nominal_dt_sec"]),
            "horizon_rounding": "exact_steps",
        }
        if records:
            numeric_names = ["log_g_max", "g_max", "gamma_1", "d_resp", "c_1", "actual_horizon_sec"]
            if family_indices is not None:
                numeric_names.extend(
                    [
                        key
                        for key, value in records[0].items()
                        if key.startswith("transfer_") and isinstance(value, (int, float, np.integer, np.floating))
                    ]
                )
            for name in numeric_names:
                summary[f"mean_{name}"] = float(np.nanmean([r[name] for r in records]))
            summary["series"] = {
                key: np.asarray([record.get(key, np.nan) for record in records], dtype=np.float32)
                for key in numeric_names
            }
            if family_indices is not None:
                for source in family_indices:
                    summary[f"transfer_status_{source}"] = [
                        str(record.get(f"transfer_status_{source}", "invalid_family_indices")) for record in records
                    ]
            summary["start_windows"] = np.asarray([r["start_window"] for r in records], dtype=np.int32)
        result["horizons"][f"h{h}"] = summary

    peaks: list[tuple[int, float, float]] = []
    for h in horizons:
        current = result["horizons"][f"h{h}"]
        if current.get("n_sequences_valid", 0):
            peaks.append((h, float(current["mean_log_g_max"]), float(current["mean_actual_horizon_sec"])))
    if peaks:
        peak_h, peak_log, peak_time = max(peaks, key=lambda item: (item[1], -item[0]))
        result["summary"] = {
            "g_peak_over_horizons": float(np.exp(peak_log)) if peak_log < np.log(np.finfo(float).max) else float("inf"),
            "log_g_peak_over_horizons": peak_log,
            "tau_peak_over_horizons": peak_time,
            "peak_horizon_steps": peak_h,
        }
    else:
        result["summary"] = {
            "g_peak_over_horizons": float("nan"),
            "log_g_peak_over_horizons": float("nan"),
            "tau_peak_over_horizons": float("nan"),
            "peak_horizon_steps": -1,
        }
    return attach_grain_for_schema(attach_certificate(result))
