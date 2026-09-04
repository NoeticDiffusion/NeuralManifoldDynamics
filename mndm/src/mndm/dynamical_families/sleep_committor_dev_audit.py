"""Pure q-free helpers for the OD-SLP-002B BOAS DEV audit."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from .sleep_committor_qualification import (
    ALL_OUTCOMES,
    FROZEN_PID_SPLIT,
    assemble_adapter_arrays,
)
from .validity import increment_pairs, validate_trajectory

CANDIDATE_GRIDS = (9, 17, 33, 65)
EVALUATION_ORDER = (65, 33, 17, 9)
SUPPORT_FLOOR = 64
MIN_TRANSITION_SEGMENTS = 20
MAX_DT_RELATIVE_DEVIATION = 0.05
MIN_EVENT_N3 = 20
MIN_EVENT_REM = 20
MIN_PID_N3 = 8
MIN_PID_REM = 8
MIN_PID_BOTH = 5
MAX_TOP3_SHARE = 0.50


def select_supported_grid(
    tq_status_by_grid: Mapping[int, str],
    support_by_grid: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Select the highest-resolution TQ-PASS grid using support only."""
    candidate_statuses: dict[str, str] = {}
    for grid in EVALUATION_ORDER:
        if tq_status_by_grid.get(int(grid)) != "PASS":
            candidate_statuses[str(grid)] = "EXCLUDED_TQ"
        elif support_by_grid.get(int(grid), {}).get("status") == "SUPPORT_PASS":
            candidate_statuses[str(grid)] = "SUPPORT_PASS"
        else:
            candidate_statuses[str(grid)] = "SUPPORT_NOT_TESTABLE"
    selected_grid = next(
        (
            int(grid)
            for grid in EVALUATION_ORDER
            if candidate_statuses[str(grid)] == "SUPPORT_PASS"
        ),
        None,
    )
    return {
        "selected_grid": selected_grid,
        "candidate_statuses": candidate_statuses,
        "selection_inputs": [
            "OD-SLP-002A-NMD-TQ candidate status",
            "q-free nearest-grid support status",
            "frozen descending candidate order",
        ],
        "endpoint_outcomes_used": False,
        "q_values_used": False,
    }


def q_free_support_audit(
    segments: Sequence[Mapping[str, Any]],
    *,
    lower: float,
    upper: float,
    grid_resolution: int,
    dev_pids: Sequence[int] = FROZEN_PID_SPLIT["DEV"],
    min_support: int = SUPPORT_FLOOR,
    min_transition_segments: int = MIN_TRANSITION_SEGMENTS,
    max_dt_relative_deviation: float = MAX_DT_RELATIVE_DEVIATION,
) -> dict[str, Any]:
    """Count nearest-grid local support and stop before drift or q estimation."""
    result: dict[str, Any] = {
        "grid_resolution": int(grid_resolution),
        "support_floor": int(min_support),
        "status": "SUPPORT_NOT_TESTABLE",
        "failure_reason": None,
        "query_grid": [],
        "support_count": [],
        "n_rows": 0,
        "n_segments": 0,
        "n_increment_pairs": 0,
        "nominal_dt_sec": None,
        "max_dt_relative_deviation": None,
    }
    if (
        int(grid_resolution) not in CANDIDATE_GRIDS
        or not np.isfinite(lower)
        or not np.isfinite(upper)
        or float(lower) >= float(upper)
    ):
        result["failure_reason"] = "invalid_support_grid_contract"
        return result
    allowed_pids = {str(int(pid)) for pid in dev_pids}
    for segment in segments:
        try:
            pid_key = str(int(float(segment.get("pid"))))
        except (TypeError, ValueError, OverflowError):
            result["failure_reason"] = "support_segment_pid_invalid"
            return result
        if pid_key not in allowed_pids:
            result["failure_reason"] = "support_segment_not_in_dev_split"
            return result
    arrays = assemble_adapter_arrays(segments)
    result["n_rows"] = int(arrays["state"].shape[0])
    result["n_segments"] = int(len(arrays["included_segment_ids"]))
    if result["n_segments"] < int(min_transition_segments):
        result["failure_reason"] = "insufficient_support_segments"
        return result
    if result["n_rows"] < 30:
        result["failure_reason"] = "insufficient_support_rows"
        return result
    x, t, segment_ids, finite, failure = validate_trajectory(
        arrays["state"],
        arrays["time"],
        min_samples=30,
        segment_id=arrays["segment_id"],
    )
    if failure or x is None or t is None or segment_ids is None or finite is None:
        result["failure_reason"] = failure or "invalid_support_trajectory"
        return result
    if not bool(np.all(finite)):
        result["failure_reason"] = "nonfinite_support_trajectory"
        return result
    source_idx, increments, dts = increment_pairs(
        x,
        t,
        segment_ids,
        max_gap_sec=None,
    )
    result["n_increment_pairs"] = int(increments.shape[0])
    if increments.shape[0] == 0:
        result["failure_reason"] = "no_within_segment_increment_pairs"
        return result
    nominal_dt = float(np.median(dts))
    relative_deviation = float(
        np.max(np.abs(dts - nominal_dt)) / nominal_dt
    )
    result["nominal_dt_sec"] = nominal_dt
    result["max_dt_relative_deviation"] = relative_deviation
    if relative_deviation > float(max_dt_relative_deviation):
        result["failure_reason"] = "materially_irregular_increment_timestep"
        return result
    query_grid = np.linspace(float(lower), float(upper), int(grid_resolution))
    source_reaction = x[source_idx, 0]
    in_bounds = (
        (source_reaction >= float(lower))
        & (source_reaction <= float(upper))
    )
    if not bool(np.any(in_bounds)):
        result["failure_reason"] = "no_in_bounds_support_sources"
        return result
    nearest = np.argmin(
        np.abs(
            source_reaction[in_bounds, None]
            - query_grid[None, :]
        ),
        axis=1,
    )
    support = np.bincount(
        nearest,
        minlength=int(grid_resolution),
    ).astype(np.int32)
    result["query_grid"] = query_grid.tolist()
    result["support_count"] = support.tolist()
    if np.any(support < int(min_support)):
        result["failure_reason"] = "dense_query_grid_has_under_supported_points"
        return result
    result["status"] = "SUPPORT_PASS"
    return result


def _endpoint_concentration(
    counts: Mapping[str, int],
    total: int,
) -> dict[str, float]:
    values = sorted((int(value) for value in counts.values()), reverse=True)
    if total <= 0:
        return {
            "top3_share": 1.0,
            "max_pid_share": 1.0,
            "n_eff": 0.0,
        }
    padded = values[:3] + [0, 0, 0]
    return {
        "top3_share": float(sum(padded[:3]) / total),
        "max_pid_share": float(max(values, default=0) / total),
        "n_eff": float(
            total**2
            / max(1, sum(value**2 for value in values))
        ),
    }


def endpoint_census(
    segments: Sequence[Mapping[str, Any]],
    *,
    dev_pids: Sequence[int] = FROZEN_PID_SPLIT["DEV"],
) -> dict[str, Any]:
    """Build the eight-way DEV census and pid-level endpoint adequacy."""
    keys = (
        "first_hit_n3",
        "first_hit_rem",
        "competing_exit_wake",
        "competing_exit_n1",
        "competing_exit_disconnection",
        "competing_exit_artifact",
        "qc_or_gap_exit",
        "right_censored",
    )
    counts = Counter()
    pid_counts = {
        str(int(pid)): {"first_hit_n3": 0, "first_hit_rem": 0}
        for pid in dev_pids
    }
    malformed = False
    for segment in segments:
        outcome = str(segment.get("outcome", ""))
        pid = segment.get("pid")
        if outcome not in ALL_OUTCOMES or pid is None:
            malformed = True
            continue
        try:
            pid_key = str(int(float(pid)))
        except (TypeError, ValueError, OverflowError):
            malformed = True
            continue
        if pid_key not in pid_counts:
            malformed = True
            continue
        counts[outcome] += 1
        if outcome in ("first_hit_n3", "first_hit_rem"):
            pid_counts[pid_key][outcome] += 1
    outcome_counts = {key: int(counts.get(key, 0)) for key in keys}
    n3_total = outcome_counts["first_hit_n3"]
    rem_total = outcome_counts["first_hit_rem"]
    n_pid_n3 = sum(value["first_hit_n3"] >= 1 for value in pid_counts.values())
    n_pid_rem = sum(value["first_hit_rem"] >= 1 for value in pid_counts.values())
    n_pid_both = sum(
        value["first_hit_n3"] >= 1 and value["first_hit_rem"] >= 1
        for value in pid_counts.values()
    )
    n_pid_n3_only = sum(
        value["first_hit_n3"] >= 1 and value["first_hit_rem"] == 0
        for value in pid_counts.values()
    )
    n_pid_rem_only = sum(
        value["first_hit_rem"] >= 1 and value["first_hit_n3"] == 0
        for value in pid_counts.values()
    )
    n_pid_neither = sum(
        value["first_hit_n3"] == 0 and value["first_hit_rem"] == 0
        for value in pid_counts.values()
    )
    n3_concentration = _endpoint_concentration(
        {pid: value["first_hit_n3"] for pid, value in pid_counts.items()},
        n3_total,
    )
    rem_concentration = _endpoint_concentration(
        {pid: value["first_hit_rem"] for pid, value in pid_counts.items()},
        rem_total,
    )
    denominator = int(sum(outcome_counts.values()))
    endpoint_fraction = (
        float((n3_total + rem_total) / denominator)
        if denominator
        else None
    )
    endpoint_failures: list[str] = []
    if n3_total < MIN_EVENT_N3:
        endpoint_failures.append("n3_event_floor")
    if rem_total < MIN_EVENT_REM:
        endpoint_failures.append("rem_event_floor")
    if n_pid_n3 < MIN_PID_N3:
        endpoint_failures.append("n3_pid_floor")
    if n_pid_rem < MIN_PID_REM:
        endpoint_failures.append("rem_pid_floor")
    if n_pid_both < MIN_PID_BOTH:
        endpoint_failures.append("both_outcome_pid_floor")
    if n3_concentration["top3_share"] > MAX_TOP3_SHARE:
        endpoint_failures.append("n3_top3_concentration")
    if rem_concentration["top3_share"] > MAX_TOP3_SHARE:
        endpoint_failures.append("rem_top3_concentration")
    if malformed:
        status = "ENDPOINT_NOT_TESTABLE"
        failure_reason = "malformed_or_unknown_endpoint_segment"
    elif endpoint_failures:
        status = "ENDPOINT_INADEQUATE"
        failure_reason = None
    else:
        status = "ENDPOINT_PASS"
        failure_reason = None
    return {
        "status": status,
        "failure_reason": failure_reason,
        "outcome_counts": outcome_counts,
        "n_dev_all_first_outcomes": denominator,
        "p_AB_DEV": endpoint_fraction,
        "n3_count": n3_total,
        "rem_count": rem_total,
        "N_pid_N3": int(n_pid_n3),
        "N_pid_REM": int(n_pid_rem),
        "N_pid_both": int(n_pid_both),
        "N_pid_N3_only": int(n_pid_n3_only),
        "N_pid_REM_only": int(n_pid_rem_only),
        "N_pid_neither_AB": int(n_pid_neither),
        "pid_counts": {
            pid: {
                "n_N3": int(value["first_hit_n3"]),
                "n_REM": int(value["first_hit_rem"]),
            }
            for pid, value in sorted(pid_counts.items(), key=lambda item: int(item[0]))
        },
        "concentration": {
            "N3": n3_concentration,
            "REM": rem_concentration,
        },
        "endpoint_failure_reasons": endpoint_failures,
        "endpoint_floor_contract": {
            "n3_events": MIN_EVENT_N3,
            "rem_events": MIN_EVENT_REM,
            "n3_pids": MIN_PID_N3,
            "rem_pids": MIN_PID_REM,
            "both_outcome_pids": MIN_PID_BOTH,
            "top3_share_max": MAX_TOP3_SHARE,
        },
    }
