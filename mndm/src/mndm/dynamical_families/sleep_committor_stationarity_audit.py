"""Pure DEV score-only stationarity helpers for OD-SLP-004.

OD-SLP-004 fits one pooled DEV q-grid elsewhere and uses this module to score
that fixed law on pooled, early, and late DEV rows.  These helpers never read
source files, fit a temporal-stratum law, access HELD_OUT/RESERVE, or use RNG.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

from .sleep_committor_dev_manifest import (
    FROZEN_PID_SPLIT,
    P_REM_DEV,
    SUPPORT_FLOOR,
    calculate_transfer_tolerance,
    dev_base_rate_brier,
    dev_segments_for_stratum,
)
from .sleep_committor_qualification import (
    ALL_OUTCOMES,
    RESOLVED_OUTCOMES,
    binary_metrics,
    resolved_scoring_rows,
)

EVALUATION_STRATA = ("pooled", "early", "late")
MIN_RESOLVED_N3 = 20
MIN_RESOLVED_REM = 20
MIN_PID_N3 = 8
MIN_PID_REM = 8
MIN_PID_BOTH = 5
RELIABILITY_EDGES = np.linspace(0.0, 1.0, 11)
PRIMARY_RELATIVE_BRIER_IMPROVEMENT = 0.05
PRIMARY_LOG_LOSS_IMPROVEMENT = 0.01
ARCHIVED_Q_TOLERANCE = 1e-12


def _empty_score_payload(
    stratum: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "status": "NOT_TESTABLE",
        "stratum": stratum,
        "failure_reason": reason,
        "evaluation_support": {
            "n_resolved_N3": 0,
            "n_resolved_REM": 0,
            "N_pid_N3": 0,
            "N_pid_REM": 0,
            "N_pid_both": 0,
            "floors": {
                "n_resolved_N3": MIN_RESOLVED_N3,
                "n_resolved_REM": MIN_RESOLVED_REM,
                "N_pid_N3": MIN_PID_N3,
                "N_pid_REM": MIN_PID_REM,
                "N_pid_both": MIN_PID_BOTH,
            },
            "status": "NOT_TESTABLE",
        },
        "n_score_rows": 0,
        "metrics": None,
        "local_success": None,
    }


def _validate_segment_temporal_metadata(
    segments: Sequence[Mapping[str, Any]],
) -> str | None:
    for segment in segments:
        try:
            start = float(segment["start_sec"])
            duration = float(segment["recording_duration_sec"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return "score_segment_temporal_metadata_missing"
        if not np.isfinite(start) or not np.isfinite(duration):
            return "score_segment_temporal_metadata_nonfinite"
        expected = (
            "early_night"
            if start < 0.5 * duration
            else "late_night"
        )
        if segment.get("night_stratum") != expected:
            return "score_segment_temporal_stratum_mismatch"
    return None


def _evaluation_support(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n3_pids = {
        str(row["pid"])
        for row in rows
        if row.get("outcome") == "first_hit_n3"
    }
    rem_pids = {
        str(row["pid"])
        for row in rows
        if row.get("outcome") == "first_hit_rem"
    }
    payload = {
        "n_resolved_N3": int(
            sum(row.get("outcome") == "first_hit_n3" for row in rows)
        ),
        "n_resolved_REM": int(
            sum(row.get("outcome") == "first_hit_rem" for row in rows)
        ),
        "N_pid_N3": len(n3_pids),
        "N_pid_REM": len(rem_pids),
        "N_pid_both": len(n3_pids & rem_pids),
        "floors": {
            "n_resolved_N3": MIN_RESOLVED_N3,
            "n_resolved_REM": MIN_RESOLVED_REM,
            "N_pid_N3": MIN_PID_N3,
            "N_pid_REM": MIN_PID_REM,
            "N_pid_both": MIN_PID_BOTH,
        },
    }
    payload["status"] = (
        "PASS"
        if (
            payload["n_resolved_N3"] >= MIN_RESOLVED_N3
            and payload["n_resolved_REM"] >= MIN_RESOLVED_REM
            and payload["N_pid_N3"] >= MIN_PID_N3
            and payload["N_pid_REM"] >= MIN_PID_REM
            and payload["N_pid_both"] >= MIN_PID_BOTH
        )
        else "NOT_TESTABLE"
    )
    return payload


def _auroc(rows: Sequence[Mapping[str, Any]]) -> float | None:
    y = np.asarray([float(row["y"]) for row in rows], dtype=float)
    prediction = np.asarray(
        [float(row["prediction"]) for row in rows],
        dtype=float,
    )
    positive = int(np.sum(y == 1.0))
    negative = int(np.sum(y == 0.0))
    if positive == 0 or negative == 0:
        return None
    ranks = rankdata(prediction, method="average")
    return float(
        (np.sum(ranks[y == 1.0]) - positive * (positive + 1) / 2.0)
        / (positive * negative)
    )


def _reliability(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    bins: list[dict[str, Any]] = []
    gaps: list[float] = []
    for index, (lower, upper) in enumerate(
        zip(RELIABILITY_EDGES[:-1], RELIABILITY_EDGES[1:])
    ):
        values = [
            row
            for row in rows
            if (
                float(row["prediction"]) >= float(lower)
                and (
                    float(row["prediction"]) < float(upper)
                    or index == len(RELIABILITY_EDGES) - 2
                )
            )
        ]
        if values:
            observed = float(np.mean([float(row["y"]) for row in values]))
            predicted = float(
                np.mean([float(row["prediction"]) for row in values])
            )
            gap = abs(predicted - observed)
            gaps.append(gap)
        else:
            observed = None
            predicted = None
            gap = None
        bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "n": len(values),
                "mean_prediction": predicted,
                "observed_fraction": observed,
                "absolute_gap": gap,
            }
        )
    return {
        "edges": RELIABILITY_EDGES.tolist(),
        "bins": bins,
        "max_absolute_gap": max(gaps) if gaps else None,
    }


def _score_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    model = binary_metrics(rows)
    base_rows = [dict(row, prediction=P_REM_DEV) for row in rows]
    base = binary_metrics(base_rows)
    brier_q = float(model["brier"])
    brier_0 = float(base["brier"])
    log_q = float(model["log_loss"])
    log_0 = float(base["log_loss"])
    improvement = brier_0 - brier_q
    return {
        "n": int(model["n"]),
        "model": model,
        "base_rate": base,
        "I_s": improvement,
        "absolute_brier_improvement": improvement,
        "R_s": (
            improvement / brier_0 if brier_0 > 0 else None
        ),
        "relative_brier_improvement": (
            improvement / brier_0 if brier_0 > 0 else None
        ),
        "L_s": log_0 - log_q,
        "log_loss_improvement": log_0 - log_q,
        "reliability": _reliability(rows),
        "auroc": _auroc(rows),
    }


def _local_success(metrics: Mapping[str, Any]) -> bool:
    relative = metrics.get("R_s")
    log_loss = metrics.get("L_s")
    return bool(
        relative is not None
        and log_loss is not None
        and float(relative) >= PRIMARY_RELATIVE_BRIER_IMPROVEMENT
        and float(log_loss) >= PRIMARY_LOG_LOSS_IMPROVEMENT
    )


def score_fixed_law_surface(
    segments: Sequence[Mapping[str, Any]],
    *,
    q_grid: Sequence[float],
    query_grid: Sequence[float],
    stratum: str,
) -> dict[str, Any]:
    """Score one fixed q-grid on one DEV evaluation surface."""
    if stratum not in EVALUATION_STRATA:
        raise ValueError(f"unknown_evaluation_stratum:{stratum}")
    metadata_failure = _validate_segment_temporal_metadata(segments)
    if metadata_failure is not None:
        return _empty_score_payload(stratum, metadata_failure)
    selected = dev_segments_for_stratum(segments, stratum)
    for segment in selected:
        outcome = segment.get("outcome")
        if outcome is None or str(outcome) not in ALL_OUTCOMES:
            return _empty_score_payload(
                stratum,
                "score_segment_outcome_missing_or_unknown",
            )
    allowed_pids = {str(int(pid)) for pid in FROZEN_PID_SPLIT["DEV"]}
    for segment in selected:
        try:
            pid = str(int(float(segment["pid"])))
        except (KeyError, TypeError, ValueError, OverflowError):
            return _empty_score_payload(stratum, "score_segment_pid_invalid")
        if pid not in allowed_pids or segment.get("split") != "DEV":
            return _empty_score_payload(
                stratum,
                "score_segment_not_in_dev_split",
            )
    rows = resolved_scoring_rows(
        selected,
        q_grid,
        query_grid,
        split="DEV",
    )
    if not rows:
        return _empty_score_payload(stratum, "no_resolved_in_support_rows")
    if not all(np.isfinite(float(row["prediction"])) for row in rows):
        return _empty_score_payload(stratum, "nonfinite_fixed_prediction")
    support = _evaluation_support(rows)
    metrics = _score_metrics(rows)
    return {
        "status": "computed" if support["status"] == "PASS" else "NOT_TESTABLE",
        "stratum": stratum,
        "failure_reason": (
            None
            if support["status"] == "PASS"
            else "evaluation_support_floor_not_met"
        ),
        "evaluation_support": support,
        "n_score_rows": len(rows),
        "metrics": metrics,
        "local_success": _local_success(metrics),
        "out_of_boundary_resolved_rows": sum(
            str(segment.get("outcome")) in RESOLVED_OUTCOMES
            and (
                not np.isfinite(float(segment.get("candidate_rc", np.nan)))
                or float(segment["candidate_rc"]) < float(query_grid[0])
                or float(segment["candidate_rc"]) > float(query_grid[-1])
            )
            for segment in selected
        ),
    }


def compare_archived_pooled_fit(
    fit: Mapping[str, Any],
    archived_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare a replayed pooled fit with the hash-bound 003 JSON payload."""
    archived_fit = archived_manifest.get("fits", {}).get("pooled", {})
    try:
        live_query = np.asarray(fit["query_grid"], dtype=float)
        archived_query = np.asarray(archived_fit["query_grid"], dtype=float)
        live_q = np.asarray(fit["q_grid"], dtype=float)
        archived_q = np.asarray(archived_fit["q_grid"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return {
            "status": "NOT_TESTABLE",
            "failure_reason": "archived_pooled_fit_payload_missing",
        }
    query_match = (
        live_query.shape == archived_query.shape
        and np.all(np.isfinite(live_query))
        and np.all(np.isfinite(archived_query))
        and np.allclose(
            live_query,
            archived_query,
            rtol=0.0,
            atol=ARCHIVED_Q_TOLERANCE,
        )
    )
    q_match = (
        live_q.shape == archived_q.shape
        and np.all(np.isfinite(live_q))
        and np.all(np.isfinite(archived_q))
        and np.allclose(
            live_q,
            archived_q,
            rtol=0.0,
            atol=ARCHIVED_Q_TOLERANCE,
        )
    )
    return {
        "status": "PASS" if query_match and q_match else "NOT_TESTABLE",
        "failure_reason": (
            None
            if query_match and q_match
            else "archived_pooled_fit_mismatch"
        ),
        "query_grid_match": bool(query_match),
        "q_grid_match": bool(q_match),
        "q_tolerance": ARCHIVED_Q_TOLERANCE,
    }


def stationarity_status(
    *,
    evaluation_status: str,
    early: Mapping[str, Any],
    late: Mapping[str, Any],
    transfer_tolerance: float | None,
) -> dict[str, str]:
    """Apply the frozen layered stationarity taxonomy."""
    if evaluation_status != "PASS":
        return {
            "stationarity_status": "NOT_TESTABLE",
            "combined_status": "NOT_TESTABLE",
        }
    if transfer_tolerance is None:
        return {
            "stationarity_status": "NOT_TESTABLE",
            "combined_status": "NOT_TESTABLE",
        }
    try:
        tolerance = float(transfer_tolerance)
    except (TypeError, ValueError, OverflowError):
        return {
            "stationarity_status": "NOT_TESTABLE",
            "combined_status": "NOT_TESTABLE",
        }
    if not np.isfinite(tolerance):
        return {
            "stationarity_status": "NOT_TESTABLE",
            "combined_status": "NOT_TESTABLE",
        }
    early_success = bool(early.get("local_success"))
    late_success = bool(late.get("local_success"))
    if not early_success or not late_success:
        return {
            "stationarity_status": "METHOD_LIMITED / LOCAL_SUCCESS_FAIL",
            "combined_status": "METHOD_LIMITED / LOCAL_SUCCESS_FAIL",
        }
    try:
        early_i = float(early["metrics"]["I_s"])
        late_i = float(late["metrics"]["I_s"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return {
            "stationarity_status": "NOT_TESTABLE",
            "combined_status": "NOT_TESTABLE",
        }
    if not np.isfinite(early_i) or not np.isfinite(late_i):
        return {
            "stationarity_status": "NOT_TESTABLE",
            "combined_status": "NOT_TESTABLE",
        }
    if abs(early_i - late_i) > tolerance:
        return {
            "stationarity_status": (
                "METHOD_LIMITED / NONSTATIONARY_ACROSS_NIGHT"
            ),
            "combined_status": (
                "METHOD_LIMITED / NONSTATIONARY_ACROSS_NIGHT"
            ),
        }
    return {
        "stationarity_status": "PASS_STATIONARY",
        "combined_status": "PASS",
    }


def transfer_payload(
    *,
    dev_base_brier: float | None,
    early: Mapping[str, Any],
    late: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the frozen scalar and stratum transfer diagnostics."""
    tolerance = calculate_transfer_tolerance(dev_base_brier)
    return {
        "DEV_base_rate_Brier": dev_base_brier,
        "transfer_tolerance": tolerance,
        "early_absolute_brier_improvement": (
            early.get("metrics", {}).get("absolute_brier_improvement")
            if early.get("metrics")
            else None
        ),
        "late_absolute_brier_improvement": (
            late.get("metrics", {}).get("absolute_brier_improvement")
            if late.get("metrics")
            else None
        ),
        "early_relative_brier_improvement": (
            early.get("metrics", {}).get("R_s")
            if early.get("metrics")
            else None
        ),
        "late_relative_brier_improvement": (
            late.get("metrics", {}).get("R_s")
            if late.get("metrics")
            else None
        ),
        "early_log_loss_improvement": (
            early.get("metrics", {}).get("L_s")
            if early.get("metrics")
            else None
        ),
        "late_log_loss_improvement": (
            late.get("metrics", {}).get("L_s")
            if late.get("metrics")
            else None
        ),
        "absolute_improvement_difference": (
            abs(
                float(
                    early["metrics"]["absolute_brier_improvement"]
                )
                - float(late["metrics"]["absolute_brier_improvement"])
            )
            if early.get("metrics") and late.get("metrics")
            else None
        ),
    }
