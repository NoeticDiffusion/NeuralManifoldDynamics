"""Synthetic contract tests for OD-SLP-004 score-only stationarity."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.dynamical_families.sleep_committor_stationarity_audit import (  # noqa: E402
    PRIMARY_LOG_LOSS_IMPROVEMENT,
    score_fixed_law_surface,
    stationarity_status,
)
from mndm.dynamical_families.sleep_committor_dev_manifest import (  # noqa: E402
    FROZEN_PID_SPLIT,
    LOWER_BOUNDARY,
    UPPER_BOUNDARY,
)


def _segment(
    index: int,
    *,
    stratum: str,
    outcome: str,
    pid: int,
) -> dict[str, object]:
    start = 10.0 if stratum == "early" else 60.0
    return {
        "segment_id": f"{stratum}-{outcome}-{index}",
        "pid": str(pid),
        "split": "DEV",
        "start_sec": start,
        "recording_duration_sec": 100.0,
        "night_stratum": f"{stratum}_night",
        "candidate_rc": 2.0,
        "outcome": outcome,
        "reaction_coordinate": np.asarray([2.0, 2.1]),
        "time": np.asarray([0.0, 1.0]),
        "regime_labels": np.asarray([2, 3 if outcome == "first_hit_n3" else 4]),
        "external_rc_finite": True,
    }


def _segments() -> list[dict[str, object]]:
    pids = list(FROZEN_PID_SPLIT["DEV"][:8])
    rows: list[dict[str, object]] = []
    for stratum in ("early", "late"):
        for index in range(20):
            rows.append(
                _segment(
                    index,
                    stratum=stratum,
                    outcome="first_hit_n3",
                    pid=pids[index % len(pids)],
                )
            )
        for index in range(30):
            rows.append(
                _segment(
                    index,
                    stratum=stratum,
                    outcome="first_hit_rem",
                    pid=pids[index % len(pids)],
                )
            )
    return rows


def _q_grid() -> tuple[list[float], list[float]]:
    return [LOWER_BOUNDARY, UPPER_BOUNDARY], [0.6, 0.6]


def test_fixed_law_scores_with_frozen_y_encoding_and_floors() -> None:
    query, q = _q_grid()
    result = score_fixed_law_surface(
        _segments(),
        q_grid=q,
        query_grid=query,
        stratum="early",
    )
    assert result["status"] == "computed"
    assert result["evaluation_support"]["n_resolved_N3"] == 20
    assert result["evaluation_support"]["n_resolved_REM"] == 30
    assert result["evaluation_support"]["N_pid_N3"] == 8
    assert result["evaluation_support"]["N_pid_REM"] == 8
    assert result["evaluation_support"]["N_pid_both"] == 8
    assert result["metrics"]["model"]["rem_rate"] == 0.6
    assert result["metrics"]["model"]["n"] == 50
    assert result["metrics"]["I_s"] == result["metrics"][
        "absolute_brier_improvement"
    ]
    assert result["metrics"]["R_s"] == result["metrics"][
        "relative_brier_improvement"
    ]
    assert result["metrics"]["L_s"] == result["metrics"][
        "log_loss_improvement"
    ]
    assert result["local_success"] is True
    assert result["metrics"]["log_loss_improvement"] >= PRIMARY_LOG_LOSS_IMPROVEMENT


def test_fixed_law_missing_duration_fails_closed() -> None:
    query, q = _q_grid()
    rows = _segments()
    rows[0] = dict(rows[0])
    rows[0].pop("recording_duration_sec")
    result = score_fixed_law_surface(
        rows,
        q_grid=q,
        query_grid=query,
        stratum="early",
    )
    assert result["status"] == "NOT_TESTABLE"
    assert result["failure_reason"] == "score_segment_temporal_metadata_missing"


def test_evaluation_floor_failure_is_not_stationarity_failure() -> None:
    query, q = _q_grid()
    rows = _segments()[:5]
    result = score_fixed_law_surface(
        rows,
        q_grid=q,
        query_grid=query,
        stratum="early",
    )
    assert result["status"] == "NOT_TESTABLE"
    statuses = stationarity_status(
        evaluation_status="NOT_TESTABLE",
        early=result,
        late=result,
        transfer_tolerance=0.06,
    )
    assert statuses == {
        "stationarity_status": "NOT_TESTABLE",
        "combined_status": "NOT_TESTABLE",
    }


def test_missing_transfer_tolerance_is_not_testable() -> None:
    metrics = {"I_s": 0.1}
    statuses = stationarity_status(
        evaluation_status="PASS",
        early={"local_success": False, "metrics": metrics},
        late={"local_success": True, "metrics": metrics},
        transfer_tolerance=None,
    )
    assert statuses == {
        "stationarity_status": "NOT_TESTABLE",
        "combined_status": "NOT_TESTABLE",
    }


def test_nonfinite_transfer_tolerance_is_not_testable() -> None:
    statuses = stationarity_status(
        evaluation_status="PASS",
        early={"local_success": True, "metrics": {"I_s": 0.1}},
        late={"local_success": True, "metrics": {"I_s": 0.1}},
        transfer_tolerance=float("nan"),
    )
    assert statuses["stationarity_status"] == "NOT_TESTABLE"


def test_stationarity_status_distinguishes_local_failure_and_nonstationarity() -> None:
    metrics = {
        "I_s": 0.1,
    }
    early = {"local_success": False, "metrics": metrics}
    late = {"local_success": True, "metrics": metrics}
    assert stationarity_status(
        evaluation_status="PASS",
        early=early,
        late=late,
        transfer_tolerance=0.06,
    )["stationarity_status"] == "METHOD_LIMITED / LOCAL_SUCCESS_FAIL"

    early = {
        "local_success": True,
        "metrics": {"I_s": 0.01},
    }
    late = {
        "local_success": True,
        "metrics": {"I_s": 0.2},
    }
    assert stationarity_status(
        evaluation_status="PASS",
        early=early,
        late=late,
        transfer_tolerance=0.06,
    )["stationarity_status"] == (
        "METHOD_LIMITED / NONSTATIONARY_ACROSS_NIGHT"
    )


def test_temporal_stratum_mismatch_fails_closed() -> None:
    query, q = _q_grid()
    rows = _segments()
    rows[0] = dict(rows[0], night_stratum="late_night")
    result = score_fixed_law_surface(
        rows,
        q_grid=q,
        query_grid=query,
        stratum="early",
    )
    assert result["status"] == "NOT_TESTABLE"
    assert result["failure_reason"] == "score_segment_temporal_stratum_mismatch"


def test_missing_outcome_fails_closed() -> None:
    query, q = _q_grid()
    rows = _segments()
    rows[0] = dict(rows[0], outcome=None)
    result = score_fixed_law_surface(
        rows,
        q_grid=q,
        query_grid=query,
        stratum="early",
    )
    assert result["status"] == "NOT_TESTABLE"
    assert result["failure_reason"] == "score_segment_outcome_missing_or_unknown"


def test_unknown_outcome_fails_closed() -> None:
    query, q = _q_grid()
    rows = _segments()
    rows[0] = dict(rows[0], outcome="unknown")
    result = score_fixed_law_surface(
        rows,
        q_grid=q,
        query_grid=query,
        stratum="early",
    )
    assert result["status"] == "NOT_TESTABLE"
    assert result["failure_reason"] == "score_segment_outcome_missing_or_unknown"
