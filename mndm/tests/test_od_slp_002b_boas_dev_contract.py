"""Synthetic fail-closed tests for the OD-SLP-002B DEV audit contract."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.dynamical_families.sleep_committor_dev_audit import (  # noqa: E402
    EVALUATION_ORDER,
    FROZEN_PID_SPLIT,
    SUPPORT_FLOOR,
    endpoint_census,
    q_free_support_audit,
    select_supported_grid,
)


def _segment(
    pid: int,
    outcome: str,
    *,
    source: float = 0.5,
    candidate_index: int = 0,
) -> dict[str, object]:
    return {
        "segment_id": f"pid-{pid}-{outcome}-{candidate_index}",
        "pid": pid,
        "outcome": outcome,
        "reaction_coordinate": np.asarray([source, source + 0.001]),
        "time": np.asarray([0.0, 0.001]),
        "regime_labels": np.asarray([2, 2], dtype=np.int16),
        "external_rc_finite": True,
    }


def _endpoint_pass_segments() -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    dev_pids = list(FROZEN_PID_SPLIT["DEV"])
    for pid in dev_pids[:10]:
        for replicate in range(2):
            segments.append(
                _segment(pid, "first_hit_n3", candidate_index=replicate)
            )
    for pid in dev_pids[:5]:
        for replicate in range(2):
            segments.append(
                _segment(pid, "first_hit_rem", candidate_index=replicate)
            )
    for pid in dev_pids[10:20]:
        segments.append(_segment(pid, "first_hit_rem"))
    segments.extend(
        [
            _segment(dev_pids[20], "competing_exit_wake"),
            _segment(dev_pids[21], "competing_exit_n1"),
            _segment(dev_pids[22], "competing_exit_disconnection"),
            _segment(dev_pids[23], "competing_exit_artifact"),
            _segment(dev_pids[24], "qc_or_gap_exit"),
            _segment(dev_pids[25], "right_censored"),
        ]
    )
    return segments


def _support_segments(
    *,
    missing_grid_index: int | None = None,
) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    dev_pids = list(FROZEN_PID_SPLIT["DEV"])
    query_grid = np.linspace(0.0, 1.0, 9)
    segment_id = 0
    for grid_index, source in enumerate(query_grid):
        if grid_index == missing_grid_index:
            continue
        for replicate in range(3):
            segment = _segment(
                dev_pids[segment_id % len(dev_pids)],
                "right_censored",
                source=float(source),
                candidate_index=replicate,
            )
            segment["segment_id"] = f"support-{segment_id}"
            segments.append(segment)
            segment_id += 1
    return segments


def test_endpoint_census_passes_pid_floors_and_all_eight_keys() -> None:
    result = endpoint_census(
        _endpoint_pass_segments(),
        dev_pids=FROZEN_PID_SPLIT["DEV"],
    )
    assert result["status"] == "ENDPOINT_PASS"
    assert set(result["outcome_counts"]) == {
        "first_hit_n3",
        "first_hit_rem",
        "competing_exit_wake",
        "competing_exit_n1",
        "competing_exit_disconnection",
        "competing_exit_artifact",
        "qc_or_gap_exit",
        "right_censored",
    }
    assert result["N_pid_N3"] == 10
    assert result["N_pid_REM"] == 15
    assert result["N_pid_both"] == 5
    assert result["concentration"]["N3"]["top3_share"] <= 0.50
    assert result["concentration"]["REM"]["top3_share"] <= 0.50
    assert result["outcome_counts"]["competing_exit_wake"] == 1
    assert result["outcome_counts"]["competing_exit_n1"] == 1
    assert result["outcome_counts"]["competing_exit_disconnection"] == 1
    assert result["outcome_counts"]["competing_exit_artifact"] == 1
    assert result["outcome_counts"]["qc_or_gap_exit"] == 1
    assert result["outcome_counts"]["right_censored"] == 1
    assert result["p_AB_DEV"] == (20 + 20) / len(_endpoint_pass_segments())
    assert len(result["pid_counts"]) == 30


def test_pid_concentration_fails_closed_even_with_event_floor() -> None:
    dev_pids = list(FROZEN_PID_SPLIT["DEV"])
    segments = [
        _segment(pid, "first_hit_n3", candidate_index=replicate)
        for pid in dev_pids[:3]
        for replicate in range(7)
    ]
    segments.extend(
        _segment(pid, "first_hit_rem", candidate_index=replicate)
        for pid in dev_pids[3:23]
        for replicate in range(1)
    )
    result = endpoint_census(segments, dev_pids=FROZEN_PID_SPLIT["DEV"])
    assert result["status"] == "ENDPOINT_INADEQUATE"
    assert result["n3_count"] == 21
    assert "n3_top3_concentration" in result["endpoint_failure_reasons"]
    assert "n3_pid_floor" in result["endpoint_failure_reasons"]


def test_event_floor_failure_is_independent_of_pid_floor() -> None:
    dev_pids = list(FROZEN_PID_SPLIT["DEV"])
    segments = [
        _segment(pid, "first_hit_n3", candidate_index=replicate)
        for pid in dev_pids[:10]
        for replicate in range(2)
    ][:-1]
    segments.extend(
        _segment(pid, "first_hit_rem", candidate_index=replicate)
        for pid in dev_pids[:10]
        for replicate in range(2)
    )
    result = endpoint_census(segments, dev_pids=dev_pids)
    assert result["n3_count"] == 19
    assert result["N_pid_N3"] == 10
    assert "n3_event_floor" in result["endpoint_failure_reasons"]


def test_both_outcome_pid_floor_is_independent() -> None:
    dev_pids = list(FROZEN_PID_SPLIT["DEV"])
    segments = [
        _segment(pid, "first_hit_n3", candidate_index=replicate)
        for pid in dev_pids[:10]
        for replicate in range(2)
    ]
    segments.extend(
        _segment(pid, "first_hit_rem", candidate_index=replicate)
        for pid in dev_pids[10:20]
        for replicate in range(2)
    )
    result = endpoint_census(segments, dev_pids=dev_pids)
    assert result["n3_count"] == 20
    assert result["rem_count"] == 20
    assert result["N_pid_both"] == 0
    assert "both_outcome_pid_floor" in result["endpoint_failure_reasons"]


def test_unknown_outcome_is_endpoint_not_testable() -> None:
    dev_pid = FROZEN_PID_SPLIT["DEV"][0]
    result = endpoint_census(
        [_segment(dev_pid, "unregistered_outcome")],
        dev_pids=FROZEN_PID_SPLIT["DEV"],
    )
    assert result["status"] == "ENDPOINT_NOT_TESTABLE"
    assert result["failure_reason"] == "malformed_or_unknown_endpoint_segment"


def test_empty_endpoint_has_defined_concentration_diagnostics() -> None:
    result = endpoint_census([], dev_pids=FROZEN_PID_SPLIT["DEV"])
    assert result["status"] == "ENDPOINT_INADEQUATE"
    assert result["concentration"]["N3"]["top3_share"] == 1.0
    assert result["concentration"]["N3"]["max_pid_share"] == 1.0
    assert result["concentration"]["N3"]["n_eff"] == 0.0


def test_held_out_pid_cannot_enter_dev_census() -> None:
    result = endpoint_census(
        [_segment(FROZEN_PID_SPLIT["HELD_OUT"][0], "first_hit_n3")],
        dev_pids=FROZEN_PID_SPLIT["DEV"],
    )
    assert result["status"] == "ENDPOINT_NOT_TESTABLE"


def test_descending_selection_is_support_only() -> None:
    assert EVALUATION_ORDER == (65, 33, 17, 9)
    selection = select_supported_grid(
        {65: "PASS", 33: "PASS", 17: "PASS", 9: "PASS"},
        {
            65: {"status": "SUPPORT_NOT_TESTABLE"},
            33: {"status": "SUPPORT_PASS"},
            17: {"status": "SUPPORT_PASS"},
            9: {"status": "SUPPORT_PASS"},
        },
    )
    assert selection["selected_grid"] == 33
    assert selection["endpoint_outcomes_used"] is False
    assert selection["q_values_used"] is False


def test_default_support_floor_is_64() -> None:
    assert SUPPORT_FLOOR == 64


def test_q_free_support_passes_without_q_fields() -> None:
    result = q_free_support_audit(
        _support_segments(),
        lower=0.0,
        upper=1.0,
        grid_resolution=9,
        min_support=1,
        min_transition_segments=20,
    )
    assert result["status"] == "SUPPORT_PASS"
    assert len(result["support_count"]) == 9
    assert "q_grid" not in result
    assert "drift_estimate_grid" not in result


def test_default_support_floor_rejects_small_synthetic_support() -> None:
    result = q_free_support_audit(
        _support_segments(),
        lower=0.0,
        upper=1.0,
        grid_resolution=9,
        min_transition_segments=20,
    )
    assert result["status"] == "SUPPORT_NOT_TESTABLE"
    assert result["failure_reason"] == "dense_query_grid_has_under_supported_points"


def test_one_window_segment_adds_no_support_pair() -> None:
    segments = _support_segments()
    segments.append(
        {
            "segment_id": "one-window",
            "pid": FROZEN_PID_SPLIT["DEV"][0],
            "outcome": "right_censored",
            "reaction_coordinate": np.asarray([0.5]),
            "time": np.asarray([0.0]),
            "regime_labels": np.asarray([2], dtype=np.int16),
            "external_rc_finite": True,
        }
    )
    result = q_free_support_audit(
        segments,
        lower=0.0,
        upper=1.0,
        grid_resolution=9,
        min_support=1,
        min_transition_segments=20,
    )
    assert result["n_increment_pairs"] == 27


def test_terminal_carrier_pair_remains_support_eligible() -> None:
    segments = _support_segments()
    for segment in segments:
        segment["regime_labels"] = np.asarray([2, 4], dtype=np.int16)
    result = q_free_support_audit(
        segments,
        lower=0.0,
        upper=1.0,
        grid_resolution=9,
        min_support=1,
        min_transition_segments=20,
    )
    assert result["n_increment_pairs"] == 27


def test_non_dev_support_segment_fails_closed() -> None:
    segments = _support_segments()
    segments[0]["pid"] = FROZEN_PID_SPLIT["HELD_OUT"][0]
    result = q_free_support_audit(
        segments,
        lower=0.0,
        upper=1.0,
        grid_resolution=9,
        min_support=1,
        min_transition_segments=20,
    )
    assert result["status"] == "SUPPORT_NOT_TESTABLE"
    assert result["failure_reason"] == "support_segment_not_in_dev_split"


def test_one_under_supported_point_rejects_grid() -> None:
    result = q_free_support_audit(
        _support_segments(missing_grid_index=4),
        lower=0.0,
        upper=1.0,
        grid_resolution=9,
        min_support=1,
        min_transition_segments=20,
    )
    assert result["status"] == "SUPPORT_NOT_TESTABLE"
    assert result["failure_reason"] == "dense_query_grid_has_under_supported_points"
    assert result["support_count"][4] == 0


def test_cross_segment_boundary_never_creates_support_increment() -> None:
    dev_pids = list(FROZEN_PID_SPLIT["DEV"])
    segments = [
        _segment(
            dev_pids[index % len(dev_pids)],
            "right_censored",
            source=0.1 if index % 2 else 0.9,
        )
        for index in range(15)
    ]
    result = q_free_support_audit(
        segments,
        lower=0.0,
        upper=1.0,
        grid_resolution=9,
        min_support=1,
        min_transition_segments=1,
    )
    assert result["n_increment_pairs"] == 15


def test_tq_fail_or_not_testable_grid_cannot_be_selected() -> None:
    selection = select_supported_grid(
        {65: "FAIL", 33: "NOT_TESTABLE", 17: "PASS", 9: "PASS"},
        {
            65: {"status": "SUPPORT_PASS"},
            33: {"status": "SUPPORT_PASS"},
            17: {"status": "SUPPORT_PASS"},
            9: {"status": "SUPPORT_PASS"},
        },
    )
    assert selection["selected_grid"] == 17


def test_no_supported_tq_grid_is_not_testable() -> None:
    selection = select_supported_grid(
        {65: "FAIL", 33: "NOT_TESTABLE", 17: "PASS", 9: "PASS"},
        {
            65: {"status": "SUPPORT_NOT_TESTABLE"},
            33: {"status": "SUPPORT_NOT_TESTABLE"},
            17: {"status": "SUPPORT_NOT_TESTABLE"},
            9: {"status": "SUPPORT_NOT_TESTABLE"},
        },
    )
    assert selection["selected_grid"] is None
