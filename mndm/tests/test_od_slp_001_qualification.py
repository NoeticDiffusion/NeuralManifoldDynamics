"""Synthetic fail-closed tests for OD-SLP-001 mechanics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.dynamical_families.sleep_committor_qualification import (  # noqa: E402
    FROZEN_PID_SPLIT,
    assemble_adapter_arrays,
    assert_frozen_pid_split,
    binary_metrics,
    build_n2_first_hit_segments,
    canonical_night_key,
    canonical_segment_id,
    greedy_wrong_pid_map,
    permute_outcomes,
    pid_digest,
    recompute_pid_split,
    resolved_scoring_rows,
    shift_reaction_coordinate,
    bootstrap_improvements,
)
from mndm.dynamical_families.sleep_first_hit_eligibility import (  # noqa: E402
    audit_stage_intervals,
)


def _record(participant_id: str = "sub-1", task: str = "Sleep") -> dict[str, str]:
    return {
        "participant_id": participant_id,
        "session": None,
        "task": task,
        "run": None,
    }


def test_frozen_pid_split_recomputes_exactly() -> None:
    pids = [pid for values in FROZEN_PID_SPLIT.values() for pid in values]
    result = assert_frozen_pid_split(pids)
    assert result["matches_frozen_lists"]
    assert recompute_pid_split(pids) == FROZEN_PID_SPLIT


def test_pid_split_mismatch_fails_closed() -> None:
    pids = [pid for values in FROZEN_PID_SPLIT.values() for pid in values]
    result = assert_frozen_pid_split(pids[:-1])
    assert not result["matches_frozen_lists"]
    assert result["reason"] == "unexpected_pid_count"


def test_pid_digest_and_night_keys_are_canonical() -> None:
    assert pid_digest("015") == pid_digest(15)
    record = _record()
    assert canonical_night_key(record) == "sub-1|NA|Sleep|NA"
    assert canonical_segment_id(record, 4) == "sub-1|NA|Sleep|NA|candidate-4"


def test_resolved_segment_has_terminal_carrier_without_exit_row() -> None:
    stage = audit_stage_intervals(
        onsets_sec=[0, 30, 60],
        durations_sec=[30, 30, 30],
        stages=[2, 2, 4],
        recording_duration_sec=90,
    )
    segments = build_n2_first_hit_segments(
        stage_audit=stage,
        reaction_coordinate=[0.1, 0.2, 0.9],
        record=_record(),
        pid=15,
        recording_duration_sec=90,
    )
    assert len(segments) == 1
    assert segments[0]["n2_window_count"] == 2
    assert segments[0]["regime_labels"].tolist() == [2, 4]
    assert segments[0]["reaction_coordinate"].tolist() == [0.1, 0.2]


def test_competing_segment_keeps_n2_labels_only() -> None:
    stage = audit_stage_intervals(
        onsets_sec=[0, 30],
        durations_sec=[30, 30],
        stages=[2, 1],
        recording_duration_sec=60,
    )
    segments = build_n2_first_hit_segments(
        stage_audit=stage,
        reaction_coordinate=[0.1, 0.2],
        record=_record(),
        pid=15,
        recording_duration_sec=60,
    )
    assert segments[0]["outcome"] == "competing_exit_n1"
    assert segments[0]["regime_labels"].tolist() == [2]


def test_assembly_does_not_create_cross_segment_increment() -> None:
    stage = audit_stage_intervals(
        onsets_sec=[0, 30, 60, 90, 120, 150],
        durations_sec=[30, 30, 30, 30, 30, 30],
        stages=[2, 2, 4, 2, 2, 3],
        recording_duration_sec=180,
    )
    segments = build_n2_first_hit_segments(
        stage_audit=stage,
        reaction_coordinate=[0.1, 0.2, 0.3, -0.1, -0.2, -0.3],
        record=_record(),
        pid=15,
        recording_duration_sec=180,
    )
    arrays = assemble_adapter_arrays(segments)
    assert arrays["segment_id"].tolist() == [0, 0, 1, 1]
    transitions = arrays["segment_id"][1:] == arrays["segment_id"][:-1]
    assert transitions.tolist() == [True, False, True]


def test_shift_is_within_night_and_deterministic() -> None:
    values = np.arange(5, dtype=float)
    assert shift_reaction_coordinate(values, 2).tolist() == [3, 4, 0, 1, 2]
    assert np.array_equal(
        shift_reaction_coordinate(values, 2),
        shift_reaction_coordinate(values, 2),
    )


def test_wrong_pid_map_is_deterministic_and_cross_pid() -> None:
    nights = [
        {"participant_id": "sub-1", "session": None, "task": "Sleep", "run": None, "pid": 1, "interval_count": 100},
        {"participant_id": "sub-2", "session": None, "task": "Sleep", "run": None, "pid": 2, "interval_count": 101},
        {"participant_id": "sub-3", "session": None, "task": "Sleep", "run": None, "pid": 3, "interval_count": 100},
        {"participant_id": "sub-4", "session": None, "task": "Sleep", "run": None, "pid": 4, "interval_count": 101},
    ]
    mapping, reason = greedy_wrong_pid_map(nights)
    assert reason is None
    assert len(mapping) == 4
    assert all(source != donor for source, donor in mapping.items())
    pid_by_key = {canonical_night_key(item): str(item["pid"]) for item in nights}
    assert all(pid_by_key[source] != pid_by_key[donor] for source, donor in mapping.items())


def test_wrong_pid_map_refuses_unmatched_source() -> None:
    nights = [
        {"participant_id": "sub-1", "session": None, "task": "Sleep", "run": None, "pid": 1, "interval_count": 100},
        {"participant_id": "sub-2", "session": None, "task": "Sleep", "run": None, "pid": 2, "interval_count": 200},
    ]
    mapping, reason = greedy_wrong_pid_map(nights)
    assert mapping == {}
    assert reason is not None


def test_label_permutation_preserves_outcomes_and_is_repeatable() -> None:
    segments = [
        {"segment_id": "b", "outcome": "first_hit_n3"},
        {"segment_id": "a", "outcome": "first_hit_rem"},
        {"segment_id": "c", "outcome": "right_censored"},
    ]
    first = permute_outcomes(segments)
    second = permute_outcomes(segments)
    assert first == second
    assert sorted(first.values()) == sorted(item["outcome"] for item in segments)


def test_scoring_excludes_out_of_boundary_candidate_starts() -> None:
    segments = [
        {
            "segment_id": "in",
            "pid": "1",
            "outcome": "first_hit_rem",
            "candidate_rc": 0.0,
            "night_stratum": "early_night",
        },
        {
            "segment_id": "out",
            "pid": "2",
            "outcome": "first_hit_n3",
            "candidate_rc": 2.0,
            "night_stratum": "late_night",
        },
    ]
    rows = resolved_scoring_rows(segments, [0.0, 1.0], [-1.0, 1.0])
    assert [row["segment_id"] for row in rows] == ["in"]


def test_binary_metrics_clip_endpoint_predictions() -> None:
    metrics = binary_metrics(
        [
            {"y": 1.0, "prediction": 1.0},
            {"y": 0.0, "prediction": 0.0},
        ]
    )
    assert np.isfinite(metrics["log_loss"])
    assert metrics["brier"] == 0.0


def test_bootstrap_requires_both_outcomes_in_each_replicate() -> None:
    rows = [
        {"segment_id": "a", "pid": "1", "y": 0.0, "prediction": 0.2},
        {"segment_id": "b", "pid": "2", "y": 1.0, "prediction": 0.8},
    ]
    result = bootstrap_improvements(
        rows,
        reference_prediction=[0.5, 0.5],
        seed=1,
        replicates=10,
    )
    assert result["status"] == "not_testable"
    assert result["reason"] == "bootstrap_replicate_without_both_outcomes"
