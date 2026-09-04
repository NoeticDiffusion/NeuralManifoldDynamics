"""Synthetic fail-closed tests for the OD-SLP-000 source contract."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mndm.dynamical_families.sleep_first_hit_eligibility import (  # noqa: E402
    SleepFirstHitProtocol,
    audit_stage_intervals,
    headband_slow_fast_logratio,
    json_safe,
    normalize_stage,
    pid_split_leaks,
)
from od_slp_000_first_hit_eligibility_gate import (  # noqa: E402
    _eligible_records_and_segments,
)


def test_numeric_stage_hum_intervals_create_one_n2_candidate() -> None:
    result = audit_stage_intervals(
        onsets_sec=[0, 30, 60, 90],
        durations_sec=[30, 30, 30, 30],
        stages=[3, 2, 2, 4],
        recording_duration_sec=120,
    )
    assert result["stage_hum_available"]
    assert result["dense_grid_available"]
    assert result["candidate_n2_blocks"] == 1
    assert result["first_hit_rem"] == 1
    assert result["first_hit_n3"] == 0


def test_competing_exits_are_counted_not_dropped() -> None:
    result = audit_stage_intervals(
        onsets_sec=[0, 30, 60],
        durations_sec=[30, 30, 30],
        stages=[2, 1, 3],
        recording_duration_sec=90,
    )
    assert result["candidate_n2_blocks"] == 1
    assert result["competing_exit_n1"] == 1
    assert result["first_hit_n3"] == 0


def test_all_competing_exit_codes_are_counted() -> None:
    for exit_code, key in (
        (0, "competing_exit_wake"),
        (8, "competing_exit_disconnection"),
        (-2, "competing_exit_artifact"),
    ):
        result = audit_stage_intervals(
            onsets_sec=[0, 30],
            durations_sec=[30, 30],
            stages=[2, exit_code],
            recording_duration_sec=60,
        )
        assert result[key] == 1


def test_terminal_unstaged_tail_is_right_censor_not_internal_gap() -> None:
    result = audit_stage_intervals(
        onsets_sec=[0, 30, 60],
        durations_sec=[30, 30, 30],
        stages=[2, 2, 2],
        recording_duration_sec=95,
    )
    assert result["dense_grid_available"]
    assert result["terminal_unstaged_sec"] == 5.0
    assert result["right_censored"] == 1


def test_internal_gap_fails_dense_grid() -> None:
    result = audit_stage_intervals(
        onsets_sec=[0, 30, 70],
        durations_sec=[30, 30, 30],
        stages=[2, 3, 4],
        recording_duration_sec=100,
    )
    assert not result["dense_grid_available"]
    assert "stage_grid_gap" in result["failure_reasons"]


def test_long_terminal_tail_fails_dense_grid() -> None:
    result = audit_stage_intervals(
        onsets_sec=[0, 30],
        durations_sec=[30, 30],
        stages=[2, 2],
        recording_duration_sec=100,
    )
    assert not result["dense_grid_available"]
    assert "stage_grid_does_not_cover_recording" in result["failure_reasons"]


def test_nonpositive_stage_duration_fails_closed() -> None:
    result = audit_stage_intervals(
        onsets_sec=[0, 30],
        durations_sec=[30, 0],
        stages=[2, 3],
        recording_duration_sec=60,
    )
    assert not result["dense_grid_available"]
    assert "stage_intervals_not_positive_duration" in result["failure_reasons"]


def test_unknown_stage_is_not_silently_replaced_by_stage_ai() -> None:
    result = audit_stage_intervals(
        onsets_sec=[0, 30],
        durations_sec=[30, 30],
        stages=[2, None],
        recording_duration_sec=60,
    )
    assert "stage_hum_unmapped" in result["failure_reasons"]
    unknown_numeric = audit_stage_intervals(
        onsets_sec=[0, 30],
        durations_sec=[30, 30],
        stages=[2, 5],
        recording_duration_sec=60,
    )
    assert "stage_hum_unmapped" in unknown_numeric["failure_reasons"]


def test_external_headband_rc_is_finite_for_clean_data() -> None:
    protocol = SleepFirstHitProtocol()
    fs = 256.0
    time = np.arange(int(60 * fs), dtype=float) / fs
    signal = np.sin(2 * np.pi * 2.0 * time)
    data = np.vstack([signal, signal])
    values = headband_slow_fast_logratio(
        data,
        sampling_frequency_hz=fs,
        intervals=[(0.0, 30.0), (30.0, 60.0)],
        protocol=protocol,
    )
    assert values.shape == (2,)
    assert np.all(np.isfinite(values))


def test_nonfinite_headband_data_fails_rc_support() -> None:
    data = np.ones((2, 256 * 30), dtype=float)
    data[0, 10] = np.nan
    values = headband_slow_fast_logratio(
        data,
        sampling_frequency_hz=256.0,
        intervals=[(0.0, 30.0)],
    )
    assert not np.isfinite(values[0])


def test_pid_split_leakage_is_detected() -> None:
    assert not pid_split_leaks({"89": "all_data"})
    assert pid_split_leaks({"89": ["train", "test"]})


def test_stage_codebook_and_json_are_explicit() -> None:
    assert normalize_stage(3.0) == 3
    assert normalize_stage("REM") == 4
    assert normalize_stage(8) == 8
    assert normalize_stage(-2) == -2
    assert json_safe({"x": float("nan"), "n": np.int64(2)}) == {
        "x": None,
        "n": 2,
    }


def test_runner_keeps_competing_and_censored_segments_in_eligibility_pool() -> None:
    records = [
        {
            "failure_reasons": [],
            "eligible_segments": 3,
            "segments": [
                {"external_rc_finite": True, "outcome": "first_hit_rem"},
                {"external_rc_finite": True, "outcome": "competing_exit_wake"},
                {"external_rc_finite": True, "outcome": "right_censored"},
            ],
        }
    ]
    eligible_records, segments = _eligible_records_and_segments(records)
    assert len(eligible_records) == 1
    assert [item["outcome"] for item in segments] == [
        "first_hit_rem",
        "competing_exit_wake",
        "right_censored",
    ]


def test_runner_segment_floor_counts_segments_not_recordings() -> None:
    records = [
        {
            "failure_reasons": [],
            "eligible_segments": 20,
            "segments": [
                {"external_rc_finite": True, "outcome": "right_censored"}
            ]
            * 20,
        }
    ]
    _, segments = _eligible_records_and_segments(records)
    assert len(segments) == 20
