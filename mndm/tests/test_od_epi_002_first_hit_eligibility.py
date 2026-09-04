"""Unit tests for the protocol-only OD-EPI-002 eligibility checks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.dynamical_families.first_hit_eligibility import (  # noqa: E402
    FirstHitEligibilityProtocol,
    audit_first_hit_windows,
    explicit_reaction_coordinate_present,
    json_safe,
    max_consecutive_true,
    pair_first_onset_with_next_offset,
    subject_split_leaks,
)
from mndm.pipeline.event_annotations import load_event_table_from_bids_events  # noqa: E402
from mndm.pipeline.intervals import (  # noqa: E402
    TimeInterval,
    WindowMembershipSpec,
    window_membership_mask,
)


def _windows(n: int = 28) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.arange(n, dtype=float) * 4.0
    return starts, starts + 8.0, np.arange(n, dtype=np.int64)


def test_full_containment_excludes_window_that_overlaps_onset() -> None:
    starts, ends, _ = _windows()
    mids = 0.5 * (starts + ends)
    fully_contained = window_membership_mask(
        t_start=starts,
        t_end=ends,
        t_mid=mids,
        interval=TimeInterval(0.0, 14.0),
        spec=WindowMembershipSpec(mode="fully_contained"),
    )
    midpoint = window_membership_mask(
        t_start=starts,
        t_end=ends,
        t_mid=mids,
        interval=TimeInterval(0.0, 14.0),
        spec=WindowMembershipSpec(mode="midpoint_in_interval"),
    )
    assert fully_contained[0]
    assert not fully_contained[2]
    assert midpoint[2]


def test_onset_smear_window_is_not_an_absorbing_core_window() -> None:
    starts, ends, epochs = _windows()
    result = audit_first_hit_windows(
        onset_sec=40.0,
        offset_sec=100.0,
        t_start=starts,
        t_end=ends,
        epoch_id=epochs,
        reaction_coordinate_column_present=True,
    )
    assert result["n_onset_smear_windows"] == 2
    assert result["n_stable_A_windows_fully_contained"] == 7
    assert result["n_B_core_windows_fully_contained"] == 12
    assert result["is_continuous_first_hit_candidate"]


def test_missing_reaction_coordinate_is_hard_blocker() -> None:
    starts, ends, epochs = _windows()
    result = audit_first_hit_windows(
        onset_sec=40.0,
        offset_sec=100.0,
        t_start=starts,
        t_end=ends,
        epoch_id=epochs,
        reaction_coordinate_column_present=False,
    )
    assert not result["is_continuous_first_hit_candidate"]
    assert "explicit_reaction_coordinate_column_absent" in result["failure_reasons"]


def test_gap_or_qc_break_rejects_stable_a_continuity() -> None:
    starts, ends, epochs = _windows()
    starts[3] += 4.0
    ends[3] += 4.0
    qc_ok = np.ones(starts.size, dtype=bool)
    qc_ok[2] = False
    result = audit_first_hit_windows(
        onset_sec=40.0,
        offset_sec=100.0,
        t_start=starts,
        t_end=ends,
        epoch_id=epochs,
        qc_ok=qc_ok,
        reaction_coordinate_column_present=True,
    )
    assert result["gap_or_qc_break_in_A"]
    assert "gap_or_qc_break_in_A" in result["failure_reasons"]


def test_b_core_and_transition_qc_breaks_are_fail_closed() -> None:
    starts, ends, epochs = _windows()
    qc_ok = np.ones(starts.size, dtype=bool)
    qc_ok[15] = False
    result = audit_first_hit_windows(
        onset_sec=40.0,
        offset_sec=100.0,
        t_start=starts,
        t_end=ends,
        epoch_id=epochs,
        qc_ok=qc_ok,
        reaction_coordinate_column_present=True,
    )
    assert result["gap_or_qc_break_in_B"]
    assert result["gap_or_qc_break_in_A_to_B"]
    assert "gap_or_qc_break_in_B" in result["failure_reasons"]


def test_window_grid_contract_is_enforced() -> None:
    starts, ends, epochs = _windows()
    starts[10] += 1.0
    ends[10] += 1.0
    result = audit_first_hit_windows(
        onset_sec=40.0,
        offset_sec=100.0,
        t_start=starts,
        t_end=ends,
        epoch_id=epochs,
        reaction_coordinate_column_present=True,
    )
    assert "window_grid_contract_violation" in result["failure_reasons"]


def test_onset_smear_never_enters_core_masks() -> None:
    starts, ends, epochs = _windows()
    result = audit_first_hit_windows(
        onset_sec=40.0,
        offset_sec=100.0,
        t_start=starts,
        t_end=ends,
        epoch_id=epochs,
        reaction_coordinate_column_present=True,
    )
    assert result["n_onset_smear_windows"] == 2
    assert "onset_smear_overlaps_core_membership" not in result["failure_reasons"]


def test_task_labels_do_not_create_transition_segments() -> None:
    starts, ends, epochs = _windows()
    result = audit_first_hit_windows(
        onset_sec=None,
        offset_sec=None,
        t_start=starts,
        t_end=ends,
        epoch_id=epochs,
        task="interictal",
        reaction_coordinate_column_present=True,
    )
    assert not result["is_continuous_first_hit_candidate"]
    assert "non_ictal_recording_not_transition_segment" in result["failure_reasons"]


def test_event_pairing_uses_next_later_offset_point() -> None:
    pair = pair_first_onset_with_next_offset(
        ["sz offset", "sz onset", "sz offset"],
        [80.0, 20.0, 60.0],
    )
    assert pair == (20.0, 60.0)


def test_event_loader_preserves_explicit_seizure_event_types(tmp_path: Path) -> None:
    path = tmp_path / "sub-test_events.tsv"
    path.write_text(
        "onset\tduration\ttrial_type\n"
        "20\t0\tsz onset\n"
        "60\t0\tsz offset\n"
        "10\t0\tother\n",
        encoding="utf-8",
    )
    table = load_event_table_from_bids_events(
        path,
        event_types=["sz onset", "sz offset"],
        trial_type_column="trial_type",
    )
    assert table.n == 2
    assert list(table.event_type) == ["sz onset", "sz offset"]
    assert np.allclose(table.duration_sec, [0.0, 0.0])


def test_support_and_subject_partition_helpers_are_fail_closed() -> None:
    assert max_consecutive_true([True, True, False, True]) == 2
    assert explicit_reaction_coordinate_present(["x", "reaction_coordinate"], "reaction_coordinate")
    assert not explicit_reaction_coordinate_present(["x"], "reaction_coordinate")
    assert not subject_split_leaks(
        {"r1": {"subject": "S1", "split": "train"}, "r2": {"subject": "S1", "split": "train"}}
    )
    assert subject_split_leaks(
        {"r1": {"subject": "S1", "split": "train"}, "r2": {"subject": "S1", "split": "test"}}
    )


def test_protocol_defaults_are_explicit() -> None:
    protocol = FirstHitEligibilityProtocol()
    assert protocol.window_sec == 8.0
    assert protocol.step_sec == 4.0
    assert protocol.core_membership == "fully_contained"
    assert protocol.onset_smear_membership == "contains_onset"
    assert protocol.reaction_coordinate_key is None


def test_json_safe_converts_nan_and_numpy_scalars() -> None:
    value = json_safe({"missing": float("nan"), "count": np.int64(3)})
    assert value == {"missing": None, "count": 3}
