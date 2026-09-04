"""Synthetic contract tests for the OD-EMP-SCOUT-2 metadata inventory."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.dynamical_families.emp_scout_2 import (  # noqa: E402
    _adjudicate,
    _select_samples,
    classify_variable,
    scan_dataset,
)


def _write_fixture(root: Path) -> None:
    (root / "README.md").write_text(
        "Task source metadata fixture\n",
        encoding="utf-8",
    )
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "fixture"}),
        encoding="utf-8",
    )
    (root / "participants.tsv").write_text(
        "participant_id\tconfidence\nsub-01\t0.8\n",
        encoding="utf-8",
    )
    eeg = root / "sub-01" / "eeg"
    beh = root / "sub-01" / "beh"
    eeg.mkdir(parents=True)
    beh.mkdir(parents=True)
    (eeg / "sub-01_task-Choice_events.tsv").write_text(
        "onset\ttrial_type\tresponse_time\tconfidence\n"
        "0\tstimulus\t1.2\t0.8\n"
        "2\tfeedback\t1.0\t0.7\n",
        encoding="utf-8",
    )
    (eeg / "sub-01_task-Choice_events.json").write_text(
        json.dumps(
            {
                "confidence": {"Description": "external confidence"},
                "trial_type": {"Levels": {"stimulus": "stimulus"}},
            }
        ),
        encoding="utf-8",
    )
    (beh / "sub-01_task-Choice_beh.tsv").write_text(
        "trial_index\tstimulus_probability\tresponse\n"
        "1\t0.8\tleft\n",
        encoding="utf-8",
    )
    (beh / "sub-01_task-Choice_physio.json").write_text(
        json.dumps({"Columns": ["ECG", "RSP", "Digital"]}),
        encoding="utf-8",
    )
    (root / "sub-01" / "code").mkdir()
    (root / "sub-01" / "code" / "task.m").write_text(
        "reward_probability = 0.8;\n",
        encoding="utf-8",
    )


def test_variable_classifier_does_not_call_response_physiology() -> None:
    assert classify_variable("response")[0] == "discrete_label"
    assert classify_variable("response_time")[0] == "discrete_label"
    assert classify_variable("trial_type")[0] == "discrete_label"
    assert classify_variable("RSP")[0] == "physiology"
    assert classify_variable("forced_choice")[0] != "continuous_external"
    assert classify_variable("value")[0] == "unevaluated"
    assert classify_variable("Trial_Stake_Jitter")[0] == "unevaluated"
    assert classify_variable("stimulus_reward_probability")[0] == (
        "ordinal_task_param"
    )
    assert classify_variable("partialScore")[0] == "ordinal_task_param"


def test_scan_is_source_driven_and_participants_are_header_only(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    spec = {
        "dataset_id": "fixture",
        "config_path": "missing.yaml",
        "configured_roots": (str(tmp_path),),
        "fallback_roots": (),
        "source_role": "raw_bids",
        "candidate_hints": (),
        "candidate_rationale": "fixture",
    }
    result = scan_dataset(spec)
    assert result["source_status"] == "AVAILABLE"
    assert result["adjudication"]["candidate_present"] is True
    assert result["adjudication"]["transition_proximity_status"] == (
        "UNVERIFIED_CANDIDATE"
    )
    names = {
        variable["name"]
        for variable in result["variables"]
    }
    assert "confidence" in names
    assert "stimulus_probability" in names
    participant_payloads = [
        payload
        for payload in result["sample_files"]
        if payload["format"] == "tsv"
        and payload["relative_path"] == "participants.tsv"
    ]
    assert participant_payloads[0]["sample_values"] == {}


def test_missing_root_is_not_tested_as_no_external_rc() -> None:
    spec = {
        "dataset_id": "missing",
        "config_path": "missing.yaml",
        "configured_roots": ("Z:/does-not-exist/scout-2",),
        "fallback_roots": (),
        "source_role": "raw_bids",
        "candidate_hints": (),
        "candidate_rationale": "fixture",
    }
    result = scan_dataset(spec)
    assert result["source_status"] == "SOURCE_UNAVAILABLE"
    assert result["adjudication"]["candidate_present"] is None
    assert result["adjudication"]["transition_proximity_status"] == (
        "SOURCE_UNAVAILABLE"
    )


def test_participant_headers_do_not_create_task_candidates(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_text("fixture", encoding="utf-8")
    (tmp_path / "participants.tsv").write_text(
        "participant_id\tBDI_score\tconfidence\nsub-01\t10\t0.8\n",
        encoding="utf-8",
    )
    spec = {
        "dataset_id": "participants-only",
        "config_path": "missing.yaml",
        "configured_roots": (str(tmp_path),),
        "fallback_roots": (),
        "source_role": "raw_bids",
        "candidate_hints": (),
        "candidate_rationale": "fixture",
    }
    result = scan_dataset(spec)
    assert result["source_status"] == "AVAILABLE"
    assert result["adjudication"]["candidate_present"] is False
    assert result["adjudication"]["transition_proximity_status"] == (
        "NO_EXTERNAL_RC"
    )


def test_unreadable_event_metadata_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_fixture(tmp_path)
    import mndm.dynamical_families.emp_scout_2 as scout

    original = scout._read_tsv

    def fake_read(path: Path, *, header_only: bool = False):
        if path.name.endswith("_events.tsv"):
            return {
                "path": str(path),
                "format": "tsv",
                "read_status": "error:fixture",
                "columns": [],
                "sample_values": {},
            }
        return original(path, header_only=header_only)

    monkeypatch.setattr(scout, "_read_tsv", fake_read)
    spec = {
        "dataset_id": "unreadable",
        "config_path": "missing.yaml",
        "configured_roots": (str(tmp_path),),
        "fallback_roots": (),
        "source_role": "raw_bids",
        "candidate_hints": (),
        "candidate_rationale": "fixture",
    }
    result = scan_dataset(spec)
    assert result["source_status"] == "SOURCE_READ_ERROR"
    assert result["adjudication"]["candidate_present"] is None
    assert result["adjudication"]["decision"] == "AUDIT_INCOMPLETE"


def test_forbidden_binary_suffix_is_never_opened(tmp_path: Path) -> None:
    (tmp_path / "README.h5").write_text(
        "must not be opened",
        encoding="utf-8",
    )
    spec = {
        "dataset_id": "binary-only",
        "config_path": "missing.yaml",
        "configured_roots": (str(tmp_path),),
        "fallback_roots": (),
        "source_role": "raw_bids",
        "candidate_hints": (),
        "candidate_rationale": "fixture",
    }
    result = scan_dataset(spec)
    assert result["source_status"] == "SOURCE_METADATA_EMPTY"
    assert result["sample_files"] == []


def test_source_walk_error_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mndm.dynamical_families.emp_scout_2 as scout

    monkeypatch.setattr(
        scout,
        "_select_samples",
        lambda root: (_ for _ in ()).throw(OSError("walk failed")),
    )
    spec = {
        "dataset_id": "walk-error",
        "config_path": "missing.yaml",
        "configured_roots": (str(tmp_path),),
        "fallback_roots": (),
        "source_role": "raw_bids",
        "candidate_hints": (),
        "candidate_rationale": "fixture",
    }
    result = scan_dataset(spec)
    assert result["source_status"] == "SOURCE_READ_ERROR"
    assert result["adjudication"]["candidate_present"] is None
    assert result["adjudication"]["decision"] == "AUDIT_INCOMPLETE"


def test_empty_hints_still_scan_candidate_classes(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    _write_fixture(root)
    variables = [
        {
            "name": "confidence",
            "classification": "continuous_external",
        }
    ]
    result = _adjudicate(
        {
            "candidate_hints": (),
            "candidate_rationale": "fixture",
        },
        variables,
    )
    assert result["candidate_present"] is True


def test_hint_cannot_promote_excluded_jitter_field() -> None:
    result = _adjudicate(
        {
            "candidate_hints": ("trial_stake",),
            "candidate_rationale": "fixture",
        },
        [
            {
                "name": "Trial_Stake_Jitter",
                "classification": "unevaluated",
            }
        ],
    )
    assert result["candidate_present"] is False
    assert result["transition_proximity_status"] == "NO_EXTERNAL_RC"
