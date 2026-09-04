"""Synthetic fail-closed tests for OD-EMP-DS004511-SEM-000."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.dynamical_families.ds004511_semantics import (  # noqa: E402
    REQUIRED_VARIABLES,
    audit_source,
)


BEHAVIOR_COLUMNS = [
    "index",
    "Trial_Type",
    "Participant_ID",
    "Trial_Probability",
    "Trial_Stake",
    "Trial_Stake_Jitter",
    "Trial_Type_Jitter",
    "Trial_Result_Jitter",
    "Balance_Update_Jitter",
    "Answer_Prediction_Agent",
    "Answer_Prediction_Agent_RT",
    "Answer_Prediction_Correct",
    "Answer_Prediction_Correct_RT",
    "Trial_Balance",
    "Trial_Balance_Effect",
    "Trial_Condition",
    "Trial_ProbableSide",
    "Prediction_Choice",
    "Prediction_Result",
    "Prediction_RT",
    "Trial_Order",
    "Trial_Datetime",
]


def _write_fixture(
    root: Path,
    *,
    include_source_code: bool = False,
    include_unknown_column: bool = False,
) -> Path:
    participant = root / "sub-S200116" / "ses-01"
    behavior_dir = participant / "beh"
    eeg_dir = participant / "eeg"
    behavior_dir.mkdir(parents=True)
    eeg_dir.mkdir(parents=True)
    (root / "README").write_text(
        "# Gambling Game\n144 rounds of dice prediction.\n",
        encoding="utf-8",
    )
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "Deception_data"}),
        encoding="utf-8",
    )
    behavior_path = (
        behavior_dir / "sub-S200116_ses-01_task-GG_run-01_beh.tsv"
    )
    rows = [
        [
            "1",
            "Self",
            "S200116",
            "0.5",
            "0.1",
            "1000",
            "1000",
            "1500",
            "1500",
            "RIGHT",
            "100",
            "LEFT",
            "200",
            "10.0",
            "0.1",
            "Probable_Truth",
            "RIGHT",
            "RIGHT",
            "2",
            "300",
            "0",
            "2020-01-01 00:00:00",
        ],
        [
            "2",
            "Computer",
            "S200116",
            "0.6666666667",
            "0.2",
            "1100",
            "1100",
            "1600",
            "1600",
            "LEFT",
            "100",
            "RIGHT",
            "200",
            "10.2",
            "0.2",
            "Probable_Lie",
            "LEFT",
            "LEFT",
            "1",
            "300",
            "1",
            "2020-01-01 00:00:20",
        ],
    ]
    source_columns = list(
        dict.fromkeys([*BEHAVIOR_COLUMNS, *REQUIRED_VARIABLES])
    )
    row_maps = [dict(zip(BEHAVIOR_COLUMNS, row)) for row in rows]
    if include_unknown_column:
        source_columns.append("confidence")
        row_maps[0]["confidence"] = "0.8"
        row_maps[1]["confidence"] = "0.7"
    lines = ["\t".join(source_columns)]
    lines.extend(
        "\t".join(row_map.get(column, "n/a") for column in source_columns)
        for row_map in row_maps
    )
    behavior_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (eeg_dir / "sub-S200116_ses-01_task-GG_run-01_events.tsv").write_text(
        "onset\tduration\ttrial_type\tvalue\tsample\n"
        "1.0\t0.0\tSync(1)\t1\t3000\n",
        encoding="utf-8",
    )
    (behavior_dir / "sub-S200116_ses-01_task-GG_run-01_physio.json").write_text(
        json.dumps(
            {
                "SamplingFrequency": 4000.0,
                "StartTime": 0,
                "Columns": [
                    "time",
                    "ECG A, X, ECG2-R",
                    "RSP A, X, RSP2-R",
                    "EDA, Y, PPGED-R",
                    "Digital input",
                ],
            }
        ),
        encoding="utf-8",
    )
    if include_source_code:
        code_dir = root / "code"
        code_dir.mkdir()
        (code_dir / "task.m").write_text("probability = 0.5;", encoding="utf-8")
    return behavior_path


def _protocol(tmp_path: Path) -> Path:
    protocol = tmp_path / "OD-EMP-DS004511-SEM-000.md"
    protocol.write_text("frozen protocol fixture", encoding="utf-8")
    return protocol


def test_trial_parameters_are_not_promoted_to_dynamic_task_rc(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    result = audit_source(
        source_root=tmp_path,
        protocol_path=_protocol(tmp_path),
    )
    assert result["gate_status"] == "NO_DYNAMIC_EXTERNAL_TASK_RC"
    assert result["candidate_dynamic_task_variables"] == []
    assert result["behavior_summary"]["n_behavior_rows"] == 2
    assert result["behavior_summary"]["trial_balance_changes_between_rows"] == 1
    assert result["behavior_summary"]["balance_effect_equals_stake"] == 2
    assert result["behavior_summary"]["balance_effect_comparisons"] == 2
    assert result["round_reconstruction"]["within_round_time_series_present"] is False
    assert result["source_metadata"]["source_files"]["events"]["sync_only"] is True
    assert result["runner_contract"]["committor_estimator_called"] is False
    assert result["runner_contract"]["outcome_association_tested"] is False
    assert result["runner_contract"]["a_b_selected"] is False


def test_peripheral_physio_is_dynamic_but_not_task_state(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    result = audit_source(
        source_root=tmp_path,
        protocol_path=_protocol(tmp_path),
    )
    physio = next(
        variable
        for variable in result["variables"]
        if variable["name"] == "ECG/RSP/EDA/EMG"
    )
    assert physio["classification"] == "CONTINUOUS_WITHIN_SEGMENT"
    assert physio["task_state_status"] == "NOT_A_TASK_STATE_RC"
    assert result["gate_status"] != "PASS_DYNAMIC_CANDIDATE"


def test_utf8_bom_event_header_remains_sync_only(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    events = (
        tmp_path
        / "sub-S200116"
        / "ses-01"
        / "eeg"
        / "sub-S200116_ses-01_task-GG_run-01_events.tsv"
    )
    text = events.read_text(encoding="utf-8")
    events.write_text("\ufeff" + text, encoding="utf-8")
    result = audit_source(
        source_root=tmp_path,
        protocol_path=_protocol(tmp_path),
    )
    assert result["source_metadata"]["source_files"]["events"]["sync_only"] is True


def test_local_source_code_presence_is_reported_but_does_not_change_gate(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path, include_source_code=True)
    result = audit_source(
        source_root=tmp_path,
        protocol_path=_protocol(tmp_path),
    )
    assert result["source_metadata"]["source_files"]["source_code_present"] is True
    assert result["gate_status"] == "NO_DYNAMIC_EXTERNAL_TASK_RC"
    assert "original GG experiment script absent locally" not in (
        result["unresolved_semantics"]
    )


def test_unlisted_candidate_column_blocks_negative_gate(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path, include_unknown_column=True)
    result = audit_source(
        source_root=tmp_path,
        protocol_path=_protocol(tmp_path),
    )
    confidence = next(
        variable
        for variable in result["variables"]
        if variable["name"] == "confidence"
    )
    assert confidence["classification"] == "UNKNOWN"
    assert result["gate_status"] == "SEMANTICS_UNRESOLVED"


def test_missing_trial_order_is_not_mistaken_for_within_round_dynamics(
    tmp_path: Path,
) -> None:
    behavior_path = _write_fixture(tmp_path)
    lines = behavior_path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    trial_order_column = header.index("Trial_Order")
    second_row = lines[2].split("\t")
    second_row[trial_order_column] = "n/a"
    lines[2] = "\t".join(second_row)
    behavior_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = audit_source(
        source_root=tmp_path,
        protocol_path=_protocol(tmp_path),
    )
    grain = result["behavior_summary"]["round_grain"]
    assert grain["all_files_one_row_per_trial"] is True
    assert grain["trial_order_complete"] is False
    assert result["round_reconstruction"]["within_round_time_series_present"] is (
        False
    )
    assert result["gate_status"] == "NO_DYNAMIC_EXTERNAL_TASK_RC"


def test_missing_required_source_file_is_audit_incomplete(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    events = (
        tmp_path
        / "sub-S200116"
        / "ses-01"
        / "eeg"
        / "sub-S200116_ses-01_task-GG_run-01_events.tsv"
    )
    events.unlink()
    result = audit_source(
        source_root=tmp_path,
        protocol_path=_protocol(tmp_path),
    )
    assert result["gate_status"] == "AUDIT_INCOMPLETE"


def test_incomplete_report_does_not_make_negative_rc_claim(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    events = (
        tmp_path
        / "sub-S200116"
        / "ses-01"
        / "eeg"
        / "sub-S200116_ses-01_task-GG_run-01_events.tsv"
    )
    events.unlink()
    result = audit_source(
        source_root=tmp_path,
        protocol_path=_protocol(tmp_path),
    )
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import od_emp_ds004511_semantics_gate as gate

    report = gate._render_report(result)
    assert "No negative RC finding is permitted" in report
    assert "No genuine within-round external task RC was identified." not in report


def test_missing_source_is_not_a_negative_semantic_finding(
    tmp_path: Path,
) -> None:
    result = audit_source(
        source_root=tmp_path / "missing",
        protocol_path=_protocol(tmp_path),
    )
    assert result["source_status"] == "SOURCE_UNAVAILABLE"
    assert result["gate_status"] == "SOURCE_UNAVAILABLE"
    assert result["candidate_dynamic_task_variables"] == []


def test_runner_refuses_overwrite(tmp_path: Path) -> None:
    _write_fixture(tmp_path / "source")
    output_json = tmp_path / "result.json"
    output_report = tmp_path / "result.md"
    output_json.write_text("sentinel", encoding="utf-8")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import od_emp_ds004511_semantics_gate as gate

    with pytest.raises(FileExistsError):
        gate.run_gate(
            repo_root=tmp_path,
            source_root=tmp_path / "source",
            output_json=output_json,
            output_report=output_report,
        )


def test_runner_writes_json_and_report_from_measured_payload(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    _write_fixture(source_root)
    protocol_path = (
        tmp_path
        / "project"
        / "orthagonal_axis"
        / "orthagonal_dynamics"
        / "OD-EMP-DS004511-SEM-000.md"
    )
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text("frozen protocol", encoding="utf-8")
    output_json = tmp_path / "result.json"
    output_report = tmp_path / "result.md"
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import od_emp_ds004511_semantics_gate as gate

    result = gate.run_gate(
        repo_root=tmp_path,
        source_root=source_root,
        output_json=output_json,
        output_report=output_report,
    )
    assert result["gate_status"] == "NO_DYNAMIC_EXTERNAL_TASK_RC"
    assert json.loads(output_json.read_text(encoding="utf-8"))["gate_status"] == (
        "NO_DYNAMIC_EXTERNAL_TASK_RC"
    )
    report = output_report.read_text(encoding="utf-8")
    assert "Trial_Datetime" in report
    assert "Behavior grain status: `True`" in report
