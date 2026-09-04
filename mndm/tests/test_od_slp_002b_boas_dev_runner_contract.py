"""Runner-level fail-closed tests for OD-SLP-002B."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402

from od_slp_002b_boas_dev_gate import run_audit  # noqa: E402


def test_empty_source_root_is_not_testable_and_writes_q_free_json(
    tmp_path: Path,
) -> None:
    output = tmp_path / "boas_dev_audit.json"
    result = run_audit(
        raw_root=tmp_path / "missing-source",
        output_path=output,
    )
    assert result["combined_status"] == "NOT_TESTABLE"
    assert result["support_status"] == "SUPPORT_NOT_TESTABLE"
    assert result["endpoint_status"] == "ENDPOINT_NOT_TESTABLE"
    assert result["fail_closed_assertions"]["q_computed"] is False
    assert result["fail_closed_assertions"]["held_out_signal_or_outcome_read"] is False
    assert result["fail_closed_assertions"]["reserve_signal_or_outcome_read"] is False
    serialized = output.read_text(encoding="utf-8")
    assert "q_grid" not in serialized
    assert "hdf5" in serialized
    assert not list(tmp_path.glob("*.h5"))


def test_runner_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "boas_dev_audit.json"
    output.write_text("sentinel\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_audit(
            raw_root=tmp_path / "missing-source",
            output_path=output,
        )
    assert output.read_text(encoding="utf-8") == "sentinel\n"


def test_empty_runner_artifact_is_valid_json(tmp_path: Path) -> None:
    output = tmp_path / "boas_dev_audit.json"
    run_audit(raw_root=tmp_path / "missing-source", output_path=output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["candidate_tq_statuses"]["65"] == "PASS"
    assert payload["selection"]["selected_grid"] is None
    assert payload["endpoint_census"]["outcome_counts"] == {
        "competing_exit_artifact": 0,
        "competing_exit_disconnection": 0,
        "competing_exit_n1": 0,
        "competing_exit_wake": 0,
        "first_hit_n3": 0,
        "first_hit_rem": 0,
        "qc_or_gap_exit": 0,
        "right_censored": 0,
    }
