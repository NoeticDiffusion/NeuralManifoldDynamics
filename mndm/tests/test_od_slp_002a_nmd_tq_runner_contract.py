"""Runner-level fail-closed tests for OD-SLP-002A-NMD-TQ."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402

import od_slp_002a_nmd_tq_gate as gate  # noqa: E402


def _computed_record(
    *,
    rmse: float = 0.1,
    mae: float = 0.08,
    e_max: float = 0.2,
) -> dict[str, object]:
    return {
        "computation_status": "computed",
        "metrics": {
            "truth_valid": True,
            "rmse_q": rmse,
            "mae_q": mae,
            "e_max": e_max,
            "endpoint_abs_error": 0.0,
            "q_range_min": 0.0,
            "q_range_max": 1.0,
            "min_q_difference": 0.0,
        },
        "appended_absorbing_boundary_rows": 0,
    }


def test_dev_thresholds_are_frozen_from_dev_records_only() -> None:
    thresholds = gate._thresholds(
        [_computed_record(rmse=0.10, mae=0.08, e_max=0.20)]
    )
    assert thresholds == {
        "rmse_q": pytest.approx(0.125),
        "mae_q": pytest.approx(0.10),
        "e_max": pytest.approx(0.25),
    }


def test_confirmation_not_testable_is_not_a_fail() -> None:
    status, reasons = gate._confirmation_pass(
        {
            "computation_status": "not_testable",
            "failure_reason": "dense_query_grid_has_under_supported_points",
            "metrics": {"truth_status": "not_scored", "truth_valid": False},
        },
        {"rmse_q": 0.2, "mae_q": 0.2, "e_max": 0.3},
    )
    assert status == "NOT_TESTABLE"
    assert reasons == ["dense_query_grid_has_under_supported_points"]


def test_confirmation_metric_breach_is_fail() -> None:
    status, reasons = gate._confirmation_pass(
        _computed_record(rmse=0.4),
        {"rmse_q": 0.2, "mae_q": 0.2, "e_max": 0.3},
    )
    assert status == "FAIL"
    assert reasons == ["rmse_q_threshold_exceeded"]


def test_compaction_keeps_diagnostics_and_drops_q_arrays() -> None:
    compact = gate._compact_candidate(
        {
            "grid_resolution": 9,
            "candidate_status": "NOT_TESTABLE",
            "dev_records": [
                {
                    "computation_status": "not_testable",
                    "failure_reason": "support",
                    "q_grid": [0.0, 1.0],
                }
            ],
            "dev_summary": {"n_records": 1, "n_not_testable": 1},
        }
    )
    serialized = json.dumps(compact)
    assert "q_grid" not in serialized
    assert compact["dev_summary"]["n_not_testable"] == 1


def test_runner_refuses_overwrite_and_writes_q_free_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gate,
        "_candidate_audit",
        lambda grid: {
            "grid_resolution": grid,
            "candidate_status": "PASS",
            "reason": None,
            "dev_thresholds": {"rmse_q": 0.1, "mae_q": 0.1, "e_max": 0.2},
            "dev_summary": {"n_records": 1},
            "confirmation_summary": {"n_records": 1},
        },
    )
    output = tmp_path / "qualification.json"
    result = gate.run_gate(output)
    assert result["decision"]["overall_status"] == "PASS"
    serialized = output.read_text(encoding="utf-8")
    assert "q_grid" not in serialized
    with pytest.raises(FileExistsError):
        gate.run_gate(output)


def test_overall_pass_uses_only_pass_eligible_grids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statuses = {65: "NOT_TESTABLE", 33: "FAIL", 17: "PASS", 9: "FAIL"}
    monkeypatch.setattr(
        gate,
        "_candidate_audit",
        lambda grid: {
            "grid_resolution": grid,
            "candidate_status": statuses[grid],
            "reason": None,
            "dev_thresholds": None,
        },
    )
    result = gate.run_gate(tmp_path / "mixed.json")
    assert result["decision"]["overall_status"] == "PASS"
    assert result["decision"]["eligible_grids"] == [17]
