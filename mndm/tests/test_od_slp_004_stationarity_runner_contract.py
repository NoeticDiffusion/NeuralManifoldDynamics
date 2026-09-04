"""Fail-closed runner tests for the OD-SLP-004 DEV audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import od_slp_004_pooled_law_stationarity_gate as gate  # noqa: E402
from mndm.dynamical_families.sleep_committor_dev_manifest import (  # noqa: E402
    CANONICAL_SOURCE_ROOT,
    FROZEN_PID_SPLIT,
)


def test_noncanonical_root_fails_before_source_access(tmp_path: Path) -> None:
    output = tmp_path / "dev_stationarity_audit.json"
    result = gate.run_audit(
        raw_root=tmp_path / "wrong-root",
        output_path=output,
    )
    assert result["combined_status"] == "NOT_TESTABLE"
    assert result["single_pooled_fit_count"] == 0
    assert result["fail_closed_assertions"]["held_out_signal_or_outcome_read"] is False
    assert result["fail_closed_assertions"]["reserve_signal_or_outcome_read"] is False
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["score_surfaces"]["early"]["status"] == "NOT_TESTABLE"
    assert "qualified" not in output.read_text(encoding="utf-8")


def test_runner_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "dev_stationarity_audit.json"
    output.write_text("sentinel\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        gate.run_audit(
            raw_root=tmp_path / "wrong-root",
            output_path=output,
        )
    assert output.read_text(encoding="utf-8") == "sentinel\n"


def _matching_prerequisites() -> dict[str, object]:
    return {
        "status": "MATCH",
        "protocol_sha256": "protocol",
        "records": {
            "od_slp_004_protocol": {"status": "MATCH", "actual_sha256": "a"},
            "od_slp_003_protocol": {"status": "MATCH", "actual_sha256": "b"},
            "od_slp_003_dev_manifest": {
                "status": "MATCH",
                "actual_sha256": "c",
            },
            "od_slp_001_protocol": {"status": "MATCH", "actual_sha256": "d"},
            "od_slp_002b_protocol": {"status": "MATCH", "actual_sha256": "e"},
            "od_slp_002a_nmd_tq_result": {
                "status": "MATCH",
                "actual_sha256": "f",
            },
            "od_slp_002b_boas_dev_audit": {
                "status": "MATCH",
                "actual_sha256": "g",
            },
        },
    }


def test_prerequisite_mismatch_skips_source_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "dev_stationarity_audit.json"
    calls: list[str] = []
    monkeypatch.setattr(
        gate,
        "_verify_prerequisites",
        lambda root: {
            "status": "MISMATCH",
            "protocol_sha256": None,
            "records": {},
        },
    )
    monkeypatch.setattr(
        gate,
        "_load_pid_map",
        lambda root: calls.append("participants"),
    )
    result = gate.run_audit(
        raw_root=Path(CANONICAL_SOURCE_ROOT),
        output_path=output,
    )
    assert result["combined_status"] == "NOT_TESTABLE"
    assert calls == []
    assert "prerequisite_hash_mismatch" in result["failure_reasons"]


def test_runner_calls_only_pooled_fit_and_writes_score_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "dev_stationarity_audit.json"
    all_pids = [
        pid
        for arm in ("DEV", "HELD_OUT", "RESERVE")
        for pid in FROZEN_PID_SPLIT[arm]
    ]
    participants = {f"sub-{pid}": pid for pid in all_pids}
    pairs = [
        {"participant_id": f"sub-{FROZEN_PID_SPLIT['DEV'][0]}"}
    ]
    archived = json.loads(
        (
            gate._repo_root()
            / gate.OD003_MANIFEST_PATH
        ).read_text(encoding="utf-8")
    )
    archived_fit = archived["fits"]["pooled"]
    fit_calls: list[str] = []
    monkeypatch.setattr(gate, "_load_pid_map", lambda root: participants)
    monkeypatch.setattr(gate, "_pair_acquisitions", lambda root: pairs)
    monkeypatch.setattr(
        gate,
        "_load_night",
        lambda pair, **kwargs: {
            "participant_id": pair["participant_id"],
            "pid": participants[pair["participant_id"]],
            "split": "DEV",
            "failure_reasons": [],
            "segments": [],
        },
    )
    monkeypatch.setattr(
        gate,
        "_all_segments",
        lambda records, split: [{"pid": 15, "split": "DEV"}],
    )
    monkeypatch.setattr(
        gate,
        "_clean_json_record",
        lambda record: {"pid": record["pid"]},
    )
    monkeypatch.setattr(
        gate,
        "fit_dev_grid",
        lambda segments, stratum: (
            fit_calls.append(stratum)
            or {
                "status": "computed",
                "q_grid": archived_fit["q_grid"],
                "query_grid": archived_fit["query_grid"],
                "support_count": archived_fit["support_count"],
                "appended_absorbing_boundary_rows": 0,
            }
        ),
    )
    monkeypatch.setattr(
        gate,
        "dev_base_rate_brier",
        lambda segments, fit: {
            "status": "computed",
            "brier": archived["dev_base_rate_brier"]["brier"],
            "n_rows": 434,
        },
    )
    monkeypatch.setattr(
        gate,
        "score_fixed_law_surface",
        lambda segments, q_grid, query_grid, stratum: {
            "status": "computed",
            "stratum": stratum,
            "evaluation_support": {"status": "PASS"},
            "n_score_rows": 50,
            "metrics": {
                "I_s": 0.1,
                "R_s": 0.1,
                "L_s": 0.02,
                "absolute_brier_improvement": 0.1,
                "relative_brier_improvement": 0.1,
                "log_loss_improvement": 0.02,
            },
            "local_success": True,
        },
    )
    result = gate.run_audit(
        raw_root=Path(CANONICAL_SOURCE_ROOT),
        output_path=output,
    )
    assert fit_calls == ["pooled"]
    assert result["fit_request_log"] == ["pooled"]
    assert result["single_pooled_fit_count"] == 1
    assert result["evaluation_status"] == "PASS"
    assert result["stationarity_status"] == "PASS_STATIONARY"
    assert result["fail_closed_assertions"]["early_fit_attempted"] is False
    assert result["fail_closed_assertions"]["late_fit_attempted"] is False
    assert set(result["score_surfaces"]) == {"pooled", "early", "late"}


def test_mixed_arm_pairs_are_filtered_before_load_night(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "dev_stationarity_audit.json"
    dev_pid = FROZEN_PID_SPLIT["DEV"][0]
    held_out_pid = FROZEN_PID_SPLIT["HELD_OUT"][0]
    participants = {
        f"sub-{pid}": pid
        for arm in ("DEV", "HELD_OUT", "RESERVE")
        for pid in FROZEN_PID_SPLIT[arm]
    }
    pairs = [
        {"participant_id": f"sub-{dev_pid}"},
        {"participant_id": f"sub-{held_out_pid}"},
    ]
    loaded: list[str] = []
    monkeypatch.setattr(
        gate,
        "_verify_prerequisites",
        lambda root: _matching_prerequisites(),
    )
    monkeypatch.setattr(
        gate,
        "verify_frozen_geometry",
        lambda root: {"status": "MATCH", "failure_reasons": []},
    )
    monkeypatch.setattr(gate, "_load_pid_map", lambda root: participants)
    monkeypatch.setattr(gate, "_pair_acquisitions", lambda root: pairs)

    def fake_load(pair, **kwargs):
        loaded.append(pair["participant_id"])
        return {
            "participant_id": pair["participant_id"],
            "pid": participants[pair["participant_id"]],
            "split": "DEV",
            "failure_reasons": [],
            "segments": [],
        }

    monkeypatch.setattr(gate, "_load_night", fake_load)
    monkeypatch.setattr(gate, "_all_segments", lambda records, split: [])
    result = gate.run_audit(
        raw_root=Path(CANONICAL_SOURCE_ROOT),
        output_path=output,
    )
    assert loaded == [f"sub-{dev_pid}"]
    assert result["source_inventory"]["non_dev_pair_paths_not_loaded"] == 1
    assert result["fail_closed_assertions"]["non_dev_edf_loaded"] is False


def test_split_mismatch_skips_pair_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "dev_stationarity_audit.json"
    participants = {f"sub-{index}": index for index in range(100)}
    pair_calls: list[str] = []
    monkeypatch.setattr(
        gate,
        "_verify_prerequisites",
        lambda root: _matching_prerequisites(),
    )
    monkeypatch.setattr(
        gate,
        "verify_frozen_geometry",
        lambda root: {"status": "MATCH", "failure_reasons": []},
    )
    monkeypatch.setattr(gate, "_load_pid_map", lambda root: participants)
    monkeypatch.setattr(
        gate,
        "_pair_acquisitions",
        lambda root: pair_calls.append("pairs") or [],
    )
    result = gate.run_audit(
        raw_root=Path(CANONICAL_SOURCE_ROOT),
        output_path=output,
    )
    assert pair_calls == []
    assert result["combined_status"] == "NOT_TESTABLE"
    assert any(
        "frozen_pid_lists_mismatch" in reason
        for reason in result["failure_reasons"]
    )
