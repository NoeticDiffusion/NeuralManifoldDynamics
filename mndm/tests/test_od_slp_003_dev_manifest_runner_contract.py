"""Fail-closed runner tests for the OD-SLP-003 DEV manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import od_slp_003_dev_manifest_gate as gate  # noqa: E402
from mndm.dynamical_families.sleep_committor_dev_manifest import (  # noqa: E402
    CANONICAL_SOURCE_ROOT,
    DEV_STRATA,
    FROZEN_PID_SPLIT,
    GRID_RESOLUTION,
    P_REM_DEV,
)


def test_noncanonical_root_fails_before_source_access(tmp_path: Path) -> None:
    output = tmp_path / "dev_manifest.json"
    result = gate.run_manifest(
        raw_root=tmp_path / "not-canonical",
        output_path=output,
    )
    assert result["dev_adapter_status"] == "NOT_TESTABLE"
    assert result["source_root_canonical"] is False
    assert result["fail_closed_assertions"]["held_out_signal_or_outcome_read"] is False
    assert result["fail_closed_assertions"]["reserve_signal_or_outcome_read"] is False
    assert result["fail_closed_assertions"]["non_dev_edf_loaded"] is False
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["held_out_eligibility_inventory"] is None
    assert payload["held_out_scoring_deferred"] is True
    assert "qualification" not in output.read_text(encoding="utf-8")
    assert "qualified" not in output.read_text(encoding="utf-8")
    assert not list(tmp_path.glob("*.h5"))


def test_runner_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "dev_manifest.json"
    output.write_text("sentinel\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        gate.run_manifest(
            raw_root=tmp_path / "not-canonical",
            output_path=output,
        )
    assert output.read_text(encoding="utf-8") == "sentinel\n"


def test_prerequisite_mismatch_skips_source_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "dev_manifest.json"
    calls: list[str] = []
    monkeypatch.setattr(
        gate,
        "_load_pid_map",
        lambda root: calls.append("participants") or {},
    )
    monkeypatch.setattr(
        gate,
        "verify_prerequisite_hashes",
        lambda root: {
            "status": "MISMATCH",
            "protocol_sha256": "sentinel",
            "records": {},
        },
    )
    result = gate.run_manifest(
        raw_root=Path(CANONICAL_SOURCE_ROOT),
        output_path=output,
    )
    assert result["dev_adapter_status"] == "NOT_TESTABLE"
    assert calls == []
    assert "prerequisite_hash_mismatch" in result["failure_reasons"]


def test_manifest_serializes_three_q_grids_without_heldout_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "dev_manifest.json"
    all_pids = [
        pid
        for arm in ("DEV", "HELD_OUT", "RESERVE")
        for pid in FROZEN_PID_SPLIT[arm]
    ]
    participants = {
        f"sub-{pid}": pid
        for pid in all_pids
    }
    pairs = [
        {"participant_id": f"sub-{pid}"}
        for pid in FROZEN_PID_SPLIT["DEV"]
    ]

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
        lambda record: {"pid": record["pid"], "split": record["split"]},
    )
    monkeypatch.setattr(
        gate,
        "verify_frozen_geometry",
        lambda root: {"status": "MATCH", "failure_reasons": []},
    )
    monkeypatch.setattr(
        gate,
        "endpoint_census",
        lambda segments, dev_pids: {
            "status": "ENDPOINT_PASS",
            "n3_count": 305,
            "rem_count": 206,
        },
    )
    monkeypatch.setattr(
        gate,
        "fit_dev_grid",
        lambda segments, stratum: {
            "status": "computed",
            "stratum": stratum,
            "q_grid": np.linspace(0.1, 0.9, GRID_RESOLUTION),
            "query_grid": np.linspace(0.0, 1.0, GRID_RESOLUTION),
            "support_count": np.full(GRID_RESOLUTION, 64),
            "n_rows": 100,
            "n_segments": 50,
            "n_transition_segments": 20,
            "appended_absorbing_boundary_rows": 0,
        },
    )
    monkeypatch.setattr(
        gate,
        "dev_base_rate_brier",
        lambda segments, pooled_fit: {
            "status": "computed",
            "brier": 0.25,
            "n_rows": 100,
        },
    )
    result = gate.run_manifest(
        raw_root=Path(CANONICAL_SOURCE_ROOT),
        output_path=output,
    )
    assert result["dev_adapter_status"] == "PASS"
    assert result["transfer_tolerance"] == 0.0625
    assert set(result["fits"]) == set(DEV_STRATA)
    serialized = output.read_text(encoding="utf-8")
    assert '"q_grid"' in serialized
    assert '"held_out_scoring_deferred": true' in serialized
    assert '"qualified"' not in serialized
    assert P_REM_DEV == result["frozen_base_rate"]["p_REM_DEV"]


def test_mixed_arm_pairs_are_filtered_before_load_night(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "dev_manifest.json"
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
    monkeypatch.setattr(gate, "_load_pid_map", lambda root: participants)
    monkeypatch.setattr(gate, "_pair_acquisitions", lambda root: pairs)
    monkeypatch.setattr(
        gate,
        "verify_frozen_geometry",
        lambda root: {"status": "MATCH", "failure_reasons": []},
    )

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
    monkeypatch.setattr(
        gate,
        "_clean_json_record",
        lambda record: {"pid": record["pid"]},
    )
    result = gate.run_manifest(
        raw_root=Path(CANONICAL_SOURCE_ROOT),
        output_path=output,
    )
    assert result["dev_adapter_status"] == "NOT_TESTABLE"
    assert loaded == [f"sub-{dev_pid}"]
    assert result["source_inventory"]["non_dev_pair_paths_not_loaded"] == 1
    assert result["fail_closed_assertions"]["non_dev_edf_loaded"] is False


def test_split_mismatch_skips_pair_discovery_and_edf_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "dev_manifest.json"
    participants = {f"sub-{index}": index for index in range(100)}
    pair_calls: list[str] = []
    monkeypatch.setattr(gate, "_load_pid_map", lambda root: participants)
    monkeypatch.setattr(
        gate,
        "_pair_acquisitions",
        lambda root: pair_calls.append("pairs") or [],
    )
    monkeypatch.setattr(
        gate,
        "verify_frozen_geometry",
        lambda root: {"status": "MATCH", "failure_reasons": []},
    )
    result = gate.run_manifest(
        raw_root=Path(CANONICAL_SOURCE_ROOT),
        output_path=output,
    )
    assert result["dev_adapter_status"] == "NOT_TESTABLE"
    assert pair_calls == []
    assert any(
        "frozen_pid_lists_mismatch" in reason
        for reason in result["failure_reasons"]
    )
