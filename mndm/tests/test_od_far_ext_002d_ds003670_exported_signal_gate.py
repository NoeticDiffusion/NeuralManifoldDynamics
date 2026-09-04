from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import od_far_ext_002d_ds003670_exported_signal_gate as gate  # noqa: E402


def _repo_far_root() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "project"
        / "orthagonal_axis"
        / "orthagonal_dynamics"
        / "finite-amplitude_resilience"
    )


def _fake_audit(**kwargs: object) -> dict[str, object]:
    del kwargs
    return {
        "protocol_id": "FAR-EXT-002D",
        "global_status": "NOT_TESTABLE",
        "global_reason": "fixture",
        "promoted_event_count": 0,
        "primary_family_id": None,
        "exported_signal_technical_status": {
            "selected_population_status": "TECHNICAL_STATUS_UNRESOLVED"
        },
        "clock_status": {"status": "CLOCK_UNRESOLVED"},
        "nmd_timebase_status": "NMD_TIMEBASE_METHOD_LIMITED",
        "post_stimulation_biological_interpretability": "NOT_ESTABLISHED",
        "audit_scope": {
            "signal_payloads_opened": False,
            "outcome_tables_opened": False,
            "nmd_outputs_opened": False,
            "mnps_calculated": False,
            "far_calculated": False,
        },
    }


def test_gate_binds_scope_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    far_root = _repo_far_root()
    scope = far_root / "FAR-EXT-002D-SCOPE-0.1.json"
    prereg = far_root / "012_far_ext_002d_ds003670_exported_signal_eligibility_prereg.md"
    certificate = far_root / "results" / "far_ext_002a" / "far_ext_002a.json"
    monkeypatch.setattr(gate, "audit_far_ext_002d", _fake_audit)
    output = tmp_path / "result.json"
    report = tmp_path / "result.md"
    payload = gate.run_gate(
        repo_root=Path(__file__).resolve().parents[2],
        metadata_root=tmp_path / "metadata",
        eeg_root=tmp_path / "eeg",
        far002a_path=certificate,
        scope_path=scope,
        prereg_path=prereg,
        output_json=output,
        output_report=report,
    )
    assert payload["scope_sha256"]
    assert payload["scope_execution_status"] == "PREREGISTRATION_FROZEN"
    assert output.is_file()
    assert report.is_file()
    with pytest.raises(
        FileExistsError,
        match="refusing_to_overwrite_far_ext_002d_archive",
    ):
        gate.run_gate(
            repo_root=Path(__file__).resolve().parents[2],
            metadata_root=tmp_path / "metadata",
            eeg_root=tmp_path / "eeg",
            far002a_path=certificate,
            scope_path=scope,
            prereg_path=prereg,
            output_json=output,
            output_report=tmp_path / "new-report.md",
        )


def test_gate_rejects_scope_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    far_root = _repo_far_root()
    bad_scope = tmp_path / "scope.json"
    bad_scope.write_text(json.dumps({"protocol_id": "FAR-EXT-002D"}), encoding="utf-8")
    monkeypatch.setattr(gate, "audit_far_ext_002d", _fake_audit)
    with pytest.raises(ValueError, match="scope_hash_mismatch"):
        gate.run_gate(
            repo_root=Path(__file__).resolve().parents[2],
            metadata_root=tmp_path / "metadata",
            eeg_root=tmp_path / "eeg",
            far002a_path=far_root
            / "results"
            / "far_ext_002a"
            / "far_ext_002a.json",
            scope_path=bad_scope,
            prereg_path=far_root
            / "012_far_ext_002d_ds003670_exported_signal_eligibility_prereg.md",
            output_json=tmp_path / "result.json",
            output_report=tmp_path / "result.md",
        )
