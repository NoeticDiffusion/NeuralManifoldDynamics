from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import od_far_ext_002c_ds003670_native_rail_methods_gate as gate  # noqa: E402


def test_gate_writes_certificate_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_root = tmp_path / "metadata"
    eeg_root = tmp_path / "eeg"
    metadata_root.mkdir()
    eeg_root.mkdir()
    far002b_path = tmp_path / "far_ext_002b.json"
    far002b_path.write_text(
        json.dumps(
            {
                "protocol_id": "FAR-EXT-002B",
                "dataset_id": "ds003670",
                "global_status": "NOT_TESTABLE",
            }
        ),
        encoding="utf-8",
    )
    prereg = tmp_path / "prereg.md"
    prereg.write_text("fixture\n", encoding="utf-8")
    monkeypatch.setattr(gate, "PREREG_PATH", prereg)
    output_json = tmp_path / "result.json"
    output_report = tmp_path / "result.md"
    result = gate.run_gate(
        metadata_root=metadata_root,
        eeg_root=eeg_root,
        far002b_path=far002b_path,
        output_json=output_json,
        output_report=output_report,
        fetch_documents=False,
        probe_remote_header=False,
    )
    assert result["protocol_id"] == "FAR-EXT-002C"
    assert result["global_status"] == "AUDIT_SCOPE_FAILED"
    assert result["protocol_sha256"]
    assert output_json.is_file()
    assert output_report.is_file()
    with pytest.raises(
        FileExistsError,
        match="refusing_to_overwrite_far_ext_002c_archive",
    ):
        gate.run_gate(
            metadata_root=metadata_root,
            eeg_root=eeg_root,
            far002b_path=far002b_path,
            output_json=output_json,
            output_report=tmp_path / "new-report.md",
        )


def test_gate_rejects_wrong_entry_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prereg = tmp_path / "prereg.md"
    prereg.write_text("fixture\n", encoding="utf-8")
    monkeypatch.setattr(gate, "PREREG_PATH", prereg)
    far002b_path = tmp_path / "far_ext_002b.json"
    far002b_path.write_text(
        json.dumps(
            {
                "protocol_id": "FAR-EXT-002B",
                "dataset_id": "ds003670",
                "global_status": "SIGNAL_TIMEBASE_PASS",
            }
        ),
        encoding="utf-8",
    )
    result = gate.run_gate(
        metadata_root=tmp_path / "metadata",
        eeg_root=tmp_path / "eeg",
        far002b_path=far002b_path,
        output_json=tmp_path / "result.json",
        output_report=tmp_path / "result.md",
        fetch_documents=False,
        probe_remote_header=False,
    )
    assert result["global_status"] == "AUDIT_SCOPE_FAILED"
    assert result["global_reason"] == "far_ext_002b_entry_binding_failed"


def test_cli_skip_flags_bind_to_runner_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, object] = {}

    def fake_run_gate(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {
            "protocol_id": "FAR-EXT-002C",
            "global_status": "NOT_TESTABLE",
            "global_reason": "fixture",
            "native_float_rail_status": "UNRESOLVED",
            "reconciliation_authorized": False,
        }

    monkeypatch.setattr(gate, "run_gate", fake_run_gate)
    assert (
        gate.main(
            [
                "--skip-documents",
                "--skip-zenodo-header",
                "--output-json",
                str(tmp_path / "result.json"),
                "--output-report",
                str(tmp_path / "result.md"),
            ]
        )
        == 0
    )
    assert seen["fetch_documents"] is False
    assert seen["probe_remote_header"] is False
    assert "FAR-EXT-002C" in capsys.readouterr().out
