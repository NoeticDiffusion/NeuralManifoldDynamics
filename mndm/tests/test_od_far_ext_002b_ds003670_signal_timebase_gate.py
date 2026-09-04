from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import od_far_ext_002b_ds003670_signal_timebase_gate as gate  # noqa: E402
from mndm.dynamical_families import far_ext_002b_ds003670_signal_timebase as far_ext  # noqa: E402
from test_far_ext_002b_ds003670_signal_timebase import _write_bound_fixture  # noqa: E402


def test_gate_writes_certificate_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_root, eeg_root, repo_root, far002_path = _write_bound_fixture(tmp_path)
    monkeypatch.setattr(far_ext, "PROMOTED_EVENT_COUNT", 4)
    prereg = tmp_path / "prereg.md"
    prereg.write_text("FAR-EXT-002B fixture prereg\n", encoding="utf-8")
    monkeypatch.setattr(gate, "PREREG_PATH", prereg)
    output_json = tmp_path / "result.json"
    output_report = tmp_path / "result.md"
    result = gate.run_gate(
        repo_root=repo_root,
        metadata_root=metadata_root,
        eeg_root=eeg_root,
        far002_path=far002_path,
        output_json=output_json,
        output_report=output_report,
        audit_existing_download=True,
    )
    assert result["protocol_id"] == "FAR-EXT-002B"
    assert result["global_status"] == "SIGNAL_TIMEBASE_PASS"
    assert result["protocol_sha256"]
    assert result["payload_download"] == {
        "performed": True,
        "mode": "audited_previously_downloaded_root",
        "payload_root": str(eeg_root),
    }
    assert output_json.is_file()
    assert output_report.is_file()
    with pytest.raises(
        FileExistsError,
        match="refusing_to_overwrite_far_ext_002b_archive",
    ):
        gate.run_gate(
            repo_root=repo_root,
            metadata_root=metadata_root,
            eeg_root=eeg_root,
            far002_path=far002_path,
            output_json=output_json,
            output_report=tmp_path / "new-report.md",
        )


def test_gate_rejects_non_pass_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_root, eeg_root, repo_root, far002_path = _write_bound_fixture(tmp_path)
    monkeypatch.setattr(far_ext, "PROMOTED_EVENT_COUNT", 4)
    payload = json.loads(far002_path.read_text(encoding="utf-8"))
    payload["global_status"] = "POINT_SEMANTICS_ONLY"
    far002_path.write_text(json.dumps(payload), encoding="utf-8")
    prereg = tmp_path / "prereg.md"
    prereg.write_text("fixture\n", encoding="utf-8")
    monkeypatch.setattr(gate, "PREREG_PATH", prereg)
    with pytest.raises(ValueError, match="global_status_not_pass"):
        gate.run_gate(
            repo_root=repo_root,
            metadata_root=metadata_root,
            eeg_root=eeg_root,
            far002_path=far002_path,
            output_json=tmp_path / "result.json",
            output_report=tmp_path / "result.md",
        )
