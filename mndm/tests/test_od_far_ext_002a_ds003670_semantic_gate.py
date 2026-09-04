from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from test_far_ext_002a_ds003670_semantic_join import _write_bound_fixture


def _load_gate():
    path = Path(__file__).with_name("od_far_ext_002a_ds003670_semantic_gate.py")
    spec = importlib.util.spec_from_file_location("od_far_ext_002a_gate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("gate_module_spec_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_writes_semantic_certificate(tmp_path: Path):
    gate = _load_gate()
    source, entry_payload = _write_bound_fixture(tmp_path, files=("0104",))
    entry_payload["gate_status"] = "CENSUS_COMPLETE"
    entry = tmp_path / "far_ext_001.json"
    entry.write_text(json.dumps(entry_payload), encoding="utf-8")

    output = tmp_path / "far_ext_002a.json"
    report = tmp_path / "far_ext_002a.md"
    result = gate.run_gate(
        metadata_root=source,
        entry_path=entry,
        output_path=output,
        report_path=report,
    )

    assert result["protocol_id"] == "FAR-EXT-002A"
    assert result["entry_criteria"]["status"] == "CENSUS_COMPLETE"
    assert result["audit_scope"]["signal_payloads_opened"] is False
    assert output.exists()
    assert report.exists()


def test_gate_refuses_overwrite(tmp_path: Path):
    gate = _load_gate()
    source, entry_payload = _write_bound_fixture(tmp_path, files=("0104",))
    entry_payload["gate_status"] = "CENSUS_COMPLETE"
    entry = tmp_path / "far_ext_001.json"
    entry.write_text(json.dumps(entry_payload), encoding="utf-8")
    output = tmp_path / "far_ext_002a.json"
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing_to_overwrite"):
        gate.run_gate(
            metadata_root=source,
            entry_path=entry,
            output_path=output,
            report_path=tmp_path / "far_ext_002a.md",
        )


def test_gate_rejects_non_complete_entry(tmp_path: Path):
    gate = _load_gate()
    source, entry_payload = _write_bound_fixture(tmp_path, files=("0104",))
    entry_payload["gate_status"] = "NOT_TESTABLE"
    entry = tmp_path / "far_ext_001.json"
    entry.write_text(json.dumps(entry_payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="entry_status_not_accepted"):
        gate.run_gate(
            metadata_root=source,
            entry_path=entry,
            output_path=tmp_path / "far_ext_002a.json",
            report_path=tmp_path / "far_ext_002a.md",
        )
