from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import od_far_ext_002e_ds003670_post_ramp_interpretation_gate as gate  # noqa: E402
from mndm.dynamical_families import (  # noqa: E402
    far_ext_002e_ds003670_post_ramp_interpretation as far_ext,
)


def _scope() -> dict[str, object]:
    return {
        "protocol_id": "FAR-EXT-002E",
        "execution_status": "PREREGISTRATION_FROZEN",
        "source_documents": [
            dict(document) for document in far_ext.SOURCE_DOCUMENTS
        ],
        "inherited_002d": {
            "result_path": "far_ext_002d.json",
            "result_sha256": "result-hash",
            "f30_subset_sha256": "subset-hash",
            "f30_subset_count": 1,
            "f30_direct_subset_sha256": "direct-subset-hash",
            "f30_direct_subset_count": 1,
        },
    }


def _write_prereg(path: Path, scope_hash: str) -> None:
    path.write_text(
        "\n".join(
            [
                "# FAR-EXT-002E",
                "",
                "scope manifest SHA256:",
                f"  {scope_hash}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _fake_audit(**kwargs: object) -> dict[str, object]:
    del kwargs
    return {
        "protocol_id": "FAR-EXT-002E",
        "dataset_id": "ds003670",
        "global_status": "POST_RAMP_INTERPRETABILITY_UNRESOLVED",
        "global_reason": "fixture",
        "source_documents": [],
        "source_semantics": {},
        "audit_scope": {
            "signal_payloads_opened": False,
            "outcome_tables_opened": False,
            "nmd_outputs_opened": False,
            "mnps_calculated": False,
            "far_calculated": False,
        },
    }


def test_gate_binds_scope_writes_report_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope_path = tmp_path / "scope.json"
    scope_path.write_text(json.dumps(_scope()), encoding="utf-8")
    prereg_path = tmp_path / "prereg.md"
    _write_prereg(prereg_path, gate.sha256_file(scope_path))
    result_path = tmp_path / "far_ext_002d.json"
    result_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(gate, "audit_far_ext_002e", _fake_audit)
    output_json = tmp_path / "result.json"
    output_report = tmp_path / "result.md"
    payload = gate.run_gate(
        repo_root=tmp_path,
        result_002d=result_path,
        scope_path=scope_path,
        prereg_path=prereg_path,
        output_json=output_json,
        output_report=output_report,
        fetch_documents=False,
    )
    assert payload["scope_execution_status"] == "PREREGISTRATION_FROZEN"
    assert output_json.is_file()
    assert output_report.is_file()
    with pytest.raises(
        FileExistsError,
        match="refusing_to_overwrite_far_ext_002e_archive",
    ):
        gate.run_gate(
            repo_root=tmp_path,
            result_002d=result_path,
            scope_path=scope_path,
            prereg_path=prereg_path,
            output_json=output_json,
            output_report=tmp_path / "new-report.md",
            fetch_documents=False,
        )


def test_gate_rejects_scope_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope_path = tmp_path / "scope.json"
    scope_path.write_text(json.dumps(_scope()), encoding="utf-8")
    prereg_path = tmp_path / "prereg.md"
    _write_prereg(prereg_path, "0" * 64)
    monkeypatch.setattr(gate, "audit_far_ext_002e", _fake_audit)
    with pytest.raises(ValueError, match="far_ext_002e_scope_hash_mismatch"):
        gate.run_gate(
            repo_root=tmp_path,
            result_002d=tmp_path / "far_ext_002d.json",
            scope_path=scope_path,
            prereg_path=prereg_path,
            output_json=tmp_path / "result.json",
            output_report=tmp_path / "result.md",
            fetch_documents=False,
        )
