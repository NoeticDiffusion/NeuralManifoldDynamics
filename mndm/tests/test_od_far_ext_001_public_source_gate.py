from __future__ import annotations

import json
from pathlib import Path

import pytest

from od_far_ext_001_public_source_gate import run_gate


def _write_entries(root: Path) -> dict[str, Path]:
    statuses = {
        "far_scout_002": ("FAR-SCOUT-002", "NOT_TESTABLE"),
        "far_003a": ("FAR-003A", "METHOD_LIMITED"),
        "far_scout_003": ("FAR-SCOUT-003", "NOT_TESTABLE"),
    }
    root.mkdir(parents=True)
    paths = {}
    for name, (protocol_id, status) in statuses.items():
        path = root / f"{name}.json"
        path.write_text(
            json.dumps({"protocol_id": protocol_id, "gate_status": status}),
            encoding="utf-8",
        )
        paths[name] = path
    return paths


def _write_metadata_roots(root: Path) -> Path:
    for dataset_id in ("ds003670", "ds006519", "ds005169", "ds008037"):
        dataset = root / dataset_id
        eeg = dataset / "sub-01" / "eeg"
        eeg.mkdir(parents=True)
        (dataset / "dataset_description.json").write_text("{}", encoding="utf-8")
        (eeg / "sub-01_task-stim_events.tsv").write_text(
            "onset\tduration\tcurrent\ttrial_type\n"
            "0\t1\t0.5 mA\tstimulation\n"
            "20\t1\t1 mA\tstimulation\n",
            encoding="utf-8",
        )
    return root


def test_gate_writes_census_and_requires_all_entries(tmp_path: Path):
    metadata_root = _write_metadata_roots(tmp_path / "metadata")
    entries = _write_entries(tmp_path / "entries")
    output = tmp_path / "result.json"
    report = tmp_path / "result.md"

    payload = run_gate(
        metadata_root=metadata_root,
        output_path=output,
        report_path=report,
        entry_paths=entries,
    )

    assert payload["protocol_id"] == "FAR-EXT-001"
    assert payload["gate_status"] == "CENSUS_COMPLETE"
    assert payload["promotion_candidates"] == []
    assert payload["metadata_download"]["performed"] is False
    assert output.exists()
    assert report.exists()


def test_gate_refuses_overwrite(tmp_path: Path):
    metadata_root = _write_metadata_roots(tmp_path / "metadata")
    entries = _write_entries(tmp_path / "entries")
    output = tmp_path / "result.json"
    report = tmp_path / "result.md"
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing_to_overwrite"):
        run_gate(
            metadata_root=metadata_root,
            output_path=output,
            report_path=report,
            entry_paths=entries,
        )


def test_gate_rejects_wrong_entry_protocol(tmp_path: Path):
    metadata_root = _write_metadata_roots(tmp_path / "metadata")
    entries = _write_entries(tmp_path / "entries")
    entries["far_003a"].write_text(
        json.dumps({"protocol_id": "WRONG", "gate_status": "METHOD_LIMITED"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="entry_protocol_not_accepted"):
        run_gate(
            metadata_root=metadata_root,
            output_path=tmp_path / "result.json",
            report_path=tmp_path / "result.md",
            entry_paths=entries,
        )
