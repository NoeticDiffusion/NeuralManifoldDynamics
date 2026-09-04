"""Synthetic contract tests for the FAR-001 source scout."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.dynamical_families.far_source_scout import (  # noqa: E402
    DIRECTION_ONLY,
    SOURCE_UNCERTAIN,
    _scan_ds006036,
    run_inventory,
)
from od_far_001_source_scout_gate import run_gate  # noqa: E402


def _write_ds006036_fixture(root: Path) -> None:
    (root / "sub-001" / "eeg").mkdir(parents=True)
    (root / "README").write_text(
        "The protocol uses incremental frequencies from 5 Hz to 10 Hz.\n",
        encoding="utf-8",
    )
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "photic fixture"}),
        encoding="utf-8",
    )
    (root / "task-photomark_events.json").write_text(
        json.dumps(
            {
                "value": {
                    "Levels": {
                        "PHOTO 5Hz": "Presentation onset",
                        "PHOTO 10Hz": "Presentation onset",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "participants.tsv").write_text(
        "participant_id\tGroup\nsub-001\tC\n",
        encoding="utf-8",
    )
    (root / "sub-001" / "eeg" / "sub-001_task-photomark_events.tsv").write_text(
        "onset\tduration\tsample\tvalue\n"
        "2.0\t0\t1000\tPHOTO 5Hz\n"
        "12.0\t0\t6000\tPHOTO 10Hz\n",
        encoding="utf-8",
    )
    (root / "sub-001" / "eeg" / "sub-001_task-photomark_eeg.set").write_text(
        "signal placeholder; scout must not open payload\n",
        encoding="utf-8",
    )


def test_frequency_levels_are_direction_only_without_rho(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ds006036"
    _write_ds006036_fixture(root)

    result = _scan_ds006036(root, tmp_path / "config.yaml")

    assert result["source_status"] == "AVAILABLE"
    assert result["classification"] == DIRECTION_ONLY
    assert result["photo_frequency_levels_hz"] == [5.0, 10.0]
    assert result["intensity_or_luminance_fields"] == []
    assert result["source_semantics"]["rho_known"] is False
    assert result["source_semantics"]["multiple_rho_levels"] is False
    assert result["source_semantics"]["v_known"] is True
    assert result["source_semantics"]["offset_known"] is False
    assert result["source_semantics"]["continuous_neural_signal"] is True


def test_missing_roots_are_source_uncertain_not_negative_evidence(
    tmp_path: Path,
) -> None:
    protocol = tmp_path / "protocol.md"
    protocol.write_text("FAR-001 fixture protocol\n", encoding="utf-8")

    result = run_inventory(
        repo_root=tmp_path,
        source_roots={"ds006036": tmp_path / "missing-ds006036"},
        protocol_path=protocol,
    )

    assert result["gate_status"] == "NOT_TESTABLE"
    assert result["gate_reason"] == "no_genuine_amplitude_candidate"
    assert all(
        dataset["classification"] == SOURCE_UNCERTAIN
        for dataset in result["datasets"]
    )
    assert result["fail_closed_assertions"]["frequency_promoted_to_amplitude"] is False


def test_inventory_finds_no_curve_candidate_from_frequency_only_fixture(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ds006036"
    _write_ds006036_fixture(root)
    protocol = tmp_path / "protocol.md"
    protocol.write_text("FAR-001 fixture protocol\n", encoding="utf-8")

    result = run_inventory(
        repo_root=tmp_path,
        source_roots={"ds006036": root},
        protocol_path=protocol,
    )

    by_id = {dataset["dataset_id"]: dataset for dataset in result["datasets"]}
    assert result["gate_status"] == "NOT_TESTABLE"
    assert by_id["ds006036"]["classification"] == DIRECTION_ONLY
    assert result["classification_counts"][DIRECTION_ONLY] == 1
    assert result["fail_closed_assertions"]["resilience_curve_estimated"] is False


def test_gate_writes_read_only_inventory_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "ds006036"
    _write_ds006036_fixture(source_root)
    protocol = (
        tmp_path
        / "project/orthagonal_axis/orthagonal_dynamics/"
        "finite-amplitude_resilience/002_far_source_scout_prereg.md"
    )
    protocol.parent.mkdir(parents=True)
    protocol.write_text("FAR-001 fixture protocol\n", encoding="utf-8")
    far000_certificate = tmp_path / "far000.json"
    far000_certificate.write_text(
        json.dumps(
            {
                "status": "PASS",
                "decision": {"far_001_authorized": True},
            }
        ),
        encoding="utf-8",
    )
    output_json = tmp_path / "result.json"
    output_report = tmp_path / "result.md"

    result = run_gate(
        repo_root=tmp_path,
        output_json=output_json,
        output_report=output_report,
        source_roots={"ds006036": source_root},
        far000_certificate=far000_certificate,
    )

    assert result["gate_status"] == "NOT_TESTABLE"
    assert output_json.is_file()
    assert output_report.is_file()
    assert json.loads(output_json.read_text(encoding="utf-8"))["gate_status"] == (
        "NOT_TESTABLE"
    )
    with pytest.raises(FileExistsError, match="refusing_to_overwrite"):
        run_gate(
            repo_root=tmp_path,
            output_json=output_json,
            output_report=tmp_path / "new-report.md",
            source_roots={"ds006036": source_root},
            far000_certificate=far000_certificate,
        )
