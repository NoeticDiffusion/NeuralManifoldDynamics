"""Tests for the MNDM prerequisite preflight checks."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def test_run_dataset_prerequisite_check_reports_ok_for_minimal_dataset(tmp_path):
    from mndm.prerequisite_check import run_dataset_prerequisite_check

    received_dir = tmp_path / "received"
    processed_dir = tmp_path / "processed"
    dataset_root = received_dir / "ds000001"
    eeg_dir = dataset_root / "sub-001" / "eeg"
    eeg_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    (dataset_root / "participants.tsv").write_text("participant_id\tgroup\nsub-001\tcontrol\n", encoding="utf-8")
    (eeg_dir / "sub-001_task-rest_eeg.edf").write_bytes(b"dummy")

    config = {
        "datasets": ["ds000001"],
        "paths": {
            "received_dir": str(received_dir),
            "processed_dir": str(processed_dir),
        },
        "epoching": {"length_s": 8.0},
        "mnps_projection": {"weights": {"m": {"eeg_alpha": 1.0}}},
        "mnps": {"enabled": True},
    }

    report = run_dataset_prerequisite_check(config=config, dataset_id="ds000001")

    assert report.ok is True
    check_map = {item.name: item for item in report.checks}
    assert check_map["dataset_root"].status == "ok"
    assert check_map["participants_table"].status == "ok"
    assert check_map["index_preview"].status == "ok"
    assert check_map["index_preview"].details["total_after_exclude"] == 1


def test_run_dataset_prerequisite_check_fails_for_invalid_exclude_regex(tmp_path):
    from mndm.prerequisite_check import run_dataset_prerequisite_check

    received_dir = tmp_path / "received"
    processed_dir = tmp_path / "processed"
    dataset_root = received_dir / "ds000002"
    dataset_root.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    config = {
        "datasets": ["ds000002"],
        "paths": {
            "received_dir": str(received_dir),
            "processed_dir": str(processed_dir),
        },
        "epoching": {"length_s": 8.0},
        "mnps_projection": {"weights": {"m": {"eeg_alpha": 1.0}}},
        "mnps": {"enabled": True},
        "exclude-files": ["("],
    }

    report = run_dataset_prerequisite_check(config=config, dataset_id="ds000002")

    assert report.ok is False
    check_map = {item.name: item for item in report.checks}
    assert check_map["exclude_files"].status == "fail"
