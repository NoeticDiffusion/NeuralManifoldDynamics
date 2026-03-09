from __future__ import annotations

import json
from pathlib import Path

import pytest

h5py = pytest.importorskip("h5py")

from mndm.pipeline.check_structure import check_run_dir


def _write_min_summary(path: Path, fmri: bool, regional_available: bool = False) -> None:
    payload = {
        "dataset_id": "dsX:sub-001:cond_task_run-01",
        "subject": "sub-001",
        "session": "ses-01",
        "run": "run-01",
        "condition": "cond",
        "task": "task",
        "extensions": {"regional_data_available": bool(regional_available)},
    }
    if fmri:
        payload["fmri_modularity_provisional_frac"] = 0.0
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_required_qc_files(rec_dir: Path) -> None:
    (rec_dir / "qc_summary.json").write_text("{}", encoding="utf-8")
    (rec_dir / "qc_reliability.json").write_text("{}", encoding="utf-8")


def test_check_structure_flags_missing_fmri_raw_regions(tmp_path: Path):
    run_dir = tmp_path / "mnps_dsX_20260101_000000"
    rec = run_dir / "sub-001_cond_task_run-01"
    rec.mkdir(parents=True)

    _write_min_summary(rec / "summary.json", fmri=True)
    _write_required_qc_files(rec)

    # H5 without /regions
    h5_path = rec / "sub-001_cond_task_run-01.h5"
    with h5py.File(h5_path, "w") as h:
        h.create_dataset("time", data=[0.0, 1.0])
        h.create_dataset("mnps_3d", data=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        h.create_dataset("mnps_3d_dot", data=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        h.require_group("jacobian")

    spec = {
        "common": {
            "required_files": ["summary.json", "qc_summary.json", "qc_reliability.json"],
            "require_single_h5": True,
        },
        "fmri": {
            "enabled": True,
            "h5_required_paths": [
                "/time",
                "/mnps_3d",
                "/mnps_3d_dot",
                "/jacobian",
                "/features_raw/values",
                "/features_raw/names",
                "/features_robust_z/values",
                "/features_robust_z/names",
            ],
            "require_regions": True,
            "regions": {"required_paths": ["/regions/bold", "/regions/names"], "required_attrs": {"/regions": ["sfreq"]}},
        },
        "eeg": {
            "enabled": True,
            "h5_required_paths": [
                "/time",
                "/mnps_3d",
                "/mnps_3d_dot",
                "/jacobian",
                "/features_raw/values",
                "/features_raw/names",
                "/features_robust_z/values",
                "/features_robust_z/names",
            ],
        },
    }

    report = check_run_dir("dsX", run_dir, spec)
    assert report.n_recordings == 1
    assert report.ok is False
    assert report.recordings[0].modality == "fmri"
    assert any(i.startswith("missing_h5_path:/regions") for i in report.recordings[0].issues)


def test_check_structure_passes_with_fmri_raw_regions(tmp_path: Path):
    run_dir = tmp_path / "mnps_dsX_20260101_000001"
    rec = run_dir / "sub-001_cond_task_run-01"
    rec.mkdir(parents=True)

    _write_min_summary(rec / "summary.json", fmri=True, regional_available=True)
    _write_required_qc_files(rec)

    h5_path = rec / "sub-001_cond_task_run-01.h5"
    with h5py.File(h5_path, "w") as h:
        h.create_dataset("time", data=[0.0, 1.0])
        h.create_dataset("mnps_3d", data=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        h.create_dataset("mnps_3d_dot", data=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        h.require_group("jacobian")
        fr = h.require_group("features_raw")
        fr.create_dataset("values", data=[[1.0, 2.0], [3.0, 4.0]])
        fr.create_dataset("names", data=[b"feat_a", b"feat_b"])
        fz = h.require_group("features_robust_z")
        fz.create_dataset("values", data=[[0.0, 1.0], [1.0, 0.0]])
        fz.create_dataset("names", data=[b"feat_a", b"feat_b"])
        rg = h.require_group("regions")
        rg.create_dataset("bold", data=[[0.0, 1.0], [1.0, 0.0]])
        rg.create_dataset("names", data=[b"VIS_ROI_A", b"DMN_ROI_B"])
        rg.attrs["sfreq"] = 0.5

    spec = {
        "common": {
            "required_files": ["summary.json", "qc_summary.json", "qc_reliability.json"],
            "require_single_h5": True,
        },
        "fmri": {
            "enabled": True,
            "h5_required_paths": [
                "/time",
                "/mnps_3d",
                "/mnps_3d_dot",
                "/jacobian",
                "/features_raw/values",
                "/features_raw/names",
                "/features_robust_z/values",
                "/features_robust_z/names",
            ],
            "require_regions": True,
            "regions": {"required_paths": ["/regions/bold", "/regions/names"], "required_attrs": {"/regions": ["sfreq"]}},
        },
        "eeg": {
            "enabled": True,
            "h5_required_paths": [
                "/time",
                "/mnps_3d",
                "/mnps_3d_dot",
                "/jacobian",
                "/features_raw/values",
                "/features_raw/names",
                "/features_robust_z/values",
                "/features_robust_z/names",
            ],
        },
    }

    report = check_run_dir("dsX", run_dir, spec)
    assert report.ok is True


def test_check_structure_flags_missing_regional_outputs_independent_of_modality(tmp_path: Path):
    run_dir = tmp_path / "mnps_dsX_20260101_000002"
    rec = run_dir / "sub-001_cond_task_run-01"
    rec.mkdir(parents=True)

    _write_min_summary(rec / "summary.json", fmri=False, regional_available=True)
    _write_required_qc_files(rec)

    h5_path = rec / "sub-001_cond_task_run-01.h5"
    with h5py.File(h5_path, "w") as h:
        h.create_dataset("time", data=[0.0, 1.0])
        h.create_dataset("mnps_3d", data=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        h.create_dataset("mnps_3d_dot", data=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        h.require_group("jacobian")
        fr = h.require_group("features_raw")
        fr.create_dataset("values", data=[[1.0], [2.0]])
        fr.create_dataset("names", data=[b"feat_a"])
        fz = h.require_group("features_robust_z")
        fz.create_dataset("values", data=[[0.0], [1.0]])
        fz.create_dataset("names", data=[b"feat_a"])

    spec = {
        "common": {
            "required_files": ["summary.json", "qc_summary.json", "qc_reliability.json"],
            "require_single_h5": True,
            "require_regional_outputs": True,
            "regional_outputs": {"required_paths": ["/regional_mnps"]},
        },
        "fmri": {"enabled": True, "require_regions": False},
        "eeg": {
            "enabled": True,
            "h5_required_paths": [
                "/time",
                "/mnps_3d",
                "/mnps_3d_dot",
                "/jacobian",
                "/features_raw/values",
                "/features_raw/names",
                "/features_robust_z/values",
                "/features_robust_z/names",
            ],
        },
    }

    report = check_run_dir("dsX", run_dir, spec)
    assert report.n_recordings == 1
    assert report.recordings[0].modality == "eeg"
    assert report.ok is False
    assert "missing_h5_path:/regional_mnps" in report.recordings[0].issues


def test_check_structure_passes_with_regional_outputs_independent_of_modality(tmp_path: Path):
    run_dir = tmp_path / "mnps_dsX_20260101_000003"
    rec = run_dir / "sub-001_cond_task_run-01"
    rec.mkdir(parents=True)

    _write_min_summary(rec / "summary.json", fmri=False, regional_available=True)
    _write_required_qc_files(rec)

    h5_path = rec / "sub-001_cond_task_run-01.h5"
    with h5py.File(h5_path, "w") as h:
        h.create_dataset("time", data=[0.0, 1.0])
        h.create_dataset("mnps_3d", data=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        h.create_dataset("mnps_3d_dot", data=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        h.require_group("jacobian")
        fr = h.require_group("features_raw")
        fr.create_dataset("values", data=[[1.0], [2.0]])
        fr.create_dataset("names", data=[b"feat_a"])
        fz = h.require_group("features_robust_z")
        fz.create_dataset("values", data=[[0.0], [1.0]])
        fz.create_dataset("names", data=[b"feat_a"])
        reg = h.require_group("regional_mnps")
        dmn = reg.require_group("DMN")
        dmn.create_dataset("mnps", data=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    spec = {
        "common": {
            "required_files": ["summary.json", "qc_summary.json", "qc_reliability.json"],
            "require_single_h5": True,
            "require_regional_outputs": True,
            "regional_outputs": {"required_paths": ["/regional_mnps"]},
        },
        "fmri": {"enabled": True, "require_regions": False},
        "eeg": {
            "enabled": True,
            "h5_required_paths": [
                "/time",
                "/mnps_3d",
                "/mnps_3d_dot",
                "/jacobian",
                "/features_raw/values",
                "/features_raw/names",
                "/features_robust_z/values",
                "/features_robust_z/names",
            ],
        },
    }

    report = check_run_dir("dsX", run_dir, spec)
    assert report.ok is True


