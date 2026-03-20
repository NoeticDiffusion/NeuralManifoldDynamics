from __future__ import annotations

import json
from pathlib import Path

import pytest

h5py = pytest.importorskip("h5py")

from mndm.pipeline.run_manifest import write_run_manifest


def _base_config() -> dict:
    return {
        "datasets": ["dsX"],
        "paths": {},
        "preprocess": {},
        "epoching": {},
        "features": {},
        "mnps_projection": {},
        "robustness": {},
        "mnps": {},
        "source": {"name": "OpenNeuro"},
    }


def _write_min_summary_json(path: Path) -> None:
    payload = {
        "subject": "sub-001",
        "task": "task",
        "condition": "cond",
        "group": "Healthy",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_run_manifest_uses_regional_mnps_as_canonical_regional_output(tmp_path: Path):
    mnps_dir = tmp_path / "mnps_dsX_20260101_000000"
    rec_dir = mnps_dir / "sub-001_cond_task_run-01"
    rec_dir.mkdir(parents=True)
    _write_min_summary_json(rec_dir / "summary.json")

    with h5py.File(rec_dir / "sub-001_cond_task_run-01.h5", "w") as h5:
        fr = h5.require_group("features_raw")
        fr.create_dataset("values", data=[[1.0], [2.0]])
        fr.create_dataset("names", data=[b"feat_a"])
        fz = h5.require_group("features_robust_z")
        fz.create_dataset("values", data=[[0.0], [1.0]])
        fz.create_dataset("names", data=[b"feat_a"])
        reg = h5.require_group("regional_mnps")
        dmn = reg.require_group("DMN")
        dmn.create_dataset("mnps", data=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    out_path = write_run_manifest(
        mnps_dir=mnps_dir,
        config=_base_config(),
        ds_id="dsX",
        received_dir=tmp_path / "received",
        processed_dir=tmp_path / "processed",
        h5_mode="subject",
    )

    manifest = json.loads(out_path.read_text(encoding="utf-8"))
    caps = manifest["capabilities"]
    assert manifest["schema"] == "mndm.run_manifest.v2"
    assert caps["regional_outputs"] is True
    assert caps["regional_outputs_path"] == "/regional_mnps"
    assert caps["raw_region_signals"] is False
    assert caps["raw_features"] is True
    assert caps["raw_features_path"] == "/features_raw"
    assert caps["robust_z_features"] is True
    assert caps["robust_z_features_path"] == "/features_robust_z"
    assert caps["counts"]["h5_with_regional_outputs"] == 1
    assert caps["counts"]["h5_with_raw_region_signals"] == 0
    assert caps["counts"]["h5_with_raw_features"] == 1
    assert caps["counts"]["h5_with_robust_z_features"] == 1


def test_run_manifest_tracks_raw_region_signals_separately_from_regional_outputs(tmp_path: Path):
    mnps_dir = tmp_path / "mnps_dsX_20260101_000001"
    rec_dir = mnps_dir / "sub-001_cond_task_run-01"
    rec_dir.mkdir(parents=True)
    _write_min_summary_json(rec_dir / "summary.json")

    with h5py.File(rec_dir / "sub-001_cond_task_run-01.h5", "w") as h5:
        fr = h5.require_group("features_raw")
        fr.create_dataset("values", data=[[1.0], [2.0]])
        fr.create_dataset("names", data=[b"feat_a"])
        fz = h5.require_group("features_robust_z")
        fz.create_dataset("values", data=[[0.0], [1.0]])
        fz.create_dataset("names", data=[b"feat_a"])
        regions = h5.require_group("regions")
        regions.create_dataset("bold", data=[[0.0, 1.0], [1.0, 0.0]])
        regions.create_dataset("names", data=[b"ROI_A", b"ROI_B"])
        regions.attrs["sfreq"] = 0.5

    out_path = write_run_manifest(
        mnps_dir=mnps_dir,
        config=_base_config(),
        ds_id="dsX",
        received_dir=tmp_path / "received",
        processed_dir=tmp_path / "processed",
        h5_mode="subject",
    )

    manifest = json.loads(out_path.read_text(encoding="utf-8"))
    caps = manifest["capabilities"]
    assert caps["regional_outputs"] is False
    assert caps["raw_region_signals"] is True
    assert caps["raw_features"] is True
    assert caps["robust_z_features"] is True
    assert caps["counts"]["h5_with_regional_outputs"] == 0
    assert caps["counts"]["h5_with_raw_region_signals"] == 1
    assert caps["counts"]["h5_with_raw_features"] == 1
    assert caps["counts"]["h5_with_robust_z_features"] == 1


def test_run_manifest_includes_reproducibility_block_and_merges_extra(tmp_path: Path):
    mnps_dir = tmp_path / "mnps_dsX_20260101_000002"
    rec_dir = mnps_dir / "sub-001_cond_task_run-01"
    rec_dir.mkdir(parents=True)
    _write_min_summary_json(rec_dir / "summary.json")

    config = _base_config()
    config["reproducibility"] = {"seed": 123}

    out_path = write_run_manifest(
        mnps_dir=mnps_dir,
        config=config,
        ds_id="dsX",
        received_dir=tmp_path / "received",
        processed_dir=tmp_path / "processed",
        h5_mode="subject",
        extra={"reproducibility": {"n_jobs": 4}},
    )

    manifest = json.loads(out_path.read_text(encoding="utf-8"))
    assert manifest["reproducibility"]["seed"] == 123
    assert manifest["reproducibility"]["seed_source"] == "reproducibility.seed"
    assert manifest["reproducibility"]["n_jobs"] == 4
