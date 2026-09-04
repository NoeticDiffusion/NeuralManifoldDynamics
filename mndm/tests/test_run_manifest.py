from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from mndm.pipeline.run_manifest import write_run_manifest


def _base_config() -> dict:
    """Internal helper: base config."""
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
    """Internal helper: write min summary json."""
    payload = {
        "subject": "sub-001",
        "task": "task",
        "condition": "cond",
        "group": "Healthy",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_run_manifest_uses_regional_mnps_as_canonical_regional_output(tmp_path: Path):
    """Test run manifest uses regional mnps as canonical regional output."""
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
    assert caps["time_reference"] is False
    assert caps["time_reference_path"] == "/extensions/time_reference"
    assert caps["counts"]["h5_with_regional_outputs"] == 1
    assert caps["counts"]["h5_with_raw_region_signals"] == 0
    assert caps["counts"]["h5_with_raw_features"] == 1
    assert caps["counts"]["h5_with_robust_z_features"] == 1
    assert caps["counts"]["h5_with_time_reference"] == 0


def test_run_manifest_detects_standard_dynamical_families(tmp_path: Path):
    """The run manifest must advertise the additive standard namespace."""
    mnps_dir = tmp_path / "mnps_dsX_20260101_000002"
    rec_dir = mnps_dir / "sub-001"
    rec_dir.mkdir(parents=True)
    with h5py.File(rec_dir / "sub-001.h5", "w") as h5:
        h5.require_group("dynamical_families")

    out_path = write_run_manifest(
        mnps_dir=mnps_dir,
        config=_base_config(),
        ds_id="dsX",
        received_dir=tmp_path / "received",
        processed_dir=tmp_path / "processed",
        h5_mode="subject",
    )
    capabilities = json.loads(out_path.read_text(encoding="utf-8"))["capabilities"]
    assert capabilities["dynamical_families"] is True
    assert capabilities["dynamical_families_path"] == "/dynamical_families"


def test_run_manifest_detects_legacy_orthogonal_dynamics_tree(tmp_path: Path):
    """Old HDF5 containers still count as family capability."""
    mnps_dir = tmp_path / "mnps_dsX_20260101_000003"
    rec_dir = mnps_dir / "sub-001"
    rec_dir.mkdir(parents=True)
    with h5py.File(rec_dir / "sub-001.h5", "w") as h5:
        h5.require_group("orthogonal_dynamics")

    out_path = write_run_manifest(
        mnps_dir=mnps_dir,
        config=_base_config(),
        ds_id="dsX",
        received_dir=tmp_path / "received",
        processed_dir=tmp_path / "processed",
        h5_mode="subject",
    )
    capabilities = json.loads(out_path.read_text(encoding="utf-8"))["capabilities"]
    assert capabilities["dynamical_families"] is True
    assert capabilities["dynamical_families_path"] == "/dynamical_families"


def test_run_manifest_tracks_raw_region_signals_separately_from_regional_outputs(tmp_path: Path):
    """Test run manifest tracks raw region signals separately from regional outputs."""
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
    assert caps["time_reference"] is False
    assert caps["counts"]["h5_with_regional_outputs"] == 0
    assert caps["counts"]["h5_with_raw_region_signals"] == 1
    assert caps["counts"]["h5_with_raw_features"] == 1
    assert caps["counts"]["h5_with_robust_z_features"] == 1
    assert caps["counts"]["h5_with_time_reference"] == 0


def test_run_manifest_includes_reproducibility_block_and_merges_extra(tmp_path: Path):
    """Test run manifest includes reproducibility block and merges extra."""
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


def test_run_manifest_detects_time_reference_capability(tmp_path: Path):
    """Detect `/extensions/time_reference/*` in capability probe + field guide."""
    mnps_dir = tmp_path / "mnps_dsX_20260101_000003"
    rec_dir = mnps_dir / "sub-001_cond_task_run-01"
    rec_dir.mkdir(parents=True)
    _write_min_summary_json(rec_dir / "summary.json")

    with h5py.File(rec_dir / "sub-001_cond_task_run-01.h5", "w") as h5:
        tr = h5.require_group("extensions").require_group("time_reference")
        run = tr.require_group("run")
        run.create_dataset("status", data=np.array("ok", dtype=h5py.string_dtype(encoding="utf-8")))
        windows = tr.require_group("windows")
        windows.create_dataset("window_start_from_anchor_sec", data=[0.0, 2.0, 4.0])

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
    assert caps["time_reference"] is True
    assert caps["time_reference_path"] == "/extensions/time_reference"
    assert caps["counts"]["h5_with_time_reference"] == 1
    h5_paths = manifest["field_guide"]["h5_paths"]
    assert "extensions/time_reference/run/*" in h5_paths
    assert "extensions/time_reference/windows/*" in h5_paths


def test_run_manifest_copies_yaml_and_records_filename(tmp_path: Path):
    """Copy active YAML into run dir and record copied filename."""
    mnps_dir = tmp_path / "mnps_dsX_20260101_000004"
    rec_dir = mnps_dir / "sub-001_cond_task_run-01"
    rec_dir.mkdir(parents=True)
    _write_min_summary_json(rec_dir / "summary.json")

    config_path = tmp_path / "config_ingest_dsX.yaml"
    config_path.write_text("datasets:\n  - dsX\n", encoding="utf-8")

    out_path = write_run_manifest(
        mnps_dir=mnps_dir,
        config=_base_config(),
        ds_id="dsX",
        received_dir=tmp_path / "received",
        processed_dir=tmp_path / "processed",
        h5_mode="subject",
        config_path=config_path,
    )

    manifest = json.loads(out_path.read_text(encoding="utf-8"))
    yaml_info = manifest["config"]["yaml_source"]
    assert yaml_info["copied_filename"] == config_path.name
    assert yaml_info["status"] in {"copied", "already_in_run_dir", "already_present"}
    copied_path = mnps_dir / config_path.name
    assert copied_path.exists()
    assert copied_path.read_text(encoding="utf-8") == config_path.read_text(encoding="utf-8")


def test_run_manifest_reports_requested_vs_realized_coordinate_contracts(tmp_path: Path):
    """Coordinate contracts should expose requested/realized/skipped breakdown."""
    mnps_dir = tmp_path / "mnps_dsX_20260101_000005"
    rec_dir = mnps_dir / "sub-001_cond_task_run-01"
    rec_dir.mkdir(parents=True)
    _write_min_summary_json(rec_dir / "summary.json")

    with h5py.File(rec_dir / "sub-001_cond_task_run-01.h5", "w") as h5:
        subject_layer = h5.require_group("coords_3d_subject_anchored")
        subject_layer.create_dataset("values", data=np.zeros((2, 3), dtype=np.float32))

    config = _base_config()
    config["mnps_projection"] = {
        "export_contracts": {
            "subject_anchored": True,
            "cohort_anchored": True,
        }
    }

    out_path = write_run_manifest(
        mnps_dir=mnps_dir,
        config=config,
        ds_id="dsX",
        received_dir=tmp_path / "received",
        processed_dir=tmp_path / "processed",
        h5_mode="subject",
    )

    manifest = json.loads(out_path.read_text(encoding="utf-8"))
    contracts = manifest["capabilities"]["coordinate_contracts"]
    assert contracts["requested_contracts"] == ["subject_anchored", "cohort_anchored"]
    assert contracts["realized_contracts"] == ["subject_anchored"]
    skipped = contracts["skipped_contracts_with_reason"]
    assert isinstance(skipped, list) and len(skipped) == 1
    assert skipped[0]["contract"] == "cohort_anchored"


def test_run_manifest_probes_anchor_capabilities(tmp_path: Path):
    """Run manifest should expose AnchorState capability flags separately from feature anchors."""
    mnps_dir = tmp_path / "mnps_dsX_20260101_000006"
    rec_dir = mnps_dir / "sub-001_cond_task_run-01"
    rec_dir.mkdir(parents=True)
    _write_min_summary_json(rec_dir / "summary.json")

    with h5py.File(rec_dir / "sub-001_cond_task_run-01.h5", "w") as h5:
        anchor_state = h5.require_group("anchor_state")
        anchor_state.create_dataset("values", data=np.zeros((2, 3), dtype=np.float32))
        anchor_quality = h5.require_group("anchor_quality")
        anchor_quality.create_dataset("values", data=np.zeros((2, 2), dtype=np.float32))
        anchor_coupling = h5.require_group("anchor_coupling")
        anchor_coupling.create_dataset("metrics", data=np.zeros((1, 4), dtype=np.float32))

    out_path = write_run_manifest(
        mnps_dir=mnps_dir,
        config=_base_config(),
        ds_id="dsX",
        received_dir=tmp_path / "received",
        processed_dir=tmp_path / "processed",
        h5_mode="subject",
    )

    manifest = json.loads(out_path.read_text(encoding="utf-8"))
    capabilities = manifest["capabilities"]
    assert capabilities["anchor_state"] is True
    assert capabilities["anchor_state_path"] == "/anchor_state"
    assert capabilities["anchor_quality"] is True
    assert capabilities["anchor_quality_path"] == "/anchor_quality"
    assert capabilities["anchor_coupling"] is True
    assert capabilities["anchor_coupling_path"] == "/anchor_coupling"


def test_run_manifest_excerpt_includes_meg_mapping_contract(tmp_path: Path):
    """Run manifest excerpt should preserve MEG shadow-mapping contract metadata."""
    mnps_dir = tmp_path / "mnps_ds003645_20260101_000007"
    rec_dir = mnps_dir / "sub-002_meeg_faces_run-01"
    rec_dir.mkdir(parents=True)
    _write_min_summary_json(rec_dir / "summary.json")

    config = _base_config()
    config["modality"] = "meg"
    config["meg_mapping"] = {
        "enabled": True,
        "mapping_family": "electrophysiology_shadow",
        "mapping_reference": "eeg_contract_v2",
        "sensor_types": ["mag", "grad"],
        "feature_combination": "robust_z_then_median",
    }

    out_path = write_run_manifest(
        mnps_dir=mnps_dir,
        config=config,
        ds_id="ds003645",
        received_dir=tmp_path / "received",
        processed_dir=tmp_path / "processed",
        h5_mode="subject",
    )

    manifest = json.loads(out_path.read_text(encoding="utf-8"))
    excerpt = manifest["config"]["excerpt"]
    assert excerpt["meg_mapping"]["mapping_family"] == "electrophysiology_shadow"
    assert excerpt["meg_mapping"]["mapping_reference"] == "eeg_contract_v2"
    assert excerpt["meg_mapping"]["sensor_types"] == ["mag", "grad"]
