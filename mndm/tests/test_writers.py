"""Tests for HDF5 and JSON writers."""

from pathlib import Path
import sys
import tempfile

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def test_write_json_summary():
    """Test write json summary."""
    from core.io.json_writer import build_manifest, write_json_summary
    from mndm.schema import MNPSPayload

    payload = MNPSPayload(
        time=np.arange(4, dtype=np.float64),
        x=np.zeros((4, 3), dtype=np.float32),
        x_dot=np.zeros((4, 3), dtype=np.float32),
        stage=None,
        z=None,
        events={},
        nn_indices=None,
        jacobian=None,
        jacobian_dot=None,
        jacobian_centers=None,
        attrs={"fs_out": 4.0, "window_sec": 8.0, "overlap": 0.5},
    )

    manifest = build_manifest("test", payload)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test.json"
        result = write_json_summary(manifest, out_path)

        assert result == out_path
        assert out_path.exists()

        import json
        with out_path.open() as f:
            loaded = json.load(f)
        assert loaded["dataset_id"] == "test"
        assert loaded["meta_indices"]["windows"] == 0


def test_write_json_summary_sanitizes_nonfinite_and_paths():
    """Test write json summary sanitizes nonfinite and paths."""
    from core.io.json_writer import write_json_summary
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "sanitized.json"
        summary = {
            "a": float("nan"),
            "b": float("inf"),
            "c": Path(tmpdir),
            "d": b"ok-bytes",
            "e": [1.0, float("-inf")],
        }
        write_json_summary(summary, out_path)
        with out_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["a"] is None
        assert loaded["b"] is None
        assert loaded["e"][1] is None
        assert isinstance(loaded["c"], str)
        assert loaded["d"] == "ok-bytes"


def test_write_h5(require_real_h5py):
    """Test write h5."""
    from core.io.h5_writer import write_h5
    from mndm.schema import MNPSPayload

    payload = MNPSPayload(
        time=np.linspace(0, 1, 5, dtype=np.float64),
        x=np.random.rand(5, 3).astype(np.float32),
        x_dot=np.random.rand(5, 3).astype(np.float32),
        stage=np.arange(5, dtype=np.int8),
        z=None,
        events={"so": np.array([1, 3], dtype=np.int64)},
        nn_indices=np.zeros((5, 2), dtype=np.int32),
        jacobian=np.random.rand(3, 3, 3).astype(np.float32),
        jacobian_dot=np.random.rand(3, 3, 3).astype(np.float32),
        jacobian_centers=np.array([1, 2, 3], dtype=np.int32),
        features_raw_values=np.random.rand(5, 2).astype(np.float32),
        features_raw_names=["eeg_alpha", "eeg_alpha__g_frontal"],
        features_robust_z_values=np.random.rand(5, 2).astype(np.float32),
        features_robust_z_names=["eeg_alpha", "eeg_alpha__g_frontal"],
        feature_metadata={
            "feature_name": np.array(["eeg_alpha", "eeg_alpha"], dtype=object),
            "group_label": np.array(["", "frontal"], dtype=object),
            "used_by_mnps_3d": np.array([1, 0], dtype=np.int8),
        },
        coordinate_layers={
            "coords_3d_subject_anchored": {
                "values": np.random.rand(5, 3).astype(np.float32),
                "names": ["m", "d", "e"],
                "attrs": {"coordinate_contract": "subject_anchored"},
            },
            "coords_3d_cohort_anchored": {
                "values": np.random.rand(5, 3).astype(np.float32),
                "names": ["m", "d", "e"],
                "attrs": {"coordinate_contract": "cohort_anchored", "anchor_id": "unit-test"},
            },
        },
        event_windows={
            "event_id": np.array([0, 0, 1], dtype=np.int32),
            "window_id": np.array([0, 1, 3], dtype=np.int32),
            "bin_label": np.array(["event", "post_near", "event"], dtype=object),
            "event_label": np.array(["Standard Tone", "Standard Tone", "Target Tone"], dtype=object),
            "window_contains_event_onset": np.array([1, 0, 1], dtype=np.int8),
        },
        event_windows_attrs={
            "_schema_version": "mndm.event_windows.v1",
            "reference": "onset",
        },
        codebooks={
            "stage": {
                "codes": np.array([10, 11], dtype=np.int32),
                "labels": ["Eyes Closed: Every 1000 ms", "Eyes Open: Every 1000 ms"],
                "label_keys": ["eyes_closed", "eyes_open"],
                "attrs": {"source": "consensus"},
            }
        },
        feature_anchors={
            "spec": {
                "anchor_id": "unit-test",
                "anchor_hash": "hash",
                "schema_version": "mndm.feature_anchors.v2.1",
            },
            "features": [
                {"feature_name": "eeg_alpha", "center": 0.0, "scale": 1.0, "n_subjects": 3},
            ],
        },
        jacobian_layers={
            "jacobian_subject_anchored": {
                "J_hat": np.random.rand(3, 3, 3).astype(np.float32),
                "J_dot": np.random.rand(3, 3, 3).astype(np.float32),
                "centers": np.array([1, 2, 3], dtype=np.int32),
                "attrs": {"coordinate_contract": "subject_anchored"},
            },
            "jacobian_cohort_anchored": {
                "J_hat": np.random.rand(3, 3, 3).astype(np.float32),
                "J_dot": np.random.rand(3, 3, 3).astype(np.float32),
                "centers": np.array([1, 2, 3], dtype=np.int32),
                "attrs": {"coordinate_contract": "cohort_anchored", "anchor_id": "unit-test"},
            },
        },
        attrs={
            "fs_out": 4.0,
            "window_sec": 8.0,
            "overlap": 0.5,
            "reproducibility_seed": 123,
            "reproducibility_seed_source": "reproducibility.seed",
            "jacobian_hash_saved": "abc123",
            "stage_codebook": {"W": 0},
            "participant_meta": {
                "participant_id": "sub-001",
                "type": "Control",
                "age": 21,
            },
            "participant_meta_source": {
                "source_path": "H:/data/dsX/participants.csv",
                "source_format": "csv",
                "subject_id_column": "participant_id",
            },
            "participant_mapped_meta": {
                "group": "Control",
                "condition": "rest",
                "task": "rest",
            },
        },
        extensions={
            "e_kappa": {
                "time": np.linspace(0, 1, 5, dtype=np.float32),
                "energy": np.linspace(0, 1, 5, dtype=np.float32),
                "kappa": np.zeros(5, dtype=np.float32),
            },
            "time_reference": {
                "run": {
                    "status": "ok",
                    "schema_version": "time_reference.v1",
                },
                "windows": {
                    "window_start_from_run_sec": np.linspace(0, 8, 5, dtype=np.float32),
                    "window_end_from_run_sec": np.linspace(2, 10, 5, dtype=np.float32),
                    "window_bin_id": np.array([0, 0, 1, 1, 1], dtype=np.int32),
                },
            },
        },
        participant_clinical_meta={
            "age": 21,
            "sex": "F",
            "medication_status": "OFF",
            "session_order": 1,
        },
        provenance={
            "contract": {
                "export_contract_version": "mndm.eeg_h5_contract.v1",
                "run_manifest_ref": "../run_manifest.json",
            },
            "anchoring": {
                "available_coordinate_contracts": ["subject_anchored", "cohort_anchored"],
            },
        },
        coverage={
            "axis_fraction": np.ones((5, 3), dtype=np.float32),
            "axis_names": np.array(["m", "d", "e"], dtype=object),
            "coordinate_layers_present": np.array(
                ["coords_3d_subject_anchored", "coords_3d_cohort_anchored"],
                dtype=object,
            ),
        },
        qc_windows={
            "retained_after_qc": np.ones(5, dtype=np.int8),
            "coverage_ok": np.array([1, 1, 1, 1, 1], dtype=np.int8),
            "stage_transition_flag": np.array([0, 1, 0, 0, 0], dtype=np.int8),
        },
        regional_mnps={
            "DMN": {
                "mnps": np.random.rand(5, 3).astype(np.float32),
                "mnps_dot": np.random.rand(5, 3).astype(np.float32),
                "jacobian": np.random.rand(3, 3, 3).astype(np.float32),
                "stratified": np.random.rand(5, 9).astype(np.float32),
                "metrics": {"m_mean": 0.1},
                "n_timepoints": 5,
                "anchor_layers": {
                    "subject_anchored": {
                        "mnps": np.random.rand(5, 3).astype(np.float32),
                        "mnps_dot": np.random.rand(5, 3).astype(np.float32),
                        "jacobian": np.random.rand(3, 3, 3).astype(np.float32),
                        "stratified": np.random.rand(5, 9).astype(np.float32),
                        "metrics": {"m_mean": 0.2},
                        "n_timepoints": 5,
                        "attrs": {"coordinate_contract": "subject_anchored"},
                    },
                    "cohort_anchored": {
                        "mnps": np.random.rand(5, 3).astype(np.float32),
                        "mnps_dot": np.random.rand(5, 3).astype(np.float32),
                        "jacobian": np.random.rand(3, 3, 3).astype(np.float32),
                        "stratified": np.random.rand(5, 9).astype(np.float32),
                        "metrics": {"m_mean": 0.3},
                        "n_timepoints": 5,
                        "attrs": {"coordinate_contract": "cohort_anchored", "anchor_id": "unit-test"},
                    },
                },
                "primary_coordinate_contract": "cohort_anchored",
            }
        },
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test.h5"
        result = write_h5(out_path, "test", payload, manifest={"note": "demo"})

        assert result == out_path
        assert out_path.exists()

        import h5py
        with h5py.File(out_path, "r") as f:
            assert "dataset_id" in f.attrs
            assert "time" in f
            assert "mnps_3d" in f
            assert "jacobian" in f
            assert "features_raw" in f
            assert "features_robust_z" in f
            assert "feature_anchors" in f
            assert "spec" in f["feature_anchors"]
            assert "per_feature" in f["feature_anchors"]
            assert f["feature_anchors"]["spec"].attrs["anchor_id"] == "unit-test"
            assert "coords_3d_subject_anchored" in f
            assert "coords_3d_cohort_anchored" in f
            assert "values" in f["coords_3d_cohort_anchored"]
            assert f["coords_3d_cohort_anchored"].attrs["coordinate_contract"] == "cohort_anchored"
            assert "values" in f["features_raw"]
            assert "names" in f["features_raw"]
            assert "metadata" in f["features_raw"]
            assert "feature_name" in f["features_raw"]["metadata"]
            assert "labels" in f
            assert "codebooks" in f
            assert "stage" in f["codebooks"]
            assert "codes" in f["codebooks"]["stage"]
            assert "labels" in f["codebooks"]["stage"]
            assert "event_windows" in f
            assert "window_id" in f["event_windows"]
            assert "event_label" in f["event_windows"]
            assert f["event_windows"].attrs["reference"] == "onset"
            assert "coverage" in f
            assert "axis_fraction" in f["coverage"]
            assert "provenance" in f
            assert "contract" in f["provenance"]
            assert "jacobian_subject_anchored" in f
            assert "J_hat" in f["jacobian_subject_anchored"]
            assert "jacobian_cohort_anchored" in f
            assert f["jacobian_cohort_anchored"].attrs["coordinate_contract"] == "cohort_anchored"
            assert "qc" in f
            assert "windows" in f["qc"]
            assert "coverage_ok" in f["qc"]["windows"]
            # Extensions group should be present when extensions are provided
            assert "extensions" in f
            assert "e_kappa" in f["extensions"]
            assert "time" in f["extensions"]["e_kappa"]
            assert "kappa" in f["extensions"]["e_kappa"]
            assert "time_reference" in f["extensions"]
            assert "run" in f["extensions"]["time_reference"]
            assert "windows" in f["extensions"]["time_reference"]
            assert "window_bin_id" in f["extensions"]["time_reference"]["windows"]
            assert "participant" in f
            assert "row_json" in f["participant"]
            assert "mapped_json" in f["participant"]
            assert "source_json" in f["participant"]
            assert "clinical_json" in f["participant"]
            assert "regional_mnps" in f
            assert "DMN" in f["regional_mnps"]
            assert "subject_anchored" in f["regional_mnps"]["DMN"]
            assert "cohort_anchored" in f["regional_mnps"]["DMN"]
            assert "mnps" in f["regional_mnps"]["DMN"]["subject_anchored"]
            assert f["regional_mnps"]["DMN"].attrs["primary_coordinate_contract"] == "cohort_anchored"
            assert f["participant"].attrs["field_participant_id"] == "sub-001"
            assert f["participant"].attrs["mapped_group"] == "Control"
            assert f["participant"].attrs["clinical_medication_status"] == "OFF"
            assert f.attrs["meta_type"] == "Control"
            assert f.attrs["group"] == "Control"
            assert f.attrs["condition"] == "rest"
            assert f.attrs["task"] == "rest"
            assert int(f.attrs["reproducibility_seed"]) == 123
            assert f.attrs["reproducibility_seed_source"] == "reproducibility.seed"
            assert f.attrs["jacobian_hash_saved"] == "abc123"


def test_write_h5_writes_jacobian_diagnostics_group(require_real_h5py):
    """Test write h5 writes jacobian diagnostics group."""
    from core.io.h5_writer import write_h5
    from mndm.schema import MNPSPayload

    payload = MNPSPayload(
        time=np.linspace(0, 1, 5, dtype=np.float64),
        x=np.random.rand(5, 3).astype(np.float32),
        x_dot=np.random.rand(5, 3).astype(np.float32),
        stage=None,
        z=None,
        events={},
        nn_indices=None,
        jacobian=np.random.rand(3, 3, 3).astype(np.float32),
        jacobian_dot=np.random.rand(3, 3, 3).astype(np.float32),
        jacobian_centers=np.array([1, 2, 3], dtype=np.int32),
        attrs={"fs_out": 4.0, "window_sec": 8.0, "overlap": 0.5},
    )
    jac_diag = {
        "windows": 3.0,
        "rel_mse_baseline_median": 0.92,
        "rel_mse_baseline_windows": np.array([0.8, 0.9, 1.1], dtype=np.float32),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "diag.h5"
        write_h5(out_path, "test", payload, manifest={"note": "diag"}, jacobian_diagnostics=jac_diag)

        import h5py
        with h5py.File(out_path, "r") as f:
            assert "jacobian" in f
            assert "diagnostics" in f["jacobian"]
            dgrp = f["jacobian"]["diagnostics"]
            assert "rel_mse_baseline_windows" in dgrp
            assert np.allclose(dgrp["rel_mse_baseline_windows"][()], np.array([0.8, 0.9, 1.1], dtype=np.float32))
            assert float(dgrp.attrs["rel_mse_baseline_median"]) == pytest.approx(0.92)


def test_write_h5_supports_unicode_label_arrays(require_real_h5py):
    """Test write h5 supports unicode label arrays."""
    from core.io.h5_writer import write_h5
    from mndm.schema import MNPSPayload

    payload = MNPSPayload(
        time=np.linspace(0, 1, 3, dtype=np.float64),
        x=np.zeros((3, 3), dtype=np.float32),
        x_dot=np.zeros((3, 3), dtype=np.float32),
        stage=None,
        z=None,
        events={},
        labels={"condition": np.array(["CC", "GG", "Rest"], dtype="<U4")},
        nn_indices=None,
        jacobian=None,
        jacobian_dot=None,
        jacobian_centers=None,
        attrs={"fs_out": 4.0, "window_sec": 8.0, "overlap": 0.5},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "unicode_labels.h5"
        write_h5(out_path, "test", payload, manifest={"note": "unicode"})

        import h5py

        with h5py.File(out_path, "r") as f:
            assert "labels" in f
            assert "condition" in f["labels"]
            raw = f["labels"]["condition"][()]
            decoded = [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in raw]
            assert decoded == ["CC", "GG", "Rest"]

