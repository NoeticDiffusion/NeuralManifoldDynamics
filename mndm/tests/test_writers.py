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
            assert "values" in f["features_raw"]
            assert "names" in f["features_raw"]
            assert "metadata" in f["features_raw"]
            assert "feature_name" in f["features_raw"]["metadata"]
            assert "labels" in f
            # Extensions group should be present when extensions are provided
            assert "extensions" in f
            assert "e_kappa" in f["extensions"]
            assert "time" in f["extensions"]["e_kappa"]
            assert "kappa" in f["extensions"]["e_kappa"]
            assert "participant" in f
            assert "row_json" in f["participant"]
            assert "mapped_json" in f["participant"]
            assert "source_json" in f["participant"]
            assert f["participant"].attrs["field_participant_id"] == "sub-001"
            assert f["participant"].attrs["mapped_group"] == "Control"
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

