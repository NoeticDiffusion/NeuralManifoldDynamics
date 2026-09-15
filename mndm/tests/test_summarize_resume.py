"""Tests for summarize --resume-run missing-H5 filtering."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.pipeline.summary_utils import h5_recording_is_complete, recording_h5_path
from mndm.pipeline.summary import DatasetSummaryRunner


@pytest.fixture
def dummy_ctx(tmp_path):
    return SimpleNamespace(
        config={"robustness": {"coverage": {}}},
        received_dir=tmp_path,
        processed_dir=tmp_path,
        coverage=SimpleNamespace(min_seconds=0.0, min_epochs=0),
        weights={"m": {}, "d": {}, "e": {}},
        normalize_override=None,
        ingest_meta={},
        reproducibility={"seed": 42, "seed_source": "default"},
        mnps_cfg={
            "window_sec": 4.0,
            "overlap": 0.25,
            "fs_out": 4.0,
            "derivative": {"method": "sav_gol", "window": 5, "polyorder": 2},
            "knn_k": 5,
            "knn_metric": "euclidean",
            "ridge_alpha": 1.0,
            "super_window": 3,
            "stage_codebook": {},
            "embodied": {"enabled": False},
            "surrogates": {},
            "reliability": {},
            "whiten": True,
        },
        extensions_cfg={},
        derivative_cfg={"method": "sav_gol", "window": 5, "polyorder": 2},
    )


def test_h5_recording_is_complete_rejects_missing_and_truncated(tmp_path):
    missing = tmp_path / "missing.h5"
    assert h5_recording_is_complete(missing) is False

    empty = tmp_path / "empty.h5"
    empty.write_bytes(b"")
    assert h5_recording_is_complete(empty) is False

    garbage = tmp_path / "garbage.h5"
    garbage.write_bytes(b"not hdf5" * 40)
    assert h5_recording_is_complete(garbage) is False


def test_h5_recording_is_complete_requires_mnps_3d(tmp_path):
    h5py = pytest.importorskip("h5py")
    import numpy as np

    complete = tmp_path / "ok.h5"
    with h5py.File(complete, "w") as handle:
        handle.create_dataset("mnps_3d", data=np.zeros((4, 3), dtype=np.float32))
    assert h5_recording_is_complete(complete) is True

    incomplete = tmp_path / "no_coords.h5"
    with h5py.File(incomplete, "w") as handle:
        handle.create_dataset("time", data=np.arange(4, dtype=np.float64))
    assert h5_recording_is_complete(incomplete) is False


def test_filter_resume_groupings_skips_complete_h5(dummy_ctx, tmp_path):
    h5py = pytest.importorskip("h5py")
    import numpy as np

    runner = DatasetSummaryRunner(dummy_ctx, "ds001", None, "subject")
    runner._session_meta_map = {}
    runner.resume_run_dir = tmp_path

    sub_id = "sub-001"
    dir_suffix = "rest_run-001"
    h5_path = recording_h5_path(tmp_path, sub_id, dir_suffix)
    h5_path.parent.mkdir(parents=True)
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("mnps_3d", data=np.zeros((3, 3), dtype=np.float32))

    complete_key = (sub_id, None, "rest", "run-001", None)
    missing_key = (sub_id, None, "rest", "run-002", None)
    frame = pd.DataFrame({"file": ["sub-001_task-rest_run-001_eeg.set"]})
    missing_frame = pd.DataFrame({"file": ["sub-001_task-rest_run-002_eeg.set"]})

    remaining = runner._filter_resume_groupings(
        [(complete_key, frame), (missing_key, missing_frame)],
        tmp_path,
    )
    assert len(remaining) == 1
    assert remaining[0][0] == missing_key
    report = json.loads((tmp_path / "resume_report.json").read_text(encoding="utf-8"))
    assert report["complete_skipped"] == 1
    assert report["remaining"] == 1
    assert report["remaining_groupings"][0]["grouping_key"][-2] == "run-002"


def test_create_output_dir_reuses_resume_run(dummy_ctx, tmp_path):
    runner = DatasetSummaryRunner(
        dummy_ctx, "ds001", None, "subject", resume_run_dir=tmp_path
    )
    assert runner._create_output_dir(tmp_path / "ds001") == tmp_path.resolve()
