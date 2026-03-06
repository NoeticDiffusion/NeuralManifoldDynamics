"""Tests for DatasetSummaryRunner and SubjectSummaryRunner."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("mne")

from mndm.pipeline.summary import DatasetSummaryRunner, SubjectSummaryRunner
from mndm.pipeline.extensions_compute import compute_extensions


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
        mnps_cfg={
            "window_sec": 4.0,
            "overlap": 0.25,
            "fs_out": 4.0,
            "derivative": {"method": "sav_gol", "window": 5, "polyorder": 2},
            "knn_k": 5,
            "knn_metric": "euclidean",
            "super_window": 3,
            "stage_codebook": {},
            "embodied": {"enabled": False},
            "surrogates": {},
            "reliability": {},
            "whiten": True,
        },
        extensions_cfg={"tig": {"enabled": True, "max_lag_sec": 8.0, "n_lags": 4}},
        derivative_cfg={"method": "sav_gol", "window": 5, "polyorder": 2},
    )


def test_dataset_runner_subject_filter(dummy_ctx):
    runner = DatasetSummaryRunner(dummy_ctx, "ds001", "001", "subject")
    frame = pd.DataFrame(
        {
            "file": [
                "sub-001_ses-01_task-rest_eeg.set",
                "sub-002_ses-01_task-rest_eeg.set",
            ]
        }
    )

    filtered = runner._apply_subject_filter(frame)
    assert len(filtered) == 1
    assert filtered["file"].iloc[0].startswith("sub-001")


def test_dataset_runner_groupings_from_file_column(dummy_ctx):
    runner = DatasetSummaryRunner(dummy_ctx, "ds001", None, "subject")
    frame = pd.DataFrame(
        {
            "file": [
                "sub-001_ses-01_task-rest_eeg.set",
                "sub-001_ses-02_task-rest_eeg.set",
                "sub-002_ses-01_task-rest_eeg.set",
            ]
        }
    )
    grouping = runner._build_groupings(frame)

    subjects = [key[0] for key, _ in grouping]
    assert subjects.count("sub-001") == 2
    assert subjects.count("sub-002") == 1


def test_tig_extension_computation(dummy_ctx, tmp_path):
    """Test TIG extension computation via compute_extensions."""
    extensions_cfg = {"tig": {"enabled": True, "max_lag_sec": 8.0, "n_lags": 4}}

    x = np.ones((10, 3), dtype=np.float32)
    time = np.linspace(0, 9, 10, dtype=np.float32)
    
    payload, summary = compute_extensions(
        dataset_label="ds001:sub-001",
        extensions_cfg=extensions_cfg,
        x=x,
        sub_frame=pd.DataFrame({"dummy": np.arange(10)}),
        time=time,
        dt=1.0,
        coords_9d=None,
        coords_9d_names=[],
        regions_bold=None,
        regions_sfreq=None,
        group_ts={},
        group_matrix=None,
        group_names=[],
        region_groups={},
    )

    assert "tig" in payload
    assert "tig" in summary
    assert payload["tig"]["tau"] == pytest.approx(summary["tig"]["tau"])

