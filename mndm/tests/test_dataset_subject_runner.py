"""Tests for DatasetSummaryRunner and SubjectSummaryRunner."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("mne")

import mndm.pipeline.summary as summary_mod
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


def test_dataset_runner_uses_requested_worker_count(dummy_ctx, monkeypatch, tmp_path):
    runner = DatasetSummaryRunner(dummy_ctx, "ds001", None, "subject", n_jobs=3)
    ds_path = tmp_path / "ds001"
    mnps_dir = ds_path / "MNPS"
    ds_path.mkdir(parents=True, exist_ok=True)
    mnps_dir.mkdir(parents=True, exist_ok=True)

    grouping_items = [
        (("sub-001", "ses-01", "rest", "run-01", None), pd.DataFrame({"file": ["sub-001_task-rest_eeg.set"]})),
        (("sub-002", "ses-01", "rest", "run-01", None), pd.DataFrame({"file": ["sub-002_task-rest_eeg.set"]})),
        (("sub-003", "ses-01", "rest", "run-01", None), pd.DataFrame({"file": ["sub-003_task-rest_eeg.set"]})),
    ]
    executor_calls: dict[str, int] = {"max_workers": 0, "submitted": 0}
    processed: list[tuple[str, int]] = []

    class _ImmediateFuture:
        def result(self):
            return None

    class _FakeExecutor:
        def __init__(self, max_workers: int):
            executor_calls["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            executor_calls["submitted"] += 1
            fn(*args, **kwargs)
            return _ImmediateFuture()

    monkeypatch.setattr(summary_mod, "load_participant_table", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(runner, "_read_index", lambda _ds_path: pd.DataFrame())
    monkeypatch.setattr(runner, "_read_features", lambda _ds_path: pd.DataFrame({"file": ["ignored"]}))
    monkeypatch.setattr(runner, "_apply_subject_filter", lambda frame: frame)
    monkeypatch.setattr(runner, "_apply_qc_filters", lambda frame: frame)
    monkeypatch.setattr(runner, "_build_groupings", lambda frame: grouping_items)
    monkeypatch.setattr(runner, "_create_output_dir", lambda _ds_path: mnps_dir)
    monkeypatch.setattr(runner, "_write_features_snapshot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "_process_grouping_item",
        lambda _ds_path, _mnps_dir, grouping_key, sub_frame: processed.append((grouping_key[0], len(sub_frame))),
    )
    monkeypatch.setattr(summary_mod, "write_run_manifest", lambda **_kwargs: None)
    monkeypatch.setattr(summary_mod, "ThreadPoolExecutor", _FakeExecutor)

    runner.run()

    assert executor_calls["max_workers"] == 3
    assert executor_calls["submitted"] == 3
    assert processed == [("sub-001", 1), ("sub-002", 1), ("sub-003", 1)]

