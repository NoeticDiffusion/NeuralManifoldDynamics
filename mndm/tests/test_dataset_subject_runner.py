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
    """Handle dummy ctx."""
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
        extensions_cfg={"tig": {"enabled": True, "max_lag_sec": 8.0, "n_lags": 4}},
        derivative_cfg={"method": "sav_gol", "window": 5, "polyorder": 2},
    )


def test_dataset_runner_subject_filter(dummy_ctx):
    """Test dataset runner subject filter."""
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
    """Test dataset runner groupings from file column."""
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
    """Test dataset runner uses requested worker count."""
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
            """Handle result."""
            return None

    class _FakeExecutor:
        def __init__(self, max_workers: int):
            """Initialize the instance."""
            executor_calls["max_workers"] = max_workers

        def __enter__(self):
            """Dunder method __enter__."""
            return self

        def __exit__(self, exc_type, exc, tb):
            """Dunder method __exit__."""
            return False

        def submit(self, fn, *args, **kwargs):
            """Handle submit."""
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


def test_dataset_runner_keeps_jacobian_hashes_stable_across_n_jobs(dummy_ctx, monkeypatch, tmp_path):
    """Test dataset runner keeps jacobian hashes stable across n jobs."""
    captures: dict[int, dict[str, str]] = {}

    class _ImmediateFuture:
        def result(self):
            """Handle result."""
            return None

    class _FakeExecutor:
        def __init__(self, max_workers: int):
            """Initialize the instance."""
            self.max_workers = max_workers

        def __enter__(self):
            """Dunder method __enter__."""
            return self

        def __exit__(self, exc_type, exc, tb):
            """Dunder method __exit__."""
            return False

        def submit(self, fn, *args, **kwargs):
            """Handle submit."""
            fn(*args, **kwargs)
            return _ImmediateFuture()

    def _run_once(n_jobs: int) -> dict[str, str]:
        """Internal helper: run once."""
        runner = DatasetSummaryRunner(dummy_ctx, "ds001", None, "subject", n_jobs=n_jobs)
        ds_path = tmp_path / f"ds001_{n_jobs}"
        mnps_dir = ds_path / "MNPS"
        ds_path.mkdir(parents=True, exist_ok=True)
        mnps_dir.mkdir(parents=True, exist_ok=True)
        grouping_items = [
            (("sub-001", "ses-01", "rest", "run-01", None), pd.DataFrame({"file": ["sub-001_task-rest_eeg.set"]})),
            (("sub-002", "ses-01", "rest", "run-01", None), pd.DataFrame({"file": ["sub-002_task-rest_eeg.set"]})),
        ]

        monkeypatch.setattr(summary_mod, "ThreadPoolExecutor", _FakeExecutor)
        monkeypatch.setattr(summary_mod, "load_participant_table", lambda *_args, **_kwargs: pd.DataFrame())
        monkeypatch.setattr(runner, "_read_index", lambda _ds_path: pd.DataFrame())
        monkeypatch.setattr(runner, "_read_features", lambda _ds_path: pd.DataFrame({"file": ["ignored"]}))
        monkeypatch.setattr(runner, "_apply_subject_filter", lambda frame: frame)
        monkeypatch.setattr(runner, "_apply_qc_filters", lambda frame: frame)
        monkeypatch.setattr(runner, "_build_groupings", lambda frame: grouping_items)
        monkeypatch.setattr(runner, "_create_output_dir", lambda _ds_path: mnps_dir)
        monkeypatch.setattr(runner, "_write_features_snapshot", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(summary_mod, "write_run_manifest", lambda **_kwargs: None)

        def _fake_subject_run(self, sub_id, ses_id, raw_task, run_id, acq_id, sub_frame):
            """Internal helper: fake subject run."""
            x = np.array(
                [[0.0, 0.1, 0.2], [0.2, 0.0, 0.1], [0.4, -0.1, 0.0], [0.6, -0.2, -0.1]],
                dtype=np.float32,
            )
            x_dot = np.gradient(x, axis=0).astype(np.float32)
            nn_idx = np.tile(np.arange(len(x)), (len(x), 1)).astype(np.int32)
            jac = summary_mod.jacobian.estimate_local_jacobians(x, x_dot, nn_idx, super_window=3, ridge_alpha=1e-4)
            captures.setdefault(n_jobs, {})[sub_id] = summary_mod._stable_hash_array(jac.j_hat)

        monkeypatch.setattr(summary_mod.SubjectSummaryRunner, "run", _fake_subject_run)
        runner.run()
        return dict(captures.get(n_jobs, {}))

    hashes_seq = _run_once(1)
    hashes_parallel = _run_once(2)

    assert hashes_seq == hashes_parallel
    assert set(hashes_seq.keys()) == {"sub-001", "sub-002"}


def test_subject_runner_exports_reproducibility_provenance(dummy_ctx, monkeypatch, tmp_path):
    """Test subject runner exports reproducibility provenance."""
    runner = DatasetSummaryRunner(dummy_ctx, "ds001", None, "subject", n_jobs=1)
    runner.participants_df = pd.DataFrame()
    monkeypatch.setattr(runner, "participant_meta_for", lambda _sub_id: {})
    monkeypatch.setattr(runner, "participant_meta_source_info", lambda: {})
    monkeypatch.setattr(runner, "resolve_coverage_policy", lambda **_kwargs: {"min_epochs": 0, "min_seconds": 0.0, "tag": "default"})
    monkeypatch.setattr(runner, "write_regional_csv_outputs_threadsafe", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "write_stratified_blocks_csv_output_threadsafe", lambda **_kwargs: None)

    subject_runner = SubjectSummaryRunner(
        dataset_runner=runner,
        ds_path=tmp_path,
        mnps_dir=tmp_path / "mnps",
        index_df=None,
    )
    subject_runner.mnps_dir.mkdir(parents=True, exist_ok=True)

    sub_frame = pd.DataFrame(
        {
            "file": ["sub-001_task-rest_eeg.set"] * 6,
            "epoch_id": np.arange(6, dtype=int),
            "t_start": np.arange(6, dtype=float),
            "t_end": np.arange(1, 7, dtype=float),
        }
    )
    x = np.array(
        [
            [0.0, 0.1, 0.2],
            [0.2, 0.0, 0.1],
            [0.4, -0.1, 0.0],
            [0.6, -0.2, -0.1],
            [0.8, -0.1, -0.2],
            [1.0, 0.0, -0.3],
        ],
        dtype=np.float32,
    )
    captures: dict[str, object] = {}

    monkeypatch.setattr(summary_mod, "extract_mapped_metadata", lambda *_args, **_kwargs: {"group": None, "condition": "rest", "task": "rest"})
    monkeypatch.setattr(summary_mod, "build_dataset_label", lambda **_kwargs: "ds001:sub-001:rest_rest")
    monkeypatch.setattr(summary_mod.projection, "project_features_with_coverage", lambda *args, **kwargs: (x, np.ones_like(x, dtype=np.float32), {}))
    monkeypatch.setattr(summary_mod.projection, "build_feature_export_bundle", lambda *args, **kwargs: {"raw_values": np.zeros((len(sub_frame), 0), dtype=np.float32), "raw_names": [], "robust_z_values": np.zeros((len(sub_frame), 0), dtype=np.float32), "robust_z_names": [], "metadata": {}})
    monkeypatch.setattr(summary_mod, "extract_stage_array", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(subject_runner, "_infer_stage_from_bids_events", lambda *_args, **_kwargs: (None, None, None, None))
    monkeypatch.setattr(summary_mod, "extract_embodied_array", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(summary_mod, "extract_events", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(subject_runner, "_load_regional_fmri", lambda **_kwargs: (None, None, None))
    monkeypatch.setattr(summary_mod, "compute_regional_context", lambda **_kwargs: ({}, None, [], {}, None))
    monkeypatch.setattr(summary_mod, "compute_extensions", lambda **_kwargs: ({}, {}))
    monkeypatch.setattr(summary_mod, "compute_ensemble_summary_for_subject", lambda **_kwargs: None)
    monkeypatch.setattr(summary_mod, "compute_robust_and_reliability_summaries", lambda **_kwargs: {})
    monkeypatch.setattr(summary_mod, "compute_dist_summary", lambda **_kwargs: None)
    monkeypatch.setattr(summary_mod, "compute_feature_baseline_comparisons", lambda **_kwargs: None)
    monkeypatch.setattr(summary_mod, "compute_tau_summary", lambda *args, **kwargs: {})
    monkeypatch.setattr(summary_mod, "compute_tier2_jacobian_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(summary_mod, "compute_emmi_metrics", lambda **_kwargs: None)
    monkeypatch.setattr(summary_mod, "compute_null_sanity_tests", lambda **_kwargs: None)
    monkeypatch.setattr(summary_mod, "compute_psd_multiverse_stability", lambda **_kwargs: None)
    monkeypatch.setattr(summary_mod.robustness, "entropy_sanity_checks", lambda *args, **kwargs: {})
    monkeypatch.setattr(summary_mod, "_get_env_provenance", lambda: {})
    monkeypatch.setattr(subject_runner, "_write_qc_files", lambda **_kwargs: None)

    def _capture_write(*, target_dir, dataset_label, manifest, payload, **kwargs):
        """Internal helper: capture write."""
        captures["manifest"] = manifest
        captures["payload"] = payload

    monkeypatch.setattr(summary_mod, "write_summary_manifest_and_h5", _capture_write)

    subject_runner.run(
        sub_id="sub-001",
        ses_id="ses-01",
        raw_task="rest",
        run_id="run-01",
        acq_id=None,
        sub_frame=sub_frame,
    )

    payload = captures["payload"]
    manifest = captures["manifest"]
    assert payload.attrs["jacobian_hash_saved"]
    assert payload.attrs["jacobian_dot_hash_saved"]
    assert payload.attrs["reproducibility_seed"] == 42
    repro = manifest["provenance"]["reproducibility"]
    assert repro["seed"] == 42
    assert repro["jacobian_hash_saved"] == payload.attrs["jacobian_hash_saved"]

