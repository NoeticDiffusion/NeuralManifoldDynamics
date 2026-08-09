"""Tests for DatasetSummaryRunner and SubjectSummaryRunner."""

import json
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


def test_dataset_runner_subject_filter_accepts_variable_bids_padding(dummy_ctx):
    """Numeric subject filters should match BIDS labels with different padding."""
    runner = DatasetSummaryRunner(dummy_ctx, "ds001", "01", "subject")
    frame = pd.DataFrame(
        {
            "file": [
                "sub-01_ses-01_task-rest_eeg.set",
                "sub-02_ses-01_task-rest_eeg.set",
            ]
        }
    )

    filtered = runner._apply_subject_filter(frame)

    assert filtered is not None
    assert len(filtered) == 1
    assert filtered["file"].iloc[0].startswith("sub-01")


def test_dataset_runner_grouping_accepts_variable_bids_padding(dummy_ctx):
    """Summary grouping must preserve numeric subject matching after filtering."""
    runner = DatasetSummaryRunner(dummy_ctx, "ds001", "01", "subject")
    frame = pd.DataFrame(
        {
            "file": [
                "sub-01_ses-01_task-rest_eeg.set",
                "sub-02_ses-01_task-rest_eeg.set",
            ]
        }
    )

    groups = runner._build_groupings(frame)

    assert len(groups) == 1
    assert groups[0][0][0] == "sub-01"


def test_dataset_runner_subject_filter_non_bids_filename_regex(dummy_ctx):
    """Subject filter should work for non-BIDS filenames via filename_parse."""
    dummy_ctx.config = {
        "robustness": {"coverage": {}},
        "metadata_extraction": {
            "datasets": {
                "physionet_icare_2_1": {
                    "filename_parse": {
                        "regex": r"^(?P<subject>\d{4})_(?P<run>\d{3})_(?P<acq>\d{3})_EEG\.hea$",
                        "subject_pad": 4,
                    }
                }
            }
        },
    }
    runner = DatasetSummaryRunner(dummy_ctx, "physionet_icare_2_1", "0332", "subject")
    frame = pd.DataFrame(
        {
            "file": [
                "0332_001_022_EEG.hea",
                "0333_001_022_EEG.hea",
            ]
        }
    )

    filtered = runner._apply_subject_filter(frame)
    assert filtered is not None
    assert len(filtered) == 1
    assert filtered["file"].iloc[0] == "0332_001_022_EEG.hea"


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


def test_dataset_runner_applies_combat_normalization(dummy_ctx):
    """ComBat should reduce simple site offsets when metadata is present."""
    pytest.importorskip("neuroCombat")
    dummy_ctx.config = {
        "robustness": {"coverage": {}},
        "normalization": {
            "enabled": True,
            "method": "combat",
            "scope": "post_features",
            "batch_key": "hospital",
            "covariates": [],
            "combat": {
                "chunk_size": 2,
                "min_batch_size": 2,
                "min_feature_observations": 4,
                "winsorize_quantiles": [0.0, 1.0],
                "family_wise": {
                    "enabled": True,
                    "strategy": "prefix",
                    "delimiter": "_",
                    "min_family_columns": 1,
                },
            },
            "validation": {
                "enabled": True,
                "max_rows": 200,
                "max_features": 8,
                "min_group_size": 2,
                "max_levels": 8,
                "batch_key": "auto",
                "target_keys": ["outcome"],
                "metrics": {
                    "batch_eta2": True,
                    "target_eta2": True,
                    "perturbation": True,
                },
            },
        },
    }
    runner = DatasetSummaryRunner(dummy_ctx, "physionet_icare_2_1", None, "subject")
    runner.participants_df = pd.DataFrame(
        {
            "participant_id": ["sub-0001", "sub-0002", "sub-0003", "sub-0004"],
            "hospital": ["A", "A", "B", "B"],
            "age": [55, 61, 58, 63],
            "sex": ["F", "M", "F", "M"],
            "outcome": ["good", "good", "poor", "poor"],
        }
    )
    runner._build_participant_meta_map()
    hospital_by_subject = dict(
        zip(
            runner.participants_df["participant_id"].astype(str),
            runner.participants_df["hospital"].astype(str),
        )
    )

    rows = []
    for subject, subject_shift in [("sub-0001", 0.10), ("sub-0002", -0.05), ("sub-0003", 0.20), ("sub-0004", -0.12)]:
        site_shift = 0.0 if hospital_by_subject[subject] == "A" else 6.0
        for epoch in range(10):
            rows.append(
                {
                    "file": f"{subject}_task-rest_eeg.set",
                    "subject": subject,
                    "epoch_id": len(rows),
                    "eeg_alpha": site_shift + subject_shift + 0.02 * epoch,
                        "eeg_beta": 0.5 * site_shift + subject_shift + 0.03 * epoch,
                    "ecg_rmssd": 2.0 * site_shift + subject_shift + 0.05 * epoch,
                }
            )
    features_df = pd.DataFrame(rows)

    hosp_series = features_df["subject"].map(hospital_by_subject)
    before_gap = float(
        abs(
            features_df.loc[hosp_series == "A", "eeg_alpha"].mean()
            - features_df.loc[hosp_series == "B", "eeg_alpha"].mean()
        )
    )

    out_df = runner._apply_feature_normalization(features_df.copy())
    after_gap = float(
        abs(
            out_df.loc[hosp_series == "A", "eeg_alpha"].mean()
            - out_df.loc[hosp_series == "B", "eeg_alpha"].mean()
        )
    )

    assert runner._normalization_report["status"] == "applied"
    assert runner._normalization_report["feature_columns_harmonized"] >= 2
    assert runner._normalization_report["family_wise"]["enabled"] is True
    assert runner._normalization_report["family_wise"]["family_count"] >= 2
    assert runner._normalization_report["validation"]["status"] == "computed"
    assert "batch_eta2" in runner._normalization_report["validation"]["probes"]
    assert "target_eta2" in runner._normalization_report["validation"]["probes"]
    assert after_gap < before_gap


def test_dataset_runner_combat_preserves_single_feature_family(dummy_ctx):
    """Single-feature family chunks should be left unchanged (not NaN-harmonized)."""
    pytest.importorskip("neuroCombat")
    dummy_ctx.config = {
        "robustness": {"coverage": {}},
        "normalization": {
            "enabled": True,
            "method": "combat",
            "scope": "post_features",
            "batch_key": "hospital",
            "covariates": ["age", "sex"],
            "combat": {
                "chunk_size": 24,
                "min_batch_size": 2,
                "min_feature_observations": 4,
                "winsorize_quantiles": [0.005, 0.995],
                "family_wise": {
                    "enabled": True,
                    "strategy": "prefix",
                    "delimiter": "_",
                    "min_family_columns": 1,
                },
            },
            "validation": {"enabled": False},
        },
    }
    runner = DatasetSummaryRunner(dummy_ctx, "physionet_icare_2_1", None, "subject")
    runner.participants_df = pd.DataFrame(
        {
            "participant_id": ["sub-0001", "sub-0002", "sub-0003", "sub-0004"],
            "hospital": ["A", "A", "B", "B"],
            "age": [55, 61, 58, 63],
            "sex": ["F", "M", "F", "M"],
        }
    )
    runner._build_participant_meta_map()
    hospital_by_subject = dict(
        zip(
            runner.participants_df["participant_id"].astype(str),
            runner.participants_df["hospital"].astype(str),
        )
    )

    rows = []
    for subject, subject_shift in [("sub-0001", 0.10), ("sub-0002", -0.05), ("sub-0003", 0.20), ("sub-0004", -0.12)]:
        site_shift = 0.0 if hospital_by_subject[subject] == "A" else 6.0
        for epoch in range(10):
            rows.append(
                {
                    "file": f"{subject}_task-rest_eeg.set",
                    "subject": subject,
                    "epoch_id": len(rows),
                    "eeg_alpha": site_shift + subject_shift + 0.02 * epoch,
                        "eeg_beta": 0.5 * site_shift + subject_shift + 0.03 * epoch,
                    # This lands in __other__ and is a single-feature family chunk.
                    "embodied_arousal_proxy": (site_shift * 20.0) + (0.5 * epoch) + subject_shift,
                }
            )
    features_df = pd.DataFrame(rows)
    embodied_before = features_df["embodied_arousal_proxy"].to_numpy(dtype=float)

    out_df = runner._apply_feature_normalization(features_df.copy())
    embodied_after = out_df["embodied_arousal_proxy"].to_numpy(dtype=float)

    assert np.isfinite(embodied_after).all()
    assert np.allclose(embodied_after, embodied_before, equal_nan=False)
    assert runner._normalization_report["status"] == "applied"
    assert runner._normalization_report["skipped_columns"].get("single_feature_family", 0) >= 1
    families = runner._normalization_report["family_wise"]["families"]
    assert any(
        int(stats.get("feature_columns_total", 0)) == 1
        and int(stats.get("feature_columns_harmonized", 0)) == 0
        and int(stats.get("chunks_skipped", 0)) >= 1
        for stats in families.values()
    )


def test_dataset_runner_writes_normalization_report_file(dummy_ctx, tmp_path):
    """Normalization report sidecar should always be writable."""
    runner = DatasetSummaryRunner(dummy_ctx, "ds001", None, "subject")
    runner._normalization_report = {
        "enabled": False,
        "status": "disabled",
        "method": "combat",
        "scope": "post_features",
    }
    info = runner._write_normalization_report_file(tmp_path)
    assert info["status"] == "written"
    assert info["path"] == "normalization_report.json"
    report_path = tmp_path / "normalization_report.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "mndm.normalization_report.v1"
    assert payload["normalization"]["status"] == "disabled"


def test_resolve_mnps_9d_runtime_config_replaces_versioned_subcoords():
    """Versioned subcoords should replace, not blend with, legacy root subcoords."""
    enabled, version, selected, subcoords = summary_mod._resolve_mnps_9d_runtime_config(
        {
            "enabled": True,
            "definition_version": "2.0",
            "subcoords": {
                "m_a": {"legacy_theta": 1.0},
                "e_s": {"legacy_alpha_theta": 0.5, "legacy_gamma": 0.5},
            },
            "metric_policies": {"e_e": {"preferred": "legacy_entropy"}},
            "versions": {
                "2.0": {
                    "subcoords": {
                        "m_a": {"modern_delta": -0.5, "modern_theta": -0.5},
                        "m_e": {"modern_alpha": -1.0},
                        "d_s": {"modern_alpha_theta": 1.0},
                        "e_s": {"modern_hjorth_complexity": 1.0},
                    },
                    "metric_policies": {"e_e": {"preferred": "permutation_entropy"}},
                }
            },
            "datasets": {
                "dsX": {
                    "subcoords": {
                        "e_s": {"dataset_hjorth_complexity": 1.0},
                    }
                }
            },
        },
        "dsX",
    )

    assert enabled is True
    assert version == "2.0"
    assert selected["subcoords"]["m_a"] == {"modern_delta": -0.5, "modern_theta": -0.5}
    assert selected["subcoords"]["m_e"] == {"modern_alpha": -1.0}
    assert selected["subcoords"]["e_s"] == {"dataset_hjorth_complexity": 1.0}
    assert "legacy_theta" not in selected["subcoords"]["m_a"]
    assert "legacy_alpha_theta" not in selected["subcoords"]["e_s"]
    assert subcoords["e_s"] == {"dataset_hjorth_complexity": 1.0}
    assert selected["metric_policies"]["e_e"]["preferred"] == "permutation_entropy"


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


def test_dataset_runner_writes_manifest_and_run_errors_on_group_failure(dummy_ctx, monkeypatch, tmp_path):
    """Dataset runner should still emit run manifest + run_errors on subject failures."""
    runner = DatasetSummaryRunner(dummy_ctx, "ds001", None, "subject", n_jobs=1)
    ds_path = tmp_path / "ds001"
    mnps_dir = ds_path / "MNPS"
    ds_path.mkdir(parents=True, exist_ok=True)
    mnps_dir.mkdir(parents=True, exist_ok=True)

    grouping_items = [
        (("sub-001", "ses-01", "rest", "run-01", None), pd.DataFrame({"file": ["sub-001_task-rest_eeg.set"]})),
        (("sub-002", "ses-01", "rest", "run-01", None), pd.DataFrame({"file": ["sub-002_task-rest_eeg.set"]})),
    ]

    monkeypatch.setattr(summary_mod, "load_participant_table", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(runner, "_read_index", lambda _ds_path: pd.DataFrame())
    monkeypatch.setattr(runner, "_read_features", lambda _ds_path: pd.DataFrame({"file": ["ignored"]}))
    monkeypatch.setattr(runner, "_apply_subject_filter", lambda frame: frame)
    monkeypatch.setattr(runner, "_apply_qc_filters", lambda frame: frame)
    monkeypatch.setattr(runner, "_build_groupings", lambda frame: grouping_items)
    monkeypatch.setattr(runner, "_create_output_dir", lambda _ds_path: mnps_dir)
    monkeypatch.setattr(runner, "_prepare_one_shot_anchor", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_write_features_snapshot", lambda *_args, **_kwargs: None)

    def _fake_subject_run(self, sub_id, ses_id, raw_task, run_id, acq_id, sub_frame):
        """Internal helper: fake subject run with one forced failure."""
        if sub_id == "sub-002":
            raise RuntimeError("intentional failure for smoke test")

    monkeypatch.setattr(summary_mod.SubjectSummaryRunner, "run", _fake_subject_run)

    runner.run()

    manifest_path = mnps_dir / "run_manifest.json"
    errors_path = mnps_dir / "run_errors.json"
    normalization_path = mnps_dir / "normalization_report.json"
    assert manifest_path.exists()
    assert errors_path.exists()
    assert normalization_path.exists()

    run_errors = json.loads(errors_path.read_text(encoding="utf-8"))
    assert run_errors["counts"]["errors_total"] == 1
    assert run_errors["counts"]["groupings_total"] == 2
    assert run_errors["errors"][0]["subject"] == "sub-002"
    assert run_errors["errors"][0]["stage"] == "grouping"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["extra"]["run_status"] == "completed_with_errors"
    assert manifest["extra"]["run_errors"]["count"] == 1
    assert manifest["extra"]["run_errors"]["path"] == "run_errors.json"
    assert manifest["extra"]["normalization_report"]["path"] == "normalization_report.json"


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
    dummy_ctx.config = {
        "robustness": {"coverage": {}},
        "anchor_state": {"enabled": True},
    }
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
            "t_start": np.arange(0, 18, 3, dtype=float),
            "t_end": np.arange(4, 22, 3, dtype=float),
            "ecg_hr_bpm": np.linspace(60.0, 70.0, 6),
            "ecg_rmssd": np.linspace(40.0, 30.0, 6),
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
    monkeypatch.setattr(
        summary_mod.projection,
        "build_feature_export_bundle",
        lambda *args, **kwargs: {
            "raw_values": sub_frame[["ecg_hr_bpm", "ecg_rmssd"]].to_numpy(dtype=np.float32),
            "raw_names": ["ecg_hr_bpm", "ecg_rmssd"],
            "robust_z_values": np.column_stack(
                [
                    np.linspace(-1.0, 1.0, len(sub_frame), dtype=np.float32),
                    np.linspace(1.0, -1.0, len(sub_frame), dtype=np.float32),
                ]
            ),
            "robust_z_names": ["ecg_hr_bpm", "ecg_rmssd"],
            "projection_z_values": np.zeros((len(sub_frame), 2), dtype=np.float32),
            "projection_z_names": ["ecg_hr_bpm", "ecg_rmssd"],
            "metadata": {
                "robust_z_valid": np.ones(2, dtype=np.int8),
                "robust_z_invalid_reason": np.asarray(["", ""], dtype=object),
            },
        },
    )
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
    assert manifest["anchor_state_validation"]["status"] == "ok"
    assert manifest["anchor_state_validation"]["guard_policy_version"] == "mndm.anchor_guard.v1"


def test_subject_runner_exports_meg_mapping_provenance(dummy_ctx, monkeypatch, tmp_path):
    """MEG shadow runs should stamp mapping contract metadata into attrs and provenance."""
    dummy_ctx.config = {
        "modality": "meg",
        "robustness": {"coverage": {}},
        "meg_mapping": {
            "enabled": True,
            "primary_surface": "meg_sensor_shadow",
            "paired_surfaces": ["eeg"],
            "mapping_family": "electrophysiology_shadow",
            "mapping_reference": "eeg_contract_v2",
            "sensor_types": ["mag", "grad"],
            "feature_combination": "robust_z_then_median",
            "validation_pilot": {"subjects": ["sub-002", "sub-003", "sub-004", "sub-005", "sub-006"]},
        },
    }
    runner = DatasetSummaryRunner(dummy_ctx, "ds003645", None, "subject", n_jobs=1)
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
            "file": ["sub-002_task-facerecognition_meg.fif"] * 6,
            "epoch_id": np.arange(6, dtype=int),
            "t_start": np.arange(0, 18, 3, dtype=float),
            "t_end": np.arange(4, 22, 3, dtype=float),
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

    monkeypatch.setattr(summary_mod, "extract_mapped_metadata", lambda *_args, **_kwargs: {"group": None, "condition": "meeg", "task": "facerecognition"})
    monkeypatch.setattr(summary_mod, "build_dataset_label", lambda **_kwargs: "ds003645:sub-002:meeg_facerecognition")
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
        captures["manifest"] = manifest
        captures["payload"] = payload

    monkeypatch.setattr(summary_mod, "write_summary_manifest_and_h5", _capture_write)

    subject_runner.run(
        sub_id="sub-002",
        ses_id="ses-01",
        raw_task="facerecognition",
        run_id="run-01",
        acq_id=None,
        sub_frame=sub_frame,
    )

    payload = captures["payload"]
    manifest = captures["manifest"]
    assert payload.attrs["modality"] == "meg"
    assert payload.attrs["mapping_family"] == "electrophysiology_shadow"
    assert payload.attrs["mapping_reference"] == "eeg_contract_v2"
    assert payload.attrs["sensor_types"] == ["mag", "grad"]
    assert payload.provenance["mapping"]["feature_combination"] == "robust_z_then_median"
    assert payload.provenance["mapping"]["validation_pilot"]["subjects"] == [
        "sub-002",
        "sub-003",
        "sub-004",
        "sub-005",
        "sub-006",
    ]
    assert manifest["provenance"]["mapping"]["mapping_family"] == "electrophysiology_shadow"


def test_subject_runner_exports_time_reference_extension(dummy_ctx, monkeypatch, tmp_path):
    """Subject runner exports time-reference attrs + extension + manifest."""
    dataset_root = tmp_path / "received_physio" / "training"
    header_dir = dataset_root / "0332"
    header_dir.mkdir(parents=True, exist_ok=True)
    header_path = header_dir / "0332_001_022_EEG.hea"
    header_path.write_text(
        "\n".join(
            [
                "0332_001_022_EEG 1 200 1000",
                "#Start time: 22:59:08",
                "#End time: 23:10:08",
            ]
        ),
        encoding="utf-8",
    )

    dummy_ctx.config = {
        "robustness": {"coverage": {}},
        "paths": {
            "dataset_received_dirs": {
                "physionet_icare_2_1": str(dataset_root),
            }
        },
        "time_reference": {
            "enabled": True,
            "schema_version": "time_reference.v1",
            "parser": "wfdb_header",
            "anchor": "first_recording",
            "bins_hours": [0, 24, 48, 72],
            "datasets": {"physionet_icare_2_1": {"enabled": True}},
        },
    }
    runner = DatasetSummaryRunner(dummy_ctx, "physionet_icare_2_1", None, "subject", n_jobs=1)
    runner.participants_df = pd.DataFrame()
    runner._build_participant_meta_map()
    monkeypatch.setattr(runner, "participant_meta_for", lambda _sub_id: {})
    monkeypatch.setattr(runner, "participant_meta_source_info", lambda: {})
    monkeypatch.setattr(
        runner,
        "resolve_coverage_policy",
        lambda **_kwargs: {"min_epochs": 0, "min_seconds": 0.0, "tag": "default"},
    )
    monkeypatch.setattr(runner, "write_regional_csv_outputs_threadsafe", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "write_stratified_blocks_csv_output_threadsafe", lambda **_kwargs: None)

    index_df = pd.DataFrame(
        [
            {
                "path": "0332/0332_001_022_EEG.hea",
                "subject": "0332",
                "run": "001",
                "acq": "022",
                "modality": "eeg",
            }
        ]
    )
    subject_runner = SubjectSummaryRunner(
        dataset_runner=runner,
        ds_path=tmp_path,
        mnps_dir=tmp_path / "mnps",
        index_df=index_df,
    )
    subject_runner.mnps_dir.mkdir(parents=True, exist_ok=True)

    sub_frame = pd.DataFrame(
        {
            "file": ["0332/0332_001_022_EEG.hea"] * 6,
            "epoch_id": np.arange(6, dtype=int),
            "t_start": np.arange(0, 12, 2, dtype=float),
            "t_end": np.arange(2, 14, 2, dtype=float),
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

    monkeypatch.setattr(
        summary_mod,
        "extract_mapped_metadata",
        lambda *_args, **_kwargs: {"group": None, "condition": "rest", "task": "continuous_eeg"},
    )
    monkeypatch.setattr(summary_mod, "build_dataset_label", lambda **_kwargs: "physionet_icare_2_1:sub-0332:rest")
    monkeypatch.setattr(
        summary_mod.projection,
        "project_features_with_coverage",
        lambda *args, **kwargs: (x, np.ones_like(x, dtype=np.float32), {}),
    )
    monkeypatch.setattr(
        summary_mod.projection,
        "build_feature_export_bundle",
        lambda frame, *args, **kwargs: {
            "raw_values": np.zeros((len(frame), 0), dtype=np.float32),
            "raw_names": [],
            "robust_z_values": np.zeros((len(frame), 0), dtype=np.float32),
            "robust_z_names": [],
            "metadata": {},
        },
    )
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
        """Capture summary write arguments for assertions."""
        captures["manifest"] = manifest
        captures["payload"] = payload

    monkeypatch.setattr(summary_mod, "write_summary_manifest_and_h5", _capture_write)

    subject_runner.run(
        sub_id="sub-0332",
        ses_id=None,
        raw_task="continuous_eeg",
        run_id="001",
        acq_id="022",
        sub_frame=sub_frame,
    )

    payload = captures["payload"]
    manifest = captures["manifest"]
    assert "time_reference" in payload.extensions
    assert "run" in payload.extensions["time_reference"]
    assert "windows" in payload.extensions["time_reference"]
    assert payload.attrs["time_reference_schema_version"] == "time_reference.v1"
    assert payload.attrs["time_reference_status"] == "ok"
    assert "time_reference" in manifest
    assert manifest["time_reference"]["status"] == "ok"


def test_subject_runner_exports_conventional_eeg_summary(dummy_ctx, monkeypatch, tmp_path):
    """Conventional EEG summaries should be exported separately from MNPS geometry."""
    dummy_ctx.config = {
        "robustness": {"coverage": {}},
        "conventional_eeg": {
            "enabled": True,
            "packs": ["tier1"],
            "export": {"per_epoch_columns": True, "summaries": True},
            "tier1": {
                "relative_bandpower": True,
                "ratios": ["theta_alpha", "slowing_index"],
                "peak_frequency": {
                    "alpha_peak_frequency": True,
                    "median_frequency": True,
                    "spectral_edge_95": True,
                },
            },
        },
    }
    runner = DatasetSummaryRunner(dummy_ctx, "ds001", None, "subject", n_jobs=1)
    runner.participants_df = pd.DataFrame()
    monkeypatch.setattr(runner, "participant_meta_for", lambda _sub_id: {})
    monkeypatch.setattr(runner, "participant_meta_source_info", lambda: {})
    monkeypatch.setattr(
        runner,
        "resolve_coverage_policy",
        lambda **_kwargs: {"min_epochs": 0, "min_seconds": 0.0, "tag": "default"},
    )
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
            "eeg_conventional_relative_alpha": np.linspace(0.20, 0.45, 6),
            "eeg_conventional_relative_theta": np.linspace(0.30, 0.10, 6),
            "eeg_conventional_ratio_theta_alpha": np.linspace(1.5, 0.4, 6),
            "eeg_conventional_ratio_slowing_index": np.linspace(1.2, 0.5, 6),
            "eeg_conventional_peak_alpha_frequency": np.linspace(9.0, 10.0, 6),
            "eeg_conventional_peak_median_frequency": np.linspace(11.0, 13.0, 6),
            "eeg_conventional_peak_spectral_edge_95": np.linspace(24.0, 28.0, 6),
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
    monkeypatch.setattr(
        summary_mod.projection,
        "project_features_with_coverage",
        lambda *args, **kwargs: (x, np.ones_like(x, dtype=np.float32), {}),
    )
    monkeypatch.setattr(
        summary_mod.projection,
        "build_feature_export_bundle",
        lambda frame, *args, **kwargs: {
            "raw_values": np.zeros((len(frame), 0), dtype=np.float32),
            "raw_names": [],
            "robust_z_values": np.zeros((len(frame), 0), dtype=np.float32),
            "robust_z_names": [],
            "metadata": {},
        },
    )
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
        """Capture summary write arguments for assertions."""
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
    assert "conventional_eeg" in payload.extensions
    assert payload.extensions["conventional_eeg"]["schema_version"] == "mndm.conventional_eeg.v1"
    assert "relative" in payload.extensions["conventional_eeg"]["families"]
    assert "ratio" in payload.extensions["conventional_eeg"]["families"]
    assert "peak" in payload.extensions["conventional_eeg"]["families"]
    assert "conventional_eeg" in manifest
    assert manifest["conventional_eeg"]["column_count"] == 7
    assert "alpha" in manifest["conventional_eeg"]["families"]["relative"]
    assert manifest["conventional_eeg"]["families"]["peak"]["alpha_frequency"]["column"] == "eeg_conventional_peak_alpha_frequency"


def test_subject_runner_exports_conventional_eeg_complexity_summary(dummy_ctx, monkeypatch, tmp_path):
    """Complexity-only conventional packs should still be summarized."""
    dummy_ctx.config = {
        "robustness": {"coverage": {}},
        "conventional_eeg": {
            "enabled": True,
            "packs": ["complexity"],
            "export": {"per_epoch_columns": True, "summaries": True},
            "complexity": {
                "spectral_entropy": True,
                "permutation_entropy": True,
                "hjorth_complexity": True,
                "hjorth_mobility": True,
            },
        },
    }
    runner = DatasetSummaryRunner(dummy_ctx, "ds001", None, "subject", n_jobs=1)
    runner.participants_df = pd.DataFrame()
    monkeypatch.setattr(runner, "participant_meta_for", lambda _sub_id: {})
    monkeypatch.setattr(runner, "participant_meta_source_info", lambda: {})
    monkeypatch.setattr(
        runner,
        "resolve_coverage_policy",
        lambda **_kwargs: {"min_epochs": 0, "min_seconds": 0.0, "tag": "default"},
    )
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
            "eeg_conventional_complexity_spectral_entropy": np.linspace(2.1, 2.6, 6),
            "eeg_conventional_complexity_permutation_entropy": np.linspace(0.6, 0.8, 6),
            "eeg_conventional_complexity_hjorth_complexity": np.linspace(1.3, 1.7, 6),
            "eeg_conventional_complexity_hjorth_mobility": np.linspace(0.8, 1.1, 6),
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
    monkeypatch.setattr(
        summary_mod.projection,
        "project_features_with_coverage",
        lambda *args, **kwargs: (x, np.ones_like(x, dtype=np.float32), {}),
    )
    monkeypatch.setattr(
        summary_mod.projection,
        "build_feature_export_bundle",
        lambda frame, *args, **kwargs: {
            "raw_values": np.zeros((len(frame), 0), dtype=np.float32),
            "raw_names": [],
            "robust_z_values": np.zeros((len(frame), 0), dtype=np.float32),
            "robust_z_names": [],
            "metadata": {},
        },
    )
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
        """Capture summary write arguments for assertions."""
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
    assert "conventional_eeg" in payload.extensions
    assert payload.extensions["conventional_eeg"]["schema_version"] == "mndm.conventional_eeg.v1"
    assert payload.extensions["conventional_eeg"]["packs"] == ["complexity"]
    assert "complexity" in payload.extensions["conventional_eeg"]["families"]
    assert "conventional_eeg" in manifest
    assert manifest["conventional_eeg"]["column_count"] == 4
    assert "spectral_entropy" in manifest["conventional_eeg"]["families"]["complexity"]
    assert (
        manifest["conventional_eeg"]["families"]["complexity"]["hjorth_complexity"]["column"]
        == "eeg_conventional_complexity_hjorth_complexity"
    )


def test_subject_runner_exports_conventional_eeg_connectivity_summary(dummy_ctx, monkeypatch, tmp_path):
    """Connectivity-only conventional packs should be summarized."""
    dummy_ctx.config = {
        "robustness": {"coverage": {}},
        "conventional_eeg": {
            "enabled": True,
            "packs": ["connectivity"],
            "export": {"per_epoch_columns": True, "summaries": True},
            "connectivity": {
                "roi_pairs": [
                    {"name": "FP", "channels": ["F3", "P3"]},
                    {"name": "FB", "channels": ["Fz", "POz"]},
                ],
                "metrics": {"plv": True, "coherence": True},
                "outputs": {"summary_stats": ["mean", "std"]},
            },
        },
    }
    runner = DatasetSummaryRunner(dummy_ctx, "ds001", None, "subject", n_jobs=1)
    runner.participants_df = pd.DataFrame()
    monkeypatch.setattr(runner, "participant_meta_for", lambda _sub_id: {})
    monkeypatch.setattr(runner, "participant_meta_source_info", lambda: {})
    monkeypatch.setattr(
        runner,
        "resolve_coverage_policy",
        lambda **_kwargs: {"min_epochs": 0, "min_seconds": 0.0, "tag": "default"},
    )
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
            "eeg_conventional_connectivity_alpha_FP_plv_mean": np.linspace(0.45, 0.60, 6),
            "eeg_conventional_connectivity_alpha_FB_coh_mean": np.linspace(0.20, 0.35, 6),
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
    monkeypatch.setattr(
        summary_mod.projection,
        "project_features_with_coverage",
        lambda *args, **kwargs: (x, np.ones_like(x, dtype=np.float32), {}),
    )
    monkeypatch.setattr(
        summary_mod.projection,
        "build_feature_export_bundle",
        lambda *args, **kwargs: {
            "raw_values": np.zeros((len(sub_frame), 0), dtype=np.float32),
            "raw_names": [],
            "robust_z_values": np.zeros((len(sub_frame), 0), dtype=np.float32),
            "robust_z_names": [],
            "metadata": {},
        },
    )
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
        """Capture summary write arguments for assertions."""
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
    assert "conventional_eeg" in payload.extensions
    assert payload.extensions["conventional_eeg"]["packs"] == ["connectivity"]
    assert "connectivity" in payload.extensions["conventional_eeg"]["families"]
    assert "conventional_eeg" in manifest
    assert manifest["conventional_eeg"]["column_count"] == 2
    assert "alpha_FP_plv_mean" in manifest["conventional_eeg"]["families"]["connectivity"]
    assert (
        manifest["conventional_eeg"]["families"]["connectivity"]["alpha_FB_coh_mean"]["column"]
        == "eeg_conventional_connectivity_alpha_FB_coh_mean"
    )


def test_subject_runner_exports_conventional_eeg_coma_summary(dummy_ctx, monkeypatch, tmp_path):
    """Coma pack should expose family summaries and unavailable clinical marker metadata."""
    dummy_ctx.config = {
        "robustness": {"coverage": {}},
        "conventional_eeg": {
            "enabled": True,
            "packs": ["coma"],
            "export": {"per_epoch_columns": True, "summaries": True},
            "coma": {
                "suppression_ratio": True,
                "burst_suppression_proxy": True,
                "continuity_proxy": True,
                "alpha_delta_ratio": True,
                "reactivity_proxy": {"enabled": True},
            },
        },
    }
    runner = DatasetSummaryRunner(dummy_ctx, "ds001", None, "subject", n_jobs=1)
    runner.participants_df = pd.DataFrame()
    monkeypatch.setattr(runner, "participant_meta_for", lambda _sub_id: {})
    monkeypatch.setattr(runner, "participant_meta_source_info", lambda: {})
    monkeypatch.setattr(
        runner,
        "resolve_coverage_policy",
        lambda **_kwargs: {"min_epochs": 0, "min_seconds": 0.0, "tag": "default"},
    )
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
            "eeg_conventional_coma_suppression_ratio": np.linspace(0.75, 0.25, 6),
            "eeg_conventional_coma_continuity_proxy": np.linspace(0.25, 0.75, 6),
            "eeg_conventional_coma_burst_suppression_proxy": np.linspace(0.15, 0.35, 6),
            "eeg_conventional_coma_alpha_delta_ratio": np.linspace(0.35, 0.55, 6),
            "eeg_conventional_coma_reactivity_proxy": np.linspace(0.10, 0.50, 6),
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
    monkeypatch.setattr(
        summary_mod.projection,
        "project_features_with_coverage",
        lambda *args, **kwargs: (x, np.ones_like(x, dtype=np.float32), {}),
    )
    monkeypatch.setattr(
        summary_mod.projection,
        "build_feature_export_bundle",
        lambda *args, **kwargs: {
            "raw_values": np.zeros((len(sub_frame), 0), dtype=np.float32),
            "raw_names": [],
            "robust_z_values": np.zeros((len(sub_frame), 0), dtype=np.float32),
            "robust_z_names": [],
            "metadata": {},
        },
    )
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
        """Capture summary write arguments for assertions."""
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
    assert "conventional_eeg" in payload.extensions
    assert payload.extensions["conventional_eeg"]["schema_version"] == "mndm.conventional_eeg.v1"
    assert payload.extensions["conventional_eeg"]["packs"] == ["coma"]
    assert "coma" in payload.extensions["conventional_eeg"]["families"]
    clinical_markers = payload.extensions["conventional_eeg"]["clinical_markers"]["markers"]
    for marker in ("ssep", "nse", "gcs", "s100b"):
        assert clinical_markers[marker]["status"] == "unavailable"
        assert isinstance(clinical_markers[marker]["reason"], str)
        assert clinical_markers[marker]["reason"]
    assert "conventional_eeg" in manifest
    assert "clinical_markers" in manifest["conventional_eeg"]
    assert "suppression_ratio" in manifest["conventional_eeg"]["families"]["coma"]
    assert (
        manifest["conventional_eeg"]["families"]["coma"]["reactivity_proxy"]["column"]
        == "eeg_conventional_coma_reactivity_proxy"
    )


def test_subject_runner_exports_dual_anchor_jacobian_and_regional_layers(dummy_ctx, monkeypatch, tmp_path):
    """Anchored runs should keep both subject/cohort Jacobian and regional layers."""
    dummy_ctx.config = {
        "robustness": {"coverage": {}},
        "mnps_projection": {
            "anchor": {
                "enabled": True,
                "path": str(tmp_path / "anchor.json"),
                "scale_method": "iqr",
                "min_subjects": 3,
            }
        },
        "regional_mnps": {"enabled": True},
    }
    runner = DatasetSummaryRunner(dummy_ctx, "ds001", None, "subject", n_jobs=1)
    runner.participants_df = pd.DataFrame()
    monkeypatch.setattr(runner, "participant_meta_for", lambda _sub_id: {})
    monkeypatch.setattr(runner, "participant_meta_source_info", lambda: {})
    monkeypatch.setattr(
        runner,
        "resolve_coverage_policy",
        lambda **_kwargs: {"min_epochs": 0, "min_seconds": 0.0, "tag": "default"},
    )
    monkeypatch.setattr(runner, "write_regional_csv_outputs_threadsafe", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "write_stratified_blocks_csv_output_threadsafe", lambda **_kwargs: None)

    subject_runner = SubjectSummaryRunner(
        dataset_runner=runner,
        ds_path=tmp_path,
        mnps_dir=tmp_path / "mnps",
        index_df=None,
    )
    subject_runner.mnps_dir.mkdir(parents=True, exist_ok=True)
    anchor_path = tmp_path / "anchor.json"
    anchor_path.write_text("{}", encoding="utf-8")

    sub_frame = pd.DataFrame(
        {
            "file": ["sub-001_task-rest_eeg.set"] * 6,
            "epoch_id": np.arange(6, dtype=int),
            "t_start": np.arange(6, dtype=float),
            "t_end": np.arange(1, 7, dtype=float),
            "feat_m": np.linspace(0.1, 0.6, 6),
            "feat_d": np.linspace(1.1, 1.6, 6),
            "feat_e": np.linspace(2.1, 2.6, 6),
        }
    )
    x_subject = np.array(
        [
            [0.0, 0.1, 0.2],
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5],
            [0.4, 0.5, 0.6],
            [0.5, 0.6, 0.7],
        ],
        dtype=np.float32,
    )
    x_cohort = x_subject + 10.0
    coords_subject = np.arange(54, dtype=np.float32).reshape(6, 9)
    coords_cohort = coords_subject + 100.0
    captures: dict[str, object] = {}
    ordered_names = ["m_a", "m_e", "m_o", "d_n", "d_l", "d_s", "e_e", "e_s", "e_m"]

    monkeypatch.setattr(summary_mod, "extract_mapped_metadata", lambda *_args, **_kwargs: {"group": None, "condition": "rest", "task": "rest"})
    monkeypatch.setattr(summary_mod, "build_dataset_label", lambda **_kwargs: "ds001:sub-001:rest_rest")
    monkeypatch.setattr(
        summary_mod.anchors,
        "load_anchor_file",
        lambda _path: {"spec": {"anchor_id": "unit-test", "anchor_hash": "hash", "anchor_source": "fixture"}},
    )
    monkeypatch.setattr(
        summary_mod.anchors,
        "anchor_mapping",
        lambda *_args, **_kwargs: {"feat_m": {"center": 0.0, "scale": 1.0}},
    )

    def _fake_project_features_with_coverage(*_args, **kwargs):
        external_anchor = kwargs.get("external_anchor")
        data = x_cohort if external_anchor is not None else x_subject
        return data, np.ones_like(data, dtype=np.float32), {}

    def _fake_project_features_v2(*_args, **kwargs):
        external_anchor = kwargs.get("external_anchor")
        data = coords_cohort if external_anchor is not None else coords_subject
        return data, list(ordered_names), {}

    def _fake_estimate_local_jacobians(x, *_args, **_kwargs):
        fill = float(np.asarray(x, dtype=np.float32)[0, 0])
        dim = int(np.asarray(x).shape[1])
        return SimpleNamespace(
            j_hat=np.full((3, dim, dim), fill, dtype=np.float32),
            j_dot=np.full((3, dim, dim), fill + 0.5, dtype=np.float32),
            centers=np.array([1, 2, 3], dtype=np.int32),
            diagnostics={},
        )

    def _fake_compute_regional_context(**kwargs):
        contract = "cohort_anchored" if kwargs.get("external_anchor") is not None else "subject_anchored"
        offset = 5.0 if contract == "cohort_anchored" else 0.0
        result = SimpleNamespace(
            mnps=np.full((6, 3), offset + 1.0, dtype=np.float32),
            mnps_dot=np.full((6, 3), offset + 2.0, dtype=np.float32),
            jacobian=np.full((3, 3, 3), offset + 3.0, dtype=np.float32),
            stratified=np.full((6, 9), offset + 4.0, dtype=np.float32),
            metrics={"m_mean": offset + 0.1},
            n_timepoints=6,
        )
        summary = SimpleNamespace(results={"DMN": result}, n_networks=1, n_dropped=0)
        return {}, None, [], {}, summary

    monkeypatch.setattr(summary_mod.projection, "project_features_with_coverage", _fake_project_features_with_coverage)
    monkeypatch.setattr(summary_mod.projection, "project_features_v2", _fake_project_features_v2)
    monkeypatch.setattr(summary_mod.projection, "build_knn_indices", lambda arr, **_kwargs: np.zeros((len(arr), 2), dtype=np.int32))
    monkeypatch.setattr(summary_mod.jacobian, "estimate_local_jacobians", _fake_estimate_local_jacobians)
    monkeypatch.setattr(
        summary_mod,
        "_resolve_mnps_9d_runtime_config",
        lambda *_args, **_kwargs: (
            True,
            "2.0",
            {"jacobian": {"enabled": True}},
            {
                "m_a": {"feat_m": 1.0},
                "m_e": {"feat_m": 1.0},
                "m_o": {"feat_m": 1.0},
                "d_n": {"feat_d": 1.0},
                "d_l": {"feat_d": 1.0},
                "d_s": {"feat_d": 1.0},
                "e_e": {"eeg_permutation_entropy": 1.0},
                "e_s": {"feat_e": 1.0},
                "e_m": {"feat_e": 1.0},
            },
        ),
    )
    monkeypatch.setattr(summary_mod, "_validate_e_e_subcoord_construct", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        summary_mod.projection,
        "build_feature_export_bundle",
        lambda *args, **kwargs: {
            "raw_values": np.zeros((len(sub_frame), 0), dtype=np.float32),
            "raw_names": [],
            "robust_z_values": np.zeros((len(sub_frame), 0), dtype=np.float32),
            "robust_z_names": [],
            "metadata": {},
        },
    )
    monkeypatch.setattr(summary_mod, "extract_stage_array", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(subject_runner, "_infer_stage_from_bids_events", lambda *_args, **_kwargs: (None, None, None, None))
    monkeypatch.setattr(summary_mod, "extract_embodied_array", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(summary_mod, "extract_events", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(subject_runner, "_load_regional_fmri", lambda **_kwargs: (None, None, None))
    monkeypatch.setattr(summary_mod, "compute_regional_context", _fake_compute_regional_context)
    monkeypatch.setattr(summary_mod, "summary_to_dataframe_rows", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(summary_mod, "compute_block_jacobian_rows", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(summary_mod, "compute_stratified_blocks_and_cross_partials", lambda **_kwargs: None)
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
    assert "coords_3d_subject_anchored" in payload.coordinate_layers
    assert "coords_3d_cohort_anchored" in payload.coordinate_layers
    assert "jacobian_subject_anchored" in payload.jacobian_layers
    assert "jacobian_cohort_anchored" in payload.jacobian_layers
    assert "jacobian_9D_subject_anchored" in payload.jacobian_layers
    assert "jacobian_9D_cohort_anchored" in payload.jacobian_layers
    assert "DMN" in payload.regional_mnps
    assert payload.regional_mnps["DMN"]["primary_coordinate_contract"] == "cohort_anchored"
    assert "subject_anchored" in payload.regional_mnps["DMN"]["anchor_layers"]
    assert "cohort_anchored" in payload.regional_mnps["DMN"]["anchor_layers"]
    assert payload.provenance["anchoring"]["available_jacobian_layers"] == [
        "jacobian_9D_cohort_anchored",
        "jacobian_9D_subject_anchored",
        "jacobian_cohort_anchored",
        "jacobian_subject_anchored",
    ]
    assert manifest["regional_outputs_h5"]["primary_coordinate_contract"] == "cohort_anchored"
    assert sorted(manifest["jacobian_h5"]["layer_paths"]) == [
        "/jacobian_9D_cohort_anchored",
        "/jacobian_9D_subject_anchored",
        "/jacobian_cohort_anchored",
        "/jacobian_subject_anchored",
    ]


def test_subject_runner_exports_mnps_mnj_sanity_to_manifest_and_qc(dummy_ctx, monkeypatch, tmp_path):
    """MNPS/MNJ sanity output should be exported to manifest and QC write path."""
    dummy_ctx.config = {
        "robustness": {
            "coverage": {},
            "review_qc": {
                "mnps_mnj_sanity": {
                    "enabled": True,
                    "robustified_variant": {"enabled": True},
                }
            },
        }
    }
    runner = DatasetSummaryRunner(dummy_ctx, "ds001", None, "subject", n_jobs=1)
    runner.participants_df = pd.DataFrame()
    monkeypatch.setattr(runner, "participant_meta_for", lambda _sub_id: {})
    monkeypatch.setattr(runner, "participant_meta_source_info", lambda: {})
    monkeypatch.setattr(
        runner,
        "resolve_coverage_policy",
        lambda **_kwargs: {"min_epochs": 0, "min_seconds": 0.0, "tag": "default"},
    )
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
            "t_start": np.arange(0, 18, 3, dtype=float),
            "t_end": np.arange(4, 22, 3, dtype=float),
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
    monkeypatch.setattr(
        summary_mod.projection,
        "project_features_with_coverage",
        lambda *args, **kwargs: (x, np.ones_like(x, dtype=np.float32), {}),
    )
    monkeypatch.setattr(
        summary_mod.projection,
        "build_feature_export_bundle",
        lambda *args, **kwargs: {
            "raw_values": np.zeros((len(sub_frame), 0), dtype=np.float32),
            "raw_names": [],
            "robust_z_values": np.zeros((len(sub_frame), 0), dtype=np.float32),
            "robust_z_names": [],
            "metadata": {},
        },
    )
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
    monkeypatch.setattr(
        summary_mod,
        "compute_mnps_mnj_sanity",
        lambda **_kwargs: {
            "status": "warning",
            "degeneracy_flags": {"coords_9d_has_degenerate_subcoord": True},
            "robustified_comparison": {"enabled": True},
        },
    )
    monkeypatch.setattr(summary_mod, "compute_emmi_metrics", lambda **_kwargs: None)
    monkeypatch.setattr(summary_mod, "compute_null_sanity_tests", lambda **_kwargs: None)
    monkeypatch.setattr(summary_mod, "compute_psd_multiverse_stability", lambda **_kwargs: None)
    monkeypatch.setattr(summary_mod.robustness, "entropy_sanity_checks", lambda *args, **kwargs: {})
    monkeypatch.setattr(summary_mod, "_get_env_provenance", lambda: {})

    def _capture_qc_write(**kwargs):
        captures["qc_kwargs"] = kwargs

    monkeypatch.setattr(subject_runner, "_write_qc_files", _capture_qc_write)

    def _capture_write(*, target_dir, dataset_label, manifest, payload, **kwargs):
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

    manifest = captures["manifest"]
    qc_kwargs = captures["qc_kwargs"]
    assert manifest["geometry_contract"]["policy_version"] == "standard_invalidity_v1"
    assert qc_kwargs["geometry_contract"]["policy_version"] == "standard_invalidity_v1"
    assert manifest["mnps_mnj_sanity"]["status"] == "warning"
    assert manifest["mnps_mnj_sanity"]["robustified_comparison"]["enabled"] is True
    assert qc_kwargs["mnps_mnj_sanity"]["status"] == "warning"
    assert qc_kwargs["mnps_mnj_sanity"]["degeneracy_flags"]["coords_9d_has_degenerate_subcoord"] is True


def test_subject_runner_drops_standard_invalid_geometry_before_knn(dummy_ctx, monkeypatch, tmp_path):
    """Always-on geometry policy should drop invalid rows before kNN/Jacobian."""
    dummy_ctx.config = {
        "robustness": {"coverage": {}},
        "mnps_projection": {"missing_axis_policy": "off"},
    }
    runner = DatasetSummaryRunner(dummy_ctx, "ds001", None, "subject", n_jobs=1)
    runner.participants_df = pd.DataFrame()
    monkeypatch.setattr(runner, "participant_meta_for", lambda _sub_id: {})
    monkeypatch.setattr(runner, "participant_meta_source_info", lambda: {})
    monkeypatch.setattr(
        runner,
        "resolve_coverage_policy",
        lambda **_kwargs: {"min_epochs": 0, "min_seconds": 0.0, "tag": "default"},
    )
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
            "t_start": np.arange(0, 18, 3, dtype=float),
            "t_end": np.arange(4, 22, 3, dtype=float),
        }
    )
    x = np.array(
        [
            [0.0, 0.1, 0.2],
            [0.2, 0.0, 0.1],
            [np.nan, 0.3, 0.4],
            [0.6, -0.2, -0.1],
            [0.8, -0.1, -0.2],
            [1.0, 0.0, -0.3],
        ],
        dtype=np.float32,
    )
    captures: dict[str, object] = {}

    monkeypatch.setattr(summary_mod, "extract_mapped_metadata", lambda *_args, **_kwargs: {"group": None, "condition": "rest", "task": "rest"})
    monkeypatch.setattr(summary_mod, "build_dataset_label", lambda **_kwargs: "ds001:sub-001:rest_rest")
    monkeypatch.setattr(
        summary_mod.projection,
        "project_features_with_coverage",
        lambda *args, **kwargs: (x, np.ones_like(x, dtype=np.float32), {}),
    )
    monkeypatch.setattr(
        summary_mod.projection,
        "build_feature_export_bundle",
        lambda frame, *args, **kwargs: {
            "raw_values": np.zeros((len(frame), 0), dtype=np.float32),
            "raw_names": [],
            "robust_z_values": np.zeros((len(frame), 0), dtype=np.float32),
            "robust_z_names": [],
            "metadata": {},
        },
    )
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

    def _capture_knn(arr, **_kwargs):
        captures["knn_input"] = np.asarray(arr, dtype=np.float32)
        n = len(arr)
        return np.tile(np.arange(n, dtype=np.int32), (n, 1))

    monkeypatch.setattr(summary_mod.projection, "build_knn_indices", _capture_knn)
    monkeypatch.setattr(
        summary_mod.jacobian,
        "estimate_local_jacobians",
        lambda x, *_args, **_kwargs: SimpleNamespace(
            j_hat=np.repeat(np.eye(x.shape[1], dtype=np.float32)[None, :, :], max(0, len(x) - 2), axis=0),
            j_dot=np.repeat(np.eye(x.shape[1], dtype=np.float32)[None, :, :], max(0, len(x) - 2), axis=0),
            centers=np.arange(max(0, len(x) - 2), dtype=np.int32),
            diagnostics={"condition_number_windows": np.ones((max(0, len(x) - 2),), dtype=np.float32)},
        ),
    )

    def _capture_qc_write(**kwargs):
        captures["qc_kwargs"] = kwargs

    monkeypatch.setattr(subject_runner, "_write_qc_files", _capture_qc_write)

    def _capture_write(*, target_dir, dataset_label, manifest, payload, **kwargs):
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

    knn_input = captures["knn_input"]
    manifest = captures["manifest"]
    payload = captures["payload"]
    qc_kwargs = captures["qc_kwargs"]

    assert knn_input.shape[0] == 5
    assert np.isfinite(knn_input).all()
    assert manifest["geometry_contract"]["shared_time_grid"]["epochs_dropped"] == 1
    assert manifest["geometry_contract"]["time_grid"]["status"] == "ok"
    assert np.isclose(manifest["geometry_contract"]["time_grid"]["dt_intervals_sec"]["median"], 3.0)
    assert np.isclose(manifest["geometry_contract"]["time_grid"]["window_lengths_sec"]["median"], 4.0)
    assert payload.attrs["dropped_geometry_invalid_epochs"] == 1
    assert qc_kwargs["geometry_contract"]["status"] == "adjusted"

