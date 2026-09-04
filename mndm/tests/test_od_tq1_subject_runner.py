"""Thin SubjectSummaryRunner replay for OD-TQ1 pipeline semantics."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mndm.pipeline.summary as summary_mod
from mndm.pipeline.summary import DatasetSummaryRunner, SubjectSummaryRunner


def _context(tmp_path: Path) -> SimpleNamespace:
    """Create the smallest runner context with OD-TQ1 explicitly enabled."""
    return SimpleNamespace(
        config={
            "modality": "eeg",
            "robustness": {"coverage": {"min_epochs": 0, "min_seconds": 0.0}},
            "mnps": {"jacobian": {"enabled": False}},
            "dynamical_families": {
                "enabled": True,
                "coordinate_layer": "subject_anchored",
                "diffusion": {
                    "enabled": True,
                    "neighborhood": {"k": 20},
                    "min_samples": 10,
                    "min_neighborhood_samples": 10,
                    "translation_qualification": {
                        "qualified": True,
                        "qualification_id": "OD-TQ1-subject-runner-fixture",
                        "qualification_contract_hash": "subject-runner-fixture-hash",
                    },
                },
            },
        },
        received_dir=tmp_path,
        processed_dir=tmp_path,
        coverage=SimpleNamespace(min_seconds=0.0, min_epochs=0),
        weights={"m": {}, "d": {}, "e": {}},
        normalize_override=None,
        ingest_meta={},
        reproducibility={"seed": 42, "seed_source": "test"},
        mnps_cfg={
            "window_sec": 1.0,
            "overlap": 0.0,
            "fs_out": 1.0,
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


def test_od_tq1_subject_runner_masks_qc_and_segments_by_file(tmp_path: Path, monkeypatch) -> None:
    """Production runner masking must precede file-segmented OD estimation."""
    ctx = _context(tmp_path)
    runner = DatasetSummaryRunner(ctx, "ds-tq1", None, "subject", n_jobs=1)
    runner.participants_df = pd.DataFrame()
    monkeypatch.setattr(runner, "participant_meta_for", lambda _sub_id: {})
    monkeypatch.setattr(runner, "participant_meta_source_info", lambda: {})
    monkeypatch.setattr(
        runner,
        "resolve_coverage_policy",
        lambda **_kwargs: {"min_epochs": 0, "min_seconds": 0.0, "tag": "fixture"},
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

    n_rows = 24
    rng = np.random.default_rng(905)
    state = np.vstack(
        [
            np.zeros((1, 3)),
            np.cumsum(rng.normal(0.0, 0.3 * np.sqrt(1.0), (n_rows - 1, 3)), axis=0),
        ]
    ).astype(np.float32)
    sub_frame = pd.DataFrame(
        {
            "file": ["first.set"] * 12 + ["second.set"] * 12,
            "epoch_id": np.arange(n_rows, dtype=np.int64),
            "t_start": np.arange(n_rows, dtype=np.float64),
            "t_end": np.arange(n_rows, dtype=np.float64) + 0.5,
            "ecg_hr_bpm": np.linspace(60.0, 70.0, n_rows),
            "ecg_rmssd": np.linspace(40.0, 30.0, n_rows),
        }
    )
    coverage = np.ones_like(state, dtype=np.float32)
    coverage[12] = 0.0
    captures: dict[str, object] = {}

    monkeypatch.setattr(
        summary_mod,
        "extract_mapped_metadata",
        lambda *_args, **_kwargs: {"group": None, "condition": "rest", "task": "rest"},
    )
    monkeypatch.setattr(summary_mod, "build_dataset_label", lambda **_kwargs: "ds-tq1:sub-001")
    monkeypatch.setattr(
        summary_mod.projection,
        "project_features_with_coverage",
        lambda *_args, **_kwargs: (state.copy(), coverage.copy(), {}),
    )

    def _feature_bundle(*args, **_kwargs):
        frame = next((arg for arg in args if isinstance(arg, pd.DataFrame)), sub_frame)
        values = frame[["ecg_hr_bpm", "ecg_rmssd"]].to_numpy(dtype=np.float32)
        return {
            "raw_values": values,
            "raw_names": ["ecg_hr_bpm", "ecg_rmssd"],
            "robust_z_values": np.zeros_like(values),
            "robust_z_names": ["ecg_hr_bpm", "ecg_rmssd"],
            "projection_z_values": np.zeros_like(values),
            "projection_z_names": ["ecg_hr_bpm", "ecg_rmssd"],
            "metadata": {
                "robust_z_valid": np.ones(2, dtype=np.int8),
                "robust_z_invalid_reason": np.asarray(["", ""], dtype=object),
            },
        }

    monkeypatch.setattr(summary_mod.projection, "build_feature_export_bundle", _feature_bundle)
    monkeypatch.setattr(summary_mod, "extract_stage_array", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        subject_runner,
        "_infer_stage_from_bids_events",
        lambda *_args, **_kwargs: (None, None, None, None),
    )
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

    original_write = summary_mod.write_summary_manifest_and_h5

    def _capture_write(**kwargs):
        captures.update(kwargs)
        return original_write(**kwargs)

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
    assert payload.dynamical_families["diffusion"]["computation_status"] == "computed"
    diffusion = payload.dynamical_families["diffusion"]
    assert diffusion["measurement_validity"] == "not_assessed"
    assert diffusion["summary"]["n_timepoints"] == n_rows - 1
    assert diffusion["summary"]["n_increment_pairs"] == 21
    assert diffusion["provenance"]["qualification_id"] == "OD-TQ1-subject-runner-fixture"

    h5_paths = sorted(Path(captures["target_dir"]).glob("*.h5"))
    assert len(h5_paths) == 1
    with h5py.File(h5_paths[0], "r") as handle:
        family = handle["dynamical_families/diffusion/v1"]
        assert family["computation_status"][()].decode() == "computed"
        assert family["measurement_validity"][()].decode() == "not_assessed"
        assert family["summary/n_timepoints"][()] == n_rows - 1
        assert family["summary/n_increment_pairs"][()] == 21
        assert family["provenance/qualification_id"][()].decode() == "OD-TQ1-subject-runner-fixture"
