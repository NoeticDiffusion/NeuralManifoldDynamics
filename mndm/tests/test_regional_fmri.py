"""Tests for regional fMRI integration in MNPS summarization."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("mne")

from mndm.pipeline import summary as summary_mod
from mndm.pipeline.summary import DatasetSummaryRunner, SubjectSummaryRunner
from mndm.pipeline.regions import group_region_indices


def _build_ctx(tmp_path):
    received = tmp_path / "received_data"
    processed = tmp_path / "processed_data"
    ctx = SimpleNamespace(
        config={
            "paths": {
                "received_dir": str(received),
                "processed_dir": str(processed),
            },
            "robustness": {"coverage": {}},
        },
        received_dir=received,
        processed_dir=processed,
        coverage=SimpleNamespace(min_seconds=0.0, min_epochs=0),
        weights={
            "m": {"feat_m": 1.0},
            "d": {"feat_d": 1.0},
            "e": {"feat_e": 1.0},
        },
        normalize_override=None,
        ingest_meta={},
        mnps_cfg={
            "window_sec": 4.0,
            "overlap": 0.25,
            "fs_out": 4.0,
            "derivative": {"method": "central", "window": 5, "polyorder": 2},
            "knn_k": 3,
            "knn_metric": "euclidean",
            "ridge_alpha": 1.0,
            "super_window": 3,
            "stage_codebook": {},
            "embodied": {"enabled": False},
            "surrogates": {},
            "reliability": {},
            "whiten": True,
        },
        extensions_cfg={
            "e_kappa": {"enabled": True, "fmri": {"enabled": True, "regional": True, "min_regions": 1}},
        },
        derivative_cfg={"method": "central", "window": 5, "polyorder": 2},
    )
    return ctx


def test_subject_runner_ingests_fmri_regions(monkeypatch, tmp_path):
    ctx = _build_ctx(tmp_path)
    ds_id = "ds001"
    received_ds = ctx.received_dir / ds_id
    processed_ds = ctx.processed_dir / ds_id
    received_ds.mkdir(parents=True, exist_ok=True)
    processed_ds.mkdir(parents=True, exist_ok=True)

    # Minimal participants + index files
    (received_ds / "participants.tsv").write_text("participant_id\tgroup\n001\tcontrol\n", encoding="utf-8")
    index_df = pd.DataFrame(
        [
            {
                "path": "sub-001_task-rest_bold.nii.gz",
                "modality": "fmri",
                "subject": "001",
            }
        ]
    )
    index_df.to_csv(processed_ds / "file_index.csv", index=False)

    bold_path = received_ds / "sub-001_task-rest_bold.nii.gz"
    bold_path.parent.mkdir(parents=True, exist_ok=True)
    bold_path.write_bytes(b"")

    features_df = pd.DataFrame(
        {
            "file": ["sub-001_ses-01_task-rest_eeg.set"] * 8,
            "qc_ok_eeg": [1] * 8,
            "feat_m": np.linspace(0, 1, 8),
            "feat_d": np.linspace(1, 2, 8),
            "feat_e": np.linspace(2, 3, 8),
        }
    )
    features_df.to_csv(processed_ds / "features.csv", index=False)

    fmri_data = np.vstack(
        [
            np.linspace(0.0, 1.0, 8),
            np.linspace(1.0, 0.0, 8),
        ]
    ).astype(np.float32)

    def fake_preprocess_fmri(path, config):
        assert Path(path) == bold_path
        return SimpleNamespace(
            signals={"fmri": fmri_data},
            channels={"fmri": ["VIS_ROI_A", "DMN_ROI_B"]},
            sfreq=0.5,
        )

    monkeypatch.setattr(summary_mod.preprocess, "preprocess_fmri", fake_preprocess_fmri)

    captured = {}

    monkeypatch.setattr(
        summary_mod,
        "write_summary_manifest_and_h5",
        lambda **kwargs: captured.setdefault("payload", kwargs["payload"]),
    )
    monkeypatch.setattr(summary_mod.json_writer, "build_manifest", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(summary_mod.json_writer, "write_json_summary", lambda *_, **__: None)

    dataset_runner = DatasetSummaryRunner(ctx, ds_id, None, "subject")
    dataset_runner.participants_df = pd.DataFrame({"participant_id": ["sub-001"], "group": ["control"]})
    dataset_runner.min_seconds = 0
    dataset_runner.min_epochs = 0
    subject_runner = SubjectSummaryRunner(
        dataset_runner=dataset_runner,
        ds_path=processed_ds,
        mnps_dir=processed_ds,
        index_df=index_df,
    )
    subject_runner.run(
        sub_id="sub-001",
        ses_id=None,
        raw_task=None,
        run_id=None,
        acq_id=None,
        sub_frame=features_df,
    )

    payload = captured.get("payload")
    assert payload is not None
    assert payload.regions_bold.shape == fmri_data.shape
    assert payload.regions_names == ["VIS_ROI_A", "DMN_ROI_B"]
    assert payload.regions_sfreq == pytest.approx(0.5)

    ek_regional = payload.extensions["e_kappa_fmri"]["regional"]
    assert "VIS" in ek_regional and ek_regional["VIS"]["n_regions"] == 1
    assert "DMN" in ek_regional and ek_regional["DMN"]["n_regions"] == 1


def test_group_region_indices_aliases():
    groups = group_region_indices(["Vis_left", "default_mode_1", "SAL_ROI"])
    assert "VIS" in groups and groups["VIS"] == [0]
    assert "DMN" in groups and groups["DMN"] == [1]
    assert "SAL" in groups and groups["SAL"] == [2]


def test_merge_fd_from_confounds_per_epoch(tmp_path):
    ctx = _build_ctx(tmp_path)
    ds_id = "ds001"
    received_ds = ctx.received_dir / ds_id
    processed_ds = ctx.processed_dir / ds_id
    received_ds.mkdir(parents=True, exist_ok=True)
    processed_ds.mkdir(parents=True, exist_ok=True)

    bold_rel = Path("sub-001/func/sub-001_task-rest_bold.nii.gz")
    bold_path = received_ds / bold_rel
    bold_path.parent.mkdir(parents=True, exist_ok=True)
    bold_path.write_bytes(b"")

    conf_path = received_ds / "derivatives" / "fmriprep" / "sub-001" / "func" / "sub-001_task-rest_desc-confounds_timeseries.tsv"
    conf_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"framewise_displacement": [0.0, 0.2, 0.7, 0.1, 0.6, 0.3]}).to_csv(
        conf_path, sep="\t", index=False
    )

    index_df = pd.DataFrame(
        [
            {
                "path": str(bold_rel).replace("\\", "/"),
                "modality": "fmri",
                "subject": "001",
            }
        ]
    )
    index_df.to_csv(processed_ds / "file_index.csv", index=False)

    dataset_runner = DatasetSummaryRunner(ctx, ds_id, None, "subject")
    subject_runner = SubjectSummaryRunner(
        dataset_runner=dataset_runner,
        ds_path=processed_ds,
        mnps_dir=processed_ds,
        index_df=index_df,
    )

    sub_frame = pd.DataFrame(
        {
            "file": ["sub-001_task-rest_bold.nii.gz"] * 3,
            "epoch_id": [0, 1, 2],
            "t_start": [0.0, 2.0, 4.0],
            "t_end": [2.0, 4.0, 6.0],
            "fmri_sfreq": [1.0, 1.0, 1.0],
        }
    )
    merged = subject_runner._merge_fd_from_confounds(
        sub_frame=sub_frame,
        raw_task="rest",
        condition=None,
        session=None,
        run_id=None,
        acq_id=None,
    )

    assert "framewise_displacement" in merged.columns
    vals = merged["framewise_displacement"].to_numpy(dtype=float)
    assert vals[0] == pytest.approx(0.2)
    assert vals[1] == pytest.approx(0.7)
    assert vals[2] == pytest.approx(0.6)

