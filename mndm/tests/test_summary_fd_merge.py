"""Tests for fMRI confounds -> FD merge in summary pipeline."""

from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("scipy")

from mndm.pipeline.summary import DatasetSummaryRunner, SubjectSummaryRunner


def _build_ctx(tmp_path):
    """Internal helper: build ctx."""
    received = tmp_path / "received_data"
    processed = tmp_path / "processed_data"
    return SimpleNamespace(
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
        weights={"m": {"feat_m": 1.0}, "d": {"feat_d": 1.0}, "e": {"feat_e": 1.0}},
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
        extensions_cfg={},
        derivative_cfg={"method": "central", "window": 5, "polyorder": 2},
    )


def test_merge_fd_from_confounds_per_epoch(tmp_path):
    """Test merge fd from confounds per epoch."""
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


def test_merge_fd_uses_configured_derivatives_dir(tmp_path):
    """Test merge fd uses configured derivatives dir."""
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

    # Custom derivatives root outside dataset_root/derivatives.
    custom_root = tmp_path / "external_derivatives"
    conf_path = custom_root / "sub-001" / "func" / "sub-001_task-rest_desc-confounds_timeseries.tsv"
    conf_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"framewise_displacement": [0.4, 0.1, 0.5, 0.2]}).to_csv(
        conf_path, sep="\t", index=False
    )
    ctx.config["preprocess"] = {
        "fmri": {
            "datasets": {
                ds_id: {
                    "confounds": {
                        "derivatives_dir": str(custom_root),
                    }
                }
            }
        }
    }

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
            "file": ["sub-001_task-rest_bold.nii.gz"] * 2,
            "epoch_id": [0, 1],
            "t_start": [0.0, 2.0],
            "t_end": [2.0, 4.0],
            "fmri_sfreq": [1.0, 1.0],
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
    vals = merged["framewise_displacement"].to_numpy(dtype=float)
    assert vals[0] == pytest.approx(0.4)
    assert vals[1] == pytest.approx(0.5)
