from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("mne")

import mndm.pipeline.summary as summary_mod
from mndm.pipeline.state_labels import build_within_run_labels
from mndm.pipeline.summary import DatasetSummaryRunner, SubjectSummaryRunner


def _build_ctx(tmp_path: Path, config: dict) -> SimpleNamespace:
    received = tmp_path / "received"
    processed = tmp_path / "processed"
    return SimpleNamespace(
        config=config,
        received_dir=received,
        processed_dir=processed,
        coverage=SimpleNamespace(min_seconds=0.0, min_epochs=0),
        weights={"m": {"feat_m": 1.0}, "d": {"feat_d": 1.0}, "e": {"feat_e": 1.0}},
        normalize_override=None,
        ingest_meta={},
        mnps_cfg={
            "window_sec": 8.0,
            "overlap": 0.5,
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
        derivative_cfg={"method": "central", "window": 5, "polyorder": 2},
        extensions_cfg={},
        reproducibility={"seed": 42, "seed_source": "default"},
    )


def test_summary_runner_applies_ds006623_within_run_labels(monkeypatch, tmp_path: Path):
    """ds006623 boundary-table rules should populate stage and aligned labels."""
    timing_csv = tmp_path / "LOR_ROR_Timing.csv"
    timing_csv.write_text(
        "Subject,LOR time (TR in task2),ROR time (TR in task3)\n"
        "sub-003,3,5\n",
        encoding="utf-8",
    )
    config = {
        "paths": {"received_dir": str(tmp_path / "received"), "processed_dir": str(tmp_path / "processed")},
        "robustness": {"coverage": {}},
        "within_run_labels": {
            "datasets": {
                "ds006623": {
                    "enabled": True,
                    "output_name": "within_run_state_v1",
                    "write_to_stage": True,
                    "write_to_labels": True,
                    "codebook_name": "within_run_state_v1",
                    "rules": [
                        {
                            "id": "task2",
                            "match": {"task": "imagery", "run": "run-2"},
                            "source": {
                                "type": "boundary_table",
                                "path": str(timing_csv),
                                "subject_column": "Subject",
                                "boundaries": {
                                    "lor": {
                                        "column": "LOR time (TR in task2)",
                                        "units": "tr_index",
                                        "index_origin": 1,
                                    }
                                },
                            },
                            "segments": [
                                {"label": "pre_lor", "end": "lor", "on_missing_boundary": "expand"},
                                {"label": "unresponsive", "start": "lor", "on_missing_boundary": "skip"},
                            ],
                        }
                    ],
                }
            }
        },
    }
    ctx = _build_ctx(tmp_path, config)
    ds_id = "ds006623"

    received_ds = ctx.received_dir / ds_id / "sub-003" / "func"
    received_ds.mkdir(parents=True, exist_ok=True)
    bold_name = "sub-003_task-imagery_run-2_bold.nii.gz"
    (received_ds / bold_name).write_bytes(b"")

    index_df = pd.DataFrame([{"path": str(Path("sub-003/func") / bold_name), "modality": "fmri", "subject": "003"}])
    features_df = pd.DataFrame(
        {
            "file": [bold_name] * 4,
            "t_start": [0.0, 1.0, 2.0, 3.0],
            "t_end": [1.0, 2.0, 3.0, 4.0],
            "fmri_sfreq": [1.0] * 4,
            "feat_m": np.linspace(0.0, 1.0, 4),
            "feat_d": np.linspace(1.0, 2.0, 4),
            "feat_e": np.linspace(2.0, 3.0, 4),
        }
    )

    captured = {}
    monkeypatch.setattr(
        summary_mod,
        "write_summary_manifest_and_h5",
        lambda **kwargs: captured.setdefault("payload", kwargs["payload"]),
    )
    monkeypatch.setattr(summary_mod.json_writer, "build_manifest", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(summary_mod.json_writer, "write_json_summary", lambda *_, **__: None)

    dataset_runner = DatasetSummaryRunner(ctx, ds_id, None, "subject")
    dataset_runner.participants_df = None
    dataset_runner.min_seconds = 0
    dataset_runner.min_epochs = 0

    subject_runner = SubjectSummaryRunner(
        dataset_runner=dataset_runner,
        ds_path=ctx.processed_dir / ds_id,
        mnps_dir=ctx.processed_dir / ds_id,
        index_df=index_df,
    )
    monkeypatch.setattr(subject_runner, "_load_regional_fmri", lambda **_kwargs: (None, None, None))
    subject_runner.run(
        sub_id="sub-003",
        ses_id=None,
        raw_task="imagery",
        run_id="run-2",
        acq_id=None,
        sub_frame=features_df,
    )

    payload = captured["payload"]
    assert payload.stage is not None
    assert payload.attrs["stage_source"] == "within_run_labels"
    assert payload.attrs["stage_codebook"]["pre_lor"] == 1
    assert payload.labels["within_run_state_v1"].tolist() == [
        "pre_lor",
        "pre_lor",
        "unresponsive",
        "unresponsive",
    ]
    assert payload.stage.tolist() == [1, 1, 2, 2]


def test_boundary_table_na_ror_expands_unresponsive_to_run_end(tmp_path: Path):
    """Missing boundary values should support 'expand' for open-ended segments."""
    timing_csv = tmp_path / "LOR_ROR_Timing.csv"
    timing_csv.write_text(
        "Subject,LOR time (TR in task2),ROR time (TR in task3)\n"
        "sub-028,1385,N/A\n",
        encoding="utf-8",
    )
    config = {
        "within_run_labels": {
            "datasets": {
                "ds006623": {
                    "enabled": True,
                    "output_name": "within_run_state_v1",
                    "write_to_stage": True,
                    "write_to_labels": True,
                    "codebook_name": "within_run_state_v1",
                    "rules": [
                        {
                            "id": "task3",
                            "match": {"task": "imagery", "run": "run-3"},
                            "source": {
                                "type": "boundary_table",
                                "path": str(timing_csv),
                                "subject_column": "Subject",
                                "boundaries": {
                                    "ror": {
                                        "column": "ROR time (TR in task3)",
                                        "units": "tr_index",
                                        "index_origin": 1,
                                        "na_values": ["N/A"],
                                    }
                                },
                            },
                            "segments": [
                                {"label": "unresponsive", "end": "ror", "on_missing_boundary": "expand"},
                                {"label": "post_ror", "start": "ror", "on_missing_boundary": "skip"},
                            ],
                        }
                    ],
                }
            }
        }
    }
    result = build_within_run_labels(
        config=config,
        dataset_id="ds006623",
        dataset_root=tmp_path,
        sub_id="sub-028",
        ses_id=None,
        task="imagery",
        raw_task="imagery",
        run_id="run-3",
        acq_id=None,
        tr_sec=1.0,
        time=np.arange(4, dtype=float),
        window_start=np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
        window_end=np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
        sub_frame=pd.DataFrame({"dummy": [1, 2, 3, 4]}),
    )

    assert result.labels["within_run_state_v1"].tolist() == [
        "unresponsive",
        "unresponsive",
        "unresponsive",
        "unresponsive",
    ]
    assert result.stage.tolist() == [2, 2, 2, 2]


def test_interval_table_supports_sleep_like_within_run_stages(tmp_path: Path):
    """Generic interval tables should support sleep-like stage labels."""
    interval_csv = tmp_path / "sleep_intervals.csv"
    interval_csv.write_text(
        "subject,start,end,label\n"
        "sub-001,0,30,Wake\n"
        "sub-001,30,60,N2\n",
        encoding="utf-8",
    )
    config = {
        "within_run_labels": {
            "datasets": {
                "dsSleep": {
                    "enabled": True,
                    "output_name": "sleep_stage_v1",
                    "write_to_stage": True,
                    "write_to_labels": True,
                    "codebook": {"Wake": 0, "N2": 2},
                    "rules": [
                        {
                            "id": "sleep_intervals",
                            "match": {"task": "sleep", "run": "run-1"},
                            "source": {
                                "type": "interval_table",
                                "path": str(interval_csv),
                                "subject_column": "subject",
                                "start_column": "start",
                                "end_column": "end",
                                "label_column": "label",
                                "units": "seconds",
                            },
                        }
                    ],
                }
            }
        }
    }

    result = build_within_run_labels(
        config=config,
        dataset_id="dsSleep",
        dataset_root=tmp_path,
        sub_id="sub-001",
        ses_id=None,
        task="sleep",
        raw_task="sleep",
        run_id="run-1",
        acq_id=None,
        tr_sec=None,
        time=np.array([15.0, 45.0], dtype=float),
        window_start=np.array([0.0, 30.0], dtype=float),
        window_end=np.array([30.0, 60.0], dtype=float),
        sub_frame=pd.DataFrame({"dummy": [1, 2]}),
    )

    assert result.labels["sleep_stage_v1"].tolist() == ["Wake", "N2"]
    assert result.stage.tolist() == [0, 2]
