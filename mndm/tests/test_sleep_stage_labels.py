from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from mndm.pipeline.summary import DatasetSummaryRunner, SubjectSummaryRunner
from mndm.pipeline import summary as summary_mod


def _build_ctx(tmp_path: Path):
    received = tmp_path / "received"
    processed = tmp_path / "processed"
    return SimpleNamespace(
        config={
            "paths": {"received_dir": str(received), "processed_dir": str(processed)},
            "epoching": {"datasets": {"ds005555": {"sampling": {"stage_columns": ["stage_hum", "stage_ai", "stage"]}}}},
            "mnps": {"stage_codebook": {"Wake": 0, "N2": 2, "N3": 3, "REM": 4}},
            "robustness": {"coverage": {}},
        },
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
            "stage_codebook": {"Wake": 0, "N2": 2, "N3": 3, "REM": 4},
            "embodied": {"enabled": False},
            "surrogates": {},
            "reliability": {},
            "whiten": True,
        },
        derivative_cfg={"method": "central", "window": 5, "polyorder": 2},
        extensions_cfg={},
    )


def test_stage_labels_written_from_events_tsv(monkeypatch, tmp_path: Path):
    ctx = _build_ctx(tmp_path)
    ds_id = "ds005555"

    # Build fake received dataset structure with EEG + events TSV
    received_ds = ctx.received_dir / ds_id / "sub-001" / "eeg"
    received_ds.mkdir(parents=True, exist_ok=True)
    eeg_name = "sub-001_task-Sleep_acq-psg_eeg.edf"
    eeg_path = received_ds / eeg_name
    eeg_path.write_bytes(b"")  # dummy file for existence checks

    events_path = received_ds / "sub-001_task-Sleep_acq-psg_events.tsv"
    events_path.write_text(
        "onset\tduration\tstage_hum\n0\t30\tWake\n30\t30\tN2\n",
        encoding="utf-8",
    )

    # Minimal processed index: must include the EEG path
    index_df = pd.DataFrame([{"path": str(Path("sub-001/eeg") / eeg_name), "modality": "eeg", "subject": "001"}])

    # Minimal features for the same file (t_start/t_end match scoring bins)
    features_df = pd.DataFrame(
        {
            "file": [eeg_name, eeg_name],
            "t_start": [0.0, 30.0],
            "t_end": [30.0, 60.0],
            "qc_ok_eeg": [1, 1],
            "feat_m": np.linspace(0, 1, 2),
            "feat_d": np.linspace(1, 2, 2),
            "feat_e": np.linspace(2, 3, 2),
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

    subject_runner.run(sub_id="sub-001", ses_id=None, raw_task="Sleep", run_id=None, acq_id="acq-psg", sub_frame=features_df)

    payload = captured.get("payload")
    assert payload is not None
    assert payload.stage is not None
    assert payload.stage.shape[0] == 2
    # codebook mapping: Wake -> 0, N2 -> 2
    assert int(payload.stage[0]) == 0
    assert int(payload.stage[1]) == 2


def test_primary_mnps_jacobian_can_be_disabled_via_config(monkeypatch, tmp_path: Path):
    ctx = _build_ctx(tmp_path)
    ctx.config["mnps"] = {"jacobian": {"enabled": False}}
    ctx.config["mnps_9d"] = {"enabled": False}
    ds_id = "ds005555"

    received_ds = ctx.received_dir / ds_id / "sub-001" / "eeg"
    received_ds.mkdir(parents=True, exist_ok=True)
    eeg_name = "sub-001_task-Sleep_acq-psg_eeg.edf"
    eeg_path = received_ds / eeg_name
    eeg_path.write_bytes(b"")

    index_df = pd.DataFrame([{"path": str(Path("sub-001/eeg") / eeg_name), "modality": "eeg", "subject": "001"}])
    features_df = pd.DataFrame(
        {
            "file": [eeg_name, eeg_name, eeg_name],
            "t_start": [0.0, 30.0, 60.0],
            "t_end": [30.0, 60.0, 90.0],
            "qc_ok_eeg": [1, 1, 1],
            "feat_m": np.linspace(0, 1, 3),
            "feat_d": np.linspace(1, 2, 3),
            "feat_e": np.linspace(2, 3, 3),
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

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("estimate_local_jacobians should not run when mnps.jacobian.enabled=false")

    monkeypatch.setattr(summary_mod.jacobian, "estimate_local_jacobians", _fail_if_called)

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

    subject_runner.run(sub_id="sub-001", ses_id=None, raw_task="Sleep", run_id=None, acq_id="acq-psg", sub_frame=features_df)

    payload = captured.get("payload")
    assert payload is not None
    assert payload.jacobian is None
    assert payload.jacobian_dot is None
    assert payload.jacobian_centers is None

