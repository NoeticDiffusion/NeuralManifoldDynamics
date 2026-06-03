from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from mndm.pipeline.summary import DatasetSummaryRunner, SubjectSummaryRunner
from mndm.pipeline import summary as summary_mod


def _build_ctx(tmp_path: Path):
    """Internal helper: build ctx."""
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
    """Test stage labels written from events tsv."""
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


def test_stage_labels_written_from_point_events_tsv(monkeypatch, tmp_path: Path):
    """Point-events in events.tsv should map to containing feature windows."""
    ctx = _build_ctx(tmp_path)
    ds_id = "ds005555"

    received_ds = ctx.received_dir / ds_id / "sub-001" / "eeg"
    received_ds.mkdir(parents=True, exist_ok=True)
    eeg_name = "sub-001_task-Sleep_acq-psg_eeg.edf"
    eeg_path = received_ds / eeg_name
    eeg_path.write_bytes(b"")

    events_path = received_ds / "sub-001_task-Sleep_acq-psg_events.tsv"
    events_path.write_text(
        "onset\tduration\tstage_hum\n5\t0\tWake\n35\t0\tN2\n",
        encoding="utf-8",
    )

    index_df = pd.DataFrame([{"path": str(Path("sub-001/eeg") / eeg_name), "modality": "eeg", "subject": "001"}])
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
    assert int(payload.stage[0]) == 0
    assert int(payload.stage[1]) == 2


def test_photic_event_provenance_and_qc_are_emitted(monkeypatch, tmp_path: Path):
    """Photomark ingest should expose auditable event provenance and QC."""
    ctx = _build_ctx(tmp_path)
    ds_id = "ds006036"
    ctx.config["epoching"]["datasets"][ds_id] = {
        "sampling": {
            "onset_column": "onset",
            "duration_column": "duration",
            "stage_columns": ["value"],
            "stage_blocking": {
                "enabled": True,
                "stage_event_regex": r"(?i)^PHOTO\s*(\d+)\s*Hz$",
                "bridge_marker_labels": ["Photo/HV mark"],
                "use_bridge_markers": True,
                "hv_tail_sec": 0.5,
                "min_block_sec": 2.0,
                "max_block_sec": 20.0,
                "preserve_block_assignments": True,
                "expected_stage_frequencies_hz": [5, 10, 25, 30],
            },
        }
    }
    ctx.config["mnps"]["stage_codebook"] = {
        "PHOTO 5Hz": 50,
        "PHOTO 10Hz": 51,
        "Photo/HV mark": 54,
        "open eyes": 60,
    }
    ctx.mnps_cfg["stage_codebook"] = dict(ctx.config["mnps"]["stage_codebook"])

    received_ds = ctx.received_dir / ds_id / "sub-001" / "eeg"
    received_ds.mkdir(parents=True, exist_ok=True)
    eeg_name = "sub-001_task-photomark_eeg.edf"
    eeg_path = received_ds / eeg_name
    eeg_path.write_bytes(b"")

    events_path = received_ds / "sub-001_task-photomark_events.tsv"
    events_path.write_text(
        "\n".join(
            [
                "onset\tduration\tvalue",
                "0.0\t0\tPHOTO 5Hz",
                "0.2\t0\tPhoto/HV mark",
                "4.0\t0\topen eyes",
                "8.0\t0\tPhoto/HV mark",
                "20.0\t0\tPHOTO 10Hz",
                "20.2\t0\tPhoto/HV mark",
                "28.0\t0\tPhoto/HV mark",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    index_df = pd.DataFrame([{"path": str(Path("sub-001/eeg") / eeg_name), "modality": "eeg", "subject": "001"}])
    features_df = pd.DataFrame(
        {
            "file": [eeg_name, eeg_name, eeg_name, eeg_name],
            "t_start": [0.0, 8.0, 16.0, 24.0],
            "t_end": [8.0, 16.0, 24.0, 32.0],
            "qc_ok_eeg": [1, 1, 1, 1],
            "feat_m": np.linspace(0, 1, 4),
            "feat_d": np.linspace(1, 2, 4),
            "feat_e": np.linspace(2, 3, 4),
        }
    )

    captured = {}
    monkeypatch.setattr(
        summary_mod,
        "write_summary_manifest_and_h5",
        lambda **kwargs: captured.update({"payload": kwargs["payload"], "manifest": kwargs["manifest"]}),
    )
    monkeypatch.setattr(summary_mod.json_writer, "build_manifest", lambda *_args, **_kwargs: _args[3])
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

    subject_runner.run(
        sub_id="sub-001",
        ses_id=None,
        raw_task="photomark",
        run_id=None,
        acq_id=None,
        sub_frame=features_df,
    )

    payload = captured.get("payload")
    manifest = captured.get("manifest", {})
    assert payload is not None
    assert payload.stage is not None
    # Inferred continuous block surface should preserve photic labels.
    assert int(payload.stage[0]) == 50
    assert int(payload.stage[2]) == 51
    assert int(payload.stage[3]) == 51

    assert isinstance(payload.event_table_columns, dict)
    assert "raw_event_label" in payload.event_table_columns
    assert "mapped_stage_code" in payload.event_table_columns
    assert "mapping_mode" in payload.event_table_columns
    assert len(payload.event_table_columns["raw_event_label"]) == 7

    stage_qc = manifest.get("stage_mapping_qc", {})
    assert bool(stage_qc)
    assert stage_qc.get("raw_has_25hz") is False
    assert stage_qc.get("raw_has_30hz") is False
    assert 25 in stage_qc.get("missing_expected_frequencies_hz_raw", [])
    assert 30 in stage_qc.get("missing_expected_frequencies_hz_raw", [])

    event_prov = manifest.get("event_provenance", {})
    assert event_prov.get("status") == "available"
    assert event_prov.get("event_rows") == 7


def test_photic_qc_marks_25_and_30_when_present(monkeypatch, tmp_path: Path):
    """QC should explicitly report 25/30 Hz when present in raw events."""
    ctx = _build_ctx(tmp_path)
    ds_id = "ds006036"
    ctx.config["epoching"]["datasets"][ds_id] = {
        "sampling": {
            "onset_column": "onset",
            "duration_column": "duration",
            "stage_columns": ["value"],
            "stage_blocking": {
                "enabled": True,
                "stage_event_regex": r"(?i)^PHOTO\s*(\d+)\s*Hz$",
                "bridge_marker_labels": ["Photo/HV mark"],
                "use_bridge_markers": True,
                "hv_tail_sec": 0.5,
                "min_block_sec": 2.0,
                "max_block_sec": 20.0,
                "preserve_block_assignments": True,
                "expected_stage_frequencies_hz": [25, 30],
            },
        }
    }
    ctx.config["mnps"]["stage_codebook"] = {
        "PHOTO 25Hz": 55,
        "PHOTO 30Hz": 56,
        "Photo/HV mark": 54,
    }
    ctx.mnps_cfg["stage_codebook"] = dict(ctx.config["mnps"]["stage_codebook"])

    received_ds = ctx.received_dir / ds_id / "sub-001" / "eeg"
    received_ds.mkdir(parents=True, exist_ok=True)
    eeg_name = "sub-001_task-photomark_eeg.edf"
    (received_ds / eeg_name).write_bytes(b"")
    events_path = received_ds / "sub-001_task-photomark_events.tsv"
    events_path.write_text(
        "\n".join(
            [
                "onset\tduration\tvalue",
                "0.0\t0\tPHOTO 25Hz",
                "0.2\t0\tPhoto/HV mark",
                "8.0\t0\tPhoto/HV mark",
                "20.0\t0\tPHOTO 30Hz",
                "20.2\t0\tPhoto/HV mark",
                "28.0\t0\tPhoto/HV mark",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    index_df = pd.DataFrame([{"path": str(Path("sub-001/eeg") / eeg_name), "modality": "eeg", "subject": "001"}])
    features_df = pd.DataFrame(
        {
            "file": [eeg_name, eeg_name, eeg_name, eeg_name],
            "t_start": [0.0, 8.0, 16.0, 24.0],
            "t_end": [8.0, 16.0, 24.0, 32.0],
            "qc_ok_eeg": [1, 1, 1, 1],
            "feat_m": np.linspace(0, 1, 4),
            "feat_d": np.linspace(1, 2, 4),
            "feat_e": np.linspace(2, 3, 4),
        }
    )

    captured = {}
    monkeypatch.setattr(
        summary_mod,
        "write_summary_manifest_and_h5",
        lambda **kwargs: captured.update({"payload": kwargs["payload"], "manifest": kwargs["manifest"]}),
    )
    monkeypatch.setattr(summary_mod.json_writer, "build_manifest", lambda *_args, **_kwargs: _args[3])
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
    subject_runner.run(
        sub_id="sub-001",
        ses_id=None,
        raw_task="photomark",
        run_id=None,
        acq_id=None,
        sub_frame=features_df,
    )

    payload = captured.get("payload")
    manifest = captured.get("manifest", {})
    assert payload is not None
    assert payload.stage is not None
    assert 55 in set(int(v) for v in payload.stage.tolist())
    assert 56 in set(int(v) for v in payload.stage.tolist())

    stage_qc = manifest.get("stage_mapping_qc", {})
    assert stage_qc.get("raw_has_25hz") is True
    assert stage_qc.get("raw_has_30hz") is True
    assert stage_qc.get("missing_expected_frequencies_hz_raw", []) == []


def test_ds003490_exports_event_windows_and_self_describing_stage_labels(monkeypatch, tmp_path: Path):
    """ds003490 should export explicit event-window joins and EO/EC labels."""
    ctx = _build_ctx(tmp_path)
    ds_id = "ds003490"
    ctx.config["epoching"]["datasets"][ds_id] = {
        "sampling": {
            "onset_column": "onset",
            "duration_column": "duration",
            "stage_columns": ["trial_type"],
            "prefer_events_stage_in_summary": True,
            "stage_map": {
                "Eyes Closed: Every 1000 ms": 10,
                "Eyes Open: Every 1000 ms": 11,
            },
        }
    }
    ctx.config["event_mapping"] = {"datasets": {ds_id: {"enabled": True}}}
    ctx.config["mnps"]["stage_codebook"] = {
        "Eyes Closed: Every 1000 ms": 10,
        "Eyes Open: Every 1000 ms": 11,
    }
    ctx.mnps_cfg["stage_codebook"] = dict(ctx.config["mnps"]["stage_codebook"])

    received_ds = ctx.received_dir / ds_id / "sub-001" / "ses-01" / "eeg"
    received_ds.mkdir(parents=True, exist_ok=True)
    eeg_name = "sub-001_ses-01_task-Rest_eeg.edf"
    (received_ds / eeg_name).write_bytes(b"")
    (received_ds / "sub-001_ses-01_task-Rest_events.tsv").write_text(
        "\n".join(
            [
                "onset\tduration\ttrial_type",
                "0.0\t8.0\tEyes Closed: Every 1000 ms",
                "8.0\t8.0\tEyes Open: Every 1000 ms",
                "2.0\t0.0\tStandard Tone",
                "10.0\t0.0\tNovel Tone",
                "14.0\t0.0\tTarget Tone",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    index_df = pd.DataFrame(
        [
            {
                "path": str(Path("sub-001/ses-01/eeg") / eeg_name),
                "modality": "eeg",
                "subject": "001",
            }
        ]
    )
    features_df = pd.DataFrame(
        {
            "file": [eeg_name] * 4,
            "t_start": [0.0, 4.0, 8.0, 12.0],
            "t_end": [8.0, 12.0, 16.0, 20.0],
            "qc_ok_eeg": [1, 1, 1, 1],
            "feat_m": np.linspace(0.0, 1.0, 4),
            "feat_d": np.linspace(1.0, 2.0, 4),
            "feat_e": np.linspace(2.0, 3.0, 4),
        }
    )

    captured = {}
    monkeypatch.setattr(
        summary_mod,
        "write_summary_manifest_and_h5",
        lambda **kwargs: captured.update({"payload": kwargs["payload"], "manifest": kwargs["manifest"]}),
    )
    monkeypatch.setattr(summary_mod.json_writer, "build_manifest", lambda *_args, **_kwargs: _args[3])
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
    subject_runner.run(
        sub_id="sub-001",
        ses_id="ses-01",
        raw_task="Rest",
        run_id=None,
        acq_id=None,
        sub_frame=features_df,
    )

    payload = captured.get("payload")
    manifest = captured.get("manifest", {})
    assert payload is not None
    assert "stage" in payload.codebooks
    assert payload.codebooks["stage"]["label_keys"] == ["eyes_closed", "eyes_open"]
    assert "eyes_closed" in payload.labels
    assert "eyes_open" in payload.labels
    assert int(payload.labels["eyes_closed"].sum()) > 0
    assert int(payload.labels["eyes_open"].sum()) > 0

    event_windows = payload.event_windows
    assert "event_label" in event_windows
    assert "window_contains_event_onset" in event_windows
    assert "event_start_window_index" in event_windows
    assert "event_stop_window_index" in event_windows

    event_labels = np.asarray(event_windows["event_label"]).astype(str)
    contains = np.asarray(event_windows["window_contains_event_onset"], dtype=np.int32)
    window_ids = np.asarray(event_windows["window_id"], dtype=np.int32)
    start_windows = np.asarray(event_windows["event_start_window_index"], dtype=np.int32)
    stop_windows = np.asarray(event_windows["event_stop_window_index"], dtype=np.int32)

    standard_mask = event_labels == "Standard Tone"
    assert standard_mask.any()
    assert set(window_ids[standard_mask & (contains == 1)].tolist()) == {0}

    novel_mask = event_labels == "Novel Tone"
    assert novel_mask.any()
    assert set(window_ids[novel_mask & (contains == 1)].tolist()) == {1, 2}
    assert set(start_windows[novel_mask].tolist()) == {1}
    assert set(stop_windows[novel_mask].tolist()) == {2}

    target_mask = event_labels == "Target Tone"
    assert target_mask.any()
    assert set(window_ids[target_mask & (contains == 1)].tolist()) == {2, 3}
    assert set(start_windows[target_mask].tolist()) == {2}
    assert set(stop_windows[target_mask].tolist()) == {3}

    assert manifest["event_windows"]["path"] == "/event_windows"
    assert manifest["codebooks"]["stage_path"] == "/codebooks/stage"


def test_prefer_events_stage_in_summary_overrides_features_stage(monkeypatch, tmp_path: Path):
    """When configured, summarize should override stale feature stage with events stage."""
    ctx = _build_ctx(tmp_path)
    ds_id = "ds006036"
    ctx.config["epoching"]["datasets"][ds_id] = {
        "sampling": {
            "onset_column": "onset",
            "duration_column": "duration",
            "stage_columns": ["value"],
            "prefer_events_stage_in_summary": True,
            "stage_blocking": {
                "enabled": True,
                "stage_event_regex": r"(?i)^PHOTO\s*(\d+)\s*Hz$",
                "bridge_marker_labels": ["Photo/HV mark"],
                "use_bridge_markers": True,
                "hv_tail_sec": 0.5,
                "min_block_sec": 2.0,
                "max_block_sec": 20.0,
                "preserve_block_assignments": True,
            },
        }
    }
    ctx.config["mnps"]["stage_codebook"] = {
        "PHOTO 5Hz": 50,
        "PHOTO 10Hz": 51,
        "Photo/HV mark": 54,
        "open eyes": 60,
    }
    ctx.mnps_cfg["stage_codebook"] = dict(ctx.config["mnps"]["stage_codebook"])

    received_ds = ctx.received_dir / ds_id / "sub-001" / "eeg"
    received_ds.mkdir(parents=True, exist_ok=True)
    eeg_name = "sub-001_task-photomark_eeg.edf"
    (received_ds / eeg_name).write_bytes(b"")
    (received_ds / "sub-001_task-photomark_events.tsv").write_text(
        "\n".join(
            [
                "onset\tduration\tvalue",
                "0.0\t0\tPHOTO 5Hz",
                "0.2\t0\tPhoto/HV mark",
                "8.0\t0\tPhoto/HV mark",
                "20.0\t0\tPHOTO 10Hz",
                "20.2\t0\tPhoto/HV mark",
                "28.0\t0\tPhoto/HV mark",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    index_df = pd.DataFrame([{"path": str(Path("sub-001/eeg") / eeg_name), "modality": "eeg", "subject": "001"}])
    # Deliberately stale stage surface from features.csv (all open-eyes code).
    features_df = pd.DataFrame(
        {
            "file": [eeg_name, eeg_name, eeg_name, eeg_name],
            "t_start": [0.0, 8.0, 16.0, 24.0],
            "t_end": [8.0, 16.0, 24.0, 32.0],
            "stage": [60, 60, 60, 60],
            "qc_ok_eeg": [1, 1, 1, 1],
            "feat_m": np.linspace(0, 1, 4),
            "feat_d": np.linspace(1, 2, 4),
            "feat_e": np.linspace(2, 3, 4),
        }
    )

    captured = {}
    monkeypatch.setattr(
        summary_mod,
        "write_summary_manifest_and_h5",
        lambda **kwargs: captured.update({"payload": kwargs["payload"], "manifest": kwargs["manifest"]}),
    )
    monkeypatch.setattr(summary_mod.json_writer, "build_manifest", lambda *_args, **_kwargs: _args[3])
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
    subject_runner.run(
        sub_id="sub-001",
        ses_id=None,
        raw_task="photomark",
        run_id=None,
        acq_id=None,
        sub_frame=features_df,
    )

    payload = captured.get("payload")
    assert payload is not None and payload.stage is not None
    # If override worked, stage surface is no longer all 60 from stale features.csv.
    assert set(int(v) for v in payload.stage.tolist()) != {60}
    assert 50 in set(int(v) for v in payload.stage.tolist())
    assert 51 in set(int(v) for v in payload.stage.tolist())


def test_primary_mnps_jacobian_can_be_disabled_via_config(monkeypatch, tmp_path: Path):
    """Test primary mnps jacobian can be disabled via config."""
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
        """Internal helper: fail if called."""
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

