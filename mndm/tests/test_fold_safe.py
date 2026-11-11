from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parents[2] / "core" / "src"))

from mndm.pipeline.fold_safe import (
    FoldSplitError,
    fit_fold_anchor_artifact,
    load_fold_split,
    load_signal_support_ledger,
    select_fold_rows,
    validate_fold_anchor_artifact,
    validate_fold_signal_support,
    verify_source_file_hashes,
)
from mndm import anchors
from mndm.__about__ import __version__ as _MNDM_VERSION
from mndm.projection import project_features


def _manifest_payload(*, apply_start: float = 4.0, time_status: str = "verified") -> dict:
    return {
        "schema_version": "mndm.fold_split.v1",
        "split_id": "synthetic-fold-a",
        "time_reference": {
            "status": time_status,
            "start_column": "t_start",
            "end_column": "t_end",
            "unit": "seconds",
            "origin": "recording_start",
        },
        "source_file_sha256": {"sub-001_run-1.edf": hashlib.sha256(b"raw-recording").hexdigest()},
        "training_ranges": [
            {"subject_id": "sub-001", "recording_id": "sub-001_run-1.edf", "start_sec": 0.0, "end_sec": apply_start}
        ],
        "apply_ranges": [
            {"subject_id": "sub-001", "recording_id": "sub-001_run-1.edf", "start_sec": apply_start, "end_sec": 8.0}
        ],
    }


def _write_manifest(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(json.dumps(payload))
    if payload.pop("_with_signal_support", False):
        ledger = {"schema": "mndm.signal_support_export.v1", "temporal_support_status": "finite_causal_conservative_bound"}
        ledger_rows = []
        for ranges in (payload["training_ranges"], payload["apply_ranges"]):
            for interval in ranges:
                recording = interval["recording_id"]
                subject = interval["subject_id"]
                source_hash = payload["source_file_sha256"][recording]
                start = int(round(float(interval["start_sec"]) / 2.0))
                stop = int(round(float(interval["end_sec"]) / 2.0))
                for epoch_number in range(start, stop):
                    nominal_start = epoch_number * 2.0
                    nominal_end = nominal_start + 2.0
                    ledger_rows.append(
                        {
                            "raw_file": recording,
                            "recording_id": Path(recording).stem,
                            "epoch_id": f"e{epoch_number}",
                            "nominal_time_start_sec": nominal_start,
                            "nominal_time_end_sec": nominal_end,
                            "sample_mapping_status": "nominal_bounds_mapped",
                            "actual_sample_contributors_status": "finite_causal_conservative_bound",
                            "actual_contributor_interval_convention": (
                                "half_open_0_based_source_array_indices; conservative_dependency_superset"
                            ),
                            "original_sample_interval_convention": (
                                "half_open_0_based_source_array_indices_within_nominal_window"
                            ),
                            "source_file_sha256": source_hash,
                            "source_sample_rate_hz": 100.0,
                            "original_sample_start": int(nominal_start * 100),
                            "original_sample_end": int(nominal_end * 100),
                            "actual_source_sample_start": int(nominal_start * 100),
                            "actual_source_sample_end_exclusive": int(nominal_end * 100),
                        }
                    )
        ledger["per_epoch_input_extent"] = ledger_rows
        ledger["execution_records"] = [
            {
                "raw_file": recording,
                "validation_status": "valid",
                "record": {
                    "source": {"sha256": digest},
                    "raw_clock_before_crop": {"n_times": 800, "sfreq_hz": 100.0},
                },
            }
            for recording, digest in payload["source_file_sha256"].items()
        ]
        ledger_path = path.with_name("signal_support_ledger.json")
        ledger_bytes = json.dumps(ledger, sort_keys=True).encode("utf-8")
        ledger_path.write_bytes(ledger_bytes)
        payload["signal_support_ledger_sha256"] = hashlib.sha256(ledger_bytes).hexdigest()
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _features(apply_values: tuple[float, float] = (500.0, 900.0)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": ["sub-001"] * 4,
            "file": ["sub-001_run-1.edf"] * 4,
            "epoch_id": ["e0", "e1", "e2", "e3"],
            "t_start": [0.0, 2.0, 4.0, 6.0],
            "t_end": [2.0, 4.0, 6.0, 8.0],
            "eeg_alpha": [2.0, 4.0, apply_values[0], apply_values[1]],
            "CPC": [1, 1, 5, 5],
        }
    )


def _fit(
    selection,
    split,
    *,
    transform_config=None,
    implementation_hash="synthetic-implementation",
    signal_support_validation=None,
):
    verified_hashes = {
        name: split.payload["source_file_sha256"][name]
        for name in sorted({item["recording_id"] for item in selection.row_records})
    }
    return fit_fold_anchor_artifact(
        selection,
        split,
        anchor_id="synthetic-training-anchor",
        anchor_source="training_ranges_only",
        cohort_filter="manifest-selected training rows",
        feature_standardization={"eeg_alpha": ["log10", "robust_z", "clip"]},
        min_subjects=1,
        scale_method="iqr",
        transform_config=transform_config or {"normalization": {"enabled": False}, "clip_threshold": 6.0},
        code_version=_MNDM_VERSION,
        implementation_hash=implementation_hash,
        verified_source_file_sha256=verified_hashes,
        feature_columns=["eeg_alpha"],
        signal_support_validation=signal_support_validation,
    )


def test_fold_anchor_bytes_do_not_depend_on_heldout_rows_or_values(tmp_path: Path):
    manifest = load_fold_split(_write_manifest(tmp_path / "split.json", _manifest_payload()))
    with_heldout = _features()
    fewer_heldout = with_heldout.iloc[:3].copy()
    train_a = select_fold_rows(with_heldout, manifest, "training")
    train_b = select_fold_rows(fewer_heldout, manifest, "training")
    train_c = select_fold_rows(_features((1e6, -1e6)), manifest, "training")

    artifact_a = _fit(train_a, manifest)
    artifact_b = _fit(train_b, manifest)
    artifact_c = _fit(train_c, manifest)

    assert json.dumps(artifact_a, sort_keys=True, separators=(",", ":")) == json.dumps(
        artifact_b, sort_keys=True, separators=(",", ":")
    )
    assert json.dumps(artifact_a, sort_keys=True, separators=(",", ":")) == json.dumps(
        artifact_c, sort_keys=True, separators=(",", ":")
    )
    record = validate_fold_anchor_artifact(artifact_a, manifest)
    assert record["fit_status"] == "complete"
    assert record["fit_population_count"] == 2
    assert record["fit_population_hash"]
    assert record["feature_fit_populations"]["eeg_alpha"]["fit_count"] == 1
    assert record["feature_fit_populations"]["eeg_alpha"]["fit_row_count"] == 2
    assert len(record["fit_time_ranges"]) == 2
    assert record["fit_time_ranges"][0]["start_sec"] == 0.0
    assert record["fit_time_ranges"][1]["end_sec"] == 4.0
    assert set(record["feature_fit_populations"]) == {"eeg_alpha"}
    assert record["fit_source_file_sha256"]["sub-001_run-1.edf"] == hashlib.sha256(b"raw-recording").hexdigest()

    tampered = json.loads(json.dumps(artifact_a))
    tampered["features"][0]["center"] = -123.0
    with pytest.raises(FoldSplitError, match="artifact hash does not match"):
        validate_fold_anchor_artifact(tampered, manifest)


def test_fold_anchor_uses_filename_subject_when_subject_cell_is_missing(tmp_path: Path):
    split = load_fold_split(_write_manifest(tmp_path / "split.json", _manifest_payload()))
    frame = _features()
    frame["subject"] = pd.NA
    selection = select_fold_rows(frame, split, "training")

    artifact = _fit(selection, split)

    record = validate_fold_anchor_artifact(artifact, split)
    assert record["feature_fit_populations"]["eeg_alpha"]["fit_population_ids"] == ["sub-001"]


def test_fold_split_rejects_overlapping_ranges_before_selection(tmp_path: Path):
    payload = _manifest_payload(apply_start=3.0)
    payload["training_ranges"][0]["end_sec"] = 4.0
    with pytest.raises(FoldSplitError, match="ranges overlap"):
        load_fold_split(_write_manifest(tmp_path / "overlap.json", payload))


def test_fold_selection_refuses_rows_crossing_partition_boundary(tmp_path: Path):
    payload = _manifest_payload(apply_start=4.0)
    split = load_fold_split(_write_manifest(tmp_path / "split.json", payload))
    frame = _features()
    frame.loc[1, "t_end"] = 4.5

    with pytest.raises(FoldSplitError, match="straddles a training range boundary"):
        select_fold_rows(frame, split, "training")


def test_fold_selection_refuses_recording_id_that_disagrees_with_source_file(tmp_path: Path):
    split = load_fold_split(_write_manifest(tmp_path / "split.json", _manifest_payload()))
    frame = _features()
    frame["recording_id"] = frame["file"]
    frame.loc[0, "file"] = "sub-002_run-1.edf"

    with pytest.raises(FoldSplitError, match="recording_id does not match"):
        select_fold_rows(frame, split, "training")


def test_fold_split_requires_verified_clock_and_source_hashes(tmp_path: Path):
    with pytest.raises(FoldSplitError, match="time_reference.status must be verified"):
        load_fold_split(_write_manifest(tmp_path / "unverified.json", _manifest_payload(time_status="ambiguous")))

    payload = _manifest_payload()
    payload["source_file_sha256"] = {}
    with pytest.raises(FoldSplitError, match="source_file_sha256 missing"):
        load_fold_split(_write_manifest(tmp_path / "missing-hash.json", payload))
    with pytest.raises(FoldSplitError, match="Could not read fold split manifest"):
        load_fold_split(tmp_path / "absent-split.json")


def test_raw_source_read_failure_is_a_fold_refusal(tmp_path: Path):
    with pytest.raises(FoldSplitError, match="raw source file is missing"):
        verify_source_file_hashes(
            {"sub-001_run-1.edf": hashlib.sha256(b"raw-recording").hexdigest()},
            {"sub-001_run-1.edf": tmp_path / "missing.edf"},
        )


def test_fold_projection_refuses_local_fit_when_external_feature_anchor_is_missing():
    with pytest.raises(ValueError, match="no valid fitted robust-z anchor"):
        project_features(
            pd.DataFrame({"eeg_alpha": [1.0, 3.0]}),
            {"m": {"eeg_alpha": 1.0}},
            normalize="robust_z",
            external_anchor={},
            require_external_anchor=True,
        )


def _runner_for_fold(config: dict, manifest_path: Path, tmp_path: Path) -> DatasetSummaryRunner:
    from mndm.pipeline.summary import DatasetSummaryRunner

    runner = DatasetSummaryRunner.__new__(DatasetSummaryRunner)
    runner.config = config
    dataset_root = tmp_path / "raw" / "synthetic"
    rel_path = Path("sub-001") / "eeg" / "sub-001_run-1.edf"
    source_path = dataset_root / rel_path
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"raw-recording")
    config.setdefault("paths", {})["dataset_received_dirs"] = {"synthetic": str(dataset_root)}
    runner.ctx = SimpleNamespace(
        weights={"m": {"eeg_alpha": 1.0}},
        normalize_override="robust_z",
        received_dir=tmp_path / "raw",
    )
    runner.received_dir = tmp_path / "raw"
    runner.config_path = manifest_path.parent / "pilot.yaml"
    runner.ds_id = "synthetic"
    runner._fold_split = None
    runner._fold_selection = None
    runner._fold_role = None
    runner._fold_anchor_artifact = None
    runner._fold_verified_source_hashes = {}
    runner._fold_signal_support_validation = None
    runner.index_df = pd.DataFrame({"path": [str(rel_path)], "subject": ["001"]})
    runner._index_paths_by_basename = {rel_path.name: [str(rel_path)]}
    runner._normalization_report = {}
    return runner


def _fold_config(manifest_path: Path, *, role: str, anchor_path: Path | None = None) -> dict:
    anchor = {"enabled": False}
    auto = {"enabled": role == "training", "anchor_id": "synthetic-training-anchor", "min_subjects": 1}
    if anchor_path is not None:
        anchor = {"enabled": True, "path": str(anchor_path), "scale_method": "iqr", "min_subjects": 1}
    return {
        "mnps_projection": {
            "fold_safe": {
                "enabled": True,
                "manifest": str(manifest_path),
                "signal_support_ledger": str(manifest_path.with_name("signal_support_ledger.json")),
                "role": role,
            },
            "export_contracts": {"subject_anchored": False, "cohort_anchored": True},
            "anchor": anchor,
            "anchor_auto_fit": auto,
            "feature_standardization": {"eeg_alpha": ["log10", "robust_z", "clip"]},
        },
        "normalization": {"enabled": False},
    }


def test_summary_runner_selects_manifest_rows_and_refuses_combat_fit(tmp_path: Path, monkeypatch):
    payload = _manifest_payload()
    payload["_with_signal_support"] = True
    manifest_path = _write_manifest(tmp_path / "split.json", payload)
    split = load_fold_split(manifest_path)
    training_runner = _runner_for_fold(_fold_config(manifest_path, role="training"), manifest_path, tmp_path)
    training_frame = training_runner._apply_fold_safe_partition(_features())
    assert training_frame["epoch_id"].tolist() == ["e0", "e1"]
    assert training_runner._fold_role == "training"
    assert training_runner._fold_signal_support_validation["status"] == "verified_no_shared_source_samples"
    assert training_runner._fold_signal_support_validation["minimum_source_sample_gap_by_recording"] == {
        "sub-001_run-1.edf": 0
    }

    mismatched_subject_runner = _runner_for_fold(
        _fold_config(manifest_path, role="training"), manifest_path, tmp_path
    )
    mismatched_subject_runner.index_df.loc[0, "subject"] = "002"
    with pytest.raises(FoldSplitError, match="file_index subject does not match"):
        mismatched_subject_runner._apply_fold_safe_partition(_features())

    artifact = _fit(
        select_fold_rows(_features(), split, "training"),
        split,
        transform_config=training_runner._fold_transform_config(),
        implementation_hash=training_runner._fold_transform_implementation_hash(),
        signal_support_validation=training_runner._fold_signal_support_validation,
    )
    anchor_path = anchors.save_anchor_file(artifact, tmp_path / "anchor.json")
    # The one-subject fixture intentionally cannot estimate a usable cohort
    # scale; bypass that independent gate to exercise the ComBat refusal path.
    monkeypatch.setattr(anchors, "anchor_mapping", lambda *args, **kwargs: {"eeg_alpha": {"center": 0.0, "scale": 1.0}})
    apply_config = _fold_config(manifest_path, role="apply", anchor_path=anchor_path)
    apply_runner = _runner_for_fold(apply_config, manifest_path, tmp_path)
    apply_frame = apply_runner._apply_fold_safe_partition(_features())
    assert apply_frame["epoch_id"].tolist() == ["e2", "e3"]
    assert apply_runner._fold_anchor_artifact is not None
    apply_frame = apply_frame.reset_index(drop=True)
    chart_provenance = apply_runner._fold_chart_provenance(apply_frame)
    assert chart_provenance["apply_status"] == "applied"
    assert chart_provenance["fit_status"] == "complete"
    assert set(chart_provenance["fit_row_ids"]).isdisjoint(chart_provenance["apply_row_ids"])
    assert chart_provenance["signal_support_validation"]["signal_support_ledger_sha256"] == split.payload[
        "signal_support_ledger_sha256"
    ]

    from mndm.pipeline.summary import SubjectSummaryRunner

    verified_artifact = apply_runner._fold_anchor_artifact
    anchor_path.write_text("{}", encoding="utf-8")
    subject_runner = SubjectSummaryRunner.__new__(SubjectSummaryRunner)
    subject_runner.dataset = apply_runner
    monkeypatch.setattr(anchors, "load_anchor_file", lambda _: pytest.fail("fold anchor should not be reloaded"))
    assert subject_runner._resolve_projection_anchor_artifact(anchor_path) is verified_artifact

    combat_config = _fold_config(manifest_path, role="apply", anchor_path=anchor_path)
    combat_config["normalization"] = {"enabled": True, "method": "combat", "scope": "post_features"}
    combat_runner = _runner_for_fold(combat_config, manifest_path, tmp_path)
    combat_runner._fold_split = split
    combat_runner._fold_role = "apply"
    combat_frame = _features()
    with pytest.raises(FoldSplitError, match="not_testable"):
        combat_runner._apply_feature_normalization(combat_frame)
    assert combat_runner._normalization_report["status"] == "not_testable"
    refusal_path = combat_runner._persist_fold_refusal_report(tmp_path / "refused-output")
    refusal = json.loads(refusal_path.read_text(encoding="utf-8"))
    assert refusal["schema"] == "mndm.fold_safe_refusal.v1"
    assert refusal["apply_status"] == "not_testable"
    assert "no saved fit/apply artifact" in refusal["reason"]
    assert (refusal_path.parent / "normalization_report.json").is_file()


def test_apply_refuses_transform_config_or_raw_source_mismatch(tmp_path: Path):
    payload = _manifest_payload()
    payload["_with_signal_support"] = True
    manifest_path = _write_manifest(tmp_path / "split.json", payload)
    split = load_fold_split(manifest_path)
    training_runner = _runner_for_fold(_fold_config(manifest_path, role="training"), manifest_path, tmp_path)
    training_runner._apply_fold_safe_partition(_features())
    artifact = _fit(
        select_fold_rows(_features(), split, "training"),
        split,
        transform_config=training_runner._fold_transform_config(),
        implementation_hash=training_runner._fold_transform_implementation_hash(),
        signal_support_validation=training_runner._fold_signal_support_validation,
    )
    anchor_path = anchors.save_anchor_file(artifact, tmp_path / "anchor.json")

    changed_config = _fold_config(manifest_path, role="apply", anchor_path=anchor_path)
    changed_config["mnps_projection"]["feature_standardization"]["eeg_alpha"] = ["robust_z", "clip"]
    apply_runner = _runner_for_fold(changed_config, manifest_path, tmp_path)
    with pytest.raises(FoldSplitError, match="configuration differs"):
        apply_runner._apply_fold_safe_partition(_features())

    changed_source_runner = _runner_for_fold(
        _fold_config(manifest_path, role="apply", anchor_path=anchor_path), manifest_path, tmp_path
    )
    source_path = tmp_path / "raw" / "synthetic" / "sub-001" / "eeg" / "sub-001_run-1.edf"
    source_path.write_bytes(b"changed-raw-recording")
    with pytest.raises(FoldSplitError, match="raw source SHA256 mismatch"):
        changed_source_runner._apply_fold_safe_partition(_features())


def test_summary_runner_refuses_overlapping_conservative_raw_sample_support(tmp_path: Path):
    payload = _manifest_payload()
    payload["_with_signal_support"] = True
    manifest_path = _write_manifest(tmp_path / "split.json", payload)
    ledger_path = manifest_path.with_name("signal_support_ledger.json")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    apply_row = next(row for row in ledger["per_epoch_input_extent"] if row["epoch_id"] == "e2")
    apply_row["actual_source_sample_start"] = 350
    ledger_bytes = json.dumps(ledger, sort_keys=True).encode("utf-8")
    ledger_path.write_bytes(ledger_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["signal_support_ledger_sha256"] = hashlib.sha256(ledger_bytes).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    runner = _runner_for_fold(_fold_config(manifest_path, role="training"), manifest_path, tmp_path)
    with pytest.raises(FoldSplitError, match="conservative raw-sample support overlaps"):
        runner._apply_fold_safe_partition(_features())


def test_summary_runner_refuses_stale_or_ambiguous_support_ledger(tmp_path: Path):
    payload = _manifest_payload()
    payload["_with_signal_support"] = True
    manifest_path = _write_manifest(tmp_path / "stale.json", payload)
    ledger_path = manifest_path.with_name("signal_support_ledger.json")
    with ledger_path.open("ab") as stream:
        stream.write(b" ")
    runner = _runner_for_fold(_fold_config(manifest_path, role="training"), manifest_path, tmp_path)
    with pytest.raises(FoldSplitError, match="ledger SHA256 does not match"):
        runner._apply_fold_safe_partition(_features())

    ambiguous_path = _write_manifest(tmp_path / "ambiguous.json", payload)
    ambiguous_ledger_path = ambiguous_path.with_name("signal_support_ledger.json")
    ledger = json.loads(ambiguous_ledger_path.read_text(encoding="utf-8"))
    ledger["per_epoch_input_extent"][0]["actual_sample_contributors_status"] = "unknown"
    ledger_bytes = json.dumps(ledger, sort_keys=True).encode("utf-8")
    ambiguous_ledger_path.write_bytes(ledger_bytes)
    manifest = json.loads(ambiguous_path.read_text(encoding="utf-8"))
    manifest["signal_support_ledger_sha256"] = hashlib.sha256(ledger_bytes).hexdigest()
    ambiguous_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    ambiguous_runner = _runner_for_fold(_fold_config(ambiguous_path, role="training"), ambiguous_path, tmp_path)
    with pytest.raises(FoldSplitError, match="unknown/noncausal support"):
        ambiguous_runner._apply_fold_safe_partition(_features())

    duplicate_path = _write_manifest(tmp_path / "duplicate.json", payload)
    duplicate_ledger_path = duplicate_path.with_name("signal_support_ledger.json")
    ledger = json.loads(duplicate_ledger_path.read_text(encoding="utf-8"))
    duplicate = next(row for row in ledger["per_epoch_input_extent"] if row["epoch_id"] == "e2")
    ledger["per_epoch_input_extent"].append(dict(duplicate))
    ledger_bytes = json.dumps(ledger, sort_keys=True).encode("utf-8")
    duplicate_ledger_path.write_bytes(ledger_bytes)
    manifest = json.loads(duplicate_path.read_text(encoding="utf-8"))
    manifest["signal_support_ledger_sha256"] = hashlib.sha256(ledger_bytes).hexdigest()
    duplicate_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    duplicate_runner = _runner_for_fold(_fold_config(duplicate_path, role="training"), duplicate_path, tmp_path)
    with pytest.raises(FoldSplitError, match="2 matching support ledger rows"):
        duplicate_runner._apply_fold_safe_partition(_features())

    clock_path = _write_manifest(tmp_path / "clock.json", payload)
    clock_ledger_path = clock_path.with_name("signal_support_ledger.json")
    ledger = json.loads(clock_ledger_path.read_text(encoding="utf-8"))
    ledger["per_epoch_input_extent"][0]["source_sample_rate_hz"] = 200.0
    ledger_bytes = json.dumps(ledger, sort_keys=True).encode("utf-8")
    clock_ledger_path.write_bytes(ledger_bytes)
    manifest = json.loads(clock_path.read_text(encoding="utf-8"))
    manifest["signal_support_ledger_sha256"] = hashlib.sha256(ledger_bytes).hexdigest()
    clock_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    clock_runner = _runner_for_fold(_fold_config(clock_path, role="training"), clock_path, tmp_path)
    with pytest.raises(FoldSplitError, match="sample rate disagrees with execution raw clock"):
        clock_runner._apply_fold_safe_partition(_features())


def test_successful_training_replay_uses_frozen_anchor(tmp_path: Path):
    payload = _manifest_payload()
    payload["split_id"] = "two-subject-fold"
    payload["source_file_sha256"] = {
        "sub-001_run-1.edf": hashlib.sha256(b"raw-1").hexdigest(),
        "sub-002_run-1.edf": hashlib.sha256(b"raw-2").hexdigest(),
    }
    payload["training_ranges"].append(
        {"subject_id": "sub-002", "recording_id": "sub-002_run-1.edf", "start_sec": 0.0, "end_sec": 4.0}
    )
    payload["apply_ranges"].append(
        {"subject_id": "sub-002", "recording_id": "sub-002_run-1.edf", "start_sec": 4.0, "end_sec": 8.0}
    )
    split = load_fold_split(_write_manifest(tmp_path / "two-subject-split.json", payload))
    frame = pd.DataFrame(
        {
            "subject": ["sub-001"] * 4 + ["sub-002"] * 4,
            "file": ["sub-001_run-1.edf"] * 4 + ["sub-002_run-1.edf"] * 4,
            "epoch_id": ["e0", "e1", "e2", "e3"] * 2,
            "t_start": [0.0, 2.0, 4.0, 6.0] * 2,
            "t_end": [2.0, 4.0, 6.0, 8.0] * 2,
            "eeg_alpha": [2.0, 4.0, 64.0, 128.0, 8.0, 16.0, 256.0, 512.0],
        }
    )
    selection = select_fold_rows(frame, split, "training")
    transform = {"eeg_alpha": ["log10", "robust_z", "clip"]}
    artifact = fit_fold_anchor_artifact(
        selection,
        split,
        anchor_id="two-subject-anchor",
        anchor_source="training_ranges_only",
        cohort_filter="manifest-selected training rows",
        feature_standardization=transform,
        min_subjects=2,
        scale_method="iqr",
        transform_config={"test": "transform"},
        code_version="test-version",
        implementation_hash="synthetic-implementation",
        verified_source_file_sha256={
            name: split.payload["source_file_sha256"][name]
            for name in ("sub-001_run-1.edf", "sub-002_run-1.edf")
        },
        feature_columns=["eeg_alpha"],
    )
    external_anchor = anchors.anchor_mapping(artifact, scale_method="iqr", min_subjects=2)
    chart, _ = project_features(
        selection.frame,
        {"m": {"eeg_alpha": 1.0}},
        normalize="robust_z",
        feature_standardization=transform,
        external_anchor=external_anchor,
        require_external_anchor=True,
    )
    transformed = np.log10(selection.frame["eeg_alpha"].to_numpy(dtype=float))
    anchor = external_anchor["eeg_alpha"]
    expected = np.clip((transformed - anchor["center"]) / (anchor["scale"] + 1e-9), -6.0, 6.0)
    np.testing.assert_allclose(chart[:, 0], expected, rtol=1e-6, atol=1e-6)


def test_numeric_zero_epoch_id_matches_fold_signal_support_ledger(tmp_path: Path):
    payload = _manifest_payload()
    payload["_with_signal_support"] = True
    manifest_path = _write_manifest(tmp_path / "numeric-epoch-id.json", payload)

    ledger_path = manifest_path.with_name("signal_support_ledger.json")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    for row in ledger["per_epoch_input_extent"]:
        row["epoch_id"] = int(str(row["epoch_id"])[1:])
    ledger_bytes = json.dumps(ledger, sort_keys=True).encode("utf-8")
    ledger_path.write_bytes(ledger_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["signal_support_ledger_sha256"] = hashlib.sha256(ledger_bytes).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    features = _features()
    features["epoch_id"] = [0, 1, 2, 3]
    split = load_fold_split(manifest_path)
    training = select_fold_rows(features, split, "training")
    apply = select_fold_rows(features, split, "apply")
    assert training.row_records[0]["epoch_id"] == "0"
    assert apply.row_records[0]["epoch_id"] == "2"
    ledger_ref = load_signal_support_ledger(
        ledger_path,
        expected_sha256=split.payload["signal_support_ledger_sha256"],
    )
    validation = validate_fold_signal_support(training, apply, split, ledger_ref)
    assert validation["status"] == "verified_no_shared_source_samples"
    assert validation["training_rows_verified"] == 2
    assert validation["apply_rows_verified"] == 2
