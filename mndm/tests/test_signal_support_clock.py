"""Cross-file and cropped nominal quality overlap, never effective support."""
import numpy as np
import pandas as pd
from mndm.signal_support_provenance import (
    build_signal_support_provenance,
    map_preprocessed_window_to_source_samples,
    refresh_source_hash,
)
from mndm.pipeline.signal_support_export import build_signal_support_export, validate_support_record


def record(tmp_path, name, onset):
    path = tmp_path / name
    path.write_bytes(name.encode())
    rec = build_signal_support_provenance(path)
    refresh_source_hash(rec)
    rec.update(raw_file=name, status="completed",
               crop={"applied": True, "actual_tmin_sec": 100.0},
               feature_time_clock={"reference": "preprocessed_array_start"},
               source_quality_intervals={"status": "observed", "intervals": [
                   {"raw_file": name, "clock": "original_raw_seconds", "start_sec": onset, "end_sec": onset + 10.0}
               ]})
    return rec


def test_crop_offset_and_source_identity_are_both_required(tmp_path):
    a = record(tmp_path, "a.edf", 110.)
    b = record(tmp_path, "b.edf", 100.)
    frame = pd.DataFrame({"file": ["a.edf", "a.edf"], "epoch_id": [0, 2],
                          "t_start": [0., 60.], "t_end": [30., 90.]})
    result = build_signal_support_export(frame, [a, b])
    first, second = result["per_epoch_input_extent"]
    assert first["nominal_original_time_start_sec"] == 100.
    assert len(first["source_quality_overlap"]) == 1
    assert first["source_quality_overlap"][0]["interval"]["raw_file"] == "a.edf"
    assert second["nominal_original_time_start_sec"] == 160.
    assert second["source_quality_overlap"] == []
    assert first["support_status"] == second["support_status"] == "unknown"
    assert first["original_sample_start"] is None


def test_unverified_clock_and_unstable_source_never_claim_overlap(tmp_path):
    rec = record(tmp_path, "a.edf", 110.)
    frame = pd.DataFrame({"file": ["a.edf"], "t_start": [0.], "t_end": [30.]})
    rec.pop("feature_time_clock")
    extent = build_signal_support_export(frame, [rec])["per_epoch_input_extent"][0]
    assert extent["source_quality_overlap_status"] == "unknown"
    rec["source"]["hash_stable"] = False
    result = build_signal_support_export(frame, [rec])
    assert result["source_quality_intervals"] == []
    assert result["per_epoch_input_extent"][0]["source_identity"] is None


def test_numpy_nonfinite_metadata_and_false_support_claim_fail_closed(tmp_path):
    rec = record(tmp_path, "a.edf", 110.)
    rec["temporal_support"] = {"status": "exact_finite"}
    frame = pd.DataFrame({"file": ["a.edf"], "t_start": [np.nan], "t_end": [np.inf]})
    result = build_signal_support_export(frame, [rec])
    row = result["per_epoch_input_extent"][0]
    assert row["support_status"] == "unknown"
    assert row["nominal_time_start_sec"] is None
    assert row["nominal_time_end_sec"] is None


def test_preprocessed_indices_map_nominal_window_through_crop_and_resampling():
    provenance = {
        "raw_clock_before_crop": {
            "sfreq_hz": 100.0,
            "first_time_sec": 5.0,
            "first_samp": 500,
            "n_times": 10_000,
        },
        "raw_clock_after_crop": {
            "sfreq_hz": 50.0,
            "first_time_sec": 15.0,
            "first_samp": 750,
            "n_times": 4_500,
        },
    }
    mapped = map_preprocessed_window_to_source_samples(
        50, 150, preprocessed_sfreq_hz=50.0, provenance=provenance
    )
    assert mapped["source_sample_mapping_status"] == "nominal_bounds_mapped"
    assert mapped["nominal_source_sample_start"] == 1_100
    assert mapped["nominal_source_sample_end_exclusive"] == 1_300
    assert mapped["nominal_source_sample_absolute_start"] == 1_600
    assert mapped["nominal_source_time_start_sec"] == 11.0
    assert mapped["nominal_source_time_end_sec"] == 13.0
    assert mapped["actual_sample_contributors_status"] == "unknown"


def test_preprocessed_window_outside_original_clock_fails_closed():
    provenance = {
        "raw_clock_before_crop": {"sfreq_hz": 10.0, "first_time_sec": 0.0, "first_samp": 0, "n_times": 100},
        "raw_clock_after_crop": {"sfreq_hz": 10.0, "first_time_sec": 9.0, "n_times": 20},
    }
    mapped = map_preprocessed_window_to_source_samples(
        10, 20, preprocessed_sfreq_hz=10.0, provenance=provenance
    )
    assert mapped["source_sample_mapping_status"] == "unknown"
    assert mapped["source_sample_mapping_reason"] == "epoch_clock_bounds_outside_original_source"


def test_unsupported_source_grid_keeps_array_bounds_but_refuses_mapping():
    mapped = map_preprocessed_window_to_source_samples(
        40,
        80,
        preprocessed_sfreq_hz=10.0,
        provenance={},
        allow_source_grid=False,
        refusal_reason="flattened_eeglab_epochs_have_no_continuous_source_sample_map",
    )
    assert mapped["source_sample_mapping_status"] == "unknown"
    assert mapped["source_sample_mapping_reason"] == "flattened_eeglab_epochs_have_no_continuous_source_sample_map"
    assert mapped["preprocessed_sample_start"] == 40
    assert mapped["preprocessed_sample_end_exclusive"] == 80


def test_export_persists_explicit_source_samples_and_prefers_sample_clock_over_crop_offset(tmp_path):
    rec = record(tmp_path, "mapped.edf", 10.0)
    rec["raw_clock_before_crop"] = {
        "sfreq_hz": 100.0,
        "first_time_sec": 100.0,
        "first_samp": 10_000,
        "n_times": 10_000,
    }
    rec["raw_clock_after_crop"] = {
        "sfreq_hz": 50.0,
        "first_time_sec": 110.0,
        "first_samp": 5_500,
        "n_times": 500,
    }
    frame = pd.DataFrame({
        "file": ["mapped.edf"],
        "epoch_id": [3],
        "t_start": [0.0],
        "t_end": [4.0],
        "preprocessed_sample_start": [50],
        "preprocessed_sample_end_exclusive": [250],
        "preprocessed_sfreq_hz": [50.0],
        "nominal_source_sample_start": [1_000],
        "nominal_source_sample_end_exclusive": [1_400],
        "nominal_source_sample_absolute_start": [11_000],
        "nominal_source_sample_absolute_end_exclusive": [11_400],
        "source_first_samp": [10_000],
        "source_sfreq_hz": [100.0],
        "nominal_source_time_start_sec": [10.0],
        "nominal_source_time_end_sec": [14.0],
        "source_sample_mapping_status": ["nominal_bounds_mapped"],
        "source_sample_mapping_reason": ["mapped_from_extractor_indices"],
        "actual_sample_contributors_status": ["unknown"],
        "input_sample_finite_status": ["all_finite"],
        "input_sample_finite_fraction": [1.0],
        "input_sample_gap_count": [1],
        "input_sample_gap_spans": ['[{"preprocessed_sample_start":70,"preprocessed_sample_end_exclusive":72,"source_sample_mapping_status":"nominal_bounds_mapped","nominal_source_sample_start":1040,"nominal_source_sample_end_exclusive":1044}]'],
    })
    row = build_signal_support_export(frame, [rec])["per_epoch_input_extent"][0]
    assert row["original_sample_start"] == 1_000
    assert row["original_sample_end"] == 1_400
    assert row["sample_mapping_status"] == "nominal_bounds_mapped"
    assert row["nominal_original_time_start_sec"] == 10.0
    assert row["nominal_original_time_end_sec"] == 14.0
    assert row["source_file_sha256"] == rec["source"]["sha256"]
    assert row["recording_id"] == "mapped"
    assert row["source_sample_rate_hz"] == 100.0
    assert row["source_first_samp"] == 10_000
    assert row["source_sample_absolute_start"] == 11_000
    assert row["actual_sample_contributors_status"] == "unknown"
    assert row["input_sample_finite_status"] == "all_finite"
    assert row["input_sample_finite_fraction"] == 1.0
    assert row["input_sample_gap_count"] == 1
    assert row["input_sample_gap_spans"][0]["nominal_source_sample_start"] == 1040
    assert row["support_status"] == "unknown"
    assert len(row["source_quality_overlap"]) == 1


def test_duplicate_raw_basenames_fail_closed_without_mixing_provenance(tmp_path):
    records = []
    for directory, contents in (("recording-a", b"a"), ("recording-b", b"b")):
        path = tmp_path / directory / "same.edf"
        path.parent.mkdir(parents=True)
        path.write_bytes(contents)
        rec = build_signal_support_provenance(path)
        refresh_source_hash(rec)
        rec.update(
            raw_file="same.edf",
            source_quality_intervals={"status": "observed", "intervals": [
                {"clock": "original_raw_seconds", "start_sec": 0.0, "end_sec": 5.0}
            ]},
        )
        records.append(rec)
    frame = pd.DataFrame({"file": ["same.edf"], "epoch_id": [0], "t_start": [0.0], "t_end": [2.0]})
    result = build_signal_support_export(frame, records)
    extent = result["per_epoch_input_extent"][0]
    assert extent["source_identity"] is None
    assert extent["source_file_sha256"] is None
    assert extent["sample_mapping_status"] == "unknown"
    assert extent["sample_mapping_reason"] == "support_basename_ambiguous"
    assert extent["source_quality_overlap"] == []
    assert result["source_quality_intervals"] == []
    assert {record["validation_reason"] for record in result["execution_records"]} == {"support_basename_ambiguous"}


def test_legacy_crop_offset_is_made_relative_to_nonzero_source_first_time(tmp_path):
    rec = record(tmp_path, "legacy.edf", 10.0)
    rec["raw_clock_before_crop"] = {"first_time_sec": 100.0, "sfreq_hz": 100.0, "first_samp": 10_000, "n_times": 10_000}
    rec["crop"]["actual_tmin_sec"] = 110.0
    frame = pd.DataFrame({"file": ["legacy.edf"], "t_start": [0.0], "t_end": [5.0]})
    row = build_signal_support_export(frame, [rec])["per_epoch_input_extent"][0]
    assert row["nominal_original_time_start_sec"] == 10.0
    assert row["nominal_original_time_end_sec"] == 15.0
    assert len(row["source_quality_overlap"]) == 1


def test_flattened_eeglab_epochs_do_not_fall_back_to_continuous_crop_time(tmp_path):
    rec = record(tmp_path, "epoched.set", 10.0)
    rec["crop"] = {
        "applied": False,
        "reason": "not_configured",
        "source_sample_mapping_status": "unknown",
        "source_sample_mapping_reason": "flattened_eeglab_epochs_have_no_continuous_source_sample_map",
    }
    rec["feature_time_clock"] = {"reference": "preprocessed_array_start"}
    rec["source_quality_intervals"] = {"status": "observed", "intervals": [
        {"clock": "original_raw_seconds", "start_sec": 0.0, "end_sec": 50.0}
    ]}
    frame = pd.DataFrame({
        "file": ["epoched.set"], "epoch_id": [0], "t_start": [0.0], "t_end": [8.0],
        "source_sample_mapping_status": ["unknown"],
        "source_sample_mapping_reason": ["flattened_eeglab_epochs_have_no_continuous_source_sample_map"],
    })
    row = build_signal_support_export(frame, [rec])["per_epoch_input_extent"][0]
    assert row["nominal_original_time_start_sec"] is None
    assert row["nominal_original_time_end_sec"] is None
    assert row["sample_mapping_status"] == "unknown"
    assert row["sample_mapping_reason"] == "flattened_eeglab_epochs_have_no_continuous_source_sample_map"
    assert row["source_quality_overlap"] == []


def test_failed_feature_clock_mapping_does_not_fall_back_to_crop_offset(tmp_path):
    rec = record(tmp_path, "bad-clock.edf", 10.0)
    rec["feature_time_clock"] = {"reference": "preprocessed_array_start"}
    rec["crop"] = {"applied": True, "actual_tmin_from_source_start_sec": 100.0}
    frame = pd.DataFrame({
        "file": ["bad-clock.edf"], "t_start": [0.0], "t_end": [8.0],
        "source_sample_mapping_status": ["unknown"],
        "source_sample_mapping_reason": ["source_clock_invalid:KeyError"],
    })
    row = build_signal_support_export(frame, [rec])["per_epoch_input_extent"][0]
    assert row["nominal_original_time_start_sec"] is None
    assert row["nominal_original_time_end_sec"] is None
    assert row["sample_mapping_reason"] == "source_clock_invalid:KeyError"
    assert row["source_quality_overlap"] == []


def test_wfdb_support_requires_stable_associated_signal_payload_hash(tmp_path):
    rec = record(tmp_path, "record.hea", 10.0)
    rec["source"]["associated_signal_files"] = [{
        "path": str(tmp_path / "record.mat"),
        "exists": True,
        "sha256": "a" * 64,
        "sha256_after": "a" * 64,
        "hash_stable": True,
    }]
    assert validate_support_record(rec, "record.hea") == (True, None)

    rec["source"]["associated_signal_files"][0]["sha256_after"] = "b" * 64
    assert validate_support_record(rec, "record.hea") == (False, "support_associated_signal_hash_unstable")

    rec["source"]["associated_signal_files"] = []
    assert validate_support_record(rec, "record.hea") == (False, "support_associated_signal_identity_missing")


def test_summary_path_marks_duplicate_basename_unknown_before_sidecar_read(tmp_path):
    import json
    from mndm.pipeline.summary import SubjectSummaryRunner

    qc_dir = tmp_path / "qc_artifacts"
    qc_dir.mkdir()
    (qc_dir / "same_qc_artifacts.json").write_text(json.dumps({
        "signal_support_provenance": {
            "schema": "mndm.signal_support_provenance.v1",
            "source": {"path": "a/same.edf", "exists": True, "sha256": "a" * 64,
                       "sha256_after": "a" * 64, "hash_stable": True},
        }
    }), encoding="utf-8")
    runner = SubjectSummaryRunner.__new__(SubjectSummaryRunner)
    runner.ds_path = tmp_path
    runner._index_paths_by_basename = {"same.edf": ["a/same.edf", "b/same.edf"]}
    runner._lookup_rel_paths_by_file_value = lambda _: ["a/same.edf", "b/same.edf"]
    records = runner._signal_support_provenance(pd.DataFrame({"file": ["same.edf"]}))
    assert records["records"][0]["status"] == "unknown"
    assert records["records"][0]["reason"] == "duplicate_raw_basename_in_file_index"

    exported = build_signal_support_export(
        pd.DataFrame({"file": ["same.edf"], "t_start": [0.0], "t_end": [8.0]}),
        records["records"],
    )["per_epoch_input_extent"][0]
    assert exported["source_identity"] is None
    assert exported["sample_mapping_status"] == "unknown"
    assert exported["sample_mapping_reason"] == "duplicate_raw_basename_in_file_index"


def test_sample_support_metadata_is_not_a_projection_feature():
    from mndm.projection import select_export_feature_columns

    frame = pd.DataFrame({
        "eeg_alpha": [1.0],
        "nominal_source_sample_start": [1_100],
        "source_sample_absolute_start": [1_600],
        "source_sfreq_hz": [100.0],
        "preprocessed_sample_start": [50],
        "input_sample_finite_fraction": [1.0],
        "technical_eeg_finite_fraction": [1.0],
        "coma_proxy_threshold_value": [0.01],
        "eeg_conventional_coma_suppression_ratio": [0.2],
        "eeg_conventional_coma_amplitude_rms": [10.0],
    })
    assert select_export_feature_columns(frame) == ["eeg_alpha"]
