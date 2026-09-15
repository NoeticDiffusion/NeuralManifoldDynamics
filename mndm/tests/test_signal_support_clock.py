"""Cross-file and cropped nominal quality overlap, never effective support."""
import numpy as np
import pandas as pd
from mndm.signal_support_provenance import build_signal_support_provenance, refresh_source_hash
from mndm.pipeline.signal_support_export import build_signal_support_export


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
