"""Independent contract tests for executed preprocessing provenance."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from mndm.signal_support_provenance import (
    SCHEMA_VERSION,
    actual_crop_bounds,
    build_signal_support_provenance,
    finish_operation,
    start_operation,
    quality_interval,
    original_raw_annotation_start,
)
from mndm.features.ppg import compute_ppg_features


def test_source_identity_and_unknown_support_are_serializable(tmp_path: Path) -> None:
    source = tmp_path / "synthetic.edf"
    source.write_bytes(b"synthetic-signal-header-and-payload")

    provenance = build_signal_support_provenance(
        source,
        input_sfreq=np.float64(256.0),
        target_sfreq=np.float64(250.0),
        source_datatype="eeg",
    )
    operation = start_operation(
        provenance,
        "bandpass",
        input_sfreq=256.0,
        parameters={"bandpass_hz": [1.0, 45.0]},
        actual_kwargs={"picks": np.int64(3), "path": source},
    )
    finish_operation(operation, status="applied", output_sfreq=250.0)

    assert provenance["schema"] == SCHEMA_VERSION
    assert provenance["source"]["sha256"]
    assert provenance["operations"][0]["status"] == "applied"
    assert provenance["operations"][0]["actual_kwargs"]["picks"] == 3
    assert provenance["temporal_support"]["status"] == "unknown"
    assert provenance["temporal_support"]["epoch_bounds"] == "not_recorded_here"
    json.dumps(provenance, allow_nan=False)

    nonfinite = start_operation(provenance, "edge_values", parameters={"nan": math.nan, "inf": math.inf})
    assert nonfinite["parameters"] == {"nan": None, "inf": None}


def test_failed_and_fallback_operations_remain_explicit(tmp_path: Path) -> None:
    source = tmp_path / "synthetic.edf"
    source.write_bytes(b"source")
    provenance = build_signal_support_provenance(source)

    failed = start_operation(provenance, "notch_filter")
    finish_operation(failed, status="failed", error=ValueError("edge failure"))
    fallback = start_operation(provenance, "eeg_bandpass")
    finish_operation(
        fallback,
        status="fallback",
        fallback="mne.filter.filter_data",
        error="raw.filter failed",
    )

    assert provenance["operations"][0]["status"] == "failed"
    assert "edge failure" in provenance["operations"][0]["error"]
    assert provenance["operations"][1]["status"] == "fallback"
    assert provenance["operations"][1]["fallback"] == "mne.filter.filter_data"


def test_spatial_operation_does_not_claim_temporal_support(tmp_path: Path) -> None:
    source = tmp_path / "synthetic.edf"
    source.write_bytes(b"source")
    provenance = build_signal_support_provenance(source)
    operation = start_operation(provenance, "eeg_csd", operation_class="spatial")
    finish_operation(operation, status="skipped", fallback="insufficient_scalp_channels")

    assert operation["temporal_support_status"] == "not_applicable"
    assert operation["status"] == "skipped"


def test_non_grid_crop_reports_actual_sample_bounds() -> None:
    bounds = actual_crop_bounds(
        original_first_samp=100,
        actual_first_samp=113,
        actual_n_times=37,
        sfreq_hz=10.0,
        original_first_time_sec=2.0,
    )
    assert bounds["actual_last_samp_exclusive"] == 150
    assert bounds["actual_tmin_sec"] == 3.3
    assert bounds["actual_tmax_sec"] == 7.0


def test_quality_intervals_preserve_source_and_reject_invalid_values() -> None:
    interval = quality_interval(
        source_kind="source_stage_hum_code_8",
        start_sec=30.0,
        duration_sec=30.0,
        source_column="stage_hum",
        stage_code=8,
    )
    assert interval["end_sec"] == 60.0
    assert interval["clock"] == "original_raw_seconds"
    try:
        quality_interval(source_kind="bad", start_sec=0.0, duration_sec=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative source interval was accepted")


def test_nonzero_mne_first_time_is_removed_from_annotation_onset() -> None:
    # MNE annotations are expressed in the raw time frame; a nonzero
    # first_samp contributes Raw.first_time and must not become file-relative.
    os.environ.setdefault("MNE_USE_NUMBA", "false")
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    import mne

    info = mne.create_info(["x"], 10.0, ["eeg"])
    raw = mne.io.RawArray(np.zeros((1, 100)), info, first_samp=50)
    raw.set_annotations(mne.Annotations([0.0], [1.0], ["BAD_synthetic"]))
    assert raw.first_time == 5.0
    assert original_raw_annotation_start(float(raw.annotations.onset[0]), raw.first_time) == 0.0


def test_ppg_records_filter_fallback_and_global_peak_threshold() -> None:
    sfreq = 50.0
    time = np.arange(6000, dtype=float) / sfreq
    # A deterministic pulse-like signal gives several epochs and peaks.
    pulse = np.sin(2.0 * np.pi * 1.2 * time) + 0.05 * np.sin(2.0 * np.pi * 0.2 * time)
    signals = {
        "signals": {"ppg": pulse[None, :]},
        "sfreq": sfreq,
        "dataset_id": "synthetic",
        "meta": {},
    }
    config = {
        "epoching": {"length_s": 30.0, "step_s": 30.0},
        "features": {"ppg": {"bandpass_low_hz": 0.4, "bandpass_high_hz": 8.0, "bandpass_order": 3}},
    }
    result = compute_ppg_features(signals, config)
    provenance = result.attrs["signal_support_provenance"]
    operations = {item["name"]: item for item in provenance["operations"]}
    assert operations["ppg_bandpass_filtfilt"]["status"] == "applied"
    assert operations["ppg_peak_detection"]["status"] == "applied"
    assert operations["ppg_peak_detection"]["parameters"]["threshold_fit_scope"] == "whole_stream_median_and_MAD"
    assert provenance["temporal_support"]["status"] == "unknown"
    assert len(result) == 4
