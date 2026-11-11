"""Behavioral tests for the ordinary EEG pipeline's separate v3.1 surface."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import json
from types import SimpleNamespace

import h5py
import numpy as np

os.environ.setdefault("MNE_USE_NUMBA", "false")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parents[2] / "core" / "src"))

from mndm.mnps.v3_1.pipeline_measurements import (
    compute_acquisition_qc,
    compute_pipeline_m2_temporal_measurement,
    compute_pipeline_recording_level_experimental_measurements,
    compute_pipeline_window_measurements,
    _m1_window_local_filter_support_qc,
)
from mndm.mnps.v3_1.realizations import CORE19_CHANNELS


def _synthetic_window(seed: int = 901, *, sample_rate_hz: float = 128.0, duration_sec: float = 4.0):
    rng = np.random.default_rng(seed)
    labels = ("F3", "F4", "C3", "C4")
    sample_count = int(sample_rate_hz * duration_sec)
    time = np.arange(sample_count) / sample_rate_hz
    phase = np.linspace(0.1, 1.1, len(labels))
    values = np.asarray([
        np.sin(2 * np.pi * 10.0 * time + offset) + 0.3 * rng.normal(size=sample_count)
        for offset in phase
    ])
    return labels, values


def test_m1_candidate_edge_support_remains_unknown_when_upstream_filter_support_is_true():
    qc = _m1_window_local_filter_support_qc(
        sample_count=512,
        sample_rate_hz=128.0,
        window_start_sec=4.0,
        window_end_sec=8.0,
        upstream_support_qc={"effective_filter_support_valid": True},
    )

    assert qc["upstream_preprocessing_filter_support_valid"] is True
    assert qc["candidate_filter_edge_support_valid"] is None
    assert qc["exact_finite_edge_safe_support_established"] is False


def test_pipeline_measurements_keep_names_native_support_and_qualification_separate():
    labels, values = _synthetic_window()
    support_qc = {
        "source_is_continuous": None,
        "timestamps_uniform": True,
        "gap_free": True,
        "bad_interval_free": None,
        "within_recording_bounds": True,
        "effective_filter_support_valid": None,
    }
    branch = {
        "branch_id": "synthetic-preprocessed-eeg",
        "reference_mode": "source_reference_as_loaded",
        "reference_set_sha256": None,
        "filter_band_hz": [1.0, 45.0],
    }

    result = compute_pipeline_window_measurements(
        values,
        labels,
        sample_rate_hz=128.0,
        window_start_sec=12.0,
        window_end_sec=16.0,
        segment_id="synthetic-run",
        source_identity="test-source",
        input_branch=branch,
        support_qc=support_qc,
        sequence_complexity_reference_count=8,
        activity_block_sec=1.0,
    )

    rows = {row["candidate_id"]: row for row in result["measurements"]}
    assert set(rows) == {"M1", "M2", "M3", "D1", "D3", "E1", "E2", "E3"}
    assert "D2" not in rows
    assert result["support_audit_status"] == "incomplete"
    assert result["realization"]["surface_id"] == "native"
    assert result["realization"]["channel_count"] == 4
    assert rows["M1"]["construct_name"] == "Phase Coordination Variability"
    assert rows["M1"]["construct_id"] == "phase_coordination_variability"
    assert rows["E2"]["construct_name"] == "Sequence Complexity"
    assert rows["E3"]["construct_name"] == "Effective Dimensionality"
    assert all(row["qualified"] is False for row in rows.values())
    assert all(row["computed"] for row in rows.values())
    assert rows["M1"]["details"]["support_qc"]["effective_filter_support_valid"] is None
    assert rows["M1"]["details"]["filter_support_qc_scope"].startswith("upstream_support_is_separate")
    local_filter_qc = rows["M1"]["details"]["filter_support_qc"]
    assert local_filter_qc["input_sample_count"] == values.shape[-1]
    assert local_filter_qc["external_left_context_samples_used"] == 0
    assert local_filter_qc["external_right_context_samples_used"] == 0
    assert local_filter_qc["padlen_samples"] == 0
    assert local_filter_qc["edge_samples_excluded"] == 0
    assert local_filter_qc["computational_support_complete_for_current_window"] is True
    assert local_filter_qc["full_declared_window_supplied_to_estimator"] is True
    assert "does not establish edge fidelity" in local_filter_qc["computational_support_scope"]
    assert local_filter_qc["analytic_transform_dependency"] == (
        "full_supplied_window_fft_with_periodic_boundary_assumption"
    )
    assert local_filter_qc["exact_finite_edge_safe_support_established"] is False
    assert local_filter_qc["candidate_filter_edge_support_valid"] is None
    assert local_filter_qc["upstream_preprocessing_filter_support_valid"] is None
    assert local_filter_qc["edge_effects_assessed_or_corrected"] is False
    assert np.isfinite(rows["M1"]["value"])
    assert isinstance(rows["M3"]["value"], list) and len(rows["M3"]["value"]) == 3

    separate_m2 = compute_pipeline_m2_temporal_measurement(
        values,
        labels,
        sample_rate_hz=128.0,
        window_start_sec=12.0,
        window_end_sec=16.0,
        segment_id="synthetic-run",
        input_branch=branch,
        support_qc=support_qc,
    )
    assert separate_m2 == rows["M2"]


def test_pipeline_window_output_is_identical_with_binary_lz78_fast_path(monkeypatch):
    from mndm.mnps.v3_1 import pipeline_measurements as measurement_module
    from mndm.mnps.v3_1.eeg_metrics import _lz78_phrase_count

    labels, values = _synthetic_window(seed=410018)
    common = dict(
        channel_names=labels,
        sample_rate_hz=128.0,
        window_start_sec=12.0,
        window_end_sec=16.0,
        segment_id="synthetic-run",
        source_identity="pipeline-speed-001-parity",
        input_branch={
            "branch_id": "synthetic-preprocessed-eeg",
            "reference_mode": "source_reference_as_loaded",
            "reference_set_sha256": None,
            "filter_band_hz": [1.0, 45.0],
        },
        support_qc={"gap_free": True, "effective_filter_support_valid": None},
        sequence_complexity_reference_count=8,
        activity_block_sec=1.0,
    )
    optimized = compute_pipeline_window_measurements(values, **common)

    def generic_binary_reference(symbols):
        return _lz78_phrase_count(tuple(int(value) for value in symbols))

    monkeypatch.setattr(measurement_module, "_lz78_phrase_count_binary", generic_binary_reference)
    reference = compute_pipeline_window_measurements(values, **common)

    assert optimized == reference


def test_optional_metric_timing_callback_preserves_exact_window_output():
    labels, values = _synthetic_window(seed=410019)
    common = dict(
        channel_names=labels,
        sample_rate_hz=128.0,
        window_start_sec=12.0,
        window_end_sec=16.0,
        segment_id="synthetic-run",
        source_identity="pipeline-speed-002-timing-parity",
        input_branch={
            "branch_id": "synthetic-preprocessed-eeg",
            "reference_mode": "source_reference_as_loaded",
            "reference_set_sha256": None,
            "filter_band_hz": [1.0, 45.0],
        },
        support_qc={"gap_free": True, "effective_filter_support_valid": None},
        sequence_complexity_reference_count=8,
        activity_block_sec=1.0,
    )
    baseline = compute_pipeline_window_measurements(values, **common)
    timing_rows = []
    instrumented = compute_pipeline_window_measurements(
        values,
        **common,
        metric_timing_callback=lambda candidate, elapsed: timing_rows.append((candidate, elapsed)),
    )

    assert instrumented == baseline
    assert {name for name, _ in timing_rows} == {
        "PCV", "TP_main_window", "AR", "NIE", "RD", "OI", "SC", "ED",
    }
    assert all(elapsed >= 0.0 for _, elapsed in timing_rows)


def test_pipeline_measurements_do_not_replace_nonfinite_input_with_values():
    labels, values = _synthetic_window()
    values[0, 20] = np.nan

    result = compute_pipeline_window_measurements(
        values,
        labels,
        sample_rate_hz=128.0,
        window_start_sec=0.0,
        window_end_sec=4.0,
        segment_id="synthetic-run",
        source_identity="test-source",
        input_branch={"branch_id": "test"},
        support_qc={"gap_free": False},
    )

    assert all(row["computed"] is False for row in result["measurements"])
    assert all(row["value"] is None for row in result["measurements"])
    assert {row["reason"] for row in result["measurements"]} == {"non_finite_input_samples"}


def test_recording_scg_emits_separate_core19_and_native_experimental_rows():
    sfreq = 128.0
    sample_count = int(sfreq * 48.0)
    time = np.arange(sample_count, dtype=np.float64) / sfreq
    rng = np.random.default_rng(441)
    labels = tuple(CORE19_CHANNELS)
    values = np.asarray([
        np.sin(2.0 * np.pi * 10.0 * time + (index * 0.17))
        + 0.2 * rng.normal(size=sample_count)
        for index in range(len(labels))
    ])
    result = compute_pipeline_recording_level_experimental_measurements(
        values,
        labels,
        sample_rate_hz=sfreq,
        support_qc={
            "source_is_continuous": True,
            "timestamps_uniform": True,
            "gap_free": True,
            "bad_interval_free": True,
            "within_recording_bounds": True,
        },
        source_identity="synthetic-source",
        dataset_id="synthetic",
        input_branch={"branch_id": "synthetic-recording"},
        signal_units="microvolt",
        include_core19=True,
        include_ds004148_core60=False,
    )

    assert result["schema"] == "mndm.mnps_v3_1.recording_level_experimental.v1"
    assert result["qualified"] is False
    assert result["aggregation_level"] == "recording"
    assert result["step_sec"] is None
    assert result["surface_count"] == 2
    assert [surface["surface_id"] for surface in result["surfaces"]] == ["core19", "native"]
    for surface in result["surfaces"]:
        measurements = {row["estimator_id"]: row for row in surface["measurements"]}
        assert measurements["spatial_dwpli_distance_spearman_alpha.v1"]["computed"] is True
        assert measurements["spatial_dwpli_distance_spearman_alpha.v1"]["qualified"] is False
        assert measurements[
            "local_distributed_coupling_contrast_distance_quartiles_dwpli2_alpha.v1"
        ]["computed"] is True
        assert surface["support_window_sec"] == 48.0


def test_recording_scg_computes_available_small_native_surface_with_channel_warning():
    sfreq = 128.0
    sample_count = int(sfreq * 48.0)
    time = np.arange(sample_count, dtype=np.float64) / sfreq
    rng = np.random.default_rng(442)
    labels = tuple(CORE19_CHANNELS[:4])
    values = np.asarray([
        np.sin(2.0 * np.pi * 10.0 * time + (index * 0.3))
        + 0.2 * rng.normal(size=sample_count)
        for index in range(len(labels))
    ])
    result = compute_pipeline_recording_level_experimental_measurements(
        values,
        labels,
        sample_rate_hz=sfreq,
        support_qc={
            "source_is_continuous": True,
            "timestamps_uniform": True,
            "gap_free": True,
            "bad_interval_free": True,
            "within_recording_bounds": True,
        },
        source_identity="synthetic-source",
        dataset_id="synthetic",
        input_branch={"branch_id": "synthetic-recording"},
        include_core19=True,
        include_ds004148_core60=False,
    )

    assert result["computed_scg_surface_count"] == 2
    assert result["surfaces"][0]["surface_id"] == "core19_available_4"
    assert result["surfaces"][0]["measurements"][0]["computed"] is True
    assert "below_core19_channel_count" in result["surfaces"][0]["surface_qc"]["warnings"]


def test_recording_scg_computes_with_unknown_support_and_omits_only_nonfinite_channel():
    sfreq = 128.0
    sample_count = int(sfreq * 48.0)
    time = np.arange(sample_count, dtype=np.float64) / sfreq
    rng = np.random.default_rng(443)
    labels = tuple(CORE19_CHANNELS[:4])
    values = np.asarray([
        np.sin(2.0 * np.pi * 10.0 * time + (index * 0.31))
        + 0.2 * rng.normal(size=sample_count)
        for index in range(len(labels))
    ])
    values[0, 100] = np.nan

    result = compute_pipeline_recording_level_experimental_measurements(
        values,
        labels,
        sample_rate_hz=sfreq,
        support_qc={
            "source_is_continuous": True,
            "timestamps_uniform": True,
            "gap_free": None,
            "bad_interval_free": None,
            "within_recording_bounds": True,
            "effective_filter_support_valid": None,
        },
        source_identity="synthetic-source-unknown-support",
        dataset_id="synthetic",
        input_branch={"branch_id": "synthetic-recording"},
        include_core19=True,
        include_ds004148_core60=False,
    )

    core19, native = result["surfaces"]
    for surface in (core19, native):
        scg, _ldcb = surface["measurements"]
        assert scg["computed"] is True
        assert scg["qualified"] is False
        assert len(surface["effective_channel_labels"]) == 3
        assert "Fp1" in surface["surface_qc"]["omitted_nonfinite_channel_labels"]
        assert "recording_support_audit_incomplete" in surface["surface_qc"]["warnings"]
        assert "gap_free" in scg["details"]["support_audit_unknown_fields"]


def test_recording_scg_still_rejects_known_discontinuity_with_other_support_unknown():
    sfreq = 128.0
    sample_count = int(sfreq * 48.0)
    time = np.arange(sample_count, dtype=np.float64) / sfreq
    labels = tuple(CORE19_CHANNELS[:4])
    values = np.asarray([
        np.sin(2.0 * np.pi * 10.0 * time + (index * 0.29))
        for index in range(len(labels))
    ])
    result = compute_pipeline_recording_level_experimental_measurements(
        values,
        labels,
        sample_rate_hz=sfreq,
        support_qc={
            "source_is_continuous": False,
            "timestamps_uniform": True,
            "gap_free": None,
            "bad_interval_free": None,
            "within_recording_bounds": True,
            "effective_filter_support_valid": None,
        },
        source_identity="synthetic-discontinuous-source",
        dataset_id="synthetic",
        input_branch={"branch_id": "synthetic-recording"},
        include_core19=True,
        include_ds004148_core60=False,
    )

    assert all(surface["measurements"][0]["computed"] is False for surface in result["surfaces"])
    assert all(surface["measurements"][0]["reason"] == "non_continuous_source" for surface in result["surfaces"])


def test_pipeline_measurement_estimator_failures_are_candidate_local(monkeypatch):
    import mndm.mnps.v3_1.pipeline_measurements as module

    labels, values = _synthetic_window(duration_sec=8.0)

    def broken_corrcoef(*_args, **_kwargs):
        raise RuntimeError("synthetic D1 failure")

    monkeypatch.setattr(module.np, "corrcoef", broken_corrcoef)
    result = compute_pipeline_window_measurements(
        values,
        labels,
        sample_rate_hz=128.0,
        window_start_sec=0.0,
        window_end_sec=8.0,
        segment_id="synthetic-run",
        source_identity="test-source",
        input_branch={"branch_id": "test"},
        support_qc={"gap_free": None},
        sequence_complexity_reference_count=4,
        activity_block_sec=1.0,
    )

    rows = {row["candidate_id"]: row for row in result["measurements"]}
    assert rows["D1"]["computed"] is False
    assert rows["D1"]["reason"] == "candidate_estimator_exception"
    assert rows["D1"]["details"]["exception_type"] == "RuntimeError"
    assert rows["M2"]["computed"] is True
    assert rows["D3"]["computed"] is True
    assert rows["E3"]["computed"] is True


def test_acquisition_qc_reports_frequency_support_and_descriptive_noise_ratio():
    sfreq = 256.0
    time = np.arange(int(sfreq * 20.0)) / sfreq
    clean = np.sin(2 * np.pi * 10.0 * time)
    with_line = np.asarray([
        clean + 0.1 * np.sin(2 * np.pi * 50.0 * time + phase)
        for phase in (0.0, 0.4, 0.8)
    ])
    no_line = with_line - 0.1 * np.sin(2 * np.pi * 50.0 * time)[None, :]

    original = with_line.copy()
    raw_qc = compute_acquisition_qc(
        with_line,
        sample_rate_hz=sfreq,
        stage="synthetic_prefilter",
        amplitude_unit="microvolt",
    )
    filtered_qc = compute_acquisition_qc(
        no_line,
        sample_rate_hz=sfreq,
        stage="synthetic_postfilter",
        amplitude_unit="microvolt",
    )
    unsupported = compute_acquisition_qc(
        with_line,
        sample_rate_hz=100.0,
        stage="below_line_band_nyquist",
        amplitude_unit="microvolt",
    )

    assert raw_qc["line_noise_48_52_ratio_status"] == "computed"
    assert raw_qc["line_noise_48_52_ratio"] > filtered_qc["line_noise_48_52_ratio"]
    assert raw_qc["broadband_noise_proxy_status"] == "computed"
    assert np.array_equal(with_line, original)
    assert unsupported["line_noise_48_52_ratio"] is None
    assert unsupported["line_noise_48_52_ratio_status"] == "unsupported"


def test_acquisition_qc_preserves_psd_input_units_and_microvolt_aliases():
    sfreq = 256.0
    time = np.arange(1024) / sfreq
    data_uv = np.sin(2 * np.pi * 37.0 * time)[None, :]
    qc_v = compute_acquisition_qc(data_uv * 1e-6, sample_rate_hz=sfreq,
                                 stage="synthetic", amplitude_unit="volt")
    # A unit-amplitude sinusoid has integrated power 0.5 in the Welch PSD.
    for unit in ("microvolt", "uv", "µv", "μv"):
        qc_uv = compute_acquisition_qc(data_uv, sample_rate_hz=sfreq,
                                      stage="synthetic", amplitude_unit=unit)
        assert np.isclose(qc_uv["broadband_noise_proxy"], 0.5 / (61 * 0.25))
        assert np.isclose(qc_uv["broadband_noise_proxy"],
                          qc_v["broadband_noise_proxy"] * 1e12)
        assert qc_uv["broadband_noise_proxy_units"] == "microvolt_squared_per_hz"
        assert np.isclose(qc_uv["line_noise_48_52_ratio"], qc_v["line_noise_48_52_ratio"],
                          atol=1e-20)


def test_standard_eeg_extractor_carries_v3_1_rows_without_adding_feature_columns():
    from mndm.features.eeg import compute_eeg_features

    labels, values = _synthetic_window(sample_rate_hz=128.0, duration_sec=4.0)
    signals = {
        "signals": {"eeg": values},
        "sfreq": 128.0,
        "channels": {"eeg": list(labels)},
        "dataset_id": "synthetic_v31",
        "file_path": "synthetic.edf",
        "meta": {
            "file": "synthetic.edf",
            "eeg_reference_mode": "average",
            "eeg_reference_status": "verified",
            "reference_set_hash": "a" * 64,
            "channel_set_hash": "b" * 64,
        },
    }
    config = {
        "epoching": {"length_s": 4.0, "step_s": 4.0},
        "features": {"eeg_psd": {"method": "welch"}, "eeg_bands": {"alpha": [8.0, 12.0]}},
        "mnps_v3_1_measurements": {
            "enabled": True,
            "sequence_complexity_reference_count": 8,
            "activity_block_sec": 1.0,
        },
    }

    frame = compute_eeg_features(signals, config)

    assert len(frame) == 1
    assert not any(str(column).startswith(("phase_coordination_variability", "sequence_complexity")) for column in frame.columns)
    exported = frame.attrs["mnps_v3_1_measurements"]
    assert exported["schema"] == "mndm.mnps_v3_1.pipeline_measurements.v1"
    assert len(exported["windows"]) == 1
    window = exported["windows"][0]
    assert window["source_sample_mapping"]["source_sample_mapping_status"] == "unknown"
    assert window["source_identity"]["identity_basis"] == "dataset_and_source_path_sha256"
    assert exported["source_identity"]["source_identity_id"] == window["source_identity"]["source_identity_id"]
    assert window["support_qc"]["gap_free"] is None
    assert window["support_qc"]["bad_interval_free"] is None
    assert window["support_qc"]["timestamps_uniform"] is None
    assert window["support_qc"]["effective_filter_support_valid"] is None
    assert {row["candidate_id"] for row in exported["windows"][0]["measurements"]} == {
        "M1", "M2", "M3", "D1", "D3", "E1", "E2", "E3"
    }
    assert "D2" not in {row["candidate_id"] for row in exported["windows"][0]["measurements"]}
    recording_d2 = exported["recording_level_experimental_measurements"]
    assert recording_d2["schema"] == "mndm.mnps_v3_1.recording_level_experimental.v1"
    assert recording_d2["aggregation_level"] == "recording"
    assert recording_d2["qualified"] is False
    assert all(not row["computed"] for surface in recording_d2["surfaces"] for row in surface["measurements"])


def test_m2_win4s_realization_is_separate_from_shared_measurement_grid():
    from mndm.features.eeg import compute_eeg_features

    labels, values = _synthetic_window(duration_sec=12.0)
    signals = {
        "signals": {"eeg": values},
        "sfreq": 128.0,
        "channels": {"eeg": list(labels)},
        "dataset_id": "synthetic_v31_temporal_realization",
        "file_path": "synthetic.edf",
        "meta": {
            "file": "synthetic.edf",
            "eeg_reference_mode": "average",
            "eeg_reference_status": "verified",
        },
    }
    config = {
        "epoching": {"length_s": 12.0, "step_s": 12.0},
        "features": {"eeg_psd": {"method": "welch"}},
        "mnps_v3_1_measurements": {
            "enabled": True,
            "window_length_sec": 20.0,
            "step_sec": 20.0,
            "recording_level_experimental": {"d2_scg": {"enabled": False}},
        },
    }

    frame = compute_eeg_features(signals, config)
    payload = frame.attrs["mnps_v3_1_measurements"]

    assert len(payload["windows"]) == 1
    assert payload["windows"][0]["window_end_sec"] == 12.0
    shared_m2 = next(row for row in payload["windows"][0]["measurements"] if row["candidate_id"] == "M2")
    assert shared_m2["details"]["measurement_window"]["sample_count"] == int(12.0 * 128.0)

    realization, = payload["temporal_realizations"]
    assert realization["realization_id"] == "temporal_persistence.win4s.v1"
    assert realization["candidate_id"] == "M2"
    assert realization["support_window_sec_requested"] == 4.0
    assert realization["step_sec_requested"] == 4.0
    assert realization["qualified"] is False
    assert realization["coordinate_assignment"] is None
    assert len(realization["windows"]) == 3
    assert [(window["window_start_sec"], window["window_end_sec"]) for window in realization["windows"]] == [
        (0.0, 4.0), (4.0, 8.0), (8.0, 12.0)
    ]
    assert all(window["measurements"][0]["candidate_id"] == "M2" for window in realization["windows"])
    assert all(window["measurements"][0]["qualified"] is False for window in realization["windows"])


def test_measurements_serialize_under_separate_hdf5_group(tmp_path: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))
    from core.io.h5_writer import write_h5
    from mndm.schema import MNPSPayload

    measurement_payload = {
        "schema": "mndm.mnps_v3_1.measurement_export.v1",
        "qualified": False,
        "coordinate_assignment": None,
        "recordings": [{
            "raw_file": "sub-01_task-rest_eeg.edf",
            "measurement_payload": {
                "windows": [{"measurements": [{"construct_id": "sequence_complexity", "value": 1.02}]}]
            },
        }],
    }
    payload = MNPSPayload(
        time=np.asarray([0.0, 1.0], dtype=np.float64),
        x=np.zeros((2, 3), dtype=np.float32),
        x_dot=np.zeros((2, 3), dtype=np.float32),
        measurement_exports={"schema_version": "mndm.measurements.v1", "v3_1": measurement_payload},
    )
    output = write_h5(tmp_path / "measurements.h5", "synthetic", payload)

    with h5py.File(output, "r") as h5:
        group = h5["measurements/v3_1"]
        assert group.attrs["_schema_version"] == "mndm.mnps_v3_1.measurement_export.v1"
        assert "recordings_json" in group
        recordings = json.loads(group["recordings_json"][()].decode("utf-8"))
        assert recordings[0]["measurement_payload"]["windows"][0]["measurements"][0]["construct_id"] == "sequence_complexity"
        assert "features_raw" not in h5


def test_hdf5_extensions_skip_absent_optional_values(tmp_path: Path):
    from core.io.h5_writer import _write_extensions_group

    output = tmp_path / "extensions.h5"
    with h5py.File(output, "w") as h5:
        _write_extensions_group(h5, {
            "conventional_eeg": {
                "row_level_status": {
                    "reason": None,
                    "rows": [{"status": "computed", "reason": None}],
                }
            }
        })

    with h5py.File(output, "r") as h5:
        row_status = h5["extensions/conventional_eeg/row_level_status"]
        assert "reason" not in row_status
        rows = json.loads(row_status["rows_json"][()].decode("utf-8"))
        assert rows == [{"status": "computed", "reason": None}]


def test_worker_and_summary_handoff_preserve_measurement_sidecar(tmp_path: Path, monkeypatch):
    import pandas as pd

    from mndm import parallel
    from mndm.pipeline.summary import SubjectSummaryRunner

    payload = {
        "schema": "mndm.mnps_v3_1.pipeline_measurements.v1",
        "status": "computed",
        "qualified": False,
        "windows": [{"epoch_id": 0, "measurements": [{"candidate_id": "E2", "value": 1.0}]}],
        "recording_level_experimental_measurements": {
            "schema": "mndm.mnps_v3_1.recording_level_experimental.v1",
            "historical_slot": "D2",
            "aggregation_level": "recording",
            "qualified": False,
            "surfaces": [{"surface_id": "core19", "measurements": [{"value": 0.1}]}],
        },
    }

    def fake_eeg_handler(_signals, _config):
        frame = pd.DataFrame({"epoch_id": [0], "t_start": [0.0], "t_end": [1.0]})
        frame.attrs["mnps_v3_1_measurements"] = payload
        return frame

    monkeypatch.setattr(parallel, "_get_modality_handlers", lambda: {"eeg": fake_eeg_handler})
    preprocessed = SimpleNamespace(
        signals={"eeg": np.ones((2, 8), dtype=np.float64)},
        sfreq=8.0,
        channels={"eeg": ["F3", "F4"]},
        meta={"file": "sub-01_task-rest_eeg.edf", "signal_support_provenance": {}},
    )
    frame, _ = parallel._compute_features_from_preprocessed(preprocessed, {})
    assert frame is not None
    assert frame.attrs["mnps_v3_1_measurements"] == payload

    qc_dir = tmp_path / "qc_artifacts"
    parallel.write_qc_json({"mnps_v3_1_measurements": payload}, qc_dir, "sub-01_task-rest_eeg.edf")
    assert json.loads((qc_dir / "sub-01_task-rest_eeg_qc_artifacts.json").read_text())["mnps_v3_1_measurements"] == payload

    runner = object.__new__(SubjectSummaryRunner)
    runner.ds_path = tmp_path
    runner._lookup_rel_paths_by_file_value = lambda _file_name: []
    sub_frame = pd.DataFrame({"file": ["sub-01_task-rest_eeg.edf"]})
    export = runner._mnps_v3_1_measurement_export(sub_frame)
    assert export is not None
    assert export["qualified"] is False
    assert "D2" in export["experimental_candidate_ids"]
    assert export["recordings"][0]["measurement_payload"] == payload
