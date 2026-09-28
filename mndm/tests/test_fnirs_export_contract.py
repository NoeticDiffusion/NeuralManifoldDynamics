"""Contract freeze tests for native fNIRS HDF5 output."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.pipeline.fnirs_export_contract import (  # noqa: E402
    FNIRS_H5_CHANNEL_MATRIX_FLOAT64,
    FNIRS_H5_CHANNEL_MATRIX_INT64,
    FNIRS_H5_CHANNEL_UTF8_AXIS,
    FNIRS_H5_CHANNEL_UTF8_MATRIX,
    FNIRS_H5_CONTRACT_VERSION,
    FNIRS_H5_REQUIRED_PATHS,
    FNIRS_H5_ROOT_ATTRIBUTES,
    FNIRS_H5_WINDOW_FLOAT64,
    FNIRS_H5_WINDOW_INT64,
    FNIRS_H5_WINDOW_INT8,
    FNIRS_H5_WINDOW_UINT8,
    FNIRS_H5_WINDOW_UTF8,
    FNIRS_ROW_SOURCE_SCHEMA_VERSION,
)


def test_fnirs_h5_contract_id_and_modality_separation_are_frozen():
    assert FNIRS_H5_CONTRACT_VERSION == "mndm.fnirs_native_h5_contract.v1"
    assert FNIRS_ROW_SOURCE_SCHEMA_VERSION == "mndm.row_source.v2"
    assert "coords_9d" not in FNIRS_H5_REQUIRED_PATHS
    assert "/mnps_3d" not in FNIRS_H5_REQUIRED_PATHS
    assert "/time" not in FNIRS_H5_REQUIRED_PATHS
    assert FNIRS_H5_ROOT_ATTRIBUTES == (
        "export_contract_version",
        "modality",
        "has_fnirs",
        "has_eeg",
        "has_meg",
        "is_simultaneous",
        "capability_status",
        "mnps_qualified",
        "measurement_validity",
        "qualification_status",
        "claim_status",
    )


def test_fnirs_h5_contract_freezes_window_and_channel_axes():
    assert FNIRS_H5_WINDOW_FLOAT64 == (
        "t_start_s", "t_end_s", "actual_t_start_s", "actual_t_end_s",
        "run_relative_t_start_s", "run_relative_t_end_s",
        "actual_covered_duration_s", "clock_support_fraction",
        "min_valid_support_fraction",
    )
    assert FNIRS_H5_WINDOW_INT64 == (
        "window_id", "segment_id", "sample_count", "timestamp_count"
    )
    assert FNIRS_H5_CHANNEL_MATRIX_FLOAT64 == (
        "mean_M", "sd_M", "valid_support_fraction"
    )
    assert FNIRS_H5_CHANNEL_MATRIX_INT64 == (
        "valid_sample_count", "valid_timestamp_count"
    )
    assert FNIRS_H5_WINDOW_UTF8 == (
        "window_status", "rejection_reason", "computation_status",
        "measurement_validity", "qualification_status", "qc_status",
    )
    assert FNIRS_H5_WINDOW_INT8 == ("qc_ok_fnirs",)
    assert FNIRS_H5_WINDOW_UINT8 == ("is_qualified",)
    assert FNIRS_H5_CHANNEL_UTF8_AXIS == ("channel_name", "chromophore", "unit")
    assert FNIRS_H5_CHANNEL_UTF8_MATRIX == ("rejection_reason",)
    assert len(FNIRS_H5_REQUIRED_PATHS) == len(set(FNIRS_H5_REQUIRED_PATHS))


def test_fnirs_h5_contract_freezes_complete_dataset_tree():
    assert FNIRS_H5_REQUIRED_PATHS == (
        "/row_source/row_source", "/row_source/source_format", "/row_source/raw_file",
        "/row_source/has_fnirs", "/row_source/has_eeg", "/row_source/has_meg",
        "/row_source/is_simultaneous", "/fnirs/windows/window_id",
        "/fnirs/windows/t_start_s", "/fnirs/windows/t_end_s",
        "/fnirs/windows/actual_t_start_s", "/fnirs/windows/actual_t_end_s",
        "/fnirs/windows/run_relative_t_start_s", "/fnirs/windows/run_relative_t_end_s",
        "/fnirs/windows/actual_covered_duration_s", "/fnirs/windows/clock_support_fraction",
        "/fnirs/windows/min_valid_support_fraction", "/fnirs/windows/segment_id",
        "/fnirs/windows/sample_count", "/fnirs/windows/timestamp_count",
        "/fnirs/windows/window_status", "/fnirs/windows/rejection_reason",
        "/fnirs/windows/computation_status", "/fnirs/windows/measurement_validity",
        "/fnirs/windows/qualification_status", "/fnirs/windows/qc_status",
        "/fnirs/windows/qc_ok_fnirs", "/fnirs/windows/is_qualified",
        "/fnirs/channels/channel_name", "/fnirs/channels/chromophore",
        "/fnirs/channels/unit", "/fnirs/channels/mean_M", "/fnirs/channels/sd_M",
        "/fnirs/channels/valid_support_fraction", "/fnirs/channels/valid_sample_count",
        "/fnirs/channels/valid_timestamp_count", "/fnirs/channels/rejection_reason",
        "/fnirs/provenance/source_sha256", "/fnirs/provenance/channel_map_json",
        "/fnirs/provenance/preprocessing_provenance_json",
    )
