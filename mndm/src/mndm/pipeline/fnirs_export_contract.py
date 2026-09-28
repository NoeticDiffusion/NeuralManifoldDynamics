"""Frozen identifiers and dimensions for native fNIRS HDF5 export."""

from __future__ import annotations

FNIRS_H5_CONTRACT_VERSION = "mndm.fnirs_native_h5_contract.v1"
FNIRS_ROW_SOURCE_SCHEMA_VERSION = "mndm.row_source.v2"

FNIRS_H5_ROOT_ATTRIBUTES = (
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

FNIRS_H5_REQUIRED_PATHS = (
    "/row_source/row_source",
    "/row_source/source_format",
    "/row_source/raw_file",
    "/row_source/has_fnirs",
    "/row_source/has_eeg",
    "/row_source/has_meg",
    "/row_source/is_simultaneous",
    "/fnirs/windows/window_id",
    "/fnirs/windows/t_start_s",
    "/fnirs/windows/t_end_s",
    "/fnirs/windows/actual_t_start_s",
    "/fnirs/windows/actual_t_end_s",
    "/fnirs/windows/run_relative_t_start_s",
    "/fnirs/windows/run_relative_t_end_s",
    "/fnirs/windows/actual_covered_duration_s",
    "/fnirs/windows/clock_support_fraction",
    "/fnirs/windows/min_valid_support_fraction",
    "/fnirs/windows/segment_id",
    "/fnirs/windows/sample_count",
    "/fnirs/windows/timestamp_count",
    "/fnirs/windows/window_status",
    "/fnirs/windows/rejection_reason",
    "/fnirs/windows/computation_status",
    "/fnirs/windows/measurement_validity",
    "/fnirs/windows/qualification_status",
    "/fnirs/windows/qc_status",
    "/fnirs/windows/qc_ok_fnirs",
    "/fnirs/windows/is_qualified",
    "/fnirs/channels/channel_name",
    "/fnirs/channels/chromophore",
    "/fnirs/channels/unit",
    "/fnirs/channels/mean_M",
    "/fnirs/channels/sd_M",
    "/fnirs/channels/valid_support_fraction",
    "/fnirs/channels/valid_sample_count",
    "/fnirs/channels/valid_timestamp_count",
    "/fnirs/channels/rejection_reason",
    "/fnirs/provenance/source_sha256",
    "/fnirs/provenance/channel_map_json",
    "/fnirs/provenance/preprocessing_provenance_json",
)

FNIRS_H5_WINDOW_FLOAT64 = (
    "t_start_s",
    "t_end_s",
    "actual_t_start_s",
    "actual_t_end_s",
    "run_relative_t_start_s",
    "run_relative_t_end_s",
    "actual_covered_duration_s",
    "clock_support_fraction",
    "min_valid_support_fraction",
)

FNIRS_H5_WINDOW_INT64 = (
    "window_id",
    "segment_id",
    "sample_count",
    "timestamp_count",
)

FNIRS_H5_WINDOW_UTF8 = (
    "window_status",
    "rejection_reason",
    "computation_status",
    "measurement_validity",
    "qualification_status",
    "qc_status",
)

FNIRS_H5_WINDOW_INT8 = ("qc_ok_fnirs",)
FNIRS_H5_WINDOW_UINT8 = ("is_qualified",)

FNIRS_H5_CHANNEL_UTF8_AXIS = ("channel_name", "chromophore", "unit")
FNIRS_H5_CHANNEL_UTF8_MATRIX = ("rejection_reason",)

FNIRS_H5_CHANNEL_MATRIX_FLOAT64 = (
    "mean_M",
    "sd_M",
    "valid_support_fraction",
)

FNIRS_H5_CHANNEL_MATRIX_INT64 = (
    "valid_sample_count",
    "valid_timestamp_count",
)
