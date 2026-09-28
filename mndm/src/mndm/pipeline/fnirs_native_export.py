"""Native, unqualified HDF5 export for one fNIRS recording's window features."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote

import h5py
import numpy as np
import pandas as pd

from .fnirs_export_contract import (
    FNIRS_H5_CHANNEL_MATRIX_FLOAT64,
    FNIRS_H5_CHANNEL_MATRIX_INT64,
    FNIRS_H5_CONTRACT_VERSION,
    FNIRS_H5_REQUIRED_PATHS,
    FNIRS_H5_ROOT_ATTRIBUTES,
    FNIRS_H5_CHANNEL_UTF8_AXIS,
    FNIRS_H5_CHANNEL_UTF8_MATRIX,
    FNIRS_H5_WINDOW_FLOAT64,
    FNIRS_H5_WINDOW_INT8,
    FNIRS_H5_WINDOW_INT64,
    FNIRS_H5_WINDOW_UINT8,
    FNIRS_H5_WINDOW_UTF8,
    FNIRS_ROW_SOURCE_SCHEMA_VERSION,
)


class FnirsExportContractError(ValueError):
    """Input table or HDF5 file does not satisfy the frozen native contract."""


_WINDOW_COLUMNS = {
    "window_id": "fnirs_window_id",
    "t_start_s": "fnirs_window_start_s",
    "t_end_s": "fnirs_window_end_s",
    "actual_t_start_s": "fnirs_actual_t_start_s",
    "actual_t_end_s": "fnirs_actual_t_end_s",
    "run_relative_t_start_s": "fnirs_run_relative_t_start_s",
    "run_relative_t_end_s": "fnirs_run_relative_t_end_s",
    "actual_covered_duration_s": "fnirs_actual_covered_duration_s",
    "clock_support_fraction": "fnirs_clock_support_fraction",
    "min_valid_support_fraction": "fnirs_min_valid_support_fraction",
    "segment_id": "fnirs_segment_id",
    "sample_count": "fnirs_window_sample_count",
    "timestamp_count": "fnirs_window_timestamp_count",
    "window_status": "fnirs_window_status",
    "rejection_reason": "fnirs_rejection_reason",
    "computation_status": "fnirs_computation_status",
    "measurement_validity": "fnirs_measurement_validity",
    "qualification_status": "fnirs_qualification_status",
    "qc_status": "fnirs_qc_status",
    "qc_ok_fnirs": "qc_ok_fnirs",
}

_CHANNEL_COLUMN_PREFIXES = {
    "mean_M": "fnirs_mean_M__",
    "sd_M": "fnirs_sd_M__",
    "valid_support_fraction": "fnirs_valid_support_fraction__",
    "valid_sample_count": "fnirs_valid_sample_count__",
    "valid_timestamp_count": "fnirs_valid_timestamp_count__",
    "rejection_reason": "fnirs_rejection_reason__",
}

_UTF8 = h5py.string_dtype(encoding="utf-8")


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise FnirsExportContractError(f"fNIRS feature table is missing required columns: {missing}")


def _one_value(frame: pd.DataFrame, column: str) -> str:
    _require_columns(frame, [column])
    values = [_text(value) for value in frame[column].tolist()]
    unique = list(dict.fromkeys(values))
    if len(unique) != 1 or not unique[0]:
        raise FnirsExportContractError(f"{column} must have one non-empty value across the exported source.")
    return unique[0]


def _parse_json_object(value: str, label: str) -> dict[str, Any]:
    try:
        result = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise FnirsExportContractError(f"{label} must contain valid JSON.") from exc
    if not isinstance(result, dict):
        raise FnirsExportContractError(f"{label} must encode a JSON object.")
    return result


def _validate_preprocessing_provenance(provenance: dict[str, Any], source_sha256: str) -> None:
    if provenance.get("source_sha256") != source_sha256:
        raise FnirsExportContractError("Source digest disagrees with preprocessing provenance.")
    if provenance.get("schema") != "mndm.fnirs_preprocessing_provenance.v1":
        raise FnirsExportContractError("Preprocessing provenance schema is missing or unsupported.")
    if provenance.get("haemoglobin_units") != "M (molar; MNE FIFF_UNIT_MOL)":
        raise FnirsExportContractError("Preprocessing provenance must declare MNE molar Hb units.")
    if provenance.get("measurement_validity") != "not_assessed" or provenance.get("claim_status") != "no_biological_claim":
        raise FnirsExportContractError("fNIRS provenance cannot upgrade measurement validity or biological claim status.")
    software = provenance.get("software")
    if not isinstance(software, Mapping) or not all(_text(software.get(name)) for name in ("mne", "mne_nirs")):
        raise FnirsExportContractError("Preprocessing provenance must record MNE and MNE-NIRS versions.")
    operations = provenance.get("operations")
    if not isinstance(operations, list) or any(
        not isinstance(operation, Mapping)
        or not _text(operation.get("name"))
        or not _text(operation.get("status"))
        or (
            operation.get("status") == "not_applicable"
            and not _text(operation.get("reason"))
        )
        or (
            operation.get("status") != "not_applicable"
            and not isinstance(operation.get("parameters"), Mapping)
        )
        or (
            "parameters" in operation
            and not isinstance(operation.get("parameters"), Mapping)
        )
        for operation in operations
    ):
        raise FnirsExportContractError("Preprocessing operations must retain names, statuses, and applicable parameters or reasons.")
    input_classification = provenance.get("input_classification")
    optical_density = next(
        (operation for operation in operations if operation.get("name") == "intensity_to_optical_density"),
        None,
    )
    expected_od_status = {
        "cw_amplitude": "applied",
        "optical_density": "not_applicable",
    }.get(input_classification)
    if optical_density is None or optical_density.get("status") != expected_od_status:
        raise FnirsExportContractError("Optical-density provenance status must match the declared input classification.")
    if input_classification == "optical_density" and optical_density.get("reason") != "input_already_optical_density":
        raise FnirsExportContractError("Already-optical-density input must explain why intensity conversion is not applicable.")
    beer_lambert = next(
        (
            operation
            for operation in operations
            if operation.get("name") == "beer_lambert_law" and operation.get("status") == "applied"
        ),
        None,
    )
    if beer_lambert is None:
        raise FnirsExportContractError("Preprocessing provenance must record applied Beer-Lambert conversion.")
    ppf = beer_lambert["parameters"].get("ppf")
    try:
        ppf_value = float(ppf)
    except (TypeError, ValueError) as exc:
        raise FnirsExportContractError("Beer-Lambert provenance must record a positive finite PPF.") from exc
    if not np.isfinite(ppf_value) or ppf_value <= 0:
        raise FnirsExportContractError("Beer-Lambert provenance must record a positive finite PPF.")


def _source_and_provenance(frame: pd.DataFrame) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    raw_file = _one_value(frame, "file")
    if Path(raw_file).suffix.lower() != ".snirf":
        raise FnirsExportContractError("Native fNIRS export requires a .snirf source file identity.")
    source_sha256 = _one_value(frame, "fnirs_source_sha256").lower()
    if len(source_sha256) != 64 or any(character not in "0123456789abcdef" for character in source_sha256):
        raise FnirsExportContractError("fnirs_source_sha256 must be a 64-character SHA-256 digest.")
    channel_map_json = _one_value(frame, "fnirs_channel_map_json")
    preprocessing_json = _one_value(frame, "fnirs_preprocessing_provenance_json")
    channel_map = _parse_json_object_or_array(channel_map_json, "fnirs_channel_map_json")
    provenance = _parse_json_object(preprocessing_json, "fnirs_preprocessing_provenance_json")
    _validate_preprocessing_provenance(provenance, source_sha256)
    return raw_file, source_sha256, channel_map, provenance


def _parse_json_object_or_array(value: str, label: str) -> list[dict[str, Any]]:
    try:
        result = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise FnirsExportContractError(f"{label} must contain valid JSON.") from exc
    if not isinstance(result, list) or not result or any(not isinstance(item, dict) for item in result):
        raise FnirsExportContractError(f"{label} must encode a non-empty JSON array of channel objects.")
    return result


def _validate_qc(frame: pd.DataFrame, *, require_qc_pass: bool) -> np.ndarray:
    values = frame[_WINDOW_COLUMNS["qc_ok_fnirs"]].to_numpy(dtype=object)
    result = np.full(len(values), -1, dtype=np.int8)
    for index, value in enumerate(values):
        if pd.isna(value):
            continue
        if not isinstance(value, (bool, np.bool_)):
            raise FnirsExportContractError("qc_ok_fnirs must contain nullable booleans; numeric truthy values are forbidden.")
        result[index] = 1 if bool(value) else 0
    if require_qc_pass and np.any(result != 1):
        raise FnirsExportContractError("QC-gated export requires qc_ok_fnirs=true for every row; unknown/failing QC is not pass.")
    qc_states = frame[_WINDOW_COLUMNS["qc_status"]].astype("string").fillna("").unique().tolist()
    if np.all(result == -1) and qc_states != ["not_assessed"]:
        raise FnirsExportContractError("All-unknown qc_ok_fnirs requires qc_status=not_assessed.")
    return result


def _prepare_fnirs_table(frame: pd.DataFrame, *, require_qc_pass: bool = False) -> dict[str, Any]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise FnirsExportContractError("A non-empty fNIRS feature DataFrame is required.")
    if "modality" in frame.columns:
        modalities = set(frame["modality"].dropna().astype(str).str.lower())
        if modalities - {"fnirs"}:
            raise FnirsExportContractError(f"Mixed or unsupported modalities cannot be exported: {sorted(modalities)}")
    if "row_source" in frame.columns:
        sources = set(frame["row_source"].dropna().astype(str))
        if sources != {"snirf_fnirs"}:
            raise FnirsExportContractError("row_source must identify only snirf_fnirs rows.")

    required = ["file", *_WINDOW_COLUMNS.values(), "fnirs_source_sha256", "fnirs_channel_map_json", "fnirs_preprocessing_provenance_json"]
    _require_columns(frame, required)
    raw_file, source_sha256, channel_map, provenance = _source_and_provenance(frame)
    if frame["file"].astype("string").nunique(dropna=True) != 1:
        raise FnirsExportContractError("Exactly one raw recording is supported per native fNIRS HDF5 file.")
    if "row_source" in frame.columns and set(frame["row_source"].dropna().astype(str)) != {"snirf_fnirs"}:
        raise FnirsExportContractError("Unknown or mixed source rows are unsupported.")

    channel_names: list[str] = []
    chromophores: list[str] = []
    channel_keys: list[str] = []
    for item in channel_map:
        name = str(item.get("channel_name", ""))
        chromophore = str(item.get("chromophore", "")).lower()
        key = str(item.get("column_key", ""))
        if not name or not key or chromophore not in {"hbo", "hbr"} or item.get("unit") != "M":
            raise FnirsExportContractError("Channel map requires unique channel_name, hbo/hbr chromophore, column_key and M unit.")
        if key != quote(name, safe=""):
            raise FnirsExportContractError("Channel column_key must be the URL-quoted channel_name frozen by G3.")
        channel_names.append(name)
        chromophores.append(chromophore)
        channel_keys.append(key)
    if len(set(zip(channel_names, chromophores))) != len(channel_names) or len(set(channel_keys)) != len(channel_keys):
        raise FnirsExportContractError("fNIRS channel identities and feature column keys must be unique.")

    window_id = pd.to_numeric(frame[_WINDOW_COLUMNS["window_id"]], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(window_id).all() or not np.equal(window_id, np.floor(window_id)).all() or len(np.unique(window_id)) != len(window_id):
        raise FnirsExportContractError("window_id must contain unique finite integers.")
    n_windows = len(frame)
    n_channels = len(channel_names)
    windows: dict[str, Any] = {}
    for target, source in _WINDOW_COLUMNS.items():
        column = frame[source]
        if target in FNIRS_H5_WINDOW_FLOAT64:
            values = pd.to_numeric(column, errors="coerce").to_numpy(dtype=float)
            allow_no_support = target in {"actual_t_start_s", "actual_t_end_s"}
            if np.isinf(values).any() or (not allow_no_support and not np.isfinite(values).all()):
                raise FnirsExportContractError(f"{source} must be finite for every window.")
            windows[target] = values.astype(np.float64, copy=False)
        elif target in FNIRS_H5_WINDOW_INT64:
            values = pd.to_numeric(column, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            if target == "segment_id":
                values = np.where(np.isfinite(values), values, -1)
            if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
                raise FnirsExportContractError(f"{source} must contain finite integers (or null segment IDs).")
            if target in {"window_id", "sample_count", "timestamp_count"} and np.any(values < 0):
                raise FnirsExportContractError(f"{source} may not contain negative values.")
            if target == "segment_id" and np.any(values < -1):
                raise FnirsExportContractError("fnirs_segment_id may use -1 as the only unavailable sentinel.")
            windows[target] = values.astype(np.int64)
        elif target in {"qc_ok_fnirs"}:
            windows[target] = _validate_qc(frame, require_qc_pass=require_qc_pass)
        else:
            windows[target] = np.asarray([_text(value) for value in column.tolist()], dtype=object)

    starts = windows["t_start_s"]
    ends = windows["t_end_s"]
    actual_starts = windows["actual_t_start_s"]
    actual_ends = windows["actual_t_end_s"]
    support = windows["clock_support_fraction"]
    covered = windows["actual_covered_duration_s"]
    if np.any(ends <= starts) or np.any(np.diff(starts) < 0):
        raise FnirsExportContractError("Window intervals must be positive and ordered by SNIRF time.")
    no_actual_bounds = np.isnan(actual_starts) | np.isnan(actual_ends)
    if np.any(np.isnan(actual_starts) != np.isnan(actual_ends)):
        raise FnirsExportContractError("Actual support bounds must either both be finite or both be unavailable.")
    rejected = np.asarray([status == "clock_rejected" for status in windows["window_status"]], dtype=bool)
    empty_support = (
        rejected
        & (support == 0)
        & (covered == 0)
        & (windows["sample_count"] == 0)
        & (windows["timestamp_count"] == 0)
    )
    if np.any(no_actual_bounds & ~empty_support):
        raise FnirsExportContractError("Unavailable actual bounds require a rejected zero-support window.")
    has_actual_bounds = ~no_actual_bounds
    if np.any(actual_starts[has_actual_bounds] < starts[has_actual_bounds] - 1e-9) or np.any(actual_ends[has_actual_bounds] > ends[has_actual_bounds] + 1e-9) or np.any(actual_ends[has_actual_bounds] < actual_starts[has_actual_bounds]):
        raise FnirsExportContractError("Actual supported time bounds must lie within nominal window bounds.")
    if np.any((support < -1e-10) | (support > 1.0 + 1e-10)) or np.any(covered < 0):
        raise FnirsExportContractError("Clock support fractions/durations are outside their valid range.")
    if not np.allclose(covered, support * (ends - starts), rtol=1e-8, atol=1e-8):
        raise FnirsExportContractError("Covered duration must agree with clock support fraction and nominal window duration.")
    if not np.allclose(windows["run_relative_t_start_s"], starts - starts[0], rtol=0, atol=1e-8):
        raise FnirsExportContractError("Run-relative window start does not agree with SNIRF clock origin.")
    statuses = windows["window_status"]
    if not set(statuses).issubset({"clock_supported", "clock_rejected"}):
        raise FnirsExportContractError("window_status contains an unsupported value.")
    rejected = np.asarray([status == "clock_rejected" for status in statuses], dtype=bool)
    reasons = windows["rejection_reason"]
    if np.any(rejected & np.asarray([not reason for reason in reasons], dtype=bool)):
        raise FnirsExportContractError("Rejected windows require an explicit rejection reason.")
    if not np.all(np.asarray(windows["computation_status"]) == "computed"):
        raise FnirsExportContractError("G4 v1 accepts computed windows only.")
    if not np.all(np.asarray(windows["measurement_validity"]) == "not_assessed"):
        raise FnirsExportContractError("G4 v1 cannot upgrade measurement validity.")
    if not np.all(np.asarray(windows["qualification_status"]) == "not_assessed"):
        raise FnirsExportContractError("G4 v1 cannot qualify measurement rows.")

    channels: dict[str, Any] = {
        "channel_name": np.asarray(channel_names, dtype=object),
        "chromophore": np.asarray(chromophores, dtype=object),
        "unit": np.asarray(["M"] * n_channels, dtype=object),
    }
    for target, prefix in _CHANNEL_COLUMN_PREFIXES.items():
        arrays = []
        for key in channel_keys:
            column = f"{prefix}{key}"
            _require_columns(frame, [column])
            if target in FNIRS_H5_CHANNEL_MATRIX_FLOAT64:
                values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
                if np.isinf(values).any():
                    raise FnirsExportContractError(f"{column} contains infinite values.")
                arrays.append(values.astype(np.float64, copy=False))
            elif target in FNIRS_H5_CHANNEL_MATRIX_INT64:
                values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
                    raise FnirsExportContractError(f"{column} must contain finite integer counts.")
                arrays.append(values.astype(np.int64))
            else:
                arrays.append(np.asarray([_text(value) for value in frame[column].tolist()], dtype=object))
        channels[target] = np.column_stack(arrays)

    for stat_name in ("mean_M", "sd_M"):
        if np.any(np.isfinite(channels[stat_name][rejected])):
            raise FnirsExportContractError("Clock-rejected windows must not carry computed HbO/HbR statistics.")
    if "is_qualified" in frame.columns and frame["is_qualified"].fillna(False).astype(bool).any():
        raise FnirsExportContractError("Feature input must not claim qualified fNIRS measurements.")

    return {
        "raw_file": raw_file,
        "source_sha256": source_sha256,
        "channel_map_json": json.dumps(channel_map, sort_keys=True, separators=(",", ":")),
        "preprocessing_provenance_json": json.dumps(provenance, sort_keys=True, separators=(",", ":")),
        "windows": windows,
        "channels": channels,
    }


def _write_strings(group: h5py.Group, name: str, values: np.ndarray) -> None:
    array = np.asarray(values, dtype=object)
    group.create_dataset(name, data=array, dtype=_UTF8)


def _decode_strings(dataset: h5py.Dataset) -> list[str]:
    values = np.asarray(dataset)
    return [value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value) for value in values]


def validate_fnirs_h5(path: Path) -> None:
    """Fail closed unless an HDF5 file conforms to the frozen fNIRS contract."""
    path = Path(path)
    try:
        with h5py.File(path, "r") as h5:
            expected_root = {
                "export_contract_version": FNIRS_H5_CONTRACT_VERSION,
                "modality": "fnirs",
                "has_fnirs": 1,
                "has_eeg": 0,
                "has_meg": 0,
                "is_simultaneous": 0,
                "capability_status": "native_only",
                "mnps_qualified": 0,
                "measurement_validity": "not_assessed",
                "qualification_status": "not_assessed",
                "claim_status": "no_biological_claim",
            }
            for name in FNIRS_H5_ROOT_ATTRIBUTES:
                if name not in h5.attrs:
                    raise FnirsExportContractError(f"Root attribute is missing: {name}")
                value = h5.attrs[name]
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                if value != expected_root[name]:
                    raise FnirsExportContractError(f"Unexpected root attribute {name}={value!r}.")
            if h5.attrs.get("_schema_version") != FNIRS_H5_CONTRACT_VERSION:
                raise FnirsExportContractError("Root _schema_version does not match the fNIRS contract.")
            missing = [name for name in FNIRS_H5_REQUIRED_PATHS if name not in h5]
            if missing:
                raise FnirsExportContractError(f"Required HDF5 paths are missing: {missing}")
            if any(forbidden in h5 for forbidden in ("time", "mnps_3d", "coords_9d")):
                raise FnirsExportContractError("Native fNIRS files may not contain generic time/MNPS coordinate surfaces.")
            fnirs = h5["fnirs"]
            if fnirs.attrs.get("_schema_version") != FNIRS_H5_CONTRACT_VERSION:
                raise FnirsExportContractError("/fnirs schema version does not match the fNIRS contract.")
            if fnirs.attrs.get("signal_representation") != "window_features" or fnirs.attrs.get("axis_order") != "window,channel":
                raise FnirsExportContractError("/fnirs must declare window-feature representation and axis order.")
            if h5["row_source"].attrs.get("_schema_version") != FNIRS_ROW_SOURCE_SCHEMA_VERSION:
                raise FnirsExportContractError("/row_source schema version is invalid.")

            n_windows = len(h5["fnirs/windows/window_id"])
            n_channels = len(h5["fnirs/channels/channel_name"])
            if n_windows == 0 or n_channels == 0:
                raise FnirsExportContractError("Window and channel axes must be non-empty.")
            window_ids = h5["fnirs/windows/window_id"][:]
            if len(np.unique(window_ids)) != n_windows:
                raise FnirsExportContractError("HDF5 window_id values must be unique.")
            for name in ("mean_M", "sd_M", "valid_support_fraction", "valid_sample_count", "valid_timestamp_count", "rejection_reason"):
                if h5[f"fnirs/channels/{name}"].shape != (n_windows, n_channels):
                    raise FnirsExportContractError(f"/fnirs/channels/{name} has an invalid [window, channel] shape.")
            for name in ("row_source/row_source", "row_source/source_format", "row_source/raw_file", "row_source/has_fnirs", "row_source/has_eeg", "row_source/has_meg", "row_source/is_simultaneous"):
                if h5[name].shape != (n_windows,):
                    raise FnirsExportContractError(f"/{name} is not aligned to the window axis.")
            for name in FNIRS_H5_WINDOW_FLOAT64:
                dataset = h5[f"fnirs/windows/{name}"]
                if dataset.shape != (n_windows,) or dataset.dtype != np.dtype("float64"):
                    raise FnirsExportContractError(f"/fnirs/windows/{name} must be float64 [window].")
            for name in FNIRS_H5_WINDOW_INT64:
                dataset = h5[f"fnirs/windows/{name}"]
                if dataset.shape != (n_windows,) or dataset.dtype != np.dtype("int64"):
                    raise FnirsExportContractError(f"/fnirs/windows/{name} must be int64 [window].")
            for name in FNIRS_H5_WINDOW_INT8:
                dataset = h5[f"fnirs/windows/{name}"]
                if dataset.shape != (n_windows,) or dataset.dtype != np.dtype("int8"):
                    raise FnirsExportContractError(f"/fnirs/windows/{name} must be int8 [window].")
            for name in FNIRS_H5_WINDOW_UINT8:
                dataset = h5[f"fnirs/windows/{name}"]
                if dataset.shape != (n_windows,) or dataset.dtype != np.dtype("uint8"):
                    raise FnirsExportContractError(f"/fnirs/windows/{name} must be uint8 [window].")
            for name in FNIRS_H5_WINDOW_UTF8:
                dataset = h5[f"fnirs/windows/{name}"]
                if dataset.shape != (n_windows,) or h5py.check_string_dtype(dataset.dtype) is None:
                    raise FnirsExportContractError(f"/fnirs/windows/{name} must be UTF-8 [window].")
            for name in FNIRS_H5_CHANNEL_MATRIX_FLOAT64:
                dataset = h5[f"fnirs/channels/{name}"]
                if dataset.shape != (n_windows, n_channels) or dataset.dtype != np.dtype("float64"):
                    raise FnirsExportContractError(f"/fnirs/channels/{name} must be float64 [window, channel].")
            for name in FNIRS_H5_CHANNEL_MATRIX_INT64:
                dataset = h5[f"fnirs/channels/{name}"]
                if dataset.shape != (n_windows, n_channels) or dataset.dtype != np.dtype("int64"):
                    raise FnirsExportContractError(f"/fnirs/channels/{name} must be int64 [window, channel].")
            for name in FNIRS_H5_CHANNEL_UTF8_AXIS:
                dataset = h5[f"fnirs/channels/{name}"]
                if dataset.shape != (n_channels,) or h5py.check_string_dtype(dataset.dtype) is None:
                    raise FnirsExportContractError(f"/fnirs/channels/{name} must be UTF-8 [channel].")
            for name in FNIRS_H5_CHANNEL_UTF8_MATRIX:
                dataset = h5[f"fnirs/channels/{name}"]
                if dataset.shape != (n_windows, n_channels) or h5py.check_string_dtype(dataset.dtype) is None:
                    raise FnirsExportContractError(f"/fnirs/channels/{name} must be UTF-8 [window, channel].")

            channel_names = _decode_strings(h5["fnirs/channels/channel_name"])
            chromophores = _decode_strings(h5["fnirs/channels/chromophore"])
            units = _decode_strings(h5["fnirs/channels/unit"])
            if len(set(channel_names)) != n_channels or any(chrom not in {"hbo", "hbr"} for chrom in chromophores):
                raise FnirsExportContractError("Channel names must be unique and chromophores must be hbo/hbr.")
            if units != ["M"] * n_channels:
                raise FnirsExportContractError("All exported Hb channels must use M units.")
            channel_map_value = h5["fnirs/provenance/channel_map_json"][()]
            if isinstance(channel_map_value, bytes):
                channel_map_value = channel_map_value.decode("utf-8")
            channel_map = _parse_json_object_or_array(str(channel_map_value), "HDF5 channel map")
            map_identity = [
                (str(item.get("channel_name", "")), str(item.get("chromophore", "")).lower(), item.get("unit"))
                for item in channel_map
            ]
            if map_identity != list(zip(channel_names, chromophores, units)):
                raise FnirsExportContractError("HDF5 channel map does not agree with the channel axis.")
            if any(str(item.get("column_key", "")) != quote(str(item.get("channel_name", "")), safe="") for item in channel_map):
                raise FnirsExportContractError("HDF5 channel map contains a column key inconsistent with channel identity.")
            windows = h5["fnirs/windows"]
            window_status = windows["window_status"].asstr()[:]
            computation_status = windows["computation_status"].asstr()[:]
            measurement_validity = windows["measurement_validity"].asstr()[:]
            qualification_status = windows["qualification_status"].asstr()[:]
            qc_status = windows["qc_status"].asstr()[:]
            if not set(window_status).issubset({"clock_supported", "clock_rejected"}):
                raise FnirsExportContractError("HDF5 window_status contains an unsupported value.")
            if not np.all(computation_status == "computed"):
                raise FnirsExportContractError("G4 v1 only accepts computed window rows.")
            if not np.all(measurement_validity == "not_assessed") or not np.all(qualification_status == "not_assessed"):
                raise FnirsExportContractError("G4 v1 cannot upgrade measurement validity or qualification status.")
            if not set(qc_status).issubset({"not_assessed", "assessed"}):
                raise FnirsExportContractError("HDF5 qc_status contains an unsupported value.")
            qc_values = windows["qc_ok_fnirs"][:]
            if np.any(~np.isin(qc_values, [-1, 0, 1])):
                raise FnirsExportContractError("qc_ok_fnirs codes must be -1 unknown, 0 fail, or 1 pass.")
            if np.any((qc_values == -1) != (qc_status == "not_assessed")):
                raise FnirsExportContractError("qc_status must agree with unknown versus assessed QC codes.")
            if np.any(windows["is_qualified"][:] != 0):
                raise FnirsExportContractError("G4 v1 output must keep every row unqualified.")
            if _decode_strings(h5["row_source/row_source"]) != ["snirf_fnirs"] * n_windows:
                raise FnirsExportContractError("Every row must be identified as snirf_fnirs.")
            if _decode_strings(h5["row_source/source_format"]) != ["snirf"] * n_windows:
                raise FnirsExportContractError("Every row must declare source_format=snirf.")
            raw_files = _decode_strings(h5["row_source/raw_file"])
            if len(set(raw_files)) != 1 or Path(raw_files[0]).suffix.lower() != ".snirf":
                raise FnirsExportContractError("Every row must identify the same .snirf source recording.")
            if np.any(h5["row_source/has_fnirs"][:] != 1) or np.any(h5["row_source/has_eeg"][:] != 0) or np.any(h5["row_source/has_meg"][:] != 0) or np.any(h5["row_source/is_simultaneous"][:] != 0):
                raise FnirsExportContractError("Per-row modality flags do not match pure fNIRS.")
            start = windows["t_start_s"][:]
            end = windows["t_end_s"][:]
            float_window_data = {name: windows[name][:] for name in FNIRS_H5_WINDOW_FLOAT64}
            if any(np.any(~np.isfinite(values)) for name, values in float_window_data.items() if name not in {"actual_t_start_s", "actual_t_end_s"}):
                raise FnirsExportContractError("HDF5 window timing and support fields must be finite.")
            actual_start = windows["actual_t_start_s"][:]
            actual_end = windows["actual_t_end_s"][:]
            if np.any(np.isinf(actual_start)) or np.any(np.isinf(actual_end)) or np.any(np.isnan(actual_start) != np.isnan(actual_end)):
                raise FnirsExportContractError("HDF5 actual support bounds must be paired finite values or paired missing values.")
            no_actual_bounds = np.isnan(actual_start)
            rejected = window_status == "clock_rejected"
            empty_support = (
                rejected
                & (windows["clock_support_fraction"][:] == 0)
                & (windows["actual_covered_duration_s"][:] == 0)
                & (windows["sample_count"][:] == 0)
                & (windows["timestamp_count"][:] == 0)
            )
            if np.any(no_actual_bounds & ~empty_support):
                raise FnirsExportContractError("Missing HDF5 actual bounds require a rejected zero-support window.")
            has_actual_bounds = ~no_actual_bounds
            if np.any(end <= start) or np.any(np.diff(start) < 0):
                raise FnirsExportContractError("HDF5 window bounds are not positive and time ordered.")
            if np.any(actual_start[has_actual_bounds] < start[has_actual_bounds] - 1e-9) or np.any(actual_end[has_actual_bounds] > end[has_actual_bounds] + 1e-9) or np.any(actual_end[has_actual_bounds] < actual_start[has_actual_bounds]):
                raise FnirsExportContractError("HDF5 actual support bounds do not lie inside nominal window bounds.")
            if not np.allclose(windows["run_relative_t_start_s"][:], start - start[0], rtol=0, atol=1e-8):
                raise FnirsExportContractError("HDF5 run-relative window start disagrees with SNIRF time origin.")
            if not np.allclose(windows["run_relative_t_end_s"][:], end - start[0], rtol=0, atol=1e-8):
                raise FnirsExportContractError("HDF5 run-relative window end disagrees with SNIRF time origin.")
            support = h5["fnirs/windows/clock_support_fraction"][:]
            if np.any(~np.isfinite(support)) or np.any((support < -1e-10) | (support > 1.0 + 1e-10)):
                raise FnirsExportContractError("HDF5 clock support fraction is invalid.")
            covered = windows["actual_covered_duration_s"][:]
            if np.any(covered < 0) or not np.allclose(covered, support * (end - start), rtol=1e-8, atol=1e-8):
                raise FnirsExportContractError("HDF5 covered duration disagrees with clock support fraction and nominal duration.")
            min_support = h5["fnirs/windows/min_valid_support_fraction"][:]
            if np.any(~np.isfinite(min_support)) or np.any((min_support < 0) | (min_support > 1)):
                raise FnirsExportContractError("HDF5 minimum valid support fraction is invalid.")
            for name in ("mean_M", "sd_M"):
                values = h5[f"fnirs/channels/{name}"][:]
                if np.any(np.isinf(values)):
                    raise FnirsExportContractError(f"HDF5 {name} contains infinite values.")
                rejected = h5["fnirs/windows/window_status"].asstr()[:] == "clock_rejected"
                if np.any(np.isfinite(values[rejected])):
                    raise FnirsExportContractError("Clock-rejected windows must not contain finite Hb statistics.")
            channel_support = h5["fnirs/channels/valid_support_fraction"][:]
            if np.any(np.isinf(channel_support)) or np.any(np.isfinite(channel_support) & ((channel_support < -1e-10) | (channel_support > 1 + 1e-10))):
                raise FnirsExportContractError("Per-channel valid support fraction is invalid.")
            for name in ("valid_sample_count", "valid_timestamp_count"):
                if np.any(h5[f"fnirs/channels/{name}"][:] < 0):
                    raise FnirsExportContractError(f"HDF5 {name} may not be negative.")
            for name in ("window_id", "sample_count", "timestamp_count"):
                if np.any(windows[name][:] < 0):
                    raise FnirsExportContractError(f"HDF5 {name} may not be negative.")
            if np.any(windows["segment_id"][:] < -1):
                raise FnirsExportContractError("HDF5 segment_id may use -1 as the only unavailable sentinel.")
            preprocessing_json = h5["fnirs/provenance/preprocessing_provenance_json"][()]
            if isinstance(preprocessing_json, bytes):
                preprocessing_json = preprocessing_json.decode("utf-8")
            provenance = _parse_json_object(str(preprocessing_json), "HDF5 preprocessing provenance")
            source_digest = h5["fnirs/provenance/source_sha256"][()]
            if isinstance(source_digest, bytes):
                source_digest = source_digest.decode("utf-8")
            if len(str(source_digest)) != 64 or any(character not in "0123456789abcdef" for character in str(source_digest).lower()):
                raise FnirsExportContractError("HDF5 source SHA-256 must be a 64-character hexadecimal digest.")
            if provenance.get("source_sha256") != source_digest:
                raise FnirsExportContractError("HDF5 source SHA-256 disagrees with preprocessing provenance.")
            _validate_preprocessing_provenance(provenance, str(source_digest))
    except OSError as exc:
        raise FnirsExportContractError(f"Cannot open fNIRS HDF5 file {path}: {exc}") from exc


def write_fnirs_native_h5(
    frame: pd.DataFrame,
    out_path: Path,
    *,
    require_qc_pass: bool = False,
    overwrite: bool = False,
) -> Path:
    """Write and validate one native fNIRS window-feature table as HDF5."""
    target = Path(out_path)
    if target.suffix.lower() not in {".h5", ".hdf5"}:
        raise FnirsExportContractError("Native fNIRS output path must end in .h5 or .hdf5.")
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {target}")
    data = _prepare_fnirs_table(frame, require_qc_pass=require_qc_pass)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{target.stem}.", suffix=".tmp.h5", dir=target.parent, delete=False) as stream:
            temporary_path = Path(stream.name)
        with h5py.File(temporary_path, "w") as h5:
            h5.attrs["_schema_version"] = FNIRS_H5_CONTRACT_VERSION
            h5.attrs["export_contract_version"] = FNIRS_H5_CONTRACT_VERSION
            h5.attrs["modality"] = "fnirs"
            h5.attrs["has_fnirs"] = np.uint8(1)
            h5.attrs["has_eeg"] = np.uint8(0)
            h5.attrs["has_meg"] = np.uint8(0)
            h5.attrs["is_simultaneous"] = np.uint8(0)
            h5.attrs["capability_status"] = "native_only"
            h5.attrs["mnps_qualified"] = np.uint8(0)
            h5.attrs["measurement_validity"] = "not_assessed"
            h5.attrs["qualification_status"] = "not_assessed"
            h5.attrs["claim_status"] = "no_biological_claim"

            row_source = h5.create_group("row_source")
            row_source.attrs["_schema_version"] = FNIRS_ROW_SOURCE_SCHEMA_VERSION
            row_source.attrs["alignment"] = "per_window"
            n_windows = len(data["windows"]["window_id"])
            for name, value in {
                "row_source": ["snirf_fnirs"] * n_windows,
                "source_format": ["snirf"] * n_windows,
                "raw_file": [data["raw_file"]] * n_windows,
            }.items():
                _write_strings(row_source, name, np.asarray(value, dtype=object))
            for name, value in {
                "has_fnirs": np.ones(n_windows, dtype=np.uint8),
                "has_eeg": np.zeros(n_windows, dtype=np.uint8),
                "has_meg": np.zeros(n_windows, dtype=np.uint8),
                "is_simultaneous": np.zeros(n_windows, dtype=np.uint8),
            }.items():
                row_source.create_dataset(name, data=value)

            fnirs = h5.create_group("fnirs")
            fnirs.attrs["_schema_version"] = FNIRS_H5_CONTRACT_VERSION
            fnirs.attrs["signal_representation"] = "window_features"
            fnirs.attrs["axis_order"] = "window,channel"
            fnirs.attrs["clock_basis"] = "SNIRF_time_seconds"
            fnirs.attrs["signal_unit"] = "M"
            windows = fnirs.create_group("windows")
            for name, values in data["windows"].items():
                if name in {"window_status", "rejection_reason", "computation_status", "measurement_validity", "qualification_status", "qc_status"}:
                    _write_strings(windows, name, values)
                elif name in FNIRS_H5_WINDOW_FLOAT64:
                    windows.create_dataset(name, data=np.asarray(values, dtype=np.float64))
                elif name in FNIRS_H5_WINDOW_INT64:
                    windows.create_dataset(name, data=np.asarray(values, dtype=np.int64))
                else:
                    windows.create_dataset(name, data=np.asarray(values, dtype=np.int8))
            qualified = np.zeros(n_windows, dtype=np.uint8)
            windows.create_dataset("is_qualified", data=qualified)

            channels = fnirs.create_group("channels")
            for name in ("channel_name", "chromophore", "unit", "rejection_reason"):
                if name == "rejection_reason":
                    _write_strings(channels, name, data["channels"][name])
                else:
                    _write_strings(channels, name, data["channels"][name])
            for name, values in data["channels"].items():
                if name in {"channel_name", "chromophore", "unit", "rejection_reason"}:
                    continue
                if name in FNIRS_H5_CHANNEL_MATRIX_FLOAT64:
                    channels.create_dataset(name, data=np.asarray(values, dtype=np.float64), compression="gzip", compression_opts=4)
                elif name in FNIRS_H5_CHANNEL_MATRIX_INT64:
                    channels.create_dataset(name, data=np.asarray(values, dtype=np.int64), compression="gzip", compression_opts=4)
                else:
                    _write_strings(channels, name, values)
            channels.attrs["axis_order"] = "window,channel"
            channels.attrs["channel_identity"] = "channel_name+chromophore"
            channels.attrs["unit"] = "M"

            provenance_group = fnirs.create_group("provenance")
            for name, value in {
                "source_sha256": data["source_sha256"],
                "channel_map_json": data["channel_map_json"],
                "preprocessing_provenance_json": data["preprocessing_provenance_json"],
            }.items():
                provenance_group.create_dataset(name, data=value, dtype=_UTF8)

        validate_fnirs_h5(temporary_path)
        if target.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {target}")
        os.replace(temporary_path, target)
        temporary_path = None
        validate_fnirs_h5(target)
        return target
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def export_fnirs_feature_table(
    feature_table: Path,
    out_path: Path,
    *,
    require_qc_pass: bool = False,
    overwrite: bool = False,
) -> Path:
    """Read a feature table using its schema sidecar when present and export it."""
    from ..parallel import _read_feature_table

    table = _read_feature_table(Path(feature_table))
    return write_fnirs_native_h5(
        table,
        out_path,
        require_qc_pass=require_qc_pass,
        overwrite=overwrite,
    )
