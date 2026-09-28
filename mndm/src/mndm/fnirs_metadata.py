"""Read-only validation of one BIDS fNIRS SNIRF recording and its sidecars.

This module checks file structure, channel metadata, optode geometry, sample
time support, and event support.  It deliberately does not read signal values
for QC, transform them, or decide whether a recording is scientifically good.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


_LENGTH_TO_METERS = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "mm": 1e-3,
    "millimeter": 1e-3,
    "millimeters": 1e-3,
    "cm": 1e-2,
    "centimeter": 1e-2,
    "centimeters": 1e-2,
}
_SUPPORTED_DATA_TYPES = {1, 99999}
_SHORT_CHANNEL_CUTOFF_M = 0.010


@dataclass(frozen=True)
class FnirsValidationIssue:
    """One reason-coded validation finding."""

    code: str
    severity: str
    field: str
    message: str


@dataclass(frozen=True)
class FnirsMetadataReport:
    """Structured result for validating one fNIRS recording and sidecars.

    ``valid`` means that the file and metadata are internally consistent.
    ``admissible`` additionally requires an input representation supported by
    the frozen v0 reference path (continuous-wave amplitude or optical
    density). Already-converted HbO/HbR is classified but not admissible.
    """

    classification: str | None
    supported_for_v0: bool
    issues: tuple[FnirsValidationIssue, ...]
    summary: Mapping[str, Any]

    @property
    def valid(self) -> bool:
        """Whether structural and metadata checks passed."""
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def admissible(self) -> bool:
        """Whether the recording passes validation and the v0 input policy."""
        return self.valid and self.supported_for_v0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of this report."""
        return {
            "classification": self.classification,
            "supported_for_v0": self.supported_for_v0,
            "valid": self.valid,
            "admissible": self.admissible,
            "issues": [asdict(issue) for issue in self.issues],
            "summary": dict(self.summary),
        }


class _Validation:
    def __init__(self) -> None:
        self.issues: list[FnirsValidationIssue] = []

    def add(self, code: str, field: str, message: str, severity: str = "error") -> None:
        self.issues.append(FnirsValidationIssue(code, severity, field, message))


class _UnsupportedSnirfLayout(Exception):
    """An identifiable SNIRF structure variant this validator does not parse."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _read_json(path: str | Path, field: str, check: _Validation) -> dict[str, Any] | None:
    p = _path(path)
    try:
        value = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        check.add("sidecar_unreadable", field, f"Could not read {p.name} ({type(exc).__name__}).")
        return None
    if not isinstance(value, dict):
        check.add("sidecar_not_object", field, f"{p.name} must contain a JSON object.")
        return None
    return value


def _merge_json_sidecars(
    value: str | Path | Sequence[str | Path] | None,
    field: str,
    check: _Validation,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Read one or more BIDS inheritance sidecars in least-to-most order."""
    if value is None:
        check.add(f"{field}_path_missing", field, "No JSON sidecar path was resolved.")
        return None, []
    paths = [value] if isinstance(value, (str, Path)) else list(value)
    if not paths:
        check.add(f"{field}_path_sequence_empty", field, "At least one JSON sidecar path is required.")
        return None, []
    paths_with_parent: dict[Path, set[Path]] = {}
    for sidecar_path in paths:
        resolved = _path(sidecar_path).resolve(strict=False)
        paths_with_parent.setdefault(resolved.parent, set()).add(resolved)
    same_parent_candidates = [
        sorted(candidate.name for candidate in candidates)
        for candidates in paths_with_parent.values()
        if len(candidates) > 1
    ]
    if same_parent_candidates:
        check.add(
            f"{field}_ambiguous",
            field,
            "The JSON inheritance chain contains multiple distinct sidecars from the same directory: "
            f"{same_parent_candidates}.",
        )
        return None, [_path(path).name for path in paths]
    merged: dict[str, Any] = {}
    loaded_any = False
    for index, sidecar_path in enumerate(paths):
        sidecar = _read_json(sidecar_path, f"{field}[{index}]", check)
        if sidecar is not None:
            # BIDS inheritance order is least-specific first; later values
            # override conflicts while other inherited keys remain.
            merged.update(sidecar)
            loaded_any = True
    return (merged if loaded_any else None), [_path(path).name for path in paths]


def _read_tsv(path: str | Path | None, field: str, check: _Validation) -> tuple[list[str], list[dict[str, str]]] | None:
    if path is None:
        check.add(f"{field}_path_missing", field, "No TSV sidecar path was resolved.")
        return None
    p = _path(path)
    try:
        with p.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            headers = list(reader.fieldnames or [])
            rows = [
                {str(key): (value or "").strip() for key, value in row.items() if key is not None}
                for row in reader
            ]
    except (OSError, UnicodeError, csv.Error) as exc:
        check.add("sidecar_unreadable", field, f"Could not read {p.name} ({type(exc).__name__}).")
        return None
    if not headers:
        check.add("sidecar_missing_header", field, f"{p.name} has no TSV header.")
        return None
    return headers, rows


def _h5_text(dataset: Any) -> str:
    import numpy as np

    array = np.asarray(dataset)
    if array.size == 0:
        return ""
    value = array.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _h5_scalar(dataset: Any) -> Any:
    import numpy as np

    array = np.asarray(dataset)
    if array.size != 1:
        raise ValueError("expected scalar HDF5 value")
    return array.reshape(-1)[0].item()


def _label_list(group: Any, key: str, count: int, prefix: str) -> list[str]:
    import numpy as np

    if key not in group:
        return [f"{prefix}{index}" for index in range(1, count + 1)]
    values = np.asarray(group[key]).reshape(-1)
    if len(values) != count:
        raise ValueError(f"{key} has {len(values)} labels for {count} optodes")
    labels = [value.decode("utf-8", errors="replace").strip() if isinstance(value, bytes) else str(value).strip() for value in values]
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError(f"{key} labels must be nonempty and unique")
    return labels


def _positions(group: Any, key: str, count: int) -> Any:
    import numpy as np

    if key not in group:
        raise ValueError(f"missing {key}")
    values = np.asarray(group[key], dtype=float)
    if values.shape != (count, 3) or not np.isfinite(values).all():
        raise ValueError(f"{key} must be a finite {count} by 3 matrix")
    return values


def _canonical_channel_suffix(value: str, classification: str) -> str | None:
    token = value.strip().lower().replace(" ", "")
    if classification in {"cw_amplitude", "optical_density"}:
        try:
            wavelength = float(token)
        except ValueError:
            return None
        return format(wavelength, "g")
    if classification == "haemoglobin":
        if token in {"hbo", "hb_o", "oxy", "oxyhaemoglobin", "oxyhemoglobin"}:
            return "hbo"
        if token in {"hbr", "hb_r", "deoxy", "deoxyhaemoglobin", "deoxyhemoglobin"}:
            return "hbr"
    return None


def _sidecar_channel_key(row: Mapping[str, str], classification: str) -> tuple[str, str, str] | None:
    source = row.get("source", "").strip()
    detector = row.get("detector", "").strip()
    if not source or not detector:
        return None
    if classification == "haemoglobin":
        channel_type = row.get("type", "").strip().lower()
        suffix = "hbo" if channel_type in {"nirscwhbo", "hbo"} else "hbr" if channel_type in {"nirscwhbr", "hbr"} else None
    else:
        suffix = _canonical_channel_suffix(row.get("wavelength_nominal", ""), classification)
    return (source, detector, suffix) if suffix is not None else None


def _normalized_unit(value: str) -> str:
    return value.strip().lower().replace("μ", "u").replace("µ", "u").replace(" ", "")


def _allowed_units(classification: str) -> set[str]:
    if classification == "cw_amplitude":
        return {"v", "volt", "volts"}
    if classification == "optical_density":
        return {"unitless", "od", "a.u.", "au"}
    if classification == "haemoglobin":
        return {"m", "mol/l", "mol/liter", "mol/litre", "um", "umol/l", "umol/liter", "umol/litre"}
    return set()


def _coordinate_rounding_tolerance(raw_value: str, scale_to_m: float, coord_m: float) -> float:
    """Bound TSV/HDF5 comparison by text precision plus float representation."""
    token = raw_value.strip().lower()
    if "e" in token:
        mantissa, exponent_text = token.split("e", 1)
        exponent = int(exponent_text)
    else:
        mantissa, exponent = token, 0
    decimals = len(mantissa.split(".", 1)[1]) if "." in mantissa else 0
    text_resolution = (10.0 ** (exponent - decimals)) * scale_to_m / 2.0
    return max(text_resolution, math.ulp(max(abs(coord_m), 1.0)) * 8.0)


def validate_fnirs_recording(
    snirf_path: str | Path,
    *,
    nirs_json: str | Path | Sequence[str | Path] | None,
    channels_tsv: str | Path | None,
    optodes_tsv: str | Path | None,
    coordsystem_json: str | Path | Sequence[str | Path] | None,
    events_tsv: str | Path | None,
    nirs_json_ambiguous: bool = False,
    channels_tsv_ambiguous: bool = False,
    events_tsv_ambiguous: bool = False,
    optodes_tsv_ambiguous: bool = False,
    coordsystem_json_ambiguous: bool = False,
) -> FnirsMetadataReport:
    """Validate one BIDS/SNIRF fNIRS recording without transforming signals.

    The function validates SNIRF channel descriptors against BIDS sidecars,
    verifies optode labels/coordinates, checks the measured time vector and
    BIDS event support, and reports the acquisition geometry class using the
    frozen ``< 10 mm`` short-channel selector. Numeric data values are never
    used for signal-quality decisions.

    Parameters are paths to one ``.snirf`` file and its matching BIDS
    one ``*_nirs.json`` path or an ordered sequence of inheritance sidecars
    from least-specific to most-specific, then ``*_channels.tsv``,
    ``*_optodes.tsv``, one ``*_coordsystem.json`` path or an ordered sequence
    of inheritance sidecars, and ``*_events.tsv``. Set the matching ambiguity
    flag when an index cannot select one unambiguous sidecar.
    """
    check = _Validation()
    snirf_file = _path(snirf_path)
    summary: dict[str, Any] = {
        "snirf_path": snirf_file.name,
        "snirf_sha256": None,
        "channel_count": None,
        "sample_count": None,
        "source_count": None,
        "detector_count": None,
        "wavelengths_nm": [],
        "measurement_pair_count": None,
        "short_channel_pair_count": None,
        "sampling_frequency_hz": None,
        "median_sample_interval_s": None,
        "time_start_s": None,
        "time_end_s": None,
        "time_gap_sample_indices": [],
        "event_count": None,
        "event_supported_count": None,
    }
    classification: str | None = None
    supported_for_v0 = False

    try:
        digest = hashlib.sha256()
        with snirf_file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        summary["snirf_sha256"] = digest.hexdigest()
    except OSError as exc:
        check.add("snirf_source_hash_unavailable", "snirf_path", f"Could not hash SNIRF source ({type(exc).__name__}).")

    if nirs_json_ambiguous:
        nirs, nirs_sources = None, []
        check.add("nirs_json_ambiguous", "nirs_json", "Multiple equally specific NIRS JSON sidecars match this recording.")
    else:
        nirs, nirs_sources = _merge_json_sidecars(nirs_json, "nirs_json", check)
    if coordsystem_json_ambiguous:
        coordsystem, coordsystem_sources = None, []
        check.add("coordsystem_json_ambiguous", "coordsystem_json", "Multiple equally specific coordinate-system JSON sidecars match this recording.")
    else:
        coordsystem, coordsystem_sources = _merge_json_sidecars(coordsystem_json, "coordsystem_json", check)
    summary["nirs_json_sources"] = nirs_sources
    summary["coordsystem_json_sources"] = coordsystem_sources
    if channels_tsv_ambiguous:
        channel_table = None
        check.add("channels_tsv_ambiguous", "channels_tsv", "Multiple equally specific channels.tsv sidecars match this recording.")
    else:
        channel_table = _read_tsv(channels_tsv, "channels_tsv", check)
    if optodes_tsv_ambiguous:
        optode_table = None
        check.add("optodes_tsv_ambiguous", "optodes_tsv", "Multiple equally specific optodes.tsv sidecars match this recording.")
    else:
        optode_table = _read_tsv(optodes_tsv, "optodes_tsv", check)
    if events_tsv_ambiguous:
        event_table = None
        check.add("events_tsv_ambiguous", "events_tsv", "Multiple equally specific events.tsv sidecars match this recording.")
    else:
        event_table = _read_tsv(events_tsv, "events_tsv", check)

    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        check.add("dependency_missing_h5py", "snirf_path", f"SNIRF validation requires h5py ({type(exc).__name__}).")
        return FnirsMetadataReport(None, False, tuple(check.issues), summary)

    measurements: list[dict[str, Any]] = []
    time_values = None
    source_labels: list[str] = []
    detector_labels: list[str] = []
    source_positions = detector_positions = None
    snirf_length_scale = None
    snirf_time_unit = None
    try:
        with h5py.File(snirf_file, "r") as h5:
            if "nirs" not in h5:
                alternate_nirs = [name for name in h5 if name.lower().startswith("nirs")]
                if alternate_nirs:
                    raise _UnsupportedSnirfLayout("snirf_group_layout_unsupported", f"Unsupported SNIRF top-level NIRS groups: {alternate_nirs}.")
                raise ValueError("missing /nirs")
            nirs_group = h5["nirs"]
            if "data1" not in nirs_group:
                alternate_data = [name for name in nirs_group if name.lower().startswith("data")]
                if alternate_data:
                    raise _UnsupportedSnirfLayout("snirf_data_block_layout_unsupported", f"Unsupported SNIRF data block layout: {alternate_data}; expected /nirs/data1.")
                raise ValueError("missing /nirs/data1")
            data_group = nirs_group["data1"]
            if "data2" in nirs_group:
                check.add("multiple_data_blocks_unsupported", "snirf_path", "SNIRF contains multiple data blocks; v0 validates one block only.")
            if "dataTimeSeries" not in data_group:
                raise _UnsupportedSnirfLayout("snirf_signal_layout_unsupported", "Unsupported SNIRF signal layout; expected /nirs/data1/dataTimeSeries.")
            if "time" not in data_group:
                raise _UnsupportedSnirfLayout("snirf_time_layout_unsupported", "Unsupported SNIRF time layout; expected /nirs/data1/time.")
            data_shape = tuple(data_group["dataTimeSeries"].shape)
            if len(data_shape) != 2 or data_shape[0] < 1 or data_shape[1] < 1:
                raise ValueError("dataTimeSeries must have shape samples by channels")
            time_values = np.asarray(data_group["time"], dtype=float).reshape(-1)
            summary["sample_count"] = int(data_shape[0])
            summary["channel_count"] = int(data_shape[1])

            list_names = sorted(
                (name for name in data_group if name.startswith("measurementList")),
                key=lambda name: int(name.removeprefix("measurementList")) if name.removeprefix("measurementList").isdigit() else math.inf,
            )
            expected_list_names = [f"measurementList{index}" for index in range(1, data_shape[1] + 1)]
            if list_names != expected_list_names:
                raise _UnsupportedSnirfLayout("snirf_measurement_layout_unsupported", f"Unsupported SNIRF measurement-list layout; expected numbered groups {expected_list_names[:3]}…{expected_list_names[-1:]}, found {list_names[:3]}.")
            if len(list_names) != data_shape[1]:
                check.add("measurement_count_mismatch", "snirf_path", f"Found {len(list_names)} measurement descriptors for {data_shape[1]} signal columns.")
            probe = nirs_group.get("probe")
            if probe is None:
                raise ValueError("missing /nirs/probe")
            if "sourcePos3D" not in probe or "detectorPos3D" not in probe:
                check.add("optode_3d_positions_missing", "snirf_path", "v0 geometry checks require sourcePos3D and detectorPos3D.")
            else:
                src_pos_raw = np.asarray(probe["sourcePos3D"], dtype=float)
                det_pos_raw = np.asarray(probe["detectorPos3D"], dtype=float)
                if src_pos_raw.ndim != 2 or src_pos_raw.shape[1:] != (3,) or not np.isfinite(src_pos_raw).all():
                    check.add("source_positions_invalid", "snirf_path", "sourcePos3D must be a finite N by 3 matrix.")
                if det_pos_raw.ndim != 2 or det_pos_raw.shape[1:] != (3,) or not np.isfinite(det_pos_raw).all():
                    check.add("detector_positions_invalid", "snirf_path", "detectorPos3D must be a finite N by 3 matrix.")

                length_unit = _h5_text(nirs_group["metaDataTags/LengthUnit"]) if "metaDataTags/LengthUnit" in nirs_group else ""
                snirf_length_scale = _LENGTH_TO_METERS.get(length_unit.lower())
                if snirf_length_scale is None:
                    check.add("snirf_length_unit_unsupported", "snirf_path", f"Unsupported or missing SNIRF LengthUnit {length_unit!r}.")
                else:
                    source_positions = src_pos_raw * snirf_length_scale
                    detector_positions = det_pos_raw * snirf_length_scale
                try:
                    source_labels = _label_list(probe, "sourceLabels", len(src_pos_raw), "S")
                    detector_labels = _label_list(probe, "detectorLabels", len(det_pos_raw), "D")
                except (ValueError, TypeError) as exc:
                    check.add("optode_labels_invalid", "snirf_path", str(exc))
                summary["source_count"] = int(len(src_pos_raw))
                summary["detector_count"] = int(len(det_pos_raw))

            snirf_time_unit = _h5_text(nirs_group["metaDataTags/TimeUnit"]) if "metaDataTags/TimeUnit" in nirs_group else ""
            if snirf_time_unit.lower() not in {"s", "sec", "second", "seconds"}:
                check.add("snirf_time_unit_unsupported", "snirf_path", f"SNIRF time vector must be in seconds; found {snirf_time_unit!r}.")

            wavelengths = np.asarray(probe["wavelengths"], dtype=float).reshape(-1) if "wavelengths" in probe else np.asarray([], dtype=float)
            if not np.isfinite(wavelengths).all() or np.any(wavelengths <= 0):
                check.add("wavelengths_invalid", "snirf_path", "Probe wavelengths must be finite positive values in nanometers.")
            summary["wavelengths_nm"] = [float(value) for value in wavelengths]

            data_types: set[int] = set()
            for name in list_names:
                item = data_group[name]
                try:
                    data_type = int(_h5_scalar(item["dataType"]))
                    source_index = int(_h5_scalar(item["sourceIndex"]))
                    detector_index = int(_h5_scalar(item["detectorIndex"]))
                    label = _h5_text(item["dataTypeLabel"]) if "dataTypeLabel" in item else ""
                    wavelength_index = int(_h5_scalar(item["wavelengthIndex"])) if "wavelengthIndex" in item else None
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    check.add("measurement_descriptor_invalid", name, f"Measurement descriptor is incomplete or invalid ({type(exc).__name__}).")
                    continue
                data_types.add(data_type)
                measurements.append({
                    "name": name,
                    "data_type": data_type,
                    "source_index": source_index,
                    "detector_index": detector_index,
                    "label": label,
                    "wavelength_index": wavelength_index,
                })
            if len(data_types) > 1:
                check.add("mixed_snirf_data_types", "snirf_path", "Measurement descriptors mix SNIRF dataType values.")
            elif data_types:
                data_type = next(iter(data_types))
                if data_type == 1:
                    classification = "cw_amplitude"
                    supported_for_v0 = True
                elif data_type == 99999:
                    labels = {measurement["label"].strip().lower().replace(" ", "") for measurement in measurements}
                    if labels and labels <= {"dod", "od", "fnirs_od", "nirsdod"}:
                        classification = "optical_density"
                        supported_for_v0 = True
                    elif labels and labels <= {"hbo", "hbr"} and labels == {"hbo", "hbr"}:
                        classification = "haemoglobin"
                        check.add("processed_hb_not_supported_v0", "snirf_path", "HbO/HbR input is classified but unsupported by the v0 reference path.", "unsupported")
                    else:
                        classification = "unknown_processed"
                        check.add("processed_data_type_label_unsupported", "snirf_path", f"Unrecognized or mixed processed labels: {sorted(labels)}.")
                else:
                    classification = "unsupported_data_type"
                    check.add("snirf_data_type_unsupported", "snirf_path", f"SNIRF dataType {data_type} is outside the v0 CW amplitude/OD/Hb classifier.")
            else:
                check.add("snirf_data_type_missing", "snirf_path", "No readable measurement dataType descriptors were found.")
    except _UnsupportedSnirfLayout as exc:
        check.add(exc.code, "snirf_path", str(exc), "unsupported")
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        check.add("snirf_unreadable", "snirf_path", f"Could not inspect SNIRF structure ({type(exc).__name__}: {exc}).")

    if time_values is not None:
        if len(time_values) != summary["sample_count"]:
            check.add("time_vector_length_mismatch", "snirf_path", f"time has {len(time_values)} values for {summary['sample_count']} samples.")
        elif len(time_values) < 2 or not np.isfinite(time_values).all():
            check.add("time_vector_invalid", "snirf_path", "time must contain at least two finite values.")
        elif np.any(np.diff(time_values) <= 0):
            check.add("time_vector_not_increasing", "snirf_path", "SNIRF time values must be strictly increasing.")
        else:
            deltas = np.diff(time_values)
            median_dt = float(np.median(deltas))
            summary["median_sample_interval_s"] = median_dt
            summary["time_start_s"] = float(time_values[0])
            summary["time_end_s"] = float(time_values[-1])
            sampling_frequency = None
            if nirs is not None:
                try:
                    sampling_frequency = float(nirs["SamplingFrequency"])
                except (KeyError, TypeError, ValueError):
                    check.add("sampling_frequency_missing_or_invalid", "nirs_json.SamplingFrequency", "SamplingFrequency must be a finite positive number.")
                if sampling_frequency is not None and (not math.isfinite(sampling_frequency) or sampling_frequency <= 0):
                    check.add("sampling_frequency_missing_or_invalid", "nirs_json.SamplingFrequency", "SamplingFrequency must be a finite positive number.")
                    sampling_frequency = None
            if sampling_frequency is not None:
                summary["sampling_frequency_hz"] = sampling_frequency
                expected_dt = 1.0 / sampling_frequency
                if not math.isclose(median_dt, expected_dt, rel_tol=1e-6, abs_tol=1e-12):
                    check.add("sampling_frequency_mismatch", "time", f"Median time step {median_dt:g}s disagrees with SamplingFrequency {sampling_frequency:g}Hz.")
                # Allow tiny floating-point jitter relative to one sample
                # period; this is a representation tolerance, not a signal-QC
                # threshold. A skipped sample remains much larger than this.
                time_tolerance = max(
                    expected_dt * 1e-6,
                    math.ulp(max(abs(float(time_values[-1])), expected_dt, 1.0)) * 16.0,
                )
                gap_indices = [int(index + 1) for index, delta in enumerate(deltas) if delta > expected_dt + time_tolerance]
                summary["time_gap_sample_indices"] = gap_indices
                if gap_indices:
                    check.add("time_gaps_detected", "time", f"Time intervals exceed the declared sample period after sample indices {gap_indices[:20]}.")
            else:
                check.add("sampling_frequency_missing_or_invalid", "nirs_json.SamplingFrequency", "A declared sampling frequency is required to assess time gaps.")

    # Validate channel sidecar types/units and its semantic measurement keys.
    channel_rows: list[dict[str, str]] = []
    if channel_table is not None:
        headers, channel_rows = channel_table
        required_columns = ("name", "type", "source", "detector", "wavelength_nominal", "units")
        for required in required_columns:
            if required not in headers:
                check.add("channels_column_missing", f"channels_tsv.{required}", f"channels.tsv is missing required column {required!r}.")
        present_required = [header for header in headers if header in required_columns]
        if len(present_required) == len(required_columns) and tuple(present_required) != required_columns:
            check.add("channels_column_order_invalid", "channels_tsv", "Required BIDS NIRS columns must occur in the order name, type, source, detector, wavelength_nominal, units.")
        if len(channel_rows) != summary["channel_count"]:
            check.add("channels_row_count_mismatch", "channels_tsv", f"Found {len(channel_rows)} channel rows for {summary['channel_count']} signal columns.")
        if "name" in headers:
            names = [row.get("name", "") for row in channel_rows]
            if any(not name for name in names):
                check.add("channel_names_invalid", "channels_tsv.name", "Channel names must be nonempty; repeated pair names across wavelengths are allowed.")
        if classification is not None and "type" in headers and "units" in headers:
            expected_types = {
                "cw_amplitude": {"nirscwamplitude"},
                # NIRSCWOPTICALDENSITY is the BIDS channel type; NIRSDOD is
                # retained for older datasets that used MNE's shorter label.
                "optical_density": {"nirscwopticaldensity", "nirsdod"},
                "haemoglobin": None,
            }.get(classification)
            allowed_units = _allowed_units(classification)
            for index, row in enumerate(channel_rows):
                channel_type = row.get("type", "").strip().lower()
                channel_unit = _normalized_unit(row.get("units", ""))
                if expected_types is not None and channel_type not in expected_types:
                    check.add("channel_type_mismatch", f"channels_tsv.row[{index + 1}].type", f"Expected one of {sorted(expected_types)} for {classification}; found {row.get('type', '')!r}.")
                elif classification == "haemoglobin" and channel_type not in {"nirscwhbo", "nirscwhbr", "hbo", "hbr"}:
                    check.add("channel_type_mismatch", f"channels_tsv.row[{index + 1}].type", f"Expected NIRSCWHBO or NIRSCWHBR for haemoglobin input; found {row.get('type', '')!r}.")
                if channel_unit not in allowed_units:
                    check.add("channel_unit_mismatch", f"channels_tsv.row[{index + 1}].units", f"Unit {row.get('units', '')!r} is not recognized for {classification}.")

    # Validate measurement indices, wavelengths, and paired channels.
    measurement_keys: list[tuple[str, str, str]] = []
    pair_to_suffixes: dict[tuple[str, str], set[str]] = {}
    ordered_measurement_keys: list[dict[str, Any]] = []
    if classification in {"cw_amplitude", "optical_density", "haemoglobin"}:
        if classification in {"cw_amplitude", "optical_density"} and len(summary["wavelengths_nm"]) != 2:
            check.add("wavelength_count_unsupported", "snirf_path", f"Expected two wavelengths for {classification}; found {len(summary['wavelengths_nm'])}.")
        for index, measurement in enumerate(measurements):
            src_i, det_i = measurement["source_index"], measurement["detector_index"]
            if not (1 <= src_i <= len(source_labels)) or not (1 <= det_i <= len(detector_labels)):
                check.add("measurement_optode_index_invalid", measurement["name"], "Source/detector indices must refer to declared optodes.")
                continue
            source, detector = source_labels[src_i - 1], detector_labels[det_i - 1]
            if classification in {"cw_amplitude", "optical_density"}:
                wavelength_index = measurement["wavelength_index"]
                if wavelength_index is None or not (1 <= wavelength_index <= len(summary["wavelengths_nm"])):
                    check.add("wavelength_index_invalid", measurement["name"], "Each CW amplitude/OD channel must reference a declared wavelength.")
                    continue
                suffix = format(float(summary["wavelengths_nm"][wavelength_index - 1]), "g")
            else:
                label = measurement["label"].strip().lower().replace(" ", "")
                suffix = "hbo" if label == "hbo" else "hbr" if label == "hbr" else ""
                if not suffix:
                    continue
            key = (source, detector, suffix)
            if key in measurement_keys:
                check.add("measurement_duplicate", measurement["name"], f"Duplicate channel measurement for {key}.")
            measurement_keys.append(key)
            ordered_measurement_keys.append(
                {
                    "source": source,
                    "detector": detector,
                    "wavelength_or_chromophore": suffix,
                }
            )
            pair_to_suffixes.setdefault((source, detector), set()).add(suffix)
        summary["measurement_pair_count"] = len(pair_to_suffixes)
        summary["channel_semantic_keys"] = ordered_measurement_keys
        summary["measurement_pair_keys"] = [
            {"source": source, "detector": detector}
            for source, detector in pair_to_suffixes
        ]
        if classification in {"cw_amplitude", "optical_density"} and len(summary["wavelengths_nm"]) == 2:
            expected = {format(float(value), "g") for value in summary["wavelengths_nm"]}
            for pair, suffixes in pair_to_suffixes.items():
                if suffixes != expected:
                    check.add("wavelength_pair_incomplete", "snirf_path", f"Pair {pair} has wavelength set {sorted(suffixes)}, expected {sorted(expected)}.")
        elif classification == "haemoglobin":
            for pair, suffixes in pair_to_suffixes.items():
                if suffixes != {"hbo", "hbr"}:
                    check.add("haemoglobin_pair_incomplete", "snirf_path", f"Pair {pair} does not have exactly one HbO and one HbR channel.")

        semantic_columns = {"source", "detector"} <= set(channel_table[0] if channel_table else [])
        if classification in {"cw_amplitude", "optical_density"}:
            semantic_columns = semantic_columns and "wavelength_nominal" in (channel_table[0] if channel_table else [])
        if channel_rows and semantic_columns:
            sidecar_keys = [_sidecar_channel_key(row, classification) for row in channel_rows]
            expected_keys = set(measurement_keys)
            if any(key is None for key in sidecar_keys) or len(set(sidecar_keys)) != len(sidecar_keys) or set(sidecar_keys) != expected_keys:
                check.add("channel_measurement_mapping_mismatch", "channels_tsv", "BIDS source/detector/wavelength-or-type fields do not map one-to-one to SNIRF measurements.")

    # Match source/detector labels and 3-D coordinates to BIDS optodes.tsv.
    if optode_table is not None and coordsystem is not None:
        headers, optode_rows = optode_table
        for required in ("name", "type", "x", "y", "z"):
            if required not in headers:
                check.add("optodes_column_missing", f"optodes_tsv.{required}", f"optodes.tsv is missing required column {required!r}.")
        coordinate_system = coordsystem.get("NIRSCoordinateSystem")
        coordinate_unit = str(coordsystem.get("NIRSCoordinateUnits", "")).strip().lower()
        coordinate_scale = _LENGTH_TO_METERS.get(coordinate_unit)
        if not isinstance(coordinate_system, str) or not coordinate_system.strip():
            check.add("coordinate_system_missing", "coordsystem_json.NIRSCoordinateSystem", "NIRSCoordinateSystem must be a nonempty string.")
        if coordinate_scale is None:
            check.add("coordinate_unit_unsupported", "coordsystem_json.NIRSCoordinateUnits", f"Unsupported or missing NIRSCoordinateUnits {coordinate_unit!r}.")
        if "name" in headers and "type" in headers:
            sidecar_optodes: dict[tuple[str, str], dict[str, str]] = {}
            for index, row in enumerate(optode_rows):
                name, kind = row.get("name", ""), row.get("type", "").strip().lower()
                if kind not in {"source", "detector"} or not name:
                    check.add("optode_row_invalid", f"optodes_tsv.row[{index + 1}]", "Each optode needs a name and type source or detector.")
                    continue
                key = (kind, name)
                if key in sidecar_optodes:
                    check.add("optode_duplicate", f"optodes_tsv.row[{index + 1}]", f"Duplicate optode {key}.")
                sidecar_optodes[key] = row
            expected_optodes = {("source", label) for label in source_labels} | {("detector", label) for label in detector_labels}
            if expected_optodes and set(sidecar_optodes) != expected_optodes:
                check.add("optode_label_set_mismatch", "optodes_tsv", "BIDS optode labels/types do not match SNIRF source and detector labels.")
            if coordinate_scale is not None and snirf_length_scale is not None and source_positions is not None and detector_positions is not None:
                coordinate_mismatches: list[str] = []
                for kind, labels, positions in (("source", source_labels, source_positions), ("detector", detector_labels, detector_positions)):
                    for index, label in enumerate(labels):
                        row = sidecar_optodes.get((kind, label))
                        if row is None:
                            continue
                        for axis, coordinate in enumerate(("x", "y", "z")):
                            raw_value = row.get(coordinate, "")
                            try:
                                sidecar_m = float(raw_value) * coordinate_scale
                            except (TypeError, ValueError):
                                check.add("optode_coordinate_invalid", f"optodes_tsv.{label}.{coordinate}", "Optode coordinates must be finite numeric values.")
                                continue
                            if not math.isfinite(sidecar_m):
                                check.add("optode_coordinate_invalid", f"optodes_tsv.{label}.{coordinate}", "Optode coordinates must be finite numeric values.")
                                continue
                            expected_m = float(positions[index, axis])
                            tolerance = _coordinate_rounding_tolerance(raw_value, coordinate_scale, expected_m)
                            if abs(sidecar_m - expected_m) > tolerance:
                                coordinate_mismatches.append(f"{kind} {label} {coordinate}")
                if coordinate_mismatches:
                    check.add("optode_coordinates_mismatch", "optodes_tsv", "Coordinates disagree between BIDS optodes.tsv and SNIRF probe positions: " + ", ".join(coordinate_mismatches[:12]) + ".")
                elif not any(issue.code == "optode_coordinate_invalid" for issue in check.issues):
                    pair_distances: dict[str, float] = {}
                    for source, detector in pair_to_suffixes:
                        source_index = source_labels.index(source)
                        detector_index = detector_labels.index(detector)
                        distance = float(np.linalg.norm(source_positions[source_index] - detector_positions[detector_index]))
                        pair_distances[f"{source}_{detector}"] = distance
                    summary["short_channel_pair_count"] = sum(distance < _SHORT_CHANNEL_CUTOFF_M for distance in pair_distances.values())
                    summary["short_channel_selector_m"] = _SHORT_CHANNEL_CUTOFF_M
                    summary["source_detector_distances_m"] = pair_distances

    # Validate event times against the SNIRF's actual sample support.
    if event_table is not None:
        headers, events = event_table
        for required in ("onset", "duration"):
            if required not in headers:
                check.add("events_column_missing", f"events_tsv.{required}", f"events.tsv is missing required column {required!r}.")
        summary["event_count"] = len(events)
        supported_events = 0
        if time_values is not None and len(time_values) >= 2 and np.isfinite(time_values).all() and np.all(np.diff(time_values) > 0) and nirs is not None:
            try:
                event_sfreq = float(nirs["SamplingFrequency"])
            except (KeyError, TypeError, ValueError):
                event_sfreq = None
            for index, event in enumerate(events):
                onset_token = event.get("onset", "").strip().lower()
                if onset_token in {"n/a", "na"}:
                    check.add("event_onset_unavailable", f"events_tsv.row[{index + 1}].onset", "BIDS onset is n/a, so this event has no supported time location.", "unsupported")
                    continue
                try:
                    onset = float(onset_token)
                    if not math.isfinite(onset):
                        raise ValueError
                except (TypeError, ValueError):
                    check.add("event_onset_invalid", f"events_tsv.row[{index + 1}].onset", "Event onset must be finite numeric seconds.")
                    continue
                duration_token = event.get("duration", "").strip().lower()
                duration = None
                if duration_token not in {"", "n/a", "na"}:
                    try:
                        duration = float(duration_token)
                        if not math.isfinite(duration) or duration < 0:
                            raise ValueError
                    except (TypeError, ValueError):
                        check.add("event_duration_invalid", f"events_tsv.row[{index + 1}].duration", "Event duration must be finite nonnegative seconds or n/a.")
                        continue
                # BIDS onset/sample values are relative to run start, whereas
                # SNIRF's time vector can carry a nonzero absolute origin.
                run_duration = float(time_values[-1] - time_values[0])
                end = onset if duration is None else onset + duration
                if end > run_duration:
                    check.add("event_duration_outside_time_support", f"events_tsv.row[{index + 1}]", "Event duration extends beyond the SNIRF recording support.")
                    continue
                if not 0.0 <= onset <= run_duration:
                    check.add("event_onset_outside_time_support", f"events_tsv.row[{index + 1}].onset", "BIDS event onset is outside the measured run support and has no corresponding SNIRF sample.", "unsupported")
                    continue
                if event.get("sample", "").strip() and event_sfreq is not None and math.isfinite(event_sfreq) and event_sfreq > 0:
                    try:
                        sample_value = float(event["sample"])
                        expected_sample = int(round(onset * event_sfreq))
                        if not math.isfinite(sample_value) or not sample_value.is_integer() or int(sample_value) != expected_sample:
                            check.add("event_sample_onset_mismatch", f"events_tsv.row[{index + 1}].sample", f"Sample index does not match onset at SamplingFrequency; expected {expected_sample}.")
                            continue
                    except (TypeError, ValueError, OverflowError):
                        check.add("event_sample_invalid", f"events_tsv.row[{index + 1}].sample", "Event sample must be an integer when provided.")
                        continue
                supported_events += 1
        summary["event_supported_count"] = supported_events

    return FnirsMetadataReport(classification, supported_for_v0, tuple(check.issues), summary)
