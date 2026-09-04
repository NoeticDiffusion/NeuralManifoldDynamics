"""Outcome-blind native EEG/time-base audit for FAR-EXT-002B.

This module intentionally stops before preprocessing, feature extraction,
MNPS construction, or FAR calculation.  It binds the semantic event ledger
from FAR-EXT-002A and audits only native EEGLAB samples, native event
latencies, recording continuity, technical signal QC, and the frozen NMD
temporal lattice.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping

import numpy as np


PROTOCOL_ID = "FAR-EXT-002B"
DATASET_ID = "ds003670"
NMD_WINDOW_SEC = 8.0
NMD_OVERLAP = 0.5
NMD_STEP_SEC = 4.0
PRE_SEC = 30.0
HORIZONS_SEC = (60.0, 30.0, 20.0, 16.0)
MIN_BIOLOGICAL_UNITS = 2
PROMOTED_EVENT_COUNT = 323
EXPECTED_EEG_CHANNEL_COUNT = 32

SIGNAL_TIMEBASE_PASS = "SIGNAL_TIMEBASE_PASS"
RAW_SIGNAL_ONLY_PASS = "RAW_SIGNAL_ONLY_PASS"
POST_STIM_ARTIFACT_NOT_TESTABLE = "POST_STIM_ARTIFACT_NOT_TESTABLE"
CLOCK_UNRESOLVED = "CLOCK_UNRESOLVED"
NMD_TIMEBASE_METHOD_LIMITED = "NMD_TIMEBASE_METHOD_LIMITED"
INSUFFICIENT_BIOLOGICAL_SUPPORT = "INSUFFICIENT_BIOLOGICAL_SUPPORT"
SOURCE_BINDING_FAILED = "SOURCE_BINDING_FAILED"
SOURCE_SYNC_UNRESOLVED = "SOURCE_SYNC_UNRESOLVED"

_EEGLAB_EVENT_CODES = {"16", "32"}
_AUXILIARY_NAMES = {"ecg", "eog", "resp"}
_TARGET_ORDER = {"frontal": 0, "motor": 1, "parietal": 2}
_FREQUENCY_ORDER = {0: 0, 5: 1, 30: 2}
_DOWNLOAD_BOOKKEEPING_NAMES = {
    "README",
    "README.md",
    "CHANGES",
    "LICENSE",
    "dataset_description.json",
    "participants.json",
    "participants.tsv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_relative(value: object) -> str:
    return str(value).replace("\\", "/").lstrip("./")


def _finite_number(value: object) -> float | None:
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.reshape(-1)[0]
    if isinstance(value, np.generic):
        value = value.item()
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        if value.size == 1:
            return _text(value.reshape(-1)[0])
        if value.dtype.kind in {"U", "S"}:
            return "".join(str(item) for item in value.reshape(-1)).strip()
    if isinstance(value, np.generic):
        value = value.item()
    text = str(value).strip()
    return text or None


def _unwrap(value: object) -> object:
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _field(value: object, name: str) -> object | None:
    value = _unwrap(value)
    if isinstance(value, Mapping):
        return value.get(name)
    if isinstance(value, np.void) and value.dtype.names and name in value.dtype.names:
        return value[name]
    try:
        return getattr(value, name)
    except AttributeError:
        return None


def _records(value: object) -> list[object]:
    value = _unwrap(value)
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        if value.dtype.names:
            return list(value.reshape(-1))
        if value.dtype == object:
            return list(value.reshape(-1))
    return [value]


def _code(value: object) -> str | None:
    text = _text(value)
    if text is None:
        return None
    match = re.search(r"(?<!\d)(\d+)(?:\.0+)?(?!\d)", text)
    return match.group(1) if match else text.strip().lower()


def _family_id(v: Mapping[str, Any]) -> str:
    target = str(v.get("target", "")).strip().lower()
    waveform_raw = str(v.get("waveform", "")).strip().lower()
    if waveform_raw in {"dc", "direct_current", "direct-current"}:
        waveform = "dc"
    elif "sin" in waveform_raw:
        waveform = "sinusoidal"
    else:
        waveform = re.sub(r"[^a-z0-9]+", "_", waveform_raw).strip("_")
    frequency = _finite_number(v.get("frequency_hz"))
    if frequency is None:
        frequency_text = "unknown"
    elif frequency.is_integer():
        frequency_text = str(int(frequency))
    else:
        frequency_text = format(frequency, "g")
    return f"{target}_{waveform}_{frequency_text}hz"


def _family_sort_key(family_id: str) -> tuple[int, int, str]:
    parts = family_id.split("_")
    target = parts[0] if parts else ""
    frequency_match = re.search(r"_(\d+(?:\.\d+)?)hz$", family_id)
    frequency = float(frequency_match.group(1)) if frequency_match else math.inf
    return (
        _TARGET_ORDER.get(target, 99),
        _FREQUENCY_ORDER.get(int(frequency) if frequency.is_integer() else -1, 99),
        family_id,
    )


def _is_auxiliary_channel(name: str) -> bool:
    normalized = name.strip().lower()
    return normalized in _AUXILIARY_NAMES or normalized.startswith(
        ("bip", "resp")
    )


def _event_key(event: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "source_file": _normalise_relative(event.get("source_file", "")),
            "subject": event.get("subject"),
            "session": event.get("session"),
            "block": event.get("block"),
            "trial_in_block": event.get("trial_in_block"),
            "trigger_start": event.get("trigger_start"),
            "v": event.get("v"),
            "rho": event.get("rho"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def promoted_event_ledger(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the immutable 002A promotion ledger, or fail closed."""
    if payload.get("protocol_id") != "FAR-EXT-002A":
        raise ValueError("far_ext_002a_protocol_id_mismatch")
    if payload.get("global_status") != "DS003670_LIMITED_CURVE_SEMANTICS_PASS":
        raise ValueError("far_ext_002a_global_status_not_pass")
    ledger: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_event in payload.get("events", []):
        if not isinstance(source_event, Mapping):
            continue
        isolated = _finite_number(source_event.get("T_isolated"))
        if not (
            source_event.get("semantic_status") == "PASS"
            and bool(source_event.get("rho_contrast_eligible"))
            and isolated is not None
            and isolated > NMD_WINDOW_SEC
            and int(source_event.get("experiment", 0) or 0) == 1
        ):
            continue
        v = source_event.get("v")
        if not isinstance(v, Mapping):
            raise ValueError("far_ext_002a_promoted_event_missing_v")
        event = dict(source_event)
        event["source_file"] = _normalise_relative(event.get("source_file", ""))
        event["family_id"] = _family_id(v)
        key = _event_key(event)
        if key in seen:
            raise ValueError("far_ext_002a_duplicate_promoted_event")
        seen.add(key)
        event["event_key"] = key
        ledger.append(event)
    if len(ledger) != PROMOTED_EVENT_COUNT:
        raise ValueError(
            f"far_ext_002a_promoted_event_count_mismatch:{len(ledger)}"
        )
    return ledger


def _load_mat(set_path: Path) -> object:
    try:
        import scipy.io as sio
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("scipy_required_for_eeglab_set") from exc
    try:
        matrix = sio.loadmat(
            str(set_path),
            squeeze_me=True,
            struct_as_record=False,
        )
    except Exception as exc:  # noqa: BLE001 - source-format boundary
        raise RuntimeError(f"eeglab_set_unreadable:{set_path}") from exc
    eeg = matrix.get("EEG")
    if eeg is not None:
        return _unwrap(eeg)
    # EEGLAB also writes a flat MATLAB workspace for some legacy exports,
    # including ds003670.  Keep only the fields needed by this audit so the
    # potentially very large `times` vector is not retained.
    required_fields = {
        "data",
        "datfile",
        "srate",
        "pnts",
        "nbchan",
        "xmin",
        "chanlocs",
        "event",
        "reject",
    }
    flat = {
        key: value
        for key, value in matrix.items()
        if key in required_fields
    }
    if {"srate", "pnts", "nbchan"}.issubset(flat):
        return flat
    raise RuntimeError(f"eeglab_set_missing_EEG_struct:{set_path}")


def _data_reference(eeg: object) -> str | None:
    for name in ("datfile", "data"):
        raw_candidate = _field(eeg, name)
        if (
            isinstance(raw_candidate, np.ndarray)
            and raw_candidate.size > 1
            and np.issubdtype(raw_candidate.dtype, np.number)
        ):
            continue
        candidate = _text(raw_candidate)
        if candidate and candidate.lower().endswith(".fdt"):
            return candidate
    return None


def referenced_fdt_path(set_path: Path) -> Path | None:
    """Read only the EEGLAB header and resolve its declared FDT companion."""
    eeg = _load_mat(set_path)
    reference = _data_reference(eeg)
    if reference is None:
        return None
    reference = reference.replace("\\", "/")
    relative = PurePosixPath(reference)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"eeglab_fdt_reference_outside_recording:{set_path}")
    return set_path.parent / Path(*relative.parts)


@dataclass
class NativeEEG:
    set_path: Path
    sfreq: float
    n_samples: int
    n_channels: int
    channel_names: list[str]
    events: list[dict[str, Any]]
    boundary_times: list[float]
    bad_intervals: list[tuple[float, float]]
    bad_channel_indices: list[int]
    data: Any
    data_dtype: np.dtype
    fdt_path: Path | None
    rail_min: int | None
    rail_max: int | None
    xmin: float


def _channel_names(eeg: object) -> list[str]:
    values = _field(eeg, "chanlocs")
    names: list[str] = []
    for record in _records(values):
        label = _text(_field(record, "labels"))
        names.append(label or f"channel_{len(names):03d}")
    return names


def _native_events(
    eeg: object,
    sfreq: float,
) -> tuple[list[dict[str, Any]], list[float], list[tuple[float, float]]]:
    events: list[dict[str, Any]] = []
    boundaries: list[float] = []
    bad_intervals: list[tuple[float, float]] = []
    for record in _records(_field(eeg, "event")):
        raw_type = _text(_field(record, "type")) or ""
        code = _code(raw_type)
        latency = _finite_number(_field(record, "latency"))
        if code is None or latency is None:
            continue
        sample_index = int(round(latency)) - 1
        time_sec = sample_index / sfreq
        duration = _finite_number(_field(record, "duration"))
        item = {
            "code": code,
            "type": raw_type,
            "latency_eeglab": latency,
            "sample_index_0": sample_index,
            "time_sec": time_sec,
            "duration_sec": duration,
        }
        events.append(item)
        lower_type = raw_type.lower()
        is_boundary = code == "boundary" or "boundary" in lower_type
        is_bad = (
            is_boundary
            or "artifact" in lower_type
            or lower_type.startswith("bad")
        )
        if is_boundary:
            boundaries.append(time_sec)
        if is_bad:
            bad_intervals.append(
                (time_sec, time_sec + max(0.0, duration or 0.0))
            )
    return events, boundaries, bad_intervals


def _bad_channel_indices(eeg: object, n_channels: int) -> list[int]:
    reject = _field(eeg, "reject")
    for field_name in ("rejchan", "badchans"):
        values = _field(reject, field_name)
        if values is None:
            continue
        array = np.asarray(values).reshape(-1)
        if array.size != n_channels:
            continue
        try:
            return [
                index
                for index, value in enumerate(array)
                if bool(value)
            ]
        except (TypeError, ValueError):
            continue
    return []


def _embedded_data(value: object, n_channels: int, n_samples: int) -> np.ndarray | None:
    value = _unwrap(value)
    if not isinstance(value, np.ndarray) or not np.issubdtype(value.dtype, np.number):
        return None
    array = np.asarray(value)
    if array.size != n_channels * n_samples:
        return None
    return array.reshape((n_channels, n_samples), order="F")


def read_native_eeglab(set_path: Path) -> NativeEEG:
    eeg = _load_mat(set_path)
    sfreq = _finite_number(_field(eeg, "srate"))
    n_samples_number = _finite_number(_field(eeg, "pnts"))
    n_channels_number = _finite_number(_field(eeg, "nbchan"))
    if sfreq is None or sfreq <= 0 or n_samples_number is None or n_channels_number is None:
        raise RuntimeError(f"eeglab_header_incomplete:{set_path}")
    n_samples = int(n_samples_number)
    n_channels = int(n_channels_number)
    names = _channel_names(eeg)
    events, boundaries, bad_intervals = _native_events(eeg, sfreq)
    embedded = _embedded_data(_field(eeg, "data"), n_channels, n_samples)
    fdt_path = None
    rail_min: int | None = None
    rail_max: int | None = None
    if embedded is not None:
        data: Any = embedded
        data_dtype = data.dtype
    else:
        fdt_path = referenced_fdt_path(set_path)
        if fdt_path is None:
            raise RuntimeError(f"eeglab_external_data_reference_missing:{set_path}")
        if not fdt_path.is_file():
            raise RuntimeError(f"eeglab_referenced_fdt_missing:{fdt_path}")
        expected_values = n_channels * n_samples
        size = fdt_path.stat().st_size
        if size == expected_values * np.dtype("<f4").itemsize:
            data_dtype = np.dtype("<f4")
        elif size == expected_values * np.dtype("<f8").itemsize:
            data_dtype = np.dtype("<f8")
        else:
            raise RuntimeError(f"eeglab_fdt_size_mismatch:{fdt_path}")
        data = np.memmap(
            fdt_path,
            dtype=data_dtype,
            mode="r",
            shape=(n_channels, n_samples),
            order="F",
        )
    if np.issubdtype(data_dtype, np.integer):
        info = np.iinfo(data_dtype)
        rail_min, rail_max = int(info.min), int(info.max)
    xmin = _finite_number(_field(eeg, "xmin"))
    return NativeEEG(
        set_path=set_path,
        sfreq=sfreq,
        n_samples=n_samples,
        n_channels=n_channels,
        channel_names=names,
        events=events,
        boundary_times=boundaries,
        bad_intervals=bad_intervals,
        bad_channel_indices=_bad_channel_indices(eeg, n_channels),
        data=data,
        data_dtype=np.dtype(data_dtype),
        fdt_path=fdt_path,
        rail_min=rail_min,
        rail_max=rail_max,
        xmin=0.0 if xmin is None else xmin,
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"metadata_json_unreadable:{path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"metadata_json_not_object:{path}")
    return value


def _sidecar_path(root: Path, relative_set: str) -> Path:
    return root / Path(relative_set.replace("_eeg.set", "_eeg.json"))


def _required_eeg_indices(
    native: NativeEEG,
    metadata_root: Path,
    relative_set: str,
) -> tuple[list[int], dict[str, Any]]:
    sidecar = _read_json(_sidecar_path(metadata_root, relative_set))
    declared = _finite_number(sidecar.get("EEGChannelCount"))
    if declared is None or int(declared) != EXPECTED_EEG_CHANNEL_COUNT:
        raise RuntimeError("eeg_channel_count_not_32")
    indices = [
        index
        for index, name in enumerate(native.channel_names)
        if not _is_auxiliary_channel(name)
    ]
    if len(indices) != EXPECTED_EEG_CHANNEL_COUNT:
        raise RuntimeError(
            f"eeg_channel_name_selection_not_exact:{len(indices)}"
        )
    if max(indices, default=-1) >= native.n_channels:
        raise RuntimeError("eeg_channel_index_out_of_range")
    if set(indices).intersection(native.bad_channel_indices):
        raise RuntimeError("native_bad_eeg_channel_annotation")
    return indices, {
        "sidecar_path": str(_sidecar_path(metadata_root, relative_set)),
        "declared_eeg_channel_count": int(declared),
        "selected_channel_names": [native.channel_names[index] for index in indices],
    }


def _event_files_from_ledger(ledger: Iterable[Mapping[str, Any]]) -> list[str]:
    values = {
        _normalise_relative(event.get("source_file", ""))
        for event in ledger
    }
    if any(not value.endswith("_events.tsv") for value in values):
        raise ValueError("far_ext_002a_source_file_not_events_tsv")
    return sorted(values)


def eeg_payload_allowlist(ledger: Iterable[Mapping[str, Any]]) -> list[str]:
    """Return exact `.set` paths required by the frozen event ledger."""
    result = []
    for event_file in _event_files_from_ledger(ledger):
        result.append(event_file.replace("_events.tsv", "_eeg.set"))
    return result


def _inventory_payload_files(root: Path) -> tuple[list[str], list[str]]:
    payload_suffixes = {".set", ".fdt"}
    payloads: list[str] = []
    unexpected: list[str] = []
    if not root.is_dir():
        return payloads, unexpected
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = _normalise_relative(path.relative_to(root))
        if path.suffix.lower() in payload_suffixes:
            payloads.append(relative)
        elif path.name not in _DOWNLOAD_BOOKKEEPING_NAMES:
            unexpected.append(relative)
    return payloads, unexpected


def bind_payload_root(
    root: Path,
    set_paths: Iterable[str],
) -> dict[str, Any]:
    expected_sets = {_normalise_relative(path) for path in set_paths}
    if not root.is_dir():
        return {
            "status": SOURCE_BINDING_FAILED,
            "reason": "eeg_payload_root_missing",
            "expected_payloads": sorted(expected_sets),
            "actual_payloads": [],
            "unexpected_files": [],
        }
    actual_payloads, unexpected = _inventory_payload_files(root)
    if unexpected:
        return {
            "status": SOURCE_BINDING_FAILED,
            "reason": "unexpected_non_payload_files",
            "expected_payloads": sorted(expected_sets),
            "actual_payloads": actual_payloads,
            "unexpected_files": unexpected,
        }
    expected_payloads = set(expected_sets)
    reference_errors: list[str] = []
    for relative_set in sorted(expected_sets):
        set_path = root / Path(relative_set)
        try:
            fdt_path = referenced_fdt_path(set_path)
        except RuntimeError as exc:
            reference_errors.append(str(exc))
            continue
        if fdt_path is not None:
            expected_payloads.add(
                _normalise_relative(fdt_path.relative_to(root))
            )
    missing_payloads = sorted(expected_payloads - set(actual_payloads))
    extra_payloads = sorted(set(actual_payloads) - expected_payloads)
    if reference_errors or missing_payloads or extra_payloads:
        return {
            "status": SOURCE_BINDING_FAILED,
            "reason": (
                "eeglab_fdt_reference_unresolved"
                if reference_errors
                else "allowlisted_payload_inventory_mismatch"
            ),
            "expected_payloads": sorted(expected_payloads),
            "actual_payloads": actual_payloads,
            "missing": missing_payloads,
            "unexpected_files": extra_payloads,
            "reference_errors": reference_errors,
        }
    return {
        "status": "PASS",
        "reason": "exact_payload_inventory",
        "expected_payloads": sorted(expected_payloads),
        "actual_payloads": actual_payloads,
        "unexpected_files": [],
    }


def _clock_for_events(
    native: NativeEEG,
    events: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    available = [
        item
        for item in native.events
        if item.get("code") in _EEGLAB_EVENT_CODES
    ]
    used: set[int] = set()
    matches: list[dict[str, Any]] = []
    failures: list[str] = []
    by_event: dict[str, dict[str, Any]] = {}
    ordered_events = sorted(
        events,
        key=lambda item: float(item.get("trigger_start", math.inf)),
    )
    for event in ordered_events:
        event_key = str(event.get("event_key"))
        event_matches: list[dict[str, Any]] = []
        event_failures: list[str] = []
        for code, field in (("16", "trigger_start"), ("32", "trigger_max_end")):
            expected = _finite_number(event.get(field))
            if expected is None:
                failure = f"{field}_missing"
                event_failures.append(failure)
                failures.append(f"{event_key}:{failure}")
                continue
            candidates = [
                (index, item)
                for index, item in enumerate(available)
                if index not in used and item["code"] == code
            ]
            if not candidates:
                failure = f"{code}_missing"
                event_failures.append(failure)
                failures.append(f"{event_key}:{failure}")
                continue
            index, item = min(
                candidates,
                key=lambda pair: abs(float(pair[1]["time_sec"]) - expected),
            )
            difference = abs(float(item["time_sec"]) - expected)
            if difference > (1.0 / native.sfreq):
                failure = f"{code}_difference_{difference:g}"
                event_failures.append(failure)
                failures.append(f"{event_key}:{failure}")
                continue
            used.add(index)
            match = {
                "event_key": event.get("event_key"),
                "code": code,
                "expected_sec": expected,
                "native_sec": item["time_sec"],
                "difference_sec": difference,
                "sample_index_0": item["sample_index_0"],
            }
            matches.append(match)
            event_matches.append(match)
        by_event[event_key] = {
            "status": "PASS" if not event_failures else CLOCK_UNRESOLVED,
            "matches": event_matches,
            "failures": event_failures,
        }
    return {
        "status": "PASS" if not failures else CLOCK_UNRESOLVED,
        "promoted_event_count": len(ordered_events),
        "matched_trigger_count": len(matches),
        "expected_trigger_count": len(ordered_events) * 2,
        "matches": matches,
        "failures": failures,
        "by_event": by_event,
        "latency_conversion": "round(latency_eeglab) - 1; t = sample_index_0 / sfreq",
        "tolerance_sec": 1.0 / native.sfreq,
    }


def _sample_bounds(start_sec: float, end_sec: float, sfreq: float, n_samples: int) -> tuple[int, int] | None:
    start = int(math.ceil(start_sec * sfreq - 1.0e-9))
    end = int(math.ceil(end_sec * sfreq - 1.0e-9))
    if start < 0 or end > n_samples or end <= start:
        return None
    return start, end


def _window_qc(
    native: NativeEEG,
    channel_indices: list[int],
    start_sec: float,
    end_sec: float,
) -> dict[str, Any]:
    bounds = _sample_bounds(start_sec, end_sec, native.sfreq, native.n_samples)
    if bounds is None:
        return {
            "status": POST_STIM_ARTIFACT_NOT_TESTABLE,
            "reason": "recording_boundary",
            "start_sec": start_sec,
            "end_sec": end_sec,
        }
    start, end = bounds
    if any(start_sec <= boundary <= end_sec for boundary in native.boundary_times) or any(
        interval_start <= end_sec and interval_end >= start_sec
        for interval_start, interval_end in native.bad_intervals
    ):
        return {
            "status": POST_STIM_ARTIFACT_NOT_TESTABLE,
            "reason": "native_bad_segment_annotation",
            "start_sec": start_sec,
            "end_sec": end_sec,
        }
    try:
        values = np.asarray(native.data[channel_indices, start:end])
    except (OSError, ValueError, RuntimeError) as exc:
        return {
            "status": POST_STIM_ARTIFACT_NOT_TESTABLE,
            "reason": f"signal_read_error:{type(exc).__name__}",
            "start_sec": start_sec,
            "end_sec": end_sec,
        }
    finite = bool(np.isfinite(values).all())
    rail_hits = 0
    rail_status = "PASS"
    if native.rail_min is not None and native.rail_max is not None:
        rail_hits = int(
            np.count_nonzero((values == native.rail_min) | (values == native.rail_max))
        )
    else:
        rail_status = "UNRESOLVED"
    status = (
        "PASS"
        if finite and rail_hits == 0 and rail_status == "PASS"
        else POST_STIM_ARTIFACT_NOT_TESTABLE
    )
    reason = (
        "technical_qc_pass"
        if status == "PASS"
        else "non_finite_samples"
        if not finite
        else "native_rail_hit"
        if rail_hits
        else "native_rail_criterion_unresolved"
    )
    return {
        "status": status,
        "reason": reason,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "sample_start": start,
        "sample_end": end,
        "n_samples": int(end - start),
        "finite": finite,
        "rail_status": rail_status,
        "rail_hits": rail_hits,
        "variance_rule": "not_used",
        "blanking_rule": "native_metadata_only",
    }


def _grid_starts(
    interval_start: float,
    interval_end: float,
    *,
    window_sec: float = NMD_WINDOW_SEC,
    step_sec: float = NMD_STEP_SEC,
) -> list[float]:
    first = max(0, int(math.ceil(interval_start / step_sec - 1.0e-9)))
    last = int(math.floor((interval_end - window_sec) / step_sec + 1.0e-9))
    return [
        round(index * step_sec, 9)
        for index in range(first, last + 1)
        if index >= 0
    ]


def _nmd_windows(
    *,
    trigger_start: float,
    t0: float,
    horizon: float,
) -> dict[str, Any]:
    pre = _grid_starts(trigger_start - PRE_SEC, trigger_start)
    post = _grid_starts(t0, t0 + horizon)
    return {
        "pre_starts": pre,
        "post_starts": post,
        "pre_count": len(pre),
        "post_count": len(post),
        "pre_required": len(pre) >= 1,
        "post_required": len(post) >= 2,
    }


def _event_candidates(
    event: Mapping[str, Any],
    native: NativeEEG,
    channel_indices: list[int],
    clock_pass: bool,
    nmd_grid_pass: bool,
) -> dict[str, Any]:
    trigger_start = _finite_number(event.get("trigger_start"))
    t0 = _finite_number(event.get("ramp_down_end"))
    next_stim = _finite_number(event.get("next_stim_start"))
    if trigger_start is None or t0 is None:
        return {
            "event_key": event.get("event_key"),
            "clock_status": CLOCK_UNRESOLVED if not clock_pass else "PASS",
            "native_signal_status": POST_STIM_ARTIFACT_NOT_TESTABLE,
            "candidate_by_horizon": {},
            "reason": "event_timing_missing",
        }
    candidate_by_horizon: dict[str, dict[str, Any]] = {}
    for horizon in HORIZONS_SEC:
        post_end = t0 + horizon
        if next_stim is not None and post_end > next_stim + 1.0e-9:
            candidate_by_horizon[str(horizon)] = {
                "eligible": False,
                "reason": "next_stimulation_within_horizon",
                "horizon_sec": horizon,
            }
            continue
        pre_qc = _window_qc(
            native,
            channel_indices,
            trigger_start - PRE_SEC,
            trigger_start,
        )
        post_qc = _window_qc(native, channel_indices, t0, post_end)
        windows = _nmd_windows(
            trigger_start=trigger_start,
            t0=t0,
            horizon=horizon,
        )
        technical_eligible = bool(
            clock_pass
            and pre_qc["status"] == "PASS"
            and post_qc["status"] == "PASS"
        )
        eligible = bool(
            technical_eligible
            and nmd_grid_pass
            and windows["pre_required"]
            and windows["post_required"]
        )
        reason = (
            "eligible"
            if eligible
            else "nmd_lattice_unresolved"
            if technical_eligible and not nmd_grid_pass
            else "clock_unresolved"
            if not clock_pass
            else pre_qc["reason"]
            if pre_qc["status"] != "PASS"
            else post_qc["reason"]
            if post_qc["status"] != "PASS"
            else "insufficient_complete_nmd_windows"
        )
        candidate_by_horizon[str(horizon)] = {
            "eligible": eligible,
            "technical_eligible": technical_eligible,
            "reason": reason,
            "horizon_sec": horizon,
            "pre_qc": pre_qc,
            "post_qc": post_qc,
            "nmd": windows,
        }
    return {
        "event_key": event.get("event_key"),
        "clock_status": "PASS" if clock_pass else CLOCK_UNRESOLVED,
        "native_signal_status": (
            "PASS"
            if any(
                item.get("technical_eligible")
                for item in candidate_by_horizon.values()
            )
            else POST_STIM_ARTIFACT_NOT_TESTABLE
        ),
        "candidate_by_horizon": candidate_by_horizon,
    }


def _family_result(
    family_id: str,
    events: list[dict[str, Any]],
    event_candidates: list[dict[str, Any]],
    *,
    nmd_grid_pass: bool,
    nmd_binding: Mapping[str, Any],
) -> dict[str, Any]:
    by_horizon: dict[float, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for horizon in HORIZONS_SEC:
        by_horizon[horizon] = [
            (event, candidate["candidate_by_horizon"].get(str(horizon), {}))
            for event, candidate in zip(events, event_candidates)
            if candidate["candidate_by_horizon"].get(str(horizon), {}).get("eligible")
        ]
    selected_horizon: float | None = None
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    support_by_horizon: dict[str, dict[str, int]] = {}
    for horizon in HORIZONS_SEC:
        pairs = by_horizon[horizon]
        support: dict[str, set[str]] = defaultdict(set)
        for event, _candidate in pairs:
            rho = _finite_number(event.get("rho"))
            if rho is not None:
                support[str(rho)].add(str(event.get("biological_unit")))
        support_by_horizon[str(horizon)] = {
            rho: len(units) for rho, units in sorted(support.items())
        }
        if (
            selected_horizon is None
            and len(support.get("0.5", set())) >= MIN_BIOLOGICAL_UNITS
            and len(support.get("1.0", set())) >= MIN_BIOLOGICAL_UNITS
        ):
            selected_horizon = horizon
            selected = pairs
    rho_values = sorted(
        {
            float(event.get("rho"))
            for event in events
            if _finite_number(event.get("rho")) is not None
        }
    )
    subjects_per_rho: dict[str, list[str]] = {}
    events_per_rho: dict[str, int] = {}
    for rho in (0.5, 1.0):
        selected_units = sorted(
            {
                str(event.get("biological_unit"))
                for event, _candidate in selected
                if math.isclose(float(event.get("rho", math.nan)), rho)
            }
        )
        subjects_per_rho[str(rho)] = selected_units
        events_per_rho[str(rho)] = sum(
            1
            for event, _candidate in selected
            if math.isclose(float(event.get("rho", math.nan)), rho)
        )
    post_statuses = [
        candidate["candidate_by_horizon"]
        .get(str(selected_horizon), {})
        .get("post_qc", {})
        .get("status")
        for candidate in event_candidates
        if selected_horizon is not None
    ]
    any_clock_unresolved = any(
        candidate.get("clock_status") != "PASS"
        for candidate in event_candidates
    )
    selected_event_keys = {event.get("event_key") for event, _ in selected}
    selected_clock_unresolved = any(
        candidate.get("clock_status") != "PASS"
        for candidate in event_candidates
        if candidate.get("event_key") in selected_event_keys
    )
    technical_support_exists = any(
        candidate.get("native_signal_status") == "PASS"
        for candidate in event_candidates
    )
    if selected_clock_unresolved:
        status = CLOCK_UNRESOLVED
        reason = "promoted_event_clock_not_proven"
    elif selected_horizon is not None and nmd_grid_pass:
        status = SIGNAL_TIMEBASE_PASS
        reason = "joint_signal_clock_common_horizon_nmd_pass"
    elif not nmd_grid_pass and technical_support_exists:
        status = RAW_SIGNAL_ONLY_PASS
        reason = "native_signal_pass_nmd_lattice_unresolved"
    elif any_clock_unresolved and not technical_support_exists:
        status = CLOCK_UNRESOLVED
        reason = "promoted_event_clock_not_proven"
    elif any(item == POST_STIM_ARTIFACT_NOT_TESTABLE for item in post_statuses) or not technical_support_exists:
        status = POST_STIM_ARTIFACT_NOT_TESTABLE
        reason = "post_ramp_technical_qc_not_testable"
    else:
        status = INSUFFICIENT_BIOLOGICAL_SUPPORT
        reason = "no_common_horizon_with_two_subjects_per_rho"
    clean_post = (
        min(
            (
                pair[1]["horizon_sec"]
                for pair in selected
                if pair[1].get("post_qc", {}).get("status") == "PASS"
            ),
            default=None,
        )
    )
    return {
        "family_id": family_id,
        "v": events[0].get("v") if events else {},
        "target": events[0].get("target"),
        "waveform": events[0].get("waveform"),
        "frequency": events[0].get("frequency_hz"),
        "rho_levels": rho_values,
        "subjects_per_rho": subjects_per_rho,
        "events_per_rho": events_per_rho,
        "min_subjects_across_rho": min(
            (len(values) for values in subjects_per_rho.values()),
            default=0,
        ),
        "event_count": len(events),
        "promoted_event_keys": [event.get("event_key") for event in events],
        "candidate_support_by_horizon": support_by_horizon,
        "selected_post_horizon": selected_horizon,
        "clean_pre_sec": PRE_SEC if selected_horizon is not None else None,
        "clean_post_sec": clean_post,
        "clock_status": (
            "PASS"
            if selected_horizon is not None and not selected_clock_unresolved
            else CLOCK_UNRESOLVED
            if any_clock_unresolved
            else "PASS"
        ),
        "native_signal_status": (
            "PASS" if technical_support_exists else POST_STIM_ARTIFACT_NOT_TESTABLE
        ),
        "native_continuity_status": (
            "PASS" if technical_support_exists else "NOT_TESTABLE"
        ),
        "pre_signal_status": (
            "PASS"
            if technical_support_exists
            else POST_STIM_ARTIFACT_NOT_TESTABLE
        ),
        "during_stim_status": "DURING_STIM_ARTIFACT_EXPECTED",
        "post_ramp_artifact_status": (
            "PASS"
            if technical_support_exists
            else POST_STIM_ARTIFACT_NOT_TESTABLE
        ),
        "nmd_window_sec": NMD_WINDOW_SEC,
        "nmd_step_sec": NMD_STEP_SEC,
        "nmd_overlap": NMD_OVERLAP,
        "nmd_pre_window_count": min(
            (
                pair[1].get("nmd", {}).get("pre_count", 0)
                for pair in selected
            ),
            default=0,
        ),
        "nmd_post_window_count": min(
            (
                pair[1].get("nmd", {}).get("post_count", 0)
                for pair in selected
            ),
            default=0,
        ),
        "nmd_timebase_status": "PASS" if nmd_grid_pass else NMD_TIMEBASE_METHOD_LIMITED,
        "eligibility_status": status,
        "reason": reason,
        "tolerability_confound": True,
        "nmd_binding": dict(nmd_binding),
        "event_eligibility": event_candidates,
    }


def _config_binding(repo_root: Path) -> dict[str, Any]:
    config_path = repo_root / "mndm" / "config" / "config_ingest_common_eeg.yaml"
    if not config_path.is_file():
        return {
            "status": NMD_TIMEBASE_METHOD_LIMITED,
            "reason": "nmd_config_missing",
            "path": str(config_path),
        }
    try:
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - configuration boundary
        return {
            "status": NMD_TIMEBASE_METHOD_LIMITED,
            "reason": f"nmd_config_unreadable:{type(exc).__name__}",
            "path": str(config_path),
        }
    mnps = config.get("mnps", {}) if isinstance(config, Mapping) else {}
    epoching = config.get("epoching", {}) if isinstance(config, Mapping) else {}
    values = {
        "mnps.window_sec": _finite_number(mnps.get("window_sec")),
        "mnps.overlap": _finite_number(mnps.get("overlap")),
        "epoching.length_s": _finite_number(epoching.get("length_s")),
        "epoching.step_s": _finite_number(epoching.get("step_s")),
    }
    valid = (
        math.isclose(values["mnps.window_sec"] or math.nan, NMD_WINDOW_SEC)
        and math.isclose(values["mnps.overlap"] or math.nan, NMD_OVERLAP)
        and math.isclose(values["epoching.length_s"] or math.nan, NMD_WINDOW_SEC)
        and math.isclose(values["epoching.step_s"] or math.nan, NMD_STEP_SEC)
    )
    return {
        "status": "PASS" if valid else NMD_TIMEBASE_METHOD_LIMITED,
        "reason": "frozen_config_match" if valid else "frozen_config_mismatch",
        "path": str(config_path),
        "sha256": sha256_file(config_path),
        "values": values,
        "window_origin": "native_recording_sample_0",
        "half_open_intervals": True,
    }


def _dataset_source_binding(
    metadata_root: Path,
    far002_payload: Mapping[str, Any],
) -> dict[str, Any]:
    expected_root = Path(str(far002_payload.get("source_root", "")))
    expected_root_resolved = expected_root.resolve()
    actual_root_resolved = metadata_root.resolve()
    binding = far002_payload.get("source_binding", {})
    status = (
        binding.get("status") == "PASS"
        and expected_root_resolved == actual_root_resolved
        and not binding.get("signal_payloads")
        and not binding.get("unexpected_files")
    )
    return {
        "status": "PASS" if status else SOURCE_BINDING_FAILED,
        "expected_root": str(expected_root),
        "actual_root": str(metadata_root),
        "far_ext_002a_source_binding_status": binding.get("status"),
        "far_ext_002a_certificate_source_binding": binding,
    }


def audit_far_ext_002b(
    *,
    metadata_root: Path,
    eeg_root: Path,
    far002_payload: Mapping[str, Any],
    repo_root: Path,
    far002_path: Path,
) -> dict[str, Any]:
    """Audit the frozen 002A event population against native EEG payloads."""
    source_binding = _dataset_source_binding(metadata_root, far002_payload)
    ledger: list[dict[str, Any]] = []
    binding_error: str | None = None
    try:
        ledger = promoted_event_ledger(far002_payload)
    except ValueError as exc:
        binding_error = str(exc)
    set_paths = eeg_payload_allowlist(ledger) if ledger else []
    payload_binding = bind_payload_root(eeg_root, set_paths)
    nmd_binding = _config_binding(repo_root)
    nmd_grid_pass = nmd_binding.get("status") == "PASS"
    if binding_error or source_binding["status"] != "PASS" or payload_binding["status"] != "PASS":
        return {
            "schema": "mndm.far_ext_002b_ds003670_signal_timebase.v1",
            "protocol_id": PROTOCOL_ID,
            "dataset_id": DATASET_ID,
            "global_status": SOURCE_BINDING_FAILED,
            "global_reason": (
                binding_error
                or payload_binding.get("reason")
                or source_binding.get("reason")
                or source_binding.get("status")
            ),
            "entry_far_ext_002a": {
                "path": str(far002_path),
                "sha256": sha256_file(far002_path),
                "protocol_id": far002_payload.get("protocol_id"),
                "global_status": far002_payload.get("global_status"),
            },
            "source_binding": source_binding,
            "payload_binding": payload_binding,
            "nmd_binding": nmd_binding,
            "promoted_event_count": len(ledger),
            "family_ledger": [],
            "event_ledger": ledger,
            "audit_scope": {
                "metadata_opened": True,
                "signal_payloads_opened": False,
                "outcome_tables_opened": False,
                "nmd_outputs_opened": False,
                "mnps_calculated": False,
                "far_calculated": False,
            },
            "far_003b_authorized": False,
        }
    events_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    events_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in ledger:
        events_by_file[event["source_file"].replace("_events.tsv", "_eeg.set")].append(event)
        events_by_family[event["family_id"]].append(event)
    native_by_file: dict[str, NativeEEG] = {}
    file_audits: dict[str, dict[str, Any]] = {}
    all_file_errors: list[str] = []
    for relative_set, file_events in sorted(events_by_file.items()):
        set_path = eeg_root / Path(relative_set)
        try:
            native = read_native_eeglab(set_path)
            indices, channel_meta = _required_eeg_indices(
                native,
                metadata_root,
                relative_set,
            )
            clock = _clock_for_events(native, file_events)
            native_by_file[relative_set] = native
            file_audits[relative_set] = {
                "status": "PASS",
                "set_path": str(set_path),
                "set_sha256": sha256_file(set_path),
                "fdt_path": str(native.fdt_path) if native.fdt_path else None,
                "fdt_sha256": sha256_file(native.fdt_path) if native.fdt_path else None,
                "sampling_frequency_hz": native.sfreq,
                "n_samples": native.n_samples,
                "duration_sec": native.n_samples / native.sfreq,
                "n_channels": native.n_channels,
                "channel_metadata": channel_meta,
                "clock": clock,
                "native_event_count": len(native.events),
                "boundary_count": len(native.boundary_times),
            }
        except (OSError, RuntimeError, ValueError) as exc:
            message = f"{relative_set}:{type(exc).__name__}:{exc}"
            all_file_errors.append(message)
            file_audits[relative_set] = {
                "status": SOURCE_SYNC_UNRESOLVED,
                "set_path": str(set_path),
                "reason": str(exc),
            }
    family_ledger: list[dict[str, Any]] = []
    for family_id in sorted(events_by_family, key=_family_sort_key):
        family_events = events_by_family[family_id]
        candidate_rows: list[dict[str, Any]] = []
        for event in family_events:
            relative_set = event["source_file"].replace("_events.tsv", "_eeg.set")
            native = native_by_file.get(relative_set)
            audit = file_audits.get(relative_set, {})
            event_clock = audit.get("clock", {}).get("by_event", {}).get(
                event["event_key"],
                {},
            )
            if native is None:
                candidate_rows.append(
                    {
                        "event_key": event["event_key"],
                        "clock_status": SOURCE_SYNC_UNRESOLVED,
                        "native_signal_status": POST_STIM_ARTIFACT_NOT_TESTABLE,
                        "candidate_by_horizon": {},
                    }
                )
                continue
            indices = [
                index
                for index, name in enumerate(native.channel_names)
                if not _is_auxiliary_channel(name)
            ]
            candidate_rows.append(
                _event_candidates(
                    event,
                    native,
                    indices,
                    event_clock.get("status") == "PASS",
                    nmd_grid_pass,
                )
            )
        family_ledger.append(
            _family_result(
                family_id,
                family_events,
                candidate_rows,
                nmd_grid_pass=nmd_grid_pass,
                nmd_binding=nmd_binding,
            )
        )
    passing = [
        family
        for family in family_ledger
        if family["eligibility_status"] == SIGNAL_TIMEBASE_PASS
        and family["min_subjects_across_rho"] >= MIN_BIOLOGICAL_UNITS
        and set(family["rho_levels"]) == {0.5, 1.0}
    ]
    method_limited = [
        family
        for family in family_ledger
        if family["eligibility_status"]
        in {RAW_SIGNAL_ONLY_PASS, NMD_TIMEBASE_METHOD_LIMITED}
    ]
    if passing:
        global_status = SIGNAL_TIMEBASE_PASS
        global_reason = "at_least_one_jointly_eligible_fixed_v_family"
    elif method_limited:
        global_status = "METHOD_LIMITED"
        global_reason = "native_signal_and_clock_pass_but_nmd_lattice_not_auditable"
    else:
        global_status = "NOT_TESTABLE"
        global_reason = "no_fixed_v_family_meets_joint_signal_timebase_criteria"
    return {
        "schema": "mndm.far_ext_002b_ds003670_signal_timebase.v1",
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "global_status": global_status,
        "global_reason": global_reason,
        "entry_far_ext_002a": {
            "path": str(far002_path),
            "sha256": sha256_file(far002_path),
            "protocol_id": far002_payload.get("protocol_id"),
            "global_status": far002_payload.get("global_status"),
        },
        "source_binding": source_binding,
        "payload_binding": payload_binding,
        "nmd_binding": nmd_binding,
        "promoted_event_count": len(ledger),
        "family_count": len(family_ledger),
        "family_ledger": family_ledger,
        "event_ledger": ledger,
        "file_audits": file_audits,
        "file_errors": all_file_errors,
        "primary_family_id": (
            sorted(
                passing,
                key=lambda family: (
                    -family["min_subjects_across_rho"],
                    -min(family["events_per_rho"].values(), default=0),
                    -(family["clean_post_sec"] or 0.0),
                    _family_sort_key(family["family_id"]),
                ),
            )[0]["family_id"]
            if passing
            else None
        ),
        "horizons_sec": list(HORIZONS_SEC),
        "pre_sec": PRE_SEC,
        "tolerability_confound": True,
        "audit_scope": {
            "metadata_opened": True,
            "signal_payloads_opened": True,
            "outcome_tables_opened": False,
            "nmd_outputs_opened": False,
            "mnps_calculated": False,
            "far_calculated": False,
            "home_away_constructed": False,
        },
        "far_003b_authorized": False,
        "claim_boundary": (
            "FAR-EXT-002B establishes only native signal/time-base eligibility "
            "under the frozen ds003670 semantic ledger. It establishes no "
            "MNPS trajectory, response magnitude, home/away state, resilience "
            "value, or causal amplitude effect."
        ),
    }
