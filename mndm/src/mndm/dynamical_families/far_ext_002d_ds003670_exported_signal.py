"""Prospective FAR-EXT-002D exported-signal/time-base eligibility audit.

This module audits only the exact Experiment-1 event population inherited from
FAR-EXT-002A.  It applies the qualified NMD-QC-FLOAT contract to complete
pre/post interval slices of the already exported EEGLAB float surface.  It
does not run preprocessing, feature extraction, MNPS, or FAR.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import numpy as np

from .far_ext_002b_ds003670_signal_timebase import (
    NativeEEG,
    read_native_eeglab,
)
from .nmd_qc_float import (
    CONTRACT_VERSION as NMD_QC_CONTRACT_VERSION,
    RULE_MANIFEST_SHA256,
    TECHNICALLY_ADMISSIBLE,
    TECHNICAL_INVALID,
    TECHNICAL_STATUS_UNRESOLVED,
    audit_exported_float,
)


PROTOCOL_ID = "FAR-EXT-002D"
DATASET_ID = "ds003670"
NMD_WINDOW_SEC = 8.0
NMD_OVERLAP = 0.5
NMD_STEP_SEC = 4.0
PRE_SEC = 30.0
HORIZONS_SEC = (60.0, 30.0, 20.0, 16.0)
RHO_LEVELS = (0.5, 1.0)
PROMOTED_EVENT_COUNT = 323
EXPECTED_EEG_CHANNEL_COUNT = 32

SIGNAL_TIMEBASE_PASS = "SIGNAL_TIMEBASE_PASS"
CLOCK_UNRESOLVED = "CLOCK_UNRESOLVED"
NMD_TIMEBASE_METHOD_LIMITED = "NMD_TIMEBASE_METHOD_LIMITED"
INSUFFICIENT_BIOLOGICAL_SUPPORT = "INSUFFICIENT_BIOLOGICAL_SUPPORT"
SOURCE_BINDING_FAILED = "SOURCE_BINDING_FAILED"
NOT_TESTABLE = "NOT_TESTABLE"

_TARGET_ORDER = {"frontal": 0, "motor": 1, "parietal": 2}
_FREQUENCY_ORDER = {0: 0, 5: 1, 30: 2}
_EEG_CHANNEL_NAMES = (
    "Fp1",
    "Fpz",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "FC5",
    "FC1",
    "FC2",
    "FC6",
    "M1",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "M2",
    "CP5",
    "CP1",
    "CP2",
    "CP6",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "POz",
    "O1",
    "Oz",
    "O2",
)
_AUXILIARY_NAMES = {"ecg", "eog", "resp"}
_BOOKKEEPING_NAMES = {
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


def _json_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _normalise_relative(value: object) -> str:
    return str(value).replace("\\", "/").lstrip("./")


def _finite(value: object) -> float | None:
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


def _family_id(v: Mapping[str, Any]) -> str:
    target = str(v.get("target", "")).strip().lower()
    waveform_raw = str(v.get("waveform", "")).strip().lower()
    if waveform_raw in {"dc", "direct_current", "direct-current"}:
        waveform = "dc"
    elif "sin" in waveform_raw:
        waveform = "sinusoidal"
    else:
        waveform = re.sub(r"[^a-z0-9]+", "_", waveform_raw).strip("_")
    frequency = _finite(v.get("frequency_hz"))
    frequency_text = (
        "unknown"
        if frequency is None
        else str(int(frequency))
        if frequency.is_integer()
        else format(frequency, "g")
    )
    return f"{target}_{waveform}_{frequency_text}hz"


def _family_sort_key(family_id: str) -> tuple[int, int, str]:
    target = family_id.split("_", 1)[0]
    match = re.search(r"_(\d+(?:\.\d+)?)hz$", family_id)
    frequency = float(match.group(1)) if match else math.inf
    return (
        _TARGET_ORDER.get(target, 99),
        _FREQUENCY_ORDER.get(int(frequency) if frequency.is_integer() else -1, 99),
        family_id,
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


def canonical_002a_promoted_events(
    payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    """Return the 002A-only promoted rows and their frozen canonical hash."""
    if payload.get("protocol_id") != "FAR-EXT-002A":
        raise ValueError("far_ext_002a_protocol_id_mismatch")
    if payload.get("dataset_id") != DATASET_ID:
        raise ValueError("far_ext_002a_dataset_id_mismatch")
    if payload.get("global_status") != "DS003670_LIMITED_CURVE_SEMANTICS_PASS":
        raise ValueError("far_ext_002a_global_status_not_pass")

    promoted: list[dict[str, Any]] = []
    for source_event in payload.get("events", []):
        if not isinstance(source_event, Mapping):
            continue
        isolated = _finite(source_event.get("T_isolated"))
        if not (
            source_event.get("semantic_status") == "PASS"
            and bool(source_event.get("rho_contrast_eligible"))
            and int(source_event.get("experiment", 0) or 0) == 1
            and isolated is not None
            and isolated > NMD_WINDOW_SEC
        ):
            continue
        event = dict(source_event)
        event["source_file"] = _normalise_relative(event.get("source_file", ""))
        promoted.append(event)

    ledger_hash = _json_hash(promoted)
    return promoted, ledger_hash


def promoted_event_ledger(
    payload: Mapping[str, Any],
    *,
    expected_hash: str | None = None,
) -> list[dict[str, Any]]:
    """Bind the exact 002A rows and add only reporting-derived fields."""
    promoted, ledger_hash = canonical_002a_promoted_events(payload)
    if len(promoted) != PROMOTED_EVENT_COUNT:
        raise ValueError(f"far_ext_002a_promoted_event_count_mismatch:{len(promoted)}")
    if expected_hash is not None and ledger_hash != expected_hash:
        raise ValueError("far_ext_002a_promoted_ledger_hash_mismatch")

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_event in promoted:
        event = dict(source_event)
        v = event.get("v")
        if not isinstance(v, Mapping):
            raise ValueError("far_ext_002a_promoted_event_missing_v")
        event["family_id"] = _family_id(v)
        event["event_key"] = _event_key(event)
        if event["event_key"] in seen:
            raise ValueError("far_ext_002a_duplicate_promoted_event")
        seen.add(event["event_key"])
        result.append(event)
    return result


def eeg_payload_allowlist(ledger: Iterable[Mapping[str, Any]]) -> list[str]:
    values = {
        _normalise_relative(event.get("source_file", "")).replace(
            "_events.tsv", "_eeg.set"
        )
        for event in ledger
    }
    if any(not value.endswith("_eeg.set") for value in values):
        raise ValueError("far_ext_002a_source_file_not_events_tsv")
    return sorted(values)


def load_scope(scope_path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(scope_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("far_ext_002d_scope_not_object")
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("far_ext_002d_scope_protocol_mismatch")
    if payload.get("execution_status") != "PREREGISTRATION_FROZEN":
        raise ValueError("far_ext_002d_scope_not_frozen")
    return payload, sha256_file(scope_path)


def _load_json_object(path: Path, reason: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(reason) from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{reason}:not_object")
    return value


def _config_binding(repo_root: Path, scope: Mapping[str, Any]) -> dict[str, Any]:
    config_info = scope["inherited_config"]
    path = repo_root / Path(str(config_info["path"]))
    result: dict[str, Any] = {
        "path": str(path),
        "expected_sha256": config_info["sha256"],
        "sha256": None,
        "status": NMD_TIMEBASE_METHOD_LIMITED,
        "reason": None,
        "values": {},
        "consumed_surface": [
            "mnps.window_sec",
            "mnps.overlap",
            "epoching.length_s",
            "epoching.step_s",
        ],
    }
    if not path.is_file():
        result["reason"] = "nmd_config_missing"
        return result
    result["sha256"] = sha256_file(path)
    if result["sha256"] != config_info["sha256"]:
        result["reason"] = "nmd_config_sha256_mismatch"
        return result
    try:
        import yaml

        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - config boundary
        result["reason"] = f"nmd_config_unreadable:{type(exc).__name__}"
        return result
    mnps = config.get("mnps", {}) if isinstance(config, Mapping) else {}
    epoching = config.get("epoching", {}) if isinstance(config, Mapping) else {}
    values = {
        "mnps.window_sec": _finite(mnps.get("window_sec")),
        "mnps.overlap": _finite(mnps.get("overlap")),
        "epoching.length_s": _finite(epoching.get("length_s")),
        "epoching.step_s": _finite(epoching.get("step_s")),
    }
    result["values"] = values
    expected = config_info["required_values"]
    valid = all(
        values[key] is not None
        and math.isclose(float(values[key]), float(expected[key]), rel_tol=0.0, abs_tol=1e-12)
        for key in expected
    )
    result["status"] = "PASS" if valid else NMD_TIMEBASE_METHOD_LIMITED
    result["reason"] = "frozen_four_key_match" if valid else "frozen_four_key_mismatch"
    return result


def _inventory_map(inventory: Mapping[str, Any]) -> dict[str, str]:
    files = inventory.get("files")
    if not isinstance(files, list):
        raise RuntimeError("payload_inventory_files_missing")
    result: dict[str, str] = {}
    for item in files:
        if not isinstance(item, Mapping):
            raise RuntimeError("payload_inventory_file_row_invalid")
        relative = _normalise_relative(item.get("path", ""))
        digest = str(item.get("sha256", ""))
        if not relative or len(digest) != 64 or relative in result:
            raise RuntimeError("payload_inventory_file_row_invalid")
        result[relative] = digest
    return result


def verify_payload_inventory(
    *,
    eeg_root: Path,
    scope: Mapping[str, Any],
    repo_root: Path,
    ledger: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    payload_info = scope["payload"]
    inventory_path = repo_root / Path(str(payload_info["inventory_path"]))
    if not inventory_path.is_file():
        return {
            "status": SOURCE_BINDING_FAILED,
            "reason": "payload_inventory_missing",
            "inventory_path": str(inventory_path),
        }
    inventory_hash = sha256_file(inventory_path)
    if inventory_hash != payload_info["inventory_sha256"]:
        return {
            "status": SOURCE_BINDING_FAILED,
            "reason": "payload_inventory_sha256_mismatch",
            "inventory_path": str(inventory_path),
            "sha256": inventory_hash,
        }
    try:
        inventory = _load_json_object(inventory_path, "payload_inventory_unreadable")
        expected = _inventory_map(inventory)
    except RuntimeError as exc:
        return {
            "status": SOURCE_BINDING_FAILED,
            "reason": str(exc),
            "inventory_path": str(inventory_path),
        }
    expected_sets = set(eeg_payload_allowlist(ledger))
    expected_from_manifest = {
        path for path in expected if path.endswith(".set")
    }
    if expected_sets != expected_from_manifest:
        return {
            "status": SOURCE_BINDING_FAILED,
            "reason": "payload_inventory_set_allowlist_mismatch",
            "expected_sets": sorted(expected_sets),
            "inventory_sets": sorted(expected_from_manifest),
        }
    expected_file_count = payload_info.get(
        "expected_payload_file_count",
        len(expected),
    )
    if len(expected) != int(expected_file_count):
        return {
            "status": SOURCE_BINDING_FAILED,
            "reason": "payload_inventory_file_count_mismatch",
            "expected_file_count": int(expected_file_count),
            "inventory_file_count": len(expected),
        }
    expected_root = Path(str(payload_info["existing_payload_root"]))
    if expected_root.resolve() != eeg_root.resolve():
        return {
            "status": SOURCE_BINDING_FAILED,
            "reason": "payload_root_mismatch",
            "expected_root": str(expected_root),
            "actual_root": str(eeg_root),
        }
    if not eeg_root.is_dir():
        return {
            "status": SOURCE_BINDING_FAILED,
            "reason": "payload_root_missing",
            "expected_root": str(expected_root),
        }
    actual: dict[str, str] = {}
    unexpected: list[str] = []
    for path in sorted(item for item in eeg_root.rglob("*") if item.is_file()):
        relative = _normalise_relative(path.relative_to(eeg_root))
        if relative not in expected:
            if path.name not in _BOOKKEEPING_NAMES:
                unexpected.append(relative)
            continue
        actual[relative] = sha256_file(path)
    missing = sorted(set(expected) - set(actual))
    mismatched = sorted(
        relative
        for relative in set(expected).intersection(actual)
        if expected[relative] != actual[relative]
    )
    status = not missing and not mismatched and not unexpected and actual == expected
    return {
        "status": "PASS" if status else SOURCE_BINDING_FAILED,
        "reason": "exact_payload_inventory" if status else "payload_inventory_mismatch",
        "inventory_path": str(inventory_path),
        "inventory_sha256": inventory_hash,
        "expected_file_count": len(expected),
        "actual_file_count": len(actual),
        "missing": missing,
        "mismatched": mismatched,
        "unexpected": unexpected,
        "file_hashes": dict(sorted(actual.items())),
    }


def _select_channels(
    native: NativeEEG,
    *,
    metadata_root: Path,
    relative_set: str,
) -> tuple[list[int], dict[str, Any]]:
    sidecar_path = metadata_root / Path(
        relative_set.replace("_eeg.set", "_eeg.json")
    )
    sidecar = _load_json_object(sidecar_path, "eeg_sidecar_unreadable")
    if int(float(sidecar.get("EEGChannelCount", -1))) != EXPECTED_EEG_CHANNEL_COUNT:
        raise RuntimeError("eeg_channel_count_not_32")
    names = list(native.channel_names)
    selected = [
        index
        for index, name in enumerate(names)
        if str(name).strip().lower() not in _AUXILIARY_NAMES
        and not str(name).strip().lower().startswith("bip")
        and not str(name).strip().lower().startswith("resp")
    ]
    selected_names = [names[index] for index in selected]
    if selected_names != list(_EEG_CHANNEL_NAMES):
        raise RuntimeError("native_eeg_channel_order_mismatch")
    if len(selected) != EXPECTED_EEG_CHANNEL_COUNT:
        raise RuntimeError("native_eeg_channel_selection_not_32")
    if set(selected).intersection(native.bad_channel_indices):
        raise RuntimeError("native_bad_eeg_channel_annotation")
    return selected, {
        "sidecar_path": str(sidecar_path),
        "declared_eeg_channel_count": EXPECTED_EEG_CHANNEL_COUNT,
        "selected_channel_indices": selected,
        "selected_channel_names": selected_names,
        "native_name_spelling_is_authoritative": True,
        "bids_channels_tsv_spelling_used": False,
        "required_channel_indices_for_nmd_qc": list(range(32)),
        "required_channel_names_for_nmd_qc": None,
        "provenance_non_empty": True,
    }


def _clock_matches(
    native: NativeEEG,
    events: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    available = [
        event
        for event in native.events
        if event.get("code") in {"16", "32"}
    ]
    used: set[int] = set()
    matches: list[dict[str, Any]] = []
    failures: list[str] = []
    by_event: dict[str, dict[str, Any]] = {}
    ordered = sorted(events, key=lambda item: float(item.get("trigger_start", math.inf)))
    for event in ordered:
        key = str(event["event_key"])
        event_matches: list[dict[str, Any]] = []
        event_failures: list[str] = []
        for code, field in (("16", "trigger_start"), ("32", "trigger_max_end")):
            expected = _finite(event.get(field))
            if expected is None:
                reason = f"{field}_missing"
                event_failures.append(reason)
                failures.append(f"{key}:{reason}")
                continue
            candidates = [
                (index, candidate)
                for index, candidate in enumerate(available)
                if index not in used and candidate.get("code") == code
            ]
            if not candidates:
                reason = f"{code}_missing"
                event_failures.append(reason)
                failures.append(f"{key}:{reason}")
                continue
            index, candidate = min(
                candidates,
                key=lambda pair: abs(float(pair[1]["time_sec"]) - expected),
            )
            difference = abs(float(candidate["time_sec"]) - expected)
            if difference > 1.0 / native.sfreq:
                reason = f"{code}_difference_{difference:g}"
                event_failures.append(reason)
                failures.append(f"{key}:{reason}")
                continue
            used.add(index)
            match = {
                "event_key": key,
                "code": code,
                "mapped_field": field,
                "expected_sec": expected,
                "native_sec": candidate["time_sec"],
                "difference_sec": difference,
                "sample_index_0": candidate["sample_index_0"],
            }
            event_matches.append(match)
            matches.append(match)
        by_event[key] = {
            "status": "PASS" if not event_failures else CLOCK_UNRESOLVED,
            "matches": event_matches,
            "failures": event_failures,
        }
    return {
        "status": "PASS" if not failures else CLOCK_UNRESOLVED,
        "promoted_event_count": len(ordered),
        "matched_trigger_count": len(matches),
        "expected_trigger_count": len(ordered) * 2,
        "matches": matches,
        "failures": failures,
        "by_event": by_event,
        "matching_scope": "promoted events only",
        "leftover_native_16_32_ignored": True,
        "latency_conversion": "round(latency_eeglab) - 1; t = sample_index_0 / sfreq",
        "tolerance_sec": 1.0 / native.sfreq,
        "t0_binding": "002A ramp_down_end; no third native trigger",
    }


def grid_starts(
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


def nmd_window_support(
    *,
    trigger_start: float,
    t0: float,
    horizon: float,
) -> dict[str, Any]:
    pre = grid_starts(trigger_start - PRE_SEC, trigger_start)
    post = grid_starts(t0, t0 + horizon)
    return {
        "pre_starts": pre,
        "post_starts": post,
        "pre_count": len(pre),
        "post_count": len(post),
        "pre_required": len(pre) >= 1,
        "post_required": len(post) >= 2,
    }


def _sample_bounds(
    start_sec: float,
    end_sec: float,
    sfreq: float,
    n_samples: int,
) -> tuple[int, int] | None:
    start = int(math.ceil(start_sec * sfreq - 1.0e-9))
    end = int(math.ceil(end_sec * sfreq - 1.0e-9))
    if start < 0 or end > n_samples or end <= start:
        return None
    return start, end


def _compact_qc(qc: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the sample-free QC certificate without duplicating raw inputs."""
    fields = (
        "protocol_id",
        "contract_id",
        "contract_version",
        "rule_manifest_path",
        "rule_manifest_sha256",
        "input_shape",
        "input_dtype",
        "time_dtype",
        "sampling_frequency",
        "input_provenance",
        "implementation_identity",
        "runtime_identity",
        "fixture_manifest_sha256",
        "input_hash",
        "time_hash",
        "channel_flags",
        "segment_flags",
        "invalid_reasons",
        "unresolved_reasons",
        "timebase_status",
        "effective_sampling_frequency",
        "required_channel_indices",
        "required_channel_names",
        "excluded_channel_indices",
        "excluded_channel_names",
        "channel_selection_provenance",
        "segment_boundaries",
        "evaluated_support",
        "uncovered_support",
        "recording_status",
        "determinism_hash",
    )
    return {field: qc[field] for field in fields if field in qc}


def audit_interval(
    native: NativeEEG,
    *,
    channel_indices: list[int],
    channel_names: list[str],
    start_sec: float,
    end_sec: float,
    input_label: str,
) -> dict[str, Any]:
    """Slice one complete absolute interval, then run NMD-QC-FLOAT."""
    bounds = _sample_bounds(
        start_sec,
        end_sec,
        native.sfreq,
        native.n_samples,
    )
    base = {
        "start_sec": start_sec,
        "end_sec": end_sec,
        "input_label": input_label,
        "status": TECHNICAL_STATUS_UNRESOLVED,
        "reason": None,
        "sample_start": None,
        "sample_end": None,
        "qc": None,
    }
    if bounds is None:
        base["reason"] = "recording_boundary"
        return base
    start, end = bounds
    base["sample_start"] = start
    base["sample_end"] = end
    if any(start_sec <= boundary <= end_sec for boundary in native.boundary_times):
        base["reason"] = "native_recording_boundary_annotation"
        return base
    if any(
        interval_start <= end_sec and interval_end >= start_sec
        for interval_start, interval_end in native.bad_intervals
    ):
        base["reason"] = "native_bad_segment_annotation"
        return base
    try:
        values = np.asarray(native.data[channel_indices, start:end])
    except (OSError, RuntimeError, ValueError) as exc:
        base["reason"] = f"signal_read_error:{type(exc).__name__}"
        return base
    time = np.arange(start, end, dtype=np.float64) / float(native.sfreq)
    qc = audit_exported_float(
        values,
        time,
        required_channel_indices=list(range(EXPECTED_EEG_CHANNEL_COUNT)),
        channel_labels=channel_names,
        channel_selection_provenance={
            "source": "FAR-EXT-002D frozen native EEGLAB channel order",
            "input_label": input_label,
            "selected_channel_indices": channel_indices,
            "selected_channel_names": channel_names,
        },
        sampling_frequency=float(native.sfreq),
        segment_boundaries=[(0, int(values.shape[1]))],
        input_provenance={
            "record_id": native.set_path.stem,
            "source": "EEGLAB exported float payload",
            "interval": input_label,
        },
        implementation_identity="mndm.dynamical_families.far_ext_002d",
        runtime_identity={
            "nmd_qc_float_contract_version": NMD_QC_CONTRACT_VERSION,
        },
    )
    base["status"] = qc["recording_status"]
    base["reason"] = (
        "nmd_qc_float_pass"
        if base["status"] == TECHNICALLY_ADMISSIBLE
        else "nmd_qc_float_invalid"
        if base["status"] == TECHNICAL_INVALID
        else "nmd_qc_float_unresolved"
    )
    base["sample_count"] = int(values.shape[1])
    base["dtype"] = str(values.dtype)
    base["time_start_sec"] = float(time[0])
    base["time_end_sec"] = float(time[-1])
    base["qc"] = _compact_qc(qc)
    return base


def _event_candidate_rows(
    event: Mapping[str, Any],
    native: NativeEEG,
    *,
    channel_indices: list[int],
    channel_names: list[str],
    clock_status: str,
    nmd_config_pass: bool,
) -> dict[str, Any]:
    trigger_start = _finite(event.get("trigger_start"))
    t0 = _finite(event.get("ramp_down_end"))
    next_stim = _finite(event.get("next_stim_start"))
    result: dict[str, Any] = {
        "event_key": event["event_key"],
        "clock_status": clock_status,
        "candidate_by_horizon": {},
    }
    if trigger_start is None or t0 is None:
        for horizon in HORIZONS_SEC:
            result["candidate_by_horizon"][str(horizon)] = {
                "eligible": False,
                "reason": "event_timing_missing",
                "horizon_sec": horizon,
            }
        result["native_signal_status"] = TECHNICAL_STATUS_UNRESOLVED
        return result

    pre_qc = audit_interval(
        native,
        channel_indices=channel_indices,
        channel_names=channel_names,
        start_sec=trigger_start - PRE_SEC,
        end_sec=trigger_start,
        input_label="pre",
    )
    result["pre_qc"] = pre_qc
    for horizon in HORIZONS_SEC:
        post_end = t0 + horizon
        row: dict[str, Any] = {
            "eligible": False,
            "technical_eligible": False,
            "horizon_sec": horizon,
            "pre_qc_status": pre_qc["status"],
            "post_qc": None,
            "nmd": nmd_window_support(
                trigger_start=trigger_start,
                t0=t0,
                horizon=horizon,
            ),
        }
        post_qc = audit_interval(
            native,
            channel_indices=channel_indices,
            channel_names=channel_names,
            start_sec=t0,
            end_sec=post_end,
            input_label=f"post_{horizon:g}s",
        )
        row["post_qc"] = post_qc
        next_stim_within_horizon = (
            next_stim is not None and post_end > next_stim + 1.0e-9
        )
        row["technical_eligible"] = bool(
            clock_status == "PASS"
            and pre_qc["status"] == TECHNICALLY_ADMISSIBLE
            and post_qc["status"] == TECHNICALLY_ADMISSIBLE
            and not next_stim_within_horizon
        )
        row["eligible"] = bool(
            row["technical_eligible"]
            and nmd_config_pass
            and row["nmd"]["pre_required"]
            and row["nmd"]["post_required"]
        )
        row["reason"] = (
            "next_stimulation_within_horizon"
            if next_stim_within_horizon
            else "eligible"
            if row["eligible"]
            else "nmd_timebase_unresolved"
            if row["technical_eligible"] and not nmd_config_pass
            else "insufficient_complete_nmd_windows"
            if row["technical_eligible"]
            else "clock_unresolved"
            if clock_status != "PASS"
            else pre_qc["reason"]
            if pre_qc["status"] != TECHNICALLY_ADMISSIBLE
            else post_qc["reason"]
        )
        result["candidate_by_horizon"][str(horizon)] = row
    result["native_signal_status"] = (
        TECHNICALLY_ADMISSIBLE
        if any(
            row.get("technical_eligible")
            for row in result["candidate_by_horizon"].values()
        )
        else pre_qc["status"]
    )
    return result


def _status_rollup(statuses: Iterable[str]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    for status in statuses:
        counts[str(status)] += 1
    if counts.get(TECHNICAL_INVALID, 0):
        status = TECHNICAL_INVALID
    elif counts.get(TECHNICAL_STATUS_UNRESOLVED, 0):
        status = TECHNICAL_STATUS_UNRESOLVED
    elif counts.get(TECHNICALLY_ADMISSIBLE, 0):
        status = TECHNICALLY_ADMISSIBLE
    else:
        status = TECHNICAL_STATUS_UNRESOLVED
    return {
        "status": status,
        "counts": dict(sorted(counts.items())),
    }


def _family_result(
    family_id: str,
    events: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    nmd_config_pass: bool,
    nmd_binding: Mapping[str, Any],
) -> dict[str, Any]:
    support_by_horizon: dict[str, dict[str, Any]] = {}
    selected_horizon: float | None = None
    selected_rows: list[
        tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
    ] = []
    for horizon in HORIZONS_SEC:
        eligible_pairs = [
            (event, row["candidate_by_horizon"].get(str(horizon), {}), row)
            for event, row in zip(events, candidate_rows)
            if row["candidate_by_horizon"].get(str(horizon), {}).get("eligible")
        ]
        subjects = {
            str(rho): sorted(
                {
                    str(event.get("biological_unit"))
                    for event, _row, _candidate in eligible_pairs
                    if math.isclose(float(event.get("rho", math.nan)), rho)
                }
            )
            for rho in RHO_LEVELS
        }
        event_counts = {
            str(rho): sum(
                math.isclose(float(event.get("rho", math.nan)), rho)
                for event, _row, _candidate in eligible_pairs
            )
            for rho in RHO_LEVELS
        }
        support_by_horizon[str(horizon)] = {
            "subjects_per_rho": subjects,
            "events_per_rho": event_counts,
            "min_subjects_across_rho": min(len(value) for value in subjects.values()),
            "eligible_event_count": len(eligible_pairs),
        }
        if (
            selected_horizon is None
            and all(len(subjects[str(rho)]) >= 2 for rho in RHO_LEVELS)
        ):
            selected_horizon = horizon
            selected_rows = eligible_pairs

    subjects_per_rho = {
        str(rho): sorted(
            {
                str(event.get("biological_unit"))
                for event, _row, _candidate in selected_rows
                if math.isclose(float(event.get("rho", math.nan)), rho)
            }
        )
        for rho in RHO_LEVELS
    }
    events_per_rho = {
        str(rho): sum(
            math.isclose(float(event.get("rho", math.nan)), rho)
            for event, _row, _candidate in selected_rows
        )
        for rho in RHO_LEVELS
    }
    has_clock = any(row.get("clock_status") == "PASS" for row in candidate_rows)
    has_technical_support = any(
        row.get("candidate_by_horizon", {})
        and any(
            horizon_row.get("technical_eligible")
            for horizon_row in row["candidate_by_horizon"].values()
        )
        for row in candidate_rows
    )
    has_lattice_support = any(
        (
            horizon_row.get("nmd", {}).get("pre_required")
            or horizon_row.get("nmd", {}).get("pre_count", 0) >= 1
        )
        and (
            horizon_row.get("nmd", {}).get("post_required")
            or horizon_row.get("nmd", {}).get("post_count", 0) >= 2
        )
        for candidate in candidate_rows
        for horizon_row in candidate.get("candidate_by_horizon", {}).values()
    )
    if selected_horizon is not None:
        eligibility_status = SIGNAL_TIMEBASE_PASS
        reason = "common_horizon_with_two_subjects_per_rho"
    elif not nmd_config_pass and has_technical_support:
        eligibility_status = NMD_TIMEBASE_METHOD_LIMITED
        reason = "frozen_nmd_configuration_not_auditable"
    elif has_technical_support and not has_lattice_support:
        eligibility_status = NOT_TESTABLE
        reason = "nmd_lattice_window_support_missing"
    elif has_technical_support:
        eligibility_status = INSUFFICIENT_BIOLOGICAL_SUPPORT
        reason = "no_common_horizon_with_two_subjects_per_rho"
    elif not has_clock:
        eligibility_status = CLOCK_UNRESOLVED
        reason = "promoted_event_clock_not_proven"
    else:
        eligibility_status = NOT_TESTABLE
        reason = "no_technically_admissible_pre_post_intervals"

    pre_statuses = [
        str(candidate["pre_qc"]["status"])
        for candidate in candidate_rows
        if isinstance(candidate.get("pre_qc"), Mapping)
    ]
    post_statuses_by_horizon: dict[str, list[str]] = {
        str(horizon): []
        for horizon in HORIZONS_SEC
    }
    for candidate in candidate_rows:
        for horizon, row in candidate.get("candidate_by_horizon", {}).items():
            post_qc = row.get("post_qc")
            if isinstance(post_qc, Mapping):
                post_statuses_by_horizon[str(horizon)].append(
                    str(post_qc.get("status"))
                )
    pre_status = _status_rollup(pre_statuses)
    post_status = {
        horizon: _status_rollup(statuses)
        for horizon, statuses in post_statuses_by_horizon.items()
    }
    if selected_horizon is not None:
        selected_statuses = [
            str(candidate["pre_qc"]["status"])
            for _event, _row, candidate in selected_rows
        ]
        selected_statuses.extend(
            str(row["post_qc"]["status"])
            for _event, row, _candidate in selected_rows
            if isinstance(row.get("post_qc"), Mapping)
        )
        exported_status = _status_rollup(selected_statuses)["status"]
    else:
        all_statuses = list(pre_statuses)
        all_statuses.extend(
            status
            for statuses in post_statuses_by_horizon.values()
            for status in statuses
        )
        exported_status = _status_rollup(all_statuses)["status"]

    return {
        "family_id": family_id,
        "v": events[0].get("v") if events else {},
        "target": events[0].get("target"),
        "waveform": events[0].get("waveform"),
        "frequency": events[0].get("frequency_hz"),
        "rho_levels": sorted(
            {
                float(event["rho"])
                for event in events
                if _finite(event.get("rho")) is not None
            }
        ),
        "subjects_per_rho": subjects_per_rho,
        "events_per_rho": events_per_rho,
        "min_subjects_across_rho": min(
            (len(value) for value in subjects_per_rho.values()),
            default=0,
        ),
        "event_count": len(events),
        "promoted_event_keys": [event["event_key"] for event in events],
        "candidate_support_by_horizon": support_by_horizon,
        "selected_post_horizon": selected_horizon,
        "clean_pre_sec": PRE_SEC if selected_horizon is not None else None,
        "clean_post_sec": selected_horizon,
        "clock_status": "PASS" if has_clock else CLOCK_UNRESOLVED,
        "exported_signal_technical_status": exported_status,
        "pre_qc_status": pre_status,
        "post_qc_status_by_horizon": post_status,
        "nmd_timebase_status": (
            "PASS"
            if nmd_config_pass and selected_horizon is not None
            else NMD_TIMEBASE_METHOD_LIMITED
        ),
        "during_stim_status": "DURING_STIM_ARTIFACT_EXPECTED",
        "post_stimulation_biological_interpretability": "NOT_ESTABLISHED",
        "eligibility_status": eligibility_status,
        "reason": reason,
        "tolerability_confound": True,
        "nmd_binding": dict(nmd_binding),
        "event_eligibility": candidate_rows,
    }


def _source_binding(
    metadata_root: Path,
    far002_payload: Mapping[str, Any],
) -> dict[str, Any]:
    expected_root = Path(str(far002_payload.get("source_root", "")))
    binding = far002_payload.get("source_binding", {})
    passed = (
        binding.get("status") == "PASS"
        and expected_root.resolve() == metadata_root.resolve()
        and not binding.get("signal_payloads")
        and not binding.get("unexpected_files")
    )
    return {
        "status": "PASS" if passed else SOURCE_BINDING_FAILED,
        "expected_root": str(expected_root),
        "actual_root": str(metadata_root),
        "far_ext_002a_source_binding_status": binding.get("status"),
    }


def _global_status(
    family_ledger: Iterable[Mapping[str, Any]],
) -> tuple[str, str]:
    families = list(family_ledger)
    passing = [
        family
        for family in families
        if family.get("eligibility_status") == SIGNAL_TIMEBASE_PASS
    ]
    method_limited = [
        family
        for family in families
        if family.get("eligibility_status") == NMD_TIMEBASE_METHOD_LIMITED
    ]
    has_clock = any(family.get("clock_status") == "PASS" for family in families)
    has_technical_support = any(
        family.get("eligibility_status")
        in {
            SIGNAL_TIMEBASE_PASS,
            INSUFFICIENT_BIOLOGICAL_SUPPORT,
            NMD_TIMEBASE_METHOD_LIMITED,
        }
        for family in families
    )
    if passing:
        return SIGNAL_TIMEBASE_PASS, "at_least_one_common_horizon_family"
    if method_limited:
        return (
            NMD_TIMEBASE_METHOD_LIMITED,
            "signal_and_clock_support_but_nmd_lattice_unresolved",
        )
    if not has_clock:
        return CLOCK_UNRESOLVED, "no_family_has_proven_promoted_event_clock"
    if has_technical_support:
        return (
            INSUFFICIENT_BIOLOGICAL_SUPPORT,
            "technical_support_without_two_units_per_rho",
        )
    return NOT_TESTABLE, "no_family_has_admissible_pre_post_intervals"


def _select_primary_family(
    families: Iterable[Mapping[str, Any]],
) -> str | None:
    passing = [
        family
        for family in families
        if family.get("eligibility_status") == SIGNAL_TIMEBASE_PASS
    ]
    if not passing:
        return None
    return sorted(
        passing,
        key=lambda family: (
            -int(family.get("min_subjects_across_rho", 0)),
            -min(
                (
                    int(value)
                    for value in family.get("events_per_rho", {}).values()
                ),
                default=0,
            ),
            -float(family.get("selected_post_horizon") or 0.0),
            _family_sort_key(str(family["family_id"])),
        ),
    )[0]["family_id"]


def _early_result(
    *,
    scope_path: Path,
    scope_hash: str | None,
    far002_path: Path,
    far002_payload: Mapping[str, Any] | None,
    global_reason: str,
    source_binding: Mapping[str, Any] | None = None,
    payload_binding: Mapping[str, Any] | None = None,
    nmd_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "mndm.far_ext_002d_ds003670_exported_signal.v1",
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "global_status": SOURCE_BINDING_FAILED,
        "global_reason": global_reason,
        "scope_path": str(scope_path),
        "scope_sha256": scope_hash,
        "entry_far_ext_002a": {
            "path": str(far002_path),
            "sha256": sha256_file(far002_path) if far002_path.is_file() else None,
            "protocol_id": far002_payload.get("protocol_id")
            if far002_payload
            else None,
            "global_status": far002_payload.get("global_status")
            if far002_payload
            else None,
        },
        "source_binding": dict(source_binding or {}),
        "payload_binding": dict(payload_binding or {}),
        "nmd_binding": dict(nmd_binding or {}),
        "exported_signal_technical_status": {
            "selected_population_status": TECHNICAL_STATUS_UNRESOLVED,
            "counts_by_interval_status": {},
            "event_horizon_status_table": [],
        },
        "clock_status": {"status": CLOCK_UNRESOLVED},
        "nmd_timebase_status": NMD_TIMEBASE_METHOD_LIMITED,
        "post_stimulation_biological_interpretability": "NOT_ESTABLISHED",
        "subjects_and_events_retained_at_each_rho": {},
        "family_ledger": [],
        "event_ledger": [],
        "audit_scope": {
            "metadata_opened": True,
            "signal_payloads_opened": False,
            "outcome_tables_opened": False,
            "nmd_outputs_opened": False,
            "mnps_calculated": False,
            "far_calculated": False,
            "home_away_constructed": False,
        },
        "tolerability_confound": True,
        "far_003b_authorized": False,
    }


def audit_far_ext_002d(
    *,
    metadata_root: Path,
    eeg_root: Path,
    far002_path: Path,
    scope_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Run the frozen 002D audit against an explicitly bound payload root."""
    try:
        scope, scope_hash = load_scope(scope_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return _early_result(
            scope_path=scope_path,
            scope_hash=None,
            far002_path=far002_path,
            far002_payload=None,
            global_reason=f"scope_binding_failed:{type(exc).__name__}:{exc}",
        )
    if (
        scope.get("nmd_qc_float", {}).get("contract_version")
        != NMD_QC_CONTRACT_VERSION
        or scope.get("nmd_qc_float", {}).get("rule_manifest_sha256")
        != RULE_MANIFEST_SHA256
    ):
        return _early_result(
            scope_path=scope_path,
            scope_hash=scope_hash,
            far002_path=far002_path,
            far002_payload=None,
            global_reason="nmd_qc_float_contract_binding_failed",
        )
    try:
        far002_payload = _load_json_object(
            far002_path,
            "far_ext_002a_certificate_unreadable",
        )
        source_binding = _source_binding(metadata_root, far002_payload)
        ledger = promoted_event_ledger(
            far002_payload,
            expected_hash=scope["inherited_002a"]["promoted_event_ledger_sha256"],
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return _early_result(
            scope_path=scope_path,
            scope_hash=scope_hash,
            far002_path=far002_path,
            far002_payload=locals().get("far002_payload"),
            global_reason=f"far_ext_002a_binding_failed:{type(exc).__name__}:{exc}",
            source_binding=locals().get("source_binding"),
        )
    if (
        far002_payload.get("global_status")
        != scope["inherited_002a"]["certificate_global_status"]
        or sha256_file(far002_path)
        != scope["inherited_002a"]["certificate_sha256"]
        or source_binding["status"] != "PASS"
    ):
        return _early_result(
            scope_path=scope_path,
            scope_hash=scope_hash,
            far002_path=far002_path,
            far002_payload=far002_payload,
            global_reason="far_ext_002a_certificate_or_source_binding_failed",
            source_binding=source_binding,
        )
    nmd_binding = _config_binding(repo_root, scope)
    payload_binding = verify_payload_inventory(
        eeg_root=eeg_root,
        scope=scope,
        repo_root=repo_root,
        ledger=ledger,
    )
    if payload_binding["status"] != "PASS":
        return _early_result(
            scope_path=scope_path,
            scope_hash=scope_hash,
            far002_path=far002_path,
            far002_payload=far002_payload,
            global_reason="payload_binding_failed",
            source_binding=source_binding,
            payload_binding=payload_binding,
            nmd_binding=nmd_binding,
        ) | {"event_ledger": ledger}

    events_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    events_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in ledger:
        relative_set = _normalise_relative(event["source_file"]).replace(
            "_events.tsv", "_eeg.set"
        )
        events_by_file[relative_set].append(event)
        events_by_family[event["family_id"]].append(event)

    native_by_file: dict[str, NativeEEG] = {}
    file_audits: dict[str, dict[str, Any]] = {}
    for relative_set, file_events in sorted(events_by_file.items()):
        set_path = eeg_root / Path(relative_set)
        try:
            native = read_native_eeglab(set_path)
            channel_indices, channel_meta = _select_channels(
                native,
                metadata_root=metadata_root,
                relative_set=relative_set,
            )
            clock = _clock_matches(native, file_events)
            native_by_file[relative_set] = native
            file_audits[relative_set] = {
                "status": "PASS",
                "set_path": str(set_path),
                "set_sha256": sha256_file(set_path),
                "fdt_path": str(native.fdt_path) if native.fdt_path else None,
                "sampling_frequency_hz": native.sfreq,
                "n_samples": native.n_samples,
                "duration_sec": native.n_samples / native.sfreq,
                "n_channels": native.n_channels,
                "data_dtype": str(native.data_dtype),
                "channel_selection": channel_meta,
                "clock": clock,
                "native_event_count": len(native.events),
                "boundary_count": len(native.boundary_times),
            }
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            file_audits[relative_set] = {
                "status": "SOURCE_SYNC_UNRESOLVED",
                "set_path": str(set_path),
                "reason": f"{type(exc).__name__}:{exc}",
            }

    family_ledger: list[dict[str, Any]] = []
    interval_status_counts: dict[str, int] = defaultdict(int)
    event_horizon_table: list[dict[str, Any]] = []
    for family_id in sorted(events_by_family, key=_family_sort_key):
        family_events = events_by_family[family_id]
        candidate_rows: list[dict[str, Any]] = []
        for event in family_events:
            relative_set = _normalise_relative(event["source_file"]).replace(
                "_events.tsv", "_eeg.set"
            )
            native = native_by_file.get(relative_set)
            file_audit = file_audits.get(relative_set, {})
            clock = file_audit.get("clock", {})
            event_clock = clock.get("by_event", {}).get(event["event_key"], {})
            if native is None:
                candidate = {
                    "event_key": event["event_key"],
                    "clock_status": SOURCE_BINDING_FAILED,
                    "native_signal_status": TECHNICAL_STATUS_UNRESOLVED,
                    "candidate_by_horizon": {
                        str(horizon): {
                            "eligible": False,
                            "reason": "native_eeg_file_unavailable",
                            "horizon_sec": horizon,
                        }
                        for horizon in HORIZONS_SEC
                    },
                }
            else:
                channel_meta = file_audit["channel_selection"]
                candidate = _event_candidate_rows(
                    event,
                    native,
                    channel_indices=list(channel_meta["selected_channel_indices"]),
                    channel_names=list(channel_meta["selected_channel_names"]),
                    clock_status=(
                        "PASS"
                        if event_clock.get("status") == "PASS"
                        else CLOCK_UNRESOLVED
                    ),
                    nmd_config_pass=nmd_binding.get("status") == "PASS",
                )
            candidate_rows.append(candidate)
            pre_qc = candidate.get("pre_qc")
            if isinstance(pre_qc, Mapping) and pre_qc.get("status"):
                interval_status_counts[str(pre_qc["status"])] += 1
            for horizon, row in candidate.get("candidate_by_horizon", {}).items():
                post_qc = row.get("post_qc")
                if isinstance(post_qc, Mapping) and post_qc.get("status"):
                    interval_status_counts[str(post_qc["status"])] += 1
                event_horizon_table.append(
                    {
                        "family_id": family_id,
                        "event_key": event["event_key"],
                        "rho": event.get("rho"),
                        "horizon_sec": float(horizon),
                        "eligible": bool(row.get("eligible")),
                        "technical_eligible": bool(row.get("technical_eligible")),
                        "reason": row.get("reason"),
                        "pre_status": pre_qc.get("status")
                        if isinstance(pre_qc, Mapping)
                        else None,
                        "post_status": post_qc.get("status")
                        if isinstance(post_qc, Mapping)
                        else None,
                    }
                )
        family_ledger.append(
            _family_result(
                family_id,
                family_events,
                candidate_rows,
                nmd_config_pass=nmd_binding.get("status") == "PASS",
                nmd_binding=nmd_binding,
            )
        )

    global_status, global_reason = _global_status(family_ledger)
    primary = _select_primary_family(family_ledger)
    has_clock = any(family["clock_status"] == "PASS" for family in family_ledger)
    if primary is not None:
        selected_status = TECHNICALLY_ADMISSIBLE
    elif interval_status_counts.get(TECHNICAL_INVALID):
        selected_status = TECHNICAL_INVALID
    else:
        selected_status = TECHNICAL_STATUS_UNRESOLVED
    clock_counts = {
        "families_with_proven_clock": sum(
            family["clock_status"] == "PASS" for family in family_ledger
        ),
        "families_total": len(family_ledger),
    }
    return {
        "schema": "mndm.far_ext_002d_ds003670_exported_signal.v1",
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "global_status": global_status,
        "global_reason": global_reason,
        "entry_far_ext_002a": {
            "path": str(far002_path),
            "sha256": sha256_file(far002_path),
            "protocol_id": far002_payload["protocol_id"],
            "global_status": far002_payload["global_status"],
            "promoted_event_count": len(ledger),
        },
        "promoted_event_ledger_sha256": scope["inherited_002a"][
            "promoted_event_ledger_sha256"
        ],
        "scope_path": str(scope_path),
        "scope_sha256": scope_hash,
        "source_binding": source_binding,
        "payload_binding": payload_binding,
        "nmd_binding": nmd_binding,
        "nmd_qc_float": {
            "contract_version": NMD_QC_CONTRACT_VERSION,
            "rule_manifest_sha256": RULE_MANIFEST_SHA256,
            "input_grain": "slice_then_audit_complete_pre_or_post_interval",
            "required_status": TECHNICALLY_ADMISSIBLE,
        },
        "exported_signal_technical_status": {
            "selected_population_status": selected_status,
            "counts_by_interval_status": dict(sorted(interval_status_counts.items())),
            "event_horizon_status_table": event_horizon_table,
        },
        "clock_status": {
            "status": "PASS" if has_clock else CLOCK_UNRESOLVED,
            **clock_counts,
            "matching_scope": "promoted events only",
        },
        "nmd_timebase_status": (
            "PASS"
            if nmd_binding.get("status") == "PASS"
            and any(
                family["nmd_timebase_status"] == "PASS"
                for family in family_ledger
            )
            else NMD_TIMEBASE_METHOD_LIMITED
        ),
        "post_stimulation_biological_interpretability": "NOT_ESTABLISHED",
        "during_stimulation_status": "DURING_STIM_ARTIFACT_EXPECTED",
        "subjects_and_events_retained_at_each_rho": {
            family["family_id"]: {
                "subjects_per_rho": family["subjects_per_rho"],
                "events_per_rho": family["events_per_rho"],
                "selected_horizon": family["selected_post_horizon"],
            }
            for family in family_ledger
        },
        "promoted_event_count": len(ledger),
        "family_count": len(family_ledger),
        "family_ledger": family_ledger,
        "event_ledger": ledger,
        "file_audits": file_audits,
        "file_count": len(file_audits),
        "primary_family_id": primary,
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
            "FAR-EXT-002D establishes only exported-signal technical, clock, "
            "and NMD-timebase eligibility under the frozen 002A ledger and "
            "NMD-QC-FLOAT contract. It establishes no artifact-free biological "
            "signal, MNPS trajectory, response magnitude, home/away state, "
            "resilience value, or causal amplitude effect."
        ),
    }


__all__ = [
    "CLOCK_UNRESOLVED",
    "DATASET_ID",
    "EXPECTED_EEG_CHANNEL_COUNT",
    "HORIZONS_SEC",
    "INSUFFICIENT_BIOLOGICAL_SUPPORT",
    "NMD_TIMEBASE_METHOD_LIMITED",
    "NOT_TESTABLE",
    "PRE_SEC",
    "PROTOCOL_ID",
    "SIGNAL_TIMEBASE_PASS",
    "SOURCE_BINDING_FAILED",
    "TECHNICALLY_ADMISSIBLE",
    "TECHNICAL_INVALID",
    "TECHNICAL_STATUS_UNRESOLVED",
    "_EEG_CHANNEL_NAMES",
    "_clock_matches",
    "_family_result",
    "_sample_bounds",
    "audit_far_ext_002d",
    "audit_interval",
    "canonical_002a_promoted_events",
    "eeg_payload_allowlist",
    "grid_starts",
    "load_scope",
    "nmd_window_support",
    "promoted_event_ledger",
    "sha256_file",
    "verify_payload_inventory",
]
