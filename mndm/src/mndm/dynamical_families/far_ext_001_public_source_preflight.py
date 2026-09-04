"""Metadata-only FAR-EXT-001 audit for public OpenNeuro sources.

This module deliberately stops before signal processing.  It consumes only
BIDS metadata and ``*_events.tsv`` tables downloaded through the strict
metadata-only OpenNeuro wrapper.  The output is a source-semantics census, not
a FAR outcome and not an authorization to download neural payloads.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

PROTOCOL_ID = "FAR-EXT-001"
MIN_BIOLOGICAL_UNITS = 2
NMD_WINDOW_BY_MODALITY_SEC = {"eeg": 8.0, "ieeg": 8.0}

SOURCE_SPECS: dict[str, dict[str, Any]] = {
    "ds003670": {
        "modality": "eeg",
        "rho_terms": ("current", "intensity", "amplitude"),
        "v_terms": ("target", "site", "waveform", "frequency", "polarity"),
        "stim_terms": ("tes", "tdcs", "tacs", "stimulation", "stim"),
        "trigger_codes": True,
        "allow_timing_without_rho": True,
        "selection_on_outcome": "tolerability_selected_amplitude_reported_externally",
    },
    "ds006519": {
        "modality": "ieeg",
        "rho_terms": ("current", "intensity", "amplitude"),
        "v_terms": (
            "contact",
            "electrode",
            "site",
            "frequency",
            "pulse",
            "polarity",
        ),
        "stim_terms": ("stimulation", "stim", "electrical"),
        "prestim_poststim_grammar": True,
        "selection_on_outcome": "negative_motor_response_selected_source_reported_externally",
    },
    "ds005169": {
        "modality": "ieeg",
        "rho_terms": ("current", "intensity", "amplitude"),
        "v_terms": (
            "contact",
            "electrode",
            "site",
            "frequency",
            "pulse",
            "polarity",
        ),
        "stim_terms": ("stimulation", "stim", "electrical"),
        "prestim_poststim_grammar": True,
        "selection_on_outcome": "visual_effect_selected_source_reported_externally",
    },
    "ds008037": {
        "modality": "eeg",
        "rho_terms": ("current", "intensity", "amplitude", "energy"),
        "v_terms": ("target", "site", "coil", "protocol", "waveform"),
        "stim_terms": ("tms", "stimulation", "stim", "pulse"),
        "selection_on_outcome": "not_established_in_metadata_preflight",
    },
}

_EVENT_SUFFIX = "_events.tsv"
_EVENT_SIDECAR_SUFFIX = "_events.json"
_SIGNAL_SUFFIXES = (
    ".edf",
    ".edf.gz",
    ".eeg",
    ".fif",
    ".fif.gz",
    ".mat",
    ".nii",
    ".nii.gz",
    ".set",
    ".fdt",
    ".bdf",
    ".vhdr",
    ".vmrk",
    ".dat",
    ".npy",
    ".npz",
    ".nwb",
    ".h5",
    ".hdf5",
    ".bin",
    ".raw",
    ".mff",
    ".mef",
    ".nev",
    ".ncs",
    ".ns1",
    ".ns2",
    ".ns3",
    ".ns4",
    ".ns5",
    ".ns6",
    ".sif",
    ".trc",
    ".cnt",
    ".con",
    ".wav",
    ".mp4",
)
_METADATA_SUFFIXES = (
    ".json",
    ".tsv",
    ".txt",
    ".md",
)
_SUBJECT_RE = re.compile(r"(?:^|[/\\])sub-([^/\\]+)", re.IGNORECASE)
_SESSION_RE = re.compile(r"(?:^|[/\\])ses-([^/\\]+)", re.IGNORECASE)
_RUN_RE = re.compile(r"(?:^|[/\\])run-([^/\\_]+)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_download_config() -> dict[str, Any]:
    """Return the explicit, signal-free download contract for FAR-EXT-001."""
    return {
        "download": {
            "use_uvx": True,
            "retries": 2,
            "metadata_only_patterns": [
                "**/dataset_description.json",
                "**/*_events.tsv",
            ],
            "metadata_only_optional_patterns": [
                "**/*_events.json",
                "**/*_eeg.json",
                "**/*_ieeg.json",
                "**/*_channels.tsv",
                "**/*_electrodes.tsv",
            ],
        }
    }


def _nmd_window_binding() -> tuple[dict[str, float], dict[str, Any]]:
    """Resolve the current EEG/iEEG NMD window from the shared config."""
    repo_root = Path(__file__).resolve().parents[4]
    config_path = repo_root / "mndm" / "config" / "config_ingest_common_eeg.yaml"
    try:
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        window = float(config["mnps"]["window_sec"])
    except (ImportError, KeyError, OSError, TypeError, ValueError) as exc:
        raise RuntimeError(f"nmd_window_binding_failed:{config_path}") from exc
    return (
        {"eeg": window, "ieeg": window},
        {
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "field": "mnps.window_sec",
        },
    )


def _norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _clean(value: object) -> str | None:
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "none", "n/a", "na"} else None


def _number(value: object) -> tuple[float | None, str | None]:
    text = _clean(value)
    if text is None:
        return None, None
    if re.search(r"\d\s*[-–]\s*\d", text):
        return None, None
    match = _NUMBER_RE.search(text.replace(",", "."))
    if not match:
        return None, None
    try:
        number = float(match.group(0))
    except ValueError:
        return None, None
    unit_match = re.search(r"(µa|μa|ua|ma|microamp(?:ere)?s?|milliamp(?:ere)?s?)", text, re.I)
    unit = unit_match.group(1).lower() if unit_match else None
    if unit in {"µa", "μa", "ua", "microamp", "microamps", "microampere", "microamperes"}:
        unit = "ua"
    elif unit in {"ma", "milliamp", "milliamps", "milliampere", "milliamperes"}:
        unit = "ma"
    return number, unit


def _canonical_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    normalized = _norm(unit)
    if normalized in {"ma", "milliamp", "milliamps", "milliampere", "milliamperes"}:
        return "mA"
    if normalized in {"ua", "microamp", "microamps", "microampere", "microamperes"}:
        return "uA"
    return unit


def _subject_session_run(path: Path, root: Path) -> tuple[str | None, str | None, str | None]:
    relative = str(path.relative_to(root))
    subject_match = _SUBJECT_RE.search(relative)
    session_match = _SESSION_RE.search(relative)
    run_match = _RUN_RE.search(relative)
    subject = subject_match.group(1) if subject_match else None
    session = session_match.group(1) if session_match else None
    run = run_match.group(1) if run_match else None
    return subject, session, run


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_tsv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            rows = [dict(row) for row in reader]
            return rows, list(reader.fieldnames or [])
    except (OSError, csv.Error):
        return [], []


def _metadata_files(root: Path) -> tuple[list[Path], list[Path]]:
    metadata: list[Path] = []
    payloads: list[Path] = []
    if not root.exists():
        return metadata, payloads
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        name = path.name.lower()
        if any(name.endswith(suffix) for suffix in _SIGNAL_SUFFIXES):
            payloads.append(path)
        elif name.endswith(_METADATA_SUFFIXES):
            metadata.append(path)
    return metadata, payloads


def _field_name(columns: Iterable[str], terms: Iterable[str]) -> str | None:
    normalized = [(column, _norm(column)) for column in columns]
    for term in terms:
        term_norm = _norm(term)
        for column, column_norm in normalized:
            if column_norm == term_norm:
                return column
    for term in terms:
        term_norm = _norm(term)
        for column, column_norm in normalized:
            if term_norm and term_norm in column_norm:
                return column
    return None


def _sidecar_field(sidecar: Mapping[str, Any], field: str) -> tuple[str | None, str | None]:
    value = sidecar.get(field)
    if value is None:
        for key, candidate in sidecar.items():
            if _norm(key) == _norm(field):
                value = candidate
                break
    if isinstance(value, Mapping):
        units = value.get("Units", value.get("units"))
        description = value.get("Description", value.get("description"))
        return _clean(units), _clean(description)
    return None, _clean(value)


def _recording_durations(root: Path, record: Mapping[str, Any]) -> list[float]:
    durations: list[float] = []
    event_path = root / str(record["source_file"])
    for path in sorted(root.rglob("*_eeg.json")) + sorted(root.rglob("*_ieeg.json")):
        if path.parent != event_path.parent:
            continue
        path_subject, path_session, path_run = _subject_session_run(path, root)
        if record["subject"] and path_subject and record["subject"] != path_subject:
            continue
        if record["session"] and path_session and record["session"] != path_session:
            continue
        if record["run"] and path_run and record["run"] != path_run:
            continue
        metadata = _read_json(path)
        value, _ = _number(metadata.get("RecordingDuration"))
        if value is not None and math.isfinite(value) and value > 0:
            durations.append(value)
    return durations


def _recording_duration_index(root: Path) -> dict[tuple[str, str | None, str | None, str | None], list[float]]:
    durations: dict[tuple[str, str | None, str | None, str | None], list[float]] = defaultdict(list)
    for path in sorted(root.rglob("*_eeg.json")) + sorted(root.rglob("*_ieeg.json")):
        subject, session, run = _subject_session_run(path, root)
        metadata = _read_json(path)
        value, _ = _number(metadata.get("RecordingDuration"))
        if value is not None and math.isfinite(value) and value > 0:
            durations[(str(path.parent), subject, session, run)].append(value)
    return durations


def _stimulus_like(row: Mapping[str, str], spec: Mapping[str, Any], rho_field: str | None) -> bool:
    values = " ".join(
        str(value).lower()
        for key, value in row.items()
        if (
            "type" in _norm(key)
            or "stim" in _norm(key)
            or "task" in _norm(key)
            or "trigger" in _norm(key)
            or "code" in _norm(key)
            or _norm(key) == "value"
        )
    )
    tokens = set(re.findall(r"[a-z0-9]+", values))
    explicit_stimulus = any(_norm(term) in tokens for term in spec["stim_terms"])
    if spec.get("trigger_codes") and re.search(r"\b(?:16|32)\b", values):
        explicit_stimulus = True
    if not explicit_stimulus:
        return False
    if not rho_field:
        return bool(spec.get("allow_timing_without_rho"))
    value, unit = _number(row.get(rho_field))
    if value is None:
        return bool(spec.get("allow_timing_without_rho"))
    field_name = _norm(rho_field)
    if "amplitude" in field_name and unit not in {"ma", "ua"}:
        return False
    return True


def _event_records(
    root: Path,
    dataset_id: str,
    spec: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    columns_seen: set[str] = set()
    event_files: list[str] = []
    unit_evidence: dict[str, set[str]] = defaultdict(set)
    for path in sorted(root.rglob(f"*{_EVENT_SUFFIX}")):
        rows, columns = _read_tsv(path)
        event_files.append(str(path.relative_to(root)))
        columns_seen.update(columns)
        sidecar = _read_json(path.with_name(path.name[:-4] + ".json"))
        rho_field = _field_name(columns, spec["rho_terms"])
        onset_field = _field_name(columns, ("onset", "stim_onset", "start_time"))
        duration_field = _field_name(columns, ("duration", "stim_duration", "pulse_duration"))
        offset_field = _field_name(columns, ("offset", "stim_offset", "stop_time"))
        label_field = _field_name(columns, ("label", "trial_type", "event_type", "stim_type"))
        subject, session, run = _subject_session_run(path, root)
        for row in rows:
            label = _clean(row.get(label_field)) if label_field else None
            label_norm = _norm(label) if label else ""
            if spec.get("prestim_poststim_grammar"):
                include_row = bool(label and label_norm in {"prestim", "poststim"})
                if not include_row and dataset_id == "ds005169" and rho_field:
                    include_row = _number(row.get(rho_field))[0] is not None
                if not include_row:
                    continue
            elif not _stimulus_like(row, spec, rho_field):
                continue
            rho, inline_unit = _number(row.get(rho_field)) if rho_field else (None, None)
            sidecar_unit, sidecar_description = (
                _sidecar_field(sidecar, rho_field) if rho_field else (None, None)
            )
            unit = _canonical_unit(inline_unit or sidecar_unit)
            if rho_field and unit:
                unit_evidence[rho_field].add(unit)
            onset, _ = _number(row.get(onset_field)) if onset_field else (None, None)
            duration, _ = _number(row.get(duration_field)) if duration_field else (None, None)
            offset, _ = _number(row.get(offset_field)) if offset_field else (None, None)
            if offset is None and onset is not None and duration is not None:
                offset = onset + duration
            v: dict[str, str] = {}
            for column in columns:
                if any(term in _norm(column) for term in spec["v_terms"]):
                    value = _clean(row.get(column))
                    if value is not None:
                        v[_norm(column)] = value
            if (
                spec.get("prestim_poststim_grammar")
                and label
                and label_norm not in {"prestim", "poststim"}
            ):
                v["label"] = label
            records.append(
                {
                    "source_file": str(path.relative_to(root)),
                    "subject": subject,
                    "session": session,
                    "run": run,
                    "rho": rho,
                    "rho_field": rho_field,
                    "rho_unit": unit,
                    "rho_description": sidecar_description,
                    "v": v,
                    "onset_sec": onset,
                    "duration_sec": duration,
                    "offset_sec": offset,
                    "timing_grammar": "explicit_duration_or_offset",
                    "event_label": label,
                    "row": {str(key): _clean(value) for key, value in row.items()},
                }
            )
    return records, {
        "event_files": event_files,
        "event_file_count": len(event_files),
        "event_columns": sorted(columns_seen),
        "rho_unit_evidence": {key: sorted(values) for key, values in unit_evidence.items()},
    }


def _trigger_code(record: Mapping[str, Any]) -> str | None:
    for key, value in record["row"].items():
        normalized = _norm(key)
        if normalized in {"trigger", "trigger_code", "event_code", "code", "value"}:
            cleaned = _clean(value)
            if cleaned is not None:
                return cleaned.lower()
    return None


def _apply_event_grammar(
    dataset_id: str,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply source-specific event semantics before temporal isolation."""
    if dataset_id == "ds006519":
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[
                (
                    record["source_file"],
                    record["subject"],
                    record["session"],
                    record["run"],
                )
            ].append(record)
        paired_records: list[dict[str, Any]] = []
        for group in grouped.values():
            ordered = sorted(
                group,
                key=lambda item: item["onset_sec"]
                if item["onset_sec"] is not None
                else float("inf"),
            )
            for index, prestim in enumerate(ordered):
                if str(prestim.get("event_label") or "").lower() != "prestim":
                    continue
                poststim = next(
                    (
                        candidate
                        for candidate in ordered[index + 1 :]
                        if str(candidate.get("event_label") or "").lower() == "poststim"
                        and candidate["onset_sec"] is not None
                    ),
                    None,
                )
                if poststim is None:
                    continue
                paired = dict(poststim)
                paired["onset_sec"] = (
                    prestim["onset_sec"] + prestim["duration_sec"]
                    if prestim["onset_sec"] is not None
                    and prestim["duration_sec"] is not None
                    else None
                )
                paired["offset_sec"] = poststim["onset_sec"]
                paired["duration_sec"] = (
                    paired["offset_sec"] - paired["onset_sec"]
                    if paired["onset_sec"] is not None
                    else None
                )
                paired["event_label"] = "prestim_to_poststim"
                paired["timing_grammar"] = "prestim_end_to_poststim_onset"
                paired_records.append(paired)
        return paired_records

    if dataset_id == "ds005169":
        grouped = defaultdict(list)
        for record in records:
            grouped[
                (
                    record["source_file"],
                    record["subject"],
                    record["session"],
                    record["run"],
                )
            ].append(record)
        paired_records = []
        for group in grouped.values():
            ordered = sorted(
                group,
                key=lambda item: item["onset_sec"]
                if item["onset_sec"] is not None
                else float("inf"),
            )
            poststim_rows = [
                record
                for record in ordered
                if str(record.get("event_label") or "").lower() == "poststim"
                and record["onset_sec"] is not None
            ]
            for record in ordered:
                label = str(record.get("event_label") or "").lower()
                if label in {"prestim", "poststim"} or record["rho"] is None:
                    continue
                poststim = next(
                    (
                        candidate
                        for candidate in poststim_rows
                        if candidate["onset_sec"] > (record["onset_sec"] or -math.inf)
                    ),
                    None,
                )
                paired = dict(record)
                if poststim is None:
                    paired["timing_grammar"] = "unresolved_missing_poststim_boundary"
                    paired["offset_sec"] = None
                else:
                    paired["offset_sec"] = poststim["onset_sec"]
                    paired["duration_sec"] = (
                        paired["offset_sec"] - paired["onset_sec"]
                        if paired["onset_sec"] is not None
                        else None
                    )
                    paired["timing_grammar"] = "stim_to_poststim_onset"
                paired_records.append(paired)
        return paired_records

    if dataset_id != "ds003670":
        return records

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                record["source_file"],
                record["subject"],
                record["session"],
                record["run"],
            )
        ].append(record)

    normalized: list[dict[str, Any]] = []
    for group in grouped.values():
        has_trigger_grammar = any(_trigger_code(record) in {"16", "32", "start", "stop"} for record in group)
        if not has_trigger_grammar:
            for record in group:
                record = dict(record)
                record["timing_grammar"] = "unresolved_missing_start_stop_trigger_grammar"
                record["offset_sec"] = None
                normalized.append(record)
            continue
        ordered = sorted(
            group,
            key=lambda item: item["onset_sec"] if item["onset_sec"] is not None else float("inf"),
        )
        index = 0
        while index < len(ordered):
            start = ordered[index]
            if _trigger_code(start) not in {"16", "start"}:
                index += 1
                continue
            stop_index = next(
                (
                    candidate_index
                    for candidate_index in range(index + 1, len(ordered))
                    if _trigger_code(ordered[candidate_index]) in {"32", "stop"}
                ),
                None,
            )
            if stop_index is None or start["onset_sec"] is None or ordered[stop_index]["onset_sec"] is None:
                unresolved = dict(start)
                unresolved["timing_grammar"] = "unresolved_unpaired_start_stop_trigger"
                unresolved["offset_sec"] = None
                normalized.append(unresolved)
                index += 1
                continue
            stop = ordered[stop_index]
            paired = dict(start)
            paired["rho"] = start["rho"] if start["rho"] is not None else stop["rho"]
            if not paired["v"] and stop["v"]:
                paired["v"] = dict(stop["v"])
            paired["offset_sec"] = float(stop["onset_sec"]) + 5.0
            paired["duration_sec"] = paired["offset_sec"] - float(paired["onset_sec"])
            paired["timing_grammar"] = "trigger_16_32_plus_locked_5s_ramp_down"
            paired["paired_stop_onset_sec"] = stop["onset_sec"]
            normalized.append(paired)
            index = stop_index + 1
    return normalized


def _add_temporal_bounds(records: list[dict[str, Any]], root: Path) -> None:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    duration_index = _recording_duration_index(root)
    for record in records:
        key = (
            record["source_file"],
            record["subject"],
            record["session"],
            record["run"],
        )
        grouped[key].append(record)
    for group in grouped.values():
        group.sort(key=lambda item: item["onset_sec"] if item["onset_sec"] is not None else float("inf"))
        for index, record in enumerate(group):
            onset = record["onset_sec"]
            offset = record["offset_sec"]
            record["stim_end_to_next_stim_sec"] = None
            record["remaining_recording_sec"] = None
            record["overlap_with_next_stim"] = False
            record["isolated_post_horizon_sec"] = None
            if onset is not None and offset is not None:
                if index + 1 < len(group):
                    next_onset = group[index + 1]["onset_sec"]
                    if next_onset is not None and next_onset < offset:
                        record["overlap_with_next_stim"] = True
                    elif next_onset is not None:
                        record["stim_end_to_next_stim_sec"] = next_onset - offset
                        record["isolated_post_horizon_sec"] = next_onset - offset
                durations = duration_index.get(
                    (
                        str((root / str(record["source_file"])).parent),
                        record["subject"],
                        record["session"],
                        record["run"],
                    ),
                    [],
                )
                if durations:
                    remaining = min(durations) - offset
                    if remaining >= 0:
                        record["remaining_recording_sec"] = remaining


def _family_key(record: Mapping[str, Any]) -> str:
    values = record.get("v") or {}
    if not values:
        return "v_unresolved"
    return json.dumps(values, sort_keys=True, ensure_ascii=False)


def _family_census(
    records: list[dict[str, Any]],
    spec: Mapping[str, Any],
    nmd_window_sec: float,
) -> list[dict[str, Any]]:
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        families[_family_key(record)].append(record)
    result: list[dict[str, Any]] = []
    for family_key, family_records in sorted(families.items()):
        bounded_records = [
            record
            for record in family_records
            if record.get("stim_end_to_next_stim_sec") is not None
            and not record.get("overlap_with_next_stim")
        ]
        rho_records = [record for record in family_records if record["rho"] is not None]
        rho_levels = sorted({round(float(record["rho"]), 12) for record in rho_records})
        subjects_by_rho: dict[str, list[str]] = {}
        sessions_by_rho: dict[str, list[str]] = {}
        events_by_rho: dict[str, int] = {}
        for rho in rho_levels:
            selected = [record for record in rho_records if math.isclose(record["rho"], rho)]
            subjects_by_rho[str(rho)] = sorted(
                {record["subject"] for record in selected if record["subject"] is not None}
            )
            sessions_by_rho[str(rho)] = sorted(
                {
                    f"{record['subject']}:{record['session'] or 'n/a'}"
                    for record in selected
                }
            )
            events_by_rho[str(rho)] = len(selected)
        horizons = [
            record["isolated_post_horizon_sec"]
            for record in bounded_records
            if record.get("isolated_post_horizon_sec") is not None
        ]
        isolated = min(horizons) if horizons else None
        min_subjects = (
            min((len(values) for values in subjects_by_rho.values()), default=0)
            if rho_levels
            else 0
        )
        rho_units = {record["rho_unit"] for record in rho_records}
        units_known = bool(rho_records) and None not in rho_units and len(rho_units) == 1
        timing_known = all(
            record["onset_sec"] is not None
            and record["offset_sec"] is not None
            and record["timing_grammar"] != "unresolved_missing_start_stop_trigger_grammar"
            and record["timing_grammar"] != "unresolved_unpaired_start_stop_trigger"
            for record in bounded_records
        )
        v_resolved = family_key != "v_unresolved"
        if not v_resolved:
            classification = "SOURCE_UNCERTAIN"
            reason = "fixed_perturbation_identity_v_is_not_explicit"
        elif not rho_records or not units_known:
            classification = "SOURCE_UNCERTAIN"
            reason = "explicit_numeric_rho_or_units_not_established_in_downloaded_metadata"
        elif not timing_known or isolated is None:
            classification = "TIMING_UNRESOLVED"
            reason = "stimulus_timing_or_post_event_bound_not_established"
        elif isolated <= nmd_window_sec:
            classification = "NMD_TIMEBASE_INCOMPATIBLE"
            reason = "clean_isolated_post_horizon_does_not_exceed_modality_matched_nmd_window"
        elif len(rho_levels) >= 2 and min_subjects >= MIN_BIOLOGICAL_UNITS:
            classification = "CURVE_CANDIDATE"
            reason = "fixed_v_family_has_multiple_rho_levels_and_biological_support"
        elif len(rho_levels) >= 1:
            classification = "POINT_OR_LIMITED_CANDIDATE"
            reason = "explicit_rho_is_present_but_curve_support_is_limited"
        else:
            classification = "SOURCE_UNCERTAIN"
            reason = "no_usable_rho_levels"
        result.append(
            {
                "v_key": family_key,
                "rho_levels": rho_levels,
                "subjects_per_rho": subjects_by_rho,
                "sessions_per_rho": sessions_by_rho,
                "events_per_rho": events_by_rho,
                "min_subjects_across_rho": min_subjects,
                "rho_units_known": units_known,
                "rho_units": sorted(rho_units),
                "v_resolved": v_resolved,
                "timing_known": timing_known,
                "events_audited": len(family_records),
                "events_with_next_stim_bound": len(bounded_records),
                "overlap_event_count": sum(
                    1 for record in family_records if record.get("overlap_with_next_stim")
                ),
                "clean_isolated_post_horizon_sec": isolated,
                "nmd_window_sec": nmd_window_sec,
                "isolated_to_nmd_ratio": (
                    isolated / nmd_window_sec if isolated is not None else None
                ),
                "classification": classification,
                "classification_reason": reason,
                "zero_or_sham_present": any(
                    token in {"zero", "sham"}
                    for record in family_records
                    for value in record["row"].values()
                    if value is not None
                    for token in re.findall(r"[a-z0-9]+", value.lower())
                )
                or any(math.isclose(record["rho"], 0.0) for record in rho_records),
                "selection_on_outcome_status": spec["selection_on_outcome"],
            }
        )
    return result


def _event_tuple(record: Mapping[str, Any]) -> dict[str, Any]:
    state_fields: dict[str, str] = {}
    for key, value in record["row"].items():
        if not any(term in _norm(key) for term in ("state", "condition", "awake", "anesthesia")):
            continue
        cleaned = _clean(value)
        if cleaned is not None:
            state_fields[_norm(key)] = cleaned
    return {
        "w": {
            "session": record["session"],
            "run": record["run"],
            "state_fields": state_fields,
        },
        "v": record["v"],
        "rho": record["rho"],
        "rho_unit": record["rho_unit"],
        "t_on": record["onset_sec"],
        "t_off": record["offset_sec"],
        "stim_end_to_next_stim_sec": record.get("stim_end_to_next_stim_sec"),
        "remaining_recording_sec": record.get("remaining_recording_sec"),
        "T_isolated": record.get("isolated_post_horizon_sec"),
        "overlap_with_next_stim": record.get("overlap_with_next_stim", False),
        "timing_grammar": record["timing_grammar"],
        "subject": record["subject"],
        "source_file": record["source_file"],
    }


def _source_result(
    dataset_id: str,
    root: Path,
    nmd_windows: Mapping[str, float],
) -> dict[str, Any]:
    spec = SOURCE_SPECS[dataset_id]
    metadata_files, payloads = _metadata_files(root)
    records, event_inventory = _event_records(root, dataset_id, spec)
    records = _apply_event_grammar(dataset_id, records)
    _add_temporal_bounds(records, root)
    nmd_window_sec = nmd_windows[spec["modality"]]
    families = _family_census(records, spec, nmd_window_sec)
    classifications = [family["classification"] for family in families]
    if payloads:
        classification = "SOURCE_UNCERTAIN"
        reason = "signal_payload_detected_in_metadata_only_root"
    elif not families:
        classification = "SOURCE_UNCERTAIN"
        reason = "no_stimulation_event_rows_found_in_downloaded_metadata"
    elif any(value == "CURVE_CANDIDATE" for value in classifications):
        classification = "CURVE_CANDIDATE"
        reason = "at_least_one_fixed_v_family_cleared_metadata_semantics"
    elif any(value == "POINT_OR_LIMITED_CANDIDATE" for value in classifications):
        classification = "POINT_OR_LIMITED_CANDIDATE"
        reason = "explicit_rho_present_but_curve_support_is_limited"
    elif any(value == "NMD_TIMEBASE_INCOMPATIBLE" for value in classifications):
        classification = "NMD_TIMEBASE_INCOMPATIBLE"
        reason = "physical_and_timed_family_failed_current_nmd_window"
    elif any(value == "TIMING_UNRESOLVED" for value in classifications):
        classification = "TIMING_UNRESOLVED"
        reason = "physical_amplitude is present but isolated timing is unresolved"
    else:
        classification = "SOURCE_UNCERTAIN"
        reason = "downloaded metadata did not establish a promotable source family"
    return {
        "dataset_id": dataset_id,
        "source_family": "openneuro",
        "source_status": "LOCAL_METADATA_ONLY",
        "source_root": str(root),
        "modality": spec["modality"],
        "metadata_inventory": {
            "metadata_file_count": len(metadata_files),
            "metadata_bytes_scanned": sum(path.stat().st_size for path in metadata_files),
            "metadata_file_records": [
                {
                    "relative_path": str(path.relative_to(root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in metadata_files
            ],
            "signal_payload_count": len(payloads),
        },
        "event_inventory": event_inventory,
        "records_audited": len(records),
        "event_tuples": [_event_tuple(record) for record in records],
        "families": families,
        "classification": classification,
        "classification_reason": reason,
        "audit_scope": {
            "metadata_only": True,
            "signal_payloads_opened": False,
            "signal_payloads_present": bool(payloads),
            "outcome_tables_opened": False,
            "far_calculated": False,
        },
    }


def run_inventory(source_roots: Mapping[str, str | Path]) -> dict[str, Any]:
    """Audit the four FAR-EXT-001 sources from local metadata-only roots."""
    nmd_windows, nmd_binding = _nmd_window_binding()
    datasets: list[dict[str, Any]] = []
    for dataset_id in SOURCE_SPECS:
        root_value = source_roots.get(dataset_id)
        root = Path(root_value) if root_value is not None else None
        if root is None or not root.exists():
            datasets.append(
                {
                    "dataset_id": dataset_id,
                    "source_status": "LOCAL_UNAVAILABLE",
                    "classification": "SOURCE_UNCERTAIN",
                    "classification_reason": "metadata-only download root is unavailable",
                    "audit_scope": {
                        "metadata_only": True,
                        "signal_payloads_opened": False,
                        "outcome_tables_opened": False,
                        "far_calculated": False,
                    },
                }
            )
        else:
            datasets.append(_source_result(dataset_id, root, nmd_windows))
    counts: dict[str, int] = defaultdict(int)
    for dataset in datasets:
        counts[dataset["classification"]] += 1
    compatible = [
        dataset
        for dataset in datasets
        if dataset["classification"] in {"CURVE_CANDIDATE", "POINT_OR_LIMITED_CANDIDATE"}
    ]
    complete = all(
        dataset.get("source_status") == "LOCAL_METADATA_ONLY"
        and dataset.get("event_inventory", {}).get("event_file_count", 0) > 0
        and not dataset.get("audit_scope", {}).get("signal_payloads_present", False)
        for dataset in datasets
    )
    return {
        "schema": "mndm.far_ext_001_public_source_preflight.v1",
        "protocol_id": PROTOCOL_ID,
        "source_scope": list(SOURCE_SPECS),
        "nmd_window_sec": nmd_windows,
        "nmd_window_binding": nmd_binding,
        "datasets": datasets,
        "classification_counts": dict(sorted(counts.items())),
        "gate_status": "CENSUS_COMPLETE" if complete else "NOT_TESTABLE",
        "gate_reason": (
            "all four public sources were audited from metadata/events-only roots"
            if complete
            else "the four-source metadata/events census is incomplete or violated its payload guard"
        ),
        "promotion_candidates": [dataset["dataset_id"] for dataset in compatible],
        "fail_closed_assertions": {
            "signal_payloads_opened": False,
            "outcome_tables_opened": False,
            "far_calculated": False,
            "inferred_rho": False,
            "condition_promoted_to_rho": False,
        },
        "claim_boundary": (
            "FAR-EXT-001 is a metadata/events-only public-source preflight. "
            "It does not open neural payloads, calculate FAR outcomes, or authorize "
            "a downstream perturbation measurement."
        ),
    }


def download_public_metadata(out_dir: str | Path) -> dict[str, Path]:
    """Download the explicit metadata-only subset through openneuro_ingest."""
    repo_root = Path(__file__).resolve().parents[4]
    openneuro_src = repo_root / "openneuro_ingest" / "src"
    if not openneuro_src.exists():
        raise RuntimeError(f"openneuro_ingest_source_missing:{openneuro_src}")
    source_text = str(openneuro_src)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    from openneuro.download import download_metadata_only

    return download_metadata_only(
        SOURCE_SPECS.keys(),
        metadata_download_config(),
        Path(out_dir),
    )
