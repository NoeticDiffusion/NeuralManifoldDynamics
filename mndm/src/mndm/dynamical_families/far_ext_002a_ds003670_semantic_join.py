"""FAR-EXT-002A source-semantic join for OpenNeuro ds003670.

The join deliberately combines only:

* the byte-bound BIDS metadata/events root from FAR-EXT-001;
* the published Experiment 1/2 session tables;
* the published trigger/block protocol.

It does not open EEG, ECG, EOG, behavioral payloads, NMD outputs, or outcome
tables.  A successful result is a semantic qualification only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .far_ext_001_public_source_preflight import _metadata_files

PROTOCOL_ID = "FAR-EXT-002A"
DATASET_ID = "ds003670"
NMD_WINDOW_SEC = 8.0
RHO_LEVELS_MA = (0.5, 1.0)
MIN_BIOLOGICAL_UNITS = 2
RAMP_DOWN_SEC = 5.0

SOURCE_DOCUMENTS = [
    {
        "source_id": "publication",
        "kind": "published_data_descriptor",
        "url": "https://doi.org/10.1038/s41597-021-01046-y",
        "claims": ["Experiment 1/2 session tables", "rho levels", "block design", "trigger grammar"],
    },
    {
        "source_id": "dataset_readme",
        "kind": "source_readme",
        "url": "https://github.com/OpenNeuroDatasets/ds003670",
        "claims": ["BIDS dataset identity", "experiment/session context"],
    },
    {
        "source_id": "supporting_repository",
        "kind": "protocol_and_analysis_repository",
        "url": "https://github.com/ngebodh/GX_tES_EEG_Physio_Behavior",
        "revision": "358804f5b8b47f4753a51fc9990f60d8f37a5b27",
        "claims": ["GX source protocol and session naming"],
    },
    {
        "source_id": "zenodo",
        "kind": "supporting_data_record",
        "url": "https://zenodo.org/api/records/4456079",
        "claims": ["GX supporting-data identity and BIDS linkage"],
    },
]

# Table 1: Experiment 1.  Each row is file number, block stim types, and
# source-reported amplitude in mA.  The table is transcribed from the cited
# publication and is kept as data so the join is deterministic and auditable.
_EXP1_ROWS = {
    "0101": (("M30", "F30", "F0"), (1.0, 1.0, 1.0), (3,)),
    "0102": (("M30", "F30", "F0"), (0.5, 0.5, 0.5), (3,)),
    "0103": (("P30", "P0", "P5"), (0.5, 0.5, 0.5), ()),
    "0104": (("F5", "M5", "M0"), (0.5, 0.5, 0.5), ()),
    "0105": (("F5", "M5", "M0"), (1.0, 1.0, 1.0), ()),
    "0106": (("P30", "P0", "P5"), (1.0, 1.0, 1.0), ()),
    "0201": (("F30", "F5", "P30"), (0.5, 0.5, 0.5), ()),
    "0202": (("P0", "M5", "M30"), (0.5, 0.5, 0.5), ()),
    "0301": (("M0", "P0", "F30"), (0.5, 0.5, 0.5), ()),
    "0302": (("F5", "P5", "P30"), (0.5, 0.5, 0.5), ()),
    "0303": (("M5", "F0", "M30"), (0.5, 0.5, 0.5), ()),
    "0401": (("P30", "M30", "F5"), (0.5, 0.5, 0.5), ()),
    "0402": (("P0", "F0", "M0"), (1.0, 1.0, 1.0), ()),
    "0403": (("F30", "M5", "P5"), (1.0, 1.0, 1.0), ()),
    "0501": (("M5", "P0", "P5"), (0.5, 1.0, 1.0), ()),
    "0504": (("P30", "M30", "M0"), (1.0, 1.0, 1.0), ()),
    "0505": (("F30", "F0", "F5"), (1.0, 1.0, 0.5), ()),
    "0601": (("F0", "F5", "P5"), (1.0, 0.5, 1.0), ()),
    "0602": (("F30", "M30", "P30"), (0.5, 1.0, 1.0), ()),
    "0603": (("P0", "M5", "M0"), (1.0, 1.0, 1.0), ()),
    "0701": (("P5", "F5", "M5"), (0.5, 0.5, 0.5), ()),
    "0702": (("M0", "P0", "F0"), (0.5, 0.5, 0.5), ()),
    "0703": (("F30", "P30", "M30"), (0.5, 0.5, 0.5), ()),
    "0801": (("F5", "M30", "M5"), (1.0, 1.0, 1.0), ()),
    "0802": (("P30", "F0", "P0"), (1.0, 1.0, 1.0), ()),
    "0803": (("P5", "F30", "M0"), (1.0, 1.0, 1.0), ()),
    "0901": (("P30", "F5", "F0"), (0.5, 0.5, 1.0), ()),
    "0902": (("M5", "F30", "P0"), (0.5, 1.0, 1.0), ()),
    "0903": (("P5", "M0", "M30"), (1.0, 1.0, 1.0), ()),
    "1001": (("M0", "P0", "M5"), (1.0, 1.0, 1.0), ()),
    "1002": (("F0", "P30", "F30"), (1.0, 1.0, 1.0), ()),
    "1003": (("F5", "M30", "P5"), (1.0, 1.0, 1.0), ()),
}

# Table 2: Experiment 2.  Its all-1-mA sessions are repeated observations,
# not 0.5-vs-1-mA amplitude-contrast evidence.
_EXP2_ROWS = {
    "1101": ("F30",),
    "1102": ("M30",),
    "1201": ("M30",),
    "1202": ("F30",),
    "1301": ("F30",),
    "1302": ("M30",),
    "1401": ("M30",),
    "1402": ("F30",),
    "1501": ("F30",),
    "1502": ("M30",),
    "1601": ("M30",),
    "1602": ("F30",),
    "1801": ("M30",),
    "1802": ("F30",),
    "1901": ("F30",),
    "1902": ("M30",),
    "2001": ("F30",),
    "2002": ("M30",),
    "2101": ("F30",),
    "2102": ("M30",),
    "2201": ("M30",),
    "2202": ("F30",),
    "2301": ("F30",),
    "2302": ("M30",),
    "2401": ("M30",),
    "2402": ("F30",),
    "2501": ("M30",),
    "2502": ("F30",),
    "2601": ("F30",),
    "2602": ("M30",),
}

# The paper documents two interrupted Experiment 1 F0 blocks and one
# Experiment 1 0201 first-trial ramp anomaly.  They remain visible in the
# event ledger and cannot contribute to semantic support.
_TECHNICAL_TRIALS = {("0201", 1, 1)}
_RAMP_DOWN_OVERRIDES_SEC = {("0201", 1, 1): 30.0}

# Publication notes identify repeated participant numbers as the same
# biological unit.  This map is used only for support counting.
_BIOLOGICAL_ID = {
    "012": "bio-012",
    "019": "bio-012",
    "015": "bio-015",
    "018": "bio-015",
    "021": "bio-021",
    "025": "bio-021",
    "026": "bio-021",
    "022": "bio-022",
    "023": "bio-022",
    "024": "bio-022",
}

_TRIGGER_RE = re.compile(r"^\s*(16|32)(?:\.0+)?\s*$")
_BLOCK_RE = re.compile(r"^\s*2(?:\.0+)?\s*$")
_SUBJECT_RE = re.compile(r"(?:^|[/\\])sub-([^/\\]+)", re.IGNORECASE)
_SESSION_RE = re.compile(r"(?:^|[/\\])ses-([^/\\]+)", re.IGNORECASE)
_METADATA_NAMES = {"README", "CHANGES", "LICENSE", "CITATION.cff"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean(value: object) -> str | None:
    text = str(value).strip()
    return text if text and text.lower() not in {"n/a", "na", "none", "nan"} else None


def _number(value: object) -> float | None:
    text = _clean(value)
    if text is None:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _file_context(path: Path, root: Path) -> tuple[str | None, str | None, str]:
    relative = str(path.relative_to(root))
    subject_match = _SUBJECT_RE.search(relative)
    session_match = _SESSION_RE.search(relative)
    subject = subject_match.group(1) if subject_match else None
    session = session_match.group(1) if session_match else None
    file_num = (
        f"{int(subject):02d}{int(session):02d}"
        if subject and session and subject.isdigit() and session.isdigit()
        else ""
    )
    return subject, session, file_num


def _protocol_for_file(file_num: str) -> dict[str, Any] | None:
    if file_num in _EXP1_ROWS:
        stim_types, amplitudes, technical_blocks = _EXP1_ROWS[file_num]
        return {
            "experiment": 1,
            "rho_source": "publication_table_1",
            "rho_contrast_eligible": True,
            "blocks": [
                {
                    "block": index + 1,
                    "stim_type": stim_type,
                    "rho": amplitudes[index],
                    "technical_error_reported": index + 1 in technical_blocks,
                }
                for index, stim_type in enumerate(stim_types)
            ],
        }
    if file_num in _EXP2_ROWS:
        return {
            "experiment": 2,
            "rho_source": "publication_table_2",
            "rho_contrast_eligible": False,
            "blocks": [
                {
                    "block": 1,
                    "stim_type": stim_type,
                    "rho": 1.0,
                    "technical_error_reported": False,
                }
                for stim_type in _EXP2_ROWS[file_num]
            ],
        }
    return None


def _stim_identity(stim_type: str) -> dict[str, Any]:
    target = {"F": "frontal", "M": "motor", "P": "parietal"}.get(stim_type[:1])
    frequency = int(stim_type[1:]) if stim_type[1:].isdigit() else None
    waveform = "DC" if frequency == 0 else "sinusoidal" if frequency in {5, 30} else None
    return {
        "target": target,
        "waveform": waveform,
        "frequency_hz": frequency,
        "stim_type": stim_type,
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]
    except (OSError, csv.Error):
        return []


def _trigger_value(row: Mapping[str, str]) -> str | None:
    for key in ("value", "stim_file", "trigger", "event_code", "code"):
        if key in row:
            value = _clean(row[key])
            if value is not None:
                return value
    return None


def _trigger_code(value: str | None) -> str | None:
    if value is None:
        return None
    match = _TRIGGER_RE.fullmatch(value)
    return match.group(1) if match else None


def _read_eeg_sidecar(path: Path) -> dict[str, Any] | None:
    sidecar = path.with_name(path.name.replace("_events.tsv", "_eeg.json"))
    try:
        value = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    description = value.get("SubjectArtefactDescription")
    if not isinstance(description, str):
        return None
    file_match = re.search(r"\bFile:\s*([0-9]+)", description, re.IGNORECASE)
    montage_match = re.search(
        r"\bMontages:\s*([^;]+)",
        description,
        re.IGNORECASE,
    )
    amplitude_match = re.search(
        r"\bStim\s+Amplitude\s*\(\s*mA\s*\)\s*:\s*([^;]+)",
        description,
        re.IGNORECASE,
    )
    if not file_match or not montage_match or not amplitude_match:
        return None
    montages = tuple(
        token.strip().replace("♦", "").strip()
        for token in montage_match.group(1).split(",")
        if token.strip()
    )
    amplitudes: list[float] = []
    for token in amplitude_match.group(1).split(","):
        number = _number(token)
        if number is None:
            return None
        amplitudes.append(number)
    return {
        "file_num": file_match.group(1).zfill(4),
        "montages": montages,
        "amplitudes_mA": tuple(amplitudes),
        "sidecar_path": str(sidecar.name),
    }


def _protocol_sidecar_match(
    protocol: Mapping[str, Any] | None,
    sidecar: Mapping[str, Any] | None,
) -> bool:
    if protocol is None or sidecar is None:
        return False
    expected_montages = tuple(block["stim_type"] for block in protocol["blocks"])
    expected_amplitudes = tuple(float(block["rho"]) for block in protocol["blocks"])
    actual_montages = tuple(sidecar.get("montages", ()))
    actual_amplitudes = tuple(sidecar.get("amplitudes_mA", ()))
    return (
        actual_montages == expected_montages
        and len(actual_amplitudes) == len(expected_amplitudes)
        and all(
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
            for actual, expected in zip(actual_amplitudes, expected_amplitudes)
        )
    )


def _event_table(path: Path, root: Path) -> list[dict[str, Any]]:
    rows = _read_rows(path)
    if not rows:
        return []
    subject, session, path_file_num = _file_context(path, root)
    sidecar = _read_eeg_sidecar(path)
    file_num = sidecar["file_num"] if sidecar else path_file_num
    protocol = _protocol_for_file(file_num)
    sidecar_matches_protocol = _protocol_sidecar_match(protocol, sidecar)
    columns = list(rows[0])
    onset_field = "onset" if "onset" in columns else None
    block_starts: list[float] = []
    trigger_rows: list[tuple[float, str]] = []
    for row in rows:
        onset = _number(row.get(onset_field)) if onset_field else None
        if onset is None:
            continue
        value = _trigger_value(row)
        if value is None:
            continue
        if _BLOCK_RE.match(value):
            block_starts.append(onset)
            continue
        code = _trigger_code(value)
        if code is not None:
            trigger_rows.append((onset, code))

    block_starts.sort()
    trigger_rows.sort()
    starts = [onset for onset, code in trigger_rows if code == "16"]
    stops = [onset for onset, code in trigger_rows if code == "32"]
    paired: list[dict[str, Any]] = []
    stop_index = 0
    for start in starts:
        while stop_index < len(stops) and stops[stop_index] <= start:
            stop_index += 1
        if stop_index >= len(stops):
            paired.append({"trigger_start": start, "trigger_max_end": None})
            continue
        stop = stops[stop_index]
        stop_index += 1
        paired.append({"trigger_start": start, "trigger_max_end": stop})

    active_block_starts = sorted(
        {
            max((block_start for block_start in block_starts if block_start <= event["trigger_start"]), default=None)
            for event in paired
        }
        - {None}
    )
    block_number = {onset: index + 1 for index, onset in enumerate(active_block_starts)}
    expected_blocks = (
        len(protocol["blocks"])
        if protocol is not None
        else None
    )
    block_cardinality_ok = expected_blocks is not None and len(active_block_starts) == expected_blocks
    trials_in_block: dict[float, int] = defaultdict(int)
    result: list[dict[str, Any]] = []
    for index, event in enumerate(paired):
        start = event["trigger_start"]
        max_end = event["trigger_max_end"]
        active_start = max(
            (block_start for block_start in block_starts if block_start <= start),
            default=None,
        )
        block = block_number.get(active_start)
        trials_in_block[active_start] += 1 if active_start is not None else 0
        trial_in_block = trials_in_block.get(active_start, 0) if active_start is not None else None
        next_start = paired[index + 1]["trigger_start"] if index + 1 < len(paired) else None
        ramp_down_sec = _RAMP_DOWN_OVERRIDES_SEC.get(
            (file_num, block, trial_in_block),
            RAMP_DOWN_SEC,
        )
        timing_status = "PASS" if max_end is not None else "TIMING_UNRESOLVED"
        if max_end is None:
            ramp_down_end = None
            isolated = None
        else:
            ramp_down_end = max_end + ramp_down_sec
            isolated = (
                next_start - ramp_down_end
                if next_start is not None and next_start >= ramp_down_end
                else None
            )
        block_info = (
            protocol["blocks"][block - 1]
            if (
                block_cardinality_ok
                and protocol is not None
                and block is not None
                and 0 < block <= len(protocol["blocks"])
            )
            else None
        )
        semantic_status = timing_status
        if semantic_status == "PASS" and not block_cardinality_ok:
            semantic_status = "SOURCE_SEMANTICS_UNRESOLVED"
        if semantic_status == "PASS" and block_info is None:
            semantic_status = "SOURCE_SEMANTICS_UNRESOLVED"
        technical_trial = (file_num, block, trial_in_block) in _TECHNICAL_TRIALS
        if semantic_status == "PASS" and (
            not sidecar_matches_protocol
            or block_info["technical_error_reported"]
            or technical_trial
        ):
            semantic_status = (
                "SOURCE_SEMANTICS_UNRESOLVED"
                if not sidecar_matches_protocol
                else "TECHNICAL_ERROR_REPORTED"
            )
        identity = _stim_identity(block_info["stim_type"]) if block_info else {}
        result.append(
            {
                "subject": subject,
                "biological_unit": _BIOLOGICAL_ID.get(subject or "", f"bio-{subject}"),
                "experiment": protocol["experiment"] if protocol else None,
                "session": session,
                "path_file_num": path_file_num,
                "source_file_num": file_num,
                "sidecar_path": sidecar.get("sidecar_path") if sidecar else None,
                "sidecar_matches_protocol": sidecar_matches_protocol,
                "file_num": file_num,
                "block": block,
                "trial_in_block": trial_in_block,
                "w": {
                    "experiment": protocol["experiment"] if protocol else None,
                    "session": session,
                    "block": block,
                },
                "v": {
                    "target": identity.get("target"),
                    "waveform": identity.get("waveform"),
                    "frequency_hz": identity.get("frequency_hz"),
                },
                "stim_type": identity.get("stim_type"),
                "target": identity.get("target"),
                "waveform": identity.get("waveform"),
                "frequency_hz": identity.get("frequency_hz"),
                "rho": block_info["rho"] if block_info else None,
                "rho_units": "mA" if block_info else None,
                "rho_source": protocol["rho_source"] if protocol else None,
                "rho_contrast_eligible": protocol["rho_contrast_eligible"] if protocol else False,
                "trigger_start": start,
                "trigger_max_end": max_end,
                "ramp_down_end": (
                    max_end + ramp_down_sec if max_end is not None else None
                ),
                "ramp_down_sec": ramp_down_sec if max_end is not None else None,
                "next_stim_start": next_start,
                "T_isolated": isolated,
                "timing_status": timing_status,
                "semantic_status": semantic_status,
                "technical_error_reported": bool(
                    block_info
                    and (
                        block_info["technical_error_reported"]
                        or technical_trial
                    )
                ),
                "block_cardinality_ok": block_cardinality_ok,
                "source_file": str(path.relative_to(root)),
            }
        )
    return result


def _source_binding(root: Path, far001_payload: Mapping[str, Any]) -> dict[str, Any]:
    expected = next(
        (
            dataset
            for dataset in far001_payload.get("datasets", [])
            if dataset.get("dataset_id") == DATASET_ID
        ),
        None,
    )
    if expected is None:
        return {"status": "ENTRY_SOURCE_MISSING"}
    expected_root = Path(expected.get("source_root", ""))
    expected_records = {
        str(record["relative_path"]).replace("\\", "/"): record.get("sha256")
        for record in expected.get("metadata_inventory", {}).get("metadata_file_records", [])
    }
    metadata_files, payload_paths = _metadata_files(root)
    actual_records = {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in metadata_files
    }
    payloads = [
        str(path.relative_to(root)).replace("\\", "/")
        for path in payload_paths
    ]
    known_paths = set(metadata_files) | set(payload_paths)
    unexpected_files: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root)).replace("\\", "/")
        if path not in known_paths and path.name not in _METADATA_NAMES:
            unexpected_files.append(relative)
    return {
        "status": (
            "PASS"
            if str(expected_root.resolve()) == str(root.resolve())
            and actual_records == expected_records
            and not payloads
            and not unexpected_files
            else "SOURCE_BINDING_FAILED"
        ),
        "expected_root": str(expected_root),
        "actual_root": str(root),
        "expected_metadata_file_count": len(expected_records),
        "actual_metadata_file_count": len(actual_records),
        "signal_payloads": payloads,
        "unexpected_files": unexpected_files,
    }


def audit_ds003670(
    root: str | Path,
    far001_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Join ds003670 BIDS triggers to source protocol semantics."""
    source_root = Path(root)
    binding = _source_binding(source_root, far001_payload)
    event_files = sorted(source_root.rglob("*_events.tsv"))
    if binding["status"] != "PASS":
        return {
            "schema": "mndm.far_ext_002a_ds003670_semantic_join.v1",
            "protocol_id": PROTOCOL_ID,
            "dataset_id": DATASET_ID,
            "source_root": str(source_root),
            "source_binding": binding,
            "source_documents": SOURCE_DOCUMENTS,
            "event_file_count": len(event_files),
            "events": [],
            "families": [],
            "global_status": "SOURCE_SEMANTICS_UNRESOLVED",
            "claim_boundary": (
                "The source root failed the FAR-EXT-001 byte/payload binding; "
                "no semantic promotion was attempted."
            ),
            "audit_scope": {
                "metadata_only": True,
                "signal_payloads_opened": False,
                "outcome_tables_opened": False,
                "nmd_outputs_opened": False,
                "far_calculated": False,
            },
        }
    events: list[dict[str, Any]] = []
    for path in event_files:
        events.extend(_event_table(path, source_root))
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    nmd_window = float(
        far001_payload.get("nmd_window_sec", {}).get("eeg", NMD_WINDOW_SEC)
    )
    for event in events:
        if (
            event["semantic_status"] == "PASS"
            and event["rho_contrast_eligible"]
            and event["T_isolated"] is not None
            and event["T_isolated"] > nmd_window
        ):
            key = json.dumps(
                {
                    "target": event["target"],
                    "waveform": event["waveform"],
                    "frequency_hz": event["frequency_hz"],
                },
                sort_keys=True,
            )
            families[key].append(event)

    family_results: list[dict[str, Any]] = []
    for v_key, family_events in sorted(families.items()):
        rho_values = sorted({event["rho"] for event in family_events})
        subjects_by_rho = {
            str(rho): sorted(
                {event["biological_unit"] for event in family_events if event["rho"] == rho}
            )
            for rho in rho_values
        }
        horizons = [event["T_isolated"] for event in family_events]
        min_subjects = min(
            (len(subjects) for subjects in subjects_by_rho.values()),
            default=0,
        )
        family_status = (
            "DS003670_LIMITED_CURVE_SEMANTICS_PASS"
            if set(rho_values) == set(RHO_LEVELS_MA)
            and min_subjects >= MIN_BIOLOGICAL_UNITS
            and horizons
            and min(horizons) > nmd_window
            else "POINT_SEMANTICS_ONLY"
            if rho_values
            else "SOURCE_SEMANTICS_UNRESOLVED"
        )
        family_results.append(
            {
                "v": json.loads(v_key),
                "rho_levels_mA": rho_values,
                "subjects_per_rho": subjects_by_rho,
                "min_biological_units_across_rho": min_subjects,
                "events": len(family_events),
                "min_T_isolated_sec": min(horizons) if horizons else None,
                "nmd_window_sec": nmd_window,
                "semantic_status": family_status,
                "tolerability_confound": True,
            }
        )

    curve_families = [
        family
        for family in family_results
        if family["semantic_status"] == "DS003670_LIMITED_CURVE_SEMANTICS_PASS"
    ]
    global_status = (
        "DS003670_LIMITED_CURVE_SEMANTICS_PASS"
        if binding["status"] == "PASS" and curve_families
        else "POINT_SEMANTICS_ONLY"
        if binding["status"] == "PASS" and family_results
        else "SOURCE_SEMANTICS_UNRESOLVED"
    )
    return {
        "schema": "mndm.far_ext_002a_ds003670_semantic_join.v1",
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "source_root": str(source_root),
        "source_binding": binding,
        "source_documents": SOURCE_DOCUMENTS,
        "event_file_count": len(event_files),
        "events": events,
        "families": family_results,
        "experiment_semantics": {
            "experiment_1": {
                "rho_levels_mA": list(RHO_LEVELS_MA),
                "rho_contrast_eligible": True,
                "rho_source": "publication_table_1",
            },
            "experiment_2": {
                "rho_levels_mA": [1.0],
                "rho_contrast_eligible": False,
                "rho_source": "publication_table_2",
            },
        },
        "global_status": global_status,
        "tolerability_confound": True,
        "next_gate": (
            "A separately authorized signal/time-base eligibility gate may be "
            "proposed; FAR-EXT-002A does not authorize FAR-003B."
        ),
        "claim_boundary": (
            "FAR-EXT-002A is an outcome-blind ds003670 protocol-semantic join. "
            "It establishes no neural response, MNPS trajectory, FAR value, or "
            "payload eligibility."
        ),
        "audit_scope": {
            "metadata_only": True,
            "signal_payloads_opened": False,
            "outcome_tables_opened": False,
            "nmd_outputs_opened": False,
            "far_calculated": False,
        },
    }
