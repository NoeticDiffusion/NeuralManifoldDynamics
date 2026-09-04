"""Read-only FAR-001 perturbation-source inventory.

The scout audits source semantics only.  It does not read NMD outputs, open
signal payloads, infer outcomes, fit a curve, or promote stimulation frequency
to perturbation amplitude.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping


PROTOCOL_ID = "FAR-001"
CURVE_ELIGIBLE = "CURVE_ELIGIBLE_CANDIDATE"
SINGLE_AMPLITUDE = "SINGLE_AMPLITUDE_CANDIDATE"
DIRECTION_ONLY = "DIRECTION_ONLY_CANDIDATE"
NO_EXPLICIT = "NO_EXPLICIT_PERTURBATION"
SOURCE_UNCERTAIN = "SOURCE_UNCERTAIN"

PHOTO_PATTERN = re.compile(r"^PHOTO\s*(?P<frequency>\d+(?:\.\d+)?)\s*Hz$", re.I)
SIGNAL_SUFFIXES = {".edf", ".bdf", ".set", ".fdt", ".eeg", ".vhdr", ".nwb"}
EXCLUDED_SOURCE_DIRECTORIES = {".git", ".datalad", "derivatives"}
AMPLITUDE_TERMS = ("intens", "lumin", "irradi", "amplitude", "current", "dose", "sham")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(_read_text(path))
    if not isinstance(value, dict):
        raise ValueError(f"json_root_not_object:{path}")
    return value


def _read_participant_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        return sum(1 for row in rows if any(str(value).strip() for value in row.values()))


def _walk_source_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for directory, subdirectories, filenames in os.walk(root):
        subdirectories[:] = [
            name
            for name in subdirectories
            if name not in EXCLUDED_SOURCE_DIRECTORIES
        ]
        directory_path = Path(directory)
        paths.extend(directory_path / filename for filename in filenames)
    return paths


def _read_event_file(path: Path) -> dict[str, Any]:
    fields: set[str] = set()
    photo_frequencies: list[float] = []
    photo_onsets: list[float] = []
    photo_durations: list[float] = []
    row_count = 0
    read_error: str | None = None
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        fields.update(str(field) for field in rows.fieldnames or [])
        for row in rows:
            row_count += 1
            label = str(row.get("value", "")).strip()
            match = PHOTO_PATTERN.fullmatch(label)
            if match is None:
                continue
            photo_frequencies.append(float(match.group("frequency")))
            try:
                onset = float(str(row.get("onset", "")).strip())
                duration = float(str(row.get("duration", "")).strip())
            except (TypeError, ValueError):
                read_error = "non_numeric_photo_timing"
                continue
            if onset == onset:
                photo_onsets.append(onset)
            if duration == duration:
                photo_durations.append(duration)
    return {
        "path": str(path),
        "row_count": row_count,
        "fields": sorted(fields),
        "photo_frequencies_hz": photo_frequencies,
        "photo_onsets_sec": photo_onsets,
        "photo_durations_sec": photo_durations,
        "read_error": read_error,
    }


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "relative_path": str(path.relative_to(root)).replace("\\", "/"),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _scan_ds006036(root: Path, config_path: Path) -> dict[str, Any]:
    required = [
        root / "README",
        root / "dataset_description.json",
        root / "task-photomark_events.json",
        root / "participants.tsv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {
            "dataset_id": "ds006036",
            "source_status": "SOURCE_READ_ERROR",
            "source_root": str(root),
            "classification": SOURCE_UNCERTAIN,
            "failure_reason": "missing_required_metadata",
            "missing_files": missing,
        }

    source_files = _walk_source_files(root)
    event_files = sorted(
        path for path in source_files if path.name.endswith("_events.tsv")
    )
    event_records = [_read_event_file(path) for path in event_files]
    all_photo_frequencies = [
        frequency
        for record in event_records
        for frequency in record["photo_frequencies_hz"]
    ]
    all_photo_onsets = [
        onset for record in event_records for onset in record["photo_onsets_sec"]
    ]
    all_photo_durations = [
        duration
        for record in event_records for duration in record["photo_durations_sec"]
    ]
    fields = sorted(
        {
            field
            for record in event_records
            for field in record["fields"]
        }
    )
    task_metadata = _read_json(root / "task-photomark_events.json")
    task_metadata_text = json.dumps(task_metadata, ensure_ascii=False).lower()
    source_metadata = [
        root / "README",
        root / "dataset_description.json",
        root / "task-photomark_events.json",
        root / "participants.tsv",
    ]
    source_metadata_records = [
        _file_record(path, root) for path in source_metadata if path.is_file()
    ]
    signal_files = sorted(
        path for path in source_files if path.suffix.lower() in SIGNAL_SUFFIXES
    )
    frequency_levels = sorted(set(all_photo_frequencies))
    read_errors = sorted(
        {
            str(record["read_error"])
            for record in event_records
            if record["read_error"] is not None
        }
    )
    intensity_fields = sorted(
        field
        for field in fields
        if any(token in field.lower() for token in AMPLITUDE_TERMS)
    )
    amplitude_terms_in_metadata = sorted(
        token for token in AMPLITUDE_TERMS if token in task_metadata_text
    )
    readme = _read_text(root / "README")
    source_defines_incremental_frequency = (
        "incremental frequencies" in readme.lower()
        or "progressing to" in readme.lower()
    )
    explicit_photic_protocol = bool(all_photo_frequencies)
    if read_errors:
        classification = SOURCE_UNCERTAIN
        classification_reason = "critical_event_metadata_read_error"
    elif not explicit_photic_protocol:
        classification = NO_EXPLICIT
        classification_reason = (
            "No explicit PHOTO frequency event was found in the audited "
            "source event stream."
        )
    elif intensity_fields or amplitude_terms_in_metadata:
        classification = SOURCE_UNCERTAIN
        classification_reason = (
            "The source contains an amplitude-related metadata term, but "
            "this scout does not adjudicate its physical units, level "
            "ordering, sham, or onset/offset semantics."
        )
    else:
        classification = DIRECTION_ONLY
        classification_reason = (
            "The source explicitly imposes photic stimulation and records "
            "multiple frequency-defined perturbation types, but the audited "
            "metadata contains no physical intensity/luminance/amplitude "
            "variable, no rho=0/sham level, and no explicit offset. Frequency "
            "is retained as v, not promoted to rho."
        )
    return {
        "dataset_id": "ds006036",
        "source_status": "AVAILABLE" if not read_errors else "SOURCE_READ_ERROR",
        "source_kind": "openneuro_bids_eeg",
        "source_root": str(root),
        "config_path": str(config_path),
        "source_metadata": source_metadata_records,
        "event_file_count": len(event_files),
        "event_row_count": sum(int(record["row_count"]) for record in event_records),
        "photo_event_count": len(all_photo_frequencies),
        "photo_frequency_levels_hz": frequency_levels,
        "photo_frequency_event_counts": {
            str(level): int(all_photo_frequencies.count(level))
            for level in frequency_levels
        },
        "event_fields": fields,
        "intensity_or_luminance_fields": intensity_fields,
        "participant_count": _read_participant_count(root / "participants.tsv"),
        "signal_file_count": len(signal_files),
        "signal_file_suffixes": sorted({path.suffix.lower() for path in signal_files}),
        "first_photo_onset_sec": min(all_photo_onsets) if all_photo_onsets else None,
        "photo_onsets_with_pre_recording": bool(
            all_photo_onsets and min(all_photo_onsets) > 0.0
        ),
        "photo_duration_values_sec": sorted(set(all_photo_durations)),
        "source_defines_incremental_frequency": source_defines_incremental_frequency,
        "source_semantics": {
            "perturbation_explicitly_imposed": explicit_photic_protocol,
            "rho_known": False,
            "rho_unit": None,
            "multiple_rho_levels": False,
            "rho_zero_or_sham": False,
            "v_known": bool(frequency_levels),
            "v_definition": "stimulation_frequency_hz",
            "onset_known": bool(all_photo_onsets),
            "offset_known": False,
            "pre_perturbation_baseline_observed": bool(
                all_photo_onsets and min(all_photo_onsets) > 0.0
            ),
            "post_perturbation_observation_present": bool(signal_files),
            "repeated_perturbations": len(all_photo_frequencies) > len(frequency_levels),
            "continuous_neural_signal": bool(signal_files),
            "pid_held_out_feasible": _read_participant_count(root / "participants.tsv") >= 3,
        },
        "amplitude_terms_in_task_metadata": amplitude_terms_in_metadata,
        "classification": classification,
        "classification_reason": classification_reason,
        "read_errors": read_errors,
    }


def _candidate_specs(repo_root: Path) -> list[dict[str, Any]]:
    return [
        {
            "dataset_id": "ds006036",
            "config_path": repo_root / "mndm/config/sources/openneuro/config_ingest_ds006036.yaml",
            "roots": [Path(r"M:\datasets\received\openneuro\ds006036")],
            "source_kind": "openneuro_bids_eeg",
        },
        {
            "dataset_id": "dandi_000458",
            "config_path": repo_root / "mndm/config/sources/dandi/config_ingest_dandi_000458.yaml",
            "roots": [
                repo_root / "data/dandi/raw/000458",
                Path(r"M:\datasets\received\dandi\000458"),
            ],
            "source_kind": "dandi_nwb_electrical_stimulation",
        },
        {
            "dataset_id": "vitaldb",
            "config_path": repo_root / "mndm/config/sources/other/config_ingest_vitaldb.yaml",
            "roots": [
                Path(r"E:\Science_Datasets\vitaldb\received\cases"),
                repo_root / "data/vitaldb/raw",
            ],
            "source_kind": "vitaldb_propofol_waveforms",
        },
        {
            "dataset_id": "ds006623",
            "config_path": repo_root / "mndm/config/sources/openneuro/config_ingest_ds006623.yaml",
            "roots": [Path(r"M:\datasets\received\openneuro\ds006623")],
            "source_kind": "openneuro_propofol_fmri",
        },
        {
            "dataset_id": "ds005917",
            "config_path": repo_root / "mndm/config/sources/openneuro/config_ingest_ds005917.yaml",
            "roots": [Path(r"M:\datasets\received\openneuro\ds005917")],
            "source_kind": "openneuro_ketamine_fmri",
        },
        {
            "dataset_id": "dandi_000009",
            "config_path": repo_root / "mndm/config/sources/dandi/config_ingest_dandi_000009.yaml",
            "roots": [
                repo_root / "data/dandi/raw/000009",
                Path(r"M:\datasets\received\dandi\000009"),
            ],
            "source_kind": "dandi_neuropixel_photostimulation",
        },
    ]


def _unavailable_candidate(spec: Mapping[str, Any]) -> dict[str, Any]:
    roots = [Path(root) for root in spec["roots"]]
    existing_roots = [str(root) for root in roots if root.is_dir()]
    source_status = "AVAILABLE" if existing_roots else "SOURCE_UNAVAILABLE"
    return {
        "dataset_id": spec["dataset_id"],
        "source_status": source_status,
        "source_kind": spec["source_kind"],
        "source_root": existing_roots[0] if existing_roots else None,
        "config_path": str(spec["config_path"]),
        "config_exists": Path(spec["config_path"]).is_file(),
        "candidate_roots": [str(root) for root in roots],
        "existing_roots": existing_roots,
        "classification": SOURCE_UNCERTAIN,
        "classification_reason": (
            (
                "The configured source root is not locally available. No "
                "amplitude, sham, direction, onset, offset, or recovery "
                "claim is made from configuration alone."
                if not existing_roots
                else
                "The source root is available but this FAR-001 scout has no "
                "source-specific metadata adapter for it. No amplitude, "
                "sham, direction, onset, offset, or recovery claim is made "
                "from configuration alone."
            )
        ),
        "source_semantics": {
            "perturbation_explicitly_imposed": None,
            "rho_known": None,
            "rho_unit": None,
            "multiple_rho_levels": None,
            "rho_zero_or_sham": None,
            "v_known": None,
            "onset_known": None,
            "offset_known": None,
            "pre_perturbation_baseline_observed": None,
            "post_perturbation_observation_present": None,
            "repeated_perturbations": None,
            "continuous_neural_signal": None,
            "pid_held_out_feasible": None,
        },
    }


def run_inventory(
    *,
    repo_root: Path,
    source_roots: Mapping[str, Path] | None = None,
    protocol_path: Path | None = None,
) -> dict[str, Any]:
    """Run the FAR-001 inventory without opening signal payloads."""
    protocol_path = protocol_path or (
        repo_root
        / "project/orthagonal_axis/orthagonal_dynamics/finite-amplitude_resilience/"
        "002_far_source_scout_prereg.md"
    )
    if not protocol_path.is_file():
        raise FileNotFoundError(f"missing_protocol:{protocol_path}")
    candidates: list[dict[str, Any]] = []
    for spec in _candidate_specs(repo_root):
        override = (source_roots or {}).get(spec["dataset_id"])
        roots = [override] if override is not None else spec["roots"]
        available_root = next((root for root in roots if root.is_dir()), None)
        if spec["dataset_id"] == "ds006036" and available_root is not None:
            candidates.append(
                _scan_ds006036(
                    available_root,
                    Path(spec["config_path"]),
                )
            )
        else:
            updated_spec = dict(spec)
            updated_spec["roots"] = roots
            candidates.append(_unavailable_candidate(updated_spec))

    classifications = [candidate["classification"] for candidate in candidates]
    n_curve = classifications.count(CURVE_ELIGIBLE)
    n_single = classifications.count(SINGLE_AMPLITUDE)
    if n_curve:
        gate_status = "PASS"
        gate_reason = "curve_eligible_candidate_present"
    elif n_single:
        gate_status = "CONDITIONAL_PASS"
        gate_reason = "single_amplitude_candidate_only"
    else:
        gate_status = "NOT_TESTABLE"
        gate_reason = "no_genuine_amplitude_candidate"
    return {
        "schema": "mndm.far_001_source_scout.v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": sha256_file(protocol_path),
        "gate_status": gate_status,
        "gate_reason": gate_reason,
        "datasets": candidates,
        "classification_counts": {
            classification: classifications.count(classification)
            for classification in sorted(set(classifications))
        },
        "fail_closed_assertions": {
            "empirical_outputs_read": False,
            "nmd_outputs_read": False,
            "signal_payloads_opened": False,
            "downloads_performed": False,
            "models_fit": False,
            "resilience_curve_estimated": False,
            "home_away_defined": False,
            "frequency_promoted_to_amplitude": False,
            "outcome_table_created": False,
        },
        "claim_boundary": (
            "FAR-001 is a source inventory only. It classifies explicit "
            "perturbation semantics and does not qualify a dataset, define "
            "home/away, estimate R(rho), or authorize FAR-002."
        ),
    }
