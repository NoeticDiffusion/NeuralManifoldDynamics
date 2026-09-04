"""FAR-001B native metadata audit for local perturbation sources.

Only source metadata is read.  NWB trial columns are accessed through h5py;
acquisition data, units/spike times, BOLD images, physio, and NMD outputs are
never opened or read.
"""

from __future__ import annotations

import csv
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping


PROTOCOL_ID = "FAR-001B"
CURVE_CANDIDATE = "CURVE_CANDIDATE"
POINT_OR_LIMITED_CANDIDATE = "POINT_OR_LIMITED_CANDIDATE"
DIRECTION_ONLY = "DIRECTION_ONLY"
NO_RHO = "NO_RHO"
SOURCE_UNCERTAIN = "SOURCE_UNCERTAIN"

EXCLUDED_DIRECTORIES = {".git", ".datalad", "derivatives"}
UNIT_PATTERN = re.compile(r"(?:μ|µ|u)A|mW|mg/?kg|ng/?kg|mg|μg|ug|%")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk_files(root: Path, *, suffixes: set[str] | None = None) -> list[Path]:
    files: list[Path] = []
    for directory, subdirectories, filenames in os.walk(root):
        subdirectories[:] = [
            name for name in subdirectories if name not in EXCLUDED_DIRECTORIES
        ]
        directory_path = Path(directory)
        for filename in filenames:
            path = directory_path / filename
            if suffixes is None or path.suffix.lower() in suffixes:
                files.append(path)
    return sorted(files)


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _finite_float(value: Any) -> float | None:
    try:
        number = float(_decode(value))
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _numeric_text(value: Any) -> float | None:
    text = str(_decode(value)).strip()
    if not text or text.lower() in {"n/a", "na", "nan", "none", "null"}:
        return None
    return _finite_float(text)


def _truthy(value: Any) -> bool:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, bool):
        return value
    text = str(_decode(value)).strip().lower()
    return text not in {"0", "false", "no", "n", "nan", ""}


def _unit_tokens(values: list[Any]) -> list[str]:
    return sorted(
        {
            match.group(0)
            for value in values
            for match in UNIT_PATTERN.finditer(str(_decode(value)))
        }
    )


def _text_set(values: list[Any]) -> list[str]:
    return sorted(
        {
            str(_decode(value)).strip()
            for value in values
            if str(_decode(value)).strip()
            and str(_decode(value)).strip().lower()
            not in {"n/a", "na", "nan", "none", "null"}
        }
    )


def _metadata_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "relative_path": str(path.relative_to(root)).replace("\\", "/"),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _nwb_trial_table(
    path: Path,
    columns: list[str],
) -> tuple[dict[str, list[Any]], dict[str, str], list[str]]:
    """Read only selected `/intervals/trials` columns and descriptions."""
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("h5py_required_for_far_001b_nwb_audit") from error

    with h5py.File(path, "r") as handle:
        table = handle.get("/intervals/trials")
        if table is None or not hasattr(table, "keys"):
            return {}, {}, ["missing_intervals_trials"]
        available = set(str(name) for name in table.keys())
        missing = sorted(set(columns) - available)
        selected: dict[str, list[Any]] = {}
        descriptions: dict[str, str] = {}
        row_count = None
        for name in columns:
            if name not in available:
                continue
            dataset = table[name]
            values = [_decode(value) for value in dataset[:]]
            selected[name] = values
            description = dataset.attrs.get("description", "")
            descriptions[name] = str(_decode(description))
            row_count = len(values) if row_count is None else row_count
        if row_count is not None:
            selected["_row_count"] = [row_count]
        return selected, descriptions, missing


def _nwb_file_records(paths: list[Path], root: Path) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": str(path.relative_to(root)).replace("\\", "/"),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]


def _nwb_group_names(path: Path, group_paths: list[str]) -> dict[str, list[str]]:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("h5py_required_for_far_001b_nwb_audit") from error
    with h5py.File(path, "r") as handle:
        output: dict[str, list[str]] = {}
        for group_path in group_paths:
            group = handle.get(group_path)
            output[group_path] = (
                sorted(str(name) for name in group.keys())
                if group is not None and hasattr(group, "keys")
                else []
            )
        return output


def _classify_levels(
    *,
    levels: list[float],
    unit: str | None,
    timing_complete: bool,
    repetitions: int,
) -> str:
    nonzero = sorted({level for level in levels if level != 0.0})
    if unit and timing_complete and len(nonzero) >= 3 and repetitions > 0:
        return CURVE_CANDIDATE
    if unit and timing_complete and len(nonzero) in {1, 2} and repetitions > 0:
        return POINT_OR_LIMITED_CANDIDATE
    if levels:
        return SOURCE_UNCERTAIN
    return NO_RHO


def _audit_dandi_000458(root: Path) -> dict[str, Any]:
    files = _walk_files(root, suffixes={".nwb"})
    if not files:
        return {
            "dataset_id": "dandi_000458",
            "source_status": "SOURCE_UNCERTAIN",
            "source_root": str(root),
            "classification": SOURCE_UNCERTAIN,
            "failure_reason": "no_native_nwb_files",
        }

    columns = [
        "behavioral_epoch",
        "estim_current",
        "is_valid",
        "start_time",
        "stimulus_type",
        "stop_time",
    ]
    required_columns = [
        "behavioral_epoch",
        "estim_current",
        "start_time",
        "stimulus_type",
        "stop_time",
    ]
    current_levels: list[float] = []
    current_counts: Counter[float] = Counter()
    unique_levels_by_epoch: dict[str, set[float]] = {}
    unit_descriptions: set[str] = set()
    timing_rows = 0
    electrical_rows_total = 0
    electrical_rows_valid = 0
    electrical_rows_invalid = 0
    visual_rows_total = 0
    isoflurane_labels: set[str] = set()
    file_errors: list[dict[str, Any]] = []
    per_file: list[dict[str, Any]] = []
    subject_names: set[str] = set()

    for path in files:
        try:
            table, descriptions, missing = _nwb_trial_table(path, columns)
            required_missing = sorted(set(required_columns) & set(missing))
            if required_missing:
                file_errors.append(
                    {
                        "relative_path": str(path.relative_to(root)).replace("\\", "/"),
                        "missing_columns": required_missing,
                    }
                )
                continue
            row_count = int(table["_row_count"][0])
            types = table["stimulus_type"]
            epochs = table["behavioral_epoch"]
            currents = table["estim_current"]
            starts = table["start_time"]
            stops = table["stop_time"]
            validity = table.get("is_valid", [True] * row_count)
            validity_field_present = "is_valid" in table
            descriptions_text = descriptions.get("estim_current", "")
            if descriptions_text:
                unit_descriptions.add(descriptions_text)
            file_levels: list[float] = []
            file_invalid_electrical = 0
            subject_names.add(path.parent.name)
            for index in range(row_count):
                stimulus_type = str(_decode(types[index])).strip().lower()
                epoch_raw = str(_decode(epochs[index])).strip()
                epoch = epoch_raw.lower()
                current = _numeric_text(currents[index])
                if "isoflurane" in epoch:
                    isoflurane_labels.add(epoch_raw)
                if stimulus_type == "electrical":
                    electrical_rows_total += 1
                    valid = _truthy(validity[index])
                    if not valid:
                        electrical_rows_invalid += 1
                        file_invalid_electrical += 1
                    if valid and current is not None:
                        electrical_rows_valid += 1
                        current_levels.append(current)
                        file_levels.append(current)
                        current_counts[current] += 1
                        unique_levels_by_epoch.setdefault(epoch_raw, set()).add(current)
                    if valid:
                        start = _finite_float(starts[index])
                        stop = _finite_float(stops[index])
                        if start is not None and stop is not None and stop >= start:
                            timing_rows += 1
                elif stimulus_type == "visual":
                    visual_rows_total += 1
            per_file.append(
                {
                    "relative_path": str(path.relative_to(root)).replace("\\", "/"),
                    "row_count": row_count,
                    "electrical_current_levels_ua": sorted(set(file_levels)),
                    "stimulus_types": _text_set(types),
                    "behavioral_epochs": _text_set(epochs),
                    "is_valid_field_present": validity_field_present,
                    "invalid_electrical_rows": file_invalid_electrical,
                }
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            file_errors.append(
                {
                    "relative_path": str(path.relative_to(root)).replace("\\", "/"),
                    "error": f"{type(error).__name__}:{error}",
                }
            )

    unit_text = " ".join(sorted(unit_descriptions))
    unit_known = bool(re.search(r"(?:μ|µ|u)A", unit_text))
    electrical_levels = sorted(set(current_levels))
    electrical_classification = _classify_levels(
        levels=electrical_levels,
        unit="μA" if unit_known else None,
        timing_complete=(
            timing_rows == electrical_rows_valid and electrical_rows_valid > 0
        ),
        repetitions=electrical_rows_valid,
    )
    file_level_count_distribution = Counter(
        len(record["electrical_current_levels_ua"]) for record in per_file
    )
    identities = [
        {
            "identity": "electrical_stimulation_current",
            "rho_field": "estim_current",
            "rho_unit": "μA" if unit_known else None,
            "levels": electrical_levels,
            "level_counts": {str(level): int(current_counts[level]) for level in electrical_levels},
            "nonzero_level_count": len([level for level in electrical_levels if level != 0.0]),
            "rho_zero_or_sham": 0.0 in electrical_levels,
            "onset_field": "start_time",
            "offset_field": "stop_time",
            "timing_complete": (
                timing_rows == electrical_rows_valid and electrical_rows_valid > 0
            ),
            "subject_count": len(subject_names),
            "unique_levels_by_behavioral_epoch": {
                epoch: sorted(levels)
                for epoch, levels in sorted(unique_levels_by_epoch.items())
            },
            "file_level_count_distribution": {
                str(level_count): int(file_count)
                for level_count, file_count in sorted(file_level_count_distribution.items())
            },
            "files_with_at_least_three_levels": sum(
                count
                for level_count, count in file_level_count_distribution.items()
                if level_count >= 3
            ),
            "candidate_scope": (
                "source-level metadata candidate; FAR-002 must freeze "
                "session/state strata before a within-session curve claim"
            ),
            "invalid_electrical_rows_excluded": electrical_rows_invalid,
            "classification": electrical_classification,
        },
        {
            "identity": "isoflurane_condition",
            "rho_field": None,
            "rho_unit": None,
            "levels": [],
            "direction_or_condition_levels": sorted(isoflurane_labels),
            "classification": DIRECTION_ONLY,
            "reason": "isoflurane labels have no per-trial concentration field",
        },
        {
            "identity": "visual_stimulation",
            "rho_field": None,
            "rho_unit": None,
            "levels": [],
            "classification": DIRECTION_ONLY,
            "reason": "visual stimulus type has no audited intensity field",
        },
    ]
    top_classification = (
        CURVE_CANDIDATE
        if electrical_classification == CURVE_CANDIDATE
        else electrical_classification
    )
    return {
        "dataset_id": "dandi_000458",
        "source_status": "AVAILABLE" if not file_errors else "SOURCE_UNCERTAIN",
        "source_root": str(root),
        "native_nwb_file_count": len(files),
        "native_nwb_files": _nwb_file_records(files, root),
        "subject_count": len(subject_names),
        "trial_rows_electrical_total": electrical_rows_total,
        "trial_rows_electrical_valid": electrical_rows_valid,
        "trial_rows_electrical_invalid": electrical_rows_invalid,
        "trial_rows_visual": visual_rows_total,
        "trial_row_count": int(sum(record["row_count"] for record in per_file)),
        "unit_descriptions": sorted(unit_descriptions),
        "identities": identities,
        "classification": top_classification,
        "classification_reason": (
            "Electrical stimulation current has native μA units, explicit "
            "trial onset/offset, and multiple nonzero levels. This is a "
            "source-level candidate, not a completed within-session FAR curve. "
            "Isoflurane and visual conditions remain separate "
            "direction/condition identities."
        ),
        "file_records": per_file,
        "file_errors": file_errors,
        "audit_scope": {
            "opened_paths": ["/intervals/trials"],
            "forbidden_paths_not_opened": [
                "/acquisition",
                "/units",
                "/processing",
            ],
        },
    }


def _audit_dandi_000009(root: Path) -> dict[str, Any]:
    files = _walk_files(root, suffixes={".nwb"})
    if not files:
        return {
            "dataset_id": "dandi_000009",
            "source_status": SOURCE_UNCERTAIN,
            "source_root": str(root),
            "classification": SOURCE_UNCERTAIN,
            "failure_reason": "no_native_nwb_files",
        }
    columns = [
        "photo_stim_period",
        "photo_stim_power",
        "photo_stim_type",
        "start_time",
        "stop_time",
        "stim_present",
    ]
    required_columns = set(columns)
    powers: list[float] = []
    periods: list[Any] = []
    stimulus_types: list[Any] = []
    stim_present_values: list[Any] = []
    timing_rows = 0
    trial_row_count = 0
    descriptions: dict[str, str] = {}
    file_errors: list[dict[str, Any]] = []
    context_names: dict[str, set[str]] = {
        "/intervals": set(),
        "/stimulus": set(),
        "/general/optogenetics": set(),
        "/acquisition": set(),
    }
    for path in files:
        try:
            table, file_descriptions, missing = _nwb_trial_table(path, columns)
            if missing:
                file_errors.append(
                    {
                        "relative_path": str(path.relative_to(root)).replace("\\", "/"),
                        "missing_columns": sorted(required_columns & set(missing)),
                    }
                )
                continue
            descriptions.update(file_descriptions)
            trial_row_count += int(table["_row_count"][0])
            powers.extend(
                value
                for value in (_finite_float(item) for item in table["photo_stim_power"])
                if value is not None
            )
            periods.extend(table["photo_stim_period"])
            stimulus_types.extend(table["photo_stim_type"])
            stim_present_values.extend(table["stim_present"])
            timing_rows += sum(
                _finite_float(start) is not None and _finite_float(stop) is not None
                for start, stop in zip(table["start_time"], table["stop_time"])
            )
            for group_path, names in _nwb_group_names(
                path,
                list(context_names),
            ).items():
                context_names[group_path].update(names)
        except (OSError, ValueError, KeyError, TypeError) as error:
            file_errors.append(
                {
                    "relative_path": str(path.relative_to(root)).replace("\\", "/"),
                    "error": f"{type(error).__name__}:{error}",
                }
            )
    power_levels = sorted({value for value in powers if value is not None})
    power_counts = Counter(value for value in powers if value is not None)
    stim_present_counts = Counter(
        str(_decode(value)).strip() for value in stim_present_values
    )
    unit_known = "mW" in descriptions.get("photo_stim_power", "")
    return {
        "dataset_id": "dandi_000009",
        "source_status": "AVAILABLE" if not file_errors else SOURCE_UNCERTAIN,
        "source_root": str(root),
        "native_nwb_file_count": len(files),
        "native_nwb_files": _nwb_file_records(files, root),
        "trial_row_count": trial_row_count,
        "identities": [
            {
                "identity": "optogenetic_stimulation_power",
                "rho_field": "photo_stim_power",
                "rho_unit": "mW" if unit_known else None,
                "levels": power_levels,
                "level_counts": {
                    str(level): int(power_counts[level]) for level in power_levels
                },
                "rho_zero_or_sham": 0.0 in power_levels,
                "onset_field": None,
                "offset_field": None,
                "timing_complete": False,
                "classification": SOURCE_UNCERTAIN,
                "conditional_classification_if_trial_window_accepted": (
                    POINT_OR_LIMITED_CANDIDATE
                ),
            }
        ],
        "rho_field": "photo_stim_power",
        "rho_unit": "mW" if unit_known else None,
        "rho_levels": power_levels,
        "rho_level_counts": {
            str(level): int(power_counts[level]) for level in power_levels
        },
        "rho_zero_or_sham": 0.0 in power_levels,
        "stim_present_counts": dict(stim_present_counts),
        "photo_stim_period_values": _text_set(periods),
        "photo_stim_type_values": _text_set(stimulus_types),
        "trial_window_timing_rows": timing_rows,
        "pulse_onset_offset_fields": [],
        "context_group_names": {
            group_path: sorted(names)
            for group_path, names in sorted(context_names.items())
        },
        "acquisition_group_empty": not context_names["/acquisition"],
        "conditional_classification_if_trial_window_accepted": (
            POINT_OR_LIMITED_CANDIDATE
        ),
        "file_errors": file_errors,
        "classification": SOURCE_UNCERTAIN,
        "classification_reason": (
            "Power and mW units are explicit, with a zero-power level, but "
            "native photostimulation pulse onset/offset fields are absent. "
            "Trial start/stop is not silently substituted for pulse timing."
        ),
        "audit_scope": {
            "opened_paths": [
                "/intervals/trials",
                "/intervals",
                "/stimulus",
                "/general/optogenetics",
                "/acquisition",
            ],
            "forbidden_paths_not_opened": [
                "/acquisition/*/data",
                "/units/spike_times",
            ],
        },
    }


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return [
            {str(key): str(value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _metadata_hashes(paths: list[Path], root: Path) -> list[dict[str, Any]]:
    return [_metadata_record(path, root) for path in paths if path.is_file()]


def _session_names(root: Path) -> list[str]:
    names: set[str] = set()
    for directory, subdirectories, _filenames in os.walk(root):
        subdirectories[:] = [
            name for name in subdirectories if name not in EXCLUDED_DIRECTORIES
        ]
        names.update(name for name in subdirectories if name.startswith("ses-"))
        directory_name = Path(directory).name
        if directory_name.startswith("ses-"):
            names.add(directory_name)
    return sorted(names)


def _audit_ds006623(root: Path) -> dict[str, Any]:
    required = [
        root / "README.md",
        root / "dataset_description.json",
        root / "Participant_Info.csv",
        root / "LOR_ROR_Timing.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {
            "dataset_id": "ds006623",
            "source_status": SOURCE_UNCERTAIN,
            "source_root": str(root),
            "classification": SOURCE_UNCERTAIN,
            "missing_files": missing,
        }
    participant_rows = _read_csv_rows(root / "Participant_Info.csv")
    timing_rows = _read_csv_rows(root / "LOR_ROR_Timing.csv")
    bold_jsons = _walk_files(root, suffixes={".json"})
    event_files = [path for path in _walk_files(root) if path.name.endswith("_events.tsv")]
    bold_metadata = []
    repetition_times: set[float] = set()
    for path in bold_jsons:
        if not path.name.endswith("_bold.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and _finite_float(payload.get("RepetitionTime")):
                repetition_times.add(float(payload["RepetitionTime"]))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        bold_metadata.append(path)
    info_fields = (
        sorted(
            {
                str(field)
                for field in participant_rows[0].keys()
                if str(field).strip()
            }
        )
        if participant_rows
        else []
    )
    candidate_fields = [
        field
        for field in ("Infusion Protocol", "LOR ESC", "Propofol dosage", "Infusion Stop")
        if field in info_fields
    ]
    unit_tokens_in_values = _unit_tokens(
        [
            row.get(field, "")
            for row in participant_rows
            for field in candidate_fields
        ]
    )
    return {
        "dataset_id": "ds006623",
        "source_status": "AVAILABLE",
        "source_root": str(root),
        "participant_count": sum(bool(row.get("Subjects")) for row in participant_rows),
        "participant_info_fields": info_fields,
        "candidate_rho_fields": candidate_fields,
        "candidate_rho_units": [],
        "unit_tokens_in_free_text_values": unit_tokens_in_values,
        "timed_dose_series_present": False,
        "behavioral_timing_row_count": len(timing_rows),
        "bold_metadata_file_count": len(bold_metadata),
        "bold_repetition_times_sec": sorted(repetition_times),
        "event_file_count": len(event_files),
        "effect_site_concentration_text_claim": (
            "effect-site concentrations"
            in (root / "README.md").read_text(encoding="utf-8", errors="replace").lower()
        ),
        "rho_known": False,
        "v_definition": "graded_propofol_sedation",
        "identities": [
            {
                "identity": "propofol_sedation",
                "rho_field": None,
                "rho_unit": None,
                "levels": [],
                "direction_or_condition_levels": ["graded_propofol_sedation"],
                "timing_complete": False,
                "classification": SOURCE_UNCERTAIN,
            }
        ],
        "classification": SOURCE_UNCERTAIN,
        "classification_reason": (
            "Numeric-looking propofol protocol fields lack explicit physical "
            "units and a timed administration/effect-site series. LOR/ROR "
            "timing is behavioral timing, not perturbation amplitude timing."
        ),
        "metadata_hashes": _metadata_hashes(
            [
                root / "README.md",
                root / "dataset_description.json",
                root / "Participant_Info.csv",
                root / "LOR_ROR_Timing.csv",
                *bold_metadata,
            ],
            root,
        ),
        "audit_scope": {
            "opened_file_classes": ["README", "JSON", "CSV"],
            "forbidden_file_classes_not_opened": ["NIfTI", "physio_time_series"],
        },
    }


def _audit_ds005917(root: Path) -> dict[str, Any]:
    required = [
        root / "README",
        root / "dataset_description.json",
        root / "participants.tsv",
        root / "phenotype" / "phenotype.tsv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {
            "dataset_id": "ds005917",
            "source_status": SOURCE_UNCERTAIN,
            "source_root": str(root),
            "classification": SOURCE_UNCERTAIN,
            "missing_files": missing,
        }
    with (root / "participants.tsv").open(
        "r", encoding="utf-8", errors="replace", newline=""
    ) as handle:
        participant_rows = list(csv.DictReader(handle, delimiter="\t"))
    infusion_values = sorted(
        {
            str(value).strip()
            for row in participant_rows
            for key, value in row.items()
            if key in {"infusion_1", "infusion_2"}
            and str(value).strip()
            and str(value).strip().lower() not in {"n/a", "na"}
        }
    )
    session_names = _session_names(root)
    bold_jsons = [
        path
        for path in _walk_files(root, suffixes={".json"})
        if path.name.endswith("_bold.json")
    ]
    physio_jsons = [
        path
        for path in _walk_files(root, suffixes={".json"})
        if path.name.endswith("_physio.json")
    ]
    event_files = [
        path for path in _walk_files(root) if path.name.endswith("_events.tsv")
    ]
    return {
        "dataset_id": "ds005917",
        "source_status": "AVAILABLE",
        "source_root": str(root),
        "participant_count": len(participant_rows),
        "infusion_labels": infusion_values,
        "session_names": session_names,
        "placebo_sessions_present": any("ses-p" in name for name in session_names),
        "baseline_sessions_present": any("ses-b0" in name for name in session_names),
        "bold_metadata_file_count": len(bold_jsons),
        "physio_metadata_file_count": len(physio_jsons),
        "event_file_count": len(event_files),
        "rho_known": False,
        "rho_units": [],
        "timed_dose_series_present": False,
        "v_definition": "ketamine_vs_placebo_session_type",
        "identities": [
            {
                "identity": "ketamine_placebo_session_type",
                "rho_field": None,
                "rho_unit": None,
                "levels": [],
                "direction_or_condition_levels": infusion_values,
                "timing_complete": False,
                "classification": DIRECTION_ONLY,
            }
        ],
        "classification": DIRECTION_ONLY,
        "classification_reason": (
            "Ketamine/placebo and d2/d10/p2/p10 session labels define "
            "perturbation type and schedule, not amplitude. No dose field "
            "with units and timing was found."
        ),
        "metadata_hashes": _metadata_hashes(
            [
                root / "README",
                root / "dataset_description.json",
                root / "participants.tsv",
                root / "phenotype" / "phenotype.tsv",
                *bold_jsons,
                *physio_jsons,
            ],
            root,
        ),
        "audit_scope": {
            "opened_file_classes": ["README", "JSON", "TSV"],
            "forbidden_file_classes_not_opened": [
                "NIfTI",
                "physio_time_series",
            ],
        },
    }


def _unavailable(
    dataset_id: str,
    root: Path,
    reason: str,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "source_status": SOURCE_UNCERTAIN,
        "source_root": str(root),
        "classification": SOURCE_UNCERTAIN,
        "failure_reason": reason,
    }


def _default_roots() -> dict[str, Path]:
    return {
        "dandi_000458": Path(
            r"K:\ExternalReceivedDatasets\DANDI\dandi_000458\raw\000458"
        ),
        "dandi_000009": Path(
            r"H:\SourceRepo2\NeuralManifoldDynamics\data\dandi\raw\000009"
        ),
        "ds006623": Path(r"M:\datasets\received\openneuro\ds006623"),
        "ds005917": Path(r"M:\datasets\received\openneuro\ds005917"),
    }


def run_inventory(
    *,
    source_roots: Mapping[str, Path] | None = None,
    protocol_path: Path,
) -> dict[str, Any]:
    roots = _default_roots()
    roots.update(source_roots or {})
    audits: list[dict[str, Any]] = []
    for dataset_id, root in roots.items():
        if not root.is_dir():
            audits.append(_unavailable(dataset_id, root, "source_root_missing"))
            continue
        if dataset_id == "dandi_000458":
            audits.append(_audit_dandi_000458(root))
        elif dataset_id == "dandi_000009":
            audits.append(_audit_dandi_000009(root))
        elif dataset_id == "ds006623":
            audits.append(_audit_ds006623(root))
        elif dataset_id == "ds005917":
            audits.append(_audit_ds005917(root))
    identity_classifications = [
        str(identity.get("classification"))
        for audit in audits
        for identity in audit.get("identities", [])
        if identity.get("classification")
    ]
    dataset_classifications = [str(audit.get("classification")) for audit in audits]
    eligible = [
        {
            "dataset_id": audit["dataset_id"],
            "identity": identity.get("identity"),
            "classification": identity.get("classification"),
        }
        for audit in audits
        for identity in audit.get("identities", [])
        if identity.get("classification")
        in {CURVE_CANDIDATE, POINT_OR_LIMITED_CANDIDATE}
    ]
    gate_status = "PASS" if eligible else "NOT_TESTABLE"
    return {
        "schema": "mndm.far_001b_source_metadata.v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": sha256_file(protocol_path),
        "gate_status": gate_status,
        "gate_reason": (
            "explicit_physical_perturbation_candidate_present"
            if eligible
            else "no_explicit_physical_perturbation_candidate"
        ),
        "datasets": audits,
        "classification_counts": {
            classification: identity_classifications.count(classification)
            for classification in sorted(set(identity_classifications))
        },
        "dataset_classification_counts": {
            classification: dataset_classifications.count(classification)
            for classification in sorted(set(dataset_classifications))
        },
        "eligible_candidates": eligible,
        "far_002_authorized": bool(eligible),
        "next_gate": "FAR-002" if eligible else "external_perturbation_source_scout",
        "fail_closed_assertions": {
            "signal_payloads_opened": False,
            "nmd_outputs_read": False,
            "nifti_payloads_opened": False,
            "physio_time_series_opened": False,
            "spike_times_opened": False,
            "dose_reconstructed": False,
            "condition_promoted_to_rho": False,
            "frequency_promoted_to_rho": False,
            "neural_outcomes_analyzed": False,
        },
        "claim_boundary": (
            "FAR-001B establishes only whether native source metadata contains "
            "explicit physical perturbation magnitude, units, and timing. It "
            "does not define home/away, estimate resilience, or perform "
            "outcome analysis."
        ),
    }
