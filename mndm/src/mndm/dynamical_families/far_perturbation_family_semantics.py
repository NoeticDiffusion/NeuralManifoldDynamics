"""FAR-002 metadata-only perturbation-family semantics audit.

This module reads DANDI 000458 NWB metadata and trial annotations only.  It
never reads acquisition data, spike times, waveforms, or any NMD-derived
output.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .far_source_metadata_audit import (
    _decode,
    _finite_float,
    _nwb_file_records,
    _nwb_group_names,
    _nwb_trial_table,
    _numeric_text,
    _truthy,
    _walk_files,
)


PROTOCOL_ID = "FAR-002"
CURVE_SEMANTICS_PASS = "CURVE_SEMANTICS_PASS"
POINT_SEMANTICS_PASS = "POINT_SEMANTICS_PASS"
DIRECTION_ONLY = "DIRECTION_ONLY"
AMPLITUDE_CONFOUNDED = "AMPLITUDE_CONFOUNDED"
TIMING_UNRESOLVED = "TIMING_UNRESOLVED"
STATE_CONFOUNDED = "STATE_CONFOUNDED"
INSUFFICIENT_LEVELS = "INSUFFICIENT_LEVELS"
SOURCE_UNCERTAIN = "SOURCE_UNCERTAIN"

MAX_SINGLE_PULSE_DURATION_SEC = 0.1
SHORT_PULSE_DURATION_BOUNDARY_SEC = 0.002
MIN_BIOLOGICAL_UNITS = 2
OPERATING_STATES = {"awake", "isoflurane", "recovery"}
V_FIELDS = (
    "stimulus_type",
    "stimulus_description",
    "estim_target_region",
    "estim_target_depth",
    "observed_stim_duration_class",
)
OPTIONAL_PHYSICAL_V_FIELDS = (
    "polarity",
    "pulse_width",
    "pulse_frequency",
    "n_pulses",
    "train_duration",
    "waveform",
)
TRIAL_COLUMNS = [
    "behavioral_epoch",
    "estim_current",
    "estim_target_depth",
    "estim_target_region",
    "is_valid",
    "start_time",
    "stimulus_description",
    "stimulus_type",
    "stop_time",
    *OPTIONAL_PHYSICAL_V_FIELDS,
]
REQUIRED_TRIAL_COLUMNS = [
    "behavioral_epoch",
    "estim_current",
    "estim_target_depth",
    "estim_target_region",
    "is_valid",
    "start_time",
    "stimulus_description",
    "stimulus_type",
    "stop_time",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar_text(handle: Any, path: str) -> str:
    obj = handle.get(path)
    if obj is None or not hasattr(obj, "shape") or obj.shape != ():
        return ""
    try:
        value = _decode(obj[()])
    except Exception:
        return ""
    return str(value).strip()


def _protocol_metadata(path: Path) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("h5py_required_for_far_002_audit") from error
    with h5py.File(path, "r") as handle:
        group_names = _nwb_group_names(
            path,
            [
                "/intervals",
                "/stimulus",
                "/stimulus/presentation",
                "/stimulus/templates",
                "/general/extracellular_ephys",
                "/acquisition",
            ],
        )
        series = handle.get("/acquisition/ElectricalSeriesEEG")
        series_metadata = {}
        if series is not None and hasattr(series, "attrs"):
            series_metadata = {
                str(key): str(_decode(value))
                for key, value in series.attrs.items()
                if str(key) in {"description", "comments", "neurodata_type"}
            }
        return {
            "session_description": _scalar_text(handle, "/session_description"),
            "experiment_description": _scalar_text(
                handle, "/general/experiment_description"
            ),
            "stimulus_protocol": _scalar_text(handle, "/general/stimulus"),
            "group_names": group_names,
            "electrical_series_metadata": series_metadata,
        }


def _session_id(path: Path) -> str:
    match = re.search(r"(ses-[^_]+)", path.name)
    return match.group(1) if match else path.stem


def _subject_id(path: Path) -> str:
    if path.parent.name.startswith("sub-"):
        return path.parent.name
    match = re.search(r"(sub-[^_]+)", path.name)
    return match.group(1) if match else path.parent.name


def _clean_text(value: Any) -> str:
    text = str(_decode(value)).strip()
    return text if text and text.lower() not in {"n/a", "na", "nan", "none"} else ""


def _v_key(values: dict[str, str]) -> str:
    extra_fields = sorted(set(values) - set(V_FIELDS))
    ordered_fields = (*V_FIELDS, *extra_fields)
    return "|".join(f"{field}={values[field]}" for field in ordered_fields)


def _family_id(v_key: str) -> str:
    digest = hashlib.sha1(v_key.encode("utf-8")).hexdigest()[:12]
    return f"family-{digest}"


def _zero_control_type(levels: set[float]) -> str:
    if 0.0 in levels:
        return "TRUE_ZERO_AMPLITUDE"
    return "NO_ZERO_CONTROL"


def _timing_semantics(
    *,
    protocol_text: str,
    durations: list[float],
    starts_complete: bool,
) -> tuple[str, str]:
    protocol_lower = protocol_text.lower()
    single_pulse_protocol = (
        "single pulse" in protocol_lower and "electrical" in protocol_lower
    )
    duration_complete = bool(durations) and all(
        0.0 <= duration <= MAX_SINGLE_PULSE_DURATION_SEC for duration in durations
    )
    if starts_complete and single_pulse_protocol and duration_complete:
        return (
            "TRUE_STIMULATION_EVENT",
            "TRUE_STIMULATION_EVENT",
        )
    return ("TIMING_UNRESOLVED", "TIMING_UNRESOLVED")


def _duration_class(duration: float | None) -> str:
    if duration is None or duration < 0:
        return "unresolved"
    if duration <= SHORT_PULSE_DURATION_BOUNDARY_SEC:
        return "sub_2ms"
    if duration <= MAX_SINGLE_PULSE_DURATION_SEC:
        return "2_to_100ms"
    return "broad_interval"


def _family_status(
    *,
    levels: set[float],
    subject_count: int,
    timing_status: str,
    negative_rho: bool,
    state_known: bool,
) -> tuple[str, str | None]:
    if timing_status != "TRUE_STIMULATION_EVENT":
        return TIMING_UNRESOLVED, "physical_stimulation_timing_unresolved"
    if not state_known:
        return STATE_CONFOUNDED, "unknown_operating_state"
    if negative_rho:
        return SOURCE_UNCERTAIN, "negative_current_requires_signedness_and_polarity_rule"
    nonzero_levels = {level for level in levels if level != 0.0}
    if subject_count < MIN_BIOLOGICAL_UNITS:
        if len(nonzero_levels) >= 1:
            return INSUFFICIENT_LEVELS, "fewer_than_two_subjects"
        return SOURCE_UNCERTAIN, "fewer_than_two_subjects_and_no_nonzero_rho"
    if len(nonzero_levels) >= 3:
        return CURVE_SEMANTICS_PASS, None
    if len(nonzero_levels) in {1, 2}:
        return POINT_SEMANTICS_PASS, "fewer_than_three_nonzero_rho_levels"
    return SOURCE_UNCERTAIN, "no_nonzero_rho_levels"


def audit_dandi_000458(
    *,
    source_root: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    files = _walk_files(source_root, suffixes={".nwb"})
    if not files:
        return {
            "dataset_id": "dandi_000458",
            "source_status": "SOURCE_UNCERTAIN",
            "classification": "BLOCKED",
            "failure_reason": "no_native_nwb_files",
            "source_root": str(source_root),
        }

    family_rows: dict[tuple[str, str], dict[str, Any]] = {}
    file_errors: list[dict[str, Any]] = []
    source_subjects: set[str] = set()
    source_sessions: set[str] = set()
    source_protocol_texts: set[str] = set()
    source_group_names: dict[str, set[str]] = defaultdict(set)
    source_series_metadata: dict[str, str] = {}
    available_trial_columns: set[str] = set()
    unresolved_v_fields: set[str] = set()
    current_values_all: list[float] = []

    for path in files:
        try:
            table, _descriptions, missing = _nwb_trial_table(path, TRIAL_COLUMNS)
            required_missing = sorted(set(REQUIRED_TRIAL_COLUMNS) & set(missing))
            if required_missing:
                file_errors.append(
                    {
                        "relative_path": str(path.relative_to(source_root)).replace(
                            "\\", "/"
                        ),
                        "missing_required_columns": required_missing,
                    }
                )
                continue
            available_trial_columns.update(
                column for column in TRIAL_COLUMNS if column not in missing
            )
            protocol_metadata = _protocol_metadata(path)
            protocol_text = protocol_metadata["stimulus_protocol"]
            source_protocol_texts.add(protocol_text)
            for group_path, names in protocol_metadata["group_names"].items():
                source_group_names[group_path].update(names)
            source_series_metadata.update(protocol_metadata["electrical_series_metadata"])
            subject = _subject_id(path)
            session = _session_id(path)
            source_subjects.add(subject)
            source_sessions.add(session)
            n_rows = int(table["_row_count"][0])
            validity = table["is_valid"]
            for index in range(n_rows):
                stimulus_type = _clean_text(table["stimulus_type"][index]).lower()
                if stimulus_type != "electrical":
                    continue
                valid = _truthy(validity[index])
                state = _clean_text(table["behavioral_epoch"][index]).lower()
                current = _numeric_text(table["estim_current"][index])
                if current is None:
                    continue
                current_values_all.append(current)
                start = _finite_float(table["start_time"][index])
                stop = _finite_float(table["stop_time"][index])
                duration = (
                    stop - start
                    if start is not None and stop is not None
                    else None
                )
                v_values = {
                    field: _clean_text(table[field][index])
                    for field in V_FIELDS
                    if field in table
                }
                v_values["observed_stim_duration_class"] = _duration_class(duration)
                for field in V_FIELDS:
                    v_values.setdefault(field, "")
                for field in OPTIONAL_PHYSICAL_V_FIELDS:
                    if field in table:
                        v_values[field] = _clean_text(table[field][index])
                if any(not value for value in v_values.values()):
                    unresolved_v_fields.update(
                        field for field, value in v_values.items() if not value
                    )
                for field in OPTIONAL_PHYSICAL_V_FIELDS:
                    if field not in available_trial_columns:
                        unresolved_v_fields.add(field)
                key = _v_key(v_values)
                family = _family_id(key)
                row_key = (family, state)
                row = family_rows.setdefault(
                    row_key,
                    {
                        "family_id": family,
                        "operating_state_w": state,
                        "perturbation_identity_v": v_values,
                        "v_key": key,
                        "rho_values": [],
                        "rho_counts": Counter(),
                        "rho_subjects": defaultdict(set),
                        "rho_sessions": defaultdict(set),
                        "durations": [],
                        "subjects": set(),
                        "sessions": set(),
                        "n_trials_total": 0,
                        "n_trials_valid": 0,
                        "invalid_trials": 0,
                        "starts_complete": True,
                        "stimulus_descriptions": set(),
                        "protocol_texts": set(),
                        "file_paths": set(),
                    },
                )
                row["n_trials_total"] += 1
                row["stimulus_descriptions"].add(
                    _clean_text(table["stimulus_description"][index])
                )
                row["protocol_texts"].add(protocol_text)
                row["file_paths"].add(
                    str(path.relative_to(source_root)).replace("\\", "/")
                )
                if not valid:
                    row["invalid_trials"] += 1
                    continue
                row["n_trials_valid"] += 1
                row["rho_values"].append(current)
                row["rho_counts"][current] += 1
                row["subjects"].add(subject)
                row["sessions"].add(session)
                row["rho_subjects"][current].add(subject)
                row["rho_sessions"][current].add(session)
                if start is None or stop is None or stop < start:
                    row["starts_complete"] = False
                else:
                    row["durations"].append(duration)
        except (OSError, ValueError, KeyError, TypeError) as error:
            file_errors.append(
                {
                    "relative_path": str(path.relative_to(source_root)).replace(
                        "\\", "/"
                    ),
                    "error": f"{type(error).__name__}:{error}",
                }
            )

    ledger: list[dict[str, Any]] = []
    for row_key in sorted(family_rows):
        row = family_rows[row_key]
        levels = set(row["rho_values"])
        onset_status, offset_status = _timing_semantics(
            protocol_text=" ".join(sorted(row["protocol_texts"])),
            durations=row["durations"],
            starts_complete=row["starts_complete"],
        )
        status, exclusion_reason = _family_status(
            levels=levels,
            subject_count=len(row["subjects"]),
            timing_status=onset_status,
            negative_rho=any(value < 0 for value in row["rho_values"]),
            state_known=row["operating_state_w"] in OPERATING_STATES,
        )
        rho_level_ledger = [
            {
                "rho": level,
                "n_trials_valid": int(row["rho_counts"][level]),
                "n_subjects": len(row["rho_subjects"][level]),
                "n_sessions": len(row["rho_sessions"][level]),
            }
            for level in sorted(levels)
        ]
        ledger.append(
            {
                "family_id": row["family_id"],
                "operating_state_w": row["operating_state_w"],
                "perturbation_identity_v": row["perturbation_identity_v"],
                "v_key": row["v_key"],
                "rho_field": "estim_current",
                "rho_units": "μA",
                "rho_signed": any(value < 0 for value in row["rho_values"]),
                "rho_levels": sorted(levels),
                "rho_level_ledger": rho_level_ledger,
                "n_nonzero_levels": len(
                    {value for value in levels if value != 0.0}
                ),
                "zero_control_type": _zero_control_type(levels),
                "n_trials_total": row["n_trials_total"],
                "n_trials_valid": row["n_trials_valid"],
                "n_trials_invalid": row["invalid_trials"],
                "n_subjects": len(row["subjects"]),
                "subjects": sorted(row["subjects"]),
                "n_sessions": len(row["sessions"]),
                "sessions": sorted(row["sessions"]),
                "stim_duration_min_sec": (
                    min(row["durations"]) if row["durations"] else None
                ),
                "stim_duration_max_sec": (
                    max(row["durations"]) if row["durations"] else None
                ),
                "stim_onset_semantics": onset_status,
                "stim_offset_semantics": offset_status,
                "site": row["perturbation_identity_v"].get("estim_target_region"),
                "polarity": row["perturbation_identity_v"].get("polarity"),
                "pulse_width": row["perturbation_identity_v"].get("pulse_width"),
                "pulse_frequency": row["perturbation_identity_v"].get(
                    "pulse_frequency"
                ),
                "train_duration": row["perturbation_identity_v"].get(
                    "train_duration"
                ),
                "other_fixed_parameters": {
                    "stimulus_type": row["perturbation_identity_v"].get(
                        "stimulus_type"
                    ),
                    "stimulus_description": row["perturbation_identity_v"].get(
                        "stimulus_description"
                    ),
                    "estim_target_depth": row["perturbation_identity_v"].get(
                        "estim_target_depth"
                    ),
                    "optional_physical_v_fields": {
                        field: row["perturbation_identity_v"].get(field)
                        for field in OPTIONAL_PHYSICAL_V_FIELDS
                        if field in row["perturbation_identity_v"]
                    },
                    "protocol_text": sorted(row["protocol_texts"]),
                },
                "unresolved_v_fields": sorted(unresolved_v_fields),
                "current_min_ua": min(levels) if levels else None,
                "current_max_ua": max(levels) if levels else None,
                "curve_status": status,
                "exclusion_reason": exclusion_reason,
            }
        )

    statuses = [str(row["curve_status"]) for row in ledger]
    if CURVE_SEMANTICS_PASS in statuses:
        global_status = "PASS"
        global_reason = "coherent_curve_family_present"
    elif POINT_SEMANTICS_PASS in statuses:
        global_status = "LIMITED_PASS"
        global_reason = "only_coherent_point_or_limited_families_present"
    elif statuses:
        global_status = "NOT_TESTABLE"
        global_reason = "no_coherent_perturbation_family"
    else:
        global_status = "BLOCKED"
        global_reason = "no_electrical_trial_family_rows"

    return {
        "schema": "mndm.far_002_perturbation_family_ledger.v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": sha256_file(protocol_path),
        "dataset_id": "dandi_000458",
        "source_status": "AVAILABLE" if not file_errors else SOURCE_UNCERTAIN,
        "source_root": str(source_root),
        "native_nwb_file_count": len(files),
        "native_nwb_files": _nwb_file_records(files, source_root),
        "source_subject_count": len(source_subjects),
        "source_session_count": len(source_sessions),
        "available_trial_columns": sorted(available_trial_columns),
        "unresolved_v_fields": sorted(unresolved_v_fields),
        "rho_observed_nonnegative": bool(current_values_all) and all(
            value >= 0 for value in current_values_all
        ),
        "rho_negative_value_count": sum(
            value < 0 for value in current_values_all
        ),
        "rho_transform": "identity_native_value",
        "rho_command_vs_estimate": "SOURCE_UNCERTAIN",
        "protocol_metadata": {
            "stimulus_protocol_texts": sorted(source_protocol_texts),
            "group_names": {
                group_path: sorted(names)
                for group_path, names in sorted(source_group_names.items())
            },
            "electrical_series_metadata": source_series_metadata,
        },
        "perturbation_family_ledger": ledger,
        "global_status": global_status,
        "global_reason": global_reason,
        "file_errors": file_errors,
        "audit_scope": {
            "opened_paths": [
                "/intervals/trials",
                "/session_description",
                "/general/experiment_description",
                "/general/stimulus",
                "/general/extracellular_ephys",
                "/acquisition (group names and series attributes only)",
            ],
            "forbidden_paths_not_opened": [
                "/acquisition/*/data",
                "/units/spike_times",
                "/processing",
                "all NMD outputs",
            ],
        },
        "fail_closed_assertions": {
            "signal_payloads_opened": False,
            "spike_times_opened": False,
            "waveforms_opened": False,
            "nmd_outputs_read": False,
            "outcomes_constructed": False,
            "home_away_constructed": False,
            "rho_abs_transform_applied": False,
            "states_pooled": False,
            "ordinary_baseline_promoted_to_zero": False,
        },
        "claim_boundary": (
            "FAR-002 establishes only native perturbation-family semantics. "
            "It does not establish signal coverage, home/away regions, "
            "resilience, or a biological dose-response."
        ),
    }
