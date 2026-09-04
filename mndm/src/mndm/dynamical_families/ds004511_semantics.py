"""Read-only source-semantics audit for OD-EMP-DS004511-SEM-000."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "mndm.od_emp_ds004511_semantics.v1"
PROTOCOL_ID = "OD-EMP-DS004511-SEM-000"
DATASET_ID = "ds004511"
MAX_UNIQUE_VALUES = 12
CANONICAL_PARTICIPANT = "sub-S200116"

REQUIRED_VARIABLES = (
    "index",
    "Participant_ID",
    "Trial_Probability",
    "Question_Position_Prediction_Correct",
    "Question_Position_Prediction_Agent",
    "Question_Position_Comparison_Self",
    "Question_Position_Comparison_Irrelevant",
    "Trial_Stake",
    "Trial_Balance",
    "Trial_Balance_Effect",
    "Question_Comparison_Condition",
    "Question_Comparison_Type",
    "Question_Comparison_Question",
    "Question_Comparison_Label",
    "Question_Comparison_Item",
    "Question_Comparison_Proportion_Yes",
    "Question_Factual_Condition",
    "Question_Factual_Item",
    "Question_Factual_Question",
    "Question_Factual_Label",
    "Question_Factual_Proportion_DontKnow",
    "Question_Factual_Type",
    "Prediction_Result",
    "Answer_Prediction_Correct",
    "Answer_Prediction_Agent",
    "Answer_Irrelevant",
    "Answer_Irrelevant_RT",
    "Answer_Self",
    "Answer_Self_RT",
    "Answer_Prediction_Correct_RT",
    "Answer_Prediction_Agent_RT",
    "Trial_Condition",
    "Trial_Type",
    "Trial_ProbableSide",
    "Prediction_Choice",
    "Prediction_RT",
    "Trial_Order",
    "Trial_Datetime",
    "Trial_Stake_Jitter",
    "Trial_Type_Jitter",
    "Trial_Result_Jitter",
    "Balance_Update_Jitter",
)

VARIABLE_CLASSIFICATIONS: dict[str, dict[str, str]] = {
    "index": {
        "dynamicality": "TRIAL_CONSTANT",
        "semantic_role": "source row identifier",
    },
    "Participant_ID": {
        "dynamicality": "TRIAL_CONSTANT",
        "semantic_role": "participant identifier",
    },
    "Trial_Probability": {
        "dynamicality": "TRIAL_CONSTANT",
        "semantic_role": "experiment-supplied probability parameter",
    },
    "Question_Position_Prediction_Correct": {
        "dynamicality": "TRIAL_CONSTANT",
        "semantic_role": "post-result question order parameter",
    },
    "Question_Position_Prediction_Agent": {
        "dynamicality": "TRIAL_CONSTANT",
        "semantic_role": "post-trial question order parameter",
    },
    "Question_Position_Comparison_Self": {
        "dynamicality": "TRIAL_CONSTANT",
        "semantic_role": "cover-question order parameter",
    },
    "Question_Position_Comparison_Irrelevant": {
        "dynamicality": "TRIAL_CONSTANT",
        "semantic_role": "cover-question order parameter",
    },
    "Trial_Stake": {
        "dynamicality": "TRIAL_CONSTANT",
        "semantic_role": "bet size / payoff magnitude",
    },
    "Trial_Balance": {
        "dynamicality": "BETWEEN_TRIAL_ONLY",
        "semantic_role": "post-round displayed balance",
    },
    "Trial_Balance_Effect": {
        "dynamicality": "TRIAL_CONSTANT",
        "semantic_role": "stored balance-effect magnitude; compare with stake",
    },
    "Prediction_Result": {
        "dynamicality": "TERMINAL_ONLY",
        "semantic_role": "single die-face result",
    },
    "Answer_Prediction_Correct": {
        "dynamicality": "TERMINAL_ONLY",
        "semantic_role": (
            "post-result participant report; exact left/right codebook "
            "unresolved"
        ),
    },
    "Answer_Prediction_Agent": {
        "dynamicality": "TERMINAL_ONLY",
        "semantic_role": (
            "post-trial agent report; exact left/right codebook unresolved"
        ),
    },
    "Answer_Prediction_Correct_RT": {
        "dynamicality": "TERMINAL_ONLY",
        "semantic_role": "post-result report latency",
    },
    "Answer_Prediction_Agent_RT": {
        "dynamicality": "TERMINAL_ONLY",
        "semantic_role": "post-trial report latency",
    },
    "Answer_Irrelevant": {
        "dynamicality": "TERMINAL_ONLY",
        "semantic_role": "cover-question response",
    },
    "Answer_Irrelevant_RT": {
        "dynamicality": "TERMINAL_ONLY",
        "semantic_role": "cover-question response latency",
    },
    "Answer_Self": {
        "dynamicality": "TERMINAL_ONLY",
        "semantic_role": "cover-question response",
    },
    "Answer_Self_RT": {
        "dynamicality": "TERMINAL_ONLY",
        "semantic_role": "cover-question response latency",
    },
    "Trial_Condition": {
        "dynamicality": "UNKNOWN",
        "semantic_role": (
            "condition label; exact source derivation is unresolved"
        ),
    },
    "Trial_Type": {
        "dynamicality": "TRIAL_CONSTANT",
        "semantic_role": "Self versus Computer trial factor",
    },
    "Trial_ProbableSide": {
        "dynamicality": "TRIAL_CONSTANT",
        "semantic_role": "trial-supplied probable-side label",
    },
    "Prediction_Choice": {
        "dynamicality": "TERMINAL_ONLY",
        "semantic_role": "single prediction choice",
    },
    "Prediction_RT": {
        "dynamicality": "TERMINAL_ONLY",
        "semantic_role": "single response latency",
    },
    "Trial_Datetime": {
        "dynamicality": "BETWEEN_TRIAL_ONLY",
        "semantic_role": "one round timestamp",
    },
    "Trial_Order": {
        "dynamicality": "TRIAL_CONSTANT",
        "semantic_role": "within-session round order",
    },
    "Question_Comparison_Condition": {
        "dynamicality": "TRIAL_CONSTANT",
        "semantic_role": "cover-question condition parameter",
    },
    "Question_Comparison_Type": {
        "dynamicality": "TRIAL_CONSTANT",
        "semantic_role": "cover-question type parameter",
    },
    "Question_Comparison_Question": {
        "dynamicality": "TRIAL_CONSTANT",
        "semantic_role": "cover-question prompt",
    },
    "Question_Comparison_Label": {
        "dynamicality": "TRIAL_CONSTANT",
        "semantic_role": "cover-question label",
    },
    "Question_Comparison_Item": {
        "dynamicality": "TRIAL_CONSTANT",
        "semantic_role": "cover-question item identifier",
    },
    "Question_Comparison_Proportion_Yes": {
        "dynamicality": "TRIAL_CONSTANT",
        "semantic_role": "source-supplied question proportion",
    },
    "Question_Factual_Condition": {
        "dynamicality": "TRIAL_CONSTANT",
        "semantic_role": "cover-question condition parameter",
    },
    "Question_Factual_Item": {
        "dynamicality": "TRIAL_CONSTANT",
        "semantic_role": "cover-question item identifier",
    },
    "Question_Factual_Question": {
        "dynamicality": "TRIAL_CONSTANT",
        "semantic_role": "cover-question prompt",
    },
    "Question_Factual_Label": {
        "dynamicality": "TRIAL_CONSTANT",
        "semantic_role": "cover-question label",
    },
    "Question_Factual_Proportion_DontKnow": {
        "dynamicality": "TRIAL_CONSTANT",
        "semantic_role": "source-supplied question proportion",
    },
    "Question_Factual_Type": {
        "dynamicality": "TRIAL_CONSTANT",
        "semantic_role": "cover-question type parameter",
    },
    "Trial_Stake_Jitter": {
        "dynamicality": "TRIAL_CONSTANT",
        "semantic_role": "display-duration parameter",
    },
    "Trial_Type_Jitter": {
        "dynamicality": "TRIAL_CONSTANT",
        "semantic_role": "display-duration parameter",
    },
    "Trial_Result_Jitter": {
        "dynamicality": "TRIAL_CONSTANT",
        "semantic_role": "display-duration parameter",
    },
    "Balance_Update_Jitter": {
        "dynamicality": "TRIAL_CONSTANT",
        "semantic_role": "display-duration parameter",
    },
}

TERMINAL_CANDIDATES = (
    {
        "name": "prediction_left_right",
        "source_fields": ["Prediction_Choice"],
        "timestamp_fields": ["Prediction_RT", "Trial_Datetime"],
        "classification": "TERMINAL_ONLY",
    },
    {
        "name": "post_result_report",
        "source_fields": ["Answer_Prediction_Correct"],
        "timestamp_fields": ["Answer_Prediction_Correct_RT", "Trial_Datetime"],
        "classification": "TERMINAL_ONLY",
        "semantic_status": "SEMANTICS_UNRESOLVED",
    },
    {
        "name": "result_or_condition_label",
        "source_fields": ["Prediction_Result", "Trial_Condition"],
        "timestamp_fields": ["Trial_Datetime"],
        "classification": "TERMINAL_ONLY",
        "semantic_status": "SEMANTICS_UNRESOLVED",
    },
    {
        "name": "gain_loss",
        "source_fields": ["Trial_Balance", "Trial_Stake"],
        "timestamp_fields": ["Trial_Datetime", "Balance_Update_Jitter"],
        "classification": "BETWEEN_TRIAL_ONLY",
    },
    {
        "name": "timeout",
        "source_fields": ["Time_Max_Exceeded"],
        "timestamp_fields": ["Trial_Datetime"],
        "classification": "ABSENT",
        "semantic_status": "ABSENT",
    },
    {
        "name": "computer_self",
        "source_fields": ["Trial_Type"],
        "timestamp_fields": ["Trial_Datetime"],
        "classification": "TRIAL_CONSTANT",
    },
    {
        "name": "prediction_category",
        "source_fields": ["Trial_Condition"],
        "timestamp_fields": ["Trial_Datetime"],
        "classification": "UNKNOWN",
        "semantic_status": "SEMANTICS_UNRESOLVED",
    },
)

PAIR_CLASSIFICATIONS = (
    {
        "reaction_coordinate": "Trial_Probability",
        "proposed_endpoints": "result_or_condition_label",
        "classification": [
            "SEMANTICS_UNRESOLVED",
            "INSUFFICIENT_TEMPORAL_DYNAMICS",
        ],
        "reason": (
            "Probability is supplied at round level and does not evolve "
            "within the round."
        ),
    },
    {
        "reaction_coordinate": "Trial_Probability",
        "proposed_endpoints": "prediction_left_right",
        "classification": [
            "STATIC_PSYCHOMETRIC_MAPPING",
            "INSUFFICIENT_TEMPORAL_DYNAMICS",
        ],
        "reason": (
            "One choice is recorded per round; no within-round trajectory "
            "is present."
        ),
    },
    {
        "reaction_coordinate": "Trial_Probability",
        "proposed_endpoints": "post_result_report",
        "classification": ["INSUFFICIENT_TEMPORAL_DYNAMICS"],
        "reason": "The scalar remains a trial parameter.",
    },
    {
        "reaction_coordinate": "Trial_Stake",
        "proposed_endpoints": "gain_loss",
        "classification": ["INSUFFICIENT_TEMPORAL_DYNAMICS"],
        "reason": "Stake is fixed for the round and is payoff magnitude.",
    },
    {
        "reaction_coordinate": "Trial_Balance_Effect",
        "proposed_endpoints": "gain_loss",
        "classification": ["INSUFFICIENT_TEMPORAL_DYNAMICS"],
        "reason": "Stored effect duplicates unsigned stake magnitude.",
    },
    {
        "reaction_coordinate": "Trial_Balance",
        "proposed_endpoints": "gain_loss",
        "classification": ["INSUFFICIENT_TEMPORAL_DYNAMICS"],
        "reason": "Balance changes between rounds and the jump is the event.",
    },
    {
        "reaction_coordinate": "Prediction_Result",
        "proposed_endpoints": "any_terminal",
        "classification": [
            "EXPERIMENT_GENERATOR_TRIVIAL",
            "INSUFFICIENT_TEMPORAL_DYNAMICS",
        ],
        "reason": "One terminal die-face value is recorded per round.",
    },
    {
        "reaction_coordinate": "Trial_Condition",
        "proposed_endpoints": "any_terminal",
        "classification": [
            "SEMANTICS_UNRESOLVED",
            "INSUFFICIENT_TEMPORAL_DYNAMICS",
        ],
        "reason": (
            "The condition label's source derivation is unresolved and it "
            "does not evolve within a round."
        ),
    },
    {
        "reaction_coordinate": "ECG/RSP/EDA/EMG",
        "proposed_endpoints": "gambling_terminal",
        "classification": ["SEMANTICS_UNRESOLVED"],
        "reason": (
            "Physiology is dynamic but transition proximity and state "
            "sufficiency are unverified; it is not a task-state RC."
        ),
    },
)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "n/a", "na"}:
            continue
        if text not in seen:
            output.append(text)
            seen.add(text)
        if len(output) >= MAX_UNIQUE_VALUES:
            break
    return output


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = [
            str(value).lstrip("\ufeff")
            for value in (reader.fieldnames or [])
        ]
        reader.fieldnames = columns
        rows = [
            {str(key): str(value) for key, value in row.items()}
            for row in reader
        ]
    return columns, rows


def _as_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed == parsed else None


def _numeric_summary(rows: Sequence[Mapping[str, str]], field: str) -> dict[str, Any]:
    values = [
        parsed
        for row in rows
        if (parsed := _as_float(row.get(field))) is not None
    ]
    if not values:
        return {
            "n_finite": 0,
            "unique_values": [],
            "min": None,
            "max": None,
        }
    return {
        "n_finite": len(values),
        "unique_values": _unique(values),
        "min": min(values),
        "max": max(values),
    }


def _parse_datetime(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _timing_summary(rows_by_pid: Mapping[str, Sequence[Mapping[str, str]]]) -> dict[str, Any]:
    gaps: list[float] = []
    timestamps = 0
    for rows in rows_by_pid.values():
        parsed = [
            timestamp
            for row in rows
            if (timestamp := _parse_datetime(row.get("Trial_Datetime", "")))
            is not None
        ]
        timestamps += len(parsed)
        gaps.extend(
            (current - previous).total_seconds()
            for previous, current in zip(parsed, parsed[1:])
        )
    return {
        "finite_trial_datetime_rows": timestamps,
        "n_inter_round_gaps": len(gaps),
        "median_gap_sec": statistics.median(gaps) if gaps else None,
        "min_gap_sec": min(gaps) if gaps else None,
        "max_gap_sec": max(gaps) if gaps else None,
    }


def _behavior_summary(paths: Sequence[Path]) -> dict[str, Any]:
    all_rows: list[dict[str, str]] = []
    columns: set[str] = set()
    file_summaries: list[dict[str, Any]] = []
    for path in sorted(paths):
        fieldnames, rows = _read_tsv(path)
        columns.update(fieldnames)
        all_rows.extend(rows)
        file_summaries.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "n_rows": len(rows),
                "n_unique_index": len(
                    {
                        row.get("index", "")
                        for row in rows
                        if row.get("index", "").strip()
                    }
                ),
                "columns": fieldnames,
                "n_unique_trial_order": len(
                    {
                        row.get("Trial_Order", "")
                        for row in rows
                        if row.get("Trial_Order", "").strip()
                    }
                ),
                "n_missing_trial_order": sum(
                    not row.get("Trial_Order", "").strip()
                    or row.get("Trial_Order", "").strip().lower()
                    in {"n/a", "na", "nan"}
                    for row in rows
                ),
            }
        )
    by_pid: dict[str, list[dict[str, str]]] = {}
    for row in all_rows:
        pid = row.get("Participant_ID", "")
        by_pid.setdefault(pid, []).append(row)
    for rows in by_pid.values():
        rows.sort(
            key=lambda row: (
                _parse_datetime(row.get("Trial_Datetime", ""))
                is None,
                _parse_datetime(row.get("Trial_Datetime", ""))
                or datetime.min,
            )
        )
    balance_changes = 0
    balance_pairs = 0
    effect_equals_stake = 0
    effect_comparisons = 0
    for rows in by_pid.values():
        previous_balance: float | None = None
        for row in rows:
            balance = _as_float(row.get("Trial_Balance"))
            if balance is not None and previous_balance is not None:
                balance_pairs += 1
                if balance != previous_balance:
                    balance_changes += 1
            if balance is not None:
                previous_balance = balance
            effect = _as_float(row.get("Trial_Balance_Effect"))
            stake = _as_float(row.get("Trial_Stake"))
            if effect is not None and stake is not None:
                effect_comparisons += 1
                if effect == stake:
                    effect_equals_stake += 1
    field_summaries = {
        field: _numeric_summary(all_rows, field)
        for field in REQUIRED_VARIABLES
        if field in columns
    }
    categorical_values = {
        field: _unique(row.get(field, "") for row in all_rows)
        for field in (
            "Trial_Type",
            "Trial_Condition",
            "Trial_ProbableSide",
            "Prediction_Choice",
            "Prediction_Result",
            "Answer_Prediction_Correct",
            "Answer_Prediction_Agent",
        )
        if field in columns
    }
    return {
        "n_behavior_files": len(paths),
        "n_behavior_rows": len(all_rows),
        "n_participants_with_behavior": len(
            {pid for pid in by_pid if pid}
        ),
        "columns": sorted(columns),
        "file_summaries": file_summaries,
        "round_grain": {
            "row_identifier_field": "index",
            "trial_order_present": "Trial_Order" in columns,
            "all_files_one_row_per_trial": bool(file_summaries)
            and all(
                summary["n_rows"] == summary["n_unique_index"]
                for summary in file_summaries
            ),
            "files_with_duplicate_row_identifier": [
                summary["path"]
                for summary in file_summaries
                if summary["n_rows"] != summary["n_unique_index"]
            ],
            "trial_order_complete": bool(file_summaries)
            and all(
                summary["n_missing_trial_order"] == 0
                for summary in file_summaries
            ),
            "files_with_missing_trial_order": [
                {
                    "path": summary["path"],
                    "n_missing_trial_order": summary["n_missing_trial_order"],
                }
                for summary in file_summaries
                if summary["n_missing_trial_order"] > 0
            ],
        },
        "field_summaries": field_summaries,
        "categorical_values": categorical_values,
        "rows_by_participant": {
            pid: len(rows) for pid, rows in sorted(by_pid.items())
        },
        "round_count_distribution": _unique(
            len(rows) for rows in by_pid.values()
        ),
        "timing": _timing_summary(by_pid),
        "trial_balance_changes_between_rows": balance_changes,
        "trial_balance_adjacent_pairs": balance_pairs,
        "balance_effect_equals_stake": effect_equals_stake,
        "balance_effect_comparisons": effect_comparisons,
    }


def _source_file_inventory(
    root: Path,
    canonical_behavior: Path,
    canonical_events: Path,
    canonical_physio: Path,
) -> dict[str, Any]:
    all_files = sorted(path for path in root.rglob("*") if path.is_file())
    source_code = [
        str(path.relative_to(root)).replace("\\", "/")
        for path in all_files
        if path.suffix.lower() in {".m", ".py", ".psyexp"}
        or "code" in path.parts
    ]
    event_columns, event_rows = _read_tsv(canonical_events)
    physio = json.loads(canonical_physio.read_text(encoding="utf-8"))
    return {
        "readme": str(root / "README"),
        "dataset_description": str(root / "dataset_description.json"),
        "canonical_behavior": str(canonical_behavior),
        "canonical_events": str(canonical_events),
        "canonical_physio_json": str(canonical_physio),
        "source_code_paths": source_code,
        "source_code_present": bool(source_code),
        "events": {
            "sha256": sha256_file(canonical_events),
            "columns": event_columns,
            "n_rows": len(event_rows),
            "trial_type_values": _unique(
                row.get("trial_type", "") for row in event_rows
            ),
            "sync_only": set(event_columns) >= {
                "onset",
                "duration",
                "trial_type",
                "value",
                "sample",
            }
            and set(
                row.get("trial_type", "") for row in event_rows
            )
            <= {"Sync(1)"},
        },
        "physio": {
            "sha256": sha256_file(canonical_physio),
            "sampling_frequency": physio.get("SamplingFrequency"),
            "start_time": physio.get("StartTime"),
            "columns": physio.get("Columns", []),
            "continuous_signal_columns": [
                column
                for column in physio.get("Columns", [])
                if column not in {"time", "Digital input"}
            ],
            "auxiliary_or_sync_columns": [
                column
                for column in physio.get("Columns", [])
                if column in {"time", "Digital input"}
            ],
        },
    }


def _variable_inventory(
    behavior: Mapping[str, Any],
    physio_columns: Sequence[str],
) -> list[dict[str, Any]]:
    columns = set(behavior.get("columns", []))
    variables: list[dict[str, Any]] = []
    for field in REQUIRED_VARIABLES:
        definition = VARIABLE_CLASSIFICATIONS[field]
        variables.append(
            {
                "name": field,
                **definition,
                "observed_in_behavior": field in columns,
                "classification": (
                    definition["dynamicality"]
                    if field in columns
                    else "ABSENT"
                ),
                "numeric_summary": behavior.get(
                    "field_summaries", {}
                ).get(field),
            }
        )
    known_columns = set(REQUIRED_VARIABLES)
    for field in sorted(columns - known_columns):
        variables.append(
            {
                "name": field,
                "dynamicality": "UNKNOWN",
                "semantic_role": (
                    "unlisted source column; semantics require source review"
                ),
                "observed_in_behavior": True,
                "classification": "UNKNOWN",
                "numeric_summary": behavior.get(
                    "field_summaries", {}
                ).get(field),
            }
        )
    variables.append(
        {
            "name": "ECG/RSP/EDA/EMG",
            "dynamicality": "CONTINUOUS_WITHIN_SEGMENT",
            "semantic_role": "peripheral physiology",
            "observed_in_physio_metadata": bool(physio_columns),
            "physio_columns": [
                column
                for column in physio_columns
                if column not in {"time", "Digital input"}
            ],
            "auxiliary_or_sync_columns": [
                column
                for column in physio_columns
                if column in {"time", "Digital input"}
            ],
            "classification": "CONTINUOUS_WITHIN_SEGMENT",
            "task_state_status": "NOT_A_TASK_STATE_RC",
        }
    )
    observed_columns = set(behavior.get("columns", []))
    search_terms = (
        "confidence",
        "cursor",
        "force",
        "trajectory",
        "evidence",
        "running",
        "accumulated",
    )
    searched = sorted(
        column
        for column in observed_columns
        if any(term in column.lower() for term in search_terms)
    )
    variables.append(
        {
            "name": "continuous_task_state_search",
            "searched_terms": list(search_terms),
            "matched_columns": searched,
            "classification": (
                "UNKNOWN" if searched else "ABSENT"
            ),
            "candidate_search_unknown_columns": searched,
        }
    )
    return variables


def audit_source(
    *,
    source_root: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    if not source_root.is_dir():
        return {
            "schema": SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "protocol_path": str(protocol_path),
            "protocol_sha256": sha256_file(protocol_path),
            "dataset": DATASET_ID,
            "source_root": str(source_root),
            "source_status": "SOURCE_UNAVAILABLE",
            "gate_status": "SOURCE_UNAVAILABLE",
            "candidate_dynamic_task_variables": [],
            "failure_reasons": ["source_root_unavailable"],
        }
    canonical_behavior = (
        source_root
        / CANONICAL_PARTICIPANT
        / "ses-01"
        / "beh"
        / f"{CANONICAL_PARTICIPANT}_ses-01_task-GG_run-01_beh.tsv"
    )
    canonical_events = (
        source_root
        / CANONICAL_PARTICIPANT
        / "ses-01"
        / "eeg"
        / f"{CANONICAL_PARTICIPANT}_ses-01_task-GG_run-01_events.tsv"
    )
    canonical_physio = (
        source_root
        / CANONICAL_PARTICIPANT
        / "ses-01"
        / "beh"
        / f"{CANONICAL_PARTICIPANT}_ses-01_task-GG_run-01_physio.json"
    )
    required = [source_root / "README", source_root / "dataset_description.json",
                canonical_behavior, canonical_events, canonical_physio]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {
            "schema": SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "protocol_path": str(protocol_path),
            "protocol_sha256": sha256_file(protocol_path),
            "dataset": DATASET_ID,
            "source_root": str(source_root),
            "source_status": "AUDIT_INCOMPLETE",
            "gate_status": "AUDIT_INCOMPLETE",
            "failure_reasons": ["required_source_file_missing"],
            "missing_files": missing,
        }
    behavior_paths = sorted(
        path
        for path in source_root.rglob("*_task-GG_run-01_beh.tsv")
        if path.is_file()
    )
    if not behavior_paths:
        return {
            "schema": SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "protocol_path": str(protocol_path),
            "protocol_sha256": sha256_file(protocol_path),
            "dataset": DATASET_ID,
            "source_root": str(source_root),
            "source_status": "AUDIT_INCOMPLETE",
            "gate_status": "AUDIT_INCOMPLETE",
            "failure_reasons": ["no_GG_behavior_tables"],
        }
    readme_text = (source_root / "README").read_text(
        encoding="utf-8",
        errors="replace",
    )
    source_files = _source_file_inventory(
        source_root,
        canonical_behavior,
        canonical_events,
        canonical_physio,
    )
    behavior = _behavior_summary(behavior_paths)
    variables = _variable_inventory(
        behavior,
        source_files["physio"]["columns"],
    )
    missing_expected_fields = [
        field
        for field in REQUIRED_VARIABLES
        if field not in behavior["columns"]
    ]
    classification_by_name = {
        variable["name"]: variable["classification"]
        for variable in variables
    }
    dynamic_task = [
        name
        for name, classification in classification_by_name.items()
        if classification
        in {"CONTINUOUS_WITHIN_SEGMENT", "PIECEWISE_DYNAMIC_WITHIN_SEGMENT"}
        and name != "ECG/RSP/EDA/EMG"
    ]
    grain = behavior["round_grain"]
    search_summary = next(
        (
            variable
            for variable in variables
            if variable["name"] == "continuous_task_state_search"
        ),
        {},
    )
    unknown_task_candidates = search_summary.get(
        "candidate_search_unknown_columns", []
    )
    if missing_expected_fields:
        gate_status = "AUDIT_INCOMPLETE"
    elif not grain["trial_order_present"] or not grain[
        "all_files_one_row_per_trial"
    ]:
        gate_status = "SEMANTICS_UNRESOLVED"
    elif unknown_task_candidates:
        gate_status = "SEMANTICS_UNRESOLVED"
    elif dynamic_task:
        gate_status = "PASS_DYNAMIC_CANDIDATE"
    else:
        gate_status = "NO_DYNAMIC_EXTERNAL_TASK_RC"
    source_code_present = source_files["source_code_present"]
    unresolved_semantics = [
        "exact face-to-side mapping is not stored in the canonical behavior table",
        "computer payment rule is not reconstructed from an original script",
        "exact EEG Sync-to-stage mapping is unavailable",
    ]
    if not source_code_present:
        unresolved_semantics.insert(
            0,
            "original GG experiment script absent locally",
        )
    return {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "dataset": DATASET_ID,
        "dataset_version": "OpenNeuro v1.0.2",
        "source_root": str(source_root),
        "source_status": "AVAILABLE",
        "gate_status": gate_status,
        "source_metadata": {
            "readme_sha256": sha256_file(source_root / "README"),
            "readme_task_description_present": (
                "Gambling Game" in readme_text
                and "144 rounds" in readme_text
            ),
            "dataset_description_sha256": sha256_file(
                source_root / "dataset_description.json"
            ),
            "source_files": source_files,
        },
        "behavior_summary": behavior,
        "missing_expected_fields": missing_expected_fields,
        "variables": variables,
        "round_reconstruction": {
            "source": "README + one-row-per-round GG behavior tables",
            "rounds_per_protocol": 144,
            "stages": [
                "round_onset",
                "trial_type",
                "stake_and_probability",
                "prediction_choice",
                "prediction_result",
                "post_result_reports",
                "balance_update",
                "round_end",
            ],
            "within_round_time_series_present": not grain[
                "all_files_one_row_per_trial"
            ],
            "eeg_stage_events_present": not source_files["events"]["sync_only"],
            "eeg_event_semantics": (
                "Sync(1) only"
                if source_files["events"]["sync_only"]
                else "stage labels present"
            ),
            "source_codebook_present": bool(
                "Gambling Game" in readme_text
                and "144 rounds" in readme_text
            ),
            "original_experiment_script_present": bool(
                source_files["source_code_present"]
            ),
            "behavior_grain": grain,
            "available_timestamp_fields": [
                field
                for field in behavior["columns"]
                if field == "Trial_Datetime"
                or field.endswith("_RT")
                or field.endswith("_Jitter")
            ],
        },
        "terminal_candidates": list(TERMINAL_CANDIDATES),
        "rc_pair_classifications": list(PAIR_CLASSIFICATIONS),
        "candidate_dynamic_task_variables": dynamic_task,
        "unresolved_semantics": unresolved_semantics,
        "runner_contract": {
            "read_only": True,
            "downloaded": False,
            "nmd_outputs_read": False,
            "committor_estimator_called": False,
            "models_fit": False,
            "outcome_association_tested": False,
            "a_b_selected": False,
            "a_b_frozen": False,
            "held_out_opened": False,
            "derived_events_written": False,
        },
        "claim_boundary": (
            "Source semantics only. "
            + (
                "No genuine first-passage RC was identified; no A/B pair "
                "was selected."
                if gate_status == "NO_DYNAMIC_EXTERNAL_TASK_RC"
                else "No negative RC finding is permitted from this audit."
            )
        ),
    }
