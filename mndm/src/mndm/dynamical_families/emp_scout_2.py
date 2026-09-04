"""Read-only source-metadata inventory for OD-EMP-SCOUT-2.

This module deliberately operates below the ingest/output boundary. It reads
small, representative source metadata files only; it never opens processed
NMD artifacts, fits a model, downloads data, or constructs a committor.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "mndm.od_emp_scout_2_inventory.v1"
PROTOCOL_ID = "OD-EMP-SCOUT-2"
MAX_SAMPLE_FILES_PER_CLASS = 1
MAX_SAMPLE_ROWS = 20
MAX_SAMPLE_VALUES = 8
MAX_TEXT_CHARS = 1200
MAX_CODE_LINES = 24
NEVER_OPEN_SUFFIXES = {".h5", ".hdf5", ".parquet", ".feather"}

DATASET_SPECS: tuple[dict[str, Any], ...] = (
    {
        "dataset_id": "ds004511",
        "config_path": "mndm/config/sources/openneuro/config_ingest_ds004511.yaml",
        "configured_roots": ("M:/datasets/received/openneuro/ds004511",),
        "fallback_roots": (),
        "source_role": "raw_bids",
        "candidate_hints": (
            "trial_probability",
            "trial_stake",
            "trial_probableside",
        ),
        "candidate_rationale": (
            "Gambling-task behavior exposes experimentally supplied "
            "probability/stake fields; transition proximity remains "
            "unverified."
        ),
    },
    {
        "dataset_id": "ds003474",
        "config_path": "mndm/config/sources/openneuro/config_ingest_ds003474.yaml",
        "configured_roots": ("N:/received/ds003474",),
        "fallback_roots": (),
        "source_role": "raw_bids",
        "candidate_hints": (),
        "candidate_rationale": (
            "Inspected task metadata exposes discrete stimulus, response, "
            "and feedback labels but no external continuous task scalar."
        ),
    },
    {
        "dataset_id": "ds003838",
        "config_path": "mndm/config/sources/openneuro/config_ingest_ds003838.yaml",
        "configured_roots": (
            "G:/Science_Datasets_longtime_storage/openneuro/received/ds003838",
        ),
        "fallback_roots": (),
        "source_role": "raw_bids",
        "candidate_hints": (),
        "candidate_rationale": (
            "No conclusion is permitted while the configured source root is "
            "unavailable."
        ),
    },
    {
        "dataset_id": "ds006848",
        "config_path": "mndm/config/sources/openneuro/config_ingest_ds006848.yaml",
        "configured_roots": (
            "K:/ExternalReceivedDatasets/openneuro/received/ds006848",
        ),
        "fallback_roots": (),
        "source_role": "raw_bids",
        "candidate_hints": ("encoding_digitvalue", "partial_score", "score"),
        "candidate_rationale": (
            "Source metadata exposes stepwise digit-value stimulus markers "
            "and possible behavior scores; transition proximity is "
            "unverified."
        ),
    },
    {
        "dataset_id": "ds004587",
        "config_path": "mndm/config/sources/openneuro/config_ingest_ds004587.yaml",
        "configured_roots": (
            "K:/ExternalReceivedDatasets/openneuro/received/ds004587",
        ),
        "fallback_roots": (),
        "source_role": "raw_bids",
        "candidate_hints": ("illusion_strength", "illusion_difference"),
        "candidate_rationale": (
            "Illusion Game behavior exposes task-defined stimulus "
            "parameters; whether they are proximal to a future A/B "
            "transition is unverified."
        ),
    },
    {
        "dataset_id": "ds003506",
        "config_path": "mndm/config/sources/openneuro/config_ingest_ds003506.yaml",
        "configured_roots": ("M:/datasets/received/openneuro/ds003506",),
        "fallback_roots": (
            "M:/datasets/received/openneuro/ds003506_trigger_enriched",
        ),
        "source_role": "raw_bids_preferred_event_staged_fallback",
        "candidate_hints": (),
        "candidate_rationale": (
            "Reward/punishment/timeout and response variables are discrete "
            "outcomes or labels; no qualifying external scalar was found "
            "in the inspected contract."
        ),
    },
    {
        "dataset_id": "ds003509",
        "config_path": "mndm/config/sources/openneuro/config_ingest_ds003509.yaml",
        "configured_roots": ("M:/datasets/received/openneuro/ds003509",),
        "fallback_roots": (
            "M:/datasets/received/openneuro/ds003509_trigger_enriched",
        ),
        "source_role": "raw_bids_preferred_event_staged_fallback",
        "candidate_hints": ("stimulus_reward_probability",),
        "candidate_rationale": (
            "The Simon task exposes an experimentally scheduled reward "
            "probability. It is stepwise and trial-wise; transition "
            "proximity remains unverified."
        ),
    },
)

_SKIP_PARTS = {".git", ".datalad", "__pycache__", "derivatives"}
_PHYSIO_TERMS = (
    "ecg",
    "respiration",
    "rsp",
    "eda",
    "ppg",
    "pupil",
    "hrv",
    "heart",
    "emg",
)
_FORBIDDEN_TERMS = (
    "latent",
    "fitted",
    "model_derived",
    "prediction_error",
    "q_value",
    "rl_value",
)
_CONTINUOUS_TERMS = (
    "evidence",
    "confidence",
    "cursor",
    "force",
    "trajectory",
    "intensity",
)
_ORDINAL_TERMS = (
    "probability",
    "strength",
    "difference",
    "digitvalue",
    "stake",
)
_DISCRETE_TERMS = (
    "trial_type",
    "event",
    "feedback",
    "response",
    "correct",
    "accuracy",
    "condition",
    "phase",
    "block",
    "stimulus",
    "outcome",
    "choice",
)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truncate(value: Any, limit: int = MAX_TEXT_CHARS) -> str:
    text = str(value).replace("\x00", "")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _unique_values(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = _truncate(value, 160)
        if not text or text.lower() in {"nan", "n/a", "na"}:
            continue
        if text not in seen:
            output.append(text)
            seen.add(text)
        if len(output) >= MAX_SAMPLE_VALUES:
            break
    return output


def classify_variable(name: str) -> tuple[str, str]:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    tokens = set(normalized.split("_"))
    if normalized in {"trial_type", "event_type", "task_type"}:
        return (
            "discrete_label",
            "Name identifies a discrete task/event type label.",
        )
    if "jitter" in tokens or normalized in {
        "value",
        "sample",
        "sample_value",
    }:
        return (
            "unevaluated",
            "Generic trigger/timing field; not treated as a task scalar.",
        )
    if any(term in normalized for term in _FORBIDDEN_TERMS):
        return (
            "model_derived_forbidden",
            "Name suggests a fitted or latent model quantity.",
        )
    if any(term in tokens for term in _PHYSIO_TERMS) or any(
        normalized.startswith(prefix)
        for prefix in ("resp_", "respiration_", "ecg_", "ppg_", "pupil_")
    ):
        return (
            "physiology",
            "Name indicates a physiological or biosignal measurement.",
        )
    if any(term in tokens for term in _CONTINUOUS_TERMS):
        return (
            "continuous_external",
            "Name suggests an externally supplied continuous task variable.",
        )
    if any(term in tokens for term in _ORDINAL_TERMS):
        return (
            "ordinal_task_param",
            "Name suggests a supplied probability, value, strength, or score.",
        )
    if (
        "score" in tokens
        and any(
            term in tokens for term in {"partial", "trial", "task", "behavior"}
        )
    ) or normalized in {
        "partialscore",
        "trialscore",
        "taskscore",
        "behaviorscore",
    }:
        return (
            "ordinal_task_param",
            "Name suggests a supplied task performance score.",
        )
    if any(term in tokens for term in _DISCRETE_TERMS):
        return (
            "discrete_label",
            "Name suggests a task/event/outcome label.",
        )
    return ("unevaluated", "No frozen semantic rule matched this name.")


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def _iter_files(root: Path) -> Iterable[Path]:
    paths = sorted(root.rglob("*"))
    return (
        path
        for path in paths
        if path.is_file()
        and not any(part.lower() in _SKIP_PARTS for part in path.parts)
    )


def _file_class(path: Path) -> str | None:
    name = path.name.lower()
    if path.suffix.lower() in NEVER_OPEN_SUFFIXES:
        return None
    if name.startswith("readme"):
        return "readme"
    if name == "dataset_description.json":
        return "dataset_description"
    if name == "participants.tsv":
        return "participants"
    if name.endswith("_events.tsv"):
        return "events_tsv"
    if name.endswith("_events.json"):
        return "events_json"
    if name.endswith("_physio.json"):
        return "physio_json"
    if name.endswith("_beh.tsv"):
        return "beh_tsv"
    if name.endswith("_beh.json"):
        return "beh_json"
    if path.suffix.lower() == ".m":
        return "matlab_code"
    return None


def _task_key(path: Path) -> str:
    match = re.search(r"(?:^|_)task-([^_]+)", path.name, flags=re.IGNORECASE)
    return match.group(1).lower() if match else "unspecified"


def _select_samples(root: Path) -> tuple[dict[str, int], list[tuple[str, Path]]]:
    counts: dict[str, int] = {}
    grouped: dict[str, list[Path]] = {}
    for path in _iter_files(root):
        kind = _file_class(path)
        if kind is None:
            continue
        counts[kind] = counts.get(kind, 0) + 1
        group = (
            _task_key(path)
            if kind in {
                "events_tsv",
                "events_json",
                "beh_tsv",
                "beh_json",
                "physio_json",
            }
            else kind
        )
        grouped.setdefault(f"{kind}:{group}", []).append(path)

    selected: list[tuple[str, Path]] = []
    for group, paths in sorted(grouped.items()):
        kind = group.split(":", 1)[0]
        selected.extend((kind, path) for path in paths[:MAX_SAMPLE_FILES_PER_CLASS])
    return counts, selected


def _read_tsv(path: Path, *, header_only: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": str(path),
        "format": "tsv",
        "read_status": "ok",
        "columns": [],
        "sample_values": {},
    }
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader, [])
            columns = [str(value).strip() for value in header if str(value).strip()]
            rows = []
            if not header_only:
                for row in reader:
                    if row:
                        rows.append(row)
                    if len(rows) >= MAX_SAMPLE_ROWS:
                        break
    except (OSError, UnicodeError, csv.Error) as error:
        payload["read_status"] = f"error:{type(error).__name__}"
        return payload
    payload["columns"] = columns
    payload["sample_values"] = {
        column: _unique_values(
            row[index] if index < len(row) else ""
            for row in rows
        )
        for index, column in enumerate(columns)
        if _unique_values(
            row[index] if index < len(row) else ""
            for row in rows
        )
    }
    return payload


def _read_json_metadata(
    path: Path,
    *,
    expose_top_level_keys_as_columns: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": str(path),
        "format": "json",
        "read_status": "ok",
    }
    try:
        raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        payload["read_status"] = f"error:{type(error).__name__}"
        return payload
    if isinstance(raw, dict):
        keys = sorted(str(key) for key in raw)[:80]
        payload["keys"] = keys
        if expose_top_level_keys_as_columns:
            payload["columns"] = keys
        columns = raw.get("Columns")
        if isinstance(columns, dict):
            payload["column_metadata"] = sorted(str(key) for key in columns)[:80]
            payload["columns"] = sorted(str(key) for key in columns)[:80]
        elif isinstance(columns, list):
            payload["columns"] = [
                str(value) for value in columns[:80]
            ]
        if "Description" in raw:
            payload["description"] = _truncate(raw["Description"])
    else:
        payload["top_level_type"] = type(raw).__name__
    return payload


def _read_text_snippet(path: Path, *, code: bool = False) -> dict[str, Any]:
    payload = {
        "path": str(path),
        "format": "matlab_code" if code else "readme",
        "read_status": "ok",
    }
    try:
        lines = path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
    except (OSError, UnicodeError) as error:
        payload["read_status"] = f"error:{type(error).__name__}"
        return payload
    if code:
        keywords = (
            "prob",
            "reward",
            "value",
            "stim",
            "feedback",
            "choice",
            "response",
            "confidence",
            "force",
            "cursor",
            "evidence",
        )
        selected = [
            line.strip()
            for line in lines
            if any(keyword in line.lower() for keyword in keywords)
        ][:MAX_CODE_LINES]
        payload["keyword_lines"] = [_truncate(line, 240) for line in selected]
    else:
        payload["snippet"] = _truncate(
            "\n".join(line.strip() for line in lines[:30] if line.strip())
        )
    return payload


def _variable_observations(
    file_payloads: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for file_payload in file_payloads:
        if file_payload.get("kind") == "participants":
            continue
        path = str(file_payload.get("path", ""))
        source_columns = file_payload.get("columns", [])
        samples = file_payload.get("sample_values", {})
        if not isinstance(source_columns, list):
            continue
        for name in source_columns:
            classification, reason = classify_variable(str(name))
            values = samples.get(str(name), []) if isinstance(samples, dict) else []
            observations.append(
                {
                    "name": str(name),
                    "classification": classification,
                    "reason": reason,
                    "source_file": path,
                    "sample_values": values,
                }
            )
            for value in values:
                normalized = str(value).lower().replace(" ", "_")
                if "encoding_digitvalue_" in normalized:
                    observations.append(
                        {
                            "name": "Encoding_DigitValue_*",
                            "classification": "ordinal_task_param",
                            "reason": (
                                "Observed stepwise digit-value task marker "
                                "in trial_type values."
                            ),
                            "source_file": path,
                            "sample_values": [value],
                        }
                    )
    deduplicated: dict[tuple[str, str], dict[str, Any]] = {}
    for observation in observations:
        key = (
            observation["name"],
            observation["classification"],
        )
        deduplicated.setdefault(key, observation)
    return list(deduplicated.values())


def _adjudicate(
    spec: Mapping[str, Any],
    variables: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    matches = []
    for variable in variables:
        classification = variable.get("classification")
        if classification in {"continuous_external", "ordinal_task_param"}:
            matches.append(variable)
    if matches:
        return {
            "candidate_present": True,
            "transition_proximity_status": "UNVERIFIED_CANDIDATE",
            "decision": "AUDIT_FIRST",
            "matched_candidates": matches,
            "rationale": (
                "A non-physiological task-variable candidate was visible in "
                "the inspected source metadata. Transition proximity and "
                "A/B suitability remain unverified."
            ),
        }
    return {
        "candidate_present": False,
        "transition_proximity_status": "NO_EXTERNAL_RC",
        "decision": "NO_EXTERNAL_RC",
        "matched_candidates": [],
        "rationale": (
            "No non-physiological continuous or ordinal task scalar was "
            "visible in the inspected source metadata."
        ),
    }


def scan_dataset(
    spec: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    configured = [Path(value) for value in spec.get("configured_roots", ())]
    fallback = [Path(value) for value in spec.get("fallback_roots", ())]
    root_probe_errors: list[str] = []
    selected_root = None
    for path in configured:
        try:
            if path.is_dir():
                selected_root = path
                break
        except OSError as error:
            root_probe_errors.append(
                f"configured:{path}:{type(error).__name__}:{error}"
            )
    root_kind = "configured"
    if selected_root is None:
        for path in fallback:
            try:
                if path.is_dir():
                    selected_root = path
                    break
            except OSError as error:
                root_probe_errors.append(
                    f"fallback:{path}:{type(error).__name__}:{error}"
                )
        root_kind = "fallback" if selected_root is not None else None
    result: dict[str, Any] = {
        "dataset_id": spec["dataset_id"],
        "config_path": spec["config_path"],
        "config_sha256": (
            sha256_file(repo_root / spec["config_path"])
            if repo_root is not None
            else None
        ),
        "configured_roots": [str(path) for path in configured],
        "fallback_roots": [str(path) for path in fallback],
        "source_root": str(selected_root) if selected_root else None,
        "source_root_kind": root_kind,
        "source_role": spec["source_role"],
        "source_status": "AVAILABLE" if selected_root else "SOURCE_UNAVAILABLE",
        "source_walk_error": None,
        "root_probe_errors": root_probe_errors,
        "file_class_counts": {},
        "sample_files": [],
        "variables": [],
        "adjudication": None,
    }
    if selected_root is None and root_probe_errors:
        result["source_status"] = "SOURCE_READ_ERROR"
        result["source_walk_error"] = "root_probe_failed"
        result["adjudication"] = {
            "candidate_present": None,
            "transition_proximity_status": "SOURCE_READ_ERROR",
            "decision": "AUDIT_INCOMPLETE",
            "matched_candidates": [],
            "rationale": (
                "One or more source-root probes failed; candidate presence "
                "is unknown."
            ),
        }
        return result
    if selected_root is None:
        result["adjudication"] = {
            "candidate_present": None,
            "transition_proximity_status": "SOURCE_UNAVAILABLE",
            "decision": "AUDIT_FIRST",
            "matched_candidates": [],
            "rationale": (
                "No configured or fallback source root was available; "
                "candidate presence is unknown."
            ),
        }
        return result

    try:
        counts, selected = _select_samples(selected_root)
    except OSError as error:
        result["source_status"] = "SOURCE_READ_ERROR"
        result["source_walk_error"] = f"{type(error).__name__}:{error}"
        result["adjudication"] = {
            "candidate_present": None,
            "transition_proximity_status": "SOURCE_READ_ERROR",
            "decision": "AUDIT_INCOMPLETE",
            "matched_candidates": [],
            "rationale": (
                "The source root existed but could not be enumerated."
            ),
        }
        return result
    result["file_class_counts"] = counts
    if not counts:
        result["source_status"] = "SOURCE_METADATA_EMPTY"
        result["adjudication"] = {
            "candidate_present": None,
            "transition_proximity_status": "SOURCE_METADATA_EMPTY",
            "decision": "AUDIT_INCOMPLETE",
            "matched_candidates": [],
            "rationale": (
                "The source root existed but contained no recognized source "
                "metadata classes."
            ),
        }
        return result
    file_payloads: list[dict[str, Any]] = []
    for kind, path in selected:
        if kind in {"events_tsv", "beh_tsv", "participants"}:
            payload = _read_tsv(path, header_only=kind == "participants")
            payload["relative_path"] = _relative(path, selected_root)
            payload["kind"] = kind
            file_payloads.append(payload)
        elif kind in {
            "events_json",
            "beh_json",
            "physio_json",
            "dataset_description",
        }:
            payload = _read_json_metadata(
                path,
                expose_top_level_keys_as_columns=kind
                in {"events_json", "beh_json", "physio_json"},
            )
            payload["relative_path"] = _relative(path, selected_root)
            payload["kind"] = kind
            file_payloads.append(payload)
        elif kind == "readme":
            payload = _read_text_snippet(path)
            payload["relative_path"] = _relative(path, selected_root)
            payload["kind"] = kind
            file_payloads.append(payload)
        elif kind == "matlab_code":
            payload = _read_text_snippet(path, code=True)
            payload["relative_path"] = _relative(path, selected_root)
            payload["kind"] = kind
            file_payloads.append(payload)
    result["sample_files"] = file_payloads
    critical_kinds = {
        "events_tsv",
        "events_json",
        "beh_tsv",
        "beh_json",
    }
    critical_errors = [
        payload
        for payload in file_payloads
        if payload.get("kind") in critical_kinds
        and payload.get("read_status") != "ok"
    ]
    if critical_errors:
        result["source_status"] = "SOURCE_READ_ERROR"
        result["source_walk_error"] = "critical_metadata_read_failed"
        result["adjudication"] = {
            "candidate_present": None,
            "transition_proximity_status": "SOURCE_READ_ERROR",
            "decision": "AUDIT_INCOMPLETE",
            "matched_candidates": [],
            "rationale": (
                "A representative events/behavior metadata file could not "
                "be read; candidate presence is unknown."
            ),
        }
        return result
    result["variables"] = _variable_observations(file_payloads)
    result["adjudication"] = _adjudicate(spec, result["variables"])
    return result


def _rank_key(result: Mapping[str, Any]) -> tuple[int, int, str]:
    status = result.get("adjudication", {}).get(
        "transition_proximity_status"
    )
    classifications = {
        str(variable.get("classification"))
        for variable in result.get("adjudication", {}).get(
            "matched_candidates", []
        )
    }
    names = {
        str(variable.get("name", "")).lower()
        for variable in result.get("adjudication", {}).get(
            "matched_candidates", []
        )
    }
    if any(
        term in name
        for name in names
        for term in ("evidence", "probability", "trajectory")
    ):
        candidate_rank = 0
    elif classifications & {"continuous_external", "ordinal_task_param"}:
        candidate_rank = 1
    else:
        candidate_rank = 2
    order = {
        "UNVERIFIED_CANDIDATE": 0,
        "NO_EXTERNAL_RC": 1,
        "SOURCE_UNAVAILABLE": 2,
        "SOURCE_METADATA_EMPTY": 3,
        "SOURCE_READ_ERROR": 4,
    }
    return (
        order.get(str(status), 5),
        candidate_rank,
        str(result.get("dataset_id")),
    )


def run_inventory(
    *,
    repo_root: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    protocol_hash = sha256_file(protocol_path)
    datasets = []
    for spec in DATASET_SPECS:
        try:
            datasets.append(scan_dataset(spec, repo_root=repo_root))
        except (OSError, RuntimeError) as error:
            datasets.append(
                {
                    "dataset_id": spec["dataset_id"],
                    "config_path": spec["config_path"],
                    "config_sha256": sha256_file(
                        repo_root / spec["config_path"]
                    ),
                    "configured_roots": list(spec["configured_roots"]),
                    "fallback_roots": list(spec["fallback_roots"]),
                    "source_root": None,
                    "source_root_kind": None,
                    "source_role": spec["source_role"],
                    "source_status": "SOURCE_READ_ERROR",
                    "source_walk_error": (
                        f"{type(error).__name__}:{error}"
                    ),
                    "root_probe_errors": [],
                    "file_class_counts": {},
                    "sample_files": [],
                    "variables": [],
                    "adjudication": {
                        "candidate_present": None,
                        "transition_proximity_status": "SOURCE_READ_ERROR",
                        "decision": "AUDIT_INCOMPLETE",
                        "matched_candidates": [],
                        "rationale": "Dataset scan failed closed.",
                    },
                }
            )
    reranked = sorted(datasets, key=_rank_key)
    reranked_ids = [str(dataset["dataset_id"]) for dataset in reranked]
    rank_position = {
        dataset_id: index + 1
        for index, dataset_id in enumerate(reranked_ids)
    }
    for dataset in datasets:
        dataset["prioritized_rank"] = rank_position[dataset["dataset_id"]]
    invalid_statuses = [
        dataset["dataset_id"]
        for dataset in datasets
        if dataset["adjudication"]["transition_proximity_status"]
        not in {
            "UNVERIFIED_CANDIDATE",
            "NO_EXTERNAL_RC",
            "SOURCE_UNAVAILABLE",
            "SOURCE_METADATA_EMPTY",
            "SOURCE_READ_ERROR",
        }
    ]
    incomplete_statuses = [
        dataset["dataset_id"]
        for dataset in datasets
        if dataset["adjudication"]["transition_proximity_status"]
        in {"SOURCE_READ_ERROR", "SOURCE_METADATA_EMPTY"}
    ]
    return {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_hash,
        "runner_version": "emp_scout_2.v1",
        "file_selection_rules": {
            "max_sample_files_per_class_or_task": MAX_SAMPLE_FILES_PER_CLASS,
            "max_sample_rows": MAX_SAMPLE_ROWS,
            "max_sample_values": MAX_SAMPLE_VALUES,
            "participant_file": "header_only",
            "events_beh_physio": "one_representative_file_per_task",
            "readme_dataset_description_code": "one_representative_per_class",
            "skip_parts": sorted(_SKIP_PARTS),
            "never_open_suffixes": sorted(NEVER_OPEN_SUFFIXES),
            "root_resolution": (
                "raw configured root first; event-staged root only as "
                "fallback when raw root is unavailable"
            ),
        },
        "runner_contract": {
            "max_sample_rows": MAX_SAMPLE_ROWS,
            "max_sample_values": MAX_SAMPLE_VALUES,
            "read_only": True,
            "downloaded": False,
            "nmd_outputs_read": False,
            "models_fit": False,
            "committor_estimator_called": False,
            "derived_events_written": False,
        },
        "datasets": datasets,
        "reranked_dataset_ids": reranked_ids,
        "richsleep_fallback": {
            "status": "FALLBACK_ONLY_IF_NO_CANDIDATE_IS_CONFIRMED",
            "risk_flag": "HIGH_RISK_RC",
            "reason": (
                "Do not open RichSleep before this task-source audit is "
                "reviewed; if no transition-proximal candidate survives, "
                "RichSleep remains an explicitly high-risk fallback."
            ),
        },
        "gate_status": (
            "PASS"
            if not invalid_statuses and not incomplete_statuses
            else "AUDIT_INCOMPLETE"
        ),
        "gate_failure_reasons": (
            [
                f"invalid_adjudication_status:{dataset_id}"
                for dataset_id in invalid_statuses
            ]
            + [
                f"source_metadata_incomplete:{dataset_id}"
                for dataset_id in incomplete_statuses
            ]
        ),
        "claim_boundary": (
            "Source metadata inventory only. UNVERIFIED_CANDIDATE does not "
            "mean eligibility, committor support, empirical qualification, "
            "or state sufficiency."
        ),
    }
