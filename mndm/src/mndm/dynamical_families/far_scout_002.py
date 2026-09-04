"""FAR-SCOUT-002 time-compatible perturbation-source scout.

This is a metadata-only scout.  It evaluates explicit physical amplitude,
native event timing, repeated biological units, and compatibility with the
existing modality-specific NMD window.  It never opens neural signal payloads,
NMD HDF5 outputs, or fits an outcome model.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from statistics import median
from typing import Any, Mapping

from .far_source_metadata_audit import (
    _audit_dandi_000009,
    _audit_ds005917,
    _audit_ds006623,
    _decode,
    _finite_float,
    _nwb_trial_table,
    _walk_files,
    _numeric_text,
    _truthy,
)
from .far_source_scout import (
    DIRECTION_ONLY as FAR001_DIRECTION_ONLY,
    _candidate_specs,
    _scan_ds006036,
)


PROTOCOL_ID = "FAR-SCOUT-002"
TIME_COMPATIBLE_CURVE = "TIME_COMPATIBLE_CURVE_CANDIDATE"
TIME_COMPATIBLE_POINT = "TIME_COMPATIBLE_POINT_CANDIDATE"
TIME_INCOMPATIBLE = "PERTURBATION_VALID_NMD_TIMEBASE_INCOMPATIBLE"
DIRECTION_ONLY = "DIRECTION_ONLY"
NO_EXPLICIT = "NO_EXPLICIT_PERTURBATION"
SOURCE_UNCERTAIN = "SOURCE_UNCERTAIN"

HARD_ISOLATED_POST_SEC = 4.0
PREFERRED_ISOLATED_POST_SEC = 8.0
MIN_CURVE_LEVELS = 3
MIN_SUBJECTS_PER_LEVEL = 2
EXTERNAL_DANDI_000458_ROOT = Path(
    r"K:\ExternalReceivedDatasets\DANDI\dandi_000458\raw\000458"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_text(value: Any) -> str:
    return str(_decode(value)).strip()


def _resolve_config_text(
    path: Path,
    *,
    seen: set[Path] | None = None,
) -> str:
    seen = seen or set()
    path = path.resolve()
    if path in seen or not path.is_file():
        return ""
    seen.add(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    chunks = [text]
    for import_value in re.findall(
        r'^\s*-\s*["\']([^"\']+)["\']',
        text,
        flags=re.MULTILINE,
    ):
        chunks.append(
            _resolve_config_text(
                path.parent / import_value,
                seen=seen,
            )
        )
    return "\n".join(chunks)


def _nmd_window_sec(config_path: Path) -> float | None:
    text = _resolve_config_text(config_path)
    blocks = re.findall(
        r"(?ms)^mnps:\s*\n((?:^[ \t]+.*(?:\n|$))*)",
        text,
    )
    for block in blocks:
        match = re.search(
            r"^\s+window_sec:\s*([0-9]+(?:\.[0-9]+)?)",
            block,
            re.MULTILINE,
        )
        if match:
            return float(match.group(1))
    return None


def _source_contract(
    *,
    dataset_id: str,
    source_kind: str,
    config_path: Path,
) -> dict[str, Any]:
    window_sec = _nmd_window_sec(config_path)
    continuous = source_kind in {
        "openneuro_bids_eeg",
        "dandi_nwb_electrical_stimulation",
    }
    return {
        "nmd_window_sec": window_sec,
        "nmd_contract_source": str(config_path),
        "continuous_neural_signal_compatible": continuous,
        "hard_isolated_post_sec": HARD_ISOLATED_POST_SEC,
        "preferred_isolated_post_sec": PREFERRED_ISOLATED_POST_SEC,
        "time_scale_rule": (
            "T_isolated >= matching NMD window is hard minimum; "
            "T_isolated >= 2 * NMD window is preferred"
        ),
    }


def _source_has_payload(dataset_id: str, root: Path) -> bool:
    if not root.is_dir():
        return False
    if dataset_id in {"dandi_000458", "dandi_000009"}:
        return any(root.rglob("*.nwb"))
    if dataset_id == "ds006036":
        return (root / "dataset_description.json").is_file() or any(
            root.rglob("*_events.tsv")
        )
    return any(root.iterdir())


def _select_root(
    *,
    dataset_id: str,
    roots: list[Path],
    override: Path | None,
) -> Path | None:
    if override is not None:
        return override if override.is_dir() else None
    return next(
        (
            root
            for root in roots
            if _source_has_payload(dataset_id, root)
        ),
        None,
    )


def _frozen_far003a_native_horizon(
    payload: Mapping[str, Any] | None,
) -> float | None:
    if not payload:
        return None
    values = [
        float(row["candidate_horizon"])
        for row in payload.get("family_ledger", [])
        if row.get("candidate_horizon") is not None
        and row.get("status")
        in {
            "RAW_SIGNAL_ONLY_PASS",
            "RAW_SIGNAL_COVERAGE_PASS",
            "SIGNAL_TIMEBASE_PASS",
        }
    ]
    return min(values) if values else None


def _empty_semantics() -> dict[str, Any]:
    return {
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
        "repeated_biological_units": None,
        "continuous_neural_signal": None,
        "isolated_post_horizon_sec": None,
    }


def _classification_reason(
    *,
    continuous: bool,
    timing_known: bool,
    level_count: int,
    supported_level_count: int,
    n_subjects: int,
    window_sec: float | None,
    max_isolated: float | None,
) -> tuple[str, str]:
    if not continuous:
        return (
            SOURCE_UNCERTAIN,
            "source is not continuous EEG/LFP-compatible with the active NMD EEG contract",
        )
    if not timing_known:
        return (
            SOURCE_UNCERTAIN,
            "native perturbation onset/offset could not be established without substituting a trial window",
        )
    if window_sec is None:
        return SOURCE_UNCERTAIN, "matching NMD window could not be resolved from config imports"
    if max_isolated is None or max_isolated <= max(
        HARD_ISOLATED_POST_SEC,
        window_sec,
    ):
        return (
            TIME_INCOMPATIBLE,
            "explicit perturbation is valid but no strict isolated post period "
            f"longer than the active {window_sec:g} s NMD window was observed",
        )
    if supported_level_count >= MIN_CURVE_LEVELS and n_subjects >= MIN_SUBJECTS_PER_LEVEL:
        if max_isolated > window_sec:
            return (
                TIME_COMPATIBLE_CURVE,
                "explicit physical amplitude, repeated units, timing, and NMD-window-compatible isolation",
            )
    if level_count > 0 and supported_level_count > 0:
        return (
            TIME_COMPATIBLE_POINT,
            "explicit physical amplitude and timing are present but curve-level support is limited",
        )
    return (
        TIME_INCOMPATIBLE,
        "physical perturbation is present but isolated support is below the active NMD window",
    )


def _audit_dandi_000458_time_compatibility(
    root: Path,
    config_path: Path,
    *,
    frozen_far003a_horizon_sec: float | None = None,
) -> dict[str, Any]:
    files = _walk_files(root, suffixes={".nwb"})
    contract = _source_contract(
        dataset_id="dandi_000458",
        source_kind="dandi_nwb_electrical_stimulation",
        config_path=config_path,
    )
    if not files:
        return {
            "dataset_id": "dandi_000458",
            "source_status": SOURCE_UNCERTAIN,
            "source_root": str(root),
            "classification": SOURCE_UNCERTAIN,
            "classification_reason": "no_native_nwb_files",
            "nmd_contract": contract,
        }

    columns = [
        "behavioral_epoch",
        "estim_current",
        "is_valid",
        "start_time",
        "stimulus_type",
        "stop_time",
    ]
    events: list[dict[str, Any]] = []
    file_errors: list[dict[str, Any]] = []
    unit_descriptions: set[str] = set()
    for path in files:
        relative = str(path.relative_to(root)).replace("\\", "/")
        try:
            table, descriptions, missing = _nwb_trial_table(path, columns)
            required = {
                "behavioral_epoch",
                "estim_current",
                "start_time",
                "stimulus_type",
                "stop_time",
            }
            if required & set(missing):
                file_errors.append(
                    {
                        "relative_path": relative,
                        "missing_columns": sorted(required & set(missing)),
                    }
                )
                continue
            if descriptions.get("estim_current"):
                unit_descriptions.add(descriptions["estim_current"])
            n_rows = int(table["_row_count"][0])
            valid_values = table.get("is_valid", [True] * n_rows)
            file_events: list[dict[str, Any]] = []
            for index in range(n_rows):
                if _clean_text(table["stimulus_type"][index]).lower() != "electrical":
                    continue
                onset = _finite_float(table["start_time"][index])
                offset = _finite_float(table["stop_time"][index])
                current = _numeric_text(table["estim_current"][index])
                if onset is None or offset is None or current is None:
                    continue
                file_events.append(
                    {
                        "onset": onset,
                        "offset": offset,
                        "rho": current,
                        "valid": _truthy(valid_values[index]),
                        "subject": path.parent.name,
                        "session": re.search(
                            r"(ses-[^_]+)",
                            path.name,
                        ).group(1)
                        if re.search(r"(ses-[^_]+)", path.name)
                        else path.stem,
                    }
                )
            file_events.sort(key=lambda event: event["onset"])
            for index, event in enumerate(file_events):
                next_onset = (
                    file_events[index + 1]["onset"]
                    if index + 1 < len(file_events)
                    else None
                )
                isolated = (
                    next_onset - event["offset"]
                    if next_onset is not None
                    else None
                )
                if event["valid"]:
                    events.append({**event, "isolated_post_sec": isolated})
        except (OSError, ValueError, KeyError, TypeError) as error:
            file_errors.append(
                {
                    "relative_path": relative,
                    "error": f"{type(error).__name__}:{error}",
                }
            )

    finite_isolation = [
        float(event["isolated_post_sec"])
        for event in events
        if event["isolated_post_sec"] is not None
        and event["isolated_post_sec"] >= 0
    ]
    levels = sorted({float(event["rho"]) for event in events})
    level_support: dict[float, set[str]] = defaultdict(set)
    for event in events:
        if (
            event["isolated_post_sec"] is not None
            and event["isolated_post_sec"] >= HARD_ISOLATED_POST_SEC
        ):
            level_support[float(event["rho"])].add(event["subject"])
    supported_levels = sorted(
        level
        for level, subjects in level_support.items()
        if len(subjects) >= MIN_SUBJECTS_PER_LEVEL
    )
    max_isolated = max(finite_isolation) if finite_isolation else None
    nmd_window = contract["nmd_window_sec"]
    classification, reason = _classification_reason(
        continuous=True,
        timing_known=bool(events),
        level_count=len(levels),
        supported_level_count=len(supported_levels),
        n_subjects=len({event["subject"] for event in events}),
        window_sec=nmd_window,
        max_isolated=max_isolated,
    )
    if (
        frozen_far003a_horizon_sec is not None
        and contract["nmd_window_sec"] is not None
        and frozen_far003a_horizon_sec < contract["nmd_window_sec"]
    ):
        classification = TIME_INCOMPATIBLE
        reason = (
            "FAR-003A frozen native support horizon is "
            f"{frozen_far003a_horizon_sec:g} s, below the active "
            f"{contract['nmd_window_sec']:g} s NMD window"
        )
    semantics = _empty_semantics()
    semantics.update(
        {
            "perturbation_explicitly_imposed": True,
            "rho_known": True,
            "rho_unit": "μA",
            "multiple_rho_levels": len(levels) >= 2,
            "rho_zero_or_sham": 0.0 in levels,
            "v_known": True,
            "onset_known": True,
            "offset_known": True,
            "pre_perturbation_baseline_observed": True,
            "post_perturbation_observation_present": True,
            "repeated_biological_units": len({event["subject"] for event in events})
            >= MIN_SUBJECTS_PER_LEVEL,
            "continuous_neural_signal": True,
            "isolated_post_horizon_sec": max_isolated,
        }
    )
    return {
        "dataset_id": "dandi_000458",
        "source_kind": "dandi_nwb_electrical_stimulation",
        "source_status": "AVAILABLE" if not file_errors else SOURCE_UNCERTAIN,
        "source_root": str(root),
        "native_nwb_file_count": len(files),
        "native_nwb_files": [
            {
                "relative_path": str(path.relative_to(root)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
            }
            for path in files
        ],
        "rho_field": "estim_current",
        "rho_unit": "μA",
        "rho_levels": levels,
        "rho_level_support_at_hard_minimum": {
            str(level): len(level_support[level]) for level in sorted(level_support)
        },
        "supported_rho_levels_at_hard_minimum": supported_levels,
        "n_valid_timed_events": len(events),
        "n_subjects": len({event["subject"] for event in events}),
        "n_sessions": len({event["session"] for event in events}),
        "isolated_post_horizon_sec": {
            "max": max_isolated,
            "median": median(finite_isolation) if finite_isolation else None,
            "n_events_ge_4s": sum(
                value >= HARD_ISOLATED_POST_SEC for value in finite_isolation
            ),
            "n_events_ge_8s": sum(
                value >= PREFERRED_ISOLATED_POST_SEC for value in finite_isolation
            ),
            "recording_end_known": False,
            "calculation": "next_valid_electrical_onset - current_valid_electrical_offset",
        },
        "unit_descriptions": sorted(unit_descriptions),
        "nmd_contract": contract,
        "frozen_far003a_native_horizon_sec": frozen_far003a_horizon_sec,
        "source_semantics": semantics,
        "classification": classification,
        "classification_reason": reason,
        "file_errors": file_errors,
        "audit_scope": {
            "opened_paths": ["/intervals/trials"],
            "forbidden_paths_not_opened": [
                "/acquisition/*/data",
                "/units/spike_times",
                "/processing",
                "NMD HDF5 outputs",
            ],
        },
    }


def _adapt_existing_metadata_audit(
    audit: dict[str, Any],
    *,
    config_path: Path,
    source_kind: str,
    continuous: bool,
) -> dict[str, Any]:
    contract = _source_contract(
        dataset_id=str(audit.get("dataset_id")),
        source_kind=source_kind,
        config_path=config_path,
    )
    classification = audit.get("classification", SOURCE_UNCERTAIN)
    if classification == FAR001_DIRECTION_ONLY:
        classification = DIRECTION_ONLY
    if not continuous and classification not in {DIRECTION_ONLY, NO_EXPLICIT}:
        classification = SOURCE_UNCERTAIN
    semantics = _empty_semantics()
    semantics.update(
        {
            "rho_known": audit.get("rho_known", False),
            "rho_unit": audit.get("rho_unit"),
            "multiple_rho_levels": len(audit.get("rho_levels", [])) >= 2,
            "v_known": classification == DIRECTION_ONLY,
            "continuous_neural_signal": continuous,
        }
    )
    adapted = dict(audit)
    adapted.update(
        {
            "nmd_contract": contract,
            "continuous_neural_signal_compatible": continuous,
            "source_semantics": semantics,
            "classification": classification,
            "classification_reason": (
                audit.get("classification_reason")
                or "existing metadata-only audit did not establish time-compatible amplitude"
            ),
        }
    )
    return adapted


def _unavailable_candidate(
    spec: Mapping[str, Any],
    *,
    config_path: Path,
) -> dict[str, Any]:
    roots = [Path(root) for root in spec["roots"]]
    existing = [str(root) for root in roots if root.is_dir()]
    contract = _source_contract(
        dataset_id=str(spec["dataset_id"]),
        source_kind=str(spec["source_kind"]),
        config_path=config_path,
    )
    semantics = _empty_semantics()
    semantics["continuous_neural_signal"] = contract[
        "continuous_neural_signal_compatible"
    ]
    return {
        "dataset_id": spec["dataset_id"],
        "source_kind": spec["source_kind"],
        "source_status": "AVAILABLE" if existing else "SOURCE_UNCERTAIN",
        "source_root": existing[0] if existing else None,
        "candidate_roots": [str(root) for root in roots],
        "nmd_contract": contract,
        "source_semantics": semantics,
        "classification": SOURCE_UNCERTAIN,
        "classification_reason": (
            "source root is unavailable or has no source-specific metadata "
            "adapter; no amplitude or time-scale claim is made from config alone"
        ),
    }


def run_inventory(
    *,
    repo_root: Path,
    source_roots: Mapping[str, Path] | None = None,
    protocol_path: Path,
    far003a_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_roots = source_roots or {}
    audits: list[dict[str, Any]] = []
    specs = _candidate_specs(repo_root)
    for spec in specs:
        dataset_id = str(spec["dataset_id"])
        config_path = Path(spec["config_path"])
        override = source_roots.get(dataset_id)
        candidate_roots = [Path(candidate) for candidate in spec["roots"]]
        if dataset_id == "dandi_000458" and override is None:
            candidate_roots.append(EXTERNAL_DANDI_000458_ROOT)
        roots = (
            [override]
            if override is not None
            else candidate_roots
        )
        root = _select_root(
            dataset_id=dataset_id,
            roots=[Path(candidate) for candidate in roots],
            override=override,
        )
        if root is None:
            updated = dict(spec)
            updated["roots"] = roots
            audits.append(_unavailable_candidate(updated, config_path=config_path))
            continue
        if dataset_id == "dandi_000458":
            audits.append(
                _audit_dandi_000458_time_compatibility(
                    root,
                    config_path,
                    frozen_far003a_horizon_sec=_frozen_far003a_native_horizon(
                        far003a_payload
                    ),
                )
            )
        elif dataset_id == "ds006036":
            audit = _scan_ds006036(root, config_path)
            audits.append(
                _adapt_existing_metadata_audit(
                    audit,
                    config_path=config_path,
                    source_kind=str(spec["source_kind"]),
                    continuous=True,
                )
            )
        elif dataset_id == "dandi_000009":
            audits.append(
                _adapt_existing_metadata_audit(
                    _audit_dandi_000009(root),
                    config_path=config_path,
                    source_kind=str(spec["source_kind"]),
                    continuous=True,
                )
            )
        elif dataset_id == "ds006623":
            audits.append(
                _adapt_existing_metadata_audit(
                    _audit_ds006623(root),
                    config_path=config_path,
                    source_kind=str(spec["source_kind"]),
                    continuous=False,
                )
            )
        elif dataset_id == "ds005917":
            audits.append(
                _adapt_existing_metadata_audit(
                    _audit_ds005917(root),
                    config_path=config_path,
                    source_kind=str(spec["source_kind"]),
                    continuous=False,
                )
            )
        else:
            updated = dict(spec)
            updated["roots"] = roots
            audits.append(_unavailable_candidate(updated, config_path=config_path))

    classifications = [str(audit["classification"]) for audit in audits]
    n_curve = classifications.count(TIME_COMPATIBLE_CURVE)
    n_point = classifications.count(TIME_COMPATIBLE_POINT)
    if n_curve:
        gate_status = "PASS"
        gate_reason = "time_compatible_curve_candidate_present"
    elif n_point:
        gate_status = "CONDITIONAL_PASS"
        gate_reason = "time_compatible_point_candidate_only"
    else:
        gate_status = "NOT_TESTABLE"
        gate_reason = "no_time_compatible_explicit_amplitude_candidate"
    return {
        "schema": "mndm.far_scout_002_time_compatible.v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": sha256_file(protocol_path),
        "gate_status": gate_status,
        "gate_reason": gate_reason,
        "hard_filters": {
            "explicit_rho_with_physical_units": True,
            "real_perturbation_onset": True,
            "minimum_nonzero_amplitudes": 1,
            "repeated_biological_units": True,
            "continuous_eeg_or_lfp_compatible_with_nmd": True,
            "hard_isolated_post_sec": HARD_ISOLATED_POST_SEC,
            "preferred_isolated_post_sec": PREFERRED_ISOLATED_POST_SEC,
            "preferred_rho_levels": MIN_CURVE_LEVELS,
            "source_sham_or_zero": "bonus",
            "multiple_operating_states": "bonus",
        },
        "datasets": audits,
        "classification_counts": {
            classification: classifications.count(classification)
            for classification in sorted(set(classifications))
        },
        "fail_closed_assertions": {
            "signal_payloads_opened": False,
            "nmd_outputs_opened": False,
            "downloads_performed": False,
            "frequency_promoted_to_rho": False,
            "condition_promoted_to_rho": False,
            "rho_reconstructed_from_nmd": False,
            "response_statistics_computed": False,
            "home_away_constructed": False,
            "models_fit": False,
            "fast_nmd_authorized": False,
        },
        "claim_boundary": (
            "FAR-SCOUT-002 classifies explicit perturbation metadata and "
            "source-defined isolated timing against the existing NMD window. "
            "It does not open signal payloads, validate artifact recovery, "
            "estimate resilience, or authorize FAR-003B."
        ),
    }
