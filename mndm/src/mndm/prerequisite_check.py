"""Preflight checks for new MNDM dataset/config setups."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd
from core.paths import resolve_paths

from . import bids_index
from .file_filters import apply_exclude_file_filters, resolve_exclude_file_patterns

@dataclass
class CheckItem:
    name: str
    status: str
    message: str
    details: Dict[str, Any]


@dataclass
class DatasetPrerequisiteReport:
    dataset_id: str
    ok: bool
    checks: List[CheckItem]


def _check(name: str, status: str, message: str, **details: Any) -> CheckItem:
    """Internal helper: check."""
    return CheckItem(name=name, status=status, message=message, details=details)


def _has_fail(checks: List[CheckItem]) -> bool:
    """Internal helper: has fail."""
    return any(item.status == "fail" for item in checks)


def _normalize_counts(mapping: Mapping[str, Any]) -> Dict[str, int]:
    """Internal helper: normalize counts."""
    out: Dict[str, int] = {}
    for key, value in mapping.items():
        try:
            out[str(key)] = int(value)
        except Exception:
            continue
    return out


def _validate_exclude_patterns(patterns: list[str]) -> CheckItem:
    """Internal helper: validate exclude patterns."""
    if not patterns:
        return _check("exclude_files", "ok", "No exclude-files configured.", patterns=[])
    invalid: list[dict[str, str]] = []
    for pattern in patterns:
        try:
            re.compile(pattern)
        except re.error as exc:
            invalid.append({"pattern": pattern, "error": str(exc)})
    if invalid:
        return _check(
            "exclude_files",
            "fail",
            "One or more exclude-files regex patterns are invalid.",
            patterns=patterns,
            invalid=invalid,
        )
    return _check(
        "exclude_files",
        "ok",
        "Exclude-files regex patterns compiled successfully.",
        patterns=patterns,
    )


def _check_required_sections(config: Mapping[str, Any]) -> CheckItem:
    """Internal helper: check required sections."""
    required = ["paths", "epoching", "mnps_projection", "mnps"]
    missing = [name for name in required if not isinstance(config.get(name), Mapping)]
    if missing:
        return _check(
            "config_sections",
            "fail",
            "Required config sections are missing.",
            missing=missing,
        )
    return _check("config_sections", "ok", "Required config sections are present.", missing=[])


def _load_participants_from_dataset_root(
    dataset_root: Path,
    config: Mapping[str, Any],
    dataset_id: str,
) -> Optional[pd.DataFrame]:
    """Internal helper: load participants via shared extraction pipeline."""
    from .pipeline.extractors import load_participant_table

    cfg = dict(config) if isinstance(config, Mapping) else {}
    paths_cfg = dict(cfg.get("paths", {})) if isinstance(cfg.get("paths", {}), Mapping) else {}
    ds_dirs = dict(paths_cfg.get("dataset_received_dirs", {})) if isinstance(paths_cfg.get("dataset_received_dirs", {}), Mapping) else {}
    ds_dirs[dataset_id] = str(dataset_root)
    paths_cfg["dataset_received_dirs"] = ds_dirs
    cfg["paths"] = paths_cfg
    return load_participant_table(dataset_root.parent, dataset_id, cfg)


def run_dataset_prerequisite_check(
    *,
    config: Mapping[str, Any],
    dataset_id: str,
    out_dir: Path | None = None,
    data_dir: Path | None = None,
) -> DatasetPrerequisiteReport:
    """Handle run dataset prerequisite check."""
    checks: List[CheckItem] = []

    received_dir, processed_dir = resolve_paths(config, out_dir, data_dir)
    checks.append(
        _check(
            "paths",
            "ok",
            "Resolved received/processed directories.",
            received_dir=str(received_dir),
            processed_dir=str(processed_dir),
        )
    )
    checks.append(_check_required_sections(config))

    configured_datasets = config.get("datasets", []) if isinstance(config, Mapping) else []
    configured_dataset_ids = [str(item) for item in configured_datasets if isinstance(item, str)]
    if configured_dataset_ids and dataset_id not in configured_dataset_ids:
        checks.append(
            _check(
                "dataset_membership",
                "warn",
                "Dataset was requested on the CLI but is not listed in config.datasets.",
                configured_datasets=configured_dataset_ids,
            )
        )
    else:
        checks.append(
            _check(
                "dataset_membership",
                "ok",
                "Dataset is present in config.datasets or config.datasets is empty.",
                configured_datasets=configured_dataset_ids,
            )
        )

    patterns = resolve_exclude_file_patterns(config)
    checks.append(_validate_exclude_patterns(patterns))

    dataset_root = bids_index.resolve_dataset_root(config, Path(received_dir), dataset_id)
    if not dataset_root.exists():
        checks.append(
            _check(
                "dataset_root",
                "fail",
                "Resolved dataset root does not exist.",
                dataset_root=str(dataset_root),
            )
        )
        return DatasetPrerequisiteReport(dataset_id=dataset_id, ok=not _has_fail(checks), checks=checks)
    if not dataset_root.is_dir():
        checks.append(
            _check(
                "dataset_root",
                "fail",
                "Resolved dataset root is not a directory.",
                dataset_root=str(dataset_root),
            )
        )
        return DatasetPrerequisiteReport(dataset_id=dataset_id, ok=not _has_fail(checks), checks=checks)
    checks.append(
        _check(
            "dataset_root",
            "ok",
            "Resolved dataset root exists.",
            dataset_root=str(dataset_root),
        )
    )

    try:
        participants_df = _load_participants_from_dataset_root(dataset_root, config, dataset_id)
    except Exception:
        participants_df = None
    if participants_df is None:
        checks.append(
            _check(
                "participants_table",
                "warn",
                "No readable participants table was found.",
                searched_under=str(dataset_root),
            )
        )
    else:
        checks.append(
            _check(
                "participants_table",
                "ok",
                "Participants table loaded successfully.",
                rows=int(len(participants_df)),
                source_path=str(participants_df.attrs.get("source_path", "")),
                source_format=str(participants_df.attrs.get("source_format", "")),
                subject_id_column=str(participants_df.attrs.get("subject_id_column", "")),
            )
        )

    processed_ds = Path(processed_dir) / dataset_id
    file_index_path = processed_ds / "file_index.csv"
    if file_index_path.exists():
        checks.append(
            _check(
                "file_index",
                "ok",
                "Existing file_index.csv found in processed directory.",
                path=str(file_index_path),
            )
        )
    else:
        checks.append(
            _check(
                "file_index",
                "warn",
                "No existing file_index.csv found yet; prerequisite check will build an in-memory preview only.",
                expected_path=str(file_index_path),
            )
        )

    try:
        preview_df = bids_index.build_file_index(dataset_root, config=config, dataset_id=dataset_id)
        preview_total = int(len(preview_df))
        filtered_df, excluded_rows, _ = apply_exclude_file_filters(
            preview_df,
            config=config,
            candidate_columns=("path",),
        )
        preview_modalities = (
            _normalize_counts(filtered_df["modality"].value_counts().to_dict())
            if "modality" in filtered_df.columns
            else {}
        )
        preview_subjects = int(filtered_df["subject"].nunique()) if "subject" in filtered_df.columns and not filtered_df.empty else 0
        if filtered_df.empty:
            checks.append(
                _check(
                    "index_preview",
                    "fail",
                    "Index preview found no usable files after applying exclude-files.",
                    total_before_exclude=preview_total,
                    excluded_by_pattern=excluded_rows,
                    modalities={},
                    subjects=0,
                )
            )
        else:
            checks.append(
                _check(
                    "index_preview",
                    "ok",
                    "Index preview found usable dataset files.",
                    total_before_exclude=preview_total,
                    total_after_exclude=int(len(filtered_df)),
                    excluded_by_pattern=excluded_rows,
                    modalities=preview_modalities,
                    subjects=preview_subjects,
                )
            )
    except Exception as exc:
        checks.append(
            _check(
                "index_preview",
                "fail",
                "Failed to build dataset index preview.",
                error=f"{type(exc).__name__}: {exc}",
            )
        )

    return DatasetPrerequisiteReport(dataset_id=dataset_id, ok=not _has_fail(checks), checks=checks)


def run_prerequisite_check(
    *,
    config: Mapping[str, Any],
    dataset_ids: list[str],
    out_dir: Path | None = None,
    data_dir: Path | None = None,
) -> Dict[str, Any]:
    """Handle run prerequisite check."""
    reports = [
        run_dataset_prerequisite_check(
            config=config,
            dataset_id=dataset_id,
            out_dir=out_dir,
            data_dir=data_dir,
        )
        for dataset_id in dataset_ids
    ]
    ok = all(report.ok for report in reports) if reports else False
    return {
        "ok": ok,
        "datasets": [asdict(report) for report in reports],
    }


def format_prerequisite_report(report: Mapping[str, Any]) -> str:
    """Render a compact human-readable preflight report."""
    lines: list[str] = []
    lines.append(f"overall_ok: {bool(report.get('ok', False))}")
    for ds_report in report.get("datasets", []) or []:
        ds_id = ds_report.get("dataset_id", "unknown")
        ds_ok = bool(ds_report.get("ok", False))
        lines.append(f"[{ds_id}] ok={ds_ok}")
        for check in ds_report.get("checks", []) or []:
            name = str(check.get("name", "unknown"))
            status = str(check.get("status", "unknown"))
            message = str(check.get("message", ""))
            lines.append(f"  - {status}: {name}: {message}")
    return "\n".join(lines)


def report_to_json(report: Mapping[str, Any]) -> str:
    """Handle report to json."""
    return json.dumps(report, indent=2, ensure_ascii=False)
