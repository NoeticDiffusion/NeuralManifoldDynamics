"""Reusable orchestration for event-locked sidecar exports."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from mndm.schema import MNPSPayload

from .control_matching import ControlMatchResult, build_matched_controls
from .event_alignment import AlignmentResult, align_events_to_windows
from .event_annotations import (
    EventTable,
    event_table_from_stage_block_intervals,
    load_event_table_from_bids_events,
    load_event_table_from_csv,
)
from .event_locked_config import (
    EventLockedProfile,
    EventSourceConfig,
    _resolve_event_locked_dataset_cfg,
    alignment_config_from_profile,
    event_locked_profile_from_config,
    event_source_config_from_config,
    export_config_from_profile,
    matching_config_from_profile,
)
from .event_locked_export import (
    event_locked_export_manifest_entry,
    build_event_locked_table,
    write_event_locked_csv,
    write_event_locked_parquet,
)
from .stage_blocking import (
    infer_stage_block_intervals,
    normalize_block_key,
    stage_blocking_config_from_mapping,
)

logger = logging.getLogger(__name__)


@dataclass
class EventLockedRunResult:
    """Outputs from one event-locked export run."""

    profile: EventLockedProfile
    source_config: EventSourceConfig
    event_table: EventTable
    alignment: AlignmentResult
    controls: ControlMatchResult
    rows: List[Dict[str, Any]]
    manifest_entry: Dict[str, Any]
    output_paths: List[Path] = field(default_factory=list)


def _resolve_sampling_cfg(config: Mapping[str, Any], dataset_id: str) -> Mapping[str, Any]:
    epoching = config.get("epoching", {}) if isinstance(config, Mapping) else {}
    datasets = epoching.get("datasets", {}) if isinstance(epoching, Mapping) else {}
    ds_cfg = datasets.get(dataset_id, {}) if isinstance(datasets, Mapping) else {}
    sampling = ds_cfg.get("sampling", {}) if isinstance(ds_cfg, Mapping) else {}
    return sampling if isinstance(sampling, Mapping) else {}


def _resolve_stage_map(config: Mapping[str, Any], dataset_id: str) -> Dict[str, int]:
    sampling_cfg = _resolve_sampling_cfg(config, dataset_id)
    stage_map_raw = sampling_cfg.get("stage_map", {}) if isinstance(sampling_cfg.get("stage_map"), Mapping) else {}
    mnps_cfg = config.get("mnps", {}) if isinstance(config, Mapping) else {}
    codebook_raw = mnps_cfg.get("stage_codebook", {}) if isinstance(mnps_cfg.get("stage_codebook"), Mapping) else {}

    merged: Dict[str, int] = {}
    for raw_map in (stage_map_raw, codebook_raw):
        for key, value in raw_map.items():
            try:
                merged[normalize_block_key(key)] = int(value)
            except Exception:
                continue
    return merged


def _resolve_stage_codes(labels_raw: Sequence[Any], stage_map: Mapping[str, int]) -> np.ndarray:
    stage_codes = np.full((len(labels_raw),), np.nan, dtype=np.float64)
    for idx, raw in enumerate(labels_raw):
        key = normalize_block_key(raw)
        if key in stage_map:
            stage_codes[idx] = float(stage_map[key])
            continue
        try:
            stage_codes[idx] = float(raw)
        except Exception:
            stage_codes[idx] = np.nan
    return stage_codes


def _load_tabular_events(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=sep)


def _filter_stage_block_intervals(
    intervals: Sequence[Any],
    *,
    stage_codes: Sequence[int] = (),
    block_parameters: Sequence[float] = (),
) -> List[Any]:
    out: List[Any] = []
    allowed_stage_codes = {int(v) for v in stage_codes}
    allowed_block_params = {float(v) for v in block_parameters}
    for interval in intervals:
        if allowed_stage_codes and int(getattr(interval, "stage_code", -1)) not in allowed_stage_codes:
            continue
        if allowed_block_params:
            block_parameter = float(getattr(interval, "block_parameter", np.nan))
            if not np.isfinite(block_parameter) or block_parameter not in allowed_block_params:
                continue
        out.append(interval)
    return out


def resolve_event_table_for_event_locked(
    *,
    config: Mapping[str, Any],
    dataset_id: str,
    source_path: Optional[Path] = None,
    event_table: Optional[EventTable] = None,
) -> tuple[EventSourceConfig, EventTable]:
    """Resolve the EventTable used by event-locked analysis."""
    stage_map = _resolve_stage_map(config, dataset_id)
    source_cfg = event_source_config_from_config(config, dataset_id, stage_codebook=stage_map)

    if source_cfg.kind == "csv":
        if event_table is not None:
            return source_cfg, event_table
        effective_path = Path(source_cfg.source_path) if source_cfg.source_path else source_path
        if effective_path is None:
            raise ValueError("CSV event source requires either event_table or source_path.")
        table = load_event_table_from_csv(effective_path, event_type_filter=source_cfg.event_type or None)
        return source_cfg, table

    if source_cfg.kind == "bids_events":
        # Direct BIDS events.tsv event-locking: no derived:task_state_label required.
        # Event onsets are read straight from the companion *_events.tsv file,
        # filtered by the ``event_types`` list configured under
        # ``event_locked.datasets.<id>.event_types``.
        effective_path = Path(source_cfg.source_path) if source_cfg.source_path else source_path
        if effective_path is None:
            raise ValueError(
                "bids_events event source requires source_path to a BIDS *_events.tsv file."
            )
        ds_cfg = _resolve_event_locked_dataset_cfg(config, dataset_id)
        event_types_raw = ds_cfg.get("event_types", [])
        if not isinstance(event_types_raw, list):
            event_types_raw = [event_types_raw]
        event_types = [str(t).strip() for t in event_types_raw if str(t).strip()]
        event_source_raw = (
            ds_cfg.get("event_source", {})
            if isinstance(ds_cfg.get("event_source"), Mapping)
            else {}
        )
        trial_type_col = str(event_source_raw.get("trial_type_column", "trial_type") or "trial_type")
        onset_col = str(event_source_raw.get("onset_column", "onset") or "onset")
        duration_col = str(event_source_raw.get("duration_column", "duration") or "duration")
        exclude_types_raw = event_source_raw.get("exclude_types", []) or []
        exclude_types = [str(t).strip() for t in (exclude_types_raw if isinstance(exclude_types_raw, list) else [exclude_types_raw])]
        table = load_event_table_from_bids_events(
            path=effective_path,
            event_types=event_types or None,
            trial_type_column=trial_type_col,
            onset_column=onset_col,
            duration_column=duration_col,
            exclude_types=exclude_types or None,
        )
        return source_cfg, table

    if source_cfg.kind != "derived_stage_block_end":
        raise ValueError(f"Unsupported event_locked event_source.kind '{source_cfg.kind}'.")

    effective_path = Path(source_cfg.source_path) if source_cfg.source_path else source_path
    if effective_path is None:
        raise ValueError("derived_stage_block_end requires source_path to a raw events TSV/CSV.")
    if not effective_path.exists():
        raise FileNotFoundError(f"Raw event source not found: {effective_path}")

    sampling_cfg = _resolve_sampling_cfg(config, dataset_id)
    onset_col = str(sampling_cfg.get("onset_column", "onset") or "onset")
    duration_col = str(sampling_cfg.get("duration_column", "duration") or "duration")
    stage_columns = sampling_cfg.get("stage_columns", ["trial_type", "value", "event_type", "label"])
    if not isinstance(stage_columns, list):
        stage_columns = ["trial_type", "value", "event_type", "label"]

    events_df = _load_tabular_events(effective_path)
    if onset_col not in events_df.columns:
        raise ValueError(f"Missing onset column '{onset_col}' in {effective_path}")
    stage_col = next((col for col in stage_columns if col in events_df.columns), None)
    if stage_col is None:
        raise ValueError(f"Could not resolve any stage column from {stage_columns} in {effective_path}")

    onsets = pd.to_numeric(events_df[onset_col], errors="coerce").to_numpy(dtype=np.float64)
    if duration_col in events_df.columns:
        durations = pd.to_numeric(events_df[duration_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    else:
        durations = np.zeros((len(events_df),), dtype=np.float64)
    labels_raw = events_df[stage_col].where(events_df[stage_col].notna(), "").tolist()
    stage_codes = _resolve_stage_codes(labels_raw, stage_map)

    stage_blocking_cfg = stage_blocking_config_from_mapping(sampling_cfg.get("stage_blocking", {}))
    intervals = infer_stage_block_intervals(
        onsets=onsets,
        durations=durations,
        labels_raw=labels_raw,
        stage_codes=stage_codes,
        cfg=stage_blocking_cfg,
    )
    intervals = _filter_stage_block_intervals(
        intervals,
        stage_codes=source_cfg.stage_codes,
        block_parameters=source_cfg.block_parameters,
    )
    table = event_table_from_stage_block_intervals(
        intervals,
        event_kind=source_cfg.event_kind,
        event_type=source_cfg.event_type,
        source=source_cfg.source_label,
        source_path=str(effective_path),
    )
    return source_cfg, table


def run_event_locked_export(
    *,
    payload: MNPSPayload,
    config: Mapping[str, Any],
    dataset_id: str,
    source_path: Optional[Path] = None,
    event_table: Optional[EventTable] = None,
    subject_id: str = "",
    session_id: str = "",
    run_id: str = "",
    out_prefix: Optional[Path] = None,
) -> EventLockedRunResult:
    """Resolve, align, match, and export one event-locked analysis sidecar."""
    profile = event_locked_profile_from_config(config, dataset_id)
    source_cfg, resolved_table = resolve_event_table_for_event_locked(
        config=config,
        dataset_id=dataset_id,
        source_path=source_path,
        event_table=event_table,
    )

    alignment = align_events_to_windows(
        resolved_table,
        time=payload.time,
        stage=payload.stage,
        window_start=payload.window_start,
        window_end=payload.window_end,
        config=alignment_config_from_profile(profile, config=config, dataset_id=dataset_id),
    )
    controls = build_matched_controls(
        resolved_table,
        time=payload.time,
        window_start=payload.window_start,
        window_end=payload.window_end,
        stage=payload.stage,
        config=matching_config_from_profile(profile),
    )
    export_cfg = export_config_from_profile(profile, config=config, dataset_id=dataset_id)
    rows = build_event_locked_table(
        payload=payload,
        alignment=alignment,
        controls=controls,
        event_table=resolved_table,
        subject_id=subject_id,
        session_id=session_id,
        run_id=run_id,
        dataset_id=dataset_id,
        config=export_cfg,
    )

    output_paths: List[Path] = []
    if out_prefix is not None:
        out_prefix = Path(out_prefix)
        if export_cfg.write_parquet:
            parquet_path = out_prefix.with_suffix(".parquet")
            parquet_written = write_event_locked_parquet(rows, parquet_path)
            if parquet_written is not None:
                output_paths.append(parquet_written)
        if export_cfg.write_csv:
            csv_path = out_prefix.with_suffix(".csv")
            csv_written = write_event_locked_csv(rows, csv_path)
            if csv_written is not None:
                output_paths.append(csv_written)

    manifest_entry = event_locked_export_manifest_entry(
        rows,
        alignment=alignment,
        controls=controls,
        out_paths=output_paths,
    )
    manifest_entry["event_source"] = source_cfg.to_dict()
    manifest_entry["profile"] = profile.to_dict()

    logger.info(
        "Event-locked run complete: dataset=%s source_kind=%s rows=%d",
        dataset_id,
        source_cfg.kind,
        len(rows),
    )
    return EventLockedRunResult(
        profile=profile,
        source_config=source_cfg,
        event_table=resolved_table,
        alignment=alignment,
        controls=controls,
        rows=rows,
        manifest_entry=manifest_entry,
        output_paths=output_paths,
    )
