""" 
Summary.py
Dataset- and subject-level MNPS summarization runners.
"""

from __future__ import annotations

import copy
import multiprocessing
import hashlib
import json
import logging
import platform
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import re
from threading import Lock
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from core.bids import parse_subject_session, parse_subject_session_task_run_acq
from .context import SummarizeContext
from .. import bids_index
from ..file_filters import apply_exclude_file_filters
from .extractors import (
    build_dataset_label,
    derive_pseudo_stage_array,
    extract_embodied_array,
    extract_events,
    extract_mapped_metadata,
    extract_stage_array,
    load_participant_table,
)
from .anchor_state import build_anchor_state_exports
from .extensions_compute import compute_extensions
from .regional_mnps import (
    compute_block_jacobian_rows,
    summary_to_dataframe_rows,
)
from .summary_regional import compute_regional_context
from .stratified_blocks import (
    compute_stratified_blocks_and_cross_partials,
)
from .summary_io import (
    write_regional_csv_outputs,
    write_stratified_blocks_csv_output,
    write_summary_manifest_and_h5,
)
from .summary_selectors import (
    load_regional_fmri_signals,
    resolve_bold_path_for_subframe,
)
from .summary_events import (
    infer_stage_from_bids_events,
    build_bids_event_stage_provenance,
    estimate_coverage_seconds,
    map_events_to_labels,
)
from .event_alignment import AlignmentConfig, align_events_to_windows
from .event_annotations import EventTable
from .state_labels import (
    build_label_segment_event_table,
    build_within_run_labels,
    summarize_within_run_manifest,
)
from .summary_qc import write_qc_files
from .summary_utils import (
    apply_fd_censoring,
    build_dir_suffix,
    extract_time_bounds,
)
from .time_reference import build_time_reference_for_run
from .conventional_summary import compute_conventional_eeg_summary
from .baseline_qc import (
    compute_feature_baseline_comparisons,
    compute_null_sanity_tests,
)
from .robustness_helpers import (
    STANDARD_GEOMETRY_POLICY_VERSION,
    STANDARD_GEOMETRY_MAX_JACOBIAN_CONDITION,
    apply_anchor_coupling_window_policy,
    apply_standard_jacobian_window_policy,
    compute_dist_summary,
    compute_emmi_metrics,
    compute_mnps_mnj_sanity,
    compute_window_time_audit,
    compute_standard_geometry_contract,
    compute_tau_summary,
    compute_tier2_jacobian_metrics,
    compute_ensemble_summary_for_subject,
    compute_psd_multiverse_stability,
    compute_robust_and_reliability_summaries,
)
from .. import nwb_intervals, preprocess
from core.io import json_writer
from .. import anchors, jacobian, projection, robustness, schema
from .run_manifest import write_run_manifest
from ..reproducibility import resolve_reproducibility_policy

logger = logging.getLogger(__name__)


class _RunnerContextProxy:
    """Per-dataset context wrapper with an overrideable config mapping."""

    def __init__(self, base: Any, config: Mapping[str, Any]):
        self._base = base
        self.config = config

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def _stable_hash_mapping(value: Mapping[str, Any]) -> str:
    """Hash a mapping deterministically for provenance/versioning."""
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    except Exception:
        payload = str(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_hash_array(value: np.ndarray) -> str:
    """Hash an ndarray deterministically for provenance checks."""
    arr = np.ascontiguousarray(value)
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()


def _resolve_event_locked_source_kind(event_locked_cfg: Mapping[str, Any]) -> str:
    """Return normalized event source kind from dataset-level event_locked config."""
    source_cfg = (
        event_locked_cfg.get("event_source", {})
        if isinstance(event_locked_cfg.get("event_source"), Mapping)
        else {}
    )
    kind = str(source_cfg.get("kind", "csv") or "csv").strip().lower()
    return kind or "csv"


def _event_locked_slug_from_csv_path(csv_path: Path) -> str:
    """Derive legacy channel slug from spindle CSV filename when possible."""
    marker = "_spindles_yasa_v1_"
    stem = csv_path.stem
    if marker not in stem:
        return ""
    slug = stem.split(marker, 1)[1].strip().lower()
    return slug


def _safe_filename_token(value: Any) -> str:
    """Return a conservative token safe for filesystem filenames."""
    text = str(value).strip()
    if not text:
        return ""
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    return text.strip("_")


def _resolve_event_locked_csv_sources(
    *,
    stage_events_path: Optional[str],
    event_locked_cfg: Mapping[str, Any],
) -> List[tuple[Path, str]]:
    """Resolve per-run CSV annotation sources for event-locked export.

    Resolution order:
    1) ``event_source.source_path`` when configured.
    2) ``csv_source_glob`` / ``csv_source_globs`` dataset-level hints.
    3) Sleep-spindle default sibling pattern next to ``*_events.tsv``.
    """
    if not stage_events_path:
        return []

    events_path = Path(str(stage_events_path))
    events_dir = events_path.parent
    events_stem = events_path.stem
    events_core = events_stem[:-7] if events_stem.endswith("_events") else events_stem

    source_cfg = (
        event_locked_cfg.get("event_source", {})
        if isinstance(event_locked_cfg.get("event_source"), Mapping)
        else {}
    )
    patterns: List[str] = []

    source_path_raw = str(source_cfg.get("source_path", "") or "").strip()
    if source_path_raw:
        patterns.append(source_path_raw)

    glob_one = event_locked_cfg.get("csv_source_glob")
    if isinstance(glob_one, str) and glob_one.strip():
        patterns.append(glob_one.strip())

    glob_many = event_locked_cfg.get("csv_source_globs", [])
    if isinstance(glob_many, Sequence) and not isinstance(glob_many, (str, bytes)):
        for item in glob_many:
            text = str(item).strip()
            if text:
                patterns.append(text)

    if not patterns:
        event_types_raw = event_locked_cfg.get("event_types", [])
        if not isinstance(event_types_raw, list):
            event_types_raw = [event_types_raw]
        event_types = {str(v).strip().lower() for v in event_types_raw if str(v).strip()}
        if "sleep_spindle" in event_types:
            patterns.append("{events_core}_spindles_yasa_v1_*.csv")

    discovered: List[Path] = []
    seen: set[str] = set()
    format_vars = {
        "events_dir": str(events_dir).replace("\\", "/"),
        "events_stem": events_stem,
        "events_core": events_core,
    }

    for raw_pattern in patterns:
        try:
            rendered = str(raw_pattern).format(**format_vars).strip()
        except Exception:
            rendered = str(raw_pattern).strip()
        if not rendered:
            continue

        candidates: List[Path] = []
        if any(ch in rendered for ch in "*?[]"):
            rendered_path = Path(rendered)
            if rendered_path.is_absolute():
                parent = rendered_path.parent
                if parent.exists():
                    candidates = sorted(parent.glob(rendered_path.name))
            else:
                candidates = sorted(events_dir.glob(rendered))
        else:
            candidate = Path(rendered)
            if not candidate.is_absolute():
                candidate = events_dir / candidate
            if candidate.exists():
                candidates = [candidate]

        for candidate in candidates:
            try:
                key = str(candidate.resolve())
            except Exception:
                key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            discovered.append(candidate)

    return [(path, _event_locked_slug_from_csv_path(path)) for path in discovered]


def _resolve_meg_mapping_contract(config: Mapping[str, Any], dataset_id: Optional[str]) -> Dict[str, Any]:
    """Resolve optional MEG shadow-mapping metadata with dataset overrides."""
    root = config.get("meg_mapping", {}) if isinstance(config, Mapping) else {}
    if not isinstance(root, Mapping):
        return {}
    merged: Dict[str, Any] = {k: root[k] for k in root if k != "datasets"}
    ds_map = root.get("datasets", {})
    if dataset_id and isinstance(ds_map, Mapping):
        ds_cfg = ds_map.get(dataset_id)
        if isinstance(ds_cfg, Mapping):
            merged.update(dict(ds_cfg))
    if not merged:
        return {}
    if merged.get("enabled", None) is False:
        return {}
    return merged


def _resolve_anchor_path(path_raw: Any, *, config: Mapping[str, Any]) -> Optional[Path]:
    """Resolve a configured feature-anchor path."""
    if path_raw is None:
        return None
    text = str(path_raw).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    base = config.get("config_dir") if isinstance(config, Mapping) else None
    if base:
        candidate = Path(str(base)) / path
        if candidate.exists():
            return candidate
    return Path.cwd() / path


def _rows_to_columnar_table(rows: list[Mapping[str, Any]]) -> Dict[str, np.ndarray]:
    """Convert row dicts into a columnar mapping suitable for HDF5 datasets."""
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    out: Dict[str, np.ndarray] = {}
    for col in frame.columns:
        series = frame[col]
        if pd.api.types.is_numeric_dtype(series):
            out[str(col)] = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32)
        else:
            out[str(col)] = series.fillna("").astype(str).to_numpy(dtype=str)
    return out


def _decode_text_scalar(value: Any) -> Optional[str]:
    """Decode bytes/NumPy byte scalars into plain text."""
    if value is None:
        return None
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    if isinstance(value, bytes):
        return value.decode("utf-8")
    text = str(value)
    return text if text else None


def _columnar_records_to_arrays(records: Sequence[Mapping[str, Any]]) -> Dict[str, np.ndarray]:
    """Convert flat row dicts into dtype-stable 1-D arrays."""
    if not records:
        return {}
    keys: list[str] = []
    seen: set[str] = set()
    for row in records:
        for key in row.keys():
            skey = str(key)
            if skey not in seen:
                seen.add(skey)
                keys.append(skey)
    out: Dict[str, np.ndarray] = {}
    for key in keys:
        values = [row.get(key) for row in records]
        non_null = [v for v in values if v is not None]
        if non_null and all(isinstance(v, (bool, np.bool_)) for v in non_null):
            out[key] = np.asarray(values, dtype=np.int8)
        elif non_null and all(isinstance(v, (int, np.integer, bool, np.bool_)) for v in non_null):
            out[key] = np.asarray(values, dtype=np.int32)
        elif non_null and all(isinstance(v, (int, float, np.integer, np.floating, bool, np.bool_)) for v in non_null):
            out[key] = np.asarray(values, dtype=np.float32)
        else:
            out[key] = np.asarray(["" if v is None else str(v) for v in values], dtype=object)
    return out


def _stage_label_to_bool_label_name(raw_label: Any) -> str:
    """Return a concise per-window label name for a stage codebook entry."""
    text = re.sub(r"\s+", " ", str(raw_label or "").strip().lower())
    if not text:
        return "unknown_stage"
    if "eyes closed" in text:
        return "eyes_closed"
    if "eyes open" in text:
        return "eyes_open"
    if text in {"w", "wake"}:
        return "wake"
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_") or "unknown_stage"


def _build_stage_bool_labels(
    stage: Optional[np.ndarray],
    stage_codebook: Mapping[str, Any],
) -> Dict[str, np.ndarray]:
    """Export stage-codebook membership as explicit per-window bool labels."""
    if stage is None or not isinstance(stage_codebook, Mapping) or not stage_codebook:
        return {}
    stage_arr = np.asarray(stage, dtype=np.int32)
    out: Dict[str, np.ndarray] = {}
    used_names: set[str] = set()
    for raw_label, raw_code in stage_codebook.items():
        try:
            code = int(raw_code)
        except Exception:
            continue
        label_name = _stage_label_to_bool_label_name(raw_label)
        if label_name in used_names:
            continue
        mask = (stage_arr == code).astype(np.int8)
        if int(mask.sum()) <= 0:
            continue
        out[label_name] = mask
        used_names.add(label_name)
    return out


def _build_stage_codebook_export(
    stage_codebook: Mapping[str, Any],
    *,
    stage_source: Optional[str],
    stage_column: Optional[str],
    stage_events_path: Optional[str],
) -> Dict[str, Any]:
    """Build the `/codebooks/stage` payload."""
    if not isinstance(stage_codebook, Mapping) or not stage_codebook:
        return {}
    rows: list[tuple[int, str, str]] = []
    for raw_label, raw_code in stage_codebook.items():
        try:
            code = int(raw_code)
        except Exception:
            continue
        label = str(raw_label)
        rows.append((code, label, _stage_label_to_bool_label_name(label)))
    rows.sort(key=lambda item: (item[0], item[1]))
    if not rows:
        return {}
    return {
        "stage": {
            "codes": np.asarray([code for code, _, _ in rows], dtype=np.int32),
            "labels": [label for _, label, _ in rows],
            "label_keys": [label_key for _, _, label_key in rows],
            "attrs": {
                "source": stage_source,
                "column": stage_column,
                "events_path": stage_events_path,
            },
        }
    }


def _build_event_windows_export(
    *,
    event_table_columns: Mapping[str, Any],
    events_path: Optional[str],
    time: np.ndarray,
    window_start: np.ndarray,
    window_end: np.ndarray,
    stage: Optional[np.ndarray],
    window_sec: float,
    overlap: float,
) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Build an explicit event->window alignment table for H5 export."""
    onset_col = event_table_columns.get("onset_sec")
    if onset_col is None:
        return {}, {}
    onsets = np.asarray(onset_col, dtype=np.float64)
    if onsets.ndim != 1 or onsets.size == 0:
        return {}, {}
    duration_col = event_table_columns.get("duration_sec")
    durations = np.asarray(duration_col, dtype=np.float64) if duration_col is not None else None
    raw_labels = (
        np.asarray(event_table_columns.get("raw_event_label"), dtype=object)
        if event_table_columns.get("raw_event_label") is not None
        else np.asarray([""] * len(onsets), dtype=object)
    )
    normalized_labels = (
        np.asarray(event_table_columns.get("normalized_event_label"), dtype=object)
        if event_table_columns.get("normalized_event_label") is not None
        else np.asarray([""] * len(onsets), dtype=object)
    )
    source_event_column = (
        np.asarray(event_table_columns.get("source_event_column"), dtype=object)
        if event_table_columns.get("source_event_column") is not None
        else np.asarray([""] * len(onsets), dtype=object)
    )
    mapped_stage_code = (
        np.asarray(event_table_columns.get("mapped_stage_code"), dtype=np.float64)
        if event_table_columns.get("mapped_stage_code") is not None
        else None
    )
    event_table = EventTable(
        onset_sec=onsets,
        duration_sec=durations,
        event_type=raw_labels,
        source=np.asarray(["bids_event_provenance"] * len(onsets), dtype=object),
        source_path=str(events_path or "bids_event_provenance"),
        n_events_loaded=int(len(onsets)),
    )
    align_cfg = AlignmentConfig(reference="onset", stage_transition_margin_sec=0.0)
    alignment = align_events_to_windows(
        event_table,
        window_start=np.asarray(window_start, dtype=np.float64),
        window_end=np.asarray(window_end, dtype=np.float64),
        time=np.asarray(time, dtype=np.float64),
        stage=np.asarray(stage, dtype=np.int32) if stage is not None else None,
        config=align_cfg,
    )
    records = alignment.to_records()
    if not records:
        return {}, {
            "_schema_version": "mndm.event_windows.v1",
            "reference": align_cfg.reference,
            "bins_json": json.dumps(
                [{"label": b.label, "lo": b.lo, "hi": b.hi} for b in align_cfg.bins],
                ensure_ascii=False,
            ),
            "source_events_path": events_path,
            "source_event_table_schema": _decode_text_scalar(event_table_columns.get("_schema_version")),
            "window_sec": float(window_sec),
            "window_step_sec": float(window_sec * (1.0 - overlap)),
            "n_rows": 0,
        }
    for row in records:
        ev_idx = int(row["event_id"])
        w_idx = int(row["window_id"])
        event_onset = float(onsets[ev_idx]) if ev_idx < len(onsets) else np.nan
        event_duration = float(durations[ev_idx]) if durations is not None and ev_idx < len(durations) else 0.0
        event_stop = event_onset + event_duration if np.isfinite(event_duration) and event_duration > 0 else event_onset
        containing_mask = (window_start <= event_onset) & (window_end > event_onset)
        overlapping_mask = (window_end > event_onset) & (window_start < event_stop) if event_stop > event_onset else containing_mask
        start_window_indices = np.where(containing_mask)[0]
        stop_window_indices = np.where(overlapping_mask)[0]
        row["event_label"] = str(raw_labels[ev_idx]) if ev_idx < len(raw_labels) else ""
        row["event_label_key"] = str(normalized_labels[ev_idx]) if ev_idx < len(normalized_labels) else ""
        row["event_onset_sec"] = event_onset
        row["event_duration_sec"] = float(durations[ev_idx]) if durations is not None and ev_idx < len(durations) else np.nan
        row["event_mapped_stage_code"] = (
            float(mapped_stage_code[ev_idx]) if mapped_stage_code is not None and ev_idx < len(mapped_stage_code) else np.nan
        )
        row["source_event_column"] = str(source_event_column[ev_idx]) if ev_idx < len(source_event_column) else ""
        row["window_start_sec"] = float(window_start[w_idx]) if w_idx < len(window_start) else np.nan
        row["window_end_sec"] = float(window_end[w_idx]) if w_idx < len(window_end) else np.nan
        row["window_contains_event_onset"] = int(bool(w_idx < len(containing_mask) and containing_mask[w_idx]))
        row["event_start_window_index"] = int(start_window_indices[0]) if start_window_indices.size else -1
        row["event_stop_window_index"] = int(stop_window_indices[-1]) if stop_window_indices.size else -1
    return _columnar_records_to_arrays(records), {
        "_schema_version": "mndm.event_windows.v1",
        "reference": align_cfg.reference,
        "bins_json": json.dumps(
            [{"label": b.label, "lo": b.lo, "hi": b.hi} for b in align_cfg.bins],
            ensure_ascii=False,
        ),
        "overlap_threshold": float(align_cfg.overlap_threshold),
        "stage_transition_margin_sec": float(align_cfg.stage_transition_margin_sec),
        "source_events_path": events_path,
        "source_event_table_schema": _decode_text_scalar(event_table_columns.get("_schema_version")),
        "window_sec": float(window_sec),
        "window_step_sec": float(window_sec * (1.0 - overlap)),
        "n_rows": int(len(records)),
        "n_events_input": int(alignment.qc.get("n_events_input", len(onsets))),
    }


def _build_participant_clinical_meta(
    *,
    participant_meta: Mapping[str, Any],
    participant_meta_source: Mapping[str, Any],
    mapped_meta: Mapping[str, Any],
    session: Optional[str],
    condition: Optional[str],
    task: Optional[str],
    run_id: Optional[str],
    acq_id: Optional[str],
) -> Dict[str, Any]:
    """Build a richer participant/session metadata payload for H5 export."""
    out: Dict[str, Any] = {
        "session": session,
        "condition": condition,
        "task": task,
        "run": run_id,
        "acq": acq_id,
    }
    if isinstance(participant_meta, Mapping) and participant_meta:
        out["participant"] = dict(participant_meta)
    if isinstance(mapped_meta, Mapping) and mapped_meta:
        out["mapped"] = dict(mapped_meta)
    if isinstance(participant_meta_source, Mapping) and participant_meta_source:
        out["source"] = dict(participant_meta_source)
    return {k: v for k, v in out.items() if v is not None}


def _build_qc_windows_export(
    *,
    sub_frame: pd.DataFrame,
    stage: Optional[np.ndarray],
    x: np.ndarray,
    coords_9d: Optional[np.ndarray],
    x_coverage: np.ndarray,
    min_axis_coverage: float,
) -> Dict[str, np.ndarray]:
    """Build a minimal per-window QC surface aligned to `/time`."""
    n_time = int(len(sub_frame))
    out: Dict[str, np.ndarray] = {
        "retained_after_qc": np.ones(n_time, dtype=np.int8),
        "rejected_flag": np.zeros(n_time, dtype=np.int8),
    }
    for key in ("qc_ok_eeg", "qc_ok_ecg", "qc_ok_eog"):
        if key in sub_frame.columns:
            arr = pd.to_numeric(sub_frame[key], errors="coerce").fillna(1).to_numpy()
            out[key] = np.asarray(arr, dtype=np.int8)
    if x_coverage.size:
        coverage_ok = np.all(np.isfinite(x_coverage) & (x_coverage >= float(min_axis_coverage)), axis=1).astype(np.int8)
        out["coverage_ok"] = coverage_ok
    if np.size(x):
        out["mnps_3d_valid"] = np.all(np.isfinite(np.asarray(x, dtype=float)), axis=1).astype(np.int8)
    if coords_9d is not None and np.size(coords_9d):
        out["coords_9d_valid"] = np.all(np.isfinite(np.asarray(coords_9d, dtype=float)), axis=1).astype(np.int8)
        if "mnps_3d_valid" in out and len(out["mnps_3d_valid"]) == len(out["coords_9d_valid"]):
            out["geometry_valid"] = (out["mnps_3d_valid"] & out["coords_9d_valid"]).astype(np.int8)
    elif "mnps_3d_valid" in out:
        out["geometry_valid"] = np.asarray(out["mnps_3d_valid"], dtype=np.int8)
    if stage is not None and len(stage) == n_time:
        transitions = np.zeros(n_time, dtype=np.int8)
        if n_time > 1:
            transitions[1:] = (np.diff(np.asarray(stage, dtype=np.int32)) != 0).astype(np.int8)
        out["stage_transition_flag"] = transitions
    return out


def _build_coverage_export(
    *,
    x_coverage: np.ndarray,
    min_axis_coverage: float,
    coordinate_layers: Mapping[str, Any],
    jacobian_centers: Optional[np.ndarray],
    jacobian_9d_centers: Optional[np.ndarray],
) -> Dict[str, Any]:
    """Build explicit cross-layer coverage metadata for H5 export."""
    out: Dict[str, Any] = {
        "axis_fraction": np.asarray(x_coverage, dtype=np.float32),
        "axis_names": np.asarray(["m", "d", "e"], dtype=object),
        "min_axis_coverage": np.asarray(float(min_axis_coverage), dtype=np.float32),
        "coordinate_layers_present": np.asarray(sorted([str(k) for k in coordinate_layers.keys()]), dtype=object),
        "coordinate_contracts_present": np.asarray(
            sorted(
                {
                    str((layer.get("attrs", {}) or {}).get("coordinate_contract"))
                    for layer in coordinate_layers.values()
                    if isinstance(layer, Mapping) and isinstance(layer.get("attrs", {}), Mapping)
                    and (layer.get("attrs", {}) or {}).get("coordinate_contract") is not None
                }
            ),
            dtype=object,
        ),
        "shared_time_grid": np.asarray(1, dtype=np.int8),
    }
    if jacobian_centers is not None and np.size(jacobian_centers) > 0:
        out["jacobian_centers"] = np.asarray(jacobian_centers, dtype=np.int32)
    if jacobian_9d_centers is not None and np.size(jacobian_9d_centers) > 0:
        out["jacobian_9d_centers"] = np.asarray(jacobian_9d_centers, dtype=np.int32)
    return out


def _resolve_export_contract_preferences(
    proj_cfg: Mapping[str, Any],
    *,
    external_anchor_available: bool,
) -> Dict[str, Any]:
    """Resolve which anchor contracts should be exported for this run."""
    export_cfg = proj_cfg.get("export_contracts", {}) if isinstance(proj_cfg, Mapping) else {}
    export_cfg = export_cfg if isinstance(export_cfg, Mapping) else {}

    subject_requested = bool(export_cfg.get("subject_anchored", True))
    cohort_requested = bool(export_cfg.get("cohort_anchored", external_anchor_available))
    subject_enabled = bool(subject_requested)
    cohort_enabled = bool(cohort_requested and external_anchor_available)
    skipped_contracts_with_reason: List[Dict[str, str]] = []

    if cohort_requested and not external_anchor_available:
        logger.warning(
            "mnps_projection.export_contracts.cohort_anchored=true requested but no external anchor is active; "
            "skipping cohort_anchored export."
        )
        skipped_contracts_with_reason.append(
            {
                "contract": "cohort_anchored",
                "reason": "requested_but_no_external_anchor",
            }
        )

    if not subject_enabled and not cohort_enabled:
        raise ValueError(
            "mnps_projection.export_contracts must enable at least one effective contract. "
            "Set subject_anchored=true, or enable cohort_anchored together with an active anchor."
        )

    primary_coordinate_contract = "cohort_anchored" if cohort_enabled else "subject_anchored"
    requested_contracts = []
    if subject_requested:
        requested_contracts.append("subject_anchored")
    if cohort_requested:
        requested_contracts.append("cohort_anchored")
    realized_contracts = []
    if subject_enabled:
        realized_contracts.append("subject_anchored")
    if cohort_enabled:
        realized_contracts.append("cohort_anchored")
    return {
        "subject_anchored": subject_enabled,
        "cohort_anchored": cohort_enabled,
        "primary_coordinate_contract": primary_coordinate_contract,
        "requested_contracts": requested_contracts,
        "realized_contracts": realized_contracts,
        "skipped_contracts_with_reason": skipped_contracts_with_reason,
    }


def _regional_result_to_h5_payload(
    result: Any,
    *,
    coordinate_contract: str,
    anchor_spec: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert a RegionalMNPSResult into H5-friendly mapping payload."""
    payload: Dict[str, Any] = {
        "mnps": getattr(result, "mnps", None),
        "mnps_dot": getattr(result, "mnps_dot", None),
        "jacobian": getattr(result, "jacobian", None),
        "stratified": getattr(result, "stratified", None),
        "metrics": dict(getattr(result, "metrics", {}) or {}),
        "n_timepoints": getattr(result, "n_timepoints", None),
        "attrs": {
            "coordinate_contract": str(coordinate_contract),
        },
    }
    if coordinate_contract == "cohort_anchored" and isinstance(anchor_spec, Mapping):
        payload["attrs"].update(
            {
                "anchor_id": anchor_spec.get("anchor_id"),
                "anchor_hash": anchor_spec.get("anchor_hash"),
                "anchor_source": anchor_spec.get("anchor_source"),
            }
        )
    return payload


def _build_regional_dual_contract_export(
    *,
    primary_coordinate_contract: str,
    subject_summary: Optional[Any],
    cohort_summary: Optional[Any],
    anchor_spec: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build backward-compatible regional export with explicit anchor sublayers."""
    summaries_by_contract = {
        "subject_anchored": subject_summary,
        "cohort_anchored": cohort_summary,
    }
    network_names: set[str] = set()
    for summary in summaries_by_contract.values():
        results = getattr(summary, "results", None)
        if isinstance(results, Mapping):
            network_names.update([str(k) for k in results.keys()])

    out: Dict[str, Any] = {}
    for network_name in sorted(network_names):
        network_entry: Dict[str, Any] = {
            "primary_coordinate_contract": str(primary_coordinate_contract),
        }
        anchor_layers: Dict[str, Any] = {}
        for contract_name, summary in summaries_by_contract.items():
            results = getattr(summary, "results", None)
            if not isinstance(results, Mapping):
                continue
            result = results.get(network_name)
            if result is None:
                continue
            anchor_layers[contract_name] = _regional_result_to_h5_payload(
                result,
                coordinate_contract=contract_name,
                anchor_spec=anchor_spec,
            )
        if not anchor_layers:
            continue
        primary_payload = anchor_layers.get(primary_coordinate_contract)
        if primary_payload is None:
            primary_payload = anchor_layers.get("subject_anchored") or next(iter(anchor_layers.values()))
        for legacy_key in ("mnps", "mnps_dot", "jacobian", "stratified", "metrics", "n_timepoints", "attrs"):
            if legacy_key in primary_payload:
                network_entry[legacy_key] = primary_payload[legacy_key]
        network_entry["anchor_layers"] = anchor_layers
        out[network_name] = network_entry
    return out


def _missing_weighted_feature_rate(
    frame: pd.DataFrame,
    weighted_features: list[str],
) -> float:
    """Fraction of missing/non-finite weighted feature cells.

    Features absent from the frame count as fully missing across all rows.
    """
    if len(frame) == 0 or not weighted_features:
        return 0.0
    total = float(len(frame) * len(weighted_features))
    missing = 0.0
    for feat in weighted_features:
        if feat not in frame.columns:
            missing += float(len(frame))
            continue
        col = pd.to_numeric(frame[feat], errors="coerce")
        missing += float((~np.isfinite(col.to_numpy(dtype=np.float64, copy=False))).sum())
    return float(missing / total) if total > 0 else 0.0


def _validate_e_e_subcoord_construct(subcoords_spec: Mapping[str, Any]) -> None:
    """Validate e_e subcoordinate against allowed energetic-complexity features.

    For fMRI v2.0 we allow signal-power and related robust proxies in addition
    to entropy-named metrics.
    """
    if not isinstance(subcoords_spec, Mapping):
        return
    e_e_weights = subcoords_spec.get("e_e")
    if e_e_weights is None:
        return
    if not isinstance(e_e_weights, Mapping) or not e_e_weights:
        raise ValueError("mnps_9d.subcoords.e_e must map to at least one supported energetic-complexity feature")
    allowed_exact = {
        "fmri_signal_power",
        "fmri_slow4_slow5_ratio",
        "fmri_ar1_coefficient",
    }
    invalid = []
    for name in e_e_weights.keys():
        key = str(name).strip()
        low = key.lower()
        if "entropy" in low:
            continue
        if key in allowed_exact:
            continue
        invalid.append(key)
    if invalid:
        raise ValueError(
            "mnps_9d.subcoords.e_e must map to supported energetic-complexity feature(s); "
            f"invalid entries: {invalid}"
        )


def _resolve_entropy_provenance(frame: pd.DataFrame) -> Dict[str, Any]:
    """Extract energetic-complexity metric provenance from feature frame."""
    construct = "energetic_complexity"
    metric = "permutation_entropy"
    backend = "numpy"
    degraded_mode = False
    reason = None
    if len(frame) == 0:
        return {
            "construct": construct,
            "metric": metric,
            "backend": backend,
            "degraded_mode": degraded_mode,
            "reason": reason,
        }

    def _first_mode(col_name: str) -> Optional[str]:
        """Internal helper: first mode."""
        if col_name not in frame.columns:
            return None
        series = frame[col_name].dropna()
        if series.empty:
            return None
        return str(series.astype(str).mode(dropna=True).iloc[0])

    construct = _first_mode("eeg_entropy_construct") or construct
    metric = _first_mode("eeg_entropy_metric") or metric
    backend = _first_mode("eeg_entropy_backend") or backend
    reason = _first_mode("eeg_entropy_reason")
    if "eeg_entropy_degraded_mode" in frame.columns:
        degraded_series = pd.to_numeric(frame["eeg_entropy_degraded_mode"], errors="coerce")
        degraded_mode = bool(np.nanmax(degraded_series.to_numpy(dtype=np.float64, copy=False)) > 0)

    return {
        "construct": construct,
        "metric": metric,
        "backend": backend,
        "degraded_mode": degraded_mode,
        "reason": reason,
    }


def _canonical_mde_from_v2_map() -> Dict[str, List[str]]:
    """Canonical v2->3D mapping used when mnps_3d.from_v2.map is omitted."""
    return {
        "m": ["m_a", "m_e", "m_o"],
        "d": ["d_n", "d_l", "d_s"],
        "e": ["e_e", "e_s", "e_m"],
    }


def _resolve_mnps_3d_cfg(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve 3D derivation strategy from config with stable defaults."""
    m3d_cfg = config.get("mnps_3d", {}) if isinstance(config, Mapping) else {}
    if not isinstance(m3d_cfg, Mapping):
        m3d_cfg = {}
    mode = str(m3d_cfg.get("mode", "direct_features")).strip().lower() or "direct_features"
    if mode not in {"from_v2", "direct_features"}:
        logger.warning("Unknown mnps_3d.mode '%s'; using direct_features", mode)
        mode = "direct_features"
    from_v2_cfg = m3d_cfg.get("from_v2", {}) if isinstance(m3d_cfg.get("from_v2", {}), Mapping) else {}
    pooling = str(from_v2_cfg.get("pooling", "mean")).strip().lower() or "mean"
    if pooling not in {"mean", "sum"}:
        logger.warning("Unknown mnps_3d.from_v2.pooling '%s'; using mean", pooling)
        pooling = "mean"
    aggregation_requested = str(from_v2_cfg.get("aggregation", "auto")).strip().lower() or "auto"
    if aggregation_requested not in {
        "auto",
        "fixed_weighted_projection",
        "group_pooling_mean",
        "group_pooling_sum",
    }:
        logger.warning(
            "Unknown mnps_3d.from_v2.aggregation '%s'; using auto",
            aggregation_requested,
        )
        aggregation_requested = "auto"
    map_cfg = from_v2_cfg.get("map", {})
    default_map = _canonical_mde_from_v2_map()
    resolved_map: Dict[str, List[str]] = {}
    for axis in ("m", "d", "e"):
        raw = map_cfg.get(axis) if isinstance(map_cfg, Mapping) else None
        if isinstance(raw, list) and raw:
            resolved_map[axis] = [str(v) for v in raw]
        else:
            resolved_map[axis] = list(default_map[axis])

    # New fixed linear mapping policy (preferred): mnps_projection.v1_mapping
    proj_cfg = config.get("mnps_projection", {}) if isinstance(config, Mapping) else {}
    v1_mapping_cfg = proj_cfg.get("v1_mapping", {}) if isinstance(proj_cfg, Mapping) else {}
    v1_mapping: Dict[str, Dict[str, float]] = {}
    v1_mapping_source = "mnps_projection.v1_mapping"
    if isinstance(v1_mapping_cfg, Mapping) and v1_mapping_cfg:
        for axis in ("m", "d", "e"):
            row = v1_mapping_cfg.get(axis, {})
            if isinstance(row, Mapping):
                v1_mapping[axis] = {str(k): float(v) for k, v in row.items()}
            else:
                v1_mapping[axis] = {}
    else:
        # Keep empty so runtime can fail fast for from_v2 mode when mapping is missing.
        for axis in ("m", "d", "e"):
            v1_mapping[axis] = {}
        v1_mapping_source = "missing"
    has_v1_mapping = any(bool(v1_mapping.get(axis, {})) for axis in ("m", "d", "e"))
    if aggregation_requested == "auto":
        aggregation = "fixed_weighted_projection" if has_v1_mapping else f"group_pooling_{pooling}"
    else:
        aggregation = aggregation_requested

    return {
        "mode": mode,
        "legacy_pooling": pooling,
        "aggregation_requested": aggregation_requested,
        "aggregation": aggregation,
        "map": resolved_map,
        "v1_mapping": v1_mapping,
        "v1_mapping_source": v1_mapping_source,
        "has_v1_mapping": bool(has_v1_mapping),
    }


def _coerce_v1_mapping_to_v2_subcoords(
    v1_mapping: Mapping[str, Any],
    subcoords_spec: Mapping[str, Any],
) -> Dict[str, Dict[str, float]]:
    """Map feature-level V1 weights onto v2 subcoordinate names when possible."""
    out: Dict[str, Dict[str, float]] = {"m": {}, "d": {}, "e": {}}
    if not isinstance(v1_mapping, Mapping):
        return out

    feature_to_subcoords: Dict[str, Dict[str, List[str]]] = {"m": {}, "d": {}, "e": {}}
    feature_to_subcoords_any: Dict[str, List[str]] = {}
    if isinstance(subcoords_spec, Mapping):
        for sub_name, sub_weights in subcoords_spec.items():
            sub = str(sub_name)
            axis = sub[:1]
            if axis not in {"m", "d", "e"}:
                continue
            if not isinstance(sub_weights, Mapping):
                continue
            for feat_name in sub_weights.keys():
                feat = str(feat_name)
                feature_to_subcoords[axis].setdefault(feat, []).append(sub)
                feature_to_subcoords_any.setdefault(feat, []).append(sub)

    for axis in ("m", "d", "e"):
        row = v1_mapping.get(axis, {}) if isinstance(v1_mapping, Mapping) else {}
        if not isinstance(row, Mapping):
            continue
        for name, weight_raw in row.items():
            try:
                weight = float(weight_raw)
            except Exception:
                continue
            if not np.isfinite(weight):
                continue
            key = str(name)
            if key.startswith(f"{axis}_"):
                out[axis][key] = out[axis].get(key, 0.0) + weight
                continue
            mapped = feature_to_subcoords[axis].get(key, [])
            if not mapped:
                # Allow explicit cross-block priors (e.g. m-axis can include d_s-derived signal).
                mapped = feature_to_subcoords_any.get(key, [])
            if mapped:
                per = weight / float(len(mapped))
                for sub in mapped:
                    out[axis][sub] = out[axis].get(sub, 0.0) + per
            else:
                # Preserve unknown keys so runtime validation can fail explicitly.
                out[axis][key] = out[axis].get(key, 0.0) + weight
    return out


def _align_v2_subcoords(
    coords_9d: np.ndarray,
    names: List[str],
    ordered_names: List[str],
) -> np.ndarray:
    """Align v2 subcoordinate matrix columns to canonical ordering when available."""
    arr = np.asarray(coords_9d, dtype=np.float64)
    if arr.ndim != 2 or not names:
        return arr
    idx = {str(name): i for i, name in enumerate(names)}
    if all(name in idx for name in ordered_names):
        order_idx = [idx[name] for name in ordered_names]
        return arr[:, order_idx]
    return arr


@lru_cache(maxsize=1)
def _get_env_provenance() -> Dict[str, Any]:
    """Collect lightweight runtime environment provenance."""
    py_ver = sys.version.replace("\n", " ").strip()
    plat = platform.platform()
    pip_freeze_hash = None
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
        )
        pip_freeze_hash = hashlib.sha256(out.encode("utf-8")).hexdigest()
    except Exception:
        pip_freeze_hash = None
    env_payload = {
        "python_version": py_ver,
        "platform": plat,
        "pip_freeze_hash": pip_freeze_hash,
    }
    env_payload["env_hash"] = _stable_hash_mapping(env_payload)
    return env_payload


def _deep_merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep merge dictionaries: override wins; nested mappings are merged recursively."""
    out: Dict[str, Any] = dict(base) if isinstance(base, Mapping) else {}
    if not isinstance(override, Mapping):
        return out
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge_dict(out.get(k, {}), v)
        else:
            out[k] = v
    return out


@lru_cache(maxsize=128)
def _load_mnps_9d_policy(policy_dir: str, dataset_id: str) -> Dict[str, Any]:
    """Load optional per-dataset mnps_9d policy from YAML."""
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    if not policy_dir or not dataset_id:
        return {}

    root = Path(policy_dir)
    ds_path = root / f"{dataset_id}_mnps_9d.yml"
    map_path = root / "datasets.yml"
    path = ds_path if ds_path.exists() else (map_path if map_path.exists() else None)
    if path is None:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

    payload: Dict[str, Any] = {}
    if path.name == "datasets.yml":
        if isinstance(data, Mapping):
            # Supported shapes:
            # - {datasets: {<ds>: {...}}}
            # - {mnps_9d: {datasets: {<ds>: {...}}}}
            # - {<ds>: {...}}
            if isinstance(data.get("mnps_9d"), Mapping):
                v2 = data.get("mnps_9d") or {}
                ds_map = v2.get("datasets", {}) if isinstance(v2, Mapping) else {}
                if isinstance(ds_map, Mapping):
                    payload = dict(ds_map.get(dataset_id, {}) or {})
            if not payload and isinstance(data.get("datasets"), Mapping):
                payload = dict((data.get("datasets") or {}).get(dataset_id, {}) or {})
            if not payload and isinstance(data.get(dataset_id), Mapping):
                payload = dict(data.get(dataset_id) or {})
    else:
        if isinstance(data, Mapping) and isinstance(data.get("mnps_9d"), Mapping):
            v2 = data.get("mnps_9d") or {}
            if isinstance(v2, Mapping):
                payload = dict(v2)
        else:
            payload = dict(data) if isinstance(data, Mapping) else {}

    for k in ("schema", "schema_version", "dataset_id", "description"):
        payload.pop(k, None)
    return payload


def _resolve_mnps_9d_runtime_config(
    v2_cfg: Mapping[str, Any],
    dataset_id: str,
) -> tuple[bool, str, Dict[str, Any], Dict[str, Any]]:
    """Resolve versioned and dataset-specific MNPS 9D config for one dataset."""
    cfg_map: Dict[str, Any] = dict(v2_cfg) if isinstance(v2_cfg, Mapping) else {}
    v2_enabled = bool(cfg_map.get("enabled", False))
    v2_definition_version = str(cfg_map.get("definition_version", cfg_map.get("mnps_9d_definition_version", "2.2")))

    selected_v2_cfg: Dict[str, Any] = dict(cfg_map)
    v2_versions = cfg_map.get("versions", {}) if isinstance(cfg_map, Mapping) else {}
    if isinstance(v2_versions, Mapping):
        candidate = v2_versions.get(v2_definition_version)
        if isinstance(candidate, Mapping):
            selected_v2_cfg = _deep_merge_dict(cfg_map, candidate)
            # Versioned subcoords replace the legacy/root map instead of merging into it.
            if isinstance(candidate.get("subcoords"), Mapping):
                selected_v2_cfg["subcoords"] = dict(candidate.get("subcoords") or {})
            if isinstance(candidate.get("metric_policies"), Mapping):
                selected_v2_cfg["metric_policies"] = _deep_merge_dict(
                    cfg_map.get("metric_policies", {}) if isinstance(cfg_map.get("metric_policies", {}), Mapping) else {},
                    candidate.get("metric_policies", {}) if isinstance(candidate.get("metric_policies", {}), Mapping) else {},
                )

    subcoords_spec: Dict[str, Any] = (
        dict(selected_v2_cfg.get("subcoords", {}) or {})
        if isinstance(selected_v2_cfg.get("subcoords", {}), Mapping)
        else {}
    )
    ds_overrides: Dict[str, Any] = {}
    inline = (cfg_map.get("datasets", {}) or {}).get(dataset_id, {})
    if isinstance(inline, Mapping):
        ds_overrides = dict(inline)
    policy_dir = cfg_map.get("policy_dir")
    if isinstance(policy_dir, (str, Path)):
        policy_cfg = _load_mnps_9d_policy(str(policy_dir), dataset_id)
        if policy_cfg:
            ds_overrides = _deep_merge_dict(ds_overrides, policy_cfg)
    if isinstance(ds_overrides, Mapping):
        if "enabled" in ds_overrides:
            v2_enabled = bool(ds_overrides.get("enabled", v2_enabled))
        if "subcoords" in ds_overrides and isinstance(ds_overrides["subcoords"], Mapping):
            merged = dict(subcoords_spec)
            merged.update(ds_overrides["subcoords"])
            subcoords_spec = merged
            selected_v2_cfg["subcoords"] = merged
        if "metric_policies" in ds_overrides and isinstance(ds_overrides["metric_policies"], Mapping):
            selected_v2_cfg = _deep_merge_dict(
                selected_v2_cfg,
                {"metric_policies": ds_overrides["metric_policies"]},
            )

    return v2_enabled, v2_definition_version, selected_v2_cfg, subcoords_spec


class DatasetSummaryRunner:
    """Encapsulate dataset-level summarization logic."""

    def __init__(self, ctx: SummarizeContext, ds_id: str, subject_filter: Optional[str], h5_mode: str, n_jobs: int = 1):
        """Initialize the instance."""
        config_copy = copy.deepcopy(ctx.config) if isinstance(ctx.config, Mapping) else ctx.config
        self.ctx = _RunnerContextProxy(ctx, config_copy)
        self.ds_id = ds_id
        self.subject_filter = self._normalize_subject(subject_filter) if subject_filter else None
        self.h5_mode = h5_mode
        self.n_jobs = max(1, int(n_jobs or 1))
        self.config = self.ctx.config
        config_path_raw = getattr(ctx, "config_path", None)
        self.config_path: Optional[Path] = (
            Path(str(config_path_raw)).expanduser()
            if isinstance(config_path_raw, (str, Path))
            else None
        )
        self.received_dir = ctx.received_dir
        self.processed_dir = ctx.processed_dir
        self._dataset_csv_lock = Lock()
        self._run_errors_lock = Lock()
        self._run_errors: List[Dict[str, Any]] = []
        self._stage_mapping_qc_lock = Lock()
        self._stage_mapping_qc_entries: List[Dict[str, Any]] = []
        self._block_native_qc_lock = Lock()
        self._block_native_qc_entries: List[Dict[str, Any]] = []
        # Global coverage defaults, with optional dataset-specific overrides
        self.min_seconds = self.ctx.coverage.min_seconds
        self.min_epochs = self.ctx.coverage.min_epochs
        self.coverage_optional_rules: List[Dict[str, Any]] = []
        self.coverage_rule_source: str = "default"
        try:
            robustness_cfg = self.config.get("robustness", {}) if isinstance(self.config, Mapping) else {}
            coverage_cfg = robustness_cfg.get("coverage", {}) if isinstance(robustness_cfg, Mapping) else {}
            ds_overrides = coverage_cfg.get("datasets", {}) if isinstance(coverage_cfg, Mapping) else {}
            if isinstance(ds_overrides, Mapping):
                ds_cfg = ds_overrides.get(self.ds_id, {})
                if isinstance(ds_cfg, Mapping):
                    if "min_seconds" in ds_cfg:
                        self.min_seconds = float(ds_cfg.get("min_seconds", self.min_seconds) or self.min_seconds)
                    if "min_epochs" in ds_cfg:
                        self.min_epochs = int(ds_cfg.get("min_epochs", self.min_epochs) or self.min_epochs)
                    optional_rules = ds_cfg.get("optional_rules", [])
                    if isinstance(optional_rules, list):
                        self.coverage_optional_rules.extend(
                            [r for r in optional_rules if isinstance(r, Mapping)]
                        )
        except Exception:
            logger.exception("Failed to apply dataset-specific coverage overrides; using global defaults")

        # Optional external per-dataset config:
        #   openneuro_ingest/config/config_<dataset>.yaml
        # Used for dataset-specific, auditable policy (e.g. ds005620 sed2 short segments).
        external_cfg = self._load_external_dataset_config()
        ext_cov = (
            external_cfg.get("robustness", {}).get("coverage", {})
            if isinstance(external_cfg, Mapping)
            else {}
        )
        if isinstance(ext_cov, Mapping):
            if "min_seconds" in ext_cov:
                self.min_seconds = float(ext_cov.get("min_seconds", self.min_seconds) or self.min_seconds)
            if "min_epochs" in ext_cov:
                self.min_epochs = int(ext_cov.get("min_epochs", self.min_epochs) or self.min_epochs)
            ext_rules = ext_cov.get("optional_rules", [])
            if isinstance(ext_rules, list):
                self.coverage_optional_rules.extend([r for r in ext_rules if isinstance(r, Mapping)])
            if external_cfg:
                self.coverage_rule_source = f"external:{self.ds_id}"

        self.participants_df: Optional[pd.DataFrame] = None
        self.index_df: Optional[pd.DataFrame] = None
        summarize_cfg = self.config.get("summarize", {}) if isinstance(self.config, Mapping) else {}
        if not isinstance(summarize_cfg, Mapping):
            summarize_cfg = {}
        # Safer default: do not silently merge multiple source files into one summarize key.
        self.allow_group_collisions = bool(summarize_cfg.get("allow_group_collisions", False))
        self.qc_policy = str(summarize_cfg.get("qc_policy", "eeg_only")).strip().lower() or "eeg_only"
        self.grouping_collision_info: Dict[str, Any] = {"count": 0, "merged_extra_files": 0, "examples": []}
        self._participant_meta_map: Dict[str, Dict[str, Any]] = {}
        self._file_entity_cache: Dict[str, tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]] = {}
        self._index_paths_by_basename: Dict[str, List[str]] = {}
        self._normalization_report: Dict[str, Any] = {
            "enabled": False,
            "status": "disabled",
            "method": None,
            "scope": None,
        }

    @staticmethod
    def _normalize_subject(value: Optional[str]) -> Optional[str]:
        """Internal helper: normalize subject."""
        if value is None:
            return None
        value = str(value)
        return value if value.startswith("sub-") else f"sub-{value.zfill(3)}"

    def _parse_file_entities(
        self,
        file_name: str,
    ) -> tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Parse grouping entities from BIDS, with optional non-BIDS regex fallback."""
        subject, session, task, run, acq = parse_subject_session_task_run_acq(file_name)
        if subject != "sub-unknown":
            return subject, session, task, run, acq

        try:
            md_spec = self.config.get("metadata_extraction", {}) if isinstance(self.config, Mapping) else {}
            ds_spec = (md_spec.get("datasets", {}) or {}).get(self.ds_id, {}) if isinstance(md_spec, Mapping) else {}
            parse_cfg = ds_spec.get("filename_parse", {}) if isinstance(ds_spec, Mapping) else {}
            regex = str(parse_cfg.get("regex", "")).strip() if isinstance(parse_cfg, Mapping) else ""
            if not regex:
                return subject, session, task, run, acq
            m = re.search(regex, str(file_name))
            if not m:
                return subject, session, task, run, acq

            gd = m.groupdict()
            subj_raw = gd.get("subject")
            pad = int(parse_cfg.get("subject_pad", 3)) if isinstance(parse_cfg, Mapping) else 3
            if subj_raw:
                subj_s = str(subj_raw)
                subject = subj_s if subj_s.startswith("sub-") else f"sub-{subj_s.zfill(pad)}"

            ses_raw = gd.get("session")
            if ses_raw:
                ses_s = str(ses_raw)
                session = ses_s if ses_s.startswith("ses-") else f"ses-{ses_s}"

            task_raw = gd.get("task")
            if task_raw:
                task = str(task_raw)

            run_raw = gd.get("run")
            if run_raw:
                run_s = str(run_raw)
                run = run_s if run_s.startswith("run-") else f"run-{run_s}"

            acq_raw = gd.get("acq")
            if acq_raw:
                acq_s = str(acq_raw)
                acq = acq_s if acq_s.startswith("acq-") else f"acq-{acq_s}"
        except Exception:
            logger.exception("Failed non-BIDS filename parsing for dataset %s", self.ds_id)

        return subject, session, task, run, acq

    def _load_external_dataset_config(self) -> Dict[str, Any]:
        """Internal helper: load external dataset config."""
        cfg_path = Path(__file__).resolve().parents[4] / "openneuro_ingest" / "config" / f"config_{self.ds_id}.yaml"
        if not cfg_path.exists():
            return {}
        try:
            import yaml  # type: ignore

            with cfg_path.open("r", encoding="utf-8") as f:
                parsed = yaml.safe_load(f) or {}
            if isinstance(parsed, Mapping):
                logger.info("Loaded external dataset config: %s", cfg_path)
                return dict(parsed)
            return {}
        except Exception as exc:
            logger.warning("Failed to load external dataset config %s: %s", cfg_path, exc)
            return {}

    @staticmethod
    def _match_value(rule_val: Any, actual_val: Optional[str]) -> bool:
        """Internal helper: match value."""
        if rule_val is None:
            return True
        if isinstance(rule_val, (list, tuple, set)):
            targets = {str(v).lower() for v in rule_val}
            return str(actual_val).lower() in targets
        return str(actual_val).lower() == str(rule_val).lower()

    def resolve_coverage_policy(
        self,
        *,
        condition: Optional[str],
        task: Optional[str],
        run_id: Optional[str],
        acq_id: Optional[str],
    ) -> Dict[str, Any]:
        """Handle resolve coverage policy."""
        policy: Dict[str, Any] = {
            "min_seconds": float(self.min_seconds),
            "min_epochs": int(self.min_epochs),
            "tag": "default",
            "source": self.coverage_rule_source,
        }
        for i, rule in enumerate(self.coverage_optional_rules):
            match = rule.get("match", {}) if isinstance(rule, Mapping) else {}
            if not isinstance(match, Mapping):
                match = {}
            if not self._match_value(match.get("condition"), condition):
                continue
            if not self._match_value(match.get("task"), task):
                continue
            if not self._match_value(match.get("run"), run_id):
                continue
            if not self._match_value(match.get("acq"), acq_id):
                continue

            if "min_seconds" in rule:
                policy["min_seconds"] = float(rule.get("min_seconds", policy["min_seconds"]) or policy["min_seconds"])
            if "min_epochs" in rule:
                policy["min_epochs"] = int(rule.get("min_epochs", policy["min_epochs"]) or policy["min_epochs"])
            policy["tag"] = str(rule.get("tag", f"optional_rule_{i}"))
            break
        return policy

    def _resolve_normalization_cfg(self) -> Dict[str, Any]:
        """Resolve normalization config with optional per-dataset overrides."""
        root = self.config.get("normalization", {}) if isinstance(self.config, Mapping) else {}
        if not isinstance(root, Mapping):
            return {}
        resolved: Dict[str, Any] = dict(root)
        ds_overrides = root.get("datasets", {})
        if isinstance(ds_overrides, Mapping):
            ds_cfg = ds_overrides.get(self.ds_id, {})
            if isinstance(ds_cfg, Mapping):
                resolved = _deep_merge_dict(resolved, ds_cfg)
        resolved.pop("datasets", None)
        return resolved

    @staticmethod
    def _normalization_key_candidates(raw_key: str) -> List[str]:
        """Build fallback metadata key candidates for normalization fields."""
        key = str(raw_key or "").strip()
        if not key:
            return []
        key_low = key.lower()
        candidates = [key]
        alias_map: Dict[str, List[str]] = {
            "site_or_hospital": [
                "hospital",
                "site",
                "site_id",
                "hospital_id",
                "center",
                "centre",
                "institution",
                "site_or_hospital",
            ],
        }
        aliases = alias_map.get(key_low, [])
        if aliases:
            candidates.extend(aliases)
        seen: set[str] = set()
        out: List[str] = []
        for cand in candidates:
            name = str(cand).strip()
            if not name:
                continue
            low = name.lower()
            if low in seen:
                continue
            seen.add(low)
            out.append(name)
        return out

    @staticmethod
    def _lookup_meta_value(meta: Mapping[str, Any], key_candidates: Sequence[str]) -> Any:
        """Lookup one metadata value using case-insensitive candidate keys."""
        if not isinstance(meta, Mapping):
            return None
        lower_map = {str(k).strip().lower(): v for k, v in meta.items()}
        for key in key_candidates:
            if key in meta:
                return meta.get(key)
            hit = lower_map.get(str(key).strip().lower())
            if hit is not None:
                return hit
        return None

    @staticmethod
    def _is_missing_meta_value(value: Any) -> bool:
        """True when metadata values should be treated as missing."""
        if value is None:
            return True
        try:
            if bool(pd.isna(value)):
                return True
        except Exception:
            pass
        if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
            return True
        if isinstance(value, str) and not value.strip():
            return True
        return False

    @staticmethod
    def _prepare_combat_covariate(
        series: pd.Series,
    ) -> tuple[pd.Series, str]:
        """Coerce one covariate to categorical/continuous for neuroCombat."""
        raw = series.copy()
        numeric = pd.to_numeric(raw, errors="coerce")
        finite_ratio = float(numeric.notna().mean()) if len(raw) else 0.0
        unique_numeric = int(numeric.dropna().nunique())
        if finite_ratio >= 0.95 and unique_numeric > 4:
            fill_value = float(np.nanmedian(numeric.to_numpy(dtype=np.float64))) if numeric.notna().any() else 0.0
            filled = numeric.fillna(fill_value).astype(float)
            return filled, "continuous"

        cleaned = raw.astype(str)
        cleaned = cleaned.where(~raw.isna(), other="unknown")
        cleaned = cleaned.str.strip()
        cleaned = cleaned.where(cleaned != "", other="unknown")
        return cleaned.astype(str), "categorical"

    @staticmethod
    def _parse_winsorize_quantiles(cfg_value: Any) -> Optional[tuple[float, float]]:
        """Parse optional winsorization quantiles for pre-ComBat clipping."""
        if not isinstance(cfg_value, (list, tuple)) or len(cfg_value) != 2:
            return None
        try:
            low = float(cfg_value[0])
            high = float(cfg_value[1])
        except Exception:
            return None
        if not (0.0 <= low < high <= 1.0):
            return None
        return low, high

    @staticmethod
    def _resolve_normalization_validation_cfg(norm_cfg: Mapping[str, Any]) -> Dict[str, Any]:
        """Resolve and sanitize normalization validation config."""
        raw = norm_cfg.get("validation", {}) if isinstance(norm_cfg.get("validation", {}), Mapping) else {}
        metrics_raw = raw.get("metrics", {}) if isinstance(raw.get("metrics", {}), Mapping) else {}
        target_keys_raw = raw.get("target_keys", [])
        if not isinstance(target_keys_raw, list):
            target_keys_raw = []
        target_keys = [str(v).strip() for v in target_keys_raw if str(v).strip()]

        return {
            "enabled": bool(raw.get("enabled", False)),
            "max_rows": max(1000, int(raw.get("max_rows", 150000) or 150000)),
            "max_features": max(8, int(raw.get("max_features", 256) or 256)),
            "min_group_size": max(2, int(raw.get("min_group_size", 20) or 20)),
            "max_levels": max(2, int(raw.get("max_levels", 24) or 24)),
            "batch_key": str(raw.get("batch_key", "auto")).strip() or "auto",
            "target_keys": target_keys,
            "metrics": {
                "batch_eta2": bool(metrics_raw.get("batch_eta2", True)),
                "target_eta2": bool(metrics_raw.get("target_eta2", True)),
                "perturbation": bool(metrics_raw.get("perturbation", True)),
            },
        }

    def _sample_probe_indices(self, candidate_index: pd.Index, cfg: Mapping[str, Any]) -> pd.Index:
        """Downsample probe rows deterministically when needed."""
        max_rows = max(1, int(cfg.get("max_rows", 150000) or 150000))
        if len(candidate_index) <= max_rows:
            return candidate_index
        seed = 42
        try:
            repro = getattr(self.ctx, "reproducibility", {}) or {}
            seed = int(repro.get("seed", 42) or 42)
        except Exception:
            seed = 42
        rng = np.random.default_rng(seed)
        picks = np.sort(rng.choice(len(candidate_index), size=max_rows, replace=False))
        return candidate_index.take(picks)

    def _sample_probe_feature_columns(self, feature_cols: Sequence[str], cfg: Mapping[str, Any]) -> List[str]:
        """Downsample probe feature columns deterministically when needed."""
        cols = [str(c) for c in feature_cols]
        max_features = max(1, int(cfg.get("max_features", 256) or 256))
        if len(cols) <= max_features:
            return cols
        seed = 42
        try:
            repro = getattr(self.ctx, "reproducibility", {}) or {}
            seed = int(repro.get("seed", 42) or 42)
        except Exception:
            seed = 42
        rng = np.random.default_rng(seed + 101)
        picks = np.sort(rng.choice(len(cols), size=max_features, replace=False))
        return [cols[i] for i in picks]

    def _compute_eta2_probe_summary(
        self,
        frame: pd.DataFrame,
        labels: pd.Series,
        *,
        min_group_size: int,
        max_levels: int,
    ) -> Dict[str, Any]:
        """Compute per-feature eta^2 summary for a categorical label."""
        if frame.empty:
            return {"status": "no_rows"}
        label_series = labels.reindex(frame.index)
        cleaned = label_series.map(lambda v: None if self._is_missing_meta_value(v) else str(v).strip())
        observed = cleaned.dropna()
        if observed.empty:
            return {"status": "no_labels"}

        counts = observed.value_counts()
        eligible = counts[counts >= int(min_group_size)]
        if int(eligible.shape[0]) < 2:
            return {
                "status": "insufficient_groups",
                "groups_observed": int(counts.shape[0]),
                "groups_eligible": int(eligible.shape[0]),
                "group_counts": {str(k): int(v) for k, v in counts.to_dict().items()},
            }
        if int(eligible.shape[0]) > int(max_levels):
            eligible = eligible.iloc[: int(max_levels)]
        levels = [str(v) for v in eligible.index.tolist()]
        mask_labels = cleaned.isin(levels).to_numpy(dtype=bool)
        label_arr = cleaned.to_numpy(dtype=object)

        eta_rows: List[tuple[str, float]] = []
        min_points = max(20, int(2 * len(levels)))
        for col in frame.columns:
            values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            mask = np.isfinite(values) & mask_labels
            if int(mask.sum()) < min_points:
                continue
            x = values[mask]
            y = label_arr[mask]
            mu = float(np.nanmean(x))
            sst = float(np.nansum((x - mu) ** 2))
            if not np.isfinite(sst) or sst <= 1e-12:
                continue
            ssb = 0.0
            for level in levels:
                idx = np.asarray([val == level for val in y], dtype=bool)
                n_g = int(idx.sum())
                if n_g <= 0:
                    continue
                mu_g = float(np.nanmean(x[idx]))
                ssb += float(n_g) * float((mu_g - mu) ** 2)
            eta2 = float(ssb / sst)
            if np.isfinite(eta2):
                eta_rows.append((str(col), eta2))

        if not eta_rows:
            return {
                "status": "no_numeric_signal",
                "group_counts": {str(k): int(v) for k, v in eligible.to_dict().items()},
            }

        eta_vals = np.asarray([v for _, v in eta_rows], dtype=np.float64)
        top = sorted(eta_rows, key=lambda row: abs(float(row[1])), reverse=True)[:10]
        return {
            "status": "computed",
            "groups_used": int(len(levels)),
            "group_counts": {str(k): int(v) for k, v in eligible.to_dict().items()},
            "features_evaluated": int(len(eta_rows)),
            "eta2_median": float(np.nanmedian(eta_vals)),
            "eta2_mean": float(np.nanmean(eta_vals)),
            "eta2_p95": float(np.nanpercentile(eta_vals, 95)),
            "top_features": [{"name": name, "eta2": float(val)} for name, val in top],
        }

    @staticmethod
    def _compute_perturbation_probe_summary(
        pre_df: pd.DataFrame,
        post_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Summarize pre/post perturbation strength across probe features."""
        if pre_df.empty or post_df.empty:
            return {"status": "no_rows"}
        common_cols = [c for c in pre_df.columns if c in post_df.columns]
        if not common_cols:
            return {"status": "no_features"}

        pct_shifts: List[float] = []
        mad_scaled: List[float] = []
        corrs: List[float] = []
        top_rows: List[tuple[str, float]] = []
        for col in common_cols:
            pre = pd.to_numeric(pre_df[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            post = pd.to_numeric(post_df[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            mask = np.isfinite(pre) & np.isfinite(post)
            if int(mask.sum()) < 10:
                continue
            x = pre[mask]
            y = post[mask]
            delta = y - x
            pre_med = float(np.nanmedian(x))
            pct = float(np.nanmedian(np.abs(delta) / (abs(pre_med) + 1e-9) * 100.0))
            scale = float(np.nanmedian(np.abs(x - pre_med)) * projection.ROBUST_MAD_TO_SIGMA)
            if not np.isfinite(scale) or scale <= 1e-9:
                scale = float(np.nanstd(x))
            if not np.isfinite(scale) or scale <= 1e-9:
                scale = 1.0
            mad_val = float(np.nanmedian(np.abs(delta) / (scale + 1e-9)))
            if np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
                corr = np.nan
            else:
                corr = float(np.corrcoef(x, y)[0, 1])
            pct_shifts.append(pct)
            mad_scaled.append(mad_val)
            if np.isfinite(corr):
                corrs.append(corr)
            top_rows.append((str(col), pct))

        if not pct_shifts:
            return {"status": "no_numeric_signal"}

        pct_arr = np.asarray(pct_shifts, dtype=np.float64)
        mad_arr = np.asarray(mad_scaled, dtype=np.float64)
        corr_arr = np.asarray(corrs, dtype=np.float64) if corrs else np.asarray([], dtype=np.float64)
        top = sorted(top_rows, key=lambda row: abs(float(row[1])), reverse=True)[:10]
        out: Dict[str, Any] = {
            "status": "computed",
            "features_evaluated": int(len(pct_shifts)),
            "median_abs_pct_shift": float(np.nanmedian(pct_arr)),
            "p95_abs_pct_shift": float(np.nanpercentile(pct_arr, 95)),
            "median_mad_scaled_shift": float(np.nanmedian(mad_arr)),
            "p95_mad_scaled_shift": float(np.nanpercentile(mad_arr, 95)),
            "top_features_abs_pct_shift": [
                {"name": name, "abs_pct_shift": float(val)} for name, val in top
            ],
        }
        if corr_arr.size:
            out["feature_corr_median"] = float(np.nanmedian(corr_arr))
            out["feature_corr_p05"] = float(np.nanpercentile(corr_arr, 5))
        return out

    def _compute_normalization_validation_report(
        self,
        *,
        pre_probe_df: pd.DataFrame,
        post_probe_df: pd.DataFrame,
        batch_probe_series: Optional[pd.Series],
        target_probe_map: Mapping[str, pd.Series],
        validation_cfg: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Compute configured pre/post validation probes for normalization."""
        report: Dict[str, Any] = {
            "enabled": True,
            "status": "computed",
            "rows_sampled": int(len(pre_probe_df)),
            "features_sampled": int(pre_probe_df.shape[1]),
            "config": {
                "max_rows": int(validation_cfg.get("max_rows", 0) or 0),
                "max_features": int(validation_cfg.get("max_features", 0) or 0),
                "min_group_size": int(validation_cfg.get("min_group_size", 0) or 0),
                "max_levels": int(validation_cfg.get("max_levels", 0) or 0),
                "batch_key": str(validation_cfg.get("batch_key", "auto")),
                "target_keys": [str(v) for v in validation_cfg.get("target_keys", [])],
            },
            "probes": {},
        }
        metrics_cfg = validation_cfg.get("metrics", {}) if isinstance(validation_cfg.get("metrics", {}), Mapping) else {}
        min_group_size = int(validation_cfg.get("min_group_size", 20) or 20)
        max_levels = int(validation_cfg.get("max_levels", 24) or 24)

        if bool(metrics_cfg.get("batch_eta2", True)) and batch_probe_series is not None:
            pre_batch = self._compute_eta2_probe_summary(
                pre_probe_df,
                batch_probe_series,
                min_group_size=min_group_size,
                max_levels=max_levels,
            )
            post_batch = self._compute_eta2_probe_summary(
                post_probe_df,
                batch_probe_series,
                min_group_size=min_group_size,
                max_levels=max_levels,
            )
            batch_probe: Dict[str, Any] = {"pre": pre_batch, "post": post_batch}
            if pre_batch.get("status") == "computed" and post_batch.get("status") == "computed":
                batch_probe["delta_eta2_median"] = float(post_batch["eta2_median"] - pre_batch["eta2_median"])
            report["probes"]["batch_eta2"] = batch_probe

        if bool(metrics_cfg.get("target_eta2", True)):
            targets: Dict[str, Any] = {}
            for target_key, target_series in target_probe_map.items():
                pre_target = self._compute_eta2_probe_summary(
                    pre_probe_df,
                    target_series,
                    min_group_size=min_group_size,
                    max_levels=max_levels,
                )
                post_target = self._compute_eta2_probe_summary(
                    post_probe_df,
                    target_series,
                    min_group_size=min_group_size,
                    max_levels=max_levels,
                )
                target_probe: Dict[str, Any] = {"pre": pre_target, "post": post_target}
                if pre_target.get("status") == "computed" and post_target.get("status") == "computed":
                    target_probe["delta_eta2_median"] = float(post_target["eta2_median"] - pre_target["eta2_median"])
                targets[str(target_key)] = target_probe
            report["probes"]["target_eta2"] = targets

        if bool(metrics_cfg.get("perturbation", True)):
            report["probes"]["perturbation"] = self._compute_perturbation_probe_summary(
                pre_probe_df,
                post_probe_df,
            )

        if not report["probes"]:
            report["status"] = "disabled_by_metrics"
        return report

    @staticmethod
    def _resolve_combat_family_cfg(combat_cfg: Mapping[str, Any]) -> Dict[str, Any]:
        """Resolve family-wise ComBat grouping config."""
        raw = combat_cfg.get("family_wise", {}) if isinstance(combat_cfg.get("family_wise", {}), Mapping) else {}
        regex_map_raw = raw.get("regex_map", {}) if isinstance(raw.get("regex_map", {}), Mapping) else {}
        regex_map: Dict[str, List[str]] = {}
        for family, patterns in regex_map_raw.items():
            if isinstance(patterns, list):
                normalized = [str(p).strip() for p in patterns if str(p).strip()]
                if normalized:
                    regex_map[str(family)] = normalized
        strategy = str(raw.get("strategy", "prefix")).strip().lower() or "prefix"
        if strategy not in {"prefix", "regex_map"}:
            strategy = "prefix"
        delimiter = str(raw.get("delimiter", "_"))
        return {
            "enabled": bool(raw.get("enabled", False)),
            "strategy": strategy,
            "delimiter": delimiter if delimiter else "_",
            "min_family_columns": max(1, int(raw.get("min_family_columns", 1) or 1)),
            "regex_map": regex_map,
        }

    def _feature_family_name(self, feature_name: str, family_cfg: Mapping[str, Any]) -> str:
        """Resolve one feature column to a family name."""
        base = str(feature_name).split("__g_", 1)[0]
        regex_map = family_cfg.get("regex_map", {}) if isinstance(family_cfg.get("regex_map", {}), Mapping) else {}
        if regex_map:
            for family, patterns in regex_map.items():
                if not isinstance(patterns, list):
                    continue
                for pattern in patterns:
                    try:
                        if re.search(str(pattern), base):
                            return str(family)
                    except re.error:
                        continue
            if str(family_cfg.get("strategy", "prefix")) == "regex_map":
                return "unmatched"
        delimiter = str(family_cfg.get("delimiter", "_"))
        if delimiter and delimiter in base:
            head = base.split(delimiter, 1)[0]
        else:
            head = base
        family = str(head).strip().lower() or "unknown"
        return family

    def _build_combat_family_groups(
        self,
        feature_cols: Sequence[str],
        family_cfg: Mapping[str, Any],
    ) -> Dict[str, List[str]]:
        """Group feature columns for optional family-wise ComBat fitting."""
        cols = [str(c) for c in feature_cols]
        if not bool(family_cfg.get("enabled", False)):
            return {"__all__": sorted(cols)}
        groups: Dict[str, List[str]] = {}
        for col in cols:
            family = self._feature_family_name(col, family_cfg)
            groups.setdefault(family, []).append(col)
        for family in list(groups.keys()):
            groups[family] = sorted(groups[family])

        min_cols = max(1, int(family_cfg.get("min_family_columns", 1) or 1))
        if min_cols > 1:
            small = [family for family, family_cols in groups.items() if len(family_cols) < min_cols]
            if small:
                merged: List[str] = []
                for family in small:
                    merged.extend(groups.pop(family, []))
                if merged:
                    groups.setdefault("__other__", [])
                    groups["__other__"].extend(merged)
                    groups["__other__"] = sorted(set(groups["__other__"]))
        if not groups:
            groups["__all__"] = sorted(cols)
        return {family: sorted(family_cols) for family, family_cols in sorted(groups.items(), key=lambda row: row[0])}

    def _apply_feature_normalization(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply optional post-features normalization (ComBat pilot)."""
        norm_cfg = self._resolve_normalization_cfg()
        validation_cfg = self._resolve_normalization_validation_cfg(norm_cfg if isinstance(norm_cfg, Mapping) else {})
        if not norm_cfg:
            self._normalization_report = {
                "enabled": False,
                "status": "missing_config",
                "method": None,
                "scope": None,
                "validation": {"enabled": False, "status": "missing_config"},
            }
            return features_df

        enabled = bool(norm_cfg.get("enabled", False))
        method = str(norm_cfg.get("method", "")).strip().lower()
        scope = str(norm_cfg.get("scope", "")).strip().lower()
        strict = bool(norm_cfg.get("strict", False))
        self._normalization_report = {
            "enabled": bool(enabled),
            "status": "disabled" if not enabled else "pending",
            "method": method or None,
            "scope": scope or None,
            "strict": bool(strict),
            "validation": {
                "enabled": bool(validation_cfg.get("enabled", False)),
                "status": "pending" if bool(validation_cfg.get("enabled", False)) and enabled else "disabled",
            },
        }
        if not enabled:
            return features_df
        if method != "combat":
            self._normalization_report["status"] = "unsupported_method"
            self._normalization_report["reason"] = f"method={method or '<empty>'}"
            logger.warning(
                "Normalization enabled for %s but method '%s' is unsupported; skipping",
                self.ds_id,
                method or "<empty>",
            )
            return features_df
        if scope != "post_features":
            self._normalization_report["status"] = "unsupported_scope"
            self._normalization_report["reason"] = f"scope={scope or '<empty>'}"
            logger.info(
                "Normalization scope '%s' is not implemented for summarize (%s); skipping",
                scope or "<empty>",
                self.ds_id,
            )
            return features_df

        try:
            from neuroCombat import neuroCombat  # type: ignore
        except Exception as exc:
            message = f"neuroCombat import failed: {exc}"
            self._normalization_report["status"] = "failed_import"
            self._normalization_report["reason"] = message
            if strict:
                raise RuntimeError(message) from exc
            logger.warning("%s; continuing without normalization", message)
            return features_df

        if len(features_df) == 0:
            self._normalization_report["status"] = "skipped_empty_features"
            return features_df

        batch_key = str(norm_cfg.get("batch_key", "")).strip()
        covariate_keys = [
            str(v).strip()
            for v in (norm_cfg.get("covariates", []) if isinstance(norm_cfg.get("covariates", []), list) else [])
            if str(v).strip()
        ]
        combat_cfg = norm_cfg.get("combat", {}) if isinstance(norm_cfg.get("combat", {}), Mapping) else {}
        family_cfg = self._resolve_combat_family_cfg(combat_cfg)
        chunk_size = max(1, int(combat_cfg.get("chunk_size", 24) or 24))
        min_batch_size = max(1, int(combat_cfg.get("min_batch_size", 2) or 2))
        min_feature_observations = max(4, int(combat_cfg.get("min_feature_observations", 16) or 16))
        variance_epsilon = float(combat_cfg.get("variance_epsilon", 1e-9) or 1e-9)
        winsorize_q = self._parse_winsorize_quantiles(
            combat_cfg.get("winsorize_quantiles", [0.005, 0.995])
        )

        if not batch_key:
            self._normalization_report["status"] = "skipped_missing_batch_key"
            logger.warning("Normalization enabled for %s but batch_key is empty; skipping ComBat", self.ds_id)
            return features_df

        subject_ids = self._anchor_subject_ids(features_df)
        if len(subject_ids) != len(features_df):
            self._normalization_report["status"] = "skipped_subject_resolution_failed"
            logger.warning("Could not align subjects to rows for ComBat (%s); skipping", self.ds_id)
            return features_df
        subject_series = pd.Series(subject_ids, index=features_df.index, dtype="object")
        unique_subjects = pd.unique(subject_series)
        meta_by_subject: Dict[str, Dict[str, Any]] = {
            str(subject): self.participant_meta_for(str(subject))
            for subject in unique_subjects
        }

        batch_candidates = self._normalization_key_candidates(batch_key)
        batch_by_subject: Dict[str, Any] = {
            sub: self._lookup_meta_value(meta_by_subject.get(sub, {}), batch_candidates)
            for sub in meta_by_subject.keys()
        }
        batch_series = subject_series.map(batch_by_subject)
        valid_mask = ~batch_series.map(self._is_missing_meta_value).to_numpy(dtype=bool)
        if not np.any(valid_mask):
            self._normalization_report["status"] = "skipped_no_batch_values"
            self._normalization_report["batch_key"] = batch_key
            logger.warning(
                "ComBat skipped for %s: no rows had batch metadata key '%s'",
                self.ds_id,
                batch_key,
            )
            return features_df

        batch_series_valid = batch_series.loc[valid_mask].astype(str).str.strip()
        batch_counts = batch_series_valid.value_counts()
        keep_batches = batch_counts[batch_counts >= int(min_batch_size)].index.tolist()
        if keep_batches:
            keep_mask = batch_series.astype(str).isin(keep_batches).to_numpy(dtype=bool)
            valid_mask = valid_mask & keep_mask
            batch_series_valid = batch_series.loc[valid_mask].astype(str).str.strip()
            batch_counts = batch_series_valid.value_counts()
        if int(batch_counts.shape[0]) < 2:
            self._normalization_report["status"] = "skipped_single_batch"
            self._normalization_report["batch_key"] = batch_key
            self._normalization_report["batch_counts"] = {
                str(k): int(v) for k, v in batch_counts.to_dict().items()
            }
            logger.warning(
                "ComBat skipped for %s: expected >=2 batches after filtering, got %d",
                self.ds_id,
                int(batch_counts.shape[0]),
            )
            return features_df

        valid_index = features_df.index[valid_mask]
        covars_df = pd.DataFrame(
            {"batch": batch_series.loc[valid_mask].astype(str).str.strip().tolist()},
            index=np.arange(int(valid_mask.sum())),
        )
        categorical_cols: List[str] = []
        continuous_cols: List[str] = []
        covariate_coverage: Dict[str, float] = {}
        for cov_key in covariate_keys:
            cov_candidates = self._normalization_key_candidates(cov_key)
            cov_by_subject = {
                sub: self._lookup_meta_value(meta_by_subject.get(sub, {}), cov_candidates)
                for sub in meta_by_subject.keys()
            }
            cov_full = subject_series.map(cov_by_subject)
            cov_valid = cov_full.loc[valid_mask]
            observed = cov_valid.map(lambda v: 0 if self._is_missing_meta_value(v) else 1)
            coverage = float(observed.mean()) if len(observed) else 0.0
            covariate_coverage[cov_key] = coverage
            if coverage <= 0.0:
                continue
            coerced, kind = self._prepare_combat_covariate(cov_valid)
            covars_df[cov_key] = coerced.to_list()
            if kind == "continuous":
                continuous_cols.append(cov_key)
            else:
                categorical_cols.append(cov_key)

        feature_cols = projection.select_export_feature_columns(features_df)
        if not feature_cols:
            self._normalization_report["status"] = "skipped_no_numeric_features"
            logger.warning("ComBat skipped for %s: no numeric feature columns found", self.ds_id)
            return features_df
        family_groups = self._build_combat_family_groups(feature_cols, family_cfg)
        family_stats: Dict[str, Dict[str, Any]] = {
            family: {
                "feature_columns_total": int(len(cols)),
                "feature_columns_harmonized": 0,
                "chunks_total": 0,
                "chunks_skipped": 0,
            }
            for family, cols in family_groups.items()
        }
        chunks_total = int(
            sum(
                (len(cols) + chunk_size - 1) // chunk_size
                for cols in family_groups.values()
                if len(cols) > 0
            )
        )

        validation_enabled = bool(validation_cfg.get("enabled", False))
        pre_probe_df: Optional[pd.DataFrame] = None
        batch_probe_series: Optional[pd.Series] = None
        target_probe_map: Dict[str, pd.Series] = {}
        if validation_enabled:
            probe_index = self._sample_probe_indices(valid_index, validation_cfg)
            probe_feature_cols = self._sample_probe_feature_columns(feature_cols, validation_cfg)
            if len(probe_index) > 0 and probe_feature_cols:
                pre_probe_df = features_df.loc[probe_index, probe_feature_cols].apply(pd.to_numeric, errors="coerce")
                probe_batch_key = str(validation_cfg.get("batch_key", "auto")).strip().lower() or "auto"
                if probe_batch_key == "auto":
                    batch_probe_series = batch_series.loc[probe_index]
                else:
                    probe_batch_candidates = self._normalization_key_candidates(probe_batch_key)
                    probe_batch_by_subject = {
                        sub: self._lookup_meta_value(meta_by_subject.get(sub, {}), probe_batch_candidates)
                        for sub in meta_by_subject.keys()
                    }
                    probe_batch_series_full = subject_series.map(probe_batch_by_subject)
                    batch_probe_series = probe_batch_series_full.loc[probe_index]
                for target_key in validation_cfg.get("target_keys", []):
                    target_name = str(target_key).strip()
                    if not target_name:
                        continue
                    target_candidates = self._normalization_key_candidates(target_name)
                    target_by_subject = {
                        sub: self._lookup_meta_value(meta_by_subject.get(sub, {}), target_candidates)
                        for sub in meta_by_subject.keys()
                    }
                    target_full = subject_series.map(target_by_subject)
                    target_probe_map[target_name] = target_full.loc[probe_index]
            else:
                self._normalization_report["validation"] = {
                    "enabled": True,
                    "status": "insufficient_probe_sample",
                    "rows_sampled": int(len(probe_index)),
                    "features_sampled": int(len(probe_feature_cols)),
                }

        harmonized_columns = 0
        skipped_columns: Dict[str, int] = {"low_support": 0, "zero_variance": 0}
        chunk_counter = 0
        for family_name, family_cols in family_groups.items():
            family_state = family_stats.get(
                family_name,
                {
                    "feature_columns_total": int(len(family_cols)),
                    "feature_columns_harmonized": 0,
                    "chunks_total": 0,
                    "chunks_skipped": 0,
                },
            )
            if not family_cols:
                family_stats[family_name] = family_state
                continue

            for i in range(0, len(family_cols), chunk_size):
                chunk_counter += 1
                chunk_cols = family_cols[i : i + chunk_size]
                family_state["chunks_total"] = int(family_state.get("chunks_total", 0)) + 1
                chunk_df = features_df.loc[valid_index, chunk_cols].apply(pd.to_numeric, errors="coerce")
                chunk_matrix = chunk_df.to_numpy(dtype=np.float64, copy=True)
                prepared_rows: List[np.ndarray] = []
                prepared_cols: List[str] = []
                missing_masks: List[np.ndarray] = []

                for col_idx, col_name in enumerate(chunk_cols):
                    col_values = chunk_matrix[:, col_idx]
                    finite_mask = np.isfinite(col_values)
                    finite_count = int(finite_mask.sum())
                    if finite_count < min_feature_observations:
                        skipped_columns["low_support"] = skipped_columns.get("low_support", 0) + 1
                        continue
                    finite_vals = col_values[finite_mask]
                    if winsorize_q is not None:
                        q_low, q_high = winsorize_q
                        try:
                            lo, hi = np.nanquantile(finite_vals, [q_low, q_high])
                        except Exception:
                            lo, hi = np.nan, np.nan
                        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                            finite_vals = np.clip(finite_vals, lo, hi)
                    if float(np.nanstd(finite_vals)) <= variance_epsilon:
                        skipped_columns["zero_variance"] = skipped_columns.get("zero_variance", 0) + 1
                        continue
                    filled = col_values.copy()
                    fill_value = float(np.nanmedian(finite_vals))
                    filled[finite_mask] = finite_vals
                    filled[~finite_mask] = fill_value
                    prepared_rows.append(filled.astype(np.float64))
                    prepared_cols.append(str(col_name))
                    missing_masks.append(~finite_mask)

                if not prepared_rows:
                    family_state["chunks_skipped"] = int(family_state.get("chunks_skipped", 0)) + 1
                    continue
                # neuroCombat becomes numerically unstable for a single-feature chunk
                # (observed as all-NaN output), so keep that column unchanged.
                if len(prepared_rows) == 1:
                    skipped_columns["single_feature_family"] = skipped_columns.get("single_feature_family", 0) + 1
                    family_state["chunks_skipped"] = int(family_state.get("chunks_skipped", 0)) + 1
                    logger.warning(
                        "Skipping ComBat chunk %d/%d (%s) for %s: single eligible feature (%s) would yield unstable harmonization",
                        chunk_counter,
                        chunks_total,
                        family_name,
                        self.ds_id,
                        prepared_cols[0],
                    )
                    continue
                dat = np.vstack(prepared_rows)
                combat_result = None
                try:
                    combat_result = neuroCombat(
                        dat=dat,
                        covars=covars_df,
                        batch_col="batch",
                        categorical_cols=categorical_cols or None,
                        continuous_cols=continuous_cols or None,
                    )
                except Exception as exc:
                    chunk_msg = (
                        f"ComBat failed for dataset={self.ds_id}, family={family_name}, chunk={chunk_counter}/{chunks_total}: {exc}"
                    )
                    if categorical_cols or continuous_cols:
                        logger.warning("%s; retrying with batch-only model", chunk_msg)
                        try:
                            combat_result = neuroCombat(
                                dat=dat,
                                covars=covars_df[["batch"]].copy(),
                                batch_col="batch",
                                categorical_cols=None,
                                continuous_cols=None,
                            )
                        except Exception as retry_exc:
                            if strict:
                                raise RuntimeError(f"{chunk_msg}; batch-only fallback also failed: {retry_exc}") from retry_exc
                            skipped_columns["combat_failed"] = skipped_columns.get("combat_failed", 0) + int(len(prepared_cols))
                            family_state["chunks_skipped"] = int(family_state.get("chunks_skipped", 0)) + 1
                            logger.warning(
                                "Skipping ComBat chunk %d/%d (%s) for %s after fallback failure: %s",
                                chunk_counter,
                                chunks_total,
                                family_name,
                                self.ds_id,
                                retry_exc,
                            )
                            continue
                    else:
                        if strict:
                            raise RuntimeError(chunk_msg) from exc
                        skipped_columns["combat_failed"] = skipped_columns.get("combat_failed", 0) + int(len(prepared_cols))
                        family_state["chunks_skipped"] = int(family_state.get("chunks_skipped", 0)) + 1
                        logger.warning(
                            "Skipping ComBat chunk %d/%d (%s) for %s: %s",
                            chunk_counter,
                            chunks_total,
                            family_name,
                            self.ds_id,
                            exc,
                        )
                        continue

                harmonized = np.asarray(combat_result["data"], dtype=np.float64)
                for row_idx, col_name in enumerate(prepared_cols):
                    updated = harmonized[row_idx, :].copy()
                    missing = missing_masks[row_idx]
                    if missing.any():
                        updated[missing] = np.nan
                    features_df.loc[valid_index, col_name] = updated.astype(np.float32)
                    harmonized_columns += 1
                    family_state["feature_columns_harmonized"] = int(
                        family_state.get("feature_columns_harmonized", 0)
                    ) + 1
            family_stats[family_name] = family_state

        if harmonized_columns == 0:
            self._normalization_report = {
                "enabled": True,
                "status": "skipped_no_eligible_columns",
                "method": "combat",
                "scope": "post_features",
                "batch_key": batch_key,
                "batch_counts": {str(k): int(v) for k, v in batch_counts.to_dict().items()},
                "rows_total": int(len(features_df)),
                "rows_harmonized": int(valid_mask.sum()),
                "feature_columns_total": int(len(feature_cols)),
                "feature_columns_harmonized": 0,
                "covariates_used": [str(c) for c in covars_df.columns if c != "batch"],
                "covariate_coverage": covariate_coverage,
                "family_wise": {
                    "enabled": bool(family_cfg.get("enabled", False)),
                    "strategy": str(family_cfg.get("strategy", "prefix")),
                    "families": family_stats,
                    "family_count": int(len(family_stats)),
                },
                "validation": {
                    "enabled": bool(validation_enabled),
                    "status": "not_run",
                    "reason": "no_harmonized_columns",
                },
            }
            logger.warning("ComBat skipped for %s: no eligible feature columns", self.ds_id)
            return features_df

        validation_report: Dict[str, Any] = {
            "enabled": bool(validation_enabled),
            "status": "disabled" if not validation_enabled else "not_run",
        }
        if validation_enabled:
            if pre_probe_df is not None and not pre_probe_df.empty:
                post_probe_df = features_df.loc[pre_probe_df.index, pre_probe_df.columns].apply(
                    pd.to_numeric, errors="coerce"
                )
                validation_report = self._compute_normalization_validation_report(
                    pre_probe_df=pre_probe_df,
                    post_probe_df=post_probe_df,
                    batch_probe_series=batch_probe_series,
                    target_probe_map=target_probe_map,
                    validation_cfg=validation_cfg,
                )
            else:
                validation_report = {
                    "enabled": True,
                    "status": "insufficient_probe_sample",
                }

        self._normalization_report = {
            "enabled": True,
            "status": "applied",
            "method": "combat",
            "scope": "post_features",
            "batch_key": batch_key,
            "batch_counts": {str(k): int(v) for k, v in batch_counts.to_dict().items()},
            "rows_total": int(len(features_df)),
            "rows_harmonized": int(valid_mask.sum()),
            "feature_columns_total": int(len(feature_cols)),
            "feature_columns_harmonized": int(harmonized_columns),
            "chunk_size": int(chunk_size),
            "min_batch_size": int(min_batch_size),
            "min_feature_observations": int(min_feature_observations),
            "winsorize_quantiles": list(winsorize_q) if winsorize_q is not None else None,
            "covariates_used": [str(c) for c in covars_df.columns if c != "batch"],
            "covariate_coverage": covariate_coverage,
            "skipped_columns": skipped_columns,
            "family_wise": {
                "enabled": bool(family_cfg.get("enabled", False)),
                "strategy": str(family_cfg.get("strategy", "prefix")),
                "families": family_stats,
                "family_count": int(len(family_stats)),
            },
            "validation": validation_report,
        }
        logger.info(
            "Applied ComBat normalization for %s (rows=%d/%d, columns=%d/%d, batches=%d)",
            self.ds_id,
            int(valid_mask.sum()),
            int(len(features_df)),
            int(harmonized_columns),
            int(len(feature_cols)),
            int(batch_counts.shape[0]),
        )
        return features_df

    def run(self) -> None:
        """Run the main workflow for this component."""
        logger.info(f"Summarizing {self.ds_id}")
        ds_path = self.processed_dir / self.ds_id
        self.participants_df = load_participant_table(self.received_dir, self.ds_id, self.config)
        self._build_participant_meta_map()
        self.index_df = self._read_index(ds_path)
        self._build_index_basename_cache()

        features_df = self._read_features(ds_path)
        if features_df is None:
            return

        features_df = self._apply_subject_filter(features_df)
        if features_df is None:
            return

        features_df = self._apply_qc_filters(features_df)
        features_df = self._apply_feature_normalization(features_df)

        try:
            grouping_items = self._build_groupings(features_df)
        except Exception:
            logger.exception("Failed to build groupings for %s", self.ds_id)
            return
        if not grouping_items:
            logger.warning(f"No groups found for {self.subject_filter or 'any subject'} in {self.ds_id}")
            return

        mnps_dir = self._create_output_dir(ds_path)
        normalization_report_info = self._write_normalization_report_file(mnps_dir)
        self._normalization_report["report_file"] = dict(normalization_report_info)
        stage_mapping_qc_info: Dict[str, Any] = {
            "schema": "mndm.stage_mapping_qc.v1",
            "status": "none",
            "path": None,
            "subjects": 0,
        }
        block_native_qc_info: Dict[str, Any] = {
            "schema": "mndm.block_native_qc.v1",
            "status": "none",
            "path": None,
            "subjects": 0,
        }
        run_fatal_error: Optional[Exception] = None
        try:
            self._prepare_one_shot_anchor(features_df, mnps_dir)
            self._write_features_snapshot(mnps_dir, features_df)
            max_workers = min(max(1, self.n_jobs), len(grouping_items), multiprocessing.cpu_count())
            if max_workers > 1:
                logger.info(
                    "Using %d summarize workers for %s (%d grouped recordings)",
                    max_workers,
                    self.ds_id,
                    len(grouping_items),
                )
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = [
                        ex.submit(self._process_grouping_item, ds_path, mnps_dir, grouping_key, sub_frame)
                        for grouping_key, sub_frame in grouping_items
                    ]
                    for fut in futures:
                        fut.result()
            else:
                for grouping_key, sub_frame in grouping_items:
                    self._process_grouping_item(ds_path, mnps_dir, grouping_key, sub_frame)
        except Exception as exc:
            run_fatal_error = exc
            self._record_run_error(
                {
                    "stage": "dataset_run",
                    "dataset_id": self.ds_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                }
            )
            logger.exception("Summarize run failed for %s", self.ds_id)
        finally:
            stage_mapping_qc_info = self._write_stage_mapping_qc_file(mnps_dir)
            block_native_qc_info = self._write_block_native_qc_file(mnps_dir)
            run_errors_info = self._write_run_errors_file(
                mnps_dir,
                total_groupings=len(grouping_items),
            )
            run_status = "completed"
            if run_fatal_error is not None:
                run_status = "failed"
            elif int(run_errors_info.get("count", 0) or 0) > 0:
                run_status = "completed_with_errors"
            fatal_error_summary = (
                {
                    "type": type(run_fatal_error).__name__,
                    "message": str(run_fatal_error),
                }
                if run_fatal_error is not None
                else None
            )

            # Write a run-level manifest for quick inspection (humans + LLMs).
            try:
                write_run_manifest(
                    mnps_dir=mnps_dir,
                    config=self.config,
                    ds_id=self.ds_id,
                    received_dir=self.received_dir,
                    processed_dir=self.processed_dir,
                    h5_mode=self.h5_mode,
                    config_path=self.config_path,
                    extra={
                        "summarize_policy": {
                            "allow_group_collisions": bool(self.allow_group_collisions),
                            "qc_policy": self.qc_policy,
                            "n_jobs": int(self.n_jobs),
                            "fit_anchor": bool(self._anchor_auto_fit_enabled()),
                        },
                        "reproducibility": self.ctx.reproducibility,
                        "grouping_collisions": self.grouping_collision_info,
                        "normalization": self._normalization_report,
                        "normalization_report": normalization_report_info,
                        "stage_mapping_qc": stage_mapping_qc_info,
                        "block_native_qc": block_native_qc_info,
                        "run_status": run_status,
                        "run_errors": run_errors_info,
                        "fatal_error": fatal_error_summary,
                    },
                )
            except Exception:
                logger.exception("Failed to write run_manifest.json for %s (%s)", self.ds_id, mnps_dir)

        if run_fatal_error is not None:
            raise run_fatal_error

    def _normalize_grouping_key(
        self,
        grouping_key: tuple[Any, Any, Any, Any, Any],
    ) -> tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Internal helper: normalize grouping key."""
        sub_id, ses_id, raw_task, run_id, acq_id = grouping_key
        if pd.isna(ses_id):
            ses_id = None
        if pd.isna(raw_task):
            raw_task = None
        if pd.isna(run_id):
            run_id = None
        if pd.isna(acq_id):
            acq_id = None
        sub_id = sub_id if str(sub_id).startswith("sub-") else f"sub-{str(sub_id).zfill(3)}"
        return str(sub_id), ses_id, raw_task, run_id, acq_id

    def _process_grouping_item(
        self,
        ds_path: Path,
        mnps_dir: Path,
        grouping_key: tuple[Any, Any, Any, Any, Any],
        sub_frame: pd.DataFrame,
    ) -> None:
        """Internal helper: process grouping item."""
        sub_id, ses_id, raw_task, run_id, acq_id = self._normalize_grouping_key(grouping_key)
        try:
            runner = SubjectSummaryRunner(
                dataset_runner=self,
                ds_path=ds_path,
                mnps_dir=mnps_dir,
                index_df=self.index_df,
            )
            runner.run(
                sub_id=sub_id,
                ses_id=ses_id,
                raw_task=raw_task,
                run_id=run_id,
                acq_id=acq_id,
                sub_frame=sub_frame,
            )
        except Exception as exc:
            representative_file = None
            if "file" in sub_frame.columns and len(sub_frame) > 0:
                representative_file = str(sub_frame["file"].iloc[0])
            self._record_run_error(
                {
                    "stage": "grouping",
                    "dataset_id": self.ds_id,
                    "subject": sub_id,
                    "session": ses_id,
                    "task": raw_task,
                    "run": run_id,
                    "acq": acq_id,
                    "rows": int(len(sub_frame)),
                    "representative_file": representative_file,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                }
            )
            logger.exception(
                "Summarize failed for %s",
                build_dataset_label(
                    ds_id=self.ds_id,
                    sub_id=sub_id,
                    ses_id=ses_id,
                    condition=None,
                    task=raw_task,
                    run=run_id,
                    acq=acq_id,
                ),
            )

    def _record_run_error(self, error_entry: Mapping[str, Any]) -> None:
        """Store one run error entry in a thread-safe way."""
        with self._run_errors_lock:
            self._run_errors.append(dict(error_entry))

    def _record_stage_mapping_qc_entry(self, entry: Mapping[str, Any]) -> None:
        """Store one per-subject stage-mapping QC entry in a thread-safe way."""
        with self._stage_mapping_qc_lock:
            self._stage_mapping_qc_entries.append(dict(entry))

    def _record_block_native_qc_entry(self, entry: Mapping[str, Any]) -> None:
        """Store one per-subject block-native QC entry in a thread-safe way."""
        with self._block_native_qc_lock:
            self._block_native_qc_entries.append(dict(entry))

    def _write_stage_mapping_qc_file(self, mnps_dir: Path) -> Dict[str, Any]:
        """Write run-level stage_mapping_qc.json when per-subject entries exist."""
        with self._stage_mapping_qc_lock:
            entries = [dict(e) for e in self._stage_mapping_qc_entries]

        summary: Dict[str, Any] = {
            "schema": "mndm.stage_mapping_qc.v1",
            "status": "none",
            "path": None,
            "subjects": int(len(entries)),
        }
        if not entries:
            return summary

        expected_freqs: set[int] = set()
        detected_freqs: set[int] = set()
        has_25 = 0
        has_30 = 0
        for row in entries:
            exp = row.get("expected_frequencies_hz", [])
            if isinstance(exp, Sequence) and not isinstance(exp, (str, bytes)):
                for v in exp:
                    try:
                        expected_freqs.add(int(v))
                    except Exception:
                        continue
            det = row.get("detected_raw_frequencies_hz", [])
            if isinstance(det, Sequence) and not isinstance(det, (str, bytes)):
                for v in det:
                    try:
                        detected_freqs.add(int(v))
                    except Exception:
                        continue
            if bool(row.get("raw_has_25hz", False)):
                has_25 += 1
            if bool(row.get("raw_has_30hz", False)):
                has_30 += 1

        expected_list = sorted(expected_freqs)
        detected_list = sorted(detected_freqs)
        missing_expected = [int(v) for v in expected_list if int(v) not in detected_freqs]

        payload = {
            "schema": "mndm.stage_mapping_qc.v1",
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "dataset_id": self.ds_id,
            "run_dir": str(mnps_dir),
            "aggregate": {
                "subjects_total": int(len(entries)),
                "subjects_with_raw_25hz": int(has_25),
                "subjects_with_raw_30hz": int(has_30),
                "expected_frequencies_hz": [int(v) for v in expected_list],
                "detected_raw_frequencies_hz": [int(v) for v in detected_list],
                "missing_expected_frequencies_hz_raw": [int(v) for v in missing_expected],
            },
            "subjects": entries,
        }
        out_path = mnps_dir / "stage_mapping_qc.json"
        try:
            json_writer.write_json_summary(payload, out_path)
            summary.update(
                {
                    "status": "written",
                    "path": out_path.name,
                    "aggregate": payload.get("aggregate", {}),
                }
            )
        except Exception as exc:
            summary["status"] = "write_failed"
            summary["error"] = str(exc)
            logger.exception("Failed to write stage_mapping_qc.json for %s (%s)", self.ds_id, out_path)
        return summary

    def _write_block_native_qc_file(self, mnps_dir: Path) -> Dict[str, Any]:
        """Write run-level block_native_qc.json when per-subject entries exist."""
        with self._block_native_qc_lock:
            entries = [dict(e) for e in self._block_native_qc_entries]

        summary: Dict[str, Any] = {
            "schema": "mndm.block_native_qc.v1",
            "status": "none",
            "path": None,
            "subjects": int(len(entries)),
        }
        if not entries:
            return summary

        blocks_total = 0
        windows_total = 0
        source_match_total = 0
        source_total = 0
        block_stage_counts: Dict[str, int] = {}
        window_stage_counts: Dict[str, int] = {}
        block_frequency_counts: Dict[str, int] = {}
        end_reason_counts: Dict[str, int] = {}
        mapping_status_counts: Dict[str, int] = {}

        def _merge_counts(target: Dict[str, int], source: Any) -> None:
            if not isinstance(source, Mapping):
                return
            for key, value in source.items():
                try:
                    iv = int(value)
                except Exception:
                    continue
                sk = str(key)
                target[sk] = target.get(sk, 0) + iv

        for row in entries:
            blocks_total += int(row.get("n_blocks", 0) or 0)
            windows_total += int(row.get("n_windows", 0) or 0)
            _merge_counts(block_stage_counts, row.get("block_counts_by_stage"))
            _merge_counts(window_stage_counts, row.get("window_counts_by_stage"))
            _merge_counts(block_frequency_counts, row.get("block_counts_by_frequency_hz"))
            _merge_counts(end_reason_counts, row.get("end_reason_counts"))
            source_stats = row.get("source_window_index", {})
            if isinstance(source_stats, Mapping):
                source_match_total += int(source_stats.get("matched", 0) or 0)
                source_total += int(source_stats.get("total", 0) or 0)
            label_cleaning = row.get("label_cleaning", {})
            if isinstance(label_cleaning, Mapping):
                _merge_counts(mapping_status_counts, label_cleaning.get("mapping_status_counts"))

        payload = {
            "schema": "mndm.block_native_qc.v1",
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "dataset_id": self.ds_id,
            "run_dir": str(mnps_dir),
            "aggregate": {
                "subjects_total": int(len(entries)),
                "blocks_total": int(blocks_total),
                "windows_total": int(windows_total),
                "source_window_match_total": int(source_match_total),
                "source_window_total": int(source_total),
                "source_window_match_fraction": float(source_match_total / source_total) if source_total > 0 else 0.0,
                "block_counts_by_stage": {str(k): int(v) for k, v in sorted(block_stage_counts.items(), key=lambda kv: kv[0])},
                "window_counts_by_stage": {str(k): int(v) for k, v in sorted(window_stage_counts.items(), key=lambda kv: kv[0])},
                "block_counts_by_frequency_hz": {
                    str(k): int(v) for k, v in sorted(block_frequency_counts.items(), key=lambda kv: kv[0])
                },
                "end_reason_counts": {str(k): int(v) for k, v in sorted(end_reason_counts.items(), key=lambda kv: kv[0])},
                "label_mapping_status_counts": {
                    str(k): int(v) for k, v in sorted(mapping_status_counts.items(), key=lambda kv: kv[0])
                },
            },
            "subjects": entries,
        }
        out_path = mnps_dir / "block_native_qc.json"
        try:
            json_writer.write_json_summary(payload, out_path)
            summary.update(
                {
                    "status": "written",
                    "path": out_path.name,
                    "aggregate": payload.get("aggregate", {}),
                }
            )
        except Exception as exc:
            summary["status"] = "write_failed"
            summary["error"] = str(exc)
            logger.exception("Failed to write block_native_qc.json for %s (%s)", self.ds_id, out_path)
        return summary

    def _write_run_errors_file(self, mnps_dir: Path, *, total_groupings: int) -> Dict[str, Any]:
        """Write run_errors.json when subject-level failures were captured."""
        with self._run_errors_lock:
            errors = [dict(entry) for entry in self._run_errors]

        summary: Dict[str, Any] = {
            "schema": "mndm.run_errors.v1",
            "status": "none",
            "path": None,
            "count": int(len(errors)),
            "groupings_total": int(total_groupings),
        }
        if not errors:
            return summary

        stage_counts: Dict[str, int] = {}
        for entry in errors:
            stage = str(entry.get("stage", "unknown"))
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

        payload = {
            "schema": "mndm.run_errors.v1",
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "dataset_id": self.ds_id,
            "run_dir": str(mnps_dir),
            "counts": {
                "errors_total": int(len(errors)),
                "groupings_total": int(total_groupings),
                "errors_by_stage": stage_counts,
            },
            "errors": errors,
        }

        out_path = mnps_dir / "run_errors.json"
        try:
            json_writer.write_json_summary(payload, out_path)
            summary["status"] = "written"
            summary["path"] = out_path.name
        except Exception as exc:
            summary["status"] = "write_failed"
            summary["error"] = str(exc)
            logger.exception("Failed to write run_errors.json for %s (%s)", self.ds_id, out_path)
        return summary

    def _write_normalization_report_file(self, mnps_dir: Path) -> Dict[str, Any]:
        """Write normalization_report.json with pre/post probe metadata."""
        summary: Dict[str, Any] = {
            "schema": "mndm.normalization_report.v1",
            "status": "none",
            "path": None,
        }
        payload = {
            "schema": "mndm.normalization_report.v1",
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "dataset_id": self.ds_id,
            "run_dir": str(mnps_dir),
            "normalization": dict(self._normalization_report),
        }
        out_path = mnps_dir / "normalization_report.json"
        try:
            json_writer.write_json_summary(payload, out_path)
            summary["status"] = "written"
            summary["path"] = out_path.name
        except Exception as exc:
            summary["status"] = "write_failed"
            summary["error"] = str(exc)
            logger.exception("Failed to write normalization_report.json for %s (%s)", self.ds_id, out_path)
        return summary

    def write_regional_csv_outputs_threadsafe(
        self,
        *,
        regional_mnps_results: Any,
        regional_mnps_cfg: Mapping[str, Any],
        mnps_dir: Path,
        config: Mapping[str, Any],
        dataset_label: str,
    ) -> None:
        """Handle write regional csv outputs threadsafe."""
        with self._dataset_csv_lock:
            write_regional_csv_outputs(
                regional_mnps_results=regional_mnps_results,
                regional_mnps_cfg=regional_mnps_cfg,
                mnps_dir=mnps_dir,
                config=config,
                dataset_label=dataset_label,
            )

    def write_stratified_blocks_csv_output_threadsafe(
        self,
        *,
        stratified_blocks_result: Any,
        config: Mapping[str, Any],
        dataset_id: str,
        mnps_dir: Path,
        dataset_label: str,
    ) -> None:
        """Handle write stratified blocks csv output threadsafe."""
        with self._dataset_csv_lock:
            write_stratified_blocks_csv_output(
                stratified_blocks_result=stratified_blocks_result,
                config=config,
                dataset_id=dataset_id,
                mnps_dir=mnps_dir,
                dataset_label=dataset_label,
            )

    def _build_features_snapshot(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Build a compact per-run features snapshot for provenance/Test-C."""
        snapshot: Dict[str, Any] = {
            "dataset_id": self.ds_id,
            "rows": int(len(features_df)),
            "columns": sorted([str(c) for c in features_df.columns]),
            "column_stats": {},
            "normalization": dict(self._normalization_report),
        }
        for col in snapshot["columns"]:
            s = features_df[col]
            col_info: Dict[str, Any] = {
                "missing_rate": float(s.isna().mean()) if len(s) else 0.0,
                "dtype": str(s.dtype),
            }
            num = pd.to_numeric(s, errors="coerce")
            finite = num[np.isfinite(num.to_numpy(dtype=np.float64, copy=False))]
            if len(finite):
                med = float(np.nanmedian(finite))
                col_info.update(
                    {
                        "mean": float(np.nanmean(finite)),
                        "std": float(np.nanstd(finite)),
                        "median": med,
                        "mad": float(np.nanmedian(np.abs(finite - med))),
                    }
                )
            snapshot["column_stats"][col] = col_info
        snapshot["features_snapshot_hash"] = _stable_hash_mapping(snapshot)
        return snapshot

    def _anchor_auto_fit_enabled(self) -> bool:
        """Return whether one-shot anchor fitting is enabled for this dataset run."""
        proj_cfg = self.config.get("mnps_projection", {}) if isinstance(self.config, Mapping) else {}
        auto_cfg = proj_cfg.get("anchor_auto_fit", {}) if isinstance(proj_cfg, Mapping) else {}
        return bool(auto_cfg.get("enabled", False)) if isinstance(auto_cfg, Mapping) else False

    def _anchor_auto_fit_config(self) -> dict[str, Any]:
        """Return the normalized one-shot anchor-fit config block."""
        proj_cfg = self.config.get("mnps_projection", {}) if isinstance(self.config, Mapping) else {}
        auto_cfg = proj_cfg.get("anchor_auto_fit", {}) if isinstance(proj_cfg, Mapping) else {}
        return dict(auto_cfg) if isinstance(auto_cfg, Mapping) else {}

    def _anchor_subject_ids(self, features_df: pd.DataFrame) -> list[str]:
        """Resolve per-row subject ids for one-shot anchor fitting."""
        if "subject" in features_df.columns:
            return [self._normalize_subject(v) or "sub-unknown" for v in features_df["subject"].tolist()]
        if "subject_id" in features_df.columns:
            return [self._normalize_subject(v) or "sub-unknown" for v in features_df["subject_id"].tolist()]
        if "file" in features_df.columns:
            subject_ids: list[str] = []
            for raw in features_df["file"].tolist():
                subject, _, _, _, _ = self._parse_file_entities(str(raw or ""))
                subject_ids.append(subject if subject and subject != "sub-unknown" else "sub-unknown")
            return subject_ids
        return ["sub-unknown"] * len(features_df)

    def _anchor_group_by_subject(
        self,
        features_df: pd.DataFrame,
        subject_ids: list[str],
    ) -> dict[str, str]:
        """Resolve subject->group provenance for one-shot anchor fitting."""
        out: dict[str, str] = {}
        if len(subject_ids) != len(features_df):
            return out
        file_values = features_df["file"].tolist() if "file" in features_df.columns else [None] * len(features_df)
        first_file_by_subject: dict[str, str] = {}
        first_session_by_subject: dict[str, Optional[str]] = {}
        for subject, raw_file in zip(subject_ids, file_values):
            if not subject or subject in first_file_by_subject:
                continue
            file_text = str(raw_file or "")
            first_file_by_subject[subject] = file_text
            _, ses_id, _, _, _ = self._parse_file_entities(file_text)
            first_session_by_subject[subject] = ses_id
        for subject in sorted(set(subject_ids)):
            if not subject or subject == "sub-unknown":
                continue
            participant_meta = self.participant_meta_for(subject)
            file_text = first_file_by_subject.get(subject)
            mapped = extract_mapped_metadata(
                participant_meta,
                self.config,
                self.ds_id,
                first_session_by_subject.get(subject),
                filename=file_text,
            )
            group = str(mapped.get("group") or "").strip()
            if group:
                out[subject] = group
        return out

    def _prepare_one_shot_anchor(self, features_df: pd.DataFrame, mnps_dir: Path) -> Optional[Path]:
        """Fit, freeze, and inject a one-shot cohort anchor before worker launch."""
        if not self._anchor_auto_fit_enabled():
            return None
        proj_cfg = self.config.setdefault("mnps_projection", {})
        if not isinstance(proj_cfg, Mapping):
            raise ValueError("mnps_projection must be a mapping for one-shot anchor fitting")
        proj_cfg = dict(proj_cfg)
        self.config["mnps_projection"] = proj_cfg
        existing_anchor_cfg = proj_cfg.get("anchor", {})
        existing_enabled = bool(existing_anchor_cfg.get("enabled", False)) if isinstance(existing_anchor_cfg, Mapping) else False
        if existing_enabled:
            raise ValueError("Cannot combine mnps_projection.anchor.enabled with mnps_projection.anchor_auto_fit.enabled")

        auto_cfg = self._anchor_auto_fit_config()
        scale_method = str(auto_cfg.get("scale_method", "iqr") or "iqr").strip().lower()
        min_subjects = int(auto_cfg.get("min_subjects", 3) or 3)
        anchor_id = str(auto_cfg.get("anchor_id") or f"{self.ds_id}_all_subjects_{scale_method}_v2_1").strip()
        anchor_source = str(auto_cfg.get("anchor_source") or "all_subjects_features_table").strip()
        cohort_filter = str(auto_cfg.get("cohort_filter") or "all usable rows after summarize QC filters").strip()
        file_ids = [str(v or "") for v in features_df["file"].tolist()] if "file" in features_df.columns else None
        subject_ids = self._anchor_subject_ids(features_df)
        group_by_subject = self._anchor_group_by_subject(features_df, subject_ids)

        anchor_artifact = anchors.fit_feature_anchors_from_features_df(
            features_df,
            anchor_id=anchor_id,
            anchor_source=anchor_source,
            cohort_filter=cohort_filter,
            feature_standardization=proj_cfg.get("feature_standardization") if isinstance(proj_cfg, Mapping) else None,
            min_subjects=min_subjects,
            scale_method=scale_method,
            subject_ids=subject_ids,
            file_ids=file_ids,
            group_by_subject=group_by_subject,
        )
        anchors_dir = mnps_dir / "anchors"
        anchor_path = anchors.save_anchor_file(anchor_artifact, anchors_dir / f"{anchor_id}.json")
        anchor_cfg = dict(existing_anchor_cfg) if isinstance(existing_anchor_cfg, Mapping) else {}
        anchor_cfg.update(
            {
                "enabled": True,
                "path": str(anchor_path),
                "scale_method": scale_method,
                "min_subjects": min_subjects,
            }
        )
        proj_cfg["anchor"] = anchor_cfg
        logger.info("Fitted one-shot feature anchor for %s: %s", self.ds_id, anchor_path)
        return anchor_path

    def _write_features_snapshot(self, mnps_dir: Path, features_df: pd.DataFrame) -> None:
        """Write features_snapshot.json under the current run directory."""
        try:
            payload = self._build_features_snapshot(features_df)
            out_path = mnps_dir / "features_snapshot.json"
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("Wrote features snapshot: %s", out_path)
        except Exception:
            logger.exception("Failed to write features snapshot for %s", self.ds_id)

    def _build_participant_meta_map(self) -> None:
        """Build O(1) participant metadata lookup by participant_id."""
        self._participant_meta_map = {}
        if self.participants_df is None or self.participants_df.empty:
            return
        if "participant_id" not in self.participants_df.columns:
            return
        try:
            for _, row in self.participants_df.iterrows():
                pid = str(row.get("participant_id", "")).strip()
                if not pid or pid in self._participant_meta_map:
                    continue
                self._participant_meta_map[pid] = row.to_dict()
        except Exception:
            self._participant_meta_map = {}

    def _build_index_basename_cache(self) -> None:
        """Build O(1) basename -> relative-path list cache from file_index."""
        self._index_paths_by_basename = {}
        if self.index_df is None or self.index_df.empty or "path" not in self.index_df.columns:
            return
        try:
            for p in self.index_df["path"].astype(str).tolist():
                b = Path(p).name
                if not b:
                    continue
                self._index_paths_by_basename.setdefault(b, []).append(p)
        except Exception:
            self._index_paths_by_basename = {}

    def _lookup_rel_paths_by_file_value(self, file_value: str) -> List[str]:
        """Return candidate relative paths for a `features.csv` file value."""
        if not file_value:
            return []
        if not self._index_paths_by_basename:
            return []
        name = Path(str(file_value).replace("\\", "/")).name
        return list(self._index_paths_by_basename.get(name, []))

    def _read_index(self, ds_path: Path) -> Optional[pd.DataFrame]:
        """Internal helper: read index."""
        index_path = ds_path / "file_index.csv"
        index_df: Optional[pd.DataFrame] = None
        if not index_path.exists():
            index_df = self._build_index_from_received(ds_path)
        elif index_path.stat().st_size == 0:
            logger.warning("Empty file_index.csv for %s; rebuilding", self.ds_id)
            try:
                index_path.unlink()
            except Exception as e:
                logger.warning("Failed to remove empty file_index.csv for %s: %s", self.ds_id, e)
            index_df = self._build_index_from_received(ds_path)
        else:
            try:
                index_df = pd.read_csv(index_path)
            except pd.errors.EmptyDataError:
                logger.warning("Empty file_index.csv for %s; rebuilding", self.ds_id)
                try:
                    index_path.unlink()
                except Exception as e:
                    logger.warning("Failed to remove empty file_index.csv for %s: %s", self.ds_id, e)
                index_df = self._build_index_from_received(ds_path)
            except Exception:
                logger.exception("Failed to read file_index.csv for %s", self.ds_id)
                return None
        if index_df is None:
            return None
        filtered, excluded_count, excluded_patterns = apply_exclude_file_filters(
            index_df,
            config=self.config,
            candidate_columns=("path",),
        )
        if excluded_count > 0:
            logger.info(
                "Excluded %s indexed files for %s via exclude-files=%s",
                excluded_count,
                self.ds_id,
                excluded_patterns,
            )
        return filtered

    def _build_index_from_received(self, ds_path: Path) -> Optional[pd.DataFrame]:
        """Internal helper: build index from received."""
        ds_root = bids_index.resolve_dataset_root(self.config, self.ctx.received_dir, self.ds_id)
        if not ds_root.exists():
            logger.warning("Dataset root missing at %s; cannot build file_index.csv", ds_root)
            return None
        try:
            logger.info("Building file index for %s from %s", self.ds_id, ds_root)
            index_df = bids_index.build_file_index(ds_root, config=self.config, dataset_id=self.ds_id)
            index_df.to_csv(ds_path / "file_index.csv", index=False)
            logger.info("Saved file index: %s", ds_path / "file_index.csv")
            return index_df
        except Exception:
            logger.exception("Failed to build file_index.csv for %s from %s", self.ds_id, ds_root)
            return None

    def _dataset_root(self) -> Path:
        """Resolve effective dataset root, honoring per-dataset overrides."""
        return bids_index.resolve_dataset_root(self.config, self.ctx.received_dir, self.ds_id)

    def _read_features(self, ds_path: Path) -> Optional[pd.DataFrame]:
        """Internal helper: read features."""
        storage_cfg = self.config.get("feature_storage", {}) if isinstance(self.config, Mapping) else {}
        read_prefer = str(storage_cfg.get("read_prefer", "parquet")).strip().lower() if isinstance(storage_cfg, Mapping) else "parquet"
        if read_prefer not in {"csv", "parquet"}:
            read_prefer = "parquet"

        candidates = [ds_path / "features.parquet", ds_path / "features.csv"]
        if read_prefer == "csv":
            candidates = [ds_path / "features.csv", ds_path / "features.parquet"]
        features_path = next((p for p in candidates if p.exists()), None)
        if features_path is None:
            logger.warning(f"No features found for {self.ds_id}, skipping")
            return None
        try:
            if features_path.suffix.lower() == ".parquet":
                features_df = pd.read_parquet(features_path)
            else:
                features_df = pd.read_csv(features_path)
        except Exception:
            logger.exception("Failed to read features table %s for %s", features_path, self.ds_id)
            return None
        features_df, excluded_count, excluded_patterns = apply_exclude_file_filters(
            features_df,
            config=self.config,
            candidate_columns=("file", "path"),
        )
        if excluded_count > 0:
            logger.info(
                "Excluded %s feature rows for %s via exclude-files=%s",
                excluded_count,
                self.ds_id,
                excluded_patterns,
            )
        if features_df.empty:
            logger.warning(f"Features dataframe empty for {self.ds_id}")
            return None
        return features_df

    def _apply_subject_filter(self, features_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Internal helper: apply subject filter."""
        if not self.subject_filter:
            return features_df
        if "file" not in features_df.columns:
            logger.warning("Subject filter requested but 'file' column missing; skipping subject filter")
            return features_df
        file_series = features_df["file"].astype(str)

        # Robust path: resolve effective subject per file using the same parser
        # as grouping (BIDS + optional metadata_extraction filename_parse regex).
        unique_files = pd.unique(file_series)
        for file_name in unique_files:
            if file_name not in self._file_entity_cache:
                self._file_entity_cache[file_name] = self._parse_file_entities(file_name)
        parsed_subjects = file_series.map(lambda f: self._file_entity_cache.get(f, ("sub-unknown", None, None, None, None))[0])
        mask = parsed_subjects == self.subject_filter

        # Backward-compatible fallback for legacy datasets where file parsing is
        # intentionally minimal and subject ID appears directly in filename text.
        if not bool(mask.any()):
            legacy_mask = file_series.str.contains(self.subject_filter, na=False)
            if self.subject_filter.startswith("sub-"):
                legacy_mask = legacy_mask | file_series.str.contains(self.subject_filter[4:], na=False)
            mask = legacy_mask

        filtered = features_df[mask]
        if filtered.empty:
            logger.warning(f"No epochs for subject {self.subject_filter} in {self.ds_id}")
            return None
        return filtered

    def _apply_qc_filters(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Internal helper: apply qc filters."""
        qc_cols = [c for c in features_df.columns if str(c).startswith("qc_ok_")]
        if not qc_cols:
            return features_df

        policy = self.qc_policy
        before = len(features_df)
        if policy == "eeg_only":
            if "qc_ok_eeg" not in features_df.columns:
                logger.info("QC policy 'eeg_only' requested but qc_ok_eeg missing; skipping QC filtering")
                return features_df
            features_df = features_df[features_df["qc_ok_eeg"].fillna(1).astype(int) == 1]
            dropped = before - len(features_df)
            if dropped > 0:
                logger.info("Dropped %d/%d epochs by qc_ok_eeg (policy=eeg_only)", dropped, before)
            return features_df

        if policy == "all_qc_ok":
            mask = np.ones(len(features_df), dtype=bool)
            dropped_by: Dict[str, int] = {}
            for c in qc_cols:
                col_ok = features_df[c].fillna(1).astype(int) == 1
                dropped_by[str(c)] = int((~col_ok).sum())
                mask &= col_ok
            features_df = features_df[mask]
            dropped = before - len(features_df)
            if dropped > 0:
                logger.info(
                    "Dropped %d/%d epochs by all_qc_ok policy; dropped_by=%s",
                    dropped,
                    before,
                    dropped_by,
                )
            return features_df

        if policy == "any_ok":
            mask = np.zeros(len(features_df), dtype=bool)
            for c in qc_cols:
                mask |= features_df[c].fillna(1).astype(int) == 1
            features_df = features_df[mask]
            dropped = before - len(features_df)
            if dropped > 0:
                logger.info("Dropped %d/%d epochs by any_ok policy", dropped, before)
            return features_df

        logger.warning(
            "Unknown summarize.qc_policy='%s'; falling back to eeg_only",
            policy,
        )
        if "qc_ok_eeg" in features_df.columns:
            before = len(features_df)
            features_df = features_df[features_df["qc_ok_eeg"].fillna(1) == 1]
            dropped = before - len(features_df)
            if dropped > 0:
                logger.info("Dropped %d/%d epochs by qc_ok_eeg", dropped, before)
        return features_df

    def _build_groupings(self, features_df: pd.DataFrame):
        """Group features by (subject, session, task, run, acq) for separate H5 output per combination."""
        if "file" in features_df.columns:
            # Parse subject/session/task/run/acq once per unique file value.
            file_series = features_df["file"].astype(str)
            unique_files = pd.unique(file_series)
            for f in unique_files:
                if f not in self._file_entity_cache:
                    self._file_entity_cache[f] = self._parse_file_entities(f)
            parsed_rows = [self._file_entity_cache[f] for f in file_series.tolist()]
            if parsed_rows:
                subj_vals, ses_vals, task_vals, run_vals, acq_vals = zip(*parsed_rows)
            else:
                subj_vals, ses_vals, task_vals, run_vals, acq_vals = ([], [], [], [], [])
            features_df = features_df.assign(
                _subject=list(subj_vals),
                _session=list(ses_vals),
                _task=list(task_vals),
                _run=list(run_vals),
                _acq=list(acq_vals),
            )
            unknown_mask = features_df["_subject"] == "sub-unknown"
            if unknown_mask.any():
                if "subject" in features_df.columns:
                    features_df.loc[unknown_mask, "_subject"] = (
                        features_df.loc[unknown_mask, "subject"].astype(str).apply(lambda s: f"sub-{s.zfill(3)}")
                    )
                else:
                    logger.warning("Some feature rows lack subject identifiers; they will be skipped")
                    features_df = features_df.loc[~unknown_mask]

            # Guardrail for non-BIDS sources: if multiple distinct files resolve to
            # the same summarize key, they are merged into one output run.
            group_cols = ["_subject", "_session", "_task", "_run", "_acq"]
            key_file = features_df[group_cols + ["file"]].drop_duplicates()
            files_per_key = key_file.groupby(group_cols, dropna=False)["file"].nunique()
            collisions = files_per_key[files_per_key > 1]
            if not collisions.empty:
                merged_extra_files = int((collisions - 1).sum())
                examples: list[str] = []
                for key, n_files in collisions.head(5).items():
                    if isinstance(key, tuple) and len(key) == 5:
                        sub_id, ses_id, task_id, run_id, acq_id = key
                    else:
                        sub_id, ses_id, task_id, run_id, acq_id = key, None, None, None, None
                    examples.append(
                        f"(sub={sub_id}, ses={ses_id}, task={task_id}, run={run_id}, acq={acq_id}) -> {int(n_files)} files"
                    )
                self.grouping_collision_info = {
                    "count": int(collisions.shape[0]),
                    "merged_extra_files": int(merged_extra_files),
                    "examples": examples,
                }
                if not self.allow_group_collisions:
                    raise RuntimeError(
                        f"Detected {int(collisions.shape[0])} grouping-key collisions in {self.ds_id}. "
                        f"Set summarize.allow_group_collisions=true to merge explicitly. "
                        f"Examples: {'; '.join(examples)}"
                    )
                logger.warning(
                    "Detected %d grouping-key collisions in %s; merging is enabled. %d extra files will be merged. Examples: %s",
                    int(collisions.shape[0]),
                    self.ds_id,
                    merged_extra_files,
                    "; ".join(examples),
                )
            # Group by subject + session + task + run + acq for separate H5 per recording stream
            grouping = features_df.groupby(["_subject", "_session", "_task", "_run", "_acq"], dropna=False)
        elif "subject" in features_df.columns:
            features_df = features_df.assign(_session=None, _task=None, _run=None, _acq=None)
            grouping = features_df.groupby(["subject", "_session", "_task", "_run", "_acq"], dropna=False)
        else:
            # No subject info: treat entire dataset as single anonymous subject
            features_df = features_df.assign(_subject="sub-unknown", _session="ses-unknown", _task=None, _run=None, _acq=None)
            grouping = features_df.groupby(["_subject", "_session", "_task", "_run", "_acq"], dropna=False)

        if self.subject_filter:
            grouping_items = [item for item in grouping if item[0][0] == self.subject_filter]
        else:
            grouping_items = list(grouping)
        return grouping_items

    def _create_output_dir(self, ds_path: Path) -> Path:
        """Internal helper: create output dir."""
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        mnps_dir = ds_path / f"neuralmanifolddynamics_{self.ds_id}_{ts}"
        mnps_dir.mkdir(parents=True, exist_ok=True)
        return mnps_dir

    def participant_meta_for(self, sub_id: str) -> Dict[str, Any]:
        """Handle participant meta for."""
        if self._participant_meta_map:
            for candidate in self._participant_lookup_candidates(sub_id):
                hit = self._participant_meta_map.get(candidate)
                if hit:
                    return dict(hit)
            return {}
        if self.participants_df is None:
            return {}
        candidates = set(self._participant_lookup_candidates(sub_id))
        lookup = self.participants_df[self.participants_df["participant_id"].astype(str).isin(candidates)]
        if lookup.empty:
            return {}
        return lookup.iloc[0].to_dict()

    @staticmethod
    def _participant_lookup_candidates(sub_id: str) -> list[str]:
        """Build candidate participant-id variants for robust matching."""
        value = str(sub_id or "").strip()
        if not value:
            return []
        candidates: list[str] = [value]

        if value.lower().startswith("sub-"):
            bare = value[4:]
            if bare:
                candidates.append(bare)
        else:
            bare = value
            candidates.append(f"sub-{bare}")

        if bare.isdigit():
            stripped = bare.lstrip("0") or "0"
            for width in (3, 4, len(bare), len(stripped)):
                if width <= 0:
                    continue
                padded = stripped.zfill(width)
                candidates.append(padded)
                candidates.append(f"sub-{padded}")

        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = candidate.strip()
            if not key or key in seen:
                continue
            deduped.append(key)
            seen.add(key)
        return deduped

    def participant_meta_source_info(self) -> Dict[str, Any]:
        """Handle participant meta source info."""
        if self.participants_df is None:
            return {}
        attrs = getattr(self.participants_df, "attrs", {}) or {}
        out: Dict[str, Any] = {}
        for key in ("source_path", "source_format", "subject_id_column"):
            value = attrs.get(key)
            if value is not None:
                out[key] = value
        return out


class SubjectSummaryRunner:
    """Subject/session-level summarization."""

    def __init__(
        self,
        dataset_runner: DatasetSummaryRunner,
        ds_path: Path,
        mnps_dir: Path,
        index_df: Optional[pd.DataFrame],
    ):
        """Initialize the instance."""
        self.dataset = dataset_runner
        self.ctx = dataset_runner.ctx
        self.ds_path = ds_path
        self.mnps_dir = mnps_dir
        self.index_df = index_df
        self._confounds_path_cache: Dict[str, Optional[Path]] = {}

    def _dataset_root(self) -> Path:
        """Resolve effective dataset root, honoring per-dataset overrides."""
        return bids_index.resolve_dataset_root(self.ctx.config, self.ctx.received_dir, self.dataset.ds_id)

    def run(
        self,
        sub_id: str,
        ses_id: Optional[str],
        raw_task: Optional[str],
        run_id: Optional[str],
        acq_id: Optional[str],
        sub_frame: pd.DataFrame,
    ) -> None:
        """Run the main workflow for this component."""
        config = self.ctx.config
        mnps_cfg = self.ctx.mnps_cfg
        normalize_mode = self.ctx.normalize_override
        proj_cfg = config.get("mnps_projection", {}) if isinstance(config, Mapping) else {}
        missing_axis_policy = str(proj_cfg.get("missing_axis_policy", "nan_mask_v1")).strip().lower() or "nan_mask_v1"

        sub_id = sub_id if str(sub_id).startswith("sub-") else f"sub-{str(sub_id).zfill(3)}"
        participant_meta = self.dataset.participant_meta_for(sub_id)
        participant_meta_source = self.dataset.participant_meta_source_info()

        # Extract representative filename for task parsing (if available)
        representative_file = None
        if "file" in sub_frame.columns and len(sub_frame) > 0:
            representative_file = str(sub_frame["file"].iloc[0])

        mapped_meta = extract_mapped_metadata(
            participant_meta, config, self.dataset.ds_id, ses_id, filename=representative_file
        )

        # Resolve condition/task:
        # - Prefer explicit mapped metadata (participants.tsv + config rules)
        # - Fall back to the task parsed from filenames during grouping, to avoid
        #   overwriting multiple (subject, task) outputs into the same directory.
        condition = mapped_meta.get("condition")
        task = mapped_meta.get("task") or raw_task

        modality = str(config.get("modality", "")).strip().lower() if isinstance(config, Mapping) else ""
        if modality == "fmri":
            sub_frame = self._merge_fd_from_confounds(
                sub_frame=sub_frame,
                raw_task=raw_task,
                condition=condition,
                session=ses_id,
                run_id=run_id,
                acq_id=acq_id,
            )

        # Optional FD-based censoring (drop high-motion epochs and neighbours)
        n_before_any = int(len(sub_frame))
        regional_mnps_cfg = config.get("regional_mnps", {}) if isinstance(config, Mapping) else {}
        fd_required = (
            bool(regional_mnps_cfg.get("require_framewise_displacement", True))
            if modality == "fmri"
            else False
        )
        sub_frame = self._apply_fd_censoring(
            sub_frame,
            require_fd=fd_required,
            context_label=f"{self.dataset.ds_id}:{sub_id}:{ses_id or '-'}:{raw_task or '-'}:{run_id or '-'}",
        )
        n_after_qc = int(len(sub_frame))

        # Build target directory: use condition/task for datasets without sessions (like ds003171)
        dir_suffix = self._build_dir_suffix(ses_id, condition, task, run_id, acq_id)
        target_dir = self.mnps_dir / (f"{sub_id}_{dir_suffix}" if dir_suffix else sub_id)
        target_dir.mkdir(parents=True, exist_ok=True)

        # Build descriptive dataset_label with condition/task
        dataset_label = build_dataset_label(
            ds_id=self.dataset.ds_id,
            sub_id=sub_id,
            ses_id=ses_id,
            condition=condition,
            task=task,
            run=run_id,
            acq=acq_id,
        )
        _cfg_dt = mnps_cfg["window_sec"] * (1.0 - mnps_cfg["overlap"])
        if "t_start" in sub_frame.columns and "t_end" in sub_frame.columns and len(sub_frame) > 1:
            _t_s_raw = pd.to_numeric(sub_frame["t_start"], errors="coerce")
            _measured_dt = float(_t_s_raw.diff().dropna().median())
            if np.isfinite(_measured_dt) and _measured_dt > 0:
                dt = _measured_dt
                if abs(dt - _cfg_dt) > 0.1:
                    logger.info(
                        "Epoch step %.3f s (from t_start) differs from mnps config formula"
                        " %.3f s (window_sec=%.1f, overlap=%.4f). "
                        "Using measured step for time axis and Jacobian dt.",
                        dt, _cfg_dt, mnps_cfg["window_sec"], mnps_cfg["overlap"],
                    )
            else:
                dt = _cfg_dt
        else:
            dt = _cfg_dt
        coverage_seconds_measured, coverage_method = self._estimate_coverage_seconds(sub_frame, dt)
        coverage_seconds_assumed = float(len(sub_frame) * dt)
        coverage_seconds_effective = (
            coverage_seconds_measured
            if np.isfinite(coverage_seconds_measured) and coverage_seconds_measured > 0
            else coverage_seconds_assumed
        )
        coverage_policy = self.dataset.resolve_coverage_policy(
            condition=condition,
            task=task,
            run_id=run_id,
            acq_id=acq_id,
        )
        min_epochs_eff = int(coverage_policy.get("min_epochs", self.dataset.min_epochs))
        min_seconds_eff = float(coverage_policy.get("min_seconds", self.dataset.min_seconds))
        coverage_tag = str(coverage_policy.get("tag", "default"))
        if len(sub_frame) < min_epochs_eff or coverage_seconds_effective < min_seconds_eff:
            logger.warning(
                "Skipping %s (coverage too low; tag=%s): epochs=%d, seconds_effective=%.1f (%s), seconds_assumed=%.1f, required_epochs=%d, required_seconds=%.1f",
                dataset_label,
                coverage_tag,
                len(sub_frame),
                coverage_seconds_effective,
                coverage_method,
                coverage_seconds_assumed,
                min_epochs_eff,
                min_seconds_eff,
            )
            return

        # Track provisional modularity fraction if available
        modularity_provisional_frac = None
        if "fmri_modularity_provisional" in sub_frame.columns and len(sub_frame):
            modularity_provisional_frac = float(sub_frame["fmri_modularity_provisional"].fillna(0).mean())

        # Stratified MNPS v2 config
        v2_cfg = config.get("mnps_9d", {}) if isinstance(config, Mapping) else {}
        v2_enabled, v2_definition_version, selected_v2_cfg, subcoords_spec = _resolve_mnps_9d_runtime_config(
            v2_cfg if isinstance(v2_cfg, Mapping) else {},
            self.dataset.ds_id,
        )
        _validate_e_e_subcoord_construct(subcoords_spec if isinstance(subcoords_spec, Mapping) else {})
        entropy_meta = _resolve_entropy_provenance(sub_frame)
        features_cfg = config.get("features", {}) if isinstance(config, Mapping) else {}
        pe_cfg = features_cfg.get("permutation_entropy", {}) if isinstance(features_cfg, Mapping) else {}
        pe_params = {
            "order": int(pe_cfg.get("order", 5)) if isinstance(pe_cfg, Mapping) else 5,
            "delay": int(pe_cfg.get("delay", 1)) if isinstance(pe_cfg, Mapping) else 1,
            "normalize": bool(pe_cfg.get("normalize", True)) if isinstance(pe_cfg, Mapping) else True,
        }
        v2_metric_policies = (
            selected_v2_cfg.get("metric_policies", {})
            if isinstance(selected_v2_cfg, Mapping)
            else {}
        )
        e_e_policy = v2_metric_policies.get("e_e", {}) if isinstance(v2_metric_policies, Mapping) else {}
        e_e_intended_metric = (
            str(e_e_policy.get("preferred", "permutation_entropy"))
            if isinstance(e_e_policy, Mapping)
            else "permutation_entropy"
        )
        mnps_9d_constructs = {
            "e_e": {
                "intended_metric": e_e_intended_metric,
                "actual_metric_used": str(entropy_meta.get("metric", "permutation_entropy")),
                "metric_backend": str(entropy_meta.get("backend", "numpy")),
                "params": pe_params,
                "degraded_mode": bool(entropy_meta.get("degraded_mode", False)),
                "reason": entropy_meta.get("reason"),
            }
        }
        m3d_cfg = _resolve_mnps_3d_cfg(config if isinstance(config, Mapping) else {})
        mde_mode_requested = str(m3d_cfg.get("mode", "direct_features"))
        mde_mode_effective = mde_mode_requested
        mde_from_v2_reason = None
        v1_mapping_hash = None
        v1_mapping_normalized = None
        v1_mapping_matrix = None
        v1_mapping_matrix_rows = None
        mde_from_v2_aggregation = None
        mde_from_v2_aggregation_requested = None
        mde_from_v2_pooling_legacy = None

        direct_weighted_features = sorted(
            {
                str(feat)
                for axis_weights in (self.ctx.weights or {}).values()
                if isinstance(axis_weights, Mapping)
                for feat in axis_weights.keys()
            }
        )
        v2_weighted_features = sorted(
            {
                str(feat)
                for subcoord_weights in (subcoords_spec or {}).values()
                if isinstance(subcoord_weights, Mapping)
                for feat in subcoord_weights.keys()
            }
        )
        missing_rate_direct = _missing_weighted_feature_rate(sub_frame, direct_weighted_features)
        missing_rate_v2 = _missing_weighted_feature_rate(sub_frame, v2_weighted_features)

        proj_cfg = config.get("mnps_projection", {}) if isinstance(config, Mapping) else {}
        clip_threshold = float(proj_cfg.get("clip_threshold", 6.0)) if isinstance(proj_cfg, Mapping) else 6.0
        feature_standardization = proj_cfg.get("feature_standardization", {}) if isinstance(proj_cfg, Mapping) else {}
        anchor_cfg = proj_cfg.get("anchor", {}) if isinstance(proj_cfg, Mapping) else {}
        anchor_artifact: Optional[dict[str, Any]] = None
        external_anchor: Optional[dict[str, Any]] = None
        anchor_enabled = bool(anchor_cfg.get("enabled", False)) if isinstance(anchor_cfg, Mapping) else False
        if anchor_enabled:
            anchor_path = _resolve_anchor_path(anchor_cfg.get("path"), config=config) if isinstance(anchor_cfg, Mapping) else None
            if anchor_path is None or not anchor_path.exists():
                raise FileNotFoundError(f"mnps_projection.anchor.enabled=true but anchor path is missing: {anchor_path}")
            anchor_artifact = anchors.load_anchor_file(anchor_path)
            external_anchor = anchors.anchor_mapping(
                anchor_artifact,
                scale_method=str(anchor_cfg.get("scale_method", "iqr")) if isinstance(anchor_cfg, Mapping) else "iqr",
                min_subjects=int(anchor_cfg.get("min_subjects", 1)) if isinstance(anchor_cfg, Mapping) else 1,
            )
            if not external_anchor:
                raise ValueError(f"Feature anchor file produced no usable anchors: {anchor_path}")
        export_contracts = _resolve_export_contract_preferences(
            proj_cfg if isinstance(proj_cfg, Mapping) else {},
            external_anchor_available=bool(external_anchor),
        )
        export_subject_anchored = bool(export_contracts.get("subject_anchored", True))
        export_cohort_anchored = bool(export_contracts.get("cohort_anchored", False))
        primary_coordinate_layer = str(export_contracts.get("primary_coordinate_contract", "subject_anchored"))

        # Project direct MNPS coordinates first (always available as fallback/provenance).
        x_direct, x_direct_coverage, feature_baselines_v1 = projection.project_features_with_coverage(
            sub_frame, 
            self.ctx.weights, 
            normalize=normalize_mode,
            feature_standardization=feature_standardization,
            clip_threshold=clip_threshold
        )
        x = x_direct
        x_coverage = x_direct_coverage
        x_definition = "direct_mde_v1"
        direct_axis_renorm = "abs_weight"
        coords_9d = None
        coords_9d_names: list[str] = []
        feature_baselines_v2 = {}
        v2_missing_policy = "renorm"
        v2_all_non_finite_names: list[str] = []
        v2_all_non_finite_count = 0
        v2_duplicate_pairs: dict[str, str] = {}
        v2_duplicate_count = 0
        v2_duplicate_constant_pairs: dict[str, str] = {}
        v2_duplicate_constant_count = 0
        if v2_enabled and subcoords_spec:
            v2_missing_policy = (
                str(selected_v2_cfg.get("missing_policy", "renorm")).strip().lower()
                if isinstance(selected_v2_cfg, Mapping)
                else "renorm"
            )
            coords_9d, coords_9d_names, feature_baselines_v2 = projection.project_features_v2(
                sub_frame, 
                subcoords_spec, 
                normalize=normalize_mode, 
                missing_policy=v2_missing_policy,
                feature_standardization=feature_standardization,
                clip_threshold=clip_threshold
            )
            if coords_9d.size and coords_9d_names:
                try:
                    (
                        coords_9d,
                        coords_9d_names,
                        coords_9d_diag,
                    ) = schema._normalize_coords_9d(
                        coords_9d,
                        coords_9d_names,
                        allow_all_non_finite_columns=True,
                        allow_duplicate_columns=True,
                        allow_duplicate_constant_columns=True,
                        return_diagnostics=True,
                    )
                    v2_all_non_finite_names = list(coords_9d_diag.get("all_non_finite_names", []) or [])
                    v2_all_non_finite_count = int(coords_9d_diag.get("all_non_finite_count", 0) or 0)
                    v2_duplicate_pairs = dict(coords_9d_diag.get("duplicate_pairs", {}) or {})
                    v2_duplicate_count = int(coords_9d_diag.get("duplicate_count", 0) or 0)
                    v2_duplicate_constant_pairs = dict(coords_9d_diag.get("duplicate_constant_pairs", {}) or {})
                    v2_duplicate_constant_count = int(coords_9d_diag.get("duplicate_constant_count", 0) or 0)
                    if v2_all_non_finite_count > 0:
                        logger.warning(
                            "Stratified MNPS coords_9d has %d all-non-finite subcoordinate(s) for %s: %s. "
                            "Proceeding in degraded mode and flagging provenance.",
                            v2_all_non_finite_count,
                            dataset_label,
                            ", ".join(v2_all_non_finite_names),
                        )
                    if v2_duplicate_count > 0:
                        dup_desc = ", ".join(f"{dst}->{src}" for dst, src in sorted(v2_duplicate_pairs.items()))
                        logger.warning(
                            "Stratified MNPS coords_9d has %d exact duplicate subcoordinate(s) for %s: %s. "
                            "Proceeding in degraded mode and flagging provenance.",
                            v2_duplicate_count,
                            dataset_label,
                            dup_desc,
                        )
                    if v2_duplicate_constant_count > 0:
                        dup_desc = ", ".join(f"{dst}->{src}" for dst, src in sorted(v2_duplicate_constant_pairs.items()))
                        logger.warning(
                            "Stratified MNPS coords_9d has %d duplicate constant subcoordinate(s) for %s: %s. "
                            "Proceeding in degraded mode and flagging provenance.",
                            v2_duplicate_constant_count,
                            dataset_label,
                            dup_desc,
                        )
                except Exception as e:
                    logger.error("Failed to normalize Stratified MNPS coords_9d for %s. Explicit failure enforced to prevent silent degradation.", dataset_label)
                    raise RuntimeError(f"Stratified MNPS normalization failed: {e}") from e

        # Merge baselines (v2 preferred if active, else v1)
        merged_baselines = dict(feature_baselines_v1)
        merged_baselines.update(feature_baselines_v2)

        if mde_mode_requested == "from_v2":
            if coords_9d is not None and coords_9d.size and coords_9d_names:
                mde_from_v2_aggregation_requested = m3d_cfg.get("aggregation_requested")
                mde_from_v2_pooling_legacy = m3d_cfg.get("legacy_pooling")
                aggregation_effective = str(m3d_cfg.get("aggregation", "fixed_weighted_projection"))
                if aggregation_effective == "fixed_weighted_projection":
                    cfg_weights = _coerce_v1_mapping_to_v2_subcoords(
                        m3d_cfg.get("v1_mapping", {}),
                        subcoords_spec if isinstance(subcoords_spec, Mapping) else {},
                    )
                    has_any_weight = any(
                        isinstance(cfg_weights, Mapping)
                        and isinstance(cfg_weights.get(axis, {}), Mapping)
                        and len(cfg_weights.get(axis, {})) > 0
                        for axis in ("m", "d", "e")
                    )
                    if not has_any_weight:
                        raise ValueError(
                            "mnps_3d.mode=from_v2 with aggregation=fixed_weighted_projection requires a non-empty V1 mapping "
                            "(direct subcoord names or feature names resolvable via mnps_9d.subcoords)."
                        )
                    axis_map_for_v1 = cfg_weights
                else:
                    axis_map_for_v1 = m3d_cfg.get("map", {})
                x, x_coverage, v1_map_info = projection.derive_mde_from_v2(
                    coords_9d,
                    coords_9d_names,
                    axis_map_for_v1,
                    pooling=str(m3d_cfg.get("legacy_pooling", "mean")),
                    normalize_columns_l2=True,
                    enforce_block_selective=False,
                    return_mapping_info=True,
                )
                x_definition = f"derived_mde_from_v2_{str((v1_map_info or {}).get('aggregation', aggregation_effective))}"
                mde_from_v2_aggregation = (v1_map_info or {}).get("aggregation")
                direct_axis_renorm = str(mde_from_v2_aggregation or aggregation_effective)
                v1_mapping_normalized = (v1_map_info or {}).get("weights_normalized")
                v1_mapping_matrix = (v1_map_info or {}).get("matrix")
                v1_mapping_matrix_rows = (v1_map_info or {}).get("coords_9d_names")
                if isinstance(v1_mapping_normalized, Mapping):
                    v1_mapping_hash = _stable_hash_mapping(v1_mapping_normalized)
            else:
                mde_mode_effective = "direct_features"
                mde_from_v2_reason = "coords_9d_unavailable"
                logger.warning(
                    "mnps_3d.mode=from_v2 requested for %s but coords_9d unavailable; falling back to direct_features",
                    dataset_label,
                )

        x_subject_anchored = np.asarray(x, dtype=np.float32).copy()
        x_subject_coverage = np.asarray(x_coverage, dtype=np.float32).copy()
        coords_9d_subject_anchored = (
            np.asarray(coords_9d, dtype=np.float32).copy()
            if coords_9d is not None and coords_9d_names
            else None
        )
        coords_9d_subject_names = list(coords_9d_names) if coords_9d_names else []
        x_cohort_anchored = None
        x_cohort_coverage = None
        coords_9d_cohort_anchored = None
        coords_9d_cohort_names: list[str] = []
        feature_baselines_anchor: dict[str, dict] = {}

        if external_anchor and export_cohort_anchored:
            x_direct_anchor, x_direct_anchor_coverage, feature_baselines_anchor_v1 = projection.project_features_with_coverage(
                sub_frame,
                self.ctx.weights,
                normalize=normalize_mode,
                feature_standardization=feature_standardization,
                clip_threshold=clip_threshold,
                external_anchor=external_anchor,
            )
            x_anchor = x_direct_anchor
            x_anchor_coverage = x_direct_anchor_coverage
            if v2_enabled and subcoords_spec:
                coords_9d_anchor, coords_9d_anchor_names, feature_baselines_anchor_v2 = projection.project_features_v2(
                    sub_frame,
                    subcoords_spec,
                    normalize=normalize_mode,
                    missing_policy=v2_missing_policy,
                    feature_standardization=feature_standardization,
                    clip_threshold=clip_threshold,
                    external_anchor=external_anchor,
                )
                feature_baselines_anchor.update(feature_baselines_anchor_v2)
                if coords_9d_anchor.size and coords_9d_anchor_names:
                    coords_9d_anchor, coords_9d_anchor_names, _ = schema._normalize_coords_9d(
                        coords_9d_anchor,
                        coords_9d_anchor_names,
                        allow_all_non_finite_columns=True,
                        allow_duplicate_columns=True,
                        allow_duplicate_constant_columns=True,
                        return_diagnostics=True,
                    )
                    if mde_mode_requested == "from_v2":
                        aggregation_effective = str(m3d_cfg.get("aggregation", "fixed_weighted_projection"))
                        if aggregation_effective == "fixed_weighted_projection":
                            axis_map_anchor = _coerce_v1_mapping_to_v2_subcoords(
                                m3d_cfg.get("v1_mapping", {}),
                                subcoords_spec if isinstance(subcoords_spec, Mapping) else {},
                            )
                        else:
                            axis_map_anchor = m3d_cfg.get("map", {})
                        x_anchor, x_anchor_coverage = projection.derive_mde_from_v2(
                            coords_9d_anchor,
                            coords_9d_anchor_names,
                            axis_map_anchor,
                            pooling=str(m3d_cfg.get("legacy_pooling", "mean")),
                            normalize_columns_l2=True,
                            enforce_block_selective=False,
                        )
                    coords_9d_cohort_anchored = np.asarray(coords_9d_anchor, dtype=np.float32)
                    coords_9d_cohort_names = list(coords_9d_anchor_names)
            feature_baselines_anchor.update(feature_baselines_anchor_v1)
            x_cohort_anchored = np.asarray(x_anchor, dtype=np.float32)
            x_cohort_coverage = np.asarray(x_anchor_coverage, dtype=np.float32)
            merged_baselines.update({f"{k}__cohort_anchor": v for k, v in feature_baselines_anchor.items()})
            if primary_coordinate_layer == "cohort_anchored":
                x = x_cohort_anchored
                x_coverage = x_cohort_coverage
                x_definition = f"{x_definition}_cohort_anchored"
                if coords_9d_cohort_anchored is not None and coords_9d_cohort_names:
                    coords_9d = coords_9d_cohort_anchored
                    coords_9d_names = coords_9d_cohort_names

        axis_cov_labels = ["m", "d", "e"]
        axis_cov_stats = {
            f"{lbl}_mean": (
                float(np.nanmean(x_coverage[:, i])) if x_coverage.size and not np.all(np.isnan(x_coverage[:, i])) else float("nan")
            )
            for i, lbl in enumerate(axis_cov_labels)
        }

        def _apply_shared_epoch_mask(mask: np.ndarray) -> None:
            """Apply a shared time-grid mask across aligned coordinate surfaces."""
            nonlocal x, x_coverage, x_subject_anchored, x_subject_coverage
            nonlocal x_cohort_anchored, x_cohort_coverage, sub_frame
            nonlocal coords_9d, coords_9d_subject_anchored, coords_9d_cohort_anchored

            x = x[mask]
            x_coverage = x_coverage[mask]
            if len(x_subject_anchored) == len(mask):
                x_subject_anchored = x_subject_anchored[mask]
                x_subject_coverage = x_subject_coverage[mask]
            if x_cohort_anchored is not None and len(x_cohort_anchored) == len(mask):
                x_cohort_anchored = x_cohort_anchored[mask]
            if x_cohort_coverage is not None and len(x_cohort_coverage) == len(mask):
                x_cohort_coverage = x_cohort_coverage[mask]
            sub_frame = sub_frame.loc[mask].reset_index(drop=True)
            if coords_9d is not None and len(coords_9d) == len(mask):
                coords_9d = coords_9d[mask]
            if coords_9d_subject_anchored is not None and len(coords_9d_subject_anchored) == len(mask):
                coords_9d_subject_anchored = coords_9d_subject_anchored[mask]
            if coords_9d_cohort_anchored is not None and len(coords_9d_cohort_anchored) == len(mask):
                coords_9d_cohort_anchored = coords_9d_cohort_anchored[mask]

        dropped_missing_axis_epochs = 0
        dropped_geometry_invalid_epochs = 0
        min_axis_coverage = float(proj_cfg.get("min_axis_coverage", 0.3)) if isinstance(proj_cfg, Mapping) else 0.3
        if missing_axis_policy == "nan_mask_v1":
            valid_x = np.all(np.isfinite(x), axis=1)
            cov_ok = np.all(x_coverage >= min_axis_coverage, axis=1)
            mask = valid_x & cov_ok
            dropped_missing_axis_epochs = int((~mask).sum())
            if dropped_missing_axis_epochs > 0:
                logger.warning(
                    "Dropping %d epochs with missing direct-axis support/coverage for %s (policy=%s, min_axis_coverage=%.3f)",
                    dropped_missing_axis_epochs,
                    dataset_label,
                    missing_axis_policy,
                    min_axis_coverage,
                )
                _apply_shared_epoch_mask(mask)
            coverage_seconds_measured_post, coverage_method_post = self._estimate_coverage_seconds(sub_frame, dt)
            coverage_seconds_assumed_post = float(len(sub_frame) * dt)
            coverage_seconds_effective_post = (
                coverage_seconds_measured_post
                if np.isfinite(coverage_seconds_measured_post) and coverage_seconds_measured_post > 0
                else coverage_seconds_assumed_post
            )
            if len(sub_frame) < min_epochs_eff or coverage_seconds_effective_post < min_seconds_eff:
                logger.warning(
                    "Skipping %s after nan/cov masking (tag=%s): epochs=%d, seconds_effective=%.1f (%s), required_epochs=%d, required_seconds=%.1f",
                    dataset_label,
                    coverage_tag,
                    len(sub_frame),
                    coverage_seconds_effective_post,
                    coverage_method_post,
                    min_epochs_eff,
                    min_seconds_eff,
                )
                return
            if len(sub_frame) == 0:
                logger.warning("Skipping %s: all epochs dropped by missing-axis policy", dataset_label)
                return

        geometry_contract = compute_standard_geometry_contract(
            x=x,
            coords_9d=coords_9d,
            coords_9d_names=coords_9d_names,
            primary_requires_coords_9d=bool(str(mde_mode_effective).strip().lower() == "from_v2"),
        )
        if v2_duplicate_count > 0:
            coords_contract = geometry_contract.setdefault("coords_9d", {})
            if isinstance(coords_contract, Mapping):
                coords_contract = dict(coords_contract)
                geometry_contract["coords_9d"] = coords_contract
            coords_contract["duplicate_pairs"] = dict(v2_duplicate_pairs)
            coords_contract["duplicate_count"] = int(v2_duplicate_count)
            coords_contract["duplicate_constant_pairs"] = dict(v2_duplicate_constant_pairs)
            coords_contract["duplicate_constant_count"] = int(v2_duplicate_constant_count)
            geometry_contract["status"] = "adjusted"
        geometry_keep_mask = np.asarray(geometry_contract.pop("_row_keep_mask", np.ones((len(sub_frame),), dtype=bool)), dtype=bool)
        dropped_geometry_invalid_epochs = int((~geometry_keep_mask).sum())
        if dropped_geometry_invalid_epochs > 0:
            logger.warning(
                "Dropping %d mathematically invalid geometry epochs for %s (policy=%s)",
                dropped_geometry_invalid_epochs,
                dataset_label,
                STANDARD_GEOMETRY_POLICY_VERSION,
            )
            _apply_shared_epoch_mask(geometry_keep_mask)
            geometry_contract["shared_time_grid"]["epochs_retained"] = int(len(sub_frame))
            geometry_contract["shared_time_grid"]["epochs_dropped"] = int(dropped_geometry_invalid_epochs)
            geometry_contract["shared_time_grid"]["drop_fraction"] = (
                float(dropped_geometry_invalid_epochs / max(1, int(geometry_contract["shared_time_grid"].get("epochs_before_policy", 0))))
            )
            coverage_seconds_measured_post, coverage_method_post = self._estimate_coverage_seconds(sub_frame, dt)
            coverage_seconds_assumed_post = float(len(sub_frame) * dt)
            coverage_seconds_effective_post = (
                coverage_seconds_measured_post
                if np.isfinite(coverage_seconds_measured_post) and coverage_seconds_measured_post > 0
                else coverage_seconds_assumed_post
            )
            if len(sub_frame) < min_epochs_eff or coverage_seconds_effective_post < min_seconds_eff:
                logger.warning(
                    "Skipping %s after standard geometry invalidity policy (tag=%s): epochs=%d, seconds_effective=%.1f (%s), required_epochs=%d, required_seconds=%.1f",
                    dataset_label,
                    coverage_tag,
                    len(sub_frame),
                    coverage_seconds_effective_post,
                    coverage_method_post,
                    min_epochs_eff,
                    min_seconds_eff,
                )
                return
            if len(sub_frame) == 0:
                logger.warning("Skipping %s: all epochs dropped by standard geometry invalidity policy", dataset_label)
                return

        # Time index and derivatives.
        # Prefer feature-epoch midpoints (t_start + t_end) / 2 when available so
        # /time is the true window-center regardless of the mnps.window_sec config.
        if "t_start" in sub_frame.columns and "t_end" in sub_frame.columns:
            _t_s = pd.to_numeric(sub_frame["t_start"], errors="coerce")
            _t_e = pd.to_numeric(sub_frame["t_end"], errors="coerce")
            time = ((_t_s + _t_e) / 2.0).to_numpy(dtype=np.float64)
        else:
            time = projection.build_time_index(len(sub_frame), mnps_cfg["window_sec"], mnps_cfg["overlap"])
        window_start, window_end = self._extract_time_bounds(sub_frame, time, mnps_cfg["window_sec"])
        geometry_contract["time_grid"] = compute_window_time_audit(
            time=time,
            window_start=window_start,
            window_end=window_end,
            dt_sec_runtime=float(dt),
            dt_sec_config=float(_cfg_dt),
            window_sec_config=float(mnps_cfg["window_sec"]),
        )
        time_reference_result = build_time_reference_for_run(
            config=config if isinstance(config, Mapping) else {},
            dataset_id=self.dataset.ds_id,
            dataset_root=self._dataset_root(),
            index_df=self.index_df,
            lookup_rel_paths_by_file_value=self.dataset._lookup_rel_paths_by_file_value,
            sub_id=sub_id,
            run_id=run_id,
            acq_id=acq_id,
            representative_file=representative_file,
            sub_frame=sub_frame,
            window_start=window_start,
            window_end=window_end,
        )
        
        # Explicitly prevent derivative estimation across file boundaries (time aliasing protection)
        def _compute_dot(features_array: np.ndarray) -> np.ndarray:
            """Internal helper: compute dot."""
            dot_cfg = ((config.get("mnps", {}) or {}).get("derivative_robust", {}) or {}) if isinstance(config, Mapping) else {}
            use_segmented = bool(dot_cfg.get("enabled", True))
            dot_array = np.zeros_like(features_array)
            if "file" in sub_frame.columns and sub_frame["file"].nunique() > 1:
                logger.info("Computing derivatives per-file to avoid boundary crossing (%d files)", sub_frame["file"].nunique())
                file_series = sub_frame["file"].to_numpy()
                for f_val in np.unique(file_series):
                    mask = (file_series == f_val)
                    sub_array = features_array[mask]
                    if len(sub_array) > 0:
                        if use_segmented:
                            dot_array[mask] = projection.estimate_derivatives_segmented(
                                sub_array,
                                dt,
                                method=self.ctx.derivative_cfg["method"],
                                max_jump=float(dot_cfg.get("max_jump", 5.0)),
                                min_seg=int(dot_cfg.get("min_seg", 9)),
                                savgol_window=int(self.ctx.derivative_cfg["window"]),
                                polyorder=int(self.ctx.derivative_cfg["polyorder"]),
                            )
                        else:
                            dot_array[mask] = projection.estimate_derivatives(
                                sub_array,
                                dt,
                                method=self.ctx.derivative_cfg["method"],
                                window=self.ctx.derivative_cfg["window"],
                                polyorder=self.ctx.derivative_cfg["polyorder"],
                            )
            else:
                if use_segmented:
                    dot_array = projection.estimate_derivatives_segmented(
                        features_array,
                        dt,
                        method=self.ctx.derivative_cfg["method"],
                        max_jump=float(dot_cfg.get("max_jump", 5.0)),
                        min_seg=int(dot_cfg.get("min_seg", 9)),
                        savgol_window=int(self.ctx.derivative_cfg["window"]),
                        polyorder=int(self.ctx.derivative_cfg["polyorder"]),
                    )
                else:
                    dot_array = projection.estimate_derivatives(
                        features_array,
                        dt,
                        method=self.ctx.derivative_cfg["method"],
                        window=self.ctx.derivative_cfg["window"],
                        polyorder=self.ctx.derivative_cfg["polyorder"],
                    )
            return dot_array

        x_dot = _compute_dot(x)

        # KNN and optional primary Jacobian
        whiten_flag = bool(mnps_cfg.get("whiten", True))
        nn_indices = projection.build_knn_indices(
            x,
            k=mnps_cfg["knn_k"],
            metric=mnps_cfg["knn_metric"],
            whiten=whiten_flag,
        )
        primary_jac_cfg = (
            ((config.get("mnps", {}) or {}).get("jacobian", {}) or {})
            if isinstance(config, Mapping)
            else {}
        )
        primary_jac_enabled = bool(primary_jac_cfg.get("enabled", True))
        jac_res = None
        if primary_jac_enabled:
            jac_res = jacobian.estimate_local_jacobians(
                x,
                x_dot,
                nn_indices,
                super_window=mnps_cfg["super_window"],
                ridge_alpha=mnps_cfg["ridge_alpha"],
                distance_weighted=bool(config.get("mnps", {}).get("ridge", {}).get("distance_weighted", True)),
                j_dot_dt=float(dt),
            )
            jac_res, primary_geometry_jacobian = apply_standard_jacobian_window_policy(
                jac_res,
                condition_number_max=STANDARD_GEOMETRY_MAX_JACOBIAN_CONDITION,
            )
        else:
            primary_geometry_jacobian = {
                "policy_version": STANDARD_GEOMETRY_POLICY_VERSION,
                "status": "not_available",
                "windows_raw": 0,
                "windows_retained": 0,
                "invalid_windows": 0,
                "condition_number_threshold": float(STANDARD_GEOMETRY_MAX_JACOBIAN_CONDITION),
                "invalid_reason_counts": {},
            }

        # Optional Jacobian for v2 coordinates
        jac_res_v2 = None
        coords_9d_geometry_jacobian = {
            "policy_version": STANDARD_GEOMETRY_POLICY_VERSION,
            "status": "not_available",
            "windows_raw": 0,
            "windows_retained": 0,
            "invalid_windows": 0,
            "condition_number_threshold": float(STANDARD_GEOMETRY_MAX_JACOBIAN_CONDITION),
            "invalid_reason_counts": {},
        }
        if v2_enabled and coords_9d is not None and coords_9d_names:
            v2_jac_cfg = v2_cfg.get("jacobian", {}) if isinstance(v2_cfg, Mapping) else {}
            v2_jac_enabled = bool(v2_jac_cfg.get("enabled", True))
            if v2_jac_enabled and coords_9d.size:
                if np.isfinite(coords_9d).all():
                    coords_9d_dot = _compute_dot(coords_9d)
                    nn_indices_v2 = projection.build_knn_indices(
                        coords_9d,
                        k=mnps_cfg["knn_k"],
                        metric=mnps_cfg["knn_metric"],
                        whiten=whiten_flag,
                    )
                    jac_res_v2 = jacobian.estimate_local_jacobians(
                        coords_9d,
                        coords_9d_dot,
                        nn_indices_v2,
                        super_window=mnps_cfg["super_window"],
                        ridge_alpha=mnps_cfg["ridge_alpha"],
                        distance_weighted=bool(config.get("mnps", {}).get("ridge", {}).get("distance_weighted", True)),
                        j_dot_dt=float(dt),
                    )
                    jac_res_v2, coords_9d_geometry_jacobian = apply_standard_jacobian_window_policy(
                        jac_res_v2,
                        condition_number_max=STANDARD_GEOMETRY_MAX_JACOBIAN_CONDITION,
                    )
                else:
                    logger.warning(
                        "Skipping v2 Jacobian for %s due to non-finite coords_9d rows "
                        "(degraded v2 coverage).",
                        dataset_label,
                    )
        geometry_contract["jacobian"] = primary_geometry_jacobian
        geometry_contract["jacobian_9d"] = coords_9d_geometry_jacobian
        geometry_contract["status"] = (
            "adjusted"
            if (
                int((geometry_contract.get("shared_time_grid") or {}).get("epochs_dropped", 0)) > 0
                or bool((geometry_contract.get("mnps_3d") or {}).get("degenerate_axes"))
                or bool((geometry_contract.get("coords_9d") or {}).get("degenerate_axes"))
                or int((geometry_contract.get("coords_9d") or {}).get("nonfinite_rows_retained_on_shared_grid", 0)) > 0
                or str((geometry_contract.get("time_grid") or {}).get("status", "ok")).strip().lower() != "ok"
                or int((primary_geometry_jacobian or {}).get("invalid_windows", 0)) > 0
                or int((coords_9d_geometry_jacobian or {}).get("invalid_windows", 0)) > 0
            )
            else "ok"
        )

        # Contract invariants: provenance hashes for "inputs used to compute outputs"
        # must match the representation saved to H5.
        x_saved = np.asarray(x, dtype=np.float32)
        x_hash_saved = _stable_hash_array(x_saved)
        x_hash_knn_input = _stable_hash_array(np.asarray(x, dtype=np.float32))
        x_hash_jac_input = _stable_hash_array(np.asarray(x, dtype=np.float32))
        if not (x_hash_saved == x_hash_knn_input == x_hash_jac_input):
            raise RuntimeError("Direct x contract violation: saved x differs from kNN/Jacobian input.")

        coords_9d_hash_saved = None
        coords_9d_hash_knn_input = None
        coords_9d_hash_jac_input = None
        if coords_9d is not None and coords_9d_names and coords_9d.size:
            coords_9d_saved = np.asarray(coords_9d, dtype=np.float32)
            coords_9d_hash_saved = _stable_hash_array(coords_9d_saved)
            if jac_res_v2 is not None:
                coords_9d_hash_knn_input = _stable_hash_array(np.asarray(coords_9d, dtype=np.float32))
                coords_9d_hash_jac_input = _stable_hash_array(np.asarray(coords_9d, dtype=np.float32))
                if not (
                    coords_9d_hash_saved == coords_9d_hash_knn_input == coords_9d_hash_jac_input
                ):
                    raise RuntimeError("v2 contract violation: saved coords_9d differs from v2 kNN/Jacobian input.")
        nn_indices_hash_saved = _stable_hash_array(np.asarray(nn_indices, dtype=np.int32)) if nn_indices is not None else None
        jacobian_hash_saved = _stable_hash_array(np.asarray(jac_res.j_hat, dtype=np.float32)) if jac_res is not None else None
        jacobian_dot_hash_saved = _stable_hash_array(np.asarray(jac_res.j_dot, dtype=np.float32)) if jac_res is not None else None
        jacobian_9d_hash_saved = (
            _stable_hash_array(np.asarray(jac_res_v2.j_hat, dtype=np.float32)) if jac_res_v2 is not None else None
        )
        jacobian_9d_dot_hash_saved = (
            _stable_hash_array(np.asarray(jac_res_v2.j_dot, dtype=np.float32)) if jac_res_v2 is not None else None
        )
        reproducibility_policy = resolve_reproducibility_policy(config, dataset_id=self.dataset.ds_id)

        # Optional Stratified (v2) block-Jacobian summaries and cross-partials
        stratified_blocks_result = None
        if jac_res_v2 is not None and coords_9d_names and getattr(jac_res_v2, "j_hat", None) is not None:
            try:
                stratified_blocks_result = compute_stratified_blocks_and_cross_partials(
                    ds_id=self.dataset.ds_id,
                    dataset_label=dataset_label,
                    subject=sub_id,
                    session=ses_id,
                    condition=condition,
                    task=task,
                    coords_9d_names=coords_9d_names,
                    jacobian_9D=jac_res_v2.j_hat,
                    config=config,
                )
            except Exception:
                logger.exception(
                    "Failed to compute stratified (v2) block summaries / cross-partials for %s",
                    dataset_label,
                )

        def _estimate_jacobian_layer(coords_arr: Optional[np.ndarray]) -> Any:
            """Estimate a Jacobian stack for an explicit anchor layer."""
            if coords_arr is None or not np.size(coords_arr):
                return None
            arr = np.asarray(coords_arr, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] == 0:
                return None
            if not np.isfinite(arr).all():
                return None
            dot_arr = _compute_dot(arr)
            nn_idx = projection.build_knn_indices(
                arr,
                k=mnps_cfg["knn_k"],
                metric=mnps_cfg["knn_metric"],
                whiten=whiten_flag,
            )
            layer_jac = jacobian.estimate_local_jacobians(
                arr,
                dot_arr,
                nn_idx,
                super_window=mnps_cfg["super_window"],
                ridge_alpha=mnps_cfg["ridge_alpha"],
                distance_weighted=bool(config.get("mnps", {}).get("ridge", {}).get("distance_weighted", True)),
                j_dot_dt=float(dt),
            )
            layer_jac, _ = apply_standard_jacobian_window_policy(
                layer_jac,
                condition_number_max=STANDARD_GEOMETRY_MAX_JACOBIAN_CONDITION,
            )
            return layer_jac

        anchor_spec = anchor_artifact.get("spec", {}) if isinstance(anchor_artifact, Mapping) else {}
        exported_anchor_spec = anchor_spec if export_cohort_anchored else {}
        jacobian_layers: Dict[str, Any] = {}
        if primary_jac_enabled:
            jac_res_subject = jac_res if primary_coordinate_layer == "subject_anchored" else _estimate_jacobian_layer(x_subject_anchored)
            jac_res_cohort = None
            if x_cohort_anchored is not None:
                jac_res_cohort = jac_res if primary_coordinate_layer == "cohort_anchored" else _estimate_jacobian_layer(x_cohort_anchored)
            if export_subject_anchored and jac_res_subject is not None:
                jacobian_layers["jacobian_subject_anchored"] = {
                    "J_hat": jac_res_subject.j_hat,
                    "J_dot": jac_res_subject.j_dot,
                    "centers": jac_res_subject.centers,
                    "attrs": {
                        "coordinate_contract": "subject_anchored",
                        "coordinate_space": "mnps_3d",
                    },
                }
            if export_cohort_anchored and jac_res_cohort is not None:
                jacobian_layers["jacobian_cohort_anchored"] = {
                    "J_hat": jac_res_cohort.j_hat,
                    "J_dot": jac_res_cohort.j_dot,
                    "centers": jac_res_cohort.centers,
                    "attrs": {
                        "coordinate_contract": "cohort_anchored",
                        "coordinate_space": "mnps_3d",
                        "anchor_id": exported_anchor_spec.get("anchor_id"),
                        "anchor_hash": exported_anchor_spec.get("anchor_hash"),
                        "anchor_source": exported_anchor_spec.get("anchor_source"),
                    },
                }
        if v2_enabled:
            jac_res_v2_subject = (
                jac_res_v2
                if primary_coordinate_layer == "subject_anchored"
                else _estimate_jacobian_layer(coords_9d_subject_anchored)
            ) if coords_9d_subject_anchored is not None and coords_9d_subject_names else None
            jac_res_v2_cohort = (
                jac_res_v2
                if primary_coordinate_layer == "cohort_anchored"
                else _estimate_jacobian_layer(coords_9d_cohort_anchored)
            ) if coords_9d_cohort_anchored is not None and coords_9d_cohort_names else None
            if export_subject_anchored and jac_res_v2_subject is not None:
                jacobian_layers["jacobian_9D_subject_anchored"] = {
                    "J_hat": jac_res_v2_subject.j_hat,
                    "J_dot": jac_res_v2_subject.j_dot,
                    "centers": jac_res_v2_subject.centers,
                    "attrs": {
                        "coordinate_contract": "subject_anchored",
                        "coordinate_space": "coords_9d",
                    },
                }
            if export_cohort_anchored and jac_res_v2_cohort is not None:
                jacobian_layers["jacobian_9D_cohort_anchored"] = {
                    "J_hat": jac_res_v2_cohort.j_hat,
                    "J_dot": jac_res_v2_cohort.j_dot,
                    "centers": jac_res_v2_cohort.centers,
                    "attrs": {
                        "coordinate_contract": "cohort_anchored",
                        "coordinate_space": "coords_9d",
                        "anchor_id": exported_anchor_spec.get("anchor_id"),
                        "anchor_hash": exported_anchor_spec.get("anchor_hash"),
                        "anchor_source": exported_anchor_spec.get("anchor_source"),
                    },
                }

        # Extract auxiliary arrays
        effective_stage_codebook = mnps_cfg["stage_codebook"]
        stage = extract_stage_array(sub_frame, mnps_cfg["stage_codebook"])
        stage_source: Optional[str] = None
        stage_column: Optional[str] = None
        stage_events_path: Optional[str] = None
        stage_proxy_info: Optional[Dict[str, Any]] = None
        if stage is None:
            try:
                stage, stage_source, stage_column, stage_events_path = self._infer_stage_from_bids_events(sub_frame)
            except Exception:
                logger.exception("Failed to infer stage labels from BIDS events for %s", dataset_label)
        else:
            stage_source = "features_csv"
            for c in ["stage", "stage_code", "sleep_stage", "labels_stage"]:
                if c in sub_frame.columns:
                    stage_column = c
                    break
            try:
                state_cfg = nwb_intervals.resolve_state_labels_config(config, self.dataset.ds_id)
                state_codebook = state_cfg.get("codebook", {}) if isinstance(state_cfg, Mapping) else {}
                if bool(state_cfg.get("enabled", False)) and "nwb_state" in sub_frame.columns and isinstance(state_codebook, Mapping):
                    effective_stage_codebook = {
                        str(k): int(v)
                        for k, v in state_codebook.items()
                    }
                    stage_source = "features_csv:nwb_intervals"
            except Exception:
                logger.exception("Failed resolving NWB state codebook for %s", dataset_label)
        within_run_labels = self._build_within_run_labels(
            config=config,
            sub_id=sub_id,
            ses_id=ses_id,
            task=task,
            raw_task=raw_task,
            run_id=run_id,
            acq_id=acq_id,
            tr_sec=self._resolve_run_tr_seconds(sub_frame),
            time=time,
            window_start=window_start,
            window_end=window_end,
            sub_frame=sub_frame,
        )
        if stage is None and within_run_labels.stage is not None:
            stage = within_run_labels.stage
            stage_source = within_run_labels.stage_source
            stage_column = within_run_labels.stage_column
            if within_run_labels.stage_codebook:
                effective_stage_codebook = within_run_labels.stage_codebook
        if stage is None:
            try:
                stage_proxy_candidate, stage_proxy_info = derive_pseudo_stage_array(
                    sub_frame,
                    config=config,
                    dataset_id=self.dataset.ds_id,
                    stage_codebook=effective_stage_codebook if isinstance(effective_stage_codebook, Mapping) else {},
                )
                if stage_proxy_candidate is not None:
                    stage = stage_proxy_candidate
                    stage_source = "pseudo_stage_proxy"
                    proxy_cols = []
                    if stage_proxy_info.get("spindle_column"):
                        proxy_cols.append(str(stage_proxy_info.get("spindle_column")))
                    if stage_proxy_info.get("sigma_column"):
                        proxy_cols.append(str(stage_proxy_info.get("sigma_column")))
                    if stage_proxy_info.get("emg_column"):
                        proxy_cols.append(str(stage_proxy_info.get("emg_column")))
                    if proxy_cols:
                        proxy_cols = list(dict.fromkeys(proxy_cols))
                    stage_column = ",".join(proxy_cols) if proxy_cols else "proxy"

                    label = stage_proxy_info.get("label")
                    code = stage_proxy_info.get("code")
                    if label is not None and code is not None:
                        try:
                            current_codebook = {
                                str(k): int(v)
                                for k, v in (effective_stage_codebook.items() if isinstance(effective_stage_codebook, Mapping) else [])
                            }
                        except Exception:
                            current_codebook = {}
                        current_codebook[str(label)] = int(code)
                        effective_stage_codebook = current_codebook
                elif stage_proxy_info and stage_proxy_info.get("status") not in {"ok", None}:
                    logger.info(
                        "Pseudo-stage skipped for %s (status=%s)",
                        dataset_label,
                        stage_proxy_info.get("status"),
                    )
            except Exception:
                logger.exception("Failed to derive pseudo-stage labels for %s", dataset_label)

        event_table_columns: Dict[str, Any] = {}
        event_provenance_events: Dict[str, np.ndarray] = {}
        stage_mapping_qc_entry: Optional[Dict[str, Any]] = None
        try:
            event_provenance = self._build_bids_event_stage_provenance(
                sub_frame=sub_frame,
                stage_for_windows=stage,
            )
            if event_provenance is not None:
                prefer_events_stage = self._prefer_events_stage_in_summary()
                if event_provenance.get("stage_inferred") is not None and (stage is None or prefer_events_stage):
                    stage = np.asarray(event_provenance.get("stage_inferred"), dtype=np.int8)
                    inferred_source = event_provenance.get("stage_source")
                    inferred_column = event_provenance.get("stage_column")
                    if inferred_source:
                        stage_source = str(inferred_source)
                        if prefer_events_stage:
                            stage_source = f"{stage_source}:override_features_stage"
                    if inferred_column:
                        stage_column = str(inferred_column)
                if stage_events_path is None and event_provenance.get("events_path"):
                    stage_events_path = str(event_provenance.get("events_path"))
                if stage_column is None and event_provenance.get("stage_column"):
                    stage_column = str(event_provenance.get("stage_column"))
                if stage_source is None and event_provenance.get("stage_source"):
                    stage_source = str(event_provenance.get("stage_source"))
                event_table_columns = dict(event_provenance.get("event_table_columns", {}) or {})
                event_provenance_events = dict(event_provenance.get("legacy_events", {}) or {})
                stage_mapping_qc_entry = dict(event_provenance.get("stage_mapping_qc", {}) or {})
        except Exception:
            logger.exception("Failed to build BIDS event provenance for %s", dataset_label)

        stage_frac_labeled = None
        if stage is not None and len(stage) > 0:
            try:
                stage_frac_labeled = float(np.mean(np.asarray(stage) != -1))
            except Exception:
                stage_frac_labeled = None
        z = extract_embodied_array(sub_frame, mnps_cfg["embodied"])
        events = dict(extract_events(sub_frame))
        if event_provenance_events:
            for key, arr in event_provenance_events.items():
                safe_key = str(key).replace("/", "_").replace("\\", "_")
                if safe_key in events:
                    continue
                try:
                    events[safe_key] = np.asarray(arr, dtype=np.float64)
                except Exception:
                    continue

        # Load regional fMRI if available (pass sub_frame for correct file matching)
        regions_bold, regions_names, regions_sfreq = self._load_regional_fmri(
            sub_id=sub_id,
            dataset_label=dataset_label,
            config=config,
            sub_frame=sub_frame,
            raw_task=raw_task,
            condition=condition,
            session=ses_id,
            run_id=run_id,
        )

        # Group regions and compute optional regional MNPS/MNJ context.
        regional_mnps_cfg = config.get("regional_mnps", {}) if isinstance(config, Mapping) else {}
        group_ts, group_matrix, group_names, region_groups, regional_mnps_results_subject = compute_regional_context(
            sub_frame=sub_frame,
            regions_bold=regions_bold,
            regions_names=regions_names,
            regions_sfreq=regions_sfreq,
            config=config,
            regional_mnps_cfg=regional_mnps_cfg if isinstance(regional_mnps_cfg, Mapping) else {},
            subcoords_spec=subcoords_spec if isinstance(subcoords_spec, Mapping) else {},
            axis_weights=self.ctx.weights if isinstance(self.ctx.weights, Mapping) else {},
            dataset_id=self.dataset.ds_id,
            dataset_label=dataset_label,
            proj_cfg=proj_cfg if isinstance(proj_cfg, Mapping) else {},
            normalize_mode=normalize_mode,
            external_anchor=None,
            subject=sub_id,
            session=ses_id,
            condition=condition,
            task=task,
            resolve_mnps_3d_cfg=_resolve_mnps_3d_cfg,
            coerce_v1_mapping_to_v2_subcoords=_coerce_v1_mapping_to_v2_subcoords,
            align_v2_subcoords=_align_v2_subcoords,
        )
        regional_mnps_results_cohort = None
        if external_anchor:
            _, _, _, _, regional_mnps_results_cohort = compute_regional_context(
                sub_frame=sub_frame,
                regions_bold=regions_bold,
                regions_names=regions_names,
                regions_sfreq=regions_sfreq,
                config=config,
                regional_mnps_cfg=regional_mnps_cfg if isinstance(regional_mnps_cfg, Mapping) else {},
                subcoords_spec=subcoords_spec if isinstance(subcoords_spec, Mapping) else {},
                axis_weights=self.ctx.weights if isinstance(self.ctx.weights, Mapping) else {},
                dataset_id=self.dataset.ds_id,
                dataset_label=dataset_label,
                proj_cfg=proj_cfg if isinstance(proj_cfg, Mapping) else {},
                normalize_mode=normalize_mode,
                external_anchor=external_anchor,
                subject=sub_id,
                session=ses_id,
                condition=condition,
                task=task,
                resolve_mnps_3d_cfg=_resolve_mnps_3d_cfg,
                coerce_v1_mapping_to_v2_subcoords=_coerce_v1_mapping_to_v2_subcoords,
                align_v2_subcoords=_align_v2_subcoords,
            )
        regional_mnps_results = (
            regional_mnps_results_cohort
            if primary_coordinate_layer == "cohort_anchored" and regional_mnps_results_cohort is not None
            else regional_mnps_results_subject
        )

        if regional_mnps_cfg.get("enabled", False) and regional_mnps_results is not None:
            # Persist regional MNPS and block-Jacobian summaries at the
            # dataset level so they can be consumed by analysis code
            # without re-estimating Jacobians.
            self.dataset.write_regional_csv_outputs_threadsafe(
                regional_mnps_results=regional_mnps_results,
                regional_mnps_cfg=regional_mnps_cfg if isinstance(regional_mnps_cfg, Mapping) else {},
                mnps_dir=self.mnps_dir,
                config=config,
                dataset_label=dataset_label,
            )

        tabular_exports_h5: Dict[str, Any] = {}
        if regional_mnps_results is not None:
            regional_rows = summary_to_dataframe_rows(regional_mnps_results)
            if regional_rows:
                tabular_exports_h5["regional_mnps_subjects"] = _rows_to_columnar_table(regional_rows)
            regional_block_rows = compute_block_jacobian_rows(
                regional_mnps_results,
                config,
                include_self=False,
            )
            if regional_block_rows:
                tabular_exports_h5["regional_block_jacobians_subjects"] = _rows_to_columnar_table(regional_block_rows)
        if stratified_blocks_result is not None and stratified_blocks_result.block_rows:
            tabular_exports_h5["stratified_block_jacobians_subjects"] = _rows_to_columnar_table(
                stratified_blocks_result.block_rows
            )

        # Optional: write Stratified (v2) block-Jacobian CSV into the MNPS run directory
        self.dataset.write_stratified_blocks_csv_output_threadsafe(
            stratified_blocks_result=stratified_blocks_result,
            config=config,
            dataset_id=self.dataset.ds_id,
            mnps_dir=self.mnps_dir,
            dataset_label=dataset_label,
        )

        # Compute extensions (E-Kappa, RFM, O-Koh, TIG)
        extensions_payload, extensions_summary = compute_extensions(
            dataset_label=dataset_label,
            extensions_cfg=self.ctx.extensions_cfg,
            x=x,
            sub_frame=sub_frame,
            time=time,
            dt=dt,
            coords_9d=coords_9d,
            coords_9d_names=coords_9d_names,
            regions_bold=regions_bold,
            regions_sfreq=regions_sfreq,
            group_ts=group_ts,
            group_matrix=group_matrix,
            group_names=group_names,
            region_groups=region_groups,
        )
        merged_extensions = dict(extensions_payload) if isinstance(extensions_payload, Mapping) else {}
        time_reference_extension = time_reference_result.get("extension")
        if isinstance(time_reference_extension, Mapping) and time_reference_extension:
            merged_extensions["time_reference"] = dict(time_reference_extension)
        if tabular_exports_h5:
            existing_tables = merged_extensions.get("tabular_exports")
            merged_tables = dict(existing_tables) if isinstance(existing_tables, Mapping) else {}
            merged_tables.update(tabular_exports_h5)
            merged_extensions["tabular_exports"] = merged_tables
        extensions_payload = merged_extensions

        # Ensemble and robustness summaries
        ensemble_summary = None
        if v2_enabled and coords_9d is not None and coords_9d_names:
            ensemble_summary = compute_ensemble_summary_for_subject(
                config=config,
                dataset_id=self.dataset.ds_id,
                sub_frame=sub_frame,
                coords_9d_names=coords_9d_names,
                subcoords_spec=subcoords_spec,
                normalize_mode=normalize_mode,
            )

        robust_summary = compute_robust_and_reliability_summaries(
            config=config,
            mnps_cfg=mnps_cfg,
            x=x,
            coords_9d=coords_9d,
            coords_9d_names=coords_9d_names,
        )

        # Neutral distributional descriptives (mean/median/std/iqr + delta).
        dist_summary = None
        try:
            dist_summary = compute_dist_summary(x=x, coords_9d=coords_9d, coords_9d_names=coords_9d_names)
        except Exception:
            logger.exception("Failed to compute dist_summary for %s", dataset_label)

        conventional_eeg_summary = None
        try:
            conventional_eeg_summary = compute_conventional_eeg_summary(
                sub_frame=sub_frame,
                config=config,
                dataset_id=self.dataset.ds_id,
            )
            if conventional_eeg_summary is not None:
                extensions_payload = dict(extensions_payload) if isinstance(extensions_payload, Mapping) else {}
                extensions_payload["conventional_eeg"] = conventional_eeg_summary
        except Exception:
            logger.exception("Failed to compute conventional_eeg summary for %s", dataset_label)

        review_qc_cfg = (
            (config.get("robustness", {}) or {}).get("review_qc", {})
            if isinstance(config, Mapping)
            else {}
        )
        mnps_3d_manifest_block = {
            "mode_requested": mde_mode_requested,
            "mode_effective": mde_mode_effective,
            "x_definition": x_definition,
            "from_v2": {
                "aggregation_requested": mde_from_v2_aggregation_requested,
                "aggregation": mde_from_v2_aggregation,
                "legacy_pooling": mde_from_v2_pooling_legacy,
                "map": m3d_cfg.get("map"),
                "v1_mapping_source": m3d_cfg.get("v1_mapping_source"),
                "v1_mapping_input": m3d_cfg.get("v1_mapping"),
                "v1_mapping_normalized": v1_mapping_normalized,
                "v1_mapping_matrix": v1_mapping_matrix,
                "v1_mapping_matrix_rows": v1_mapping_matrix_rows,
                "v1_mapping_hash": v1_mapping_hash,
                "fallback_reason": mde_from_v2_reason,
            },
        }
        baseline_comparisons = None
        try:
            baseline_comparisons = compute_feature_baseline_comparisons(
                sub_frame=sub_frame,
                x=x,
                dt_sec=float(dt),
                review_qc_cfg=review_qc_cfg,
            )
        except Exception:
            logger.exception("Failed to compute baseline_comparisons for %s", dataset_label)

        # Tier-1 time structure: autocorrelation length (tau)
        tau_summary = None
        try:
            tau_cfg = (config.get("robustness", {}) or {}).get("tau", {}) if isinstance(config, Mapping) else {}
            tau_nan_policy = str(tau_cfg.get("nan_policy", "strict")).strip().lower()
            tau_axes = compute_tau_summary(x, ["m", "d", "e"], dt_sec=float(dt), nan_policy=tau_nan_policy)
            tau_v2 = (
                compute_tau_summary(coords_9d, list(coords_9d_names), dt_sec=float(dt), nan_policy=tau_nan_policy)
                if coords_9d is not None and coords_9d_names
                else {}
            )
            tau_summary = {"axes": tau_axes, "subcoords": tau_v2} if (tau_axes or tau_v2) else None
        except Exception:
            logger.exception("Failed to compute tau_summary for %s", dataset_label)

        # Tier-2 MNJ-adjacent metrics from the primary Jacobian + derived indices
        tier2_jac = None
        try:
            if jac_res is not None:
                tier2_jac = compute_tier2_jacobian_metrics(
                    jac_res.j_hat,
                    jacobian_diagnostics=jac_res.diagnostics,
                )
        except Exception:
            logger.exception("Failed to compute tier2_jacobian_metrics for %s", dataset_label)

        mnps_mnj_sanity = None
        try:
            mnps_mnj_sanity = compute_mnps_mnj_sanity(
                x=x,
                x_dot=x_dot,
                time=time,
                dt_sec=float(dt),
                coords_9d=coords_9d,
                coords_9d_names=coords_9d_names,
                jacobian=jac_res.j_hat if jac_res is not None else None,
                jacobian_diagnostics=jac_res.diagnostics if jac_res is not None else None,
                review_qc_cfg=review_qc_cfg,
                projection_contract=mnps_3d_manifest_block,
                file_labels=sub_frame["file"].to_numpy() if "file" in sub_frame.columns else None,
                derivative_cfg=self.ctx.derivative_cfg,
            )
        except Exception:
            logger.exception("Failed to compute mnps_mnj_sanity for %s", dataset_label)

        emmi = None
        try:
            emmi = compute_emmi_metrics(x=x, x_dot=x_dot)
        except Exception:
            logger.exception("Failed to compute EMMI metrics for %s", dataset_label)

        null_sanity_tests = None
        try:
            file_labels = sub_frame["file"].to_numpy() if "file" in sub_frame.columns else None
            null_sanity_tests = compute_null_sanity_tests(
                x=x,
                dt_sec=float(dt),
                derivative_cfg=self.ctx.derivative_cfg,
                derivative_robust_cfg=((config.get("mnps", {}) or {}).get("derivative_robust", {}) or {})
                if isinstance(config, Mapping)
                else {},
                file_labels=file_labels,
                knn_k=mnps_cfg["knn_k"],
                knn_metric=mnps_cfg["knn_metric"],
                whiten=bool(mnps_cfg.get("whiten", True)),
                super_window=mnps_cfg["super_window"],
                ridge_alpha=mnps_cfg["ridge_alpha"],
                distance_weighted=bool(config.get("mnps", {}).get("ridge", {}).get("distance_weighted", True)),
                review_qc_cfg=review_qc_cfg,
                config=config,
            )
        except Exception:
            logger.exception("Failed to compute null_sanity_tests for %s", dataset_label)

        multiverse_psd = None
        if v2_enabled and coords_9d is not None and coords_9d_names and subcoords_spec:
            multiverse_psd = compute_psd_multiverse_stability(
                config=config,
                ds_id=self.dataset.ds_id,
                sub_frame=sub_frame,
                coords_9d=coords_9d,
                coords_9d_names=coords_9d_names,
                subcoords_spec=subcoords_spec,
                normalize_mode=normalize_mode,
            )

        env_meta = _get_env_provenance()
        feature_export_bundle = projection.build_feature_export_bundle(
            sub_frame,
            direct_features=direct_weighted_features,
            v2_features=v2_weighted_features,
            normalize_mode=normalize_mode,
            feature_standardization=feature_standardization if isinstance(feature_standardization, Mapping) else None,
            clip_threshold=clip_threshold,
            entropy_meta=entropy_meta,
        )
        features_raw_values = np.asarray(feature_export_bundle.get("raw_values"), dtype=np.float32)
        features_raw_names = list(feature_export_bundle.get("raw_names", []) or [])
        features_robust_z_values = np.asarray(feature_export_bundle.get("robust_z_values"), dtype=np.float32)
        features_robust_z_names = list(feature_export_bundle.get("robust_z_names", []) or [])
        features_projection_z_values = np.asarray(feature_export_bundle.get("projection_z_values"), dtype=np.float32)
        features_projection_z_names = list(feature_export_bundle.get("projection_z_names", []) or [])
        feature_metadata = dict(feature_export_bundle.get("metadata", {}) or {})
        (
            anchor_state_export,
            anchor_state_dot_export,
            anchor_quality_export,
            anchor_state_diagnostics,
        ) = build_anchor_state_exports(
            features_df=sub_frame,
            robust_z_values=features_robust_z_values,
            robust_z_names=features_robust_z_names,
            time=np.asarray(time, dtype=np.float64),
            config=config,
        )
        anchor_coupling_export: Dict[str, Any] = {}
        anchor_coupling_policy: Dict[str, Any] = {}
        anchor_coupling_cfg = (
            config.get("anchor_coupling", {})
            if isinstance(config, Mapping) and isinstance(config.get("anchor_coupling", {}), Mapping)
            else {}
        )
        if (
            anchor_coupling_cfg.get("enabled", False)
            and isinstance(anchor_state_export, Mapping)
            and isinstance(anchor_state_dot_export, Mapping)
            and anchor_state_export.get("values") is not None
            and anchor_state_dot_export.get("values") is not None
            and nn_indices is not None
        ):
            try:
                coupling_raw = jacobian.estimate_anchor_coupling(
                    np.asarray(x, dtype=np.float32),
                    np.asarray(x_dot, dtype=np.float32),
                    np.asarray(anchor_state_export.get("values"), dtype=np.float32),
                    np.asarray(anchor_state_dot_export.get("values"), dtype=np.float32),
                    np.asarray(nn_indices, dtype=np.int32),
                    super_window=int(anchor_coupling_cfg.get("super_window", 3) or 3),
                    ridge_alpha=float(anchor_coupling_cfg.get("ridge_alpha", 1.0) or 1.0),
                    distance_weighted=bool(anchor_coupling_cfg.get("distance_weighted", False)),
                    j_dot_dt=float(dt) if np.isfinite(float(dt)) and float(dt) > 0 else None,
                )
                anchor_coupling_export, anchor_coupling_policy = apply_anchor_coupling_window_policy(
                    coupling_raw,
                    condition_number_max=float(
                        anchor_coupling_cfg.get(
                            "condition_number_max",
                            STANDARD_GEOMETRY_MAX_JACOBIAN_CONDITION,
                        )
                    ),
                    min_windows=int(anchor_coupling_cfg.get("min_windows", 3) or 3),
                )
                if anchor_coupling_export:
                    anchor_coupling_export.setdefault("diagnostics", {})
                    if isinstance(anchor_coupling_export.get("diagnostics"), Mapping):
                        anchor_coupling_export["diagnostics"] = {
                            **dict(anchor_coupling_export.get("diagnostics", {}) or {}),
                            "policy": anchor_coupling_policy,
                        }
            except Exception:
                logger.exception("Failed to build anchor coupling export for %s", dataset_label)
        feature_names_hash = (
            hashlib.sha256(
                json.dumps(features_raw_names, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if features_raw_names
            else None
        )
        features_raw_hash_saved = _stable_hash_array(features_raw_values) if features_raw_values.size else None
        features_robust_z_hash_saved = (
            _stable_hash_array(features_robust_z_values) if features_robust_z_values.size else None
        )
        coordinate_layers: Dict[str, Any] = {
            **(
                {
                    "coords_3d_subject_anchored": {
                        "values": x_subject_anchored,
                        "names": ["m", "d", "e"],
                        "attrs": {
                            "coordinate_contract": "subject_anchored",
                            "normalize_mode": normalize_mode,
                            "role": "within_subject_geometry",
                        },
                    }
                }
                if export_subject_anchored
                else {}
            )
        }
        if export_subject_anchored and coords_9d_subject_anchored is not None and coords_9d_subject_names:
            coordinate_layers["coords_9d_subject_anchored"] = {
                "values": coords_9d_subject_anchored,
                "names": coords_9d_subject_names,
                "attrs": {
                    "coordinate_contract": "subject_anchored",
                    "normalize_mode": normalize_mode,
                    "role": "within_subject_stratified_geometry",
                },
            }
        if export_cohort_anchored and x_cohort_anchored is not None:
            coordinate_layers["coords_3d_cohort_anchored"] = {
                "values": x_cohort_anchored,
                "names": ["m", "d", "e"],
                "attrs": {
                    "coordinate_contract": "cohort_anchored",
                    "anchor_id": exported_anchor_spec.get("anchor_id"),
                    "anchor_hash": exported_anchor_spec.get("anchor_hash"),
                    "anchor_source": exported_anchor_spec.get("anchor_source"),
                    "role": "clinical_group_comparison",
                },
            }
        if export_cohort_anchored and coords_9d_cohort_anchored is not None and coords_9d_cohort_names:
            coordinate_layers["coords_9d_cohort_anchored"] = {
                "values": coords_9d_cohort_anchored,
                "names": coords_9d_cohort_names,
                "attrs": {
                    "coordinate_contract": "cohort_anchored",
                    "anchor_id": exported_anchor_spec.get("anchor_id"),
                    "anchor_hash": exported_anchor_spec.get("anchor_hash"),
                    "anchor_source": exported_anchor_spec.get("anchor_source"),
                    "role": "clinical_group_comparison_stratified",
                },
            }

        stage_codebook_export = _build_stage_codebook_export(
            effective_stage_codebook if isinstance(effective_stage_codebook, Mapping) else {},
            stage_source=stage_source,
            stage_column=stage_column,
            stage_events_path=stage_events_path,
        )
        event_windows_columns, event_windows_attrs = _build_event_windows_export(
            event_table_columns=event_table_columns,
            events_path=stage_events_path,
            time=time,
            window_start=window_start,
            window_end=window_end,
            stage=stage,
            window_sec=float(mnps_cfg["window_sec"]),
            overlap=float(mnps_cfg["overlap"]),
        )
        participant_clinical_meta = _build_participant_clinical_meta(
            participant_meta=participant_meta if isinstance(participant_meta, Mapping) else {},
            participant_meta_source=participant_meta_source if isinstance(participant_meta_source, Mapping) else {},
            mapped_meta=mapped_meta if isinstance(mapped_meta, Mapping) else {},
            session=ses_id,
            condition=condition,
            task=task,
            run_id=run_id,
            acq_id=acq_id,
        )
        qc_windows_export = _build_qc_windows_export(
            sub_frame=sub_frame,
            stage=stage,
            x=np.asarray(x, dtype=np.float32),
            coords_9d=np.asarray(coords_9d, dtype=np.float32) if coords_9d is not None else None,
            x_coverage=np.asarray(x_coverage, dtype=np.float32),
            min_axis_coverage=float(min_axis_coverage),
        )
        coverage_export = _build_coverage_export(
            x_coverage=np.asarray(x_coverage, dtype=np.float32),
            min_axis_coverage=float(min_axis_coverage),
            coordinate_layers=coordinate_layers,
            jacobian_centers=jac_res.centers if jac_res is not None else None,
            jacobian_9d_centers=jac_res_v2.centers if jac_res_v2 is not None else None,
        )
        available_coordinate_layers = sorted([str(name) for name in coordinate_layers.keys()])
        available_jacobian_layers = sorted([str(name) for name in jacobian_layers.keys()])
        available_coordinate_contracts = sorted(
            {
                str((layer.get("attrs", {}) or {}).get("coordinate_contract"))
                for layer in coordinate_layers.values()
                if isinstance(layer, Mapping)
                and isinstance(layer.get("attrs", {}), Mapping)
                and (layer.get("attrs", {}) or {}).get("coordinate_contract") is not None
            }
        )
        event_table_schema_version = _decode_text_scalar(event_table_columns.get("_schema_version"))
        normalization_report = dict(getattr(self.dataset, "_normalization_report", {}) or {})
        normalization_report_info = normalization_report.get("report_file", {}) if isinstance(normalization_report, Mapping) else {}
        mapping_contract = _resolve_meg_mapping_contract(
            self.ctx.config if isinstance(self.ctx.config, Mapping) else {},
            self.dataset.ds_id,
        )
        mapping_provenance: Dict[str, Any] = {}
        if mapping_contract:
            mapping_provenance = {
                "modality": str(config.get("modality", "")).strip().lower() or "meg",
                "primary_surface": mapping_contract.get("primary_surface"),
                "paired_surfaces": list(mapping_contract.get("paired_surfaces", []) or []),
                "mapping_family": mapping_contract.get("mapping_family"),
                "mapping_reference": mapping_contract.get("mapping_reference"),
                "sensor_types": list(mapping_contract.get("sensor_types", []) or []),
                "feature_combination": mapping_contract.get("feature_combination"),
                "validation_pilot": dict(mapping_contract.get("validation_pilot", {}) or {}),
            }
        provenance_export: Dict[str, Any] = {
            "contract": {
                "export_contract_version": "mndm.eeg_h5_contract.v1",
                "config_digest_sha256": _stable_hash_mapping(self.ctx.config if isinstance(self.ctx.config, Mapping) else {}),
                "config_filename": getattr(getattr(self.dataset, "config_path", None), "name", None),
                "run_manifest_ref": "../run_manifest.json",
                "geometry_invalidity_policy": STANDARD_GEOMETRY_POLICY_VERSION,
                "geometry_contract_status": geometry_contract.get("status"),
            },
            "anchoring": {
                "available_coordinate_layers": available_coordinate_layers,
                "available_jacobian_layers": available_jacobian_layers,
                "available_coordinate_contracts": available_coordinate_contracts,
                "requested_contracts": list(export_contracts.get("requested_contracts", []) or []),
                "realized_contracts": list(export_contracts.get("realized_contracts", []) or []),
                "skipped_contracts_with_reason": list(
                    export_contracts.get("skipped_contracts_with_reason", []) or []
                ),
                "primary_coordinate_layer": (
                    "coords_3d_cohort_anchored" if primary_coordinate_layer == "cohort_anchored" else "coords_3d_subject_anchored"
                ),
                "primary_coordinate_contract": primary_coordinate_layer,
                "anchor_id": exported_anchor_spec.get("anchor_id"),
                "anchor_hash": exported_anchor_spec.get("anchor_hash"),
                "anchor_source": exported_anchor_spec.get("anchor_source"),
            },
            "normalization": {
                "status": normalization_report.get("status") if isinstance(normalization_report, Mapping) else None,
                "method": normalization_report.get("method") if isinstance(normalization_report, Mapping) else None,
                "scope": normalization_report.get("scope") if isinstance(normalization_report, Mapping) else None,
                "batch_key": normalization_report.get("batch_key") if isinstance(normalization_report, Mapping) else None,
                "report_ref": (
                    "../normalization_report.json"
                    if isinstance(normalization_report_info, Mapping) and normalization_report_info.get("status") == "written"
                    else None
                ),
            },
            "event_stage_mapping": {
                "event_mapping_version": event_table_schema_version,
                "stage_mapping_version": event_table_schema_version,
                "stage_source": stage_source,
                "stage_column": stage_column,
                "stage_events_path": stage_events_path,
                "stage_codebook_hash": _stable_hash_mapping(effective_stage_codebook)
                if isinstance(effective_stage_codebook, Mapping) and effective_stage_codebook
                else None,
            },
            "anchor_state": anchor_state_diagnostics,
            "geometry_contract": geometry_contract,
        }
        if mapping_provenance:
            provenance_export["mapping"] = mapping_provenance
        regional_mnps_export = _build_regional_dual_contract_export(
            primary_coordinate_contract=primary_coordinate_layer,
            subject_summary=regional_mnps_results_subject if export_subject_anchored else None,
            cohort_summary=regional_mnps_results_cohort if export_cohort_anchored else None,
            anchor_spec=exported_anchor_spec if isinstance(exported_anchor_spec, Mapping) else {},
        )

        # Build per-row source provenance (replaces implicit stacked-half slicing).
        # The file column encodes which raw file each row was computed from.
        _row_source_cols: dict = {}
        if "file" in sub_frame.columns and len(sub_frame) > 0:
            _raw_files = sub_frame["file"].fillna("").astype(str).to_numpy()
            def _classify_source(fname: str) -> str:
                fl = fname.lower()
                if fl.endswith(".fif") or fl.endswith(".fif.gz"):
                    return "fif_meeg"
                if fl.endswith(".set") or fl.endswith(".fdt"):
                    return "set_eeg"
                return "unknown"
            _row_src = np.array([_classify_source(f) for f in _raw_files], dtype=object)
            _has_meg  = (_row_src == "fif_meeg").astype(np.int8)
            _has_eeg  = np.ones(len(_raw_files), dtype=np.int8)
            _has_mag  = _has_meg.copy()
            _has_grad = _has_meg.copy()
            _src_fmt  = np.array(
                ["neuromag_fif" if r == "fif_meeg" else "eeglab_set" if r == "set_eeg" else "unknown"
                 for r in _row_src], dtype=object
            )
            _row_source_cols = {
                "row_source":    _row_src,
                "has_meg":       _has_meg,
                "has_eeg":       _has_eeg,
                "has_mag":       _has_mag,
                "has_grad":      _has_grad,
                "raw_file":      np.array([Path(f).name for f in _raw_files], dtype=object),
                "source_format": _src_fmt,
            }

        # Build payload
        payload = schema.MNPSPayload(
            time=time,
            x=x,
            x_dot=x_dot,
            window_start=window_start,
            window_end=window_end,
            stage=stage,
            z=z,
            events=events,
            event_table_columns=event_table_columns,
            event_windows=event_windows_columns,
            event_windows_attrs=event_windows_attrs,
            codebooks=stage_codebook_export,
            nn_indices=nn_indices,
            jacobian=jac_res.j_hat if jac_res is not None else None,
            jacobian_dot=jac_res.j_dot if jac_res is not None else None,
            jacobian_centers=jac_res.centers if jac_res is not None else None,
            jacobian_9D=jac_res_v2.j_hat if jac_res_v2 is not None else None,
            jacobian_9D_dot=jac_res_v2.j_dot if jac_res_v2 is not None else None,
            jacobian_9D_centers=jac_res_v2.centers if jac_res_v2 is not None else None,
            feature_baselines=merged_baselines,
            features_raw_values=features_raw_values,
            features_raw_names=features_raw_names,
            features_robust_z_values=features_robust_z_values,
            features_robust_z_names=features_robust_z_names,
            features_projection_z_values=features_projection_z_values,
            features_projection_z_names=features_projection_z_names,
            feature_metadata=feature_metadata,
            coordinate_layers=coordinate_layers,
            feature_anchors=(anchor_artifact or {}) if export_cohort_anchored else {},
            jacobian_layers=jacobian_layers,
            anchor_state=anchor_state_export,
            anchor_state_dot=anchor_state_dot_export,
            anchor_quality=anchor_quality_export,
            anchor_coupling=anchor_coupling_export,
            participant_clinical_meta=participant_clinical_meta,
            provenance=provenance_export,
            coverage=coverage_export,
            qc_windows=qc_windows_export,
            regional_mnps=regional_mnps_export,
            row_source_columns=_row_source_cols,
            attrs={
                # Stable identity fields (used downstream for grouping/contrasts).
                "dataset": self.dataset.ds_id,
                "subject_id": sub_id,
                "session": ses_id,
                "fs_out": mnps_cfg["fs_out"],
                "window_sec": mnps_cfg["window_sec"],
                "overlap": mnps_cfg["overlap"],
                "stage_codebook": effective_stage_codebook,
                "stage_source": stage_source,
                "stage_column": stage_column,
                "stage_events_path": stage_events_path,
                "stage_frac_labeled": stage_frac_labeled,
                "pseudo_stage_status": stage_proxy_info.get("status") if isinstance(stage_proxy_info, Mapping) else None,
                "pseudo_stage_label": stage_proxy_info.get("label") if isinstance(stage_proxy_info, Mapping) else None,
                "pseudo_stage_code": stage_proxy_info.get("code") if isinstance(stage_proxy_info, Mapping) else None,
                "pseudo_stage_labeled_fraction": stage_proxy_info.get("labeled_fraction") if isinstance(stage_proxy_info, Mapping) else None,
                "pseudo_stage_sigma_column": stage_proxy_info.get("sigma_column") if isinstance(stage_proxy_info, Mapping) else None,
                "pseudo_stage_emg_column": stage_proxy_info.get("emg_column") if isinstance(stage_proxy_info, Mapping) else None,
                "pseudo_stage_spindle_column": stage_proxy_info.get("spindle_column") if isinstance(stage_proxy_info, Mapping) else None,
                "participant_meta": participant_meta,
                "participant_meta_source": participant_meta_source,
                "participant_mapped_meta": mapped_meta,
                "group": mapped_meta.get("group"),
                "condition": condition,
                "task": task,
                "run": run_id,
                "acq": acq_id,
                "modality": str(config.get("modality", "")).strip().lower() if isinstance(config, Mapping) else None,
                "coverage_rule_tag": coverage_tag,
                "coverage_min_seconds_effective": min_seconds_eff,
                "coverage_min_epochs_effective": min_epochs_eff,
                "coverage_seconds_effective": coverage_seconds_effective,
                "coverage_seconds_measured": coverage_seconds_measured,
                "coverage_seconds_assumed": coverage_seconds_assumed,
                "coverage_seconds_method": coverage_method,
                "epochs_raw": n_before_any,
                "epochs_after_qc": n_after_qc,
                "epochs_after_nan_mask": int(len(sub_frame)),
                "epochs_after_geometry_policy": int(len(sub_frame)),
                "mndm_version": "2.1",
                "export_contract_version": "mndm.eeg_h5_contract.v1",
                "primary_coordinate_layer": (
                    "coords_3d_cohort_anchored" if primary_coordinate_layer == "cohort_anchored" else "coords_3d_subject_anchored"
                ),
                "primary_coordinate_contract": primary_coordinate_layer,
                "available_jacobian_layers": available_jacobian_layers,
                "anchor_id": (
                    exported_anchor_spec.get("anchor_id")
                ),
                "anchor_hash": (
                    exported_anchor_spec.get("anchor_hash")
                ),
                "anchor_state_names": list(anchor_state_export.get("names", []) or []),
                "x_definition": x_definition,
                "mde_mode_requested": mde_mode_requested,
                "mde_mode_effective": mde_mode_effective,
                "mde_from_v2_aggregation_requested": mde_from_v2_aggregation_requested,
                "mde_from_v2_aggregation": mde_from_v2_aggregation,
                "mde_from_v2_pooling_legacy": mde_from_v2_pooling_legacy,
                "mde_from_v2_map": m3d_cfg.get("map"),
                "mde_from_v2_v1_mapping_source": m3d_cfg.get("v1_mapping_source"),
                "mde_from_v2_v1_mapping_input": m3d_cfg.get("v1_mapping"),
                "mde_from_v2_v1_mapping_normalized": v1_mapping_normalized,
                "mde_from_v2_v1_mapping_matrix": v1_mapping_matrix,
                "mde_from_v2_v1_mapping_matrix_rows": v1_mapping_matrix_rows,
                "mde_from_v2_v1_mapping_hash": v1_mapping_hash,
                "mde_from_v2_fallback_reason": mde_from_v2_reason,
                "v2_definition": f"subcoords_9d_v{str(v2_definition_version).replace('.', '_')}",
                "mnps_9d_definition_version": v2_definition_version,
                "mnps_9d_constructs": mnps_9d_constructs,
                "normalize_mode": normalize_mode,
                "missing_axis_policy": missing_axis_policy,
                "dropped_missing_axis_epochs": dropped_missing_axis_epochs,
                "geometry_invalidity_policy": STANDARD_GEOMETRY_POLICY_VERSION,
                "geometry_contract_status": geometry_contract.get("status"),
                "dropped_geometry_invalid_epochs": dropped_geometry_invalid_epochs,
                "geometry_jacobian_invalid_windows": int((geometry_contract.get("jacobian") or {}).get("invalid_windows", 0)),
                "geometry_jacobian_9d_invalid_windows": int((geometry_contract.get("jacobian_9d") or {}).get("invalid_windows", 0)),
                "weights_hash_direct": _stable_hash_mapping(self.ctx.weights or {}),
                "subcoords_hash_v2": _stable_hash_mapping(subcoords_spec if isinstance(subcoords_spec, Mapping) else {}),
                "missing_weighted_feature_rate_direct": missing_rate_direct,
                "missing_weighted_feature_rate_v2": missing_rate_v2,
                "direct_axis_renorm": direct_axis_renorm,
                "direct_axis_coverage_m_mean": axis_cov_stats["m_mean"],
                "direct_axis_coverage_d_mean": axis_cov_stats["d_mean"],
                "direct_axis_coverage_e_mean": axis_cov_stats["e_mean"],
                "direct_axis_coverage_m_min": float(np.nanmin(x_coverage[:, 0])) if x_coverage.size and not np.all(np.isnan(x_coverage[:, 0])) else float("nan"),
                "direct_axis_coverage_d_min": float(np.nanmin(x_coverage[:, 1])) if x_coverage.size and not np.all(np.isnan(x_coverage[:, 1])) else float("nan"),
                "direct_axis_coverage_e_min": float(np.nanmin(x_coverage[:, 2])) if x_coverage.size and not np.all(np.isnan(x_coverage[:, 2])) else float("nan"),
                "min_axis_coverage": float(min_axis_coverage),
                "v2_missing_policy": v2_missing_policy if v2_enabled and subcoords_spec else None,
                "coords_9d_allow_all_non_finite_columns": True if v2_enabled and subcoords_spec else False,
                "coords_9d_allow_duplicate_columns": True if v2_enabled and subcoords_spec else False,
                "coords_9d_allow_duplicate_constant_columns": True if v2_enabled and subcoords_spec else False,
                "coords_9d_degraded_mode": bool(v2_all_non_finite_count > 0 or v2_duplicate_count > 0),
                "coords_9d_all_non_finite_count": int(v2_all_non_finite_count),
                "coords_9d_all_non_finite_names": v2_all_non_finite_names if v2_all_non_finite_names else None,
                "coords_9d_duplicate_count": int(v2_duplicate_count),
                "coords_9d_duplicate_pairs": v2_duplicate_pairs if v2_duplicate_pairs else None,
                "coords_9d_duplicate_constant_count": int(v2_duplicate_constant_count),
                "coords_9d_duplicate_constant_pairs": v2_duplicate_constant_pairs if v2_duplicate_constant_pairs else None,
                "e_e_construct": entropy_meta.get("construct"),
                "e_e_metric": entropy_meta.get("metric"),
                "e_e_backend": entropy_meta.get("backend"),
                "e_e_degraded_mode": bool(entropy_meta.get("degraded_mode", False)),
                "e_e_reason": entropy_meta.get("reason"),
                "python_version": env_meta.get("python_version"),
                "platform": env_meta.get("platform"),
                "pip_freeze_hash": env_meta.get("pip_freeze_hash"),
                "env_hash": env_meta.get("env_hash"),
                "x_hash_saved": x_hash_saved,
                "x_hash_knn_input": x_hash_knn_input,
                "x_hash_jacobian_input": x_hash_jac_input,
                "nn_indices_hash_saved": nn_indices_hash_saved,
                "jacobian_hash_saved": jacobian_hash_saved,
                "jacobian_dot_hash_saved": jacobian_dot_hash_saved,
                "coords_9d_hash_saved": coords_9d_hash_saved,
                "coords_9d_hash_knn_input": coords_9d_hash_knn_input,
                "coords_9d_hash_jacobian_input": coords_9d_hash_jac_input,
                "jacobian_9d_hash_saved": jacobian_9d_hash_saved,
                "jacobian_9d_dot_hash_saved": jacobian_9d_dot_hash_saved,
                "coords_9d_names": coords_9d_names if coords_9d_names else None,
                "feature_export_scope": "all_numeric_feature_columns_excluding_metadata",
                "feature_export_names_hash": feature_names_hash,
                "features_raw_hash_saved": features_raw_hash_saved,
                "features_robust_z_hash_saved": features_robust_z_hash_saved,
                "features_raw_column_count": int(len(features_raw_names)),
                "features_robust_z_column_count": int(len(features_robust_z_names)),
                "feature_metadata_fields": sorted(feature_metadata.keys()) if feature_metadata else [],
                "reproducibility_seed": int(reproducibility_policy.get("seed", 42)),
                "reproducibility_seed_source": str(reproducibility_policy.get("seed_source", "default")),
                "mapping_family": mapping_provenance.get("mapping_family"),
                "mapping_reference": mapping_provenance.get("mapping_reference"),
                "mapping_primary_surface": mapping_provenance.get("primary_surface"),
                "mapping_paired_surfaces": mapping_provenance.get("paired_surfaces"),
                "sensor_types": mapping_provenance.get("sensor_types"),
                "feature_combination": mapping_provenance.get("feature_combination"),
                "validation_pilot_subjects": (
                    list((mapping_provenance.get("validation_pilot") or {}).get("subjects", []) or [])
                    if isinstance(mapping_provenance.get("validation_pilot"), Mapping)
                    else None
                ),
                **(
                    dict(time_reference_result.get("attrs"))
                    if isinstance(time_reference_result.get("attrs"), Mapping)
                    else {}
                ),
            },
        )
        if v2_enabled and coords_9d_names and coords_9d is not None and coords_9d.size:
            payload.coords_9d = coords_9d.astype(np.float32)
            payload.coords_9d_names = coords_9d_names
        if stratified_blocks_result is not None and stratified_blocks_result.cross_partials_series:
            payload.jacobian_9D_cross_partials = stratified_blocks_result.cross_partials_series
        if regions_bold is not None:
            payload.regions_bold = regions_bold
            if regions_names is not None:
                payload.regions_names = regions_names
            if regions_sfreq is not None:
                payload.regions_sfreq = regions_sfreq
        if extensions_payload:
            payload.extensions = extensions_payload

        # Event mapping to labels (opt-in)
        labels_combined: Dict[str, np.ndarray] = {}
        if within_run_labels.labels:
            labels_combined.update(within_run_labels.labels)
        for label_col in ("task_state_label", "task_load_label", "task_load_n"):
            if label_col in sub_frame.columns:
                try:
                    values = sub_frame[label_col].to_numpy()
                    if np.asarray(values).shape == (len(time),):
                        labels_combined.setdefault(label_col, values)
                except Exception:
                    logger.exception("Failed to export label column '%s' for %s", label_col, dataset_label)
        stage_bool_labels = _build_stage_bool_labels(
            stage,
            effective_stage_codebook if isinstance(effective_stage_codebook, Mapping) else {},
        )
        for label_name, label_arr in stage_bool_labels.items():
            labels_combined.setdefault(label_name, label_arr)
        labels_mapped = self._map_events_to_labels(config, time, window_start, window_end, events)
        if labels_mapped:
            labels_combined.update(labels_mapped)
        if labels_combined:
            payload.labels = labels_combined

        # Entropy QC checks
        entropy_qc = {}
        if coords_9d is not None and coords_9d_names:
            try:
                entropy_qc = robustness.entropy_sanity_checks(coords_9d, coords_9d_names)
            except Exception:
                logger.exception("Entropy sanity checks failed for %s", dataset_label)

        # Build manifest
        manifest_extra = {
            "subject": sub_id,
            "session": ses_id,
            "run": run_id,
            "acq": acq_id,
            "participant_meta": participant_meta,
            "participant_meta_source": participant_meta_source,
            "participant_mapped_meta": mapped_meta,
            "group": mapped_meta.get("group"),
            "condition": condition,
            "task": task,
            "stage_source": stage_source,
            "stage_column": stage_column,
            "stage_events_path": stage_events_path,
            "stage_frac_labeled": stage_frac_labeled,
            "pseudo_stage": stage_proxy_info if isinstance(stage_proxy_info, Mapping) else None,
            "within_run_labels": summarize_within_run_manifest(within_run_labels.manifest) if within_run_labels.manifest else None,
            "coverage": {
                "rule_tag": coverage_tag,
                "min_seconds_effective": min_seconds_eff,
                "min_epochs_effective": min_epochs_eff,
                "seconds_effective": coverage_seconds_effective,
                "seconds_measured": coverage_seconds_measured,
                "seconds_assumed": coverage_seconds_assumed,
                "seconds_method": coverage_method,
            },
            "coverage_h5": {
                "status": "available",
                "path": "/coverage",
                "axis_fraction_path": "/coverage/axis_fraction",
                "coordinate_layers_present": available_coordinate_layers,
                "jacobian_layers_present": available_jacobian_layers,
            },
            "provenance_h5": {
                "status": "available",
                "path": "/provenance",
                "contract_path": "/provenance/contract",
                "anchoring_path": "/provenance/anchoring",
            },
            "jacobian_h5": {
                "primary_path": "/jacobian",
                "primary_v2_path": "/jacobian_9D" if jac_res_v2 is not None else None,
                "layer_paths": [f"/{name}" for name in available_jacobian_layers],
            },
            "participant_h5": {
                "clinical_json_path": "/participant/clinical_json",
            },
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        if isinstance(time_reference_result.get("manifest"), Mapping):
            manifest_extra["time_reference"] = dict(time_reference_result["manifest"])
        if stage_mapping_qc_entry:
            stage_mapping_qc_entry = dict(stage_mapping_qc_entry)
            stage_mapping_qc_entry.update(
                {
                    "subject": sub_id,
                    "session": ses_id,
                    "run": run_id,
                    "acq": acq_id,
                    "task": task,
                    "condition": condition,
                    "stage_source": stage_source,
                    "stage_column": stage_column,
                    "stage_frac_labeled": stage_frac_labeled,
                }
            )
            manifest_extra["stage_mapping_qc"] = stage_mapping_qc_entry
            self.dataset._record_stage_mapping_qc_entry(stage_mapping_qc_entry)
        if event_table_columns:
            manifest_extra["event_provenance"] = {
                "status": "available",
                "events_path": stage_events_path,
                "event_table_columns": sorted([str(k) for k in event_table_columns.keys()]),
                "event_rows": int(
                    len(np.asarray(event_table_columns.get("onset_sec", [])))
                    if "onset_sec" in event_table_columns
                    else 0
                ),
            }
        if event_windows_columns or event_windows_attrs:
            manifest_extra["event_windows"] = {
                "status": "available",
                "path": "/event_windows",
                "rows": int(event_windows_attrs.get("n_rows", 0) or 0),
                "reference": event_windows_attrs.get("reference"),
                "bins_json": event_windows_attrs.get("bins_json"),
            }
        if stage_codebook_export:
            manifest_extra["codebooks"] = {
                "available": sorted([str(k) for k in stage_codebook_export.keys()]),
                "stage_path": "/codebooks/stage" if "stage" in stage_codebook_export else None,
            }
        if regional_mnps_export:
            manifest_extra["regional_outputs_h5"] = {
                "path": "/regional_mnps",
                "available_coordinate_contracts": available_coordinate_contracts,
                "primary_coordinate_contract": primary_coordinate_layer,
            }
        manifest_extra["coordinate_contracts"] = {
            "requested_contracts": list(export_contracts.get("requested_contracts", []) or []),
            "realized_contracts": list(export_contracts.get("realized_contracts", []) or []),
            "skipped_contracts_with_reason": list(
                export_contracts.get("skipped_contracts_with_reason", []) or []
            ),
            "available_coordinate_contracts": available_coordinate_contracts,
            "primary_coordinate_contract": primary_coordinate_layer,
        }
        if stratified_blocks_result is not None:
            if stratified_blocks_result.blocks_manifest:
                rows_light: list[dict[str, Any]] = []
                for r in stratified_blocks_result.block_rows:
                    rows_light.append(
                        {
                            "out_group": r.get("out_group"),
                            "in_group": r.get("in_group"),
                            "out_dim": r.get("out_dim"),
                            "in_dim": r.get("in_dim"),
                            "n_timepoints": r.get("n_timepoints"),
                            "block_trace_mean": r.get("block_trace_mean"),
                            "block_frobenius_mean": r.get("block_frobenius_mean"),
                            "block_anisotropy_mean": r.get("block_anisotropy_mean"),
                            "c_sym_mean": r.get("c_sym_mean"),
                            "c_rot_mean": r.get("c_rot_mean"),
                        }
                    )
                manifest_extra["jacobian_9D_blocks"] = {
                    "config": stratified_blocks_result.blocks_manifest,
                    "rows": rows_light,
                }
            if stratified_blocks_result.cross_partials_manifest:
                manifest_extra["jacobian_9D_cross_partials"] = stratified_blocks_result.cross_partials_manifest
        if ensemble_summary is not None:
            manifest_extra["ensemble_robustness"] = ensemble_summary
        if robust_summary:
            manifest_extra["robust_summary"] = robust_summary
        if dist_summary:
            manifest_extra["dist_summary"] = dist_summary
        if conventional_eeg_summary:
            manifest_extra["conventional_eeg"] = conventional_eeg_summary
        if baseline_comparisons is not None:
            manifest_extra["baseline_comparisons"] = baseline_comparisons
        if tau_summary:
            manifest_extra["tau_summary"] = tau_summary
        if tier2_jac:
            manifest_extra["tier2_jacobian"] = tier2_jac
        if emmi:
            manifest_extra["tier2_emmi"] = emmi
        if null_sanity_tests is not None:
            manifest_extra["null_sanity_tests"] = null_sanity_tests
        if multiverse_psd is not None:
            manifest_extra["multiverse_psd"] = multiverse_psd
        if entropy_qc:
            manifest_extra["entropy_qc"] = entropy_qc
        if extensions_summary:
            manifest_extra["extensions"] = extensions_summary
        if self.ctx.ingest_meta:
            manifest_extra["ndt_ingest"] = self.ctx.ingest_meta
        if modularity_provisional_frac is not None:
            manifest_extra["fmri_modularity_provisional_frac"] = modularity_provisional_frac
        if labels_mapped:
            manifest_extra["events_mapped"] = sorted(labels_mapped.keys())
        manifest_extra["mnps_3d"] = mnps_3d_manifest_block
        manifest_extra["geometry_contract"] = geometry_contract
        if mnps_mnj_sanity is not None:
            manifest_extra["mnps_mnj_sanity"] = mnps_mnj_sanity
        manifest_extra["provenance"] = {
            "mnps_9d_definition_version": v2_definition_version,
            "mnps_9d_constructs": mnps_9d_constructs,
            "reproducibility": {
                **reproducibility_policy,
                "nn_indices_hash_saved": nn_indices_hash_saved,
                "x_hash_saved": x_hash_saved,
                "x_hash_knn_input": x_hash_knn_input,
                "x_hash_jacobian_input": x_hash_jac_input,
                "jacobian_hash_saved": jacobian_hash_saved,
                "jacobian_dot_hash_saved": jacobian_dot_hash_saved,
                "coords_9d_hash_saved": coords_9d_hash_saved,
                "coords_9d_hash_knn_input": coords_9d_hash_knn_input,
                "coords_9d_hash_jacobian_input": coords_9d_hash_jac_input,
                "jacobian_9d_hash_saved": jacobian_9d_hash_saved,
                "jacobian_9d_dot_hash_saved": jacobian_9d_dot_hash_saved,
            },
        }
        if mapping_provenance:
            manifest_extra["provenance"]["mapping"] = mapping_provenance
        manifest_extra["feature_exports"] = {
            "raw_h5_path": "/features_raw",
            "robust_z_h5_path": "/features_robust_z",
            "scope": "all_numeric_feature_columns_excluding_metadata",
            "column_count": int(len(features_raw_names)),
            "names_hash": feature_names_hash,
            "metadata_fields": sorted(feature_metadata.keys()) if feature_metadata else [],
        }
        hrv_feature_names = [str(name) for name in features_raw_names if str(name).startswith("ecg_hrv_")]
        if hrv_feature_names:
            features_cfg = config.get("features", {}) if isinstance(config, Mapping) else {}
            ecg_cfg = features_cfg.get("ecg", {}) if isinstance(features_cfg, Mapping) else {}
            hrv_cfg = ecg_cfg.get("hrv", {}) if isinstance(ecg_cfg, Mapping) else {}
            if not isinstance(hrv_cfg, Mapping):
                hrv_cfg = {}
            manifest_extra["anchor_hrv_v0_1"] = {
                "enabled": True,
                "source": "ecg",
                "window_sec": float(hrv_cfg.get("superwindow_s", 60.0) or 60.0),
                "window_mode": str(hrv_cfg.get("window_mode", "centered") or "centered"),
                "min_nn_intervals": int(hrv_cfg.get("min_nn_intervals", 20) or 20),
                "min_coverage_fraction": float(hrv_cfg.get("min_coverage_fraction", 0.5) or 0.5),
                "max_artifact_fraction": float(hrv_cfg.get("max_artifact_fraction", 0.25) or 0.25),
                "feature_names": hrv_feature_names,
                "quality_feature_names": [
                    name
                    for name in hrv_feature_names
                    if name.endswith("_quality_score")
                    or name.endswith("_artifact_fraction")
                    or name.endswith("_coverage_fraction")
                    or name.endswith("_nn_count")
                ],
                "features_h5_path": "/features_raw",
            }
        if anchor_state_diagnostics:
            manifest_extra["anchor_state"] = {
                **anchor_state_diagnostics,
                "h5_paths": {
                    "anchor_state": "/anchor_state",
                    "anchor_state_dot": "/anchor_state_dot",
                    "anchor_quality": "/anchor_quality",
                },
            }
        if anchor_coupling_policy:
            manifest_extra["anchor_coupling"] = {
                **anchor_coupling_policy,
                "path": "/anchor_coupling",
            }

        # Add a clear note indicating that these tier-2 metrics are tentative and belong in the analysis repo
        if any([tau_summary, tier2_jac, emmi, dist_summary, baseline_comparisons, null_sanity_tests, mnps_mnj_sanity]):
            manifest_extra["_TENTATIVE_NOTE"] = (
                "Metrics such as baseline_comparisons, mnps_mnj_sanity, null_sanity_tests, "
                "tier2_jacobian, tier2_emmi, tau_summary, and dist_summary are provided as tentative QA summaries only. "
                "Real statistical verification "
                "and interpretation must be performed downstream in the analysis repository."
            )

        primary_jac_diagnostics = jac_res.diagnostics if jac_res is not None else None

        self._write_qc_files(
            target_dir=target_dir,
            dataset_label=dataset_label,
            sub_id=sub_id,
            ses_id=ses_id,
            sub_frame=sub_frame,
            dt=dt,
            ensemble_summary=ensemble_summary,
            robust_summary=robust_summary,
            dist_summary=dist_summary,
            baseline_comparisons=baseline_comparisons,
            tau_summary=tau_summary,
            tier2_jacobian=tier2_jac,
            tier2_emmi=emmi,
            null_sanity_tests=null_sanity_tests,
            entropy_qc=entropy_qc,
            geometry_contract=geometry_contract,
            mnps_mnj_sanity=mnps_mnj_sanity,
        )
        event_locked_cfg = {}
        if isinstance(config, Mapping):
            event_locked_root = config.get("event_locked", {})
            if isinstance(event_locked_root, Mapping):
                event_locked_datasets = event_locked_root.get("datasets", {})
                if isinstance(event_locked_datasets, Mapping):
                    event_locked_cfg = event_locked_datasets.get(self.dataset.ds_id, {})
                    if not isinstance(event_locked_cfg, Mapping):
                        event_locked_cfg = {}
        event_source_kind = (
            _resolve_event_locked_source_kind(event_locked_cfg)
            if isinstance(event_locked_cfg, Mapping)
            else "csv"
        )
        if event_locked_cfg.get("enabled", False):
            try:
                from .event_locked_runner import run_event_locked_export

                event_locked_entries: List[Dict[str, Any]] = []
                base_config = config if isinstance(config, Mapping) else {}
                legacy_name_parts = [
                    _safe_filename_token(sub_id or "sub-unknown"),
                    _safe_filename_token(raw_task or task or ""),
                    _safe_filename_token(acq_id or ""),
                    _safe_filename_token(run_id or ""),
                ]
                legacy_name_parts = [p for p in legacy_name_parts if p]
                legacy_event_locked_prefix = "_".join(legacy_name_parts) or "event_locked"

                if event_source_kind == "csv":
                    csv_sources = _resolve_event_locked_csv_sources(
                        stage_events_path=stage_events_path,
                        event_locked_cfg=event_locked_cfg,
                    )
                    for csv_path, csv_slug in csv_sources:
                        out_prefix = target_dir / "event_locked"
                        if csv_slug:
                            out_prefix = target_dir / f"{legacy_event_locked_prefix}_event_locked_v1_{csv_slug}"
                        event_locked_result = run_event_locked_export(
                            payload=payload,
                            config=base_config,
                            dataset_id=self.dataset.ds_id,
                            source_path=Path(csv_path),
                            event_table=None,
                            subject_id=sub_id or "",
                            session_id=ses_id or "",
                            run_id=run_id or "",
                            out_prefix=out_prefix,
                        )
                        entry = dict(event_locked_result.manifest_entry)
                        entry["resolved_source_path"] = str(csv_path)
                        if csv_slug:
                            entry["channel_slug"] = csv_slug
                        event_locked_entries.append(entry)

                elif event_source_kind == "bids_events":
                    # Direct BIDS events.tsv event-locking.
                    # Uses the companion *_events.tsv file identified by
                    # stage_events_path (already resolved per-subject above).
                    # No derived:task_state_label column required.
                    bids_events_src = stage_events_path or None
                    if bids_events_src:
                        out_prefix = (
                            target_dir
                            / f"{legacy_event_locked_prefix}_event_locked_bids_v1"
                        )
                        event_locked_result = run_event_locked_export(
                            payload=payload,
                            config=base_config,
                            dataset_id=self.dataset.ds_id,
                            source_path=Path(bids_events_src),
                            event_table=None,
                            subject_id=sub_id or "",
                            session_id=ses_id or "",
                            run_id=run_id or "",
                            out_prefix=out_prefix,
                        )
                        entry = dict(event_locked_result.manifest_entry)
                        entry["resolved_source_path"] = str(bids_events_src)
                        entry["event_source_kind"] = "bids_events"
                        event_locked_entries.append(entry)
                    else:
                        logger.warning(
                            "event_locked bids_events: no BIDS events.tsv path resolved for %s",
                            dataset_label,
                        )

                elif event_source_kind == "derived_stage_block_end" and stage_events_path:
                    event_locked_result = run_event_locked_export(
                        payload=payload,
                        config=base_config,
                        dataset_id=self.dataset.ds_id,
                        source_path=Path(stage_events_path),
                        event_table=None,
                        subject_id=sub_id or "",
                        session_id=ses_id or "",
                        run_id=run_id or "",
                        out_prefix=target_dir / "event_locked",
                    )
                    event_locked_entries.append(dict(event_locked_result.manifest_entry))

                derived_event_table = None
                task_state_series = payload.labels.get("task_state_label") if isinstance(payload.labels, Mapping) else None
                if (
                    task_state_series is not None
                    and payload.window_start is not None
                    and payload.window_end is not None
                ):
                    derived_event_table = build_label_segment_event_table(
                        labels=task_state_series,
                        window_start=np.asarray(payload.window_start, dtype=np.float64),
                        window_end=np.asarray(payload.window_end, dtype=np.float64),
                        source_path=str(stage_events_path or "derived:task_state_label"),
                        source_label="derived:task_state_label",
                    )
                if not event_locked_entries and derived_event_table is not None and len(derived_event_table) > 0:
                    event_locked_result = run_event_locked_export(
                        payload=payload,
                        config=base_config,
                        dataset_id=self.dataset.ds_id,
                        source_path=Path(stage_events_path) if stage_events_path else None,
                        event_table=derived_event_table,
                        subject_id=sub_id or "",
                        session_id=ses_id or "",
                        run_id=run_id or "",
                        out_prefix=target_dir / "event_locked",
                    )
                    entry = dict(event_locked_result.manifest_entry)
                    entry["derived_from"] = "task_state_label_segments"
                    event_locked_entries.append(entry)

                if event_locked_entries:
                    if len(event_locked_entries) == 1:
                        manifest_extra["event_locked"] = event_locked_entries[0]
                    else:
                        manifest_extra["event_locked"] = {
                            "n_exports": len(event_locked_entries),
                            "entries": event_locked_entries,
                        }
            except Exception as _el_exc:
                logger.debug("event_locked export skipped: %s", _el_exc)
        block_native_cfg = {}
        if isinstance(config, Mapping):
            block_native_root = config.get("block_native", {})
            if isinstance(block_native_root, Mapping):
                block_native_datasets = block_native_root.get("datasets", {})
                if isinstance(block_native_datasets, Mapping):
                    block_native_cfg = block_native_datasets.get(self.dataset.ds_id, {})
                    if not isinstance(block_native_cfg, Mapping):
                        block_native_cfg = {}
        # M6 — Block-native payload injection (additive, no-op when disabled)
        # Must run after payload is built and before manifest/H5 write so that
        # block-native contract fields and summary sections stay in sync.
        if stage_events_path or block_native_cfg.get("enabled", False):
            try:
                import pandas as _pd
                from .block_native_config import block_native_dataset_config_from_config
                from .block_native_export import inject_block_native_into_payload
                from .event_locked_runner import _resolve_sampling_cfg, _resolve_stage_map
                _bn_cfg = block_native_dataset_config_from_config(
                    config if isinstance(config, Mapping) else {},
                    self.dataset.ds_id,
                )
                _events_df_bn = None
                _block_native_derived_from = None
                if stage_events_path:
                    _events_df_bn = _pd.read_csv(
                        stage_events_path,
                        sep="\t" if str(stage_events_path).endswith(".tsv") else ",",
                    )
                if _bn_cfg.enabled and _bn_cfg.source.use_derived_task_state_segments:
                    _task_state_series_bn = payload.labels.get("task_state_label") if isinstance(payload.labels, Mapping) else None
                    if (
                        _task_state_series_bn is not None
                        and payload.window_start is not None
                        and payload.window_end is not None
                    ):
                        _derived_event_table_bn = build_label_segment_event_table(
                            labels=_task_state_series_bn,
                            window_start=np.asarray(payload.window_start, dtype=np.float64),
                            window_end=np.asarray(payload.window_end, dtype=np.float64),
                            source_path=str(stage_events_path or "derived:task_state_label"),
                            source_label="derived:task_state_label",
                        )
                        if len(_derived_event_table_bn) > 0:
                            _events_df_bn = _pd.DataFrame(
                                {
                                    _bn_cfg.source.onset_column: np.asarray(_derived_event_table_bn.onset_sec, dtype=np.float64),
                                    _bn_cfg.source.duration_column: (
                                        np.asarray(_derived_event_table_bn.duration_sec, dtype=np.float64)
                                        if _derived_event_table_bn.duration_sec is not None
                                        else np.zeros((_derived_event_table_bn.n,), dtype=np.float64)
                                    ),
                                    _bn_cfg.source.label_column: (
                                        np.asarray(_derived_event_table_bn.event_type, dtype=object)
                                        if _derived_event_table_bn.event_type is not None
                                        else np.full((_derived_event_table_bn.n,), "", dtype=object)
                                    ),
                                }
                            )
                            _events_df_bn.attrs["source_path"] = str(_derived_event_table_bn.source_path)
                            _block_native_derived_from = "task_state_label_segments"
                if _events_df_bn is None:
                    raise ValueError("block_native requires either stage events or derived task-state segments.")
                _bn_manifest = inject_block_native_into_payload(
                    payload,
                    events_df=_events_df_bn,
                    config=config if isinstance(config, Mapping) else {},
                    dataset_id=self.dataset.ds_id,
                    subject_id=sub_id or "",
                    session_id=ses_id or "",
                    run_id=run_id or "",
                    sampling_cfg=_resolve_sampling_cfg(
                        config if isinstance(config, Mapping) else {}, self.dataset.ds_id
                    ),
                    stage_map=_resolve_stage_map(
                        config if isinstance(config, Mapping) else {}, self.dataset.ds_id
                    ),
                    output_dir=target_dir,
                )
                if isinstance(_bn_manifest, Mapping):
                    if _bn_manifest.get("n_blocks", 0) or _bn_manifest.get("status") in {"no_blocks", "error"}:
                        manifest_extra["block_native"] = dict(_bn_manifest)
                        if _block_native_derived_from:
                            manifest_extra["block_native"]["derived_from"] = _block_native_derived_from

                    if _bn_manifest.get("n_blocks", 0):
                        manifest_extra["block_native_coverage"] = {
                            "raw_stage_label_counts": (
                                dict(stage_mapping_qc_entry.get("raw_event_label_counts", {}))
                                if isinstance(stage_mapping_qc_entry, Mapping)
                                else {}
                            ),
                            "recovered_blocks_by_stage": dict(_bn_manifest.get("block_counts_by_stage", {})),
                            "block_native_windows_by_stage": dict(_bn_manifest.get("window_counts_by_stage", {})),
                            "block_counts_by_frequency_hz": dict(_bn_manifest.get("block_counts_by_frequency_hz", {})),
                        }

                        qc_entry = _bn_manifest.get("qc_entry", {})
                        if isinstance(qc_entry, Mapping):
                            qc_entry = dict(qc_entry)
                            qc_entry.update(
                                {
                                    "subject": sub_id,
                                    "session": ses_id,
                                    "run": run_id,
                                    "acq": acq_id,
                                    "task": task,
                                    "condition": condition,
                                    "dataset_id": self.dataset.ds_id,
                                }
                            )
                            if isinstance(stage_mapping_qc_entry, Mapping):
                                raw_label_counts = stage_mapping_qc_entry.get("raw_event_label_counts", {})
                                cleaned_counts: Dict[str, int] = {}
                                normalized_counts: Dict[str, int] = {}
                                if isinstance(raw_label_counts, Mapping):
                                    for raw_label, raw_count in raw_label_counts.items():
                                        try:
                                            count_int = int(raw_count)
                                        except Exception:
                                            continue
                                        cleaned = re.sub(
                                            r"\s+",
                                            " ",
                                            "".join(
                                                ch if str(ch).isprintable() else " "
                                                for ch in str(raw_label)
                                            ),
                                        ).strip()
                                        normalized = cleaned.lower()
                                        if cleaned:
                                            cleaned_counts[cleaned] = cleaned_counts.get(cleaned, 0) + count_int
                                        if normalized:
                                            normalized_counts[normalized] = normalized_counts.get(normalized, 0) + count_int
                                qc_entry["label_cleaning"] = {
                                    "raw_event_label_counts": dict(raw_label_counts) if isinstance(raw_label_counts, Mapping) else {},
                                    "cleaned_event_label_counts": {
                                        str(k): int(v) for k, v in sorted(cleaned_counts.items(), key=lambda kv: kv[0])
                                    },
                                    "normalized_event_label_counts": {
                                        str(k): int(v) for k, v in sorted(normalized_counts.items(), key=lambda kv: kv[0])
                                    },
                                    "mapping_status_counts": (
                                        dict(stage_mapping_qc_entry.get("mapping_mode_counts", {}))
                                        if isinstance(stage_mapping_qc_entry.get("mapping_mode_counts", {}), Mapping)
                                        else {}
                                    ),
                                }
                            manifest_extra["block_native_qc"] = qc_entry
                            self.dataset._record_block_native_qc_entry(qc_entry)
            except Exception as _bn_exc:
                logger.debug("block_native injection skipped: %s", _bn_exc)

        manifest = json_writer.build_manifest(dataset_label, payload, primary_jac_diagnostics, manifest_extra)

        write_summary_manifest_and_h5(
            target_dir=target_dir,
            dataset_label=dataset_label,
            manifest=manifest,
            payload=payload,
            jacobian_diagnostics=primary_jac_diagnostics,
            sub_id=sub_id,
            ses_id=ses_id,
            condition=condition,
            task=task,
            run_id=run_id,
            acq_id=acq_id,
            build_dir_suffix=self._build_dir_suffix,
        )
    @staticmethod
    def _apply_fd_censoring(
        sub_frame: pd.DataFrame,
        fd_thresh: float = 0.5,
        pad: int = 1,
        *,
        require_fd: bool = False,
        context_label: Optional[str] = None,
    ) -> pd.DataFrame:
        """Drop epochs with framewise displacement above threshold (and pad neighbours)."""
        return apply_fd_censoring(
            sub_frame=sub_frame,
            fd_thresh=fd_thresh,
            pad=pad,
            require_fd=require_fd,
            context_label=context_label,
            logger=logger,
        )

    @staticmethod
    def _build_dir_suffix(
        ses_id: Optional[str],
        condition: Optional[str],
        task: Optional[str],
        run_id: Optional[str] = None,
        acq_id: Optional[str] = None,
    ) -> Optional[str]:
        """Build directory/filename suffix from session, condition, and task.

        Priority:
        1. If condition+task: "condition_task" (e.g., "awake_audio")
        2. If only condition: "condition" (e.g., "awake")
        3. If only session: "ses_id" (e.g., "ses-01")
        4. If only task (no condition): "task" (e.g., "rest")
        5. None if nothing available
        """
        return build_dir_suffix(ses_id, condition, task, run_id, acq_id)

    def _resolve_bold_path_for_subframe(
        self,
        sub_frame: pd.DataFrame,
        raw_task: Optional[str],
        condition: Optional[str],
        session: Optional[str],
        run_id: Optional[str],
        acq_id: Optional[str],
    ) -> Optional[Path]:
        """Resolve BOLD NIfTI path for the current grouped sub_frame."""
        return resolve_bold_path_for_subframe(
            sub_frame=sub_frame,
            raw_task=raw_task,
            condition=condition,
            session=session,
            run_id=run_id,
            acq_id=acq_id,
            dataset_root=self._dataset_root(),
            index_df=self.index_df,
            lookup_rel_paths_by_file_value=self.dataset._lookup_rel_paths_by_file_value,
        )

    def _resolve_confounds_path_for_bold(self, bold_path: Path) -> Optional[Path]:
        """Resolve BIDS confounds TSV corresponding to a BOLD file."""
        cache_key = str(bold_path)
        if cache_key in self._confounds_path_cache:
            return self._confounds_path_cache[cache_key]

        dataset_root = self._dataset_root()
        bold_name = bold_path.name
        if not bold_name.endswith((".nii", ".nii.gz")):
            self._confounds_path_cache[cache_key] = None
            return None

        bold_base = bold_name.replace(".nii.gz", "").replace(".nii", "")
        if bold_base.endswith("_bold"):
            conf_name = f"{bold_base[:-5]}_desc-confounds_timeseries.tsv"
        else:
            conf_name = f"{bold_base}_desc-confounds_timeseries.tsv"

        cfg = self.ctx.config if isinstance(self.ctx.config, Mapping) else {}
        preprocess_cfg = cfg.get("preprocess", {}) if isinstance(cfg, Mapping) else {}
        fmri_cfg = preprocess_cfg.get("fmri", {}) if isinstance(preprocess_cfg, Mapping) else {}
        if not isinstance(fmri_cfg, Mapping):
            fmri_cfg = {}
        ds_overrides = fmri_cfg.get("datasets", {}) if isinstance(fmri_cfg.get("datasets", {}), Mapping) else {}
        ds_cfg = ds_overrides.get(self.dataset.ds_id, {}) if isinstance(ds_overrides, Mapping) else {}
        if not isinstance(ds_cfg, Mapping):
            ds_cfg = {}
        merged_fmri_cfg: Dict[str, Any] = {k: v for k, v in fmri_cfg.items() if k != "datasets"}
        merged_fmri_cfg.update(dict(ds_cfg))
        conf_cfg = merged_fmri_cfg.get("confounds", {}) if isinstance(merged_fmri_cfg.get("confounds", {}), Mapping) else {}

        candidates: List[Path] = [bold_path.parent / conf_name]

        roots: List[Path] = []

        def _append_root(v: Any) -> None:
            """Internal helper: append root."""
            if not isinstance(v, (str, Path)):
                return
            p = Path(str(v))
            if not p.is_absolute():
                p = (dataset_root / p).resolve()
            roots.append(p)

        _append_root(conf_cfg.get("confounds_root"))
        _append_root(conf_cfg.get("derivatives_dir"))
        _append_root(merged_fmri_cfg.get("confounds_root"))
        _append_root(merged_fmri_cfg.get("derivatives_dir"))
        roots.append(dataset_root / "derivatives")

        try:
            rel_parent = bold_path.parent.relative_to(dataset_root)
            for root in roots:
                candidates.append(root / rel_parent / conf_name)
                candidates.append(root / "fmriprep" / rel_parent / conf_name)
                candidates.append(root / self.dataset.ds_id / rel_parent / conf_name)
                candidates.append(root / self.dataset.ds_id / "fmriprep" / rel_parent / conf_name)
        except Exception:
            pass

        for c in candidates:
            if c.exists():
                self._confounds_path_cache[cache_key] = c
                return c

        token = conf_name.replace("_desc-confounds_timeseries.tsv", "")
        for root in roots:
            if root.exists():
                for c in root.rglob("*desc-confounds_timeseries.tsv"):
                    name = c.name
                    if token in name:
                        self._confounds_path_cache[cache_key] = c
                        return c
        self._confounds_path_cache[cache_key] = None
        return None

    def _merge_fd_from_confounds(
        self,
        sub_frame: pd.DataFrame,
        raw_task: Optional[str],
        condition: Optional[str],
        session: Optional[str],
        run_id: Optional[str],
        acq_id: Optional[str],
    ) -> pd.DataFrame:
        """Populate framewise_displacement per epoch from BIDS confounds TSV."""
        if sub_frame.empty or "framewise_displacement" in sub_frame.columns:
            return sub_frame

        bold_path = self._resolve_bold_path_for_subframe(
            sub_frame=sub_frame,
            raw_task=raw_task,
            condition=condition,
            session=session,
            run_id=run_id,
            acq_id=acq_id,
        )
        if bold_path is None:
            logger.warning("FD merge skipped: could not resolve BOLD path for current fMRI segment")
            return sub_frame

        conf_path = self._resolve_confounds_path_for_bold(bold_path)
        if conf_path is None:
            logger.warning("FD merge skipped: no confounds TSV found for %s", bold_path.name)
            return sub_frame

        try:
            conf_df = pd.read_csv(conf_path, sep="\t")
        except Exception:
            logger.warning("FD merge skipped: failed reading confounds TSV %s", conf_path)
            return sub_frame
        if conf_df.empty:
            logger.warning("FD merge skipped: empty confounds TSV %s", conf_path)
            return sub_frame

        fd_col = None
        for col in conf_df.columns:
            if str(col).strip().lower() == "framewise_displacement":
                fd_col = col
                break
        if fd_col is None:
            logger.warning("FD merge skipped: framewise_displacement missing in %s", conf_path)
            return sub_frame

        fd = pd.to_numeric(conf_df[fd_col], errors="coerce").to_numpy(dtype=float)
        if fd.size == 0:
            logger.warning("FD merge skipped: no FD values in %s", conf_path)
            return sub_frame

        sfreq = np.nan
        if "fmri_sfreq" in sub_frame.columns:
            try:
                sfreq = float(pd.to_numeric(sub_frame["fmri_sfreq"], errors="coerce").dropna().iloc[0])
            except Exception:
                sfreq = np.nan
        if not np.isfinite(sfreq) or sfreq <= 0:
            try:
                import nibabel as nib  # type: ignore

                bold_img = nib.load(str(bold_path))
                zooms = bold_img.header.get_zooms()
                tr = float(zooms[3]) if len(zooms) > 3 else np.nan
                if np.isfinite(tr) and tr > 0:
                    sfreq = 1.0 / tr
            except Exception:
                sfreq = np.nan
        if not np.isfinite(sfreq) or sfreq <= 0:
            logger.warning("FD merge skipped: could not infer fMRI sfreq for %s", bold_path.name)
            return sub_frame

        tr = 1.0 / float(sfreq)
        frame_t = np.arange(fd.size, dtype=float) * tr

        out = sub_frame.copy()
        if "t_start" in out.columns and "t_end" in out.columns:
            t_start = pd.to_numeric(out["t_start"], errors="coerce").to_numpy(dtype=float)
            t_end = pd.to_numeric(out["t_end"], errors="coerce").to_numpy(dtype=float)
        else:
            epoch_ids = (
                pd.to_numeric(out["epoch_id"], errors="coerce").fillna(-1).astype(int).to_numpy()
                if "epoch_id" in out.columns
                else np.arange(len(out), dtype=int)
            )
            if "fmri_step_sec" in out.columns:
                step_sec = float(pd.to_numeric(out["fmri_step_sec"], errors="coerce").dropna().iloc[0])
            else:
                step_sec = tr
            if "fmri_window_sec" in out.columns:
                window_sec = float(pd.to_numeric(out["fmri_window_sec"], errors="coerce").dropna().iloc[0])
            else:
                window_sec = step_sec
            t_start = epoch_ids.astype(float) * float(step_sec)
            t_end = t_start + float(window_sec)

        fd_epoch = np.full((len(out),), np.nan, dtype=float)
        for i, (s, e) in enumerate(zip(t_start, t_end)):
            if not (np.isfinite(s) and np.isfinite(e) and e > s):
                continue
            mask = (frame_t >= s) & (frame_t < e)
            vals = fd[mask]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            fd_epoch[i] = float(np.nanmax(vals))

        out["framewise_displacement"] = fd_epoch.astype(np.float32)
        logger.info(
            "Merged framewise_displacement from confounds for %s (%d/%d finite epochs)",
            bold_path.name,
            int(np.isfinite(fd_epoch).sum()),
            int(len(fd_epoch)),
        )
        return out

    def _infer_stage_from_bids_events(
        self, sub_frame: pd.DataFrame
    ) -> tuple[Optional[np.ndarray], Optional[str], Optional[str], Optional[str]]:
        """Infer per-epoch sleep stage codes from the BIDS *_events.tsv file.

        Intended for sleep datasets such as ds005555 (BOAS) where stage labels are
        provided as columns (e.g., stage_hum / stage_ai) in the events TSV.
        """
        return infer_stage_from_bids_events(
            sub_frame=sub_frame,
            index_df=self.index_df,
            dataset_root=self._dataset_root(),
            lookup_rel_paths_by_file_value=self.dataset._lookup_rel_paths_by_file_value,
            ctx_config=self.ctx.config if isinstance(self.ctx.config, Mapping) else {},
            mnps_cfg=self.ctx.mnps_cfg if isinstance(self.ctx.mnps_cfg, Mapping) else {},
            dataset_id=self.dataset.ds_id,
        )

    def _build_bids_event_stage_provenance(
        self,
        *,
        sub_frame: pd.DataFrame,
        stage_for_windows: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Build event provenance + stage mapping QC from BIDS events TSV."""
        result = build_bids_event_stage_provenance(
            sub_frame=sub_frame,
            stage_for_windows=stage_for_windows,
            index_df=self.index_df,
            dataset_root=self._dataset_root(),
            lookup_rel_paths_by_file_value=self.dataset._lookup_rel_paths_by_file_value,
            ctx_config=self.ctx.config if isinstance(self.ctx.config, Mapping) else {},
            mnps_cfg=self.ctx.mnps_cfg if isinstance(self.ctx.mnps_cfg, Mapping) else {},
            dataset_id=self.dataset.ds_id,
        )
        return {
            "stage_inferred": result.stage_inferred,
            "stage_source": result.stage_source,
            "stage_column": result.stage_column,
            "events_path": result.events_path,
            "event_table_columns": result.event_table_columns,
            "legacy_events": result.legacy_events,
            "stage_mapping_qc": result.stage_mapping_qc,
        }

    def _prefer_events_stage_in_summary(self) -> bool:
        """Return whether summarize should override feature CSV stage from BIDS events."""
        epoching = self.ctx.config.get("epoching", {}) if isinstance(self.ctx.config, Mapping) else {}
        if not isinstance(epoching, Mapping):
            return False
        ds_map = epoching.get("datasets", {})
        if not isinstance(ds_map, Mapping):
            return False
        ds_cfg = ds_map.get(self.dataset.ds_id, {})
        if not isinstance(ds_cfg, Mapping):
            return False
        sampling_cfg = ds_cfg.get("sampling", {})
        if not isinstance(sampling_cfg, Mapping):
            return False
        return bool(sampling_cfg.get("prefer_events_stage_in_summary", False))

    @staticmethod
    def _estimate_coverage_seconds(sub_frame: pd.DataFrame, dt_fallback: float) -> tuple[float, str]:
        """Estimate coverage from timestamps when available; otherwise fallback to len*dt."""
        return estimate_coverage_seconds(sub_frame, dt_fallback)

    def _load_regional_fmri(
        self,
        sub_id: str,
        dataset_label: str,
        config: Mapping[str, Any],
        sub_frame: pd.DataFrame,
        raw_task: Optional[str],
        condition: Optional[str],
        session: Optional[str],
        run_id: Optional[str],
    ) -> tuple[Optional[np.ndarray], Optional[List[str]], Optional[float]]:
        """Load regional fMRI signals if available.

        Uses the file from sub_frame to ensure we load the correct BOLD file
        for this specific (subject, condition, task) combination.
        """
        return load_regional_fmri_signals(
            sub_id=sub_id,
            dataset_label=dataset_label,
            config=config,
            sub_frame=sub_frame,
            raw_task=raw_task,
            condition=condition,
            session=session,
            run_id=run_id,
            dataset_root=self._dataset_root(),
            index_df=self.index_df,
            lookup_rel_paths_by_file_value=self.dataset._lookup_rel_paths_by_file_value,
            preprocess_fmri=preprocess.preprocess_fmri,
            logger=logger,
        )

    @staticmethod
    def _extract_time_bounds(sub_frame: pd.DataFrame, time: np.ndarray, window_sec: float) -> tuple[np.ndarray, np.ndarray]:
        """Return per-window start/end seconds, using t_start/t_end if available."""
        return extract_time_bounds(sub_frame, time, window_sec)

    def _map_events_to_labels(
        self,
        config: Mapping[str, Any],
        time: np.ndarray,
        window_start: np.ndarray,
        window_end: np.ndarray,
        events: Mapping[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Map event timestamps to MNPS window-aligned binary labels (opt-in)."""
        return map_events_to_labels(
            config=config,
            time=time,
            window_start=window_start,
            window_end=window_end,
            events=events,
            dataset_id=self.dataset.ds_id,
        )

    def _build_within_run_labels(
        self,
        *,
        config: Mapping[str, Any],
        sub_id: str,
        ses_id: Optional[str],
        task: Optional[str],
        raw_task: Optional[str],
        run_id: Optional[str],
        acq_id: Optional[str],
        tr_sec: Optional[float],
        time: np.ndarray,
        window_start: np.ndarray,
        window_end: np.ndarray,
        sub_frame: pd.DataFrame,
    ) -> Any:
        """Build optional within-run labels aligned to the MNPS time axis."""
        return build_within_run_labels(
            config=config,
            dataset_id=self.dataset.ds_id,
            dataset_root=self._dataset_root(),
            sub_id=sub_id,
            ses_id=ses_id,
            task=task,
            raw_task=raw_task,
            run_id=run_id,
            acq_id=acq_id,
            tr_sec=tr_sec,
            time=time,
            window_start=window_start,
            window_end=window_end,
            sub_frame=sub_frame,
        )

    @staticmethod
    def _resolve_run_tr_seconds(sub_frame: pd.DataFrame) -> Optional[float]:
        """Best-effort TR resolution for within-run label sources using TR indices."""
        if "fmri_sfreq" in sub_frame.columns:
            try:
                sfreq = pd.to_numeric(sub_frame["fmri_sfreq"], errors="coerce").to_numpy(dtype=float)
                finite = sfreq[np.isfinite(sfreq) & (sfreq > 0)]
                if finite.size:
                    return float(1.0 / finite[0])
            except Exception:
                pass
        return None

    def _write_qc_files(
        self,
        target_dir: Path,
        dataset_label: str,
        sub_id: str,
        ses_id: Optional[str],
        sub_frame: pd.DataFrame,
        dt: float,
        ensemble_summary: Optional[Dict[str, Any]],
        robust_summary: Optional[Dict[str, Any]],
        dist_summary: Optional[Dict[str, Any]],
        baseline_comparisons: Optional[Dict[str, Any]],
        tau_summary: Optional[Dict[str, Any]],
        tier2_jacobian: Optional[Dict[str, Any]],
        tier2_emmi: Optional[Dict[str, Any]],
        null_sanity_tests: Optional[Dict[str, Any]],
        entropy_qc: Optional[Dict[str, Any]],
        geometry_contract: Optional[Dict[str, Any]],
        mnps_mnj_sanity: Optional[Dict[str, Any]],
    ) -> None:
        """Write QC-related JSON files."""
        write_qc_files(
            target_dir=target_dir,
            dataset_label=dataset_label,
            ds_path=self.ds_path,
            sub_id=sub_id,
            ses_id=ses_id,
            sub_frame=sub_frame,
            dt=dt,
            ensemble_summary=ensemble_summary,
            robust_summary=robust_summary,
            dist_summary=dist_summary,
            baseline_comparisons=baseline_comparisons,
            tau_summary=tau_summary,
            tier2_jacobian=tier2_jacobian,
            tier2_emmi=tier2_emmi,
            null_sanity_tests=null_sanity_tests,
            entropy_qc=entropy_qc,
            geometry_contract=geometry_contract,
            mnps_mnj_sanity=mnps_mnj_sanity,
        )
