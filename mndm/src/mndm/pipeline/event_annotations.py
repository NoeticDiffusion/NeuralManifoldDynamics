"""Generic event annotation table for NeuralManifoldDynamics.

Defines a columnar ``EventTable`` dataclass and helpers to load, validate, and
serialize event annotations (e.g. sleep spindles, slow oscillations, or any
other time-stamped event) to/from CSV/TSV and HDF5-compatible dicts.

Design contract
---------------
* No hard-coded event types. ``event_type`` is a free string column.
* Absence of event annotations must never break a normal MNDM run.
* Source provenance is always recorded (``source`` column).
* HDF5 layout: columnar datasets under ``/events/`` with a
  ``_schema_version`` attribute on the group.

Typical CSV/TSV columns (all optional except ``onset_sec``)::

    onset_sec   duration_sec  offset_sec  event_type     confidence  source
    3.4         1.2           4.6         sleep_spindle  0.87        detector:sigma_v1
    ...
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "1.0"

# Columns that are always float arrays when present.
_FLOAT_COLS = frozenset(
    {
        "onset_sec",
        "duration_sec",
        "offset_sec",
        "peak_sec",
        "confidence",
        "frequency_hz",
        "amplitude",
        "sigma_power",
        "sigma_power_z_n2",
        "so_onset_sec",
        "so_upstate_sec",
        "so_amplitude",
        "so_phase_at_spindle_peak",
        "so_spindle_latency_sec",
        "so_spindle_coupling_score",
        "so_partner_missing",
    }
)

# Columns stored as variable-length UTF-8 strings.
_STR_COLS = frozenset(
    {
        "event_type",
        "source",
        "channel",
        "stage",
        "metadata_json",
    }
)


def _sec_to_ms(value_sec: float) -> float:
    """Convert seconds to milliseconds, preserving NaN."""
    return float(value_sec) * 1000.0 if np.isfinite(float(value_sec)) else np.nan


@dataclass
class EventTable:
    """Columnar representation of time-stamped event annotations.

    All arrays share the same length ``n``. Arrays that are absent are ``None``.

    Required
    --------
    onset_sec : float64 array ``[n]``
        Event onset time in seconds (aligned to the same clock as MNPS windows).

    Optional float columns
    ----------------------
    duration_sec, offset_sec, peak_sec, confidence, frequency_hz,
    amplitude, sigma_power, sigma_power_z_n2, so_onset_sec, so_upstate_sec,
    so_amplitude, so_phase_at_spindle_peak, so_spindle_latency_sec,
    so_spindle_coupling_score, so_partner_missing

    Optional string columns
    -----------------------
    event_type, source, channel, stage, metadata_json

    Provenance
    ----------
    source_path : str
        Path to the file the table was loaded from (or ``"synthetic"``).
    n_events_loaded : int
        How many rows were in the raw file before any filtering.
    """

    onset_sec: np.ndarray

    duration_sec: Optional[np.ndarray] = None
    offset_sec: Optional[np.ndarray] = None
    peak_sec: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    frequency_hz: Optional[np.ndarray] = None
    amplitude: Optional[np.ndarray] = None
    sigma_power: Optional[np.ndarray] = None
    sigma_power_z_n2: Optional[np.ndarray] = None
    so_onset_sec: Optional[np.ndarray] = None
    so_upstate_sec: Optional[np.ndarray] = None
    so_amplitude: Optional[np.ndarray] = None
    so_phase_at_spindle_peak: Optional[np.ndarray] = None
    so_spindle_latency_sec: Optional[np.ndarray] = None
    so_spindle_coupling_score: Optional[np.ndarray] = None
    so_partner_missing: Optional[np.ndarray] = None

    event_type: Optional[np.ndarray] = None
    source: Optional[np.ndarray] = None
    channel: Optional[np.ndarray] = None
    stage: Optional[np.ndarray] = None
    metadata_json: Optional[np.ndarray] = None

    source_path: str = "unknown"
    n_events_loaded: int = 0

    def __len__(self) -> int:
        return int(self.onset_sec.shape[0])

    @property
    def n(self) -> int:
        return len(self)

    def is_empty(self) -> bool:
        return self.n == 0


def make_empty_event_table(source_path: str = "empty") -> EventTable:
    """Return a zero-row EventTable."""
    return EventTable(onset_sec=np.empty(0, dtype=np.float64), source_path=source_path)


def event_table_from_stage_block_intervals(
    intervals: Sequence[Any],
    *,
    event_kind: str = "end",
    event_type: str = "stage_block_end",
    source: str = "derived:stage_blocking",
    source_path: str = "synthetic:stage_blocking",
) -> EventTable:
    """Build a synthetic EventTable from inferred stage-block intervals.

    Parameters
    ----------
    intervals:
        Sequence of objects exposing at least ``start_sec``, ``end_sec``,
        ``stage_code``, ``block_id``, ``block_parameter``, ``source_event_idx``,
        and optionally ``support_event_indices``.
    event_kind:
        Which point to emit for each interval. Supported: ``"start"``,
        ``"end"``. The emitted event is always a point-event with
        ``duration_sec = 0``.
    event_type:
        Value written into the EventTable ``event_type`` column.
    source:
        Value written into the EventTable ``source`` column.
    source_path:
        Provenance path for the synthetic table. For derived block events this
        should typically be the raw BIDS ``*_events.tsv`` source path.
    """
    intervals = list(intervals or [])
    if not intervals:
        return make_empty_event_table(source_path)

    if event_kind not in {"start", "end"}:
        raise ValueError(f"Unsupported event_kind '{event_kind}'. Expected 'start' or 'end'.")

    onset_vals: List[float] = []
    freq_vals: List[float] = []
    stage_vals: List[str] = []
    metadata_vals: List[str] = []

    for interval in intervals:
        start_sec = float(getattr(interval, "start_sec"))
        end_sec = float(getattr(interval, "end_sec"))
        onset_sec = start_sec if event_kind == "start" else end_sec
        block_id = int(getattr(interval, "block_id", -1))
        stage_code = int(getattr(interval, "stage_code", -1))
        source_event_idx = int(getattr(interval, "source_event_idx", -1))
        block_parameter = float(getattr(interval, "block_parameter", np.nan))
        support_event_indices = tuple(int(v) for v in getattr(interval, "support_event_indices", ()) or ())
        derived_from = str(getattr(interval, "derived_from", "stage_blocking") or "stage_blocking")
        end_reason = str(getattr(interval, "end_reason", "unknown") or "unknown")
        membership_mode = str(getattr(interval, "membership_mode", "") or "")
        bridge_tail_sec = float(getattr(interval, "bridge_tail_sec", np.nan))
        bridge_tail_cap_sec = float(getattr(interval, "bridge_tail_cap_sec", np.nan))
        is_inferred = bool(getattr(interval, "is_inferred", True))
        start_ms = _sec_to_ms(start_sec)
        end_ms = _sec_to_ms(end_sec)
        bridge_tail_ms = _sec_to_ms(bridge_tail_sec)
        bridge_tail_cap_ms = _sec_to_ms(bridge_tail_cap_sec)

        onset_vals.append(onset_sec)
        freq_vals.append(block_parameter if np.isfinite(block_parameter) else np.nan)
        stage_vals.append(str(stage_code))
        metadata_vals.append(
            json.dumps(
                {
                    "derived_event_kind": str(event_kind),
                    "derived_from": derived_from,
                    "is_inferred": is_inferred,
                    "end_reason": end_reason,
                    "membership_mode": membership_mode,
                    "bridge_tail_sec": (bridge_tail_sec if np.isfinite(bridge_tail_sec) else None),
                    "bridge_tail_cap_sec": (bridge_tail_cap_sec if np.isfinite(bridge_tail_cap_sec) else None),
                    "bridge_tail_ms": (bridge_tail_ms if np.isfinite(bridge_tail_ms) else None),
                    "bridge_tail_cap_ms": (bridge_tail_cap_ms if np.isfinite(bridge_tail_cap_ms) else None),
                    "block_id": block_id,
                    "block_start_sec": start_sec,
                    "block_end_sec": end_sec,
                    "block_start_ms": (start_ms if np.isfinite(start_ms) else None),
                    "block_end_ms": (end_ms if np.isfinite(end_ms) else None),
                    "block_duration_ms": ((end_ms - start_ms) if np.isfinite(start_ms) and np.isfinite(end_ms) else None),
                    "stage_code": stage_code,
                    "block_parameter": (block_parameter if np.isfinite(block_parameter) else None),
                    "source_event_idx": source_event_idx,
                    "support_event_indices": list(support_event_indices),
                },
                ensure_ascii=False,
            )
        )

    n = len(onset_vals)
    onset_arr = np.asarray(onset_vals, dtype=np.float64)
    zeros = np.zeros((n,), dtype=np.float64)
    return EventTable(
        onset_sec=onset_arr,
        duration_sec=zeros.copy(),
        offset_sec=onset_arr.copy(),
        frequency_hz=np.asarray(freq_vals, dtype=np.float64),
        event_type=np.asarray([str(event_type)] * n, dtype=object),
        source=np.asarray([str(source)] * n, dtype=object),
        stage=np.asarray(stage_vals, dtype=object),
        metadata_json=np.asarray(metadata_vals, dtype=object),
        source_path=str(source_path),
        n_events_loaded=n,
    )


def load_event_table_from_csv(
    path: Path,
    *,
    sep: Optional[str] = None,
    onset_col: str = "onset_sec",
    event_type_filter: Optional[str] = None,
    max_duration_sec: Optional[float] = None,
    min_duration_sec: Optional[float] = None,
) -> EventTable:
    """Load an event table from a CSV/TSV file.

    The file must contain at least a column matching ``onset_col``.
    All standard column names are detected automatically regardless of order.
    Unknown columns are stored serialized as ``metadata_json`` if no
    ``metadata_json`` column already exists.

    Parameters
    ----------
    path:
        Path to a ``.csv`` or ``.tsv`` file.
    sep:
        Delimiter. Auto-detected from extension if ``None`` (``.tsv`` → tab,
        ``.csv`` → comma).
    onset_col:
        Name of the onset column in the file (default ``"onset_sec"``).
        If the file uses ``"onset"`` instead, pass ``onset_col="onset"``.
    event_type_filter:
        Keep only rows where ``event_type == event_type_filter``.
    max_duration_sec / min_duration_sec:
        Optional hard duration bounds applied after loading.

    Returns
    -------
    EventTable
    """
    path = Path(path)
    if not path.exists():
        logger.warning("Event annotation file not found: %s", path)
        return make_empty_event_table(str(path))

    if sep is None:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","

    rows: List[Dict[str, str]] = []
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter=sep)
            for row in reader:
                rows.append(dict(row))
    except Exception:
        logger.exception("Failed to read event file: %s", path)
        return make_empty_event_table(str(path))

    n_loaded = len(rows)
    if n_loaded == 0:
        return EventTable(onset_sec=np.empty(0, dtype=np.float64), source_path=str(path), n_events_loaded=0)

    known_cols = _FLOAT_COLS | _STR_COLS | {onset_col}
    all_file_cols = set(rows[0].keys())
    extra_cols = sorted(all_file_cols - known_cols)

    def _to_float_col(key: str) -> Optional[np.ndarray]:
        col_key = key if key in all_file_cols else None
        if key == "onset_sec" and onset_col != "onset_sec" and onset_col in all_file_cols:
            col_key = onset_col
        elif key not in all_file_cols:
            return None
        vals = []
        for r in rows:
            raw = r.get(col_key or key, "")
            try:
                vals.append(float(raw) if raw not in ("", "nan", "NaN", "NA", "N/A") else np.nan)
            except ValueError:
                vals.append(np.nan)
        return np.asarray(vals, dtype=np.float64)

    def _to_str_col(key: str) -> Optional[np.ndarray]:
        if key not in all_file_cols:
            return None
        return np.asarray([str(r.get(key, "")) for r in rows], dtype=object)

    onset = _to_float_col("onset_sec")
    if onset is None or not np.any(np.isfinite(onset)):
        logger.warning("No finite onset values found in %s", path)
        return make_empty_event_table(str(path))

    duration = _to_float_col("duration_sec")
    offset = _to_float_col("offset_sec")
    peak = _to_float_col("peak_sec")
    confidence = _to_float_col("confidence")
    frequency_hz = _to_float_col("frequency_hz")
    amplitude = _to_float_col("amplitude")
    sigma_power = _to_float_col("sigma_power")
    sigma_power_z_n2 = _to_float_col("sigma_power_z_n2")
    so_onset_sec = _to_float_col("so_onset_sec")
    so_upstate_sec = _to_float_col("so_upstate_sec")
    so_amplitude = _to_float_col("so_amplitude")
    so_phase_at_spindle_peak = _to_float_col("so_phase_at_spindle_peak")
    so_spindle_latency_sec = _to_float_col("so_spindle_latency_sec")
    so_spindle_coupling_score = _to_float_col("so_spindle_coupling_score")
    so_partner_missing = _to_float_col("so_partner_missing")

    event_type = _to_str_col("event_type")
    source = _to_str_col("source")
    channel = _to_str_col("channel")
    stage = _to_str_col("stage")
    metadata_json_col = _to_str_col("metadata_json")

    if extra_cols and metadata_json_col is None:
        extra_payloads = []
        for r in rows:
            extra_payloads.append(json.dumps({k: r.get(k, "") for k in extra_cols}, ensure_ascii=False))
        metadata_json_col = np.asarray(extra_payloads, dtype=object)

    keep = np.ones(n_loaded, dtype=bool)

    if event_type_filter is not None and event_type is not None:
        keep &= np.asarray([str(v).strip() == event_type_filter for v in event_type])

    if min_duration_sec is not None and duration is not None:
        valid_dur = np.isfinite(duration)
        keep &= ~valid_dur | (duration >= min_duration_sec)

    if max_duration_sec is not None and duration is not None:
        valid_dur = np.isfinite(duration)
        keep &= ~valid_dur | (duration <= max_duration_sec)

    def _apply_mask(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        return arr[keep] if arr is not None else None

    table = EventTable(
        onset_sec=onset[keep],
        duration_sec=_apply_mask(duration),
        offset_sec=_apply_mask(offset),
        peak_sec=_apply_mask(peak),
        confidence=_apply_mask(confidence),
        frequency_hz=_apply_mask(frequency_hz),
        amplitude=_apply_mask(amplitude),
        sigma_power=_apply_mask(sigma_power),
        sigma_power_z_n2=_apply_mask(sigma_power_z_n2),
        so_onset_sec=_apply_mask(so_onset_sec),
        so_upstate_sec=_apply_mask(so_upstate_sec),
        so_amplitude=_apply_mask(so_amplitude),
        so_phase_at_spindle_peak=_apply_mask(so_phase_at_spindle_peak),
        so_spindle_latency_sec=_apply_mask(so_spindle_latency_sec),
        so_spindle_coupling_score=_apply_mask(so_spindle_coupling_score),
        so_partner_missing=_apply_mask(so_partner_missing),
        event_type=_apply_mask(event_type),
        source=_apply_mask(source),
        channel=_apply_mask(channel),
        stage=_apply_mask(stage),
        metadata_json=_apply_mask(metadata_json_col),
        source_path=str(path),
        n_events_loaded=n_loaded,
    )
    logger.info(
        "Loaded %d/%d events from %s",
        table.n,
        n_loaded,
        path.name,
    )
    return table


def validate_event_table(table: EventTable) -> List[str]:
    """Return a list of validation warnings (empty = OK).

    Does not raise; the caller decides how to handle warnings.
    """
    warnings: List[str] = []
    if table.is_empty():
        warnings.append("EventTable has zero rows.")
        return warnings

    n = table.n
    if not np.any(np.isfinite(table.onset_sec)):
        warnings.append("No finite onset_sec values.")

    if table.duration_sec is not None:
        n_neg = int(np.sum(table.duration_sec[np.isfinite(table.duration_sec)] < 0))
        if n_neg:
            warnings.append(f"{n_neg} events have negative duration_sec.")

    # Onset monotonicity is not required (events may be out-of-order in source).

    if table.source is None:
        warnings.append("No 'source' column — provenance cannot be audited.")

    if table.event_type is None:
        warnings.append("No 'event_type' column — event classification unknown.")

    return warnings


def event_table_to_hdf5_columns(table: EventTable) -> Dict[str, Any]:
    """Convert EventTable to a dict of column arrays suitable for HDF5 writing.

    The resulting dict can be written to HDF5 as columnar datasets under
    an ``/events/`` group. A ``_schema_version`` key is included so readers
    can detect the columnar layout vs. the legacy 1-D index layout.

    Returns an empty dict if ``table`` is empty.
    """
    if table.is_empty():
        return {}

    out: Dict[str, Any] = {"_schema_version": np.bytes_(_SCHEMA_VERSION)}
    out["onset_sec"] = table.onset_sec.astype(np.float64)

    def _add_float(key: str, arr: Optional[np.ndarray]) -> None:
        if arr is not None:
            out[key] = arr.astype(np.float64)

    def _add_str(key: str, arr: Optional[np.ndarray]) -> None:
        if arr is not None:
            out[key] = np.asarray([str(v) for v in arr], dtype=object)

    _add_float("duration_sec", table.duration_sec)
    _add_float("offset_sec", table.offset_sec)
    _add_float("peak_sec", table.peak_sec)
    _add_float("confidence", table.confidence)
    _add_float("frequency_hz", table.frequency_hz)
    _add_float("amplitude", table.amplitude)
    _add_float("sigma_power", table.sigma_power)
    _add_float("sigma_power_z_n2", table.sigma_power_z_n2)
    _add_float("so_onset_sec", table.so_onset_sec)
    _add_float("so_upstate_sec", table.so_upstate_sec)
    _add_float("so_amplitude", table.so_amplitude)
    _add_float("so_phase_at_spindle_peak", table.so_phase_at_spindle_peak)
    _add_float("so_spindle_latency_sec", table.so_spindle_latency_sec)
    _add_float("so_spindle_coupling_score", table.so_spindle_coupling_score)
    _add_float("so_partner_missing", table.so_partner_missing)
    _add_str("event_type", table.event_type)
    _add_str("source", table.source)
    _add_str("channel", table.channel)
    _add_str("stage", table.stage)
    _add_str("metadata_json", table.metadata_json)

    return out


def event_table_manifest_entry(table: EventTable) -> Dict[str, Any]:
    """Return a small JSON-serializable dict for ``run_manifest.json``."""
    if table.is_empty():
        return {"n_events": 0, "source_path": table.source_path}

    event_types: List[str] = []
    if table.event_type is not None:
        event_types = sorted(set(str(v) for v in table.event_type if v))

    sources: List[str] = []
    if table.source is not None:
        sources = sorted(set(str(v) for v in table.source if v))

    dur_stats: Dict[str, Any] = {}
    if table.duration_sec is not None:
        finite = table.duration_sec[np.isfinite(table.duration_sec)]
        if finite.size:
            dur_stats = {
                "duration_mean_sec": float(np.mean(finite)),
                "duration_min_sec": float(np.min(finite)),
                "duration_max_sec": float(np.max(finite)),
            }

    return {
        "n_events": table.n,
        "n_events_loaded": table.n_events_loaded,
        "source_path": table.source_path,
        "event_types": event_types,
        "sources": sources,
        "schema_version": _SCHEMA_VERSION,
        **dur_stats,
    }


def load_event_table_from_bids_events(
    path: Path,
    *,
    event_types: Optional[Sequence[str]] = None,
    trial_type_column: str = "trial_type",
    onset_column: str = "onset",
    duration_column: str = "duration",
    exclude_types: Optional[Sequence[str]] = None,
) -> EventTable:
    """Load an EventTable from a BIDS ``*_events.tsv`` file.

    Reads the BIDS events file directly using the raw ``onset`` / ``trial_type``
    / ``duration`` columns (standard BIDS naming).  Does NOT require a
    ``derived:task_state_label`` column — event onsets are taken straight from
    the file.

    This is the generic, dataset-agnostic path for event-locked analysis in
    BIDS datasets.  Any recording with a companion ``*_events.tsv`` can use it.

    Parameters
    ----------
    path : Path
        Path to the BIDS ``*_events.tsv`` file.
    event_types : sequence of str, optional
        Whitelist of ``trial_type`` values to keep.  If ``None`` or empty, all
        non-empty rows are kept.
    trial_type_column : str
        Name of the column holding event labels (default ``"trial_type"``).
    onset_column : str
        Name of the onset column in seconds (default ``"onset"``).
    duration_column : str
        Name of the duration column in seconds (default ``"duration"``).
    exclude_types : sequence of str, optional
        Blacklist of ``trial_type`` values to drop (applied after whitelist).

    Returns
    -------
    EventTable
        A valid (possibly empty) EventTable with ``event_type`` = ``trial_type``.
        Provenance fields are populated so callers can report why events were
        excluded.
    """
    path = Path(path)
    if not path.exists():
        logger.warning("BIDS events file not found: %s", path)
        return make_empty_event_table(str(path))

    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        df = pd.read_csv(path, sep=sep)
    except Exception:
        logger.exception("Failed to read BIDS events file: %s", path)
        return make_empty_event_table(str(path))

    if df.empty:
        return EventTable(
            onset_sec=np.empty(0, dtype=np.float64),
            source_path=str(path),
            n_events_loaded=0,
        )

    n_loaded = len(df)

    # Report available trial types for diagnostics
    if trial_type_column in df.columns:
        found_types = sorted(df[trial_type_column].dropna().astype(str).unique().tolist())
        logger.debug(
            "BIDS events (%s): %d rows, trial_types=%s",
            path.name,
            n_loaded,
            found_types,
        )
    else:
        logger.warning(
            "BIDS events (%s): missing column '%s'; available=%s",
            path.name,
            trial_type_column,
            sorted(df.columns.tolist()),
        )
        return make_empty_event_table(str(path))

    if onset_column not in df.columns:
        logger.warning(
            "BIDS events (%s): missing onset column '%s'", path.name, onset_column
        )
        return make_empty_event_table(str(path))

    # Build whitelist / blacklist sets
    keep_set: Optional[set] = None
    if event_types:
        keep_set = {str(t).strip() for t in event_types if str(t).strip()}
    drop_set: set = set()
    if exclude_types:
        drop_set = {str(t).strip() for t in exclude_types if str(t).strip()}

    onset_arr = pd.to_numeric(df[onset_column], errors="coerce").to_numpy(dtype=np.float64)
    type_arr = df[trial_type_column].fillna("").astype(str).tolist()

    if duration_column in df.columns:
        dur_arr: Optional[np.ndarray] = pd.to_numeric(
            df[duration_column], errors="coerce"
        ).to_numpy(dtype=np.float64)
    else:
        dur_arr = None

    onset_out: List[float] = []
    type_out: List[str] = []
    dur_out: List[float] = []
    n_excluded_type = 0
    n_excluded_onset = 0

    for idx, (onset, label) in enumerate(zip(onset_arr, type_arr)):
        label = label.strip()
        if not np.isfinite(onset):
            n_excluded_onset += 1
            continue
        if not label or label.lower() in {"nan", "n/a", ""}:
            n_excluded_type += 1
            continue
        if keep_set is not None and label not in keep_set:
            n_excluded_type += 1
            continue
        if label in drop_set:
            n_excluded_type += 1
            continue
        onset_out.append(float(onset))
        type_out.append(label)
        dur_out.append(float(dur_arr[idx]) if dur_arr is not None and np.isfinite(dur_arr[idx]) else float("nan"))

    n_kept = len(onset_out)
    logger.info(
        "BIDS event-lock (%s): %d/%d events kept (excluded_type=%d, excluded_onset=%d)",
        path.name,
        n_kept,
        n_loaded,
        n_excluded_type,
        n_excluded_onset,
    )

    if not onset_out:
        return EventTable(
            onset_sec=np.empty(0, dtype=np.float64),
            event_type=np.empty(0, dtype=object),
            source_path=str(path),
            n_events_loaded=n_loaded,
        )

    return EventTable(
        onset_sec=np.asarray(onset_out, dtype=np.float64),
        event_type=np.asarray(type_out, dtype=object),
        duration_sec=np.asarray(dur_out, dtype=np.float64),
        source_path=str(path),
        source=np.full(n_kept, "bids_events_tsv", dtype=object),
        n_events_loaded=n_loaded,
    )
