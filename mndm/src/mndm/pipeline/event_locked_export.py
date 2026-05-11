"""Event-locked MNPS summary export (M5).

Assembles a flat, auditable analysis table from:

* ``MNPSPayload``      — MNPS tensors for one subject/run
* ``AlignmentResult``  — event-to-window bin assignments
* ``ControlMatchResult`` — matched non-event control window assignments
* subject/run metadata — identifiers for joining back to HDF5 outputs
* export config        — format, optional columns, provenance flags

Output rows
-----------
One row per ``(subject_id, condition, event_id, bin_label, window_id)``.

``condition`` values:

* ``"spindle_event"``   — a window whose bin assignment came from a real event
* ``"matched_control"`` — a control window matched to a real event

Mandatory columns (always present)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``subject_id``, ``session_id``, ``run_id``, ``dataset_id``,
``condition``, ``event_id``, ``window_id``, ``bin_label``,
``window_start_sec``, ``window_end_sec``, ``window_center_sec``,
``rel_time_sec``, ``overlap_sec``, ``overlap_frac``, ``is_event_window``,
``stage``,
``m``, ``d``, ``e``, ``m_dot``, ``d_dot``, ``e_dot``,
``mnps_finite``,
``alignment_reference``, ``alignment_bins_json``,
``control_seed``, ``event_source_path``, ``annotation_source_hash``,
``n_events_input``, ``n_events_aligned``, ``n_events_excluded_transition``,
``match_success_rate``

Optional columns (present when payload tensors are available)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``m_a``, ``m_e``, ``m_o``, ``d_n``, ``d_l``, ``d_s``, ``e_e``, ``e_s``, ``e_m``,
``coords_9d_finite``,
``event_onset_sec``, ``event_duration_sec``, ``event_source``,
``event_type``, ``event_channel``, ``event_confidence``,
``match_rank``, ``match_distance``, ``matched_event_id``

Design contract
---------------
* No statistics. No hypothesis tests. No plotting.
* Never silently drop failed controls or missing tensors — use NaN / flags.
* Provenance columns are mandatory and non-nullable.
* ``coords_9d`` and Jacobian tensors are optional; absence is explicit.
* Output is Parquet-first; CSV is opt-in for debug/inspection.
* Output is deterministic for a fixed input (no hidden RNG).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..schema import MNPSPayload
from .event_alignment import AlignmentResult
from .control_matching import ControlMatchResult
from .event_annotations import EventTable

logger = logging.getLogger(__name__)

# Canonical 9-D subcoordinate names in order
_COORDS_9D_NAMES = ("m_a", "m_e", "m_o", "d_n", "d_l", "d_s", "e_e", "e_s", "e_m")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class ExportConfig:
    """Lightweight config for the event-locked export.

    Parameters
    ----------
    write_parquet:
        Write ``.parquet`` output (requires ``pyarrow``).
    write_csv:
        Write ``.csv`` output alongside parquet (always possible).
    include_coords_9d:
        Include stratified 9-D subcoordinates when present in payload.
    provenance_in_every_row:
        Repeat all provenance columns in every row (enables independent
        per-row traceability; costs some storage). Default True.
    profile_name:
        Named analysis profile (e.g. ``"sleep_spindle_event_locked_v1"``).
        Recorded in every provenance row. ``None`` if not driven by a profile.
    window_length_s:
        Feature-epoch window length in seconds. Recorded for provenance.
    window_step_s:
        Feature-epoch step in seconds. Recorded for provenance.
    bins_json:
        JSON-serialized bin definitions (overrides auto-serialization from
        alignment QC when provided).
    alignment_reference:
        Event reference point (``"onset"``, ``"peak"``, ``"offset"``).
        Overrides the value derived from alignment QC when provided.
    control_seed:
        RNG seed used for control matching. Overrides the value from
        control QC when provided.
    annotation_source_hash:
        SHA-256 hex digest of the annotation source file (e.g. the YASA
        spindle CSV).  Computed automatically from ``event_table.source_path``
        when ``None`` and the path exists; pass an explicit value to override
        or skip computation.  Stored in every provenance row so that the
        exact detector output file can always be identified.
    """

    def __init__(
        self,
        *,
        write_parquet: bool = True,
        write_csv: bool = False,
        include_coords_9d: bool = True,
        provenance_in_every_row: bool = True,
        profile_name: Optional[str] = None,
        window_length_s: Optional[float] = None,
        window_step_s: Optional[float] = None,
        bins_json: Optional[str] = None,
        alignment_reference: Optional[str] = None,
        control_seed: Optional[int] = None,
        annotation_source_hash: Optional[str] = None,
    ) -> None:
        self.write_parquet = write_parquet
        self.write_csv = write_csv
        self.include_coords_9d = include_coords_9d
        self.provenance_in_every_row = provenance_in_every_row
        self.profile_name = profile_name
        self.window_length_s = window_length_s
        self.window_step_s = window_step_s
        self.bins_json = bins_json
        self.alignment_reference = alignment_reference
        self.control_seed = control_seed
        self.annotation_source_hash = annotation_source_hash

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any]) -> "ExportConfig":
        return cls(
            write_parquet=bool(cfg.get("write_parquet", True)),
            write_csv=bool(cfg.get("write_csv", False)),
            include_coords_9d=bool(cfg.get("include_coords_9d", True)),
            provenance_in_every_row=bool(cfg.get("provenance_in_every_row", True)),
        )


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def _sha256_of_file(path: Path, chunk: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of *path* (streaming, 1 MiB chunks)."""
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            while True:
                block = fh.read(chunk)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()
    except OSError as exc:
        logger.warning("Could not hash %s: %s", path, exc)
        return ""


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------

def _mnps_at(payload: MNPSPayload, w_idx: int) -> Dict[str, Any]:
    """Extract MNPS 3D values and derivatives at window index ``w_idx``."""
    out: Dict[str, Any] = {}
    x = payload.x
    xd = payload.x_dot
    if w_idx < len(x):
        row = x[w_idx]
        out["m"] = _f32(row[0]) if len(row) > 0 else np.nan
        out["d"] = _f32(row[1]) if len(row) > 1 else np.nan
        out["e"] = _f32(row[2]) if len(row) > 2 else np.nan
        out["mnps_finite"] = int(np.isfinite([out["m"], out["d"], out["e"]]).all())
    else:
        out.update({"m": np.nan, "d": np.nan, "e": np.nan, "mnps_finite": 0})

    if xd is not None and w_idx < len(xd):
        drow = xd[w_idx]
        out["m_dot"] = _f32(drow[0]) if len(drow) > 0 else np.nan
        out["d_dot"] = _f32(drow[1]) if len(drow) > 1 else np.nan
        out["e_dot"] = _f32(drow[2]) if len(drow) > 2 else np.nan
    else:
        out["m_dot"] = out["d_dot"] = out["e_dot"] = np.nan
    return out


def _coords9d_at(payload: MNPSPayload, w_idx: int, *, include: bool) -> Dict[str, Any]:
    """Extract 9-D subcoordinates at ``w_idx``. Returns NaN if absent/include=False."""
    if not include:
        return {}
    c9 = getattr(payload, "coords_9d", None)
    if c9 is None or not np.size(c9):
        return {name: np.nan for name in _COORDS_9D_NAMES}

    names_attr = getattr(payload, "coords_9d_names", None) or list(_COORDS_9D_NAMES)
    out: Dict[str, Any] = {}
    if w_idx < c9.shape[0]:
        row = c9[w_idx]
        for i, name in enumerate(names_attr):
            out[str(name)] = _f32(row[i]) if i < len(row) else np.nan
        out["coords_9d_finite"] = int(np.isfinite(list(out.values())).all())
    else:
        for name in names_attr:
            out[str(name)] = np.nan
        out["coords_9d_finite"] = 0
    return out


def _timing_at(payload: MNPSPayload, w_idx: int, *, time: np.ndarray) -> Dict[str, Any]:
    """Extract window timing columns."""
    w_center = float(time[w_idx]) if w_idx < len(time) else np.nan
    w_start = np.nan
    w_end = np.nan
    if getattr(payload, "window_start", None) is not None and w_idx < len(payload.window_start):
        w_start = float(payload.window_start[w_idx])
    if getattr(payload, "window_end", None) is not None and w_idx < len(payload.window_end):
        w_end = float(payload.window_end[w_idx])
    return {
        "window_center_sec": w_center,
        "window_start_sec": w_start,
        "window_end_sec": w_end,
    }


def _stage_at(payload: MNPSPayload, w_idx: int) -> int:
    s = getattr(payload, "stage", None)
    if s is not None and w_idx < len(s):
        return int(s[w_idx])
    return -1


def _event_meta(table: Optional[EventTable], ev_idx: int) -> Dict[str, Any]:
    """Extract per-event metadata columns from EventTable."""
    if table is None or table.is_empty() or ev_idx < 0 or ev_idx >= table.n:
        return {
            "event_onset_sec": np.nan,
            "event_duration_sec": np.nan,
            "event_source": "",
            "event_type": "",
            "event_channel": "",
            "event_confidence": np.nan,
        }
    return {
        "event_onset_sec": float(table.onset_sec[ev_idx]),
        "event_duration_sec": float(table.duration_sec[ev_idx]) if table.duration_sec is not None else np.nan,
        "event_source": str(table.source[ev_idx]) if table.source is not None else "",
        "event_type": str(table.event_type[ev_idx]) if table.event_type is not None else "",
        "event_channel": str(table.channel[ev_idx]) if table.channel is not None else "",
        "event_confidence": float(table.confidence[ev_idx]) if table.confidence is not None else np.nan,
    }


def _f32(v: Any) -> float:
    """Cast to Python float, preserving NaN."""
    try:
        return float(v)
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_event_locked_table(
    *,
    payload: MNPSPayload,
    alignment: AlignmentResult,
    controls: ControlMatchResult,
    event_table: Optional[EventTable] = None,
    subject_id: str = "",
    session_id: str = "",
    run_id: str = "",
    dataset_id: str = "",
    config: Optional[ExportConfig] = None,
) -> List[Dict[str, Any]]:
    """Assemble the flat event-locked analysis table.

    Parameters
    ----------
    payload:
        MNPS tensors for one subject/run.
    alignment:
        Output of ``align_events_to_windows()``.
    controls:
        Output of ``build_matched_controls()``.
    event_table:
        Source event annotations (for per-event metadata columns).
        May be ``None`` — event metadata columns will be NaN/empty.
    subject_id, session_id, run_id, dataset_id:
        Identifier strings repeated in every row for join-back capability.
    config:
        Export configuration. Uses defaults if ``None``.

    Returns
    -------
    List[Dict[str, Any]]
        Each element is one flat row. Suitable for ``pd.DataFrame(rows)`` or
        Parquet writing.
    """
    if config is None:
        config = ExportConfig()

    time = payload.time

    # --- Provenance block (repeated in every row) --------------------------
    a_qc = alignment.qc
    c_qc = controls.qc

    # Bins: prefer explicit override from ExportConfig, else derive from alignment QC.
    if config.bins_json is not None:
        bins_serialized = config.bins_json
    elif "_bins_objects" in a_qc:
        bins_serialized = json.dumps(
            [{"label": b.label, "lo": b.lo, "hi": b.hi} for b in a_qc["_bins_objects"]],
            ensure_ascii=False,
        )
    else:
        bins_serialized = json.dumps(a_qc.get("bins_used", []), ensure_ascii=False)

    # Reference: prefer ExportConfig override, else alignment QC.
    eff_reference = config.alignment_reference or str(a_qc.get("reference", "onset"))

    # Control seed: prefer ExportConfig override (which comes from EventLockedProfile),
    # else fall back to control QC.
    raw_seed = config.control_seed if config.control_seed is not None else c_qc.get("seed", -1)
    eff_seed = int(raw_seed) if raw_seed is not None else -1

    # Annotation source: path and SHA-256 hash.
    source_path_str = str(event_table.source_path) if event_table is not None else ""
    if config.annotation_source_hash is not None:
        source_hash = config.annotation_source_hash
    elif event_table is not None and event_table.source_path is not None:
        sp = Path(event_table.source_path)
        source_hash = _sha256_of_file(sp) if sp.exists() else ""
    else:
        source_hash = ""

    provenance: Dict[str, Any] = {
        "subject_id": subject_id,
        "session_id": session_id,
        "run_id": run_id,
        "dataset_id": dataset_id,
        # Profile-level provenance — populated when driven by EventLockedProfile.
        "profile_name": config.profile_name,
        "window_length_s": config.window_length_s,
        "window_step_s": config.window_step_s,
        "alignment_reference": eff_reference,
        "alignment_bins_json": bins_serialized,
        "control_seed": eff_seed,
        "event_source_path": source_path_str,
        "annotation_source_hash": source_hash,
        "n_events_input": int(a_qc.get("n_events_input", 0)),
        "n_events_aligned": int(a_qc.get("n_events_aligned", 0)),
        "n_events_excluded_transition": int(a_qc.get("n_events_excluded_stage_transition", 0)),
        "match_success_rate": float(c_qc.get("match_success_rate", np.nan)),
    }

    rows: List[Dict[str, Any]] = []

    # --- Spindle-event rows -----------------------------------------------
    for arow in alignment.rows:
        w_idx = arow.window_id
        ev_idx = arow.event_id

        row: Dict[str, Any] = {}
        if config.provenance_in_every_row:
            row.update(provenance)

        row["condition"] = "spindle_event"
        row["event_id"] = ev_idx
        row["window_id"] = w_idx
        row["bin_label"] = arow.bin_label
        row["rel_time_sec"] = arow.rel_time_sec
        row["overlap_sec"] = arow.overlap_sec
        row["overlap_frac"] = arow.overlap_frac
        row["is_event_window"] = int(arow.is_event_window)
        row["stage"] = arow.stage

        row.update(_timing_at(payload, w_idx, time=time))
        row.update(_mnps_at(payload, w_idx))
        row.update(_coords9d_at(payload, w_idx, include=config.include_coords_9d))
        row.update(_event_meta(event_table, ev_idx))

        # Control metadata not applicable
        row["match_rank"] = -1
        row["match_distance"] = np.nan
        row["matched_event_id"] = -1

        rows.append(row)

    # --- Matched-control rows ---------------------------------------------
    # Build a lookup: control_window_id → list of (event_id, rank, distance)
    control_lookup: Dict[int, List[Dict[str, Any]]] = {}
    for crow in controls.rows:
        control_lookup.setdefault(crow.control_window_id, []).append(
            {
                "event_id": crow.event_id,
                "match_rank": crow.match_rank,
                "match_distance": crow.match_distance,
            }
        )

    for crow in controls.rows:
        w_idx = crow.control_window_id
        ev_idx = crow.event_id

        row = {}
        if config.provenance_in_every_row:
            row.update(provenance)

        row["condition"] = "matched_control"
        row["event_id"] = -1
        row["window_id"] = w_idx
        # bin_label and rel_time for a control window are relative to the
        # matched spindle event's reference time. We emit NaN/empty here
        # because the control window was not binned — it is merely matched.
        row["bin_label"] = "control"
        row["rel_time_sec"] = np.nan
        row["overlap_sec"] = 0.0
        row["overlap_frac"] = 0.0
        row["is_event_window"] = 0
        row["stage"] = crow.stage

        row.update(_timing_at(payload, w_idx, time=time))
        row.update(_mnps_at(payload, w_idx))
        row.update(_coords9d_at(payload, w_idx, include=config.include_coords_9d))
        row.update(_event_meta(None, -1))  # no event metadata for control windows

        row["match_rank"] = crow.match_rank
        row["match_distance"] = crow.match_distance
        row["matched_event_id"] = ev_idx

        rows.append(row)

    logger.info(
        "Event-locked table: %d spindle rows + %d control rows = %d total (subject=%s)",
        len(alignment.rows),
        len(controls.rows),
        len(rows),
        subject_id,
    )
    return rows


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_event_locked_parquet(
    rows: List[Dict[str, Any]],
    out_path: Path,
) -> Optional[Path]:
    """Write rows to Parquet. Returns path on success, None if pyarrow unavailable.

    Missing columns are filled with ``None`` so all rows share the same schema.
    """
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        logger.warning("pandas not installed; cannot write Parquet. Install pandas + pyarrow.")
        return None

    if not rows:
        logger.info("No rows to write — skipping Parquet output at %s", out_path)
        return None

    try:
        df = pd.DataFrame(rows)
    except Exception:
        logger.exception("Failed to build DataFrame for Parquet export")
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_parquet(out_path, index=False)
        logger.info("Wrote event-locked Parquet: %s (%d rows, %d cols)", out_path, len(df), len(df.columns))
        return out_path
    except Exception:
        logger.exception("Failed to write Parquet to %s", out_path)
        # Fallback: write CSV so the data is not silently lost
        csv_path = out_path.with_suffix(".csv")
        try:
            df.to_csv(csv_path, index=False)
            logger.info("Fell back to CSV: %s", csv_path)
        except Exception:
            logger.exception("CSV fallback also failed")
        return None


def write_event_locked_csv(
    rows: List[Dict[str, Any]],
    out_path: Path,
) -> Optional[Path]:
    """Write rows to CSV using csv module (no pandas required).

    Missing columns in any row are filled with empty string.
    """
    if not rows:
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect full column set preserving insertion order from first row.
    all_cols: list[str] = list(rows[0].keys())
    seen = set(all_cols)
    for r in rows[1:]:
        for k in r:
            if k not in seen:
                all_cols.append(k)
                seen.add(k)

    import csv as _csv

    try:
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = _csv.DictWriter(fh, fieldnames=all_cols, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow({k: ("" if v is None else v) for k, v in r.items()})
        logger.info("Wrote event-locked CSV: %s (%d rows)", out_path, len(rows))
        return out_path
    except Exception:
        logger.exception("Failed to write CSV to %s", out_path)
        return None


# ---------------------------------------------------------------------------
# Provenance / manifest helper
# ---------------------------------------------------------------------------

def event_locked_export_manifest_entry(
    rows: List[Dict[str, Any]],
    *,
    alignment: AlignmentResult,
    controls: ControlMatchResult,
    out_paths: Optional[Sequence[Optional[Path]]] = None,
) -> Dict[str, Any]:
    """Return a small JSON-serializable dict for ``run_manifest.json``.

    Records row counts per condition, QC summaries, and output paths.
    """
    n_spindle = sum(1 for r in rows if r.get("condition") == "spindle_event")
    n_control = sum(1 for r in rows if r.get("condition") == "matched_control")

    paths_entry = []
    if out_paths:
        paths_entry = [str(p) for p in out_paths if p is not None]

    return {
        "n_rows_total": len(rows),
        "n_spindle_event_rows": n_spindle,
        "n_matched_control_rows": n_control,
        "alignment_qc": {k: v for k, v in alignment.qc.items() if k != "_bins_objects"},
        "control_qc": controls.qc,
        "output_paths": paths_entry,
    }
