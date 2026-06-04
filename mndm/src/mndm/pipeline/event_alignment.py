"""Event-to-MNPS-window alignment with relative time bins.

Takes an ``EventTable`` and the MNPS window arrays (``window_start``,
``window_end``, ``time``, optional ``stage``) and produces a row-per-window
alignment table.

Output columns
--------------
event_id        int   — index into the EventTable (0-based)
window_id       int   — index into the MNPS time axis
rel_time_sec    float — window center minus event reference time (seconds)
bin_label       str   — named temporal bin (e.g. ``"pre_near"``, ``"event"``)
overlap_sec     float — seconds of overlap between window and event interval
overlap_frac    float — overlap_sec / window duration
stage           int   — sleep/task stage code for the window (-1 = unknown)
is_event_window bool  — True when overlap_frac >= overlap_threshold

Design contract
---------------
* Pure functions; no I/O.
* Absent events → empty result (never raises).
* Bins are fully configurable; the default set follows the architect spec.
* Boundary/stage-transition exclusion is opt-in and counted.
* All exclusion counts are surfaced in a QC dict for manifests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .event_annotations import EventTable
from .intervals import overlap_frac, overlap_sec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default bin definitions  (edges are relative to event reference in seconds)
# ---------------------------------------------------------------------------

DEFAULT_BINS: List[Tuple[str, float, float]] = [
    ("pre_far", -30.0, -10.0),
    ("pre_near", -10.0, 0.0),
    ("event", 0.0, 3.0),
    ("post_near", 3.0, 10.0),
    ("post_far", 10.0, 30.0),
]


@dataclass
class BinSpec:
    """One named temporal bin relative to an event reference."""

    label: str
    lo: float  # seconds, inclusive lower bound relative to event ref
    hi: float  # seconds, exclusive upper bound relative to event ref

    def contains(self, rel_sec: float) -> bool:
        return self.lo <= rel_sec < self.hi


@dataclass
class AlignmentConfig:
    """Configuration for event→window alignment.

    Parameters
    ----------
    reference:
        Which event timestamp to use as ``t=0``. One of ``"onset"``,
        ``"peak"``, ``"offset"``. Falls back to ``onset_sec`` when the
        requested column is absent.
    bins:
        Ordered list of ``BinSpec``. Windows may fall into at most one bin
        (first match wins). Windows outside all bins receive label ``"outside"``.
    overlap_threshold:
        Minimum overlap fraction to mark a window as ``is_event_window``.
        Default 0.0 (any overlap counts).
    stage_transition_margin_sec:
        Exclude events whose reference time is within this many seconds of a
        sleep-stage transition. Set to 0 to disable. Counted in QC.
    stage_filter:
        If not empty, only process events where the MNPS-window stage code
        is one of these values. ``None`` means accept any stage.
    """

    reference: str = "onset"
    bins: List[BinSpec] = field(default_factory=lambda: [BinSpec(l, lo, hi) for l, lo, hi in DEFAULT_BINS])
    overlap_threshold: float = 0.0
    stage_transition_margin_sec: float = 30.0
    stage_filter: Optional[Sequence[int]] = None


def _parse_bins_from_config(cfg: Mapping[str, Any]) -> List[BinSpec]:
    """Parse bins from YAML config dict, falling back to DEFAULT_BINS."""
    raw = cfg.get("bins_sec", {})
    if not isinstance(raw, Mapping) or not raw:
        return [BinSpec(l, lo, hi) for l, lo, hi in DEFAULT_BINS]
    bins = []
    for label, bounds in raw.items():
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            try:
                bins.append(BinSpec(str(label), float(bounds[0]), float(bounds[1])))
            except (TypeError, ValueError):
                logger.warning("Skipping invalid bin spec for '%s': %s", label, bounds)
    return bins if bins else [BinSpec(l, lo, hi) for l, lo, hi in DEFAULT_BINS]


def alignment_config_from_dict(cfg: Mapping[str, Any]) -> AlignmentConfig:
    """Build an ``AlignmentConfig`` from a YAML/dict sub-section.

    Expected keys (all optional)::

        reference: "onset"   # or "peak" / "offset"
        bins_sec:
            pre_far:   [-30, -10]
            pre_near:  [-10,   0]
            event:     [  0,   3]
            post_near: [  3,  10]
            post_far:  [ 10,  30]
        min_overlap_fraction: 0.0
        exclude_stage_transition_margin_sec: 30
        stage_filter: [2]   # optional: only N2 windows
    """
    return AlignmentConfig(
        reference=str(cfg.get("reference", "onset")),
        bins=_parse_bins_from_config(cfg),
        overlap_threshold=float(cfg.get("min_overlap_fraction", cfg.get("overlap_threshold", 0.0))),
        stage_transition_margin_sec=float(cfg.get("exclude_stage_transition_margin_sec", 30.0)),
        stage_filter=cfg.get("stage_filter", None),
    )


@dataclass
class AlignmentRow:
    """One row of the event-window alignment table."""

    event_id: int
    window_id: int
    rel_time_sec: float
    bin_label: str
    overlap_sec: float
    overlap_frac: float
    stage: int
    is_event_window: bool


@dataclass
class AlignmentResult:
    """Output of ``align_events_to_windows``."""

    rows: List[AlignmentRow]
    qc: Dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return len(self.rows) == 0

    def to_records(self) -> List[Dict[str, Any]]:
        """Serialize rows to plain dicts (suitable for Parquet/CSV export)."""
        return [
            {
                "event_id": r.event_id,
                "window_id": r.window_id,
                "rel_time_sec": r.rel_time_sec,
                "bin_label": r.bin_label,
                "overlap_sec": r.overlap_sec,
                "overlap_frac": r.overlap_frac,
                "stage": r.stage,
                "is_event_window": int(r.is_event_window),
            }
            for r in self.rows
        ]


def _event_reference_time(table: EventTable, idx: int, reference: str) -> float:
    """Return the reference timestamp for event ``idx``."""
    if reference == "peak" and table.peak_sec is not None:
        v = float(table.peak_sec[idx])
        if np.isfinite(v):
            return v
    if reference == "offset" and table.offset_sec is not None:
        v = float(table.offset_sec[idx])
        if np.isfinite(v):
            return v
    return float(table.onset_sec[idx])


def _detect_stage_transitions(stage: np.ndarray) -> np.ndarray:
    """Return indices where stage changes (transition *starts* at that index)."""
    if stage is None or stage.size < 2:
        return np.empty(0, dtype=np.int64)
    changes = np.where(np.diff(stage.astype(np.int32)) != 0)[0] + 1
    return changes.astype(np.int64)


def align_events_to_windows(
    table: EventTable,
    *,
    window_start: np.ndarray,
    window_end: np.ndarray,
    time: np.ndarray,
    stage: Optional[np.ndarray] = None,
    config: Optional[AlignmentConfig] = None,
) -> AlignmentResult:
    """Align events to MNPS windows and assign relative-time bin labels.

    Parameters
    ----------
    table:
        Event annotations. If empty, an empty ``AlignmentResult`` is returned.
    window_start / window_end:
        Per-window time bounds in seconds, shape ``[T]``.
    time:
        Window center times in seconds, shape ``[T]``.
    stage:
        Optional per-window stage codes (int), shape ``[T]``.
        -1 or missing = unknown.
    config:
        Alignment configuration. Uses defaults if ``None``.

    Returns
    -------
    AlignmentResult
        ``rows`` contains one entry per (event, window) pair that falls
        within any bin. ``qc`` summarises exclusion counts.
    """
    if config is None:
        config = AlignmentConfig()

    if table.is_empty():
        return AlignmentResult(rows=[], qc={
            "n_events_input": 0,
            "n_events_excluded_non_finite": 0,
            "n_events_excluded_stage_transition": 0,
            "n_events_aligned": 0,
            "n_rows_total": 0,
            "bins_used": [b.label for b in config.bins],
            "reference": config.reference,
        })

    n_windows = len(time)
    has_stage = stage is not None and stage.size == n_windows

    stage_arr = stage if has_stage else np.full(n_windows, -1, dtype=np.int16)

    # Gracefully handle absent window bounds: derive from time axis spacing.
    if window_start is None or window_end is None:
        if n_windows > 1:
            dt = float(np.median(np.diff(time)))
        else:
            dt = 1.0
        half = dt / 2.0
        window_start = time - half
        window_end = time + half

    window_dur = window_end - window_start

    transition_times: np.ndarray = np.empty(0, dtype=np.float64)
    if config.stage_transition_margin_sec > 0 and has_stage:
        t_idx = _detect_stage_transitions(stage_arr)
        if t_idx.size:
            transition_times = time[t_idx]

    qc: Dict[str, Any] = {
        "n_events_input": table.n,
        "n_events_excluded_non_finite": 0,
        "n_events_excluded_stage_transition": 0,
        "n_events_aligned": 0,
        "n_rows_total": 0,
        "bins_used": [b.label for b in config.bins],
        "reference": config.reference,
    }

    rows: List[AlignmentRow] = []

    for ev_idx in range(table.n):
        onset = float(table.onset_sec[ev_idx])
        if not np.isfinite(onset):
            qc["n_events_excluded_non_finite"] += 1
            continue

        ref_t = _event_reference_time(table, ev_idx, config.reference)
        if not np.isfinite(ref_t):
            qc["n_events_excluded_non_finite"] += 1
            continue

        # Stage-transition margin exclusion
        if transition_times.size and config.stage_transition_margin_sec > 0:
            near_transition = np.any(np.abs(transition_times - ref_t) < config.stage_transition_margin_sec)
            if near_transition:
                qc["n_events_excluded_stage_transition"] += 1
                continue

        # Event interval for overlap computation
        ev_end = onset
        if table.duration_sec is not None:
            d = float(table.duration_sec[ev_idx])
            if np.isfinite(d) and d > 0:
                ev_end = onset + d
        elif table.offset_sec is not None:
            off = float(table.offset_sec[ev_idx])
            if np.isfinite(off) and off > onset:
                ev_end = off

        qc["n_events_aligned"] += 1
        ev_rows = 0

        for w_idx in range(n_windows):
            w_start = float(window_start[w_idx])
            w_end = float(window_end[w_idx])
            w_dur = float(window_dur[w_idx])
            w_center = float(time[w_idx])
            w_stage = int(stage_arr[w_idx])

            rel_sec = w_center - ref_t

            # Assign bin (first match wins)
            bin_label = "outside"
            for b in config.bins:
                if b.contains(rel_sec):
                    bin_label = b.label
                    break

            if bin_label == "outside":
                continue

            # Overlap with event interval
            overlap_sec_val = float(overlap_sec(w_start, w_end, onset, ev_end))
            overlap_frac_val = float(overlap_frac(w_start, w_end, onset, ev_end))

            is_event_window = overlap_frac_val >= config.overlap_threshold

            rows.append(
                AlignmentRow(
                    event_id=ev_idx,
                    window_id=w_idx,
                    rel_time_sec=rel_sec,
                    bin_label=bin_label,
                    overlap_sec=overlap_sec_val,
                    overlap_frac=overlap_frac_val,
                    stage=w_stage,
                    is_event_window=is_event_window,
                )
            )
            ev_rows += 1

        qc["n_rows_total"] += ev_rows

    logger.info(
        "Event alignment: %d/%d events aligned, %d rows produced",
        qc["n_events_aligned"],
        qc["n_events_input"],
        qc["n_rows_total"],
    )
    return AlignmentResult(rows=rows, qc=qc)
