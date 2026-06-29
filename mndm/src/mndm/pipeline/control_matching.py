"""Matched non-event control window sampling for event-locked MNPS analysis.

For each spindle (or other) event, this module samples one or more N2 windows
that share the same subject/session/stage but do *not* overlap with any event.

Design contract
---------------
* Deterministic under a user-supplied seed; seed is always recorded.
* Failed matches are never silently dropped — they are counted and flagged.
* Matching is done at the window level (indices into the MNPS time axis).
* No statistical inference here; output is a table for downstream analysis.
* Works with any event type, not just sleep spindles.

Output
------
``ControlMatchResult.rows`` — one row per (event_id, control_window_id) pair::

    event_id          int  — index into EventTable
    control_window_id int  — index into MNPS time axis
    match_rank        int  — 1-based rank (1 = best match)
    match_distance    float — distance metric used for matching (lower = better)
    stage             int  — stage code of the matched window

``ControlMatchResult.qc`` — matching coverage statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .event_annotations import EventTable

logger = logging.getLogger(__name__)


@dataclass
class MatchingConfig:
    """Parameters for control-window matching.

    Parameters
    ----------
    n_controls_per_event:
        Number of control windows to sample per event.
    target_stage:
        Stage code to restrict candidate pool (e.g. 2 = N2). ``None`` means
        accept windows of any stage.
    exclusion_margin_sec:
        Windows whose center falls within this many seconds of *any* event
        onset are excluded from the candidate pool.
    time_of_night_quartile_match:
        If ``True``, prefer candidates in the same quartile of recording time.
        Implemented as a soft penalty in match distance, not a hard filter.
    seed:
        RNG seed for deterministic sampling. Always recorded in QC.
    """

    n_controls_per_event: int = 3
    target_stage: Optional[int] = 2
    exclusion_margin_sec: float = 30.0
    time_of_night_quartile_match: bool = True
    seed: int = 1729


def matching_config_from_dict(cfg: Mapping[str, Any]) -> MatchingConfig:
    """Build a ``MatchingConfig`` from a YAML/dict sub-section.

    Expected keys (all optional)::

        n_controls_per_event: 3
        target_stage: 2
        exclusion_margin_sec: 30
        time_of_night_quartile_match: true
        seed: 1729
    """
    return MatchingConfig(
        n_controls_per_event=int(cfg.get("n_controls_per_event", 3)),
        target_stage=cfg.get("target_stage", 2),
        exclusion_margin_sec=float(cfg.get("exclusion_margin_sec", 30.0)),
        time_of_night_quartile_match=bool(cfg.get("time_of_night_quartile_match", True)),
        seed=int(cfg.get("seed", 1729)),
    )


@dataclass
class ControlMatchRow:
    """One matched control window entry."""

    event_id: int
    control_window_id: int
    match_rank: int
    match_distance: float
    stage: int


@dataclass
class ControlMatchResult:
    """Output of ``build_matched_controls``."""

    rows: List[ControlMatchRow]
    qc: Dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return len(self.rows) == 0

    def to_records(self) -> List[Dict[str, Any]]:
        return [
            {
                "event_id": r.event_id,
                "control_window_id": r.control_window_id,
                "match_rank": r.match_rank,
                "match_distance": r.match_distance,
                "stage": r.stage,
            }
            for r in self.rows
        ]


def _quartile(value: float, lo: float, hi: float) -> int:
    """Return 0-3 quartile index of value in [lo, hi]."""
    if not np.isfinite(value) or not np.isfinite(lo) or not np.isfinite(hi):
        return 0
    if hi <= lo:
        return 0
    frac = (value - lo) / (hi - lo)
    if not np.isfinite(frac):
        return 0
    return int(min(3, max(0, int(frac * 4))))


def build_matched_controls(
    table: EventTable,
    *,
    time: np.ndarray,
    window_start: np.ndarray,
    window_end: np.ndarray,
    stage: Optional[np.ndarray] = None,
    config: Optional[MatchingConfig] = None,
) -> ControlMatchResult:
    """Sample matched non-event control windows for each event.

    For each event in ``table``, the function:

    1. Builds a candidate pool of windows that pass stage and proximity filters.
    2. Scores candidates by distance to the event's recording-time quartile.
    3. Samples ``config.n_controls_per_event`` candidates without replacement.
    4. Records the result including match rank and distance.

    Events with fewer than ``n_controls_per_event`` valid candidates are
    flagged in ``qc`` but still returned (partially matched).

    Parameters
    ----------
    table:
        Event annotations. Empty → empty result.
    time:
        MNPS window center times, shape ``[T]``.
    window_start / window_end:
        Per-window time bounds, shape ``[T]``.
    stage:
        Per-window stage codes, shape ``[T]``. ``None`` → all -1.
    config:
        Matching configuration. Uses defaults if ``None``.

    Returns
    -------
    ControlMatchResult
    """
    if config is None:
        config = MatchingConfig()

    rng = np.random.default_rng(config.seed)

    n_windows = len(time)
    has_stage = stage is not None and stage.size == n_windows
    stage_arr = np.asarray(stage, dtype=np.int16) if has_stage else np.full(n_windows, -1, dtype=np.int16)

    qc: Dict[str, Any] = {
        "seed": config.seed,
        "n_events_input": table.n if not table.is_empty() else 0,
        "n_events_with_full_match": 0,
        "n_events_with_partial_match": 0,
        "n_events_with_no_match": 0,
        "n_controls_requested_per_event": config.n_controls_per_event,
        "target_stage": config.target_stage,
        "exclusion_margin_sec": config.exclusion_margin_sec,
    }

    if table.is_empty() or n_windows == 0:
        return ControlMatchResult(rows=[], qc=qc)

    finite_time = np.asarray(time, dtype=np.float64)
    finite_time = finite_time[np.isfinite(finite_time)]
    if finite_time.size == 0:
        return ControlMatchResult(rows=[], qc=qc)
    t_min = float(np.min(finite_time))
    t_max = float(np.max(finite_time))

    # Pre-compute all event onset times for exclusion mask.
    event_onsets = table.onset_sec.astype(np.float64)
    # Include event offsets/durations in exclusion zone where available.
    event_ends: np.ndarray = event_onsets.copy()
    if table.duration_sec is not None:
        finite_dur = np.isfinite(table.duration_sec)
        event_ends[finite_dur] = event_onsets[finite_dur] + table.duration_sec[finite_dur]

    rows: List[ControlMatchRow] = []

    for ev_idx in range(table.n):
        onset = float(event_onsets[ev_idx])
        if not np.isfinite(onset):
            qc["n_events_with_no_match"] += 1
            continue

        ev_end = float(event_ends[ev_idx])
        ev_quartile = _quartile(onset, t_min, t_max)

        # Build exclusion mask: any window overlapping any event's exclusion zone.
        in_exclusion = np.zeros(n_windows, dtype=bool)
        for i in range(table.n):
            eo = float(event_onsets[i])
            ee = float(event_ends[i])
            if not np.isfinite(eo):
                continue
            margin = config.exclusion_margin_sec
            in_exclusion |= (time >= (eo - margin)) & (time <= (ee + margin))

        # Build candidate mask.
        candidate_mask = ~in_exclusion & np.isfinite(time)
        if config.target_stage is not None:
            candidate_mask &= stage_arr == config.target_stage

        candidate_indices = np.where(candidate_mask)[0]
        if candidate_indices.size == 0:
            qc["n_events_with_no_match"] += 1
            logger.debug("Event %d: no candidates after filtering", ev_idx)
            continue

        # Score: absolute quartile distance (soft match).
        if config.time_of_night_quartile_match:
            cand_quartiles = np.array(
                [_quartile(float(time[i]), t_min, t_max) for i in candidate_indices],
                dtype=np.int32,
            )
            distances = np.abs(cand_quartiles - ev_quartile).astype(np.float32)
        else:
            distances = np.zeros(candidate_indices.size, dtype=np.float32)

        # Add small uniform noise to break ties reproducibly.
        distances += rng.uniform(0, 1e-4, size=distances.size).astype(np.float32)

        # Sort candidates by distance.
        order = np.argsort(distances, kind="stable")
        sorted_indices = candidate_indices[order]
        sorted_distances = distances[order]

        n_to_sample = min(config.n_controls_per_event, sorted_indices.size)
        selected = sorted_indices[:n_to_sample]
        sel_distances = sorted_distances[:n_to_sample]

        if n_to_sample == config.n_controls_per_event:
            qc["n_events_with_full_match"] += 1
        elif n_to_sample > 0:
            qc["n_events_with_partial_match"] += 1
        else:
            qc["n_events_with_no_match"] += 1
            continue

        for rank, (w_idx, dist) in enumerate(zip(selected, sel_distances), start=1):
            rows.append(
                ControlMatchRow(
                    event_id=ev_idx,
                    control_window_id=int(w_idx),
                    match_rank=rank,
                    match_distance=float(dist),
                    stage=int(stage_arr[w_idx]),
                )
            )

    match_rate = (
        (qc["n_events_with_full_match"] + qc["n_events_with_partial_match"]) / qc["n_events_input"]
        if qc["n_events_input"] > 0
        else 0.0
    )
    qc["match_success_rate"] = round(match_rate, 4)

    logger.info(
        "Control matching: %d full, %d partial, %d no-match (rate=%.2f, seed=%d)",
        qc["n_events_with_full_match"],
        qc["n_events_with_partial_match"],
        qc["n_events_with_no_match"],
        match_rate,
        config.seed,
    )
    return ControlMatchResult(rows=rows, qc=qc)
