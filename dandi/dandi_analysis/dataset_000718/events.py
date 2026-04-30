"""High-synchrony event detection and ensemble recruitment scoring.

Replaces the whole-window mean/burst-score approach with an event-first
pipeline:

1. Detect high-synchrony frames in the population activity trace.
2. Merge contiguous high-synchrony frames into candidate offline events.
3. Score each event by projecting it onto encoding-defined ensemble spatial
   profiles.
4. Return a per-event, per-ensemble recruitment score table.

Null is a circular-shift applied to the population activity trace before event
detection, preserving autocorrelation structure while destroying temporal
alignment with ensemble profiles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SynchronyEvent:
    """One candidate offline reactivation event."""
    event_idx: int
    start_frame: int
    stop_frame: int               # exclusive
    duration_frames: int
    peak_population_activity: float
    mean_population_activity: float
    n_active_units: int           # units above 1-sigma during event


@dataclass
class EventRecruitmentScore:
    """Projection of one ensemble onto one event."""
    event_idx: int
    ensemble_idx: int
    peak_projection: float
    mean_projection: float
    norm_score: float             # (observed - null_mean) / null_std
    null_mean: float
    null_std: float


@dataclass
class EventDetectionResult:
    """All events detected in one session/window."""
    session_id: str
    n_frames: int
    n_units: int
    threshold_sigma: float
    threshold_value: float
    n_events: int
    events: list[SynchronyEvent]


@dataclass
class H1EventResult:
    """Full event-based H1 result for one NE→Offline pair."""
    session_ne: str
    session_offline: str
    n_registered_units: int
    n_offline_frames: int
    n_events: int
    n_ensembles: int
    recruitment_scores: list[EventRecruitmentScore]
    null_n: int
    # Summary
    n_significant_pairs: int = 0   # (event, ensemble) pairs with norm_score > 1.96
    max_norm_score: float = 0.0
    session_summary: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------

def detect_synchrony_events(
    data: np.ndarray,
    *,
    threshold_sigma: float = 2.0,
    min_duration_frames: int = 3,
    min_gap_frames: int = 10,
    activity_sigma_unit: float = 1.0,
) -> EventDetectionResult:
    """Detect high-synchrony frames in a (T, N) activity matrix.

    Population activity at each frame = fraction of units exceeding
    *activity_sigma_unit* standard deviations above their mean.  Frames
    where this fraction exceeds mean + *threshold_sigma* * std are
    high-synchrony candidates.

    Parameters
    ----------
    data:
        z-scored activity matrix, shape (T, N).
    threshold_sigma:
        Population-activity threshold in units of std above mean.
    min_duration_frames:
        Minimum consecutive frames to constitute an event.
    min_gap_frames:
        Minimum gap between events (frames below threshold that separate them).
    activity_sigma_unit:
        Per-unit activation threshold for counting a unit as "active" in a frame.
    """
    T, N = data.shape
    unit_active = (data > activity_sigma_unit)          # (T, N) bool
    pop_frac = unit_active.mean(axis=1)                 # (T,) fraction active

    pop_mean = float(pop_frac.mean())
    pop_std = float(pop_frac.std())
    threshold = pop_mean + threshold_sigma * pop_std

    above = pop_frac > threshold
    events: list[SynchronyEvent] = []

    # Find contiguous above-threshold runs
    i = 0
    while i < T:
        if above[i]:
            j = i + 1
            while j < T and above[j]:
                j += 1
            duration = j - i
            if duration >= min_duration_frames:
                chunk = data[i:j]
                pop_chunk = pop_frac[i:j]
                events.append(SynchronyEvent(
                    event_idx=len(events),
                    start_frame=i,
                    stop_frame=j,
                    duration_frames=duration,
                    peak_population_activity=float(pop_chunk.max()),
                    mean_population_activity=float(pop_chunk.mean()),
                    n_active_units=int(unit_active[i:j].any(axis=0).sum()),
                ))
            i = j
        else:
            i += 1

    # Merge events separated by fewer than min_gap_frames
    if events and min_gap_frames > 0:
        merged: list[SynchronyEvent] = [events[0]]
        for ev in events[1:]:
            prev = merged[-1]
            gap = ev.start_frame - prev.stop_frame
            if gap < min_gap_frames:
                # Merge
                chunk = data[prev.start_frame:ev.stop_frame]
                pop_chunk = pop_frac[prev.start_frame:ev.stop_frame]
                merged[-1] = SynchronyEvent(
                    event_idx=prev.event_idx,
                    start_frame=prev.start_frame,
                    stop_frame=ev.stop_frame,
                    duration_frames=ev.stop_frame - prev.start_frame,
                    peak_population_activity=float(pop_chunk.max()),
                    mean_population_activity=float(pop_chunk.mean()),
                    n_active_units=int((data[prev.start_frame:ev.stop_frame] > activity_sigma_unit).any(axis=0).sum()),
                )
            else:
                merged.append(ev)
        events = merged

    return EventDetectionResult(
        session_id="",
        n_frames=T,
        n_units=N,
        threshold_sigma=threshold_sigma,
        threshold_value=float(threshold),
        n_events=len(events),
        events=events,
    )


# ---------------------------------------------------------------------------
# Ensemble recruitment scoring
# ---------------------------------------------------------------------------

def score_event_recruitment(
    offline_data: np.ndarray,
    detection: EventDetectionResult,
    ensemble_weights: np.ndarray,
    ensemble_idx: int,
    *,
    null_n: int = 200,
    rng_seed: int = 42,
) -> list[EventRecruitmentScore]:
    """Score each event by projection onto one ensemble weight vector.

    Parameters
    ----------
    offline_data:
        Activity matrix (T, N), units aligned with ensemble_weights.
    detection:
        EventDetectionResult from detect_synchrony_events.
    ensemble_weights:
        Spatial profile of the encoding ensemble, shape (N,).
    ensemble_idx:
        Index of this ensemble (for bookkeeping).
    null_n:
        Number of circular-shift null draws.

    Returns
    -------
    list of EventRecruitmentScore, one per detected event.
    """
    if not detection.events:
        return []

    weights = ensemble_weights / (ensemble_weights.max() + 1e-12)
    projection = offline_data @ weights      # (T,)

    rng = np.random.default_rng(rng_seed + ensemble_idx)
    T = len(projection)
    min_shift = max(10, T // 20)
    max_shift = max(min_shift + 1, T - min_shift)

    scores: list[EventRecruitmentScore] = []
    for ev in detection.events:
        ev_proj = projection[ev.start_frame:ev.stop_frame]
        observed_peak = float(ev_proj.max()) if len(ev_proj) > 0 else 0.0
        observed_mean = float(ev_proj.mean()) if len(ev_proj) > 0 else 0.0

        # Null: draw the same duration window from a shifted projection
        dur = ev.duration_frames
        null_peaks: list[float] = []
        for _ in range(null_n):
            shift = int(rng.integers(min_shift, max_shift))
            proj_shifted = np.roll(projection, shift)
            # Sample at a random position of same duration from the shifted series
            start_null = int(rng.integers(0, max(1, T - dur)))
            null_peaks.append(float(proj_shifted[start_null:start_null + dur].max()))

        null_arr = np.array(null_peaks)
        null_mean = float(null_arr.mean())
        null_std = float(null_arr.std())
        norm_score = (observed_peak - null_mean) / null_std if null_std > 0 else float("nan")

        scores.append(EventRecruitmentScore(
            event_idx=ev.event_idx,
            ensemble_idx=ensemble_idx,
            peak_projection=observed_peak,
            mean_projection=observed_mean,
            norm_score=norm_score,
            null_mean=null_mean,
            null_std=null_std,
        ))

    return scores


def run_event_h1(
    offline_data: np.ndarray,
    encoding_ensemble_weights: list[np.ndarray],
    session_ne: str,
    session_offline: str,
    n_registered_units: int,
    *,
    threshold_sigma: float = 2.0,
    min_duration_frames: int = 3,
    min_gap_frames: int = 10,
    null_n: int = 200,
    rng_seed: int = 42,
) -> H1EventResult:
    """Run the full event-based H1 pipeline for one NE→Offline pair.

    1. Detect synchrony events in offline_data.
    2. Score each event against each encoding ensemble.
    3. Return an H1EventResult with all scores and a session summary.
    """
    detection = detect_synchrony_events(
        offline_data,
        threshold_sigma=threshold_sigma,
        min_duration_frames=min_duration_frames,
        min_gap_frames=min_gap_frames,
    )
    detection.session_id = session_offline

    all_scores: list[EventRecruitmentScore] = []
    for k, weights in enumerate(encoding_ensemble_weights):
        scores = score_event_recruitment(
            offline_data, detection, weights, ensemble_idx=k,
            null_n=null_n, rng_seed=rng_seed,
        )
        all_scores.extend(scores)

    valid = [s for s in all_scores if s.norm_score == s.norm_score]
    n_sig = sum(1 for s in valid if s.norm_score > 1.96)
    max_z = float(max((s.norm_score for s in valid), default=float("nan")))

    result = H1EventResult(
        session_ne=session_ne,
        session_offline=session_offline,
        n_registered_units=n_registered_units,
        n_offline_frames=detection.n_frames,
        n_events=detection.n_events,
        n_ensembles=len(encoding_ensemble_weights),
        recruitment_scores=all_scores,
        null_n=null_n,
        n_significant_pairs=n_sig,
        max_norm_score=max_z,
    )

    total_pairs = len(valid)
    result.session_summary = {
        "n_events": detection.n_events,
        "n_ensembles": len(encoding_ensemble_weights),
        "total_scored_pairs": total_pairs,
        "n_significant_z196": n_sig,
        "fraction_significant": round(n_sig / max(1, total_pairs), 3),
        "max_norm_score": round(max_z, 3) if max_z == max_z else None,
        "threshold_sigma": threshold_sigma,
        "event_mean_duration_frames": round(
            float(np.mean([e.duration_frames for e in detection.events])), 1
        ) if detection.events else 0,
    }

    return result
