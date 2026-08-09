"""Seeded, stratified non-event timestamps for point-process analyses."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .phase_continuous import PhaseContinuousState, sample_phase_state


def _stage_at_time(timestamp_sec: float, stage_codes: Sequence[int], stage_epoch_sec: float) -> int:
    index = int(timestamp_sec // stage_epoch_sec)
    return int(stage_codes[index]) if 0 <= index < len(stage_codes) else -1


def _quartile(timestamp_sec: float, duration_sec: float) -> int:
    if duration_sec <= 0:
        return -1
    return min(3, max(0, int(4.0 * timestamp_sec / duration_sec)))


def build_non_event_risk_catalog(
    state: PhaseContinuousState,
    *,
    subject: str,
    stage_codes: Sequence[int],
    spindle_onsets_sec: Iterable[float],
    stage_epoch_sec: float = 30.0,
    stage_filter: Iterable[int] = (2,),
    exclusion_margin_sec: float = 30.0,
    n_per_event: int = 3,
    seed: int = 42,
    require_both_phase_valid: bool = True,
) -> pd.DataFrame:
    """Draw a stage- and time-of-night-stratified non-event risk catalogue.

    The candidate population is the continuous phase grid, not the 30-s
    geometry control pool.  Candidate timestamps inside an event exclusion
    interval are removed before sampling.
    """
    columns = [
        "subject",
        "risk_id",
        "timestamp_sec",
        "stage",
        "time_of_night_quartile",
        "phi_cardiac",
        "phi_resp",
        "cardiac_phase_valid",
        "resp_phase_valid",
        "seed",
        "exclusion_margin_sec",
        "selection_status",
    ]
    if state.time_sec.size == 0 or len(stage_codes) == 0 or n_per_event < 1:
        return pd.DataFrame(columns=columns)

    time = state.time_sec
    duration_sec = float(time[-1]) if time.size else 0.0
    stages = np.asarray([_stage_at_time(t, stage_codes, stage_epoch_sec) for t in time], dtype=int)
    allowed = np.isin(stages, list(stage_filter))
    phase_valid = state.cardiac_valid & state.resp_valid if require_both_phase_valid else (state.cardiac_valid | state.resp_valid)
    candidates = allowed & phase_valid
    onsets = np.asarray([float(value) for value in spindle_onsets_sec if np.isfinite(float(value))], dtype=float)
    if onsets.size:
        for onset in onsets:
            candidates &= np.abs(time - onset) > float(exclusion_margin_sec)

    rng = np.random.default_rng(seed)
    available = np.flatnonzero(candidates)
    selected_indices: list[int] = []
    for onset in onsets:
        target_stage = _stage_at_time(float(onset), stage_codes, stage_epoch_sec)
        target_quartile = _quartile(float(onset), duration_sec)
        matching = available[
            (stages[available] == target_stage)
            & np.asarray([_quartile(float(time[index]), duration_sec) == target_quartile for index in available])
        ]
        take = min(int(n_per_event), int(matching.size))
        if take:
            chosen = rng.choice(matching, size=take, replace=False)
            selected_indices.extend(int(value) for value in chosen)
            available = available[~np.isin(available, chosen)]

    sampled = sample_phase_state(state, time[np.asarray(selected_indices, dtype=int)]) if selected_indices else pd.DataFrame()
    if sampled.empty:
        return pd.DataFrame(columns=columns)
    sampled.insert(0, "subject", str(subject))
    sampled.insert(1, "risk_id", np.arange(len(sampled), dtype=int))
    sampled["stage"] = [_stage_at_time(t, stage_codes, stage_epoch_sec) for t in sampled["timestamp_sec"]]
    sampled["time_of_night_quartile"] = [_quartile(t, duration_sec) for t in sampled["timestamp_sec"]]
    sampled["seed"] = int(seed)
    sampled["exclusion_margin_sec"] = float(exclusion_margin_sec)
    sampled["selection_status"] = "selected"
    return sampled[columns]
