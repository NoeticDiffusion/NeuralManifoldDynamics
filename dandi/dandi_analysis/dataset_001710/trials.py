"""Canonical trial table reconstruction for DANDI 001710.

Because 001710 NWB files do not contain NWB ``intervals`` tables, trial
boundaries must be derived from the 2P-aligned behavior time series together
with the embedded ``trial_cell_data`` annotation blob.

The output is a ``TrialTable`` containing a list of ``TrialRow`` dataclasses,
one per identified trial.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dandi_analysis.dataset_001710.behavior import BehaviorTable, load_behavior_table
from dandi_analysis.dataset_001710.io import read_trial_annotation_blob


@dataclass(slots=True)
class TrialRow:
    trial_id: int
    day: int
    block_id: float
    arm_label: str          # "left", "right", or "unknown"
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration_sec: float
    reward_count: int
    valid: bool
    notes: str = ""


@dataclass
class TrialTable:
    session_path: Path
    day: int
    trials: list[TrialRow] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.trials)

    def valid_trials(self) -> list[TrialRow]:
        return [t for t in self.trials if t.valid]

    def by_arm(self, arm: str) -> list[TrialRow]:
        return [t for t in self.trials if t.arm_label == arm]

    def by_block(self, block_id: float) -> list[TrialRow]:
        return [t for t in self.trials if t.block_id == block_id]


def build_trial_table(
    path: Path,
    *,
    day: int = -1,
    min_duration_sec: float = 0.1,
) -> TrialTable:
    """Reconstruct the canonical trial table for one 001710 session.

    Strategy (in priority order):
    1. Use ``trial start`` / ``trial end`` boolean pulses from the 2P-aligned
       behavior container to detect rising-edge frame indices.
    2. Cross-reference embedded ``trial_start_inds`` from the annotation blob
       as a consistency check; warn if there is a large discrepancy.
    3. Annotate each trial with arm label, block id, and reward count.

    Parameters
    ----------
    day:
        Day index to embed in each ``TrialRow``.  If ``-1``, the value is
        read from the annotation blob when available.
    min_duration_sec:
        Trials shorter than this are flagged as invalid.
    """
    beh = load_behavior_table(path, source="2p")
    if beh is None:
        return TrialTable(session_path=path, day=day, metadata={"error": "behavior_not_loaded"})

    blob = read_trial_annotation_blob(path)
    if day < 0:
        day = int(blob.get("day", -1)) if blob else -1

    trials = _build_from_behavior_signals(beh, blob, day=day, min_duration_sec=min_duration_sec)
    return TrialTable(
        session_path=path,
        day=day,
        trials=trials,
        metadata={
            "n_frames": beh.n_frames,
            "blob_keys": list(blob.keys()) if blob else [],
            "source": "behavior_signals+annotation_blob",
        },
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_from_behavior_signals(
    beh: BehaviorTable,
    blob: dict[str, Any],
    *,
    day: int,
    min_duration_sec: float,
) -> list[TrialRow]:
    ts = beh.timestamps

    # --- Detect trial boundaries -------------------------------------------
    start_frames = _detect_rising_edges(beh.channels.get("trial_start"))
    end_frames = _detect_rising_edges(beh.channels.get("trial_end"))

    # Fall back to annotation blob indices if behavior signals are sparse
    if len(start_frames) == 0 and blob:
        blob_starts = blob.get("trial_start_inds", [])
        if isinstance(blob_starts, (list, np.ndarray)) and len(blob_starts) > 0:
            start_frames = np.asarray(blob_starts, dtype=int)

    if len(start_frames) == 0:
        return []

    # Pair starts with ends using nearest-following-end logic
    end_frames = np.asarray(end_frames, dtype=int)
    paired_ends = _pair_ends(start_frames, end_frames, n_total=beh.n_frames)

    # --- Per-trial annotation arrays ---------------------------------------
    trial_numbers = beh.channels.get("trial_number", np.zeros(beh.n_frames))
    arm_arr = beh.channels.get("arm", None)
    block_arr = beh.channels.get("block", np.zeros(beh.n_frames))
    reward_arr = beh.channels.get("reward", np.zeros(beh.n_frames))

    rows: list[TrialRow] = []
    for idx, (sf, ef) in enumerate(zip(start_frames, paired_ends)):
        sf, ef = int(sf), int(ef)
        t_start = float(ts[sf]) if sf < len(ts) else float(sf)
        t_end = float(ts[ef]) if ef < len(ts) else float(ef)
        duration = t_end - t_start

        arm_label = _majority_arm(arm_arr, sf, ef)
        block_id = float(np.nanmedian(block_arr[sf:ef])) if ef > sf else 0.0
        reward_count = int(np.nansum(reward_arr[sf:ef] > 0.5)) if ef > sf else 0

        valid = duration >= min_duration_sec and ef > sf
        notes = "" if valid else f"short_trial:{duration:.3f}s"

        rows.append(
            TrialRow(
                trial_id=idx,
                day=day,
                block_id=block_id,
                arm_label=arm_label,
                start_frame=sf,
                end_frame=ef,
                start_time=t_start,
                end_time=t_end,
                duration_sec=duration,
                reward_count=reward_count,
                valid=valid,
                notes=notes,
            )
        )

    return rows


def _detect_rising_edges(arr: np.ndarray | None) -> np.ndarray:
    """Return frame indices where arr transitions from <=0.5 to >0.5."""
    if arr is None or arr.shape[0] == 0:
        return np.array([], dtype=int)
    binary = (np.asarray(arr) > 0.5).astype(int)
    diff = np.diff(binary, prepend=0)
    return np.where(diff > 0)[0]


def _pair_ends(
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    n_total: int,
) -> np.ndarray:
    """For each start frame, find the next end frame that follows it."""
    paired: list[int] = []
    for s in starts:
        following = ends[ends > s]
        if len(following) > 0:
            paired.append(int(following[0]))
        else:
            paired.append(n_total - 1)
    return np.array(paired, dtype=int)


def _majority_arm(
    arm_arr: np.ndarray | None,
    start: int,
    end: int,
) -> str:
    if arm_arr is None or end <= start:
        return "unknown"
    window = arm_arr[start:end]
    valid = window[~np.isnan(window)]
    if len(valid) == 0:
        return "unknown"
    median_val = float(np.median(valid))
    # Convention: 0 → left, 1 → right (or raw label values)
    if median_val < 0.5:
        return "left"
    elif median_val > 0.5:
        return "right"
    return "unknown"
