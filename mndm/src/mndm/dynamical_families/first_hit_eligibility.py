"""Protocol-only eligibility checks for continuous first-hit committors.

This module deliberately does not estimate a committor.  It evaluates whether
windowed recordings satisfy the frozen OD-EPI-002 protocol before any explicit
reaction-coordinate and first-hit labels may be passed to the OD-TQ2b adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..pipeline.intervals import (
    TimeInterval,
    WindowMembershipSpec,
    window_membership_mask,
)

OD_EPI_002_PROTOCOL_ID = "OD-EPI-002-first-hit-v1"
OD_EPI_002_SCHEMA_VERSION = "mndm.od_epi_002_first_hit_eligibility.v1"


@dataclass(frozen=True)
class FirstHitEligibilityProtocol:
    """Frozen metadata contract for an onset first-hit eligibility audit."""

    protocol_id: str = OD_EPI_002_PROTOCOL_ID
    window_sec: float = 8.0
    step_sec: float = 4.0
    stable_a_buffer_sec: float = 8.0
    min_stable_a_windows: int = 5
    min_b_core_windows: int = 5
    adapter_min_transition_segments: int = 5
    analysis_min_transition_segments: int = 20
    onset_event_type: str = "sz onset"
    offset_event_type: str = "sz offset"
    core_membership: str = "fully_contained"
    onset_smear_membership: str = "contains_onset"
    reaction_coordinate_key: str | None = None
    partition_grain: str = "subject"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of the frozen protocol."""
        return {
            "protocol_id": self.protocol_id,
            "window_sec": self.window_sec,
            "step_sec": self.step_sec,
            "stable_a_buffer_sec": self.stable_a_buffer_sec,
            "min_stable_a_windows": self.min_stable_a_windows,
            "min_b_core_windows": self.min_b_core_windows,
            "adapter_min_transition_segments": self.adapter_min_transition_segments,
            "analysis_min_transition_segments": self.analysis_min_transition_segments,
            "onset_event_type": self.onset_event_type,
            "offset_event_type": self.offset_event_type,
            "core_membership": self.core_membership,
            "onset_smear_membership": self.onset_smear_membership,
            "reaction_coordinate_key": self.reaction_coordinate_key,
            "partition_grain": self.partition_grain,
        }


def pair_first_onset_with_next_offset(
    event_types: Sequence[str],
    event_times_sec: Sequence[float],
    *,
    onset_event_type: str = "sz onset",
    offset_event_type: str = "sz offset",
) -> tuple[float, float] | None:
    """Pair the first onset with the next later offset point event."""
    types = np.asarray([str(value).strip() for value in event_types], dtype=object)
    times = np.asarray(event_times_sec, dtype=float).reshape(-1)
    if types.size != times.size:
        return None
    onset_indices = np.flatnonzero(
        (types == onset_event_type) & np.isfinite(times)
    )
    if onset_indices.size == 0:
        return None
    onset_index = int(onset_indices[np.argmin(times[onset_indices])])
    onset = float(times[onset_index])
    offset_indices = np.flatnonzero(
        (types == offset_event_type)
        & np.isfinite(times)
        & (times > onset)
    )
    if offset_indices.size == 0:
        return None
    offset_index = int(offset_indices[np.argmin(times[offset_indices])])
    return onset, float(times[offset_index])


def max_consecutive_true(mask: Sequence[bool]) -> int:
    """Return the longest consecutive run of true values."""
    values = np.asarray(mask, dtype=bool).reshape(-1)
    if values.size == 0:
        return 0
    padded = np.concatenate(([False], values, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return int(np.max(ends - starts)) if starts.size else 0


def explicit_reaction_coordinate_present(
    columns: Iterable[str],
    reaction_coordinate_key: str | None,
) -> bool:
    """Return true only for an explicitly named available column."""
    key = str(reaction_coordinate_key or "").strip()
    return bool(key) and key in {str(column) for column in columns}


def subject_split_leaks(
    assignments: Mapping[str, str],
    *,
    subject_key: str = "subject",
) -> bool:
    """Return true when a subject is assigned to more than one split.

    ``assignments`` maps a recording identifier to a mapping-like record with
    ``subject`` and ``split`` fields.  The helper accepts a plain mapping of
    recording identifiers to ``(subject, split)`` tuples as a compact fixture
    form as well.
    """
    by_subject: dict[str, set[str]] = {}
    for value in assignments.values():
        if isinstance(value, Mapping):
            subject = str(value.get(subject_key, ""))
            split = str(value.get("split", ""))
        else:
            subject, split = (str(item) for item in value)
        by_subject.setdefault(subject, set()).add(split)
    return any(len(splits) > 1 for splits in by_subject.values() if splits)


def _continuity_break(
    *,
    t_start: np.ndarray,
    epoch_id: np.ndarray,
    qc_ok: np.ndarray,
    selected: np.ndarray,
    step_sec: float,
    tolerance_sec: float,
) -> bool:
    """Check continuity and QC over the selected stable-A windows."""
    selected_indices = np.flatnonzero(selected)
    if selected_indices.size == 0:
        return False
    if not np.all(qc_ok[selected_indices]):
        return True
    if selected_indices.size < 2:
        return False
    start_diffs = np.diff(t_start[selected_indices])
    epoch_diffs = np.diff(epoch_id[selected_indices])
    return bool(
        np.any(np.abs(start_diffs - float(step_sec)) > float(tolerance_sec))
        or np.any(epoch_diffs != 1)
    )


def audit_first_hit_windows(
    *,
    onset_sec: float | None,
    offset_sec: float | None,
    t_start: Sequence[float],
    t_end: Sequence[float],
    epoch_id: Sequence[int] | None = None,
    qc_ok: Sequence[bool] | None = None,
    reaction_coordinate_column_present: bool = False,
    task: str = "ictal",
    protocol: FirstHitEligibilityProtocol | None = None,
    continuity_tolerance_sec: float = 1e-3,
) -> dict[str, Any]:
    """Audit one recording against OD-EPI-002 without estimating ``q``."""
    protocol = protocol or FirstHitEligibilityProtocol()
    starts = np.asarray(t_start, dtype=float).reshape(-1)
    ends = np.asarray(t_end, dtype=float).reshape(-1)
    if starts.size != ends.size:
        raise ValueError("t_start and t_end must have equal lengths")
    if epoch_id is None:
        epochs = np.arange(starts.size, dtype=np.int64)
    else:
        epochs = np.asarray(epoch_id, dtype=np.int64).reshape(-1)
        if epochs.size != starts.size:
            raise ValueError("epoch_id must align with window arrays")
    if qc_ok is None:
        qc = np.ones(starts.size, dtype=bool)
    else:
        qc = np.asarray(qc_ok, dtype=bool).reshape(-1)
        if qc.size != starts.size:
            raise ValueError("qc_ok must align with window arrays")

    order = np.argsort(starts, kind="stable")
    starts = starts[order]
    ends = ends[order]
    epochs = epochs[order]
    qc = qc[order]
    mids = 0.5 * (starts + ends)
    finite_windows = np.isfinite(starts) & np.isfinite(ends) & (ends > starts)
    reasons: list[str] = []
    grid_contract_break = not bool(np.all(finite_windows))
    if np.any(finite_windows):
        durations = ends[finite_windows] - starts[finite_windows]
        grid_contract_break |= bool(
            np.any(
                np.abs(durations - float(protocol.window_sec))
                > float(continuity_tolerance_sec)
            )
        )
    finite_starts = starts[finite_windows]
    finite_epochs = epochs[finite_windows]
    if finite_starts.size > 1:
        grid_contract_break |= bool(
            np.any(
                np.abs(np.diff(finite_starts) - float(protocol.step_sec))
                > float(continuity_tolerance_sec)
            )
        )
        grid_contract_break |= bool(np.any(np.diff(finite_epochs) != 1))
    if grid_contract_break:
        reasons.append("window_grid_contract_violation")
    pair_available = (
        onset_sec is not None
        and offset_sec is not None
        and np.isfinite(float(onset_sec))
        and np.isfinite(float(offset_sec))
        and float(offset_sec) > float(onset_sec)
    )
    if str(task).strip().lower() != "ictal":
        reasons.append("non_ictal_recording_not_transition_segment")
    if not pair_available:
        reasons.append("ordered_onset_offset_pair_required")

    onset_smear = np.zeros(starts.size, dtype=bool)
    stable_a = np.zeros(starts.size, dtype=bool)
    b_core = np.zeros(starts.size, dtype=bool)
    gap_a = False
    gap_b = False
    gap_span = False
    if pair_available:
        onset = float(onset_sec)
        offset = float(offset_sec)
        onset_smear = window_membership_mask(
            t_start=starts,
            t_end=ends,
            t_mid=mids,
            interval=TimeInterval(onset, onset),
            spec=WindowMembershipSpec(mode=protocol.onset_smear_membership),
        )
        stable_a = window_membership_mask(
            t_start=starts,
            t_end=ends,
            t_mid=mids,
            interval=TimeInterval(0.0, onset - protocol.stable_a_buffer_sec),
            spec=WindowMembershipSpec(mode=protocol.core_membership),
        ) & finite_windows
        b_core = window_membership_mask(
            t_start=starts,
            t_end=ends,
            t_mid=mids,
            interval=TimeInterval(
                onset + protocol.stable_a_buffer_sec,
                offset,
            ),
            spec=WindowMembershipSpec(mode=protocol.core_membership),
        ) & finite_windows
        if bool(np.any(onset_smear & (stable_a | b_core))):
            reasons.append("onset_smear_overlaps_core_membership")
        gap_a = _continuity_break(
            t_start=starts,
            epoch_id=epochs,
            qc_ok=qc,
            selected=stable_a,
            step_sec=protocol.step_sec,
            tolerance_sec=continuity_tolerance_sec,
        )
        gap_b = _continuity_break(
            t_start=starts,
            epoch_id=epochs,
            qc_ok=qc,
            selected=b_core,
            step_sec=protocol.step_sec,
            tolerance_sec=continuity_tolerance_sec,
        )
        a_indices = np.flatnonzero(stable_a)
        b_indices = np.flatnonzero(b_core)
        if a_indices.size and b_indices.size:
            span = np.zeros(starts.size, dtype=bool)
            span[a_indices[0] : b_indices[-1] + 1] = True
            gap_span = _continuity_break(
                t_start=starts,
                epoch_id=epochs,
                qc_ok=qc,
                selected=span,
                step_sec=protocol.step_sec,
                tolerance_sec=continuity_tolerance_sec,
            )
        if not bool(np.any(onset_smear)):
            reasons.append("onset_not_aligned_to_window_grid")
        if int(np.sum(stable_a)) < protocol.min_stable_a_windows:
            reasons.append("insufficient_fully_contained_stable_A")
        if max_consecutive_true(stable_a) < protocol.min_stable_a_windows:
            reasons.append("stable_A_windows_not_consecutive")
        if int(np.sum(b_core)) < protocol.min_b_core_windows:
            reasons.append("insufficient_fully_contained_B_core")
        if gap_a:
            reasons.append("gap_or_qc_break_in_A")
        if gap_b:
            reasons.append("gap_or_qc_break_in_B")
        if gap_span:
            reasons.append("gap_or_qc_break_in_A_to_B")

    if not reaction_coordinate_column_present:
        reasons.append("explicit_reaction_coordinate_column_absent")

    stable_indices = np.flatnonzero(stable_a)
    stable_span = 0.0
    if stable_indices.size:
        stable_span = float(ends[stable_indices[-1]] - starts[stable_indices[0]])
    return {
        "task": str(task),
        "onset_sec": float(onset_sec) if onset_sec is not None else None,
        "offset_sec": float(offset_sec) if offset_sec is not None else None,
        "n_windows": int(starts.size),
        "n_onset_smear_windows": int(np.sum(onset_smear)),
        "onset_aligned_to_grid": bool(np.any(onset_smear)),
        "n_stable_A_windows_fully_contained": int(np.sum(stable_a)),
        "max_consecutive_stable_A_windows": max_consecutive_true(stable_a),
        "pre_onset_usable_span_sec": stable_span,
        "n_B_core_windows_fully_contained": int(np.sum(b_core)),
        "gap_or_qc_break_in_A": gap_a,
        "gap_or_qc_break_in_B": gap_b,
        "gap_or_qc_break_in_A_to_B": gap_span,
        "explicit_reaction_coordinate_column_present": bool(
            reaction_coordinate_column_present
        ),
        "is_continuous_first_hit_candidate": not reasons,
        "failure_reasons": reasons,
    }


def finite_quantiles(values: Sequence[float]) -> dict[str, float | int | None]:
    """Return stable descriptive quantiles for an audit report."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "p25": float(np.quantile(finite, 0.25)),
        "median": float(np.quantile(finite, 0.50)),
        "p75": float(np.quantile(finite, 0.75)),
        "max": float(np.max(finite)),
    }


def json_safe(value: Any) -> Any:
    """Convert NumPy/pandas values and NaN to strict JSON-safe values."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, TimeInterval):
        return {"start_sec": value.start_sec, "end_sec": value.end_sec}
    return value
