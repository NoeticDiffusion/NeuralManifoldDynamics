"""Outcome-blind event-block and window scheduling helpers for v3.2 pilots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Mapping


STATE_MARKER_SUFFIX = ": Every 2000 ms"


@dataclass(frozen=True)
class ConditionBlock:
    block_id: str
    condition: str
    start_sample: int
    end_sample_exclusive: int
    start_event_onset_seconds: float

    @property
    def sample_count(self) -> int:
        return self.end_sample_exclusive - self.start_sample

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["sample_count"] = self.sample_count
        return payload


@dataclass(frozen=True)
class CandidateWindow:
    window_id: str
    block_id: str
    condition: str
    start_sample: int
    end_sample_exclusive: int

    @property
    def sample_count(self) -> int:
        return self.end_sample_exclusive - self.start_sample

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["sample_count"] = self.sample_count
        return payload


def reconstruct_condition_blocks(
    events: Iterable[Mapping[str, object]],
    *,
    sample_rate_hz: float,
    recording_sample_count: int,
) -> tuple[ConditionBlock, ...]:
    """Reconstruct instructed EO/EC blocks from the 2-second state markers.

    The 500 ms cues, generic status events and other task markers are not
    treated as state transitions. A block ends at the next distinct state
    marker or at the source recording boundary.
    """
    fs = float(sample_rate_hz)
    if not math.isfinite(fs) or fs <= 0.0:
        raise ValueError("sample_rate_hz must be finite and positive")
    if isinstance(recording_sample_count, bool) or not isinstance(recording_sample_count, int) or recording_sample_count <= 0:
        raise ValueError("recording_sample_count must be a positive integer")
    parsed: list[tuple[float, int, str]] = []
    for source_index, row in enumerate(events):
        trial_type = str(row.get("trial_type", "")).strip()
        if not trial_type.endswith(STATE_MARKER_SUFFIX):
            continue
        lowered = trial_type.casefold()
        if lowered.startswith("eyes open"):
            condition = "EO"
        elif lowered.startswith("eyes closed"):
            condition = "EC"
        else:
            continue
        try:
            onset = float(row.get("onset", "nan"))
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(onset) or onset < 0.0:
            continue
        sample_float = onset * fs
        sample = int(round(sample_float))
        if abs(sample_float - sample) > 0.5000001:
            raise ValueError(f"state event at {onset} seconds is not representable on the sample grid")
        if sample >= recording_sample_count:
            continue
        parsed.append((onset, sample, condition))
    parsed.sort(key=lambda item: (item[0], item[1]))
    transitions: list[tuple[float, int, str]] = []
    previous: str | None = None
    for item in parsed:
        if item[2] != previous:
            transitions.append(item)
            previous = item[2]
    blocks: list[ConditionBlock] = []
    for index, (onset, start_sample, condition) in enumerate(transitions):
        end_sample = transitions[index + 1][1] if index + 1 < len(transitions) else recording_sample_count
        if end_sample <= start_sample:
            continue
        blocks.append(ConditionBlock(
            block_id=f"stateblock-{index + 1:03d}",
            condition=condition,
            start_sample=start_sample,
            end_sample_exclusive=min(end_sample, recording_sample_count),
            start_event_onset_seconds=onset,
        ))
    return tuple(blocks)


def enumerate_full_windows(
    blocks: Iterable[ConditionBlock],
    *,
    sample_rate_hz: float,
    window_seconds: float,
    window_prefix: str = "window",
) -> tuple[CandidateWindow, ...]:
    """Enumerate every nonoverlapping complete window within each state block."""
    fs = float(sample_rate_hz)
    seconds = float(window_seconds)
    if not math.isfinite(fs) or fs <= 0.0 or not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError("sample rate and window duration must be finite and positive")
    exact_samples = seconds * fs
    window_samples = int(round(exact_samples))
    if window_samples < 1 or abs(exact_samples - window_samples) > 1e-9:
        raise ValueError("window duration must lie exactly on the declared sample grid")
    output: list[CandidateWindow] = []
    for block in blocks:
        full_count = block.sample_count // window_samples
        for index in range(full_count):
            start_sample = block.start_sample + index * window_samples
            end_sample = start_sample + window_samples
            output.append(CandidateWindow(
                window_id=f"{block.block_id}-{window_prefix}-{index + 1:03d}",
                block_id=block.block_id,
                condition=block.condition,
                start_sample=start_sample,
                end_sample_exclusive=end_sample,
            ))
    return tuple(output)


__all__ = [
    "CandidateWindow", "ConditionBlock", "enumerate_full_windows",
    "reconstruct_condition_blocks",
]
