"""Source-level BOAS first-hit eligibility helpers.

The functions in this module are deliberately estimator-free. They validate
source stage intervals, count first-hit outcomes, and calculate the
predeclared external headband reaction-coordinate values used only by the
OD-SLP-000 eligibility audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

STAGE_WAKE = 0
STAGE_N1 = 1
STAGE_N2 = 2
STAGE_N3 = 3
STAGE_REM = 4
STAGE_DISCONNECTION = 8
STAGE_ARTIFACT = -2

STAGE_NAMES = {
    STAGE_WAKE: "wake",
    STAGE_N1: "n1",
    STAGE_N2: "n2",
    STAGE_N3: "n3",
    STAGE_REM: "rem",
    STAGE_DISCONNECTION: "disconnection",
    STAGE_ARTIFACT: "artifact",
}


@dataclass(frozen=True)
class SleepFirstHitProtocol:
    """Frozen OD-SLP-000 source-level contract."""

    protocol_id: str = "OD-SLP-000-first-hit-v1"
    dataset_id: str = "ds005555"
    stage_column: str = "stage_hum"
    n2_code: int = STAGE_N2
    n3_code: int = STAGE_N3
    rem_code: int = STAGE_REM
    window_tolerance_sec: float = 1.0
    synchronization_tolerance_sec: float = 1.0
    terminal_censor_max_sec: float = 30.0
    minimum_eligible_segments: int = 20
    minimum_eligible_pids: int = 10
    minimum_stratum_segments: int = 5
    rc_name: str = "headband_slow_fast_logratio"
    rc_epsilon: float = 1e-12
    rc_channels: tuple[str, str] = ("HB_1", "HB_2")
    delta_band_hz: tuple[float, float] = (1.0, 4.0)
    alpha_band_hz: tuple[float, float] = (8.0, 12.0)
    beta_band_hz: tuple[float, float] = (13.0, 30.0)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe protocol representation."""
        return {
            "protocol_id": self.protocol_id,
            "dataset_id": self.dataset_id,
            "stage_column": self.stage_column,
            "n2_code": self.n2_code,
            "n3_code": self.n3_code,
            "rem_code": self.rem_code,
            "window_tolerance_sec": self.window_tolerance_sec,
            "synchronization_tolerance_sec": self.synchronization_tolerance_sec,
            "terminal_censor_max_sec": self.terminal_censor_max_sec,
            "minimum_eligible_segments": self.minimum_eligible_segments,
            "minimum_eligible_pids": self.minimum_eligible_pids,
            "minimum_stratum_segments": self.minimum_stratum_segments,
            "rc_name": self.rc_name,
            "rc_epsilon": self.rc_epsilon,
            "rc_channels": list(self.rc_channels),
            "delta_band_hz": list(self.delta_band_hz),
            "alpha_band_hz": list(self.alpha_band_hz),
            "beta_band_hz": list(self.beta_band_hz),
        }


def normalize_stage(value: Any) -> int | None:
    """Normalize numeric or codebook stage values without AI fallback."""
    if value is None:
        return None
    try:
        numeric = float(value)
        if np.isfinite(numeric) and numeric.is_integer():
            code = int(numeric)
            return code if code in STAGE_NAMES else None
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    aliases = {
        "wake": STAGE_WAKE,
        "n1": STAGE_N1,
        "n2": STAGE_N2,
        "n3": STAGE_N3,
        "rem": STAGE_REM,
        "psg_disconnection": STAGE_DISCONNECTION,
        "disconnection": STAGE_DISCONNECTION,
        "artifact_or_missing": STAGE_ARTIFACT,
        "artifact": STAGE_ARTIFACT,
    }
    return aliases.get(text)


def _as_intervals(
    onsets_sec: Sequence[float],
    durations_sec: Sequence[float],
    stages: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Normalize source rows and return ordered interval arrays."""
    onsets = np.asarray(onsets_sec, dtype=float).reshape(-1)
    durations = np.asarray(durations_sec, dtype=float).reshape(-1)
    if onsets.size != durations.size or onsets.size != len(stages):
        raise ValueError("stage interval columns must have equal lengths")
    codes = np.asarray(
        [normalize_stage(value) for value in stages], dtype=object
    )
    reasons: list[str] = []
    if onsets.size == 0:
        reasons.append("stage_hum_empty")
    if not np.all(np.isfinite(onsets)):
        reasons.append("stage_onset_nonfinite")
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0):
        reasons.append("stage_intervals_not_positive_duration")
    if any(value is None for value in codes):
        reasons.append("stage_hum_unmapped")
    numeric_codes = np.asarray(
        [int(value) if value is not None else -999 for value in codes],
        dtype=np.int16,
    )
    order = np.argsort(onsets, kind="stable")
    return (
        onsets[order],
        durations[order],
        numeric_codes[order],
        reasons,
    )


def audit_stage_intervals(
    *,
    onsets_sec: Sequence[float],
    durations_sec: Sequence[float],
    stages: Sequence[Any],
    recording_duration_sec: float | None,
    protocol: SleepFirstHitProtocol | None = None,
) -> dict[str, Any]:
    """Audit stage intervals and count first-hit outcomes."""
    protocol = protocol or SleepFirstHitProtocol()
    onsets, durations, codes, reasons = _as_intervals(
        onsets_sec, durations_sec, stages
    )
    ends = onsets + durations
    if onsets.size:
        gaps = np.diff(onsets) - durations[:-1]
        if np.any(gaps > protocol.window_tolerance_sec):
            reasons.append("stage_grid_gap")
        if np.any(gaps < -protocol.window_tolerance_sec):
            reasons.append("stage_grid_overlap")
        if abs(float(onsets[0])) > protocol.window_tolerance_sec:
            reasons.append("stage_grid_does_not_start_at_zero")
        if recording_duration_sec is None or not np.isfinite(
            float(recording_duration_sec)
        ):
            reasons.append("recording_duration_missing")
        else:
            terminal_tail = float(recording_duration_sec) - float(ends[-1])
            if terminal_tail < -protocol.window_tolerance_sec:
                reasons.append("stage_grid_exceeds_recording")
            elif terminal_tail > protocol.terminal_censor_max_sec:
                reasons.append("stage_grid_does_not_cover_recording")
    else:
        reasons.append("stage_grid_empty")

    dense_grid_available = not reasons
    n3_mask = codes == protocol.n3_code
    rem_mask = codes == protocol.rem_code
    n2_mask = codes == protocol.n2_code
    candidates: list[int] = []
    if n2_mask.size:
        starts = np.flatnonzero(n2_mask)
        if starts.size:
            candidates = [
                int(index)
                for index in starts
                if index == 0 or not n2_mask[index - 1]
            ]

    outcome_counts = {
        "first_hit_n3": 0,
        "first_hit_rem": 0,
        "competing_exit_wake": 0,
        "competing_exit_n1": 0,
        "competing_exit_disconnection": 0,
        "competing_exit_artifact": 0,
        "qc_or_gap_exit": 0,
        "right_censored": 0,
    }
    segments: list[dict[str, Any]] = []
    for candidate_index in candidates:
        outcome = "right_censored"
        end_index = len(codes) - 1
        for index in range(candidate_index + 1, len(codes)):
            code = int(codes[index])
            if code == protocol.n3_code:
                outcome = "first_hit_n3"
                end_index = index
                break
            if code == protocol.rem_code:
                outcome = "first_hit_rem"
                end_index = index
                break
            if code == STAGE_WAKE:
                outcome = "competing_exit_wake"
                end_index = index
                break
            if code == STAGE_N1:
                outcome = "competing_exit_n1"
                end_index = index
                break
            if code == STAGE_DISCONNECTION:
                outcome = "competing_exit_disconnection"
                end_index = index
                break
            if code == STAGE_ARTIFACT:
                outcome = "competing_exit_artifact"
                end_index = index
                break
            if index > 0 and (
                onsets[index] - ends[index - 1] > protocol.window_tolerance_sec
            ):
                outcome = "qc_or_gap_exit"
                end_index = index
                break
        outcome_counts[outcome] += 1
        segments.append(
            {
                "candidate_interval_index": candidate_index,
                "outcome_interval_index": end_index,
                "start_sec": float(onsets[candidate_index]),
                "end_sec": float(ends[end_index]),
                "outcome": outcome,
            }
        )

    return {
        "dense_grid_available": bool(dense_grid_available),
        "terminal_unstaged_sec": (
            max(
                0.0,
                float(recording_duration_sec) - float(ends[-1]),
            )
            if onsets.size
            and recording_duration_sec is not None
            and np.isfinite(float(recording_duration_sec))
            else None
        ),
        "stage_hum_available": bool(onsets.size and not any(
            reason in {"stage_hum_empty", "stage_hum_unmapped"}
            for reason in reasons
        )),
        "n3_core_windows": int(np.sum(n3_mask)),
        "rem_core_windows": int(np.sum(rem_mask)),
        "n2_interior_windows": int(np.sum(n2_mask)),
        "candidate_n2_blocks": int(len(candidates)),
        **outcome_counts,
        "failure_reasons": list(dict.fromkeys(reasons)),
        "segments": segments,
        "interval_onsets_sec": onsets.tolist(),
        "interval_ends_sec": ends.tolist(),
        "interval_stages": codes.tolist(),
    }


def headband_slow_fast_logratio(
    data: np.ndarray,
    *,
    sampling_frequency_hz: float,
    intervals: Sequence[tuple[float, float]],
    protocol: SleepFirstHitProtocol | None = None,
) -> np.ndarray:
    """Calculate the frozen headband-only scalar for source intervals."""
    protocol = protocol or SleepFirstHitProtocol()
    values = np.full(len(intervals), np.nan, dtype=float)
    array = np.asarray(data, dtype=float)
    if array.ndim != 2 or array.shape[0] != 2:
        raise ValueError("headband data must have shape [2, samples]")
    if not np.isfinite(float(sampling_frequency_hz)) or sampling_frequency_hz <= 0:
        return values
    fs = float(sampling_frequency_hz)
    for index, (start_sec, end_sec) in enumerate(intervals):
        start = max(0, int(round(float(start_sec) * fs)))
        stop = min(array.shape[1], int(round(float(end_sec) * fs)))
        if stop <= start:
            continue
        segment = array[:, start:stop]
        if not np.all(np.isfinite(segment)):
            continue
        segment = segment - np.mean(segment, axis=1, keepdims=True)
        taper = np.hanning(segment.shape[1])
        if not np.any(taper):
            continue
        spectrum = np.fft.rfft(segment * taper[None, :], axis=1)
        power = (np.abs(spectrum) ** 2) / max(float(np.sum(taper**2)), 1.0)
        frequencies = np.fft.rfftfreq(segment.shape[1], d=1.0 / fs)

        def band_power(bounds: tuple[float, float]) -> np.ndarray:
            mask = (frequencies >= bounds[0]) & (frequencies < bounds[1])
            if not np.any(mask):
                return np.full(2, np.nan)
            return np.mean(power[:, mask], axis=1)

        delta = band_power(protocol.delta_band_hz)
        alpha = band_power(protocol.alpha_band_hz)
        beta = band_power(protocol.beta_band_hz)
        if not np.all(np.isfinite([delta, alpha, beta])):
            continue
        delta_mean = float(np.mean(delta))
        denominator = float(np.mean(alpha + beta))
        values[index] = np.log(
            (delta_mean + protocol.rc_epsilon)
            / (denominator + protocol.rc_epsilon)
        )
    return values


def pid_split_leaks(assignments: Mapping[str, str]) -> bool:
    """Return true if one person appears in multiple partition labels."""
    by_pid: dict[str, set[str]] = {}
    for pid, split in assignments.items():
        if pid is None or str(pid).strip() == "":
            continue
        if isinstance(split, (list, tuple, set)):
            values = {str(value) for value in split}
        else:
            values = {str(split)}
        by_pid.setdefault(str(pid), set()).update(values)
    return any(len(values) > 1 for values in by_pid.values())


def json_safe(value: Any) -> Any:
    """Convert NumPy values and non-finite floats to strict JSON values."""
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
    if isinstance(value, np.bool_):
        return bool(value)
    return value
