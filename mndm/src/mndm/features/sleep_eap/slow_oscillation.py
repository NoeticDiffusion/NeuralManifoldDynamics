"""Slow-oscillation detection with explicit stage and quality provenance."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, hilbert


def _bandpass(signal: np.ndarray, sfreq: float, band_hz: tuple[float, float]) -> np.ndarray:
    nyquist = sfreq / 2.0
    lo = max(float(band_hz[0]) / nyquist, 1e-6)
    hi = min(float(band_hz[1]) / nyquist, 1.0 - 1e-6)
    if not lo < hi:
        raise ValueError("slow-oscillation band is invalid for sampling frequency")
    b, a = butter(3, [lo, hi], btype="bandpass")
    return filtfilt(b, a, signal)


def slow_oscillation_phase(
    eeg: np.ndarray,
    *,
    sfreq: float,
    band_hz: tuple[float, float] = (0.3, 1.25),
) -> np.ndarray:
    """Return the analytic slow-oscillation phase for a mono or averaged EEG."""
    data = np.asarray(eeg, dtype=float)
    if data.ndim == 2:
        data = np.nanmean(data, axis=0)
    if data.ndim != 1 or not np.isfinite(data).all():
        return np.full(data.shape[-1] if data.ndim else 0, np.nan, dtype=float)
    return np.angle(hilbert(_bandpass(data, sfreq, band_hz))).astype(np.float64)


def detect_slow_oscillations(
    eeg: np.ndarray,
    *,
    sfreq: float,
    stage_codes: Sequence[int] | None = None,
    stage_epoch_sec: float = 30.0,
    include_stages: Iterable[int] = (2, 3),
    band_hz: tuple[float, float] = (0.3, 1.25),
    min_peak_to_peak_uv: float = 40.0,
    min_duration_sec: float = 0.8,
    max_duration_sec: float = 2.0,
) -> pd.DataFrame:
    """Detect trough-to-following-up-state slow oscillations from raw EEG.

    Candidate detection is intentionally signal-based, then stage-gated at the
    trough.  This avoids filtering discontinuities at 30-s stage boundaries and
    retains an auditable stage label for every retained event.
    """
    data = np.asarray(eeg, dtype=float)
    if data.ndim == 2:
        data = np.nanmean(data, axis=0)
    if data.ndim != 1 or data.size < max(16, int(3 * sfreq)):
        return pd.DataFrame(columns=["onset_sec", "peak_sec", "upstate_sec", "duration_sec", "amplitude", "event_type", "stage", "source", "so_qc_flag"])
    if not np.isfinite(data).all():
        return pd.DataFrame(columns=["onset_sec", "peak_sec", "upstate_sec", "duration_sec", "amplitude", "event_type", "stage", "source", "so_qc_flag"])

    filtered = _bandpass(data, sfreq, band_hz)
    min_distance = max(1, int(round(min_duration_sec * sfreq)))
    troughs, _ = find_peaks(-filtered, distance=min_distance)
    upstates, _ = find_peaks(filtered, distance=min_distance)
    allowed = {int(stage) for stage in include_stages}
    rows: list[dict[str, object]] = []

    for trough in troughs:
        following = upstates[upstates > trough]
        if following.size == 0:
            continue
        upstate = int(following[0])
        next_troughs = troughs[troughs > upstate]
        if next_troughs.size == 0:
            continue
        duration = (int(next_troughs[0]) - trough) / float(sfreq)
        if not min_duration_sec <= duration <= max_duration_sec:
            continue
        amplitude = float(filtered[upstate] - filtered[trough])
        if not np.isfinite(amplitude) or amplitude < min_peak_to_peak_uv:
            continue
        stage = ""
        if stage_codes is not None:
            index = int((trough / float(sfreq)) // stage_epoch_sec)
            if index < 0 or index >= len(stage_codes):
                continue
            stage_code = int(stage_codes[index])
            if stage_code not in allowed:
                continue
            stage = str(stage_code)
        trough_sec = float(trough / sfreq)
        upstate_sec = float(upstate / sfreq)
        rows.append(
            {
                "onset_sec": trough_sec,
                "peak_sec": upstate_sec,
                "upstate_sec": upstate_sec,
                "duration_sec": duration,
                "amplitude": amplitude,
                "event_type": "slow_oscillation",
                "stage": stage,
                "source": "detector:sleep_eap_so_v1",
                "so_qc_flag": "ok",
            }
        )
    return pd.DataFrame(rows)
