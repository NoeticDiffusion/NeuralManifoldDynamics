"""REM-restricted continuous theta carrier for stage-general Sleep-EAP exports."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def rem_theta_phase(
    eeg: np.ndarray,
    *,
    sfreq: float,
    stage_codes: Sequence[int],
    stage_epoch_sec: float = 30.0,
    rem_stage_code: int = 4,
    band_hz: tuple[float, float] = (4.0, 8.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Compute theta phase and validity mask, retaining only REM samples."""
    data = np.asarray(eeg, dtype=float)
    if data.ndim == 2:
        data = np.nanmean(data, axis=0)
    if data.ndim != 1 or data.size == 0 or not np.isfinite(data).all():
        return np.full(data.shape[-1] if data.ndim else 0, np.nan), np.zeros(data.shape[-1] if data.ndim else 0, dtype=bool)
    nyquist = sfreq / 2.0
    lo, hi = float(band_hz[0]) / nyquist, float(band_hz[1]) / nyquist
    if not 0.0 < lo < hi < 1.0:
        raise ValueError("REM theta band is invalid for sampling frequency")
    b, a = butter(3, [lo, hi], btype="bandpass")
    phase = np.angle(hilbert(filtfilt(b, a, data))).astype(np.float64)
    sample_times = np.arange(data.size, dtype=float) / float(sfreq)
    epoch_indices = (sample_times // float(stage_epoch_sec)).astype(int)
    stage_array = np.asarray(stage_codes, dtype=int)
    valid = np.zeros(data.size, dtype=bool)
    in_bounds = epoch_indices < stage_array.size
    valid[in_bounds] = stage_array[epoch_indices[in_bounds]] == int(rem_stage_code)
    phase[~valid] = np.nan
    return phase, valid
