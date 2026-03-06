"""
fmri_continuous.py
Continuous-session preprocessing helpers for fMRI ROI time series."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def process_session_signals(
    roi_ts: np.ndarray,
    sfreq: float,
    config: Mapping[str, Any] | None,
) -> Dict[str, np.ndarray]:
    """Apply continuous-domain transforms before epoch slicing.

    Parameters
    ----------
    roi_ts
        ROI time series with shape ``(n_regions, n_times)``.
    sfreq
        Sampling frequency in Hz.
    config
        Optional preprocessing configuration. Supported keys:
        - ``f_low`` / ``f_high`` under this mapping
        - ``bandpass`` as ``[f_low, f_high]``
        - ``compute_phase`` (default: True)

    Returns
    -------
    dict
        Contains:
        - ``filtered_ts``: bandpass-filtered series (or raw series if filter invalid)
        - ``phase_ts``: phase angles from Hilbert transform over the full session
    """
    if roi_ts.ndim != 2:
        raise ValueError("roi_ts must be 2-D (n_regions, n_times)")
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("sfreq must be finite and positive")

    cfg = config if isinstance(config, Mapping) else {}
    band = cfg.get("bandpass")
    if isinstance(band, (list, tuple)) and len(band) >= 2:
        f_low = float(band[0])
        f_high = float(band[1])
    else:
        f_low = float(cfg.get("f_low", 0.01))
        f_high = float(cfg.get("f_high", 0.1))
    compute_phase = bool(cfg.get("compute_phase", True))

    filtered_ts = _bandpass_signal(roi_ts, sfreq=sfreq, f_low=f_low, f_high=f_high)
    result: Dict[str, np.ndarray] = {"filtered_ts": filtered_ts}

    if compute_phase:
        analytic_signal = hilbert(filtered_ts, axis=1)
        result["phase_ts"] = np.angle(analytic_signal)
    else:
        result["phase_ts"] = np.full_like(filtered_ts, np.nan, dtype=float)

    return result


def _bandpass_signal(
    data: np.ndarray,
    sfreq: float,
    f_low: float,
    f_high: float,
    order: int = 4,
) -> np.ndarray:
    """Bandpass filter full session; raise on invalid filter settings."""
    nyq = sfreq / 2.0
    if not (np.isfinite(f_low) and np.isfinite(f_high) and 0 < f_low < f_high < nyq):
        raise ValueError(
            "Invalid bandpass parameters for given sfreq: "
            f"sfreq={sfreq}, nyq={nyq}, f_low={f_low}, f_high={f_high}"
        )
    b, a = butter(order, [f_low / nyq, f_high / nyq], btype="band")
    return filtfilt(b, a, np.asarray(data, dtype=float), axis=1)

