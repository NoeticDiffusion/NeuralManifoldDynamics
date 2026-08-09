"""Run-level continuous cardiac and respiratory phase state for Sleep-EAP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from ..phase_anchor import detect_rpeaks_chunked, resp_phase_continuous
from .contracts import PHASE_CONTINUOUS_V1_CONTRACT


@dataclass(frozen=True)
class PhaseContinuousState:
    """A quality-gated continuous phase state on a regular time grid."""

    time_sec: np.ndarray
    phi_cardiac: np.ndarray
    phi_resp: np.ndarray
    cardiac_valid: np.ndarray
    resp_valid: np.ndarray
    rpeaks_sec: np.ndarray
    sample_hz: float
    contract: str = PHASE_CONTINUOUS_V1_CONTRACT


def cardiac_phase_series(time_sec: np.ndarray, rpeaks_sec: np.ndarray) -> np.ndarray:
    """Return RR-interpolated cardiac phase in ``[0, 2π)`` at each time."""
    time = np.asarray(time_sec, dtype=float)
    rpeaks = np.asarray(rpeaks_sec, dtype=float)
    out = np.full(time.shape, np.nan, dtype=np.float64)
    if rpeaks.size < 2:
        return out
    index = np.searchsorted(rpeaks, time, side="right") - 1
    valid = (index >= 0) & (index < rpeaks.size - 1)
    valid_indices = index[valid]
    rr = rpeaks[valid_indices + 1] - rpeaks[valid_indices]
    positive = rr > 0
    output_indices = np.flatnonzero(valid)[positive]
    out[output_indices] = (
        2.0
        * np.pi
        * (time[output_indices] - rpeaks[index[output_indices]])
        / rr[positive]
    )
    return out


def build_phase_continuous_state(
    *,
    ecg_1d: Optional[np.ndarray],
    resp_1d: Optional[np.ndarray],
    sfreq: float,
    phase_anchor_config: Mapping[str, Any] | None = None,
    sample_hz: float = 10.0,
) -> PhaseContinuousState:
    """Build a downsampled, quality-gated phase state from raw recording signals.

    ``sample_hz`` defaults to 10 Hz, which resolves the cardiac cycle at 100 ms
    without exporting full-rate PSG arrays.  The manifest records it explicitly,
    so downstream hazard models cannot assume a different sampling grid.
    """
    if sample_hz <= 0:
        raise ValueError("sample_hz must be positive")
    if sfreq <= 0:
        raise ValueError("sfreq must be positive")

    cfg = dict(phase_anchor_config or {})
    lengths = [len(np.asarray(signal).ravel()) for signal in (ecg_1d, resp_1d) if signal is not None]
    if not lengths:
        empty = np.empty(0, dtype=np.float64)
        return PhaseContinuousState(empty, empty, empty, empty.astype(bool), empty.astype(bool), empty, sample_hz)

    n_samples = min(lengths)
    duration_sec = n_samples / float(sfreq)
    time_sec = np.arange(0.0, duration_sec, 1.0 / float(sample_hz), dtype=np.float64)

    rpeaks_sec = np.empty(0, dtype=np.float64)
    if ecg_1d is not None:
        rpeaks_sec = detect_rpeaks_chunked(np.asarray(ecg_1d, dtype=float).ravel()[:n_samples], sfreq, cfg)
        rpeaks_sec = rpeaks_sec.astype(np.float64) / float(sfreq)
    phi_cardiac = cardiac_phase_series(time_sec, rpeaks_sec)
    cardiac_valid = np.isfinite(phi_cardiac)

    phi_resp = np.full(time_sec.shape, np.nan, dtype=np.float64)
    if resp_1d is not None:
        continuous = resp_phase_continuous(np.asarray(resp_1d, dtype=float).ravel()[:n_samples], sfreq, cfg)
        indices = np.clip(np.rint(time_sec * sfreq).astype(np.int64), 0, max(0, len(continuous) - 1))
        if len(continuous):
            phi_resp = continuous[indices]
    resp_valid = np.isfinite(phi_resp)
    return PhaseContinuousState(
        time_sec=time_sec,
        phi_cardiac=phi_cardiac,
        phi_resp=phi_resp,
        cardiac_valid=cardiac_valid,
        resp_valid=resp_valid,
        rpeaks_sec=rpeaks_sec,
        sample_hz=float(sample_hz),
    )


def sample_phase_state(state: PhaseContinuousState, timestamps_sec: np.ndarray) -> pd.DataFrame:
    """Sample the nearest valid state sample at each requested timestamp."""
    timestamps = np.asarray(timestamps_sec, dtype=float)
    out = pd.DataFrame({"timestamp_sec": timestamps})
    if state.time_sec.size == 0:
        out["phi_cardiac"] = np.nan
        out["phi_resp"] = np.nan
        out["cardiac_phase_valid"] = False
        out["resp_phase_valid"] = False
        return out
    idx = np.searchsorted(state.time_sec, timestamps, side="left")
    idx = np.clip(idx, 0, state.time_sec.size - 1)
    previous = np.maximum(idx - 1, 0)
    use_previous = np.abs(timestamps - state.time_sec[previous]) < np.abs(timestamps - state.time_sec[idx])
    idx = np.where(use_previous, previous, idx)
    out["phi_cardiac"] = state.phi_cardiac[idx]
    out["phi_resp"] = state.phi_resp[idx]
    out["cardiac_phase_valid"] = state.cardiac_valid[idx]
    out["resp_phase_valid"] = state.resp_valid[idx]
    return out


def phase_state_to_frame(state: PhaseContinuousState) -> pd.DataFrame:
    """Serialize a continuous phase state into the ``phase_continuous_v1`` schema."""
    return pd.DataFrame(
        {
            "timestamp_sec": state.time_sec,
            "phi_cardiac": state.phi_cardiac,
            "phi_resp": state.phi_resp,
            "cardiac_phase_valid": state.cardiac_valid.astype(np.int8),
            "resp_phase_valid": state.resp_valid.astype(np.int8),
        }
    )
