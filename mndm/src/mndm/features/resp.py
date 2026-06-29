"""Respiration v0.1 feature extraction.

Extracts per-epoch respiratory features from a ``resp``-typed channel.

Feature families
----------------
Core kinematic features
    resp_rate_bpm, resp_period_s, resp_period_cv,
    resp_amplitude_median, resp_amplitude_cv,
    resp_peak_count, resp_trough_count

Regularity and phase features
    resp_regular_index, resp_pause_fraction,
    resp_inhale_fraction, resp_exhale_fraction,
    resp_phase_mean, resp_phase_consistency

Signal quality
    resp_signal_quality, resp_source_channel, qc_ok_resp

Derived anchor-relevant indices
    resp_slowing_index, resp_depth_index, resp_anchor_index

Note
----
These are computed per MNPS epoch window, not per individual breath cycle.
Phase-resolved cardiorespiratory coupling lives in ``cardioresp.py``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

from . import epoch_selection

logger = logging.getLogger(__name__)


def _resolve_epoch_params(
    config: Mapping[str, Any], dataset_id: Optional[str]
) -> tuple[float, float]:
    return epoch_selection.resolve_epoch_params(config, dataset_id)


def _resolve_resp_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    features_cfg = config.get("features", {}) if isinstance(config, Mapping) else {}
    resp_cfg = features_cfg.get("resp", {}) if isinstance(features_cfg, Mapping) else {}
    out: Dict[str, Any] = dict(resp_cfg) if isinstance(resp_cfg, Mapping) else {}
    out.setdefault("lowpass_hz", 1.0)
    out.setdefault("peak_min_distance_s", 1.5)
    out.setdefault("peak_min_prominence_frac", 0.15)
    out.setdefault("pause_flatness_frac", 0.05)
    out.setdefault("min_breaths", 2)
    out.setdefault("min_signal_quality", 0.2)
    return out


def _resolve_chosen_epochs(
    *,
    config: Mapping[str, Any],
    dataset_id: Optional[str],
    raw_file_path: Optional[str],
    sfreq: float,
    n_samples: int,
    step_s: float,
    epoch_length_samples: int,
    epoch_step_samples: int,
) -> Optional[set[int]]:
    if not raw_file_path:
        return None
    try:
        return epoch_selection.resolve_stage_stratified_epoch_set(
            config=config,
            dataset_id=dataset_id,
            raw_file_path=raw_file_path,
            sfreq=float(sfreq),
            n_samples=int(n_samples),
            step_s=float(step_s),
            epoch_length_samples=int(epoch_length_samples),
            epoch_step_samples=int(epoch_step_samples),
        )
    except Exception:
        logger.exception("RESP stage-stratified selection failed; continuing with full epochs")
        return None


def _signal_quality(resp_1d: np.ndarray, sfreq: float) -> float:
    """Simple quality score: variance explained by the 0.05–1 Hz band.

    Returns a value in [0, 1].  Low values indicate a flat or heavily
    noise-dominated channel.
    """
    try:
        nyq = sfreq * 0.5
        lo, hi = 0.05, min(1.0, nyq * 0.95)
        if hi <= lo or len(resp_1d) < int(sfreq * 3):
            return 0.0
        b, a = sp_signal.butter(3, [lo / nyq, hi / nyq], btype="bandpass")
        filt = sp_signal.filtfilt(b, a, resp_1d)
        total_var = float(np.var(resp_1d))
        if total_var < 1e-12:
            return 0.0
        filtered_var = float(np.var(filt))
        return float(np.clip(filtered_var / total_var, 0.0, 1.0))
    except Exception:
        return 0.0


def _circular_mean_and_consistency(phases: np.ndarray) -> tuple[float, float]:
    """Compute circular mean (rad, -π..π) and mean resultant length (0..1)."""
    if len(phases) == 0:
        return float("nan"), float("nan")
    sin_mean = float(np.mean(np.sin(phases)))
    cos_mean = float(np.mean(np.cos(phases)))
    mean_phase = float(np.arctan2(sin_mean, cos_mean))
    consistency = float(np.sqrt(sin_mean**2 + cos_mean**2))
    return mean_phase, consistency


def _extract_resp_features_epoch(
    epoch: np.ndarray,
    sfreq: float,
    *,
    lowpass_hz: float,
    peak_min_distance_s: float,
    peak_min_prominence_frac: float,
    pause_flatness_frac: float,
    min_breaths: int,
    source_channel: str,
) -> Dict[str, Any]:
    """Extract respiration features from one epoch array."""
    epoch_s = len(epoch) / sfreq
    nan_row: Dict[str, Any] = {
        "resp_rate_bpm": np.nan,
        "resp_period_s": np.nan,
        "resp_period_cv": np.nan,
        "resp_amplitude_median": np.nan,
        "resp_amplitude_cv": np.nan,
        "resp_regular_index": np.nan,
        "resp_pause_fraction": np.nan,
        "resp_inhale_fraction": np.nan,
        "resp_exhale_fraction": np.nan,
        "resp_phase_mean": np.nan,
        "resp_phase_consistency": np.nan,
        "resp_signal_quality": np.nan,
        "resp_peak_count": 0,
        "resp_trough_count": 0,
        "resp_source_channel": source_channel,
        "qc_ok_resp": False,
        "resp_slowing_index": np.nan,
        "resp_depth_index": np.nan,
        "resp_anchor_index": np.nan,
    }

    sq = _signal_quality(epoch, sfreq)
    nan_row["resp_signal_quality"] = float(sq)
    if sq < pause_flatness_frac:
        return nan_row

    # Low-pass filter
    try:
        nyq = sfreq * 0.5
        hi = min(lowpass_hz, nyq * 0.95)
        b, a = sp_signal.butter(3, hi / nyq, btype="low")
        filt = sp_signal.filtfilt(b, a, epoch)
    except Exception:
        filt = epoch.copy()

    filt = filt - np.mean(filt)

    # Peak/trough detection
    signal_range = float(np.max(filt) - np.min(filt))
    if signal_range < 1e-10:
        return nan_row

    prominence_threshold = max(1e-8, peak_min_prominence_frac * signal_range)
    min_distance_samples = max(1, int(peak_min_distance_s * sfreq))

    peaks, peak_props = sp_signal.find_peaks(
        filt,
        distance=min_distance_samples,
        prominence=prominence_threshold,
    )
    troughs, _ = sp_signal.find_peaks(
        -filt,
        distance=min_distance_samples,
        prominence=prominence_threshold,
    )

    n_peaks = len(peaks)
    n_troughs = len(troughs)
    nan_row["resp_peak_count"] = int(n_peaks)
    nan_row["resp_trough_count"] = int(n_troughs)

    if n_peaks < min_breaths:
        return nan_row

    # Breath periods from inter-peak intervals
    ipi = np.diff(peaks) / sfreq  # seconds between successive peaks
    period_mean = float(np.mean(ipi)) if len(ipi) > 0 else float("nan")
    period_cv = (
        float(np.std(ipi) / (np.mean(ipi) + 1e-8)) if len(ipi) > 0 else float("nan")
    )
    rate_bpm = 60.0 / period_mean if period_mean > 0 else float("nan")

    # Amplitudes at detected peaks
    amplitudes = filt[peaks] - np.min(filt)
    amp_median = float(np.median(amplitudes))
    amp_cv = float(np.std(amplitudes) / (np.mean(amplitudes) + 1e-8))

    # Pause fraction: samples where |signal| < flatness threshold
    flatness_thr = max(1e-8, pause_flatness_frac * signal_range)
    pause_frac = float(np.mean(np.abs(filt) < flatness_thr))

    # Inhale/exhale fractions from zero-crossing analysis
    pos_frac = float(np.mean(filt > 0))
    neg_frac = float(np.mean(filt < 0))

    # Instantaneous phase via Hilbert transform
    try:
        analytic = sp_signal.hilbert(filt)
        inst_phase = np.angle(analytic)  # -π..π
        phase_mean, phase_consistency = _circular_mean_and_consistency(inst_phase)
    except Exception:
        phase_mean, phase_consistency = float("nan"), float("nan")

    # Regularity: combination of period stability and phase consistency
    period_cv_clamped = float(np.clip(period_cv, 0.0, 5.0)) if np.isfinite(period_cv) else 2.0
    phase_cons = float(phase_consistency) if np.isfinite(phase_consistency) else 0.0
    # Higher = more regular: low period_cv and high phase consistency
    regular_index = (1.0 / (1.0 + period_cv_clamped)) + phase_cons / 2.0

    # Derived indices (signed, on the same relative scale):
    # resp_slowing_index: high rate → negative
    slowing_index = float(-rate_bpm / 20.0) if np.isfinite(rate_bpm) else float("nan")
    # resp_depth_index: amplitude proxy
    depth_index = float(amp_median / (signal_range + 1e-8))
    # resp_anchor_index: regularity + depth − rate_normalized
    if np.isfinite(regular_index) and np.isfinite(depth_index) and np.isfinite(slowing_index):
        anchor_index = regular_index + depth_index + slowing_index
    else:
        anchor_index = float("nan")

    return {
        "resp_rate_bpm": float(rate_bpm),
        "resp_period_s": float(period_mean),
        "resp_period_cv": float(period_cv),
        "resp_amplitude_median": float(amp_median),
        "resp_amplitude_cv": float(amp_cv),
        "resp_regular_index": float(regular_index),
        "resp_pause_fraction": float(pause_frac),
        "resp_inhale_fraction": float(pos_frac),
        "resp_exhale_fraction": float(neg_frac),
        "resp_phase_mean": float(phase_mean),
        "resp_phase_consistency": float(phase_consistency),
        "resp_signal_quality": float(sq),
        "resp_peak_count": int(n_peaks),
        "resp_trough_count": int(n_troughs),
        "resp_source_channel": source_channel,
        "qc_ok_resp": bool(np.isfinite(rate_bpm) and sq >= pause_flatness_frac),
        "resp_slowing_index": float(slowing_index),
        "resp_depth_index": float(depth_index),
        "resp_anchor_index": float(anchor_index),
    }


def compute_resp_features(
    signals: Mapping[str, Any], config: Mapping[str, Any]
) -> pd.DataFrame:
    """Compute per-epoch Respiration v0.1 features.

    Parameters
    ----------
    signals:
        Preprocessed signals dict with keys ``signals``, ``sfreq``,
        ``channels``, ``dataset_id``, ``file_path``.
    config:
        Pipeline configuration dict.

    Returns
    -------
    pd.DataFrame
        Per-epoch respiration features.  Empty DataFrame when no ``resp``
        channel is available.
    """
    if "resp" not in signals.get("signals", {}):
        return pd.DataFrame()

    resp_data = signals["signals"].get("resp", None)
    if resp_data is None:
        return pd.DataFrame()

    resp_arr = np.asarray(resp_data, dtype=float)
    if resp_arr.ndim == 1:
        resp_arr = resp_arr[None, :]
    if resp_arr.ndim != 2 or resp_arr.shape[1] <= 0:
        return pd.DataFrame()

    sfreq = float(signals.get("sfreq", 250))
    dataset_id = signals.get("dataset_id")
    raw_file_path = signals.get("file_path")

    length_s, step_s = _resolve_epoch_params(config, dataset_id)
    rcfg = _resolve_resp_config(config)

    epoch_length_samples = int(length_s * sfreq)
    epoch_step_samples = int(step_s * sfreq)
    n_samples = resp_arr.shape[1]

    if epoch_length_samples <= 0 or epoch_step_samples <= 0 or n_samples < epoch_length_samples:
        return pd.DataFrame()

    n_epochs = (n_samples - epoch_length_samples) // epoch_step_samples + 1

    chosen_epochs = _resolve_chosen_epochs(
        config=config,
        dataset_id=dataset_id,
        raw_file_path=raw_file_path,
        sfreq=sfreq,
        n_samples=n_samples,
        step_s=float(step_s),
        epoch_length_samples=epoch_length_samples,
        epoch_step_samples=epoch_step_samples,
    )

    # Resolve channel names for provenance
    channels = signals.get("channels", {})
    resp_channel_names: List[str] = []
    if isinstance(channels, Mapping):
        resp_channel_names = channels.get("resp", []) or []
    if isinstance(resp_channel_names, str):
        resp_channel_names = [resp_channel_names]

    source_channel = str(resp_channel_names[0]) if resp_channel_names else "resp_0"

    # Use first RESP channel
    resp_1d = resp_arr[0]

    records: List[Dict[str, Any]] = []

    for epoch_idx in range(n_epochs):
        if chosen_epochs is not None and epoch_idx not in chosen_epochs:
            continue

        start_idx = epoch_idx * epoch_step_samples
        end_idx = start_idx + epoch_length_samples
        if end_idx > n_samples:
            break

        epoch = resp_1d[start_idx:end_idx]
        feats = _extract_resp_features_epoch(
            epoch,
            sfreq,
            lowpass_hz=float(rcfg["lowpass_hz"]),
            peak_min_distance_s=float(rcfg["peak_min_distance_s"]),
            peak_min_prominence_frac=float(rcfg["peak_min_prominence_frac"]),
            pause_flatness_frac=float(rcfg["pause_flatness_frac"]),
            min_breaths=int(rcfg["min_breaths"]),
            source_channel=source_channel,
        )
        feats["epoch_id"] = epoch_idx
        feats["t_start"] = start_idx / sfreq
        feats["t_end"] = end_idx / sfreq
        records.append(feats)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    logger.info(
        "RESP v0.1: computed %d epochs; qc_ok fraction=%.2f",
        len(df),
        float(df["qc_ok_resp"].mean()) if "qc_ok_resp" in df.columns else 0.0,
    )
    return df
