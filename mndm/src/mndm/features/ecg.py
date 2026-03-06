"""
ecg.py
ECG feature extraction (HRV SDNN/RMSSD approximations).

Inputs: signals dict, config dict.
Outputs: DataFrame with per-epoch HRV features.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
from scipy import signal

from . import epoch_selection

logger = logging.getLogger(__name__)


def _resolve_epoch_params(config: Mapping[str, Any], dataset_id: Optional[str]) -> tuple[float, float]:
    return epoch_selection.resolve_epoch_params(config, dataset_id)


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
        logger.exception("ECG stage-stratified selection failed; continuing with full epochs")
        return None


def compute_ecg_features(signals: Mapping[str, Any], config: Mapping[str, Any]) -> pd.DataFrame:
    """Compute per-epoch ECG features (HRV SDNN/RMSSD).
    
    Parameters
    ----------
    signals
        PreprocessedSignals or dict with 'signals' and 'sfreq' keys.
    config
        Configuration with epoching parameters.
    
    Returns
    -------
    DataFrame with columns: epoch_id, ecg_sdnn, ecg_rmssd
    """
    if "ecg" not in signals.get("signals", {}):
        return pd.DataFrame()
    
    ecg_arr = np.asarray(signals["signals"]["ecg"], dtype=float)
    if ecg_arr.ndim == 1:
        ecg_arr = ecg_arr[None, :]
    if ecg_arr.ndim != 2 or ecg_arr.shape[1] <= 0:
        return pd.DataFrame()

    sfreq = float(signals.get("sfreq", 250))
    dataset_id = signals.get("dataset_id")
    raw_file_path = signals.get("file_path")
    
    # Get epoching parameters with dataset overrides
    length_s, step_s = _resolve_epoch_params(config, dataset_id)
    
    # Use first ECG channel
    ecg_channel = np.asarray(ecg_arr[0], dtype=float)

    features_cfg = config.get("features", {}) if isinstance(config, Mapping) else {}
    ecg_cfg = features_cfg.get("ecg", {}) if isinstance(features_cfg, Mapping) else {}
    if not isinstance(ecg_cfg, Mapping):
        ecg_cfg = {}

    bandpass_low_hz = float(ecg_cfg.get("bandpass_low_hz", 5.0) or 5.0)
    bandpass_high_hz = float(ecg_cfg.get("bandpass_high_hz", 20.0) or 20.0)
    bandpass_order = int(ecg_cfg.get("bandpass_order", 3) or 3)
    refractory_s = float(ecg_cfg.get("refractory_s", 0.3) or 0.3)
    prominence_mult = float(ecg_cfg.get("prominence_mult", 1.0) or 1.0)
    rr_min_s = float(ecg_cfg.get("rr_min_s", 0.3) or 0.3)
    rr_max_s = float(ecg_cfg.get("rr_max_s", 2.0) or 2.0)
    min_rr_for_sdnn = int(ecg_cfg.get("min_rr_for_sdnn", 2) or 2)
    min_rr_for_rmssd = int(ecg_cfg.get("min_rr_for_rmssd", 3) or 3)
    
    # Epoch the data
    epoch_length_samples = int(length_s * sfreq)
    epoch_step_samples = int(step_s * sfreq)
    n_samples = int(len(ecg_channel))
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

    # Run-level R-peak detection (boundary-robust vs per-epoch detection).
    nyquist = sfreq * 0.5
    hi = min(bandpass_high_hz, nyquist * 0.99)
    lo = max(0.01, bandpass_low_hz)
    if hi <= lo:
        lo = 5.0
        hi = min(20.0, nyquist * 0.99)
    try:
        b, a = signal.butter(bandpass_order, [lo / nyquist, hi / nyquist], btype="bandpass")
        filtered_full = signal.filtfilt(b, a, ecg_channel)
    except Exception:
        logger.exception("ECG bandpass failed; falling back to demeaned signal for R-peak detection")
        filtered_full = ecg_channel - np.median(ecg_channel)

    centered = filtered_full - np.median(filtered_full)
    sig = np.abs(centered)
    mad = float(np.median(np.abs(centered))) + 1e-8
    robust_sigma = 1.4826 * mad
    prominence = max(1e-6, prominence_mult * robust_sigma)
    min_dist = max(1, int(round(refractory_s * sfreq)))
    peaks, _ = signal.find_peaks(sig, distance=min_dist, prominence=prominence)
    peaks = np.asarray(peaks, dtype=int)

    records: List[Dict[str, Any]] = []
    
    for epoch_idx in range(n_epochs):
        if chosen_epochs is not None and epoch_idx not in chosen_epochs:
            continue
        start_idx = epoch_idx * epoch_step_samples
        end_idx = start_idx + epoch_length_samples
        
        if end_idx > n_samples:
            break

        # Count run-level peaks within this epoch and derive epoch-local RR.
        left = int(np.searchsorted(peaks, start_idx, side="left"))
        right = int(np.searchsorted(peaks, end_idx, side="left"))
        epoch_peaks = peaks[left:right]
        rr_intervals = np.diff(epoch_peaks.astype(float)) / sfreq if epoch_peaks.size >= 2 else np.asarray([], dtype=float)
        if rr_intervals.size:
            rr_intervals = rr_intervals[(rr_intervals >= rr_min_s) & (rr_intervals <= rr_max_s)]

        sdnn = (
            float(np.std(rr_intervals, ddof=1))
            if rr_intervals.size >= max(min_rr_for_sdnn, 2)
            else np.nan
        )
        drr = np.diff(rr_intervals) if rr_intervals.size >= max(min_rr_for_rmssd, 3) else np.asarray([], dtype=float)
        rmssd = float(np.sqrt(np.mean(drr ** 2))) if drr.size > 0 else np.nan
        
        records.append({
            "epoch_id": epoch_idx,
            "t_start": start_idx / sfreq,
            "t_end": end_idx / sfreq,
            "ecg_sdnn": sdnn,
            "ecg_rmssd": rmssd,
            "qc_ok_ecg": bool(np.isfinite(sdnn) and np.isfinite(rmssd)),
        })
    
    df = pd.DataFrame(records)
    logger.info(f"Computed {len(df)} ECG epochs")
    return df


