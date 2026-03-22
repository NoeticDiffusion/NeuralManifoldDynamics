"""EOG feature extraction (blink detection and rate).

Inputs: signals dict, config dict.
Outputs: DataFrame with per-epoch blink rates.
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
    """Resolve epoching length/step with optional per-dataset overrides."""
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
    """Return selected epoch ids from stage-stratified policy, if enabled."""
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
        logger.exception("EOG stage-stratified selection failed; continuing with full epochs")
        return None


def compute_eog_features(signals: Mapping[str, Any], config: Mapping[str, Any]) -> pd.DataFrame:
    """Compute per-epoch EOG features (blink rate).

    Args:
        signals: Preprocessed signals dict with ``signals`` and ``sfreq``.
        config: Configuration with epoching and optional ``features.eog``.

    Returns:
        DataFrame with ``epoch_id``, ``eog_blink_rate``, etc.
    """
    if "eog" not in signals.get("signals", {}):
        return pd.DataFrame()
    
    eog_arr = np.asarray(signals["signals"]["eog"], dtype=float)
    if eog_arr.ndim == 1:
        eog_arr = eog_arr[None, :]
    if eog_arr.ndim != 2 or eog_arr.shape[1] <= 0:
        return pd.DataFrame()
    sfreq = signals.get("sfreq", 250)
    dataset_id = signals.get("dataset_id")
    raw_file_path = signals.get("file_path")
    
    # Get epoching parameters (with dataset overrides)
    length_s, step_s = _resolve_epoch_params(config, dataset_id)
    features_cfg = config.get("features", {}) if isinstance(config, Mapping) else {}
    eog_cfg = features_cfg.get("eog", {}) if isinstance(features_cfg, Mapping) else {}
    if not isinstance(eog_cfg, Mapping):
        eog_cfg = {}
    filter_low_hz = float(eog_cfg.get("filter_low_hz", 0.5) or 0.5)
    filter_high_hz = float(eog_cfg.get("filter_high_hz", 10.0) or 10.0)
    filter_order = int(eog_cfg.get("filter_order", 3) or 3)
    std_mult = float(eog_cfg.get("std_mult", 2.0) or 2.0)
    mad_mult = float(eog_cfg.get("mad_mult", 4.0) or 4.0)
    prominence_mult = float(eog_cfg.get("prominence_mult", 1.0) or 1.0)
    refractory_s = float(eog_cfg.get("refractory_s", 0.25) or 0.25)
    
    # Use first EOG channel if multiple
    eog_channel = eog_arr[0]
    
    # Epoch the data
    epoch_length_samples = int(length_s * sfreq)
    epoch_step_samples = int(step_s * sfreq)
    n_samples = int(len(eog_channel))
    if epoch_length_samples <= 0 or epoch_step_samples <= 0 or n_samples < epoch_length_samples:
        return pd.DataFrame()
    n_epochs = (n_samples - epoch_length_samples) // epoch_step_samples + 1
    chosen_epochs = _resolve_chosen_epochs(
        config=config,
        dataset_id=dataset_id,
        raw_file_path=raw_file_path,
        sfreq=float(sfreq),
        n_samples=n_samples,
        step_s=float(step_s),
        epoch_length_samples=epoch_length_samples,
        epoch_step_samples=epoch_step_samples,
    )

    # Run-level blink detection (more stable than per-epoch peak finding).
    nyquist = float(sfreq) * 0.5
    hi = min(filter_high_hz, nyquist * 0.99)
    lo = max(0.01, filter_low_hz)
    if hi <= lo:
        lo = 0.5
        hi = min(10.0, nyquist * 0.99)
    try:
        b, a = signal.butter(filter_order, [lo / nyquist, hi / nyquist], btype="bandpass")
        filtered_full = signal.filtfilt(b, a, eog_channel)
    except Exception:
        logger.exception("EOG bandpass failed; falling back to demeaned signal for blink detection")
        filtered_full = eog_channel - np.median(eog_channel)

    centered = filtered_full - np.median(filtered_full)
    abs_sig = np.abs(centered)
    mad = float(np.median(np.abs(centered))) + 1e-8
    std = float(np.std(filtered_full))
    thr = max(std_mult * std, mad_mult * mad)
    min_dist = max(1, int(round(refractory_s * float(sfreq))))
    prominence = max(1e-8, prominence_mult * mad)
    peaks, _ = signal.find_peaks(abs_sig, height=thr, distance=min_dist, prominence=prominence)
    peaks = np.asarray(peaks, dtype=int)

    records: List[Dict[str, Any]] = []
    
    for epoch_idx in range(n_epochs):
        if chosen_epochs is not None and epoch_idx not in chosen_epochs:
            continue
        start_idx = epoch_idx * epoch_step_samples
        end_idx = start_idx + epoch_length_samples
        
        if end_idx > n_samples:
            break

        # Count run-detected peaks inside this epoch interval.
        left = int(np.searchsorted(peaks, start_idx, side="left"))
        right = int(np.searchsorted(peaks, end_idx, side="left"))
        n_blinks = max(0, right - left)
        blink_rate = float(n_blinks / max(float(length_s), 1e-6))
        
        records.append({
            "epoch_id": epoch_idx,
            "t_start": start_idx / float(sfreq),
            "t_end": end_idx / float(sfreq),
            "eog_blink_rate": blink_rate,
            "qc_ok_eog": bool(np.isfinite(blink_rate))
        })
    
    df = pd.DataFrame(records)
    logger.info(f"Computed {len(df)} EOG epochs")
    return df


