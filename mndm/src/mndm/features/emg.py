"""EMG feature extraction (RMS, activation proxies).

Inputs: signals dict, config dict.
Outputs: DataFrame with per-epoch RMS values.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from . import epoch_selection

logger = logging.getLogger(__name__)


def _resolve_epoch_params(config: Mapping[str, Any], dataset_id: Optional[str]) -> tuple[float, float]:
    """Internal helper: resolve epoch params."""
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
    """Internal helper: resolve chosen epochs."""
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
        logger.exception("EMG stage-stratified selection failed; continuing with full epochs")
        return None


def compute_emg_features(signals: Mapping[str, Any], config: Mapping[str, Any]) -> pd.DataFrame:
    """Compute per-epoch EMG features (RMS).

    Args:
        signals: Preprocessed signals dict with ``signals`` and ``sfreq``.
        config: Configuration with epoching parameters.

    Returns:
        DataFrame with ``epoch_id``, ``emg_rms``, etc.
    """
    if "emg" not in signals.get("signals", {}):
        return pd.DataFrame()
    
    emg_arr = np.asarray(signals["signals"]["emg"], dtype=float)
    if emg_arr.ndim == 1:
        emg_arr = emg_arr[None, :]
    if emg_arr.ndim != 2 or emg_arr.shape[1] <= 0:
        return pd.DataFrame()
    sfreq = signals.get("sfreq", 250)
    dataset_id = signals.get("dataset_id")
    raw_file_path = signals.get("file_path")
    
    # Get epoching parameters with dataset overrides
    length_s, step_s = _resolve_epoch_params(config, dataset_id)
    
    # Epoch the data
    epoch_length_samples = int(length_s * sfreq)
    epoch_step_samples = int(step_s * sfreq)
    n_samples = int(emg_arr.shape[1])
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
    
    records: List[Dict[str, Any]] = []
    
    for epoch_idx in range(n_epochs):
        if chosen_epochs is not None and epoch_idx not in chosen_epochs:
            continue
        start_idx = epoch_idx * epoch_step_samples
        end_idx = start_idx + epoch_length_samples
        
        if end_idx > n_samples:
            break
        
        epoch_data = emg_arr[:, start_idx:end_idx]
        # Remove per-channel baseline offset before RMS aggregation.
        epoch_data = epoch_data - np.median(epoch_data, axis=1, keepdims=True)
        
        # RMS across channels and time
        rms = float(np.sqrt(np.mean(epoch_data ** 2)))
        
        records.append({
            "epoch_id": epoch_idx,
            "t_start": start_idx / float(sfreq),
            "t_end": end_idx / float(sfreq),
            "emg_rms": rms,
            "qc_ok_emg": bool(np.isfinite(rms)),
        })
    
    df = pd.DataFrame(records)
    logger.info(f"Computed {len(df)} EMG epochs")
    return df


