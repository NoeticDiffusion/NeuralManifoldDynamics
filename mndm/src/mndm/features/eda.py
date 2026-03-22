"""Electrodermal activity features (tonic median, phasic energy).

Inputs: signals dict, config dict.
Outputs: DataFrame with per-epoch EDA features.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd
from scipy import signal

logger = logging.getLogger(__name__)


def compute_eda_features(signals: Mapping[str, Any], config: Mapping[str, Any]) -> pd.DataFrame:
    """Compute per-epoch EDA features (tonic median, phasic energy).

    Args:
        signals: Preprocessed signals dict containing ``eda`` under ``signals``.
        config: Configuration with epoching parameters.

    Returns:
        Per-epoch DataFrame with tonic/phasic summary columns.
    """
    if "eda" not in signals.get("signals", {}):
        return pd.DataFrame()
    
    eda_data = signals["signals"].get("eda", None)
    if eda_data is None:
        return pd.DataFrame()
    
    sfreq = signals.get("sfreq", 250)
    
    # Get epoching parameters
    epoching = config.get("epoching", {}) if isinstance(config, dict) else {}
    length_s = epoching.get("length_s", 8.0)
    step_s = epoching.get("step_s", 4.0)
    
    # Use first EDA channel
    eda_channel = eda_data[0] if len(eda_data) > 0 else np.array([])
    
    # Epoch the data
    epoch_length_samples = int(length_s * sfreq)
    epoch_step_samples = int(step_s * sfreq)
    n_epochs = (len(eda_channel) - epoch_length_samples) // epoch_step_samples + 1
    
    records: List[Dict[str, Any]] = []
    
    for epoch_idx in range(n_epochs):
        start_idx = epoch_idx * epoch_step_samples
        end_idx = start_idx + epoch_length_samples
        
        if end_idx > len(eda_channel):
            break
        
        epoch_data = eda_channel[start_idx:end_idx]
        
        # Tonic component: low-pass filtered median
        b_low, a_low = signal.butter(3, 0.5 / (sfreq / 2), 'low')
        tonic = signal.filtfilt(b_low, a_low, epoch_data)
        tonic_med = np.median(tonic)
        
        # Phasic component: bandpass filtered energy
        b_band, a_band = signal.butter(3, [0.05 / (sfreq / 2), 2.0 / (sfreq / 2)], 'band')
        phasic = signal.filtfilt(b_band, a_band, epoch_data)
        phasic_energy = np.mean(phasic ** 2)
        
        records.append({
            "epoch_id": epoch_idx,
            "eda_tonic_med": tonic_med,
            "eda_phasic_energy": phasic_energy
        })
    
    df = pd.DataFrame(records)
    logger.info(f"Computed {len(df)} EDA epochs")
    return df


