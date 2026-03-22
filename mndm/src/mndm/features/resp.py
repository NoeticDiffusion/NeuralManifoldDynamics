"""Respiration feature extraction (rate variance, stability proxies).

Inputs: signals dict, config dict.
Outputs: DataFrame with per-epoch respiration rate variance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd
from scipy import signal

logger = logging.getLogger(__name__)


def compute_resp_features(signals: Mapping[str, Any], config: Mapping[str, Any]) -> pd.DataFrame:
    """Compute per-epoch respiration features (rate variance).

    Args:
        signals: Preprocessed signals dict with ``signals`` and ``sfreq``.
        config: Configuration with epoching parameters.

    Returns:
        DataFrame with ``epoch_id``, ``resp_rate_var``, etc.
    """
    if "resp" not in signals.get("signals", {}):
        return pd.DataFrame()
    
    resp_data = signals["signals"].get("resp", None)
    if resp_data is None:
        return pd.DataFrame()
    
    sfreq = signals.get("sfreq", 250)
    
    # Get epoching parameters
    epoching = config.get("epoching", {}) if isinstance(config, dict) else {}
    length_s = epoching.get("length_s", 8.0)
    step_s = epoching.get("step_s", 4.0)
    
    # Use first RESP channel
    resp_channel = resp_data[0] if len(resp_data) > 0 else np.array([])
    
    # Epoch the data
    epoch_length_samples = int(length_s * sfreq)
    epoch_step_samples = int(step_s * sfreq)
    n_epochs = (len(resp_channel) - epoch_length_samples) // epoch_step_samples + 1
    
    records: List[Dict[str, Any]] = []
    
    for epoch_idx in range(n_epochs):
        start_idx = epoch_idx * epoch_step_samples
        end_idx = start_idx + epoch_length_samples
        
        if end_idx > len(resp_channel):
            break
        
        epoch_data = resp_channel[start_idx:end_idx]
        
        # Detect breaths (peaks in respiration signal)
        peaks, _ = signal.find_peaks(epoch_data, height=np.mean(epoch_data))
        
        if len(peaks) < 2:
            records.append({
                "epoch_id": epoch_idx,
                "resp_rate_var": np.nan
            })
            continue
        
        # Compute respiration rate over time
        breath_intervals = np.diff(peaks) / sfreq
        breath_rates = 60 / breath_intervals  # breaths per minute
        
        # Variance of respiration rate
        rate_var = np.var(breath_rates)
        
        records.append({
            "epoch_id": epoch_idx,
            "resp_rate_var": rate_var
        })
    
    df = pd.DataFrame(records)
    logger.info(f"Computed {len(df)} RESP epochs")
    return df


