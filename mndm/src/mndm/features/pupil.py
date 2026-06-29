"""Pupil feature extraction (diameter, volatility, blink proxies, quality)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

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
        logger.exception("Pupil stage-stratified selection failed; continuing with full epochs")
        return None


def compute_pupil_features(signals: Mapping[str, Any], config: Mapping[str, Any]) -> pd.DataFrame:
    """Compute per-epoch pupil features."""
    if "pupil" not in signals.get("signals", {}):
        return pd.DataFrame()

    pupil_arr = np.asarray(signals["signals"]["pupil"], dtype=float)
    if pupil_arr.ndim == 1:
        pupil_arr = pupil_arr[None, :]
    if pupil_arr.ndim != 2 or pupil_arr.shape[1] <= 0:
        return pd.DataFrame()

    sfreq = float(signals.get("sfreq", 120))
    dataset_id = signals.get("dataset_id")
    raw_file_path = signals.get("file_path")
    length_s, step_s = _resolve_epoch_params(config, dataset_id)

    valid_counts = np.sum(np.isfinite(pupil_arr), axis=0)
    summed = np.nansum(pupil_arr, axis=0)
    pupil_channel = np.divide(
        summed,
        np.maximum(valid_counts, 1),
        out=np.full(pupil_arr.shape[1], np.nan, dtype=float),
        where=valid_counts > 0,
    )
    epoch_length_samples = int(length_s * sfreq)
    epoch_step_samples = int(step_s * sfreq)
    n_samples = int(len(pupil_channel))
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

    records: List[Dict[str, Any]] = []
    for epoch_idx in range(n_epochs):
        if chosen_epochs is not None and epoch_idx not in chosen_epochs:
            continue
        start_idx = epoch_idx * epoch_step_samples
        end_idx = start_idx + epoch_length_samples
        if end_idx > n_samples:
            break

        epoch = pupil_channel[start_idx:end_idx]
        valid = np.isfinite(epoch) & (epoch > 0)
        valid_frac = float(np.mean(valid)) if epoch.size else 0.0
        if np.any(valid):
            clean = epoch[valid]
            diffs = np.diff(clean) * sfreq if clean.size >= 2 else np.asarray([], dtype=float)
            diameter_mean = float(np.mean(clean))
            diameter_std = float(np.std(clean, ddof=1)) if clean.size >= 2 else np.nan
            dilation_velocity = float(np.mean(np.abs(diffs))) if diffs.size else np.nan
        else:
            diameter_mean = np.nan
            diameter_std = np.nan
            dilation_velocity = np.nan

        invalid_mask = ~valid
        blink_starts = np.diff(np.concatenate(([0], invalid_mask.astype(np.int8), [0])))
        blink_count = int(np.sum(blink_starts == 1))
        blink_rate = float(blink_count / max(length_s, 1e-6))

        records.append(
            {
                "epoch_id": epoch_idx,
                "t_start": start_idx / sfreq,
                "t_end": end_idx / sfreq,
                "pupil_diameter_mean": diameter_mean,
                "pupil_diameter_std": diameter_std,
                "pupil_dilation_velocity": dilation_velocity,
                "pupil_blink_fraction": float(1.0 - valid_frac),
                "pupil_blink_rate": blink_rate,
                "pupil_quality_score": valid_frac,
                "qc_ok_pupil": bool(valid_frac >= 0.5 and np.isfinite(diameter_mean)),
            }
        )

    df = pd.DataFrame(records)
    logger.info("Computed %d pupil epochs", len(df))
    return df
