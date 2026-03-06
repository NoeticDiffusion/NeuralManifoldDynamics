"""
fmri.py
fMRI feature extraction with continuous preprocessing before epoching.

Ingest contract:
- continuous time-domain transforms are applied at session level first
- a single epoch loop defines temporal resolution
- local per-window metrics are then computed from sliced continuous outputs
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from . import fmri_continuous
from . import fmri_epoch_metrics

logger = logging.getLogger(__name__)


def compute_fmri_features(signals: Mapping[str, Any], config: Mapping[str, Any]) -> pd.DataFrame:
    """Compute per-window fMRI metrics from regional BOLD time series.

    Parameters
    ----------
    signals
        Mapping with keys:
        - "signals": dict containing "fmri": array [n_regions, n_times].
        - "sfreq": sampling frequency in Hz (1 / TR).
        - "channels": dict containing "fmri": list of region names (optional).
        - "dataset_id": optional dataset identifier.
    config
        Configuration dict. Window settings are read from ``features.fmri`` /
        ``epoching``. Continuous bandpass/phase settings are read from either
        ``preprocessing`` or ``preprocess.fmri``.

    Returns
    -------
    DataFrame
        One row per time window, with columns:
        - epoch_id, t_start, t_end
        - fmri_variance_global, fmri_FC_mean, fmri_kuramoto_global
        - fmri_entropy_global (legacy alias for fmri_region_var_mean)
        - fmri_lf_power (legacy alias for fmri_signal_power)
        - fmri_region_var_mean
        - fmri_signal_power
        - fmri_lf_power_delta, fmri_lf_power_delta_valid
        - fmri_window_sec, fmri_step_sec, fmri_window_samples, fmri_step_samples, fmri_sfreq
    """
    fmri_data = _validate_and_extract_data(signals)
    n_regions, n_times = fmri_data.shape
    if n_regions == 0 or n_times == 0:
        return pd.DataFrame()

    sfreq = float(signals.get("sfreq", 1.0) or 1.0)
    dataset_id = signals.get("dataset_id")

    window_sec, step_sec, window_samples, step_samples = _compute_window_params(
        config=config,
        sfreq=sfreq,
        dataset_id=dataset_id,
    )
    if window_samples <= 1 or step_samples <= 0:
        logger.warning("Invalid fMRI windowing parameters; returning empty table")
        return pd.DataFrame()
    if n_times < window_samples:
        logger.warning(
            "fMRI run shorter than one window; returning empty feature table (n_times=%d, window_samples=%d)",
            n_times,
            window_samples,
        )
        return pd.DataFrame()

    # 1) Session-level continuous processing before any epoch slices.
    continuous_cfg = _resolve_continuous_cfg(config)
    continuous_out = fmri_continuous.process_session_signals(
        roi_ts=fmri_data,
        sfreq=sfreq,
        config=continuous_cfg,
    )
    filtered_ts = np.asarray(continuous_out.get("filtered_ts", fmri_data), dtype=float)
    phase_ts = continuous_out.get("phase_ts")
    phase_arr = np.asarray(phase_ts, dtype=float) if phase_ts is not None else None

    roi_names = None
    channels_map = signals.get("channels") if isinstance(signals, Mapping) else None
    if isinstance(channels_map, Mapping):
        roi_names = channels_map.get("fmri")

    metrics_cfg = {}
    if isinstance(config, Mapping):
        metrics_cfg = config.get("metrics", {}) or {}
        if not metrics_cfg:
            features_cfg = config.get("features", {}) or {}
            # Preferred v2 path: features.metrics
            if isinstance(features_cfg, Mapping):
                metrics_cfg = features_cfg.get("metrics", {}) or {}
            # Backward-compatible fallback for existing configs.
            if not metrics_cfg:
                metrics_cfg = ((features_cfg.get("fmri", {}) if isinstance(features_cfg, Mapping) else {}) or {}).get(
                    "metrics", {}
                ) or {}

    records: list[Dict[str, Any]] = []
    prev_power_global: Optional[float] = None

    # 2) Single epoch loop (no nested windows).
    for epoch_id, start_idx in enumerate(range(0, n_times - window_samples + 1, step_samples)):
        end_idx = start_idx + window_samples
        epoch_raw = fmri_data[:, start_idx:end_idx]
        epoch_filtered = filtered_ts[:, start_idx:end_idx]
        epoch_phase = phase_arr[:, start_idx:end_idx] if phase_arr is not None else None

        # 3) Local per-window metrics from already processed continuous signals.
        epoch_metrics = fmri_epoch_metrics.compute_local_metrics(
            epoch_raw=epoch_raw,
            epoch_filtered=epoch_filtered,
            epoch_phase=epoch_phase,
            sfreq=float(sfreq),
            config=metrics_cfg,
            roi_names=roi_names if isinstance(roi_names, (list, tuple)) else None,
        )

        # Hard scrub epochs with excessive motion proxy (DVARS) on raw BOLD.
        dvars_val = float(epoch_metrics.get("fmri_dvars", float("nan")))
        dvars_threshold = float(metrics_cfg.get("dvars_threshold", 5.0))
        if np.isfinite(dvars_val) and dvars_val > dvars_threshold:
            for key in list(epoch_metrics.keys()):
                if key != "fmri_dvars":
                    epoch_metrics[key] = float("nan")
            logger.debug(
                "Epoch %d scrubbed due to high DVARS (%.2f > %.2f)",
                int(epoch_id),
                dvars_val,
                dvars_threshold,
            )

        fmri_variance_global = float(epoch_metrics.get("fmri_variance_global", float("nan")))
        fmri_signal_power = float(epoch_metrics.get("fmri_signal_power", float("nan")))
        if prev_power_global is None or not np.isfinite(prev_power_global) or not np.isfinite(fmri_signal_power):
            lf_power_delta = float("nan")
            lf_power_delta_valid = 0
        else:
            lf_power_delta = float(abs(fmri_signal_power - prev_power_global))
            lf_power_delta_valid = 1

        records.append(
            {
                "epoch_id": int(epoch_id),
                "t_start": start_idx / sfreq,
                "t_end": end_idx / sfreq,
                "fmri_variance_global": fmri_variance_global,
                "fmri_region_var_mean": fmri_variance_global,
                "fmri_entropy_global": fmri_variance_global,
                "fmri_signal_power": fmri_signal_power,
                "fmri_lf_power": fmri_signal_power,
                "fmri_lf_power_delta": lf_power_delta,
                "fmri_lf_power_delta_valid": int(lf_power_delta_valid),
                "fmri_window_sec": window_sec,
                "fmri_step_sec": step_sec,
                "fmri_window_samples": int(window_samples),
                "fmri_step_samples": int(step_samples),
                "fmri_sfreq": float(sfreq),
                "dataset_id": dataset_id,
                **epoch_metrics,
            }
        )

        if np.isfinite(fmri_signal_power):
            prev_power_global = fmri_signal_power

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        logger.info("Computed %d fMRI epochs", len(df))
    return df


def _validate_and_extract_data(signals: Mapping[str, Any]) -> np.ndarray:
    sig_dict = signals.get("signals", {})
    if "fmri" not in sig_dict:
        return np.zeros((0, 0), dtype=float)
    fmri_data = np.asarray(sig_dict["fmri"], dtype=float)
    if fmri_data.ndim != 2:
        raise ValueError("signals['fmri'] must have shape (n_regions, n_times)")
    return fmri_data


def _compute_window_params(config: Mapping[str, Any], sfreq: float, dataset_id: Any) -> Tuple[float, float, int, int]:
    features_cfg = config.get("features", {}) if isinstance(config, Mapping) else {}
    fmri_cfg = features_cfg.get("fmri", {}) if isinstance(features_cfg, Mapping) else {}
    epoching_cfg = config.get("epoching", {}) if isinstance(config, Mapping) else {}

    ds_overrides: Dict[str, Any] = {}
    if isinstance(fmri_cfg, Mapping):
        ds_map = fmri_cfg.get("datasets", {})
        if isinstance(ds_map, Mapping) and dataset_id in ds_map and isinstance(ds_map[dataset_id], Mapping):
            ds_overrides = dict(ds_map[dataset_id])  # type: ignore[arg-type]

    window_sec = float(
        ds_overrides.get("window_sec", fmri_cfg.get("window_sec", epoching_cfg.get("length_s", 30.0))) or 30.0
    )
    step_sec = float(ds_overrides.get("step_sec", fmri_cfg.get("step_sec", epoching_cfg.get("step_s", 15.0))) or 15.0)
    if not np.isfinite(window_sec) or not np.isfinite(step_sec) or not np.isfinite(sfreq) or sfreq <= 0:
        return window_sec, step_sec, 0, 0

    window_samples = max(2, int(np.floor(window_sec * sfreq)))
    step_samples = max(1, int(np.floor(step_sec * sfreq)))
    return window_sec, step_sec, window_samples, step_samples


def _resolve_continuous_cfg(config: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if not isinstance(config, Mapping):
        return result

    preprocess_cfg = config.get("preprocess", {})
    if isinstance(preprocess_cfg, Mapping):
        legacy_band = preprocess_cfg.get("fmri_bandpass")
        if isinstance(legacy_band, (list, tuple)) and len(legacy_band) >= 2:
            result["bandpass"] = [float(legacy_band[0]), float(legacy_band[1])]
        preprocess_fmri = preprocess_cfg.get("fmri", {})
        if isinstance(preprocess_fmri, Mapping):
            band = preprocess_fmri.get("bandpass")
            if isinstance(band, (list, tuple)) and len(band) >= 2:
                result["bandpass"] = [float(band[0]), float(band[1])]

    # Optional direct override (preferred when provided).
    user_cfg = config.get("preprocessing")
    if isinstance(user_cfg, Mapping):
        for key in ("f_low", "f_high", "bandpass", "compute_phase"):
            if key in user_cfg:
                result[key] = user_cfg[key]
    return result

