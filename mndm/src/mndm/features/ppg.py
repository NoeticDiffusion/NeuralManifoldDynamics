"""PPG feature extraction (rate, amplitude, variability, quality)."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
from scipy import signal

from . import epoch_selection
from .temporal_support import apply_to_finite_segments, finite_slices
from ..signal_support_provenance import build_signal_support_provenance, finish_operation, start_operation

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
        logger.exception("PPG stage-stratified selection failed; continuing with full epochs")
        return None


def compute_ppg_features(signals: Mapping[str, Any], config: Mapping[str, Any]) -> pd.DataFrame:
    """Compute per-epoch PPG features."""
    if "ppg" not in signals.get("signals", {}):
        return pd.DataFrame()

    ppg_arr = np.asarray(signals["signals"]["ppg"], dtype=float)
    if ppg_arr.ndim == 1:
        ppg_arr = ppg_arr[None, :]
    if ppg_arr.ndim != 2 or ppg_arr.shape[1] <= 0:
        return pd.DataFrame()

    sfreq = float(signals.get("sfreq", 250))
    dataset_id = signals.get("dataset_id")
    raw_file_path = signals.get("file_path")
    length_s, step_s = _resolve_epoch_params(config, dataset_id)

    features_cfg = config.get("features", {}) if isinstance(config, Mapping) else {}
    ppg_cfg = features_cfg.get("ppg", {}) if isinstance(features_cfg, Mapping) else {}
    if not isinstance(ppg_cfg, Mapping):
        ppg_cfg = {}
    preprocess_cfg = config.get("preprocess", {}) if isinstance(config, Mapping) else {}
    temporal_support_mode = (
        str(preprocess_cfg.get("temporal_support_mode", "default")).strip().lower()
        if isinstance(preprocess_cfg, Mapping)
        else "default"
    )
    estimator = str(ppg_cfg.get("estimator", "recording_stream_v1")).strip().lower()
    window_local_mode = estimator == "window_local_present_only_v1"
    if temporal_support_mode == "causal_finite" and not window_local_mode:
        raise ValueError(
            "causal_finite preprocessing requires features.ppg.estimator=window_local_present_only_v1"
        )
    if window_local_mode and temporal_support_mode != "causal_finite":
        raise ValueError(
            "window_local_present_only_v1 requires preprocess.temporal_support_mode=causal_finite"
        )
    source_channels = (
        list(signals.get("channels", {}).get("ppg", []))
        if isinstance(signals.get("channels", {}), Mapping)
        else []
    )
    if window_local_mode and (
        len(source_channels) != 1
        or not re.search(r"(?i)(ppg|pleth|pulse|photopleth)", str(source_channels[0]))
    ):
        raise ValueError(
            "window_local_present_only_v1 requires exactly one source channel identified by its pulse/PPG name"
        )
    if window_local_mode and (ppg_arr.ndim != 2 or ppg_arr.shape[0] != len(source_channels) or ppg_arr.shape[0] != 1):
        raise ValueError(
            "window_local_present_only_v1 requires exactly one PPG signal row matching its source-channel metadata"
        )
    bandpass_low_hz = float(ppg_cfg.get("bandpass_low_hz", 0.4) or 0.4)
    bandpass_high_hz = float(ppg_cfg.get("bandpass_high_hz", 8.0) or 8.0)
    bandpass_order = int(ppg_cfg.get("bandpass_order", 3) or 3)
    refractory_s = float(ppg_cfg.get("refractory_s", 0.35) or 0.35)
    prominence_mult = float(ppg_cfg.get("prominence_mult", 0.75) or 0.75)

    ppg_channel = np.asarray(ppg_arr[0], dtype=float)
    if not np.isfinite(ppg_channel).any():
        return pd.DataFrame()
    epoch_length_samples = int(length_s * sfreq)
    epoch_step_samples = int(step_s * sfreq)
    n_samples = int(len(ppg_channel))
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

    nyquist = sfreq * 0.5
    hi = min(bandpass_high_hz, nyquist * 0.99)
    lo = max(0.01, bandpass_low_hz)
    if hi <= lo:
        lo = 0.4
        hi = min(8.0, nyquist * 0.99)
    inherited = signals.get("meta", {}).get("signal_support_provenance") if isinstance(signals.get("meta", {}), Mapping) else None
    provenance = deepcopy(inherited) if isinstance(inherited, Mapping) else build_signal_support_provenance(
        Path(str(signals.get("file_path"))) if signals.get("file_path") else Path("<unknown-ppg-source>"),
        input_sfreq=sfreq,
        target_sfreq=sfreq,
        source_datatype="ppg",
    )
    filter_runs: list[slice] = []
    filter_fallbacks: list[slice] = []

    def _demean_segment(segment: np.ndarray) -> np.ndarray:
        return segment - np.median(segment)

    if window_local_mode:
        ppg_filter = start_operation(
            provenance,
            "ppg_bandpass",
            operation_class="temporal",
            input_sfreq=sfreq,
            parameters={"enabled": False, "reason": "present_only_window_local_estimator"},
            actual_kwargs={"skipped": True, "reason": "raw PPG is analyzed within each current measurement window"},
        )
        finish_operation(ppg_filter, status="skipped", output_sfreq=sfreq)
        filtered_full = ppg_channel.copy()
        filter_runs = finite_slices(filtered_full)
        provenance["ppg_feature_contract"] = {
            "schema": "mndm.ppg_window_feature.v1",
            "status": "computed_unqualified",
            "estimator": "window_local_present_only_v1",
            "source_channels": source_channels,
            "source_channel_types_after_dataset_typing": {
                str(name): str(value)
                for name, value in provenance.get("channel_types_after_dataset_typing", {}).items()
                if name in source_channels
            },
            "source_channel_selection": (
                provenance.get("ppg_source_selection", {}).get("rule") or "not_recorded"
                if isinstance(provenance.get("ppg_source_selection", {}), Mapping)
                else "not_recorded"
            ),
            "feature_window_sec": float(length_s),
            "feature_step_sec": float(step_s),
            "signal_filter": "none",
            "threshold_scope": "each_current_window_median_and_MAD",
            "information_scope": "present_only_within_[t_start,t_end); no full-recording threshold or filter",
            "rate_unit": "beats_per_minute",
            "qc_rule": "finite rate and at least two detected peaks in the current window",
        }
    else:
        ppg_filter = start_operation(
            provenance,
            "ppg_bandpass_filtfilt",
            input_sfreq=sfreq,
            parameters={"low_hz": lo, "high_hz": hi, "order": bandpass_order},
            actual_kwargs={
                "method": "scipy.signal.filtfilt",
                "finite_segment_policy": "filter_each_contiguous_finite_run_independently",
                "b": "recorded_below",
                "a": "recorded_below",
            },
        )
        try:
            b, a = signal.butter(bandpass_order, [lo / nyquist, hi / nyquist], btype="bandpass")
            ppg_filter["actual_kwargs"]["b"] = np.asarray(b).tolist()
            ppg_filter["actual_kwargs"]["a"] = np.asarray(a).tolist()
            filtered_full, filter_runs, filter_fallbacks = apply_to_finite_segments(
                ppg_channel,
                lambda segment: signal.filtfilt(b, a, segment),
                fallback=_demean_segment,
            )
            ppg_filter["actual_kwargs"]["finite_segment_count"] = len(filter_runs)
            ppg_filter["actual_kwargs"]["fallback_segment_count"] = len(filter_fallbacks)
            finish_operation(
                ppg_filter,
                status="fallback" if filter_fallbacks else "applied",
                output_sfreq=sfreq,
                fallback="demeaned_finite_segments" if filter_fallbacks else None,
            )
        except Exception as primary_exc:
            logger.exception("PPG bandpass failed; falling back to demeaned signal for pulse detection")
            filtered_full, filter_runs, _ = apply_to_finite_segments(ppg_channel, _demean_segment)
            ppg_filter["actual_kwargs"]["finite_segment_count"] = len(filter_runs)
            ppg_filter["actual_kwargs"]["fallback_segment_count"] = len(filter_runs)
            finish_operation(
                ppg_filter,
                status="fallback",
                output_sfreq=sfreq,
                fallback="demeaned_finite_segments",
                error=primary_exc,
            )

    min_dist = max(1, int(round(refractory_s * sfreq)))
    segmented_support = not bool(np.isfinite(ppg_channel).all())
    peak_parts: list[np.ndarray] = []
    prominence_parts: list[np.ndarray] = []
    segment_prominences: list[float] = []
    if window_local_mode:
        peak_operation = start_operation(
            provenance,
            "ppg_peak_detection",
            operation_class="feature",
            input_sfreq=sfreq,
            parameters={
                "distance_samples": min_dist,
                "prominence_mult": prominence_mult,
                "threshold_fit_scope": "per_measurement_window_median_and_MAD",
                "feature_window_sec": float(length_s),
                "feature_step_sec": float(step_s),
            },
            actual_kwargs={"signal": "raw_window_centered_by_median", "peak_input": "absolute_centered_signal"},
        )
        peak_operation["information_scope"] = "present_only_within_each_current_feature_window"
        finish_operation(peak_operation, status="applied", output_sfreq=sfreq)
        peak_operation["result"] = {
            "global_threshold": False,
            "threshold_fitted_per_window": True,
            "finite_segment_count": len(filter_runs),
        }
        peaks = np.asarray([], dtype=int)
        prominences = np.asarray([], dtype=float)
    else:
        for run in finite_slices(filtered_full):
            segment = filtered_full[run]
            centered = segment - np.median(segment)
            mad = float(np.median(np.abs(centered))) + 1e-8
            prominence = max(1e-6, prominence_mult * mad)
            segment_prominences.append(float(prominence))
            local_peaks, props = signal.find_peaks(
                np.abs(centered), distance=min_dist, prominence=prominence
            )
            peak_parts.append(local_peaks.astype(int) + int(run.start or 0))
            prominence_parts.append(
                np.asarray(props.get("prominences", np.asarray([], dtype=float)), dtype=float)
            )
        peaks = np.concatenate(peak_parts) if peak_parts else np.asarray([], dtype=int)
        prominences = (
            np.concatenate(prominence_parts) if prominence_parts else np.asarray([], dtype=float)
        )
        representative_prominence = (
            float(np.median(segment_prominences)) if segment_prominences else float("nan")
        )
        peak_operation = start_operation(
            provenance,
            "ppg_peak_detection",
            input_sfreq=sfreq,
            parameters={
                "distance_samples": min_dist,
                "prominence": representative_prominence,
                "prominence_mult": prominence_mult,
                "threshold_fit_scope": (
                    "per_contiguous_finite_segment_median_and_MAD"
                    if segmented_support
                    else "whole_stream_median_and_MAD"
                ),
                "finite_segment_count": len(filter_runs),
                "prominence_by_segment": segment_prominences,
            },
            actual_kwargs={"distance": min_dist, "prominence_by_segment": segment_prominences},
        )
        finish_operation(peak_operation, status="applied", output_sfreq=sfreq)
        peak_operation["result"] = {
            "peaks_count": int(peaks.size),
            "global_threshold": not segmented_support,
            "finite_segment_count": len(filter_runs),
        }

    records: List[Dict[str, Any]] = []
    for epoch_idx in range(n_epochs):
        if chosen_epochs is not None and epoch_idx not in chosen_epochs:
            continue
        start_idx = epoch_idx * epoch_step_samples
        end_idx = start_idx + epoch_length_samples
        if end_idx > n_samples:
            break
        if not np.isfinite(ppg_channel[start_idx:end_idx]).all():
            continue

        if window_local_mode:
            current_window = ppg_channel[start_idx:end_idx]
            centered_window = current_window - np.median(current_window)
            window_mad = float(np.median(np.abs(centered_window))) + 1e-8
            window_prominence = max(1e-6, prominence_mult * window_mad)
            epoch_peaks, props = signal.find_peaks(
                np.abs(centered_window), distance=min_dist, prominence=window_prominence
            )
            epoch_prom = np.asarray(props.get("prominences", np.asarray([], dtype=float)), dtype=float)
        else:
            left = int(np.searchsorted(peaks, start_idx, side="left"))
            right = int(np.searchsorted(peaks, end_idx, side="left"))
            epoch_peaks = peaks[left:right]
            epoch_prom = prominences[left:right] if prominences.size == peaks.size else np.asarray([], dtype=float)
        pulse_intervals = np.diff(epoch_peaks.astype(float)) / sfreq if epoch_peaks.size >= 2 else np.asarray([], dtype=float)
        rate_bpm = float(60.0 / np.mean(pulse_intervals)) if pulse_intervals.size > 0 and np.mean(pulse_intervals) > 0 else np.nan
        amp_mean = float(np.mean(epoch_prom)) if epoch_prom.size > 0 else np.nan
        amp_std = float(np.std(epoch_prom, ddof=1)) if epoch_prom.size >= 2 else np.nan
        amp_cv = float(amp_std / amp_mean) if np.isfinite(amp_std) and np.isfinite(amp_mean) and amp_mean != 0 else np.nan
        quality_score = float(min(1.0, epoch_peaks.size / max(length_s * 1.5, 1.0)))

        records.append(
            {
                "epoch_id": epoch_idx,
                "t_start": start_idx / sfreq,
                "t_end": end_idx / sfreq,
                "ppg_rate_bpm": rate_bpm,
                "ppg_pulse_count": int(epoch_peaks.size),
                "ppg_amplitude_mean": amp_mean,
                "ppg_amplitude_std": amp_std,
                "ppg_amplitude_cv": amp_cv,
                "ppg_quality_score": quality_score,
                "qc_ok_ppg": bool(np.isfinite(rate_bpm) and epoch_peaks.size >= 2),
            }
        )

    df = pd.DataFrame(records)
    df.attrs["signal_support_provenance"] = provenance
    logger.info("Computed %d PPG epochs", len(df))
    return df
