"""OcularState v0.1 — EOG feature extraction.

Blink detection, saccade proxy, eye stability, and per-epoch QC.

Feature families
----------------
Blink features
    eog_blink_rate (blinks/s), eog_blink_amplitude (mean peak amplitude),
    eog_blink_duration_mean (ms), eog_blink_count

Saccade proxy (from HEOG channels)
    eog_heog_saccade_rate (saccades/s), eog_heog_saccade_amplitude (mean)

Eye stability
    eog_eye_stability_index    (1 = perfectly still)

Signal quality
    eog_artifact_fraction, qc_ok_eog

Note
----
Saccade detection is a proxy based on rapid HEOG deflections.  For precise
saccade kinematics a dedicated eye-tracker would be required.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
from scipy import signal

from . import epoch_selection
from .temporal_support import apply_to_finite_segments, finite_slices
from ..signal_support_provenance import build_signal_support_provenance, finish_operation, start_operation

logger = logging.getLogger(__name__)


def _resolve_epoch_params(
    config: Mapping[str, Any], dataset_id: Optional[str]
) -> tuple[float, float]:
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
        logger.exception("EOG stage-stratified selection failed; continuing with full epochs")
        return None


def _resolve_eog_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    features_cfg = config.get("features", {}) if isinstance(config, Mapping) else {}
    eog_cfg = features_cfg.get("eog", {}) if isinstance(features_cfg, Mapping) else {}
    out: Dict[str, Any] = dict(eog_cfg) if isinstance(eog_cfg, Mapping) else {}
    # Blink detection thresholds
    out.setdefault("filter_low_hz", 0.5)
    out.setdefault("filter_high_hz", 10.0)
    out.setdefault("filter_order", 3)
    out.setdefault("std_mult", 2.0)
    out.setdefault("mad_mult", 4.0)
    out.setdefault("prominence_mult", 1.0)
    out.setdefault("refractory_s", 0.25)
    # Blink duration window (samples either side of peak)
    out.setdefault("blink_halfwidth_s", 0.15)
    # Saccade detection (HEOG)
    out.setdefault("saccade_filter_low_hz", 1.0)
    out.setdefault("saccade_filter_high_hz", 30.0)
    out.setdefault("saccade_std_mult", 2.5)
    out.setdefault("saccade_refractory_s", 0.15)
    # Artifact saturation threshold (fraction of max amplitude)
    out.setdefault("artifact_sat_frac", 0.95)
    return out


def _bandpass(
    data: np.ndarray,
    sfreq: float,
    lo: float,
    hi: float,
    order: int = 3,
) -> np.ndarray:
    filtered, _, _ = _bandpass_with_info(data, sfreq, lo, hi, order)
    return filtered


def _bandpass_with_info(
    data: np.ndarray,
    sfreq: float,
    lo: float,
    hi: float,
    order: int = 3,
) -> tuple[np.ndarray, list[slice], list[slice]]:
    nyq = sfreq * 0.5
    hi_safe = min(hi, nyq * 0.99)
    lo_safe = max(0.01, lo)

    def _demean(segment: np.ndarray) -> np.ndarray:
        return segment - np.median(segment)

    def _filter(segment: np.ndarray) -> np.ndarray:
        if hi_safe <= lo_safe:
            return _demean(segment)
        b, a = signal.butter(order, [lo_safe / nyq, hi_safe / nyq], btype="bandpass")
        return signal.filtfilt(b, a, segment)

    return apply_to_finite_segments(data, _filter, fallback=_demean)


def _detect_blinks(
    veog_1d: np.ndarray,
    sfreq: float,
    *,
    filter_low_hz: float,
    filter_high_hz: float,
    filter_order: int,
    std_mult: float,
    mad_mult: float,
    prominence_mult: float,
    refractory_s: float,
    filtered_data: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (peak_indices, prominences) on the full recording."""
    filtered = (
        np.asarray(filtered_data, dtype=float)
        if filtered_data is not None
        else _bandpass(veog_1d, sfreq, filter_low_hz, filter_high_hz, filter_order)
    )
    min_dist = max(1, int(round(refractory_s * sfreq)))
    peak_parts: list[np.ndarray] = []
    prominence_parts: list[np.ndarray] = []
    for run in finite_slices(filtered):
        segment = filtered[run]
        centered = segment - np.median(segment)
        abs_sig = np.abs(centered)
        mad = float(np.median(np.abs(centered))) + 1e-8
        std = float(np.std(segment))
        thr = max(std_mult * std, mad_mult * mad)
        prominence = max(1e-8, prominence_mult * mad)
        peaks, props = signal.find_peaks(
            abs_sig, height=thr, distance=min_dist, prominence=prominence
        )
        peak_parts.append(peaks.astype(int) + int(run.start or 0))
        prominence_parts.append(np.asarray(props.get("prominences", np.zeros(len(peaks))), dtype=float))
    if not peak_parts:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)
    return np.concatenate(peak_parts), np.concatenate(prominence_parts)


def _detect_saccades(
    heog_1d: np.ndarray,
    sfreq: float,
    *,
    filter_low_hz: float,
    filter_high_hz: float,
    std_mult: float,
    refractory_s: float,
    filtered_data: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rapid-deflection proxy detector for HEOG saccades.

    Works on *velocity* (first derivative) of the HEOG signal.
    Returns (peak_indices, amplitudes_abs).
    """
    filtered = (
        np.asarray(filtered_data, dtype=float)
        if filtered_data is not None
        else _bandpass(heog_1d, sfreq, filter_low_hz, filter_high_hz)
    )
    min_dist = max(1, int(round(refractory_s * sfreq)))
    peak_parts: list[np.ndarray] = []
    amplitude_parts: list[np.ndarray] = []
    for run in finite_slices(filtered):
        segment = filtered[run]
        velocity = np.diff(segment, prepend=segment[0]) * sfreq
        abs_vel = np.abs(velocity)
        thr = float(std_mult * np.std(abs_vel))
        peaks, _ = signal.find_peaks(abs_vel, height=thr, distance=min_dist)
        peak_parts.append(peaks.astype(int) + int(run.start or 0))
        amplitude_parts.append(abs_vel[peaks])
    if not peak_parts:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)
    return np.concatenate(peak_parts), np.concatenate(amplitude_parts)


def _blink_duration_ms(
    signal_arr: np.ndarray,
    peak_idx: int,
    sfreq: float,
    halfwidth_s: float,
    *,
    segment_start: int = 0,
    segment_stop: Optional[int] = None,
) -> float:
    """Estimate blink duration as the half-amplitude width around a peak."""
    hw = max(1, int(halfwidth_s * sfreq))
    stop = len(signal_arr) if segment_stop is None else min(len(signal_arr), int(segment_stop))
    lo = max(int(segment_start), peak_idx - hw)
    hi = min(stop - 1, peak_idx + hw)
    window = signal_arr[lo : hi + 1]
    if len(window) == 0:
        return float("nan")
    peak_val = float(signal_arr[peak_idx])
    half_val = peak_val * 0.5
    crossing = np.where(window < half_val)[0]
    if len(crossing) == 0:
        return float(2 * hw / sfreq * 1000)
    duration_samples = float(len(window) - len(crossing))
    return float(duration_samples / sfreq * 1000)


def compute_eog_features(
    signals: Mapping[str, Any], config: Mapping[str, Any]
) -> pd.DataFrame:
    """Compute per-epoch OcularState v0.1 features.

    Parameters
    ----------
    signals:
        Preprocessed signals dict.
    config:
        Pipeline configuration dict.

    Returns
    -------
    pd.DataFrame
        Per-epoch ocular features.  Empty DataFrame when no ``eog`` channel
        is available.
    """
    if "eog" not in signals.get("signals", {}):
        return pd.DataFrame()

    eog_arr = np.asarray(signals["signals"]["eog"], dtype=float)
    if eog_arr.ndim == 1:
        eog_arr = eog_arr[None, :]
    if eog_arr.ndim != 2 or eog_arr.shape[1] <= 0:
        return pd.DataFrame()

    sfreq = float(signals.get("sfreq", 250))
    dataset_id = signals.get("dataset_id")
    raw_file_path = signals.get("file_path")

    length_s, step_s = _resolve_epoch_params(config, dataset_id)
    ecfg = _resolve_eog_config(config)

    epoch_length_samples = int(length_s * sfreq)
    epoch_step_samples = int(step_s * sfreq)
    n_samples = eog_arr.shape[1]

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

    # -------------------------------------------------------------------
    # Channel role classification
    # For ds003969: EXG1/EXG2 = HEOG, EXG3/EXG4 = VEOG.
    # Heuristic: if >1 channel available, use the first as VEOG and others
    # for HEOG (saccade proxy); VEOG used for blink detection.
    # -------------------------------------------------------------------
    n_chans = eog_arr.shape[0]
    if not np.isfinite(eog_arr).any():
        return pd.DataFrame()
    veog_ch = eog_arr[0]
    heog_ch = eog_arr[1] if n_chans > 1 else None

    inherited = signals.get("meta", {}).get("signal_support_provenance") if isinstance(signals.get("meta", {}), Mapping) else None
    provenance = deepcopy(inherited) if isinstance(inherited, Mapping) else build_signal_support_provenance(
        Path(str(raw_file_path)) if raw_file_path else Path("<unknown-eog-source>"),
        input_sfreq=sfreq,
        target_sfreq=sfreq,
        source_datatype="eog",
    )

    def _filter_channel(
        channel: np.ndarray,
        *,
        operation_name: str,
        low_hz: float,
        high_hz: float,
        order: int,
    ) -> np.ndarray:
        operation = start_operation(
            provenance,
            operation_name,
            input_sfreq=sfreq,
            parameters={"low_hz": low_hz, "high_hz": high_hz, "order": order},
            actual_kwargs={
                "method": "scipy.signal.filtfilt",
                "finite_segment_policy": "filter_each_contiguous_finite_run_independently",
            },
        )
        filtered, runs, fallback_runs = _bandpass_with_info(
            channel, sfreq, low_hz, high_hz, order
        )
        operation["actual_kwargs"]["finite_segment_count"] = len(runs)
        operation["actual_kwargs"]["fallback_segment_count"] = len(fallback_runs)
        finish_operation(
            operation,
            status="fallback" if fallback_runs else "applied",
            output_sfreq=sfreq,
            fallback="demeaned_finite_segments" if fallback_runs else None,
        )
        return filtered

    # -------------------------------------------------------------------
    # Run-level blink detection on VEOG
    # -------------------------------------------------------------------
    veog_filt = _filter_channel(
        veog_ch,
        operation_name="eog_veog_bandpass_filtfilt",
        low_hz=float(ecfg["filter_low_hz"]),
        high_hz=float(ecfg["filter_high_hz"]),
        order=int(ecfg["filter_order"]),
    )
    blink_peaks, blink_prominences = _detect_blinks(
        veog_ch,
        sfreq,
        filter_low_hz=float(ecfg["filter_low_hz"]),
        filter_high_hz=float(ecfg["filter_high_hz"]),
        filter_order=int(ecfg["filter_order"]),
        std_mult=float(ecfg["std_mult"]),
        mad_mult=float(ecfg["mad_mult"]),
        prominence_mult=float(ecfg["prominence_mult"]),
        refractory_s=float(ecfg["refractory_s"]),
        filtered_data=veog_filt,
    )

    # Pre-compute blink amplitudes from filtered VEOG for amplitude stats
    veog_abs = np.full(veog_filt.shape, np.nan, dtype=float)
    for run in finite_slices(veog_filt):
        segment = veog_filt[run]
        veog_abs[run] = np.abs(segment - np.median(segment))

    # Blink durations (run-level, indexed for fast per-epoch lookup)
    halfwidth_s = float(ecfg["blink_halfwidth_s"])

    # -------------------------------------------------------------------
    # Run-level saccade detection on HEOG
    # -------------------------------------------------------------------
    sacc_peaks: np.ndarray = np.array([], dtype=int)
    sacc_amplitudes: np.ndarray = np.array([], dtype=float)
    if heog_ch is not None:
        heog_filt = _filter_channel(
            heog_ch,
            operation_name="eog_heog_bandpass_filtfilt",
            low_hz=float(ecfg["saccade_filter_low_hz"]),
            high_hz=float(ecfg["saccade_filter_high_hz"]),
            order=int(ecfg["filter_order"]),
        )
        sacc_peaks, sacc_amplitudes = _detect_saccades(
            heog_ch,
            sfreq,
            filter_low_hz=float(ecfg["saccade_filter_low_hz"]),
            filter_high_hz=float(ecfg["saccade_filter_high_hz"]),
            std_mult=float(ecfg["saccade_std_mult"]),
            refractory_s=float(ecfg["saccade_refractory_s"]),
            filtered_data=heog_filt,
        )

    # Artifact saturation detection (run-level, per channel).
    # Operate on the DC-removed signal so that Biosemi/BDF DC offsets do not
    # dominate max_abs and cause every sample to be flagged.
    saturation_mask = np.zeros(eog_arr.shape, dtype=bool)
    if np.isfinite(eog_arr).all():
        centered_full = eog_arr - np.median(eog_arr, axis=1, keepdims=True)
        max_abs = float(np.max(np.abs(centered_full))) + 1e-8
        sat_threshold = float(ecfg["artifact_sat_frac"]) * max_abs
        saturation_mask = np.abs(centered_full) > sat_threshold
    else:
        for channel_idx in range(n_chans):
            for run in finite_slices(eog_arr[channel_idx]):
                segment = eog_arr[channel_idx, run]
                centered_segment = segment - np.median(segment)
                segment_max = float(np.max(np.abs(centered_segment))) + 1e-8
                threshold = float(ecfg["artifact_sat_frac"]) * segment_max
                saturation_mask[channel_idx, run] = np.abs(centered_segment) > threshold

    records: List[Dict[str, Any]] = []

    for epoch_idx in range(n_epochs):
        if chosen_epochs is not None and epoch_idx not in chosen_epochs:
            continue

        start_idx = epoch_idx * epoch_step_samples
        end_idx = start_idx + epoch_length_samples
        if end_idx > n_samples:
            break
        if not np.isfinite(eog_arr[:, start_idx:end_idx]).all():
            continue

        # --- blinks in this epoch ---
        lo = int(np.searchsorted(blink_peaks, start_idx, side="left"))
        hi = int(np.searchsorted(blink_peaks, end_idx, side="left"))
        epoch_blink_peaks = blink_peaks[lo:hi]
        n_blinks = max(0, hi - lo)
        blink_rate = float(n_blinks / max(length_s, 1e-6))

        blink_amp: float = float("nan")
        blink_dur: float = float("nan")
        if n_blinks > 0:
            blink_amp = float(np.mean(veog_abs[epoch_blink_peaks]))
            durations = []
            veog_runs = finite_slices(veog_abs)
            for peak in epoch_blink_peaks:
                run = next(
                    (item for item in veog_runs if int(item.start or 0) <= int(peak) < int(item.stop)),
                    None,
                )
                if run is None:
                    continue
                durations.append(
                    _blink_duration_ms(
                        veog_abs,
                        int(peak),
                        sfreq,
                        halfwidth_s,
                        segment_start=int(run.start or 0),
                        segment_stop=int(run.stop),
                    )
                )
            finite_durs = [d for d in durations if np.isfinite(d)]
            blink_dur = float(np.mean(finite_durs)) if finite_durs else float("nan")

        # --- saccades in this epoch (HEOG proxy) ---
        slo = int(np.searchsorted(sacc_peaks, start_idx, side="left"))
        shi = int(np.searchsorted(sacc_peaks, end_idx, side="left"))
        n_saccades = max(0, shi - slo)
        saccade_rate = float(n_saccades / max(length_s, 1e-6))
        saccade_amp: float = float("nan")
        if n_saccades > 0 and len(sacc_amplitudes) > slo:
            saccade_amp = float(np.mean(sacc_amplitudes[slo:shi]))

        # --- artifact fraction ---
        artifact_frac = float(np.mean(np.any(saturation_mask[:, start_idx:end_idx], axis=0)))

        # --- eye stability index ---
        # High blink + saccade rate → lower stability.
        # Signed (NOT clipped to zero) so the full distribution has non-zero
        # MAD, making robust-z normalization stable.  Epochs with extreme
        # eye movement receive negative values; quiet epochs receive positive.
        # Reference: ~0.5 blinks/s and ~1 saccade/s = typical awake resting.
        normalized_movement = (blink_rate / 0.5 + saccade_rate / 1.0) / 2.0
        stability_index = float(1.0 - normalized_movement)

        qc_ok = bool(np.isfinite(blink_rate) and artifact_frac < 0.5)

        records.append(
            {
                "epoch_id": epoch_idx,
                "t_start": start_idx / sfreq,
                "t_end": end_idx / sfreq,
                "eog_blink_rate": blink_rate,
                "eog_blink_count": int(n_blinks),
                "eog_blink_amplitude": blink_amp,
                "eog_blink_duration_mean": blink_dur,
                "eog_heog_saccade_rate": saccade_rate,
                "eog_heog_saccade_amplitude": saccade_amp,
                "eog_eye_stability_index": stability_index,
                "eog_artifact_fraction": artifact_frac,
                "qc_ok_eog": qc_ok,
            }
        )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df.attrs["signal_support_provenance"] = provenance
    logger.info(
        "EOG v0.1: computed %d epochs; qc_ok fraction=%.2f",
        len(df),
        float(df["qc_ok_eog"].mean()) if "qc_ok_eog" in df.columns else 0.0,
    )
    return df
