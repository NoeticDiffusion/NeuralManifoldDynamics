"""Electrodermal activity (EDA / GSR) feature extraction.

Extracts a tonic/phasic decomposition and skin-conductance-response (SCR)
events from an ``eda``-typed channel, then summarises them per MNPS epoch
window.

Decomposition
-------------
The full session is decomposed once (session-level tonic/phasic separation),
then per-epoch statistics are read from the continuous arrays.  This avoids
the edge-filtering artifacts that per-epoch decomposition would introduce,
and matches the design already used by ``resp.py`` / ``ecg.py`` (run-level
detection, epoch-level summarisation).

Two decomposition backends are supported:

``neurokit2`` (preferred, used when installed)
    ``neurokit2.eda_process()`` with the default ``"neurokit"`` method
    (high-pass-style tonic/phasic separation) plus its built-in SCR detector.

``scipy`` (fallback, always available)
    A simple low-pass tonic estimate (``phasic = raw - tonic``) and a
    prominence-based peak detector on the phasic residual as an SCR proxy.
    Less accurate than neurokit2 but requires no extra dependency.

Feature families
----------------
Tonic (slow, baseline arousal)
    ``eda_tonic_scl``, ``eda_tonic_slope``
Phasic (fast, event-related SCRs)
    ``eda_phasic_scr_rate``, ``eda_phasic_scr_amp``, ``eda_phasic_scr_count``,
    ``eda_phasic_auc``
Derived
    ``eda_arousal_index`` (``scr_rate + |tonic_slope|``, unnormalised)
Signal quality / provenance
    ``eda_source_channel``, ``qc_ok_eda``

Note
----
This module is dataset-agnostic: it consumes whatever channel the upstream
pipeline exposes under ``signals["signals"]["eda"]`` (see
``preprocess.py``'s modality-collection step, which matches channels by name
since EDA has no native MNE channel type). It does not know or care whether
that channel originated from a native EDF/BDF channel or was injected from a
companion physio file (e.g. via ``preprocess.datasets.<id>.physio_tsv_inject``)
— any dataset-specific parsing (column layout, sampling rate, compression)
must happen upstream of this extractor.
"""

from __future__ import annotations

import logging
from math import gcd
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy import stats as sp_stats

from . import epoch_selection

logger = logging.getLogger(__name__)


def _resolve_epoch_params(config: Mapping[str, Any], dataset_id: Optional[str]) -> tuple[float, float]:
    return epoch_selection.resolve_epoch_params(config, dataset_id)


def _resolve_eda_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve EDA-specific feature config with additive defaults."""
    features_cfg = config.get("features", {}) if isinstance(config, Mapping) else {}
    eda_cfg = features_cfg.get("eda", {}) if isinstance(features_cfg, Mapping) else {}
    out: Dict[str, Any] = dict(eda_cfg) if isinstance(eda_cfg, Mapping) else {}
    out.setdefault("enabled", True)
    out.setdefault("target_sfreq_hz", 50.0)
    out.setdefault("decomposition_method", "neurokit")
    out.setdefault("min_signal_range_uS", 0.01)
    out.setdefault("min_mean_uS", 0.0)
    out.setdefault("max_mean_uS", 100.0)
    out.setdefault("min_epoch_coverage_s", 2.0)
    out.setdefault("scr_min_distance_s", 1.0)
    out.setdefault("scr_prominence_mult", 5.0)
    out.setdefault("tonic_window_s", 10.0)
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
        logger.exception("EDA stage-stratified selection failed; continuing with full epochs")
        return None


def _downsample_1d(x: np.ndarray, sfreq: float, target_sfreq: float) -> tuple[np.ndarray, float]:
    """Downsample a 1-D signal to (approximately) ``target_sfreq`` Hz.

    Uses ``scipy.signal.resample_poly`` with a GCD-reduced rational ratio —
    the same resampling approach already used elsewhere in this codebase
    (see ``preprocess.py``'s early-resample step). Handles arbitrary sfreq /
    target_sfreq ratios without the "factor <= 13" caveat of iterative
    ``scipy.signal.decimate``.

    Returns the (possibly unchanged) array and its actual resulting sfreq.
    """
    if target_sfreq <= 0 or target_sfreq >= sfreq or x.size == 0:
        return x, sfreq
    try:
        up = int(round(target_sfreq * 1000))
        down = int(round(sfreq * 1000))
        g = gcd(up, down) or 1
        up //= g
        down //= g
        ds = sp_signal.resample_poly(x.astype(np.float64), up, down)
        return ds, sfreq * up / down
    except Exception:
        logger.debug("EDA downsample failed; using native sfreq")
        return x, sfreq


def _decompose_session_neurokit2(
    eda_1d: np.ndarray, sfreq: float, method: str
) -> Optional[Dict[str, np.ndarray]]:
    """Session-level tonic/phasic decomposition via NeuroKit2. None on failure."""
    try:
        import warnings

        import neurokit2 as nk  # type: ignore

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nk_signals, info = nk.eda_process(eda_1d, sampling_rate=int(round(sfreq)), method=method)
        return {
            "tonic": np.asarray(nk_signals["EDA_Tonic"].values, dtype=float),
            "phasic": np.asarray(nk_signals["EDA_Phasic"].values, dtype=float),
            "scr_peaks": np.asarray(info.get("SCR_Peaks", []), dtype=int),
            "scr_amplitudes": np.asarray(info.get("SCR_Amplitude", []), dtype=float),
        }
    except ImportError:
        logger.debug("neurokit2 not installed; falling back to scipy EDA decomposition")
        return None
    except Exception as exc:
        logger.warning("neurokit2 EDA decomposition failed: %s — falling back to scipy", exc)
        return None


def _decompose_session_scipy(
    eda_1d: np.ndarray,
    sfreq: float,
    *,
    scr_min_distance_s: float,
    scr_prominence_mult: float,
    tonic_window_s: float = 10.0,
) -> Dict[str, np.ndarray]:
    """Fallback tonic/phasic split + prominence-based SCR proxy (no neurokit2 required).

    Tonic: sliding-median baseline over a ``tonic_window_s``-wide window.
    A median filter is used instead of a low-pass IIR filter: at the very
    low cutoffs needed to isolate tonic drift (~0.05 Hz), a Butterworth
    filter's impulse response can span longer than a typical recording,
    producing ringing across the *entire* signal rather than just at the
    edges. The median filter is unconditionally stable and robust to the
    phasic SCR bumps riding on top of the tonic baseline.
    Phasic: residual (raw - tonic).
    SCR events: positive-going peaks in the phasic residual above a
    MAD-scaled prominence threshold, at least ``scr_min_distance_s`` apart.
    """
    try:
        from scipy.ndimage import median_filter

        win = max(3, int(round(tonic_window_s * sfreq)))
        if win % 2 == 0:
            win += 1
        tonic = median_filter(eda_1d, size=win, mode="nearest")
    except Exception:
        tonic = np.full_like(eda_1d, float(np.nanmean(eda_1d)) if eda_1d.size else np.nan)
    phasic = eda_1d - tonic

    try:
        phasic_mad = float(np.median(np.abs(phasic - np.median(phasic)))) + 1e-9
        min_dist = max(1, int(round(scr_min_distance_s * sfreq)))
        prom = max(1e-6, scr_prominence_mult * 1.4826 * phasic_mad)
        peaks, props = sp_signal.find_peaks(phasic, distance=min_dist, prominence=prom)
        amps = np.asarray(props.get("prominences", np.array([])), dtype=float)
    except Exception:
        peaks = np.array([], dtype=int)
        amps = np.array([], dtype=float)

    return {
        "tonic": tonic,
        "phasic": phasic,
        "scr_peaks": peaks.astype(int),
        "scr_amplitudes": amps,
    }


def _nan_epoch_row(source_channel: str) -> Dict[str, Any]:
    return {
        "eda_tonic_scl": np.nan,
        "eda_tonic_slope": np.nan,
        "eda_phasic_scr_rate": np.nan,
        "eda_phasic_scr_amp": np.nan,
        "eda_phasic_scr_count": np.nan,
        "eda_phasic_auc": np.nan,
        "eda_arousal_index": np.nan,
        "eda_source_channel": source_channel,
        "qc_ok_eda": False,
    }


def _epoch_features(
    session: Mapping[str, np.ndarray],
    i0: int,
    i1: int,
    sfreq: float,
    min_samples: int,
    source_channel: str,
) -> Dict[str, Any]:
    """Extract scalar EDA features for one epoch window on the decomposed session."""
    if i1 - i0 < min_samples:
        return _nan_epoch_row(source_channel)

    tonic_seg = session["tonic"][i0:i1]
    phasic_seg = session["phasic"][i0:i1]
    if tonic_seg.size == 0 or not np.any(np.isfinite(tonic_seg)):
        return _nan_epoch_row(source_channel)

    tonic_scl = float(np.nanmean(tonic_seg))
    t_arr = np.arange(len(tonic_seg)) / sfreq
    valid = np.isfinite(tonic_seg)
    slope = float("nan")
    if valid.sum() >= 4:
        slope, *_ = sp_stats.linregress(t_arr[valid], tonic_seg[valid])
        slope = float(slope)

    phasic_auc = float(np.nanmean(np.abs(phasic_seg))) if phasic_seg.size else float("nan")

    duration_s = (i1 - i0) / sfreq
    scr_peaks = session["scr_peaks"]
    scr_amplitudes = session["scr_amplitudes"]
    mask = (scr_peaks >= i0) & (scr_peaks < i1)
    scr_count = int(np.sum(mask))
    scr_rate = float(scr_count / duration_s * 60.0) if duration_s > 0 else float("nan")

    scr_amp = 0.0
    if scr_count > 0 and scr_amplitudes.size == scr_peaks.size:
        amps = scr_amplitudes[mask]
        amps = amps[np.isfinite(amps) & (amps > 0)]
        if amps.size:
            scr_amp = float(amps.mean())

    arousal_index = (
        float(scr_rate + abs(slope)) if np.isfinite(scr_rate) and np.isfinite(slope) else float("nan")
    )

    return {
        "eda_tonic_scl": tonic_scl,
        "eda_tonic_slope": slope,
        "eda_phasic_scr_rate": scr_rate,
        "eda_phasic_scr_amp": scr_amp,
        "eda_phasic_scr_count": float(scr_count),
        "eda_phasic_auc": phasic_auc,
        "eda_arousal_index": arousal_index,
        "eda_source_channel": source_channel,
        "qc_ok_eda": True,
    }


def compute_eda_features(signals: Mapping[str, Any], config: Mapping[str, Any]) -> pd.DataFrame:
    """Compute per-epoch EDA features (tonic SCL, phasic SCR events).

    Args:
        signals: Preprocessed signals dict with ``signals``, ``sfreq``,
            ``channels``, ``dataset_id``, ``file_path``.
        config: Pipeline configuration (optional ``features.eda`` block).

    Returns:
        Per-epoch DataFrame. Empty when no ``eda`` channel is available or
        ``features.eda.enabled`` is explicitly set to ``false``.
    """
    if "eda" not in signals.get("signals", {}):
        return pd.DataFrame()

    eda_data = signals["signals"].get("eda")
    if eda_data is None:
        return pd.DataFrame()

    eda_arr = np.asarray(eda_data, dtype=float)
    if eda_arr.ndim == 1:
        eda_arr = eda_arr[None, :]
    if eda_arr.ndim != 2 or eda_arr.shape[1] <= 0:
        return pd.DataFrame()

    ecfg = _resolve_eda_config(config)
    if not bool(ecfg.get("enabled", True)):
        return pd.DataFrame()

    sfreq = float(signals.get("sfreq", 250))
    dataset_id = signals.get("dataset_id")
    raw_file_path = signals.get("file_path")

    length_s, step_s = _resolve_epoch_params(config, dataset_id)

    channels = signals.get("channels", {})
    eda_channel_names: List[str] = []
    if isinstance(channels, Mapping):
        eda_channel_names = channels.get("eda", []) or []
    if isinstance(eda_channel_names, str):
        eda_channel_names = [eda_channel_names]
    source_channel = str(eda_channel_names[0]) if eda_channel_names else "eda_0"

    eda_native = eda_arr[0]

    epoch_length_samples = int(length_s * sfreq)
    epoch_step_samples = int(step_s * sfreq)
    n_samples = int(len(eda_native))
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

    # --- Session-level decomposition (once per file, not per epoch) ---
    target_sfreq = float(ecfg.get("target_sfreq_hz", 50.0) or 50.0)
    eda_ds, eda_sfreq = _downsample_1d(eda_native, sfreq, target_sfreq)

    # Flatness/saturation QC runs on the *native* signal, not the downsampled
    # one: resample_poly's edge padding can introduce Gibbs-like ringing at
    # the start/end of a genuinely flat channel, which would otherwise mask
    # a truly dead sensor.
    eda_range = float(eda_native.max() - eda_native.min()) if eda_native.size else 0.0
    eda_mean = float(np.nanmean(eda_native)) if eda_native.size else float("nan")
    min_range = float(ecfg.get("min_signal_range_uS", 0.01))
    min_mean = float(ecfg.get("min_mean_uS", 0.0))
    max_mean = float(ecfg.get("max_mean_uS", 100.0))
    session_quality_ok = bool(
        eda_range >= min_range and np.isfinite(eda_mean) and min_mean < eda_mean < max_mean
    )

    session: Optional[Dict[str, np.ndarray]] = None
    if session_quality_ok:
        method = str(ecfg.get("decomposition_method", "neurokit"))
        session = _decompose_session_neurokit2(eda_ds, eda_sfreq, method)
        if session is None:
            session = _decompose_session_scipy(
                eda_ds,
                eda_sfreq,
                scr_min_distance_s=float(ecfg.get("scr_min_distance_s", 1.0)),
                scr_prominence_mult=float(ecfg.get("scr_prominence_mult", 0.05)),
                tonic_window_s=float(ecfg.get("tonic_window_s", 10.0)),
            )

    min_epoch_coverage_s = float(ecfg.get("min_epoch_coverage_s", 2.0))
    min_samples_ds = max(1, int(round(min_epoch_coverage_s * eda_sfreq)))

    records: List[Dict[str, Any]] = []
    for epoch_idx in range(n_epochs):
        if chosen_epochs is not None and epoch_idx not in chosen_epochs:
            continue
        start_idx = epoch_idx * epoch_step_samples
        end_idx = start_idx + epoch_length_samples
        if end_idx > n_samples:
            break

        if session is not None:
            # Map original-sfreq sample bounds to downsampled-sfreq indices.
            i0 = int(round(start_idx / sfreq * eda_sfreq))
            i1 = min(int(round(end_idx / sfreq * eda_sfreq)), len(session["tonic"]))
            row = _epoch_features(session, i0, i1, eda_sfreq, min_samples_ds, source_channel)
        else:
            row = _nan_epoch_row(source_channel)

        row["epoch_id"] = epoch_idx
        row["t_start"] = start_idx / sfreq
        row["t_end"] = end_idx / sfreq
        records.append(row)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    logger.info(
        "EDA v0.1: computed %d epochs; qc_ok fraction=%.2f (session_quality_ok=%s)",
        len(df),
        float(df["qc_ok_eda"].mean()) if "qc_ok_eda" in df.columns else 0.0,
        session_quality_ok,
    )
    return df
