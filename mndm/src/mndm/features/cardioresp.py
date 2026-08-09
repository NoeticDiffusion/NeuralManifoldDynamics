"""Cardiorespiratory Coupling v0.1 feature extraction.

Computes per-epoch metrics that characterise the coupling between cardiac
rhythm (ECG / R-peaks) and respiration (RESP channel).

Feature families
----------------
RSA (Respiratory Sinus Arrhythmia)
    cardioresp_rsa_amplitude   – std of IBI in the respiration frequency band
    cardioresp_rsa_band_lo     – lower edge of the respiratory band used (Hz)
    cardioresp_rsa_band_hi     – upper edge of the respiratory band used (Hz)

Coherence
    cardioresp_coherence       – magnitude-squared coherence between interpolated
                                 IBI time series and RESP signal at respiratory freq

R-peak respiratory phase locking
    cardioresp_rpeak_resp_plv  – phase-locking value of R-peaks w.r.t. RESP
                                 instantaneous phase

HR modulation
    cardioresp_hr_resp_xcorr_peak   – peak Pearson r between HR and RESP
    cardioresp_hr_resp_xcorr_lag_s  – lag (seconds) at peak cross-correlation

Composite anchor index
    cardioresp_anchor_index    – coherence + RSA proxy, reflecting vagal efficiency

Quality
    cardioresp_nn_count        – number of R-peaks in epoch
    cardioresp_resp_quality    – resp signal quality proxy (variance ratio)
    qc_ok_cardioresp           – True when enough R-peaks and RESP quality is OK

Design notes
------------
* The module re-runs R-peak detection on the ECG channel (same detector as
  ``features.ecg``), because the parallel extractor dispatch processes each
  modality independently and does not share intermediate results.
* Coherence is computed with Welch's method on (IBI_interpolated, RESP).
* PLV uses the Hilbert transform to get instantaneous RESP phase at each
  R-peak; the mean resultant length of e^{i·phase} is the PLV.
* For short epochs these estimates are noisy; ``qc_ok_cardioresp`` is set to
  False when there are fewer than 5 cardiac cycles in the epoch.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.interpolate import interp1d

from . import epoch_selection

logger = logging.getLogger(__name__)

_RESP_BAND_LO_DEFAULT = 0.1   # Hz
_RESP_BAND_HI_DEFAULT = 0.5   # Hz
_MIN_RPEAKS_FOR_QC = 5


def _resolve_epoch_params(
    config: Mapping[str, Any], dataset_id: Optional[str]
) -> tuple[float, float]:
    return epoch_selection.resolve_epoch_params(config, dataset_id)


def _resolve_cardioresp_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    features_cfg = config.get("features", {}) if isinstance(config, Mapping) else {}
    cr_cfg = features_cfg.get("cardioresp", {}) if isinstance(features_cfg, Mapping) else {}
    out: Dict[str, Any] = dict(cr_cfg) if isinstance(cr_cfg, Mapping) else {}
    out.setdefault("resp_band_lo", _RESP_BAND_LO_DEFAULT)
    out.setdefault("resp_band_hi", _RESP_BAND_HI_DEFAULT)
    out.setdefault("ibi_interp_hz", 4.0)     # resample rate for IBI time series
    out.setdefault("coherence_nperseg", 128)  # Welch segment length
    out.setdefault("xcorr_max_lag_s", 10.0)  # max lag for HR–RESP cross-correlation
    out.setdefault("min_rpeaks", _MIN_RPEAKS_FOR_QC)
    # Peak detector reused from ECG config
    out.setdefault("peak_detector", "scipy_polarity")
    return out


def _detect_rpeaks_simple(
    ecg_1d: np.ndarray, sfreq: float
) -> np.ndarray:
    """R-peak detector for cardioresp: prefers neurokit2, falls back to scipy_polarity.

    Uses a stricter refractory period (0.45 s → max 133 bpm) and higher
    prominence threshold than the plain fallback to suppress T-wave
    double-detection.
    """
    try:
        from .ecg import _detect_rpeaks  # type: ignore
        # Try neurokit2 first (most reliable, avoids T-wave double-detection).
        peaks, _, detector_used = _detect_rpeaks(
            ecg_1d,
            sfreq,
            peak_detector="neurokit2",
            bandpass_low_hz=5.0,
            bandpass_high_hz=min(30.0, sfreq * 0.45),
            bandpass_order=3,
            refractory_s=0.45,   # min 450 ms between beats (max 133 bpm)
            prominence_mult=1.0,  # stricter than default 0.5
        )
        return np.asarray(peaks, dtype=int)
    except Exception:
        pass

    # Inline fallback: polarity-aware scipy detector
    nyq = sfreq * 0.5
    lo, hi = 5.0, min(30.0, nyq * 0.95)
    try:
        b, a = sp_signal.butter(3, [lo / nyq, hi / nyq], btype="bandpass")
        filt = sp_signal.filtfilt(b, a, ecg_1d)
    except Exception:
        filt = ecg_1d - np.median(ecg_1d)
    centered = filt - np.median(filt)
    mad = float(np.median(np.abs(centered))) + 1e-8
    min_dist = max(1, int(0.45 * sfreq))
    # Use polarity: detect on the dominant-amplitude side only
    pos_side = np.clip(centered, 0.0, None)
    neg_side = np.clip(-centered, 0.0, None)
    score_pos = float(np.sum(pos_side > 1.4826 * mad))
    score_neg = float(np.sum(neg_side > 1.4826 * mad))
    sig = pos_side if score_pos >= score_neg else neg_side
    peaks, _ = sp_signal.find_peaks(
        sig,
        distance=min_dist,
        prominence=1.4826 * mad * 1.0,
    )
    return peaks.astype(int)


def _ibi_series(
    peaks: np.ndarray, sfreq: float, n_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute IBI time series (seconds) at peak times (seconds)."""
    if len(peaks) < 2:
        return np.array([]), np.array([])
    peak_times = peaks / sfreq
    ibi = np.diff(peak_times)        # in seconds
    ibi_times = peak_times[1:]       # time at the *end* of each IBI
    return ibi_times, ibi


def _interpolated_ibi(
    ibi_times: np.ndarray, ibi_vals: np.ndarray, target_hz: float, t_start: float, t_end: float
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate IBI onto a uniform grid."""
    if len(ibi_times) < 2:
        return np.array([]), np.array([])
    mask = (ibi_times >= t_start) & (ibi_times <= t_end)
    t = ibi_times[mask]
    v = ibi_vals[mask]
    if len(t) < 2:
        return np.array([]), np.array([])
    interp_fn = interp1d(t, v, kind="linear", bounds_error=False, fill_value=(v[0], v[-1]))
    t_uniform = np.arange(t_start, t_end, 1.0 / target_hz)
    return t_uniform, interp_fn(t_uniform)


def _magnitude_squared_coherence_at_band(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    band_lo: float,
    band_hi: float,
    nperseg: int,
) -> float:
    """Mean magnitude-squared coherence between x and y in (band_lo, band_hi)."""
    if len(x) < nperseg or len(y) < nperseg:
        nperseg = max(4, min(len(x), len(y)) // 2)
    try:
        f, cxy = sp_signal.coherence(x, y, fs=fs, nperseg=nperseg)
        mask = (f >= band_lo) & (f <= band_hi)
        if not np.any(mask):
            return float("nan")
        return float(np.mean(cxy[mask]))
    except Exception:
        return float("nan")


def _rsa_amplitude(
    ibi: np.ndarray,
    fs: float,
    band_lo: float,
    band_hi: float,
) -> float:
    """Std of IBI after bandpass filtering in the respiratory band (RSA proxy)."""
    if len(ibi) < 8:
        return float("nan")
    nyq = fs * 0.5
    hi = min(band_hi, nyq * 0.95)
    lo = max(0.001, band_lo)
    if hi <= lo:
        return float("nan")
    try:
        b, a = sp_signal.butter(3, [lo / nyq, hi / nyq], btype="bandpass")
        filtered = sp_signal.filtfilt(b, a, ibi)
        return float(np.std(filtered))
    except Exception:
        return float("nan")


def _bandpass_resp(resp_1d: np.ndarray, sfreq: float, lo: float, hi: float) -> np.ndarray:
    """Bandpass the RESP signal for phase estimation.

    Removes DC and slow drift before Hilbert so instantaneous phase reflects
    actual respiratory oscillation, not the carrier DC level.
    """
    nyq = sfreq * 0.5
    hi_safe = min(hi, nyq * 0.95)
    lo_safe = max(0.005, lo)
    if hi_safe <= lo_safe or len(resp_1d) < int(sfreq * 2):
        return resp_1d - np.mean(resp_1d)
    try:
        b, a = sp_signal.butter(3, [lo_safe / nyq, hi_safe / nyq], btype="bandpass")
        return sp_signal.filtfilt(b, a, resp_1d)
    except Exception:
        return resp_1d - np.mean(resp_1d)


def _rpeak_resp_plv(
    peaks: np.ndarray, resp_1d: np.ndarray, sfreq: float,
    band_lo: float = _RESP_BAND_LO_DEFAULT,
    band_hi: float = _RESP_BAND_HI_DEFAULT,
) -> float:
    """Phase-locking value of R-peaks w.r.t. instantaneous RESP phase.

    The RESP signal is bandpass-filtered in (band_lo, band_hi) before the
    Hilbert transform so that DC offsets and slow drifts do not produce
    spuriously high PLV.

    Returns mean resultant length of e^{i * phi(t_rpeak)} in [0, 1].
    """
    if len(peaks) < 3:
        return float("nan")
    try:
        resp_filtered = _bandpass_resp(resp_1d, sfreq, band_lo, band_hi)
        analytic = sp_signal.hilbert(resp_filtered)
        inst_phase = np.angle(analytic)  # -π .. π
        # Clamp to valid sample range
        valid_peaks = peaks[(peaks >= 0) & (peaks < len(resp_1d))]
        if len(valid_peaks) < 3:
            return float("nan")
        phases_at_peaks = inst_phase[valid_peaks]
        # PLV = |mean(e^{i*phi})|
        plv = float(np.abs(np.mean(np.exp(1j * phases_at_peaks))))
        return float(np.clip(plv, 0.0, 1.0))
    except Exception:
        return float("nan")


def _hr_resp_xcorr(
    peaks: np.ndarray,
    resp_1d: np.ndarray,
    sfreq: float,
    max_lag_s: float,
    ibi_interp_hz: float,
) -> tuple[float, float]:
    """Peak cross-correlation and lag between interpolated HR and RESP."""
    if len(peaks) < 4:
        return float("nan"), float("nan")
    try:
        duration_s = len(resp_1d) / sfreq
        ibi_times, ibi_vals = _ibi_series(peaks, sfreq, len(resp_1d))
        if len(ibi_times) < 2:
            return float("nan"), float("nan")
        t_uniform, ibi_interp = _interpolated_ibi(
            ibi_times, ibi_vals, ibi_interp_hz, 0.0, duration_s
        )
        if len(ibi_interp) < 4:
            return float("nan"), float("nan")

        # Downsample resp to match IBI grid
        resp_times = np.arange(len(resp_1d)) / sfreq
        resp_fn = interp1d(
            resp_times, resp_1d, kind="linear",
            bounds_error=False, fill_value=(resp_1d[0], resp_1d[-1])
        )
        resp_interp = resp_fn(t_uniform)

        # Normalise
        ibi_z = (ibi_interp - np.mean(ibi_interp)) / (np.std(ibi_interp) + 1e-8)
        resp_z = (resp_interp - np.mean(resp_interp)) / (np.std(resp_interp) + 1e-8)

        max_lag_samples = int(max_lag_s * ibi_interp_hz)
        xcorr = np.correlate(ibi_z, resp_z, mode="full")
        lags = np.arange(-(len(ibi_z) - 1), len(ibi_z))
        in_range = np.abs(lags) <= max_lag_samples
        xcorr_clipped = xcorr[in_range]
        lags_clipped = lags[in_range]
        if len(xcorr_clipped) == 0:
            return float("nan"), float("nan")
        peak_idx = int(np.argmax(np.abs(xcorr_clipped)))
        peak_r = float(xcorr_clipped[peak_idx]) / max(len(ibi_z), 1)
        peak_lag_s = float(lags_clipped[peak_idx]) / ibi_interp_hz
        return float(np.clip(peak_r, -1.0, 1.0)), peak_lag_s
    except Exception:
        return float("nan"), float("nan")


def compute_cardioresp_features(
    signals: Mapping[str, Any], config: Mapping[str, Any]
) -> pd.DataFrame:
    """Compute per-epoch Cardiorespiratory Coupling v0.1 features.

    Requires both ``ecg`` and ``resp`` channels in *signals*.

    Parameters
    ----------
    signals:
        Preprocessed signals dict.
    config:
        Pipeline configuration dict.

    Returns
    -------
    pd.DataFrame
        Per-epoch cardiorespiratory coupling features.  Empty DataFrame when
        ECG or RESP channels are not available.
    """
    sigs = signals.get("signals", {})
    if "ecg" not in sigs or "resp" not in sigs:
        return pd.DataFrame()

    ecg_arr = np.asarray(sigs["ecg"], dtype=float)
    if ecg_arr.ndim == 1:
        ecg_arr = ecg_arr[None, :]
    if ecg_arr.ndim != 2 or ecg_arr.shape[1] <= 0:
        return pd.DataFrame()

    resp_arr = np.asarray(sigs["resp"], dtype=float)
    if resp_arr.ndim == 1:
        resp_arr = resp_arr[None, :]
    if resp_arr.ndim != 2 or resp_arr.shape[1] <= 0:
        return pd.DataFrame()

    sfreq = float(signals.get("sfreq", 250))
    dataset_id = signals.get("dataset_id")

    length_s, step_s = _resolve_epoch_params(config, dataset_id)
    crcfg = _resolve_cardioresp_config(config)

    epoch_length_samples = int(length_s * sfreq)
    epoch_step_samples = int(step_s * sfreq)
    n_samples = ecg_arr.shape[1]

    if epoch_length_samples <= 0 or epoch_step_samples <= 0 or n_samples < epoch_length_samples:
        return pd.DataFrame()

    n_epochs = (n_samples - epoch_length_samples) // epoch_step_samples + 1

    # Use first channel from each modality
    ecg_1d = ecg_arr[0]
    resp_1d = resp_arr[0]

    # Run-level R-peak detection (shared across epochs for efficiency)
    run_peaks = _detect_rpeaks_simple(ecg_1d, sfreq)
    run_ibi_times, run_ibi_vals = _ibi_series(run_peaks, sfreq, n_samples)

    band_lo = float(crcfg["resp_band_lo"])
    band_hi = float(crcfg["resp_band_hi"])
    ibi_hz = float(crcfg["ibi_interp_hz"])
    nperseg = int(crcfg["coherence_nperseg"])
    max_lag_s = float(crcfg["xcorr_max_lag_s"])
    min_rpeaks = int(crcfg["min_rpeaks"])

    records: List[Dict[str, Any]] = []

    for epoch_idx in range(n_epochs):
        start_idx = epoch_idx * epoch_step_samples
        end_idx = start_idx + epoch_length_samples
        if end_idx > n_samples:
            break

        t_start_s = start_idx / sfreq
        t_end_s = end_idx / sfreq

        epoch_ecg = ecg_1d[start_idx:end_idx]
        epoch_resp = resp_1d[start_idx:end_idx]

        # Peaks within this epoch (from run-level detection)
        epoch_peaks_global = run_peaks[
            (run_peaks >= start_idx) & (run_peaks < end_idx)
        ]
        epoch_peaks_local = epoch_peaks_global - start_idx
        n_peaks = len(epoch_peaks_local)

        qc_ok = bool(n_peaks >= min_rpeaks)

        null_row: Dict[str, Any] = {
            "epoch_id": epoch_idx,
            "t_start": t_start_s,
            "t_end": t_end_s,
            "cardioresp_rsa_amplitude": float("nan"),
            "cardioresp_rsa_band_lo": band_lo,
            "cardioresp_rsa_band_hi": band_hi,
            "cardioresp_coherence": float("nan"),
            "cardioresp_rpeak_resp_plv": float("nan"),
            "cardioresp_hr_resp_xcorr_peak": float("nan"),
            "cardioresp_hr_resp_xcorr_lag_s": float("nan"),
            "cardioresp_coupling_index": float("nan"),
            "cardioresp_nn_count": n_peaks,
            "qc_ok_cardioresp": qc_ok,
        }

        if not qc_ok:
            records.append(null_row)
            continue

        # IBI for this epoch from run-level series
        mask = (run_ibi_times >= t_start_s) & (run_ibi_times <= t_end_s)
        epoch_ibi_times = run_ibi_times[mask]
        epoch_ibi_vals = run_ibi_vals[mask]

        # Interpolated IBI
        t_uniform, ibi_interp = _interpolated_ibi(
            epoch_ibi_times, epoch_ibi_vals, ibi_hz, t_start_s, t_end_s
        )

        rsa_amp = float("nan")
        coherence = float("nan")

        if len(ibi_interp) >= 8:
            rsa_amp = _rsa_amplitude(ibi_interp, ibi_hz, band_lo, band_hi)
            # Downsample RESP to match IBI grid
            resp_times_epoch = np.arange(len(epoch_resp)) / sfreq
            resp_fn = interp1d(
                resp_times_epoch, epoch_resp, kind="linear",
                bounds_error=False,
                fill_value=(epoch_resp[0], epoch_resp[-1]),
            )
            resp_on_ibi_grid = resp_fn(t_uniform - t_start_s)
            coherence = _magnitude_squared_coherence_at_band(
                ibi_interp, resp_on_ibi_grid, ibi_hz, band_lo, band_hi, nperseg
            )

        plv = _rpeak_resp_plv(epoch_peaks_local, epoch_resp, sfreq, band_lo, band_hi)
        xcorr_peak, xcorr_lag = _hr_resp_xcorr(
            epoch_peaks_local, epoch_resp, sfreq, max_lag_s, ibi_hz
        )

        # Composite anchor index: coherence + PLV proxy (both in [0,1])
        anchor_components = [v for v in [coherence, plv] if np.isfinite(v)]
        anchor_idx = float(np.mean(anchor_components)) if anchor_components else float("nan")

        records.append(
            {
                "epoch_id": epoch_idx,
                "t_start": t_start_s,
                "t_end": t_end_s,
                "cardioresp_rsa_amplitude": float(rsa_amp) if np.isfinite(rsa_amp) else float("nan"),
                "cardioresp_rsa_band_lo": band_lo,
                "cardioresp_rsa_band_hi": band_hi,
                "cardioresp_coherence": float(coherence) if np.isfinite(coherence) else float("nan"),
                "cardioresp_rpeak_resp_plv": float(plv) if np.isfinite(plv) else float("nan"),
                "cardioresp_hr_resp_xcorr_peak": float(xcorr_peak) if np.isfinite(xcorr_peak) else float("nan"),
                "cardioresp_hr_resp_xcorr_lag_s": float(xcorr_lag) if np.isfinite(xcorr_lag) else float("nan"),
                "cardioresp_coupling_index": anchor_idx,
                "cardioresp_nn_count": n_peaks,
                "qc_ok_cardioresp": True,
            }
        )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    logger.info(
        "Cardioresp v0.1: computed %d epochs; qc_ok fraction=%.2f",
        len(df),
        float(df["qc_ok_cardioresp"].mean()) if "qc_ok_cardioresp" in df.columns else 0.0,
    )
    return df
