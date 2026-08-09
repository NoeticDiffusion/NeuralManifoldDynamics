"""Phase anchor features: cardiac and respiratory phase state (Mål B / C).

Computes per-epoch cardiac phase (linear RR interpolation), respiratory phase
(Hilbert transform), heart rate, respiratory rate, inhale fraction, and
heartbeat-evoked potential (HEP) amplitude.  Designed as an *optional* extractor
that integrates with the standard ``mndm.cli features`` pipeline.

Dataset support
---------------
- **RichSleep**: ECG ✓, Resp (Chest) ✓, frontal EEG ✓ → full output
- **ANPHY**:     ECG ✓ (bipolar ECG1−ECG2), Resp ✗ → cardiac arm only
- **BOAS**:      ECG ✗, Resp (PSG_THOR) ✓ → respiratory arm only

Missing signals produce NaN-filled columns, never an error.

Config section (top-level ``phase_anchor`` key in YAML)::

    phase_anchor:
      enabled: false                  # DISABLED by default — opt-in per dataset
      ecg_bipolar: false              # true → compute row[0] − row[1] for 2-ch ECG
      frontal_eeg_channels: []        # channel names for HEP; empty → global mean
      hep_window_lo_s: 0.200          # HEP onset relative to R-peak (s)
      hep_window_hi_s: 0.600          # HEP offset relative to R-peak (s)
      resp_bandpass_lo_hz: 0.10       # respiratory bandpass low cutoff (Hz)
      resp_bandpass_hi_hz: 0.50       # respiratory bandpass high cutoff (Hz)
      chunk_minutes: 5                # R-peak detection chunk size (0 → no chunking)
      min_rpeaks_epoch: 5             # below this → cardiac NaN for that epoch
      min_rpeaks_hep: 3               # minimum peaks for HEP average

Output columns (all float32 except epoch_id and n_rpeaks_in_epoch)
-------------------------------------------------------------------
    epoch_id            int   — epoch index (matches features.parquet join key)
    t_start             float — epoch start time (s)
    t_end               float — epoch end time (s)
    phi_cardiac_mean    float — circular mean cardiac phase ∈ [−π, π)
    phi_resp_mean       float — circular mean respiratory phase ∈ [−π, π)
    rr_interval_ms      float — mean RR interval in epoch (ms)
    hr_bpm              float — mean heart rate (bpm)
    resp_rate_bpm       float — respiratory rate (bpm) from Hilbert phase advance
    inhale_fraction     float — fraction of epoch samples in inhale (0–1)
    hep_amplitude       float — HEP mean amplitude (200–600 ms post-R, EEG units)
    n_rpeaks_in_epoch   int   — R-peak count in epoch
    pa_cardiac_quality  float — fraction of expected beats detected (0–1)
    pa_resp_quality     float — finite fraction of phi_resp samples (0–1)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert

from . import epoch_selection

logger = logging.getLogger(__name__)

_RESP_BP_LO_DEFAULT      = 0.10   # Hz
_RESP_BP_HI_DEFAULT      = 0.50   # Hz
_HEP_WIN_LO_DEFAULT      = 0.200  # s
_HEP_WIN_HI_DEFAULT      = 0.600  # s
_CHUNK_MINUTES_DEFAULT   = 5
_MIN_RPEAKS_EPOCH_DEFAULT = 5
_MIN_RPEAKS_HEP_DEFAULT  = 3


# ─────────────────────────────── Config resolution ───────────────────────────

def _resolve_phase_anchor_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Read the top-level ``phase_anchor`` block with additive defaults."""
    pa = config.get("phase_anchor", {}) if isinstance(config, Mapping) else {}
    if not isinstance(pa, Mapping):
        pa = {}
    out: Dict[str, Any] = dict(pa)
    out.setdefault("enabled", False)
    out.setdefault("ecg_bipolar", False)
    out.setdefault("frontal_eeg_channels", [])
    out.setdefault("hep_window_lo_s", _HEP_WIN_LO_DEFAULT)
    out.setdefault("hep_window_hi_s", _HEP_WIN_HI_DEFAULT)
    out.setdefault("resp_bandpass_lo_hz", _RESP_BP_LO_DEFAULT)
    out.setdefault("resp_bandpass_hi_hz", _RESP_BP_HI_DEFAULT)
    out.setdefault("chunk_minutes", _CHUNK_MINUTES_DEFAULT)
    out.setdefault("min_rpeaks_epoch", _MIN_RPEAKS_EPOCH_DEFAULT)
    out.setdefault("min_rpeaks_hep", _MIN_RPEAKS_HEP_DEFAULT)
    return out


# ─────────────────────────────── Signal extraction ───────────────────────────

def _extract_ecg_1d(
    signals: Mapping[str, Any],
    cfg: Dict[str, Any],
) -> Optional[np.ndarray]:
    """Return 1-D ECG array or None if the ecg modality is absent."""
    ecg_data = signals.get("signals", {}).get("ecg")
    if ecg_data is None:
        return None
    arr = np.asarray(ecg_data, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        if arr.shape[0] == 0:
            return None
        if arr.shape[0] == 1:
            return arr[0]
        if bool(cfg.get("ecg_bipolar", False)):
            logger.debug("phase_anchor ECG: bipolar difference (ch0 − ch1)")
            return arr[0] - arr[1]
        logger.debug("phase_anchor ECG: %d channels, using first", arr.shape[0])
        return arr[0]
    return arr.ravel()


def _extract_resp_1d(signals: Mapping[str, Any]) -> Optional[np.ndarray]:
    """Return 1-D primary resp array or None if the resp modality is absent."""
    resp_data = signals.get("signals", {}).get("resp")
    if resp_data is None:
        return None
    arr = np.asarray(resp_data, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[0] >= 1:
        return arr[0]   # first row = primary channel (preprocessor-ordered)
    return arr.ravel() if arr.size > 0 else None


def _extract_eeg_frontal(
    signals: Mapping[str, Any],
    frontal_names: List[str],
) -> Optional[np.ndarray]:
    """Return mean frontal EEG (1-D) for HEP, or global mean if names unmatched.

    Returns None when the eeg modality is absent.
    """
    eeg_data = signals.get("signals", {}).get("eeg")
    if eeg_data is None:
        return None
    arr = np.asarray(eeg_data, dtype=np.float64)
    if arr.ndim == 1:
        return arr

    # Try to match configured frontal channel names
    if frontal_names:
        channels_map = signals.get("channels")
        eeg_ch_names: List[str] = []
        if isinstance(channels_map, Mapping):
            raw = channels_map.get("eeg", [])
            if isinstance(raw, (list, tuple)):
                eeg_ch_names = [str(n) for n in raw]
        if eeg_ch_names:
            indices = [i for i, n in enumerate(eeg_ch_names) if n in frontal_names]
            if indices:
                logger.debug("phase_anchor HEP: using %d frontal channels %s",
                             len(indices), [eeg_ch_names[i] for i in indices])
                return arr[indices, :].mean(axis=0)
            logger.debug("phase_anchor: frontal channels %s not found in EEG "
                         "(available: %s …); using global mean",
                         frontal_names, eeg_ch_names[:5])

    return arr.mean(axis=0)


# ─────────────────────────────── R-peak detection ────────────────────────────

def _detect_rpeaks_chunked(
    ecg_1d: np.ndarray,
    sfreq: float,
    cfg: Dict[str, Any],
) -> np.ndarray:
    """Chunked R-peak detection for whole-night recordings.

    Delegates to :func:`mndm.features.ecg._detect_rpeaks` (NeuroKit2 /
    scipy_polarity) in ``chunk_minutes``-length windows to avoid memory and
    runtime issues on signals with ~5.7 M samples at 200 Hz.

    Returns an int64 array of R-peak sample indices (global, not per-chunk).
    """
    chunk_min = float(cfg.get("chunk_minutes", _CHUNK_MINUTES_DEFAULT))
    chunk_samples = (max(1, int(chunk_min * 60.0 * sfreq))
                     if chunk_min > 0 else len(ecg_1d))
    n_total  = len(ecg_1d)
    n_chunks = max(1, int(np.ceil(n_total / chunk_samples)))
    all_peaks: List[np.ndarray] = []

    # Try to reuse the pipeline's R-peak detector (avoids code duplication)
    try:
        from .ecg import _detect_rpeaks as _pipeline_detect  # type: ignore
        _have_pipeline_detector = True
    except Exception:
        _pipeline_detect = None
        _have_pipeline_detector = False

    for i in range(n_chunks):
        s = i * chunk_samples
        e = min(s + chunk_samples, n_total)
        chunk = ecg_1d[s:e]
        peaked = False

        if _have_pipeline_detector:
            try:
                peaks, _, _ = _pipeline_detect(
                    chunk,
                    sfreq,
                    peak_detector="neurokit2",
                    bandpass_low_hz=5.0,
                    bandpass_high_hz=min(30.0, sfreq * 0.45),
                    bandpass_order=3,
                    refractory_s=0.45,
                    prominence_mult=1.0,
                )
                all_peaks.append(np.asarray(peaks, dtype=np.int64) + s)
                peaked = True
            except Exception as exc:
                logger.debug("phase_anchor: pipeline detector failed on chunk %d/%d (%s)",
                             i + 1, n_chunks, exc)

        if not peaked:
            # Inline polarity-aware scipy fallback
            try:
                from scipy.signal import find_peaks as _fp
                from scipy.signal import butter as _butter, filtfilt as _ff
                nyq = sfreq * 0.5
                lo, hi = 5.0, min(20.0, nyq * 0.99)
                b, a   = _butter(3, [lo / nyq, hi / nyq], btype="bandpass")
                filt   = _ff(b, a, chunk)
                cen    = filt - np.median(filt)
                mad    = float(np.median(np.abs(cen))) + 1e-8
                prom   = 1.4826 * mad
                d      = max(1, int(0.45 * sfreq))
                pos_s  = np.clip(cen, 0.0, None)
                neg_s  = np.clip(-cen, 0.0, None)
                pp, _p = _fp(pos_s, distance=d, prominence=prom)
                np_, _ = _fp(neg_s, distance=d, prominence=prom)
                peaks  = pp if len(pp) >= len(np_) else np_
                all_peaks.append(peaks.astype(np.int64) + s)
            except Exception as exc2:
                logger.debug("phase_anchor: scipy fallback failed on chunk %d/%d (%s)",
                             i + 1, n_chunks, exc2)

    if not all_peaks:
        return np.array([], dtype=np.int64)
    combined = np.concatenate(all_peaks).astype(np.int64)
    combined.sort()
    return combined


# ─────────────────────────────── Phase helpers ───────────────────────────────

def _resp_phase_continuous(
    resp_1d: np.ndarray,
    sfreq: float,
    cfg: Dict[str, Any],
) -> np.ndarray:
    """Instantaneous respiratory phase via Hilbert on bandpassed signal.

    Returns an array of length ``len(resp_1d)`` with values in [−π, π).
    """
    lo = float(cfg.get("resp_bandpass_lo_hz", _RESP_BP_LO_DEFAULT))
    hi = float(cfg.get("resp_bandpass_hi_hz", _RESP_BP_HI_DEFAULT))
    nyq = sfreq / 2.0
    lo_n = max(lo / nyq, 1e-6)
    hi_n = min(hi / nyq, 1.0 - 1e-6)
    if lo_n < hi_n:
        b, a   = butter(3, [lo_n, hi_n], btype="band")
        resp_f = filtfilt(b, a, resp_1d)
    else:
        logger.debug("phase_anchor: resp bandpass limits invalid; using raw signal")
        resp_f = resp_1d
    return np.angle(hilbert(resp_f)).astype(np.float64)


def _cardiac_phase_at(t_sec: float, rpeaks_sec: np.ndarray) -> float:
    """Cardiac phase phi ∈ [0, 2π) at ``t_sec`` via linear RR interpolation."""
    if len(rpeaks_sec) < 2:
        return float("nan")
    k = int(np.searchsorted(rpeaks_sec, t_sec, side="right")) - 1
    if k < 0 or k >= len(rpeaks_sec) - 1:
        return float("nan")
    rr = rpeaks_sec[k + 1] - rpeaks_sec[k]
    if rr <= 0.0:
        return float("nan")
    return float(2.0 * np.pi * (t_sec - rpeaks_sec[k]) / rr)


def detect_rpeaks_chunked(
    ecg_1d: np.ndarray,
    sfreq: float,
    config: Mapping[str, Any],
) -> np.ndarray:
    """Public run-level R-peak helper shared with Sleep-EAP sidecar exports."""
    return _detect_rpeaks_chunked(ecg_1d, sfreq, dict(config))


def resp_phase_continuous(
    resp_1d: np.ndarray,
    sfreq: float,
    config: Mapping[str, Any],
) -> np.ndarray:
    """Public continuous respiratory phase helper shared with sidecars."""
    return _resp_phase_continuous(resp_1d, sfreq, dict(config))


def cardiac_phase_at(t_sec: float, rpeaks_sec: np.ndarray) -> float:
    """Public scalar cardiac phase helper shared with sidecars."""
    return _cardiac_phase_at(t_sec, rpeaks_sec)


def _circular_mean(angles: np.ndarray) -> float:
    """Circular mean of radian angles → result ∈ [−π, π)."""
    valid = angles[np.isfinite(angles)]
    if len(valid) == 0:
        return float("nan")
    return float(np.angle(np.mean(np.exp(1j * valid))))


def _hep_amplitude(
    eeg_1d: np.ndarray,
    rpeaks_samples: np.ndarray,
    sfreq: float,
    win_lo_s: float,
    win_hi_s: float,
    min_rpeaks: int,
) -> float:
    """Heartbeat-Evoked Potential amplitude for one EEG segment.

    Extracts [win_lo_s, win_hi_s] post-R-peak windows, averages across peaks
    (ERP), then returns the mean of that waveform.  Returns NaN if fewer than
    ``min_rpeaks`` peaks produce valid windows.
    """
    n = len(eeg_1d)
    lo_s = int(round(win_lo_s * sfreq))
    hi_s = int(round(win_hi_s * sfreq))
    if hi_s <= lo_s:
        return float("nan")
    epochs: List[np.ndarray] = []
    for r in rpeaks_samples:
        s, e = int(r) + lo_s, int(r) + hi_s
        if s >= 0 and e <= n:
            epochs.append(eeg_1d[s:e])
    if len(epochs) < min_rpeaks:
        return float("nan")
    erp = np.mean(np.stack(epochs, axis=0), axis=0)
    return float(np.mean(erp))


# ─────────────────────────────── Per-epoch row ───────────────────────────────

def _epoch_row(
    epoch_idx: int,
    s0: int,
    s1: int,
    sfreq: float,
    rpeaks_sec: Optional[np.ndarray],
    phi_resp: Optional[np.ndarray],
    eeg_frontal: Optional[np.ndarray],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute a single epoch row dict."""
    t0 = s0 / sfreq
    t1 = s1 / sfreq
    min_ep  = int(cfg.get("min_rpeaks_epoch", _MIN_RPEAKS_EPOCH_DEFAULT))
    min_hep = int(cfg.get("min_rpeaks_hep", _MIN_RPEAKS_HEP_DEFAULT))
    win_lo  = float(cfg.get("hep_window_lo_s", _HEP_WIN_LO_DEFAULT))
    win_hi  = float(cfg.get("hep_window_hi_s", _HEP_WIN_HI_DEFAULT))

    row: Dict[str, Any] = {"epoch_id": epoch_idx, "t_start": t0, "t_end": t1}

    # ── Cardiac ──────────────────────────────────────────────────────────────
    if rpeaks_sec is not None:
        ep_r  = rpeaks_sec[(rpeaks_sec >= t0) & (rpeaks_sec <= t1)]
        n_r   = len(ep_r)
        row["n_rpeaks_in_epoch"] = n_r

        if n_r >= 2:
            rr_ms = float(np.mean(np.diff(ep_r)) * 1000.0)
            row["rr_interval_ms"] = rr_ms
            row["hr_bpm"]         = 60_000.0 / rr_ms if rr_ms > 0 else float("nan")
        else:
            row["rr_interval_ms"] = float("nan")
            row["hr_bpm"]         = float("nan")

        # Sample phase at epoch midpoint — R-peaks are by definition at phase 0
        # in the linear-interpolation model, so sampling at ep_r times always
        # yields 0 (diary 181 bug A, diary 183 fix).
        row["phi_cardiac_mean"] = _cardiac_phase_at(0.5 * (t0 + t1), rpeaks_sec)
        # Quality: fraction of expected beats at 75 bpm baseline
        expected              = max(1.0, (t1 - t0) / 0.8)
        row["pa_cardiac_quality"] = float(min(1.0, n_r / expected))

        # HEP — only when enough peaks and frontal EEG is available
        if eeg_frontal is not None and n_r >= min_hep:
            ep_samp = (np.round(ep_r * sfreq).astype(np.int64) - s0)
            ep_samp = ep_samp[(ep_samp >= 0) & (ep_samp < (s1 - s0))]
            row["hep_amplitude"] = _hep_amplitude(
                eeg_frontal[s0:s1], ep_samp, sfreq, win_lo, win_hi, min_hep
            )
        else:
            row["hep_amplitude"] = float("nan")
    else:
        for col in ("phi_cardiac_mean", "rr_interval_ms", "hr_bpm",
                    "n_rpeaks_in_epoch", "pa_cardiac_quality", "hep_amplitude"):
            row[col] = float("nan")
        row["n_rpeaks_in_epoch"] = 0

    # ── Respiratory ──────────────────────────────────────────────────────────
    if phi_resp is not None:
        ep_phi = phi_resp[s0 : min(s1, len(phi_resp))]
        row["phi_resp_mean"]  = _circular_mean(ep_phi)
        row["pa_resp_quality"] = (
            float(np.isfinite(ep_phi).mean()) if len(ep_phi) > 0 else float("nan")
        )
        if len(ep_phi) > 1:
            # Inhale = Hilbert phase in (−π, 0): signal rising toward its
            # maximum.  np.diff(unwrap) > 0 is almost always True for a
            # monotonically increasing phase signal (diary 181 bug B fix).
            row["inhale_fraction"] = float(np.mean(ep_phi < 0))
            dur_s  = (min(s1, len(phi_resp)) - s0) / sfreq
            cycles = (np.unwrap(ep_phi)[-1] - np.unwrap(ep_phi)[0]) / (2.0 * np.pi)
            row["resp_rate_bpm"] = float(cycles / dur_s * 60.0) if dur_s > 0 else float("nan")
        else:
            row["inhale_fraction"] = float("nan")
            row["resp_rate_bpm"]   = float("nan")
    else:
        for col in ("phi_resp_mean", "pa_resp_quality", "inhale_fraction", "resp_rate_bpm"):
            row[col] = float("nan")

    return row


# ─────────────────────────────── Public entry point ──────────────────────────

def compute_phase_anchor_features(
    signals: Mapping[str, Any],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Compute per-epoch cardiac and respiratory phase anchor features.

    Parameters
    ----------
    signals : Mapping
        Preprocessed signals dict with ``signals``, ``sfreq``, ``channels``,
        ``dataset_id`` keys (same format used by ecg / resp / cardioresp extractors).
    config : Mapping
        Full ingest configuration dict.  Reads from top-level ``phase_anchor:``
        key only.

    Returns
    -------
    pd.DataFrame
        Per-epoch rows with phase anchor columns aligned on ``epoch_id``.
        Returns an empty DataFrame when ``phase_anchor.enabled: false`` (default)
        or when neither ECG nor resp signals are present.
    """
    pa_cfg = _resolve_phase_anchor_config(config)
    if not bool(pa_cfg.get("enabled", False)):
        return pd.DataFrame()

    sfreq      = float(signals.get("sfreq", 250))
    dataset_id = signals.get("dataset_id")

    ecg_1d      = _extract_ecg_1d(signals, pa_cfg)
    resp_1d     = _extract_resp_1d(signals)
    eeg_frontal = _extract_eeg_frontal(
        signals, list(pa_cfg.get("frontal_eeg_channels") or [])
    )

    if ecg_1d is None and resp_1d is None:
        logger.info("phase_anchor: no ECG and no resp in signals — skipping (%s)", dataset_id)
        return pd.DataFrame()

    # Run-level R-peak detection (chunked, handles whole-night recordings)
    rpeaks_sec: Optional[np.ndarray] = None
    if ecg_1d is not None:
        logger.info("phase_anchor: detecting R-peaks  len=%d sfreq=%.0f Hz …",
                    len(ecg_1d), sfreq)
        rp_samples = _detect_rpeaks_chunked(ecg_1d, sfreq, pa_cfg)
        if len(rp_samples) >= 2:
            rpeaks_sec = rp_samples.astype(np.float64) / sfreq
            mean_hr    = 60.0 / float(np.mean(np.diff(rpeaks_sec)))
            logger.info("phase_anchor: %d R-peaks, mean HR = %.1f bpm",
                        len(rpeaks_sec), mean_hr)
        else:
            logger.warning("phase_anchor: <2 R-peaks detected; cardiac columns → NaN")

    # Continuous respiratory phase via Hilbert transform
    phi_resp: Optional[np.ndarray] = None
    if resp_1d is not None:
        logger.info("phase_anchor: computing respiratory phase (Hilbert) …")
        phi_resp = _resp_phase_continuous(resp_1d, sfreq, pa_cfg)

    # Determine recording length from available signals
    n_samples = int(
        ecg_1d.shape[-1] if ecg_1d is not None
        else resp_1d.shape[-1] if resp_1d is not None
        else eeg_frontal.shape[-1] if eeg_frontal is not None
        else 0
    )

    # Build full epoch index — NO stage-stratified sub-sampling here.
    # The outer-join merge in parallel._merge_feature_frames aligns all epochs
    # using epoch_id, so we produce all epochs and let the EEG rows determine
    # the final sampled set.
    length_s, step_s = epoch_selection.resolve_epoch_params(config, dataset_id)
    epoch_len  = int(length_s * sfreq)
    epoch_step = int(step_s   * sfreq)
    if epoch_len <= 0 or n_samples < epoch_len:
        logger.warning("phase_anchor: no valid epochs (n_samples=%d, epoch_len=%d)",
                       n_samples, epoch_len)
        return pd.DataFrame()

    n_epochs = (n_samples - epoch_len) // epoch_step + 1

    rows: List[Dict[str, Any]] = []
    for idx in range(max(n_epochs, 0)):
        s0 = idx * epoch_step
        s1 = s0 + epoch_len
        if s1 > n_samples:
            break
        rows.append(_epoch_row(idx, s0, s1, sfreq, rpeaks_sec, phi_resp, eeg_frontal, pa_cfg))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Cast to float32 for memory efficiency; keep epoch_id and count as int
    int_cols = {"epoch_id", "n_rpeaks_in_epoch"}
    for col in df.columns:
        if col not in int_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
    return df
