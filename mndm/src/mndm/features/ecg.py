"""ECG feature extraction (HR/HRV/quality approximations).

Expects a ``signals`` mapping (preprocessed dict) and ingest ``config``; returns
per-epoch HRV columns.

R-peak detection
----------------
Three detectors are available, selected by ``features.ecg.peak_detector`` in config:

``"neurokit2"`` (default when neurokit2 is installed)
    Uses ``neurokit2.ecg_peaks()`` which implements a Pan-Tompkins-style detector.
    Avoids T-wave double-detection.  Requires the ``neurokit2`` package.

``"scipy_polarity"``
    Polarity-aware scipy fallback: applies bandpass, then detects peaks on only
    the dominant-polarity side of the signal (no ``np.abs``).  This avoids
    T-wave double-detection without requiring neurokit2.

``"scipy_abs"`` (legacy — not recommended)
    The original ``np.abs(centered)`` approach.  Prone to T-wave double-detection
    if the T-wave prominence exceeds the refractory threshold.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
from scipy import signal

from . import epoch_selection

logger = logging.getLogger(__name__)


def _resolve_epoch_params(config: Mapping[str, Any], dataset_id: Optional[str]) -> tuple[float, float]:
    """Internal helper: resolve epoch params."""
    return epoch_selection.resolve_epoch_params(config, dataset_id)


def _resolve_ecg_feature_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve ECG-specific feature config with additive defaults."""
    features_cfg = config.get("features", {}) if isinstance(config, Mapping) else {}
    ecg_cfg = features_cfg.get("ecg", {}) if isinstance(features_cfg, Mapping) else {}
    out = dict(ecg_cfg) if isinstance(ecg_cfg, Mapping) else {}
    # Default peak detector: neurokit2 when available, else polarity-aware scipy.
    out.setdefault("peak_detector", "neurokit2")
    return out


def _detect_rpeaks_neurokit2(
    ecg_1d: np.ndarray,
    sfreq: float,
) -> Optional[np.ndarray]:
    """R-peak detection via NeuroKit2 (Pan-Tompkins-style). Returns None on failure."""
    try:
        import warnings
        import neurokit2 as nk  # type: ignore
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, info = nk.ecg_process(ecg_1d, sampling_rate=int(sfreq))
        peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
        return peaks
    except ImportError:
        logger.debug("neurokit2 not installed; falling back to scipy detector")
        return None
    except Exception as exc:
        logger.warning("neurokit2 R-peak detection failed: %s — falling back", exc)
        return None


def _detect_rpeaks_scipy_polarity(
    ecg_1d: np.ndarray,
    sfreq: float,
    *,
    bandpass_low_hz: float,
    bandpass_high_hz: float,
    bandpass_order: int,
    refractory_s: float,
    prominence_mult: float,
) -> np.ndarray:
    """Polarity-aware scipy R-peak detector.

    Applies a bandpass filter then finds peaks on *only* the dominant side of the
    signal (positive or negative polarity), never on ``np.abs()``.  This prevents
    T-wave deflections of the opposite polarity from being double-counted.
    """
    nyquist = sfreq * 0.5
    hi = min(bandpass_high_hz, nyquist * 0.99)
    lo = max(0.01, bandpass_low_hz)
    if hi <= lo:
        lo, hi = 5.0, min(20.0, nyquist * 0.99)
    try:
        b, a = signal.butter(bandpass_order, [lo / nyquist, hi / nyquist], btype="bandpass")
        filt = signal.filtfilt(b, a, ecg_1d)
    except Exception:
        logger.debug("bandpass failed; using demeaned signal")
        filt = ecg_1d - np.median(ecg_1d)
    centered = filt - np.median(filt)
    min_dist = max(1, int(round(refractory_s * sfreq)))

    def _find_side(sig: np.ndarray) -> tuple[np.ndarray, float]:
        clipped = np.clip(sig, 0.0, None)
        pos_vals = clipped[clipped > 0]
        if pos_vals.size == 0:
            return np.array([], dtype=int), 0.0
        mad = float(np.median(np.abs(pos_vals))) + 1e-8
        prom = max(1e-6, prominence_mult * 1.4826 * mad)
        peaks, props = signal.find_peaks(clipped, distance=min_dist, prominence=prom)
        total_prom = float(np.sum(props["prominences"])) if peaks.size > 0 else 0.0
        return peaks.astype(int), total_prom

    peaks_pos, score_pos = _find_side(centered)
    peaks_neg, score_neg = _find_side(-centered)
    # Choose the side whose detected peaks have the greater total prominence.
    peaks = peaks_pos if score_pos >= score_neg else peaks_neg
    return peaks


def _detect_rpeaks_scipy_abs(
    ecg_1d: np.ndarray,
    sfreq: float,
    *,
    bandpass_low_hz: float,
    bandpass_high_hz: float,
    bandpass_order: int,
    refractory_s: float,
    prominence_mult: float,
) -> np.ndarray:
    """Legacy abs-signal detector (prone to T-wave double-detection — kept for reference)."""
    nyquist = sfreq * 0.5
    hi = min(bandpass_high_hz, nyquist * 0.99)
    lo = max(0.01, bandpass_low_hz)
    if hi <= lo:
        lo, hi = 5.0, min(20.0, nyquist * 0.99)
    try:
        b, a = signal.butter(bandpass_order, [lo / nyquist, hi / nyquist], btype="bandpass")
        filt = signal.filtfilt(b, a, ecg_1d)
    except Exception:
        filt = ecg_1d - np.median(ecg_1d)
    centered = filt - np.median(filt)
    sig = np.abs(centered)
    mad = float(np.median(np.abs(centered))) + 1e-8
    prom = max(1e-6, prominence_mult * 1.4826 * mad)
    min_dist = max(1, int(round(refractory_s * sfreq)))
    peaks, _ = signal.find_peaks(sig, distance=min_dist, prominence=prom)
    return peaks.astype(int)


def _detect_rpeaks(
    ecg_1d: np.ndarray,
    sfreq: float,
    *,
    peak_detector: str,
    bandpass_low_hz: float,
    bandpass_high_hz: float,
    bandpass_order: int,
    refractory_s: float,
    prominence_mult: float,
) -> tuple[np.ndarray, bool, str]:
    """Dispatch to the configured R-peak detector with automatic polarity correction.

    Returns
    -------
    peaks : np.ndarray
        Sample indices of detected R-peaks (on the *original* signal scale).
    polarity_inverted : bool
        True when the ECG signal was inverted before detection because QRS
        deflections were predominantly negative.
    detector_used : str
        Name of the detector that produced the final peaks.
    """
    detector = str(peak_detector or "neurokit2").strip().lower()

    # --- generic polarity detection (detector-agnostic) ---
    polarity, ecg_for_detection = _apply_polarity_correction(
        ecg_1d,
        sfreq=sfreq,
        bandpass_low_hz=bandpass_low_hz,
        bandpass_high_hz=bandpass_high_hz,
        bandpass_order=bandpass_order,
    )
    polarity_inverted = polarity == -1

    scipy_kwargs = dict(
        bandpass_low_hz=bandpass_low_hz,
        bandpass_high_hz=bandpass_high_hz,
        bandpass_order=bandpass_order,
        refractory_s=refractory_s,
        prominence_mult=prominence_mult,
    )
    if detector == "neurokit2":
        peaks = _detect_rpeaks_neurokit2(ecg_for_detection, sfreq)
        if peaks is None:
            logger.debug("Falling back to scipy_polarity detector")
            peaks = _detect_rpeaks_scipy_polarity(ecg_for_detection, sfreq, **scipy_kwargs)
            return peaks, polarity_inverted, "scipy_polarity"
        return peaks, polarity_inverted, "neurokit2"
    if detector == "scipy_polarity":
        return _detect_rpeaks_scipy_polarity(ecg_for_detection, sfreq, **scipy_kwargs), polarity_inverted, "scipy_polarity"
    if detector == "scipy_abs":
        return _detect_rpeaks_scipy_abs(ecg_1d, sfreq, **scipy_kwargs), False, "scipy_abs"
    logger.warning("Unknown peak_detector '%s'; using scipy_polarity", detector)
    return _detect_rpeaks_scipy_polarity(ecg_for_detection, sfreq, **scipy_kwargs), polarity_inverted, "scipy_polarity"


def _apply_polarity_correction(
    ecg_1d: np.ndarray,
    *,
    sfreq: float,
    bandpass_low_hz: float,
    bandpass_high_hz: float,
    bandpass_order: int,
) -> tuple[int, np.ndarray]:
    """Detect QRS polarity and return (polarity, signal_for_detection).

    Parameters
    ----------
    ecg_1d : 1-D signal array.

    Returns
    -------
    polarity : int
        +1 if QRS peaks are predominantly positive, -1 if negative.
    signal_for_detection : np.ndarray
        The signal ready for peak detection (inverted when polarity == -1).

    Notes
    -----
    Uses a simple percentile heuristic on the bandpass-filtered signal:
    compare the 99th percentile (positive excursion) against the absolute
    value of the 1st percentile (negative excursion).  The larger side
    is assumed to contain the QRS complex.  This is robust to baseline
    wander and does not require any prior peak detection.
    """
    try:
        nyquist = sfreq * 0.5
        hi = min(bandpass_high_hz, nyquist * 0.99)
        lo = max(0.01, bandpass_low_hz)
        if hi <= lo:
            lo, hi = 5.0, min(20.0, nyquist * 0.99)
        b, a = signal.butter(bandpass_order, [lo / nyquist, hi / nyquist], btype="bandpass")
        filt = signal.filtfilt(b, a, ecg_1d)
    except Exception:
        filt = ecg_1d - float(np.median(ecg_1d))

    p99 = float(np.percentile(filt, 99))
    p01 = float(np.percentile(filt, 1))
    if abs(p01) > abs(p99) * 1.2:
        # Negative QRS dominant — invert for detection
        return -1, -ecg_1d
    return 1, ecg_1d
    if detector == "scipy_abs":
        return _detect_rpeaks_scipy_abs(ecg_1d, sfreq, **scipy_kwargs)
    logger.warning("Unknown peak_detector '%s'; using scipy_polarity", detector)
    return _detect_rpeaks_scipy_polarity(ecg_1d, sfreq, **scipy_kwargs)


def _resolve_hrv_superwindow_config(ecg_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve optional ECG HRV superwindow policy."""
    raw = ecg_cfg.get("hrv", {}) if isinstance(ecg_cfg, Mapping) else {}
    cfg = dict(raw) if isinstance(raw, Mapping) else {}
    cfg.setdefault("enabled", False)
    cfg.setdefault("superwindow_s", 60.0)
    cfg.setdefault("window_mode", "centered")
    cfg.setdefault("min_nn_intervals", 20)
    cfg.setdefault("min_coverage_fraction", 0.5)
    cfg.setdefault("max_artifact_fraction", 0.25)
    cfg.setdefault("pnn50_threshold_ms", 50.0)
    # Nonlinear complexity metrics (opt-in, requires antropy + nolds).
    complexity_raw = cfg.get("complexity", {})
    complexity = dict(complexity_raw) if isinstance(complexity_raw, Mapping) else {}
    complexity.setdefault("enabled", False)
    complexity.setdefault("sampen_order", 2)
    complexity.setdefault("sampen_tolerance_mult", 0.2)
    complexity.setdefault("min_nn_for_sampen", 50)
    complexity.setdefault("dfa_short_nvals_lo", 4)
    complexity.setdefault("dfa_short_nvals_hi", 12)
    complexity.setdefault("min_nn_for_dfa", 16)
    cfg["complexity"] = complexity
    return cfg


def _window_bounds(
    *,
    center_s: float,
    window_s: float,
    total_duration_s: float,
    mode: str,
) -> tuple[float, float]:
    """Resolve superwindow bounds around one epoch center."""
    if window_s <= 0:
        return max(0.0, min(center_s, total_duration_s)), max(0.0, min(center_s, total_duration_s))

    mode_norm = str(mode or "centered").strip().lower()
    if mode_norm == "trailing":
        start = center_s - window_s
        end = center_s
    else:
        half = 0.5 * window_s
        start = center_s - half
        end = center_s + half
    if total_duration_s <= window_s:
        return 0.0, max(total_duration_s, 0.0)
    if start < 0.0:
        end = min(total_duration_s, end - start)
        start = 0.0
    if end > total_duration_s:
        start = max(0.0, start - (end - total_duration_s))
        end = total_duration_s
    return float(max(0.0, start)), float(min(total_duration_s, end))


def _quality_score_from_support(
    *,
    nn_count: int,
    min_nn_intervals: int,
    coverage_fraction: float,
    artifact_fraction: float,
) -> float:
    """Build a compact 0-1 quality score for HRV superwindow estimates."""
    nn_support = min(1.0, float(nn_count) / max(float(min_nn_intervals), 1.0))
    coverage = float(np.clip(coverage_fraction, 0.0, 1.0))
    artifact_penalty = 1.0 - float(np.clip(artifact_fraction, 0.0, 1.0))
    return float(np.clip(nn_support * coverage * artifact_penalty, 0.0, 1.0))


def _compute_hrv_complexity(
    nn: np.ndarray,
    *,
    sampen_order: int = 2,
    sampen_tolerance_mult: float = 0.2,
    min_nn_for_sampen: int = 50,
    dfa_short_nvals_lo: int = 4,
    dfa_short_nvals_hi: int = 12,
    min_nn_for_dfa: int = 16,
) -> Dict[str, Any]:
    """Compute nonlinear HRV complexity metrics from an NN interval series.

    Metrics
    -------
    ``ecg_hrv_sampen``
        Sample Entropy (order ``sampen_order``, tolerance ``sampen_tolerance_mult × std(nn)``).
        Measures signal irregularity: higher values → more complex / less predictable
        beat-to-beat pattern.  Requires ``antropy``.
    ``ecg_hrv_dfa_alpha1``
        Short-range Detrended Fluctuation Analysis scaling exponent (α₁), computed
        over lag range ``[dfa_short_nvals_lo, dfa_short_nvals_hi)``.
        α₁ ≈ 0.5 → uncorrelated; α₁ ≈ 1.0 → 1/f (healthy HRV); α₁ > 1.5 → correlated.
        Requires ``nolds``.

    All metrics return NaN when fewer than the minimum required samples are
    available, or when the external library is not installed.

    Parameters
    ----------
    nn:
        Valid NN intervals (seconds), already artifact-filtered.
    sampen_order:
        Embedding dimension *m* for Sample Entropy (default 2).
    sampen_tolerance_mult:
        Tolerance *r* = ``sampen_tolerance_mult × std(nn)`` (default 0.2).
    min_nn_for_sampen:
        Minimum NN count required to compute SampEn (default 50).
    dfa_short_nvals_lo, dfa_short_nvals_hi:
        Inclusive lower and exclusive upper bound for the short-range DFA lag
        window [lo, hi) (default [4, 12)).
    min_nn_for_dfa:
        Minimum NN count required to compute DFA α₁ (default 16).

    Returns
    -------
    Dict with keys ``ecg_hrv_sampen`` and ``ecg_hrv_dfa_alpha1``.
    """
    _nan = float("nan")
    out: Dict[str, Any] = {"ecg_hrv_sampen": _nan, "ecg_hrv_dfa_alpha1": _nan}
    n = int(nn.size)

    # Sample Entropy
    if n >= min_nn_for_sampen:
        try:
            import antropy  # type: ignore
            tolerance = sampen_tolerance_mult * float(np.std(nn, ddof=1))
            if tolerance > 0:
                out["ecg_hrv_sampen"] = float(
                    antropy.sample_entropy(nn, order=sampen_order, tolerance=tolerance)
                )
        except ImportError:
            logger.debug("antropy not installed; skipping SampEn")
        except Exception as exc:
            logger.debug("SampEn computation failed: %s", exc)

    # DFA α₁ (short-range)
    if n >= min_nn_for_dfa:
        try:
            import nolds  # type: ignore
            import warnings
            nvals = range(dfa_short_nvals_lo, dfa_short_nvals_hi)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                alpha1 = nolds.dfa(nn, nvals=nvals, fit_exp="poly")
            if np.isfinite(alpha1):
                out["ecg_hrv_dfa_alpha1"] = float(alpha1)
        except ImportError:
            logger.debug("nolds not installed; skipping DFA")
        except Exception as exc:
            logger.debug("DFA computation failed: %s", exc)

    return out


def _compute_hrv_superwindow_metrics(
    *,
    peak_times_s: np.ndarray,
    rr_all_s: np.ndarray,
    center_s: float,
    total_duration_s: float,
    rr_min_s: float,
    rr_max_s: float,
    cfg: Mapping[str, Any],
    task_label_intervals: Optional[Dict[str, List[tuple]]] = None,
) -> Dict[str, Any]:
    """Compute one aligned HRV estimate on a longer superwindow.

    Parameters
    ----------
    task_label_intervals : dict, optional
        Mapping of task-label name -> list of (start_s, end_s) intervals
        active during the recording.  Used to compute per-label overlap
        fractions for the HRV superwindow (contamination provenance).
        Example: {"Digits_Retrieval": [(18.0, 36.0), (64.0, 82.0)], ...}
    """
    window_s = float(cfg.get("superwindow_s", 60.0) or 60.0)
    min_nn_intervals = int(cfg.get("min_nn_intervals", 20) or 20)
    min_coverage_fraction = float(cfg.get("min_coverage_fraction", 0.5) or 0.5)
    max_artifact_fraction = float(cfg.get("max_artifact_fraction", 0.25) or 0.25)
    pnn50_threshold_s = float(cfg.get("pnn50_threshold_ms", 50.0) or 50.0) / 1000.0
    window_mode = str(cfg.get("window_mode", "centered") or "centered")
    exclude_labels: List[str] = list(cfg.get("exclude_labels", []) or [])

    sw_start_s, sw_end_s = _window_bounds(
        center_s=center_s,
        window_s=window_s,
        total_duration_s=total_duration_s,
        mode=window_mode,
    )
    sw_duration = max(sw_end_s - sw_start_s, 1e-6)

    left = int(np.searchsorted(peak_times_s, sw_start_s, side="left"))
    right = int(np.searchsorted(peak_times_s, sw_end_s, side="left"))
    raw_rr = rr_all_s[left : max(left, right - 1)]
    valid_mask = (raw_rr >= rr_min_s) & (raw_rr <= rr_max_s)
    nn = raw_rr[valid_mask]
    invalid_count = int(np.sum(~valid_mask)) if raw_rr.size > 0 else 0
    nn_count = int(nn.size)
    coverage_fraction = float((sw_end_s - sw_start_s) / window_s) if window_s > 0 else 0.0
    artifact_fraction = float(invalid_count / raw_rr.size) if raw_rr.size > 0 else np.nan
    ibi_mean_ms = float(np.mean(nn) * 1000.0) if nn_count > 0 else np.nan
    hr_mean_bpm = float(60.0 / np.mean(nn)) if nn_count > 0 and np.mean(nn) > 0 else np.nan
    sdnn_ms = float(np.std(nn, ddof=1) * 1000.0) if nn_count >= max(min_nn_intervals, 2) else np.nan
    dnn = np.diff(nn) if nn_count >= 3 else np.asarray([], dtype=float)
    rmssd_ms = float(np.sqrt(np.mean(dnn ** 2)) * 1000.0) if dnn.size > 0 else np.nan
    pnn50 = float(np.mean(np.abs(dnn) > pnn50_threshold_s)) if dnn.size > 0 else np.nan
    quality_score = _quality_score_from_support(
        nn_count=nn_count,
        min_nn_intervals=min_nn_intervals,
        coverage_fraction=coverage_fraction,
        artifact_fraction=artifact_fraction if np.isfinite(artifact_fraction) else 1.0,
    )
    qc_ok = bool(
        nn_count >= min_nn_intervals
        and coverage_fraction >= min_coverage_fraction
        and (not np.isfinite(artifact_fraction) or artifact_fraction <= max_artifact_fraction)
    )
    result: Dict[str, Any] = {
        "ecg_hrv_hr_mean_bpm": hr_mean_bpm,
        "ecg_hrv_ibi_mean_ms": ibi_mean_ms,
        "ecg_hrv_sdnn_ms": sdnn_ms,
        "ecg_hrv_rmssd_ms": rmssd_ms,
        "ecg_hrv_pnn50": pnn50,
        "ecg_hrv_nn_count": nn_count,
        "ecg_hrv_artifact_fraction": artifact_fraction,
        "ecg_hrv_coverage_fraction": coverage_fraction,
        "ecg_hrv_quality_score": quality_score,
        "qc_ok_ecg_hrv": qc_ok,
    }

    # --- Contamination provenance (generic, config-driven) ---
    # Compute overlap of the HRV superwindow with each task-label interval.
    # Exports: ecg_hrv_dominant_stage_label, ecg_hrv_dominant_stage_frac,
    #          ecg_hrv_n_stage_labels, ecg_hrv_contains_excluded_label.
    if task_label_intervals:
        label_overlaps: Dict[str, float] = {}
        for label, intervals in task_label_intervals.items():
            overlap_s = 0.0
            for seg_start, seg_end in intervals:
                overlap_start = max(sw_start_s, float(seg_start))
                overlap_end = min(sw_end_s, float(seg_end))
                if overlap_end > overlap_start:
                    overlap_s += overlap_end - overlap_start
            label_overlaps[label] = min(1.0, overlap_s / sw_duration)

        if label_overlaps:
            dominant_label = max(label_overlaps, key=lambda k: label_overlaps[k])
            dominant_frac = label_overlaps[dominant_label]
            n_labels_present = int(sum(1 for v in label_overlaps.values() if v > 0.01))
            contains_excluded = bool(
                any(
                    label_overlaps.get(lbl, 0.0) > 0.0
                    for lbl in exclude_labels
                )
            )
            result["ecg_hrv_dominant_stage_label"] = dominant_label if dominant_frac > 0.0 else ""
            result["ecg_hrv_dominant_stage_frac"] = dominant_frac
            result["ecg_hrv_n_stage_labels"] = n_labels_present
            result["ecg_hrv_contains_excluded_label"] = contains_excluded
    else:
        result["ecg_hrv_dominant_stage_label"] = ""
        result["ecg_hrv_dominant_stage_frac"] = float("nan")
        result["ecg_hrv_n_stage_labels"] = 0
        result["ecg_hrv_contains_excluded_label"] = False

    # Nonlinear complexity (opt-in via features.ecg.hrv.complexity.enabled).
    complexity_cfg = cfg.get("complexity", {})
    if isinstance(complexity_cfg, Mapping) and bool(complexity_cfg.get("enabled", False)):
        result.update(
            _compute_hrv_complexity(
                nn,
                sampen_order=int(complexity_cfg.get("sampen_order", 2)),
                sampen_tolerance_mult=float(complexity_cfg.get("sampen_tolerance_mult", 0.2)),
                min_nn_for_sampen=int(complexity_cfg.get("min_nn_for_sampen", 50)),
                dfa_short_nvals_lo=int(complexity_cfg.get("dfa_short_nvals_lo", 4)),
                dfa_short_nvals_hi=int(complexity_cfg.get("dfa_short_nvals_hi", 12)),
                min_nn_for_dfa=int(complexity_cfg.get("min_nn_for_dfa", 16)),
            )
        )

    return result


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
        logger.exception("ECG stage-stratified selection failed; continuing with full epochs")
        return None


def _build_task_label_intervals(
    *,
    raw_file_path: Optional[str],
    total_duration_s: float,
    onset_column: str = "onset",
    duration_column: str = "duration",
    trial_type_column: str = "trial_type",
) -> Optional[Dict[str, List[tuple]]]:
    """Build per-label time intervals from a companion BIDS events.tsv.

    Reads the ``*_events.tsv`` file adjacent to ``raw_file_path``, then applies
    last-value-carried-forward (LVCF) to produce continuous ``(start_s, end_s)``
    intervals per ``trial_type`` label.

    Returns ``None`` when no events file is found or the file cannot be parsed.
    This is fully dataset-agnostic — any BIDS recording with a companion events
    file benefits automatically.

    The result can be used by ``_compute_hrv_superwindow_metrics`` to report
    which task phases contributed to each HRV superwindow.
    """
    if not raw_file_path:
        return None
    try:
        import pathlib
        fp = pathlib.Path(str(raw_file_path))
        stem = fp.stem
        # Strip common modality suffixes (_eeg, _ecg, _meg, _ieeg)
        for suffix in ("_eeg", "_ecg", "_meg", "_ieeg", "_bold", "_physio"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        events_path = fp.parent / f"{stem}_events.tsv"
        if not events_path.exists():
            # Try without suffix stripping
            events_path = fp.parent / f"{fp.stem}_events.tsv"
        if not events_path.exists():
            return None

        events_df = pd.read_csv(events_path, sep="\t")
        if onset_column not in events_df.columns or trial_type_column not in events_df.columns:
            return None

        onset_arr = pd.to_numeric(events_df[onset_column], errors="coerce").to_numpy(dtype=float)
        labels_arr = events_df[trial_type_column].astype(str).fillna("").tolist()
        n = len(onset_arr)

        # Read durations when present; fall back to 0 (triggers LVCF below).
        if trial_type_column in events_df.columns and duration_column in events_df.columns:
            dur_arr = pd.to_numeric(events_df[duration_column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        else:
            dur_arr = np.zeros(n, dtype=float)

        intervals: Dict[str, List[tuple]] = {}
        for idx in range(n):
            if not np.isfinite(onset_arr[idx]):
                continue
            label = str(labels_arr[idx]).strip()
            if not label or label.lower() in {"nan", "n/a", "boundary", ""}:
                continue
            start_s = float(onset_arr[idx])
            # Prefer explicit duration when non-zero; otherwise use LVCF
            # (next-event onset), which is correct for instantaneous markers.
            dur = float(dur_arr[idx]) if idx < len(dur_arr) else 0.0
            if dur > 0.0:
                end_s = start_s + dur
            else:
                end_s = float(onset_arr[idx + 1]) if (
                    idx + 1 < n and np.isfinite(onset_arr[idx + 1])
                ) else total_duration_s
            if label not in intervals:
                intervals[label] = []
            intervals[label].append((start_s, end_s))

        return intervals if intervals else None
    except Exception as exc:
        logger.debug("Could not build task label intervals from events: %s", exc)
        return None


def compute_ecg_features(signals: Mapping[str, Any], config: Mapping[str, Any]) -> pd.DataFrame:
    """Compute per-epoch ECG features (HR/HRV/quality).

    Args:
        signals: Preprocessed signals dict with ``signals`` and ``sfreq``.
        config: Configuration with epoching (and optional ``features.ecg``).

    Returns:
        DataFrame with columns such as ``epoch_id``, ``ecg_hr_bpm``, ``ecg_sdnn``, ``ecg_rmssd``.
    """
    if "ecg" not in signals.get("signals", {}):
        return pd.DataFrame()
    
    ecg_arr = np.asarray(signals["signals"]["ecg"], dtype=float)
    if ecg_arr.ndim == 1:
        ecg_arr = ecg_arr[None, :]
    if ecg_arr.ndim != 2 or ecg_arr.shape[1] <= 0:
        return pd.DataFrame()

    sfreq = float(signals.get("sfreq", 250))
    dataset_id = signals.get("dataset_id")
    raw_file_path = signals.get("file_path")
    
    # Get epoching parameters with dataset overrides
    length_s, step_s = _resolve_epoch_params(config, dataset_id)
    
    # Use first ECG channel
    ecg_channel = np.asarray(ecg_arr[0], dtype=float)

    ecg_cfg = _resolve_ecg_feature_config(config)
    hrv_cfg = _resolve_hrv_superwindow_config(ecg_cfg)

    peak_detector = str(ecg_cfg.get("peak_detector", "neurokit2"))
    bandpass_low_hz = float(ecg_cfg.get("bandpass_low_hz", 5.0) or 5.0)
    bandpass_high_hz = float(ecg_cfg.get("bandpass_high_hz", 20.0) or 20.0)
    bandpass_order = int(ecg_cfg.get("bandpass_order", 3) or 3)
    refractory_s = float(ecg_cfg.get("refractory_s", 0.3) or 0.3)
    prominence_mult = float(ecg_cfg.get("prominence_mult", 1.0) or 1.0)
    rr_min_s = float(ecg_cfg.get("rr_min_s", 0.3) or 0.3)
    rr_max_s = float(ecg_cfg.get("rr_max_s", 2.0) or 2.0)
    min_rr_for_sdnn = int(ecg_cfg.get("min_rr_for_sdnn", 2) or 2)
    min_rr_for_rmssd = int(ecg_cfg.get("min_rr_for_rmssd", 3) or 3)
    
    # Epoch the data
    epoch_length_samples = int(length_s * sfreq)
    epoch_step_samples = int(step_s * sfreq)
    n_samples = int(len(ecg_channel))
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

    # Run-level R-peak detection (returns peaks, polarity_inverted, detector_used)
    peaks, ecg_polarity_inverted, ecg_detector_used = _detect_rpeaks(
        ecg_channel,
        sfreq,
        peak_detector=peak_detector,
        bandpass_low_hz=bandpass_low_hz,
        bandpass_high_hz=bandpass_high_hz,
        bandpass_order=bandpass_order,
        refractory_s=refractory_s,
        prominence_mult=prominence_mult,
    )
    logger.debug(
        "R-peak detection (%s, polarity_inverted=%s): %d peaks detected",
        ecg_detector_used, ecg_polarity_inverted, peaks.size,
    )
    peak_times_s = peaks.astype(np.float64) / sfreq
    rr_all_s = np.diff(peak_times_s) if peak_times_s.size >= 2 else np.asarray([], dtype=np.float64)
    total_duration_s = float(n_samples / sfreq)

    # Build task-label intervals for HRV contamination provenance.
    # Reads the companion BIDS events.tsv (if it exists) using LVCF,
    # building per-label time intervals over the full recording.
    # This is dataset-agnostic: any BIDS file with a *_events.tsv works.
    task_label_intervals: Optional[Dict[str, List[tuple]]] = None
    if bool(hrv_cfg.get("enabled", False)):
        task_label_intervals = _build_task_label_intervals(
            raw_file_path=raw_file_path,
            total_duration_s=float(n_samples / sfreq),
            onset_column=str(ecg_cfg.get("events_onset_column", "onset")),
            duration_column=str(ecg_cfg.get("events_duration_column", "duration")),
            trial_type_column=str(ecg_cfg.get("events_trial_type_column", "trial_type")),
        )

    records: List[Dict[str, Any]] = []
    
    for epoch_idx in range(n_epochs):
        if chosen_epochs is not None and epoch_idx not in chosen_epochs:
            continue
        start_idx = epoch_idx * epoch_step_samples
        end_idx = start_idx + epoch_length_samples
        
        if end_idx > n_samples:
            break

        # Count run-level peaks within this epoch and derive epoch-local RR.
        left = int(np.searchsorted(peaks, start_idx, side="left"))
        right = int(np.searchsorted(peaks, end_idx, side="left"))
        epoch_peaks = peaks[left:right]
        rr_intervals = rr_all_s[left : max(left, right - 1)]
        if rr_intervals.size:
            rr_intervals = rr_intervals[(rr_intervals >= rr_min_s) & (rr_intervals <= rr_max_s)]

        sdnn = (
            float(np.std(rr_intervals, ddof=1))
            if rr_intervals.size >= max(min_rr_for_sdnn, 2)
            else np.nan
        )
        drr = np.diff(rr_intervals) if rr_intervals.size >= max(min_rr_for_rmssd, 3) else np.asarray([], dtype=float)
        rmssd = float(np.sqrt(np.mean(drr ** 2))) if drr.size > 0 else np.nan
        rr_mean = float(np.mean(rr_intervals)) if rr_intervals.size > 0 else np.nan
        hr_bpm = float(60.0 / rr_mean) if np.isfinite(rr_mean) and rr_mean > 0 else np.nan
        rr_cv = float(np.std(rr_intervals, ddof=1) / rr_mean) if rr_intervals.size >= 2 and rr_mean > 0 else np.nan
        peak_count = int(epoch_peaks.size)
        quality_score = float(min(1.0, rr_intervals.size / max(length_s * 1.5, 1.0))) if rr_intervals.size > 0 else 0.0
        
        record = {
            "epoch_id": epoch_idx,
            "t_start": start_idx / sfreq,
            "t_end": end_idx / sfreq,
            "ecg_hr_bpm": hr_bpm,
            "ecg_rr_mean": rr_mean,
            "ecg_rr_cv": rr_cv,
            "ecg_sdnn": sdnn,
            "ecg_rmssd": rmssd,
            "ecg_peak_count": peak_count,
            "ecg_quality_score": quality_score,
            "ecg_peak_detector": ecg_detector_used,
            "ecg_polarity_inverted": ecg_polarity_inverted,
            "qc_ok_ecg": bool(np.isfinite(hr_bpm) and peak_count >= 2),
        }
        if bool(hrv_cfg.get("enabled", False)):
            center_s = 0.5 * (float(start_idx / sfreq) + float(end_idx / sfreq))
            record.update(
                _compute_hrv_superwindow_metrics(
                    peak_times_s=peak_times_s,
                    rr_all_s=rr_all_s,
                    center_s=center_s,
                    total_duration_s=total_duration_s,
                    rr_min_s=rr_min_s,
                    rr_max_s=rr_max_s,
                    cfg=hrv_cfg,
                    task_label_intervals=task_label_intervals,
                )
            )
        records.append(record)
    
    df = pd.DataFrame(records)
    logger.info(f"Computed {len(df)} ECG epochs")
    return df


