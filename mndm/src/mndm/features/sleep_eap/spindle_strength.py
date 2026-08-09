"""Raw-EEG spindle strength measures for the Sleep-EAP Phase 2 contract."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.signal import welch

from ..eeg import _integrated_bandpower


def _as_mono(eeg: np.ndarray) -> np.ndarray:
    values = np.asarray(eeg, dtype=float)
    if values.ndim == 1:
        return values
    if values.ndim == 2:
        return np.nanmean(values, axis=0)
    raise ValueError("eeg must be a one- or two-dimensional array")


def sigma_power_in_window(
    eeg: np.ndarray,
    *,
    sfreq: float,
    center_sec: float,
    window_sec: float = 2.0,
    band_hz: tuple[float, float] = (12.0, 15.0),
) -> tuple[float, float, str]:
    """Compute absolute 12–15 Hz power and peak-to-peak amplitude around an event."""
    signal = _as_mono(eeg)
    if sfreq <= 0 or window_sec <= 0:
        return np.nan, np.nan, "invalid_parameters"
    half = int(round(window_sec * sfreq / 2.0))
    center = int(round(center_sec * sfreq))
    start, end = center - half, center + half
    if start < 0 or end > signal.size or end - start < max(8, int(sfreq)):
        return np.nan, np.nan, "window_out_of_bounds"
    segment = signal[start:end]
    if not np.isfinite(segment).all():
        return np.nan, np.nan, "nonfinite_eeg"
    nperseg = min(segment.size, max(8, int(round(sfreq * 2.0))))
    freqs, psd = welch(segment, fs=float(sfreq), nperseg=nperseg)
    power = _integrated_bandpower(psd, freqs, float(band_hz[0]), float(band_hz[1]))
    amplitude = float(np.ptp(segment))
    return float(power), amplitude, "ok" if np.isfinite(power) else "insufficient_band_resolution"


def robust_z_against_reference(
    values: Iterable[float],
    reference_values: Iterable[float],
) -> tuple[np.ndarray, float, float]:
    """Robustly standardize values using a fixed reference distribution.

    Returns z scores, reference median, and MAD-derived scale.  An unusable
    reference yields NaNs rather than silently substituting a global cohort
    normalizer.
    """
    value_array = np.asarray(list(values), dtype=float)
    reference = np.asarray(list(reference_values), dtype=float)
    reference = reference[np.isfinite(reference)]
    if reference.size < 3:
        return np.full(value_array.shape, np.nan), np.nan, np.nan
    median = float(np.median(reference))
    scale = float(1.4826 * np.median(np.abs(reference - median)))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        return np.full(value_array.shape, np.nan), median, np.nan
    return (value_array - median) / scale, median, scale


def compute_spindle_strength(
    spindles: pd.DataFrame,
    *,
    eeg: np.ndarray,
    sfreq: float,
    reference_centers_sec: Iterable[float],
    window_sec: float = 2.0,
    band_hz: tuple[float, float] = (12.0, 15.0),
) -> pd.DataFrame:
    """Attach raw sigma power, amplitude, and N2-reference robust z scores.

    The returned table preserves input rows and exposes the normalization
    parameters as repeated provenance columns, allowing downstream consumers to
    recompute or audit every standardised value.
    """
    if "onset_sec" not in spindles:
        raise ValueError("spindles must contain onset_sec")
    out = spindles.copy()
    centers = pd.to_numeric(out.get("peak_sec", out["onset_sec"]), errors="coerce").to_numpy(dtype=float)
    powers: list[float] = []
    amplitudes: list[float] = []
    flags: list[str] = []
    for center in centers:
        power, amplitude, flag = sigma_power_in_window(
            eeg, sfreq=sfreq, center_sec=float(center), window_sec=window_sec, band_hz=band_hz
        )
        powers.append(power)
        amplitudes.append(amplitude)
        flags.append(flag)

    reference_powers = [
        sigma_power_in_window(eeg, sfreq=sfreq, center_sec=float(center), window_sec=window_sec, band_hz=band_hz)[0]
        for center in reference_centers_sec
    ]
    z_scores, ref_median, ref_scale = robust_z_against_reference(powers, reference_powers)
    out["sigma_power"] = np.asarray(powers, dtype=float)
    out["amplitude"] = np.asarray(amplitudes, dtype=float)
    out["sigma_power_z_n2"] = z_scores
    out["sigma_reference_median"] = ref_median
    out["sigma_reference_mad_scale"] = ref_scale
    out["strength_qc_flag"] = flags
    return out
