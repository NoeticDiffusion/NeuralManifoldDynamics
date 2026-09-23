"""Continuous-session preprocessing helpers for fMRI ROI time series."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

logger = logging.getLogger(__name__)

FILTER_STAGE_CONTINUOUS = "fmri_continuous"
FILTER_STAGE_DEFERRED = "deferred_to_fmri_continuous"


def _effective_bandpass_order(n_times: int, order: int = 4) -> int:
    """Largest Butterworth order whose filtfilt padlen fits ``n_times``."""
    eff_order = int(order)
    while eff_order >= 1 and n_times <= 3 * (2 * eff_order + 1):
        eff_order -= 1
    return eff_order


def process_session_signals(
    roi_ts: np.ndarray,
    sfreq: float,
    config: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Apply continuous-domain transforms before epoch slicing.

    Args:
        roi_ts: ROI time series with shape ``(n_regions, n_times)``.
        sfreq: Sampling frequency in Hz.
        config: Optional preprocessing mapping; supports ``f_low``/``f_high``,
            ``bandpass`` as ``[f_low, f_high]``, ``compute_phase`` (default True),
            and ``skip_bandpass`` when an upstream stage already filtered.

    Returns:
        Dict with ``filtered_ts``, ``phase_ts``, and filter provenance keys.
    """
    if roi_ts.ndim != 2:
        raise ValueError("roi_ts must be 2-D (n_regions, n_times)")
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("sfreq must be finite and positive")

    cfg = config if isinstance(config, Mapping) else {}
    band = cfg.get("bandpass")
    if isinstance(band, (list, tuple)) and len(band) >= 2:
        f_low = float(band[0])
        f_high = float(band[1])
    else:
        f_low = float(cfg.get("f_low", 0.01))
        f_high = float(cfg.get("f_high", 0.1))
    compute_phase = bool(cfg.get("compute_phase", True))
    skip_bandpass = bool(cfg.get("skip_bandpass", False))
    n_times = int(np.asarray(roi_ts).shape[1]) if np.asarray(roi_ts).ndim == 2 else int(np.asarray(roi_ts).shape[-1])

    if skip_bandpass:
        filtered_ts = np.asarray(roi_ts, dtype=float)
        filter_skipped = True
        filter_applied = bool(cfg.get("filter_applied", True))
        filter_stage = str(cfg.get("filter_stage") or FILTER_STAGE_CONTINUOUS)
        filter_order = None
    else:
        eff_order = _effective_bandpass_order(n_times, order=4)
        if eff_order < 1:
            filtered_ts = np.asarray(roi_ts, dtype=float)
            filter_applied = False
            filter_skipped = True
            filter_stage = "skipped_too_short"
            filter_order = 0
        else:
            filtered_ts = _bandpass_signal(
                roi_ts, sfreq=sfreq, f_low=f_low, f_high=f_high, order=eff_order
            )
            filter_applied = True
            filter_skipped = False
            filter_stage = FILTER_STAGE_CONTINUOUS
            filter_order = int(eff_order)

    result: Dict[str, Any] = {
        "filtered_ts": filtered_ts,
        "filter_applied": filter_applied,
        "filter_skipped": filter_skipped,
        "filter_stage": filter_stage,
        "filter_bandpass": [float(f_low), float(f_high)],
        "filter_order": filter_order,
    }

    if compute_phase:
        analytic_signal = hilbert(filtered_ts, axis=1)
        result["phase_ts"] = np.angle(analytic_signal)
    else:
        result["phase_ts"] = np.full_like(filtered_ts, np.nan, dtype=float)

    return result


def _bandpass_signal(
    data: np.ndarray,
    sfreq: float,
    f_low: float,
    f_high: float,
    order: int = 4,
) -> np.ndarray:
    """Bandpass filter full session; raise on invalid Nyquist settings.

    ``filtfilt`` default padlen is ``3 * ntaps`` (ntaps = ``2*order+1`` for a
    band filter). Session length that cannot support order 4 degrades order
    the same way as ``fmri_connectivity._bandpass`` (ALFF/fALFF). Order stays
    4 for typical rest runs. If even order 1 does not fit, return unfiltered
    data rather than raising.
    """
    nyq = sfreq / 2.0
    if not (np.isfinite(f_low) and np.isfinite(f_high) and 0 < f_low < f_high < nyq):
        raise ValueError(
            "Invalid bandpass parameters for given sfreq: "
            f"sfreq={sfreq}, nyq={nyq}, f_low={f_low}, f_high={f_high}"
        )
    arr = np.asarray(data, dtype=float)
    n_times = arr.shape[1] if arr.ndim == 2 else arr.shape[-1]
    eff_order = _effective_bandpass_order(n_times, order=order)
    if eff_order < 1:
        logger.warning(
            "Session too short (%d samples) for any stable bandpass order; returning unfiltered data",
            n_times,
        )
        return arr
    if eff_order != int(order):
        logger.debug(
            "Reduced session bandpass order from %d to %d for n_times=%d",
            order,
            eff_order,
            n_times,
        )
    b, a = butter(eff_order, [f_low / nyq, f_high / nyq], btype="band")
    return filtfilt(b, a, arr, axis=1)


def fmri_filter_attrs_from_frame(sub_frame: Any) -> dict[str, Any]:
    """Copy session-filter provenance from a feature table onto HDF5 root attrs."""
    if sub_frame is None or getattr(sub_frame, "empty", True):
        return {}
    attrs: dict[str, Any] = {}
    if "fmri_filter_stage" in sub_frame.columns:
        stages = [
            str(v).strip()
            for v in sub_frame["fmri_filter_stage"].tolist()
            if v is not None and str(v).strip() and str(v).strip().lower() not in {"nan", "none"}
        ]
        if stages:
            attrs["filter_stage"] = stages[0]
    if "fmri_filter_applied" in sub_frame.columns:
        applied = [
            int(bool(v))
            for v in sub_frame["fmri_filter_applied"].tolist()
            if v is not None and str(v).strip().lower() not in {"", "nan", "none"}
        ]
        if applied:
            attrs["filter_applied"] = int(applied[0])
    if "fmri_filter_skipped" in sub_frame.columns:
        skipped = [
            int(bool(v))
            for v in sub_frame["fmri_filter_skipped"].tolist()
            if v is not None and str(v).strip().lower() not in {"", "nan", "none"}
        ]
        if skipped:
            attrs["filter_skipped"] = int(skipped[0])
    if "fmri_filter_order" in sub_frame.columns:
        order = _first_finite(sub_frame, "fmri_filter_order")
        if order is not None:
            attrs["filter_order"] = int(order)
    low = _first_finite(sub_frame, "fmri_filter_bandpass_low")
    high = _first_finite(sub_frame, "fmri_filter_bandpass_high")
    if low is not None:
        attrs["filter_bandpass_low"] = low
    if high is not None:
        attrs["filter_bandpass_high"] = high
    return attrs


def _first_finite(sub_frame: Any, column: str) -> Optional[float]:
    if column not in getattr(sub_frame, "columns", ()):
        return None
    for raw in list(sub_frame[column]):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value == value and value not in (float("inf"), float("-inf")):
            return value
    return None


def maybe_apply_preprocess_bandpass(
    region_array: np.ndarray,
    fmri_cfg: Mapping[str, Any] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """P0.4: never bandpass in preprocess_fmri; return the array unchanged."""
    return np.asarray(region_array), preprocess_filter_meta(fmri_cfg)


def preprocess_filter_meta(fmri_cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    """Provenance when preprocess_fmri defers bandpass to fmri_continuous."""
    ignored: Optional[list[float]] = None
    if isinstance(fmri_cfg, Mapping):
        band = fmri_cfg.get("bandpass")
        if isinstance(band, Sequence) and not isinstance(band, (str, bytes)) and len(band) >= 2:
            try:
                ignored = [float(band[0]), float(band[1])]
            except (TypeError, ValueError):
                ignored = None
    return {
        "filter_applied": False,
        "filter_stage": FILTER_STAGE_DEFERRED,
        "filter_skipped_reason": "canonical_stage_fmri_continuous",
        "filter_ignored_preprocess_bandpass": ignored,
    }
