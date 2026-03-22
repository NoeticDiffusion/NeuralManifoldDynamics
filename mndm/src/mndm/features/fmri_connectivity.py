"""Static connectivity metrics for fMRI (FC, ALFF/fALFF)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np
from scipy.signal import butter, filtfilt


@dataclass
class ConnectivityConfig:
    compute_fc: bool
    compute_alff: bool
    compute_falff: bool
    bandpass: Tuple[float, float]


def compute_static_connectivity_features(
    roi_ts: np.ndarray,
    sfreq: float,
    config: Mapping[str, object] | None,
) -> Tuple[Dict[str, float], np.ndarray | None]:
    """Return dict with FC/ALFF summaries and FC matrix (optional)."""

    if roi_ts.ndim != 2:
        raise ValueError("roi_ts must be 2-D (n_rois, n_times)")

    cfg = _parse_config(config)
    features: Dict[str, float] = {}
    fc_matrix = None

    if cfg.compute_fc:
        fc_matrix = np.corrcoef(roi_ts)
        triu = fc_matrix[np.triu_indices_from(fc_matrix, k=1)]
        if triu.size:
            features["fmri_global_FC_mean"] = float(np.nanmean(triu))
            features["fmri_global_FC_std"] = float(np.nanstd(triu))

    if cfg.compute_alff or cfg.compute_falff:
        band_data = _bandpass(roi_ts, sfreq, cfg.bandpass[0], cfg.bandpass[1])
        alff = np.std(band_data, axis=1)
        if cfg.compute_alff:
            features["fmri_ALFF_mean"] = float(np.nanmean(alff))
            features["fmri_ALFF_std"] = float(np.nanstd(alff))
        if cfg.compute_falff:
            full_std = np.std(roi_ts, axis=1)
            falff = alff / (full_std + 1e-12)
            features["fmri_fALFF_mean"] = float(np.nanmean(falff))
            features["fmri_fALFF_std"] = float(np.nanstd(falff))

    return features, fc_matrix


def _parse_config(config: Mapping[str, object] | None) -> ConnectivityConfig:
    """Internal helper: parse config."""
    cfg = config or {}
    return ConnectivityConfig(
        compute_fc=bool(cfg.get("FC_pearson", True)),
        compute_alff=bool(cfg.get("ALFF", True)),
        compute_falff=bool(cfg.get("fALFF", True)),
        bandpass=tuple(cfg.get("alff_band", (0.01, 0.1))),
    )


def _bandpass(data: np.ndarray, sfreq: float, f_low: float, f_high: float, order: int = 4) -> np.ndarray:
    """Internal helper: bandpass."""
    nyq = sfreq / 2.0
    if not (0 < f_low < f_high < nyq):
        return data
    b, a = butter(order, [f_low / nyq, f_high / nyq], btype="band")
    return filtfilt(b, a, data, axis=1)

