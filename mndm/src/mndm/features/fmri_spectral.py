"""Spectral complexity metrics for fMRI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
from scipy.signal import welch

@dataclass
class SpectralConfig:
    spectral_entropy: bool


def compute_spectral_fractal_features(
    roi_ts: np.ndarray,
    sfreq: float,
    config: Mapping[str, object] | None,
) -> Dict[str, float]:
    """Compute spectral entropy only (no DFA/PLE in ingest)."""

    if roi_ts.ndim != 2:
        raise ValueError("roi_ts must be 2-D (n_rois, n_times)")
    if sfreq <= 0:
        raise ValueError("sfreq must be positive")

    cfg = _parse_config(config)
    features: Dict[str, float] = {}

    freqs, psd = welch(roi_ts, fs=sfreq, nperseg=min(roi_ts.shape[1], 256), axis=1)

    if cfg.spectral_entropy:
        se = [_spectral_entropy(psd_row, freqs) for psd_row in psd]
        features["fmri_spectral_entropy_global"] = float(np.nanmean(se))

    return features


def _parse_config(config: Mapping[str, object] | None) -> SpectralConfig:
    cfg = config or {}
    return SpectralConfig(
        spectral_entropy=bool(cfg.get("spectral_entropy", True)),
    )


def _spectral_entropy(psd: np.ndarray, freqs: np.ndarray) -> float:
    psd = np.abs(psd)
    psd /= np.sum(psd) + 1e-12
    return float(-np.sum(psd * np.log(psd + 1e-12)))

