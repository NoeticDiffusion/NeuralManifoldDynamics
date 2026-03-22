"""
fmri_phase.py
Phase synchrony metrics (Kuramoto) for fMRI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np
from scipy.signal import butter, filtfilt, hilbert


@dataclass
class PhaseConfig:
    f_low: float
    f_high: float
    regional_sets: Sequence[Mapping[str, object]]


def compute_phase_synchrony_features(
    roi_ts: np.ndarray,
    sfreq: float,
    roi_names: Sequence[str] | None,
    config: Mapping[str, object] | None,
) -> Dict[str, float]:
    """Return Kuramoto-based synchrony metrics."""

    if roi_ts.ndim != 2:
        raise ValueError("roi_ts must be 2-D (n_rois, n_times)")
    if sfreq <= 0:
        raise ValueError("sfreq must be positive")

    cfg = _parse_config(config)
    phases = _bandpass_phase(roi_ts, sfreq, cfg.f_low, cfg.f_high)
    gamma_global = _kuramoto_order_parameter(phases)
    features: Dict[str, float] = {
        "fmri_kuramoto_global_mean": float(np.nanmean(gamma_global)),
        "fmri_kuramoto_global_std": float(np.nanstd(gamma_global)),
    }

    if roi_names and cfg.regional_sets:
        name_to_idx = {name: idx for idx, name in enumerate(roi_names)}
        for entry in cfg.regional_sets:
            set_name = str(entry.get("name"))
            members = entry.get("members") or entry.get("rois_from_network")
            if not members:
                continue
            indices = _resolve_indices(members, name_to_idx)
            if not indices:
                continue
            gamma_reg = _kuramoto_order_parameter(phases[indices])
            features[f"fmri_kuramoto_{set_name}_mean"] = float(np.nanmean(gamma_reg))
            features[f"fmri_kuramoto_{set_name}_std"] = float(np.nanstd(gamma_reg))
    return features


def _parse_config(config: Mapping[str, object] | None) -> PhaseConfig:
    """Internal helper: parse config."""
    cfg = config or {}
    band = cfg.get("bandpass_hz", {"f_low": 0.01, "f_high": 0.1})
    regional_sets = cfg.get("regional_sets", []) if isinstance(cfg, Mapping) else []
    return PhaseConfig(
        f_low=float(band.get("f_low", 0.01)),
        f_high=float(band.get("f_high", 0.1)),
        regional_sets=regional_sets,
    )


def _bandpass_phase(data: np.ndarray, sfreq: float, f_low: float, f_high: float, order: int = 4) -> np.ndarray:
    """Internal helper: bandpass phase."""
    nyq = sfreq / 2.0
    if not (0 < f_low < f_high < nyq):
        filtered = data
    else:
        b, a = butter(order, [f_low / nyq, f_high / nyq], btype="band")
        filtered = filtfilt(b, a, data, axis=1)
    analytic = hilbert(filtered, axis=1)
    return np.angle(analytic)


def _kuramoto_order_parameter(phases: np.ndarray) -> np.ndarray:
    """Internal helper: kuramoto order parameter."""
    n_nodes, n_times = phases.shape
    if n_nodes == 0:
        return np.zeros(n_times, dtype=float)
    z = np.exp(1j * phases)
    order_param = np.abs(np.mean(z, axis=0))
    return order_param


def _resolve_indices(members, name_to_idx) -> Sequence[int]:
    """Internal helper: resolve indices."""
    indices = []
    if isinstance(members, Sequence) and not isinstance(members, str):
        for name in members:
            idx = name_to_idx.get(str(name))
            if idx is not None:
                indices.append(idx)
    else:
        for name, idx in name_to_idx.items():
            if str(name).upper().startswith(str(members).upper()):
                indices.append(idx)
    return indices

