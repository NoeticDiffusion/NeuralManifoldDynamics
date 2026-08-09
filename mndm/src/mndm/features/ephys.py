"""Per-window population features for binned extracellular spike rates."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


def compute_ephys_features(signals: Mapping[str, Any], config: Mapping[str, Any]) -> pd.DataFrame:
    """Compute population-rate metrics from ``signals['ephys']``.

    The input has shape ``[n_units, n_rate_bins]`` in Hz.  This intentionally
    follows the fMRI feature-extractor window contract while retaining features
    meaningful for sparse spike-derived population activity.
    """
    signal_map = signals.get("signals", {})
    if "ephys" not in signal_map:
        return pd.DataFrame()
    rates = np.asarray(signal_map["ephys"], dtype=float)
    if rates.ndim != 2:
        raise ValueError("signals['ephys'] must have shape (n_units, n_rate_bins)")
    n_units, n_times = rates.shape
    if n_units == 0 or n_times == 0:
        return pd.DataFrame()
    sfreq = float(signals.get("sfreq", 1.0) or 1.0)
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("ephys sampling frequency must be positive.")

    ephys_cfg = _ephys_config(config)
    window_sec = float(ephys_cfg.get("window_sec", _epoch_value(config, "length_s", 2.0)))
    step_sec = float(ephys_cfg.get("step_sec", _epoch_value(config, "step_s", window_sec)))
    window_samples = max(2, int(round(window_sec * sfreq)))
    step_samples = max(1, int(round(step_sec * sfreq)))
    if n_times < window_samples:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for epoch_id, start in enumerate(range(0, n_times - window_samples + 1, step_samples)):
        end = start + window_samples
        epoch = rates[:, start:end]
        records.append(
            {
                "epoch_id": epoch_id,
                "t_start": start / sfreq,
                "t_end": end / sfreq,
                "ephys_mean_rate_hz": float(np.nanmean(epoch)),
                "ephys_rate_std_hz": float(np.nanstd(epoch)),
                "ephys_population_cv": _coefficient_of_variation(epoch),
                "ephys_participation_ratio": _participation_ratio(epoch),
                "ephys_top_pc_variance_fraction": _top_pc_fraction(epoch),
                "ephys_pairwise_corr_mean": _pairwise_corr_mean(epoch),
                "ephys_rate_entropy": _rate_entropy(epoch),
                "ephys_ar1": _population_ar1(epoch),
                "ephys_active_unit_fraction": float(np.mean(np.nanmean(epoch, axis=1) > 0.0)),
                "ephys_n_units": int(n_units),
                "ephys_window_sec": window_sec,
                "ephys_step_sec": step_sec,
                "ephys_window_samples": window_samples,
                "ephys_step_samples": step_samples,
                "ephys_sfreq": sfreq,
                "dataset_id": signals.get("dataset_id"),
            }
        )
    return pd.DataFrame.from_records(records)


def _ephys_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    features = config.get("features", {}) if isinstance(config, Mapping) else {}
    return features.get("ephys", {}) if isinstance(features, Mapping) else {}


def _epoch_value(config: Mapping[str, Any], key: str, default: float) -> float:
    epoching = config.get("epoching", {}) if isinstance(config, Mapping) else {}
    return float(epoching.get(key, default) if isinstance(epoching, Mapping) else default)


def _coefficient_of_variation(epoch: np.ndarray) -> float:
    mean = float(np.nanmean(epoch))
    return float(np.nanstd(epoch) / mean) if np.isfinite(mean) and mean > 0 else float("nan")


def _covariance_eigenvalues(epoch: np.ndarray) -> np.ndarray:
    if epoch.shape[0] < 2 or epoch.shape[1] < 2:
        return np.empty(0)
    clean = np.nan_to_num(epoch, nan=0.0)
    covariance = np.cov(clean)
    return np.clip(np.linalg.eigvalsh(np.atleast_2d(covariance)), 0.0, None)


def _participation_ratio(epoch: np.ndarray) -> float:
    values = _covariance_eigenvalues(epoch)
    denominator = float(np.sum(values**2))
    return float(np.sum(values) ** 2 / denominator) if denominator > 0 else float("nan")


def _top_pc_fraction(epoch: np.ndarray) -> float:
    values = _covariance_eigenvalues(epoch)
    total = float(np.sum(values))
    return float(values[-1] / total) if len(values) and total > 0 else float("nan")


def _pairwise_corr_mean(epoch: np.ndarray) -> float:
    if epoch.shape[0] < 2:
        return float("nan")
    corr = np.corrcoef(np.nan_to_num(epoch, nan=0.0))
    values = corr[np.triu_indices_from(corr, k=1)]
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else float("nan")


def _rate_entropy(epoch: np.ndarray) -> float:
    distribution = np.clip(np.nanmean(epoch, axis=1), 0.0, None)
    total = float(distribution.sum())
    if total <= 0:
        return float("nan")
    probabilities = distribution / total
    positive = probabilities[probabilities > 0]
    return float(-np.sum(positive * np.log2(positive)) / np.log2(len(probabilities))) if len(probabilities) > 1 else 0.0


def _population_ar1(epoch: np.ndarray) -> float:
    population = np.nanmean(epoch, axis=0)
    if len(population) < 3 or np.nanstd(population[:-1]) == 0 or np.nanstd(population[1:]) == 0:
        return float("nan")
    return float(np.corrcoef(population[:-1], population[1:])[0, 1])
