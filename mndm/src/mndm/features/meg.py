"""MEG feature extraction for electrophysiology shadow mapping.

This v0.1 surface mirrors the minimal EEG feature namespace used by the
existing MNPS stack while preserving explicit MAG/GRAD diagnostics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import signal

from .eeg import (
    _compute_entropy_feature,
    _compute_hjorth_metrics,
    _integrated_bandpower,
    _run_multitaper_psd_safely,
    psd_array_multitaper,
)
from .. import ensembles

logger = logging.getLogger(__name__)

_DEFAULT_MEG_BANDS: Dict[str, list[float]] = {
    "delta": [1.0, 4.0],
    "theta": [4.0, 8.0],
    "alpha": [8.0, 12.0],
    "beta": [13.0, 30.0],
    "gamma": [30.0, 45.0],
}
_POSITIVE_COMBINED_FEATURES = {
    "delta",
    "theta",
    "alpha",
    "beta",
    "gamma",
    "highfreq_power_30_45",
}
_MEG_COMBINED_FEATURES = (
    "delta",
    "theta",
    "alpha",
    "beta",
    "gamma",
    "alpha_theta",
    "beta_alpha",
    "hjorth_mobility",
    "hjorth_complexity",
    "permutation_entropy",
    "sample_entropy",
    "highfreq_power_30_45",
)


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return a finite ratio or NaN when the denominator is not usable."""
    try:
        num = float(numerator)
        den = float(denominator)
    except Exception:
        return np.nan
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0:
        return np.nan
    return float(num / den)


def _resolve_epoch_meta(
    n_samples: int,
    sfreq: float,
    config: Mapping[str, Any],
    dataset_id: Optional[str],
) -> list[tuple[int, int, int]]:
    """Resolve the shared epoch grid used for both MAG and GRAD."""
    epoching = config.get("epoching", {}) if isinstance(config, Mapping) else {}
    length_s = float(epoching.get("length_s", 8.0) or 8.0)
    step_s = float(epoching.get("step_s", 4.0) or 4.0)
    if dataset_id and isinstance(epoching, Mapping):
        ds_map = epoching.get("datasets", {})
        if isinstance(ds_map, Mapping):
            ds_cfg = ds_map.get(dataset_id, {})
            if isinstance(ds_cfg, Mapping):
                if "length_s" in ds_cfg:
                    length_s = float(ds_cfg.get("length_s", length_s) or length_s)
                if "step_s" in ds_cfg:
                    step_s = float(ds_cfg.get("step_s", step_s) or step_s)

    epoch_length_samples = int(length_s * sfreq)
    epoch_step_samples = int(step_s * sfreq)
    if epoch_length_samples <= 0 or epoch_step_samples <= 0 or n_samples < epoch_length_samples:
        return []

    meta: list[tuple[int, int, int]] = []
    n_epochs = (n_samples - epoch_length_samples) // epoch_step_samples + 1
    for epoch_idx in range(max(n_epochs, 0)):
        start_idx = epoch_idx * epoch_step_samples
        end_idx = start_idx + epoch_length_samples
        if end_idx > n_samples:
            break
        meta.append((epoch_idx, start_idx, end_idx))
    return meta


def _resolve_meg_bands(features_cfg: Mapping[str, Any]) -> Dict[str, tuple[float, float]]:
    """Return a complete MEG band dictionary with EEG fallback semantics."""
    bands_cfg = features_cfg.get("meg_bands")
    if not isinstance(bands_cfg, Mapping):
        bands_cfg = features_cfg.get("eeg_bands", {})
    merged: Dict[str, Sequence[float]] = dict(_DEFAULT_MEG_BANDS)
    if isinstance(bands_cfg, Mapping):
        for band_name, band_range in bands_cfg.items():
            if isinstance(band_range, Sequence) and len(band_range) >= 2:
                merged[str(band_name)] = [float(band_range[0]), float(band_range[1])]
    return {
        str(name): (float(bounds[0]), float(bounds[1]))
        for name, bounds in merged.items()
    }


def _resolve_meg_psd_cfg(
    features_cfg: Mapping[str, Any],
    sfreq: float,
) -> tuple[str, Optional[float], float, float]:
    """Resolve PSD settings with MEG-first, EEG-second fallback."""
    psd_cfg = features_cfg.get("meg_psd")
    if not isinstance(psd_cfg, Mapping):
        psd_cfg = features_cfg.get("eeg_psd", {})
    if not isinstance(psd_cfg, Mapping):
        psd_cfg = {}
    psd_method = str(psd_cfg.get("method", "multitaper")).strip().lower()
    mt_bandwidth = psd_cfg.get("bandwidth")
    psd_fmin = float(psd_cfg.get("fmin", 0.0) or 0.0)
    psd_fmax = float(psd_cfg.get("fmax", 50.0) or 50.0)
    nyquist = float(sfreq) * 0.5
    psd_fmax = min(psd_fmax, nyquist)
    if psd_fmax <= psd_fmin:
        psd_fmin = 0.0
        psd_fmax = nyquist
    return psd_method, mt_bandwidth, psd_fmin, psd_fmax


def _sample_entropy(
    data: np.ndarray,
    m: int = 2,
    r_scale: float = 0.2,
    max_n: int = 300,
) -> float:
    """Compute sample entropy (SampEn) via vectorised template matching.

    SampEn = -log(A/B) where B counts pairs of length-m templates within
    Chebyshev distance r, and A counts pairs at length m+1.  Unlike permutation
    entropy it measures regularity via template matching, not ordinal patterns,
    so it is orthogonal to ``_compute_entropy_feature`` (permutation entropy).

    Long signals are uniformly subsampled to ``max_n`` points so that the
    O(n²) template comparison stays fast even for high-sfreq MEG windows.

    Returns NaN when the signal is too short or the B count is zero.
    """
    arr = np.asarray(data, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 2 * (m + 1):
        return np.nan
    if n > max_n:
        idx = np.round(np.linspace(0, n - 1, max_n)).astype(int)
        arr = arr[idx]
        n = len(arr)
    r = r_scale * float(np.std(arr, ddof=1))
    if not np.isfinite(r) or r <= 0:
        return np.nan
    # Build template matrix: shape (n-m, m+1)
    templates = np.lib.stride_tricks.sliding_window_view(arr, m + 1)  # (n-m, m+1)
    tm = templates[:, :m]   # length-m prefixes
    tp1 = templates[:, m]   # m+1-th elements
    b_count = 0
    a_count = 0
    for i in range(len(tm) - 1):
        # Chebyshev distance between template i and all later templates
        diffs = np.abs(tm[i + 1 :] - tm[i])   # (n-m-1-i, m)
        chebyshev = diffs.max(axis=1)           # (n-m-1-i,)
        match_b = chebyshev <= r
        b_count += int(match_b.sum())
        if match_b.any():
            a_count += int((np.abs(tp1[i + 1 :][match_b] - tp1[i]) <= r).sum())
    if b_count == 0:
        return np.nan
    ratio = a_count / b_count
    if ratio <= 0:
        return np.nan
    return float(-np.log(ratio))


def _robust_z(values: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Robust z-score a vector with stable low-variance fallback."""
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return out
    finite = arr[mask]
    center = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= eps:
        std = float(np.nanstd(finite))
        scale = std if np.isfinite(std) and std > eps else 1.0
    out[mask] = (finite - center) / scale
    return out


def _infer_meg_family_indices(names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Fallback split for MEG channel lists when preprocess did not expose families."""
    mag_idx: list[int] = []
    grad_idx: list[int] = []
    for idx, raw_name in enumerate(names):
        name = str(raw_name).strip()
        if not name:
            continue
        if name[-1:] == "1":
            mag_idx.append(idx)
        elif name[-1:] in {"2", "3"}:
            grad_idx.append(idx)
    return np.asarray(mag_idx, dtype=int), np.asarray(grad_idx, dtype=int)


def _resolve_meg_family_arrays(signals: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """Resolve MAG/GRAD arrays from preprocess output with safe fallback."""
    signal_map = signals.get("signals", {}) if isinstance(signals, Mapping) else {}
    channels_map = signals.get("channels", {}) if isinstance(signals, Mapping) else {}
    families: Dict[str, np.ndarray] = {}

    meg_mag = signal_map.get("meg_mag")
    meg_grad = signal_map.get("meg_grad")
    if isinstance(meg_mag, np.ndarray) and meg_mag.ndim == 2 and meg_mag.size:
        families["mag"] = meg_mag
    if isinstance(meg_grad, np.ndarray) and meg_grad.ndim == 2 and meg_grad.size:
        families["grad"] = meg_grad
    if families:
        return families

    meg = signal_map.get("meg")
    if not isinstance(meg, np.ndarray) or meg.ndim != 2 or meg.size == 0:
        return {}

    meg_names = channels_map.get("meg", []) if isinstance(channels_map, Mapping) else []
    if isinstance(meg_names, Sequence) and meg_names:
        mag_idx, grad_idx = _infer_meg_family_indices([str(name) for name in meg_names])
        if mag_idx.size > 0:
            families["mag"] = meg[mag_idx, :]
        if grad_idx.size > 0:
            families["grad"] = meg[grad_idx, :]
    if not families:
        families["grad"] = meg
    return families


def _compute_sensor_family_features(
    data: np.ndarray,
    sfreq: float,
    meta: Sequence[tuple[int, int, int]],
    *,
    features_cfg: Mapping[str, Any],
    prefix: str,
) -> pd.DataFrame:
    """Compute raw per-epoch features for one MEG sensor family."""
    if data.ndim != 2 or data.size == 0 or not meta:
        return pd.DataFrame()

    bands = _resolve_meg_bands(features_cfg)
    psd_method, mt_bandwidth, psd_fmin, psd_fmax = _resolve_meg_psd_cfg(features_cfg, sfreq)
    pe_cfg = features_cfg.get("permutation_entropy", {}) if isinstance(features_cfg, Mapping) else {}
    pe_order = int(pe_cfg.get("order", 5))
    pe_delay = int(pe_cfg.get("delay", 1))
    pe_normalize = bool(pe_cfg.get("normalize", True))

    epochs_agg: list[np.ndarray] = []
    valid_meta: list[tuple[int, int, int]] = []
    for epoch_id, start_idx, end_idx in meta:
        epoch_data = data[:, start_idx:end_idx]
        if epoch_data.size == 0:
            continue
        epochs_agg.append(np.median(epoch_data, axis=0))
        valid_meta.append((epoch_id, start_idx, end_idx))
    if not epochs_agg:
        return pd.DataFrame()

    epochs_arr = np.stack(epochs_agg, axis=0)
    epoch_length_samples = int(valid_meta[0][2] - valid_meta[0][1])
    if psd_method == "multitaper" and psd_array_multitaper is not None:
        psd, freqs = _run_multitaper_psd_safely(
            epochs_arr,
            sfreq=sfreq,
            fmin=psd_fmin,
            fmax=psd_fmax,
            bandwidth=mt_bandwidth,
            adaptive=True,
            normalization="full",
            verbose=False,
        )
    else:
        psds: list[np.ndarray] = []
        freqs = None
        nperseg = min(epoch_length_samples, 512)
        noverlap = nperseg // 2
        for epoch_row in epochs_arr:
            freq_row, psd_row = signal.welch(
                epoch_row,
                fs=sfreq,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                detrend="constant",
                scaling="density",
            )
            in_band = (freq_row >= psd_fmin) & (freq_row <= psd_fmax)
            freq_row = freq_row[in_band]
            psd_row = psd_row[in_band]
            if freqs is None:
                freqs = freq_row
            psds.append(psd_row)
        psd = np.stack(psds, axis=0)
    if freqs is None:
        return pd.DataFrame()

    records: list[Dict[str, Any]] = []
    for idx, (epoch_id, start_idx, end_idx) in enumerate(valid_meta):
        row: Dict[str, Any] = {}
        bandpowers: Dict[str, float] = {}
        for band_name, (low, high) in bands.items():
            lo = max(float(low), float(psd_fmin))
            hi = min(float(high), float(psd_fmax))
            bandpower = _integrated_bandpower(psd[idx, :], freqs, lo, hi)
            bandpowers[band_name] = bandpower
            row[f"{prefix}_{band_name}"] = bandpower
        row[f"{prefix}_highfreq_power_30_45"] = _integrated_bandpower(
            psd[idx, :],
            freqs,
            max(30.0, float(psd_fmin)),
            min(45.0, float(psd_fmax)),
        )
        row[f"{prefix}_alpha_theta"] = _safe_ratio(
            bandpowers.get("alpha", np.nan),
            bandpowers.get("theta", np.nan),
        )
        row[f"{prefix}_beta_alpha"] = _safe_ratio(
            bandpowers.get("beta", np.nan),
            bandpowers.get("alpha", np.nan),
        )
        hj_mobility, hj_complexity = _compute_hjorth_metrics(epochs_arr[idx, :])
        row[f"{prefix}_hjorth_mobility"] = hj_mobility
        row[f"{prefix}_hjorth_complexity"] = hj_complexity
        entropy_value, entropy_meta = _compute_entropy_feature(
            epochs_arr[idx, :],
            sfreq=float(sfreq),
            order=pe_order,
            delay=pe_delay,
            normalize=pe_normalize,
        )
        row[f"{prefix}_permutation_entropy"] = entropy_value
        # True sample entropy via template matching (independent of permutation entropy).
        se_val = _sample_entropy(epochs_arr[idx, :])
        row[f"{prefix}_sample_entropy"] = se_val if np.isfinite(se_val) else entropy_value
        row[f"{prefix}_entropy_construct"] = str(entropy_meta.get("construct", "energetic_complexity"))
        row[f"{prefix}_entropy_metric"] = str(entropy_meta.get("metric", "permutation_entropy"))
        row[f"{prefix}_entropy_backend"] = str(entropy_meta.get("backend", "numpy"))
        row[f"{prefix}_entropy_degraded_mode"] = bool(entropy_meta.get("degraded_mode", False))
        row[f"{prefix}_entropy_reason"] = entropy_meta.get("reason")
        core_cols = [f"{prefix}_{band}" for band in ("delta", "theta", "alpha", "beta", "gamma")]
        row[f"qc_ok_{prefix}"] = bool(all(np.isfinite(row.get(col, np.nan)) for col in core_cols))
        records.append(row)
    return pd.DataFrame(records)


def _combine_sensor_family_features(frame: pd.DataFrame, families: Sequence[str]) -> pd.DataFrame:
    """Combine MAG/GRAD surfaces after robust standardization.

    Positive band-power features (e.g. meg_delta, meg_alpha) use the
    geometric mean across sensor families rather than robust_z of the raw
    physical values.  MEG data arrives in Tesla (≈1e-13 T), so raw values
    are far below the ``eps=1e-9`` threshold in ``_robust_z``, causing that
    function's fallback path to set scale=1.0 and return un-standardised
    residuals which are then exponentiated to 10^(~0) = 1.0 for every epoch.

    The geometric-mean approach (log-mean then 10^) is scale-invariant: it
    produces the correct result whether the inputs are in T² (~1e-26) or
    µV² (~10), and is consistent with the single-family path which stores
    the raw physical value directly for downstream ``["log10","robust_z","clip"]``
    standardisation.
    """
    combined = pd.DataFrame(index=frame.index)
    for feature_name in _MEG_COMBINED_FEATURES:
        family_cols = [
            f"meg_{family}_{feature_name}"
            for family in families
            if f"meg_{family}_{feature_name}" in frame.columns
        ]
        if not family_cols:
            continue
        is_positive = feature_name in _POSITIVE_COMBINED_FEATURES
        if len(family_cols) == 1:
            # Single family: store raw value; downstream log10→robust_z handles any scale.
            combined[f"meg_{feature_name}"] = pd.to_numeric(
                frame[family_cols[0]],
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            continue
        if is_positive:
            # Geometric mean across families in log10-space.
            # Scale-invariant: works for any physical unit (T², µV², …).
            log_arrays: list[np.ndarray] = []
            for col in family_cols:
                raw = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64, copy=True)
                with np.errstate(divide="ignore", invalid="ignore"):
                    log_arrays.append(np.where(raw > 0, np.log10(raw), np.nan))
            log_mean = np.nanmean(np.stack(log_arrays, axis=1), axis=1)
            combined[f"meg_{feature_name}"] = np.power(10.0, log_mean)
        else:
            standardized = np.stack(
                [
                    _robust_z(pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64, copy=True))
                    for col in family_cols
                ],
                axis=1,
            )
            combined_z = np.nanmedian(standardized, axis=1)
            combined[f"meg_{feature_name}"] = combined_z.astype(np.float32)

    # Combined entropy provenance is deterministic and sensor-family aware.
    entropy_backend_cols = [f"meg_{family}_entropy_backend" for family in families if f"meg_{family}_entropy_backend" in frame.columns]
    entropy_reason_cols = [f"meg_{family}_entropy_reason" for family in families if f"meg_{family}_entropy_reason" in frame.columns]
    degraded_cols = [f"meg_{family}_entropy_degraded_mode" for family in families if f"meg_{family}_entropy_degraded_mode" in frame.columns]
    construct_cols = [f"meg_{family}_entropy_construct" for family in families if f"meg_{family}_entropy_construct" in frame.columns]
    metric_cols = [f"meg_{family}_entropy_metric" for family in families if f"meg_{family}_entropy_metric" in frame.columns]
    if construct_cols:
        combined["meg_entropy_construct"] = frame[construct_cols[0]].astype(object)
    if metric_cols:
        combined["meg_entropy_metric"] = frame[metric_cols[0]].astype(object)
    if entropy_backend_cols:
        combined["meg_entropy_backend"] = frame[entropy_backend_cols[0]].astype(object)
    if degraded_cols:
        degraded = np.zeros(len(frame), dtype=bool)
        for col in degraded_cols:
            degraded |= frame[col].astype(bool).to_numpy(copy=False)
        combined["meg_entropy_degraded_mode"] = degraded
    if entropy_reason_cols:
        reasons = np.full(len(frame), "", dtype=object)
        for col in entropy_reason_cols:
            vals = frame[col].fillna("").astype(str).to_numpy(dtype=object, copy=False)
            use = (reasons == "") & (vals != "")
            if np.any(use):
                reasons[use] = vals[use]
        combined["meg_entropy_reason"] = reasons
    if "meg_permutation_entropy" in combined.columns:
        combined["qc_ok_meg"] = np.isfinite(
            combined[[col for col in combined.columns if col in {
                "meg_delta",
                "meg_theta",
                "meg_alpha",
                "meg_beta",
                "meg_gamma",
            }]].to_numpy(dtype=np.float64)
        ).all(axis=1)
    return combined


def _compute_meg_group_features(
    signals: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    sfreq: float,
    meta: Sequence[tuple[int, int, int]],
    dataset_id: Optional[str],
) -> pd.DataFrame:
    """Compute config-defined helmet-group features from one MEG family.

    These are sensor-topographic aggregates, not source-localized cortical
    regions.  Groups intentionally use their own ``meg_ensembles`` config so
    that EEG 10-20 groups under ``robustness.ensembles`` cannot accidentally
    be applied to MEG channel names.
    """
    robustness_cfg = config.get("robustness", {}) if isinstance(config, Mapping) else {}
    groups_cfg = robustness_cfg.get("meg_ensembles", {}) if isinstance(robustness_cfg, Mapping) else {}
    if not isinstance(groups_cfg, Mapping) or not bool(groups_cfg.get("enabled", False)):
        return pd.DataFrame()

    family_name = str(groups_cfg.get("sensor_family", "mag")).strip().lower()
    if family_name not in {"mag", "grad", "meg"}:
        logger.warning("Unsupported MEG ensemble sensor_family=%r; expected mag, grad, or meg", family_name)
        return pd.DataFrame()

    signal_map = signals.get("signals", {}) if isinstance(signals, Mapping) else {}
    channels_map = signals.get("channels", {}) if isinstance(signals, Mapping) else {}
    data_key = "meg" if family_name == "meg" else f"meg_{family_name}"
    group_data = signal_map.get(data_key)
    group_channels = channels_map.get(data_key, [])
    if not isinstance(group_data, np.ndarray) or group_data.ndim != 2 or not group_data.size:
        logger.warning("MEG ensembles requested %s but no matching sensor data were available", data_key)
        return pd.DataFrame()
    if not isinstance(group_channels, Sequence) or isinstance(group_channels, (str, bytes)):
        return pd.DataFrame()

    min_channels = max(1, int(groups_cfg.get("min_channels", 3) or 3))
    group_defs = ensembles.realize_ensemble_groups(groups_cfg, dataset_id, [str(name) for name in group_channels])
    if not group_defs:
        return pd.DataFrame()

    features_cfg = config.get("features", {}) if isinstance(config, Mapping) else {}
    group_frames: list[pd.DataFrame] = []
    for group in group_defs:
        if len(group.indices) < min_channels:
            logger.warning(
                "Skipping MEG helmet group %s: %d channels resolved, minimum is %d",
                group.safe_name,
                len(group.indices),
                min_channels,
            )
            continue
        group_frame = _compute_sensor_family_features(
            group_data[np.asarray(group.indices, dtype=int), :],
            sfreq,
            meta,
            features_cfg=features_cfg if isinstance(features_cfg, Mapping) else {},
            prefix="meg",
        )
        if group_frame.empty:
            continue
        group_frame = group_frame.rename(columns={col: f"{col}__g_{group.safe_name}" for col in group_frame.columns})
        group_frames.append(group_frame)
    return pd.concat(group_frames, axis=1) if group_frames else pd.DataFrame()


def compute_meg_features(signals: Mapping[str, Any], config: Mapping[str, Any]) -> pd.DataFrame:
    """Compute minimal MEG shadow features with MAG/GRAD diagnostics."""
    family_arrays = _resolve_meg_family_arrays(signals)
    if not family_arrays:
        return pd.DataFrame()

    sfreq = float(signals.get("sfreq", 250.0) or 250.0)
    dataset_id = signals.get("dataset_id")
    features_cfg = config.get("features", {}) if isinstance(config, Mapping) else {}
    if not isinstance(features_cfg, Mapping):
        features_cfg = {}

    sample_counts = [arr.shape[1] for arr in family_arrays.values() if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.size]
    if not sample_counts:
        return pd.DataFrame()
    meta = _resolve_epoch_meta(min(sample_counts), sfreq, config, dataset_id)
    if not meta:
        return pd.DataFrame()

    meta_df = pd.DataFrame(
        {
            "epoch_id": [epoch_id for epoch_id, _, _ in meta],
            "t_start": [start / sfreq for _, start, _ in meta],
            "t_end": [end / sfreq for _, _, end in meta],
        }
    )
    out = meta_df.copy()
    used_families: list[str] = []
    for family_name in ("mag", "grad"):
        family_data = family_arrays.get(family_name)
        if not isinstance(family_data, np.ndarray) or family_data.ndim != 2 or family_data.size == 0:
            continue
        family_df = _compute_sensor_family_features(
            family_data,
            sfreq,
            meta,
            features_cfg=features_cfg,
            prefix=f"meg_{family_name}",
        )
        if family_df.empty:
            continue
        used_families.append(family_name)
        out = pd.concat([out, family_df], axis=1)
    if not used_families:
        return pd.DataFrame()

    combined_df = _combine_sensor_family_features(out, used_families)
    if not combined_df.empty:
        out = pd.concat([out, combined_df], axis=1)
    group_df = _compute_meg_group_features(
        signals,
        config,
        sfreq=sfreq,
        meta=meta,
        dataset_id=str(dataset_id) if dataset_id is not None else None,
    )
    if not group_df.empty:
        out = pd.concat([out, group_df], axis=1)
    return out
