"""Per-epoch local fMRI metrics computed from preprocessed session slices."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from scipy.signal import welch

from ..reproducibility import resolve_component_seed

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    nx = None

logger = logging.getLogger(__name__)

_NETWORK_ALIASES: Dict[str, str] = {
    "VIS": "VIS",
    "VISUAL": "VIS",
    "SOMMOT": "SMN",
    "SOMATOMOTOR": "SMN",
    "DORSATTN": "DAN",
    "VENTATTN": "VAN",
    "SALVENTATTN": "SAL",
    "SAL": "SAL",
    "LIMBIC": "LIM",
    "CONT": "FPN",
    "CONTROL": "FPN",
    "DEFAULT": "DMN",
    "DMN": "DMN",
    "FP": "FPN",
}


def compute_local_metrics(
    epoch_raw: np.ndarray,
    epoch_filtered: np.ndarray,
    epoch_phase: np.ndarray | None,
    sfreq: float,
    config: Mapping[str, Any] | None,
    roi_names: Sequence[str] | None = None,
) -> Dict[str, float]:
    """Compute robust local metrics for one epoch.

    This function intentionally excludes DFA/PLE and any nested-window logic.
    """
    if epoch_raw.ndim != 2 or epoch_filtered.ndim != 2:
        raise ValueError("epoch_raw and epoch_filtered must be 2-D")
    if epoch_raw.shape != epoch_filtered.shape:
        raise ValueError("epoch_raw and epoch_filtered must have identical shape")

    cfg = config if isinstance(config, Mapping) else {}
    metrics: Dict[str, float] = {}
    min_timepoints_fc = int(cfg.get("min_timepoints_fc", 10))

    compute_variance = bool(cfg.get("compute_variance", True))
    compute_power = bool(cfg.get("compute_power", True))
    compute_fc = bool(cfg.get("compute_fc", True))
    compute_kuramoto = bool(cfg.get("compute_kuramoto", True))
    compute_modularity = bool(cfg.get("compute_modularity", True))
    compute_ar1 = bool(cfg.get("compute_ar1", True))
    compute_slow_band_ratio = bool(cfg.get("compute_slow_band_ratio", True))
    compute_gradient_proxy = bool(cfg.get("compute_gradient_proxy", True))
    compute_dfc_variance = bool(cfg.get("compute_dfc_variance", True))
    compute_dvars = bool(cfg.get("compute_dvars", True))
    modularity_seed, _ = resolve_component_seed(
        cfg,
        fallback_seed=cfg.get("modularity_seed", cfg.get("seed")),
        fallback_source="features.metrics.modularity_seed",
    )

    # 0) Motion surrogate from raw, unfiltered BOLD.
    if compute_dvars:
        metrics["fmri_dvars"] = _compute_epoch_dvars(epoch_raw)

    # 1) Variance proxy over filtered data.
    if compute_variance:
        var_regions = np.nanvar(epoch_filtered, axis=1)
        variance_global = float(np.nanmean(var_regions)) if np.isfinite(var_regions).any() else float("nan")
        metrics["fmri_variance_global"] = variance_global

    # 2) Signal power over filtered data.
    if compute_power:
        metrics["fmri_signal_power"] = float(np.nanmean(epoch_filtered**2)) if epoch_filtered.size else float("nan")

    # 3) Window-level FC summary from filtered data.
    fc: np.ndarray | None = None
    if epoch_filtered.shape[0] >= 2 and epoch_filtered.shape[1] >= max(min_timepoints_fc, 2):
        fc = np.corrcoef(epoch_filtered)
        triu = fc[np.triu_indices_from(fc, k=1)]
        valid_triu = triu[np.isfinite(triu)]
        if compute_fc and valid_triu.size > 0:
            metrics["fmri_FC_mean"] = float(np.nanmean(valid_triu))
            metrics["fmri_FC_std"] = float(np.nanstd(valid_triu))

    # 4) Louvain modularity on non-negative FC adjacency.
    if compute_modularity:
        metrics["fmri_modularity"] = float("nan")
        if fc is not None and np.any(np.isfinite(fc)) and nx is not None:
            try:
                adj = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
                np.fill_diagonal(adj, 0.0)
                adj = np.maximum(adj, 0.0)
                if np.any(adj > 0):
                    G = nx.from_numpy_array(adj)
                    communities = nx.algorithms.community.louvain_communities(
                        G,
                        weight="weight",
                        seed=int(modularity_seed),
                    )
                    metrics["fmri_modularity"] = float(nx.algorithms.community.modularity(G, communities, weight="weight"))
            except Exception:
                logger.debug("Louvain modularity failed for epoch", exc_info=True)

    # 5) dFC variance from internal subwindows (or FC-std proxy when insufficient windows).
    if compute_dfc_variance:
        metrics["fmri_dFC_variance"] = _compute_dfc_variance(epoch_filtered, sfreq, min_timepoints_fc)

    # 6) Kuramoto order parameter from precomputed phase.
    if compute_kuramoto and epoch_phase is not None and epoch_phase.shape == epoch_filtered.shape:
        z = np.exp(1j * epoch_phase)
        gamma_t = np.abs(np.mean(z, axis=0))
        metrics["fmri_kuramoto_global"] = float(np.nanmean(gamma_t))
        metrics["fmri_kuramoto_global_std"] = float(np.nanstd(gamma_t))

    # 7) Slow-4 / Slow-5 ratio from filtered epoch.
    if compute_slow_band_ratio:
        metrics["fmri_slow4_slow5_ratio"] = _compute_slow4_slow5_ratio(epoch_filtered, sfreq)

    # 8) AR(1) coefficient (mean over ROIs).
    if compute_ar1:
        metrics["fmri_ar1_coefficient"] = _compute_ar1_mean(epoch_filtered)

    # 9) Gradient proxy: unimodal intra-network FC / transmodal intra-network FC.
    if compute_gradient_proxy:
        metrics["fmri_gradient_ratio"] = _compute_gradient_ratio(fc, roi_names)

    return metrics


def _compute_epoch_dvars(epoch_raw: np.ndarray) -> float:
    """Compute mean epoch DVARS (RMS temporal derivative over ROIs)."""
    if epoch_raw.ndim != 2 or epoch_raw.shape[1] < 2:
        return float("nan")
    diffs = np.diff(np.asarray(epoch_raw, dtype=float), axis=1)
    dvars_t = np.sqrt(np.mean(diffs**2, axis=0))
    finite = dvars_t[np.isfinite(dvars_t)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _compute_dfc_variance(epoch_filtered: np.ndarray, sfreq: float, min_timepoints_fc: int) -> float:
    """Internal helper: compute dfc variance."""
    n_time = int(epoch_filtered.shape[1])
    win_len = max(int(round(15.0 * sfreq)), min_timepoints_fc)
    step = max(int(round(5.0 * sfreq)), 1)
    fc_means: list[float] = []
    start = 0
    while start + win_len <= n_time:
        seg = epoch_filtered[:, start : start + win_len]
        if seg.shape[1] >= max(min_timepoints_fc, 2) and seg.shape[0] >= 2:
            fc = np.corrcoef(seg)
            triu = fc[np.triu_indices_from(fc, k=1)]
            valid = triu[np.isfinite(triu)]
            if valid.size:
                fc_means.append(float(np.nanmean(valid)))
        start += step
    if len(fc_means) >= 2:
        return float(np.nanvar(np.asarray(fc_means, dtype=float)))

    # Fallback proxy when not enough subwindows: dispersion of static FC edges.
    if epoch_filtered.shape[1] >= max(min_timepoints_fc, 2) and epoch_filtered.shape[0] >= 2:
        fc = np.corrcoef(epoch_filtered)
        triu = fc[np.triu_indices_from(fc, k=1)]
        valid = triu[np.isfinite(triu)]
        if valid.size:
            return float(np.nanvar(valid))
    return float("nan")


def _compute_slow4_slow5_ratio(epoch_filtered: np.ndarray, sfreq: float) -> float:
    """Internal helper: compute slow4 slow5 ratio."""
    if sfreq <= 0 or epoch_filtered.shape[1] < 4:
        return float("nan")
    try:
        freqs, psd = welch(epoch_filtered, fs=sfreq, nperseg=min(epoch_filtered.shape[1], 128), axis=1)
    except Exception:
        return float("nan")
    if freqs.size == 0 or psd.size == 0:
        return float("nan")
    slow5 = np.nansum(psd[:, (freqs >= 0.01) & (freqs < 0.027)], axis=1)
    slow4 = np.nansum(psd[:, (freqs >= 0.027) & (freqs <= 0.073)], axis=1)
    ratio = slow4 / (slow5 + 1e-12)
    finite = ratio[np.isfinite(ratio)]
    return float(np.nanmean(finite)) if finite.size else float("nan")


def _compute_ar1_mean(epoch_filtered: np.ndarray) -> float:
    """Internal helper: compute ar1 mean."""
    vals: list[float] = []
    for i in range(epoch_filtered.shape[0]):
        x = np.asarray(epoch_filtered[i], dtype=float)
        if x.size < 3:
            continue
        x0 = x[:-1]
        x1 = x[1:]
        if np.nanstd(x0) <= 0 or np.nanstd(x1) <= 0:
            continue
        corr = np.corrcoef(x0, x1)[0, 1]
        if np.isfinite(corr):
            vals.append(float(corr))
    if not vals:
        return float("nan")
    return float(np.nanmean(np.asarray(vals, dtype=float)))


def _infer_network_label(region_name: str) -> str:
    """Internal helper: infer network label."""
    if not region_name:
        return "ROI"
    tokens = [tok for tok in str(region_name).split("_") if tok]
    for tok in tokens:
        key = tok.upper()
        if key in _NETWORK_ALIASES:
            return _NETWORK_ALIASES[key]
    if len(tokens) >= 3:
        return tokens[2].upper()
    return tokens[0].upper()


def _mean_intra_fc(fc: np.ndarray, idxs: list[int]) -> float:
    """Internal helper: mean intra fc."""
    if len(idxs) < 2:
        return float("nan")
    sub = fc[np.ix_(idxs, idxs)]
    triu = sub[np.triu_indices_from(sub, k=1)]
    valid = triu[np.isfinite(triu)]
    return float(np.nanmean(valid)) if valid.size else float("nan")


def _compute_gradient_ratio(fc: np.ndarray | None, roi_names: Sequence[str] | None) -> float:
    """Internal helper: compute gradient ratio."""
    if fc is None or roi_names is None:
        return float("nan")
    if len(roi_names) != fc.shape[0]:
        return float("nan")

    groups: Dict[str, list[int]] = {}
    for idx, name in enumerate(roi_names):
        label = _infer_network_label(str(name))
        groups.setdefault(label, []).append(idx)

    uni_vals = [_mean_intra_fc(fc, groups.get("VIS", [])), _mean_intra_fc(fc, groups.get("SMN", []))]
    trans_vals = [_mean_intra_fc(fc, groups.get("DMN", [])), _mean_intra_fc(fc, groups.get("FPN", []))]
    uni = np.asarray([v for v in uni_vals if np.isfinite(v)], dtype=float)
    trans = np.asarray([v for v in trans_vals if np.isfinite(v)], dtype=float)
    if uni.size == 0 or trans.size == 0:
        return float("nan")
    denom = float(np.nanmean(trans))
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return float("nan")
    return float(np.nanmean(uni) / denom)

