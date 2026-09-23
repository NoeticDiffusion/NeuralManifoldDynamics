"""Per-epoch local fMRI metrics computed from preprocessed session slices."""

from __future__ import annotations

import itertools
import logging
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from scipy.signal import welch

from ..reproducibility import resolve_component_seed
from . import fmri_connectivity

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    nx = None

logger = logging.getLogger(__name__)

# NMD-fMRI-v2.5 Stage 2 (sciencelead/003.md + 0004.md) contract-repair note:
# these three column names have historically been assigned the exact same
# value as another feature in features/fmri.py (fmri_region_var_mean and
# fmri_entropy_global both literally copy fmri_variance_global; fmri_lf_power
# literally copies fmri_signal_power). They are NOT independent measurements
# despite the names. Kept for backward compatibility with anything reading
# these column names, but must never be treated as real entropy / regional
# variance / band-limited low-frequency-power features in any mapping,
# registry classification, or new analysis. See results/ds007216/
# v25_feature_registry.csv (family="rejected_alias") and diary #168.
LEGACY_ALIAS_FEATURES: Dict[str, str] = {
    "fmri_region_var_mean": "fmri_variance_global",
    "fmri_entropy_global": "fmri_variance_global",
    "fmri_lf_power": "fmri_signal_power",
}

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
    # NMD-fMRI-v2.5 Stage 2 candidate feature families (sciencelead/003.md +
    # 0004.md). All default False: opt-in only, so no existing dataset config
    # changes output schema/cost unless it explicitly enables them.
    compute_spectral_entropy = bool(cfg.get("compute_spectral_entropy", False))
    compute_permutation_entropy_fmri = bool(cfg.get("compute_permutation_entropy_fmri", False))
    compute_sample_entropy_fmri = bool(cfg.get("compute_sample_entropy_fmri", False))
    compute_network_fc = bool(cfg.get("compute_network_fc", False))
    compute_ar2 = bool(cfg.get("compute_ar2", False))
    compute_temporal_smoothness = bool(cfg.get("compute_temporal_smoothness", False))
    compute_hurst = bool(cfg.get("compute_hurst", False))
    compute_alff = bool(cfg.get("compute_alff", False))
    compute_falff = bool(cfg.get("compute_falff", False))
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
            metrics["fmri_FC_mean"] = fc_triu_mean(fc)
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
        metrics["fmri_dFC_variance"] = _compute_dfc_variance(
            epoch_filtered, sfreq, min_timepoints_fc, fc_epoch=fc
        )

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

    # 10) True entropy/complexity family (v2.5 Stage 2). Computed on the
    # bandpassed epoch signal per ROI, then averaged -- NOT a copy of
    # variance/power (contrast with the legacy fmri_entropy_global alias).
    if compute_spectral_entropy:
        metrics["fmri_spectral_entropy"] = _compute_spectral_entropy(epoch_filtered, sfreq)
    if compute_permutation_entropy_fmri:
        metrics["fmri_permutation_entropy"] = _compute_permutation_entropy(epoch_filtered)
    if compute_sample_entropy_fmri:
        # Known-risky at short window lengths (~20-45 samples); this project
        # intentionally excludes full DFA/PLE nested-window analysis (see
        # module docstring) -- sample entropy here is a single-window,
        # explicitly opt-in estimate to be screened by the v2.5 feature
        # registry's finite-fraction/variance checks, not assumed stable.
        metrics["fmri_sample_entropy"] = _compute_sample_entropy(epoch_filtered)

    # 11) Connectivity/segregation family: within/between Yeo-network FC,
    # system segregation index (Chan et al. 2014 formula), and
    # Guimera-Amaral participation coefficient using the fixed Yeo-network
    # partition (not a data-driven community detection, for stability at
    # short window lengths).
    if compute_network_fc:
        metrics.update(_compute_network_fc_metrics(fc, roi_names))

    # 12) Temporal-persistence family: lag-2 autocorrelation and a bounded
    # Hjorth-mobility-derived smoothness score (complements lag-1 AR1 below).
    if compute_ar2:
        metrics["fmri_ar2_coefficient"] = _compute_ar2_mean(epoch_filtered)
    if compute_temporal_smoothness:
        metrics["fmri_temporal_smoothness"] = _compute_temporal_smoothness(epoch_filtered)
    if compute_hurst:
        # Coarse single-window R/S proxy, not a multi-scale DFA/PLE estimate
        # (see module docstring); expected unstable at this window length --
        # kept opt-in and screened via the registry rather than assumed valid.
        metrics["fmri_hurst_exponent"] = _compute_hurst_mean(epoch_filtered)

    # 13) ALFF/fALFF (amplitude/low-frequency family). Computed on RAW
    # (unfiltered) epoch data so fmri_connectivity.py's own internal bandpass
    # and full-band denominator are meaningful; treated as motion-risk per
    # the v2.5 Stage 1 audit (raw amplitude/power features correlate ~0.58
    # with FD in ds007216) until independently screened.
    if compute_alff or compute_falff:
        alff_features, _ = fmri_connectivity.compute_static_connectivity_features(
            epoch_raw,
            sfreq,
            {"FC_pearson": False, "ALFF": compute_alff, "fALFF": compute_falff},
        )
        if compute_alff:
            metrics["fmri_ALFF"] = alff_features.get("fmri_ALFF_mean", float("nan"))
        if compute_falff:
            metrics["fmri_fALFF"] = alff_features.get("fmri_fALFF_mean", float("nan"))

    return metrics


def fc_triu_mean(fc: np.ndarray) -> float:
    """Mean of the finite strict-upper-triangle of a correlation matrix."""
    arr = np.asarray(fc, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 2:
        return float("nan")
    triu = arr[np.triu_indices_from(arr, k=1)]
    valid = triu[np.isfinite(triu)]
    return float(np.nanmean(valid)) if valid.size else float("nan")


def fc_mean_for_indices(fc: np.ndarray, indices: Sequence[int]) -> float:
    """``fmri_FC_mean`` of a ROI subset as a slice of a session FC matrix."""
    idx = np.asarray(indices, dtype=int)
    if idx.size < 2:
        return float("nan")
    arr = np.asarray(fc, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return float("nan")
    if np.any(idx < 0) or np.any(idx >= arr.shape[0]):
        return float("nan")
    return fc_triu_mean(arr[np.ix_(idx, idx)])


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


def _compute_dfc_variance(
    epoch_filtered: np.ndarray,
    sfreq: float,
    min_timepoints_fc: int,
    fc_epoch: np.ndarray | None = None,
) -> float:
    """dFC variance from inner subwindows; fallback reuses the epoch FC matrix."""
    n_time = int(epoch_filtered.shape[1])
    win_len = max(int(round(15.0 * sfreq)), min_timepoints_fc)
    step = max(int(round(5.0 * sfreq)), 1)
    fc_means: list[float] = []
    start = 0
    while start + win_len <= n_time:
        seg = epoch_filtered[:, start : start + win_len]
        if seg.shape[1] >= max(min_timepoints_fc, 2) and seg.shape[0] >= 2:
            mean = fc_triu_mean(np.corrcoef(seg))
            if np.isfinite(mean):
                fc_means.append(mean)
        start += step
    if len(fc_means) >= 2:
        return float(np.nanvar(np.asarray(fc_means, dtype=float)))

    # Fallback proxy when not enough subwindows: dispersion of static FC edges.
    fc = fc_epoch
    if fc is None and epoch_filtered.shape[1] >= max(min_timepoints_fc, 2) and epoch_filtered.shape[0] >= 2:
        fc = np.corrcoef(epoch_filtered)
    if fc is None:
        return float("nan")
    arr = np.asarray(fc, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 2:
        return float("nan")
    triu = arr[np.triu_indices_from(arr, k=1)]
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


def _mean_lag_corr(epoch_filtered: np.ndarray, lag: int) -> float:
    """Mean per-ROI lag-``lag`` Pearson correlation (AR1/AR2).

    Same NaN policy as the historical per-ROI ``corrcoef`` loop: skip a row
    when ``nanstd`` of either lag slice is ``<= 0``; correlation itself
    matches ``np.corrcoef`` (ordinary mean, ddof=1).
    """
    arr = np.asarray(epoch_filtered, dtype=float)
    if arr.ndim != 2 or lag < 1 or arr.shape[1] < lag + 2:
        return float("nan")
    x0 = arr[:, :-lag]
    x1 = arr[:, lag:]
    with np.errstate(invalid="ignore", divide="ignore"):
        std0 = np.nanstd(x0, axis=1)
        std1 = np.nanstd(x1, axis=1)
    ok = (std0 > 0) & (std1 > 0)
    n = int(x0.shape[1])
    if n < 2:
        return float("nan")
    a = x0 - np.mean(x0, axis=1, keepdims=True)
    b = x1 - np.mean(x1, axis=1, keepdims=True)
    denom = float(n - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = np.sum(a * b, axis=1) / denom
        sa = np.sqrt(np.sum(a * a, axis=1) / denom)
        sb = np.sqrt(np.sum(b * b, axis=1) / denom)
        corr = cov / (sa * sb)
    corr = np.where(ok, corr, np.nan)
    finite = corr[np.isfinite(corr)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _compute_ar1_mean(epoch_filtered: np.ndarray) -> float:
    """Mean lag-1 autocorrelation across ROIs."""
    return _mean_lag_corr(epoch_filtered, lag=1)


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


def _compute_spectral_entropy(epoch_filtered: np.ndarray, sfreq: float) -> float:
    """Normalized Shannon entropy of the Welch PSD, averaged over ROIs.

    In [0, 1] (0 = single spectral line, 1 = flat/white spectrum). Genuinely
    independent of amplitude/variance (PSD is normalized to a probability
    distribution before computing entropy) -- contrast with the legacy
    fmri_entropy_global alias, which is just a copy of variance."""
    if sfreq <= 0 or epoch_filtered.shape[1] < 8:
        return float("nan")
    try:
        freqs, psd = welch(epoch_filtered, fs=sfreq, nperseg=min(epoch_filtered.shape[1], 64), axis=1)
    except Exception:
        return float("nan")
    if freqs.size < 2 or psd.size == 0:
        return float("nan")
    psd_sum = np.nansum(psd, axis=1)
    valid_rows = np.isfinite(psd_sum) & (psd_sum > 0)
    if not np.any(valid_rows):
        return float("nan")
    p = psd[valid_rows] / psd_sum[valid_rows, None]
    p = np.clip(p, 1e-12, 1.0)
    ent = -np.sum(p * np.log(p), axis=1) / np.log(p.shape[1])
    finite = ent[np.isfinite(ent)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _permutation_entropy_1d(x: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """Bandt-Pompe permutation entropy, normalized to [0, 1] by log(order!)."""
    n = len(x)
    if n < order * delay + 1 or order < 2:
        return float("nan")
    permutations = list(itertools.permutations(range(order)))
    perm_to_idx = {p: i for i, p in enumerate(permutations)}
    counts = np.zeros(len(permutations), dtype=float)
    for i in range(n - (order - 1) * delay):
        window = x[i : i + order * delay : delay]
        if len(window) < order or not np.all(np.isfinite(window)):
            continue
        rank = tuple(np.argsort(window))
        counts[perm_to_idx[rank]] += 1
    total = counts.sum()
    if total <= 0:
        return float("nan")
    p = counts[counts > 0] / total
    ent = -np.sum(p * np.log(p))
    return float(ent / np.log(len(permutations)))


def _compute_permutation_entropy(epoch_filtered: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """Mean permutation entropy across ROIs. Order kept low (3) given the
    short (~20-45 sample) epoch length used throughout this pipeline."""
    vals: list[float] = []
    for i in range(epoch_filtered.shape[0]):
        v = _permutation_entropy_1d(np.asarray(epoch_filtered[i], dtype=float), order=order, delay=delay)
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def _sample_entropy_1d(x: np.ndarray, m: int = 2, r_frac: float = 0.2) -> float:
    """Classic SampEn (Richman & Moorman, 2000). Known to be biased/unstable
    for short series (N << 100); kept simple/explicit here rather than
    approximated, so instability shows up directly in the v2.5 registry's
    finite-fraction/variance checks."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < m + 2:
        return float("nan")
    std = np.nanstd(x)
    if not np.isfinite(std) or std <= 0:
        return float("nan")
    r = r_frac * std

    def _count_matches(m_len: int) -> int:
        if n - m_len + 1 < 2:
            return 0
        templates = np.array([x[i : i + m_len] for i in range(n - m_len + 1)])
        count = 0
        for i in range(len(templates) - 1):
            dist = np.max(np.abs(templates[i + 1 :] - templates[i]), axis=1)
            count += int(np.sum(dist <= r))
        return count

    b_count = _count_matches(m)
    a_count = _count_matches(m + 1)
    if b_count == 0 or a_count == 0:
        return float("nan")
    return float(-np.log(a_count / b_count))


def _compute_sample_entropy(epoch_filtered: np.ndarray, m: int = 2, r_frac: float = 0.2) -> float:
    vals: list[float] = []
    for i in range(epoch_filtered.shape[0]):
        v = _sample_entropy_1d(epoch_filtered[i], m=m, r_frac=r_frac)
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def _compute_network_fc_metrics(fc: np.ndarray | None, roi_names: Sequence[str] | None) -> Dict[str, float]:
    """Within/between Yeo-network FC, system segregation index (Chan et al.
    2014: (within - between) / within), and Guimera-Amaral participation
    coefficient on the fixed Yeo-network partition (not data-driven
    community detection, for stability at short window lengths)."""
    out = {
        "fmri_within_network_fc": float("nan"),
        "fmri_between_network_fc": float("nan"),
        "fmri_network_segregation_index": float("nan"),
        "fmri_participation_coefficient": float("nan"),
    }
    if fc is None or roi_names is None or len(roi_names) != fc.shape[0]:
        return out

    groups: Dict[str, list[int]] = {}
    for idx, name in enumerate(roi_names):
        label = _infer_network_label(str(name))
        groups.setdefault(label, []).append(idx)
    group_names = [g for g, idxs in groups.items() if len(idxs) >= 2]
    if len(group_names) >= 2:
        within_vals = [v for g in group_names if np.isfinite(v := _mean_intra_fc(fc, groups[g]))]
        between_vals: list[float] = []
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                sub = fc[np.ix_(groups[group_names[i]], groups[group_names[j]])]
                valid = sub[np.isfinite(sub)]
                if valid.size:
                    between_vals.append(float(np.nanmean(valid)))

        within_mean = float(np.nanmean(within_vals)) if within_vals else float("nan")
        between_mean = float(np.nanmean(between_vals)) if between_vals else float("nan")
        out["fmri_within_network_fc"] = within_mean
        out["fmri_between_network_fc"] = between_mean
        if np.isfinite(within_mean) and np.isfinite(between_mean) and abs(within_mean) > 1e-9:
            out["fmri_network_segregation_index"] = (within_mean - between_mean) / within_mean

    out["fmri_participation_coefficient"] = _participation_coefficient(fc, groups)
    return out


def _participation_coefficient(fc: np.ndarray, groups: Mapping[str, Sequence[int]]) -> float:
    """Guimera-Amaral PC on the named partition, including singleton networks."""
    n = int(fc.shape[0])
    idx_to_group = {int(idx): g for g, idxs in groups.items() for idx in idxs}
    absfc = np.abs(np.nan_to_num(fc, nan=0.0))
    np.fill_diagonal(absfc, 0.0)
    node_groups = [idx_to_group.get(j) for j in range(n)]
    uniq = sorted({g for g in node_groups if g is not None})
    if not uniq:
        return float("nan")
    g_index = {g: k for k, g in enumerate(uniq)}
    indicator = np.zeros((n, len(uniq)), dtype=float)
    for j, g in enumerate(node_groups):
        if g is not None:
            indicator[j, g_index[g]] = 1.0
    group_sums = absfc @ indicator
    totals = absfc.sum(axis=1)
    ok = totals > 1e-12
    if not np.any(ok):
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = group_sums[ok] / totals[ok, None]
    pc = 1.0 - np.sum(frac * frac, axis=1)
    finite = pc[np.isfinite(pc)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _compute_ar2_mean(epoch_filtered: np.ndarray) -> float:
    """Mean lag-2 autocorrelation across ROIs."""
    return _mean_lag_corr(epoch_filtered, lag=2)


def _compute_temporal_smoothness(epoch_filtered: np.ndarray) -> float:
    """Internal helper: Hjorth-mobility-derived smoothness, bounded in (0, 1].
    Higher = smoother/more temporally persistent; complements AR1/AR2."""
    vals: list[float] = []
    for i in range(epoch_filtered.shape[0]):
        x = np.asarray(epoch_filtered[i], dtype=float)
        if x.size < 3:
            continue
        var_x = np.nanvar(x)
        if not np.isfinite(var_x) or var_x <= 0:
            continue
        var_dx = np.nanvar(np.diff(x))
        mobility = np.sqrt(var_dx / var_x)
        if np.isfinite(mobility) and mobility >= 0:
            vals.append(1.0 / (1.0 + mobility))
    return float(np.mean(vals)) if vals else float("nan")


def _hurst_rs_1d(x: np.ndarray) -> float:
    """Single-window rescaled-range (R/S) Hurst proxy. NOT a multi-scale
    DFA/PLE estimate (this module intentionally excludes those -- see module
    docstring); a coarse, single-point estimate expected to be noisy/unstable
    at the ~20-45 sample epoch lengths used throughout this pipeline."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 8:
        return float("nan")
    mean_x = np.nanmean(x)
    if not np.isfinite(mean_x):
        return float("nan")
    y = np.cumsum(x - mean_x)
    r = float(np.max(y) - np.min(y))
    s = float(np.nanstd(x))
    if s <= 0 or r <= 0:
        return float("nan")
    return float(np.log(r / s) / np.log(n))


def _compute_hurst_mean(epoch_filtered: np.ndarray) -> float:
    vals: list[float] = []
    for i in range(epoch_filtered.shape[0]):
        v = _hurst_rs_1d(epoch_filtered[i])
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


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

