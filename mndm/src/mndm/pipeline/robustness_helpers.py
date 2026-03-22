"""
robustness_helpers.py
Robustness, reliability, and QC summary utilities for MNPS."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from .. import ensembles, projection, robustness

logger = logging.getLogger(__name__)

def _distributional_descriptives(
    values: np.ndarray,
    names: list[str],
) -> Dict[str, Dict[str, Any]]:
    """Compute neutral distributional descriptives for each column.

    This is intentionally analysis-agnostic: no hypothesis tests, no group comparisons,
    just basic functionals of the per-epoch distribution.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(names):
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for j, name in enumerate(names):
        col = arr[:, j]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            out[str(name)] = {
                "n": 0,
                "nan_frac": 1.0,
                "mean": float("nan"),
                "median": float("nan"),
                "std": float("nan"),
                "iqr": float("nan"),
                "delta_mean_median": float("nan"),
            }
            continue

        mean = float(np.mean(finite))
        median = float(np.median(finite))
        std = float(np.std(finite, ddof=0))
        # MAD: median(|x - median(x)|); robust scale. Also export scaled MAD_sigma
        # for approximate comparability to std under normality.
        try:
            mad = float(np.median(np.abs(finite - median)))
        except Exception:
            mad = float("nan")
        mad_sigma = float(1.4826 * mad) if np.isfinite(mad) else float("nan")
        # Shape descriptors use central moments (outlier-sensitive by design).
        skewness = float("nan")
        kurtosis_excess = float("nan")
        if np.isfinite(std) and std > 0:
            try:
                z = (finite - mean) / std
                skewness = float(np.mean(z ** 3))
                kurtosis_excess = float(np.mean(z ** 4) - 3.0)
            except Exception:
                skewness = float("nan")
                kurtosis_excess = float("nan")
        try:
            q25 = float(np.percentile(finite, 25))
            q75 = float(np.percentile(finite, 75))
            iqr = float(q75 - q25)
        except Exception:
            iqr = float("nan")
        out[str(name)] = {
            "n": int(finite.size),
            "nan_frac": float(1.0 - (finite.size / col.size if col.size else 1.0)),
            "mean": mean,
            "median": median,
            "std": std,
            "iqr": iqr,
            "mad": mad,
            "mad_sigma": mad_sigma,
            "skewness": skewness,
            "kurtosis_excess": kurtosis_excess,
            "delta_mean_median": float(mean - median),
        }
    return out


def compute_dist_summary(
    x: np.ndarray,
    coords_9d: Optional[np.ndarray],
    coords_9d_names: list[str],
) -> Dict[str, Any]:
    """Compute distributional summaries for MNPS axes and optional v2 subcoords."""
    result: Dict[str, Any] = {}
    try:
        result["axes"] = _distributional_descriptives(np.asarray(x, dtype=float), ["m", "d", "e"])
    except Exception:
        logger.exception("Failed to compute distributional descriptives for MNPS axes")
        result["axes"] = {}

    if coords_9d is not None and coords_9d_names:
        try:
            result["subcoords"] = _distributional_descriptives(
                np.asarray(coords_9d, dtype=float),
                list(coords_9d_names),
            )
        except Exception:
            logger.exception("Failed to compute distributional descriptives for Stratified MNPS subcoords")
            result["subcoords"] = {}
    return result


def compute_tau_summary(
    values: np.ndarray,
    names: list[str],
    dt_sec: float,
    max_lag_sec: float = 60.0,
    threshold: float = 1.0 / np.e,
    nan_policy: str = "strict",
) -> Dict[str, Dict[str, Any]]:
    """Compute an autocorrelation length (tau) per column.

    Definition used here (simple, reproducible): the first lag where ACF falls below `threshold`
    (default: 1/e). Reported in seconds.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(names):
        return {}
    if not np.isfinite(dt_sec) or dt_sec <= 0:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    max_lag = int(max(1, round(max_lag_sec / dt_sec)))
    policy = str(nan_policy).strip().lower() or "strict"
    if policy not in {"strict", "interpolate"}:
        policy = "strict"

    def _acf_tau_1d(col: np.ndarray) -> float:
        """Internal helper: acf tau 1d."""
        x = np.asarray(col, dtype=float)
        finite_mask = np.isfinite(x)
        if not np.all(finite_mask):
            if policy == "strict":
                return float("nan")
            # Interpolate in index space without changing sample positions.
            if not np.any(finite_mask):
                return float("nan")
            idx = np.arange(x.size, dtype=float)
            x = np.interp(idx, idx[finite_mask], x[finite_mask])
        if x.size < 8:
            return float("nan")
        x = x - float(np.mean(x))
        var = float(np.mean(x * x))
        if not np.isfinite(var) or var <= 0:
            return float("nan")
        n = int(x.size)
        L = min(max_lag, n - 1)
        if L < 1:
            return float("nan")
        # FFT-based autocorrelation (biased) for speed and numerical stability
        nfft = 1
        while nfft < 2 * n:
            nfft *= 2
        fx = np.fft.rfft(x, n=nfft)
        ac = np.fft.irfft(fx * np.conj(fx), n=nfft)[: n]
        acf = ac / (ac[0] if ac[0] != 0 else (var * n))
        acf = np.asarray(acf, dtype=float)
        # find first lag where ACF < threshold
        below = np.where(acf[1 : L + 1] < float(threshold))[0]
        if below.size == 0:
            return float(L * dt_sec)
        lag = int(below[0] + 1)
        return float(lag * dt_sec)

    for j, name in enumerate(names):
        col = arr[:, j]
        tau = _acf_tau_1d(col)
        out[str(name)] = {
            "tau_sec": float(tau),
            "dt_sec": float(dt_sec),
            "max_lag_sec": float(max_lag_sec),
            "threshold": float(threshold),
            "nan_frac": float(1.0 - np.mean(np.isfinite(col))) if col.size else 1.0,
            "nan_policy": policy,
        }
    return out


def compute_tier2_jacobian_metrics(
    jacobian: Optional[np.ndarray],
    jacobian_diagnostics: Optional[Mapping[str, Any]] = None,
    max_windows_for_condition_number: int = 5000,
) -> Dict[str, Any]:
    """Tier-2 MNJ-adjacent metrics from the primary (typically 3×3) Jacobian."""
    if jacobian is None:
        return {}
    J = np.asarray(jacobian, dtype=float)
    if J.ndim != 3 or J.shape[0] == 0 or J.shape[1] != J.shape[2]:
        return {}
    W, D, _ = J.shape

    # Trace series (divergence)
    trace = np.trace(J, axis1=1, axis2=2)
    trace_f = trace[np.isfinite(trace)]
    signed_div: Dict[str, float] = {
        "n": int(trace_f.size),
        "frac_pos": float(np.mean(trace_f > 0)) if trace_f.size else float("nan"),
        "frac_neg": float(np.mean(trace_f < 0)) if trace_f.size else float("nan"),
        "mean_pos": float(np.mean(trace_f[trace_f > 0])) if np.any(trace_f > 0) else float("nan"),
        "mean_neg": float(np.mean(trace_f[trace_f < 0])) if np.any(trace_f < 0) else float("nan"),
        "mean_abs": float(np.mean(np.abs(trace_f))) if trace_f.size else float("nan"),
    }

    # Condition number per window: kappa(J)=sigma_max/sigma_min
    cond = np.full((W,), np.nan, dtype=float)
    finite_rows = np.all(np.isfinite(J), axis=(1, 2))
    total_finite_windows = int(np.sum(finite_rows))
    cond_windows = np.flatnonzero(finite_rows)
    max_w = int(max_windows_for_condition_number) if max_windows_for_condition_number is not None else 0
    subsampled = False
    if max_w > 0 and cond_windows.size > max_w:
        pick = np.linspace(0, cond_windows.size - 1, num=max_w, dtype=int)
        cond_windows = cond_windows[pick]
        subsampled = True

    if cond_windows.size > 0:
        try:
            svals = np.linalg.svd(J[cond_windows], compute_uv=False)  # [Ws, D]
            smin = np.min(svals, axis=1)
            smax = np.max(svals, axis=1)
            ok = np.isfinite(smin) & np.isfinite(smax) & (smin > 0)
            cond_vals = np.full(smin.shape, np.nan, dtype=float)
            cond_vals[ok] = smax[ok] / smin[ok]
            cond[cond_windows] = cond_vals
        except Exception:
            logger.exception("Failed vectorized SVD for Jacobian condition number")
    cond_desc = _distributional_descriptives(cond.reshape(-1, 1), ["kappa"])
    cond_desc = cond_desc.get("kappa", {}) if isinstance(cond_desc, dict) else {}
    if isinstance(cond_desc, dict):
        cond_desc["estimated_on_windows"] = int(cond_windows.size)
        cond_desc["total_finite_windows"] = total_finite_windows
        cond_desc["subsampled"] = bool(subsampled)

    # Rotation coherence (3D only): axis stability of antisymmetric component
    rot: Dict[str, Any] = {}
    if D == 3:
        omega = 0.5 * (J - np.transpose(J, (0, 2, 1)))
        wvec = np.stack([omega[:, 2, 1], omega[:, 0, 2], omega[:, 1, 0]], axis=1)  # [W,3]
        norms = np.linalg.norm(wvec, axis=1)
        mask = np.isfinite(norms) & (norms > 0)
        if np.any(mask):
            axes = (wvec[mask] / norms[mask, None]).astype(float)
            mean_axis = np.mean(axes, axis=0)
            mnorm = float(np.linalg.norm(mean_axis))
            rot = {
                "n": int(axes.shape[0]),
                "mean_resultant_length": mnorm,  # 0..1
                "mean_axis": [float(x) for x in (mean_axis / (mnorm if mnorm > 0 else 1.0))],
            }
        else:
            rot = {"n": 0, "mean_resultant_length": float("nan"), "mean_axis": [float("nan")] * 3}

    rel_mse_summary: Dict[str, Any] = {}
    if isinstance(jacobian_diagnostics, Mapping):
        rel_raw = jacobian_diagnostics.get("rel_mse_baseline_windows", None)
        if rel_raw is not None:
            rel_arr = np.asarray(rel_raw, dtype=float).reshape(-1)
            rel_desc = _distributional_descriptives(rel_arr.reshape(-1, 1), ["rel"])
            rel_desc = rel_desc.get("rel", {}) if isinstance(rel_desc, dict) else {}
            if isinstance(rel_desc, dict):
                rel_desc["note"] = "Relative local-fit MSE vs intercept-only baseline (<1 better than baseline)."
                rel_mse_summary = rel_desc
        elif "rel_mse_baseline_median" in jacobian_diagnostics:
            rel_mse_summary = {
                "median": float(jacobian_diagnostics.get("rel_mse_baseline_median")),
                "note": "Relative local-fit MSE vs intercept-only baseline (<1 better than baseline).",
            }

    return {
        "signed_divergence_balance": signed_div,
        "jacobian_condition_number": cond_desc,
        "rotation_coherence": rot,
        "rel_mse_baseline": rel_mse_summary,
    }


def compute_emmi_metrics(x: np.ndarray, x_dot: np.ndarray) -> Dict[str, float]:
    """Tier-2 derived indices (control/sensitivity proxies) from MNPS + speed."""
    X = np.asarray(x, dtype=float)
    Xd = np.asarray(x_dot, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3 or Xd.ndim != 2 or Xd.shape[1] != 3:
        return {}

    m = X[:, 0]
    d = X[:, 1]
    e = X[:, 2]
    speed = np.linalg.norm(Xd, axis=1)

    def _finite_median(arr: np.ndarray) -> float:
        """Internal helper: finite median."""
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        return float(np.median(a)) if a.size else float("nan")

    def _finite_mean(arr: np.ndarray) -> float:
        """Internal helper: finite mean."""
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        return float(np.mean(a)) if a.size else float("nan")

    m_med = _finite_median(m)
    d_med = _finite_median(d)
    e_med = _finite_median(e)
    sp_med = _finite_median(speed)
    sp_mean = _finite_mean(speed)

    mv_med = float("nan")
    mv_abs_med = float("nan")
    if np.isfinite(m_med) and np.isfinite(d_med) and np.isfinite(e_med):
        mv_med = float(m_med * d_med * e_med)
        mv_abs_med = float(abs(m_med) * abs(d_med) * abs(e_med))

    def _safe_ratio(num: float, den: float, abs_den_floor: float = 1e-6) -> float:
        """Internal helper: safe ratio."""
        if not (np.isfinite(num) and np.isfinite(den)):
            return float("nan")
        if abs(float(den)) < float(abs_den_floor):
            return float("nan")
        return float(num / den)

    emmi_e_over_m = _safe_ratio(e_med, m_med, abs_den_floor=1e-6)
    mv_over_speed = _safe_ratio(mv_med, sp_med, abs_den_floor=1e-6)

    return {
        "speed_mean": sp_mean,
        "speed_median": sp_med,
        "mv_median": mv_med,
        "mv_abs_median": mv_abs_med,
        "emmi_e_over_m_median": emmi_e_over_m,
        "mv_over_speed_median": mv_over_speed,
    }


def compute_ensemble_summary_for_subject(
    config: Mapping[str, Any],
    dataset_id: str,
    sub_frame: pd.DataFrame,
    coords_9d_names: list[str],
    subcoords_spec: Mapping[str, Mapping[str, float]],
    normalize_mode: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Compute ensemble-mean and variance over Stratified MNPS subcoordinates."""

    if not coords_9d_names or not subcoords_spec:
        return None

    robustness_cfg = config.get("robustness", {}) if isinstance(config, Mapping) else {}
    ensembles_cfg = robustness_cfg.get("ensembles", {}) if isinstance(robustness_cfg, Mapping) else {}
    if not isinstance(ensembles_cfg, Mapping) or not ensembles_cfg.get("enabled", False):
        return None

    config_groups = ensembles.resolve_config_groups(ensembles_cfg, dataset_id)
    if not config_groups:
        return None

    used_features: set[str] = set()
    for _, weight_map in subcoords_spec.items():
        if isinstance(weight_map, Mapping):
            for feat_name in weight_map.keys():
                used_features.add(str(feat_name))
    if not used_features:
        return None

    group_summaries: list[np.ndarray] = []
    realised_groups: list[str] = []

    for group_name in config_groups.keys():
        safe_name = ensembles.sanitize_group_name(group_name)
        suffix = f"__g_{safe_name}"

        needed_cols = {c for c in used_features if c in sub_frame.columns}
        needed_cols.update({f"{feat}{suffix}" for feat in used_features if f"{feat}{suffix}" in sub_frame.columns})
        df_group = sub_frame.loc[:, sorted(needed_cols)].copy() if needed_cols else pd.DataFrame(index=sub_frame.index)
        for feat in used_features:
            group_col = f"{feat}{suffix}"
            if group_col in df_group.columns:
                df_group[feat] = df_group[group_col]

        try:
            coords_g, names_g, _ = projection.project_features_v2(
                df_group,
                subcoords_spec,
                normalize=normalize_mode,
            )
        except Exception as exc:
            logger.warning("Failed to project v2 features for ensemble group %s in %s: %s", group_name, dataset_id, exc)
            continue

        if coords_g.size == 0:
            continue

        if coords_9d_names and list(names_g) != list(coords_9d_names):
            logger.warning(
                "coords_9d name mismatch for ensemble group %s in %s; skipping group", group_name, dataset_id
            )
            continue

        summary_g = np.nanmedian(coords_g, axis=0)
        if not np.all(np.isfinite(summary_g)):
            if np.all(np.isnan(summary_g)):
                continue
            summary_g = np.where(np.isfinite(summary_g), summary_g, np.nan)

        group_summaries.append(summary_g.astype(np.float32))
        realised_groups.append(str(group_name))

    if not group_summaries:
        return None

    stack = np.stack(group_summaries, axis=0)
    ensemble_mean = np.nanmean(stack, axis=0)
    ensemble_var = np.nanvar(stack, axis=0)

    return {
        "groups_config": config_groups,
        "groups_realised": realised_groups,
        "subcoord_names": list(coords_9d_names),
        "mean": {name: float(ensemble_mean[i]) for i, name in enumerate(coords_9d_names)},
        "var": {name: float(ensemble_var[i]) for i, name in enumerate(coords_9d_names)},
    }


def compute_robust_and_reliability_summaries(
    config: Mapping[str, Any],
    mnps_cfg: Mapping[str, Any],
    x: np.ndarray,
    coords_9d: Optional[np.ndarray],
    coords_9d_names: list[str],
) -> Dict[str, Any]:
    """Compute robust summaries and split-half reliability for MNPS coordinates."""

    result: Dict[str, Any] = {}

    try:
        axes_names = ["m", "d", "e"]
        axes_summary = robustness.summarize_array(x, axes_names, config)
        axes_reliability = robustness.split_half_reliability(x, axes_names)
        result["axes"] = {
            "summary": axes_summary,
            "reliability": axes_reliability,
        }
    except Exception:
        logger.exception("Failed to compute robustness summaries for MNPS axes")

    if coords_9d is not None and coords_9d_names:
        try:
            sub_summary = robustness.summarize_array(coords_9d, coords_9d_names, config)
            sub_reliability = robustness.split_half_reliability(coords_9d, coords_9d_names)
            result["subcoords"] = {
                "summary": sub_summary,
                "reliability": sub_reliability,
            }
        except Exception:
            logger.exception("Failed to compute robustness summaries for Stratified MNPS subcoordinates")

    return result


def compute_psd_multiverse_stability(
    config: Mapping[str, Any],
    ds_id: str,
    sub_frame: pd.DataFrame,
    coords_9d: Optional[np.ndarray],
    coords_9d_names: list[str],
    subcoords_spec: Mapping[str, Mapping[str, float]],
    normalize_mode: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Compute PSD multiverse stability indices for Stratified subcoordinates."""

    if coords_9d is None or not coords_9d_names or not subcoords_spec:
        return None

    robustness_cfg = config.get("robustness", {}) if isinstance(config, Mapping) else {}
    multiverse_cfg = robustness_cfg.get("multiverse", {}) if isinstance(robustness_cfg, Mapping) else {}
    psd_mv_cfg = multiverse_cfg.get("psd", {}) if isinstance(multiverse_cfg, Mapping) else {}
    if not isinstance(psd_mv_cfg, Mapping) or not psd_mv_cfg.get("enabled", False):
        return None

    used_features: set[str] = set()
    for _, weight_map in subcoords_spec.items():
        if isinstance(weight_map, Mapping):
            for feat_name in weight_map.keys():
                used_features.add(str(feat_name))

    alt_candidates = {feat for feat in used_features if f"{feat}__psd_alt" in sub_frame.columns}
    if not alt_candidates:
        return None

    needed_cols = set(used_features)
    needed_cols.update({f"{feat}__psd_alt" for feat in alt_candidates})
    needed_cols = {c for c in needed_cols if c in sub_frame.columns}
    df_alt = sub_frame.loc[:, sorted(needed_cols)].copy() if needed_cols else pd.DataFrame(index=sub_frame.index)
    for feat in alt_candidates:
        alt_col = f"{feat}__psd_alt"
        df_alt[feat] = df_alt[alt_col]

    try:
        coords_alt, names_alt, _ = projection.project_features_v2(
            df_alt,
            subcoords_spec,
            normalize=normalize_mode,
        )
    except Exception as exc:
        logger.warning("Failed to project v2 features for PSD multiverse in %s: %s", ds_id, exc)
        return None

    if coords_alt.size == 0 or list(names_alt) != list(coords_9d_names):
        return None

    T, K = coords_9d.shape
    if T != coords_alt.shape[0] or K != coords_alt.shape[1]:
        return None

    stability: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(coords_9d_names):
        primary = coords_9d[:, idx]
        alt = coords_alt[:, idx]
        mask = np.isfinite(primary) & np.isfinite(alt)
        primary = primary[mask]
        alt = alt[mask]
        if primary.size < 2:
            stability[str(name)] = {"corr": float("nan"), "mean_abs_diff": float("nan")}
            continue
        p_mean = float(primary.mean())
        a_mean = float(alt.mean())
        num = float(np.sum((primary - p_mean) * (alt - a_mean)))
        den = float(np.sqrt(np.sum((primary - p_mean) ** 2) * np.sum((alt - a_mean) ** 2)))
        corr = float("nan") if den == 0 else float(num / den)
        mad = float(np.mean(np.abs(primary - alt)))
        stability[str(name)] = {"corr": corr, "mean_abs_diff": mad}

    primary_method = str(config.get("features", {}).get("eeg_psd", {}).get("method", "multitaper")).lower()
    secondary_method = str(psd_mv_cfg.get("secondary_method", "welch")).lower()

    return {
        "methods": {
            "primary": primary_method,
            "secondary": secondary_method,
        },
        "stability": stability,
    }


def build_qc_summary(
    dataset_label: str,
    ds_path: Path,
    sub_id: str,
    ses_id: Optional[str],
    sub_frame: pd.DataFrame,
    dt: float,
    ensemble_summary: Optional[Dict[str, Any]],
    robust_summary: Dict[str, Any] | None,
    dist_summary: Dict[str, Any] | None,
    entropy_qc: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Aggregate per-subject robustness into a compact QC summary."""

    epochs = int(len(sub_frame))
    seconds = float(epochs * dt)

    artifact_summary: Dict[str, Any] = {}
    try:
        qc_dir = ds_path / "qc_artifacts"
        methods: set[str] = set()
        bad_channels: set[str] = set()
        if qc_dir.exists() and "file" in sub_frame.columns:
            file_values = [str(f) for f in sub_frame["file"].dropna().astype(str).unique()]
            stem_to_names: Dict[str, set[str]] = {}
            for file_name in file_values:
                stem_to_names.setdefault(Path(file_name).stem, set()).add(Path(file_name).name)
            stem_collisions = {s: sorted(list(v)) for s, v in stem_to_names.items() if len(v) > 1}
            if stem_collisions:
                logger.warning(
                    "QC artifact stem collisions detected for %s: %s",
                    dataset_label,
                    stem_collisions,
                )
            file_stems = set(stem_to_names.keys())
            import json  # local import to avoid top-level dependency

            for stem in file_stems:
                qc_path = qc_dir / f"{stem}_qc_artifacts.json"
                if not qc_path.exists():
                    continue
                try:
                    with qc_path.open("r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    continue
                art = meta.get("artifact", {}) if isinstance(meta, dict) else {}
                method = art.get("method")
                if isinstance(method, str):
                    methods.add(method)
                for ch in art.get("bad_eeg_channels", []) or []:
                    bad_channels.add(str(ch))
        artifact_summary = {
            "methods": sorted(methods) if methods else [],
            "bad_eeg_channels": sorted(bad_channels) if bad_channels else [],
            "n_bad_eeg_channels": len(bad_channels),
            "stem_collisions": stem_collisions if "stem_collisions" in locals() else {},
        }
    except Exception:
        logger.exception("Failed to aggregate artifact metrics for %s", dataset_label)
        artifact_summary = {}

    ensemble_metrics: Dict[str, Any] = {}
    if ensemble_summary is not None:
        var_map = ensemble_summary.get("var", {}) or {}
        try:
            vals = np.asarray(list(var_map.values()), dtype=float)
            ensemble_metrics = {
                "var_by_subcoord": var_map,
                "var_mean": float(np.nanmean(vals)) if vals.size else float("nan"),
                "var_max": float(np.nanmax(vals)) if vals.size else float("nan"),
            }
        except Exception:
            ensemble_metrics = {"var_by_subcoord": var_map}

    reliability_axes = {}
    reliability_subcoords = {}
    if robust_summary:
        reliability_axes = (robust_summary.get("axes") or {}).get("reliability", {}) or {}
        reliability_subcoords = (robust_summary.get("subcoords") or {}).get("reliability", {}) or {}

    provisional_axes: Dict[str, Any] = {}
    if entropy_qc:
        for name, info in entropy_qc.items():
            if bool(info.get("provisional", False)):
                provisional_axes[name] = info

    return {
        "dataset_id": dataset_label,
        "subject": sub_id,
        "session": ses_id,
        "coverage": {
            "epochs": epochs,
            "seconds": seconds,
        },
        "artifacts": artifact_summary,
        "ensemble": ensemble_metrics,
        "reliability": {
            "axes": reliability_axes,
            "subcoords": reliability_subcoords,
        },
        # Neutral distributional descriptives (mean/median/std/iqr + delta).
        "dist_summary": dist_summary or {},
        "entropy_provisional": provisional_axes,
    }

