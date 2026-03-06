"""Projection helpers for MNPS coordinates and derivatives.

This module now exposes both the legacy DataFrame-facing helper
``project_to_mnps`` as well as array-oriented utilities consumed by the new
tensor pipeline:

* ``project_features`` returns an ``[T, 3]`` float32 array ``x`` with
  coordinates ``[m, d, e]``.
* ``estimate_derivatives`` supports Savitzky–Golay and central-difference
  derivative estimation for ``x``.
* ``build_knn_indices`` constructs neighbour indices using ``scipy``'s
  ``cKDTree`` for downstream Jacobian estimation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover - SciPy is installed in production envs
    savgol_filter = None  # type: ignore

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None  # type: ignore

logger = logging.getLogger(__name__)


AXES = ("m", "d", "e")
ROBUST_MAD_TO_SIGMA = 1.4826


def _normalize_matrix_unit_interval(
    matrix: np.ndarray,
    eps: float = 1e-6,
    min_finite_count: int = 3,
) -> np.ndarray:
    """Map columns of ``matrix`` into [0, 1] using a robust sigmoidal transform.

    Each column is robustly centred (median) and scaled (1.4826*MAD with std fallback),
    then passed through a logistic squashing function to guarantee bounded
    output. NaNs are preserved.
    """

    if matrix.size == 0:
        return matrix

    out = matrix.astype(np.float32, copy=True)

    for col_idx in range(out.shape[1]):
        col = out[:, col_idx]
        mask = np.isfinite(col)
        if not mask.any():
            continue

        finite_vals = col[mask]
        # Low-support columns are unstable to robust scaling; keep them as NaN.
        if finite_vals.size < int(min_finite_count):
            out[:, col_idx] = np.nan
            continue
        median = float(np.nanmedian(finite_vals))
        mad = float(np.nanmedian(np.abs(finite_vals - median)))
        if not np.isfinite(mad) or mad <= eps:
            std = float(np.nanstd(finite_vals))
            scale = std if np.isfinite(std) and std > eps else 1.0
        else:
            # Approximate robust standard deviation under normality.
            scale = ROBUST_MAD_TO_SIGMA * mad

        z_scores = (finite_vals - median) / scale
        clipped = np.clip(z_scores, -60.0, 60.0)
        mapped = 1.0 / (1.0 + np.exp(-clipped))
        out[mask, col_idx] = np.clip(mapped, 0.0, 1.0).astype(np.float32)

    return out


def _normalize_used_columns(
    df: pd.DataFrame, 
    used_cols: Sequence[str], 
    normalize: Optional[str],
    pipeline_map: Optional[Mapping[str, Sequence[str]]] = None,
    clip_thresh: float = 6.0
) -> tuple[pd.DataFrame, dict]:
    """Normalize selected columns in a DataFrame copy.
    
    Applies deterministic standardization (Log10 -> Robust Z -> Clip) to raw features prior to 
    MNPS projection. Ensures numerical stability and commensurate scaling across disparate metrics 
    (power, ratios, complexity) to prevent singular matrices during downstream local Jacobian estimation, 
    while retaining absolute scaling parameters as metadata for macro-state baseline comparisons.
    """
    if not normalize or not used_cols:
        return df, {}
        
    out = df.copy()
    baselines = {}
    
    for col in used_cols:
        col_data = out[col].astype(np.float32).values
        mask = np.isfinite(col_data)
        
        if not mask.any():
            baselines[col] = {
                "abs_median": float('nan'),
                "abs_mad": float('nan'),
                "transformation_applied": "none"
            }
            continue
            
        finite_vals = col_data[mask]
        
        # Capture absolute baseline
        raw_median = float(np.nanmedian(finite_vals))
        raw_mad = float(np.nanmedian(np.abs(finite_vals - raw_median))) * 1.4826
        
        baseline_info = {
            "abs_median": raw_median,
            "abs_mad": raw_mad,
        }
        
        # Determine pipeline
        pipeline = None
        if pipeline_map and col in pipeline_map:
            pipeline = pipeline_map[col]
        elif pipeline_map and "default" in pipeline_map:
            pipeline = pipeline_map["default"]
            
        if pipeline is None:
            # Fallback if no specific rule applies. Entropy should NEVER be log10-transformed blind.
            # Ratios and power are usually >0 but log10 is now explicitly configured. 
            # We default to just robust_z and clip unless explicitly requested otherwise.
            pipeline = ["robust_z", "clip"]
                
        transformed = col_data.copy()
        applied_steps = []
        
        for step in pipeline:
            step_str = str(step).strip().lower()
            if step_str == "log10":
                transformed[mask] = np.log10(np.clip(transformed[mask], 1e-9, None))
                applied_steps.append("log10")
            elif step_str in ("robust_z", "robust"):
                t_finite = transformed[mask]
                t_median = np.nanmedian(t_finite)
                t_mad = np.nanmedian(np.abs(t_finite - t_median)) * 1.4826
                transformed[mask] = (t_finite - t_median) / (t_mad + 1e-9)
                applied_steps.append("robust_z")
            elif step_str == "z":
                t_finite = transformed[mask]
                mu = np.nanmean(t_finite)
                sigma = np.nanstd(t_finite)
                transformed[mask] = (t_finite - mu) / (sigma + 1e-9)
                applied_steps.append("z")
            elif step_str == "clip":
                transformed[mask] = np.clip(transformed[mask], -clip_thresh, clip_thresh)
                applied_steps.append(f"clip_{clip_thresh}")

        baseline_info["transformation_applied"] = " -> ".join(applied_steps) if applied_steps else "none"
        baselines[col] = baseline_info
        
        # Assign back
        new_col_data = np.zeros_like(col_data)
        new_col_data[mask] = transformed[mask]
        out[col] = new_col_data
        
    return out, baselines


def project_features(
    features_df: pd.DataFrame, 
    weights: Mapping[str, Mapping[str, float]], 
    normalize: Optional[str] = None,
    feature_standardization: Optional[Mapping[str, Sequence[str]]] = None,
    clip_threshold: float = 6.0
) -> tuple[np.ndarray, dict[str, dict]]:
    """Return MNPS coordinates ``x=[m,d,e]`` as a float32 array, and baseline metadata.

    Parameters
    ----------
    features_df
        DataFrame with feature columns
    weights
        Mapping of axis to feature weights
    normalize
        Optional normalization for feature columns used in weights: 'z', 'robust_z' or None
    feature_standardization
        Optional mapping of feature -> sequence of operations
    clip_threshold
        Threshold for clipping features
    """

    if len(features_df) == 0:
        return np.zeros((0, 3), dtype=np.float32), {}

    used_cols = []
    missing_by_axis: Dict[str, list[str]] = {axis: [] for axis in AXES}
    for axis in AXES:
        for feat_name in weights.get(axis, {}).keys():
            if feat_name in features_df.columns:
                used_cols.append(feat_name)
            else:
                missing_by_axis[axis].append(str(feat_name))
    for axis in AXES:
        missing = sorted(set(missing_by_axis[axis]))
        if missing:
            logger.debug("Missing weighted features for axis '%s': %s", axis, ", ".join(missing))
    used_cols = sorted(set(used_cols))
    if not used_cols:
        return np.zeros((len(features_df), 3), dtype=np.float32), {}

    baselines = {}
    if normalize:
        features_df, baselines = _normalize_used_columns(
            features_df, 
            used_cols, 
            normalize,
            pipeline_map=feature_standardization,
            clip_thresh=clip_threshold
        )

    # Keep missing values as NaN here; per-axis aggregation below renormalizes
    # by present weights (sum of abs(weights)) to avoid silent fillna(0)-bias.
    X = features_df.loc[:, used_cols].to_numpy(dtype=np.float32, copy=True)
    X[~np.isfinite(X)] = np.nan
    W = np.zeros((len(used_cols), 3), dtype=np.float32)
    col_idx_map = {c: i for i, c in enumerate(used_cols)}
    for axis_idx, axis in enumerate(AXES):
        for feat_name, weight in weights.get(axis, {}).items():
            i = col_idx_map.get(feat_name)
            if i is not None:
                W[i, axis_idx] = np.float32(weight)
    out = np.zeros((len(features_df), 3), dtype=np.float32)
    for axis_idx in range(3):
        w = W[:, axis_idx]
        nz = np.abs(w) > 0
        if not np.any(nz):
            continue
        x_axis = X[:, nz]
        w_axis = w[nz]
        finite = np.isfinite(x_axis)
        numerator = np.where(finite, x_axis * w_axis[None, :], 0.0).sum(axis=1)
        denom = np.where(finite, np.abs(w_axis)[None, :], 0.0).sum(axis=1)
        out[:, axis_idx] = np.divide(
            numerator,
            denom,
            out=np.full_like(numerator, np.nan, dtype=np.float32),
            where=denom > 0,
        ).astype(np.float32)
    return out, baselines


def project_features_with_coverage(
    features_df: pd.DataFrame,
    weights: Mapping[str, Mapping[str, float]],
    normalize: Optional[str] = None,
    feature_standardization: Optional[Mapping[str, Sequence[str]]] = None,
    clip_threshold: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict]]:
    """Project direct MNPS coordinates and return per-epoch axis coverage.

    Coverage is defined per axis and epoch as:
        sum(|w_i| for finite weighted features) / sum(|w_i| for configured axis weights)
    and lies in [0, 1]. It is NaN when an axis has no configured non-zero weights.
    """
    x, baselines = project_features(
        features_df, 
        weights, 
        normalize=normalize,
        feature_standardization=feature_standardization,
        clip_threshold=clip_threshold
    )
    coverage = np.full_like(x, np.nan, dtype=np.float32)
    if len(features_df) == 0:
        return x, coverage, baselines

    used_cols = sorted(
        {
            str(feat_name)
            for axis in AXES
            for feat_name in (weights.get(axis, {}) or {}).keys()
            if feat_name in features_df.columns
        }
    )
    if not used_cols:
        return x, coverage, baselines

    baselines = {}
    if normalize:
        features_df_norm, _ = _normalize_used_columns(
            features_df, 
            used_cols, 
            normalize,
            pipeline_map=feature_standardization,
            clip_thresh=clip_threshold
        )
    else:
        features_df_norm = features_df
        
    X = features_df_norm.loc[:, used_cols].to_numpy(dtype=np.float32, copy=True)
    X[~np.isfinite(X)] = np.nan
    col_idx_map = {c: i for i, c in enumerate(used_cols)}

    for axis_idx, axis in enumerate(AXES):
        w_map = weights.get(axis, {}) or {}
        axis_indices = []
        axis_weights = []
        for feat_name, weight in w_map.items():
            i = col_idx_map.get(str(feat_name))
            if i is None:
                continue
            w = float(weight)
            if abs(w) <= 0:
                continue
            axis_indices.append(i)
            axis_weights.append(abs(w))
        if not axis_indices:
            continue
        x_axis = X[:, axis_indices]
        abs_w = np.asarray(axis_weights, dtype=np.float32)
        denom_full = float(np.sum(abs_w))
        finite = np.isfinite(x_axis)
        denom_present = np.where(finite, abs_w[None, :], 0.0).sum(axis=1)
        coverage[:, axis_idx] = np.divide(
            denom_present,
            denom_full,
            out=np.full_like(denom_present, np.nan, dtype=np.float32),
            where=denom_full > 0,
        ).astype(np.float32)
    return x, coverage, baselines


def project_features_v2(
    features_df: pd.DataFrame,
    subcoords: Mapping[str, Mapping[str, float]],
    normalize: Optional[str] = None,
    missing_policy: str = "renorm",
    feature_standardization: Optional[Mapping[str, Sequence[str]]] = None,
    clip_threshold: float = 6.0,
) -> tuple[np.ndarray, list[str], dict[str, dict]]:
    """Return MNPS v2 subcoordinates as a float32 array, their names, and feature baselines.

    Parameters
    ----------
    features_df
        Feature dataframe
    subcoords
        Mapping subcoord_name -> {feature_name: weight}
    normalize
        Optional normalization for used columns: 'z', 'robust_z' or None
    feature_standardization
        Optional mapping of feature -> sequence of operations
    clip_threshold
        Threshold for clipping features
    """
    if len(features_df) == 0 or not subcoords:
        return np.zeros((0, 0), dtype=np.float32), [], {}

    names = list(subcoords.keys())
    used_cols = []
    for weight_map in subcoords.values():
        for feat_name in weight_map.keys():
            if feat_name in features_df.columns:
                used_cols.append(feat_name)
            else:
                logger.debug("Skipping missing feature '%s'", feat_name)
    used_cols = sorted(set(used_cols))
    if not used_cols:
        return np.zeros((len(features_df), len(names)), dtype=np.float32), names, {}

    baselines = {}
    if normalize:
        features_df, baselines = _normalize_used_columns(features_df, used_cols, normalize)

    X = features_df.loc[:, used_cols].to_numpy(dtype=np.float32, copy=True)
    X[~np.isfinite(X)] = np.nan
    policy = str(missing_policy).strip().lower() or "renorm"
    if policy not in {"renorm"}:
        raise ValueError(f"Unsupported v2 missing_policy '{missing_policy}'")
    W = np.zeros((len(used_cols), len(names)), dtype=np.float32)
    col_idx_map = {c: i for i, c in enumerate(used_cols)}
    for sc_idx, sc_name in enumerate(names):
        for feat_name, weight in subcoords.get(sc_name, {}).items():
            i = col_idx_map.get(feat_name)
            if i is not None:
                W[i, sc_idx] = np.float32(weight)

    Xv2 = np.full((len(features_df), len(names)), np.nan, dtype=np.float32)
    for sc_idx in range(len(names)):
        w = W[:, sc_idx]
        nz = np.abs(w) > 0
        if not np.any(nz):
            continue
        x_sc = X[:, nz]
        w_sc = w[nz]
        finite = np.isfinite(x_sc)
        numerator = np.where(finite, x_sc * w_sc[None, :], 0.0).sum(axis=1)
        denom = np.where(finite, np.abs(w_sc)[None, :], 0.0).sum(axis=1)
        Xv2[:, sc_idx] = np.divide(
            numerator,
            denom,
            out=np.full_like(numerator, np.nan, dtype=np.float32),
            where=denom > 0,
        ).astype(np.float32)

    # In Step 3 we removed the global post-projection normalization/scaling,
    # because features are pre-standardized before projection!
    return Xv2, names, baselines


def construct_fixed_projection_matrix(
    coords_9d_names: Sequence[str],
    config_weights: Mapping[str, Any],
    *,
    enforce_block_selective: bool = True,
    normalize_columns_l2: bool = True,
    l2_epsilon: float = 1e-9,
) -> tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    """Construct and L2-normalize fixed V2->V1 projection matrix from config."""
    idx_map = {str(name): i for i, name in enumerate(coords_9d_names)}
    P = np.zeros((len(coords_9d_names), 3), dtype=np.float32)

    for axis_idx, axis in enumerate(AXES):
        raw = config_weights.get(axis, {}) if isinstance(config_weights, Mapping) else {}
        if isinstance(raw, Mapping):
            items = [(str(k), float(v)) for k, v in raw.items()]
        else:
            seq = raw if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else []
            items = [(str(name), 1.0) for name in seq]

        if enforce_block_selective:
            bad = [name for name, _ in items if not str(name).startswith(f"{axis}_")]
            if bad:
                raise ValueError(
                    f"V1 mapping for axis '{axis}' includes cross-block subcoords: {bad}"
                )
        for sub_name, weight in items:
            if sub_name in idx_map and np.isfinite(weight):
                P[idx_map[sub_name], axis_idx] = np.float32(weight)

    # Enforce L2 normalization per V1 axis column during runtime.
    if normalize_columns_l2:
        col_norms = np.linalg.norm(P, ord=2, axis=0).astype(np.float32)
        safe_den = np.where(
            np.isfinite(col_norms),
            col_norms + np.float32(max(l2_epsilon, 0.0)),
            np.float32(1.0),
        )
        P = (P / safe_den[None, :]).astype(np.float32)

    weights_normalized: Dict[str, Dict[str, float]] = {axis: {} for axis in AXES}
    for axis_idx, axis in enumerate(AXES):
        row: Dict[str, float] = {}
        for name, i in idx_map.items():
            val = float(P[i, axis_idx])
            if abs(val) > 0:
                row[name] = val
        weights_normalized[axis] = row
    return P, weights_normalized


def apply_fixed_projection(
    v2_tensor: np.ndarray,
    projection_matrix: np.ndarray,
) -> np.ndarray:
    """Apply fixed projection matrix to V2 tensor (x = V2 @ P)."""
    X = np.asarray(v2_tensor, dtype=np.float32)
    P = np.asarray(projection_matrix, dtype=np.float32)
    if X.ndim != 2 or P.ndim != 2:
        raise ValueError("apply_fixed_projection expects 2D tensors")
    if X.shape[1] != P.shape[0]:
        raise ValueError(
            f"Dimension mismatch in apply_fixed_projection: X has {X.shape[1]} columns but P has {P.shape[0]} rows"
        )
    return (X @ P).astype(np.float32)


def derive_mde_from_v2(
    coords_9d: np.ndarray,
    coords_9d_names: Sequence[str],
    axis_map: Mapping[str, Any],
    pooling: str = "mean",
    *,
    normalize_columns_l2: bool = True,
    enforce_block_selective: bool = True,
    l2_epsilon: float = 1e-9,
    return_mapping_info: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Derive canonical ``x=[m,d,e]`` from stratified ``coords_9d``.

    The preferred runtime path is a fixed weighted V2->V1 projection matrix built
    from config and L2-normalized per output axis. This preserves configured
    asymmetries between subcoordinates and avoids documenting the result as a
    trivial equal-weight mean. Legacy ``pooling`` values are only descriptive for
    simpler group-pooling configurations; when a weighted mapping is supplied,
    the effective aggregation is the fixed weighted projection.

    Returns
    -------
    x, coverage
        ``x`` is ``[T,3]`` and ``coverage`` is per-axis finite fraction in ``[0,1]``.
        Coverage is NaN for axes with no mapped columns in ``coords_9d_names``.
    """
    X = np.asarray(coords_9d, dtype=np.float32)
    T = int(X.shape[0]) if X.ndim == 2 else 0
    if X.ndim != 2:
        raise ValueError("derive_mde_from_v2 expects coords_9d with shape [T, S]")

    mode = str(pooling).strip().lower() or "mean"
    if mode not in {"mean", "sum"}:
        raise ValueError(f"Unsupported from_v2 pooling '{pooling}'")

    out = np.full((T, 3), np.nan, dtype=np.float32)
    cov = np.full((T, 3), np.nan, dtype=np.float32)
    if T == 0 or len(coords_9d_names) == 0:
        return out, cov

    idx_map = {str(name): i for i, name in enumerate(coords_9d_names)}
    mapping_mode = "weighted"
    if isinstance(axis_map, Mapping):
        for axis in AXES:
            raw = axis_map.get(axis, {})
            if not isinstance(raw, Mapping):
                mapping_mode = "group_pooling"
                break

    use_l2_normalization = bool(normalize_columns_l2)
    if mapping_mode == "group_pooling" and mode == "sum":
        use_l2_normalization = False

    P, axis_weight_rows = construct_fixed_projection_matrix(
        coords_9d_names,
        axis_map,
        enforce_block_selective=enforce_block_selective,
        normalize_columns_l2=use_l2_normalization,
        l2_epsilon=l2_epsilon,
    )

    X_safe = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    numer_all = apply_fixed_projection(X_safe, P)
    finite_all = np.isfinite(X)
    absP = np.abs(P)
    for axis_idx, axis in enumerate(AXES):
        w = P[:, axis_idx]
        nz = np.abs(w) > 0
        if not np.any(nz):
            continue
        numer = numer_all[:, axis_idx]
        denom = np.where(finite_all[:, nz], absP[nz, axis_idx][None, :], 0.0).sum(axis=1)
        denom_full = float(np.sum(absP[nz, axis_idx]))
        if mapping_mode == "group_pooling" and mode == "sum":
            out[:, axis_idx] = np.where(
                denom > 0,
                numer,
                np.full_like(numer, np.nan, dtype=np.float32),
            ).astype(np.float32)
        else:
            out[:, axis_idx] = np.divide(
                numer,
                denom,
                out=np.full_like(numer, np.nan, dtype=np.float32),
                where=denom > 0,
            )
        cov[:, axis_idx] = np.divide(
            denom,
            denom_full,
            out=np.full_like(denom, np.nan, dtype=np.float32),
            where=denom_full > 0,
        )
    mapping_info = {
        "mode": mapping_mode,
        "pooling_requested": mode,
        "aggregation": (
            "fixed_weighted_projection"
            if mapping_mode == "weighted"
            else f"group_pooling_{mode}"
        ),
        "normalize_columns_l2": bool(use_l2_normalization),
        "l2_epsilon": float(l2_epsilon),
        "enforce_block_selective": bool(enforce_block_selective),
        "weights_normalized": axis_weight_rows,
        "matrix": P.astype(np.float32).tolist(),
        "coords_9d_names": [str(n) for n in coords_9d_names],
    }
    if return_mapping_info:
        return out, cov, mapping_info
    return out, cov

def project_to_mnps(features_df: pd.DataFrame, weights: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    """Legacy helper returning a DataFrame with MNPS coordinates.

    The returned DataFrame contains canonical ``m/d/e`` columns.
    """

    if len(features_df) == 0:
        return pd.DataFrame(columns=["epoch_id", "m", "d", "e"])
    if "epoch_id" not in features_df.columns:
        raise ValueError("features_df must contain 'epoch_id'")

    x, _ = project_features(features_df, weights)
    out_df = features_df[["epoch_id"]].copy()
    out_df[["m", "d", "e"]] = x

    logger.info("Projected %s epochs to MNPS", len(out_df))
    return out_df


def estimate_derivatives(x: np.ndarray, dt: float, method: str = "sav_gol", window: int = 7, polyorder: int = 3) -> np.ndarray:
    """Estimate ``x_dot`` for MNPS coordinates.

    Parameters
    ----------
    x:
        Array with shape ``[T, 3]``.
    dt:
        Sampling interval in seconds for the MNPS time base.
    method:
        Either ``sav_gol`` (default) or ``central``.
    window:
        Window length for Savitzky–Golay (must be odd).
    polyorder:
        Polynomial order for Savitzky–Golay.
    """

    if x.size == 0:
        return np.zeros_like(x, dtype=np.float32)

    finite_rows = np.isfinite(x).all(axis=1)
    if not np.all(finite_rows):
        # Do not impute hidden values: preserve NaN rows and compute derivatives only on
        # contiguous finite segments.
        out = np.full_like(x, np.nan, dtype=np.float32)
        idx = np.flatnonzero(finite_rows)
        if idx.size == 0:
            return out
        split_points = np.where(np.diff(idx) > 1)[0]
        starts = np.concatenate(([0], split_points + 1))
        stops = np.concatenate((split_points + 1, [idx.size]))
        for s, e in zip(starts, stops):
            seg_idx = idx[s:e]
            seg = x[seg_idx]
            if seg.shape[0] < 2:
                continue
            out_seg = estimate_derivatives(seg, dt, method=method, window=window, polyorder=polyorder)
            out[seg_idx] = out_seg
        logger.debug(
            "estimate_derivatives propagated NaNs and processed %d finite segment(s)",
            int(len(starts)),
        )
        return out

    if method == "central":
        grad = np.gradient(x, dt, axis=0)
        return grad.astype(np.float32)

    if method != "sav_gol":
        raise ValueError(f"Unknown derivative method '{method}'")

    if savgol_filter is None:
        raise ImportError("scipy.signal.savgol_filter is required for sav_gol derivatives")

    # Robustly adapt window/polyorder to sequence length
    T = x.shape[0]
    orig_window = window
    orig_poly = polyorder

    # Compute the maximum valid odd window length not exceeding T
    max_window = T if (T % 2 == 1) else max(0, T - 1)

    # If too few points for Savitzky–Golay, fall back to central differences
    if max_window < 3:
        logger.info("Too few epochs (T=%s) for Savitzky–Golay; using central differences", T)
        grad = np.gradient(x, dt, axis=0)
        return grad.astype(np.float32)

    # Make requested window odd and cap to max_window
    if window % 2 == 0:
        window -= 1
    window = max(3, min(window, max_window))

    # Ensure polyorder < window, clamp to at least 1
    if polyorder >= window:
        polyorder = max(1, window - 1)

    if window != orig_window or polyorder != orig_poly:
        logger.debug(
            "Adjusted SavGol params for T=%s: window %s->%s, polyorder %s->%s",
            T, orig_window, window, orig_poly, polyorder
        )

    deriv = savgol_filter(
        x,
        window_length=window,
        polyorder=polyorder,
        deriv=1,
        delta=dt,
        axis=0,
        mode="interp",
    )
    return np.asarray(deriv, dtype=np.float32)


def estimate_derivatives_segmented(
    x: np.ndarray,
    dt: float,
    method: str = "sav_gol",
    *,
    max_jump: float = 5.0,
    min_seg: int = 9,
    savgol_window: int = 7,
    polyorder: int = 3,
) -> np.ndarray:
    """Estimate derivatives on robust segments split by large trajectory jumps.

    This keeps the measurement model intact while reducing derivative smearing
    across discontinuities/outliers.
    """
    X = np.asarray(x, dtype=np.float32)
    if X.size == 0:
        return np.zeros_like(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("estimate_derivatives_segmented expects a 2-D array")
    T = X.shape[0]
    if T < 3:
        return np.zeros_like(X, dtype=np.float32)

    finite_rows = np.isfinite(X).all(axis=1)
    out = np.full_like(X, np.nan, dtype=np.float32)
    if not np.any(finite_rows):
        return out

    # Work only on contiguous finite segments.
    idx = np.flatnonzero(finite_rows)
    split_points = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate(([0], split_points + 1))
    stops = np.concatenate((split_points + 1, [idx.size]))
    for s, e in zip(starts, stops):
        seg_idx = idx[s:e]
        seg = X[seg_idx]
        if seg.shape[0] < 3:
            continue

        # Detect large in-segment jumps and split further.
        d = np.linalg.norm(np.diff(seg, axis=0), axis=1)
        med = float(np.nanmedian(d)) if d.size else 0.0
        mad = float(np.nanmedian(np.abs(d - med))) if d.size else 0.0
        sigma = ROBUST_MAD_TO_SIGMA * mad if np.isfinite(mad) and mad > 0 else float(np.nanstd(d))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0
        thr = med + float(max_jump) * sigma
        breaks = np.where(d > thr)[0]
        sub_starts = np.concatenate(([0], breaks + 1))
        sub_ends = np.concatenate((breaks + 1, [seg.shape[0]]))
        for ss, ee in zip(sub_starts, sub_ends):
            sub = seg[ss:ee]
            if sub.shape[0] < 2:
                continue
            if method == "sav_gol" and sub.shape[0] >= int(max(3, min_seg)):
                # Keep odd window and valid polyorder for each subsegment.
                win = int(max(3, savgol_window))
                if win % 2 == 0:
                    win -= 1
                max_win = sub.shape[0] if (sub.shape[0] % 2 == 1) else max(0, sub.shape[0] - 1)
                win = max(3, min(win, max_win))
                p = int(min(max(1, polyorder), win - 1))
                sub_dot = estimate_derivatives(sub, dt, method="sav_gol", window=win, polyorder=p)
            else:
                sub_dot = estimate_derivatives(sub, dt, method="central")
            out[seg_idx[ss:ee]] = sub_dot

    out[~np.isfinite(out)] = 0.0
    return out


def build_time_index(num_epochs: int, window_sec: float, overlap: float, origin: float = 0.0) -> np.ndarray:
    """Construct the canonical MNPS time vector."""

    if num_epochs == 0:
        return np.zeros((0,), dtype=np.float64)
    if overlap < 0.0 or overlap >= 1.0:
        raise ValueError("overlap must be in [0.0, 1.0)")
    step = window_sec * (1.0 - overlap)
    centers = origin + window_sec / 2.0 + step * np.arange(num_epochs)
    return centers.astype(np.float64)


def build_knn_indices(x: np.ndarray, k: int = 20, metric: str = "euclidean", whiten: bool = True) -> np.ndarray:
    """Return ``[T, k]`` array of nearest-neighbour indices."""

    if metric != "euclidean":
        logger.warning("Only euclidean metric supported; received %s", metric)

    if x.size == 0:
        return np.zeros((0, k), dtype=np.int32)
    T = int(x.shape[0])
    finite_rows = np.isfinite(x).all(axis=1)
    if not np.all(finite_rows):
        n_bad = int((~finite_rows).sum())
        raise ValueError(
            f"build_knn_indices requires finite x; found {n_bad}/{T} rows with non-finite values"
        )
    if T < 2:
        return np.zeros((T, 0), dtype=np.int32)

    k = min(k, max(1, T - 1))

    xx = x
    if whiten:
        mu = np.mean(x, axis=0, keepdims=True)
        sigma = np.std(x, axis=0, keepdims=True)
        sigma[sigma == 0] = 1.0
        xx = (x - mu) / sigma

    if cKDTree is None:
        logger.warning("scipy.spatial.cKDTree unavailable; using blockwise O(T^2) kNN fallback")
        xx = np.asarray(xx, dtype=np.float32)
        norms = np.sum(xx * xx, axis=1, dtype=np.float32)
        out = np.empty((T, k), dtype=np.int32)

        # Keep temporary distance matrix near ~128MB: chunk_size * T * 8 bytes.
        target_bytes = 128 * 1024 * 1024
        chunk_size = max(1, int(target_bytes // (max(1, T) * 8)))

        for start in range(0, T, chunk_size):
            stop = min(T, start + chunk_size)
            chunk = xx[start:stop]
            d2 = norms[start:stop, None] + norms[None, :] - np.float32(2.0) * (chunk @ xx.T)
            np.maximum(d2, 0.0, out=d2)
            row_idx = np.arange(start, stop)
            d2[np.arange(stop - start), row_idx] = np.inf
            part = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
            part_d = np.take_along_axis(d2, part, axis=1)
            order = np.argsort(part_d, axis=1)
            out[start:stop] = np.take_along_axis(part, order, axis=1).astype(np.int32)
        return out

    tree = cKDTree(xx)
    _, idx = tree.query(xx, k=k + 1, workers=-1)
    idx = np.asarray(idx, dtype=np.int64)
    cand = idx[:, 1 : k + 1].astype(np.int32, copy=False)
    # Fast path for the common case where query returns k non-self neighbours directly.
    bad_rows = np.any(cand == np.arange(T, dtype=np.int32)[:, None], axis=1)
    if not np.any(bad_rows):
        return cand

    indices = cand.copy()
    for i in np.flatnonzero(bad_rows):
        row = idx[i]
        row = row[row != i]
        if row.size < k:
            extras = np.setdiff1d(np.arange(T, dtype=np.int64), np.concatenate(([i], row)), assume_unique=False)
            row = np.concatenate([row, extras[: max(0, k - row.size)]])
        indices[i] = row[:k].astype(np.int32)
    return indices



