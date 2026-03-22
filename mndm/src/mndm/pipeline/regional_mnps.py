"""Regional MNPS/MNJ estimation from precomputed trajectories only."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from .. import jacobian, projection

logger = logging.getLogger(__name__)


@dataclass
class RegionalMNPSResult:
    """Result container for a single network's MNPS/MNJ computation."""

    network: str
    mnps: np.ndarray  # [T, 3] coordinates (m, d, e)
    mnps_dot: np.ndarray  # [T, 3] derivatives
    jacobian: Optional[np.ndarray] = None  # [T, 3, 3] local Jacobians
    stratified: Optional[np.ndarray] = None  # [T, 9] stratified sub-coords
    jacobian_diagnostics: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    n_timepoints: int = 0
    valid: bool = True
    drop_reason: Optional[str] = None


@dataclass
class RegionalMNPSSummary:
    """Summary of regional MNPS/MNJ for all networks in a segment."""

    subject: str
    session: Optional[str]
    condition: Optional[str]
    task: Optional[str]
    results: Dict[str, RegionalMNPSResult] = field(default_factory=dict)
    n_networks: int = 0
    n_dropped: int = 0


def _as_mnps_array(mnps_trajectory: np.ndarray) -> np.ndarray:
    """Validate/normalize precomputed MNPS trajectory to [T,3]."""
    arr = np.asarray(mnps_trajectory, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("Expected precomputed MNPS trajectory with shape [T, >=3]")
    out = arr[:, :3].copy()
    # Fill sparse NaNs robustly before derivatives/Jacobian.
    for col in range(out.shape[1]):
        vals = out[:, col]
        bad = ~np.isfinite(vals)
        if bad.any():
            finite = vals[~bad]
            fill_val = float(np.nanmedian(finite)) if finite.size else 0.0
            vals[bad] = fill_val
            out[:, col] = vals
    return out


def compute_jacobian_metrics(j_hat: np.ndarray) -> Dict[str, float]:
    """Compute summary metrics from a Jacobian field.

    Args:
        j_hat: Jacobian stack with shape ``(T, 3, 3)``.

    Returns:
        Dict with ``trace_mean``, ``frobenius_mean``, ``rotation_norm_mean``,
        ``anisotropy_mean``, etc.
    """
    if j_hat.size == 0 or j_hat.shape[0] == 0:
        return {
            "trace_mean": np.nan,
            "trace_std": np.nan,
            "frobenius_mean": np.nan,
            "rotation_norm_mean": np.nan,
            "anisotropy_mean": np.nan,
        }

    n_windows = j_hat.shape[0]
    traces = []
    frobenius_norms = []
    rotation_norms = []
    anisotropies = []

    for t in range(n_windows):
        J = j_hat[t]

        # Trace (sum of diagonal elements)
        tr = np.trace(J)
        traces.append(tr)

        # Frobenius norm
        fro = np.linalg.norm(J, ord="fro")
        frobenius_norms.append(fro)

        # Rotation norm (antisymmetric part)
        J_antisym = 0.5 * (J - J.T)
        rot_norm = np.linalg.norm(J_antisym, ord="fro")
        rotation_norms.append(rot_norm)

        # Anisotropy (condition-number style): ratio of max to min singular value.
        #
        # Important: for near-rank-deficient Jacobians (common in very smooth /
        # stationary segments), the smallest singular value can hit 0 in float32.
        # In that case the per-window anisotropy is undefined; we store NaN and
        # later fall back to 1.0 *only if all windows are undefined*.
        try:
            svd = np.linalg.svd(J, compute_uv=False)
            s0 = float(svd[0]) if svd.size else np.nan
            smin = float(svd[-1]) if svd.size else np.nan
            if not np.isfinite(s0) or not np.isfinite(smin):
                aniso = np.nan
            elif smin > 1e-10:
                aniso = float(np.clip(s0 / smin, 1.0, 1e6))
            else:
                aniso = np.nan
        except np.linalg.LinAlgError:
            aniso = np.nan
        anisotropies.append(aniso)

    def _safe_nanmean(arr: list) -> float:
        """Compute nanmean without warning on empty arrays."""
        arr_np = np.array(arr)
        if arr_np.size == 0 or np.all(np.isnan(arr_np)):
            return np.nan
        return float(np.nanmean(arr_np))

    def _safe_nanstd(arr: list) -> float:
        """Compute nanstd without warning on empty arrays."""
        arr_np = np.array(arr)
        if arr_np.size == 0 or np.all(np.isnan(arr_np)):
            return np.nan
        return float(np.nanstd(arr_np))

    return {
        "trace_mean": _safe_nanmean(traces),
        "trace_std": _safe_nanstd(traces),
        "frobenius_mean": _safe_nanmean(frobenius_norms),
        "rotation_norm_mean": _safe_nanmean(rotation_norms),
        # If *all* windows are undefined (e.g. all svd[-1] ~ 0), return 1.0 as a
        # stable "no anisotropy signal" sentinel instead of propagating all-NaN.
        "anisotropy_mean": (lambda v: 1.0 if np.isnan(v) else v)(_safe_nanmean(anisotropies)),
    }


def compute_mnps_metrics(mnps: np.ndarray) -> Dict[str, float]:
    """Compute summary metrics from MNPS coordinates.

    Args:
        mnps: MNPS array with shape ``(T, 3)`` for axes ``(m, d, e)``.

    Returns:
        Per-axis mean, std, median, IQR, MAD keys (``m_mean``, ``d_*``, ``e_*``).
    """
    if mnps.size == 0 or mnps.shape[0] == 0:
        return {
            "m_mean": np.nan, "m_std": np.nan, "m_median": np.nan, "m_iqr": np.nan, "m_mad": np.nan,
            "d_mean": np.nan, "d_std": np.nan, "d_median": np.nan, "d_iqr": np.nan, "d_mad": np.nan,
            "e_mean": np.nan, "e_std": np.nan, "e_median": np.nan, "e_iqr": np.nan, "e_mad": np.nan,
        }

    def _axis_metrics(values: np.ndarray, prefix: str) -> Dict[str, float]:
        """Internal helper: axis metrics."""
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return {
                f"{prefix}_mean": np.nan,
                f"{prefix}_std": np.nan,
                f"{prefix}_median": np.nan,
                f"{prefix}_iqr": np.nan,
                f"{prefix}_mad": np.nan,
            }
        median = float(np.nanmedian(finite))
        q75, q25 = np.nanpercentile(finite, [75, 25])
        mad = float(np.nanmedian(np.abs(finite - median)))
        return {
            f"{prefix}_mean": float(np.nanmean(finite)),
            f"{prefix}_std": float(np.nanstd(finite)),
            f"{prefix}_median": median,
            f"{prefix}_iqr": float(q75 - q25),
            f"{prefix}_mad": mad,
        }

    metrics: Dict[str, float] = {}
    metrics.update(_axis_metrics(mnps[:, 0], "m"))
    metrics.update(_axis_metrics(mnps[:, 1], "d"))
    metrics.update(_axis_metrics(mnps[:, 2], "e"))
    return metrics


def compute_stratified_metrics(
    mnps: np.ndarray,
    subcoords: Mapping[str, Mapping[str, float]],
    *,
    prefix: str = "strat",
) -> Dict[str, float]:
    """Compute summary metrics for stratified subcoordinates over (m, d, e).

    The subcoords mapping defines linear combinations of the MNPS axes:
        name: {m: w1, d: w2, e: w3}

    If no m/d/e weights are provided for a subcoord, fall back to the
    prefix in the subcoord name (m_*, d_*, e_*) to choose the axis.
    """
    if mnps.size == 0 or mnps.shape[0] == 0:
        return {}

    axis_map = {"m": 0, "d": 1, "e": 2}

    def _sanitize(name: str) -> str:
        """Internal helper: sanitize."""
        out = str(name).strip().replace(" ", "_").replace("-", "_")
        return "".join(ch for ch in out if ch.isalnum() or ch == "_")

    def _axis_metrics(values: np.ndarray, base: str) -> Dict[str, float]:
        """Internal helper: axis metrics."""
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return {
                f"{base}_mean": np.nan,
                f"{base}_std": np.nan,
                f"{base}_median": np.nan,
                f"{base}_iqr": np.nan,
                f"{base}_mad": np.nan,
            }
        median = float(np.nanmedian(finite))
        q75, q25 = np.nanpercentile(finite, [75, 25])
        mad = float(np.nanmedian(np.abs(finite - median)))
        return {
            f"{base}_mean": float(np.nanmean(finite)),
            f"{base}_std": float(np.nanstd(finite)),
            f"{base}_median": median,
            f"{base}_iqr": float(q75 - q25),
            f"{base}_mad": mad,
        }

    metrics: Dict[str, float] = {}
    for name, weights in (subcoords or {}).items():
        axis_weights: Dict[str, float] = {}
        if isinstance(weights, Mapping):
            for axis_name, w in weights.items():
                axis_key = str(axis_name).lower().strip()
                if axis_key not in axis_map:
                    continue
                try:
                    axis_weights[axis_key] = float(w)
                except Exception:
                    continue

        if not axis_weights:
            name_key = str(name).lower().strip()
            if name_key.startswith("m"):
                axis_weights = {"m": 1.0}
            elif name_key.startswith("d"):
                axis_weights = {"d": 1.0}
            elif name_key.startswith("e"):
                axis_weights = {"e": 1.0}

        if not axis_weights:
            continue

        vec = np.zeros((mnps.shape[0],), dtype=np.float32)
        for axis_key, weight in axis_weights.items():
            vec += weight * mnps[:, axis_map[axis_key]].astype(np.float32, copy=False)
        clean_name = _sanitize(name)
        base = f"{prefix}_{clean_name}"
        metrics.update(_axis_metrics(vec, base))
    return metrics


def _build_stratified_trajectory(
    mnps: np.ndarray,
    subcoords: Mapping[str, Mapping[str, float]],
    ordered_names: List[str],
) -> np.ndarray:
    """Build a T x K stratified trajectory from m/d/e MNPS coordinates."""
    axis_map = {"m": 0, "d": 1, "e": 2}
    rows: List[np.ndarray] = []
    for name in ordered_names:
        weights = subcoords.get(name, {}) if isinstance(subcoords, Mapping) else {}
        axis_weights: Dict[str, float] = {}
        if isinstance(weights, Mapping):
            for axis_name, w in weights.items():
                axis_key = str(axis_name).lower().strip()
                if axis_key in axis_map:
                    try:
                        axis_weights[axis_key] = float(w)
                    except Exception:
                        continue
        if not axis_weights:
            # Fall back to subcoord prefix when explicit m/d/e weights are missing.
            key = str(name).lower()
            if key.startswith("m"):
                axis_weights = {"m": 1.0}
            elif key.startswith("d"):
                axis_weights = {"d": 1.0}
            elif key.startswith("e"):
                axis_weights = {"e": 1.0}

        vec = np.zeros((mnps.shape[0],), dtype=np.float64)
        if axis_weights:
            for axis_key, w in axis_weights.items():
                vec += w * mnps[:, axis_map[axis_key]].astype(np.float64, copy=False)
        else:
            vec[:] = np.nan
        rows.append(vec)

    if not rows:
        return np.zeros((mnps.shape[0], 0), dtype=np.float64)
    return np.stack(rows, axis=1)


def _compute_9d_conditioning_diagnostics(x_9d: np.ndarray) -> Dict[str, float]:
    """Compute condition number and PCA variance diagnostics for T x 9 matrix."""
    diag: Dict[str, float] = {
        "strat9_condition_number": np.nan,
        "strat9_pc1_var": np.nan,
        "strat9_pc2_var": np.nan,
        "strat9_pc3_var": np.nan,
        "strat9_pc1_3_cumulative_var": np.nan,
        "strat9_falsified": 0.0,
    }
    if x_9d.ndim != 2 or x_9d.shape[0] < 3 or x_9d.shape[1] < 3:
        return diag

    x = np.asarray(x_9d, dtype=np.float64)
    # Standardize columns (z-score). NaNs are median-filled first.
    for c in range(x.shape[1]):
        col = x[:, c]
        bad = ~np.isfinite(col)
        if bad.any():
            finite = col[~bad]
            col[bad] = np.nanmedian(finite) if finite.size else 0.0
            x[:, c] = col
    mu = np.mean(x, axis=0, keepdims=True)
    sigma = np.std(x, axis=0, keepdims=True)
    sigma[sigma < 1e-12] = 1.0
    z = (x - mu) / sigma

    try:
        svals = np.linalg.svd(z, compute_uv=False)
    except np.linalg.LinAlgError:
        return diag
    if svals.size == 0:
        return diag

    smax = float(svals[0])
    smin = float(svals[-1])
    cond = float(np.inf) if smin <= 1e-12 else float(smax / smin)
    var = svals**2
    total = float(np.sum(var))
    if total <= 0:
        return diag
    ratios = var / total
    pc1 = float(ratios[0]) if ratios.size > 0 else np.nan
    pc2 = float(ratios[1]) if ratios.size > 1 else np.nan
    pc3 = float(ratios[2]) if ratios.size > 2 else np.nan
    cum3 = float(np.sum(ratios[:3]))

    diag.update(
        {
            "strat9_condition_number": cond,
            "strat9_pc1_var": pc1,
            "strat9_pc2_var": pc2,
            "strat9_pc3_var": pc3,
            "strat9_pc1_3_cumulative_var": cum3,
            "strat9_falsified": 1.0 if cum3 > 0.90 else 0.0,
        }
    )
    return diag


def compute_regional_mnps_for_network(
    network_label: str,
    mnps_trajectory: np.ndarray,
    config: Mapping[str, Any],
    min_length: int = 3,
    stratified_trajectory: Optional[np.ndarray] = None,
) -> RegionalMNPSResult:
    """Compute MNPS/MNJ for one network from precomputed trajectory only."""
    result = RegionalMNPSResult(network=network_label, mnps=np.array([]), mnps_dot=np.array([]))
    try:
        mnps = _as_mnps_array(mnps_trajectory)
    except ValueError as exc:
        result.valid = False
        result.drop_reason = str(exc)
        return result

    if mnps.shape[0] < min_length:
        result.valid = False
        result.drop_reason = f"too few MNPS samples ({mnps.shape[0]} < {min_length})"
        logger.debug("Dropping %s: %s", network_label, result.drop_reason)
        return result

    mnps_cfg = config.get("mnps", {}) if isinstance(config, Mapping) else {}
    dt = float(mnps_cfg.get("time_step_sec", mnps_cfg.get("dt_sec", 1.0)))
    if not np.isfinite(dt) or dt <= 0:
        dt = 1.0

    result.mnps = mnps
    result.n_timepoints = mnps.shape[0]

    deriv_cfg = mnps_cfg.get("derivative", {})
    method = deriv_cfg.get("method", "sav_gol")
    window = int(deriv_cfg.get("window", 7))
    polyorder = int(deriv_cfg.get("polyorder", 3))
    result.mnps_dot = projection.estimate_derivatives(mnps, dt, method, window, polyorder)

    jac_cfg = config.get("jacobian", {}) if isinstance(config, Mapping) else {}
    if jac_cfg.get("enabled", True):
        knn_cfg = mnps_cfg.get("knn", {})
        k = int(knn_cfg.get("k", 10))
        metric = knn_cfg.get("metric", "euclidean")

        nn_indices = projection.build_knn_indices(mnps, k=k, metric=metric, whiten=True)

        super_window = int(jac_cfg.get("super_window", 3))
        ridge_alpha = float(jac_cfg.get("ridge_alpha", 1.0))
        dist_weighted = bool(jac_cfg.get("distance_weighted", True))

        jac_result = jacobian.estimate_local_jacobians(
            mnps,
            result.mnps_dot,
            nn_indices,
            super_window=super_window,
            ridge_alpha=ridge_alpha,
            distance_weighted=dist_weighted,
            j_dot_dt=float(dt),
        )

        result.jacobian = jac_result.j_hat
        result.jacobian_diagnostics = jac_result.diagnostics

        result.metrics.update(compute_jacobian_metrics(jac_result.j_hat))

    result.metrics.update(compute_mnps_metrics(mnps))

    strat_cfg = config.get("stratified", {}) if isinstance(config, Mapping) else {}
    if isinstance(strat_cfg, Mapping) and bool(strat_cfg.get("enabled", False)):
        subcoords = (
            strat_cfg.get("subcoords", {})
            if isinstance(strat_cfg.get("subcoords", {}), Mapping)
            else {}
        )
        result.metrics.update(compute_stratified_metrics(mnps, subcoords, prefix="strat"))

        ordered_subcoords = [
            "m_a",
            "m_e",
            "m_o",
            "d_n",
            "d_l",
            "d_s",
            "e_e",
            "e_s",
            "e_m",
        ]
        if stratified_trajectory is not None:
            strat_traj = np.asarray(stratified_trajectory, dtype=np.float64)
        else:
            # Do not silently fabricate 9D from 3D fallback when upstream plumbing
            # did not provide explicit stratified trajectories.
            logger.warning(
                "Stratified enabled but network_stratified not provided for %s. "
                "Skipping 9D diagnostics.",
                network_label,
            )
            strat_traj = np.zeros((mnps.shape[0], 0), dtype=np.float64)
        if strat_traj.shape[1] == 9:
            # Persist the stratified trajectory so it can be written to H5.
            result.stratified = strat_traj.astype(np.float32)
            diag = _compute_9d_conditioning_diagnostics(strat_traj)
            result.metrics.update(diag)
            logger.info(
                "9D conditioning [%s]: kappa=%.4g, PC1-3 cumulative variance=%.4f (PC1=%.4f, PC2=%.4f, PC3=%.4f)",
                network_label,
                diag.get("strat9_condition_number", float("nan")),
                diag.get("strat9_pc1_3_cumulative_var", float("nan")),
                diag.get("strat9_pc1_var", float("nan")),
                diag.get("strat9_pc2_var", float("nan")),
                diag.get("strat9_pc3_var", float("nan")),
            )
            if float(diag.get("strat9_pc1_3_cumulative_var", np.nan)) > 0.90:
                logger.critical(
                    "CRITICAL WARNING: 9D MNPS falsified for this session. Matrix is effectively 3-dimensional."
                )

    return result


def compute_all_regional_mnps(
    group_ts: Optional[Dict[str, np.ndarray]],
    sfreq: Optional[float],
    config: Mapping[str, Any],
    subject: str,
    session: Optional[str] = None,
    condition: Optional[str] = None,
    task: Optional[str] = None,
    *,
    network_mnps: Optional[Mapping[str, np.ndarray]] = None,
    network_stratified: Optional[Mapping[str, np.ndarray]] = None,
) -> RegionalMNPSSummary:
    """Compute regional MNPS/MNJ from precomputed network trajectories only."""
    summary = RegionalMNPSSummary(
        subject=subject,
        session=session,
        condition=condition,
        task=task,
    )

    if not network_mnps:
        logger.warning(
            "Regional MNPS skipped for %s: no precomputed network_mnps provided "
            "(raw group_ts/sfreq input is no longer supported).",
            subject,
        )
        return summary

    min_length = int(config.get("min_segment_length_tr", 3))
    enabled_networks = config.get("networks", [])

    for network_label, mnps in network_mnps.items():
        if enabled_networks and network_label not in enabled_networks:
            continue

        result = compute_regional_mnps_for_network(
            network_label=network_label,
            mnps_trajectory=np.asarray(mnps, dtype=np.float32),
            config=config,
            min_length=min_length,
            stratified_trajectory=(
                np.asarray(network_stratified.get(network_label), dtype=np.float64)
                if isinstance(network_stratified, Mapping) and network_label in network_stratified
                else None
            ),
        )

        if result.valid:
            summary.results[network_label] = result
            summary.n_networks += 1
        else:
            summary.n_dropped += 1
            logger.debug(
                "Dropped %s for %s/%s/%s: %s",
                network_label, subject, condition, task, result.drop_reason
            )

    logger.info(
        "Regional MNPS for %s/%s/%s: %d networks computed, %d dropped",
        subject, condition or "-", task or "-",
        summary.n_networks, summary.n_dropped,
    )

    return summary


def summary_to_dataframe_rows(summary: RegionalMNPSSummary) -> List[Dict[str, Any]]:
    """Convert a RegionalMNPSSummary to a list of row dictionaries for CSV output.

    Each row represents one (subject, session, condition, task, network) combination.
    """
    rows = []
    for network_label, result in summary.results.items():
        row = {
            "subject_id": summary.subject,
            "session_id": summary.session,
            "condition_label": summary.condition,
            "task_label": summary.task,
            "network_label": network_label,
            "n_timepoints": result.n_timepoints,
        }
        # Add all metrics
        row.update(result.metrics)
        rows.append(row)
    return rows


def write_regional_mnps_csv(
    summaries: List[RegionalMNPSSummary],
    output_path: Path,
    append: bool = False,
) -> None:
    """Write regional MNPS summaries to a CSV file.

    Args:
        summaries: One or more :class:`RegionalMNPSSummary` instances.
        output_path: Destination CSV path.
        append: If True, append without rewriting header when file exists.
    """
    all_rows = []
    for summary in summaries:
        all_rows.extend(summary_to_dataframe_rows(summary))

    if not all_rows:
        logger.warning("No regional MNPS data to write to %s", output_path)
        return

    df = pd.DataFrame(all_rows)

    # Ensure consistent column order
    column_order = [
        "subject_id", "session_id", "condition_label", "task_label", "network_label",
        "n_timepoints",
        "m_mean", "m_std", "m_median", "m_iqr", "m_mad",
        "d_mean", "d_std", "d_median", "d_iqr", "d_mad",
        "e_mean", "e_std", "e_median", "e_iqr", "e_mad",
        "trace_mean", "trace_std", "frobenius_mean", "rotation_norm_mean", "anisotropy_mean",
    ]
    # Only include columns that exist
    columns = [c for c in column_order if c in df.columns]
    # Add any extra columns not in the standard order
    extra_cols = [c for c in df.columns if c not in columns]
    # Keep stratified columns grouped for readability
    strat_cols = sorted([c for c in extra_cols if c.startswith("strat_")])
    extra_cols = [c for c in extra_cols if c not in strat_cols]
    columns.extend(strat_cols)
    columns.extend(extra_cols)
    df = df[columns]

    mode = "a" if append and output_path.exists() else "w"
    header = not (append and output_path.exists())
    df.to_csv(output_path, mode=mode, header=header, index=False)
    logger.info("Wrote %d regional MNPS rows to %s", len(df), output_path)


# NDT canonical block index mapping over the 9 stratified sub-coordinates.
# Index order: [m_a, m_e, m_o, d_n, d_l, d_s, e_e, e_s, e_m]
_BLOCK_INDICES: Dict[str, List[int]] = {
    "m": [0, 1, 2],  # m_a, m_e, m_o  — Macrostates
    "d": [3, 4, 5],  # d_n, d_l, d_s  — Dynamics / Dispersion
    "e": [6, 7, 8],  # e_e, e_s, e_m  — Entropy / Energy / Embodiment
}


def _iter_block_pairs(
    block_names: List[str],
    pairs_cfg: Any,
    include_self: bool,
) -> List[tuple]:
    """Expand pairs_cfg to a list of (source, target) block name tuples."""
    legacy_aliases = {"M": "m", "D": "d", "E": "e", "m": "m", "d": "d", "e": "e"}
    if pairs_cfg == "all":
        return [
            (out_g, in_g)
            for out_g in block_names
            for in_g in block_names
            if include_self or out_g != in_g
        ]
    if isinstance(pairs_cfg, (list, tuple)):
        out_pairs: List[tuple] = []
        for item in pairs_cfg:
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                continue
            src_raw, tgt_raw = str(item[0]), str(item[1])
            src = legacy_aliases.get(src_raw, legacy_aliases.get(src_raw.upper(), src_raw.lower()))
            tgt = legacy_aliases.get(tgt_raw, legacy_aliases.get(tgt_raw.upper(), tgt_raw.lower()))
            if src not in _BLOCK_INDICES or tgt not in _BLOCK_INDICES:
                continue
            if include_self or src != tgt:
                out_pairs.append((src, tgt))
        return out_pairs
    raise ValueError("block_jacobians.pairs must be 'all' or a list of [source, target]")


def _compute_block_metrics(block_field: np.ndarray, is_diagonal_block: bool) -> Dict[str, float]:
    """Summaries for a block Jacobian field [T, p_out, p_in].

    ``is_diagonal_block`` must be True only when out_group == in_group.
    Trace and anisotropy are geometrically meaningful *only* for diagonal
    (intra-block) Jacobians; for off-diagonal blocks they are physically
    undefined and are returned as NaN.  Frobenius norm is always valid.
    """
    if block_field.size == 0 or block_field.shape[0] == 0:
        return {
            "block_trace_mean": float("nan"),
            "block_frobenius_mean": float("nan"),
            "block_anisotropy_mean": float("nan"),
        }
    frob = np.linalg.norm(block_field, ord="fro", axis=(1, 2))
    metrics: Dict[str, float] = {
        "block_frobenius_mean": float(np.nanmean(frob)) if frob.size else float("nan"),
        "block_trace_mean": float("nan"),
        "block_anisotropy_mean": float("nan"),
    }
    if not is_diagonal_block:
        return metrics
    p_out, p_in = int(block_field.shape[1]), int(block_field.shape[2])
    if p_out != p_in:
        return metrics
    traces = np.trace(block_field, axis1=1, axis2=2)
    metrics["block_trace_mean"] = float(np.nanmean(traces))
    aniso_vals: List[float] = []
    for t in range(block_field.shape[0]):
        try:
            sv = np.linalg.svd(block_field[t], compute_uv=False)
            aniso_vals.append(float(sv[0] / sv[-1]) if sv[-1] > 1e-10 else float("nan"))
        except np.linalg.LinAlgError:
            aniso_vals.append(float("nan"))
    aniso_arr = np.array(aniso_vals)
    valid = aniso_arr[np.isfinite(aniso_arr)]
    metrics["block_anisotropy_mean"] = float(np.mean(valid)) if valid.size else float("nan")
    return metrics


def compute_block_jacobian_rows(
    summary: RegionalMNPSSummary,
    config: Mapping[str, Any],
    include_self: bool = False,
) -> List[Dict[str, Any]]:
    """Compute cross-block Jacobians from per-network stratified ``(T, 9)`` trajectories.

    Args:
        summary: Regional results including per-network ``stratified`` arrays.
        config: Full pipeline config (``modality``, ``mnps``, ``regional_mnps``, …).
        include_self: Whether to include intra-block pairs (``M→M``, …).

    Returns:
        List of row dicts (empty for fMRI or when block Jacobians are disabled).
    """
    # Hard-exit for fMRI: 9D sub-trajectories are rank-deficient per network due
    # to the smooth BOLD signal; cross-block Jacobians have no empirical basis.
    modality = str(config.get("modality", "")).lower() if isinstance(config, Mapping) else ""
    if modality == "fmri":
        logger.debug(
            "Block Jacobians skipped for fMRI (9D rank deficiency; no empirical basis)."
        )
        return []

    regional_cfg: Mapping[str, Any] = (
        config.get("regional_mnps", {}) if isinstance(config, Mapping) else {}
    )
    block_cfg: Mapping[str, Any] = (
        regional_cfg.get("block_jacobians", {}) if isinstance(regional_cfg, Mapping) else {}
    )
    if not bool(block_cfg.get("enabled", False)):
        return []

    pairs_cfg = block_cfg.get("pairs", "all")
    include_self_cfg = bool(block_cfg.get("include_self", include_self))
    include_sym_rot = bool(block_cfg.get("include_sym_rot", True))

    # Derivative parameters (reuse global MNPS settings as defaults)
    mnps_cfg: Mapping[str, Any] = (
        config.get("mnps", {}) if isinstance(config, Mapping) else {}
    )
    dt = float(mnps_cfg.get("time_step_sec", mnps_cfg.get("dt_sec", 1.0)))
    if not np.isfinite(dt) or dt <= 0:
        dt = 1.0
    deriv_cfg: Mapping[str, Any] = (
        mnps_cfg.get("derivative", {}) if isinstance(mnps_cfg, Mapping) else {}
    )
    method = str(deriv_cfg.get("method", "sav_gol"))
    window = int(deriv_cfg.get("window", 7))
    polyorder = int(deriv_cfg.get("polyorder", 3))

    # KNN / ridge params (prefer per-network Jacobian config; fall back to block_jacobians)
    jac_cfg: Mapping[str, Any] = (
        regional_cfg.get("jacobian", {}) if isinstance(regional_cfg, Mapping) else {}
    )
    knn_k = int(
        (jac_cfg.get("knn") or block_cfg.get("knn") or {}).get("k", 10)
    )
    ridge_alpha = float(jac_cfg.get("ridge_alpha", block_cfg.get("ridge_alpha", 1.0)))
    super_window = int(jac_cfg.get("super_window", block_cfg.get("super_window", 3)))

    block_names = list(_BLOCK_INDICES.keys())
    try:
        pairs = _iter_block_pairs(block_names, pairs_cfg, include_self=include_self_cfg)
    except ValueError as exc:
        logger.warning("Block Jacobian pair expansion failed: %s", exc)
        return []

    rows: List[Dict[str, Any]] = []

    for network_label, result in summary.results.items():
        strat = result.stratified
        if strat is None or strat.ndim != 2 or strat.shape[1] != 9:
            logger.debug(
                "Skipping block Jacobians for %s: stratified unavailable (shape=%s)",
                network_label,
                None if strat is None else strat.shape,
            )
            continue

        strat_f = np.asarray(strat, dtype=np.float32)
        n_t = int(strat_f.shape[0])

        for src_name, tgt_name in pairs:
            src_idxs = _BLOCK_INDICES[src_name]
            tgt_idxs = _BLOCK_INDICES[tgt_name]

            X = strat_f[:, src_idxs]  # [T, 3] — source position in block phase space
            Y = strat_f[:, tgt_idxs]  # [T, 3] — target whose dynamics we model

            Y_dot = projection.estimate_derivatives(
                Y.astype(np.float64), dt, method, window, polyorder
            ).astype(np.float32)

            try:
                nn_indices = projection.build_knn_indices(X, k=knn_k, metric="euclidean", whiten=True)
            except Exception as exc:
                logger.debug(
                    "KNN failed for block %s→%s @ %s: %s", src_name, tgt_name, network_label, exc
                )
                continue

            try:
                jac_result = jacobian.estimate_local_jacobians(
                    X,
                    Y_dot,
                    nn_indices,
                    super_window=super_window,
                    ridge_alpha=ridge_alpha,
                    distance_weighted=False,
                    j_dot_dt=float(dt),
                )
            except Exception as exc:
                logger.debug(
                    "Jacobian failed for block %s→%s @ %s: %s",
                    src_name, tgt_name, network_label, exc,
                )
                continue

            metrics = _compute_block_metrics(jac_result.j_hat, is_diagonal_block=(src_name == tgt_name))

            # Symmetric / antisymmetric coupling: only meaningful for cross-block pairs
            c_sym_mean = float("nan")
            c_rot_mean = float("nan")
            if include_sym_rot and src_name != tgt_name:
                X_rev = strat_f[:, tgt_idxs]
                Y_rev_dot = projection.estimate_derivatives(
                    strat_f[:, src_idxs].astype(np.float64), dt, method, window, polyorder
                ).astype(np.float32)
                try:
                    nn_rev = projection.build_knn_indices(
                        X_rev, k=knn_k, metric="euclidean", whiten=True
                    )
                    jac_rev = jacobian.estimate_local_jacobians(
                        X_rev, Y_rev_dot, nn_rev,
                        super_window=super_window, ridge_alpha=ridge_alpha,
                        distance_weighted=False, j_dot_dt=float(dt),
                    )
                    T_min = min(jac_result.j_hat.shape[0], jac_rev.j_hat.shape[0])
                    sym_vals: List[float] = []
                    rot_vals: List[float] = []
                    for t_idx in range(T_min):
                        J_fwd = jac_result.j_hat[t_idx]
                        J_rev = jac_rev.j_hat[t_idx]
                        sym_block = 0.5 * (J_fwd + J_rev.T)
                        rot_block = 0.5 * (J_fwd - J_rev.T)
                        sym_vals.append(float(np.linalg.norm(sym_block, ord="fro")))
                        rot_vals.append(float(np.linalg.norm(rot_block, ord="fro")))
                    if sym_vals:
                        c_sym_mean = float(np.nanmean(sym_vals))
                        c_rot_mean = float(np.nanmean(rot_vals))
                except Exception as exc:
                    logger.debug(
                        "Sym/rot coupling failed for %s↔%s @ %s: %s",
                        src_name, tgt_name, network_label, exc,
                    )

            row: Dict[str, Any] = {
                "subject_id": summary.subject,
                "session_id": summary.session,
                "condition_label": summary.condition,
                "task_label": summary.task,
                "network_label": network_label,
                "source_block": src_name,
                "target_block": tgt_name,
                "n_timepoints": n_t,
                "c_sym_mean": c_sym_mean,
                "c_rot_mean": c_rot_mean,
            }
            row.update(metrics)
            rows.append(row)

        if rows:
            logger.info(
                "Block Jacobians for %s/%s: %d rows for %d networks",
                summary.subject, network_label, len(rows), len(summary.results),
            )

    return rows


def write_block_jacobians_csv(
    summaries: List[RegionalMNPSSummary],
    config: Mapping[str, Any],
    output_path: Path,
    append: bool = False,
    include_self: bool = False,
) -> None:
    """Compute and write cross-block Jacobian rows to CSV (EEG-oriented).

    Args:
        summaries: Regional MNPS summaries from one or more segments.
        config: Full pipeline config including ``regional_mnps.block_jacobians``.
        output_path: Output CSV path.
        append: Append to an existing CSV when True.
        include_self: Default for intra-block pairs (config may override).
    """
    all_rows: List[Dict[str, Any]] = []
    for summary in summaries:
        all_rows.extend(compute_block_jacobian_rows(summary, config, include_self=include_self))

    if not all_rows:
        return

    df = pd.DataFrame(all_rows)

    column_order = [
        "subject_id", "session_id", "condition_label", "task_label",
        "network_label", "source_block", "target_block", "n_timepoints",
        "block_trace_mean", "block_frobenius_mean", "block_anisotropy_mean",
        "c_sym_mean", "c_rot_mean",
    ]
    columns = [c for c in column_order if c in df.columns]
    extra_cols = [c for c in df.columns if c not in columns]
    columns.extend(extra_cols)
    df = df[columns]

    mode = "a" if append and output_path.exists() else "w"
    header = not (append and output_path.exists())
    df.to_csv(output_path, mode=mode, header=header, index=False)
    logger.info("Wrote %d block Jacobian rows to %s", len(df), output_path)

