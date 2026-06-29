"""Block-native window sidecar export.

Assembles a flat, auditable analysis table from:

* ``Sequence[StageBlockInterval]`` — inferred block intervals for one subject/run
* ``BlockWindowProfileConfig``     — window profile (sliding / tail / post_offset / partitioned)
* ``MNPSPayload`` (optional)       — MNPS tensors for coordinate lookup
* subject/run metadata             — identifiers for joining back to HDF5 outputs

Output rows
-----------
One row per block-native window.

Mandatory columns (always present)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``subject_id``, ``session_id``, ``run_id``, ``dataset_id``,
``block_id``, ``window_id_within_block``, ``stage_code``,
``block_start_sec``, ``block_end_sec``, ``block_duration_sec``,
``window_start_sec``, ``window_end_sec``, ``window_center_sec``,
``relative_time_in_block_sec``, ``distance_to_block_end_sec``,
``relative_pos_0_1``, ``partition_label``, ``is_post_offset``.

Optional MNPS columns (when payload provided)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``m``, ``d``, ``e``, ``m_dot``, ``d_dot``, ``e_dot``, ``mnps_finite``.

Design contract
---------------
* No statistics. No hypothesis tests. No plotting.
* Sidecar output only — does not modify the canonical HDF5 contract.
* Parquet-first; CSV is opt-in.
* Deterministic for a fixed input.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .block_native_config import BlockWindowProfileConfig
from .block_windows import BlockWindowSpec, generate_block_windows

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spec conversion
# ---------------------------------------------------------------------------

def block_window_spec_from_profile(profile: BlockWindowProfileConfig) -> BlockWindowSpec:
    """Convert a :class:`BlockWindowProfileConfig` to a :class:`BlockWindowSpec`."""
    return BlockWindowSpec(
        kind=profile.kind,
        window_length_sec=profile.window_length_sec,
        step_sec=profile.step_sec,
        emit_relative_position=profile.emit_relative_position,
        tail_sec=profile.tail_sec,
        post_offset_bins=profile.post_offset_bins,
        partitions=profile.partitions,
        min_windows_per_block=profile.min_windows_per_block,
        min_block_sec=profile.min_block_sec,
    )


# ---------------------------------------------------------------------------
# MNPS coordinate helpers
# ---------------------------------------------------------------------------

def _find_closest_window_idx(payload: Any, win_start: float, win_end: float) -> int:
    """Return the payload time-axis index closest to the window center.

    Returns ``-1`` when no valid match is found within one window length.

    Notes
    -----
    Uses ``np.nanargmin`` rather than ``np.argmin`` so that NaN-padded time
    arrays (produced when the session grid is longer than the run) do not
    corrupt the result.  In NumPy ≥ 2.x, ``np.argmin`` on an array containing
    NaN values returns the *first* NaN index rather than the index of the
    finite minimum, which caused all block-native windows to map to the same
    constant ``source_window_index`` equal to the first NaN position.
    """
    if payload is None:
        return -1
    time = getattr(payload, "time", None)
    if time is None or len(time) == 0:
        return -1
    win_center = (win_start + win_end) * 0.5
    t = np.asarray(time, dtype=float)
    diffs = np.abs(t - win_center)
    if not np.any(np.isfinite(diffs)):
        return -1
    idx = int(np.nanargmin(diffs))
    win_len = win_end - win_start
    if not np.isfinite(diffs[idx]) or diffs[idx] > win_len:
        return -1
    return idx


def _mnps_at(payload: Any, w_idx: int) -> Dict[str, Any]:
    """Extract MNPS 3-D coordinates and derivatives at window index."""
    out: Dict[str, Any] = {}
    x = getattr(payload, "x", None)
    xd = getattr(payload, "x_dot", None)
    if x is not None and w_idx < len(x):
        row = x[w_idx]
        out["m"] = float(row[0]) if len(row) > 0 else float("nan")
        out["d"] = float(row[1]) if len(row) > 1 else float("nan")
        out["e"] = float(row[2]) if len(row) > 2 else float("nan")
        out["mnps_finite"] = int(np.isfinite([out["m"], out["d"], out["e"]]).all())
    else:
        out.update({"m": float("nan"), "d": float("nan"), "e": float("nan"), "mnps_finite": 0})
    if xd is not None and w_idx < len(xd):
        drow = xd[w_idx]
        out["m_dot"] = float(drow[0]) if len(drow) > 0 else float("nan")
        out["d_dot"] = float(drow[1]) if len(drow) > 1 else float("nan")
        out["e_dot"] = float(drow[2]) if len(drow) > 2 else float("nan")
    else:
        out.update({"m_dot": float("nan"), "d_dot": float("nan"), "e_dot": float("nan")})
    return out


def _mnps_nan() -> Dict[str, Any]:
    return {
        "m": float("nan"), "d": float("nan"), "e": float("nan"), "mnps_finite": 0,
        "m_dot": float("nan"), "d_dot": float("nan"), "e_dot": float("nan"),
    }


def _scalar_value(value: Any) -> Any:
    """Convert numpy scalars/bytes into plain row-friendly Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.decode("utf-8", errors="ignore")
    return value


def _named_group_at(group: Any, w_idx: int, *, suffix: str = "") -> Dict[str, Any]:
    """Extract one row from a ``{"values","names"}`` payload group."""
    if not isinstance(group, dict):
        return {}
    values = group.get("values")
    names = group.get("names")
    if values is None or names is None:
        return {}
    arr = np.asarray(values)
    if arr.ndim != 2 or w_idx < 0 or w_idx >= arr.shape[0]:
        return {}
    out: Dict[str, Any] = {}
    for idx, raw_name in enumerate(list(names)[: arr.shape[1]]):
        out[f"{str(raw_name)}{suffix}"] = _scalar_value(arr[w_idx, idx])
    return out


def _labels_at(payload: Any, w_idx: int) -> Dict[str, Any]:
    """Extract aligned 1-D labels from ``payload.labels``."""
    labels = getattr(payload, "labels", None)
    if not isinstance(labels, dict) or w_idx < 0:
        return {}
    out: Dict[str, Any] = {}
    for key, values in labels.items():
        arr = np.asarray(values)
        if arr.ndim != 1 or w_idx >= arr.shape[0]:
            continue
        out[str(key)] = _scalar_value(arr[w_idx])
    return out


def _anchor_exports_at(payload: Any, w_idx: int) -> Dict[str, Any]:
    """Extract aligned anchor-state, derivative, and quality values for one window."""
    if w_idx < 0:
        return {}
    out: Dict[str, Any] = {}
    out.update(_named_group_at(getattr(payload, "anchor_state", None), w_idx))
    out.update(_named_group_at(getattr(payload, "anchor_state_dot", None), w_idx, suffix="_dot"))
    out.update(_named_group_at(getattr(payload, "anchor_quality", None), w_idx, suffix="_quality"))
    return out


def _raw_feature_exports_at(
    payload: Any,
    w_idx: int,
    *,
    prefixes: Sequence[str] = ("ecg_hrv_", "pupil_"),
    explicit_names: Sequence[str] = ("qc_ok_ecg_hrv",),
) -> Dict[str, Any]:
    """Extract selected raw feature columns aligned to one payload window."""
    if w_idx < 0:
        return {}
    values = getattr(payload, "features_raw_values", None)
    names = getattr(payload, "features_raw_names", None)
    if values is None or names is None:
        return {}
    arr = np.asarray(values)
    if arr.ndim != 2 or w_idx >= arr.shape[0]:
        return {}
    out: Dict[str, Any] = {}
    prefixes_norm = tuple(str(v) for v in prefixes)
    explicit_norm = {str(v) for v in explicit_names}
    for idx, raw_name in enumerate(list(names)[: arr.shape[1]]):
        name = str(raw_name)
        if not (name.startswith(prefixes_norm) or name in explicit_norm):
            continue
        out[name] = _scalar_value(arr[w_idx, idx])
    return out


def _as_frequency_hz(block: Any) -> float:
    """Best-effort conversion of block parameter to frequency_hz."""
    try:
        value = float(getattr(block, "block_parameter", float("nan")))
    except Exception:
        value = float("nan")
    return value if np.isfinite(value) else float("nan")


def _summarize_block_native_qc(
    *,
    blocks: Sequence[Any],
    window_rows: Sequence[Any],
    source_indices: Sequence[int],
) -> Dict[str, Any]:
    """Build compact per-subject block-native QC stats."""
    block_stage_counts = Counter(int(getattr(b, "stage_code", -1)) for b in blocks)
    block_freq_counts = Counter(
        int(round(float(freq)))
        for freq in [_as_frequency_hz(b) for b in blocks]
        if np.isfinite(freq)
    )
    block_end_reason_counts = Counter(str(getattr(b, "end_reason", "") or "") for b in blocks)
    window_stage_counts = Counter(int(getattr(r, "stage_code", -1)) for r in window_rows)

    starts = np.asarray([float(getattr(b, "start_sec", float("nan"))) for b in blocks], dtype=np.float64)
    ends = np.asarray([float(getattr(b, "end_sec", float("nan"))) for b in blocks], dtype=np.float64)
    durations = ends - starts
    durations = durations[np.isfinite(durations)]
    gaps = np.asarray([], dtype=np.float64)
    if starts.size > 1 and ends.size > 1:
        order = np.argsort(starts, kind="stable")
        sorted_starts = starts[order]
        sorted_ends = ends[order]
        gaps = sorted_starts[1:] - sorted_ends[:-1]
        gaps = gaps[np.isfinite(gaps)]

    source_arr = np.asarray(list(source_indices), dtype=np.int64) if source_indices else np.asarray([], dtype=np.int64)
    matched_source = int(np.sum(source_arr >= 0)) if source_arr.size else 0
    unmatched_source = int(np.sum(source_arr < 0)) if source_arr.size else 0

    def _dist(values: np.ndarray) -> Dict[str, Optional[float]]:
        if values.size == 0:
            return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}
        return {
            "count": int(values.size),
            "min": float(np.nanmin(values)),
            "p25": float(np.nanpercentile(values, 25)),
            "median": float(np.nanpercentile(values, 50)),
            "p75": float(np.nanpercentile(values, 75)),
            "max": float(np.nanmax(values)),
        }

    return {
        "schema": "mndm.block_native_qc.v1",
        "n_blocks": int(len(blocks)),
        "n_windows": int(len(window_rows)),
        "block_counts_by_stage": {str(k): int(v) for k, v in sorted(block_stage_counts.items(), key=lambda kv: kv[0])},
        "window_counts_by_stage": {str(k): int(v) for k, v in sorted(window_stage_counts.items(), key=lambda kv: kv[0])},
        "block_counts_by_frequency_hz": {
            str(k): int(v) for k, v in sorted(block_freq_counts.items(), key=lambda kv: kv[0])
        },
        "end_reason_counts": {
            str(k): int(v) for k, v in sorted(block_end_reason_counts.items(), key=lambda kv: kv[0])
        },
        "duration_sec_distribution": _dist(durations),
        "gap_sec_distribution": _dist(gaps),
        "source_window_index": {
            "matched": int(matched_source),
            "unmatched": int(unmatched_source),
            "total": int(source_arr.size),
            "matched_fraction": float(matched_source / source_arr.size) if source_arr.size else 0.0,
        },
        "filters": {
            "min_block_sec": None,
            "min_windows_per_block": None,
        },
    }


# ---------------------------------------------------------------------------
# Trajectory geometry helpers
# ---------------------------------------------------------------------------

def _trajectory_stats_by_block(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Compute per-block manifold trajectory geometry from assembled rows.

    For each block, extracts the ordered sequence of ``(m, d, e)`` coordinates
    from windows where ``mnps_finite == 1``, then computes:

    * ``traj_path_length``      — total Euclidean arc length across windows.
    * ``traj_endpoint_dist``    — straight-line distance from first to last window.
    * ``traj_efficiency``       — ``endpoint_dist / path_length`` (1 = perfectly straight).
    * ``traj_mean_curvature``   — mean turning angle (radians) between consecutive steps.
    * ``traj_max_curvature``    — maximum turning angle (radians) between consecutive steps.
    * ``traj_n_finite_windows`` — number of windows with valid MNPS coordinates.

    Blocks with fewer than two finite windows get NaN for all geometry metrics.
    Blocks with fewer than three finite windows get NaN for curvature metrics.
    All values are replicated to every window in the block.

    Returns
    -------
    Dict[int, Dict[str, Any]]
        Mapping ``block_id → trajectory stats dict``.
    """
    _nan = float("nan")
    _traj_keys = (
        "traj_path_length",
        "traj_endpoint_dist",
        "traj_efficiency",
        "traj_mean_curvature",
        "traj_max_curvature",
        "traj_n_finite_windows",
    )
    _nan_stats: Dict[str, Any] = {k: _nan for k in _traj_keys}
    _nan_stats["traj_n_finite_windows"] = 0

    # Group rows by block_id, preserving window order (rows are already ordered).
    from collections import defaultdict
    block_rows: Dict[int, list] = defaultdict(list)
    for row in rows:
        bid = int(row.get("block_id", -1))
        block_rows[bid].append(row)

    result: Dict[int, Dict[str, Any]] = {}
    for bid, brows in block_rows.items():
        # Extract valid MNPS coords in window order.
        coords = []
        for r in brows:
            if int(r.get("mnps_finite", 0)) == 1:
                m = r.get("m")
                d = r.get("d")
                e = r.get("e")
                if m is not None and d is not None and e is not None:
                    if np.isfinite(m) and np.isfinite(d) and np.isfinite(e):
                        coords.append((float(m), float(d), float(e)))

        n = len(coords)
        stats: Dict[str, Any] = dict(_nan_stats)
        stats["traj_n_finite_windows"] = n

        if n < 2:
            result[bid] = stats
            continue

        pts = np.array(coords, dtype=np.float64)   # (n, 3)
        deltas = np.diff(pts, axis=0)               # (n-1, 3)
        step_lengths = np.linalg.norm(deltas, axis=1)

        path_length = float(np.sum(step_lengths))
        endpoint_dist = float(np.linalg.norm(pts[-1] - pts[0]))
        efficiency = float(endpoint_dist / path_length) if path_length > 1e-12 else _nan

        stats["traj_path_length"] = path_length
        stats["traj_endpoint_dist"] = endpoint_dist
        stats["traj_efficiency"] = efficiency

        if n >= 3:
            # Turning angle between consecutive displacement vectors.
            d1 = deltas[:-1]   # (n-2, 3)
            d2 = deltas[1:]    # (n-2, 3)
            norms1 = np.linalg.norm(d1, axis=1)
            norms2 = np.linalg.norm(d2, axis=1)
            valid = (norms1 > 1e-12) & (norms2 > 1e-12)
            if np.any(valid):
                cos_angles = np.einsum("ij,ij->i", d1[valid], d2[valid]) / (
                    norms1[valid] * norms2[valid]
                )
                cos_angles = np.clip(cos_angles, -1.0, 1.0)
                angles = np.arccos(cos_angles)
                stats["traj_mean_curvature"] = float(np.mean(angles))
                stats["traj_max_curvature"] = float(np.max(angles))

        result[bid] = stats

    return result


# ---------------------------------------------------------------------------
# Inter-network Jacobian coupling helpers
# ---------------------------------------------------------------------------

#: Short abbreviations used in coupling column names.
_NETWORK_ABBREV: Dict[str, str] = {
    "central": "cntr",
    "frontal": "frnt",
    "parietal_occipital": "par",
    "temporal": "temp",
}


def _abbrev_network(name: str) -> str:
    return _NETWORK_ABBREV.get(name, name[:8])


def _inter_network_coupling_by_block(
    rows: List[Dict[str, Any]],
    payload: Any,
) -> Dict[int, Dict[str, Any]]:
    """Compute per-block inter-network Jacobian off-diagonal coupling metrics.

    Stacks the per-network regional MNPS trajectories into a joint *k*-D
    vector (k = 3 × n_networks, sorted alphabetically by network name) and
    estimates a single block Jacobian via least-squares on consecutive-window
    differences:

        ΔX ≈ J · X_prev      (least-squares, shape k×k)

    Each ordered off-diagonal 3×3 block J[3i:3i+3, 3j:3j+3] describes how
    network *j*'s position predicts network *i*'s velocity.  Its Frobenius
    norm is stored as ``coupl_{abbr_i}_from_{abbr_j}``.

    Parameters
    ----------
    rows:
        Already-assembled flat rows (must contain ``block_id`` and
        ``source_window_index``).
    payload:
        ``MNPSPayload`` instance; must expose ``regional_mnps`` mapping at
        least two networks, each with a ``"mnps"`` array of shape *(T, 3)*.

    Returns
    -------
    Dict[int, Dict[str, Any]]
        ``block_id → coupling stats dict``.  Empty dict if fewer than two
        networks are available.  Blocks with too few finite samples carry
        NaN for every coupling column.
    """
    _nan = float("nan")
    regional = getattr(payload, "regional_mnps", None)
    if not isinstance(regional, dict) or len(regional) < 2:
        return {}

    net_labels = sorted(regional.keys())
    n_nets = len(net_labels)
    abbrevs = [_abbrev_network(n) for n in net_labels]

    # Extract per-network MNPS arrays (T, 3).
    net_mnps: Dict[str, Any] = {}
    for net in net_labels:
        entry = regional[net]
        mnps_arr = None
        if isinstance(entry, dict):
            mnps_arr = entry.get("mnps")
            if mnps_arr is None:
                # fall back to primary anchor layer
                for layer in entry.get("anchor_layers", {}).values():
                    if isinstance(layer, dict) and layer.get("mnps") is not None:
                        mnps_arr = layer["mnps"]
                        break
        if mnps_arr is None:
            return {}
        net_mnps[net] = np.asarray(mnps_arr, dtype=np.float64)

    # Build ordered coupling column names: all ordered (i→j) off-diagonal pairs.
    coupl_cols: List[tuple] = []  # (i, j, col_name)
    for i in range(n_nets):
        for j in range(n_nets):
            if i != j:
                coupl_cols.append((i, j, f"coupl_{abbrevs[i]}_from_{abbrevs[j]}"))

    _nan_stats: Dict[str, Any] = {col: _nan for _, _, col in coupl_cols}

    # Group rows by block_id preserving window order.
    from collections import defaultdict
    block_rows: Dict[int, list] = defaultdict(list)
    for row in rows:
        block_rows[int(row.get("block_id", -1))].append(row)

    # Minimum samples for a well-conditioned k×k least-squares.
    k = 3 * n_nets
    min_samples = k + 1

    def _estimate_coupling_from_widxs(w_idxs: List[int]) -> Dict[str, Any]:
        """Estimate coupling stats from a list of source_window_indices.

        Returns NaN dict on failure or insufficient samples.
        """
        if len(w_idxs) < min_samples:
            return dict(_nan_stats)
        try:
            parts = []
            for net in net_labels:
                mnps = net_mnps[net]
                rows_arr = np.full((len(w_idxs), 3), _nan)
                for ii, widx in enumerate(w_idxs):
                    if widx < mnps.shape[0]:
                        rows_arr[ii] = mnps[widx]
                parts.append(rows_arr)
            X = np.concatenate(parts, axis=1)
            finite_mask = np.isfinite(X).all(axis=1)
            X = X[finite_mask]
            if X.shape[0] < min_samples:
                return dict(_nan_stats)
            dX = np.diff(X, axis=0)
            X_prev = X[:-1]
            J_T, _, _, _ = np.linalg.lstsq(X_prev, dX, rcond=None)
            J = J_T.T
            out: Dict[str, Any] = {}
            for i, j, col in coupl_cols:
                block_ij = J[i * 3: (i + 1) * 3, j * 3: (j + 1) * 3]
                out[col] = float(np.linalg.norm(block_ij, "fro"))
            return out
        except Exception:
            return dict(_nan_stats)

    result: Dict[int, Dict[str, Any]] = {}
    for bid, brows in block_rows.items():
        result[bid] = dict(_nan_stats)

    # --- First pass: per-block estimation (preferred) ---
    for bid, brows in block_rows.items():
        w_idxs = [
            int(r["source_window_index"])
            for r in brows
            if int(r.get("source_window_index", -1)) >= 0
        ]
        coupling = _estimate_coupling_from_widxs(w_idxs)
        if any(np.isfinite(v) for v in coupling.values() if isinstance(v, float)):
            result[bid] = coupling

    # --- Stage-level fallback: pool windows within stage when per-block fails ---
    # Group blocks by their stage_code / task_state_label for pooling.
    stage_to_widxs: Dict[Any, List[int]] = {}
    stage_to_bids: Dict[Any, List[int]] = {}
    for bid, brows in block_rows.items():
        if any(np.isfinite(v) for v in result[bid].values() if isinstance(v, float)):
            continue  # already solved per-block
        stage_key = brows[0].get("stage_code", brows[0].get("task_state_label", "unknown")) if brows else "unknown"
        if stage_key not in stage_to_widxs:
            stage_to_widxs[stage_key] = []
            stage_to_bids[stage_key] = []
        for r in brows:
            swi = int(r.get("source_window_index", -1))
            if swi >= 0:
                stage_to_widxs[stage_key].append(swi)
        stage_to_bids[stage_key].append(bid)

    for stage_key, w_idxs_pooled in stage_to_widxs.items():
        coupling = _estimate_coupling_from_widxs(w_idxs_pooled)
        # Replicate stage-level estimate to every block that needed the fallback.
        for bid in stage_to_bids[stage_key]:
            result[bid] = coupling

    return result


# ---------------------------------------------------------------------------
# Main table builder
# ---------------------------------------------------------------------------

def build_block_native_table(
    *,
    blocks: Sequence,
    profile: BlockWindowProfileConfig,
    payload: Optional[Any] = None,
    subject_id: str = "",
    session_id: str = "",
    run_id: str = "",
    dataset_id: str = "",
    include_mnps: bool = True,
) -> List[Dict[str, Any]]:
    """Build a flat block-native window table for one subject/run.

    Parameters
    ----------
    blocks:
        Iterable of :class:`~mndm.pipeline.stage_blocking.StageBlockInterval`.
    profile:
        Window profile configuration.
    payload:
        Optional ``MNPSPayload`` for MNPS coordinate lookup.
    subject_id, session_id, run_id, dataset_id:
        Identifier strings repeated in every row.
    include_mnps:
        If True and payload is provided, attempt to attach MNPS coordinates
        by nearest-center lookup.

    Returns
    -------
    List[Dict[str, Any]]
        Flat rows suitable for ``pd.DataFrame(rows)`` or Parquet writing.
    """
    spec = block_window_spec_from_profile(profile)
    window_rows = generate_block_windows(list(blocks), spec)
    block_by_id: Dict[int, Any] = {
        int(getattr(b, "block_id", idx)): b
        for idx, b in enumerate(blocks)
    }

    if not window_rows:
        logger.debug(
            "block_native_export: no windows generated for subject=%s run=%s",
            subject_id, run_id,
        )
        return []

    rows: List[Dict[str, Any]] = []
    for brow in window_rows:
        row: Dict[str, Any] = {
            "subject_id": subject_id,
            "session_id": session_id,
            "run_id": run_id,
            "dataset_id": dataset_id,
            "block_id": brow.block_id,
            "window_id_within_block": brow.window_id_within_block,
            "stage_code": brow.stage_code,
            "block_start_sec": brow.block_start_sec,
            "block_end_sec": brow.block_end_sec,
            "block_duration_sec": brow.block_duration_sec,
            "window_start_sec": brow.window_start_sec,
            "window_end_sec": brow.window_end_sec,
            "window_center_sec": brow.window_center_sec,
            "relative_time_in_block_sec": brow.relative_time_in_block_sec,
            "distance_to_block_end_sec": brow.distance_to_block_end_sec,
            "relative_pos_0_1": brow.relative_pos_0_1,
            "partition_label": brow.partition_label,
            "is_post_offset": int(brow.is_post_offset),
        }
        src_block = block_by_id.get(int(getattr(brow, "block_id", -1)))
        row["frequency_hz"] = _as_frequency_hz(src_block) if src_block is not None else float("nan")
        row["derived_from"] = str(getattr(src_block, "derived_from", "")) if src_block is not None else ""
        row["end_reason"] = str(getattr(src_block, "end_reason", "")) if src_block is not None else ""

        w_idx = -1
        if payload is not None:
            w_idx = _find_closest_window_idx(
                payload, brow.window_start_sec, brow.window_end_sec
            )
        row["source_window_index"] = int(w_idx)
        if include_mnps and payload is not None:
            row.update(_mnps_at(payload, w_idx) if w_idx >= 0 else _mnps_nan())
        if payload is not None and w_idx >= 0:
            row.update(_raw_feature_exports_at(payload, w_idx))
            row.update(_anchor_exports_at(payload, w_idx))
            row.update(_labels_at(payload, w_idx))

        rows.append(row)

    # Attach per-block trajectory geometry (requires MNPS columns to be present).
    if rows and any("mnps_finite" in r for r in rows):
        traj_by_block = _trajectory_stats_by_block(rows)
        for row in rows:
            bid = int(row.get("block_id", -1))
            row.update(traj_by_block.get(bid, {
                "traj_path_length": float("nan"),
                "traj_endpoint_dist": float("nan"),
                "traj_efficiency": float("nan"),
                "traj_mean_curvature": float("nan"),
                "traj_max_curvature": float("nan"),
                "traj_n_finite_windows": 0,
            }))

    # Attach inter-network Jacobian off-diagonal coupling (requires regional_mnps in payload).
    if payload is not None:
        coupling_by_block = _inter_network_coupling_by_block(rows, payload)
        if coupling_by_block:
            for row in rows:
                bid = int(row.get("block_id", -1))
                row.update(coupling_by_block.get(bid, {}))

    logger.info(
        "block_native_export: %d blocks → %d windows (subject=%s run=%s)",
        len({brow.block_id for brow in window_rows}),
        len(rows),
        subject_id,
        run_id,
    )
    return rows


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_block_native_parquet(
    rows: List[Dict[str, Any]],
    out_path: Path,
) -> Optional[Path]:
    """Write block-native rows to Parquet. Returns path on success, ``None`` otherwise."""
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not installed; cannot write Parquet.")
        return None

    if not rows:
        logger.info("No block-native rows to write — skipping %s", out_path)
        return None

    try:
        df = pd.DataFrame(rows)
    except Exception:
        logger.exception("Failed to build DataFrame for block-native export")
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_parquet(out_path, index=False)
        logger.info(
            "Wrote block-native Parquet: %s (%d rows, %d cols)",
            out_path, len(df), len(df.columns),
        )
        return out_path
    except Exception:
        logger.exception("Failed to write Parquet to %s", out_path)
        csv_path = out_path.with_suffix(".csv")
        try:
            df.to_csv(csv_path, index=False)
            logger.info("Fell back to CSV: %s", csv_path)
        except Exception:
            logger.exception("CSV fallback also failed")
        return None


def write_block_native_csv(
    rows: List[Dict[str, Any]],
    out_path: Path,
) -> Optional[Path]:
    """Write block-native rows to CSV (no pandas required)."""
    if not rows:
        return None

    import csv as _csv

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_cols: list[str] = list(rows[0].keys())
    seen = set(all_cols)
    for r in rows[1:]:
        for k in r:
            if k not in seen:
                all_cols.append(k)
                seen.add(k)

    try:
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = _csv.DictWriter(fh, fieldnames=all_cols, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow({k: ("" if v is None else v) for k, v in r.items()})
        logger.info("Wrote block-native CSV: %s (%d rows)", out_path, len(rows))
        return out_path
    except Exception:
        logger.exception("Failed to write CSV to %s", out_path)
        return None


# ---------------------------------------------------------------------------
# Manifest helper
# ---------------------------------------------------------------------------

def block_native_export_manifest_entry(
    rows: List[Dict[str, Any]],
    *,
    dataset_id: str = "",
    out_paths: Optional[Sequence[Optional[Path]]] = None,
    profile: Optional[BlockWindowProfileConfig] = None,
) -> Dict[str, Any]:
    """Return a JSON-serializable manifest entry for one block-native export run."""
    block_ids: set = set()
    stage_codes: set = set()
    for row in rows:
        block_ids.add(row.get("block_id"))
        stage_codes.add(row.get("stage_code"))
    raw_hrv_feature_columns = sorted(
        {
            str(key)
            for row in rows
            for key in row.keys()
            if str(key).startswith("ecg_hrv_") or str(key) == "qc_ok_ecg_hrv"
        }
    )

    paths_entry: list[str] = []
    if out_paths:
        paths_entry = [str(p) for p in out_paths if p is not None]

    return {
        "n_rows_total": len(rows),
        "n_blocks": len(block_ids),
        "n_stage_codes": len(stage_codes),
        "dataset_id": dataset_id,
        "window_profile_kind": profile.kind if profile is not None else "unknown",
        "window_length_sec": profile.window_length_sec if profile is not None else None,
        "step_sec": profile.step_sec if profile is not None else None,
        "raw_hrv_feature_columns": raw_hrv_feature_columns,
        "output_paths": paths_entry,
    }


# ---------------------------------------------------------------------------
# Columnar conversion helpers (for MNPSPayload H5 contract)
# ---------------------------------------------------------------------------

def blocks_to_columns(blocks: Sequence) -> Dict[str, Any]:
    """Convert a list of StageBlockInterval to a columnar dict for H5 export.

    Returns a dict mapping column name → 1-D numpy array, suitable for
    ``payload.block_table_columns``.
    """
    if not blocks:
        return {}
    import numpy as np

    block_ids = np.array([int(b.block_id) for b in blocks], dtype=np.int32)
    stage_codes = np.array([int(b.stage_code) for b in blocks], dtype=np.int32)
    start_sec = np.array([float(b.start_sec) for b in blocks], dtype=np.float32)
    end_sec = np.array([float(b.end_sec) for b in blocks], dtype=np.float32)
    duration_sec = end_sec - start_sec
    frequency_hz = np.array([_as_frequency_hz(b) for b in blocks], dtype=np.float32)
    derived_from = np.array([str(getattr(b, "derived_from", "")) for b in blocks], dtype=object)
    end_reason = np.array([str(getattr(b, "end_reason", "")) for b in blocks], dtype=object)
    source_event_idx = np.array([int(getattr(b, "source_event_idx", -1)) for b in blocks], dtype=np.int32)
    support_event_count = np.array(
        [
            int(len(getattr(b, "support_event_indices", ()) or ()))
            for b in blocks
        ],
        dtype=np.int32,
    )
    membership_mode = np.array([str(getattr(b, "membership_mode", "")) for b in blocks], dtype=object)
    bridge_tail_sec = np.array([float(getattr(b, "bridge_tail_sec", float("nan"))) for b in blocks], dtype=np.float32)
    bridge_tail_cap_sec = np.array([float(getattr(b, "bridge_tail_cap_sec", float("nan"))) for b in blocks], dtype=np.float32)
    is_inferred = np.array([int(bool(getattr(b, "is_inferred", True))) for b in blocks], dtype=np.int8)

    return {
        "block_id": block_ids,
        "stage_code": stage_codes,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_sec": duration_sec,
        "frequency_hz": frequency_hz,
        "source_event_idx": source_event_idx,
        "support_event_count": support_event_count,
        "derived_from": derived_from,
        "end_reason": end_reason,
        "membership_mode": membership_mode,
        "bridge_tail_sec": bridge_tail_sec,
        "bridge_tail_cap_sec": bridge_tail_cap_sec,
        "is_inferred": is_inferred,
    }


def block_windows_to_columns(
    rows: "List[BlockWindowRow]",
    *,
    source_window_index: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Convert a list of BlockWindowRow to a columnar dict for H5 export.

    Returns a dict mapping column name → 1-D numpy array, suitable for
    ``payload.block_window_table_columns``.
    """
    if not rows:
        return {}
    from .block_windows import BlockWindowRow
    import numpy as np

    source_idx = np.asarray(
        list(source_window_index) if source_window_index is not None else [-1] * len(rows),
        dtype=np.int32,
    )
    if source_idx.shape[0] != len(rows):
        source_idx = np.full((len(rows),), -1, dtype=np.int32)

    return {
        "block_id": np.array([r.block_id for r in rows], dtype=np.int32),
        "window_id_within_block": np.array([r.window_id_within_block for r in rows], dtype=np.int32),
        "stage_code": np.array([r.stage_code for r in rows], dtype=np.int32),
        "block_start_sec": np.array([r.block_start_sec for r in rows], dtype=np.float32),
        "block_end_sec": np.array([r.block_end_sec for r in rows], dtype=np.float32),
        "block_duration_sec": np.array([r.block_duration_sec for r in rows], dtype=np.float32),
        "window_start_sec": np.array([r.window_start_sec for r in rows], dtype=np.float32),
        "window_end_sec": np.array([r.window_end_sec for r in rows], dtype=np.float32),
        "window_center_sec": np.array([r.window_center_sec for r in rows], dtype=np.float32),
        "relative_time_in_block_sec": np.array([r.relative_time_in_block_sec for r in rows], dtype=np.float32),
        "distance_to_block_end_sec": np.array([r.distance_to_block_end_sec for r in rows], dtype=np.float32),
        "relative_pos_0_1": np.array([r.relative_pos_0_1 for r in rows], dtype=np.float32),
        "source_window_index": source_idx,
        "partition_label": np.array([r.partition_label for r in rows], dtype=object),
        "is_post_offset": np.array([int(r.is_post_offset) for r in rows], dtype=np.int8),
    }


def inject_block_native_into_payload(
    payload: Any,
    *,
    events_df: Any,
    config: "Mapping[str, Any]",
    dataset_id: str,
    subject_id: str = "",
    session_id: str = "",
    run_id: str = "",
    sampling_cfg: Optional["Mapping[str, Any]"] = None,
    stage_map: Optional["Mapping[str, int]"] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Infer blocks and inject block-native tables into an MNPSPayload.

    This function is the M6 injection point: it is called after the payload
    is assembled in summary.py and before it is written to H5.  It mutates
    ``payload.block_table_columns`` and ``payload.block_window_table_columns``
    in place when block-native mode is enabled for the dataset.

    Parameters
    ----------
    payload:
        :class:`~mndm.schema.MNPSPayload` instance (mutated in place).
    events_df:
        ``pandas.DataFrame`` of BIDS events for this run.
    config:
        Fully resolved config dict.
    dataset_id:
        Dataset identifier.
    subject_id, session_id, run_id:
        Identifiers for logging only.
    sampling_cfg:
        Optional ``epoching.datasets.<id>.sampling`` dict for stage-blocking.
    stage_map:
        Optional label → int mapping for stage code inference.

    Returns
    -------
    Dict[str, Any]
        Manifest entry dict (may be empty if block_native is disabled).
    """
    try:
        from .block_native_config import (
            block_native_dataset_config_from_config,
            infer_blocks_from_events,
        )
        from .block_windows import generate_block_windows
        from typing import Mapping as _Mapping
    except ImportError as exc:
        logger.warning("block_native inject: import error: %s", exc)
        return {}

    if not isinstance(config, _Mapping):
        return {}

    bn_cfg = block_native_dataset_config_from_config(config, dataset_id)
    if not bn_cfg.enabled:
        return {}

    try:
        blocks = infer_blocks_from_events(
            events_df,
            bn_cfg.source,
            stage_map=stage_map,
            sampling_cfg=sampling_cfg,
        )
        if not blocks:
            logger.info(
                "block_native inject: no blocks found (subject=%s run=%s dataset=%s)",
                subject_id, run_id, dataset_id,
            )
            return {
                "n_blocks": 0,
                "n_windows": 0,
                "status": "no_blocks",
                "block_source_kind": bn_cfg.source.kind,
                "window_profile_kind": bn_cfg.window_profile.kind,
                "named_window_profile": bn_cfg.window_profile.named_profile or None,
            }

        spec = block_window_spec_from_profile(bn_cfg.window_profile)
        window_rows = generate_block_windows(blocks, spec)
        source_indices = [
            int(_find_closest_window_idx(payload, row.window_start_sec, row.window_end_sec))
            for row in window_rows
        ]

        payload.block_table_columns = blocks_to_columns(blocks)
        payload.block_window_table_columns = block_windows_to_columns(
            window_rows,
            source_window_index=source_indices,
        )

        export_paths: list[str] = []
        if output_dir is not None and (bn_cfg.export_parquet or bn_cfg.export_csv):
            out_dir = Path(output_dir)
            table_rows = build_block_native_table(
                blocks=blocks,
                profile=bn_cfg.window_profile,
                payload=payload,
                subject_id=subject_id,
                session_id=session_id,
                run_id=run_id,
                dataset_id=dataset_id,
                include_mnps=True,
            )
            if bn_cfg.export_parquet:
                parquet_path = write_block_native_parquet(
                    table_rows,
                    out_dir / "block_native_windows.parquet",
                )
                if parquet_path is not None:
                    export_paths.append(str(parquet_path))
            if bn_cfg.export_csv:
                csv_path = write_block_native_csv(
                    table_rows,
                    out_dir / "block_native_windows.csv",
                )
                if csv_path is not None:
                    export_paths.append(str(csv_path))

        block_stage_counts = Counter(int(getattr(b, "stage_code", -1)) for b in blocks)
        block_freq_counts = Counter(
            int(round(float(freq)))
            for freq in [_as_frequency_hz(b) for b in blocks]
            if np.isfinite(freq)
        )
        window_stage_counts = Counter(int(getattr(r, "stage_code", -1)) for r in window_rows)
        qc_entry = _summarize_block_native_qc(
            blocks=blocks,
            window_rows=window_rows,
            source_indices=source_indices,
        )
        qc_entry["filters"] = {
            "min_block_sec": float(bn_cfg.window_profile.min_block_sec),
            "min_windows_per_block": int(bn_cfg.window_profile.min_windows_per_block),
        }

        logger.info(
            "block_native inject: %d blocks → %d windows (subject=%s run=%s dataset=%s)",
            len(blocks), len(window_rows), subject_id, run_id, dataset_id,
        )
        return {
            "n_blocks": len(blocks),
            "n_windows": len(window_rows),
            "status": "ok",
            "window_profile_kind": bn_cfg.window_profile.kind,
            "named_window_profile": bn_cfg.window_profile.named_profile or None,
            "block_source_kind": bn_cfg.source.kind,
            "block_counts_by_stage": {str(k): int(v) for k, v in sorted(block_stage_counts.items(), key=lambda kv: kv[0])},
            "window_counts_by_stage": {str(k): int(v) for k, v in sorted(window_stage_counts.items(), key=lambda kv: kv[0])},
            "block_counts_by_frequency_hz": {str(k): int(v) for k, v in sorted(block_freq_counts.items(), key=lambda kv: kv[0])},
            "source_window_match_count": int(np.sum(np.asarray(source_indices, dtype=np.int32) >= 0)) if source_indices else 0,
            "source_window_total": int(len(source_indices)),
            "export_paths": export_paths,
            "qc_entry": qc_entry,
        }
    except Exception as exc:
        logger.warning(
            "block_native inject: failed for subject=%s run=%s: %s",
            subject_id, run_id, exc,
        )
        return {"status": "error", "error": str(exc)}
