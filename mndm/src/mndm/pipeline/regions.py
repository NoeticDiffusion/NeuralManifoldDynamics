"""
regions.py
Network and region utilities for fMRI parcellation mapping."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import svd


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


def infer_network_label(region_name: str) -> str:
    """Best-effort mapping from atlas ROI names to canonical network labels."""
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


def group_region_indices(region_names: Optional[List[str]]) -> Dict[str, List[int]]:
    """Group region indices by inferred network label."""
    groups: Dict[str, List[int]] = {}
    if not region_names:
        return groups
    for idx, name in enumerate(region_names):
        label = infer_network_label(name)
        groups.setdefault(label, []).append(idx)
    return groups


def select_complete_group_rois(
    regions_bold: np.ndarray,
    indices: Sequence[int],
    region_names: Optional[Sequence[str]] = None,
) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """Keep only fully finite ROI rows for a network.

    Returns ``(None, None)`` when no ROI is fully finite. Partial-NaN and
    all-NaN rows are omitted rather than median- or zero-filled.
    """
    if not indices:
        return None, None
    idx = np.asarray(indices, dtype=int)
    roi_block = np.asarray(regions_bold)[idx, :]
    if roi_block.ndim != 2 or roi_block.shape[1] == 0:
        return None, None
    complete = np.isfinite(roi_block).all(axis=1)
    if not np.any(complete):
        return None, None
    kept = np.asarray(roi_block[complete], dtype=float)
    kept_names: Optional[List[str]] = None
    if region_names is not None:
        kept_idx = idx[complete]
        n_names = len(region_names)
        kept_names = [
            str(region_names[int(i)])
            for i in kept_idx
            if 0 <= int(i) < n_names
        ]
    return kept, kept_names


def complete_group_roi_indices(
    regions_bold: np.ndarray,
    indices: Sequence[int],
) -> np.ndarray:
    """Original ROI indices that are fully finite (no imputation)."""
    if not indices:
        return np.zeros((0,), dtype=int)
    idx = np.asarray(indices, dtype=int)
    roi_block = np.asarray(regions_bold)[idx, :]
    if roi_block.ndim != 2 or roi_block.shape[1] == 0:
        return np.zeros((0,), dtype=int)
    complete = np.isfinite(roi_block).all(axis=1)
    return idx[complete]


def aggregate_group_timeseries(
    regions_bold: Optional[np.ndarray],
    groups: Mapping[str, Sequence[int]],
) -> Dict[str, np.ndarray]:
    """Compute first-principal-component time series per network/group.

    ROIs that are not fully finite are dropped. A group with no remaining
    ROI is omitted rather than median- or zero-filled into a fake PC.
    """
    aggregated: Dict[str, np.ndarray] = {}
    if regions_bold is None or not groups:
        return aggregated
    for label, indices in groups.items():
        complete, _ = select_complete_group_rois(regions_bold, indices)
        if complete is None:
            continue

        centered = complete - np.mean(complete, axis=1, keepdims=True)
        if not np.isfinite(centered).all():
            continue

        try:
            _, s, vt = svd(centered, full_matrices=False, check_finite=True)
        except Exception:
            continue
        if s.size == 0 or vt.size == 0:
            continue

        # First temporal PC score: sign-aligned for deterministic orientation.
        pc1 = vt[0, :] * s[0]
        if np.nansum(pc1) < 0:
            pc1 = -pc1
        aggregated[label] = np.asarray(pc1, dtype=float)
    return aggregated


def stack_group_matrix(group_ts: Mapping[str, np.ndarray]) -> Tuple[Optional[np.ndarray], List[str]]:
    """Return stacked matrix [n_groups, n_times] sorted by label."""
    if not group_ts:
        return None, []
    names = sorted(group_ts.keys())
    matrix = np.stack([group_ts[name] for name in names], axis=0)
    return matrix, names

