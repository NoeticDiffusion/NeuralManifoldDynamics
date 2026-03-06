"""Regional MNPS helpers extracted from summary.py."""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .regions import aggregate_group_timeseries, group_region_indices, stack_group_matrix
from .regional_mnps import compute_all_regional_mnps, RegionalMNPSSummary
from ..features import fmri as fmri_features
from .. import projection

logger = logging.getLogger(__name__)


def extract_eeg_group_feature_frames(sub_frame: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Build per-group EEG feature frames from `__g_<group>` suffixed columns."""
    if sub_frame is None or sub_frame.empty:
        return {}
    suffix_re = re.compile(r"^(?P<base>.+)__g_(?P<group>[A-Za-z0-9_]+)$")
    by_group: Dict[str, Dict[str, str]] = {}
    suffixed_cols: List[str] = []
    for col in sub_frame.columns:
        match = suffix_re.match(str(col))
        if not match:
            continue
        base_name = str(match.group("base")).strip()
        group_name = str(match.group("group")).strip()
        if not base_name or not group_name:
            continue
        by_group.setdefault(group_name, {})[base_name] = str(col)
        suffixed_cols.append(str(col))
    if not by_group:
        return {}

    base_frame = sub_frame.drop(columns=sorted(set(suffixed_cols)), errors="ignore").copy()
    out: Dict[str, pd.DataFrame] = {}
    for group_name, mapping in by_group.items():
        g_frame = base_frame.copy()
        for base_name, source_col in mapping.items():
            g_frame[base_name] = sub_frame[source_col]
        out[group_name] = g_frame
    return out


def build_precomputed_eeg_group_trajectories(
    group_frames: Mapping[str, pd.DataFrame],
    axis_weights: Mapping[str, Mapping[str, float]],
    config: Mapping[str, Any],
    proj_cfg: Mapping[str, Any],
    normalize_mode: Optional[str],
    subcoords_spec: Mapping[str, Any],
    v2_enabled: bool,
    *,
    resolve_mnps_3d_cfg: Callable[[Mapping[str, Any]], Dict[str, Any]],
    coerce_v1_mapping_to_v2_subcoords: Callable[[Mapping[str, Any], Mapping[str, Any]], Dict[str, Dict[str, float]]],
    align_v2_subcoords: Callable[[np.ndarray, List[str], List[str]], np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Build per-group MNPS/v2 trajectories from grouped EEG feature frames."""
    network_mnps: Dict[str, np.ndarray] = {}
    network_stratified: Dict[str, np.ndarray] = {}
    if not group_frames:
        return network_mnps, network_stratified

    clip_threshold = float(proj_cfg.get("clip_threshold", 6.0)) if isinstance(proj_cfg, Mapping) else 6.0
    feature_standardization = (
        proj_cfg.get("feature_standardization")
        if isinstance(proj_cfg, Mapping) and isinstance(proj_cfg.get("feature_standardization"), Mapping)
        else None
    )
    ordered_subcoords = ["m_a", "m_e", "m_o", "d_n", "d_l", "d_s", "e_e", "e_s", "e_m"]
    m3d_cfg = resolve_mnps_3d_cfg(config)
    mde_mode = m3d_cfg.get("mode", "direct_features")

    for group_label, g_frame in group_frames.items():
        if g_frame is None or g_frame.empty:
            continue
        x_net, _ = projection.project_features(
            g_frame,
            axis_weights,
            normalize=normalize_mode,
            feature_standardization=feature_standardization,
            clip_threshold=clip_threshold,
        )
        if x_net.ndim == 2 and x_net.shape[0] > 0 and x_net.shape[1] >= 3:
            network_mnps[group_label] = np.asarray(x_net[:, :3], dtype=np.float32)

        if v2_enabled and isinstance(subcoords_spec, Mapping) and subcoords_spec:
            coords_9d_net, names_v2, _ = projection.project_features_v2(
                g_frame,
                subcoords_spec,
                normalize=normalize_mode,
                feature_standardization=feature_standardization,
                clip_threshold=clip_threshold,
            )
            if coords_9d_net.ndim == 2 and coords_9d_net.shape[0] > 0 and names_v2:
                network_stratified[group_label] = align_v2_subcoords(
                    coords_9d_net, list(names_v2), ordered_subcoords
                )
                if mde_mode == "from_v2":
                    cfg_weights_net = coerce_v1_mapping_to_v2_subcoords(
                        m3d_cfg.get("v1_mapping", {}),
                        subcoords_spec,
                    )
                    has_weights = any(
                        isinstance(cfg_weights_net.get(ax, {}), Mapping) and cfg_weights_net.get(ax)
                        for ax in ("m", "d", "e")
                    )
                    if has_weights:
                        try:
                            x_net_v2, _, _ = projection.derive_mde_from_v2(
                                coords_9d_net,
                                list(names_v2),
                                cfg_weights_net,
                                pooling=str(m3d_cfg.get("legacy_pooling", "mean")),
                                normalize_columns_l2=True,
                                enforce_block_selective=False,
                                return_mapping_info=True,
                            )
                            if x_net_v2.ndim == 2 and x_net_v2.shape[0] > 0 and x_net_v2.shape[1] >= 3:
                                network_mnps[group_label] = np.asarray(x_net_v2[:, :3], dtype=np.float32)
                        except Exception:
                            logger.exception(
                                "derive_mde_from_v2 failed for EEG group %s; retaining direct-feature result",
                                group_label,
                            )
    return network_mnps, network_stratified


def build_precomputed_network_trajectories(
    regions_bold: Optional[np.ndarray],
    regions_names: Optional[List[str]],
    regions_sfreq: Optional[float],
    region_groups: Mapping[str, List[int]],
    axis_weights: Mapping[str, Mapping[str, float]],
    dataset_id: str,
    config: Mapping[str, Any],
    sub_frame: pd.DataFrame,
    proj_cfg: Mapping[str, Any],
    normalize_mode: Optional[str],
    subcoords_spec: Mapping[str, Any],
    v2_enabled: bool,
    *,
    resolve_mnps_3d_cfg: Callable[[Mapping[str, Any]], Dict[str, Any]],
    coerce_v1_mapping_to_v2_subcoords: Callable[[Mapping[str, Any], Mapping[str, Any]], Dict[str, Dict[str, float]]],
    align_v2_subcoords: Callable[[np.ndarray, List[str], List[str]], np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Build per-network precomputed trajectories for regional fMRI MNPS."""
    network_mnps: Dict[str, np.ndarray] = {}
    network_stratified: Dict[str, np.ndarray] = {}
    if regions_bold is None or regions_sfreq is None or not region_groups:
        return network_mnps, network_stratified

    clip_threshold = float(proj_cfg.get("clip_threshold", 6.0)) if isinstance(proj_cfg, Mapping) else 6.0
    feature_standardization = (
        proj_cfg.get("feature_standardization")
        if isinstance(proj_cfg, Mapping) and isinstance(proj_cfg.get("feature_standardization"), Mapping)
        else None
    )
    ordered_subcoords = ["m_a", "m_e", "m_o", "d_n", "d_l", "d_s", "e_e", "e_s", "e_m"]
    m3d_cfg = resolve_mnps_3d_cfg(config)
    mde_mode = m3d_cfg.get("mode", "direct_features")
    kept_epochs = (
        set(pd.to_numeric(sub_frame["epoch_id"], errors="coerce").dropna().astype(int).tolist())
        if "epoch_id" in sub_frame.columns
        else None
    )

    for network_label, indices in region_groups.items():
        if not indices:
            continue
        idx = np.asarray(indices, dtype=int)
        if idx.size == 0:
            continue
        roi_slice = np.asarray(regions_bold[idx, :], dtype=np.float32)
        if roi_slice.ndim != 2 or roi_slice.shape[1] < 2:
            continue
        roi_names_slice = None
        if isinstance(regions_names, list) and regions_names:
            roi_names_slice = [regions_names[i] for i in idx if 0 <= i < len(regions_names)]

        try:
            net_features = fmri_features.compute_fmri_features(
                {
                    "signals": {"fmri": roi_slice},
                    "sfreq": float(regions_sfreq),
                    "channels": {"fmri": roi_names_slice or []},
                    "dataset_id": dataset_id,
                },
                config,
            )
        except Exception:
            logger.exception("Failed to compute network-level fMRI features for %s", network_label)
            continue

        if net_features is None or net_features.empty:
            continue
        if kept_epochs is not None and "epoch_id" in net_features.columns:
            net_features = net_features[
                pd.to_numeric(net_features["epoch_id"], errors="coerce").fillna(-1).astype(int).isin(kept_epochs)
            ].copy()
        if net_features.empty:
            continue

        x_net, _ = projection.project_features(
            net_features,
            axis_weights,
            normalize=normalize_mode,
            feature_standardization=feature_standardization,
            clip_threshold=clip_threshold,
        )
        if x_net.ndim == 2 and x_net.shape[0] > 0 and x_net.shape[1] >= 3:
            network_mnps[network_label] = np.asarray(x_net[:, :3], dtype=np.float32)

        if v2_enabled and isinstance(subcoords_spec, Mapping) and subcoords_spec:
            coords_9d_net, names_v2, _ = projection.project_features_v2(
                net_features,
                subcoords_spec,
                normalize=normalize_mode,
                feature_standardization=feature_standardization,
                clip_threshold=clip_threshold,
            )
            if coords_9d_net.ndim == 2 and coords_9d_net.shape[0] > 0 and names_v2:
                network_stratified[network_label] = align_v2_subcoords(
                    coords_9d_net, list(names_v2), ordered_subcoords
                )
                if mde_mode == "from_v2":
                    cfg_weights_net = coerce_v1_mapping_to_v2_subcoords(
                        m3d_cfg.get("v1_mapping", {}),
                        subcoords_spec,
                    )
                    has_weights = any(
                        isinstance(cfg_weights_net.get(ax, {}), Mapping) and cfg_weights_net.get(ax)
                        for ax in ("m", "d", "e")
                    )
                    if has_weights:
                        try:
                            x_net_v2, _, _ = projection.derive_mde_from_v2(
                                coords_9d_net,
                                list(names_v2),
                                cfg_weights_net,
                                pooling=str(m3d_cfg.get("legacy_pooling", "mean")),
                                normalize_columns_l2=True,
                                enforce_block_selective=False,
                                return_mapping_info=True,
                            )
                            if x_net_v2.ndim == 2 and x_net_v2.shape[0] > 0 and x_net_v2.shape[1] >= 3:
                                network_mnps[network_label] = np.asarray(x_net_v2[:, :3], dtype=np.float32)
                        except Exception:
                            logger.exception(
                                "derive_mde_from_v2 failed for network %s; retaining direct-feature result",
                                network_label,
                            )
    return network_mnps, network_stratified


def compute_regional_context(
    *,
    sub_frame: pd.DataFrame,
    regions_bold: Optional[np.ndarray],
    regions_names: Optional[List[str]],
    regions_sfreq: Optional[float],
    config: Mapping[str, Any],
    regional_mnps_cfg: Mapping[str, Any],
    subcoords_spec: Mapping[str, Any],
    axis_weights: Mapping[str, Mapping[str, float]],
    dataset_id: str,
    dataset_label: str,
    proj_cfg: Mapping[str, Any],
    normalize_mode: Optional[str],
    subject: str,
    session: Optional[str],
    condition: Optional[str],
    task: Optional[str],
    resolve_mnps_3d_cfg: Callable[[Mapping[str, Any]], Dict[str, Any]],
    coerce_v1_mapping_to_v2_subcoords: Callable[[Mapping[str, Any], Mapping[str, Any]], Dict[str, Dict[str, float]]],
    align_v2_subcoords: Callable[[np.ndarray, List[str], List[str]], np.ndarray],
) -> Tuple[
    Dict[str, np.ndarray],
    Optional[np.ndarray],
    List[str],
    Dict[str, List[int]],
    Optional[RegionalMNPSSummary],
]:
    """Compute regional grouping context and optional regional MNPS result."""
    region_groups = group_region_indices(regions_names)
    group_ts = aggregate_group_timeseries(regions_bold, region_groups)
    group_matrix, group_names = stack_group_matrix(group_ts)

    regional_mnps_results: Optional[RegionalMNPSSummary] = None
    if not bool(regional_mnps_cfg.get("enabled", False)):
        return group_ts, group_matrix, group_names, region_groups, regional_mnps_results

    min_regions_required = int(regional_mnps_cfg.get("min_regions_required", 195) or 0)
    if regions_bold is not None and min_regions_required > 0:
        n_regions = int(np.asarray(regions_bold).shape[0])
        if n_regions < min_regions_required:
            raise ValueError(
                "Regional MNPS aborted: insufficient fMRI ROI coverage "
                f"(n_regions={n_regions}, required>={min_regions_required}). "
                "Use MNI-normalized derivatives / valid atlas coverage before network analysis."
            )

    regional_strat_cfg = (
        regional_mnps_cfg.get("stratified", {})
        if isinstance(regional_mnps_cfg.get("stratified", {}), Mapping)
        else {}
    )
    regional_subcoords_spec = (
        regional_strat_cfg.get("subcoords", {})
        if isinstance(regional_strat_cfg.get("subcoords", {}), Mapping)
        else {}
    )
    regional_strat_enabled = bool(regional_strat_cfg.get("enabled", False))
    strat_subcoords_for_regions = (
        regional_subcoords_spec
        if regional_strat_enabled and regional_subcoords_spec
        else (subcoords_spec if isinstance(subcoords_spec, Mapping) else {})
    )

    network_mnps: Dict[str, np.ndarray] = {}
    network_stratified: Dict[str, np.ndarray] = {}
    if group_ts and regions_sfreq:
        network_mnps, network_stratified = build_precomputed_network_trajectories(
            regions_bold=regions_bold,
            regions_names=regions_names,
            regions_sfreq=regions_sfreq,
            region_groups=region_groups,
            axis_weights=axis_weights,
            dataset_id=dataset_id,
            config=config,
            sub_frame=sub_frame,
            proj_cfg=proj_cfg,
            normalize_mode=normalize_mode,
            subcoords_spec=strat_subcoords_for_regions if isinstance(strat_subcoords_for_regions, Mapping) else {},
            v2_enabled=bool(strat_subcoords_for_regions),
            resolve_mnps_3d_cfg=resolve_mnps_3d_cfg,
            coerce_v1_mapping_to_v2_subcoords=coerce_v1_mapping_to_v2_subcoords,
            align_v2_subcoords=align_v2_subcoords,
        )

    modality = str(config.get("modality", "")).strip().lower() if isinstance(config, Mapping) else ""
    if modality == "eeg":
        eeg_group_frames = extract_eeg_group_feature_frames(sub_frame)
        if eeg_group_frames:
            eeg_mnps, eeg_stratified = build_precomputed_eeg_group_trajectories(
                group_frames=eeg_group_frames,
                axis_weights=axis_weights,
                config=config,
                proj_cfg=proj_cfg,
                normalize_mode=normalize_mode,
                subcoords_spec=(strat_subcoords_for_regions if isinstance(strat_subcoords_for_regions, Mapping) else {}),
                v2_enabled=bool(strat_subcoords_for_regions),
                resolve_mnps_3d_cfg=resolve_mnps_3d_cfg,
                coerce_v1_mapping_to_v2_subcoords=coerce_v1_mapping_to_v2_subcoords,
                align_v2_subcoords=align_v2_subcoords,
            )
            network_mnps.update(eeg_mnps)
            network_stratified.update(eeg_stratified)

    if network_mnps:
        regional_mnps_results = compute_all_regional_mnps(
            group_ts=group_ts,
            sfreq=regions_sfreq,
            config=regional_mnps_cfg,
            subject=subject,
            session=session,
            condition=condition,
            task=task,
            network_mnps=network_mnps,
            network_stratified=network_stratified,
        )
    else:
        logger.info(
            "Regional MNPS enabled for %s but no regional trajectories were available",
            dataset_label,
        )

    return group_ts, group_matrix, group_names, region_groups, regional_mnps_results
