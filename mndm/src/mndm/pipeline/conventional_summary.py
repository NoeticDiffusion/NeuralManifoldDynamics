"""Helpers for config-driven conventional EEG comparator summaries."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from .robustness_helpers import _distributional_descriptives


def _deep_merge_mapping(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge two config mappings."""
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge_mapping(dict(merged.get(key, {})), dict(value))
        else:
            merged[key] = value
    return merged


def resolve_conventional_eeg_cfg(config: Mapping[str, Any], dataset_id: Optional[str]) -> Dict[str, Any]:
    """Resolve conventional EEG config with optional dataset overrides."""
    root = config.get("conventional_eeg", {}) if isinstance(config, Mapping) else {}
    if not isinstance(root, Mapping):
        return {}
    merged: Dict[str, Any] = {k: root[k] for k in root if k != "datasets"}
    ds_map = root.get("datasets", {})
    if dataset_id and isinstance(ds_map, Mapping):
        ds_cfg = ds_map.get(dataset_id)
        if isinstance(ds_cfg, Mapping):
            merged = _deep_merge_mapping(merged, dict(ds_cfg))
    return merged


def _normalize_packs(conventional_cfg: Mapping[str, Any]) -> set[str]:
    """Return normalized conventional EEG pack names."""
    packs_raw = conventional_cfg.get("packs", ["tier1"]) if isinstance(conventional_cfg, Mapping) else ["tier1"]
    if isinstance(packs_raw, (str, bytes)):
        return {str(packs_raw).strip().lower()}
    if isinstance(packs_raw, list):
        return {str(v).strip().lower() for v in packs_raw if str(v).strip()}
    return {"tier1"}


def compute_conventional_eeg_summary(
    *,
    sub_frame: pd.DataFrame,
    config: Mapping[str, Any],
    dataset_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Compute summarize-time descriptives for conventional EEG comparator columns."""
    conventional_cfg = resolve_conventional_eeg_cfg(config, dataset_id)
    if not conventional_cfg or not bool(conventional_cfg.get("enabled", False)):
        return None

    packs = _normalize_packs(conventional_cfg)

    export_cfg = conventional_cfg.get("export", {})
    if isinstance(export_cfg, Mapping) and not bool(export_cfg.get("summaries", True)):
        return None

    conventional_cols = [
        str(col)
        for col in sub_frame.columns
        if str(col).startswith("eeg_conventional_")
    ]
    if not conventional_cols:
        return None

    families: Dict[str, Dict[str, Any]] = {}
    for col in sorted(conventional_cols):
        suffix = str(col)[len("eeg_conventional_") :]
        family, sep, feature_name = suffix.partition("_")
        if not sep:
            family = "misc"
            feature_name = suffix
        families.setdefault(family or "misc", {})[feature_name] = col

    family_payload: Dict[str, Any] = {}
    for family_name, feature_cols in families.items():
        ordered_items = sorted(feature_cols.items(), key=lambda item: item[0])
        feature_names = [feature_name for feature_name, _ in ordered_items]
        values = np.column_stack(
            [
                pd.to_numeric(sub_frame[col_name], errors="coerce").to_numpy(dtype=float)
                for _, col_name in ordered_items
            ]
        )
        descriptives = _distributional_descriptives(values, feature_names)
        family_payload[family_name] = {
            feature_name: {
                "column": feature_cols[feature_name],
                **stats,
            }
            for feature_name, stats in descriptives.items()
        }

    return {
        "schema_version": "mndm.conventional_eeg.v1",
        "packs": sorted(packs),
        "column_count": int(len(conventional_cols)),
        "columns": sorted(conventional_cols),
        "families": family_payload,
    }
