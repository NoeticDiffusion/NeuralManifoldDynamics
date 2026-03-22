"""Helpers for channel-shift ensembles (spatial robustness).

This module provides small utilities to:

- Parse ensemble group definitions from the config (with per-dataset overrides).
- Resolve those group definitions against the available EEG channels.
- Normalise group names into safe suffixes for feature column names.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class EnsembleGroupDef:
    """Resolved ensemble group definition for a given recording.

    Attributes:
        name: Raw group name from the config (for example ``frontal``).
        safe_name: Normalised identifier used in feature column suffixes
            (for example ``parietal_occipital``).
        indices: Indices of EEG channels belonging to the group.
        channels: Resolved EEG channel names as present in the data.
    """

    name: str
    safe_name: str
    indices: List[int]
    channels: List[str]


def sanitize_group_name(name: str) -> str:
    """Return a safe group identifier for use in column suffixes.

    Examples:
        >>> sanitize_group_name("Frontal")
        'frontal'
        >>> sanitize_group_name("Parietal/Occipital")
        'parietal_occipital'
    """

    base = str(name).strip().lower()
    # Replace any non-alphanumeric sequence with a single underscore
    base = re.sub(r"[^0-9a-zA-Z]+", "_", base)
    base = base.strip("_")
    return base or "group"


def _canonical_channel(label: str) -> str:
    """Canonicalize a channel label for robust matching.

    Upper-cases, strips whitespace, and drops non-alphanumeric characters so
    labels such as ``Fp1``, ``FP1``, and ``Fp1-REF`` map to the same key.

    Args:
        label: Raw channel label string.

    Returns:
        Canonical key string.
    """

    return re.sub(r"[^0-9A-Z]+", "", str(label).strip().upper())


def resolve_config_groups(cfg: Optional[Mapping[str, Any]], dataset_id: Optional[str]) -> Dict[str, List[str]]:
    """Resolve ensemble group definitions from config for a dataset.

    Args:
        cfg: ``robustness.ensembles`` (or similar) mapping with ``groups`` and
            optional per-dataset overrides under ``datasets.<id>``.
        dataset_id: Current dataset id, or None to use only global groups.

    Returns:
        Mapping from group name to list of channel label strings.
    """

    if not isinstance(cfg, Mapping):
        return {}

    groups: Dict[str, List[str]] = {}

    base_groups = cfg.get("groups", {})
    if isinstance(base_groups, Mapping):
        for g_name, ch_list in base_groups.items():
            if isinstance(ch_list, Sequence) and not isinstance(ch_list, (str, bytes)):
                groups[str(g_name)] = [str(ch) for ch in ch_list]

    ds_all = cfg.get("datasets", {})
    if dataset_id and isinstance(ds_all, Mapping):
        ds_cfg = ds_all.get(dataset_id)
        if isinstance(ds_cfg, Mapping):
            ds_groups = ds_cfg.get("groups", ds_cfg)
            if isinstance(ds_groups, Mapping):
                for g_name, ch_list in ds_groups.items():
                    if isinstance(ch_list, Sequence) and not isinstance(ch_list, (str, bytes)):
                        groups[str(g_name)] = [str(ch) for ch in ch_list]

    return groups


def realize_ensemble_groups(
    cfg: Optional[Mapping[str, Any]],
    dataset_id: Optional[str],
    available_channels: Sequence[str],
) -> List[EnsembleGroupDef]:
    """Resolve configured groups against ``available_channels`` and return defs.

    Args:
        cfg: Ensemble configuration (see :func:`resolve_config_groups`).
        dataset_id: Dataset id for per-dataset overrides.
        available_channels: Channel names in recording order.

    Returns:
        List of :class:`EnsembleGroupDef` entries with non-empty channel sets.
    """

    if not available_channels:
        return []

    groups = resolve_config_groups(cfg, dataset_id)
    if not groups:
        return []

    available = [str(ch) for ch in available_channels]
    canonical_map = {_canonical_channel(ch): idx for idx, ch in enumerate(available)}

    resolved: List[EnsembleGroupDef] = []
    for group_name, channel_list in groups.items():
        if not channel_list:
            continue
        indices: List[int] = []
        resolved_channels: List[str] = []
        for ch in channel_list:
            key = _canonical_channel(ch)
            if key in canonical_map:
                idx = canonical_map[key]
                indices.append(idx)
                resolved_channels.append(available[idx])
        if not indices:
            logger.debug("Ensemble group '%s' resolved to 0 channels for dataset %s", group_name, dataset_id)
            continue
        resolved.append(
            EnsembleGroupDef(
                name=str(group_name),
                safe_name=sanitize_group_name(str(group_name)),
                indices=indices,
                channels=resolved_channels,
            )
        )

    return resolved

