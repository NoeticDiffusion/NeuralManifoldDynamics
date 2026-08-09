"""Report-only QC for frozen MEG sensor sectors.

Sector labels denote helmet measurement geometry only.  This module neither
writes MNPS coordinates nor assigns cortical/biological regional labels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from .meg_transform_replay import validate_meg_h5


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _names(data: np.ndarray) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in data]


def _split_half_corr(values: np.ndarray) -> float:
    """Odd/even temporal split-half reliability, sign-preserving by design."""
    odd, even = values[::2], values[1::2]
    n = min(len(odd), len(even))
    if n < 3:
        return float("nan")
    a, b = odd[:n], even[:n]
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[finite], b[finite])[0, 1])


def run_sensor_topography_qc(
    h5_path: Path,
    config: Mapping[str, Any],
    *,
    dataset_id: str,
    cross_modal: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """Run an opt-in frozen-sector QC report from exported H5 surfaces."""
    qc_cfg = config.get("sensor_topography_qc", {})
    if not isinstance(qc_cfg, Mapping) or not bool(qc_cfg.get("enabled", False)):
        return {"status": "disabled", "reason": "sensor_topography_qc.enabled is false"}
    if cross_modal:
        contract = qc_cfg.get("sensor_topography_contract")
        if not isinstance(contract, Mapping) or not contract:
            raise ValueError("Cross-modal QC requires a frozen sensor_topography_contract")
        # This implementation deliberately keeps cross-modal inference out of
        # default runs; callers must provide independently aligned inputs.
        raise ValueError("Cross-modal QC requires the dedicated paired-H5 workflow; it is not enabled by a single-H5 call")

    replay = validate_meg_h5(h5_path, config, dataset_id=dataset_id)
    with h5py.File(h5_path, "r") as h5:
        raw = np.asarray(h5["features_raw/values"], dtype=float)
        names = _names(h5["features_raw/names"][:])
    groups_cfg = config.get("meg_ensembles", {})
    groups = groups_cfg.get("groups", {}) if isinstance(groups_cfg, Mapping) else {}
    group_names = list(groups)
    grouped_columns = {
        group: [i for i, name in enumerate(names) if name.startswith("meg_") and f"__g_{group}" in name]
        for group in group_names
    }
    coverage = {group: len(indices) for group, indices in grouped_columns.items()}
    reliability: dict[str, float] = {}
    for group, indices in grouped_columns.items():
        if not indices:
            reliability[group] = float("nan")
            continue
        per_feature = [_split_half_corr(raw[:, idx]) for idx in indices]
        reliability[group] = float(np.nanmedian(per_feature)) if np.isfinite(per_feature).any() else float("nan")
    return {
        "status": "ok" if replay["status"] == "ok" and all(count > 0 for count in coverage.values()) else "failed",
        "claim_boundary": "Frozen sensor-topographic measurement QC only; no cortical localization or EEG-MEG harmonization claim.",
        "h5_path": str(h5_path),
        "dataset_id": dataset_id,
        "transform_validation": replay,
        "frozen_group_coverage": coverage,
        "odd_even_split_half_reliability": reliability,
        "config_hash": _hash(config),
        "sector_contract_hash": _hash(groups),
        "random_seed": int(seed),
        "cross_modal": False,
    }
