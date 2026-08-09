"""Gated Phase 2 sidecar-to-H5 extension bridge.

The bridge is intentionally read-only with respect to Phase 2 sidecars: raw
signals are processed by the dedicated sidecar runner, then summarize embeds
the versioned product under ``/extensions`` when explicitly enabled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from mndm.features.sleep_eap.contracts import (
    NON_EVENT_RISK_V1_CONTRACT,
    PHASE_CONTINUOUS_V1_CONTRACT,
)


def _resolve_block(config: Mapping[str, Any], name: str, dataset_id: str) -> Dict[str, Any]:
    root = config.get(name, {}) if isinstance(config, Mapping) else {}
    if not isinstance(root, Mapping):
        return {"enabled": False}
    base = {key: value for key, value in root.items() if key != "datasets"}
    datasets = root.get("datasets", {})
    override = datasets.get(dataset_id, {}) if isinstance(datasets, Mapping) else {}
    if isinstance(override, Mapping):
        base.update(dict(override))
    base.setdefault("enabled", False)
    return base


def _table_payload(frame: pd.DataFrame) -> Dict[str, np.ndarray]:
    payload: Dict[str, np.ndarray] = {}
    for column in frame.columns:
        values = frame[column].to_numpy()
        if values.dtype.kind in {"O", "U"}:
            payload[str(column)] = np.asarray([str(value) for value in values], dtype=object)
        else:
            payload[str(column)] = values
    return payload


def _load_sidecar(directory: str | Path | None, filename: str) -> tuple[Optional[pd.DataFrame], str]:
    if not directory:
        return None, ""
    path = Path(str(directory)) / filename
    if not path.exists():
        return None, str(path)
    return pd.read_parquet(path), str(path)


def _sidecar_subject_id(subject_id: str) -> str:
    """Map BIDS-style subject labels to the bare Phase 2 sidecar naming key."""
    return str(subject_id).strip().removeprefix("sub-")


def build_phase2_extensions_for_run(
    *,
    config: Mapping[str, Any],
    dataset_id: str,
    subject_id: str,
) -> Dict[str, Any]:
    """Load enabled Phase 2 sidecars into extension/manifest/attribute payloads."""
    phase_cfg = _resolve_block(config, "phase_continuous", dataset_id)
    risk_cfg = _resolve_block(config, "non_event_risk", dataset_id)
    sidecar_subject_id = _sidecar_subject_id(subject_id)
    result: Dict[str, Any] = {
        "extensions": {},
        "attrs": {},
        "manifest": {
            "phase_continuous": {"enabled": bool(phase_cfg.get("enabled", False)), "status": "disabled"},
            "non_event_risk": {"enabled": bool(risk_cfg.get("enabled", False)), "status": "disabled"},
        },
    }

    if bool(phase_cfg.get("enabled", False)):
        contract = str(phase_cfg.get("contract", PHASE_CONTINUOUS_V1_CONTRACT))
        frame, source_path = _load_sidecar(
            phase_cfg.get("sidecar_dir"),
            f"{sidecar_subject_id}_phase_continuous_v1.parquet",
        )
        status = "available" if frame is not None else "sidecar_missing"
        result["manifest"]["phase_continuous"] = {
            "enabled": True,
            "status": status,
            "contract": contract,
            "source_path": source_path,
            "n_samples": int(len(frame)) if frame is not None else 0,
            "h5_path": "/extensions/phase_continuous_v1",
        }
        result["attrs"].update(
            {
                "phase_continuous_enabled": True,
                "phase_continuous_contract": contract,
                "phase_continuous_status": status,
            }
        )
        if frame is not None:
            payload = _table_payload(frame)
            payload["contract"] = contract
            payload["source_path"] = source_path
            result["extensions"]["phase_continuous_v1"] = payload

    if bool(risk_cfg.get("enabled", False)):
        contract = str(risk_cfg.get("contract", NON_EVENT_RISK_V1_CONTRACT))
        frame, source_path = _load_sidecar(
            risk_cfg.get("sidecar_dir"),
            f"{sidecar_subject_id}_non_event_risk_v1.parquet",
        )
        status = "available" if frame is not None else "sidecar_missing"
        result["manifest"]["non_event_risk"] = {
            "enabled": True,
            "status": status,
            "contract": contract,
            "source_path": source_path,
            "n_rows": int(len(frame)) if frame is not None else 0,
            "h5_path": "/extensions/non_event_risk_v1",
        }
        result["attrs"].update(
            {
                "non_event_risk_enabled": True,
                "non_event_risk_contract": contract,
                "non_event_risk_status": status,
            }
        )
        if frame is not None:
            payload = _table_payload(frame)
            payload["contract"] = contract
            payload["source_path"] = source_path
            result["extensions"]["non_event_risk_v1"] = payload
    return result
