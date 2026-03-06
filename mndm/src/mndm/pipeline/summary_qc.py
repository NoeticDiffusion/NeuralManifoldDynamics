"""QC output helpers extracted from summary.py."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from core.io import json_writer
from .robustness_helpers import build_qc_summary

logger = logging.getLogger(__name__)


def write_qc_files(
    *,
    target_dir: Path,
    dataset_label: str,
    ds_path: Path,
    sub_id: str,
    ses_id: Optional[str],
    sub_frame: pd.DataFrame,
    dt: float,
    ensemble_summary: Optional[Dict[str, Any]],
    robust_summary: Optional[Dict[str, Any]],
    dist_summary: Optional[Dict[str, Any]],
    tau_summary: Optional[Dict[str, Any]],
    tier2_jacobian: Optional[Dict[str, Any]],
    tier2_emmi: Optional[Dict[str, Any]],
    entropy_qc: Optional[Dict[str, Any]],
) -> None:
    """Write QC-related JSON files."""
    qc_rel_path = target_dir / "qc_reliability.json"
    try:
        rel_payload = {
            "dataset_id": dataset_label,
            "subject": sub_id,
            "session": ses_id,
            "reliability_axes": robust_summary.get("axes", {}).get("reliability", {}) if robust_summary else {},
            "reliability_subcoords": robust_summary.get("subcoords", {}).get("reliability", {}) if robust_summary else {},
        }
        json_writer.write_json_summary(rel_payload, qc_rel_path)
    except Exception:
        logger.exception("Failed to write qc_reliability.json for %s", dataset_label)

    qc_summary_path = target_dir / "qc_summary.json"
    try:
        qc_summary = build_qc_summary(
            dataset_label=dataset_label,
            ds_path=ds_path,
            sub_id=sub_id,
            ses_id=ses_id,
            sub_frame=sub_frame,
            dt=dt,
            ensemble_summary=ensemble_summary,
            robust_summary=robust_summary,
            dist_summary=dist_summary,
            entropy_qc=entropy_qc,
        )
        if tau_summary:
            qc_summary["tau_summary"] = tau_summary
        if tier2_jacobian:
            qc_summary["tier2_jacobian"] = tier2_jacobian
        if tier2_emmi:
            qc_summary["tier2_emmi"] = tier2_emmi
        json_writer.write_json_summary(qc_summary, qc_summary_path)
    except Exception:
        logger.exception("Failed to write qc_summary.json for %s", dataset_label)
