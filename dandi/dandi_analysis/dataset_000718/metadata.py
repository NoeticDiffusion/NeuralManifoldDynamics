from __future__ import annotations

from pathlib import Path
from typing import Any

from dandi_analysis.dataset_000718.io import open_nwb_readonly


def extract_nwb_metadata(path: Path) -> dict[str, Any]:
    """Extract a comprehensive metadata dict from an NWB file.

    Returns an empty dict on any failure.  The dict is suitable for
    JSON serialisation (all values are scalars, lists, or nested dicts).
    """
    try:
        with open_nwb_readonly(path) as nwb:
            return _extract(nwb, path)
    except Exception as exc:
        return {"error": str(exc), "path": str(path)}


def _extract(nwb: Any, path: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {"path": str(path)}

    for attr in (
        "session_description",
        "identifier",
        "experimenter",
        "institution",
        "lab",
        "experiment_description",
    ):
        val = getattr(nwb, attr, None)
        if val is not None:
            meta[attr] = val if isinstance(val, (str, list)) else str(val)

    start = getattr(nwb, "session_start_time", None)
    if start is not None:
        meta["session_start_time"] = str(start)

    # Subject
    subject = getattr(nwb, "subject", None)
    if subject is not None:
        meta["subject"] = {
            "subject_id": getattr(subject, "subject_id", None),
            "species": getattr(subject, "species", None),
            "sex": getattr(subject, "sex", None),
            "age": getattr(subject, "age", None),
        }

    # Top-level groups
    try:
        meta["acquisitions"] = list(nwb.acquisition.keys())
    except Exception:
        meta["acquisitions"] = []

    try:
        meta["processing_keys"] = list(nwb.processing.keys())
    except Exception:
        meta["processing_keys"] = []

    try:
        meta["interval_names"] = list(nwb.intervals.keys())
    except Exception:
        meta["interval_names"] = []

    try:
        meta["imaging_planes"] = list(nwb.imaging_planes.keys())
    except Exception:
        meta["imaging_planes"] = []

    try:
        meta["electrode_groups"] = list(nwb.electrode_groups.keys())
    except Exception:
        meta["electrode_groups"] = []

    try:
        meta["devices"] = list(nwb.devices.keys())
    except Exception:
        meta["devices"] = []

    # Units table info
    units = getattr(nwb, "units", None)
    if units is not None:
        try:
            meta["units_count"] = len(units)
            meta["units_columns"] = list(units.colnames)
        except Exception:
            pass

    return meta
