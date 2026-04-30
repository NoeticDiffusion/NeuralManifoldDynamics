from __future__ import annotations

from pathlib import Path
from typing import Any

from dandi_analysis.dataset_000336.io import open_nwb_readonly


def extract_nwb_metadata(path: Path) -> dict[str, Any]:
    """Extract a comprehensive metadata dict from a 000336 NWB file.

    Extended relative to 000718 to capture two-plane / somato-dendritic
    plane information and stimulus/trial structure.
    """
    try:
        with open_nwb_readonly(path) as nwb:
            return _extract(nwb, path)
    except Exception as exc:
        return {"error": str(exc), "path": str(path)}


def _extract(nwb: Any, path: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {"path": str(path)}

    for attr in ("session_description", "identifier", "experimenter",
                 "institution", "lab", "experiment_description"):
        val = getattr(nwb, attr, None)
        if val is not None:
            meta[attr] = val if isinstance(val, (str, list)) else str(val)

    start = getattr(nwb, "session_start_time", None)
    if start is not None:
        meta["session_start_time"] = str(start)

    subject = getattr(nwb, "subject", None)
    if subject is not None:
        meta["subject"] = {
            "subject_id": getattr(subject, "subject_id", None),
            "species": getattr(subject, "species", None),
            "sex": getattr(subject, "sex", None),
            "age": getattr(subject, "age", None),
        }

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

    # Per-imaging-plane depth and description (key for dendritic vs somatic)
    plane_info: dict[str, dict] = {}
    try:
        for plane_name, plane in nwb.imaging_planes.items():
            plane_info[plane_name] = {
                "description": getattr(plane, "description", None),
                "imaging_rate": getattr(plane, "imaging_rate", None),
                "indicator": getattr(plane, "indicator", None),
                "location": getattr(plane, "location", None),
            }
    except Exception:
        pass
    if plane_info:
        meta["imaging_plane_info"] = plane_info

    # Fluorescence ROI response series names and shapes
    rrs_info: list[dict] = []
    try:
        ophys = nwb.processing.get("ophys") or nwb.processing.get("Ophys")
        if ophys:
            for iface_name, iface in ophys.data_interfaces.items():
                if hasattr(iface, "roi_response_series"):
                    for rrs_name, rrs in iface.roi_response_series.items():
                        rrs_info.append({
                            "interface": iface_name,
                            "name": rrs_name,
                            "shape": list(rrs.data.shape),
                        })
    except Exception:
        pass
    if rrs_info:
        meta["roi_response_series"] = rrs_info

    # Stimulus / trial table
    try:
        interval_info = {}
        for name, table in nwb.intervals.items():
            try:
                interval_info[name] = {
                    "n_rows": len(table),
                    "columns": list(table.colnames),
                }
            except Exception:
                pass
        if interval_info:
            meta["interval_tables"] = interval_info
    except Exception:
        pass

    return meta
