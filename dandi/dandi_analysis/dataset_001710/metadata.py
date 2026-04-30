"""Lightweight metadata extraction for DANDI 001710 NWB files.

Targets: behavior channels, ophys interfaces, matrix shapes, sampling rate,
ROI counts, novel_arm, day, and embedded place-cell annotation keys.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from dandi_analysis.dataset_001710.io import open_nwb_readonly, _parse_annotation_blob


def extract_nwb_metadata(path: Path) -> dict[str, Any]:
    """Extract a comprehensive metadata dict from a 001710 NWB file."""
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
        meta["devices"] = list(nwb.devices.keys())
    except Exception:
        meta["devices"] = []

    # ---- Behavior channels ------------------------------------------------
    beh_channels: dict[str, list[str]] = {}
    try:
        beh_mod = nwb.processing["behavior"]
        for container_name, container in beh_mod.data_interfaces.items():
            try:
                beh_channels[container_name] = list(container.time_series.keys())
            except Exception:
                beh_channels[container_name] = []
    except Exception:
        pass
    if beh_channels:
        meta["behavior_channels"] = beh_channels

    # ---- Ophys interfaces and RRS shapes ----------------------------------
    rrs_info: list[dict[str, Any]] = []
    try:
        ophys = nwb.processing["ophys"]
        for iface_name, iface in ophys.data_interfaces.items():
            if hasattr(iface, "roi_response_series"):
                for rrs_name, rrs in iface.roi_response_series.items():
                    shape = list(rrs.data.shape)
                    rate = getattr(rrs, "rate", None)
                    rrs_info.append(
                        {
                            "interface": iface_name,
                            "name": rrs_name,
                            "shape": shape,
                            "n_frames": shape[0] if shape else None,
                            "n_rois": shape[1] if len(shape) > 1 else None,
                            "sampling_rate": float(rate) if rate else 15.4609375,
                        }
                    )
    except Exception:
        pass
    if rrs_info:
        meta["roi_response_series"] = rrs_info

    # ---- Imaging plane info -----------------------------------------------
    plane_info: dict[str, dict[str, Any]] = {}
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

    # ---- Annotation blob keys (trial_cell_data) ---------------------------
    blob = _parse_annotation_blob(nwb)
    if blob:
        meta["annotation_blob_keys"] = list(blob.keys())
        # Scalar fields that are safe to surface directly
        for key in ("day", "novel_arm", "mouse", "mux"):
            if key in blob:
                meta[f"blob_{key}"] = blob[key]

    return meta
