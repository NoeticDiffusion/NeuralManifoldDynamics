"""Session identity normalization for DANDI 001710.

001710 paths look like:
    sub-Cre-1/sub-Cre-1_ses-ymaze-day0-scan0-novel-arm-1_behavior+ophys.nwb

The preferred ``session_label`` comes from the NWB-native ``session_id``
(e.g. ``ymaze_day0_scan0_novel_arm-1``) rather than the path-derived form.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Sequence

from dandi_analysis.contracts import ReadyNwbAsset, SessionIndexRow
from dandi_analysis.dataset_001710.io import open_nwb_readonly
from dandi_analysis.dataset_001710.metadata import extract_nwb_metadata


_SUBJECT_RE = re.compile(r"(?:^|[\\/])sub-([^/\\]+)")
_DAY_RE = re.compile(r"day(\d+)", re.IGNORECASE)
_NOVEL_ARM_RE = re.compile(r"novel-arm-(\d+)", re.IGNORECASE)
_SCAN_RE = re.compile(r"scan(\d+)", re.IGNORECASE)


def subject_group(subject_id: str) -> str:
    """Collapse a 001710 subject id into its genotype group.

    Examples
    --------
    ``Cre-1`` -> ``Cre``
    ``sub-Ctrl-9`` -> ``Ctrl``
    ``SparseKO-7`` -> ``SparseKO``
    """
    normalized = str(subject_id).strip()
    if normalized.startswith("sub-"):
        normalized = normalized[4:]

    token = normalized.split("-", 1)[0].lower()
    if token == "cre":
        return "Cre"
    if token == "ctrl":
        return "Ctrl"
    if token == "sparseko":
        return "SparseKO"
    return "Unknown"


def parse_subject_session(path: Path) -> tuple[str, str, int, int]:
    """Extract ``(subject_id, session_label, day, novel_arm)`` from a 001710 NWB path.

    ``session_label`` is the path-derived form; callers that have the NWB file
    open should prefer the NWB-native ``session_id`` via ``build_session_index``.
    """
    s = str(path)
    m_sub = _SUBJECT_RE.search(s)
    subject_id = m_sub.group(1) if m_sub else "unknown"

    m_day = _DAY_RE.search(s)
    day = int(m_day.group(1)) if m_day else -1

    m_arm = _NOVEL_ARM_RE.search(s)
    novel_arm = int(m_arm.group(1)) if m_arm else -1

    m_scan = _SCAN_RE.search(s)
    scan = int(m_scan.group(1)) if m_scan else 0

    session_label = (
        f"ymaze_day{day}_scan{scan}_novel_arm-{novel_arm}"
        if day >= 0
        else path.stem
    )
    return subject_id, session_label, day, novel_arm


def _nwb_session_id(path: Path) -> str | None:
    """Attempt to read the NWB-native session_id without raising."""
    try:
        with open_nwb_readonly(path) as nwb:
            sid = getattr(nwb, "session_id", None)
            return str(sid) if sid else None
    except Exception:
        return None


def build_session_index(
    ready_assets: Sequence[ReadyNwbAsset],
    *,
    read_metadata: bool = True,
) -> list[SessionIndexRow]:
    """Build a normalized session index from ready 001710 assets."""
    rows: list[SessionIndexRow] = []
    for asset in ready_assets:
        if not asset.is_ready:
            continue

        subject_id, session_label_path, day, novel_arm = parse_subject_session(asset.path)

        # Prefer NWB-native session_id when it differs from the path-derived label
        nwb_sid = _nwb_session_id(asset.path)
        session_label = nwb_sid if nwb_sid else session_label_path

        meta: dict = {}
        if read_metadata:
            try:
                meta = extract_nwb_metadata(asset.path)
            except Exception as exc:
                warnings.warn(f"metadata read failed for {asset.path}: {exc}")

        rows.append(
            SessionIndexRow(
                subject_id=subject_id,
                session_label=session_label,
                local_path=asset.path,
                size=asset.size,
                state="ready",
                description=meta.get("session_description", ""),
                start_time=meta.get("session_start_time", ""),
                processing_keys=tuple(meta.get("processing_keys", [])),
                interval_names=tuple(meta.get("interval_names", [])),
                imaging_planes=tuple(meta.get("imaging_planes", [])),
                metadata={
                    "day": day,
                    "novel_arm": novel_arm,
                    "blob_day": meta.get("blob_day"),
                    "blob_novel_arm": meta.get("blob_novel_arm"),
                    "annotation_blob_keys": meta.get("annotation_blob_keys", []),
                    "rrs_shapes": {
                        r["interface"]: r["shape"]
                        for r in meta.get("roi_response_series", [])
                    },
                },
            )
        )

    rows.sort(key=lambda r: (r.subject_id, r.metadata.get("day", -1)))
    return rows
