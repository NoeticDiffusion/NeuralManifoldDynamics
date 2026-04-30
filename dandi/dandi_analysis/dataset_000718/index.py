from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Sequence

from dandi_analysis.contracts import ReadyNwbAsset, SessionIndexRow
from dandi_analysis.dataset_000718.io import safe_read_session_metadata


_SUBJECT_RE = re.compile(r"(?:^|[\\/])sub-([^/\\]+)")
_SESSION_RE = re.compile(r"ses-([^_./\\]+)")


def parse_subject_session(path: Path) -> tuple[str, str]:
    """Extract subject_id and session_label from a canonical NWB path.

    Falls back to filename stem if the expected tokens are absent.
    """
    s = str(path)
    m_sub = _SUBJECT_RE.search(s)
    subject_id = m_sub.group(1) if m_sub else "unknown"
    m_ses = _SESSION_RE.search(s)
    session_label = m_ses.group(1) if m_ses else path.stem
    return subject_id, session_label


def _session_id(subject_id: str, session_label: str) -> str:
    return f"{subject_id}__{session_label}"


def build_session_index(
    ready_assets: Sequence[ReadyNwbAsset],
    *,
    read_metadata: bool = True,
) -> list[SessionIndexRow]:
    """Turn a list of ready NWB assets into ``SessionIndexRow`` objects.

    When *read_metadata* is True the function opens each file to extract
    lightweight metadata.  Set to False for fast index builds in tests.
    """
    rows: list[SessionIndexRow] = []
    for asset in ready_assets:
        if not asset.is_ready:
            continue
        subject_id, session_label = parse_subject_session(asset.path)
        meta: dict = {}
        if read_metadata:
            try:
                meta = safe_read_session_metadata(asset.path)
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
            )
        )
    return rows
