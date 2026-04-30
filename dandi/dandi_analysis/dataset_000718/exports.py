from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from dandi_analysis.contracts import (
    ActivityMatrix,
    OfflineWindow,
    QcIssue,
    SessionIndexRow,
)


def write_session_index_csv(rows: Sequence[SessionIndexRow], output: Path) -> None:
    """Write a CSV with one row per session."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject_id", "session_label", "local_path", "size", "state",
        "description", "start_time", "processing_keys", "interval_names",
        "imaging_planes",
    ]
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "subject_id": row.subject_id,
                "session_label": row.session_label,
                "local_path": str(row.local_path),
                "size": row.size,
                "state": row.state,
                "description": row.description,
                "start_time": row.start_time,
                "processing_keys": "|".join(row.processing_keys),
                "interval_names": "|".join(row.interval_names),
                "imaging_planes": "|".join(row.imaging_planes),
            })


def write_epoch_csv(windows: Sequence[OfflineWindow], output: Path) -> None:
    """Write a CSV with one row per offline window."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["session_id", "label", "start_sec", "stop_sec", "duration_sec", "epoch_type"]
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for w in windows:
            writer.writerow({
                "session_id": w.session_id,
                "label": w.label,
                "start_sec": w.start_sec,
                "stop_sec": w.stop_sec,
                "duration_sec": w.duration_sec,
                "epoch_type": w.epoch_type,
            })


def write_metadata_json(metadata: dict, output: Path) -> None:
    """Write a metadata dict as formatted JSON."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)


def write_qc_report(issues: Sequence[QcIssue], output: Path) -> None:
    """Write a markdown QC report."""
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# QC Report",
        "",
        f"Total issues: {len(issues)}",
        f"Errors: {sum(1 for i in issues if i.severity == 'error')}",
        f"Warnings: {sum(1 for i in issues if i.severity == 'warning')}",
        "",
        "| Path | Issue type | Severity | Message |",
        "| --- | --- | :---: | --- |",
    ]
    for issue in issues:
        lines.append(
            f"| `{issue.path}` | `{issue.issue_type}` | {issue.severity} | {issue.message} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_activity_npz(matrix: ActivityMatrix, output: Path) -> None:
    """Write an ActivityMatrix to a compressed NPZ archive."""
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output),
        data=np.array(matrix.data),
        timestamps=np.array(matrix.timestamps),
        unit_ids=np.array(matrix.unit_ids),
        sampling_rate=np.array([matrix.sampling_rate]),
        session_id=np.array([matrix.session_id]),
        window_label=np.array([matrix.window_label]),
    )
