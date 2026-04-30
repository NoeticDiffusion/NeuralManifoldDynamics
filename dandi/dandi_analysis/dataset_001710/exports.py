"""Plain-output exports for DANDI 001710 analysis artifacts.

Writes human-inspectable CSV, JSON, NPZ, and Markdown files.  All functions
accept a ``output_dir`` argument and create it if necessary.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from dandi_analysis.contracts import SessionIndexRow
from dandi_analysis.dataset_001710.behavior import BehaviorTable
from dandi_analysis.dataset_001710.ophys import OphysMatrix
from dandi_analysis.dataset_001710.placecode import TuningCurveSet
from dandi_analysis.dataset_001710.remapping import SimilarityMatrix
from dandi_analysis.dataset_001710.trials import TrialRow, TrialTable


def export_session_index(
    rows: list[SessionIndexRow],
    output_dir: Path,
    *,
    filename: str = "session_index_001710.csv",
) -> Path:
    """Write a CSV session index."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "subject_id",
                "session_label",
                "day",
                "novel_arm",
                "size",
                "state",
                "n_frames",
                "n_rois",
                "local_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            rrs = row.metadata.get("rrs_shapes", {})
            n_frames = n_rois = ""
            for shape in rrs.values():
                if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                    n_frames = shape[0]
                    n_rois = shape[1]
                    break
            writer.writerow(
                {
                    "subject_id": row.subject_id,
                    "session_label": row.session_label,
                    "day": row.metadata.get("day", ""),
                    "novel_arm": row.metadata.get("novel_arm", ""),
                    "size": row.size,
                    "state": row.state,
                    "n_frames": n_frames,
                    "n_rois": n_rois,
                    "local_path": str(row.local_path),
                }
            )
    return out_path


def export_trial_table(
    trials: TrialTable,
    output_dir: Path,
    *,
    filename: str | None = None,
) -> Path:
    """Write a per-trial CSV table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = filename or f"trials_day{trials.day:02d}.csv"
    out_path = output_dir / fname
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "trial_id",
                "day",
                "block_id",
                "arm_label",
                "start_frame",
                "end_frame",
                "start_time",
                "end_time",
                "duration_sec",
                "reward_count",
                "valid",
                "notes",
            ],
        )
        writer.writeheader()
        for t in trials.trials:
            writer.writerow(
                {
                    "trial_id": t.trial_id,
                    "day": t.day,
                    "block_id": t.block_id,
                    "arm_label": t.arm_label,
                    "start_frame": t.start_frame,
                    "end_frame": t.end_frame,
                    "start_time": f"{t.start_time:.4f}",
                    "end_time": f"{t.end_time:.4f}",
                    "duration_sec": f"{t.duration_sec:.4f}",
                    "reward_count": t.reward_count,
                    "valid": int(t.valid),
                    "notes": t.notes,
                }
            )
    return out_path


def export_metadata_json(
    metadata: dict[str, Any],
    output_dir: Path,
    *,
    filename: str = "metadata_001710.json",
) -> Path:
    """Write a JSON metadata summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=_json_default)
    return out_path


def export_dff_matrix(
    ophys: OphysMatrix,
    output_dir: Path,
    *,
    filename: str | None = None,
) -> Path:
    """Write the dF matrix as a compressed NPZ file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = filename or f"dff_day{ophys.metadata.get('day', 'X')}.npz"
    out_path = output_dir / fname
    np.savez_compressed(
        str(out_path),
        data=ophys.data,
        timestamps=ophys.timestamps,
        roi_ids=np.array(ophys.roi_ids),
        sampling_rate=np.array([ophys.sampling_rate]),
    )
    return out_path


def export_tuning_summary(
    tc: TuningCurveSet,
    output_dir: Path,
    *,
    filename: str | None = None,
    label: str = "",
) -> Path:
    """Write tuning curves and occupancy as a CSV summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = filename or f"tuning_curves{'_' + label if label else ''}.csv"
    out_path = output_dir / fname

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        bin_headers = [f"bin_{i}" for i in range(tc.n_bins)]
        writer = csv.writer(fh)
        writer.writerow(["roi_id"] + bin_headers)
        for roi_i in range(tc.n_rois):
            row = [roi_i] + [
                "" if not np.isfinite(v) else f"{v:.6f}"
                for v in tc.tuning_curves[roi_i]
            ]
            writer.writerow(row)
    return out_path


def export_similarity_matrix(
    sim_mat: SimilarityMatrix,
    output_dir: Path,
    *,
    filename: str = "day_similarity_matrix.csv",
) -> Path:
    """Write the similarity matrix as a labelled CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([""] + sim_mat.labels)
        for i, label in enumerate(sim_mat.labels):
            row = [label] + [
                "" if not np.isfinite(v) else f"{v:.6f}"
                for v in sim_mat.matrix[i]
            ]
            writer.writerow(row)
    return out_path


def export_qc_report(
    report_text: str,
    output_dir: Path,
    *,
    filename: str = "qc_report.md",
) -> Path:
    """Write a QC report to a Markdown file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    out_path.write_text(report_text, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Robustness package exports
# ---------------------------------------------------------------------------

def export_robustness_csv(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    filename: str,
) -> Path:
    """Write a list of dicts to a CSV file.

    Keys of the first row are used as column headers.  All values are
    coerced to strings so the file is always human-readable.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    if not rows:
        out_path.write_text("", encoding="utf-8")
        return out_path

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: ("" if (isinstance(v, float) and not np.isfinite(v)) else v)
                 for k, v in row.items()}
            )
    return out_path


def export_null_json(
    null_data: list[dict[str, Any]],
    output_dir: Path,
    *,
    filename: str,
) -> Path:
    """Write null-test result dicts to a JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(null_data, fh, indent=2, default=_json_default)
    return out_path


def export_figure(fig: Any, output_dir: Path, *, filename: str) -> Path:
    """Save a matplotlib figure to *output_dir/filename* and close it.

    Parameters
    ----------
    fig:
        A ``matplotlib.figure.Figure`` instance.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass
    return out_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)
