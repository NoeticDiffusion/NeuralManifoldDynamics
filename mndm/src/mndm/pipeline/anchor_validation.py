"""Downstream validation helpers for noetic anchor analyses.

These utilities intentionally operate on flat sidecar tables rather than the
core H5 ingest contract.  They provide small, auditable summaries and null
controls for reviewer-facing analyses while keeping stronger dynamical claims
downstream-first.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_ANCHOR_COLUMNS = (
    "sympathetic_index",
    "vagal_index",
    "vascular_index",
    "pupil_arousal_index",
    "anchor_index",
)


def resolve_anchor_columns(
    frame: pd.DataFrame,
    anchor_columns: Optional[Sequence[str]] = None,
) -> list[str]:
    """Resolve the anchor columns present in a sidecar table."""
    candidates = list(anchor_columns) if anchor_columns is not None else list(DEFAULT_ANCHOR_COLUMNS)
    return [col for col in candidates if col in frame.columns]


def add_geometry_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with simple derived geometry summaries added when possible."""
    out = frame.copy()
    if {"m", "d", "e"}.issubset(out.columns):
        coords = out[["m", "d", "e"]].to_numpy(dtype=float)
        out["mnps_radius"] = np.linalg.norm(coords, axis=1)
    if {"m_dot", "d_dot", "e_dot"}.issubset(out.columns):
        deriv = out[["m_dot", "d_dot", "e_dot"]].to_numpy(dtype=float)
        out["mnps_speed"] = np.linalg.norm(deriv, axis=1)
    return out


def summarize_anchor_by_load(
    frame: pd.DataFrame,
    *,
    anchor_columns: Optional[Sequence[str]] = None,
    group_columns: Sequence[str] = ("dataset_id", "subject_id", "condition", "task_state_label", "task_load_n"),
) -> pd.DataFrame:
    """Compute reviewer-friendly grouped summaries for anchor/load analyses."""
    if frame.empty:
        return pd.DataFrame()
    work = add_geometry_columns(frame)
    anchors = resolve_anchor_columns(work, anchor_columns=anchor_columns)
    metrics = anchors + [col for col in ("mnps_radius", "mnps_speed") if col in work.columns]
    if not metrics:
        return pd.DataFrame()

    safe_groups = [col for col in group_columns if col in work.columns]
    if not safe_groups:
        safe_groups = ["condition"] if "condition" in work.columns else []
    grouped = work.groupby(safe_groups, dropna=False) if safe_groups else [((), work)]

    rows = []
    for key, part in grouped:
        row: Dict[str, Any] = {}
        if safe_groups:
            if not isinstance(key, tuple):
                key = (key,)
            row.update({col: key[idx] for idx, col in enumerate(safe_groups)})
        row["n_rows"] = int(len(part))
        for metric in metrics:
            values = pd.to_numeric(part[metric], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else np.nan
            row[f"{metric}_median"] = float(np.median(finite)) if finite.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def make_time_shift_null(
    frame: pd.DataFrame,
    *,
    anchor_columns: Optional[Sequence[str]] = None,
    group_columns: Sequence[str] = ("subject_id", "run_id"),
    shift: Optional[int] = None,
) -> pd.DataFrame:
    """Circularly shift anchor columns within subject/run groups."""
    if frame.empty:
        return frame.copy()
    work = frame.copy()
    anchors = resolve_anchor_columns(work, anchor_columns=anchor_columns)
    if not anchors:
        return work
    safe_groups = [col for col in group_columns if col in work.columns]
    if not safe_groups:
        safe_groups = ["subject_id"] if "subject_id" in work.columns else []
    if not safe_groups:
        safe_groups = ["__all__"]
        work["__all__"] = "all"

    for _, index in work.groupby(safe_groups, dropna=False).groups.items():
        positions = np.asarray(list(index), dtype=int)
        if positions.size <= 1:
            continue
        delta = int(shift) if shift is not None else max(1, positions.size // 2)
        for column in anchors:
            values = work.iloc[positions][column].to_numpy(copy=True)
            work.iloc[positions, work.columns.get_loc(column)] = np.roll(values, delta)

    work["null_control"] = "time_shift"
    if "__all__" in work.columns:
        work = work.drop(columns=["__all__"])
    return work


def make_subject_shuffle_null(
    frame: pd.DataFrame,
    *,
    anchor_columns: Optional[Sequence[str]] = None,
    subject_column: str = "subject_id",
) -> pd.DataFrame:
    """Deterministically shuffle anchor columns across subjects."""
    if frame.empty or subject_column not in frame.columns:
        return frame.copy()
    work = frame.copy()
    anchors = resolve_anchor_columns(work, anchor_columns=anchor_columns)
    if not anchors:
        return work

    subjects = [subject for subject in pd.unique(work[subject_column]) if pd.notna(subject)]
    if len(subjects) <= 1:
        return work
    rotated = {subjects[idx]: subjects[(idx + 1) % len(subjects)] for idx in range(len(subjects))}

    donor_frames = {
        subject: work.loc[work[subject_column] == rotated[subject], anchors].reset_index(drop=True)
        for subject in subjects
    }
    for subject in subjects:
        mask = work[subject_column] == subject
        positions = np.where(mask.to_numpy())[0]
        donor = donor_frames.get(subject)
        if donor is None or donor.empty:
            continue
        for col in anchors:
            donor_values = donor[col].to_numpy(copy=True)
            if donor_values.size == 0:
                continue
            if donor_values.size < positions.size:
                donor_values = np.resize(donor_values, positions.size)
            work.iloc[positions, work.columns.get_loc(col)] = donor_values[: positions.size]
    work["null_control"] = "subject_shuffle"
    return work


def summarize_null_controls(
    frame: pd.DataFrame,
    *,
    anchor_columns: Optional[Sequence[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Build the primary reviewer-facing anchor summary plus null-control variants."""
    anchors = resolve_anchor_columns(frame, anchor_columns=anchor_columns)
    return {
        "observed": summarize_anchor_by_load(frame, anchor_columns=anchors),
        "time_shift": summarize_anchor_by_load(
            make_time_shift_null(frame, anchor_columns=anchors),
            anchor_columns=anchors,
        ),
        "subject_shuffle": summarize_anchor_by_load(
            make_subject_shuffle_null(frame, anchor_columns=anchors),
            anchor_columns=anchors,
        ),
    }
