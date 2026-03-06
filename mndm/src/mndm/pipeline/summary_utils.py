"""General helpers extracted from summary.py."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def apply_fd_censoring(
    sub_frame: pd.DataFrame,
    fd_thresh: float = 0.5,
    pad: int = 1,
    *,
    require_fd: bool = False,
    context_label: Optional[str] = None,
    logger: Optional[object] = None,
) -> pd.DataFrame:
    """Drop epochs with framewise displacement above threshold (and pad neighbours)."""
    if "framewise_displacement" not in sub_frame.columns:
        if require_fd:
            label = context_label or "fMRI segment"
            raise ValueError(
                f"FD censoring requires 'framewise_displacement' for {label}, but column is missing. "
                "Point pipeline to BIDS confounds (*desc-confounds_timeseries.tsv) and merge FD before summary."
            )
        if logger is not None:
            logger.info("FD censoring skipped: framewise_displacement column missing")
        return sub_frame

    df = sub_frame.copy()
    if "epoch_id" in df.columns:
        df = df.sort_values("epoch_id")
    fd = pd.to_numeric(df["framewise_displacement"], errors="coerce").fillna(0.0)
    bad = fd > fd_thresh
    if pad > 0:
        for k in range(1, pad + 1):
            bad |= bad.shift(k, fill_value=False) | bad.shift(-k, fill_value=False)
    kept = df.loc[~bad].copy()
    dropped = len(df) - len(kept)
    if dropped > 0 and logger is not None:
        logger.info(
            "FD censoring dropped %d/%d epochs (threshold=%.3f, pad=%d)",
            dropped,
            len(df),
            fd_thresh,
            pad,
        )
    return kept


def build_dir_suffix(
    ses_id: Optional[str],
    condition: Optional[str],
    task: Optional[str],
    run_id: Optional[str] = None,
    acq_id: Optional[str] = None,
) -> Optional[str]:
    """Build directory/filename suffix from session, condition, and task."""
    parts: list[str] = []
    if condition:
        parts.append(condition)
    elif ses_id:
        parts.append(ses_id)
    if task:
        parts.append(task)
    if run_id:
        parts.append(run_id)
    if acq_id:
        parts.append(acq_id)
    return "_".join(parts) if parts else None


def extract_time_bounds(
    sub_frame: pd.DataFrame,
    time: np.ndarray,
    window_sec: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-window start/end seconds, using t_start/t_end if available."""
    t_len = len(time)
    start = np.asarray(sub_frame["t_start"].to_numpy(), dtype=np.float32) if "t_start" in sub_frame.columns else None
    end = np.asarray(sub_frame["t_end"].to_numpy(), dtype=np.float32) if "t_end" in sub_frame.columns else None
    if start is None or start.shape[0] != t_len:
        start = (time - 0.5 * window_sec).astype(np.float32)
    if end is None or end.shape[0] != t_len:
        end = (time + 0.5 * window_sec).astype(np.float32)
    return start, end
