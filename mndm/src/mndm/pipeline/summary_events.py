"""Event/stage helpers extracted from summary.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


def infer_stage_from_bids_events(
    *,
    sub_frame: pd.DataFrame,
    index_df: Optional[pd.DataFrame],
    dataset_root: Path,
    lookup_rel_paths_by_file_value: Any,
    ctx_config: Mapping[str, Any],
    mnps_cfg: Mapping[str, Any],
    dataset_id: str,
) -> Tuple[Optional[np.ndarray], Optional[str], Optional[str], Optional[str]]:
    """Infer per-epoch sleep stage codes from BIDS ``*_events.tsv`` for a segment."""
    if "file" not in sub_frame.columns or len(sub_frame) == 0:
        return None, None, None, None

    filename = str(sub_frame["file"].iloc[0])
    rel_path = None
    try:
        rel_candidates = lookup_rel_paths_by_file_value(filename)
        if rel_candidates:
            rel_path = str(rel_candidates[0])
    except Exception:
        rel_path = None
    if rel_path is None and index_df is not None and "path" in index_df.columns:
        try:
            mask = index_df["path"].astype(str).str.endswith(filename)
            if mask.any():
                rel_path = str(index_df.loc[mask, "path"].iloc[0])
        except Exception:
            rel_path = None

    if not rel_path:
        return None, None, None, None

    file_path = dataset_root / rel_path
    if not file_path.exists():
        return None, None, None, None

    base_stem = file_path.stem
    base_core = base_stem[:-4] if base_stem.endswith("_eeg") else base_stem
    events_path = file_path.parent / f"{base_core}_events.tsv"
    if not events_path.exists():
        events_path = file_path.parent / "events.tsv"
        if not events_path.exists():
            return None, None, None, None

    try:
        events_df = pd.read_csv(events_path, sep="\t")
    except Exception:
        return None, None, None, None
    if events_df.empty:
        return None, None, None, None

    epoching = ctx_config.get("epoching", {}) if isinstance(ctx_config, Mapping) else {}
    ds_cfg = epoching.get("datasets", {}).get(dataset_id, {})
    sampling_cfg = ds_cfg.get("sampling", {})
    candidate_cols = sampling_cfg.get("stage_columns", ["stage_hum", "stage_ai", "stage", "sleep_stage"])
    if not isinstance(candidate_cols, list):
        candidate_cols = [str(candidate_cols)]

    stage_col = None
    for col in candidate_cols:
        if col in events_df.columns:
            stage_col = col
            break
    if stage_col is None:
        return None, None, None, None

    if "hum" in stage_col:
        stage_source = "hum"
    elif "ai" in stage_col:
        stage_source = "ai"
    else:
        stage_source = "consensus"

    if "onset" not in events_df.columns:
        return None, None, None, None

    onsets = pd.to_numeric(events_df["onset"], errors="coerce").to_numpy(dtype=float)
    dur_col = sampling_cfg.get("duration_column", "duration")
    if dur_col in events_df.columns:
        durations = pd.to_numeric(events_df[dur_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        durations = np.zeros_like(onsets)

    raw_stages = events_df[stage_col]
    codebook = mnps_cfg.get("stage_codebook", {}) if isinstance(mnps_cfg, Mapping) else {}
    if raw_stages.dtype.kind in {"U", "S", "O"} and codebook:
        mapping = {str(k).lower(): int(v) for k, v in codebook.items()}
        stages = np.array([mapping.get(str(v).strip().lower(), -1) for v in raw_stages], dtype=float)
    else:
        stages = pd.to_numeric(raw_stages, errors="coerce").to_numpy(dtype=float)

    if "t_start" in sub_frame.columns and "t_end" in sub_frame.columns:
        t_mid_arr = (
            pd.to_numeric(sub_frame["t_start"], errors="coerce")
            + pd.to_numeric(sub_frame["t_end"], errors="coerce")
        ).to_numpy(dtype=float) / 2.0
    else:
        dt = float(mnps_cfg["window_sec"]) * (1.0 - float(mnps_cfg["overlap"]))
        time_idx = np.arange(len(sub_frame)) * dt
        t_mid_arr = time_idx + 0.5 * float(mnps_cfg["window_sec"])

    out = np.full((t_mid_arr.shape[0],), -1, dtype=np.int16)
    valid = np.isfinite(onsets) & np.isfinite(durations) & np.isfinite(stages)
    onsets = onsets[valid]
    durations = durations[valid]
    stages = stages[valid]
    if onsets.size == 0:
        return None, None, None, None

    for o, d, s in zip(onsets, durations, stages):
        end = o + (d if d > 0 else 0.0)
        if end <= o:
            mask = np.isclose(t_mid_arr, o, atol=1e-3)
        else:
            mask = (t_mid_arr >= o) & (t_mid_arr < end)
        if mask.any():
            out[mask] = int(s)

    return out.astype(np.int8, copy=False), stage_source, stage_col, str(events_path)


def estimate_coverage_seconds(sub_frame: pd.DataFrame, dt_fallback: float) -> Tuple[float, str]:
    """Estimate coverage from timestamps when available; else fallback to len*dt."""
    if "t_start" in sub_frame.columns and "t_end" in sub_frame.columns:
        try:
            t_start = pd.to_numeric(sub_frame["t_start"], errors="coerce").to_numpy(dtype=float)
            t_end = pd.to_numeric(sub_frame["t_end"], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(t_start) & np.isfinite(t_end)
            if np.any(valid):
                starts = t_start[valid]
                ends = t_end[valid]
                span = float(np.nanmax(ends) - np.nanmin(starts))
                if np.isfinite(span) and span > 0:
                    return span, "timestamps_span"
        except Exception:
            pass
    return float(len(sub_frame) * float(dt_fallback)), "assumed_len_dt"


def map_events_to_labels(
    *,
    config: Mapping[str, Any],
    time: np.ndarray,
    window_start: np.ndarray,
    window_end: np.ndarray,
    events: Mapping[str, np.ndarray],
    dataset_id: str,
) -> Dict[str, np.ndarray]:
    """Map event timestamps to MNPS window-aligned binary labels (opt-in)."""
    ev_cfg = config.get("event_mapping", {}) if isinstance(config, Mapping) else {}
    enabled = bool(ev_cfg.get("enabled", False))
    ds_override = (ev_cfg.get("datasets", {}) or {}).get(dataset_id, {})
    if isinstance(ds_override, Mapping) and "enabled" in ds_override:
        enabled = bool(ds_override.get("enabled", enabled))
    if not enabled or not events:
        return {}

    tol = ev_cfg.get("tolerance_sec", None)
    if isinstance(ds_override, Mapping) and "tolerance_sec" in ds_override:
        tol = ds_override.get("tolerance_sec", tol)
    if tol is None:
        tol = float(max(np.diff(time).min(initial=0.0), 0.0)) / 2.0 if len(time) > 1 else 0.0

    labels: Dict[str, np.ndarray] = {}
    for name, vals in events.items():
        arr = np.zeros_like(time, dtype=bool)
        for v in np.asarray(vals, dtype=float):
            mask = (window_start <= v) & (window_end >= v)
            if not mask.any() and tol > 0:
                mask = np.abs(time - v) <= tol
            arr |= mask
        if arr.any():
            labels[name] = arr.astype(np.int8)
    return labels
