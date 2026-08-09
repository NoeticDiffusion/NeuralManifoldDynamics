"""Continuous state and stimulation annotations for DANDI 000458 LFP."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .nwb_units import read_trials


STATE_SCHEMA = "mndm.lfp_state_annotations.v1"


@dataclass(frozen=True)
class LfpStateContract:
    """Continuous behavioral blocks and sparse stimulation events."""

    blocks: pd.DataFrame
    stim_events: pd.DataFrame
    time_origin_sec: float
    recording_stop_sec: float
    pharmacological_intervals: pd.DataFrame = field(default_factory=pd.DataFrame)


def read_000458_state_contract(nwb_path: str | Path, series_path: str) -> LfpStateContract:
    """Read continuous state blocks from trial transitions, not point-event labels."""
    path = Path(nwb_path)
    trials = read_trials(path)
    required = {"start_time", "behavioral_epoch"}
    if not required.issubset(trials):
        raise ValueError(f"000458 trials lack required columns: {sorted(required - set(trials))}")
    events = trials.copy()
    events["start_time"] = pd.to_numeric(events["start_time"], errors="coerce")
    events = events.loc[events["start_time"].notna() & events["behavioral_epoch"].notna()].sort_values("start_time")
    events["behavioral_epoch"] = events["behavioral_epoch"].astype(str).str.lower().str.strip()
    origin, recording_stop = _series_clock(path, series_path)
    transitions = events.loc[events["behavioral_epoch"].ne(events["behavioral_epoch"].shift())]
    rows: list[dict[str, Any]] = []
    starts = transitions["start_time"].to_numpy(dtype=float)
    labels = transitions["behavioral_epoch"].tolist()
    for index, (start, label) in enumerate(zip(starts, labels, strict=True)):
        stop = starts[index + 1] if index + 1 < len(starts) else recording_stop
        rows.append({"state": label, "start_sec": float(start), "stop_sec": float(stop)})
    event_cols = [name for name in ("start_time", "estim_current", "stimulus_type", "stimulus_description") if name in events]
    return LfpStateContract(
        blocks=pd.DataFrame(rows),
        stim_events=events[event_cols].reset_index(drop=True),
        time_origin_sec=origin,
        recording_stop_sec=recording_stop,
        pharmacological_intervals=_read_pharmacological_intervals(path),
    )


def annotate_lfp_features(
    features: pd.DataFrame,
    contract: LfpStateContract,
    *,
    stim_guard_sec: float = 1.0,
    exclude_induction: bool = False,
) -> pd.DataFrame:
    """Join feature windows to state blocks and nearest-stimulation metadata."""
    result = features.copy()
    starts = pd.to_numeric(result["t_start"], errors="coerce").to_numpy(dtype=float) + contract.time_origin_sec
    ends = pd.to_numeric(result["t_end"], errors="coerce").to_numpy(dtype=float) + contract.time_origin_sec
    midpoint = (starts + ends) / 2.0
    result["lfp_epoch_midpoint_sec"] = midpoint
    result["lfp_behavioral_state"] = _assign_blocks(midpoint, contract.blocks)
    stim_times = contract.stim_events["start_time"].to_numpy(dtype=float)
    nearest_index = np.searchsorted(stim_times, midpoint)
    left = np.clip(nearest_index - 1, 0, len(stim_times) - 1)
    right = np.clip(nearest_index, 0, len(stim_times) - 1)
    closest = np.where(np.abs(midpoint - stim_times[left]) <= np.abs(midpoint - stim_times[right]), left, right)
    distance = np.abs(midpoint - stim_times[closest])
    result["lfp_nearest_stim_distance_sec"] = distance
    result["lfp_stim_adjacent"] = distance <= float(stim_guard_sec)
    result["lfp_stim_contains_onset"] = _contains_event(starts, ends, stim_times)
    if "estim_current" in contract.stim_events:
        result["lfp_nearest_estim_current"] = pd.to_numeric(
            contract.stim_events["estim_current"], errors="coerce"
        ).to_numpy(dtype=float)[closest]
    result["lfp_interstim_primary"] = result["lfp_behavioral_state"].notna() & ~result["lfp_stim_adjacent"]
    if exclude_induction:
        induction = contract.pharmacological_intervals
        mask = np.zeros(len(midpoint), dtype=bool)
        for row in induction.loc[induction.get("tag", pd.Series(dtype=str)).eq("isoflurane_induction")].itertuples(index=False):
            mask |= (midpoint >= row.start_sec) & (midpoint < row.stop_sec)
        result["lfp_induction_excluded"] = mask
        result["lfp_interstim_primary"] &= ~result["lfp_induction_excluded"]
    return result


def write_annotated_lfp_features(
    features_path: str | Path,
    nwb_path: str | Path,
    series_path: str,
    output_path: str | Path,
    *,
    stim_guard_sec: float = 1.0,
    exclude_induction: bool = False,
) -> dict[str, Any]:
    """Annotate a feature parquet and write a compact coverage sidecar."""
    frame = pd.read_parquet(features_path)
    contract = read_000458_state_contract(nwb_path, series_path)
    annotated = annotate_lfp_features(
        frame, contract, stim_guard_sec=stim_guard_sec, exclude_induction=exclude_induction
    )
    annotated["lfp_running_speed_mean"] = _running_speed_epoch_mean(
        Path(nwb_path),
        annotated["lfp_epoch_midpoint_sec"].to_numpy(dtype=float),
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    annotated.to_parquet(output, index=False)
    coverage = {
        "schema": STATE_SCHEMA,
        "time_origin_sec": contract.time_origin_sec,
        "stim_guard_sec": stim_guard_sec,
        "state_blocks": contract.blocks.to_dict(orient="records"),
        "n_epochs": len(annotated),
        "n_interstim_primary": int(annotated["lfp_interstim_primary"].sum()),
        "state_counts": {
            str(key): int(value) for key, value in annotated["lfp_behavioral_state"].value_counts(dropna=False).items()
        },
    }
    output.with_suffix(".state_coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    return coverage


def _series_clock(path: Path, series_path: str) -> tuple[float, float]:
    import h5py

    with h5py.File(path, "r") as handle:
        group = handle[series_path.strip("/")]
        timestamps = group.get("timestamps")
        if timestamps is not None:
            return float(timestamps[0]), float(timestamps[-1])
        starting = group.get("starting_time")
        if starting is None or "rate" not in starting.attrs:
            raise ValueError(f"Cannot determine recording clock for {series_path!r}")
        rate = float(starting.attrs["rate"])
        return float(starting[()]), float(starting[()]) + (len(group["data"]) - 1) / rate


def _assign_blocks(midpoint: np.ndarray, blocks: pd.DataFrame) -> np.ndarray:
    labels = np.full(len(midpoint), None, dtype=object)
    for row in blocks.itertuples(index=False):
        labels[(midpoint >= row.start_sec) & (midpoint < row.stop_sec)] = row.state
    return labels


def _contains_event(starts: np.ndarray, ends: np.ndarray, event_times: np.ndarray) -> np.ndarray:
    index = np.searchsorted(event_times, starts, side="left")
    return (index < len(event_times)) & (event_times[np.clip(index, 0, len(event_times) - 1)] < ends)


def _running_speed_epoch_mean(path: Path, midpoint_sec: np.ndarray) -> np.ndarray:
    """Sample the 100-Hz wheel trace at each feature midpoint."""
    import h5py

    location = "processing/behavior/BehavioralTimeSeries/running_speed"
    try:
        with h5py.File(path, "r") as handle:
            group = handle[location]
            data = np.asarray(group["data"][:], dtype=float)
            if "timestamps" in group:
                timestamps = np.asarray(group["timestamps"][:], dtype=float)
                return np.interp(midpoint_sec, timestamps, data, left=np.nan, right=np.nan)
            start = float(group["starting_time"][()])
            rate = float(group["starting_time"].attrs["rate"])
            index = np.rint((midpoint_sec - start) * rate).astype(int)
            output = np.full(len(index), np.nan)
            valid = (index >= 0) & (index < len(data))
            output[valid] = data[index[valid]]
            return output
    except (KeyError, OSError, ValueError):
        return np.full(len(midpoint_sec), np.nan)


def _read_pharmacological_intervals(path: Path) -> pd.DataFrame:
    """Read tagged NWB epochs without assuming a fixed session timeline."""
    try:
        from pynwb import NWBHDF5IO

        with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
            table = io.read().intervals.get("epochs")
            if table is None:
                return pd.DataFrame(columns=["tag", "start_sec", "stop_sec"])
            frame = table.to_dataframe().reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["tag", "start_sec", "stop_sec"])
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        tags = getattr(row, "tags", [])
        if isinstance(tags, (str, bytes)):
            tags = [tags]
        for tag in tags or []:
            text = tag.decode("utf-8", errors="replace") if isinstance(tag, bytes) else str(tag)
            rows.append({"tag": text.lower(), "start_sec": float(row.start_time), "stop_sec": float(row.stop_time)})
    return pd.DataFrame(rows, columns=["tag", "start_sec", "stop_sec"])
