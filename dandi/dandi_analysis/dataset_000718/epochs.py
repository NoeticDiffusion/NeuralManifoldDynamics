from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Sequence

from dandi_analysis.contracts import OfflineWindow


_OFFLINE_KEYWORDS = re.compile(
    r"(sleep|rest|offline|nrem|rem|quiet|immobile|pause|inter.?trial)",
    re.IGNORECASE,
)
# sleep_state values in DANDI 000718 that represent offline / consolidation periods
_OFFLINE_STATES = {"quiet wake", "nrem", "rem", "sleep", "rest", "offline"}
_FALLBACK_MIN_DURATION_SEC = 60.0  # minimum window length for heuristic fallback


def extract_offline_windows(
    path: Path,
    session_id: str,
    *,
    min_duration_sec: float = 30.0,
) -> list[OfflineWindow]:
    """Extract candidate offline / rest windows from an NWB file.

    Strategy
    --------
    1. Inspect ``nwb.intervals`` for tables whose name matches offline keywords.
    2. Inspect all ``nwb.processing`` modules for ``TimeIntervals`` objects
       (e.g. ``processing['sleep']['SleepIntervals']`` in DANDI 000718).
       Within each such table, rows whose ``sleep_state`` value is in
       ``_OFFLINE_STATES`` are extracted as individual windows.
    3. If neither source produces windows, apply a heuristic fallback.
    """
    try:
        import pynwb

        with pynwb.NWBHDF5IO(str(path), mode="r", load_namespaces=True) as io:
            nwb = io.read()
            return _extract_from_nwb(nwb, session_id, min_duration_sec)
    except Exception as exc:
        warnings.warn(f"epoch extraction failed for {path}: {exc}")
        return []


def _extract_from_nwb(
    nwb: Any,
    session_id: str,
    min_duration_sec: float,
) -> list[OfflineWindow]:
    windows: list[OfflineWindow] = []

    # --- Pass 1: nwb.intervals ---
    try:
        interval_keys = list(nwb.intervals.keys())
    except Exception:
        interval_keys = []

    for key in interval_keys:
        if not _OFFLINE_KEYWORDS.search(key):
            continue
        try:
            table = nwb.intervals[key]
            windows.extend(
                _windows_from_table(table, key, session_id, min_duration_sec)
            )
        except Exception as exc:
            warnings.warn(f"could not read interval '{key}': {exc}")

    # --- Pass 2: nwb.processing[*][TimeIntervals] ---
    try:
        for mod_name, module in nwb.processing.items():
            for iface_name, iface in module.data_interfaces.items():
                if not _is_time_intervals(iface):
                    continue
                if not (_OFFLINE_KEYWORDS.search(mod_name)
                        or _OFFLINE_KEYWORDS.search(iface_name)):
                    # Still scan if there's a sleep_state column with offline values
                    if not _has_offline_state_col(iface):
                        continue
                label_prefix = f"{mod_name}.{iface_name}"
                windows.extend(
                    _windows_from_table(
                        iface, label_prefix, session_id, min_duration_sec
                    )
                )
    except Exception as exc:
        warnings.warn(f"processing module scan failed: {exc}")

    if not windows:
        windows = _heuristic_fallback(nwb, session_id, min_duration_sec)

    return windows


def _is_time_intervals(obj: Any) -> bool:
    """Return True if *obj* looks like a pynwb TimeIntervals / DynamicTable."""
    try:
        # pynwb.epoch.TimeIntervals is the correct location (not pynwb.misc)
        return hasattr(obj, "colnames") and (
            "start_time" in obj.colnames and "stop_time" in obj.colnames
        )
    except Exception:
        return False


def _has_offline_state_col(table: Any) -> bool:
    """Return True if the table has a ``sleep_state`` column with offline rows."""
    try:
        states = [s.lower() for s in _col_to_list(table, "sleep_state")]
        return any(s in _OFFLINE_STATES for s in states)
    except Exception:
        return False


def _windows_from_table(
    table: Any,
    label_prefix: str,
    session_id: str,
    min_duration_sec: float,
) -> list[OfflineWindow]:
    """Extract OfflineWindow objects from a single interval table.

    If the table has a ``sleep_state`` column, only rows whose state is in
    ``_OFFLINE_STATES`` are included.  Otherwise all rows are included.
    """
    starts = _col_to_list(table, "start_time")
    stops  = _col_to_list(table, "stop_time")
    states = _col_to_list(table, "sleep_state")

    has_state_filter = len(states) == len(starts)
    out: list[OfflineWindow] = []

    for i, (t0, t1) in enumerate(zip(starts, stops)):
        if has_state_filter:
            raw_state: str = str(states[i]).lower()
            if raw_state not in _OFFLINE_STATES:
                continue
            epoch_type = _classify_state(raw_state)
        else:
            epoch_type = _classify_epoch_type(label_prefix)

        duration = float(t1) - float(t0)
        if duration < min_duration_sec:
            continue

        out.append(
            OfflineWindow(
                session_id=session_id,
                label=f"{label_prefix}_{i}",
                start_sec=float(t0),
                stop_sec=float(t1),
                epoch_type=epoch_type,
            )
        )
    return out


def _classify_state(state: str) -> str:
    """Map a raw sleep_state string to a canonical epoch_type."""
    if "nrem" in state:
        return "NREM"
    if state == "rem":
        return "REM"
    if "quiet" in state:
        return "quiet_wake"
    if "sleep" in state:
        return "sleep"
    return "rest"


def _classify_epoch_type(name: str) -> str:
    name_lower = name.lower()
    if "nrem" in name_lower:
        return "NREM"
    if "rem" in name_lower and "nrem" not in name_lower:
        return "REM"
    if "sleep" in name_lower:
        return "sleep"
    if "rest" in name_lower or "quiet" in name_lower or "immobile" in name_lower:
        return "rest"
    return "offline"


def _col_to_list(table: Any, col: str) -> list:
    """Extract a column from a DynamicTable as a plain Python list."""
    try:
        return list(table[col].data[:])
    except Exception:
        try:
            return list(getattr(table, col)[:])
        except Exception:
            return []


def _heuristic_fallback(
    nwb: Any,
    session_id: str,
    min_duration_sec: float,
) -> list[OfflineWindow]:
    """Return a single window covering the full recording if long enough."""
    t_start: float | None = None
    t_stop: float | None = None

    try:
        # Try units spike times as a proxy for recording span
        spike_times = nwb.units["spike_times"]
        all_times: list[float] = []
        for row_times in spike_times.data:
            all_times.extend(row_times)
        if all_times:
            t_start = float(min(all_times))
            t_stop = float(max(all_times))
    except Exception:
        pass

    if t_start is None:
        return []

    duration = (t_stop or t_start) - t_start
    if duration < max(min_duration_sec, _FALLBACK_MIN_DURATION_SEC):
        return []

    return [
        OfflineWindow(
            session_id=session_id,
            label="heuristic_full_session",
            start_sec=t_start,
            stop_sec=t_stop or t_start,
            epoch_type="offline",
        )
    ]
