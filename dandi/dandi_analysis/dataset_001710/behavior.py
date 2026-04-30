"""Standardized behavior table extraction for DANDI 001710.

Produces a ``BehaviorTable`` dataclass (wrapping a dict of aligned arrays)
from either the 2P-aligned or full-resolution behavior container.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dandi_analysis.dataset_001710.io import SessionChannel, read_behavior_series


# Canonical column names used downstream.  Values in parentheses are the raw
# NWB series names in the 2P-aligned container.
_CHANNEL_MAP_2P = {
    "timestamps": None,          # synthetic from position timestamps
    "position": "position",
    "x_position": "x position",
    "y_position": "y position",
    "speed": "speed",
    "reward": "reward",
    "licks": "licks",
    "trial_number": "trial number",
    "trial_start": "trial start",
    "trial_end": "trial end",
    "arm": "left or right",
    "block": "block",
}

_CHANNEL_MAP_FULLRES = {
    "timestamps": None,
    "position": "position",
    "x_position": "x position",
    "y_position": "y position",
    "speed": "speed",
    "reward": "reward",
    "consummatory_licks": "consummatory licks",
    "non_consummatory_licks": "non-consummatory licks",
    "manual_rewards": "manual rewards",
    "trial_number": "trial number",
    "trial_start": "trial start",
    "trial_end": "trial end",
    "arm": "left or right",
    "block": "block",
    "rotary_encoder": "rotary encoder reading",
}


@dataclass
class BehaviorTable:
    """Aligned behavior arrays for one 001710 session."""

    session_path: Path
    source: str  # "2p" or "fullres"
    n_frames: int
    timestamps: np.ndarray          # shape (T,)
    channels: dict[str, np.ndarray] = field(default_factory=dict)
    channel_id: str = "single"      # imaging channel this table belongs to
    metadata: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.channels[key]

    def keys(self) -> list[str]:
        return list(self.channels.keys())


def load_behavior_table(
    path: Path,
    *,
    source: str = "2p",
    channel: SessionChannel | None = None,
) -> BehaviorTable | None:
    """Load and normalize a behavior table from one 001710 NWB file.

    Parameters
    ----------
    source:
        ``"2p"`` (default) or ``"fullres"``.  Ignored when *channel* is given.
    channel:
        ``SessionChannel`` descriptor.  When provided, routes to the correct
        behavior container automatically (multi-channel support).

    Returns ``None`` if the requested container is not present.
    """
    raw = read_behavior_series(path, source=source, channel=channel)
    if not raw:
        return None

    channel_id = channel.channel_id if channel is not None else "single"
    channel_map = _CHANNEL_MAP_2P if source == "2p" else _CHANNEL_MAP_FULLRES

    # Determine timestamps from the position channel (longest reliable series)
    ts_arr: np.ndarray | None = None
    for ts_key in ("position", "x position", "x_position"):
        if ts_key in raw:
            ts_arr, _ = raw[ts_key]
            break
    if ts_arr is None:
        # Fall back to first available series
        first_key = next(iter(raw))
        ts_arr, _ = raw[first_key]

    n = int(ts_arr.shape[0])

    channels: dict[str, np.ndarray] = {}
    for canonical, raw_name in channel_map.items():
        if canonical == "timestamps" or raw_name is None:
            continue
        if raw_name in raw:
            _, data = raw[raw_name]
            arr = np.asarray(data, dtype=float)
            # Trim or pad to match n if minor length mismatch
            if arr.shape[0] > n:
                arr = arr[:n]
            elif arr.shape[0] < n:
                pad = np.full(n - arr.shape[0], np.nan)
                arr = np.concatenate([arr, pad])
            channels[canonical] = arr

    return BehaviorTable(
        session_path=path,
        source=source,
        n_frames=n,
        timestamps=ts_arr,
        channels=channels,
        channel_id=channel_id,
        metadata={"raw_keys": list(raw.keys())},
    )
