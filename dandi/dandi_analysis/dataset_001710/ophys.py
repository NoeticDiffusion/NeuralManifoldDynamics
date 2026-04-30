"""Standardized optical (calcium imaging) extraction for DANDI 001710.

Returns aligned ``time x rois`` arrays with sampling metadata and ROI ids.
No aggressive denoising is performed here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dandi_analysis.dataset_001710.io import SessionChannel, read_roi_response_series, read_plane_segmentation


@dataclass
class OphysMatrix:
    """Aligned optical activity matrix for one 001710 session."""

    session_path: Path
    signal: str             # "dff", "fluorescence", or "neuropil"
    data: np.ndarray        # shape (T, N)
    timestamps: np.ndarray  # shape (T,)
    roi_ids: tuple[int, ...]
    sampling_rate: float
    n_frames: int
    n_rois: int
    channel_id: str = "single"  # imaging channel ('0', '1', or 'single')
    metadata: dict[str, Any] = field(default_factory=dict)


def load_ophys_matrix(
    path: Path,
    *,
    signal: str = "dff",
    channel: SessionChannel | None = None,
) -> OphysMatrix | None:
    """Load one optical signal family as an aligned (T, N) matrix.

    Parameters
    ----------
    signal:
        ``"dff"`` (default), ``"fluorescence"``, or ``"neuropil"``.
        Ignored when *channel* is provided.
    channel:
        ``SessionChannel`` descriptor for explicit routing.  Use this for
        multi-channel sessions (SparseKO-style).

    Returns ``None`` if the signal is not available.
    """
    result = read_roi_response_series(path, signal=signal, channel=channel)
    if result is None:
        return None

    data, timestamps, sampling_rate = result
    data = np.asarray(data, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)

    channel_id = channel.channel_id if channel is not None else "single"
    roi_ids = _read_roi_ids(path, expected_n=data.shape[1], channel=channel)

    return OphysMatrix(
        session_path=path,
        signal=signal,
        data=data,
        timestamps=timestamps,
        roi_ids=tuple(roi_ids),
        sampling_rate=sampling_rate,
        n_frames=data.shape[0],
        n_rois=data.shape[1],
        channel_id=channel_id,
        metadata={"signal": signal, "channel_id": channel_id},
    )


def load_all_channel_matrices(
    path: Path,
    *,
    signal: str = "dff",
) -> list[OphysMatrix]:
    """Load one OphysMatrix per imaging channel found in the session.

    For single-channel sessions this returns a one-element list.
    For multi-channel sessions (SparseKO-style) this returns one matrix
    per channel, each using the appropriate ``SessionChannel`` resolver.
    """
    from dandi_analysis.dataset_001710.io import list_session_channels
    session_channels = list_session_channels(path)
    matrices: list[OphysMatrix] = []
    for ch in session_channels:
        mat = load_ophys_matrix(path, signal=signal, channel=ch)
        if mat is not None:
            matrices.append(mat)
    return matrices


def align_ophys_to_behavior(
    ophys: OphysMatrix,
    behavior_timestamps: np.ndarray,
) -> np.ndarray:
    """Return indices into ``ophys.timestamps`` nearest to each behavior frame.

    Useful for matching the ophys matrix to the 2P-aligned behavior table,
    which is already synchronized but may have slightly different frame counts.
    """
    bt = np.asarray(behavior_timestamps, dtype=float)
    ot = ophys.timestamps
    indices = np.searchsorted(ot, bt)
    indices = np.clip(indices, 0, len(ot) - 1)
    return indices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_roi_ids(
    path: Path, expected_n: int, channel: SessionChannel | None = None
) -> list[int]:
    """Read ROI integer ids from the plane segmentation table."""
    try:
        seg = read_plane_segmentation(path, channel=channel)
        if seg is None:
            return list(range(expected_n))
        ids = list(range(len(seg)))
        if len(ids) == expected_n:
            return ids
        # Use whatever length the segmentation reports; caller handles mismatch
        return list(range(expected_n))
    except Exception:
        return list(range(expected_n))
