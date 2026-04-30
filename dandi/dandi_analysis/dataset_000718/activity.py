from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

from dandi_analysis.contracts import ActivityMatrix, OfflineWindow


def build_activity_matrix(
    path: Path,
    session_id: str,
    window: OfflineWindow | None = None,
    *,
    z_score: bool = True,
) -> ActivityMatrix | None:
    """Build an ActivityMatrix from a fluorescence or spike container.

    Strategy (in order of preference):
    1. ``nwb.processing['ophys']['Fluorescence']`` — Ca-imaging fluorescence
    2. ``nwb.units`` spike times converted to binned firing rate

    Parameters
    ----------
    window:
        If provided, only the time span covered by the window is extracted.
        Pass ``None`` (default) to extract the **full session** — required for
        NeutralExposure sessions that have no offline interval annotations.

    Returns None if no compatible data is found.  Set *z_score* to True
    (default) to normalise each unit trace to zero mean, unit variance.
    """
    try:
        import pynwb

        with pynwb.NWBHDF5IO(str(path), mode="r", load_namespaces=True) as io:
            nwb = io.read()
            return _build(nwb, session_id, window, z_score=z_score)
    except Exception as exc:
        warnings.warn(f"activity matrix build failed for {path}: {exc}")
        return None


def build_full_session_activity_matrix(
    path: Path,
    session_id: str,
    *,
    z_score: bool = True,
) -> ActivityMatrix | None:
    """Build an ActivityMatrix for the complete session (no window restriction).

    Convenience wrapper around :func:`build_activity_matrix` with
    ``window=None``.  Use this for NeutralExposure (encoding) sessions that
    lack offline interval annotations.
    """
    return build_activity_matrix(path, session_id, window=None, z_score=z_score)
    try:
        import pynwb

        with pynwb.NWBHDF5IO(str(path), mode="r", load_namespaces=True) as io:
            nwb = io.read()
            return _build(nwb, session_id, window, z_score=z_score)
    except Exception as exc:
        warnings.warn(f"activity matrix build failed for {path}: {exc}")
        return None


def _build(
    nwb: Any,
    session_id: str,
    window: OfflineWindow | None,
    *,
    z_score: bool,
) -> ActivityMatrix | None:
    # --- Try fluorescence first ---
    mat = _try_fluorescence(nwb, session_id, window, z_score=z_score)
    if mat is not None:
        return mat

    # --- Fall back to spike-rate ---
    mat = _try_spikes(nwb, session_id, window, z_score=z_score)
    return mat


# ---------------------------------------------------------------------------
# Fluorescence path
# ---------------------------------------------------------------------------

def _try_fluorescence(
    nwb: Any,
    session_id: str,
    window: OfflineWindow | None,
    *,
    z_score: bool,
) -> ActivityMatrix | None:
    try:
        ophys = nwb.processing["ophys"]
        fluor = ophys["Fluorescence"]
        # Prefer deconvolved or denoised over raw baseline traces
        rrs_keys = list(fluor.roi_response_series.keys())
        preferred = ["Deconvolved", "deconvolved", "Denoised", "denoised",
                     "dF_F", "dff", "DfOverF"]
        roi_key = next(
            (k for k in preferred if k in rrs_keys),
            rrs_keys[0],
        )
        rrs = fluor.roi_response_series[roi_key]
        data = np.array(rrs.data[:], dtype=float)  # (T, N)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        timestamps = _get_timestamps(rrs, data.shape[0])
        unit_ids = tuple(str(i) for i in range(data.shape[1]))

        if window is not None:
            mask = (timestamps >= window.start_sec) & (timestamps <= window.stop_sec)
            data = data[mask]
            timestamps = timestamps[mask]

        if z_score:
            data = _zscore(data)

        sr = _infer_sampling_rate(timestamps)
        return ActivityMatrix(
            session_id=session_id,
            data=data,
            unit_ids=unit_ids,
            timestamps=timestamps,
            sampling_rate=sr,
            window_label=window.label if window else "",
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Spike-rate path
# ---------------------------------------------------------------------------

_BIN_SIZE_SEC = 0.1  # 100 ms bins


def _try_spikes(
    nwb: Any,
    session_id: str,
    window: OfflineWindow | None,
    *,
    z_score: bool,
) -> ActivityMatrix | None:
    try:
        units = nwb.units
        n_units = len(units)
        if n_units == 0:
            return None

        all_spikes: list[np.ndarray] = []
        for i in range(n_units):
            try:
                st = np.array(units["spike_times"][i], dtype=float)
            except Exception:
                st = np.array([], dtype=float)
            all_spikes.append(st)

        flat = np.concatenate([s for s in all_spikes if len(s) > 0])
        if flat.size == 0:
            return None

        t_start = float(flat.min())
        t_stop = float(flat.max())

        if window is not None:
            t_start = max(t_start, window.start_sec)
            t_stop = min(t_stop, window.stop_sec)

        if t_stop <= t_start:
            return None

        edges = np.arange(t_start, t_stop + _BIN_SIZE_SEC, _BIN_SIZE_SEC)
        timestamps = (edges[:-1] + edges[1:]) / 2.0
        n_bins = len(timestamps)

        data = np.zeros((n_bins, n_units), dtype=float)
        for j, spikes in enumerate(all_spikes):
            if len(spikes) == 0:
                continue
            counts, _ = np.histogram(spikes, bins=edges)
            data[:, j] = counts / _BIN_SIZE_SEC  # convert to Hz

        unit_ids = tuple(str(i) for i in range(n_units))

        if z_score:
            data = _zscore(data)

        return ActivityMatrix(
            session_id=session_id,
            data=data,
            unit_ids=unit_ids,
            timestamps=timestamps,
            sampling_rate=1.0 / _BIN_SIZE_SEC,
            window_label=window.label if window else "",
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_timestamps(rrs: Any, n_samples: int) -> np.ndarray:
    try:
        ts = np.array(rrs.timestamps[:], dtype=float)
        if len(ts) == n_samples:
            return ts
    except Exception:
        pass
    rate = float(getattr(rrs, "rate", None) or 1.0)
    starting = float(getattr(rrs, "starting_time", None) or 0.0)
    return np.linspace(starting, starting + n_samples / rate, n_samples)


def _infer_sampling_rate(timestamps: np.ndarray) -> float:
    if len(timestamps) < 2:
        return 1.0
    dt = float(np.median(np.diff(timestamps)))
    return 1.0 / dt if dt > 0 else 1.0


def _zscore(data: np.ndarray) -> np.ndarray:
    mu = data.mean(axis=0, keepdims=True)
    sd = data.std(axis=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (data - mu) / sd
