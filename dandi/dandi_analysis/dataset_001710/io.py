"""Thin read-only NWB wrappers for DANDI 001710.

Handles both single-channel sessions (sub-Cre-*, sub-Ctrl-*) and
multi-channel sessions (sub-SparseKO-*) through resolver functions that
discover the correct interface names at runtime rather than hardcoding them.

Key entry point for callers: ``resolve_session_channels(nwb)`` returns a list
of ``SessionChannel`` descriptors, one per imaging channel.  All other loaders
accept an optional ``channel`` argument that routes to the right interfaces.
"""
from __future__ import annotations

import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator


# ---------------------------------------------------------------------------
# SessionChannel descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionChannel:
    """All NWB interface names needed to analyse one imaging channel."""

    channel_id: str              # '0', '1', ... or 'single' for standard files
    behavior_container: str      # NWB interface name inside processing['behavior']
    ophys_interface: str         # NWB interface name inside processing['ophys']
    ophys_series: str            # series name inside ophys_interface
    imaging_plane: str           # key in nwb.imaging_planes
    segmentation_interface: str  # NWB interface name for ImageSegmentation
    is_multichannel: bool        # True for SparseKO-style dual-plane files


# ---------------------------------------------------------------------------
# Core NWB context manager
# ---------------------------------------------------------------------------

@contextmanager
def open_nwb_readonly(path: Path) -> Generator[Any, None, None]:
    """Context manager that yields a read-only NWB file object."""
    import pynwb
    io = pynwb.NWBHDF5IO(str(path), mode="r", load_namespaces=True)
    try:
        yield io.read()
    finally:
        io.close()


# ---------------------------------------------------------------------------
# Channel resolver
# ---------------------------------------------------------------------------

def resolve_session_channels(nwb: Any) -> list[SessionChannel]:
    """Inspect an open NWB object and return one ``SessionChannel`` per channel.

    For standard single-channel sessions this returns a one-element list with
    ``channel_id='single'``.  For multi-channel sessions (SparseKO-style) it
    returns one entry per detected channel index.
    """
    try:
        beh_mod = nwb.processing["behavior"]
        beh_ifaces = list(beh_mod.data_interfaces.keys())
    except (KeyError, AttributeError):
        beh_ifaces = []

    try:
        ophys_mod = nwb.processing["ophys"]
        ophys_ifaces = list(ophys_mod.data_interfaces.keys())
    except (KeyError, AttributeError):
        ophys_ifaces = []

    try:
        plane_names = list(nwb.imaging_planes.keys())
    except (KeyError, AttributeError):
        plane_names = []

    # Detect multi-channel by presence of 'channel N df' in ophys interfaces
    channel_ids = _detect_channel_ids(ophys_ifaces, beh_ifaces, plane_names)
    multichannel = len(channel_ids) > 1 or (
        len(channel_ids) == 1 and channel_ids[0] != "single"
    )

    channels: list[SessionChannel] = []
    for ch_id in channel_ids:
        beh_cont = _resolve_behavior_container(beh_ifaces, ch_id)
        oph_iface, oph_series = _resolve_ophys_dff(ophys_ifaces, ch_id)
        plane = _resolve_imaging_plane(plane_names, ch_id)
        seg = _resolve_segmentation(ophys_ifaces, ch_id)
        channels.append(SessionChannel(
            channel_id=ch_id,
            behavior_container=beh_cont,
            ophys_interface=oph_iface,
            ophys_series=oph_series,
            imaging_plane=plane,
            segmentation_interface=seg,
            is_multichannel=multichannel,
        ))
    return channels


def list_session_channels(path: Path) -> list[SessionChannel]:
    """Open *path* and return its channel descriptors without raising."""
    try:
        with open_nwb_readonly(path) as nwb:
            return resolve_session_channels(nwb)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Resolver helpers
# ---------------------------------------------------------------------------

def _detect_channel_ids(
    ophys_ifaces: list[str],
    beh_ifaces: list[str],
    plane_names: list[str],
) -> list[str]:
    """Return an ordered list of channel identifiers found in this session."""
    # Look for 'channel N df' pattern (SparseKO-style)
    found: set[str] = set()
    for name in ophys_ifaces:
        m = re.match(r"channel\s+(\d+)\s+df$", name.strip(), re.IGNORECASE)
        if m:
            found.add(m.group(1))

    if found:
        return sorted(found)

    # Fallback: look for channelN in imaging plane names
    for name in plane_names:
        m = re.search(r"channel(\d+)", name, re.IGNORECASE)
        if m:
            found.add(m.group(1))

    if found:
        return sorted(found)

    return ["single"]


def _resolve_behavior_container(beh_ifaces: list[str], ch_id: str) -> str:
    """Pick the best 2P-aligned behavior container for this channel."""
    if ch_id == "single":
        # Standard: prefer '2P-aligned behavior' exactly
        if "2P-aligned behavior" in beh_ifaces:
            return "2P-aligned behavior"
        # Fallback: any non-fullres container
        for name in beh_ifaces:
            nl = name.lower()
            if "2p" in nl and "full" not in nl:
                return name
        return beh_ifaces[0] if beh_ifaces else "2P-aligned behavior"
    else:
        # Multi-channel: look for 'channel_N' suffix or 'channel N' in name
        for name in beh_ifaces:
            nl = name.lower()
            if ("2p" in nl or "2p-aligned" in nl) and (
                f"channel_{ch_id}" in nl.replace(" ", "_")
                or f"channel {ch_id}" in nl
                or nl.endswith(f"_{ch_id}")
            ):
                return name
        # Fallback: first 2P container
        for name in beh_ifaces:
            if "2p" in name.lower() and "full" not in name.lower():
                return name
        return beh_ifaces[0] if beh_ifaces else f"2P-aligned behavior channel_{ch_id}"


def _resolve_ophys_dff(
    ophys_ifaces: list[str], ch_id: str
) -> tuple[str, str]:
    """Return (interface_name, series_name) for the preferred dF signal."""
    if ch_id == "single":
        # Standard: 'dF' interface with 'dF' series
        if "dF" in ophys_ifaces:
            return "dF", "dF"
        # Fallback: any interface that looks like dff but not deconvolved
        for name in ophys_ifaces:
            nl = name.lower()
            if "df" in nl and "deconv" not in nl and "neuropil" not in nl:
                return name, name
        # Last resort
        for name in ophys_ifaces:
            if "deconv" not in name.lower() and "neuropil" not in name.lower():
                return name, name
        return ophys_ifaces[0], ophys_ifaces[0]
    else:
        # Multi-channel: 'channel N df' (exact match preferred, not deconvolved)
        target = f"channel {ch_id} df"
        for name in ophys_ifaces:
            if name.lower().strip() == target:
                return name, name
        # Partial match
        for name in ophys_ifaces:
            nl = name.lower()
            if f"channel {ch_id}" in nl and "df" in nl and "deconv" not in nl and "neuropil" not in nl:
                return name, name
        return ophys_ifaces[0], ophys_ifaces[0]


def _resolve_imaging_plane(plane_names: list[str], ch_id: str) -> str:
    """Pick the imaging plane for this channel."""
    if ch_id == "single":
        if "ImagingPlane" in plane_names:
            return "ImagingPlane"
        return plane_names[0] if plane_names else "ImagingPlane"
    else:
        # Look for 'ImagingPlaneChannelN' or 'channel_N' in name
        for name in plane_names:
            if re.search(rf"channel\s*{ch_id}", name, re.IGNORECASE):
                return name
        return plane_names[0] if plane_names else f"ImagingPlaneChannel{ch_id}"


def _resolve_segmentation(ophys_ifaces: list[str], ch_id: str) -> str:
    """Pick the ImageSegmentation interface for this channel."""
    if ch_id == "single":
        if "ImageSegmentation" in ophys_ifaces:
            return "ImageSegmentation"
        for name in ophys_ifaces:
            if "segmentation" in name.lower() and "channel" not in name.lower():
                return name
        for name in ophys_ifaces:
            if "segmentation" in name.lower():
                return name
        return "ImageSegmentation"
    else:
        for name in ophys_ifaces:
            if "segmentation" in name.lower() and re.search(
                rf"channel\s*{ch_id}", name, re.IGNORECASE
            ):
                return name
        for name in ophys_ifaces:
            if "segmentation" in name.lower():
                return name
        return f"ImageSegmentationChannel{ch_id}"


# ---------------------------------------------------------------------------
# Annotation blob
# ---------------------------------------------------------------------------

def read_trial_annotation_blob(path: Path) -> dict[str, Any]:
    """Read and parse the serialized metadata blob in ``acquisition.trial_cell_data``.

    Returns an empty dict if the annotation series is absent or unparseable.
    """
    try:
        with open_nwb_readonly(path) as nwb:
            return _parse_annotation_blob(nwb)
    except Exception:
        return {}


def _parse_annotation_blob(nwb: Any) -> dict[str, Any]:
    try:
        ann = nwb.acquisition["trial_cell_data"]
    except (KeyError, AttributeError):
        return {}

    raw: str | None = None
    try:
        data = ann.data
        if hasattr(data, "__getitem__"):
            raw = str(data[0])
        else:
            raw = str(data)
    except Exception:
        pass

    if raw is None:
        try:
            raw = ann.description
        except Exception:
            return {}

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        import ast
        return ast.literal_eval(raw)
    except Exception:
        return {"_raw": raw}


# ---------------------------------------------------------------------------
# Behavior series reader (channel-aware)
# ---------------------------------------------------------------------------

def read_behavior_series(
    path: Path,
    *,
    source: str = "2p",
    channel: SessionChannel | None = None,
) -> dict[str, Any]:
    """Return a dict of ``{series_name: (timestamps, data)}`` for one behavior container.

    Parameters
    ----------
    source:
        ``"2p"`` uses the 2P-aligned container (default); ``"fullres"`` uses the
        full-resolution container.  Ignored when *channel* is provided.
    channel:
        If provided, uses ``channel.behavior_container`` directly.
    """
    try:
        with open_nwb_readonly(path) as nwb:
            if channel is not None:
                container_name = channel.behavior_container
            elif source == "fullres":
                container_name = "Full temporal resolution behavior"
            else:
                channels = resolve_session_channels(nwb)
                container_name = channels[0].behavior_container if channels else "2P-aligned behavior"
            return _extract_behavior_container(nwb, container_name)
    except Exception:
        return {}


def _extract_behavior_container(nwb: Any, container_name: str) -> dict[str, Any]:
    try:
        beh_mod = nwb.processing["behavior"]
        container = beh_mod.data_interfaces[container_name]
        series_dict = container.time_series
    except (KeyError, AttributeError):
        return {}

    result: dict[str, Any] = {}
    for name, ts in series_dict.items():
        try:
            timestamps = ts.timestamps
            if timestamps is None:
                rate = getattr(ts, "starting_time_rate", None) or getattr(ts, "rate", None)
                n = len(ts.data[:])
                import numpy as np
                start = getattr(ts, "starting_time", 0.0) or 0.0
                timestamps = (
                    start + np.arange(n) / float(rate)
                    if rate
                    else np.arange(n, dtype=float)
                )
            else:
                timestamps = timestamps[:]
            result[name] = (timestamps, ts.data[:])
        except Exception:
            pass
    return result


# ---------------------------------------------------------------------------
# Ophys reader (channel-aware)
# ---------------------------------------------------------------------------

def read_roi_response_series(
    path: Path,
    *,
    signal: str = "dff",
    channel: SessionChannel | None = None,
) -> tuple[Any, Any, float] | None:
    """Return ``(data, timestamps, sampling_rate)`` for one ophys signal.

    Parameters
    ----------
    signal:
        Used when *channel* is None: ``"dff"``, ``"fluorescence"``, ``"neuropil"``.
    channel:
        If provided, loads ``channel.ophys_interface`` / ``channel.ophys_series``.
    """
    try:
        with open_nwb_readonly(path) as nwb:
            if channel is not None:
                return _extract_rrs(nwb, channel.ophys_interface, channel.ophys_series)
            # Legacy path for single-channel sessions
            channels = resolve_session_channels(nwb)
            ch = channels[0] if channels else None
            if ch is None:
                return None
            if signal == "dff":
                return _extract_rrs(nwb, ch.ophys_interface, ch.ophys_series)
            # Other signals: map by keyword
            iface_name, series_name = _legacy_signal_map(
                list(nwb.processing["ophys"].data_interfaces.keys()), signal
            )
            return _extract_rrs(nwb, iface_name, series_name)
    except Exception:
        return None


def _legacy_signal_map(
    ophys_ifaces: list[str], signal: str
) -> tuple[str, str]:
    """Map a signal keyword to interface/series names for single-channel files."""
    _map = {
        "fluorescence": ("fluorescence", "fluorescence"),
        "neuropil": ("neuropil", "neuropil fluorescence"),
        "dff": ("dF", "dF"),
    }
    if signal in _map:
        iface, series = _map[signal]
        if iface in ophys_ifaces:
            return iface, series
    return ophys_ifaces[0], ophys_ifaces[0]


def _extract_rrs(
    nwb: Any,
    iface_name: str,
    rrs_name: str,
) -> tuple[Any, Any, float] | None:
    import numpy as np

    try:
        ophys = nwb.processing["ophys"]
        iface = ophys.data_interfaces[iface_name]
        rrs = iface.roi_response_series[rrs_name]
    except (KeyError, AttributeError):
        return None

    data = rrs.data[:]
    rate = getattr(rrs, "rate", None)
    timestamps = getattr(rrs, "timestamps", None)
    if timestamps is not None:
        timestamps = timestamps[:]
    elif rate:
        start = getattr(rrs, "starting_time", 0.0) or 0.0
        timestamps = start + np.arange(data.shape[0]) / float(rate)
    else:
        timestamps = np.arange(data.shape[0], dtype=float)

    sampling_rate = float(rate) if rate else 15.4609375

    return data, timestamps, sampling_rate


# ---------------------------------------------------------------------------
# Plane segmentation reader (channel-aware)
# ---------------------------------------------------------------------------

def read_plane_segmentation(
    path: Path,
    *,
    channel: SessionChannel | None = None,
) -> Any | None:
    """Return the ``PlaneSegmentation`` table for this channel."""
    try:
        with open_nwb_readonly(path) as nwb:
            if channel is not None:
                seg_iface_name = channel.segmentation_interface
            else:
                channels = resolve_session_channels(nwb)
                seg_iface_name = channels[0].segmentation_interface if channels else "ImageSegmentation"
            ophys = nwb.processing["ophys"]
            seg = ophys.data_interfaces[seg_iface_name]
            # Get the first PlaneSegmentation available
            return next(iter(seg.plane_segmentations.values()))
    except (KeyError, AttributeError, StopIteration):
        return None
