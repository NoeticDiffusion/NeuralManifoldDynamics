from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator


@contextmanager
def open_nwb_readonly(path: Path) -> Generator[Any, None, None]:
    import pynwb
    io = pynwb.NWBHDF5IO(str(path), mode="r", load_namespaces=True)
    try:
        yield io.read()
    finally:
        io.close()


def safe_read_session_metadata(path: Path) -> dict[str, Any]:
    try:
        with open_nwb_readonly(path) as nwb:
            return _extract(nwb)
    except Exception:
        return {}


def _extract(nwb: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for attr in ("session_description", "identifier"):
        val = getattr(nwb, attr, None)
        if val is not None:
            meta[attr] = str(val)
    start = getattr(nwb, "session_start_time", None)
    if start is not None:
        meta["session_start_time"] = str(start)
    try:
        meta["processing_keys"] = list(nwb.processing.keys())
    except Exception:
        meta["processing_keys"] = []
    try:
        meta["interval_names"] = list(nwb.intervals.keys())
    except Exception:
        meta["interval_names"] = []
    try:
        meta["imaging_planes"] = list(nwb.imaging_planes.keys())
    except Exception:
        meta["imaging_planes"] = []
    return meta
