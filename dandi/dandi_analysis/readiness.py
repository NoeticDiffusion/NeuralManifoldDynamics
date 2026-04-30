from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

from dandi_analysis.contracts import DiscoveredNwbAsset, ReadyNwbAsset


_MIN_FILE_SIZE_BYTES: int = 1024  # files smaller than 1 KiB are likely stubs
_STABILITY_SLEEP_SECONDS: float = 0.3


def check_readiness(path: Path) -> ReadyNwbAsset:
    """Runs a series of readiness checks on the specified NWB file.

    The following checks are performed in order:
    1. File existence.
    2. File size (must be above a minimum threshold).
    3. File stability (size must not change over a short interval).
    4. HDF5 openability (using h5py).
    5. NWB structure validation (using pynwb).

    Args:
        path: Path to the NWB file to check.

    Returns:
        A ReadyNwbAsset object containing the results of all checks.
    """
    p = Path(path)

    if not p.exists():
        return ReadyNwbAsset(
            path=p, size=0, is_h5_openable=False, is_nwb_openable=False,
            error="file_not_found",
        )

    size1 = p.stat().st_size
    if size1 < _MIN_FILE_SIZE_BYTES:
        return ReadyNwbAsset(
            path=p, size=size1, is_h5_openable=False, is_nwb_openable=False,
            error=f"file_too_small:{size1}",
        )

    time.sleep(_STABILITY_SLEEP_SECONDS)
    size2 = p.stat().st_size
    if size2 != size1:
        return ReadyNwbAsset(
            path=p, size=size2, is_h5_openable=False, is_nwb_openable=False,
            error=f"file_size_unstable:{size1}->{size2}",
        )

    # --- h5py check ---
    is_h5 = False
    h5_error: str | None = None
    try:
        import h5py
        with h5py.File(p, "r"):
            is_h5 = True
    except Exception as exc:
        h5_error = f"h5py_open_failed:{type(exc).__name__}"

    if not is_h5:
        return ReadyNwbAsset(
            path=p, size=size2, is_h5_openable=False, is_nwb_openable=False,
            error=h5_error,
        )

    # --- pynwb check ---
    is_nwb = False
    nwb_error: str | None = None
    try:
        import pynwb
        with pynwb.NWBHDF5IO(str(p), mode="r", load_namespaces=True) as io:
            _ = io.read()
            is_nwb = True
    except Exception as exc:
        nwb_error = f"pynwb_open_failed:{type(exc).__name__}"

    return ReadyNwbAsset(
        path=p,
        size=size2,
        is_h5_openable=True,
        is_nwb_openable=is_nwb,
        error=nwb_error,
    )


def filter_ready(
    assets: Sequence[DiscoveredNwbAsset],
    *,
    canonical_only: bool = True,
) -> list[ReadyNwbAsset]:
    """Checks readiness for a sequence of assets and returns those that pass.

    Args:
        assets: Sequence of discovered NWB assets.
        canonical_only: If True, only check readiness for canonical files.

    Returns:
        List of ReadyNwbAsset objects that are fully "ready".
    """
    candidates = [a for a in assets if (not canonical_only or a.is_canonical)]
    results: list[ReadyNwbAsset] = []
    for asset in candidates:
        r = check_readiness(asset.path)
        if r.is_ready:
            results.append(r)
    return results


def build_readiness_report(results: Sequence[ReadyNwbAsset]) -> str:
    """Generates a compact markdown report of the readiness results.

    Args:
        results: Sequence of ReadyNwbAsset objects.

    Returns:
        A string containing the markdown readiness report.
    """
    lines = [
        "# Readiness Report",
        "",
        f"Files checked: {len(results)}",
        f"Ready: {sum(1 for r in results if r.is_ready)}",
        f"Not ready: {sum(1 for r in results if not r.is_ready)}",
        "",
        "| Path | Size (bytes) | h5py | pynwb | Error |",
        "| --- | ---: | :---: | :---: | --- |",
    ]
    for r in results:
        lines.append(
            f"| `{r.path}` | {r.size} | "
            f"{'ok' if r.is_h5_openable else 'FAIL'} | "
            f"{'ok' if r.is_nwb_openable else 'FAIL'} | "
            f"{r.error or ''} |"
        )
    return "\n".join(lines) + "\n"
