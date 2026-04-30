from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Sequence

from dandi_analysis.contracts import DiscoveredNwbAsset


_CANONICAL_PATTERN = re.compile(r"^sub-[^/\\]+[/\\].+\.nwb$", re.IGNORECASE)


def discover_nwb_assets(root: Path) -> list[DiscoveredNwbAsset]:
    """Recursively discover all NWB files under the given root directory.

    Canonical assets sit under a ``sub-*/`` subdirectory; loose files directly
    at root or with non-canonical paths are flagged as duplicates and matched
    to their canonical counterpart when possible based on file stem.

    Args:
        root: The directory to search for NWB files.

    Returns:
        A list of DiscoveredNwbAsset objects sorted by canonical status and path.
    """
    root = Path(root)
    if not root.exists():
        return []

    all_nwb: list[Path] = sorted(root.rglob("*.nwb"))

    canonical: list[Path] = []
    non_canonical: list[Path] = []

    for p in all_nwb:
        rel = p.relative_to(root)
        if _CANONICAL_PATTERN.match(str(rel)):
            canonical.append(p)
        else:
            non_canonical.append(p)

    canonical_by_stem: dict[str, Path] = {p.stem: p for p in canonical}

    assets: list[DiscoveredNwbAsset] = []

    for p in canonical:
        stat = p.stat()
        assets.append(
            DiscoveredNwbAsset(
                path=p,
                size=stat.st_size,
                mtime=stat.st_mtime,
                is_canonical=True,
            )
        )

    for p in non_canonical:
        stat = p.stat()
        dup_of = canonical_by_stem.get(p.stem)
        assets.append(
            DiscoveredNwbAsset(
                path=p,
                size=stat.st_size,
                mtime=stat.st_mtime,
                is_canonical=False,
                duplicate_of=dup_of,
            )
        )

    assets.sort(key=lambda a: (not a.is_canonical, str(a.path)))
    return assets


def canonical_assets(assets: Sequence[DiscoveredNwbAsset]) -> list[DiscoveredNwbAsset]:
    """Returns only the canonical assets from a sequence.

    Args:
        assets: Sequence of discovered NWB assets.

    Returns:
        List of canonical DiscoveredNwbAsset objects.
    """
    return [a for a in assets if a.is_canonical]


def duplicate_assets(assets: Sequence[DiscoveredNwbAsset]) -> list[DiscoveredNwbAsset]:
    """Returns only the non-canonical (duplicate or loose) assets from a sequence.

    Args:
        assets: Sequence of discovered NWB assets.

    Returns:
        List of non-canonical DiscoveredNwbAsset objects.
    """
    return [a for a in assets if not a.is_canonical]


def build_inventory_report(assets: Sequence[DiscoveredNwbAsset]) -> str:
    """Generates a markdown table summarizing the discovered assets.

    Args:
        assets: Sequence of DiscoveredNwbAsset objects.

    Returns:
        A string containing the markdown inventory report.
    """
    lines = [
        "# NWB Asset Inventory",
        "",
        f"Total files found: {len(assets)}",
        f"Canonical: {sum(1 for a in assets if a.is_canonical)}",
        f"Non-canonical / duplicate: {sum(1 for a in assets if not a.is_canonical)}",
        "",
        "| Path | Size (bytes) | Canonical | Duplicate of |",
        "| --- | ---: | :---: | --- |",
    ]
    for asset in assets:
        dup = str(asset.duplicate_of) if asset.duplicate_of else ""
        lines.append(
            f"| `{asset.path}` | {asset.size} | "
            f"{'yes' if asset.is_canonical else 'no'} | {dup} |"
        )
    return "\n".join(lines) + "\n"
