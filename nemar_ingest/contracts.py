"""Small, dependency-free contracts used by the NEMAR downloader."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One file advertised by a NEMAR release manifest."""

    path: str
    url: str | None = None
    size: int | None = None
    sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    checksum: str | None = None
    checksum_algorithm: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetDownload:
    """Result for one downloaded dataset."""

    dataset_id: str
    version: str
    root: str
    selected_files: tuple[str, ...]
    manifest_url: str
    failed_files: tuple[str, ...] = ()

