"""Small contracts for the Zenodo downloader."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class FileEntry:
    """One file advertised by a Zenodo record."""

    key: str
    url: str
    size: int | None = None
    checksum: str | None = None
    algorithm: str | None = None
    digest: str | None = None


@dataclass(frozen=True, slots=True)
class ZenodoRecord:
    """Resolved Zenodo record and its file list."""

    record_id: str
    files: tuple[FileEntry, ...]
    title: str = ""
    doi: str | None = None
    concept_doi: str | None = None
    api_url: str = ""


@dataclass(frozen=True, slots=True)
class DownloadOptions:
    """Transfer settings for one download run."""

    chunk_bytes: int = 8 * 1024 * 1024
    retries: int = 5
    retry_backoff_seconds: float = 2.0
    timeout_seconds: float = 120.0
    resume: bool = True
    verify_checksum: bool = True
    overwrite: bool = False
    progress_bytes: int = 256 * 1024 * 1024
    workers: int = 2
    continue_on_error: bool = True


@dataclass(frozen=True, slots=True)
class FileTransfer:
    """Outcome for one selected file."""

    key: str
    local_path: str
    size: int | None
    checksum: str | None
    status: str
    checksum_verified: bool = False
    error: str = ""


@dataclass(frozen=True, slots=True)
class RecordDownload:
    """Result for one Zenodo record."""

    record_id: str
    root: str
    title: str
    doi: str | None
    files: tuple[FileTransfer, ...] = field(default_factory=tuple)
    failed_files: tuple[str, ...] = ()
    dry_run: bool = False
