from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence


@dataclass(slots=True)
class DatasetSpec:
    """Metadata specification for a target DANDI dataset."""

    config_id: str
    adapter: str
    dandiset_id: str
    version: str = "draft"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StorageSpec:
    """Defines the local filesystem layout for an ingestion job."""

    output_root: Path
    cache_root: Path
    raw_root: Path
    manifest_root: Path
    triage_root: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SelectionSpec:
    """Defines which assets should be included in an ingestion job."""

    path_filters: tuple[str, ...] = ()
    subject_filters: tuple[str, ...] = ()
    session_filters: tuple[str, ...] = ()
    asset_limit: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionSpec:
    """Governs the runtime behavior of the ingestion process."""

    metadata_only: bool = True
    streaming_allowed: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OutputSpec:
    """Paths for the various files generated during ingestion."""

    manifest_json: Path
    manifest_csv: Path
    triage_markdown: Path
    probe_json: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DandiIngestionConfig:
    """Consolidated configuration for a DANDI ingestion pipeline."""

    dataset: DatasetSpec
    storage: StorageSpec
    selection: SelectionSpec
    execution: ExecutionSpec
    outputs: OutputSpec
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AssetRecord:
    """Record of a single asset from the DANDI archive."""

    dandiset_id: str
    version: str
    identifier: str
    path: str
    size: int | None = None
    asset_url: str | None = None
    download_url: str | None = None
    subject_id: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def local_path(self, raw_root: Path) -> Path:
        """Return the local filesystem path for this asset given a root directory."""
        return raw_root / Path(self.path)


@dataclass(slots=True)
class ProbeSummary:
    """Summary of structural and metadata information probed from an NWB file."""

    path: str
    local_path: Path
    exists: bool
    file_size: int | None = None
    subject_id: str | None = None
    session_id: str | None = None
    top_level_groups: tuple[str, ...] = ()
    acquisitions: tuple[str, ...] = ()
    processing_modules: tuple[str, ...] = ()
    intervals: tuple[str, ...] = ()
    imaging_planes: tuple[str, ...] = ()
    devices: tuple[str, ...] = ()
    lab_meta_data: tuple[str, ...] = ()
    modality_hints: tuple[str, ...] = ()
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TriageResult:
    """Result of the triage process identifying which assets to process."""

    adapter_id: str
    dandiset_id: str
    selected_assets: tuple[AssetRecord, ...] = ()
    notes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetAdapter(Protocol):
    """Protocol defining the interface for dataset-specific logic."""

    adapter_id: str

    def select_assets(
        self,
        records: Sequence[AssetRecord],
        config: DandiIngestionConfig,
    ) -> list[AssetRecord]:
        """Select relevant assets from a collection of records."""
        ...

    def build_triage(
        self,
        records: Sequence[AssetRecord],
        config: DandiIngestionConfig,
        *,
        probes: Sequence[ProbeSummary] | None = None,
    ) -> TriageResult:
        """Analyze selected records and optional probes."""
        ...

    def render_triage_markdown(self, triage: TriageResult) -> str:
        """Render a triage result as a human-readable markdown report."""
        ...
