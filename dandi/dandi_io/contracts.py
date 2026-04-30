from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence


@dataclass(slots=True)
class DatasetSpec:
    """Metadata specification for a target DANDI dataset.

    Attributes:
        config_id: Unique identifier for this configuration.
        adapter: Name of the adapter to use for dataset-specific logic.
        dandiset_id: DANDI identifier (e.g., '000718').
        version: Version of the dandiset to ingest (default 'draft').
        metadata: Arbitrary dictionary for additional spec data.
    """

    config_id: str
    adapter: str
    dandiset_id: str
    version: str = "draft"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StorageSpec:
    """Defines the local filesystem layout for an ingestion job.

    Attributes:
        output_root: Root directory for processed outputs.
        cache_root: Root directory for cached files.
        raw_root: Directory where raw assets (NWB files) are stored.
        manifest_root: Directory for manifest and index files.
        triage_root: Directory for triage reports and logs.
        metadata: Arbitrary dictionary for additional storage data.
    """

    output_root: Path
    cache_root: Path
    raw_root: Path
    manifest_root: Path
    triage_root: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SelectionSpec:
    """Defines which assets should be included in an ingestion job.

    Attributes:
        path_filters: List of string fragments to match against file paths.
        subject_filters: List of fragments to match against subject IDs.
        session_filters: List of fragments to match against session IDs.
        asset_limit: Maximum number of assets to process.
        metadata: Arbitrary dictionary for additional selection data.
    """

    path_filters: tuple[str, ...] = ()
    subject_filters: tuple[str, ...] = ()
    session_filters: tuple[str, ...] = ()
    asset_limit: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionSpec:
    """Governs the runtime behavior of the ingestion process.

    Attributes:
        metadata_only: If True, do not download or process heavy data arrays.
        streaming_allowed: If True, use S3 streaming for NWB probing.
        metadata: Arbitrary dictionary for execution parameters.
    """

    metadata_only: bool = True
    streaming_allowed: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OutputSpec:
    """Paths for the various files generated during ingestion.

    Attributes:
        manifest_json: Path to the JSON manifest of selected assets.
        manifest_csv: Path to the CSV manifest of selected assets.
        triage_markdown: Path to the generated markdown report.
        probe_json: Path to the results of NWB probing.
        metadata: Arbitrary dictionary for additional output data.
    """

    manifest_json: Path
    manifest_csv: Path
    triage_markdown: Path
    probe_json: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DandiIngestionConfig:
    """Consolidated configuration for a DANDI ingestion pipeline.

    Attributes:
        dataset: Target dataset specification.
        storage: Filesystem layout.
        selection: Asset filtering criteria.
        execution: Runtime flags.
        outputs: Generated file paths.
        metadata: Arbitrary top-level metadata.
    """

    dataset: DatasetSpec
    storage: StorageSpec
    selection: SelectionSpec
    execution: ExecutionSpec
    outputs: OutputSpec
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AssetRecord:
    """Record of a single asset (file) from the DANDI archive.

    Attributes:
        dandiset_id: ID of the parent dandiset.
        version: Dandiset version.
        identifier: Unique asset identifier.
        path: Relative path within the dandiset.
        size: File size in bytes.
        asset_url: Permanent URL to the asset metadata.
        download_url: Direct download link for the file.
        subject_id: Subject identifier if available in metadata.
        session_id: Session identifier if available in metadata.
        metadata: Full asset metadata dictionary from DANDI.
    """

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
        """Returns the local filesystem path for this asset given a root directory.

        Args:
            raw_root: The local root directory for raw assets.

        Returns:
            A Path object pointing to the local file.
        """
        return raw_root / Path(self.path)


@dataclass(slots=True)
class ProbeSummary:
    """Summary of structural and metadata information probed from an NWB file.

    Attributes:
        path: Relative path within the dandiset.
        local_path: Absolute local path to the file.
        exists: True if the file exists locally.
        file_size: Size on disk in bytes.
        subject_id: Subject ID extracted from the NWB file.
        session_id: Session ID extracted from the NWB file.
        top_level_groups: Names of top-level HDF5 groups.
        acquisitions: Names of acquisition objects.
        processing_modules: Names of processing modules.
        intervals: Names of time interval tables (epochs).
        imaging_planes: Names of imaging planes.
        devices: Names of recorded devices.
        lab_meta_data: Names of lab-specific metadata blocks.
        modality_hints: List of inferred data modalities.
        error: Descriptive error if probing failed.
        metadata: Arbitrary dictionary for additional diagnostics.
    """

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
    """Result of the triage process identifying which assets to process.

    Attributes:
        adapter_id: ID of the adapter that performed the triage.
        dandiset_id: ID of the dandiset.
        selected_assets: Collection of assets identified for downstream use.
        notes: Human-readable observations or warnings from triage.
        metadata: Arbitrary dictionary for additional results.
    """

    adapter_id: str
    dandiset_id: str
    selected_assets: tuple[AssetRecord, ...] = ()
    notes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetAdapter(Protocol):
    """Protocol defining the interface for dataset-specific logic.

    Each dataset (e.g., 000718, 001710) may require specialized logic
    for asset selection and triage reporting.
    """

    adapter_id: str

    def select_assets(
        self,
        records: Sequence[AssetRecord],
        config: DandiIngestionConfig,
    ) -> list[AssetRecord]:
        """Selects relevant assets from a collection of records.

        Args:
            records: All available asset records in the dandiset.
            config: Job configuration.

        Returns:
            List of records to be processed.
        """
        ...

    def build_triage(
        self,
        records: Sequence[AssetRecord],
        config: DandiIngestionConfig,
        *,
        probes: Sequence[ProbeSummary] | None = None,
    ) -> TriageResult:
        """Analyzes probes to build a final triage result.

        Args:
            records: All available asset records.
            config: Job configuration.
            probes: Structural summaries of the files.

        Returns:
            A TriageResult object summarizing findings.
        """
        ...

    def render_triage_markdown(self, triage: TriageResult) -> str:
        """Renders the triage result as a human-readable markdown report.

        Args:
            triage: The triage result to render.

        Returns:
            A string containing the markdown report.
        """
        ...
