from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Sequence

from dandi_io.config import ensure_storage_roots
from dandi_io.contracts import AssetRecord, DandiIngestionConfig, ProbeSummary, TriageResult


def filter_assets(
    records: Sequence[AssetRecord],
    config: DandiIngestionConfig,
) -> list[AssetRecord]:
    """Filters a sequence of asset records based on the ingestion configuration.

    Applies path, subject, and session filters defined in the config.

    Args:
        records: Sequence of AssetRecord objects to filter.
        config: Ingestion configuration containing selection criteria.

    Returns:
        List of filtered AssetRecord objects.
    """
    selected: list[AssetRecord] = []
    for record in records:
        if config.selection.path_filters and not _matches_filters(record.path, config.selection.path_filters):
            continue
        if config.selection.subject_filters:
            subject_target = record.subject_id or record.path
            if not _matches_filters(subject_target, config.selection.subject_filters):
                continue
        if config.selection.session_filters:
            session_target = record.session_id or record.path
            if not _matches_filters(session_target, config.selection.session_filters):
                continue
        selected.append(record)
    return selected


def apply_asset_limit(records: Sequence[AssetRecord], limit: int | None) -> list[AssetRecord]:
    """Limits the number of asset records to a maximum value.

    Args:
        records: Sequence of AssetRecord objects.
        limit: Maximum number of records to return (None for no limit).

    Returns:
        A list containing up to *limit* records.
    """
    if limit is None:
        return list(records)
    return list(records[: max(0, limit)])


def write_manifest(records: Sequence[AssetRecord], config: DandiIngestionConfig) -> None:
    """Writes JSON and CSV manifest files for the selected assets.

    Args:
        records: Sequence of assets to include in the manifest.
        config: Configuration defining output paths.
    """
    ensure_storage_roots(config)
    payload = [serialize_asset(record) for record in records]
    config.outputs.manifest_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    fieldnames = (
        "dandiset_id",
        "version",
        "identifier",
        "path",
        "size",
        "subject_id",
        "session_id",
        "asset_url",
        "download_url",
    )
    with config.outputs.manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_probe_summaries(summaries: Sequence[ProbeSummary], config: DandiIngestionConfig) -> None:
    """Writes a JSON file containing summaries of probed NWB files.

    Args:
        summaries: Sequence of ProbeSummary objects.
        config: Configuration defining output paths.
    """
    ensure_storage_roots(config)
    payload = [serialize_probe(summary) for summary in summaries]
    config.outputs.probe_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_triage_markdown(markdown: str, config: DandiIngestionConfig) -> None:
    """Writes a markdown report of the triage process.

    Args:
        markdown: The markdown content to write.
        config: Configuration defining output paths.
    """
    ensure_storage_roots(config)
    config.outputs.triage_markdown.write_text(markdown, encoding="utf-8")


def selected_local_paths(records: Iterable[AssetRecord], config: DandiIngestionConfig) -> list[Path]:
    """Computes the local filesystem paths for a set of assets.

    Args:
        records: Iterable of AssetRecord objects.
        config: Configuration defining the local storage root.

    Returns:
        List of Path objects representing local file locations.
    """
    return [record.local_path(config.storage.raw_root) for record in records]


def serialize_asset(record: AssetRecord) -> dict[str, object]:
    """Serializes an AssetRecord to a plain dictionary.

    Args:
        record: The AssetRecord to serialize.

    Returns:
        Dictionary representation of the record.
    """
    return {
        "dandiset_id": record.dandiset_id,
        "version": record.version,
        "identifier": record.identifier,
        "path": record.path,
        "size": record.size,
        "asset_url": record.asset_url,
        "download_url": record.download_url,
        "subject_id": record.subject_id,
        "session_id": record.session_id,
        "metadata": record.metadata,
    }


def serialize_probe(summary: ProbeSummary) -> dict[str, object]:
    """Serializes a ProbeSummary to a plain dictionary.

    Args:
        summary: The ProbeSummary to serialize.

    Returns:
        Dictionary representation of the summary.
    """
    return {
        "path": summary.path,
        "local_path": str(summary.local_path),
        "exists": summary.exists,
        "file_size": summary.file_size,
        "subject_id": summary.subject_id,
        "session_id": summary.session_id,
        "top_level_groups": list(summary.top_level_groups),
        "acquisitions": list(summary.acquisitions),
        "processing_modules": list(summary.processing_modules),
        "intervals": list(summary.intervals),
        "imaging_planes": list(summary.imaging_planes),
        "devices": list(summary.devices),
        "lab_meta_data": list(summary.lab_meta_data),
        "modality_hints": list(summary.modality_hints),
        "error": summary.error,
        "metadata": summary.metadata,
    }


def serialize_triage(triage: TriageResult) -> dict[str, object]:
    """Serializes a TriageResult to a plain dictionary.

    Args:
        triage: The TriageResult to serialize.

    Returns:
        Dictionary representation of the result.
    """
    return {
        "adapter_id": triage.adapter_id,
        "dandiset_id": triage.dandiset_id,
        "notes": list(triage.notes),
        "selected_assets": [serialize_asset(record) for record in triage.selected_assets],
        "metadata": triage.metadata,
    }


def _matches_filters(value: str, filters: Sequence[str]) -> bool:
    """Internal helper to check if a string contains any of the filter fragments.

    Args:
        value: The string to check.
        filters: Sequence of case-insensitive filter fragments.

    Returns:
        True if any fragment is found in the value, False otherwise.
    """
    text = value.lower()
    return any(fragment.lower() in text for fragment in filters)
