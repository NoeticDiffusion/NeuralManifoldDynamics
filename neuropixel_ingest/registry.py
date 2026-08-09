"""Registry for small dataset-specific NWB metadata conventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class DatasetAdapter:
    """Declarative mapping for a DANDI ephys dataset.

    The generic reader still preserves every NWB trial column.  These mappings
    only identify useful event fields for future event-aligned analyses.
    """

    dataset_id: str
    name: str
    event_columns: Mapping[str, tuple[str, ...]]
    preferred_quality_columns: tuple[str, ...] = ("quality", "label", "kilosort2_label")


_ADAPTERS: dict[str, DatasetAdapter] = {}


def register(adapter: DatasetAdapter) -> DatasetAdapter:
    """Register and return an adapter (decorator-friendly)."""
    if adapter.dataset_id in _ADAPTERS:
        raise ValueError(f"Duplicate ephys dataset adapter: {adapter.dataset_id}")
    _ADAPTERS[adapter.dataset_id] = adapter
    return adapter


def get_adapter(dataset_id: str) -> DatasetAdapter:
    """Return a registered adapter or explain available IDs."""
    try:
        return _ADAPTERS[dataset_id]
    except KeyError as exc:
        available = ", ".join(sorted(_ADAPTERS)) or "<none>"
        raise KeyError(f"Unknown ephys dataset {dataset_id!r}; known adapters: {available}") from exc


def known_adapters() -> tuple[str, ...]:
    """Return registered dataset IDs."""
    return tuple(sorted(_ADAPTERS))
