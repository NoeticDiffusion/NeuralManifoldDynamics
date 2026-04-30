"""
DANDI Data I/O and Ingestion.

Core machinery for discovering, downloading, and probing assets from the
DANDI archive, with support for dataset-specific adapters.
"""

from dandi_io.client import DandiClient
from dandi_io.config import DEFAULT_DANDI_CONFIG, ensure_storage_roots, resolve_dandi_config
from dandi_io.contracts import (
    AssetRecord,
    DandiIngestionConfig,
    ProbeSummary,
    TriageResult,
)
from dandi_io.registry import get_dataset_adapter, known_adapters


def main() -> int:
    from dandi_io.cli import main as cli_main

    return cli_main()


__all__ = [
    "AssetRecord",
    "DandiClient",
    "DandiIngestionConfig",
    "DEFAULT_DANDI_CONFIG",
    "ProbeSummary",
    "TriageResult",
    "ensure_storage_roots",
    "get_dataset_adapter",
    "known_adapters",
    "main",
    "resolve_dandi_config",
]
