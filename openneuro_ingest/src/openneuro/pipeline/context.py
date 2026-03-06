"""Shared configuration context for ingest commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from core.paths import resolve_paths


@dataclass(frozen=True)
class PathsConfig:
    received_dir: Path
    processed_dir: Path


@dataclass(frozen=True)
class ResolvedConfig:
    """Typed configuration shared across ingest commands."""

    raw: Mapping[str, Any]
    paths: PathsConfig

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any],
        out_dir: Path | None,
        cli_data_dir: Path | None = None,
    ) -> "ResolvedConfig":
        received_dir, processed_dir = resolve_paths(config, out_dir, cli_data_dir)
        return cls(
            raw=config,
            paths=PathsConfig(received_dir=received_dir, processed_dir=processed_dir),
        )

