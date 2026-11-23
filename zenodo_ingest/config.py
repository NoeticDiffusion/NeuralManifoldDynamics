"""YAML configuration loading for Zenodo downloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {"received_dir": "data/zenodo/received"},
    "selection": {},
    "download": {
        "api_base": "https://zenodo.org/api",
        "workers": 2,
        "chunk_bytes": 8 * 1024 * 1024,
        "retries": 5,
        "retry_backoff_seconds": 2.0,
        "timeout_seconds": 120,
        "resume": True,
        "verify_checksum": True,
        "overwrite": False,
        "continue_on_error": True,
        "progress_bytes": 256 * 1024 * 1024,
        "token_env": "ZENODO_TOKEN",
    },
}


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load a YAML config and fill unspecified sections from defaults."""
    if path is None:
        return deep_merge(DEFAULT_CONFIG, {})
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping at config root in {config_path}")
    return deep_merge(DEFAULT_CONFIG, loaded)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
