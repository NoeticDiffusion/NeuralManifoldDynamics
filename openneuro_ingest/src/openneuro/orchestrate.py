"""Pipeline orchestration and command handlers.

This module contains the business logic for each CLI command, keeping cli.py
thin and focused on argument parsing only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

from core import config_loader
from . import download
from .pipeline.context import ResolvedConfig

logger = logging.getLogger(__name__)




def cmd_download(
    config: dict,
    dataset_ids: list[str],
    out_dir: Path | None,
    data_dir: Path | None = None,
) -> int:
    """Download datasets into the received directory."""
    try:
        resolved = ResolvedConfig.from_mapping(config, out_dir, data_dir)
        download.download_datasets(dataset_ids, config, resolved.paths.received_dir)
        return 0
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return 1


 
