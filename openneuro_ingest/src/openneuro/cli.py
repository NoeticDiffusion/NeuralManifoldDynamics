"""Command-line interface for the Noetic Ingest toolkit.

Subcommands:
  - download: Fetch dataset(s) from OpenNeuro.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from . import __version__
from core import config_loader
from core import datasets
from . import orchestrate

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", nargs="*", default=None, help="OpenNeuro dataset id(s), e.g., ds003490")
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "config" / "config_ingest.yaml", help="Path to ingest YAML config")
    p.add_argument("--out-dir", type=Path, default=None, help="Processed output base directory (defaults to config paths.processed_dir)")
    p.add_argument("--data-dir", type=Path, default=None, help="Received/raw data base directory (defaults to config paths.received_dir)")
    p.add_argument("--include-pca-results", action="store_true", help="Include datasets marked with pca_results: true in the config file")


def build_parser(argv: Sequence[str] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ingest", description="Noetic Diffusion OpenNeuro Ingest CLI")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="Download dataset(s) and index files")
    _add_common_args(p_dl)

    return parser




def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser(argv)
    args = parser.parse_args(argv)
    
    # Load config
    config = config_loader.load_config(args.config)
    
    # Get dataset IDs
    if args.dataset:
        dataset_ids = args.dataset
    else:
        dataset_ids = datasets.list_datasets(config, include_pca_results=args.include_pca_results)
    
    if not dataset_ids:
        logger.error("No datasets specified")
        return 1
    
    # Execute command
    if args.command == "download":
        return orchestrate.cmd_download(config, dataset_ids, args.out_dir, args.data_dir)
    else:
        parser.error("Unknown command")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


