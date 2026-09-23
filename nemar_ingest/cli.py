"""Command line entry point for NEMAR dataset downloads."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import load_config
from .download import download_datasets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nemar-ingest", description="Download public NEMAR BIDS datasets")
    parser.add_argument("--config", type=Path, required=True, help="NEMAR YAML configuration")
    parser.add_argument("--dataset", nargs="*", default=None, help="NEMAR dataset id(s), e.g. on005385")
    parser.add_argument("--out-dir", type=Path, default=None, help="Override paths.received_dir")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and select the manifest but do not download payload files")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    dataset_cfg = config.get("dataset", {})
    datasets = args.dataset or [str(dataset_cfg.get("id", ""))]
    datasets = [item for item in datasets if item]
    if not datasets:
        raise SystemExit("No NEMAR dataset specified")
    out_dir = args.out_dir or Path(config.get("paths", {}).get("received_dir", "data/nemar/received"))
    if args.dry_run:
        # Reuse manifest parsing and selection, but replace transfer with an
        # explicit error if a caller accidentally reaches the payload stage.
        from .client import NemarClient
        from .download import select_manifest_entries
        from .client import validate_dataset_id, validate_version
        from .download import _mapping
        selection = _mapping(config, "selection")
        version = validate_version(str(dataset_cfg.get("version", "latest")))
        for dataset in datasets:
            dataset = validate_dataset_id(dataset)
            manifest_url, entries = NemarClient().get_manifest(dataset, version)
            selected = select_manifest_entries(entries, selection)
            print(f"{dataset}: {len(selected)} files selected from {manifest_url}")
        return 0
    results = download_datasets(datasets, config, out_dir)
    incomplete = False
    for result in results.values():
        failed = result.failed_files
        ok = len(result.selected_files) - len(failed)
        print(f"Downloaded {result.dataset_id} {result.version}: {ok} files, {len(failed)} failed -> {result.root}")
        if failed:
            incomplete = True
            print("Failed files are in acquisition_receipt.json under failed_entries. A rerun retries them.")
            for path in failed[:20]:
                print(f"  {path}")
            if len(failed) > 20:
                print(f"  ... {len(failed) - 20} more")
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())

