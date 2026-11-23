"""Command line entry point for Zenodo record downloads."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from zenodo_ingest.client import ZenodoError, format_bytes
from zenodo_ingest.config import load_config
from zenodo_ingest.download import configured_records, download_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zenodo-ingest",
        description="List and download files from public Zenodo records",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("list", "Show the files a download would select"),
        ("download", "Stream selected files, resuming incomplete transfers"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--config", type=Path, default=None, help="YAML config")
        command.add_argument(
            "--record",
            action="append",
            default=None,
            help="Zenodo recid, DOI, or record URL. Repeat for more than one record.",
        )
        command.add_argument("--api-base", default=None, help="https://zenodo.org/api or the sandbox API")
        command.add_argument("--include", action="append", default=None, help="Glob kept when selecting files")
        command.add_argument("--exclude", action="append", default=None, help="Glob removed when selecting files")
        if name == "download":
            command.add_argument("--out-dir", type=Path, default=None, help="Override paths.received_dir")
            command.add_argument("--workers", type=int, default=None, help="Parallel file transfers, 1-8")
            command.add_argument("--overwrite", action="store_true", help="Download again even when the local file matches")
            command.add_argument("--dry-run", action="store_true", help="Select files and print sizes without transferring bytes")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.api_base:
        config.setdefault("download", {})["api_base"] = args.api_base
    if args.include is not None:
        config.setdefault("selection", {})["include"] = args.include
    if args.exclude is not None:
        config.setdefault("selection", {})["exclude"] = args.exclude
    if getattr(args, "workers", None) is not None:
        config.setdefault("download", {})["workers"] = args.workers
    if getattr(args, "overwrite", False):
        config.setdefault("download", {})["overwrite"] = True

    records = configured_records(config, args.record)
    if not records:
        raise SystemExit("No Zenodo record specified. Pass --record or set record: in the config.")
    out_dir = getattr(args, "out_dir", None) or Path(config.get("paths", {}).get("received_dir", "data/zenodo/received"))
    dry_run = args.command == "list" or bool(getattr(args, "dry_run", False))
    try:
        results = download_records(records, config, out_dir, dry_run=dry_run)
    except ZenodoError as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1

    incomplete = False
    for result in results.values():
        total = sum(item.size or 0 for item in result.files)
        print(f"{result.record_id}  {result.title}")
        if result.doi:
            print(f"doi: {result.doi}")
        print(f"{len(result.files)} files, {format_bytes(total)}")
        if dry_run:
            for item in result.files:
                checksum = item.checksum or "-"
                print(f"  {format_bytes(item.size):>12}  {checksum:<40}  {item.key}")
            print(f"dry-run: nothing written under {result.root}")
            continue
        failed = result.failed_files
        ok = len(result.files) - len(failed)
        print(f"received {ok} files, {len(failed)} failed -> {result.root}")
        if failed:
            incomplete = True
            print("Failed files are recorded in acquisition_receipt.json. A rerun retries them.")
            for key in failed[:20]:
                print(f"  {key}")
            if len(failed) > 20:
                print(f"  ... {len(failed) - 20} more")
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
