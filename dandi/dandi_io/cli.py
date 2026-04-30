from __future__ import annotations

import argparse

from dandi_io.client import DandiClient
from dandi_io.config import ensure_storage_roots, resolve_dandi_config
from dandi_io.download import (
    filter_assets,
    write_manifest,
    write_probe_summaries,
    write_triage_markdown,
)
from dandi_io.probe import probe_assets
from dandi_io.registry import get_dataset_adapter


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config, config_path = resolve_dandi_config(args.config)
    ensure_storage_roots(config)
    adapter = get_dataset_adapter(config.dataset.adapter)
    client = DandiClient()

    records = client.list_assets(config.dataset.dandiset_id, config.dataset.version)
    filtered = filter_assets(records, config)
    selected = adapter.select_assets(filtered, config)
    triage = adapter.build_triage(selected, config)

    write_manifest(filtered, config)
    write_triage_markdown(adapter.render_triage_markdown(triage), config)

    if args.command == "list":
        print(f"Loaded config: {config_path}")
        print(f"Dandiset: {config.dataset.dandiset_id} ({config.dataset.version})")
        print(f"Assets listed: {len(records)}")
        print(f"Assets after filters: {len(filtered)}")
        print(f"Assets selected for triage: {len(selected)}")
        print(f"Manifest written: {config.outputs.manifest_json}")
        print(f"Triage written: {config.outputs.triage_markdown}")
        return 0

    if args.command == "download":
        targets = selected or filtered
        print(f"Loaded config: {config_path}")
        print(f"Dandiset: {config.dataset.dandiset_id} ({config.dataset.version})")
        print(f"Assets selected for download: {len(targets)}")
        downloaded = []
        for record in targets:
            print(f"Downloading: {record.path}")
            downloaded.extend(client.download_assets([record], output_root=config.storage.raw_root))
            print(f"Completed: {record.path}")
        print(f"Downloaded or confirmed local assets: {len(downloaded)}")
        for path in downloaded:
            print(path)
        return 0

    if args.command == "probe":
        targets = selected or filtered
        summaries = probe_assets(targets, raw_root=config.storage.raw_root)
        write_probe_summaries(summaries, config)
        triage_with_probes = adapter.build_triage(selected, config, probes=summaries)
        write_triage_markdown(adapter.render_triage_markdown(triage_with_probes), config)
        print(f"Probe summary written: {config.outputs.probe_json}")
        for summary in summaries:
            status = "present" if summary.exists else "missing"
            print(f"{status}: {summary.local_path}")
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generic DANDI listing, probing, and download workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("list", "download", "probe"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--config", required=True, help="Path to a DANDI YAML config.")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
