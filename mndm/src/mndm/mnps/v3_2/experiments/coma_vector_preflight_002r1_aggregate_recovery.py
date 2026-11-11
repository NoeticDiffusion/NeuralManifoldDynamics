"""Rebuild COMA-VECTOR-PREFLIGHT-002-R1 tables from verified gzip segment JSON.

This recovery step aggregates already-computed segment outputs only. It does
not read EEG data or recompute measurements. The original runner and kernel
hashes remain the measurement provenance; this script's hash is recorded as
the aggregation recovery implementation.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[6]
for source_root in (REPO_ROOT / "core" / "src", REPO_ROOT / "mndm" / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from mndm.mnps.v3_2.experiments import coma_vector_preflight_002r1 as runner


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    selection_path = runner.OUTPUT_ROOT / "selection.json"
    selection = runner._read_json(selection_path)
    if selection.get("runner_and_kernel_sha256") != runner._code_hashes():
        raise RuntimeError("original_runner_or_kernel_hash_mismatch")
    if selection.get("preprocessing_config_sha256") != runner._config_hashes():
        raise RuntimeError("frozen_preprocessing_config_hash_mismatch")
    if selection.get("protocol_sha256") != runner.sha256_file(runner.PROTOCOL_PATH):
        raise RuntimeError("frozen_protocol_hash_mismatch")

    run_id = "run_20261014T052958Z_coma_vector_preflight002r1_first24h"
    run_dir = runner.OUTPUT_ROOT / "cohort" / run_id
    segment_dir = run_dir / "segments"
    compressed_segments = sorted(segment_dir.glob("*.json.gz"), key=lambda path: path.name.lower())
    if len(compressed_segments) != 156:
        raise RuntimeError(f"unexpected_compressed_segment_count:{len(compressed_segments)}")

    original_read_json = runner._read_json

    def read_json(path: Path) -> dict:
        if Path(path).suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                value = json.load(stream)
            if not isinstance(value, dict):
                raise ValueError(f"expected_json_object:{path}")
            return value
        return original_read_json(path)

    runner._read_json = read_json
    runner._collect_segment_files = lambda _run_dir: compressed_segments
    runner._aggregate_run(
        run_dir,
        selection,
        mode="first_24h",
        run_id=run_id,
    )

    # Correct the inventory-count label in the standard participant summary:
    # its original aggregate counted processed first-day segments, not every
    # source header in the frozen inventory.
    source_manifest = original_read_json(runner.OUTPUT_ROOT / "source_manifest.json")
    inventory_counts = {str(subject_id): 0 for subject_id in selection["selected_subject_ids_in_draw_order"]}
    for source_record in source_manifest.get("records", []):
        subject_id = str(source_record.get("subject_id", ""))
        if subject_id in inventory_counts:
            inventory_counts[subject_id] += 1
    segment_counts = {subject_id: 0 for subject_id in inventory_counts}
    row_counts = {subject_id: 0 for subject_id in inventory_counts}
    supported_counts = {subject_id: 0 for subject_id in inventory_counts}
    valid_counts = {
        subject_id: {family: 0 for family in ("ED", "OI", "NIE")}
        for subject_id in inventory_counts
    }
    for path in compressed_segments:
        segment = read_json(path)
        subject_id = str(segment.get("subject_id", ""))
        if subject_id not in segment_counts:
            continue
        segment_counts[subject_id] += 1
        for row in segment.get("rows", []):
            row_counts[subject_id] += 1
            if row.get("support_status") != "unsupported":
                supported_counts[subject_id] += 1
            for family in ("ED", "OI", "NIE"):
                if row.get("metrics", {}).get(family, {}).get("valid"):
                    valid_counts[subject_id][family] += 1
    participant_rows = []
    for subject_id in selection["selected_subject_ids_in_draw_order"]:
        subject_id = str(subject_id)
        participant_rows.append({
            "subject_id": subject_id,
            "source_segment_count_in_inventory": inventory_counts[subject_id],
            "first_day_segment_count_processed": segment_counts[subject_id],
            "window_realization_row_count": row_counts[subject_id],
            "supported_window_realization_rows": supported_counts[subject_id],
            **{f"{family}_valid_value_count": valid_counts[subject_id][family] for family in ("ED", "OI", "NIE")},
        })
    runner._write_csv(run_dir / "participant_summary.csv", participant_rows)

    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = runner._read_json(run_manifest_path)
    run_manifest["output_file_sha256"]["participant_summary.csv"] = runner.sha256_file(
        run_dir / "participant_summary.csv"
    )
    run_manifest["participant_summary_count_semantics"] = (
        "source_segment_count_in_inventory counts all selected source headers; "
        "first_day_segment_count_processed counts retained first-day segment outputs"
    )
    run_manifest["segment_storage_format"] = "gzip-compressed JSON; decompression round-trip SHA-256 verified before source JSON removal"
    run_manifest["aggregation_recovery"] = {
        "reason": "H drive exhausted space while writing the aggregate CSV after all segments were processed",
        "measurement_values_recomputed": False,
        "original_runner_and_kernel_hashes_verified": True,
        "recovery_script_sha256": _sha256_file(Path(__file__).resolve()),
    }
    runner._write_json(run_manifest_path, run_manifest)


if __name__ == "__main__":
    main()
