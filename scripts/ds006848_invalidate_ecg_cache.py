"""Invalidate ds006848 ECG feature cache.

Deletes the intermediate JSON files for ds006848 so that the next
`mndm.cli features` run regenerates them with new columns:
  ecg_polarity_inverted, ecg_peak_detector,
  ecg_hrv_dominant_stage_label, ecg_hrv_dominant_stage_frac,
  ecg_hrv_n_stage_labels, ecg_hrv_contains_excluded_label.

Also deletes features.csv and features.parquet so they are rebuilt
from scratch.  Run this script ONCE before the re-run, then start
the features pipeline in an external terminal.

Usage:
  python scripts/ds006848_invalidate_ecg_cache.py [--dry-run]
"""
import argparse
import pathlib
import shutil
import sys

PROCESSED_DIR = pathlib.Path("J:/processed/openneuro/ds006848")
INTERMEDIATE_DIR = PROCESSED_DIR / "intermediate"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print what would be deleted without deleting")
    args = parser.parse_args()

    dry = args.dry_run
    prefix = "[DRY-RUN] " if dry else ""

    if not PROCESSED_DIR.exists():
        print(f"ERROR: Processed dir not found: {PROCESSED_DIR}", file=sys.stderr)
        sys.exit(1)

    deleted = 0

    # Delete intermediate JSONs
    if INTERMEDIATE_DIR.exists():
        json_files = sorted(INTERMEDIATE_DIR.glob("*.json"))
        print(f"{prefix}Deleting {len(json_files)} intermediate JSON files from {INTERMEDIATE_DIR}")
        for jf in json_files:
            print(f"  {prefix}rm {jf.name}")
            if not dry:
                jf.unlink()
        deleted += len(json_files)
    else:
        print(f"WARNING: Intermediate dir not found: {INTERMEDIATE_DIR}")

    # Delete features.csv and features.parquet
    for fname in ("features.csv", "features.parquet"):
        fp = PROCESSED_DIR / fname
        if fp.exists():
            print(f"{prefix}Deleting {fp}")
            if not dry:
                fp.unlink()
            deleted += 1

    print(f"\n{prefix}Done. {deleted} files {'would be ' if dry else ''}deleted.")
    if not dry:
        print("\nNext step — run in an EXTERNAL PowerShell window:")
        print(
            "  cd H:\\SourceRepo2\\NeuralManifoldDynamics\n"
            "  $env:PYTHONPATH='H:\\SourceRepo2\\NeuralManifoldDynamics\\mndm\\src;H:\\SourceRepo2\\NeuralManifoldDynamics\\core\\src'\n"
            "  .venv\\Scripts\\python.exe -m mndm.cli features `\n"
            "    --config mndm/config/config_ingest_ds006848.yaml `\n"
            "    --dataset ds006848 `\n"
            "    --data-dir K:/ExternalReceivedDatasets/openneuro/received `\n"
            "    --out-dir J:/processed/openneuro `\n"
            "    --n-jobs 4 2>&1 | Tee-Object -FilePath logs/ds006848_features_rerun.txt"
        )


if __name__ == "__main__":
    main()
