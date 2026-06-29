"""Resumable per-subject summarize wrapper for large datasets.

Runs `mndm summarize --subject NNN` for each subject in sequence, skipping
subjects whose H5 output already exists in the most recent run directory.

Usage (PowerShell):
    $env:PYTHONPATH = "H:/SourceRepo2/NeuralManifoldDynamics/mndm/src;..."
    python project/scripts/summarize_per_subject.py --dataset ds003838 `
        --config mndm/config/config_ingest_ds003838.yaml

If interrupted, simply re-run the same command — already-completed subjects
are skipped automatically.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _find_all_run_dirs(processed_root: Path, dataset_id: str) -> list[Path]:
    """Return all run directories for the dataset, newest first."""
    return sorted(
        (processed_root / dataset_id).glob(f"neuralmanifolddynamics_{dataset_id}_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _find_latest_run_dir(processed_root: Path, dataset_id: str) -> Path | None:
    """Return the most recently modified run directory for the dataset."""
    dirs = _find_all_run_dirs(processed_root, dataset_id)
    return dirs[0] if dirs else None


def _already_done(processed_root: Path, dataset_id: str, subject: str) -> bool:
    """Return True if an H5 file for this subject exists in ANY run directory."""
    pattern = f"*{subject}*.h5"
    for run_dir in _find_all_run_dirs(processed_root, dataset_id):
        if any(run_dir.glob(pattern)):
            return True
    return False


def _subject_ids_from_parquet(processed_root: Path, dataset_id: str) -> list[str]:
    """Load subject IDs from features.parquet and return zero-padded strings."""
    import pandas as pd
    parquet = processed_root / dataset_id / "features.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"features.parquet not found: {parquet}")
    df = pd.read_parquet(parquet, columns=["subject"])
    ids = sorted(df["subject"].dropna().unique())
    return [str(int(s)).zfill(3) for s in ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Dataset ID, e.g. ds003838")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--out-dir", default=None, help="Processed output dir override")
    parser.add_argument("--processed-root", default="G:/Science_Datasets_longtime_storage/processed/openneuro",
                        help="Root of processed outputs (default: %(default)s)")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[],
                        help="Extra arguments passed verbatim to mndm summarize")
    parser.add_argument("--dry-run", action="store_true", help="Print subjects without running")
    args = parser.parse_args()

    processed_root = Path(args.processed_root)
    dataset_dir = processed_root / args.dataset

    subjects = _subject_ids_from_parquet(processed_root, args.dataset)
    print(f"Found {len(subjects)} subjects in features.parquet", flush=True)

    # Locate most recent run dir to check for completed H5 files.
    run_dir = _find_latest_run_dir(processed_root, args.dataset)
    if run_dir:
        all_run_dirs = _find_all_run_dirs(processed_root, args.dataset)
        print(f"Found {len(all_run_dirs)} run dir(s); latest: {run_dir.name}", flush=True)
    else:
        print("No existing run dir found — starting fresh.", flush=True)

    skipped, done, failed = 0, 0, 0
    for sub in subjects:
        if _already_done(processed_root, args.dataset, sub):
            print(f"[SKIP] sub-{sub} (H5 already exists)", flush=True)
            skipped += 1
            continue

        cmd = [
            sys.executable, "-m", "mndm.cli", "summarize",
            "--dataset", args.dataset,
            "--config", args.config,
            "--subject", sub,
        ]
        if args.out_dir:
            cmd += ["--out-dir", args.out_dir]
        cmd += args.extra_args

        if args.dry_run:
            print(f"[DRY] {' '.join(cmd)}", flush=True)
            continue

        print(f"[RUN] sub-{sub} ...", end=" ", flush=True)
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=False)
        elapsed = time.time() - t0

        if result.returncode == 0:
            print(f"OK ({elapsed:.0f}s)", flush=True)
            done += 1
        else:
            print(f"FAILED (exit {result.returncode})", flush=True)
            failed += 1

    print(f"\nDone. skipped={skipped}, completed={done}, failed={failed}", flush=True)


if __name__ == "__main__":
    main()
