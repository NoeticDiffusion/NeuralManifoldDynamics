"""Command-line interface for the MNPS toolkit.

Subcommands:
  - features: Compute per-epoch features from indexed files.
  - summarize: Project features to MNPS and compute robust summaries.
  - pack: Pack a run directory into a single H5.
  - check-structure: Validate summarized outputs.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
from pathlib import Path
from typing import Sequence

from . import __version__
from core import config_loader
from core import datasets
from core.paths import resolve_paths
from .tools.pack_h5 import pack_run  # lightweight post-processing utility

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _set_blas_thread_env_defaults() -> None:
    """Set conservative BLAS/OpenMP thread defaults before heavy imports."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", nargs="*", default=None, help="OpenNeuro dataset id(s), e.g., ds003490")
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "config" / "config_ingest.yaml", help="Path to ingest YAML config")
    p.add_argument("--out-dir", type=Path, default=None, help="Processed output base directory (defaults to config paths.processed_dir)")
    p.add_argument("--data-dir", type=Path, default=None, help="Received/raw data base directory (defaults to config paths.received_dir)")
    p.add_argument("--include-pca-results", action="store_true", help="Include datasets marked with pca_results: true in the config file")
    p.add_argument("--subject", type=str, default=None, help="Optional subject filter, e.g., 001")
    p.add_argument("--h5-mode", choices=["dataset", "subject"], default="subject", help="Write one H5 per 'dataset' or per 'subject' (default: subject)")
    p.add_argument("--n-jobs", type=int, default=None, help="Number of parallel workers (default: min(cores, 6))")
    p.add_argument("--mem-budget-gb", type=float, default=0.0, help="Memory budget in GB for worker scaling (default: 0 = auto-detect from available RAM)")
    p.add_argument("--mnps-k", type=int, default=None, help="Override k for MNPS kNN (default from config)")
    p.add_argument("--mnps-super-window", type=int, default=None, help="Override MNPS super-window length")
    p.add_argument("--mnps-derivative", choices=["sav_gol", "central"], default=None, help="Override derivative method")
    p.add_argument("--mnps-derivative-window", type=int, default=None, help="Savitzky–Golay window length override")
    p.add_argument("--mnps-derivative-poly", type=int, default=None, help="Savitzky–Golay polynomial order override")
    p.add_argument(
        "--fmri-dvars-threshold",
        type=float,
        default=None,
        help="Override hard DVARS scrub threshold (features.metrics.dvars_threshold)",
    )


def _mnps_overrides_from_args(args) -> dict:
    overrides = {}
    if args.mnps_k is not None:
        overrides["knn_k"] = args.mnps_k
    if args.mnps_super_window is not None:
        overrides["super_window"] = args.mnps_super_window
    if args.mnps_derivative is not None:
        overrides.setdefault("derivative", {})["method"] = args.mnps_derivative
    if args.mnps_derivative_window is not None:
        overrides.setdefault("derivative", {})["window"] = args.mnps_derivative_window
    if args.mnps_derivative_poly is not None:
        overrides.setdefault("derivative", {})["polyorder"] = args.mnps_derivative_poly
    return overrides


def _apply_cli_feature_overrides(config: dict, args) -> None:
    """Apply CLI feature-related overrides in-place."""
    if args.fmri_dvars_threshold is None:
        return
    features_cfg = config.setdefault("features", {})
    metrics_cfg = features_cfg.setdefault("metrics", {})
    metrics_cfg["dvars_threshold"] = float(args.fmri_dvars_threshold)
    metrics_cfg.setdefault("compute_dvars", True)
    logger.info("CLI override: features.metrics.dvars_threshold=%.3f", float(args.fmri_dvars_threshold))


def build_parser(argv: Sequence[str] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mndm", description="MNDM computation CLI")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    p_feat = sub.add_parser("features", help="Compute per-epoch features")
    _add_common_args(p_feat)

    p_sum = sub.add_parser("summarize", help="Project to MNPS and summarize")
    _add_common_args(p_sum)

    p_resum = sub.add_parser("resummarize", help="Re-run summarize only (alias of summarize)")
    _add_common_args(p_resum)

    p_pack = sub.add_parser("pack", help="Pack a summarized run (many small H5) into a single H5 container")
    _add_common_args(p_pack)
    p_pack.add_argument("--run-dir", type=Path, default=None, help="Path to a specific run directory. If omitted, pack the latest summarized run under processed/<dataset>/.")
    p_pack.add_argument("--overwrite", action="store_true", help="Overwrite packed.h5 if it exists")

    p_check = sub.add_parser("check-structure", help="Validate summarized outputs (run folders) against a structure spec")
    _add_common_args(p_check)
    p_check.add_argument("--check-config", type=Path, default=Path(__file__).resolve().parents[2] / "config" / "check_structure.yaml", help="Path to structure check YAML spec")
    p_check.add_argument("--run-selector", choices=["latest", "all"], default=None, help="Which summarized run(s) to check (default: from check-config)")
    p_check.add_argument("--out", type=Path, default=None, help="Optional combined JSON report output path")

    p_all = sub.add_parser("all", help="Run features → summarize")
    _add_common_args(p_all)

    return parser




def main(argv: Sequence[str] | None = None) -> int:
    # Must happen before importing modules that may initialize NumPy/SciPy/BLAS.
    _set_blas_thread_env_defaults()

    parser = build_parser(argv)
    args = parser.parse_args(argv)
    
    # Load config
    config = config_loader.load_config(args.config)
    if isinstance(config, dict):
        _apply_cli_feature_overrides(config, args)
    
    # Get dataset IDs
    if args.dataset:
        dataset_ids = args.dataset
    else:
        dataset_ids = datasets.list_datasets(config, include_pca_results=args.include_pca_results)
    
    if not dataset_ids:
        logger.error("No datasets specified")
        return 1
    
    # Determine n_jobs if not specified
    n_jobs = args.n_jobs
    if n_jobs is None:
        n_jobs = min(multiprocessing.cpu_count(), 6)

    # Execute command
    if args.command == "features":
        from . import orchestrate
        return orchestrate.cmd_features(
            config, dataset_ids, args.out_dir, args.data_dir,
            subject=args.subject, n_jobs=n_jobs, mem_budget_gb=args.mem_budget_gb
        )
    elif args.command == "summarize":
        from . import orchestrate
        return orchestrate.cmd_summarize(config, dataset_ids, args.out_dir, args.data_dir, subject=args.subject, h5_mode=args.h5_mode, mnps_overrides=_mnps_overrides_from_args(args))
    elif args.command == "resummarize":
        from . import orchestrate
        return orchestrate.cmd_summarize(config, dataset_ids, args.out_dir, args.data_dir, subject=args.subject, h5_mode=args.h5_mode, mnps_overrides=_mnps_overrides_from_args(args))
    elif args.command == "pack":
        _, processed_base = resolve_paths(config, args.out_dir, args.data_dir)
        ok = True
        for ds_id in dataset_ids:
            ds_path = processed_base / ds_id
            run_dir = Path(args.run_dir) if args.run_dir else None
            if run_dir is None:
                # Pick latest summarized run under dataset processed dir.
                candidates = [
                    p
                    for pattern in ("neuralmanifolddynamics_*", "mnps_*")
                    for p in ds_path.glob(pattern)
                    if p.is_dir()
                ]
                if not candidates:
                    logger.error("No summarized run directories found under %s", ds_path)
                    ok = False
                    continue
                run_dir = max(candidates, key=lambda p: p.stat().st_mtime)
            out_path = run_dir / "packed.h5"
            try:
                pack_run(run_dir, out_path, overwrite=bool(args.overwrite))
                logger.info("Wrote packed H5: %s", out_path)
            except Exception as exc:
                logger.error("Pack failed for %s (%s): %s", ds_id, run_dir, exc)
                ok = False
        return 0 if ok else 1
    elif args.command == "check-structure":
        from . import orchestrate
        return orchestrate.cmd_check_structure(
            config,
            dataset_ids,
            args.out_dir,
            args.data_dir,
            check_config_path=args.check_config,
            run_selector=args.run_selector,
            out_report=args.out,
        )
    elif args.command == "all":
        from . import orchestrate
        if orchestrate.cmd_features(
            config, dataset_ids, args.out_dir, args.data_dir,
            subject=args.subject, n_jobs=n_jobs, mem_budget_gb=args.mem_budget_gb
        ) != 0:
            return 1
        return orchestrate.cmd_summarize(
            config, dataset_ids, args.out_dir, args.data_dir,
            subject=args.subject, h5_mode=args.h5_mode, mnps_overrides=_mnps_overrides_from_args(args)
        )
    else:
        parser.error("Unknown command")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


