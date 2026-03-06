"""Fallback downloader that uses DataLad/git-annex to fetch OpenNeuro datasets.

This script clones the public OpenNeuroDatasets git-annex repositories and
downloads their full contents (or a user-specified subset) using DataLad.
It is intended as a workaround when the openneuro-py API cannot resolve
include patterns for certain datasets.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

logger = logging.getLogger("datalad_fallback")


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "config_ingest.yaml"


def _load_config(config_path: Path) -> dict:
    try:
        from core import config_loader

        return config_loader.load_config(config_path)
    except Exception as exc:
        logger.warning("Failed to load config %s: %s", config_path, exc)
        return {}


def _resolve_data_dir(config: dict, override: Path | None) -> Path:
    if override is not None:
        return override
    paths_cfg = config.get("paths", {}) if isinstance(config, dict) else {}
    data_dir = paths_cfg.get("received_dir")
    if data_dir:
        return Path(data_dir)
    return Path.cwd()


def _resolve_processed_dir(config: dict, override: Path | None, default_base: Path) -> Path:
    if override is not None:
        return override
    paths_cfg = config.get("paths", {}) if isinstance(config, dict) else {}
    processed_dir = paths_cfg.get("processed_dir")
    if processed_dir:
        return Path(processed_dir)
    return default_base.parent / "processed"


def _resolve_datasets(config: dict, overrides: Sequence[str] | None) -> list[str]:
    if overrides:
        return [str(ds) for ds in overrides]
    datasets_cfg = config.get("datasets") if isinstance(config, dict) else None
    if isinstance(datasets_cfg, Iterable):
        return [str(ds) for ds in datasets_cfg]
    raise ValueError("No datasets specified via --dataset and none found in config.")


def _ensure_datalad_available() -> None:
    try:
        import datalad  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "DataLad is required for the fallback downloader. "
            "Install it via `pip install datalad datalad-installer`."
        ) from exc


def _configure_datalad_logging(level: str) -> None:
    """Best-effort suppression of DataLad's internal logging noise.

    DataLad emits a lot of INFO via Python logging. We clamp its logger levels.
    """
    try:
        lvl = getattr(logging, str(level).upper(), logging.WARNING)
    except Exception:
        lvl = logging.WARNING

    # Common DataLad logger namespaces.
    for name in ("datalad", "datalad.core", "datalad.annex", "datalad.ui", "git", "git.cmd"):
        try:
            logging.getLogger(name).setLevel(lvl)
        except Exception:
            pass


def _call_datalad(fn, **kwargs):
    """Call a DataLad API function while only passing supported kwargs.

    DataLad's Python API differs across versions; this prevents failures when
    optional args like `result_renderer` are unsupported.
    """
    try:
        import inspect

        params = set(inspect.signature(fn).parameters.keys())
    except Exception:  # pragma: no cover
        params = set()

    filtered = {k: v for k, v in kwargs.items() if (not params) or (k in params)}
    return fn(**filtered)


def _chunk(seq: Sequence[str], size: int) -> list[list[str]]:
    if size <= 0:
        return [list(seq)]
    out: list[list[str]] = []
    buf: list[str] = []
    for item in seq:
        buf.append(item)
        if len(buf) >= size:
            out.append(buf)
            buf = []
    if buf:
        out.append(buf)
    return out


@dataclass(frozen=True)
class GetPlan:
    targets: list[str]
    derivatives_skipped: bool


def _build_get_plan(dataset: "Dataset", paths: Sequence[str] | None, include_derivatives: bool) -> GetPlan:
    # Choose what to fetch:
    # - explicit subset overrides everything
    # - default: fetch all top-level entries except derivatives
    if paths:
        targets = [str(p) for p in paths if str(p).strip()]
        if not targets:
            targets = ["."]
        return GetPlan(targets=targets, derivatives_skipped=not include_derivatives)

    root = Path(dataset.pathobj)
    entries = [p for p in root.iterdir()]
    targets: list[str] = []
    for entry in entries:
        if entry.name == "derivatives" and not include_derivatives:
            continue
        targets.append(str(entry.relative_to(root)))
    if not targets:
        targets = ["."]
    return GetPlan(targets=targets, derivatives_skipped=not include_derivatives)


def _clone_if_needed(dataset_id: str, base_dir: Path, update: bool) -> "Dataset":
    from datalad import api as dl_api
    from datalad.support.exceptions import IncompleteResultsError

    repo_url = f"https://github.com/OpenNeuroDatasets/{dataset_id}.git"
    dest = base_dir / dataset_id
    dest.parent.mkdir(parents=True, exist_ok=True)

    ds = dl_api.Dataset(dest)
    if ds.is_installed():
        logger.info("Dataset %s already cloned at %s", dataset_id, dest)
        if update:
            logger.info("Updating %s from upstream...", dataset_id)
            try:
                # DataLad's Dataset.update() signature varies across versions.
                # Newer versions accept `to_default=...`; older ones do not.
                update_fn = getattr(ds, "update", None)
                if update_fn is None:
                    raise RuntimeError("DataLad Dataset object has no update() method")
                try:
                    import inspect

                    params = set(inspect.signature(update_fn).parameters.keys())
                except Exception:  # pragma: no cover - introspection can fail
                    params = set()

                # Try to disable DataLad result rendering across versions
                extra = {}
                if "result_renderer" in params:
                    extra["result_renderer"] = "disabled"
                if "to_default" in params:
                    update_fn(to_default=True, merge=True, **extra)
                elif "merge" in params:
                    update_fn(merge=True, **extra)
                else:
                    update_fn(**extra)
            except IncompleteResultsError as exc:  # pragma: no cover - runtime
                raise RuntimeError(f"Failed to update dataset {dataset_id}") from exc
            except TypeError:
                # Last-resort fallback if introspection missed (e.g. wrapped callables)
                try:  # pragma: no cover - environment specific
                    ds.update(merge=True, result_renderer="disabled")
                except TypeError:
                    try:
                        ds.update(result_renderer="disabled")
                    except TypeError:
                        ds.update()
        return ds

    logger.info("Cloning %s into %s", repo_url, dest)
    try:
        ds = _call_datalad(dl_api.clone, source=repo_url, path=dest, result_renderer="disabled")
    except Exception as exc:  # pragma: no cover - runtime
        raise RuntimeError(f"Clone failed for {dataset_id}: {exc}") from exc
    return ds


def _get_content(
    dataset: "Dataset",
    jobs: int | None,
    paths: Sequence[str] | None,
    include_derivatives: bool,
    on_failure: str,
    result_renderer: str,
    batch_targets: int,
    report_every: int,
    preview: int,
) -> None:
    from datalad.support.exceptions import IncompleteResultsError

    plan = _build_get_plan(dataset, paths, include_derivatives)
    targets = plan.targets

    # Avoid logging thousands of targets; show count + small preview only.
    if preview < 0:
        preview = 0
    if preview > 0 and len(targets) > 1:
        shown = targets[:preview]
        suffix = f" (+{len(targets) - len(shown)} more)" if len(targets) > len(shown) else ""
        target_msg = ", ".join(shown) + suffix
    else:
        target_msg = targets[0] if targets else "."

    logger.info(
        "Fetching content for %s (targets=%d; %s)%s",
        dataset.pathobj,
        len(targets),
        target_msg,
        "" if include_derivatives else " [derivatives skipped by default]",
    )

    base_kwargs = {"recursive": True, "on_failure": on_failure, "result_renderer": result_renderer}
    if jobs is not None:
        base_kwargs["jobs"] = jobs

    batches = _chunk(targets, int(batch_targets or 0))
    total = len(batches)
    if total > 1:
        logger.info("Downloading in %d batch(es) (batch_targets=%d)", total, batch_targets)
    try:
        for i, batch in enumerate(batches, start=1):
            kwargs = dict(base_kwargs)
            kwargs["path"] = batch
            _call_datalad(dataset.get, **kwargs)
            if report_every and (i % int(report_every) == 0 or i == total):
                logger.info("Completed batch %d/%d", i, total)
    except IncompleteResultsError as exc:  # pragma: no cover - runtime
        # When on_failure != "stop", DataLad may still raise IncompleteResultsError.
        # In that case, keep going but surface the issue.
        if str(on_failure).lower() != "stop":
            logger.warning("Data fetch had incomplete results for %s: %s", dataset.pathobj, exc)
            return
        raise RuntimeError(f"Data fetch failed for {dataset.pathobj}") from exc


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download OpenNeuro datasets via DataLad/git-annex."
    )
    parser.add_argument(
        "--dataset",
        nargs="*",
        default=None,
        help="Dataset IDs (e.g., ds005114). Defaults to config datasets.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config_path(),
        help="Path to ingest YAML config (used for defaults).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override received/raw base directory.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Number of parallel jobs for git-annex transfers (default: 4).",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Fetch latest commits before downloading content.",
    )
    parser.add_argument(
        "--subset",
        action="append",
        default=None,
        help=(
            "Optional dataset-relative path(s) to fetch (default: full dataset). "
            "Can be provided multiple times, e.g. --subset sub-01/func --subset sub-02/func"
        ),
    )
    parser.add_argument(
        "--on-failure",
        default="stop",
        choices=["stop", "ignore", "continue"],
        help=(
            "How to handle missing/unavailable annex content during download. "
            "'stop' aborts on first failure (default). "
            "'ignore' continues and returns success when some files are unavailable."
        ),
    )
    parser.add_argument(
        "--include-derivatives",
        action="store_true",
        help="Include derivatives directory (default: skip derivatives to reduce size).",
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="After download, build BIDS file index (file_index.csv).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Override processed/output base directory (used when --build-index).",
    )
    parser.add_argument(
        "--skip-get",
        action="store_true",
        help="Skip DataLad get step (assumes content already present).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    parser.add_argument(
        "--datalad-log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Clamp DataLad internal logging to this level (default: WARNING).",
    )
    parser.add_argument(
        "--result-renderer",
        default="disabled",
        choices=["disabled", "generic", "tailored", "json", "json_pp"],
        help="DataLad result renderer (default: disabled to avoid spam).",
    )
    parser.add_argument(
        "--batch-targets",
        type=int,
        default=0,
        help=(
            "Optional: split the get() call into batches of this many top-level targets. "
            "Useful to get periodic 'batch completed' progress without printing per-file. "
            "0 means a single get() call (default)."
        ),
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=1,
        help="When using --batch-targets, log progress every N batches (default: 1).",
    )
    parser.add_argument(
        "--targets-preview",
        type=int,
        default=8,
        help="How many targets to include in the log preview (default: 8). Set 0 to disable.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    _configure_datalad_logging(args.datalad_log_level)

    try:
        _ensure_datalad_available()
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    config = _load_config(args.config)
    try:
        dataset_ids = _resolve_datasets(config, args.dataset)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    base_dir = _resolve_data_dir(config, args.data_dir)
    processed_dir = _resolve_processed_dir(config, args.processed_dir, base_dir)
    logger.info("Base data directory: %s", base_dir)
    if args.build_index:
        logger.info("Processed directory: %s", processed_dir)

    exit_code = 0
    for ds_id in dataset_ids:
        try:
            dataset = _clone_if_needed(ds_id, base_dir, update=args.update)
            if args.skip_get:
                logger.info("Skipping DataLad get for %s (per --skip-get)", ds_id)
            else:
                _get_content(
                    dataset,
                    jobs=args.jobs,
                    paths=args.subset,
                    include_derivatives=args.include_derivatives,
                    on_failure=args.on_failure,
                    result_renderer=args.result_renderer,
                    batch_targets=int(args.batch_targets or 0),
                    report_every=int(args.report_every or 0),
                    preview=int(args.targets_preview or 0),
                )
            if args.build_index:
                try:
                    from openneuro import bids_index
                except Exception as exc:  # pragma: no cover - import failure
                    raise RuntimeError("Unable to import bids_index module") from exc
                ds_path = Path(dataset.pathobj)
                logger.info("Building file_index.csv for %s", ds_path)
                index_df = bids_index.build_file_index(ds_path)
                out_dir = processed_dir / ds_id
                out_dir.mkdir(parents=True, exist_ok=True)
                index_path = out_dir / "file_index.csv"
                index_df.to_csv(index_path, index=False)
                logger.info("Saved index to %s (%d rows)", index_path, len(index_df))
        except Exception as exc:  # pragma: no cover - runtime
            logger.error("Failed processing %s: %s", ds_id, exc)
            exit_code = 1

    if exit_code == 0:
        logger.info("All requested datasets fetched successfully.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

