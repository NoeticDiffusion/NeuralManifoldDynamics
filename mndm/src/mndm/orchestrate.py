"""
orchestrate.py
Pipeline orchestration and command handlers.

This module contains the business logic for each CLI command, keeping cli.py
thin and focused on argument parsing only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from core import config_loader
from . import bids_index
from .file_filters import apply_exclude_file_filters
from .parallel import (
    merge_temp_features,
    process_single_file,
    resolve_feature_io_policy,
    write_intermediate_json,
    write_merged_features,
    write_qc_json,
    write_temp_features_csv,
)
from .pipeline.context import ResolvedConfig, SummarizeContext
from .pipeline.summary import DatasetSummaryRunner
from .pipeline.check_structure import run_structure_check
from .prerequisite_check import format_prerequisite_report, report_to_json, run_prerequisite_check

logger = logging.getLogger(__name__)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logging.warning("psutil not available; memory monitoring disabled")


def _monitor_memory() -> float:
    """Get current memory usage in GB."""
    if not HAS_PSUTIL:
        return 0.0
    try:
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 3)
    except Exception:
        return 0.0


def _available_ram_gb() -> float:
    """Return GB of RAM currently available to this process."""
    if HAS_PSUTIL:
        try:
            return psutil.virtual_memory().available / (1024 ** 3)
        except Exception:
            pass
    return 8.0  # safe fallback when psutil is absent


def _auto_mem_budget_gb() -> float:
    """Compute a safe memory budget from currently available RAM.

    Uses 70 % of available RAM so the OS and other processes keep headroom.
    Always at least 2 GB so small machines don't get clamped to 1 worker.
    """
    return max(2.0, _available_ram_gb() * 0.70)


def _parse_nifti_header_stats(file_path: Path) -> Optional[Dict[str, float]]:
    """Read NIfTI/NIfTI-gz header to obtain array dimensions without loading data.

    Returns a dict with 'raw_array_gb' (uncompressed float32 size in GB) when
    the header can be parsed, else None.
    """
    try:
        import nibabel as nib
        import numpy as np
        img = nib.load(str(file_path))           # header-only, no pixel IO
        shape = img.header.get_data_shape()
        dtype = img.header.get_data_dtype()
        itemsize = int(np.dtype(dtype).itemsize) if dtype is not None else 4
        n_total = 1
        for dim in shape:
            n_total *= max(1, int(dim))
        raw_gb = (n_total * itemsize) / (1024.0 ** 3)
        return {"raw_array_gb": float(raw_gb)}
    except Exception:
        return None


def _estimate_peak_ram_per_file_gb(file_path: Path, file_size_bytes: int) -> float:
    """Backward-compatible wrapper returning only estimated GB."""
    est_gb, _ = _estimate_peak_ram_per_file(file_path, file_size_bytes)
    return float(est_gb)


def _parse_edf_header_stats(file_path: Path) -> Optional[Dict[str, float]]:
    """Parse lightweight EDF header fields for memory estimation."""
    try:
        with file_path.open("rb") as f:
            fixed = f.read(256)
            if len(fixed) < 256:
                return None
            n_records = int((fixed[236:244] or b"0").decode("ascii", errors="ignore").strip() or "0")
            rec_dur = float((fixed[244:252] or b"0").decode("ascii", errors="ignore").strip() or "0")
            n_signals = int((fixed[252:256] or b"0").decode("ascii", errors="ignore").strip() or "0")
            if n_signals <= 0:
                return None
            # EDF per-signal "samples per record" fields start after:
            # labels(16), transducer(80), phys_dim(8), phys_min(8), phys_max(8),
            # dig_min(8), dig_max(8), prefilter(80) => 216 bytes per signal.
            f.seek(256 + 216 * n_signals)
            samples_blob = f.read(8 * n_signals)
            if len(samples_blob) < 8 * n_signals:
                return None
            spr: list[int] = []
            for i in range(n_signals):
                raw = samples_blob[i * 8 : (i + 1) * 8]
                try:
                    spr.append(int(raw.decode("ascii", errors="ignore").strip() or "0"))
                except Exception:
                    spr.append(0)
            spr = [v for v in spr if v > 0]
            if not spr:
                return None
            return {
                "n_records": float(max(n_records, 0)),
                "record_duration_sec": float(max(rec_dur, 0.0)),
                "n_signals": float(n_signals),
                "max_samples_per_record": float(max(spr)),
                "sum_samples_per_record": float(sum(spr)),
            }
    except Exception:
        return None


def _estimate_peak_ram_per_file(file_path: Path, file_size_bytes: int) -> tuple[float, str]:
    """Heuristic peak RAM estimate (GB) with model label.

    Two-step strategy:
    1) Header-aware estimate when EDF header is available.
    2) Conservative size-based fallback by suffix.
    """
    size_gb = max(float(file_size_bytes) / (1024.0 ** 3), 1e-3)
    suffixes = "".join(file_path.suffixes).lower()

    if suffixes.endswith(".edf"):
        hdr = _parse_edf_header_stats(file_path)
        if hdr:
            n_signals = float(hdr.get("n_signals", 0.0) or 0.0)
            n_records = float(hdr.get("n_records", 0.0) or 0.0)
            max_spr = float(hdr.get("max_samples_per_record", 0.0) or 0.0)
            if n_signals > 0 and n_records > 0 and max_spr > 0:
                # Approximate peak worker RAM for EDF preprocessing.
                #
                # The old model assumed a very pessimistic float64-dense path with
                # multiple transient copies and a hard >=1.2 GB floor. Empirical
                # runs have shown typical EEG workers are closer to ~0.5 GB, so we
                # keep a safety margin but calibrate the estimate toward observed
                # worker peaks instead of a worst-case synthetic allocation stack.
                dense_matrix_gb = (n_signals * n_records * max_spr * 8.0) / (1024.0 ** 3)
                # One main dense allocation plus moderate transient overhead.
                est = dense_matrix_gb * 1.15 + 0.12
                # Guardrail against underestimation from malformed headers.
                est = max(est, size_gb * 1.05, 0.45)
                return float(est), "edf_header_v2"
        # Fallback when header parsing fails.
        return float(max(0.45, size_gb * 1.8)), "size_fallback_edf_v2"
    if suffixes.endswith(".vhdr") or suffixes.endswith(".eeg") or suffixes.endswith(".set"):
        return float(max(0.9, size_gb * 3.5)), "size_fallback_eeg_v1"
    if suffixes.endswith(".nii") or suffixes.endswith(".nii.gz"):
        hdr = _parse_nifti_header_stats(file_path)
        if hdr:
            raw_gb = float(hdr.get("raw_array_gb", 0.0) or 0.0)
            if raw_gb > 0:
                # Peak during atlas parcellation:
                #   - full 4D array in RAM (raw_gb)
                #   - one transient copy for resampling/masking (×1.5)
                #   - Python/nilearn/nibabel overhead (+ 0.3 GB)
                est = raw_gb * 1.5 + 0.3
                est = max(est, 0.3)
                return float(est), "nifti_header_v1"
        # Fallback when nibabel header read fails.
        # .nii.gz: compressed on disk, so size_gb is much smaller than RAM usage.
        # Typical gz compression ratio for BOLD data is 4–8×, so we apply 5×
        # as a conservative mid-point, then add 50 % overhead.
        if suffixes.endswith(".nii.gz"):
            return float(max(0.3, size_gb * 7.5)), "size_fallback_nii_gz_v2"
        return float(max(0.3, size_gb * 2.0)), "size_fallback_nii_v2"
    return float(max(0.5, size_gb * 2.5)), "size_fallback_generic_v1"


def _cap_workers_by_memory(
    requested_workers: int,
    file_tasks: list[tuple[Path, str, int]],
    mem_budget_gb: float,
) -> tuple[int, float]:
    """Return a memory-aware worker cap and estimated per-worker peak GB.

    When *mem_budget_gb* is 0 (the CLI default) the budget is determined
    automatically from the RAM currently available to the OS via psutil.
    """
    if requested_workers <= 1 or not file_tasks:
        return max(1, requested_workers), 0.0

    # 0 means "auto": derive budget from available system RAM.
    if mem_budget_gb <= 0:
        mem_budget_gb = _auto_mem_budget_gb()

    worst_case_per_worker_gb = max(
        _estimate_peak_ram_per_file(file_path, file_size)[0]
        for file_path, _, file_size in file_tasks
    )
    if worst_case_per_worker_gb <= 0:
        return max(1, requested_workers), 0.0

    # Reserve 20 % headroom for parent process + transient allocations.
    effective_budget_gb = max(mem_budget_gb * 0.80, 0.5)
    max_by_memory = max(1, int(effective_budget_gb // worst_case_per_worker_gb))
    return min(max(1, requested_workers), max_by_memory), worst_case_per_worker_gb


def cmd_features(
    config: dict,
    dataset_ids: list[str],
    out_dir: Path | None,
    data_dir: Path | None = None,
    subject: str | None = None,
    n_jobs: int = 1,
    mem_budget_gb: float = 4.0,
) -> int:
    """Compute features for preprocessed files with parallel processing."""
    def _normalize_subject_token(value: Any) -> str:
        """Internal helper: normalize subject token."""
        s = str(value).strip()
        if s.lower().startswith("sub-"):
            s = s[4:]
        if s.endswith(".0"):
            try:
                f = float(s)
                if f.is_integer():
                    s = str(int(f))
            except Exception:
                pass
        elif s.isdigit():
            try:
                s = str(int(s))
            except Exception:
                pass
        return s

    try:
        requested_workers = min(n_jobs, multiprocessing.cpu_count())
        logger.info("Requested %s workers for feature computation", requested_workers)

        # Resolve effective memory budget once so both _cap_workers_by_memory and
        # the in-loop memory monitor use the same value.
        effective_mem_budget_gb = _auto_mem_budget_gb() if mem_budget_gb <= 0 else mem_budget_gb
        logger.info("Effective memory budget: %.2f GB", effective_mem_budget_gb)

        parallel_supported = True
        try:
            from joblib import Parallel  # noqa: F401
        except ImportError:
            parallel_supported = False
        except TypeError:
            parallel_supported = False

        if parallel_supported:
            from joblib import Parallel, delayed

        resolved = ResolvedConfig.from_mapping(config, out_dir, data_dir)
        received_dir = resolved.paths.received_dir
        processed_dir = resolved.paths.processed_dir
        def _build_index(ds_id: str, ds_path: Path) -> Optional[pd.DataFrame]:
            """Internal helper: build index."""
            ds_root = bids_index.resolve_dataset_root(config, received_dir, ds_id)
            if not ds_root.exists():
                logger.warning("No file index found for %s and dataset root missing at %s; skipping", ds_id, ds_root)
                return None
            try:
                logger.info("No file index found for %s; building index from %s", ds_id, ds_root)
                index_df = bids_index.build_file_index(ds_root, config=config, dataset_id=ds_id)
                index_df.to_csv(ds_path / "file_index.csv", index=False)
                logger.info("Saved file index: %s", ds_path / "file_index.csv")
                return index_df
            except Exception as e:
                logger.warning("Failed to build file index for %s from %s: %s", ds_id, ds_root, e)
                return None

        for ds_id in dataset_ids:
            logger.info("Computing features for %s", ds_id)
            ds_path = processed_dir / ds_id
            ds_path.mkdir(parents=True, exist_ok=True)
            ds_root = bids_index.resolve_dataset_root(config, received_dir, ds_id)

            io_policy = resolve_feature_io_policy(config, ds_path)
            leftover_temps = list(ds_path.glob("features_*.csv")) + list(ds_path.glob("features_*.parquet"))
            if leftover_temps:
                logger.info("Found %s leftover temp files, cleaning up", len(leftover_temps))
                for temp_file in leftover_temps:
                    try:
                        temp_file.unlink()
                    except Exception as e:
                        logger.warning("Failed to delete %s: %s", temp_file, e)

            index_path = ds_path / "file_index.csv"
            if index_path.exists() and index_path.stat().st_size == 0:
                logger.warning("Empty file_index.csv for %s; rebuilding", ds_id)
                try:
                    index_path.unlink()
                except Exception as e:
                    logger.warning("Failed to remove empty file_index.csv for %s: %s", ds_id, e)
            if not index_path.exists():
                index_df = _build_index(ds_id, ds_path)
                if index_df is None:
                    continue
            else:
                try:
                    index_df = pd.read_csv(index_path)
                except pd.errors.EmptyDataError:
                    logger.warning("Empty file_index.csv for %s; rebuilding", ds_id)
                    try:
                        index_path.unlink()
                    except Exception as e:
                        logger.warning("Failed to remove empty file_index.csv for %s: %s", ds_id, e)
                    index_df = _build_index(ds_id, ds_path)
                    if index_df is None:
                        continue
                except Exception as e:
                    logger.warning("Failed to read file_index.csv for %s: %s", ds_id, e)
                    continue
            if subject:
                subj_str = _normalize_subject_token(subject)
                index_df = index_df[
                    index_df["subject"].map(_normalize_subject_token) == subj_str
                ]
                logger.info("Filtered to subject %s: %s files", subject, len(index_df))
            index_df, excluded_count, excluded_patterns = apply_exclude_file_filters(
                index_df,
                config=config,
                candidate_columns=("path",),
            )
            if excluded_count > 0:
                logger.info(
                    "Excluded %s indexed files for %s via exclude-files=%s",
                    excluded_count,
                    ds_id,
                    excluded_patterns,
                )

            io_policy = resolve_feature_io_policy(config, ds_path, planned_files=len(index_df))
            already_processed_stems: set[str] = set()
            features_candidates = [ds_path / "features.parquet", ds_path / "features.csv"]
            if io_policy.get("read_prefer") == "csv":
                features_candidates = [ds_path / "features.csv", ds_path / "features.parquet"]
            existing_features_path = next((p for p in features_candidates if p.exists()), None)
            if existing_features_path is not None:
                try:
                    existing_features = (
                        pd.read_parquet(existing_features_path)
                        if existing_features_path.suffix.lower() == ".parquet"
                        else pd.read_csv(existing_features_path)
                    )
                    if "file" in existing_features.columns:
                        normalized_files = (
                            existing_features["file"]
                            .dropna()
                            .astype(str)
                            .str.replace("\\", "/", regex=False)
                            .str.split("/")
                            .str[-1]
                        )
                        stems = normalized_files.str.rsplit(".", n=1).str[0]
                        already_processed_stems.update(stems.unique())
                        logger.info(
                            "Found %s already processed files in %s",
                            len(already_processed_stems),
                            existing_features_path.name,
                        )
                except Exception as e:
                    logger.warning("Could not read existing features table to skip files: %s", e)

            intermediate_dir = ds_path / "intermediate"
            if intermediate_dir.exists():
                json_stems = {path.stem for path in intermediate_dir.glob("*.json")}
                if json_stems:
                    already_processed_stems.update(json_stems)
                    logger.info(
                        "Found %s intermediate JSON files (will skip duplicates)",
                        len(json_stems),
                    )

            file_tasks = []
            skipped_count = 0
            for _, row in index_df.iterrows():
                rel = str(row.get("path", "") or "")
                rel_norm = rel.replace("\\", "/")
                base = rel_norm.split("/")[-1] if rel_norm else ""
                if not rel_norm or base.startswith("._") or base.startswith(".") or "/._" in rel_norm:
                    skipped_count += 1
                    continue

                file_path = ds_root / rel
                if file_path.exists():
                    file_stem = file_path.stem
                    if file_stem in already_processed_stems:
                        skipped_count += 1
                        continue
                    file_size = file_path.stat().st_size
                    sub_id = str(row.get("subject", "") or "unknown")
                    file_tasks.append((file_path, rel, file_size, sub_id))

            if skipped_count > 0:
                logger.info("Skipped %s already processed files", skipped_count)

            if not file_tasks:
                logger.warning("No valid files found for %s", ds_id)
                continue

            # Group by subject and sort alphabetically so each worker handles
            # one complete subject at a time rather than random file ordering.
            subject_groups: Dict[str, list[tuple[Path, str, int]]] = {}
            for file_path, rel, file_size, sub_id in file_tasks:
                subject_groups.setdefault(sub_id, []).append((file_path, rel, file_size))
            subject_task_list = sorted(subject_groups.items())  # alphabetical by subject id

            # Flatten back to file_tasks without subject for memory estimation
            file_tasks_flat = [(fp, rel, fs) for _, files in subject_task_list for fp, rel, fs in files]
            n_subjects = len(subject_task_list)
            n_files = len(file_tasks_flat)
            logger.info("Processing %s subjects (%s files) in alphabetical order", n_subjects, n_files)
            file_mem_estimates: Dict[str, Dict[str, Any]] = {}
            for file_path, _, file_size in file_tasks_flat:
                est_gb_i, model_i = _estimate_peak_ram_per_file(file_path, file_size)
                file_mem_estimates[str(file_path)] = {
                    "est_gb": float(est_gb_i),
                    "model": str(model_i),
                }
            worst_mem_path = None
            worst_mem_gb = 0.0
            worst_mem_model = "n/a"
            if file_mem_estimates:
                worst_mem_path, worst_item = max(
                    file_mem_estimates.items(),
                    key=lambda kv: float((kv[1] or {}).get("est_gb", 0.0) or 0.0),
                )
                worst_mem_gb = float((worst_item or {}).get("est_gb", 0.0) or 0.0)
                worst_mem_model = str((worst_item or {}).get("model", "n/a"))

            max_workers, est_worker_gb = _cap_workers_by_memory(
                requested_workers=requested_workers,
                file_tasks=file_tasks_flat,
                mem_budget_gb=effective_mem_budget_gb,
            )
            if max_workers < requested_workers:
                logger.warning(
                    "Reducing workers from %s to %s for %s due to memory budget "
                    "(%.2f GB; est peak per worker %.2f GB)",
                    requested_workers,
                    max_workers,
                    ds_id,
                    effective_mem_budget_gb,
                    est_worker_gb,
                )
                if worst_mem_path:
                    logger.warning(
                        "Memory heuristic driver for %s: file=%s, model=%s, estimate=%.2f GB",
                        ds_id,
                        Path(worst_mem_path).name,
                        worst_mem_model,
                        worst_mem_gb,
                    )
            else:
                logger.info(
                    "Using %s workers for %s (%s subjects, mem budget %.2f GB; est peak per worker %.2f GB)",
                    max_workers,
                    ds_id,
                    n_subjects,
                    effective_mem_budget_gb,
                    est_worker_gb,
                )
                if worst_mem_path:
                    logger.info(
                        "Memory heuristic driver for %s: file=%s, model=%s, estimate=%.2f GB",
                        ds_id,
                        Path(worst_mem_path).name,
                        worst_mem_model,
                        worst_mem_gb,
                    )

            failed_files = []
            successful_count = 0

            class MemoryLimitExceeded(Exception):
                pass

            def _handle_result(res: Any) -> None:
                """Internal helper: handle result."""
                nonlocal successful_count
                try:
                    file_path_obj = Path(res.file_path)
                except Exception:
                    file_path_obj = None

                if res.success:
                    t_write0 = time.perf_counter()
                    t_temp = 0.0
                    t_intermediate = 0.0
                    t_qc = 0.0
                    if res.features_df is not None and len(res.features_df) > 0:
                        t0 = time.perf_counter()
                        file_hash = hashlib.md5(str(res.file_path).encode()).hexdigest()[:8]
                        write_temp_features_csv(
                            res.features_df,
                            ds_path / "features.csv",
                            file_hash,
                            io_policy=io_policy,
                        )
                        t_temp = float(time.perf_counter() - t0)
                    t1 = time.perf_counter()
                    write_intermediate_json(
                        res.features_df,
                        intermediate_dir,
                        file_path_obj.name if file_path_obj is not None else "unknown",
                    )
                    t_intermediate = float(time.perf_counter() - t1)
                    qc_dir = ds_path / "qc_artifacts"
                    t2 = time.perf_counter()
                    write_qc_json(
                        getattr(res, "meta", None),
                        qc_dir,
                        file_path_obj.name if file_path_obj is not None else "unknown",
                    )
                    t_qc = float(time.perf_counter() - t2)
                    t_write_total = float(time.perf_counter() - t_write0)
                    worker_total = float((getattr(res, "timings", {}) or {}).get("worker_total", 0.0) or 0.0)
                    end_to_end = worker_total + t_write_total
                    if end_to_end > 0:
                        logger.info(
                            "Stage timings for %s (end-to-end): worker=%.2fs (%.1f%%), write_temp=%.2fs (%.1f%%), write_intermediate=%.2fs (%.1f%%), write_qc=%.2fs (%.1f%%), write_total=%.2fs (%.1f%%), total=%.2fs (100.0%%)",
                            file_path_obj.name if file_path_obj is not None else str(res.file_path),
                            worker_total,
                            100.0 * worker_total / end_to_end,
                            t_temp,
                            100.0 * t_temp / end_to_end,
                            t_intermediate,
                            100.0 * t_intermediate / end_to_end,
                            t_qc,
                            100.0 * t_qc / end_to_end,
                            t_write_total,
                            100.0 * t_write_total / end_to_end,
                            end_to_end,
                        )
                    # Calibration log: compare pre-run heuristic vs observed per-worker RSS.
                    try:
                        meta = getattr(res, "meta", {}) or {}
                        obs_peak = float(meta.get("worker_rss_gb_peak", 0.0) or 0.0)
                        obs_delta = float(meta.get("worker_rss_gb_delta_peak", 0.0) or 0.0)
                        est_item = file_mem_estimates.get(str(file_path_obj)) if file_path_obj is not None else None
                        if est_item and obs_peak > 0:
                            logger.info(
                                "Memory calibration for %s: heuristic=%.2f GB (model=%s), observed_worker_peak_rss=%.2f GB, observed_delta=%.2f GB",
                                file_path_obj.name if file_path_obj is not None else str(res.file_path),
                                float(est_item.get("est_gb", 0.0) or 0.0),
                                str(est_item.get("model", "n/a")),
                                obs_peak,
                                obs_delta,
                            )
                    except Exception:
                        logger.debug("Memory calibration log skipped for %s", res.file_path)
                    successful_count += 1
                else:
                    try:
                        rel_base = ds_root
                        rel_path = str(Path(res.file_path).resolve().relative_to(rel_base.resolve()))
                    except Exception:
                        rel_path = Path(res.file_path).name if file_path_obj is not None else str(res.file_path)
                    failed_files.append((rel_path, res.error))
                    logger.error("Failed to process %s: %s", rel_path, res.error)

                if successful_count > 0 and successful_count % 10 == 0:
                    mem_gb = _monitor_memory()
                    if mem_gb > effective_mem_budget_gb * 0.95:
                        logger.error("Memory limit exceeded: %.2f GB (budget: %.2f GB). Aborting to prevent OOM.", mem_gb, effective_mem_budget_gb)
                        raise MemoryLimitExceeded("Memory limit exceeded.")
                    elif mem_gb > effective_mem_budget_gb * 0.8:
                        logger.warning("Memory usage high: %.2f GB (budget: %.2f GB)", mem_gb, effective_mem_budget_gb)

            def _process_subject_files(
                sub_id: str,
                files: list[tuple[Path, str, int]],
                cfg: dict,
            ) -> list[Any]:
                """Process all files for one subject sequentially; return list of results."""
                results_list = []
                for fp, _rel, _sz in files:
                    results_list.append(process_single_file(fp, cfg))
                return results_list

            try:
                if parallel_supported:
                    run_cfg = config.get("run", {}) if isinstance(config, dict) else {}
                    batch_size = int(run_cfg.get("parallel_batch_size", 200) or 200)
                    batch_size = max(1, batch_size)
                    for start in range(0, len(subject_task_list), batch_size):
                        batch = subject_task_list[start : start + batch_size]
                        try:
                            iter_results = Parallel(
                                n_jobs=max_workers,
                                backend="loky",
                                prefer="processes",
                                verbose=10,
                                return_as="generator",
                            )(
                                delayed(_process_subject_files)(sub_id, files, config)
                                for sub_id, files in batch
                            )
                            for subject_results in iter_results:
                                for res in (subject_results or []):
                                    _handle_result(res)
                        except TypeError:
                            logger.warning(
                                "Parallel streaming unsupported in this joblib version; "
                                "falling back to non-streaming parallel execution"
                            )
                            results = Parallel(
                                n_jobs=max_workers,
                                backend="loky",
                                prefer="processes",
                                verbose=10,
                            )(
                                delayed(_process_subject_files)(sub_id, files, config)
                                for sub_id, files in batch
                            )
                            for subject_results in results:
                                for res in (subject_results or []):
                                    _handle_result(res)
                        except MemoryLimitExceeded:
                            raise
                        except Exception:
                            logger.exception(
                                "Parallel batch failed (%s-%s); falling back to sequential for this batch only",
                                start,
                                start + len(batch) - 1,
                            )
                            for sub_id, files in batch:
                                for res in _process_subject_files(sub_id, files, config):
                                    _handle_result(res)
                else:
                    for sub_id, files in subject_task_list:
                        logger.info("Processing subject %s (%s file(s))", sub_id, len(files))
                        for res in _process_subject_files(sub_id, files, config):
                            _handle_result(res)
            except MemoryLimitExceeded as e:
                logger.error("Soft abort triggered due to memory limits: %s. Proceeding to merge completed files.", e)

            if failed_files:
                failed_path = ds_path / "failed_files.txt"
                with failed_path.open("w") as f:
                    for file_rel, error in failed_files:
                        f.write(f"{file_rel}: {error}\n")
                logger.warning("Failed files logged to %s (%s failures)", failed_path, len(failed_files))

            temp_count = len(list(ds_path.glob("features_*.csv"))) + len(list(ds_path.glob("features_*.parquet")))
            merged_df = merge_temp_features(ds_path, io_policy=io_policy)
            if len(merged_df) > 0:
                out_paths = write_merged_features(merged_df, ds_path, io_policy=io_policy)
                logger.info(
                    "Saved features tables: %s (%s epochs, merged %s temp files, policy=%s)",
                    ", ".join(sorted(str(p) for p in out_paths.values())) if out_paths else "none",
                    len(merged_df),
                    temp_count,
                    io_policy.get("primary_format"),
                )
            else:
                logger.warning("No features generated for %s", ds_id)

        return 0
    except Exception as e:
        logger.error("Features computation failed: %s", e)
        return 1






def cmd_summarize(
    config: dict,
    dataset_ids: list[str],
    out_dir: Path | None,
    data_dir: Path | None = None,
    subject: str | None = None,
    h5_mode: str = "subject",
    n_jobs: int = 1,
    mnps_overrides: Optional[Dict[str, Any]] = None,
) -> int:
    """Project features to MNPS tensors and write HDF5/JSON outputs."""
    try:
        resolved = ResolvedConfig.from_mapping(config, out_dir, data_dir, mnps_overrides)
        ctx = SummarizeContext.from_resolved(resolved)
    except Exception as exc:
        logger.error(f"Summarization failed: {exc}")
        return 1
    return _summarize_with_context(ctx, dataset_ids, subject, h5_mode, n_jobs=n_jobs)


def cmd_check_structure(
    config: dict,
    dataset_ids: list[str],
    out_dir: Path | None,
    data_dir: Path | None = None,
    check_config_path: Path | None = None,
    run_selector: str | None = None,
    out_report: Path | None = None,
) -> int:
    """Validate summarized outputs (run folders) against a check spec."""
    try:
        resolved = ResolvedConfig.from_mapping(config, out_dir, data_dir)
        processed_base = resolved.paths.processed_dir
        if check_config_path is None:
            raise ValueError("--check-config is required")

        check_cfg = config_loader.load_config(check_config_path)
        scan_cfg = check_cfg.get("scan", {}) if isinstance(check_cfg, dict) else {}
        selector = run_selector or str((scan_cfg.get("run_selector", "latest")) or "latest")

        report = run_structure_check(
            ingest_config=config,
            check_spec=check_cfg,
            dataset_ids=dataset_ids,
            processed_base=processed_base,
            run_selector=selector,
        )
        if out_report is not None:
            out_report.parent.mkdir(parents=True, exist_ok=True)
            out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
            logger.info("Wrote combined report: %s", out_report)

        return 0 if bool(report.get("ok", False)) else 2
    except Exception as exc:
        logger.error("Structure check failed: %s", exc)
        return 1


def cmd_prerequisite_check(
    config: dict,
    dataset_ids: list[str],
    out_dir: Path | None,
    data_dir: Path | None = None,
    out_report: Path | None = None,
) -> int:
    """Run preflight checks before starting an MNDM pipeline run."""
    try:
        report = run_prerequisite_check(
            config=config,
            dataset_ids=dataset_ids,
            out_dir=out_dir,
            data_dir=data_dir,
        )
        logger.info("\n%s", format_prerequisite_report(report))
        if out_report is not None:
            out_report.parent.mkdir(parents=True, exist_ok=True)
            out_report.write_text(report_to_json(report), encoding="utf-8")
            logger.info("Wrote prerequisite report: %s", out_report)
        return 0 if bool(report.get("ok", False)) else 2
    except Exception as exc:
        logger.error("Prerequisite check failed: %s", exc)
        return 1


def _summarize_with_context(
    ctx: SummarizeContext,
    dataset_ids: list[str],
    subject: str | None = None,
    h5_mode: str = "subject",
    n_jobs: int = 1,
) -> int:
    """Internal helper that assumes `SummarizeContext` has been resolved."""
    try:
        for ds_id in dataset_ids:
            DatasetSummaryRunner(ctx, ds_id, subject, h5_mode, n_jobs=n_jobs).run()

        return 0
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return 1
