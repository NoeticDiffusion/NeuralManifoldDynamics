"""
parallel.py
Worker functions for parallel file processing."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
from .reproducibility import resolve_base_seed

logger = logging.getLogger(__name__)

try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except Exception:  # pragma: no cover
    _HAS_PSUTIL = False

def _looks_like_git_annex_placeholder(file_path: Path) -> bool:
    """Detect git-annex/datalad placeholder files (common on Windows).

    On Windows, datalad/git-annex can leave a small text file whose content looks like:
      /annex/objects/SHA256E-s12345--<hash>/<hash>

    These are not valid EDF/VHDR/etc and should be treated as missing data.
    """
    try:
        st = file_path.stat()
        if st.st_size <= 0 or st.st_size > 2048:
            return False
        with file_path.open("rb") as f:
            head = f.read(128)
        return head.startswith(b"/annex/objects/") or b"/annex/objects/" in head
    except Exception:
        return False


def _dedupe_columns(cols: list[str]) -> list[str]:
    """Return a list of unique column names by suffixing duplicates deterministically."""
    seen: Dict[str, int] = {}
    out: list[str] = []
    for c in cols:
        base = str(c)
        n = seen.get(base, 0)
        if n == 0:
            out.append(base)
        else:
            out.append(f"{base}__dup{n}")
        seen[base] = n + 1
    return out


def _json_numpy_default(obj: Any) -> Any:
    """JSON serializer for NumPy scalars/arrays used in QC metadata."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _process_rss_gb() -> float:
    """Return current worker RSS in GB (0 when unavailable)."""
    if not _HAS_PSUTIL:
        return 0.0
    try:
        return float(psutil.Process().memory_info().rss) / (1024.0 ** 3)
    except Exception:
        return 0.0


def resolve_feature_io_policy(
    config: Mapping[str, Any] | None,
    ds_path: Path,
    planned_files: Optional[int] = None,
) -> Dict[str, Any]:
    """Resolve feature table IO policy from config with auto-size heuristics."""
    cfg = (config or {}).get("feature_storage", {}) if isinstance(config, Mapping) else {}
    default_format = str(cfg.get("default_format", "csv")).strip().lower() or "csv"
    if default_format not in {"csv", "parquet", "auto"}:
        default_format = "csv"
    read_prefer = str(cfg.get("read_prefer", "parquet")).strip().lower() or "parquet"
    if read_prefer not in {"csv", "parquet"}:
        read_prefer = "parquet"
    write_both = bool(cfg.get("write_both", False))

    auto_cfg = cfg.get("auto_large_dataset", {}) if isinstance(cfg, Mapping) else {}
    min_files = int(auto_cfg.get("min_files", 3000) or 3000)
    min_total_mb = float(auto_cfg.get("min_total_mb", 512.0) or 512.0)

    total_mb = 0.0
    try:
        for p in ds_path.glob("features_*.*"):
            if p.suffix.lower() in {".csv", ".parquet"} and p.is_file():
                total_mb += float(p.stat().st_size) / (1024.0 * 1024.0)
        for p in (ds_path / "features.csv", ds_path / "features.parquet"):
            if p.exists() and p.is_file():
                total_mb += float(p.stat().st_size) / (1024.0 * 1024.0)
    except Exception:
        pass

    is_large = (int(planned_files or 0) >= min_files) or (total_mb >= min_total_mb)
    primary_format = "parquet" if (default_format == "auto" and is_large) else default_format
    if primary_format == "auto":
        primary_format = "csv"

    write_csv = write_both or primary_format == "csv"
    write_parquet = write_both or primary_format == "parquet"

    return {
        "primary_format": primary_format,
        "read_prefer": read_prefer,
        "write_csv": bool(write_csv),
        "write_parquet": bool(write_parquet),
        "write_both": bool(write_both),
        "is_large_dataset": bool(is_large),
        "planned_files": int(planned_files or 0),
        "estimated_existing_features_mb": float(total_mb),
    }


def _read_feature_table(path: Path) -> pd.DataFrame:
    """Read CSV/Parquet feature table by suffix."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _merge_feature_frames(feature_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge per-modality feature frames into a single frame.

    Strategy:
    - If frames expose `epoch_id`, align on it by using it as an index (outer join).
      This avoids duplicate `epoch_id` columns when concatenating EEG/EOG/ECG/etc.
    - As a last resort, enforce unique column names to keep JSON export robust.
    """
    dfs = [df for df in feature_dfs if isinstance(df, pd.DataFrame) and not df.empty]
    if not dfs:
        return pd.DataFrame()

    has_epoch = any("epoch_id" in df.columns for df in dfs)
    if has_epoch:
        # Choose a base epoch_id sequence to align any df missing epoch_id but
        # having the same length (rare, but keeps things resilient).
        base_epoch: Optional[np.ndarray] = None
        for df in dfs:
            if "epoch_id" in df.columns:
                base_epoch = df["epoch_id"].to_numpy()
                break

        aligned: list[pd.DataFrame] = []
        for df in dfs:
            df2 = df.copy()
            if "epoch_id" in df2.columns:
                df2 = df2.set_index("epoch_id")
            elif base_epoch is not None and len(df2) == len(base_epoch):
                logger.warning(
                    "Forcing epoch_id alignment by length-only fallback (len=%s); "
                    "verify modality timebases for potential shifts.",
                    len(df2),
                )
                df2.index = base_epoch
                df2.index.name = "epoch_id"
            aligned.append(df2)

        merged = pd.concat(aligned, axis=1, join="outer")
        merged = merged.reset_index()
    else:
        merged = pd.concat(dfs, axis=1, join="outer")

    # Ensure unique columns for safe JSON export and later downstream reads.
    if merged.columns.duplicated().any():
        merged.columns = _dedupe_columns([str(c) for c in merged.columns])

    # Cross-modality resolver for embodiment axis:
    # prefer ECG RMSSD, else EOG blink-rate, else EEG high-frequency power.
    if len(merged) > 0:
        try:
            proxy = np.full(len(merged), np.nan, dtype=np.float64)
            source = np.full(len(merged), "", dtype=object)
            for col_name, source_name in (
                ("ecg_rmssd", "ecg_rmssd"),
                ("eog_blink_rate", "eog_blink_rate"),
                ("eeg_highfreq_power_30_45", "eeg_highfreq_power_30_45"),
            ):
                if col_name not in merged.columns:
                    continue
                vals = pd.to_numeric(merged[col_name], errors="coerce").to_numpy(dtype=np.float64, copy=False)
                use = np.isfinite(vals) & ~np.isfinite(proxy)
                if np.any(use):
                    proxy[use] = vals[use]
                    source[use] = source_name
            merged["embodied_arousal_proxy"] = proxy.astype(np.float32)
            merged["embodied_arousal_proxy_source"] = source
        except Exception:
            logger.exception("Failed to compute embodied_arousal_proxy; continuing without resolver feature")

    return merged


@dataclass
class WorkerResult:
    """Result from worker processing."""
    success: bool
    file_path: str
    features_df: pd.DataFrame | None = None
    error: str | None = None
    meta: Optional[Dict[str, Any]] = None
    timings: Optional[Dict[str, float]] = None


def worker_init(base_seed: int, worker_id: int):
    """Initialize worker process with BLAS threads and seed.
    
    Parameters
    ----------
    base_seed
        Base seed from config.
    worker_id
        Unique worker identifier.
    """
    # Set BLAS threads before importing NumPy/SciPy
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    # Do not set RNG here: per-file seeds in process_single_file ensure
    # deterministic outputs independent of task scheduling/worker assignment.
    logger.debug("Worker %s initialized (BLAS threads pinned to 1)", worker_id)


def _set_blas_threads():
    """Set BLAS environment variables to prevent oversubscription."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _stable_seed_token(file_path: Path, config: Dict[str, Any]) -> str:
    """Build a machine-stable token for per-file RNG seeding.

    Prefer dataset-relative path (<dataset_id>/<relative_path>) when possible
    so seeds remain stable across different absolute mount roots.
    """
    try:
        path_norm = file_path.as_posix()
        cfg_datasets = config.get("datasets", [])
        dataset_ids = [str(d.get("id", d)) if isinstance(d, dict) else str(d) for d in cfg_datasets]
        for ds in dataset_ids:
            token = f"/{ds}/"
            idx = path_norm.find(token)
            if idx >= 0:
                rel = path_norm[idx + 1 :]  # keep dataset id in token
                return rel.replace("\\", "/").lower()
    except Exception:
        pass
    # Fallback: filename only (stable enough for reproducibility, but less unique).
    return file_path.name.lower()


def process_single_file(file_path: Path, config: Dict[str, Any]) -> WorkerResult:
    """Process a single file and return features DataFrame.
    
    This function is designed to be called in a worker process.
    It sets BLAS threads to 1 and performs cleanup after processing.
    
    Parameters
    ----------
    file_path
        Path to the EEG file to process.
    config
        Configuration dictionary.
    
    Returns
    -------
    WorkerResult with success status and features DataFrame.
    """
    _set_blas_threads()
    
    preprocessed = None
    stage_times: Dict[str, float] = {}
    rss_start = _process_rss_gb()
    rss_peak = rss_start

    def _track_rss() -> None:
        nonlocal rss_peak
        r = _process_rss_gb()
        if r > rss_peak:
            rss_peak = r

    def _attach_memory_meta(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        out = dict(meta or {})
        start = float(rss_start) if rss_start > 0 else None
        peak = float(rss_peak) if rss_peak > 0 else None
        out["worker_rss_gb_start"] = start
        out["worker_rss_gb_peak"] = peak
        out["worker_rss_gb_delta_peak"] = (
            float(rss_peak - rss_start) if (rss_start > 0 and rss_peak > 0) else None
        )
        return out
    try:
        # Deterministic per-file RNG seed (so parallel vs sequential runs match).
        try:
            base_seed, _ = resolve_base_seed(config)
            seed_token = _stable_seed_token(file_path, config)
            h = hashlib.md5(seed_token.encode("utf-8")).hexdigest()
            file_seed = base_seed + (int(h[:8], 16) % 1_000_000)
            np.random.seed(file_seed)
        except Exception:
            pass

        # Robust skip for git-annex/datalad placeholder files (unfetched content).
        try:
            if _looks_like_git_annex_placeholder(file_path):
                logger.warning("Skipping git-annex placeholder (unfetched): %s", file_path)
                return WorkerResult(
                    success=True,
                    file_path=str(file_path),
                    features_df=None,
                    meta={"skipped": True, "reason": "git_annex_placeholder"},
                )
        except Exception:
            pass

        # Robust skip for macOS AppleDouble / dotfiles that can sneak into datasets
        # when copied from macOS (e.g. "._sub-..._bold.nii.gz").
        try:
            name = file_path.name
            if name.startswith("._") or name.startswith(".") or "/._" in str(file_path).replace("\\", "/"):
                return WorkerResult(
                    success=True,
                    file_path=str(file_path),
                    features_df=None,
                    meta={"skipped": True, "reason": "dotfile/appledouble"},
                )
        except Exception:
            pass

        t0 = time.perf_counter()
        from . import preprocess
        from .features import eeg, eog, emg, ecg, resp, eda, fmri
        
        preprocessed = preprocess.preprocess_file(file_path, config)
        _track_rss()
        t1 = time.perf_counter()
        stage_times["preprocess"] = float(t1 - t0)
        try:
            pre_t = (preprocessed.meta or {}).get("timings", {})
            if isinstance(pre_t, Mapping):
                for key, value in pre_t.items():
                    try:
                        stage_times[f"pre_{str(key)}"] = float(value)
                    except Exception:
                        continue
        except Exception:
            pass
        
        # Normalize payload to plain dict and include metadata needed for
        # robustness features (e.g. channel-shift ensembles).
        sig_payload = {
            "signals": preprocessed.signals,
            "sfreq": preprocessed.sfreq,
            "channels": preprocessed.channels,
            "dataset_id": preprocessed.meta.get("dataset_id"),
            # Feature extractors sometimes need to locate sidecars (events/channels)
            # relative to the raw file path.
            "file_path": preprocessed.meta.get("file"),
        }
        
        feature_dfs = []
        feature_stage_times: Dict[str, float] = {}
        mod_handlers = {
            "eeg": eeg.compute_eeg_features,
            "eog": eog.compute_eog_features,
            "emg": emg.compute_emg_features,
            "ecg": ecg.compute_ecg_features,
            "resp": resp.compute_resp_features,
            "eda": eda.compute_eda_features,
            "fmri": fmri.compute_fmri_features,
        }
        
        # Only compute features for modalities that exist in the signals
        available_modalities = set(preprocessed.signals.keys())
        for mod_name, mod_func in mod_handlers.items():
            if mod_name in available_modalities:
                t_mod0 = time.perf_counter()
                mod_features = mod_func(sig_payload, config)
                _track_rss()
                feature_stage_times[f"feature_{mod_name}"] = float(time.perf_counter() - t_mod0)
                if len(mod_features) > 0:
                    feature_dfs.append(mod_features)
        t2 = time.perf_counter()
        stage_times.update(feature_stage_times)
        stage_times["feature_extract_total"] = float(t2 - t1)
        stage_times["merge_modalities"] = 0.0
        
        if feature_dfs:
            t_merge0 = time.perf_counter()
            merged = _merge_feature_frames(feature_dfs)
            _track_rss()
            stage_times["merge_modalities"] = float(time.perf_counter() - t_merge0)
            merged["file"] = str(file_path.name)
            total = float(time.perf_counter() - t0)
            stage_times["worker_total"] = total
            logger.info(
                "Stage timings for %s (worker): %s",
                file_path.name,
                _format_timing_breakdown(stage_times, total_key="worker_total"),
            )
            return WorkerResult(
                success=True,
                file_path=str(file_path),
                features_df=merged,
                meta=_attach_memory_meta(preprocessed.meta),
                timings=stage_times,
            )
        
        total = float(time.perf_counter() - t0)
        stage_times["worker_total"] = total
        logger.info(
            "Stage timings for %s (worker): %s",
            file_path.name,
            _format_timing_breakdown(stage_times, total_key="worker_total"),
        )
        return WorkerResult(
            success=True,
            file_path=str(file_path),
            features_df=None,
            meta=_attach_memory_meta(preprocessed.meta),
            timings=stage_times,
        )
    except Exception as e:
        # Log full traceback for easier debugging, but return a structured error
        logger.exception("Failed to process %s", file_path)
        return WorkerResult(
            success=False,
            file_path=str(file_path),
            error=str(e),
            meta=_attach_memory_meta(preprocessed.meta if preprocessed is not None else None),
            timings=stage_times if stage_times else None,
        )
    finally:
        # Aggressive cleanup removed as it can trigger crashes during worker shutdown on Windows.
        pass


def _format_timing_breakdown(stage_times: Mapping[str, float], total_key: str = "worker_total") -> str:
    """Format stage timing as `name: sec (pct)` sorted by duration."""
    total = float(stage_times.get(total_key, 0.0) or 0.0)
    if total <= 0:
        return "timing unavailable"

    entries = []
    for key, value in stage_times.items():
        if key == total_key:
            continue
        sec = float(value or 0.0)
        pct = 100.0 * sec / total if total > 0 else 0.0
        entries.append((sec, f"{key}={sec:.2f}s ({pct:.1f}%)"))
    entries.sort(key=lambda x: x[0], reverse=True)
    parts = [x[1] for x in entries]
    parts.append(f"{total_key}={total:.2f}s (100.0%)")
    return ", ".join(parts)


def write_temp_features_csv(
    features_df: pd.DataFrame,
    csv_path: Path,
    file_hash: str,
    io_policy: Optional[Mapping[str, Any]] = None,
):
    """Write temp features in CSV/Parquet according to IO policy.
    
    Parameters
    ----------
    features_df
        DataFrame to write.
    csv_path
        Base path for CSV file.
    file_hash
        Hash to make filename unique.
    """
    if len(features_df) == 0:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    base = csv_path.parent / f"features_{file_hash}"
    write_csv = bool((io_policy or {}).get("write_csv", True))
    write_parquet = bool((io_policy or {}).get("write_parquet", False))

    if write_csv:
        temp_csv = base.with_suffix(".csv")
        features_df.to_csv(temp_csv, index=False)
        logger.debug("Wrote temp features CSV: %s", temp_csv)
    if write_parquet:
        temp_parquet = base.with_suffix(".parquet")
        try:
            features_df.to_parquet(temp_parquet, index=False)
            logger.debug("Wrote temp features Parquet: %s", temp_parquet)
        except Exception as exc:
            logger.warning("Failed to write temp parquet %s: %s", temp_parquet, exc)


def write_qc_json(meta: Optional[Dict[str, Any]], target_dir: Path, file_name: str) -> None:
    """Persist per-file QC / artifact metadata as JSON.

    Parameters
    ----------
    meta
        Metadata dict returned from preprocessing (PreprocessedSignals.meta).
    target_dir
        Directory where QC JSON files should be written.
    file_name
        Original EEG file name (used to derive QC filename).
    """
    if not meta:
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(file_name).stem
    qc_path = target_dir / f"{stem}_qc_artifacts.json"
    try:
        with qc_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=_json_numpy_default)
        logger.info("Wrote QC JSON: %s", qc_path)
    except Exception as exc:
        logger.warning("Failed to write QC JSON %s: %s", qc_path, exc)


def write_intermediate_json(features_df: pd.DataFrame, target_dir: Path, file_name: str) -> None:
    """Persist per-file features as JSON for resilience and debugging."""
    if features_df is None or len(features_df) == 0:
        logger.info("Skipping intermediate JSON for %s: empty features", file_name)
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / f"{Path(file_name).stem}.json"
    try:
        features_df.to_json(json_path, orient="records")
        logger.info("Wrote intermediate JSON: %s (%d rows)", json_path, len(features_df))
    except Exception as exc:
        logger.warning("Failed to write intermediate JSON %s: %s", json_path, exc)


def merge_temp_features(ds_path: Path, io_policy: Optional[Mapping[str, Any]] = None) -> pd.DataFrame:
    """Merge temporary feature CSV files into single DataFrame.
    
    Parameters
    ----------
    ds_path
        Dataset directory path.
    
    Returns
    -------
    Merged DataFrame of all features.
    """
    read_prefer = str((io_policy or {}).get("read_prefer", "parquet")).lower()
    if read_prefer not in {"csv", "parquet"}:
        read_prefer = "parquet"

    temp_candidates = sorted(
        [
            p
            for p in ds_path.glob("features_*.*")
            if p.suffix.lower() in {".csv", ".parquet"}
        ]
    )
    by_stem: Dict[str, Dict[str, Path]] = {}
    for p in temp_candidates:
        ext = p.suffix.lower().lstrip(".")
        by_stem.setdefault(p.stem, {})[ext] = p
    temp_files: list[Path] = []
    for stem, exts in by_stem.items():
        chosen = exts.get(read_prefer) or exts.get("parquet") or exts.get("csv")
        if chosen is not None:
            temp_files.append(chosen)
    temp_files = sorted(temp_files)

    features_parquet = ds_path / "features.parquet"
    features_csv = ds_path / "features.csv"
    features_path = features_parquet if (read_prefer == "parquet" and features_parquet.exists()) else features_csv
    if not features_path.exists() and read_prefer == "csv" and features_parquet.exists():
        features_path = features_parquet
    if not features_path.exists() and read_prefer == "parquet" and features_csv.exists():
        features_path = features_csv

    if not temp_files and not features_path.exists():
        return pd.DataFrame()

    dfs = []
    if features_path.exists():
        try:
            existing_df = _read_feature_table(features_path)
            if not existing_df.empty:
                dfs.append(existing_df)
        except Exception as e:
            logger.warning("Failed to read existing %s: %s", features_path.name, e)

    for temp_file in temp_files:
        try:
            df = _read_feature_table(temp_file)
            dfs.append(df)
        except Exception as e:
            logger.warning("Failed to read %s: %s", temp_file, e)

    if not dfs:
        return pd.DataFrame()

    merged = pd.concat(dfs, ignore_index=True)
    if {"file", "epoch_id"}.issubset(set(merged.columns)):
        merged = merged.drop_duplicates(subset=["file", "epoch_id"], keep="last")
    else:
        merged = merged.drop_duplicates()

    # Clean up temp files (all temp variants, not just chosen read-preference).
    for temp_file in temp_candidates:
        try:
            temp_file.unlink()
        except Exception as e:
            logger.warning("Failed to delete %s: %s", temp_file, e)

    # Optional ordering for readability
    if "file" in merged.columns and "epoch_id" in merged.columns:
        merged = merged.sort_values(by=["file", "epoch_id"]).reset_index(drop=True)

    return merged


def write_merged_features(
    merged_df: pd.DataFrame,
    ds_path: Path,
    io_policy: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Path]:
    """Write merged features in configured output formats."""
    out_paths: Dict[str, Path] = {}
    if merged_df is None or len(merged_df) == 0:
        return out_paths
    write_csv = bool((io_policy or {}).get("write_csv", True))
    write_parquet = bool((io_policy or {}).get("write_parquet", False))
    if write_csv:
        csv_path = ds_path / "features.csv"
        merged_df.to_csv(csv_path, index=False)
        out_paths["csv"] = csv_path
    if write_parquet:
        parquet_path = ds_path / "features.parquet"
        try:
            merged_df.to_parquet(parquet_path, index=False)
            out_paths["parquet"] = parquet_path
        except Exception as exc:
            logger.warning("Failed to write merged parquet %s: %s", parquet_path, exc)
    return out_paths

