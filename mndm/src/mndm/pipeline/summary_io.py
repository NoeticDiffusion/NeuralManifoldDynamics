"""I/O helpers extracted from summary.py."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from core.io import h5_writer, json_writer

from .regional_mnps import write_block_jacobians_csv, write_regional_mnps_csv
from .stratified_blocks import write_stratified_block_jacobians_csv

logger = logging.getLogger(__name__)


def _with_timestamp(base_name: str, default: str, ts_suffix: str) -> str:
    name = base_name or default
    root, ext = os.path.splitext(name)
    if not ext:
        ext = ".csv"
    return f"{root}_{ts_suffix}{ext}" if ts_suffix else f"{root}{ext}"


def write_regional_csv_outputs(
    *,
    regional_mnps_results: Any,
    regional_mnps_cfg: Mapping[str, Any],
    mnps_dir: Path,
    config: Mapping[str, Any],
    dataset_label: str,
) -> None:
    """Write regional MNPS and block-Jacobian CSV outputs."""
    ts_suffix = mnps_dir.name.split("_")[-1] if "_" in mnps_dir.name else ""
    try:
        csv_name = str(regional_mnps_cfg.get("csv_output", "regional_mnps_subjects.csv"))
        regional_csv = mnps_dir / _with_timestamp(csv_name, "regional_mnps_subjects.csv", ts_suffix)
        write_regional_mnps_csv([regional_mnps_results], regional_csv, append=True)

        block_csv_name = str(
            regional_mnps_cfg.get("block_csv_output", "regional_block_jacobians_subjects.csv")
        )
        block_csv = mnps_dir / _with_timestamp(block_csv_name, "regional_block_jacobians_subjects.csv", ts_suffix)
        write_block_jacobians_csv([regional_mnps_results], config, block_csv, append=True)
    except Exception:
        logger.exception("Failed to write regional MNPS / block Jacobian CSVs for %s", dataset_label)


def write_stratified_blocks_csv_output(
    *,
    stratified_blocks_result: Any,
    config: Mapping[str, Any],
    dataset_id: str,
    mnps_dir: Path,
    dataset_label: str,
) -> None:
    """Write optional stratified (v2) block-Jacobian CSV."""
    if stratified_blocks_result is None or not getattr(stratified_blocks_result, "block_rows", None):
        return
    try:
        v2_cfg_local = config.get("mnps_9d", {}) if isinstance(config, Mapping) else {}
        blocks_cfg: dict[str, Any] = {}
        if isinstance(v2_cfg_local, Mapping):
            base_cfg = (
                v2_cfg_local.get("block_jacobians", {})
                or v2_cfg_local.get("jacobian_blocks", {})
                or {}
            )
            if isinstance(base_cfg, Mapping):
                blocks_cfg.update(dict(base_cfg))

            datasets_cfg = v2_cfg_local.get("datasets", {}) or {}
            if isinstance(datasets_cfg, Mapping):
                ds_cfg = datasets_cfg.get(dataset_id, {}) or {}
                if isinstance(ds_cfg, Mapping):
                    ds_blocks = ds_cfg.get("block_jacobians", {}) or ds_cfg.get("jacobian_blocks", {}) or {}
                    if isinstance(ds_blocks, Mapping):
                        blocks_cfg.update(dict(ds_blocks))

        if bool(blocks_cfg.get("enabled", False)):
            csv_name = str(blocks_cfg.get("csv_output", "stratified_block_jacobians_subjects.csv"))
            ts_suffix = mnps_dir.name.split("_")[-1] if "_" in mnps_dir.name else ""
            out_csv = mnps_dir / _with_timestamp(csv_name, "stratified_block_jacobians_subjects.csv", ts_suffix)
            write_stratified_block_jacobians_csv(stratified_blocks_result.block_rows, out_csv, append=True)
    except Exception:
        logger.exception("Failed to write stratified (v2) block Jacobian CSV for %s", dataset_label)


def write_summary_manifest_and_h5(
    *,
    target_dir: Path,
    dataset_label: str,
    manifest: Mapping[str, Any],
    payload: Any,
    jacobian_diagnostics: Mapping[str, Any],
    sub_id: str,
    ses_id: Optional[str],
    condition: Optional[str],
    task: Optional[str],
    run_id: Optional[str],
    acq_id: Optional[str],
    build_dir_suffix: Callable[[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]], Optional[str]],
) -> None:
    """Write summary JSON and subject H5 outputs."""
    json_path = target_dir / "summary.json"
    json_writer.write_json_summary(manifest, json_path)

    h5_suffix = build_dir_suffix(ses_id, condition, task, run_id, acq_id)
    h5_basename = f"{sub_id}_{h5_suffix}.h5" if h5_suffix else f"{sub_id}.h5"
    h5_path = target_dir / h5_basename
    h5_writer.write_h5(
        h5_path,
        dataset_label,
        payload,
        manifest,
        jacobian_diagnostics=jacobian_diagnostics,
    )
