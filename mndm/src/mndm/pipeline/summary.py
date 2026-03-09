""" 
Summary.py
Dataset- and subject-level MNPS summarization runners.
"""

from __future__ import annotations

import multiprocessing
import hashlib
import json
import logging
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import re
from threading import Lock
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from core.bids import parse_subject_session, parse_subject_session_task_run_acq
from .context import SummarizeContext
from .. import bids_index
from .extractors import (
    build_dataset_label,
    extract_embodied_array,
    extract_events,
    extract_mapped_metadata,
    extract_stage_array,
    load_participant_table,
)
from .extensions_compute import compute_extensions
from .regional_mnps import (
    compute_block_jacobian_rows,
    summary_to_dataframe_rows,
)
from .summary_regional import compute_regional_context
from .stratified_blocks import (
    compute_stratified_blocks_and_cross_partials,
)
from .summary_io import (
    write_regional_csv_outputs,
    write_stratified_blocks_csv_output,
    write_summary_manifest_and_h5,
)
from .summary_selectors import (
    load_regional_fmri_signals,
    resolve_bold_path_for_subframe,
)
from .summary_events import (
    infer_stage_from_bids_events,
    estimate_coverage_seconds,
    map_events_to_labels,
)
from .summary_qc import write_qc_files
from .summary_utils import (
    apply_fd_censoring,
    build_dir_suffix,
    extract_time_bounds,
)
from .robustness_helpers import (
    compute_dist_summary,
    compute_emmi_metrics,
    compute_tau_summary,
    compute_tier2_jacobian_metrics,
    compute_ensemble_summary_for_subject,
    compute_psd_multiverse_stability,
    compute_robust_and_reliability_summaries,
)
from .. import preprocess
from core.io import json_writer
from .. import jacobian, projection, robustness, schema
from .run_manifest import write_run_manifest

logger = logging.getLogger(__name__)


def _stable_hash_mapping(value: Mapping[str, Any]) -> str:
    """Hash a mapping deterministically for provenance/versioning."""
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    except Exception:
        payload = str(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_hash_array(value: np.ndarray) -> str:
    """Hash an ndarray deterministically for provenance checks."""
    arr = np.ascontiguousarray(value)
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()


def _rows_to_columnar_table(rows: list[Mapping[str, Any]]) -> Dict[str, np.ndarray]:
    """Convert row dicts into a columnar mapping suitable for HDF5 datasets."""
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    out: Dict[str, np.ndarray] = {}
    for col in frame.columns:
        series = frame[col]
        if pd.api.types.is_numeric_dtype(series):
            out[str(col)] = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32)
        else:
            out[str(col)] = series.fillna("").astype(str).to_numpy(dtype=str)
    return out


def _missing_weighted_feature_rate(
    frame: pd.DataFrame,
    weighted_features: list[str],
) -> float:
    """Fraction of missing/non-finite weighted feature cells.

    Features absent from the frame count as fully missing across all rows.
    """
    if len(frame) == 0 or not weighted_features:
        return 0.0
    total = float(len(frame) * len(weighted_features))
    missing = 0.0
    for feat in weighted_features:
        if feat not in frame.columns:
            missing += float(len(frame))
            continue
        col = pd.to_numeric(frame[feat], errors="coerce")
        missing += float((~np.isfinite(col.to_numpy(dtype=np.float64, copy=False))).sum())
    return float(missing / total) if total > 0 else 0.0


def _validate_e_e_subcoord_construct(subcoords_spec: Mapping[str, Any]) -> None:
    """Validate e_e subcoordinate against allowed energetic-complexity features.

    For fMRI v2.0 we allow signal-power and related robust proxies in addition
    to entropy-named metrics.
    """
    if not isinstance(subcoords_spec, Mapping):
        return
    e_e_weights = subcoords_spec.get("e_e")
    if e_e_weights is None:
        return
    if not isinstance(e_e_weights, Mapping) or not e_e_weights:
        raise ValueError("mnps_9d.subcoords.e_e must map to at least one supported energetic-complexity feature")
    allowed_exact = {
        "fmri_signal_power",
        "fmri_slow4_slow5_ratio",
        "fmri_ar1_coefficient",
    }
    invalid = []
    for name in e_e_weights.keys():
        key = str(name).strip()
        low = key.lower()
        if "entropy" in low:
            continue
        if key in allowed_exact:
            continue
        invalid.append(key)
    if invalid:
        raise ValueError(
            "mnps_9d.subcoords.e_e must map to supported energetic-complexity feature(s); "
            f"invalid entries: {invalid}"
        )


def _resolve_entropy_provenance(frame: pd.DataFrame) -> Dict[str, Any]:
    """Extract energetic-complexity metric provenance from feature frame."""
    construct = "energetic_complexity"
    metric = "permutation_entropy"
    backend = "numpy"
    degraded_mode = False
    reason = None
    if len(frame) == 0:
        return {
            "construct": construct,
            "metric": metric,
            "backend": backend,
            "degraded_mode": degraded_mode,
            "reason": reason,
        }

    def _first_mode(col_name: str) -> Optional[str]:
        if col_name not in frame.columns:
            return None
        series = frame[col_name].dropna()
        if series.empty:
            return None
        return str(series.astype(str).mode(dropna=True).iloc[0])

    construct = _first_mode("eeg_entropy_construct") or construct
    metric = _first_mode("eeg_entropy_metric") or metric
    backend = _first_mode("eeg_entropy_backend") or backend
    reason = _first_mode("eeg_entropy_reason")
    if "eeg_entropy_degraded_mode" in frame.columns:
        degraded_series = pd.to_numeric(frame["eeg_entropy_degraded_mode"], errors="coerce")
        degraded_mode = bool(np.nanmax(degraded_series.to_numpy(dtype=np.float64, copy=False)) > 0)

    return {
        "construct": construct,
        "metric": metric,
        "backend": backend,
        "degraded_mode": degraded_mode,
        "reason": reason,
    }


def _canonical_mde_from_v2_map() -> Dict[str, List[str]]:
    """Canonical v2->3D mapping used when mnps_3d.from_v2.map is omitted."""
    return {
        "m": ["m_a", "m_e", "m_o"],
        "d": ["d_n", "d_l", "d_s"],
        "e": ["e_e", "e_s", "e_m"],
    }


def _resolve_mnps_3d_cfg(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve 3D derivation strategy from config with stable defaults."""
    m3d_cfg = config.get("mnps_3d", {}) if isinstance(config, Mapping) else {}
    if not isinstance(m3d_cfg, Mapping):
        m3d_cfg = {}
    mode = str(m3d_cfg.get("mode", "direct_features")).strip().lower() or "direct_features"
    if mode not in {"from_v2", "direct_features"}:
        logger.warning("Unknown mnps_3d.mode '%s'; using direct_features", mode)
        mode = "direct_features"
    from_v2_cfg = m3d_cfg.get("from_v2", {}) if isinstance(m3d_cfg.get("from_v2", {}), Mapping) else {}
    pooling = str(from_v2_cfg.get("pooling", "mean")).strip().lower() or "mean"
    if pooling not in {"mean", "sum"}:
        logger.warning("Unknown mnps_3d.from_v2.pooling '%s'; using mean", pooling)
        pooling = "mean"
    aggregation_requested = str(from_v2_cfg.get("aggregation", "auto")).strip().lower() or "auto"
    if aggregation_requested not in {
        "auto",
        "fixed_weighted_projection",
        "group_pooling_mean",
        "group_pooling_sum",
    }:
        logger.warning(
            "Unknown mnps_3d.from_v2.aggregation '%s'; using auto",
            aggregation_requested,
        )
        aggregation_requested = "auto"
    map_cfg = from_v2_cfg.get("map", {})
    default_map = _canonical_mde_from_v2_map()
    resolved_map: Dict[str, List[str]] = {}
    for axis in ("m", "d", "e"):
        raw = map_cfg.get(axis) if isinstance(map_cfg, Mapping) else None
        if isinstance(raw, list) and raw:
            resolved_map[axis] = [str(v) for v in raw]
        else:
            resolved_map[axis] = list(default_map[axis])

    # New fixed linear mapping policy (preferred): mnps_projection.v1_mapping
    proj_cfg = config.get("mnps_projection", {}) if isinstance(config, Mapping) else {}
    v1_mapping_cfg = proj_cfg.get("v1_mapping", {}) if isinstance(proj_cfg, Mapping) else {}
    v1_mapping: Dict[str, Dict[str, float]] = {}
    v1_mapping_source = "mnps_projection.v1_mapping"
    if isinstance(v1_mapping_cfg, Mapping) and v1_mapping_cfg:
        for axis in ("m", "d", "e"):
            row = v1_mapping_cfg.get(axis, {})
            if isinstance(row, Mapping):
                v1_mapping[axis] = {str(k): float(v) for k, v in row.items()}
            else:
                v1_mapping[axis] = {}
    else:
        # Keep empty so runtime can fail fast for from_v2 mode when mapping is missing.
        for axis in ("m", "d", "e"):
            v1_mapping[axis] = {}
        v1_mapping_source = "missing"
    has_v1_mapping = any(bool(v1_mapping.get(axis, {})) for axis in ("m", "d", "e"))
    if aggregation_requested == "auto":
        aggregation = "fixed_weighted_projection" if has_v1_mapping else f"group_pooling_{pooling}"
    else:
        aggregation = aggregation_requested

    return {
        "mode": mode,
        "legacy_pooling": pooling,
        "aggregation_requested": aggregation_requested,
        "aggregation": aggregation,
        "map": resolved_map,
        "v1_mapping": v1_mapping,
        "v1_mapping_source": v1_mapping_source,
        "has_v1_mapping": bool(has_v1_mapping),
    }


def _coerce_v1_mapping_to_v2_subcoords(
    v1_mapping: Mapping[str, Any],
    subcoords_spec: Mapping[str, Any],
) -> Dict[str, Dict[str, float]]:
    """Map feature-level V1 weights onto v2 subcoordinate names when possible."""
    out: Dict[str, Dict[str, float]] = {"m": {}, "d": {}, "e": {}}
    if not isinstance(v1_mapping, Mapping):
        return out

    feature_to_subcoords: Dict[str, Dict[str, List[str]]] = {"m": {}, "d": {}, "e": {}}
    feature_to_subcoords_any: Dict[str, List[str]] = {}
    if isinstance(subcoords_spec, Mapping):
        for sub_name, sub_weights in subcoords_spec.items():
            sub = str(sub_name)
            axis = sub[:1]
            if axis not in {"m", "d", "e"}:
                continue
            if not isinstance(sub_weights, Mapping):
                continue
            for feat_name in sub_weights.keys():
                feat = str(feat_name)
                feature_to_subcoords[axis].setdefault(feat, []).append(sub)
                feature_to_subcoords_any.setdefault(feat, []).append(sub)

    for axis in ("m", "d", "e"):
        row = v1_mapping.get(axis, {}) if isinstance(v1_mapping, Mapping) else {}
        if not isinstance(row, Mapping):
            continue
        for name, weight_raw in row.items():
            try:
                weight = float(weight_raw)
            except Exception:
                continue
            if not np.isfinite(weight):
                continue
            key = str(name)
            if key.startswith(f"{axis}_"):
                out[axis][key] = out[axis].get(key, 0.0) + weight
                continue
            mapped = feature_to_subcoords[axis].get(key, [])
            if not mapped:
                # Allow explicit cross-block priors (e.g. m-axis can include d_s-derived signal).
                mapped = feature_to_subcoords_any.get(key, [])
            if mapped:
                per = weight / float(len(mapped))
                for sub in mapped:
                    out[axis][sub] = out[axis].get(sub, 0.0) + per
            else:
                # Preserve unknown keys so runtime validation can fail explicitly.
                out[axis][key] = out[axis].get(key, 0.0) + weight
    return out


def _align_v2_subcoords(
    coords_9d: np.ndarray,
    names: List[str],
    ordered_names: List[str],
) -> np.ndarray:
    """Align v2 subcoordinate matrix columns to canonical ordering when available."""
    arr = np.asarray(coords_9d, dtype=np.float64)
    if arr.ndim != 2 or not names:
        return arr
    idx = {str(name): i for i, name in enumerate(names)}
    if all(name in idx for name in ordered_names):
        order_idx = [idx[name] for name in ordered_names]
        return arr[:, order_idx]
    return arr


@lru_cache(maxsize=1)
def _get_env_provenance() -> Dict[str, Any]:
    """Collect lightweight runtime environment provenance."""
    py_ver = sys.version.replace("\n", " ").strip()
    plat = platform.platform()
    pip_freeze_hash = None
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
        )
        pip_freeze_hash = hashlib.sha256(out.encode("utf-8")).hexdigest()
    except Exception:
        pip_freeze_hash = None
    env_payload = {
        "python_version": py_ver,
        "platform": plat,
        "pip_freeze_hash": pip_freeze_hash,
    }
    env_payload["env_hash"] = _stable_hash_mapping(env_payload)
    return env_payload


def _deep_merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep merge dictionaries: override wins; nested mappings are merged recursively."""
    out: Dict[str, Any] = dict(base) if isinstance(base, Mapping) else {}
    if not isinstance(override, Mapping):
        return out
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge_dict(out.get(k, {}), v)
        else:
            out[k] = v
    return out


@lru_cache(maxsize=128)
def _load_mnps_9d_policy(policy_dir: str, dataset_id: str) -> Dict[str, Any]:
    """Load optional per-dataset mnps_9d policy from YAML."""
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    if not policy_dir or not dataset_id:
        return {}

    root = Path(policy_dir)
    ds_path = root / f"{dataset_id}_mnps_9d.yml"
    map_path = root / "datasets.yml"
    path = ds_path if ds_path.exists() else (map_path if map_path.exists() else None)
    if path is None:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

    payload: Dict[str, Any] = {}
    if path.name == "datasets.yml":
        if isinstance(data, Mapping):
            # Supported shapes:
            # - {datasets: {<ds>: {...}}}
            # - {mnps_9d: {datasets: {<ds>: {...}}}}
            # - {<ds>: {...}}
            if isinstance(data.get("mnps_9d"), Mapping):
                v2 = data.get("mnps_9d") or {}
                ds_map = v2.get("datasets", {}) if isinstance(v2, Mapping) else {}
                if isinstance(ds_map, Mapping):
                    payload = dict(ds_map.get(dataset_id, {}) or {})
            if not payload and isinstance(data.get("datasets"), Mapping):
                payload = dict((data.get("datasets") or {}).get(dataset_id, {}) or {})
            if not payload and isinstance(data.get(dataset_id), Mapping):
                payload = dict(data.get(dataset_id) or {})
    else:
        if isinstance(data, Mapping) and isinstance(data.get("mnps_9d"), Mapping):
            v2 = data.get("mnps_9d") or {}
            if isinstance(v2, Mapping):
                payload = dict(v2)
        else:
            payload = dict(data) if isinstance(data, Mapping) else {}

    for k in ("schema", "schema_version", "dataset_id", "description"):
        payload.pop(k, None)
    return payload


class DatasetSummaryRunner:
    """Encapsulate dataset-level summarization logic."""

    def __init__(self, ctx: SummarizeContext, ds_id: str, subject_filter: Optional[str], h5_mode: str, n_jobs: int = 1):
        self.ctx = ctx
        self.ds_id = ds_id
        self.subject_filter = self._normalize_subject(subject_filter) if subject_filter else None
        self.h5_mode = h5_mode
        self.n_jobs = max(1, int(n_jobs or 1))
        self.config = ctx.config
        self.received_dir = ctx.received_dir
        self.processed_dir = ctx.processed_dir
        self._dataset_csv_lock = Lock()
        # Global coverage defaults, with optional dataset-specific overrides
        self.min_seconds = self.ctx.coverage.min_seconds
        self.min_epochs = self.ctx.coverage.min_epochs
        self.coverage_optional_rules: List[Dict[str, Any]] = []
        self.coverage_rule_source: str = "default"
        try:
            robustness_cfg = self.config.get("robustness", {}) if isinstance(self.config, Mapping) else {}
            coverage_cfg = robustness_cfg.get("coverage", {}) if isinstance(robustness_cfg, Mapping) else {}
            ds_overrides = coverage_cfg.get("datasets", {}) if isinstance(coverage_cfg, Mapping) else {}
            if isinstance(ds_overrides, Mapping):
                ds_cfg = ds_overrides.get(self.ds_id, {})
                if isinstance(ds_cfg, Mapping):
                    if "min_seconds" in ds_cfg:
                        self.min_seconds = float(ds_cfg.get("min_seconds", self.min_seconds) or self.min_seconds)
                    if "min_epochs" in ds_cfg:
                        self.min_epochs = int(ds_cfg.get("min_epochs", self.min_epochs) or self.min_epochs)
                    optional_rules = ds_cfg.get("optional_rules", [])
                    if isinstance(optional_rules, list):
                        self.coverage_optional_rules.extend(
                            [r for r in optional_rules if isinstance(r, Mapping)]
                        )
        except Exception:
            logger.exception("Failed to apply dataset-specific coverage overrides; using global defaults")

        # Optional external per-dataset config:
        #   openneuro_ingest/config/config_<dataset>.yaml
        # Used for dataset-specific, auditable policy (e.g. ds005620 sed2 short segments).
        external_cfg = self._load_external_dataset_config()
        ext_cov = (
            external_cfg.get("robustness", {}).get("coverage", {})
            if isinstance(external_cfg, Mapping)
            else {}
        )
        if isinstance(ext_cov, Mapping):
            if "min_seconds" in ext_cov:
                self.min_seconds = float(ext_cov.get("min_seconds", self.min_seconds) or self.min_seconds)
            if "min_epochs" in ext_cov:
                self.min_epochs = int(ext_cov.get("min_epochs", self.min_epochs) or self.min_epochs)
            ext_rules = ext_cov.get("optional_rules", [])
            if isinstance(ext_rules, list):
                self.coverage_optional_rules.extend([r for r in ext_rules if isinstance(r, Mapping)])
            if external_cfg:
                self.coverage_rule_source = f"external:{self.ds_id}"

        self.participants_df: Optional[pd.DataFrame] = None
        self.index_df: Optional[pd.DataFrame] = None
        summarize_cfg = self.config.get("summarize", {}) if isinstance(self.config, Mapping) else {}
        if not isinstance(summarize_cfg, Mapping):
            summarize_cfg = {}
        # Safer default: do not silently merge multiple source files into one summarize key.
        self.allow_group_collisions = bool(summarize_cfg.get("allow_group_collisions", False))
        self.qc_policy = str(summarize_cfg.get("qc_policy", "eeg_only")).strip().lower() or "eeg_only"
        self.grouping_collision_info: Dict[str, Any] = {"count": 0, "merged_extra_files": 0, "examples": []}
        self._participant_meta_map: Dict[str, Dict[str, Any]] = {}
        self._file_entity_cache: Dict[str, tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]] = {}
        self._index_paths_by_basename: Dict[str, List[str]] = {}

    @staticmethod
    def _normalize_subject(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = str(value)
        return value if value.startswith("sub-") else f"sub-{value.zfill(3)}"

    def _parse_file_entities(
        self,
        file_name: str,
    ) -> tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Parse grouping entities from BIDS, with optional non-BIDS regex fallback."""
        subject, session, task, run, acq = parse_subject_session_task_run_acq(file_name)
        if subject != "sub-unknown":
            return subject, session, task, run, acq

        try:
            md_spec = self.config.get("metadata_extraction", {}) if isinstance(self.config, Mapping) else {}
            ds_spec = (md_spec.get("datasets", {}) or {}).get(self.ds_id, {}) if isinstance(md_spec, Mapping) else {}
            parse_cfg = ds_spec.get("filename_parse", {}) if isinstance(ds_spec, Mapping) else {}
            regex = str(parse_cfg.get("regex", "")).strip() if isinstance(parse_cfg, Mapping) else ""
            if not regex:
                return subject, session, task, run, acq
            m = re.search(regex, str(file_name))
            if not m:
                return subject, session, task, run, acq

            gd = m.groupdict()
            subj_raw = gd.get("subject")
            pad = int(parse_cfg.get("subject_pad", 3)) if isinstance(parse_cfg, Mapping) else 3
            if subj_raw:
                subj_s = str(subj_raw)
                subject = subj_s if subj_s.startswith("sub-") else f"sub-{subj_s.zfill(pad)}"

            ses_raw = gd.get("session")
            if ses_raw:
                ses_s = str(ses_raw)
                session = ses_s if ses_s.startswith("ses-") else f"ses-{ses_s}"

            task_raw = gd.get("task")
            if task_raw:
                task = str(task_raw)

            run_raw = gd.get("run")
            if run_raw:
                run_s = str(run_raw)
                run = run_s if run_s.startswith("run-") else f"run-{run_s}"

            acq_raw = gd.get("acq")
            if acq_raw:
                acq_s = str(acq_raw)
                acq = acq_s if acq_s.startswith("acq-") else f"acq-{acq_s}"
        except Exception:
            logger.exception("Failed non-BIDS filename parsing for dataset %s", self.ds_id)

        return subject, session, task, run, acq

    def _load_external_dataset_config(self) -> Dict[str, Any]:
        cfg_path = Path(__file__).resolve().parents[4] / "openneuro_ingest" / "config" / f"config_{self.ds_id}.yaml"
        if not cfg_path.exists():
            return {}
        try:
            import yaml  # type: ignore

            with cfg_path.open("r", encoding="utf-8") as f:
                parsed = yaml.safe_load(f) or {}
            if isinstance(parsed, Mapping):
                logger.info("Loaded external dataset config: %s", cfg_path)
                return dict(parsed)
            return {}
        except Exception as exc:
            logger.warning("Failed to load external dataset config %s: %s", cfg_path, exc)
            return {}

    @staticmethod
    def _match_value(rule_val: Any, actual_val: Optional[str]) -> bool:
        if rule_val is None:
            return True
        if isinstance(rule_val, (list, tuple, set)):
            targets = {str(v).lower() for v in rule_val}
            return str(actual_val).lower() in targets
        return str(actual_val).lower() == str(rule_val).lower()

    def resolve_coverage_policy(
        self,
        *,
        condition: Optional[str],
        task: Optional[str],
        run_id: Optional[str],
        acq_id: Optional[str],
    ) -> Dict[str, Any]:
        policy: Dict[str, Any] = {
            "min_seconds": float(self.min_seconds),
            "min_epochs": int(self.min_epochs),
            "tag": "default",
            "source": self.coverage_rule_source,
        }
        for i, rule in enumerate(self.coverage_optional_rules):
            match = rule.get("match", {}) if isinstance(rule, Mapping) else {}
            if not isinstance(match, Mapping):
                match = {}
            if not self._match_value(match.get("condition"), condition):
                continue
            if not self._match_value(match.get("task"), task):
                continue
            if not self._match_value(match.get("run"), run_id):
                continue
            if not self._match_value(match.get("acq"), acq_id):
                continue

            if "min_seconds" in rule:
                policy["min_seconds"] = float(rule.get("min_seconds", policy["min_seconds"]) or policy["min_seconds"])
            if "min_epochs" in rule:
                policy["min_epochs"] = int(rule.get("min_epochs", policy["min_epochs"]) or policy["min_epochs"])
            policy["tag"] = str(rule.get("tag", f"optional_rule_{i}"))
            break
        return policy

    def run(self) -> None:
        logger.info(f"Summarizing {self.ds_id}")
        ds_path = self.processed_dir / self.ds_id
        self.participants_df = load_participant_table(self.received_dir, self.ds_id)
        self._build_participant_meta_map()
        self.index_df = self._read_index(ds_path)
        self._build_index_basename_cache()

        features_df = self._read_features(ds_path)
        if features_df is None:
            return

        features_df = self._apply_subject_filter(features_df)
        if features_df is None:
            return

        features_df = self._apply_qc_filters(features_df)

        try:
            grouping_items = self._build_groupings(features_df)
        except Exception:
            logger.exception("Failed to build groupings for %s", self.ds_id)
            return
        if not grouping_items:
            logger.warning(f"No groups found for {self.subject_filter or 'any subject'} in {self.ds_id}")
            return

        mnps_dir = self._create_output_dir(ds_path)
        self._write_features_snapshot(mnps_dir, features_df)
        max_workers = min(max(1, self.n_jobs), len(grouping_items), multiprocessing.cpu_count())
        if max_workers > 1:
            logger.info(
                "Using %d summarize workers for %s (%d grouped recordings)",
                max_workers,
                self.ds_id,
                len(grouping_items),
            )
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [
                    ex.submit(self._process_grouping_item, ds_path, mnps_dir, grouping_key, sub_frame)
                    for grouping_key, sub_frame in grouping_items
                ]
                for fut in futures:
                    fut.result()
        else:
            for grouping_key, sub_frame in grouping_items:
                self._process_grouping_item(ds_path, mnps_dir, grouping_key, sub_frame)

        # Write a run-level manifest for quick inspection (humans + LLMs).
        try:
            write_run_manifest(
                mnps_dir=mnps_dir,
                config=self.ctx.config,
                ds_id=self.ds_id,
                received_dir=self.received_dir,
                processed_dir=self.processed_dir,
                h5_mode=self.h5_mode,
                extra={
                    "summarize_policy": {
                        "allow_group_collisions": bool(self.allow_group_collisions),
                        "qc_policy": self.qc_policy,
                    },
                    "grouping_collisions": self.grouping_collision_info,
                },
            )
        except Exception:
            logger.exception("Failed to write run_manifest.json for %s (%s)", self.ds_id, mnps_dir)

    def _normalize_grouping_key(
        self,
        grouping_key: tuple[Any, Any, Any, Any, Any],
    ) -> tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
        sub_id, ses_id, raw_task, run_id, acq_id = grouping_key
        if pd.isna(ses_id):
            ses_id = None
        if pd.isna(raw_task):
            raw_task = None
        if pd.isna(run_id):
            run_id = None
        if pd.isna(acq_id):
            acq_id = None
        sub_id = sub_id if str(sub_id).startswith("sub-") else f"sub-{str(sub_id).zfill(3)}"
        return str(sub_id), ses_id, raw_task, run_id, acq_id

    def _process_grouping_item(
        self,
        ds_path: Path,
        mnps_dir: Path,
        grouping_key: tuple[Any, Any, Any, Any, Any],
        sub_frame: pd.DataFrame,
    ) -> None:
        sub_id, ses_id, raw_task, run_id, acq_id = self._normalize_grouping_key(grouping_key)
        runner = SubjectSummaryRunner(
            dataset_runner=self,
            ds_path=ds_path,
            mnps_dir=mnps_dir,
            index_df=self.index_df,
        )
        runner.run(
            sub_id=sub_id,
            ses_id=ses_id,
            raw_task=raw_task,
            run_id=run_id,
            acq_id=acq_id,
            sub_frame=sub_frame,
        )

    def write_regional_csv_outputs_threadsafe(
        self,
        *,
        regional_mnps_results: Any,
        regional_mnps_cfg: Mapping[str, Any],
        mnps_dir: Path,
        config: Mapping[str, Any],
        dataset_label: str,
    ) -> None:
        with self._dataset_csv_lock:
            write_regional_csv_outputs(
                regional_mnps_results=regional_mnps_results,
                regional_mnps_cfg=regional_mnps_cfg,
                mnps_dir=mnps_dir,
                config=config,
                dataset_label=dataset_label,
            )

    def write_stratified_blocks_csv_output_threadsafe(
        self,
        *,
        stratified_blocks_result: Any,
        config: Mapping[str, Any],
        dataset_id: str,
        mnps_dir: Path,
        dataset_label: str,
    ) -> None:
        with self._dataset_csv_lock:
            write_stratified_blocks_csv_output(
                stratified_blocks_result=stratified_blocks_result,
                config=config,
                dataset_id=dataset_id,
                mnps_dir=mnps_dir,
                dataset_label=dataset_label,
            )

    def _build_features_snapshot(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Build a compact per-run features snapshot for provenance/Test-C."""
        snapshot: Dict[str, Any] = {
            "dataset_id": self.ds_id,
            "rows": int(len(features_df)),
            "columns": sorted([str(c) for c in features_df.columns]),
            "column_stats": {},
        }
        for col in snapshot["columns"]:
            s = features_df[col]
            col_info: Dict[str, Any] = {
                "missing_rate": float(s.isna().mean()) if len(s) else 0.0,
                "dtype": str(s.dtype),
            }
            num = pd.to_numeric(s, errors="coerce")
            finite = num[np.isfinite(num.to_numpy(dtype=np.float64, copy=False))]
            if len(finite):
                med = float(np.nanmedian(finite))
                col_info.update(
                    {
                        "mean": float(np.nanmean(finite)),
                        "std": float(np.nanstd(finite)),
                        "median": med,
                        "mad": float(np.nanmedian(np.abs(finite - med))),
                    }
                )
            snapshot["column_stats"][col] = col_info
        snapshot["features_snapshot_hash"] = _stable_hash_mapping(snapshot)
        return snapshot

    def _write_features_snapshot(self, mnps_dir: Path, features_df: pd.DataFrame) -> None:
        """Write features_snapshot.json under the current run directory."""
        try:
            payload = self._build_features_snapshot(features_df)
            out_path = mnps_dir / "features_snapshot.json"
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("Wrote features snapshot: %s", out_path)
        except Exception:
            logger.exception("Failed to write features snapshot for %s", self.ds_id)

    def _build_participant_meta_map(self) -> None:
        """Build O(1) participant metadata lookup by participant_id."""
        self._participant_meta_map = {}
        if self.participants_df is None or self.participants_df.empty:
            return
        if "participant_id" not in self.participants_df.columns:
            return
        try:
            for _, row in self.participants_df.iterrows():
                pid = str(row.get("participant_id", "")).strip()
                if not pid or pid in self._participant_meta_map:
                    continue
                self._participant_meta_map[pid] = row.to_dict()
        except Exception:
            self._participant_meta_map = {}

    def _build_index_basename_cache(self) -> None:
        """Build O(1) basename -> relative-path list cache from file_index."""
        self._index_paths_by_basename = {}
        if self.index_df is None or self.index_df.empty or "path" not in self.index_df.columns:
            return
        try:
            for p in self.index_df["path"].astype(str).tolist():
                b = Path(p).name
                if not b:
                    continue
                self._index_paths_by_basename.setdefault(b, []).append(p)
        except Exception:
            self._index_paths_by_basename = {}

    def _lookup_rel_paths_by_file_value(self, file_value: str) -> List[str]:
        """Return candidate relative paths for a `features.csv` file value."""
        if not file_value:
            return []
        if not self._index_paths_by_basename:
            return []
        name = Path(str(file_value).replace("\\", "/")).name
        return list(self._index_paths_by_basename.get(name, []))

    def _read_index(self, ds_path: Path) -> Optional[pd.DataFrame]:
        index_path = ds_path / "file_index.csv"
        if not index_path.exists():
            return self._build_index_from_received(ds_path)
        if index_path.stat().st_size == 0:
            logger.warning("Empty file_index.csv for %s; rebuilding", self.ds_id)
            try:
                index_path.unlink()
            except Exception as e:
                logger.warning("Failed to remove empty file_index.csv for %s: %s", self.ds_id, e)
            return self._build_index_from_received(ds_path)
        try:
            return pd.read_csv(index_path)
        except pd.errors.EmptyDataError:
            logger.warning("Empty file_index.csv for %s; rebuilding", self.ds_id)
            try:
                index_path.unlink()
            except Exception as e:
                logger.warning("Failed to remove empty file_index.csv for %s: %s", self.ds_id, e)
            return self._build_index_from_received(ds_path)
        except Exception:
            logger.exception("Failed to read file_index.csv for %s", self.ds_id)
            return None

    def _build_index_from_received(self, ds_path: Path) -> Optional[pd.DataFrame]:
        ds_root = bids_index.resolve_dataset_root(self.config, self.ctx.received_dir, self.ds_id)
        if not ds_root.exists():
            logger.warning("Dataset root missing at %s; cannot build file_index.csv", ds_root)
            return None
        try:
            logger.info("Building file index for %s from %s", self.ds_id, ds_root)
            index_df = bids_index.build_file_index(ds_root, config=self.config, dataset_id=self.ds_id)
            index_df.to_csv(ds_path / "file_index.csv", index=False)
            logger.info("Saved file index: %s", ds_path / "file_index.csv")
            return index_df
        except Exception:
            logger.exception("Failed to build file_index.csv for %s from %s", self.ds_id, ds_root)
            return None

    def _dataset_root(self) -> Path:
        """Resolve effective dataset root, honoring per-dataset overrides."""
        return bids_index.resolve_dataset_root(self.config, self.ctx.received_dir, self.ds_id)

    def _read_features(self, ds_path: Path) -> Optional[pd.DataFrame]:
        storage_cfg = self.config.get("feature_storage", {}) if isinstance(self.config, Mapping) else {}
        read_prefer = str(storage_cfg.get("read_prefer", "parquet")).strip().lower() if isinstance(storage_cfg, Mapping) else "parquet"
        if read_prefer not in {"csv", "parquet"}:
            read_prefer = "parquet"

        candidates = [ds_path / "features.parquet", ds_path / "features.csv"]
        if read_prefer == "csv":
            candidates = [ds_path / "features.csv", ds_path / "features.parquet"]
        features_path = next((p for p in candidates if p.exists()), None)
        if features_path is None:
            logger.warning(f"No features found for {self.ds_id}, skipping")
            return None
        try:
            if features_path.suffix.lower() == ".parquet":
                features_df = pd.read_parquet(features_path)
            else:
                features_df = pd.read_csv(features_path)
        except Exception:
            logger.exception("Failed to read features table %s for %s", features_path, self.ds_id)
            return None
        if features_df.empty:
            logger.warning(f"Features dataframe empty for {self.ds_id}")
            return None
        return features_df

    def _apply_subject_filter(self, features_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if not self.subject_filter:
            return features_df
        if "file" not in features_df.columns:
            logger.warning("Subject filter requested but 'file' column missing; skipping subject filter")
            return features_df
        filtered = features_df[features_df["file"].astype(str).str.contains(self.subject_filter)]
        if filtered.empty:
            logger.warning(f"No epochs for subject {self.subject_filter} in {self.ds_id}")
            return None
        return filtered

    def _apply_qc_filters(self, features_df: pd.DataFrame) -> pd.DataFrame:
        qc_cols = [c for c in features_df.columns if str(c).startswith("qc_ok_")]
        if not qc_cols:
            return features_df

        policy = self.qc_policy
        before = len(features_df)
        if policy == "eeg_only":
            if "qc_ok_eeg" not in features_df.columns:
                logger.info("QC policy 'eeg_only' requested but qc_ok_eeg missing; skipping QC filtering")
                return features_df
            features_df = features_df[features_df["qc_ok_eeg"].fillna(1).astype(int) == 1]
            dropped = before - len(features_df)
            if dropped > 0:
                logger.info("Dropped %d/%d epochs by qc_ok_eeg (policy=eeg_only)", dropped, before)
            return features_df

        if policy == "all_qc_ok":
            mask = np.ones(len(features_df), dtype=bool)
            dropped_by: Dict[str, int] = {}
            for c in qc_cols:
                col_ok = features_df[c].fillna(1).astype(int) == 1
                dropped_by[str(c)] = int((~col_ok).sum())
                mask &= col_ok
            features_df = features_df[mask]
            dropped = before - len(features_df)
            if dropped > 0:
                logger.info(
                    "Dropped %d/%d epochs by all_qc_ok policy; dropped_by=%s",
                    dropped,
                    before,
                    dropped_by,
                )
            return features_df

        if policy == "any_ok":
            mask = np.zeros(len(features_df), dtype=bool)
            for c in qc_cols:
                mask |= features_df[c].fillna(1).astype(int) == 1
            features_df = features_df[mask]
            dropped = before - len(features_df)
            if dropped > 0:
                logger.info("Dropped %d/%d epochs by any_ok policy", dropped, before)
            return features_df

        logger.warning(
            "Unknown summarize.qc_policy='%s'; falling back to eeg_only",
            policy,
        )
        if "qc_ok_eeg" in features_df.columns:
            before = len(features_df)
            features_df = features_df[features_df["qc_ok_eeg"].fillna(1) == 1]
            dropped = before - len(features_df)
            if dropped > 0:
                logger.info("Dropped %d/%d epochs by qc_ok_eeg", dropped, before)
        return features_df

    def _build_groupings(self, features_df: pd.DataFrame):
        """Group features by (subject, session, task, run, acq) for separate H5 output per combination."""
        if "file" in features_df.columns:
            # Parse subject/session/task/run/acq once per unique file value.
            file_series = features_df["file"].astype(str)
            unique_files = pd.unique(file_series)
            for f in unique_files:
                if f not in self._file_entity_cache:
                    self._file_entity_cache[f] = self._parse_file_entities(f)
            parsed_rows = [self._file_entity_cache[f] for f in file_series.tolist()]
            if parsed_rows:
                subj_vals, ses_vals, task_vals, run_vals, acq_vals = zip(*parsed_rows)
            else:
                subj_vals, ses_vals, task_vals, run_vals, acq_vals = ([], [], [], [], [])
            features_df = features_df.assign(
                _subject=list(subj_vals),
                _session=list(ses_vals),
                _task=list(task_vals),
                _run=list(run_vals),
                _acq=list(acq_vals),
            )
            unknown_mask = features_df["_subject"] == "sub-unknown"
            if unknown_mask.any():
                if "subject" in features_df.columns:
                    features_df.loc[unknown_mask, "_subject"] = (
                        features_df.loc[unknown_mask, "subject"].astype(str).apply(lambda s: f"sub-{s.zfill(3)}")
                    )
                else:
                    logger.warning("Some feature rows lack subject identifiers; they will be skipped")
                    features_df = features_df.loc[~unknown_mask]

            # Guardrail for non-BIDS sources: if multiple distinct files resolve to
            # the same summarize key, they are merged into one output run.
            group_cols = ["_subject", "_session", "_task", "_run", "_acq"]
            key_file = features_df[group_cols + ["file"]].drop_duplicates()
            files_per_key = key_file.groupby(group_cols, dropna=False)["file"].nunique()
            collisions = files_per_key[files_per_key > 1]
            if not collisions.empty:
                merged_extra_files = int((collisions - 1).sum())
                examples: list[str] = []
                for key, n_files in collisions.head(5).items():
                    if isinstance(key, tuple) and len(key) == 5:
                        sub_id, ses_id, task_id, run_id, acq_id = key
                    else:
                        sub_id, ses_id, task_id, run_id, acq_id = key, None, None, None, None
                    examples.append(
                        f"(sub={sub_id}, ses={ses_id}, task={task_id}, run={run_id}, acq={acq_id}) -> {int(n_files)} files"
                    )
                self.grouping_collision_info = {
                    "count": int(collisions.shape[0]),
                    "merged_extra_files": int(merged_extra_files),
                    "examples": examples,
                }
                if not self.allow_group_collisions:
                    raise RuntimeError(
                        f"Detected {int(collisions.shape[0])} grouping-key collisions in {self.ds_id}. "
                        f"Set summarize.allow_group_collisions=true to merge explicitly. "
                        f"Examples: {'; '.join(examples)}"
                    )
                logger.warning(
                    "Detected %d grouping-key collisions in %s; merging is enabled. %d extra files will be merged. Examples: %s",
                    int(collisions.shape[0]),
                    self.ds_id,
                    merged_extra_files,
                    "; ".join(examples),
                )
            # Group by subject + session + task + run + acq for separate H5 per recording stream
            grouping = features_df.groupby(["_subject", "_session", "_task", "_run", "_acq"], dropna=False)
        elif "subject" in features_df.columns:
            features_df = features_df.assign(_session=None, _task=None, _run=None, _acq=None)
            grouping = features_df.groupby(["subject", "_session", "_task", "_run", "_acq"], dropna=False)
        else:
            # No subject info: treat entire dataset as single anonymous subject
            features_df = features_df.assign(_subject="sub-unknown", _session="ses-unknown", _task=None, _run=None, _acq=None)
            grouping = features_df.groupby(["_subject", "_session", "_task", "_run", "_acq"], dropna=False)

        if self.subject_filter:
            grouping_items = [item for item in grouping if item[0][0] == self.subject_filter]
        else:
            grouping_items = list(grouping)
        return grouping_items

    def _create_output_dir(self, ds_path: Path) -> Path:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        mnps_dir = ds_path / f"neuralmanifolddynamics_{self.ds_id}_{ts}"
        mnps_dir.mkdir(parents=True, exist_ok=True)
        return mnps_dir

    def participant_meta_for(self, sub_id: str) -> Dict[str, Any]:
        if self._participant_meta_map:
            return dict(self._participant_meta_map.get(sub_id, {}))
        if self.participants_df is None:
            return {}
        lookup = self.participants_df[self.participants_df["participant_id"].astype(str) == sub_id]
        if lookup.empty:
            return {}
        return lookup.iloc[0].to_dict()


class SubjectSummaryRunner:
    """Subject/session-level summarization."""

    def __init__(
        self,
        dataset_runner: DatasetSummaryRunner,
        ds_path: Path,
        mnps_dir: Path,
        index_df: Optional[pd.DataFrame],
    ):
        self.dataset = dataset_runner
        self.ctx = dataset_runner.ctx
        self.ds_path = ds_path
        self.mnps_dir = mnps_dir
        self.index_df = index_df
        self._confounds_path_cache: Dict[str, Optional[Path]] = {}

    def _dataset_root(self) -> Path:
        """Resolve effective dataset root, honoring per-dataset overrides."""
        return bids_index.resolve_dataset_root(self.ctx.config, self.ctx.received_dir, self.dataset.ds_id)

    def run(
        self,
        sub_id: str,
        ses_id: Optional[str],
        raw_task: Optional[str],
        run_id: Optional[str],
        acq_id: Optional[str],
        sub_frame: pd.DataFrame,
    ) -> None:
        config = self.ctx.config
        mnps_cfg = self.ctx.mnps_cfg
        normalize_mode = self.ctx.normalize_override
        proj_cfg = config.get("mnps_projection", {}) if isinstance(config, Mapping) else {}
        missing_axis_policy = str(proj_cfg.get("missing_axis_policy", "nan_mask_v1")).strip().lower() or "nan_mask_v1"

        sub_id = sub_id if str(sub_id).startswith("sub-") else f"sub-{str(sub_id).zfill(3)}"
        participant_meta = self.dataset.participant_meta_for(sub_id)

        # Extract representative filename for task parsing (if available)
        representative_file = None
        if "file" in sub_frame.columns and len(sub_frame) > 0:
            representative_file = str(sub_frame["file"].iloc[0])

        mapped_meta = extract_mapped_metadata(
            participant_meta, config, self.dataset.ds_id, ses_id, filename=representative_file
        )

        # Resolve condition/task:
        # - Prefer explicit mapped metadata (participants.tsv + config rules)
        # - Fall back to the task parsed from filenames during grouping, to avoid
        #   overwriting multiple (subject, task) outputs into the same directory.
        condition = mapped_meta.get("condition")
        task = mapped_meta.get("task") or raw_task

        modality = str(config.get("modality", "")).strip().lower() if isinstance(config, Mapping) else ""
        if modality == "fmri":
            sub_frame = self._merge_fd_from_confounds(
                sub_frame=sub_frame,
                raw_task=raw_task,
                condition=condition,
                session=ses_id,
                run_id=run_id,
                acq_id=acq_id,
            )

        # Optional FD-based censoring (drop high-motion epochs and neighbours)
        n_before_any = int(len(sub_frame))
        regional_mnps_cfg = config.get("regional_mnps", {}) if isinstance(config, Mapping) else {}
        fd_required = (
            bool(regional_mnps_cfg.get("require_framewise_displacement", True))
            if modality == "fmri"
            else False
        )
        sub_frame = self._apply_fd_censoring(
            sub_frame,
            require_fd=fd_required,
            context_label=f"{self.dataset.ds_id}:{sub_id}:{ses_id or '-'}:{raw_task or '-'}:{run_id or '-'}",
        )
        n_after_qc = int(len(sub_frame))

        # Build target directory: use condition/task for datasets without sessions (like ds003171)
        dir_suffix = self._build_dir_suffix(ses_id, condition, task, run_id, acq_id)
        target_dir = self.mnps_dir / (f"{sub_id}_{dir_suffix}" if dir_suffix else sub_id)
        target_dir.mkdir(parents=True, exist_ok=True)

        # Build descriptive dataset_label with condition/task
        dataset_label = build_dataset_label(
            ds_id=self.dataset.ds_id,
            sub_id=sub_id,
            ses_id=ses_id,
            condition=condition,
            task=task,
            run=run_id,
            acq=acq_id,
        )
        dt = mnps_cfg["window_sec"] * (1.0 - mnps_cfg["overlap"])
        coverage_seconds_measured, coverage_method = self._estimate_coverage_seconds(sub_frame, dt)
        coverage_seconds_assumed = float(len(sub_frame) * dt)
        coverage_seconds_effective = (
            coverage_seconds_measured
            if np.isfinite(coverage_seconds_measured) and coverage_seconds_measured > 0
            else coverage_seconds_assumed
        )
        coverage_policy = self.dataset.resolve_coverage_policy(
            condition=condition,
            task=task,
            run_id=run_id,
            acq_id=acq_id,
        )
        min_epochs_eff = int(coverage_policy.get("min_epochs", self.dataset.min_epochs))
        min_seconds_eff = float(coverage_policy.get("min_seconds", self.dataset.min_seconds))
        coverage_tag = str(coverage_policy.get("tag", "default"))
        if len(sub_frame) < min_epochs_eff or coverage_seconds_effective < min_seconds_eff:
            logger.warning(
                "Skipping %s (coverage too low; tag=%s): epochs=%d, seconds_effective=%.1f (%s), seconds_assumed=%.1f, required_epochs=%d, required_seconds=%.1f",
                dataset_label,
                coverage_tag,
                len(sub_frame),
                coverage_seconds_effective,
                coverage_method,
                coverage_seconds_assumed,
                min_epochs_eff,
                min_seconds_eff,
            )
            return

        # Track provisional modularity fraction if available
        modularity_provisional_frac = None
        if "fmri_modularity_provisional" in sub_frame.columns and len(sub_frame):
            modularity_provisional_frac = float(sub_frame["fmri_modularity_provisional"].fillna(0).mean())

        # Stratified MNPS v2 config
        v2_cfg = config.get("mnps_9d", {}) if isinstance(config, Mapping) else {}
        v2_enabled = bool(v2_cfg.get("enabled", False))
        v2_definition_version = str(v2_cfg.get("definition_version", v2_cfg.get("mnps_9d_definition_version", "2.2")))
        v2_versions = v2_cfg.get("versions", {}) if isinstance(v2_cfg, Mapping) else {}
        selected_v2_cfg: Mapping[str, Any] = v2_cfg if isinstance(v2_cfg, Mapping) else {}
        if isinstance(v2_versions, Mapping):
            candidate = v2_versions.get(v2_definition_version)
            if isinstance(candidate, Mapping):
                selected_v2_cfg = _deep_merge_dict(v2_cfg, candidate)
        subcoords_spec = selected_v2_cfg.get("subcoords", {}) if isinstance(selected_v2_cfg, Mapping) else {}
        ds_overrides: Dict[str, Any] = {}
        if isinstance(v2_cfg, Mapping):
            inline = (v2_cfg.get("datasets", {}) or {}).get(self.dataset.ds_id, {})
            if isinstance(inline, Mapping):
                ds_overrides = dict(inline)
            policy_dir = v2_cfg.get("policy_dir")
            if isinstance(policy_dir, (str, Path)):
                policy_cfg = _load_mnps_9d_policy(str(policy_dir), self.dataset.ds_id)
                if policy_cfg:
                    ds_overrides = _deep_merge_dict(ds_overrides, policy_cfg)
        if isinstance(ds_overrides, Mapping):
            if "enabled" in ds_overrides:
                v2_enabled = bool(ds_overrides.get("enabled", v2_enabled))
            if "subcoords" in ds_overrides and isinstance(ds_overrides["subcoords"], Mapping):
                merged = dict(subcoords_spec)
                merged.update(ds_overrides["subcoords"])
                subcoords_spec = merged
            if "metric_policies" in ds_overrides and isinstance(ds_overrides["metric_policies"], Mapping):
                selected_v2_cfg = _deep_merge_dict(
                    selected_v2_cfg if isinstance(selected_v2_cfg, Mapping) else {},
                    {"metric_policies": ds_overrides["metric_policies"]},
                )
        _validate_e_e_subcoord_construct(subcoords_spec if isinstance(subcoords_spec, Mapping) else {})
        entropy_meta = _resolve_entropy_provenance(sub_frame)
        features_cfg = config.get("features", {}) if isinstance(config, Mapping) else {}
        pe_cfg = features_cfg.get("permutation_entropy", {}) if isinstance(features_cfg, Mapping) else {}
        pe_params = {
            "order": int(pe_cfg.get("order", 5)) if isinstance(pe_cfg, Mapping) else 5,
            "delay": int(pe_cfg.get("delay", 1)) if isinstance(pe_cfg, Mapping) else 1,
            "normalize": bool(pe_cfg.get("normalize", True)) if isinstance(pe_cfg, Mapping) else True,
        }
        v2_metric_policies = (
            selected_v2_cfg.get("metric_policies", {})
            if isinstance(selected_v2_cfg, Mapping)
            else {}
        )
        e_e_policy = v2_metric_policies.get("e_e", {}) if isinstance(v2_metric_policies, Mapping) else {}
        e_e_intended_metric = (
            str(e_e_policy.get("preferred", "permutation_entropy"))
            if isinstance(e_e_policy, Mapping)
            else "permutation_entropy"
        )
        mnps_9d_constructs = {
            "e_e": {
                "intended_metric": e_e_intended_metric,
                "actual_metric_used": str(entropy_meta.get("metric", "permutation_entropy")),
                "metric_backend": str(entropy_meta.get("backend", "numpy")),
                "params": pe_params,
                "degraded_mode": bool(entropy_meta.get("degraded_mode", False)),
                "reason": entropy_meta.get("reason"),
            }
        }
        m3d_cfg = _resolve_mnps_3d_cfg(config if isinstance(config, Mapping) else {})
        mde_mode_requested = str(m3d_cfg.get("mode", "direct_features"))
        mde_mode_effective = mde_mode_requested
        mde_from_v2_reason = None
        v1_mapping_hash = None
        v1_mapping_normalized = None
        v1_mapping_matrix = None
        v1_mapping_matrix_rows = None
        mde_from_v2_aggregation = None
        mde_from_v2_aggregation_requested = None
        mde_from_v2_pooling_legacy = None

        direct_weighted_features = sorted(
            {
                str(feat)
                for axis_weights in (self.ctx.weights or {}).values()
                if isinstance(axis_weights, Mapping)
                for feat in axis_weights.keys()
            }
        )
        v2_weighted_features = sorted(
            {
                str(feat)
                for subcoord_weights in (subcoords_spec or {}).values()
                if isinstance(subcoord_weights, Mapping)
                for feat in subcoord_weights.keys()
            }
        )
        missing_rate_direct = _missing_weighted_feature_rate(sub_frame, direct_weighted_features)
        missing_rate_v2 = _missing_weighted_feature_rate(sub_frame, v2_weighted_features)

        proj_cfg = config.get("mnps_projection", {}) if isinstance(config, Mapping) else {}
        clip_threshold = float(proj_cfg.get("clip_threshold", 6.0)) if isinstance(proj_cfg, Mapping) else 6.0
        feature_standardization = proj_cfg.get("feature_standardization", {}) if isinstance(proj_cfg, Mapping) else {}

        # Project direct MNPS coordinates first (always available as fallback/provenance).
        x_direct, x_direct_coverage, feature_baselines_v1 = projection.project_features_with_coverage(
            sub_frame, 
            self.ctx.weights, 
            normalize=normalize_mode,
            feature_standardization=feature_standardization,
            clip_threshold=clip_threshold
        )
        x = x_direct
        x_coverage = x_direct_coverage
        x_definition = "direct_mde_v1"
        direct_axis_renorm = "abs_weight"
        coords_9d = None
        coords_9d_names: list[str] = []
        feature_baselines_v2 = {}
        v2_missing_policy = "renorm"
        v2_all_non_finite_names: list[str] = []
        v2_all_non_finite_count = 0
        if v2_enabled and subcoords_spec:
            v2_missing_policy = (
                str(selected_v2_cfg.get("missing_policy", "renorm")).strip().lower()
                if isinstance(selected_v2_cfg, Mapping)
                else "renorm"
            )
            coords_9d, coords_9d_names, feature_baselines_v2 = projection.project_features_v2(
                sub_frame, 
                subcoords_spec, 
                normalize=normalize_mode, 
                missing_policy=v2_missing_policy,
                feature_standardization=feature_standardization,
                clip_threshold=clip_threshold
            )
            if coords_9d.size and coords_9d_names:
                try:
                    (
                        coords_9d,
                        coords_9d_names,
                        coords_9d_diag,
                    ) = schema._normalize_coords_9d(
                        coords_9d,
                        coords_9d_names,
                        allow_all_non_finite_columns=True,
                        return_diagnostics=True,
                    )
                    v2_all_non_finite_names = list(coords_9d_diag.get("all_non_finite_names", []) or [])
                    v2_all_non_finite_count = int(coords_9d_diag.get("all_non_finite_count", 0) or 0)
                    if v2_all_non_finite_count > 0:
                        logger.warning(
                            "Stratified MNPS coords_9d has %d all-non-finite subcoordinate(s) for %s: %s. "
                            "Proceeding in degraded mode and flagging provenance.",
                            v2_all_non_finite_count,
                            dataset_label,
                            ", ".join(v2_all_non_finite_names),
                        )
                except Exception as e:
                    logger.error("Failed to normalize Stratified MNPS coords_9d for %s. Explicit failure enforced to prevent silent degradation.", dataset_label)
                    raise RuntimeError(f"Stratified MNPS normalization failed: {e}") from e

        # Merge baselines (v2 preferred if active, else v1)
        merged_baselines = dict(feature_baselines_v1)
        merged_baselines.update(feature_baselines_v2)

        if mde_mode_requested == "from_v2":
            if coords_9d is not None and coords_9d.size and coords_9d_names:
                mde_from_v2_aggregation_requested = m3d_cfg.get("aggregation_requested")
                mde_from_v2_pooling_legacy = m3d_cfg.get("legacy_pooling")
                aggregation_effective = str(m3d_cfg.get("aggregation", "fixed_weighted_projection"))
                if aggregation_effective == "fixed_weighted_projection":
                    cfg_weights = _coerce_v1_mapping_to_v2_subcoords(
                        m3d_cfg.get("v1_mapping", {}),
                        subcoords_spec if isinstance(subcoords_spec, Mapping) else {},
                    )
                    has_any_weight = any(
                        isinstance(cfg_weights, Mapping)
                        and isinstance(cfg_weights.get(axis, {}), Mapping)
                        and len(cfg_weights.get(axis, {})) > 0
                        for axis in ("m", "d", "e")
                    )
                    if not has_any_weight:
                        raise ValueError(
                            "mnps_3d.mode=from_v2 with aggregation=fixed_weighted_projection requires a non-empty V1 mapping "
                            "(direct subcoord names or feature names resolvable via mnps_9d.subcoords)."
                        )
                    axis_map_for_v1 = cfg_weights
                else:
                    axis_map_for_v1 = m3d_cfg.get("map", {})
                x, x_coverage, v1_map_info = projection.derive_mde_from_v2(
                    coords_9d,
                    coords_9d_names,
                    axis_map_for_v1,
                    pooling=str(m3d_cfg.get("legacy_pooling", "mean")),
                    normalize_columns_l2=True,
                    enforce_block_selective=False,
                    return_mapping_info=True,
                )
                x_definition = f"derived_mde_from_v2_{str((v1_map_info or {}).get('aggregation', aggregation_effective))}"
                mde_from_v2_aggregation = (v1_map_info or {}).get("aggregation")
                direct_axis_renorm = str(mde_from_v2_aggregation or aggregation_effective)
                v1_mapping_normalized = (v1_map_info or {}).get("weights_normalized")
                v1_mapping_matrix = (v1_map_info or {}).get("matrix")
                v1_mapping_matrix_rows = (v1_map_info or {}).get("coords_9d_names")
                if isinstance(v1_mapping_normalized, Mapping):
                    v1_mapping_hash = _stable_hash_mapping(v1_mapping_normalized)
            else:
                mde_mode_effective = "direct_features"
                mde_from_v2_reason = "coords_9d_unavailable"
                logger.warning(
                    "mnps_3d.mode=from_v2 requested for %s but coords_9d unavailable; falling back to direct_features",
                    dataset_label,
                )

        axis_cov_labels = ["m", "d", "e"]
        axis_cov_stats = {
            f"{lbl}_mean": (
                float(np.nanmean(x_coverage[:, i])) if x_coverage.size and not np.all(np.isnan(x_coverage[:, i])) else float("nan")
            )
            for i, lbl in enumerate(axis_cov_labels)
        }

        dropped_missing_axis_epochs = 0
        min_axis_coverage = float(proj_cfg.get("min_axis_coverage", 0.3)) if isinstance(proj_cfg, Mapping) else 0.3
        if missing_axis_policy == "nan_mask_v1":
            valid_x = np.all(np.isfinite(x), axis=1)
            cov_ok = np.all(x_coverage >= min_axis_coverage, axis=1)
            mask = valid_x & cov_ok
            dropped_missing_axis_epochs = int((~mask).sum())
            if dropped_missing_axis_epochs > 0:
                logger.warning(
                    "Dropping %d epochs with missing direct-axis support/coverage for %s (policy=%s, min_axis_coverage=%.3f)",
                    dropped_missing_axis_epochs,
                    dataset_label,
                    missing_axis_policy,
                    min_axis_coverage,
                )
                x = x[mask]
                x_coverage = x_coverage[mask]
                sub_frame = sub_frame.loc[mask].reset_index(drop=True)
                if coords_9d is not None and len(coords_9d) == len(mask):
                    coords_9d = coords_9d[mask]
            coverage_seconds_measured_post, coverage_method_post = self._estimate_coverage_seconds(sub_frame, dt)
            coverage_seconds_assumed_post = float(len(sub_frame) * dt)
            coverage_seconds_effective_post = (
                coverage_seconds_measured_post
                if np.isfinite(coverage_seconds_measured_post) and coverage_seconds_measured_post > 0
                else coverage_seconds_assumed_post
            )
            if len(sub_frame) < min_epochs_eff or coverage_seconds_effective_post < min_seconds_eff:
                logger.warning(
                    "Skipping %s after nan/cov masking (tag=%s): epochs=%d, seconds_effective=%.1f (%s), required_epochs=%d, required_seconds=%.1f",
                    dataset_label,
                    coverage_tag,
                    len(sub_frame),
                    coverage_seconds_effective_post,
                    coverage_method_post,
                    min_epochs_eff,
                    min_seconds_eff,
                )
                return
            if len(sub_frame) == 0:
                logger.warning("Skipping %s: all epochs dropped by missing-axis policy", dataset_label)
                return

        # Time index and derivatives
        time = projection.build_time_index(len(sub_frame), mnps_cfg["window_sec"], mnps_cfg["overlap"])
        window_start, window_end = self._extract_time_bounds(sub_frame, time, mnps_cfg["window_sec"])
        
        # Explicitly prevent derivative estimation across file boundaries (time aliasing protection)
        def _compute_dot(features_array: np.ndarray) -> np.ndarray:
            dot_cfg = ((config.get("mnps", {}) or {}).get("derivative_robust", {}) or {}) if isinstance(config, Mapping) else {}
            use_segmented = bool(dot_cfg.get("enabled", True))
            dot_array = np.zeros_like(features_array)
            if "file" in sub_frame.columns and sub_frame["file"].nunique() > 1:
                logger.info("Computing derivatives per-file to avoid boundary crossing (%d files)", sub_frame["file"].nunique())
                file_series = sub_frame["file"].to_numpy()
                for f_val in np.unique(file_series):
                    mask = (file_series == f_val)
                    sub_array = features_array[mask]
                    if len(sub_array) > 0:
                        if use_segmented:
                            dot_array[mask] = projection.estimate_derivatives_segmented(
                                sub_array,
                                dt,
                                method=self.ctx.derivative_cfg["method"],
                                max_jump=float(dot_cfg.get("max_jump", 5.0)),
                                min_seg=int(dot_cfg.get("min_seg", 9)),
                                savgol_window=int(self.ctx.derivative_cfg["window"]),
                                polyorder=int(self.ctx.derivative_cfg["polyorder"]),
                            )
                        else:
                            dot_array[mask] = projection.estimate_derivatives(
                                sub_array,
                                dt,
                                method=self.ctx.derivative_cfg["method"],
                                window=self.ctx.derivative_cfg["window"],
                                polyorder=self.ctx.derivative_cfg["polyorder"],
                            )
            else:
                if use_segmented:
                    dot_array = projection.estimate_derivatives_segmented(
                        features_array,
                        dt,
                        method=self.ctx.derivative_cfg["method"],
                        max_jump=float(dot_cfg.get("max_jump", 5.0)),
                        min_seg=int(dot_cfg.get("min_seg", 9)),
                        savgol_window=int(self.ctx.derivative_cfg["window"]),
                        polyorder=int(self.ctx.derivative_cfg["polyorder"]),
                    )
                else:
                    dot_array = projection.estimate_derivatives(
                        features_array,
                        dt,
                        method=self.ctx.derivative_cfg["method"],
                        window=self.ctx.derivative_cfg["window"],
                        polyorder=self.ctx.derivative_cfg["polyorder"],
                    )
            return dot_array

        x_dot = _compute_dot(x)

        # KNN and Jacobian
        whiten_flag = bool(mnps_cfg.get("whiten", True))
        nn_indices = projection.build_knn_indices(
            x,
            k=mnps_cfg["knn_k"],
            metric=mnps_cfg["knn_metric"],
            whiten=whiten_flag,
        )
        jac_res = jacobian.estimate_local_jacobians(
            x,
            x_dot,
            nn_indices,
            super_window=mnps_cfg["super_window"],
            ridge_alpha=mnps_cfg["ridge_alpha"],
            distance_weighted=bool(config.get("mnps", {}).get("ridge", {}).get("distance_weighted", True)),
            j_dot_dt=float(dt),
        )

        # Optional Jacobian for v2 coordinates
        jac_res_v2 = None
        if v2_enabled and coords_9d is not None and coords_9d_names:
            v2_jac_cfg = v2_cfg.get("jacobian", {}) if isinstance(v2_cfg, Mapping) else {}
            v2_jac_enabled = bool(v2_jac_cfg.get("enabled", True))
            if v2_jac_enabled and coords_9d.size:
                if np.isfinite(coords_9d).all():
                    coords_9d_dot = _compute_dot(coords_9d)
                    nn_indices_v2 = projection.build_knn_indices(
                        coords_9d,
                        k=mnps_cfg["knn_k"],
                        metric=mnps_cfg["knn_metric"],
                        whiten=whiten_flag,
                    )
                    jac_res_v2 = jacobian.estimate_local_jacobians(
                        coords_9d,
                        coords_9d_dot,
                        nn_indices_v2,
                        super_window=mnps_cfg["super_window"],
                        ridge_alpha=mnps_cfg["ridge_alpha"],
                        distance_weighted=bool(config.get("mnps", {}).get("ridge", {}).get("distance_weighted", True)),
                        j_dot_dt=float(dt),
                    )
                else:
                    logger.warning(
                        "Skipping v2 Jacobian for %s due to non-finite coords_9d rows "
                        "(degraded v2 coverage).",
                        dataset_label,
                    )

        # Contract invariants: provenance hashes for "inputs used to compute outputs"
        # must match the representation saved to H5.
        x_saved = np.asarray(x, dtype=np.float32)
        x_hash_saved = _stable_hash_array(x_saved)
        x_hash_knn_input = _stable_hash_array(np.asarray(x, dtype=np.float32))
        x_hash_jac_input = _stable_hash_array(np.asarray(x, dtype=np.float32))
        if not (x_hash_saved == x_hash_knn_input == x_hash_jac_input):
            raise RuntimeError("Direct x contract violation: saved x differs from kNN/Jacobian input.")

        coords_9d_hash_saved = None
        coords_9d_hash_knn_input = None
        coords_9d_hash_jac_input = None
        if coords_9d is not None and coords_9d_names and coords_9d.size:
            coords_9d_saved = np.asarray(coords_9d, dtype=np.float32)
            coords_9d_hash_saved = _stable_hash_array(coords_9d_saved)
            if jac_res_v2 is not None:
                coords_9d_hash_knn_input = _stable_hash_array(np.asarray(coords_9d, dtype=np.float32))
                coords_9d_hash_jac_input = _stable_hash_array(np.asarray(coords_9d, dtype=np.float32))
                if not (
                    coords_9d_hash_saved == coords_9d_hash_knn_input == coords_9d_hash_jac_input
                ):
                    raise RuntimeError("v2 contract violation: saved coords_9d differs from v2 kNN/Jacobian input.")

        # Optional Stratified (v2) block-Jacobian summaries and cross-partials
        stratified_blocks_result = None
        if jac_res_v2 is not None and coords_9d_names and getattr(jac_res_v2, "j_hat", None) is not None:
            try:
                stratified_blocks_result = compute_stratified_blocks_and_cross_partials(
                    ds_id=self.dataset.ds_id,
                    dataset_label=dataset_label,
                    subject=sub_id,
                    session=ses_id,
                    condition=condition,
                    task=task,
                    coords_9d_names=coords_9d_names,
                    jacobian_9D=jac_res_v2.j_hat,
                    config=config,
                )
            except Exception:
                logger.exception(
                    "Failed to compute stratified (v2) block summaries / cross-partials for %s",
                    dataset_label,
                )

        # Extract auxiliary arrays
        stage = extract_stage_array(sub_frame, mnps_cfg["stage_codebook"])
        stage_source: Optional[str] = None
        stage_column: Optional[str] = None
        stage_events_path: Optional[str] = None
        if stage is None:
            try:
                stage, stage_source, stage_column, stage_events_path = self._infer_stage_from_bids_events(sub_frame)
            except Exception:
                logger.exception("Failed to infer stage labels from BIDS events for %s", dataset_label)
        else:
            stage_source = "features_csv"
            for c in ["stage", "stage_code", "sleep_stage", "labels_stage"]:
                if c in sub_frame.columns:
                    stage_column = c
                    break
        stage_frac_labeled = None
        if stage is not None and len(stage) > 0:
            try:
                stage_frac_labeled = float(np.mean(np.asarray(stage) != -1))
            except Exception:
                stage_frac_labeled = None
        z = extract_embodied_array(sub_frame, mnps_cfg["embodied"])
        events = extract_events(sub_frame)

        # Load regional fMRI if available (pass sub_frame for correct file matching)
        regions_bold, regions_names, regions_sfreq = self._load_regional_fmri(
            sub_id=sub_id,
            dataset_label=dataset_label,
            config=config,
            sub_frame=sub_frame,
            raw_task=raw_task,
            condition=condition,
            session=ses_id,
            run_id=run_id,
        )

        # Group regions and compute optional regional MNPS/MNJ context.
        regional_mnps_cfg = config.get("regional_mnps", {}) if isinstance(config, Mapping) else {}
        group_ts, group_matrix, group_names, region_groups, regional_mnps_results = compute_regional_context(
            sub_frame=sub_frame,
            regions_bold=regions_bold,
            regions_names=regions_names,
            regions_sfreq=regions_sfreq,
            config=config,
            regional_mnps_cfg=regional_mnps_cfg if isinstance(regional_mnps_cfg, Mapping) else {},
            subcoords_spec=subcoords_spec if isinstance(subcoords_spec, Mapping) else {},
            axis_weights=self.ctx.weights if isinstance(self.ctx.weights, Mapping) else {},
            dataset_id=self.dataset.ds_id,
            dataset_label=dataset_label,
            proj_cfg=proj_cfg if isinstance(proj_cfg, Mapping) else {},
            normalize_mode=normalize_mode,
            subject=sub_id,
            session=ses_id,
            condition=condition,
            task=task,
            resolve_mnps_3d_cfg=_resolve_mnps_3d_cfg,
            coerce_v1_mapping_to_v2_subcoords=_coerce_v1_mapping_to_v2_subcoords,
            align_v2_subcoords=_align_v2_subcoords,
        )

        if regional_mnps_cfg.get("enabled", False) and regional_mnps_results is not None:
            # Persist regional MNPS and block-Jacobian summaries at the
            # dataset level so they can be consumed by analysis code
            # without re-estimating Jacobians.
            self.dataset.write_regional_csv_outputs_threadsafe(
                regional_mnps_results=regional_mnps_results,
                regional_mnps_cfg=regional_mnps_cfg if isinstance(regional_mnps_cfg, Mapping) else {},
                mnps_dir=self.mnps_dir,
                config=config,
                dataset_label=dataset_label,
            )

        tabular_exports_h5: Dict[str, Any] = {}
        if regional_mnps_results is not None:
            regional_rows = summary_to_dataframe_rows(regional_mnps_results)
            if regional_rows:
                tabular_exports_h5["regional_mnps_subjects"] = _rows_to_columnar_table(regional_rows)
            regional_block_rows = compute_block_jacobian_rows(
                regional_mnps_results,
                config,
                include_self=False,
            )
            if regional_block_rows:
                tabular_exports_h5["regional_block_jacobians_subjects"] = _rows_to_columnar_table(regional_block_rows)
        if stratified_blocks_result is not None and stratified_blocks_result.block_rows:
            tabular_exports_h5["stratified_block_jacobians_subjects"] = _rows_to_columnar_table(
                stratified_blocks_result.block_rows
            )

        # Optional: write Stratified (v2) block-Jacobian CSV into the MNPS run directory
        self.dataset.write_stratified_blocks_csv_output_threadsafe(
            stratified_blocks_result=stratified_blocks_result,
            config=config,
            dataset_id=self.dataset.ds_id,
            mnps_dir=self.mnps_dir,
            dataset_label=dataset_label,
        )

        # Compute extensions (E-Kappa, RFM, O-Koh, TIG)
        extensions_payload, extensions_summary = compute_extensions(
            dataset_label=dataset_label,
            extensions_cfg=self.ctx.extensions_cfg,
            x=x,
            sub_frame=sub_frame,
            time=time,
            dt=dt,
            coords_9d=coords_9d,
            coords_9d_names=coords_9d_names,
            regions_bold=regions_bold,
            regions_sfreq=regions_sfreq,
            group_ts=group_ts,
            group_matrix=group_matrix,
            group_names=group_names,
            region_groups=region_groups,
        )
        if tabular_exports_h5:
            merged_extensions = dict(extensions_payload) if isinstance(extensions_payload, Mapping) else {}
            existing_tables = merged_extensions.get("tabular_exports")
            merged_tables = dict(existing_tables) if isinstance(existing_tables, Mapping) else {}
            merged_tables.update(tabular_exports_h5)
            merged_extensions["tabular_exports"] = merged_tables
            extensions_payload = merged_extensions

        # Ensemble and robustness summaries
        ensemble_summary = None
        if v2_enabled and coords_9d is not None and coords_9d_names:
            ensemble_summary = compute_ensemble_summary_for_subject(
                config=config,
                dataset_id=self.dataset.ds_id,
                sub_frame=sub_frame,
                coords_9d_names=coords_9d_names,
                subcoords_spec=subcoords_spec,
                normalize_mode=normalize_mode,
            )

        robust_summary = compute_robust_and_reliability_summaries(
            config=config,
            mnps_cfg=mnps_cfg,
            x=x,
            coords_9d=coords_9d,
            coords_9d_names=coords_9d_names,
        )

        # Neutral distributional descriptives (mean/median/std/iqr + delta).
        dist_summary = None
        try:
            dist_summary = compute_dist_summary(x=x, coords_9d=coords_9d, coords_9d_names=coords_9d_names)
        except Exception:
            logger.exception("Failed to compute dist_summary for %s", dataset_label)

        # Tier-1 time structure: autocorrelation length (tau)
        tau_summary = None
        try:
            tau_cfg = (config.get("robustness", {}) or {}).get("tau", {}) if isinstance(config, Mapping) else {}
            tau_nan_policy = str(tau_cfg.get("nan_policy", "strict")).strip().lower()
            tau_axes = compute_tau_summary(x, ["m", "d", "e"], dt_sec=float(dt), nan_policy=tau_nan_policy)
            tau_v2 = (
                compute_tau_summary(coords_9d, list(coords_9d_names), dt_sec=float(dt), nan_policy=tau_nan_policy)
                if coords_9d is not None and coords_9d_names
                else {}
            )
            tau_summary = {"axes": tau_axes, "subcoords": tau_v2} if (tau_axes or tau_v2) else None
        except Exception:
            logger.exception("Failed to compute tau_summary for %s", dataset_label)

        # Tier-2 MNJ-adjacent metrics from the primary Jacobian + derived indices
        tier2_jac = None
        try:
            tier2_jac = compute_tier2_jacobian_metrics(
                jac_res.j_hat,
                jacobian_diagnostics=jac_res.diagnostics,
            )
        except Exception:
            logger.exception("Failed to compute tier2_jacobian_metrics for %s", dataset_label)

        emmi = None
        try:
            emmi = compute_emmi_metrics(x=x, x_dot=x_dot)
        except Exception:
            logger.exception("Failed to compute EMMI metrics for %s", dataset_label)

        multiverse_psd = None
        if v2_enabled and coords_9d is not None and coords_9d_names and subcoords_spec:
            multiverse_psd = compute_psd_multiverse_stability(
                config=config,
                ds_id=self.dataset.ds_id,
                sub_frame=sub_frame,
                coords_9d=coords_9d,
                coords_9d_names=coords_9d_names,
                subcoords_spec=subcoords_spec,
                normalize_mode=normalize_mode,
            )

        env_meta = _get_env_provenance()
        feature_export_bundle = projection.build_feature_export_bundle(
            sub_frame,
            direct_features=direct_weighted_features,
            v2_features=v2_weighted_features,
            normalize_mode=normalize_mode,
            feature_standardization=feature_standardization if isinstance(feature_standardization, Mapping) else None,
            clip_threshold=clip_threshold,
            entropy_meta=entropy_meta,
        )
        features_raw_values = np.asarray(feature_export_bundle.get("raw_values"), dtype=np.float32)
        features_raw_names = list(feature_export_bundle.get("raw_names", []) or [])
        features_robust_z_values = np.asarray(feature_export_bundle.get("robust_z_values"), dtype=np.float32)
        features_robust_z_names = list(feature_export_bundle.get("robust_z_names", []) or [])
        feature_metadata = dict(feature_export_bundle.get("metadata", {}) or {})
        feature_names_hash = (
            hashlib.sha256(
                json.dumps(features_raw_names, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if features_raw_names
            else None
        )
        features_raw_hash_saved = _stable_hash_array(features_raw_values) if features_raw_values.size else None
        features_robust_z_hash_saved = (
            _stable_hash_array(features_robust_z_values) if features_robust_z_values.size else None
        )

        # Build payload
        payload = schema.MNPSPayload(
            time=time,
            x=x,
            x_dot=x_dot,
            window_start=window_start,
            window_end=window_end,
            stage=stage,
            z=z,
            events=events,
            nn_indices=nn_indices,
            jacobian=jac_res.j_hat,
            jacobian_dot=jac_res.j_dot,
            jacobian_centers=jac_res.centers,
            jacobian_9D=jac_res_v2.j_hat if jac_res_v2 is not None else None,
            jacobian_9D_dot=jac_res_v2.j_dot if jac_res_v2 is not None else None,
            jacobian_9D_centers=jac_res_v2.centers if jac_res_v2 is not None else None,
            feature_baselines=merged_baselines,
            features_raw_values=features_raw_values,
            features_raw_names=features_raw_names,
            features_robust_z_values=features_robust_z_values,
            features_robust_z_names=features_robust_z_names,
            feature_metadata=feature_metadata,
            attrs={
                # Stable identity fields (used downstream for grouping/contrasts).
                "dataset": self.dataset.ds_id,
                "subject_id": sub_id,
                "session": ses_id,
                "fs_out": mnps_cfg["fs_out"],
                "window_sec": mnps_cfg["window_sec"],
                "overlap": mnps_cfg["overlap"],
                "stage_codebook": mnps_cfg["stage_codebook"],
                "stage_source": stage_source,
                "stage_column": stage_column,
                "stage_events_path": stage_events_path,
                "stage_frac_labeled": stage_frac_labeled,
                "participant_meta": participant_meta,
                "group": mapped_meta.get("group"),
                "condition": condition,
                "task": task,
                "run": run_id,
                "acq": acq_id,
                "coverage_rule_tag": coverage_tag,
                "coverage_min_seconds_effective": min_seconds_eff,
                "coverage_min_epochs_effective": min_epochs_eff,
                "coverage_seconds_effective": coverage_seconds_effective,
                "coverage_seconds_measured": coverage_seconds_measured,
                "coverage_seconds_assumed": coverage_seconds_assumed,
                "coverage_seconds_method": coverage_method,
                "epochs_raw": n_before_any,
                "epochs_after_qc": n_after_qc,
                "epochs_after_nan_mask": int(len(sub_frame)),
                "x_definition": x_definition,
                "mde_mode_requested": mde_mode_requested,
                "mde_mode_effective": mde_mode_effective,
                "mde_from_v2_aggregation_requested": mde_from_v2_aggregation_requested,
                "mde_from_v2_aggregation": mde_from_v2_aggregation,
                "mde_from_v2_pooling_legacy": mde_from_v2_pooling_legacy,
                "mde_from_v2_map": m3d_cfg.get("map"),
                "mde_from_v2_v1_mapping_source": m3d_cfg.get("v1_mapping_source"),
                "mde_from_v2_v1_mapping_input": m3d_cfg.get("v1_mapping"),
                "mde_from_v2_v1_mapping_normalized": v1_mapping_normalized,
                "mde_from_v2_v1_mapping_matrix": v1_mapping_matrix,
                "mde_from_v2_v1_mapping_matrix_rows": v1_mapping_matrix_rows,
                "mde_from_v2_v1_mapping_hash": v1_mapping_hash,
                "mde_from_v2_fallback_reason": mde_from_v2_reason,
                "v2_definition": f"subcoords_9d_v{str(v2_definition_version).replace('.', '_')}",
                "mnps_9d_definition_version": v2_definition_version,
                "mnps_9d_constructs": mnps_9d_constructs,
                "normalize_mode": normalize_mode,
                "missing_axis_policy": missing_axis_policy,
                "dropped_missing_axis_epochs": dropped_missing_axis_epochs,
                "weights_hash_direct": _stable_hash_mapping(self.ctx.weights or {}),
                "subcoords_hash_v2": _stable_hash_mapping(subcoords_spec if isinstance(subcoords_spec, Mapping) else {}),
                "missing_weighted_feature_rate_direct": missing_rate_direct,
                "missing_weighted_feature_rate_v2": missing_rate_v2,
                "direct_axis_renorm": direct_axis_renorm,
                "direct_axis_coverage_m_mean": axis_cov_stats["m_mean"],
                "direct_axis_coverage_d_mean": axis_cov_stats["d_mean"],
                "direct_axis_coverage_e_mean": axis_cov_stats["e_mean"],
                "direct_axis_coverage_m_min": float(np.nanmin(x_coverage[:, 0])) if x_coverage.size and not np.all(np.isnan(x_coverage[:, 0])) else float("nan"),
                "direct_axis_coverage_d_min": float(np.nanmin(x_coverage[:, 1])) if x_coverage.size and not np.all(np.isnan(x_coverage[:, 1])) else float("nan"),
                "direct_axis_coverage_e_min": float(np.nanmin(x_coverage[:, 2])) if x_coverage.size and not np.all(np.isnan(x_coverage[:, 2])) else float("nan"),
                "min_axis_coverage": float(min_axis_coverage),
                "v2_missing_policy": v2_missing_policy if v2_enabled and subcoords_spec else None,
                "coords_9d_allow_all_non_finite_columns": True if v2_enabled and subcoords_spec else False,
                "coords_9d_degraded_mode": bool(v2_all_non_finite_count > 0),
                "coords_9d_all_non_finite_count": int(v2_all_non_finite_count),
                "coords_9d_all_non_finite_names": v2_all_non_finite_names if v2_all_non_finite_names else None,
                "e_e_construct": entropy_meta.get("construct"),
                "e_e_metric": entropy_meta.get("metric"),
                "e_e_backend": entropy_meta.get("backend"),
                "e_e_degraded_mode": bool(entropy_meta.get("degraded_mode", False)),
                "e_e_reason": entropy_meta.get("reason"),
                "python_version": env_meta.get("python_version"),
                "platform": env_meta.get("platform"),
                "pip_freeze_hash": env_meta.get("pip_freeze_hash"),
                "env_hash": env_meta.get("env_hash"),
                "x_hash_saved": x_hash_saved,
                "x_hash_knn_input": x_hash_knn_input,
                "x_hash_jacobian_input": x_hash_jac_input,
                "coords_9d_hash_saved": coords_9d_hash_saved,
                "coords_9d_hash_knn_input": coords_9d_hash_knn_input,
                "coords_9d_hash_jacobian_input": coords_9d_hash_jac_input,
                "coords_9d_names": coords_9d_names if coords_9d_names else None,
                "feature_export_scope": "all_numeric_feature_columns_excluding_metadata",
                "feature_export_names_hash": feature_names_hash,
                "features_raw_hash_saved": features_raw_hash_saved,
                "features_robust_z_hash_saved": features_robust_z_hash_saved,
                "features_raw_column_count": int(len(features_raw_names)),
                "features_robust_z_column_count": int(len(features_robust_z_names)),
                "feature_metadata_fields": sorted(feature_metadata.keys()) if feature_metadata else [],
            },
        )
        if v2_enabled and coords_9d_names and coords_9d is not None and coords_9d.size:
            payload.coords_9d = coords_9d.astype(np.float32)
            payload.coords_9d_names = coords_9d_names
        if stratified_blocks_result is not None and stratified_blocks_result.cross_partials_series:
            payload.jacobian_9D_cross_partials = stratified_blocks_result.cross_partials_series
        if regions_bold is not None:
            payload.regions_bold = regions_bold
            if regions_names is not None:
                payload.regions_names = regions_names
            if regions_sfreq is not None:
                payload.regions_sfreq = regions_sfreq
        if extensions_payload:
            payload.extensions = extensions_payload

        # Event mapping to labels (opt-in)
        labels_mapped = self._map_events_to_labels(config, time, window_start, window_end, events)
        if labels_mapped:
            payload.labels = labels_mapped

        # Add regional MNPS/MNJ results
        if regional_mnps_results and regional_mnps_results.n_networks > 0:
            for network_label, result in regional_mnps_results.results.items():
                payload.regional_mnps[network_label] = {
                    "mnps": result.mnps,
                    "mnps_dot": result.mnps_dot,
                    "jacobian": result.jacobian,
                    "stratified": result.stratified,  # [T, 9] – None when 9D not run
                    "metrics": result.metrics,
                    "n_timepoints": result.n_timepoints,
                }

        # Entropy QC checks
        entropy_qc = {}
        if coords_9d is not None and coords_9d_names:
            try:
                entropy_qc = robustness.entropy_sanity_checks(coords_9d, coords_9d_names)
            except Exception:
                logger.exception("Entropy sanity checks failed for %s", dataset_label)

        # Build manifest
        manifest_extra = {
            "subject": sub_id,
            "session": ses_id,
            "run": run_id,
            "acq": acq_id,
            "participant_meta": participant_meta,
            "group": mapped_meta.get("group"),
            "condition": condition,
            "task": task,
            "stage_source": stage_source,
            "stage_column": stage_column,
            "stage_events_path": stage_events_path,
            "stage_frac_labeled": stage_frac_labeled,
            "coverage": {
                "rule_tag": coverage_tag,
                "min_seconds_effective": min_seconds_eff,
                "min_epochs_effective": min_epochs_eff,
                "seconds_effective": coverage_seconds_effective,
                "seconds_measured": coverage_seconds_measured,
                "seconds_assumed": coverage_seconds_assumed,
                "seconds_method": coverage_method,
            },
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        if stratified_blocks_result is not None:
            if stratified_blocks_result.blocks_manifest:
                rows_light: list[dict[str, Any]] = []
                for r in stratified_blocks_result.block_rows:
                    rows_light.append(
                        {
                            "out_group": r.get("out_group"),
                            "in_group": r.get("in_group"),
                            "out_dim": r.get("out_dim"),
                            "in_dim": r.get("in_dim"),
                            "n_timepoints": r.get("n_timepoints"),
                            "block_trace_mean": r.get("block_trace_mean"),
                            "block_frobenius_mean": r.get("block_frobenius_mean"),
                            "block_anisotropy_mean": r.get("block_anisotropy_mean"),
                            "c_sym_mean": r.get("c_sym_mean"),
                            "c_rot_mean": r.get("c_rot_mean"),
                        }
                    )
                manifest_extra["jacobian_9D_blocks"] = {
                    "config": stratified_blocks_result.blocks_manifest,
                    "rows": rows_light,
                }
            if stratified_blocks_result.cross_partials_manifest:
                manifest_extra["jacobian_9D_cross_partials"] = stratified_blocks_result.cross_partials_manifest
        if ensemble_summary is not None:
            manifest_extra["ensemble_robustness"] = ensemble_summary
        if robust_summary:
            manifest_extra["robust_summary"] = robust_summary
        if dist_summary:
            manifest_extra["dist_summary"] = dist_summary
        if tau_summary:
            manifest_extra["tau_summary"] = tau_summary
        if tier2_jac:
            manifest_extra["tier2_jacobian"] = tier2_jac
        if emmi:
            manifest_extra["tier2_emmi"] = emmi
        if multiverse_psd is not None:
            manifest_extra["multiverse_psd"] = multiverse_psd
        if entropy_qc:
            manifest_extra["entropy_qc"] = entropy_qc
        if extensions_summary:
            manifest_extra["extensions"] = extensions_summary
        if self.ctx.ingest_meta:
            manifest_extra["ndt_ingest"] = self.ctx.ingest_meta
        if modularity_provisional_frac is not None:
            manifest_extra["fmri_modularity_provisional_frac"] = modularity_provisional_frac
        if labels_mapped:
            manifest_extra["events_mapped"] = sorted(labels_mapped.keys())
        manifest_extra["mnps_3d"] = {
            "mode_requested": mde_mode_requested,
            "mode_effective": mde_mode_effective,
            "x_definition": x_definition,
            "from_v2": {
                "aggregation_requested": mde_from_v2_aggregation_requested,
                "aggregation": mde_from_v2_aggregation,
                "legacy_pooling": mde_from_v2_pooling_legacy,
                "map": m3d_cfg.get("map"),
                "v1_mapping_source": m3d_cfg.get("v1_mapping_source"),
                "v1_mapping_input": m3d_cfg.get("v1_mapping"),
                "v1_mapping_normalized": v1_mapping_normalized,
                "v1_mapping_matrix": v1_mapping_matrix,
                "v1_mapping_matrix_rows": v1_mapping_matrix_rows,
                "v1_mapping_hash": v1_mapping_hash,
                "fallback_reason": mde_from_v2_reason,
            },
        }
        manifest_extra["provenance"] = {
            "mnps_9d_definition_version": v2_definition_version,
            "mnps_9d_constructs": mnps_9d_constructs,
        }
        manifest_extra["feature_exports"] = {
            "raw_h5_path": "/features_raw",
            "robust_z_h5_path": "/features_robust_z",
            "scope": "all_numeric_feature_columns_excluding_metadata",
            "column_count": int(len(features_raw_names)),
            "names_hash": feature_names_hash,
            "metadata_fields": sorted(feature_metadata.keys()) if feature_metadata else [],
        }

        # Add a clear note indicating that these tier-2 metrics are tentative and belong in the analysis repo
        if any([tau_summary, tier2_jac, emmi, dist_summary]):
            manifest_extra["_TENTATIVE_NOTE"] = (
                "Metrics such as tier2_jacobian, tier2_emmi, tau_summary, and dist_summary "
                "are provided as tentative QA summaries only. Real statistical verification "
                "and interpretation must be performed downstream in the analysis repository."
            )

        manifest = json_writer.build_manifest(dataset_label, payload, jac_res.diagnostics, manifest_extra)

        self._write_qc_files(
            target_dir=target_dir,
            dataset_label=dataset_label,
            sub_id=sub_id,
            ses_id=ses_id,
            sub_frame=sub_frame,
            dt=dt,
            ensemble_summary=ensemble_summary,
            robust_summary=robust_summary,
            dist_summary=dist_summary,
            tau_summary=tau_summary,
            tier2_jacobian=tier2_jac,
            tier2_emmi=emmi,
            entropy_qc=entropy_qc,
        )
        write_summary_manifest_and_h5(
            target_dir=target_dir,
            dataset_label=dataset_label,
            manifest=manifest,
            payload=payload,
            jacobian_diagnostics=jac_res.diagnostics,
            sub_id=sub_id,
            ses_id=ses_id,
            condition=condition,
            task=task,
            run_id=run_id,
            acq_id=acq_id,
            build_dir_suffix=self._build_dir_suffix,
        )

    @staticmethod
    def _apply_fd_censoring(
        sub_frame: pd.DataFrame,
        fd_thresh: float = 0.5,
        pad: int = 1,
        *,
        require_fd: bool = False,
        context_label: Optional[str] = None,
    ) -> pd.DataFrame:
        """Drop epochs with framewise displacement above threshold (and pad neighbours)."""
        return apply_fd_censoring(
            sub_frame=sub_frame,
            fd_thresh=fd_thresh,
            pad=pad,
            require_fd=require_fd,
            context_label=context_label,
            logger=logger,
        )

    @staticmethod
    def _build_dir_suffix(
        ses_id: Optional[str],
        condition: Optional[str],
        task: Optional[str],
        run_id: Optional[str] = None,
        acq_id: Optional[str] = None,
    ) -> Optional[str]:
        """Build directory/filename suffix from session, condition, and task.

        Priority:
        1. If condition+task: "condition_task" (e.g., "awake_audio")
        2. If only condition: "condition" (e.g., "awake")
        3. If only session: "ses_id" (e.g., "ses-01")
        4. If only task (no condition): "task" (e.g., "rest")
        5. None if nothing available
        """
        return build_dir_suffix(ses_id, condition, task, run_id, acq_id)

    def _resolve_bold_path_for_subframe(
        self,
        sub_frame: pd.DataFrame,
        raw_task: Optional[str],
        condition: Optional[str],
        session: Optional[str],
        run_id: Optional[str],
        acq_id: Optional[str],
    ) -> Optional[Path]:
        """Resolve BOLD NIfTI path for the current grouped sub_frame."""
        return resolve_bold_path_for_subframe(
            sub_frame=sub_frame,
            raw_task=raw_task,
            condition=condition,
            session=session,
            run_id=run_id,
            acq_id=acq_id,
            dataset_root=self._dataset_root(),
            index_df=self.index_df,
            lookup_rel_paths_by_file_value=self.dataset._lookup_rel_paths_by_file_value,
        )

    def _resolve_confounds_path_for_bold(self, bold_path: Path) -> Optional[Path]:
        """Resolve BIDS confounds TSV corresponding to a BOLD file."""
        cache_key = str(bold_path)
        if cache_key in self._confounds_path_cache:
            return self._confounds_path_cache[cache_key]

        dataset_root = self._dataset_root()
        bold_name = bold_path.name
        if not bold_name.endswith((".nii", ".nii.gz")):
            self._confounds_path_cache[cache_key] = None
            return None

        bold_base = bold_name.replace(".nii.gz", "").replace(".nii", "")
        if bold_base.endswith("_bold"):
            conf_name = f"{bold_base[:-5]}_desc-confounds_timeseries.tsv"
        else:
            conf_name = f"{bold_base}_desc-confounds_timeseries.tsv"

        cfg = self.ctx.config if isinstance(self.ctx.config, Mapping) else {}
        preprocess_cfg = cfg.get("preprocess", {}) if isinstance(cfg, Mapping) else {}
        fmri_cfg = preprocess_cfg.get("fmri", {}) if isinstance(preprocess_cfg, Mapping) else {}
        if not isinstance(fmri_cfg, Mapping):
            fmri_cfg = {}
        ds_overrides = fmri_cfg.get("datasets", {}) if isinstance(fmri_cfg.get("datasets", {}), Mapping) else {}
        ds_cfg = ds_overrides.get(self.dataset.ds_id, {}) if isinstance(ds_overrides, Mapping) else {}
        if not isinstance(ds_cfg, Mapping):
            ds_cfg = {}
        merged_fmri_cfg: Dict[str, Any] = {k: v for k, v in fmri_cfg.items() if k != "datasets"}
        merged_fmri_cfg.update(dict(ds_cfg))
        conf_cfg = merged_fmri_cfg.get("confounds", {}) if isinstance(merged_fmri_cfg.get("confounds", {}), Mapping) else {}

        candidates: List[Path] = [bold_path.parent / conf_name]

        roots: List[Path] = []

        def _append_root(v: Any) -> None:
            if not isinstance(v, (str, Path)):
                return
            p = Path(str(v))
            if not p.is_absolute():
                p = (dataset_root / p).resolve()
            roots.append(p)

        _append_root(conf_cfg.get("confounds_root"))
        _append_root(conf_cfg.get("derivatives_dir"))
        _append_root(merged_fmri_cfg.get("confounds_root"))
        _append_root(merged_fmri_cfg.get("derivatives_dir"))
        roots.append(dataset_root / "derivatives")

        try:
            rel_parent = bold_path.parent.relative_to(dataset_root)
            for root in roots:
                candidates.append(root / rel_parent / conf_name)
                candidates.append(root / "fmriprep" / rel_parent / conf_name)
                candidates.append(root / self.dataset.ds_id / rel_parent / conf_name)
                candidates.append(root / self.dataset.ds_id / "fmriprep" / rel_parent / conf_name)
        except Exception:
            pass

        for c in candidates:
            if c.exists():
                self._confounds_path_cache[cache_key] = c
                return c

        token = conf_name.replace("_desc-confounds_timeseries.tsv", "")
        for root in roots:
            if root.exists():
                for c in root.rglob("*desc-confounds_timeseries.tsv"):
                    name = c.name
                    if token in name:
                        self._confounds_path_cache[cache_key] = c
                        return c
        self._confounds_path_cache[cache_key] = None
        return None

    def _merge_fd_from_confounds(
        self,
        sub_frame: pd.DataFrame,
        raw_task: Optional[str],
        condition: Optional[str],
        session: Optional[str],
        run_id: Optional[str],
        acq_id: Optional[str],
    ) -> pd.DataFrame:
        """Populate framewise_displacement per epoch from BIDS confounds TSV."""
        if sub_frame.empty or "framewise_displacement" in sub_frame.columns:
            return sub_frame

        bold_path = self._resolve_bold_path_for_subframe(
            sub_frame=sub_frame,
            raw_task=raw_task,
            condition=condition,
            session=session,
            run_id=run_id,
            acq_id=acq_id,
        )
        if bold_path is None:
            logger.warning("FD merge skipped: could not resolve BOLD path for current fMRI segment")
            return sub_frame

        conf_path = self._resolve_confounds_path_for_bold(bold_path)
        if conf_path is None:
            logger.warning("FD merge skipped: no confounds TSV found for %s", bold_path.name)
            return sub_frame

        try:
            conf_df = pd.read_csv(conf_path, sep="\t")
        except Exception:
            logger.warning("FD merge skipped: failed reading confounds TSV %s", conf_path)
            return sub_frame
        if conf_df.empty:
            logger.warning("FD merge skipped: empty confounds TSV %s", conf_path)
            return sub_frame

        fd_col = None
        for col in conf_df.columns:
            if str(col).strip().lower() == "framewise_displacement":
                fd_col = col
                break
        if fd_col is None:
            logger.warning("FD merge skipped: framewise_displacement missing in %s", conf_path)
            return sub_frame

        fd = pd.to_numeric(conf_df[fd_col], errors="coerce").to_numpy(dtype=float)
        if fd.size == 0:
            logger.warning("FD merge skipped: no FD values in %s", conf_path)
            return sub_frame

        sfreq = np.nan
        if "fmri_sfreq" in sub_frame.columns:
            try:
                sfreq = float(pd.to_numeric(sub_frame["fmri_sfreq"], errors="coerce").dropna().iloc[0])
            except Exception:
                sfreq = np.nan
        if not np.isfinite(sfreq) or sfreq <= 0:
            try:
                import nibabel as nib  # type: ignore

                bold_img = nib.load(str(bold_path))
                zooms = bold_img.header.get_zooms()
                tr = float(zooms[3]) if len(zooms) > 3 else np.nan
                if np.isfinite(tr) and tr > 0:
                    sfreq = 1.0 / tr
            except Exception:
                sfreq = np.nan
        if not np.isfinite(sfreq) or sfreq <= 0:
            logger.warning("FD merge skipped: could not infer fMRI sfreq for %s", bold_path.name)
            return sub_frame

        tr = 1.0 / float(sfreq)
        frame_t = np.arange(fd.size, dtype=float) * tr

        out = sub_frame.copy()
        if "t_start" in out.columns and "t_end" in out.columns:
            t_start = pd.to_numeric(out["t_start"], errors="coerce").to_numpy(dtype=float)
            t_end = pd.to_numeric(out["t_end"], errors="coerce").to_numpy(dtype=float)
        else:
            epoch_ids = (
                pd.to_numeric(out["epoch_id"], errors="coerce").fillna(-1).astype(int).to_numpy()
                if "epoch_id" in out.columns
                else np.arange(len(out), dtype=int)
            )
            if "fmri_step_sec" in out.columns:
                step_sec = float(pd.to_numeric(out["fmri_step_sec"], errors="coerce").dropna().iloc[0])
            else:
                step_sec = tr
            if "fmri_window_sec" in out.columns:
                window_sec = float(pd.to_numeric(out["fmri_window_sec"], errors="coerce").dropna().iloc[0])
            else:
                window_sec = step_sec
            t_start = epoch_ids.astype(float) * float(step_sec)
            t_end = t_start + float(window_sec)

        fd_epoch = np.full((len(out),), np.nan, dtype=float)
        for i, (s, e) in enumerate(zip(t_start, t_end)):
            if not (np.isfinite(s) and np.isfinite(e) and e > s):
                continue
            mask = (frame_t >= s) & (frame_t < e)
            vals = fd[mask]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            fd_epoch[i] = float(np.nanmax(vals))

        out["framewise_displacement"] = fd_epoch.astype(np.float32)
        logger.info(
            "Merged framewise_displacement from confounds for %s (%d/%d finite epochs)",
            bold_path.name,
            int(np.isfinite(fd_epoch).sum()),
            int(len(fd_epoch)),
        )
        return out

    def _infer_stage_from_bids_events(
        self, sub_frame: pd.DataFrame
    ) -> tuple[Optional[np.ndarray], Optional[str], Optional[str], Optional[str]]:
        """Infer per-epoch sleep stage codes from the BIDS *_events.tsv file.

        Intended for sleep datasets such as ds005555 (BOAS) where stage labels are
        provided as columns (e.g., stage_hum / stage_ai) in the events TSV.
        """
        return infer_stage_from_bids_events(
            sub_frame=sub_frame,
            index_df=self.index_df,
            dataset_root=self._dataset_root(),
            lookup_rel_paths_by_file_value=self.dataset._lookup_rel_paths_by_file_value,
            ctx_config=self.ctx.config if isinstance(self.ctx.config, Mapping) else {},
            mnps_cfg=self.ctx.mnps_cfg if isinstance(self.ctx.mnps_cfg, Mapping) else {},
            dataset_id=self.dataset.ds_id,
        )

    @staticmethod
    def _estimate_coverage_seconds(sub_frame: pd.DataFrame, dt_fallback: float) -> tuple[float, str]:
        """Estimate coverage from timestamps when available; otherwise fallback to len*dt."""
        return estimate_coverage_seconds(sub_frame, dt_fallback)

    def _load_regional_fmri(
        self,
        sub_id: str,
        dataset_label: str,
        config: Mapping[str, Any],
        sub_frame: pd.DataFrame,
        raw_task: Optional[str],
        condition: Optional[str],
        session: Optional[str],
        run_id: Optional[str],
    ) -> tuple[Optional[np.ndarray], Optional[List[str]], Optional[float]]:
        """Load regional fMRI signals if available.

        Uses the file from sub_frame to ensure we load the correct BOLD file
        for this specific (subject, condition, task) combination.
        """
        return load_regional_fmri_signals(
            sub_id=sub_id,
            dataset_label=dataset_label,
            config=config,
            sub_frame=sub_frame,
            raw_task=raw_task,
            condition=condition,
            session=session,
            run_id=run_id,
            dataset_root=self._dataset_root(),
            index_df=self.index_df,
            lookup_rel_paths_by_file_value=self.dataset._lookup_rel_paths_by_file_value,
            preprocess_fmri=preprocess.preprocess_fmri,
            logger=logger,
        )

    @staticmethod
    def _extract_time_bounds(sub_frame: pd.DataFrame, time: np.ndarray, window_sec: float) -> tuple[np.ndarray, np.ndarray]:
        """Return per-window start/end seconds, using t_start/t_end if available."""
        return extract_time_bounds(sub_frame, time, window_sec)

    def _map_events_to_labels(
        self,
        config: Mapping[str, Any],
        time: np.ndarray,
        window_start: np.ndarray,
        window_end: np.ndarray,
        events: Mapping[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Map event timestamps to MNPS window-aligned binary labels (opt-in)."""
        return map_events_to_labels(
            config=config,
            time=time,
            window_start=window_start,
            window_end=window_end,
            events=events,
            dataset_id=self.dataset.ds_id,
        )

    def _write_qc_files(
        self,
        target_dir: Path,
        dataset_label: str,
        sub_id: str,
        ses_id: Optional[str],
        sub_frame: pd.DataFrame,
        dt: float,
        ensemble_summary: Optional[Dict[str, Any]],
        robust_summary: Optional[Dict[str, Any]],
        dist_summary: Optional[Dict[str, Any]],
        tau_summary: Optional[Dict[str, Any]],
        tier2_jacobian: Optional[Dict[str, Any]],
        tier2_emmi: Optional[Dict[str, Any]],
        entropy_qc: Optional[Dict[str, Any]],
    ) -> None:
        """Write QC-related JSON files."""
        write_qc_files(
            target_dir=target_dir,
            dataset_label=dataset_label,
            ds_path=self.ds_path,
            sub_id=sub_id,
            ses_id=ses_id,
            sub_frame=sub_frame,
            dt=dt,
            ensemble_summary=ensemble_summary,
            robust_summary=robust_summary,
            dist_summary=dist_summary,
            tau_summary=tau_summary,
            tier2_jacobian=tier2_jacobian,
            tier2_emmi=tier2_emmi,
            entropy_qc=entropy_qc,
        )
