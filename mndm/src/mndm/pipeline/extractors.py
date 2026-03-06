"""Data extraction utilities for MNPS summarization."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
def _load_condition_task_policy(policy_dir: str, dataset_id: str) -> Dict[str, Any]:
    """Load an optional per-dataset condition/task setup YAML.

    Intended to standardize task/condition/group normalization across runs without
    editing the monolithic ingest YAML. The file is expected at:
      <policy_dir>/<dataset_id>_conditions_task_setup.yml
    """
    try:
        import yaml  # type: ignore
    except Exception:
        return {}

    if not policy_dir or not dataset_id:
        return {}

    root = Path(policy_dir)
    ds_path = root / f"{dataset_id}_conditions_task_setup.yml"
    map_path = root / "datasets.yml"
    path = ds_path if ds_path.exists() else (map_path if map_path.exists() else None)
    if path is None:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Failed to parse condition/task policy %s: %s", path, exc)
        return {}

    payload: Dict[str, Any] = {}
    if path.name == "datasets.yml":
        # Supported shapes:
        # - {datasets: {<ds>: {...}}}
        # - {metadata_extraction: {datasets: {<ds>: {...}}}}
        # - {<ds>: {...}}
        if isinstance(data, Mapping):
            if isinstance(data.get("metadata_extraction"), Mapping):
                md = data.get("metadata_extraction") or {}
                ds_map = md.get("datasets", {}) if isinstance(md, Mapping) else {}
                if isinstance(ds_map, Mapping):
                    payload = dict(ds_map.get(dataset_id, {}) or {})
            if not payload and isinstance(data.get("datasets"), Mapping):
                payload = dict((data.get("datasets") or {}).get(dataset_id, {}) or {})
            if not payload and isinstance(data.get(dataset_id), Mapping):
                payload = dict(data.get(dataset_id) or {})
    else:
        # Allow either:
        # - top-level {group/condition/task/...}
        # - or {metadata_extraction: {group/condition/task/...}}
        if isinstance(data, Mapping) and isinstance(data.get("metadata_extraction"), Mapping):
            payload = dict(data.get("metadata_extraction") or {})
        else:
            payload = dict(data) if isinstance(data, Mapping) else {}

    for k in ("schema", "schema_version", "dataset_id", "description"):
        payload.pop(k, None)
    return payload


def extract_mapped_metadata(
    participant_meta: Dict[str, Any],
    config: Mapping[str, Any],
    dataset_id: str,
    session: Optional[str],
    filename: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """Derive generic metadata fields (group/condition/task) from config rules and participant_meta.

    Parameters
    ----------
    participant_meta : dict
        Row from participants.tsv as a dictionary.
    config : Mapping
        Full ingest configuration.
    dataset_id : str
        Dataset identifier (e.g., "ds003171").
    session : str, optional
        Session identifier (e.g., "ses-01").
    filename : str, optional
        BIDS filename for parsing task (e.g., "sub-01_ses-01_task-rest_bold.nii.gz").

    Returns
    -------
    dict
        Keys: group, condition, task - each may be None if not found.
    """
    result: Dict[str, Optional[str]] = {"group": None, "condition": None, "task": None}
    meta = participant_meta or {}

    spec = (config.get("metadata_extraction", {}) if isinstance(config, Mapping) else {}) or {}
    default = spec.get("default", {})
    per_ds = (spec.get("datasets", {}) or {}).get(dataset_id, {})

    # Optional: overlay a per-dataset policy YAML (stored outside the main ingest config)
    policy_dir = spec.get("policy_dir")
    if policy_dir and isinstance(policy_dir, (str, Path)):
        policy = _load_condition_task_policy(str(policy_dir), dataset_id)
        if policy:
            per_ds = _deep_merge_dict(per_ds if isinstance(per_ds, Mapping) else {}, policy)

    def _normalize(value: Optional[str], mapping: Dict[str, str]) -> Optional[str]:
        if value is None:
            return None
        raw = str(value).strip().lower()
        stripped = raw.replace("-", "").replace("_", "")
        return mapping.get(raw, mapping.get(stripped, value))

    def _as_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            if isinstance(value, (int, float, np.floating, np.integer)):
                return float(value)
            s = str(value).strip()
            if s == "" or s.lower() in {"nan", "none", "null"}:
                return None
            return float(s)
        except Exception:
            return None

    def _apply_numeric_rules(rules: Any) -> Optional[str]:
        """Apply simple numeric threshold rules for group assignment.

        Expected format:
          numeric_rules:
            - {column: attention, op: \">=\", value: 1.0, label: ADHD_proxy}
            - {column: attention, op: \"<=\", value: -1.0, label: Control_proxy}
        """
        if not isinstance(rules, (list, tuple)):
            return None
        for rule in rules:
            if not isinstance(rule, Mapping):
                continue
            col = rule.get("column")
            op = str(rule.get("op", "")).strip()
            label = rule.get("label")
            if not col or label is None:
                continue
            x = _as_float(meta.get(str(col)))
            thr = _as_float(rule.get("value"))
            if x is None or thr is None:
                continue
            ok = False
            if op == ">=":
                ok = x >= thr
            elif op == ">":
                ok = x > thr
            elif op == "<=":
                ok = x <= thr
            elif op == "<":
                ok = x < thr
            elif op in ("==", "="):
                ok = x == thr
            elif op == "!=":
                ok = x != thr
            if ok:
                return str(label)
        return None

    def _parse_task_from_filename(fname: str) -> Optional[str]:
        """Extract task label from BIDS filename (e.g., 'task-rest' → 'rest')."""
        match = re.search(r"(?:^|[_/\\])task-([A-Za-z0-9]+)", fname)
        return match.group(1) if match else None

    def _parse_session_from_filename(fname: str) -> Optional[str]:
        """Extract session label from BIDS filename (e.g., 'ses-awake' → 'ses-awake')."""
        match = re.search(r"(ses-[a-zA-Z0-9]+)", fname)
        return match.group(1) if match else None

    def _parse_acq_from_filename(fname: str) -> Optional[str]:
        """Extract acquisition label from BIDS filename (e.g., 'acq-eyesopen' → 'eyesopen')."""
        match = re.search(r"acq-([a-zA-Z0-9]+)", fname)
        return match.group(1) if match else None

    def _parse_compound_task(task_str: str, known_conditions: list[str]) -> tuple[Optional[str], Optional[str]]:
        """Parse compound task string into (task, condition).
        
        Some datasets (e.g., ds003171) encode condition in the task name:
        'audioawake' → ('audio', 'awake')
        'restdeep' → ('rest', 'deep')
        'rest' → ('rest', None)  # no condition suffix
        """
        if not task_str:
            return None, None
        task_lower = task_str.lower()
        for cond in sorted(known_conditions, key=lambda x: len(str(x)), reverse=True):
            cond_lower = cond.lower()
            if task_lower.endswith(cond_lower):
                actual_task = task_str[:-len(cond)]
                return actual_task if actual_task else None, cond
        return task_str, None

    # --- Group extraction ---
    group_spec = dict(default.get("group", {}))
    group_spec.update(per_ds.get("group", {}))

    # Priority 0: numeric threshold rules (useful for proxy groupings in datasets
    # that provide dimensional measures but no explicit diagnosis column).
    if result["group"] is None and "numeric_rules" in group_spec:
        result["group"] = _apply_numeric_rules(group_spec.get("numeric_rules"))

    for key in group_spec.get("candidates", []):
        if key in meta and isinstance(meta[key], (str, int, float)):
            result["group"] = str(meta[key]).strip()
            break
    if result["group"] is None and "default" in group_spec:
        result["group"] = group_spec["default"]
    if "normalize" in group_spec and result["group"]:
        result["group"] = _normalize(result["group"], group_spec["normalize"])

    # --- Condition extraction ---
    cond_spec = dict(default.get("condition", {}))
    cond_spec.update(per_ds.get("condition", {}))

    # Determine effective session (prefer argument, fallback to filename)
    effective_session = session
    if not effective_session and filename:
        effective_session = _parse_session_from_filename(filename)

    # Priority 1: Direct session_map (session → condition)
    session_map = cond_spec.get("session_map", {})
    if effective_session and effective_session in session_map:
        result["condition"] = session_map[effective_session]
    else:
        # Priority 2: Session-specific candidates from participants.tsv
        sess_default = cond_spec.get("session_candidates", {})
        sess_override = per_ds.get("condition", {}).get("session_candidates", {})
        merged_sess = dict(sess_default)
        merged_sess.update(sess_override)

        if effective_session and effective_session in merged_sess:
            for key in merged_sess[effective_session]:
                if key in meta and isinstance(meta[key], (str, int, float)):
                    result["condition"] = str(meta[key]).strip()
                    break

        # Priority 3: Generic candidates from participants.tsv
        if result["condition"] is None:
            for key in cond_spec.get("candidates", []):
                if key in meta and isinstance(meta[key], (str, int, float)):
                    result["condition"] = str(meta[key]).strip()
                    break

    # Apply default if still None
    if result["condition"] is None and "default" in cond_spec:
        result["condition"] = cond_spec["default"]
    if "normalize" in cond_spec and result["condition"]:
        result["condition"] = _normalize(result["condition"], cond_spec["normalize"])

    # Optional: infer condition from filename (best-effort).
    # Some datasets encode condition in the task label (e.g., task-awake → condition=awake).
    if (
        result["condition"] is None
        and cond_spec.get("from_filename", False)
        and cond_spec.get("allow_task_as_condition", False)
        and filename
    ):
        try:
            inferred = _parse_task_from_filename(filename)
        except Exception:
            inferred = None
        if inferred:
            result["condition"] = inferred
            if "normalize" in cond_spec and result["condition"]:
                result["condition"] = _normalize(result["condition"], cond_spec["normalize"])

    # Optional: infer condition from acquisition label (acq-*) in filename.
    # Useful when datasets keep task constant (e.g. task-rest) and encode
    # eyes-open/eyes-closed via acq-eyesopen / acq-eyesclosed.
    if result["condition"] is None and cond_spec.get("from_acq", False) and filename:
        try:
            inferred_acq = _parse_acq_from_filename(filename)
        except Exception:
            inferred_acq = None
        if inferred_acq:
            result["condition"] = inferred_acq
            if "normalize" in cond_spec and result["condition"]:
                result["condition"] = _normalize(result["condition"], cond_spec["normalize"])

    # --- Task extraction ---
    task_spec = dict(default.get("task", {}))
    task_spec.update(per_ds.get("task", {}))

    # Priority 1: Parse from filename if enabled
    raw_task = None
    if task_spec.get("from_filename", False) and filename:
        raw_task = _parse_task_from_filename(filename)

    # Check for compound task (e.g., "audioawake" → task="audio", condition="awake")
    compound_conditions = task_spec.get("compound_conditions", [])
    if raw_task and compound_conditions:
        parsed_task, parsed_cond = _parse_compound_task(raw_task, compound_conditions)
        result["task"] = parsed_task
        # Only set condition from compound if not already set
        if parsed_cond and result["condition"] is None:
            result["condition"] = parsed_cond
            # Apply normalization to compound-derived condition
            if "normalize" in cond_spec and result["condition"]:
                result["condition"] = _normalize(result["condition"], cond_spec["normalize"])
    elif raw_task:
        result["task"] = raw_task

    # Priority 2: candidates from participants.tsv
    if result["task"] is None:
        for key in task_spec.get("candidates", []):
            if key in meta and isinstance(meta[key], (str, int, float)):
                result["task"] = str(meta[key]).strip()
                break

    # Apply default if still None
    if result["task"] is None and "default" in task_spec:
        result["task"] = task_spec["default"]

    # Optional task normalization (mirrors group/condition normalization).
    # Useful to standardize task labels across datasets (case/underscores/dashes).
    if "normalize" in task_spec and result["task"]:
        try:
            norm_map = task_spec.get("normalize", {})
            if isinstance(norm_map, dict):
                result["task"] = _normalize(result["task"], norm_map)
        except Exception:
            # Best-effort; never fail metadata extraction due to task normalization.
            pass

    return result


def build_dataset_label(
    ds_id: str,
    sub_id: str,
    ses_id: Optional[str],
    condition: Optional[str],
    task: Optional[str],
    run: Optional[str] = None,
    acq: Optional[str] = None,
) -> str:
    """Build a descriptive dataset label including condition/task when available.

    Format: ds_id:sub_id[:condition][_task]

    Examples:
    - ds003171:sub-02CB:awake_rest
    - ds003171:sub-02CB:deep_audio
    - ds005114:sub-001 (no condition/task)
    """
    label = f"{ds_id}:{sub_id}"

    # Add condition if present
    if condition:
        label = f"{label}:{condition}"
    elif ses_id:
        # Fall back to session if no condition
        label = f"{label}:{ses_id}"

    # Append task if present
    if task:
        label = f"{label}_{task}"

    # Append run identifier if present (to disambiguate multi-run datasets).
    if run:
        label = f"{label}_{run}"

    # Append acquisition identifier if present (to disambiguate multi-acq datasets).
    if acq:
        label = f"{label}_{acq}"

    return label


def extract_stage_array(df: pd.DataFrame, stage_codebook: Dict[str, Any]) -> Optional[np.ndarray]:
    """Extract sleep stage array from DataFrame using codebook mapping."""
    candidate_cols = ["stage", "stage_code", "sleep_stage", "labels_stage"]
    mapping = {}
    for key, value in stage_codebook.items():
        try:
            mapping[str(key).lower()] = int(value)
        except Exception:
            continue

    for col in candidate_cols:
        if col in df.columns:
            series = df[col]
            if series.dtype.kind in {"U", "S", "O"} and mapping:
                mapped = [mapping.get(str(v).strip().lower(), -1) for v in series]
            else:
                mapped = series.fillna(-1).astype(int).tolist()
            return np.asarray(mapped, dtype=np.int8)
    return None


def extract_embodied_array(df: pd.DataFrame, embodied_cfg: Dict[str, Any]) -> Optional[np.ndarray]:
    """Extract embodied/interoceptive signals from DataFrame."""
    if not embodied_cfg or not embodied_cfg.get("enabled", False):
        return None
    channels = embodied_cfg.get("channels", [])
    available = [ch for ch in channels if ch in df.columns]
    if not available:
        return None
    return df[available].fillna(0).to_numpy(dtype=np.float32)


def extract_events(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Extract event columns (prefixed with 'event_') from DataFrame."""
    events: Dict[str, np.ndarray] = {}
    for col in df.columns:
        if str(col).startswith("event_"):
            name = str(col)[6:]
            values = df[col].dropna().to_numpy()
            events[name] = values
    return events


def load_participant_table(received_dir: Path, dataset_id: str) -> Optional[pd.DataFrame]:
    """Load participants.tsv from the received dataset directory."""
    tsv_path = received_dir / dataset_id / "participants.tsv"
    if not tsv_path.exists():
        logger.warning(f"participants.tsv not found at {tsv_path}")
        return None
    try:
        df = pd.read_csv(tsv_path, sep="\t")
        # Normalize a few common variants to participant_id.
        if "participant_id" not in df.columns:
            for alt in ("subject_id", "participant", "subject"):
                if alt in df.columns:
                    df = df.rename(columns={alt: "participant_id"})
                    break
        if "participant_id" not in df.columns:
            logger.warning("participants.tsv missing participant_id-like column at %s", tsv_path)
            return None
        df["participant_id"] = df["participant_id"].astype(str)
        return df
    except Exception as exc:
        logger.warning(f"Failed to load participants.tsv: {exc}")
        return None
