"""Data extraction utilities for MNPS summarization."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
    """Derive generic metadata fields (group/condition/task) from config rules.

    Args:
        participant_meta: Row from ``participants.tsv`` as a dictionary.
        config: Full ingest configuration.
        dataset_id: Dataset identifier (for example ``ds003171``).
        session: Session identifier (for example ``ses-01``), or None.
        filename: BIDS filename for parsing task context, or None.

    Returns:
        Dict with keys ``group``, ``condition``, and ``task`` (each optional).
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

    def _is_scalar_meta_value(value: Any) -> bool:
        """Internal helper: is scalar meta value."""
        return isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_))

    def _pick_candidate_value(candidates: Any) -> Optional[str]:
        """Internal helper: pick candidate value."""
        if not isinstance(candidates, (list, tuple)):
            return None
        key_lookup: Dict[str, str] = {}
        for key in meta.keys():
            try:
                key_lookup[str(key).strip().lower()] = str(key)
            except Exception:
                continue
        for candidate in candidates:
            cand_norm = str(candidate).strip().lower()
            actual_key = key_lookup.get(cand_norm)
            if actual_key is None:
                continue
            value = meta.get(actual_key)
            if _is_scalar_meta_value(value):
                return str(value).strip()
        return None

    def _normalize(value: Optional[str], mapping: Dict[str, str]) -> Optional[str]:
        """Internal helper: normalize."""
        if value is None:
            return None
        raw = str(value).strip().lower()
        stripped = raw.replace("-", "").replace("_", "")
        return mapping.get(raw, mapping.get(stripped, value))

    def _as_float(value: Any) -> Optional[float]:
        """Internal helper: as float."""
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

    if result["group"] is None:
        result["group"] = _pick_candidate_value(group_spec.get("candidates", []))
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
                result["condition"] = _pick_candidate_value([key])
                if result["condition"] is not None:
                    break

        # Priority 3: Generic candidates from participants.tsv
        if result["condition"] is None:
            result["condition"] = _pick_candidate_value(cond_spec.get("candidates", []))

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
        result["task"] = _pick_candidate_value(task_spec.get("candidates", []))

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


def _pick_first_numeric_series(df: pd.DataFrame, candidates: Sequence[str]) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Return first finite numeric series found among candidate columns."""
    for col in candidates:
        if col in df.columns:
            arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(arr).any():
                return arr, str(col)
    return None, None


def _resolve_pseudo_stage_dataset_cfg(config: Mapping[str, Any], dataset_id: str) -> Dict[str, Any]:
    """Resolve merged pseudo-stage config with per-dataset overrides."""
    root = config.get("pseudo_stage", {}) if isinstance(config, Mapping) else {}
    if not isinstance(root, Mapping):
        return {}

    merged: Dict[str, Any] = {k: v for k, v in root.items() if k != "datasets"}
    datasets = root.get("datasets", {}) if isinstance(root.get("datasets"), Mapping) else {}
    ds_cfg = datasets.get(dataset_id, {}) if isinstance(datasets, Mapping) else {}
    if isinstance(ds_cfg, Mapping):
        merged = _deep_merge_dict(merged, ds_cfg)
    return merged


def derive_pseudo_stage_array(
    df: pd.DataFrame,
    config: Mapping[str, Any],
    dataset_id: str,
    stage_codebook: Mapping[str, Any],
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Derive an N2-like pseudo-stage from spindle/sigma/EMG proxies.

    The proxy is intended for datasets without explicit sleep staging labels.
    It combines three criteria:
    - spindle density proxy (or spindle-event density if provided),
    - sigma-like power,
    - low EMG/high-frequency proxy.
    """
    empty_meta: Dict[str, Any] = {"enabled": False}
    if df is None or len(df) == 0:
        return None, empty_meta

    pseudo_cfg = _resolve_pseudo_stage_dataset_cfg(config, dataset_id)
    if not pseudo_cfg or not bool(pseudo_cfg.get("enabled", False)):
        return None, empty_meta

    label = str(pseudo_cfg.get("label", "N2_PROXY"))
    unknown_code = int(pseudo_cfg.get("unknown_code", -1))

    fallback_code = None
    if isinstance(stage_codebook, Mapping):
        try:
            fallback_code = int(stage_codebook.get("N2"))
        except Exception:
            fallback_code = None
    if fallback_code is None:
        fallback_code = 2
    try:
        target_code = int(pseudo_cfg.get("code", fallback_code))
    except Exception:
        target_code = int(fallback_code)

    sigma_candidates = pseudo_cfg.get("sigma_columns", ["eeg_alpha", "eeg_beta", "eeg_theta"])
    emg_candidates = pseudo_cfg.get(
        "emg_columns",
        ["embodied_arousal_proxy", "eeg_highfreq_power_30_45", "eeg_gamma"],
    )
    spindle_candidates = pseudo_cfg.get(
        "spindle_columns",
        ["event_sleep_spindle_density", "spindle_density", "event_sleep_spindle_count", "event_sleep_spindle"],
    )
    if not isinstance(sigma_candidates, Sequence) or isinstance(sigma_candidates, (str, bytes)):
        sigma_candidates = ["eeg_alpha", "eeg_beta", "eeg_theta"]
    if not isinstance(emg_candidates, Sequence) or isinstance(emg_candidates, (str, bytes)):
        emg_candidates = ["embodied_arousal_proxy", "eeg_highfreq_power_30_45", "eeg_gamma"]
    if not isinstance(spindle_candidates, Sequence) or isinstance(spindle_candidates, (str, bytes)):
        spindle_candidates = ["event_sleep_spindle_density", "spindle_density", "event_sleep_spindle_count", "event_sleep_spindle"]

    sigma_arr, sigma_col = _pick_first_numeric_series(df, [str(c) for c in sigma_candidates])
    emg_arr, emg_col = _pick_first_numeric_series(df, [str(c) for c in emg_candidates])
    spindle_arr, spindle_col = _pick_first_numeric_series(df, [str(c) for c in spindle_candidates])

    # If no explicit spindle signal exists, reuse sigma-like activity as a spindle burst proxy.
    spindle_fallback_to_sigma = False
    if spindle_arr is None and sigma_arr is not None:
        spindle_arr = np.asarray(sigma_arr, dtype=float)
        spindle_col = sigma_col
        spindle_fallback_to_sigma = True

    min_criteria = int(pseudo_cfg.get("min_criteria", 2))
    min_labeled_fraction = float(pseudo_cfg.get("min_labeled_fraction", 0.05))
    sigma_q = float(pseudo_cfg.get("sigma_quantile_min", 0.60))
    emg_q = float(pseudo_cfg.get("emg_quantile_max", 0.40))
    spindle_burst_q = float(pseudo_cfg.get("spindle_burst_quantile_min", 0.75))
    spindle_density_window = max(1, int(pseudo_cfg.get("spindle_density_window_epochs", 15)))
    spindle_density_min = float(pseudo_cfg.get("spindle_density_min", 0.20))

    criteria: Dict[str, np.ndarray] = {}
    n = len(df)
    valid_mask = np.ones(n, dtype=bool)

    if sigma_arr is not None:
        sigma_thresh = float(np.nanquantile(sigma_arr, sigma_q))
        sigma_ok = np.isfinite(sigma_arr) & (sigma_arr >= sigma_thresh)
        criteria["sigma_high"] = sigma_ok
    if emg_arr is not None:
        emg_thresh = float(np.nanquantile(emg_arr, emg_q))
        emg_ok = np.isfinite(emg_arr) & (emg_arr <= emg_thresh)
        criteria["emg_low"] = emg_ok
    if spindle_arr is not None:
        spindle_burst_thresh = float(np.nanquantile(spindle_arr, spindle_burst_q))
        spindle_burst = np.isfinite(spindle_arr) & (spindle_arr >= spindle_burst_thresh)
        spindle_density = (
            pd.Series(spindle_burst.astype(float))
            .rolling(window=spindle_density_window, min_periods=1, center=True)
            .mean()
            .to_numpy(dtype=float)
        )
        spindle_ok = np.isfinite(spindle_density) & (spindle_density >= spindle_density_min)
        criteria["spindle_density_high"] = spindle_ok

    if not criteria:
        return None, {
            "enabled": True,
            "status": "no_proxy_columns_found",
            "label": label,
            "code": target_code,
        }

    criteria_mat = np.column_stack([v.astype(int) for v in criteria.values()])
    criteria_hits = np.sum(criteria_mat, axis=1)
    required_hits = min(max(1, min_criteria), criteria_mat.shape[1])
    proxy_mask = valid_mask & (criteria_hits >= required_hits)

    stage = np.full(n, unknown_code, dtype=np.int8)
    stage[proxy_mask] = np.int8(target_code)

    labeled_fraction = float(np.mean(stage != unknown_code)) if n > 0 else 0.0
    if labeled_fraction < min_labeled_fraction:
        return None, {
            "enabled": True,
            "status": "insufficient_proxy_coverage",
            "label": label,
            "code": target_code,
            "labeled_fraction": labeled_fraction,
            "min_labeled_fraction": min_labeled_fraction,
        }

    return stage, {
        "enabled": True,
        "status": "ok",
        "label": label,
        "code": target_code,
        "unknown_code": unknown_code,
        "labeled_fraction": labeled_fraction,
        "required_hits": required_hits,
        "criteria_used": list(criteria.keys()),
        "sigma_column": sigma_col,
        "emg_column": emg_col,
        "spindle_column": spindle_col,
        "spindle_fallback_to_sigma": spindle_fallback_to_sigma,
    }


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
    """Extract event columns (prefixed with ``event_``) from DataFrame."""
    events: Dict[str, np.ndarray] = {}
    for col in df.columns:
        if str(col).startswith("event_"):
            name = str(col)[6:]
            values = df[col].dropna().to_numpy()
            events[name] = values
    return events


def _resolve_dataset_root_for_participants(
    received_dir: Path,
    dataset_id: str,
    config: Optional[Mapping[str, Any]],
) -> Path:
    """Resolve dataset root, honoring per-dataset received-dir overrides."""
    try:
        from .. import bids_index

        return bids_index.resolve_dataset_root(config or {}, Path(received_dir), dataset_id)
    except Exception:
        return Path(received_dir) / dataset_id


def _normalize_participant_id_column(
    df: pd.DataFrame,
    participants_cfg: Mapping[str, Any],
    *,
    source_path: Path,
) -> Optional[pd.DataFrame]:
    """Normalize subject id aliases to canonical `participant_id`."""
    subject_id_candidates = participants_cfg.get(
        "subject_id_candidates",
        ["participant_id", "subject_id", "participant", "subject"],
    )
    subject_id_column = participants_cfg.get("subject_id_column")
    if isinstance(subject_id_column, str) and subject_id_column.strip():
        subject_id_candidates = [subject_id_column.strip(), *list(subject_id_candidates or [])]

    if "participant_id" not in df.columns:
        col_lookup = {str(col).strip().lower(): str(col) for col in df.columns}
        for alt in subject_id_candidates:
            actual = col_lookup.get(str(alt).strip().lower())
            if actual in df.columns:
                df = df.rename(columns={actual: "participant_id"})
                break
    if "participant_id" not in df.columns:
        logger.warning("Participants table missing participant_id-like column at %s", source_path)
        return None
    df["participant_id"] = df["participant_id"].astype(str)
    return df


def _parse_scalar_value(raw_value: str, cast_mode: str) -> Any:
    """Parse scalar metadata values according to cast mode."""
    text = str(raw_value).strip()
    if cast_mode != "auto":
        return text

    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if re.fullmatch(r"[-+]?\d+", text):
        try:
            return int(text)
        except Exception:
            return text
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text):
        try:
            return float(text)
        except Exception:
            return text
    return text


def _normalize_metadata_key(key: str, mode: str) -> str:
    """Normalize parsed sidecar keys to a configured shape."""
    normalized = str(key).strip()
    if mode == "lower":
        return normalized.lower()
    if mode == "snake_case":
        normalized = re.sub(r"[^0-9A-Za-z]+", "_", normalized).strip("_")
        return normalized.lower()
    return normalized


def _parse_key_value_sidecar(path: Path, sidecar_cfg: Mapping[str, Any]) -> Dict[str, str]:
    """Parse key/value sidecar file into a metadata dictionary."""
    encoding = str(sidecar_cfg.get("encoding", "utf-8"))
    separator = str(sidecar_cfg.get("key_value_separator", sidecar_cfg.get("separator", ":")))
    comment_prefixes_raw = sidecar_cfg.get("comment_prefixes", ["#", ";"])
    if isinstance(comment_prefixes_raw, Sequence) and not isinstance(comment_prefixes_raw, (str, bytes)):
        comment_prefixes = [str(x) for x in comment_prefixes_raw]
    else:
        comment_prefixes = ["#", ";"]

    try:
        text = path.read_text(encoding=encoding)
    except Exception as exc:
        logger.warning("Failed to read sidecar metadata file %s: %s", path, exc)
        return {}

    parsed: Dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped.startswith(prefix) for prefix in comment_prefixes):
            continue
        if separator not in stripped:
            continue
        key, value = stripped.split(separator, 1)
        key_norm = str(key).strip()
        if not key_norm:
            continue
        parsed[key_norm] = str(value).strip()
    return parsed


def _extract_subject_from_sidecar(
    sidecar_path: Path,
    parsed: Mapping[str, str],
    sidecar_cfg: Mapping[str, Any],
) -> Optional[str]:
    """Extract canonical participant id for one sidecar row."""
    subject_cfg = sidecar_cfg.get("subject_id", {}) if isinstance(sidecar_cfg, Mapping) else {}
    if not isinstance(subject_cfg, Mapping):
        subject_cfg = {}

    from_key = subject_cfg.get("from_key")
    from_filename = str(subject_cfg.get("from_filename", "stem"))
    regex = subject_cfg.get("regex")
    strip_prefixes_raw = subject_cfg.get("strip_prefixes", [])
    if isinstance(strip_prefixes_raw, Sequence) and not isinstance(strip_prefixes_raw, (str, bytes)):
        strip_prefixes = [str(x) for x in strip_prefixes_raw]
    else:
        strip_prefixes = []
    prefix = str(subject_cfg.get("prefix", "")).strip()
    pad_raw = subject_cfg.get("pad")
    pad = int(pad_raw) if isinstance(pad_raw, (int, str)) and str(pad_raw).strip() else None

    key_lookup = {str(k).strip().lower(): str(v) for k, v in parsed.items()}
    candidate_values: List[str] = []
    if isinstance(from_key, str) and from_key.strip():
        value = key_lookup.get(from_key.strip().lower())
        if value:
            candidate_values.append(value)

    filename_candidate: Optional[str] = None
    if from_filename == "name":
        filename_candidate = sidecar_path.name
    elif from_filename == "parent":
        filename_candidate = sidecar_path.parent.name
    elif from_filename == "path":
        filename_candidate = sidecar_path.as_posix()
    else:
        filename_candidate = sidecar_path.stem
    if filename_candidate:
        candidate_values.append(filename_candidate)

    token: Optional[str] = None
    for raw in candidate_values:
        candidate = str(raw).strip()
        if not candidate:
            continue
        if isinstance(regex, str) and regex.strip():
            match = re.search(regex, candidate)
            if not match:
                continue
            if "subject" in match.groupdict():
                candidate = str(match.group("subject"))
            elif match.groups():
                candidate = str(match.group(1))
            else:
                candidate = str(match.group(0))
        token = candidate.strip()
        if token:
            break

    if not token:
        return None

    for to_strip in strip_prefixes:
        if to_strip and token.lower().startswith(to_strip.lower()):
            token = token[len(to_strip) :]
            break

    if pad is not None and token.isdigit():
        token = token.zfill(pad)

    if prefix and not token.lower().startswith(prefix.lower()):
        token = f"{prefix}{token}"
    return token.strip() or None


def _load_sidecar_participant_table(
    dataset_root: Path,
    participants_cfg: Mapping[str, Any],
) -> Optional[pd.DataFrame]:
    """Load participant metadata from sidecar key/value files."""
    sidecar_cfg = participants_cfg.get("sidecar_files", {})
    if not isinstance(sidecar_cfg, Mapping):
        return None
    if not bool(sidecar_cfg.get("enabled", False)):
        return None

    parser_name = str(sidecar_cfg.get("parser", "key_value")).strip().lower()
    if parser_name != "key_value":
        logger.warning("Unsupported participants sidecar parser '%s' (expected key_value)", parser_name)
        return None

    globs_raw = sidecar_cfg.get("file_globs")
    file_globs: List[str] = []
    if isinstance(globs_raw, Sequence) and not isinstance(globs_raw, (str, bytes)):
        file_globs = [str(x).strip() for x in globs_raw if str(x).strip()]
    elif isinstance(globs_raw, str) and globs_raw.strip():
        file_globs = [globs_raw.strip()]
    else:
        single_glob = sidecar_cfg.get("file_glob")
        if isinstance(single_glob, str) and single_glob.strip():
            file_globs = [single_glob.strip()]
    if not file_globs:
        file_globs = ["**/*.txt"]

    cast_mode = str(sidecar_cfg.get("value_cast", "none")).strip().lower()
    key_mode = str(sidecar_cfg.get("key_normalization", "none")).strip().lower()
    include_keys_raw = sidecar_cfg.get("include_keys", [])
    exclude_keys_raw = sidecar_cfg.get("exclude_keys", [])
    include_keys_seq = (
        include_keys_raw
        if isinstance(include_keys_raw, Sequence) and not isinstance(include_keys_raw, (str, bytes))
        else []
    )
    exclude_keys_seq = (
        exclude_keys_raw
        if isinstance(exclude_keys_raw, Sequence) and not isinstance(exclude_keys_raw, (str, bytes))
        else []
    )
    include_keys = {str(x).strip().lower() for x in include_keys_seq if str(x).strip()}
    exclude_keys = {str(x).strip().lower() for x in exclude_keys_seq if str(x).strip()}
    include_source_file = bool(sidecar_cfg.get("include_source_file", False))
    source_column = str(sidecar_cfg.get("source_column", "source_file")).strip() or "source_file"

    column_map_cfg = sidecar_cfg.get("column_map", {})
    column_map: Dict[str, str] = {}
    if isinstance(column_map_cfg, Mapping):
        for key, value in column_map_cfg.items():
            key_norm = str(key).strip().lower()
            val_norm = str(value).strip()
            if key_norm and val_norm:
                column_map[key_norm] = val_norm

    sidecar_paths: List[Path] = []
    seen_paths: set[str] = set()
    for pattern in file_globs:
        try:
            candidates = dataset_root.glob(pattern)
        except Exception as exc:
            logger.warning("Invalid sidecar glob '%s' under %s: %s", pattern, dataset_root, exc)
            continue
        for path in candidates:
            if not path.is_file():
                continue
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            sidecar_paths.append(path)
    sidecar_paths.sort(key=lambda p: p.as_posix())

    rows_by_participant: Dict[str, Dict[str, Any]] = {}
    for sidecar_path in sidecar_paths:
        parsed = _parse_key_value_sidecar(sidecar_path, sidecar_cfg)
        if not parsed:
            continue
        participant_id = _extract_subject_from_sidecar(sidecar_path, parsed, sidecar_cfg)
        if not participant_id:
            continue

        row = rows_by_participant.setdefault(participant_id, {"participant_id": participant_id})
        for raw_key, raw_value in parsed.items():
            raw_key_norm = str(raw_key).strip().lower()
            if include_keys and raw_key_norm not in include_keys:
                continue
            if raw_key_norm in exclude_keys:
                continue
            mapped_key = column_map.get(raw_key_norm, str(raw_key).strip())
            mapped_key = _normalize_metadata_key(mapped_key, key_mode)
            if not mapped_key or mapped_key == "participant_id":
                continue
            row[mapped_key] = _parse_scalar_value(raw_value, cast_mode)
        if include_source_file:
            row[source_column] = sidecar_path.as_posix()

    if not rows_by_participant:
        return None

    rows = [rows_by_participant[key] for key in sorted(rows_by_participant.keys())]
    df = pd.DataFrame(rows)
    if "participant_id" not in df.columns:
        return None
    df["participant_id"] = df["participant_id"].astype(str)
    ordered_cols = ["participant_id", *[c for c in df.columns if c != "participant_id"]]
    df = df[ordered_cols]
    df.attrs["source_path"] = dataset_root.as_posix()
    df.attrs["source_format"] = "sidecar_key_value"
    df.attrs["subject_id_column"] = "participant_id"
    return df


def _merge_participant_tables(
    base_df: pd.DataFrame,
    sidecar_df: pd.DataFrame,
    *,
    merge_strategy: str,
) -> pd.DataFrame:
    """Merge a base participants table with an additional participant table."""
    left = base_df.copy()
    right = sidecar_df.copy()
    left["participant_id"] = left["participant_id"].astype(str)
    right["participant_id"] = right["participant_id"].astype(str)
    left_idx = left.set_index("participant_id")
    right_idx = right.set_index("participant_id")

    ordered_ids = list(left_idx.index)
    for pid in right_idx.index:
        if pid not in left_idx.index:
            ordered_ids.append(pid)

    merged = left_idx.reindex(ordered_ids)
    for col in right_idx.columns:
        sidecar_col = right_idx.reindex(ordered_ids)[col]
        if col not in merged.columns:
            merged[col] = sidecar_col
            continue
        if merge_strategy == "prefer_existing":
            merged[col] = merged[col].combine_first(sidecar_col)
        else:
            merged[col] = sidecar_col.combine_first(merged[col])

    out = merged.reset_index()
    out.attrs["source_path"] = str(base_df.attrs.get("source_path", ""))
    out.attrs["source_format"] = "merged_participants_and_sidecar"
    out.attrs["subject_id_column"] = "participant_id"
    return out


def _normalize_join_token(value: Any) -> str:
    """Normalize a join-key scalar to a comparable string token."""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def _merge_participant_tables_via_base_column(
    base_df: pd.DataFrame,
    sidecar_df: pd.DataFrame,
    *,
    base_column: str,
    merge_strategy: str,
) -> pd.DataFrame:
    """Merge extra metadata using a non-canonical key from the base table.

    Example: the base participants table uses canonical `participant_id`, while
    an auxiliary clinical TSV is keyed by `subject_id`. In that case we want to
    match `base_df[subject_id]` against `sidecar_df["participant_id"]`, while
    preserving the base table's canonical `participant_id` column.
    """
    left = base_df.copy()
    right = sidecar_df.copy()
    base_key = str(base_column or "").strip()
    if not base_key:
        return _merge_participant_tables(left, right, merge_strategy=merge_strategy)
    if base_key not in left.columns:
        logger.warning("Cannot join extra participants table via missing base column '%s'", base_key)
        return left
    if "participant_id" not in right.columns:
        logger.warning("Cannot join extra participants table via '%s': missing participant_id column", base_key)
        return left

    left["participant_id"] = left["participant_id"].astype(str)
    left["__join_key"] = left[base_key].map(_normalize_join_token)
    right["participant_id"] = right["participant_id"].map(_normalize_join_token)
    right = right[right["participant_id"] != ""].copy()

    duplicate_mask = right["participant_id"].duplicated(keep="last")
    if duplicate_mask.any():
        dup_count = int(duplicate_mask.sum())
        logger.warning(
            "Extra participants table %s had %d duplicate join ids for base column '%s'; keeping last occurrence",
            str(sidecar_df.attrs.get("source_path", "")),
            dup_count,
            base_key,
        )
        right = right.drop_duplicates(subset=["participant_id"], keep="last")

    right_idx = right.set_index("participant_id")
    left_join_keys = left["__join_key"]
    joinable_keys = {str(idx) for idx in right_idx.index if str(idx)}
    matched = int(left_join_keys.isin(joinable_keys).sum())
    if matched == 0 and len(left) > 0 and len(right_idx) > 0:
        logger.warning(
            "Extra participants table %s produced no matches via base column '%s'",
            str(sidecar_df.attrs.get("source_path", "")),
            base_key,
        )

    merged = left.copy()
    for col in right_idx.columns:
        sidecar_col = left_join_keys.map(right_idx[col])
        if col not in merged.columns:
            merged[col] = sidecar_col
            continue
        if merge_strategy == "prefer_existing":
            merged[col] = merged[col].combine_first(sidecar_col)
        else:
            merged[col] = sidecar_col.combine_first(merged[col])

    out = merged.drop(columns=["__join_key"], errors="ignore")
    out.attrs["source_path"] = str(base_df.attrs.get("source_path", ""))
    out.attrs["source_format"] = "merged_participants_and_extra_tables"
    out.attrs["subject_id_column"] = "participant_id"
    return out


def _apply_extra_table_column_rules(
    df: pd.DataFrame,
    table_cfg: Mapping[str, Any],
) -> pd.DataFrame:
    """Apply optional renaming/normalization rules to extra-table columns."""
    out = df.copy()
    key_mode = str(table_cfg.get("column_normalization", "none")).strip().lower()
    prefix = str(table_cfg.get("column_prefix", "")).strip()
    column_map_cfg = table_cfg.get("column_map", {})
    column_map: Dict[str, str] = {}
    if isinstance(column_map_cfg, Mapping):
        for key, value in column_map_cfg.items():
            key_norm = str(key).strip().lower()
            val_norm = str(value).strip()
            if key_norm and val_norm:
                column_map[key_norm] = val_norm

    rename: Dict[str, str] = {}
    for col in out.columns:
        original = str(col).strip()
        if not original or original == "participant_id":
            continue
        mapped = column_map.get(original.lower(), original)
        mapped = _normalize_metadata_key(mapped, key_mode)
        if prefix and mapped != "participant_id":
            mapped = f"{prefix}{mapped}"
        if mapped and mapped != original:
            rename[str(col)] = mapped
    if rename:
        out = out.rename(columns=rename)
    return out


def _load_additional_participant_tables(
    dataset_root: Path,
    participants_cfg: Mapping[str, Any],
) -> list[pd.DataFrame]:
    """Load optional extra participant tables such as phenotype TSV files."""
    extra_raw = participants_cfg.get("extra_tables", [])
    if not isinstance(extra_raw, Sequence) or isinstance(extra_raw, (str, bytes)):
        return []
    tables: list[pd.DataFrame] = []
    for item in extra_raw:
        if not isinstance(item, Mapping):
            continue
        path_value = item.get("path")
        if not isinstance(path_value, (str, Path)) or not str(path_value).strip():
            continue
        raw_path = Path(str(path_value).strip())
        candidate_path = raw_path if raw_path.is_absolute() else dataset_root / raw_path
        if not candidate_path.exists():
            logger.warning("Configured extra participants table not found: %s", candidate_path)
            continue

        sep = item.get("sep", item.get("delimiter"))
        if not isinstance(sep, str) or not sep:
            suffix = candidate_path.suffix.lower()
            if suffix == ".tsv":
                sep = "\t"
            elif suffix == ".csv":
                sep = ","
            else:
                sep = None
        read_kwargs: Dict[str, Any] = {}
        if sep is None:
            read_kwargs.update({"sep": None, "engine": "python"})
        else:
            read_kwargs["sep"] = sep
        try:
            loaded = pd.read_csv(candidate_path, **read_kwargs)
        except Exception as exc:
            logger.warning("Failed to load extra participants table %s: %s", candidate_path, exc)
            continue

        normalized = _normalize_participant_id_column(
            loaded,
            item,
            source_path=candidate_path,
        )
        if normalized is None:
            continue

        include_raw = item.get("include_columns", [])
        if isinstance(include_raw, Sequence) and not isinstance(include_raw, (str, bytes)):
            include_cols = [str(v).strip() for v in include_raw if str(v).strip()]
            if include_cols:
                keep = ["participant_id", *[c for c in include_cols if c in normalized.columns and c != "participant_id"]]
                normalized = normalized[[c for c in keep if c in normalized.columns]]

        normalized = _apply_extra_table_column_rules(normalized, item)

        normalized.attrs["source_path"] = str(candidate_path)
        normalized.attrs["source_format"] = candidate_path.suffix.lower().lstrip(".") or "text"
        normalized.attrs["subject_id_column"] = "participant_id"
        join_to_base = item.get("join_to_base_column", item.get("base_join_column"))
        if isinstance(join_to_base, str) and join_to_base.strip():
            normalized.attrs["join_to_base_column"] = join_to_base.strip()
        tables.append(normalized)
    return tables


def load_participant_table(
    received_dir: Path,
    dataset_id: str,
    config: Optional[Mapping[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    """Load participants table from the received dataset directory.

    Supported formats:
    - default dataset-root discovery: `participants.tsv`, `participants.csv`, `participants.txt`
    - YAML override via `metadata_extraction.[default|datasets.<id>].participants.path`
    - optional sidecar key/value metadata via `participants.sidecar_files`
    - optional tabular phenotype merges via `participants.extra_tables`
    """
    metadata_spec = (config.get("metadata_extraction", {}) if isinstance(config, Mapping) else {}) or {}
    default_cfg = metadata_spec.get("default", {}) if isinstance(metadata_spec, Mapping) else {}
    per_ds = (metadata_spec.get("datasets", {}) or {}).get(dataset_id, {}) if isinstance(metadata_spec, Mapping) else {}

    participants_cfg: Dict[str, Any] = {}
    if isinstance(default_cfg, Mapping) and isinstance(default_cfg.get("participants"), Mapping):
        participants_cfg.update(dict(default_cfg.get("participants") or {}))
    if isinstance(per_ds, Mapping) and isinstance(per_ds.get("participants"), Mapping):
        participants_cfg.update(dict(per_ds.get("participants") or {}))

    dataset_root = _resolve_dataset_root_for_participants(received_dir, dataset_id, config)
    path_value = participants_cfg.get("path")
    candidate_path: Optional[Path] = None
    if isinstance(path_value, (str, Path)) and str(path_value).strip():
        raw_path = Path(str(path_value).strip())
        candidate_path = raw_path if raw_path.is_absolute() else dataset_root / raw_path
        if not candidate_path.exists():
            logger.warning("Configured participants path not found for %s: %s", dataset_id, candidate_path)
            candidate_path = None
    else:
        for name in ("participants.tsv", "participants.csv", "participants.txt"):
            path = dataset_root / name
            if path.exists():
                candidate_path = path
                break

    table_df: Optional[pd.DataFrame] = None
    if candidate_path is not None:
        sep = participants_cfg.get("sep", participants_cfg.get("delimiter"))
        if not isinstance(sep, str) or not sep:
            suffix = candidate_path.suffix.lower()
            if suffix == ".tsv":
                sep = "\t"
            elif suffix == ".csv":
                sep = ","
            else:
                sep = None

        try:
            read_kwargs: Dict[str, Any] = {}
            if sep is None:
                read_kwargs.update({"sep": None, "engine": "python"})
            else:
                read_kwargs["sep"] = sep
            loaded = pd.read_csv(candidate_path, **read_kwargs)
            table_df = _normalize_participant_id_column(
                loaded,
                participants_cfg,
                source_path=candidate_path,
            )
            if table_df is not None:
                table_df.attrs["source_path"] = str(candidate_path)
                table_df.attrs["source_format"] = candidate_path.suffix.lower().lstrip(".") or "text"
                table_df.attrs["subject_id_column"] = "participant_id"
        except Exception as exc:
            logger.warning("Failed to load participants table %s: %s", candidate_path, exc)
            table_df = None

    sidecar_df = _load_sidecar_participant_table(dataset_root, participants_cfg)
    extra_tables = _load_additional_participant_tables(dataset_root, participants_cfg)

    if table_df is None and sidecar_df is None and not extra_tables:
        logger.warning("No participants table found for %s under %s", dataset_id, dataset_root)
        return None
    merged = table_df
    if merged is None:
        merged = sidecar_df
        sidecar_df = None
    elif sidecar_df is not None:
        merge_strategy = str(participants_cfg.get("sidecar_merge", "prefer_sidecar")).strip().lower()
        if merge_strategy not in {"prefer_sidecar", "prefer_existing"}:
            merge_strategy = "prefer_sidecar"
        merged = _merge_participant_tables(merged, sidecar_df, merge_strategy=merge_strategy)

    if merged is None and extra_tables:
        merged = extra_tables[0]
        extra_tables = extra_tables[1:]

    extra_merge_strategy = str(participants_cfg.get("extra_tables_merge", "prefer_sidecar")).strip().lower()
    if extra_merge_strategy not in {"prefer_sidecar", "prefer_existing"}:
        extra_merge_strategy = "prefer_sidecar"
    for extra_df in extra_tables:
        if merged is None:
            join_to_base = str(extra_df.attrs.get("join_to_base_column", "")).strip()
            if join_to_base:
                logger.warning(
                    "Skipping extra participants table %s because join_to_base_column='%s' requires a base participants table",
                    str(extra_df.attrs.get("source_path", "")),
                    join_to_base,
                )
                continue
            merged = extra_df
        else:
            join_to_base = str(extra_df.attrs.get("join_to_base_column", "")).strip()
            if join_to_base:
                merged = _merge_participant_tables_via_base_column(
                    merged,
                    extra_df,
                    base_column=join_to_base,
                    merge_strategy=extra_merge_strategy,
                )
            else:
                merged = _merge_participant_tables(merged, extra_df, merge_strategy=extra_merge_strategy)
            source_paths = [str(merged.attrs.get("source_path", "")).strip(), str(extra_df.attrs.get("source_path", "")).strip()]
            source_paths = [p for p in source_paths if p]
            if source_paths:
                merged.attrs["source_path"] = ";".join(dict.fromkeys(source_paths))
            merged.attrs["source_format"] = "merged_participants_and_extra_tables"

    return merged


def load_session_table(
    received_dir: Path,
    dataset_id: str,
    config: Optional[Mapping[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    """Load an optional per-session metadata table for non-BIDS sources.

    Configure ``metadata_extraction.datasets.<id>.sessions.path`` with a TSV
    or CSV containing ``participant_id`` and ``session_id``.  Unlike
    participant tables, this preserves repeated recordings and is joined by
    the summary runner only for its matching subject/session output.
    """
    metadata_spec = (config.get("metadata_extraction", {}) if isinstance(config, Mapping) else {}) or {}
    per_ds = (metadata_spec.get("datasets", {}) or {}).get(dataset_id, {}) if isinstance(metadata_spec, Mapping) else {}
    sessions_cfg = per_ds.get("sessions", {}) if isinstance(per_ds, Mapping) else {}
    if not isinstance(sessions_cfg, Mapping) or not sessions_cfg.get("path"):
        return None
    dataset_root = _resolve_dataset_root_for_participants(received_dir, dataset_id, config)
    raw_path = Path(str(sessions_cfg["path"]))
    path = raw_path if raw_path.is_absolute() else dataset_root / raw_path
    if not path.exists():
        logger.warning("Configured sessions path not found for %s: %s", dataset_id, path)
        return None
    try:
        frame = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    except Exception as exc:
        logger.warning("Failed to load sessions table %s: %s", path, exc)
        return None
    required = {"participant_id", "session_id"}
    if missing := required.difference(frame.columns):
        logger.warning("Sessions table %s lacks columns: %s", path, sorted(missing))
        return None
    frame = frame.copy()
    frame["participant_id"] = frame["participant_id"].astype(str)
    frame["session_id"] = frame["session_id"].astype(str)
    frame.attrs["source_path"] = str(path)
    return frame
