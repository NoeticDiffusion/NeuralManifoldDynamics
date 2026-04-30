"""Helpers for aligning NWB interval metadata to MNDM epochs."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class NwbEpochLabels:
    """NWB interval labels aligned to feature/MNPS epochs."""

    labels: Dict[str, np.ndarray] = field(default_factory=dict)
    stage: Optional[np.ndarray] = None
    stage_source: Optional[str] = None
    stage_column: Optional[str] = None
    manifest: Dict[str, Any] = field(default_factory=dict)


def resolve_state_labels_config(config: Mapping[str, Any], dataset_id: Optional[str]) -> Dict[str, Any]:
    """Resolve generic NWB state-label config with optional dataset overrides."""
    root = config.get("state_labels", {}) if isinstance(config, Mapping) else {}
    if not isinstance(root, Mapping):
        return {}
    merged: Dict[str, Any] = {str(k): v for k, v in root.items() if k != "datasets"}
    ds_map = root.get("datasets", {})
    if dataset_id and isinstance(ds_map, Mapping):
        ds_cfg = ds_map.get(dataset_id)
        if isinstance(ds_cfg, Mapping):
            merged = _deep_merge(merged, ds_cfg)

    # Reuse condition normalization when state_labels does not define its own map.
    if "normalize" not in merged:
        md = config.get("metadata_extraction", {}) if isinstance(config, Mapping) else {}
        md_ds = (md.get("datasets", {}) or {}).get(dataset_id, {}) if isinstance(md, Mapping) else {}
        condition = md_ds.get("condition", {}) if isinstance(md_ds, Mapping) else {}
        normalize = condition.get("normalize", {}) if isinstance(condition, Mapping) else {}
        if isinstance(normalize, Mapping) and normalize:
            merged["normalize"] = dict(normalize)
    return merged


def align_nwb_intervals_to_epochs(
    *,
    file_path: Path,
    config: Mapping[str, Any],
    dataset_id: Optional[str],
    epoch_meta: Sequence[tuple[int, int, int]],
    sfreq: float,
) -> NwbEpochLabels:
    """Align NWB TimeIntervals/behavioral fallback intervals to epoch rows."""
    cfg = resolve_state_labels_config(config, dataset_id)
    if not bool(cfg.get("enabled", False)):
        return NwbEpochLabels()
    if not str(file_path).lower().endswith(".nwb"):
        return NwbEpochLabels()
    if not epoch_meta:
        return NwbEpochLabels()

    output_name = str(cfg.get("output_name") or "nwb_state")
    source_name = str(cfg.get("source_output_name") or f"{output_name}_source")
    write_to_stage = bool(cfg.get("write_to_stage", True))
    write_to_features = bool(cfg.get("write_to_features", True))
    stage_codebook = _resolve_codebook(cfg)

    try:
        rows = read_nwb_interval_rows(file_path=file_path, config=config, dataset_id=dataset_id)
    except Exception:
        logger.exception("Failed reading NWB intervals for %s", file_path)
        return NwbEpochLabels(
            manifest={
                "enabled": True,
                "source": "nwb_intervals",
                "error": "read_failed",
            }
        )

    if not rows:
        return NwbEpochLabels(
            manifest={
                "enabled": True,
                "source": "nwb_intervals",
                "assigned_frac": 0.0,
                "interval_rows": 0,
            }
        )

    starts = np.asarray([float(start) / float(sfreq) for _, start, _ in epoch_meta], dtype=float)
    ends = np.asarray([float(end) / float(sfreq) for _, _, end in epoch_meta], dtype=float)
    mids = (starts + ends) / 2.0
    labels = np.full((len(epoch_meta),), "", dtype=object)
    sources = np.full((len(epoch_meta),), "", dtype=object)
    conflict_count = 0
    assigned_rows = 0
    overwrite = str(cfg.get("conflict_policy", "first")).strip().lower() == "last"
    point_events_to_windows = bool(cfg.get("point_events_to_windows", True))
    point_max = float(cfg.get("point_event_max_duration_sec", 1.0) or 1.0)

    for row in rows:
        start = _safe_float(row.get("start_time"))
        stop = _safe_float(row.get("stop_time"))
        label = row.get("label")
        if start is None or label is None:
            continue
        if stop is None or not math.isfinite(stop):
            stop = start

        duration = max(0.0, float(stop) - float(start))
        if duration > 0:
            mask = (mids >= float(start)) & (mids < float(stop))
        else:
            mask = np.isclose(mids, float(start), atol=1e-3)
        if point_events_to_windows and not mask.any() and duration <= point_max:
            mask = (starts <= float(start)) & (ends >= float(start))
        if not mask.any():
            continue

        source = f"{row.get('table', '')}:{row.get('label_column', '')}".strip(":")
        for idx in np.flatnonzero(mask):
            current = str(labels[idx]) if labels[idx] is not None else ""
            if current:
                if current != str(label):
                    conflict_count += 1
                    if overwrite:
                        labels[idx] = str(label)
                        sources[idx] = source
                continue
            labels[idx] = str(label)
            sources[idx] = source
        assigned_rows += 1

    assigned_mask = np.asarray([bool(str(v)) for v in labels], dtype=bool)
    out_labels: Dict[str, np.ndarray] = {}
    if write_to_features and assigned_mask.any():
        out_labels[output_name] = labels
        out_labels[source_name] = sources

    stage = None
    if write_to_stage and assigned_mask.any() and stage_codebook:
        stage = _encode_stage(labels, stage_codebook)
        if stage is not None and not np.any(stage != -1):
            stage = None

    manifest = {
        "enabled": True,
        "source": "nwb_intervals",
        "output_name": output_name,
        "source_output_name": source_name,
        "write_to_stage": write_to_stage,
        "write_to_features": write_to_features,
        "assigned_frac": float(np.mean(assigned_mask)) if assigned_mask.size else 0.0,
        "assigned_interval_rows": int(assigned_rows),
        "interval_rows": int(len(rows)),
        "conflict_count": int(conflict_count),
        "tables": sorted({str(row.get("table")) for row in rows if row.get("table")}),
    }
    if stage_codebook:
        manifest["stage_codebook"] = stage_codebook

    return NwbEpochLabels(
        labels=out_labels,
        stage=stage,
        stage_source="nwb_intervals" if stage is not None else None,
        stage_column=output_name if stage is not None else None,
        manifest=manifest,
    )


def read_nwb_interval_rows(
    *,
    file_path: Path,
    config: Mapping[str, Any],
    dataset_id: Optional[str],
) -> List[Dict[str, Any]]:
    """Read candidate NWB interval rows as normalized dictionaries."""
    cfg = resolve_state_labels_config(config, dataset_id)
    if not bool(cfg.get("enabled", False)):
        return []

    try:
        from pynwb import NWBHDF5IO
    except ImportError as exc:
        raise RuntimeError("pynwb is required for NWB interval extraction.") from exc

    rows: List[Dict[str, Any]] = []
    with NWBHDF5IO(str(file_path), "r", load_namespaces=True) as io:
        nwbfile = io.read()
        rows.extend(_read_time_intervals(nwbfile, cfg))
        if not rows and bool(cfg.get("behavioral_fallback", True)):
            rows.extend(_read_behavioral_interval_series(nwbfile, cfg))
    return rows


def _read_time_intervals(nwbfile: Any, cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    intervals = getattr(nwbfile, "intervals", {}) or {}
    try:
        available = [str(name) for name in intervals.keys()]
    except Exception:
        available = []
    if not available:
        return []

    requested = cfg.get("candidate_intervals", ["epochs", "trials"])
    requested_names = [str(x) for x in requested] if isinstance(requested, Sequence) and not isinstance(requested, (str, bytes)) else []
    ordered = [name for name in requested_names if name in available]
    if bool(cfg.get("include_custom_intervals", True)):
        ordered.extend([name for name in available if name not in ordered])
    ordered = list(dict.fromkeys(ordered))

    out: List[Dict[str, Any]] = []
    for table_name in ordered:
        try:
            table = intervals[table_name]
            frame = table.to_dataframe()
        except Exception:
            logger.exception("Failed reading NWB interval table %s", table_name)
            continue
        if frame is None or frame.empty:
            continue
        out.extend(_frame_to_interval_rows(frame, table_name=table_name, cfg=cfg))
    return out


def _frame_to_interval_rows(frame: pd.DataFrame, *, table_name: str, cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    label_candidates = _label_candidates(cfg)
    normalize = _normalization_map(cfg)
    out: List[Dict[str, Any]] = []
    for row_idx, row in frame.reset_index(drop=False).iterrows():
        start = _safe_float(row.get("start_time"))
        stop = _safe_float(row.get("stop_time"))
        if start is None:
            continue
        label = None
        label_col = None
        for candidate in label_candidates:
            if candidate not in frame.columns:
                continue
            label = _normalize_label(row.get(candidate), normalize)
            if label is not None:
                label_col = candidate
                break
        if label is None and bool(cfg.get("use_table_name_as_label", False)):
            label = _normalize_label(table_name, normalize)
            label_col = "table_name"
        if label is None:
            continue
        out.append(
            {
                "table": table_name,
                "row_index": int(row_idx),
                "start_time": float(start),
                "stop_time": float(stop) if stop is not None else float(start),
                "label": label,
                "label_column": label_col,
            }
        )
    return out


def _read_behavioral_interval_series(nwbfile: Any, cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Best-effort fallback for BehavioralEpochs/IntervalSeries interfaces."""
    rows: List[Dict[str, Any]] = []
    try:
        processing = getattr(nwbfile, "processing", {}) or {}
        containers: List[tuple[str, Any]] = []
        for mod_name, module in processing.items():
            containers.extend(_iter_named_children(module, str(mod_name)))
        for path, obj in containers:
            class_name = obj.__class__.__name__.lower()
            if "intervalseries" not in class_name:
                continue
            rows.extend(_interval_series_to_rows(path, obj, cfg))
    except Exception:
        logger.exception("Failed reading NWB behavioral interval fallback")
    return rows


def _iter_named_children(container: Any, prefix: str) -> Iterable[tuple[str, Any]]:
    try:
        items = container.items() if hasattr(container, "items") else []
    except Exception:
        items = []
    for name, obj in items:
        path = f"{prefix}/{name}" if prefix else str(name)
        yield path, obj
        if hasattr(obj, "data_interfaces"):
            try:
                yield from _iter_named_children(obj.data_interfaces, path)
            except Exception:
                pass
        if hasattr(obj, "interval_series"):
            try:
                yield from _iter_named_children(obj.interval_series, path)
            except Exception:
                pass


def _interval_series_to_rows(path: str, obj: Any, cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    try:
        timestamps = np.asarray(obj.timestamps[:], dtype=float)
        data = np.asarray(obj.data[:])
    except Exception:
        return []
    if timestamps.size == 0 or data.size == 0:
        return []
    n = min(int(timestamps.size), int(data.size))
    timestamps = timestamps[:n]
    data = data[:n]
    normalize = _normalization_map(cfg)
    label = _normalize_label(getattr(obj, "name", Path(path).name), normalize)
    if label is None:
        return []

    out: List[Dict[str, Any]] = []
    open_start: Optional[float] = None
    for ts, value in zip(timestamps, data):
        try:
            marker = float(value)
        except Exception:
            marker = 0.0
        if marker > 0 and open_start is None:
            open_start = float(ts)
        elif marker < 0 and open_start is not None:
            out.append(
                {
                    "table": path,
                    "row_index": len(out),
                    "start_time": float(open_start),
                    "stop_time": float(ts),
                    "label": label,
                    "label_column": "interval_series",
                }
            )
            open_start = None
    return out


def _label_candidates(cfg: Mapping[str, Any]) -> List[str]:
    raw = cfg.get(
        "label_candidates",
        [
            "condition",
            "state",
            "epoch",
            "anesthesia_state",
            "behavioral_epoch",
            "tags",
            "label",
            "stage",
            "sleep_stage",
            "trial_type",
            "stimulus_type",
            "stimulus_description",
        ],
    )
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [str(x) for x in raw]
    return ["condition", "state", "behavioral_epoch", "tags"]


def _normalization_map(cfg: Mapping[str, Any]) -> Dict[str, str]:
    raw = cfg.get("normalize", {})
    if not isinstance(raw, Mapping):
        return {}
    out: Dict[str, str] = {}
    for key, value in raw.items():
        val = _stringify_label(value)
        if val is None:
            continue
        for variant in _label_variants(key):
            out[variant] = val
    return out


def _resolve_codebook(cfg: Mapping[str, Any]) -> Dict[str, int]:
    raw = cfg.get("codebook", {})
    if not isinstance(raw, Mapping):
        return {}
    out: Dict[str, int] = {}
    for key, value in raw.items():
        try:
            code = int(value)
        except Exception:
            continue
        for variant in _label_variants(key):
            out[variant] = code
    return out


def _normalize_label(value: Any, normalize: Mapping[str, str]) -> Optional[str]:
    text = _stringify_label(value)
    if text is None:
        return None
    variants = _label_variants(text)
    for variant in variants:
        mapped = normalize.get(variant)
        if mapped is not None:
            return str(mapped)
    return variants[0]


def _encode_stage(labels: np.ndarray, codebook: Mapping[str, int]) -> Optional[np.ndarray]:
    out = np.full((labels.shape[0],), -1, dtype=np.int16)
    for idx, label in enumerate(labels):
        text = _stringify_label(label)
        if text is None:
            continue
        for variant in _label_variants(text):
            if variant in codebook:
                out[idx] = int(codebook[variant])
                break
    return out.astype(np.int8, copy=False)


def _stringify_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        for item in value:
            text = _stringify_label(item)
            if text is not None:
                return text
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "n/a", "na"}:
        return None
    return text


def _label_variants(value: Any) -> List[str]:
    text = _stringify_label(value)
    if text is None:
        return []
    lowered = re.sub(r"\s+", "_", text.strip().lower())
    compact = re.sub(r"[\s_\-]+", "", lowered)
    return list(dict.fromkeys([lowered, compact, text.strip()]))


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out.get(key, {}), value)
        else:
            out[key] = value
    return out
