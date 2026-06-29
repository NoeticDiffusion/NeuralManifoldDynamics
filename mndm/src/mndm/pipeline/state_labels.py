"""Generic builders for within-run labels aligned to the MNPS time axis."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WithinRunLabelsResult:
    """Time-aligned labels derived from within-run state rules."""

    stage: Optional[np.ndarray] = None
    stage_codebook: Optional[Dict[str, int]] = None
    stage_source: Optional[str] = None
    stage_column: Optional[str] = None
    labels: Dict[str, np.ndarray] = field(default_factory=dict)
    manifest: Dict[str, Any] = field(default_factory=dict)


def build_within_run_labels(
    *,
    config: Mapping[str, Any],
    dataset_id: str,
    dataset_root: Path,
    sub_id: str,
    ses_id: Optional[str],
    task: Optional[str],
    raw_task: Optional[str],
    run_id: Optional[str],
    acq_id: Optional[str],
    tr_sec: Optional[float],
    time: np.ndarray,
    window_start: np.ndarray,
    window_end: np.ndarray,
    sub_frame: pd.DataFrame,
) -> WithinRunLabelsResult:
    """Build within-run labels from generic config rules."""
    cfg = _resolve_within_run_labels_cfg(config, dataset_id)
    if not cfg.get("enabled", False):
        return WithinRunLabelsResult()

    rules = cfg.get("rules", [])
    if not isinstance(rules, Sequence) or isinstance(rules, (str, bytes)):
        return WithinRunLabelsResult()

    output_name = str(cfg.get("output_name") or cfg.get("codebook_name") or "within_run_state")
    write_to_stage = bool(cfg.get("write_to_stage", True))
    write_to_labels = bool(cfg.get("write_to_labels", True))
    codebook = _resolve_codebook(
        dataset_id=dataset_id,
        cfg=cfg,
    )
    midpoints = _compute_midpoints(time=time, window_start=window_start, window_end=window_end)

    merged = np.full((midpoints.shape[0],), None, dtype=object)
    matched_rules: list[dict[str, Any]] = []
    conflict_count = 0

    for idx, rule in enumerate(rules):
        if not isinstance(rule, Mapping):
            continue
        if not _rule_matches(
            rule=rule,
            sub_id=sub_id,
            ses_id=ses_id,
            task=task,
            raw_task=raw_task,
            run_id=run_id,
            acq_id=acq_id,
        ):
            continue
        try:
            series, rule_meta = _evaluate_rule(
                rule=rule,
                dataset_root=dataset_root,
                sub_id=sub_id,
                task=task,
                raw_task=raw_task,
                tr_sec=tr_sec,
                midpoints=midpoints,
                sub_frame=sub_frame,
            )
        except Exception:
            logger.exception(
                "Failed within-run label rule %s for %s:%s/%s",
                rule.get("id", idx),
                dataset_id,
                sub_id,
                run_id or "run-unknown",
            )
            continue
        if series is None:
            continue

        filled = 0
        for pos, value in enumerate(series):
            if value is None:
                continue
            if merged[pos] is None:
                merged[pos] = value
                filled += 1
            elif merged[pos] != value:
                conflict_count += 1
        if rule_meta:
            rule_meta["filled_windows"] = filled
        matched_rules.append(rule_meta)

    labeled_mask = np.array([value is not None for value in merged], dtype=bool)
    if not labeled_mask.any():
        return WithinRunLabelsResult(
            manifest={
                "enabled": True,
                "output_name": output_name,
                "write_to_stage": write_to_stage,
                "write_to_labels": write_to_labels,
                "matched_rules": matched_rules,
                "conflict_count": conflict_count,
                "assigned_frac": 0.0,
            }
        )

    labels_out: Dict[str, np.ndarray] = {}
    if write_to_labels:
        labels_out[output_name] = np.asarray(
            ["" if value is None else str(value) for value in merged],
            dtype=object,
        )

    stage = None
    if write_to_stage:
        stage = _encode_stage_series(merged, codebook)

    assigned_frac = float(np.mean(labeled_mask)) if labeled_mask.size else 0.0
    manifest = {
        "enabled": True,
        "output_name": output_name,
        "write_to_stage": write_to_stage,
        "write_to_labels": write_to_labels,
        "codebook_name": cfg.get("codebook_name"),
        "stage_codebook": codebook if codebook else None,
        "matched_rules": matched_rules,
        "conflict_count": int(conflict_count),
        "assigned_frac": assigned_frac,
    }
    return WithinRunLabelsResult(
        stage=stage,
        stage_codebook=codebook if stage is not None and codebook else None,
        stage_source="within_run_labels" if stage is not None else None,
        stage_column=output_name if stage is not None else None,
        labels=labels_out,
        manifest=manifest,
    )


def _resolve_within_run_labels_cfg(config: Mapping[str, Any], dataset_id: str) -> Dict[str, Any]:
    """Merge global and per-dataset within-run label config."""
    root = config.get("within_run_labels", {}) if isinstance(config, Mapping) else {}
    if not isinstance(root, Mapping):
        return {}
    merged = {k: v for k, v in root.items() if k != "datasets"}
    ds_override = (root.get("datasets", {}) or {}).get(dataset_id, {})
    if isinstance(ds_override, Mapping):
        for key, value in ds_override.items():
            if key == "rules" and isinstance(value, list):
                merged["rules"] = list(value)
            else:
                merged[key] = value
    return merged


def _rule_matches(
    *,
    rule: Mapping[str, Any],
    sub_id: str,
    ses_id: Optional[str],
    task: Optional[str],
    raw_task: Optional[str],
    run_id: Optional[str],
    acq_id: Optional[str],
) -> bool:
    """Return True when a rule matches the current grouping."""
    match = rule.get("match", {})
    if not isinstance(match, Mapping):
        return True
    checks = {
        "subject": sub_id,
        "session": ses_id,
        "task": task,
        "raw_task": raw_task,
        "run": run_id,
        "acq": acq_id,
    }
    for key, actual in checks.items():
        expected = match.get(key)
        if expected is None:
            continue
        if isinstance(expected, (list, tuple, set)):
            targets = {str(v).lower() for v in expected}
            if str(actual).lower() not in targets:
                return False
        elif str(actual).lower() != str(expected).lower():
            return False
    return True


def _evaluate_rule_events_tsv(
    *,
    source: Mapping[str, Any],
    dataset_root: Path,
    sub_id: str,
    task: Optional[str],
    raw_task: Optional[str],
    midpoints: np.ndarray,
    meta: Dict[str, Any],
) -> tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Label windows from a BIDS events.tsv using last-value-carried-forward.

    Searches for the events file automatically when no explicit ``path`` is
    given, trying BIDS sub-{sub_id}/eeg and sub-{sub_id}/ directories.
    The ``trial_type`` column (configurable) is mapped to state labels via
    LVCF: each event's label extends until the onset of the next event.
    An optional ``label_set`` whitelist restricts which trial_type values
    are propagated; others emit ``None`` (unlabeled).
    """
    events_df = _load_bids_events_tsv(
        source=source,
        dataset_root=dataset_root,
        sub_id=sub_id,
        task=task,
        raw_task=raw_task,
    )
    if events_df is None or events_df.empty:
        logger.debug(
            "events_tsv: no events file found for sub-%s task=%s", sub_id, task or raw_task
        )
        return None, meta

    onset_col = str(source.get("onset_column", "onset") or "onset")
    trial_type_col = str(source.get("trial_type_column", "trial_type") or "trial_type")
    label_set_raw = source.get("label_set")
    label_set: Optional[set] = (
        {str(v).strip() for v in label_set_raw if str(v).strip()}
        if isinstance(label_set_raw, (list, tuple, set))
        else None
    )
    reset_on_raw = source.get("reset_on_labels")
    reset_on: set = (
        {str(v).strip() for v in reset_on_raw if str(v).strip()}
        if isinstance(reset_on_raw, (list, tuple, set))
        else set()
    )

    if onset_col not in events_df.columns:
        logger.warning("events_tsv: onset column %r not found", onset_col)
        return None, meta
    if trial_type_col not in events_df.columns:
        logger.warning("events_tsv: trial_type column %r not found", trial_type_col)
        return None, meta

    onset = pd.to_numeric(events_df[onset_col], errors="coerce").to_numpy(dtype=float)
    raw_labels = events_df[trial_type_col].astype(str).fillna("").tolist()
    n_events = len(onset)

    out = np.full((midpoints.shape[0],), None, dtype=object)
    current_label: Optional[str] = None

    for idx in range(n_events):
        if not np.isfinite(onset[idx]):
            continue
        ev_label = raw_labels[idx].strip() if idx < len(raw_labels) else ""
        if not ev_label or ev_label.lower() in {"nan", "n/a", "boundary"}:
            continue

        if ev_label in reset_on:
            current_label = None
            continue

        if label_set is not None and ev_label not in label_set:
            current_label = None
            continue

        current_label = ev_label
        start_sec = float(onset[idx])
        end_sec = float(onset[idx + 1]) if (
            idx + 1 < n_events and np.isfinite(onset[idx + 1])
        ) else np.inf
        mask = _segment_mask(midpoints, start_sec, end_sec)
        out[mask] = current_label

    n_assigned = int(np.sum(out != None))  # noqa: E711
    meta.update({
        "source": "events_tsv",
        "trial_type_column": trial_type_col,
        "n_events": n_events,
        "n_assigned_windows": n_assigned,
    })
    return out, meta


def _load_bids_events_tsv(
    *,
    source: Mapping[str, Any],
    dataset_root: Path,
    sub_id: str,
    task: Optional[str],
    raw_task: Optional[str],
) -> Optional[pd.DataFrame]:
    """Locate and load a BIDS events TSV, with auto-resolution from subject/task."""
    path_value = source.get("path")
    if isinstance(path_value, (str, Path)) and str(path_value).strip():
        raw_path = Path(str(path_value).strip())
        path = raw_path if raw_path.is_absolute() else (dataset_root / raw_path).resolve()
        if path.exists():
            try:
                return pd.read_csv(path, sep="\t")
            except Exception:
                logger.exception("events_tsv: failed reading %s", path)
                return None

    # Auto-resolve BIDS path: sub-{sub_id}/{datatype}/sub-{sub_id}_task-{task}_events.tsv
    candidate_tasks = [t for t in [raw_task, task] if t]
    # Normalise: "verbal_wm" → "verbalwm", strip underscores/spaces
    def _normalise(t: str) -> str:
        return re.sub(r"[_\s-]", "", t.lower())

    expanded_tasks: List[str] = []
    for t in candidate_tasks:
        expanded_tasks.append(t)
        n = _normalise(t)
        if n != t:
            expanded_tasks.append(n)

    sub_prefix = sub_id if sub_id.startswith("sub-") else f"sub-{sub_id}"
    for t in expanded_tasks:
        for subdir in ["eeg", "ieeg", "meg", "."]:
            p = dataset_root / sub_prefix / subdir / f"{sub_prefix}_task-{t}_events.tsv"
            if p.exists():
                try:
                    return pd.read_csv(p, sep="\t")
                except Exception:
                    logger.exception("events_tsv: failed reading %s", p)
    logger.debug(
        "events_tsv: events file not found for sub=%s task=%s in %s",
        sub_id, candidate_tasks, dataset_root,
    )
    return None


def _evaluate_rule(
    *,
    rule: Mapping[str, Any],
    dataset_root: Path,
    sub_id: str,
    task: Optional[str],
    raw_task: Optional[str],
    tr_sec: Optional[float],
    midpoints: np.ndarray,
    sub_frame: pd.DataFrame,
) -> tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Evaluate one within-run rule to a per-window label series."""
    source = rule.get("source", {})
    if not isinstance(source, Mapping):
        return None, {}
    source_type = str(source.get("type", "")).strip().lower()
    rule_id = str(rule.get("id") or source_type or "within_run_rule")
    meta: Dict[str, Any] = {"id": rule_id, "source_type": source_type}

    if source_type == "events_tsv":
        return _evaluate_rule_events_tsv(
            source=source,
            dataset_root=dataset_root,
            sub_id=sub_id,
            task=task,
            raw_task=raw_task,
            midpoints=midpoints,
            meta=meta,
        )

    if source_type in {"column_from_features", "feature_column"}:
        column = str(source.get("column", "")).strip()
        if not column or column not in sub_frame.columns:
            return None, meta
        series = [
            _normalize_label_value(value)
            for value in sub_frame[column].tolist()
        ]
        meta["column"] = column
        meta["source"] = "features"
        return np.asarray(series, dtype=object), meta

    if source_type == "interval_table":
        table = _load_table(source, dataset_root)
        if table is None or table.empty:
            return None, meta
        row_subject_col = str(source.get("subject_column", "subject"))
        intervals = table.loc[
            table[row_subject_col].astype(str).str.strip().str.lower() == str(sub_id).lower()
        ].copy()
        if intervals.empty:
            return None, meta
        label_col = str(source.get("label_column", "label"))
        start_col = str(source.get("start_column", "start"))
        end_col = str(source.get("end_column", "end"))
        units = str(source.get("units", "seconds")).strip().lower() or "seconds"
        index_origin = int(source.get("index_origin", 0) or 0)
        out = np.full((midpoints.shape[0],), None, dtype=object)
        for _, row in intervals.iterrows():
            label_value = _normalize_label_value(row.get(label_col))
            if label_value is None:
                continue
            start_sec = _convert_unit_value(row.get(start_col), units=units, tr_sec=tr_sec, index_origin=index_origin)
            end_sec = _convert_unit_value(row.get(end_col), units=units, tr_sec=tr_sec, index_origin=index_origin)
            if start_sec is None and end_sec is None:
                continue
            mask = _segment_mask(midpoints, start_sec, end_sec)
            out[mask] = label_value
        meta.update({"path": str(source.get("path", "")), "units": units, "source": "interval_table"})
        return out, meta

    if source_type == "boundary_table":
        table = _load_table(source, dataset_root)
        if table is None or table.empty:
            return None, meta
        subject_column = str(source.get("subject_column", "subject"))
        rows = table.loc[
            table[subject_column].astype(str).str.strip().str.lower() == str(sub_id).lower()
        ]
        if rows.empty:
            return None, meta
        row = rows.iloc[0]
        boundaries = {}
        boundary_specs = source.get("boundaries", {})
        if not isinstance(boundary_specs, Mapping):
            return None, meta
        for name, spec in boundary_specs.items():
            if not isinstance(spec, Mapping):
                continue
            units = str(spec.get("units", "seconds")).strip().lower() or "seconds"
            index_origin = int(spec.get("index_origin", 0) or 0)
            boundaries[str(name)] = _convert_unit_value(
                row.get(spec.get("column")),
                units=units,
                tr_sec=tr_sec,
                index_origin=index_origin,
                na_values=spec.get("na_values"),
            )
        segments = rule.get("segments", [])
        if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
            return None, meta
        out = np.full((midpoints.shape[0],), None, dtype=object)
        for segment in segments:
            if not isinstance(segment, Mapping):
                continue
            label_value = _normalize_label_value(segment.get("label"))
            if label_value is None:
                continue
            start_sec, start_missing = _resolve_segment_endpoint(
                ref=segment.get("start"),
                boundaries=boundaries,
            )
            end_sec, end_missing = _resolve_segment_endpoint(
                ref=segment.get("end"),
                boundaries=boundaries,
            )
            missing_policy = str(segment.get("on_missing_boundary", "expand")).strip().lower() or "expand"
            if missing_policy == "skip" and (start_missing or end_missing):
                continue
            mask = _segment_mask(midpoints, start_sec, end_sec)
            out[mask] = label_value
        meta.update(
            {
                "path": str(source.get("path", "")),
                "source": "boundary_table",
                "boundaries": boundaries,
            }
        )
        return out, meta

    logger.warning("Unsupported within-run source type '%s'", source_type)
    return None, meta


def _compute_midpoints(
    *,
    time: np.ndarray,
    window_start: np.ndarray,
    window_end: np.ndarray,
) -> np.ndarray:
    """Compute per-window midpoints."""
    if window_start.shape == window_end.shape == time.shape:
        return ((window_start + window_end) / 2.0).astype(np.float64, copy=False)
    return np.asarray(time, dtype=np.float64)


def _segment_mask(midpoints: np.ndarray, start_sec: Optional[float], end_sec: Optional[float]) -> np.ndarray:
    """Return mask for a half-open interval on midpoint timestamps."""
    mask = np.ones((midpoints.shape[0],), dtype=bool)
    if start_sec is not None and math.isfinite(start_sec):
        mask &= midpoints >= float(start_sec)
    if end_sec is not None and math.isfinite(end_sec):
        mask &= midpoints < float(end_sec)
    return mask


def _resolve_segment_endpoint(
    *,
    ref: Any,
    boundaries: Mapping[str, Optional[float]],
) -> tuple[Optional[float], bool]:
    """Resolve a segment endpoint reference to seconds."""
    if ref is None:
        return None, False
    if isinstance(ref, (int, float)):
        value = float(ref)
        return (value if math.isfinite(value) else None), not math.isfinite(value)
    key = str(ref).strip()
    if not key:
        return None, False
    value = boundaries.get(key)
    return value, value is None


def _normalize_label_value(value: Any) -> Optional[Any]:
    """Normalize raw label values to None or a scalar."""
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            decoded = value.decode("utf-8")
        except Exception:
            decoded = value.decode("utf-8", errors="ignore")
        stripped = decoded.strip()
        return stripped or None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if pd.isna(value):
        return None
    return str(value)


def _encode_stage_series(
    values: np.ndarray,
    codebook: Optional[Mapping[str, int]],
) -> Optional[np.ndarray]:
    """Encode object labels to an int8 stage series."""
    if values.size == 0:
        return None
    out = np.full((values.shape[0],), -1, dtype=np.int16)
    mapping = {str(k).strip().lower(): int(v) for k, v in (codebook or {}).items()}
    for idx, value in enumerate(values):
        if value is None:
            continue
        if isinstance(value, (np.integer, int)):
            out[idx] = int(value)
            continue
        if isinstance(value, (np.floating, float)):
            if math.isfinite(float(value)):
                out[idx] = int(value)
            continue
        mapped = mapping.get(str(value).strip().lower())
        if mapped is not None:
            out[idx] = int(mapped)
    if not np.any(out != -1):
        return None
    return out.astype(np.int8, copy=False)


def _resolve_codebook(
    *,
    dataset_id: str,
    cfg: Mapping[str, Any],
) -> Optional[Dict[str, int]]:
    """Resolve codebook from inline config or dataset codebook YAML."""
    inline = cfg.get("codebook")
    if isinstance(inline, Mapping) and inline:
        return {str(k): int(v) for k, v in inline.items()}

    codebook_name = cfg.get("codebook_name")
    if not isinstance(codebook_name, str) or not codebook_name.strip():
        return None

    codebook_path = Path(__file__).resolve().parents[3] / "config" / "codebook" / f"{dataset_id}.yaml"
    if not codebook_path.exists():
        logger.warning("Within-run codebook file not found for %s: %s", dataset_id, codebook_path)
        return None
    try:
        import yaml  # type: ignore

        parsed = yaml.safe_load(codebook_path.read_text(encoding="utf-8")) or {}
        codes = parsed.get("codes", {}) if isinstance(parsed, Mapping) else {}
        entry = codes.get(codebook_name, {}) if isinstance(codes, Mapping) else {}
        if isinstance(entry, Mapping) and entry:
            return {str(k): int(v) for k, v in entry.items()}
    except Exception:
        logger.exception("Failed loading within-run codebook '%s' for %s", codebook_name, dataset_id)
    return None


@lru_cache(maxsize=32)
def _read_table_cached(path_str: str, sep_value: Optional[str]) -> pd.DataFrame:
    """Read and cache external label tables."""
    path = Path(path_str)
    read_kwargs: Dict[str, Any] = {}
    if sep_value:
        read_kwargs["sep"] = sep_value
    else:
        suffix = path.suffix.lower()
        if suffix == ".tsv":
            read_kwargs["sep"] = "\t"
        elif suffix == ".csv":
            read_kwargs["sep"] = ","
        else:
            read_kwargs.update({"sep": None, "engine": "python"})
    return pd.read_csv(path, **read_kwargs)


def _load_table(source: Mapping[str, Any], dataset_root: Path) -> Optional[pd.DataFrame]:
    """Load an external table referenced by a rule source."""
    path_value = source.get("path")
    if not isinstance(path_value, (str, Path)) or not str(path_value).strip():
        return None
    raw_path = Path(str(path_value).strip())
    path = raw_path if raw_path.is_absolute() else (dataset_root / raw_path).resolve()
    if not path.exists():
        logger.warning("Within-run label table not found: %s", path)
        return None
    sep = source.get("sep")
    try:
        return _read_table_cached(str(path), str(sep) if isinstance(sep, str) and sep else None).copy()
    except Exception:
        logger.exception("Failed reading within-run label table %s", path)
        return None


def _convert_unit_value(
    raw_value: Any,
    *,
    units: str,
    tr_sec: Optional[float],
    index_origin: int = 0,
    na_values: Any = None,
) -> Optional[float]:
    """Convert raw values from seconds/TR/sample to seconds."""
    if raw_value is None or pd.isna(raw_value):
        return None
    na_tokens = {"", "na", "n/a", "nan", "none"}
    if isinstance(na_values, Sequence) and not isinstance(na_values, (str, bytes)):
        na_tokens |= {str(v).strip().lower() for v in na_values}
    text = str(raw_value).strip()
    if text.lower() in na_tokens:
        return None
    try:
        value = float(text)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    units = units.strip().lower()
    if units in {"seconds", "sec", "s"}:
        return value
    if units in {"tr", "tr_index", "tr_indices"}:
        if tr_sec is None or not math.isfinite(float(tr_sec)) or float(tr_sec) <= 0:
            return None
        return max(0.0, value - float(index_origin)) * float(tr_sec)
    if units in {"sample", "samples"}:
        sample_rate_hz = None
        if isinstance(na_values, Mapping):
            sample_rate_hz = na_values.get("sample_rate_hz")
        try:
            rate = float(sample_rate_hz)
        except Exception:
            return None
        if not math.isfinite(rate) or rate <= 0:
            return None
        return max(0.0, value - float(index_origin)) / rate
    return value


def summarize_within_run_manifest(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a compact, JSON-stable manifest fragment."""
    compact: Dict[str, Any] = {}
    for key in (
        "enabled",
        "output_name",
        "write_to_stage",
        "write_to_labels",
        "codebook_name",
        "assigned_frac",
        "conflict_count",
    ):
        if key in manifest:
            compact[key] = manifest.get(key)
    matched = manifest.get("matched_rules")
    if isinstance(matched, Sequence) and not isinstance(matched, (str, bytes)):
        rows = []
        for row in matched:
            if not isinstance(row, Mapping):
                continue
            slim = {
                "id": row.get("id"),
                "source_type": row.get("source_type"),
                "column": row.get("column"),
                "filled_windows": row.get("filled_windows"),
            }
            rows.append({k: v for k, v in slim.items() if v is not None})
        compact["matched_rules"] = rows
    if manifest.get("stage_codebook"):
        compact["stage_codebook"] = manifest.get("stage_codebook")
    return json.loads(json.dumps(compact))


def build_label_segment_event_table(
    *,
    labels: Sequence[Any],
    window_start: np.ndarray,
    window_end: np.ndarray,
    source_path: str = "derived:within_run_labels",
    source_label: str = "derived:within_run_labels",
) -> "EventTable":
    """Convert a per-window label series into contiguous segment events."""
    from .event_annotations import EventTable

    label_arr = np.asarray(labels, dtype=object)
    start_arr = np.asarray(window_start, dtype=np.float64)
    end_arr = np.asarray(window_end, dtype=np.float64)
    n = min(label_arr.shape[0], start_arr.shape[0], end_arr.shape[0])
    if n <= 0:
        return EventTable(
            onset_sec=np.zeros((0,), dtype=np.float64),
            event_type=np.zeros((0,), dtype=object),
            source=np.zeros((0,), dtype=object),
            source_path=str(source_path),
            n_events_loaded=0,
        )

    events_onset: List[float] = []
    events_duration: List[float] = []
    events_peak: List[float] = []
    events_type: List[str] = []
    events_stage: List[str] = []
    events_source: List[str] = []

    current_label: Optional[str] = None
    seg_start = -1
    seg_end = -1

    def _flush_segment() -> None:
        if current_label is None or seg_start < 0 or seg_end < seg_start:
            return
        onset = float(start_arr[seg_start])
        offset = float(end_arr[seg_end])
        if not np.isfinite(onset):
            return
        if not np.isfinite(offset) or offset < onset:
            offset = onset
        duration = max(0.0, offset - onset)
        peak = onset + 0.5 * duration
        events_onset.append(onset)
        events_duration.append(duration)
        events_peak.append(peak)
        events_type.append(current_label)
        events_stage.append(current_label)
        events_source.append(str(source_label))

    for idx in range(n):
        raw = label_arr[idx]
        label_value = _normalize_label_value(raw)
        label = str(label_value) if label_value is not None else None
        if label is None:
            _flush_segment()
            current_label = None
            seg_start = -1
            seg_end = -1
            continue
        if current_label is None:
            current_label = label
            seg_start = idx
            seg_end = idx
            continue
        if label == current_label and idx == seg_end + 1:
            seg_end = idx
            continue
        _flush_segment()
        current_label = label
        seg_start = idx
        seg_end = idx
    _flush_segment()

    onsets = np.asarray(events_onset, dtype=np.float64)
    durations = np.asarray(events_duration, dtype=np.float64)
    peaks = np.asarray(events_peak, dtype=np.float64)
    event_types = np.asarray(events_type, dtype=object)
    sources = np.asarray(events_source, dtype=object)
    stages = np.asarray(events_stage, dtype=object)
    return EventTable(
        onset_sec=onsets,
        duration_sec=durations,
        peak_sec=peaks,
        event_type=event_types,
        source=sources,
        stage=stages,
        source_path=str(source_path),
        n_events_loaded=int(onsets.shape[0]),
    )
