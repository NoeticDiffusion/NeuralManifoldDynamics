"""Time-reference extraction and alignment helpers for summary outputs."""

from __future__ import annotations

import logging
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_SCHEMA_VERSION = "time_reference.v1"
_DEFAULT_BINS_HOURS: tuple[float, ...] = (0.0, 24.0, 48.0, 72.0)
_SECONDS_PER_DAY = 24.0 * 3600.0
_CLOCK_RE = re.compile(
    r"^\s*(?P<hours>\d{1,2}):(?P<minutes>\d{2})(?::(?P<seconds>\d{2}(?:\.\d+)?))?\s*(?P<plus>\+*)\s*$"
)
_WFDB_START_RE = re.compile(r"^#\s*start\s*time\s*:\s*(?P<value>.+?)\s*$", re.IGNORECASE)
_WFDB_END_RE = re.compile(r"^#\s*end\s*time\s*:\s*(?P<value>.+?)\s*$", re.IGNORECASE)


def _deep_merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge two mappings."""
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge_dict(existing, value)
        else:
            merged[key] = value
    return merged


def _normalize_token(value: Any, prefixes: Sequence[str] = ()) -> Optional[str]:
    """Normalize run/acq-like tokens for matching."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    low = text.lower()
    for pref in prefixes:
        if low.startswith(pref):
            text = text[len(pref) :]
            low = text.lower()
            break
    return text or None


def _normalize_subject_candidates(sub_id: str) -> list[str]:
    """Return subject-id variants for robust matching."""
    value = str(sub_id or "").strip()
    if not value:
        return []
    out: list[str] = []
    out.append(value)
    if value.lower().startswith("sub-"):
        bare = value[4:]
        if bare:
            out.append(bare)
    else:
        bare = value
        out.append(f"sub-{bare}")
    if bare.isdigit():
        stripped = bare.lstrip("0") or "0"
        for width in (3, 4, len(bare), len(stripped)):
            if width <= 0:
                continue
            padded = stripped.zfill(width)
            out.append(padded)
            out.append(f"sub-{padded}")
    deduped: list[str] = []
    seen: set[str] = set()
    for item in out:
        key = str(item).strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def _normalize_bins_hours(raw_bins: Any) -> list[float]:
    """Normalize configured hour-bin boundaries."""
    if not isinstance(raw_bins, Sequence) or isinstance(raw_bins, (str, bytes)):
        return list(_DEFAULT_BINS_HOURS)
    parsed: list[float] = []
    for value in raw_bins:
        try:
            hour = float(value)
        except Exception:
            continue
        if not np.isfinite(hour):
            continue
        parsed.append(hour)
    if not parsed:
        return list(_DEFAULT_BINS_HOURS)
    parsed = sorted(set(parsed))
    if parsed[0] > 0.0:
        parsed.insert(0, 0.0)
    return parsed


def resolve_time_reference_config(config: Optional[Mapping[str, Any]], dataset_id: str) -> Dict[str, Any]:
    """Resolve global + dataset-specific time-reference config."""
    time_cfg = config.get("time_reference", {}) if isinstance(config, Mapping) else {}
    if not isinstance(time_cfg, Mapping):
        return {
            "enabled": False,
            "schema_version": _DEFAULT_SCHEMA_VERSION,
            "parser": "wfdb_header",
            "anchor": "first_recording",
            "bins_hours": list(_DEFAULT_BINS_HOURS),
        }

    base_cfg: Dict[str, Any] = {k: v for k, v in time_cfg.items() if k != "datasets"}
    ds_map = time_cfg.get("datasets", {})
    ds_override = ds_map.get(dataset_id, {}) if isinstance(ds_map, Mapping) else {}
    merged = _deep_merge_dict(base_cfg, ds_override) if isinstance(ds_override, Mapping) else dict(base_cfg)

    parser = str(merged.get("parser", "wfdb_header")).strip().lower() or "wfdb_header"
    anchor = str(merged.get("anchor", "first_recording")).strip().lower() or "first_recording"
    out: Dict[str, Any] = {
        "enabled": bool(merged.get("enabled", False)),
        "schema_version": str(merged.get("schema_version", _DEFAULT_SCHEMA_VERSION)),
        "parser": parser,
        "anchor": anchor,
        "bins_hours": _normalize_bins_hours(merged.get("bins_hours", list(_DEFAULT_BINS_HOURS))),
    }
    wfdb_cfg = merged.get("wfdb_header", {})
    if isinstance(wfdb_cfg, Mapping):
        out["wfdb_header"] = dict(wfdb_cfg)
    return out


def parse_wfdb_clock_value(raw_value: str) -> Optional[float]:
    """Parse WFDB clock value (e.g. ``22:59:08`` or ``24:00:00+``) into seconds."""
    text = str(raw_value or "").strip()
    if not text:
        return None
    match = _CLOCK_RE.match(text)
    if match is None:
        return None
    try:
        hours = int(match.group("hours"))
        minutes = int(match.group("minutes"))
        seconds = float(match.group("seconds") or "0")
    except Exception:
        return None
    if minutes < 0 or minutes >= 60:
        return None
    if seconds < 0 or seconds >= 60:
        return None
    total = float(hours * 3600 + minutes * 60 + seconds)
    plus_count = len(match.group("plus") or "")
    if plus_count > 0:
        # ``24:00:00+`` should map to exactly one-day rollover (86400 sec),
        # while values like ``00:10:00+`` add one full day.
        day_rollovers = plus_count if hours < 24 else max(0, plus_count - 1)
        total += float(day_rollovers) * _SECONDS_PER_DAY
    return total


def parse_wfdb_header_clocks(header_path: Path) -> Dict[str, Any]:
    """Parse ``#Start time`` / ``#End time`` from a WFDB header."""
    path = Path(header_path)
    out: Dict[str, Any] = {
        "status": "header_missing",
        "header_path": str(path),
        "run_start_clock_raw": None,
        "run_end_clock_raw": None,
        "run_start_clock_sec": None,
        "run_end_clock_sec": None,
        "run_duration_sec": None,
        "parse_errors": [],
    }
    if not path.exists():
        return out

    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception as exc:
        out["status"] = "header_read_error"
        out["parse_errors"] = [str(exc)]
        return out

    start_raw: Optional[str] = None
    end_raw: Optional[str] = None
    for line in lines:
        if start_raw is None:
            match_start = _WFDB_START_RE.match(line.strip())
            if match_start is not None:
                start_raw = str(match_start.group("value")).strip()
                continue
        if end_raw is None:
            match_end = _WFDB_END_RE.match(line.strip())
            if match_end is not None:
                end_raw = str(match_end.group("value")).strip()
        if start_raw is not None and end_raw is not None:
            break

    out["run_start_clock_raw"] = start_raw
    out["run_end_clock_raw"] = end_raw

    parse_errors: list[str] = []
    start_sec = parse_wfdb_clock_value(start_raw or "")
    end_sec = parse_wfdb_clock_value(end_raw or "")

    if start_raw is None:
        parse_errors.append("missing_start_time")
    elif start_sec is None:
        parse_errors.append("invalid_start_time")
    if end_raw is None:
        parse_errors.append("missing_end_time")
    elif end_sec is None:
        parse_errors.append("invalid_end_time")

    if start_sec is None and end_sec is None:
        out["status"] = "missing_or_invalid_start_end"
        out["parse_errors"] = parse_errors
        return out
    if start_sec is None:
        out["status"] = "missing_or_invalid_start"
        out["run_end_clock_sec"] = float(end_sec) if end_sec is not None else None
        out["parse_errors"] = parse_errors
        return out

    out["run_start_clock_sec"] = float(start_sec)
    if end_sec is not None:
        end_aligned = float(end_sec)
        while end_aligned < float(start_sec):
            end_aligned += _SECONDS_PER_DAY
        out["run_end_clock_sec"] = end_aligned
        out["run_duration_sec"] = float(end_aligned - float(start_sec))
    out["parse_errors"] = parse_errors
    out["status"] = "ok" if not parse_errors else "ok_with_warnings"
    return out


@lru_cache(maxsize=8192)
def _parse_wfdb_header_clocks_cached(path_str: str, mtime_ns: int, size_bytes: int) -> Dict[str, Any]:
    """Cached wrapper around WFDB header parsing."""
    del mtime_ns, size_bytes
    return parse_wfdb_header_clocks(Path(path_str))


def _parse_wfdb_header_with_cache(path: Path) -> Dict[str, Any]:
    """Parse WFDB header with mtime/size cache key."""
    stat = path.stat()
    parsed = _parse_wfdb_header_clocks_cached(str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
    return dict(parsed)


def _build_bin_vectors(
    window_start_from_anchor_sec: np.ndarray,
    bins_hours: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign window-level bins from anchor-relative start times."""
    bin_edges_hours = np.asarray(list(bins_hours), dtype=np.float32)
    bin_edges_sec = np.asarray(bin_edges_hours * np.float32(3600.0), dtype=np.float32)
    bin_ids = np.full(window_start_from_anchor_sec.shape, -1, dtype=np.int32)
    labels = np.full(window_start_from_anchor_sec.shape, "unassigned", dtype="<U64")
    if window_start_from_anchor_sec.size == 0:
        return bin_ids, labels, bin_edges_sec

    valid = np.isfinite(window_start_from_anchor_sec)
    if not valid.any():
        return bin_ids, labels, bin_edges_sec

    if len(bin_edges_sec) == 0:
        return bin_ids, labels, bin_edges_sec

    for idx in range(len(bin_edges_sec) - 1):
        lo = float(bin_edges_sec[idx])
        hi = float(bin_edges_sec[idx + 1])
        mask = valid & (window_start_from_anchor_sec >= lo) & (window_start_from_anchor_sec < hi)
        if mask.any():
            bin_ids[mask] = int(idx)
            labels[mask] = f"{float(bin_edges_hours[idx]):g}-{float(bin_edges_hours[idx + 1]):g}h"
    last_idx = int(len(bin_edges_sec) - 1)
    last_mask = valid & (window_start_from_anchor_sec >= float(bin_edges_sec[last_idx]))
    if last_mask.any():
        bin_ids[last_mask] = last_idx
        labels[last_mask] = f"{float(bin_edges_hours[last_idx]):g}h_plus"

    before_mask = valid & (window_start_from_anchor_sec < float(bin_edges_sec[0]))
    if before_mask.any():
        labels[before_mask] = f"before_{float(bin_edges_hours[0]):g}h"
    return bin_ids, labels, bin_edges_sec


def _extract_run_acq_from_name(path_like: str) -> tuple[Optional[str], Optional[str]]:
    """Extract run/acq tokens from non-BIDS I-CARE style filenames."""
    name = Path(str(path_like)).name
    stem = name[:-4] if name.lower().endswith(".hea") else name
    parts = stem.split("_")
    if len(parts) < 4:
        return None, None
    run = _normalize_token(parts[1], prefixes=("run-",))
    acq = _normalize_token(parts[2], prefixes=("acq-",))
    return run, acq


def _collect_subject_header_rows(
    *,
    dataset_root: Path,
    index_df: Optional[pd.DataFrame],
    subject_candidates: Sequence[str],
) -> list[dict[str, Any]]:
    """Collect subject-scoped WFDB header rows from file index."""
    if index_df is None or index_df.empty or "path" not in index_df.columns:
        return []
    rows = index_df.copy()
    if "modality" in rows.columns:
        rows = rows[rows["modality"].astype(str).str.lower() == "eeg"]
    rows = rows[rows["path"].astype(str).str.lower().str.endswith(".hea")]
    if rows.empty:
        return []
    if "subject" in rows.columns and subject_candidates:
        subj_set = set(subject_candidates)
        rows = rows[rows["subject"].astype(str).isin(subj_set)]
    out: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        rel = str(row.get("path", "")).replace("\\", "/")
        if not rel:
            continue
        abs_path = dataset_root / rel
        if not abs_path.exists():
            continue
        run_norm = _normalize_token(row.get("run"), prefixes=("run-",))
        acq_norm = _normalize_token(row.get("acq"), prefixes=("acq-",))
        if run_norm is None or acq_norm is None:
            run_from_name, acq_from_name = _extract_run_acq_from_name(rel)
            run_norm = run_norm or run_from_name
            acq_norm = acq_norm or acq_from_name
        out.append(
            {
                "path": abs_path,
                "rel_path": rel,
                "run": run_norm,
                "acq": acq_norm,
            }
        )
    return out


def _resolve_header_from_features(
    *,
    dataset_root: Path,
    sub_frame: pd.DataFrame,
    representative_file: Optional[str],
    lookup_rel_paths_by_file_value: Callable[[str], List[str]],
) -> Optional[Path]:
    """Resolve current run header path from feature-file values + index cache."""
    values: list[str] = []
    if representative_file:
        values.append(str(representative_file))
    if "file" in sub_frame.columns:
        values.extend([str(v) for v in sub_frame["file"].dropna().astype(str).unique().tolist()])

    seen: set[str] = set()
    for value in values:
        key = value.strip()
        if not key or key in seen:
            continue
        seen.add(key)

        try:
            abs_candidate = Path(key)
            if abs_candidate.is_absolute() and abs_candidate.exists() and abs_candidate.suffix.lower() == ".hea":
                return abs_candidate
        except Exception:
            pass

        rel_norm = key.replace("\\", "/")
        if rel_norm.lower().endswith(".hea"):
            rel_candidate = dataset_root / rel_norm
            if rel_candidate.exists():
                return rel_candidate

        rel_paths = []
        try:
            rel_paths = list(lookup_rel_paths_by_file_value(key))
        except Exception:
            rel_paths = []
        for rel in rel_paths:
            rel_norm = str(rel).replace("\\", "/")
            if not rel_norm.lower().endswith(".hea"):
                continue
            candidate = dataset_root / rel_norm
            if candidate.exists():
                return candidate
    return None


def _resolve_current_header_row(
    *,
    dataset_root: Path,
    index_df: Optional[pd.DataFrame],
    subject_candidates: Sequence[str],
    run_id: Optional[str],
    acq_id: Optional[str],
    current_header_path: Optional[Path],
) -> Optional[dict[str, Any]]:
    """Resolve the current run row from index metadata."""
    if current_header_path is None:
        return None
    current_rel: Optional[str] = None
    try:
        current_rel = str(current_header_path.relative_to(dataset_root)).replace("\\", "/")
    except Exception:
        current_rel = None

    rows = _collect_subject_header_rows(
        dataset_root=dataset_root,
        index_df=index_df,
        subject_candidates=subject_candidates,
    )
    if not rows:
        run_from_name, acq_from_name = _extract_run_acq_from_name(current_header_path.name)
        return {
            "path": current_header_path,
            "rel_path": current_rel or str(current_header_path),
            "run": run_from_name,
            "acq": acq_from_name,
        }

    if current_rel:
        for row in rows:
            if str(row.get("rel_path", "")) == current_rel:
                return row

    run_norm = _normalize_token(run_id, prefixes=("run-",))
    acq_norm = _normalize_token(acq_id, prefixes=("acq-",))
    filtered = rows
    if run_norm is not None:
        run_filtered = [r for r in filtered if _normalize_token(r.get("run"), prefixes=("run-",)) == run_norm]
        if run_filtered:
            filtered = run_filtered
    if acq_norm is not None:
        acq_filtered = [r for r in filtered if _normalize_token(r.get("acq"), prefixes=("acq-",)) == acq_norm]
        if acq_filtered:
            filtered = acq_filtered

    if filtered:
        return filtered[0]
    for row in rows:
        if Path(str(row.get("path", ""))).name == current_header_path.name:
            return row
    return {
        "path": current_header_path,
        "rel_path": current_rel or str(current_header_path),
        "run": run_norm,
        "acq": acq_norm,
    }


def _as_sort_int(token: Optional[str]) -> tuple[int, str]:
    """Convert run/acq tokens into stable sort tuple."""
    if token is None:
        return (10**9, "")
    text = str(token).strip()
    if not text:
        return (10**9, "")
    if text.isdigit():
        return (int(text), text)
    return (10**9, text)


def _unwrap_subject_timeline(entries: list[dict[str, Any]]) -> tuple[Dict[str, tuple[float, Optional[float]]], Optional[float]]:
    """Unwrap daily clock times into monotonic elapsed seconds."""
    if not entries:
        return {}, None
    sorted_entries = sorted(
        entries,
        key=lambda row: (
            _as_sort_int(_normalize_token(row.get("run"), prefixes=("run-",))),
            _as_sort_int(_normalize_token(row.get("acq"), prefixes=("acq-",))),
            str(row.get("rel_path", row.get("path", ""))),
        ),
    )
    out: Dict[str, tuple[float, Optional[float]]] = {}
    day_offset = 0.0
    prev_start: Optional[float] = None
    for row in sorted_entries:
        start_clock = row.get("run_start_clock_sec")
        if start_clock is None:
            continue
        start_val = float(start_clock)
        candidate_start = start_val + day_offset
        while prev_start is not None and candidate_start < prev_start - 1e-6:
            day_offset += _SECONDS_PER_DAY
            candidate_start = start_val + day_offset
        end_clock = row.get("run_end_clock_sec")
        candidate_end: Optional[float] = None
        if end_clock is not None:
            candidate_end = float(end_clock) + day_offset
            while candidate_end < candidate_start - 1e-6:
                candidate_end += _SECONDS_PER_DAY
        path_key = str(row.get("path"))
        out[path_key] = (candidate_start, candidate_end)
        prev_start = candidate_start
    if not out:
        return {}, None
    anchor_start = min(v[0] for v in out.values())
    return out, float(anchor_start)


def _build_disabled_result(status: str, schema_version: str, source: str, anchor: str) -> Dict[str, Any]:
    """Construct a no-op result for disabled/unsupported states."""
    attrs = {
        "time_reference_schema_version": schema_version,
        "time_reference_source": source,
        "time_reference_status": status,
        "time_reference_anchor_mode": anchor,
        "time_reference_enabled": False,
    }
    return {
        "enabled": False,
        "status": status,
        "attrs": attrs,
        "manifest": {"enabled": False, "status": status, "source": source, "anchor": anchor, "schema_version": schema_version},
        "extension": None,
    }


def build_time_reference_for_run(
    *,
    config: Mapping[str, Any],
    dataset_id: str,
    dataset_root: Path,
    index_df: Optional[pd.DataFrame],
    lookup_rel_paths_by_file_value: Callable[[str], List[str]],
    sub_id: str,
    run_id: Optional[str],
    acq_id: Optional[str],
    representative_file: Optional[str],
    sub_frame: pd.DataFrame,
    window_start: np.ndarray,
    window_end: np.ndarray,
) -> Dict[str, Any]:
    """Build time-reference extension/attrs/manifest payload for one run."""
    resolved_cfg = resolve_time_reference_config(config, dataset_id)
    schema_version = str(resolved_cfg.get("schema_version", _DEFAULT_SCHEMA_VERSION))
    parser = str(resolved_cfg.get("parser", "wfdb_header"))
    anchor_mode = str(resolved_cfg.get("anchor", "first_recording"))
    if not bool(resolved_cfg.get("enabled", False)):
        return _build_disabled_result("disabled", schema_version, parser, anchor_mode)
    if parser != "wfdb_header":
        return _build_disabled_result("unsupported_parser", schema_version, parser, anchor_mode)

    window_start_from_run_sec = np.asarray(window_start, dtype=np.float32)
    window_end_from_run_sec = np.asarray(window_end, dtype=np.float32)
    bins_hours = _normalize_bins_hours(resolved_cfg.get("bins_hours", list(_DEFAULT_BINS_HOURS)))

    current_header_path = _resolve_header_from_features(
        dataset_root=dataset_root,
        sub_frame=sub_frame,
        representative_file=representative_file,
        lookup_rel_paths_by_file_value=lookup_rel_paths_by_file_value,
    )
    if current_header_path is None:
        status = "header_not_found"
        window_start_from_anchor_sec = np.full(window_start_from_run_sec.shape, np.nan, dtype=np.float32)
        window_end_from_anchor_sec = np.full(window_end_from_run_sec.shape, np.nan, dtype=np.float32)
        window_bin_id, window_bin_label, bins_sec = _build_bin_vectors(window_start_from_anchor_sec, bins_hours)
        run_payload = {
            "schema_version": schema_version,
            "parser": parser,
            "status": status,
            "anchor_mode": anchor_mode,
            "header_path": "",
            "run_start_clock_raw": "",
            "run_end_clock_raw": "",
            "run_start_clock_sec": np.float32(np.nan),
            "run_end_clock_sec": np.float32(np.nan),
            "run_start_elapsed_sec": np.float32(np.nan),
            "run_end_elapsed_sec": np.float32(np.nan),
            "run_duration_sec": np.float32(np.nan),
            "subject_anchor_start_sec": np.float32(np.nan),
            "bins_hours": np.asarray(bins_hours, dtype=np.float32),
            "bins_seconds": bins_sec,
            "parse_errors": np.asarray(["header_not_found"], dtype="<U64"),
        }
        extension = {
            "run": run_payload,
            "windows": {
                "window_start_from_run_sec": window_start_from_run_sec,
                "window_end_from_run_sec": window_end_from_run_sec,
                "window_start_from_anchor_sec": window_start_from_anchor_sec,
                "window_end_from_anchor_sec": window_end_from_anchor_sec,
                "window_bin_id": window_bin_id,
                "window_bin_label": window_bin_label,
            },
        }
        attrs = {
            "time_reference_schema_version": schema_version,
            "time_reference_source": parser,
            "time_reference_status": status,
            "time_reference_anchor_mode": anchor_mode,
            "time_reference_enabled": True,
            "time_reference_header_path": "",
        }
        manifest = {
            "enabled": True,
            "schema_version": schema_version,
            "source": parser,
            "status": status,
            "anchor": anchor_mode,
            "header_path": "",
            "bins_hours": bins_hours,
        }
        return {
            "enabled": True,
            "status": status,
            "attrs": attrs,
            "manifest": manifest,
            "extension": extension,
        }

    current_parse = _parse_wfdb_header_with_cache(current_header_path)
    subject_candidates = _normalize_subject_candidates(sub_id)
    current_row = _resolve_current_header_row(
        dataset_root=dataset_root,
        index_df=index_df,
        subject_candidates=subject_candidates,
        run_id=run_id,
        acq_id=acq_id,
        current_header_path=current_header_path,
    )
    if current_row is None:
        current_row = {
            "path": current_header_path,
            "rel_path": str(current_header_path),
            "run": _normalize_token(run_id, prefixes=("run-",)),
            "acq": _normalize_token(acq_id, prefixes=("acq-",)),
        }

    subject_rows = _collect_subject_header_rows(
        dataset_root=dataset_root,
        index_df=index_df,
        subject_candidates=subject_candidates,
    )
    if not subject_rows:
        subject_rows = [current_row]

    parsed_entries: list[dict[str, Any]] = []
    for row in subject_rows:
        row_path = Path(str(row.get("path")))
        if not row_path.exists():
            continue
        parsed = _parse_wfdb_header_with_cache(row_path)
        merged = dict(row)
        merged.update(parsed)
        parsed_entries.append(merged)

    timeline_by_path, anchor_start_abs = _unwrap_subject_timeline(parsed_entries)
    current_start_abs: Optional[float] = None
    current_end_abs: Optional[float] = None
    cur_path_key = str(current_header_path)
    if cur_path_key in timeline_by_path:
        current_start_abs, current_end_abs = timeline_by_path[cur_path_key]
    elif current_parse.get("run_start_clock_sec") is not None:
        current_start_abs = float(current_parse["run_start_clock_sec"])
        end_clock = current_parse.get("run_end_clock_sec")
        current_end_abs = float(end_clock) if end_clock is not None else None

    if anchor_mode != "first_recording":
        anchor_start_abs = current_start_abs
    if current_start_abs is not None and anchor_start_abs is None:
        anchor_start_abs = current_start_abs

    run_start_elapsed_sec: Optional[float] = None
    run_end_elapsed_sec: Optional[float] = None
    if current_start_abs is not None and anchor_start_abs is not None:
        run_start_elapsed_sec = float(current_start_abs - anchor_start_abs)
    if current_end_abs is not None and anchor_start_abs is not None:
        run_end_elapsed_sec = float(current_end_abs - anchor_start_abs)

    if run_start_elapsed_sec is not None and math.isfinite(run_start_elapsed_sec):
        window_start_from_anchor_sec = np.asarray(window_start_from_run_sec + np.float32(run_start_elapsed_sec), dtype=np.float32)
        window_end_from_anchor_sec = np.asarray(window_end_from_run_sec + np.float32(run_start_elapsed_sec), dtype=np.float32)
    else:
        window_start_from_anchor_sec = np.full(window_start_from_run_sec.shape, np.nan, dtype=np.float32)
        window_end_from_anchor_sec = np.full(window_end_from_run_sec.shape, np.nan, dtype=np.float32)

    window_bin_id, window_bin_label, bins_sec = _build_bin_vectors(window_start_from_anchor_sec, bins_hours)

    status = str(current_parse.get("status", "header_parse_unknown"))
    if status == "ok_with_warnings":
        status = "ok_with_warnings"
    elif status == "ok" and run_start_elapsed_sec is None:
        status = "ok_without_anchor_alignment"

    header_rel = str(current_row.get("rel_path", ""))
    if not header_rel:
        try:
            header_rel = str(current_header_path.relative_to(dataset_root)).replace("\\", "/")
        except Exception:
            header_rel = str(current_header_path)

    run_duration_sec = current_parse.get("run_duration_sec")
    if run_duration_sec is None and run_start_elapsed_sec is not None and run_end_elapsed_sec is not None:
        run_duration_sec = float(run_end_elapsed_sec - run_start_elapsed_sec)

    run_payload = {
        "schema_version": schema_version,
        "parser": parser,
        "status": status,
        "anchor_mode": anchor_mode,
        "header_path": header_rel,
        "run_start_clock_raw": str(current_parse.get("run_start_clock_raw") or ""),
        "run_end_clock_raw": str(current_parse.get("run_end_clock_raw") or ""),
        "run_start_clock_sec": np.float32(current_parse.get("run_start_clock_sec", np.nan)),
        "run_end_clock_sec": np.float32(current_parse.get("run_end_clock_sec", np.nan)),
        "run_start_elapsed_sec": np.float32(run_start_elapsed_sec if run_start_elapsed_sec is not None else np.nan),
        "run_end_elapsed_sec": np.float32(run_end_elapsed_sec if run_end_elapsed_sec is not None else np.nan),
        "run_duration_sec": np.float32(run_duration_sec if run_duration_sec is not None else np.nan),
        "subject_anchor_start_sec": np.float32(anchor_start_abs if anchor_start_abs is not None else np.nan),
        "bins_hours": np.asarray(bins_hours, dtype=np.float32),
        "bins_seconds": bins_sec,
        "parse_errors": np.asarray(list(current_parse.get("parse_errors", []) or []), dtype="<U64"),
    }
    extension = {
        "run": run_payload,
        "windows": {
            "window_start_from_run_sec": window_start_from_run_sec,
            "window_end_from_run_sec": window_end_from_run_sec,
            "window_start_from_anchor_sec": window_start_from_anchor_sec,
            "window_end_from_anchor_sec": window_end_from_anchor_sec,
            "window_bin_id": window_bin_id,
            "window_bin_label": window_bin_label,
        },
    }

    attrs = {
        "time_reference_schema_version": schema_version,
        "time_reference_source": parser,
        "time_reference_status": status,
        "time_reference_anchor_mode": anchor_mode,
        "time_reference_enabled": True,
        "time_reference_header_path": header_rel,
        "time_reference_run_start_elapsed_sec": run_start_elapsed_sec,
        "time_reference_run_end_elapsed_sec": run_end_elapsed_sec,
        "time_reference_subject_anchor_start_sec": anchor_start_abs,
    }
    manifest = {
        "enabled": True,
        "schema_version": schema_version,
        "source": parser,
        "status": status,
        "anchor": anchor_mode,
        "header_path": header_rel,
        "run_start_elapsed_sec": run_start_elapsed_sec,
        "run_end_elapsed_sec": run_end_elapsed_sec,
        "bins_hours": bins_hours,
        "parse_errors": list(current_parse.get("parse_errors", []) or []),
    }
    return {
        "enabled": True,
        "status": status,
        "attrs": attrs,
        "manifest": manifest,
        "extension": extension,
    }
