"""Event/stage helpers extracted from summary.py."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class BidsEventStageProvenance:
    """Event-provenance payload derived from BIDS ``*_events.tsv``."""

    stage_inferred: Optional[np.ndarray] = None
    stage_source: Optional[str] = None
    stage_column: Optional[str] = None
    events_path: Optional[str] = None
    event_table_columns: Dict[str, Any] = field(default_factory=dict)
    legacy_events: Dict[str, np.ndarray] = field(default_factory=dict)
    stage_mapping_qc: Dict[str, Any] = field(default_factory=dict)


def _normalize_label(raw: Any) -> str:
    """Return normalized event label text."""
    if raw is None:
        return ""
    text = str(raw).strip()
    return text


def _normalize_key(raw: Any) -> str:
    """Return lower-cased normalized key."""
    text = _normalize_label(raw)
    # Normalize control/noise characters to spaces for stable matching.
    text = "".join(ch if ch.isprintable() else " " for ch in text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _sanitize_event_key(label: str) -> str:
    """Return a stable ASCII-ish key for legacy event arrays."""
    raw = _normalize_key(label)
    sanitized = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    return sanitized or "unknown_event"


def _resolve_sampling_cfg(ctx_config: Mapping[str, Any], dataset_id: str) -> Dict[str, Any]:
    """Resolve per-dataset epoching sampling config."""
    epoching = ctx_config.get("epoching", {}) if isinstance(ctx_config, Mapping) else {}
    ds_cfg = epoching.get("datasets", {}).get(dataset_id, {}) if isinstance(epoching, Mapping) else {}
    sampling_cfg = ds_cfg.get("sampling", {}) if isinstance(ds_cfg, Mapping) else {}
    return dict(sampling_cfg) if isinstance(sampling_cfg, Mapping) else {}


def _resolve_stage_source_name(stage_col: str) -> str:
    """Resolve a compact stage source tag from a stage column name."""
    col = _normalize_key(stage_col)
    if "hum" in col:
        return "hum"
    if "ai" in col:
        return "ai"
    return "consensus"


def _locate_events_path(
    *,
    sub_frame: pd.DataFrame,
    index_df: Optional[pd.DataFrame],
    dataset_root: Path,
    lookup_rel_paths_by_file_value: Any,
) -> Optional[Path]:
    """Locate ``*_events.tsv`` for the representative file in sub_frame."""
    if "file" not in sub_frame.columns or len(sub_frame) == 0:
        return None
    filename = str(sub_frame["file"].iloc[0])
    rel_path = None
    try:
        rel_candidates = lookup_rel_paths_by_file_value(filename)
        if rel_candidates:
            rel_path = str(rel_candidates[0])
    except Exception:
        rel_path = None

    if rel_path is None and index_df is not None and "path" in index_df.columns:
        try:
            mask = index_df["path"].astype(str).str.endswith(filename)
            if mask.any():
                rel_path = str(index_df.loc[mask, "path"].iloc[0])
        except Exception:
            rel_path = None
    if not rel_path:
        return None

    file_path = dataset_root / rel_path
    if not file_path.exists():
        return None
    base_stem = file_path.stem
    base_core = base_stem[:-4] if base_stem.endswith("_eeg") else base_stem
    events_path = file_path.parent / f"{base_core}_events.tsv"
    if events_path.exists():
        return events_path
    legacy = file_path.parent / "events.tsv"
    if legacy.exists():
        return legacy
    return None


def _event_mask_for_window(
    *,
    onset: float,
    duration: float,
    t_start: np.ndarray,
    t_end: np.ndarray,
    t_mid: np.ndarray,
) -> np.ndarray:
    """Return per-window mask for one event."""
    end = float(onset) + (float(duration) if float(duration) > 0 else 0.0)
    if end <= float(onset):
        mask = (t_start <= float(onset)) & (t_end > float(onset))
        if not mask.any():
            mask = np.isclose(t_mid, float(onset), atol=1e-3)
        return mask
    return (t_mid >= float(onset)) & (t_mid < end)


def _build_legacy_event_arrays(
    onsets: np.ndarray,
    labels: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Build legacy ``payload.events`` onset arrays grouped by raw label."""
    grouped: Dict[str, List[float]] = {}
    for onset, raw_label in zip(onsets, labels):
        if not np.isfinite(float(onset)):
            continue
        label = _normalize_label(raw_label)
        if not label:
            continue
        key = f"raw_{_sanitize_event_key(label)}_onset_sec"
        grouped.setdefault(key, []).append(float(onset))
    out: Dict[str, np.ndarray] = {}
    for key, vals in grouped.items():
        out[key] = np.asarray(sorted(vals), dtype=np.float64)
    return out


def _build_stage_with_blocking(
    *,
    onsets: np.ndarray,
    durations: np.ndarray,
    labels_raw: Sequence[str],
    direct_codes: np.ndarray,
    t_start: np.ndarray,
    t_end: np.ndarray,
    sampling_cfg: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build per-window stage with optional photic block expansion."""
    n_events = int(len(onsets))
    t_mid = (t_start + t_end) / 2.0
    out = np.full((t_mid.shape[0],), -1, dtype=np.int16)

    mapped_codes = np.full((n_events,), np.nan, dtype=np.float64)
    mapping_mode = np.asarray(["unmapped"] * n_events, dtype=object)
    block_ids = np.full((n_events,), -1, dtype=np.int32)
    freq_hz = np.full((n_events,), np.nan, dtype=np.float64)
    window_assign_counts = np.zeros((n_events,), dtype=np.int32)

    blocking_cfg = sampling_cfg.get("stage_blocking", {}) if isinstance(sampling_cfg, Mapping) else {}
    blocking_cfg = dict(blocking_cfg) if isinstance(blocking_cfg, Mapping) else {}
    blocking_enabled = bool(blocking_cfg.get("enabled", False))

    labels_norm = np.asarray([_normalize_label(v) for v in labels_raw], dtype=object)
    labels_key = np.asarray([_normalize_key(v) for v in labels_raw], dtype=object)
    stage_vals = np.asarray(direct_codes, dtype=np.float64)

    order = np.argsort(onsets, kind="stable")

    # Generic stage-blocking config supports multiple key aliases.
    event_regex = str(
        blocking_cfg.get(
            "stage_event_regex",
            blocking_cfg.get(
                "carrier_event_regex",
                blocking_cfg.get("photic_regex", ""),
            ),
        )
        or ""
    ).strip()
    photic_re: Optional[re.Pattern[str]] = None
    if event_regex:
        try:
            photic_re = re.compile(event_regex, flags=re.IGNORECASE)
        except Exception:
            photic_re = None

    def _match_freq(label: str) -> Optional[float]:
        if not label or photic_re is None:
            return None
        m = photic_re.match(str(label))
        if not m:
            return None
        try:
            return float(m.group(1))
        except Exception:
            return None

    hv_labels_raw = blocking_cfg.get(
        "bridge_marker_labels",
        blocking_cfg.get("marker_labels", blocking_cfg.get("hv_mark_labels", [])),
    )
    if isinstance(hv_labels_raw, Sequence) and not isinstance(hv_labels_raw, (str, bytes)):
        hv_values = list(hv_labels_raw)
    elif hv_labels_raw in (None, ""):
        hv_values = []
    else:
        hv_values = [hv_labels_raw]
    hv_labels = {_normalize_key(v) for v in hv_values if _normalize_key(v)}
    protect_non_photic = bool(
        blocking_cfg.get(
            "preserve_block_assignments",
            blocking_cfg.get("preserve_photic_blocks", True),
        )
    )
    min_block_sec = float(blocking_cfg.get("min_block_sec", 2.0) or 2.0)
    max_block_sec = float(blocking_cfg.get("max_block_sec", 20.0) or 20.0)
    hv_tail_sec = float(blocking_cfg.get("hv_tail_sec", 0.5) or 0.5)
    hv_tail_cap_sec = float(blocking_cfg.get("hv_tail_cap_sec", 1.0) or 1.0)
    use_hv_marks = bool(blocking_cfg.get("use_bridge_markers", blocking_cfg.get("use_hv_marks", True)))

    photic_indices: List[int] = []
    for idx in order.tolist():
        freq = _match_freq(str(labels_norm[idx]))
        if freq is None:
            continue
        freq_hz[idx] = float(freq)
        if np.isfinite(stage_vals[idx]):
            photic_indices.append(int(idx))

    if not blocking_enabled:
        for idx in order.tolist():
            if not np.isfinite(stage_vals[idx]):
                continue
            mask = _event_mask_for_window(
                onset=float(onsets[idx]),
                duration=float(durations[idx]),
                t_start=t_start,
                t_end=t_end,
                t_mid=t_mid,
            )
            assigned = int(mask.sum())
            if assigned > 0:
                out[mask] = int(stage_vals[idx])
                window_assign_counts[idx] = int(assigned)
            mapped_codes[idx] = float(stage_vals[idx])
            mapping_mode[idx] = "direct"
        return out.astype(np.int16, copy=False), mapped_codes, mapping_mode, block_ids, freq_hz, window_assign_counts

    block_counter = 0
    for pos, idx in enumerate(photic_indices):
        start = float(onsets[idx])
        next_start = (
            float(onsets[photic_indices[pos + 1]])
            if (pos + 1) < len(photic_indices)
            else float("inf")
        )
        end_direct = (
            float(start + durations[idx])
            if np.isfinite(durations[idx]) and float(durations[idx]) > 0
            else float("nan")
        )
        end_hv = float("nan")
        if use_hv_marks and hv_labels:
            hv_mask = np.array(
                [
                    (_normalize_key(v) in hv_labels)
                    for v in labels_norm
                ],
                dtype=bool,
            )
            hv_onsets = onsets[
                hv_mask
                & np.isfinite(onsets)
                & (onsets >= start)
                & (onsets < next_start if np.isfinite(next_start) else np.ones_like(onsets, dtype=bool))
            ]
            hv_onsets = np.asarray(hv_onsets, dtype=np.float64)
            if hv_onsets.size:
                hv_onsets = np.sort(hv_onsets)
                hv_step = float("nan")
                if hv_onsets.size >= 2:
                    diffs = np.diff(hv_onsets)
                    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
                    if diffs.size:
                        hv_step = float(np.median(diffs))
                tail = float(hv_tail_sec)
                if np.isfinite(hv_step) and hv_step > 0:
                    tail = max(tail, min(hv_step, hv_tail_cap_sec))
                end_hv = float(hv_onsets[-1] + tail)

        if np.isfinite(end_direct) and end_direct > start:
            end = float(end_direct)
        elif np.isfinite(end_hv) and end_hv > start:
            end = float(end_hv)
        elif np.isfinite(next_start):
            end = min(float(next_start), float(start + max_block_sec))
        else:
            end = float(start + max_block_sec)

        if np.isfinite(next_start):
            end = min(end, float(next_start))
        if end - start < min_block_sec:
            if np.isfinite(next_start) and (next_start - start) >= min_block_sec:
                end = min(float(next_start), float(start + max_block_sec))
            else:
                end = float(start + min_block_sec)
        if end <= start:
            end = float(start + max(min_block_sec, 1e-3))

        block_mask = (t_mid >= start) & (t_mid < end)
        if block_mask.any():
            out[block_mask] = int(stage_vals[idx])
        mapped_codes[idx] = float(stage_vals[idx])
        mapping_mode[idx] = "direct_photic_block"
        block_ids[idx] = int(block_counter)
        window_assign_counts[idx] = int(block_mask.sum())

        hv_row_mask = np.array(
            [
                (_normalize_key(v) in hv_labels)
                for v in labels_norm
            ],
            dtype=bool,
        ) & np.isfinite(onsets) & (onsets >= start) & (onsets < end)
        hv_indices = np.where(hv_row_mask)[0]
        for hv_idx in hv_indices.tolist():
            if not np.isfinite(mapped_codes[hv_idx]):
                mapped_codes[hv_idx] = float(stage_vals[idx])
                mapping_mode[hv_idx] = "inferred_photic_block"
                block_ids[hv_idx] = int(block_counter)
                freq_hz[hv_idx] = freq_hz[idx]
        block_counter += 1

    for idx in order.tolist():
        if not np.isfinite(stage_vals[idx]):
            continue
        is_hv = _normalize_key(labels_norm[idx]) in hv_labels
        if is_hv and photic_indices:
            if np.isnan(mapped_codes[idx]):
                mapped_codes[idx] = float(stage_vals[idx])
                mapping_mode[idx] = "direct_hv_support"
            continue
        mask = _event_mask_for_window(
            onset=float(onsets[idx]),
            duration=float(durations[idx]),
            t_start=t_start,
            t_end=t_end,
            t_mid=t_mid,
        )
        if not mask.any():
            if np.isnan(mapped_codes[idx]):
                mapped_codes[idx] = float(stage_vals[idx])
                mapping_mode[idx] = "direct_no_window_match"
            continue

        if np.isfinite(freq_hz[idx]):
            mask_apply = mask & ((out == -1) | (out == int(stage_vals[idx])))
        elif protect_non_photic:
            mask_apply = mask & (out == -1)
        else:
            mask_apply = mask

        assigned = int(mask_apply.sum())
        if assigned > 0:
            out[mask_apply] = int(stage_vals[idx])
            window_assign_counts[idx] = max(int(window_assign_counts[idx]), int(assigned))
        if np.isnan(mapped_codes[idx]):
            mapped_codes[idx] = float(stage_vals[idx])
            mapping_mode[idx] = "direct"

    return out.astype(np.int16, copy=False), mapped_codes, mapping_mode, block_ids, freq_hz, window_assign_counts


def _build_stage_mapping_qc(
    *,
    labels_raw: Sequence[str],
    mapped_codes: np.ndarray,
    mapping_mode: np.ndarray,
    inferred_freq_hz: np.ndarray,
    stage_for_windows: Optional[np.ndarray],
    sampling_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build stage/event mapping QC dict."""
    raw_counts = Counter(_normalize_label(v) for v in labels_raw if _normalize_label(v))

    mapped_code_counts: Dict[str, int] = {}
    for v in mapped_codes.tolist():
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if not np.isfinite(fv):
            continue
        key = str(int(fv))
        mapped_code_counts[key] = mapped_code_counts.get(key, 0) + 1

    unmapped_label_counts: Dict[str, int] = {}
    for raw_label, code in zip(labels_raw, mapped_codes):
        lbl = _normalize_label(raw_label)
        if not lbl:
            continue
        try:
            code_f = float(code)
        except Exception:
            code_f = float("nan")
        if not np.isfinite(code_f):
            unmapped_label_counts[lbl] = unmapped_label_counts.get(lbl, 0) + 1

    mode_counts = Counter(str(v) for v in mapping_mode.tolist() if str(v))

    raw_freq_vals = inferred_freq_hz[np.isfinite(inferred_freq_hz)]
    raw_freq_counts: Dict[str, int] = {}
    for fv in raw_freq_vals.tolist():
        key = str(int(round(float(fv))))
        raw_freq_counts[key] = raw_freq_counts.get(key, 0) + 1

    window_stage_counts: Dict[str, int] = {}
    if stage_for_windows is not None and len(stage_for_windows) > 0:
        for v in np.asarray(stage_for_windows).tolist():
            try:
                key = str(int(v))
            except Exception:
                continue
            window_stage_counts[key] = window_stage_counts.get(key, 0) + 1

    stage_blocking = sampling_cfg.get("stage_blocking", {}) if isinstance(sampling_cfg, Mapping) else {}
    expected_raw = (
        stage_blocking.get(
            "expected_stage_frequencies_hz",
            stage_blocking.get("expected_frequencies_hz", []),
        )
        if isinstance(stage_blocking, Mapping)
        else []
    )
    expected_freqs: List[int] = []
    if isinstance(expected_raw, Sequence) and not isinstance(expected_raw, (str, bytes)):
        for v in expected_raw:
            try:
                expected_freqs.append(int(v))
            except Exception:
                continue
    expected_freqs = sorted(set(expected_freqs))
    detected_freqs = sorted({int(round(float(v))) for v in raw_freq_vals.tolist()})

    missing_expected_raw = [int(v) for v in expected_freqs if int(v) not in detected_freqs]

    return {
        "raw_event_label_counts": {str(k): int(v) for k, v in sorted(raw_counts.items(), key=lambda kv: kv[0])},
        "mapped_stage_code_counts": {str(k): int(v) for k, v in sorted(mapped_code_counts.items(), key=lambda kv: kv[0])},
        "unmapped_event_label_counts": {str(k): int(v) for k, v in sorted(unmapped_label_counts.items(), key=lambda kv: kv[0])},
        "mapping_mode_counts": {str(k): int(v) for k, v in sorted(mode_counts.items(), key=lambda kv: kv[0])},
        "stage_blocking_enabled": bool(
            stage_blocking.get("enabled", False) if isinstance(stage_blocking, Mapping) else False
        ),
        "raw_stage_frequency_event_counts_hz": {
            str(k): int(v)
            for k, v in sorted(raw_freq_counts.items(), key=lambda kv: int(kv[0]))
        },
        # Backward-compatible alias kept for existing downstream notebooks.
        "raw_photic_frequency_event_counts_hz": {str(k): int(v) for k, v in sorted(raw_freq_counts.items(), key=lambda kv: int(kv[0]))},
        "window_stage_counts": {str(k): int(v) for k, v in sorted(window_stage_counts.items(), key=lambda kv: int(kv[0]))},
        "expected_frequencies_hz": [int(v) for v in expected_freqs],
        "detected_raw_frequencies_hz": [int(v) for v in detected_freqs],
        "missing_expected_frequencies_hz_raw": [int(v) for v in missing_expected_raw],
        "raw_has_25hz": bool(25 in detected_freqs),
        "raw_has_30hz": bool(30 in detected_freqs),
    }


def build_bids_event_stage_provenance(
    *,
    sub_frame: pd.DataFrame,
    stage_for_windows: Optional[np.ndarray],
    index_df: Optional[pd.DataFrame],
    dataset_root: Path,
    lookup_rel_paths_by_file_value: Any,
    ctx_config: Mapping[str, Any],
    mnps_cfg: Mapping[str, Any],
    dataset_id: str,
) -> BidsEventStageProvenance:
    """Build stage/event provenance object from BIDS ``*_events.tsv``."""
    events_path = _locate_events_path(
        sub_frame=sub_frame,
        index_df=index_df,
        dataset_root=dataset_root,
        lookup_rel_paths_by_file_value=lookup_rel_paths_by_file_value,
    )
    if events_path is None:
        return BidsEventStageProvenance()

    try:
        events_df = pd.read_csv(events_path, sep="\t")
    except Exception:
        return BidsEventStageProvenance(events_path=str(events_path))
    if events_df.empty:
        return BidsEventStageProvenance(events_path=str(events_path))

    sampling_cfg = _resolve_sampling_cfg(ctx_config, dataset_id)
    candidate_cols = sampling_cfg.get("stage_columns", ["stage_hum", "stage_ai", "stage", "sleep_stage"])
    if not isinstance(candidate_cols, list):
        candidate_cols = [str(candidate_cols)]
    stage_col = None
    for col in candidate_cols:
        if col in events_df.columns:
            stage_col = str(col)
            break
    if stage_col is None or "onset" not in events_df.columns:
        return BidsEventStageProvenance(
            stage_column=stage_col,
            events_path=str(events_path),
        )

    if "t_start" in sub_frame.columns and "t_end" in sub_frame.columns:
        t_start_arr = pd.to_numeric(sub_frame["t_start"], errors="coerce").to_numpy(dtype=float)
        t_end_arr = pd.to_numeric(sub_frame["t_end"], errors="coerce").to_numpy(dtype=float)
    else:
        dt = float(mnps_cfg["window_sec"]) * (1.0 - float(mnps_cfg["overlap"]))
        time_idx = np.arange(len(sub_frame)) * dt
        t_mid_arr = time_idx + 0.5 * float(mnps_cfg["window_sec"])
        half = 0.5 * float(mnps_cfg["window_sec"])
        t_start_arr = t_mid_arr - half
        t_end_arr = t_mid_arr + half

    raw_onsets = pd.to_numeric(events_df["onset"], errors="coerce").to_numpy(dtype=float)
    dur_col = str(sampling_cfg.get("duration_column", "duration"))
    if dur_col in events_df.columns:
        raw_durations = pd.to_numeric(events_df[dur_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        raw_durations = np.zeros_like(raw_onsets)
    labels = events_df[stage_col].astype(str).fillna("").to_numpy(dtype=object)

    stage_codebook = mnps_cfg.get("stage_codebook", {}) if isinstance(mnps_cfg, Mapping) else {}
    mapping = {
        _normalize_key(k): int(v)
        for k, v in (stage_codebook.items() if isinstance(stage_codebook, Mapping) else [])
        if str(k).strip() != ""
    }
    numeric_codes = pd.to_numeric(events_df[stage_col], errors="coerce").to_numpy(dtype=float)
    direct_codes = np.full_like(numeric_codes, np.nan, dtype=float)
    for i, text in enumerate(labels.tolist()):
        key = _normalize_key(text)
        if key in mapping:
            direct_codes[i] = float(mapping[key])
    stage_codes = np.where(np.isfinite(numeric_codes), numeric_codes, direct_codes)

    valid = np.isfinite(raw_onsets) & np.isfinite(raw_durations)
    onsets = raw_onsets[valid]
    durations = raw_durations[valid]
    labels_valid = labels[valid]
    stage_codes_valid = stage_codes[valid]
    if onsets.size == 0:
        return BidsEventStageProvenance(
            stage_column=stage_col,
            stage_source=_resolve_stage_source_name(stage_col),
            events_path=str(events_path),
        )

    stage_inferred, mapped_codes, mapping_mode, block_ids, freq_hz, window_assign_counts = _build_stage_with_blocking(
        onsets=onsets,
        durations=durations,
        labels_raw=labels_valid,
        direct_codes=stage_codes_valid,
        t_start=t_start_arr,
        t_end=t_end_arr,
        sampling_cfg=sampling_cfg,
    )

    stage_used = (
        np.asarray(stage_for_windows, dtype=np.int16)
        if stage_for_windows is not None and len(stage_for_windows) == len(stage_inferred)
        else stage_inferred
    )
    qc = _build_stage_mapping_qc(
        labels_raw=[str(v) for v in labels_valid.tolist()],
        mapped_codes=mapped_codes,
        mapping_mode=mapping_mode,
        inferred_freq_hz=freq_hz,
        stage_for_windows=stage_used,
        sampling_cfg=sampling_cfg,
    )
    qc["events_path"] = str(events_path)
    qc["source_event_column"] = str(stage_col)
    qc["duration_column"] = str(dur_col)
    qc["n_event_rows"] = int(len(onsets))
    qc["n_event_rows_mapped"] = int(np.isfinite(mapped_codes).sum())
    qc["n_event_rows_unmapped"] = int(np.size(mapped_codes) - int(np.isfinite(mapped_codes).sum()))
    qc["n_windows"] = int(len(stage_used)) if stage_used is not None else int(len(stage_inferred))
    blocking_cfg = sampling_cfg.get("stage_blocking", {}) if isinstance(sampling_cfg, Mapping) else {}
    blocking_enabled = bool(blocking_cfg.get("enabled", False)) if isinstance(blocking_cfg, Mapping) else False
    mapping_rule_name = "stage_map_plus_stage_blocking" if blocking_enabled else "stage_map_only"

    event_table_columns: Dict[str, Any] = {
        "_schema_version": np.bytes_("bids_event_provenance.v1"),
        "onset_sec": np.asarray(onsets, dtype=np.float64),
        "duration_sec": np.asarray(durations, dtype=np.float64),
        "raw_event_label": np.asarray([_normalize_label(v) for v in labels_valid.tolist()], dtype=object),
        "normalized_event_label": np.asarray([_normalize_key(v) for v in labels_valid.tolist()], dtype=object),
        "mapped_stage_code": np.asarray(mapped_codes, dtype=np.float64),
        "mapping_mode": np.asarray([str(v) for v in mapping_mode.tolist()], dtype=object),
        "source_event_column": np.asarray([str(stage_col)] * len(onsets), dtype=object),
        "mapping_rule": np.asarray([mapping_rule_name] * len(onsets), dtype=object),
        "is_stage_block_event": np.asarray(np.isfinite(freq_hz), dtype=np.int8),
        "stage_block_frequency_hz": np.asarray(freq_hz, dtype=np.float64),
        # Backward-compatible aliases kept for existing downstream notebooks.
        "is_photic": np.asarray(np.isfinite(freq_hz), dtype=np.int8),
        "photic_frequency_hz": np.asarray(freq_hz, dtype=np.float64),
        "inferred_block_id": np.asarray(block_ids, dtype=np.int32),
        "window_assignment_count": np.asarray(window_assign_counts, dtype=np.int32),
    }

    legacy_events = _build_legacy_event_arrays(
        onsets=np.asarray(onsets, dtype=np.float64),
        labels=[str(v) for v in labels_valid.tolist()],
    )

    return BidsEventStageProvenance(
        stage_inferred=stage_inferred.astype(np.int8, copy=False),
        stage_source=_resolve_stage_source_name(stage_col),
        stage_column=str(stage_col),
        events_path=str(events_path),
        event_table_columns=event_table_columns,
        legacy_events=legacy_events,
        stage_mapping_qc=qc,
    )


def infer_stage_from_bids_events(
    *,
    sub_frame: pd.DataFrame,
    index_df: Optional[pd.DataFrame],
    dataset_root: Path,
    lookup_rel_paths_by_file_value: Any,
    ctx_config: Mapping[str, Any],
    mnps_cfg: Mapping[str, Any],
    dataset_id: str,
) -> Tuple[Optional[np.ndarray], Optional[str], Optional[str], Optional[str]]:
    """Infer per-epoch stage codes from BIDS ``*_events.tsv`` for a segment."""
    provenance = build_bids_event_stage_provenance(
        sub_frame=sub_frame,
        stage_for_windows=None,
        index_df=index_df,
        dataset_root=dataset_root,
        lookup_rel_paths_by_file_value=lookup_rel_paths_by_file_value,
        ctx_config=ctx_config,
        mnps_cfg=mnps_cfg,
        dataset_id=dataset_id,
    )
    return (
        provenance.stage_inferred,
        provenance.stage_source,
        provenance.stage_column,
        provenance.events_path,
    )


def estimate_coverage_seconds(sub_frame: pd.DataFrame, dt_fallback: float) -> Tuple[float, str]:
    """Estimate coverage from timestamps when available; else fallback to len*dt."""
    if "t_start" in sub_frame.columns and "t_end" in sub_frame.columns:
        try:
            t_start = pd.to_numeric(sub_frame["t_start"], errors="coerce").to_numpy(dtype=float)
            t_end = pd.to_numeric(sub_frame["t_end"], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(t_start) & np.isfinite(t_end)
            if np.any(valid):
                starts = t_start[valid]
                ends = t_end[valid]
                span = float(np.nanmax(ends) - np.nanmin(starts))
                if np.isfinite(span) and span > 0:
                    return span, "timestamps_span"
        except Exception:
            pass
    return float(len(sub_frame) * float(dt_fallback)), "assumed_len_dt"


def map_events_to_labels(
    *,
    config: Mapping[str, Any],
    time: np.ndarray,
    window_start: np.ndarray,
    window_end: np.ndarray,
    events: Mapping[str, np.ndarray],
    dataset_id: str,
) -> Dict[str, np.ndarray]:
    """Map event timestamps to MNPS window-aligned binary labels (opt-in)."""
    ev_cfg = config.get("event_mapping", {}) if isinstance(config, Mapping) else {}
    enabled = bool(ev_cfg.get("enabled", False))
    ds_override = (ev_cfg.get("datasets", {}) or {}).get(dataset_id, {})
    if isinstance(ds_override, Mapping) and "enabled" in ds_override:
        enabled = bool(ds_override.get("enabled", enabled))
    if not enabled or not events:
        return {}

    tol = ev_cfg.get("tolerance_sec", None)
    if isinstance(ds_override, Mapping) and "tolerance_sec" in ds_override:
        tol = ds_override.get("tolerance_sec", tol)
    if tol is None:
        tol = float(max(np.diff(time).min(initial=0.0), 0.0)) / 2.0 if len(time) > 1 else 0.0

    labels: Dict[str, np.ndarray] = {}
    for name, vals in events.items():
        arr = np.zeros_like(time, dtype=bool)
        for v in np.asarray(vals, dtype=float):
            mask = (window_start <= v) & (window_end >= v)
            if not mask.any() and tol > 0:
                mask = np.abs(time - v) <= tol
            arr |= mask
        if arr.any():
            labels[name] = arr.astype(np.int8)
    return labels
