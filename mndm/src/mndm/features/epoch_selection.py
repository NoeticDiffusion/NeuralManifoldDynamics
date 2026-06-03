"""Shared epoch/stage selection helpers for multimodal feature extractors."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..reproducibility import resolve_component_seed


def resolve_epoch_params(config: Mapping[str, Any], dataset_id: Optional[str]) -> tuple[float, float]:
    """Resolve epoch length/step with optional per-dataset overrides."""
    epoching = config.get("epoching", {}) if isinstance(config, Mapping) else {}
    length_s = float(epoching.get("length_s", 8.0) or 8.0) if isinstance(epoching, Mapping) else 8.0
    step_s = float(epoching.get("step_s", 4.0) or 4.0) if isinstance(epoching, Mapping) else 4.0
    if dataset_id and isinstance(epoching, Mapping):
        ds_map = epoching.get("datasets", {})
        if isinstance(ds_map, Mapping):
            ds_cfg = ds_map.get(dataset_id, {})
            if isinstance(ds_cfg, Mapping):
                if "length_s" in ds_cfg:
                    length_s = float(ds_cfg.get("length_s", length_s) or length_s)
                if "step_s" in ds_cfg:
                    step_s = float(ds_cfg.get("step_s", step_s) or step_s)
    return length_s, step_s


def build_epoch_meta(n_samples: int, epoch_length_samples: int, epoch_step_samples: int) -> List[tuple[int, int, int]]:
    """Build (epoch_id, start_idx, end_idx) tuples for valid epochs."""
    if epoch_length_samples <= 0 or epoch_step_samples <= 0 or n_samples < epoch_length_samples:
        return []
    n_epochs = (n_samples - epoch_length_samples) // epoch_step_samples + 1
    out: List[tuple[int, int, int]] = []
    for epoch_idx in range(max(n_epochs, 0)):
        start_idx = epoch_idx * epoch_step_samples
        end_idx = start_idx + epoch_length_samples
        if end_idx > n_samples:
            break
        out.append((epoch_idx, start_idx, end_idx))
    return out


def resolve_epoching_sampling_cfg(config: Mapping[str, Any], dataset_id: Optional[str]) -> Dict[str, Any]:
    """Resolve epoch sampling config with optional per-dataset overrides."""
    epoching = config.get("epoching", {}) if isinstance(config, Mapping) else {}
    merged: Dict[str, Any] = {}
    if isinstance(epoching, Mapping):
        sampling = epoching.get("sampling", {})
        if isinstance(sampling, Mapping):
            merged.update(dict(sampling))
        if dataset_id:
            ds_map = epoching.get("datasets", {})
            if isinstance(ds_map, Mapping):
                ds_cfg = ds_map.get(dataset_id, {})
                if isinstance(ds_cfg, Mapping):
                    ds_sampling = ds_cfg.get("sampling", {})
                    if isinstance(ds_sampling, Mapping):
                        merged.update(dict(ds_sampling))
    repro_cfg = config.get("reproducibility", {}) if isinstance(config, Mapping) else {}
    if isinstance(repro_cfg, Mapping) and repro_cfg:
        merged["reproducibility"] = dict(repro_cfg)
    return merged


def default_stage_map() -> Dict[str, int]:
    """Handle default stage map."""
    return {"W": 0, "Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "R": 4, "L": -1}


def _normalize_stage_key(value: Any) -> str:
    """Normalize stage/event labels for robust key matching."""
    text = str(value).strip()
    text = "".join(ch if ch.isprintable() else " " for ch in text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def find_events_tsv_for_raw(file_path: str) -> Optional[Path]:
    """Best-effort locate BIDS ``*_events.tsv`` next to a raw file path."""
    if not file_path:
        return None
    try:
        p = Path(file_path)
    except Exception:
        return None
    parent = p.parent
    stem = p.stem
    base_core = stem
    if stem.endswith("_eeg"):
        base_core = stem[:-4]
    elif stem.endswith("_ieeg"):
        base_core = stem[:-5]
    candidate = parent / f"{base_core}_events.tsv"
    if candidate.exists():
        return candidate
    legacy = parent / "events.tsv"
    if legacy.exists():
        return legacy
    return None


def resolve_sleep_annotation_cfg(config: Mapping[str, Any], dataset_id: Optional[str]) -> Dict[str, Any]:
    """Resolve optional sleep-annotation config with dataset overrides."""
    root = config.get("sleep_annotations", {}) if isinstance(config, Mapping) else {}
    if not isinstance(root, Mapping):
        return {}
    merged: Dict[str, Any] = {k: root[k] for k in root if k != "datasets"}
    ds_map = root.get("datasets", {})
    if dataset_id and isinstance(ds_map, Mapping):
        ds_cfg = ds_map.get(dataset_id)
        if isinstance(ds_cfg, Mapping):
            merged.update(dict(ds_cfg))
    return merged


def find_sleep_annotation_for_raw(file_path: str, annotation_cfg: Mapping[str, Any]) -> Optional[Path]:
    """Best-effort locate a text sleep annotation sidecar near the raw file."""
    try:
        raw_path = Path(file_path)
    except Exception:
        return None
    if not raw_path.exists():
        return None

    if "annotation_file" in annotation_cfg and annotation_cfg.get("annotation_file"):
        fixed = raw_path.parent / str(annotation_cfg.get("annotation_file"))
        if fixed.exists():
            return fixed

    globs = annotation_cfg.get("file_globs", ["*.txt", "*.ann"])
    if not isinstance(globs, list) or not globs:
        globs = ["*.txt", "*.ann"]

    parent_levels = int(annotation_cfg.get("search_parent_levels", 1) or 1)
    parent_levels = max(0, min(parent_levels, 4))
    dirs: List[Path] = [raw_path.parent]
    for i in range(1, parent_levels + 1):
        if i < len(raw_path.parents):
            dirs.append(raw_path.parents[i])
    dirs = list(dict.fromkeys(dirs))

    candidates: List[Path] = []
    for d in dirs:
        for pattern in globs:
            try:
                candidates.extend([p for p in d.glob(str(pattern)) if p.is_file()])
            except Exception:
                continue
    if not candidates:
        return None

    uniq = sorted({str(p.resolve()): p for p in candidates}.values(), key=lambda p: str(p))
    preferred = annotation_cfg.get("preferred_files", [])
    if isinstance(preferred, list):
        preferred_norm = {str(x).lower() for x in preferred}
        for p in uniq:
            if p.name.lower() in preferred_norm:
                return p

    raw_stem = raw_path.stem.lower()
    for p in uniq:
        if raw_stem and raw_stem in p.stem.lower():
            return p

    subj_hint = raw_path.parent.name.lower()
    for p in uniq:
        if subj_hint and subj_hint in p.stem.lower():
            return p
    return uniq[0]


def label_epochs_from_sleep_annotation(
    epoch_meta: Sequence[tuple[int, int, int]],
    sfreq: float,
    annotation_path: Path,
    stage_map: Mapping[str, int],
) -> Optional[np.ndarray]:
    """Label epochs using annotation text rows: <label> <start_sec> <duration_sec>."""
    rows: List[tuple[float, float, int]] = []
    try:
        with annotation_path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = str(raw).strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                label = str(parts[0]).strip()
                try:
                    onset = float(parts[1])
                    duration = float(parts[2])
                except Exception:
                    continue
                code = int(stage_map.get(label, stage_map.get(label.upper(), -1)))
                rows.append((onset, duration, code))
    except Exception:
        return None
    if not rows:
        return None

    t_start = np.asarray([s / sfreq for (_, s, _) in epoch_meta], dtype=float)
    t_end = np.asarray([e / sfreq for (_, _, e) in epoch_meta], dtype=float)
    t_mid = (t_start + t_end) * 0.5
    out = np.full((t_mid.shape[0],), -1, dtype=np.int16)
    for onset, duration, code in rows:
        end = onset + (duration if duration > 0 else 0.0)
        if end <= onset:
            # Point-events should map to the epoch window containing onset.
            mask = (t_start <= onset) & (t_end > onset)
            if not mask.any():
                mask = np.isclose(t_mid, onset, atol=1e-3)
        else:
            mask = (t_mid >= onset) & (t_mid < end)
        if mask.any():
            out[mask] = int(code)
    return out


def label_epochs_with_stages(
    epoch_meta: Sequence[tuple[int, int, int]],
    sfreq: float,
    events_df: pd.DataFrame,
    stage_columns: Sequence[str],
    onset_column: str = "onset",
    duration_column: str = "duration",
    stage_map: Optional[Mapping[str, int]] = None,
    sampling_cfg: Optional[Mapping[str, Any]] = None,
) -> Optional[np.ndarray]:
    """Return per-epoch stage codes using events onset/duration + stage columns."""
    if events_df is None or events_df.empty or onset_column not in events_df.columns:
        return None
    stage_col = None
    for c in stage_columns:
        if c in events_df.columns:
            stage_col = c
            break
    if stage_col is None:
        return None

    onset = pd.to_numeric(events_df[onset_column], errors="coerce").to_numpy(dtype=float)
    duration = (
        pd.to_numeric(events_df[duration_column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if duration_column in events_df.columns
        else np.zeros((len(events_df),), dtype=float)
    )
    stage_series = events_df[stage_col].astype(str).fillna("")
    stage_num = pd.to_numeric(stage_series, errors="coerce")
    if stage_map is None:
        stage_map = default_stage_map()
    mapping = {
        _normalize_stage_key(k): int(v)
        for k, v in (stage_map.items() if isinstance(stage_map, Mapping) else [])
        if _normalize_stage_key(k) != ""
    }
    mapped = stage_series.astype(str).map(lambda s: mapping.get(_normalize_stage_key(s), np.nan))
    stage = stage_num.fillna(pd.to_numeric(mapped, errors="coerce")).to_numpy(dtype=float)

    valid = np.isfinite(onset) & np.isfinite(duration)
    onset = onset[valid]
    duration = duration[valid]
    stage = stage[valid]
    labels = stage_series.to_numpy(dtype=object)[valid]
    if onset.size == 0:
        return None

    t_start = np.asarray([s / sfreq for (_, s, _) in epoch_meta], dtype=float)
    t_end = np.asarray([e / sfreq for (_, _, e) in epoch_meta], dtype=float)
    t_mid = (t_start + t_end) * 0.5
    out = np.full((t_mid.shape[0],), -1, dtype=np.int16)

    stage_blocking_cfg = {}
    if isinstance(sampling_cfg, Mapping):
        cfg = sampling_cfg.get("stage_blocking", {})
        if isinstance(cfg, Mapping):
            stage_blocking_cfg = dict(cfg)
    blocking_enabled = bool(stage_blocking_cfg.get("enabled", False))

    if not blocking_enabled:
        finite = np.isfinite(stage)
        onset_f = onset[finite]
        duration_f = duration[finite]
        stage_f = stage[finite].astype(int, copy=False)
        if onset_f.size == 0:
            return None
        for o, d, s in zip(onset_f, duration_f, stage_f):
            end = o + (d if d > 0 else 0.0)
            if end <= o:
                # Point-events should map to the epoch window containing onset.
                mask = (t_start <= o) & (t_end > o)
                if not mask.any():
                    mask = np.isclose(t_mid, o, atol=1e-3)
            else:
                mask = (t_mid >= o) & (t_mid < end)
            if mask.any():
                out[mask] = int(s)
        return out.astype(np.int16, copy=False)

    event_regex = str(
        stage_blocking_cfg.get(
            "stage_event_regex",
            stage_blocking_cfg.get(
                "carrier_event_regex",
                stage_blocking_cfg.get("photic_regex", ""),
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

    marker_labels_raw = stage_blocking_cfg.get(
        "bridge_marker_labels",
        stage_blocking_cfg.get("marker_labels", stage_blocking_cfg.get("hv_mark_labels", [])),
    )
    if isinstance(marker_labels_raw, Sequence) and not isinstance(marker_labels_raw, (str, bytes)):
        marker_labels = [_normalize_stage_key(v) for v in marker_labels_raw]
    elif marker_labels_raw in (None, ""):
        marker_labels = []
    else:
        marker_labels = [_normalize_stage_key(marker_labels_raw)]
    hv_labels = {v for v in marker_labels if v}

    use_hv_marks = bool(stage_blocking_cfg.get("use_bridge_markers", stage_blocking_cfg.get("use_hv_marks", True)))
    preserve_photic_blocks = bool(
        stage_blocking_cfg.get(
            "preserve_block_assignments",
            stage_blocking_cfg.get("preserve_photic_blocks", True),
        )
    )
    min_block_sec = float(stage_blocking_cfg.get("min_block_sec", 2.0) or 2.0)
    max_block_sec = float(stage_blocking_cfg.get("max_block_sec", 20.0) or 20.0)
    hv_tail_sec = float(stage_blocking_cfg.get("hv_tail_sec", 0.5) or 0.5)
    hv_tail_cap_sec = float(stage_blocking_cfg.get("hv_tail_cap_sec", 1.0) or 1.0)

    labels_norm = np.asarray([str(v).strip() for v in labels], dtype=object)
    labels_key = np.asarray([_normalize_stage_key(v) for v in labels], dtype=object)
    order = np.argsort(onset, kind="stable")

    freq_hz = np.full((len(labels_norm),), np.nan, dtype=np.float64)
    photic_indices: List[int] = []
    for idx in order.tolist():
        if photic_re is None:
            continue
        m = photic_re.match(str(labels_norm[idx]))
        if not m:
            continue
        try:
            freq_hz[idx] = float(m.group(1))
        except Exception:
            freq_hz[idx] = np.nan
        if np.isfinite(stage[idx]):
            photic_indices.append(int(idx))

    for pos, idx in enumerate(photic_indices):
        start = float(onset[idx])
        next_start = float(onset[photic_indices[pos + 1]]) if (pos + 1) < len(photic_indices) else float("inf")
        end_direct = (
            float(start + duration[idx])
            if np.isfinite(duration[idx]) and float(duration[idx]) > 0
            else float("nan")
        )
        end_hv = float("nan")
        if use_hv_marks and hv_labels:
            hv_mask = np.array([_normalize_stage_key(v) in hv_labels for v in labels_norm], dtype=bool)
            in_range = hv_mask & np.isfinite(onset) & (onset >= start)
            if np.isfinite(next_start):
                in_range &= onset < next_start
            hv_onsets = np.sort(np.asarray(onset[in_range], dtype=np.float64))
            if hv_onsets.size:
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
            out[block_mask] = int(stage[idx])

    for idx in order.tolist():
        if not np.isfinite(stage[idx]):
            continue
        is_hv = _normalize_stage_key(labels_key[idx]) in hv_labels
        if is_hv and photic_indices:
            # With explicit photic block inference active, treat HV marks as
            # support/provenance signals rather than direct window labels.
            continue
        o = float(onset[idx])
        d = float(duration[idx])
        end = o + (d if d > 0 else 0.0)
        if end <= o:
            mask = (t_start <= o) & (t_end > o)
            if not mask.any():
                mask = np.isclose(t_mid, o, atol=1e-3)
        else:
            mask = (t_mid >= o) & (t_mid < end)
        if not mask.any():
            continue
        if np.isfinite(freq_hz[idx]):
            mask_apply = mask & ((out == -1) | (out == int(stage[idx])))
        elif preserve_photic_blocks:
            mask_apply = mask & (out == -1)
        else:
            mask_apply = mask
        if mask_apply.any():
            out[mask_apply] = int(stage[idx])

    return out.astype(np.int16, copy=False)


def _contiguous_runs(indices: np.ndarray) -> List[tuple[int, int]]:
    """Internal helper: contiguous runs."""
    if indices.size == 0:
        return []
    idx = np.sort(indices.astype(int))
    runs: List[tuple[int, int]] = []
    start = int(idx[0])
    prev = int(idx[0])
    for v in idx[1:]:
        v = int(v)
        if v == prev + 1:
            prev = v
            continue
        runs.append((start, prev))
        start = v
        prev = v
    runs.append((start, prev))
    return runs


def select_stage_stratified_blocks(
    stage_per_epoch: np.ndarray,
    epoch_step_sec: float,
    sampling_cfg: Mapping[str, Any],
) -> Optional[np.ndarray]:
    """Select epoch indices via stage-stratified contiguous blocks."""
    if stage_per_epoch is None or stage_per_epoch.size == 0:
        return None
    target_minutes = sampling_cfg.get("target_minutes", {})
    if not isinstance(target_minutes, Mapping) or not target_minutes:
        return None
    block_minutes = float(sampling_cfg.get("block_minutes", 5) or 5.0)
    block_epochs = max(1, int(round((block_minutes * 60.0) / max(epoch_step_sec, 1e-6))))
    seed, _ = resolve_component_seed(
        {"reproducibility": sampling_cfg.get("reproducibility", {})} if isinstance(sampling_cfg, Mapping) else None,
        fallback_seed=sampling_cfg.get("seed"),
        fallback_source="epoching.sampling.seed",
    )
    rng = np.random.default_rng(seed)
    stage_code_map = {"Wake": 0, "W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}

    chosen: List[int] = []
    for stage_key, minutes in target_minutes.items():
        code = stage_code_map.get(str(stage_key), None)
        if code is None:
            try:
                code = int(stage_key)  # type: ignore[arg-type]
            except Exception:
                continue
        minutes_f = float(minutes or 0.0)
        if minutes_f <= 0:
            continue
        target_epochs = max(1, int(round((minutes_f * 60.0) / max(epoch_step_sec, 1e-6))))
        idxs = np.where(stage_per_epoch.astype(int) == int(code))[0]
        if idxs.size == 0:
            continue
        runs = _contiguous_runs(idxs)
        runs = sorted(runs, key=lambda r: (r[1] - r[0] + 1), reverse=True)
        need = target_epochs
        for (a, b) in runs:
            run_len = (b - a + 1)
            if run_len < block_epochs:
                continue
            max_blocks = max(1, need // block_epochs)
            max_blocks = min(max_blocks, (run_len // block_epochs))
            if max_blocks <= 1:
                start = a + (run_len - block_epochs) // 2
                chosen.extend(list(range(start, start + block_epochs)))
                need -= block_epochs
            else:
                span = run_len - block_epochs
                qs = np.linspace(0.0, 1.0, num=max_blocks, endpoint=True)
                starts = [a + int(round(q * span)) for q in qs]
                jitter = rng.integers(-1, 2, size=len(starts))
                for s0, j in zip(starts, jitter):
                    s1 = int(np.clip(int(s0) + int(j), a, b - block_epochs + 1))
                    chosen.extend(list(range(s1, s1 + block_epochs)))
                    need -= block_epochs
                    if need <= 0:
                        break
            if need <= 0:
                break

    if not chosen:
        return None
    return np.unique(np.asarray(chosen, dtype=int))


def resolve_stage_stratified_epoch_set(
    *,
    config: Mapping[str, Any],
    dataset_id: Optional[str],
    raw_file_path: Optional[str],
    sfreq: float,
    n_samples: int,
    step_s: float,
    epoch_length_samples: int,
    epoch_step_samples: int,
) -> Optional[set[int]]:
    """Return selected epoch ids from stage-stratified policy, if enabled."""
    if not raw_file_path:
        return None

    sampling_cfg = resolve_epoching_sampling_cfg(config, dataset_id)
    if not (
        isinstance(sampling_cfg, Mapping)
        and bool(sampling_cfg.get("enabled", False))
        and str(sampling_cfg.get("method", "")).lower() == "stage_stratified_blocks"
    ):
        return None

    meta_all = build_epoch_meta(
        n_samples=int(n_samples),
        epoch_length_samples=int(epoch_length_samples),
        epoch_step_samples=int(epoch_step_samples),
    )
    if not meta_all:
        return None

    stage_map = default_stage_map()
    cfg_stage_map = sampling_cfg.get("stage_map", {}) if isinstance(sampling_cfg, Mapping) else {}
    if isinstance(cfg_stage_map, Mapping):
        for k, v in cfg_stage_map.items():
            try:
                stage_map[str(k)] = int(v)
            except Exception:
                continue

    stage_per_epoch = None
    events_path = find_events_tsv_for_raw(str(raw_file_path))
    if events_path is not None:
        events_df = pd.read_csv(events_path, sep="\t")
        stage_cols = sampling_cfg.get("stage_columns", ["stage_hum", "stage_ai", "stage"])
        stage_cols_list = [str(c) for c in stage_cols] if isinstance(stage_cols, list) else ["stage_hum", "stage_ai", "stage"]
        stage_per_epoch = label_epochs_with_stages(
            epoch_meta=meta_all,
            sfreq=float(sfreq),
            events_df=events_df,
            stage_columns=stage_cols_list,
            onset_column=str(sampling_cfg.get("onset_column", "onset")),
            duration_column=str(sampling_cfg.get("duration_column", "duration")),
            stage_map=stage_map,
            sampling_cfg=sampling_cfg if isinstance(sampling_cfg, Mapping) else None,
        )

    if stage_per_epoch is None:
        ann_cfg = resolve_sleep_annotation_cfg(config, dataset_id)
        if bool(ann_cfg.get("enabled", False)):
            cfg_map = ann_cfg.get("stage_map", {})
            if isinstance(cfg_map, Mapping):
                for k, v in cfg_map.items():
                    try:
                        stage_map[str(k)] = int(v)
                    except Exception:
                        continue
            ann_path = find_sleep_annotation_for_raw(str(raw_file_path), ann_cfg)
            if ann_path is not None:
                stage_per_epoch = label_epochs_from_sleep_annotation(
                    epoch_meta=meta_all,
                    sfreq=float(sfreq),
                    annotation_path=ann_path,
                    stage_map=stage_map,
                )

    chosen_idx = select_stage_stratified_blocks(
        stage_per_epoch=stage_per_epoch if stage_per_epoch is not None else np.asarray([], dtype=int),
        epoch_step_sec=float(step_s),
        sampling_cfg=sampling_cfg,
    )
    if chosen_idx is None or chosen_idx.size == 0:
        return None
    return set(int(i) for i in chosen_idx.tolist())
