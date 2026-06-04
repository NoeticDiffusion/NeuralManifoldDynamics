"""Generic helpers for inferring stage blocks from sparse event streams."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .intervals import WindowMembershipSpec, membership_spec_from_mapping


def normalize_block_label(raw: Any) -> str:
    """Return normalized block/event label text."""
    if raw is None:
        return ""
    return str(raw).strip()


def normalize_block_key(raw: Any) -> str:
    """Return lower-cased normalized key for stable comparisons."""
    text = normalize_block_label(raw)
    text = "".join(ch if ch.isprintable() else " " for ch in text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


@dataclass(frozen=True)
class StageBlockInterval:
    """One inferred stage block on the run timeline."""

    start_sec: float
    end_sec: float
    stage_code: int
    source_event_idx: int
    block_id: int
    block_parameter: float = float("nan")
    support_event_indices: Tuple[int, ...] = ()
    derived_from: str = "stage_blocking"
    end_reason: str = "unknown"
    membership_mode: str = ""
    bridge_tail_sec: float = float("nan")
    bridge_tail_cap_sec: float = float("nan")
    is_inferred: bool = True


@dataclass(frozen=True)
class StageBlockingConfig:
    """Normalized stage-block inference config."""

    enabled: bool = False
    stage_event_regex: str = ""
    bridge_marker_labels: Tuple[str, ...] = ()
    use_bridge_markers: bool = True
    preserve_block_assignments: bool = True
    min_block_sec: float = 2.0
    max_block_sec: float = 20.0
    bridge_tail_sec: float = 0.5
    bridge_tail_cap_sec: float = 1.0
    expected_stage_frequencies_hz: Tuple[int, ...] = ()
    window_membership: WindowMembershipSpec = field(default_factory=WindowMembershipSpec)


def stage_blocking_config_from_mapping(cfg: Mapping[str, Any] | None) -> StageBlockingConfig:
    """Normalize a stage-blocking config mapping with backward-compatible aliases."""
    raw = dict(cfg) if isinstance(cfg, Mapping) else {}
    marker_labels_raw = raw.get(
        "bridge_marker_labels",
        raw.get("marker_labels", raw.get("hv_mark_labels", [])),
    )
    if isinstance(marker_labels_raw, Sequence) and not isinstance(marker_labels_raw, (str, bytes)):
        marker_values = tuple(normalize_block_key(v) for v in marker_labels_raw if normalize_block_key(v))
    elif marker_labels_raw in (None, ""):
        marker_values = ()
    else:
        norm = normalize_block_key(marker_labels_raw)
        marker_values = (norm,) if norm else ()

    expected_raw = raw.get(
        "expected_stage_frequencies_hz",
        raw.get("expected_frequencies_hz", raw.get("expected_stage_values", [])),
    )
    expected_vals: List[int] = []
    if isinstance(expected_raw, Sequence) and not isinstance(expected_raw, (str, bytes)):
        for value in expected_raw:
            try:
                expected_vals.append(int(value))
            except Exception:
                continue

    window_membership_cfg = raw.get("window_membership", {})
    if not isinstance(window_membership_cfg, Mapping):
        window_membership_cfg = {}

    return StageBlockingConfig(
        enabled=bool(raw.get("enabled", False)),
        stage_event_regex=str(
            raw.get(
                "stage_event_regex",
                raw.get(
                    "carrier_event_regex",
                    raw.get("photic_regex", ""),
                ),
            )
            or ""
        ).strip(),
        bridge_marker_labels=marker_values,
        use_bridge_markers=bool(raw.get("use_bridge_markers", raw.get("use_hv_marks", True))),
        preserve_block_assignments=bool(
            raw.get("preserve_block_assignments", raw.get("preserve_photic_blocks", True))
        ),
        min_block_sec=float(raw.get("min_block_sec", 2.0) or 2.0),
        max_block_sec=float(raw.get("max_block_sec", 20.0) or 20.0),
        bridge_tail_sec=float(raw.get("bridge_tail_sec", raw.get("hv_tail_sec", 0.5)) or 0.5),
        bridge_tail_cap_sec=float(raw.get("bridge_tail_cap_sec", raw.get("hv_tail_cap_sec", 1.0)) or 1.0),
        expected_stage_frequencies_hz=tuple(sorted(set(expected_vals))),
        window_membership=membership_spec_from_mapping(window_membership_cfg),
    )


def _compile_stage_block_regex(cfg: StageBlockingConfig) -> Optional[re.Pattern[str]]:
    if not cfg.stage_event_regex:
        return None
    try:
        return re.compile(cfg.stage_event_regex, flags=re.IGNORECASE)
    except Exception:
        return None


def match_stage_block_parameters(
    labels_raw: Sequence[Any],
    cfg: StageBlockingConfig,
) -> np.ndarray:
    """Return regex-captured numeric block parameters for raw labels."""
    out = np.full((len(labels_raw),), np.nan, dtype=np.float64)
    regex = _compile_stage_block_regex(cfg)
    if regex is None:
        return out
    for idx, raw in enumerate(labels_raw):
        match = regex.match(normalize_block_label(raw))
        if not match:
            continue
        try:
            out[idx] = float(match.group(1))
        except Exception:
            out[idx] = np.nan
    return out


def infer_stage_block_intervals(
    *,
    onsets: np.ndarray,
    durations: np.ndarray,
    labels_raw: Sequence[Any],
    stage_codes: np.ndarray,
    cfg: StageBlockingConfig,
) -> List[StageBlockInterval]:
    """Infer absolute stage-block intervals from sparse carrier and bridge events."""
    if not cfg.enabled:
        return []

    onsets_arr = np.asarray(onsets, dtype=float)
    durations_arr = np.asarray(durations, dtype=float)
    stage_codes_arr = np.asarray(stage_codes, dtype=float)
    labels_norm = np.asarray([normalize_block_label(v) for v in labels_raw], dtype=object)
    labels_key = np.asarray([normalize_block_key(v) for v in labels_raw], dtype=object)
    block_params = match_stage_block_parameters(labels_norm.tolist(), cfg)
    order = np.argsort(onsets_arr, kind="stable")
    carrier_indices = [
        int(idx)
        for idx in order.tolist()
        if np.isfinite(block_params[idx]) and np.isfinite(stage_codes_arr[idx])
    ]
    bridge_labels = set(cfg.bridge_marker_labels)

    intervals: List[StageBlockInterval] = []
    block_id = 0
    for pos, idx in enumerate(carrier_indices):
        start = float(onsets_arr[idx])
        next_start = float(onsets_arr[carrier_indices[pos + 1]]) if (pos + 1) < len(carrier_indices) else float("inf")
        end_direct = (
            float(start + durations_arr[idx])
            if np.isfinite(durations_arr[idx]) and float(durations_arr[idx]) > 0
            else float("nan")
        )
        end_bridge = float("nan")
        if cfg.use_bridge_markers and bridge_labels:
            bridge_mask = np.asarray([value in bridge_labels for value in labels_key.tolist()], dtype=bool)
            in_range = bridge_mask & np.isfinite(onsets_arr) & (onsets_arr >= start)
            if np.isfinite(next_start):
                in_range &= onsets_arr < next_start
            bridge_onsets = np.sort(np.asarray(onsets_arr[in_range], dtype=np.float64))
            if bridge_onsets.size:
                bridge_step = float("nan")
                if bridge_onsets.size >= 2:
                    diffs = np.diff(bridge_onsets)
                    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
                    if diffs.size:
                        bridge_step = float(np.median(diffs))
                tail = float(cfg.bridge_tail_sec)
                if np.isfinite(bridge_step) and bridge_step > 0:
                    tail = max(tail, min(bridge_step, cfg.bridge_tail_cap_sec))
                end_bridge = float(bridge_onsets[-1] + tail)

        end_reason = "unknown"
        if np.isfinite(end_direct) and end_direct > start:
            end = float(end_direct)
            end_reason = "carrier_duration"
        elif np.isfinite(end_bridge) and end_bridge > start:
            end = float(end_bridge)
            end_reason = "bridge_tail"
        elif np.isfinite(next_start):
            end = min(float(next_start), float(start + cfg.max_block_sec))
            end_reason = "next_carrier" if next_start <= float(start + cfg.max_block_sec) else "max_block_sec"
        else:
            end = float(start + cfg.max_block_sec)
            end_reason = "max_block_sec"

        if np.isfinite(next_start) and end > float(next_start):
            end = float(next_start)
            end_reason = "next_carrier"
        if end - start < cfg.min_block_sec:
            if np.isfinite(next_start) and (next_start - start) >= cfg.min_block_sec:
                end = min(float(next_start), float(start + cfg.max_block_sec))
                end_reason = "min_block_sec"
            else:
                end = float(start + cfg.min_block_sec)
                end_reason = "min_block_sec"
        if end <= start:
            end = float(start + max(cfg.min_block_sec, 1e-3))
            end_reason = "min_block_sec"

        support_mask = np.asarray([value in bridge_labels for value in labels_key.tolist()], dtype=bool)
        support_mask &= np.isfinite(onsets_arr) & (onsets_arr >= start) & (onsets_arr < end)
        support_indices = tuple(int(v) for v in np.where(support_mask)[0].tolist())
        intervals.append(
            StageBlockInterval(
                start_sec=start,
                end_sec=end,
                stage_code=int(stage_codes_arr[idx]),
                source_event_idx=int(idx),
                block_id=int(block_id),
                block_parameter=float(block_params[idx]),
                support_event_indices=support_indices,
                derived_from="stage_blocking",
                end_reason=end_reason,
                membership_mode=str(cfg.window_membership.mode),
                bridge_tail_sec=float(cfg.bridge_tail_sec),
                bridge_tail_cap_sec=float(cfg.bridge_tail_cap_sec),
                is_inferred=True,
            )
        )
        block_id += 1

    return intervals
