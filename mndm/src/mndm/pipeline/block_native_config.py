"""Config parsing and block-inference dispatch for block-native analysis mode.

This module is the single config-facing entry point for the block-native
pipeline.  It covers two responsibilities:

1. **Config parsing** — translates raw YAML dicts from the loaded config into
   typed dataclasses (``BlockWindowProfileConfig``, ``BlockSourceConfig``,
   ``BlockNativeDatasetConfig``).

2. **Block-inference dispatch** — ``infer_blocks_from_events()`` routes to the
   correct inference function based on ``source.kind``:

   - ``stage_blocking`` — re-uses the existing
     :func:`~mndm.pipeline.stage_blocking.infer_stage_block_intervals` pipeline.
   - ``duration_events`` — infers one block per event row with an explicit
     non-zero duration, filtered by label.  Suitable for datasets where
     task/condition blocks are stored as single long events (e.g. ds003490
     EO/EC resting blocks).
   - ``task_phase`` — groups consecutive events that share a configured label
     prefix into phase blocks.  A boundary is emitted when the prefix changes
     or the inter-event gap exceeds ``gap_tolerance_sec``.  Suitable for
     cognitive-task datasets where phases are distinguished by event-family
     prefixes (e.g. ``training_`` vs ``test_`` in ds003509).

Design contract
---------------
* No I/O here.  Reads only from already-loaded config dicts and
  ``pandas.DataFrame`` event tables.
* All fields have defaults; absent YAML keys never raise.
* Returns :class:`~mndm.pipeline.stage_blocking.StageBlockInterval` objects
  for all three source kinds so that downstream consumers work identically
  regardless of which source was used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .stage_blocking import (
    StageBlockInterval,
    infer_stage_block_intervals,
    normalize_block_key,
    stage_blocking_config_from_mapping,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

# Named profile presets used for block-native analysis.
# These are intentionally additive aliases over the existing primitive profile
# kinds (sliding/tail/post_offset/partitioned) so existing configs stay valid.
NAMED_BLOCK_NATIVE_WINDOW_PROFILES: Dict[str, Dict[str, Any]] = {
    "whole_block": {"kind": "sliding"},
    "early_block": {
        "kind": "partitioned",
        "partitions": {"early_block": [0.0, 8.0]},
    },
    "mid_block": {
        "kind": "partitioned",
        "partitions": {"mid_block": [8.0, 16.0]},
    },
    "late_block": {
        "kind": "partitioned",
        "partitions": {"late_block": [-8.0, 0.0]},
    },
    "last5": {"kind": "tail", "tail_sec": 5.0},
    "tail8": {"kind": "tail", "tail_sec": 8.0},
    "post_offset_0_8": {
        "kind": "post_offset",
        "post_offset_bins": {"post_offset_0_8": [0.0, 8.0]},
    },
    "post_offset_8_16": {
        "kind": "post_offset",
        "post_offset_bins": {"post_offset_8_16": [8.0, 16.0]},
    },
}


def available_named_window_profiles() -> Tuple[str, ...]:
    """Return the supported named block-native profile ids."""
    return tuple(sorted(NAMED_BLOCK_NATIVE_WINDOW_PROFILES.keys()))

@dataclass(frozen=True)
class BlockWindowProfileConfig:
    """How windows are generated from inferred blocks.

    Maps directly to the ``block_native.datasets.<id>.window_profile`` YAML key.
    """

    kind: str = "sliding"
    window_length_sec: float = 4.0
    step_sec: float = 2.0
    emit_relative_position: bool = True
    tail_sec: float = 8.0
    post_offset_bins: Tuple[Tuple[str, float, float], ...] = field(default_factory=tuple)
    partitions: Tuple[Tuple[str, float, float], ...] = field(default_factory=tuple)
    min_windows_per_block: int = 1
    min_block_sec: float = 0.0
    named_profile: str = ""


@dataclass(frozen=True)
class BlockSourceConfig:
    """Which block-inference strategy to use.

    Maps to the ``block_native.datasets.<id>.source`` YAML key.

    Attributes
    ----------
    kind:
        One of ``"stage_blocking"``, ``"duration_events"``, or
        ``"task_phase"``.
    block_event_labels:
        Labels to accept when ``kind == "duration_events"``.  Empty means
        accept all labeled events that have a matching stage code.
    block_event_stage_codes:
        Explicit label → stage_code mapping for ``"duration_events"`` kind.
        Stored as a tuple of ``(label, code)`` pairs.
    phase_prefixes:
        Prefix → phase_name pairs for ``"task_phase"`` kind.  A tuple of
        ``(phase_name, prefix)`` pairs.
    gap_tolerance_sec:
        Maximum inter-event gap before splitting a ``task_phase`` block.
    min_block_sec:
        Minimum block duration.  Shorter inferred blocks are discarded.
    max_block_sec:
        Maximum block duration cap.
    onset_column:
        Column name for event onsets in the events DataFrame.
    duration_column:
        Column name for event durations.
    label_column:
        Column name for event labels used by ``duration_events`` and
        ``task_phase`` kinds.
    """

    kind: str = "stage_blocking"
    block_event_labels: Tuple[str, ...] = field(default_factory=tuple)
    block_event_stage_codes: Tuple[Tuple[str, int], ...] = field(default_factory=tuple)
    phase_prefixes: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    gap_tolerance_sec: float = 5.0
    min_block_sec: float = 2.0
    max_block_sec: float = 600.0
    onset_column: str = "onset"
    duration_column: str = "duration"
    label_column: str = "trial_type"
    use_derived_task_state_segments: bool = False


@dataclass(frozen=True)
class BlockNativeDatasetConfig:
    """Top-level block-native config for one dataset."""

    enabled: bool = False
    source: BlockSourceConfig = field(default_factory=BlockSourceConfig)
    window_profile: BlockWindowProfileConfig = field(default_factory=BlockWindowProfileConfig)
    export_parquet: bool = True
    export_csv: bool = True


# ---------------------------------------------------------------------------
# Top-level helpers
# ---------------------------------------------------------------------------

def analysis_mode_from_config(config: Mapping[str, Any]) -> str:
    """Return the analysis mode: ``"global"`` (default) or ``"block_native"``."""
    if not isinstance(config, Mapping):
        return "global"
    return str(config.get("analysis_mode", "global") or "global").strip().lower()


def is_block_native_enabled(config: Mapping[str, Any], dataset_id: str) -> bool:
    """Return True if block-native mode is enabled for *dataset_id*."""
    return block_native_dataset_config_from_config(config, dataset_id).enabled


def block_native_dataset_config_from_config(
    config: Mapping[str, Any],
    dataset_id: str,
) -> BlockNativeDatasetConfig:
    """Parse ``block_native.datasets.<dataset_id>`` from a loaded config dict.

    Parameters
    ----------
    config:
        Fully resolved config dict (after ``load_config()``).
    dataset_id:
        Dataset identifier key.

    Returns
    -------
    BlockNativeDatasetConfig
        Always returns a valid config object; absent keys use defaults.
    """
    bn = config.get("block_native", {}) if isinstance(config, Mapping) else {}
    if not isinstance(bn, Mapping):
        return BlockNativeDatasetConfig()
    datasets = bn.get("datasets", {})
    if not isinstance(datasets, Mapping):
        return BlockNativeDatasetConfig()
    ds_cfg = datasets.get(dataset_id, {})
    if not isinstance(ds_cfg, Mapping):
        return BlockNativeDatasetConfig()

    enabled = bool(ds_cfg.get("enabled", False))

    source_raw = ds_cfg.get("source", {})
    source = _parse_block_source_config(
        source_raw if isinstance(source_raw, Mapping) else {}
    )

    profile_raw = ds_cfg.get("window_profile", {})
    profile = _parse_window_profile_config(
        profile_raw if isinstance(profile_raw, Mapping) else {}
    )

    export_raw = ds_cfg.get("export", {})
    export_parquet = True
    export_csv = True
    if isinstance(export_raw, Mapping):
        export_parquet = bool(export_raw.get("write_parquet", True))
        export_csv = bool(export_raw.get("write_csv", True))

    return BlockNativeDatasetConfig(
        enabled=enabled,
        source=source,
        window_profile=profile,
        export_parquet=export_parquet,
        export_csv=export_csv,
    )


# ---------------------------------------------------------------------------
# Internal config parsers
# ---------------------------------------------------------------------------

def _parse_block_source_config(raw: Mapping[str, Any]) -> BlockSourceConfig:
    kind = str(raw.get("kind", "stage_blocking") or "stage_blocking").strip().lower()

    raw_labels = raw.get("block_event_labels", [])
    if isinstance(raw_labels, str):
        raw_labels = [raw_labels]
    block_event_labels = tuple(
        str(v) for v in (raw_labels if isinstance(raw_labels, (list, tuple)) else [raw_labels])
        if str(v).strip()
    )

    stage_codes_raw = raw.get("block_event_stage_codes", {})
    block_event_stage_codes: List[Tuple[str, int]] = []
    if isinstance(stage_codes_raw, Mapping):
        for label, code in stage_codes_raw.items():
            try:
                block_event_stage_codes.append((str(label), int(code)))
            except (TypeError, ValueError):
                pass

    phase_prefixes_raw = raw.get("phase_prefixes", {})
    phase_prefixes: List[Tuple[str, str]] = []
    if isinstance(phase_prefixes_raw, Mapping):
        for name, prefix in phase_prefixes_raw.items():
            phase_prefixes.append((str(name), str(prefix)))

    return BlockSourceConfig(
        kind=kind,
        block_event_labels=block_event_labels,
        block_event_stage_codes=tuple(block_event_stage_codes),
        phase_prefixes=tuple(phase_prefixes),
        gap_tolerance_sec=float(raw.get("gap_tolerance_sec", 5.0) or 5.0),
        min_block_sec=float(raw.get("min_block_sec", 2.0) or 2.0),
        max_block_sec=float(raw.get("max_block_sec", 600.0) or 600.0),
        onset_column=str(raw.get("onset_column", "onset") or "onset"),
        duration_column=str(raw.get("duration_column", "duration") or "duration"),
        label_column=str(raw.get("label_column", "trial_type") or "trial_type"),
        use_derived_task_state_segments=bool(raw.get("use_derived_task_state_segments", False)),
    )


def _parse_window_profile_config(raw: Mapping[str, Any]) -> BlockWindowProfileConfig:
    profile_raw = dict(raw) if isinstance(raw, Mapping) else {}
    named_profile = str(
        profile_raw.get("profile", profile_raw.get("named_profile", profile_raw.get("preset", "")))
        or ""
    ).strip().lower()
    if named_profile:
        preset = NAMED_BLOCK_NATIVE_WINDOW_PROFILES.get(named_profile)
        if isinstance(preset, Mapping):
            merged_profile = dict(preset)
            for k, v in profile_raw.items():
                if k in {"profile", "named_profile", "preset"}:
                    continue
                merged_profile[k] = v
            profile_raw = merged_profile
        else:
            logger.warning(
                "Unknown block_native window_profile profile=%r; supported: %s",
                named_profile,
                ", ".join(available_named_window_profiles()),
            )

    kind = str(profile_raw.get("kind", "sliding") or "sliding").strip().lower()

    raw_bins = profile_raw.get("post_offset_bins", {})
    post_offset_bins: List[Tuple[str, float, float]] = []
    if isinstance(raw_bins, Mapping):
        for name, bounds in raw_bins.items():
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                try:
                    post_offset_bins.append((str(name), float(bounds[0]), float(bounds[1])))
                except (TypeError, ValueError):
                    pass
    elif isinstance(raw_bins, Sequence) and not isinstance(raw_bins, (str, bytes)):
        for entry in raw_bins:
            if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)) and len(entry) == 3:
                try:
                    post_offset_bins.append((str(entry[0]), float(entry[1]), float(entry[2])))
                except (TypeError, ValueError):
                    pass

    raw_parts = profile_raw.get("partitions", {})
    partitions: List[Tuple[str, float, float]] = []
    if isinstance(raw_parts, Mapping):
        for name, bounds in raw_parts.items():
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                try:
                    partitions.append((str(name), float(bounds[0]), float(bounds[1])))
                except (TypeError, ValueError):
                    pass
    elif isinstance(raw_parts, Sequence) and not isinstance(raw_parts, (str, bytes)):
        for entry in raw_parts:
            if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)) and len(entry) == 3:
                try:
                    partitions.append((str(entry[0]), float(entry[1]), float(entry[2])))
                except (TypeError, ValueError):
                    pass

    return BlockWindowProfileConfig(
        kind=kind,
        window_length_sec=float(profile_raw.get("window_length_sec", 4.0) or 4.0),
        step_sec=float(profile_raw.get("step_sec", 2.0) or 2.0),
        emit_relative_position=bool(profile_raw.get("emit_relative_position", True)),
        tail_sec=float(profile_raw.get("tail_sec", 8.0) or 8.0),
        post_offset_bins=tuple(post_offset_bins),
        partitions=tuple(partitions),
        min_windows_per_block=int(profile_raw.get("min_windows_per_block", 1) or 1),
        min_block_sec=float(profile_raw.get("min_block_sec", 0.0) or 0.0),
        named_profile=named_profile,
    )


# ---------------------------------------------------------------------------
# Block-inference dispatch (M2)
# ---------------------------------------------------------------------------

def infer_blocks_from_events(
    events_df: Any,
    source_cfg: BlockSourceConfig,
    *,
    stage_map: Optional[Mapping[str, int]] = None,
    sampling_cfg: Optional[Mapping[str, Any]] = None,
) -> List[StageBlockInterval]:
    """Infer block intervals from an events DataFrame.

    Dispatches to the correct inference function based on
    ``source_cfg.kind``.

    Parameters
    ----------
    events_df:
        ``pandas.DataFrame`` of BIDS events for one run.
    source_cfg:
        Block source configuration parsed from YAML.
    stage_map:
        Optional label → int mapping used by ``stage_blocking`` and
        ``duration_events`` kinds.
    sampling_cfg:
        Raw ``epoching.datasets.<id>.sampling`` dict, forwarded to the
        ``stage_blocking`` sub-function when the kind is ``"stage_blocking"``.

    Returns
    -------
    List[StageBlockInterval]
        Inferred block intervals in onset order.
    """
    kind = source_cfg.kind
    if kind == "stage_blocking":
        return _infer_stage_blocking(
            events_df, source_cfg,
            stage_map=stage_map, sampling_cfg=sampling_cfg,
        )
    if kind == "duration_events":
        return _infer_duration_events(events_df, source_cfg, stage_map=stage_map)
    if kind == "task_phase":
        return _infer_task_phase(events_df, source_cfg)
    logger.warning("Unknown block source kind %r — returning empty block list", kind)
    return []


# ---------------------------------------------------------------------------
# Kind-specific inference implementations
# ---------------------------------------------------------------------------

def _infer_stage_blocking(
    events_df: Any,
    source_cfg: BlockSourceConfig,
    *,
    stage_map: Optional[Mapping[str, int]] = None,
    sampling_cfg: Optional[Mapping[str, Any]] = None,
) -> List[StageBlockInterval]:
    """Delegate to the existing stage-blocking infrastructure."""
    try:
        import pandas as pd
    except ImportError:
        return []

    if not isinstance(events_df, pd.DataFrame) or events_df.empty:
        return []

    onset_col = source_cfg.onset_column
    dur_col = source_cfg.duration_column
    if onset_col not in events_df.columns:
        return []

    blocking_cfg = stage_blocking_config_from_mapping(
        sampling_cfg.get("stage_blocking", {})
        if isinstance(sampling_cfg, Mapping)
        else {}
    )
    if not blocking_cfg.enabled:
        return []

    onset = pd.to_numeric(events_df[onset_col], errors="coerce").to_numpy(dtype=float)
    duration = (
        pd.to_numeric(events_df[dur_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if dur_col in events_df.columns
        else np.zeros(len(events_df), dtype=float)
    )

    label_col = source_cfg.label_column
    if label_col not in events_df.columns:
        label_col = "value" if "value" in events_df.columns else events_df.columns[0]
    labels_raw = events_df[label_col].astype(str).fillna("").to_numpy(dtype=object)

    stage_codes = np.full(len(events_df), np.nan, dtype=np.float64)
    if stage_map:
        for idx, label in enumerate(labels_raw):
            key = normalize_block_key(label)
            if key in stage_map:
                stage_codes[idx] = float(stage_map[key])

    valid = np.isfinite(onset)
    return infer_stage_block_intervals(
        onsets=onset[valid],
        durations=duration[valid],
        labels_raw=labels_raw[valid].tolist(),
        stage_codes=stage_codes[valid],
        cfg=blocking_cfg,
    )


def _infer_duration_events(
    events_df: Any,
    source_cfg: BlockSourceConfig,
    *,
    stage_map: Optional[Mapping[str, int]] = None,
) -> List[StageBlockInterval]:
    """Infer one block per event with an explicit non-zero duration."""
    try:
        import pandas as pd
    except ImportError:
        return []

    if not isinstance(events_df, pd.DataFrame) or events_df.empty:
        return []

    onset_col = source_cfg.onset_column
    dur_col = source_cfg.duration_column
    label_col = source_cfg.label_column

    if onset_col not in events_df.columns:
        return []
    if label_col not in events_df.columns:
        return []

    label_to_code: Dict[str, int] = {}
    for label, code in source_cfg.block_event_stage_codes:
        label_to_code[normalize_block_key(label)] = int(code)
    if stage_map:
        for label, code in stage_map.items():
            label_to_code[normalize_block_key(label)] = int(code)

    target_label_keys = {normalize_block_key(v) for v in source_cfg.block_event_labels}

    onset = pd.to_numeric(events_df[onset_col], errors="coerce").to_numpy(dtype=float)
    duration = (
        pd.to_numeric(events_df[dur_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if dur_col in events_df.columns
        else np.zeros(len(events_df), dtype=float)
    )
    labels = events_df[label_col].astype(str).fillna("").to_numpy(dtype=object)

    intervals: List[StageBlockInterval] = []
    block_id = 0
    order = np.argsort(onset, kind="stable")

    for idx in order.tolist():
        if not np.isfinite(onset[idx]):
            continue
        label_key = normalize_block_key(str(labels[idx]))
        if target_label_keys and label_key not in target_label_keys:
            continue
        if label_key not in label_to_code:
            continue

        start = float(onset[idx])
        dur = float(duration[idx]) if np.isfinite(duration[idx]) and duration[idx] > 0 else 0.0
        if dur < source_cfg.min_block_sec:
            continue

        end = start + min(dur, source_cfg.max_block_sec)
        stage_code = label_to_code[label_key]

        intervals.append(
            StageBlockInterval(
                start_sec=start,
                end_sec=end,
                stage_code=stage_code,
                source_event_idx=int(idx),
                block_id=block_id,
                block_parameter=float("nan"),
                derived_from="duration_events",
                end_reason="carrier_duration",
                is_inferred=False,
            )
        )
        block_id += 1

    return intervals


def _infer_task_phase(
    events_df: Any,
    source_cfg: BlockSourceConfig,
) -> List[StageBlockInterval]:
    """Group prefix-matching events into task-phase blocks."""
    try:
        import pandas as pd
    except ImportError:
        return []

    if not isinstance(events_df, pd.DataFrame) or events_df.empty:
        return []

    onset_col = source_cfg.onset_column
    label_col = source_cfg.label_column

    if onset_col not in events_df.columns or label_col not in events_df.columns:
        return []
    if not source_cfg.phase_prefixes:
        return []

    prefix_map: List[Tuple[str, str, int]] = [
        (normalize_block_key(prefix), phase_name, phase_idx + 1)
        for phase_idx, (phase_name, prefix) in enumerate(source_cfg.phase_prefixes)
    ]

    onset = pd.to_numeric(events_df[onset_col], errors="coerce").to_numpy(dtype=float)
    labels = events_df[label_col].astype(str).fillna("").to_numpy(dtype=object)

    phase_assignments: List[Optional[Tuple[str, int]]] = [None] * len(events_df)
    for idx in range(len(events_df)):
        label_key = normalize_block_key(str(labels[idx]))
        for prefix, phase_name, stage_code in prefix_map:
            if label_key.startswith(prefix):
                phase_assignments[idx] = (phase_name, stage_code)
                break

    order = np.argsort(onset, kind="stable")
    intervals: List[StageBlockInterval] = []
    block_id = 0

    current_phase: Optional[str] = None
    current_stage_code: int = 0
    current_start: float = float("nan")
    current_end: float = float("nan")
    current_source_idx: int = 0
    support_indices: List[int] = []

    def _flush(end_sec: float) -> None:
        nonlocal block_id
        if current_phase is None or not np.isfinite(current_start):
            return
        dur = end_sec - current_start
        if dur < source_cfg.min_block_sec:
            return
        intervals.append(
            StageBlockInterval(
                start_sec=current_start,
                end_sec=end_sec,
                stage_code=current_stage_code,
                source_event_idx=current_source_idx,
                block_id=block_id,
                block_parameter=float("nan"),
                support_event_indices=tuple(support_indices),
                derived_from="task_phase",
                end_reason="phase_end",
                is_inferred=True,
            )
        )
        block_id += 1

    for idx in order.tolist():
        if not np.isfinite(onset[idx]):
            continue
        assignment = phase_assignments[idx]
        if assignment is None:
            continue

        phase_name, stage_code = assignment
        event_onset = float(onset[idx])

        gap_too_large = (
            current_phase is not None
            and np.isfinite(current_end)
            and (event_onset - current_end) > source_cfg.gap_tolerance_sec
        )

        if current_phase is None:
            current_phase = phase_name
            current_stage_code = stage_code
            current_start = event_onset
            current_end = event_onset
            current_source_idx = int(idx)
            support_indices = []
        elif phase_name != current_phase or gap_too_large:
            _flush(current_end)
            current_phase = phase_name
            current_stage_code = stage_code
            current_start = event_onset
            current_end = event_onset
            current_source_idx = int(idx)
            support_indices = []
        else:
            support_indices.append(int(idx))
            current_end = event_onset

    if current_phase is not None:
        _flush(current_end)

    return intervals
