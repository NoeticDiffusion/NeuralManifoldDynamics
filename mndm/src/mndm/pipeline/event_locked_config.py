"""Config parser for the generic ``event_locked`` YAML section.

Reads the ``event_locked.datasets.<dataset_id>`` block from a loaded config dict
and returns typed objects consumed by the event-locked pipeline:

* ``EventLockedProfile``   — immutable provenance record, written to manifest
* ``AlignmentConfig``      — passed to ``align_events_to_windows()``
* ``MatchingConfig``       — passed to ``build_matched_controls()``
* ``ExportConfig``         — passed to ``build_event_locked_table()``

Design contract
---------------
* No I/O here. Reads only from already-loaded config dicts.
* All fields have sensible defaults; absent YAML keys never raise.
* Profile name and all analysis parameters appear verbatim in provenance.
* Stage filter stored as list[int] (converted from ["N2"] via stage_codebook).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .event_alignment import AlignmentConfig, BinSpec, DEFAULT_BINS
from .control_matching import MatchingConfig
from .event_locked_export import ExportConfig

# Default stage codebook (matches schema.py / features/eeg.py convention)
_DEFAULT_STAGE_CODEBOOK: Dict[str, int] = {
    "W": 0, "Wake": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "REM": 4, "R": 4,
}


@dataclass(frozen=True)
class EventLockedProfile:
    """Immutable provenance record for one event-locked analysis run.

    Written verbatim to ``run_manifest.json`` and repeated in every export row.

    Attributes
    ----------
    profile_name:
        The ``profile`` key from YAML — ties together epoching and event_locked configs.
    dataset_id:
        Dataset identifier.
    event_types:
        Which event types are selected from the EventTable.
    stage_filter_labels:
        Human-readable stage names from YAML (e.g. ``["N2"]``).
    stage_filter_codes:
        Corresponding integer codes used in pipeline comparisons.
    reference:
        Event timestamp used as ``t=0`` for bins (``"peak"``, ``"onset"``, ``"offset"``).
    bins_json:
        JSON-serialised ordered bin list for exact reproducibility.
    window_length_s:
        Epoch length used in MNDM run.
    window_step_s:
        Epoch step used in MNDM run.
    control_seed:
        RNG seed for matched control sampling.
    n_controls_per_event:
        Controls requested per event.
    control_exclusion_margin_sec:
        Exclusion zone around events during control sampling.
    """

    profile_name: str = "unknown"
    dataset_id: str = ""
    event_types: tuple = ()
    stage_filter_labels: tuple = ("N2",)
    stage_filter_codes: tuple = (2,)
    reference: str = "peak"
    bins_json: str = "[]"
    window_length_s: float = 6.0
    window_step_s: float = 2.0
    control_seed: int = 42
    n_controls_per_event: int = 3
    control_exclusion_margin_sec: float = 30.0
    event_source_kind: str = "csv"

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict for manifest / provenance export."""
        return {
            "profile_name": self.profile_name,
            "dataset_id": self.dataset_id,
            "event_source_kind": self.event_source_kind,
            "event_types": list(self.event_types),
            "stage_filter_labels": list(self.stage_filter_labels),
            "stage_filter_codes": list(self.stage_filter_codes),
            "reference": self.reference,
            "bins": json.loads(self.bins_json),
            "window_length_s": self.window_length_s,
            "window_step_s": self.window_step_s,
            "control_seed": self.control_seed,
            "n_controls_per_event": self.n_controls_per_event,
            "control_exclusion_margin_sec": self.control_exclusion_margin_sec,
        }


@dataclass(frozen=True)
class EventSourceConfig:
    """Configuration for resolving an EventTable for event-locked export."""

    kind: str = "csv"
    event_type: str = "sleep_spindle"
    event_kind: str = "end"
    stage_codes: tuple = ()
    block_parameters: tuple = ()
    source_label: str = "derived:stage_blocking"
    source_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "event_type": self.event_type,
            "event_kind": self.event_kind,
            "stage_codes": list(self.stage_codes),
            "block_parameters": list(self.block_parameters),
            "source_label": self.source_label,
            "source_path": self.source_path,
        }


def _resolve_event_locked_dataset_cfg(
    config: Mapping[str, Any],
    dataset_id: str,
) -> Dict[str, Any]:
    """Return the ``event_locked.datasets.<dataset_id>`` sub-dict, or ``{}``."""
    el = config.get("event_locked", {}) if isinstance(config, Mapping) else {}
    if not isinstance(el, Mapping):
        return {}
    datasets = el.get("datasets", {})
    if not isinstance(datasets, Mapping):
        return {}
    ds_cfg = datasets.get(dataset_id, {})
    return dict(ds_cfg) if isinstance(ds_cfg, Mapping) else {}


def _resolve_epoching_profile(
    config: Mapping[str, Any],
    dataset_id: str,
) -> Dict[str, Any]:
    """Return the ``epoching.datasets.<dataset_id>`` sub-dict, or ``{}``."""
    ep = config.get("epoching", {}) if isinstance(config, Mapping) else {}
    if not isinstance(ep, Mapping):
        return {}
    datasets = ep.get("datasets", {})
    if not isinstance(datasets, Mapping):
        return {}
    ds_cfg = datasets.get(dataset_id, {})
    return dict(ds_cfg) if isinstance(ds_cfg, Mapping) else {}


def _parse_stage_filter(
    labels: Sequence[str],
    codebook: Mapping[str, int],
) -> List[int]:
    """Convert stage label strings to integer codes via codebook."""
    codes = []
    for label in labels:
        key = str(label).strip()
        if key in codebook:
            codes.append(int(codebook[key]))
        else:
            # Try parsing as integer directly
            try:
                codes.append(int(key))
            except ValueError:
                pass
    return codes


def _parse_bins(raw_bins: Any) -> List[BinSpec]:
    """Convert YAML or JSON bin config to an ordered list of BinSpec."""
    if isinstance(raw_bins, list):
        result = []
        for item in raw_bins:
            if not isinstance(item, Mapping):
                continue
            try:
                result.append(BinSpec(str(item["label"]), float(item["lo"]), float(item["hi"])))
            except (KeyError, TypeError, ValueError):
                continue
        if result:
            return result
        return [BinSpec(label, lo, hi) for label, lo, hi in DEFAULT_BINS]

    if not isinstance(raw_bins, Mapping) or not raw_bins:
        return [BinSpec(label, lo, hi) for label, lo, hi in DEFAULT_BINS]
    result = []
    for label, bounds in raw_bins.items():
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            try:
                result.append(BinSpec(str(label), float(bounds[0]), float(bounds[1])))
            except (TypeError, ValueError):
                pass
    return result if result else [BinSpec(label, lo, hi) for label, lo, hi in DEFAULT_BINS]


def event_locked_profile_from_config(
    config: Mapping[str, Any],
    dataset_id: str,
    *,
    stage_codebook: Optional[Mapping[str, int]] = None,
) -> EventLockedProfile:
    """Build an ``EventLockedProfile`` from the loaded config dict.

    Parameters
    ----------
    config:
        Fully resolved config dict (after ``load_config()``).
    dataset_id:
        Which dataset's ``event_locked`` block to parse.
    stage_codebook:
        Optional override for stage-label → int mapping.
        Defaults to the standard W/N1/N2/N3/REM codebook.

    Returns
    -------
    EventLockedProfile
        Always returns a valid profile; absent YAML keys use defaults.
    """
    codebook = dict(stage_codebook) if stage_codebook else _DEFAULT_STAGE_CODEBOOK

    ds_cfg = _resolve_event_locked_dataset_cfg(config, dataset_id)
    ep_cfg = _resolve_epoching_profile(config, dataset_id)

    profile_name = str(
        ds_cfg.get("profile") or ep_cfg.get("profile") or "event_locked_default"
    )

    event_types_raw = ds_cfg.get("event_types", ["sleep_spindle"])
    event_types = tuple(str(t) for t in (event_types_raw if isinstance(event_types_raw, list) else [event_types_raw]))

    stage_filter_raw = ds_cfg.get("stage_filter", ["N2"])
    if not isinstance(stage_filter_raw, list):
        stage_filter_raw = [stage_filter_raw]
    stage_filter_labels = tuple(str(s) for s in stage_filter_raw)
    stage_filter_codes = tuple(_parse_stage_filter(stage_filter_labels, codebook))

    reference = str(ds_cfg.get("reference", "peak"))
    event_source_cfg = ds_cfg.get("event_source", {}) if isinstance(ds_cfg.get("event_source"), Mapping) else {}
    event_source_kind = str(event_source_cfg.get("kind", "csv") or "csv")

    raw_bins = ds_cfg.get("bins", {})
    bins = _parse_bins(raw_bins)
    bins_json = json.dumps(
        [{"label": b.label, "lo": b.lo, "hi": b.hi} for b in bins],
        ensure_ascii=False,
    )

    window_length_s = float(ep_cfg.get("length_s", 6.0))
    window_step_s = float(ep_cfg.get("step_s", 2.0))

    controls_cfg = ds_cfg.get("controls", {}) if isinstance(ds_cfg.get("controls"), Mapping) else {}
    control_seed = int(controls_cfg.get("seed", 42))
    n_controls = int(controls_cfg.get("n_controls_per_event", 3))
    exclusion_margin = float(controls_cfg.get("exclusion_margin_sec", 30.0))

    return EventLockedProfile(
        profile_name=profile_name,
        dataset_id=dataset_id,
        event_types=event_types,
        event_source_kind=event_source_kind,
        stage_filter_labels=stage_filter_labels,
        stage_filter_codes=stage_filter_codes,
        reference=reference,
        bins_json=bins_json,
        window_length_s=window_length_s,
        window_step_s=window_step_s,
        control_seed=control_seed,
        n_controls_per_event=n_controls,
        control_exclusion_margin_sec=exclusion_margin,
    )


def event_source_config_from_config(
    config: Mapping[str, Any],
    dataset_id: str,
    *,
    stage_codebook: Optional[Mapping[str, int]] = None,
) -> EventSourceConfig:
    """Build an ``EventSourceConfig`` from the loaded config dict."""
    codebook = dict(stage_codebook) if stage_codebook else _DEFAULT_STAGE_CODEBOOK
    ds_cfg = _resolve_event_locked_dataset_cfg(config, dataset_id)
    raw = ds_cfg.get("event_source", {}) if isinstance(ds_cfg.get("event_source"), Mapping) else {}

    kind = str(raw.get("kind", "csv") or "csv").strip().lower()
    if not kind:
        kind = "csv"

    event_types_raw = ds_cfg.get("event_types", ["sleep_spindle"])
    if not isinstance(event_types_raw, list):
        event_types_raw = [event_types_raw]
    configured_event_types = [str(v) for v in event_types_raw if str(v).strip()]
    default_event_type = "stage_block_end" if kind == "derived_stage_block_end" else "sleep_spindle"
    event_type = str(raw.get("event_type", configured_event_types[0] if configured_event_types else default_event_type) or default_event_type)

    event_kind = str(raw.get("event_kind", "end") or "end").strip().lower()

    stage_codes_raw = raw.get("stage_codes", raw.get("block_stage_codes", []))
    if not isinstance(stage_codes_raw, list):
        stage_codes_raw = [stage_codes_raw]
    stage_codes = tuple(_parse_stage_filter([str(v) for v in stage_codes_raw], codebook))

    block_parameters_raw = raw.get("block_parameters", raw.get("block_parameter_values", []))
    if not isinstance(block_parameters_raw, list):
        block_parameters_raw = [block_parameters_raw]
    block_parameters: List[float] = []
    for value in block_parameters_raw:
        try:
            block_parameters.append(float(value))
        except (TypeError, ValueError):
            continue

    return EventSourceConfig(
        kind=kind,
        event_type=event_type,
        event_kind=event_kind,
        stage_codes=tuple(stage_codes),
        block_parameters=tuple(block_parameters),
        source_label=str(raw.get("source_label", "derived:stage_blocking") or "derived:stage_blocking"),
        source_path=str(raw.get("source_path", "") or ""),
    )


def alignment_config_from_profile(
    profile: EventLockedProfile,
    config: Optional[Mapping[str, Any]] = None,
    dataset_id: str = "",
) -> AlignmentConfig:
    """Build an ``AlignmentConfig`` from an ``EventLockedProfile``.

    Parameters
    ----------
    profile:
        Parsed event-locked profile.
    config:
        Full config dict (for additional alignment overrides, optional).
    dataset_id:
        Used to look up per-dataset overrides.
    """
    bins = _parse_bins(json.loads(profile.bins_json))  # reconstruct from JSON

    margin = 30.0
    if config is not None:
        ds_cfg = _resolve_event_locked_dataset_cfg(config, dataset_id)
        # For direct BIDS event-locking the events ARE the state transitions,
        # so the sleep-spindle transition exclusion should default to 0.
        default_margin = 0.0 if profile.event_source_kind == "bids_events" else 30.0
        margin = float(ds_cfg.get("exclude_stage_transition_margin_sec", default_margin))
    elif profile.event_source_kind == "bids_events":
        margin = 0.0

    overlap_threshold = 0.0
    if config is not None:
        ds_cfg = _resolve_event_locked_dataset_cfg(config, dataset_id)
        overlap_threshold = float(ds_cfg.get("min_overlap_fraction", 0.0))

    return AlignmentConfig(
        reference=profile.reference,
        bins=bins,
        overlap_threshold=overlap_threshold,
        stage_transition_margin_sec=margin,
        stage_filter=list(profile.stage_filter_codes) if profile.stage_filter_codes else None,
    )


def matching_config_from_profile(profile: EventLockedProfile) -> MatchingConfig:
    """Build a ``MatchingConfig`` from an ``EventLockedProfile``."""
    target_stage = int(profile.stage_filter_codes[0]) if profile.stage_filter_codes else None
    return MatchingConfig(
        n_controls_per_event=profile.n_controls_per_event,
        target_stage=target_stage,
        exclusion_margin_sec=profile.control_exclusion_margin_sec,
        time_of_night_quartile_match=True,
        seed=profile.control_seed,
    )


def export_config_from_yaml(
    config: Mapping[str, Any],
    dataset_id: str,
) -> ExportConfig:
    """Build an ``ExportConfig`` from the ``event_locked.datasets.<id>.export`` block.

    Provenance fields (profile_name, window_length_s, etc.) are not populated here;
    use :func:`export_config_from_profile` when an ``EventLockedProfile`` is available.
    """
    ds_cfg = _resolve_event_locked_dataset_cfg(config, dataset_id)
    exp_cfg = ds_cfg.get("export", {}) if isinstance(ds_cfg.get("export"), Mapping) else {}
    return ExportConfig(
        write_parquet=bool(exp_cfg.get("write_parquet", True)),
        write_csv=bool(exp_cfg.get("write_csv", True)),
        include_coords_9d=bool(exp_cfg.get("include_coords_9d", True)),
        provenance_in_every_row=bool(exp_cfg.get("provenance_in_every_row", True)),
        event_condition_label=str(exp_cfg.get("event_condition_label", "event") or "event"),
        control_condition_label=str(exp_cfg.get("control_condition_label", "matched_control") or "matched_control"),
    )


def export_config_from_profile(
    profile: EventLockedProfile,
    config: Optional[Mapping[str, Any]] = None,
    dataset_id: str = "",
) -> ExportConfig:
    """Build an ``ExportConfig`` from an ``EventLockedProfile``.

    All provenance fields (profile_name, window_length_s, window_step_s, bins_json,
    alignment_reference, control_seed) are populated from the profile so they appear
    verbatim in every exported row.  I/O flags (write_parquet, write_csv, etc.) are
    read from the ``event_locked.datasets.<id>.export`` YAML block when provided.
    """
    io_cfg: ExportConfig
    if config is not None:
        io_cfg = export_config_from_yaml(config, dataset_id)
    else:
        io_cfg = ExportConfig()

    return ExportConfig(
        write_parquet=io_cfg.write_parquet,
        write_csv=io_cfg.write_csv,
        include_coords_9d=io_cfg.include_coords_9d,
        provenance_in_every_row=io_cfg.provenance_in_every_row,
        # Profile provenance — never None after this call.
        profile_name=profile.profile_name,
        window_length_s=profile.window_length_s,
        window_step_s=profile.window_step_s,
        bins_json=profile.bins_json,
        alignment_reference=profile.reference,
        control_seed=profile.control_seed,
        event_condition_label=io_cfg.event_condition_label,
        control_condition_label=io_cfg.control_condition_label,
    )


def is_event_locked_enabled(
    config: Mapping[str, Any],
    dataset_id: str,
) -> bool:
    """Return True if event-locked analysis is enabled for this dataset."""
    ds_cfg = _resolve_event_locked_dataset_cfg(config, dataset_id)
    return bool(ds_cfg.get("enabled", False))
