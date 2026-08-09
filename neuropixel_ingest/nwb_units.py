"""Schema-tolerant NWB Units, trials, and event extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .anatomy import enrich_units_anatomy, read_probe_geometry
from .contracts import EphysReadConfig, SessionBundle
from .quality import filter_units, normalize_quality_tier, unit_quality_summary

_COMMON_UNIT_COLUMNS = (
    "quality",
    "label",
    "kilosort2_label",
    "ibl_quality_score",
    "firing_rate",
    "spike_count",
    "presence_ratio",
    "isi_violations_ratio",
    "amplitude",
    "peak_to_trough_duration_ms",
    "depth",
    "cell_type",
    "distance_from_probe_tip_um",
    "probe_name",
    "electrode_group",
    "location",
)


def read_units(path: str | Path, config: EphysReadConfig = EphysReadConfig()) -> pd.DataFrame:
    """Read an NWB Units table into one row per unit.

    The required output is ``unit_id`` plus ``spike_times`` (seconds).  Common
    scalar metadata are retained whenever present; unknown requested columns
    are ignored so adapters can support lab-specific schemas.
    """
    path = Path(path)
    try:
        from pynwb import NWBHDF5IO
    except ImportError as exc:
        raise RuntimeError("pynwb is required to read NWB Units tables.") from exc

    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwbfile = io.read()
        table = getattr(nwbfile, "units", None)
        if table is None:
            return pd.DataFrame(columns=["unit_id", "spike_times"])
        try:
            unit_ids = np.asarray(table.id[:])
        except Exception:
            return pd.DataFrame(columns=["unit_id", "spike_times"])

        rows: dict[str, Any] = {"unit_id": unit_ids}
        spikes: list[np.ndarray] = []
        for index in range(len(unit_ids)):
            try:
                values = np.asarray(table["spike_times"][index], dtype=float).reshape(-1)
            except Exception:
                values = np.empty(0, dtype=float)
            spikes.append(np.sort(values[np.isfinite(values)]))
        rows["spike_times"] = spikes

        wanted = tuple(dict.fromkeys((*_COMMON_UNIT_COLUMNS, *config.include_unit_columns)))
        for name in wanted:
            values = _read_scalar_column(table, name, len(unit_ids))
            if values is not None:
                rows[name] = values

        units = pd.DataFrame(rows)
    units = enrich_units_anatomy(path, units, config.anatomy)
    if "depth" in units:
        units = units.rename(columns={"depth": "depth_um"})
    labels = units.get("quality", units.get("label", units.get("kilosort2_label", pd.Series(None, index=units.index))))
    scores = units.get("ibl_quality_score", pd.Series(None, index=units.index))
    units["quality_tier"] = [
        normalize_quality_tier(label, score) for label, score in zip(labels, scores, strict=True)
    ]
    return filter_units(units, config)


def read_trials(path: str | Path) -> pd.DataFrame:
    """Return all NWB trial columns without task-specific column assumptions."""
    path = Path(path)
    try:
        from pynwb import NWBHDF5IO
    except ImportError as exc:
        raise RuntimeError("pynwb is required to read NWB trial tables.") from exc
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwbfile = io.read()
        table = getattr(nwbfile, "trials", None)
        if table is None:
            return pd.DataFrame()
        return _dynamic_table_to_frame(table)


def read_session_bundle(
    path: str | Path,
    config: EphysReadConfig = EphysReadConfig(),
    *,
    dataset_id: str | None = None,
) -> SessionBundle:
    """Read a portable session bundle from an NWB file."""
    path = Path(path)
    raw_units = read_units(path, EphysReadConfig(
        quality_policy="all",
        min_firing_rate_hz=config.min_firing_rate_hz,
        min_presence_ratio=config.min_presence_ratio,
        max_isi_violations_ratio=config.max_isi_violations_ratio,
        include_unit_columns=config.include_unit_columns,
        anatomy=config.anatomy,
    ))
    units = filter_units(raw_units, config)
    trials = read_trials(path)
    events = _resolve_events(trials, dataset_id)
    geometry = read_probe_geometry(path)
    return SessionBundle(
        units=units,
        trials=trials,
        events=events,
        probe_geometry=geometry,
        unit_qc={
            "n_units_raw": len(raw_units),
            "n_units_retained": len(units),
            "quality_policy": config.quality_policy,
            "quality_summary": unit_quality_summary(raw_units),
        },
        metadata={"file": str(path), "source_format": "NWB", "n_units": len(units), "dataset_id": dataset_id},
    )


def read_unit_qc_inventory(path: str | Path, config: EphysReadConfig = EphysReadConfig()) -> pd.DataFrame:
    """Read a compact QC/anatomy inventory without extracting spike trains."""
    units = read_units(path, EphysReadConfig(
        quality_policy="all",
        include_unit_columns=config.include_unit_columns,
        anatomy=config.anatomy,
    ))
    return units.drop(columns=["spike_times"], errors="ignore")


def _read_scalar_column(table: Any, name: str, n_units: int) -> np.ndarray | None:
    try:
        column = table[name]
        values = np.asarray(column.data[:])
    except (KeyError, AttributeError, TypeError):
        return None
    if len(values) != n_units or values.ndim > 1:
        return None
    if values.dtype.kind == "S":
        return np.asarray([item.decode("utf-8", errors="replace") for item in values], dtype=object)
    return values


def _add_electrode_metadata(table: Any, units: pd.DataFrame) -> None:
    """Add unit electrode-group/location columns when pynwb exposes them."""
    if "location" in units:
        return
    try:
        electrodes = table["electrodes"]
    except (KeyError, AttributeError):
        return
    locations: list[str] = []
    groups: list[str] = []
    for index in range(len(units)):
        try:
            rows = electrodes[index]
            locations.append(_first_electrode_value(rows, "location", "unknown"))
            groups.append(_first_electrode_value(rows, "group_name", "unknown"))
        except Exception:
            locations.append("unknown")
            groups.append("unknown")
    units["location"] = locations
    units["electrode_group"] = groups


def _first_electrode_value(rows: Any, field: str, default: str) -> str:
    try:
        values = rows[field]
        value = values.iloc[0] if hasattr(values, "iloc") and len(values) else (values[0] if len(values) else default)
    except Exception:
        value = default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _dynamic_table_to_frame(table: Any) -> pd.DataFrame:
    names = [str(name) for name in getattr(table, "colnames", ())]
    rows: Mapping[str, Any] = {}
    for name in names:
        try:
            values = np.asarray(table[name].data[:])
        except (KeyError, AttributeError, TypeError):
            continue
        if values.dtype.kind == "S":
            values = np.asarray([item.decode("utf-8", errors="replace") for item in values], dtype=object)
        if values.ndim == 1:
            rows[name] = values
    return pd.DataFrame(rows)


def _resolve_events(trials: pd.DataFrame, dataset_id: str | None) -> dict[str, np.ndarray]:
    if trials.empty or not dataset_id:
        return {}
    try:
        from . import datasets  # noqa: F401 - import registers adapters
        from .registry import get_adapter

        adapter = get_adapter(dataset_id)
    except (ImportError, KeyError):
        return {}
    result: dict[str, np.ndarray] = {}
    for event_name, candidates in adapter.event_columns.items():
        column = next((name for name in candidates if name in trials), None)
        if column is None:
            continue
        values = pd.to_numeric(trials[column], errors="coerce").to_numpy(dtype=float)
        result[event_name] = values[np.isfinite(values)]
    return result
