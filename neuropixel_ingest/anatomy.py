"""NWB electrode-table and unit-anatomy enrichment helpers."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .contracts import AnatomyReadConfig


def normalize_location(value: Any, parser: str = "auto") -> str | None:
    """Return a conservative region token while preserving the raw label elsewhere."""
    text = _text(value)
    if not text or text.lower() in {"unknown", "none", "nan", "void", "root"}:
        return None
    if parser == "none":
        return None
    if ":" in text:
        for part in text.split(";"):
            key, _, candidate = part.partition(":")
            if key.strip().lower() in {"brain_region", "region", "location"} and candidate.strip():
                return candidate.strip().split()[0]
    return text.split()[0]


def read_probe_geometry(path: str | Path) -> pd.DataFrame:
    """Read portable electrode geometry rows directly from an NWB HDF5 file."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for NWB probe geometry.") from exc
    with h5py.File(Path(path), "r") as handle:
        root = handle.get("/general/extracellular_ephys/electrodes")
        if root is None:
            return pd.DataFrame()
        n_rows = _dataset_length(root.get("id"))
        if n_rows == 0:
            return pd.DataFrame()
        columns = {
            name: _read_column(root, name, n_rows)
            for name in (
                "id",
                "x",
                "y",
                "z",
                "location",
                "group_name",
                "imp",
                "probe_vertical_position",
                "probe_horizontal_position",
                "is_data_valid",
                "reference",
            )
        }
        rows: list[dict[str, Any]] = []
        xyz = np.column_stack(
            [pd.to_numeric(pd.Series(columns.get(name, [np.nan] * n_rows)), errors="coerce") for name in ("x", "y", "z")]
        )
        placeholder = bool(np.isfinite(xyz).any() and np.nanmax(np.abs(xyz)) == 0)
        for idx in range(n_rows):
            raw_location = _text(columns.get("location", [None] * n_rows)[idx]) or None
            probe_id = _text(columns.get("group_name", [None] * n_rows)[idx]) or "unknown"
            rows.append(
                {
                    "probe_id": probe_id,
                    "electrode_id": _as_int(columns.get("id", [idx] * n_rows)[idx], idx),
                    "x_um": _as_float(columns.get("x", [None] * n_rows)[idx]),
                    "y_um": _as_float(columns.get("y", [None] * n_rows)[idx]),
                    "z_um": _as_float(columns.get("z", [None] * n_rows)[idx]),
                    "location_raw": raw_location,
                    "location_acronym": normalize_location(raw_location),
                    "group_name": probe_id,
                    "impedance": _as_float(columns.get("imp", [None] * n_rows)[idx]),
                    "probe_vertical_position": _as_float(
                        columns.get("probe_vertical_position", [None] * n_rows)[idx]
                    ),
                    "probe_horizontal_position": _as_float(
                        columns.get("probe_horizontal_position", [None] * n_rows)[idx]
                    ),
                    "is_data_valid": _as_bool(columns.get("is_data_valid", [None] * n_rows)[idx]),
                    "reference": _text(columns.get("reference", [None] * n_rows)[idx]) or None,
                    "geometry_status": "placeholder" if placeholder else "available",
                }
            )
    return pd.DataFrame(rows)


def read_lfp_series_channels(path: str | Path, series_path: str) -> pd.DataFrame:
    """Map an NWB ElectricalSeries channel axis to its electrode-table rows."""
    geometry = read_probe_geometry(path)
    if geometry.empty:
        return geometry
    try:
        import h5py

        with h5py.File(Path(path), "r") as handle:
            group = handle[series_path.strip("/")]
            electrode_indices = np.asarray(group["electrodes"][:], dtype=int).reshape(-1)
    except Exception as exc:
        raise ValueError(f"Unable to resolve electrodes for NWB series {series_path!r}") from exc
    if (electrode_indices < 0).any() or (electrode_indices >= len(geometry)).any():
        raise ValueError(f"Series {series_path!r} references electrode rows outside the electrodes table")
    # A DynamicTableRegion stores table-row indices, not necessarily the table's
    # public ``id`` values.
    mapped = geometry.iloc[electrode_indices].reset_index(drop=True).copy()
    mapped["channel_index"] = np.arange(len(mapped), dtype=int)
    mapped["series_path"] = series_path
    return mapped


def enrich_units_anatomy(
    path: str | Path, units: pd.DataFrame, config: AnatomyReadConfig
) -> pd.DataFrame:
    """Attach electrode-derived location/probe/depth columns to units."""
    if units.empty:
        return units.copy()
    geometry = read_probe_geometry(path)
    output = units.copy()
    references = _read_unit_electrode_references(path, len(output))
    raw_locations: list[str | None] = []
    groups: list[str | None] = []
    for refs in references:
        selected = geometry.loc[geometry["electrode_id"].isin(refs)] if len(geometry) else geometry
        raw_locations.append(_aggregate(selected.get("location_raw", []), config.multi_electrode_policy))
        groups.append(_aggregate(selected.get("group_name", []), config.multi_electrode_policy))
    existing = output.get("location")
    if existing is not None:
        raw_locations = [
            _text(value) or candidate for value, candidate in zip(existing.tolist(), raw_locations, strict=True)
        ]
    output["location_raw"] = raw_locations
    output["location_acronym"] = [normalize_location(value, config.acronym_parser) for value in raw_locations]
    output["electrode_group"] = groups
    output["anatomy_source"] = np.where(
        pd.Series(raw_locations).notna(), "electrodes_table", "missing"
    )
    return output


def _read_unit_electrode_references(path: str | Path, n_units: int) -> list[list[int]]:
    try:
        import h5py
        with h5py.File(Path(path), "r") as handle:
            flat = np.asarray(handle["/units/electrodes"][:], dtype=int)
            ends = np.asarray(handle["/units/electrodes_index"][:], dtype=int)
    except Exception:
        return [[] for _ in range(n_units)]
    out: list[list[int]] = []
    start = 0
    for end in ends[:n_units]:
        out.append([int(value) for value in flat[start:int(end)]])
        start = int(end)
    out.extend([[] for _ in range(max(0, n_units - len(out)))])
    return out


def _aggregate(values: Iterable[Any], policy: str) -> str | None:
    valid = [_text(value) for value in values if _text(value)]
    if not valid:
        return None
    if policy == "first":
        return valid[0]
    if policy == "union":
        return "|".join(sorted(set(valid)))
    return Counter(valid).most_common(1)[0][0]


def _read_column(group: Any, name: str, n_rows: int) -> list[Any]:
    dataset = group.get(name)
    if dataset is None:
        return [None] * n_rows
    values = dataset[:]
    return list(values[:n_rows])


def _dataset_length(dataset: Any) -> int:
    return int(len(dataset)) if dataset is not None else 0


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip() if value is not None else ""


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    try:
        return bool(value)
    except (TypeError, ValueError):
        return None
