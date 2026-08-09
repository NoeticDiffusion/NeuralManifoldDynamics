"""Data contracts shared by NWB extracellular-electrophysiology readers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EphysReadConfig:
    """Controls schema-tolerant extraction of an NWB Units table."""

    quality_policy: str = "all"
    min_firing_rate_hz: float | None = None
    min_presence_ratio: float | None = None
    max_isi_violations_ratio: float | None = None
    include_unit_columns: tuple[str, ...] = ()
    anatomy: "AnatomyReadConfig" = field(default_factory=lambda: AnatomyReadConfig())


@dataclass(frozen=True)
class AnatomyReadConfig:
    """Controls optional anatomy and geometry enrichment from NWB electrodes."""

    location_paths: tuple[str, ...] = (
        "/general/localization/ElectrodesIBLBregma/beryl_location",
        "/general/extracellular_ephys/electrodes/location",
    )
    acronym_parser: str = "auto"
    multi_electrode_policy: str = "mode"
    allow_h5py_enrichment: bool = True


@dataclass(frozen=True)
class ProbeGeometry:
    """Portable row contract for one NWB electrode/channel."""

    probe_id: str
    electrode_id: int
    x_um: float | None = None
    y_um: float | None = None
    z_um: float | None = None
    location_raw: str | None = None
    location_acronym: str | None = None
    group_name: str | None = None
    impedance: float | None = None
    geometry_status: str = "missing"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class SessionBundle:
    """Portable representation of one NWB ephys session.

    ``units`` contains a ``spike_times`` column of one-dimensional seconds
    arrays.  The remaining tables preserve NWB-native trial/event metadata
    without imposing a task-specific schema.
    """

    units: pd.DataFrame
    trials: pd.DataFrame = field(default_factory=pd.DataFrame)
    events: Mapping[str, np.ndarray] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    probe_geometry: pd.DataFrame = field(default_factory=pd.DataFrame)
    unit_qc: Mapping[str, Any] = field(default_factory=dict)

    def spike_times(self) -> list[np.ndarray]:
        """Return finite, sorted spike-time arrays for every retained unit."""
        if "spike_times" not in self.units:
            return []
        result: list[np.ndarray] = []
        for values in self.units["spike_times"]:
            arr = np.asarray(values, dtype=float).reshape(-1)
            result.append(np.sort(arr[np.isfinite(arr)]))
        return result
