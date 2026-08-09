"""Backward-compatible ds003645 aliases for generic HDF5 lineage helpers."""

from ..provenance.h5_lineage import (
    REQUIRED_ROW_SOURCE_COLUMNS,
    SimultaneousFeatureRows,
    load_simultaneous_feature_rows,
    require_exact_window_match,
    required_feature_columns,
    window_key_frame,
)

__all__ = [
    "REQUIRED_ROW_SOURCE_COLUMNS",
    "SimultaneousFeatureRows",
    "load_simultaneous_feature_rows",
    "require_exact_window_match",
    "required_feature_columns",
    "window_key_frame",
]
