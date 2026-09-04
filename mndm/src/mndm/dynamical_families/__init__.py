"""Standard non-MNPS dynamical measurement families.

This package is intentionally separate from :mod:`mndm.dynamics`, whose
measurements derive from local Jacobians.  Family objects require
family-specific support.  Destination and resilience still need protocol
inputs plus an adapter stamp before ``computed``.  Diffusion, when enabled,
computes if the estimator has support; OD-TQ1 tags in provenance are
method-validation, not a compute gate.
"""

from .chart_drift import resolve_chart_drift, resolve_ingest_chart_drift
from .committor import estimate_committor, estimate_committor_local_law_dense_grid_o2b
from .contracts import (
    COMMITTOR_SCHEMA_VERSION,
    DIFFUSION_GEOMETRY_SCHEMA_VERSION,
    FINITE_AMPLITUDE_RESILIENCE_SCHEMA_VERSION,
)
from .diffusion_geometry import estimate_local_diffusion_geometry
from .io import family_tree_present, read_inferential_grain, read_measurement_certificate, resolve_family_group
from .registry import FAMILIES, WRITABLE_FAMILY_IDS, family_forbids, get_family
from .resilience import summarize_finite_amplitude_resilience

__all__ = [
    "COMMITTOR_SCHEMA_VERSION",
    "DIFFUSION_GEOMETRY_SCHEMA_VERSION",
    "FAMILIES",
    "FINITE_AMPLITUDE_RESILIENCE_SCHEMA_VERSION",
    "WRITABLE_FAMILY_IDS",
    "estimate_committor",
    "estimate_committor_local_law_dense_grid_o2b",
    "estimate_local_diffusion_geometry",
    "family_forbids",
    "family_tree_present",
    "get_family",
    "read_inferential_grain",
    "read_measurement_certificate",
    "resolve_chart_drift",
    "resolve_family_group",
    "resolve_ingest_chart_drift",
    "summarize_finite_amplitude_resilience",
]
