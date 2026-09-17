"""Standard non-MNPS dynamical measurement families.

This package is intentionally separate from :mod:`mndm.dynamics`, whose
measurements derive from local Jacobians.  Family objects require
family-specific support.  Destination and resilience still need protocol
inputs plus an adapter stamp before ``computed``.  Diffusion, when enabled,
computes if the estimator has support; OD-TQ1 tags in provenance are
method-validation, not a compute gate.
"""

from .one_step_operator import estimate_affine_one_step_family
from .amplification import estimate_neighbor_gain_q90
from .history import estimate_history_predictive_gain
from .turning import estimate_turning_rate
from .chart_drift import resolve_chart_drift, resolve_ingest_chart_drift
from .committor import estimate_committor, estimate_committor_local_law_dense_grid_o2b
from .contracts import (
    AFFINE_ONE_STEP_SCHEMA_VERSION,
    AMPLIFICATION_SCHEMA_VERSION,
    HISTORY_SCHEMA_VERSION,
    TURNING_SCHEMA_VERSION,
    CHART_DRIFT_SCHEMA_VERSION,
    COMMITTOR_SCHEMA_VERSION,
    DIFFUSION_GEOMETRY_SCHEMA_VERSION,
    FINITE_AMPLITUDE_RESILIENCE_SCHEMA_VERSION,
)
from .diffusion_geometry import estimate_local_diffusion_geometry
from .finite_lag_drift import (
    estimate_blocked_crossfit_mean_rate,
    estimate_chart_drift_family,
    estimate_conditional_mean_rate_level1,
    estimate_lag_diagnostics,
    estimate_realized_velocity_level0,
)
from .transition_support import (
    TransitionSupport,
    build_transition_support,
)
from .io import family_tree_present, read_inferential_grain, read_measurement_certificate, resolve_family_group
from .registry import FAMILIES, WRITABLE_FAMILY_IDS, family_forbids, get_family
from .resilience import summarize_finite_amplitude_resilience

__all__ = [
    "AFFINE_ONE_STEP_SCHEMA_VERSION",
    "AMPLIFICATION_SCHEMA_VERSION",
    "HISTORY_SCHEMA_VERSION",
    "TURNING_SCHEMA_VERSION",
    "CHART_DRIFT_SCHEMA_VERSION",
    "COMMITTOR_SCHEMA_VERSION",
    "DIFFUSION_GEOMETRY_SCHEMA_VERSION",
    "FAMILIES",
    "FINITE_AMPLITUDE_RESILIENCE_SCHEMA_VERSION",
    "TransitionSupport",
    "WRITABLE_FAMILY_IDS",
    "build_transition_support",
    "estimate_affine_one_step_family",
    "estimate_blocked_crossfit_mean_rate",
    "estimate_chart_drift_family",
    "estimate_committor",
    "estimate_committor_local_law_dense_grid_o2b",
    "estimate_conditional_mean_rate_level1",
    "estimate_lag_diagnostics",
    "estimate_local_diffusion_geometry",
    "estimate_neighbor_gain_q90",
    "estimate_history_predictive_gain",
    "estimate_turning_rate",
    "estimate_realized_velocity_level0",
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
