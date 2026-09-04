"""Versioned mathematical layers derived from estimated local Jacobians.

This package deliberately does not estimate Jacobians.  Estimation remains in
``mndm.jacobian``; these modules compute contract-versioned measurements from
the exported ``J_hat`` fields.
"""

from .jacobian_metrics import JACOBIAN_METRICS_SCHEMA_VERSION, compute_jacobian_metrics
from .transition_residuals import TRANSITION_RESIDUALS_SCHEMA_VERSION, compute_transition_residuals

__all__ = [
    "JACOBIAN_METRICS_SCHEMA_VERSION",
    "compute_jacobian_metrics",
    "TRANSITION_RESIDUALS_SCHEMA_VERSION",
    "compute_transition_residuals",
]
