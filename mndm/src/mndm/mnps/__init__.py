"""MNPS chart namespace.

This is a compatibility-facing namespace for the canonical MNPS coordinate
chart.  The implementation remains in :mod:`mndm.projection`,
:mod:`mndm.jacobian`, and :mod:`mndm.schema` until a separately validated
refactor can move code without changing the public ``mndm.*`` import surface.
"""

from ..projection import project_features, project_features_v2, project_to_mnps
from ..schema import MNPS_AXIS_NAMES, mnps_9d_CANONICAL_ORDER

__all__ = [
    "MNPS_AXIS_NAMES",
    "mnps_9d_CANONICAL_ORDER",
    "project_features",
    "project_features_v2",
    "project_to_mnps",
]
