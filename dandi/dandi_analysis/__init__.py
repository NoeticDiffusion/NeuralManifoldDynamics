"""
DANDI Analysis Tools.

Utilities for extracting and analyzing neural activity from DANDI NWB assets,
specifically focusing on coreactivation and replay-like window identification.
"""

from dandi_analysis.contracts import (
    ActivityMatrix,
    DiscoveredNwbAsset,
    OfflineWindow,
    PairwiseCoreactivationResult,
    QcIssue,
    ReadyNwbAsset,
    SessionIndexRow,
)
from dandi_analysis.inventory import (
    build_inventory_report,
    canonical_assets,
    discover_nwb_assets,
    duplicate_assets,
)
from dandi_analysis.readiness import (
    build_readiness_report,
    check_readiness,
    filter_ready,
)

__all__ = [
    "ActivityMatrix",
    "DiscoveredNwbAsset",
    "OfflineWindow",
    "PairwiseCoreactivationResult",
    "QcIssue",
    "ReadyNwbAsset",
    "SessionIndexRow",
    "build_inventory_report",
    "build_readiness_report",
    "canonical_assets",
    "check_readiness",
    "discover_nwb_assets",
    "duplicate_assets",
    "filter_ready",
]
