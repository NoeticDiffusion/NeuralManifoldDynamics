"""Experimental MNPS 3.4 flat EEG coordinates.

This namespace is deliberately separate from the canonical MNPS v2 projection.
"""

from .candidates import compute_eeg_candidates, resolve_v3_4_config
from .coordinates import build_measurement_export, fit_anchor_manifest
from .meg_candidates import compute_meg_v3_4_features, resolve_meg_v3_4_config
from .meg_export import build_meg_measurement_export

__all__ = [
    "build_measurement_export",
    "build_meg_measurement_export",
    "compute_eeg_candidates",
    "compute_meg_v3_4_features",
    "fit_anchor_manifest",
    "resolve_meg_v3_4_config",
    "resolve_v3_4_config",
]
