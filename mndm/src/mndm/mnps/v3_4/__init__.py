"""Experimental MNPS 3.4 flat EEG coordinates.

This namespace is deliberately separate from the canonical MNPS v2 projection.
"""

from .candidates import compute_eeg_candidates, resolve_v3_4_config
from .coordinates import build_measurement_export, fit_anchor_manifest

__all__ = [
    "build_measurement_export",
    "compute_eeg_candidates",
    "fit_anchor_manifest",
    "resolve_v3_4_config",
]
