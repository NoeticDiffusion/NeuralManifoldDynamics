"""Sleep-EAP Phase 2 extraction primitives and versioned data contracts."""

from .contracts import (
    EVENT_PHASE_N3_SO_V1_CONTRACT,
    EVENT_PHASE_REM_THETA_V1_CONTRACT,
    EVENT_PHASE_V3_CONTRACT,
    NON_EVENT_RISK_V1_CONTRACT,
    PHASE_CONTINUOUS_V1_CONTRACT,
    SLOW_OSCILLATION_V1_CONTRACT,
)
from .event_phase_coupling import build_n3_slow_oscillation_event_phase, build_rem_theta_event_phase
from .non_event_risk import build_non_event_risk_catalog
from .phase_continuous import (
    PhaseContinuousState,
    build_phase_continuous_state,
    cardiac_phase_series,
    phase_state_to_frame,
    sample_phase_state,
)
from .rem_theta import rem_theta_phase
from .slow_oscillation import detect_slow_oscillations, slow_oscillation_phase
from .so_spindle_coupling import couple_spindles_to_slow_oscillations
from .spindle_strength import compute_spindle_strength, robust_z_against_reference, sigma_power_in_window

__all__ = [
    "EVENT_PHASE_V3_CONTRACT",
    "EVENT_PHASE_N3_SO_V1_CONTRACT",
    "EVENT_PHASE_REM_THETA_V1_CONTRACT",
    "NON_EVENT_RISK_V1_CONTRACT",
    "PHASE_CONTINUOUS_V1_CONTRACT",
    "SLOW_OSCILLATION_V1_CONTRACT",
    "PhaseContinuousState",
    "build_non_event_risk_catalog",
    "build_n3_slow_oscillation_event_phase",
    "build_rem_theta_event_phase",
    "build_phase_continuous_state",
    "cardiac_phase_series",
    "compute_spindle_strength",
    "couple_spindles_to_slow_oscillations",
    "detect_slow_oscillations",
    "phase_state_to_frame",
    "rem_theta_phase",
    "robust_z_against_reference",
    "sample_phase_state",
    "sigma_power_in_window",
    "slow_oscillation_phase",
]
