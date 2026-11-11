"""Sleep-EAP contracts with lazily loaded extraction primitives.

Importing a contract should not load every optional sleep estimator. In
particular, the spindle-strength implementation imports the general EEG stack
and its optional JIT-backed complexity dependency. Resolve public functions on
first use so non-sleep pipelines can read contract constants without paying
that startup cost.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .contracts import (
    EVENT_PHASE_N3_SO_V1_CONTRACT,
    EVENT_PHASE_REM_THETA_V1_CONTRACT,
    EVENT_PHASE_V3_CONTRACT,
    NON_EVENT_RISK_V1_CONTRACT,
    PHASE_CONTINUOUS_V1_CONTRACT,
    SLOW_OSCILLATION_V1_CONTRACT,
)

_LAZY_EXPORT_MODULES = {
    "build_n3_slow_oscillation_event_phase": ".event_phase_coupling",
    "build_rem_theta_event_phase": ".event_phase_coupling",
    "build_non_event_risk_catalog": ".non_event_risk",
    "PhaseContinuousState": ".phase_continuous",
    "build_phase_continuous_state": ".phase_continuous",
    "cardiac_phase_series": ".phase_continuous",
    "phase_state_to_frame": ".phase_continuous",
    "sample_phase_state": ".phase_continuous",
    "rem_theta_phase": ".rem_theta",
    "detect_slow_oscillations": ".slow_oscillation",
    "slow_oscillation_phase": ".slow_oscillation",
    "couple_spindles_to_slow_oscillations": ".so_spindle_coupling",
    "compute_spindle_strength": ".spindle_strength",
    "robust_z_against_reference": ".spindle_strength",
    "sigma_power_in_window": ".spindle_strength",
}

__all__ = [
    "EVENT_PHASE_V3_CONTRACT",
    "EVENT_PHASE_N3_SO_V1_CONTRACT",
    "EVENT_PHASE_REM_THETA_V1_CONTRACT",
    "NON_EVENT_RISK_V1_CONTRACT",
    "PHASE_CONTINUOUS_V1_CONTRACT",
    "SLOW_OSCILLATION_V1_CONTRACT",
    *_LAZY_EXPORT_MODULES,
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
