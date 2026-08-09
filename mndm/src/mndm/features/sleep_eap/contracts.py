"""Versioned contracts for the RichSleep Sleep-EAP Phase 2 sidecars.

The contracts in this module describe data products only.  They deliberately do
not encode an inference model or an interpretation of the EAP hypothesis.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping


EVENT_PHASE_V3_CONTRACT = "rvc_v1_phase_resolved_v03"
PHASE_CONTINUOUS_V1_CONTRACT = "phase_continuous_v1"
NON_EVENT_RISK_V1_CONTRACT = "non_event_risk_v1"
SLOW_OSCILLATION_V1_CONTRACT = "slow_oscillation_v1"
EVENT_PHASE_N3_SO_V1_CONTRACT = "event_phase_n3_so_v1"
EVENT_PHASE_REM_THETA_V1_CONTRACT = "event_phase_rem_theta_v1"

SPINDLE_STRENGTH_COLUMNS = (
    "frequency_hz",
    "amplitude",
    "sigma_power",
    "sigma_power_z_n2",
    "yasa_amplitude",
    "yasa_abs_power",
    "yasa_rel_power",
    "strength_qc_flag",
)

SO_COUPLING_COLUMNS = (
    "so_onset_sec",
    "so_upstate_sec",
    "so_amplitude",
    "so_phase_at_spindle_peak",
    "so_spindle_latency_sec",
    "so_spindle_coupling_score",
    "so_partner_missing",
    "so_coupling_qc_flag",
)


def resolve_phase2_config(
    config: Mapping[str, Any] | None,
    dataset_id: str,
) -> Dict[str, Any]:
    """Resolve additive global and dataset-specific ``sleep_eap_phase2`` config."""
    root = config.get("sleep_eap_phase2", {}) if isinstance(config, Mapping) else {}
    if not isinstance(root, Mapping):
        return {"enabled": False}
    base = {key: value for key, value in root.items() if key != "datasets"}
    datasets = root.get("datasets", {})
    override = datasets.get(dataset_id, {}) if isinstance(datasets, Mapping) else {}
    if isinstance(override, Mapping):
        base.update(dict(override))
    base.setdefault("enabled", False)
    base.setdefault("event_phase_contract", EVENT_PHASE_V3_CONTRACT)
    base.setdefault("phase_continuous_contract", PHASE_CONTINUOUS_V1_CONTRACT)
    base.setdefault("non_event_risk_contract", NON_EVENT_RISK_V1_CONTRACT)
    base.setdefault("slow_oscillation_contract", SLOW_OSCILLATION_V1_CONTRACT)
    base.setdefault("event_phase_n3_so_contract", EVENT_PHASE_N3_SO_V1_CONTRACT)
    base.setdefault("event_phase_rem_theta_contract", EVENT_PHASE_REM_THETA_V1_CONTRACT)
    return base
