"""Canonical dynamical-family registry and dependency DAG.

This is the Round-1 semantic kernel: identifiers, schemas, namespaces,
dependencies, and forbidden substitutions. Round 3 stores default grain
on each family as documentation metadata; writers still go through
``attach_grain``. Support signatures remain a later round.
"""

from __future__ import annotations

from typing import Any, Mapping

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


STOCHASTIC_REACHABILITY_SCHEMA_VERSION = "mndm.stochastic_reachability.v1"

CANONICAL_HDF5_ROOT = "dynamical_families"
LEGACY_HDF5_ROOT = "orthogonal_dynamics"

FAMILIES: dict[str, dict[str, Any]] = {
    "diffusion": {
        "canonical_id": "diffusion",
        "family_object": "diffusion_geometry",
        "schema": DIFFUSION_GEOMETRY_SCHEMA_VERSION,
        "namespace": f"/{CANONICAL_HDF5_ROOT}/diffusion/v1",
        "legacy_namespace": f"/{LEGACY_HDF5_ROOT}/diffusion_geometry/v1",
        "implementation": "landed",
        "default_enabled": False,
        "requires": ("chart_trajectory", "transition_increments"),
        "forbids": (
            "jacobian_derivative_residual_as_diffusion",
            "mnps_xdot_as_sde_drift",
            "jacobian_intercept_as_sde_drift",
            "local_increment_mean_as_sde_drift",
            "increment_covariance_level0",
            "ito_diffusion_tensor_level3",
            "ito_qualified",
            "diffusion_effective_dimension_level1",
            "diffusion_condition_number_level1",
            "diffusion_directional_entropy_level1",
            "d_diff_as_diffusion_effective_dimension_level1",
        ),
        "grain": {
            "native": "window",
            "parent": "recording",
            "repeated_measure": "true",
        },
    },
    "spread": {
        "canonical_id": "spread",
        "family_object": "stochastic_reachability",
        "schema": STOCHASTIC_REACHABILITY_SCHEMA_VERSION,
        # Gate F frozen 2026-09-02: family YAML key `spread` stays refused.
        # Opt-in W_Q is local_dynamics.stochastic_reachability and writes
        # /stochastic_reachability/v1, not /dynamical_families/spread.
        "namespace": None,
        "out_of_family_namespace": "/stochastic_reachability/v1",
        "legacy_namespace": None,
        "implementation": "gate_closed",
        "gate": "F",
        "default_enabled": False,
        "requires": ("local_transition_operator", "admissible_process_covariance"),
        "forbids": (
            "derivative_residual_covariance",
            "gate_e_proxy_as_process_noise",
            "observed_future_spread_as_level0",
            "w_q_as_empirical_future_covariance",
            "w_q_as_controllability",
            "w_q_as_occupancy",
            "w_q_as_generator_spread",
            "w_q_as_conditional_future_spread_level1",
            "multi_step_w_q_as_level2",
            "one_step_w_q_as_level4",
            "observed_future_spread_level0",
            "observed_future_effective_dimension_level0",
            "observed_future_log_volume_level0",
            "observed_future_anisotropy_level0",
            "conditional_future_spread_level1",
            "transition_reachability_covariance_level2",
            "transition_reachability_effective_dimension_level2",
            "transition_predictive_spread_level2",
            "generator_predictive_spread_level3",
            "finite_time_reachability_level4",
            "reachability_effective_dimension_level4",
            "reachability_anisotropy_level4",
            "d_eff_as_reachability_effective_dimension_level4",
        ),
        "grain": {
            "native": "recording_horizon",
            "parent": "recording",
            "repeated_measure": "true",
        },
    },
    "persistence": {
        "canonical_id": "persistence",
        "family_object": "region_persistence",
        "schema": None,
        "namespace": None,
        "legacy_namespace": None,
        "implementation": "withheld",
        "default_enabled": False,
        "forbids": (
            "visual_clustering_as_attractor",
            "state_return_probability_as_attractor",
            "region_survival_as_attractor",
            "transformed_retention_as_escape_rate",
            "attractor",
            "basin",
            "persistence",
            "recurrence",
            "state_geometry",
            "basin_attractor_geometry_level4",
            "state_return_probability_level0",
            "local_recurrence_rate_level0",
            "region_survival_probability_level1",
            "region_dwell_time_level1",
            "transition_self_retention_level2",
            "transition_escape_rate_level2",
            "state_recurrence_level0",
            "residence_persistence_level1",
            "transition_metastability_level2",
        ),
    },
    "hysteresis": {
        "canonical_id": "hysteresis",
        "family_object": "observational_recovery",
        "schema": None,
        "namespace": None,
        "legacy_namespace": None,
        "implementation": "withheld",
        "default_enabled": False,
        "forbids": (
            "hysteresis_as_far",
            "return_distance_as_far",
            "path_alignment_as_level2_model",
            "matched_return_as_level2_model",
            "recovery_time_as_far_recovery_time",
            "descriptive_return_as_intervention",
            "hysteresis",
            "recovery",
            "return_distance_level0",
            "matched_return_distance_level1",
            "recovery_time_level1",
            "induction_recovery_path_asymmetry_level2",
            "induction_recovery_path_asymmetry_level1",
        ),
    },
    "destination": {
        "canonical_id": "destination",
        "family_object": "committor",
        "schema": COMMITTOR_SCHEMA_VERSION,
        "namespace": f"/{CANONICAL_HDF5_ROOT}/destination/v1",
        "legacy_namespace": f"/{LEGACY_HDF5_ROOT}/committor/v1",
        "implementation": "restricted",
        "default_enabled": False,
        "requires": (
            "explicit_reaction_coordinate",
            "explicit_set_A",
            "explicit_set_B",
            "first_hit_certificate",
        ),
        "forbids": (
            "stage_labels_as_destination_truth",
            "computed_q_as_generator_committor",
            "state_conditioned_first_hit_as_level0",
            "committor",
            "generator_committor_level3",
            "transition_model_first_hit_probability_level2",
            "destination_first_hit_fraction_level0",
            "destination_hit_probability_level1",
            "destination_unresolved_fraction_level1",
            "transition_model_first_hit_probability_level4",
            "resolved_q_as_destination_hit_probability_level1",
            "unresolved_outcomes_as_destination_unresolved_fraction_level1",
            "include_unresolved_in_first_hit_q",
        ),
        "grain": {
            "native": "window",
            "parent": "recording",
            "repeated_measure": "true",
        },
    },
    "resilience": {
        "canonical_id": "resilience",
        "family_object": "finite_amplitude_resilience",
        "schema": FINITE_AMPLITUDE_RESILIENCE_SCHEMA_VERSION,
        "namespace": f"/{CANONICAL_HDF5_ROOT}/resilience/v1",
        "legacy_namespace": f"/{LEGACY_HDF5_ROOT}/finite_amplitude_resilience/v1",
        "implementation": "serializer_only",
        "default_enabled": False,
        "requires": (
            "perturbation_amplitude",
            "outcome_definition",
            "recovery_or_survival_criterion",
        ),
        "forbids": (
            "inference_from_spontaneous_trajectories",
            "inference_from_jacobian",
            "inference_from_finite_time_response",
            "spontaneous_return_as_far",
            "matched_perturbation_as_level2_model",
            "discrete_r50_as_far_threshold_p50",
            "amplitude_curve_as_spontaneous_return",
            "r50_as_far_threshold_p50",
            "spontaneous_return_fraction_level0",
            "excursion_recovery_probability_level1",
            "excursion_recovery_time_median_level1",
            "matched_perturbation_recovery_level2",
            "matched_perturbation_recovery_level1",
            "far_threshold_p50_level4",
            "far_threshold_p90_level4",
        ),
        "grain": {
            "native": "event",
            "parent": "recording",
            "repeated_measure": "true",
        },
    },
    "drift": {
        "canonical_id": "drift",
        "family_object": "chart_drift",
        "schema": CHART_DRIFT_SCHEMA_VERSION,
        "namespace": f"/{CANONICAL_HDF5_ROOT}/drift/v1",
        "legacy_namespace": None,
        "implementation": "landed",
        "default_enabled": False,
        "requires": ("chart_trajectory", "transition_increments"),
        "forbids": (
            "mnps_xdot_as_sde_drift",
            "jacobian_intercept_as_sde_drift",
            "local_increment_mean_as_sde_drift",
            "ito_qualified_auto_promotion",
            "ito_drift_level3",
            "ito_qualified",
            "crossfit_local_chart_b",
            "source_crossfit_as_independent_b",
        ),
        "grain": {
            "native": "window",
            "parent": "recording",
            "repeated_measure": "true",
        },
    },
    "one_step": {
        "canonical_id": "one_step",
        "family_object": "affine_one_step_operator",
        "schema": AFFINE_ONE_STEP_SCHEMA_VERSION,
        "namespace": f"/{CANONICAL_HDF5_ROOT}/one_step/v1",
        "legacy_namespace": None,
        "implementation": "landed",
        "default_enabled": False,
        "requires": ("chart_trajectory", "transition_increments"),
        "forbids": (
            "mnps_xdot_as_one_step_map",
            "jacobian_expm_as_fitted_phi",
            "one_step_iteration_as_level2",
            "phi_one_composed_as_lag2",
            "affine_two_step",
            "two_step",
            "affine_two_step_map_level2",
            "ito_qualified_auto_promotion",
            "ito_drift_level3",
            "ito_qualified",
            "abscissa_from_unidentified_operator",
            "innovation_as_diffusion_a_hat",
            "peak_gain_from_phi_powers",
            "finite_time_peak_gain_level4",
            "operator_gain_as_spectral_abscissa",
            "epsilon_rescued_volume_gain",
            "reactivity_gap_level3",
            "abscissa_difference_as_reactivity_gap",
            "jacobian_reactivity_gap_as_level3",
            "operator_gain_anisotropy_level2",
            "generator_symmetric_anisotropy_level3",
        ),
        "grain": {
            "native": "recording",
            "parent": "recording",
            "repeated_measure": "false",
        },
    },
    "amplification": {
        "canonical_id": "amplification",
        "family_object": "neighbor_gain",
        "schema": AMPLIFICATION_SCHEMA_VERSION,
        "namespace": f"/{CANONICAL_HDF5_ROOT}/amplification/v1",
        "legacy_namespace": None,
        "implementation": "landed",
        "default_enabled": False,
        "requires": ("chart_trajectory", "transition_increments"),
        "forbids": (
            "resample_neighbors_at_target",
            "neighbor_gain_as_operator_max_gain",
            "neighbor_gain_as_spectral_abscissa",
            "history_predictive_gain_level1",
            "neighbor_separation_rate_level1",
            "neighbor_gain_rate_q90_level1",
            "cloud_volume_change_rate_level1",
            "neighbor_separation_as_spectral_abscissa",
            "cloud_volume_as_operator_volume",
        ),
        "grain": {
            "native": "window",
            "parent": "recording",
            "repeated_measure": "true",
        },
    },
    "history": {
        "canonical_id": "history",
        "family_object": "history_predictive_gain",
        "schema": HISTORY_SCHEMA_VERSION,
        "namespace": f"/{CANONICAL_HDF5_ROOT}/history/v1",
        "legacy_namespace": None,
        "implementation": "landed",
        "default_enabled": False,
        "requires": ("chart_trajectory", "transition_increments"),
        "forbids": (
            "history_as_markov_restoration",
            "history_conditioned_operator_level2",
            "history_augmented_generator_level3",
            "history_augmented_propagator_level4",
            "mnps_xdot_as_history_model",
            "one_step_phi_as_history_m0",
            "logm_of_history_m1",
            "history_m1_as_ito_drift",
            "m1_iteration_as_history_propagator",
        ),
        "grain": {
            "native": "recording",
            "parent": "recording",
            "repeated_measure": "false",
        },
    },
    "turning": {
        "canonical_id": "turning",
        "family_object": "turning_rate",
        "schema": TURNING_SCHEMA_VERSION,
        "namespace": f"/{CANONICAL_HDF5_ROOT}/turning/v1",
        "legacy_namespace": None,
        "implementation": "landed",
        "default_enabled": False,
        "requires": ("chart_trajectory", "transition_increments"),
        "forbids": (
            "operator_rotation_as_turning",
            "generator_rotation_as_turning",
            "mnps_xdot_as_turning",
            "zero_fill_undefined_direction",
            "cloud_volume_change_rate_level1",
        ),
        "grain": {
            "native": "window",
            "parent": "recording",
            "repeated_measure": "true",
        },
    },
}

WRITABLE_FAMILY_IDS = (
    "diffusion",
    "destination",
    "resilience",
    "drift",
    "one_step",
    "amplification",
    "history",
    "turning",
)

LEGACY_FAMILY_HDF5_NAMES = {
    "diffusion": "diffusion_geometry",
    "destination": "committor",
    "resilience": "finite_amplitude_resilience",
}

LEGACY_FAMILY_CONFIG_KEYS = {
    "diffusion_geometry": "diffusion",
    "committor": "destination",
    "finite_amplitude_resilience": "resilience",
}


def get_family(family_id: str) -> dict[str, Any]:
    """Return a copy of one family registry record."""
    try:
        return dict(FAMILIES[family_id])
    except KeyError as exc:
        raise KeyError(f"Unknown dynamical family: {family_id}") from exc


def family_config(root: Mapping[str, Any] | None, family_id: str) -> dict[str, Any]:
    """Return the config mapping for one canonical family id."""
    if not isinstance(root, Mapping):
        return {}
    block = root.get(family_id, {})
    return dict(block) if isinstance(block, Mapping) else {}


def family_forbids(family_id: str, substitution: str) -> bool:
    """Return True when ``substitution`` is a forbidden input for the family."""
    return substitution in get_family(family_id)["forbids"]


def validate_writable_family_payload(mapping: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a copy of a family payload that may be written to HDF5.

    Unknown keys, Gate F ``spread``, and pre-v3 nested names are refused.
    """
    if not isinstance(mapping, Mapping):
        return {}
    out = dict(mapping)
    if "spread" in out:
        raise ValueError(
            "Family 'spread' is gate_closed (Gate F) and is not written under "
            "/dynamical_families."
        )
    withheld = [
        key
        for key in (
            "attractor",
            "basin",
            "persistence",
            "recurrence",
            "hysteresis",
            "recovery",
        )
        if key in out
    ]
    if withheld:
        raise ValueError(
            "dynamical_families payload keys "
            f"{withheld} are withheld; ingest does not write attractor, basin, "
            "persistence, recurrence, hysteresis, or recovery groups. "
            "Recurrence is not an attractor. Hysteresis is not FAR."
        )
    for old, new in LEGACY_FAMILY_CONFIG_KEYS.items():
        if old in out:
            raise ValueError(
                f"dynamical_families payload key '{old}' was renamed to '{new}'. "
                "No alias is provided."
            )
    unknown = [str(key) for key in out if str(key) not in WRITABLE_FAMILY_IDS]
    if unknown:
        raise ValueError(
            "dynamical_families payload keys must be one of "
            f"{list(WRITABLE_FAMILY_IDS)}; unknown: {unknown}"
        )
    from .measurement_register import COMPATIBILITY_ROWS, RELATION_WITHHELD

    withheld_leaves = {
        name
        for name, row in COMPATIBILITY_ROWS.items()
        if row.get("relation") == RELATION_WITHHELD
    }
    for family_name, family_payload in out.items():
        if family_payload is None:
            continue
        if not isinstance(family_payload, Mapping):
            raise ValueError(
                f"dynamical_families.{family_name} must be a mapping; "
                "list-valued family payloads are not serialized."
            )
        nested_keys = _compatibility_keys_in_payload(family_payload)
        if _payload_contains_key(family_payload, "qualified"):
            raise ValueError(
                f"dynamical_families.{family_name} nested key 'qualified' is "
                "not a payload field; use qualification_status. YAML "
                "translation_qualification.qualified is not a global flag."
            )
        found = [key for key in nested_keys if key in withheld_leaves]
        if found:
            raise ValueError(
                f"dynamical_families.{family_name} nested keys {found} are "
                "withheld_not_written and are not serialized."
            )
        namespace = str((FAMILIES.get(family_name) or {}).get("namespace") or "")
        misplaced = []
        for key in nested_keys:
            row = COMPATIBILITY_ROWS.get(key)
            path = row.get("physical_path") if isinstance(row, Mapping) else None
            if path and namespace and not str(path).startswith(namespace):
                misplaced.append(key)
        if misplaced:
            raise ValueError(
                f"dynamical_families.{family_name} nested keys {misplaced} "
                "belong to another family path and are not serialized here."
            )
        spec = (FAMILIES.get(family_name) or {}).get("grain") or {}
        native = str(spec.get("native") or "")
        if not str(family_payload.get("computation_status") or "").strip():
            raise ValueError(
                f"dynamical_families.{family_name} requires computation_status"
            )
        if not str(family_payload.get("qualification_status") or "").strip():
            raise ValueError(
                f"dynamical_families.{family_name} requires qualification_status; "
                "it is not computation_status and not YAML qualified"
            )
        grain = family_payload.get("grain")
        if not isinstance(grain, Mapping):
            raise ValueError(
                f"dynamical_families.{family_name} requires nested grain/"
            )
        missing_grain = [
            field
            for field in (
                "native",
                "parent",
                "biological_unit",
                "repeated_measure",
                "direct_between_subject_inference",
            )
            if not str(grain.get(field) or "").strip()
        ]
        if missing_grain:
            raise ValueError(
                f"dynamical_families.{family_name} grain missing {missing_grain}"
            )
        if native and str(grain.get("native") or "") != native:
            raise ValueError(
                f"dynamical_families.{family_name} grain.native must be {native}"
            )
    return out


def refuse_withheld_compatibility_keys(payload: Any, *, label: str) -> None:
    """Refuse withheld 001 names in any serialized mapping or list of mappings."""
    from .measurement_register import COMPATIBILITY_ROWS, RELATION_WITHHELD

    if payload is None or isinstance(payload, (str, bytes)):
        return
    if not isinstance(payload, (Mapping, list, tuple)):
        return
    withheld_leaves = {
        name
        for name, row in COMPATIBILITY_ROWS.items()
        if row.get("relation") == RELATION_WITHHELD
    }
    found = [key for key in _compatibility_keys_in_payload(payload) if key in withheld_leaves]
    if found:
        raise ValueError(
            f"{label} nested keys {found} are withheld_not_written and are not serialized."
        )


def _payload_contains_key(payload: Any, name: str) -> bool:
    """True if ``name`` appears as a mapping key at any depth."""
    stack: list[Any] = [payload]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            if name in current:
                return True
            stack.extend(current.values())
        elif isinstance(current, (list, tuple)) and not isinstance(current, (str, bytes)):
            stack.extend(current)
    return False


def _compatibility_keys_in_payload(payload: Any) -> list[str]:
    """Collect compatibility-table keys at any mapping or list-of-mapping depth."""
    from .measurement_register import COMPATIBILITY_ROWS

    found: list[str] = []
    stack: list[Any] = [payload]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            for key, value in current.items():
                name = str(key)
                if name in COMPATIBILITY_ROWS:
                    found.append(name)
                stack.append(value)
        elif isinstance(current, (list, tuple)) and not isinstance(current, (str, bytes)):
            stack.extend(current)
    return found
