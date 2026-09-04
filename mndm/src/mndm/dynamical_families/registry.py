"""Canonical dynamical-family registry and dependency DAG.

This is the Round-1 semantic kernel: identifiers, schemas, namespaces,
dependencies, and forbidden substitutions. Round 3 stores default grain
on each family as documentation metadata; writers still go through
``attach_grain``. Support signatures remain a later round.
"""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import (
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
        ),
        "grain": {
            "native": "recording_horizon",
            "parent": "recording",
            "repeated_measure": "true",
        },
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
        "forbids": ("stage_labels_as_destination_truth",),
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
        ),
        "grain": {
            "native": "event",
            "parent": "recording",
            "repeated_measure": "true",
        },
    },
}

WRITABLE_FAMILY_IDS = ("diffusion", "destination", "resilience")

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
    return out
