"""Shared standard contracts for non-MNPS dynamical measurements.

These measurement families consume an exported coordinate trajectory but do not
alter its MNPS definition.  In particular, a transition-residual covariance
proxy is not an Itô diffusion tensor, and finite-time tangent response is not
finite-amplitude resilience.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..measurement_certificate import attach_certificate
from ..inferential_grain import attach_grain_for_schema


DIFFUSION_GEOMETRY_SCHEMA_VERSION = "mndm.diffusion_geometry.v1"
COMMITTOR_SCHEMA_VERSION = "mndm.committor.v1"
FINITE_AMPLITUDE_RESILIENCE_SCHEMA_VERSION = "mndm.finite_amplitude_resilience.v1"

VALID_COMPUTATION_STATUSES = frozenset(
    {"computed", "not_requested", "not_testable", "insufficient_support", "invalid"}
)
# Schema/contract class for these family objects. This is not an empirical
# license and is not the same as a scientific "experimental estimator" label.
CONTRACT_STATUS = "standard"
SIMULATOR_VALIDATED = "simulator_validated"
MNDM_TRANSLATION_VALIDATED = "mndm_translation_validated"
CLAIM_STATUS = "no_biological_claim"


def unavailable_result(
    schema_version: str,
    *,
    status: str,
    failure_reason: str,
    coordinate_layer: str | None = None,
    coordinate_names: list[str] | None = None,
) -> dict[str, Any]:
    """Return a schema-stable non-computed result without fabricated values."""
    if status not in VALID_COMPUTATION_STATUSES or status == "computed":
        raise ValueError(f"Unsupported computation status: {status}")
    result: dict[str, Any] = {
        "schema_version": schema_version,
        "computation_status": status,
        "failure_reason": failure_reason,
        "series": {},
        "summary": {},
        "provenance": {
            "coordinate_layer": str(coordinate_layer) if coordinate_layer is not None else None,
            "coordinate_names": [str(name) for name in coordinate_names] if coordinate_names is not None else [],
            "contract_status": CONTRACT_STATUS,
            "contract_version": "v1",
            "validation_level": SIMULATOR_VALIDATED,
            "claim_status": CLAIM_STATUS,
            "adapter_version": "mndm.dynamical_families.adapter.v1",
            "qualification_id": None,
            "qualification_contract_hash": None,
        },
    }
    return attach_grain_for_schema(attach_certificate(result))


def build_provenance(
    *,
    coordinate_layer: str,
    coordinate_names: list[str],
    time_semantics: str,
    estimator: str,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Build explicit, JSON/HDF5-safe provenance for one estimator result."""
    return {
        "coordinate_layer": str(coordinate_layer),
        "coordinate_names": [str(name) for name in coordinate_names],
        "time_semantics": str(time_semantics),
        "estimator": str(estimator),
        "settings": dict(settings),
        "contract_status": CONTRACT_STATUS,
        "contract_version": "v1",
        "validation_level": SIMULATOR_VALIDATED,
        "claim_status": CLAIM_STATUS,
        "adapter_version": "mndm.dynamical_families.adapter.v1",
        "qualification_id": None,
        "qualification_contract_hash": None,
    }
