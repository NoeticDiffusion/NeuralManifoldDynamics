"""Inferential grain for family and local-dynamics exports.

Round 3 makes native observation grain first-class so a window or HDF5 file
is not treated as a biological replicate.

```text
one window / one recording  ≠  one participant
```

Grain is schema metadata: it is written even when computation_status is not
computed. Readers never infer ``window`` or ``subject`` from series or paths.
"""

from __future__ import annotations

from typing import Any, Mapping

from .measurement_certificate import NOT_RECORDED, _group_scalar

GRAIN_NATIVE = "native"
GRAIN_PARENT = "parent"
GRAIN_BIOLOGICAL_UNIT = "biological_unit"
GRAIN_REPEATED_MEASURE = "repeated_measure"
GRAIN_BETWEEN_SUBJECT = "direct_between_subject_inference"

GRAIN_FIELDS = (
    GRAIN_NATIVE,
    GRAIN_PARENT,
    GRAIN_BIOLOGICAL_UNIT,
    GRAIN_REPEATED_MEASURE,
    GRAIN_BETWEEN_SUBJECT,
)

VALID_GRAIN_UNITS = frozenset(
    {"window", "transition", "recording_horizon", "event", "recording", "subject"}
)
VALID_REPEATED_MEASURE_WRITE = frozenset({"true", "false"})
BIOLOGICAL_UNIT_SUBJECT = "subject"
BETWEEN_SUBJECT_FORBIDDEN = "forbidden"

GRAIN_ORIGIN_CANONICAL = "canonical"
GRAIN_ORIGIN_ABSENT = "legacy_absent"

# Schema → native grain. Property of the contract, not of whether values exist.
GRAIN_BY_SCHEMA: dict[str, dict[str, str]] = {
    "mndm.jacobian_metrics.v1": {
        "native": "window",
        "parent": "recording",
        "repeated_measure": "true",
    },
    "mndm.finite_time_response.v1": {
        "native": "recording_horizon",
        "parent": "recording",
        "repeated_measure": "true",
    },
    "mndm.transition_residuals.v1": {
        "native": "transition",
        "parent": "recording",
        "repeated_measure": "true",
    },
    "mndm.residual_covariance_proxy.v1": {
        "native": "recording",
        "parent": "subject",
        "repeated_measure": "false",
    },
    "mndm.transition_residual_covariance_proxy.v1": {
        "native": "recording",
        "parent": "subject",
        "repeated_measure": "false",
    },
    "mndm.stochastic_reachability.v1": {
        "native": "recording_horizon",
        "parent": "recording",
        "repeated_measure": "true",
    },
    "mndm.diffusion_geometry.v1": {
        "native": "window",
        "parent": "recording",
        "repeated_measure": "true",
    },
    "mndm.committor.v1": {
        "native": "window",
        "parent": "recording",
        "repeated_measure": "true",
    },
    "mndm.finite_amplitude_resilience.v1": {
        "native": "event",
        "parent": "recording",
        "repeated_measure": "true",
    },
}

GRAIN_BY_FAMILY_ID: dict[str, dict[str, str]] = {
    "diffusion": GRAIN_BY_SCHEMA["mndm.diffusion_geometry.v1"],
    "destination": GRAIN_BY_SCHEMA["mndm.committor.v1"],
    "resilience": GRAIN_BY_SCHEMA["mndm.finite_amplitude_resilience.v1"],
}


def validate_write_grain(
    *,
    native: str,
    parent: str,
    biological_unit: str,
    repeated_measure: str,
    direct_between_subject_inference: str,
) -> None:
    """Reject unknown grain tokens on write."""
    if native not in VALID_GRAIN_UNITS:
        raise ValueError(f"Unsupported grain.native for write: {native}")
    if parent not in VALID_GRAIN_UNITS:
        raise ValueError(f"Unsupported grain.parent for write: {parent}")
    if biological_unit != BIOLOGICAL_UNIT_SUBJECT:
        raise ValueError(
            f"Unsupported grain.biological_unit for write: {biological_unit}"
        )
    if repeated_measure not in VALID_REPEATED_MEASURE_WRITE:
        raise ValueError(
            f"Unsupported grain.repeated_measure for write: {repeated_measure}"
        )
    if direct_between_subject_inference != BETWEEN_SUBJECT_FORBIDDEN:
        raise ValueError(
            "Unsupported grain.direct_between_subject_inference for write: "
            f"{direct_between_subject_inference}"
        )


def attach_grain(
    result: Mapping[str, Any],
    *,
    native: str,
    parent: str,
    repeated_measure: str,
    biological_unit: str = BIOLOGICAL_UNIT_SUBJECT,
    direct_between_subject_inference: str = BETWEEN_SUBJECT_FORBIDDEN,
) -> dict[str, Any]:
    """Attach a nested grain mapping as a sibling of computation_status."""
    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping")
    validate_write_grain(
        native=native,
        parent=parent,
        biological_unit=biological_unit,
        repeated_measure=repeated_measure,
        direct_between_subject_inference=direct_between_subject_inference,
    )
    out = dict(result)
    out["grain"] = {
        GRAIN_NATIVE: native,
        GRAIN_PARENT: parent,
        GRAIN_BIOLOGICAL_UNIT: biological_unit,
        GRAIN_REPEATED_MEASURE: repeated_measure,
        GRAIN_BETWEEN_SUBJECT: direct_between_subject_inference,
    }
    return out


def attach_grain_for_schema(
    result: Mapping[str, Any],
    schema_version: str | None = None,
) -> dict[str, Any]:
    """Attach grain from the result schema, or an explicit schema_version."""
    schema = str(schema_version or result.get("schema_version") or "").strip()
    spec = GRAIN_BY_SCHEMA.get(schema)
    if spec is None:
        raise ValueError(f"No inferential grain registered for schema: {schema}")
    return attach_grain(result, **spec)


def read_inferential_grain(group: Any) -> dict[str, str]:
    """Read grain fields without inferring window or subject from series/path.

    Missing ``grain/`` or missing child fields become ``not_recorded``.
    """
    grain_group = None
    if group is not None:
        try:
            grain_present = "grain" in group
        except Exception:
            grain_present = False
        if grain_present:
            candidate = group["grain"]
            if isinstance(candidate, Mapping) or hasattr(candidate, "keys"):
                grain_group = candidate

    fields: dict[str, str] = {}
    missing = grain_group is None
    for key in GRAIN_FIELDS:
        value = _group_scalar(grain_group, key) if grain_group is not None else None
        if value is None:
            missing = True
            fields[key] = NOT_RECORDED
        else:
            fields[key] = value

    return {
        **fields,
        "grain_origin": GRAIN_ORIGIN_ABSENT if missing else GRAIN_ORIGIN_CANONICAL,
    }
