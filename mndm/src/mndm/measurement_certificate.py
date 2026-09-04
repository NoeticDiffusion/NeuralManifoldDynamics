"""Three-way measurement certificate for family and local-dynamics exports.

Round 2 makes these sibling fields first-class:

```text
computation_status     value exists / why not
measurement_validity   valid under this observation regime
claim_status           NDT / biology licensed
```

``validation_level`` is a method-validation tag and is not rewritten here.
Write paths never emit ``ndt_licensed`` or infer regime validity that was not
explicitly requested by the caller.
"""

from __future__ import annotations

from typing import Any, Mapping

COMPUTATION_STATUS_COMPUTED = "computed"
NOT_RECORDED = "not_recorded"
NOT_APPLICABLE = "not_applicable"
NOT_ASSESSED = "not_assessed"
TRANSLATION_QUALIFIED = "translation_qualified"
NO_BIOLOGICAL_CLAIM = "no_biological_claim"

VALID_COMPUTATION_STATUSES = frozenset(
    {
        "computed",
        "not_requested",
        "not_testable",
        "insufficient_support",
        "invalid",
        "unavailable",
    }
)
VALID_MEASUREMENT_VALIDITY_WRITE = frozenset(
    {NOT_APPLICABLE, NOT_ASSESSED, TRANSLATION_QUALIFIED}
)
VALID_MEASUREMENT_VALIDITY_READ = VALID_MEASUREMENT_VALIDITY_WRITE | {NOT_RECORDED}
VALID_CLAIM_STATUS_WRITE = frozenset({NO_BIOLOGICAL_CLAIM})
VALID_CLAIM_STATUS_READ = VALID_CLAIM_STATUS_WRITE | {NOT_RECORDED}

CERTIFICATE_ORIGIN_CANONICAL = "canonical"
CERTIFICATE_ORIGIN_PROMOTED = "legacy_promoted_provenance"
CERTIFICATE_ORIGIN_ABSENT = "legacy_absent"


def measurement_validity_for_write(
    computation_status: str,
    *,
    translation_qualified: bool = False,
) -> str:
    """Return the write-side measurement_validity for a computation status."""
    if translation_qualified:
        if computation_status != COMPUTATION_STATUS_COMPUTED:
            raise ValueError(
                "translation_qualified requires computation_status='computed'"
            )
        return TRANSLATION_QUALIFIED
    if computation_status == COMPUTATION_STATUS_COMPUTED:
        return NOT_ASSESSED
    return NOT_APPLICABLE


def attach_certificate(
    result: Mapping[str, Any],
    *,
    translation_qualified: bool = False,
) -> dict[str, Any]:
    """Attach write-side certificate fields as siblings of computation_status."""
    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping")
    out = dict(result)
    status = str(out.get("computation_status") or "").strip()
    if not status:
        raise ValueError("computation_status is required before attaching a certificate")
    validity = measurement_validity_for_write(
        status,
        translation_qualified=translation_qualified,
    )
    validate_write_certificate(
        measurement_validity=validity,
        claim_status=NO_BIOLOGICAL_CLAIM,
    )
    out["measurement_validity"] = validity
    out["claim_status"] = NO_BIOLOGICAL_CLAIM
    return out


def validate_write_certificate(
    *,
    measurement_validity: str,
    claim_status: str,
) -> None:
    """Reject reader-only or unknown tokens on write."""
    if measurement_validity not in VALID_MEASUREMENT_VALIDITY_WRITE:
        raise ValueError(
            f"Unsupported measurement_validity for write: {measurement_validity}"
        )
    if claim_status not in VALID_CLAIM_STATUS_WRITE:
        raise ValueError(f"Unsupported claim_status for write: {claim_status}")


def _decode_scalar(value: Any) -> str | None:
    if value is None:
        return None
    raw = value
    if hasattr(value, "shape") and getattr(value, "shape", ()) == ():
        try:
            raw = value[()]
        except Exception:
            raw = value
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    if isinstance(raw, str):
        return raw
    if raw is None:
        return None
    return str(raw)


def _group_scalar(group: Any, key: str) -> str | None:
    if group is None:
        return None
    try:
        present = key in group
    except Exception:
        return None
    if not present:
        return None
    try:
        return _decode_scalar(group[key])
    except Exception:
        return None


def read_measurement_certificate(group: Any) -> dict[str, str]:
    """Read certificate fields without inferring scientific validity.

    Missing top-level fields become ``not_recorded``. ``claim_status`` may be
    copied from ``provenance/claim_status`` when that dataset exists. Qualification
    IDs and ``validation_level`` are never promoted to ``translation_qualified``.
    """
    computation = _group_scalar(group, "computation_status")
    validity = _group_scalar(group, "measurement_validity")
    claim = _group_scalar(group, "claim_status")
    provenance_claim = None
    if group is not None:
        try:
            provenance_present = "provenance" in group
        except Exception:
            provenance_present = False
        if provenance_present:
            provenance_claim = _group_scalar(group["provenance"], "claim_status")

    origin = CERTIFICATE_ORIGIN_CANONICAL
    if computation is None or validity is None or claim is None:
        origin = CERTIFICATE_ORIGIN_ABSENT
        if claim is None and provenance_claim is not None:
            claim = provenance_claim
            origin = CERTIFICATE_ORIGIN_PROMOTED
        if computation is None:
            computation = NOT_RECORDED
        if validity is None:
            validity = NOT_RECORDED
        if claim is None:
            claim = NOT_RECORDED

    return {
        "computation_status": computation,
        "measurement_validity": validity,
        "claim_status": claim,
        "certificate_origin": origin,
    }
