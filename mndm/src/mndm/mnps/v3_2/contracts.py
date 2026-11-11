"""Shared, serializable result envelope for experimental MNPS v3.2 kernels.

The envelope carries estimator output together with the evidence a kernel
actually receives. Unknown recording support, realization identity, or
reproducibility fields stay null and are not inferred from a numeric value.
Pipeline wrappers may fill source and time support after independently
auditing those inputs; mathematical kernels must not invent it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal, TypeAlias


CONTRACT_ID = "nmdm.mnps_v3_2.measurement_record.v1"

SupportStatus: TypeAlias = Literal["known", "unknown", "unsupported"]
RecordStatus: TypeAlias = Literal["computed_exploratory", "unsupported"]
RealizationAuditStatus: TypeAlias = Literal["complete", "incomplete", "unknown"]


@dataclass(frozen=True)
class TimeSupportRecord:
    """Audited observation support; unset values mean unknown, not zero."""

    support_status: SupportStatus
    source_id: str | None = None
    source_sha256: str | None = None
    segment_id: str | None = None
    sample_grid_id: str | None = None
    sample_rate_hz: float | None = None
    start_sample: int | None = None
    end_sample_exclusive: int | None = None
    sample_count: int | None = None
    nominal_window_seconds: float | None = None
    dependency_start_sample: int | None = None
    dependency_end_sample_exclusive: int | None = None
    gap_free: bool | None = None
    coverage: float | None = None
    availability: str | None = None
    filter_warmup_samples: int | None = None
    filter_delay_samples: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RealizationRecord:
    """Actual ordered observation set and its separate audit state."""

    audit_status: RealizationAuditStatus
    kind: str | None
    requested_id: str | None
    actual_id: str | None
    ordered_labels: tuple[str, ...] | None
    order_sha256: str | None
    set_sha256: str | None
    count: int | None
    coverage: float | None = None
    reference_id: str | None = None
    reference_set_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["ordered_labels"] = (
            list(self.ordered_labels) if self.ordered_labels is not None else None
        )
        return payload


@dataclass(frozen=True)
class ReproducibilityRecord:
    """Run-level identifiers supplied by a protocol or pipeline wrapper."""

    status: Literal["complete", "partial", "not_supplied"]
    run_id: str | None = None
    input_sha256: str | None = None
    code_sha256: str | None = None
    protocol_sha256: str | None = None
    config_id: str | None = None
    config_sha256: str | None = None
    dependency_versions: tuple[tuple[str, str], ...] | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["dependency_versions"] = (
            [list(pair) for pair in self.dependency_versions]
            if self.dependency_versions is not None
            else None
        )
        return payload


MeasurementValue: TypeAlias = float | tuple[float, ...] | None
MethodParameterValue: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True)
class MeasurementRecord:
    """Common result contract shared by v3.2 measurement kernels.

    ``value`` is either one scalar, a stable-order vector, or null. Vector
    outputs must provide matching component names; this prevents silent
    averaging or conversion to an opaque serialized string.
    """

    family_id: str
    estimator_id: str
    schema_id: str
    value: MeasurementValue
    units: str
    computed: bool
    valid: bool
    qualified: bool
    status: RecordStatus
    reason: str | None
    support: TimeSupportRecord
    realization: RealizationRecord
    qc: tuple[str, ...]
    reproducibility: ReproducibilityRecord
    value_components: tuple[str, ...] | None = None
    method_parameters: tuple[tuple[str, MethodParameterValue], ...] | None = None
    direction: str | None = None
    qualification_decision_ref: str | None = None
    claim_status: str = "no_biological_claim"
    contract_id: str = CONTRACT_ID

    def __post_init__(self) -> None:
        if isinstance(self.value, tuple):
            if self.value_components is None or len(self.value_components) != len(self.value):
                raise ValueError("vector values require one ordered component name per value")
            if any(not isinstance(name, str) or not name.strip() for name in self.value_components):
                raise ValueError("vector component names must be nonempty strings")
            normalized = tuple(name.strip().casefold() for name in self.value_components)
            if len(set(normalized)) != len(normalized):
                raise ValueError("vector component names must be unique")
            if any(not math.isfinite(component) for component in self.value):
                raise ValueError("vector values must be finite")
        elif self.value_components is not None:
            raise ValueError("scalar or null values cannot declare vector components")
        elif self.value is not None and not math.isfinite(self.value):
            raise ValueError("scalar values must be finite")
        if self.valid != (self.value is not None):
            raise ValueError("valid must indicate whether a finite measurement value is present")
        if self.qualified and not self.valid:
            raise ValueError("an invalid measurement cannot be scientifically qualified")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if isinstance(self.value, tuple):
            payload["value"] = list(self.value)
        payload["value_components"] = (
            list(self.value_components) if self.value_components is not None else None
        )
        payload["method_parameters"] = (
            [list(item) for item in self.method_parameters]
            if self.method_parameters is not None
            else None
        )
        payload["qc"] = list(self.qc)
        payload["support"] = self.support.to_dict()
        payload["realization"] = self.realization.to_dict()
        payload["reproducibility"] = self.reproducibility.to_dict()
        return payload


__all__ = [
    "CONTRACT_ID",
    "MeasurementRecord",
    "MethodParameterValue",
    "MeasurementValue",
    "RealizationRecord",
    "ReproducibilityRecord",
    "TimeSupportRecord",
]
