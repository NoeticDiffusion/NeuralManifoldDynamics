"""Channel-count-normalized ED diagnostic for experimental MNPS v3.2.

``ED/N`` is a separate derived estimator identity. It is a descriptive
normalization by the sensor count; it does not make different montages,
references, or covariance supports equivalent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from .contracts import (
    MeasurementRecord,
    RealizationRecord,
    ReproducibilityRecord,
    TimeSupportRecord,
)
from .ed import ED_RAW_ESTIMATOR_ID, ED_RAW_SCHEMA_ID, EDRawResult


ED_NORMALIZED_ESTIMATOR_ID = (
    "ED.sensor_covariance_participation_ratio_normalized_by_channel_count.v1"
)
ED_NORMALIZED_SCHEMA_ID = "nmdm.mnps_v3_2.ed_normalized_result.v1"


@dataclass(frozen=True)
class EDNormalizedResult:
    """A diagnostic ``raw ED / N`` value derived from one raw ED result."""

    estimator_id: str
    schema_id: str
    source_estimator_id: str
    value: float | None
    units: str
    computed: bool
    valid: bool
    qualified: Literal[False]
    status: Literal["computed_exploratory", "unsupported"]
    reason: str | None
    support_status: Literal["unknown"]
    channel_count: int
    sample_count_per_channel: int
    channel_labels: tuple[str, ...] | None
    channel_order_sha256: str | None
    channel_set_sha256: str | None
    normalization_denominator: Literal["channel_count"]
    qc: tuple[str, ...]
    claim_status: Literal["no_biological_claim"]

    def to_measurement_record(self) -> MeasurementRecord:
        if self.channel_labels is None:
            audit_status = "unknown"
        elif self.channel_order_sha256 is not None and self.channel_set_sha256 is not None:
            audit_status = "complete"
        else:
            audit_status = "incomplete"
        return MeasurementRecord(
            family_id="ED",
            estimator_id=self.estimator_id,
            schema_id=self.schema_id,
            value=self.value,
            units=self.units,
            computed=self.computed,
            valid=self.valid,
            qualified=self.qualified,
            status=self.status,
            reason=self.reason,
            support=TimeSupportRecord(
                support_status=self.support_status,
                sample_count=self.sample_count_per_channel,
            ),
            realization=RealizationRecord(
                audit_status=audit_status,
                kind="sensor_channels",
                requested_id=None,
                actual_id=None,
                ordered_labels=self.channel_labels,
                order_sha256=self.channel_order_sha256,
                set_sha256=self.channel_set_sha256,
                count=self.channel_count,
            ),
            qc=self.qc,
            reproducibility=ReproducibilityRecord(status="not_supplied"),
            method_parameters=(
                ("source_estimator_id", self.source_estimator_id),
                ("normalization_denominator", self.normalization_denominator),
                ("channel_count", self.channel_count),
            ),
            claim_status=self.claim_status,
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["channel_labels"] = (
            list(self.channel_labels) if self.channel_labels is not None else None
        )
        payload["qc"] = list(self.qc)
        payload["measurement_record"] = self.to_measurement_record().to_dict()
        return payload


def compute_ed_normalized_by_channel_count(
    raw_result: EDRawResult,
) -> EDNormalizedResult:
    """Return ``ED/N`` as a distinct, unqualified diagnostic measurement."""
    if not isinstance(raw_result, EDRawResult):
        return EDNormalizedResult(
            estimator_id=ED_NORMALIZED_ESTIMATOR_ID,
            schema_id=ED_NORMALIZED_SCHEMA_ID,
            source_estimator_id=ED_RAW_ESTIMATOR_ID,
            value=None,
            units="proportion_of_channels",
            computed=False,
            valid=False,
            qualified=False,
            status="unsupported",
            reason="raw_ed_result_required",
            support_status="unknown",
            channel_count=0,
            sample_count_per_channel=0,
            channel_labels=None,
            channel_order_sha256=None,
            channel_set_sha256=None,
            normalization_denominator="channel_count",
            qc=("raw_ed_result_required",),
            claim_status="no_biological_claim",
        )

    value: float | None = None
    reason: str | None = None
    computed = raw_result.computed
    if raw_result.value is None:
        reason = raw_result.reason or "raw_ed_value_unavailable"
    elif raw_result.channel_count <= 0:
        reason = "invalid_channel_count"
    else:
        candidate = float(raw_result.value) / raw_result.channel_count
        if np.isfinite(candidate):
            value = candidate
        else:
            reason = "non_finite_normalized_participation_ratio"

    valid = value is not None
    return EDNormalizedResult(
        estimator_id=ED_NORMALIZED_ESTIMATOR_ID,
        schema_id=ED_NORMALIZED_SCHEMA_ID,
        source_estimator_id=raw_result.estimator_id,
        value=value,
        units="proportion_of_channels",
        computed=computed,
        valid=valid,
        qualified=False,
        status="computed_exploratory" if valid else "unsupported",
        reason=reason,
        support_status=raw_result.support_status,
        channel_count=raw_result.channel_count,
        sample_count_per_channel=raw_result.sample_count_per_channel,
        channel_labels=raw_result.channel_labels,
        channel_order_sha256=raw_result.channel_order_sha256,
        channel_set_sha256=raw_result.channel_set_sha256,
        normalization_denominator="channel_count",
        qc=raw_result.qc,
        claim_status="no_biological_claim",
    )


__all__ = [
    "ED_NORMALIZED_ESTIMATOR_ID",
    "ED_NORMALIZED_SCHEMA_ID",
    "EDNormalizedResult",
    "compute_ed_normalized_by_channel_count",
]
