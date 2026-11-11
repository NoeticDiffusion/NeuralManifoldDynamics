"""Isotropic-shrinkage challenger for experimental MNPS v3.2 ED.

This is a separate estimator from raw ED v1. It uses the same centered
sample-covariance input rule, then applies
``C_lambda = (1-lambda) C + lambda trace(C)/N I``. The shrinkage parameter is
explicit and must not be selected from recording outcomes. The ``lambda=1``
arm is an algebraic control: for positive trace its ED equals the sensor count.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import numpy as np

from .contracts import (
    MeasurementRecord,
    RealizationRecord,
    ReproducibilityRecord,
    TimeSupportRecord,
)
from .ed import (
    ED_UNITS,
    EDRawResult,
    _PSD_NEGATIVE_EIGENVALUE_RTOL,
    compute_ed_raw_sensor_covariance_participation_ratio,
)


ED_ISOTROPIC_SHRINKAGE_ESTIMATOR_ID = (
    "ED.sensor_covariance_participation_ratio_isotropic_shrinkage.v1"
)
ED_ISOTROPIC_SHRINKAGE_SCHEMA_ID = "nmdm.mnps_v3_2.ed_isotropic_shrinkage_result.v1"


@dataclass(frozen=True)
class EDIsotropicShrinkageResult:
    """One raw-data ED attempt with an explicitly named shrinkage parameter."""

    estimator_id: str
    schema_id: str
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
    shrinkage_lambda: float | None
    covariance: str
    shrinkage: Literal["isotropic_trace_preserving"]
    normalized_by_channel_count: Literal[False]
    psd_negative_eigenvalue_rtol: float
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
                ("covariance", self.covariance),
                ("shrinkage", self.shrinkage),
                ("shrinkage_lambda", self.shrinkage_lambda),
                ("normalized_by_channel_count", self.normalized_by_channel_count),
                ("psd_negative_eigenvalue_rtol", self.psd_negative_eigenvalue_rtol),
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


def _result_from_raw(
    raw: EDRawResult,
    *,
    value: float | None,
    computed: bool,
    reason: str | None,
    shrinkage_lambda: float | None,
    qc: tuple[str, ...],
) -> EDIsotropicShrinkageResult:
    valid = value is not None
    return EDIsotropicShrinkageResult(
        estimator_id=ED_ISOTROPIC_SHRINKAGE_ESTIMATOR_ID,
        schema_id=ED_ISOTROPIC_SHRINKAGE_SCHEMA_ID,
        value=value,
        units=ED_UNITS,
        computed=computed,
        valid=valid,
        qualified=False,
        status="computed_exploratory" if valid else "unsupported",
        reason=reason,
        support_status="unknown",
        channel_count=raw.channel_count,
        sample_count_per_channel=raw.sample_count_per_channel,
        channel_labels=raw.channel_labels,
        channel_order_sha256=raw.channel_order_sha256,
        channel_set_sha256=raw.channel_set_sha256,
        shrinkage_lambda=shrinkage_lambda,
        covariance=raw.covariance,
        shrinkage="isotropic_trace_preserving",
        normalized_by_channel_count=False,
        psd_negative_eigenvalue_rtol=_PSD_NEGATIVE_EIGENVALUE_RTOL,
        qc=qc,
        claim_status="no_biological_claim",
    )


def compute_ed_isotropic_shrinkage(
    signals: np.ndarray,
    shrinkage_lambda: float,
    channel_names: Sequence[str] | None = None,
) -> EDIsotropicShrinkageResult:
    """Compute participation ratio after explicit isotropic covariance shrinkage.

    The input uses the raw ED v1 array contract: real finite ``(channels,
    samples)`` values, at least two channels, and more samples than channels.
    ``shrinkage_lambda`` must be a finite numeric scalar in ``[0,1]``. Labels
    remain provenance only. This result is exploratory and has unknown
    temporal/source support.
    """
    raw = compute_ed_raw_sensor_covariance_participation_ratio(signals, channel_names)
    if raw.value is None:
        return _result_from_raw(
            raw,
            value=None,
            computed=raw.computed,
            reason=raw.reason,
            shrinkage_lambda=None,
            qc=raw.qc,
        )

    if isinstance(shrinkage_lambda, (bool, np.bool_)) or not isinstance(
        shrinkage_lambda, (int, float, np.integer, np.floating)
    ):
        return _result_from_raw(
            raw,
            value=None,
            computed=False,
            reason="invalid_shrinkage_lambda",
            shrinkage_lambda=None,
            qc=raw.qc + ("shrinkage_lambda_must_be_finite_numeric_in_0_1",),
        )
    lam = float(shrinkage_lambda)
    if not np.isfinite(lam) or not 0.0 <= lam <= 1.0:
        return _result_from_raw(
            raw,
            value=None,
            computed=False,
            reason="invalid_shrinkage_lambda",
            shrinkage_lambda=lam if np.isfinite(lam) else None,
            qc=raw.qc + ("shrinkage_lambda_must_be_finite_numeric_in_0_1",),
        )

    values = np.asarray(signals, dtype=np.float64)
    covariance = np.cov(values, rowvar=True, ddof=1)
    channel_count = int(values.shape[0])
    trace = float(np.trace(covariance))
    if not np.isfinite(covariance).all() or not np.isfinite(trace) or trace <= 0.0:
        return _result_from_raw(
            raw,
            value=None,
            computed=True,
            reason="undefined_covariance_spectrum",
            shrinkage_lambda=lam,
            qc=raw.qc + ("invalid_covariance_for_isotropic_shrinkage",),
        )

    shrunk_covariance = (1.0 - lam) * covariance + lam * (trace / channel_count) * np.eye(
        channel_count, dtype=np.float64
    )
    try:
        eigenvalues = np.linalg.eigvalsh(shrunk_covariance)
    except np.linalg.LinAlgError:
        return _result_from_raw(
            raw,
            value=None,
            computed=True,
            reason="covariance_eigendecomposition_failed",
            shrinkage_lambda=lam,
            qc=raw.qc,
        )
    if not np.isfinite(eigenvalues).all():
        return _result_from_raw(
            raw,
            value=None,
            computed=True,
            reason="non_finite_covariance_spectrum",
            shrinkage_lambda=lam,
            qc=raw.qc,
        )
    spectral_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -_PSD_NEGATIVE_EIGENVALUE_RTOL * spectral_scale:
        return _result_from_raw(
            raw,
            value=None,
            computed=True,
            reason="covariance_not_positive_semidefinite",
            shrinkage_lambda=lam,
            qc=raw.qc,
        )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    spectral_trace = float(np.sum(eigenvalues))
    sum_squares = float(np.sum(eigenvalues * eigenvalues))
    if (
        not np.isfinite(spectral_trace)
        or not np.isfinite(sum_squares)
        or spectral_trace <= 0.0
        or sum_squares <= 0.0
    ):
        return _result_from_raw(
            raw,
            value=None,
            computed=True,
            reason="undefined_covariance_spectrum",
            shrinkage_lambda=lam,
            qc=raw.qc,
        )
    value = float(spectral_trace * spectral_trace / sum_squares)
    if not np.isfinite(value):
        return _result_from_raw(
            raw,
            value=None,
            computed=True,
            reason="non_finite_participation_ratio",
            shrinkage_lambda=lam,
            qc=raw.qc,
        )
    return _result_from_raw(
        raw,
        value=value,
        computed=True,
        reason=None,
        shrinkage_lambda=lam,
        qc=raw.qc,
    )


__all__ = [
    "ED_ISOTROPIC_SHRINKAGE_ESTIMATOR_ID",
    "ED_ISOTROPIC_SHRINKAGE_SCHEMA_ID",
    "EDIsotropicShrinkageResult",
    "compute_ed_isotropic_shrinkage",
]
