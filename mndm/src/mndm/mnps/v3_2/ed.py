"""Raw sensor-covariance participation ratio for experimental MNPS v3.2.

This module implements only the matrix-level estimator. It does not filter,
window, repair, or resample signals and it does not infer temporal support from
the array. A returned finite value therefore has ``support_status="unknown"``
until a caller supplies an independently audited window contract. The value
is an observed sensor-space participation ratio, not intrinsic manifold
dimension or a scientifically qualified measurement.

The numerical definition follows the v3.1 E3 legacy formula (centered sample
covariance, ``ddof=1``, no shrinkage), but this raw-kernel result has a distinct
v3.2 estimator ID because temporal support and realization are not established
here. ``ED/N`` and covariance shrinkage are not implemented by this estimator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Literal, Sequence

import numpy as np

from .contracts import (
    MeasurementRecord,
    RealizationRecord,
    ReproducibilityRecord,
    TimeSupportRecord,
)


ED_RAW_ESTIMATOR_ID = "ED.sensor_covariance_participation_ratio_raw_matrix.v1"
ED_RAW_SCHEMA_ID = "nmdm.mnps_v3_2.ed_raw_matrix_result.v1"
ED_UNITS = "sensor_directions"

# Retained from the inspected v3.1 E3 implementation for numerical parity.
# Float64 is required for both covariance accumulation and eigendecomposition.
_PSD_NEGATIVE_EIGENVALUE_RTOL = 1e-10


@dataclass(frozen=True)
class EDRawResult:
    """Result of one raw matrix-level ED attempt.

    ``computed`` means covariance/spectrum arithmetic was attempted. ``valid``
    means the estimator produced a finite value after its declared array,
    support-count, and spectrum checks. Audit completeness is separate:
    temporal/source support is always unknown in this raw API, and
    ``qualified`` is always false.
    """

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
    qc: tuple[str, ...]
    covariance: str
    shrinkage: Literal["none"]
    normalized_by_channel_count: Literal[False]
    effective_sample_support_gate: Literal["not_assessed"]
    psd_negative_eigenvalue_rtol: float
    claim_status: Literal["no_biological_claim"]

    def to_measurement_record(self) -> MeasurementRecord:
        """Return the common v3.2 result envelope without inventing support."""
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
                ("normalized_by_channel_count", self.normalized_by_channel_count),
                ("psd_negative_eigenvalue_rtol", self.psd_negative_eigenvalue_rtol),
            ),
            claim_status=self.claim_status,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible result without converting nulls to zero."""
        payload = asdict(self)
        payload["channel_labels"] = (
            list(self.channel_labels) if self.channel_labels is not None else None
        )
        payload["qc"] = list(self.qc)
        payload["measurement_record"] = self.to_measurement_record().to_dict()
        return payload


def _label_metadata(
    channel_names: Sequence[str] | None,
    channel_count: int,
) -> tuple[tuple[str, ...] | None, str | None, str | None, tuple[str, ...]]:
    if channel_names is None:
        return None, None, None, ("sensor_labels_unavailable",)
    if isinstance(channel_names, (str, bytes, bytearray)):
        return None, None, None, ("sensor_labels_must_be_sequence",)

    try:
        raw_labels = tuple(channel_names)
    except (TypeError, ValueError):
        return None, None, None, ("sensor_labels_unreadable",)

    warnings: list[str] = []
    if len(raw_labels) != channel_count:
        warnings.append("sensor_label_count_mismatch")
    if any(not isinstance(label, str) for label in raw_labels):
        warnings.append("sensor_labels_must_be_strings")
        return None, None, None, tuple(warnings)

    labels = tuple(raw_labels)
    if any(not label.strip() for label in labels):
        warnings.append("empty_sensor_label")
    normalized = tuple(label.strip().casefold() for label in labels)
    if len(set(normalized)) != len(normalized):
        warnings.append("duplicate_sensor_label")
    if warnings:
        # Labels are audit metadata rather than inputs to the covariance math.
        # Preserve a computable value, while declining to invent valid hashes.
        return labels, None, None, tuple(warnings)

    # Canonical form is compact UTF-8 JSON. Order hash preserves the caller's
    # row order; set hash sorts the same exact labels and is intentionally
    # separate from the ordered realization identity.
    ordered = json.dumps(labels, ensure_ascii=False, separators=(",", ":"))
    unordered = json.dumps(
        sorted(labels), ensure_ascii=False, separators=(",", ":")
    )
    order_hash = hashlib.sha256(ordered.encode("utf-8")).hexdigest()
    set_hash = hashlib.sha256(unordered.encode("utf-8")).hexdigest()
    return labels, order_hash, set_hash, ()


def _result(
    *,
    value: float | None,
    computed: bool,
    reason: str | None,
    channel_count: int,
    sample_count: int,
    labels: tuple[str, ...] | None,
    order_hash: str | None,
    set_hash: str | None,
    qc: tuple[str, ...],
) -> EDRawResult:
    valid = value is not None
    return EDRawResult(
        estimator_id=ED_RAW_ESTIMATOR_ID,
        schema_id=ED_RAW_SCHEMA_ID,
        value=value,
        units=ED_UNITS,
        computed=computed,
        valid=valid,
        qualified=False,
        status="computed_exploratory" if valid else "unsupported",
        reason=reason,
        support_status="unknown",
        channel_count=channel_count,
        sample_count_per_channel=sample_count,
        channel_labels=labels,
        channel_order_sha256=order_hash,
        channel_set_sha256=set_hash,
        qc=qc,
        covariance="centered_sample_covariance_ddof_1",
        shrinkage="none",
        normalized_by_channel_count=False,
        effective_sample_support_gate="not_assessed",
        psd_negative_eigenvalue_rtol=_PSD_NEGATIVE_EIGENVALUE_RTOL,
        claim_status="no_biological_claim",
    )


def compute_ed_raw_sensor_covariance_participation_ratio(
    signals: np.ndarray,
    channel_names: Sequence[str] | None = None,
) -> EDRawResult:
    """Compute raw covariance participation ratio for one channel-by-time array.

    Parameters
    ----------
    signals:
        Finite, float64-convertible array shaped ``(channels, samples)``.
        The legacy sensor-covariance support rule is ``N >= 2`` and ``T > N``.
    channel_names:
        Optional labels in the same row order. Labels are provenance only and
        do not alter the estimator; absent or inconsistent labels are reported
        as QC while retaining a computable value.

    Returns
    -------
    EDRawResult
        Unsupported inputs return ``value=None`` with a reason. A finite result
        is exploratory and has unknown temporal/source support until an
        independent window auditor supplies that information.

    Notes
    -----
    For ``Y`` shaped ``(N, T)``, the calculation is
    ``C = cov(Y, rowvar=True, ddof=1)`` and
    ``ED = trace(C)**2 / trace(C @ C)``. Tiny negative eigenvalues no smaller
    than ``-1e-10 * max(max(abs(eigenvalues)), 1)`` are clipped to zero, which
    matches the current v3.1 E3 numerical tolerance for parity. No
    effective-independent-sample correction is attempted.
    """
    try:
        raw_values = np.asarray(signals)
    except (TypeError, ValueError, OverflowError):
        return _result(
            value=None,
            computed=False,
            reason="invalid_signal_array",
            channel_count=0,
            sample_count=0,
            labels=None,
            order_hash=None,
            set_hash=None,
            qc=(),
        )

    complex_input = np.iscomplexobj(raw_values)
    if raw_values.dtype.kind == "O":
        complex_input = complex_input or any(
            isinstance(value, (complex, np.complexfloating))
            for value in raw_values.flat
        )
    if complex_input:
        channel_count = int(raw_values.shape[0]) if raw_values.ndim >= 1 else 0
        sample_count = int(raw_values.shape[1]) if raw_values.ndim == 2 else 0
        labels, order_hash, set_hash, label_qc = _label_metadata(
            channel_names, channel_count
        )
        return _result(
            value=None,
            computed=False,
            reason="complex_signal_not_supported",
            channel_count=channel_count,
            sample_count=sample_count,
            labels=labels,
            order_hash=order_hash,
            set_hash=set_hash,
            qc=label_qc,
        )

    try:
        values = np.asarray(raw_values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return _result(
            value=None,
            computed=False,
            reason="invalid_signal_array",
            channel_count=int(raw_values.shape[0]) if raw_values.ndim >= 1 else 0,
            sample_count=int(raw_values.shape[1]) if raw_values.ndim == 2 else 0,
            labels=None,
            order_hash=None,
            set_hash=None,
            qc=(),
        )

    if values.ndim != 2:
        return _result(
            value=None,
            computed=False,
            reason="signal_must_be_channels_by_samples",
            channel_count=int(values.shape[0]) if values.ndim else 0,
            sample_count=int(values.shape[1]) if values.ndim == 2 else 0,
            labels=None,
            order_hash=None,
            set_hash=None,
            qc=(),
        )

    channel_count, sample_count = map(int, values.shape)
    labels, order_hash, set_hash, label_qc = _label_metadata(
        channel_names, channel_count
    )

    def unsupported(reason: str, *, computed: bool = False) -> EDRawResult:
        return _result(
            value=None,
            computed=computed,
            reason=reason,
            channel_count=channel_count,
            sample_count=sample_count,
            labels=labels,
            order_hash=order_hash,
            set_hash=set_hash,
            qc=label_qc,
        )

    if channel_count < 2:
        return unsupported("insufficient_channels_for_sensor_covariance")
    if sample_count <= channel_count:
        return unsupported("insufficient_samples_for_sensor_covariance")
    if not np.isfinite(values).all():
        return unsupported("non_finite_sample")

    # Input conversion and support checks are complete. The following flag
    # distinguishes a numeric attempt that failed from an unattempted input.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        covariance = np.cov(values, rowvar=True, ddof=1)
    if not np.isfinite(covariance).all():
        return unsupported("non_finite_covariance", computed=True)

    try:
        eigenvalues = np.linalg.eigvalsh(covariance)
    except np.linalg.LinAlgError:
        return unsupported("covariance_eigendecomposition_failed", computed=True)
    if not np.isfinite(eigenvalues).all():
        return unsupported("non_finite_covariance_spectrum", computed=True)

    spectral_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -_PSD_NEGATIVE_EIGENVALUE_RTOL * spectral_scale:
        return unsupported("covariance_not_positive_semidefinite", computed=True)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    trace = float(np.sum(eigenvalues))
    sum_squares = float(np.sum(eigenvalues * eigenvalues))
    if (
        not np.isfinite(trace)
        or not np.isfinite(sum_squares)
        or trace <= 0.0
        or sum_squares <= 0.0
    ):
        return unsupported("undefined_covariance_spectrum", computed=True)

    value = float(trace * trace / sum_squares)
    if not np.isfinite(value):
        return unsupported("non_finite_participation_ratio", computed=True)
    return _result(
        value=value,
        computed=True,
        reason=None,
        channel_count=channel_count,
        sample_count=sample_count,
        labels=labels,
        order_hash=order_hash,
        set_hash=set_hash,
        qc=label_qc + ("temporal_and_source_support_not_audited",),
    )


__all__ = [
    "ED_RAW_ESTIMATOR_ID",
    "ED_RAW_SCHEMA_ID",
    "EDRawResult",
    "compute_ed_raw_sensor_covariance_participation_ratio",
]
