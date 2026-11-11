"""Standalone matrix-level ordinal irregularity for experimental MNPS v3.2.

This kernel computes normalized order-5, delay-1 permutation entropy for each
row of a channel-by-sample matrix, then takes the median over every supplied
row. Ordinal ties use stable mergesort, so equal samples retain their original
time order. The formula is explicit, but the matrix API does not select or
verify preprocessing, a channel set, a reference, a sample grid, or recording
support. Its realization is therefore unresolved and its result is never
scientifically qualified here.

This is a new v3.2 matrix-input estimator. It does not relabel either the
fixed-83-channel v3.1 E1 estimator or the separate native-window v3.1 pipeline
measurement. It rejects non-finite samples instead of silently deleting them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Literal, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .contracts import (
    MeasurementRecord,
    RealizationRecord,
    ReproducibilityRecord,
    TimeSupportRecord,
)


OI_MATRIX_ESTIMATOR_ID = "OI.matrix_input_channel_median_normalized_PE5_delay1.v1"
OI_MATRIX_SCHEMA_ID = "nmdm.mnps_v3_2.oi_matrix_result.v1"
OI_UNITS = "normalized_entropy"
OI_ORDER = 5
OI_DELAY_SAMPLES = 1
OI_PATTERN_COUNT = 120


def _stable_ordinal_pattern_counts(row: np.ndarray) -> np.ndarray:
    """Count stable order-5 patterns without sorting all pattern rows.

    The ranks below reproduce ``argsort(..., kind="mergesort")`` exactly:
    equal values retain their original sample order. Each rank tuple is then
    encoded in base 5, so its histogram has the same lexicographic count order
    returned by ``np.unique(..., axis=0)`` while avoiding that full row sort.
    """
    embedded = sliding_window_view(row, window_shape=OI_ORDER)
    pattern_count = embedded.shape[0]
    ranks = np.zeros((pattern_count, OI_ORDER), dtype=np.uint8)
    for column in range(OI_ORDER):
        rank = np.zeros(pattern_count, dtype=np.uint8)
        for other in range(OI_ORDER):
            if other == column:
                continue
            rank += (
                (embedded[:, other] < embedded[:, column])
                | ((embedded[:, other] == embedded[:, column]) & (other < column))
            )
        ranks[:, column] = rank

    # Ranks are unique within a pattern, so sort stability is immaterial here.
    order = np.argsort(ranks, axis=1, kind="quicksort")
    codes = (((order[:, 0] * OI_ORDER + order[:, 1]) * OI_ORDER + order[:, 2])
             * OI_ORDER + order[:, 3]) * OI_ORDER + order[:, 4]
    counts = np.bincount(codes, minlength=OI_ORDER ** OI_ORDER)
    return counts[counts > 0]


@dataclass(frozen=True)
class OIMatrixResult:
    """Result of one matrix-level OI attempt.

    The input array establishes its row and column counts, but not its actual
    recording time support. ``support_status`` consequently remains unknown.
    Optional channel labels describe matrix row order only; they do not identify
    a standardized/native sensor realization or its reference.
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
    per_channel_entropy: tuple[float, ...] | None
    pattern_count_per_channel: int | None
    qc: tuple[str, ...]
    claim_status: Literal["no_biological_claim"]

    def to_measurement_record(self) -> MeasurementRecord:
        """Return the shared envelope without inferring the upstream setup."""
        if self.channel_labels is None:
            audit_status = "unknown"
        elif self.channel_order_sha256 is not None and self.channel_set_sha256 is not None:
            # The labels and their order are known, but the actual sensor
            # realization and reference have not been audited by this kernel.
            audit_status = "incomplete"
        else:
            audit_status = "incomplete"

        return MeasurementRecord(
            family_id="OI",
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
                kind="matrix_input_rows_unresolved_sensor_realization",
                requested_id=None,
                actual_id=None,
                ordered_labels=self.channel_labels,
                order_sha256=self.channel_order_sha256,
                set_sha256=self.channel_set_sha256,
                count=self.channel_count,
                reference_id=None,
                reference_set_sha256=None,
            ),
            qc=self.qc,
            reproducibility=ReproducibilityRecord(status="not_supplied"),
            method_parameters=(
                ("input_orientation", "channels_by_samples"),
                ("ordinal_order", OI_ORDER),
                ("delay_samples", OI_DELAY_SAMPLES),
                ("tie_handling", "stable_mergesort_original_sample_order"),
                ("patterns_per_channel", self.pattern_count_per_channel),
                ("entropy", "shannon_natural_log"),
                ("normalization_denominator", "ln(5!)"),
                ("channel_aggregation", "median_over_all_supplied_matrix_rows"),
                ("non_finite_sample_policy", "reject_matrix"),
            ),
            direction="higher_more_ordinally_irregular",
            claim_status=self.claim_status,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible result with nulls retained as nulls."""
        payload = asdict(self)
        payload["channel_labels"] = (
            list(self.channel_labels) if self.channel_labels is not None else None
        )
        payload["per_channel_entropy"] = (
            list(self.per_channel_entropy)
            if self.per_channel_entropy is not None
            else None
        )
        payload["qc"] = list(self.qc)
        payload["measurement_record"] = self.to_measurement_record().to_dict()
        return payload


def _label_metadata(
    channel_names: Sequence[str] | None,
    channel_count: int,
) -> tuple[tuple[str, ...] | None, str | None, str | None, tuple[str, ...]]:
    """Validate optional row labels and hash exact order and unordered set."""
    if channel_names is None:
        return None, None, None, ("channel_labels_unavailable",)
    if isinstance(channel_names, (str, bytes, bytearray)):
        return None, None, None, ("channel_labels_must_be_sequence",)

    try:
        raw_labels = tuple(channel_names)
    except (TypeError, ValueError):
        return None, None, None, ("channel_labels_unreadable",)

    warnings: list[str] = []
    if len(raw_labels) != channel_count:
        warnings.append("channel_label_count_mismatch")
    if any(not isinstance(label, str) for label in raw_labels):
        warnings.append("channel_labels_must_be_strings")
        return None, None, None, tuple(warnings)

    labels = tuple(raw_labels)
    if any(not label.strip() for label in labels):
        warnings.append("empty_channel_label")
    normalized = tuple(label.strip().casefold() for label in labels)
    if len(set(normalized)) != len(normalized):
        warnings.append("duplicate_channel_label")
    if warnings:
        # Keep computable entropy when only optional audit metadata is bad.
        return labels, None, None, tuple(warnings)

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
    per_channel_entropy: tuple[float, ...] | None,
    pattern_count: int | None,
    qc: tuple[str, ...],
) -> OIMatrixResult:
    valid = value is not None
    return OIMatrixResult(
        estimator_id=OI_MATRIX_ESTIMATOR_ID,
        schema_id=OI_MATRIX_SCHEMA_ID,
        value=value,
        units=OI_UNITS,
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
        per_channel_entropy=per_channel_entropy,
        pattern_count_per_channel=pattern_count,
        qc=qc,
        claim_status="no_biological_claim",
    )


def compute_oi_matrix(
    signals: np.ndarray,
    channel_names: Sequence[str] | None = None,
) -> OIMatrixResult:
    """Compute normalized order-5/delay-1 permutation entropy on matrix rows.

    ``signals`` must be a finite real-valued ``(channels, samples)`` matrix.
    Each row is embedded into all consecutive length-5 patterns, giving
    ``T - 4`` patterns for ``T`` samples. Exact ordinal ties retain the
    original time-index order, matching a stable mergesort. Pattern counts are
    accumulated from their rank codes. Shannon entropy uses natural logarithms
    and is divided by ``ln(5!)``. The scalar is the median across every
    supplied matrix row, with no fixed channel list and no implicit
    missing-channel handling.

    The matrix-only API cannot establish preprocessing, reference, source
    identity, sample rate, gap-free support, or a named channel realization.
    Those fields remain unknown/incomplete in the common measurement record.
    Invalid optional labels are QC and do not discard otherwise computable
    values; non-finite signal samples and arrays with fewer than five samples
    are unsupported.
    """
    try:
        raw = np.asarray(signals)
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
            per_channel_entropy=None,
            pattern_count=None,
            qc=(
                "preprocessing_not_audited",
                "reference_not_audited",
                "sensor_realization_not_declared",
                "temporal_and_source_support_not_audited",
            ),
        )

    is_complex = np.iscomplexobj(raw)
    if raw.dtype.kind == "O":
        is_complex = is_complex or any(
            isinstance(item, (complex, np.complexfloating)) for item in raw.flat
        )

    channel_count = int(raw.shape[0]) if raw.ndim >= 1 else 0
    sample_count = int(raw.shape[1]) if raw.ndim == 2 else 0
    labels, order_hash, set_hash, label_qc = _label_metadata(
        channel_names, channel_count
    )
    audit_qc = (
        "preprocessing_not_audited",
        "reference_not_audited",
        "sensor_realization_not_declared",
        "temporal_and_source_support_not_audited",
    )

    def unsupported(reason: str, extra_qc: tuple[str, ...] = ()) -> OIMatrixResult:
        return _result(
            value=None,
            computed=False,
            reason=reason,
            channel_count=channel_count,
            sample_count=sample_count,
            labels=labels,
            order_hash=order_hash,
            set_hash=set_hash,
            per_channel_entropy=None,
            pattern_count=None,
            qc=label_qc + extra_qc + audit_qc,
        )

    if is_complex:
        return unsupported("complex_signal_not_supported")
    if raw.ndim != 2:
        return unsupported("signal_must_be_channels_by_samples")
    if channel_count < 1:
        return unsupported("no_channels")
    if sample_count < OI_ORDER:
        return unsupported("insufficient_samples_for_one_ordinal_pattern")

    try:
        values = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return unsupported("invalid_signal_array")
    if not np.isfinite(values).all():
        return unsupported("non_finite_sample")

    pattern_count = sample_count - OI_ORDER + 1
    per_channel: list[float] = []
    denominator = float(np.log(float(OI_PATTERN_COUNT)))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for row in values:
            counts = _stable_ordinal_pattern_counts(row)
            probabilities = counts.astype(np.float64) / float(pattern_count)
            entropy = -float(np.sum(probabilities * np.log(probabilities)))
            per_channel.append(entropy / denominator)

    entropies = np.asarray(per_channel, dtype=np.float64)
    if entropies.shape != (channel_count,) or not np.isfinite(entropies).all():
        return _result(
            value=None,
            computed=True,
            reason="non_finite_permutation_entropy",
            channel_count=channel_count,
            sample_count=sample_count,
            labels=labels,
            order_hash=order_hash,
            set_hash=set_hash,
            per_channel_entropy=None,
            pattern_count=pattern_count,
            qc=label_qc + ("permutation_entropy_numeric_failure",) + audit_qc,
        )

    value = float(np.median(entropies))
    if not np.isfinite(value):
        return _result(
            value=None,
            computed=True,
            reason="non_finite_channel_median",
            channel_count=channel_count,
            sample_count=sample_count,
            labels=labels,
            order_hash=order_hash,
            set_hash=set_hash,
            per_channel_entropy=tuple(float(item) for item in entropies),
            pattern_count=pattern_count,
            qc=label_qc + ("channel_aggregation_numeric_failure",) + audit_qc,
        )

    return _result(
        value=value,
        computed=True,
        reason=None,
        channel_count=channel_count,
        sample_count=sample_count,
        labels=labels,
        order_hash=order_hash,
        set_hash=set_hash,
        per_channel_entropy=tuple(float(item) for item in entropies),
        pattern_count=pattern_count,
        qc=label_qc + audit_qc,
    )


__all__ = [
    "OI_MATRIX_ESTIMATOR_ID",
    "OI_MATRIX_SCHEMA_ID",
    "OI_ORDER",
    "OI_DELAY_SAMPLES",
    "OIMatrixResult",
    "compute_oi_matrix",
]
