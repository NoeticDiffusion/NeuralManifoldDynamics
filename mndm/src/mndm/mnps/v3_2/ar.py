"""Experimental, low-level Activity Reconfiguration (AR) formula kernel.

The caller supplies one explicitly constructed activity vector per adjacent
block. This module does not choose the feature construction, sensors, scaling,
block partition, filtering, or recording support. It returns every adjacent
pair in order and never aggregates the vector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Sequence

import numpy as np

from .contracts import MeasurementRecord, RealizationRecord, ReproducibilityRecord, TimeSupportRecord


AR_ACTIVITY_ESTIMATOR_ID = "AR.sensor_activity_adjacent_pearson_change_vector.v1"
AR_ACTIVITY_SCHEMA_ID = "nmdm.mnps_v3_2.ar_activity_vector_result.v1"


def _labels_metadata(labels: Sequence[str] | None, channel_count: int):
    if labels is None:
        return None, None, None, ("channel_labels_unavailable",)
    if isinstance(labels, (str, bytes, bytearray)):
        return None, None, None, ("channel_labels_must_be_sequence",)
    try:
        result = tuple(labels)
    except (TypeError, ValueError):
        return None, None, None, ("channel_labels_unreadable",)
    if any(not isinstance(label, str) or not label.strip() for label in result):
        return None, None, None, ("channel_labels_must_be_nonempty_strings",)
    if len(result) != channel_count:
        return result, None, None, ("channel_label_count_mismatch",)
    normalized = tuple(label.strip().casefold() for label in result)
    if len(set(normalized)) != len(normalized):
        return result, None, None, ("duplicate_channel_label",)
    ordered = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    unordered = json.dumps(sorted(result), ensure_ascii=False, separators=(",", ":"))
    return (
        result,
        hashlib.sha256(ordered.encode("utf-8")).hexdigest(),
        hashlib.sha256(unordered.encode("utf-8")).hexdigest(),
        (),
    )


@dataclass(frozen=True)
class ARActivityResult:
    """Ordered component result, including pair-local undefined values."""

    estimator_id: str
    schema_id: str
    component_values: tuple[float | None, ...]
    component_names: tuple[str, ...]
    component_status: tuple[str, ...]
    component_reasons: tuple[str | None, ...]
    computed: bool
    valid: bool
    qualified: bool
    status: str
    reason: str | None
    block_intervals_sec: tuple[tuple[float, float], ...] | None
    channel_count: int
    vector_construction_id: str | None
    scaling_id: str | None
    channel_labels: tuple[str, ...] | None
    channel_order_sha256: str | None
    channel_set_sha256: str | None
    qc: tuple[str, ...]

    def to_measurement_record(self) -> MeasurementRecord:
        all_components_valid = bool(self.component_values) and all(
            value is not None for value in self.component_values
        )
        value = (
            tuple(float(item) for item in self.component_values if item is not None)
            if all_components_valid
            else None
        )
        return MeasurementRecord(
            family_id="AR",
            estimator_id=self.estimator_id,
            schema_id=self.schema_id,
            value=value,
            units="1",
            computed=self.computed,
            valid=value is not None,
            qualified=False,
            status="computed_exploratory" if value is not None else "unsupported",
            reason=None if value is not None else self.reason or "no_complete_component_vector",
            support=TimeSupportRecord(support_status="unknown"),
            realization=RealizationRecord(
                audit_status="complete" if self.channel_order_sha256 else "incomplete",
                kind="caller_constructed_activity_vectors",
                requested_id=None,
                actual_id=None,
                ordered_labels=self.channel_labels,
                order_sha256=self.channel_order_sha256,
                set_sha256=self.channel_set_sha256,
                count=self.channel_count,
            ),
            qc=self.qc,
            reproducibility=ReproducibilityRecord(status="not_supplied"),
            value_components=self.component_names if value is not None else None,
            method_parameters=(
                ("formula", "1 - PearsonR_sensor(v_b, v_b_plus_1)"),
                ("vector_construction_id", self.vector_construction_id),
                ("scaling_id", self.scaling_id),
                ("aggregation", "none_ordered_adjacent_pairs"),
                ("undefined_pair_policy", "component_local_null_preserved_in_result"),
            ),
            direction="higher_more_activity_pattern_reconfiguration",
            claim_status="no_biological_claim",
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["channel_labels"] = list(self.channel_labels) if self.channel_labels is not None else None
        payload["block_intervals_sec"] = (
            [list(pair) for pair in self.block_intervals_sec]
            if self.block_intervals_sec is not None else None
        )
        payload["component_values"] = list(self.component_values)
        payload["component_names"] = list(self.component_names)
        payload["component_status"] = list(self.component_status)
        payload["component_reasons"] = list(self.component_reasons)
        payload["qc"] = list(self.qc)
        payload["measurement_record"] = self.to_measurement_record().to_dict()
        return payload


def _unsupported(
    reason: str,
    *,
    channel_count: int = 0,
    vector_construction_id: str | None = None,
    scaling_id: str | None = None,
    intervals: tuple[tuple[float, float], ...] | None = None,
    labels: tuple[str, ...] | None = None,
    order_hash: str | None = None,
    set_hash: str | None = None,
    qc: tuple[str, ...] = (),
) -> ARActivityResult:
    return ARActivityResult(
        estimator_id=AR_ACTIVITY_ESTIMATOR_ID,
        schema_id=AR_ACTIVITY_SCHEMA_ID,
        component_values=(), component_names=(), component_status=(), component_reasons=(),
        computed=False, valid=False, qualified=False, status="unsupported", reason=reason,
        block_intervals_sec=intervals, channel_count=channel_count,
        vector_construction_id=vector_construction_id, scaling_id=scaling_id,
        channel_labels=labels, channel_order_sha256=order_hash,
        channel_set_sha256=set_hash, qc=qc,
    )


def compute_ar_activity_transition_vector(
    block_vectors: np.ndarray,
    block_intervals_sec: Sequence[Sequence[float]],
    *,
    vector_construction_id: str | None,
    scaling_id: str | None,
    channel_labels: Sequence[str] | None = None,
) -> ARActivityResult:
    """Compute ordered adjacent changes ``1 - Pearson r`` over sensors.

    ``block_vectors`` is ``(blocks, sensors)`` and is constructed upstream.
    Its feature definition and scaling must be named. Blocks are required to
    be positive, equal-duration, contiguous intervals. Float-second interval
    boundaries use a 1 ns absolute tolerance (no scale-dependent relative
    allowance);
    sample-index boundaries are preferable when available. A
    constant-across-sensor vector makes only its adjacent component undefined;
    no channel is dropped and the component order is never aggregated.
    """
    construction = vector_construction_id.strip() if isinstance(vector_construction_id, str) and vector_construction_id.strip() else None
    scaling = scaling_id.strip() if isinstance(scaling_id, str) and scaling_id.strip() else None
    try:
        raw = np.asarray(block_vectors)
        if np.iscomplexobj(raw):
            return _unsupported("block_vectors_must_be_real_2d", vector_construction_id=construction, scaling_id=scaling)
        values = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return _unsupported("invalid_block_vectors", vector_construction_id=construction, scaling_id=scaling)
    if values.ndim != 2:
        return _unsupported("block_vectors_must_be_real_2d", vector_construction_id=construction, scaling_id=scaling)
    block_count, channel_count = values.shape
    if block_count < 2:
        return _unsupported("at_least_two_blocks_required", channel_count=channel_count, vector_construction_id=construction, scaling_id=scaling)
    if channel_count < 2:
        return _unsupported("at_least_two_sensor_features_required", channel_count=channel_count, vector_construction_id=construction, scaling_id=scaling)
    labels, order_hash, set_hash, label_qc = _labels_metadata(channel_labels, channel_count)
    if not np.isfinite(values).all():
        return _unsupported("non_finite_block_vector", channel_count=channel_count, vector_construction_id=construction, scaling_id=scaling, labels=labels, order_hash=order_hash, set_hash=set_hash, qc=label_qc)
    try:
        intervals = np.asarray(block_intervals_sec, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return _unsupported("invalid_block_intervals", channel_count=channel_count, vector_construction_id=construction, scaling_id=scaling, labels=labels, order_hash=order_hash, set_hash=set_hash, qc=label_qc)
    if intervals.shape != (block_count, 2) or not np.isfinite(intervals).all():
        return _unsupported("block_interval_shape_or_values_invalid", channel_count=channel_count, vector_construction_id=construction, scaling_id=scaling, labels=labels, order_hash=order_hash, set_hash=set_hash, qc=label_qc)
    durations = intervals[:, 1] - intervals[:, 0]
    if np.any(durations <= 0.0):
        return _unsupported("block_intervals_must_have_positive_duration", channel_count=channel_count, vector_construction_id=construction, scaling_id=scaling, labels=labels, order_hash=order_hash, set_hash=set_hash, qc=label_qc)
    interval_tolerance_sec = 1e-9
    if not np.allclose(durations, durations[0], rtol=0.0, atol=interval_tolerance_sec):
        return _unsupported("block_durations_must_match", channel_count=channel_count, vector_construction_id=construction, scaling_id=scaling, labels=labels, order_hash=order_hash, set_hash=set_hash, qc=label_qc)
    if not np.allclose(intervals[:-1, 1], intervals[1:, 0], rtol=0.0, atol=interval_tolerance_sec):
        return _unsupported("blocks_not_adjacent_or_gap_free", channel_count=channel_count, vector_construction_id=construction, scaling_id=scaling, labels=labels, order_hash=order_hash, set_hash=set_hash, qc=label_qc)

    values_out: list[float | None] = []
    names: list[str] = []
    statuses: list[str] = []
    reasons: list[str | None] = []
    for index in range(block_count - 1):
        names.append(f"block_{index:03d}_to_{index + 1:03d}")
        left = values[index] - np.mean(values[index])
        right = values[index + 1] - np.mean(values[index + 1])
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm == 0.0 or right_norm == 0.0:
            values_out.append(None)
            statuses.append("unsupported")
            reasons.append("constant_sensor_vector_correlation_undefined")
            continue
        correlation = float(np.dot(left, right) / (left_norm * right_norm))
        if not np.isfinite(correlation):
            values_out.append(None)
            statuses.append("unsupported")
            reasons.append("non_finite_sensor_correlation")
            continue
        values_out.append(float(1.0 - correlation))
        statuses.append("computed_exploratory")
        reasons.append(None)
    partial = any(value is None for value in values_out)
    return ARActivityResult(
        estimator_id=AR_ACTIVITY_ESTIMATOR_ID,
        schema_id=AR_ACTIVITY_SCHEMA_ID,
        component_values=tuple(values_out), component_names=tuple(names),
        component_status=tuple(statuses), component_reasons=tuple(reasons),
        computed=True, valid=not partial, qualified=False,
        status="partial" if partial else "computed_exploratory",
        reason="one_or_more_adjacent_pairs_undefined" if partial else None,
        block_intervals_sec=tuple((float(start), float(end)) for start, end in intervals),
        channel_count=channel_count, vector_construction_id=construction,
        scaling_id=scaling, channel_labels=labels, channel_order_sha256=order_hash,
        channel_set_sha256=set_hash,
        qc=label_qc + (() if construction is not None and scaling is not None else ("vector_construction_or_scaling_provenance_missing",)),
    )


__all__ = ["AR_ACTIVITY_ESTIMATOR_ID", "AR_ACTIVITY_SCHEMA_ID", "ARActivityResult", "compute_ar_activity_transition_vector"]
