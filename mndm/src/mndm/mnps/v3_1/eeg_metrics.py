"""Unqualified EEG sensor-space prototypes for MNPS v3.1.

These functions implement separate, explicitly named estimands so they can
be compared on audited windows later. The sensor-space measures are not
anatomical regions, do not fill chart slots, and never set ``qualified=True``.
They consume arrays assembled by a caller; extraction, full provenance, and
scientific qualification remain separate gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from typing import Literal, Sequence

import numpy as np

from .candidates import MNPS31WindowSupport


@dataclass(frozen=True)
class EEGCandidateMeasurement:
    """A computed value with technical and scientific status kept separate."""

    candidate_id: str
    estimator_id: str
    value: float | tuple[float, ...] | None
    units: str
    observation_space: str
    computed: bool
    valid: bool
    qualified: bool
    status: Literal["computed_exploratory", "unsupported"]
    reason: str | None
    support: MNPS31WindowSupport
    details: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["claim_status"] = "no_biological_claim"
        payload["coordinate_assignment"] = None
        return payload


def _labels_hash(channel_names: Sequence[str]) -> str:
    return hashlib.sha256(
        "\n".join(str(name).strip().casefold() for name in channel_names).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdefABCDEF" for char in value)


def _unsupported(
    *,
    candidate_id: str,
    estimator_id: str,
    support: MNPS31WindowSupport,
    reason: str,
    units: str = "1",
    observation_space: str = "sensor_only",
    details: dict[str, object] | None = None,
) -> EEGCandidateMeasurement:
    return EEGCandidateMeasurement(
        candidate_id=candidate_id,
        estimator_id=estimator_id,
        value=None,
        units=units,
        observation_space=observation_space,
        computed=False,
        valid=False,
        qualified=False,
        status="unsupported",
        reason=reason,
        support=support,
        details=details or {},
    )


def _computed(
    *,
    candidate_id: str,
    estimator_id: str,
    value: float | tuple[float, ...],
    support: MNPS31WindowSupport,
    units: str = "1",
    observation_space: str = "sensor_only",
    details: dict[str, object] | None = None,
) -> EEGCandidateMeasurement:
    return EEGCandidateMeasurement(
        candidate_id=candidate_id,
        estimator_id=estimator_id,
        value=value,
        units=units,
        observation_space=observation_space,
        computed=True,
        valid=True,
        qualified=False,
        status="computed_exploratory",
        reason=None,
        support=support,
        details=details or {},
    )


def _support_reason(support: MNPS31WindowSupport) -> str | None:
    flags = (
        support.source_is_continuous,
        support.timestamps_uniform,
        support.gap_free,
        support.bad_interval_free,
        support.within_recording_bounds,
        support.effective_filter_support_valid,
    )
    if any(flag is None for flag in flags):
        return "support_audit_incomplete"
    if any(not isinstance(flag, (bool, np.bool_)) for flag in flags):
        return "support_audit_flag_not_boolean"
    checks = (
        (support.source_is_continuous, "non_continuous_source"),
        (support.timestamps_uniform, "irregular_sample_times"),
        (support.gap_free, "gap_or_segment_boundary"),
        (support.bad_interval_free, "bad_interval_overlap"),
        (support.within_recording_bounds, "recording_boundary_overrun"),
        (support.effective_filter_support_valid, "insufficient_filter_edge_support"),
    )
    for passed, reason in checks:
        if not bool(passed):
            return reason
    if not support.segment_id.strip():
        return "missing_segment_id"
    if support.sample_count < 2:
        return "insufficient_sample_count"
    if not np.isfinite(support.sample_rate_hz) or support.sample_rate_hz <= 0:
        return "invalid_sampling_rate"
    if not np.isfinite(support.window_start_sec) or not np.isfinite(support.window_end_sec):
        return "invalid_window_bounds"
    expected_duration = support.sample_count / support.sample_rate_hz
    actual_duration = support.window_end_sec - support.window_start_sec
    if actual_duration <= 0 or not np.isclose(
        actual_duration, expected_duration, rtol=0.0, atol=1e-6
    ):
        return "window_time_support_mismatch"
    return None


def _validate_labels(channel_names: Sequence[str]) -> str | None:
    keys = [str(name).strip().casefold() for name in channel_names]
    if not keys or any(not name for name in keys):
        return "empty_channel_set_or_label"
    if len(set(keys)) != len(keys):
        return "duplicate_channel_label"
    return None


def _validate_matrix(
    values: np.ndarray,
    support: MNPS31WindowSupport,
    *,
    first_dimension: int | None = None,
) -> tuple[np.ndarray | None, str | None]:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return None, "invalid_numeric_input"
    if array.ndim != 2:
        return None, "input_must_be_two_dimensional"
    if first_dimension is not None and array.shape[0] != first_dimension:
        return None, "channel_count_mismatch"
    if array.shape[1] != support.sample_count:
        return None, "sample_count_mismatch"
    if not np.isfinite(array).all():
        return None, "non_finite_input"
    return array, None


def compute_m1_sensor_phase_variability(
    phase_radians: np.ndarray,
    channel_names: Sequence[str],
    support: MNPS31WindowSupport,
    *,
    phase_estimator_id: str,
    band_id: str,
) -> EEGCandidateMeasurement:
    """Compute SD over time of the sensor-set phase order parameter.

    ``phase_estimator_id`` and ``band_id`` must identify the upstream phase
    construction. The result is sensor-space coordination variability; it is
    not regional metastability or a chart coordinate.
    """
    candidate_id, estimator_id = "M1", "M1.sensor_phase_order_parameter_sd.v0"
    support_error = _support_reason(support)
    if support_error:
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason=support_error)
    label_error = _validate_labels(channel_names)
    if label_error:
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason=label_error)
    if len(channel_names) < 2:
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="insufficient_channels")
    if not phase_estimator_id.strip() or not band_id.strip():
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="phase_or_band_provenance_missing")
    phase, error = _validate_matrix(phase_radians, support, first_dimension=len(channel_names))
    if error:
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason=error)
    if np.any(np.abs(phase) > np.pi + 1e-12):
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="phase_outside_wrapped_radian_range")
    order_parameter = np.abs(np.mean(np.exp(1j * phase), axis=0))
    value = float(np.std(order_parameter, ddof=0))
    if not np.isfinite(value):
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="non_finite_estimate")
    return _computed(
        candidate_id=candidate_id,
        estimator_id=estimator_id,
        value=value,
        support=support,
        details={
            "channel_order_sha256": _labels_hash(channel_names),
            "channel_count": len(channel_names),
            "phase_estimator_id": phase_estimator_id,
            "band_id": band_id,
            "aggregation": "population_sd_over_samples",
            "space_claim": "sensor_only",
        },
    )


def _adjacent_vector_changes(vectors: np.ndarray) -> tuple[float, ...] | None:
    changes: list[float] = []
    for left, right in zip(vectors[:-1], vectors[1:]):
        left_scale = float(np.max(np.abs(left)))
        right_scale = float(np.max(np.abs(right)))
        if left_scale == 0.0 or right_scale == 0.0:
            return None
        left_scaled = left / left_scale
        right_scaled = right / right_scale
        left_centered = left_scaled - np.mean(left_scaled)
        right_centered = right_scaled - np.mean(right_scaled)
        denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
        if denominator == 0.0:
            return None
        correlation = float(np.dot(left_centered, right_centered) / denominator)
        if not np.isfinite(correlation):
            return None
        changes.append(1.0 - float(np.clip(correlation, -1.0, 1.0)))
    return tuple(changes)


def _validate_adjacent_blocks(
    vectors: np.ndarray,
    block_intervals_sec: np.ndarray,
    support: MNPS31WindowSupport,
) -> str | None:
    if vectors.ndim != 2 or vectors.shape[0] < 2 or vectors.shape[1] < 2:
        return "insufficient_blocks_or_vector_features"
    try:
        intervals = np.asarray(block_intervals_sec, dtype=np.float64)
    except (TypeError, ValueError):
        return "invalid_block_intervals"
    if intervals.shape != (vectors.shape[0], 2) or not np.isfinite(intervals).all():
        return "invalid_block_intervals"
    durations = intervals[:, 1] - intervals[:, 0]
    if np.any(durations <= 0) or not np.allclose(durations, durations[0], rtol=0.0, atol=1e-9):
        return "blocks_must_have_equal_positive_duration"
    if not np.allclose(intervals[:-1, 1], intervals[1:, 0], rtol=0.0, atol=1e-6):
        return "blocks_not_adjacent_or_gap_free"
    if intervals[0, 0] < support.window_start_sec or intervals[-1, 1] > support.window_end_sec:
        return "block_outside_audited_window"
    if not np.isclose(intervals[0, 0], support.window_start_sec, rtol=0.0, atol=1e-6) or not np.isclose(
        intervals[-1, 1], support.window_end_sec, rtol=0.0, atol=1e-6
    ):
        return "blocks_do_not_cover_audited_window"
    return None


def compute_m3_sensor_activity_reconfiguration(
    block_activity_vectors: np.ndarray,
    block_intervals_sec: np.ndarray,
    channel_names: Sequence[str],
    support: MNPS31WindowSupport,
    *,
    vector_construction_id: str,
    vector_scaling_id: str,
) -> EEGCandidateMeasurement:
    """Return 1−Pearson-r for each adjacent sensor activity-vector pair.

    Values are returned per pair; no across-pair aggregation is imposed. The
    input vectors must use the same declared channels and construction.
    """
    candidate_id, estimator_id = "M3", "M3.sensor_activity_adjacent_pearson_change.v0"
    support_error = _support_reason(support)
    if support_error:
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason=support_error)
    label_error = _validate_labels(channel_names)
    if label_error:
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason=label_error)
    try:
        vectors = np.asarray(block_activity_vectors, dtype=np.float64)
    except (TypeError, ValueError):
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="invalid_numeric_input")
    if not np.isfinite(vectors).all():
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="non_finite_input")
    if vectors.ndim != 2 or vectors.shape[1] != len(channel_names):
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="channel_count_mismatch")
    if not vector_construction_id.strip() or not vector_scaling_id.strip():
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="vector_construction_provenance_missing")
    blocks_error = _validate_adjacent_blocks(vectors, block_intervals_sec, support)
    if blocks_error:
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason=blocks_error)
    intervals = np.asarray(block_intervals_sec, dtype=np.float64)
    changes = _adjacent_vector_changes(vectors)
    if changes is None:
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="constant_block_vector")
    return _computed(
        candidate_id=candidate_id,
        estimator_id=estimator_id,
        value=changes,
        support=support,
        details={
            "channel_order_sha256": _labels_hash(channel_names),
            "vector_construction_id": vector_construction_id,
            "vector_scaling_id": vector_scaling_id,
            "block_intervals_sec": intervals.tolist(),
            "block_duration_sec": float(intervals[0, 1] - intervals[0, 0]),
            "block_pair_count": len(changes),
            "aggregation": "none_per_adjacent_pair",
            "space_claim": "sensor_only",
        },
    )


def compute_m3_sensor_connectivity_reconfiguration(
    block_connectivity_vectors: np.ndarray,
    block_intervals_sec: np.ndarray,
    support: MNPS31WindowSupport,
    *,
    edge_order_sha256: str,
    edge_estimator_id: str,
    edge_scaling_id: str,
) -> EEGCandidateMeasurement:
    """Return adjacent-block 1−Pearson-r values for a fixed sensor-edge set.

    This is a separate M3 connectivity variant, not interchangeable with the
    activity-vector variant. The edge matrix, estimator, and ordering are
    supplied by the caller and remain part of the study provenance.
    """
    candidate_id, estimator_id = "M3", "M3.sensor_connectivity_adjacent_pearson_change.v0"
    support_error = _support_reason(support)
    if support_error:
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason=support_error)
    try:
        vectors = np.asarray(block_connectivity_vectors, dtype=np.float64)
    except (TypeError, ValueError):
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="invalid_numeric_input")
    if vectors.ndim != 2 or vectors.shape[0] < 2 or vectors.shape[1] < 2:
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="insufficient_blocks_or_edges")
    if not np.isfinite(vectors).all():
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="non_finite_input")
    if not _is_sha256(edge_order_sha256) or not edge_estimator_id.strip() or not edge_scaling_id.strip():
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="edge_provenance_missing")
    blocks_error = _validate_adjacent_blocks(vectors, block_intervals_sec, support)
    if blocks_error:
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason=blocks_error)
    intervals = np.asarray(block_intervals_sec, dtype=np.float64)
    changes = _adjacent_vector_changes(vectors)
    if changes is None:
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="constant_block_vector")
    return _computed(
        candidate_id=candidate_id,
        estimator_id=estimator_id,
        value=changes,
        support=support,
        details={
            "edge_order_sha256": edge_order_sha256,
            "edge_estimator_id": edge_estimator_id,
            "edge_scaling_id": edge_scaling_id,
            "block_intervals_sec": intervals.tolist(),
            "block_duration_sec": float(intervals[0, 1] - intervals[0, 0]),
            "block_pair_count": len(changes),
            "aggregation": "none_per_adjacent_pair",
            "space_claim": "sensor_only",
        },
    )


def compute_d1_sensor_weighted_global_efficiency(
    edge_weights: np.ndarray,
    support: MNPS31WindowSupport,
    *,
    node_order_sha256: str,
    edge_estimator_id: str,
    edge_rule_id: str,
) -> EEGCandidateMeasurement:
    """Compute weighted global efficiency with length ``1 / weight``.

    Zero weights denote absent edges; weights must be finite, symmetric, and
    in [0,1]. Disconnected node pairs contribute zero. This sensor-graph
    statistic is a sidecar and makes no cortical graph claim.
    """
    candidate_id, estimator_id = "D1", "D1.sensor_weighted_global_efficiency_invweight.v0"
    support_error = _support_reason(support)
    if support_error:
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason=support_error)
    try:
        weights = np.asarray(edge_weights, dtype=np.float64)
    except (TypeError, ValueError):
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="invalid_numeric_input")
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1] or weights.shape[0] < 2:
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="invalid_graph_shape")
    if not np.isfinite(weights).all():
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="non_finite_input")
    if not _is_sha256(node_order_sha256):
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="node_order_hash_missing_or_invalid")
    if not edge_estimator_id.strip() or not edge_rule_id.strip():
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="edge_provenance_missing")
    if np.any(weights < 0.0) or np.any(weights > 1.0) or not np.allclose(weights, weights.T, rtol=0.0, atol=1e-8):
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="weights_must_be_symmetric_in_unit_interval")
    if not np.allclose(np.diag(weights), 0.0, rtol=0.0, atol=1e-12):
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="nonzero_graph_diagonal")

    distances = np.full(weights.shape, np.inf, dtype=np.float64)
    np.fill_diagonal(distances, 0.0)
    connected = weights > 0.0
    with np.errstate(over="ignore", divide="ignore"):
        distances[connected] = 1.0 / weights[connected]
    if not np.isfinite(distances[connected]).all():
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="edge_length_overflow")
    for pivot in range(weights.shape[0]):
        with np.errstate(over="ignore", invalid="ignore"):
            candidate = distances[:, pivot, None] + distances[None, pivot, :]
        if np.any(np.isfinite(distances[:, pivot, None]) & np.isfinite(distances[None, pivot, :]) & ~np.isfinite(candidate)):
            return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="path_length_overflow")
        distances = np.minimum(distances, candidate)
    inverse = np.zeros_like(distances)
    np.divide(1.0, distances, out=inverse, where=np.isfinite(distances) & (distances > 0.0))
    value = float(np.sum(inverse) / (weights.shape[0] * (weights.shape[0] - 1)))
    return _computed(
        candidate_id=candidate_id,
        estimator_id=estimator_id,
        value=value,
        support=support,
        details={
            "node_count": int(weights.shape[0]),
            "node_order_sha256": node_order_sha256.lower(),
            "edge_estimator_id": edge_estimator_id,
            "edge_rule_id": edge_rule_id,
            "weight_to_length": "1/weight",
            "disconnected_pair_value": 0.0,
            "space_claim": "sensor_graph_only",
        },
    )


def compute_d2_sensor_pairclass_difference(
    coupling: np.ndarray,
    class_a: np.ndarray,
    class_b: np.ndarray,
    support: MNPS31WindowSupport,
    *,
    variant: Literal["near_far", "within_between"],
    pair_definition_id: str,
    coupling_estimator_id: str,
    node_order_sha256: str,
) -> EEGCandidateMeasurement:
    """Compute mean coupling(class A) minus mean coupling(class B).

    The caller supplies separate, disjoint, symmetric masks for both classes.
    This permits intermediate/unclassified pairs to remain excluded. Pair
    geometry/community construction is not learned from the coupling window
    by this function.
    """
    candidate_id = "D2"
    if variant not in {"near_far", "within_between"}:
        return _unsupported(candidate_id=candidate_id, estimator_id="D2.sensor_unknown_pairclass.v0", support=support, reason="unknown_pairclass_variant")
    estimator_id = f"D2.sensor_{variant}_coupling_difference.v0"
    support_error = _support_reason(support)
    if support_error:
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason=support_error)
    try:
        values = np.asarray(coupling, dtype=np.float64)
        mask_a = np.asarray(class_a)
        mask_b = np.asarray(class_b)
    except (TypeError, ValueError):
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="invalid_numeric_input")
    if (
        values.ndim != 2
        or values.shape[0] != values.shape[1]
        or mask_a.shape != values.shape
        or mask_b.shape != values.shape
        or values.shape[0] < 2
    ):
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="invalid_pair_matrix_shape")
    if mask_a.dtype.kind != "b" or mask_b.dtype.kind != "b":
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="pair_class_mask_must_be_boolean")
    if not np.isfinite(values).all():
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="non_finite_input")
    if (
        not np.allclose(values, values.T, rtol=0.0, atol=1e-8)
        or not np.array_equal(mask_a, mask_a.T)
        or not np.array_equal(mask_b, mask_b.T)
    ):
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="pair_matrices_must_be_symmetric")
    if np.any(mask_a.diagonal()) or np.any(mask_b.diagonal()):
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="diagonal_in_pair_class")
    if np.any(mask_a & mask_b):
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="overlapping_pair_classes")
    if np.any(values < -1.0) or np.any(values > 1.0):
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="coupling_outside_signed_unit_interval")
    if not pair_definition_id.strip() or not coupling_estimator_id.strip() or not _is_sha256(node_order_sha256):
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="pair_or_coupling_provenance_missing")
    upper = np.triu(np.ones(values.shape, dtype=bool), k=1)
    selected_a = mask_a & upper
    selected_b = mask_b & upper
    if not np.any(selected_a) or not np.any(selected_b):
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="insufficient_pair_class_support")
    value = float(np.mean(values[selected_a]) - np.mean(values[selected_b]))
    return _computed(
        candidate_id=candidate_id,
        estimator_id=estimator_id,
        value=value,
        support=support,
        details={
            "variant": variant,
            "pair_definition_id": pair_definition_id,
            "coupling_estimator_id": coupling_estimator_id,
            "node_order_sha256": node_order_sha256.lower(),
            "class_a_pair_count": int(np.count_nonzero(selected_a)),
            "class_b_pair_count": int(np.count_nonzero(selected_b)),
            "contrast_order": "class_a_minus_class_b",
            "space_claim": "sensor_only",
        },
    )


def compute_d3_sensor_rdm_dispersion(
    feature_patterns: np.ndarray,
    channel_names: Sequence[str],
    support: MNPS31WindowSupport,
    *,
    feature_construction_id: str,
    scaling_id: str,
    feature_units: str,
) -> EEGCandidateMeasurement:
    """Compute mean off-diagonal Euclidean distance between channel patterns."""
    candidate_id, estimator_id = "D3", "D3.sensor_rdm_mean_euclidean.v0"
    support_error = _support_reason(support)
    if support_error:
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason=support_error, units=feature_units)
    label_error = _validate_labels(channel_names)
    if label_error:
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason=label_error, units=feature_units)
    try:
        patterns = np.asarray(feature_patterns, dtype=np.float64)
    except (TypeError, ValueError):
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="invalid_numeric_input", units=feature_units)
    if patterns.ndim != 2 or patterns.shape[0] != len(channel_names) or patterns.shape[0] < 2 or patterns.shape[1] < 1:
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="invalid_pattern_matrix_shape", units=feature_units)
    if not np.isfinite(patterns).all():
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="non_finite_input", units=feature_units)
    if not feature_construction_id.strip() or not scaling_id.strip() or not feature_units.strip():
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="feature_or_scaling_provenance_missing", units=feature_units)
    with np.errstate(over="ignore", invalid="ignore"):
        differences = patterns[:, None, :] - patterns[None, :, :]
        distances = np.sqrt(np.sum(differences * differences, axis=-1))
    if not np.isfinite(distances).all():
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="distance_overflow", units=feature_units)
    upper = np.triu(np.ones(distances.shape, dtype=bool), k=1)
    value = float(np.mean(distances[upper]))
    return _computed(
        candidate_id=candidate_id,
        estimator_id=estimator_id,
        value=value,
        support=support,
        units=feature_units,
        details={
            "channel_order_sha256": _labels_hash(channel_names),
            "feature_construction_id": feature_construction_id,
            "scaling_id": scaling_id,
            "feature_units": feature_units,
            "distance": "euclidean",
            "pair_count": int(np.count_nonzero(upper)),
            "space_claim": "sensor_pattern_only",
        },
    )


def _lz78_phrase_count(symbols: tuple[object, ...]) -> int:
    # Keep the same incremental LZ78 dictionary semantics without allocating
    # and hashing every growing tuple slice. Each trie node is a dictionary;
    # a private sentinel marks phrases already present in the dictionary.
    terminal = object()
    root: dict[object, Any] = {}
    cursor = 0
    phrase_count = 0
    symbol_count = len(symbols)
    while cursor < symbol_count:
        node = root
        end = cursor
        while end < symbol_count:
            symbol = symbols[end]
            child = node.get(symbol)
            if child is None:
                child = {}
                node[symbol] = child
                child[terminal] = True
                phrase_count += 1
                end += 1
                break
            node = child
            end += 1
            if terminal not in node:
                node[terminal] = True
                phrase_count += 1
                break
        cursor = end
    return phrase_count


def _lz78_phrase_count_binary(symbols: np.ndarray) -> int:
    """Exact compact-trie fast path for one-dimensional binary symbols.

    Pipeline E2 obtains these symbols from a strict comparison against each
    channel's median. The parser follows the same insertion points as
    :func:`_lz78_phrase_count`, using two integer transitions per trie node
    instead of nested Python dictionaries. Converting the uint8 stream to
    ``bytes`` avoids allocating one Python integer per sample. RNG generation,
    symbol order, phrase count, and all aggregation remain unchanged.
    """
    values = np.asarray(symbols)
    if values.ndim != 1 or values.dtype != np.uint8:
        raise ValueError("binary LZ78 fast path expects a one-dimensional uint8 array")
    if np.any(values > 1):
        raise ValueError("binary LZ78 fast path accepts only symbols 0 and 1")

    data = values.tobytes(order="C")
    children = [-1, -1]  # root's zero and one transitions
    cursor = 0
    phrase_count = 0
    size = len(data)
    while cursor < size:
        node = 0
        end = cursor
        while end < size:
            edge = (node << 1) | data[end]
            child = children[edge]
            if child < 0:
                child = len(children) >> 1
                children.extend((-1, -1))
                children[edge] = child
                phrase_count += 1
                end += 1
                break
            node = child
            end += 1
        cursor = end
    return phrase_count


def compute_e2_sensor_lz78_phrase_density(
    symbols: Sequence[int | str],
    support: MNPS31WindowSupport,
    *,
    channel_name: str,
    symbolization_id: str,
) -> EEGCandidateMeasurement:
    """Compute unadjusted LZ78 phrase density on one pre-symbolized EEG stream.

    This is an explicitly unadjusted E2 algorithm probe. It does not choose a
    symbolization rule, channel aggregation, or the matched-reference
    normalization specified as a future E2 decision.
    """
    candidate_id, estimator_id = "E2", "E2.sensor_lz78_phrase_density_unadjusted.v0"
    support_error = _support_reason(support)
    if support_error:
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason=support_error, observation_space="sensor_channel")
    if len(symbols) != support.sample_count:
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="sample_count_mismatch", observation_space="sensor_channel")
    if not channel_name.strip() or not symbolization_id.strip():
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="symbolization_provenance_missing", observation_space="sensor_channel")
    if any(not isinstance(symbol, (int, str, np.integer)) for symbol in symbols):
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="symbols_must_be_discrete_integers_or_strings", observation_space="sensor_channel")
    sequence = tuple(symbols)
    alphabet_size = len(set(sequence))
    if alphabet_size < 2:
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="symbol_sequence_has_fewer_than_two_symbols", observation_space="sensor_channel")
    phrase_count = _lz78_phrase_count(sequence)
    value = float(phrase_count / len(sequence))
    return _computed(
        candidate_id=candidate_id,
        estimator_id=estimator_id,
        value=value,
        support=support,
        observation_space="sensor_channel",
        details={
            "channel_name": channel_name,
            "symbolization_id": symbolization_id,
            "lz_variant": "LZ78 incremental dictionary parsing",
            "phrase_count": phrase_count,
            "sequence_length": len(sequence),
            "alphabet_size_observed": alphabet_size,
            "normalization": "phrase_count_per_symbol_only",
            "matched_reference_normalization": "not_applied",
            "channel_aggregation": "not_applied_single_channel",
        },
    )


def compute_e2_sensor_lz78_permutation_reference_ratio(
    symbols: Sequence[int | str],
    support: MNPS31WindowSupport,
    *,
    channel_name: str,
    symbolization_id: str,
    reference_replicates: int = 32,
    random_seed: int = 310905,
) -> EEGCandidateMeasurement:
    """Compare LZ78 phrase count with exact-multiset shuffled references.

    This is an unratified E2 normalization probe, not the selected E2 scalar.
    The reference sequences have the same length and exact symbol counts as
    the observed sequence; only temporal order is randomized. The ratio is
    observed phrase count divided by the median reference phrase count.
    Symbolization and channel aggregation remain caller-owned and unresolved.
    """
    base = compute_e2_sensor_lz78_phrase_density(
        symbols,
        support,
        channel_name=channel_name,
        symbolization_id=symbolization_id,
    )
    if not base.computed:
        return _unsupported(
            candidate_id="E2",
            estimator_id="E2.sensor_lz78_permutation_reference_ratio.v0",
            support=support,
            reason=base.reason or "unsupported_base_sequence",
            units=base.units,
            observation_space=base.observation_space,
            details={
                **base.details,
                "selection_status": "unratified_g1_candidate_probe",
                "normalization": "not_computed_base_sequence_unsupported",
            },
        )
    if (
        isinstance(reference_replicates, bool)
        or not isinstance(reference_replicates, (int, np.integer))
        or reference_replicates < 8
    ):
        return _unsupported(
            candidate_id="E2",
            estimator_id="E2.sensor_lz78_permutation_reference_ratio.v0",
            support=support,
            reason="reference_replicates_must_be_integer_at_least_8",
            observation_space="sensor_channel",
        )
    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, (int, np.integer))
        or random_seed < 0
    ):
        return _unsupported(
            candidate_id="E2",
            estimator_id="E2.sensor_lz78_permutation_reference_ratio.v0",
            support=support,
            reason="random_seed_must_be_nonnegative_integer",
            observation_space="sensor_channel",
        )
    sequence = np.asarray(tuple(symbols), dtype=object)
    observed_count = int(base.details["phrase_count"])
    rng = np.random.default_rng(int(random_seed))
    reference_counts = np.asarray(
        [
            _lz78_phrase_count(tuple(rng.permutation(sequence).tolist()))
            for _ in range(int(reference_replicates))
        ],
        dtype=np.float64,
    )
    reference_median = float(np.median(reference_counts))
    if not np.isfinite(reference_counts).all() or reference_median <= 0:
        return _unsupported(
            candidate_id="E2",
            estimator_id="E2.sensor_lz78_permutation_reference_ratio.v0",
            support=support,
            reason="undefined_permutation_reference",
            observation_space="sensor_channel",
        )
    ratio = float(observed_count / reference_median)
    return _computed(
        candidate_id="E2",
        estimator_id="E2.sensor_lz78_permutation_reference_ratio.v0",
        value=ratio,
        support=support,
        observation_space="sensor_channel",
        details={
            **base.details,
            "normalization": "observed_phrase_count_divided_by_median_exact_multiset_permutation_phrase_count",
            "reference_generator": "uniform_permutation_preserving_exact_symbol_counts",
            "reference_replicates": int(reference_replicates),
            "reference_seed": int(random_seed),
            "reference_rng_bit_generator": type(rng.bit_generator).__name__,
            "numpy_version": np.__version__,
            "reference_phrase_count_median": reference_median,
            "reference_phrase_count_q25": float(np.quantile(reference_counts, 0.25)),
            "reference_phrase_count_q75": float(np.quantile(reference_counts, 0.75)),
            "channel_aggregation": "not_applied_single_channel",
            "selection_status": "unratified_g1_candidate_probe",
        },
    )


E2_EXACT_MEDIAN_LZ78_ESTIMATOR_ID = "E2.sensor_median_lz78_exact_multiset_ratio_83.v1"
E2_EXACT_MEDIAN_LZ78_FIXED_SET_ID = "anphy_epctl01_e1_g2_frozen_ordered_83_labels"
E2_EXACT_MEDIAN_LZ78_CHANNEL_COUNT = 83
E2_EXACT_MEDIAN_LZ78_SAMPLE_COUNT = 5000
E2_EXACT_MEDIAN_LZ78_SAMPLE_RATE_HZ = 250.0
E2_EXACT_MEDIAN_LZ78_REFERENCE_COUNT = 32
E2_EXACT_MEDIAN_LZ78_MASTER_SEED = 310930


def _e2_exact_lz78_seed(
    *,
    source_identity: str,
    support: MNPS31WindowSupport,
    channel_name: str,
    master_seed: int,
) -> int:
    """Derive a stable per-window/per-channel PCG64 seed for E2 v1."""
    seed_material = "\0".join(
        (
            "mndm.mnps.v3_1.e2_seed_derivation.v1",
            str(master_seed),
            source_identity,
            support.segment_id,
            float(support.window_start_sec).hex(),
            float(support.window_end_sec).hex(),
            channel_name,
            E2_EXACT_MEDIAN_LZ78_ESTIMATOR_ID,
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big", signed=False)


def _e2_exact_channel_result(
    row: np.ndarray,
    *,
    channel_name: str,
    source_identity: str,
    support: MNPS31WindowSupport,
    master_seed: int,
) -> dict[str, object]:
    """Compute one channel of E2 v1; private helper shared with synthetic gates."""
    threshold = float(np.median(row))
    symbols = np.asarray(row > threshold, dtype=np.uint8)
    zero_count = int(symbols.size - np.count_nonzero(symbols))
    one_count = int(np.count_nonzero(symbols))
    tie_count = int(np.count_nonzero(row == threshold))
    symbol_counts = {"0": zero_count, "1": one_count}
    if zero_count == 0 or one_count == 0:
        return {
            "computed": False,
            "valid": False,
            "reason": "symbol_sequence_has_fewer_than_two_symbols",
            "channel_name": channel_name,
            "threshold": threshold,
            "tie_count": tie_count,
            "symbol_counts": symbol_counts,
        }
    seed = _e2_exact_lz78_seed(
        source_identity=source_identity,
        support=support,
        channel_name=channel_name,
        master_seed=master_seed,
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    observed_count = _lz78_phrase_count(tuple(int(value) for value in symbols))
    reference_counts: list[int] = []
    reference_draw_sha256: list[str] = []
    source_histogram = (zero_count, one_count)
    for _ in range(E2_EXACT_MEDIAN_LZ78_REFERENCE_COUNT):
        indices = rng.permutation(E2_EXACT_MEDIAN_LZ78_SAMPLE_COUNT)
        transcript = np.asarray(indices, dtype="<u4")
        reference_draw_sha256.append(hashlib.sha256(transcript.tobytes()).hexdigest())
        permuted = symbols[indices]
        if (int(permuted.size - np.count_nonzero(permuted)), int(np.count_nonzero(permuted))) != source_histogram:
            return {
                "computed": False,
                "valid": False,
                "reason": "reference_histogram_changed",
                "channel_name": channel_name,
            }
        reference_counts.append(_lz78_phrase_count(tuple(int(value) for value in permuted)))
    reference_median = float(np.median(np.asarray(reference_counts, dtype=np.float64)))
    if not np.isfinite(reference_median) or reference_median <= 0.0:
        return {
            "computed": False,
            "valid": False,
            "reason": "undefined_exact_multiset_reference",
            "channel_name": channel_name,
        }
    ratio = float(observed_count / reference_median)
    if not np.isfinite(ratio):
        return {"computed": False, "valid": False, "reason": "non_finite_ratio", "channel_name": channel_name}
    return {
        "computed": True,
        "valid": True,
        "reason": None,
        "channel_name": channel_name,
        "threshold": threshold,
        "tie_count": tie_count,
        "symbol_counts": symbol_counts,
        "symbol_sequence_sha256": hashlib.sha256(symbols.tobytes()).hexdigest(),
        "observed_phrase_count": observed_count,
        "reference_phrase_counts": reference_counts,
        "reference_phrase_count_median": reference_median,
        "reference_phrase_count_q25": float(np.quantile(reference_counts, 0.25)),
        "reference_phrase_count_q75": float(np.quantile(reference_counts, 0.75)),
        "ratio": ratio,
        "derived_seed": seed,
        "seed_derivation_id": "mndm.mnps.v3_1.e2_seed_derivation.v1",
        "reference_rng_bit_generator": "PCG64",
        "reference_draw_sha256": reference_draw_sha256,
    }


def compute_e2_sensor_median_lz78_exact_multiset_ratio(
    signals: np.ndarray,
    channel_names: Sequence[str],
    support: MNPS31WindowSupport,
    *,
    source_identity: str,
    fixed_set_id: str,
    source_reference: str,
    master_seed: int = E2_EXACT_MEDIAN_LZ78_MASTER_SEED,
) -> EEGCandidateMeasurement:
    """Compute the selected but unqualified 83-sensor E2 technical estimator.

    Each sensor's 5,000 finite samples are binarized against their arithmetic
    median (strictly greater maps to one; ties map to zero).  The LZ78 phrase
    count is divided by the median count from 32 PCG64 permutations of that
    sensor's exact binary multiset.  The returned scalar is the median of all
    83 per-sensor ratios.  Any unsupported sensor invalidates the whole fixed
    set.  This function does not read or preprocess EEG and cannot qualify E2.
    """
    estimator_id = E2_EXACT_MEDIAN_LZ78_ESTIMATOR_ID
    support_error = _support_reason(support)
    if support_error:
        return _unsupported(candidate_id="E2", estimator_id=estimator_id, support=support, reason=support_error, observation_space="sensor_only_fixed_set")
    if support.sample_count != E2_EXACT_MEDIAN_LZ78_SAMPLE_COUNT:
        return _unsupported(candidate_id="E2", estimator_id=estimator_id, support=support, reason="sample_count_mismatch", observation_space="sensor_only_fixed_set")
    if not np.isclose(support.sample_rate_hz, E2_EXACT_MEDIAN_LZ78_SAMPLE_RATE_HZ, rtol=0.0, atol=1e-12):
        return _unsupported(candidate_id="E2", estimator_id=estimator_id, support=support, reason="sampling_rate_mismatch", observation_space="sensor_only_fixed_set")
    label_error = _validate_labels(channel_names)
    if label_error:
        return _unsupported(candidate_id="E2", estimator_id=estimator_id, support=support, reason=label_error, observation_space="sensor_only_fixed_set")
    if len(channel_names) != E2_EXACT_MEDIAN_LZ78_CHANNEL_COUNT:
        return _unsupported(candidate_id="E2", estimator_id=estimator_id, support=support, reason="channel_count_mismatch", observation_space="sensor_only_fixed_set")
    if not isinstance(source_identity, str) or not source_identity.strip():
        return _unsupported(candidate_id="E2", estimator_id=estimator_id, support=support, reason="source_identity_missing", observation_space="sensor_only_fixed_set")
    if fixed_set_id != E2_EXACT_MEDIAN_LZ78_FIXED_SET_ID:
        return _unsupported(candidate_id="E2", estimator_id=estimator_id, support=support, reason="fixed_sensor_set_id_mismatch", observation_space="sensor_only_fixed_set")
    if source_reference != "average_over_complete_fixed_83_channel_set":
        return _unsupported(candidate_id="E2", estimator_id=estimator_id, support=support, reason="source_reference_mismatch", observation_space="sensor_only_fixed_set")
    if any(not isinstance(name, str) for name in channel_names):
        return _unsupported(candidate_id="E2", estimator_id=estimator_id, support=support, reason="invalid_channel_label", observation_space="sensor_only_fixed_set")
    if isinstance(master_seed, bool) or not isinstance(master_seed, (int, np.integer)) or int(master_seed) != E2_EXACT_MEDIAN_LZ78_MASTER_SEED:
        return _unsupported(candidate_id="E2", estimator_id=estimator_id, support=support, reason="master_seed_mismatch", observation_space="sensor_only_fixed_set")
    try:
        values = np.asarray(signals, dtype=np.float64)
    except (TypeError, ValueError):
        return _unsupported(candidate_id="E2", estimator_id=estimator_id, support=support, reason="invalid_numeric_input", observation_space="sensor_only_fixed_set")
    if values.ndim != 2 or values.shape != (E2_EXACT_MEDIAN_LZ78_CHANNEL_COUNT, E2_EXACT_MEDIAN_LZ78_SAMPLE_COUNT):
        return _unsupported(candidate_id="E2", estimator_id=estimator_id, support=support, reason="invalid_fixed_sensor_matrix_shape", observation_space="sensor_only_fixed_set")
    if not np.isfinite(values).all():
        return _unsupported(candidate_id="E2", estimator_id=estimator_id, support=support, reason="non_finite_input", observation_space="sensor_only_fixed_set")

    per_channel: list[dict[str, object]] = []
    ratios: list[float] = []
    for row, channel_name in zip(values, channel_names, strict=True):
        channel = _e2_exact_channel_result(
            row,
            channel_name=channel_name,
            source_identity=source_identity,
            support=support,
            master_seed=int(master_seed),
        )
        if not channel["computed"]:
            return _unsupported(
                candidate_id="E2",
                estimator_id=estimator_id,
                support=support,
                reason=str(channel["reason"]),
                observation_space="sensor_only_fixed_set",
                details={
                    "failed_channel": channel_name,
                    "channel_measurement": channel,
                    "aggregation_failure_policy": "any_channel_unsupported_invalidates_whole_fixed_set",
                },
            )
        ratios.append(float(channel["ratio"]))
        per_channel.append(channel)

    value = float(np.median(np.asarray(ratios, dtype=np.float64)))
    return _computed(
        candidate_id="E2",
        estimator_id=estimator_id,
        value=value,
        support=support,
        units="dimensionless_ratio",
        observation_space="sensor_only_fixed_set",
        details={
            "construct": "temporal_order_compressibility_relative_to_exact_symbol_multiset_shuffles",
            "fixed_set_id": fixed_set_id,
            "source_reference": source_reference,
            "channel_count": E2_EXACT_MEDIAN_LZ78_CHANNEL_COUNT,
            "channel_order_sha256": _labels_hash(channel_names),
            "source_identity": source_identity,
            "symbolization_id": "median_gt_ties_low",
            "symbolization_threshold": "arithmetic_median_of_all_5000_samples",
            "symbolization_rule": "symbol_1_iff_sample_strictly_greater_than_threshold_else_0; exact ties map to 0",
            "lz_variant": "LZ78_incremental_dictionary_parsing",
            "reference_generator": "uniform_random_permutations_of_exact_binary_symbol_multiset",
            "reference_count_per_channel": E2_EXACT_MEDIAN_LZ78_REFERENCE_COUNT,
            "reference_summary": "median_reference_phrase_count",
            "seed_derivation_id": "mndm.mnps.v3_1.e2_seed_derivation.v1",
            "master_seed": int(master_seed),
            "reference_rng_bit_generator": "PCG64",
            "per_channel_formula": "observed_phrase_count_divided_by_median_exact_multiset_reference_phrase_count",
            "aggregate_formula": "median_of_all_83_supported_per_channel_ratios",
            "anchor": "ratio_1_equals_median_exact_multiset_shuffled_phrase_count",
            "direction": "larger_is_less_compressible_relative_to_own_shuffled_reference",
            "aggregation_failure_policy": "any_channel_unsupported_invalidates_whole_fixed_set",
            "qualified": False,
            "coordinate_assignment": None,
            "claim_status": "no_biological_claim",
            "per_channel": per_channel,
        },
    )


def compute_e3_sensor_participation_ratio(
    signals: np.ndarray,
    channel_names: Sequence[str],
    support: MNPS31WindowSupport,
) -> EEGCandidateMeasurement:
    """Compute raw-covariance participation ratio for sensor observations.

    The output is an observed sensor-space effective dimension. No `N_eff/R`
    gate, shrinkage, or cross-window spectrum stability is implied.
    """
    candidate_id, estimator_id = "E3", "E3.sensor_covariance_participation_ratio.v0"
    units = "sensor_directions"
    support_error = _support_reason(support)
    if support_error:
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason=support_error, units=units)
    label_error = _validate_labels(channel_names)
    if label_error:
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason=label_error, units=units)
    if len(channel_names) < 2 or support.sample_count <= len(channel_names):
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="insufficient_samples_for_sensor_covariance", units=units)
    values, error = _validate_matrix(signals, support, first_dimension=len(channel_names))
    if error:
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason=error, units=units)
    with np.errstate(over="ignore", invalid="ignore"):
        covariance = np.cov(values, rowvar=True, ddof=1)
    if not np.isfinite(covariance).all():
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="non_finite_covariance", units=units)
    try:
        eigenvalues = np.linalg.eigvalsh(covariance)
    except np.linalg.LinAlgError:
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="covariance_eigendecomposition_failed", units=units)
    if not np.isfinite(eigenvalues).all():
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="non_finite_covariance_spectrum", units=units)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -1e-10 * scale:
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="covariance_not_positive_semidefinite", units=units)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    trace = float(np.sum(eigenvalues))
    sum_squares = float(np.sum(eigenvalues * eigenvalues))
    if not np.isfinite(trace) or not np.isfinite(sum_squares) or trace <= 0 or sum_squares <= 0:
        return _unsupported(candidate_id=candidate_id, estimator_id=estimator_id, support=support, reason="undefined_covariance_spectrum", units=units)
    value = float(trace * trace / sum_squares)
    return _computed(
        candidate_id=candidate_id,
        estimator_id=estimator_id,
        value=value,
        support=support,
        units=units,
        details={
            "channel_order_sha256": _labels_hash(channel_names),
            "channel_count": len(channel_names),
            "covariance": "centered_sample_covariance_ddof_1",
            "shrinkage": "none",
            "normalized_by_channel_count": False,
            "effective_sample_support_gate": "not_assessed",
            "space_claim": "sensor_only",
        },
    )


__all__ = [
    "EEGCandidateMeasurement",
    "compute_m1_sensor_phase_variability",
    "compute_m3_sensor_activity_reconfiguration",
    "compute_m3_sensor_connectivity_reconfiguration",
    "compute_d1_sensor_weighted_global_efficiency",
    "compute_d2_sensor_pairclass_difference",
    "compute_d3_sensor_rdm_dispersion",
    "compute_e2_sensor_median_lz78_exact_multiset_ratio",
    "compute_e2_sensor_lz78_phrase_density",
    "compute_e3_sensor_participation_ratio",
]
