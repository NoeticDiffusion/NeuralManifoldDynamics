"""Matrix-level network integration efficiency for experimental MNPS v3.2.

This is the frozen legacy D1 weighted-global-efficiency formula only. The
caller must provide an already constructed coupling matrix and identify its
edge estimator and rule. This module does not construct correlations, wPLI,
thresholds, graph density, topology nulls, or temporal support.

For positive off-diagonal weight ``w_ij``, the edge length is ``1 / w_ij``;
zero weights are absent edges. The returned NIE is the mean reciprocal
shortest-path distance across ordered distinct node pairs. A computed value is
exploratory, has unknown temporal/source support, and is never qualified here.
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


NIE_ESTIMATOR_ID = "D1.sensor_weighted_global_efficiency_invweight.v0"
NIE_SCHEMA_ID = "nmdm.mnps_v3_2.nie_matrix_result.v1"
NIE_UNITS = "global_efficiency"


@dataclass(frozen=True)
class NIERawResult:
    """Result of one matrix-level NIE attempt.

    ``computed`` is true only after all declared matrix and provenance inputs
    pass validation and the shortest-path calculation is attempted. The raw
    API has no recording/time metadata, so ``support_status`` is always
    ``"unknown"`` and ``qualified`` is always false.
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
    node_count: int
    node_labels: tuple[str, ...] | None
    node_order_sha256: str | None
    edge_estimator_id: str | None
    edge_rule_id: str | None
    qc: tuple[str, ...]
    edge_length_rule: Literal["inverse_positive_weight_zero_is_absent"]
    symmetry_rule: Literal["exact"]
    diagonal_rule: Literal["exact_zero"]
    weight_bounds: tuple[float, float]
    claim_status: Literal["no_biological_claim"]

    def to_measurement_record(self) -> MeasurementRecord:
        """Return the common v3.2 result envelope without inventing support."""
        if self.node_labels is None:
            audit_status = "unknown"
        elif self.node_order_sha256 is not None:
            audit_status = "complete"
        else:
            audit_status = "incomplete"
        return MeasurementRecord(
            family_id="NIE",
            estimator_id=self.estimator_id,
            schema_id=self.schema_id,
            value=self.value,
            units=self.units,
            computed=self.computed,
            valid=self.valid,
            qualified=self.qualified,
            status=self.status,
            reason=self.reason,
            support=TimeSupportRecord(support_status=self.support_status),
            realization=RealizationRecord(
                audit_status=audit_status,
                kind="network_nodes",
                requested_id=None,
                actual_id=None,
                ordered_labels=self.node_labels,
                order_sha256=self.node_order_sha256,
                set_sha256=None,
                count=self.node_count,
            ),
            qc=self.qc,
            reproducibility=ReproducibilityRecord(status="not_supplied"),
            method_parameters=(
                ("edge_length_rule", self.edge_length_rule),
                ("symmetry_rule", self.symmetry_rule),
                ("diagonal_rule", self.diagonal_rule),
                ("weight_min", self.weight_bounds[0]),
                ("weight_max", self.weight_bounds[1]),
            ),
            claim_status=self.claim_status,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible payload; unsupported values remain null."""
        payload = asdict(self)
        payload["node_labels"] = (
            list(self.node_labels) if self.node_labels is not None else None
        )
        payload["qc"] = list(self.qc)
        payload["weight_bounds"] = list(self.weight_bounds)
        payload["measurement_record"] = self.to_measurement_record().to_dict()
        return payload


def _label_metadata(
    node_labels: Sequence[str] | None,
    node_count: int,
) -> tuple[tuple[str, ...] | None, str | None, tuple[str, ...]]:
    """Validate labels and hash their exact declared order as compact JSON."""
    if node_labels is None:
        return None, None, ("node_labels_required",)
    if isinstance(node_labels, (str, bytes, bytearray)):
        return None, None, ("node_labels_must_be_sequence",)

    try:
        raw_labels = tuple(node_labels)
    except (TypeError, ValueError):
        return None, None, ("node_labels_unreadable",)

    qc: list[str] = []
    if len(raw_labels) != node_count:
        qc.append("node_label_count_mismatch")
    if any(not isinstance(label, str) for label in raw_labels):
        qc.append("node_labels_must_be_strings")
        return None, None, tuple(qc)

    labels = tuple(raw_labels)
    if any(not label.strip() for label in labels):
        qc.append("empty_node_label")
    normalized = tuple(label.strip().casefold() for label in labels)
    if len(set(normalized)) != len(normalized):
        qc.append("duplicate_node_label")
    if qc:
        return labels, None, tuple(qc)

    # This order-sensitive canonical form matches the raw v3.2 ED label rule.
    serialized = json.dumps(labels, ensure_ascii=False, separators=(",", ":"))
    order_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return labels, order_hash, ()


def _metadata_id(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _result(
    *,
    value: float | None,
    computed: bool,
    reason: str | None,
    node_count: int,
    labels: tuple[str, ...] | None,
    order_hash: str | None,
    edge_estimator_id: str | None,
    edge_rule_id: str | None,
    qc: tuple[str, ...],
) -> NIERawResult:
    valid = value is not None
    return NIERawResult(
        estimator_id=NIE_ESTIMATOR_ID,
        schema_id=NIE_SCHEMA_ID,
        value=value,
        units=NIE_UNITS,
        computed=computed,
        valid=valid,
        qualified=False,
        status="computed_exploratory" if valid else "unsupported",
        reason=reason,
        support_status="unknown",
        node_count=node_count,
        node_labels=labels,
        node_order_sha256=order_hash,
        edge_estimator_id=edge_estimator_id,
        edge_rule_id=edge_rule_id,
        qc=qc,
        edge_length_rule="inverse_positive_weight_zero_is_absent",
        symmetry_rule="exact",
        diagonal_rule="exact_zero",
        weight_bounds=(0.0, 1.0),
        claim_status="no_biological_claim",
    )


def compute_nie_raw_weighted_global_efficiency(
    weight_matrix: np.ndarray,
    node_labels: Sequence[str] | None,
    node_order_sha256: str | None,
    edge_estimator_id: str | None,
    edge_rule_id: str | None,
) -> NIERawResult:
    """Compute inverse-weight shortest-path global efficiency from ``W``.

    Parameters
    ----------
    weight_matrix:
        Finite, real, float64-convertible square matrix. The frozen D1 contract
        requires exact symmetry, an exactly zero diagonal, and every
        off-diagonal weight in ``[0, 1]``. Zero means no edge.
    node_labels:
        Required, unique, nonempty string labels in exactly the matrix's node
        order. A scalar string is not treated as a sequence of labels.
    node_order_sha256:
        Required declared SHA-256 of compact UTF-8 JSON serialization of the
        ordered labels (``json.dumps(labels, ensure_ascii=False,
        separators=(",", ":"))``). It is checked against the labels.
    edge_estimator_id, edge_rule_id:
        Required nonempty identifiers describing how the caller constructed
        the matrix. This function does not infer or validate their definitions.

    Returns
    -------
    NIERawResult
        Invalid inputs return a null value and a machine-readable reason. A
        finite result is explicitly exploratory and has unknown temporal and
        source support because the matrix API receives no such evidence.

    Notes
    -----
    For positive ``W[i, j]``, edge length is ``1 / W[i, j]``; otherwise the
    edge is absent. Floyd-Warshall computes all-pairs shortest distances. NIE
    averages ``1 / D[i, j]`` over ordered pairs ``i != j``, assigning zero to
    disconnected pairs. No tolerance is specified by the legacy formula, so
    symmetry, diagonal, and weight-bound checks are exact rather than silently
    repairing or clipping the caller's matrix.
    """
    try:
        raw_matrix = np.asarray(weight_matrix)
    except (TypeError, ValueError, OverflowError):
        return _result(
            value=None,
            computed=False,
            reason="invalid_weight_matrix",
            node_count=0,
            labels=None,
            order_hash=None,
            edge_estimator_id=_metadata_id(edge_estimator_id),
            edge_rule_id=_metadata_id(edge_rule_id),
            qc=(),
        )

    is_complex = np.iscomplexobj(raw_matrix)
    if raw_matrix.dtype.kind == "O":
        is_complex = is_complex or any(
            isinstance(item, (complex, np.complexfloating))
            for item in raw_matrix.flat
        )
    node_count = int(raw_matrix.shape[0]) if raw_matrix.ndim >= 1 else 0
    if is_complex:
        labels, order_hash, label_qc = _label_metadata(node_labels, node_count)
        return _result(
            value=None,
            computed=False,
            reason="complex_weight_matrix_not_supported",
            node_count=node_count,
            labels=labels,
            order_hash=order_hash,
            edge_estimator_id=_metadata_id(edge_estimator_id),
            edge_rule_id=_metadata_id(edge_rule_id),
            qc=label_qc,
        )

    try:
        weights = np.asarray(raw_matrix, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return _result(
            value=None,
            computed=False,
            reason="invalid_weight_matrix",
            node_count=node_count,
            labels=None,
            order_hash=None,
            edge_estimator_id=_metadata_id(edge_estimator_id),
            edge_rule_id=_metadata_id(edge_rule_id),
            qc=(),
        )

    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        return _result(
            value=None,
            computed=False,
            reason="weight_matrix_must_be_square",
            node_count=node_count,
            labels=None,
            order_hash=None,
            edge_estimator_id=_metadata_id(edge_estimator_id),
            edge_rule_id=_metadata_id(edge_rule_id),
            qc=(),
        )

    node_count = int(weights.shape[0])
    labels, calculated_order_hash, label_qc = _label_metadata(
        node_labels, node_count
    )
    edge_estimator = _metadata_id(edge_estimator_id)
    edge_rule = _metadata_id(edge_rule_id)

    def unsupported(reason: str, extra_qc: tuple[str, ...] = ()) -> NIERawResult:
        return _result(
            value=None,
            computed=False,
            reason=reason,
            node_count=node_count,
            labels=labels,
            order_hash=calculated_order_hash,
            edge_estimator_id=edge_estimator,
            edge_rule_id=edge_rule,
            qc=label_qc + extra_qc,
        )

    if node_count < 2:
        return unsupported("insufficient_nodes")
    if labels is None or calculated_order_hash is None:
        reason = label_qc[0] if label_qc else "node_labels_required"
        return unsupported(reason)
    if node_order_sha256 is None:
        return unsupported("node_order_hash_required", ("node_order_hash_missing",))
    if not isinstance(node_order_sha256, str) or len(node_order_sha256) != 64:
        return unsupported("invalid_node_order_hash", ("node_order_hash_invalid",))
    try:
        int(node_order_sha256, 16)
    except ValueError:
        return unsupported("invalid_node_order_hash", ("node_order_hash_invalid",))
    if node_order_sha256.casefold() != calculated_order_hash:
        return unsupported(
            "node_order_hash_mismatch", ("node_order_hash_mismatch",)
        )
    if edge_estimator is None:
        return unsupported("edge_estimator_id_required", ("edge_estimator_id_missing",))
    if edge_rule is None:
        return unsupported("edge_rule_id_required", ("edge_rule_id_missing",))
    if not np.isfinite(weights).all():
        return unsupported("non_finite_weight")
    if not np.array_equal(weights, weights.T):
        return unsupported("weight_matrix_not_symmetric")
    if not np.all(np.diag(weights) == 0.0):
        return unsupported("weight_matrix_diagonal_must_be_zero")
    off_diagonal = weights[~np.eye(node_count, dtype=bool)]
    if np.any((off_diagonal < 0.0) | (off_diagonal > 1.0)):
        return unsupported("off_diagonal_weight_out_of_bounds")

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        lengths = np.full(weights.shape, np.inf, dtype=np.float64)
        positive = weights > 0.0
        lengths[positive] = 1.0 / weights[positive]
    np.fill_diagonal(lengths, 0.0)
    if np.any(positive & ~np.isfinite(lengths)):
        return unsupported("edge_length_overflow", ("positive_weight_length_unrepresentable",))

    # Keep reachability distinct from numeric infinity so overflow in a path
    # sum cannot be mistaken for a disconnected node pair.
    reachable = positive.copy()
    np.fill_diagonal(reachable, True)
    distances = lengths
    with np.errstate(over="ignore", invalid="ignore"):
        for pivot in range(node_count):
            through_pivot = distances[:, pivot, None] + distances[None, pivot, :]
            distances = np.minimum(distances, through_pivot)
            reachable |= reachable[:, pivot, None] & reachable[None, pivot, :]

    off_diagonal_mask = ~np.eye(node_count, dtype=bool)
    if np.any(reachable[off_diagonal_mask] & ~np.isfinite(distances[off_diagonal_mask])):
        return unsupported(
            "shortest_path_distance_overflow",
            ("connected_pair_distance_unrepresentable",),
        )

    inverse_distances = np.zeros_like(distances)
    finite_positive = np.isfinite(distances) & (distances > 0.0)
    inverse_distances[finite_positive] = 1.0 / distances[finite_positive]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        value = float(np.sum(inverse_distances[off_diagonal_mask]) / (node_count * (node_count - 1)))
    if not np.isfinite(value):
        return _result(
            value=None,
            computed=True,
            reason="non_finite_network_efficiency",
            node_count=node_count,
            labels=labels,
            order_hash=calculated_order_hash,
            edge_estimator_id=edge_estimator,
            edge_rule_id=edge_rule,
            qc=label_qc + ("shortest_path_numeric_failure",),
        )

    return _result(
        value=value,
        computed=True,
        reason=None,
        node_count=node_count,
        labels=labels,
        order_hash=calculated_order_hash,
        edge_estimator_id=edge_estimator,
        edge_rule_id=edge_rule,
        qc=("temporal_and_source_support_not_audited",),
    )


__all__ = [
    "NIE_ESTIMATOR_ID",
    "NIE_SCHEMA_ID",
    "NIERawResult",
    "compute_nie_raw_weighted_global_efficiency",
]
