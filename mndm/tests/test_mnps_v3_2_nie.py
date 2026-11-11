"""Deterministic contract tests for the MNPS v3.2 raw NIE matrix kernel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.mnps.v3_2.nie import (
    NIE_ESTIMATOR_ID,
    compute_nie_raw_weighted_global_efficiency,
)


def _order_hash(labels: tuple[str, ...]) -> str:
    serialized = json.dumps(labels, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _compute(
    weights: np.ndarray,
    labels: tuple[str, ...] = ("F3", "C3", "P3"),
    **overrides: object,
):
    metadata: dict[str, object] = {
        "node_labels": labels,
        "node_order_sha256": _order_hash(labels),
        "edge_estimator_id": "test.absolute_pearson.v1",
        "edge_rule_id": "test.no_threshold_abs.v1",
    }
    metadata.update(overrides)
    return compute_nie_raw_weighted_global_efficiency(weights, **metadata)  # type: ignore[arg-type]


def test_raw_nie_known_weighted_chain_uses_inverse_weight_shortest_paths() -> None:
    # D12=1, D23=2, D13=3; average over six ordered pairs is 11/18.
    weights = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.5], [0.0, 0.5, 0.0]],
        dtype=np.float64,
    )

    result = _compute(weights)

    assert result.value == pytest.approx(11.0 / 18.0, abs=1e-15)
    assert result.estimator_id == NIE_ESTIMATOR_ID
    assert result.computed and result.valid
    assert result.qualified is False
    assert result.support_status == "unknown"
    assert result.status == "computed_exploratory"
    assert result.node_order_sha256 == _order_hash(("F3", "C3", "P3"))
    assert result.edge_estimator_id == "test.absolute_pearson.v1"
    assert result.edge_rule_id == "test.no_threshold_abs.v1"
    assert "temporal_and_source_support_not_audited" in result.qc
    assert result.to_dict()["value"] == pytest.approx(11.0 / 18.0)


def test_raw_nie_disconnected_pairs_contribute_zero() -> None:
    labels = ("F3", "C3", "P3", "O1")
    weights = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5],
            [0.0, 0.0, 0.5, 0.0],
        ]
    )

    result = _compute(weights, labels)

    # Three units of ordered-pair efficiency over N*(N-1)=12 pairs.
    assert result.value == pytest.approx(0.25, abs=1e-15)

    all_disconnected = _compute(np.zeros((3, 3), dtype=np.float64))
    assert all_disconnected.value == 0.0


def test_raw_nie_complete_unit_weight_graph_has_efficiency_one() -> None:
    weights = np.ones((3, 3), dtype=np.float64) - np.eye(3)

    result = _compute(weights)

    assert result.value == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize(
    ("weights", "labels", "reason"),
    [
        (np.zeros((1, 1)), ("F3",), "insufficient_nodes"),
        (np.zeros((2, 3)), ("F3", "C3"), "weight_matrix_must_be_square"),
        (
            np.array([[0.0, 0.5], [0.4, 0.0]]),
            ("F3", "C3"),
            "weight_matrix_not_symmetric",
        ),
        (
            np.array([[0.1, 0.5], [0.5, 0.0]]),
            ("F3", "C3"),
            "weight_matrix_diagonal_must_be_zero",
        ),
        (
            np.array([[0.0, -0.1], [-0.1, 0.0]]),
            ("F3", "C3"),
            "off_diagonal_weight_out_of_bounds",
        ),
        (
            np.array([[0.0, 1.01], [1.01, 0.0]]),
            ("F3", "C3"),
            "off_diagonal_weight_out_of_bounds",
        ),
        (
            np.array([[0.0, np.nan], [np.nan, 0.0]]),
            ("F3", "C3"),
            "non_finite_weight",
        ),
    ],
)
def test_raw_nie_rejects_unsupported_matrix_contracts(
    weights: np.ndarray, labels: tuple[str, ...], reason: str
) -> None:
    result = _compute(weights, labels)

    assert result.value is None
    assert result.computed is False
    assert result.valid is False
    assert result.qualified is False
    assert result.status == "unsupported"
    assert result.reason == reason
    assert result.to_dict()["value"] is None


def test_raw_nie_rejects_complex_weights_before_real_conversion() -> None:
    weights = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.complex128)

    result = _compute(weights, ("F3", "C3"))

    assert result.value is None
    assert result.reason == "complex_weight_matrix_not_supported"
    assert result.computed is False


@pytest.mark.parametrize(
    ("overrides", "reason", "qc"),
    [
        ({"node_labels": None}, "node_labels_required", "node_labels_required"),
        ({"node_labels": "F3"}, "node_labels_must_be_sequence", "node_labels_must_be_sequence"),
        (
            {"node_labels": ("F3", "C3")},
            "node_label_count_mismatch",
            "node_label_count_mismatch",
        ),
        (
            {"node_labels": ("F3", None, "P3")},
            "node_labels_must_be_strings",
            "node_labels_must_be_strings",
        ),
        (
            {"node_labels": ("F3", " ", "P3")},
            "empty_node_label",
            "empty_node_label",
        ),
        (
            {"node_labels": ("F3", "f3", "P3")},
            "duplicate_node_label",
            "duplicate_node_label",
        ),
        ({"node_order_sha256": None}, "node_order_hash_required", "node_order_hash_missing"),
        (
            {"node_order_sha256": "0" * 64},
            "node_order_hash_mismatch",
            "node_order_hash_mismatch",
        ),
        ({"edge_estimator_id": None}, "edge_estimator_id_required", "edge_estimator_id_missing"),
        ({"edge_rule_id": " "}, "edge_rule_id_required", "edge_rule_id_missing"),
    ],
)
def test_raw_nie_requires_valid_order_and_edge_provenance(
    overrides: dict[str, object], reason: str, qc: str
) -> None:
    weights = np.zeros((3, 3), dtype=np.float64)
    result = _compute(weights, **overrides)

    assert result.value is None
    assert result.computed is False
    assert result.reason == reason
    assert qc in result.qc


def test_raw_nie_hash_is_order_sensitive() -> None:
    weights = np.array(
        [[0.0, 0.5, 0.0], [0.5, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    result = _compute(
        weights,
        ("P3", "C3", "F3"),
        node_order_sha256=_order_hash(("F3", "C3", "P3")),
    )

    assert result.value is None
    assert result.reason == "node_order_hash_mismatch"
    assert result.node_order_sha256 == _order_hash(("P3", "C3", "F3"))
