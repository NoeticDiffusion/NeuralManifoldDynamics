"""Frozen synthetic topology control for the experimental v3.2 NIE kernel.

Reproduce the data artifact with:
    python mndm/tests/test_mnps_v3_2_nie_topology_001.py --output-dir <path>
"""

from __future__ import annotations

import argparse
from itertools import combinations, permutations
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterator

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.mnps.v3_2.nie import (  # noqa: E402
    NIE_ESTIMATOR_ID,
    NIE_SCHEMA_ID,
    compute_nie_raw_weighted_global_efficiency,
)


NODE_LABELS = ("A", "B", "C", "D", "E")
NODES = len(NODE_LABELS)
EDGE_POSITIONS = tuple(combinations(range(NODES), 2))
POSITIVE_WEIGHTS = (0.2, 0.4, 0.6, 0.8)
EDGE_WEIGHT_MULTISET = (0.0,) * 6 + POSITIVE_WEIGHTS
EDGE_ESTIMATOR_ID = "synthetic.fixed_weight_multiset.v1"
EDGE_RULE_ID = "synthetic.undirected_weight_placement_no_threshold.v1"
PROTOCOL_RELATIVE_PATH = Path(
    "project/new_mnps_v.3.2/protocols/NIE-TOPOLOGY-001.md"
)
NIE_SOURCE_RELATIVE_PATH = Path("mndm/src/mndm/mnps/v3_2/nie.py")
RUNNER_RELATIVE_PATH = Path("mndm/tests/test_mnps_v3_2_nie_topology_001.py")
EXPECTED_ASSIGNMENTS = 5_040

EXEMPLARS: dict[str, tuple[tuple[str, str, float], ...]] = {
    "star": (
        ("A", "B", 0.2),
        ("A", "C", 0.4),
        ("A", "D", 0.6),
        ("A", "E", 0.8),
    ),
    "path": (
        ("A", "B", 0.2),
        ("B", "C", 0.4),
        ("C", "D", 0.6),
        ("D", "E", 0.8),
    ),
    "triangle_plus_disjoint_edge": (
        ("A", "B", 0.2),
        ("A", "C", 0.4),
        ("B", "C", 0.6),
        ("D", "E", 0.8),
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _order_hash(labels: tuple[str, ...]) -> str:
    payload = json.dumps(labels, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _matrix_from_edge_values(edge_values: tuple[float, ...]) -> np.ndarray:
    if len(edge_values) != len(EDGE_POSITIONS):
        raise ValueError("edge_values must follow all ten frozen edge positions")
    matrix = np.zeros((NODES, NODES), dtype=np.float64)
    for (left, right), weight in zip(EDGE_POSITIONS, edge_values, strict=True):
        matrix[left, right] = weight
        matrix[right, left] = weight
    return matrix


def _exemplar_matrix(edges: tuple[tuple[str, str, float], ...]) -> np.ndarray:
    weights = {pair: 0.0 for pair in EDGE_POSITIONS}
    label_to_index = {label: index for index, label in enumerate(NODE_LABELS)}
    for left, right, value in edges:
        pair = tuple(sorted((label_to_index[left], label_to_index[right])))
        weights[pair] = value
    return _matrix_from_edge_values(tuple(weights[pair] for pair in EDGE_POSITIONS))


def _upper_triangle(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(NODES, k=1)]


def _strength_summary(matrix: np.ndarray) -> dict[str, object]:
    upper = _upper_triangle(matrix)
    return {
        "node_count": int(matrix.shape[0]),
        "upper_triangle_weight_multiset": [float(value) for value in np.sort(upper)],
        "mean_upper_triangle_edge_strength_including_zeros": math.fsum(
            float(value) for value in upper
        ) / len(upper),
        "mean_positive_edge_strength": math.fsum(
            float(value) for value in upper if value > 0.0
        ) / np.count_nonzero(upper),
        "positive_edge_count": int(np.count_nonzero(upper)),
    }


def _kernel_value(matrix: np.ndarray, labels: tuple[str, ...] = NODE_LABELS) -> float:
    result = compute_nie_raw_weighted_global_efficiency(
        matrix,
        node_labels=labels,
        node_order_sha256=_order_hash(labels),
        edge_estimator_id=EDGE_ESTIMATOR_ID,
        edge_rule_id=EDGE_RULE_ID,
    )
    if not (result.computed and result.valid and result.value is not None):
        raise AssertionError(f"NIE kernel rejected frozen control: {result.to_dict()}")
    if result.qualified is not False or result.estimator_id != NIE_ESTIMATOR_ID:
        raise AssertionError("kernel identity or experimental qualification changed")
    if result.schema_id != NIE_SCHEMA_ID:
        raise AssertionError("kernel result schema changed")
    return float(result.value)


def iter_exact_edge_assignments() -> Iterator[tuple[float, ...]]:
    """Yield all 5,040 labeled assignments in the frozen stable order."""
    for positive_positions in combinations(range(len(EDGE_POSITIONS)), len(POSITIVE_WEIGHTS)):
        for assignment in permutations(POSITIVE_WEIGHTS):
            edge_values = [0.0] * len(EDGE_POSITIONS)
            for position, weight in zip(positive_positions, assignment, strict=True):
                edge_values[position] = weight
            yield tuple(edge_values)


def _midrank_percentile(values: np.ndarray, target: float) -> float:
    less = int(np.count_nonzero(values < target))
    equal = int(np.count_nonzero(values == target))
    return 100.0 * (less + 0.5 * equal) / len(values)


def run_exact_enumeration(output_dir: Path) -> dict[str, object]:
    """Compute every frozen assignment and write the reproducible run files."""
    repo_root = Path(__file__).resolve().parents[2]
    source_hashes = {
        "protocol_sha256": _sha256(repo_root / PROTOCOL_RELATIVE_PATH),
        "nie_source_sha256": _sha256(repo_root / NIE_SOURCE_RELATIVE_PATH),
        "runner_source_sha256": _sha256(repo_root / RUNNER_RELATIVE_PATH),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    null_values: list[float] = []
    seen_assignments: set[tuple[float, ...]] = set()
    with (output_dir / "exact_null_distribution.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["assignment_index", "upper_triangle_weights_lexicographic", "nie"]
        )
        for assignment_index, edge_values in enumerate(iter_exact_edge_assignments()):
            if edge_values in seen_assignments:
                raise AssertionError("duplicate assignment in exact enumerator")
            seen_assignments.add(edge_values)
            matrix = _matrix_from_edge_values(edge_values)
            stats = _strength_summary(matrix)
            if tuple(stats["upper_triangle_weight_multiset"]) != tuple(
                sorted(EDGE_WEIGHT_MULTISET)
            ):
                raise AssertionError("enumerated graph changed the frozen weight multiset")
            if stats["mean_upper_triangle_edge_strength_including_zeros"] != 0.2:
                raise AssertionError("enumerated graph changed mean edge strength")
            value = _kernel_value(matrix)
            null_values.append(value)
            writer.writerow(
                [
                    assignment_index,
                    json.dumps(edge_values, separators=(",", ":")),
                    format(value, ".17g"),
                ]
            )

    if len(seen_assignments) != EXPECTED_ASSIGNMENTS:
        raise AssertionError(f"expected {EXPECTED_ASSIGNMENTS} unique assignments")
    values = np.asarray(null_values, dtype=np.float64)
    if values.size != EXPECTED_ASSIGNMENTS or not np.isfinite(values).all():
        raise AssertionError("exact NIE enumeration is incomplete or non-finite")

    exemplar_results: dict[str, object] = {}
    exemplar_values: dict[str, float] = {}
    for name, edges in EXEMPLARS.items():
        matrix = _exemplar_matrix(edges)
        strength = _strength_summary(matrix)
        if tuple(strength["upper_triangle_weight_multiset"]) != tuple(
            sorted(EDGE_WEIGHT_MULTISET)
        ):
            raise AssertionError(f"exemplar {name} does not preserve full multiset")
        value = _kernel_value(matrix)
        exemplar_values[name] = value
        exemplar_results[name] = {
            "edges": [
                {"left": left, "right": right, "weight": weight}
                for left, right, weight in edges
            ],
            **strength,
            "nie": value,
            "exact_null_inclusive_midrank_percentile": _midrank_percentile(values, value),
        }

    null_summary: dict[str, object] = {
        "assignment_count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "population_standard_deviation_ddof_0": float(np.std(values, ddof=0)),
        "percentiles_linear": {
            str(percentile): float(np.percentile(values, percentile, method="linear"))
            for percentile in (5, 25, 50, 75, 95)
        },
    }
    summary: dict[str, object] = {
        "experiment_id": "NIE-TOPOLOGY-001",
        "run_id": "2026-10-12_frozen_multiset_n5_v1",
        "protocol_status": "frozen_before_execution",
        "estimator_id": NIE_ESTIMATOR_ID,
        "schema_id": NIE_SCHEMA_ID,
        "edge_estimator_id": EDGE_ESTIMATOR_ID,
        "edge_rule_id": EDGE_RULE_ID,
        "node_labels": list(NODE_LABELS),
        "node_count": NODES,
        "edge_positions_lexicographic": [
            [NODE_LABELS[left], NODE_LABELS[right]] for left, right in EDGE_POSITIONS
        ],
        "complete_edge_weight_multiset_including_zero": list(EDGE_WEIGHT_MULTISET),
        "mean_upper_triangle_edge_strength_including_zeros": 0.2,
        "mean_positive_edge_strength": 0.5,
        "enumeration": {
            "kind": "exact_labeled_weight_placement_enumeration",
            "random_seed": None,
            "algorithm": "choose four of ten edge slots lexicographically; assign distinct positive weights in itertools.permutations order; set remaining six weights to zero",
            "expected_unique_assignments": EXPECTED_ASSIGNMENTS,
            "observed_unique_assignments": len(seen_assignments),
            "isomorphic_graphs_collapsed": False,
        },
        "source_hashes_sha256": source_hashes,
        "null_distribution": null_summary,
        "exemplars": exemplar_results,
        "interpretation": {
            "synthetic_discrimination_only": True,
            "biological_value_established": False,
            "scientific_qualification": False,
            "claim": "Differences under a fixed complete weight multiset demonstrate sensitivity to edge placement/path structure beyond mean strength for this synthetic graph family only.",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def test_exact_enumerator_covers_5040_weight_preserving_assignments() -> None:
    assignments = list(iter_exact_edge_assignments())
    assert len(assignments) == EXPECTED_ASSIGNMENTS
    assert len(set(assignments)) == EXPECTED_ASSIGNMENTS
    expected_multiset = tuple(sorted(EDGE_WEIGHT_MULTISET))
    for assignment in assignments:
        assert tuple(sorted(assignment)) == expected_multiset
        assert math.fsum(assignment) / len(assignment) == 0.2


def test_relabeling_preserves_multiset_mean_strength_and_nie() -> None:
    permutation = (4, 2, 0, 3, 1)
    expected_multiset = tuple(sorted(EDGE_WEIGHT_MULTISET))
    for edges in EXEMPLARS.values():
        original = _exemplar_matrix(edges)
        relabeled = original[np.ix_(permutation, permutation)]
        relabeled_labels = tuple(NODE_LABELS[index] for index in permutation)
        assert original.shape == relabeled.shape == (NODES, NODES)
        assert tuple(np.sort(_upper_triangle(original))) == expected_multiset
        assert tuple(np.sort(_upper_triangle(relabeled))) == expected_multiset
        assert math.fsum(float(value) for value in _upper_triangle(original)) / 10 == 0.2
        assert math.fsum(float(value) for value in _upper_triangle(relabeled)) / 10 == 0.2
        assert abs(
            _kernel_value(original) - _kernel_value(relabeled, relabeled_labels)
        ) <= 1e-15


def test_frozen_exemplars_share_complete_weight_multiset() -> None:
    for edges in EXEMPLARS.values():
        stats = _strength_summary(_exemplar_matrix(edges))
        assert stats["node_count"] == NODES
        assert tuple(stats["upper_triangle_weight_multiset"]) == tuple(
            sorted(EDGE_WEIGHT_MULTISET)
        )
        assert stats["mean_upper_triangle_edge_strength_including_zeros"] == 0.2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = run_exact_enumeration(args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
