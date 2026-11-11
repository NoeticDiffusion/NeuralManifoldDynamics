"""Deterministic contract tests for the v3.2 matrix-level OI kernel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.mnps.v3_2.oi import (
    OI_MATRIX_ESTIMATOR_ID,
    OI_MATRIX_SCHEMA_ID,
    compute_oi_matrix,
)
from mndm.features.eeg_entropy import compute_permutation_entropy


def _labels_hash(labels: tuple[str, ...]) -> str:
    payload = json.dumps(labels, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _set_hash(labels: tuple[str, ...]) -> str:
    payload = json.dumps(sorted(labels), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _frozen_mergesort_entropy(row: np.ndarray) -> float:
    from numpy.lib.stride_tricks import sliding_window_view

    patterns = np.argsort(sliding_window_view(row, 5), axis=1, kind="mergesort")
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    probabilities = counts.astype(np.float64) / float(row.size - 4)
    return -float(np.sum(probabilities * np.log(probabilities))) / np.log(120.0)


def test_oi_uses_order5_delay1_stable_ties_and_median_channel_aggregation() -> None:
    # The first row has two order patterns. Stable sorting gives
    # [2,3,0,1,4] and [1,2,4,0,3], each once. The monotone second row has one
    # repeated pattern and zero entropy. Ties are intentional in the first row.
    values = np.array(
        [
            [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        ],
        dtype=np.float64,
    )

    result = compute_oi_matrix(values, ("F3", "C3"))

    expected_first = np.log(2.0) / np.log(120.0)
    assert result.estimator_id == OI_MATRIX_ESTIMATOR_ID
    assert result.estimator_id != "mndm.mnps_v3_1.e1_channel_median_pe.v0"
    assert result.schema_id == OI_MATRIX_SCHEMA_ID
    assert result.per_channel_entropy == pytest.approx((expected_first, 0.0), abs=1e-15)
    assert result.value == pytest.approx(expected_first / 2.0, abs=1e-15)
    assert result.pattern_count_per_channel == 2
    assert result.computed and result.valid
    assert result.qualified is False
    assert result.status == "computed_exploratory"
    assert result.support_status == "unknown"


def test_oi_accepts_the_minimal_five_samples_as_one_tied_pattern() -> None:
    result = compute_oi_matrix(np.ones((1, 5), dtype=np.float64), ("F3",))

    assert result.pattern_count_per_channel == 1
    assert result.per_channel_entropy == (0.0,)
    assert result.value == 0.0
    assert result.valid


def test_oi_matches_the_v3_1_entropy_helper_on_supported_finite_rows() -> None:
    values = np.array(
        [
            [0.0, 1.0, 1.0, 0.0, 2.0, 3.0, 2.0, 1.0],
            [3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0],
        ],
        dtype=np.float64,
    )
    expected = tuple(
        compute_permutation_entropy(row, order=5, delay=1, normalize=True)
        for row in values
    )

    result = compute_oi_matrix(values, ("F3", "C3"))

    assert result.per_channel_entropy == pytest.approx(expected, abs=1e-15)
    assert result.value == pytest.approx(float(np.median(expected)), abs=1e-15)


def test_oi_rank_histogram_is_exactly_equal_to_frozen_mergesort_counts() -> None:
    rng = np.random.default_rng(270216)
    values = np.round(rng.normal(size=(4, 1200)), decimals=1)
    values[0] = np.repeat(np.array([-1.0, 0.0, 1.0, 0.0]), 300)
    values[1] = np.arange(values.shape[1], dtype=np.float64) % 7

    expected_entropies = tuple(_frozen_mergesort_entropy(row) for row in values)
    result = compute_oi_matrix(values, ("F3", "C3", "P3", "O1"))

    assert result.per_channel_entropy == expected_entropies
    assert result.value == float(np.median(expected_entropies))


def test_oi_preserves_row_order_hash_and_set_hash_separately() -> None:
    values = np.array(
        [
            [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        ]
    )
    labels = ("F3", "C3")
    forward = compute_oi_matrix(values, labels)
    reverse = compute_oi_matrix(values[::-1], labels[::-1])

    assert forward.value == pytest.approx(reverse.value, abs=0.0)
    assert forward.per_channel_entropy == pytest.approx(tuple(reversed(reverse.per_channel_entropy)))
    assert forward.channel_labels == labels
    assert reverse.channel_labels == tuple(reversed(labels))
    assert forward.channel_order_sha256 == _labels_hash(labels)
    assert reverse.channel_order_sha256 == _labels_hash(tuple(reversed(labels)))
    assert forward.channel_order_sha256 != reverse.channel_order_sha256
    assert forward.channel_set_sha256 == reverse.channel_set_sha256 == _set_hash(labels)


def test_oi_common_envelope_keeps_unknown_support_and_unresolved_realization() -> None:
    labels = ("F3", "C3")
    result = compute_oi_matrix(
        np.array(
            [
                [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            ]
        ),
        labels,
    )

    record = result.to_measurement_record()
    payload = result.to_dict()
    assert record.contract_id == "nmdm.mnps_v3_2.measurement_record.v1"
    assert record.family_id == "OI"
    assert record.estimator_id == OI_MATRIX_ESTIMATOR_ID
    assert record.value == result.value
    assert record.support.support_status == "unknown"
    assert record.support.sample_count == 6
    assert record.realization.audit_status == "incomplete"
    assert record.realization.kind == "matrix_input_rows_unresolved_sensor_realization"
    assert record.realization.actual_id is None
    assert record.realization.reference_id is None
    assert record.realization.ordered_labels == labels
    assert record.reproducibility.status == "not_supplied"
    assert record.qualified is False
    assert "preprocessing_not_audited" in record.qc
    assert "temporal_and_source_support_not_audited" in record.qc
    assert payload["measurement_record"]["value"] == pytest.approx(result.value)
    assert payload["measurement_record"]["qualified"] is False
    params = dict(record.method_parameters or ())
    assert params["tie_handling"] == "stable_mergesort_original_sample_order"
    assert params["channel_aggregation"] == "median_over_all_supplied_matrix_rows"


@pytest.mark.parametrize(
    ("signals", "reason"),
    [
        (np.ones((2, 4), dtype=np.float64), "insufficient_samples_for_one_ordinal_pattern"),
        (
            np.array([[0.0, 1.0, np.nan, 3.0, 4.0]], dtype=np.float64),
            "non_finite_sample",
        ),
        (
            np.array([[0.0 + 1.0j, 1.0, 2.0, 3.0, 4.0]], dtype=np.complex128),
            "complex_signal_not_supported",
        ),
        (np.zeros((0, 5), dtype=np.float64), "no_channels"),
    ],
)
def test_oi_rejects_unsupported_signal_inputs_without_substituting_values(
    signals: np.ndarray, reason: str
) -> None:
    result = compute_oi_matrix(signals)
    record = result.to_measurement_record()

    assert result.value is None
    assert result.computed is False
    assert result.valid is False
    assert result.qualified is False
    assert result.status == "unsupported"
    assert result.reason == reason
    assert result.to_dict()["value"] is None
    assert record.value is None
    assert record.support.support_status == "unknown"


def test_bad_optional_labels_remain_qc_without_suppressing_computable_oi() -> None:
    result = compute_oi_matrix(np.arange(10, dtype=np.float64).reshape(2, 5), ("F3",))

    assert result.value == pytest.approx(0.0, abs=0.0)
    assert result.valid
    assert result.channel_labels == ("F3",)
    assert result.channel_order_sha256 is None
    assert "channel_label_count_mismatch" in result.qc
    assert result.to_measurement_record().realization.audit_status == "incomplete"

