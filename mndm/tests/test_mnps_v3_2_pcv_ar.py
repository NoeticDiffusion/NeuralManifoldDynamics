"""Focused tests for low-level experimental v3.2 PCV and AR kernels."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.mnps.v3_2.ar import compute_ar_activity_transition_vector
from mndm.mnps.v3_2.pcv import compute_pcv_phase_order_parameter_sd


def test_pcv_constant_order_parameter_has_zero_population_variability() -> None:
    phase = np.array([[0.0, 1.0, -2.0], [0.0, 1.0, -2.0]])
    result = compute_pcv_phase_order_parameter_sd(
        phase,
        phase_estimator_id="test.phase.v1",
        band_id="test.band.v1",
        channel_labels=("F3", "C3"),
    )

    assert result.computed and result.valid and not result.qualified
    assert result.value == 0.0
    assert result.to_measurement_record().support.support_status == "unknown"
    assert result.to_measurement_record().realization.audit_status == "complete"


def test_pcv_formula_is_computable_when_phase_provenance_is_unknown() -> None:
    phase = np.array([[0.0, 0.0, 1.0], [0.0, np.pi, 1.0]])
    result = compute_pcv_phase_order_parameter_sd(phase)

    assert result.valid
    assert result.value is not None
    assert "phase_estimator_provenance_missing" in result.qc
    assert "phase_band_provenance_missing" in result.qc
    assert result.to_dict()["measurement_record"]["qualified"] is False


def test_pcv_nonfinite_input_is_candidate_local_null() -> None:
    result = compute_pcv_phase_order_parameter_sd(np.array([[0.0, np.nan]]))

    assert not result.computed
    assert not result.valid
    assert result.value is None
    assert result.reason == "non_finite_phase"


def test_ar_preserves_all_ordered_adjacent_components_without_aggregation() -> None:
    vectors = np.array(
        [[1.0, 2.0, 3.0], [3.0, 6.0, 9.0], [3.0, 2.0, 1.0]],
    )
    result = compute_ar_activity_transition_vector(
        vectors,
        ((0.0, 2.0), (2.0, 4.0), (4.0, 6.0)),
        vector_construction_id="test.rms.v1",
        scaling_id="test.raw.v1",
        channel_labels=("F3", "C3", "P3"),
    )

    assert result.valid
    assert result.component_values == (0.0, 2.0)
    record = result.to_measurement_record()
    assert record.value == (0.0, 2.0)
    assert record.value_components == ("block_000_to_001", "block_001_to_002")
    assert record.method_parameters[3] == ("aggregation", "none_ordered_adjacent_pairs")


def test_ar_constant_pair_is_null_only_for_that_component() -> None:
    vectors = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    result = compute_ar_activity_transition_vector(
        vectors,
        ((0.0, 2.0), (2.0, 4.0), (4.0, 6.0)),
        vector_construction_id="test.activity.v1",
        scaling_id="test.scale.v1",
    )

    assert result.computed
    assert result.component_values[0] is None
    assert result.component_status == ("unsupported", "computed_exploratory")
    assert result.component_values[1] == pytest.approx(2.0)
    assert result.to_measurement_record().value is None
    assert result.to_dict()["component_values"][0] is None
    assert result.to_dict()["component_values"][1] == pytest.approx(2.0)


def test_ar_rejects_nonadjacent_or_unequal_blocks() -> None:
    vectors = np.array([[1.0, 2.0], [2.0, 1.0]])
    result = compute_ar_activity_transition_vector(
        vectors,
        ((0.0, 2.0), (3.0, 5.0)),
        vector_construction_id="test.vectors.v1",
        scaling_id="test.scale.v1",
    )

    assert not result.computed
    assert result.reason == "blocks_not_adjacent_or_gap_free"


def test_ar_rejects_unequal_duration_blocks() -> None:
    vectors = np.array([[1.0, 2.0], [2.0, 1.0]])
    result = compute_ar_activity_transition_vector(
        vectors,
        ((0.0, 2.0), (2.0, 5.0)),
        vector_construction_id="test.vectors.v1",
        scaling_id="test.scale.v1",
    )

    assert not result.computed
    assert result.reason == "block_durations_must_match"


def test_ar_accepts_decimal_timestamp_rounding_within_one_nanosecond() -> None:
    vectors = np.array([[1.0, 2.0], [2.0, 1.0]])
    result = compute_ar_activity_transition_vector(
        vectors,
        ((0.0, 0.1), (0.1000000005, 0.2)),
        vector_construction_id="test.vectors.v1",
        scaling_id="test.scale.v1",
    )

    assert result.valid
