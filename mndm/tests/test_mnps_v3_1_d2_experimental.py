"""Synthetic D2 label-family pair-mask and coupling-contrast controls."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.mnps.v3_1.candidates import MNPS31WindowSupport  # noqa: E402
from mndm.mnps.v3_1.d2_experimental import (  # noqa: E402
    D2_LABEL_FAMILY_ESTIMATOR_ID,
    D2_LABEL_FAMILY_PAIR_DEFINITION_ID,
    build_d2_sensor_label_family_pair_masks,
    compute_d2_sensor_label_family_from_signals,
    compute_d2_sensor_label_family_local_distributed,
)


def _support(sample_count: int = 100, sample_rate_hz: float = 100.0) -> MNPS31WindowSupport:
    return MNPS31WindowSupport(
        window_start_sec=0.0,
        window_end_sec=sample_count / sample_rate_hz,
        sample_rate_hz=sample_rate_hz,
        sample_count=sample_count,
        segment_id="synthetic-d2-segment",
        source_is_continuous=True,
        timestamps_uniform=True,
        gap_free=True,
        bad_interval_free=True,
        within_recording_bounds=True,
        effective_filter_support_valid=True,
    )


def _known_geometry_fixture() -> tuple[list[str], np.ndarray]:
    """Two synthetic label families with known close-within/far-between layout."""
    labels = ["Fp1", "Fp2", "P3", "P4"]
    # Arbitrary synthetic sensor-space coordinates, not electrode metadata.
    positions = np.asarray([[-1.00, 0.00], [-0.95, 0.00], [1.00, 0.00], [1.05, 0.00]])
    return labels, positions


def _synthetic_coupling() -> np.ndarray:
    coupling = np.full((4, 4), 0.2, dtype=np.float64)
    np.fill_diagonal(coupling, 0.0)
    coupling[0, 1] = coupling[1, 0] = 0.8
    coupling[2, 3] = coupling[3, 2] = 0.8
    return coupling


def test_fixed_prefix_masks_match_declared_synthetic_geometry_and_partition_pairs() -> None:
    labels, positions = _known_geometry_fixture()
    masks = build_d2_sensor_label_family_pair_masks(labels)

    assert masks.supported is True
    assert masks.reason is None
    assert masks.label_families == ("FP", "FP", "P", "P")
    assert masks.local_mask is not None and masks.distributed_mask is not None
    assert masks.local_pair_count == 2
    assert masks.distributed_pair_count == 4
    assert np.array_equal(masks.local_mask, masks.local_mask.T)
    assert np.array_equal(masks.distributed_mask, masks.distributed_mask.T)
    assert not np.any(masks.local_mask & masks.distributed_mask)
    assert not np.any(np.diag(masks.local_mask))
    assert not np.any(np.diag(masks.distributed_mask))

    upper = np.triu(np.ones((len(labels), len(labels)), dtype=bool), k=1)
    assert np.array_equal(masks.local_mask | masks.distributed_mask, upper | upper.T)
    local_distances = [
        float(np.linalg.norm(positions[i] - positions[j]))
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
        if masks.local_mask[i, j]
    ]
    distributed_distances = [
        float(np.linalg.norm(positions[i] - positions[j]))
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
        if masks.distributed_mask[i, j]
    ]
    assert max(local_distances) < min(distributed_distances)


def test_synthetic_known_geometry_contrast_is_local_minus_distributed_and_unqualified() -> None:
    result = compute_d2_sensor_label_family_local_distributed(
        _synthetic_coupling(),
        ["Fp1", "Fp2", "P3", "P4"],
        _support(),
        coupling_estimator_id="synthetic_fixed_matrix.v0",
    )

    assert result.computed is True
    assert result.valid is True
    assert result.qualified is False
    assert result.value == pytest.approx(0.6)
    assert result.estimator_id == D2_LABEL_FAMILY_ESTIMATOR_ID
    assert result.details["pair_definition_id"] == D2_LABEL_FAMILY_PAIR_DEFINITION_ID
    assert result.details["local_pair_count"] == 2
    assert result.details["distributed_pair_count"] == 4
    assert result.details["channel_names"] == ["Fp1", "Fp2", "P3", "P4"]
    assert result.details["channel_count"] == 4
    assert len(result.details["channel_order_sha256"]) == 64
    assert len(result.details["channel_set_sha256"]) == 64
    assert result.details["contrast_order"] == "local_minus_distributed"
    assert result.details["position_metadata_used"] is False
    assert result.details["coordinate_units"] is None
    assert result.details["coordinate_frame"] is None
    assert result.details["space_claim"] == "sensor_label_family_proxy_only"
    assert result.to_dict()["coordinate_assignment"] is None


def test_label_family_masks_and_value_are_node_relabel_invariant() -> None:
    labels = ["Fp1", "Fp2", "P3", "P4"]
    matrix = _synthetic_coupling()
    original = compute_d2_sensor_label_family_local_distributed(
        matrix,
        labels,
        _support(),
        coupling_estimator_id="synthetic_fixed_matrix.v0",
    )
    permutation = np.asarray([2, 0, 3, 1])
    relabeled = compute_d2_sensor_label_family_local_distributed(
        matrix[np.ix_(permutation, permutation)],
        [labels[index] for index in permutation],
        _support(),
        coupling_estimator_id="synthetic_fixed_matrix.v0",
    )

    assert original.value == pytest.approx(relabeled.value)
    assert original.details["node_order_sha256"] != relabeled.details["node_order_sha256"]
    assert relabeled.qualified is False


def test_pair_support_is_unsupported_when_one_label_class_is_empty() -> None:
    no_local_pairs = build_d2_sensor_label_family_pair_masks(["Fp1", "C3", "P3"])
    no_distributed_pairs = build_d2_sensor_label_family_pair_masks(["F1", "F3", "F7"])

    assert no_local_pairs.supported is False
    assert no_local_pairs.reason == "insufficient_pair_class_support"
    assert no_local_pairs.local_pair_count == 0
    assert no_local_pairs.distributed_pair_count == 3
    assert no_distributed_pairs.supported is False
    assert no_distributed_pairs.reason == "insufficient_pair_class_support"
    assert no_distributed_pairs.local_pair_count == 3
    assert no_distributed_pairs.distributed_pair_count == 0

    result = compute_d2_sensor_label_family_local_distributed(
        _synthetic_coupling()[:3, :3],
        ["Fp1", "C3", "P3"],
        _support(),
        coupling_estimator_id="synthetic_fixed_matrix.v0",
    )
    assert result.computed is False
    assert result.valid is False
    assert result.qualified is False
    assert result.value is None
    assert result.reason == "insufficient_pair_class_support"


@pytest.mark.parametrize(
    ("labels", "expected_reason"),
    [
        (["Fp1", "EOG1"], "unsupported_sensor_label_for_fixed_pair_rule"),
        (["Fp1", " fp1 "], "duplicate_channel_label"),
        (["Fp1"], "insufficient_channels_for_pair_classes"),
        (["Fp1", ""], "invalid_channel_label"),
        ([], "insufficient_channels_for_pair_classes"),
    ],
)
def test_unusable_labels_fail_closed(labels: list[str], expected_reason: str) -> None:
    masks = build_d2_sensor_label_family_pair_masks(labels)
    result = compute_d2_sensor_label_family_local_distributed(
        np.zeros((len(labels), len(labels)), dtype=np.float64),
        labels,
        _support(),
        coupling_estimator_id="synthetic_fixed_matrix.v0",
    )

    assert masks.supported is False
    assert masks.reason == expected_reason
    assert result.computed is False
    assert result.value is None
    assert result.qualified is False
    assert result.reason == expected_reason


def test_nonfinite_and_malformed_coupling_remain_unsupported() -> None:
    labels = ["Fp1", "Fp2", "P3", "P4"]
    nonfinite = _synthetic_coupling()
    nonfinite[0, 2] = nonfinite[2, 0] = np.nan
    malformed = np.zeros((3, 3), dtype=np.float64)

    nonfinite_result = compute_d2_sensor_label_family_local_distributed(
        nonfinite,
        labels,
        _support(),
        coupling_estimator_id="synthetic_fixed_matrix.v0",
    )
    malformed_result = compute_d2_sensor_label_family_local_distributed(
        malformed,
        labels,
        _support(),
        coupling_estimator_id="synthetic_fixed_matrix.v0",
    )
    missing_estimator = compute_d2_sensor_label_family_local_distributed(
        _synthetic_coupling(),
        labels,
        _support(),
        coupling_estimator_id="",
    )

    assert nonfinite_result.reason == "non_finite_input"
    assert malformed_result.reason == "invalid_pair_matrix_shape"
    assert missing_estimator.reason == "pair_or_coupling_provenance_missing"
    assert all(item.value is None and not item.qualified for item in
               (nonfinite_result, malformed_result, missing_estimator))


def test_support_failure_is_preserved_before_pair_or_coupling_evaluation() -> None:
    bad_support = replace(_support(), gap_free=False)
    result = compute_d2_sensor_label_family_local_distributed(
        _synthetic_coupling(),
        ["Fp1", "Fp2", "P3", "P4"],
        bad_support,
        coupling_estimator_id="synthetic_fixed_matrix.v0",
    )
    assert result.computed is False
    assert result.value is None
    assert result.reason == "gap_or_segment_boundary"
    assert result.details["channel_names"] == ["Fp1", "Fp2", "P3", "P4"]
    assert result.details["channel_count"] == 4
    assert len(result.details["channel_set_sha256"]) == 64
    assert result.qualified is False



def _compute_from_signals(
    sensor_signals: np.ndarray,
    channel_names: list[str],
    support: MNPS31WindowSupport,
    *,
    reference_mode: str = "recorded",
    reference_id: str = "synthetic-recorded-reference-v0",
    source_id: str = "synthetic-d2-source",
    source_content_sha256: str = "a" * 64,
    preprocessing_id: str = "synthetic-unfiltered-v0",
    signal_units: str = "arbitrary_amplitude",
):
    return compute_d2_sensor_label_family_from_signals(
        sensor_signals,
        channel_names,
        support,
        surface_id="synthetic-four-sensor-surface-v0",
        source_id=source_id,
        source_content_sha256=source_content_sha256,
        preprocessing_id=preprocessing_id,
        signal_units=signal_units,
        reference_mode=reference_mode,  # type: ignore[arg-type]
        reference_id=reference_id,
    )


def test_end_to_end_pearson_matrix_and_contrast_match_known_synthetic_signals() -> None:
    a = np.asarray([-1.0, -1.0, 1.0, 1.0])
    b = np.asarray([-1.0, 1.0, -1.0, 1.0])
    signals = np.vstack([a, a, b, b])
    result = _compute_from_signals(
        signals,
        ["Fp1", "Fp2", "P3", "P4"],
        _support(sample_count=4, sample_rate_hz=4.0),
    )

    expected_matrix = np.asarray(
        [[1.0, 1.0, 0.0, 0.0],
         [1.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 1.0],
         [0.0, 0.0, 1.0, 1.0]]
    )
    assert result.computed is True
    assert result.value == pytest.approx(1.0)
    assert result.details["coupling_estimator_id"] == "D2.sensor_signed_zero_lag_pearson_matrix.v0"
    assert result.details["local_mean_coupling"] == pytest.approx(1.0)
    assert result.details["distributed_mean_coupling"] == pytest.approx(0.0)
    assert np.allclose(result.details["coupling_matrix"], expected_matrix, atol=1e-12)
    assert result.details["channel_order_sha256"]
    assert result.details["channel_set_sha256"]
    assert result.details["sample_count"] == 4
    assert result.details["surface_id"] == "synthetic-four-sensor-surface-v0"
    assert result.details["source_content_sha256"] == "a" * 64
    assert result.details["reference_mode"] == "recorded"
    assert result.details["reference_channel_names"] is None
    assert result.details["qc"]["input_finite"] is True
    assert result.qualified is False


def test_declared_average_reference_uses_exact_supplied_surface() -> None:
    a = np.asarray([-1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    b = np.asarray([-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    c = np.asarray([0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 1.0, 1.0])
    d = np.asarray([0.0, 0.0, 0.0, 0.0, -1.0, 1.0, -1.0, 1.0])
    common = np.asarray([1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0])
    signals = np.vstack([a + common, a + 0.5 * b + common,
                         c + common, c + 0.5 * d + common])
    names = ["Fp1", "Fp2", "P3", "P4"]
    result = _compute_from_signals(
        signals,
        names,
        _support(sample_count=8, sample_rate_hz=8.0),
        reference_mode="average_over_exact_surface",
        reference_id="average-over-exact-surface-v0",
    )

    assert result.computed is True
    assert result.qualified is False
    assert result.details["reference_channel_names"] == names
    assert result.details["reference_set_sha256"] == result.details["channel_order_sha256"]
    assert result.details["qc"]["reference_applied_to_exact_surface_only"] is True
    referenced = signals - np.mean(signals, axis=0, keepdims=True)
    expected_coupling = np.corrcoef(referenced)
    observed_coupling = np.asarray(result.details["coupling_matrix"])
    assert np.allclose(observed_coupling, expected_coupling, atol=1e-12)
    assert not np.allclose(observed_coupling, np.corrcoef(signals), atol=1e-6)


def test_signal_path_fails_closed_for_constant_channels_bad_provenance_labels_and_support() -> None:
    names = ["Fp1", "Fp2", "P3", "P4"]
    support = _support(sample_count=4, sample_rate_hz=4.0)
    valid = np.asarray([
        [-1.0, -1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0, 1.0],
    ])
    constant = valid.copy()
    constant[0] = 1.0
    unsupported_labels = _compute_from_signals(valid, ["Fp1", "Fp2", "P3", "EOG1"], support)
    constant_channel = _compute_from_signals(constant, names, support)
    hash_missing = _compute_from_signals(valid, names, support, source_content_sha256="bad")
    provenance_missing = _compute_from_signals(valid, names, support, preprocessing_id=" ")
    gap = _compute_from_signals(valid, names, replace(support, gap_free=False))
    sample_mismatch = _compute_from_signals(valid[:, :-1], names, support)

    assert unsupported_labels.reason == "unsupported_sensor_label_for_fixed_pair_rule"
    assert constant_channel.reason == "constant_sensor_channel"
    assert hash_missing.reason == "source_content_hash_missing_or_invalid"
    assert provenance_missing.reason == "measurement_provenance_missing"
    assert gap.reason == "gap_or_segment_boundary"
    assert gap.details["source_content_sha256"] == "a" * 64
    assert gap.details["preprocessing_id"] == "synthetic-unfiltered-v0"
    assert gap.details["reference_id"] == "synthetic-recorded-reference-v0"
    assert sample_mismatch.reason == "sample_count_mismatch"
    assert all(not item.computed and item.value is None and not item.qualified
               for item in (unsupported_labels, constant_channel, hash_missing,
                            provenance_missing, gap, sample_mismatch))
