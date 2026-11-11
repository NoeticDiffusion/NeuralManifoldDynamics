"""Known-truth and failure-path tests for the MNPS v3.2 raw ED kernel."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.mnps.v3_1 import MNPS31WindowSupport, compute_e3_sensor_participation_ratio
from mndm.mnps.v3_2.ed import (
    ED_RAW_ESTIMATOR_ID,
    compute_ed_raw_sensor_covariance_participation_ratio,
)


def _rank_k_signals(*, channels: int, rank: int, samples: int) -> np.ndarray:
    rng = np.random.default_rng(20261011)
    temporal_basis = rng.standard_normal((samples, rank))
    temporal_basis -= temporal_basis.mean(axis=0, keepdims=True)
    orthonormal, _ = np.linalg.qr(temporal_basis)
    active = np.sqrt(samples - 1.0) * orthonormal[:, :rank].T
    return np.vstack((active, np.zeros((channels - rank, samples))))


def _signals_with_covariance_eigenvalues(
    eigenvalues: tuple[float, ...], *, samples: int
) -> np.ndarray:
    rng = np.random.default_rng(20261012)
    temporal_basis = rng.standard_normal((samples, len(eigenvalues)))
    temporal_basis -= temporal_basis.mean(axis=0, keepdims=True)
    orthonormal, _ = np.linalg.qr(temporal_basis)
    return (
        np.sqrt(samples - 1.0)
        * np.sqrt(np.asarray(eigenvalues, dtype=np.float64))[:, None]
        * orthonormal[:, : len(eigenvalues)].T
    )


def test_raw_ed_recovers_rank_one_and_equal_eigenvalue_rank() -> None:
    rank_one_base = np.sin(2.0 * np.pi * np.arange(32) / 32.0)
    rank_one = np.vstack((rank_one_base, 2.0 * rank_one_base, -rank_one_base))
    rank_three = _rank_k_signals(channels=5, rank=3, samples=40)

    one = compute_ed_raw_sensor_covariance_participation_ratio(
        rank_one, ["F3", "C3", "P3"]
    )
    three = compute_ed_raw_sensor_covariance_participation_ratio(
        rank_three, ["F3", "C3", "P3", "O1", "O2"]
    )

    assert one.value == pytest.approx(1.0, abs=1e-12)
    assert three.value == pytest.approx(3.0, abs=1e-12)
    assert one.estimator_id == ED_RAW_ESTIMATOR_ID
    assert one.valid and one.computed
    assert one.qualified is False
    assert one.support_status == "unknown"
    assert one.status == "computed_exploratory"


def test_raw_ed_preserves_unknown_audit_and_null_serialization() -> None:
    values = _rank_k_signals(channels=4, rank=2, samples=20)
    result = compute_ed_raw_sensor_covariance_participation_ratio(values)

    assert result.value == pytest.approx(2.0, abs=1e-12)
    assert "temporal_and_source_support_not_audited" in result.qc
    payload = result.to_dict()
    assert payload["value"] == pytest.approx(2.0, abs=1e-12)
    assert payload["channel_labels"] is None
    assert payload["channel_order_sha256"] is None
    assert payload["qualified"] is False

    unsupported = compute_ed_raw_sensor_covariance_participation_ratio(
        np.zeros((4, 12)), ["F3", "C3", "P3", "O1"]
    )
    assert unsupported.value is None
    assert unsupported.to_dict()["value"] is None
    assert unsupported.reason == "undefined_covariance_spectrum"
    assert unsupported.valid is False


def test_raw_ed_rejects_complex_inputs_before_real_conversion() -> None:
    real = _rank_k_signals(channels=4, rank=2, samples=20)
    complex_array = real.astype(np.complex128) + 1j * np.ones_like(real)
    object_array = np.asarray(complex_array, dtype=object)

    for values in (complex_array, object_array):
        result = compute_ed_raw_sensor_covariance_participation_ratio(values)
        assert result.value is None
        assert result.reason == "complex_signal_not_supported"
        assert result.computed is False


@pytest.mark.parametrize(
    ("values", "reason"),
    [
        (np.ones((1, 8)), "insufficient_channels_for_sensor_covariance"),
        (np.ones((4, 4)), "insufficient_samples_for_sensor_covariance"),
        (np.full((3, 8), np.nan), "non_finite_sample"),
        (np.ones((3, 8)), "undefined_covariance_spectrum"),
    ],
)
def test_raw_ed_returns_typed_unsupported_for_declared_input_failures(
    values: np.ndarray, reason: str
) -> None:
    result = compute_ed_raw_sensor_covariance_participation_ratio(values)

    assert result.value is None
    assert result.valid is False
    assert result.reason == reason


def test_label_qc_does_not_suppress_computable_ed_and_hashes_are_distinct() -> None:
    values = _rank_k_signals(channels=4, rank=2, samples=20)
    labels = ("F3", "C3", "P3", "O1")
    ordered = compute_ed_raw_sensor_covariance_participation_ratio(values, labels)
    permuted = compute_ed_raw_sensor_covariance_participation_ratio(
        values[[2, 0, 3, 1]], (labels[2], labels[0], labels[3], labels[1])
    )
    incomplete = compute_ed_raw_sensor_covariance_participation_ratio(
        values, ("F3", "C3")
    )

    assert ordered.value == pytest.approx(permuted.value, abs=1e-12)
    assert ordered.channel_order_sha256 != permuted.channel_order_sha256
    assert ordered.channel_set_sha256 == permuted.channel_set_sha256
    assert incomplete.value == pytest.approx(ordered.value, abs=1e-12)
    assert incomplete.channel_order_sha256 is None
    assert "sensor_label_count_mismatch" in incomplete.qc


@pytest.mark.parametrize(
    ("labels", "expected_qc"),
    [
        (("F3", None, "P3", "O1"), "sensor_labels_must_be_strings"),
        (("F3", 2, "P3", "O1"), "sensor_labels_must_be_strings"),
        ("F3", "sensor_labels_must_be_sequence"),
    ],
)
def test_invalid_label_types_do_not_create_hashes_or_suppress_value(
    labels: object, expected_qc: str
) -> None:
    values = _rank_k_signals(channels=4, rank=2, samples=20)
    unlabeled = compute_ed_raw_sensor_covariance_participation_ratio(values)
    result = compute_ed_raw_sensor_covariance_participation_ratio(values, labels)  # type: ignore[arg-type]

    assert result.value == pytest.approx(unlabeled.value, abs=1e-12)
    assert result.channel_labels is None
    assert result.channel_order_sha256 is None
    assert result.channel_set_sha256 is None
    assert expected_qc in result.qc


def test_raw_ed_matches_accepted_v31_e3_numerics() -> None:
    rng = np.random.default_rng(49)
    values = rng.standard_normal((4, 100))
    support = MNPS31WindowSupport(
        window_start_sec=0.0,
        window_end_sec=1.0,
        sample_rate_hz=100.0,
        sample_count=100,
        segment_id="synthetic-parity",
        source_is_continuous=True,
        timestamps_uniform=True,
        gap_free=True,
        bad_interval_free=True,
        within_recording_bounds=True,
        effective_filter_support_valid=True,
    )
    labels = ("F3", "C3", "P3", "O1")

    old = compute_e3_sensor_participation_ratio(values, labels, support)
    current = compute_ed_raw_sensor_covariance_participation_ratio(values, labels)

    assert old.value is not None
    assert current.value == pytest.approx(old.value, rel=1e-13, abs=1e-13)
    assert old.estimator_id == "E3.sensor_covariance_participation_ratio.v0"
    assert current.estimator_id != old.estimator_id


def test_raw_ed_reports_colored_finite_samples_without_independence_correction() -> None:
    rng = np.random.default_rng(91)
    innovations = rng.standard_normal((4, 512))
    colored = np.zeros_like(innovations)
    for index in range(1, innovations.shape[1]):
        colored[:, index] = 0.95 * colored[:, index - 1] + innovations[:, index]

    result = compute_ed_raw_sensor_covariance_participation_ratio(colored)

    assert result.value is not None
    assert 1.0 <= result.value <= result.channel_count
    assert result.effective_sample_support_gate == "not_assessed"
    assert result.support_status == "unknown"


def test_near_singular_ed_is_stable_across_uniform_signal_scales() -> None:
    near_singular = _signals_with_covariance_eigenvalues(
        (1.0, 0.5, 0.25, 1e-10), samples=64
    )
    results = [
        compute_ed_raw_sensor_covariance_participation_ratio(
            near_singular * scale, ("F3", "C3", "P3", "O1")
        )
        for scale in (1e-8, 1.0, 1e8)
    ]

    assert all(result.value is not None for result in results)
    assert results[0].value == pytest.approx(results[1].value, rel=1e-8)
    assert results[2].value == pytest.approx(results[1].value, rel=1e-8)
