from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.mnps.v3_1 import (
    M2InputBranch,
    MNPS31WindowSupport,
    compute_m2_surface_lag1_decorrelation,
)
from mndm.mnps.v3_1.m2_acf_mobility import (
    M2_LAG1_DECORRELATION_ESTIMATOR_ID,
    compute_m2_sensor_lag1_decorrelation,
)


def _support(
    n: int,
    sample_rate_hz: float = 250.0,
    **overrides: object,
) -> MNPS31WindowSupport:
    fields: dict[str, object] = {
        "window_start_sec": 10.0,
        "window_end_sec": 10.0 + n / sample_rate_hz,
        "sample_rate_hz": sample_rate_hz,
        "sample_count": n,
        "segment_id": "synthetic:m2-test",
        "source_is_continuous": True,
        "timestamps_uniform": True,
        "gap_free": True,
        "bad_interval_free": True,
        "within_recording_bounds": True,
        "effective_filter_support_valid": True,
    }
    fields.update(overrides)
    return MNPS31WindowSupport(**fields)  # type: ignore[arg-type]


def _ar1(n: int, sample_rate_hz: float, tau_seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    phi = np.exp(-1.0 / (sample_rate_hz * tau_seconds))
    innovation_scale = np.sqrt(1.0 - phi * phi)
    innovations = rng.normal(size=n)
    values = np.empty(n, dtype=np.float64)
    values[0] = rng.normal()
    for index in range(1, n):
        values[index] = phi * values[index - 1] + innovation_scale * innovations[index]
    return values


def test_exact_lag1_definition_units_and_unqualified_status() -> None:
    # Centered x = [-1.5, -0.5, 0.5, 1.5]; rho(1) = 1.25 / 5 = .25.
    result = compute_m2_sensor_lag1_decorrelation(
        np.array([1.0, 2.0, 3.0, 4.0]), _support(4)
    )

    assert result.estimator_id == M2_LAG1_DECORRELATION_ESTIMATOR_ID
    assert result.candidate_id == "M2"
    assert result.value == 0.75
    assert result.lag1_autocorrelation == 0.25
    assert result.units == "1"
    assert result.lag_seconds == 1.0 / 250.0
    assert result.computed is True
    assert result.valid is True
    assert result.qualified is False
    assert result.status == "computed_exploratory"
    assert result.to_dict()["claim_status"] == "no_biological_claim"


def test_ar1_population_scaling_with_physical_timescale_and_sampling_rate() -> None:
    tau_seconds = 0.2
    for sample_rate_hz in (100.0, 250.0):
        values = _ar1(100_000, sample_rate_hz, tau_seconds, seed=813)
        result = compute_m2_sensor_lag1_decorrelation(
            values, _support(values.size, sample_rate_hz)
        )
        expected_rho1 = np.exp(-1.0 / (sample_rate_hz * tau_seconds))

        assert result.value is not None
        assert result.lag1_autocorrelation == pytest.approx(expected_rho1, abs=0.015)
        assert result.value == pytest.approx(1.0 - expected_rho1, abs=0.015)


def test_value_is_invariant_to_positive_negative_and_extreme_amplitude_scaling() -> None:
    values = np.array([0.2, -0.3, 1.4, 0.1, -0.8, 0.7, 1.0])
    expected = compute_m2_sensor_lag1_decorrelation(values, _support(values.size))

    for scale in (7.5, -3.0, 1e-250, 1e250):
        actual = compute_m2_sensor_lag1_decorrelation(values * scale, _support(values.size))
        assert actual.value == pytest.approx(expected.value, abs=1e-14)
        assert actual.lag1_autocorrelation == pytest.approx(
            expected.lag1_autocorrelation, abs=1e-14
        )


def test_periodic_signal_can_have_high_mobility_without_being_random() -> None:
    sample_rate_hz = 250.0
    values = np.tile(np.array([1.0, -1.0]), 500)
    result = compute_m2_sensor_lag1_decorrelation(
        values, _support(values.size, sample_rate_hz)
    )

    assert result.value == pytest.approx(1.999)
    assert result.lag1_autocorrelation == pytest.approx(-0.999)


def test_support_and_degenerate_failures_return_no_numeric_substitute() -> None:
    values = np.random.default_rng(17).normal(size=32)
    cases = (
        (values, _support(values.size, gap_free=False), "gap_or_segment_boundary"),
        (values, _support(values.size, effective_filter_support_valid=None), "support_audit_incomplete"),
        (values, _support(values.size, bad_interval_free=False), "bad_interval_overlap"),
        (values, _support(values.size, sample_count=31), "sample_count_mismatch"),
        (values, _support(values.size, window_end_sec=11.0), "window_time_support_mismatch"),
        (np.ones(32), _support(32), "zero_variance_signal"),
        (np.array([1.0, np.nan, 2.0]), _support(3), "non_finite_sample"),
        (values[:2], _support(2), "at_least_three_samples_required"),
    )

    for signal, support, reason in cases:
        result = compute_m2_sensor_lag1_decorrelation(signal, support)
        assert result.value is None
        assert result.lag1_autocorrelation is None
        assert result.computed is False
        assert result.valid is False
        assert result.qualified is False
        assert result.reason == reason


def test_surface_median_preserves_realization_and_channel_coverage() -> None:
    rng = np.random.default_rng(991)
    signals = np.vstack(
        (
            rng.normal(size=64),
            np.ones(64),
            np.linspace(-1.0, 1.0, 64),
            np.sin(np.arange(64) * np.pi / 2.0),
        )
    )
    labels = ("EEG-A", "EEG-B", "EEG-C", "EEG-D")
    support = _support(64, 250.0)
    branch = M2InputBranch(
        branch_id="synthetic_unfiltered_native_reference_v0",
        sample_rate_hz=250.0,
        filter_band_hz=None,
        reference="native_reference_preserved",
        reference_set_sha256=None,
    )

    result = compute_m2_surface_lag1_decorrelation(
        signals, labels, support, branch
    )

    assert result.estimator_id == "M2.surface_channel_median_lag1_decorrelation.v0"
    assert result.computed is True
    assert result.valid is True
    assert result.qualified is False
    assert result.channel_labels == labels
    assert result.channel_count == 4
    assert result.valid_channel_count == 3
    assert result.valid_channel_coverage == 0.75
    assert result.per_channel[1].reason == "zero_variance_signal"
    assert result.qc["minimum_coverage_threshold"] is None
    serialized = result.to_dict()
    assert serialized["input_branch"]["reference"] == "native_reference_preserved"
    assert len(serialized["channel_order_sha256"]) == 64
    assert len(serialized["channel_set_sha256"]) == 64


def test_surface_input_branch_rate_mismatch_is_unsupported() -> None:
    values = np.random.default_rng(2).normal(size=(2, 32))
    branch = M2InputBranch(
        branch_id="synthetic",
        sample_rate_hz=200.0,
        filter_band_hz=(1.0, 45.0),
        reference="synthetic_reference",
        reference_set_sha256="a" * 64,
    )

    result = compute_m2_surface_lag1_decorrelation(
        values, ("A", "B"), _support(32, 250.0), branch
    )

    assert result.value is None
    assert result.computed is False
    assert result.reason == "input_branch_sample_rate_mismatch"
