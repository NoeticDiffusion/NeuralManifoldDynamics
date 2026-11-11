"""Synthetic contract checks for the first MNPS v3.1 candidate estimator."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.mnps.v3_1 import (
    E1_CANDIDATE_ID,
    E1_METHOD_VERSION,
    MNPS31E1WindowSupport,
    MNPS31_CANDIDATES,
    compute_e1_channel_median_permutation_entropy,
)


def _fixture(seed: int = 5):
    labels = tuple(f"EEG{i:02d}" for i in range(83))
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=(len(labels), 5000))
    support = MNPS31E1WindowSupport(
        window_start_sec=40.0,
        window_end_sec=60.0,
        sample_rate_hz=250.0,
        sample_count=5000,
        segment_id="synthetic-run-segment-1",
        source_is_continuous=True,
        timestamps_uniform=True,
        gap_free=True,
        bad_interval_free=True,
        within_recording_bounds=True,
        effective_filter_support_valid=True,
    )
    return labels, signal, support


def test_e1_computes_fixed_order_channel_median_and_never_qualifies() -> None:
    labels, signal, support = _fixture()

    result = compute_e1_channel_median_permutation_entropy(signal, labels, labels, support)

    assert result.candidate_id == E1_CANDIDATE_ID
    assert result.method_version == E1_METHOD_VERSION
    assert result.status == "computed_exploratory"
    assert result.computed is True
    assert result.valid is True
    assert result.qualified is False
    assert result.reason is None
    assert result.value is not None and 0.0 <= result.value <= 1.0
    assert result.to_dict()["claim_status"] == "no_biological_claim"


def test_e1_is_invariant_to_input_channel_order_when_labels_are_preserved() -> None:
    labels, signal, support = _fixture()
    forward = compute_e1_channel_median_permutation_entropy(signal, labels, labels, support)
    reverse = compute_e1_channel_median_permutation_entropy(
        signal[::-1], labels[::-1], labels, support
    )

    assert forward.value == reverse.value
    assert forward.channel_order_sha256 == reverse.channel_order_sha256


def test_e1_fails_closed_on_unverified_or_invalid_support() -> None:
    labels, signal, support = _fixture()
    missing_audit = MNPS31E1WindowSupport(
        window_start_sec=support.window_start_sec,
        window_end_sec=support.window_end_sec,
        sample_rate_hz=support.sample_rate_hz,
        sample_count=support.sample_count,
        segment_id=support.segment_id,
    )
    gap = MNPS31E1WindowSupport(**{**support.__dict__, "gap_free": False})
    malformed_audit = MNPS31E1WindowSupport(**{**support.__dict__, "gap_free": "false"})
    nonfinite = signal.copy()
    nonfinite[0, 100] = np.nan

    assert (
        compute_e1_channel_median_permutation_entropy(signal, labels, labels, missing_audit).reason
        == "support_audit_incomplete"
    )
    assert (
        compute_e1_channel_median_permutation_entropy(signal, labels, labels, gap).reason
        == "gap_or_segment_boundary"
    )
    assert (
        compute_e1_channel_median_permutation_entropy(signal, labels, labels, malformed_audit).reason
        == "support_audit_flag_not_boolean"
    )
    assert (
        compute_e1_channel_median_permutation_entropy(nonfinite, labels, labels, support).reason
        == "non_finite_sample"
    )


def test_e1_rejects_missing_channels_constant_channels_and_wrong_sample_count() -> None:
    labels, signal, support = _fixture()
    missing = compute_e1_channel_median_permutation_entropy(
        signal[:-1], labels[:-1], labels, support
    )
    constant_signal = signal.copy()
    constant_signal[7] = 2.0
    constant = compute_e1_channel_median_permutation_entropy(
        constant_signal, labels, labels, support
    )
    short_support = MNPS31E1WindowSupport(**{**support.__dict__, "sample_count": 4999})
    short_signal = compute_e1_channel_median_permutation_entropy(
        signal[:, :-1], labels, labels, short_support
    )

    assert missing.reason == "fixed_channel_set_mismatch"
    assert constant.reason == "zero_variance_channel"
    assert short_signal.reason == "unsupported_window_sample_count"


def test_candidate_registry_marks_prototypes_without_qualifying_any_family() -> None:
    assert len(MNPS31_CANDIDATES) == 9
    assert all(candidate.coordinate_assignment is None for candidate in MNPS31_CANDIDATES)
    assert all(candidate.qualified is False for candidate in MNPS31_CANDIDATES)
    estimator_ids = [
        estimator_id
        for candidate in MNPS31_CANDIDATES
        for estimator_id in candidate.estimator_ids
    ]
    assert len(estimator_ids) == len(set(estimator_ids))
    assert {candidate.candidate_id for candidate in MNPS31_CANDIDATES} == {
        "M1", "M2", "M3", "D1", "D2", "D3", "E1", "E2", "E3"
    }
    m2 = next(item for item in MNPS31_CANDIDATES if item.candidate_id == "M2")
    assert m2.implementation_status == "prototype_available"
    assert m2.qualified is False
    assert m2.estimator_ids[:2] == (
        "M2.sensor_lag1_decorrelation.v0",
        "M2.surface_channel_median_lag1_decorrelation.v0",
    )
    assert (m2.construct_name, m2.short_name, m2.public_id) == (
        "Temporal Persistence", "TP", "temporal_persistence"
    )
    assert m2.standard_availability == "standard_when_supported"
    e2 = next(item for item in MNPS31_CANDIDATES if item.candidate_id == "E2")
    assert e2.implementation_status == "core_probe_available"
    assert e2.estimator_ids[:3] == (
        "E2.sensor_median_lz78_exact_multiset_ratio_83.v1",
        "E2.sensor_lz78_phrase_density_unadjusted.v0",
        "E2.sensor_lz78_permutation_reference_ratio.v0",
    )
    d2 = next(item for item in MNPS31_CANDIDATES if item.candidate_id == "D2")
    assert (d2.construct_name, d2.short_name, d2.public_id) == (
        "Spatial Coupling Gradient", "SCG", "spatial_coupling_gradient"
    )
    assert d2.historical_aliases == ("D2", "LDCB")
    assert "spatial_dwpli_distance_spearman_alpha.v1" in d2.estimator_ids
    assert d2.standard_availability == "experimental_only"
