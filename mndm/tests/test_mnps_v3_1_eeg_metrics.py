"""Synthetic semantics and failure controls for v3.1 EEG sensor prototypes."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.mnps.v3_1 import (
    MNPS31WindowSupport,
    compute_d1_sensor_weighted_global_efficiency,
    compute_d2_sensor_pairclass_difference,
    compute_d3_sensor_rdm_dispersion,
    compute_e2_sensor_lz78_phrase_density,
    compute_e2_sensor_lz78_permutation_reference_ratio,
    compute_e3_sensor_participation_ratio,
    compute_m1_sensor_phase_variability,
    compute_m3_sensor_activity_reconfiguration,
    compute_m3_sensor_connectivity_reconfiguration,
)


def _support(sample_count: int = 800, sample_rate_hz: float = 100.0) -> MNPS31WindowSupport:
    return MNPS31WindowSupport(
        window_start_sec=0.0,
        window_end_sec=sample_count / sample_rate_hz,
        sample_rate_hz=sample_rate_hz,
        sample_count=sample_count,
        segment_id="synthetic-eeg-segment-1",
        source_is_continuous=True,
        timestamps_uniform=True,
        gap_free=True,
        bad_interval_free=True,
        within_recording_bounds=True,
        effective_filter_support_valid=True,
    )


def _assert_exploratory(result) -> None:
    assert result.computed is True
    assert result.valid is True
    assert result.qualified is False
    assert result.status == "computed_exploratory"
    assert result.to_dict()["claim_status"] == "no_biological_claim"
    assert result.to_dict()["coordinate_assignment"] is None


def test_m1_sensor_phase_order_parameter_detects_switching_and_rejects_unidentified_phase() -> None:
    support = _support()
    channels = [f"EEG{i:02d}" for i in range(4)]
    phases = np.zeros((4, support.sample_count), dtype=float)
    phases[:, support.sample_count // 2 :] = np.asarray([0.0, np.pi / 2, np.pi, -np.pi / 2])[:, None]

    switching = compute_m1_sensor_phase_variability(
        phases,
        channels,
        support,
        phase_estimator_id="hilbert-v0",
        band_id="alpha-example",
    )
    malformed_support = MNPS31WindowSupport(**{**support.__dict__, "gap_free": "false"})
    malformed_audit = compute_m1_sensor_phase_variability(
        phases,
        channels,
        malformed_support,
        phase_estimator_id="hilbert-v0",
        band_id="alpha-example",
    )
    missing_method = compute_m1_sensor_phase_variability(
        phases,
        channels,
        support,
        phase_estimator_id="",
        band_id="alpha-example",
    )

    _assert_exploratory(switching)
    assert switching.value is not None and switching.value > 0.4
    assert switching.details["space_claim"] == "sensor_only"
    assert malformed_audit.reason == "support_audit_flag_not_boolean"
    assert missing_method.reason == "phase_or_band_provenance_missing"


def test_m1_phase_order_parameter_is_invariant_to_common_phase_rotation() -> None:
    support = _support()
    channel_names = ["F3", "C3", "P3", "O1"]
    offsets = np.asarray([0.0, 0.4, 1.1, 2.0])[:, None]
    locked = np.broadcast_to(offsets, (4, support.sample_count)).copy()
    common_rotation = np.linspace(-np.pi, np.pi, support.sample_count)
    rotating = np.angle(np.exp(1j * (offsets + common_rotation[None, :])))

    locked_result = compute_m1_sensor_phase_variability(
        locked,
        channel_names,
        support,
        phase_estimator_id="synthetic-phase-v0",
        band_id="synthetic-band-v0",
    )
    rotating_result = compute_m1_sensor_phase_variability(
        rotating,
        channel_names,
        support,
        phase_estimator_id="synthetic-phase-v0",
        band_id="synthetic-band-v0",
    )

    _assert_exploratory(locked_result)
    _assert_exploratory(rotating_result)
    assert locked_result.value is not None and abs(locked_result.value) < 1e-12
    assert rotating_result.value is not None and rotating_result.value < 1e-12


def test_m3_keeps_activity_and_connectivity_variants_separate_and_returns_pairwise_values() -> None:
    support = _support(sample_count=800, sample_rate_hz=100.0)
    channel_names = ["F3", "C3", "P3"]
    intervals = np.asarray([[0, 2], [2, 4], [4, 6], [6, 8]], dtype=float)
    activity = np.asarray([[1, 2, 4], [1.1, 2.1, 4.1], [4, 2, 1], [4.1, 2.1, 1.1]])
    connectivity = np.asarray([[0.1, 0.2, 0.3], [0.11, 0.21, 0.31], [0.3, 0.2, 0.1], [0.31, 0.21, 0.11]])

    activity_result = compute_m3_sensor_activity_reconfiguration(
        activity,
        intervals,
        channel_names,
        support,
        vector_construction_id="bandpower-v0",
        vector_scaling_id="fixed-channel-scale-v0",
    )
    connectivity_result = compute_m3_sensor_connectivity_reconfiguration(
        connectivity,
        intervals,
        support,
        edge_order_sha256="a" * 64,
        edge_estimator_id="coherence-v0",
        edge_scaling_id="fixed-edge-scale-v0",
    )
    gap = intervals.copy()
    gap[2] += 0.05
    gap_result = compute_m3_sensor_activity_reconfiguration(
        activity,
        gap,
        channel_names,
        support,
        vector_construction_id="bandpower-v0",
        vector_scaling_id="fixed-channel-scale-v0",
    )

    _assert_exploratory(activity_result)
    _assert_exploratory(connectivity_result)
    assert activity_result.estimator_id != connectivity_result.estimator_id
    assert isinstance(activity_result.value, tuple) and len(activity_result.value) == 3
    assert activity_result.value[0] < 0.01
    assert activity_result.value[1] > 1.9
    assert gap_result.reason == "blocks_not_adjacent_or_gap_free"


def test_d1_weighted_efficiency_and_d2_pair_class_contrast_have_expected_values() -> None:
    support = _support()
    complete = np.ones((4, 4), dtype=float)
    np.fill_diagonal(complete, 0.0)
    efficiency = compute_d1_sensor_weighted_global_efficiency(
        complete,
        support,
        node_order_sha256="a" * 64,
        edge_estimator_id="synthetic-edge-v0",
        edge_rule_id="complete-unit-weights-v0",
    )

    coupling = np.full((4, 4), 0.2, dtype=float)
    class_a = np.zeros((4, 4), dtype=bool)
    class_b = np.zeros((4, 4), dtype=bool)
    for i, j in ((0, 1), (2, 3)):
        coupling[i, j] = coupling[j, i] = 0.8
        class_a[i, j] = class_a[j, i] = True
    for i, j in ((0, 2), (0, 3), (1, 2), (1, 3)):
        class_b[i, j] = class_b[j, i] = True
    np.fill_diagonal(coupling, 0.0)
    contrast = compute_d2_sensor_pairclass_difference(
        coupling,
        class_a,
        class_b,
        support,
        variant="near_far",
        pair_definition_id="synthetic-near-far-v0",
        coupling_estimator_id="synthetic-coherence-v0",
        node_order_sha256="b" * 64,
    )
    invalid_pair_mask = compute_d2_sensor_pairclass_difference(
        coupling,
        class_a.astype(np.int8),
        class_b,
        support,
        variant="near_far",
        pair_definition_id="synthetic-near-far-v0",
        coupling_estimator_id="synthetic-coherence-v0",
        node_order_sha256="b" * 64,
    )

    _assert_exploratory(efficiency)
    _assert_exploratory(contrast)
    assert np.isclose(efficiency.value, 1.0)
    assert np.isclose(contrast.value, 0.6)
    assert contrast.details["contrast_order"] == "class_a_minus_class_b"
    assert invalid_pair_mask.reason == "pair_class_mask_must_be_boolean"


def test_d1_inverse_weight_efficiency_handles_path_disconnection_and_node_relabeling() -> None:
    support = _support()
    path = np.asarray([[0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]])
    kwargs = {
        "node_order_sha256": "c" * 64,
        "edge_estimator_id": "synthetic-coupling-v0",
        "edge_rule_id": "synthetic-inverse-weight-v0",
    }
    result = compute_d1_sensor_weighted_global_efficiency(path, support, **kwargs)
    order = np.asarray([2, 0, 1])
    relabeled = compute_d1_sensor_weighted_global_efficiency(
        path[np.ix_(order, order)],
        support,
        **{**kwargs, "node_order_sha256": "d" * 64},
    )
    scaled = compute_d1_sensor_weighted_global_efficiency(
        0.5 * path,
        support,
        **{**kwargs, "edge_rule_id": "synthetic-half-weight-inverse-v0"},
    )
    disconnected = np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    disconnected_result = compute_d1_sensor_weighted_global_efficiency(
        disconnected, support, **kwargs
    )
    negative = path.copy()
    negative[0, 1] = negative[1, 0] = -0.1
    negative_result = compute_d1_sensor_weighted_global_efficiency(
        negative, support, **kwargs
    )
    gap_result = compute_d1_sensor_weighted_global_efficiency(
        path,
        MNPS31WindowSupport(**{**support.__dict__, "gap_free": False}),
        **kwargs,
    )

    for value in (result, relabeled, scaled, disconnected_result):
        _assert_exploratory(value)
    assert np.isclose(result.value, 5.0 / 12.0)
    assert np.isclose(relabeled.value, result.value)
    assert np.isclose(scaled.value, result.value / 2.0)
    assert np.isclose(disconnected_result.value, 1.0 / 3.0)
    assert negative_result.value is None
    assert negative_result.reason == "weights_must_be_symmetric_in_unit_interval"
    assert gap_result.value is None and gap_result.reason == "gap_or_segment_boundary"


def test_d2_pairclass_difference_preserves_labels_sign_and_pair_permutation() -> None:
    support = _support()
    coupling = np.asarray(
        [
            [0.0, 0.8, 0.2, 0.2],
            [0.8, 0.0, 0.2, 0.2],
            [0.2, 0.2, 0.0, 0.8],
            [0.2, 0.2, 0.8, 0.0],
        ]
    )
    names = ["F3", "C3", "P3", "O1"]
    class_a = np.zeros((4, 4), dtype=bool)
    class_b = np.zeros((4, 4), dtype=bool)
    for left, right in ((0, 1), (2, 3)):
        class_a[left, right] = class_a[right, left] = True
    for left, right in ((0, 2), (0, 3), (1, 2), (1, 3)):
        class_b[left, right] = class_b[right, left] = True
    original_order_hash = hashlib.sha256(
        "\n".join(name.casefold() for name in names).encode("utf-8")
    ).hexdigest()
    kwargs = {
        "variant": "within_between",
        "pair_definition_id": "synthetic-independent-groups-v0",
        "coupling_estimator_id": "synthetic-coupling-v0",
        "node_order_sha256": original_order_hash,
    }
    result = compute_d2_sensor_pairclass_difference(
        coupling, class_a, class_b, support, **kwargs
    )
    order = np.asarray([2, 0, 3, 1])
    permuted_names = [names[index] for index in order]
    permuted_order_hash = hashlib.sha256(
        "\n".join(name.casefold() for name in permuted_names).encode("utf-8")
    ).hexdigest()
    relabeled = compute_d2_sensor_pairclass_difference(
        coupling[np.ix_(order, order)],
        class_a[np.ix_(order, order)],
        class_b[np.ix_(order, order)],
        support,
        **{**kwargs, "node_order_sha256": permuted_order_hash},
    )
    reversed_classes = compute_d2_sensor_pairclass_difference(
        coupling, class_b, class_a, support, **kwargs
    )
    overlap = compute_d2_sensor_pairclass_difference(
        coupling, class_a, class_a, support, **kwargs
    )
    no_class_b = compute_d2_sensor_pairclass_difference(
        coupling, class_a, np.zeros_like(class_b), support, **kwargs
    )

    for value in (result, relabeled, reversed_classes):
        _assert_exploratory(value)
    assert np.isclose(result.value, 0.6)
    assert np.isclose(relabeled.value, result.value)
    assert np.isclose(reversed_classes.value, -result.value)
    assert overlap.value is None and overlap.reason == "overlapping_pair_classes"
    assert no_class_b.value is None and no_class_b.reason == "insufficient_pair_class_support"


def test_d3_rdm_and_e2_lz78_probe_record_unresolved_transform_choices() -> None:
    support = _support()
    channels = ["F3", "C3", "P3"]
    patterns = np.asarray([[0, 0], [3, 4], [0, 4]], dtype=float)
    rdm = compute_d3_sensor_rdm_dispersion(
        patterns,
        channels,
        support,
        feature_construction_id="synthetic-feature-v0",
        scaling_id="fixed-units-v0",
        feature_units="z-score",
    )
    rng = np.random.default_rng(91)
    random_symbols = rng.integers(0, 2, size=support.sample_count).tolist()
    lz = compute_e2_sensor_lz78_phrase_density(
        random_symbols,
        support,
        channel_name="F3",
        symbolization_id="synthetic-binary-sequence-v0",
    )
    constant_lz = compute_e2_sensor_lz78_phrase_density(
        [1] * support.sample_count,
        support,
        channel_name="F3",
        symbolization_id="synthetic-constant-sequence-v0",
    )

    _assert_exploratory(rdm)
    _assert_exploratory(lz)
    assert np.isclose(rdm.value, 4.0)
    assert rdm.units == "z-score"
    unsupported_rdm = compute_d3_sensor_rdm_dispersion(
        patterns,
        channels,
        MNPS31WindowSupport(**{**support.__dict__, "gap_free": False}),
        feature_construction_id="synthetic-feature-v0",
        scaling_id="fixed-units-v0",
        feature_units="z-score",
    )
    assert unsupported_rdm.reason == "gap_or_segment_boundary"
    assert unsupported_rdm.units == "z-score"
    assert lz.details["matched_reference_normalization"] == "not_applied"
    assert lz.details["channel_aggregation"] == "not_applied_single_channel"
    assert constant_lz.reason == "symbol_sequence_has_fewer_than_two_symbols"


def test_d3_sensor_rdm_exact_geometry_permutation_scaling_and_unsupported_controls() -> None:
    support = _support()
    names = ["F3", "C3", "P3"]
    patterns = np.asarray([[0.0, 0.0], [3.0, 4.0], [0.0, 4.0]])
    kwargs = {
        "feature_construction_id": "synthetic-fixed-v0",
        "scaling_id": "synthetic-euclidean-units-v0",
        "feature_units": "a.u.",
    }
    base = compute_d3_sensor_rdm_dispersion(patterns, names, support, **kwargs)
    permuted = compute_d3_sensor_rdm_dispersion(
        patterns[::-1], names[::-1], support, **kwargs
    )
    scaled = compute_d3_sensor_rdm_dispersion(
        2.0 * patterns,
        names,
        support,
        **{**kwargs, "scaling_id": "synthetic-doubled-units-v0"},
    )
    duplicated = compute_d3_sensor_rdm_dispersion(
        np.vstack([patterns, patterns[0]]),
        [*names, "O1"],
        support,
        **kwargs,
    )
    identical = compute_d3_sensor_rdm_dispersion(
        np.zeros_like(patterns), names, support, **kwargs
    )
    nonfinite = compute_d3_sensor_rdm_dispersion(
        np.asarray([[0.0, 0.0], [np.nan, 1.0], [0.0, 4.0]]), names, support, **kwargs
    )
    gap = compute_d3_sensor_rdm_dispersion(
        patterns,
        names,
        MNPS31WindowSupport(**{**support.__dict__, "gap_free": False}),
        **kwargs,
    )

    for result in (base, permuted, scaled, duplicated, identical):
        _assert_exploratory(result)
    assert base.value == 4.0
    assert permuted.value == base.value
    assert scaled.value == 8.0
    assert duplicated.value == 3.5
    assert identical.value == 0.0 and identical.valid is True
    assert nonfinite.value is None and nonfinite.reason == "non_finite_input"
    assert gap.value is None and gap.reason == "gap_or_segment_boundary"


def test_e2_lz78_permutation_reference_is_reproducible_and_keeps_probe_unratified() -> None:
    support = _support()
    rng = np.random.default_rng(123)
    periodic = [0, 1] * (support.sample_count // 2)
    iid = rng.integers(0, 2, size=support.sample_count).tolist()
    markov = np.empty(support.sample_count, dtype=int)
    markov[0] = 0
    for index in range(1, support.sample_count):
        markov[index] = markov[index - 1] if rng.random() < 0.95 else 1 - markov[index - 1]
    kwargs = {
        "channel_name": "F3",
        "symbolization_id": "synthetic-binary-sequence-v0",
        "reference_replicates": 32,
        "random_seed": 310905,
    }
    first = compute_e2_sensor_lz78_permutation_reference_ratio(
        periodic, support, **kwargs
    )
    replay = compute_e2_sensor_lz78_permutation_reference_ratio(
        periodic, support, **kwargs
    )
    invalid_replicates = compute_e2_sensor_lz78_permutation_reference_ratio(
        periodic, support, **{**kwargs, "reference_replicates": 4}
    )
    iid_result = compute_e2_sensor_lz78_permutation_reference_ratio(
        iid, support, **{**kwargs, "reference_replicates": 64}
    )
    markov_result = compute_e2_sensor_lz78_permutation_reference_ratio(
        markov.tolist(), support, **{**kwargs, "reference_replicates": 64}
    )
    constant_result = compute_e2_sensor_lz78_permutation_reference_ratio(
        [1] * support.sample_count, support, **kwargs
    )
    gap_support = MNPS31WindowSupport(**{**support.__dict__, "gap_free": False})
    gap_result = compute_e2_sensor_lz78_permutation_reference_ratio(
        periodic, gap_support, **kwargs
    )

    _assert_exploratory(first)
    assert first.value == replay.value
    assert first.value is not None and first.value < 1.0
    assert first.estimator_id == "E2.sensor_lz78_permutation_reference_ratio.v0"
    assert first.details["normalization"] == (
        "observed_phrase_count_divided_by_median_exact_multiset_permutation_phrase_count"
    )
    assert first.details["selection_status"] == "unratified_g1_candidate_probe"
    assert first.details["reference_replicates"] == 32
    assert invalid_replicates.value is None
    assert invalid_replicates.reason == "reference_replicates_must_be_integer_at_least_8"
    _assert_exploratory(iid_result)
    _assert_exploratory(markov_result)
    assert iid_result.value is not None and np.isfinite(iid_result.value)
    assert markov_result.value is not None and np.isfinite(markov_result.value)
    assert constant_result.value is None
    assert constant_result.reason == "symbol_sequence_has_fewer_than_two_symbols"
    assert gap_result.value is None
    assert gap_result.reason == "gap_or_segment_boundary"
    assert gap_result.estimator_id == "E2.sensor_lz78_permutation_reference_ratio.v0"


def test_e3_sensor_participation_ratio_recovers_rank_one_fixture_and_rejects_gap() -> None:
    support = _support(sample_count=400, sample_rate_hz=100.0)
    rng = np.random.default_rng(13)
    base = rng.normal(size=support.sample_count)
    signals = np.vstack([base, 2.0 * base, -0.5 * base, 3.0 * base])
    channels = ["F3", "C3", "P3", "O1"]
    result = compute_e3_sensor_participation_ratio(signals, channels, support)
    gap_support = MNPS31WindowSupport(**{**support.__dict__, "gap_free": False})
    rejected = compute_e3_sensor_participation_ratio(signals, channels, gap_support)

    _assert_exploratory(result)
    assert np.isclose(result.value, 1.0, atol=1e-8)
    assert result.units == "sensor_directions"
    assert result.details["effective_sample_support_gate"] == "not_assessed"
    assert rejected.reason == "gap_or_segment_boundary"
    assert rejected.units == "sensor_directions"
