"""Synthetic estimator controls for experimental D2/SCG."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy import signal
from scipy.stats import rankdata

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.mnps.v3_1.candidates import MNPS31WindowSupport  # noqa: E402
from mndm.mnps.v3_1.d2_spatial_gradient import (  # noqa: E402
    SCG_ESTIMATOR_ID,
    _dwpli2_debiased_matrix,
    compute_d2_spatial_coupling_gradient,
    spherical_geodesic_distances_m,
    standard_template_positions_m,
)
from mndm.mnps.v3_1.d2_position_templates import (  # noqa: E402
    DS004148_CORE60_CHANNELS,
    DS004148_CORE60_SET_SHA256,
)
from mndm.mnps.v3_1.realizations import CORE19_CHANNELS  # noqa: E402


SAMPLE_RATE_HZ = 128.0
DURATION_SEC = 60


def _support(sample_count: int) -> MNPS31WindowSupport:
    return MNPS31WindowSupport(
        window_start_sec=0.0,
        window_end_sec=sample_count / SAMPLE_RATE_HZ,
        sample_rate_hz=SAMPLE_RATE_HZ,
        sample_count=sample_count,
        segment_id="synthetic-scg",
        source_is_continuous=True,
        timestamps_uniform=True,
        gap_free=True,
        bad_interval_free=True,
        within_recording_bounds=True,
        effective_filter_support_valid=True,
    )


def _core19_positions() -> tuple[np.ndarray, np.ndarray]:
    positions, _, error = standard_template_positions_m(CORE19_CHANNELS)
    assert error is None and positions is not None
    distance_matrix = spherical_geodesic_distances_m(positions)
    return positions, distance_matrix


def _synthesize_alpha_network(seed: int, profile: str) -> np.ndarray:
    """Build pair-specific phase-lagged alpha sources with known distance trend."""
    rng = np.random.default_rng(seed)
    sample_count = int(SAMPLE_RATE_HZ * DURATION_SEC)
    sos = signal.butter(4, [8.0, 13.0], btype="bandpass", fs=SAMPLE_RATE_HZ, output="sos")
    _, distances = _core19_positions()
    pair_distances = distances[np.triu_indices(len(CORE19_CHANNELS), k=1)]
    max_distance = float(np.max(pair_distances))
    values = np.zeros((len(CORE19_CHANNELS), sample_count), dtype=np.float64)

    for left in range(len(CORE19_CHANNELS)):
        for right in range(left + 1, len(CORE19_CHANNELS)):
            distance = float(distances[left, right])
            if profile == "near":
                amplitude = 0.05 + 2.5 * np.exp(-distance / 0.025)
            elif profile == "neutral":
                amplitude = 0.75
            elif profile == "far":
                amplitude = 0.05 + 2.5 * np.exp(-(max_distance - distance) / 0.025)
            else:  # pragma: no cover - test helper guard
                raise ValueError(profile)

            source = signal.sosfiltfilt(sos, rng.normal(size=sample_count))
            quadrature = np.imag(signal.hilbert(source))
            values[left] += amplitude * source
            values[right] += amplitude * quadrature

    values += 0.3 * rng.normal(size=values.shape)
    return values


def _pair_metric_signals(values: np.ndarray) -> tuple[float, float, np.ndarray]:
    positions, distance_matrix = _core19_positions()
    del positions
    pair_indices = np.triu_indices(values.shape[0], k=1)
    pair_distances = distance_matrix[pair_indices]
    referenced = values - np.mean(values, axis=0, keepdims=True)
    coupling, _, error = _dwpli2_debiased_matrix(referenced, SAMPLE_RATE_HZ)
    assert error is None and coupling is not None
    scg = _negative_spearman(pair_distances, coupling[pair_indices])
    pearson = np.corrcoef(referenced)
    pearson_scg = _negative_spearman(pair_distances, np.abs(pearson[pair_indices]))
    return scg, pearson_scg, coupling


def _negative_spearman(distance: np.ndarray, coupling: np.ndarray) -> float:
    return float(-np.corrcoef(rankdata(distance), rankdata(coupling))[0, 1])


def test_scg_synthetic_near_neutral_and_far_phase_lag_controls() -> None:
    values = {
        profile: np.asarray([_pair_metric_signals(_synthesize_alpha_network(seed, profile))[0] for seed in range(4)])
        for profile in ("near", "neutral", "far")
    }

    assert np.median(values["near"]) > 0.30
    assert abs(float(np.mean(values["neutral"]))) < 0.10
    assert np.median(values["far"]) < -0.10


def test_scg_zero_lag_source_control_uses_same_reference_for_both_measures() -> None:
    """At moderate smooth-source amplitude, dwPLI2 SCG shifts less than Pearson SCG."""
    positions, _ = _core19_positions()
    directions = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    source_loadings = np.exp(-np.sum((directions - directions[0]) ** 2, axis=1) / (2.0 * 0.8**2))
    source_filter = signal.butter(4, [8.0, 13.0], btype="bandpass", fs=SAMPLE_RATE_HZ, output="sos")
    deltas = []

    for seed in range(4):
        base = _synthesize_alpha_network(seed, "near")
        source = signal.sosfiltfilt(source_filter, np.random.default_rng(1000 + seed).normal(size=base.shape[1]))
        baseline = _pair_metric_signals(base)
        injected = _pair_metric_signals(base + 1.0 * source_loadings[:, None] * source[None, :])
        deltas.append((abs(injected[0] - baseline[0]), abs(injected[1] - baseline[1])))

    mean_dwpli2_change, mean_pearson_change = np.mean(np.asarray(deltas), axis=0)
    assert mean_pearson_change > 0.20
    assert mean_dwpli2_change < 0.25 * mean_pearson_change


def test_exact_common_mode_is_removed_by_the_declared_average_reference() -> None:
    base = _synthesize_alpha_network(2, "near")
    common = signal.sosfiltfilt(
        signal.butter(4, [8.0, 13.0], btype="bandpass", fs=SAMPLE_RATE_HZ, output="sos"),
        np.random.default_rng(99).normal(size=base.shape[1]),
    )
    np.testing.assert_allclose(
        base - base.mean(axis=0, keepdims=True),
        base + 20.0 * common[None, :] - (base + 20.0 * common[None, :]).mean(axis=0, keepdims=True),
        atol=1e-12,
        rtol=0.0,
    )


def test_scg_label_permutation_breaks_the_geometry_coupling_association_on_average() -> None:
    _, distances = _core19_positions()
    values = _synthesize_alpha_network(0, "near")
    _, _, coupling = _pair_metric_signals(values)
    pair_indices = np.triu_indices(len(CORE19_CHANNELS), k=1)
    coupling_values = coupling[pair_indices]
    rng = np.random.default_rng(402)
    permuted_scg = []

    for _ in range(100):
        permutation = rng.permutation(len(CORE19_CHANNELS))
        permuted_distances = distances[np.ix_(permutation, permutation)][pair_indices]
        permuted_scg.append(_negative_spearman(permuted_distances, coupling_values))

    assert abs(float(np.mean(permuted_scg))) < 0.06


def test_scg_end_to_end_uses_declared_core19_and_preserves_provenance() -> None:
    values = _synthesize_alpha_network(0, "near")
    positions, _ = _core19_positions()
    result = compute_d2_spatial_coupling_gradient(
        values,
        CORE19_CHANNELS,
        positions,
        _support(values.shape[1]),
        surface_id="core19",
        source_id="synthetic-source",
        source_content_sha256="a" * 64,
        preprocessing_id="synthetic-unfiltered-average-reference-v1",
        reference_mode="average_over_exact_surface",
        reference_id="average-reference-core19-v1",
        position_source="standard_template",
        coordinate_frame="mri",
        position_template_id="mne_standard_1005_mne_1.6.1_frozen_v1",
    )

    assert result.scg.computed is True
    assert result.scg.estimator_id == SCG_ESTIMATOR_ID
    assert result.scg.value is not None and result.scg.value > 0.30
    assert result.scg.qualified is False
    assert result.scg.details["channel_count"] == 19
    assert result.scg.details["pair_count"] == 171
    assert result.scg.details["position_source"] == "standard_template"
    assert result.scg.details["coordinate_frame"] == "mri"
    assert result.scg.details["position_sha256"]
    assert result.scg.details["frequency_band_aggregation"] == "unweighted_arithmetic_mean_of_per_frequency_dWPLI2"
    assert result.scg.details["segments_nonoverlapping"] is True
    assert result.scg.details["segment_count"] == 30
    assert result.ldcb_sidecar.computed is True
    assert result.pair_channel_indices is not None and result.pair_channel_indices.shape == (171, 2)
    assert result.pair_dwpli2_debiased is not None and result.pair_dwpli2_debiased.shape == (171,)


def test_computable_small_native_surface_is_emitted_with_core19_qc_warning() -> None:
    labels = CORE19_CHANNELS[:5]
    values = _synthesize_alpha_network(0, "near")[:5]
    positions, _, error = standard_template_positions_m(labels)
    assert error is None and positions is not None
    result = compute_d2_spatial_coupling_gradient(
        values,
        labels,
        positions,
        _support(values.shape[1]),
        surface_id="core19_available_five",
        requested_surface_id="core19",
        requested_channel_names=CORE19_CHANNELS,
        surface_qc={"warnings": ["requested_surface_missing_channels"], "omitted_channel_labels": list(CORE19_CHANNELS[5:])},
        source_id="synthetic-source",
        source_content_sha256="b" * 64,
        preprocessing_id="synthetic-unfiltered",
        reference_mode="average_over_exact_surface",
        reference_id="average-reference-five-v1",
        position_source="standard_template",
        coordinate_frame="mri",
        position_template_id="mne_standard_1005_mne_1.6.1_frozen_v1",
    )

    assert result.scg.computed is True
    assert result.scg.details["channel_count"] == 5
    assert result.scg.details["surface_realization"] == "nonstandard_exploratory_set"
    assert result.scg.details["surface_qc_warnings"] == ["below_core19_channel_count", "requested_surface_missing_channels"]
    assert result.scg.details["requested_surface_id"] == "core19"
    assert result.scg.details["requested_channel_count"] == 19
    assert result.scg.details["missing_requested_channel_labels"] == list(CORE19_CHANNELS[5:])


def test_dWPLI2_technical_epoch_floor_has_explicit_23_and_24_segment_behavior() -> None:
    rng = np.random.default_rng(773)
    at_floor, floor_details, floor_error = _dwpli2_debiased_matrix(
        rng.normal(size=(3, 48 * int(SAMPLE_RATE_HZ))), SAMPLE_RATE_HZ
    )
    below_floor, below_details, below_error = _dwpli2_debiased_matrix(
        rng.normal(size=(3, 46 * int(SAMPLE_RATE_HZ))), SAMPLE_RATE_HZ
    )

    assert floor_error is None and at_floor is not None
    assert floor_details["segment_count"] == 24
    assert floor_details["spectral_observation_count_per_frequency"] == 24
    assert floor_details["minimum_nonoverlapping_segment_count"] == 24
    assert below_floor is None
    assert below_error == "insufficient_nonoverlapping_dwpli_segments"
    assert below_details["segment_count"] == 23


def test_core60_template_is_a_distinct_pinned_exploratory_surface_realization() -> None:
    rng = np.random.default_rng(404)
    sample_count = int(SAMPLE_RATE_HZ * DURATION_SEC)
    values = rng.normal(size=(len(DS004148_CORE60_CHANNELS), sample_count))
    positions, _, error = standard_template_positions_m(DS004148_CORE60_CHANNELS)
    assert error is None and positions is not None

    result = compute_d2_spatial_coupling_gradient(
        values,
        DS004148_CORE60_CHANNELS,
        positions,
        _support(sample_count),
        surface_id="core60",
        requested_surface_id="core60",
        requested_channel_names=DS004148_CORE60_CHANNELS,
        source_id="synthetic-source",
        source_content_sha256="c" * 64,
        preprocessing_id="synthetic-unfiltered",
        reference_mode="average_over_exact_surface",
        reference_id=f"average-reference:{DS004148_CORE60_SET_SHA256}",
        position_source="standard_template",
        coordinate_frame="mri",
        position_template_id="mne_standard_1005_mne_1.6.1_frozen_v1",
    )

    assert result.scg.details["surface_realization"] == "ds004148_core60_study_common_set"
    assert result.scg.details["channel_set_sha256"] == DS004148_CORE60_SET_SHA256
    assert result.scg.details["requested_channel_count"] == 60
    assert result.scg.details["missing_requested_channel_labels"] == []
