"""Experimental spatial coupling gradient (SCG) for EEG sensor surfaces.

The primary scalar is ``-Spearman(distance, dwPLI2_debiased)`` over all
unordered sensor pairs. The distances are origin-centered spherical-geodesic
distances from an explicitly identified sensor-position realization. The
secondary LDCB value contrasts the closest and farthest distance quartiles.

This is a sensor-observation-space measurement, not a cortical network or
anatomical connectivity claim. It remains D2/SCG experimental and unqualified.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from typing import Literal, Mapping, Sequence

import numpy as np
from scipy import signal
from scipy.stats import rankdata

from .candidates import MNPS31WindowSupport
from .d2_position_templates import (
    COORDINATE_FRAME as STANDARD_TEMPLATE_COORDINATE_FRAME,
    COORDINATE_UNITS as STANDARD_TEMPLATE_COORDINATE_UNITS,
    DS004148_CORE60_CHANNELS,
    DS004148_CORE60_SET_SHA256,
    POSITION_TABLE_SHA256 as STANDARD_TEMPLATE_POSITION_TABLE_SHA256,
    STANDARD_1005_POSITIONS_M,
    TEMPLATE_ID as STANDARD_1005_TEMPLATE_ID,
)
from .eeg_metrics import EEGCandidateMeasurement, _computed, _support_reason, _unsupported
from .realizations import CORE19_CHANNELS


SCG_ESTIMATOR_ID = "spatial_dwpli_distance_spearman_alpha.v1"
SCG_CONNECTIVITY_ESTIMATOR_ID = "dwpli2_debiased_per_frequency_alpha_8_13hz_nonoverlapping_2sec.v1"
SCG_GEOMETRY_ESTIMATOR_ID = "spherical_geodesic_normalized_position_vectors.v1"
LDCB_SIDECAR_ESTIMATOR_ID = "local_distributed_coupling_contrast_distance_quartiles_dwpli2_alpha.v1"
SCG_MIN_CHANNELS = 3
SCG_STANDARD_MIN_CHANNELS = 19
SCG_ALPHA_BAND_HZ = (8.0, 13.0)
SCG_SEGMENT_LENGTH_SEC = 2.0
SCG_SEGMENT_STEP_SEC = 2.0
SCG_MIN_SEGMENT_COUNT = 24

PositionSource = Literal["measured", "bids", "standard_template"]


@dataclass(frozen=True)
class D2SpatialGradientResult:
    """Primary SCG scalar and its descriptive LDCB sidecar."""

    scg: EEGCandidateMeasurement
    ldcb_sidecar: EEGCandidateMeasurement
    pair_channel_indices: np.ndarray | None
    pair_distances_m: np.ndarray | None
    pair_dwpli2_debiased: np.ndarray | None
    coupling_matrix: np.ndarray | None


def _sha256_labels(labels: Sequence[str], *, ordered: bool) -> str:
    normalized = [str(label).strip().casefold() for label in labels]
    if not ordered:
        normalized.sort()
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()


def _position_hash(labels: Sequence[str], positions_m: np.ndarray) -> str:
    rows = []
    for label, xyz in zip(labels, positions_m, strict=True):
        rows.append("\0".join((str(label).strip().casefold(), *(float(value).hex() for value in xyz))))
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def spherical_geodesic_distances_m(positions_m: np.ndarray) -> np.ndarray:
    """Map centered 3-D sensor coordinates to an origin-centered sphere.

    Each position is projected radially to the median radius of the supplied
    surface. Pairwise distances are then great-circle arc lengths. The input
    frame origin is part of the geometry contract and is never estimated from
    the selected surface, so Core19 and Core60 can share one template frame.
    """
    positions = np.asarray(positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 2:
        raise ValueError("positions_must_be_n_channels_by_3")
    if not np.isfinite(positions).all():
        raise ValueError("positions_must_be_finite")
    radii = np.linalg.norm(positions, axis=1)
    if not np.isfinite(radii).all() or np.any(radii <= 0.0):
        raise ValueError("position_radius_must_be_positive")
    directions = positions / radii[:, None]
    radius_m = float(np.median(radii))
    cosine = np.clip(directions @ directions.T, -1.0, 1.0)
    distances = np.arccos(cosine) * radius_m
    np.fill_diagonal(distances, 0.0)
    return distances


def standard_template_positions_m(channel_names: Sequence[str]) -> tuple[np.ndarray | None, dict[str, object], str | None]:
    """Resolve labels from the frozen standard_1005 position table.

    Only the explicit Core19/Core60 template label set is present in this
    version. Unknown labels stay unsupported rather than receiving guessed
    coordinates.
    """
    if isinstance(channel_names, (str, bytes)):
        return None, {}, "invalid_channel_label_sequence"
    labels = tuple(str(label).strip() for label in channel_names)
    normalized = tuple(label.casefold() for label in labels)
    if not labels or any(not label for label in labels):
        return None, {}, "invalid_channel_label"
    if len(set(normalized)) != len(normalized):
        return None, {}, "duplicate_channel_label"
    missing = sorted({label for label in normalized if label not in STANDARD_1005_POSITIONS_M})
    if missing:
        return None, {
            "position_source": "standard_template",
            "position_template_id": STANDARD_1005_TEMPLATE_ID,
            "missing_template_labels": missing,
        }, "unknown_position_template_labels"
    positions = np.asarray([STANDARD_1005_POSITIONS_M[label] for label in normalized], dtype=np.float64)
    metadata = {
        "position_source": "standard_template",
        "position_template_id": STANDARD_1005_TEMPLATE_ID,
        "coordinate_frame": STANDARD_TEMPLATE_COORDINATE_FRAME,
        "coordinate_units": STANDARD_TEMPLATE_COORDINATE_UNITS,
        "template_position_table_sha256": STANDARD_TEMPLATE_POSITION_TABLE_SHA256,
        "position_sha256": _position_hash(labels, positions),
        "template_label_count": len(STANDARD_1005_POSITIONS_M),
        "position_labels": list(labels),
    }
    return positions, metadata, None


def _dwpli2_debiased_matrix(
    values: np.ndarray,
    sample_rate_hz: float,
    *,
    band_hz: tuple[float, float] = SCG_ALPHA_BAND_HZ,
    segment_length_sec: float = SCG_SEGMENT_LENGTH_SEC,
    segment_step_sec: float = SCG_SEGMENT_STEP_SEC,
    min_segment_count: int = SCG_MIN_SEGMENT_COUNT,
) -> tuple[np.ndarray | None, dict[str, object], str | None]:
    """Estimate alpha-band mean dWPLI² from nonoverlapping epochs.

    For each pair and frequency bin, the debiased estimator is
    ``(sum(I)**2 - sum(I**2)) / (sum(abs(I))**2 - sum(I**2))`` over the
    declared nonoverlapping epochs, where ``I`` is the imaginary
    cross-spectrum. The returned band value is the unweighted arithmetic mean
    of those per-frequency estimates. Finite negative values are retained
    because dWPLI² is not constrained to be nonnegative in finite samples. An
    exactly zero denominator maps to zero coupling by definition.
    """
    data = np.asarray(values, dtype=np.float64)
    sfreq = float(sample_rate_hz)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 3:
        return None, {}, "invalid_signal_matrix_shape_or_sample_count"
    if not np.isfinite(data).all():
        return None, {}, "non_finite_input"
    if not math.isfinite(sfreq) or sfreq <= 0.0 or sfreq / 2.0 < band_hz[1]:
        return None, {}, "sampling_frequency_does_not_support_alpha_band"
    if not math.isfinite(segment_length_sec) or segment_length_sec <= 0.0:
        return None, {}, "invalid_segment_length"
    if not math.isfinite(segment_step_sec) or segment_step_sec <= 0.0:
        return None, {}, "invalid_segment_step"

    nperseg = int(round(segment_length_sec * sfreq))
    step = int(round(segment_step_sec * sfreq))
    if nperseg < 8 or step < 1 or data.shape[1] < nperseg:
        return None, {}, "insufficient_samples_for_declared_segments"
    if step != nperseg:
        return None, {}, "overlapping_segments_not_supported_by_scg_v1"
    noverlap = nperseg - step
    frequencies, _, spectra = signal.spectrogram(
        data,
        fs=sfreq,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        mode="complex",
        axis=-1,
    )
    keep = (frequencies >= band_hz[0]) & (frequencies <= band_hz[1])
    segment_count = int(spectra.shape[-1])
    frequency_bin_count = int(np.count_nonzero(keep))
    details: dict[str, object] = {
        "coupling_estimator_id": SCG_CONNECTIVITY_ESTIMATOR_ID,
        "coupling": "debiased_squared_weighted_phase_lag_index_dWPLI2",
        "debiased_wpli_formula": "(sum(I)^2-sum(I^2))/(sum(abs(I))^2-sum(I^2))",
        "imaginary_cross_spectrum_observations": "nonoverlapping_segments_estimated_separately_per_frequency_bin",
        "frequency_band_hz": [float(band_hz[0]), float(band_hz[1])],
        "frequency_bin_count": frequency_bin_count,
        "frequency_band_aggregation": "unweighted_arithmetic_mean_of_per_frequency_dWPLI2",
        "segment_length_sec": nperseg / sfreq,
        "segment_step_sec": step / sfreq,
        "segment_count": segment_count,
        "spectral_observation_count_per_frequency": segment_count,
        "minimum_nonoverlapping_segment_count": int(min_segment_count),
        "minimum_segment_count_role": "technical_computability_floor_not_independence_or_bias_calibration",
        "nonoverlapping_epochs_independence_claim": False,
        "segments_nonoverlapping": True,
        "spectrogram_window": "hann",
        "spectrogram_detrend": "constant_per_segment",
        "zero_imaginary_energy_pair_rule": "dWPLI2=0_when_denominator_is_exactly_zero",
    }
    if not np.any(keep):
        return None, details, "alpha_band_frequency_bins_unavailable"
    if segment_count < min_segment_count:
        return None, details, "insufficient_nonoverlapping_dwpli_segments"
    selected = spectra[:, keep, :]
    if not np.isfinite(selected.real).all() or not np.isfinite(selected.imag).all():
        return None, details, "non_finite_cross_spectral_estimates"

    n_channels = data.shape[0]
    coupling = np.eye(n_channels, dtype=np.float64)
    zero_energy_pair_frequency_count = 0
    zero_energy_pair_count = 0
    for left in range(n_channels):
        left_spectrum = selected[left]
        for right in range(left + 1, n_channels):
            imag_by_frequency = np.imag(left_spectrum * np.conjugate(selected[right]))
            frequency_estimates = []
            pair_has_zero_energy_bin = False
            for imag in imag_by_frequency:
                imag = np.asarray(imag, dtype=np.float64).reshape(-1)
                sum_imag = float(np.sum(imag, dtype=np.float64))
                sum_imag_squared = float(np.dot(imag, imag))
                sum_abs = float(np.sum(np.abs(imag), dtype=np.float64))
                numerator = sum_imag * sum_imag - sum_imag_squared
                denominator = sum_abs * sum_abs - sum_imag_squared
                if not math.isfinite(denominator) or not math.isfinite(numerator):
                    return None, details, "non_finite_dwpli_intermediate"
                if denominator <= 0.0:
                    if denominator == 0.0 and numerator == 0.0:
                        estimate = 0.0
                        zero_energy_pair_frequency_count += 1
                        pair_has_zero_energy_bin = True
                    else:
                        return None, details, "invalid_dwpli_debiasing_denominator"
                else:
                    estimate = numerator / denominator
                frequency_estimates.append(float(np.clip(estimate, -1.0, 1.0)))
            if pair_has_zero_energy_bin:
                zero_energy_pair_count += 1
            band_estimate = float(np.mean(frequency_estimates, dtype=np.float64))
            coupling[left, right] = coupling[right, left] = band_estimate
    details["zero_imaginary_energy_pair_frequency_count"] = zero_energy_pair_frequency_count
    details["zero_imaginary_energy_pair_count"] = zero_energy_pair_count
    details["finite_negative_dwpli2_pair_count"] = int(np.count_nonzero(coupling[np.triu_indices(n_channels, k=1)] < 0.0))
    return coupling, details, None


def compute_d2_spatial_coupling_gradient(
    sensor_signals: np.ndarray,
    channel_names: Sequence[str],
    positions_m: np.ndarray,
    support: MNPS31WindowSupport,
    *,
    surface_id: str,
    source_id: str,
    source_content_sha256: str,
    preprocessing_id: str,
    reference_mode: Literal["average_over_exact_surface"],
    reference_id: str,
    position_source: PositionSource,
    coordinate_frame: str,
    position_template_id: str | None = None,
    requested_surface_id: str | None = None,
    requested_channel_names: Sequence[str] | None = None,
    surface_qc: Mapping[str, object] | None = None,
    allow_unknown_support_for_computation: bool = False,
    signal_units: str = "microvolt",
    alpha_band_hz: tuple[float, float] = SCG_ALPHA_BAND_HZ,
    segment_length_sec: float = SCG_SEGMENT_LENGTH_SEC,
    segment_step_sec: float = SCG_SEGMENT_STEP_SEC,
    min_segment_count: int = SCG_MIN_SEGMENT_COUNT,
) -> D2SpatialGradientResult:
    """Compute the experimental primary SCG scalar and LDCB quartile sidecar.

    ``sensor_signals`` must be the complete declared surface in ``channel_names``
    order. The proposal's reference contract is the instantaneous mean of that
    exact surface. Position arrays are in meters in a declared, origin-centered
    frame. This function reads no raw files or sidecars.
    """
    labels = tuple(str(name).strip() for name in channel_names) if not isinstance(channel_names, (str, bytes)) else ()
    normalized_labels = tuple(label.casefold() for label in labels)
    core19_order = tuple(label.casefold() for label in CORE19_CHANNELS)
    core60_order = tuple(label.casefold() for label in DS004148_CORE60_CHANNELS)
    is_core19_set = set(normalized_labels) == set(core19_order)
    is_core60_set = set(normalized_labels) == set(core60_order)
    if is_core19_set:
        surface_realization = "core19_standard_set"
    elif is_core60_set:
        surface_realization = "ds004148_core60_study_common_set"
    else:
        surface_realization = "nonstandard_exploratory_set"
    requested_labels = (
        tuple(str(label).strip() for label in requested_channel_names)
        if requested_channel_names is not None and not isinstance(requested_channel_names, (str, bytes))
        else labels
    )
    normalized_label_set = set(normalized_labels)
    missing_requested_labels = [label for label in requested_labels if label.casefold() not in normalized_label_set]
    qc_details = dict(surface_qc or {})
    qc_warnings = list(qc_details.get("warnings", [])) if isinstance(qc_details.get("warnings", []), (list, tuple)) else []
    if len(labels) < SCG_STANDARD_MIN_CHANNELS:
        qc_warnings.append("below_core19_channel_count")
    if is_core19_set and normalized_labels != core19_order:
        qc_warnings.append("core19_channel_order_differs_from_frozen_order")
    if is_core60_set and normalized_labels != core60_order:
        qc_warnings.append("core60_channel_order_differs_from_frozen_order")
    base_details: dict[str, object] = {
        "construct_id": "spatial_coupling_gradient",
        "construct_name": "Spatial Coupling Gradient",
        "short_name": "SCG",
        "historical_aliases": ["D2", "LDCB"],
        "surface_id": str(surface_id),
        "requested_surface_id": str(requested_surface_id or surface_id),
        "surface_realization": surface_realization,
        "surface_qc_warnings": sorted(set(str(warning) for warning in qc_warnings)),
        "surface_qc": qc_details,
        "channel_labels": list(labels),
        "channel_count": len(labels),
        "channel_order_sha256": _sha256_labels(labels, ordered=True),
        "channel_set_sha256": _sha256_labels(labels, ordered=False),
        "requested_channel_labels": list(requested_labels),
        "requested_channel_count": len(requested_labels),
        "requested_channel_set_sha256": _sha256_labels(requested_labels, ordered=False),
        "missing_requested_channel_labels": missing_requested_labels,
        "source_id": source_id,
        "source_content_sha256": source_content_sha256,
        "preprocessing_id": preprocessing_id,
        "signal_units": signal_units,
        "reference_mode": reference_mode,
        "reference_id": reference_id,
        "position_source": position_source,
        "coordinate_frame": coordinate_frame,
        "coordinate_units": "m",
        "position_template_id": position_template_id,
        "geometry_estimator_id": SCG_GEOMETRY_ESTIMATOR_ID,
        "qualified": False,
        "support": support.__dict__,
    }

    def unsupported(reason: str, *, extra: dict[str, object] | None = None) -> D2SpatialGradientResult:
        details = {**base_details, **(extra or {})}
        scg = _unsupported(
            candidate_id="D2", estimator_id=SCG_ESTIMATOR_ID, support=support,
            reason=reason, observation_space="sensor_geometry_sensor_connectivity",
            details=details,
        )
        ldcb = _unsupported(
            candidate_id="D2", estimator_id=LDCB_SIDECAR_ESTIMATOR_ID, support=support,
            reason=reason, observation_space="sensor_geometry_sensor_connectivity",
            details=details,
        )
        return D2SpatialGradientResult(scg, ldcb, None, None, None, None)

    support_fields = (
        ("source_is_continuous", support.source_is_continuous, "non_continuous_source"),
        ("timestamps_uniform", support.timestamps_uniform, "irregular_sample_times"),
        ("gap_free", support.gap_free, "gap_or_segment_boundary"),
        ("bad_interval_free", support.bad_interval_free, "bad_interval_overlap"),
        ("within_recording_bounds", support.within_recording_bounds, "recording_boundary_overrun"),
        ("effective_filter_support_valid", support.effective_filter_support_valid, "insufficient_filter_edge_support"),
    )
    unknown_support_fields = [name for name, value, _ in support_fields if value is None]
    if allow_unknown_support_for_computation and unknown_support_fields:
        invalid_fields = [
            name for name, value, _ in support_fields
            if value is not None and not isinstance(value, (bool, np.bool_))
        ]
        if invalid_fields:
            return unsupported("support_audit_flag_not_boolean", extra={"invalid_support_fields": invalid_fields})
        known_failure = next(
            (reason for _, value, reason in support_fields if isinstance(value, (bool, np.bool_)) and not bool(value)),
            None,
        )
        if known_failure:
            return unsupported(known_failure, extra={"support_audit_unknown_fields": unknown_support_fields})
        # Unknown provenance is retained on ``support`` but does not suppress
        # numerical computation. Run the normal bounds checks with only those
        # unknown flags provisionally filled as true; known false flags above
        # still reject the measurement.
        validation_support = replace(
            support,
            **{name: True for name in unknown_support_fields},
        )
        support_error = _support_reason(validation_support)
        if support_error:
            return unsupported(support_error, extra={"support_audit_unknown_fields": unknown_support_fields})
        base_details["support_audit_status"] = "incomplete_computed_exploratory"
        base_details["support_audit_unknown_fields"] = unknown_support_fields
    else:
        support_error = _support_reason(support)
        if support_error:
            return unsupported(support_error)
    if len(labels) < SCG_MIN_CHANNELS:
        return unsupported("fewer_than_3_channels_for_spatial_gradient")
    if any(not label for label in labels) or len({x.casefold() for x in labels}) != len(labels):
        return unsupported("invalid_or_duplicate_channel_labels")
    if any(not str(value).strip() for value in (surface_id, source_id, preprocessing_id, reference_id, coordinate_frame)):
        return unsupported("measurement_provenance_missing")
    if (
        not isinstance(source_content_sha256, str)
        or len(source_content_sha256) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in source_content_sha256)
    ):
        return unsupported("source_content_hash_missing_or_invalid")
    if reference_mode != "average_over_exact_surface":
        return unsupported("complete_set_average_reference_required")
    if position_source not in {"measured", "bids", "standard_template"}:
        return unsupported("unknown_position_source")
    if position_source == "standard_template" and not str(position_template_id or "").strip():
        return unsupported("position_template_identity_missing")
    if str(surface_id).strip().casefold() == "core19" and not is_core19_set:
        return unsupported("declared_core19_surface_channel_set_mismatch")
    if str(surface_id).strip().casefold() == "core60":
        if _sha256_labels(DS004148_CORE60_CHANNELS, ordered=False) != DS004148_CORE60_SET_SHA256:
            return unsupported("pinned_core60_channel_inventory_hash_mismatch")
        if not is_core60_set:
            return unsupported("declared_core60_surface_channel_set_mismatch")

    try:
        values = np.asarray(sensor_signals, dtype=np.float64)
        positions = np.asarray(positions_m, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return unsupported("invalid_numeric_input")
    if values.ndim != 2 or values.shape != (len(labels), support.sample_count):
        return unsupported("signal_shape_or_sample_count_mismatch", extra={"signal_shape": list(values.shape)})
    if positions.shape != (len(labels), 3):
        return unsupported("position_count_or_shape_mismatch", extra={"position_shape": list(positions.shape)})
    if not np.isfinite(values).all():
        return unsupported("non_finite_input")
    if not np.isfinite(positions).all():
        return unsupported("non_finite_sensor_positions")
    if not math.isfinite(support.sample_rate_hz) or support.sample_rate_hz / 2.0 < alpha_band_hz[1]:
        return unsupported("sampling_frequency_does_not_support_alpha_band")

    try:
        distances = spherical_geodesic_distances_m(positions)
    except (TypeError, ValueError, FloatingPointError, OverflowError) as exc:
        return unsupported("invalid_sensor_geometry", extra={"geometry_error": str(exc)})
    position_hash = _position_hash(labels, positions)
    pairs = np.triu_indices(len(labels), k=1)
    pair_distances = distances[pairs]
    if not np.isfinite(pair_distances).all() or np.any(pair_distances <= 0.0):
        return unsupported("nonpositive_or_nonfinite_pair_distance", extra={"position_sha256": position_hash})

    # Average reference is applied to the exact declared surface. No channel
    # is dropped, reordered, interpolated, or substituted by this estimator.
    referenced = values - np.mean(values, axis=0, keepdims=True)
    coupling, coupling_details, coupling_error = _dwpli2_debiased_matrix(
        referenced,
        support.sample_rate_hz,
        band_hz=alpha_band_hz,
        segment_length_sec=segment_length_sec,
        segment_step_sec=segment_step_sec,
        min_segment_count=min_segment_count,
    )
    shared_details = {
        **base_details,
        "position_sha256": position_hash,
        "position_count": len(labels),
        "pair_count": int(len(pair_distances)),
        "pair_order": "upper_triangle_row_major_i_lt_j_in_channel_order",
        "distance_estimator_id": SCG_GEOMETRY_ESTIMATOR_ID,
        "distance_definition": "arccos(dot(unit_position_i,unit_position_j))*median_position_radius",
        "position_radial_origin": "declared_coordinate_frame_origin_not_estimated_from_surface",
        "pair_distance_min_m": float(np.min(pair_distances)),
        "pair_distance_median_m": float(np.median(pair_distances)),
        "pair_distance_max_m": float(np.max(pair_distances)),
        "reference_channel_labels": list(labels),
        "reference_set_sha256": _sha256_labels(labels, ordered=True),
        **coupling_details,
    }
    if coupling_error or coupling is None:
        return unsupported(coupling_error or "dwpli_estimation_failed", extra=shared_details)

    pair_couplings = coupling[pairs]
    if not np.isfinite(pair_couplings).all():
        return unsupported("non_finite_pairwise_dwpli", extra=shared_details)
    distance_ranks = rankdata(pair_distances, method="average")
    coupling_ranks = rankdata(pair_couplings, method="average")
    if np.ptp(distance_ranks) == 0.0:
        return unsupported("constant_pairwise_sensor_distance", extra=shared_details)
    if np.ptp(coupling_ranks) == 0.0:
        return unsupported("constant_pairwise_coupling", extra=shared_details)
    rho = float(np.corrcoef(distance_ranks, coupling_ranks)[0, 1])
    if not math.isfinite(rho):
        return unsupported("spatial_spearman_correlation_undefined", extra=shared_details)
    scg_value = float(-rho)

    pair_count = len(pair_distances)
    quartile_count = max(1, pair_count // 4)
    pair_index = np.arange(pair_count)
    ordering = np.lexsort((pair_index, pair_distances))
    near_indices = ordering[:quartile_count]
    far_indices = ordering[-quartile_count:]
    local_median = float(np.median(pair_couplings[near_indices]))
    distributed_median = float(np.median(pair_couplings[far_indices]))
    ldcb_value = local_median - distributed_median

    scg = _computed(
        candidate_id="D2", estimator_id=SCG_ESTIMATOR_ID, value=scg_value,
        support=support, observation_space="sensor_geometry_sensor_connectivity",
        details={
            **shared_details,
            "scalar_definition": "negative_spearman_rank_correlation_of_pair_distance_and_alpha_mean_debiased_squared_wPLI_dWPLI2",
            "spearman_rho_distance_vs_dwpli2": rho,
            "short_distance_positive_interpretation": True,
            "long_distance_positive_interpretation": False,
            "pairwise_coupling_min": float(np.min(pair_couplings)),
            "pairwise_coupling_median": float(np.median(pair_couplings)),
            "pairwise_coupling_max": float(np.max(pair_couplings)),
            "coupling_matrix": "returned_separately_in_D2SpatialGradientResult",
        },
    )
    ldcb = _computed(
        candidate_id="D2", estimator_id=LDCB_SIDECAR_ESTIMATOR_ID, value=ldcb_value,
        support=support, observation_space="sensor_geometry_sensor_connectivity",
        details={
            **shared_details,
            "sidecar_name": "Local–Distributed Coupling Balance",
            "scalar_definition": "median_alpha_mean_dWPLI2_nearest_distance_quartile_minus_median_alpha_mean_dWPLI2_farthest_distance_quartile",
            "quartile_rule": "floor(pair_count/4) nearest and farthest pairs; distance ties use upper-triangle order",
            "near_pair_count": quartile_count,
            "far_pair_count": quartile_count,
            "near_max_distance_m": float(np.max(pair_distances[near_indices])),
            "far_min_distance_m": float(np.min(pair_distances[far_indices])),
            "local_median_dwpli2": local_median,
            "distributed_median_dwpli2": distributed_median,
            "position_source": position_source,
            "space_claim": "sensor_observation_space_only",
        },
    )
    return D2SpatialGradientResult(scg, ldcb, np.column_stack(pairs), pair_distances, pair_couplings, coupling)


__all__ = [
    "SCG_ESTIMATOR_ID",
    "SCG_CONNECTIVITY_ESTIMATOR_ID",
    "SCG_GEOMETRY_ESTIMATOR_ID",
    "LDCB_SIDECAR_ESTIMATOR_ID",
    "SCG_MIN_CHANNELS",
    "SCG_STANDARD_MIN_CHANNELS",
    "SCG_ALPHA_BAND_HZ",
    "SCG_SEGMENT_LENGTH_SEC",
    "SCG_SEGMENT_STEP_SEC",
    "SCG_MIN_SEGMENT_COUNT",
    "D2SpatialGradientResult",
    "spherical_geodesic_distances_m",
    "standard_template_positions_m",
    "compute_d2_spatial_coupling_gradient",
]
