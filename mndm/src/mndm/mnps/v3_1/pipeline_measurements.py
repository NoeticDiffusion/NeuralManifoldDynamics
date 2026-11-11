"""Named, window-level v3.1 EEG measurements for the ordinary feature path.

These values are a separate measurement surface. They are not appended to the
legacy feature table and do not populate MNPS or dynamical coordinates. A
finite estimator result can be emitted with incomplete temporal/filter audit;
that uncertainty is carried as QC and does not imply qualification.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import signal

from ...features.eeg_entropy import compute_permutation_entropy
from .candidates import (
    MNPS31_CANDIDATES_BY_ID,
    MNPS31_STANDARD_CANDIDATE_IDS,
    MNPS31WindowSupport,
)
from .eeg_metrics import _lz78_phrase_count_binary
from .d2_position_templates import (
    DS004148_CORE60_CHANNELS,
    STANDARD_1005_POSITIONS_M,
    TEMPLATE_ID as STANDARD_1005_TEMPLATE_ID,
)
from .d2_spatial_gradient import compute_d2_spatial_coupling_gradient
from .realizations import CORE19_CHANNELS


RECORDING_LEVEL_EXPERIMENTAL_SCHEMA = "mndm.mnps_v3_1.recording_level_experimental.v1"
D2_PIPELINE_PREPROCESSING_ID = "mndm.pipeline_preprocessed_eeg_recording.v1"


PIPELINE_MEASUREMENT_SCHEMA = "mndm.mnps_v3_1.pipeline_measurements.v1"
PIPELINE_E2_SEED_DERIVATION_ID = "mndm.mnps_v3_1.pipeline_e2_seed_derivation.v1"
PIPELINE_E2_MASTER_SEED = 310930


def _label_hash(labels: Sequence[str], *, ordered: bool) -> str:
    normalized = tuple(str(label).strip().casefold() for label in labels)
    if not ordered:
        normalized = tuple(sorted(normalized))
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()


def compute_acquisition_qc(
    eeg_window: np.ndarray,
    *,
    sample_rate_hz: float,
    stage: str,
    window_start_sec: float = 0.0,
    amplitude_unit: str | None = None,
) -> dict[str, object]:
    """Compute descriptive spectral QC without changing the input signal.

    The line-noise ratio is median over channels of mean PSD in 48–52 Hz
    divided by mean PSD in 1–45 Hz. The broadband proxy is median channel
    mean PSD in 30–45 Hz. The function never applies a threshold or rejects a
    window. It returns explicit unsupported statuses when the frequency range
    is unavailable.
    """
    try:
        data = np.asarray(eeg_window, dtype=np.float64)
        sfreq = float(sample_rate_hz)
    except (TypeError, ValueError, OverflowError):
        data = np.empty((0, 0), dtype=np.float64)
        sfreq = math.nan
    if (
        data.ndim != 2
        or data.shape[0] < 1
        or data.shape[1] < 8
        or not np.isfinite(data).all()
        or not math.isfinite(sfreq)
        or sfreq <= 0.0
    ):
        reason = "invalid_or_nonfinite_qc_window"
        return {
            "line_noise_48_52_ratio": None,
            "line_noise_48_52_ratio_status": "unsupported",
            "line_noise_48_52_ratio_reason": reason,
            "broadband_noise_proxy": None,
            "broadband_noise_proxy_status": "unsupported",
            "broadband_noise_proxy_reason": reason,
            "qc_window_start_sec": float(window_start_sec),
            "qc_window_end_sec": None,
            "qc_sample_rate_hz": sfreq if math.isfinite(sfreq) else None,
            "qc_stage": str(stage),
        }
    nperseg = min(data.shape[1], max(8, int(round(4.0 * sfreq))))
    frequencies, psd = signal.welch(
        data,
        fs=sfreq,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
        axis=-1,
    )
    low = (frequencies >= 1.0) & (frequencies <= 45.0)
    line = (frequencies >= 48.0) & (frequencies <= 52.0)
    broad = (frequencies >= 30.0) & (frequencies <= 45.0)
    line_value: float | None = None
    line_status = "computed"
    line_reason: str | None = None
    if sfreq / 2.0 < 52.0 or not np.any(line) or not np.any(low):
        line_status, line_reason = "unsupported", "sampling_rate_or_window_does_not_resolve_48_52_and_1_45_hz"
    else:
        denominator = np.mean(psd[:, low], axis=1)
        numerator = np.mean(psd[:, line], axis=1)
        usable = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0.0)
        if not np.any(usable):
            line_status, line_reason = "unsupported", "line_noise_ratio_denominator_unavailable"
        else:
            line_value = float(np.median(numerator[usable] / denominator[usable]))
    broad_value: float | None = None
    broad_status = "computed"
    broad_reason: str | None = None
    if not np.any(broad):
        broad_status, broad_reason = "unsupported", "30_45_hz_band_unavailable"
    else:
        per_channel_broad = np.mean(psd[:, broad], axis=1)
        finite_broad = per_channel_broad[np.isfinite(per_channel_broad)]
        if not finite_broad.size:
            broad_status, broad_reason = "unsupported", "broadband_psd_unavailable"
        else:
            # Welch already returns the square of the input amplitude unit / Hz.
            broad_value = float(np.median(finite_broad))
    return {
        "line_noise_48_52_ratio": line_value,
        "line_noise_48_52_ratio_status": line_status,
        "line_noise_48_52_ratio_reason": line_reason,
        "broadband_noise_proxy": broad_value,
        "broadband_noise_proxy_status": broad_status,
        "broadband_noise_proxy_reason": broad_reason,
        "broadband_noise_proxy_units": "microvolt_squared_per_hz" if str(amplitude_unit or "").strip().casefold() in {"microvolt", "uv", "µv", "μv"} else "input_unit_squared_per_hz",
        "line_noise_ratio_formula": "median_channel_mean_PSD_48_52_divided_by_mean_PSD_1_45",
        "broadband_noise_proxy_formula": "median_channel_mean_PSD_30_45",
        "welch_nperseg": int(nperseg),
        "qc_window_start_sec": float(window_start_sec),
        "qc_window_end_sec": float(window_start_sec + data.shape[1] / sfreq),
        "qc_sample_rate_hz": sfreq,
        "qc_stage": str(stage),
        "amplitude_unit": amplitude_unit,
    }


def _numeric_or_none(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _row(
    candidate_id: str,
    estimator_id: str,
    value: object,
    *,
    units: str,
    reason: str | None = None,
    details: Mapping[str, object] | None = None,
) -> dict[str, object]:
    candidate = MNPS31_CANDIDATES_BY_ID[candidate_id]
    finite_value: object = value
    if isinstance(value, np.ndarray):
        finite_value = value.tolist()
    if isinstance(finite_value, tuple):
        finite_value = list(finite_value)
    if isinstance(finite_value, list):
        is_finite = bool(finite_value) and all(_numeric_or_none(item) is not None for item in finite_value)
    else:
        is_finite = _numeric_or_none(finite_value) is not None
    computed = reason is None and is_finite
    return {
        "construct_id": candidate.public_id,
        "construct_name": candidate.construct_name,
        "short_name": candidate.short_name,
        "historical_slot": candidate.candidate_id,
        "historical_family": candidate.family,
        "standard_availability": candidate.standard_availability,
        "candidate_id": candidate.candidate_id,
        "estimator_id": estimator_id,
        "value": finite_value if computed else None,
        "units": units,
        "computed": computed,
        "valid": computed,
        "qualified": False,
        "status": "computed_exploratory" if computed else "unsupported",
        "reason": reason if reason is not None else (None if computed else "non_finite_estimate"),
        "details": dict(details or {}),
        "coordinate_assignment": None,
        "claim_status": "no_biological_claim",
    }


def _m1_window_local_filter_support_qc(
    *,
    sample_count: int,
    sample_rate_hz: float,
    window_start_sec: float,
    window_end_sec: float,
    upstream_support_qc: Mapping[str, object],
) -> dict[str, object]:
    """Describe M1's actual window-local temporal dependency without qualifying it.

    The SOS forward/backward IIR and FFT Hilbert transform have no established
    finite edge-safe interval in this implementation. This records that all
    samples of the declared window are present while keeping boundary fidelity
    unknown and upstream preprocessing support separate.
    """
    return {
        "schema": "mndm.mnps_v3_1.m1_window_local_filter_support_qc.v2",
        "filter_id": "4th_order_Butterworth_8_13_Hz_sosfiltfilt",
        "filter_implementation": "scipy.signal.sosfiltfilt",
        "filter_order": 4,
        "filter_band_hz": [8.0, 13.0],
        "filter_context": "entire_current_window",
        "filter_boundary_handling": "endpoint_dependent_forward_backward_IIR_no_explicit_padding",
        "padtype": None,
        "padlen_samples": 0,
        "analytic_transform": "scipy.signal.hilbert_fft",
        "analytic_transform_dependency": "full_supplied_window_fft_with_periodic_boundary_assumption",
        "input_window_start_sec": float(window_start_sec),
        "input_window_end_sec": float(window_end_sec),
        "input_window_duration_sec": float(window_end_sec - window_start_sec),
        "input_sample_rate_hz": float(sample_rate_hz),
        "input_sample_count": int(sample_count),
        "input_samples_finite_and_available": int(sample_count),
        "external_left_context_samples_used": 0,
        "external_right_context_samples_used": 0,
        "edge_samples_excluded": 0,
        "full_declared_window_supplied_to_estimator": True,
        # Retained for readers of the v1 field; the explicit scope below
        # prevents interpreting this as continuous-series edge validation.
        "computational_support_complete_for_current_window": True,
        "computational_support_scope": (
            "all samples of the declared window were supplied; this does not "
            "establish edge fidelity relative to a continuous recording"
        ),
        "exact_finite_edge_safe_support_established": False,
        "candidate_filter_edge_support_valid": None,
        "candidate_filter_edge_support_status": "unknown_unassessed",
        "upstream_preprocessing_filter_support_valid": upstream_support_qc.get(
            "effective_filter_support_valid"
        ),
        "upstream_support_qc": dict(upstream_support_qc),
        "edge_effects_assessed_or_corrected": False,
        "status": "full_window_computed; edge_fidelity_unassessed",
        "warnings": [
            "forward_backward_iir_has_no_established_finite_edge_safe_interval",
            "fft_hilbert_uses_the_full_window_fft_boundary_assumption",
            "no_external_context_or_edge_exclusion",
            "alpha_filter_edge_effects_unassessed",
        ],
    }


def _unsupported_rows(reason: str, details: Mapping[str, object] | None = None) -> list[dict[str, object]]:
    estimator_ids = {
        "M1": "PCV.pipeline_native_window_alpha_phase_order_parameter_sd.v1",
        "M2": "TP.pipeline_native_window_lag1_decorrelation.v1",
        "M3": "AR.pipeline_native_window_adjacent_block_rms_change.v1",
        "D1": "NIE.pipeline_native_window_abs_pearson_weighted_efficiency.v1",
        "D3": "RD.pipeline_native_window_centered_log_psd_dispersion.v1",
        "E1": "OI.pipeline_native_window_permutation_entropy_order5_delay1.v1",
        "E2": "SC.pipeline_native_window_lz78_exact_multiset_ratio.v1",
        "E3": "ED.pipeline_native_window_covariance_participation_ratio.v1",
    }
    units = {
        "M1": "1", "M2": "1", "M3": "1", "D1": "1",
        "D3": "centered_log10_power", "E1": "normalized_entropy",
        "E2": "dimensionless_ratio", "E3": "sensor_directions",
    }
    return [
        _row(candidate_id, estimator_ids[candidate_id], None, units=units[candidate_id], reason=reason, details=details)
        for candidate_id in MNPS31_STANDARD_CANDIDATE_IDS
    ]


def _temporal_persistence_row(
    values: np.ndarray,
    *,
    sample_rate_hz: float,
    n_channels: int,
    base_details: Mapping[str, object],
) -> dict[str, object]:
    """Compute the production M2 formula for one already-validated EEG window."""
    tp_values: list[float] = []
    tp_failures: list[str] = []
    try:
        for channel in values:
            scale = float(np.max(np.abs(channel)))
            if scale == 0.0:
                tp_failures.append("zero_variance_channel")
                continue
            centered = channel / scale
            centered = centered - float(np.mean(centered))
            energy = float(np.dot(centered, centered))
            if not math.isfinite(energy) or energy <= 0.0:
                tp_failures.append("zero_variance_channel")
                continue
            rho = float(np.dot(centered[:-1], centered[1:]) / energy)
            if not math.isfinite(rho) or abs(rho) > 1.0 + 1e-12:
                tp_failures.append("undefined_lag1_autocorrelation")
                continue
            tp_values.append(1.0 - min(1.0, max(-1.0, rho)))
        tp_value = float(np.median(np.asarray(tp_values, dtype=np.float64))) if tp_values else None
        return _row(
            "M2",
            "TP.pipeline_native_window_lag1_decorrelation.v1",
            tp_value,
            units="1",
            reason=None if tp_values else "no_channel_has_defined_lag1_autocorrelation",
            details={
                **dict(base_details),
                "definition": "median_channel(1 - mean_centered_lag1_autocorrelation)",
                "lag_seconds": 1.0 / sample_rate_hz,
                "valid_channel_count": len(tp_values),
                "valid_channel_coverage": len(tp_values) / n_channels if n_channels else 0.0,
                "invalid_channel_reasons": tp_failures,
            },
        )
    except Exception as exc:
        return _row(
            "M2",
            "TP.pipeline_native_window_lag1_decorrelation.v1",
            None,
            units="1",
            reason="candidate_estimator_exception",
            details={**dict(base_details), "exception_type": type(exc).__name__},
        )


def compute_pipeline_m2_temporal_measurement(
    eeg_window: np.ndarray,
    channel_names: Sequence[str],
    *,
    sample_rate_hz: float,
    window_start_sec: float,
    window_end_sec: float,
    segment_id: str,
    input_branch: Mapping[str, object],
    support_qc: Mapping[str, object],
    acquisition_qc: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Compute only M2 on a separately defined temporal realization window."""
    try:
        values = np.asarray(eeg_window, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        values = np.empty((0, 0), dtype=np.float64)
    labels = tuple(str(label).strip() for label in channel_names)
    sfreq = float(sample_rate_hz)
    start_sec = float(window_start_sec)
    end_sec = float(window_end_sec)
    segment = str(segment_id or "")
    n_channels = int(values.shape[0]) if values.ndim == 2 else 0
    n_samples = int(values.shape[1]) if values.ndim == 2 else 0
    label_keys = tuple(label.casefold() for label in labels)
    input_error = None
    if values.ndim != 2:
        input_error = "signals_must_be_channels_by_samples"
    elif not labels or len(labels) != n_channels or any(not label for label in labels):
        input_error = "channel_label_count_mismatch_or_empty"
    elif len(set(label_keys)) != len(label_keys):
        input_error = "duplicate_channel_labels"
    elif n_samples < 2:
        input_error = "insufficient_samples"
    elif not math.isfinite(sfreq) or sfreq <= 0.0:
        input_error = "invalid_sampling_frequency"
    elif not math.isfinite(start_sec) or not math.isfinite(end_sec) or end_sec <= start_sec:
        input_error = "invalid_window_bounds"
    elif not np.isclose(end_sec - start_sec, n_samples / sfreq, rtol=0.0, atol=max(1e-6, 1.0 / sfreq)):
        input_error = "window_time_sample_count_mismatch"

    channel_order_hash = _label_hash(labels, ordered=True) if labels else ""
    channel_set_hash = _label_hash(labels, ordered=False) if labels else ""
    support_qc = dict(support_qc)
    support_status = "complete" if support_qc and all(value is not None for value in support_qc.values()) else "incomplete"
    if any(value is False for value in support_qc.values()):
        support_status = "known_limitations"
    window = {
        "start_sec": start_sec,
        "end_sec": end_sec,
        "sample_rate_hz": sfreq,
        "sample_count": n_samples,
        "segment_id": segment,
    }
    base_details: dict[str, object] = {
        "channel_count": n_channels,
        "channel_order_sha256": channel_order_hash,
        "channel_set_sha256": channel_set_hash,
        "input_reference_mode": input_branch.get("reference_mode"),
        "reference_set_sha256": input_branch.get("reference_set_sha256"),
        "input_filter_band_hz": input_branch.get("filter_band_hz"),
        "support_audit_status": support_status,
        "support_qc": support_qc,
        "acquisition_qc": dict(acquisition_qc or {}),
        "measurement_window": window,
        "realization": {
            "surface_id": "native",
            "surface_kind": "native",
            "channel_labels": list(labels),
            "channel_count": n_channels,
            "channel_order_sha256": channel_order_hash,
            "channel_set_sha256": channel_set_hash,
            "reference_mode": input_branch.get("reference_mode"),
            "reference_set_sha256": input_branch.get("reference_set_sha256"),
            "input_branch_id": input_branch.get("branch_id"),
            "filter_band_hz": input_branch.get("filter_band_hz"),
        },
        "qualification": "not_assessed",
    }
    if input_error:
        return _row(
            "M2", "TP.pipeline_native_window_lag1_decorrelation.v1", None,
            units="1", reason=input_error, details=base_details,
        )
    if not np.isfinite(values).all():
        return _row(
            "M2", "TP.pipeline_native_window_lag1_decorrelation.v1", None,
            units="1", reason="non_finite_input_samples", details=base_details,
        )
    return _temporal_persistence_row(
        values,
        sample_rate_hz=sfreq,
        n_channels=n_channels,
        base_details=base_details,
    )


def _recording_input_sha256(values: np.ndarray, labels: Sequence[str], sample_rate_hz: float) -> str:
    """Hash the exact preprocessed signal matrix consumed by recording metrics."""
    canonical = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(json.dumps({
        "schema": "mndm.preprocessed_signal_matrix_hash.v1",
        "shape": list(canonical.shape),
        "dtype": "little_endian_float64",
        "sample_rate_hz": float(sample_rate_hz).hex(),
        "ordered_channel_labels": [str(label).strip().casefold() for label in labels],
    }, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def compute_pipeline_recording_level_experimental_measurements(
    eeg_recording: np.ndarray,
    channel_names: Sequence[str],
    *,
    sample_rate_hz: float,
    support_qc: Mapping[str, object],
    source_identity: str,
    dataset_id: str | None,
    input_branch: Mapping[str, object],
    preprocessing_metadata: Mapping[str, object] | None = None,
    signal_units: str = "unknown",
    include_core19: bool = True,
    include_ds004148_core60: bool = True,
) -> dict[str, object]:
    """Compute experimental SCG/LDCB once over a supported recording.

    This stays outside the window chart and MNJ coordinates. The native
    template-supported set is preserved as its own realization; fixed Core19
    and the ds004148 Core60 set are emitted separately when present. Channels
    without frozen template coordinates, duplicate labels, and nonfinite
    channels are reported in surface QC and omitted only from the affected
    realization.
    """
    try:
        values = np.asarray(eeg_recording, dtype=np.float64)
        sfreq = float(sample_rate_hz)
    except (TypeError, ValueError, OverflowError):
        values = np.empty((0, 0), dtype=np.float64)
        sfreq = math.nan
    labels = tuple(str(label).strip() for label in channel_names) if not isinstance(channel_names, (str, bytes)) else ()
    n_channels = values.shape[0] if values.ndim == 2 else 0
    n_samples = values.shape[1] if values.ndim == 2 else 0
    input_shape_valid = values.ndim == 2 and len(labels) == n_channels
    if not input_shape_valid:
        values = np.empty((0, 0), dtype=np.float64)
        labels = ()
        n_channels = 0
        n_samples = 0
    duration = n_samples / sfreq if math.isfinite(sfreq) and sfreq > 0.0 else 0.0
    segment_id = str(source_identity or "")
    support_values = dict(support_qc)
    support = MNPS31WindowSupport(
        window_start_sec=0.0,
        window_end_sec=float(duration),
        sample_rate_hz=sfreq,
        sample_count=int(n_samples),
        segment_id=segment_id,
        source_is_continuous=support_values.get("source_is_continuous"),
        timestamps_uniform=support_values.get("timestamps_uniform"),
        gap_free=support_values.get("gap_free"),
        bad_interval_free=support_values.get("bad_interval_free"),
        within_recording_bounds=support_values.get("within_recording_bounds"),
        # Preserve missing upstream edge provenance as unknown; the pipeline
        # mode below permits computation while keeping qualification separate.
        effective_filter_support_valid=support_values.get("effective_filter_support_valid"),
    )
    content_hash = _recording_input_sha256(values, labels, sfreq) if np.isfinite(sfreq) and sfreq > 0 else "0" * 64

    occurrence_indices: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        occurrence_indices.setdefault(label.casefold(), []).append(index)
    duplicate_source_labels = sorted(
        labels[index]
        for indices in occurrence_indices.values() if len(indices) > 1
        for index in indices[:1]
    )
    finite_by_channel = (
        np.isfinite(values).all(axis=1)
        if values.ndim == 2 and values.shape[1] > 0
        else np.zeros((n_channels,), dtype=bool)
    )

    template_labels = set(STANDARD_1005_POSITIONS_M)
    core19_labels = tuple(CORE19_CHANNELS)
    core60_labels = tuple(DS004148_CORE60_CHANNELS)
    requested_specs: list[tuple[str, tuple[str, ...]]] = []
    core19_overlap = set(label.casefold() for label in labels).intersection(label.casefold() for label in core19_labels)
    if include_core19 and core19_overlap:
        requested_specs.append(("core19", core19_labels))
    if include_ds004148_core60 and str(dataset_id or "").casefold() == "ds004148":
        requested_specs.append(("core60", core60_labels))
    requested_specs.append(("native", labels))

    surfaces: list[dict[str, object]] = []
    for requested_surface_id, requested_labels in requested_specs:
        effective_labels: list[str] = []
        effective_indices: list[int] = []
        missing_labels: list[str] = []
        duplicate_labels: list[str] = []
        nonfinite_labels: list[str] = []
        unmapped_position_labels: list[str] = []
        for requested_label in requested_labels:
            key = requested_label.casefold()
            matches = occurrence_indices.get(key, [])
            if not matches:
                missing_labels.append(requested_label)
            elif len(matches) > 1:
                duplicate_labels.append(requested_label)
            elif not finite_by_channel[matches[0]]:
                nonfinite_labels.append(requested_label)
            elif key not in template_labels:
                unmapped_position_labels.append(requested_label)
            else:
                effective_labels.append(requested_label)
                effective_indices.append(matches[0])

        complete = (
            len(effective_labels) == len(requested_labels)
            and {label.casefold() for label in effective_labels} == {label.casefold() for label in requested_labels}
        )
        if requested_surface_id == "core19":
            actual_surface_id = "core19" if complete else f"core19_available_{len(effective_labels)}"
        elif requested_surface_id == "core60":
            actual_surface_id = "core60" if complete else f"core60_available_{len(effective_labels)}"
        else:
            actual_surface_id = "native" if complete else f"native_template_available_{len(effective_labels)}"

        warnings: list[str] = []
        if missing_labels:
            warnings.append("requested_surface_missing_channels")
        if duplicate_labels:
            warnings.append("requested_surface_duplicate_source_labels")
        if nonfinite_labels:
            warnings.append("requested_surface_omits_nonfinite_channels")
        if unmapped_position_labels:
            warnings.append("requested_surface_has_channels_without_frozen_template_positions")
        if len(effective_labels) < 19:
            warnings.append("below_core19_channel_count")
        unknown_support_fields = sorted(
            key for key in (
                "source_is_continuous",
                "timestamps_uniform",
                "gap_free",
                "bad_interval_free",
                "within_recording_bounds",
                "effective_filter_support_valid",
            )
            if support_values.get(key) is None
        )
        if unknown_support_fields:
            warnings.append("recording_support_audit_incomplete")
        surface_qc = {
            "requested_channel_labels": list(requested_labels),
            "effective_channel_labels": effective_labels,
            "missing_requested_channel_labels": missing_labels,
            "duplicate_requested_channel_labels": duplicate_labels,
            "omitted_nonfinite_channel_labels": nonfinite_labels,
            "channels_without_template_positions": unmapped_position_labels,
            "duplicate_source_channel_labels": duplicate_source_labels,
            "position_source": "frozen_standard_template",
            "position_template_id": STANDARD_1005_TEMPLATE_ID,
            "input_shape_valid": input_shape_valid,
            "support_audit_unknown_fields": unknown_support_fields,
            "warnings": sorted(set(warnings)),
        }
        positions = np.asarray(
            [STANDARD_1005_POSITIONS_M[label.casefold()] for label in effective_labels],
            dtype=np.float64,
        ).reshape((-1, 3))
        signal_values = values[effective_indices, :] if effective_indices and values.ndim == 2 else np.empty((0, n_samples), dtype=np.float64)
        reference_order_hash = _label_hash(effective_labels, ordered=True)
        reference_id = f"instantaneous-average-over-exact-sensor-set.v1:{reference_order_hash}"
        measured = compute_d2_spatial_coupling_gradient(
            signal_values,
            effective_labels,
            positions,
            support,
            surface_id=actual_surface_id,
            requested_surface_id=requested_surface_id,
            requested_channel_names=requested_labels,
            surface_qc=surface_qc,
            source_id=segment_id,
            source_content_sha256=content_hash,
            preprocessing_id=D2_PIPELINE_PREPROCESSING_ID,
            reference_mode="average_over_exact_surface",
            reference_id=reference_id,
            position_source="standard_template",
            coordinate_frame="mri",
            position_template_id=STANDARD_1005_TEMPLATE_ID,
            allow_unknown_support_for_computation=True,
            signal_units=signal_units,
        )
        surfaces.append({
            "requested_surface_id": requested_surface_id,
            "surface_id": actual_surface_id,
            "aggregation_level": "recording",
            "support_window_sec": float(duration),
            "step_sec": None,
            "effective_sample_count": int(n_samples),
            "effective_channel_labels": effective_labels,
            "source_content_sha256": content_hash,
            "preprocessing_id": D2_PIPELINE_PREPROCESSING_ID,
            "input_branch": dict(input_branch),
            "preprocessing_metadata": dict(preprocessing_metadata or {}),
            "support_qc": support_values,
            "surface_qc": surface_qc,
            "measurements": [measured.scg.to_dict(), measured.ldcb_sidecar.to_dict()],
        })

    computed_scg_count = sum(
        bool(row.get("computed"))
        for surface in surfaces
        for row in surface["measurements"]
        if isinstance(row, Mapping) and row.get("estimator_id") == "spatial_dwpli_distance_spearman_alpha.v1"
    )
    status = "computed" if computed_scg_count == len(surfaces) and surfaces else (
        "partial" if computed_scg_count else "unsupported"
    )
    return {
        "schema": RECORDING_LEVEL_EXPERIMENTAL_SCHEMA,
        "status": status,
        "qualified": False,
        "coordinate_assignment": None,
        "historical_slot": "D2",
        "construct_id": "spatial_coupling_gradient",
        "construct_name": "Spatial Coupling Gradient",
        "short_name": "SCG",
        "aggregation_level": "recording",
        "support_window_sec": float(duration),
        "step_sec": None,
        "surface_count": len(surfaces),
        "computed_scg_surface_count": int(computed_scg_count),
        "source_identity": str(source_identity),
        "source_content_hash_basis": "preprocessed_float64_signal_array_with_shape_sample_rate_and_ordered_labels",
        "input_branch": dict(input_branch),
        "support_qc": support_values,
        "surfaces": surfaces,
        "claim_status": "sensor_observation_space_no_biological_claim",
    }


def _sequence_complexity(
    values: np.ndarray,
    labels: Sequence[str],
    *,
    source_identity: str,
    segment_id: str,
    start_sec: float,
    end_sec: float,
    reference_count: int,
) -> tuple[float | None, str | None, dict[str, object]]:
    if isinstance(reference_count, bool) or not isinstance(reference_count, (int, np.integer)) or reference_count < 8:
        return None, "reference_count_must_be_integer_at_least_8", {}
    channel_ratios: list[float] = []
    channel_rows: list[dict[str, object]] = []
    for row, label in zip(values, labels, strict=True):
        threshold = float(np.median(row))
        symbols = np.asarray(row > threshold, dtype=np.uint8)
        zero_count = int(symbols.size - np.count_nonzero(symbols))
        one_count = int(np.count_nonzero(symbols))
        if not zero_count or not one_count:
            return None, "sequence_has_fewer_than_two_symbols", {"failed_channel": label}
        seed_material = "\0".join((
            PIPELINE_E2_SEED_DERIVATION_ID,
            str(PIPELINE_E2_MASTER_SEED),
            source_identity,
            segment_id,
            float(start_sec).hex(),
            float(end_sec).hex(),
            str(label).strip().casefold(),
            str(reference_count),
        )).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big", signed=False)
        rng = np.random.Generator(np.random.PCG64(seed))
        observed = _lz78_phrase_count_binary(symbols)
        reference_counts = [
            _lz78_phrase_count_binary(symbols[rng.permutation(symbols.size)])
            for _ in range(reference_count)
        ]
        reference_median = float(np.median(np.asarray(reference_counts, dtype=np.float64)))
        if not math.isfinite(reference_median) or reference_median <= 0:
            return None, "undefined_exact_multiset_reference", {"failed_channel": label}
        ratio = float(observed / reference_median)
        channel_ratios.append(ratio)
        channel_rows.append({
            "channel_label": str(label),
            "threshold": threshold,
            "tie_count": int(np.count_nonzero(row == threshold)),
            "symbol_counts": {"0": zero_count, "1": one_count},
            "observed_phrase_count": observed,
            "reference_phrase_count_median": reference_median,
            "ratio": ratio,
            "derived_seed": seed,
        })
    return float(np.median(np.asarray(channel_ratios, dtype=np.float64))), None, {
        "symbolization": "per_channel_arithmetic_median_strictly_greater_is_1_ties_are_0",
        "lz_variant": "LZ78_incremental_dictionary_parsing",
        "reference_generator": "PCG64_permutations_of_exact_binary_symbol_multiset",
        "reference_count_per_channel": int(reference_count),
        "reference_summary": "median_reference_phrase_count",
        "master_seed": PIPELINE_E2_MASTER_SEED,
        "seed_derivation_id": PIPELINE_E2_SEED_DERIVATION_ID,
        "aggregation": "median_of_supported_channel_ratios",
        "per_channel": channel_rows,
    }


def compute_pipeline_window_measurements(
    eeg_window: np.ndarray,
    channel_names: Sequence[str],
    *,
    sample_rate_hz: float,
    window_start_sec: float,
    window_end_sec: float,
    segment_id: str,
    source_identity: str,
    input_branch: Mapping[str, object],
    support_qc: Mapping[str, object],
    acquisition_qc: Mapping[str, object] | None = None,
    sequence_complexity_reference_count: int = 32,
    activity_block_sec: float = 2.0,
    metric_timing_callback: Callable[[str, float], None] | None = None,
) -> dict[str, object]:
    """Compute the eight standard-availability candidates on one actual EEG window.

    The channel realization is native and stays exactly as presented to the
    standard EEG feature extractor. No channel is reordered, filled, selected,
    or silently removed. Support audit gaps are emitted as QC; they do not
    suppress a mathematically computable value.
    """
    try:
        values = np.asarray(eeg_window, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        values = np.empty((0, 0), dtype=np.float64)
    labels = tuple(str(label).strip() for label in channel_names)
    start_sec = float(window_start_sec)
    end_sec = float(window_end_sec)
    sfreq = float(sample_rate_hz)
    segment = str(segment_id or "")
    n_channels = int(values.shape[0]) if values.ndim == 2 else 0
    n_samples = int(values.shape[1]) if values.ndim == 2 else 0
    label_keys = tuple(label.casefold() for label in labels)
    input_error = None
    if values.ndim != 2:
        input_error = "signals_must_be_channels_by_samples"
    elif not labels or len(labels) != n_channels or any(not label for label in labels):
        input_error = "channel_label_count_mismatch_or_empty"
    elif len(set(label_keys)) != len(label_keys):
        input_error = "duplicate_channel_labels"
    elif n_samples < 2:
        input_error = "insufficient_samples"
    elif not np.isfinite(sfreq) or sfreq <= 0:
        input_error = "invalid_sampling_frequency"
    elif not np.isfinite(start_sec) or not np.isfinite(end_sec) or end_sec <= start_sec:
        input_error = "invalid_window_bounds"
    elif not np.isclose(end_sec - start_sec, n_samples / sfreq, rtol=0.0, atol=max(1e-6, 1.0 / sfreq)):
        input_error = "window_time_sample_count_mismatch"

    channel_order_hash = _label_hash(labels, ordered=True) if labels else ""
    channel_set_hash = _label_hash(labels, ordered=False) if labels else ""
    window = {
        "start_sec": start_sec,
        "end_sec": end_sec,
        "sample_rate_hz": sfreq,
        "sample_count": n_samples,
        "segment_id": segment,
    }
    realization = {
        "surface_id": "native",
        "surface_kind": "native",
        "channel_labels": list(labels),
        "channel_count": n_channels,
        "channel_order_sha256": channel_order_hash,
        "channel_set_sha256": channel_set_hash,
        "reference_mode": input_branch.get("reference_mode"),
        "reference_set_sha256": input_branch.get("reference_set_sha256"),
        "input_branch_id": input_branch.get("branch_id"),
        "filter_band_hz": input_branch.get("filter_band_hz"),
    }
    support_qc = dict(support_qc)
    support_status = "complete" if support_qc and all(value is not None for value in support_qc.values()) else "incomplete"
    if any(value is False for value in support_qc.values()):
        support_status = "known_limitations"
    if input_error:
        rows = _unsupported_rows(input_error)
        all_finite = False
    elif not np.isfinite(values).all():
        rows = _unsupported_rows("non_finite_input_samples")
        all_finite = False
    else:
        rows = []
        all_finite = True

    def add(candidate_id: str, estimator: str, value: object, units: str, *, reason: str | None = None, details: Mapping[str, object] | None = None) -> None:
        rows.append(_row(candidate_id, estimator, value, units=units, reason=reason, details=details))

    def metric_timer_start() -> float | None:
        return time.perf_counter() if metric_timing_callback is not None else None

    def metric_timer_finish(candidate: str, started: float | None) -> None:
        if metric_timing_callback is None or started is None:
            return
        try:
            metric_timing_callback(candidate, time.perf_counter() - started)
        except Exception:
            # Profiling must never change or suppress a measurement result.
            pass

    if all_finite:
        base_details = {
            "channel_count": n_channels,
            "channel_order_sha256": channel_order_hash,
            "channel_set_sha256": channel_set_hash,
            "input_reference_mode": input_branch.get("reference_mode"),
            "reference_set_sha256": input_branch.get("reference_set_sha256"),
            "input_filter_band_hz": input_branch.get("filter_band_hz"),
            "support_audit_status": support_status,
            "support_qc": support_qc,
            "acquisition_qc": dict(acquisition_qc or {}),
            "measurement_window": window,
            "qualification": "not_assessed",
        }

        # Phase Coordination Variability: window-local alpha band and analytic phase.
        metric_started = metric_timer_start()
        try:
            pcv_reason = None
            pcv_value = None
            pcv_details: dict[str, object] = {}
            if n_channels < 2:
                pcv_reason = "at_least_two_channels_required"
            elif sfreq <= 26.0:
                pcv_reason = "sampling_frequency_does_not_support_8_13_hz_band"
            else:
                try:
                    sos = signal.butter(4, [8.0, 13.0], btype="bandpass", fs=sfreq, output="sos")
                    alpha = signal.sosfiltfilt(sos, values, axis=-1, padtype=None, padlen=0)
                    analytic = signal.hilbert(alpha, axis=-1)
                    amplitude = np.abs(analytic)
                    if not np.isfinite(amplitude).all() or np.any(amplitude == 0.0):
                        pcv_reason = "invalid_or_zero_analytic_amplitude"
                    else:
                        phase_order = np.abs(np.mean(analytic / amplitude, axis=0))
                        pcv_value = float(np.std(phase_order, ddof=0))
                        pcv_details = {
                            "phase_estimator": "4th_order_Butterworth_8_13_Hz_sosfiltfilt_padlen0_then_Hilbert",
                            "aggregation": "population_sd_over_window_samples_of_native_sensor_phase_order_parameter",
                            "filter_context": "window_local",
                            "filter_support_qc_scope": "upstream_support_is_separate; candidate_local_filter_dependency_is_recorded_below",
                            "filter_support_qc": _m1_window_local_filter_support_qc(
                                sample_count=n_samples,
                                sample_rate_hz=sfreq,
                                window_start_sec=start_sec,
                                window_end_sec=end_sec,
                                upstream_support_qc=support_qc,
                            ),
                        }
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    pcv_reason = "alpha_phase_estimation_failed"
            add("M1", "PCV.pipeline_native_window_alpha_phase_order_parameter_sd.v1", pcv_value,
                "1", reason=pcv_reason, details={**base_details, **pcv_details})

        except Exception as exc:
            rows[:] = [row for row in rows if row.get("candidate_id") != "M1"]
            add("M1", "PCV.pipeline_native_window_alpha_phase_order_parameter_sd.v1", None, "1",
                reason="candidate_estimator_exception",
                details={**base_details, "exception_type": type(exc).__name__})
        finally:
            metric_timer_finish("PCV", metric_started)

        # Temporal Persistence: median channel-wise 1 - lag-one autocorrelation.
        metric_started = metric_timer_start()
        rows.append(_temporal_persistence_row(
            values,
            sample_rate_hz=sfreq,
            n_channels=n_channels,
            base_details=base_details,
        ))
        metric_timer_finish("TP_main_window", metric_started)

        # Activity Reconfiguration: native channel RMS vectors in equal adjacent blocks.
        metric_started = metric_timer_start()
        try:
            ar_value = None
            ar_reason = None
            ar_details: dict[str, object] = {**base_details}
            block_samples = int(round(float(activity_block_sec) * sfreq))
            if block_samples < 1 or n_samples % block_samples != 0 or n_samples // block_samples < 2:
                ar_reason = "window_must_contain_at_least_two_equal_activity_blocks"
            else:
                activity = np.sqrt(np.mean(values.reshape(n_channels, -1, block_samples) ** 2, axis=2)).T
                changes: list[float] = []
                for left, right in zip(activity[:-1], activity[1:], strict=True):
                    left_scale = float(np.max(np.abs(left)))
                    right_scale = float(np.max(np.abs(right)))
                    if left_scale == 0.0 or right_scale == 0.0:
                        ar_reason = "zero_activity_block_vector"
                        break
                    left_norm = left / left_scale
                    right_norm = right / right_scale
                    left_centered = left_norm - np.mean(left_norm)
                    right_centered = right_norm - np.mean(right_norm)
                    denom = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
                    if denom <= 0.0:
                        ar_reason = "constant_activity_block_vector"
                        break
                    correlation = float(np.dot(left_centered, right_centered) / denom)
                    changes.append(1.0 - min(1.0, max(-1.0, correlation)))
                if ar_reason is None and changes:
                    ar_value = changes
                    ar_details.update({"vector_construction": "native_channel_RMS_per_equal_contiguous_block",
                                       "block_duration_sec": block_samples / sfreq,
                                       "block_count": int(activity.shape[0]),
                                       "aggregation": "none_one_value_per_adjacent_block_pair"})
                elif ar_reason is None:
                    ar_reason = "no_adjacent_block_pairs"
            add("M3", "AR.pipeline_native_window_adjacent_block_rms_change.v1", ar_value,
                "1", reason=ar_reason, details=ar_details)

        except Exception as exc:
            rows[:] = [row for row in rows if row.get("candidate_id") != "M3"]
            add("M3", "AR.pipeline_native_window_adjacent_block_rms_change.v1", None, "1",
                reason="candidate_estimator_exception",
                details={**base_details, "exception_type": type(exc).__name__})
        finally:
            metric_timer_finish("AR", metric_started)

        # Network Integration Efficiency: absolute zero-lag Pearson weights, no threshold.
        metric_started = metric_timer_start()
        try:
            nie_value = None
            nie_reason = None
            nie_details: dict[str, object] = {**base_details}
            if n_channels < 2:
                nie_reason = "at_least_two_channels_required"
            else:
                with np.errstate(invalid="ignore", divide="ignore"):
                    correlation = np.corrcoef(values)
                if not np.isfinite(correlation).all():
                    nie_reason = "constant_or_nonfinite_channel_coupling"
                else:
                    weights = np.abs(correlation)
                    np.fill_diagonal(weights, 0.0)
                    distances = np.full(weights.shape, np.inf, dtype=np.float64)
                    np.fill_diagonal(distances, 0.0)
                    edges = weights > 0.0
                    distances[edges] = 1.0 / weights[edges]
                    for pivot in range(n_channels):
                        distances = np.minimum(distances, distances[:, pivot, None] + distances[None, pivot, :])
                    inverse = np.zeros_like(distances)
                    np.divide(1.0, distances, out=inverse, where=np.isfinite(distances) & (distances > 0.0))
                    nie_value = float(np.sum(inverse) / (n_channels * (n_channels - 1)))
                    nie_details.update({"edge_estimator": "absolute_zero_lag_Pearson_r",
                                        "threshold": "none",
                                        "weight_to_length": "1/weight",
                                        "disconnected_pair_value": 0.0,
                                        "node_order_sha256": channel_order_hash,
                                        "graph_claim": "sensor_graph_only"})
            add("D1", "NIE.pipeline_native_window_abs_pearson_weighted_efficiency.v1", nie_value,
                "1", reason=nie_reason, details=nie_details)

        except Exception as exc:
            rows[:] = [row for row in rows if row.get("candidate_id") != "D1"]
            add("D1", "NIE.pipeline_native_window_abs_pearson_weighted_efficiency.v1", None, "1",
                reason="candidate_estimator_exception",
                details={**base_details, "exception_type": type(exc).__name__})
        finally:
            metric_timer_finish("NIE", metric_started)

        # Representational Dispersion: sensor-centered log Welch PSD patterns.
        metric_started = metric_timer_start()
        try:
            rd_value = None
            rd_reason = None
            rd_details: dict[str, object] = {**base_details}
            if n_channels < 2:
                rd_reason = "at_least_two_channels_required"
            else:
                nperseg = min(n_samples, max(8, int(round(4.0 * sfreq))))
                frequencies, psd = signal.welch(values, fs=sfreq, window="hann", nperseg=nperseg,
                                                 noverlap=nperseg // 2, detrend="constant", scaling="density", axis=-1)
                keep = (frequencies >= 1.0) & (frequencies <= min(45.0, sfreq / 2.0))
                if np.count_nonzero(keep) < 2 or not np.isfinite(psd[:, keep]).all() or np.any(psd[:, keep] <= 0.0):
                    rd_reason = "insufficient_positive_log_psd_frequency_support"
                else:
                    profiles = np.log10(psd[:, keep])
                    profiles = profiles - np.mean(profiles, axis=1, keepdims=True)
                    differences = profiles[:, None, :] - profiles[None, :, :]
                    distances = np.sqrt(np.sum(differences * differences, axis=-1))
                    upper = np.triu(np.ones(distances.shape, dtype=bool), k=1)
                    rd_value = float(np.mean(distances[upper]))
                    rd_details.update({"feature_construction": "within_sensor_centered_log10_Welch_PSD",
                                      "frequency_band_hz": [1.0, float(min(45.0, sfreq / 2.0))],
                                      "welch_nperseg": int(nperseg), "welch_noverlap": int(nperseg // 2),
                                      "retained_frequency_bin_count": int(np.count_nonzero(keep)),
                                      "distance": "mean_pairwise_Euclidean",
                                      "feature_units": "within_sensor_centered_log10_power",
                                      "space_claim": "sensor_pattern_only"})
            add("D3", "RD.pipeline_native_window_centered_log_psd_dispersion.v1", rd_value,
                "centered_log10_power", reason=rd_reason, details=rd_details)

        except Exception as exc:
            rows[:] = [row for row in rows if row.get("candidate_id") != "D3"]
            add("D3", "RD.pipeline_native_window_centered_log_psd_dispersion.v1", None, "centered_log10_power",
                reason="candidate_estimator_exception",
                details={**base_details, "exception_type": type(exc).__name__})
        finally:
            metric_timer_finish("RD", metric_started)

        # Ordinal Irregularity: normalized order-5, delay-1 permutation entropy.
        metric_started = metric_timer_start()
        try:
            try:
                entropy = np.asarray([compute_permutation_entropy(row, order=5, delay=1, normalize=True) for row in values], dtype=np.float64)
                oi_value = float(np.median(entropy)) if entropy.size and np.isfinite(entropy).all() else None
                oi_reason = None if oi_value is not None else "permutation_entropy_undefined"
            except Exception as exc:  # keep this metric isolated from the main feature path
                entropy = np.empty((0,), dtype=np.float64)
                oi_value = None
                oi_reason = f"permutation_entropy_failed:{type(exc).__name__}"
            add("E1", "OI.pipeline_native_window_permutation_entropy_order5_delay1.v1", oi_value,
                "normalized_entropy", reason=oi_reason,
                details={**base_details, "ordinal_order": 5, "delay_samples": 1,
                         "normalization": "divide_by_log2_factorial_order",
                         "aggregation": "median_over_all_native_channels",
                         "per_channel_computed_count": int(entropy.size)})

        except Exception as exc:
            rows[:] = [row for row in rows if row.get("candidate_id") != "E1"]
            add("E1", "OI.pipeline_native_window_permutation_entropy_order5_delay1.v1", None, "normalized_entropy",
                reason="candidate_estimator_exception",
                details={**base_details, "exception_type": type(exc).__name__})
        finally:
            metric_timer_finish("OI", metric_started)

        # Sequence Complexity: exact-multiset shuffled LZ78 reference ratio.
        metric_started = metric_timer_start()
        try:
            try:
                sc_value, sc_reason, sc_details = _sequence_complexity(
                    values, labels, source_identity=str(source_identity), segment_id=segment,
                    start_sec=start_sec, end_sec=end_sec,
                    reference_count=int(sequence_complexity_reference_count),
                )
            except Exception as exc:
                sc_value, sc_reason, sc_details = None, f"exact_multiset_reference_failed:{type(exc).__name__}", {}
            add("E2", "SC.pipeline_native_window_lz78_exact_multiset_ratio.v1", sc_value,
                "dimensionless_ratio", reason=sc_reason,
                details={**base_details, "construct": "temporal_order_complexity_relative_to_exact_multiset_shuffles", **sc_details})

        except Exception as exc:
            rows[:] = [row for row in rows if row.get("candidate_id") != "E2"]
            add("E2", "SC.pipeline_native_window_lz78_exact_multiset_ratio.v1", None, "dimensionless_ratio",
                reason="candidate_estimator_exception",
                details={**base_details, "exception_type": type(exc).__name__})
        finally:
            metric_timer_finish("SC", metric_started)

        # Effective Dimensionality: participation ratio of raw sensor covariance.
        metric_started = metric_timer_start()
        try:
            ed_value = None
            ed_reason = None
            ed_details: dict[str, object] = {**base_details}
            if n_samples < 2 or n_channels < 2:
                ed_reason = "at_least_two_channels_and_two_samples_required"
            else:
                try:
                    with np.errstate(over="ignore", invalid="ignore"):
                        covariance = np.cov(values, rowvar=True, ddof=1)
                    eigenvalues = np.linalg.eigvalsh(covariance) if np.isfinite(covariance).all() else np.asarray([np.nan])
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    covariance = np.asarray([np.nan])
                    eigenvalues = np.asarray([np.nan])
                if not np.isfinite(covariance).all() or not np.isfinite(eigenvalues).all():
                    ed_reason = "nonfinite_sensor_covariance"
                else:
                    eigenvalues = np.maximum(eigenvalues, 0.0)
                    trace = float(np.sum(eigenvalues))
                    squared_trace = float(np.sum(eigenvalues * eigenvalues))
                    if trace <= 0.0 or squared_trace <= 0.0:
                        ed_reason = "undefined_covariance_spectrum"
                    else:
                        ed_value = float(trace * trace / squared_trace)
                        ed_details.update({"covariance": "centered_sample_covariance_ddof_1",
                                           "shrinkage": "none", "normalized_by_channel_count": False,
                                           "effective_sample_support_gate": "not_assessed",
                                           "space_claim": "observed_sensor_space"})
            add("E3", "ED.pipeline_native_window_covariance_participation_ratio.v1", ed_value,
                "sensor_directions", reason=ed_reason, details=ed_details)

        except Exception as exc:
            rows[:] = [row for row in rows if row.get("candidate_id") != "E3"]
            add("E3", "ED.pipeline_native_window_covariance_participation_ratio.v1", None, "sensor_directions",
                reason="candidate_estimator_exception",
                details={**base_details, "exception_type": type(exc).__name__})
        finally:
            metric_timer_finish("ED", metric_started)

    for row in rows:
        row["details"] = {
            **dict(row.get("details", {})),
            "realization": realization,
        }
    return {
        "schema": PIPELINE_MEASUREMENT_SCHEMA,
        "status": "computed" if any(row["computed"] for row in rows) else "completed_no_values",
        "window": window,
        "realization": realization,
        "support_audit_status": support_status,
        "support_qc": support_qc,
        "acquisition_qc": dict(acquisition_qc or {}),
        "qualified": False,
        "coordinate_assignment": None,
        "measurements": rows,
    }


def failed_pipeline_window_measurements(
    *,
    channel_names: Sequence[str],
    sample_rate_hz: float,
    window_start_sec: float,
    window_end_sec: float,
    segment_id: str,
    input_branch: Mapping[str, object],
    support_qc: Mapping[str, object],
    error: BaseException,
) -> dict[str, object]:
    """Build explicit unsupported rows if an unexpected candidate error occurs."""
    labels = tuple(str(label).strip() for label in channel_names)
    realization = {
        "surface_id": "native",
        "surface_kind": "native",
        "channel_labels": list(labels),
        "channel_count": len(labels),
        "channel_order_sha256": _label_hash(labels, ordered=True) if labels else "",
        "channel_set_sha256": _label_hash(labels, ordered=False) if labels else "",
        "reference_mode": input_branch.get("reference_mode"),
        "reference_set_sha256": input_branch.get("reference_set_sha256"),
        "input_branch_id": input_branch.get("branch_id"),
        "filter_band_hz": input_branch.get("filter_band_hz"),
    }
    rows = _unsupported_rows(
        "candidate_pipeline_exception",
        {
            "exception_type": type(error).__name__,
            "support_qc": dict(support_qc),
            "realization": realization,
        },
    )
    return {
        "schema": PIPELINE_MEASUREMENT_SCHEMA,
        "status": "candidate_pipeline_exception",
        "window": {
            "start_sec": float(window_start_sec),
            "end_sec": float(window_end_sec),
            "sample_rate_hz": float(sample_rate_hz),
            "segment_id": str(segment_id),
        },
        "realization": realization,
        "support_audit_status": "incomplete",
        "support_qc": dict(support_qc),
        "qualified": False,
        "coordinate_assignment": None,
        "measurements": rows,
    }


__all__ = [
    "PIPELINE_MEASUREMENT_SCHEMA",
    "PIPELINE_E2_MASTER_SEED",
    "PIPELINE_E2_SEED_DERIVATION_ID",
    "compute_acquisition_qc",
    "compute_pipeline_window_measurements",
    "failed_pipeline_window_measurements",
]
