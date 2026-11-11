"""Fixed-sensor alpha phase extraction for the MNPS v3.1 M1 sidecar.

This opt-in helper consumes one contiguous, already resampled context around a
nominal window. It does not read EEG files, infer channel labels, rereference,
or qualify an M1 result. Source support and fixed-label identity are audited by
the caller before this function is called.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Sequence

import numpy as np
from scipy import signal


M1_ALPHA_SAMPLE_RATE_HZ = 250.0
M1_ALPHA_BAND_HZ = (8.0, 13.0)
M1_ALPHA_CHANNEL_COUNT = 83
M1_ALPHA_WINDOW_SECONDS = 20.0
M1_ALPHA_WINDOW_SAMPLES = 5000
M1_ALPHA_FILTER_ORDER = 4
M1_ALPHA_SOURCE_MARGIN_CYCLES = 3.0
M1_ALPHA_SOURCE_MARGIN_SAMPLES = math.ceil(
    M1_ALPHA_SOURCE_MARGIN_CYCLES
    * M1_ALPHA_SAMPLE_RATE_HZ
    / M1_ALPHA_BAND_HZ[0]
)
M1_ALPHA_PHASE_ESTIMATOR_ID = "M1.sensor_alpha_8_13_hilbert_butterworth_sos_filtfilt.v2"
M1_ALPHA_BAND_ID = "alpha_8_13_hz_fixed_83_label_sensor_set"


@dataclass(frozen=True)
class M1AlphaPhaseResult:
    """A central 20 s phase matrix or a null result with a stable reason."""

    phase_radians: np.ndarray | None
    reason: str | None
    details: dict[str, object]


def _label_key(value: object) -> str:
    return str(value).strip().casefold()


def extract_m1_alpha_hilbert_phase(
    extended_sensor_signals: np.ndarray,
    channel_names: Sequence[str],
    expected_channel_names: Sequence[str],
    *,
    sample_rate_hz: float = M1_ALPHA_SAMPLE_RATE_HZ,
    margin_samples_each_side: int = M1_ALPHA_SOURCE_MARGIN_SAMPLES,
) -> M1AlphaPhaseResult:
    """Extract central-window alpha phase from a fixed 83-sensor context.

    Input shape must be exactly ``(83, 5000 + 2*94)``. The first and last 94
    samples are real source context, not synthesized padding. The caller must
    establish that this context is within a single continuous, non-BAD source
    interval and includes the finite resampling dependencies. No fallback,
    channel dropping, alternate band, amplitude threshold, or padding is used.
    """
    try:
        values = np.asarray(extended_sensor_signals, dtype=np.float64)
    except (TypeError, ValueError):
        return M1AlphaPhaseResult(None, "invalid_numeric_input", {})

    if values.ndim != 2:
        return M1AlphaPhaseResult(None, "input_must_be_two_dimensional", {})
    if len(channel_names) != M1_ALPHA_CHANNEL_COUNT or len(expected_channel_names) != M1_ALPHA_CHANNEL_COUNT:
        return M1AlphaPhaseResult(None, "fixed_channel_set_must_contain_83_labels", {})
    actual_keys = tuple(_label_key(name) for name in channel_names)
    expected_keys = tuple(_label_key(name) for name in expected_channel_names)
    if (
        any(not key for key in actual_keys + expected_keys)
        or len(set(actual_keys)) != M1_ALPHA_CHANNEL_COUNT
        or len(set(expected_keys)) != M1_ALPHA_CHANNEL_COUNT
        or actual_keys != expected_keys
    ):
        return M1AlphaPhaseResult(None, "fixed_channel_set_mismatch", {})
    if values.shape[0] != M1_ALPHA_CHANNEL_COUNT:
        return M1AlphaPhaseResult(None, "channel_count_mismatch", {})
    if not math.isfinite(float(sample_rate_hz)) or not math.isclose(
        float(sample_rate_hz), M1_ALPHA_SAMPLE_RATE_HZ, rel_tol=0.0, abs_tol=1e-12
    ):
        return M1AlphaPhaseResult(None, "sample_rate_mismatch", {})
    if margin_samples_each_side != M1_ALPHA_SOURCE_MARGIN_SAMPLES:
        return M1AlphaPhaseResult(None, "frozen_source_margin_mismatch", {})
    expected_samples = M1_ALPHA_WINDOW_SAMPLES + 2 * M1_ALPHA_SOURCE_MARGIN_SAMPLES
    if values.shape != (M1_ALPHA_CHANNEL_COUNT, expected_samples):
        return M1AlphaPhaseResult(None, "insufficient_filter_edge_support", {})
    if not np.isfinite(values).all():
        return M1AlphaPhaseResult(None, "non_finite_input", {})
    if float(np.max(np.abs(values))) == 0.0:
        return M1AlphaPhaseResult(None, "zero_signal", {})
    flatline_channels = np.flatnonzero(np.ptp(values, axis=1) == 0.0)
    if flatline_channels.size:
        return M1AlphaPhaseResult(
            None,
            "flatline_fixed_sensor_channel",
            {
                "flatline_rule": "exact_zero_peak_to_peak_over_full_source_context",
                "flatline_channel_labels": [str(channel_names[int(index)]) for index in flatline_channels],
            },
        )

    try:
        sos = signal.butter(
            M1_ALPHA_FILTER_ORDER,
            list(M1_ALPHA_BAND_HZ),
            btype="bandpass",
            fs=M1_ALPHA_SAMPLE_RATE_HZ,
            output="sos",
        )
        filtered = signal.sosfiltfilt(
            sos,
            values,
            axis=-1,
            padtype=None,
            padlen=0,
        )
        analytic = signal.hilbert(filtered, axis=-1)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return M1AlphaPhaseResult(None, "phase_estimator_failed", {})

    start = M1_ALPHA_SOURCE_MARGIN_SAMPLES
    stop = start + M1_ALPHA_WINDOW_SAMPLES
    analytic_central = analytic[:, start:stop]
    amplitude = np.abs(analytic_central)
    if not np.isfinite(amplitude).all():
        return M1AlphaPhaseResult(None, "non_finite_analytic_amplitude", {})
    zero_amplitude = np.argwhere(amplitude == 0.0)
    if zero_amplitude.size:
        return M1AlphaPhaseResult(
            None,
            "zero_analytic_amplitude",
            {
                "amplitude_validity_rule": "reject_any_exact_zero_analytic_amplitude_sample",
                "zero_amplitude_sample_count": int(zero_amplitude.shape[0]),
                "zero_amplitude_channel_labels": sorted(
                    {str(channel_names[int(index)]) for index in zero_amplitude[:, 0]}
                ),
            },
        )
    phase = np.asarray(np.angle(analytic_central), dtype=np.float64)
    if phase.shape != (M1_ALPHA_CHANNEL_COUNT, M1_ALPHA_WINDOW_SAMPLES):
        return M1AlphaPhaseResult(None, "phase_output_shape_mismatch", {})
    if not np.isfinite(phase).all():
        return M1AlphaPhaseResult(None, "non_finite_phase_estimate", {})

    sos_sha256 = hashlib.sha256(np.asarray(sos, dtype="<f8").tobytes()).hexdigest()
    details: dict[str, object] = {
        "phase_estimator_id": M1_ALPHA_PHASE_ESTIMATOR_ID,
        "band_id": M1_ALPHA_BAND_ID,
        "band_hz": list(M1_ALPHA_BAND_HZ),
        "sample_rate_hz": M1_ALPHA_SAMPLE_RATE_HZ,
        "filter": "scipy.signal.butter_sos_order4_then_sosfiltfilt_zero_phase",
        "sosfiltfilt_padtype": None,
        "sosfiltfilt_padlen": 0,
        "analytic_signal": "scipy.signal.hilbert_axis_last",
        "source_margin_cycles_at_low_edge": M1_ALPHA_SOURCE_MARGIN_CYCLES,
        "source_margin_samples_each_side": M1_ALPHA_SOURCE_MARGIN_SAMPLES,
        "source_margin_seconds_each_side": M1_ALPHA_SOURCE_MARGIN_SAMPLES / M1_ALPHA_SAMPLE_RATE_HZ,
        "source_margin_basis": "ceil(3_cycles*250_Hz/8_Hz), conservative transfer of the M1-G1-S1 three-cycle edge-support probe",
        "nominal_window_samples": M1_ALPHA_WINDOW_SAMPLES,
        "phase_output_samples": M1_ALPHA_WINDOW_SAMPLES,
        "phase_range_policy": "principal_angle_minus_pi_to_pi",
        "rereference": "source_reference_preserved",
        "notch_hz": None,
        "csd": False,
        "coordinates_used": False,
        "amplitude_threshold": None,
        "amplitude_validity_rule": "reject_exactly_flat_input_channel_or_any_exact_zero_analytic_amplitude_sample",
        "sos_coefficients_sha256": sos_sha256,
    }
    return M1AlphaPhaseResult(phase, None, details)
