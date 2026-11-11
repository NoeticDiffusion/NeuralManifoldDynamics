import numpy as np

from mndm.mnps.v3_1.m1_phase import (
    M1_ALPHA_CHANNEL_COUNT,
    M1_ALPHA_PHASE_ESTIMATOR_ID,
    M1_ALPHA_SOURCE_MARGIN_SAMPLES,
    M1_ALPHA_WINDOW_SAMPLES,
    extract_m1_alpha_hilbert_phase,
)
from mndm.mnps.v3_1 import m1_phase


def _labels():
    return tuple(f"EEG_{index:02d}" for index in range(M1_ALPHA_CHANNEL_COUNT))


def _signal_matrix():
    total = M1_ALPHA_WINDOW_SAMPLES + 2 * M1_ALPHA_SOURCE_MARGIN_SAMPLES
    time = np.arange(total, dtype=np.float64) / 250.0
    offsets = np.linspace(-1.2, 1.2, M1_ALPHA_CHANNEL_COUNT)
    return np.cos(2.0 * np.pi * 10.0 * time[None, :] + offsets[:, None])


def test_m1_alpha_phase_returns_exact_central_fixed_set_matrix():
    labels = _labels()
    result = extract_m1_alpha_hilbert_phase(_signal_matrix(), labels, labels)

    assert result.reason is None
    assert result.phase_radians.shape == (M1_ALPHA_CHANNEL_COUNT, M1_ALPHA_WINDOW_SAMPLES)
    assert np.isfinite(result.phase_radians).all()
    assert result.details["phase_estimator_id"] == M1_ALPHA_PHASE_ESTIMATOR_ID
    assert result.details["source_margin_samples_each_side"] == 94
    assert result.details["rereference"] == "source_reference_preserved"
    assert result.details["coordinates_used"] is False


def test_m1_alpha_phase_rejects_channel_reordering_and_missing_labels():
    labels = _labels()
    reordered = (labels[1], labels[0], *labels[2:])
    missing = labels[:-1]

    assert extract_m1_alpha_hilbert_phase(_signal_matrix(), reordered, labels).reason == "fixed_channel_set_mismatch"
    assert extract_m1_alpha_hilbert_phase(_signal_matrix(), missing, labels).reason == "fixed_channel_set_must_contain_83_labels"


def test_m1_alpha_phase_rejects_insufficient_margin_and_nonfinite_input():
    labels = _labels()
    values = _signal_matrix()
    too_short = values[:, 1:]
    nonfinite = values.copy()
    nonfinite[4, 100] = np.nan

    assert extract_m1_alpha_hilbert_phase(too_short, labels, labels).reason == "insufficient_filter_edge_support"
    assert extract_m1_alpha_hilbert_phase(
        values,
        labels,
        labels,
        margin_samples_each_side=M1_ALPHA_SOURCE_MARGIN_SAMPLES - 1,
    ).reason == "frozen_source_margin_mismatch"
    assert extract_m1_alpha_hilbert_phase(nonfinite, labels, labels).reason == "non_finite_input"


def test_m1_alpha_phase_rejects_wrong_sample_rate_and_zero_signal():
    labels = _labels()
    zeros = np.zeros_like(_signal_matrix())

    assert extract_m1_alpha_hilbert_phase(
        _signal_matrix(), labels, labels, sample_rate_hz=256.0
    ).reason == "sample_rate_mismatch"
    assert extract_m1_alpha_hilbert_phase(zeros, labels, labels).reason == "zero_signal"


def test_m1_alpha_phase_rejects_a_single_exact_flatline_channel():
    labels = _labels()
    for constant in (0.0, 1.0):
        values = _signal_matrix()
        values[17, :] = constant

        result = extract_m1_alpha_hilbert_phase(values, labels, labels)

        assert result.reason == "flatline_fixed_sensor_channel"
        assert result.details["flatline_rule"] == "exact_zero_peak_to_peak_over_full_source_context"
        assert result.details["flatline_channel_labels"] == [labels[17]]


def test_m1_alpha_phase_rejects_any_exact_zero_analytic_amplitude(monkeypatch):
    labels = _labels()
    original_hilbert = m1_phase.signal.hilbert

    def inject_exact_zero(values, axis=-1):
        analytic = original_hilbert(values, axis=axis)
        analytic[5, 100] = 0.0j
        return analytic

    monkeypatch.setattr(m1_phase.signal, "hilbert", inject_exact_zero)
    result = extract_m1_alpha_hilbert_phase(_signal_matrix(), labels, labels)

    assert result.reason == "zero_analytic_amplitude"
    assert result.details["zero_amplitude_sample_count"] == 1
    assert result.details["zero_amplitude_channel_labels"] == [labels[5]]
