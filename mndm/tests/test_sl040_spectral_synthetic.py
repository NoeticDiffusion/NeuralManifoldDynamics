"""Focused contract tests for the frozen SL-040 synthetic check."""
import importlib.util
from pathlib import Path

import numpy as np
from scipy import signal


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "project/new_mnps_v.3.2/science_lead/sl040_spectral_synthetic.py"
SPEC = importlib.util.spec_from_file_location("sl040_spectral_synthetic", SCRIPT_PATH)
sl040 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sl040)


def test_powerlaw_window_is_seed_reproducible_and_valid():
    first = sl040.make_powerlaw_window(1.5, 2, 7)
    repeat = sl040.make_powerlaw_window(1.5, 2, 7)
    other = sl040.make_powerlaw_window(1.5, 2, 8)

    assert first.shape == (5120,)
    assert np.isfinite(first).all()
    assert np.isclose(np.std(first, ddof=1), 1.0)
    np.testing.assert_array_equal(first, repeat)
    assert not np.array_equal(first, other)


def test_processing_branches_have_target_support_and_match_frozen_operations():
    source = sl040.make_powerlaw_window(1.0, 1, 3)
    resample_only = sl040.process_window(source, apply_boas_filter=False)
    actual_boas = sl040.process_window(source, apply_boas_filter=True)

    expected_resample = signal.resample_poly(source, 125, 64)
    sos = signal.butter(4, (1, 45), btype="bandpass", fs=256.0, output="sos")
    expected_boas = signal.resample_poly(signal.sosfiltfilt(sos, source), 125, 64)
    assert resample_only.shape == actual_boas.shape == (10000,)
    np.testing.assert_array_equal(resample_only, expected_resample)
    np.testing.assert_array_equal(actual_boas, expected_boas)
    assert not np.array_equal(resample_only, actual_boas)


def test_summary_keeps_requested_count_and_paired_finite_denominator():
    records = [
        {
            "known_beta": 1.0,
            "resample_only": {"aperiodic_exponent": 0.8, "exponent_valid": True},
            "boas_filter_then_resample": {"aperiodic_exponent": 1.1, "exponent_valid": True},
        },
        {
            "known_beta": 1.0,
            "resample_only": {"aperiodic_exponent": None, "exponent_valid": False},
            "boas_filter_then_resample": {"aperiodic_exponent": 1.3, "exponent_valid": True},
        },
    ]

    summary = sl040.summarize(records)["1.0"]
    resample = summary["branches"]["resample_only"]
    filtered = summary["branches"]["boas_filter_then_resample"]
    paired = summary["paired_filter_minus_resample_only"]
    assert (resample["requested"], resample["finite"], resample["finite_fraction"]) == (2, 1, 0.5)
    assert np.isclose(resample["mean_bias_vs_known_beta"], -0.2)
    assert (filtered["requested"], filtered["finite"]) == (2, 2)
    assert (paired["requested"], paired["finite_pairs"]) == (2, 1)
    assert np.isclose(paired["mean"], 0.3)
