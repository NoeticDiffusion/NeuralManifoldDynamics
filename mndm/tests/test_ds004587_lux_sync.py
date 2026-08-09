"""Tests for ds004587 LUX-photosensor trial-clock recovery."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.pipeline.ds004587_lux_sync import (
    behavioral_block_landmarks,
    build_ig_trial_event_table,
    estimate_landmark_consensus_offset,
    lux_plateau_landmarks,
    run_sync_quality_row,
)


def test_behavioral_block_landmarks_requires_finite_block_number():
    """Only gaps that also carry a real block_number count as landmarks."""
    time_elapsed = np.array([0.0, 1000.0, 6000.0, 7000.0, 20000.0, 21000.0])
    # Gap at idx1->2 (5000ms) has NaN block_number at idx1 (early instructions,
    # excluded); gap at idx3->4 (13000ms) has a real block_number at idx3 and
    # is the only qualifying landmark.
    block_number = np.array([np.nan, np.nan, np.nan, 1.0, np.nan, np.nan])

    landmarks = behavioral_block_landmarks(time_elapsed, block_number, min_gap_ms=4000.0)

    assert landmarks.tolist() == [7000.0]


def test_lux_plateau_landmarks_detects_long_high_and_low_plateaus():
    """A synthetic step signal should yield one landmark per long plateau."""
    sfreq = 1000.0
    lux = np.zeros(20_000)
    lux[5_000:11_000] = 1.0  # 6s high plateau starting at t=5000ms
    lux[14_000:14_200] = 1.0  # 200ms blip -- too short to count

    landmarks = lux_plateau_landmarks(lux, sfreq_hz=sfreq, min_duration_ms=5000.0)

    assert landmarks.size == 1
    assert abs(landmarks[0] - 5000.0) < 1.0


def test_lux_plateau_landmarks_empty_for_flat_signal():
    """A LUX channel with zero variance (sensor not functional) yields no landmarks."""
    lux = np.full(10_000, 512.0)
    landmarks = lux_plateau_landmarks(lux, sfreq_hz=1000.0)
    assert landmarks.size == 0


def test_estimate_landmark_consensus_offset_clean_case():
    """A clean constant offset should be recovered exactly with all landmarks matched."""
    beh = np.array([10_000.0, 100_000.0, 250_000.0, 400_000.0, 600_000.0])
    true_offset = 137_842.0
    lux = beh + true_offset

    result = estimate_landmark_consensus_offset(beh, lux, tolerance_ms=50.0)

    assert result.n_matched == 5
    assert abs(result.offset_ms - true_offset) < 1e-6
    assert result.max_abs_residual_ms < 1e-6
    assert result.qc_ok()
    lo, hi = result.bracket_beh_ms
    assert lo == pytest.approx(10_000.0)
    assert hi == pytest.approx(600_000.0)


def test_estimate_landmark_consensus_offset_small_jitter_still_matches():
    """Sub-tolerance per-landmark jitter should not break consensus matching."""
    rng = np.random.default_rng(0)
    beh = np.array([10_000.0, 100_000.0, 250_000.0, 400_000.0, 600_000.0])
    true_offset = -42_000.0
    jitter = rng.uniform(-15.0, 15.0, size=beh.size)
    lux = beh + true_offset + jitter

    result = estimate_landmark_consensus_offset(beh, lux, tolerance_ms=50.0)

    assert result.n_matched == 5
    assert result.max_abs_residual_ms < 50.0
    assert result.qc_ok()


def test_estimate_landmark_consensus_offset_drifted_case_fails_quality_gate():
    """Linear clock drift breaks the constant-offset assumption for most landmarks."""
    beh = np.array([0.0, 100_000.0, 200_000.0, 300_000.0, 400_000.0])
    true_offset = 50_000.0
    drift_ppm = 500.0  # 500 ppm drift accumulates far beyond tolerance over the span
    drift_ms = beh * (drift_ppm / 1.0e6)
    lux = beh + true_offset + drift_ms

    result = estimate_landmark_consensus_offset(beh, lux, tolerance_ms=50.0)

    # Only landmarks near the anchor point used for the winning offset stay
    # within tolerance; the run should not silently claim a good sync.
    assert result.n_matched < beh.size
    assert not result.qc_ok(min_matched=5)


def test_estimate_landmark_consensus_offset_insufficient_landmarks():
    """Fewer than 2 landmarks on either side cannot produce a trustworthy offset."""
    result = estimate_landmark_consensus_offset(np.array([1.0]), np.array([2.0, 3.0]))
    assert result.n_matched == 0
    assert not np.isfinite(result.offset_ms)
    assert not result.qc_ok()


def test_estimate_landmark_consensus_offset_zero_lux_landmarks():
    """A flat/non-functional LUX channel yields zero LUX landmarks and no offset."""
    beh = np.array([10_000.0, 100_000.0, 250_000.0])
    result = estimate_landmark_consensus_offset(beh, np.empty(0))
    assert result.n_matched == 0
    assert not result.qc_ok()


def _make_beh_frame(n_trials: int, spacing_ms: float, start_ms: float) -> pd.DataFrame:
    rows = []
    for i in range(n_trials):
        rows.append({
            "trial_type": "image-keyboard-response",
            "time_elapsed": start_ms + i * spacing_ms,
            "block_number": float(1 + i // 10),
            "trial_number": float(i),
            "type": "MullerLyer",
            "illusion_strength": 5.0,
            "illusion_difference": 0.1,
            "correct": bool(i % 2 == 0),
            "correct_response": "arrowup",
            "response": "arrowup",
            "rt": 500.0,
            "stimulus": f"stimuli/img_{i}.png",
            "trial_index": i,
            "block": "MullerLyer",
        })
    return pd.DataFrame(rows)


def test_build_ig_trial_event_table_recovers_onsets_within_bracket_when_qc_ok():
    """Trials inside the matched-landmark bracket get a finite onset when qc_ok."""
    beh = _make_beh_frame(n_trials=20, spacing_ms=2000.0, start_ms=0.0)
    beh_landmarks = np.array([0.0, 10_000.0, 20_000.0, 30_000.0])
    offset = 5_000.0
    lux_landmarks = beh_landmarks + offset
    estimate = estimate_landmark_consensus_offset(beh_landmarks, lux_landmarks, tolerance_ms=50.0)
    assert estimate.qc_ok()

    trials = build_ig_trial_event_table(beh, estimate, subject_id="sub-FFE999", run_id="run-01")

    assert len(trials) == 20
    assert trials["qc_ok_event_sync"].all()
    within_bracket = trials["within_sync_bracket"]
    assert within_bracket.any()
    recovered = trials.loc[within_bracket, "onset_sec"]
    assert np.isfinite(recovered).all()
    expected = (beh.loc[within_bracket, "time_elapsed"].to_numpy() + offset) / 1000.0
    np.testing.assert_allclose(recovered.to_numpy(), expected)
    # Trials outside the landmark-bracketed interval must not be extrapolated.
    outside = trials.loc[~within_bracket, "onset_sec"]
    assert np.isnan(outside.to_numpy()).all()
    # Non-EventTable columns should be preserved verbatim for downstream joins.
    assert "illusion_strength" in trials.columns
    assert "correct" in trials.columns


def test_build_ig_trial_event_table_all_nan_when_qc_fails():
    """A run whose offset fails the quality gate must not fabricate any onsets."""
    beh = _make_beh_frame(n_trials=10, spacing_ms=1000.0, start_ms=0.0)
    estimate = estimate_landmark_consensus_offset(np.array([1.0]), np.array([2.0, 3.0]))
    assert not estimate.qc_ok()

    trials = build_ig_trial_event_table(beh, estimate, subject_id="sub-FFE999", run_id="run-01")

    assert len(trials) == 10
    assert not trials["qc_ok_event_sync"].any()
    assert np.isnan(trials["onset_sec"].to_numpy()).all()


def test_run_sync_quality_row_reports_recovery_counts():
    beh = _make_beh_frame(n_trials=5, spacing_ms=1000.0, start_ms=0.0)
    beh_landmarks = np.array([0.0, 1000.0, 2000.0])
    lux_landmarks = beh_landmarks + 1000.0
    estimate = estimate_landmark_consensus_offset(beh_landmarks, lux_landmarks, tolerance_ms=50.0)
    trials = build_ig_trial_event_table(beh, estimate, subject_id="sub-FFE999", run_id="run-01")

    row = run_sync_quality_row(
        estimate,
        subject_id="sub-FFE999",
        run_id="run-01",
        n_trials_total=len(trials),
        n_trials_recovered=int(np.isfinite(trials["onset_sec"]).sum()),
    )

    assert row["subject_id"] == "sub-FFE999"
    assert row["n_trials_total"] == 5
    assert row["qc_ok_event_sync"] == estimate.qc_ok()
