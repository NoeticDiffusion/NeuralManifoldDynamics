"""P0.7: coverage is interval union, not span; derivatives do not cross time holes."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.pipeline.summary_events import estimate_coverage_seconds, union_interval_seconds
from mndm.projection import (
    TIME_GAP_TOL,
    estimate_derivatives_with_time_gaps,
    time_gap_slices,
)

WINDOW_SEC = 45.0
STEP_SEC = 15.0
HOLE_SEC = 60.0
N_PER_BLOCK = 10


def _two_blocks():
    t_a = np.arange(N_PER_BLOCK, dtype=float) * STEP_SEC
    t_b = (t_a[-1] + WINDOW_SEC) + HOLE_SEC + np.arange(N_PER_BLOCK, dtype=float) * STEP_SEC
    t_start = np.concatenate([t_a, t_b])
    t_end = t_start + WINDOW_SEC
    return t_start, t_end


def test_union_interval_seconds_merges_overlap_not_holes():
    t_start, t_end = _two_blocks()
    union = union_interval_seconds(t_start, t_end)
    span = float(t_end.max() - t_start.min())
    block_union = (t_end[N_PER_BLOCK - 1] - t_start[0]) + (t_end[-1] - t_start[N_PER_BLOCK])
    assert union == pytest.approx(block_union)
    assert union == pytest.approx(2 * (WINDOW_SEC + (N_PER_BLOCK - 1) * STEP_SEC))
    assert span == pytest.approx(union + HOLE_SEC)
    assert union < span


def test_estimate_coverage_seconds_uses_union_not_span():
    t_start, t_end = _two_blocks()
    frame = pd.DataFrame({"t_start": t_start, "t_end": t_end})
    measured, method = estimate_coverage_seconds(frame, dt_fallback=STEP_SEC)
    span = float(t_end.max() - t_start.min())
    assert method == "timestamps_union"
    assert measured == pytest.approx(union_interval_seconds(t_start, t_end))
    assert measured == pytest.approx(span - HOLE_SEC)
    assumed = float(len(frame) * STEP_SEC)
    assert measured != pytest.approx(assumed)


def test_contiguous_windows_union_equals_span():
    t_start = np.arange(8, dtype=float) * STEP_SEC
    t_end = t_start + WINDOW_SEC
    frame = pd.DataFrame({"t_start": t_start, "t_end": t_end})
    measured, method = estimate_coverage_seconds(frame, dt_fallback=STEP_SEC)
    assert method == "timestamps_union"
    assert measured == pytest.approx(float(t_end.max() - t_start.min()))


def test_time_gap_slices_split_on_step_times_one_plus_tol():
    t_start, _ = _two_blocks()
    slices = time_gap_slices(t_start, STEP_SEC, tol=TIME_GAP_TOL)
    assert slices == [slice(0, N_PER_BLOCK), slice(N_PER_BLOCK, 2 * N_PER_BLOCK)]
    # A 20% extra delay is within default tol=0.25.
    t_ok = np.array([0.0, 15.0, 15.0 * 1.2])
    assert time_gap_slices(t_ok, STEP_SEC) == [slice(0, 3)]
    # Non-positive Δt_start is a gap even if it is smaller than the forward threshold.
    t_back = np.array([0.0, 15.0, 10.0, 25.0])
    assert time_gap_slices(t_back, STEP_SEC) == [slice(0, 2), slice(2, 4)]


def test_x_dot_is_nan_at_time_hole_and_not_smeared():
    t_start, _ = _two_blocks()
    x = np.stack([t_start, t_start, t_start], axis=1).astype(np.float32)
    x_dot = estimate_derivatives_with_time_gaps(
        x,
        STEP_SEC,
        t_start,
        method="central",
        use_segmented=False,
        nan_gap_edges=True,
    )
    join_a = N_PER_BLOCK - 1
    join_b = N_PER_BLOCK
    assert np.all(np.isnan(x_dot[join_a]))
    assert np.all(np.isnan(x_dot[join_b]))
    interior = np.concatenate(
        [np.arange(1, join_a), np.arange(join_b + 1, 2 * N_PER_BLOCK - 1)]
    )
    interior_dot = x_dot[interior, 0]
    assert np.all(np.isfinite(interior_dot))
    assert np.allclose(interior_dot, 1.0, atol=0.05)
    # Crossing the hole with uniform dt would look like (60+15)/15 = 5.
    assert np.max(np.abs(interior_dot)) < 2.0


def test_inverted_intervals_are_dropped_from_union():
    frame = pd.DataFrame(
        {
            "t_start": [0.0, 15.0, 30.0],
            "t_end": [45.0, 10.0, 75.0],
        }
    )
    measured, method = estimate_coverage_seconds(frame, dt_fallback=STEP_SEC)
    assert method == "timestamps_union"
    assert measured == pytest.approx(75.0)


def test_missing_timestamps_fall_back_to_len_dt():
    frame = pd.DataFrame({"feat": np.arange(5, dtype=float)})
    measured, method = estimate_coverage_seconds(frame, dt_fallback=2.0)
    assert method == "assumed_len_dt"
    assert measured == pytest.approx(10.0)
