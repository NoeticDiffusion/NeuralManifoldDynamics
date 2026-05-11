"""
Regression tests for the summary.py time-axis fix.

Released as part of the sleep-spindle event-locked track.
These tests guard against re-introducing the bug where /time was derived
from ``mnps.window_sec / overlap`` rather than from feature-epoch midpoints.

The bug caused:
  - /time step = 4 s when feature epochs had step = 2 s
  - Wrong rel_time_sec in event alignment
  - Wrong Jacobian j_dot_dt (velocity scale error 2×)
  - coverage_seconds_assumed = 2× actual recording span

Contract under test
-------------------
When ``sub_frame`` contains ``t_start`` and ``t_end`` columns:

  time[i] == (t_start[i] + t_end[i]) / 2        for all i   [C1]
  dt       == median(diff(t_start))               [C2]
  coverage_seconds_assumed ≈ len(sub_frame) * dt  [C3]

When ``sub_frame`` does NOT contain ``t_start``/``t_end``:

  time[i] == window_sec/2 + step*i  (build_time_index fallback)  [C4]
  dt       == window_sec * (1 - overlap)                          [C5]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Helpers — replicate the dt/time derivation logic from summary.py
# so these tests don't call the full summarize pipeline (which needs many
# dependencies and real data).  The logic is extracted verbatim from the
# changed lines in summary.py; any future refactor must keep it in sync.
# ---------------------------------------------------------------------------

def _derive_dt_and_time(
    sub_frame: pd.DataFrame,
    mnps_cfg: Dict[str, Any],
) -> tuple[float, np.ndarray]:
    """Mirror the dt/time logic from SubjectSummaryRunner.run().

    Returns (dt_sec, time_array).
    """
    from mndm.projection import build_time_index

    cfg_dt = mnps_cfg["window_sec"] * (1.0 - mnps_cfg["overlap"])

    if "t_start" in sub_frame.columns and "t_end" in sub_frame.columns and len(sub_frame) > 1:
        t_s_raw = pd.to_numeric(sub_frame["t_start"], errors="coerce")
        measured_dt = float(t_s_raw.diff().dropna().median())
        if np.isfinite(measured_dt) and measured_dt > 0:
            dt = measured_dt
        else:
            dt = cfg_dt
    else:
        dt = cfg_dt

    if "t_start" in sub_frame.columns and "t_end" in sub_frame.columns:
        t_s = pd.to_numeric(sub_frame["t_start"], errors="coerce")
        t_e = pd.to_numeric(sub_frame["t_end"], errors="coerce")
        time = ((t_s + t_e) / 2.0).to_numpy(dtype=np.float64)
    else:
        time = build_time_index(len(sub_frame), mnps_cfg["window_sec"], mnps_cfg["overlap"])

    return dt, time


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_frame(n: int, length_s: float, step_s: float) -> pd.DataFrame:
    """Create a minimal features sub_frame with t_start/t_end."""
    t_start = np.arange(n, dtype=np.float64) * step_s
    t_end = t_start + length_s
    return pd.DataFrame({"t_start": t_start, "t_end": t_end, "epoch_id": np.arange(n)})


def _make_frame_no_bounds(n: int) -> pd.DataFrame:
    """Features sub_frame without t_start/t_end (fMRI / legacy)."""
    return pd.DataFrame({"epoch_id": np.arange(n)})


MNPS_CFG_LEGACY = {"window_sec": 8.0, "overlap": 0.5}    # old base config → step=4 s
MNPS_CFG_SPINDLE = {"window_sec": 6.0, "overlap": 0.6667}  # spindle overlay hint


# ---------------------------------------------------------------------------
# C1: time == (t_start + t_end) / 2
# ---------------------------------------------------------------------------

class TestTimeEqualsMidpoint:
    """[C1] /time axis == feature-epoch midpoints."""

    def test_6s_step_2s(self):
        frame = _make_frame(100, length_s=6.0, step_s=2.0)
        dt, time = _derive_dt_and_time(frame, MNPS_CFG_LEGACY)
        expected = (frame["t_start"].values + frame["t_end"].values) / 2.0
        np.testing.assert_allclose(time, expected, atol=1e-9)

    def test_30s_step_30s(self):
        """Standard 30 s staging epochs (no overlap)."""
        frame = _make_frame(50, length_s=30.0, step_s=30.0)
        _, time = _derive_dt_and_time(frame, MNPS_CFG_LEGACY)
        expected = frame["t_start"].values + 15.0  # center of 30 s window
        np.testing.assert_allclose(time, expected, atol=1e-9)

    def test_8s_step_2s(self):
        """Alternative spindle window length."""
        frame = _make_frame(200, length_s=8.0, step_s=2.0)
        _, time = _derive_dt_and_time(frame, MNPS_CFG_LEGACY)
        expected = frame["t_start"].values + 4.0
        np.testing.assert_allclose(time, expected, atol=1e-9)

    def test_single_epoch(self):
        """Single epoch: t_start/t_end present but diff() is empty → fallback to cfg_dt."""
        frame = _make_frame(1, length_s=6.0, step_s=2.0)
        dt, time = _derive_dt_and_time(frame, MNPS_CFG_LEGACY)
        # dt falls back to config formula for n=1, but time still uses midpoint
        assert time[0] == pytest.approx(3.0, abs=1e-9)
        assert dt == pytest.approx(MNPS_CFG_LEGACY["window_sec"] * (1 - MNPS_CFG_LEGACY["overlap"]))

    def test_time_step_equals_epoch_step(self):
        """time[i+1] - time[i] == step_s for all i."""
        step_s = 2.0
        frame = _make_frame(50, length_s=6.0, step_s=step_s)
        _, time = _derive_dt_and_time(frame, MNPS_CFG_LEGACY)
        diffs = np.diff(time)
        np.testing.assert_allclose(diffs, step_s, atol=1e-9)

    def test_NOT_formula_derived(self):
        """Confirm that the old formula (step = window_sec*(1-overlap) = 4 s)
        is no longer used when t_start/t_end are present."""
        frame = _make_frame(20, length_s=6.0, step_s=2.0)
        _, time = _derive_dt_and_time(frame, MNPS_CFG_LEGACY)
        # Old formula would give time[0] = 4.0, new gives 3.0
        assert time[0] == pytest.approx(3.0, abs=1e-9), (
            f"time[0]={time[0]:.3f}; expected 3.0 (midpoint), not 4.0 (old formula)"
        )


# ---------------------------------------------------------------------------
# C2: dt follows t_start step
# ---------------------------------------------------------------------------

class TestDtFromFeatureStep:
    """[C2] dt is derived from median(diff(t_start)), not from config formula."""

    def test_dt_2s_step(self):
        frame = _make_frame(100, length_s=6.0, step_s=2.0)
        dt, _ = _derive_dt_and_time(frame, MNPS_CFG_LEGACY)
        assert dt == pytest.approx(2.0, abs=1e-9), f"Expected dt=2.0, got {dt}"

    def test_dt_30s_step(self):
        frame = _make_frame(50, length_s=30.0, step_s=30.0)
        dt, _ = _derive_dt_and_time(frame, MNPS_CFG_LEGACY)
        assert dt == pytest.approx(30.0, abs=1e-9)

    def test_dt_NOT_config_formula_when_differ(self):
        """Config formula gives 4 s; measured step is 2 s → must be 2 s."""
        frame = _make_frame(50, length_s=6.0, step_s=2.0)
        dt, _ = _derive_dt_and_time(frame, MNPS_CFG_LEGACY)
        cfg_dt = MNPS_CFG_LEGACY["window_sec"] * (1.0 - MNPS_CFG_LEGACY["overlap"])
        assert dt != pytest.approx(cfg_dt, abs=0.01), (
            "dt should not equal config formula when t_start step differs"
        )

    def test_dt_equals_config_when_consistent(self):
        """When feature step matches config formula, both paths agree."""
        # window_sec=4, overlap=0.5 → formula dt=2 s; feature step=2 s
        cfg = {"window_sec": 4.0, "overlap": 0.5}
        frame = _make_frame(50, length_s=4.0, step_s=2.0)
        dt, _ = _derive_dt_and_time(frame, cfg)
        assert dt == pytest.approx(2.0, abs=1e-9)


# ---------------------------------------------------------------------------
# C3: coverage_seconds_assumed ≈ recording span
# ---------------------------------------------------------------------------

class TestCoverageSeconds:
    """[C3] coverage_seconds_assumed uses corrected dt."""

    def test_coverage_6s_step_2s(self):
        n, step_s = 100, 2.0
        frame = _make_frame(n, length_s=6.0, step_s=step_s)
        dt, _ = _derive_dt_and_time(frame, MNPS_CFG_LEGACY)
        coverage_assumed = float(n * dt)
        # Should be 100 * 2 = 200 s, not 100 * 4 = 400 s
        assert coverage_assumed == pytest.approx(200.0, abs=0.1)

    def test_coverage_matches_actual_span(self):
        """coverage_assumed is within one epoch_length of the actual recording span.

        coverage_assumed = n * step  (n epochs × step_s)
        actual_span      = (n-1)*step + length  (last window end)

        The difference is length - step = 6 - 2 = 4 s by design.
        The key assertion is that coverage is NOT 2× actual (the pre-fix bug).
        """
        n, step_s, length_s = 13727, 2.0, 6.0
        frame = _make_frame(n, length_s=length_s, step_s=step_s)
        dt, _ = _derive_dt_and_time(frame, MNPS_CFG_LEGACY)
        coverage_assumed = float(n * dt)
        actual_span = float(frame["t_end"].iloc[-1])
        # Expected: 27454 s vs 27458 s — within one epoch length
        assert abs(coverage_assumed - actual_span) <= length_s, (
            f"coverage_assumed={coverage_assumed:.1f} s, actual_span={actual_span:.1f} s; "
            f"diff={abs(coverage_assumed - actual_span):.1f} s > epoch_length={length_s} s"
        )
        # Pre-fix sanity: old formula would give 54908 s ≈ 2× actual
        old_dt = MNPS_CFG_LEGACY["window_sec"] * (1.0 - MNPS_CFG_LEGACY["overlap"])
        coverage_old = float(n * old_dt)
        assert coverage_assumed < coverage_old * 0.6, (
            "Post-fix coverage must be substantially less than pre-fix formula"
        )

    def test_coverage_old_formula_would_be_double(self):
        """Verify the pre-fix coverage was 2× actual for 6 s/2 s epochs."""
        n, step_s = 1000, 2.0
        frame = _make_frame(n, length_s=6.0, step_s=step_s)
        dt, _ = _derive_dt_and_time(frame, MNPS_CFG_LEGACY)
        coverage_new = float(n * dt)

        old_dt = MNPS_CFG_LEGACY["window_sec"] * (1.0 - MNPS_CFG_LEGACY["overlap"])  # 4 s
        coverage_old = float(n * old_dt)

        assert coverage_old == pytest.approx(2 * coverage_new, rel=0.01), (
            "Pre-fix coverage should be 2× post-fix for 6 s/2 s epochs"
        )


# ---------------------------------------------------------------------------
# C4 + C5: fallback when t_start/t_end absent
# ---------------------------------------------------------------------------

class TestFallbackWithoutBounds:
    """[C4, C5] Without t_start/t_end, behaviour reverts to build_time_index formula."""

    def test_fallback_time_uses_formula(self):
        from mndm.projection import build_time_index
        n = 20
        frame = _make_frame_no_bounds(n)
        dt, time = _derive_dt_and_time(frame, MNPS_CFG_LEGACY)
        expected = build_time_index(n, MNPS_CFG_LEGACY["window_sec"], MNPS_CFG_LEGACY["overlap"])
        np.testing.assert_allclose(time, expected, atol=1e-9)

    def test_fallback_dt_uses_formula(self):
        frame = _make_frame_no_bounds(30)
        dt, _ = _derive_dt_and_time(frame, MNPS_CFG_LEGACY)
        cfg_dt = MNPS_CFG_LEGACY["window_sec"] * (1.0 - MNPS_CFG_LEGACY["overlap"])
        assert dt == pytest.approx(cfg_dt, abs=1e-9)

    def test_fallback_time_step_equals_formula_step(self):
        frame = _make_frame_no_bounds(10)
        cfg = {"window_sec": 30.0, "overlap": 0.0}
        _, time = _derive_dt_and_time(frame, cfg)
        np.testing.assert_allclose(np.diff(time), 30.0, atol=1e-9)

    def test_only_t_start_present_still_uses_formula(self):
        """If only t_start present (no t_end), fall back to formula for time."""
        from mndm.projection import build_time_index
        n = 15
        frame = pd.DataFrame({
            "epoch_id": np.arange(n),
            "t_start": np.arange(n, dtype=float) * 2.0,  # t_end absent
        })
        dt, time = _derive_dt_and_time(frame, MNPS_CFG_LEGACY)
        expected = build_time_index(n, MNPS_CFG_LEGACY["window_sec"], MNPS_CFG_LEGACY["overlap"])
        np.testing.assert_allclose(time, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# Regression guard: the exact pre-fix / post-fix values for ds005555 sub-1
# ---------------------------------------------------------------------------

class TestDs005555Sub1RegressionValues:
    """Pin exact expected time-axis values for the ds005555 6 s/2 s config."""

    @pytest.fixture
    def frame_sub1(self):
        """Replicate the first 10 windows of the sub-1 features."""
        return _make_frame(10, length_s=6.0, step_s=2.0)

    def test_first_time_value_is_3_not_4(self, frame_sub1):
        """Pre-fix: time[0] = 4.0.  Post-fix: time[0] = 3.0."""
        _, time = _derive_dt_and_time(frame_sub1, MNPS_CFG_LEGACY)
        assert time[0] == pytest.approx(3.0, abs=1e-9), (
            f"time[0]={time[0]}; regression: pre-fix value was 4.0"
        )

    def test_time_step_is_2_not_4(self, frame_sub1):
        """Pre-fix: step = 4.0 s.  Post-fix: step = 2.0 s."""
        _, time = _derive_dt_and_time(frame_sub1, MNPS_CFG_LEGACY)
        step = float(np.median(np.diff(time)))
        assert step == pytest.approx(2.0, abs=1e-9), (
            f"time step={step}; regression: pre-fix was 4.0"
        )

    def test_dt_is_2_not_4(self, frame_sub1):
        """Pre-fix: dt = 4.0 s.  Post-fix: dt = 2.0 s."""
        dt, _ = _derive_dt_and_time(frame_sub1, MNPS_CFG_LEGACY)
        assert dt == pytest.approx(2.0, abs=1e-9), (
            f"dt={dt}; regression: pre-fix was 4.0"
        )
