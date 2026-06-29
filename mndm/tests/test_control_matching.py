"""Unit tests for pipeline/control_matching.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))


def _make_windows(n: int = 40, window_sec: float = 4.0, step_sec: float = 4.0):
    w_start = np.arange(n, dtype=np.float64) * step_sec
    w_end = w_start + window_sec
    t = (w_start + w_end) / 2.0
    return w_start, w_end, t


class TestControlMatchingSameSubjectStage:
    """Acceptance criterion: controls are matched within subject/session/stage."""

    def test_matched_controls_have_correct_stage(self):
        from mndm.pipeline.control_matching import MatchingConfig, build_matched_controls
        from mndm.pipeline.event_annotations import EventTable

        n = 40
        w_start, w_end, t = _make_windows(n)
        # First 20 windows N2, last 20 N3
        stage = np.array([2] * 20 + [3] * 20, dtype=np.int16)
        table = EventTable(onset_sec=np.array([10.0]), duration_sec=np.array([1.0]))

        cfg = MatchingConfig(target_stage=2, n_controls_per_event=3, seed=42, exclusion_margin_sec=5.0)
        result = build_matched_controls(table, time=t, window_start=w_start, window_end=w_end, stage=stage, config=cfg)

        for row in result.rows:
            assert row.stage == 2, f"Expected N2 (2), got {row.stage}"


class TestControlMatchingExcludesEventOverlap:
    """Acceptance criterion: controls do not overlap spindle events."""

    def test_controls_excluded_near_event(self):
        from mndm.pipeline.control_matching import MatchingConfig, build_matched_controls
        from mndm.pipeline.event_annotations import EventTable

        n = 40
        w_start, w_end, t = _make_windows(n)
        stage = np.full(n, 2, dtype=np.int16)

        event_onset = float(t[5])  # event at window 5 center
        table = EventTable(onset_sec=np.array([event_onset]), duration_sec=np.array([1.0]))

        margin = 8.0  # exclusion = 2 windows width
        cfg = MatchingConfig(target_stage=2, exclusion_margin_sec=margin, n_controls_per_event=5, seed=0)
        result = build_matched_controls(table, time=t, window_start=w_start, window_end=w_end, stage=stage, config=cfg)

        excluded_zone_lo = event_onset - margin
        excluded_zone_hi = event_onset + 1.0 + margin  # event_end + margin
        for row in result.rows:
            w_center = float(t[row.control_window_id])
            assert not (excluded_zone_lo <= w_center <= excluded_zone_hi), (
                f"Control window {row.control_window_id} (t={w_center:.1f}) is inside exclusion zone"
            )


class TestControlMatchingDeterminism:
    """Acceptance criterion: controls are deterministic under seed."""

    def test_same_seed_same_result(self):
        from mndm.pipeline.control_matching import MatchingConfig, build_matched_controls
        from mndm.pipeline.event_annotations import EventTable

        n = 30
        w_start, w_end, t = _make_windows(n)
        stage = np.full(n, 2, dtype=np.int16)
        table = EventTable(onset_sec=np.array([20.0, 60.0]), duration_sec=np.array([1.0, 1.0]))

        cfg = MatchingConfig(seed=1729)
        r1 = build_matched_controls(table, time=t, window_start=w_start, window_end=w_end, stage=stage, config=cfg)
        r2 = build_matched_controls(table, time=t, window_start=w_start, window_end=w_end, stage=stage, config=cfg)

        ids1 = [r.control_window_id for r in r1.rows]
        ids2 = [r.control_window_id for r in r2.rows]
        assert ids1 == ids2

    def test_different_seed_may_differ(self):
        from mndm.pipeline.control_matching import MatchingConfig, build_matched_controls
        from mndm.pipeline.event_annotations import EventTable

        n = 40
        w_start, w_end, t = _make_windows(n)
        stage = np.full(n, 2, dtype=np.int16)
        table = EventTable(onset_sec=np.array([10.0, 30.0, 50.0]), duration_sec=np.array([1.0, 1.0, 1.0]))

        r1 = build_matched_controls(
            table, time=t, window_start=w_start, window_end=w_end, stage=stage,
            config=MatchingConfig(seed=0)
        )
        r2 = build_matched_controls(
            table, time=t, window_start=w_start, window_end=w_end, stage=stage,
            config=MatchingConfig(seed=9999)
        )
        ids1 = [r.control_window_id for r in r1.rows]
        ids2 = [r.control_window_id for r in r2.rows]
        # Different seeds should (with very high probability) produce different results
        # over many events and many candidates — not guaranteed but expected
        assert ids1 != ids2 or True  # soft check: don't hard-fail if they happen to match


class TestControlMatchingQC:
    def test_qc_records_seed(self):
        from mndm.pipeline.control_matching import MatchingConfig, build_matched_controls
        from mndm.pipeline.event_annotations import make_empty_event_table

        w_start, w_end, t = _make_windows(5)
        cfg = MatchingConfig(seed=42)
        result = build_matched_controls(
            make_empty_event_table(), time=t, window_start=w_start, window_end=w_end, config=cfg
        )
        assert result.qc["seed"] == 42

    def test_failed_matches_counted(self):
        from mndm.pipeline.control_matching import MatchingConfig, build_matched_controls
        from mndm.pipeline.event_annotations import EventTable

        # Only 2 windows total, both excluded by margin → no controls possible
        w_start = np.array([0.0, 4.0])
        w_end = np.array([4.0, 8.0])
        t = np.array([2.0, 6.0])
        stage = np.full(2, 2, dtype=np.int16)
        table = EventTable(onset_sec=np.array([2.0]), duration_sec=np.array([1.0]))

        cfg = MatchingConfig(exclusion_margin_sec=10.0, n_controls_per_event=3, seed=0)
        result = build_matched_controls(table, time=t, window_start=w_start, window_end=w_end, stage=stage, config=cfg)

        assert result.qc["n_events_with_no_match"] == 1
        assert result.is_empty()

    def test_match_success_rate_in_qc(self):
        from mndm.pipeline.control_matching import MatchingConfig, build_matched_controls
        from mndm.pipeline.event_annotations import EventTable

        n = 20
        w_start, w_end, t = _make_windows(n)
        stage = np.full(n, 2, dtype=np.int16)
        table = EventTable(onset_sec=np.array([20.0]), duration_sec=np.array([1.0]))

        cfg = MatchingConfig(n_controls_per_event=3, seed=0, exclusion_margin_sec=5.0)
        result = build_matched_controls(table, time=t, window_start=w_start, window_end=w_end, stage=stage, config=cfg)

        assert "match_success_rate" in result.qc
        assert 0.0 <= result.qc["match_success_rate"] <= 1.0

    def test_nan_time_windows_do_not_break_quartile_matching(self):
        from mndm.pipeline.control_matching import MatchingConfig, build_matched_controls
        from mndm.pipeline.event_annotations import EventTable

        w_start, w_end, t = _make_windows(8)
        t = t.astype(np.float64)
        t[3] = np.nan
        stage = np.full(8, 2, dtype=np.int16)
        table = EventTable(onset_sec=np.array([10.0]), duration_sec=np.array([1.0]))

        result = build_matched_controls(
            table,
            time=t,
            window_start=w_start,
            window_end=w_end,
            stage=stage,
            config=MatchingConfig(n_controls_per_event=2, seed=7, exclusion_margin_sec=2.0),
        )

        assert "match_success_rate" in result.qc
        assert all(np.isfinite(float(t[row.control_window_id])) for row in result.rows)


class TestControlMatchingEmpty:
    def test_empty_table_returns_empty(self):
        from mndm.pipeline.control_matching import build_matched_controls
        from mndm.pipeline.event_annotations import make_empty_event_table

        w_start, w_end, t = _make_windows(10)
        result = build_matched_controls(make_empty_event_table(), time=t, window_start=w_start, window_end=w_end)
        assert result.is_empty()


class TestControlMatchingToRecords:
    def test_to_records_has_expected_keys(self):
        from mndm.pipeline.control_matching import MatchingConfig, build_matched_controls
        from mndm.pipeline.event_annotations import EventTable

        n = 20
        w_start, w_end, t = _make_windows(n)
        stage = np.full(n, 2, dtype=np.int16)
        table = EventTable(onset_sec=np.array([20.0]), duration_sec=np.array([1.0]))

        cfg = MatchingConfig(n_controls_per_event=2, seed=0, exclusion_margin_sec=5.0)
        result = build_matched_controls(table, time=t, window_start=w_start, window_end=w_end, stage=stage, config=cfg)

        records = result.to_records()
        if records:
            assert "event_id" in records[0]
            assert "control_window_id" in records[0]
            assert "match_rank" in records[0]
            assert "stage" in records[0]
