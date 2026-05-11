"""Unit tests for pipeline/event_alignment.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))


def _make_windows(n: int = 20, window_sec: float = 4.0, step_sec: float = 4.0):
    """Return (window_start, window_end, time) for n non-overlapping windows."""
    w_start = np.arange(n, dtype=np.float64) * step_sec
    w_end = w_start + window_sec
    t = (w_start + w_end) / 2.0
    return w_start, w_end, t


def _make_stage_n2(n: int = 20) -> np.ndarray:
    return np.full(n, 2, dtype=np.int16)


class TestAlignmentConfigParsing:
    def test_default_bins_count(self):
        from mndm.pipeline.event_alignment import AlignmentConfig

        cfg = AlignmentConfig()
        assert len(cfg.bins) == 5
        labels = [b.label for b in cfg.bins]
        assert "pre_far" in labels
        assert "event" in labels

    def test_parse_bins_from_config(self):
        from mndm.pipeline.event_alignment import alignment_config_from_dict

        raw = {
            "reference": "peak",
            "bins_sec": {
                "pre": [-10, 0],
                "event": [0, 2],
                "post": [2, 10],
            },
            "min_overlap_fraction": 0.1,
        }
        cfg = alignment_config_from_dict(raw)
        assert cfg.reference == "peak"
        assert len(cfg.bins) == 3
        assert cfg.overlap_threshold == pytest.approx(0.1)


class TestAlignmentPeakReference:
    """Acceptance criterion: event-window alignment peak reference."""

    def test_alignment_uses_peak_when_available(self):
        from mndm.pipeline.event_alignment import AlignmentConfig, align_events_to_windows
        from mndm.pipeline.event_annotations import EventTable

        w_start, w_end, t = _make_windows(20)
        # Peak at t=20, onset at t=10 — should centre on t=20
        table = EventTable(
            onset_sec=np.array([10.0]),
            peak_sec=np.array([20.0]),
            duration_sec=np.array([1.0]),
            source=np.array(["test"], dtype=object),
        )
        cfg = AlignmentConfig(reference="peak")
        result = align_events_to_windows(table, window_start=w_start, window_end=w_end, time=t, config=cfg)
        # Window at center=20 should have rel_time close to 0
        event_windows = [r for r in result.rows if r.bin_label == "event"]
        assert len(event_windows) > 0
        # rel_time for the window at t≈20 should be ~0
        near_zero = [r for r in result.rows if abs(r.rel_time_sec) < 3.0]
        assert len(near_zero) > 0

    def test_alignment_falls_back_to_onset(self):
        from mndm.pipeline.event_alignment import AlignmentConfig, align_events_to_windows
        from mndm.pipeline.event_annotations import EventTable

        w_start, w_end, t = _make_windows(20)
        table = EventTable(
            onset_sec=np.array([20.0]),
            duration_sec=np.array([1.0]),
        )
        cfg = AlignmentConfig(reference="peak")  # no peak_sec in table
        result = align_events_to_windows(table, window_start=w_start, window_end=w_end, time=t, config=cfg)
        assert result.qc["n_events_aligned"] == 1


class TestAlignmentOverlap:
    """Acceptance criterion: event-window alignment overlap reference."""

    def test_overlap_frac_correct(self):
        from mndm.pipeline.event_alignment import align_events_to_windows
        from mndm.pipeline.event_annotations import EventTable

        # One 4-second window [8, 12], event [10, 11] → overlap 1s, frac 0.25
        w_start = np.array([8.0])
        w_end = np.array([12.0])
        t = np.array([10.0])
        table = EventTable(
            onset_sec=np.array([10.0]),
            duration_sec=np.array([1.0]),
        )
        result = align_events_to_windows(table, window_start=w_start, window_end=w_end, time=t)
        assert len(result.rows) == 1
        assert result.rows[0].overlap_sec == pytest.approx(1.0)
        assert result.rows[0].overlap_frac == pytest.approx(0.25)

    def test_is_event_window_threshold(self):
        from mndm.pipeline.event_alignment import AlignmentConfig, align_events_to_windows
        from mndm.pipeline.event_annotations import EventTable

        w_start = np.array([0.0, 4.0, 8.0])
        w_end = np.array([4.0, 8.0, 12.0])
        t = np.array([2.0, 6.0, 10.0])
        # Event [5, 6] overlaps window [4,8] by 1s (frac=0.25)
        table = EventTable(onset_sec=np.array([5.0]), duration_sec=np.array([1.0]))

        cfg_low = AlignmentConfig(overlap_threshold=0.1)
        cfg_high = AlignmentConfig(overlap_threshold=0.5)

        res_low = align_events_to_windows(table, window_start=w_start, window_end=w_end, time=t, config=cfg_low)
        res_high = align_events_to_windows(table, window_start=w_start, window_end=w_end, time=t, config=cfg_high)

        low_event = [r for r in res_low.rows if r.is_event_window]
        high_event = [r for r in res_high.rows if r.is_event_window]

        assert len(low_event) > 0
        assert len(high_event) == 0


class TestAlignmentEmpty:
    def test_empty_table_returns_empty(self):
        from mndm.pipeline.event_alignment import align_events_to_windows
        from mndm.pipeline.event_annotations import make_empty_event_table

        w_start, w_end, t = _make_windows(10)
        result = align_events_to_windows(make_empty_event_table(), window_start=w_start, window_end=w_end, time=t)
        assert result.is_empty()

    def test_qc_populated_on_empty(self):
        from mndm.pipeline.event_alignment import align_events_to_windows
        from mndm.pipeline.event_annotations import make_empty_event_table

        w_start, w_end, t = _make_windows(5)
        result = align_events_to_windows(make_empty_event_table(), window_start=w_start, window_end=w_end, time=t)
        assert result.qc["n_events_input"] == 0


class TestAlignmentStageTransitionExclusion:
    def test_stage_transition_excludes_nearby_event(self):
        from mndm.pipeline.event_alignment import AlignmentConfig, align_events_to_windows
        from mndm.pipeline.event_annotations import EventTable

        n = 20
        w_start, w_end, t = _make_windows(n)
        # Stage changes at window 10 (t=42s)
        stage = np.full(n, 2, dtype=np.int16)
        stage[10:] = 3  # transition at index 10
        t_transition = t[10]

        # Event right at transition — should be excluded
        table = EventTable(onset_sec=np.array([t_transition]), duration_sec=np.array([1.0]))
        cfg = AlignmentConfig(stage_transition_margin_sec=30.0)
        result = align_events_to_windows(
            table, window_start=w_start, window_end=w_end, time=t, stage=stage, config=cfg
        )
        assert result.qc["n_events_excluded_stage_transition"] == 1
        assert result.qc["n_events_aligned"] == 0

    def test_event_far_from_transition_is_kept(self):
        from mndm.pipeline.event_alignment import AlignmentConfig, align_events_to_windows
        from mndm.pipeline.event_annotations import EventTable

        n = 30
        w_start, w_end, t = _make_windows(n)
        stage = np.full(n, 2, dtype=np.int16)
        stage[20:] = 3
        t_transition = t[20]

        # Event far from transition (200s away)
        far_onset = t_transition - 200.0
        if far_onset < 0:
            far_onset = t_transition + 200.0
        table = EventTable(onset_sec=np.array([far_onset]), duration_sec=np.array([1.0]))
        cfg = AlignmentConfig(stage_transition_margin_sec=30.0)
        result = align_events_to_windows(
            table, window_start=w_start, window_end=w_end, time=t, stage=stage, config=cfg
        )
        assert result.qc["n_events_excluded_stage_transition"] == 0


class TestAlignmentToRecords:
    def test_to_records_serializable(self):
        from mndm.pipeline.event_alignment import align_events_to_windows
        from mndm.pipeline.event_annotations import EventTable

        w_start, w_end, t = _make_windows(10)
        table = EventTable(onset_sec=np.array([5.0]), duration_sec=np.array([1.0]))
        result = align_events_to_windows(table, window_start=w_start, window_end=w_end, time=t)
        records = result.to_records()
        assert isinstance(records, list)
        if records:
            assert "event_id" in records[0]
            assert "window_id" in records[0]
            assert "bin_label" in records[0]
            assert "overlap_frac" in records[0]
