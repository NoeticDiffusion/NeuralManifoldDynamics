"""Tests for block_windows.py and block_native_config.py.

Covers M1 (window geometry) and M2 (block-source inference).
"""

from __future__ import annotations

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_block(block_id: int, start: float, end: float, stage_code: int = 1):
    """Minimal block-like object compatible with generate_block_windows."""
    from mndm.pipeline.stage_blocking import StageBlockInterval
    return StageBlockInterval(
        start_sec=start,
        end_sec=end,
        stage_code=stage_code,
        source_event_idx=0,
        block_id=block_id,
    )


def _make_events_df(rows):
    """Build a minimal events DataFrame from a list of dicts."""
    import pandas as pd
    return pd.DataFrame(rows)


# ===========================================================================
# M1 — BlockWindowSpec / generate_block_windows
# ===========================================================================

class TestSlidingWindows:
    def test_basic_geometry(self):
        from mndm.pipeline.block_windows import BlockWindowSpec, generate_block_windows
        block = _make_block(0, 0.0, 20.0)
        spec = BlockWindowSpec(kind="sliding", window_length_sec=4.0, step_sec=2.0)
        rows = generate_block_windows([block], spec)

        assert len(rows) == 9, f"Expected 9 windows, got {len(rows)}"
        assert rows[0].window_start_sec == pytest.approx(0.0)
        assert rows[0].window_end_sec == pytest.approx(4.0)
        assert rows[-1].window_start_sec == pytest.approx(16.0)
        assert rows[-1].window_end_sec == pytest.approx(20.0)

        for r in rows:
            assert 0.0 <= r.relative_pos_0_1 <= 1.0

    def test_block_too_short_for_one_window(self):
        from mndm.pipeline.block_windows import BlockWindowSpec, generate_block_windows
        block = _make_block(0, 0.0, 3.0)
        spec = BlockWindowSpec(kind="sliding", window_length_sec=4.0, step_sec=2.0)
        rows = generate_block_windows([block], spec)
        assert rows == []

    def test_distance_to_block_end(self):
        from mndm.pipeline.block_windows import BlockWindowSpec, generate_block_windows
        block = _make_block(0, 0.0, 20.0)
        spec = BlockWindowSpec(kind="sliding", window_length_sec=4.0, step_sec=2.0)
        rows = generate_block_windows([block], spec)
        last = rows[-1]
        # Last window center at 18.0 → distance_to_block_end_sec = 2.0
        assert last.window_center_sec == pytest.approx(18.0)
        assert last.distance_to_block_end_sec == pytest.approx(2.0)

    def test_relative_position_correctness(self):
        from mndm.pipeline.block_windows import BlockWindowSpec, generate_block_windows
        block = _make_block(0, 10.0, 30.0)
        spec = BlockWindowSpec(kind="sliding", window_length_sec=4.0, step_sec=4.0)
        rows = generate_block_windows([block], spec)
        # First window: center at 12.0, relative_time = 2.0, block_dur = 20
        assert rows[0].relative_time_in_block_sec == pytest.approx(2.0)
        assert rows[0].relative_pos_0_1 == pytest.approx(0.1)

    def test_adjacent_blocks_no_leakage(self):
        from mndm.pipeline.block_windows import BlockWindowSpec, generate_block_windows
        b0 = _make_block(0, 0.0, 10.0)
        b1 = _make_block(1, 10.0, 20.0)
        spec = BlockWindowSpec(kind="sliding", window_length_sec=4.0, step_sec=4.0)
        rows = generate_block_windows([b0, b1], spec)
        # No window starts before 0 or ends after 20
        for r in rows:
            assert r.window_start_sec >= 0.0
            assert r.window_end_sec <= 20.0
        # No window spans the block boundary 10.0
        for r in rows:
            assert not (r.window_start_sec < 10.0 and r.window_end_sec > 10.0), (
                f"Window [{r.window_start_sec}, {r.window_end_sec}] spans boundary"
            )

    def test_min_windows_per_block_filter(self):
        from mndm.pipeline.block_windows import BlockWindowSpec, generate_block_windows
        # Block fits 1 window but min_windows_per_block=2 → should be dropped
        block = _make_block(0, 0.0, 5.0)
        spec = BlockWindowSpec(
            kind="sliding", window_length_sec=4.0, step_sec=2.0,
            min_windows_per_block=2,
        )
        rows = generate_block_windows([block], spec)
        assert rows == []

    def test_stage_code_inherited(self):
        from mndm.pipeline.block_windows import BlockWindowSpec, generate_block_windows
        block = _make_block(0, 0.0, 10.0, stage_code=53)
        spec = BlockWindowSpec(kind="sliding", window_length_sec=4.0, step_sec=4.0)
        rows = generate_block_windows([block], spec)
        assert all(r.stage_code == 53 for r in rows)

    def test_block_id_propagated(self):
        from mndm.pipeline.block_windows import BlockWindowSpec, generate_block_windows
        b0 = _make_block(0, 0.0, 10.0)
        b1 = _make_block(7, 20.0, 30.0)
        spec = BlockWindowSpec(kind="sliding", window_length_sec=4.0, step_sec=4.0)
        rows = generate_block_windows([b0, b1], spec)
        b0_rows = [r for r in rows if r.block_id == 0]
        b7_rows = [r for r in rows if r.block_id == 7]
        assert len(b0_rows) > 0
        assert len(b7_rows) > 0
        assert all(r.block_id == 0 for r in b0_rows)
        assert all(r.block_id == 7 for r in b7_rows)


class TestTailWindows:
    def test_tail_anchor(self):
        from mndm.pipeline.block_windows import BlockWindowSpec, generate_block_windows
        block = _make_block(0, 0.0, 20.0)
        spec = BlockWindowSpec(
            kind="tail", window_length_sec=4.0, step_sec=2.0, tail_sec=8.0,
        )
        rows = generate_block_windows([block], spec)
        # tail_start = 20 - 8 = 12; windows at [12,16], [14,18], [16,20]
        assert len(rows) == 3
        assert all(r.window_start_sec >= 12.0 for r in rows)
        assert all(r.partition_label == "tail" for r in rows)

    def test_tail_exceeds_block_duration(self):
        from mndm.pipeline.block_windows import BlockWindowSpec, generate_block_windows
        block = _make_block(0, 0.0, 6.0)
        spec = BlockWindowSpec(
            kind="tail", window_length_sec=4.0, step_sec=2.0, tail_sec=20.0,
        )
        rows = generate_block_windows([block], spec)
        # tail_start = max(0, 6-20) = 0 → covers whole block
        assert all(r.window_start_sec >= 0.0 for r in rows)
        assert all(r.window_end_sec <= 6.0 for r in rows)


class TestPostOffsetWindows:
    def test_basic_post_offset(self):
        from mndm.pipeline.block_windows import BlockWindowSpec, generate_block_windows
        block = _make_block(0, 10.0, 30.0)
        spec = BlockWindowSpec(
            kind="post_offset",
            window_length_sec=4.0,
            step_sec=4.0,
            post_offset_bins=(
                ("post_early", 0.0, 8.0),
                ("post_late", 8.0, 16.0),
            ),
        )
        rows = generate_block_windows([block], spec)
        # All windows start at or after block_end (30.0)
        assert all(r.window_start_sec >= 30.0 for r in rows)
        assert all(r.is_post_offset for r in rows)
        labels = {r.partition_label for r in rows}
        assert "post_early" in labels
        assert "post_late" in labels

    def test_post_offset_window_count(self):
        from mndm.pipeline.block_windows import BlockWindowSpec, generate_block_windows
        block = _make_block(0, 0.0, 10.0)
        spec = BlockWindowSpec(
            kind="post_offset",
            window_length_sec=4.0,
            step_sec=4.0,
            post_offset_bins=(("bin_a", 0.0, 8.0),),
        )
        rows = generate_block_windows([block], spec)
        # [10,14] and [14,18] → 2 windows
        assert len(rows) == 2


class TestPartitionedWindows:
    def test_two_partitions(self):
        from mndm.pipeline.block_windows import BlockWindowSpec, generate_block_windows
        block = _make_block(0, 0.0, 20.0)
        spec = BlockWindowSpec(
            kind="partitioned",
            window_length_sec=2.0,
            step_sec=2.0,
            partitions=(
                ("early", 0.0, 8.0),
                ("tail", -8.0, 0.0),
            ),
        )
        rows = generate_block_windows([block], spec)
        early = [r for r in rows if r.partition_label == "early"]
        tail = [r for r in rows if r.partition_label == "tail"]
        assert len(early) > 0
        assert len(tail) > 0
        assert all(r.window_start_sec < 8.0 for r in early)
        assert all(r.window_start_sec >= 12.0 for r in tail)

    def test_empty_partitions_falls_back_to_sliding(self):
        from mndm.pipeline.block_windows import BlockWindowSpec, generate_block_windows
        block = _make_block(0, 0.0, 20.0)
        spec_part = BlockWindowSpec(
            kind="partitioned", window_length_sec=4.0, step_sec=4.0, partitions=(),
        )
        spec_slide = BlockWindowSpec(
            kind="sliding", window_length_sec=4.0, step_sec=4.0,
        )
        assert len(generate_block_windows([block], spec_part)) == len(
            generate_block_windows([block], spec_slide)
        )


# ===========================================================================
# M2 — BlockNativeDatasetConfig / infer_blocks_from_events
# ===========================================================================

class TestBlockNativeConfig:
    def test_analysis_mode_default(self):
        from mndm.pipeline.block_native_config import analysis_mode_from_config
        assert analysis_mode_from_config({}) == "global"
        assert analysis_mode_from_config({"analysis_mode": "block_native"}) == "block_native"

    def test_dataset_config_enabled(self):
        from mndm.pipeline.block_native_config import block_native_dataset_config_from_config
        config = {
            "block_native": {
                "datasets": {
                    "ds006036": {
                        "enabled": True,
                        "source": {"kind": "stage_blocking"},
                        "window_profile": {"kind": "sliding", "window_length_sec": 4.0},
                    }
                }
            }
        }
        cfg = block_native_dataset_config_from_config(config, "ds006036")
        assert cfg.enabled is True
        assert cfg.source.kind == "stage_blocking"
        assert cfg.window_profile.window_length_sec == pytest.approx(4.0)

    def test_dataset_config_absent(self):
        from mndm.pipeline.block_native_config import block_native_dataset_config_from_config
        cfg = block_native_dataset_config_from_config({}, "ds_unknown")
        assert cfg.enabled is False

    def test_source_kind_duration_events_parsed(self):
        from mndm.pipeline.block_native_config import block_native_dataset_config_from_config
        config = {
            "block_native": {
                "datasets": {
                    "ds003490": {
                        "enabled": True,
                        "source": {
                            "kind": "duration_events",
                            "label_column": "trial_type",
                            "block_event_labels": ["Eyes Closed: Every 1000 ms"],
                            "block_event_stage_codes": {"Eyes Closed: Every 1000 ms": 10},
                            "min_block_sec": 30.0,
                        },
                        "window_profile": {"kind": "sliding"},
                    }
                }
            }
        }
        cfg = block_native_dataset_config_from_config(config, "ds003490")
        assert cfg.source.kind == "duration_events"
        assert "Eyes Closed: Every 1000 ms" in cfg.source.block_event_labels
        assert cfg.source.min_block_sec == pytest.approx(30.0)

    def test_source_kind_task_phase_parsed(self):
        from mndm.pipeline.block_native_config import block_native_dataset_config_from_config
        config = {
            "block_native": {
                "datasets": {
                    "ds003509": {
                        "enabled": True,
                        "source": {
                            "kind": "task_phase",
                            "label_column": "event_subtype_role",
                            "phase_prefixes": {"training": "training_", "test": "test_"},
                            "gap_tolerance_sec": 10.0,
                        },
                        "window_profile": {"kind": "sliding"},
                    }
                }
            }
        }
        cfg = block_native_dataset_config_from_config(config, "ds003509")
        assert cfg.source.kind == "task_phase"
        phase_dict = dict(cfg.source.phase_prefixes)
        assert phase_dict.get("training") == "training_"
        assert phase_dict.get("test") == "test_"
        assert cfg.source.gap_tolerance_sec == pytest.approx(10.0)

    def test_window_profile_post_offset_bins(self):
        from mndm.pipeline.block_native_config import block_native_dataset_config_from_config
        config = {
            "block_native": {
                "datasets": {
                    "ds_test": {
                        "enabled": True,
                        "source": {"kind": "stage_blocking"},
                        "window_profile": {
                            "kind": "post_offset",
                            "window_length_sec": 4.0,
                            "post_offset_bins": {
                                "post_early": [0.0, 8.0],
                                "post_late": [8.0, 16.0],
                            },
                        },
                    }
                }
            }
        }
        cfg = block_native_dataset_config_from_config(config, "ds_test")
        bins = dict((b[0], (b[1], b[2])) for b in cfg.window_profile.post_offset_bins)
        assert "post_early" in bins
        assert bins["post_early"] == pytest.approx((0.0, 8.0))

    def test_window_profile_partitions(self):
        from mndm.pipeline.block_native_config import block_native_dataset_config_from_config
        config = {
            "block_native": {
                "datasets": {
                    "ds_test": {
                        "enabled": True,
                        "source": {"kind": "stage_blocking"},
                        "window_profile": {
                            "kind": "partitioned",
                            "partitions": {
                                "early": [0.0, 8.0],
                                "tail": [-8.0, 0.0],
                            },
                        },
                    }
                }
            }
        }
        cfg = block_native_dataset_config_from_config(config, "ds_test")
        parts = dict((p[0], (p[1], p[2])) for p in cfg.window_profile.partitions)
        assert "early" in parts
        assert "tail" in parts
        assert parts["tail"] == pytest.approx((-8.0, 0.0))

    def test_named_profile_tail8(self):
        from mndm.pipeline.block_native_config import block_native_dataset_config_from_config
        config = {
            "block_native": {
                "datasets": {
                    "ds_test": {
                        "enabled": True,
                        "source": {"kind": "stage_blocking"},
                        "window_profile": {
                            "profile": "tail8",
                            "window_length_sec": 4.0,
                            "step_sec": 2.0,
                        },
                    }
                }
            }
        }
        cfg = block_native_dataset_config_from_config(config, "ds_test")
        assert cfg.window_profile.kind == "tail"
        assert cfg.window_profile.tail_sec == pytest.approx(8.0)
        assert cfg.window_profile.named_profile == "tail8"

    def test_named_profile_post_offset(self):
        from mndm.pipeline.block_native_config import block_native_dataset_config_from_config
        config = {
            "block_native": {
                "datasets": {
                    "ds_test": {
                        "enabled": True,
                        "source": {"kind": "stage_blocking"},
                        "window_profile": {
                            "named_profile": "post_offset_0_8",
                            "window_length_sec": 2.0,
                            "step_sec": 2.0,
                        },
                    }
                }
            }
        }
        cfg = block_native_dataset_config_from_config(config, "ds_test")
        assert cfg.window_profile.kind == "post_offset"
        bins = dict((name, (lo, hi)) for name, lo, hi in cfg.window_profile.post_offset_bins)
        assert bins["post_offset_0_8"] == pytest.approx((0.0, 8.0))
        assert cfg.window_profile.named_profile == "post_offset_0_8"


class TestDurationEventsInference:
    def test_basic_block_from_explicit_duration(self):
        from mndm.pipeline.block_native_config import (
            BlockSourceConfig,
            infer_blocks_from_events,
        )
        df = _make_events_df([
            {"onset": 10.0, "duration": 60.0, "trial_type": "Eyes Closed: Every 1000 ms"},
            {"onset": 80.0, "duration": 60.0, "trial_type": "Eyes Open: Every 1000 ms"},
        ])
        cfg = BlockSourceConfig(
            kind="duration_events",
            label_column="trial_type",
            block_event_labels=("Eyes Closed: Every 1000 ms", "Eyes Open: Every 1000 ms"),
            block_event_stage_codes=(
                ("Eyes Closed: Every 1000 ms", 10),
                ("Eyes Open: Every 1000 ms", 11),
            ),
            min_block_sec=30.0,
            max_block_sec=600.0,
        )
        blocks = infer_blocks_from_events(df, cfg)
        assert len(blocks) == 2
        assert blocks[0].start_sec == pytest.approx(10.0)
        assert blocks[0].end_sec == pytest.approx(70.0)
        assert blocks[0].stage_code == 10
        assert blocks[1].stage_code == 11

    def test_max_block_sec_cap_applied(self):
        from mndm.pipeline.block_native_config import (
            BlockSourceConfig,
            infer_blocks_from_events,
        )
        df = _make_events_df([
            {"onset": 0.0, "duration": 700.0, "trial_type": "Eyes Closed: Every 1000 ms"},
        ])
        cfg = BlockSourceConfig(
            kind="duration_events",
            label_column="trial_type",
            block_event_labels=("Eyes Closed: Every 1000 ms",),
            block_event_stage_codes=(("Eyes Closed: Every 1000 ms", 10),),
            min_block_sec=2.0,
            max_block_sec=600.0,
        )
        blocks = infer_blocks_from_events(df, cfg)
        assert len(blocks) == 1
        assert blocks[0].end_sec == pytest.approx(600.0)

    def test_too_short_block_skipped(self):
        from mndm.pipeline.block_native_config import (
            BlockSourceConfig,
            infer_blocks_from_events,
        )
        df = _make_events_df([
            {"onset": 0.0, "duration": 5.0, "trial_type": "Eyes Closed: Every 1000 ms"},
        ])
        cfg = BlockSourceConfig(
            kind="duration_events",
            label_column="trial_type",
            block_event_labels=("Eyes Closed: Every 1000 ms",),
            block_event_stage_codes=(("Eyes Closed: Every 1000 ms", 10),),
            min_block_sec=30.0,
        )
        blocks = infer_blocks_from_events(df, cfg)
        assert blocks == []

    def test_unlisted_label_skipped(self):
        from mndm.pipeline.block_native_config import (
            BlockSourceConfig,
            infer_blocks_from_events,
        )
        df = _make_events_df([
            {"onset": 0.0, "duration": 60.0, "trial_type": "Unknown label"},
        ])
        cfg = BlockSourceConfig(
            kind="duration_events",
            label_column="trial_type",
            block_event_labels=("Eyes Closed: Every 1000 ms",),
            block_event_stage_codes=(("Eyes Closed: Every 1000 ms", 10),),
            min_block_sec=2.0,
        )
        blocks = infer_blocks_from_events(df, cfg)
        assert blocks == []


class TestTaskPhaseInference:
    def test_two_phase_separation(self):
        from mndm.pipeline.block_native_config import (
            BlockSourceConfig,
            infer_blocks_from_events,
        )
        events = (
            [{"onset": float(t), "event_subtype_role": "training_stimulus"} for t in range(3)]
            + [{"onset": float(100 + t), "event_subtype_role": "test_stimulus"} for t in range(3)]
        )
        df = _make_events_df(events)
        cfg = BlockSourceConfig(
            kind="task_phase",
            label_column="event_subtype_role",
            phase_prefixes=(("training", "training_"), ("test", "test_")),
            gap_tolerance_sec=5.0,
            min_block_sec=2.0,
        )
        blocks = infer_blocks_from_events(df, cfg)
        assert len(blocks) == 2
        derived_froms = {b.derived_from for b in blocks}
        assert derived_froms == {"task_phase"}

    def test_gap_breaks_same_phase(self):
        from mndm.pipeline.block_native_config import (
            BlockSourceConfig,
            infer_blocks_from_events,
        )
        events = (
            [{"onset": float(t), "event_subtype_role": "training_stimulus"} for t in [0, 1]]
            + [{"onset": float(t), "event_subtype_role": "training_stimulus"} for t in [50, 51]]
        )
        df = _make_events_df(events)
        cfg = BlockSourceConfig(
            kind="task_phase",
            label_column="event_subtype_role",
            phase_prefixes=(("training", "training_"),),
            gap_tolerance_sec=5.0,
            min_block_sec=0.5,
        )
        blocks = infer_blocks_from_events(df, cfg)
        assert len(blocks) == 2

    def test_choose_match_phases(self):
        from mndm.pipeline.block_native_config import (
            BlockSourceConfig,
            infer_blocks_from_events,
        )
        events = (
            [{"onset": float(t), "event_subtype_role": "choose_stimulus"} for t in range(5)]
            + [{"onset": float(100 + t), "event_subtype_role": "match_stimulus"} for t in range(5)]
        )
        df = _make_events_df(events)
        cfg = BlockSourceConfig(
            kind="task_phase",
            label_column="event_subtype_role",
            phase_prefixes=(("choose", "choose_"), ("match", "match_")),
            gap_tolerance_sec=5.0,
            min_block_sec=2.0,
        )
        blocks = infer_blocks_from_events(df, cfg)
        assert len(blocks) == 2
        assert blocks[0].stage_code == 1  # choose (first phase)
        assert blocks[1].stage_code == 2  # match (second phase)
