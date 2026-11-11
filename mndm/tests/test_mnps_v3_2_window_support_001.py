from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.mnps.v3_2.candidate_grid import ConditionBlock
from mndm.mnps.v3_2.experiments import window_support_001 as ws


def test_technical_roster_selection_is_deterministic_and_id_only() -> None:
    roster = [{"participant_id": f"sub-{index:03d}", "technical": index} for index in range(1, 10)]

    first = ws.choose_technical_roster(roster, count=4)
    second = ws.choose_technical_roster(list(reversed(roster)), count=4)

    assert [row["participant_id"] for row in first] == [row["participant_id"] for row in second]
    assert all("technical" in row for row in first)
    with pytest.raises(ValueError, match="duplicate_participant_ids"):
        ws.choose_technical_roster([roster[0], roster[0]], count=1)


def test_boundary_markers_split_only_the_interior_of_condition_blocks() -> None:
    block = ConditionBlock("stateblock-001", "EO", 0, 40, 0.0)

    split = ws.split_blocks_at_boundaries([block], [0, 12, 30, 40])

    assert [(item.start_sample, item.end_sample_exclusive) for item in split] == [(0, 12), (12, 30), (30, 40)]
    assert all(item.condition == "EO" for item in split)


def test_duration_schedule_reuses_matched_prefixes_and_keeps_anchor_start() -> None:
    block = ConditionBlock("stateblock-001", "EO", 0, 40, 0.0)

    scheduled = ws.schedule_duration_windows([block], sample_rate_hz=1, duration_seconds=8)

    assert len(scheduled) == 6
    matched = [row for row in scheduled if "matched_20s_anchor_prefix" in row.grid_roles]
    assert [row.window.start_sample for row in matched] == [0, 20]
    assert all(row.matched_anchor_start_sample == row.window.start_sample for row in matched)
    assert all(row.window.sample_count == 8 for row in matched)
    assert "cadence" in scheduled[0].grid_roles
    assert "matched_20s_anchor_prefix" in scheduled[0].grid_roles


def test_ar_duration_realizations_preserve_distinct_component_shapes() -> None:
    expected = {4: 1, 8: 3, 12: 5, 20: 9}

    for duration, component_count in expected.items():
        metadata = ws.ar_realization_metadata(duration)
        assert metadata["component_count"] == component_count
        assert metadata["block_count"] == component_count + 1
        assert metadata["duration_realization_id"].endswith(f"{component_count + 1}blocks.v1")
    assert len({ws.duration_realization_id("AR", duration) for duration in expected}) == 4


def test_ar_computed_exploratory_components_count_as_support_without_scalarizing_partial_vectors() -> None:
    complete = {
        "family_id": "AR", "computed": True, "valid": True,
        "component_values": [0.2, 0.3], "component_names": ["a", "b"],
        "component_status": ["computed_exploratory", "computed_exploratory"],
    }
    partial = {
        "family_id": "AR", "computed": True, "valid": False,
        "component_values": [0.2, None], "component_names": ["a", "b"],
        "component_status": ["computed_exploratory", "unsupported"],
    }

    assert ws._support_is_valid(complete)
    assert not ws._support_is_valid(partial)
    assert ws._ar_component_value(partial, 0) == 0.2
    assert ws._ar_component_value(partial, 1) is None


def test_ar_condition_descriptions_are_component_wise_even_for_partial_rows() -> None:
    common = {
        "participant_id": "sub-001", "run_id": "run-01", "duration_seconds": 8,
        "family_id": "AR", "window_grid_roles": ["cadence"],
        "component_names": ["c0", "c1", "c2"],
    }
    rows = [
        {
            **common, "condition": "EO", "state_block_id": "stateblock-001",
            "support_start_sample": 0, "support_end_sample_exclusive": 4000,
            "computed": True, "valid": False, "status": "partial",
            "component_values": [0.1, None, 0.3],
            "component_status": ["computed_exploratory", "unsupported", "computed_exploratory"],
        },
        {
            **common, "condition": "EC", "state_block_id": "stateblock-002",
            "support_start_sample": 5000, "support_end_sample_exclusive": 9000,
            "computed": True, "valid": False, "status": "partial",
            "component_values": [0.4, 0.5, None],
            "component_status": ["computed_exploratory", "computed_exploratory", "unsupported"],
        },
    ]

    temporal, contrasts = ws.build_temporal_and_condition_descriptives(rows)

    assert len(temporal) == 6
    assert len(contrasts) == 3
    by_index = {row["component_index"]: row for row in contrasts}
    assert by_index[0]["ec_minus_eo"] == pytest.approx(0.3)
    assert by_index[0]["eo_valid_windows"] == 1
    assert by_index[0]["ec_valid_windows"] == 1
    assert by_index[1]["ec_minus_eo"] is None
    assert by_index[2]["ec_minus_eo"] is None


def test_float_value_parity_uses_frozen_tight_tolerance_and_compares_shapes() -> None:
    assert ws._values_match([0.1, 0.2], [0.1 + 1e-13, 0.2])
    assert not ws._values_match([0.1, 0.2], [0.1])
    assert not ws._values_match(0.1, 0.1 + 1e-8)
