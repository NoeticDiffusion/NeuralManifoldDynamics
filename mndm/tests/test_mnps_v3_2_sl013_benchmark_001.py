from __future__ import annotations

import numpy as np

from mndm.mnps.v3_2.experiments.sl013_benchmark_001 import (
    build_common7_and_ar,
    build_row_accounting,
    equal_span_offsets,
    intervals_cover_anchor,
    record_class,
    synthetic_recovery,
    trajectory_design_diagnostics,
)


def test_equal_span_grids_cover_exactly_twenty_seconds() -> None:
    anchor_start = 10_000
    anchor_end = anchor_start + 20 * 500
    for duration in (4, 8, 12):
        intervals = [
            (anchor_start + offset * 500, anchor_start + (offset + duration) * 500)
            for offset in equal_span_offsets(duration)
        ]
        assert intervals_cover_anchor(anchor_start, anchor_end, intervals)
        assert all(start >= anchor_start and end <= anchor_end for start, end in intervals)


def test_equal_span_cover_rejects_gap_or_out_of_anchor_window() -> None:
    assert not intervals_cover_anchor(0, 10, [(0, 4), (5, 10)])
    assert not intervals_cover_anchor(0, 10, [(-1, 5), (5, 10)])


def test_parent_record_classes_are_explicit_and_reconcile() -> None:
    assert record_class({"window_grid_roles": ["cadence", "matched_20s_anchor_prefix"]}) == "cadence_and_matched"
    assert record_class({"window_grid_roles": ["cadence"]}) == "cadence_only"
    assert record_class({"window_grid_roles": ["matched_20s_anchor_prefix"]}) == "matched_only"
    rows = [
        {"participant_id": "sub-1", "run_id": "run-01", "condition": "EO", "state_block_id": "b",
         "duration_seconds": 4, "support_start_sample": start, "support_end_sample_exclusive": start + 2000,
         "family_id": family, "window_grid_roles": roles}
        for start, roles in ((0, ["cadence"]), (2000, ["cadence", "matched_20s_anchor_prefix"]))
        for family in ("PCV",)
    ]
    details, summary = build_row_accounting(rows)
    assert sum(row["family_rows"] for row in details) == 2
    assert sum(row["unique_windows"] for row in summary) == 2
    assert sum(row["family_rows"] for row in summary) == 2


def test_common7_representation_keeps_ar_separate() -> None:
    base = {
        "participant_id": "sub-1", "run_id": "run-01", "condition": "EO", "state_block_id": "b",
        "duration_seconds": 4, "support_start_sample": 0, "support_end_sample_exclusive": 2000,
        "window_id": "w", "window_grid_roles": ["cadence"], "sample_rate_hz": 500,
        "channel_count": 2, "channel_order_sha256": "order", "channel_set_sha256": "set",
        "reference_id": "ref", "reference_realization_sha256": "ref-hash", "reference_set_sha256": "ref-set",
        "valid": True, "computed": True, "status": "valid", "reason": None,
    }
    rows = [
        {**base, "family_id": family, "value": float(index)}
        for index, family in enumerate(("PCV", "TP", "NIE", "RD", "OI", "SC", "ED"))
    ]
    rows.append({**base, "family_id": "AR", "component_names": ["AR_1"], "component_values": [0.7],
                 "duration_realization_id": "ar4", "value_shape": [1]})
    common, ar = build_common7_and_ar(rows)
    assert len(common) == 1
    assert common[0]["coordinate_order"] == '["PCV", "TP", "NIE", "RD", "OI", "SC", "ED"]'
    assert common[0]["ED_value"] == 6.0
    assert "AR_value" not in common[0]
    assert len(ar) == 1 and ar[0]["component_values"] == "[0.7]"


def test_design_diagnostics_detect_full_and_deficient_affine_designs() -> None:
    rng = np.random.default_rng(41)
    full = rng.normal(size=(32, 8))
    full_result = trajectory_design_diagnostics(full)
    assert full_result["raw"]["rank"] == 9
    assert full_result["training_prefix_standardized"]["available"] is True
    deficient = full.copy()
    deficient[:, 7] = deficient[:, 6]
    deficient_result = trajectory_design_diagnostics(deficient)
    assert deficient_result["raw"]["rank"] < 9


def test_synthetic_linear_dynamics_recovery_and_rank_control() -> None:
    result = synthetic_recovery()
    assert result["full_rank_recovery_pass"] is True
    assert result["duplicate_coordinate_control_rejected"] is True
    assert result["design_rank"] == 9
    assert result["duplicate_coordinate_control_rank"] < 9
