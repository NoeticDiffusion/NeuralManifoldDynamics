from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_id": ["ds003838"] * 6,
            "subject_id": ["sub-01"] * 3 + ["sub-02"] * 3,
            "run_id": ["run-1"] * 6,
            "condition": ["event"] * 6,
            "task_state_label": ["listen", "mem5", "mem9", "listen", "mem5", "mem9"],
            "task_load_n": [0.0, 5.0, 9.0, 0.0, 5.0, 9.0],
            "anchor_index": [0.1, 0.3, 0.5, 1.1, 1.3, 1.5],
            "vagal_index": [0.2, 0.4, 0.6, 1.2, 1.4, 1.6],
            "m": np.linspace(0.0, 1.0, 6),
            "d": np.linspace(1.0, 2.0, 6),
            "e": np.linspace(2.0, 3.0, 6),
            "m_dot": np.linspace(0.0, 0.5, 6),
            "d_dot": np.linspace(0.5, 1.0, 6),
            "e_dot": np.linspace(1.0, 1.5, 6),
        }
    )


def test_summarize_anchor_by_load_emits_grouped_anchor_metrics():
    from mndm.pipeline.anchor_validation import summarize_anchor_by_load

    summary = summarize_anchor_by_load(_make_frame())

    assert not summary.empty
    assert "anchor_index_mean" in summary.columns
    assert "mnps_radius_mean" in summary.columns
    listen_rows = summary.loc[summary["task_state_label"] == "listen"]
    assert len(listen_rows) == 2


def test_time_shift_null_rotates_anchor_columns_within_subject():
    from mndm.pipeline.anchor_validation import make_time_shift_null

    frame = _make_frame()
    shifted = make_time_shift_null(frame, shift=1)

    assert shifted["null_control"].eq("time_shift").all()
    assert shifted.loc[:2, "anchor_index"].tolist() == [0.5, 0.1, 0.3]
    assert shifted.loc[3:, "anchor_index"].tolist() == [1.5, 1.1, 1.3]


def test_subject_shuffle_null_swaps_anchor_donors_between_subjects():
    from mndm.pipeline.anchor_validation import make_subject_shuffle_null

    frame = _make_frame()
    shuffled = make_subject_shuffle_null(frame)

    assert shuffled["null_control"].eq("subject_shuffle").all()
    assert shuffled.loc[:2, "anchor_index"].tolist() == [1.1, 1.3, 1.5]
    assert shuffled.loc[3:, "anchor_index"].tolist() == [0.1, 0.3, 0.5]


def test_summarize_null_controls_returns_observed_and_null_variants():
    from mndm.pipeline.anchor_validation import summarize_null_controls

    result = summarize_null_controls(_make_frame())

    assert set(result.keys()) == {"observed", "time_shift", "subject_shuffle"}
    assert all(isinstance(frame, pd.DataFrame) for frame in result.values())
