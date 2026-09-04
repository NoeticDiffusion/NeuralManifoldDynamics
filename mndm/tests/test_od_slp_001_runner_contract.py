"""Runner-side fail-closed tests without opening BOAS source data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from od_slp_001_empirical_qualification_gate import (  # noqa: E402
    _clone_segments_with_night_rc,
    _comparison_pass,
    _object2_competing_diagnostic,
    run_qualification,
)


def test_comparison_pass_requires_point_and_lower_ci_floors() -> None:
    passing = {
        "status": "computed",
        "point_relative_brier_improvement": 0.10,
        "point_log_loss_improvement": 0.02,
        "brier_lower_95": 0.01,
        "log_loss_lower_95": 0.001,
    }
    failing = dict(passing, brier_lower_95=-0.01)
    assert _comparison_pass(passing)
    assert not _comparison_pass(failing)


def test_competing_diagnostic_fails_closed_on_empty_bin() -> None:
    segments = [
        {
            "pid": "1",
            "candidate_rc": -0.9,
            "outcome": "first_hit_n3",
            "external_rc_finite": True,
        }
    ]
    result = _object2_competing_diagnostic(
        segments,
        lower=-1.0,
        upper=1.0,
    )
    assert result["status"] == "NOT_TESTABLE"
    assert result["reason"] == "competing_risk_empty_rc_bin"


def test_wrong_pid_prefix_truncation_does_not_pad_tail() -> None:
    record = {
        "night_key": "sub-1|NA|Sleep|NA",
        "split": "DEV",
        "segments": [
            {
                "segment_id": "seg",
                "candidate_interval_index": 0,
                "n2_window_count": 4,
                "reaction_coordinate": np.zeros(4),
                "candidate_rc": 0.0,
                "external_rc_finite": True,
            }
        ],
    }
    cloned = _clone_segments_with_night_rc(
        [record],
        reaction_by_night={
            "sub-1|NA|Sleep|NA": np.asarray([1.0, 2.0])
        },
    )
    values = cloned[0]["reaction_coordinate"]
    assert values[:2].tolist() == [1.0, 2.0]
    assert np.all(np.isnan(values[2:]))
    assert not cloned[0]["external_rc_finite"]


def test_empty_source_root_is_not_testable_and_writes_no_hdf5(tmp_path: Path) -> None:
    output = tmp_path / "qualification.json"
    result = run_qualification(raw_root=tmp_path / "missing", output_path=output)
    assert result["decision"]["status"] == "NOT_TESTABLE"
    assert output.exists()
    assert not list(tmp_path.glob("*.h5"))
