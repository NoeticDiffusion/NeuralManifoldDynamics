"""Tests for Figshare cohort-coordinate age analysis."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _write_h5_session(run_dir: Path, participant: str, session: str, age: int, source: str, excluded: bool = False) -> None:
    """Create a minimal cohort-coordinate H5 plus its paired session summary."""
    directory = run_dir / f"{participant}_{session}"
    directory.mkdir(parents=True)
    path = directory / f"{participant}_{session}.h5"
    values_3d = np.column_stack(
        [
            np.linspace(age / 100, age / 100 + 0.2, 8),
            np.linspace(0, 0.1, 8),
            np.linspace(-0.1, 0, 8),
        ]
    )
    values_9d = np.column_stack([values_3d, np.tile(np.arange(6), (8, 1))])
    with h5py.File(path, "w") as h5:
        for name, values, labels in (
            ("coords_3d_cohort_anchored", values_3d, ["m", "d", "e"]),
            (
                "coords_9d_cohort_anchored",
                values_9d,
                ["m", "d", "e", "d_n", "d_l", "d_s", "e_e", "e_s", "e_m"],
            ),
        ):
            group = h5.create_group(name)
            group.create_dataset("values", data=values)
            group.create_dataset("names", data=np.asarray(labels, dtype="S"))
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "participant_meta": {
                    "participant_id": participant,
                    "session_id": session,
                    "age_days": age,
                    "age_source": source,
                    "excluded_from_age_analysis": excluded,
                }
            }
        ),
        encoding="utf-8",
    )


def test_build_session_coordinate_table_and_exclusion(tmp_path: Path):
    """The H5/session join retains source flags and cohort-coordinate medians."""
    from mndm.tools.figshare_infant_cohort import build_session_coordinate_table

    run_dir = tmp_path / "run"
    rows = []
    for index, (participant, session, age, excluded) in enumerate(
        [
            ("sub-01", "ses-01", 40, False),
            ("sub-01", "ses-02", 70, False),
            ("sub-02", "ses-01", 50, False),
            ("sub-02", "ses-02", 80, False),
            ("sub-03", "ses-01", 60, False),
            ("sub-03", "ses-02", 90, True),
        ]
    ):
        source = "spreadsheet_direct" if index % 2 == 0 else "derived_from_first_visit_age_and_meas_date"
        _write_h5_session(run_dir, participant, session, age, source, excluded)
        rows.append(
            {
                "participant_id": participant,
                "session_id": session,
                "age_days": age,
                "age_source": source,
                "excluded_from_age_analysis": excluded,
            }
        )
    sessions_path = tmp_path / "sessions.tsv"
    pd.DataFrame(rows).to_csv(sessions_path, sep="\t", index=False)

    table = build_session_coordinate_table(run_dir, sessions_path)

    assert len(table) == 6
    assert "coords_3d_cohort_anchored__m_median" in table.columns
    assert table["excluded_from_age_analysis"].sum() == 1


def test_gee_models_exclude_unresolved_and_emit_sensitivity(tmp_path: Path):
    """Age GEE uses child clusters and excludes explicit age-analysis rows."""
    pytest.importorskip("statsmodels")
    from mndm.tools.figshare_infant_cohort import fit_cohort_coordinate_age_models

    table = pd.DataFrame(
        {
            "participant_id": ["sub-01", "sub-01", "sub-02", "sub-02", "sub-03", "sub-03", "sub-04", "sub-04"],
            "age_days": [40, 70, 45, 75, 50, 80, 55, 85],
            "age_source": [
                "spreadsheet_direct",
                "derived_from_first_visit_age_and_meas_date",
                "spreadsheet_direct",
                "derived_from_first_visit_age_and_meas_date",
                "spreadsheet_direct",
                "derived_from_first_visit_age_and_meas_date",
                "spreadsheet_direct",
                "derived_from_first_visit_age_and_meas_date",
            ],
            "excluded_from_age_analysis": [False, False, False, False, False, False, False, True],
            "coords_3d_cohort_anchored__m_median": np.linspace(0.1, 0.8, 8),
        }
    )

    results, coefficients, predictions = fit_cohort_coordinate_age_models(table)

    primary = results.loc[results["analysis"] == "resolved_age_primary"].iloc[0]
    assert primary["status"] == "fit"
    assert primary["n_sessions"] == 7
    assert primary["n_children"] == 4
    assert "spreadsheet_direct_age_sensitivity" in set(results["analysis"])
    assert not coefficients.empty
    assert not predictions.empty
