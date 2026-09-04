"""Synthetic contract tests for the OD-SLP-003 DEV manifest helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.dynamical_families import sleep_committor_dev_manifest as manifest  # noqa: E402
from mndm.dynamical_families.committor import (  # noqa: E402
    estimate_committor_local_law_dense_grid_o2b,
)
from mndm.dynamical_families.sleep_committor_qualification import (  # noqa: E402
    FROZEN_PID_SPLIT,
)


def _segment(
    segment_id: str,
    *,
    pid: int = FROZEN_PID_SPLIT["DEV"][0],
    source: float = 2.0,
    outcome: str = "right_censored",
    stratum: str = "early_night",
) -> dict[str, object]:
    return {
        "segment_id": segment_id,
        "pid": str(pid),
        "split": "DEV",
        "night_stratum": stratum,
        "candidate_rc": float(source),
        "outcome": outcome,
        "reaction_coordinate": np.asarray([source, source + 0.001]),
        "time": np.asarray([0.0, 1.0]),
        "regime_labels": np.asarray([2, 2], dtype=np.int16),
        "external_rc_finite": True,
    }


def _fit_fixture() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    query = manifest.frozen_query_grid()
    counter = 0
    for value in query:
        for _ in range(manifest.SUPPORT_FLOOR):
            rows.append(
                _segment(
                    f"local-{counter}",
                    source=float(value),
                    stratum="early_night",
                )
            )
            counter += 1
    for index in range(manifest.MIN_TRANSITION_SEGMENTS):
        outcome = "first_hit_n3" if index % 2 else "first_hit_rem"
        rows.append(
            {
                **_segment(
                    f"transition-{index}",
                    source=2.0,
                    outcome=outcome,
                    stratum="early_night",
                ),
                "regime_labels": np.asarray([2, 3 if index % 2 else 4]),
            }
        )
    rows.extend(
        _segment(
            f"late-{index}",
            source=2.0,
            stratum="late_night",
        )
        for index in range(2)
    )
    return rows


def test_frozen_grid_and_split_digest_are_stable() -> None:
    query = manifest.frozen_query_grid()
    assert query.shape == (17,)
    assert query[0] == manifest.LOWER_BOUNDARY
    assert query[-1] == manifest.UPPER_BOUNDARY
    assert manifest.split_digest() == manifest.split_digest(
        {
            key: tuple(value)
            for key, value in FROZEN_PID_SPLIT.items()
        }
    )


def test_canonical_source_root_rejects_k_mapping() -> None:
    assert manifest.canonical_source_root_matches(
        r"M:/datasets/received/openneuro/ds005555"
    )
    assert not manifest.canonical_source_root_matches(
        r"K:\datasets\received\openneuro\ds005555"
    )


def test_transfer_tolerance_has_frozen_two_branch_rule() -> None:
    assert manifest.calculate_transfer_tolerance(0.05) == 0.02
    assert manifest.calculate_transfer_tolerance(0.40) == 0.1
    assert manifest.calculate_transfer_tolerance(None) is None


def test_dev_base_rate_brier_is_empirical_and_excludes_out_of_bounds() -> None:
    fit = {
        "status": "computed",
        "query_grid": [manifest.LOWER_BOUNDARY, manifest.UPPER_BOUNDARY],
        "q_grid": [0.0, 1.0],
    }
    segments = [
        {
            **_segment(
                "n3-in",
                source=manifest.LOWER_BOUNDARY,
                outcome="first_hit_n3",
            ),
            "split": "DEV",
        },
        {
            **_segment(
                "rem-in",
                source=manifest.UPPER_BOUNDARY,
                outcome="first_hit_rem",
            ),
            "split": "DEV",
        },
        {
            **_segment(
                "n3-out",
                source=manifest.UPPER_BOUNDARY + 1.0,
                outcome="first_hit_n3",
            ),
            "split": "DEV",
        },
    ]
    result = manifest.dev_base_rate_brier(segments, fit)
    expected = manifest.P_REM_DEV**2 / 2.0 + (
        1.0 - manifest.P_REM_DEV
    ) ** 2 / 2.0
    assert result["status"] == "computed"
    assert result["n_rows"] == 2
    assert np.isclose(result["brier"], expected)
    assert not np.isclose(result["brier"], manifest.P_REM_DEV * (1.0 - manifest.P_REM_DEV))


def test_fit_dev_grid_uses_audit_only_path_and_external_provenance(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_estimator(*args, **kwargs):
        captured.update(kwargs)
        return {
            "computation_status": "computed",
            "failure_reason": None,
            "series": {
                "query_grid": manifest.frozen_query_grid(),
                "q_grid": np.linspace(0.1, 0.9, 17),
                "grid_support_count": np.full(17, 64, dtype=np.int32),
            },
            "summary": {
                "n_transition_segments": 20,
                "appended_absorbing_boundary_rows": 0,
            },
        }

    monkeypatch.setattr(
        manifest,
        "estimate_audit_only_coarse_grid",
        fake_estimator,
    )
    result = manifest.fit_dev_grid(_fit_fixture(), stratum="pooled")
    assert result["status"] == "computed"
    assert len(result["q_grid"]) == 17
    assert captured["grid_resolution"] == 17
    assert captured["min_support_per_grid"] == 64
    assert result["provenance"]["coordinate_layer"] == "external_headband_rc"
    assert result["provenance"]["settings"]["audit_only"] is True
    assert result["appended_absorbing_boundary_rows"] == 0


def test_fit_dev_grid_rejects_nonfrozen_query_grid(monkeypatch) -> None:
    monkeypatch.setattr(
        manifest,
        "estimate_audit_only_coarse_grid",
        lambda *args, **kwargs: {
            "computation_status": "computed",
            "failure_reason": None,
            "series": {
                "query_grid": np.linspace(0.0, 1.0, 17),
                "q_grid": np.linspace(0.1, 0.9, 17),
                "grid_support_count": np.full(17, 64),
            },
            "summary": {
                "n_transition_segments": 20,
                "appended_absorbing_boundary_rows": 0,
            },
        },
    )
    result = manifest.fit_dev_grid(_fit_fixture(), stratum="pooled")
    assert result["status"] == "not_testable"
    assert result["failure_reason"] == "invalid_finite_dev_fit_payload"


@pytest.mark.parametrize(
    ("summary", "reason"),
    [
        ({}, "absorbing_boundary_row_count_missing"),
        (
            {"appended_absorbing_boundary_rows": 1},
            "absorbing_boundary_rows_present",
        ),
    ],
)
def test_fit_dev_grid_fails_closed_on_absorbing_row_contract(
    monkeypatch,
    summary,
    reason,
) -> None:
    monkeypatch.setattr(
        manifest,
        "estimate_audit_only_coarse_grid",
        lambda *args, **kwargs: {
            "computation_status": "computed",
            "failure_reason": None,
            "series": {
                "query_grid": manifest.frozen_query_grid(),
                "q_grid": np.linspace(0.1, 0.9, 17),
                "grid_support_count": np.full(17, 64),
            },
            "summary": {
                "n_transition_segments": 20,
                **summary,
            },
        },
    )
    result = manifest.fit_dev_grid(_fit_fixture(), stratum="pooled")
    assert result["status"] == "not_testable"
    assert result["failure_reason"] == reason


def test_fit_strata_are_disjoint_and_non_dev_pid_fails_closed(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        manifest,
        "estimate_audit_only_coarse_grid",
        lambda *args, **kwargs: {
            "computation_status": "computed",
            "failure_reason": None,
            "series": {
                "query_grid": manifest.frozen_query_grid(),
                "q_grid": np.linspace(0.1, 0.9, 17),
                "grid_support_count": np.full(17, 64),
            },
            "summary": {
                "n_transition_segments": 20,
                "appended_absorbing_boundary_rows": 0,
            },
        },
    )
    rows = _fit_fixture()
    assert all(
        row["night_stratum"] == "early_night"
        for row in manifest.dev_segments_for_stratum(rows, "early")
    )
    assert all(
        row["night_stratum"] == "late_night"
        for row in manifest.dev_segments_for_stratum(rows, "late")
    )
    invalid = [dict(rows[0], pid="999")]
    assert manifest.fit_dev_grid(invalid, stratum="pooled")["failure_reason"] == (
        "support_segment_not_in_dev_split"
    )


def test_dev_base_rate_brier_rejects_non_dev_pid_even_with_dev_split_label() -> None:
    fit = {
        "status": "computed",
        "query_grid": [manifest.LOWER_BOUNDARY, manifest.UPPER_BOUNDARY],
        "q_grid": [0.0, 1.0],
    }
    segment = _segment(
        "wrong-pid",
        pid=999,
        outcome="first_hit_rem",
    )
    segment["split"] = "DEV"
    result = manifest.dev_base_rate_brier([segment], fit)
    assert result["status"] == "not_testable"
    assert result["reason"] == "dev_base_rate_segment_not_in_dev_split"


def test_production_estimator_still_rejects_coarse_grid() -> None:
    state = np.asarray([[0.0], [0.01]] * 20)
    time = np.asarray([0.0, 1.0] * 20)
    labels = np.asarray([2, 3] * 20)
    segment_id = np.repeat(np.arange(20), 2)
    result = estimate_committor_local_law_dense_grid_o2b(
        state,
        time,
        state[:, 0],
        labels,
        set_A=[3],
        set_B=[4],
        grid_min=manifest.LOWER_BOUNDARY,
        grid_max=manifest.UPPER_BOUNDARY,
        diffusion_coefficient=manifest.A_DEV,
        segment_id=segment_id,
        grid_resolution=17,
        min_support_per_grid=manifest.SUPPORT_FLOOR,
    )
    assert result["computation_status"] == "not_testable"
    assert result["failure_reason"] == "o2b_grid_resolution_below_minimum"
