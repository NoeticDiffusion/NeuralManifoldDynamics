from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.mnps.v3_2.experiments import window_support_001_perturbation as perturb


def test_earliest_cadence_selection_is_one_window_per_participant_condition_duration() -> None:
    rows = []
    for start, window_id in ((10, "later"), (0, "earlier")):
        for family in ("AR", "ED"):
            rows.append({
                "participant_id": "sub-001", "run_id": "run-01", "condition": "EO",
                "duration_seconds": 4, "window_grid_roles": ["cadence"],
                "support_start_sample": start, "window_id": window_id, "family_id": family,
            })
    rows.append({**rows[0], "participant_id": "sub-002", "condition": "EC"})

    selected = perturb.choose_earliest_cadence_windows(rows)

    assert len(selected) == 2
    assert {(row["participant_id"], row["condition"], row["window_id"]) for row in selected} == {
        ("sub-001", "EO", "earlier"), ("sub-002", "EC", "later")
    }


def test_noise_seed_and_perturbed_signal_are_deterministic() -> None:
    signals = np.arange(80, dtype=np.float64).reshape(4, 20) + 1.0
    digest, seed = perturb.deterministic_noise_seed("sub-001", "run-01", "window", 0.001)

    left = perturb.perturb_signal(signals, 0.001, seed)
    right = perturb.perturb_signal(signals, 0.001, seed)

    assert len(digest) == 64
    assert np.array_equal(left, right)
    assert not np.array_equal(left, signals)
    with pytest.raises(ValueError, match="not_in_frozen_protocol"):
        perturb.perturb_signal(signals, 0.02, seed)


def test_scalar_effect_preserves_signed_and_relative_change() -> None:
    baseline = {
        "family_id": "ED", "participant_id": "sub-001", "run_id": "run-01",
        "condition": "EO", "duration_seconds": 4, "window_id": "window",
        "computed": True, "valid": True, "status": "computed_exploratory",
        "reason": None, "value": 2.0,
    }
    perturbed = {
        "family_id": "ED", "computed": True, "valid": True,
        "status": "computed_exploratory", "reason": None, "value": 1.8,
    }

    row = perturb.build_effect_row(
        baseline, perturbed, noise_fraction=0.001, seed_sha256="a" * 64,
        perturbed_window_sha256="b" * 64,
    )

    assert row["signed_delta"] == pytest.approx(-0.2)
    assert row["absolute_delta"] == pytest.approx(0.2)
    assert row["relative_delta"] == pytest.approx(-0.1)
    assert row["shape_match"] is True
    assert row["qualified"] is False


def test_ar_effects_and_summaries_remain_component_wise() -> None:
    baseline = {
        "family_id": "AR", "participant_id": "sub-001", "run_id": "run-01",
        "condition": "EO", "duration_seconds": 8, "window_id": "window",
        "duration_realization_id": "AR.window08s", "computed": True, "valid": True,
        "status": "computed_exploratory", "reason": None,
        "component_values": [0.2, 0.4, 0.5], "component_names": ["a", "b", "c"],
    }
    perturbed = {
        "family_id": "AR", "computed": True, "valid": True,
        "status": "computed_exploratory", "reason": None,
        "component_values": [0.3, 0.3, 0.5], "component_names": ["a", "b", "c"],
    }
    row = perturb.build_effect_row(
        baseline, perturbed, noise_fraction=0.01, seed_sha256="c" * 64,
        perturbed_window_sha256="d" * 64,
    )

    assert row["shape_match"] is True
    assert row["signed_delta"] == pytest.approx([0.1, -0.1, 0.0])
    assert row["absolute_delta"] == pytest.approx([0.1, 0.1, 0.0])
    summary = perturb.summarize_effects([row])
    assert [item["component_index"] for item in summary] == [0, 1, 2]
    assert [item["component_name"] for item in summary] == ["a", "b", "c"]
    assert all(item["windows_requested"] == 1 for item in summary)


def test_ar_shape_mismatch_has_no_component_deltas() -> None:
    baseline = {"family_id": "AR", "component_values": [0.1, 0.2], "component_names": ["a", "b"]}
    perturbed = {"family_id": "AR", "component_values": [0.1], "component_names": ["a"]}

    row = perturb.build_effect_row(
        baseline, perturbed, noise_fraction=0.001, seed_sha256="e" * 64,
        perturbed_window_sha256="f" * 64,
    )

    assert row["shape_match"] is False
    assert row["signed_delta"] is None
    assert row["absolute_delta"] is None
