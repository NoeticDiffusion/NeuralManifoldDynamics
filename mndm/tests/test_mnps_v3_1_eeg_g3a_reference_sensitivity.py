"""Synthetic checks for E1's G3a-R reference sensitivity calculations."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    REPO_ROOT
    / "project"
    / "new_mnps_v.3.1"
    / "run_anphy_epctl01_e1_g3a_reference_sensitivity.py"
)
spec = importlib.util.spec_from_file_location("mnps_v31_e1_g3a_reference_sensitivity", RUNNER_PATH)
assert spec is not None and spec.loader is not None
g3a_r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g3a_r)


COMMON_INDICES = tuple(range(19))


def test_rereference_to_common_subset_matches_direct_average_reference() -> None:
    raw = np.random.default_rng(31_904).normal(size=(83, 5000))
    reference_83 = raw - np.mean(raw, axis=0, keepdims=True)
    actual = g3a_r._reference_over_subset(reference_83, COMMON_INDICES)
    expected = raw - np.mean(raw[list(COMMON_INDICES), :], axis=0, keepdims=True)

    assert np.allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert np.max(np.abs(np.mean(actual[list(COMMON_INDICES), :], axis=0))) < 1e-13


def test_rereference_requires_exact_window_and_unique_reference_rows() -> None:
    valid = np.random.default_rng(50).normal(size=(83, 5000))

    with pytest.raises(ValueError, match="83_by_5000"):
        g3a_r._reference_over_subset(valid[:, :-1], COMMON_INDICES)
    with pytest.raises(ValueError, match="19_unique_rows"):
        g3a_r._reference_over_subset(valid, tuple(range(18)) + (17,))


def test_reference_contrast_statistics_keep_declared_orientation_and_no_inference() -> None:
    reference83 = np.asarray([0.2, 0.4, 0.6, 0.8])
    reference19 = np.asarray([0.3, 0.5, 0.7, 0.9])

    result = g3a_r._paired_statistics(
        reference83,
        reference19,
        orientation="reference19_minus_reference83",
    )

    assert result["orientation"] == "reference19_minus_reference83"
    assert result["spearman_rank_correlation"] == pytest.approx(1.0)
    assert result["mean_bias"] == pytest.approx(0.1)
    assert result["median_absolute_difference"] == pytest.approx(0.1)
    assert result["p_value"] is None
    assert result["confidence_interval"] is None


def test_replay_provenance_allows_only_pinned_helper_hash_delta() -> None:
    historical = {"window_loader": "old", "preprocess_support_helpers": "old-helper", "entropy": "same"}
    current = {"window_loader": "old", "preprocess_support_helpers": "new-helper", "entropy": "same"}
    policy = {
        "permitted_hash_difference": "preprocess_support_helpers",
        "historical_hash": "old-helper",
        "current_hash": "new-helper",
    }

    replay = g3a_r._validate_replay_implementation(historical, current, policy)

    assert replay["all_other_implementation_hashes_match"] is True
    assert replay["underlying_sample_matrix_bitwise_replay_claim"] is False


def test_replay_provenance_rejects_unpinned_or_additional_hash_delta() -> None:
    historical = {"window_loader": "old", "preprocess_support_helpers": "old-helper", "entropy": "same"}
    policy = {
        "permitted_hash_difference": "preprocess_support_helpers",
        "historical_hash": "old-helper",
        "current_hash": "new-helper",
    }
    with pytest.raises(ValueError, match="current_helper_hash_mismatch"):
        g3a_r._validate_replay_implementation(
            historical, {**historical, "preprocess_support_helpers": "unexpected"}, policy
        )
    with pytest.raises(ValueError, match="unexpected_implementation_hash_differences"):
        g3a_r._validate_replay_implementation(
            historical,
            {**historical, "preprocess_support_helpers": "new-helper", "window_loader": "changed"},
            policy,
        )
