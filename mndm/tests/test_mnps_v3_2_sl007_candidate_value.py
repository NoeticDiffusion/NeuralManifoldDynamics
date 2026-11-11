"""Focused numerical tests for the bounded SL-007 control helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.mnps.v3_2.experiments.sl007_candidate_value import (
    _verify_sl006_artifacts,
    _participant_oi_summary,
    _conventional_features,
    _iaaft_surrogate,
    _permute_edge_positions,
)


def test_edge_position_control_preserves_exact_multiset_and_symmetry() -> None:
    weights = np.array(
        [[0.0, 0.1, 0.2, 0.0], [0.1, 0.0, 0.3, 0.4], [0.2, 0.3, 0.0, 0.5], [0.0, 0.4, 0.5, 0.0]],
        dtype=np.float64,
    )

    permuted = _permute_edge_positions(weights, seed=1234)
    upper = np.triu_indices(4, k=1)
    assert np.array_equal(permuted, permuted.T)
    assert np.array_equal(np.sort(permuted[upper]), np.sort(weights[upper]))
    assert np.all(np.diag(permuted) == 0.0)


def test_iaaft_preserves_amplitude_multiset_and_reports_spectral_error() -> None:
    time = np.arange(1000, dtype=np.float64) / 250.0
    signal = np.sin(2 * np.pi * 8 * time) + 0.1 * np.sin(2 * np.pi * 23 * time)
    surrogate, spectral_error = _iaaft_surrogate(signal, seed=42, iterations=12)

    assert np.array_equal(np.sort(surrogate), np.sort(signal))
    assert np.isfinite(surrogate).all()
    assert np.isfinite(spectral_error)
    assert spectral_error >= 0.0


def test_conventional_features_are_finite_for_nonconstant_multichannel_signal() -> None:
    time = np.arange(5000, dtype=np.float64) / 250.0
    signals = np.vstack(
        (
            np.sin(2 * np.pi * 8 * time),
            np.sin(2 * np.pi * 12 * time + 0.2),
            0.5 * np.sin(2 * np.pi * 4 * time) + np.cos(2 * np.pi * 20 * time),
        )
    )

    features = _conventional_features(signals)
    assert len(features) == 9
    assert all(np.isfinite(value) for value in features.values())
    assert features["welch_power_alpha_8_13_uv2"] > 0.0


def test_sl006_artifact_manifest_checks_hashes_and_rejects_escape(tmp_path: Path) -> None:
    artifact = tmp_path / "participant.json"
    artifact.write_text("{}", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()

    _verify_sl006_artifacts(tmp_path, {"participant.json": digest})

    try:
        _verify_sl006_artifacts(tmp_path, {"participant.json": "0" * 64})
    except ValueError as error:
        assert "sl006_artifact_hash_mismatch" in str(error)
    else:
        raise AssertionError("artifact hash mismatch was not rejected")

    try:
        _verify_sl006_artifacts(tmp_path, {"../outside.json": digest})
    except ValueError as error:
        assert "sl006_artifact_path_escapes_run_root" in str(error)
    else:
        raise AssertionError("artifact path escape was not rejected")


def test_oi_participant_summary_uses_single_median_suffix_for_comparators() -> None:
    summary = _participant_oi_summary(
        "S01",
        [{
            "oi": 0.3,
            "normalized_spectral_entropy_1_45": 0.6,
            "welch_power_delta_1_4_uv2": 1.2,
            "autocorrelation_lag_1_samples": 0.4,
        }],
    )

    assert summary["normalized_spectral_entropy_1_45_median"] == 0.6
    assert summary["welch_power_delta_1_4_uv2_median"] == 1.2
    assert summary["autocorrelation_lag_1_samples_median"] == 0.4
    assert "normalized_spectral_entropy_1_45_median_median" not in summary
