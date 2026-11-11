"""Tests for distinct ED normalization and isotropic-shrinkage challengers."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.mnps.v3_2.ed import (
    compute_ed_raw_sensor_covariance_participation_ratio,
)
from mndm.mnps.v3_2.ed_normalized import (
    ED_NORMALIZED_ESTIMATOR_ID,
    compute_ed_normalized_by_channel_count,
)
from mndm.mnps.v3_2.ed_shrinkage import (
    ED_ISOTROPIC_SHRINKAGE_ESTIMATOR_ID,
    compute_ed_isotropic_shrinkage,
)


def _signals_with_covariance_eigenvalues(
    eigenvalues: tuple[float, ...], *, samples: int = 64
) -> np.ndarray:
    rng = np.random.default_rng(20261012)
    temporal_basis = rng.standard_normal((samples, len(eigenvalues)))
    temporal_basis -= temporal_basis.mean(axis=0, keepdims=True)
    orthonormal, _ = np.linalg.qr(temporal_basis)
    return (
        np.sqrt(samples - 1.0)
        * np.sqrt(np.asarray(eigenvalues, dtype=np.float64))[:, None]
        * orthonormal[:, : len(eigenvalues)].T
    )


def test_shrinkage_lambda_zero_matches_raw_ed_under_separate_identity() -> None:
    signals = _signals_with_covariance_eigenvalues((1.0, 0.5, 0.25, 0.1))
    labels = ("F3", "C3", "P3", "O1")
    raw = compute_ed_raw_sensor_covariance_participation_ratio(signals, labels)
    shrunk = compute_ed_isotropic_shrinkage(signals, 0.0, labels)

    assert raw.value is not None
    assert shrunk.value == pytest.approx(raw.value, rel=1e-13, abs=1e-13)
    assert shrunk.estimator_id == ED_ISOTROPIC_SHRINKAGE_ESTIMATOR_ID
    assert shrunk.estimator_id != raw.estimator_id
    assert shrunk.shrinkage_lambda == 0.0
    assert not shrunk.qualified


def test_lambda_one_isotropic_limit_returns_sensor_count() -> None:
    signals = _signals_with_covariance_eigenvalues((1.0, 0.5, 0.25, 1e-10))

    result = compute_ed_isotropic_shrinkage(signals, 1.0)

    assert result.value == pytest.approx(4.0, abs=1e-12)
    assert result.shrinkage_lambda == 1.0
    assert result.valid and result.computed
    assert result.to_measurement_record().method_parameters is not None


def test_lambda_half_matches_frozen_isotropic_covariance_formula() -> None:
    eigenvalues = np.array((1.0, 0.5, 0.25, 1e-10), dtype=np.float64)
    signals = _signals_with_covariance_eigenvalues(tuple(eigenvalues))
    mean_eigenvalue = float(eigenvalues.sum() / len(eigenvalues))
    shrunk_eigenvalues = 0.5 * eigenvalues + 0.5 * mean_eigenvalue
    expected = float(shrunk_eigenvalues.sum() ** 2 / np.square(shrunk_eigenvalues).sum())

    result = compute_ed_isotropic_shrinkage(signals, 0.5)

    assert result.value == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("lam", [-0.01, 1.01, float("nan"), float("inf"), True, "0.5"])
def test_shrinkage_rejects_invalid_lambda(lam: object) -> None:
    signals = _signals_with_covariance_eigenvalues((1.0, 0.5, 0.25, 0.1))

    result = compute_ed_isotropic_shrinkage(signals, lam)  # type: ignore[arg-type]

    assert result.value is None
    assert result.reason == "invalid_shrinkage_lambda"
    assert not result.valid
    assert not result.qualified


def test_normalized_ed_is_separate_diagnostic_and_preserves_unknown_support() -> None:
    signals = _signals_with_covariance_eigenvalues((1.0, 0.5, 0.25, 0.1))
    raw = compute_ed_raw_sensor_covariance_participation_ratio(
        signals, ("F3", "C3", "P3", "O1")
    )

    normalized = compute_ed_normalized_by_channel_count(raw)

    assert raw.value is not None
    assert normalized.value == pytest.approx(raw.value / 4)
    assert normalized.estimator_id == ED_NORMALIZED_ESTIMATOR_ID
    assert normalized.source_estimator_id == raw.estimator_id
    assert normalized.support_status == "unknown"
    assert normalized.channel_set_sha256 == raw.channel_set_sha256
    assert not normalized.qualified


def test_normalization_preserves_unsupported_raw_status_as_null() -> None:
    raw = compute_ed_raw_sensor_covariance_participation_ratio(np.ones((3, 12)))

    normalized = compute_ed_normalized_by_channel_count(raw)

    assert normalized.value is None
    assert normalized.reason == raw.reason
    assert not normalized.valid
    assert normalized.to_dict()["value"] is None
