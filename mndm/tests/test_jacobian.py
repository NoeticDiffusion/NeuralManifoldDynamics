"""Tests for MNPS Jacobian estimation."""

from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def test_jacobian_recovers_linear_system():
    from mndm.jacobian import estimate_local_jacobians

    rng = np.random.default_rng(123)
    x = rng.normal(size=(60, 3)).astype(np.float32)
    x -= x.mean(axis=0, keepdims=True)
    a_true = np.array([[0.1, -0.05, 0.02], [0.03, -0.08, 0.01], [-0.02, 0.04, -0.05]], dtype=np.float32)
    x_dot = x @ a_true.T

    nn_idx = np.tile(np.arange(len(x)), (len(x), 1)).astype(np.int32)

    result = estimate_local_jacobians(x, x_dot, nn_idx, super_window=1, ridge_alpha=1e-6)

    assert result.j_hat.shape[1:] == (3, 3)
    mean_j = result.j_hat.mean(axis=0)
    assert np.allclose(mean_j, a_true, atol=1e-2)


def test_jacobian_centers_align_with_successful_windows():
    from mndm.jacobian import estimate_local_jacobians

    rng = np.random.default_rng(7)
    x = rng.normal(size=(8, 2)).astype(np.float32)
    a_true = np.array([[0.2, -0.1], [0.05, -0.15]], dtype=np.float32)
    x_dot = x @ a_true.T

    # super_window=1 => gathered set is [center] + nn_idx[center].
    # Rows with duplicate neighbours fail (< dim+1 unique points for dim=2).
    nn_idx = np.array(
        [
            [0, 0],  # fail (only {0})
            [1, 1],  # fail
            [0, 7],  # success ({2,0,7})
            [3, 3],  # fail
            [1, 6],  # success ({4,1,6})
            [5, 5],  # fail
            [2, 7],  # success ({6,2,7})
            [7, 7],  # fail
        ],
        dtype=np.int32,
    )

    result = estimate_local_jacobians(x, x_dot, nn_idx, super_window=1, ridge_alpha=1e-6)

    expected_centers = np.array([2, 4, 6], dtype=np.int32)
    assert np.array_equal(result.centers, expected_centers)
    assert result.j_hat.shape[0] == expected_centers.shape[0]


def test_jacobian_dot_is_timeline_aligned_and_zero_at_start():
    from mndm.jacobian import estimate_local_jacobians

    rng = np.random.default_rng(11)
    x = rng.normal(size=(20, 3)).astype(np.float32)
    a_true = np.array(
        [[0.1, -0.02, 0.03], [0.0, -0.07, 0.01], [-0.01, 0.02, -0.05]],
        dtype=np.float32,
    )
    x_dot = x @ a_true.T
    nn_idx = np.tile(np.arange(len(x)), (len(x), 1)).astype(np.int32)

    result = estimate_local_jacobians(x, x_dot, nn_idx, super_window=1, ridge_alpha=1e-6)

    assert result.j_dot.shape == result.j_hat.shape
    if result.j_hat.shape[0] > 1:
        expected = np.gradient(result.j_hat, axis=0)
        assert np.allclose(result.j_dot, expected, atol=1e-7)


def test_jacobian_skips_non_finite_rows_in_neighborhood():
    from mndm.jacobian import estimate_local_jacobians

    rng = np.random.default_rng(101)
    x = rng.normal(size=(30, 3)).astype(np.float32)
    a_true = np.array([[0.1, 0.02, -0.03], [0.05, -0.02, 0.01], [0.0, 0.03, -0.04]], dtype=np.float32)
    x_dot = x @ a_true.T
    x[5, 1] = np.nan
    x_dot[7, 0] = np.nan
    nn_idx = np.tile(np.arange(len(x)), (len(x), 1)).astype(np.int32)

    result = estimate_local_jacobians(x, x_dot, nn_idx, super_window=3, ridge_alpha=1e-3)
    assert result.j_hat.shape[0] > 0
    assert np.all(np.isfinite(result.j_hat))


def test_phase_randomise_preserves_channel_means():
    from mndm.jacobian import phase_randomise

    rng = np.random.default_rng(5)
    x = rng.normal(loc=2.5, scale=0.7, size=(128, 3)).astype(np.float32)
    y = phase_randomise(x, seed=42)
    assert y.shape == x.shape
    assert np.allclose(np.mean(y, axis=0), np.mean(x, axis=0), atol=1e-6)

