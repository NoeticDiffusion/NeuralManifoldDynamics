"""Tests for experimental 3D→3D vs 9D→3D cross-affine maps.

The 9→3 map predicts ẋ^(3) from x^(9) on frozen support indices.
It is not a 9D Jacobian and has no spectral abscissa.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent / "src"))
sys.path.insert(0, str(TESTS))

from family_b_cross_affine import (  # noqa: E402
    estimate_local_cross_affine,
    n_cross_affine_parameters,
)


def _family_mean_p() -> np.ndarray:
    p = np.zeros((9, 3), dtype=np.float32)
    scale = float(np.sqrt(3.0))
    for axis in range(3):
        p[3 * axis : 3 * axis + 3, axis] = 1.0 / scale
    return p


def _all_nn(n: int) -> np.ndarray:
    return np.tile(np.arange(n), (n, 1)).astype(np.int32)


def test_n_cross_affine_parameters_match_sl001():
    assert n_cross_affine_parameters(3, 3) == 12
    assert n_cross_affine_parameters(9, 3) == 30
    assert n_cross_affine_parameters(9, 9) == 90


def test_cross_3_to_3_matches_square_jacobian():
    from mndm.jacobian import estimate_local_jacobians

    rng = np.random.default_rng(501)
    x = rng.normal(size=(80, 3)).astype(np.float32)
    x -= x.mean(axis=0, keepdims=True)
    a_true = np.array(
        [[0.12, -0.04, 0.02], [0.03, -0.09, 0.01], [-0.02, 0.05, -0.06]],
        dtype=np.float32,
    )
    x_dot = x @ a_true.T
    nn_idx = _all_nn(len(x))
    square = estimate_local_jacobians(x, x_dot, nn_idx, super_window=1, ridge_alpha=1e-6)
    cross = estimate_local_cross_affine(x, x_dot, nn_idx, super_window=1, ridge_alpha=1e-6)
    assert np.array_equal(square.centers, cross.centers)
    assert np.allclose(square.j_hat, cross.map_hat, atol=1e-5)
    assert cross.map_hat.shape[1:] == (3, 3)
    assert int(cross.diagnostics["n_affine_parameters"]) == 12


def test_closed_projection_both_maps_recover():
    rng = np.random.default_rng(502)
    p = _family_mean_p()
    j3 = np.array(
        [[-0.2, 0.05, 0.0], [-0.04, -0.15, 0.03], [0.02, -0.01, -0.1]],
        dtype=np.float32,
    )
    j9 = p @ j3 @ p.T
    x9 = rng.normal(size=(120, 9)).astype(np.float32)
    x3 = x9 @ p
    x3_dot = x9 @ j9.T @ p
    nn_idx = _all_nn(len(x9))
    m3 = estimate_local_cross_affine(x3, x3_dot, nn_idx, super_window=1, ridge_alpha=1e-6)
    m9 = estimate_local_cross_affine(
        x9, x3_dot, nn_idx, super_window=1, ridge_alpha=1e-6, x_metric=x3
    )
    assert float(m3.diagnostics["rel_mse_baseline_oos_median"]) < 0.05
    assert float(m9.diagnostics["rel_mse_baseline_oos_median"]) < 0.05
    assert np.allclose(m3.map_hat.mean(axis=0), j3, atol=0.08)


def test_hidden_9d_state_beats_3d_on_projected_velocity():
    rng = np.random.default_rng(503)
    p = _family_mean_p()
    j3 = np.array(
        [[-0.15, 0.02, 0.0], [0.0, -0.12, 0.04], [0.03, 0.0, -0.08]],
        dtype=np.float32,
    )
    j9 = p @ j3 @ p.T
    extra = np.zeros((9, 9), dtype=np.float32)
    extra[3, 0] = 1.4
    extra[3, 1] = -1.4
    extra[4, 0] = 1.4
    extra[4, 1] = -1.4
    extra[5, 0] = 1.4
    extra[5, 1] = -1.4
    j9 = j9 + extra
    x9 = rng.normal(size=(160, 9)).astype(np.float32)
    x3 = x9 @ p
    x3_dot = x9 @ j9.T @ p
    nn_idx = _all_nn(len(x9))
    m3 = estimate_local_cross_affine(x3, x3_dot, nn_idx, super_window=1, ridge_alpha=1e-6)
    m9 = estimate_local_cross_affine(
        x9, x3_dot, nn_idx, super_window=1, ridge_alpha=1e-6, x_metric=x3
    )
    rel3 = float(m3.diagnostics["rel_mse_baseline_oos_median"])
    rel9 = float(m9.diagnostics["rel_mse_baseline_oos_median"])
    assert rel9 < 0.15
    assert rel3 > rel9 + 0.4
    assert m9.map_hat.shape[1:] == (3, 9)


def test_time_local_9_to_3_requires_thirty_samples():
    rng = np.random.default_rng(504)
    x9 = rng.normal(size=(80, 9)).astype(np.float32)
    y = rng.normal(size=(80, 3)).astype(np.float32)
    result = estimate_local_cross_affine(
        x9,
        y,
        np.zeros((0,), dtype=np.int32),
        super_window=21,
        ridge_alpha=1.0,
        support_mode="time_local",
        x_metric=x9[:, :3],
    )
    assert int(result.diagnostics["n_affine_parameters"]) == 30
    assert int(result.diagnostics["min_samples"]) == 30
    assert result.map_hat.shape[0] == 0
    assert float(result.diagnostics["failed_insufficient_neighbours"]) > 0


def test_cross_map_has_no_spectral_abscissa_field():
    rng = np.random.default_rng(505)
    x = rng.normal(size=(40, 9)).astype(np.float32)
    y = rng.normal(size=(40, 3)).astype(np.float32)
    result = estimate_local_cross_affine(x, y, _all_nn(len(x)), super_window=1, ridge_alpha=1.0)
    assert "spectral_abscissa" not in result.diagnostics
    assert result.map_hat.ndim == 3
    assert result.map_hat.shape[1] != result.map_hat.shape[2]


def test_unknown_support_mode_rejected_for_cross_affine():
    x = np.zeros((10, 9), dtype=np.float32)
    y = np.zeros((10, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="support_mode"):
        estimate_local_cross_affine(x, y, np.zeros((0,), dtype=np.int32), support_mode="sindy")
