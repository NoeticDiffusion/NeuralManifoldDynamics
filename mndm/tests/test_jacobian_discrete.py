"""Tests for the experimental discrete one-step map estimator.

This family fits ``x_{t+h} ≈ Φ(x_t - x̄) + b`` and does not use ``x_dot``.
It is not wired into summarize and is not a production Jacobian.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _all_nn(n: int) -> np.ndarray:
    return np.tile(np.arange(n), (n, 1)).astype(np.int32)


def _paired_cloud(phi: np.ndarray, n_pairs: int, rng: np.random.Generator) -> np.ndarray:
    """Independent (x, Φx) pairs separated by NaN so horizon=1 is well-posed.

    A single iterated orbit may be rank-deficient in sampled state space and
    then cannot identify a full matrix Φ. Each pair is a 3-row block:
    source, image, gap.
    """
    dim = int(phi.shape[0])
    rows = np.full((n_pairs * 3, dim), np.nan, dtype=np.float32)
    src = rng.normal(size=(n_pairs, dim)).astype(np.float32)
    tgt = (src @ np.asarray(phi, dtype=np.float32).T).astype(np.float32)
    rows[0::3] = src
    rows[1::3] = tgt
    return rows


def test_discrete_map_recovers_linear_one_step():
    from mndm.jacobian_discrete import estimate_local_discrete_maps

    rng = np.random.default_rng(401)
    a_cont = np.array(
        [[0.10, -0.05, 0.02], [0.03, -0.08, 0.01], [-0.02, 0.04, -0.05]],
        dtype=np.float32,
    )
    dt = 0.25
    phi_true = np.eye(3, dtype=np.float32) + dt * a_cont
    x = _paired_cloud(phi_true, 40, rng)
    src = x[0::3]
    src = src[np.isfinite(src).all(axis=1)]
    assert int(np.linalg.matrix_rank(src.astype(np.float64))) == 3

    result = estimate_local_discrete_maps(
        x,
        _all_nn(len(x)),
        super_window=1,
        ridge_alpha=1e-6,
        horizon=1,
        dt_sec=dt,
    )
    assert result.phi_hat.shape[1:] == (3, 3)
    mean_phi = result.phi_hat.mean(axis=0)
    assert np.allclose(mean_phi, phi_true, atol=0.05)
    mean_euler = result.euler_j.mean(axis=0)
    assert np.allclose(mean_euler, a_cont, atol=0.08)
    assert float(result.diagnostics["rel_mse_baseline_median"]) < 0.05
    assert float(result.diagnostics["rel_mse_baseline_oos_median"]) < 0.15
    assert result.diagnostics["estimator_family"] == "discrete_one_step_map"
    assert int(result.diagnostics["horizon"]) == 1


def test_discrete_map_fails_on_independent_noise():
    from mndm.jacobian_discrete import estimate_local_discrete_maps
    from mndm.projection import build_knn_indices

    rng = np.random.default_rng(402)
    x = rng.normal(size=(200, 3)).astype(np.float32)
    nn_idx = build_knn_indices(x, k=20, metric="euclidean", whiten=True)
    knn = estimate_local_discrete_maps(
        x, nn_idx, super_window=3, ridge_alpha=1.0, knn_k=20, dt_sec=4.0
    )
    time_local = estimate_local_discrete_maps(
        x,
        np.zeros((0,), dtype=np.int32),
        super_window=41,
        ridge_alpha=1.0,
        support_mode="time_local",
        dt_sec=4.0,
    )
    assert float(knn.diagnostics["rel_mse_baseline_median"]) >= 0.9
    assert float(knn.diagnostics["rel_mse_baseline_oos_median"]) >= 0.9
    assert float(time_local.diagnostics["rel_mse_baseline_median"]) >= 0.9
    assert float(time_local.diagnostics["rel_mse_baseline_oos_median"]) >= 0.9


def test_discrete_map_is_not_the_xdot_jacobian():
    from mndm.jacobian import estimate_local_jacobians
    from mndm.jacobian_discrete import estimate_local_discrete_maps

    rng = np.random.default_rng(403)
    a_cont = np.array(
        [[0.12, -0.04, 0.02], [0.03, -0.09, 0.01], [-0.02, 0.05, -0.06]],
        dtype=np.float32,
    )
    dt = 0.25
    phi_true = np.eye(3, dtype=np.float32) + dt * a_cont
    x = _paired_cloud(phi_true, 36, rng)
    x_dot = np.full_like(x, np.nan)
    src = np.arange(0, len(x), 3)
    src = src[src + 1 < len(x)]
    x_dot[src] = (x[src + 1] - x[src]) / dt
    nn_idx = _all_nn(len(x))
    jac = estimate_local_jacobians(x, x_dot, nn_idx, super_window=1, ridge_alpha=1e-6)
    disc = estimate_local_discrete_maps(
        x, nn_idx, super_window=1, ridge_alpha=1e-6, dt_sec=dt
    )
    assert jac.j_hat.shape[0] > 0 and disc.phi_hat.shape[0] > 0
    assert not np.allclose(jac.j_hat.mean(axis=0), disc.phi_hat.mean(axis=0), atol=0.2)
    assert np.allclose(jac.j_hat.mean(axis=0), a_cont, atol=0.05)
    assert np.allclose(disc.phi_hat.mean(axis=0), phi_true, atol=0.05)


def test_time_local_discrete_beats_knn_on_switching_map():
    from mndm.jacobian_discrete import estimate_local_discrete_maps
    from mndm.projection import build_knn_indices

    rng = np.random.default_rng(404)
    a_cw = np.array([[-0.15, -1.2], [1.2, -0.15]], dtype=np.float32)
    a_ccw = np.array([[-0.15, 1.2], [-1.2, -0.15]], dtype=np.float32)
    dt = 0.2
    phi_cw = np.eye(2, dtype=np.float32) + dt * a_cw
    phi_ccw = np.eye(2, dtype=np.float32) + dt * a_ccw
    n_half = 80
    x = np.vstack(
        [_paired_cloud(phi_cw, n_half, rng), _paired_cloud(phi_ccw, n_half, rng)]
    )
    split = n_half * 3

    nn_idx = build_knn_indices(np.nan_to_num(x, nan=0.0), k=20, metric="euclidean", whiten=True)
    knn = estimate_local_discrete_maps(
        x, nn_idx, super_window=3, ridge_alpha=1e-6, knn_k=20, dt_sec=dt
    )
    time_local = estimate_local_discrete_maps(
        x,
        np.zeros((0,), dtype=np.int32),
        super_window=21,
        ridge_alpha=1e-6,
        support_mode="time_local",
        dt_sec=dt,
    )
    knn_rel = float(knn.diagnostics["rel_mse_baseline_median"])
    local_rel = float(time_local.diagnostics["rel_mse_baseline_median"])
    assert local_rel < 0.25
    assert knn_rel > local_rel + 0.15

    second_half = time_local.centers >= split + 9
    assert np.any(second_half)
    mean_second = time_local.phi_hat[second_half].mean(axis=0)
    assert np.allclose(mean_second, phi_ccw, atol=0.15)


def test_time_local_discrete_short_window_is_insufficient_for_3d_affine():
    from mndm.jacobian_discrete import estimate_local_discrete_maps

    rng = np.random.default_rng(405)
    x = rng.normal(size=(80, 3)).astype(np.float32)
    result = estimate_local_discrete_maps(
        x,
        np.zeros((0,), dtype=np.int32),
        super_window=7,
        ridge_alpha=1.0,
        support_mode="time_local",
        dt_sec=4.0,
    )
    assert int(result.diagnostics["n_affine_parameters"]) == 12
    assert int(result.diagnostics["min_samples"]) == 12
    assert result.phi_hat.shape[0] == 0
    assert float(result.diagnostics["failed_insufficient_pairs"]) > 0


def test_unknown_support_mode_is_rejected_for_discrete_maps():
    from mndm.jacobian_discrete import estimate_local_discrete_maps

    x = np.zeros((10, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="support_mode"):
        estimate_local_discrete_maps(x, np.zeros((0,), dtype=np.int32), support_mode="sindy")


def test_empty_and_short_recordings_return_empty_with_pair_failure_keys():
    from mndm.jacobian_discrete import estimate_local_discrete_maps

    empty = estimate_local_discrete_maps(
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((0,), dtype=np.int32),
        super_window=3,
        dt_sec=4.0,
    )
    assert empty.phi_hat.shape == (0, 3, 3)
    assert int(empty.diagnostics["failed_insufficient_pairs"]) == 0
    assert int(empty.diagnostics["failed_nonfinite_center"]) == 0

    short = estimate_local_discrete_maps(
        np.ones((4, 3), dtype=np.float32),
        np.zeros((0,), dtype=np.int32),
        super_window=3,
        support_mode="knn",
        dt_sec=4.0,
    )
    assert short.phi_hat.shape[0] == 0
    assert "failed_insufficient_pairs" in short.diagnostics


def test_horizon_two_recovers_placed_one_step_image():
    from mndm.jacobian_discrete import estimate_local_discrete_maps

    rng = np.random.default_rng(406)
    phi_true = np.array(
        [[0.9, -0.1, 0.05], [0.04, 0.88, 0.02], [-0.03, 0.06, 0.91]],
        dtype=np.float32,
    )
    n_pairs = 36
    horizon = 2
    block = 2 * horizon + 1
    rows = np.full((n_pairs * block, 3), np.nan, dtype=np.float32)
    src = rng.normal(size=(n_pairs, 3)).astype(np.float32)
    rows[0::block] = src
    rows[horizon::block] = src @ phi_true.T
    result = estimate_local_discrete_maps(
        rows,
        _all_nn(len(rows)),
        super_window=1,
        ridge_alpha=1e-6,
        horizon=horizon,
        dt_sec=1.0,
    )
    assert int(result.diagnostics["horizon"]) == 2
    assert result.phi_hat.shape[0] > 0
    assert np.allclose(result.phi_hat.mean(axis=0), phi_true, atol=0.05)


def test_nonfinite_center_is_rejected_when_distance_weighted():
    from mndm.jacobian_discrete import estimate_local_discrete_maps

    rng = np.random.default_rng(407)
    phi_true = np.eye(3, dtype=np.float32) * 0.9
    x = _paired_cloud(phi_true, 24, rng)
    finite = estimate_local_discrete_maps(
        x,
        _all_nn(len(x)),
        super_window=1,
        ridge_alpha=1e-6,
        distance_weighted=True,
        dt_sec=1.0,
    )
    assert finite.phi_hat.shape[0] > 0
    poisoned = x.copy()
    poisoned[int(finite.centers[0])] = np.nan
    result = estimate_local_discrete_maps(
        poisoned,
        _all_nn(len(poisoned)),
        super_window=1,
        ridge_alpha=1e-6,
        distance_weighted=True,
        dt_sec=1.0,
    )
    assert float(result.diagnostics["failed_nonfinite_center"]) >= 1
    if result.phi_hat.shape[0] > 0:
        assert np.isfinite(result.phi_hat).all()
        assert int(finite.centers[0]) not in set(int(c) for c in result.centers)
