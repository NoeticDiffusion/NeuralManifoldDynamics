"""Tests for MNPS Jacobian estimation."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_jacobian_recovers_linear_system():
    """Test jacobian recovers linear system."""
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
    """Test jacobian centers align with successful windows."""
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
    """Test jacobian dot is timeline aligned and zero at start."""
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
    """Test jacobian skips non finite rows in neighborhood."""
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
    """Test phase randomise preserves channel means."""
    from mndm.jacobian import phase_randomise

    rng = np.random.default_rng(5)
    x = rng.normal(loc=2.5, scale=0.7, size=(128, 3)).astype(np.float32)
    y = phase_randomise(x, seed=42)
    assert y.shape == x.shape
    assert np.allclose(np.mean(y, axis=0), np.mean(x, axis=0), atol=1e-6)


def test_jacobian_is_exactly_deterministic_for_identical_inputs():
    """Test jacobian is exactly deterministic for identical inputs."""
    from mndm.jacobian import estimate_local_jacobians

    rng = np.random.default_rng(2026)
    x = rng.normal(size=(48, 3)).astype(np.float32)
    a_true = np.array([[0.1, -0.05, 0.02], [0.04, -0.03, 0.01], [-0.02, 0.03, -0.06]], dtype=np.float32)
    x_dot = x @ a_true.T
    nn_idx = np.tile(np.arange(len(x)), (len(x), 1)).astype(np.int32)

    first = estimate_local_jacobians(x, x_dot, nn_idx, super_window=3, ridge_alpha=1e-4)
    second = estimate_local_jacobians(x, x_dot, nn_idx, super_window=3, ridge_alpha=1e-4)

    assert np.array_equal(first.centers, second.centers)
    assert np.array_equal(first.j_hat, second.j_hat)
    assert np.array_equal(first.j_dot, second.j_dot)


def test_estimate_local_jacobians_gathers_neighbours_once(monkeypatch):
    """The canonical estimator should reuse each center's gathered neighbours."""
    from mndm import jacobian

    rng = np.random.default_rng(2028)
    x = rng.normal(size=(30, 3)).astype(np.float32)
    x_dot = rng.normal(size=(30, 3)).astype(np.float32)
    nn_idx = np.tile(np.arange(len(x)), (len(x), 1)).astype(np.int32)
    gathered_centers = []
    original_gather = jacobian._gather_indices

    def _counting_gather(center, *args, **kwargs):
        gathered_centers.append(int(center))
        return original_gather(center, *args, **kwargs)

    monkeypatch.setattr(jacobian, "_gather_indices", _counting_gather)

    result = jacobian.estimate_local_jacobians(
        x,
        x_dot,
        nn_idx,
        super_window=3,
        ridge_alpha=1e-4,
    )

    expected_centers = np.arange(1, len(x) - 1, dtype=np.int32)
    assert np.array_equal(result.centers, expected_centers)
    assert gathered_centers == expected_centers.tolist()


def test_estimate_anchor_coupling_exports_cross_blocks_and_metrics():
    from mndm.jacobian import estimate_anchor_coupling

    rng = np.random.default_rng(2027)
    x = rng.normal(size=(40, 3)).astype(np.float32)
    a = rng.normal(size=(40, 2)).astype(np.float32)
    j_xx = np.array([[0.1, 0.0, 0.02], [0.01, -0.05, 0.0], [0.0, 0.03, -0.07]], dtype=np.float32)
    j_xa = np.array([[0.2, -0.1], [0.05, 0.04], [-0.02, 0.08]], dtype=np.float32)
    j_ax = np.array([[0.03, -0.02, 0.01], [0.04, 0.01, -0.05]], dtype=np.float32)
    j_aa = np.array([[0.02, -0.01], [0.0, 0.03]], dtype=np.float32)
    x_dot = x @ j_xx.T + a @ j_xa.T
    a_dot = x @ j_ax.T + a @ j_aa.T
    nn_idx = np.tile(np.arange(len(x)), (len(x), 1)).astype(np.int32)

    result = estimate_anchor_coupling(
        x,
        x_dot,
        a,
        a_dot,
        nn_idx,
        super_window=1,
        ridge_alpha=1e-6,
    )

    assert result["J_z"].shape[1:] == (5, 5)
    assert result["J_xa"].shape[1:] == (3, 2)
    assert result["J_ax"].shape[1:] == (2, 3)
    assert result["metrics"].shape[1] == 4
    assert result["metric_names"] == [
        "forward_drive_fro",
        "reverse_drive_fro",
        "directional_asymmetry",
        "rotational_exchange",
    ]
    assert np.isfinite(result["metrics"]).any()


def test_jacobian_diagnostics_expose_neighborhood_support():
    """Per-window neighborhood size must be inspectable next to the fit.

    The 8 s / 4 s MNPS grid is the Jacobian *center* lattice. The affine fit
    already pools unique indices from super_window time neighbors, each
    contributing knn_k chart neighbors. Diagnostics must expose that support
    rather than leaving consumers to infer it from window_start/window_end.
    """
    from mndm.jacobian import estimate_local_jacobians

    rng = np.random.default_rng(365)
    x = rng.normal(size=(40, 3)).astype(np.float32)
    a_true = np.array(
        [[0.1, -0.05, 0.02], [0.03, -0.08, 0.01], [-0.02, 0.04, -0.05]],
        dtype=np.float32,
    )
    x_dot = x @ a_true.T
    nn_idx = np.vstack(
        [rng.choice(len(x), size=5, replace=False).astype(np.int32) for _ in range(len(x))]
    )

    result = estimate_local_jacobians(
        x,
        x_dot,
        nn_idx,
        super_window=3,
        ridge_alpha=1.0,
        distance_weighted=True,
        knn_k=5,
    )

    diag = result.diagnostics
    assert result.j_hat.shape[0] > 0
    assert diag["support_mode"] == "knn"
    assert diag["knn_k"] == 5
    assert diag["super_window"] == 3
    assert diag["ridge_alpha"] == 1.0
    assert diag["distance_weighted"] is True
    assert diag["min_samples"] == 4
    assert diag["n_affine_parameters"] == 12
    n_samples = np.asarray(diag["n_neighborhood_samples"])
    assert n_samples.shape[0] == result.j_hat.shape[0]
    assert np.all(n_samples >= 4)
    assert np.isclose(diag["n_neighborhood_samples_median"], float(np.median(n_samples)))
    assert diag["n_neighborhood_samples_min"] == float(np.min(n_samples))


def test_infer_knn_k_records_effective_nn_idx_width():
    """Requested k must not overstate a short-recording cap in nn_idx."""
    from mndm.jacobian import infer_knn_k

    nn_idx = np.zeros((5, 4), dtype=np.int32)
    assert infer_knn_k(nn_idx, knn_k=20) == 4
    assert infer_knn_k(nn_idx) == 4
    assert infer_knn_k(np.zeros((0,), dtype=np.int32), knn_k=20) == 20


def test_time_local_support_mode_recovers_linear_system():
    """Time-local affine fit must recover a known linear generator."""
    from mndm.jacobian import estimate_local_jacobians

    rng = np.random.default_rng(379)
    x = rng.normal(size=(80, 3)).astype(np.float32)
    x -= x.mean(axis=0, keepdims=True)
    a_true = np.array(
        [[0.12, -0.04, 0.02], [0.03, -0.09, 0.01], [-0.02, 0.05, -0.06]],
        dtype=np.float32,
    )
    x_dot = x @ a_true.T
    empty_nn = np.zeros((0,), dtype=np.int32)

    result = estimate_local_jacobians(
        x,
        x_dot,
        empty_nn,
        super_window=15,
        ridge_alpha=1e-6,
        support_mode="time_local",
    )

    assert result.diagnostics["support_mode"] == "time_local"
    assert result.diagnostics["knn_k"] == 0
    assert int(result.diagnostics["min_samples"]) == 12
    assert result.j_hat.shape[0] > 0
    mean_j = result.j_hat.mean(axis=0)
    assert np.allclose(mean_j, a_true, atol=2e-2)
    assert float(result.diagnostics["rel_mse_baseline_median"]) < 0.05


def test_default_knn_support_mode_is_bit_identical_to_explicit_knn():
    from mndm.jacobian import estimate_local_jacobians

    rng = np.random.default_rng(380)
    x = rng.normal(size=(40, 3)).astype(np.float32)
    a_true = np.array(
        [[0.1, -0.05, 0.02], [0.03, -0.08, 0.01], [-0.02, 0.04, -0.05]],
        dtype=np.float32,
    )
    x_dot = x @ a_true.T
    nn_idx = np.vstack(
        [rng.choice(len(x), size=5, replace=False).astype(np.int32) for _ in range(len(x))]
    )

    implicit = estimate_local_jacobians(x, x_dot, nn_idx, super_window=3, ridge_alpha=1e-4)
    explicit = estimate_local_jacobians(
        x, x_dot, nn_idx, super_window=3, ridge_alpha=1e-4, support_mode="knn"
    )
    assert np.array_equal(implicit.j_hat, explicit.j_hat)
    assert implicit.diagnostics["support_mode"] == "knn"


def test_time_local_beats_knn_on_switching_linear_system():
    """Shared state cloud, two maps in time: chart kNN mixes, time-local does not."""
    from mndm.jacobian import estimate_local_jacobians
    from mndm.projection import build_knn_indices

    rng = np.random.default_rng(382)
    a_cw = np.array([[-0.15, -1.2], [1.2, -0.15]], dtype=np.float32)
    a_ccw = np.array([[-0.15, 1.2], [-1.2, -0.15]], dtype=np.float32)
    n_half = 160
    x1 = rng.normal(size=(n_half, 2)).astype(np.float32)
    x2 = rng.normal(size=(n_half, 2)).astype(np.float32)
    x = np.vstack([x1, x2])
    x_dot = np.vstack([x1 @ a_cw.T, x2 @ a_ccw.T])

    nn_idx = build_knn_indices(x, k=20, metric="euclidean", whiten=True)
    knn = estimate_local_jacobians(
        x, x_dot, nn_idx, super_window=3, ridge_alpha=1e-6, knn_k=20
    )
    time_local = estimate_local_jacobians(
        x,
        x_dot,
        np.zeros((0,), dtype=np.int32),
        super_window=11,
        ridge_alpha=1e-6,
        support_mode="time_local",
    )
    knn_rel = float(knn.diagnostics["rel_mse_baseline_median"])
    local_rel = float(time_local.diagnostics["rel_mse_baseline_median"])
    assert local_rel < 0.15
    assert knn_rel > 0.4
    assert local_rel < knn_rel - 0.2

    second_half = time_local.centers >= n_half + 5
    assert np.any(second_half)
    mean_second = time_local.j_hat[second_half].mean(axis=0)
    assert np.allclose(mean_second, a_ccw, atol=0.25)


def test_both_support_modes_fail_gate_on_pure_noise():
    from mndm.jacobian import estimate_local_jacobians
    from mndm.projection import build_knn_indices

    rng = np.random.default_rng(381)
    x = rng.normal(size=(200, 3)).astype(np.float32)
    x_dot = rng.normal(size=(200, 3)).astype(np.float32)
    nn_idx = build_knn_indices(x, k=20, metric="euclidean", whiten=True)
    knn = estimate_local_jacobians(x, x_dot, nn_idx, super_window=3, ridge_alpha=1.0, knn_k=20)
    time_local = estimate_local_jacobians(
        x, x_dot, nn_idx, super_window=41, ridge_alpha=1.0, support_mode="time_local"
    )
    assert float(knn.diagnostics["rel_mse_baseline_median"]) >= 0.9
    assert float(time_local.diagnostics["rel_mse_baseline_median"]) >= 0.9


def test_time_local_short_window_is_insufficient_for_3d_affine():
    """Fewer samples than affine parameters must not emit a 3D time-local J.

    3D affine map has 12 parameters. super_window=7 is an in-sample overfit
    trap if those rows were fitted; time_local therefore requires
    n_affine_parameters samples.
    """
    from mndm.jacobian import estimate_local_jacobians

    rng = np.random.default_rng(383)
    x = rng.normal(size=(80, 3)).astype(np.float32)
    x_dot = rng.normal(size=(80, 3)).astype(np.float32)
    result = estimate_local_jacobians(
        x,
        x_dot,
        np.zeros((0,), dtype=np.int32),
        super_window=7,
        ridge_alpha=1.0,
        support_mode="time_local",
    )
    assert int(result.diagnostics["n_affine_parameters"]) == 12
    assert int(result.diagnostics["min_samples"]) == 12
    assert result.j_hat.shape[0] == 0
    assert float(result.diagnostics["failed_insufficient_neighbours"]) > 0


def test_unknown_support_mode_is_rejected():
    from mndm.jacobian import estimate_local_jacobians
    import pytest

    x = np.zeros((10, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="support_mode"):
        estimate_local_jacobians(x, x, np.zeros((0,), dtype=np.int32), support_mode="sindy")


def test_holdout_oos_rel_mse_recovers_linear_system_without_changing_j_hat():
    from mndm.jacobian import estimate_local_jacobians

    rng = np.random.default_rng(384)
    x = rng.normal(size=(80, 3)).astype(np.float32)
    x -= x.mean(axis=0, keepdims=True)
    a_true = np.array(
        [[0.12, -0.04, 0.02], [0.03, -0.09, 0.01], [-0.02, 0.05, -0.06]],
        dtype=np.float32,
    )
    x_dot = x @ a_true.T
    nn_idx = np.tile(np.arange(len(x)), (len(x), 1)).astype(np.int32)
    result = estimate_local_jacobians(x, x_dot, nn_idx, super_window=1, ridge_alpha=1e-6)
    assert result.diagnostics["oos_holdout_stride"] == 4
    oos = np.asarray(result.diagnostics["rel_mse_baseline_oos_windows"])
    assert oos.shape[0] == result.j_hat.shape[0]
    assert float(result.diagnostics["rel_mse_baseline_oos_median"]) < 0.05
    mean_j = result.j_hat.mean(axis=0)
    assert np.allclose(mean_j, a_true, atol=1e-2)


def test_holdout_oos_rel_mse_fails_gate_on_pure_noise():
    from mndm.jacobian import estimate_local_jacobians
    from mndm.projection import build_knn_indices

    rng = np.random.default_rng(385)
    x = rng.normal(size=(200, 3)).astype(np.float32)
    x_dot = rng.normal(size=(200, 3)).astype(np.float32)
    nn_idx = build_knn_indices(x, k=20, metric="euclidean", whiten=True)
    knn = estimate_local_jacobians(x, x_dot, nn_idx, super_window=3, ridge_alpha=1.0, knn_k=20)
    time_local = estimate_local_jacobians(
        x, x_dot, nn_idx, super_window=41, ridge_alpha=1.0, support_mode="time_local"
    )
    assert float(knn.diagnostics["rel_mse_baseline_oos_median"]) >= 0.9
    assert float(time_local.diagnostics["rel_mse_baseline_oos_median"]) >= 0.9

