"""Tests for MNPS projection."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def test_project_to_mnps_empty():
    from mndm.projection import project_to_mnps

    features_df = pd.DataFrame()
    weights = {"m": {"feat1": 0.5}, "d": {}, "e": {}}

    out = project_to_mnps(features_df, weights)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 0
    assert set(out.columns) == {"epoch_id", "m", "d", "e"}


def test_project_to_mnps_math():
    from mndm.projection import project_to_mnps

    features_df = pd.DataFrame({
        "epoch_id": [0, 1, 2],
        "eeg_alpha": [1.0, 2.0, 3.0],
        "eeg_beta": [0.5, 1.0, 1.5]
    })

    weights = {
        "m": {"eeg_alpha": 1.0},
        "d": {"eeg_beta": 1.0},
        "e": {}
    }

    out = project_to_mnps(features_df, weights)

    assert np.allclose(out["m"], [1.0, 2.0, 3.0])
    assert np.allclose(out["d"], [0.5, 1.0, 1.5])
    assert np.allclose(out["e"], [0.0, 0.0, 0.0])


def test_project_features_array():
    from mndm.projection import project_features

    features_df = pd.DataFrame({
        "epoch_id": [0],
        "feat_a": [1.0],
        "feat_b": [2.0],
    })

    weights = {
        "m": {"feat_a": 0.5, "feat_b": 0.5},
        "d": {"feat_b": 1.0},
        "e": {"feat_a": -1.0},
    }

    x, baselines = project_features(features_df, weights)
    assert x.shape == (1, 3)
    assert np.allclose(x[0], [1.5, 2.0, -1.0])
    assert isinstance(baselines, dict)


def test_project_features_renormalizes_when_weighted_feature_missing():
    from mndm.projection import project_features, project_features_with_coverage

    features_df = pd.DataFrame(
        {
            "epoch_id": [0, 1],
            "feat_a": [np.nan, 2.0],
            "feat_b": [2.0, 2.0],
        }
    )
    weights = {
        "m": {"feat_a": 0.5, "feat_b": 0.5},
        "d": {},
        "e": {},
    }

    x, _ = project_features(features_df, weights)
    assert x.shape == (2, 3)
    # Row 0: feat_a missing => renormalize on feat_b only (value 2.0)
    assert np.allclose(x[0, 0], 2.0)
    # Row 1: both present => weighted mean equivalent here
    assert np.allclose(x[1, 0], 2.0)
    x2, cov, _ = project_features_with_coverage(features_df, weights)
    assert np.allclose(x2, x, equal_nan=True)
    assert np.allclose(cov[:, 0], [0.5, 1.0], atol=1e-6)


def test_project_features_all_missing_axis_returns_nan():
    from mndm.projection import project_features

    features_df = pd.DataFrame(
        {
            "epoch_id": [0],
            "feat_a": [np.nan],
        }
    )
    weights = {
        "m": {"feat_a": 1.0},
        "d": {},
        "e": {},
    }

    x, _ = project_features(features_df, weights)
    assert x.shape == (1, 3)
    assert np.isnan(x[0, 0])


def test_estimate_derivatives_central():
    from mndm.projection import estimate_derivatives

    t = np.linspace(0, 1, 5)
    x = np.vstack([t, t ** 2, np.sin(t)]).T.astype(np.float32)
    x_dot = estimate_derivatives(x, dt=0.25, method="central")
    assert x_dot.shape == x.shape
    assert np.allclose(x_dot[:, 0], 1.0, atol=0.2)


def test_estimate_derivatives_propagates_nan_rows():
    from mndm.projection import estimate_derivatives

    x = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [np.nan, np.nan, np.nan],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
        ],
        dtype=np.float32,
    )
    x_dot = estimate_derivatives(x, dt=1.0, method="sav_gol", window=3, polyorder=1)
    assert x_dot.shape == x.shape
    assert np.all(np.isnan(x_dot[2]))


def test_estimate_derivatives_segmented_handles_jumps():
    from mndm.projection import estimate_derivatives_segmented

    t = np.arange(20, dtype=np.float32)
    x = np.stack([t, t, t], axis=1).astype(np.float32)
    x[10:] += 100.0  # discontinuity
    x_dot = estimate_derivatives_segmented(
        x,
        dt=1.0,
        method="sav_gol",
        max_jump=3.0,
        min_seg=5,
        savgol_window=5,
        polyorder=2,
    )
    assert x_dot.shape == x.shape
    assert np.all(np.isfinite(x_dot))


def test_knn_indices_shape():
    from mndm.projection import build_knn_indices

    x = np.random.rand(10, 3).astype(np.float32)
    nn_idx = build_knn_indices(x, k=5)
    assert nn_idx.shape == (10, 5)


def test_knn_indices_single_sample_returns_empty_neighbors():
    from mndm.projection import build_knn_indices

    x = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    nn_idx = build_knn_indices(x, k=5)
    assert nn_idx.shape == (1, 0)


def test_knn_indices_rejects_non_finite_rows():
    from mndm.projection import build_knn_indices

    x = np.array([[0.0, 1.0, 2.0], [np.nan, 1.0, 2.0]], dtype=np.float32)
    try:
        _ = build_knn_indices(x, k=1)
        assert False, "Expected ValueError for non-finite rows"
    except ValueError:
        assert True


def test_project_features_v2_bounds():
    from mndm.projection import project_features_v2

    features_df = pd.DataFrame({
        "epoch_id": [0, 1, 2, 3, 4],
        "feat_up": np.linspace(-10.0, 10.0, 5),
        "feat_down": np.linspace(10.0, -10.0, 5),
    })
    subcoords = {
        "m_a": {"feat_up": 1.0},
        "m_e": {"feat_down": -0.5},
    }

    values, names, baselines = project_features_v2(features_df, subcoords, normalize=None)

    assert names == ["m_a", "m_e"]
    assert values.shape == (5, 2)
    assert np.all(np.isfinite(values))
    assert values[0, 0] < values[-1, 0]
    assert values[0, 1] < values[-1, 1]
    assert isinstance(baselines, dict)


def test_project_features_v2_constant_maps_to_midpoint():
    from mndm.projection import project_features_v2

    features_df = pd.DataFrame({
        "epoch_id": [0, 1, 2],
        "feat_flat": [2.0, 2.0, 2.0],
    })
    subcoords = {"m_o": {"feat_flat": 1.0}}

    values, names, _ = project_features_v2(features_df, subcoords, normalize=None)

    assert names == ["m_o"]
    assert values.shape == (3, 1)
    assert np.allclose(values, 2.0, atol=1e-6)


def test_project_features_v2_missing_weighted_feature_renormalizes():
    from mndm.projection import project_features_v2

    features_df = pd.DataFrame(
        {
            "epoch_id": [0, 1, 2],
            "feat_a": [np.nan, 2.0, 2.0],
            "feat_b": [2.0, 2.0, 2.0],
        }
    )
    subcoords = {"m_a": {"feat_a": 0.5, "feat_b": 0.5}}

    values, names, _ = project_features_v2(features_df, subcoords, normalize=None)

    assert names == ["m_a"]
    assert values.shape == (3, 1)
    assert np.all(np.isfinite(values))
    # Both rows should map to the same subcoord value after per-row renormalization.
    assert np.allclose(values[0, 0], values[1, 0], atol=1e-6)


def test_derive_mde_from_v2_group_mean():
    from mndm.projection import derive_mde_from_v2

    coords_9d_names = ["m_a", "m_e", "m_o", "d_n", "d_l", "d_s", "e_e", "e_s", "e_m"]
    coords_9d = np.array(
        [
            [0.2, 0.4, 0.6, 0.1, 0.3, 0.5, 0.7, 0.9, 0.8],
            [0.1, np.nan, 0.5, 0.2, 0.2, np.nan, np.nan, 0.6, 0.6],
        ],
        dtype=np.float32,
    )
    axis_map = {
        "m": ["m_a", "m_e", "m_o"],
        "d": ["d_n", "d_l", "d_s"],
        "e": ["e_e", "e_s", "e_m"],
    }
    x, cov = derive_mde_from_v2(coords_9d, coords_9d_names, axis_map, pooling="mean")
    assert x.shape == (2, 3)
    assert cov.shape == (2, 3)
    assert np.allclose(x[0], [0.4, 0.3, 0.8], atol=1e-6)
    assert np.allclose(cov[0], [1.0, 1.0, 1.0], atol=1e-6)
    assert np.allclose(x[1], [0.3, 0.2, 0.6], atol=1e-6)
    assert np.allclose(cov[1], [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0], atol=1e-6)


def test_derive_mde_from_v2_missing_axis_columns_returns_nan_axis():
    from mndm.projection import derive_mde_from_v2

    coords_9d_names = ["m_a", "m_e", "m_o"]
    coords_9d = np.array([[0.2, 0.4, 0.6]], dtype=np.float32)
    axis_map = {
        "m": ["m_a", "m_e", "m_o"],
        "d": ["d_n", "d_l", "d_s"],
        "e": ["e_e", "e_s", "e_m"],
    }
    x, cov = derive_mde_from_v2(coords_9d, coords_9d_names, axis_map, pooling="mean")
    assert np.allclose(x[0, 0], 0.4, atol=1e-6)
    assert np.allclose(cov[0, 0], 1.0, atol=1e-6)
    assert np.isnan(x[0, 1]) and np.isnan(x[0, 2])
    assert np.isnan(cov[0, 1]) and np.isnan(cov[0, 2])


def test_derive_mde_from_v2_weighted_mapping_normalizes_columns():
    from mndm.projection import derive_mde_from_v2

    coords_9d_names = ["m_a", "m_e", "m_o", "d_n", "d_l", "d_s", "e_e", "e_s", "e_m"]
    coords_9d = np.array([[0.2, 0.4, 0.6, 0.2, 0.3, 0.1, 0.8, 0.4, 0.2]], dtype=np.float32)
    mapping = {
        "m": {"m_a": 0.8, "m_e": 0.55, "m_o": 0.2},
        "d": {"d_n": 0.65, "d_l": 0.65, "d_s": 0.35},
        "e": {"e_e": 0.8, "e_s": 0.55, "e_m": 0.2},
    }
    x, cov, info = derive_mde_from_v2(
        coords_9d,
        coords_9d_names,
        mapping,
        return_mapping_info=True,
    )
    assert x.shape == (1, 3)
    assert cov.shape == (1, 3)
    assert np.all(np.isfinite(x))
    assert np.allclose(cov, 1.0, atol=1e-6)
    assert info["aggregation"] == "fixed_weighted_projection"
    for axis in ("m", "d", "e"):
        w = np.array(list(info["weights_normalized"][axis].values()), dtype=float)
        assert np.isclose(np.linalg.norm(w), 1.0, atol=1e-6)


def test_derive_mde_from_v2_group_pooling_sum_returns_sum_not_mean():
    from mndm.projection import derive_mde_from_v2

    coords_9d_names = ["m_a", "m_e", "m_o"]
    coords_9d = np.array([[0.2, 0.4, 0.6]], dtype=np.float32)
    axis_map = {
        "m": ["m_a", "m_e", "m_o"],
        "d": [],
        "e": [],
    }
    x, cov, info = derive_mde_from_v2(
        coords_9d,
        coords_9d_names,
        axis_map,
        pooling="sum",
        return_mapping_info=True,
    )
    assert np.allclose(x[0, 0], 1.2, atol=1e-6)
    assert np.allclose(cov[0, 0], 1.0, atol=1e-6)
    assert info["aggregation"] == "group_pooling_sum"


def test_derive_mde_from_v2_rejects_cross_block_mapping():
    from mndm.projection import derive_mde_from_v2

    coords_9d_names = ["m_a", "m_e", "m_o", "d_n", "d_l", "d_s", "e_e", "e_s", "e_m"]
    coords_9d = np.ones((2, 9), dtype=np.float32)
    bad_mapping = {
        "m": {"d_n": 1.0},  # cross-block should fail
        "d": {"d_n": 1.0},
        "e": {"e_e": 1.0},
    }
    try:
        _ = derive_mde_from_v2(coords_9d, coords_9d_names, bad_mapping)
        assert False, "Expected ValueError for cross-block mapping"
    except ValueError:
        assert True


def test_construct_and_apply_fixed_projection_runtime_l2_normalization():
    from mndm.projection import apply_fixed_projection, construct_fixed_projection_matrix

    names = ["m_a", "m_e", "m_o", "d_n", "d_l", "d_s", "e_e", "e_s", "e_m"]
    mapping = {
        "m": {"m_a": 0.8, "m_e": 0.55, "m_o": 0.2},
        "d": {"d_n": 0.65, "d_l": 0.65, "d_s": 0.35},
        "e": {"e_e": 0.8, "e_s": 0.55, "e_m": 0.2},
    }
    P, normalized = construct_fixed_projection_matrix(
        names,
        mapping,
        enforce_block_selective=True,
        normalize_columns_l2=True,
        l2_epsilon=1e-9,
    )
    assert P.shape == (9, 3)
    assert np.allclose(np.linalg.norm(P, axis=0), 1.0, atol=1e-6)
    assert set(normalized.keys()) == {"m", "d", "e"}

    Xv2 = np.ones((2, 9), dtype=np.float32)
    xv1 = apply_fixed_projection(Xv2, P)
    assert xv1.shape == (2, 3)
    assert np.all(np.isfinite(xv1))


def test_build_knn_indices_fallback_excludes_self(monkeypatch):
    import mndm.projection as projection_mod

    monkeypatch.setattr(projection_mod, "cKDTree", None)
    x = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    nn_idx = projection_mod.build_knn_indices(x, k=2)
    assert nn_idx.shape == (5, 2)
    for i in range(5):
        assert i not in set(nn_idx[i].tolist())


def test_project_to_mnps_requires_epoch_id():
    from mndm.projection import project_to_mnps

    features_df = pd.DataFrame({"feat": [1.0, 2.0]})
    weights = {"m": {"feat": 1.0}, "d": {}, "e": {}}
    try:
        _ = project_to_mnps(features_df, weights)
        assert False, "Expected ValueError for missing epoch_id"
    except ValueError:
        assert True


