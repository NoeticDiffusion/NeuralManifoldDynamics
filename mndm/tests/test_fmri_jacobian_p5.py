"""P5: fMRI Jacobian support is not_testable when the affine fit is underdetermined."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.jacobian import (
    estimate_local_jacobians,
    jacobian_segment_ids,
    restrict_indices_to_segment,
)
from mndm.pipeline.robustness_helpers import apply_standard_jacobian_window_policy
from mndm.pipeline.summary import jacobian_support_kwargs


def _dense_nn(n: int) -> np.ndarray:
    nn = np.zeros((n, max(n - 1, 1)), dtype=np.int32)
    for i in range(n):
        nn[i] = np.asarray([j for j in range(n) if j != i], dtype=np.int32)
    return nn


def test_even_super_window_is_realized_odd_and_recorded():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(40, 3)).astype(np.float32)
    x_dot = rng.normal(size=(40, 3)).astype(np.float32)
    nn = _dense_nn(40)[:, :8]
    even = estimate_local_jacobians(x, x_dot, nn, super_window=2, ridge_alpha=1e-3, knn_k=8)
    odd = estimate_local_jacobians(x, x_dot, nn, super_window=3, ridge_alpha=1e-3, knn_k=8)
    assert even.diagnostics["super_window_requested"] == 2
    assert even.diagnostics["super_window"] == 3
    assert odd.diagnostics["super_window_requested"] == 3
    np.testing.assert_allclose(even.j_hat, odd.j_hat)


def test_short_9d_trajectory_is_not_testable():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(8, 9)).astype(np.float32)
    x_dot = rng.normal(size=(8, 9)).astype(np.float32)
    result = estimate_local_jacobians(
        x,
        x_dot,
        _dense_nn(8),
        super_window=3,
        ridge_alpha=1.0,
        knn_k=7,
        require_determined_support=True,
    )
    assert result.diagnostics["computation_status"] == "not_testable"
    assert result.diagnostics["n_affine_parameters"] == 90
    assert result.j_hat.shape == (0, 9, 9)
    assert result.j_hat.size == 0
    _, policy = apply_standard_jacobian_window_policy(result)
    assert policy["status"] == "not_testable"
    assert policy["windows_retained"] == 0


def test_3d_ridge_fit_below_parameter_count_is_flagged_then_refused():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(16, 3)).astype(np.float32)
    x_dot = rng.normal(size=(16, 3)).astype(np.float32)
    nn = _dense_nn(16)[:, :3]
    flagged = estimate_local_jacobians(x, x_dot, nn, super_window=1, ridge_alpha=1.0, knn_k=3)
    assert flagged.j_hat.shape[0] > 0
    assert flagged.diagnostics["computation_status"] == "ok"
    assert flagged.diagnostics["min_samples"] == 4
    assert flagged.diagnostics["n_affine_parameters"] == 12
    assert flagged.diagnostics["ridge_underdetermined"] is True
    assert np.all(np.asarray(flagged.diagnostics["n_neighborhood_samples"]) < 12)

    refused = estimate_local_jacobians(
        x,
        x_dot,
        nn,
        super_window=1,
        ridge_alpha=1.0,
        knn_k=3,
        require_determined_support=True,
    )
    assert refused.diagnostics["computation_status"] == "not_testable"
    assert refused.diagnostics["min_samples"] == 12
    assert refused.j_hat.shape[0] == 0


def test_segment_mask_drops_cross_gap_neighbors():
    t_start = np.array([0.0, 2.0, 4.0, 40.0, 42.0, 44.0])
    ids = jacobian_segment_ids(6, t_start, dt=2.0)
    assert list(ids) == [0, 0, 0, 1, 1, 1]
    files = np.array(["a", "a", "b", "b", "b", "b"])
    split = jacobian_segment_ids(6, np.arange(6, dtype=float) * 2.0, dt=2.0, file_ids=files)
    assert split[1] != split[2]

    kept = restrict_indices_to_segment(np.arange(6, dtype=np.int32), 1, ids)
    assert list(kept) == [0, 1, 2]

    rng = np.random.default_rng(11)
    x = rng.normal(size=(20, 3)).astype(np.float32)
    x_dot = rng.normal(size=(20, 3)).astype(np.float32)
    nn = _dense_nn(20)
    seg = np.array([0] * 10 + [1] * 10, dtype=np.int32)
    closed = estimate_local_jacobians(
        x, x_dot, nn, super_window=1, ridge_alpha=1.0, knn_k=19, segment_id=seg
    )
    opened = estimate_local_jacobians(x, x_dot, nn, super_window=1, ridge_alpha=1.0, knn_k=19)
    assert closed.diagnostics["n_neighborhood_samples_min"] <= 10
    assert opened.diagnostics["n_neighborhood_samples_min"] > 10


def test_support_kwargs_inherit_fmri_flags_and_match_length():
    frame = pd.DataFrame(
        {
            "t_start": [0.0, 15.0, 80.0],
            "file": ["a", "a", "a"],
        }
    )
    child = jacobian_support_kwargs(
        frame,
        3,
        15.0,
        {"enabled": True},
        inherit={"require_determined_support": True, "forbid_cross_gap": True},
    )
    assert child["require_determined_support"] is True
    assert list(child["segment_id"]) == [0, 0, 1]
    mismatched = jacobian_support_kwargs(
        frame,
        2,
        15.0,
        {"forbid_cross_gap": True},
    )
    assert mismatched["forbid_cross_gap"] is True
    assert mismatched["segment_id"] is None
    bare = jacobian_support_kwargs(
        pd.DataFrame({"epoch": [0, 1, 2]}),
        3,
        15.0,
        {"forbid_cross_gap": True},
    )
    assert bare["segment_id"] is None
    rng = np.random.default_rng(2)
    x = rng.normal(size=(12, 3)).astype(np.float32)
    refused = estimate_local_jacobians(
        x,
        rng.normal(size=(12, 3)).astype(np.float32),
        _dense_nn(12)[:, :5],
        super_window=1,
        knn_k=5,
        forbid_cross_gap=True,
    )
    assert refused.diagnostics["computation_status"] == "not_testable"
    assert refused.diagnostics["computation_status_reason"] == "gap_boundaries_unresolved"
    assert refused.j_hat.shape[0] == 0
