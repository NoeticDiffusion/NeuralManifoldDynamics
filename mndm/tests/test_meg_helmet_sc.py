"""Tests for the claim-bounded MEG helmet SC chart utilities."""

from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.tools.meg_helmet_sc import (
    compute_helmet_sc_chart,
    degree_preserving_sc_null,
    helmet_geometry_graph,
    project_connectome_to_helmet,
    shuffled_sensitivity,
)


def test_project_connectome_to_helmet_normalizes_group_sensitivity():
    sc = np.array([[0.0, 2.0, 1.0], [2.0, 0.0, 3.0], [1.0, 3.0, 0.0]])
    sensitivity = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 1.0]])

    projected = project_connectome_to_helmet(sc, sensitivity)

    assert projected.shape == (2, 2)
    assert np.allclose(projected, projected.T)
    assert np.allclose(np.diag(projected), 0.0)
    assert projected[0, 1] > 0.0


def test_helmet_sc_chart_has_windowed_alignment_axes():
    rng = np.random.default_rng(12)
    mnps = rng.normal(size=(50, 3, 3))
    graph = helmet_geometry_graph(np.array([[-0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.1, 0.0, 0.0]]))

    chart = compute_helmet_sc_chart(mnps, graph, window=20, step=10, tau=0.1, n_low_modes=1)

    assert len(chart) == 4
    assert {"m_sc", "d_sc", "e_sc", "smoothness", "low_mode_fraction", "diffusion_r2"} <= set(chart.columns)
    assert np.isfinite(chart["smoothness"]).all()


def test_structural_nulls_keep_required_shapes():
    rng = np.random.default_rng(5)
    sc = np.array([[0.0, 1.0, 2.0, 0.0], [1.0, 0.0, 3.0, 4.0], [2.0, 3.0, 0.0, 5.0], [0.0, 4.0, 5.0, 0.0]])
    sensitivity = np.eye(4)[:3]

    assert shuffled_sensitivity(sensitivity, rng).shape == sensitivity.shape
    null_sc = degree_preserving_sc_null(sc, rng)
    assert null_sc.shape == sc.shape
    assert np.allclose(null_sc, null_sc.T)
