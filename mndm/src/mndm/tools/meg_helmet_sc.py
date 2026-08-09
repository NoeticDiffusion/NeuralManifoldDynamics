"""SC-informed charts for *sensor-topographic* MEG helmet groups.

This module deliberately does not estimate cortical sources.  It projects an
externally supplied cortical structural-connectome (SC) graph into a frozen
helmet-group sensitivity map, then measures whether the observed group MNPS
field has graph structure beyond structural and geometry controls.
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd
from scipy.linalg import expm


def _symmetric_adjacency(matrix: np.ndarray) -> np.ndarray:
    """Return a finite, symmetric non-negative adjacency with zero diagonal."""
    adjacency = np.asarray(matrix, dtype=float).copy()
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Adjacency must be square.")
    adjacency[~np.isfinite(adjacency)] = 0.0
    adjacency = np.maximum(0.0, 0.5 * (adjacency + adjacency.T))
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def project_connectome_to_helmet(sc: np.ndarray, sensitivity: np.ndarray) -> np.ndarray:
    """Project cortical SC into frozen helmet-group sensitivity space.

    ``sensitivity`` is ``[n_groups, n_cortical_parcels]`` and must be derived
    solely from sensor geometry/forward sensitivity, never from MNPS outcomes.
    Rows are normalized so output edge magnitude is not driven by group size.
    """
    adjacency = _symmetric_adjacency(sc)
    mapping = np.asarray(sensitivity, dtype=float)
    if mapping.ndim != 2 or mapping.shape[1] != adjacency.shape[0]:
        raise ValueError("Sensitivity must have shape [n_groups, n_sc_nodes].")
    mapping = np.maximum(mapping, 0.0)
    row_sum = mapping.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0.0):
        raise ValueError("Every helmet group needs non-zero cortical sensitivity.")
    mapping = mapping / row_sum
    projected = mapping @ adjacency @ mapping.T
    return _symmetric_adjacency(projected)


def helmet_geometry_graph(sensor_positions: np.ndarray, *, epsilon: float = 1e-6) -> np.ndarray:
    """Build an inverse-distance control graph from group centroid positions."""
    positions = np.asarray(sensor_positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Sensor positions must have shape [n_groups, 3].")
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    graph = np.divide(1.0, distances + epsilon, out=np.zeros_like(distances), where=distances > 0.0)
    return _symmetric_adjacency(graph)


def shuffled_sensitivity(sensitivity: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a cortical-label shuffled mapping control with group weights intact."""
    mapping = np.asarray(sensitivity, dtype=float)
    if mapping.ndim != 2:
        raise ValueError("Sensitivity must be two-dimensional.")
    return mapping[:, rng.permutation(mapping.shape[1])]


def degree_preserving_sc_null(sc: np.ndarray, rng: np.random.Generator, *, nswap: int = 50) -> np.ndarray:
    """Rewire SC topology while preserving binary node degree and edge weights.

    The null uses degree-preserving edge swaps and then redistributes the
    observed positive edge weights across the rewired edges. Dense connectomes
    with no available swaps return an edge-weight permutation and are labelled
    as such by callers.
    """
    adjacency = _symmetric_adjacency(sc)
    upper_i, upper_j = np.triu_indices_from(adjacency, k=1)
    positive = adjacency[upper_i, upper_j] > 0.0
    edges = list(zip(upper_i[positive].tolist(), upper_j[positive].tolist()))
    weights = adjacency[upper_i[positive], upper_j[positive]]
    if len(edges) < 4:
        return _symmetric_adjacency(adjacency[np.ix_(rng.permutation(len(adjacency)), rng.permutation(len(adjacency)))])
    try:
        import networkx as nx

        graph = nx.Graph()
        graph.add_nodes_from(range(adjacency.shape[0]))
        graph.add_edges_from(edges)
        nx.double_edge_swap(
            graph,
            nswap=min(max(1, nswap), max(1, len(edges) // 2)),
            max_tries=max(100, nswap * 20),
            seed=int(rng.integers(0, np.iinfo(np.int32).max)),
        )
        rewired = list(graph.edges())
    except Exception:
        rewired = edges
    null = np.zeros_like(adjacency)
    for (i, j), weight in zip(rewired, rng.permutation(weights)):
        null[i, j] = null[j, i] = float(weight)
    return _symmetric_adjacency(null)


def _laplacian(adjacency: np.ndarray) -> np.ndarray:
    adjacency = _symmetric_adjacency(adjacency)
    return np.diag(adjacency.sum(axis=1)) - adjacency


def _window_metrics(field: np.ndarray, laplacian: np.ndarray, *, tau: float, n_low_modes: int) -> Mapping[str, float]:
    """Calculate graph metrics for a ``[groups, time, coordinates]`` field."""
    eigvals, eigvecs = np.linalg.eigh(laplacian)
    n_low = max(1, min(int(n_low_modes), eigvecs.shape[1] - 1))
    flat = field.reshape(field.shape[0], -1)
    denom = np.sum(flat * flat, axis=0)
    smoothness = np.mean(np.sum(flat * (laplacian @ flat), axis=0) / np.maximum(denom, 1e-12))
    spectral = eigvecs.T @ flat
    energy = spectral * spectral
    low_fraction = float(np.mean(energy[:n_low].sum(axis=0) / np.maximum(energy.sum(axis=0), 1e-12)))
    high_energy = energy[n_low:]
    high_prob = high_energy / np.maximum(high_energy.sum(axis=0, keepdims=True), 1e-12)
    high_entropy = float(np.mean(-np.sum(high_prob * np.log(np.maximum(high_prob, 1e-12)), axis=0)))

    diffusion = expm(-float(tau) * laplacian)
    if field.shape[1] < 2:
        diffusion_r2 = np.nan
        residual_energy = np.nan
    else:
        before = field[:, :-1, :].reshape(field.shape[0], -1)
        observed = field[:, 1:, :].reshape(field.shape[0], -1)
        predicted = diffusion @ before
        residual_energy = float(np.mean((observed - predicted) ** 2))
        ss_tot = float(np.sum((observed - observed.mean()) ** 2))
        diffusion_r2 = float(1.0 - np.sum((observed - predicted) ** 2) / ss_tot) if ss_tot > 1e-12 else np.nan
    return {
        "smoothness": float(smoothness),
        "low_mode_fraction": low_fraction,
        "high_mode_entropy": high_entropy,
        "diffusion_r2": diffusion_r2,
        "residual_energy": residual_energy,
    }


def compute_helmet_sc_chart(
    group_mnps: np.ndarray,
    adjacency: np.ndarray,
    *,
    window: int = 30,
    step: int = 10,
    tau: float = 0.05,
    n_low_modes: int = 2,
) -> pd.DataFrame:
    """Return windowed SC-alignment metrics for ``[time, groups, mde]`` MNPS.

    The output is a parallel structural-alignment chart. It must never replace
    the per-group ``[m, d, e]`` observations or be interpreted as localization.
    """
    values = np.asarray(group_mnps, dtype=float)
    graph = _symmetric_adjacency(adjacency)
    if values.ndim != 3 or values.shape[1] != graph.shape[0] or values.shape[2] < 1:
        raise ValueError("group_mnps must have shape [time, n_groups, n_coordinates].")
    if window < 2 or step < 1:
        raise ValueError("window must be >=2 and step must be >=1.")
    laplacian = _laplacian(graph)
    rows = []
    for start in range(0, values.shape[0] - window + 1, step):
        group_time_coordinate = np.transpose(values[start : start + window], (1, 0, 2))
        metrics = dict(_window_metrics(group_time_coordinate, laplacian, tau=tau, n_low_modes=n_low_modes))
        # Frozen experimental compression: higher low-mode and diffusion fit
        # increase alignment; roughness and residuals decrease it.
        metrics["m_sc"] = metrics["low_mode_fraction"] + metrics["diffusion_r2"]
        metrics["d_sc"] = metrics["smoothness"] + metrics["residual_energy"]
        metrics["e_sc"] = metrics["high_mode_entropy"]
        metrics["start_index"] = start
        metrics["end_index"] = start + window
        rows.append(metrics)
    return pd.DataFrame(rows)
