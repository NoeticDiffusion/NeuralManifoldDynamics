"""Graph-theoretic metrics shared between EEG and fMRI pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import networkx as nx
import numpy as np
from networkx.algorithms import community, efficiency_measures


@dataclass
class ThresholdConfig:
    method: str
    density: float


def compute_graph_metrics(
    fc_matrix: np.ndarray,
    config: Mapping[str, object] | None,
) -> Dict[str, float]:
    """Return graph-level metrics from a functional connectivity matrix."""

    if fc_matrix.ndim != 2 or fc_matrix.shape[0] != fc_matrix.shape[1]:
        raise ValueError("fc_matrix must be square 2-D")

    cfg = config or {}
    metrics_cfg = cfg.get("metrics", {}) if isinstance(cfg, Mapping) else {}
    threshold_cfg = _parse_threshold(cfg.get("thresholding"))

    adj = _threshold_matrix(fc_matrix, threshold_cfg)
    G = nx.from_numpy_array(adj, create_using=nx.Graph)
    features: Dict[str, float] = {}

    if metrics_cfg.get("global_efficiency", True):
        features["graph_global_efficiency"] = float(efficiency_measures.global_efficiency(G))

    if metrics_cfg.get("avg_path_length", True):
        try:
            features["graph_avg_path_length"] = float(nx.average_shortest_path_length(G))
        except nx.NetworkXError:
            features["graph_avg_path_length"] = float("nan")

    communities = None
    if metrics_cfg.get("modularity", True) or metrics_cfg.get("participation_coeff", True):
        communities = list(community.greedy_modularity_communities(G, weight="weight"))
        if metrics_cfg.get("modularity", True):
            features["graph_modularity"] = float(community.modularity(G, communities, weight="weight"))

    if metrics_cfg.get("participation_coeff", True) and communities:
        pc = _participation_coeff(adj, communities)
        features["graph_participation_coeff_mean"] = float(np.nanmean(pc))
        features["graph_participation_coeff_std"] = float(np.nanstd(pc))

    if metrics_cfg.get("hubness_degree", False):
        degrees = np.array([deg for _, deg in G.degree(weight="weight")], dtype=float)
        features["graph_degree_mean"] = float(np.nanmean(degrees))
        features["graph_degree_std"] = float(np.nanstd(degrees))

    return features


def _parse_threshold(cfg) -> ThresholdConfig:
    if isinstance(cfg, Mapping):
        method = str(cfg.get("method", "proportional")).lower()
        density = float(cfg.get("density", 0.15))
    else:
        method = "proportional"
        density = 0.15
    if method != "proportional":
        raise ValueError("Only proportional thresholding is supported currently")
    if not (0 < density <= 1):
        raise ValueError("density must be in (0, 1]")
    return ThresholdConfig(method=method, density=density)


def _threshold_matrix(fc: np.ndarray, cfg: ThresholdConfig) -> np.ndarray:
    adj = np.abs(fc.copy())
    np.fill_diagonal(adj, 0.0)
    if cfg.method == "proportional":
        triu = adj[np.triu_indices_from(adj, k=1)]
        if triu.size == 0:
            return adj
        cutoff_index = max(int((1.0 - cfg.density) * triu.size), 0)
        threshold = np.partition(triu, cutoff_index)[cutoff_index]
        adj[adj < threshold] = 0.0
    return adj


def _participation_coeff(adj: np.ndarray, communities: Sequence[Sequence[int]]) -> np.ndarray:
    n = adj.shape[0]
    strengths = adj.sum(axis=1)
    if np.allclose(strengths, 0):
        return np.full(n, np.nan)

    pc = np.zeros(n, dtype=float)
    comm_lookup = np.zeros(n, dtype=int)
    for idx, nodes in enumerate(communities):
        comm_lookup[list(nodes)] = idx

    for node in range(n):
        k = strengths[node]
        if k <= 0:
            pc[node] = np.nan
            continue
        sum_sq = 0.0
        for idx in range(len(communities)):
            mask = comm_lookup == idx
            k_is = adj[node, mask].sum()
            sum_sq += (k_is / k) ** 2
        pc[node] = 1.0 - sum_sq
    return pc

