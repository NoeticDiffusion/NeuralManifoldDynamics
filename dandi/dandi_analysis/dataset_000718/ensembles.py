"""Assembly extraction for DANDI 000718 sessions.

Supports three assembly routes (WP3 benchmark):
  1. NMF  — non-negative matrix factorisation (original route).
  2. ICA  — independent component analysis (FastICA via sklearn).
  3. Graph — pairwise correlation matrix + spectral clustering / top eigenvectors.

Each route returns the same Ensemble/EnsembleResult format so downstream
scoring is method-agnostic.

Also provides cross-restart stability metrics so the benchmarking can report
whether a given assembly route produces consistent spatial profiles.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Ensemble:
    """One NMF component extracted from a session."""

    component_idx: int
    session_id: str
    unit_weights: np.ndarray          # shape (n_units,)  — spatial profile
    temporal_profile: np.ndarray      # shape (n_time,)   — activation over time
    unit_ids: tuple[str, ...]
    explained_variance_ratio: float = 0.0
    reconstruction_error: float = float("nan")
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_units(self) -> int:
        return len(self.unit_ids)

    @property
    def top_unit_indices(self, threshold: float = 0.3) -> list[int]:
        """Indices of units with weight above *threshold* * max_weight."""
        max_w = float(self.unit_weights.max()) if self.unit_weights.max() > 0 else 1.0
        return [i for i, w in enumerate(self.unit_weights) if w >= threshold * max_w]


@dataclass
class EnsembleResult:
    """Container for all ensembles extracted from one session/window."""

    session_id: str
    window_label: str
    n_components: int
    ensembles: list[Ensemble]
    reconstruction_error: float
    total_variance_explained: float
    metadata: dict[str, Any] = field(default_factory=dict)


def extract_ensembles(
    data: np.ndarray,
    unit_ids: tuple[str, ...],
    session_id: str,
    window_label: str = "",
    *,
    n_components: int = 8,
    max_iter: int = 500,
    random_state: int = 42,
    min_unit_weight: float = 1e-6,
) -> EnsembleResult:
    """Fit NMF to *data* (T × N, non-negative) and return EnsembleResult.

    The data is first shifted to be non-negative (min-subtracted per unit).
    Each NMF component corresponds to one candidate ensemble.

    Parameters
    ----------
    data:
        Activity matrix, shape (T, N). Can contain negative values (will be
        min-shifted to non-negative before NMF).
    unit_ids:
        Tuple of unit ID strings, length N.
    n_components:
        Number of NMF components (ensembles) to extract.
    """
    from sklearn.decomposition import NMF

    T, N = data.shape
    n_components = min(n_components, N, T)

    # Shift to non-negative
    data_nn = data - data.min(axis=0, keepdims=True)
    # Normalise columns to unit max (keeps relative amplitudes)
    col_max = data_nn.max(axis=0, keepdims=True)
    col_max = np.where(col_max == 0, 1.0, col_max)
    data_norm = data_nn / col_max

    model = NMF(
        n_components=n_components,
        max_iter=max_iter,
        random_state=random_state,
        init="nndsvda",
    )
    W = model.fit_transform(data_norm)  # (T, k)
    H = model.components_              # (k, N)

    # Reconstruction error
    recon = float(model.reconstruction_err_)

    # Variance explained approximation
    total_var = float(np.var(data_norm) * data_norm.size)
    residual = data_norm - W @ H
    explained = max(0.0, 1.0 - float(np.var(residual) * residual.size) / max(total_var, 1e-12))

    ensembles: list[Ensemble] = []
    for k in range(n_components):
        spatial = H[k]          # (N,) — unit weights
        temporal = W[:, k]      # (T,) — activation
        # Per-component variance explained
        recon_k = np.outer(temporal, spatial)
        resid_k = data_norm - recon_k
        ev_k = max(0.0, 1.0 - float(np.var(resid_k) * resid_k.size) / max(total_var, 1e-12))

        ensembles.append(
            Ensemble(
                component_idx=k,
                session_id=session_id,
                unit_weights=spatial,
                temporal_profile=temporal,
                unit_ids=unit_ids,
                explained_variance_ratio=ev_k,
                reconstruction_error=recon,
            )
        )

    return EnsembleResult(
        session_id=session_id,
        window_label=window_label,
        n_components=n_components,
        ensembles=ensembles,
        reconstruction_error=recon,
        total_variance_explained=explained,
    )


def _burst_score(projection: np.ndarray, top_frac: float = 0.10) -> float:
    """Mean activation of the top *top_frac* most active frames.

    This score is sensitive to whether ensemble co-activation is concentrated
    in high-amplitude bursts, unlike the plain mean which is shift-invariant.
    A circular-shift null is therefore non-degenerate for this score.
    """
    k = max(1, int(len(projection) * top_frac))
    return float(np.partition(projection, -k)[-k:].mean())


def offline_ensemble_reactivation(
    offline_data: np.ndarray,
    ensemble: Ensemble,
    *,
    null_n: int = 200,
    rng_seed: int = 42,
    top_frac: float = 0.10,
) -> dict[str, float]:
    """Measure how strongly an ensemble's spatial profile is re-expressed in offline data.

    Uses the ensemble's unit weight vector as a projection axis.  The score is
    the **mean of the top *top_frac* most active frames** (burst score), which
    is sensitive to whether co-activation is concentrated in replay-like events.

    A circular-shift null is applied *to the projection timeseries* before
    scoring.  This preserves the overall distribution of projection values while
    destroying their temporal structure — making the null non-degenerate.

    The previously used plain-mean score was shift-invariant (degenerate null).
    This version fixes that.

    Parameters
    ----------
    offline_data:
        Activity matrix for the offline period, shape (T_offline, N).  Units
        must share the same ordering as the ensemble.
    ensemble:
        Spatial weight vector source (from a different session or period).
    top_frac:
        Fraction of top-activation frames included in the burst score.

    Returns
    -------
    dict with keys: burst_score, null_mean, null_std, z_score, top_frac
    """
    weights = ensemble.unit_weights / (ensemble.unit_weights.max() + 1e-12)
    projection = offline_data @ weights          # (T_offline,)
    observed = _burst_score(projection, top_frac)

    rng = np.random.default_rng(rng_seed)
    T = len(projection)
    min_shift = max(10, T // 20)
    max_shift = max(min_shift + 1, T - min_shift)

    null_vals: list[float] = []
    for _ in range(null_n):
        shift = int(rng.integers(min_shift, max_shift))
        null_vals.append(_burst_score(np.roll(projection, shift), top_frac))

    null_arr = np.array(null_vals)
    null_mean = float(null_arr.mean())
    null_std = float(null_arr.std())
    z = (observed - null_mean) / null_std if null_std > 0 else float("nan")

    return {
        "burst_score": observed,
        "null_mean": null_mean,
        "null_std": null_std,
        "z_score": z,
        "top_frac": top_frac,
    }


def ensemble_overlap(e1: Ensemble, e2: Ensemble) -> float:
    """Cosine similarity between two ensemble spatial profiles.

    Useful for comparing ensembles across sessions when unit ordering is shared.
    """
    w1 = e1.unit_weights
    w2 = e2.unit_weights
    # Align by unit_ids intersection
    ids1 = {uid: i for i, uid in enumerate(e1.unit_ids)}
    ids2 = {uid: i for i, uid in enumerate(e2.unit_ids)}
    shared = sorted(ids1.keys() & ids2.keys())
    if not shared:
        return float("nan")
    v1 = np.array([w1[ids1[u]] for u in shared])
    v2 = np.array([w2[ids2[u]] for u in shared])
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / norm) if norm > 0 else float("nan")


# ---------------------------------------------------------------------------
# WP3: ICA-based assembly extraction
# ---------------------------------------------------------------------------

def extract_ensembles_ica(
    data: np.ndarray,
    unit_ids: tuple[str, ...],
    session_id: str,
    window_label: str = "",
    *,
    n_components: int = 8,
    max_iter: int = 500,
    random_state: int = 42,
) -> EnsembleResult:
    """Extract assemblies using FastICA.

    ICA components are not constrained to be non-negative; we take the
    absolute value of the spatial loading so that large (positive or
    negative) weights both indicate strong ensemble membership.

    Returns the same EnsembleResult format as NMF so downstream code is
    method-agnostic.
    """
    from sklearn.decomposition import FastICA

    T, N = data.shape
    n_components = min(n_components, N, T)

    # z-score (ICA prefers zero-mean unit-variance inputs)
    mu = data.mean(axis=0, keepdims=True)
    sd = data.std(axis=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    data_z = (data - mu) / sd

    try:
        ica = FastICA(
            n_components=n_components,
            max_iter=max_iter,
            random_state=random_state,
            whiten="unit-variance",
        )
        S = ica.fit_transform(data_z)   # (T, k) — sources
        A = ica.mixing_                 # (N, k) — mixing matrix (spatial loadings)
    except Exception as exc:
        warnings.warn(f"ICA failed for {session_id}: {exc}")
        return EnsembleResult(
            session_id=session_id, window_label=window_label,
            n_components=0, ensembles=[], reconstruction_error=float("nan"),
            total_variance_explained=0.0,
        )

    recon = data_z - S @ A.T
    recon_err = float(np.sqrt(np.mean(recon ** 2)))
    total_var = float(np.var(data_z))
    ev = max(0.0, 1.0 - float(np.var(recon)) / max(total_var, 1e-12))

    ensembles: list[Ensemble] = []
    for k in range(n_components):
        spatial = np.abs(A[:, k])       # absolute loading as weight
        temporal = S[:, k]
        ensembles.append(Ensemble(
            component_idx=k,
            session_id=session_id,
            unit_weights=spatial,
            temporal_profile=temporal,
            unit_ids=unit_ids,
            explained_variance_ratio=0.0,
            reconstruction_error=recon_err,
        ))

    return EnsembleResult(
        session_id=session_id,
        window_label=window_label,
        n_components=n_components,
        ensembles=ensembles,
        reconstruction_error=recon_err,
        total_variance_explained=ev,
        metadata={"method": "ica"},
    )


# ---------------------------------------------------------------------------
# WP3: Graph-based assembly extraction (top eigenvectors of correlation matrix)
# ---------------------------------------------------------------------------

def extract_ensembles_graph(
    data: np.ndarray,
    unit_ids: tuple[str, ...],
    session_id: str,
    window_label: str = "",
    *,
    n_components: int = 8,
    threshold_percentile: float = 90.0,
) -> EnsembleResult:
    """Extract assemblies from the top eigenvectors of the co-activation graph.

    Steps:
    1. Compute pairwise Pearson correlation matrix C (N × N).
    2. Threshold C at *threshold_percentile* to get a sparse adjacency matrix.
    3. Compute top *n_components* eigenvectors of the symmetrised, normalised
       adjacency matrix.
    4. Take the absolute value of each eigenvector as an assembly weight.

    This is conceptually similar to PCA on the correlation structure but
    operates on the thresholded graph rather than the raw data, making it
    more robust to dominant noise dimensions.
    """
    T, N = data.shape
    n_components = min(n_components, N - 1, T)

    # z-score per unit
    mu = data.mean(axis=0, keepdims=True)
    sd = data.std(axis=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    data_z = (data - mu) / sd

    # Pairwise Pearson correlation
    C = (data_z.T @ data_z) / max(1, T - 1)    # (N, N)
    np.fill_diagonal(C, 0.0)

    # Threshold
    thresh = float(np.percentile(np.abs(C), threshold_percentile))
    A = np.where(np.abs(C) >= thresh, C, 0.0)

    # Symmetric normalised Laplacian → top eigenvectors
    row_sum = np.abs(A).sum(axis=1)
    row_sum = np.where(row_sum == 0, 1.0, row_sum)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(row_sum))
    L_sym = D_inv_sqrt @ A @ D_inv_sqrt

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
        # Take top n_components by absolute eigenvalue
        idx = np.argsort(np.abs(eigenvalues))[::-1][:n_components]
        vecs = eigenvectors[:, idx]         # (N, n_components)
    except np.linalg.LinAlgError as exc:
        warnings.warn(f"Graph eigdecomp failed for {session_id}: {exc}")
        return EnsembleResult(
            session_id=session_id, window_label=window_label,
            n_components=0, ensembles=[], reconstruction_error=float("nan"),
            total_variance_explained=0.0,
        )

    ensembles: list[Ensemble] = []
    for k in range(vecs.shape[1]):
        spatial = np.abs(vecs[:, k])
        # Temporal: project data onto the eigenvector direction
        temporal = data_z @ vecs[:, k]
        ensembles.append(Ensemble(
            component_idx=k,
            session_id=session_id,
            unit_weights=spatial,
            temporal_profile=temporal,
            unit_ids=unit_ids,
            explained_variance_ratio=0.0,
            reconstruction_error=float("nan"),
        ))

    return EnsembleResult(
        session_id=session_id,
        window_label=window_label,
        n_components=len(ensembles),
        ensembles=ensembles,
        reconstruction_error=float("nan"),
        total_variance_explained=float("nan"),
        metadata={"method": "graph", "threshold_percentile": threshold_percentile},
    )


# ---------------------------------------------------------------------------
# WP3: Stability metric across restarts / folds
# ---------------------------------------------------------------------------

def assembly_stability(
    data: np.ndarray,
    unit_ids: tuple[str, ...],
    *,
    method: str = "nmf",
    n_components: int = 8,
    n_restarts: int = 5,
) -> dict[str, float]:
    """Measure how stable the assembly spatial profiles are across restarts.

    For NMF/ICA: run *n_restarts* times with different random seeds, compute
    pairwise cosine similarity between corresponding component pairs (after
    Hungarian matching), report mean and min similarity.

    For graph: run with *n_restarts* random subsamples (90% of frames).

    Returns
    -------
    dict with: mean_stability, min_stability, method
    """
    from scipy.optimize import linear_sum_assignment

    extract = {
        "nmf": lambda rs: extract_ensembles(data, unit_ids, "stab", n_components=n_components, random_state=rs),
        "ica": lambda rs: extract_ensembles_ica(data, unit_ids, "stab", n_components=n_components, random_state=rs),
        "graph": lambda rs: extract_ensembles_graph(data, unit_ids, "stab", n_components=n_components),
    }[method]

    T = data.shape[0]
    rng = np.random.default_rng(42)
    results: list[list[np.ndarray]] = []

    for r in range(n_restarts):
        if method == "graph":
            # Random 90% subsample of time frames
            idx = rng.choice(T, size=int(0.9 * T), replace=False)
            sub = data[np.sort(idx)]
            res = extract_ensembles_graph(sub, unit_ids, "stab", n_components=n_components)
        else:
            res = extract(r * 13 + 7)
        results.append([e.unit_weights for e in res.ensembles])

    if not results or len(results[0]) == 0:
        return {"mean_stability": float("nan"), "min_stability": float("nan"), "method": method}

    # Compare each pair of runs: Hungarian matching, then cosine similarity
    sim_vals: list[float] = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            wa = results[i]
            wb = results[j]
            k = min(len(wa), len(wb))
            # Cosine similarity matrix (k × k)
            sim_mat = np.zeros((k, k))
            for a_idx in range(k):
                for b_idx in range(k):
                    na = np.linalg.norm(wa[a_idx])
                    nb = np.linalg.norm(wb[b_idx])
                    if na > 0 and nb > 0:
                        sim_mat[a_idx, b_idx] = float(np.dot(wa[a_idx], wb[b_idx]) / (na * nb))
            row_ind, col_ind = linear_sum_assignment(-sim_mat)
            for r_i, c_i in zip(row_ind, col_ind):
                sim_vals.append(sim_mat[r_i, c_i])

    return {
        "mean_stability": round(float(np.mean(sim_vals)), 3),
        "min_stability": round(float(np.min(sim_vals)), 3),
        "method": method,
        "n_restarts": n_restarts,
    }


# ---------------------------------------------------------------------------
# WP3: Multi-method benchmark runner
# ---------------------------------------------------------------------------

def benchmark_assembly_methods(
    data: np.ndarray,
    unit_ids: tuple[str, ...],
    session_id: str,
    *,
    n_components: int = 8,
    n_stability_restarts: int = 5,
) -> dict[str, dict]:
    """Run NMF, ICA, and graph extraction and return a comparative summary.

    Also runs stability analysis for each method.

    Returns
    -------
    dict keyed by method name, each containing:
      - ``ensembles``: EnsembleResult
      - ``stability``: stability dict
    """
    methods: dict[str, Any] = {}

    # NMF
    nmf_res = extract_ensembles(data, unit_ids, session_id, n_components=n_components, random_state=42)
    nmf_stab = assembly_stability(data, unit_ids, method="nmf", n_components=n_components, n_restarts=n_stability_restarts)
    methods["nmf"] = {"ensembles": nmf_res, "stability": nmf_stab}

    # ICA
    ica_res = extract_ensembles_ica(data, unit_ids, session_id, n_components=n_components, random_state=42)
    ica_stab = assembly_stability(data, unit_ids, method="ica", n_components=n_components, n_restarts=n_stability_restarts)
    methods["ica"] = {"ensembles": ica_res, "stability": ica_stab}

    # Graph
    graph_res = extract_ensembles_graph(data, unit_ids, session_id, n_components=n_components)
    graph_stab = assembly_stability(data, unit_ids, method="graph", n_components=n_components, n_restarts=n_stability_restarts)
    methods["graph"] = {"ensembles": graph_res, "stability": graph_stab}

    return methods
