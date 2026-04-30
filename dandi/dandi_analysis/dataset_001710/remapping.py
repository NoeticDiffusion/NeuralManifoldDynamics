"""Context-sensitive remapping observables for DANDI 001710.

This is the first paper-facing target for 001710.  All observables are kept
descriptive and explicit; no manifold or classifier layers are applied here.

Key observables:
- Within-day left/right arm representation separation (population-code angle).
- Cross-day tuning correlation for matched ROI indices within the same file.
- Day-to-day population-code similarity matrices.
- Block-conditioned similarity matrices.
- Novel-arm-sensitive shifts in tuning or population geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dandi_analysis.dataset_001710.placecode import TuningCurveSet


@dataclass
class RemappingResult:
    """Population similarity between two conditions or days."""

    label_a: str
    label_b: str
    similarity: float           # mean Pearson r across ROIs, or population r
    n_rois: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimilarityMatrix:
    """Pairwise similarity matrix across days or conditions."""

    labels: list[str]
    matrix: np.ndarray          # (n, n) symmetric
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.labels)


# ---------------------------------------------------------------------------
# Within-day arm separation
# ---------------------------------------------------------------------------

def within_day_arm_separation(
    tc_left: TuningCurveSet,
    tc_right: TuningCurveSet,
    *,
    method: str = "population_vector_angle",
) -> RemappingResult:
    """Compute population-level separation between left and right arm tuning.

    Parameters
    ----------
    method:
        ``"population_vector_angle"``: cosine distance between flattened
        population vectors (most interpretable for remapping).
        ``"mean_roi_correlation"``: mean Pearson r across ROIs.
    """
    tc_l = _flatten_nan_safe(tc_left.tuning_curves)
    tc_r = _flatten_nan_safe(tc_right.tuning_curves)

    # Keep only ROIs and bins with finite values in both conditions
    valid_rois = np.isfinite(tc_l).any(axis=1) & np.isfinite(tc_r).any(axis=1)
    tc_l = tc_l[valid_rois, :]
    tc_r = tc_r[valid_rois, :]

    n_rois = int(valid_rois.sum())

    if method == "population_vector_angle":
        flat_l = tc_l.ravel()
        flat_r = tc_r.ravel()
        fin = np.isfinite(flat_l) & np.isfinite(flat_r)
        if fin.sum() < 2:
            sim = np.nan
        else:
            sim = float(np.corrcoef(flat_l[fin], flat_r[fin])[0, 1])
    else:
        cors = []
        for roi in range(n_rois):
            a, b = tc_l[roi], tc_r[roi]
            fin = np.isfinite(a) & np.isfinite(b)
            if fin.sum() >= 3:
                cors.append(float(np.corrcoef(a[fin], b[fin])[0, 1]))
        sim = float(np.nanmean(cors)) if cors else np.nan

    return RemappingResult(
        label_a="left",
        label_b="right",
        similarity=sim,
        n_rois=n_rois,
        metadata={"method": method},
    )


# ---------------------------------------------------------------------------
# Cross-day similarity
# ---------------------------------------------------------------------------

def cross_day_tuning_correlation(
    tc_day_a: TuningCurveSet,
    tc_day_b: TuningCurveSet,
    *,
    label_a: str = "day_a",
    label_b: str = "day_b",
) -> RemappingResult:
    """Mean per-ROI tuning correlation across two days.

    Assumes ROI ordering is consistent within the same NWB session file (which
    is the case for 001710 since sessions share the same field of view).
    """
    n_rois = min(tc_day_a.n_rois, tc_day_b.n_rois)
    tc_a = tc_day_a.tuning_curves[:n_rois, :]
    tc_b = tc_day_b.tuning_curves[:n_rois, :]

    # Interpolate to a common number of bins if necessary
    if tc_a.shape[1] != tc_b.shape[1]:
        n_bins = min(tc_a.shape[1], tc_b.shape[1])
        tc_a = _resample_bins(tc_a, n_bins)
        tc_b = _resample_bins(tc_b, n_bins)

    cors: list[float] = []
    for roi in range(n_rois):
        a, b = tc_a[roi], tc_b[roi]
        fin = np.isfinite(a) & np.isfinite(b)
        if fin.sum() >= 3:
            cors.append(float(np.corrcoef(a[fin], b[fin])[0, 1]))

    sim = float(np.nanmean(cors)) if cors else np.nan
    return RemappingResult(
        label_a=label_a,
        label_b=label_b,
        similarity=sim,
        n_rois=n_rois,
        metadata={"method": "mean_roi_correlation", "n_valid_rois": len(cors)},
    )


# ---------------------------------------------------------------------------
# Day-to-day population similarity matrix
# ---------------------------------------------------------------------------

def build_day_similarity_matrix(
    tc_by_day: dict[str, TuningCurveSet],
) -> SimilarityMatrix:
    """Build an all-pairs population similarity matrix across days.

    Parameters
    ----------
    tc_by_day:
        Mapping from day label (e.g. ``"day0"``) to ``TuningCurveSet``.
    """
    labels = sorted(tc_by_day.keys())
    n = len(labels)
    mat = np.full((n, n), np.nan)

    for i, la in enumerate(labels):
        for j, lb in enumerate(labels):
            if i == j:
                mat[i, j] = 1.0
                continue
            if i > j:
                mat[i, j] = mat[j, i]
                continue
            res = cross_day_tuning_correlation(
                tc_by_day[la],
                tc_by_day[lb],
                label_a=la,
                label_b=lb,
            )
            mat[i, j] = res.similarity

    return SimilarityMatrix(labels=labels, matrix=mat)


# ---------------------------------------------------------------------------
# Block-conditioned similarity
# ---------------------------------------------------------------------------

def block_conditioned_similarity(
    tc_blocks: dict[str, TuningCurveSet],
) -> SimilarityMatrix:
    """Build an all-pairs similarity matrix across block conditions.

    Same implementation as ``build_day_similarity_matrix``, factored out
    for clarity since callers often have separate block-keyed curve dicts.
    """
    return build_day_similarity_matrix(tc_blocks)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_nan_safe(tc: np.ndarray) -> np.ndarray:
    result = np.array(tc, dtype=float)
    return result


def _resample_bins(tc: np.ndarray, n_bins: int) -> np.ndarray:
    """Linearly resample tuning curves from current n_bins to target n_bins."""
    from scipy.interpolate import interp1d  # type: ignore[import]
    n_rois, src_bins = tc.shape
    x_src = np.linspace(0, 1, src_bins)
    x_dst = np.linspace(0, 1, n_bins)
    out = np.full((n_rois, n_bins), np.nan)
    for roi in range(n_rois):
        y = tc[roi]
        fin = np.isfinite(y)
        if fin.sum() >= 2:
            f = interp1d(x_src[fin], y[fin], bounds_error=False, fill_value=np.nan)
            out[roi] = f(x_dst)
    return out
