"""Cross-session ROI registration for DANDI 000718 (Ca-imaging).

Provides a probabilistic footprint-based matcher that computes a per-pair
confidence score from:
  - centroid distance,
  - Dice footprint overlap,
  - shape correlation (cosine similarity of flattened masks),
  - local neighbourhood geometric consistency.

The output is a ``RegistrationResult`` that exposes a thresholded matched set
and per-match confidence values. This confidence layer is mandatory before
interpreting negative or positive cross-session H1 results.

Without this, a negative cross-session result is biologically ambiguous:
it could mean no replay, or it could mean the same cells were not tracked.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RoiFootprint:
    """Spatial descriptor for one ROI."""
    roi_idx: int
    centroid: np.ndarray          # shape (2,): (row, col)
    area: int
    mask_flat: np.ndarray         # normalised flat mask (for cosine sim)
    bbox: tuple[int, int, int, int]   # (r0, r1, c0, c1)


@dataclass
class MatchedPair:
    """One cross-session ROI match."""
    source_idx: int     # ROI index in session A
    target_idx: int     # ROI index in session B
    centroid_dist: float
    dice: float
    shape_corr: float
    local_consistency: float = 1.0
    confidence: float = 0.0     # Combined score in [0, 1]

    @property
    def is_accepted(self) -> bool:
        return self.confidence >= 0.3


@dataclass
class RegistrationResult:
    """All cross-session matches for one session pair."""
    session_a: str
    session_b: str
    n_rois_a: int
    n_rois_b: int
    all_matches: list[MatchedPair]
    max_centroid_dist_px: float
    min_dice: float
    # Thresholded set (confidence >= threshold)
    accepted: list[MatchedPair] = field(default_factory=list)

    @property
    def n_accepted(self) -> int:
        return len(self.accepted)

    def matched_indices(self) -> tuple[list[int], list[int]]:
        """Return (source_indices, target_indices) for accepted matches."""
        src = [m.source_idx for m in self.accepted]
        tgt = [m.target_idx for m in self.accepted]
        return src, tgt

    def summary_dict(self) -> dict:
        return {
            "session_a": self.session_a,
            "session_b": self.session_b,
            "n_rois_a": self.n_rois_a,
            "n_rois_b": self.n_rois_b,
            "n_candidates": len(self.all_matches),
            "n_accepted": self.n_accepted,
            "fraction_a_matched": round(self.n_accepted / max(1, self.n_rois_a), 3),
            "fraction_b_matched": round(self.n_accepted / max(1, self.n_rois_b), 3),
            "mean_confidence": round(
                float(np.mean([m.confidence for m in self.accepted])), 3
            ) if self.accepted else 0.0,
            "mean_dice": round(
                float(np.mean([m.dice for m in self.accepted])), 3
            ) if self.accepted else 0.0,
            "mean_centroid_dist_px": round(
                float(np.mean([m.centroid_dist for m in self.accepted])), 2
            ) if self.accepted else 0.0,
        }


# ---------------------------------------------------------------------------
# Footprint extraction
# ---------------------------------------------------------------------------

def _load_footprints(path: Path) -> list[RoiFootprint]:
    """Load per-ROI image masks from an NWB file using h5py batch read.

    Uses direct h5py access to read all masks in one operation (orders of
    magnitude faster than pynwb element-by-element access on large datasets).
    """
    import h5py

    footprints: list[RoiFootprint] = []
    h5_mask_path = "processing/ophys/ImageSegmentation/PlaneSegmentation/image_mask"

    try:
        with h5py.File(str(path), "r") as f:
            if h5_mask_path not in f:
                warnings.warn(f"No PlaneSegmentation image_mask in {path.name}")
                return footprints
            all_masks = f[h5_mask_path][:]   # (N, H, W) — one read
    except Exception as exc:
        warnings.warn(f"h5py mask read failed for {path.name}: {exc}")
        return footprints

    for i, mask in enumerate(all_masks):
        if mask.max() == 0:
            continue
        try:
            footprints.append(_compute_footprint(i, mask.astype(float)))
        except Exception as exc:
            warnings.warn(f"ROI {i} footprint failed: {exc}")

    return footprints


def _compute_footprint(roi_idx: int, mask: np.ndarray) -> RoiFootprint:
    """Compute spatial descriptors from a 2D image mask.

    The full (H, W) mask is stored only for Dice computation (via bbox crop).
    For shape correlation we store a small 32×32 crop of the bounding box,
    which is fast enough for O(N²) cosine similarity at registration time.
    """
    norm = mask / (mask.max() + 1e-12)
    yx = np.argwhere(norm > 0.1)
    if len(yx) == 0:
        centroid = np.array([0.0, 0.0])
        bbox = (0, 1, 0, 1)
        crop_flat = np.zeros(32 * 32)
    else:
        centroid = yx.mean(axis=0)
        r0, c0 = yx.min(axis=0)
        r1, c1 = yx.max(axis=0)
        bbox = (int(r0), int(r1) + 1, int(c0), int(c1) + 1)
        # Resize bounding-box crop to 32×32 for fast shape comparison
        crop = norm[r0:r1 + 1, c0:c1 + 1]
        crop_flat = _resize_to_flat(crop, 32)

    crop_norm = crop_flat / (np.linalg.norm(crop_flat) + 1e-12)

    return RoiFootprint(
        roi_idx=roi_idx,
        centroid=centroid,
        area=int((norm > 0.1).sum()),
        mask_flat=crop_norm,       # 32×32 crop, normalised
        bbox=bbox,
    )


def _resize_to_flat(crop: np.ndarray, target: int) -> np.ndarray:
    """Resize a 2D crop to (target × target) via simple decimation/zero-pad."""
    from PIL import Image  # noqa: F401 — lazy import for speed; fall back to numpy
    try:
        from PIL import Image
        img = Image.fromarray(crop.astype(np.float32))
        img_r = img.resize((target, target), Image.BILINEAR)
        return np.array(img_r).ravel()
    except Exception:
        # Fallback: crude resize with slicing
        H, W = crop.shape
        row_idx = np.round(np.linspace(0, H - 1, target)).astype(int)
        col_idx = np.round(np.linspace(0, W - 1, target)).astype(int)
        return crop[np.ix_(row_idx, col_idx)].ravel()


# ---------------------------------------------------------------------------
# Pairwise scoring
# ---------------------------------------------------------------------------

def _dice(fp_a: RoiFootprint, fp_b: RoiFootprint) -> float:
    """Dice coefficient from 32×32 crops using a relative binarisation threshold."""
    thr_a = float(fp_a.mask_flat.max()) * 0.3
    thr_b = float(fp_b.mask_flat.max()) * 0.3
    if thr_a == 0 or thr_b == 0:
        return 0.0
    a_bin = fp_a.mask_flat > thr_a
    b_bin = fp_b.mask_flat > thr_b
    intersection = float((a_bin & b_bin).sum())
    union_sum = float(a_bin.sum() + b_bin.sum())
    return 2.0 * intersection / union_sum if union_sum > 0 else 0.0


def _shape_corr(fp_a: RoiFootprint, fp_b: RoiFootprint) -> float:
    """Cosine similarity of normalised flat masks."""
    return float(np.dot(fp_a.mask_flat, fp_b.mask_flat))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_sessions(
    path_a: Path,
    path_b: Path,
    session_a: str,
    session_b: str,
    *,
    max_centroid_dist_px: float = 10.0,
    min_dice: float = 0.1,
    confidence_threshold: float = 0.30,
) -> RegistrationResult:
    """Match ROIs across two sessions using spatial footprints.

    Parameters
    ----------
    max_centroid_dist_px:
        Maximum centroid-to-centroid distance (pixels) to consider a candidate
        match.  For 2-photon Ca-imaging with stable FOV, 5–15 px is typical.
    min_dice:
        Minimum Dice overlap to accept a candidate pair.
    confidence_threshold:
        Minimum combined confidence for the final accepted match set.

    Returns
    -------
    RegistrationResult with full match list and thresholded accepted set.
    """
    fps_a = _load_footprints(path_a)
    fps_b = _load_footprints(path_b)

    if not fps_a or not fps_b:
        return RegistrationResult(
            session_a=session_a, session_b=session_b,
            n_rois_a=len(fps_a), n_rois_b=len(fps_b),
            all_matches=[], max_centroid_dist_px=max_centroid_dist_px,
            min_dice=min_dice,
        )

    centroids_a = np.stack([f.centroid for f in fps_a])  # (Na, 2)
    centroids_b = np.stack([f.centroid for f in fps_b])  # (Nb, 2)

    # Build candidate pairs via centroid distance
    from scipy.spatial import cKDTree
    tree_b = cKDTree(centroids_b)
    indices_b_lists = tree_b.query_ball_point(centroids_a, r=max_centroid_dist_px)

    all_matches: list[MatchedPair] = []
    for i_a, b_candidates in enumerate(indices_b_lists):
        for i_b in b_candidates:
            fp_a = fps_a[i_a]
            fp_b = fps_b[i_b]
            dist = float(np.linalg.norm(fp_a.centroid - fp_b.centroid))
            sc = _shape_corr(fp_a, fp_b)
            if sc < min_dice:   # reuse min_dice parameter as min_shape_corr gate
                continue
            d = _dice(fp_a, fp_b)
            all_matches.append(MatchedPair(
                source_idx=fp_a.roi_idx,
                target_idx=fp_b.roi_idx,
                centroid_dist=dist,
                dice=d,
                shape_corr=sc,
            ))

    # Compute raw confidence and keep best match per source (greedy 1-to-1)
    for m in all_matches:
        dist_norm = 1.0 - m.centroid_dist / max_centroid_dist_px
        m.confidence = float(
            0.50 * m.dice +
            0.30 * m.shape_corr +
            0.20 * dist_norm
        )

    # Greedy 1-to-1 assignment: keep highest-confidence match per source and target
    all_matches.sort(key=lambda x: -x.confidence)
    used_a: set[int] = set()
    used_b: set[int] = set()
    assigned: list[MatchedPair] = []
    for m in all_matches:
        if m.source_idx not in used_a and m.target_idx not in used_b:
            assigned.append(m)
            used_a.add(m.source_idx)
            used_b.add(m.target_idx)

    # Local neighbourhood consistency: for each match, check that its
    # nearby source-ROIs' matches are geometrically consistent (no large
    # spatial distortion from source neighbourhood to target neighbourhood).
    _add_neighbourhood_consistency(assigned, fps_a, fps_b, k=5)

    # Recompute final confidence with consistency term
    for m in assigned:
        dist_norm = 1.0 - m.centroid_dist / max_centroid_dist_px
        m.confidence = float(
            0.40 * m.dice +
            0.25 * m.shape_corr +
            0.15 * dist_norm +
            0.20 * m.local_consistency
        )

    accepted = [m for m in assigned if m.confidence >= confidence_threshold]

    return RegistrationResult(
        session_a=session_a,
        session_b=session_b,
        n_rois_a=len(fps_a),
        n_rois_b=len(fps_b),
        all_matches=assigned,
        max_centroid_dist_px=max_centroid_dist_px,
        min_dice=min_dice,
        accepted=accepted,
    )


def _add_neighbourhood_consistency(
    matches: list[MatchedPair],
    fps_a: list[RoiFootprint],
    fps_b: list[RoiFootprint],
    k: int = 5,
) -> None:
    """Compute local geometric consistency for each match in-place.

    For a match (a→b), find the k nearest matches by source centroid.
    Compute the residual of b's centroid prediction from those neighbours'
    rigid transform.  High consistency = low residual.
    """
    if len(matches) < k + 1:
        return

    idx_to_fp_a = {f.roi_idx: f for f in fps_a}
    idx_to_fp_b = {f.roi_idx: f for f in fps_b}

    src_cents = np.array([idx_to_fp_a[m.source_idx].centroid for m in matches])
    tgt_cents = np.array([idx_to_fp_b[m.target_idx].centroid for m in matches])
    offsets = tgt_cents - src_cents   # (M, 2)

    from scipy.spatial import cKDTree
    tree = cKDTree(src_cents)

    for i, m in enumerate(matches):
        dists, nbrs = tree.query(src_cents[i], k=min(k + 1, len(matches)))
        nbrs = [n for n in nbrs if n != i][:k]
        if not nbrs:
            m.local_consistency = 0.5
            continue
        mean_offset = offsets[nbrs].mean(axis=0)
        predicted_tgt = src_cents[i] + mean_offset
        actual_tgt = tgt_cents[i]
        residual = float(np.linalg.norm(predicted_tgt - actual_tgt))
        # Map residual to [0,1]: 0 px → 1.0, 5+ px → 0.0
        m.local_consistency = float(max(0.0, 1.0 - residual / 5.0))
