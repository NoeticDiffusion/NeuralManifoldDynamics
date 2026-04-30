"""Within-session place-code observables for DANDI 001710.

Computes occupancy-normalized tuning curves, split-half reliability, and
left/right arm comparisons.  Uses transparent averages and explicit
position-bin logic.  Does NOT start by trusting embedded ``place_cell_info``
as ground truth; instead provides helpers to audit it against derived curves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dandi_analysis.dataset_001710.behavior import BehaviorTable
from dandi_analysis.dataset_001710.ophys import OphysMatrix
from dandi_analysis.dataset_001710.trials import TrialTable, TrialRow


@dataclass
class TuningCurveSet:
    """Occupancy-normalized tuning curves for all ROIs in one session."""

    n_rois: int
    n_bins: int
    bin_edges: np.ndarray       # shape (n_bins + 1,)
    bin_centers: np.ndarray     # shape (n_bins,)
    occupancy: np.ndarray       # shape (n_bins,) in frames
    mean_activity: np.ndarray   # shape (n_bins, n_rois)
    tuning_curves: np.ndarray   # shape (n_rois, n_bins)  occupancy-normalized
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArmTuning:
    """Tuning curves split by arm label."""

    left: TuningCurveSet | None
    right: TuningCurveSet | None
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_tuning_curves(
    ophys: OphysMatrix,
    behavior: BehaviorTable,
    *,
    n_bins: int = 50,
    pos_min: float | None = None,
    pos_max: float | None = None,
    min_occupancy_frames: int = 2,
) -> TuningCurveSet:
    """Compute occupancy-normalized tuning curves for all ROIs.

    Parameters
    ----------
    ophys:
        Aligned ``(T, N)`` dF matrix.
    behavior:
        Behavior table with a ``position`` channel.
    n_bins:
        Number of spatial bins.
    pos_min, pos_max:
        Position range; inferred from data if ``None``.
    min_occupancy_frames:
        Bins visited fewer than this many frames are set to NaN.
    """
    pos = behavior.channels.get("position")
    if pos is None:
        raise KeyError("'position' channel not found in BehaviorTable")

    pos = np.asarray(pos, dtype=float)
    activity = ophys.data  # (T, N)

    # Trim to common length
    n = min(len(pos), activity.shape[0])
    pos = pos[:n]
    activity = activity[:n, :]

    p_min = float(np.nanmin(pos)) if pos_min is None else float(pos_min)
    p_max = float(np.nanmax(pos)) if pos_max is None else float(pos_max)

    bin_edges = np.linspace(p_min, p_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = np.digitize(pos, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    occupancy = np.zeros(n_bins, dtype=float)
    mean_activity = np.zeros((n_bins, activity.shape[1]), dtype=float)

    for b in range(n_bins):
        mask = bin_idx == b
        occupancy[b] = float(np.sum(mask))
        if occupancy[b] >= min_occupancy_frames:
            with np.errstate(all="ignore"):
                mean_activity[b] = np.nanmean(activity[mask, :], axis=0)
        else:
            mean_activity[b] = np.nan

    # Occupancy-normalize: divide by occupancy (frames) for rate-like curves
    occ_safe = np.where(occupancy >= min_occupancy_frames, occupancy, np.nan)
    tuning_curves = (mean_activity / occ_safe[:, None]).T  # (N, n_bins)

    return TuningCurveSet(
        n_rois=activity.shape[1],
        n_bins=n_bins,
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        occupancy=occupancy,
        mean_activity=mean_activity,
        tuning_curves=tuning_curves,
        metadata={"pos_min": p_min, "pos_max": p_max, "n_frames": n},
    )


def split_half_reliability(
    tuning_curves: np.ndarray,
    ophys: OphysMatrix,
    behavior: BehaviorTable,
    *,
    n_bins: int = 50,
    n_splits: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """Estimate split-half reliability for each ROI as mean Pearson r.

    Randomly splits trials (or frames) into two halves and computes the
    correlation between the two resulting tuning curves.  Repeats ``n_splits``
    times and returns the mean correlation per ROI.

    Returns an array of shape ``(n_rois,)`` with values in [-1, 1].
    """
    rng = np.random.default_rng(seed)
    n_frames = min(ophys.n_frames, behavior.n_frames)
    frame_indices = np.arange(n_frames)

    n_rois = ophys.n_rois
    correlations = np.zeros((n_splits, n_rois), dtype=float)

    pos = behavior.channels.get("position", np.zeros(n_frames))[:n_frames]
    activity = ophys.data[:n_frames, :]

    for split_i in range(n_splits):
        perm = rng.permutation(n_frames)
        half = n_frames // 2
        idx_a, idx_b = perm[:half], perm[half:]

        tc_a = _fast_tuning(pos[idx_a], activity[idx_a, :], n_bins)
        tc_b = _fast_tuning(pos[idx_b], activity[idx_b, :], n_bins)

        for roi in range(n_rois):
            a, b = tc_a[roi], tc_b[roi]
            valid = np.isfinite(a) & np.isfinite(b)
            if valid.sum() < 3:
                correlations[split_i, roi] = np.nan
            else:
                correlations[split_i, roi] = float(np.corrcoef(a[valid], b[valid])[0, 1])

    return np.nanmean(correlations, axis=0)


def arm_tuning(
    ophys: OphysMatrix,
    behavior: BehaviorTable,
    trials: TrialTable,
    *,
    n_bins: int = 50,
) -> ArmTuning:
    """Compute separate tuning curves for left-arm and right-arm trials."""
    left_frames = _frames_for_arm(behavior, trials, "left")
    right_frames = _frames_for_arm(behavior, trials, "right")

    def _subset(frames: np.ndarray) -> tuple[OphysMatrix, BehaviorTable] | None:
        if len(frames) < 10:
            return None
        from copy import copy
        sub_o = copy(ophys)
        sub_o.data = ophys.data[frames, :]
        sub_o.timestamps = ophys.timestamps[frames]
        sub_o.n_frames = len(frames)

        sub_b = copy(behavior)
        sub_b.n_frames = len(frames)
        sub_b.timestamps = behavior.timestamps[frames]
        sub_b.channels = {k: v[frames] for k, v in behavior.channels.items()}
        return sub_o, sub_b

    left_tcs: TuningCurveSet | None = None
    right_tcs: TuningCurveSet | None = None

    res_l = _subset(left_frames)
    if res_l:
        try:
            left_tcs = compute_tuning_curves(res_l[0], res_l[1], n_bins=n_bins)
        except Exception:
            pass

    res_r = _subset(right_frames)
    if res_r:
        try:
            right_tcs = compute_tuning_curves(res_r[0], res_r[1], n_bins=n_bins)
        except Exception:
            pass

    return ArmTuning(left=left_tcs, right=right_tcs)


def reward_zone_summary(
    ophys: OphysMatrix,
    behavior: BehaviorTable,
    *,
    reward_radius_bins: int = 3,
    n_bins: int = 50,
) -> dict[str, Any]:
    """Return mean activity near the reward zone peak vs. non-reward bins."""
    pos = behavior.channels.get("position")
    reward = behavior.channels.get("reward")
    if pos is None or reward is None:
        return {}

    n = min(len(pos), ophys.n_frames)
    pos = pos[:n]
    reward_sig = reward[:n]
    activity = ophys.data[:n, :]

    # Reward zone: bins with the highest average reward signal
    bin_edges = np.linspace(np.nanmin(pos), np.nanmax(pos), n_bins + 1)
    bin_idx = np.clip(np.digitize(pos, bin_edges) - 1, 0, n_bins - 1)

    reward_by_bin = np.zeros(n_bins)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() > 0:
            reward_by_bin[b] = float(np.nanmean(reward_sig[mask]))

    reward_peak_bin = int(np.argmax(reward_by_bin))
    reward_bins = set(
        range(
            max(0, reward_peak_bin - reward_radius_bins),
            min(n_bins, reward_peak_bin + reward_radius_bins + 1),
        )
    )
    reward_mask = np.array([bin_idx[i] in reward_bins for i in range(n)])
    non_reward_mask = ~reward_mask

    result: dict[str, Any] = {
        "reward_peak_bin": reward_peak_bin,
        "n_reward_frames": int(reward_mask.sum()),
        "n_non_reward_frames": int(non_reward_mask.sum()),
    }
    if reward_mask.sum() > 0:
        result["mean_activity_reward"] = float(np.nanmean(activity[reward_mask, :]))
    if non_reward_mask.sum() > 0:
        result["mean_activity_non_reward"] = float(np.nanmean(activity[non_reward_mask, :]))
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fast_tuning(
    pos: np.ndarray,
    activity: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Return (n_rois, n_bins) mean-activity tuning curves (no occupancy norm)."""
    p_min, p_max = float(np.nanmin(pos)), float(np.nanmax(pos))
    if p_max <= p_min:
        return np.full((activity.shape[1], n_bins), np.nan)
    edges = np.linspace(p_min, p_max, n_bins + 1)
    bidx = np.clip(np.digitize(pos, edges) - 1, 0, n_bins - 1)
    curves = np.full((activity.shape[1], n_bins), np.nan)
    for b in range(n_bins):
        mask = bidx == b
        if mask.sum() >= 1:
            with np.errstate(all="ignore"):
                curves[:, b] = np.nanmean(activity[mask, :], axis=0)
    return curves


def _frames_for_arm(
    behavior: BehaviorTable,
    trials: TrialTable,
    arm: str,
) -> np.ndarray:
    """Return frame indices belonging to trials with the given arm label."""
    indices: list[int] = []
    for trial in trials.valid_trials():
        if trial.arm_label == arm:
            indices.extend(range(trial.start_frame, trial.end_frame))
    n = behavior.n_frames
    indices_arr = np.array(indices, dtype=int)
    return indices_arr[indices_arr < n]
