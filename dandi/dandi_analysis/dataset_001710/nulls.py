"""Cheap null baselines for 001710 context-sensitivity analyses.

All null generators preserve array shape and are deterministic under a fixed
seed.  They are designed to test whether observed context dependence exceeds
simple label or temporal structure artifacts.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def circular_time_shift(
    activity: np.ndarray,
    *,
    shift: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Roll the time axis of ``activity`` by a random (or specified) shift.

    Parameters
    ----------
    activity:
        Array of shape ``(T, N)``.
    shift:
        Number of frames to shift.  If ``None``, drawn uniformly from
        ``[T // 10, T - T // 10]`` to avoid near-identity shifts.

    Returns a copy of ``activity`` with time circularly shifted.
    """
    activity = np.asarray(activity)
    T = activity.shape[0]
    if shift is None:
        rng = np.random.default_rng(seed)
        lo = max(1, T // 10)
        hi = max(lo + 1, T - T // 10)
        shift = int(rng.integers(lo, hi))
    return np.roll(activity, shift, axis=0)


def position_bin_shuffle(
    activity: np.ndarray,
    position: np.ndarray,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return activity and position with frames shuffled within each position bin.

    This preserves the marginal distributions of both activity and position
    while breaking the spatial structure of individual tuning curves.

    Returns ``(shuffled_activity, shuffled_position)`` of the same shape.
    """
    rng = np.random.default_rng(seed)
    activity = np.asarray(activity, dtype=float)
    position = np.asarray(position, dtype=float)

    n = min(len(position), activity.shape[0])
    position = position[:n]
    activity = activity[:n]

    p_min, p_max = float(np.nanmin(position)), float(np.nanmax(position))
    n_bins = 20
    edges = np.linspace(p_min, p_max, n_bins + 1)
    bidx = np.clip(np.digitize(position, edges) - 1, 0, n_bins - 1)

    out_activity = activity.copy()
    out_position = position.copy()

    for b in range(n_bins):
        mask = np.where(bidx == b)[0]
        if len(mask) < 2:
            continue
        perm = rng.permutation(len(mask))
        out_activity[mask] = activity[mask[perm]]
        out_position[mask] = position[mask[perm]]

    return out_activity, out_position


def trial_label_shuffle(
    labels: list[str] | np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Return a permuted copy of trial labels.

    Used to test whether arm or context sensitivity depends on the actual
    label assignment or could arise from any balanced partition.
    """
    rng = np.random.default_rng(seed)
    labels_arr = np.array(labels)
    return rng.permutation(labels_arr)


def arm_label_shuffle(
    trial_arm_labels: list[str] | np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Return a permuted copy of arm labels (alias of ``trial_label_shuffle``).

    Kept as a separate entry point so callers can be explicit about which
    label type is being shuffled.
    """
    return trial_label_shuffle(trial_arm_labels, seed=seed)


def generate_null_distribution(
    func: Any,
    n_nulls: int,
    *,
    base_seed: int = 0,
    **kwargs: Any,
) -> np.ndarray:
    """Run ``func(seed=k, **kwargs)`` for ``k`` in ``range(n_nulls)`` and return results.

    ``func`` must accept a ``seed`` keyword argument and return a scalar float.
    Useful for building empirical null distributions for any of the above
    shuffles combined with a downstream metric.
    """
    results = np.empty(n_nulls, dtype=float)
    for i in range(n_nulls):
        results[i] = float(func(seed=base_seed + i, **kwargs))
    return results


def permutation_cohort_null(
    obs_diff: float,
    group_a_vals: list[float] | np.ndarray,
    group_b_vals: list[float] | np.ndarray,
    *,
    n_perms: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Permutation test for the difference in means between two groups.

    Pools all values and resamples without replacement into groups of the
    original sizes, building an empirical null distribution of
    ``mean(A) - mean(B)``.

    Parameters
    ----------
    obs_diff:
        Observed ``mean(A) - mean(B)``.
    group_a_vals, group_b_vals:
        Per-observation values (e.g. per-day off-diagonal similarities).
        Non-finite values are silently dropped before testing.
    n_perms:
        Number of permutations.
    seed:
        Random seed for reproducibility.

    Returns a dict with keys:
        ``null_mean``, ``null_std``, ``z``, ``p_empirical``,
        ``n_perms``, ``effect_direction``, ``null_distribution``, ``note``.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(group_a_vals, dtype=float)
    b = np.asarray(group_b_vals, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    na, nb = len(a), len(b)
    pooled = np.concatenate([a, b])

    null_diffs = np.empty(n_perms, dtype=float)
    for i in range(n_perms):
        perm = rng.permutation(len(pooled))
        perm_a = pooled[perm[:na]]
        perm_b = pooled[perm[na : na + nb]]
        null_diffs[i] = float(np.mean(perm_a)) - float(np.mean(perm_b))

    null_mean = float(np.mean(null_diffs))
    null_std = float(np.std(null_diffs))

    if null_std > 0:
        z = (obs_diff - null_mean) / null_std
    else:
        z = float("nan")

    # One-sided p: direction determined by sign of observed difference
    if obs_diff < 0:
        p_empirical = float(np.mean(null_diffs <= obs_diff))
        effect_direction = "negative"
    else:
        p_empirical = float(np.mean(null_diffs >= obs_diff))
        effect_direction = "positive"

    note = ""
    if na < 3 or nb < 3:
        note = (
            f"Small sample (na={na}, nb={nb}); permutation test has limited power."
        )

    return {
        "null_mean": round(null_mean, 6),
        "null_std": round(null_std, 6),
        "z": round(z, 4) if np.isfinite(z) else float("nan"),
        "p_empirical": round(p_empirical, 4),
        "n_perms": n_perms,
        "effect_direction": effect_direction,
        "null_distribution": null_diffs.tolist(),
        "note": note,
    }
