"""Robustness and null-test analyses for DANDI 001710.

Provides analysis helpers for broadening 001710 beyond a one-subject-per-label
comparison:

1. ``arm_label_audit``       — verify arm labels against NWB annotation blob.
2. ``compare_sparseko_channels`` — check whether the SparseKO effect is
   channel-independent (ch0 vs ch1 comparison).
3. ``cohort_null_tests``     — legacy single-subject null test for cross-day
   similarity differences.
4. ``group_null_tests``      — subject-level permutation tests between genotype
   groups (e.g. SparseKO vs Cre / Ctrl).
5. ``day_lag_similarity``    — day-lag profile of cross-day similarity (lags 1–5).
6. ``aggregate_group_day_lag`` — aggregate subject-level lag profiles by group.
7. ``plot_day_lag_curves``   — visualize lag profiles per subject or group.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from dandi_analysis.dataset_001710.io import (
    list_session_channels,
    read_trial_annotation_blob,
)
from dandi_analysis.dataset_001710.behavior import load_behavior_table
from dandi_analysis.dataset_001710.ophys import load_ophys_matrix
from dandi_analysis.dataset_001710.trials import build_trial_table
from dandi_analysis.dataset_001710.placecode import (
    compute_tuning_curves,
    split_half_reliability,
    arm_tuning,
)
from dandi_analysis.dataset_001710.remapping import (
    SimilarityMatrix,
    build_day_similarity_matrix,
    within_day_arm_separation,
)
from dandi_analysis.dataset_001710.nulls import permutation_cohort_null


# ---------------------------------------------------------------------------
# Analysis 1 — Arm label audit
# ---------------------------------------------------------------------------

def arm_label_audit(path: Path, day: int) -> dict[str, Any]:
    """Verify reconstructed arm labels against NWB annotation blob.

    Reads the annotation blob's ``vr_trial_info`` (if present) and compares
    per-trial arm labels with those derived by ``build_trial_table``.

    Returns a dict with keys:
        path, day, vr_trial_info_found, n_trials, n_valid, n_left, n_right,
        n_mismatch, mismatch_rate, flagged, note.
    """
    blob = read_trial_annotation_blob(path)
    trials = build_trial_table(path, day=day)
    valid = trials.valid_trials()

    result: dict[str, Any] = {
        "path": str(path),
        "day": day,
        "vr_trial_info_found": False,
        "n_trials": len(trials),
        "n_valid": len(valid),
        "n_left": len(trials.by_arm("left")),
        "n_right": len(trials.by_arm("right")),
        "n_mismatch": 0,
        "mismatch_rate": float("nan"),
        "flagged": False,
        "note": "",
    }

    # Try to locate vr_trial_info in the blob
    vr_info = None
    if blob:
        for key in ("vr_trial_info", "trial_info", "vr_trials"):
            if key in blob:
                vr_info = blob[key]
                result["vr_trial_info_found"] = True
                break

    if vr_info is None:
        result["note"] = (
            "vr_trial_info not found in annotation blob; cannot cross-validate arm labels."
        )
        return result

    # Parse annotation arm labels defensively
    annotation_arms: list[str] = []
    try:
        if isinstance(vr_info, list):
            for entry in vr_info:
                if isinstance(entry, dict):
                    raw = str(entry.get("arm", entry.get("side", "unknown"))).lower()
                else:
                    raw = str(entry).lower()
                if "left" in raw or raw == "0":
                    annotation_arms.append("left")
                elif "right" in raw or raw == "1":
                    annotation_arms.append("right")
                else:
                    annotation_arms.append("unknown")
        elif isinstance(vr_info, dict):
            arm_list = vr_info.get("arm", vr_info.get("side", []))
            for a in arm_list:
                s = str(a).lower()
                if "left" in s or s == "0":
                    annotation_arms.append("left")
                elif "right" in s or s == "1":
                    annotation_arms.append("right")
                else:
                    annotation_arms.append("unknown")
    except Exception as exc:
        result["note"] = f"Failed to parse vr_trial_info: {exc}"
        return result

    if not annotation_arms:
        result["note"] = "vr_trial_info found but contained no parseable arm labels."
        return result

    # Compare reconstructed vs annotation (up to min length)
    n_compare = min(len(valid), len(annotation_arms))
    n_mismatch = 0
    for i in range(n_compare):
        rec_arm = valid[i].arm_label
        ann_arm = annotation_arms[i]
        if ann_arm != "unknown" and rec_arm != "unknown" and rec_arm != ann_arm:
            n_mismatch += 1

    mismatch_rate = n_mismatch / n_compare if n_compare > 0 else float("nan")
    result["n_mismatch"] = n_mismatch
    result["mismatch_rate"] = round(mismatch_rate, 4) if np.isfinite(mismatch_rate) else float("nan")
    result["flagged"] = bool(mismatch_rate > 0.1) if np.isfinite(mismatch_rate) else False
    result["note"] = f"Compared {n_compare} trials against annotation blob."
    return result


# ---------------------------------------------------------------------------
# Analysis 2 — SparseKO channel comparison
# ---------------------------------------------------------------------------

def compare_sparseko_channels(paths_by_day: dict[int, Path]) -> dict[str, Any]:
    """Compare ch0 vs ch1 metrics for SparseKO-7 across days.

    For each day builds tuning curves, split-half reliability, and
    within-day arm separation separately for ch0 and ch1.  Then builds
    cross-day similarity matrices per channel.

    Parameters
    ----------
    paths_by_day:
        Mapping from day index to NWB file path for SparseKO-7 sessions.

    Returns a dict with ``per_day_metrics``, ``ch0``, ``ch1``, and
    ``ch0_minus_ch1`` summaries.
    """
    result: dict[str, Any] = {
        "days_processed": [],
        "per_day_metrics": [],
        "ch0": {},
        "ch1": {},
        "ch0_minus_ch1": {},
        "note": "",
    }

    tc_by_day_per_channel: dict[str, dict[str, Any]] = {"0": {}, "1": {}}

    for day, path in sorted(paths_by_day.items()):
        channels = list_session_channels(path)
        if not channels:
            continue

        day_label = f"day{day}"
        day_entry: dict[str, Any] = {"day": day}

        for ch in channels:
            ch_id = ch.channel_id
            if ch_id not in ("0", "1"):
                continue

            beh = load_behavior_table(path, channel=ch)
            if beh is None:
                day_entry[f"ch{ch_id}_error"] = "behavior_not_loaded"
                continue

            ophys = load_ophys_matrix(path, channel=ch)
            if ophys is None:
                day_entry[f"ch{ch_id}_error"] = "ophys_not_loaded"
                continue

            trials = build_trial_table(path, day=day)

            day_entry[f"ch{ch_id}_n_rois"] = ophys.n_rois
            day_entry[f"ch{ch_id}_n_frames"] = beh.n_frames
            day_entry[f"ch{ch_id}_n_valid_trials"] = len(trials.valid_trials())

            # Tuning + split-half
            try:
                tc = compute_tuning_curves(ophys, beh, n_bins=50)
                rel = split_half_reliability(tc.tuning_curves, ophys, beh, n_splits=5)
                finite_rel = rel[np.isfinite(rel)]
                sph_med = float(np.median(finite_rel)) if len(finite_rel) else float("nan")
                day_entry[f"ch{ch_id}_split_half_median"] = round(sph_med, 4)
                tc_by_day_per_channel[ch_id][day_label] = tc
            except Exception as exc:
                day_entry[f"ch{ch_id}_tuning_error"] = str(exc)

            # Within-day arm separation
            try:
                arm_tc = arm_tuning(ophys, beh, trials, n_bins=50)
                if arm_tc.left is not None and arm_tc.right is not None:
                    sep = within_day_arm_separation(arm_tc.left, arm_tc.right)
                    day_entry[f"ch{ch_id}_arm_separation"] = round(sep.similarity, 4)
                else:
                    day_entry[f"ch{ch_id}_arm_separation"] = None
            except Exception:
                day_entry[f"ch{ch_id}_arm_separation"] = None

        result["per_day_metrics"].append(day_entry)
        result["days_processed"].append(day)

    # Cross-day similarity matrices per channel
    for ch_id in ("0", "1"):
        ch_key = f"ch{ch_id}"
        tc_dict = tc_by_day_per_channel[ch_id]
        if len(tc_dict) < 2:
            result[ch_key]["cross_day_sim"] = "insufficient_days"
            result[ch_key]["off_diagonal_mean"] = float("nan")
            continue
        try:
            sim_mat = build_day_similarity_matrix(tc_dict)
            off_diag = _off_diagonal_mean(sim_mat.matrix)
            result[ch_key]["cross_day_sim_labels"] = sim_mat.labels
            result[ch_key]["cross_day_sim_matrix"] = sim_mat.matrix.tolist()
            result[ch_key]["off_diagonal_mean"] = round(off_diag, 4)
        except Exception as exc:
            result[ch_key]["cross_day_sim_error"] = str(exc)
            result[ch_key]["off_diagonal_mean"] = float("nan")

    # Aggregate per-channel arm separation and split-half means
    for ch_id in ("0", "1"):
        ch_key = f"ch{ch_id}"
        arm_vals = [
            d.get(f"ch{ch_id}_arm_separation")
            for d in result["per_day_metrics"]
            if d.get(f"ch{ch_id}_arm_separation") is not None
        ]
        finite_arm = [v for v in arm_vals if v is not None and np.isfinite(v)]
        result[ch_key]["arm_separation_mean"] = (
            round(float(np.mean(finite_arm)), 4) if finite_arm else float("nan")
        )
        result[ch_key]["arm_separation_std"] = (
            round(float(np.std(finite_arm)), 4) if len(finite_arm) > 1 else float("nan")
        )

        sph_vals = [
            d.get(f"ch{ch_id}_split_half_median")
            for d in result["per_day_metrics"]
            if isinstance(d.get(f"ch{ch_id}_split_half_median"), float)
        ]
        finite_sph = [v for v in sph_vals if np.isfinite(v)]
        result[ch_key]["split_half_mean"] = (
            round(float(np.mean(finite_sph)), 4) if finite_sph else float("nan")
        )

    # ch0 - ch1 differences
    for metric in ("off_diagonal_mean", "arm_separation_mean", "split_half_mean"):
        v0 = result["ch0"].get(metric, float("nan"))
        v1 = result["ch1"].get(metric, float("nan"))
        if np.isfinite(v0) and np.isfinite(v1):
            result["ch0_minus_ch1"][metric] = round(v0 - v1, 4)
        else:
            result["ch0_minus_ch1"][metric] = float("nan")

    return result


# ---------------------------------------------------------------------------
# Analysis 3 — Cohort null tests
# ---------------------------------------------------------------------------

def cohort_null_tests(
    cohort_sim_matrices: dict[str, SimilarityMatrix],
    *,
    n_perms: int = 1000,
) -> list[dict[str, Any]]:
    """Run permutation-based null tests for cross-day similarity differences.

    Tests whether SparseKO-7 has significantly lower cross-day similarity
    than each non-KO cohort.  Operates on per-day off-diagonal values
    (more observations than cohort-level scalars).

    Parameters
    ----------
    cohort_sim_matrices:
        Mapping from cohort name to ``SimilarityMatrix``.
    n_perms:
        Permutations for the null distribution.

    Returns a list of result dicts, one per comparison.
    """
    if "SparseKO-7" not in cohort_sim_matrices:
        return [{"error": "SparseKO-7 not found in cohort_sim_matrices"}]

    sko_mat = cohort_sim_matrices["SparseKO-7"]
    sko_vals = _off_diagonal_values(sko_mat.matrix)

    comparisons = [k for k in cohort_sim_matrices if k != "SparseKO-7"]
    results: list[dict[str, Any]] = []

    for ctrl_name in sorted(comparisons):
        ctrl_mat = cohort_sim_matrices[ctrl_name]
        ctrl_vals = _off_diagonal_values(ctrl_mat.matrix)

        if len(sko_vals) == 0 or len(ctrl_vals) == 0:
            results.append({
                "comparison": f"SparseKO-7 vs {ctrl_name}",
                "error": "Insufficient data for null test.",
            })
            continue

        obs_diff = float(np.mean(sko_vals)) - float(np.mean(ctrl_vals))

        null_result = permutation_cohort_null(
            obs_diff,
            sko_vals,
            ctrl_vals,
            n_perms=n_perms,
            seed=0,
        )

        entry: dict[str, Any] = {
            "comparison": f"SparseKO-7 vs {ctrl_name}",
            "sko_off_diag_mean": round(float(np.mean(sko_vals)), 4),
            "ctrl_off_diag_mean": round(float(np.mean(ctrl_vals)), 4),
            "sko_n_pairs": len(sko_vals),
            "ctrl_n_pairs": len(ctrl_vals),
            "observed_diff": round(obs_diff, 4),
            "null_mean": null_result["null_mean"],
            "null_std": null_result["null_std"],
            "z": null_result["z"],
            "p_empirical": null_result["p_empirical"],
            "n_perms": null_result["n_perms"],
            "effect_direction": null_result["effect_direction"],
            "note": null_result["note"],
        }

        # Claim boundary
        z = null_result.get("z", float("nan"))
        p = null_result.get("p_empirical", float("nan"))
        if np.isfinite(z) and z < -1.96 and np.isfinite(p) and p < 0.05:
            entry["claim"] = (
                "SparseKO shows significantly lower cross-day similarity (null-separated)."
            )
        elif np.isfinite(obs_diff) and obs_diff < 0:
            entry["claim"] = (
                "Directionally consistent: SparseKO lower, but not null-separated."
            )
        else:
            entry["claim"] = "No clear effect detected."

        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# Analysis 4 — Group null tests
# ---------------------------------------------------------------------------

def group_null_tests(
    subject_sim_matrices: dict[str, SimilarityMatrix],
    subject_groups: dict[str, str],
    *,
    target_group: str = "SparseKO",
    n_perms: int = 1000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Compare a target genotype group against other groups using subject means.

    Each subject contributes one summary value: the mean finite off-diagonal
    similarity from that subject's day-similarity matrix. This avoids treating
    repeated day-pairs from one animal as independent between-subject evidence.
    """
    group_values: dict[str, list[float]] = {}
    group_subjects: dict[str, list[str]] = {}

    for subject_id, sim_mat in subject_sim_matrices.items():
        group = subject_groups.get(subject_id, "Unknown")
        if group == "Unknown":
            continue
        off_diag_mean = _off_diagonal_mean(sim_mat.matrix)
        if not np.isfinite(off_diag_mean):
            continue
        group_values.setdefault(group, []).append(off_diag_mean)
        group_subjects.setdefault(group, []).append(subject_id)

    if target_group not in group_values:
        return [{"error": f"{target_group} not found in subject_sim_matrices"}]

    target_vals = group_values[target_group]
    target_subjects = sorted(group_subjects.get(target_group, []))
    comparisons = [group for group in sorted(group_values) if group != target_group]
    results: list[dict[str, Any]] = []

    for other_group in comparisons:
        other_vals = group_values[other_group]
        other_subjects = sorted(group_subjects.get(other_group, []))
        if len(target_vals) == 0 or len(other_vals) == 0:
            results.append(
                {
                    "comparison": f"{target_group} vs {other_group}",
                    "error": "Insufficient subject-level data for null test.",
                }
            )
            continue

        obs_diff = float(np.mean(target_vals)) - float(np.mean(other_vals))
        null_result = permutation_cohort_null(
            obs_diff,
            target_vals,
            other_vals,
            n_perms=n_perms,
            seed=seed,
        )

        entry: dict[str, Any] = {
            "comparison": f"{target_group} vs {other_group}",
            "target_group": target_group,
            "other_group": other_group,
            "target_mean": round(float(np.mean(target_vals)), 4),
            "other_mean": round(float(np.mean(other_vals)), 4),
            "target_n_subjects": len(target_vals),
            "other_n_subjects": len(other_vals),
            "target_subjects": ", ".join(target_subjects),
            "other_subjects": ", ".join(other_subjects),
            "observed_diff": round(obs_diff, 4),
            "null_mean": null_result["null_mean"],
            "null_std": null_result["null_std"],
            "z": null_result["z"],
            "p_empirical": null_result["p_empirical"],
            "n_perms": null_result["n_perms"],
            "effect_direction": null_result["effect_direction"],
            "note": null_result["note"],
        }

        z = null_result.get("z", float("nan"))
        p = null_result.get("p_empirical", float("nan"))
        if np.isfinite(z) and z < -1.96 and np.isfinite(p) and p < 0.05:
            entry["claim"] = (
                f"{target_group} shows lower subject-level cross-day stability than {other_group}."
            )
        elif np.isfinite(obs_diff) and obs_diff < 0:
            entry["claim"] = (
                f"Directionally consistent: {target_group} lower than {other_group}, but not null-separated."
            )
        else:
            entry["claim"] = "No clear effect detected."

        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# Analysis 5 — Day-lag similarity
# ---------------------------------------------------------------------------

def day_lag_similarity(
    sim_matrix: SimilarityMatrix,
) -> dict[int, dict[str, Any]]:
    """Extract cross-day similarity as a function of day-lag (1–5).

    For each lag, collects all (i, j) off-diagonal pairs where
    |day_i - day_j| == lag and computes mean, median, and n_pairs.

    Parameters
    ----------
    sim_matrix:
        ``SimilarityMatrix`` whose labels follow the pattern ``'dayN'``.

    Returns ``{lag: {"mean": ..., "median": ..., "n_pairs": ...}}``.
    """
    labels = sim_matrix.labels
    mat = sim_matrix.matrix

    # Parse day numbers from labels
    day_nums: list[int] = []
    for lab in labels:
        try:
            day_nums.append(int(str(lab).replace("day", "").strip()))
        except (ValueError, AttributeError):
            day_nums.append(-1)

    lag_data: dict[int, dict[str, Any]] = {}
    n = len(labels)

    for lag in range(1, 6):
        vals: list[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                di, dj = day_nums[i], day_nums[j]
                if di < 0 or dj < 0:
                    continue
                if abs(di - dj) == lag:
                    v = float(mat[i, j])
                    if np.isfinite(v):
                        vals.append(v)
        lag_data[lag] = {
            "mean": round(float(np.mean(vals)), 4) if vals else float("nan"),
            "median": round(float(np.median(vals)), 4) if vals else float("nan"),
            "n_pairs": len(vals),
        }

    return lag_data


def aggregate_group_day_lag(
    subject_lag_data: dict[str, dict[int, dict[str, Any]]],
    subject_groups: dict[str, str],
) -> dict[str, dict[int, dict[str, Any]]]:
    """Aggregate subject-level day-lag means into genotype-group summaries."""
    grouped: dict[str, dict[int, list[float]]] = {}

    for subject_id, lag_data in subject_lag_data.items():
        group = subject_groups.get(subject_id, "Unknown")
        if group == "Unknown":
            continue
        for lag, stats in lag_data.items():
            mean_val = float(stats.get("mean", float("nan")))
            if not np.isfinite(mean_val):
                continue
            grouped.setdefault(group, {}).setdefault(lag, []).append(mean_val)

    aggregated: dict[str, dict[int, dict[str, Any]]] = {}
    for group, lag_map in grouped.items():
        aggregated[group] = {}
        for lag in range(1, 6):
            vals = lag_map.get(lag, [])
            aggregated[group][lag] = {
                "mean": round(float(np.mean(vals)), 4) if vals else float("nan"),
                "median": round(float(np.median(vals)), 4) if vals else float("nan"),
                "n_subjects": len(vals),
            }
    return aggregated


def plot_day_lag_curves(
    lag_data_by_cohort: dict[str, dict[int, dict[str, Any]]],
    output_path: Path,
) -> None:
    """Plot mean cross-day similarity vs. day-lag per cohort.

    Parameters
    ----------
    lag_data_by_cohort:
        Mapping from cohort name to ``{lag: {"mean": ..., "n_pairs": ...}}``.
    output_path:
        Path where the PNG figure is saved.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _COLORS = {
        "Cre": "#2196F3",
        "Ctrl": "#FF9800",
        "SparseKO": "#F44336",
        "Cre-1": "#2196F3",
        "Cre-2": "#4CAF50",
        "Ctrl-9": "#FF9800",
        "SparseKO-7": "#F44336",
    }

    fig, ax = plt.subplots(figsize=(7, 4))

    for cohort, lag_data in sorted(lag_data_by_cohort.items()):
        lags = sorted(lag_data.keys())
        means = [lag_data[lag]["mean"] for lag in lags]

        finite_lags = [lags[i] for i, m in enumerate(means) if np.isfinite(m)]
        finite_means = [means[i] for i, m in enumerate(means) if np.isfinite(m)]
        if not finite_lags:
            continue

        ax.plot(
            finite_lags,
            finite_means,
            marker="o",
            label=cohort,
            color=_COLORS.get(cohort),
            linewidth=2,
            markersize=5,
        )

    ax.set_xlabel("Day lag", fontsize=12)
    ax.set_ylabel("Mean cross-day similarity (r)", fontsize=12)
    ax.set_title("Cross-day similarity by day lag", fontsize=13)
    ax.set_xticks(list(range(1, 6)))
    ax.legend(fontsize=10, loc="upper right")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _off_diagonal_mean(mat: np.ndarray) -> float:
    """Return mean of finite off-diagonal entries."""
    vals = _off_diagonal_values(mat)
    return float(np.mean(vals)) if vals else float("nan")


def _off_diagonal_values(mat: np.ndarray) -> list[float]:
    """Return all finite off-diagonal values from a square matrix."""
    n = mat.shape[0]
    return [
        float(mat[i, j])
        for i in range(n)
        for j in range(n)
        if i != j and np.isfinite(mat[i, j])
    ]
