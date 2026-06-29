"""04d — Behavioral condition review for ds006848 verbal WM.

Tests whether manifold m/d tracks difficulty, speed, or encoding success
across and within conditions.

Outputs
-------
  behavioral_condition_summary.csv    accuracy/RT by subject × condition
  trial_level_merged.csv              per-trial beh + MNPS (from 04b outputs)
  mnps_behavior_correlations.csv      Spearman r(m, NCorrect) etc. per condition
  serial_position_accuracy.csv        P(correct) by serial position × condition
  friedman_behavioral.csv             Friedman for NCorrect / partialScore
  pairwise_behavioral.csv             BH-FDR Wilcoxon for behavioral measures
  summary.txt

Usage
-----
  python project/scripts/04d_behavioral_condition_review.py \\
      --bids-dir K:/ExternalReceivedDatasets/openneuro/received/ds006848 \\
      --04b-dir J:/processed/openneuro/ds006848/04b_encoding_phase \\
      --out-dir J:/processed/openneuro/ds006848/04d_behavioral
"""
from __future__ import annotations

import argparse
import logging
import warnings
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, wilcoxon

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONDITIONS = ["Simultaneous", "Fast", "FastDelay", "Slow"]
# beh.tsv uses "Fast+delay" for FastDelay
BEH_COND_MAP = {
    "Fast+delay": "FastDelay",
    "Fast": "Fast",
    "Simultaneous": "Simultaneous",
    "Slow": "Slow",
}

# ---------------------------------------------------------------------------


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    rank = np.empty_like(order)
    rank[order] = np.arange(1, n + 1)
    q = pvals * n / rank
    q_min = np.minimum.accumulate(q[order][::-1])[::-1]
    q[order] = np.minimum(q[order], q_min)
    return np.minimum(q, 1.0)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else np.nan


def load_behavioral_data(bids_dir: Path, subjects: Optional[List[str]] = None) -> pd.DataFrame:
    """Load and concatenate beh.tsv files for all subjects."""
    frames: List[pd.DataFrame] = []
    for sub_dir in sorted(bids_dir.iterdir()):
        if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
            continue
        if subjects and sub_dir.name not in subjects:
            continue
        beh_files = list((sub_dir / "beh").glob("*task-verbalwm_beh.tsv")) if (sub_dir / "beh").exists() else []
        if not beh_files:
            # some datasets put beh files in eeg/
            beh_files = list((sub_dir / "eeg").glob("*task-verbalwm_beh.tsv"))
        if not beh_files:
            continue
        try:
            df = pd.read_csv(beh_files[0], sep="\t")
            df["subject"] = sub_dir.name
            frames.append(df)
        except Exception as exc:
            logger.warning("Could not read beh for %s: %s", sub_dir.name, exc)
    if not frames:
        raise FileNotFoundError(f"No behavioral files found under {bids_dir}")
    beh = pd.concat(frames, ignore_index=True)
    beh["condition"] = beh["condition"].map(BEH_COND_MAP).fillna(beh["condition"])
    return beh


def parse_serial_position(beh: pd.DataFrame) -> pd.DataFrame:
    """Expand triggerCorrect into per-position binary columns."""
    beh = beh.copy()
    tc = beh["triggerCorrect"].astype(str).str.zfill(7)
    for pos in range(7):
        beh[f"pos{pos+1}_correct"] = tc.str[pos].astype(int)
    return beh


def friedman_posthoc(df: pd.DataFrame, metric: str) -> tuple:
    pivot = (
        df[df["condition"].isin(CONDITIONS)]
        .pivot_table(index="subject", columns="condition", values=metric, aggfunc="median")
        .dropna()
    )
    n = len(pivot)
    avail = [c for c in CONDITIONS if c in pivot.columns]
    if n < 3 or len(avail) < 3:
        return pd.DataFrame(), pd.DataFrame()
    chi2, p = stats.friedmanchisquare(*[pivot[c].values for c in avail])
    fr = pd.DataFrame([{"metric": metric, "chi2": round(chi2, 3), "dof": len(avail) - 1,
                        "p_friedman": p, "n_subjects": n}])
    pairs: List[Dict] = []
    for ca, cb in combinations(avail, 2):
        paired = np.column_stack([pivot[ca].values, pivot[cb].values])
        paired = paired[np.isfinite(paired).all(axis=1)]
        if len(paired) < 3:
            continue
        try:
            stat, pval = wilcoxon(paired[:, 0], paired[:, 1])
        except ValueError:
            pval, stat = 1.0, np.nan
        pairs.append({"condition_a": ca, "condition_b": cb,
                      "median_a": round(np.median(paired[:, 0]), 3),
                      "median_b": round(np.median(paired[:, 1]), 3),
                      "W": stat, "p": pval,
                      "cohens_d": round(cohens_d(paired[:, 0], paired[:, 1]), 3)})
    if not pairs:
        return fr, pd.DataFrame()
    pw = pd.DataFrame(pairs)
    pw["q"] = bh_fdr(pw["p"].values)
    pw["sig"] = pw["q"] < 0.05
    pw.insert(0, "metric", metric)
    return fr, pw


def run(bids_dir: Path, dir_04b: Path, out_dir: Path,
        subjects: Optional[List[str]] = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load behavioral data
    beh = load_behavioral_data(bids_dir, subjects)
    beh = parse_serial_position(beh)
    n_subjects = beh["subject"].nunique()
    logger.info("Behavioral data: %d trials across %d subjects", len(beh), n_subjects)

    # -----------------------------------------------------------------------
    # 1. Condition-level behavioral summary
    # -----------------------------------------------------------------------
    cond_summary = (
        beh.groupby(["subject", "condition"])
        .agg(
            n_trials=("NCorrect", "count"),
            mean_NCorrect=("NCorrect", "mean"),
            median_NCorrect=("NCorrect", "median"),
            mean_partial=("partialScore", "mean"),
            median_partial=("partialScore", "median"),
        )
        .reset_index()
    )
    cond_summary.to_csv(out_dir / "behavioral_condition_summary.csv", index=False)

    # -----------------------------------------------------------------------
    # 2. Friedman + pairwise for behavioral measures
    # -----------------------------------------------------------------------
    fr_rows, pw_rows = [], []
    for metric in ["NCorrect", "partialScore"]:
        fr, pw = friedman_posthoc(cond_summary.rename(
            columns={"mean_NCorrect": "NCorrect", "mean_partial": "partialScore"}
        ) if metric == "NCorrect" else cond_summary.rename(
            columns={"mean_NCorrect": "NCorrect", "mean_partial": "partialScore"}
        ), metric)
        if not fr.empty:
            fr_rows.append(fr)
        if not pw.empty:
            pw_rows.append(pw)

    # Simpler: compute per-subject condition median then test
    fr_rows2, pw_rows2 = [], []
    for metric in ["NCorrect", "partialScore"]:
        sub_cond = beh.groupby(["subject", "condition"])[metric].median().reset_index()
        fr, pw = friedman_posthoc(sub_cond, metric)
        if not fr.empty:
            fr_rows2.append(fr)
        if not pw.empty:
            pw_rows2.append(pw)
    if fr_rows2:
        pd.concat(fr_rows2, ignore_index=True).to_csv(out_dir / "friedman_behavioral.csv", index=False)
    if pw_rows2:
        pd.concat(pw_rows2, ignore_index=True).to_csv(out_dir / "pairwise_behavioral.csv", index=False)

    # -----------------------------------------------------------------------
    # 3. Serial position accuracy
    # -----------------------------------------------------------------------
    pos_cols = [f"pos{i+1}_correct" for i in range(7)]
    sp_acc = (
        beh.groupby(["subject", "condition"])[pos_cols].mean()
        .reset_index()
    )
    sp_summary = (
        sp_acc.groupby("condition")[pos_cols].mean().reset_index()
    )
    sp_summary.to_csv(out_dir / "serial_position_accuracy.csv", index=False)
    logger.info("Serial position analysis complete")

    # -----------------------------------------------------------------------
    # 4. MNPS–behavior correlation: load 04b per-trial data and merge
    # -----------------------------------------------------------------------
    trial_merged = _load_and_merge_mnps_behavior(beh, dir_04b)
    if trial_merged is not None and not trial_merged.empty:
        trial_merged.to_csv(out_dir / "trial_level_merged.csv", index=False)
        # Spearman r(m, NCorrect) per condition per subject, then median across subjects
        corr_rows: List[Dict] = []
        for (sub, cond), grp in trial_merged.groupby(["subject", "condition"]):
            if len(grp) < 5:
                continue
            for mnps_col in ["m", "d", "e"]:
                for beh_col in ["NCorrect", "partialScore"]:
                    if mnps_col not in grp.columns:
                        continue
                    vals = grp[[mnps_col, beh_col]].dropna()
                    if len(vals) < 5:
                        continue
                    r, pval = spearmanr(vals[mnps_col], vals[beh_col])
                    corr_rows.append({"subject": sub, "condition": cond,
                                      "mnps": mnps_col, "behavior": beh_col,
                                      "spearman_r": round(r, 3), "p": round(pval, 4)})
        if corr_rows:
            corr_df = pd.DataFrame(corr_rows)
            corr_summary = (
                corr_df.groupby(["condition", "mnps", "behavior"])["spearman_r"]
                .agg(["median", "mean", "std", "count"])
                .reset_index()
            )
            corr_df.to_csv(out_dir / "mnps_behavior_correlations_raw.csv", index=False)
            corr_summary.to_csv(out_dir / "mnps_behavior_correlations.csv", index=False)
            logger.info("MNPS–behavior correlations computed for %d subject×condition pairs", len(corr_rows))
    else:
        logger.warning("Could not merge 04b trial-level data — skipping MNPS–behavior correlation")

    # -----------------------------------------------------------------------
    # 5. Summary text
    # -----------------------------------------------------------------------
    _write_summary(beh, sp_summary, fr_rows2, pw_rows2, out_dir)
    logger.info("04d complete → %s", out_dir)


def _load_and_merge_mnps_behavior(beh: pd.DataFrame, dir_04b: Path) -> Optional[pd.DataFrame]:
    """Attempt to merge 04b per-trial MNPS with behavioral data.

    04b's A_full_encoding/subject_condition_medians.csv has per-subject ×
    condition medians (not per-trial). The per-trial data needs to be
    regenerated or the per-trial CSVs need to be saved by 04b.
    Currently returns per-subject × condition merge only.
    """
    medians_path = dir_04b / "A_full_encoding" / "subject_condition_medians.csv"
    if not medians_path.exists():
        return None
    mnps = pd.read_csv(medians_path)
    beh_sub_cond = (
        beh.groupby(["subject", "condition"])
        .agg(NCorrect=("NCorrect", "mean"), partialScore=("partialScore", "mean"))
        .reset_index()
    )
    merged = mnps.merge(beh_sub_cond, on=["subject", "condition"], how="inner")
    return merged


def _write_summary(beh, sp_summary, fr_rows, pw_rows, out_dir):
    lines = ["=== 04d Behavioral Condition Review ===\n"]
    lines.append(f"Subjects: {beh['subject'].nunique()}, Trials total: {len(beh)}\n")

    lines.append("\n--- NCorrect by condition (median across subjects) ---\n")
    sc = beh.groupby("condition")["NCorrect"].agg(["median", "mean", "std"]).round(2)
    lines.append(sc.to_string())

    lines.append("\n\n--- partialScore by condition ---\n")
    sc2 = beh.groupby("condition")["partialScore"].agg(["median", "mean", "std"]).round(2)
    lines.append(sc2.to_string())

    lines.append("\n\n--- Serial position accuracy ---\n")
    lines.append(sp_summary.to_string(index=False))

    if fr_rows:
        lines.append("\n\n--- Friedman tests ---\n")
        lines.append(pd.concat(fr_rows).to_string(index=False))

    if pw_rows:
        pw_all = pd.concat(pw_rows)
        sig = pw_all[pw_all["sig"]]
        lines.append(f"\n\n--- Significant pairwise (q<0.05): {len(sig)}/{len(pw_all)} ---\n")
        if not sig.empty:
            lines.append(sig.to_string(index=False))

    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bids-dir", type=Path,
                    default=Path("K:/ExternalReceivedDatasets/openneuro/received/ds006848"))
    ap.add_argument("--04b-dir", type=Path,
                    default=Path("J:/processed/openneuro/ds006848/04b_encoding_phase"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("J:/processed/openneuro/ds006848/04d_behavioral"))
    ap.add_argument("--subjects", nargs="*", default=None)
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.bids_dir, getattr(args, "04b_dir"), args.out_dir, args.subjects)
