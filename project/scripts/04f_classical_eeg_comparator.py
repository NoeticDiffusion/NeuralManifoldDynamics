"""04f — Classical EEG comparator for ds006848 encoding-phase analysis.

Tests whether the same Fast/FastDelay > Simultaneous/Slow condition ordering
observed in MNPS m/d also appears in classical EEG features during encoding.

Features tested
---------------
  eeg_theta__g_frontal           frontal theta power (4–8 Hz)
  eeg_alpha__g_frontal           frontal alpha power (8–13 Hz)
  eeg_alpha__g_parietal_occipital  parietal-occipital alpha (alpha engagement)
  eeg_beta__g_frontal            frontal beta power
  eeg_hjorth_complexity__g_frontal  frontal signal complexity
  eeg_sample_entropy__g_frontal  frontal sample entropy
  eeg_permutation_entropy__g_frontal  frontal permutation entropy
  eeg_hjorth_complexity          global complexity

The features come from the already-computed features.parquet, which has
t_start / t_end per epoch. Encoding windows are reconstructed from events.tsv
using the same logic as 04b/04c.

Outputs
-------
  subject_condition_medians.csv   per-subject × condition EEG feature medians
  friedman_results.csv            Friedman χ² / p for each EEG feature
  pairwise_contrasts.csv          BH-FDR Wilcoxon contrasts
  condition_profile.csv           group-level medians + IQR per feature
  summary.txt

Usage
-----
  python project/scripts/04f_classical_eeg_comparator.py \\
      --features-parquet J:/processed/openneuro/ds006848/features.parquet \\
      --bids-dir K:/ExternalReceivedDatasets/openneuro/received/ds006848 \\
      --out-dir J:/processed/openneuro/ds006848/04f_eeg_comparator
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
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONDITIONS = ["Simultaneous", "Fast", "FastDelay", "Slow"]
CONDITION_MAP = {
    "Retention_Simultaneous": "Simultaneous",
    "Retention_Fast": "Fast",
    "Retention_FastDelay": "FastDelay",
    "Retention_Slow": "Slow",
}
MIN_OVERLAP_FRAC = 0.25

EEG_FEATURES = [
    "eeg_theta__g_frontal",
    "eeg_alpha__g_frontal",
    "eeg_alpha__g_parietal_occipital",
    "eeg_beta__g_frontal",
    "eeg_hjorth_complexity__g_frontal",
    "eeg_sample_entropy__g_frontal",
    "eeg_permutation_entropy__g_frontal",
    "eeg_hjorth_complexity",
]


# ---------------------------------------------------------------------------
# Trial reconstruction (copied from 04b/04c for self-containment)
# ---------------------------------------------------------------------------

def reconstruct_trials(events_df: pd.DataFrame) -> pd.DataFrame:
    events_df = events_df.sort_values("onset").reset_index(drop=True)
    trials: List[Dict] = []
    for idx, row in events_df.iterrows():
        if row["trial_type"] not in CONDITION_MAP:
            continue
        condition = CONDITION_MAP[row["trial_type"]]
        ret_start = row["onset"]
        enc_events = []
        for jdx in range(idx - 1, max(-1, idx - 30), -1):
            ttype = events_df.at[jdx, "trial_type"]
            if ttype.startswith("Encoding_"):
                enc_events.append(events_df.at[jdx, "onset"])
            elif enc_events:
                break
        if not enc_events:
            continue
        enc_start = min(enc_events)
        enc_end = ret_start
        later = events_df[
            (events_df["onset"] > ret_start + 6.0)
            & (events_df["trial_type"] == "Digits_Retrieval")
        ]
        if later.empty:
            continue
        trials.append({"trial_id": len(trials), "condition": condition,
                       "enc_start": enc_start, "enc_end": enc_end})
    return pd.DataFrame(trials)


def overlap_fraction_arr(t_start: np.ndarray, t_end: np.ndarray,
                         iv_start: float, iv_end: float) -> np.ndarray:
    win_dur = t_end - t_start
    ovlp = np.maximum(0.0, np.minimum(t_end, iv_end) - np.maximum(t_start, iv_start))
    return np.where(win_dur > 0, ovlp / win_dur, 0.0)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    rank = np.empty_like(order)
    rank[order] = np.arange(1, n + 1)
    q = pvals * n / rank
    q[order] = np.minimum(q[order], np.minimum.accumulate(q[order][::-1])[::-1])
    return np.minimum(q, 1.0)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else np.nan


def friedman_posthoc(sub_cond: pd.DataFrame, metric: str):
    pivot = (
        sub_cond[sub_cond["condition"].isin(CONDITIONS)]
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
                      "median_a": round(np.median(paired[:, 0]), 4),
                      "median_b": round(np.median(paired[:, 1]), 4),
                      "W": stat, "p": pval,
                      "cohens_d": round(cohens_d(paired[:, 0], paired[:, 1]), 3)})
    if not pairs:
        return fr, pd.DataFrame()
    pw = pd.DataFrame(pairs)
    pw["q"] = bh_fdr(pw["p"].values)
    pw["sig"] = pw["q"] < 0.05
    pw.insert(0, "metric", metric)
    return fr, pw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(features_parquet: Path, bids_dir: Path, out_dir: Path,
        subjects: Optional[List[str]] = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_parquet(features_parquet)
    # Detect file/subject column names
    file_col = "file" if "file" in features.columns else "filename"
    sub_col = "subject" if "subject" in features.columns else "subject_id"

    # Keep only verbalwm epochs
    features = features[features[file_col].str.contains("verbalwm", case=False, na=False)].copy()
    logger.info("Features loaded: %d verbalwm epochs", len(features))

    # Derive BIDS subject ID from filename (e.g. "sub-001_task-verbalwm_eeg.vhdr" → "sub-001")
    features["subject_bids"] = features[file_col].str.extract(r"(sub-\d+)")

    # Check which EEG features are available
    available = [f for f in EEG_FEATURES if f in features.columns]
    missing = [f for f in EEG_FEATURES if f not in features.columns]
    if missing:
        logger.warning("Missing EEG features: %s", missing)
    logger.info("Available EEG features: %s", available)

    all_summary: List[pd.DataFrame] = []

    for subject_id, sub_feats in features.groupby("subject_bids"):
        subject = str(subject_id)
        if subjects and subject not in subjects:
            continue

        events_file = bids_dir / subject / "eeg" / f"{subject}_task-verbalwm_events.tsv"
        if not events_file.exists():
            continue

        try:
            events_df = pd.read_csv(events_file, sep="\t")
            trials = reconstruct_trials(events_df)
        except Exception as exc:
            logger.error("Trial reconstruction failed %s: %s", subject, exc)
            continue

        if trials.empty:
            continue

        t_start = sub_feats["t_start"].values
        t_end = sub_feats["t_end"].values

        rows: List[Dict] = []
        for _, trial in trials.iterrows():
            frac = overlap_fraction_arr(t_start, t_end, trial["enc_start"], trial["enc_end"])
            sel = sub_feats[frac >= MIN_OVERLAP_FRAC]
            if sel.empty:
                continue
            r: Dict = {"trial_id": trial["trial_id"], "condition": trial["condition"]}
            for feat in available:
                r[feat] = sel[feat].median()
            rows.append(r)

        if not rows:
            logger.warning("%s: no epoch-trial matches", subject)
            continue

        per_trial = pd.DataFrame(rows)
        summary_rows = []
        for cond, grp in per_trial.groupby("condition"):
            sr: Dict = {"subject": subject, "condition": cond, "n_trials": len(grp)}
            for feat in available:
                sr[feat] = grp[feat].median()
            summary_rows.append(sr)
        all_summary.append(pd.DataFrame(summary_rows))

    if not all_summary:
        logger.error("No data collected")
        return

    summary_df = pd.concat(all_summary, ignore_index=True)
    summary_df.to_csv(out_dir / "subject_condition_medians.csv", index=False)
    logger.info("Summary: %d subjects × conditions", len(summary_df))

    # Condition profile
    cond_profile = summary_df.groupby("condition")[available].agg(["median", "std"]).reset_index()
    cond_profile.to_csv(out_dir / "condition_profile.csv", index=False)

    # Friedman + pairwise
    fr_rows, pw_rows = [], []
    for feat in available:
        fr, pw = friedman_posthoc(summary_df, feat)
        if not fr.empty:
            fr_rows.append(fr)
        if not pw.empty:
            pw_rows.append(pw)

    if fr_rows:
        pd.concat(fr_rows, ignore_index=True).to_csv(out_dir / "friedman_results.csv", index=False)
    if pw_rows:
        pd.concat(pw_rows, ignore_index=True).to_csv(out_dir / "pairwise_contrasts.csv", index=False)

    _write_summary(summary_df, available, fr_rows, pw_rows, out_dir)
    logger.info("04f complete → %s", out_dir)


def _write_summary(summary_df, available, fr_rows, pw_rows, out_dir):
    lines = ["=== 04f Classical EEG Comparator ===\n"]
    lines.append(f"Subjects: {summary_df['subject'].nunique()}\n")
    lines.append(f"Features tested: {available}\n")

    lines.append("\n--- Condition medians (group level) ---\n")
    gmed = summary_df.groupby("condition")[available].median().round(4)
    lines.append(gmed.to_string())

    if fr_rows:
        fr_all = pd.concat(fr_rows)
        lines.append("\n\n--- Friedman results ---\n")
        lines.append(fr_all[["metric", "chi2", "p_friedman", "n_subjects"]].to_string(index=False))

        sig_fr = fr_all[fr_all["p_friedman"] < 0.05]
        lines.append(f"\n\nSignificant Friedman (p<0.05): {len(sig_fr)}/{len(fr_all)}\n")
        if not sig_fr.empty:
            lines.append(sig_fr["metric"].tolist().__str__())

    if pw_rows:
        pw_all = pd.concat(pw_rows)
        sig_pw = pw_all[pw_all["sig"]]
        lines.append(f"\n\n--- Significant pairwise (q<0.05): {len(sig_pw)}/{len(pw_all)} ---\n")
        if not sig_pw.empty:
            lines.append(sig_pw.to_string(index=False))

    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-parquet", type=Path,
                    default=Path("J:/processed/openneuro/ds006848/features.parquet"))
    ap.add_argument("--bids-dir", type=Path,
                    default=Path("K:/ExternalReceivedDatasets/openneuro/received/ds006848"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("J:/processed/openneuro/ds006848/04f_eeg_comparator"))
    ap.add_argument("--subjects", nargs="*", default=None)
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.features_parquet, args.bids_dir, args.out_dir, args.subjects)
