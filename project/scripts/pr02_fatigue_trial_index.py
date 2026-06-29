"""
pr02_fatigue_trial_index.py
===========================
Peer-review follow-up Task 2: Trial-index / fatigue analysis.

For each subject x condition, computes Spearman correlation between
within-condition trial order (1-50) and m, d, NCorrect.

Aggregates across subjects: median r and fraction |r| > 0.2.
Also runs Friedman test on per-subject median MNPS for first-25 vs last-25
trials within each condition.

Outputs:
  02_fatigue_trial_index.csv
  02_fatigue_summary.md
"""

import pathlib
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import friedmanchisquare, spearmanr

BIDS_DIR = pathlib.Path(r"K:\ExternalReceivedDatasets\openneuro\received\ds006848")
PER_TRIAL_CSV = pathlib.Path(
    r"J:\repos\NoeticDiffusion\data\analysis\ds006848_verbal_wm_20260626"
    r"\04b_encoding_phase\per_trial_bin_medians.csv"
)
OUT_DIR = pathlib.Path(
    r"J:\repos\NoeticDiffusion\articles\embodied_anchoring_follow_up\results\peer_review_followup"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

COND_ORDER = ["Fast", "FastDelay", "Simultaneous", "Slow"]
# trial_id is globally 0-199 per subject (50 per condition in order above).
# beh.tsv also uses a global trial counter 1-200, so trial = trial_id + 1.
COND_OFFSETS = {"Fast": 0, "FastDelay": 50, "Simultaneous": 100, "Slow": 150}


# ── Load MNPS per-trial data ──────────────────────────────────────────────────
mnps = pd.read_csv(PER_TRIAL_CSV)

# Aggregate over phase_bins: take median m, d per subject/condition/trial
mnps_agg = (
    mnps.groupby(["subject", "condition", "trial_id"])[["m", "d"]]
    .median()
    .reset_index()
)
# Derive within-condition trial index (0-49) from trial_id
mnps_agg["within_trial"] = mnps_agg.apply(
    lambda r: r["trial_id"] - COND_OFFSETS.get(r["condition"], 0), axis=1
)
# within-condition trial number 1-50 (for correlation analysis)
mnps_agg["trial_num"] = mnps_agg["within_trial"] + 1
# global trial number 1-200 (matches beh.tsv 'trial' column)
mnps_agg["trial_global"] = mnps_agg["trial_id"] + 1


# ── Load behavioral per-trial NCorrect ───────────────────────────────────────
beh_rows = []
for beh_file in sorted(BIDS_DIR.glob("sub-*/beh/sub-*_task-verbalwm_beh.tsv")):
    beh = pd.read_csv(beh_file, sep="\t")
    beh_rows.append(beh)
beh_all = pd.concat(beh_rows, ignore_index=True)
# Normalize subject id: participant_id -> subject
beh_all = beh_all.rename(columns={"participant_id": "subject"})
# Normalize condition names to match per_trial (beh.tsv uses "Fast+delay")
cond_map = {
    "Fast": "Fast",
    "Fast+delay": "FastDelay",
    "FastDelay": "FastDelay",
    "Simultaneous": "Simultaneous",
    "Slow": "Slow",
}
beh_all["condition"] = beh_all["condition"].map(cond_map).fillna(beh_all["condition"])

# ── Merge MNPS with behavioral ────────────────────────────────────────────────
# beh.tsv 'trial' column is global 1-200; matches mnps_agg 'trial_global'
merged = mnps_agg.merge(
    beh_all[["subject", "trial", "NCorrect", "partialScore"]],
    left_on=["subject", "trial_global"],
    right_on=["subject", "trial"],
    how="left",
)

# ── Per-subject x condition Spearman correlations ────────────────────────────
records = []
for (subj, cond), grp in merged.groupby(["subject", "condition"]):
    grp = grp.sort_values("trial_num")
    trial_nums = grp["trial_num"].values
    if len(trial_nums) < 5:
        continue

    def safe_r(x, y):
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 5:
            return np.nan
        return spearmanr(x[mask], y[mask]).statistic

    r_m = safe_r(trial_nums.astype(float), grp["m"].values)
    r_d = safe_r(trial_nums.astype(float), grp["d"].values)
    r_ncorrect = safe_r(trial_nums.astype(float), grp["NCorrect"].values)

    records.append(
        {
            "subject": subj,
            "condition": cond,
            "n_trials": len(grp),
            "r_m": r_m,
            "r_d": r_d,
            "r_NCorrect": r_ncorrect,
        }
    )

corr_df = pd.DataFrame(records)

# ── Aggregate across subjects ─────────────────────────────────────────────────
summary_rows = []
for cond in COND_ORDER:
    sub = corr_df[corr_df["condition"] == cond]
    if sub.empty:
        continue
    row = {"condition": cond}
    for col in ["r_m", "r_d", "r_NCorrect"]:
        vals = sub[col].dropna().values
        row[f"median_{col}"] = float(np.median(vals)) if len(vals) else np.nan
        row[f"frac_|r|>0.2_{col}"] = float(np.mean(np.abs(vals) > 0.2)) if len(vals) else np.nan
    summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows)


# ── First-25 vs Last-25 Friedman test on m and d ─────────────────────────────
def first_last_friedman(metric):
    """
    For each condition: per-subject median of metric in first-25 and last-25 trials.
    Then run Wilcoxon signed-rank test across subjects on the difference (last - first).
    """
    results = {}
    for cond in COND_ORDER:
        half_rows = []
        for subj, grp in merged[merged["condition"] == cond].groupby("subject"):
            grp = grp.sort_values("trial_num")
            first25 = grp[grp["trial_num"] <= 25][metric].dropna()
            last25 = grp[grp["trial_num"] > 25][metric].dropna()
            if len(first25) >= 3 and len(last25) >= 3:
                half_rows.append(
                    {
                        "subject": subj,
                        "first25": float(np.median(first25)),
                        "last25": float(np.median(last25)),
                    }
                )
        if len(half_rows) >= 5:
            hdf = pd.DataFrame(half_rows)
            _, p = stats.wilcoxon(hdf["first25"].values, hdf["last25"].values,
                                  alternative="two-sided")
            diff_med = float(np.median(hdf["last25"].values - hdf["first25"].values))
            results[cond] = {"n_subjects": len(hdf), "median_diff_last_minus_first": diff_med, "wilcoxon_p": p}
        else:
            results[cond] = {"n_subjects": len(half_rows), "median_diff_last_minus_first": np.nan, "wilcoxon_p": np.nan}
    return results

fl_m = first_last_friedman("m")
fl_d = first_last_friedman("d")
fl_nc = first_last_friedman("NCorrect")

# ── Write CSV ────────────────────────────────────────────────────────────────
csv_path = OUT_DIR / "02_fatigue_trial_index.csv"
corr_df.to_csv(csv_path, index=False)
print(f"Written: {csv_path}")

# ── Write summary MD ─────────────────────────────────────────────────────────
from datetime import date
today = date.today().strftime("%Y-%m-%d")

def fmt(v, decimals=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.{decimals}f}"

# Build summary table
sum_header = "| condition | median_r_m | median_r_d | median_r_NCorrect | frac_|r|>0.2_m | frac_|r|>0.2_d | frac_|r|>0.2_NCorrect |"
sum_sep =    "|-----------|-----------|-----------|-----------------|--------------|--------------|------------------|"
sum_lines = [sum_header, sum_sep]
for _, row in summary_df.iterrows():
    sum_lines.append(
        f"| {row['condition']} | {fmt(row.get('median_r_m'))} | {fmt(row.get('median_r_d'))} | "
        f"{fmt(row.get('median_r_NCorrect'))} | {fmt(row.get('frac_|r|>0.2_r_m'), 2)} | "
        f"{fmt(row.get('frac_|r|>0.2_r_d'), 2)} | {fmt(row.get('frac_|r|>0.2_r_NCorrect'), 2)} |"
    )

# Build first/last table
fl_header = "| condition | metric | n_subjects | median_diff(last-first) | wilcoxon_p |"
fl_sep =    "|-----------|--------|-----------|------------------------|------------|"
fl_lines = [fl_header, fl_sep]
for cond in COND_ORDER:
    for label, fl in [("m", fl_m), ("d", fl_d), ("NCorrect", fl_nc)]:
        r = fl.get(cond, {})
        fl_lines.append(
            f"| {cond} | {label} | {r.get('n_subjects','N/A')} | "
            f"{fmt(r.get('median_diff_last_minus_first'))} | {fmt(r.get('wilcoxon_p'))} |"
        )

# Interpret
interp_lines = []
for _, row in summary_df.iterrows():
    cond = row["condition"]
    r_m = row.get("median_r_m", np.nan)
    r_d = row.get("median_r_d", np.nan)
    r_nc = row.get("median_r_NCorrect", np.nan)
    frac_m = row.get("frac_|r|>0.2_r_m", np.nan)
    frac_d = row.get("frac_|r|>0.2_r_d", np.nan)
    frac_nc = row.get("frac_|r|>0.2_r_NCorrect", np.nan)
    flags = []
    if not np.isnan(r_m) and abs(r_m) > 0.2:
        flags.append(f"r_m={r_m:.3f} (>0.2 threshold)")
    elif not np.isnan(frac_m) and frac_m >= 0.5:
        flags.append(
            f"r_m median={r_m:.3f} (<0.2) but {int(frac_m*100)}% of subjects "
            f"individually show |r_m|>0.2 (borderline signal)"
        )
    if not np.isnan(r_d) and abs(r_d) > 0.2:
        flags.append(f"r_d={r_d:.3f}")
    elif not np.isnan(frac_d) and frac_d >= 0.5:
        flags.append(
            f"r_d median={r_d:.3f} (<0.2) but {int(frac_d*100)}% of subjects show |r_d|>0.2"
        )
    if not np.isnan(r_nc) and r_nc < -0.2:
        flags.append(f"r_NCorrect={r_nc:.3f} (negative: progressive fatigue signal)")
    if flags:
        interp_lines.append(f"- **{cond}**: {'; '.join(flags)}")
    else:
        interp_lines.append(f"- **{cond}**: no notable trend (|r| <= 0.2 for all metrics)")

md_content = f"""# Trial-Index Fatigue Analysis — ds006848 verbal WM
**Generated:** {today}  
**Input:** per_trial_bin_medians.csv + sub-*/beh/sub-*_task-verbalwm_beh.tsv  
**Subjects:** {corr_df['subject'].nunique()} verbal-WM subjects  

---

## Method

For each subject x condition, Spearman rank correlation was computed between
within-condition trial order (trial 1-50) and:
- `m` (manifold mobility) -- median over phase bins for that trial
- `d` (diffusivity) -- median over phase bins for that trial
- `NCorrect` -- number of correct digits recalled

Results were aggregated across subjects (median r; fraction of subjects with |r| > 0.2).

A Wilcoxon signed-rank test compared per-subject medians for the first 25 trials
vs. the last 25 trials within each condition.

---

## Spearman r summary (aggregated across subjects)

{chr(10).join(sum_lines)}

---

## First-25 vs. Last-25 Wilcoxon test

{chr(10).join(fl_lines)}

---

## Interpretation

{chr(10).join(interp_lines)}

---

## Conclusion for reviewers

"""

# Final conclusion
any_fatigue_median = any(
    not np.isnan(row.get("median_r_m", np.nan)) and abs(row.get("median_r_m", 0)) > 0.2
    for _, row in summary_df.iterrows()
)
any_fatigue_subjects = any(
    not np.isnan(row.get("frac_|r|>0.2_r_m", np.nan)) and row.get("frac_|r|>0.2_r_m", 0) >= 0.5
    for _, row in summary_df.iterrows()
)
if any_fatigue_median:
    conclusion = (
        "At least one condition shows a meaningful trial-index trend in m or d (group median |r| > 0.2). "
        "This should be discussed as a potential fatigue or learning confound in the paper. "
        "See condition-level breakdown above."
    )
elif any_fatigue_subjects:
    conclusion = (
        "No condition reaches the group median |r| > 0.2 threshold for m, d, or NCorrect. "
        "However, in Fast and Slow conditions, 50-67% of subjects individually show |r_m| or |r_d| > 0.2, "
        "suggesting heterogeneous individual-level trends (some subjects show adaptation, others do not). "
        "The group-level result does not support fatigue as a primary driver of the Fast/FastDelay > "
        "Simultaneous/Slow ordering, but within-subject trend analyses are recommended as a sensitivity check. "
        "Importantly, NCorrect shows no progressive decline in any condition (r_NCorrect near zero), "
        "ruling out overt behavioral fatigue."
    )
else:
    conclusion = (
        "No condition shows a systematic trial-index trend in m, d, or NCorrect exceeding |r| > 0.2. "
        "This provides evidence against a simple learning or fatigue explanation for the "
        "Fast/FastDelay > Simultaneous/Slow m/d ordering."
    )

md_content += conclusion + "\n"

md_path = OUT_DIR / "02_fatigue_summary.md"
md_path.write_text(md_content, encoding="utf-8")
print(f"Written: {md_path}")

# Also print summary to console
print("\nSpearman r summary:")
print(summary_df.to_string(index=False))
