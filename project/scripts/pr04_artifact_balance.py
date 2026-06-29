"""
pr04_artifact_balance.py
========================
Peer-review follow-up: Condition-structured artifact audit for ds006848.

The primary concern: Fast/FastDelay involve rapid sequential visual stimuli
(400 ms/digit) while Simultaneous presents all 7 digits at once. This could
produce condition-structured differences in eye movements / blinks and muscle
artifacts that contaminate MNPS m/d.

Method:
  1. Load per-epoch EEG spectral features from features.parquet (verbalwm only).
  2. For each subject, reconstruct encoding intervals from events.tsv.
  3. Assign each epoch to a condition via vectorized overlap (>=25%).
  4. Per subject x condition: compute median of each artifact proxy.
  5. Friedman test across 4 conditions + pairwise Wilcoxon (BH-FDR).

Artifact proxies:
  - eeg_highfreq_power_30_45__g_frontal  : frontal 30-45 Hz (blink/EMG)
  - eeg_highfreq_power_30_45__g_temporal : temporal 30-45 Hz (muscle)
  - eeg_gamma__g_frontal                 : frontal gamma
  - eeg_gamma__g_temporal                : temporal gamma
  - eeg_hjorth_mobility__g_frontal       : rapid amplitude changes (blink proxy)

Outputs:
  04_artifact_balance_by_condition.csv
  04_artifact_balance_summary.md
"""

from __future__ import annotations
import warnings
import pathlib
from itertools import combinations
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

warnings.filterwarnings("ignore")

FEATURES_PARQUET = pathlib.Path(
    r"J:\repos\NoeticDiffusion\data\raw\ds006848_features_20260626.parquet"
)
BIDS_DIR = pathlib.Path(
    r"K:\ExternalReceivedDatasets\openneuro\received\ds006848"
)
OUT_DIR = pathlib.Path(
    r"J:\repos\NoeticDiffusion\articles\embodied_anchoring_follow_up\results\peer_review_followup"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SEC = 8.0
COND_ORDER = ["Fast", "FastDelay", "Simultaneous", "Slow"]
RETENTION_TO_COND = {
    "Retention_Fast": "Fast",
    "Retention_FastDelay": "FastDelay",
    "Retention_Simultaneous": "Simultaneous",
    "Retention_Slow": "Slow",
}
ARTIFACT_PROXIES = [
    "eeg_highfreq_power_30_45__g_frontal",
    "eeg_highfreq_power_30_45__g_temporal",
    "eeg_gamma__g_frontal",
    "eeg_gamma__g_temporal",
    "eeg_hjorth_mobility__g_frontal",
]


def reconstruct_trials(events_path: pathlib.Path) -> pd.DataFrame:
    ev = pd.read_csv(events_path, sep="\t").sort_values("onset").reset_index(drop=True)
    trials, enc_onset = [], None
    for _, row in ev.iterrows():
        tt = str(row.get("trial_type", ""))
        onset = float(row["onset"])
        if tt.startswith("Encoding_DigitValue_") or tt == "Encoding_Set_Simultaneous":
            if enc_onset is None:
                enc_onset = onset
        elif tt in RETENTION_TO_COND:
            if enc_onset is not None:
                trials.append({"condition": RETENTION_TO_COND[tt],
                                "enc_onset": enc_onset, "enc_offset": onset})
            enc_onset = None
        elif tt in ("Baseline_2s", "Digits_Retrieval", "Boundary", "Experiment_End"):
            enc_onset = None
    return pd.DataFrame(trials)


def assign_conditions_vectorized(epochs: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    if trials.empty or epochs.empty:
        return pd.DataFrame()

    ep = epochs[["t_start"]].copy()
    ep["t_end"] = ep["t_start"] + WINDOW_SEC
    ep["epoch_idx"] = np.arange(len(ep))

    tr = trials.reset_index(drop=True)
    cross = ep.merge(tr, how="cross")

    overlap = (
        np.minimum(cross["t_end"], cross["enc_offset"])
        - np.maximum(cross["t_start"], cross["enc_onset"])
    ).clip(lower=0) / WINDOW_SEC
    cross["enc_overlap_frac"] = overlap
    cross = cross[cross["enc_overlap_frac"] >= 0.25]
    if cross.empty:
        return pd.DataFrame()

    cross = cross.sort_values("enc_overlap_frac", ascending=False)
    cross = cross.drop_duplicates(subset=["epoch_idx"], keep="first")

    result = epochs.copy().reset_index(drop=True)
    result["epoch_idx"] = np.arange(len(result))
    return result.merge(
        cross[["epoch_idx", "condition", "enc_overlap_frac"]],
        on="epoch_idx",
        how="inner",
    )


def friedman_posthoc(pivot: pd.DataFrame) -> tuple:
    avail = [c for c in COND_ORDER if c in pivot.columns]
    clean = pivot[avail].dropna()
    if len(avail) < 3 or len(clean) < 5:
        return np.nan, np.nan, pd.DataFrame()
    try:
        chi2, p = friedmanchisquare(*[clean[c].values for c in avail])
    except Exception:
        return np.nan, np.nan, pd.DataFrame()

    pairs = list(combinations(avail, 2))
    raw_ps = []
    for c1, c2 in pairs:
        d = clean[c1].values - clean[c2].values
        try:
            _, pw = (0, 1.0) if np.all(d == 0) else wilcoxon(clean[c1].values, clean[c2].values)
        except Exception:
            pw = np.nan
        raw_ps.append(pw)

    n = len(raw_ps)
    order = np.argsort(raw_ps)
    adj = np.array(raw_ps, dtype=float)
    for rank, orig in enumerate(order):
        adj[orig] = min(1.0, raw_ps[orig] * n / (rank + 1))
    for i in range(n - 2, -1, -1):
        adj[order[i]] = min(adj[order[i]], adj[order[i + 1]])

    rows = []
    for (c1, c2), rp, ap in zip(pairs, raw_ps, adj):
        rows.append({"cond1": c1, "cond2": c2,
                     "median_cond1": float(clean[c1].median()),
                     "median_cond2": float(clean[c2].median()),
                     "p_raw": rp, "p_adj": float(ap), "sig": ap < 0.05})
    return chi2, p, pd.DataFrame(rows)


def main():
    print("Loading features.parquet...")
    feat_all = pd.read_parquet(FEATURES_PARQUET)
    if "task" in feat_all.columns:
        feat_all = feat_all[feat_all["task"].str.contains("verbalwm", na=False)]
    print(f"  {len(feat_all)} epochs after verbalwm filter")

    available_proxies = [p for p in ARTIFACT_PROXIES if p in feat_all.columns]
    print(f"  Proxies: {available_proxies}")

    keep_cols = ["subject", "t_start"] + available_proxies
    feat_all = feat_all[[c for c in keep_cols if c in feat_all.columns]]

    all_rows = []
    for sub_int in sorted(feat_all["subject"].unique()):
        subject_bids = f"sub-{int(sub_int):03d}"
        sub_feat = feat_all[feat_all["subject"] == sub_int].copy()

        ev_files = list(BIDS_DIR.glob(
            f"{subject_bids}/eeg/{subject_bids}_task-verbalwm_events.tsv"
        ))
        if not ev_files:
            ev_files = list(BIDS_DIR.glob(f"{subject_bids}/**/*verbalwm*events.tsv"))
        if not ev_files:
            print(f"  {subject_bids}: no events.tsv")
            continue

        trials = reconstruct_trials(ev_files[0])
        assigned = assign_conditions_vectorized(sub_feat, trials)
        if assigned.empty:
            print(f"  {subject_bids}: no windows assigned")
            continue

        for cond in COND_ORDER:
            cond_rows = assigned[assigned["condition"] == cond]
            if cond_rows.empty:
                continue
            row = {"subject": subject_bids, "condition": cond, "n_windows": len(cond_rows)}
            for proxy in available_proxies:
                row[proxy] = float(cond_rows[proxy].median())
            all_rows.append(row)
        print(f"  {subject_bids}: {len(assigned)} windows assigned")

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("ERROR: No data collected.")
        return

    csv_path = OUT_DIR / "04_artifact_balance_by_condition.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWritten: {csv_path}")

    fr_rows, posthoc_all = [], []
    for proxy in available_proxies:
        pivot = df.pivot_table(index="subject", columns="condition", values=proxy)
        chi2, p, ph = friedman_posthoc(pivot)
        fr_rows.append({"proxy": proxy, "chi2": chi2, "p_friedman": p,
                        "n_subjects": int(pivot.dropna().shape[0])})
        if not ph.empty:
            ph.insert(0, "proxy", proxy)
            posthoc_all.append(ph)
    fr_df = pd.DataFrame(fr_rows)
    posthoc_df = pd.concat(posthoc_all, ignore_index=True) if posthoc_all else pd.DataFrame()
    cond_medians = df.groupby("condition")[available_proxies].median().round(4)

    today = date.today().strftime("%Y-%m-%d")
    any_sig = bool((fr_df["p_friedman"].dropna() < 0.05).any())

    md_lines = [
        "# Condition-Structured Artifact Audit -- ds006848 verbal WM",
        f"**Generated:** {today}  ",
        f"**Subjects:** {df['subject'].nunique()}  ",
        "",
        "## Background",
        "",
        "Fast/FastDelay: 7 digits sequentially at 400 ms/digit.",
        "Simultaneous: all 7 at once. Rapid sequential stimuli may produce",
        "condition-structured blink/muscle artifacts that contaminate MNPS m/d.",
        "",
        "## Condition medians per artifact proxy",
        "",
        cond_medians.T.to_markdown(),
        "",
        "## Friedman test (4 conditions)",
        "",
        "| proxy | chi2 | p_friedman | n_subjects | sig? |",
        "|-------|------|-----------|-----------|------|",
    ]
    for _, row in fr_df.iterrows():
        p = row["p_friedman"]
        sig = not np.isnan(p) and p < 0.05
        p_str = f"**{p:.4f}**" if sig else (f"{p:.4f}" if not np.isnan(p) else "N/A")
        chi_str = f"{row['chi2']:.2f}" if not np.isnan(row["chi2"]) else "N/A"
        md_lines.append(
            f"| {row['proxy']} | {chi_str} | {p_str} | {int(row['n_subjects'])} | {'YES' if sig else 'no'} |"
        )

    md_lines += ["", "## Pairwise contrasts (BH-FDR p < 0.05)", ""]
    if not posthoc_df.empty:
        sig_ph = posthoc_df[posthoc_df["p_adj"] < 0.05]
        md_lines.append(
            sig_ph[["proxy", "cond1", "cond2", "median_cond1", "median_cond2", "p_adj"]].to_markdown(index=False)
            if not sig_ph.empty else "None significant."
        )
    else:
        md_lines.append("No posthoc data.")

    md_lines += ["", "## Conclusion", ""]
    if any_sig:
        sig_proxies = fr_df[fr_df["p_friedman"] < 0.05]["proxy"].tolist()
        md_lines += [
            "**WARNING: Condition-structured artifact signal detected.**",
            "",
            f"Significant Friedman effects in: {', '.join(sig_proxies)}",
            "",
            "ICA cleaning (Fp1/Fp2 EOG proxy) is required before journal submission.",
            "Config ready in `config_ingest_ds006848.yaml` (set ica.enabled: true).",
        ]
    else:
        md_lines += [
            "**No significant condition-structured artifact differences detected.**",
            "",
            "Artifact proxies (frontal/temporal 30-45 Hz, Hjorth mobility) do not",
            "differ significantly across presentation modes (Friedman p > 0.05 for all).",
            "This provides evidence against condition-structured artifact contamination.",
            "",
            "Recommended Methods statement:",
            "> 'Artifact balance analysis confirmed that epoch-level proxies for",
            "> blink/eye artifacts (frontal 30-45 Hz, Hjorth mobility) and muscle",
            "> artifacts (temporal 30-45 Hz) did not differ significantly across",
            "> presentation modes (Friedman p > 0.05), ruling out condition-structured",
            "> artifact contamination as a primary confound.'",
        ]

    md_path = OUT_DIR / "04_artifact_balance_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Written: {md_path}")
    print("\nFriedman results:")
    print(fr_df.to_string(index=False))
    print(f"\nAny significant: {any_sig}")


if __name__ == "__main__":
    main()
