"""04c — Window-overlap purity audit for the 04b encoding-phase findings.

Motivation
----------
04b uses 8 s MNPS windows on encoding phases of 2.8–7.0 s.  Many "encoding"
windows may simultaneously overlap baseline, retention, or retrieval signal.
This audit quantifies that contamination and tests whether the
Fast/FastDelay > Simultaneous/Slow m/d finding survives stricter purity gates.

For every window assigned to a trial's encoding interval the script computes:

    enc_overlap_frac    fraction of window inside encoding
    baseline_overlap    fraction inside Baseline_2s (2 s before encoding)
    retention_overlap   fraction inside retention (6 s after encoding)
    retrieval_overlap   fraction inside retrieval (~13 s after retention)
    other_overlap       remainder
    win_ctr_re_enc      window centre relative to encoding onset  (s)
    win_ctr_re_ret      window centre relative to retention onset (s)

Tests A and B from 04b are then rerun under four filter conditions:

    F0  any overlap ≥ 25 %          (04b baseline)
    F1  window centre inside encoding
    F2  encoding overlap ≥ 50 %
    F3  encoding overlap ≥ 70 %
    F4  overlap-weighted average     (weight = enc_overlap_frac)

If the m/d pattern survives F2/F3 it may be upgraded to an internally
validated encoding result.  If it disappears or weakens the current finding
should be interpreted as transition-window geometry.

Usage
-----
  python project/scripts/04c_window_overlap_purity_audit.py \\
      --run-dir J:/processed/openneuro/ds006848/neuralmanifolddynamics_ds006848_20260626_114620 \\
      --bids-dir K:/ExternalReceivedDatasets/openneuro/received/ds006848 \\
      --out-dir J:/processed/openneuro/ds006848/04c_purity_audit
"""
from __future__ import annotations

import argparse
import logging
import warnings
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*copy.*keyword.*", category=DeprecationWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
CONDITIONS = ["Simultaneous", "Fast", "FastDelay", "Slow"]
CONDITION_MAP = {
    "Retention_Simultaneous": "Simultaneous",
    "Retention_Fast": "Fast",
    "Retention_FastDelay": "FastDelay",
    "Retention_Slow": "Slow",
}
METRICS = ["m", "d", "e", "m_dot", "d_dot", "e_dot"]
COMMON_DURATION_S = 2.8
BASELINE_DURATION_S = 2.0
RETENTION_DURATION_S = 6.0
RETRIEVAL_DURATION_S = 13.0

# Filter labels and their minimum encoding-overlap fraction requirement
# (F1 uses centre-in-encoding logic instead)
FILTERS = {
    "F0_any25": {"min_enc_frac": 0.25, "centre_in": False},
    "F1_centre_in": {"min_enc_frac": 0.0, "centre_in": True},
    "F2_enc50": {"min_enc_frac": 0.50, "centre_in": False},
    "F3_enc70": {"min_enc_frac": 0.70, "centre_in": False},
}
# F4 (overlap-weighted) handled separately


# ---------------------------------------------------------------------------
# Shared utilities (inlined to keep 04c self-contained)
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
        baseline_start = enc_start - BASELINE_DURATION_S
        baseline_end = enc_start
        ret_end = ret_start + RETENTION_DURATION_S

        later = events_df[
            (events_df["onset"] > ret_end)
            & (events_df["trial_type"] == "Digits_Retrieval")
        ]
        if later.empty:
            continue
        retr_start = later.iloc[0]["onset"]
        retr_end = retr_start + RETRIEVAL_DURATION_S

        trials.append(
            {
                "trial_id": len(trials),
                "condition": condition,
                "baseline_start": baseline_start,
                "baseline_end": baseline_end,
                "enc_start": enc_start,
                "enc_end": enc_end,
                "enc_dur": enc_end - enc_start,
                "ret_start": ret_start,
                "ret_end": ret_end,
                "retr_start": retr_start,
                "retr_end": retr_end,
            }
        )
    return pd.DataFrame(trials)


def load_subject_mnps(h5_path: Path) -> pd.DataFrame:
    with h5py.File(h5_path, "r") as f:
        t_start = np.array(f["window_start"], dtype=np.float64)
        t_end = np.array(f["window_end"], dtype=np.float64)
        mnps = np.array(f["mnps_3d"], dtype=np.float64)
        mnps_dot = np.array(f["mnps_3d_dot"], dtype=np.float64)
    df = pd.DataFrame(
        {
            "win_id": np.arange(len(t_start)),
            "t_start": t_start,
            "t_end": t_end,
            "m": mnps[:, 0],
            "d": mnps[:, 1],
            "e": mnps[:, 2],
            "m_dot": mnps_dot[:, 0],
            "d_dot": mnps_dot[:, 1],
            "e_dot": mnps_dot[:, 2],
        }
    )
    return df[np.isfinite(df["m"])].copy()


def _overlap_frac(w_start: float, w_end: float, iv_start: float, iv_end: float) -> float:
    win_dur = w_end - w_start
    if win_dur <= 0:
        return 0.0
    ovlp = max(0.0, min(w_end, iv_end) - max(w_start, iv_start))
    return ovlp / win_dur


# ---------------------------------------------------------------------------
# Purity computation
# ---------------------------------------------------------------------------


def compute_purity_table(
    windows: pd.DataFrame,
    trials: pd.DataFrame,
    min_any_enc_overlap: float = 0.01,
) -> pd.DataFrame:
    """Return one row per (window, trial) with purity metrics.

    Windows are included if their encoding overlap fraction >= min_any_enc_overlap.
    """
    rows: List[Dict] = []
    t_start_arr = windows["t_start"].values
    t_end_arr = windows["t_end"].values
    win_ctr_arr = (t_start_arr + t_end_arr) / 2.0

    for _, trial in trials.iterrows():
        enc_start = trial["enc_start"]
        enc_end = trial["enc_end"]
        enc_dur = trial["enc_dur"]
        if enc_dur <= 0:
            continue

        for i, (ws, we, wc) in enumerate(zip(t_start_arr, t_end_arr, win_ctr_arr)):
            enc_frac = _overlap_frac(ws, we, enc_start, enc_end)
            if enc_frac < min_any_enc_overlap:
                continue

            bl_frac = _overlap_frac(ws, we, trial["baseline_start"], trial["baseline_end"])
            ret_frac = _overlap_frac(ws, we, trial["ret_start"], trial["ret_end"])
            retr_frac = _overlap_frac(ws, we, trial["retr_start"], trial["retr_end"])
            # "other" = any fraction not accounted for above (inter-trial gaps, etc.)
            other_frac = max(0.0, 1.0 - enc_frac - bl_frac - ret_frac - retr_frac)

            row = {
                "trial_id": trial["trial_id"],
                "condition": trial["condition"],
                "win_id": windows.iloc[i]["win_id"],
                "t_start": ws,
                "t_end": we,
                "enc_overlap_frac": round(enc_frac, 4),
                "baseline_overlap_frac": round(bl_frac, 4),
                "retention_overlap_frac": round(ret_frac, 4),
                "retrieval_overlap_frac": round(retr_frac, 4),
                "other_overlap_frac": round(other_frac, 4),
                "win_ctr_re_enc_onset": round(wc - enc_start, 3),
                "win_ctr_re_ret_onset": round(wc - trial["ret_start"], 3),
                "centre_in_encoding": enc_start <= wc < enc_end,
            }
            for m in METRICS:
                row[m] = windows.iloc[i][m]
            rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Statistics (copied from 04b for self-containment)
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
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else np.nan


def friedman_posthoc(
    sub_cond: pd.DataFrame, metric: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pivot = (
        sub_cond[sub_cond["condition"].isin(CONDITIONS)]
        .pivot_table(index="subject", columns="condition", values=metric, aggfunc="median")
        .dropna()
    )
    n = len(pivot)
    avail = [c for c in CONDITIONS if c in pivot.columns]
    if n < 3:
        return pd.DataFrame(), pd.DataFrame()

    fr = pd.DataFrame()
    if len(avail) >= 3:
        chi2, p = stats.friedmanchisquare(*[pivot[c].values for c in avail])
        fr = pd.DataFrame([{"metric": metric, "chi2": round(chi2, 3),
                            "dof": len(avail) - 1, "p_friedman": p, "n_subjects": n}])
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
        pairs.append({
            "condition_a": ca, "condition_b": cb,
            "median_a": round(np.median(paired[:, 0]), 4),
            "median_b": round(np.median(paired[:, 1]), 4),
            "W": stat, "p": pval,
            "cohens_d": round(cohens_d(paired[:, 0], paired[:, 1]), 3),
        })
    if not pairs:
        return fr, pd.DataFrame()
    pw = pd.DataFrame(pairs)
    pw["q"] = bh_fdr(pw["p"].values)
    pw["sig"] = pw["q"] < 0.05
    pw.insert(0, "metric", metric)
    return fr, pw


# ---------------------------------------------------------------------------
# Apply a filter to the purity table and compute per-subject × cond medians
# ---------------------------------------------------------------------------


def apply_filter(
    purity: pd.DataFrame,
    filter_def: Dict,
    test_end_col: str,     # "enc_end" or "enc28_end" (Test B)
    trials: pd.DataFrame,  # needed for Test B enc28_end
    subject: str,
) -> pd.DataFrame:
    """Filter purity rows and return per-subject × condition medians."""
    sel = purity.copy()

    if filter_def["centre_in"]:
        sel = sel[sel["centre_in_encoding"]]
    else:
        sel = sel[sel["enc_overlap_frac"] >= filter_def["min_enc_frac"]]

    # For Test B restrict to first 2.8 s of encoding
    if test_end_col == "enc28_end":
        # Re-evaluate: window centre must be < enc_start + 2.8 s
        # We stored win_ctr_re_enc_onset, so ctr_re_enc < 2.8
        sel = sel[sel["win_ctr_re_enc_onset"] < COMMON_DURATION_S]

    if sel.empty:
        return pd.DataFrame()

    rows = []
    for cond, grp in sel.groupby("condition"):
        # Per-trial medians first, then per-subject median
        trial_meds = grp.groupby("trial_id")[METRICS].median().reset_index()
        r: Dict = {"subject": subject, "condition": cond, "n_trials": len(trial_meds)}
        for m in METRICS:
            r[m] = trial_meds[m].median()
        rows.append(r)
    return pd.DataFrame(rows)


def apply_weighted_filter(
    purity: pd.DataFrame,
    test_end_col: str,
    subject: str,
) -> pd.DataFrame:
    """F4: overlap-weighted average of MNPS per trial, then median across trials."""
    sel = purity[purity["enc_overlap_frac"] > 0].copy()
    if test_end_col == "enc28_end":
        sel = sel[sel["win_ctr_re_enc_onset"] < COMMON_DURATION_S]
    if sel.empty:
        return pd.DataFrame()

    rows = []
    for cond, cgrp in sel.groupby("condition"):
        trial_rows = []
        for tid, tgrp in cgrp.groupby("trial_id"):
            weights = tgrp["enc_overlap_frac"].values
            w_sum = weights.sum()
            if w_sum == 0:
                continue
            r: Dict = {"trial_id": tid}
            for m in METRICS:
                r[m] = np.average(tgrp[m].values, weights=weights)
            trial_rows.append(r)
        if not trial_rows:
            continue
        tmdf = pd.DataFrame(trial_rows)
        r2: Dict = {"subject": subject, "condition": cond, "n_trials": len(tmdf)}
        for m in METRICS:
            r2[m] = tmdf[m].median()
        rows.append(r2)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_audit(
    run_dir: Path,
    bids_dir: Path,
    out_dir: Path,
    subjects: Optional[List[str]] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect per-subject purity tables and per-filter summaries
    all_purity: List[pd.DataFrame] = []

    # dict[filter_name][test_name] -> list of per-subject summary DataFrames
    filter_summaries: Dict[str, Dict[str, List[pd.DataFrame]]] = {
        fname: {"A": [], "B": []} for fname in list(FILTERS.keys()) + ["F4_weighted"]
    }

    for sub_dir in sorted(run_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        h5_files = list(sub_dir.glob("*verbal*.h5"))
        if not h5_files:
            continue
        subject = sub_dir.name.replace("_verbal_wm", "").replace("_verbalwm", "")
        if subjects is not None and subject not in subjects:
            continue

        events_file = bids_dir / subject / "eeg" / f"{subject}_task-verbalwm_events.tsv"
        if not events_file.exists():
            logger.warning("No events.tsv for %s", subject)
            continue

        try:
            events_df = pd.read_csv(events_file, sep="\t")
            trials = reconstruct_trials(events_df)
            windows = load_subject_mnps(h5_files[0])
        except Exception as exc:
            logger.error("Load failed %s: %s", subject, exc)
            continue

        if trials.empty:
            continue

        # Add enc28_end column for Test B
        trials = trials.copy()
        trials["enc28_end"] = trials["enc_start"] + COMMON_DURATION_S

        # Compute purity table (any 1% enc overlap to cast a wide net)
        purity = compute_purity_table(windows, trials, min_any_enc_overlap=0.01)
        if purity.empty:
            logger.warning("%s: empty purity table", subject)
            continue

        purity["subject"] = subject
        all_purity.append(purity)

        # Apply filters for Test A and B
        for fname, fdef in FILTERS.items():
            for test, ecol in [("A", "enc_end"), ("B", "enc28_end")]:
                summary = apply_filter(purity, fdef, ecol, trials, subject)
                if not summary.empty:
                    filter_summaries[fname][test].append(summary)

        # F4 weighted
        for test, ecol in [("A", "enc_end"), ("B", "enc28_end")]:
            summary = apply_weighted_filter(purity, ecol, subject)
            if not summary.empty:
                filter_summaries["F4_weighted"][test].append(summary)

        logger.info("  %s: %d purity rows", subject, len(purity))

    if not all_purity:
        logger.error("No data collected — check run_dir and bids_dir paths")
        return

    # ------------------------------------------------------------------
    # Save full purity table
    purity_df = pd.concat(all_purity, ignore_index=True)
    purity_df.to_parquet(out_dir / "window_purity_table.parquet", index=False)
    purity_df.to_csv(out_dir / "window_purity_table.csv", index=False)
    logger.info("Purity table: %d rows → %s", len(purity_df), out_dir / "window_purity_table.csv")

    # ------------------------------------------------------------------
    # Overlap distribution per condition
    dist_rows = []
    for cond, grp in purity_df.groupby("condition"):
        for frac_col in ["enc_overlap_frac", "baseline_overlap_frac",
                         "retention_overlap_frac", "retrieval_overlap_frac", "other_overlap_frac"]:
            dist_rows.append({
                "condition": cond,
                "phase": frac_col.replace("_overlap_frac", "").replace("_", " "),
                "mean": round(grp[frac_col].mean(), 3),
                "median": round(grp[frac_col].median(), 3),
                "p25": round(grp[frac_col].quantile(0.25), 3),
                "p75": round(grp[frac_col].quantile(0.75), 3),
                "pct_gt50": round((grp[frac_col] >= 0.50).mean() * 100, 1),
                "pct_gt70": round((grp[frac_col] >= 0.70).mean() * 100, 1),
            })
    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(out_dir / "overlap_distribution.csv", index=False)

    # ------------------------------------------------------------------
    # Per-filter condition medians comparison table
    comparison_rows = []

    for fname in list(FILTERS.keys()) + ["F4_weighted"]:
        for test in ["A", "B"]:
            frames = filter_summaries[fname][test]
            if not frames:
                continue
            sub_cond = pd.concat(frames, ignore_index=True)
            n_sub = sub_cond["subject"].nunique()
            for metric in METRICS:
                cond_meds = sub_cond.groupby("condition")[metric].median()
                row: Dict = {"filter": fname, "test": test, "metric": metric, "n_subjects": n_sub}
                for c in CONDITIONS:
                    row[c] = round(cond_meds.get(c, np.nan), 4)
                comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(out_dir / "filter_comparison.csv", index=False)

    # Save Friedman and pairwise separately
    fr_rows: List[pd.DataFrame] = []
    pw_rows: List[pd.DataFrame] = []
    for fname in list(FILTERS.keys()) + ["F4_weighted"]:
        for test in ["A", "B"]:
            frames = filter_summaries[fname][test]
            if not frames:
                continue
            sub_cond = pd.concat(frames, ignore_index=True)
            for metric in METRICS:
                fr, pw = friedman_posthoc(sub_cond, metric)
                if not fr.empty:
                    fr.insert(0, "filter", fname)
                    fr.insert(1, "test", test)
                    fr_rows.append(fr)
                if not pw.empty:
                    pw.insert(0, "filter", fname)
                    pw.insert(1, "test", test)
                    pw_rows.append(pw)

    if fr_rows:
        pd.concat(fr_rows, ignore_index=True).to_csv(out_dir / "friedman_by_filter.csv", index=False)
    if pw_rows:
        pd.concat(pw_rows, ignore_index=True).to_csv(out_dir / "pairwise_by_filter.csv", index=False)

    _write_audit_summary(purity_df, dist_df, comparison_df, fr_rows, out_dir)
    logger.info("04c audit complete → %s", out_dir)


def _write_audit_summary(
    purity_df: pd.DataFrame,
    dist_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    friedman_frames: List[pd.DataFrame],
    out_dir: Path,
) -> None:
    lines: List[str] = ["=== 04c Window-Overlap Purity Audit ===\n"]

    n_windows = len(purity_df)
    n_subjects = purity_df["subject"].nunique()
    lines.append(f"Total (window × trial) pairs: {n_windows}  across {n_subjects} subjects\n")
    lines.append(f"Window size: 8 s, step: 4 s\n")

    # Overlap summary per condition
    lines.append("\n--- Encoding overlap fraction by condition ---\n")
    enc_dist = dist_df[dist_df["phase"] == "encoding"][
        ["condition", "mean", "median", "pct_gt50", "pct_gt70"]
    ]
    lines.append(enc_dist.to_string(index=False))

    lines.append("\n\n--- Contamination summary (mean overlap fraction) ---\n")
    phases = ["encoding", "baseline", "retention", "retrieval", "other"]
    ctm = dist_df[dist_df["phase"].isin(phases)].pivot_table(
        index="condition", columns="phase", values="mean"
    )
    lines.append(ctm.round(3).to_string())

    # Filter comparison for m and d, Test A
    lines.append("\n\n--- Condition medians by filter — Test A, m ---\n")
    cm = comparison_df[(comparison_df["test"] == "A") & (comparison_df["metric"] == "m")]
    if not cm.empty:
        lines.append(cm[["filter", "n_subjects"] + CONDITIONS].to_string(index=False))

    lines.append("\n\n--- Condition medians by filter — Test A, d ---\n")
    cd = comparison_df[(comparison_df["test"] == "A") & (comparison_df["metric"] == "d")]
    if not cd.empty:
        lines.append(cd[["filter", "n_subjects"] + CONDITIONS].to_string(index=False))

    # Friedman p-values across filters for m / d
    if friedman_frames:
        fr_all = pd.concat(friedman_frames, ignore_index=True)
        lines.append("\n\n--- Friedman p-values across filters (Test A) ---\n")
        fr_a = fr_all[(fr_all["test"] == "A") & (fr_all["metric"].isin(["m", "d"]))]
        if not fr_a.empty:
            lines.append(fr_a[["filter", "metric", "chi2", "p_friedman", "n_subjects"]].to_string(index=False))

    # Summary verdict
    lines.append("\n\n--- Verdict ---\n")
    lines.append(
        "Pattern is 'surviving' a filter if Fast/FastDelay median > Simultaneous/Slow\n"
        "for both m and d under that filter.\n"
    )
    for fname in list(FILTERS.keys()) + ["F4_weighted"]:
        m_row = comparison_df[
            (comparison_df["filter"] == fname)
            & (comparison_df["test"] == "A")
            & (comparison_df["metric"] == "m")
        ]
        if m_row.empty:
            lines.append(f"  {fname}: no data\n")
            continue
        r = m_row.iloc[0]
        fast_high = (r.get("Fast", np.nan) > r.get("Simultaneous", np.nan)) and (
            r.get("FastDelay", np.nan) > r.get("Simultaneous", np.nan)
        )
        slow_low = r.get("Slow", np.nan) < r.get("Simultaneous", np.nan)
        verdict = "SURVIVES" if (fast_high and slow_low) else "WEAKENS/FAILS"
        lines.append(
            f"  {fname}: m Fast={r.get('Fast','?'):.3f} FastDelay={r.get('FastDelay','?'):.3f} "
            f"Simult={r.get('Simultaneous','?'):.3f} Slow={r.get('Slow','?'):.3f}  → {verdict}\n"
        )

    (out_dir / "audit_summary.txt").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "J:/processed/openneuro/ds006848/neuralmanifolddynamics_ds006848_20260626_114620"
        ),
    )
    ap.add_argument(
        "--bids-dir",
        type=Path,
        default=Path("K:/ExternalReceivedDatasets/openneuro/received/ds006848"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("J:/processed/openneuro/ds006848/04c_purity_audit"),
    )
    ap.add_argument("--subjects", nargs="*", default=None)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir
    if not run_dir.exists():
        parent = run_dir.parent
        dirs = sorted(parent.glob("neuralmanifolddynamics_*"))
        if dirs:
            run_dir = dirs[-1]
            logger.warning("run-dir not found; using: %s", run_dir)
        else:
            raise FileNotFoundError(f"No run dirs under {parent}")

    run_audit(
        run_dir=run_dir,
        bids_dir=args.bids_dir,
        out_dir=args.out_dir,
        subjects=args.subjects,
    )


if __name__ == "__main__":
    main()
