"""04e — Subject-level robustness for ds006848 encoding-phase m/d finding.

Analyses
--------
  1. Leave-one-out (LOO) Friedman: remove each subject in turn; count how
     many LOO runs preserve significance (p < 0.05) for m and d.
  2. Bootstrap confidence intervals: 10 000 resample-subjects iterations,
     compute per-condition medians; report 95 % CI for m and d.
  3. F1 and F4 pairwise effect sizes: Cohen's d matrix for m and d under
     both applicable purity filters.
  4. Subject-level rank consistency: for each subject, rank the four
     conditions by m and d; report how often Fast/FastDelay ranks above
     Simultaneous/Slow.

Outputs
-------
  loo_friedman.csv           LOO p-values per subject removed
  bootstrap_ci.csv           95% CIs per condition
  pairwise_effect_sizes.csv  Cohen's d per filter (F0/F1/F4)
  rank_consistency.csv       per-subject condition rank
  summary.txt

Usage
-----
  python project/scripts/04e_subject_robustness.py \\
      --04b-dir J:/processed/openneuro/ds006848/04b_encoding_phase \\
      --04c-dir J:/processed/openneuro/ds006848/04c_purity_audit \\
      --out-dir J:/processed/openneuro/ds006848/04e_robustness
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
N_BOOTSTRAP = 10_000
RNG_SEED = 42


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


def load_pivot(csv_path: Path, metric: str) -> pd.DataFrame:
    """Load subject × condition summary and pivot for a given metric."""
    df = pd.read_csv(csv_path)
    pivot = (
        df[df["condition"].isin(CONDITIONS)]
        .pivot_table(index="subject", columns="condition", values=metric, aggfunc="median")
        .dropna()
    )
    return pivot


def friedman_p(pivot: pd.DataFrame) -> Optional[float]:
    avail = [c for c in CONDITIONS if c in pivot.columns]
    if len(avail) < 3 or len(pivot) < 3:
        return None
    try:
        _, p = stats.friedmanchisquare(*[pivot[c].values for c in avail])
        return p
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 1. LOO Friedman
# ---------------------------------------------------------------------------

def loo_friedman(pivot: pd.DataFrame, metric: str) -> pd.DataFrame:
    subjects = pivot.index.tolist()
    rows: List[Dict] = []
    full_p = friedman_p(pivot)
    for subj in subjects:
        loo = pivot.drop(index=subj)
        p = friedman_p(loo)
        rows.append({"subject_removed": subj, "metric": metric,
                     "n_remaining": len(loo), "p_friedman": p,
                     "sig_05": (p < 0.05) if p is not None else None})
    result = pd.DataFrame(rows)
    n_sig = result["sig_05"].sum()
    logger.info("LOO %s: full p=%.4f, %d/%d LOO runs significant",
                metric, full_p or np.nan, n_sig, len(subjects))
    return result


# ---------------------------------------------------------------------------
# 2. Bootstrap CIs
# ---------------------------------------------------------------------------

def bootstrap_ci(pivot: pd.DataFrame, metric: str,
                 n_boot: int = N_BOOTSTRAP, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(pivot)
    avail = [c for c in CONDITIONS if c in pivot.columns]
    boot_meds = {c: np.empty(n_boot) for c in avail}

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = pivot.iloc[idx]
        for c in avail:
            boot_meds[c][i] = np.median(sample[c].values)

    rows: List[Dict] = []
    for c in avail:
        bm = boot_meds[c]
        rows.append({
            "condition": c, "metric": metric,
            "median": round(np.median(pivot[c].values), 4),
            "boot_mean": round(np.mean(bm), 4),
            "ci_lo_95": round(np.percentile(bm, 2.5), 4),
            "ci_hi_95": round(np.percentile(bm, 97.5), 4),
            "ci_lo_90": round(np.percentile(bm, 5.0), 4),
            "ci_hi_90": round(np.percentile(bm, 95.0), 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Pairwise effect sizes under each filter
# ---------------------------------------------------------------------------

def pairwise_effect_sizes(pivot: pd.DataFrame, metric: str, filter_name: str) -> pd.DataFrame:
    avail = [c for c in CONDITIONS if c in pivot.columns]
    rows: List[Dict] = []
    for ca, cb in combinations(avail, 2):
        paired = np.column_stack([pivot[ca].values, pivot[cb].values])
        paired = paired[np.isfinite(paired).all(axis=1)]
        if len(paired) < 3:
            continue
        try:
            _, pval = wilcoxon(paired[:, 0], paired[:, 1])
        except ValueError:
            pval = np.nan
        q = bh_fdr(np.array([pval]))[0]
        rows.append({
            "filter": filter_name, "metric": metric,
            "condition_a": ca, "condition_b": cb,
            "median_a": round(np.median(paired[:, 0]), 4),
            "median_b": round(np.median(paired[:, 1]), 4),
            "cohens_d": round(cohens_d(paired[:, 0], paired[:, 1]), 3),
            "p": round(pval, 5), "q": round(q, 5),
            "sig": q < 0.05,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Subject-level rank consistency
# ---------------------------------------------------------------------------

def rank_consistency(pivot: pd.DataFrame, metric: str) -> pd.DataFrame:
    avail = [c for c in CONDITIONS if c in pivot.columns]
    rows: List[Dict] = []
    for subj, row in pivot[avail].iterrows():
        ranked = row.rank(ascending=False)  # rank 1 = highest
        r: Dict = {"subject": subj, "metric": metric}
        for c in avail:
            r[f"rank_{c}"] = int(ranked[c])
        # Indicator: Fast > Simultaneous AND FastDelay > Simultaneous
        r["fast_above_simult"] = (
            (row.get("Fast", np.nan) > row.get("Simultaneous", np.nan))
            if "Fast" in row.index and "Simultaneous" in row.index else None
        )
        r["fastdelay_above_simult"] = (
            (row.get("FastDelay", np.nan) > row.get("Simultaneous", np.nan))
            if "FastDelay" in row.index and "Simultaneous" in row.index else None
        )
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(dir_04b: Path, dir_04c: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    medians_a = dir_04b / "A_full_encoding" / "subject_condition_medians.csv"
    if not medians_a.exists():
        raise FileNotFoundError(f"04b output not found: {medians_a}")

    # Load F0 (04b baseline) and F1/F4 comparison tables from 04c
    filter_comp = dir_04c / "filter_comparison.csv"
    has_04c = filter_comp.exists()

    loo_rows, boot_rows, es_rows, rank_rows = [], [], [], []
    summary_parts: List[str] = ["=== 04e Subject Robustness ===\n"]

    for metric in ["m", "d"]:
        pivot_f0 = load_pivot(medians_a, metric)
        n_subjects = len(pivot_f0)
        summary_parts.append(f"\nMetric: {metric}  N={n_subjects}\n")

        # LOO
        loo = loo_friedman(pivot_f0, metric)
        loo_rows.append(loo)
        n_loo_sig = loo["sig_05"].sum()
        summary_parts.append(
            f"  LOO: {n_loo_sig}/{n_subjects} runs remain p<0.05\n"
        )

        # Bootstrap
        boot = bootstrap_ci(pivot_f0, metric)
        boot_rows.append(boot)
        summary_parts.append("  Bootstrap 95% CIs:\n")
        for _, br in boot.iterrows():
            summary_parts.append(
                f"    {br['condition']:15s}: {br['median']:.3f}  "
                f"[{br['ci_lo_95']:.3f}, {br['ci_hi_95']:.3f}]\n"
            )

        # Effect sizes: F0
        es_f0 = pairwise_effect_sizes(pivot_f0, metric, "F0_any25")
        es_rows.append(es_f0)

        # Effect sizes: F1 and F4 from 04c comparison table
        if has_04c:
            comp = pd.read_csv(filter_comp)
            for fname in ["F1_centre_in", "F4_weighted"]:
                comp_f = comp[(comp["filter"] == fname) & (comp["test"] == "A") & (comp["metric"] == metric)]
                if comp_f.empty:
                    continue
                # Reconstruct pivot from comparison table
                # (We only have condition medians per filter, not per-subject data)
                # Use the pairwise_by_filter.csv if available
                pairwise_f = dir_04c / "pairwise_by_filter.csv"
                if pairwise_f.exists():
                    pw = pd.read_csv(pairwise_f)
                    pw_sel = pw[(pw["filter"] == fname) & (pw["test"] == "A") & (pw["metric"] == metric)]
                    if not pw_sel.empty:
                        es_rows.append(pw_sel[["filter", "metric", "condition_a", "condition_b",
                                               "median_a", "median_b", "cohens_d", "p", "q", "sig"]])

        # Rank consistency
        rank_df = rank_consistency(pivot_f0, metric)
        rank_rows.append(rank_df)
        n_fast = rank_df["fast_above_simult"].sum()
        n_fd = rank_df["fastdelay_above_simult"].sum()
        summary_parts.append(
            f"  Rank: Fast>Simult in {n_fast}/{n_subjects} subjects, "
            f"FastDelay>Simult in {n_fd}/{n_subjects}\n"
        )

    # Save
    if loo_rows:
        pd.concat(loo_rows, ignore_index=True).to_csv(out_dir / "loo_friedman.csv", index=False)
    if boot_rows:
        pd.concat(boot_rows, ignore_index=True).to_csv(out_dir / "bootstrap_ci.csv", index=False)
    if es_rows:
        pd.concat(es_rows, ignore_index=True).to_csv(out_dir / "pairwise_effect_sizes.csv", index=False)
    if rank_rows:
        pd.concat(rank_rows, ignore_index=True).to_csv(out_dir / "rank_consistency.csv", index=False)

    (out_dir / "summary.txt").write_text("".join(summary_parts), encoding="utf-8")
    logger.info("04e complete → %s", out_dir)


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--04b-dir", type=Path,
                    default=Path("J:/processed/openneuro/ds006848/04b_encoding_phase"))
    ap.add_argument("--04c-dir", type=Path,
                    default=Path("J:/processed/openneuro/ds006848/04c_purity_audit"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("J:/processed/openneuro/ds006848/04e_robustness"))
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    dir_04b = getattr(args, "04b_dir")
    dir_04c = getattr(args, "04c_dir")
    run(dir_04b, dir_04c, args.out_dir)
