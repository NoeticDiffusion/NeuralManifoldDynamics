"""Full verified statistics package for ds003838.

Combines stage contrasts, Friedman tests, and coupling analysis into one
reproducible run. Reads the corrected block_native_windows from the
specified run directory.

Outputs (all in <out_dir>/)
-----------------------------
  stage_medians.csv              per-metric group median / IQR per stage
  friedman_results.csv           Friedman chi2 + p for every metric
  pairwise_contrasts.csv         all stage-pair Wilcoxon q + Cohen's d
  coupling_friedman.csv          Friedman for all 12 coupl_* columns
  coupling_stage_medians.csv     per-coupling group median per stage
  key_stats_summary.csv          one-page summary for article supplement

Usage
-----
  python project/scripts/run_stage_contrasts.py \\
      --run-dir J:/processed/openneuro/ds003838/neuralmanifolddynamics_ds003838_20260624_151505
"""
from __future__ import annotations
import argparse, logging
from itertools import combinations
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RUN_DIR = Path("J:/processed/openneuro/ds003838/neuralmanifolddynamics_ds003838_20260624_151505")
OUT_DIR = Path("J:/processed/openneuro/ds003838/audit_ecg_ds003838/stats_package")
STAGE_ORDER = ["rest", "listen", "mem5", "mem9", "mem13"]

PRIMARY_METRICS = [
    "vagal_index", "sympathetic_index", "vascular_index", "anchor_index",
    "ecg_hrv_rmssd_ms", "ecg_hrv_hr_mean_bpm", "ecg_hrv_pnn50", "ecg_hrv_sdnn_ms",
    "traj_path_length", "traj_mean_curvature", "traj_efficiency",
]
COUPL_METRICS = [
    "coupl_cntr_from_frnt", "coupl_cntr_from_par", "coupl_cntr_from_temp",
    "coupl_frnt_from_cntr", "coupl_frnt_from_par", "coupl_frnt_from_temp",
    "coupl_par_from_cntr", "coupl_par_from_frnt", "coupl_par_from_temp",
    "coupl_temp_from_cntr", "coupl_temp_from_frnt", "coupl_temp_from_par",
]


def load_block_native(run_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for sub_dir in sorted(run_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        pq = sub_dir / "block_native_windows.parquet"
        csv = sub_dir / "block_native_windows.csv"
        for f in [pq, csv]:
            if f.exists():
                try:
                    df = pd.read_parquet(f) if str(f).endswith(".parquet") else pd.read_csv(f)
                    if not df.empty:
                        frames.append(df); break
                except Exception:
                    continue
    if not frames:
        raise FileNotFoundError(f"No block_native_windows in {run_dir}")
    return pd.concat(frames, ignore_index=True)


def resolve_stage(df: pd.DataFrame) -> pd.DataFrame:
    if "stage" not in df.columns:
        col = next((c for c in ["task_state_label", "stage_code"] if c in df.columns), None)
        if col:
            df = df.rename(columns={col: "stage"})
    if "stage" in df.columns and df["stage"].dtype.kind in ("i", "f"):
        df["stage"] = df["stage"].map({0:"rest",1:"listen",5:"mem5",9:"mem9",13:"mem13"}).fillna(df["stage"].astype(str))
    return df


def get_sub_col(df: pd.DataFrame) -> str:
    return "subject_id" if "subject_id" in df.columns else "subject"


def sub_stage_medians(df: pd.DataFrame, metric: str, sub_col: str) -> pd.Series:
    """Per-subject, per-stage medians → index = (subject, stage)."""
    return df[df[metric].notna()].groupby([sub_col, "stage"])[metric].median()


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled = np.sqrt(((na-1)*np.std(a,ddof=1)**2 + (nb-1)*np.std(b,ddof=1)**2) / (na+nb-2))
    return float((np.mean(a)-np.mean(b))/pooled) if pooled > 1e-10 else np.nan


def friedman_row(df: pd.DataFrame, metric: str, stages: list[str], sub_col: str) -> dict:
    pivot = (df[df["stage"].isin(stages)]
             .groupby([sub_col,"stage"])[metric].median()
             .unstack("stage").reindex(columns=stages).dropna())
    if pivot.shape[0] < 5 or pivot.shape[1] < 2:
        return {"metric": metric, "chi2": np.nan, "p": np.nan, "n_subjects": 0}
    chi2, p = stats.friedmanchisquare(*[pivot[s].values for s in stages])
    return {"metric": metric, "chi2": float(chi2), "p": float(p), "n_subjects": int(pivot.shape[0])}


def compute_stage_medians(df: pd.DataFrame, metrics: list[str], sub_col: str) -> pd.DataFrame:
    rows = []
    valid_stages = [s for s in STAGE_ORDER if s in df["stage"].unique()]
    for metric in metrics:
        if metric not in df.columns:
            continue
        for stage in valid_stages:
            sdf = df[df["stage"] == stage]
            sub_meds = sdf.groupby(sub_col)[metric].median().dropna()
            if sub_meds.empty:
                continue
            rows.append({
                "metric": metric, "stage": stage,
                "n_subjects": len(sub_meds),
                "group_median": sub_meds.median(),
                "group_q25": sub_meds.quantile(0.25),
                "group_q75": sub_meds.quantile(0.75),
                "group_mean": sub_meds.mean(),
                "group_sd": sub_meds.std(ddof=1),
            })
    return pd.DataFrame(rows)


def compute_pairwise_contrasts(df: pd.DataFrame, metrics: list[str], sub_col: str) -> pd.DataFrame:
    valid_stages = [s for s in STAGE_ORDER if s in df["stage"].unique()]
    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        sub_meds = sub_stage_medians(df, metric, sub_col)
        for s1, s2 in combinations(valid_stages, 2):
            a = sub_meds.xs(s1, level="stage") if s1 in sub_meds.index.get_level_values("stage") else pd.Series(dtype=float)
            b = sub_meds.xs(s2, level="stage") if s2 in sub_meds.index.get_level_values("stage") else pd.Series(dtype=float)
            common = a.index.intersection(b.index)
            av, bv = a[common].values, b[common].values
            ok = np.isfinite(av) & np.isfinite(bv)
            av, bv = av[ok], bv[ok]
            if len(av) < 5:
                continue
            try:
                _, p = stats.wilcoxon(av, bv)
            except Exception:
                p = np.nan
            rows.append({
                "metric": metric,
                "stage_a": s1, "stage_b": s2,
                "median_a": float(np.median(av)), "median_b": float(np.median(bv)),
                "cohens_d": cohens_d(av, bv),
                "wilcoxon_p": float(p),
                "n_pairs": int(len(av)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading block_native_windows from %s ...", args.run_dir)
    df = load_block_native(args.run_dir)
    df = resolve_stage(df)
    sub_col = get_sub_col(df)
    n_subs = df[sub_col].nunique()
    valid_stages = [s for s in STAGE_ORDER if s in df["stage"].unique()]
    logger.info("Loaded %d windows, %d subjects, stages: %s", len(df), n_subs, valid_stages)

    # ── 1. Stage medians ─────────────────────────────────────────────────────
    logger.info("Computing stage medians ...")
    all_metrics = [m for m in PRIMARY_METRICS if m in df.columns]
    coupl_avail  = [m for m in COUPL_METRICS if m in df.columns]

    sm = compute_stage_medians(df, all_metrics + coupl_avail, sub_col)
    sm.to_csv(args.out_dir / "stage_medians.csv", index=False)
    logger.info("stage_medians.csv: %d rows", len(sm))

    # ── 2. Friedman for primary metrics ──────────────────────────────────────
    logger.info("Running Friedman tests ...")
    fried_rows = [friedman_row(df, m, valid_stages, sub_col) for m in all_metrics]
    fried_df = pd.DataFrame(fried_rows).sort_values("p")
    fried_df.to_csv(args.out_dir / "friedman_results.csv", index=False)

    print("\n=== Friedman tests (primary metrics) ===")
    for _, r in fried_df.iterrows():
        sig = "***" if r["p"] < 0.001 else ("**" if r["p"] < 0.01 else ("*" if r["p"] < 0.05 else ""))
        print(f"  {r['metric']:35s}  chi2={r['chi2']:7.2f}  p={r['p']:.2e}  n={int(r['n_subjects'])}  {sig}")

    # ── 3. Pairwise contrasts for primary metrics ────────────────────────────
    logger.info("Computing pairwise contrasts ...")
    pw = compute_pairwise_contrasts(df, all_metrics, sub_col)
    pw.to_csv(args.out_dir / "pairwise_contrasts.csv", index=False)
    logger.info("pairwise_contrasts.csv: %d rows", len(pw))

    # Print key C3 (listen vs mem*) rows
    c3 = pw[pw["stage_a"] == "listen"]
    if not c3.empty:
        print("\n=== C3: listen vs mem* (key rows) ===")
        for _, r in c3[c3["metric"].isin(["vagal_index","ecg_hrv_rmssd_ms","anchor_index"])].iterrows():
            print(f"  {r['metric']:30s}  {r['stage_a']} vs {r['stage_b']:5s}  "
                  f"d={r['cohens_d']:+.3f}  p={r['wilcoxon_p']:.2e}  n={int(r['n_pairs'])}")

    # ── 4. Coupling analysis ─────────────────────────────────────────────────
    if coupl_avail:
        logger.info("Running coupling analysis (%d metrics) ...", len(coupl_avail))
        fried_coupl = [friedman_row(df, m, valid_stages, sub_col) for m in coupl_avail]
        fried_coupl_df = pd.DataFrame(fried_coupl).sort_values("p")
        fried_coupl_df.to_csv(args.out_dir / "coupling_friedman.csv", index=False)

        coupl_sm = compute_stage_medians(df, coupl_avail, sub_col)
        coupl_sm.to_csv(args.out_dir / "coupling_stage_medians.csv", index=False)

        print("\n=== Friedman tests (coupling metrics) ===")
        for _, r in fried_coupl_df.iterrows():
            sig = "***" if r["p"] < 0.001 else ("**" if r["p"] < 0.01 else ("*" if r["p"] < 0.05 else "ns"))
            print(f"  {r['metric']:35s}  chi2={r['chi2']:7.2f}  p={r['p']:.2e}  {sig}")
    else:
        logger.warning("No coupling columns found — skipping coupling analysis")

    # ── 5. One-page key summary ───────────────────────────────────────────────
    key_rows = []
    for m in ["vagal_index", "ecg_hrv_rmssd_ms", "anchor_index", "traj_path_length"]:
        fr = next((r for r in fried_rows if r["metric"] == m), None)
        if fr is None:
            continue
        # C3 listen vs pooled mem
        sub_meds = sub_stage_medians(df, m, sub_col)
        listen_vals = sub_meds.xs("listen", level="stage") if "listen" in sub_meds.index.get_level_values("stage") else pd.Series(dtype=float)
        mem_stages = [s for s in ["mem5","mem9","mem13"] if s in sub_meds.index.get_level_values("stage")]
        if mem_stages:
            mem_vals = df[df["stage"].isin(mem_stages)].groupby(sub_col)[m].median().dropna()
        else:
            mem_vals = pd.Series(dtype=float)
        common = listen_vals.index.intersection(mem_vals.index)
        av, bv = listen_vals[common].values, mem_vals[common].values
        ok = np.isfinite(av) & np.isfinite(bv)
        av, bv = av[ok], bv[ok]
        try:
            _, c3_p = stats.wilcoxon(av, bv) if len(av) >= 5 else (np.nan, np.nan)
        except Exception:
            c3_p = np.nan
        c3_d = cohens_d(av, bv)
        key_rows.append({
            "metric": m,
            "friedman_chi2": fr["chi2"],
            "friedman_p": fr["p"],
            "n_subjects": fr["n_subjects"],
            "c3_listen_vs_mem_cohens_d": c3_d,
            "c3_listen_vs_mem_wilcoxon_p": c3_p,
            "listen_median": float(listen_vals.median()) if len(listen_vals) > 0 else np.nan,
            "mem_pooled_median": float(mem_vals.median()) if len(mem_vals) > 0 else np.nan,
        })
    key_df = pd.DataFrame(key_rows)
    key_df.to_csv(args.out_dir / "key_stats_summary.csv", index=False)

    print("\n=== Key stats summary ===")
    print(key_df.to_string(index=False))

    logger.info("Done. All outputs in %s", args.out_dir)


if __name__ == "__main__":
    main()
