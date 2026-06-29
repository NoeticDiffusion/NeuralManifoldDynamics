"""Post-correction HRV analysis for ds003838 (Request 1 deliverables).

Compares corrected vs original HRV values and runs the key statistical tests
required by the peer reviewer.

Requires:
  features_pre_ecg_patch.parquet   (original, bad detector)
  features.parquet                 (corrected, NK2 detector)
  block_native_windows.parquet (from summarize run after patch)

Outputs
-------
  audit_ecg_ds003838/
    corrected_hrv_group_medians.csv     per-stage group median HR/RMSSD/pNN50
    corrected_hrv_stats.csv             Friedman + C3 Wilcoxon + Cohen's d
    bland_altman_rmssd.png              Bland-Altman old vs corrected RMSSD
    bland_altman_rmssd.csv              data behind the plot

Usage
-----
  python project/scripts/hrv_corrected_analysis_ds003838.py \\
      --run-dir "J:/processed/openneuro/ds003838/neuralmanifolddynamics_RRR"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURES_NEW = Path("J:/processed/openneuro/ds003838/features.parquet")
FEATURES_OLD = Path("J:/processed/openneuro/ds003838/features_pre_ecg_patch.parquet")
PROCESSED_BASE = Path("J:/processed/openneuro/ds003838")
OUT_DIR = Path("J:/processed/openneuro/ds003838/audit_ecg_ds003838")

STAGE_ORDER = ["rest", "listen", "mem5", "mem9", "mem13"]
LISTEN_LABEL = "listen"
MEM_LABELS = ["mem5", "mem9", "mem13"]


def load_block_native(run_dir: Path) -> pd.DataFrame:
    rows = []
    for sub_dir in sorted(run_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        for f in ["block_native_windows.parquet", "block_native_windows.csv"]:
            p = sub_dir / f
            if p.exists():
                try:
                    df = pd.read_parquet(p) if f.endswith(".parquet") else pd.read_csv(p)
                    if not df.empty:
                        rows.append(df)
                    break
                except Exception:
                    continue
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def find_latest_run_dir(base: Path) -> Optional[Path]:
    dirs = sorted(base.glob("neuralmanifolddynamics_*"), reverse=True)
    return dirs[0] if dirs else None


def resolve_stage(df: pd.DataFrame) -> pd.DataFrame:
    if "stage" not in df.columns and "task_state_label" in df.columns:
        df = df.rename(columns={"task_state_label": "stage"})
    stage_map = {0: "rest", 1: "listen", 5: "mem5", 9: "mem9", 13: "mem13"}
    if "stage" in df.columns and df["stage"].dtype.kind in ("i", "f"):
        df["stage"] = df["stage"].map(stage_map).fillna(df["stage"].astype(str))
    return df


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled = np.sqrt(((na - 1) * np.std(a, ddof=1) ** 2 + (nb - 1) * np.std(b, ddof=1) ** 2)
                     / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 1e-10 else np.nan


def group_medians_per_stage(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Per-subject, per-stage medians then group median/IQR."""
    sub_col = "subject_id" if "subject_id" in df.columns else "subject"
    rows = []
    valid_stages = [s for s in STAGE_ORDER if s in df["stage"].unique()]
    for stage in valid_stages:
        sdf = df[df["stage"] == stage]
        for m in metrics:
            if m not in df.columns:
                continue
            sub_meds = sdf.groupby(sub_col)[m].median().dropna()
            rows.append({
                "stage": stage, "metric": m,
                "n_subjects": len(sub_meds),
                "group_median": sub_meds.median(),
                "group_q25": sub_meds.quantile(0.25),
                "group_q75": sub_meds.quantile(0.75),
            })
    return pd.DataFrame(rows)


def friedman_chi2_p(df: pd.DataFrame, metric: str, stages: list[str],
                    sub_col: str) -> tuple[float, float]:
    pivot = (df[df["stage"].isin(stages)]
             .groupby([sub_col, "stage"])[metric]
             .median()
             .unstack("stage")
             .reindex(columns=stages)
             .dropna())
    if pivot.shape[0] < 5 or pivot.shape[1] < 2:
        return np.nan, np.nan
    stat, p = stats.friedmanchisquare(*[pivot[s].values for s in stages])
    return float(stat), float(p)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = args.run_dir or find_latest_run_dir(PROCESSED_BASE)
    if run_dir is None:
        logger.error("No run directory found; specify --run-dir")
        return
    logger.info("Using run dir: %s", run_dir)

    # Load corrected block_native_windows
    bn = load_block_native(run_dir)
    if bn.empty:
        logger.error("No block_native_windows found in %s", run_dir)
        return
    bn = resolve_stage(bn)
    sub_col = "subject_id" if "subject_id" in bn.columns else "subject"
    n_subjects = bn[sub_col].nunique()
    logger.info("block_native loaded: %d rows, %d subjects", len(bn), n_subjects)

    # ── Group medians per stage ──────────────────────────────────────────────
    metrics_of_interest = ["ecg_hrv_hr_mean_bpm", "ecg_hrv_rmssd_ms", "ecg_hrv_pnn50", "vagal_index"]
    avail = [m for m in metrics_of_interest if m in bn.columns]
    medians_df = group_medians_per_stage(bn, avail)
    med_csv = args.out_dir / "corrected_hrv_group_medians.csv"
    medians_df.to_csv(med_csv, index=False)
    logger.info("Wrote %s", med_csv)

    print("\n=== Corrected group medians per stage ===")
    print(medians_df.to_string(index=False))

    # ── Statistical tests ────────────────────────────────────────────────────
    valid_stages = [s for s in STAGE_ORDER if s in bn["stage"].unique()]
    stat_rows = []

    # Friedman for vagal_index / RMSSD
    for m in ["vagal_index", "ecg_hrv_rmssd_ms"]:
        if m not in bn.columns:
            continue
        chi2, p = friedman_chi2_p(bn, m, valid_stages, sub_col)
        stat_rows.append({"test": f"Friedman χ²  ({m})", "stat": chi2, "p": p,
                           "stages": str(valid_stages)})
        logger.info("Friedman %s: χ²=%.2f, p=%.4f", m, chi2, p)

    # C3: listen vs mem
    mem_present = [m for m in MEM_LABELS if m in bn["stage"].unique()]
    if LISTEN_LABEL in bn["stage"].unique() and mem_present:
        for m in ["vagal_index", "ecg_hrv_rmssd_ms", "ecg_hrv_hr_mean_bpm"]:
            if m not in bn.columns:
                continue
            listen_sub = bn[bn["stage"] == LISTEN_LABEL].groupby(sub_col)[m].median()
            mem_sub = bn[bn["stage"].isin(mem_present)].groupby(sub_col)[m].median()
            common = listen_sub.index.intersection(mem_sub.index)
            a, b = listen_sub[common].values, mem_sub[common].values
            valid = np.isfinite(a) & np.isfinite(b)
            a, b = a[valid], b[valid]
            if len(a) >= 5:
                _, p_wx = stats.wilcoxon(a, b)
                d = cohens_d(a, b)
                stat_rows.append({"test": f"Wilcoxon listen vs mem ({m})",
                                   "stat": d, "p": p_wx,
                                   "stages": f"{LISTEN_LABEL} vs {mem_present}"})
                logger.info("C3 Wilcoxon %s: p=%.4f, Cohen's d=%.3f", m, p_wx, d)

    stats_df = pd.DataFrame(stat_rows)
    stats_csv = args.out_dir / "corrected_hrv_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    logger.info("Wrote %s", stats_csv)
    print("\n=== Corrected HRV statistical tests ===")
    print(stats_df.to_string(index=False).encode("ascii", "replace").decode("ascii"))

    # ── Bland-Altman: old vs corrected RMSSD ────────────────────────────────
    if FEATURES_OLD.exists() and FEATURES_NEW.exists():
        logger.info("Computing Bland-Altman old vs corrected RMSSD …")
        old_df = pd.read_parquet(FEATURES_OLD)
        new_df = pd.read_parquet(FEATURES_NEW)

        sub_col_feat = "subject"
        for df_, tag in [(old_df, "old"), (new_df, "new")]:
            df_[tag + "_rmssd"] = df_["ecg_hrv_rmssd_ms"]
            df_[tag + "_hr"] = df_["ecg_hrv_hr_mean_bpm"]

        old_sub = old_df.groupby(sub_col_feat)["old_rmssd"].median().rename("rmssd_old")
        new_sub = new_df.groupby(sub_col_feat)["new_rmssd"].median().rename("rmssd_new")
        ba = pd.concat([old_sub, new_sub], axis=1).dropna()
        ba["diff"] = ba["rmssd_old"] - ba["rmssd_new"]
        ba["mean"] = 0.5 * (ba["rmssd_old"] + ba["rmssd_new"])

        bias = float(ba["diff"].mean())
        loa = 1.96 * float(ba["diff"].std(ddof=1))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(ba["mean"], ba["diff"], s=45, alpha=0.7)
        ax.axhline(bias, color="red", lw=1.5, label=f"Bias = {bias:+.1f} ms")
        ax.axhline(bias + loa, color="red", ls="--", lw=1, label=f"+1.96 SD = {bias+loa:.1f}")
        ax.axhline(bias - loa, color="red", ls="--", lw=1, label=f"-1.96 SD = {bias-loa:.1f}")
        ax.axhline(0, color="k", lw=0.8, alpha=0.4)
        ax.set_xlabel("Mean RMSSD (old + corrected)/2  [ms]", fontsize=10)
        ax.set_ylabel("Difference RMSSD (old − corrected)  [ms]", fontsize=10)
        ax.set_title("Bland-Altman: original vs NK2-corrected RMSSD (per-subject medians)", fontsize=10)
        ax.legend(fontsize=9)
        plt.tight_layout()
        ba_fig = args.out_dir / "bland_altman_rmssd.png"
        fig.savefig(str(ba_fig), dpi=130)
        plt.close(fig)
        logger.info("Bland-Altman RMSSD: bias=%.1f ms, LoA=[%.1f, %.1f] ms", bias, bias - loa, bias + loa)

        ba.to_csv(args.out_dir / "bland_altman_rmssd.csv")
        print(f"\nBland-Altman RMSSD: bias={bias:+.1f} ms, LoA=[{bias-loa:.1f}, {bias+loa:.1f}] ms")

    logger.info("Analysis complete — results in %s", args.out_dir)


if __name__ == "__main__":
    main()
