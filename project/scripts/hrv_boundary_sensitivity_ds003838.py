"""Boundary-excluded HRV sensitivity analysis for ds003838.

Peer-review concern: 60-s superwindows centred near stage transitions
(listen → mem5, etc.) mix RR intervals from adjacent stages, potentially
distorting the central listen-vs-mem contrast.

This script:
1. Loads block_native_windows.parquet (one run directory specified via --run-dir)
2. Computes distance of each window centre from the nearest stage boundary
3. Runs all stage-level analyses on TWO subsets:
     (a) all windows           (b) boundary-excluded (|distance| > 30 s)
4. Reports the comparison table and writes it as a CSV

Output columns:
  | Analysis                       | All windows | Boundary-excluded (±30s) |
  | vagal_index Friedman χ²        |             |                          |
  | C3: listen vs mem Wilcoxon q   |             |                          |
  | Cohen's d  C3                  |             |                          |
  | RMSSD listen-peak (median)     |             |                          |

Usage
-----
  python project/scripts/hrv_boundary_sensitivity_ds003838.py \\
      --run-dir "J:/processed/openneuro/ds003838/neuralmanifolddynamics_ds003838_20260610_184658" \\
      --out-dir "J:/processed/openneuro/ds003838/audit_ecg_ds003838"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_RUN_DIR = Path(
    "J:/processed/openneuro/ds003838/neuralmanifolddynamics_ds003838_20260610_184658"
)
OUT_DIR = Path("J:/processed/openneuro/ds003838/audit_ecg_ds003838")
BOUNDARY_MARGIN_S = 30.0   # exclude windows within 30s of a boundary

# Stage order for Friedman (repeated measures over subjects)
STAGE_ORDER = ["rest", "listen", "mem5", "mem9", "mem13"]

# C3 contrast: listen vs memory (any of mem5/mem9/mem13)
LISTEN_LABEL = "listen"
MEM_LABELS = ["mem5", "mem9", "mem13"]


def load_block_native(run_dir: Path) -> pd.DataFrame:
    """Load all per-subject block_native_windows parquets from a run directory."""
    rows = []
    for sub_dir in sorted(run_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        pq = sub_dir / "block_native_windows.parquet"
        csv = sub_dir / "block_native_windows.csv"
        if pq.exists():
            try:
                rows.append(pd.read_parquet(pq))
                continue
            except Exception:
                pass
        if csv.exists():
            try:
                rows.append(pd.read_csv(csv))
            except Exception:
                pass
    if not rows:
        raise FileNotFoundError(f"No block_native_windows found in {run_dir}")
    df = pd.concat(rows, ignore_index=True)
    logger.info("Loaded %d windows from %d subjects", len(df), df["subject_id"].nunique()
                if "subject_id" in df.columns else df["subject"].nunique())
    return df


def resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise subject / stage column names."""
    if "subject_id" not in df.columns and "subject" in df.columns:
        df = df.rename(columns={"subject": "subject_id"})
    if "stage" not in df.columns and "task_state_label" in df.columns:
        df = df.rename(columns={"task_state_label": "stage"})
    # Decode numeric stage codes
    stage_map = {0: "rest", 1: "listen", 5: "mem5", 9: "mem9", 13: "mem13"}
    if df["stage"].dtype.kind in ("i", "f"):
        df["stage"] = df["stage"].map(stage_map).fillna(df["stage"].astype(str))
    return df


def compute_boundary_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Add distance_to_boundary_s column (min distance to stage-boundary edge)."""
    dist_cols = []
    if "relative_time_in_block_sec" in df.columns:
        dist_cols.append(df["relative_time_in_block_sec"])
    if "distance_to_block_end_sec" in df.columns:
        dist_cols.append(df["distance_to_block_end_sec"])
    if not dist_cols:
        # Derive from block onset and offset if available
        if "block_onset_s" in df.columns and "block_offset_s" in df.columns and "t_center_s" in df.columns:
            df["relative_time_in_block_sec"] = df["t_center_s"] - df["block_onset_s"]
            df["distance_to_block_end_sec"] = df["block_offset_s"] - df["t_center_s"]
            dist_cols = [df["relative_time_in_block_sec"], df["distance_to_block_end_sec"]]
        else:
            logger.warning("No block-edge distance columns found; boundary exclusion unavailable")
            df["distance_to_boundary_s"] = np.inf
            return df
    df["distance_to_boundary_s"] = np.minimum(*[s.clip(lower=0) for s in dist_cols])
    return df


def window_center(df: pd.DataFrame) -> pd.Series:
    """Return window centre time (seconds from recording start)."""
    if "t_center_s" in df.columns:
        return df["t_center_s"]
    if "t_start" in df.columns and "t_end" in df.columns:
        return 0.5 * (df["t_start"] + df["t_end"])
    raise KeyError("Cannot determine window centre from columns: " + str(df.columns.tolist()))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for two independent samples."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_std = np.sqrt(((na - 1) * np.std(a, ddof=1) ** 2 + (nb - 1) * np.std(b, ddof=1) ** 2)
                         / (na + nb - 2))
    if pooled_std < 1e-10:
        return np.nan
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def wilcoxon_paired_q(a_vals: np.ndarray, b_vals: np.ndarray) -> float:
    """Wilcoxon signed-rank test p-value (paired, on per-subject medians)."""
    if len(a_vals) < 5:
        return np.nan
    try:
        _, p = stats.wilcoxon(a_vals, b_vals)
        return float(p)
    except Exception:
        return np.nan


def friedman_chi2(
    df: pd.DataFrame, metric: str, stages: list[str]
) -> float:
    """Friedman test: repeated measures over subjects across stages."""
    pivot = (df[df["stage"].isin(stages)]
             .groupby(["subject_id", "stage"])[metric]
             .median()
             .unstack("stage")
             .reindex(columns=stages)
             .dropna())
    if pivot.shape[0] < 5 or pivot.shape[1] < 2:
        return np.nan
    try:
        _, p = stats.friedmanchisquare(*[pivot[s].values for s in stages])
        return float(p)
    except Exception:
        return np.nan


def analyse(df: pd.DataFrame, label: str) -> Dict[str, float]:
    """Run all four analyses on a subset of windows."""
    results: Dict[str, float] = {}

    valid_stages = [s for s in STAGE_ORDER if s in df["stage"].unique()]
    metric_col = None
    for cand in ("vagal_index", "ecg_hrv_rmssd_ms", "ecg_rmssd"):
        if cand in df.columns and df[cand].notna().any():
            metric_col = cand
            break

    if metric_col is None:
        logger.warning("[%s] No usable HRV metric column found", label)
        return {}

    logger.info("[%s] Metric: %s | Stages: %s | N windows: %d",
                label, metric_col, valid_stages, len(df))

    if len(valid_stages) < 2:
        logger.warning("[%s] Too few stages after filtering (%s) — boundary margin may be too aggressive",
                       label, valid_stages)
        results["friedman_p"] = np.nan
        results["wilcoxon_p_c3"] = np.nan
        results["cohens_d_c3"] = np.nan
        results["rmssd_listen_median"] = np.nan
        results["n_windows"] = int(len(df))
        results["n_subjects"] = int(df["subject_id"].nunique())
        results["note"] = f"Only stages present: {valid_stages}"
        return results

    # 1. Friedman for vagal_index / RMSSD across stages
    results["friedman_p"] = friedman_chi2(df, metric_col, valid_stages)

    # 2 & 3. C3: listen vs mem (Wilcoxon + Cohen's d on per-subject medians)
    mem_present = [m for m in MEM_LABELS if m in df["stage"].unique()]
    if LISTEN_LABEL in df["stage"].unique() and mem_present:
        listen_sub = (df[df["stage"] == LISTEN_LABEL]
                      .groupby("subject_id")[metric_col].median())
        mem_sub = (df[df["stage"].isin(mem_present)]
                   .groupby("subject_id")[metric_col].median())
        common = listen_sub.index.intersection(mem_sub.index)
        a, b = listen_sub[common].values, mem_sub[common].values
        valid = np.isfinite(a) & np.isfinite(b)
        a, b = a[valid], b[valid]
        results["wilcoxon_p_c3"] = wilcoxon_paired_q(a, b)
        results["cohens_d_c3"] = cohens_d(a, b)
    else:
        results["wilcoxon_p_c3"] = np.nan
        results["cohens_d_c3"] = np.nan

    # 4. RMSSD listen-peak (median across all subjects)
    if "ecg_hrv_rmssd_ms" in df.columns:
        listen_rmssd = df[df["stage"] == LISTEN_LABEL]["ecg_hrv_rmssd_ms"].median()
    else:
        listen_rmssd = np.nan
    results["rmssd_listen_median"] = float(listen_rmssd) if np.isfinite(listen_rmssd) else np.nan

    results["n_windows"] = int(len(df))
    results["n_subjects"] = int(df["subject_id"].nunique())
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--margin-s", type=float, default=BOUNDARY_MARGIN_S)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_block_native(args.run_dir)
    df = resolve_columns(df)
    df = compute_boundary_distance(df)

    n_total = len(df)
    df_excl = df[df["distance_to_boundary_s"] >= args.margin_s]
    n_excl = len(df_excl)
    logger.info("All windows: %d | Boundary-excluded (>=%.0fs): %d (%.1f%% retained)",
                n_total, args.margin_s, n_excl, 100 * n_excl / max(1, n_total))

    res_all = analyse(df, "all_windows")
    res_excl = analyse(df_excl, "boundary_excluded")

    # Build comparison table
    rows = [
        {
            "Analysis": "vagal_index / RMSSD Friedman p",
            "All windows": f"{res_all.get('friedman_p', np.nan):.4f}",
            "Boundary-excluded (±30s)": f"{res_excl.get('friedman_p', np.nan):.4f}",
        },
        {
            "Analysis": "C3 listen vs mem Wilcoxon p",
            "All windows": f"{res_all.get('wilcoxon_p_c3', np.nan):.4f}",
            "Boundary-excluded (±30s)": f"{res_excl.get('wilcoxon_p_c3', np.nan):.4f}",
        },
        {
            "Analysis": "C3 Cohen's d",
            "All windows": f"{res_all.get('cohens_d_c3', np.nan):.3f}",
            "Boundary-excluded (±30s)": f"{res_excl.get('cohens_d_c3', np.nan):.3f}",
        },
        {
            "Analysis": "RMSSD listen-peak median (ms)",
            "All windows": f"{res_all.get('rmssd_listen_median', np.nan):.1f}",
            "Boundary-excluded (±30s)": f"{res_excl.get('rmssd_listen_median', np.nan):.1f}",
        },
        {
            "Analysis": "N windows",
            "All windows": str(res_all.get("n_windows", "")),
            "Boundary-excluded (±30s)": str(res_excl.get("n_windows", "")),
        },
        {
            "Analysis": "N subjects",
            "All windows": str(res_all.get("n_subjects", "")),
            "Boundary-excluded (±30s)": str(res_excl.get("n_subjects", "")),
        },
    ]
    table = pd.DataFrame(rows)

    out_csv = args.out_dir / "hrv_boundary_sensitivity.csv"
    table.to_csv(out_csv, index=False)
    logger.info("Wrote %s", out_csv)

    print("\n=== Boundary-excluded HRV sensitivity ===")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
