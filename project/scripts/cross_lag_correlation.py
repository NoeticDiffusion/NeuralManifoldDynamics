"""Cross-lag Spearman correlation: anchor indices → MNPS coordinates.

Reads block-native windows from a MNDM summarize run directory and computes
signed Spearman correlations between anchor indices and MNPS coordinates at
temporal lags ranging from -3 to +3 windows.

A positive lag means the anchor column leads the MNPS column
(anchor at time t predicts MNPS at t+lag, i.e., anchor precedes manifold).
A negative lag means MNPS leads anchor.

Lags are always computed WITHIN a single block so that cross-block window
pairs are never formed.

Usage
-----
python project/scripts/cross_lag_correlation.py \\
    --run-dir J:/processed/openneuro/ds003838/neuralmanifolddynamics_ds003838_20260610_125038 \\
    --out cross_lag_correlations.csv

Options
-------
--anchor-cols       Anchor columns to include (default: vagal_index sympathetic_index anchor_index)
--mnps-cols         MNPS columns to include (default: m d e)
--lags              Space-separated lag values in windows (default: -3 -2 -1 0 1 2 3)
--min-pairs         Minimum number of valid (anchor, mnps) pairs required to emit a row (default: 10)
--stage-col         Column used to group stages (default: task_state_label)
--sort-col          Column used to order windows within a block (default: window_start_sec)
--pool              If set, also write a pooled summary (median r across subjects per cell).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _cross_lag_within_block(
    df_block: pd.DataFrame,
    anchor_col: str,
    mnps_col: str,
    lag: int,
    sort_col: str,
) -> tuple[float, int]:
    """Return (spearman_r, n_pairs) for one block at a given lag."""
    ordered = df_block.sort_values(sort_col)
    a = ordered[anchor_col].to_numpy(dtype=float)
    m = ordered[mnps_col].to_numpy(dtype=float)
    n = len(a)

    if lag >= 0:
        t_idx = np.arange(n - lag)
        a_vals = a[t_idx]
        m_vals = m[t_idx + lag]
    else:
        abs_lag = -lag
        t_idx = np.arange(abs_lag, n)
        a_vals = a[t_idx]
        m_vals = m[t_idx + lag]  # lag is negative, so t_idx + lag = t_idx - abs_lag

    valid = np.isfinite(a_vals) & np.isfinite(m_vals)
    n_pairs = int(valid.sum())
    if n_pairs < 2:
        return float("nan"), n_pairs

    r, _ = stats.spearmanr(a_vals[valid], m_vals[valid])
    return float(r), n_pairs


def compute_cross_lag(
    df: pd.DataFrame,
    anchor_cols: list[str],
    mnps_cols: list[str],
    lags: list[int],
    stage_col: str,
    sort_col: str,
    min_pairs: int,
) -> pd.DataFrame:
    """Compute cross-lag correlations for all anchor × mnps × stage × lag combos.

    Pairs are formed within blocks only, then pooled across blocks within a
    stage using a single Spearman call on the concatenated valid pairs.
    """
    records = []
    subject_id = str(df["subject_id"].iloc[0]) if "subject_id" in df.columns else "unknown"
    task = str(df["task_state_label"].iloc[0]) if "task_state_label" in df.columns else "unknown"

    # Determine task from the parquet file name if not in data.
    stages = df[stage_col].dropna().unique() if stage_col in df.columns else ["unknown"]

    for stage in stages:
        df_stage = df[df[stage_col] == stage] if stage != "unknown" else df
        blocks = df_stage["block_id"].unique() if "block_id" in df_stage.columns else [None]

        for anchor_col in anchor_cols:
            if anchor_col not in df_stage.columns:
                continue
            for mnps_col in mnps_cols:
                if mnps_col not in df_stage.columns:
                    continue
                for lag in lags:
                    # Pool pairs across all blocks in this stage.
                    all_a, all_m = [], []
                    for bid in blocks:
                        if bid is None:
                            df_block = df_stage
                        else:
                            df_block = df_stage[df_stage["block_id"] == bid]
                        if len(df_block) < 2:
                            continue
                        ordered = df_block.sort_values(sort_col)
                        a = ordered[anchor_col].to_numpy(dtype=float)
                        m = ordered[mnps_col].to_numpy(dtype=float)
                        n = len(a)
                        if lag >= 0:
                            if n <= lag:
                                continue
                            idx = np.arange(n - lag)
                            a_vals = a[idx]
                            m_vals = m[idx + lag]
                        else:
                            abs_lag = -lag
                            if n <= abs_lag:
                                continue
                            idx = np.arange(abs_lag, n)
                            a_vals = a[idx]
                            m_vals = m[idx + lag]
                        valid = np.isfinite(a_vals) & np.isfinite(m_vals)
                        all_a.append(a_vals[valid])
                        all_m.append(m_vals[valid])

                    if not all_a:
                        continue
                    pool_a = np.concatenate(all_a)
                    pool_m = np.concatenate(all_m)
                    n_pairs = len(pool_a)
                    if n_pairs < min_pairs:
                        continue
                    r, _ = stats.spearmanr(pool_a, pool_m)
                    records.append({
                        "subject_id": subject_id,
                        "stage": str(stage),
                        "anchor_col": anchor_col,
                        "mnps_col": mnps_col,
                        "lag": lag,
                        "spearman_r": round(float(r), 6),
                        "n_pairs": n_pairs,
                    })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Discovery + I/O
# ---------------------------------------------------------------------------

def _find_parquets(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("sub-*/block_native_windows.parquet"))


def _pooled_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Median r across subjects, per (stage, anchor_col, mnps_col, lag)."""
    grp = df.groupby(["stage", "anchor_col", "mnps_col", "lag"])
    return (
        grp["spearman_r"]
        .agg(
            median_r="median",
            mean_r="mean",
            std_r="std",
            n_subjects="count",
        )
        .reset_index()
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-dir", required=True,
                   help="Path to a neuralmanifolddynamics_* run directory.")
    p.add_argument("--out", default="cross_lag_correlations.csv",
                   help="Output CSV path (default: %(default)s).")
    p.add_argument("--anchor-cols", nargs="+",
                   default=["vagal_index", "sympathetic_index", "anchor_index"],
                   help="Anchor columns to test.")
    p.add_argument("--mnps-cols", nargs="+",
                   default=["m", "d", "e"],
                   help="MNPS coordinate columns to test.")
    p.add_argument("--lags", nargs="+", type=int,
                   default=[-3, -2, -1, 0, 1, 2, 3],
                   help="Lag values in units of windows (default: -3 … +3).")
    p.add_argument("--min-pairs", type=int, default=10,
                   help="Minimum pooled pairs required to emit a row (default: 10).")
    p.add_argument("--stage-col", default="task_state_label",
                   help="Column used to identify stages (default: task_state_label).")
    p.add_argument("--sort-col", default="window_start_sec",
                   help="Column used to order windows within a block (default: window_start_sec).")
    p.add_argument("--pool", action="store_true",
                   help="Also write a pooled summary CSV (median r across subjects).")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    run_dir = Path(args.run_dir)
    out_path = Path(args.out)

    parquets = _find_parquets(run_dir)
    if not parquets:
        print(f"ERROR: No block_native_windows.parquet found under {run_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(parquets)} parquet files in {run_dir.name}")

    all_frames: list[pd.DataFrame] = []
    for i, pq in enumerate(parquets, 1):
        subject_tag = pq.parent.name  # e.g. sub-035_digit_span
        print(f"  [{i:3d}/{len(parquets)}] {subject_tag} ...", end=" ", flush=True)
        try:
            df = pd.read_parquet(pq)
            result = compute_cross_lag(
                df,
                anchor_cols=args.anchor_cols,
                mnps_cols=args.mnps_cols,
                lags=args.lags,
                stage_col=args.stage_col,
                sort_col=args.sort_col,
                min_pairs=args.min_pairs,
            )
            if not result.empty:
                # Tag with task from folder name (e.g. "digit_span" or "rest")
                task_tag = "_".join(subject_tag.split("_")[1:])
                result.insert(1, "task", task_tag)
                all_frames.append(result)
                print(f"{len(result)} rows")
            else:
                print("no rows (insufficient data)")
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)

    if not all_frames:
        print("No results produced.", file=sys.stderr)
        sys.exit(1)

    combined = pd.concat(all_frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"\nWrote {len(combined)} rows -> {out_path}")

    if args.pool:
        pooled_path = out_path.with_stem(out_path.stem + "_pooled")
        pooled = _pooled_summary(combined)
        pooled.to_csv(pooled_path, index=False)
        print(f"Wrote pooled summary ({len(pooled)} rows) -> {pooled_path}")

    # Quick sanity: peak lag distribution
    if not combined.empty:
        peak_lags = (
            combined.groupby(["subject_id", "task", "stage", "anchor_col", "mnps_col"])
            .apply(lambda g: g.loc[g["spearman_r"].abs().idxmax(), "lag"], include_groups=False)
            .value_counts()
            .sort_index()
        )
        print("\nPeak-|r| lag distribution across all subject×stage×anchor×mnps cells:")
        print(peak_lags.to_string())


if __name__ == "__main__":
    main()
