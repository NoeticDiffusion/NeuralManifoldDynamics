"""
Multi-subject descriptive event-vs-control MNPS summary.

Reads all event-locked Parquet files produced by batch_event_locked.py,
concatenates them, and produces a per-subject + pooled descriptive summary
by (bin, condition).

EXPLORATORY — no inferential statistics, no hypothesis tests.

Claim boundary:
  "Using YASA 0.7.0 on ds005555 PSG_F3 (protocol v1), event-locked MNPS
   measurements show the following descriptive patterns across sub-1 to sub-5."
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for pkg in ["mndm/src", "core/src"]:
    p = str(REPO / pkg)
    if p not in sys.path:
        sys.path.insert(0, p)

import argparse
import numpy as np
import pandas as pd

PROC_ROOT = Path(r"M:\datasets\processed\openneuro\ds005555")
SUBJECTS  = ["sub-1", "sub-2", "sub-3", "sub-4", "sub-5"]
MNPS_DIMS = ["m", "d", "e"]

parser = argparse.ArgumentParser()
parser.add_argument("--subjects", nargs="+", default=SUBJECTS)
parser.add_argument("--out", type=Path, default=PROC_ROOT / "multi_subject_descriptive.csv")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load all parquets
# ---------------------------------------------------------------------------
dfs = []
for sub in args.subjects:
    pattern = f"**/{sub}_Sleep_acq-psg_event_locked_v1.parquet"
    files = sorted(PROC_ROOT.rglob(pattern.replace("**", "*")))
    if not files:
        # try recursive
        files = list(PROC_ROOT.rglob(f"{sub}_Sleep_acq-psg_event_locked_v1.parquet"))
    if not files:
        print(f"  [{sub}] No Parquet found — skipping")
        continue
    fpath = files[-1]  # most recent
    df = pd.read_parquet(str(fpath))
    df["subject"] = sub
    dfs.append(df)
    print(f"  [{sub}] Loaded {len(df)} rows from {fpath.name}")

if not dfs:
    print("No Parquet files found. Run batch_event_locked.py first.")
    sys.exit(1)

df_all = pd.concat(dfs, ignore_index=True)
n_subjects = df_all["subject"].nunique()
print(f"\nTotal: {len(df_all)} rows across {n_subjects} subjects")

# ---------------------------------------------------------------------------
# Bins present (excluding "control" bin label used by matched controls)
# ---------------------------------------------------------------------------
EVENT_BINS = ["pre_far", "pre_near", "event", "post_near", "post_far"]
CONTROL_BIN = "control"

print("\n" + "="*72)
print("DESCRIPTIVE MULTI-SUBJECT EVENT vs CONTROL  [EXPLORATORY]")
print(f"Protocol v1 | YASA 0.7.0 | PSG_F3 | N2 only | ds005555")
print("="*72)

# ---------------------------------------------------------------------------
# 1. Per-subject, per-condition mean
# ---------------------------------------------------------------------------
print("\n[1] Per-subject mean (all bins pooled within condition)")
print(f"  {'Subject':<10} {'Cond':<18} {'N':>6}  {'m':>8} {'d':>8} {'e':>8}")
print(f"  {'-'*10} {'-'*18} {'-'*6}  {'-'*8} {'-'*8} {'-'*8}")
for sub in df_all["subject"].unique():
    s_df = df_all[df_all["subject"] == sub]
    for cond in ["spindle_event", "matched_control"]:
        c_df = s_df[s_df["condition"] == cond]
        if len(c_df) == 0:
            continue
        print(f"  {sub:<10} {cond:<18} {len(c_df):>6}  "
              f"{c_df['m'].mean():>8.4f} {c_df['d'].mean():>8.4f} {c_df['e'].mean():>8.4f}")

# ---------------------------------------------------------------------------
# 2. Pooled: event vs control per bin
# ---------------------------------------------------------------------------
print(f"\n[2] Pooled across subjects: spindle_event per bin vs all matched_controls")
ctrl_all = df_all[df_all["condition"] == "matched_control"]
ev_all   = df_all[df_all["condition"] == "spindle_event"]

print(f"\n  Control (all subjects, all bins): N={len(ctrl_all)}")
ctrl_means = {dim: ctrl_all[dim].mean() for dim in MNPS_DIMS}
print(f"  {'m':>8} {'d':>8} {'e':>8}")
print(f"  {ctrl_means['m']:>8.4f} {ctrl_means['d']:>8.4f} {ctrl_means['e']:>8.4f}")

print(f"\n  {'Bin':<22} {'N_ev':>6}  {'m':>8} {'d':>8} {'e':>8}  "
      f"{'dm':>8} {'dd':>8} {'de':>8}")
print(f"  {'-'*22} {'-'*6}  {'-'*8} {'-'*8} {'-'*8}  {'-'*8} {'-'*8} {'-'*8}")
for b in EVENT_BINS:
    b_df = ev_all[ev_all["bin_label"] == b]
    if len(b_df) == 0:
        continue
    means = {dim: b_df[dim].mean() for dim in MNPS_DIMS}
    diffs = {dim: means[dim] - ctrl_means[dim] for dim in MNPS_DIMS}
    print(f"  {b:<22} {len(b_df):>6}  "
          f"{means['m']:>8.4f} {means['d']:>8.4f} {means['e']:>8.4f}  "
          f"{diffs['m']:>+8.4f} {diffs['d']:>+8.4f} {diffs['e']:>+8.4f}")

# ---------------------------------------------------------------------------
# 3. Direction consistency across subjects (event bin vs controls)
# ---------------------------------------------------------------------------
print(f"\n[3] Direction consistency — event bin (t=0 to +3s) vs controls, per subject")
print(f"  ('+' = event > control, '-' = event < control)")
print(f"\n  {'Subject':<10}  {'dm':>8} {'dd':>8} {'de':>8}")
print(f"  {'-'*10}  {'-'*8} {'-'*8} {'-'*8}")
directions = {dim: [] for dim in MNPS_DIMS}
for sub in df_all["subject"].unique():
    s_df = df_all[df_all["subject"] == sub]
    ev_b = s_df[(s_df["condition"] == "spindle_event") & (s_df["bin_label"] == "event")]
    ct   = s_df[s_df["condition"] == "matched_control"]
    if len(ev_b) == 0 or len(ct) == 0:
        print(f"  {sub:<10}  (no event or control data)")
        continue
    diffs = {dim: ev_b[dim].mean() - ct[dim].mean() for dim in MNPS_DIMS}
    signs = {dim: "+" if diffs[dim] > 0 else "-" for dim in MNPS_DIMS}
    for dim in MNPS_DIMS:
        directions[dim].append(diffs[dim] > 0)
    print(f"  {sub:<10}  "
          f"{signs['m']}{abs(diffs['m']):.4f}  "
          f"{signs['d']}{abs(diffs['d']):.4f}  "
          f"{signs['e']}{abs(diffs['e']):.4f}")

n_subs = len(df_all["subject"].unique())
if n_subs > 1:
    print(f"\n  Direction agreement (fraction subjects with same sign):")
    for dim in MNPS_DIMS:
        d = directions[dim]
        n_pos = sum(d)
        n_neg = len(d) - n_pos
        dominant = "event>ctrl" if n_pos > n_neg else "event<ctrl"
        frac = max(n_pos, n_neg) / len(d) if len(d) > 0 else 0
        print(f"    {dim}: {n_pos}/{len(d)} positive, {dominant}, agreement={frac:.0%}")

# ---------------------------------------------------------------------------
# 4. Save pooled table
# ---------------------------------------------------------------------------
# Per-bin × condition summary
rows_out = []
for sub in list(df_all["subject"].unique()) + ["POOLED"]:
    sub_df = df_all if sub == "POOLED" else df_all[df_all["subject"] == sub]
    for cond in ["spindle_event", "matched_control"]:
        c_df = sub_df[sub_df["condition"] == cond]
        for b in (EVENT_BINS if cond == "spindle_event" else [CONTROL_BIN]):
            b_df = c_df[c_df["bin_label"] == b] if b != CONTROL_BIN else c_df
            if len(b_df) == 0:
                continue
            rows_out.append({
                "subject": sub,
                "condition": cond,
                "bin": b,
                "n": len(b_df),
                **{f"{dim}_mean": b_df[dim].mean() for dim in MNPS_DIMS},
                **{f"{dim}_std":  b_df[dim].std()  for dim in MNPS_DIMS},
            })

df_out = pd.DataFrame(rows_out)
df_out.to_csv(str(args.out), index=False)

print(f"\n[4] Summary table saved: {args.out}")
print(f"\n{'='*72}")
print("EXPLORATORY — no inferential statistics. Claim: measurement differences only.")
print(f"{'='*72}")
