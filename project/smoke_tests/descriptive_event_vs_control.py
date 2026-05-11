"""
Exploratory descriptive summary: spindle events vs. matched N2 controls.

EXPLORATORY / HYPOTHESIS-FREE — no statistical tests, no p-values.

This script reads the event-locked export Parquet and produces a plain-text
summary of MNPS (m, d, e) across bins and conditions. It is intended for
visual sanity-checking of the pipeline output, not for drawing biological
conclusions.

All findings are marked EXPLORATORY. Claim boundary:

  "Using YASA 0.7.0 detector-derived spindle events on ds005555 sub-1,
   the event-locked MNPS export shows the following descriptive patterns."

Not:

  "Sleep spindles have an MNPS effect."

Usage
-----
  python descriptive_event_vs_control.py [--parquet PATH]

If no --parquet is supplied, the script reads the default path from the
smoke-test smoke_real_h5_event_locked.py convention.
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

try:
    import pandas as pd
except ImportError:
    print("pandas required: pip install pandas")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_PARQUET = Path(r"M:\datasets\processed\mndm\ds005555"
                       r"\sub-1_task-Sleep_acq-psg_event_locked.parquet")

# Also try the smoke-test output path
FALLBACK_PARQUET = Path(r"M:\datasets\processed\mndm\ds005555"
                        r"\event_locked_smoke_output.parquet")

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Descriptive event-vs-control MNPS summary")
parser.add_argument("--parquet", type=Path, default=None, help="Path to event-locked Parquet")
parser.add_argument("--out", type=Path, default=None, help="Optional output text file")
args = parser.parse_args()

parquet_path = args.parquet
if parquet_path is None:
    for candidate in [DEFAULT_PARQUET, FALLBACK_PARQUET]:
        if candidate.exists():
            parquet_path = candidate
            break
    if parquet_path is None:
        print(f"No Parquet found at default locations. Run smoke_real_h5_event_locked.py first.")
        print(f"  Tried: {DEFAULT_PARQUET}")
        print(f"  Tried: {FALLBACK_PARQUET}")
        print("\nAlternatively, pass --parquet PATH to this script.")
        sys.exit(1)

print(f"Loading: {parquet_path}")
df = pd.read_parquet(str(parquet_path))

lines: list[str] = []

def emit(s: str = "") -> None:
    lines.append(s)
    print(s)

emit("=" * 70)
emit("DESCRIPTIVE EVENT vs CONTROL MNPS SUMMARY  [EXPLORATORY]")
emit("Source: YASA 0.7.0 detector-derived spindle events, ds005555 sub-1")
emit("Claim: measurement differences only; no causal/biological inference")
emit("=" * 70)

emit(f"\nParquet: {parquet_path}")
emit(f"Rows: {len(df)},  Columns: {len(df.columns)}")

# Check required columns
required_cols = {"condition", "bin_label", "m", "d", "e"}
missing = required_cols - set(df.columns)
if missing:
    emit(f"\nERROR: Missing required columns: {missing}")
    sys.exit(1)

# Conditions
conditions = df["condition"].unique().tolist()
emit(f"\nConditions: {conditions}")
for cond in conditions:
    n = (df["condition"] == cond).sum()
    emit(f"  {cond}: {n} rows")

# Bins
bins = df["bin_label"].dropna().unique().tolist()
bins_sorted = sorted(bins)
emit(f"\nBins: {bins_sorted}")

# ---------------------------------------------------------------------------
# MNPS descriptive statistics by (bin, condition)
# ---------------------------------------------------------------------------
emit("\n" + "-" * 70)
emit("[1] MNPS per bin × condition  (mean ± SD)")
emit("    m = mobility, d = diffusivity, e = entropy")
emit("-" * 70)

MNPS_DIMS = ["m", "d", "e"]
for cond in ["spindle_event", "matched_control"]:
    if cond not in df["condition"].values:
        continue
    sub = df[df["condition"] == cond]
    emit(f"\n  Condition: {cond}  (N={len(sub)} rows)")
    emit(f"  {'Bin':<22}  {'N':>5}  {'m mean':>10}  {'d mean':>10}  {'e mean':>10}")
    emit(f"  {'-'*22}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}")
    for b in bins_sorted:
        sub_b = sub[sub["bin_label"] == b]
        n_b = len(sub_b)
        if n_b == 0:
            emit(f"  {b:<22}  {n_b:>5}  {'(no data)':>10}")
            continue
        means = {dim: sub_b[dim].mean() for dim in MNPS_DIMS}
        stds  = {dim: sub_b[dim].std() for dim in MNPS_DIMS}
        emit(
            f"  {b:<22}  {n_b:>5}  "
            f"{means['m']:>8.4f}   {means['d']:>8.4f}   {means['e']:>8.4f}"
        )

# ---------------------------------------------------------------------------
# Difference: spindle event - matched control
# Controls are assigned bin_label="control" (not split per bin), so the
# comparison is: all event-condition rows vs all control-condition rows.
# We also show a per-bin breakdown of event rows for within-event structure.
# ---------------------------------------------------------------------------
emit("\n" + "-" * 70)
emit("[2] Descriptive difference: all spindle_event vs all matched_control")
emit("    (NOT a statistical test — descriptive only)")
emit("    Controls span all bins uniformly; comparison is aggregate.")
emit("-" * 70)

event_sub  = df[df["condition"] == "spindle_event"]
ctrl_sub   = df[df["condition"] == "matched_control"]
n_ev_total = len(event_sub)
n_ct_total = len(ctrl_sub)

if n_ev_total > 0 and n_ct_total > 0:
    emit(f"\n  Overall: spindle_event (N={n_ev_total}) vs matched_control (N={n_ct_total})")
    emit(f"\n  {'Dim':<6}  {'event mean':>12}  {'ctrl mean':>12}  {'diff (ev-ct)':>14}")
    emit(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*14}")
    for dim in MNPS_DIMS:
        ev_mean = event_sub[dim].mean()
        ct_mean = ctrl_sub[dim].mean()
        diff = ev_mean - ct_mean
        emit(f"  {dim:<6}  {ev_mean:>12.4f}  {ct_mean:>12.4f}  {diff:>+14.4f}")

# Per-bin breakdown of spindle_event rows (temporal structure)
emit(f"\n  Per-bin breakdown — spindle_event only (temporal structure):")
emit(f"  {'Bin':<22}  {'N':>5}  {'m':>10}  {'d':>10}  {'e':>10}")
emit(f"  {'-'*22}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}")
for b in bins_sorted:
    if b == "control":
        continue  # control bin belongs to matched_control condition
    sub_b = event_sub[event_sub["bin_label"] == b]
    if len(sub_b) == 0:
        continue
    emit(
        f"  {b:<22}  {len(sub_b):>5}  "
        f"{sub_b['m'].mean():>8.4f}   {sub_b['d'].mean():>8.4f}   {sub_b['e'].mean():>8.4f}"
    )

# ---------------------------------------------------------------------------
# Data quality checks
# ---------------------------------------------------------------------------
emit("\n" + "-" * 70)
emit("[3] Data quality")
emit("-" * 70)

for dim in MNPS_DIMS:
    col = df[dim]
    n_finite = col.isfinite().sum() if hasattr(col, "isfinite") else np.isfinite(col.values).sum()
    n_nan = col.isna().sum()
    emit(f"  {dim}: {n_finite}/{len(col)} finite,  {n_nan} NaN")

if "mnps_finite" in df.columns:
    n_finite_flag = df["mnps_finite"].sum()
    emit(f"  mnps_finite flag: {n_finite_flag}/{len(df)} True")

# Provenance sample
emit("\n" + "-" * 70)
emit("[4] Provenance sample (from first row)")
emit("-" * 70)
prov_cols = [
    "profile_name", "window_length_s", "window_step_s",
    "alignment_reference", "control_seed",
    "event_source_path", "annotation_source_hash",
    "n_events_input", "n_events_aligned", "n_events_excluded_transition",
    "match_success_rate",
]
for col in prov_cols:
    if col in df.columns:
        val = df[col].iloc[0]
        if col == "annotation_source_hash" and isinstance(val, str) and len(val) > 16:
            val = val[:16] + "..."
        emit(f"  {col}: {val}")

emit("\n" + "=" * 70)
emit("END OF EXPLORATORY SUMMARY")
emit("All values are descriptive measurements. No causal claims.")
emit("=" * 70)

# Optional: write to file
if args.out:
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nOutput written to: {args.out}")
