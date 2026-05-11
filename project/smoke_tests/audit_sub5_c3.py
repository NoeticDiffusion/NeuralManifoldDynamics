"""
Sub-5 C3 Δd audit — why does the event bin show Δd = -1.20?

Checks:
  1. Event and control row counts per bin
  2. Distribution of d values (event vs control) — percentiles, IQR, outliers
  3. Top-N windows by extreme d value in the event bin
  4. Stage and time distribution of event-bin rows
  5. Detector-table summary (spindle durations, amplitudes, frequencies)
  6. Whether a small number of windows dominates the mean
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for pkg in ["mndm/src", "core/src"]:
    p = str(REPO / pkg)
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd

PROC_ROOT = Path(r"M:\datasets\processed\openneuro\ds005555")
DS_ROOT   = Path(r"M:\datasets\received\openneuro\ds005555")
SUB       = "sub-5"

# Load C3 event-locked parquet
c3_files = list(PROC_ROOT.rglob(f"{SUB}_Sleep_acq-psg_event_locked_v1_psg_c3.parquet"))
if not c3_files:
    print(f"No C3 parquet for {SUB}")
    sys.exit(1)
df = pd.read_parquet(str(c3_files[-1]))
print(f"Loaded {len(df)} rows from {c3_files[-1].name}")

MNPS_DIMS = ["m", "d", "e"]
EVENT_BIN = "event"

ev = df[(df["condition"] == "spindle_event") & (df["bin_label"] == EVENT_BIN)]
ct = df[df["condition"] == "matched_control"]

print(f"\n{'='*64}")
print(f"[1] Row counts")
print(f"{'='*64}")
print(f"  Event bin rows   : {len(ev)}")
print(f"  Control rows     : {len(ct)}")
print(f"  Delta d (raw)    : {ev['d'].mean():.4f} - {ct['d'].mean():.4f} = {ev['d'].mean()-ct['d'].mean():.4f}")

print(f"\n{'='*64}")
print(f"[2] Distribution of d values (event bin vs controls)")
print(f"{'='*64}")
for label, data in [("event_bin", ev["d"]), ("control", ct["d"])]:
    pct = np.percentile(data.dropna(), [1, 5, 25, 50, 75, 95, 99])
    iqr = pct[4] - pct[2]
    print(f"\n  {label} (n={len(data)}):")
    print(f"    mean={data.mean():.4f}  std={data.std():.4f}  IQR={iqr:.4f}")
    print(f"    p1={pct[0]:.3f} p5={pct[1]:.3f} p25={pct[2]:.3f} p50={pct[3]:.3f} "
          f"p75={pct[4]:.3f} p95={pct[5]:.3f} p99={pct[6]:.3f}")
    # Outlier count (> 3 IQR from median)
    lo = pct[3] - 3*iqr
    hi = pct[3] + 3*iqr
    n_out = int(((data < lo) | (data > hi)).sum())
    print(f"    outliers (|d - median| > 3*IQR): {n_out} / {len(data)} ({100*n_out/len(data):.1f}%)")

print(f"\n{'='*64}")
print(f"[3] Top-10 extreme d rows in event bin (by |d|)")
print(f"{'='*64}")
top = ev.nlargest(5, "d")[["event_id","window_id","window_center_sec","d","m","e","stage","bin_label","rel_time_sec"]]
bot = ev.nsmallest(5, "d")[["event_id","window_id","window_center_sec","d","m","e","stage","bin_label","rel_time_sec"]]
print("\n  Top 5 highest d:")
print(top.to_string(index=False))
print("\n  Top 5 lowest d:")
print(bot.to_string(index=False))

print(f"\n{'='*64}")
print(f"[4] How many unique events contribute to the event bin?")
print(f"{'='*64}")
n_unique_ev = ev["event_id"].nunique() if "event_id" in ev.columns else "N/A"
n_unique_win = ev["window_id"].nunique() if "window_id" in ev.columns else "N/A"
print(f"  Unique event IDs in event bin: {n_unique_ev}")
print(f"  Unique window IDs in event bin: {n_unique_win}")

# Check if a few events drive the mean
if "event_id" in ev.columns:
    per_event_d = ev.groupby("event_id")["d"].mean().sort_values()
    print(f"\n  Per-event mean d (top 5 most extreme low):")
    print(per_event_d.head(5).to_string())
    print(f"\n  Per-event mean d (top 5 most extreme high):")
    print(per_event_d.tail(5).to_string())
    pct10 = float(np.percentile(per_event_d, 10))
    print(f"\n  Events with mean d < {pct10:.3f} (bottom 10%): {int((per_event_d < pct10).sum())}")

print(f"\n{'='*64}")
print(f"[5] Stage distribution of event-bin rows")
print(f"{'='*64}")
if "stage" in ev.columns:
    print(ev["stage"].value_counts().to_string())

print(f"\n{'='*64}")
print(f"[6] Time distribution of event-bin rows (recording fraction)")
print(f"{'='*64}")
if "window_center_sec" in ev.columns:
    t = ev["window_center_sec"]
    print(f"  time range: {t.min():.0f}s - {t.max():.0f}s  median={t.median():.0f}s")

# Load C3 spindle CSV for detector QC
csv_files = list((DS_ROOT / SUB / "eeg").glob("*spindles_yasa_v1_psg_c3.csv"))
if csv_files:
    sp = pd.read_csv(str(csv_files[0]))
    print(f"\n{'='*64}")
    print(f"[7] C3 detector table summary ({len(sp)} spindles)")
    print(f"{'='*64}")
    for col in ["duration_sec", "yasa_amplitude", "yasa_frequency", "yasa_relpower"]:
        if col in sp.columns:
            pct = np.percentile(sp[col].dropna(), [5, 25, 50, 75, 95])
            print(f"  {col}: p5={pct[0]:.3f} p25={pct[1]:.3f} p50={pct[2]:.3f} "
                  f"p75={pct[3]:.3f} p95={pct[4]:.3f}")
    if "stage" in sp.columns:
        print(f"  stage distribution: {sp['stage'].value_counts().to_dict()}")
else:
    print("\n  [7] No C3 spindle CSV found for sub-5")

print(f"\n{'='*64}")
print(f"[8] Compare sub-5 d absolute level: event bins vs all controls")
print(f"{'='*64}")
print(f"  All event rows d: mean={df[df['condition']=='spindle_event']['d'].mean():.4f}")
print(f"  All control rows d: mean={df[df['condition']=='matched_control']['d'].mean():.4f}")
for b in ["pre_far", "pre_near", "event", "post_near", "post_far"]:
    b_df = df[(df["condition"]=="spindle_event") & (df["bin_label"]==b)]
    print(f"  {b:<12}: n={len(b_df):>5}  d_mean={b_df['d'].mean():>8.4f}  "
          f"d_std={b_df['d'].std():>8.4f}")
