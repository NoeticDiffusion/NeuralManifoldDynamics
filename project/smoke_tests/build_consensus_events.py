"""
Build F3 intersect C3 consensus spindle event set.

Selects only spindles where both PSG_F3 and PSG_C3 detect an event
within a configurable temporal tolerance (default: 0.5 s based on
onset times). The consensus event uses F3 as the reference channel
(onset, duration, peak from F3) and records C3 overlap metadata.

This is a stricter event set than either channel alone:
- Reduces single-channel noise candidates
- Requires bilateral cortical expression (frontal + central)
- Reduces event count but improves presumed specificity

Output: a CSV in EventTable format with source="detector:yasa-0.7.0/consensus-F3-C3"

CLAIM BOUNDARY: Detector-derived events, NOT ground truth. Consensus
filtering improves presumed specificity, but does not validate against
polysomnography expert annotation.

Usage
-----
  python build_consensus_events.py [--tolerance_s FLOAT] [--out PATH]
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for pkg in ["mndm/src", "core/src"]:
    p = str(REPO / pkg)
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
F3_CSV = Path(r"M:\datasets\received\openneuro\ds005555\sub-1\eeg"
              r"\sub-1_task-Sleep_acq-psg_spindles_yasa.csv")
C3_CSV = Path(r"M:\datasets\received\openneuro\ds005555\sub-1\eeg"
              r"\sub-1_task-Sleep_acq-psg_spindles_yasa_C3.csv")
OUT_DEFAULT = Path(r"M:\datasets\received\openneuro\ds005555\sub-1\eeg"
                   r"\sub-1_task-Sleep_acq-psg_spindles_yasa_consensus_F3C3.csv")

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Build F3 intersect C3 consensus spindle events")
parser.add_argument("--tolerance_s", type=float, default=0.5,
                    help="Max onset-time difference (s) to consider events co-occurring (default: 0.5)")
parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
parser.add_argument("--f3", type=Path, default=F3_CSV)
parser.add_argument("--c3", type=Path, default=C3_CSV)
args = parser.parse_args()

print(f"F3 CSV: {args.f3}")
print(f"C3 CSV: {args.c3}")
print(f"Tolerance: {args.tolerance_s} s")

if not args.f3.exists():
    print(f"ERROR: F3 CSV not found: {args.f3}")
    sys.exit(1)
if not args.c3.exists():
    print(f"ERROR: C3 CSV not found: {args.c3}")
    print("Run detect_spindles_yasa_c3.py first.")
    sys.exit(1)

df_f3 = pd.read_csv(str(args.f3))
df_c3 = pd.read_csv(str(args.c3))
print(f"\nLoaded: F3={len(df_f3)} events, C3={len(df_c3)} events")

# ---------------------------------------------------------------------------
# Match F3 events to nearest C3 event within tolerance
# ---------------------------------------------------------------------------
f3_onsets = df_f3["onset_sec"].values
c3_onsets = df_c3["onset_sec"].values

matched_f3_idx = []
matched_c3_idx = []
c3_onset_diff = []
c3_duration_match = []

for i, f3_onset in enumerate(f3_onsets):
    diffs = np.abs(c3_onsets - f3_onset)
    nearest_j = int(np.argmin(diffs))
    nearest_diff = float(diffs[nearest_j])
    if nearest_diff <= args.tolerance_s:
        matched_f3_idx.append(i)
        matched_c3_idx.append(nearest_j)
        c3_onset_diff.append(nearest_diff)
        c3_duration_match.append(df_c3["duration_sec"].iloc[nearest_j])

n_matched = len(matched_f3_idx)
print(f"\nConsensus events: {n_matched} / {len(df_f3)} F3 events matched C3 within {args.tolerance_s} s")
print(f"  Unmatched F3 events (dropped): {len(df_f3) - n_matched}")
print(f"  Unique C3 events used: {len(set(matched_c3_idx))} / {len(df_c3)}")

# Check for double-matched C3 events (multiple F3 events matching same C3)
c3_use_counts = pd.Series(matched_c3_idx).value_counts()
n_duplicates = (c3_use_counts > 1).sum()
if n_duplicates > 0:
    print(f"  WARNING: {n_duplicates} C3 events matched by >1 F3 event (may indicate closely-spaced spindles)")

# ---------------------------------------------------------------------------
# Build consensus table
# ---------------------------------------------------------------------------
out_rows = df_f3.iloc[matched_f3_idx].copy().reset_index(drop=True)

# Update source and add consensus metadata
out_rows["source"] = "detector:yasa-0.7.0/consensus-F3-C3"
out_rows["consensus_c3_onset_diff_s"] = c3_onset_diff
out_rows["consensus_c3_duration_s"] = c3_duration_match
out_rows["consensus_tolerance_s"] = args.tolerance_s

# Reorder: standard EventTable columns first
event_table_cols = [
    "onset_sec", "duration_sec", "peak_sec",
    "event_type", "channel", "confidence", "source", "stage",
]
extra_cols = [c for c in out_rows.columns if c not in event_table_cols]
out_rows = out_rows[event_table_cols + extra_cols]

out_rows.to_csv(str(args.out), index=False)
print(f"\nSaved: {args.out}")

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
print("\n--- Consensus event statistics ---")
print(f"  N events: {len(out_rows)}")
if len(out_rows) > 0:
    print(f"  Duration: {out_rows['duration_sec'].mean():.3f} +/- {out_rows['duration_sec'].std():.3f} s")
    if "yasa_frequency" in out_rows.columns:
        print(f"  Frequency: {out_rows['yasa_frequency'].mean():.2f} Hz")
    if "yasa_amplitude" in out_rows.columns:
        print(f"  Amplitude: {out_rows['yasa_amplitude'].mean():.1f} uV")
    print(f"  C3 onset diff: {np.mean(c3_onset_diff):.3f} +/- {np.std(c3_onset_diff):.3f} s (mean +/- SD)")

print(f"\n  Retention rate vs F3 alone: {100*n_matched/len(df_f3):.1f}%")
print(f"  Note: lower retention = stricter/higher-precision event set")

# Compute implied rate (requires N2 duration from events.tsv)
try:
    EVENTS_TSV = Path(r"M:\datasets\received\openneuro\ds005555\sub-1\eeg"
                      r"\sub-1_task-Sleep_acq-psg_events.tsv")
    df_ev = pd.read_csv(str(EVENTS_TSV), sep="\t")
    stages = df_ev["stage_hum"].astype(str).map(
        {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "-1": -1}
    ).fillna(-1).astype(int)
    n2_h = (stages == 2).sum() * 30 / 3600
    rate = len(out_rows) / n2_h if n2_h > 0 else float("nan")
    print(f"  Consensus rate: {rate:.1f} /h N2  (N2={n2_h:.2f} h)")
    if 5 <= rate <= 20:
        print(f"  VERDICT: Rate {rate:.1f}/h is within plausible range (5-20/h). Suitable for exploratory use.")
    elif rate < 5:
        print(f"  VERDICT: Rate {rate:.1f}/h is below typical range. May be too strict or too few real spindles.")
    else:
        print(f"  VERDICT: Rate {rate:.1f}/h still above plausible range. Consider stricter tolerance or parameter tuning.")
except Exception as exc:
    print(f"  (Could not compute rate: {exc})")

print("\nCLAIM BOUNDARY: detector:yasa-0.7.0/consensus-F3-C3. NOT ground truth.")
print("Consensus filtering improves presumed specificity; biological interpretation deferred.")
