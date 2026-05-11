"""
QC report for YASA detector-derived spindle annotations.

Produces a minimal text report covering:
  - N spindles detected, rate per hour of N2
  - Duration distribution (mean, std, min, p25, median, p75, max, histogram)
  - Channel breakdown
  - Stage distribution (should be N2-only since we ran include=(2,))
  - Exclusion reasons from event-locked alignment
  - Whether 46 transition exclusions are a reasonable fraction

Input files
-----------
  SPINDLE_CSV   : YASA output in EventTable format
  EVENTS_TSV    : BIDS events.tsv with sleep staging
  EXPORT_PARQUET: event-locked export (to retrieve alignment QC)

CLAIM BOUNDARY: All findings are for detector-derived events (YASA 0.7.0),
not ground-truth spindles.
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SPINDLE_CSV = Path(r"M:\datasets\received\openneuro\ds005555\sub-1\eeg"
                   r"\sub-1_task-Sleep_acq-psg_spindles_yasa.csv")
EVENTS_TSV = Path(r"M:\datasets\received\openneuro\ds005555\sub-1\eeg"
                  r"\sub-1_task-Sleep_acq-psg_events.tsv")
EXPORT_PARQUET = Path(r"M:\datasets\processed\mndm\ds005555"
                      r"\sub-1_task-Sleep_acq-psg_event_locked.parquet")

# Also look for C3 CSV if already generated
C3_CSV = Path(r"M:\datasets\received\openneuro\ds005555\sub-1\eeg"
              r"\sub-1_task-Sleep_acq-psg_spindles_yasa_C3.csv")

# ---------------------------------------------------------------------------
# Load spindle CSV
# ---------------------------------------------------------------------------
print("=" * 60)
print("YASA SPINDLE DETECTION QC REPORT")
print("Source: detector:yasa-0.7.0  |  NOT ground truth")
print("=" * 60)

df_sp = pd.read_csv(str(SPINDLE_CSV))
print(f"\n[1] SPINDLE COUNT AND RATE")
print(f"  N spindles (F3):    {len(df_sp)}")
print(f"  Source column:      {df_sp['source'].unique().tolist()}")
print(f"  Channel(s):         {df_sp['channel'].unique().tolist()}")
print(f"  Stage value(s):     {df_sp['stage'].unique().tolist()}")

# ---------------------------------------------------------------------------
# N2 duration from events.tsv
# ---------------------------------------------------------------------------
df_ev = pd.read_csv(str(EVENTS_TSV), sep="\t")
STAGE_MAP = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "-1": -1}
# Accept both int and string stage columns
if "stage_hum" in df_ev.columns:
    stages = df_ev["stage_hum"].astype(str).map(STAGE_MAP).fillna(-1).astype(int)
elif "value" in df_ev.columns:
    stages = df_ev["value"].astype(str).map(STAGE_MAP).fillna(-1).astype(int)
else:
    stages = pd.Series([], dtype=int)

n2_epochs = int((stages == 2).sum())
n2_duration_sec = n2_epochs * 30.0
n2_duration_h = n2_duration_sec / 3600.0

print(f"\n  N2 epochs (×30 s):  {n2_epochs}")
print(f"  N2 duration:        {n2_duration_sec:.0f} s  ({n2_duration_h:.2f} h)")
if n2_duration_h > 0:
    rate_per_h = len(df_sp) / n2_duration_h
    print(f"  Spindle rate:       {rate_per_h:.1f} / h of N2  "
          f"(typical range: 5-15 /h in healthy adults)")
    if rate_per_h < 4:
        print("  NOTE: Rate < 4/h — possible under-detection (low-amplitude signal or wrong channel)")
    elif rate_per_h > 20:
        print("  NOTE: Rate > 20/h — possible over-detection or very clean N2 signal")

# ---------------------------------------------------------------------------
# Duration distribution
# ---------------------------------------------------------------------------
print(f"\n[2] DURATION DISTRIBUTION (seconds)")
dur = df_sp["duration_sec"]
desc = dur.describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90])
print(f"  Mean ± Std:   {dur.mean():.3f} ± {dur.std():.3f}")
print(f"  Min / Max:    {dur.min():.3f} / {dur.max():.3f}")
print(f"  p10 / p25:    {desc['10%']:.3f} / {desc['25%']:.3f}")
print(f"  Median:       {desc['50%']:.3f}")
print(f"  p75 / p90:    {desc['75%']:.3f} / {desc['90%']:.3f}")

# ASCII histogram of duration
bins = np.arange(0.4, 3.2, 0.2)
hist, edges = np.histogram(dur.values, bins=bins)
print(f"\n  Duration histogram (bin width = 0.2 s):")
for i, count in enumerate(hist):
    bar = "#" * min(count, 40)
    print(f"  [{edges[i]:.1f}–{edges[i+1]:.1f})  {bar}  ({count})")

# ---------------------------------------------------------------------------
# YASA amplitude / frequency if present
# ---------------------------------------------------------------------------
if "yasa_frequency" in df_sp.columns:
    freq = df_sp["yasa_frequency"]
    print(f"\n[3] SPINDLE FREQUENCY (Hz)")
    print(f"  Mean ± Std:   {freq.mean():.2f} ± {freq.std():.2f}")
    print(f"  Min / Max:    {freq.min():.2f} / {freq.max():.2f}")

if "yasa_amplitude" in df_sp.columns:
    amp = df_sp["yasa_amplitude"]
    print(f"\n[4] SPINDLE AMPLITUDE (µV)")
    print(f"  Mean ± Std:   {amp.mean():.1f} ± {amp.std():.1f}")
    print(f"  Min / Max:    {amp.min():.1f} / {amp.max():.1f}")

# ---------------------------------------------------------------------------
# Temporal distribution across night
# ---------------------------------------------------------------------------
print(f"\n[5] TEMPORAL DISTRIBUTION")
onset = df_sp["onset_sec"]
recording_h = onset.max() / 3600
print(f"  Onset range:  {onset.min():.0f} – {onset.max():.0f} s  ({recording_h:.2f} h)")
quarter_s = (onset.max() - onset.min()) / 4
for q in range(4):
    lo = onset.min() + q * quarter_s
    hi = lo + quarter_s
    n_q = ((onset >= lo) & (onset < hi)).sum()
    print(f"  Q{q+1} [{lo/3600:.2f}h–{hi/3600:.2f}h):  {n_q} spindles")

# ---------------------------------------------------------------------------
# Alignment QC (from export parquet, if available)
# ---------------------------------------------------------------------------
print(f"\n[6] ALIGNMENT QC (from event-locked export)")
if EXPORT_PARQUET.exists():
    df_ex = pd.read_parquet(str(EXPORT_PARQUET))
    print(f"  Export rows:   {len(df_ex)}")
    
    if "condition" in df_ex.columns:
        cond_counts = df_ex["condition"].value_counts()
        print(f"  Condition counts:")
        for cond, cnt in cond_counts.items():
            print(f"    {cond}: {cnt} rows")

    # Bin coverage
    if "bin_label" in df_ex.columns:
        bin_counts = df_ex["bin_label"].value_counts().sort_index()
        print(f"  Bin coverage:")
        for b, cnt in bin_counts.items():
            print(f"    {b}: {cnt} rows")

    # Stage distribution
    if "stage" in df_ex.columns:
        stage_counts = df_ex["stage"].value_counts().sort_index()
        print(f"  Stage distribution in export:")
        for s, cnt in stage_counts.items():
            print(f"    stage {s}: {cnt} rows")

    # Exclusion summary
    if "qc_excluded_stage_transition" in df_ex.columns:
        n_excl_trans = df_ex[df_ex["condition"] == "spindle_event"]["qc_excluded_stage_transition"].sum() \
            if "condition" in df_ex.columns else 0
        print(f"  Stage-transition exclusions: {n_excl_trans}")
    
    n_spindle_rows = len(df_ex[df_ex["condition"] == "spindle_event"]) if "condition" in df_ex.columns else "?"
    n_ctrl_rows    = len(df_ex[df_ex["condition"] == "matched_control"]) if "condition" in df_ex.columns else "?"
    print(f"  Spindle event rows: {n_spindle_rows}")
    print(f"  Control rows:       {n_ctrl_rows}")

    # Assess the 46 transition exclusions
    n_detected = len(df_sp)
    n_spindle_events_in_export = (
        df_ex[df_ex["condition"] == "spindle_event"]["event_id"].nunique()
        if "condition" in df_ex.columns and "event_id" in df_ex.columns
        else None
    )
    if n_spindle_events_in_export is not None:
        n_excluded = n_detected - n_spindle_events_in_export
        pct_excl = 100 * n_excluded / n_detected if n_detected > 0 else 0
        print(f"\n  Exclusion assessment:")
        print(f"    Detected:          {n_detected}")
        print(f"    Aligned to export: {n_spindle_events_in_export}")
        print(f"    Excluded:          {n_excluded}  ({pct_excl:.1f}%)")
        if pct_excl < 20:
            print(f"    VERDICT: Exclusion rate {pct_excl:.1f}% is within acceptable range (< 20%).")
        elif pct_excl < 40:
            print(f"    VERDICT: Exclusion rate {pct_excl:.1f}% is moderate — check stage-transition margin.")
        else:
            print(f"    VERDICT: Exclusion rate {pct_excl:.1f}% is high — review margin or staging quality.")
else:
    print(f"  (Export Parquet not found at {EXPORT_PARQUET})")
    print(f"  Known from smoke test: 46 excluded (stage transitions), 242 aligned, 242/288 = 84% retained")
    n_detected = len(df_sp)
    n_excluded = 46
    n_aligned = n_detected - n_excluded
    pct_excl = 100 * n_excluded / n_detected
    print(f"\n  Exclusion assessment (from smoke test log):")
    print(f"    Detected:    {n_detected}")
    print(f"    Excluded:    {n_excluded}  ({pct_excl:.1f}%)")
    print(f"    Aligned:     {n_aligned}")
    if pct_excl < 20:
        print(f"    VERDICT: {pct_excl:.1f}% exclusion is within acceptable range (< 20%).")
    else:
        print(f"    VERDICT: {pct_excl:.1f}% exclusion — check stage-transition margin.")

# ---------------------------------------------------------------------------
# C3 comparison (if available)
# ---------------------------------------------------------------------------
print(f"\n[7] CHANNEL COMPARISON")
print(f"  F3 (primary):  {len(df_sp)} spindles")
if C3_CSV.exists():
    df_c3 = pd.read_csv(str(C3_CSV))
    print(f"  C3 (secondary): {len(df_c3)} spindles")
    ratio = len(df_c3) / len(df_sp) if len(df_sp) > 0 else float("nan")
    print(f"  C3/F3 ratio:   {ratio:.2f}  (expect ~0.7–1.3 for homologous channels)")
    if "yasa_frequency" in df_c3.columns and "yasa_frequency" in df_sp.columns:
        print(f"  Freq F3: {df_sp['yasa_frequency'].mean():.2f} Hz  |  C3: {df_c3['yasa_frequency'].mean():.2f} Hz")
    # Overlap: spindles within 1s of each other
    n_overlapping = 0
    for onset_c3 in df_c3["onset_sec"].values:
        if any(abs(df_sp["onset_sec"].values - onset_c3) < 1.0):
            n_overlapping += 1
    pct_overlap = 100 * n_overlapping / len(df_c3) if len(df_c3) > 0 else 0
    print(f"  Temporal overlap (C3 within 1s of F3): {n_overlapping}/{len(df_c3)}  ({pct_overlap:.1f}%)")
else:
    print(f"  C3: not yet detected (run detect_spindles_yasa_c3.py)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"  Detector:     YASA 0.7.0  (detector-derived, NOT ground truth)")
print(f"  Channel:      F3 (PSG)")
print(f"  N spindles:   {len(df_sp)}")
if n2_duration_h > 0:
    print(f"  Rate:         {len(df_sp)/n2_duration_h:.1f} /h N2")
print(f"  Duration:     {dur.mean():.2f} ± {dur.std():.2f} s (mean ± SD)")
print(f"  Stage filter: N2 only (as specified)")
print(f"{'='*60}")
print(f"\nCLAIM: detector-derived events; no inference about ground-truth spindle biology.")
