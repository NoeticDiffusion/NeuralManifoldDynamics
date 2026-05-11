"""
F3 vs C3 channel robustness comparison for event-locked MNPS.

Reads per-channel Parquet files produced by batch_event_locked.py,
computes event-bin delta (event - matched_control) per subject and channel,
then reports direction consistency across subjects and F3/C3 agreement.

EXPLORATORY -- no inferential statistics.

Promotion criterion (as specified by architect):
  >= 4/5 subjects agree in direction for at least one MNPS dimension,
  OR coherent F3/C3 agreement across subjects.

Usage:
  python compare_channels.py [--channels PSG_F3 PSG_C3] [--subjects ...]
"""

from __future__ import annotations

import argparse
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
SUBJECTS  = ["sub-1", "sub-2", "sub-3", "sub-4", "sub-5"]
MNPS_DIMS = ["m", "d", "e"]
EVENT_BIN = "event"

parser = argparse.ArgumentParser()
parser.add_argument("--channels", nargs="+", default=["PSG_F3", "PSG_C3"])
parser.add_argument("--subjects", nargs="+", default=SUBJECTS)
parser.add_argument("--bin", default=EVENT_BIN,
                    help="Bin to use for event delta (default: event)")
args = parser.parse_args()

CHANNELS = args.channels
SUBJECTS = args.subjects
BIN      = args.bin


def channel_slug(ch: str) -> str:
    return ch.lower().replace(" ", "_")


def load_channel(channel: str) -> pd.DataFrame | None:
    slug = channel_slug(channel)
    dfs = []
    for sub in SUBJECTS:
        files = list(PROC_ROOT.rglob(
            f"{sub}_Sleep_acq-psg_event_locked_v1_{slug}.parquet"
        ))
        if not files:
            print(f"  [{channel}] {sub}: no parquet found")
            continue
        df = pd.read_parquet(str(files[-1]))
        df["subject"] = sub
        df["channel"] = channel
        dfs.append(df)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def event_delta(df: pd.DataFrame, bin_label: str) -> pd.DataFrame:
    """Per-subject delta: event bin mean - matched_control mean for each MNPS dim."""
    rows = []
    for sub in df["subject"].unique():
        s = df[df["subject"] == sub]
        ev = s[(s["condition"] == "spindle_event") & (s["bin_label"] == bin_label)]
        ct = s[s["condition"] == "matched_control"]
        if len(ev) == 0 or len(ct) == 0:
            continue
        row = {"subject": sub, "n_event": len(ev), "n_control": len(ct)}
        for dim in MNPS_DIMS:
            row[f"ev_{dim}"]    = ev[dim].mean()
            row[f"ct_{dim}"]    = ct[dim].mean()
            row[f"delta_{dim}"] = ev[dim].mean() - ct[dim].mean()
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print("=" * 72)
print(f"CHANNEL ROBUSTNESS: F3 vs C3 -- bin='{BIN}'  [EXPLORATORY]")
print(f"Protocol v1 | YASA 0.7.0 | N2 only | ds005555")
print("=" * 72)

channel_data: dict[str, pd.DataFrame] = {}
channel_deltas: dict[str, pd.DataFrame] = {}

for ch in CHANNELS:
    print(f"\nLoading {ch}...")
    df = load_channel(ch)
    if df is None:
        print(f"  {ch}: no data — skipping")
        continue
    channel_data[ch] = df
    channel_deltas[ch] = event_delta(df, BIN)
    print(f"  {ch}: {len(df)} rows, {df['subject'].nunique()} subjects")

if not channel_deltas:
    print("No data loaded.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Per-channel: spindle rate and QC
# ---------------------------------------------------------------------------
print("\n" + "="*72)
print("[1] Spindle rates and event counts per subject per channel")
print("="*72)
print(f"\n  {'Subject':<10}  " + "  ".join(f"{ch:<18}" for ch in CHANNELS))
print(f"  {'-'*10}  " + "  ".join(f"{'-'*18}" for _ in CHANNELS))

for sub in SUBJECTS:
    parts = []
    for ch in CHANNELS:
        if ch not in channel_data:
            parts.append(f"{'NO DATA':<18}")
            continue
        df = channel_data[ch]
        s = df[df["subject"] == sub]
        n_ev = len(s[s["condition"] == "spindle_event"])
        n_ct = len(s[s["condition"] == "matched_control"])
        if n_ev == 0:
            parts.append(f"{'(no events)':<18}")
        else:
            parts.append(f"{n_ev:>5} ev / {n_ct:>5} ct  ")
    print(f"  {sub:<10}  " + "  ".join(parts))

# ---------------------------------------------------------------------------
# Per-channel direction table
# ---------------------------------------------------------------------------
print("\n" + "="*72)
print(f"[2] Event-bin delta per subject per channel (event_mean - control_mean)")
print("="*72)

for ch in CHANNELS:
    if ch not in channel_deltas:
        continue
    d = channel_deltas[ch]
    print(f"\n  -- {ch} --")
    print(f"  {'Subject':<10}  {'dm':>8} {'dd':>8} {'de':>8}  {'n_ev':>6}")
    print(f"  {'-'*10}  {'-'*8} {'-'*8} {'-'*8}  {'-'*6}")
    for _, row in d.iterrows():
        print(f"  {row['subject']:<10}  "
              f"{row['delta_m']:>+8.4f} {row['delta_d']:>+8.4f} {row['delta_e']:>+8.4f}  "
              f"{int(row['n_event']):>6}")

# ---------------------------------------------------------------------------
# Direction consistency per channel
# ---------------------------------------------------------------------------
print("\n" + "="*72)
print("[3] Direction agreement per channel (fraction of subjects with same sign)")
print("="*72)

consistency: dict[str, dict[str, tuple]] = {}  # channel -> dim -> (frac_pos, dominant)
for ch in CHANNELS:
    if ch not in channel_deltas:
        continue
    d = channel_deltas[ch]
    print(f"\n  -- {ch} --")
    consistency[ch] = {}
    for dim in MNPS_DIMS:
        col = f"delta_{dim}"
        n_pos = int((d[col] > 0).sum())
        n_neg = int((d[col] < 0).sum())
        n_sub = len(d)
        frac_pos = n_pos / n_sub if n_sub > 0 else 0
        dominant = "event>ctrl" if n_pos >= n_neg else "event<ctrl"
        frac_agree = max(n_pos, n_neg) / n_sub if n_sub > 0 else 0
        flag = "<<" if frac_agree >= 0.8 else ("< " if frac_agree >= 0.6 else "  ")
        print(f"  {dim}: {n_pos}/{n_sub} positive  {dominant:>12}  agreement={frac_agree:.0%} {flag}")
        consistency[ch][dim] = (frac_agree, dominant, n_pos, n_sub)

# ---------------------------------------------------------------------------
# Cross-channel F3/C3 agreement per subject per dim
# ---------------------------------------------------------------------------
if len(channel_deltas) >= 2:
    chs = list(channel_deltas.keys())
    ch1, ch2 = chs[0], chs[1]
    d1 = channel_deltas[ch1].set_index("subject")
    d2 = channel_deltas[ch2].set_index("subject")
    common_subs = sorted(set(d1.index) & set(d2.index))

    print("\n" + "="*72)
    print(f"[4] Cross-channel direction agreement: {ch1} vs {ch2}")
    print("="*72)
    print(f"\n  {'Subject':<10}  " + "  ".join(
        f"{'delta_'+dim+':':>6} {ch1[:4]:>5}/{ch2[:4]:>5}" for dim in MNPS_DIMS
    ))
    print(f"  {'-'*10}  " + "  ".join(f"{'-'*18}" for dim in MNPS_DIMS))

    agree_counts = {dim: 0 for dim in MNPS_DIMS}
    for sub in common_subs:
        parts = []
        for dim in MNPS_DIMS:
            v1 = d1.loc[sub, f"delta_{dim}"]
            v2 = d2.loc[sub, f"delta_{dim}"]
            agree = (v1 > 0) == (v2 > 0)
            if agree:
                agree_counts[dim] += 1
            s1 = "+" if v1 > 0 else "-"
            s2 = "+" if v2 > 0 else "-"
            tag = "ok" if agree else "XX"
            parts.append(f"{s1}{abs(v1):.3f}/{s2}{abs(v2):.3f} {tag}")
        print(f"  {sub:<10}  " + "  ".join(parts))

    print(f"\n  Cross-channel agreement (same sign in both channels):")
    for dim in MNPS_DIMS:
        n = len(common_subs)
        frac = agree_counts[dim] / n if n > 0 else 0
        flag = "<<" if frac >= 0.8 else ("< " if frac >= 0.6 else "  ")
        print(f"  {dim}: {agree_counts[dim]}/{n} agree  ({frac:.0%}) {flag}")

# ---------------------------------------------------------------------------
# Promotion decision
# ---------------------------------------------------------------------------
print("\n" + "="*72)
print("[5] Promotion check (>= 4/5 agreement OR coherent F3/C3 agreement)")
print("="*72)

promoted = []
for ch in CHANNELS:
    if ch not in consistency:
        continue
    for dim in MNPS_DIMS:
        frac_agree, dominant, n_pos, n_sub = consistency[ch][dim]
        if n_sub >= 4 and frac_agree >= 4/5:
            promoted.append(f"{ch}/{dim}: {n_pos}/{n_sub} ({dominant}, {frac_agree:.0%})")

if len(channel_deltas) >= 2:
    for dim in MNPS_DIMS:
        n = len(common_subs)
        frac = agree_counts[dim] / n if n > 0 else 0
        if n >= 4 and frac >= 4/5:
            promoted.append(f"F3+C3/{dim}: cross-channel agreement {agree_counts[dim]}/{n} ({frac:.0%})")

if promoted:
    print("\n  PROMOTED patterns (meeting >= 4/5 threshold):")
    for p in promoted:
        print(f"    [*] {p}")
else:
    print("\n  NO promoted patterns.")
    print("  Threshold not met: no dimension shows >= 4/5 direction agreement")
    print("  in any single channel or cross-channel combination.")
    print("\n  Correct claim: PSG_F3 and PSG_C3 do not show a stable MNPS")
    print("  spindle signature at the group level on this dataset/detector.")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_rows = []
for ch in CHANNELS:
    if ch not in channel_deltas:
        continue
    d = channel_deltas[ch].copy()
    d["channel"] = ch
    out_rows.append(d)

if out_rows:
    df_out = pd.concat(out_rows, ignore_index=True)
    out_path = PROC_ROOT / "channel_comparison_deltas.csv"
    df_out.to_csv(str(out_path), index=False)
    print(f"\n  Delta table saved: {out_path}")

print(f"\n{'='*72}")
print("EXPLORATORY -- no inferential statistics. Claim: measurement differences only.")
print(f"{'='*72}")
