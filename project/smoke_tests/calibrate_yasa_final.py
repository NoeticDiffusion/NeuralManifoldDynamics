"""
Final YASA calibration pass: remove_outliers + calibrated params + consensus.

Three strategies tested:
  A. Best single-channel params + remove_outliers=True (F3)
  B. Best single-channel params + consensus F3 intersect C3 (0.3s)
  C. Combination: remove_outliers + consensus

Target: 5-20 /h N2.

CLAIM BOUNDARY: detector calibration only.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for pkg in ["mndm/src", "core/src"]:
    p = str(REPO / pkg)
    if p not in sys.path:
        sys.path.insert(0, p)

import mne, yasa
import numpy as np
import pandas as pd

EDF_PATH   = Path(r"M:\datasets\received\openneuro\ds005555\sub-1\eeg"
                  r"\sub-1_task-Sleep_acq-psg_eeg.edf")
EVENTS_TSV = Path(r"M:\datasets\received\openneuro\ds005555\sub-1\eeg"
                  r"\sub-1_task-Sleep_acq-psg_events.tsv")
OUT_DIR    = Path(r"M:\datasets\received\openneuro\ds005555\sub-1\eeg")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading EDF...")
raw = mne.io.read_raw_edf(str(EDF_PATH), preload=True, verbose=False)
sfreq = raw.info["sfreq"]

def get_chan(name):
    idx = raw.ch_names.index(name)
    data, _ = raw[idx, :]
    return data[0] * 1e6

data_f3 = get_chan("PSG_F3")
data_c3 = get_chan("PSG_C3")

df_ev = pd.read_csv(str(EVENTS_TSV), sep="\t")
hypno = df_ev["stage_hum"].map(lambda x: {0:0,1:1,2:2,3:3,4:4,-1:-1}.get(int(x),-1)).to_numpy(int)
n_rec = int(np.ceil(len(data_f3) / sfreq / 30))
if len(hypno) < n_rec:
    hypno = np.concatenate([hypno, np.full(n_rec - len(hypno), -1)])
else:
    hypno = hypno[:n_rec]

hyp_f3 = yasa.hypno_upsample_to_data(hypno, sf_hypno=1/30, data=data_f3, sf_data=sfreq)
hyp_c3 = yasa.hypno_upsample_to_data(hypno, sf_hypno=1/30, data=data_c3, sf_data=sfreq)
n2_h = (hypno == 2).sum() * 30 / 3600
print(f"N2: {n2_h:.2f} h\n")

# ---------------------------------------------------------------------------
# Best single-channel params from targeted sweep
# rel_pow=0.30, corr=0.70, rms=2.5, min_distance=700, freq_sp=(12,15)
# → 70 events, 21.1/h  (just above range)
# ---------------------------------------------------------------------------
BEST_THRESH = {"rel_pow": 0.30, "corr": 0.70, "rms": 2.5}
BEST_PARAMS = dict(freq_sp=(12,15), freq_broad=(1,30), duration=(0.5,3.0),
                   min_distance=700, thresh=BEST_THRESH, include=(2,), verbose=False)

def detect(data_uv, hypno_up, **kwargs):
    sp = yasa.spindles_detect(data_uv, sf=sfreq, hypno=hypno_up, **kwargs)
    return sp.summary() if sp is not None else pd.DataFrame()

def consensus(df_a, df_b, tolerance_s=0.3):
    """Return rows from df_a that have a df_b event within tolerance_s."""
    a_on = df_a["Start"].values if "Start" in df_a.columns else df_a["onset_sec"].values
    b_on = df_b["Start"].values if "Start" in df_b.columns else df_b["onset_sec"].values
    keep = []
    for i, a in enumerate(a_on):
        if len(b_on) > 0 and np.min(np.abs(b_on - a)) <= tolerance_s:
            keep.append(i)
    return df_a.iloc[keep].reset_index(drop=True)

def rate(df):
    return len(df) / n2_h

# ---------------------------------------------------------------------------
# Strategy A: best params + remove_outliers=True (F3 only)
# ---------------------------------------------------------------------------
print("Strategy A: best params + remove_outliers=True (F3)...")
df_a = detect(data_f3, hyp_f3, remove_outliers=True, **BEST_PARAMS)
print(f"  A: n={len(df_a)}, rate={rate(df_a):.1f}/h N2")

# Also try progressively stricter rms with remove_outliers
for rms_val in [3.0, 3.5, 4.0, 5.0]:
    thr = {**BEST_THRESH, "rms": rms_val}
    p = {**BEST_PARAMS, "thresh": thr, "remove_outliers": True}
    df_t = detect(data_f3, hyp_f3, **p)
    tag = "IN-RANGE" if 5 <= rate(df_t) <= 20 else "HIGH" if rate(df_t) > 20 else "LOW"
    print(f"  A (rms={rms_val}): n={len(df_t)}, rate={rate(df_t):.1f}/h  [{tag}]")
    if 5 <= rate(df_t) <= 20:
        df_a_best = df_t
        BEST_RMS = rms_val
        break
else:
    df_a_best = df_a
    BEST_RMS = 2.5

# ---------------------------------------------------------------------------
# Strategy B: best params, consensus F3+C3 (no remove_outliers)
# ---------------------------------------------------------------------------
print("\nStrategy B: best params, consensus F3+C3 (tol=0.3s)...")
df_b_f3 = detect(data_f3, hyp_f3, **BEST_PARAMS)
df_b_c3 = detect(data_c3, hyp_c3, **BEST_PARAMS)
print(f"  B-F3 alone: n={len(df_b_f3)}, rate={rate(df_b_f3):.1f}/h")
print(f"  B-C3 alone: n={len(df_b_c3)}, rate={rate(df_b_c3):.1f}/h")
df_b_cons = consensus(df_b_f3, df_b_c3, tolerance_s=0.3)
tag_b = "IN-RANGE" if 5 <= rate(df_b_cons) <= 20 else "HIGH" if rate(df_b_cons) > 20 else "LOW"
print(f"  B-consensus: n={len(df_b_cons)}, rate={rate(df_b_cons):.1f}/h  [{tag_b}]")

# ---------------------------------------------------------------------------
# Strategy C: remove_outliers + consensus
# ---------------------------------------------------------------------------
print("\nStrategy C: remove_outliers + consensus F3+C3...")
df_c_f3 = detect(data_f3, hyp_f3, remove_outliers=True, **BEST_PARAMS)
df_c_c3 = detect(data_c3, hyp_c3, remove_outliers=True, **BEST_PARAMS)
print(f"  C-F3 alone: n={len(df_c_f3)}, rate={rate(df_c_f3):.1f}/h")
print(f"  C-C3 alone: n={len(df_c_c3)}, rate={rate(df_c_c3):.1f}/h")
df_c_cons = consensus(df_c_f3, df_c_c3, tolerance_s=0.3)
tag_c = "IN-RANGE" if 5 <= rate(df_c_cons) <= 20 else "HIGH" if rate(df_c_cons) > 20 else "LOW"
print(f"  C-consensus: n={len(df_c_cons)}, rate={rate(df_c_cons):.1f}/h  [{tag_c}]")

# ---------------------------------------------------------------------------
# Select best strategy
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("CALIBRATION SUMMARY")
print("="*60)

candidates = [
    ("A-remove_outliers", df_a_best, rate(df_a_best)),
    ("B-consensus", df_b_cons, rate(df_b_cons)),
    ("C-remove_outliers+consensus", df_c_cons, rate(df_c_cons)),
]

for name, df, r in candidates:
    tag = "IN-RANGE" if 5 <= r <= 20 else "HIGH" if r > 20 else "LOW"
    print(f"  {name:<40}: n={len(df):4d}, rate={r:5.1f}/h  [{tag}]")

best_candidates = [(n, df, r) for n, df, r in candidates if 5 <= r <= 20]
if best_candidates:
    # Pick closest to center of range (12.5/h)
    chosen_name, chosen_df, chosen_rate = min(best_candidates, key=lambda x: abs(x[2] - 12.5))
    print(f"\nChosen strategy: {chosen_name}  ({chosen_rate:.1f}/h)")
else:
    # None in range: pick lowest rate
    chosen_name, chosen_df, chosen_rate = min(candidates, key=lambda x: x[2])
    print(f"\nNo strategy in 5-20/h range. Using lowest-rate: {chosen_name} ({chosen_rate:.1f}/h)")
    print("Recommendation: accept as best-available; flag as above-range in claims.")

# Save the chosen event set
if len(chosen_df) > 0:
    out = pd.DataFrame()
    col_map = {"Start": "onset_sec", "Duration": "duration_sec", "Peak": "peak_sec"}
    for yasa_col, et_col in col_map.items():
        if yasa_col in chosen_df.columns:
            out[et_col] = chosen_df[yasa_col].astype(float)
    out["event_type"]  = "sleep_spindle"
    out["channel"]     = "PSG_F3"
    out["confidence"]  = np.nan
    strategy_tag = chosen_name.replace("+", "_").replace("-", "_")
    out["source"] = f"detector:yasa-0.7.0/calibrated/{strategy_tag}"
    out["stage"] = 2
    for col in ["Amplitude","RMS","AbsPower","RelPower","Frequency","Oscillations","Symmetry"]:
        if col in chosen_df.columns:
            out[f"yasa_{col.lower()}"] = chosen_df[col].astype(float)

    fname = f"sub-1_task-Sleep_acq-psg_spindles_yasa_{strategy_tag}.csv"
    OUT_PATH = OUT_DIR / fname
    out.to_csv(str(OUT_PATH), index=False)
    print(f"\nCalibrated events saved: {OUT_PATH}")
    print(f"  N={len(out)}, rate={chosen_rate:.1f}/h, source={out['source'].iloc[0]}")

print("\nCLAIM: calibration run only. Biological interpretation deferred.")
