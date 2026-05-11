"""
Detect sleep spindles on ds005555 sub-1 using the C3 channel.

This is a sensitivity/robustness check against the F3 (primary) detection.
Outputs a separate CSV to avoid overwriting the primary F3 annotations.

CLAIM BOUNDARY: These are detector-derived spindles (YASA automatic
detection), NOT ground truth. Use source="detector:yasa-0.7.0" tag.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for pkg in ["mndm/src", "core/src"]:
    p = str(REPO / pkg)
    if p not in sys.path:
        sys.path.insert(0, p)

import mne
import yasa
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EDF_PATH = Path(r"M:\datasets\received\openneuro\ds005555\sub-1\eeg"
                r"\sub-1_task-Sleep_acq-psg_eeg.edf")
EVENTS_TSV = Path(r"M:\datasets\received\openneuro\ds005555\sub-1\eeg"
                  r"\sub-1_task-Sleep_acq-psg_events.tsv")
OUT_CSV = Path(r"M:\datasets\received\openneuro\ds005555\sub-1\eeg"
               r"\sub-1_task-Sleep_acq-psg_spindles_yasa_C3.csv")

TARGET_CHANNELS = ["PSG_C3", "C3", "C3-M2", "C3:A2"]   # try in order

# ---------------------------------------------------------------------------
# Load EDF
# ---------------------------------------------------------------------------
print("Loading EDF:", EDF_PATH)
raw = mne.io.read_raw_edf(str(EDF_PATH), preload=True, verbose=False)
print(f"  Available channels: {raw.ch_names}")
sfreq = raw.info["sfreq"]
print(f"  Sfreq: {sfreq} Hz,  Duration: {raw.times[-1]:.0f} s")

# Find C3 channel
ch_name = None
for candidate in TARGET_CHANNELS:
    if candidate in raw.ch_names:
        ch_name = candidate
        break

if ch_name is None:
    # Fallback: list channels and pick the one with C3 in the name
    c3_matches = [c for c in raw.ch_names if "C3" in c.upper()]
    if c3_matches:
        ch_name = c3_matches[0]
        print(f"  Auto-selected C3 channel: {ch_name}")
    else:
        print(f"  ERROR: No C3 channel found in {raw.ch_names}")
        print(f"  Fallback: using second available EEG channel if present")
        eeg_picks = mne.pick_types(raw.info, eeg=True)
        if len(eeg_picks) >= 2:
            ch_name = raw.ch_names[eeg_picks[1]]
            print(f"  Using: {ch_name}")
        else:
            print("  Cannot proceed: no C3 or alternative channel available.")
            sys.exit(1)

print(f"\nUsing channel: {ch_name}")
idx = raw.ch_names.index(ch_name)
data, _ = raw[idx, :]
data_uv = data[0] * 1e6   # V → µV

# ---------------------------------------------------------------------------
# Load sleep staging (same as F3 run)
# ---------------------------------------------------------------------------
df_events = pd.read_csv(str(EVENTS_TSV), sep="\t")
STAGE_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, -1: -1}
hypno = df_events["stage_hum"].map(lambda x: STAGE_MAP.get(int(x), -1)).to_numpy(dtype=int)

n_epochs_recording = int(np.ceil(len(data_uv) / sfreq / 30))
if len(hypno) < n_epochs_recording:
    hypno = np.concatenate([hypno, np.full(n_epochs_recording - len(hypno), -1)])
elif len(hypno) > n_epochs_recording:
    hypno = hypno[:n_epochs_recording]

hypno_upsampled = yasa.hypno_upsample_to_data(hypno, sf_hypno=1/30, data=data_uv, sf_data=sfreq)
n2_epochs = int((hypno == 2).sum())
n2_h = n2_epochs * 30 / 3600
print(f"Hypnogram: N2 epochs = {n2_epochs} ({n2_h:.2f} h)")

# ---------------------------------------------------------------------------
# Detect spindles on C3
# ---------------------------------------------------------------------------
print("\nRunning YASA spindle detection on", ch_name, "...")
sp = yasa.spindles_detect(
    data_uv,
    sf=sfreq,
    hypno=hypno_upsampled,
    include=(2,),
    freq_sp=(11, 16),
    freq_broad=(1, 30),
    duration=(0.5, 3.0),
    min_distance=500,
    verbose=False,
)

if sp is None:
    print("No spindles detected on", ch_name)
    sys.exit(1)

df = sp.summary()
print(f"Detected {len(df)} spindles on {ch_name}")
print(f"  Rate: {len(df)/n2_h:.1f} /h N2")
print(f"  Duration: {df['Duration'].mean():.3f} ± {df['Duration'].std():.3f} s")

# ---------------------------------------------------------------------------
# Convert to EventTable CSV
# ---------------------------------------------------------------------------
out = pd.DataFrame()
out["onset_sec"]    = df["Start"].astype(float)
out["duration_sec"] = df["Duration"].astype(float)
out["peak_sec"]     = df["Peak"].astype(float)
out["event_type"]   = "sleep_spindle"
out["channel"]      = ch_name
out["confidence"]   = np.nan
out["source"]       = "detector:yasa-0.7.0"
out["stage"]        = df.get("Stage", 2).astype(int)
for col in ["Amplitude", "RMS", "AbsPower", "RelPower", "Frequency", "Oscillations", "Symmetry"]:
    if col in df.columns:
        out[f"yasa_{col.lower()}"] = df[col].astype(float)

out.to_csv(str(OUT_CSV), index=False)
print(f"\nSaved: {OUT_CSV}")
print(f"Columns: {out.columns.tolist()}")
print(f"\nCLAIM BOUNDARY: detector:yasa-0.7.0 on {ch_name}, NOT ground truth.")
