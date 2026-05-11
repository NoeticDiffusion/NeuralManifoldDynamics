"""
Detect sleep spindles on ds005555 sub-1 PSG EDF using YASA.

Output: a CSV matching the EventTable contract:
  onset_sec, duration_sec, peak_sec, event_type, channel, confidence, source, stage

IMPORTANT CLAIM BOUNDARY: These are detector-derived spindles (YASA automatic
detection), NOT ground truth. They must be labeled accordingly. Any downstream
analysis comparing spindle vs. non-spindle MNPS must treat detector errors as
a known limitation.
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
               r"\sub-1_task-Sleep_acq-psg_spindles_yasa.csv")

# ---------------------------------------------------------------------------
# Load EDF
# ---------------------------------------------------------------------------
print("Loading EDF:", EDF_PATH)
raw = mne.io.read_raw_edf(str(EDF_PATH), preload=True, verbose=False)
print(f"  Channels: {raw.ch_names}")
print(f"  Duration: {raw.times[-1]:.1f} s ({raw.times[-1]/3600:.2f} h)")
print(f"  Sfreq:    {raw.info['sfreq']} Hz")

# ---------------------------------------------------------------------------
# Pick EEG-like channels
# ---------------------------------------------------------------------------
# ds005555 PSG has limited channels; pick what's available
eeg_picks = mne.pick_types(raw.info, eeg=True)
if len(eeg_picks) == 0:
    # Fallback: pick any non-stim channel
    eeg_picks = mne.pick_types(raw.info, exclude=["stim"])
print(f"  EEG picks ({len(eeg_picks)}): {[raw.ch_names[i] for i in eeg_picks]}")

# Use first EEG channel for spindle detection
ch_name = raw.ch_names[eeg_picks[0]]
data, times = raw[eeg_picks[0], :]  # shape (1, n_times)
data_uv = data[0] * 1e6             # V → µV

sfreq = raw.info["sfreq"]

# ---------------------------------------------------------------------------
# Load sleep staging
# ---------------------------------------------------------------------------
df_events = pd.read_csv(str(EVENTS_TSV), sep="\t")
print(f"\nStage annotations: {len(df_events)} epochs")

# Build hypnogram array aligned to EEG samples (1 value per 30-s epoch).
# YASA expects hypnogram in 30-s epochs; -1 = unscored.
STAGE_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, -1: -1}  # stage_hum is already int
hypno = df_events["stage_hum"].map(lambda x: STAGE_MAP.get(int(x), -1)).to_numpy(dtype=int)

# Pad or trim to match recording length in 30 s epochs
n_epochs_recording = int(np.ceil(len(data_uv) / sfreq / 30))
if len(hypno) < n_epochs_recording:
    hypno = np.concatenate([hypno, np.full(n_epochs_recording - len(hypno), -1)])
elif len(hypno) > n_epochs_recording:
    hypno = hypno[:n_epochs_recording]

# Upsample hypnogram to per-sample resolution (for YASA)
hypno_upsampled = yasa.hypno_upsample_to_data(hypno, sf_hypno=1/30, data=data_uv, sf_data=sfreq)
print(f"Hypnogram upsampled: {len(hypno_upsampled)} samples, n_N2 epochs = {(hypno == 2).sum()}")

# ---------------------------------------------------------------------------
# Detect spindles
# ---------------------------------------------------------------------------
print("\nRunning YASA spindle detection...")
sp = yasa.spindles_detect(
    data_uv,
    sf=sfreq,
    hypno=hypno_upsampled,
    include=(2,),          # N2 only
    freq_sp=(11, 16),      # sigma band
    freq_broad=(1, 30),
    duration=(0.5, 3.0),   # spindle duration range
    min_distance=500,      # ms between spindles
    verbose=False,
)

if sp is None:
    print("No spindles detected.")
    sys.exit(1)

df = sp.summary()
print(f"\nDetected {len(df)} spindles")
print(df.head())
print(f"Columns: {df.columns.tolist()}")

# ---------------------------------------------------------------------------
# Convert to EventTable CSV format
# ---------------------------------------------------------------------------
# YASA columns: Start, Peak, End, Duration, Amplitude, RMS, AbsPower,
#               RelPower, Frequency, Oscillations, Symmetry, Stage, Channel
out = pd.DataFrame()
out["onset_sec"]    = df["Start"].astype(float)
out["duration_sec"] = df["Duration"].astype(float)
out["peak_sec"]     = df["Peak"].astype(float)
out["event_type"]   = "sleep_spindle"
out["channel"]      = df.get("Channel", ch_name)
out["confidence"]   = np.nan          # YASA doesn't output per-spindle probability
out["source"]       = f"detector:yasa-0.7.0"
out["stage"]        = df.get("Stage", 2).astype(int)
# Extra YASA columns for reference (stored as metadata)
for col in ["Amplitude", "RMS", "AbsPower", "RelPower", "Frequency", "Oscillations", "Symmetry"]:
    if col in df.columns:
        out[f"yasa_{col.lower()}"] = df[col].astype(float)

out.to_csv(str(OUT_CSV), index=False)
print(f"\nSaved: {OUT_CSV}")
print(f"Columns: {out.columns.tolist()}")
print(f"\nSpindle summary:")
print(f"  N spindles: {len(out)}")
print(f"  Duration:   {out['duration_sec'].describe()[['mean','min','max']].round(3).to_dict()}")
print(f"  Onset range: {out['onset_sec'].min():.1f} – {out['onset_sec'].max():.1f} s")
print(f"  Channel:    {out['channel'].unique()}")
print(f"\nCLAIM BOUNDARY: detector:yasa-0.7.0, NOT ground truth.")
