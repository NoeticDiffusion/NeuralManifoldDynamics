"""
Calibrated sleep spindle detection on ds005555 sub-1, PSG_F3.

Parameters calibrated by sweep (see calibrate_yasa_targeted.py,
calibrate_yasa_final.py) to produce a rate within the plausible
range for healthy adult N2 sleep (5–20/h).

Calibration result
------------------
  freq_sp      = (12, 15) Hz   [narrow sigma band]
  rel_pow      = 0.30          [stricter than default 0.20]
  corr         = 0.70          [stricter than default 0.65]
  rms          = 3.0           [stricter than default 1.5]
  min_distance = 700 ms        [wider than default 500 ms]
  remove_outliers = True       [key parameter: removes feature-space outliers]

  Result: 54 events, 16.3/h N2  (target: 5–20/h)

Previous overpermissive baseline
---------------------------------
  Default YASA params → 288 events, 86.8/h  (DQF-001, rejected)

CLAIM BOUNDARY: Detector-derived events (YASA 0.7.0), NOT ground truth.
Calibration targets plausible rate, not biological validation.
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
EDF_PATH   = Path(r"M:\datasets\received\openneuro\ds005555\sub-1\eeg"
                  r"\sub-1_task-Sleep_acq-psg_eeg.edf")
EVENTS_TSV = Path(r"M:\datasets\received\openneuro\ds005555\sub-1\eeg"
                  r"\sub-1_task-Sleep_acq-psg_events.tsv")
OUT_CSV    = Path(r"M:\datasets\received\openneuro\ds005555\sub-1\eeg"
                  r"\sub-1_task-Sleep_acq-psg_spindles_yasa_calibrated.csv")

# ---------------------------------------------------------------------------
# Calibrated parameters (do not change without re-running sweep + claim update)
# ---------------------------------------------------------------------------
CALIBRATED_PARAMS = dict(
    freq_sp       = (12, 15),
    freq_broad    = (1, 30),
    duration      = (0.5, 3.0),
    min_distance  = 700,
    thresh        = {"rel_pow": 0.30, "corr": 0.70, "rms": 3.0},
    remove_outliers = True,
    include       = (2,),         # N2 only
    verbose       = False,
)

# Human-readable label for the source column
SOURCE_TAG = (
    "detector:yasa-0.7.0/calibrated"
    "/freq12-15Hz_relp0.30_corr0.70_rms3.0_md700ms_rmoutliers"
)

# ---------------------------------------------------------------------------
# Load EDF
# ---------------------------------------------------------------------------
print("Loading EDF:", EDF_PATH)
raw = mne.io.read_raw_edf(str(EDF_PATH), preload=True, verbose=False)
sfreq = raw.info["sfreq"]
print(f"  Channels: {raw.ch_names}")
print(f"  Duration: {raw.times[-1]:.0f} s  ({raw.times[-1]/3600:.2f} h)")
print(f"  Sfreq:    {sfreq} Hz")

ch_name = "PSG_F3"
idx = raw.ch_names.index(ch_name)
data, _ = raw[idx, :]
data_uv = data[0] * 1e6     # V → µV

# ---------------------------------------------------------------------------
# Load staging
# ---------------------------------------------------------------------------
df_ev = pd.read_csv(str(EVENTS_TSV), sep="\t")
STAGE_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, -1: -1}
hypno = df_ev["stage_hum"].map(lambda x: STAGE_MAP.get(int(x), -1)).to_numpy(dtype=int)
n_rec = int(np.ceil(len(data_uv) / sfreq / 30))
if len(hypno) < n_rec:
    hypno = np.concatenate([hypno, np.full(n_rec - len(hypno), -1)])
elif len(hypno) > n_rec:
    hypno = hypno[:n_rec]

hypno_up = yasa.hypno_upsample_to_data(hypno, sf_hypno=1/30, data=data_uv, sf_data=sfreq)
n2_epochs = int((hypno == 2).sum())
n2_h = n2_epochs * 30 / 3600
print(f"\nN2 epochs: {n2_epochs}  ({n2_h:.2f} h)")

# ---------------------------------------------------------------------------
# Detect spindles
# ---------------------------------------------------------------------------
print(f"\nRunning calibrated YASA detection on {ch_name}...")
print(f"  freq_sp={CALIBRATED_PARAMS['freq_sp']} Hz")
print(f"  thresh={CALIBRATED_PARAMS['thresh']}")
print(f"  min_distance={CALIBRATED_PARAMS['min_distance']} ms")
print(f"  remove_outliers={CALIBRATED_PARAMS['remove_outliers']}")

sp = yasa.spindles_detect(data_uv, sf=sfreq, hypno=hypno_up, **CALIBRATED_PARAMS)

if sp is None:
    print("No spindles detected.")
    sys.exit(1)

df = sp.summary()
n = len(df)
rate = n / n2_h
print(f"\nDetected {n} spindles  ({rate:.1f}/h N2)")
if 5 <= rate <= 20:
    print(f"  VERDICT: Rate {rate:.1f}/h is within plausible range (5-20/h).  OK")
elif rate < 5:
    print(f"  WARNING: Rate {rate:.1f}/h is below typical range — possible under-detection.")
else:
    print(f"  WARNING: Rate {rate:.1f}/h exceeds plausible range — parameters may need further tuning.")

print(f"\n  Duration: {df['Duration'].mean():.3f} +/- {df['Duration'].std():.3f} s")
if "Frequency" in df.columns:
    print(f"  Frequency: {df['Frequency'].mean():.2f} Hz")
if "Amplitude" in df.columns:
    print(f"  Amplitude: {df['Amplitude'].mean():.1f} uV")

# ---------------------------------------------------------------------------
# Save EventTable CSV
# ---------------------------------------------------------------------------
out = pd.DataFrame()
out["onset_sec"]    = df["Start"].astype(float)
out["duration_sec"] = df["Duration"].astype(float)
out["peak_sec"]     = df["Peak"].astype(float)
out["event_type"]   = "sleep_spindle"
out["channel"]      = ch_name
out["confidence"]   = np.nan
out["source"]       = SOURCE_TAG
out["stage"]        = df.get("Stage", 2).astype(int)
for col in ["Amplitude", "RMS", "AbsPower", "RelPower", "Frequency", "Oscillations", "Symmetry"]:
    if col in df.columns:
        out[f"yasa_{col.lower()}"] = df[col].astype(float)

out.to_csv(str(OUT_CSV), index=False)
print(f"\nSaved: {OUT_CSV}")
print(f"Columns: {out.columns.tolist()}")

print(f"\nCLAIM BOUNDARY:")
print(f"  Source tag:  {SOURCE_TAG}")
print(f"  Rate:        {rate:.1f}/h N2 — within calibrated plausible range")
print(f"  NOT ground truth. Biological interpretation requires replication and expert review.")
