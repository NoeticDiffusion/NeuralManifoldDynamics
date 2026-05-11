"""
YASA spindle detector calibration sweep — ds005555 sub-1.

Sweeps detection thresholds to find parameter combinations that produce
spindle rates in the plausible range (5–20/h of N2) for this PSG dataset.

Target: freq_sp=(12,15) (default) or broader (11,16); various thresh combos.

Outputs a CSV ranking all parameter combinations by rate.

CLAIM BOUNDARY: All detections are detector-derived events, not ground truth.
This sweep is a calibration step, not biological analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path
from itertools import product

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
                  r"\yasa_sweep_results.csv")

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------
# Default YASA: rel_pow=0.2, corr=0.65, rms=1.5, min_distance=500
# We want stricter thresholds. We also test freq_sp=(12,15) vs (11,16).
SWEEP = {
    "freq_sp":      [(11, 16), (12, 15)],
    "rel_pow":      [0.20, 0.25, 0.30],
    "corr":         [0.65, 0.70],
    "rms":          [1.5, 2.0, 2.5, 3.0],
    "min_distance": [500, 700, 1000],    # ms
}

TARGET_RATE_MIN, TARGET_RATE_MAX = 5.0, 20.0

# ---------------------------------------------------------------------------
# Load EDF and staging once
# ---------------------------------------------------------------------------
print("Loading EDF:", EDF_PATH)
raw = mne.io.read_raw_edf(str(EDF_PATH), preload=True, verbose=False)
sfreq = raw.info["sfreq"]
print(f"  Channels: {raw.ch_names}")

# Use both PSG_F3 and PSG_C3
channels_to_sweep = {}
for cname in ["PSG_F3", "PSG_C3"]:
    if cname in raw.ch_names:
        idx = raw.ch_names.index(cname)
        data, _ = raw[idx, :]
        channels_to_sweep[cname] = data[0] * 1e6  # V → µV

if not channels_to_sweep:
    print("No PSG_F3 or PSG_C3 found.")
    sys.exit(1)
print(f"  Sweeping channels: {list(channels_to_sweep.keys())}")

# Staging
df_events = pd.read_csv(str(EVENTS_TSV), sep="\t")
STAGE_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, -1: -1}
hypno_base = df_events["stage_hum"].map(lambda x: STAGE_MAP.get(int(x), -1)).to_numpy(dtype=int)
n2_epochs = int((hypno_base == 2).sum())
n2_duration_h = n2_epochs * 30 / 3600.0
print(f"  N2 duration: {n2_duration_h:.2f} h  ({n2_epochs} epochs)")

def make_hypno(data_uv: np.ndarray) -> np.ndarray:
    n_rec = int(np.ceil(len(data_uv) / sfreq / 30))
    hypno = hypno_base.copy()
    if len(hypno) < n_rec:
        hypno = np.concatenate([hypno, np.full(n_rec - len(hypno), -1)])
    elif len(hypno) > n_rec:
        hypno = hypno[:n_rec]
    return yasa.hypno_upsample_to_data(hypno, sf_hypno=1/30, data=data_uv, sf_data=sfreq)

# Pre-compute hypnograms
hypno_up = {ch: make_hypno(data) for ch, data in channels_to_sweep.items()}

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
results = []
combos = list(product(
    SWEEP["freq_sp"],
    SWEEP["rel_pow"],
    SWEEP["corr"],
    SWEEP["rms"],
    SWEEP["min_distance"],
))
n_total = len(combos) * len(channels_to_sweep)
print(f"\nSweeping {len(combos)} parameter combos × {len(channels_to_sweep)} channels = {n_total} runs")

for ch_name, data_uv in channels_to_sweep.items():
    hyp = hypno_up[ch_name]
    for i, (freq_sp, rel_pow, corr, rms, min_dist) in enumerate(combos):
        try:
            sp = yasa.spindles_detect(
                data_uv,
                sf=sfreq,
                hypno=hyp,
                include=(2,),
                freq_sp=freq_sp,
                freq_broad=(1, 30),
                duration=(0.5, 3.0),
                min_distance=min_dist,
                thresh={"rel_pow": rel_pow, "corr": corr, "rms": rms},
                verbose=False,
            )
            if sp is not None:
                df = sp.summary()
                n = len(df)
                rate = n / n2_duration_h
                dur_mean = df["Duration"].mean()
                dur_std  = df["Duration"].std()
                freq_mean = df["Frequency"].mean() if "Frequency" in df.columns else np.nan
                amp_mean  = df["Amplitude"].mean() if "Amplitude" in df.columns else np.nan
                in_range  = TARGET_RATE_MIN <= rate <= TARGET_RATE_MAX
            else:
                n, rate, dur_mean, dur_std, freq_mean, amp_mean, in_range = 0, 0, np.nan, np.nan, np.nan, np.nan, False
        except Exception as exc:
            n, rate, dur_mean, dur_std, freq_mean, amp_mean, in_range = -1, -1, np.nan, np.nan, np.nan, np.nan, False
            print(f"  ERROR: ch={ch_name} freq_sp={freq_sp} rel_pow={rel_pow} corr={corr} rms={rms} min_dist={min_dist}: {exc}")

        results.append({
            "channel":      ch_name,
            "freq_sp":      str(freq_sp),
            "rel_pow":      rel_pow,
            "corr":         corr,
            "rms":          rms,
            "min_distance": min_dist,
            "n_spindles":   n,
            "rate_per_h":   round(rate, 1),
            "dur_mean_s":   round(dur_mean, 3) if np.isfinite(dur_mean) else np.nan,
            "dur_std_s":    round(dur_std, 3)  if np.isfinite(dur_std)  else np.nan,
            "freq_hz":      round(freq_mean, 2) if np.isfinite(freq_mean) else np.nan,
            "amp_uv":       round(amp_mean, 1)  if np.isfinite(amp_mean)  else np.nan,
            "in_range_5_20": in_range,
        })

        if (i + 1) % 20 == 0 or i == len(combos) - 1:
            done = (i + 1) + combos.index((freq_sp, rel_pow, corr, rms, min_dist)) // len(combos)
            print(f"  [{ch_name}] combo {i+1}/{len(combos)}: "
                  f"rms={rms} rel_pow={rel_pow} n={n} rate={rate:.1f}/h", end="\r")

print()

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
df_res = pd.DataFrame(results)
df_res.to_csv(str(OUT_CSV), index=False)
print(f"\nSweep results saved: {OUT_CSV}  ({len(df_res)} rows)")

# Show in-range results
in_range = df_res[df_res["in_range_5_20"] == True].copy()
print(f"\nParameter combos with rate in [{TARGET_RATE_MIN}–{TARGET_RATE_MAX}] /h N2: {len(in_range)}")

if len(in_range) > 0:
    print("\nTop 20 in-range results (by rate ascending):")
    top = in_range.sort_values("rate_per_h").head(20)
    cols = ["channel", "freq_sp", "rel_pow", "corr", "rms", "min_distance",
            "n_spindles", "rate_per_h", "dur_mean_s", "freq_hz", "amp_uv"]
    print(top[cols].to_string(index=False))

    # Recommend: pick the combination closest to center of range (12.5/h)
    in_range["dist_from_center"] = abs(in_range["rate_per_h"] - 12.5)
    best = in_range.sort_values("dist_from_center").iloc[0]
    print(f"\nRecommended calibrated parameter set (closest to 12.5/h center):")
    for col in ["channel", "freq_sp", "rel_pow", "corr", "rms", "min_distance",
                "n_spindles", "rate_per_h", "dur_mean_s", "freq_hz"]:
        print(f"  {col}: {best[col]}")
else:
    print("No parameter combination achieved the target range. Consider:")
    print("  - Increasing rms threshold further (>3.0)")
    print("  - Narrowing freq_sp to (12,15)")
    print("  - Applying channel-consensus filtering post-detection")
    # Show the least-permissive results anyway
    print("\nLeast-permissive results (lowest rate, any channel):")
    print(df_res.sort_values("rate_per_h").head(10)[
        ["channel", "rel_pow", "corr", "rms", "min_distance", "n_spindles", "rate_per_h"]
    ].to_string(index=False))

print("\nCLAIM BOUNDARY: sweep determines calibrated detection parameters only.")
print("Biological interpretation deferred until calibrated rate achieved and replicated.")
