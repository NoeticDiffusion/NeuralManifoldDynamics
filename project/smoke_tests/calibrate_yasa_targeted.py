"""
Targeted YASA calibration: 12 focused parameter combinations.

Based on the full sweep being too slow (>10 min for 288 combos on this EDF),
this targeted run tests the most informative combinations to quickly identify
a calibrated parameter set giving 5-20/h N2.

Strategy:
- Fix freq_sp=(12,15) (narrower sigma band, more specific than 11-16)
- Vary rms=1.5, 2.0, 2.5, 3.0 (the most sensitive threshold)
- Vary min_distance=500, 1000 ms (to separate burst detections)
- Fix rel_pow=0.25 (slightly stricter than default 0.2)
- Fix corr=0.65 (keep default)
- Channel: PSG_F3 only (then apply consensus filter separately)

CLAIM BOUNDARY: detector calibration only, not biological analysis.
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

# PSG_F3
idx_f3 = raw.ch_names.index("PSG_F3")
data_f3, _ = raw[idx_f3, :]
data_f3_uv = data_f3[0] * 1e6

# Staging
df_ev = pd.read_csv(str(EVENTS_TSV), sep="\t")
hypno_base = df_ev["stage_hum"].map(lambda x: {0:0,1:1,2:2,3:3,4:4,-1:-1}.get(int(x),-1)).to_numpy(int)
n_rec = int(np.ceil(len(data_f3_uv) / sfreq / 30))
if len(hypno_base) < n_rec:
    hypno_base = np.concatenate([hypno_base, np.full(n_rec - len(hypno_base), -1)])
elif len(hypno_base) > n_rec:
    hypno_base = hypno_base[:n_rec]
hypno_up = yasa.hypno_upsample_to_data(hypno_base, sf_hypno=1/30, data=data_f3_uv, sf_data=sfreq)

n2_h = (hypno_base == 2).sum() * 30 / 3600
print(f"N2 duration: {n2_h:.2f} h")

# ---------------------------------------------------------------------------
# Targeted parameter grid — 12 combos
# ---------------------------------------------------------------------------
# (freq_sp, rel_pow, corr, rms, min_distance_ms)
COMBOS = [
    # Narrow sigma band + progressive rms escalation
    ((12, 15), 0.25, 0.65, 2.0, 500),
    ((12, 15), 0.25, 0.65, 2.5, 500),
    ((12, 15), 0.25, 0.65, 3.0, 500),
    ((12, 15), 0.25, 0.65, 3.5, 500),
    # Increase min_distance to reduce burst clusters
    ((12, 15), 0.25, 0.65, 2.0, 1000),
    ((12, 15), 0.25, 0.65, 2.5, 1000),
    ((12, 15), 0.25, 0.65, 3.0, 1000),
    ((12, 15), 0.30, 0.65, 2.0, 1000),
    ((12, 15), 0.30, 0.65, 2.5, 1000),
    ((12, 15), 0.30, 0.65, 3.0, 1000),
    # Stricter correlation
    ((12, 15), 0.25, 0.70, 2.5, 700),
    ((12, 15), 0.30, 0.70, 2.5, 700),
]

print(f"\nRunning {len(COMBOS)} targeted combinations (PSG_F3 only)...")
print(f"Target rate: 5-20 /h N2\n")

results = []
for freq_sp, rel_pow, corr, rms, min_dist in COMBOS:
    sp = yasa.spindles_detect(
        data_f3_uv, sf=sfreq, hypno=hypno_up,
        include=(2,), freq_sp=freq_sp, freq_broad=(1, 30),
        duration=(0.5, 3.0), min_distance=min_dist,
        thresh={"rel_pow": rel_pow, "corr": corr, "rms": rms},
        verbose=False,
    )
    if sp is not None:
        df = sp.summary()
        n = len(df)
        rate = n / n2_h
        dur  = df["Duration"].mean()
        freq = df["Frequency"].mean() if "Frequency" in df.columns else np.nan
        amp  = df["Amplitude"].mean() if "Amplitude" in df.columns else np.nan
        tag  = "IN-RANGE" if 5 <= rate <= 20 else ("LOW" if rate < 5 else "HIGH")
    else:
        n, rate, dur, freq, amp, tag = 0, 0.0, np.nan, np.nan, np.nan, "LOW"

    results.append({
        "freq_sp": str(freq_sp), "rel_pow": rel_pow, "corr": corr,
        "rms": rms, "min_dist_ms": min_dist,
        "n": n, "rate_h": round(rate, 1), "dur_s": round(dur, 3) if np.isfinite(dur) else None,
        "freq_hz": round(freq, 2) if np.isfinite(freq) else None,
        "amp_uv": round(amp, 1) if np.isfinite(amp) else None,
        "status": tag,
    })
    print(f"  rms={rms:.1f} rel_pow={rel_pow} corr={corr} min_dist={min_dist:4d} | "
          f"n={n:4d} rate={rate:5.1f}/h  [{tag}]")

df_res = pd.DataFrame(results)

print("\n" + "="*65)
print("TARGETED SWEEP RESULTS")
print("="*65)
in_range = df_res[df_res["status"] == "IN-RANGE"]
print(f"In-range combos (5-20/h): {len(in_range)}")
if len(in_range) > 0:
    best = in_range.loc[in_range["rate_h"].sub(12.5).abs().idxmin()]
    print(f"\nBest calibrated parameter set (closest to 12.5/h center):")
    for k, v in best.items():
        print(f"  {k}: {v}")

    # Save calibrated detections using best parameters
    best_sp = yasa.spindles_detect(
        data_f3_uv, sf=sfreq, hypno=hypno_up,
        include=(2,),
        freq_sp=eval(best["freq_sp"]),
        freq_broad=(1, 30), duration=(0.5, 3.0),
        min_distance=int(best["min_dist_ms"]),
        thresh={"rel_pow": best["rel_pow"], "corr": best["corr"], "rms": best["rms"]},
        verbose=False,
    )
    if best_sp is not None:
        df_best = best_sp.summary()
        out = pd.DataFrame()
        out["onset_sec"]    = df_best["Start"].astype(float)
        out["duration_sec"] = df_best["Duration"].astype(float)
        out["peak_sec"]     = df_best["Peak"].astype(float)
        out["event_type"]   = "sleep_spindle"
        out["channel"]      = "PSG_F3"
        out["confidence"]   = np.nan
        params_tag = (f"rms{best['rms']}_relp{best['rel_pow']}_"
                      f"corr{best['corr']}_md{int(best['min_dist_ms'])}ms_"
                      f"fsp{''.join(str(x) for x in eval(best['freq_sp']))}Hz")
        out["source"]       = f"detector:yasa-0.7.0/calibrated/{params_tag}"
        out["stage"]        = df_best.get("Stage", 2).astype(int)
        for col in ["Amplitude","RMS","AbsPower","RelPower","Frequency","Oscillations","Symmetry"]:
            if col in df_best.columns:
                out[f"yasa_{col.lower()}"] = df_best[col].astype(float)

        OUT_PATH = OUT_DIR / f"sub-1_task-Sleep_acq-psg_spindles_yasa_calibrated_{params_tag}.csv"
        out.to_csv(str(OUT_PATH), index=False)
        print(f"\nCalibrated detections saved: {OUT_PATH}")
        print(f"N={len(out)}, rate={best['rate_h']}/h N2")
else:
    print("\nNo in-range combination found. Summary:")
    print(df_res[["rms","rel_pow","min_dist_ms","n","rate_h","status"]].to_string(index=False))
    print("\nConsider: consensus F3+C3 filter reduces rate by ~4x (87 -> 20/h at 0.3s tolerance).")
    print("Combining calibrated single-channel with consensus may reach 5-20/h range.")

print("\nCLAIM: calibration only. No biological interpretation.")
