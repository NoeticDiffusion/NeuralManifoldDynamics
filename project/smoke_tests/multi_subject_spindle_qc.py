"""
Multi-subject spindle density QC — ds005555.

Reports spindles/minute per channel per sleep stage, following the
Purcell et al. 2017 (Nature Comm., N=11 630) reference metric:
  N2: ~1.88/min (central channels; range 0.70–4.80/min per PLOS One)
  N3: ~1.45/min

Runs two parameter sets side-by-side:
  DEFAULT  : YASA defaults (freq_sp=12-15, rel_pow=0.20, corr=0.65,
             rms=1.5, min_dist=500 ms, remove_outliers=False)
  CALIBRATED: stricter (rel_pow=0.30, corr=0.70, rms=3.0, min_dist=700 ms,
             remove_outliers=True)

Both on PSG_F3, N2 only (primary) and N3 separately for completeness.

CLAIM BOUNDARY: detector-derived events; calibration comparison only.
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
# Configuration
# ---------------------------------------------------------------------------
DS_ROOT   = Path(r"M:\datasets\received\openneuro\ds005555")
SUBJECTS  = ["sub-1", "sub-2", "sub-3", "sub-4", "sub-5"]
CHANNEL   = "PSG_F3"
OUT_CSV   = DS_ROOT / "multi_subject_spindle_qc.csv"

# Reference thresholds from Purcell 2017 + PLOS One validation
# (central channels; frontal may differ slightly)
REF_N2_MIN, REF_N2_MAX = 0.70, 4.80   # /min, PLOS One acceptable range
REF_N2_CENTER          = 1.88          # /min, Purcell 2017 mean
REF_N3_MIN, REF_N3_MAX = 0.80, 2.00

PARAMS = {
    "DEFAULT": dict(
        freq_sp=(12, 15), freq_broad=(1, 30), duration=(0.5, 3.0),
        min_distance=500, thresh={"rel_pow": 0.20, "corr": 0.65, "rms": 1.5},
        remove_outliers=False, verbose=False,
    ),
    "CALIBRATED": dict(
        freq_sp=(12, 15), freq_broad=(1, 30), duration=(0.5, 3.0),
        min_distance=700, thresh={"rel_pow": 0.30, "corr": 0.70, "rms": 3.0},
        remove_outliers=True, verbose=False,
    ),
}

STAGE_INT = {2: "N2", 3: "N3"}     # YASA hypnogram integers

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_subject(sub_id: str):
    """Return (data_uv, sfreq, hypno_int) or None on failure."""
    base = DS_ROOT / sub_id / "eeg"
    edf  = base / f"{sub_id}_task-Sleep_acq-psg_eeg.edf"
    tsv  = base / f"{sub_id}_task-Sleep_acq-psg_events.tsv"
    if not edf.exists() or not tsv.exists():
        return None, None, None
    raw = mne.io.read_raw_edf(str(edf), preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    if CHANNEL not in raw.ch_names:
        return None, sfreq, None
    idx = raw.ch_names.index(CHANNEL)
    data, _ = raw[idx, :]
    data_uv = data[0] * 1e6
    df_ev = pd.read_csv(str(tsv), sep="\t")
    if "stage_hum" not in df_ev.columns:
        return None, sfreq, None
    SM = {0:0,1:1,2:2,3:3,4:4,-1:-1}
    hypno = df_ev["stage_hum"].map(lambda x: SM.get(int(x), -1)).to_numpy(int)
    n_rec = int(np.ceil(len(data_uv) / sfreq / 30))
    if len(hypno) < n_rec:
        hypno = np.concatenate([hypno, np.full(n_rec - len(hypno), -1)])
    else:
        hypno = hypno[:n_rec]
    return data_uv, sfreq, hypno


def detect_per_stage(data_uv, sfreq, hypno, param_set, include_stages=(2, 3)):
    """Run YASA for each stage separately; return {stage_int: DataFrame}."""
    results = {}
    for stage_int in include_stages:
        hypno_up = yasa.hypno_upsample_to_data(
            hypno, sf_hypno=1/30, data=data_uv, sf_data=sfreq
        )
        sp = yasa.spindles_detect(
            data_uv, sf=sfreq, hypno=hypno_up,
            include=(stage_int,), **param_set,
        )
        results[stage_int] = sp.summary() if sp is not None else pd.DataFrame()
    return results


def stage_duration_min(hypno, stage_int):
    return float((hypno == stage_int).sum()) * 30 / 60

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print("=" * 72)
print("MULTI-SUBJECT SPINDLE DENSITY QC  [spindles/min per stage]")
print(f"Reference: Purcell 2017 N2={REF_N2_CENTER}/min; range {REF_N2_MIN}-{REF_N2_MAX}/min")
print(f"Channel: {CHANNEL},  Subjects: {SUBJECTS}")
print("=" * 72)

rows = []
for sub in SUBJECTS:
    print(f"\nLoading {sub}...", end=" ", flush=True)
    data_uv, sfreq, hypno = load_subject(sub)
    if data_uv is None:
        print("SKIP (missing EDF or staging)")
        continue
    n2_min = stage_duration_min(hypno, 2)
    n3_min = stage_duration_min(hypno, 3)
    print(f"N2={n2_min:.0f}min  N3={n3_min:.0f}min")

    for pname, params in PARAMS.items():
        per_stage = detect_per_stage(data_uv, sfreq, hypno, params)
        for stage_int, stage_name in STAGE_INT.items():
            df_sp = per_stage[stage_int]
            n = len(df_sp)
            dur_min = n2_min if stage_int == 2 else n3_min
            rate_min = n / dur_min if dur_min > 0 else np.nan
            rate_h   = rate_min * 60
            dur_mean = df_sp["Duration"].mean() if len(df_sp) > 0 else np.nan
            freq_m   = df_sp["Frequency"].mean() if "Frequency" in df_sp.columns and len(df_sp) > 0 else np.nan

            if stage_int == 2:
                in_range = REF_N2_MIN <= rate_min <= REF_N2_MAX
                tag = "OK" if in_range else ("LOW" if rate_min < REF_N2_MIN else "HIGH")
            else:
                in_range = REF_N3_MIN <= rate_min <= REF_N3_MAX
                tag = "OK" if in_range else ("LOW" if rate_min < REF_N3_MIN else "HIGH")

            rows.append({
                "subject": sub, "params": pname, "stage": stage_name,
                "n_spindles": n,
                "stage_dur_min": round(dur_min, 1),
                "rate_per_min": round(rate_min, 3) if np.isfinite(rate_min) else None,
                "rate_per_h":   round(rate_h, 1)   if np.isfinite(rate_h)   else None,
                "dur_mean_s":   round(dur_mean, 3)  if np.isfinite(dur_mean) else None,
                "freq_hz":      round(freq_m, 2)    if np.isfinite(freq_m)   else None,
                "in_range": in_range, "tag": tag,
            })
            print(f"  {pname:<12} {stage_name}: n={n:4d}  {rate_min:5.2f}/min  "
                  f"({rate_h:6.1f}/h)  dur={dur_mean:.2f}s  [{tag}]")

df_res = pd.DataFrame(rows)
df_res.to_csv(str(OUT_CSV), index=False)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("SUMMARY — mean rate across subjects")
print("=" * 72)
print(f"  {'Params':<14} {'Stage':<5} {'Mean /min':>10}  {'Range /min':>16}  "
      f"{'Ref center':>11}  {'N in-range':>11}")
print(f"  {'-'*14} {'-'*5} {'-'*10}  {'-'*16}  {'-'*11}  {'-'*11}")
for pname in PARAMS:
    for stage_name in ["N2", "N3"]:
        sub_df = df_res[(df_res["params"] == pname) & (df_res["stage"] == stage_name)]
        rates = sub_df["rate_per_min"].dropna()
        if len(rates) == 0:
            continue
        ref_c = REF_N2_CENTER if stage_name == "N2" else 1.45
        n_ok  = (sub_df["tag"] == "OK").sum()
        print(f"  {pname:<14} {stage_name:<5} {rates.mean():>10.2f}  "
              f"[{rates.min():.2f}–{rates.max():.2f}]     "
              f"{ref_c:>8.2f}/min   {n_ok}/{len(sub_df)} subjects")

print(f"\nFull results saved: {OUT_CSV}")

# Key conclusion
print("\n" + "-" * 72)
print("CALIBRATION ASSESSMENT")
print("-" * 72)
n2_default    = df_res[(df_res["params"]=="DEFAULT")    & (df_res["stage"]=="N2")]["rate_per_min"].mean()
n2_calibrated = df_res[(df_res["params"]=="CALIBRATED") & (df_res["stage"]=="N2")]["rate_per_min"].mean()
print(f"  DEFAULT    N2 mean: {n2_default:.2f}/min  "
      f"({'OK' if REF_N2_MIN<=n2_default<=REF_N2_MAX else 'OUTSIDE'} ref {REF_N2_MIN}-{REF_N2_MAX}/min)")
print(f"  CALIBRATED N2 mean: {n2_calibrated:.2f}/min  "
      f"({'OK' if REF_N2_MIN<=n2_calibrated<=REF_N2_MAX else 'OUTSIDE'} ref {REF_N2_MIN}-{REF_N2_MAX}/min)")
print(f"  Purcell 2017 center (C3/C4): {REF_N2_CENTER}/min")
print(f"\nCLAIM BOUNDARY: detector-derived events; reference from Purcell 2017 (N=11630).")
print("These are PSG_F3 (frontal), not C3/C4 (central). Frontal rates may differ.")
