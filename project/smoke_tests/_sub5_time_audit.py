import pandas as pd, numpy as np
from pathlib import Path
f = list(Path(r"M:\datasets\processed\openneuro\ds005555").rglob(
    "sub-5_Sleep_acq-psg_event_locked_v1_psg_c3.parquet"))[-1]
df = pd.read_parquet(str(f))

ev = df[df["condition"]=="spindle_event"]
ct = df[df["condition"]=="matched_control"]

print("== sub-5 C3: time-of-night distribution ==")
for label, data in [("event", ev), ("control", ct)]:
    p = np.percentile(data["window_center_sec"], [10,25,50,75,90])
    print(f"  {label}: p10={p[0]:.0f} p25={p[1]:.0f} p50={p[2]:.0f} p75={p[3]:.0f} p90={p[4]:.0f}")

print()
print("== d by time quartile (event vs control) ==")
all_t = df["window_center_sec"]
qs = np.percentile(all_t, [25, 50, 75])
for label, data in [("event", ev), ("control", ct)]:
    bounds = [(0, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], 1e9)]
    for i, (lo, hi) in enumerate(bounds):
        seg = data[(data["window_center_sec"] >= lo) & (data["window_center_sec"] < hi)]
        if len(seg) == 0:
            continue
        hi_str = f"{hi:.0f}" if hi < 1e9 else "end"
        print(f"  {label} Q{i+1} ({lo:.0f}-{hi_str}s): n={len(seg):>5}  d={seg['d'].mean():.4f}")

print()
print("== d correlation with time (Pearson) ==")
for label, data in [("event", ev), ("control", ct)]:
    r = float(np.corrcoef(data["window_center_sec"], data["d"])[0,1])
    print(f"  {label}: r(t,d) = {r:.4f}")
