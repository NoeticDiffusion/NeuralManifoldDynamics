"""
Baseline-corrected event-bin delta for all subjects.
Usage: python _baseline_corrected_all.py [--channel PSG_C3|PSG_F3]
"""
import argparse, json, numpy as np, pandas as pd
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--channel", default="PSG_C3")
args = ap.parse_args()

PROC   = Path(r"M:\datasets\processed\openneuro\ds005555")
SLUG   = args.channel.lower().replace("-", "_")
DIMS   = ["m", "d", "e"]
BINS   = ["pre_far", "pre_near", "event", "post_near", "post_far"]
QC_F   = PROC / f"batch_event_locked_qc_{SLUG}.json"

qc_map = {q["subject"]: q for q in json.load(open(QC_F))}

rows = []
skipped = 0
# Deduplicate: for each subject keep only the most recent parquet (by mtime).
all_parquets = list(PROC.rglob(f"*_Sleep_acq-psg_event_locked_v1_{SLUG}.parquet"))
latest: dict[str, Path] = {}
for p in all_parquets:
    sub = p.name.split("_Sleep")[0]
    if sub not in latest or p.stat().st_mtime > latest[sub].stat().st_mtime:
        latest[sub] = p
for sub, f in sorted(latest.items(), key=lambda kv: int(kv[0].split("-")[1])):
    q   = qc_map.get(sub, {})
    qc_pass = q.get("qc_pass", False)
    df  = pd.read_parquet(str(f))
    ev  = df[df["condition"] == "spindle_event"]
    if ev.empty:
        skipped += 1
        continue
    pre_far_means = {dim: ev[ev["bin_label"]=="pre_far"][dim].mean() for dim in DIMS}
    for b in BINS:
        b_df = ev[ev["bin_label"] == b]
        if b_df.empty:
            continue
        row = {"subject": sub, "bin": b, "n": len(b_df),
               "qc_pass": qc_pass, "rate_per_min": q.get("rate_per_min", np.nan)}
        for dim in DIMS:
            row[f"{dim}_mean"] = b_df[dim].mean()
            row[f"{dim}_bc"]   = b_df[dim].mean() - pre_far_means[dim]
        rows.append(row)

df_all = pd.DataFrame(rows)
n_subs = df_all["subject"].nunique()
print(f"Loaded {n_subs} subjects ({skipped} skipped), {len(df_all)} bin-rows")

# QC-passing subjects only
df_pass = df_all[df_all["qc_pass"]]
n_pass  = df_pass["subject"].nunique()
print(f"QC-passing subjects: {n_pass}")

# Event-bin baseline-corrected direction consistency
print(f"\n=== Event-bin baseline-corrected direction (event bin, QC-pass subjects) ===")
ev_pass = df_pass[df_pass["bin"] == "event"]
for dim in DIMS:
    n_pos = int((ev_pass[f"{dim}_bc"] > 0).sum())
    n_neg = int((ev_pass[f"{dim}_bc"] < 0).sum())
    n_tot = len(ev_pass)
    dominant = "event>baseline" if n_pos > n_neg else "event<baseline"
    frac = max(n_pos, n_neg) / n_tot
    med  = ev_pass[f"{dim}_bc"].median()
    print(f"  MNPS-{dim}: {n_pos}/{n_tot} positive  {dominant}  agree={frac:.0%}  median_bc={med:+.4f}")

# All subjects (including WARN, for transparency)
print(f"\n=== All subjects (including WARN) ===")
ev_all = df_all[df_all["bin"] == "event"]
for dim in DIMS:
    n_pos = int((ev_all[f"{dim}_bc"] > 0).sum())
    n_tot = len(ev_all)
    frac = max(n_pos, n_tot - n_pos) / n_tot
    med  = ev_all[f"{dim}_bc"].median()
    print(f"  MNPS-{dim}: {n_pos}/{n_tot} positive  agree={frac:.0%}  median_bc={med:+.4f}")

# Denominator audit
print(f"\n=== Denominator audit ===")
n_total_subs   = len(latest)
n_qc_pass      = len([s for s, q in qc_map.items() if q.get("qc_pass")])
n_qc_warn      = len([s for s, q in qc_map.items() if not q.get("qc_pass") and q.get("n_fails",0) >= 0])
ev_rows        = df_all[df_all["bin"] == "event"]
n_ev_pass_rows = len(df_pass[df_pass["bin"] == "event"])
n_ev_all_rows  = len(ev_rows)
assert n_total_subs == n_ev_all_rows, f"Mismatch: {n_total_subs} subjects vs {n_ev_all_rows} event-bin rows"
print(f"  Unique subjects with parquet      : {n_total_subs}")
print(f"  QC PASS (from QC JSON)            : {n_qc_pass}")
print(f"  QC WARN (from QC JSON)            : {n_qc_warn}")
print(f"  Event-bin rows (deduplicated)     : {n_ev_all_rows}  [== n_total_subs, OK]")
print(f"  Event-bin rows, QC-pass subjects  : {n_ev_pass_rows}  [== n_qc_pass, expected]")
assert n_ev_pass_rows == n_qc_pass, f"QC-pass mismatch: {n_ev_pass_rows} vs {n_qc_pass}"
print(f"  Denominators consistent: YES")

# Save
out = PROC / f"baseline_corrected_all_{SLUG}.csv"
df_all.to_csv(str(out), index=False)
print(f"\nSaved: {out}")
