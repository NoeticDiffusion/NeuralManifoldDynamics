"""ds006848 validation script — run after feature re-extraction.

Three validation gates:
  Gate 1 — Event-locking works (after summarize with bids_events)
  Gate 2 — ECG features regenerated with new columns
  Gate 3 — HRV contamination table exists and is usable

Usage:
  python scripts/ds006848_validate_infrastructure.py

Outputs to stdout and writes ds006848_validation_report.txt.
"""
import pathlib
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

PROCESSED_DIR = pathlib.Path("J:/processed/openneuro/ds006848")
REPORT_PATH = pathlib.Path("H:/SourceRepo2/NeuralManifoldDynamics/project/diary/ds006848_validation_report.txt")

lines = []

def pr(msg=""):
    print(msg)
    lines.append(msg)


# ============================================================
# Gate 2 — ECG features with new columns
# ============================================================
pr("=" * 70)
pr("GATE 2 — ECG features regenerated (new columns)")
pr("=" * 70)

feat_path = PROCESSED_DIR / "features.parquet"
if not feat_path.exists():
    feat_path = PROCESSED_DIR / "features.csv"
    if not feat_path.exists():
        pr("ERROR: features.parquet/csv not found. Run features pipeline first.")
        sys.exit(1)

df = pd.read_parquet(feat_path) if feat_path.suffix == ".parquet" else pd.read_csv(feat_path)
pr(f"Features file: {feat_path.name}, rows={len(df)}, cols={len(df.columns)}")

required_new_cols = [
    "ecg_polarity_inverted",
    "ecg_peak_detector",
    "ecg_hrv_dominant_stage_label",
    "ecg_hrv_dominant_stage_frac",
    "ecg_hrv_n_stage_labels",
    "ecg_hrv_contains_excluded_label",
]
pr("")
pr("New column presence check:")
all_present = True
for col in required_new_cols:
    present = col in df.columns
    status = "OK" if present else "MISSING"
    pr(f"  {status:7s}  {col}")
    if not present:
        all_present = False

if not all_present:
    pr("")
    pr("ERROR: New columns are missing. Feature cache was not invalidated.")
    pr("Run: python scripts/ds006848_invalidate_ecg_cache.py")
    pr("Then re-run: mndm.cli features --dataset ds006848 ...")
    sys.exit(1)

pr("")
pr("ECG polarity distribution:")
if "ecg_polarity_inverted" in df.columns:
    n_inverted = df["ecg_polarity_inverted"].sum()
    n_total = df["ecg_polarity_inverted"].notna().sum()
    pr(f"  polarity_inverted=True: {n_inverted}/{n_total} epochs ({100*n_inverted/max(n_total,1):.1f}%)")

    if "file" in df.columns:
        file_pol = df.groupby("file")["ecg_polarity_inverted"].mean().reset_index()
        file_pol.columns = ["file", "frac_inverted"]
        inverted_files = file_pol[file_pol["frac_inverted"] > 0.5].sort_values("frac_inverted", ascending=False)
        pr(f"  Files with predominantly inverted polarity: {len(inverted_files)}")
        for _, row in inverted_files.head(10).iterrows():
            pr(f"    {row['file']}: {row['frac_inverted']:.2f}")

if "ecg_peak_detector" in df.columns:
    pr("")
    pr("Peak detector usage:")
    pr(df["ecg_peak_detector"].value_counts().to_string())

pr("")
pr("HR and RMSSD summary (after polarity correction):")
for col, label in [("ecg_hrv_hr_mean_bpm", "HR bpm"), ("ecg_hrv_rmssd_ms", "RMSSD ms")]:
    if col in df.columns:
        v = df[col].dropna()
        pr(f"  {label}: median={v.median():.1f}, IQR=[{v.quantile(0.25):.1f}, {v.quantile(0.75):.1f}], "
           f"p5={v.quantile(0.05):.1f}, p95={v.quantile(0.95):.1f}")

pr("")
pr("Per-file median HR (top 10 highest — flags inverted/double-detected):")
if "file" in df.columns and "ecg_hrv_hr_mean_bpm" in df.columns:
    fhr = df.groupby("file").agg(
        hr_med=("ecg_hrv_hr_mean_bpm", "median"),
        rmssd_med=("ecg_hrv_rmssd_ms", "median"),
        pol_frac=("ecg_polarity_inverted", "mean"),
    ).reset_index().sort_values("hr_med", ascending=False)
    pr(fhr.head(10).to_string(index=False))
    suspicious = fhr[fhr["hr_med"] > 100]
    pr(f"\n  Files with HR > 100 bpm: {len(suspicious)}")

# ============================================================
# Gate 3 — HRV contamination
# ============================================================
pr("")
pr("=" * 70)
pr("GATE 3 — HRV contamination reporting")
pr("=" * 70)

hrv_ok = "qc_ok_ecg_hrv" in df.columns
contamination_ok = "ecg_hrv_contains_excluded_label" in df.columns
purity_ok = "ecg_hrv_dominant_stage_frac" in df.columns

if hrv_ok and contamination_ok and purity_ok:
    hrv_df = df[df["qc_ok_ecg_hrv"] == True].copy() if hrv_ok else df.copy()
    pr(f"qc_ok_ecg_hrv windows: {len(hrv_df)}")

    if contamination_ok:
        n_contaminated = hrv_df["ecg_hrv_contains_excluded_label"].sum()
        pr(f"Windows with excluded label (Digits_Retrieval): {n_contaminated} ({100*n_contaminated/max(len(hrv_df),1):.1f}%)")

    if "ecg_hrv_dominant_stage_label" in df.columns:
        pr("")
        pr("Dominant stage label distribution (qc_ok windows):")
        pr(hrv_df["ecg_hrv_dominant_stage_label"].value_counts().head(15).to_string())

    # Science lead's recommended gate
    gate_mask = (
        (hrv_df["qc_ok_ecg_hrv"] == True)
        & (hrv_df["ecg_hrv_contains_excluded_label"] == False)
        & (hrv_df["ecg_hrv_dominant_stage_frac"] >= 0.60)
    )
    n_gated = gate_mask.sum()
    pr("")
    pr("Science lead gate (qc_ok AND not_excluded AND dominant_frac>=0.60):")
    pr(f"  Windows passing gate: {n_gated} / {len(hrv_df)} ({100*n_gated/max(len(hrv_df),1):.1f}%)")
    if n_gated < 100:
        pr("  WARNING: Very few windows pass gate. ds006848 HRV anchor claims should be gated.")
    else:
        pr("  PASS: Sufficient clean HRV windows for analysis.")

    # Stage breakdown after gate
    if n_gated > 0 and "task_state_label" in hrv_df.columns:
        pr("")
        pr("Stage breakdown after gate:")
        pr(hrv_df[gate_mask]["task_state_label"].value_counts().to_string())
else:
    pr("SKIP: HRV contamination columns not available. Re-run features first.")

# ============================================================
# Gate 1 — Event-locking
# ============================================================
pr("")
pr("=" * 70)
pr("GATE 1 — Event-locking (bids_events kind)")
pr("=" * 70)

run_dirs = sorted(PROCESSED_DIR.glob("neuralmanifolddynamics_*"))
if run_dirs:
    latest_run = run_dirs[-1]
    pr(f"Latest run dir: {latest_run.name}")

    # Look for event-locked parquet files
    el_files = list(latest_run.rglob("*event_locked*bids*.parquet"))
    el_files += list(latest_run.rglob("*event_locked*.parquet"))
    pr(f"Event-locked parquet files found: {len(el_files)}")

    if el_files:
        # Aggregate counts
        all_el = []
        for f in el_files:
            try:
                el_df = pd.read_parquet(f)
                all_el.append(el_df)
            except Exception as e:
                pr(f"  WARNING: Could not read {f.name}: {e}")
        if all_el:
            el_combined = pd.concat(all_el, ignore_index=True)
            pr(f"Total event-locked rows: {len(el_combined)}")
            if "event_type" in el_combined.columns:
                pr("")
                pr("Event counts by type:")
                pr(el_combined["event_type"].value_counts().head(20).to_string())
            if "subject_id" in el_combined.columns or "subject" in el_combined.columns:
                sub_col = "subject_id" if "subject_id" in el_combined.columns else "subject"
                n_subs = el_combined[sub_col].nunique()
                pr(f"\nSubjects with event-locked data: {n_subs}")
    else:
        pr("No event-locked parquet files found in latest run.")
        pr("Run: python -m mndm.cli summarize --dataset ds006848 --config mndm/config/config_ingest_ds006848.yaml")
else:
    pr("No run directories found in processed dir.")

# ============================================================
# Summary
# ============================================================
pr("")
pr("=" * 70)
pr("VALIDATION SUMMARY")
pr("=" * 70)

gates = {
    "Gate 2 (new ECG columns)": all_present,
    "Gate 3 (HRV contamination)": contamination_ok and purity_ok,
    "Gate 1 (event-locking)": len(el_files) > 0 if run_dirs else False,
}
for name, ok in gates.items():
    pr(f"  {'PASS' if ok else 'FAIL/PENDING':12s}  {name}")

pr("")
pr("Proceed to 04b encoding-phase analysis only when all gates PASS.")

# Write report
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"\nReport written to {REPORT_PATH}")
