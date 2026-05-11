"""
Smoke test: event-locked export against real ds005555 sub-1 HDF5.

Steps:
  1. Load real HDF5 (6s/2s MNPS, /labels/stage).
  2. Create synthetic spindle annotations at every 3rd N2 window onset.
  3. Run align_events_to_windows.
  4. Run build_matched_controls.
  5. Build event-locked table and write to Parquet.
  6. Report bin distribution and MNPS statistics.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo path setup
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]
for pkg in ["mndm/src", "core/src"]:
    p = str(REPO / pkg)
    if p not in sys.path:
        sys.path.insert(0, p)

import h5py
import numpy as np
from pathlib import Path as _Path
from core.config_loader import load_config as _load_config

from mndm.pipeline.event_annotations import EventTable, validate_event_table
from mndm.pipeline.event_alignment import AlignmentConfig, BinSpec, align_events_to_windows
from mndm.pipeline.control_matching import MatchingConfig, build_matched_controls
from mndm.pipeline.event_locked_export import build_event_locked_table, write_event_locked_parquet
from mndm.pipeline.event_locked_config import (
    event_locked_profile_from_config,
    alignment_config_from_profile,
    matching_config_from_profile,
    export_config_from_profile,
)
from mndm.schema import MNPSPayload

# ---------------------------------------------------------------------------
# Config — driven by overlay YAML so provenance is fully populated
# ---------------------------------------------------------------------------
OVERLAY_CFG_PATH = REPO / "mndm/config/config_ingest_ds005555_sleep_spindles.yaml"

# Use the new H5 produced with fixed /time axis
H5_PATH = Path(r"M:\datasets\processed\openneuro\ds005555"
               r"\neuralmanifolddynamics_ds005555_20260508_092744"
               r"\sub-1_Sleep_acq-psg\sub-1_Sleep_acq-psg.h5")

DATASET_ID = "ds005555"
SUBJECT_ID = "sub-1"

SPINDLE_EVERY_NTH_N2 = 30  # used only as fallback if YASA CSV not found
SPINDLE_DURATION_S   = 1.5  # fallback duration
SPINDLE_CSV = Path(r"M:\datasets\received\openneuro\ds005555\sub-1\eeg"
                   r"\sub-1_task-Sleep_acq-psg_spindles_yasa.csv")
USE_REAL_SPINDLES = SPINDLE_CSV.exists()

# Load config and derive typed pipeline objects from the event_locked profile.
_cfg = _load_config(OVERLAY_CFG_PATH)
PROFILE   = event_locked_profile_from_config(_cfg, DATASET_ID)
ALIGNMENT_CFG = alignment_config_from_profile(PROFILE, _cfg, DATASET_ID)
MATCHING_CFG  = matching_config_from_profile(PROFILE)
EXPORT_CFG    = export_config_from_profile(PROFILE, _cfg, DATASET_ID)

# ---------------------------------------------------------------------------
# Load H5
# ---------------------------------------------------------------------------
print("=" * 60)
print(f"Loading HDF5: {H5_PATH}")
assert H5_PATH.exists(), f"H5 not found: {H5_PATH}"

with h5py.File(H5_PATH, "r") as f:
    time        = f["time"][:]
    mnps_3d     = f["mnps_3d"][:]
    mnps_3d_dot = f["mnps_3d_dot"][:] if "mnps_3d_dot" in f else None
    coords_9d   = f["coords_9d/values"][:] if "coords_9d" in f else None
    window_start = f["window_start"][:]
    window_end   = f["window_end"][:]
    stage_arr    = f["labels/stage"][:]

print(f"  Windows : {len(time)}")
print(f"  Duration: {window_end[-1]/3600:.2f} h")
print(f"  MNPS shape: {mnps_3d.shape}")
print(f"  coords_9d: {coords_9d.shape if coords_9d is not None else 'absent'}")

stage_counts = {s: int((stage_arr == s).sum()) for s in np.unique(stage_arr)}
print(f"  Stage dist: {stage_counts}")

# ---------------------------------------------------------------------------
# Spindle events: real (YASA) or synthetic fallback
# ---------------------------------------------------------------------------
n2_indices = np.where(stage_arr == 2)[0]

if USE_REAL_SPINDLES:
    import pandas as _pd
    print(f"\nLoading real spindle annotations: {SPINDLE_CSV}")
    sp_df = _pd.read_csv(SPINDLE_CSV)
    n_spindles = len(sp_df)
    onset_sec    = sp_df["onset_sec"].to_numpy(dtype=np.float64)
    duration_sec = sp_df["duration_sec"].to_numpy(dtype=np.float32)
    peak_sec     = sp_df["peak_sec"].to_numpy(dtype=np.float64) if "peak_sec" in sp_df.columns else onset_sec + duration_sec / 2.0
    source_label = str(sp_df["source"].iloc[0]) if "source" in sp_df.columns else "yasa"

    event_table = EventTable(
        onset_sec=onset_sec,
        duration_sec=duration_sec,
        peak_sec=peak_sec,
        event_type=np.array(["sleep_spindle"] * n_spindles, dtype=object),
        channel=sp_df.get("channel", _pd.Series(["PSG_F3"] * n_spindles)).to_numpy(dtype=object),
        confidence=sp_df.get("confidence", _pd.Series([np.nan] * n_spindles)).to_numpy(dtype=np.float32),
        source=np.array([source_label] * n_spindles, dtype=object),
        source_path=str(SPINDLE_CSV),
        n_events_loaded=n_spindles,
    )
    print(f"  Loaded {n_spindles} spindles from {source_label}")
else:
    print(f"\nFallback: synthetic spindles every {SPINDLE_EVERY_NTH_N2}th N2 window")
    spindle_idx  = n2_indices[::SPINDLE_EVERY_NTH_N2]
    n_spindles   = len(spindle_idx)
    onset_sec    = window_start[spindle_idx]
    duration_sec = np.full(n_spindles, SPINDLE_DURATION_S, dtype=np.float32)
    peak_sec     = onset_sec + SPINDLE_DURATION_S / 2.0
    event_table = EventTable(
        onset_sec=onset_sec,
        duration_sec=duration_sec,
        peak_sec=peak_sec,
        event_type=np.array(["sleep_spindle"] * n_spindles, dtype=object),
        channel=np.array(["C3"] * n_spindles, dtype=object),
        confidence=np.full(n_spindles, 0.9, dtype=np.float32),
        source=np.array([f"synthetic_every_{SPINDLE_EVERY_NTH_N2}th_N2"] * n_spindles, dtype=object),
        source_path="synthetic",
        n_events_loaded=n_spindles,
    )

warnings = validate_event_table(event_table)
if warnings:
    print(f"  Validation warnings: {warnings}")

# ---------------------------------------------------------------------------
# Build MNPSPayload
# ---------------------------------------------------------------------------
payload = MNPSPayload(
    x=mnps_3d,
    x_dot=mnps_3d_dot if mnps_3d_dot is not None else np.zeros_like(mnps_3d),
    coords_9d=coords_9d,
    time=time,
    window_start=window_start,
    window_end=window_end,
    labels={"stage": stage_arr},
)

# ---------------------------------------------------------------------------
# Align events → windows
# ---------------------------------------------------------------------------
print("\nRunning event-window alignment...")
align_result = align_events_to_windows(
    event_table,
    window_start=window_start,
    window_end=window_end,
    time=time,
    stage=stage_arr,
    config=ALIGNMENT_CFG,
)
qc = align_result.qc
print(f"  Total event-window pairs  : {len(align_result.rows)}")
print(f"  Events aligned            : {qc['n_events_aligned']}")
print(f"  Pairs by bin:")
from collections import Counter
bin_counts = Counter(r.bin_label for r in align_result.rows if r.bin_label and r.bin_label != "outside")
for bn, cnt in sorted(bin_counts.items()):
    print(f"    {bn:15s}: {cnt}")
print(f"  Stage-transition excluded : {qc.get('n_events_excluded_stage_transition', 0)}")

# ---------------------------------------------------------------------------
# Build matched controls
# ---------------------------------------------------------------------------
print("\nBuilding matched N2 controls...")
ctrl_result = build_matched_controls(
    event_table,
    time=time,
    window_start=window_start,
    window_end=window_end,
    stage=stage_arr,
    config=MATCHING_CFG,
)
n_requested = n_spindles * MATCHING_CFG.n_controls_per_event
n_matched = len(ctrl_result.rows)
n_failed = ctrl_result.qc.get("n_failed_matches", n_requested - n_matched)
print(f"  Controls requested: {n_requested}")
print(f"  Controls matched  : {n_matched}")
print(f"  Failed matches    : {n_failed}")

# ---------------------------------------------------------------------------
# Event-locked export
# ---------------------------------------------------------------------------
print("\nBuilding event-locked table...")
table = build_event_locked_table(
    payload=payload,
    alignment=align_result,
    controls=ctrl_result,
    event_table=event_table,
    subject_id=SUBJECT_ID,
    dataset_id=DATASET_ID,
    config=EXPORT_CFG,
)
print(f"  Rows in table: {len(table)}")
conditions = Counter(r["condition"] for r in table)
print(f"  Conditions: {dict(conditions)}")

bins_in_table = Counter(r["bin_label"] for r in table)
print(f"  Bins in table:")
for bn, cnt in sorted(bins_in_table.items()):
    pct = 100 * cnt / len(table)
    print(f"    {bn:15s}: {cnt} ({pct:.1f}%)")

# MNPS mean per bin for spindle_event condition
import numpy as np
spindle_rows = [r for r in table if r["condition"] == "spindle_event"]
if spindle_rows:
    m_vals = np.array([r["m"] for r in spindle_rows])
    d_vals = np.array([r["d"] for r in spindle_rows])
    e_vals = np.array([r["e"] for r in spindle_rows])
    print(f"\n  Spindle event MNPS mean: m={m_vals.mean():.4f}, d={d_vals.mean():.4f}, e={e_vals.mean():.4f}")

# Check provenance — all profile fields must be non-None
prov_check = {
    "profile_name": PROFILE.profile_name,
    "window_length_s": PROFILE.window_length_s,
    "window_step_s": PROFILE.window_step_s,
    "control_seed": PROFILE.control_seed,
    "dataset_id": DATASET_ID,
    "subject_id": SUBJECT_ID,
}
print(f"\n  Provenance in first row:")
first = table[0]
prov_ok = True
for key, expected in prov_check.items():
    actual = first.get(key)
    status = "OK" if actual == expected else f"MISMATCH (got {actual!r}, expected {expected!r})"
    print(f"    {key}: {actual!r}  [{status}]")
    if "MISMATCH" in status:
        prov_ok = False
if not prov_ok:
    print("  WARNING: provenance mismatch!")
else:
    print("  All provenance fields correct.")

# ---------------------------------------------------------------------------
# Write Parquet — always persist to a stable path alongside the HDF5
# ---------------------------------------------------------------------------
import pandas as pd

OUT_PARQUET = H5_PATH.parent / (H5_PATH.stem + "_event_locked.parquet")
write_event_locked_parquet(table, OUT_PARQUET)
size_kb = OUT_PARQUET.stat().st_size / 1024
print(f"\nParquet written: {OUT_PARQUET} ({size_kb:.1f} KB)")

# Reload and spot-check
df = pd.read_parquet(str(OUT_PARQUET))
print(f"  Parquet rows: {len(df)}, cols: {len(df.columns)}")
print(f"  Columns: {list(df.columns[:10])} ...")
print(f"  Finite MNPS rows: {df['mnps_finite'].sum()} / {len(df)}")

if "annotation_source_hash" in df.columns:
    h = df["annotation_source_hash"].iloc[0]
    print(f"  annotation_source_hash: {h[:16]}... ({len(h)} chars)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SMOKE TEST PASSED")
print(f"  Real HDF5: {H5_PATH.name}")
print(f"  N2 windows: {len(n2_indices)}, Spindles: {n_spindles}, Controls: {n_matched}")
print(f"  Bins populated: {sorted(bin_counts.keys())}")
all_expected_bins = {"pre_far", "pre_near", "event", "post_near", "post_far"}
missing = all_expected_bins - set(bin_counts.keys())
if missing:
    print(f"  WARNING - missing bins: {missing}")
else:
    print("  All 5 bins populated OK")
print("=" * 60)
