"""
Batch event-locked MNPS pipeline — ds005555 sub-1 to sub-5.

For each subject:
  1. Check if a 6s/2s MNDM H5 exists; if not, run mndm features + summarize.
  2. Detect N2 spindles with canonical YASA (freq_sp=(12,15), defaults).
  3. Write annotation CSV with source tag + SHA-256.
  4. Run event-locked export.
  5. Apply QC gate.
  6. Collect per-subject QC summary.

Protocol: protocol_n2_event_locked_v1.md
Claim:    detector-derived exploratory MNPS; no inferential statistics.

Usage:
  python batch_event_locked.py [--subjects sub-1 sub-2 ...] [--channel PSG_F3] [--skip-mndm]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for pkg in ["mndm/src", "core/src"]:
    p = str(REPO / pkg)
    if p not in sys.path:
        sys.path.insert(0, p)

import h5py
import mne
import numpy as np
import pandas as pd
import yasa

from mndm.schema import MNPSPayload
from mndm.pipeline.event_annotations import EventTable
from mndm.pipeline.event_locked_runner import run_event_locked_export
from core import config_loader

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SUBJECTS  = ["sub-1", "sub-2", "sub-3", "sub-4", "sub-5"]
DS_ROOT   = Path(r"M:\datasets\received\openneuro\ds005555")
PROC_ROOT = Path(r"M:\datasets\processed\openneuro\ds005555")
CONFIG_YAML = REPO / "mndm" / "config" / "config_ingest_ds005555_sleep_spindles.yaml"
DEFAULT_CHANNEL = "PSG_F3"

# Canonical YASA parameters (protocol_n2_event_locked_v1)
YASA_PARAMS = dict(
    freq_sp=(12, 15), freq_broad=(1, 30), duration=(0.5, 3.0),
    min_distance=500, thresh={"rel_pow": 0.20, "corr": 0.65, "rms": 1.5},
    remove_outliers=False, include=(2,), verbose=False,
)
# SOURCE_TAG is set after arg parsing (includes channel name)

# QC gate thresholds
QC_RATE_MIN, QC_RATE_MAX = 0.3, 5.0   # spindles/min N2
QC_MATCH_RATE_MIN = 0.80
QC_FINITE_MNPS_MIN = 0.99
QC_EXCL_MAX_FRAC   = 0.30             # max fraction excluded by stage transition

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--subjects", nargs="+", default=SUBJECTS)
parser.add_argument("--channel", default=DEFAULT_CHANNEL,
                    help="EEG channel to use for spindle detection (default: PSG_F3)")
parser.add_argument("--skip-mndm", action="store_true",
                    help="Skip MNDM pipeline; only run spindle detection + event-locked export")
args = parser.parse_args()
SUBJECTS = args.subjects
CHANNEL  = args.channel
# Source tag encodes channel so annotations are channel-specific
SOURCE_TAG = f"detector:yasa-0.7.0/protocol-v1/freq12-15Hz-defaults-N2/{CHANNEL}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_h5(sub: str) -> Path | None:
    """Find the most recent sub-level H5 under PROC_ROOT / ds005555."""
    pattern = f"{sub}_Sleep_acq-psg.h5"
    candidates = sorted(PROC_ROOT.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_payload(h5_path: Path, sub: str) -> MNPSPayload:
    with h5py.File(str(h5_path), "r") as f:
        time_arr  = f["time"][:]
        # MNPS 3D — stored as /mnps_3d
        x         = f["mnps_3d"][:] if "mnps_3d" in f else f["x"][:]
        x_dot     = (f["mnps_3d_dot"][:] if "mnps_3d_dot" in f
                     else f["x_dot"][:] if "x_dot" in f else None)
        # coords_9d is a group with a /values sub-dataset
        coords_9d = (f["coords_9d/values"][:]
                     if "coords_9d" in f and "coords_9d/values" in f else None)
        ws         = f["window_start"][:] if "window_start" in f else None
        we         = f["window_end"][:]   if "window_end"   in f else None
        stage_raw  = f["labels/stage"][:] if "labels/stage" in f else None
    stage = stage_raw.astype(int) if stage_raw is not None else None
    return MNPSPayload(
        time=time_arr, x=x, x_dot=x_dot, coords_9d=coords_9d,
        window_start=ws, window_end=we, stage=stage,
    )


def run_mndm(sub: str) -> bool:
    """Run mndm features + summarize for a subject. Returns True on success."""
    sub_num = sub.replace("sub-", "")
    cmd_base = [
        sys.executable, "-m", "mndm.cli",
        "--config", str(CONFIG_YAML),
        "--dataset", "ds005555",
        "--subject", sub_num,
    ]
    env = {"PYTHONPATH": f"{REPO/'mndm/src'};{REPO/'core/src'}"}
    import os
    full_env = {**os.environ, **env}

    for subcmd in ["features", "summarize"]:
        cmd = cmd_base[:1] + ["-m", "mndm.cli"] + [subcmd] + cmd_base[3:]
        # Build properly: python -m mndm.cli features --config ... --dataset ... --subject ...
        cmd = [sys.executable, "-c",
               f"from mndm.cli import main; import sys; sys.argv=['mndm','{subcmd}',"
               f"'--config',r'{CONFIG_YAML}','--dataset','ds005555','--subject','{sub_num}']; main()"]
        print(f"  [{sub}] Running mndm {subcmd}...")
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, env=full_env)
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"  [{sub}] MNDM {subcmd} FAILED ({elapsed:.0f}s):")
            print(result.stderr[-2000:])
            return False
        print(f"  [{sub}] mndm {subcmd} OK ({elapsed:.0f}s)")
    return True


def _channel_slug(ch: str) -> str:
    """Convert channel name to a safe filename fragment, e.g. PSG_F3 -> psg_f3."""
    return ch.lower().replace(" ", "_")


def detect_spindles(sub: str) -> tuple[pd.DataFrame, Path | None]:
    """Detect N2 spindles on the configured CHANNEL; return (df_spindles, csv_path)."""
    base = DS_ROOT / sub / "eeg"
    edf  = base / f"{sub}_task-Sleep_acq-psg_eeg.edf"
    tsv  = base / f"{sub}_task-Sleep_acq-psg_events.tsv"
    if not edf.exists() or not tsv.exists():
        print(f"  [{sub}] EDF or events.tsv missing — skip")
        return pd.DataFrame(), None

    raw = mne.io.read_raw_edf(str(edf), preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    if CHANNEL not in raw.ch_names:
        print(f"  [{sub}] Channel {CHANNEL} not found in {raw.ch_names}")
        return pd.DataFrame(), None
    idx = raw.ch_names.index(CHANNEL)
    data, _ = raw[idx, :]
    data_uv = data[0] * 1e6

    df_ev = pd.read_csv(str(tsv), sep="\t")
    SM = {0:0,1:1,2:2,3:3,4:4,-1:-1}
    hypno = df_ev["stage_hum"].map(lambda x: SM.get(int(x), -1)).to_numpy(int)
    n_rec = int(np.ceil(len(data_uv) / sfreq / 30))
    hypno = np.pad(hypno[:n_rec], (0, max(0, n_rec - len(hypno))), constant_values=-1)
    hypno_up = yasa.hypno_upsample_to_data(hypno, sf_hypno=1/30, data=data_uv, sf_data=sfreq)

    n2_min = float((hypno == 2).sum()) * 30 / 60
    sp = yasa.spindles_detect(data_uv, sf=sfreq, hypno=hypno_up, **YASA_PARAMS)
    if sp is None:
        print(f"  [{sub}] No spindles detected")
        return pd.DataFrame(), None

    df = sp.summary()
    rate = len(df) / n2_min if n2_min > 0 else 0
    print(f"  [{sub}] {len(df)} spindles  {rate:.2f}/min N2 ({n2_min:.0f}min N2)")

    out = pd.DataFrame({
        "onset_sec": df["Start"].astype(float),
        "duration_sec": df["Duration"].astype(float),
        "peak_sec": df["Peak"].astype(float),
        "event_type": "sleep_spindle",
        "channel": CHANNEL,
        "confidence": np.nan,
        "source": SOURCE_TAG,
        "stage": df.get("Stage", 2).astype(int),
    })
    for col in ["Amplitude","RMS","AbsPower","RelPower","Frequency","Oscillations","Symmetry"]:
        if col in df.columns:
            out[f"yasa_{col.lower()}"] = df[col].astype(float)

    csv_path = base / f"{sub}_task-Sleep_acq-psg_spindles_yasa_v1_{_channel_slug(CHANNEL)}.csv"
    out.to_csv(str(csv_path), index=False)
    return out, csv_path


def run_event_locked(sub: str, h5_path: Path, csv_path: Path, cfg: dict) -> dict:
    """Run event-locked export for one subject. Returns QC dict."""
    print(f"  [{sub}] Loading payload from H5...")
    payload = load_payload(h5_path, sub)
    out_parquet = h5_path.parent / f"{sub}_Sleep_acq-psg_event_locked_v1_{_channel_slug(CHANNEL)}.parquet"
    print(f"  [{sub}] Running reusable event-locked pipeline...")
    result = run_event_locked_export(
        payload=payload,
        config=cfg,
        dataset_id="ds005555",
        source_path=csv_path,
        subject_id=sub,
        run_id="Sleep_acq-psg",
        out_prefix=out_parquet.with_suffix(""),
    )
    event_table = result.event_table
    alignment = result.alignment
    controls = result.controls
    n_aligned = alignment.qc.get("n_events_aligned", 0)
    n_excluded = alignment.qc.get("n_events_excluded_stage_transition", 0)
    n_matched = len(controls.rows)
    match_rate = controls.qc.get("match_success_rate", 0.0)
    n_rows = len(result.rows)
    print(f"  [{sub}] Aligned: {n_aligned}, excluded (transition): {n_excluded}")
    print(f"  [{sub}] Controls: {n_matched}, match rate: {match_rate:.2f}")

    df_ex = pd.read_parquet(str(out_parquet))
    finite_frac = float(df_ex["mnps_finite"].mean()) if "mnps_finite" in df_ex.columns else 1.0
    bins_populated = list(df_ex[df_ex["condition"]!="matched_control"]["bin_label"].unique()) if "bin_label" in df_ex.columns else []
    n_spindles_total = len(event_table.onset_sec)
    excl_frac = n_excluded / n_spindles_total if n_spindles_total > 0 else 0.0

    # N2 rate check (need N2 duration from H5)
    n2_windows = int((payload.stage == 2).sum()) if payload.stage is not None else 0
    step_s = float(np.median(np.diff(payload.time))) if len(payload.time) > 1 else 2.0
    n2_min = n2_windows * step_s / 60
    rate_min = n_spindles_total / n2_min if n2_min > 0 else 0.0

    src_hash = sha256(csv_path)

    qc = {
        "subject": sub,
        "n_spindles": n_spindles_total,
        "rate_per_min": round(rate_min, 3),
        "rate_in_range": QC_RATE_MIN <= rate_min <= QC_RATE_MAX,
        "n_aligned": n_aligned,
        "n_excluded_transition": n_excluded,
        "excl_frac": round(excl_frac, 3),
        "excl_ok": excl_frac <= QC_EXCL_MAX_FRAC,
        "n_controls": n_matched,
        "match_rate": round(match_rate, 3),
        "match_rate_ok": match_rate >= QC_MATCH_RATE_MIN,
        "finite_mnps_frac": round(finite_frac, 4),
        "finite_ok": finite_frac >= QC_FINITE_MNPS_MIN,
        "bins_populated": sorted(bins_populated),
        "all_5_bins_ok": len(set(bins_populated) & {"pre_far","pre_near","event","post_near","post_far"}) == 5,
        "n_export_rows": n_rows,
        "annotation_source_hash": src_hash[:16] + "...",
        "parquet_path": str(out_parquet),
        "n_fails": 0,
    }
    n_fails = sum([
        not qc["rate_in_range"],
        not qc["all_5_bins_ok"],
        not qc["match_rate_ok"],
        not qc["finite_ok"],
        not qc["excl_ok"],
    ])
    qc["n_fails"] = n_fails
    qc["qc_pass"] = n_fails == 0

    return qc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("=" * 72)
print(f"BATCH EVENT-LOCKED PIPELINE -- ds005555  [protocol_n2_event_locked_v1]")
print(f"Subjects: {SUBJECTS}")
print(f"Channel: {CHANNEL}, YASA: freq_sp=(12,15), defaults, N2 only")
print("=" * 72)

# Load config once
cfg = config_loader.load_config(str(CONFIG_YAML))

all_qc = []
for sub in SUBJECTS:
    print(f"\n{'-'*60}")
    print(f"SUBJECT: {sub}")
    print(f"{'-'*60}")

    # --- Step 1: MNDM H5 ---
    h5_path = find_h5(sub)
    if h5_path is None:
        if args.skip_mndm:
            print(f"  [{sub}] No H5 found and --skip-mndm set. SKIPPING.")
            all_qc.append({"subject": sub, "qc_pass": False, "n_fails": -1,
                           "skip_reason": "no_h5"})
            continue
        print(f"  [{sub}] No H5 found — running MNDM pipeline...")
        success = run_mndm(sub)
        if not success:
            print(f"  [{sub}] MNDM FAILED. Skipping event-locked step.")
            all_qc.append({"subject": sub, "qc_pass": False, "n_fails": -1,
                           "skip_reason": "mndm_failed"})
            continue
        h5_path = find_h5(sub)
        if h5_path is None:
            print(f"  [{sub}] H5 still not found after MNDM. Skipping.")
            all_qc.append({"subject": sub, "qc_pass": False, "n_fails": -1,
                           "skip_reason": "h5_not_found_post_mndm"})
            continue
    print(f"  [{sub}] H5: {h5_path.name}")

    # --- Step 2: Spindle detection ---
    print(f"  [{sub}] Detecting spindles on {CHANNEL}...")
    base_eeg = DS_ROOT / sub / "eeg"
    # Check for existing channel-specific canonical CSV
    csv_path = base_eeg / f"{sub}_task-Sleep_acq-psg_spindles_yasa_v1_{_channel_slug(CHANNEL)}.csv"
    if csv_path.exists():
        print(f"  [{sub}] Using existing spindle CSV: {csv_path.name}")
    else:
        _, csv_path = detect_spindles(sub)
        if csv_path is None:
            print(f"  [{sub}] Detection failed. Skipping.")
            all_qc.append({"subject": sub, "qc_pass": False, "n_fails": -1,
                           "skip_reason": "detection_failed"})
            continue

    # --- Steps 3–4: Event-locked export + QC ---
    try:
        qc = run_event_locked(sub, h5_path, csv_path, cfg)
    except Exception as exc:
        import traceback
        print(f"  [{sub}] Event-locked FAILED: {exc}")
        traceback.print_exc()
        all_qc.append({"subject": sub, "qc_pass": False, "n_fails": -1,
                       "skip_reason": f"export_error: {exc}"})
        continue

    all_qc.append(qc)

    # --- QC gate report ---
    gate_symbol = "PASS" if qc["qc_pass"] else f"WARN ({qc['n_fails']} flags)"
    print(f"\n  [{sub}] QC gate: {gate_symbol}")
    print(f"    rate        {qc['rate_per_min']:.2f}/min  {'OK' if qc['rate_in_range'] else 'FLAG'}")
    print(f"    bins        {qc['bins_populated']}  {'OK' if qc['all_5_bins_ok'] else 'FLAG'}")
    print(f"    match_rate  {qc['match_rate']:.2f}  {'OK' if qc['match_rate_ok'] else 'FLAG'}")
    print(f"    finite_mnps {qc['finite_mnps_frac']:.4f}  {'OK' if qc['finite_ok'] else 'FLAG'}")
    print(f"    excl_frac   {qc['excl_frac']:.2f}  {'OK' if qc['excl_ok'] else 'FLAG'}")
    print(f"    rows        {qc['n_export_rows']}")

# ---------------------------------------------------------------------------
# Batch summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("BATCH QC SUMMARY")
print("=" * 72)
print(f"  {'Subject':<10} {'Rate/min':>9} {'Bins':>5} {'Match':>6} {'Finite':>7} "
      f"{'Excl':>6} {'Rows':>7} {'Gate':>8}")
print(f"  {'-'*10} {'-'*9} {'-'*5} {'-'*6} {'-'*7} {'-'*6} {'-'*7} {'-'*8}")
for q in all_qc:
    sub = q["subject"]
    if q.get("n_fails", -1) < 0:
        reason = q.get("skip_reason", "skipped")
        print(f"  {sub:<10} {'—':>9} {'—':>5} {'—':>6} {'—':>7} {'—':>6} {'—':>7} SKIP:{reason}")
        continue
    gate = "PASS" if q.get("qc_pass") else f"WARN({q.get('n_fails',0)})"
    print(f"  {sub:<10} {q.get('rate_per_min',0):>9.2f} "
          f"{'5/5' if q.get('all_5_bins_ok') else '<5/5':>5} "
          f"{q.get('match_rate',0):>6.2f} "
          f"{q.get('finite_mnps_frac',0):>7.4f} "
          f"{q.get('excl_frac',0):>6.2f} "
          f"{q.get('n_export_rows',0):>7d} "
          f"{gate:>8}")

n_pass = sum(1 for q in all_qc if q.get("qc_pass"))
n_warn = sum(1 for q in all_qc if not q.get("qc_pass") and q.get("n_fails", -1) >= 0)
n_skip = sum(1 for q in all_qc if q.get("n_fails", -1) < 0)
print(f"\n  PASS: {n_pass}  WARN: {n_warn}  SKIP: {n_skip}  Total: {len(all_qc)}")

# Save QC summary
qc_out = PROC_ROOT / f"batch_event_locked_qc_{_channel_slug(CHANNEL)}.json"
try:
    with open(str(qc_out), "w", encoding="utf-8") as f:
        json.dump(all_qc, f, indent=2, default=str)
    print(f"\n  QC summary saved: {qc_out}")
except Exception:
    pass

print("\nCLAIM: detector-derived exploratory pipeline. No inferential statistics.")
