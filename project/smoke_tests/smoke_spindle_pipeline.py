"""Smoke test: event-locked spindle MNPS pipeline against real data files.

Runs against:
  - ds005555 sub-1 PSG  (openneuro, BIDS EEG sleep dataset)
  - ANPHY EPCTL01       (custom format, tab-delimited stage annotations)

Neither dataset has real spindle annotations yet.
Synthetic spindles are injected: one per 3rd N2 epoch, centered in the epoch.
This tests pipeline mechanics, not sleep-spindle biology.

Usage
-----
    python project/smoke_tests/smoke_spindle_pipeline.py

Outputs
-------
    project/smoke_tests/results/smoke_<dataset>_<subject>.csv
    project/smoke_tests/results/smoke_<dataset>_<subject>_qc.json
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "mndm" / "src"))
sys.path.insert(0, str(REPO / "core" / "src"))

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Stage loaders
# ---------------------------------------------------------------------------

def load_ds005555_stages(
    events_tsv: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load onset_sec, duration_sec, stage_int from ds005555 events TSV.

    Stage encoding used here: W=0, N1=1, N2=2, N3=3, REM=4.
    """
    stage_map = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "R": 4}
    int_map = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
    onsets, durations, stages = [], [], []
    with events_tsv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            try:
                o = float(row["onset"])
                d = float(row.get("duration", 30))
                raw = row.get("stage_hum", row.get("stage_ai", "")).strip()
                s = stage_map.get(raw, int_map.get(raw, -1))
                onsets.append(o)
                durations.append(d)
                stages.append(s)
            except (ValueError, KeyError):
                continue
    return (
        np.asarray(onsets, dtype=np.float64),
        np.asarray(durations, dtype=np.float64),
        np.asarray(stages, dtype=np.int16),
    )


def load_anphy_stages(
    ann_file: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load stages from ANPHY tab-delimited annotation file.

    Format: stage<TAB>onset_sec<TAB>duration_sec
    """
    stage_map = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "R": 4}
    onsets, durations, stages = [], [], []
    with ann_file.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            raw_stage, raw_onset, raw_dur = parts[0], parts[1], parts[2]
            try:
                o = float(raw_onset)
                d = float(raw_dur)
                s = stage_map.get(raw_stage.strip().upper(), -1)
                onsets.append(o)
                durations.append(d)
                stages.append(s)
            except ValueError:
                continue
    return (
        np.asarray(onsets, dtype=np.float64),
        np.asarray(durations, dtype=np.float64),
        np.asarray(stages, dtype=np.int16),
    )


# ---------------------------------------------------------------------------
# Synthetic spindle generation (no real annotations available)
# ---------------------------------------------------------------------------

def make_synthetic_spindles(
    onsets: np.ndarray,
    durations: np.ndarray,
    stages: np.ndarray,
    *,
    target_stage: int = 2,
    every_nth: int = 3,
    spindle_duration_sec: float = 1.0,
    source: str = "synthetic:every_3rd_N2_epoch",
    rng_seed: int = 42,
) -> "EventTable":
    """Inject one synthetic spindle per ``every_nth`` N2 epoch.

    The spindle onset is placed at the centre of the epoch.
    This is strictly for smoke-testing pipeline mechanics, not biology.
    """
    from mndm.pipeline.event_annotations import EventTable

    n2_idx = np.where(stages == target_stage)[0]
    selected = n2_idx[::every_nth]
    if selected.size == 0:
        return EventTable(
            onset_sec=np.empty(0, dtype=np.float64),
            source_path="synthetic",
            n_events_loaded=0,
        )

    epoch_centers = onsets[selected] + durations[selected] / 2.0
    spindle_onsets = epoch_centers - spindle_duration_sec / 2.0

    n = len(selected)
    rng = np.random.default_rng(rng_seed)
    confidence = rng.uniform(0.6, 0.95, size=n)

    return EventTable(
        onset_sec=spindle_onsets,
        duration_sec=np.full(n, spindle_duration_sec),
        event_type=np.array(["sleep_spindle"] * n, dtype=object),
        source=np.array([source] * n, dtype=object),
        channel=np.array(["synthetic"] * n, dtype=object),
        confidence=confidence,
        source_path="synthetic",
        n_events_loaded=n,
    )


# ---------------------------------------------------------------------------
# Minimal MNPSPayload from staging data
# ---------------------------------------------------------------------------

def make_payload_from_stages(
    onsets: np.ndarray,
    durations: np.ndarray,
    stages: np.ndarray,
    *,
    rng_seed: int = 0,
) -> "MNPSPayload":
    """Build a synthetic MNPSPayload aligned to the staging epochs.

    MNPS values are random but stage-stratified (N2 has slightly lower entropy,
    higher mobility — purely for visual plausibility, not biology).
    """
    from mndm.schema import MNPSPayload

    n = len(onsets)
    rng = np.random.default_rng(rng_seed)

    w_start = onsets.astype(np.float64)
    w_end = (onsets + durations).astype(np.float64)
    t = (w_start + w_end) / 2.0

    # Generate stage-plausible random MNPS
    x = rng.random((n, 3)).astype(np.float32)
    # Slightly modulate N2 windows for visual interest
    is_n2 = stages == 2
    x[is_n2, 0] += 0.15  # higher mobility
    x[is_n2, 2] -= 0.1   # lower entropy
    x = np.clip(x, 0, 1).astype(np.float32)
    x_dot = rng.standard_normal((n, 3)).astype(np.float32) * 0.05

    # 9D coords
    coords_9d = rng.random((n, 9)).astype(np.float32)
    coords_9d_names = ["m_a", "m_e", "m_o", "d_n", "d_l", "d_s", "e_e", "e_s", "e_m"]

    return MNPSPayload(
        time=t,
        x=x,
        x_dot=x_dot,
        stage=stages.astype(np.int8),
        window_start=w_start,
        window_end=w_end,
        coords_9d=coords_9d,
        coords_9d_names=coords_9d_names,
        attrs={
            "fs_out": float(1.0 / np.median(np.diff(t))) if len(t) > 1 else 1.0,
            "window_sec": float(np.median(durations)),
            "overlap": 0.0,
            "note": "synthetic_mnps_from_staging",
        },
    )


# ---------------------------------------------------------------------------
# Single-subject smoke run
# ---------------------------------------------------------------------------

def run_smoke(
    *,
    dataset_label: str,
    subject_id: str,
    onsets: np.ndarray,
    durations: np.ndarray,
    stages: np.ndarray,
) -> Dict:
    """Run the full spindle pipeline for one subject and return a QC report."""
    from mndm.pipeline.event_annotations import validate_event_table
    from mndm.pipeline.event_alignment import AlignmentConfig, align_events_to_windows
    from mndm.pipeline.control_matching import MatchingConfig, build_matched_controls
    from mndm.pipeline.event_locked_export import (
        ExportConfig,
        build_event_locked_table,
        write_event_locked_csv,
        event_locked_export_manifest_entry,
    )

    t0 = time.perf_counter()

    payload = make_payload_from_stages(onsets, durations, stages)
    event_table = make_synthetic_spindles(onsets, durations, stages)

    validation_warnings = validate_event_table(event_table)

    alignment_cfg = AlignmentConfig(
        reference="onset",
        stage_transition_margin_sec=30.0,
    )
    alignment = align_events_to_windows(
        event_table,
        window_start=payload.window_start,
        window_end=payload.window_end,
        time=payload.time,
        stage=payload.stage,
        config=alignment_cfg,
    )

    control_cfg = MatchingConfig(
        n_controls_per_event=3,
        target_stage=2,
        exclusion_margin_sec=30.0,
        time_of_night_quartile_match=True,
        seed=1729,
    )
    controls = build_matched_controls(
        event_table,
        time=payload.time,
        window_start=payload.window_start,
        window_end=payload.window_end,
        stage=payload.stage,
        config=control_cfg,
    )

    rows = build_event_locked_table(
        payload=payload,
        alignment=alignment,
        controls=controls,
        event_table=event_table,
        subject_id=subject_id,
        dataset_id=dataset_label,
        config=ExportConfig(include_coords_9d=True, write_csv=True),
    )

    csv_path = OUT_DIR / f"smoke_{dataset_label}_{subject_id}.csv"
    write_event_locked_csv(rows, csv_path)

    manifest = event_locked_export_manifest_entry(
        rows, alignment=alignment, controls=controls, out_paths=[csv_path]
    )
    qc_path = OUT_DIR / f"smoke_{dataset_label}_{subject_id}_qc.json"
    with qc_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    elapsed = time.perf_counter() - t0

    spindle_rows = [r for r in rows if r["condition"] == "spindle_event"]
    control_rows = [r for r in rows if r["condition"] == "matched_control"]

    # Compute per-bin spindle counts
    bin_counts: Dict[str, int] = {}
    for r in spindle_rows:
        bin_counts[r["bin_label"]] = bin_counts.get(r["bin_label"], 0) + 1

    # Spot-check: fraction of spindle event rows with is_event_window=1
    event_window_frac = (
        sum(1 for r in spindle_rows if r.get("is_event_window", 0)) / len(spindle_rows)
        if spindle_rows else 0.0
    )

    return {
        "dataset": dataset_label,
        "subject": subject_id,
        "total_staging_epochs": len(onsets),
        "n2_epochs": int(np.sum(stages == 2)),
        "synthetic_spindles_injected": event_table.n,
        "validation_warnings": validation_warnings,
        "n_events_aligned": alignment.qc["n_events_aligned"],
        "n_events_excluded_transition": alignment.qc["n_events_excluded_stage_transition"],
        "n_spindle_rows": len(spindle_rows),
        "n_control_rows": len(control_rows),
        "match_success_rate": controls.qc["match_success_rate"],
        "n_full_matches": controls.qc["n_events_with_full_match"],
        "n_partial_matches": controls.qc["n_events_with_partial_match"],
        "n_no_matches": controls.qc["n_events_with_no_match"],
        "bin_counts": bin_counts,
        "event_window_frac": round(event_window_frac, 3),
        "mnps_finite_frac": round(
            sum(1 for r in spindle_rows if r.get("mnps_finite", 0)) / max(1, len(spindle_rows)), 3
        ),
        "coords_9d_present": any("m_a" in r for r in spindle_rows[:1]),
        "csv_out": str(csv_path),
        "qc_json_out": str(qc_path),
        "elapsed_sec": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results = []

    # --- ds005555 sub-1 PSG ------------------------------------------------
    ds5_events = Path(r"M:\datasets\received\openneuro\ds005555\sub-1\eeg\sub-1_task-Sleep_acq-psg_events.tsv")
    if ds5_events.exists():
        print("\n=== ds005555 / sub-1 (PSG) ===")
        onsets, durations, stages = load_ds005555_stages(ds5_events)
        qc = run_smoke(
            dataset_label="ds005555",
            subject_id="sub-1",
            onsets=onsets,
            durations=durations,
            stages=stages,
        )
        results.append(qc)
        _print_qc(qc)
    else:
        print(f"SKIP ds005555: {ds5_events} not found")

    # --- ANPHY EPCTL01 -----------------------------------------------------
    anphy_ann = Path(r"M:\datasets\received\ANPHY\Subjects\EPCTL01\test1.txt")
    if anphy_ann.exists():
        print("\n=== ANPHY / EPCTL01 ===")
        onsets, durations, stages = load_anphy_stages(anphy_ann)
        qc = run_smoke(
            dataset_label="ANPHY",
            subject_id="EPCTL01",
            onsets=onsets,
            durations=durations,
            stages=stages,
        )
        results.append(qc)
        _print_qc(qc)
    else:
        print(f"SKIP ANPHY: {anphy_ann} not found")

    # --- Summary -----------------------------------------------------------
    print("\n=== SMOKE TEST SUMMARY ===")
    for r in results:
        status = "PASS" if r["mnps_finite_frac"] == 1.0 and r["n_spindle_rows"] > 0 else "WARN"
        print(f"  [{status}] {r['dataset']} / {r['subject']}  "
              f"spindles={r['synthetic_spindles_injected']}  "
              f"aligned={r['n_events_aligned']}  "
              f"spindle_rows={r['n_spindle_rows']}  "
              f"control_rows={r['n_control_rows']}  "
              f"match_rate={r['match_success_rate']:.2f}  "
              f"elapsed={r['elapsed_sec']}s")

    # Write combined summary
    summary_path = OUT_DIR / "smoke_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\nSummary JSON: {summary_path}")


def _print_qc(qc: Dict) -> None:
    print(f"  Staging epochs    : {qc['total_staging_epochs']} total, {qc['n2_epochs']} N2")
    print(f"  Synthetic spindles: {qc['synthetic_spindles_injected']} injected")
    if qc["validation_warnings"]:
        print(f"  Validation warns  : {qc['validation_warnings']}")
    print(f"  Aligned events    : {qc['n_events_aligned']} ({qc['n_events_excluded_transition']} excluded at transitions)")
    print(f"  Spindle rows      : {qc['n_spindle_rows']}")
    print(f"  Control rows      : {qc['n_control_rows']}")
    print(f"  Match rate        : {qc['match_success_rate']:.2f}  "
          f"(full={qc['n_full_matches']}, partial={qc['n_partial_matches']}, none={qc['n_no_matches']})")
    print(f"  Bin counts        : {qc['bin_counts']}")
    print(f"  Event window frac : {qc['event_window_frac']}")
    print(f"  MNPS finite frac  : {qc['mnps_finite_frac']}")
    print(f"  coords_9d present : {qc['coords_9d_present']}")
    print(f"  CSV out           : {qc['csv_out']}")
    print(f"  Elapsed           : {qc['elapsed_sec']}s")


if __name__ == "__main__":
    main()
