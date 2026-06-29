"""HRV superwindow robustness sidecar for ds003838.

Re-reads the raw ECG EEGLAB .set files, re-detects R-peaks once per recording,
then sweeps over HRV superwindow sizes and NN-count cutoffs.  Outputs a single
long-format parquet that can be joined to block_native_windows.parquet via
(subject_id, block_id, source_window_index).

Usage
-----
  # All subjects, write next to features.parquet
  python project/scripts/hrv_robustness_sidecar_ds003838.py

  # A few subjects for smoke test
  python project/scripts/hrv_robustness_sidecar_ds003838.py --subjects 35 36 37

  # Custom output path
  python project/scripts/hrv_robustness_sidecar_ds003838.py --out J:/processed/openneuro/ds003838/hrv_robustness.parquet

Columns in output
-----------------
  subject_id, dataset_id, task, block_id, source_window_index,
  window_center_sec, block_duration_sec, stage_code,
  hrv_window_sec,        # 15 / 30 / 60 / block_level
  hrv_window_label,      # "15s" / "30s" / "60s" / "block"
  nn_cutoff,             # 10 / 20 / 30  (minimum NN count required)
  nn_count,              # actual NN count within this superwindow
  nn_passes_cutoff,      # bool
  ecg_hrv_rmssd_ms,
  ecg_hrv_sdnn_ms,
  ecg_hrv_hr_mean_bpm,
  ecg_hrv_pnn50,
  ecg_hrv_quality_score,
  ecg_hrv_artifact_fraction,
  ecg_hrv_coverage_fraction,
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths – edit if your installation differs
# ---------------------------------------------------------------------------
PROCESSED_ROOT = Path("J:/processed/openneuro/ds003838")
RECEIVED_ROOT = Path("G:/Science_Datasets_longtime_storage/openneuro/received/ds003838")
FEATURES_PARQUET = PROCESSED_ROOT / "features.parquet"
FILE_INDEX_CSV = PROCESSED_ROOT / "file_index.csv"

HRV_WINDOW_SIZES_S: List[float] = [15.0, 30.0, 60.0]   # + block_level added per-row
NN_CUTOFFS: List[int] = [10, 20, 30]

# ECG bandpass (same defaults as mndm/features/ecg.py)
BANDPASS_LOW_HZ = 5.0
BANDPASS_HIGH_HZ = 20.0
BANDPASS_ORDER = 3
REFRACTORY_S = 0.3
PROMINENCE_MULT = 1.0
RR_MIN_S = 0.3
RR_MAX_S = 2.0
PNN50_THRESHOLD_S = 0.050   # 50 ms


# ---------------------------------------------------------------------------
# Helpers – copied from mndm.features.ecg to keep sidecar self-contained
# ---------------------------------------------------------------------------

def _window_bounds(
    center_s: float,
    window_s: float,
    total_duration_s: float,
) -> tuple[float, float]:
    half = 0.5 * window_s
    start = center_s - half
    end = center_s + half
    if total_duration_s <= window_s:
        return 0.0, max(total_duration_s, 0.0)
    if start < 0.0:
        end = min(total_duration_s, end - start)
        start = 0.0
    if end > total_duration_s:
        start = max(0.0, start - (end - total_duration_s))
        end = total_duration_s
    return float(max(0.0, start)), float(min(total_duration_s, end))


def _quality_score(
    nn_count: int, min_nn: int, coverage_frac: float, artifact_frac: float
) -> float:
    nn_support = min(1.0, float(nn_count) / max(float(min_nn), 1.0))
    cov = float(np.clip(coverage_frac, 0.0, 1.0))
    art = 1.0 - float(np.clip(artifact_frac, 0.0, 1.0))
    return float(np.clip(nn_support * cov * art, 0.0, 1.0))


def compute_hrv_one_window(
    *,
    peak_times_s: np.ndarray,
    rr_all_s: np.ndarray,
    center_s: float,
    window_s: float,
    total_duration_s: float,
    nn_cutoff: int,
) -> Dict[str, Any]:
    """Compute HRV metrics for one (center, window_size, nn_cutoff) combination."""
    sw_start, sw_end = _window_bounds(center_s, window_s, total_duration_s)
    left = int(np.searchsorted(peak_times_s, sw_start, side="left"))
    right = int(np.searchsorted(peak_times_s, sw_end, side="left"))
    raw_rr = rr_all_s[left : max(left, right - 1)]
    valid_mask = (raw_rr >= RR_MIN_S) & (raw_rr <= RR_MAX_S)
    nn = raw_rr[valid_mask]
    invalid_count = int(np.sum(~valid_mask)) if raw_rr.size > 0 else 0
    nn_count = int(nn.size)
    coverage_frac = float((sw_end - sw_start) / window_s) if window_s > 0 else 0.0
    artifact_frac = float(invalid_count / raw_rr.size) if raw_rr.size > 0 else np.nan
    art_for_score = artifact_frac if np.isfinite(artifact_frac) else 1.0
    quality_score = _quality_score(nn_count, nn_cutoff, coverage_frac, art_for_score)

    nan = float("nan")
    if nn_count < 1:
        return dict(
            nn_count=nn_count, nn_passes_cutoff=False,
            ecg_hrv_hr_mean_bpm=nan, ecg_hrv_rmssd_ms=nan, ecg_hrv_sdnn_ms=nan,
            ecg_hrv_pnn50=nan, ecg_hrv_quality_score=quality_score,
            ecg_hrv_artifact_fraction=artifact_frac, ecg_hrv_coverage_fraction=coverage_frac,
        )

    hr_mean_bpm = float(60.0 / np.mean(nn)) if np.mean(nn) > 0 else nan
    sdnn_ms = float(np.std(nn, ddof=1) * 1000.0) if nn_count >= 2 else nan
    dnn = np.diff(nn) if nn_count >= 3 else np.asarray([], dtype=float)
    rmssd_ms = float(np.sqrt(np.mean(dnn ** 2)) * 1000.0) if dnn.size > 0 else nan
    pnn50 = float(np.mean(np.abs(dnn) > PNN50_THRESHOLD_S)) if dnn.size > 0 else nan

    passes = bool(nn_count >= nn_cutoff)
    # zero out metrics if below cutoff (they would not be used in analysis)
    if not passes:
        rmssd_ms = nan
        sdnn_ms = nan
        pnn50 = nan

    return dict(
        nn_count=nn_count, nn_passes_cutoff=passes,
        ecg_hrv_hr_mean_bpm=hr_mean_bpm, ecg_hrv_rmssd_ms=rmssd_ms,
        ecg_hrv_sdnn_ms=sdnn_ms, ecg_hrv_pnn50=pnn50,
        ecg_hrv_quality_score=quality_score,
        ecg_hrv_artifact_fraction=artifact_frac,
        ecg_hrv_coverage_fraction=coverage_frac,
    )


def detect_rpeaks(ecg_1d: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """Run-level R-peak detection using NeuroKit2 (Pan-Tompkins-style) with polarity-aware scipy fallback.

    This matches the corrected detector in mndm.features.ecg after the T-wave double-detection fix.
    """
    # Try NeuroKit2 first (avoids T-wave double-detection from np.abs)
    try:
        import warnings
        import neurokit2 as nk  # type: ignore
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, info = nk.ecg_process(ecg_1d, sampling_rate=int(sfreq))
        peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    except Exception:
        # Polarity-aware scipy fallback (no np.abs)
        nyquist = sfreq * 0.5
        hi = min(BANDPASS_HIGH_HZ, nyquist * 0.99)
        lo = max(0.01, BANDPASS_LOW_HZ)
        if hi <= lo:
            lo, hi = 5.0, min(20.0, nyquist * 0.99)
        try:
            b, a = sp_signal.butter(BANDPASS_ORDER, [lo / nyquist, hi / nyquist], btype="bandpass")
            filt = sp_signal.filtfilt(b, a, ecg_1d)
        except Exception:
            filt = ecg_1d - np.median(ecg_1d)
        centered = filt - np.median(filt)
        min_dist = max(1, int(round(REFRACTORY_S * sfreq)))

        def _side(s):
            c = np.clip(s, 0.0, None)
            pv = c[c > 0]
            if pv.size == 0:
                return np.array([], dtype=int), 0.0
            prom = max(1e-6, PROMINENCE_MULT * 1.4826 * (float(np.median(np.abs(pv))) + 1e-8))
            pk, pr = sp_signal.find_peaks(c, distance=min_dist, prominence=prom)
            return pk.astype(int), float(np.sum(pr["prominences"])) if pk.size else 0.0

        pp, sp_pos = _side(centered)
        pn, sp_neg = _side(-centered)
        peaks = pp if sp_pos >= sp_neg else pn

    peak_times_s = peaks.astype(np.float64) / sfreq
    rr_all_s = np.diff(peak_times_s) if peak_times_s.size >= 2 else np.asarray([], dtype=np.float64)
    return peak_times_s, rr_all_s


def load_ecg_from_set(set_path: Path) -> Optional[tuple[np.ndarray, float]]:
    """Load raw ECG from an EEGLAB .set file via MNE. Returns (1d_ecg, sfreq) or None."""
    try:
        import mne  # type: ignore
        mne.set_log_level("WARNING")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
    except Exception as exc:
        logger.warning("Cannot load %s: %s", set_path, exc)
        return None
    sfreq = float(raw.info["sfreq"])
    # Try to find the ECG channel by type, then by name.
    ecg_chs = mne.pick_types(raw.info, ecg=True, exclude=[])
    if len(ecg_chs) == 0:
        ecg_name_guesses = [
            n for n in raw.ch_names
            if any(kw in n.upper() for kw in ("ECG", "EKG", "CARDIAC"))
        ]
        if not ecg_name_guesses:
            logger.warning("No ECG channel found in %s", set_path.name)
            return None
        ecg_chs = [raw.ch_names.index(ecg_name_guesses[0])]
    data, _ = raw[ecg_chs[0], :]
    return np.asarray(data[0], dtype=float), sfreq


# ---------------------------------------------------------------------------
# Core sweep logic
# ---------------------------------------------------------------------------

def process_one_recording(
    ecg_path: Path,
    bn_windows: pd.DataFrame,
    subject_id: str,
    task: str,
    dataset_id: str = "ds003838",
) -> List[Dict[str, Any]]:
    """Sweep HRV windows + cutoffs for one ECG file and its block-native windows."""
    result = load_ecg_from_set(ecg_path)
    if result is None:
        return []
    ecg_1d, sfreq = result
    total_duration_s = float(len(ecg_1d) / sfreq)
    logger.info("  %s / %s: %.1f s, sfreq=%.0f", subject_id, task, total_duration_s, sfreq)
    peak_times_s, rr_all_s = detect_rpeaks(ecg_1d, sfreq)
    if rr_all_s.size == 0:
        logger.warning("  No R-peaks found for %s / %s", subject_id, task)
        return []

    rows: List[Dict[str, Any]] = []
    for _, win in bn_windows.iterrows():
        center_s = float(win.get("window_center_sec", 0.0))
        block_dur = float(win.get("block_duration_sec", 60.0))
        block_id = win.get("block_id", -1)
        src_widx = win.get("source_window_index", -1)
        stage_code = win.get("stage_code", np.nan)

        window_sizes = HRV_WINDOW_SIZES_S + [block_dur]
        window_labels = ["15s", "30s", "60s", "block"]

        for ws, wlabel in zip(window_sizes, window_labels):
            for cutoff in NN_CUTOFFS:
                metrics = compute_hrv_one_window(
                    peak_times_s=peak_times_s,
                    rr_all_s=rr_all_s,
                    center_s=center_s,
                    window_s=ws,
                    total_duration_s=total_duration_s,
                    nn_cutoff=cutoff,
                )
                rows.append({
                    "subject_id": subject_id,
                    "dataset_id": dataset_id,
                    "task": task,
                    "block_id": block_id,
                    "source_window_index": src_widx,
                    "window_center_sec": center_s,
                    "block_duration_sec": block_dur,
                    "stage_code": stage_code,
                    "hrv_window_sec": ws,
                    "hrv_window_label": wlabel,
                    "nn_cutoff": cutoff,
                    **metrics,
                })
    return rows


def find_latest_run_dir(processed_root: Path, dataset_id: str) -> Optional[Path]:
    run_dirs = sorted(processed_root.glob(f"neuralmanifolddynamics_{dataset_id}_*"))
    return run_dirs[-1] if run_dirs else None


def main() -> None:
    parser = argparse.ArgumentParser(description="HRV robustness sidecar for ds003838")
    parser.add_argument("--subjects", nargs="*", default=None, metavar="N",
                        help="Subjects to process (e.g. 35 36). Default: all in file_index.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output parquet path. Default: <processed>/hrv_robustness_sidecar.parquet")
    parser.add_argument("--received-dir", type=Path, default=RECEIVED_ROOT,
                        help="Root of received ds003838 dataset (BIDS dir).")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_ROOT,
                        help="Root of processed ds003838 outputs.")
    args = parser.parse_args()

    processed_root = args.processed_dir
    received_root = args.received_dir
    out_path = args.out or (processed_root / "hrv_robustness_sidecar.parquet")

    # Load file index
    file_index_csv = processed_root / "file_index.csv"
    if not file_index_csv.exists():
        logger.error("file_index.csv not found at %s", file_index_csv)
        sys.exit(1)
    file_df = pd.read_csv(file_index_csv)
    ecg_df = file_df[file_df["modality"].str.lower() == "ecg"].copy()
    if ecg_df.empty:
        logger.error("No ECG rows in file_index.csv")
        sys.exit(1)

    # Filter subjects (accept both "35" and "sub-035" as input)
    if args.subjects:
        wanted_raw = set(args.subjects)
        wanted_padded = {f"sub-{int(s):03d}" for s in wanted_raw if s.isdigit()}
        wanted_prefixed = {s for s in wanted_raw if s.startswith("sub-")}
        wanted = wanted_padded | wanted_prefixed
        ecg_df = ecg_df[ecg_df["subject"].apply(lambda x: f"sub-{int(x):03d}").isin(wanted)]
        if ecg_df.empty:
            logger.error("No matching ECG files for subjects %s", wanted)
            sys.exit(1)

    # Find latest run dir for block_native_windows
    run_dir = find_latest_run_dir(processed_root, "ds003838")
    if run_dir is None:
        logger.error("No neuralmanifolddynamics_ds003838_* run dir found in %s", processed_root)
        sys.exit(1)
    logger.info("Using run dir: %s", run_dir)

    # Pre-load all block_native_windows into a dict keyed by (subject_id, task)
    bn_by_key: Dict[tuple, pd.DataFrame] = {}
    for sub_dir in sorted(run_dir.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        pq = sub_dir / "block_native_windows.parquet"
        csv = sub_dir / "block_native_windows.csv"
        df = None
        if pq.exists():
            try:
                df = pd.read_parquet(pq)
            except Exception:
                df = None
        if df is None and csv.exists():
            try:
                df = pd.read_csv(csv)
            except Exception:
                pass
        if df is None or df.empty:
            continue
        # Infer task from subdirectory name when column is absent
        if "task" not in df.columns:
            dir_name = sub_dir.name
            parts = dir_name.split("_", 1)
            inferred_task = parts[1] if len(parts) == 2 else "unknown"
            df = df.copy()
            df["task"] = inferred_task
        for (s, t), grp in df.groupby(["subject_id", "task"]):
            bn_by_key[(str(s), str(t))] = grp.reset_index(drop=True)

    logger.info("Loaded block_native_windows for %d (subject, task) pairs", len(bn_by_key))

    # Task normalization: file_index uses BIDS task name, run dirs use normalized name
    TASK_NORMALIZE = {"memory": "digit_span", "rest": "rest"}

    all_rows: List[Dict[str, Any]] = []
    for _, row in ecg_df.iterrows():
        raw_path_rel = str(row["path"])
        subject_id = f"sub-{int(row['subject']):03d}"
        raw_task = str(row.get("task", "unknown"))
        task = TASK_NORMALIZE.get(raw_task, raw_task)
        set_path = received_root / raw_path_rel
        if not set_path.exists():
            logger.warning("ECG file not found: %s", set_path)
            continue
        bn_key = (subject_id, task)
        if bn_key not in bn_by_key:
            logger.warning("No block_native_windows for sub-%s / %s — skipping", subject_id, task)
            continue
        bn_windows = bn_by_key[bn_key]
        logger.info("Processing %s / %s (%d windows)", subject_id, task, len(bn_windows))
        rows = process_one_recording(
            ecg_path=set_path,
            bn_windows=bn_windows,
            subject_id=subject_id,
            task=task,
        )
        all_rows.extend(rows)

    if not all_rows:
        logger.error("No HRV rows produced — check ECG file paths and block_native_windows.")
        sys.exit(1)

    out_df = pd.DataFrame(all_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    logger.info(
        "Wrote %d rows (%d subjects, %d window/cutoff combos) → %s",
        len(out_df),
        out_df["subject_id"].nunique(),
        len(HRV_WINDOW_SIZES_S) + 1,    # +1 for block_level
        out_path,
    )

    # Quick summary: median RMSSD by window size (for the 60s-compatible nn>=20 cutoff)
    ref = out_df[out_df["nn_cutoff"] == 20]
    if not ref.empty:
        summary = (
            ref.groupby("hrv_window_label")["ecg_hrv_rmssd_ms"]
            .agg(["median", "count", lambda x: x.notna().mean()])
            .rename(columns={"<lambda_0>": "finite_frac"})
        )
        print("\n=== Quick summary: median RMSSD (ms) by window size @ nn>=20 ===")
        print(summary.to_string())

        # Pass rate by window + cutoff
        pass_rates = (
            out_df.groupby(["hrv_window_label", "nn_cutoff"])["nn_passes_cutoff"]
            .mean()
            .unstack("nn_cutoff")
        )
        print("\n=== Window pass-rate by (window, nn_cutoff) ===")
        print(pass_rates.to_string())


if __name__ == "__main__":
    main()
