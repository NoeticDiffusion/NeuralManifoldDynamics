"""Patch ECG columns in features.parquet using the corrected NeuroKit2 detector.

This script:
1. Reads the current features.parquet (which has T-wave double-detection artefacts)
2. For each (subject, task) pair in file_index.csv, loads the raw ECG .set file
3. Re-detects R-peaks with NeuroKit2 (Pan-Tompkins)
4. Recomputes per-epoch ECG features using the SAME t_start/t_end windows from parquet
5. Overwrites only the ECG columns; EEG features are preserved unchanged
6. Writes the corrected parquet alongside the original

Usage
-----
  python project/scripts/ecg_patch_features.py
  python project/scripts/ecg_patch_features.py --dry-run       # only print stats

The original features.parquet is preserved as features_pre_ecg_patch.parquet.
The patched result is saved to features.parquet (in-place).
"""

from __future__ import annotations

import argparse
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURES_PARQUET = Path("J:/processed/openneuro/ds003838/features.parquet")
FILE_INDEX_CSV = Path("J:/processed/openneuro/ds003838/file_index.csv")
RECEIVED_ROOT = Path("G:/Science_Datasets_longtime_storage/openneuro/received/ds003838")

# Must match mndm config
PROCESSED_SFREQ = 250.0
BANDPASS_LOW = 5.0
BANDPASS_HIGH = 20.0
BANDPASS_ORDER = 3
REFRACTORY_S = 0.3
RR_MIN_S = 0.30
RR_MAX_S = 2.00
HRV_SUPERWINDOW_S = 60.0
MIN_NN_INTERVALS = 20
PNN50_THRESHOLD_MS = 50.0

ECG_COLS = [
    "ecg_hr_bpm", "ecg_rr_mean", "ecg_rr_cv", "ecg_sdnn", "ecg_rmssd",
    "ecg_peak_count", "ecg_quality_score", "qc_ok_ecg",
    "ecg_hrv_hr_mean_bpm", "ecg_hrv_ibi_mean_ms", "ecg_hrv_sdnn_ms",
    "ecg_hrv_rmssd_ms", "ecg_hrv_pnn50", "ecg_hrv_nn_count",
    "ecg_hrv_artifact_fraction", "ecg_hrv_coverage_fraction",
    "ecg_hrv_quality_score", "qc_ok_ecg_hrv",
    "ecg_hrv_sampen", "ecg_hrv_dfa_alpha1",
]


def _detect_rpeaks_nk2(ecg_1d: np.ndarray, sfreq: float) -> np.ndarray:
    """Detect R-peaks via NeuroKit2. Fallback to polarity-aware scipy."""
    try:
        import neurokit2 as nk  # type: ignore
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, info = nk.ecg_process(ecg_1d, sampling_rate=int(sfreq))
        return np.asarray(info["ECG_R_Peaks"], dtype=int)
    except Exception:
        pass
    # polarity-aware scipy fallback
    nyq = sfreq * 0.5
    hi = min(BANDPASS_HIGH, nyq * 0.99)
    lo = max(0.01, BANDPASS_LOW)
    try:
        b, a = sp_signal.butter(BANDPASS_ORDER, [lo / nyq, hi / nyq], btype="bandpass")
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
        prom = max(1e-6, 1.4826 * (float(np.median(np.abs(pv))) + 1e-8))
        pk, pr = sp_signal.find_peaks(c, distance=min_dist, prominence=prom)
        return pk.astype(int), float(np.sum(pr["prominences"])) if pk.size else 0.0

    pp, sp = _side(centered)
    pn, sn = _side(-centered)
    return pp if sp >= sn else pn


def _hrv_superwindow(peak_times_s: np.ndarray, center_s: float,
                     total_dur_s: float) -> Dict[str, Any]:
    half = HRV_SUPERWINDOW_S / 2.0
    t0, t1 = max(0.0, center_s - half), min(total_dur_s, center_s + half)
    mask = (peak_times_s >= t0) & (peak_times_s <= t1)
    sub = peak_times_s[mask]
    if sub.size < 2:
        return {k: np.nan for k in [
            "ecg_hrv_hr_mean_bpm", "ecg_hrv_ibi_mean_ms", "ecg_hrv_sdnn_ms",
            "ecg_hrv_rmssd_ms", "ecg_hrv_pnn50", "ecg_hrv_nn_count",
            "ecg_hrv_artifact_fraction", "ecg_hrv_coverage_fraction",
            "ecg_hrv_quality_score", "qc_ok_ecg_hrv",
            "ecg_hrv_sampen", "ecg_hrv_dfa_alpha1",
        ]}
    rr_s = np.diff(sub)
    valid = (rr_s >= RR_MIN_S) & (rr_s <= RR_MAX_S)
    nn = rr_s[valid]
    art_frac = 1.0 - float(valid.mean()) if valid.size > 0 else 1.0
    cov_frac = float(nn.size) / max(1.0, (t1 - t0) / 0.8)

    if nn.size < MIN_NN_INTERVALS:
        row: Dict[str, Any] = {k: np.nan for k in [
            "ecg_hrv_hr_mean_bpm", "ecg_hrv_ibi_mean_ms", "ecg_hrv_sdnn_ms",
            "ecg_hrv_rmssd_ms", "ecg_hrv_pnn50", "ecg_hrv_quality_score",
            "ecg_hrv_sampen", "ecg_hrv_dfa_alpha1",
        ]}
        row.update({
            "ecg_hrv_nn_count": int(nn.size),
            "ecg_hrv_artifact_fraction": art_frac,
            "ecg_hrv_coverage_fraction": cov_frac,
            "qc_ok_ecg_hrv": False,
        })
        return row

    ibi_ms = nn * 1000.0
    drr = np.diff(nn)
    hr = float(60.0 / np.mean(nn))
    rmssd = float(np.sqrt(np.mean(drr ** 2)) * 1000.0) if drr.size > 0 else np.nan
    sdnn = float(np.std(nn, ddof=1) * 1000.0) if nn.size > 1 else np.nan
    pnn50 = float(np.mean(np.abs(drr) * 1000.0 > PNN50_THRESHOLD_MS)) if drr.size > 0 else np.nan
    quality = float(min(1.0, nn.size / max(1.0, (t1 - t0) / 0.8)))

    # HRV complexity (optional, keep nan if not computable)
    sampen = np.nan
    dfa = np.nan
    if nn.size >= 50:
        try:
            import antropy as ant  # type: ignore
            sampen = float(ant.sample_entropy(ibi_ms, order=2, metric="chebyshev"))
        except Exception:
            pass
    if nn.size >= 16:
        try:
            import nolds  # type: ignore
            dfa = float(nolds.dfa(ibi_ms, nvals=range(4, 12)))
        except Exception:
            pass

    return {
        "ecg_hrv_hr_mean_bpm": hr,
        "ecg_hrv_ibi_mean_ms": float(np.mean(ibi_ms)),
        "ecg_hrv_sdnn_ms": sdnn,
        "ecg_hrv_rmssd_ms": rmssd,
        "ecg_hrv_pnn50": pnn50,
        "ecg_hrv_nn_count": int(nn.size),
        "ecg_hrv_artifact_fraction": art_frac,
        "ecg_hrv_coverage_fraction": cov_frac,
        "ecg_hrv_quality_score": quality,
        "qc_ok_ecg_hrv": True,
        "ecg_hrv_sampen": sampen,
        "ecg_hrv_dfa_alpha1": dfa,
    }


def load_ecg_at_250hz(set_path: Path) -> Optional[Tuple[np.ndarray, float]]:
    """Load ECG from .set file and resample to 250 Hz (matches mndm processing)."""
    try:
        import mne
        mne.set_log_level("ERROR")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
        ecg_chs = mne.pick_types(raw.info, ecg=True, exclude=[])
        if len(ecg_chs) == 0:
            ecg_chs = [i for i, n in enumerate(raw.ch_names)
                       if any(k in n.upper() for k in ("ECG", "EKG", "CARDIAC"))]
        if not ecg_chs:
            return None
        if raw.info["sfreq"] != PROCESSED_SFREQ:
            raw.resample(PROCESSED_SFREQ)
        data, _ = raw[ecg_chs[0], :]
        return np.asarray(data[0], dtype=float), float(raw.info["sfreq"])
    except Exception as e:
        logger.warning("Cannot load %s: %s", set_path, e)
        return None


def compute_ecg_row(epoch_row: dict, peak_times_s: np.ndarray,
                    sfreq: float, n_samples: int) -> Dict[str, Any]:
    """Compute corrected ECG features for one epoch window."""
    t0_raw = epoch_row["t_start"]
    t1_raw = epoch_row["t_end"]
    if not (np.isfinite(t0_raw) and np.isfinite(t1_raw)):
        return {col: np.nan for col in [
            "ecg_hr_bpm", "ecg_rr_mean", "ecg_rr_cv", "ecg_sdnn", "ecg_rmssd",
            "ecg_peak_count", "ecg_quality_score", "qc_ok_ecg",
            "ecg_hrv_hr_mean_bpm", "ecg_hrv_ibi_mean_ms", "ecg_hrv_sdnn_ms",
            "ecg_hrv_rmssd_ms", "ecg_hrv_pnn50", "ecg_hrv_nn_count",
            "ecg_hrv_artifact_fraction", "ecg_hrv_coverage_fraction",
            "ecg_hrv_quality_score", "qc_ok_ecg_hrv",
            "ecg_hrv_sampen", "ecg_hrv_dfa_alpha1",
        ]}
    t0, t1 = float(t0_raw), float(t1_raw)
    length_s = t1 - t0

    # Peaks within this epoch
    mask = (peak_times_s >= t0) & (peak_times_s < t1)
    epoch_peaks = peak_times_s[mask]
    rr_s = np.diff(epoch_peaks) if epoch_peaks.size >= 2 else np.array([], dtype=float)
    rr_s = rr_s[(rr_s >= RR_MIN_S) & (rr_s <= RR_MAX_S)] if rr_s.size > 0 else rr_s

    rr_mean = float(np.mean(rr_s)) if rr_s.size > 0 else np.nan
    hr_bpm = float(60.0 / rr_mean) if np.isfinite(rr_mean) and rr_mean > 0 else np.nan
    rr_cv = (float(np.std(rr_s, ddof=1) / rr_mean)
             if rr_s.size >= 2 and rr_mean > 0 else np.nan)
    sdnn = float(np.std(rr_s, ddof=1)) if rr_s.size >= 2 else np.nan
    drr = np.diff(rr_s) if rr_s.size >= 3 else np.array([], dtype=float)
    rmssd = float(np.sqrt(np.mean(drr ** 2))) if drr.size > 0 else np.nan
    peak_count = int(epoch_peaks.size)
    quality_score = float(min(1.0, rr_s.size / max(length_s * 1.5, 1.0))) if rr_s.size > 0 else 0.0

    center_s = 0.5 * (t0 + t1)
    total_dur = float(n_samples / sfreq)
    hrv_super = _hrv_superwindow(peak_times_s, center_s, total_dur)

    row = {
        "ecg_hr_bpm": hr_bpm,
        "ecg_rr_mean": rr_mean,
        "ecg_rr_cv": rr_cv,
        "ecg_sdnn": sdnn,
        "ecg_rmssd": rmssd,
        "ecg_peak_count": peak_count,
        "ecg_quality_score": quality_score,
        "qc_ok_ecg": bool(np.isfinite(hr_bpm) and peak_count >= 2),
    }
    row.update(hrv_super)
    return row


def patch_subject_task(
    sub: int, task: str, set_path: Path, feat_df: pd.DataFrame
) -> Optional[pd.DataFrame]:
    """Recompute ECG features for one (subject, task) and return patched rows."""
    mask = (feat_df["subject"] == sub) & (feat_df["task"] == task)
    sub_df = feat_df[mask].copy()
    if sub_df.empty:
        return None

    result = load_ecg_at_250hz(set_path)
    if result is None:
        return None
    ecg_1d, sfreq = result
    peaks_idx = _detect_rpeaks_nk2(ecg_1d, sfreq)
    peak_times_s = peaks_idx.astype(np.float64) / sfreq
    n_samples = len(ecg_1d)

    hr_nk2 = 60.0 / np.mean(np.diff(peak_times_s)) if peak_times_s.size > 2 else np.nan
    logger.info("  sub-%03d %s: NK2 HR=%.1f bpm (%d peaks)", sub, task, hr_nk2, len(peaks_idx))

    new_rows = []
    for _, row in sub_df.iterrows():
        new_row = compute_ecg_row(row.to_dict(), peak_times_s, sfreq, n_samples)
        new_rows.append(new_row)

    patch_df = pd.DataFrame(new_rows, index=sub_df.index)
    return patch_df[ECG_COLS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--features", type=Path, default=FEATURES_PARQUET)
    parser.add_argument("--file-index", type=Path, default=FILE_INDEX_CSV)
    parser.add_argument("--received-dir", type=Path, default=RECEIVED_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    logger.info("Loading features.parquet …")
    feat_df = pd.read_parquet(args.features)

    # Only patch EEG-modality rows (which carry ECG features)
    feat_eeg_mask = feat_df["modality"].str.lower().isin(["eeg", "ecg"])
    logger.info("Rows to patch: %d / %d", feat_eeg_mask.sum(), len(feat_df))

    file_index = pd.read_csv(args.file_index)
    ecg_files = file_index[file_index["modality"].str.lower() == "ecg"].copy()
    ecg_files["task_norm"] = ecg_files["task"].str.lower().replace(
        {"memory": "digit_span", "digit_span": "digit_span", "rest": "rest"}
    )

    # Build mapping: (subject_int, task_in_parquet) → set_path
    task_map: Dict[Tuple[int, str], Path] = {}
    for _, row in ecg_files.iterrows():
        sub_int = int(row["subject"])
        # parquet uses 'memory' as task (as normalized by pipeline)
        raw_task = str(row["task"]).lower()
        parquet_task = "memory" if raw_task in ("memory", "digit_span") else raw_task
        set_path = args.received_dir / row["path"]
        if set_path.exists():
            task_map[(sub_int, parquet_task)] = set_path
        else:
            logger.debug("ECG file missing: %s", set_path)

    logger.info("Found %d valid ECG file paths", len(task_map))

    # Prepare patch
    unique_combos = feat_df[feat_eeg_mask][["subject", "task"]].drop_duplicates()
    logger.info("Unique (subject, task) combos to patch: %d", len(unique_combos))

    if args.dry_run:
        logger.info("DRY RUN — exiting without writing")
        return

    # Backup original
    backup_path = args.features.parent / "features_pre_ecg_patch.parquet"
    if not backup_path.exists():
        logger.info("Backing up to %s", backup_path)
        feat_df.to_parquet(backup_path, index=False)

    # Patch iteratively (in-memory)
    patched_df = feat_df.copy()
    n_patched = 0

    for _, combo_row in unique_combos.iterrows():
        sub = int(combo_row["subject"])
        task = str(combo_row["task"])
        set_path = task_map.get((sub, task))
        if set_path is None:
            logger.warning("No ECG file for sub-%03d / %s — skipping", sub, task)
            continue

        patch = patch_subject_task(sub, task, set_path, feat_df)
        if patch is None:
            logger.warning("Patch failed for sub-%03d / %s", sub, task)
            continue

        # Only overwrite the ECG columns that actually exist
        for col in ECG_COLS:
            if col in patch.columns and col in patched_df.columns:
                patched_df.loc[patch.index, col] = patch[col]
        n_patched += len(patch)

    logger.info("Patched %d rows; writing corrected features.parquet …", n_patched)
    patched_df.to_parquet(args.features, index=False)
    logger.info("Done — features.parquet updated in-place")

    # Quick validation
    orig_hr = feat_df["ecg_hr_bpm"].median()
    new_hr = patched_df["ecg_hr_bpm"].median()
    orig_rmssd = feat_df["ecg_hrv_rmssd_ms"].median()
    new_rmssd = patched_df["ecg_hrv_rmssd_ms"].median()
    logger.info("HR median: %.1f → %.1f bpm", orig_hr, new_hr)
    logger.info("RMSSD median: %.1f → %.1f ms", orig_rmssd, new_rmssd)


if __name__ == "__main__":
    main()
