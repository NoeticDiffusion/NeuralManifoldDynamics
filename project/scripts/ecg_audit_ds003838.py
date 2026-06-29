"""ECG R-peak audit for ds003838.

Compares the current mndm detector (abs-bandpass + find_peaks) against
NeuroKit2's Pan-Tompkins-style detector on 10-15 representative subjects.

Outputs
-------
  audit_ecg_ds003838/
    ecg_audit_summary.csv          # per-subject HR comparison table
    ecg_audit_rr_detail.csv        # per-subject median/std of RR intervals
    figures/sub-XXX_task_seg.png   # 30-s ECG segment with both detectors
    figures/bland_altman.png       # Bland-Altman: mndm vs NK2 median HR

Usage
-----
  python project/scripts/ecg_audit_ds003838.py
  python project/scripts/ecg_audit_ds003838.py --subjects 35 36 91  # spot-check
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sp_signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RECEIVED_ROOT = Path("G:/Science_Datasets_longtime_storage/openneuro/received/ds003838")
FILE_INDEX_CSV = Path("J:/processed/openneuro/ds003838/file_index.csv")
FEATURES_PARQUET = Path("J:/processed/openneuro/ds003838/features.parquet")
OUT_DIR = Path("J:/processed/openneuro/ds003838/audit_ecg_ds003838")

# ── mndm detector defaults (same as ecg.py) ─────────────────────────────────
BANDPASS_LOW = 5.0
BANDPASS_HIGH = 20.0
BANDPASS_ORDER = 3
REFRACTORY_S = 0.3
PROMINENCE_MULT = 1.0
RR_MIN_S = 0.30
RR_MAX_S = 2.00


# ── helpers ──────────────────────────────────────────────────────────────────

def detect_mndm(ecg_1d: np.ndarray, sfreq: float) -> np.ndarray:
    """Current mndm detector: bandpass → |x| → find_peaks."""
    nyq = sfreq * 0.5
    hi = min(BANDPASS_HIGH, nyq * 0.99)
    lo = max(0.01, BANDPASS_LOW)
    try:
        b, a = sp_signal.butter(BANDPASS_ORDER, [lo / nyq, hi / nyq], btype="bandpass")
        filt = sp_signal.filtfilt(b, a, ecg_1d)
    except Exception:
        filt = ecg_1d - np.median(ecg_1d)
    centered = filt - np.median(filt)
    sig = np.abs(centered)                     # <── the suspected problem
    mad = float(np.median(np.abs(centered))) + 1e-8
    prom = max(1e-6, PROMINENCE_MULT * 1.4826 * mad)
    min_dist = max(1, int(round(REFRACTORY_S * sfreq)))
    peaks, _ = sp_signal.find_peaks(sig, distance=min_dist, prominence=prom)
    return peaks.astype(int)


def detect_neurokit2(ecg_1d: np.ndarray, sfreq: float) -> np.ndarray:
    """NeuroKit2 ECG peaks (Pan-Tompkins-style)."""
    import neurokit2 as nk
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        signals, info = nk.ecg_process(ecg_1d, sampling_rate=int(sfreq))
    r_peaks = np.where(signals["ECG_R_Peaks"] == 1)[0]
    return r_peaks.astype(int)


def rr_stats(peaks: np.ndarray, sfreq: float) -> Dict[str, float]:
    """Compute HR and HRV stats from peak indices."""
    if peaks.size < 3:
        return dict(hr_bpm=np.nan, rmssd_ms=np.nan, sdnn_ms=np.nan, n_peaks=int(peaks.size))
    rr_s = np.diff(peaks.astype(float) / sfreq)
    rr_s = rr_s[(rr_s >= RR_MIN_S) & (rr_s <= RR_MAX_S)]
    if rr_s.size < 2:
        return dict(hr_bpm=np.nan, rmssd_ms=np.nan, sdnn_ms=np.nan, n_peaks=int(peaks.size))
    hr = float(60.0 / np.mean(rr_s))
    drr = np.diff(rr_s)
    rmssd = float(np.sqrt(np.mean(drr ** 2)) * 1000.0) if drr.size > 0 else np.nan
    sdnn = float(np.std(rr_s, ddof=1) * 1000.0)
    return dict(hr_bpm=hr, rmssd_ms=rmssd, sdnn_ms=sdnn, n_peaks=int(peaks.size))


def load_ecg(set_path: Path) -> Optional[Tuple[np.ndarray, float]]:
    try:
        import mne
        mne.set_log_level("WARNING")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
        sfreq = float(raw.info["sfreq"])
        ecg_chs = mne.pick_types(raw.info, ecg=True, exclude=[])
        if len(ecg_chs) == 0:
            ecg_chs = [raw.ch_names.index(n) for n in raw.ch_names
                       if any(k in n.upper() for k in ("ECG", "EKG", "CARDIAC"))
                       if n in raw.ch_names]
        if not ecg_chs:
            return None
        data, _ = raw[ecg_chs[0], :]
        return np.asarray(data[0], dtype=float), sfreq
    except Exception as e:
        logger.warning("Cannot load %s: %s", set_path, e)
        return None


def plot_segment(
    ecg_1d: np.ndarray, sfreq: float,
    peaks_mndm: np.ndarray, peaks_nk2: np.ndarray,
    subject: str, task: str, out_path: Path,
    seg_start_s: float = 10.0, seg_dur_s: float = 30.0,
) -> None:
    """Plot a 30-second ECG segment with both detector outputs."""
    t = np.arange(len(ecg_1d)) / sfreq
    i0 = int(seg_start_s * sfreq)
    i1 = min(int((seg_start_s + seg_dur_s) * sfreq), len(ecg_1d))
    t_seg = t[i0:i1]
    ecg_seg = ecg_1d[i0:i1]

    fig, axes = plt.subplots(2, 1, figsize=(18, 6), sharex=True)
    for ax, sig, label, color in zip(
        axes,
        [ecg_seg, ecg_seg],
        ["mndm (abs-bandpass)", "NeuroKit2 (Pan-Tompkins)"],
        ["royalblue", "darkorange"],
    ):
        ax.plot(t_seg, sig, lw=0.6, color="k", alpha=0.7)
        ax.set_ylabel("ECG (a.u.)", fontsize=9)

    # mndm peaks
    pm = peaks_mndm[(peaks_mndm >= i0) & (peaks_mndm < i1)]
    axes[0].plot(t[pm], ecg_1d[pm], "v", color="royalblue", ms=7,
                 label=f"mndm  n={len(pm)}")
    axes[0].set_title(f"{subject} / {task}  —  mndm detector  "
                      f"(HR≈{60.0/np.diff(peaks_mndm[[0,-1]]).mean()*len(peaks_mndm):.0f} bpm "
                      f"approx)", fontsize=10)
    axes[0].legend(fontsize=8)

    # NK2 peaks
    pn = peaks_nk2[(peaks_nk2 >= i0) & (peaks_nk2 < i1)]
    axes[1].plot(t[pn], ecg_1d[pn], "^", color="darkorange", ms=7,
                 label=f"NK2   n={len(pn)}")
    axes[1].set_title("NeuroKit2 detector", fontsize=10)
    axes[1].set_xlabel("Time (s)", fontsize=9)
    axes[1].legend(fontsize=8)

    # Mark inter-detector discrepancies: mndm peaks with no close NK2 partner
    for p in pm:
        nearby = np.abs(pn - p) < int(0.15 * sfreq)
        if not nearby.any():
            axes[0].axvline(t[p], color="red", alpha=0.4, lw=1.2)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)


def bland_altman(hr_mndm: np.ndarray, hr_nk2: np.ndarray, out_path: Path) -> None:
    """Bland-Altman plot: mndm vs NK2 median HR per subject."""
    diff = hr_mndm - hr_nk2
    mean = 0.5 * (hr_mndm + hr_nk2)
    bias = float(np.nanmean(diff))
    loa = 1.96 * float(np.nanstd(diff, ddof=1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(mean, diff, s=40, alpha=0.7)
    ax.axhline(bias, color="red", lw=1.5, label=f"Bias = {bias:+.1f} bpm")
    ax.axhline(bias + loa, color="red", ls="--", lw=1, label=f"+1.96 SD = {bias+loa:.1f}")
    ax.axhline(bias - loa, color="red", ls="--", lw=1, label=f"-1.96 SD = {bias-loa:.1f}")
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("Mean HR (mndm + NK2) / 2  [bpm]", fontsize=10)
    ax.set_ylabel("Difference HR (mndm − NK2)  [bpm]", fontsize=10)
    ax.set_title("Bland-Altman: mndm vs NeuroKit2 median HR", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=130)
    plt.close(fig)
    logger.info("Bland-Altman: bias=%.1f bpm, LoA=[%.1f, %.1f]", bias, bias-loa, bias+loa)


# ── main ─────────────────────────────────────────────────────────────────────

def pick_audit_subjects(n: int = 15) -> List[int]:
    """Pick subjects spanning the HR distribution: low, mid, high + extremes."""
    df = pd.read_parquet(FEATURES_PARQUET)
    sub_hr = df.groupby("subject")["ecg_hr_bpm"].median().sort_values()
    subs = sub_hr.index.tolist()
    step = max(1, len(subs) // (n - 2))
    picked = subs[::step][:n - 2]
    # Always include the lowest and highest for diagnostics
    picked = list(dict.fromkeys([subs[0]] + list(picked) + [subs[-1]]))
    return [int(s) for s in picked[:n]]


def main() -> None:
    parser = argparse.ArgumentParser(description="ECG R-peak audit for ds003838")
    parser.add_argument("--subjects", nargs="*", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--received-dir", type=Path, default=RECEIVED_ROOT)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    file_df = pd.read_csv(FILE_INDEX_CSV)
    ecg_df = file_df[file_df["modality"].str.lower() == "ecg"].copy()

    subjects = args.subjects or pick_audit_subjects(15)
    logger.info("Auditing %d subjects: %s", len(subjects), subjects)

    rows: List[Dict] = []
    hr_mndm_list, hr_nk2_list = [], []

    for sub_int in subjects:
        sub_str = f"sub-{sub_int:03d}"
        sub_rows = ecg_df[ecg_df["subject"] == sub_int]
        if sub_rows.empty:
            logger.warning("No ECG in file_index for subject %d", sub_int)
            continue

        for _, frow in sub_rows.iterrows():
            task = str(frow.get("task", "unknown"))
            set_path = args.received_dir / frow["path"]
            if not set_path.exists():
                logger.warning("Missing: %s", set_path)
                continue

            result = load_ecg(set_path)
            if result is None:
                continue
            ecg_1d, sfreq = result
            logger.info("%s / %s  %.0f s  sfreq=%.0f", sub_str, task,
                        len(ecg_1d)/sfreq, sfreq)

            peaks_mndm = detect_mndm(ecg_1d, sfreq)
            try:
                peaks_nk2 = detect_neurokit2(ecg_1d, sfreq)
            except Exception as e:
                logger.warning("NK2 failed for %s/%s: %s", sub_str, task, e)
                peaks_nk2 = np.array([], dtype=int)

            stats_m = rr_stats(peaks_mndm, sfreq)
            stats_n = rr_stats(peaks_nk2, sfreq) if peaks_nk2.size > 0 else {
                k: np.nan for k in ("hr_bpm", "rmssd_ms", "sdnn_ms", "n_peaks")
            }

            row = {
                "subject": sub_str, "task": task,
                "mndm_hr_bpm": stats_m["hr_bpm"],
                "nk2_hr_bpm": stats_n["hr_bpm"],
                "hr_ratio": (stats_m["hr_bpm"] / stats_n["hr_bpm"]
                             if np.isfinite(stats_n["hr_bpm"]) and stats_n["hr_bpm"] > 0
                             else np.nan),
                "mndm_rmssd_ms": stats_m["rmssd_ms"],
                "nk2_rmssd_ms": stats_n["rmssd_ms"],
                "mndm_sdnn_ms": stats_m["sdnn_ms"],
                "nk2_sdnn_ms": stats_n["sdnn_ms"],
                "mndm_n_peaks": stats_m["n_peaks"],
                "nk2_n_peaks": stats_n.get("n_peaks", np.nan),
                "duration_s": float(len(ecg_1d) / sfreq),
            }
            rows.append(row)
            logger.info(
                "  mndm HR=%.1f  NK2 HR=%.1f  ratio=%.2f  "
                "mndm RMSSD=%.1f ms  NK2 RMSSD=%.1f ms",
                stats_m["hr_bpm"], stats_n["hr_bpm"],
                row["hr_ratio"] if np.isfinite(row["hr_ratio"]) else -1,
                stats_m["rmssd_ms"] if np.isfinite(stats_m["rmssd_ms"]) else -1,
                stats_n["rmssd_ms"] if np.isfinite(stats_n["rmssd_ms"]) else -1,
            )

            if np.isfinite(stats_m["hr_bpm"]):
                hr_mndm_list.append(stats_m["hr_bpm"])
            if np.isfinite(stats_n["hr_bpm"]):
                hr_nk2_list.append(stats_n["hr_bpm"])

            # Plot segment (first recording per subject only)
            if task in ("rest", "memory"):
                plot_segment(
                    ecg_1d, sfreq, peaks_mndm, peaks_nk2,
                    sub_str, task,
                    fig_dir / f"{sub_str}_{task}_seg.png",
                )

    summary_df = pd.DataFrame(rows)
    summary_csv = out_dir / "ecg_audit_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info("Wrote %s", summary_csv)

    if rows:
        print("\n=== ECG audit summary ===")
        print(summary_df[["subject", "task", "mndm_hr_bpm", "nk2_hr_bpm",
                           "hr_ratio", "mndm_rmssd_ms", "nk2_rmssd_ms"]].to_string(index=False))
        print()
        print("Group mndm HR median: %.1f bpm" % summary_df["mndm_hr_bpm"].median())
        print("Group NK2  HR median: %.1f bpm" % summary_df["nk2_hr_bpm"].median())
        print("Group HR ratio (mndm/NK2) median: %.2f" %
              summary_df["hr_ratio"].median())
        print()
        print("Group mndm RMSSD median: %.1f ms" % summary_df["mndm_rmssd_ms"].median())
        print("Group NK2  RMSSD median: %.1f ms" % summary_df["nk2_rmssd_ms"].median())

    if hr_mndm_list and hr_nk2_list and len(hr_mndm_list) == len(hr_nk2_list):
        bland_altman(
            np.array(hr_mndm_list), np.array(hr_nk2_list),
            fig_dir / "bland_altman.png",
        )


if __name__ == "__main__":
    main()
