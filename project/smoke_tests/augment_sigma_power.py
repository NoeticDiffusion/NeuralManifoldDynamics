"""
Augment event-locked parquets with per-window EEG bandpower (delta, theta,
alpha, sigma, beta).

For every row in the event-locked parquet — both spindle events AND matched
controls — this script extracts the raw 6 s EEG epoch at [window_start_sec,
window_end_sec], computes the Welch PSD, and integrates it over five bands.
The result is written as an augmented parquet next to the original, suffixed
`_spow.parquet`.

Output columns added:
  eeg_delta_psd   (0.5–4 Hz,  log10 µV²/Hz mean)
  eeg_theta_psd   (4–8 Hz)
  eeg_alpha_psd   (8–12 Hz)
  eeg_sigma_psd   (12–15 Hz)   <- primary new feature
  eeg_beta_psd    (15–30 Hz)
  eeg_spow_ok     (bool) — False if epoch was out-of-range or all-zero

Usage:
  python augment_sigma_power.py [--channel PSG_C3] [--subjects sub-1 sub-2 ...]
"""
import argparse
import json
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DS_RECV  = Path(r"M:\datasets\received\openneuro\ds005555")
DS_PROC  = Path(r"M:\datasets\processed\openneuro\ds005555")

BANDS = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 15.0),
    "beta":  (15.0, 30.0),
}

NPERSEG_S = 1.0   # Welch segment length in seconds → 256 samples at 256 Hz

ap = argparse.ArgumentParser()
ap.add_argument("--channel",  default="PSG_C3",
                help="EEG channel (must match detection channel)")
ap.add_argument("--subjects", nargs="*", default=None,
                help="Restrict to specific subjects (default: all)")
args = ap.parse_args()

CHANNEL = args.channel
SLUG    = CHANNEL.lower().replace("-", "_")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bandpower_log10(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    """Mean log10 PSD within [lo, hi] Hz. Returns NaN if band has no bins."""
    mask = (freqs >= lo) & (freqs < hi)
    if mask.sum() == 0:
        return float("nan")
    return float(np.log10(np.maximum(psd[mask].mean(), 1e-30)))


def compute_epoch_bandpower(data1d: np.ndarray, sfreq: float) -> dict:
    """Compute Welch PSD and integrate over BANDS for a 1-D epoch."""
    nperseg = max(int(NPERSEG_S * sfreq), 16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        freqs, psd = welch(data1d, fs=sfreq, nperseg=nperseg,
                           noverlap=nperseg // 2, scaling="density")
    return {f"eeg_{name}_psd": _bandpower_log10(freqs, psd, lo, hi)
            for name, (lo, hi) in BANDS.items()}


def process_subject(sub: str) -> bool:
    """Augment the event-locked parquet for one subject. Returns True on success."""
    # Find latest parquet for this subject
    all_pq = list(DS_PROC.rglob(
        f"{sub}_Sleep_acq-psg_event_locked_v1_{SLUG}.parquet"))
    if not all_pq:
        print(f"  [{sub}] No parquet found — skip")
        return False
    pq_path = sorted(all_pq, key=lambda p: p.stat().st_mtime)[-1]

    out_path = pq_path.with_name(pq_path.stem + "_spow.parquet")
    if out_path.exists():
        print(f"  [{sub}] Already augmented — skip")
        return True

    # Load EDF (lazy — do not preload entire file)
    edf = DS_RECV / sub / "eeg" / f"{sub}_task-Sleep_acq-psg_eeg.edf"
    if not edf.exists():
        print(f"  [{sub}] EDF not found — skip")
        return False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_edf(str(edf), preload=False, verbose=False)

    sfreq = raw.info["sfreq"]
    if CHANNEL not in raw.ch_names:
        print(f"  [{sub}] Channel {CHANNEL} missing in EDF — skip")
        return False

    ch_idx = raw.ch_names.index(CHANNEL)
    n_rec_samples = raw.n_times

    df = pd.read_parquet(str(pq_path))
    n_rows = len(df)
    print(f"  [{sub}] {n_rows} rows  sfreq={sfreq:.0f}  EDF={n_rec_samples/sfreq/3600:.2f} h")

    band_cols = {f"eeg_{name}_psd": np.full(n_rows, np.nan) for name in BANDS}
    ok_col    = np.zeros(n_rows, dtype=bool)

    for i, row in df.iterrows():
        t_start = float(row["window_start_sec"])
        t_end   = float(row["window_end_sec"])

        s0 = max(0, int(np.round(t_start * sfreq)))
        s1 = min(n_rec_samples, int(np.round(t_end   * sfreq)))

        if s1 - s0 < 32:          # too short to compute meaningful PSD
            continue

        epoch_raw, _ = raw[ch_idx, s0:s1]
        epoch = epoch_raw[0] * 1e6    # V → µV

        if not np.isfinite(epoch).all() or np.all(epoch == 0):
            continue

        bp = compute_epoch_bandpower(epoch, sfreq)
        for col, val in bp.items():
            band_cols[col][i] = val
        ok_col[i] = True

    for col, arr in band_cols.items():
        df[col] = arr
    df["eeg_spow_ok"] = ok_col

    n_ok = int(ok_col.sum())
    print(f"  [{sub}] eeg_spow_ok: {n_ok}/{n_rows} ({100*n_ok/n_rows:.1f}%)")
    df.to_parquet(str(out_path), index=False)
    print(f"  [{sub}] Saved: {out_path.name}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if args.subjects:
    subjects = args.subjects
else:
    # Discover from parquets
    all_pq = list(DS_PROC.rglob(f"*_Sleep_acq-psg_event_locked_v1_{SLUG}.parquet"))
    seen: dict[str, Path] = {}
    for p in all_pq:
        sub = p.name.split("_Sleep")[0]
        if sub not in seen or p.stat().st_mtime > seen[sub].stat().st_mtime:
            seen[sub] = p
    subjects = sorted(seen.keys(), key=lambda s: int(s.split("-")[1]))

print(f"Augmenting sigma power: {len(subjects)} subjects, channel={CHANNEL}")
print()

n_ok = n_fail = 0
for sub in subjects:
    ok = process_subject(sub)
    if ok:
        n_ok += 1
    else:
        n_fail += 1

print()
print(f"Done. OK={n_ok}  FAIL={n_fail}")
print(f"Output parquets: *_event_locked_v1_{SLUG}_spow.parquet")
