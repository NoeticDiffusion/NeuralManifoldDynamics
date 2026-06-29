"""04b — Encoding-phase MNPS analysis for ds006848.

Three MNPS analyses (Tests A, B, C) plus retrieval-by-prior-mode.

Analyses
--------
A. Full encoding episode
   Reconstruct each trial's encoding interval from events.tsv.
   Summarise m, d, e, m_dot, d_dot, e_dot per subject × condition.
   Friedman across Simultaneous / Fast / FastDelay / Slow.
   Pairwise Wilcoxon with BH-FDR.

B. Common-duration encoding window
   First 2.8 s of encoding for every condition (= full Fast / Simultaneous
   duration). Allows fair cross-mode comparison.

C. Normalised encoding phase
   Divide each encoding episode into 4 normalised bins (0-25 %, 25-50 %,
   50-75 %, 75-100 %). Test condition × phase_bin effects on m, d, e.
   Main test for FastDelay's hypothesised encode–rehearse micro-cycle.

Retrieval-by-prior-mode
   Digits_Retrieval windows stratified by the preceding condition.

Interpretation constraints (from science lead)
----------------------------------------------
* Previous Phase-D null is a maintenance-window null, not a
  presentation-mode null.
* FastDelay pre→early result is a transition diagnostic, not yet a
  balanced condition effect.
* Do not claim item-level MNPS resolution from 8 s windows.
* Do not make WM-phase HRV / anchor claims from 60 s superwindows.
* sub-003 excluded from ECG / anchor analyses but kept for EEG/MNPS.

Usage
-----
  python project/scripts/04b_encoding_phase_analysis.py \\
      --run-dir J:/processed/openneuro/ds006848/neuralmanifolddynamics_ds006848_20260626_114620 \\
      --bids-dir K:/ExternalReceivedDatasets/openneuro/received/ds006848 \\
      --out-dir J:/processed/openneuro/ds006848/04b_encoding_phase
"""
from __future__ import annotations

import argparse
import logging
import warnings
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*copy.*keyword.*", category=DeprecationWarning)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONDITIONS = ["Simultaneous", "Fast", "FastDelay", "Slow"]
CONDITION_MAP = {
    "Retention_Simultaneous": "Simultaneous",
    "Retention_Fast": "Fast",
    "Retention_FastDelay": "FastDelay",
    "Retention_Slow": "Slow",
}
METRICS = ["m", "d", "e", "m_dot", "d_dot", "e_dot"]
COMMON_DURATION_S = 2.8   # Fast / Simultaneous full encoding duration
N_PHASE_BINS = 4
MIN_OVERLAP_FRAC = 0.25   # window must overlap the interval by ≥25 %

# ---------------------------------------------------------------------------
# Trial-structure reconstruction
# ---------------------------------------------------------------------------


def reconstruct_trials(events_df: pd.DataFrame) -> pd.DataFrame:
    """Extract per-trial encoding / retention / retrieval intervals.

    Returns a DataFrame with columns::

        trial_id, condition,
        enc_start, enc_end,          # encoding phase
        ret_start, ret_end,          # retention phase
        retr_start, retr_end         # retrieval phase
    """
    events_df = events_df.sort_values("onset").reset_index(drop=True)
    trials: List[Dict] = []

    retention_types = set(CONDITION_MAP.keys())
    for idx, row in events_df.iterrows():
        if row["trial_type"] not in retention_types:
            continue

        condition = CONDITION_MAP[row["trial_type"]]
        ret_start = row["onset"]
        ret_end = ret_start + 6.0  # retention is always 6 s

        # Find encoding start: scan backward for first Encoding_* group
        enc_events = []
        for jdx in range(idx - 1, max(-1, idx - 30), -1):
            ttype = events_df.at[jdx, "trial_type"]
            if ttype.startswith("Encoding_"):
                enc_events.append(events_df.at[jdx, "onset"])
            elif enc_events:
                # Hit a non-encoding row after collecting encoding events
                break

        if not enc_events:
            continue

        enc_start = min(enc_events)
        enc_end = ret_start   # encoding ends when Retention_* begins

        # Find retrieval start: next Digits_Retrieval after ret_start
        later = events_df[
            (events_df["onset"] > ret_end) & (events_df["trial_type"] == "Digits_Retrieval")
        ]
        if later.empty:
            continue
        retr_start = later.iloc[0]["onset"]
        retr_end = retr_start + 13.0  # typical retrieval window

        trials.append(
            {
                "trial_id": len(trials),
                "condition": condition,
                "enc_start": enc_start,
                "enc_end": enc_end,
                "enc_dur": enc_end - enc_start,
                "ret_start": ret_start,
                "ret_end": ret_end,
                "retr_start": retr_start,
                "retr_end": retr_end,
            }
        )

    return pd.DataFrame(trials)


# ---------------------------------------------------------------------------
# H5 loader
# ---------------------------------------------------------------------------


def load_subject_mnps(h5_path: Path) -> pd.DataFrame:
    """Return per-window MNPS features from an H5 file.

    Returns columns: t_start, t_end, m, d, e, m_dot, d_dot, e_dot
    """
    with h5py.File(h5_path, "r") as f:
        t_start = np.array(f["window_start"], dtype=np.float64)
        t_end = np.array(f["window_end"], dtype=np.float64)
        mnps = np.array(f["mnps_3d"], dtype=np.float64)       # (N, 3)
        mnps_dot = np.array(f["mnps_3d_dot"], dtype=np.float64)  # (N, 3)

    df = pd.DataFrame(
        {
            "t_start": t_start,
            "t_end": t_end,
            "m": mnps[:, 0],
            "d": mnps[:, 1],
            "e": mnps[:, 2],
            "m_dot": mnps_dot[:, 0],
            "d_dot": mnps_dot[:, 1],
            "e_dot": mnps_dot[:, 2],
        }
    )
    return df[np.isfinite(df["m"])]


# ---------------------------------------------------------------------------
# Window–interval matching
# ---------------------------------------------------------------------------


def overlap_fraction(w_start: np.ndarray, w_end: np.ndarray,
                     iv_start: float, iv_end: float) -> np.ndarray:
    """Fraction of the window duration that overlaps [iv_start, iv_end]."""
    win_dur = w_end - w_start
    ovlp = np.maximum(0.0, np.minimum(w_end, iv_end) - np.maximum(w_start, iv_start))
    return np.where(win_dur > 0, ovlp / win_dur, 0.0)


def windows_in_interval(
    windows: pd.DataFrame,
    iv_start: float,
    iv_end: float,
    min_overlap: float = MIN_OVERLAP_FRAC,
) -> pd.DataFrame:
    """Return rows of *windows* whose overlap fraction ≥ min_overlap."""
    frac = overlap_fraction(
        windows["t_start"].values, windows["t_end"].values, iv_start, iv_end
    )
    return windows[frac >= min_overlap]


# ---------------------------------------------------------------------------
# Per-trial aggregation helpers
# ---------------------------------------------------------------------------


def trial_medians(
    windows: pd.DataFrame,
    trials: pd.DataFrame,
    iv_start_col: str,
    iv_end_col: str,
    min_overlap: float = MIN_OVERLAP_FRAC,
) -> pd.DataFrame:
    """Compute per-trial medians of MNPS metrics.

    Returns columns: trial_id, condition + METRICS.
    Trials with zero qualifying windows are omitted.
    """
    rows: List[Dict] = []
    for _, trial in trials.iterrows():
        iv_start = trial[iv_start_col]
        iv_end = trial[iv_end_col]
        sel = windows_in_interval(windows, iv_start, iv_end, min_overlap)
        if sel.empty:
            continue
        row: Dict = {"trial_id": trial["trial_id"], "condition": trial["condition"]}
        for m in METRICS:
            row[m] = sel[m].median()
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["trial_id", "condition"] + METRICS)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR correction.  Returns q-values."""
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    rank = np.empty_like(order)
    rank[order] = np.arange(1, n + 1)
    q = pvals * n / rank
    # Enforce monotonicity from right
    q_min = np.minimum.accumulate(q[order][::-1])[::-1]
    q[order] = np.minimum(q[order], q_min)
    return np.minimum(q, 1.0)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    return (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else np.nan


def friedman_posthoc(
    subject_condition_medians: pd.DataFrame,
    metric: str,
    conditions: List[str] = CONDITIONS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Friedman test + pairwise Wilcoxon with BH-FDR.

    Parameters
    ----------
    subject_condition_medians:
        Must contain columns 'subject', 'condition', metric.
    metric:
        Column to test.

    Returns
    -------
    friedman_row : DataFrame (1 row)
        chi2, dof, p, n_subjects
    pairwise : DataFrame
        condition_a, condition_b, median_a, median_b, W, p, q, cohens_d, sig
    """
    pivot = (
        subject_condition_medians[subject_condition_medians["condition"].isin(conditions)]
        .pivot_table(index="subject", columns="condition", values=metric, aggfunc="median")
    )
    pivot = pivot.dropna()
    n_subjects = len(pivot)

    # Friedman
    cond_arrays = [pivot[c].values for c in conditions if c in pivot.columns]
    if len(cond_arrays) < 2 or n_subjects < 3:
        empty_fr = pd.DataFrame(
            [{"metric": metric, "chi2": np.nan, "dof": np.nan, "p_friedman": np.nan, "n_subjects": n_subjects}]
        )
        return empty_fr, pd.DataFrame()

    chi2, p_friedman = stats.friedmanchisquare(*cond_arrays)

    friedman_row = pd.DataFrame(
        [{"metric": metric, "chi2": chi2, "dof": len(cond_arrays) - 1,
          "p_friedman": p_friedman, "n_subjects": n_subjects}]
    )

    # Pairwise Wilcoxon
    pairs: List[Dict] = []
    available = [c for c in conditions if c in pivot.columns]
    for ca, cb in combinations(available, 2):
        a = pivot[ca].values
        b = pivot[cb].values
        paired = np.column_stack([a, b])
        paired = paired[np.isfinite(paired).all(axis=1)]
        if len(paired) < 3:
            continue
        try:
            stat, pval = wilcoxon(paired[:, 0], paired[:, 1])
        except ValueError:
            pval = 1.0
            stat = np.nan
        pairs.append(
            {
                "condition_a": ca,
                "condition_b": cb,
                "median_a": np.median(paired[:, 0]),
                "median_b": np.median(paired[:, 1]),
                "W": stat,
                "p": pval,
                "cohens_d": cohens_d(paired[:, 0], paired[:, 1]),
            }
        )

    if not pairs:
        return friedman_row, pd.DataFrame()

    pairwise = pd.DataFrame(pairs)
    pairwise["q"] = bh_fdr(pairwise["p"].values)
    pairwise["sig"] = pairwise["q"] < 0.05
    pairwise.insert(0, "metric", metric)
    return friedman_row, pairwise


# ---------------------------------------------------------------------------
# Per-subject × condition summary
# ---------------------------------------------------------------------------


def build_subject_condition_summary(
    per_trial: pd.DataFrame, subject: str
) -> pd.DataFrame:
    """Compute per-subject × condition medians from per-trial rows."""
    if per_trial.empty:
        return pd.DataFrame()
    rows: List[Dict] = []
    for cond, grp in per_trial.groupby("condition"):
        row: Dict = {"subject": subject, "condition": cond, "n_trials": len(grp)}
        for m in METRICS:
            if m in grp.columns:
                row[m] = grp[m].median()
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Event-locked parquet loader for retrieval analysis
# ---------------------------------------------------------------------------


def load_event_locked_parquets(run_dir: Path, task_tag: str = "verbalwm") -> pd.DataFrame:
    """Concatenate all event-locked parquets for *task_tag* from *run_dir*."""
    frames: List[pd.DataFrame] = []
    for sub_dir in sorted(run_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        for pq in sub_dir.glob(f"*{task_tag}*event_locked*.parquet"):
            try:
                df = pd.read_parquet(pq)
                frames.append(df)
            except Exception as exc:
                logger.warning("Could not read %s: %s", pq, exc)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Main analysis runners
# ---------------------------------------------------------------------------


def run_test(
    label: str,
    iv_start_col: str,
    iv_end_col: str,
    run_dir: Path,
    bids_dir: Path,
    out_dir: Path,
    subjects: Optional[List[str]] = None,
) -> None:
    """Generic driver for Tests A and B.

    For Test A: iv_start_col='enc_start', iv_end_col='enc_end'
    For Test B: iv_start_col='enc_start', iv_end_col='enc28_end' (added externally)
    """
    logger.info("=== Test %s ===", label)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summary: List[pd.DataFrame] = []

    for sub_dir in sorted(run_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        h5_files = list(sub_dir.glob("*verbal*.h5"))
        if not h5_files:
            continue
        subject = sub_dir.name.replace("_verbal_wm", "").replace("_verbalwm", "")
        if subjects is not None and subject not in subjects:
            continue

        events_file = (
            bids_dir / subject / "eeg" / f"{subject}_task-verbalwm_events.tsv"
        )
        if not events_file.exists():
            logger.warning("No events.tsv for %s", subject)
            continue

        try:
            events_df = pd.read_csv(events_file, sep="\t")
            trials = reconstruct_trials(events_df)
        except Exception as exc:
            logger.error("Trial reconstruction failed for %s: %s", subject, exc)
            continue

        if trials.empty:
            logger.warning("No trials reconstructed for %s", subject)
            continue

        # Clamp encoding end for Test B
        if iv_end_col == "enc28_end":
            trials["enc28_end"] = trials["enc_start"] + COMMON_DURATION_S

        try:
            windows = load_subject_mnps(h5_files[0])
        except Exception as exc:
            logger.error("H5 load failed for %s: %s", subject, exc)
            continue

        per_trial = trial_medians(windows, trials, iv_start_col, iv_end_col)
        if per_trial.empty:
            logger.warning("%s: no trial-window matches for test %s", subject, label)
            continue

        summary = build_subject_condition_summary(per_trial, subject)
        all_summary.append(summary)
        logger.info(
            "  %s: %d trials, %d conditions",
            subject, len(per_trial), per_trial["condition"].nunique(),
        )

    if not all_summary:
        logger.warning("Test %s: no data collected", label)
        return

    summary_df = pd.concat(all_summary, ignore_index=True)
    summary_df.to_csv(out_dir / "subject_condition_medians.csv", index=False)
    logger.info("Wrote %s", out_dir / "subject_condition_medians.csv")

    # Friedman + pairwise tests
    friedman_rows: List[pd.DataFrame] = []
    pairwise_rows: List[pd.DataFrame] = []
    for metric in METRICS:
        fr, pw = friedman_posthoc(summary_df, metric)
        friedman_rows.append(fr)
        if not pw.empty:
            pairwise_rows.append(pw)

    pd.concat(friedman_rows, ignore_index=True).to_csv(
        out_dir / "friedman_results.csv", index=False
    )
    if pairwise_rows:
        pd.concat(pairwise_rows, ignore_index=True).to_csv(
            out_dir / "pairwise_contrasts.csv", index=False
        )

    _write_summary_txt(label, summary_df, friedman_rows, pairwise_rows, out_dir)
    logger.info("Test %s complete → %s", label, out_dir)


def run_test_c_normalized_bins(
    run_dir: Path,
    bids_dir: Path,
    out_dir: Path,
    subjects: Optional[List[str]] = None,
) -> None:
    """Test C — normalised encoding phase bins.

    Resolution note: with 8 s windows at 4 s step, Fast / Simultaneous
    (2.8 s encoding) typically yield ≤ 1 window; FastDelay / Slow (7 s)
    yield 1–2 windows. The four-bin structure is sparsely populated for
    short conditions; treat results as exploratory.
    """
    logger.info("=== Test C: normalised encoding phase ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    bin_labels = [f"bin{i+1}" for i in range(N_PHASE_BINS)]  # bin1..bin4

    all_rows: List[Dict] = []

    for sub_dir in sorted(run_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        h5_files = list(sub_dir.glob("*verbal*.h5"))
        if not h5_files:
            continue
        subject = sub_dir.name.replace("_verbal_wm", "").replace("_verbalwm", "")
        if subjects is not None and subject not in subjects:
            continue

        events_file = (
            bids_dir / subject / "eeg" / f"{subject}_task-verbalwm_events.tsv"
        )
        if not events_file.exists():
            continue

        try:
            events_df = pd.read_csv(events_file, sep="\t")
            trials = reconstruct_trials(events_df)
            windows = load_subject_mnps(h5_files[0])
        except Exception as exc:
            logger.error("Data load failed for %s: %s", subject, exc)
            continue

        # Pre-compute window centres once per subject
        win_centers = (windows["t_start"].values + windows["t_end"].values) / 2.0

        for _, trial in trials.iterrows():
            enc_start = trial["enc_start"]
            enc_dur = trial["enc_dur"]
            if enc_dur <= 0:
                continue

            for bin_idx in range(N_PHASE_BINS):
                bin_lo = enc_start + (bin_idx / N_PHASE_BINS) * enc_dur
                bin_hi = enc_start + ((bin_idx + 1) / N_PHASE_BINS) * enc_dur

                # Assign a window to a bin when its centre falls inside the bin
                # OR when it has any non-zero overlap with the encoding episode.
                # For short conditions (Fast/Simultaneous, ~2.8 s encoding) the
                # bins are narrower than one window (8 s), so we use centre-based
                # assignment with a fallback to any overlap with the full encoding.
                in_bin = (win_centers >= bin_lo) & (win_centers < bin_hi)

                # Fallback: if no centre lands in this bin, accept windows that
                # overlap the full encoding interval and whose normalised centre
                # (relative to enc_start) maps to this bin.
                if not in_bin.any():
                    norm_ctr = (win_centers - enc_start) / enc_dur
                    in_bin = (norm_ctr >= bin_idx / N_PHASE_BINS) & (norm_ctr < (bin_idx + 1) / N_PHASE_BINS)
                    # Also require some overlap with the encoding interval
                    has_overlap = overlap_fraction(
                        windows["t_start"].values, windows["t_end"].values,
                        enc_start, trial["enc_end"]
                    ) > 0
                    in_bin = in_bin & has_overlap

                sel = windows[in_bin]
                if sel.empty:
                    continue
                row: Dict = {
                    "subject": subject,
                    "condition": trial["condition"],
                    "trial_id": trial["trial_id"],
                    "phase_bin": bin_labels[bin_idx],
                    "bin_idx": bin_idx,
                    "n_windows": len(sel),
                }
                for m in METRICS:
                    row[m] = sel[m].median()
                all_rows.append(row)

    if not all_rows:
        logger.warning("Test C: no data collected")
        return

    per_bin_df = pd.DataFrame(all_rows)
    per_bin_df.to_csv(out_dir / "per_trial_bin_medians.csv", index=False)

    # Per-subject × condition × bin medians
    sub_cond_bin = (
        per_bin_df.groupby(["subject", "condition", "phase_bin"])[METRICS]
        .median()
        .reset_index()
    )
    sub_cond_bin.to_csv(out_dir / "subject_condition_bin_medians.csv", index=False)

    # Coverage: fraction of trials with ≥1 window per bin
    coverage = (
        per_bin_df.groupby(["condition", "phase_bin"])["trial_id"]
        .nunique()
        .reset_index(name="n_trials_covered")
    )
    coverage.to_csv(out_dir / "bin_coverage.csv", index=False)
    logger.info("Bin coverage:\n%s", coverage.to_string(index=False))

    # Simple Friedman: per condition, test phase_bin effect on m / d / e
    friedman_rows: List[pd.DataFrame] = []
    for condition in CONDITIONS:
        sub_cond = sub_cond_bin[sub_cond_bin["condition"] == condition]
        for metric in ["m", "d", "e"]:
            pivot = sub_cond.pivot_table(
                index="subject", columns="phase_bin", values=metric, aggfunc="median"
            ).dropna()
            if pivot.shape[0] < 3 or pivot.shape[1] < 2:
                continue
            try:
                chi2, p = stats.friedmanchisquare(*[pivot[b].values for b in pivot.columns])
                friedman_rows.append(
                    pd.DataFrame(
                        [{"condition": condition, "metric": metric,
                          "chi2": chi2, "p": p, "n_subjects": len(pivot)}]
                    )
                )
            except Exception:
                pass

    if friedman_rows:
        pd.concat(friedman_rows, ignore_index=True).to_csv(
            out_dir / "friedman_phase_bin.csv", index=False
        )

    logger.info("Test C complete → %s", out_dir)


def run_retrieval_by_prior_mode(
    run_dir: Path,
    bids_dir: Path,
    out_dir: Path,
    subjects: Optional[List[str]] = None,
) -> None:
    """Digits_Retrieval MNPS stratified by preceding presentation mode.

    Strategy: load per-subject H5 + events.tsv, tag each retrieval
    interval with the preceding Retention_* type, then compute per-subject
    × prior-mode medians.
    """
    logger.info("=== Retrieval-by-prior-mode ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summary: List[pd.DataFrame] = []

    for sub_dir in sorted(run_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        h5_files = list(sub_dir.glob("*verbal*.h5"))
        if not h5_files:
            continue
        subject = sub_dir.name.replace("_verbal_wm", "").replace("_verbalwm", "")
        if subjects is not None and subject not in subjects:
            continue

        events_file = (
            bids_dir / subject / "eeg" / f"{subject}_task-verbalwm_events.tsv"
        )
        if not events_file.exists():
            continue

        try:
            events_df = pd.read_csv(events_file, sep="\t").sort_values("onset")
            windows = load_subject_mnps(h5_files[0])
        except Exception as exc:
            logger.error("Load failed for %s: %s", subject, exc)
            continue

        # Tag each Digits_Retrieval event with prior mode
        retrieval_rows: List[Dict] = []
        retr_events = events_df[events_df["trial_type"] == "Digits_Retrieval"]
        for _, retr in retr_events.iterrows():
            # Find most-recent Retention_* before this retrieval
            preceding = events_df[
                (events_df["onset"] < retr["onset"])
                & (events_df["trial_type"].isin(CONDITION_MAP.keys()))
            ]
            if preceding.empty:
                continue
            prior_type = preceding.iloc[-1]["trial_type"]
            prior_mode = CONDITION_MAP.get(prior_type, prior_type)

            retr_start = retr["onset"]
            retr_end = retr_start + 13.0
            sel = windows_in_interval(windows, retr_start, retr_end)
            if sel.empty:
                continue
            row: Dict = {
                "trial_id": len(retrieval_rows),
                "prior_mode": prior_mode,
            }
            for m in METRICS:
                row[m] = sel[m].median()
            retrieval_rows.append(row)

        if not retrieval_rows:
            continue

        per_retr = pd.DataFrame(retrieval_rows)
        rows: List[Dict] = []
        for mode, grp in per_retr.groupby("prior_mode"):
            r: Dict = {"subject": subject, "prior_mode": mode, "n_trials": len(grp)}
            for m in METRICS:
                r[m] = grp[m].median()
            rows.append(r)
        all_summary.append(pd.DataFrame(rows))
        logger.info("  %s: %d retrieval intervals", subject, len(per_retr))

    if not all_summary:
        logger.warning("Retrieval-by-prior-mode: no data collected")
        return

    summary_df = pd.concat(all_summary, ignore_index=True)
    summary_df.to_csv(out_dir / "subject_priormode_medians.csv", index=False)

    # Friedman across prior modes
    conditions_prior = ["Simultaneous", "Fast", "FastDelay", "Slow"]
    friedman_rows: List[pd.DataFrame] = []
    pairwise_rows: List[pd.DataFrame] = []
    for metric in METRICS:
        pivot = (
            summary_df[summary_df["prior_mode"].isin(conditions_prior)]
            .pivot_table(index="subject", columns="prior_mode", values=metric, aggfunc="median")
            .dropna()
        )
        if pivot.shape[0] < 3:
            continue
        arrays = [pivot[c].values for c in conditions_prior if c in pivot.columns]
        if len(arrays) < 2:
            continue
        chi2, p = stats.friedmanchisquare(*arrays)
        friedman_rows.append(
            pd.DataFrame(
                [{"metric": metric, "chi2": chi2, "dof": len(arrays) - 1,
                  "p_friedman": p, "n_subjects": len(pivot)}]
            )
        )
        # Pairwise
        pairs: List[Dict] = []
        for ca, cb in combinations([c for c in conditions_prior if c in pivot.columns], 2):
            paired = np.column_stack([pivot[ca].values, pivot[cb].values])
            paired = paired[np.isfinite(paired).all(axis=1)]
            if len(paired) < 3:
                continue
            try:
                stat, pval = wilcoxon(paired[:, 0], paired[:, 1])
            except ValueError:
                pval = 1.0; stat = np.nan
            pairs.append(
                {"condition_a": ca, "condition_b": cb,
                 "median_a": np.median(paired[:, 0]), "median_b": np.median(paired[:, 1]),
                 "W": stat, "p": pval, "cohens_d": cohens_d(paired[:, 0], paired[:, 1])}
            )
        if pairs:
            pw = pd.DataFrame(pairs)
            pw["q"] = bh_fdr(pw["p"].values)
            pw["sig"] = pw["q"] < 0.05
            pw.insert(0, "metric", metric)
            pairwise_rows.append(pw)

    if friedman_rows:
        pd.concat(friedman_rows, ignore_index=True).to_csv(
            out_dir / "friedman_results.csv", index=False
        )
    if pairwise_rows:
        pd.concat(pairwise_rows, ignore_index=True).to_csv(
            out_dir / "pairwise_contrasts.csv", index=False
        )
    logger.info("Retrieval-by-prior-mode complete → %s", out_dir)


# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------


def _write_summary_txt(
    label: str,
    summary_df: pd.DataFrame,
    friedman_rows: List[pd.DataFrame],
    pairwise_rows: List[pd.DataFrame],
    out_dir: Path,
) -> None:
    lines: List[str] = [f"=== Test {label} summary ===\n"]
    n_subjects = summary_df["subject"].nunique() if not summary_df.empty else 0
    lines.append(f"Subjects: {n_subjects}\n")

    lines.append("\nCondition medians (across subjects)\n")
    if not summary_df.empty:
        cond_med = (
            summary_df.groupby("condition")[["m", "d", "e"]]
            .agg(["median", "count"])
        )
        lines.append(cond_med.to_string())

    lines.append("\n\nFriedman results\n")
    if friedman_rows:
        fr = pd.concat(friedman_rows, ignore_index=True)
        lines.append(fr.to_string(index=False))

    lines.append("\n\nSignificant pairwise contrasts (q < 0.05)\n")
    if pairwise_rows:
        pw = pd.concat(pairwise_rows, ignore_index=True)
        sig = pw[pw["sig"]]
        lines.append(sig.to_string(index=False) if not sig.empty else "(none)")

    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=Path("J:/processed/openneuro/ds006848/neuralmanifolddynamics_ds006848_20260626_114620"),
        help="Path to the neuralmanifolddynamics_* run directory",
    )
    ap.add_argument(
        "--bids-dir",
        type=Path,
        default=Path("K:/ExternalReceivedDatasets/openneuro/received/ds006848"),
        help="Root of the BIDS dataset",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("J:/processed/openneuro/ds006848/04b_encoding_phase"),
        help="Output directory",
    )
    ap.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Subset of subjects to process (default: all)",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    run_dir = args.run_dir
    bids_dir = args.bids_dir
    out_dir = args.out_dir
    subjects = args.subjects

    if not run_dir.exists():
        # Fall back to latest available run
        parent = run_dir.parent
        dirs = sorted(parent.glob("neuralmanifolddynamics_*"))
        if dirs:
            run_dir = dirs[-1]
            logger.warning("run-dir not found; using latest: %s", run_dir)
        else:
            raise FileNotFoundError(f"No run directories found under {parent}")

    logger.info("Run dir: %s", run_dir)
    logger.info("BIDS dir: %s", bids_dir)
    logger.info("Out dir: %s", out_dir)

    run_test(
        "A_full_encoding",
        iv_start_col="enc_start",
        iv_end_col="enc_end",
        run_dir=run_dir,
        bids_dir=bids_dir,
        out_dir=out_dir / "A_full_encoding",
        subjects=subjects,
    )

    run_test(
        "B_28s_window",
        iv_start_col="enc_start",
        iv_end_col="enc28_end",
        run_dir=run_dir,
        bids_dir=bids_dir,
        out_dir=out_dir / "B_28s_window",
        subjects=subjects,
    )

    run_test_c_normalized_bins(
        run_dir=run_dir,
        bids_dir=bids_dir,
        out_dir=out_dir / "C_normalized_bins",
        subjects=subjects,
    )

    run_retrieval_by_prior_mode(
        run_dir=run_dir,
        bids_dir=bids_dir,
        out_dir=out_dir / "D_retrieval_by_prior_mode",
        subjects=subjects,
    )

    logger.info("04b complete.")


if __name__ == "__main__":
    main()
