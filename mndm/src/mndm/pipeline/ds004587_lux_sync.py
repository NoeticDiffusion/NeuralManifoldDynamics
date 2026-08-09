"""LUX-photosensor trial-clock recovery for ds004587 (Illusion Game).

``ds004587`` ships EEG recordings whose ``events.tsv`` contains only a single
"recording start" marker -- no hardware trigger channel marks individual
trial onsets (``TriggerChannelCount: 0`` in every ``*_eeg.json``). However,
the injected physiology stream (``physio_tsv_inject`` in
``config_ingest_ds004587.yaml``) includes a ``LUX`` channel: a PLUX/BITalino
photosensor that captures real screen-luminance transitions and is
time-aligned with the EEG recording (``physio.json StartTime: -0.0``).

This module recovers a robust, per-run linear mapping from the behavioral
log's ``time_elapsed`` clock (jsPsych, milliseconds since the browser
experiment started) to the EEG/physio sample clock, using distinctive
"block-break" landmarks that are unambiguous on both sides:

* On the behavioral side: rows immediately preceding a gap of several
  seconds in ``time_elapsed`` that also carry a valid ``block_number`` (i.e.
  a real inter-block pause, not an early instruction/preload screen).
* On the LUX side: long high/low plateaus (several seconds), which correspond
  to the block-break/instruction screens shown between blocks.

Naive sequential (index-based) matching of every individual trial or
fixation-cross screen to LUX plateaus is unreliable: trial and fixation
durations overlap in range, so index-based pairing accumulates drift and can
be off by hundreds of seconds. Only the sparse, unambiguous block-break
landmarks give a trustworthy consensus offset; this module always resolves
the offset from those landmarks, never from sequential trial matching.

Validated (see ``project/diary`` and handover documents) against the full
ds004587 cohort: ~74% of subjects with usable physiology show >=3 matched
landmarks with sub-30ms residuals across ~15 minute recordings (no
detectable drift); ~20% show no LUX variation at all (sensor presumably
non-functional for that run) and must be excluded, not guessed at.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Minimum gap (ms) in behavioral time_elapsed to treat as a candidate block-break.
DEFAULT_MIN_GAP_MS = 4000.0
#: Minimum LUX plateau duration (ms) to treat as a candidate block-break landmark.
DEFAULT_MIN_PLATEAU_MS = 5000.0
#: Tolerance (ms) for counting a landmark as "matched" once a global offset is applied.
DEFAULT_MATCH_TOLERANCE_MS = 50.0
#: Minimum number of matched landmarks required to trust a run's offset.
DEFAULT_MIN_MATCHED_LANDMARKS = 3
#: Maximum residual (ms) among matched landmarks required to trust a run's offset.
DEFAULT_MAX_RESIDUAL_MS = 30.0

#: jsPsych trial_type carrying the actual illusion-judgment stimulus + response.
IG_TRIAL_TYPE = "image-keyboard-response"
#: Columns from beh.tsv preserved verbatim on each recovered trial row.
IG_TRIAL_BEH_COLUMNS = (
    "trial_index", "block", "block_number", "trial_number", "type",
    "illusion_strength", "illusion_difference", "correct", "correct_response",
    "response", "rt", "stimulus",
)


def behavioral_block_landmarks(
    time_elapsed_ms: np.ndarray,
    block_number: np.ndarray,
    *,
    min_gap_ms: float = DEFAULT_MIN_GAP_MS,
) -> np.ndarray:
    """Return behavioral-clock timestamps (ms) of candidate block-break landmarks.

    A landmark is the ``time_elapsed`` value of a row that is immediately
    followed by a gap of at least ``min_gap_ms`` AND that itself carries a
    finite ``block_number`` (this excludes early instruction/preload gaps,
    which occur before ``block_number`` is ever set).
    """
    te = np.asarray(time_elapsed_ms, dtype=np.float64)
    bn = np.asarray(block_number, dtype=np.float64)
    if te.size < 2:
        return np.empty(0, dtype=np.float64)
    gaps = np.diff(te)
    mask = (gaps >= min_gap_ms) & np.isfinite(bn[:-1])
    landmarks = te[:-1][mask]
    return np.array(sorted(set(landmarks.tolist())), dtype=np.float64)


def lux_plateau_landmarks(
    lux: np.ndarray,
    *,
    sfreq_hz: float = 1000.0,
    min_duration_ms: float = DEFAULT_MIN_PLATEAU_MS,
    collapse_within_ms: float = 1000.0,
) -> np.ndarray:
    """Return physio-clock timestamps (ms) of long LUX plateaus (either level).

    ``lux`` is a quantized photosensor signal (few discrete levels). A
    "plateau" is any maximal run between a rising and the next falling
    transition (or vice versa) that lasts at least ``min_duration_ms``.
    Near-duplicate detections (e.g. a transient mid-level sample splitting
    one true plateau into two) are collapsed when within
    ``collapse_within_ms`` of each other.
    """
    values = np.asarray(lux, dtype=np.float64)
    if values.size < 2:
        return np.empty(0, dtype=np.float64)
    ms_per_sample = 1000.0 / float(sfreq_hz)
    diffs = np.diff(values)
    rising = np.where(diffs > 0)[0]
    falling = np.where(diffs < 0)[0]
    min_samples = min_duration_ms / ms_per_sample

    starts_samples: list[int] = []
    fi = 0
    for r in rising:
        while fi < len(falling) and falling[fi] <= r:
            fi += 1
        if fi < len(falling) and (falling[fi] - r) >= min_samples:
            starts_samples.append(int(r) + 1)  # first sample of the new (plateau) level

    if not starts_samples:
        return np.empty(0, dtype=np.float64)

    collapse_samples = collapse_within_ms / ms_per_sample
    collapsed: list[int] = []
    for s in starts_samples:
        if not collapsed or (s - collapsed[-1]) > collapse_samples:
            collapsed.append(s)
    return np.asarray(collapsed, dtype=np.float64) * ms_per_sample


@dataclass
class LandmarkOffsetEstimate:
    """Result of landmark-consensus clock-offset estimation for one run."""

    offset_ms: float = float("nan")
    n_beh_landmarks: int = 0
    n_lux_landmarks: int = 0
    n_matched: int = 0
    max_abs_residual_ms: float = float("nan")
    matched_beh_ms: np.ndarray = field(default_factory=lambda: np.empty(0))
    matched_lux_ms: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def bracket_beh_ms(self) -> tuple[float, float]:
        """Behavioral-clock (lo, hi) interval spanned by matched landmarks.

        Trial recovery should be restricted to this interval: the validated
        constant-offset assumption only holds where it was actually checked,
        and extrapolating beyond it risks unbounded drift error.
        """
        if self.matched_beh_ms.size == 0:
            return (float("nan"), float("nan"))
        return (float(np.min(self.matched_beh_ms)), float(np.max(self.matched_beh_ms)))

    def qc_ok(
        self,
        *,
        min_matched: int = DEFAULT_MIN_MATCHED_LANDMARKS,
        max_residual_ms: float = DEFAULT_MAX_RESIDUAL_MS,
    ) -> bool:
        return (
            self.n_matched >= min_matched
            and np.isfinite(self.max_abs_residual_ms)
            and self.max_abs_residual_ms < max_residual_ms
        )


def estimate_landmark_consensus_offset(
    beh_landmarks_ms: np.ndarray,
    lux_landmarks_ms: np.ndarray,
    *,
    tolerance_ms: float = DEFAULT_MATCH_TOLERANCE_MS,
) -> LandmarkOffsetEstimate:
    """Estimate the behavioral-to-physio clock offset by landmark consensus.

    Tries every (behavioral landmark, LUX landmark) pair as a candidate
    global offset, then scores each candidate by how many OTHER landmarks
    independently agree with it within ``tolerance_ms``. This is robust to
    spurious/extra landmarks on either side (unlike sequential index
    matching, which fails badly when the two landmark counts differ) because
    the correct offset is the one every real landmark pair supports, while
    wrong offsets only explain the one pair used to generate them.

    Returns an estimate with ``n_matched = 0`` (and NaN offset/residual) when
    fewer than 2 landmarks exist on either side.
    """
    beh = np.asarray(beh_landmarks_ms, dtype=np.float64)
    lux = np.asarray(lux_landmarks_ms, dtype=np.float64)
    result = LandmarkOffsetEstimate(n_beh_landmarks=int(beh.size), n_lux_landmarks=int(lux.size))
    if beh.size < 2 or lux.size < 2:
        return result

    best_n = -1
    best_offset = float("nan")
    best_residuals: Optional[np.ndarray] = None
    best_matched_mask: Optional[np.ndarray] = None

    best_nearest_idx: Optional[np.ndarray] = None

    for b in beh:
        for l in lux:
            offset = l - b
            predicted = beh + offset
            abs_diffs = np.abs(predicted[:, None] - lux[None, :])
            nearest_idx = np.argmin(abs_diffs, axis=1)
            residuals = abs_diffs[np.arange(beh.size), nearest_idx]
            matched = residuals < tolerance_ms
            n_matched = int(matched.sum())
            if n_matched > best_n:
                best_n = n_matched
                best_offset = float(offset)
                best_residuals = residuals
                best_matched_mask = matched
                best_nearest_idx = nearest_idx

    if best_matched_mask is None or best_n <= 0 or best_nearest_idx is None:
        return result

    result.offset_ms = best_offset
    result.n_matched = best_n
    result.max_abs_residual_ms = float(np.max(best_residuals[best_matched_mask]))
    result.matched_beh_ms = beh[best_matched_mask]
    result.matched_lux_ms = lux[best_nearest_idx[best_matched_mask]]
    return result


def build_ig_trial_event_table(
    beh: pd.DataFrame,
    offset_estimate: LandmarkOffsetEstimate,
    *,
    subject_id: str,
    run_id: str,
    min_matched: int = DEFAULT_MIN_MATCHED_LANDMARKS,
    max_residual_ms: float = DEFAULT_MAX_RESIDUAL_MS,
) -> pd.DataFrame:
    """Build the per-trial recovered-onset table for one ds004587 IG run.

    One row per illusion-judgment trial (``trial_type ==
    "image-keyboard-response"``). ``onset_sec`` is finite only when the run's
    landmark-consensus offset passes the quality gate AND the trial's
    ``time_elapsed`` falls within the landmark-bracketed interval; otherwise
    it is NaN so the trial is transparently excluded downstream (the
    event-locked pipeline already treats non-finite ``onset_sec`` as
    "excluded", counted in its own QC), rather than guessed at via
    extrapolation.
    """
    trials = beh.loc[beh["trial_type"] == IG_TRIAL_TYPE].copy()
    trials = trials.reset_index(drop=True)

    qc_ok = offset_estimate.qc_ok(min_matched=min_matched, max_residual_ms=max_residual_ms)
    bracket_lo, bracket_hi = offset_estimate.bracket_beh_ms

    te = pd.to_numeric(trials.get("time_elapsed"), errors="coerce").to_numpy(dtype=np.float64)
    within_bracket = np.isfinite(te) & (te >= bracket_lo) & (te <= bracket_hi) if np.isfinite(bracket_lo) else np.zeros(len(te), dtype=bool)

    onset_sec = np.full(len(te), np.nan, dtype=np.float64)
    if qc_ok:
        recoverable = within_bracket & np.isfinite(te)
        onset_sec[recoverable] = (te[recoverable] + offset_estimate.offset_ms) / 1000.0

    out = pd.DataFrame({
        "onset_sec": onset_sec,
        "event_type": "ig_trial",
        "source": "derived:lux_photosensor_block_landmark_sync",
        "subject_id": subject_id,
        "run_id": run_id,
        "within_sync_bracket": within_bracket,
        "qc_ok_event_sync": bool(qc_ok),
        "sync_offset_ms": offset_estimate.offset_ms,
        "n_landmarks_matched": offset_estimate.n_matched,
        "max_residual_ms": offset_estimate.max_abs_residual_ms,
    })
    for col in IG_TRIAL_BEH_COLUMNS:
        if col in trials.columns:
            out[col] = trials[col].to_numpy()
    return out


def run_sync_quality_row(
    offset_estimate: LandmarkOffsetEstimate,
    *,
    subject_id: str,
    run_id: str,
    n_trials_total: int,
    n_trials_recovered: int,
    min_matched: int = DEFAULT_MIN_MATCHED_LANDMARKS,
    max_residual_ms: float = DEFAULT_MAX_RESIDUAL_MS,
) -> dict:
    """Return one reviewable cohort-audit row summarizing sync quality."""
    return {
        "subject_id": subject_id,
        "run_id": run_id,
        "n_beh_landmarks": offset_estimate.n_beh_landmarks,
        "n_lux_landmarks": offset_estimate.n_lux_landmarks,
        "n_landmarks_matched": offset_estimate.n_matched,
        "max_residual_ms": offset_estimate.max_abs_residual_ms,
        "sync_offset_ms": offset_estimate.offset_ms,
        "qc_ok_event_sync": bool(offset_estimate.qc_ok(min_matched=min_matched, max_residual_ms=max_residual_ms)),
        "n_trials_total": n_trials_total,
        "n_trials_recovered": n_trials_recovered,
    }
