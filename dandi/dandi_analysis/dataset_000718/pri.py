"""Preferential Reactivation Index (PRI) for H1 selectivity testing.

The PRI asks: during an offline event, are the exact units that define a
NeutralExposure encoding ensemble overrepresented among the active units,
relative to size-matched random unit sets?

Unlike projection-based scores, PRI uses a binary unit-level criterion and
a size-matched null that explicitly controls for population burst magnitude.
This makes it insensitive to the trivial confound that large population bursts
inflate any ensemble projection.

Design
------
For each encoding ensemble:
  1. Define the ensemble "core" as the top-k units by spatial weight
     (k = top_frac × N registered units, minimum 5).
  2. For each offline event, define which units are "active":
     units whose mean z-score during the event exceeds a threshold.
  3. Observed selectivity = fraction of core units that are active.
  4. Null: repeatedly draw k random units and compute their active fraction.
  5. PRI z-score = (observed − null_mean) / null_std.

PRI > 0 means the encoding-defined core units are more frequently active
during the event than expected by chance given the total number of active units.
PRI near 0 means the active units during the event are randomly distributed
across the population — no selectivity for the encoding-defined core.

Selectivity controls
--------------------
- registration shuffle: shuffle the offline unit mapping → core units now
  correspond to random biological cells → expected PRI ≈ 0.
- event-time shuffle: random event placement → expected PRI ≈ 0.
- mismatched pair: different animal → expected PRI ≈ 0 if biology matters.

A successful specificity demonstration requires real > shuffled for
registration shuffle AND event-time shuffle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PriScore:
    """PRI result for one (event, ensemble) pair."""
    event_idx: int
    ensemble_idx: int
    k: int                      # number of core units tested
    n_active_total: int         # total active units in event
    n_active_core: int          # observed: active core units
    active_frac_core: float     # n_active_core / k
    null_mean: float
    null_std: float
    z_score: float


@dataclass
class PriSessionResult:
    """All PRI scores for one session pair."""
    session_ne: str
    session_offline: str
    n_registered: int
    n_events: int
    n_ensembles: int
    top_frac: float
    activity_threshold: float
    null_n: int
    scores: list[PriScore]
    # Session-level summary (filled by summarise())
    n_significant: int = 0
    fraction_significant: float = 0.0
    mean_z: float = float("nan")
    max_z: float = float("nan")

    def summarise(self) -> None:
        valid = [s for s in self.scores if s.z_score == s.z_score]
        self.n_significant = sum(1 for s in valid if s.z_score > 1.96)
        self.fraction_significant = round(self.n_significant / max(1, len(valid)), 3)
        zvals = [s.z_score for s in valid]
        self.mean_z = round(float(np.mean(zvals)), 3) if zvals else float("nan")
        self.max_z = round(float(max(zvals, default=float("nan"))), 3)

    def to_dict(self) -> dict[str, Any]:
        self.summarise()
        return {
            "session_ne": self.session_ne,
            "session_offline": self.session_offline,
            "n_registered": self.n_registered,
            "n_events": self.n_events,
            "n_ensembles": self.n_ensembles,
            "top_frac": self.top_frac,
            "activity_threshold": self.activity_threshold,
            "null_n": self.null_n,
            "n_significant": self.n_significant,
            "fraction_significant": self.fraction_significant,
            "mean_z": self.mean_z,
            "max_z": self.max_z,
            "total_scored_pairs": len(self.scores),
        }


# ---------------------------------------------------------------------------
# Core PRI computation
# ---------------------------------------------------------------------------

def compute_pri_event(
    event_data: np.ndarray,
    ensemble_weights: np.ndarray,
    ensemble_idx: int,
    *,
    top_frac: float = 0.15,
    activity_threshold: float = 0.0,
    null_n: int = 500,
    rng_seed: int = 42,
) -> PriScore:
    """Compute PRI for one (event, ensemble) pair.

    Parameters
    ----------
    event_data:
        Activity during the event window, shape (T_event, N). Units should be
        z-scored relative to the full offline session baseline.
    ensemble_weights:
        Spatial weights from the encoding-defined assembly, shape (N,).
        Higher = more important for that ensemble.
    top_frac:
        Fraction of N units that constitute the ensemble "core".
    activity_threshold:
        A unit is "active" during the event if its mean activity across the
        event window exceeds this z-score threshold.
    null_n:
        Number of random size-matched draws for the null.

    Returns
    -------
    PriScore with z_score, observed active fraction, and null statistics.
    """
    N = len(ensemble_weights)
    k = max(5, int(round(N * top_frac)))
    k = min(k, N)

    # Define core units: top-k by ensemble weight
    core_units = np.argsort(ensemble_weights)[-k:]

    # Define active units: mean z-score during event > threshold
    mean_act = event_data.mean(axis=0)    # (N,)
    active_mask = mean_act > activity_threshold
    n_active_total = int(active_mask.sum())

    # Observed: how many core units are active?
    n_active_core = int(active_mask[core_units].sum())
    frac_core = n_active_core / k

    # Null: size-matched random draws
    rng = np.random.default_rng(rng_seed + ensemble_idx)
    null_counts: list[float] = []
    for _ in range(null_n):
        rand_units = rng.choice(N, size=k, replace=False)
        null_counts.append(float(active_mask[rand_units].sum()) / k)

    null_arr = np.array(null_counts)
    null_mean = float(null_arr.mean())
    null_std = float(null_arr.std())
    z = (frac_core - null_mean) / null_std if null_std > 0 else float("nan")

    return PriScore(
        event_idx=-1,           # set by caller
        ensemble_idx=ensemble_idx,
        k=k,
        n_active_total=n_active_total,
        n_active_core=n_active_core,
        active_frac_core=round(frac_core, 4),
        null_mean=round(null_mean, 4),
        null_std=round(null_std, 4),
        z_score=round(z, 3) if z == z else float("nan"),
    )


def _active_frac_core(
    chunk: "np.ndarray",
    core_units: "np.ndarray",
    activity_threshold: float,
) -> float:
    """Mean activity fraction of core units in a data chunk (T, N)."""
    mean_act = chunk.mean(axis=0)
    return float((mean_act[core_units] > activity_threshold).sum()) / len(core_units)


# ---------------------------------------------------------------------------
# Intra-vs-inter-event PRI enrichment
# ---------------------------------------------------------------------------

@dataclass
class PriEnrichmentScore:
    """Enrichment of PRI at events relative to matched inter-event baseline."""
    event_idx: int
    ensemble_idx: int
    k: int
    activity_threshold: float
    # Active fractions
    event_frac: float
    inter_frac: float
    enrichment: float           # event_frac − inter_frac  (> 0 means event-enriched)


@dataclass
class PriEnrichmentResult:
    """Session-level enrichment summary."""
    session_ne: str
    session_offline: str
    n_registered: int
    n_events: int
    n_ensembles: int
    top_frac: float
    activity_threshold: float
    scores: list[PriEnrichmentScore]
    # Filled by summarise()
    mean_enrichment: float = float("nan")
    enrichment_z: float = float("nan")          # permutation-derived
    frac_positive: float = float("nan")         # fraction of pairs with enrichment > 0

    def summarise(self) -> None:
        import numpy as np
        enrichments = [s.enrichment for s in self.scores]
        if not enrichments:
            return
        arr = np.array(enrichments)
        self.mean_enrichment = round(float(arr.mean()), 4)
        self.frac_positive = round(float((arr > 0).mean()), 3)
        sd = arr.std()
        self.enrichment_z = round(float(arr.mean() / (sd / np.sqrt(len(arr)))) if sd > 0 else float("nan"), 3)

    def to_dict(self) -> dict:
        self.summarise()
        return {
            "session_ne": self.session_ne,
            "session_offline": self.session_offline,
            "n_registered": self.n_registered,
            "n_events": self.n_events,
            "n_ensembles": self.n_ensembles,
            "top_frac": self.top_frac,
            "activity_threshold": self.activity_threshold,
            "mean_enrichment": self.mean_enrichment,
            "enrichment_z": self.enrichment_z,
            "frac_positive": self.frac_positive,
            "n_scored_pairs": len(self.scores),
        }


def compute_pri_enrichment_session(
    offline_data: "np.ndarray",
    ensemble_weights_list: list["np.ndarray"],
    events: list,
    session_ne: str,
    session_offline: str,
    n_registered: int,
    *,
    top_frac: float = 0.15,
    activity_threshold: float = 0.0,
    n_inter_samples: int = 5,
    rng_seed: int = 42,
) -> PriEnrichmentResult:
    """Compute intra-vs-inter-event PRI enrichment for a session.

    For each (event, ensemble) pair:
      - measure active fraction of core units DURING the event
      - measure active fraction of core units in `n_inter_samples` duration-matched
        windows drawn uniformly from the inter-event offline frames
      - enrichment = event_frac − mean(inter_fracs)

    A positive mean enrichment across all pairs indicates that the high-synchrony
    events specifically drive reactivation of the encoding-defined core units,
    beyond the session-wide elevated baseline.

    The `enrichment_z` in `PriEnrichmentResult.summarise()` is a standard error
    t-like statistic (mean / SE) over all scored pairs, testing H0: enrichment = 0.
    """
    import numpy as np

    T = offline_data.shape[0]
    rng = np.random.default_rng(rng_seed)

    # Build inter-event frame pool (complement of all event windows)
    event_mask = np.zeros(T, dtype=bool)
    for ev in events:
        event_mask[ev.start_frame:ev.stop_frame] = True
    inter_frames = np.where(~event_mask)[0]

    scores: list[PriEnrichmentScore] = []

    for ens_idx, weights in enumerate(ensemble_weights_list):
        N = len(weights)
        k = max(5, int(round(N * top_frac)))
        k = min(k, N)
        core_units = np.argsort(weights)[-k:]

        for ev in events:
            dur = ev.duration_frames
            if dur == 0:
                continue

            event_chunk = offline_data[ev.start_frame:ev.stop_frame]
            ev_frac = _active_frac_core(event_chunk, core_units, activity_threshold)

            # Sample n_inter_samples duration-matched inter-event windows
            if len(inter_frames) < dur:
                continue

            inter_fracs: list[float] = []
            for _ in range(n_inter_samples):
                start_pos = rng.integers(0, len(inter_frames) - dur + 1)
                chunk = offline_data[inter_frames[start_pos: start_pos + dur]]
                inter_fracs.append(_active_frac_core(chunk, core_units, activity_threshold))

            mean_inter = float(np.mean(inter_fracs))
            scores.append(PriEnrichmentScore(
                event_idx=ev.event_idx,
                ensemble_idx=ens_idx,
                k=k,
                activity_threshold=activity_threshold,
                event_frac=round(ev_frac, 4),
                inter_frac=round(mean_inter, 4),
                enrichment=round(ev_frac - mean_inter, 4),
            ))

    result = PriEnrichmentResult(
        session_ne=session_ne,
        session_offline=session_offline,
        n_registered=n_registered,
        n_events=len(events),
        n_ensembles=len(ensemble_weights_list),
        top_frac=top_frac,
        activity_threshold=activity_threshold,
        scores=scores,
    )
    result.summarise()
    return result


def run_pri_session(
    ne_data: np.ndarray,
    offline_data: np.ndarray,
    ensemble_weights_list: list[np.ndarray],
    events: list,
    session_ne: str,
    session_offline: str,
    n_registered: int,
    *,
    top_frac: float = 0.15,
    activity_threshold: float = 0.0,
    null_n: int = 500,
    rng_seed: int = 42,
) -> PriSessionResult:
    """Run PRI for all (event, ensemble) pairs in one session.

    Parameters
    ----------
    ne_data:
        Encoding activity matrix (T_ne, N), z-scored.
    offline_data:
        Offline activity matrix (T_off, N), z-scored relative to full offline
        session (not just the event window).
    ensemble_weights_list:
        List of spatial weight arrays, one per encoding ensemble.
    events:
        List of SynchronyEvent objects from event detection.
    """
    scores: list[PriScore] = []

    for ev in events:
        event_chunk = offline_data[ev.start_frame:ev.stop_frame]
        if len(event_chunk) == 0:
            continue
        for ens_idx, weights in enumerate(ensemble_weights_list):
            score = compute_pri_event(
                event_chunk, weights, ens_idx,
                top_frac=top_frac,
                activity_threshold=activity_threshold,
                null_n=null_n,
                rng_seed=rng_seed,
            )
            score.event_idx = ev.event_idx
            scores.append(score)

    result = PriSessionResult(
        session_ne=session_ne,
        session_offline=session_offline,
        n_registered=n_registered,
        n_events=len(events),
        n_ensembles=len(ensemble_weights_list),
        top_frac=top_frac,
        activity_threshold=activity_threshold,
        null_n=null_n,
        scores=scores,
    )
    result.summarise()
    return result
