"""Identifiable lag-n source transitions shared by drift and diffusion.

This is a support object, not a measurement level. ``support_id`` hashes
source indices, lag, gap policy, the irregular-``dt`` threshold, and
``distance_metric_id``. It does not hash neighborhood ``k`` or kNN weights. Embargo, when applied,
is ``index_steps`` only: overlapping analysis windows or filters are not
covered (SL-LEV-MES-003 §7.1 / §8).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

import numpy as np

from .measurement_register import DISTANCE_EUCLIDEAN_CHART
from .validity import increment_pairs_at_lag

EMBARGO_SEMANTICS_INDEX_STEPS = "index_steps"
REASON_IRREGULAR_DT = "materially_irregular_increment_timestep"
REASON_NON_POSITIVE_DT = "non_positive_nominal_dt"
_SUPPORT_HASH_PREFIX = b"mndm.transition_support.v1"


def compute_support_id(
    source_idx: np.ndarray,
    *,
    lag: int,
    max_gap_sec: float | None,
    max_dt_relative_deviation: float,
    distance_metric_id: str = DISTANCE_EUCLIDEAN_CHART,
) -> str:
    """Deterministic identity of a source-transition set. Not a kNN hash."""
    hasher = hashlib.sha256()
    hasher.update(_SUPPORT_HASH_PREFIX)
    hasher.update(np.asarray(source_idx, dtype=np.int32).reshape(-1).tobytes())
    hasher.update(np.int32(lag).tobytes())
    if max_gap_sec is None:
        hasher.update(b"max_gap=none")
    else:
        hasher.update(f"max_gap={float(max_gap_sec):.17g}".encode("ascii"))
    hasher.update(f"max_dt_rel={float(max_dt_relative_deviation):.17g}".encode("ascii"))
    hasher.update(str(distance_metric_id).encode("utf-8"))
    return hasher.hexdigest()[:32]


@dataclass(frozen=True)
class TransitionSupport:
    """Lag-n increment pairs plus an identity that downstream families can share."""

    source_idx: np.ndarray
    increments: np.ndarray
    dts: np.ndarray
    lag: int
    nominal_dt_sec: float
    max_dt_relative_deviation: float
    observed_dt_relative_deviation: float
    max_gap_sec: float | None
    distance_metric_id: str
    support_id: str
    embargo_semantics: str = EMBARGO_SEMANTICS_INDEX_STEPS
    failure_reason: str | None = None

    def subset(self, mask: np.ndarray) -> "TransitionSupport":
        """Return a variant subset (for example an index-embargoed cross-fit fold union)."""
        keep = np.asarray(mask, dtype=bool).reshape(-1)
        if keep.size != int(self.source_idx.size):
            raise ValueError("subset mask must align with source_idx")
        return support_from_pairs(
            source_idx=self.source_idx[keep],
            increments=self.increments[keep],
            dts=self.dts[keep],
            lag=self.lag,
            max_gap_sec=self.max_gap_sec,
            max_dt_relative_deviation=self.max_dt_relative_deviation,
            distance_metric_id=self.distance_metric_id,
        )


def support_from_pairs(
    *,
    source_idx: np.ndarray,
    increments: np.ndarray,
    dts: np.ndarray,
    lag: int,
    max_gap_sec: float | None,
    max_dt_relative_deviation: float,
    distance_metric_id: str = DISTANCE_EUCLIDEAN_CHART,
) -> TransitionSupport:
    """Wrap already-selected increment pairs. Does not re-run gap filtering."""
    idx = np.asarray(source_idx, dtype=np.int32).reshape(-1)
    dx = np.asarray(increments, dtype=float)
    dt = np.asarray(dts, dtype=float).reshape(-1)
    support_id = compute_support_id(
        idx,
        lag=int(lag),
        max_gap_sec=max_gap_sec,
        max_dt_relative_deviation=float(max_dt_relative_deviation),
        distance_metric_id=str(distance_metric_id),
    )
    if dt.size == 0:
        return TransitionSupport(
            source_idx=idx,
            increments=dx,
            dts=dt,
            lag=int(lag),
            nominal_dt_sec=float("nan"),
            max_dt_relative_deviation=float(max_dt_relative_deviation),
            observed_dt_relative_deviation=float("nan"),
            max_gap_sec=max_gap_sec,
            distance_metric_id=str(distance_metric_id),
            support_id=support_id,
            embargo_semantics=EMBARGO_SEMANTICS_INDEX_STEPS,
            failure_reason=None,
        )
    nominal_dt = float(np.median(dt))
    if not np.isfinite(nominal_dt) or nominal_dt <= 0.0:
        return TransitionSupport(
            source_idx=idx,
            increments=dx,
            dts=dt,
            lag=int(lag),
            nominal_dt_sec=nominal_dt,
            max_dt_relative_deviation=float(max_dt_relative_deviation),
            observed_dt_relative_deviation=float("nan"),
            max_gap_sec=max_gap_sec,
            distance_metric_id=str(distance_metric_id),
            support_id=support_id,
            embargo_semantics=EMBARGO_SEMANTICS_INDEX_STEPS,
            failure_reason=REASON_NON_POSITIVE_DT,
        )
    relative_deviation = float(np.max(np.abs(dt - nominal_dt)) / nominal_dt)
    failure = None
    if relative_deviation > float(max_dt_relative_deviation):
        failure = REASON_IRREGULAR_DT
    return TransitionSupport(
        source_idx=idx,
        increments=dx,
        dts=dt,
        lag=int(lag),
        nominal_dt_sec=nominal_dt,
        max_dt_relative_deviation=float(max_dt_relative_deviation),
        observed_dt_relative_deviation=relative_deviation,
        max_gap_sec=max_gap_sec,
        distance_metric_id=str(distance_metric_id),
        support_id=support_id,
        embargo_semantics=EMBARGO_SEMANTICS_INDEX_STEPS,
        failure_reason=failure,
    )


def build_transition_support(
    state: np.ndarray,
    time: np.ndarray,
    segment_id: np.ndarray,
    *,
    lag: int = 1,
    max_gap_sec: float | None = None,
    max_dt_relative_deviation: float = 0.05,
    distance_metric_id: str = DISTANCE_EUCLIDEAN_CHART,
) -> TransitionSupport:
    """Select within-segment lag-n pairs and attach a ``support_id``.

    Pair selection is ``increment_pairs_at_lag`` (interior NaN fail-closed).
    Neighborhood size is not part of this object.
    """
    source_idx, increments, dts = increment_pairs_at_lag(
        state, time, segment_id, lag=int(lag), max_gap_sec=max_gap_sec
    )
    return support_from_pairs(
        source_idx=source_idx,
        increments=increments,
        dts=dts,
        lag=int(lag),
        max_gap_sec=max_gap_sec,
        max_dt_relative_deviation=float(max_dt_relative_deviation),
        distance_metric_id=str(distance_metric_id),
    )


def support_series_fields(support: TransitionSupport) -> dict[str, Any]:
    return {"source_idx": np.asarray(support.source_idx, dtype=np.int32)}


def support_summary_fields(support: TransitionSupport) -> dict[str, Any]:
    return {
        "transition_support_id": support.support_id,
        "transition_support_lag": int(support.lag),
        "embargo_semantics": support.embargo_semantics,
        "distance_metric_id": support.distance_metric_id,
    }


def support_settings_fields(support: TransitionSupport) -> dict[str, Any]:
    return {
        "transition_support_id": support.support_id,
        "embargo_semantics": support.embargo_semantics,
        "distance_metric_id": support.distance_metric_id,
    }


def lag1_overlap_fields(
    left_idx: np.ndarray,
    left_id: str | None,
    right_idx: np.ndarray,
    right_id: str | None,
) -> dict[str, Any]:
    """Common source-index identity. Neighborhood overlap is not compared."""
    left = np.asarray(left_idx, dtype=np.int32).reshape(-1)
    right = np.asarray(right_idx, dtype=np.int32).reshape(-1)
    return {
        "lag1_support_ids_match": bool(left_id is not None and left_id == right_id),
        "n_common_source_idx": int(np.intersect1d(left, right).size),
    }


def stamp_lag1_support_overlap(export: Mapping[str, Any]) -> None:
    """When both pooled lag-1 drift and diffusion computed, stamp shared-support fields.

    Mutates ``export`` in place. Different post-kNN validity masks are allowed;
    the comparison is source transitions, not neighborhoods.
    """
    from .measurement_register import MEASUREMENT_ID_CONDITIONAL_MEAN_RATE, VARIANT_POOLED

    diffusion = export.get("diffusion")
    drift = export.get("drift")
    if not isinstance(diffusion, Mapping) or not isinstance(drift, Mapping):
        return
    if diffusion.get("computation_status") != "computed":
        return
    pooled = (drift.get(MEASUREMENT_ID_CONDITIONAL_MEAN_RATE) or {}).get(VARIANT_POOLED)
    if not isinstance(pooled, Mapping) or pooled.get("computation_status") != "computed":
        return
    d_series = diffusion.get("series") or {}
    p_series = pooled.get("series") or {}
    if "source_idx" not in d_series or "source_idx" not in p_series:
        return
    overlap = lag1_overlap_fields(
        d_series["source_idx"],
        (diffusion.get("summary") or {}).get("transition_support_id"),
        p_series["source_idx"],
        (pooled.get("summary") or {}).get("transition_support_id"),
    )
    for result in (diffusion, pooled):
        summary = dict(result.get("summary") or {})
        summary.update(overlap)
        result["summary"] = summary
