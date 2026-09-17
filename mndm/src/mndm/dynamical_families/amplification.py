"""Observed same-pair neighbor gain, separation, and cloud volume (001).

``neighbor_gain_q90_level1`` is the per-source 0.90 quantile of
``(d_{ij}^{(1)}+ε)/(d_{ij}^{(0)}+ε)`` for Euclidean pairs selected at the
source and followed one real lag-1 step.

Nested identities on the **same** pairs, not YAML toggles:

* ``neighbor_separation_rate_level1``: per-source median of
  ``log((d1+ε)/(d0+ε))/nominal_dt``. Not spectral abscissa.
* ``neighbor_gain_rate_q90_level1``: ``log(G_{q90})/nominal_dt`` of the
  already written q90. Not a new pair set.
* ``cloud_volume_change_rate_level1``: same-cloud
  ``(logdet(C1+εI)-logdet(C0+εI))/(2 nominal_dt)``. Epsilon is a
  documented logdet floor, not operator-volume rescue.

Neighbors are not re-selected at the successor.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import AMPLIFICATION_SCHEMA_VERSION, build_provenance, unavailable_result
from .measurement_register import (
    MEASUREMENT_ID_CLOUD_VOLUME,
    MEASUREMENT_ID_NEIGHBOR_GAIN,
    MEASUREMENT_ID_NEIGHBOR_GAIN_RATE,
    MEASUREMENT_ID_NEIGHBOR_SEPARATION,
    QUALIFICATION_CLOUD_VOLUME,
    QUALIFICATION_NOT_ASSESSED,
    QUALIFICATION_SAME_PAIR_GAIN,
    QUALIFICATION_SAME_PAIR_GAIN_RATE,
    QUALIFICATION_SAME_PAIR_SEPARATION,
    stamp_register_fields,
)
from .transition_support import (
    build_transition_support,
    support_series_fields,
    support_settings_fields,
    support_summary_fields,
)
from .validity import chunked_nearest_neighbors, validate_trajectory
from ..inferential_grain import attach_grain_for_schema
from ..measurement_certificate import attach_certificate

ESTIMATOR_NAME = "same_pair_neighbor_gain_q90"
DEFAULT_Q = 0.90
DEFAULT_EPSILON = 1e-12
REASON_SUBJECT_ANCHORED_3D = "amplification_subject_anchored_3d_only"
REASON_Q = "neighbor_gain_q_not_implemented"
REASON_INSUFFICIENT = "insufficient_local_support"
REASON_EPSILON = "neighbor_gain_distance_epsilon_invalid"
_SUBJECT_ANCHORED_LAYERS = frozenset({"coords_3d_subject_anchored", "subject_anchored"})
_AMP_LEAF_IDS = (
    MEASUREMENT_ID_NEIGHBOR_GAIN,
    MEASUREMENT_ID_NEIGHBOR_SEPARATION,
    MEASUREMENT_ID_NEIGHBOR_GAIN_RATE,
    MEASUREMENT_ID_CLOUD_VOLUME,
)


def _finalize(result: Mapping[str, Any]) -> dict[str, Any]:
    return attach_grain_for_schema(attach_certificate(result), AMPLIFICATION_SCHEMA_VERSION)


def _stamp_qualification(result: dict[str, Any], token: str) -> dict[str, Any]:
    out = dict(result)
    out["qualification_status"] = token
    summary = dict(out.get("summary") or {})
    summary["qualification_status"] = token
    out["summary"] = summary
    return out


def _nested_leaf(*, measurement_id: str, qualification: str, result: Mapping[str, Any]) -> dict[str, Any]:
    leaf = stamp_register_fields(dict(result), measurement_id)
    leaf = _stamp_qualification(leaf, qualification)
    return _finalize(leaf)


def _unavailable(
    *,
    status: str,
    failure_reason: str,
    coordinate_layer: str,
    coordinate_names: list[str],
    extra_summary: Mapping[str, Any] | None = None,
    extra_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = unavailable_result(
        AMPLIFICATION_SCHEMA_VERSION,
        status=status,
        failure_reason=failure_reason,
        coordinate_layer=coordinate_layer,
        coordinate_names=coordinate_names,
    )
    summary = dict(result.get("summary") or {})
    if extra_summary:
        summary.update(dict(extra_summary))
    result["summary"] = summary
    provenance = dict(result.get("provenance") or {})
    settings = dict(provenance.get("settings") or {})
    settings["estimator"] = ESTIMATOR_NAME
    settings["not_operator_max_gain"] = True
    settings["not_resampled_at_target"] = True
    settings["not_spectral_abscissa"] = True
    settings["not_operator_volume"] = True
    settings["epsilon_logdet_regularized"] = True
    if extra_settings:
        settings.update(dict(extra_settings))
    provenance["settings"] = settings
    provenance["estimator"] = ESTIMATOR_NAME
    result["provenance"] = provenance
    parent = stamp_register_fields(result, MEASUREMENT_ID_NEIGHBOR_GAIN)
    parent = _stamp_qualification(parent, QUALIFICATION_SAME_PAIR_GAIN)
    parent = _finalize(parent)
    gain_leaf = dict(parent)
    parent[MEASUREMENT_ID_NEIGHBOR_GAIN] = gain_leaf
    for measurement_id in _AMP_LEAF_IDS:
        if measurement_id == MEASUREMENT_ID_NEIGHBOR_GAIN:
            continue
        parent[measurement_id] = _nested_leaf(
            measurement_id=measurement_id,
            qualification=QUALIFICATION_NOT_ASSESSED,
            result=gain_leaf,
        )
    return parent


def unavailable_amplification_family(
    *,
    status: str,
    failure_reason: str,
    coordinate_layer: str,
    coordinate_names: list[str],
    extra_summary: Mapping[str, Any] | None = None,
    extra_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Schema-complete unavailable amplification payload with all L1 leaves."""
    return _unavailable(
        status=status,
        failure_reason=failure_reason,
        coordinate_layer=coordinate_layer,
        coordinate_names=coordinate_names,
        extra_summary=extra_summary,
        extra_settings=extra_settings,
    )


def _identity_leaf(
    *,
    measurement_id: str,
    qualification: str,
    identified: bool,
    failure_reason: str | None,
    series: Mapping[str, Any],
    summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    status = "computed" if identified else "insufficient_support"
    result = {
        "schema_version": AMPLIFICATION_SCHEMA_VERSION,
        "computation_status": status,
        "failure_reason": failure_reason,
        "series": dict(series),
        "summary": dict(summary),
        "provenance": dict(provenance),
    }
    token = qualification
    return _nested_leaf(measurement_id=measurement_id, qualification=token, result=result)


def _cloud_logdet_rate(
    positions_0: np.ndarray,
    positions_1: np.ndarray,
    eps: float,
    dt: float,
) -> float:
    n_pts, dim = positions_0.shape
    if n_pts < dim + 1 or dt <= 0:
        return float("nan")
    cov0 = np.atleast_2d(np.cov(positions_0, rowvar=False, ddof=1))
    cov1 = np.atleast_2d(np.cov(positions_1, rowvar=False, ddof=1))
    eye = np.eye(dim, dtype=float)
    sign0, log0 = np.linalg.slogdet(cov0 + eps * eye)
    sign1, log1 = np.linalg.slogdet(cov1 + eps * eye)
    if sign0 <= 0 or sign1 <= 0 or not np.isfinite(log0) or not np.isfinite(log1):
        return float("nan")
    return float((log1 - log0) / (2.0 * dt))


def estimate_neighbor_gain_q90(
    state: np.ndarray,
    time: np.ndarray,
    *,
    segment_id: np.ndarray | None = None,
    coordinate_layer: str = "coords_3d_subject_anchored",
    coordinate_names: Sequence[str] | None = None,
    neighborhood_k: int = 20,
    min_samples: int = 30,
    min_neighborhood_samples: int = 10,
    max_gap_sec: float | None = None,
    max_dt_relative_deviation: float = 0.05,
    q: float = DEFAULT_Q,
    distance_epsilon: float = DEFAULT_EPSILON,
) -> dict[str, Any]:
    """Per-source q90 of same-pair Euclidean neighbor gains at lag 1."""
    names = list(coordinate_names or ["m", "d", "e"])
    layer = str(coordinate_layer or "").strip()
    x_in = np.asarray(state)
    if layer not in _SUBJECT_ANCHORED_LAYERS or x_in.ndim != 2 or x_in.shape[1] != 3:
        return _unavailable(
            status="not_testable",
            failure_reason=REASON_SUBJECT_ANCHORED_3D,
            coordinate_layer=layer or "unknown",
            coordinate_names=names,
        )
    if abs(float(q) - DEFAULT_Q) > 1e-12:
        return _unavailable(
            status="not_testable",
            failure_reason=REASON_Q,
            coordinate_layer=layer,
            coordinate_names=names,
            extra_summary={"requested_q": float(q), "implemented_q": DEFAULT_Q},
        )
    eps = float(distance_epsilon)
    if not np.isfinite(eps) or eps < 0.0:
        return _unavailable(
            status="invalid",
            failure_reason=REASON_EPSILON,
            coordinate_layer=layer,
            coordinate_names=names,
            extra_summary={"distance_epsilon": eps},
        )
    x, t, segments, _finite, reason = validate_trajectory(
        state, time, min_samples=min_samples, segment_id=segment_id
    )
    if reason is not None or x is None or t is None or segments is None:
        return _unavailable(
            status="insufficient_support" if reason == "insufficient_samples" else "invalid",
            failure_reason=str(reason or "invalid_trajectory"),
            coordinate_layer=layer,
            coordinate_names=names,
        )
    support = build_transition_support(
        x,
        t,
        segments,
        lag=1,
        max_gap_sec=max_gap_sec,
        max_dt_relative_deviation=float(max_dt_relative_deviation),
    )
    shared_settings = {
        "estimator": ESTIMATOR_NAME,
        "neighborhood_k": int(neighborhood_k),
        "min_neighborhood_samples": int(min_neighborhood_samples),
        "q": DEFAULT_Q,
        "distance_epsilon": float(distance_epsilon),
        "same_pair_forward": True,
        "not_resampled_at_target": True,
        "not_operator_max_gain": True,
        "not_spectral_abscissa": True,
        "not_peak_gain_level4": True,
        "not_operator_volume": True,
        "epsilon_logdet_regularized": True,
        "aggregation": "per_source_q90_not_pooled_pair_q90",
        "distance_metric_id": "euclidean_on_release_scaled_chart",
        **support_settings_fields(support),
    }
    if support.failure_reason == "non_positive_nominal_dt":
        return _unavailable(
            status="invalid",
            failure_reason="non_positive_nominal_dt",
            coordinate_layer=layer,
            coordinate_names=names,
            extra_settings=shared_settings,
        )
    if support.failure_reason == "materially_irregular_increment_timestep":
        return _unavailable(
            status="not_testable",
            failure_reason="materially_irregular_increment_timestep",
            coordinate_layer=layer,
            coordinate_names=names,
            extra_settings=shared_settings,
        )
    source_idx = np.asarray(support.source_idx, dtype=np.int32)
    n_refs = int(source_idx.size)
    k = int(neighborhood_k)
    min_pairs = int(min_neighborhood_samples)
    n_time = int(x.shape[0])
    dim = int(x.shape[1])
    dt = float(support.nominal_dt_sec)
    gain_q90 = np.full(n_time, np.nan, dtype=np.float32)
    sep_rate = np.full(n_time, np.nan, dtype=np.float32)
    gain_rate = np.full(n_time, np.nan, dtype=np.float32)
    cloud_rate = np.full(n_time, np.nan, dtype=np.float32)
    n_pairs = np.zeros(n_time, dtype=np.int32)
    if n_refs < max(min_pairs + 1, k + 1) or k < 1 or dt <= 0:
        return _unavailable(
            status="insufficient_support",
            failure_reason=REASON_INSUFFICIENT,
            coordinate_layer=layer,
            coordinate_names=names,
            extra_settings=shared_settings,
        )
    x_src = x[source_idx]
    x_tgt = x_src + np.asarray(support.increments, dtype=float)
    k_use = min(k + 1, n_refs)
    nearest_indices, _squared = chunked_nearest_neighbors(x_src, x_src, k_use)
    n_valid_gain = 0
    n_valid_sep = 0
    n_valid_cloud = 0
    min_cloud = dim + 1
    for row, center_source in enumerate(source_idx):
        neighbors = np.asarray(nearest_indices[row], dtype=np.intp)
        keep = neighbors != row
        neighbors = neighbors[keep][:k]
        if int(neighbors.size) < min_pairs:
            continue
        d0 = np.linalg.norm(x_src[row] - x_src[neighbors], axis=1)
        d1 = np.linalg.norm(x_tgt[row] - x_tgt[neighbors], axis=1)
        finite = np.isfinite(d0) & np.isfinite(d1)
        if int(np.sum(finite)) < min_pairs:
            continue
        ratios = (d1[finite] + eps) / (d0[finite] + eps)
        t_idx = int(center_source)
        n_pairs[t_idx] = np.int32(ratios.size)
        q90 = float(np.quantile(ratios, DEFAULT_Q))
        gain_q90[t_idx] = np.float32(q90)
        n_valid_gain += 1
        if np.isfinite(q90) and q90 > 0.0:
            gain_rate[t_idx] = np.float32(np.log(q90) / dt)
        finite_rates = np.log(ratios[ratios > 0.0]) / dt
        if finite_rates.size:
            sep_rate[t_idx] = np.float32(np.median(finite_rates))
            n_valid_sep += 1
        cloud_idx = np.concatenate([np.array([row], dtype=np.intp), neighbors])
        if int(cloud_idx.size) >= min_cloud:
            rate = _cloud_logdet_rate(x_src[cloud_idx], x_tgt[cloud_idx], eps, dt)
            if np.isfinite(rate):
                cloud_rate[t_idx] = np.float32(rate)
                n_valid_cloud += 1
    if n_valid_gain == 0:
        return _unavailable(
            status="insufficient_support",
            failure_reason=REASON_INSUFFICIENT,
            coordinate_layer=layer,
            coordinate_names=names,
            extra_settings=shared_settings,
        )
    finite_gains = gain_q90[np.isfinite(gain_q90)]
    support_series = support_series_fields(support)
    support_summary = support_summary_fields(support)
    common_flags = {
        "q": DEFAULT_Q,
        "distance_epsilon": eps,
        "same_pair_forward": True,
        "not_resampled_at_target": True,
        "declared_lag_steps": 1,
        "nominal_dt_sec": dt,
        "n_valid_centers": int(n_valid_gain),
    }
    provenance = build_provenance(
        coordinate_layer=layer,
        coordinate_names=names,
        time_semantics="within_segment_same_pair_lag1_neighbor_gain",
        estimator=ESTIMATOR_NAME,
        settings=shared_settings,
    )
    gain_summary = {
        **support_summary,
        **common_flags,
        "neighbor_gain_q90_median": float(np.median(finite_gains)),
        "not_operator_max_gain": True,
        "aggregation": "per_source_q90_not_pooled_pair_q90",
    }
    gain_series = {
        **support_series,
        "neighbor_gain_q90": gain_q90,
        "n_neighbor_pairs": n_pairs,
    }
    result = unavailable_result(
        AMPLIFICATION_SCHEMA_VERSION,
        status="not_testable",
        failure_reason="not_requested",
        coordinate_layer=layer,
        coordinate_names=names,
    )
    result["computation_status"] = "computed"
    result["failure_reason"] = None
    result["series"] = gain_series
    result["summary"] = gain_summary
    result["provenance"] = provenance
    leaf = stamp_register_fields(result, MEASUREMENT_ID_NEIGHBOR_GAIN)
    leaf = _stamp_qualification(leaf, QUALIFICATION_SAME_PAIR_GAIN)
    leaf = _finalize(leaf)

    finite_sep = sep_rate[np.isfinite(sep_rate)]
    sep_ok = n_valid_sep > 0
    sep_leaf = _identity_leaf(
        measurement_id=MEASUREMENT_ID_NEIGHBOR_SEPARATION,
        qualification=QUALIFICATION_SAME_PAIR_SEPARATION,
        identified=sep_ok,
        failure_reason=None if sep_ok else REASON_INSUFFICIENT,
        series={**support_series, "neighbor_separation_rate": sep_rate, "n_neighbor_pairs": n_pairs},
        summary={
            **support_summary,
            **common_flags,
            "neighbor_separation_rate_median": float(np.median(finite_sep)) if sep_ok else float("nan"),
            "n_valid_centers": int(n_valid_sep),
            "not_spectral_abscissa": True,
        },
        provenance=provenance,
    )
    finite_rate = gain_rate[np.isfinite(gain_rate)]
    rate_ok = int(finite_rate.size) > 0
    rate_leaf = _identity_leaf(
        measurement_id=MEASUREMENT_ID_NEIGHBOR_GAIN_RATE,
        qualification=QUALIFICATION_SAME_PAIR_GAIN_RATE,
        identified=rate_ok,
        failure_reason=None if rate_ok else REASON_INSUFFICIENT,
        series={**support_series, "neighbor_gain_rate_q90": gain_rate, "n_neighbor_pairs": n_pairs},
        summary={
            **support_summary,
            **common_flags,
            "neighbor_gain_rate_q90_median": float(np.median(finite_rate)) if rate_ok else float("nan"),
            "n_valid_centers": int(finite_rate.size),
            "not_operator_max_gain": True,
            "not_spectral_abscissa": True,
            "not_peak_gain_level4": True,
        },
        provenance=provenance,
    )
    finite_cloud = cloud_rate[np.isfinite(cloud_rate)]
    cloud_ok = n_valid_cloud > 0
    cloud_leaf = _identity_leaf(
        measurement_id=MEASUREMENT_ID_CLOUD_VOLUME,
        qualification=QUALIFICATION_CLOUD_VOLUME,
        identified=cloud_ok,
        failure_reason=None if cloud_ok else REASON_INSUFFICIENT,
        series={**support_series, "cloud_volume_change_rate": cloud_rate, "n_neighbor_pairs": n_pairs},
        summary={
            **support_summary,
            **common_flags,
            "cloud_volume_change_rate_median": float(np.median(finite_cloud)) if cloud_ok else float("nan"),
            "n_valid_centers": int(n_valid_cloud),
            "not_operator_volume": True,
            "epsilon_logdet_regularized": True,
        },
        provenance=provenance,
    )
    parent = dict(leaf)
    parent[MEASUREMENT_ID_NEIGHBOR_GAIN] = leaf
    parent[MEASUREMENT_ID_NEIGHBOR_SEPARATION] = sep_leaf
    parent[MEASUREMENT_ID_NEIGHBOR_GAIN_RATE] = rate_leaf
    parent[MEASUREMENT_ID_CLOUD_VOLUME] = cloud_leaf
    return _finalize(parent)
