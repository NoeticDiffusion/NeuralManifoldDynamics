"""Frozen M0/M1 history predictive gain and nested M1 operator (001).

``history_predictive_gain_level1`` is the out-of-sample mean-squared-error
reduction ``err(M0)-err(M1)`` on the same lag-1 triples:

* ``M0``: affine ``x_t -> x_{t+1}``
* ``M1``: affine ``(x_t, x_{t-1}) -> x_{t+1}``

``history_conditioned_operator_level2`` is nested under the same family.
It is identified only when M1 itself passes the frozen one-step OOS gate
(``mndm.one_step_fit_fidelity.v1``: median of two chronological blocked
holdout rel-MSE scores strictly less than 0.9 versus a mean-next-state
baseline). Positive ``H_gain`` is not identification. The map is 3×6, not
lag-1 ``Φ``. Not Markov restoration. Generator and propagator rungs remain
withheld.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import HISTORY_SCHEMA_VERSION, build_provenance, unavailable_result
from .measurement_register import (
    MEASUREMENT_ID_HISTORY_OPERATOR,
    MEASUREMENT_ID_HISTORY_PREDICTIVE_GAIN,
    QUALIFICATION_HISTORY_GAIN,
    QUALIFICATION_HISTORY_OPERATOR_IDENTIFIED,
    QUALIFICATION_HISTORY_OPERATOR_NOT_IDENTIFIED,
    QUALIFICATION_NOT_ASSESSED,
    stamp_register_fields,
)
from .one_step_operator import (
    DEFAULT_REL_MSE_THRESHOLD,
    ONE_STEP_FIT_GATE_VERSION,
    _MIN_HOLDOUT,
    _fit_affine_map,
    _temporal_block_folds,
)
from .transition_support import (
    embargo_claim_fields,
    build_transition_support,
    support_series_fields,
    support_settings_fields,
    support_summary_fields,
)
from .validity import validate_trajectory
from ..inferential_grain import attach_grain_for_schema
from ..jacobian import _affine_rel_mse
from ..measurement_certificate import attach_certificate

ESTIMATOR_NAME = "frozen_affine_m0_m1_oos"
DEFAULT_N_BLOCKS = 2
DEFAULT_EMBARGO_STEPS = 4
DEFAULT_RIDGE = 1e-4
REASON_SUBJECT_ANCHORED_3D = "history_subject_anchored_3d_only"
REASON_N_BLOCKS = "history_n_blocks_not_implemented"
REASON_EMBARGO = "history_embargo_steps_not_implemented"
REASON_THRESHOLD = "history_rel_mse_threshold_not_implemented"
REASON_TRIPLES = "insufficient_history_triples"
REASON_FOLDS = "history_both_folds_required"
REASON_INSUFFICIENT = "insufficient_local_support"
REASON_M1_BASELINE = "history_m1_fit_not_better_than_baseline"
REASON_M1_HOLDOUT = "history_m1_holdout_below_one_step_min"
_SUBJECT_ANCHORED_LAYERS = frozenset({"coords_3d_subject_anchored", "subject_anchored"})
_M1_INPUT_DIM = 6
_M1_STATE_DIM = 3


def _finalize(result: Mapping[str, Any]) -> dict[str, Any]:
    return attach_grain_for_schema(attach_certificate(result), HISTORY_SCHEMA_VERSION)


def _stamp_qualification(result: dict[str, Any], token: str) -> dict[str, Any]:
    out = dict(result)
    out["qualification_status"] = token
    summary = dict(out.get("summary") or {})
    summary["qualification_status"] = token
    out["summary"] = summary
    return out


def _nested_leaf(
    *,
    measurement_id: str,
    qualification: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    leaf = stamp_register_fields(dict(result), measurement_id)
    leaf = _stamp_qualification(leaf, qualification)
    return _finalize(leaf)


def _operator_unavailable_leaf(result: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(result)
    summary = dict(payload.get("summary") or {})
    summary.pop("not_one_step_operator", None)
    summary["history_m1_not_lag1_phi"] = True
    payload["summary"] = summary
    provenance = dict(payload.get("provenance") or {})
    settings = dict(provenance.get("settings") or {})
    settings.pop("not_one_step_operator", None)
    settings["history_m1_not_lag1_phi"] = True
    settings["not_history_augmented_generator"] = True
    provenance["settings"] = settings
    payload["provenance"] = provenance
    return _nested_leaf(
        measurement_id=MEASUREMENT_ID_HISTORY_OPERATOR,
        qualification=QUALIFICATION_NOT_ASSESSED,
        result=payload,
    )


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
        HISTORY_SCHEMA_VERSION,
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
    settings["not_markov_restoration"] = True
    settings["not_one_step_operator"] = True
    settings["history_m1_not_lag1_phi"] = True
    settings["one_step_fit_gate_version"] = ONE_STEP_FIT_GATE_VERSION
    settings["one_step_rel_mse_threshold"] = float(DEFAULT_REL_MSE_THRESHOLD)
    if extra_settings:
        settings.update(dict(extra_settings))
    provenance["settings"] = settings
    provenance["estimator"] = ESTIMATOR_NAME
    result["provenance"] = provenance
    parent = stamp_register_fields(result, MEASUREMENT_ID_HISTORY_PREDICTIVE_GAIN)
    parent = _stamp_qualification(parent, QUALIFICATION_HISTORY_GAIN)
    parent = _finalize(parent)
    gain_leaf = dict(parent)
    parent[MEASUREMENT_ID_HISTORY_PREDICTIVE_GAIN] = gain_leaf
    parent[MEASUREMENT_ID_HISTORY_OPERATOR] = _operator_unavailable_leaf(gain_leaf)
    return parent


def unavailable_history_family(
    *,
    status: str,
    failure_reason: str,
    coordinate_layer: str,
    coordinate_names: list[str],
    extra_summary: Mapping[str, Any] | None = None,
    extra_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Schema-complete unavailable history payload with L1 and L2 leaves."""
    return _unavailable(
        status=status,
        failure_reason=failure_reason,
        coordinate_layer=coordinate_layer,
        coordinate_names=coordinate_names,
        extra_summary=extra_summary,
        extra_settings=extra_settings,
    )


def _predict(phi: np.ndarray, intercept: np.ndarray, x_ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    return (x - x_ref.reshape(1, -1)) @ phi.T + intercept.reshape(1, -1)


def _operator_leaf(
    *,
    identified: bool,
    failure_reason: str | None,
    series: Mapping[str, Any],
    summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    status = "computed" if identified else "insufficient_support"
    result = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "computation_status": status,
        "failure_reason": failure_reason,
        "series": dict(series),
        "summary": dict(summary),
        "provenance": dict(provenance),
    }
    qualification = (
        QUALIFICATION_HISTORY_OPERATOR_IDENTIFIED
        if identified
        else QUALIFICATION_HISTORY_OPERATOR_NOT_IDENTIFIED
    )
    return _nested_leaf(
        measurement_id=MEASUREMENT_ID_HISTORY_OPERATOR,
        qualification=qualification,
        result=result,
    )


def estimate_history_predictive_gain(
    state: np.ndarray,
    time: np.ndarray,
    *,
    segment_id: np.ndarray | None = None,
    coordinate_layer: str = "coords_3d_subject_anchored",
    coordinate_names: Sequence[str] | None = None,
    min_samples: int = 30,
    min_triples: int = 20,
    max_gap_sec: float | None = None,
    max_dt_relative_deviation: float = 0.05,
    ridge_alpha: float = DEFAULT_RIDGE,
    n_blocks: int = DEFAULT_N_BLOCKS,
    embargo_steps: int = DEFAULT_EMBARGO_STEPS,
    one_step_rel_mse_threshold: float = DEFAULT_REL_MSE_THRESHOLD,
) -> dict[str, Any]:
    """Recording-level OOS MSE reduction from one lag of history, plus nested M1 OOS."""
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
    if int(n_blocks) != DEFAULT_N_BLOCKS:
        return _unavailable(
            status="not_testable",
            failure_reason=REASON_N_BLOCKS,
            coordinate_layer=layer,
            coordinate_names=names,
            extra_summary={"requested_n_blocks": int(n_blocks)},
        )
    if int(embargo_steps) != DEFAULT_EMBARGO_STEPS:
        return _unavailable(
            status="not_testable",
            failure_reason=REASON_EMBARGO,
            coordinate_layer=layer,
            coordinate_names=names,
            extra_summary={"requested_embargo_steps": int(embargo_steps)},
        )
    if float(one_step_rel_mse_threshold) != float(DEFAULT_REL_MSE_THRESHOLD):
        return _unavailable(
            status="not_testable",
            failure_reason=REASON_THRESHOLD,
            coordinate_layer=layer,
            coordinate_names=names,
            extra_summary={"requested_one_step_rel_mse_threshold": float(one_step_rel_mse_threshold)},
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
        "ridge_alpha": float(ridge_alpha),
        "n_blocks": DEFAULT_N_BLOCKS,
        "embargo_steps": DEFAULT_EMBARGO_STEPS,
        **embargo_claim_fields(),
        "min_triples": int(min_triples),
        "m0": "affine_x_t_to_x_t_plus_1",
        "m1": "affine_x_t_x_t_minus_1_to_x_t_plus_1",
        "error": "mean_squared_euclidean_next_state",
        "not_markov_restoration": True,
        "not_one_step_operator": True,
        "history_m1_not_lag1_phi": True,
        "one_step_fit_gate_version": ONE_STEP_FIT_GATE_VERSION,
        "one_step_rel_mse_threshold": float(DEFAULT_REL_MSE_THRESHOLD),
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
    source_set = set(int(value) for value in source_idx.tolist())
    keep = np.array([int(src) - 1 in source_set for src in source_idx], dtype=bool)
    if int(np.sum(keep)) < int(min_triples):
        return _unavailable(
            status="insufficient_support",
            failure_reason=REASON_TRIPLES,
            coordinate_layer=layer,
            coordinate_names=names,
            extra_settings=shared_settings,
            extra_summary={"n_history_triples": int(np.sum(keep))},
        )
    triple_support = support.subset(keep)
    triple_source = np.asarray(triple_support.source_idx, dtype=np.int32)
    shared_settings = {
        **shared_settings,
        **support_settings_fields(triple_support),
        "lag1_transition_support_id": support.support_id,
        "n_lag1_pairs": int(source_idx.size),
        "n_history_triples": int(triple_source.size),
    }
    x_t = x[triple_source]
    x_tm1 = x[triple_source - 1]
    x_tp1 = x_t + np.asarray(triple_support.increments, dtype=float)
    x_m1 = np.concatenate([x_t, x_tm1], axis=1)
    weights = np.ones(int(triple_source.size), dtype=np.float32)
    folds = _temporal_block_folds(triple_source, DEFAULT_EMBARGO_STEPS)
    n_time = int(x.shape[0])
    local_gain = np.full(n_time, np.nan, dtype=np.float32)
    sq_m0 = np.full(n_time, np.nan, dtype=np.float32)
    sq_m1 = np.full(n_time, np.nan, dtype=np.float32)
    holdout_m0: list[np.ndarray] = []
    holdout_m1: list[np.ndarray] = []
    fold_rel: list[float] = []
    n_holdout = 0
    n_valid_folds = 0
    n_valid_m1_folds = 0
    min_hold = max(_MIN_HOLDOUT, _M1_INPUT_DIM + 1)
    for train_idx, test_idx in folds:
        if train_idx.size < 8 or test_idx.size < 1:
            continue
        fitted_m0 = _fit_affine_map(x_t[train_idx], x_tp1[train_idx], weights[train_idx], ridge_alpha)
        fitted_m1 = _fit_affine_map(x_m1[train_idx], x_tp1[train_idx], weights[train_idx], ridge_alpha)
        if fitted_m0 is None or fitted_m1 is None:
            continue
        pred0 = _predict(*fitted_m0, x_t[test_idx])
        pred1 = _predict(*fitted_m1, x_m1[test_idx])
        err0 = np.sum((pred0 - x_tp1[test_idx]) ** 2, axis=1)
        err1 = np.sum((pred1 - x_tp1[test_idx]) ** 2, axis=1)
        holdout_m0.append(err0)
        holdout_m1.append(err1)
        n_holdout += int(test_idx.size)
        n_valid_folds += 1
        times = triple_source[test_idx]
        sq_m0[times] = err0.astype(np.float32)
        sq_m1[times] = err1.astype(np.float32)
        local_gain[times] = (err0 - err1).astype(np.float32)
        if int(test_idx.size) >= min_hold:
            phi_oos, intercept_oos, xref_oos = fitted_m1
            rel_oos = _affine_rel_mse(
                x_m1[test_idx].astype(np.float32),
                x_tp1[test_idx].astype(np.float32),
                phi_oos,
                intercept_oos,
                xref_oos,
            )
            if np.isfinite(rel_oos):
                fold_rel.append(float(rel_oos))
                n_valid_m1_folds += 1
    fold_summary = {
        "n_lag1_pairs": int(source_idx.size),
        "n_history_triples": int(triple_source.size),
        "n_oos_holdout": n_holdout,
        "n_valid_folds": n_valid_folds,
        "n_valid_m1_folds": n_valid_m1_folds,
    }
    if n_valid_folds != DEFAULT_N_BLOCKS:
        return _unavailable(
            status="insufficient_support",
            failure_reason=REASON_FOLDS,
            coordinate_layer=layer,
            coordinate_names=names,
            extra_settings=shared_settings,
            extra_summary=fold_summary,
        )
    if n_holdout < int(min_triples) // 2 or not holdout_m0:
        return _unavailable(
            status="insufficient_support",
            failure_reason=REASON_INSUFFICIENT,
            coordinate_layer=layer,
            coordinate_names=names,
            extra_settings=shared_settings,
            extra_summary=fold_summary,
        )
    mse_m0 = float(np.mean(np.concatenate(holdout_m0)))
    mse_m1 = float(np.mean(np.concatenate(holdout_m1)))
    gain = float(mse_m0 - mse_m1)
    summary = {
        **support_summary_fields(triple_support),
        **fold_summary,
        "lag1_transition_support_id": support.support_id,
        "mse_m0": mse_m0,
        "mse_m1": mse_m1,
        "history_predictive_gain": gain,
        "not_markov_restoration": True,
        "not_one_step_operator": True,
        "history_m1_not_lag1_phi": True,
        "declared_lag_steps": 1,
        "history_lags": 1,
    }
    series = {
        **support_series_fields(triple_support),
        "history_predictive_gain": local_gain,
        "mse_m0": sq_m0,
        "mse_m1": sq_m1,
    }
    result = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "computation_status": "computed",
        "failure_reason": None,
        "series": series,
        "summary": summary,
        "provenance": build_provenance(
            coordinate_layer=layer,
            coordinate_names=names,
            time_semantics="within_segment_frozen_m0_m1_oos",
            estimator=ESTIMATOR_NAME,
            settings=shared_settings,
        ),
    }
    leaf = stamp_register_fields(result, MEASUREMENT_ID_HISTORY_PREDICTIVE_GAIN)
    leaf = _stamp_qualification(leaf, QUALIFICATION_HISTORY_GAIN)
    leaf = _finalize(leaf)

    phi_hat = np.full((n_time, _M1_STATE_DIM, _M1_INPUT_DIM), np.nan, dtype=np.float32)
    affine_reference = np.full((n_time, _M1_INPUT_DIM), np.nan, dtype=np.float32)
    affine_intercept = np.full((n_time, _M1_STATE_DIM), np.nan, dtype=np.float32)
    rel_mse = np.full(n_time, np.nan, dtype=np.float32)
    median_rel = float("nan")
    identified = False
    operator_reason: str | None = REASON_M1_HOLDOUT
    n_fitted = 0
    if n_valid_m1_folds == DEFAULT_N_BLOCKS and len(fold_rel) == DEFAULT_N_BLOCKS:
        median_rel = float(np.median(np.asarray(fold_rel, dtype=float)))
        identified = bool(median_rel < float(DEFAULT_REL_MSE_THRESHOLD))
        operator_reason = None if identified else REASON_M1_BASELINE
        if identified:
            fitted_all = _fit_affine_map(x_m1, x_tp1, weights, ridge_alpha)
            if fitted_all is None:
                identified = False
                operator_reason = REASON_INSUFFICIENT
            else:
                phi, intercept, x_ref = fitted_all
                for center in triple_source:
                    t_idx = int(center)
                    phi_hat[t_idx] = phi
                    affine_reference[t_idx] = x_ref
                    affine_intercept[t_idx] = intercept
                    rel_mse[t_idx] = np.float32(median_rel)
                    n_fitted += 1
    operator_summary = {
        **support_summary_fields(triple_support),
        **fold_summary,
        "lag1_transition_support_id": support.support_id,
        "rel_mse_baseline_median": median_rel,
        "rel_mse_baseline_folds": np.asarray(fold_rel, dtype=np.float32),
        "rel_mse_scoring": "blocked_holdout",
        "one_step_identified": identified,
        "one_step_fit_gate_version": ONE_STEP_FIT_GATE_VERSION,
        "one_step_rel_mse_threshold": float(DEFAULT_REL_MSE_THRESHOLD),
        "n_fitted_windows": int(n_fitted),
        "phi_shape": [_M1_STATE_DIM, _M1_INPUT_DIM],
        "input_dim": _M1_INPUT_DIM,
        "state_dim": _M1_STATE_DIM,
        "not_markov_restoration": True,
        "history_m1_not_lag1_phi": True,
        "not_history_augmented_generator": True,
        "declared_lag_steps": 1,
        "history_lags": 1,
    }
    operator_series = {
        **support_series_fields(triple_support),
        "phi_hat": phi_hat,
        "affine_reference": affine_reference,
        "affine_intercept": affine_intercept,
        "rel_mse_baseline": rel_mse,
    }
    operator_leaf = _operator_leaf(
        identified=identified,
        failure_reason=operator_reason,
        series=operator_series,
        summary=operator_summary,
        provenance=build_provenance(
            coordinate_layer=layer,
            coordinate_names=names,
            time_semantics="within_segment_frozen_m1_one_step_oos",
            estimator=ESTIMATOR_NAME,
            settings={
                **shared_settings,
                "not_one_step_operator": False,
                "history_m1_not_lag1_phi": True,
                "not_history_augmented_generator": True,
            },
        ),
    )
    parent = dict(leaf)
    parent[MEASUREMENT_ID_HISTORY_PREDICTIVE_GAIN] = leaf
    parent[MEASUREMENT_ID_HISTORY_OPERATOR] = operator_leaf
    return _finalize(parent)
