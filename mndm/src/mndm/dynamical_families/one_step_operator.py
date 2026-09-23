"""Generic qualified affine one-step operator (SL-LEV-MES-003 level 2–4).

Fits a recording-level lag-1 map ``x_{t+Δ} ≈ Φ(x_t - x̄) + c`` and, when
requested, a **direct** lag-2 map on pairs ``(x_t, x_{t+2Δ})``. Lag 2 is not
``Φ_1^2``. Composing the lag-1 map twice is level 4
(``iterated_one_step_horizon_map_level4``) with its own horizon holdout.
This is not the production ẋ-Jacobian, not ``expm(J_hat dt)``, and not
Itô ``b``. Level-2 Euclidean functionals of a qualified ``Φ`` are
``log(σ_max(Φ))/dt``, ``log|det Φ|/dt``, and polar rotation
``||log R||_F/(√2 dt)``. They are not spectral abscissa, not peak gain,
and not generator-proxy rotation. Rank-deficient volume is not
epsilon-rescued. Level-3 generator proxies are ``logm(Φ)/nominal_dt`` from a
qualified map at the same declared lag. At lag 2, ``nominal_dt`` is the
median lag-2 span (≈ ``2Δt``). Nested under ``declared_lag_2/``. Not
``ito_drift_level3``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy.linalg import logm, polar

from .contracts import AFFINE_ONE_STEP_SCHEMA_VERSION, build_provenance, unavailable_result
from .measurement_register import (
    ALLOWED_ONE_STEP_DECLARED_LAGS,
    DEFAULT_ONE_STEP_DECLARED_LAGS,
    MEASUREMENT_ID_AFFINE_MAP,
    MEASUREMENT_ID_AFFINE_MEAN_RATE,
    MEASUREMENT_ID_DIVERGENCE,
    MEASUREMENT_ID_GENERATOR_ROTATION,
    MEASUREMENT_ID_INNOVATION_COVARIANCE,
    MEASUREMENT_ID_ITERATED_ONE_STEP_MAP,
    MEASUREMENT_ID_NUMERICAL_ABSCISSA,
    MEASUREMENT_ID_OPERATOR_MAX_GAIN,
    MEASUREMENT_ID_OPERATOR_ROTATION_RATE,
    MEASUREMENT_ID_OPERATOR_VOLUME_GAIN,
    MEASUREMENT_ID_SPECTRAL_ABSCISSA,
    ONE_STEP_LAG2_DIVERGENCE,
    ONE_STEP_LAG2_INNOVATION,
    ONE_STEP_LAG2_MAP,
    ONE_STEP_LAG2_MAX_GAIN,
    ONE_STEP_LAG2_MEAN_RATE,
    ONE_STEP_LAG2_NUMERICAL,
    ONE_STEP_LAG2_ROTATION,
    ONE_STEP_LAG2_ROTATION_RATE,
    ONE_STEP_LAG2_SPECTRAL,
    ONE_STEP_LAG2_VOLUME_GAIN,
    QUALIFICATION_GENERATOR_PROXY_NOT_ITO,
    QUALIFICATION_HORIZON_PROPAGATION_IDENTIFIED,
    QUALIFICATION_HORIZON_PROPAGATION_NOT_IDENTIFIED,
    QUALIFICATION_ONE_STEP_FUNCTIONAL,
    QUALIFICATION_ONE_STEP_IDENTIFIED,
    QUALIFICATION_ONE_STEP_NOT_IDENTIFIED,
    VARIANT_DECLARED_LAG_2,
    VARIANT_DECLARED_LAG_STEPS_1,
    VARIANT_DECLARED_LAG_STEPS_2,
    VARIANT_HORIZON_STEPS_2,
    stamp_register_fields,
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
from ..jacobian import _affine_rel_mse, _fit_ridge
from ..measurement_certificate import attach_certificate

ESTIMATOR_NAME = "recording_affine_discrete_map"
ONE_STEP_FIT_GATE_VERSION = "mndm.one_step_fit_fidelity.v1"
DEFAULT_REL_MSE_THRESHOLD = 0.9
REASON_SUBJECT_ANCHORED_3D = "one_step_subject_anchored_3d_only"
REASON_INSUFFICIENT_LOCAL = "insufficient_local_support"
REASON_NOT_BETTER_THAN_BASELINE = "one_step_fit_not_better_than_baseline"
REASON_UPSTREAM = "upstream_one_step_not_identified"
REASON_LOGM = "generator_matrix_log_not_real"
REASON_EMBARGO = "one_step_embargo_shorter_than_declared_lag"
REASON_HORIZON = "horizon_fit_not_better_than_baseline"
REASON_RANK = "operator_phi_rank_deficient"
REASON_REFLECTION = "operator_polar_reflection_not_rotation"
REASON_POLAR_LOG = "operator_polar_rotation_log_not_real"
HORIZON_STEPS = 2
_SUBJECT_ANCHORED_LAYERS = frozenset({"coords_3d_subject_anchored", "subject_anchored"})
_LOGM_IMAG_TOL = 1e-8
_RANK_REL_TOL = 1e-8
_MIN_HOLDOUT = 8
DEFAULT_N_BLOCKS = 2
DEFAULT_EMBARGO_STEPS = 4


def _finalize(result: Mapping[str, Any]) -> dict[str, Any]:
    return attach_grain_for_schema(attach_certificate(result), AFFINE_ONE_STEP_SCHEMA_VERSION)


def _unavailable(
    *,
    status: str,
    failure_reason: str,
    coordinate_layer: str,
    coordinate_names: list[str],
    measurement_id: str | None = None,
    extra_summary: Mapping[str, Any] | None = None,
    extra_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = unavailable_result(
        AFFINE_ONE_STEP_SCHEMA_VERSION,
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
    settings["not_sde_drift"] = True
    settings["not_production_jacobian"] = True
    settings["estimator"] = ESTIMATOR_NAME
    if extra_settings:
        settings.update(dict(extra_settings))
    provenance["settings"] = settings
    provenance["estimator"] = ESTIMATOR_NAME
    result["provenance"] = provenance
    if measurement_id is not None:
        result = stamp_register_fields(result, measurement_id)
    return _finalize(result)


_FUNCTIONAL_PROVENANCE = {
    "operator_polar_side": "right",
    "operator_rank_rel_tol": _RANK_REL_TOL,
    "operator_logm_imag_tol": _LOGM_IMAG_TOL,
}


def _attach_refused_functionals(
    parent: Mapping[str, Any],
    *,
    coordinate_layer: str,
    coordinate_names: list[str],
    register_max_gain_key: str,
    register_volume_gain_key: str,
    register_rotation_rate_key: str,
    variant_id: str | None,
) -> dict[str, Any]:
    """Write L2 functional leaves on early refusal. Never substitute zeros."""
    out = dict(parent)
    if MEASUREMENT_ID_OPERATOR_MAX_GAIN in out:
        return _finalize(out)
    reason = str(out.get("failure_reason") or "not_requested")
    extra_summary = dict(out.get("summary") or {})
    extra_settings = {
        **_FUNCTIONAL_PROVENANCE,
        **dict((out.get("provenance") or {}).get("settings") or {}),
    }
    for measurement_id, register_key in (
        (MEASUREMENT_ID_OPERATOR_MAX_GAIN, register_max_gain_key),
        (MEASUREMENT_ID_OPERATOR_VOLUME_GAIN, register_volume_gain_key),
        (MEASUREMENT_ID_OPERATOR_ROTATION_RATE, register_rotation_rate_key),
    ):
        out[measurement_id] = _leaf(
            measurement_id=measurement_id,
            status="not_testable",
            failure_reason=reason,
            coordinate_layer=coordinate_layer,
            coordinate_names=coordinate_names,
            extra_summary=extra_summary,
            extra_settings=extra_settings,
            qualification_override=QUALIFICATION_ONE_STEP_FUNCTIONAL,
            register_key=register_key,
            variant_id=variant_id,
        )
    return _finalize(out)


def _layer_refusal(
    *,
    state: np.ndarray,
    coordinate_layer: str,
    coordinate_names: list[str] | None,
) -> dict[str, Any] | None:
    names = list(coordinate_names or ["m", "d", "e"])
    x = np.asarray(state)
    layer = str(coordinate_layer or "").strip()
    if layer not in _SUBJECT_ANCHORED_LAYERS:
        return _unavailable(
            status="not_testable",
            failure_reason=REASON_SUBJECT_ANCHORED_3D,
            coordinate_layer=layer or "unknown",
            coordinate_names=names,
        )
    if x.ndim != 2 or x.shape[1] != 3:
        return _unavailable(
            status="not_testable",
            failure_reason=REASON_SUBJECT_ANCHORED_3D,
            coordinate_layer=layer,
            coordinate_names=names,
        )
    return None


def _fit_affine_map(
    x_src: np.ndarray,
    x_tgt: np.ndarray,
    weights: np.ndarray,
    ridge_alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if x_src.shape[0] < x_src.shape[1] + 1:
        return None
    x_mean = np.mean(x_src, axis=0)
    design = x_src - x_mean.reshape(1, -1)
    col_scale = np.std(design, axis=0, ddof=0)
    col_scale = np.where(np.isfinite(col_scale) & (col_scale > 1e-8), col_scale, 1.0)
    design_std = design / col_scale.reshape(1, -1)
    design_aug = np.hstack([design_std, np.ones((design.shape[0], 1), dtype=float)])
    w = np.asarray(weights, dtype=np.float32)
    if w.size != x_src.shape[0] or not np.all(np.isfinite(w)):
        return None
    phi_std, intercept = _fit_ridge(
        design_aug.astype(np.float32),
        np.asarray(x_tgt, dtype=np.float32),
        float(ridge_alpha),
        sample_weights=w,
    )
    phi = (np.asarray(phi_std, dtype=np.float32) / col_scale.reshape(1, -1).astype(np.float32))
    if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(intercept)):
        return None
    return phi.astype(np.float32), intercept.astype(np.float32), x_mean.astype(np.float32)


def _generator_from_phi(phi: np.ndarray, dt_sec: float) -> np.ndarray | None:
    if not np.isfinite(dt_sec) or float(dt_sec) <= 0.0 or not np.all(np.isfinite(phi)):
        return None
    try:
        logged = np.asarray(logm(np.asarray(phi, dtype=float)), dtype=complex)
    except (ValueError, np.linalg.LinAlgError):
        return None
    if float(np.max(np.abs(np.imag(logged)))) > _LOGM_IMAG_TOL:
        return None
    generator = np.real(logged) / float(dt_sec)
    if not np.all(np.isfinite(generator)):
        return None
    return generator.astype(np.float32)


def _operator_functionals_from_phi(
    phi: np.ndarray, dt_sec: float
) -> dict[str, tuple[float | None, str | None]]:
    """Euclidean SVD / det / polar functionals of Φ. Not generator proxies."""
    out: dict[str, tuple[float | None, str | None]] = {
        "max_gain": (None, REASON_RANK),
        "volume_gain": (None, REASON_RANK),
        "rotation_rate": (None, REASON_RANK),
    }
    if not np.isfinite(dt_sec) or float(dt_sec) <= 0.0 or not np.all(np.isfinite(phi)):
        return out
    phi_m = np.asarray(phi, dtype=float)
    try:
        singular = np.linalg.svd(phi_m, compute_uv=False)
    except np.linalg.LinAlgError:
        return out
    if singular.size == 0 or not np.all(np.isfinite(singular)):
        return out
    smax = float(singular[0])
    smin = float(singular[-1])
    rank_deficient = (smax <= 0.0) or (smin / max(smax, 1e-30) < _RANK_REL_TOL)
    if smax > 0.0:
        out["max_gain"] = (float(np.log(smax) / float(dt_sec)), None)
    if rank_deficient:
        out["volume_gain"] = (None, REASON_RANK)
        out["rotation_rate"] = (None, REASON_RANK)
        return out
    sign, logabs = np.linalg.slogdet(phi_m)
    if int(sign) == 0 or not np.isfinite(logabs):
        out["volume_gain"] = (None, REASON_RANK)
    else:
        out["volume_gain"] = (float(logabs / float(dt_sec)), None)
    try:
        rotation, _stretch = polar(phi_m, side="right")
    except (ValueError, np.linalg.LinAlgError):
        out["rotation_rate"] = (None, REASON_POLAR_LOG)
        return out
    det_r = float(np.linalg.det(np.asarray(rotation, dtype=float)))
    if not np.isfinite(det_r) or det_r <= 0.0:
        out["rotation_rate"] = (None, REASON_REFLECTION)
        return out
    try:
        logged = np.asarray(logm(np.asarray(rotation, dtype=float)), dtype=complex)
    except (ValueError, np.linalg.LinAlgError):
        out["rotation_rate"] = (None, REASON_POLAR_LOG)
        return out
    if float(np.max(np.abs(np.imag(logged)))) > _LOGM_IMAG_TOL:
        out["rotation_rate"] = (None, REASON_POLAR_LOG)
        return out
    real_log = np.real(logged)
    if not np.all(np.isfinite(real_log)):
        out["rotation_rate"] = (None, REASON_POLAR_LOG)
        return out
    rate = float(np.linalg.norm(real_log) / (np.sqrt(2.0) * float(dt_sec)))
    out["rotation_rate"] = (rate, None)
    return out


def _temporal_block_folds(
    source_idx: np.ndarray, embargo_steps: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Two chronological folds with an index embargo around the median split."""
    split_index = int(np.median(np.asarray(source_idx, dtype=np.int32)))
    embargo = int(embargo_steps)
    pair = np.arange(int(source_idx.size), dtype=np.int32)
    fold1 = pair[source_idx < (split_index - embargo)]
    fold2 = pair[source_idx > (split_index + embargo)]
    return [(fold1, fold2), (fold2, fold1)]


def _stamp_qualification(result: dict[str, Any], token: str) -> dict[str, Any]:
    out = dict(result)
    out["qualification_status"] = token
    summary = dict(out.get("summary") or {})
    summary["qualification_status"] = token
    out["summary"] = summary
    return out


def _leaf(
    *,
    measurement_id: str,
    status: str,
    failure_reason: str | None,
    coordinate_layer: str,
    coordinate_names: list[str],
    series: Mapping[str, Any] | None = None,
    extra_summary: Mapping[str, Any] | None = None,
    extra_settings: Mapping[str, Any] | None = None,
    qualification_override: str | None = None,
    register_key: str | None = None,
    variant_id: str | None = None,
) -> dict[str, Any]:
    result = unavailable_result(
        AFFINE_ONE_STEP_SCHEMA_VERSION,
        status=status if status != "computed" else "not_testable",
        failure_reason=failure_reason or "not_requested",
        coordinate_layer=coordinate_layer,
        coordinate_names=coordinate_names,
    )
    if status == "computed":
        result["computation_status"] = "computed"
        result["failure_reason"] = None
    result["series"] = dict(series or {})
    summary = dict(result.get("summary") or {})
    if extra_summary:
        summary.update(dict(extra_summary))
    result["summary"] = summary
    provenance = dict(result.get("provenance") or {})
    settings = dict(provenance.get("settings") or {})
    settings["not_sde_drift"] = True
    settings["not_production_jacobian"] = True
    settings["never_auto_ito_qualified"] = True
    settings["estimator"] = ESTIMATOR_NAME
    if extra_settings:
        settings.update(dict(extra_settings))
    provenance["settings"] = settings
    provenance["estimator"] = ESTIMATOR_NAME
    result["provenance"] = provenance
    result = stamp_register_fields(
        result,
        register_key or measurement_id,
        variant_id=variant_id,
    )
    if qualification_override is not None:
        result = _stamp_qualification(result, qualification_override)
    return _finalize(result)


def _recording_affine_at_lag(
    state: np.ndarray,
    time: np.ndarray,
    *,
    lag: int,
    segment_id: np.ndarray | None = None,
    coordinate_layer: str = "coords_3d_subject_anchored",
    coordinate_names: list[str] | None = None,
    neighborhood_k: int = 20,
    min_samples: int = 30,
    min_neighborhood_samples: int = 10,
    max_gap_sec: float | None = None,
    max_dt_relative_deviation: float = 0.05,
    ridge_alpha: float = 1e-4,
    one_step_rel_mse_threshold: float = DEFAULT_REL_MSE_THRESHOLD,
    n_blocks: int = DEFAULT_N_BLOCKS,
    embargo_steps: int = DEFAULT_EMBARGO_STEPS,
    epsilon: float = 1e-12,
    write_level3: bool = True,
    register_map_key: str = MEASUREMENT_ID_AFFINE_MAP,
    register_rate_key: str = MEASUREMENT_ID_AFFINE_MEAN_RATE,
    register_innov_key: str = MEASUREMENT_ID_INNOVATION_COVARIANCE,
    variant_id: str | None = VARIANT_DECLARED_LAG_STEPS_1,
) -> dict[str, Any]:
    """Fit one recording-level affine map at a declared lag. Does not compose Φ."""
    names = list(coordinate_names or ["m", "d", "e"])
    lag = int(lag)
    write_level3 = bool(write_level3) and lag in (1, 2)
    register_spectral_key = MEASUREMENT_ID_SPECTRAL_ABSCISSA
    register_numerical_key = MEASUREMENT_ID_NUMERICAL_ABSCISSA
    register_divergence_key = MEASUREMENT_ID_DIVERGENCE
    register_rotation_key = MEASUREMENT_ID_GENERATOR_ROTATION
    register_max_gain_key = MEASUREMENT_ID_OPERATOR_MAX_GAIN
    register_volume_gain_key = MEASUREMENT_ID_OPERATOR_VOLUME_GAIN
    register_rotation_rate_key = MEASUREMENT_ID_OPERATOR_ROTATION_RATE
    if lag != 1:
        if register_map_key == MEASUREMENT_ID_AFFINE_MAP:
            register_map_key = ONE_STEP_LAG2_MAP
            register_rate_key = ONE_STEP_LAG2_MEAN_RATE
            register_innov_key = ONE_STEP_LAG2_INNOVATION
        register_spectral_key = ONE_STEP_LAG2_SPECTRAL
        register_numerical_key = ONE_STEP_LAG2_NUMERICAL
        register_divergence_key = ONE_STEP_LAG2_DIVERGENCE
        register_rotation_key = ONE_STEP_LAG2_ROTATION
        register_max_gain_key = ONE_STEP_LAG2_MAX_GAIN
        register_volume_gain_key = ONE_STEP_LAG2_VOLUME_GAIN
        register_rotation_rate_key = ONE_STEP_LAG2_ROTATION_RATE
        if variant_id in (None, VARIANT_DECLARED_LAG_STEPS_1):
            variant_id = VARIANT_DECLARED_LAG_STEPS_2

    def _refuse(result: Mapping[str, Any]) -> dict[str, Any]:
        return _attach_refused_functionals(
            result,
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
            register_max_gain_key=register_max_gain_key,
            register_volume_gain_key=register_volume_gain_key,
            register_rotation_rate_key=register_rotation_rate_key,
            variant_id=variant_id,
        )

    refusal = _layer_refusal(state=state, coordinate_layer=coordinate_layer, coordinate_names=names)
    if refusal is not None:
        return _refuse(refusal)
    x, t, segments, finite, reason = validate_trajectory(
        state, time, min_samples=min_samples, segment_id=segment_id
    )
    if reason is not None or x is None or t is None or segments is None:
        return _refuse(
            _unavailable(
                status="insufficient_support" if reason == "insufficient_samples" else "invalid",
                failure_reason=str(reason or "invalid_trajectory"),
                coordinate_layer=coordinate_layer,
                coordinate_names=names,
            )
        )
    if int(n_blocks) != DEFAULT_N_BLOCKS:
        return _refuse(
            _unavailable(
                status="invalid",
                failure_reason="one_step_requires_two_temporal_blocks",
                coordinate_layer=coordinate_layer,
                coordinate_names=names,
            )
        )
    if int(embargo_steps) < lag:
        embargo_fields = {
            "declared_lag_steps": lag,
            "embargo_steps": int(embargo_steps),
            **embargo_claim_fields(),
            "not_composed_one_step": True,
        }
        if lag >= 2:
            embargo_fields["min_embargo_steps"] = 2
        return _refuse(
            _unavailable(
                status="invalid",
                failure_reason=REASON_EMBARGO,
                coordinate_layer=coordinate_layer,
                coordinate_names=names,
                extra_summary=embargo_fields,
                extra_settings=embargo_fields,
            )
        )
    support = build_transition_support(
        x,
        t,
        segments,
        lag=int(lag),
        max_gap_sec=max_gap_sec,
        max_dt_relative_deviation=float(max_dt_relative_deviation),
    )
    shared_settings = {
        "one_step_fit_gate_version": ONE_STEP_FIT_GATE_VERSION,
        "one_step_rel_mse_threshold": float(one_step_rel_mse_threshold),
        "ridge_alpha": float(ridge_alpha),
        "neighborhood_k": int(neighborhood_k),
        "operator_scope": "recording",
        "rel_mse_scoring": "blocked_holdout",
        "n_blocks": int(DEFAULT_N_BLOCKS),
        "embargo_steps": int(embargo_steps),
        **embargo_claim_fields(),
        "declared_lag_steps": int(lag),
        "not_composed_one_step": True,
        "neighborhood_not_used_for_phi": True,
        "distance_epsilon": float(epsilon),
        **_FUNCTIONAL_PROVENANCE,
        **support_settings_fields(support),
    }
    if support.failure_reason == "non_positive_nominal_dt":
        return _refuse(
            _unavailable(
                status="invalid",
                failure_reason="non_positive_nominal_dt",
                coordinate_layer=coordinate_layer,
                coordinate_names=names,
                extra_settings=shared_settings,
            )
        )
    if support.failure_reason == "materially_irregular_increment_timestep":
        return _refuse(
            _unavailable(
                status="not_testable",
                failure_reason="materially_irregular_increment_timestep",
                coordinate_layer=coordinate_layer,
                coordinate_names=names,
                extra_settings=shared_settings,
            )
        )
    n_time, dim = x.shape
    n_refs = int(support.source_idx.size)
    minimum_support = max(int(min_samples), int(min_neighborhood_samples), int(dim * (dim + 1)))
    if n_refs < minimum_support:
        return _refuse(
            _unavailable(
                status="insufficient_support",
                failure_reason=REASON_INSUFFICIENT_LOCAL,
                coordinate_layer=coordinate_layer,
                coordinate_names=names,
                extra_settings=shared_settings,
            )
        )

    source_idx = np.asarray(support.source_idx, dtype=np.int32)
    targets = x[source_idx] + np.asarray(support.increments, dtype=float)
    increment_state = x[source_idx]
    min_hold = max(_MIN_HOLDOUT, dim + 1)
    fold_scores: list[float] = []
    n_holdout = 0
    for train_idx, hold_idx in _temporal_block_folds(source_idx, embargo_steps):
        if int(train_idx.size) < minimum_support or int(hold_idx.size) < min_hold:
            return _refuse(
                _unavailable(
                    status="insufficient_support",
                    failure_reason=REASON_INSUFFICIENT_LOCAL,
                    coordinate_layer=coordinate_layer,
                    coordinate_names=names,
                    extra_settings=shared_settings,
                )
            )
        fitted_oos = _fit_affine_map(
            increment_state[train_idx],
            targets[train_idx],
            np.ones(int(train_idx.size), dtype=np.float32),
            ridge_alpha,
        )
        if fitted_oos is None:
            return _refuse(
                _unavailable(
                    status="insufficient_support",
                    failure_reason=REASON_INSUFFICIENT_LOCAL,
                    coordinate_layer=coordinate_layer,
                    coordinate_names=names,
                    extra_settings=shared_settings,
                )
            )
        phi_oos, intercept_oos, xref_oos = fitted_oos
        rel_oos = _affine_rel_mse(
            increment_state[hold_idx].astype(np.float32),
            targets[hold_idx].astype(np.float32),
            phi_oos,
            intercept_oos,
            xref_oos,
        )
        if not np.isfinite(rel_oos):
            return _refuse(
                _unavailable(
                    status="insufficient_support",
                    failure_reason=REASON_INSUFFICIENT_LOCAL,
                    coordinate_layer=coordinate_layer,
                    coordinate_names=names,
                    extra_settings=shared_settings,
                )
            )
        fold_scores.append(float(rel_oos))
        n_holdout += int(hold_idx.size)
    median_rel = float(np.median(np.asarray(fold_scores, dtype=float)))
    identified = bool(median_rel < float(one_step_rel_mse_threshold))

    phi_hat = np.full((n_time, dim, dim), np.nan, dtype=np.float32)
    affine_reference = np.full((n_time, dim), np.nan, dtype=np.float32)
    affine_intercept = np.full((n_time, dim), np.nan, dtype=np.float32)
    affine_rate = np.full((n_time, dim), np.nan, dtype=np.float32)
    innovation = np.full((n_time, dim, dim), np.nan, dtype=np.float32)
    rel_mse = np.full(n_time, np.nan, dtype=np.float32)
    spectral = np.full(n_time, np.nan, dtype=np.float32)
    numerical = np.full(n_time, np.nan, dtype=np.float32)
    divergence = np.full(n_time, np.nan, dtype=np.float32)
    rotation = np.full(n_time, np.nan, dtype=np.float32)
    max_gain = np.full(n_time, np.nan, dtype=np.float32)
    volume_gain = np.full(n_time, np.nan, dtype=np.float32)
    rotation_rate = np.full(n_time, np.nan, dtype=np.float32)
    generator_ok = np.zeros(n_time, dtype=np.int8)
    n_fitted = 0
    dt = float(support.nominal_dt_sec)
    spec_val = num_val = div_val = rot_val = np.float32(np.nan)
    gain_val = vol_val = rot_rate_val = np.float32(np.nan)
    gain_reason = vol_reason = rot_rate_reason = REASON_UPSTREAM
    gen_ok = 0

    if identified:
        all_weights = np.ones(n_refs, dtype=np.float32)
        fitted_all = _fit_affine_map(increment_state, targets, all_weights, ridge_alpha)
        if fitted_all is None:
            identified = False
        else:
            phi, intercept, x_ref = fitted_all
            resid = targets - (increment_state - x_ref.reshape(1, -1)) @ phi.T - intercept.reshape(1, -1)
            cov = np.full((dim, dim), np.nan, dtype=np.float32)
            if resid.shape[0] >= dim + 1:
                cov_raw = np.atleast_2d(np.cov(resid, rowvar=False, ddof=1)) / dt
                if cov_raw.shape == (dim, dim) and np.all(np.isfinite(cov_raw)):
                    cov = cov_raw.astype(np.float32)
            generator = _generator_from_phi(phi, dt) if write_level3 else None
            if generator is not None:
                eigvals = np.linalg.eigvals(generator)
                spec_val = np.float32(np.max(np.real(eigvals)))
                symmetric = 0.5 * (generator + generator.T)
                num_val = np.float32(np.linalg.eigvalsh(symmetric)[-1])
                div_val = np.float32(np.trace(generator))
                skew = 0.5 * (generator - generator.T)
                rot_val = np.float32(np.linalg.norm(skew))
                gen_ok = 1
            functionals = _operator_functionals_from_phi(phi, dt)
            gain_raw, gain_reason = functionals["max_gain"]
            vol_raw, vol_reason = functionals["volume_gain"]
            rot_raw, rot_rate_reason = functionals["rotation_rate"]
            if gain_raw is not None:
                gain_val = np.float32(gain_raw)
            if vol_raw is not None:
                vol_val = np.float32(vol_raw)
            if rot_raw is not None:
                rot_rate_val = np.float32(rot_raw)
            for center in source_idx:
                t_idx = int(center)
                query = x[t_idx]
                predicted = phi @ (query - x_ref) + intercept
                phi_hat[t_idx] = phi
                affine_reference[t_idx] = x_ref
                affine_intercept[t_idx] = intercept
                affine_rate[t_idx] = ((predicted - query) / dt).astype(np.float32)
                innovation[t_idx] = cov
                rel_mse[t_idx] = np.float32(median_rel)
                if write_level3:
                    spectral[t_idx] = spec_val
                    numerical[t_idx] = num_val
                    divergence[t_idx] = div_val
                    rotation[t_idx] = rot_val
                    generator_ok[t_idx] = np.int8(gen_ok)
                max_gain[t_idx] = gain_val
                volume_gain[t_idx] = vol_val
                rotation_rate[t_idx] = rot_rate_val
                n_fitted += 1

    map_status = "computed" if identified else "insufficient_support"
    map_reason = None if identified else REASON_NOT_BETTER_THAN_BASELINE
    map_qual = QUALIFICATION_ONE_STEP_IDENTIFIED if identified else QUALIFICATION_ONE_STEP_NOT_IDENTIFIED
    summary_common = {
        **support_summary_fields(support),
        "n_increment_pairs": n_refs,
        "n_fitted_windows": int(n_fitted),
        "n_holdout_pairs": n_holdout,
        "rel_mse_baseline_median": median_rel,
        "rel_mse_baseline_folds": np.asarray(fold_scores, dtype=np.float32),
        "rel_mse_scoring": "blocked_holdout",
        "n_blocks": int(DEFAULT_N_BLOCKS),
        "embargo_steps": int(embargo_steps),
        "operator_scope": "recording",
        "one_step_identified": identified,
        "declared_lag_steps": int(lag),
        "not_composed_one_step": True,
        "nominal_dt_sec": dt,
        "not_sde_drift": True,
        "not_production_jacobian": True,
        "ito_drift_level3_written": False,
    }
    series_common = {
        **support_series_fields(support),
        "rel_mse_baseline": rel_mse,
    }
    map_leaf = _leaf(
        measurement_id=MEASUREMENT_ID_AFFINE_MAP,
        status=map_status,
        failure_reason=map_reason,
        coordinate_layer=coordinate_layer,
        coordinate_names=names,
        series={
            **series_common,
            "phi_hat": phi_hat,
            "affine_reference": affine_reference,
            "affine_intercept": affine_intercept,
        },
        extra_summary=summary_common,
        extra_settings=shared_settings,
        qualification_override=map_qual,
        register_key=register_map_key,
        variant_id=variant_id,
    )
    rate_leaf = _leaf(
        measurement_id=MEASUREMENT_ID_AFFINE_MEAN_RATE,
        status=map_status,
        failure_reason=map_reason,
        coordinate_layer=coordinate_layer,
        coordinate_names=names,
        series={**series_common, "affine_mean_rate": affine_rate},
        extra_summary=summary_common,
        extra_settings=shared_settings,
        qualification_override=map_qual,
        register_key=register_rate_key,
        variant_id=variant_id,
    )
    innovation_leaf = _leaf(
        measurement_id=MEASUREMENT_ID_INNOVATION_COVARIANCE,
        status=map_status,
        failure_reason=map_reason,
        coordinate_layer=coordinate_layer,
        coordinate_names=names,
        series={**series_common, "innovation_covariance": innovation},
        extra_summary={**summary_common, "not_ito_diffusion_tensor": True, "not_diffusion_a_hat": True},
        extra_settings=shared_settings,
        qualification_override=map_qual,
        register_key=register_innov_key,
        variant_id=variant_id,
    )

    def _level3_leaf(
        measurement_id: str,
        values: np.ndarray,
        series_name: str,
        register_key: str,
    ) -> dict[str, Any]:
        if not identified:
            return _leaf(
                measurement_id=measurement_id,
                status="not_testable",
                failure_reason=REASON_UPSTREAM,
                coordinate_layer=coordinate_layer,
                coordinate_names=names,
                extra_summary=summary_common,
                extra_settings=shared_settings,
                qualification_override=QUALIFICATION_GENERATOR_PROXY_NOT_ITO,
                register_key=register_key,
                variant_id=variant_id,
            )
        n_gen = int(np.sum(generator_ok))
        status = "computed" if n_gen > 0 else "insufficient_support"
        reason = None if n_gen > 0 else REASON_LOGM
        return _leaf(
            measurement_id=measurement_id,
            status=status,
            failure_reason=reason,
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
            series={**series_common, series_name: values, "generator_log_ok": generator_ok},
            extra_summary={
                **summary_common,
                "n_generator_windows": n_gen,
                "generator_conversion": "matrix_log_over_nominal_dt",
                "not_ito_drift": True,
            },
            extra_settings=shared_settings,
            qualification_override=QUALIFICATION_GENERATOR_PROXY_NOT_ITO,
            register_key=register_key,
            variant_id=variant_id,
        )

    spectral_leaf = None
    numerical_leaf = None
    divergence_leaf = None
    rotation_leaf = None
    if write_level3:
        spectral_leaf = _level3_leaf(
            MEASUREMENT_ID_SPECTRAL_ABSCISSA, spectral, "spectral_abscissa", register_spectral_key
        )
        numerical_leaf = _level3_leaf(
            MEASUREMENT_ID_NUMERICAL_ABSCISSA, numerical, "numerical_abscissa", register_numerical_key
        )
        divergence_leaf = _level3_leaf(
            MEASUREMENT_ID_DIVERGENCE, divergence, "divergence", register_divergence_key
        )
        rotation_leaf = _level3_leaf(
            MEASUREMENT_ID_GENERATOR_ROTATION, rotation, "generator_rotation_norm", register_rotation_key
        )

    def _functional_leaf(
        measurement_id: str,
        values: np.ndarray,
        series_name: str,
        register_key: str,
        value_ok: bool,
        fail_reason: str | None,
    ) -> dict[str, Any]:
        if not identified:
            return _leaf(
                measurement_id=measurement_id,
                status="not_testable",
                failure_reason=REASON_UPSTREAM,
                coordinate_layer=coordinate_layer,
                coordinate_names=names,
                extra_summary=summary_common,
                extra_settings=shared_settings,
                qualification_override=QUALIFICATION_ONE_STEP_FUNCTIONAL,
                register_key=register_key,
                variant_id=variant_id,
            )
        status = "computed" if value_ok else "insufficient_support"
        return _leaf(
            measurement_id=measurement_id,
            status=status,
            failure_reason=None if value_ok else fail_reason,
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
            series={**series_common, series_name: values},
            extra_summary={
                **summary_common,
                "not_independent_singular_oos": True,
                "not_peak_gain_level4": True,
                "distance_metric_id": "euclidean_on_release_scaled_chart",
            },
            extra_settings=shared_settings,
            qualification_override=QUALIFICATION_ONE_STEP_FUNCTIONAL,
            register_key=register_key,
            variant_id=variant_id,
        )

    max_gain_leaf = _functional_leaf(
        MEASUREMENT_ID_OPERATOR_MAX_GAIN,
        max_gain,
        "operator_max_gain_rate",
        register_max_gain_key,
        np.any(np.isfinite(max_gain)),
        gain_reason,
    )
    volume_gain_leaf = _functional_leaf(
        MEASUREMENT_ID_OPERATOR_VOLUME_GAIN,
        volume_gain,
        "operator_volume_gain_rate",
        register_volume_gain_key,
        np.any(np.isfinite(volume_gain)),
        vol_reason,
    )
    rotation_rate_leaf = _functional_leaf(
        MEASUREMENT_ID_OPERATOR_ROTATION_RATE,
        rotation_rate,
        "operator_rotation_rate",
        register_rotation_rate_key,
        np.any(np.isfinite(rotation_rate)),
        rot_rate_reason,
    )

    parent_status = "computed" if identified else "insufficient_support"
    parent_reason = None if identified else REASON_NOT_BETTER_THAN_BASELINE
    parent = unavailable_result(
        AFFINE_ONE_STEP_SCHEMA_VERSION,
        status=parent_status if parent_status != "computed" else "not_testable",
        failure_reason=parent_reason or "not_requested",
        coordinate_layer=coordinate_layer,
        coordinate_names=names,
    )
    if parent_status == "computed":
        parent["computation_status"] = "computed"
        parent["failure_reason"] = None
    parent["series"] = {}
    parent["summary"] = {
        **summary_common,
        "affine_one_step_map_status": map_status,
        "spectral_abscissa_status": None if spectral_leaf is None else spectral_leaf.get("computation_status"),
        "numerical_abscissa_status": None if numerical_leaf is None else numerical_leaf.get("computation_status"),
    }
    parent["provenance"] = build_provenance(
        coordinate_layer=coordinate_layer,
        coordinate_names=names,
        time_semantics="within_segment_one_step_affine_map",
        estimator=ESTIMATOR_NAME,
        settings={
            **shared_settings,
            "not_sde_drift": True,
            "not_production_jacobian": True,
            "never_auto_ito_qualified": True,
            "not_jacobian_metrics_icare_gate": True,
        },
    )
    parent[MEASUREMENT_ID_AFFINE_MAP] = map_leaf
    parent[MEASUREMENT_ID_AFFINE_MEAN_RATE] = rate_leaf
    parent[MEASUREMENT_ID_INNOVATION_COVARIANCE] = innovation_leaf
    parent[MEASUREMENT_ID_OPERATOR_MAX_GAIN] = max_gain_leaf
    parent[MEASUREMENT_ID_OPERATOR_VOLUME_GAIN] = volume_gain_leaf
    parent[MEASUREMENT_ID_OPERATOR_ROTATION_RATE] = rotation_rate_leaf
    if write_level3:
        parent[MEASUREMENT_ID_SPECTRAL_ABSCISSA] = spectral_leaf
        parent[MEASUREMENT_ID_NUMERICAL_ABSCISSA] = numerical_leaf
        parent[MEASUREMENT_ID_DIVERGENCE] = divergence_leaf
        parent[MEASUREMENT_ID_GENERATOR_ROTATION] = rotation_leaf
    parent = stamp_register_fields(parent, register_map_key, variant_id=variant_id)
    parent = _stamp_qualification(parent, map_qual)
    return _finalize(parent)


_LEVEL2_LEAF_IDS = (
    MEASUREMENT_ID_AFFINE_MAP,
    MEASUREMENT_ID_AFFINE_MEAN_RATE,
    MEASUREMENT_ID_INNOVATION_COVARIANCE,
)
_LEVEL2_FUNCTIONAL_IDS = (
    MEASUREMENT_ID_OPERATOR_MAX_GAIN,
    MEASUREMENT_ID_OPERATOR_VOLUME_GAIN,
    MEASUREMENT_ID_OPERATOR_ROTATION_RATE,
)
_LEVEL3_LEAF_IDS = (
    MEASUREMENT_ID_SPECTRAL_ABSCISSA,
    MEASUREMENT_ID_NUMERICAL_ABSCISSA,
    MEASUREMENT_ID_DIVERGENCE,
    MEASUREMENT_ID_GENERATOR_ROTATION,
)


def _lag2_identity_group(lag2: Mapping[str, Any]) -> dict[str, Any]:
    """Nest lag-2 level-2 and level-3 leaves. Never attach Φ₁²."""
    if MEASUREMENT_ID_AFFINE_MAP in lag2:
        return {
            key: lag2[key]
            for key in (*_LEVEL2_LEAF_IDS, *_LEVEL2_FUNCTIONAL_IDS, *_LEVEL3_LEAF_IDS)
            if key in lag2
        }
    status = str(lag2.get("computation_status") or "not_testable")
    reason = str(lag2.get("failure_reason") or "not_requested")
    layer = str((lag2.get("provenance") or {}).get("coordinate_layer") or "coords_3d_subject_anchored")
    names = list((lag2.get("provenance") or {}).get("coordinate_names") or ["m", "d", "e"])
    extra_summary = dict(lag2.get("summary") or {})
    extra_settings = dict((lag2.get("provenance") or {}).get("settings") or {})
    placeholder_status = status if status != "computed" else "insufficient_support"
    keys = (
        (MEASUREMENT_ID_AFFINE_MAP, ONE_STEP_LAG2_MAP, None),
        (MEASUREMENT_ID_AFFINE_MEAN_RATE, ONE_STEP_LAG2_MEAN_RATE, None),
        (MEASUREMENT_ID_INNOVATION_COVARIANCE, ONE_STEP_LAG2_INNOVATION, None),
        (MEASUREMENT_ID_OPERATOR_MAX_GAIN, ONE_STEP_LAG2_MAX_GAIN, QUALIFICATION_ONE_STEP_FUNCTIONAL),
        (MEASUREMENT_ID_OPERATOR_VOLUME_GAIN, ONE_STEP_LAG2_VOLUME_GAIN, QUALIFICATION_ONE_STEP_FUNCTIONAL),
        (MEASUREMENT_ID_OPERATOR_ROTATION_RATE, ONE_STEP_LAG2_ROTATION_RATE, QUALIFICATION_ONE_STEP_FUNCTIONAL),
        (MEASUREMENT_ID_SPECTRAL_ABSCISSA, ONE_STEP_LAG2_SPECTRAL, QUALIFICATION_GENERATOR_PROXY_NOT_ITO),
        (MEASUREMENT_ID_NUMERICAL_ABSCISSA, ONE_STEP_LAG2_NUMERICAL, QUALIFICATION_GENERATOR_PROXY_NOT_ITO),
        (MEASUREMENT_ID_DIVERGENCE, ONE_STEP_LAG2_DIVERGENCE, QUALIFICATION_GENERATOR_PROXY_NOT_ITO),
        (MEASUREMENT_ID_GENERATOR_ROTATION, ONE_STEP_LAG2_ROTATION, QUALIFICATION_GENERATOR_PROXY_NOT_ITO),
    )
    return {
        measurement_id: _leaf(
            measurement_id=measurement_id,
            status=placeholder_status,
            failure_reason=reason,
            coordinate_layer=layer,
            coordinate_names=names,
            extra_summary=extra_summary,
            extra_settings=extra_settings,
            qualification_override=qualification,
            register_key=register_key,
            variant_id=VARIANT_DECLARED_LAG_STEPS_2,
        )
        for measurement_id, register_key, qualification in keys
    }


def _compose_affine(
    phi: np.ndarray, intercept: np.ndarray, x_ref: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the affine chart map twice. Not the direct lag-2 estimator."""
    phi_m = np.asarray(phi, dtype=float)
    intercept_v = np.asarray(intercept, dtype=float).reshape(-1)
    xref_v = np.asarray(x_ref, dtype=float).reshape(-1)
    phi_h = phi_m @ phi_m
    intercept_h = phi_m @ (intercept_v - xref_v) + intercept_v
    return phi_h.astype(np.float32), intercept_h.astype(np.float32), xref_v.astype(np.float32)


def _finite_affine_from_leaf(leaf: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    series = leaf.get("series") or {}
    phi_series = np.asarray(series.get("phi_hat"), dtype=float)
    ref_series = np.asarray(series.get("affine_reference"), dtype=float)
    intercept_series = np.asarray(series.get("affine_intercept"), dtype=float)
    if phi_series.ndim != 3 or phi_series.shape[0] == 0:
        return None
    finite = np.all(np.isfinite(phi_series.reshape(phi_series.shape[0], -1)), axis=1)
    finite &= np.all(np.isfinite(ref_series), axis=1)
    finite &= np.all(np.isfinite(intercept_series), axis=1)
    if not np.any(finite):
        return None
    idx = int(np.flatnonzero(finite)[0])
    return (
        phi_series[idx].astype(np.float32),
        intercept_series[idx].astype(np.float32),
        ref_series[idx].astype(np.float32),
    )


def _estimate_iterated_horizon(
    state: np.ndarray,
    time: np.ndarray,
    lag1: Mapping[str, Any],
    *,
    segment_id: np.ndarray | None,
    coordinate_layer: str,
    coordinate_names: list[str] | None,
    min_samples: int,
    min_neighborhood_samples: int,
    max_gap_sec: float | None,
    max_dt_relative_deviation: float,
    ridge_alpha: float,
    one_step_rel_mse_threshold: float,
    n_blocks: int,
    embargo_steps: int,
    lag2_map: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Qualify Φ₁ applied twice on lag-2 pairs. Does not write the lag-2 identity."""
    names = list(coordinate_names or ["m", "d", "e"])
    horizon_settings = {
        "horizon_steps": HORIZON_STEPS,
        "claim_class": "finite_horizon_propagation",
        "not_direct_lag2_map": True,
        "composed_from_lag1": True,
        "embargo_steps": int(embargo_steps),
        **embargo_claim_fields(),
        "min_embargo_steps": HORIZON_STEPS,
        "one_step_fit_gate_version": ONE_STEP_FIT_GATE_VERSION,
        "one_step_rel_mse_threshold": float(one_step_rel_mse_threshold),
        "rel_mse_scoring": "blocked_holdout",
        "n_blocks": int(DEFAULT_N_BLOCKS),
        "operator_scope": "recording",
        "not_sde_drift": True,
    }

    def _horizon_unavailable(status: str, reason: str, extra_summary: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return _leaf(
            measurement_id=MEASUREMENT_ID_ITERATED_ONE_STEP_MAP,
            status=status,
            failure_reason=reason,
            coordinate_layer=coordinate_layer,
            coordinate_names=names,
            extra_summary={**horizon_settings, **dict(extra_summary or {})},
            extra_settings=horizon_settings,
            qualification_override=QUALIFICATION_HORIZON_PROPAGATION_NOT_IDENTIFIED,
            register_key=MEASUREMENT_ID_ITERATED_ONE_STEP_MAP,
            variant_id=VARIANT_HORIZON_STEPS_2,
        )

    lag1_map = lag1.get(MEASUREMENT_ID_AFFINE_MAP) or {}
    if str(lag1_map.get("qualification_status") or "") != QUALIFICATION_ONE_STEP_IDENTIFIED:
        return _horizon_unavailable("not_testable", REASON_UPSTREAM)
    if int(n_blocks) != DEFAULT_N_BLOCKS:
        return _horizon_unavailable("invalid", "one_step_requires_two_temporal_blocks")
    if int(embargo_steps) < HORIZON_STEPS:
        return _horizon_unavailable("invalid", REASON_EMBARGO)
    fitted = _finite_affine_from_leaf(lag1_map)
    if fitted is None:
        return _horizon_unavailable("not_testable", REASON_UPSTREAM)
    x, t, segments, finite, reason = validate_trajectory(
        state, time, min_samples=min_samples, segment_id=segment_id
    )
    if reason is not None or x is None or t is None or segments is None:
        return _horizon_unavailable(
            "insufficient_support" if reason == "insufficient_samples" else "invalid",
            str(reason or "invalid_trajectory"),
        )
    support1 = build_transition_support(
        x, t, segments, lag=1, max_gap_sec=max_gap_sec,
        max_dt_relative_deviation=float(max_dt_relative_deviation),
    )
    support2 = build_transition_support(
        x, t, segments, lag=HORIZON_STEPS, max_gap_sec=max_gap_sec,
        max_dt_relative_deviation=float(max_dt_relative_deviation),
    )
    horizon_settings.update(support_settings_fields(support2))
    horizon_settings["transition_support_lag"] = int(support2.lag)
    if support2.failure_reason == "non_positive_nominal_dt":
        return _horizon_unavailable("invalid", "non_positive_nominal_dt")
    if support2.failure_reason == "materially_irregular_increment_timestep":
        return _horizon_unavailable("not_testable", "materially_irregular_increment_timestep")
    dim = int(x.shape[1])
    n_time = int(x.shape[0])
    minimum_support = max(int(min_samples), int(min_neighborhood_samples), int(dim * (dim + 1)))
    min_hold = max(_MIN_HOLDOUT, dim + 1)
    src1 = np.asarray(support1.source_idx, dtype=np.int32)
    src2 = np.asarray(support2.source_idx, dtype=np.int32)
    if src1.size < minimum_support or src2.size < minimum_support:
        return _horizon_unavailable("insufficient_support", REASON_INSUFFICIENT_LOCAL)
    targets1 = x[src1] + np.asarray(support1.increments, dtype=float)
    targets2 = x[src2] + np.asarray(support2.increments, dtype=float)
    fold_scores: list[float] = []
    n_holdout = 0
    split_index = int(np.median(src2))
    embargo = int(embargo_steps)
    fold_regions = (
        (src1 < (split_index - embargo), src2 > (split_index + embargo)),
        (src1 > (split_index + embargo), src2 < (split_index - embargo)),
    )
    for train_mask, hold_mask in fold_regions:
        if int(np.sum(train_mask)) < minimum_support or int(np.sum(hold_mask)) < min_hold:
            return _horizon_unavailable("insufficient_support", REASON_INSUFFICIENT_LOCAL)
        fitted_oos = _fit_affine_map(
            x[src1[train_mask]],
            targets1[train_mask],
            np.ones(int(np.sum(train_mask)), dtype=np.float32),
            ridge_alpha,
        )
        if fitted_oos is None:
            return _horizon_unavailable("insufficient_support", REASON_INSUFFICIENT_LOCAL)
        phi_oos, intercept_oos, xref_oos = fitted_oos
        phi_h, intercept_h, xref_h = _compose_affine(phi_oos, intercept_oos, xref_oos)
        rel_oos = _affine_rel_mse(
            x[src2[hold_mask]].astype(np.float32),
            targets2[hold_mask].astype(np.float32),
            phi_h,
            intercept_h,
            xref_h,
        )
        if not np.isfinite(rel_oos):
            return _horizon_unavailable("insufficient_support", REASON_INSUFFICIENT_LOCAL)
        fold_scores.append(float(rel_oos))
        n_holdout += int(np.sum(hold_mask))
    median_rel = float(np.median(np.asarray(fold_scores, dtype=float)))
    identified = bool(median_rel < float(one_step_rel_mse_threshold))
    phi, intercept, x_ref = fitted
    phi_h, intercept_h, xref_h = _compose_affine(phi, intercept, x_ref)
    phi_series = np.full((n_time, dim, dim), np.nan, dtype=np.float32)
    ref_series = np.full((n_time, dim), np.nan, dtype=np.float32)
    intercept_series = np.full((n_time, dim), np.nan, dtype=np.float32)
    rel_mse = np.full(n_time, np.nan, dtype=np.float32)
    n_fitted = 0
    if identified:
        for center in src2:
            t_idx = int(center)
            phi_series[t_idx] = phi_h
            ref_series[t_idx] = xref_h
            intercept_series[t_idx] = intercept_h
            rel_mse[t_idx] = np.float32(median_rel)
            n_fitted += 1
    phi2_distance = np.float32(np.nan)
    if lag2_map is not None and str(lag2_map.get("qualification_status") or "") == QUALIFICATION_ONE_STEP_IDENTIFIED:
        lag2_fitted = _finite_affine_from_leaf(lag2_map)
        if lag2_fitted is not None:
            phi2_distance = np.float32(np.linalg.norm(phi_h - lag2_fitted[0]))
    status = "computed" if identified else "insufficient_support"
    reason = None if identified else REASON_HORIZON
    qual = (
        QUALIFICATION_HORIZON_PROPAGATION_IDENTIFIED
        if identified
        else QUALIFICATION_HORIZON_PROPAGATION_NOT_IDENTIFIED
    )
    summary = {
        **horizon_settings,
        **support_summary_fields(support2),
        "n_increment_pairs": int(src2.size),
        "n_fitted_windows": int(n_fitted),
        "n_holdout_pairs": n_holdout,
        "rel_mse_baseline_median": median_rel,
        "rel_mse_baseline_folds": np.asarray(fold_scores, dtype=np.float32),
        "horizon_identified": identified,
        "composed_phi_lag2_frobenius": phi2_distance,
        "not_sde_drift": True,
        "ito_drift_level3_written": False,
    }
    return _leaf(
        measurement_id=MEASUREMENT_ID_ITERATED_ONE_STEP_MAP,
        status=status,
        failure_reason=reason,
        coordinate_layer=coordinate_layer,
        coordinate_names=names,
        series={
            **support_series_fields(support2),
            "phi_composed": phi_series,
            "affine_reference": ref_series,
            "affine_intercept": intercept_series,
            "rel_mse_baseline": rel_mse,
        },
        extra_summary=summary,
        extra_settings=horizon_settings,
        qualification_override=qual,
        register_key=MEASUREMENT_ID_ITERATED_ONE_STEP_MAP,
        variant_id=VARIANT_HORIZON_STEPS_2,
    )


def _with_iterated_horizon(
    parent: Mapping[str, Any],
    lag1: Mapping[str, Any],
    state: np.ndarray,
    time: np.ndarray,
    shared_kwargs: Mapping[str, Any],
    lag2_map: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    out = dict(parent)
    leaf = _estimate_iterated_horizon(
        state,
        time,
        lag1,
        segment_id=shared_kwargs.get("segment_id"),
        coordinate_layer=str(shared_kwargs.get("coordinate_layer") or "coords_3d_subject_anchored"),
        coordinate_names=shared_kwargs.get("coordinate_names"),
        min_samples=int(shared_kwargs.get("min_samples", 30)),
        min_neighborhood_samples=int(shared_kwargs.get("min_neighborhood_samples", 10)),
        max_gap_sec=shared_kwargs.get("max_gap_sec"),
        max_dt_relative_deviation=float(shared_kwargs.get("max_dt_relative_deviation", 0.05)),
        ridge_alpha=float(shared_kwargs.get("ridge_alpha", 1e-4)),
        one_step_rel_mse_threshold=float(shared_kwargs.get("one_step_rel_mse_threshold", DEFAULT_REL_MSE_THRESHOLD)),
        n_blocks=int(shared_kwargs.get("n_blocks", DEFAULT_N_BLOCKS)),
        embargo_steps=int(shared_kwargs.get("embargo_steps", DEFAULT_EMBARGO_STEPS)),
        lag2_map=lag2_map,
    )
    out[MEASUREMENT_ID_ITERATED_ONE_STEP_MAP] = leaf
    summary = dict(out.get("summary") or {})
    summary["iterated_horizon_status"] = leaf.get("computation_status")
    out["summary"] = summary
    return _finalize(out)


def estimate_affine_one_step_family(
    state: np.ndarray,
    time: np.ndarray,
    *,
    segment_id: np.ndarray | None = None,
    coordinate_layer: str = "coords_3d_subject_anchored",
    coordinate_names: list[str] | None = None,
    neighborhood_k: int = 20,
    min_samples: int = 30,
    min_neighborhood_samples: int = 10,
    max_gap_sec: float | None = None,
    max_dt_relative_deviation: float = 0.05,
    ridge_alpha: float = 1e-4,
    one_step_rel_mse_threshold: float = DEFAULT_REL_MSE_THRESHOLD,
    n_blocks: int = DEFAULT_N_BLOCKS,
    embargo_steps: int = DEFAULT_EMBARGO_STEPS,
    epsilon: float = 1e-12,
    declared_lags: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Fit lag-1 and, when requested, a direct lag-2 map. Does not compose Φ₁."""
    lags = tuple(DEFAULT_ONE_STEP_DECLARED_LAGS if declared_lags is None else declared_lags)
    if not lags or any(int(lag) not in ALLOWED_ONE_STEP_DECLARED_LAGS for lag in lags):
        raise ValueError(
            "declared_lags must be a non-empty subset of "
            f"{list(ALLOWED_ONE_STEP_DECLARED_LAGS)}; composing the lag-1 "
            "map is not the lag-2 identity."
        )
    shared_kwargs = {
        "segment_id": segment_id,
        "coordinate_layer": coordinate_layer,
        "coordinate_names": coordinate_names,
        "neighborhood_k": neighborhood_k,
        "min_samples": min_samples,
        "min_neighborhood_samples": min_neighborhood_samples,
        "max_gap_sec": max_gap_sec,
        "max_dt_relative_deviation": max_dt_relative_deviation,
        "ridge_alpha": ridge_alpha,
        "one_step_rel_mse_threshold": one_step_rel_mse_threshold,
        "n_blocks": n_blocks,
        "embargo_steps": embargo_steps,
        "epsilon": epsilon,
    }
    lag1 = None
    if 1 in lags:
        lag1 = _recording_affine_at_lag(
            state,
            time,
            lag=1,
            write_level3=True,
            register_map_key=MEASUREMENT_ID_AFFINE_MAP,
            register_rate_key=MEASUREMENT_ID_AFFINE_MEAN_RATE,
            register_innov_key=MEASUREMENT_ID_INNOVATION_COVARIANCE,
            variant_id=VARIANT_DECLARED_LAG_STEPS_1,
            **shared_kwargs,
        )
    lag2 = None
    if 2 in lags:
        lag2 = _recording_affine_at_lag(
            state,
            time,
            lag=2,
            write_level3=True,
            register_map_key=ONE_STEP_LAG2_MAP,
            register_rate_key=ONE_STEP_LAG2_MEAN_RATE,
            register_innov_key=ONE_STEP_LAG2_INNOVATION,
            variant_id=VARIANT_DECLARED_LAG_STEPS_2,
            **shared_kwargs,
        )
    if lag1 is not None and lag2 is None:
        return _with_iterated_horizon(lag1, lag1, state, time, shared_kwargs)
    if lag2 is not None and lag1 is None:
        parent = dict(lag2)
        for key in (*_LEVEL2_LEAF_IDS, *_LEVEL2_FUNCTIONAL_IDS, *_LEVEL3_LEAF_IDS, MEASUREMENT_ID_ITERATED_ONE_STEP_MAP):
            parent.pop(key, None)
        parent[VARIANT_DECLARED_LAG_2] = _lag2_identity_group(lag2)
        summary = dict(parent.get("summary") or {})
        summary["declared_lag_2_status"] = lag2.get("computation_status")
        parent["summary"] = summary
        return _finalize(parent)
    parent = dict(lag1)
    parent[VARIANT_DECLARED_LAG_2] = _lag2_identity_group(lag2)
    summary = dict(parent.get("summary") or {})
    summary["declared_lag_2_status"] = lag2.get("computation_status")
    parent["summary"] = summary
    lag2_map = None
    if isinstance(lag2, Mapping):
        lag2_map = (lag2.get(VARIANT_DECLARED_LAG_2) or {}).get(MEASUREMENT_ID_AFFINE_MAP)
        if lag2_map is None:
            lag2_map = lag2.get(MEASUREMENT_ID_AFFINE_MAP)
    return _with_iterated_horizon(parent, lag1, state, time, shared_kwargs, lag2_map=lag2_map)
