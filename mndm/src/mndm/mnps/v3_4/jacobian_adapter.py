"""Local Jacobian on the MNPS 3.4 flat nine coordinates.

The fit uses the same local-linear estimator as the legacy charts, on a
different coordinate layer. Results stay inside the v3.4 measurement payload
so they are written under ``/measurements/v3_4/jacobian`` and do not replace
``/jacobian`` or ``/jacobian_9D``.

Default is on when a v3.4 EEG export is built. Set ``v3_4_jacobian.enabled``
to false to skip it. Whitening stays off: the Euclidean metric is the declared
coordinate chart, not a per-recording rescaling. An incomplete window is left
out of the fit. A failed fit does not become a zero matrix.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ...dynamics.jacobian_metrics import compute_jacobian_metrics
from ...jacobian import estimate_local_jacobians, jacobian_segment_ids
from ...pipeline.robustness_helpers import apply_standard_jacobian_window_policy
from ...projection import build_knn_indices, estimate_derivatives_with_time_gaps
from .candidates import resolve_v3_4_config

SCHEMA = "mndm.mnps_v3_4.jacobian.v1"
METRICS_SCHEMA = "mndm.jacobian_metrics.v1"
_ANCHOR_APPLIED = {"computed", "computed_exploratory_transfer"}


def v3_4_jacobian_enabled(config: Mapping[str, Any] | None) -> bool:
    """Return whether the flat-nine Jacobian should be computed.

    Missing configuration is on. An explicit ``enabled: false`` is the off switch.
    """
    block = _jacobian_block(config)
    return bool(block.get("enabled", True))


def attach_v3_4_jacobian(
    payload: Mapping[str, Any] | None,
    config: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Attach ``jacobian`` to a v3.4 measurement payload.

    Anchored coordinates are used only after an anchor actually applied and at
    least one window is complete. Otherwise the raw nine are used. The two
    layers are never mixed.
    """
    if not isinstance(payload, Mapping) or not payload:
        return None if payload is None else dict(payload)
    out = dict(payload)
    if not v3_4_jacobian_enabled(config):
        return out
    out["jacobian"] = _build_jacobian(out, config if isinstance(config, Mapping) else {})
    return out


def attach_meg_v3_4_jacobians(
    payload: Mapping[str, Any] | None,
    config: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Attach a separate raw Jacobian to every nine-coordinate MEG variant.

    MEG exports nest coordinate variants under sensor families. Constructing a
    temporary flat adapter payload lets the existing estimator run unchanged,
    while keeping MAG, GRAD, and each coordinate definition separate in output.
    The seven-coordinate removal-only variant is retained with an explicit
    ``not_testable`` result rather than padded to nine coordinates.
    """
    if not isinstance(payload, Mapping) or not payload:
        return None if payload is None else dict(payload)
    out = dict(payload)
    meg_block = config.get("v3_4_meg_jacobian") if isinstance(config, Mapping) else None
    if not isinstance(meg_block, Mapping) or not bool(meg_block.get("enabled", False)):
        return out
    families = out.get("sensor_families")
    if not isinstance(families, Mapping):
        return out
    start, end = _meg_window_bounds(out)
    provenance = out.get("provenance")
    provenance = dict(provenance) if isinstance(provenance, Mapping) else {}
    provenance["anchor_application_status"] = str(
        out.get("anchor_application_status") or "not_applied"
    )
    updated_families: dict[str, Any] = {}
    for family_name, family_payload in families.items():
        if not isinstance(family_payload, Mapping):
            updated_families[str(family_name)] = family_payload
            continue
        family_out = dict(family_payload)
        variants = family_out.get("variants")
        if not isinstance(variants, Mapping):
            updated_families[str(family_name)] = family_out
            continue
        updated_variants: dict[str, Any] = {}
        for variant_id, variant_payload in variants.items():
            if not isinstance(variant_payload, Mapping):
                updated_variants[str(variant_id)] = variant_payload
                continue
            variant_out = dict(variant_payload)
            values = np.asarray(variant_out.get("raw_values", []), dtype=np.float64)
            support = np.asarray(variant_out.get("coordinate_support", []), dtype=bool)
            complete = np.asarray(variant_out.get("complete_window_mask", []), dtype=bool)
            flat_payload = {
                "coordinate_order": list(variant_out.get("coordinate_names") or []),
                "raw_coordinates": {"values": values, "support": support},
                "coordinates": {"values": np.full_like(values, np.nan) if values.ndim == 2 else None},
                "complete_9d_mask": complete,
                "window_start_sec": start,
                "window_end_sec": end,
                "provenance": provenance,
            }
            adapter_config = dict(config) if isinstance(config, Mapping) else {}
            # Keep the MEG activation/settings modality-specific. The common
            # EEG config may enable v3_4_jacobian and must not implicitly turn
            # this experimental MEG estimator on.
            adapter_config["v3_4_jacobian"] = dict(meg_block)
            jacobian = _build_jacobian(flat_payload, adapter_config)
            jacobian["sensor_family"] = str(family_name)
            jacobian["coordinate_variant"] = str(variant_id)
            jacobian["coordinate_units"] = list(variant_out.get("coordinate_units") or [])
            variant_out["jacobian"] = jacobian
            updated_variants[str(variant_id)] = variant_out
        family_out["variants"] = updated_variants
        updated_families[str(family_name)] = family_out
    out["sensor_families"] = updated_families
    return out


def _meg_window_bounds(payload: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    axis = payload.get("window_axis")
    axis = axis if isinstance(axis, Mapping) else {}
    start = np.asarray(axis.get("window_start_sec", []), dtype=np.float64).reshape(-1)
    end = np.asarray(axis.get("window_end_sec", []), dtype=np.float64).reshape(-1)
    return start, end


def _jacobian_block(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    direct = config.get("v3_4_jacobian")
    if isinstance(direct, Mapping):
        return direct
    resolved = resolve_v3_4_config(config)
    nested = resolved.get("v3_4_jacobian") if isinstance(resolved, Mapping) else None
    return nested if isinstance(nested, Mapping) else {}


def _build_jacobian(payload: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    block = _jacobian_block(config)
    names = [str(name) for name in list(payload.get("coordinate_order") or [])]
    state, valid, layer, unit_note = _select_state(payload, names)
    n_windows = int(state.shape[0]) if state.ndim == 2 else 0
    base = {
        "schema": SCHEMA,
        "coordinate_layer": layer,
        "coordinate_names": names,
        "coordinate_unit_note": unit_note,
        "n_windows": n_windows,
        "n_complete_windows": int(np.count_nonzero(valid)),
        "whiten": False,
        "metric": "euclidean_on_declared_coordinates",
    }
    if state.ndim != 2 or state.shape[1] != 9 or len(names) != 9:
        return {
            **base,
            "computation_status": "not_testable",
            "failure_reason": "v3_4_jacobian_requires_flat_nine",
        }
    kept = np.flatnonzero(valid)
    if kept.size < 2:
        return {
            **base,
            "computation_status": "insufficient_support",
            "failure_reason": "fewer_than_two_complete_v3_4_windows",
        }

    mnps = config.get("mnps") if isinstance(config.get("mnps"), Mapping) else {}
    derivative = mnps.get("derivative") if isinstance(mnps.get("derivative"), Mapping) else {}
    derivative_robust = mnps.get("derivative_robust") if isinstance(mnps.get("derivative_robust"), Mapping) else {}
    ridge = mnps.get("ridge") if isinstance(mnps.get("ridge"), Mapping) else {}
    method = str(block.get("derivative_method", derivative.get("method", "sav_gol")))
    window = int(block.get("derivative_window", derivative.get("window", 7)))
    polyorder = int(block.get("derivative_polyorder", derivative.get("polyorder", 3)))
    knn_k = int(block.get("knn_k", mnps.get("knn_k", 20)))
    knn_metric = str(block.get("knn_metric", mnps.get("knn_metric", "euclidean")))
    super_window = int(block.get("super_window", mnps.get("super_window", 3)))
    ridge_alpha = float(block.get("ridge_alpha", mnps.get("ridge_alpha", 1.0)))
    distance_weighted = bool(block.get("distance_weighted", ridge.get("distance_weighted", True)))
    whiten = bool(block.get("whiten", False))

    start = _window_time(payload, "window_start_sec", n_windows)
    end = _window_time(payload, "window_end_sec", n_windows)
    center_sec = 0.5 * (start + end)
    kept_time = center_sec[kept]
    dt = _nominal_dt(kept_time, config)
    x = np.asarray(state[kept], dtype=np.float64)
    x_dot = estimate_derivatives_with_time_gaps(
        x,
        dt,
        kept_time,
        method=method,
        window=window,
        polyorder=polyorder,
        gap_tol=float(derivative_robust.get("gap_tol", 0.25)),
        use_segmented=bool(derivative_robust.get("enabled", True)),
        max_jump=float(derivative_robust.get("max_jump", 5.0)),
        min_seg=int(derivative_robust.get("min_seg", 9)),
    )
    finite_velocity = np.isfinite(x_dot).all(axis=1)
    if int(np.count_nonzero(finite_velocity)) < 2:
        return {
            **base,
            "computation_status": "insufficient_support",
            "failure_reason": "derivative_support_below_minimum_segment",
            "dt_sec": float(dt),
            "derivative_method": method,
        }

    segment_id = jacobian_segment_ids(int(x.shape[0]), kept_time, float(dt))
    nn_indices = build_knn_indices(x, k=knn_k, metric=knn_metric, whiten=whiten)
    jac_res = estimate_local_jacobians(
        x,
        x_dot,
        nn_indices,
        super_window=super_window,
        ridge_alpha=ridge_alpha,
        distance_weighted=distance_weighted,
        j_dot_dt=float(dt),
        knn_k=knn_k,
        require_determined_support=bool(block.get("require_determined_support", False)),
        forbid_cross_gap=True,
        segment_id=segment_id,
    )
    jac_res, policy = apply_standard_jacobian_window_policy(jac_res)
    j_hat = np.asarray(jac_res.j_hat, dtype=np.float32) if jac_res is not None else np.zeros((0, 9, 9), dtype=np.float32)
    centers_local = np.asarray(jac_res.centers, dtype=np.int32) if jac_res is not None else np.zeros((0,), dtype=np.int32)
    if j_hat.ndim != 3 or j_hat.shape[0] == 0 or centers_local.size == 0:
        reason = str((jac_res.diagnostics or {}).get("computation_status_reason") or "no_retained_jacobian_windows") if jac_res is not None else "jacobian_not_returned"
        return {
            **base,
            "computation_status": "not_testable",
            "failure_reason": reason,
            "dt_sec": float(dt),
            "derivative_method": method,
            "policy": _json_policy(policy),
        }

    original_centers = kept[np.clip(centers_local, 0, kept.size - 1)].astype(np.int32)
    j_dot = np.asarray(jac_res.j_dot, dtype=np.float32)
    if j_hat.shape[0] <= 1:
        j_dot = np.full_like(j_hat, np.nan, dtype=np.float32)
    metrics = compute_jacobian_metrics(
        j_hat,
        nominal_dt_sec=float(dt),
    )
    diagnostics = jac_res.diagnostics if isinstance(jac_res.diagnostics, Mapping) else {}
    return {
        **base,
        "computation_status": "computed",
        "failure_reason": None,
        "window_index": original_centers,
        "window_center_sec": center_sec[original_centers].astype(np.float64),
        "J_hat": j_hat,
        "J_dot": j_dot,
        "derived_metrics": metrics,
        "dt_sec": float(dt),
        "derivative_method": method,
        "derivative_window": window,
        "derivative_polyorder": polyorder,
        "knn_k": knn_k,
        "super_window": super_window,
        "ridge_alpha": float(ridge_alpha),
        "distance_weighted": distance_weighted,
        "whiten": whiten,
        "require_determined_support": bool(block.get("require_determined_support", False)),
        "ridge_underdetermined": bool(diagnostics.get("ridge_underdetermined", False)),
        "n_centers": int(j_hat.shape[0]),
        "metrics_schema": str(metrics.get("schema", METRICS_SCHEMA)),
        "policy": _json_policy(policy),
    }


def _select_state(
    payload: Mapping[str, Any],
    names: list[str],
) -> tuple[np.ndarray, np.ndarray, str, str]:
    raw_group = payload.get("raw_coordinates") if isinstance(payload.get("raw_coordinates"), Mapping) else {}
    raw = np.asarray(raw_group.get("values"), dtype=np.float64) if raw_group.get("values") is not None else np.zeros((0, 0))
    support = raw_group.get("support")
    if support is None:
        raw_valid = np.isfinite(raw).all(axis=1) if raw.ndim == 2 else np.zeros((0,), dtype=bool)
    else:
        support_arr = np.asarray(support, dtype=bool)
        raw_valid = support_arr.all(axis=1) & np.isfinite(raw).all(axis=1) if raw.ndim == 2 else np.zeros((0,), dtype=bool)

    anchored_group = payload.get("coordinates") if isinstance(payload.get("coordinates"), Mapping) else {}
    anchored = (
        np.asarray(anchored_group.get("values"), dtype=np.float64)
        if anchored_group.get("values") is not None
        else np.zeros((0, 0))
    )
    complete = np.asarray(payload.get("complete_9d_mask"), dtype=bool).reshape(-1)
    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), Mapping) else {}
    anchor_status = str(provenance.get("anchor_application_status") or "")
    anchored_ok = (
        anchor_status in _ANCHOR_APPLIED
        and anchored.ndim == 2
        and anchored.shape[1] == len(names)
        and complete.size == anchored.shape[0]
        and bool(np.any(complete))
    )
    if anchored_ok:
        valid = complete & np.isfinite(anchored).all(axis=1)
        state = np.array(anchored, dtype=np.float64, copy=True)
        state[~valid] = np.nan
        return state, valid, "v3_4_anchored", "dimensionless_reference_scores"
    state = np.array(raw, dtype=np.float64, copy=True) if raw.ndim == 2 else np.zeros((0, 0), dtype=np.float64)
    if state.ndim == 2 and raw_valid.shape[0] == state.shape[0]:
        state[~raw_valid] = np.nan
    else:
        raw_valid = np.isfinite(state).all(axis=1) if state.ndim == 2 else np.zeros((0,), dtype=bool)
    return state, raw_valid, "v3_4_raw", "mixed_source_units"


def _window_time(payload: Mapping[str, Any], key: str, n_windows: int) -> np.ndarray:
    values = payload.get(key)
    if values is None:
        return np.full((n_windows,), np.nan, dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size != n_windows:
        return np.full((n_windows,), np.nan, dtype=np.float64)
    return arr


def _nominal_dt(times: np.ndarray, config: Mapping[str, Any]) -> float:
    series = np.asarray(times, dtype=np.float64).reshape(-1)
    if series.size >= 2:
        step = np.diff(series)
        positive = step[np.isfinite(step) & (step > 0)]
        if positive.size:
            return float(np.median(positive))
    surface = config.get("measurement_surface") if isinstance(config.get("measurement_surface"), Mapping) else {}
    declared = surface.get("window_step_sec") if isinstance(surface, Mapping) else None
    try:
        value = float(declared)
    except (TypeError, ValueError):
        value = 4.0
    return value if np.isfinite(value) and value > 0 else 4.0


def _json_policy(policy: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(policy, Mapping):
        return {}
    out: dict[str, Any] = {}
    for key, value in policy.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[str(key)] = value
        elif isinstance(value, np.generic):
            out[str(key)] = value.item()
        elif isinstance(value, Mapping):
            out[str(key)] = {str(k): (v.item() if isinstance(v, np.generic) else v) for k, v in value.items()}
    return out
