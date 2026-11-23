"""Experimental finite-lag dynamics on separate observed LFP charts.

This adapter consumes the contact-supported candidate export and emits a
separate LFP chart/dynamics HDF5. It does not use MNPS 3.4's canonical 9D
coordinate or Jacobian/one-step adapters.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


CHART_SCHEMA = "mndm.experimental_lfp_observed_chart.v1"
DYNAMICS_SCHEMA = "mndm.experimental_lfp_finite_lag_dynamics.v1"
_VALUE_FIELDS = (
    "spectral_centroid_hz",
    "log10_group_bandpower_v2",
    "envelope_cv",
)


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_hash(value: Mapping[str, Any]) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _decode_vector(values: Any) -> np.ndarray:
    return np.asarray([_text(value) for value in np.asarray(values).reshape(-1)], dtype=object)


def build_observed_chart(
    *,
    centroid_hz: np.ndarray,
    bandpower_v2: np.ndarray,
    envelope_cv: np.ndarray,
    support_pass: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Build a per-group 3D chart and its declared session-local calibration."""
    centroid = np.asarray(centroid_hz, dtype=np.float64)
    power = np.asarray(bandpower_v2, dtype=np.float64)
    envelope = np.asarray(envelope_cv, dtype=np.float64)
    if centroid.ndim != 1 or power.shape != centroid.shape or envelope.shape != centroid.shape:
        raise ValueError("Each group metric must be a one-dimensional vector with a shared time axis.")
    n = centroid.size
    for metric_name in ("centroid_hz", "bandpower_v2", "envelope_cv"):
        mask = np.asarray(support_pass[metric_name], dtype=bool).reshape(-1)
        if mask.size != n:
            raise ValueError(f"Support mask {metric_name!r} does not match the group time axis.")

    raw = np.column_stack((centroid, np.full(n, np.nan), envelope))
    positive_power = np.isfinite(power) & (power > 0.0)
    raw[positive_power, 1] = np.log10(power[positive_power] / 1.0)
    component_support = np.column_stack(
        (
            np.asarray(support_pass["centroid_hz"], dtype=bool).reshape(-1),
            np.asarray(support_pass["bandpower_v2"], dtype=bool).reshape(-1) & positive_power,
            np.asarray(support_pass["envelope_cv"], dtype=bool).reshape(-1),
        )
    )
    raw_finite = np.isfinite(raw)
    calibration_mask = component_support.all(axis=1) & raw_finite.all(axis=1)
    n_components = raw.shape[1]
    median = np.full(n_components, np.nan, dtype=np.float64)
    scale = np.full(n_components, np.nan, dtype=np.float64)
    scale_valid = np.zeros(n_components, dtype=bool)
    if calibration_mask.any():
        calibration_values = raw[calibration_mask]
        median = np.median(calibration_values, axis=0)
        mad = np.median(np.abs(calibration_values - median[None, :]), axis=0)
        scale = 1.4826 * mad
        scale_valid = np.isfinite(scale) & (scale > 0.0)

    scaled = np.full_like(raw, np.nan, dtype=np.float64)
    if scale_valid.all():
        scaled = (raw - median[None, :]) / scale[None, :]
    chart_complete = (
        component_support.all(axis=1)
        & raw_finite.all(axis=1)
        & scale_valid.all()
        & np.isfinite(scaled).all(axis=1)
    )
    scaled[~chart_complete] = np.nan
    missing_reason = np.full(n, "", dtype=object)
    for row in range(n):
        if not np.asarray(support_pass["centroid_hz"], dtype=bool)[row]:
            missing_reason[row] = "centroid_insufficient_group_support"
        elif not np.asarray(support_pass["bandpower_v2"], dtype=bool)[row]:
            missing_reason[row] = "bandpower_insufficient_group_support"
        elif not np.isfinite(power[row]):
            missing_reason[row] = "nonfinite_group_bandpower"
        elif power[row] <= 0.0:
            missing_reason[row] = "zero_power_log_unsupported"
        elif not np.asarray(support_pass["envelope_cv"], dtype=bool)[row]:
            missing_reason[row] = "envelope_cv_insufficient_group_support"
        elif not raw_finite[row].all():
            missing_reason[row] = "nonfinite_chart_component"
        elif not calibration_mask.any():
            missing_reason[row] = "no_calibration_support"
        elif not scale_valid.all():
            missing_reason[row] = "zero_or_nonfinite_robust_scale"
        elif not chart_complete[row]:
            missing_reason[row] = "unsupported_chart_window"

    return {
        "raw_values": raw,
        "values": scaled,
        "component_support": component_support,
        "complete_mask": chart_complete,
        "missing_reason": missing_reason,
        "calibration_median": median,
        "calibration_scale": scale,
        "calibration_scale_valid": scale_valid,
        "calibration_window_count": int(calibration_mask.sum()),
        "candidate_window_count": int(n),
        "coordinate_names": list(_VALUE_FIELDS),
    }


def derive_processing_batch_ids(
    source_sample_start: np.ndarray,
    *,
    sample_rate_hz: float,
    batch_duration_sec: float = 60.0,
) -> tuple[np.ndarray, int]:
    """Reconstruct the frozen candidate exporter batch assignment."""
    starts = np.asarray(source_sample_start, dtype=np.int64).reshape(-1)
    batch_samples = int(round(float(batch_duration_sec) * float(sample_rate_hz)))
    if batch_samples <= 0:
        raise ValueError("Processing batch sample count must be positive.")
    if np.any(starts < 0):
        raise ValueError("Source sample starts must be nonnegative.")
    return (starts // batch_samples).astype(np.int32), batch_samples


def _all_path(mask: np.ndarray, start: int, lag: int) -> bool:
    return bool(np.all(mask[start : start + lag + 1]))


def _all_same_text(values: np.ndarray, start: int, lag: int) -> bool:
    path = values[start : start + lag + 1]
    return bool(path.size == lag + 1 and path[0] and np.all(path == path[0]))


def estimate_finite_lag_group(
    *,
    chart: np.ndarray,
    complete_mask: np.ndarray,
    center_time_sec: np.ndarray,
    behavioral_state: Sequence[Any],
    stim_adjacent: np.ndarray,
    stim_contains_onset: np.ndarray,
    induction_excluded: np.ndarray,
    segment_id: Sequence[Any],
    processing_batch_id: np.ndarray,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Estimate finite-lag conditional mean and covariance rates for one group."""
    x = np.asarray(chart, dtype=np.float64)
    valid = np.asarray(complete_mask, dtype=bool).reshape(-1)
    time = np.asarray(center_time_sec, dtype=np.float64).reshape(-1)
    state = np.asarray([_text(value) for value in behavioral_state], dtype=object).reshape(-1)
    segment = np.asarray([_text(value) for value in segment_id], dtype=object).reshape(-1)
    batch = np.asarray(processing_batch_id, dtype=np.int32).reshape(-1)
    stim = np.asarray(stim_adjacent, dtype=bool).reshape(-1)
    onset = np.asarray(stim_contains_onset, dtype=bool).reshape(-1)
    induction = np.asarray(induction_excluded, dtype=bool).reshape(-1)
    n = valid.size
    if x.ndim != 2 or x.shape[0] != n or time.size != n:
        raise ValueError("Chart, validity, and time vectors must share a window axis.")
    if any(values.size != n for values in (state, segment, batch, stim, onset, induction)):
        raise ValueError("Annotation, segment, and batch arrays must align with the chart.")
    valid &= np.isfinite(x).all(axis=1) & np.isfinite(time)
    labels = sorted({str(value) for value in state if str(value)})
    estimators = spec["estimands"]
    lag_settings = spec["lag_policy"]
    neighborhood = spec["neighborhood"]
    eligibility = spec["transition_eligibility"]
    block_settings = spec["envelope_processing_block_policy"]
    max_rel_dt = float(lag_settings["max_relative_tau_deviation_from_realized_median"])
    min_rows = int(neighborhood["minimum_eligible_transitions_per_behavior_state_and_group"])
    k_max = int(neighborhood["k_unique_source_batches"])
    k_min = int(neighborhood["minimum_unique_source_batches"])
    exclusion_batch_distance = int(block_settings["minimum_absolute_batch_id_difference_from_query"])
    lags = [int(value) for value in lag_settings["predeclared_sensitivity_lag_steps"]]
    if not bool(eligibility["no_crossing_of_missing_windows_or_source_timestamp_gaps"]):
        raise ValueError("This estimator version requires gap-safe transitions.")

    query_safe = valid & (state != "") & ~stim & ~onset & ~induction
    state_results: dict[str, Any] = {}
    transition_summary: dict[str, Any] = {}
    for state_label in labels:
        state_query = query_safe & (state == state_label)
        state_result: dict[str, Any] = {"behavioral_state": state_label, "lags": {}}
        for lag in lags:
            candidate_rows: list[int] = []
            candidate_dts: list[float] = []
            ineligible_counts = {
                "unsupported_chart_or_missing_window": 0,
                "stim_or_induction_excluded": 0,
                "behavior_state_boundary": 0,
                "segment_boundary": 0,
                "processing_batch_boundary": 0,
                "invalid_timestamp_step": 0,
                "materially_irregular_lag": 0,
            }
            for source_idx in range(0, n - lag):
                if state[source_idx] != state_label:
                    continue
                if not _all_path(valid, source_idx, lag):
                    ineligible_counts["unsupported_chart_or_missing_window"] += 1
                    continue
                if not _all_path(~(stim | onset | induction), source_idx, lag):
                    ineligible_counts["stim_or_induction_excluded"] += 1
                    continue
                if not _all_same_text(state, source_idx, lag):
                    ineligible_counts["behavior_state_boundary"] += 1
                    continue
                if not _all_same_text(segment, source_idx, lag):
                    ineligible_counts["segment_boundary"] += 1
                    continue
                if not np.all(batch[source_idx : source_idx + lag + 1] == batch[source_idx]):
                    ineligible_counts["processing_batch_boundary"] += 1
                    continue
                path_dt = np.diff(time[source_idx : source_idx + lag + 1])
                if path_dt.size != lag or not np.isfinite(path_dt).all() or np.any(path_dt <= 0.0):
                    ineligible_counts["invalid_timestamp_step"] += 1
                    continue
                if np.any(np.abs(path_dt - np.median(path_dt)) / np.median(path_dt) > max_rel_dt):
                    ineligible_counts["materially_irregular_lag"] += 1
                    continue
                dt = float(time[source_idx + lag] - time[source_idx])
                candidate_rows.append(source_idx)
                candidate_dts.append(dt)

            source_idx_arr = np.asarray(candidate_rows, dtype=np.int32)
            dts = np.asarray(candidate_dts, dtype=np.float64)
            tau = float(np.median(dts)) if dts.size else float("nan")
            if dts.size and np.isfinite(tau) and tau > 0.0:
                rel_dev = np.abs(dts - tau) / tau
                keep = np.isfinite(rel_dev) & (rel_dev <= max_rel_dt)
                ineligible_counts["materially_irregular_lag"] += int(np.count_nonzero(~keep))
                source_idx_arr = source_idx_arr[keep]
                dts = dts[keep]
                tau = float(np.median(dts)) if dts.size else float("nan")

            drift = np.full((n, x.shape[1]), np.nan, dtype=np.float64)
            covariance = np.full((n, x.shape[1], x.shape[1]), np.nan, dtype=np.float64)
            support_count = np.zeros(n, dtype=np.int32)
            radius = np.full(n, np.nan, dtype=np.float64)
            reason = np.full(n, "", dtype=object)
            selected_source = np.full((n, k_max), -1, dtype=np.int32)
            selected_batch = np.full((n, k_max), -1, dtype=np.int32)
            selected_distance = np.full((n, k_max), np.nan, dtype=np.float64)

            if source_idx_arr.size < min_rows or not np.isfinite(tau) or tau <= 0.0:
                reason[state_query] = "insufficient_state_transition_support"
            else:
                increments = x[source_idx_arr + lag] - x[source_idx_arr]
                transition_states = x[source_idx_arr]
                source_batches = batch[source_idx_arr]
                transition_is_finite = np.isfinite(increments).all(axis=1) & np.isfinite(transition_states).all(axis=1)
                source_idx_arr = source_idx_arr[transition_is_finite]
                increments = increments[transition_is_finite]
                transition_states = transition_states[transition_is_finite]
                source_batches = source_batches[transition_is_finite]
                for query_idx in np.flatnonzero(state_query):
                    allowed = (
                        (np.abs(source_batches.astype(np.int64) - int(batch[query_idx])) >= exclusion_batch_distance)
                        & (segment[source_idx_arr] == segment[query_idx])
                    )
                    candidate_positions = np.flatnonzero(allowed)
                    if candidate_positions.size == 0:
                        reason[query_idx] = "insufficient_local_support"
                        continue
                    delta_state = transition_states[candidate_positions] - x[query_idx][None, :]
                    distances = np.linalg.norm(delta_state, axis=1)
                    # Keep only the closest transition from each non-overlapping source batch.
                    source_windows = source_idx_arr[candidate_positions]
                    candidate_batches = source_batches[candidate_positions]
                    order = np.lexsort((source_windows, distances))
                    ordered_batches = candidate_batches[order]
                    _, first_for_batch = np.unique(ordered_batches, return_index=True)
                    selected_order_positions = np.sort(first_for_batch)
                    selected_local_positions = order[selected_order_positions[:k_max]]
                    if selected_local_positions.size < k_min:
                        support_count[query_idx] = int(selected_local_positions.size)
                        reason[query_idx] = "insufficient_local_support"
                        continue
                    local_positions = candidate_positions[selected_local_positions]
                    distances_kept = distances[selected_local_positions]
                    selected_indices = source_idx_arr[local_positions]
                    delta = increments[local_positions]
                    n_support = int(delta.shape[0])
                    support_count[query_idx] = n_support
                    radius[query_idx] = float(np.max(distances_kept))
                    selected_source[query_idx, :n_support] = selected_indices
                    selected_batch[query_idx, :n_support] = source_batches[local_positions]
                    selected_distance[query_idx, :n_support] = distances_kept
                    mean_increment = np.mean(delta, axis=0)
                    drift[query_idx] = mean_increment / tau
                    centered = delta - mean_increment[None, :]
                    covariance[query_idx] = (centered.T @ centered) / ((n_support - 1) * tau)

            lag_result = {
                "lag_steps": lag,
                "tau_sec": tau,
                "n_eligible_transitions_in_state": int(source_idx_arr.size),
                "eligible_source_transition_index": source_idx_arr,
                "eligible_source_batch_id": batch[source_idx_arr],
                "ineligible_transition_counts": ineligible_counts,
                "query_window_index": np.arange(n, dtype=np.int32),
                "query_valid": state_query.astype(np.uint8),
                "drift": drift,
                "covariance_rate": covariance,
                "support_count": support_count,
                "local_radius": radius,
                "missing_reason": reason,
                "selected_source_transition_index": selected_source,
                "selected_source_batch_id": selected_batch,
                "selected_source_distance": selected_distance,
            }
            state_result["lags"][f"lag_{lag}"] = lag_result
            transition_summary[f"{state_label}|lag_{lag}"] = {
                "eligible_transitions": int(source_idx_arr.size),
                "tau_sec": tau,
                "query_windows": int(state_query.sum()),
                "computed_query_windows": int(np.isfinite(drift).all(axis=1).sum()),
                "insufficient_local_support": int(np.count_nonzero(reason == "insufficient_local_support")),
                "ineligible_transition_counts": ineligible_counts,
            }
        state_results[state_label] = state_result
    return {
        "state_results": state_results,
        "transition_summary": transition_summary,
        "n_windows": n,
        "n_complete_chart_windows": int(valid.sum()),
        "n_query_safe_windows": int(query_safe.sum()),
        "processing_batch_ids": batch,
    }


def _create_dataset(group: h5py.Group, name: str, values: Any) -> None:
    if isinstance(values, Mapping):
        values = json.dumps(dict(values), ensure_ascii=False, sort_keys=True)
    array = np.asarray(values)
    kwargs: dict[str, Any] = {}
    if array.ndim > 0 and array.size:
        kwargs.update(compression="gzip", compression_opts=4, shuffle=True)
    if array.dtype.kind in {"U", "O"}:
        flat = ["" if value is None else _text(value) for value in array.reshape(-1)]
        data = np.asarray(flat, dtype=object).reshape(array.shape)
        group.create_dataset(name, data=data, dtype=h5py.string_dtype("utf-8"), **kwargs)
    else:
        group.create_dataset(name, data=array, **kwargs)


def write_lfp_dynamics_h5(
    output_path: str | Path,
    *,
    groups: Mapping[str, Mapping[str, Any]],
    chart_spec: Mapping[str, Any],
    dynamics_spec: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Path:
    """Write the separate experimental chart and finite-lag HDF5 overlay."""
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite LFP dynamics export: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(target, "x") as handle:
        handle.attrs["schema"] = DYNAMICS_SCHEMA
        handle.attrs["chart_id"] = str(chart_spec["chart_id"])
        handle.attrs["estimator_id"] = str(dynamics_spec["estimator_id"])
        handle.attrs["coordinate_dimension"] = 3
        contracts = handle.create_group("contracts")
        _create_dataset(contracts, "chart_spec_json", json.dumps(chart_spec, ensure_ascii=False))
        _create_dataset(contracts, "dynamics_spec_json", json.dumps(dynamics_spec, ensure_ascii=False))
        _create_dataset(handle, "provenance_json", json.dumps(dict(provenance), ensure_ascii=False))
        root = handle.create_group("groups")
        for group_id, result in groups.items():
            group = root.create_group(str(group_id))
            chart = group.create_group("chart")
            for key in (
                "coordinate_names",
                "raw_values",
                "values",
                "component_support",
                "complete_mask",
                "missing_reason",
                "calibration_median",
                "calibration_scale",
                "calibration_scale_valid",
            ):
                _create_dataset(chart, key, result["chart"][key])
            _create_dataset(chart, "center_time_sec", result["center_time_sec"])
            _create_dataset(chart, "window_index", np.arange(result["chart"]["values"].shape[0], dtype=np.int32))
            _create_dataset(chart, "source_sample_start", result["source_sample_start"])
            _create_dataset(chart, "segment_id", result["segment_id"])
            _create_dataset(chart, "processing_batch_id", result["dynamics"]["processing_batch_ids"])
            _create_dataset(chart, "behavioral_state", result["behavioral_state"])
            _create_dataset(chart, "stim_adjacent", result["stim_adjacent"])
            _create_dataset(chart, "stim_contains_onset", result["stim_contains_onset"])
            _create_dataset(chart, "induction_excluded", result["induction_excluded"])
            dynamic_root = group.create_group("dynamical_families")
            for state_index, (state_label, state_result) in enumerate(result["dynamics"]["state_results"].items()):
                state_group = dynamic_root.create_group(f"state_{state_index:03d}")
                state_group.attrs["behavioral_state"] = state_label
                for lag_name, lag_result in state_result["lags"].items():
                    lag_group = state_group.create_group(lag_name)
                    for key, value in lag_result.items():
                        _create_dataset(lag_group, key, value)
            _create_dataset(
                group,
                "transition_summary_json",
                json.dumps(result["dynamics"]["transition_summary"], ensure_ascii=False),
            )
        handle.flush()
    return target


def run_lfp_finite_lag_dynamics(
    *,
    candidate_h5_path: str | Path,
    chart_spec_path: str | Path,
    dynamics_spec_path: str | Path,
    output_h5_path: str | Path,
) -> dict[str, Any]:
    """Build group-specific charts and finite-lag measurements from one candidate export."""
    source_path = Path(candidate_h5_path)
    chart_path = Path(chart_spec_path)
    dynamics_path = Path(dynamics_spec_path)
    chart_spec = json.loads(chart_path.read_text(encoding="utf-8"))
    dynamics_spec = json.loads(dynamics_path.read_text(encoding="utf-8"))
    if chart_spec.get("status") != "frozen_before_lfp_dynamics_computation":
        raise ValueError("LFP observed chart must be frozen before dynamics computation.")
    if dynamics_spec.get("status") != "frozen_before_lfp_dynamics_computation":
        raise ValueError("LFP finite-lag estimator must be frozen before computation.")
    if dynamics_spec.get("chart_spec_id") != chart_spec.get("chart_id"):
        raise ValueError("Dynamics spec refers to a different observed chart.")
    if dynamics_spec.get("candidate_dynamics_values_examined") is not False:
        raise ValueError("Dynamics spec must record candidate_dynamics_values_examined=false.")
    if chart_spec.get("candidate_values_used_to_select_chart") is not False:
        raise ValueError("Chart spec must record candidate_values_used_to_select_chart=false.")
    source_sha = _sha256_file(source_path)

    with h5py.File(source_path, "r") as handle:
        if _text(handle.attrs.get("schema", "")) != "mndm.experimental_lfp_mnps_3_4_candidates.v1":
            raise ValueError("Input is not the expected frozen LFP candidate export schema.")
        embedded_spec = json.loads(_text(handle["contracts/estimator_spec_json"][()]))
        if embedded_spec.get("spec_id") != chart_spec["source_candidate_estimator_spec_id"]:
            raise ValueError("Candidate export and observed chart refer to different estimator specs.")
        group_ids = _decode_vector(handle["group_metrics/group_id"][:]).tolist()
        declared_groups = chart_spec["trajectory_policy"]["group_ids"]
        if group_ids != declared_groups:
            raise ValueError("Candidate group ordering differs from the frozen LFP chart.")
        source_samples = np.asarray(handle["windows/source_sample_start"][:], dtype=np.int64)
        start_time = np.asarray(handle["windows/source_time_start_sec"][:], dtype=np.float64)
        stop_time = np.asarray(handle["windows/source_time_stop_sec"][:], dtype=np.float64)
        center_time = (start_time + stop_time) / 2.0
        segment_id = _decode_vector(handle["windows/segment_id"][:])
        state = _decode_vector(handle["window_annotations/lfp_behavioral_state"][:])
        stim_adjacent = np.asarray(handle["window_annotations/lfp_stim_adjacent"][:], dtype=bool)
        stim_contains_onset = np.asarray(handle["window_annotations/lfp_stim_contains_onset"][:], dtype=bool)
        induction_excluded = np.asarray(handle["window_annotations/lfp_induction_excluded"][:], dtype=bool)
        fs = float(embedded_spec["timebase"]["sample_rate_hz"])
        batch_id, batch_samples = derive_processing_batch_ids(source_samples, sample_rate_hz=fs)
        output_groups: dict[str, Any] = {}
        summaries: dict[str, Any] = {}
        for group_index, group_id in enumerate(group_ids):
            chart = build_observed_chart(
                centroid_hz=handle["group_metrics/centroid_hz_median"][:, group_index],
                bandpower_v2=handle["group_metrics/bandpower_v2_median"][:, group_index],
                envelope_cv=handle["group_metrics/envelope_cv_median"][:, group_index],
                support_pass={
                    "centroid_hz": handle["group_metrics/centroid_hz_support_pass"][:, group_index],
                    "bandpower_v2": handle["group_metrics/bandpower_v2_support_pass"][:, group_index],
                    "envelope_cv": handle["group_metrics/envelope_cv_support_pass"][:, group_index],
                },
            )
            dynamics = estimate_finite_lag_group(
                chart=chart["values"],
                complete_mask=chart["complete_mask"],
                center_time_sec=center_time,
                behavioral_state=state,
                stim_adjacent=stim_adjacent,
                stim_contains_onset=stim_contains_onset,
                induction_excluded=induction_excluded,
                segment_id=segment_id,
                processing_batch_id=batch_id,
                spec=dynamics_spec,
            )
            output_groups[group_id] = {
                "chart": chart,
                "center_time_sec": center_time,
                "source_sample_start": source_samples,
                "segment_id": segment_id,
                "behavioral_state": state,
                "stim_adjacent": stim_adjacent,
                "stim_contains_onset": stim_contains_onset,
                "induction_excluded": induction_excluded,
                "dynamics": dynamics,
            }
            summaries[group_id] = {
                "chart_complete_window_count": chart["calibration_window_count"],
                "chart_scale_valid": bool(chart["calibration_scale_valid"].all()),
                "state_lag_summaries": dynamics["transition_summary"],
            }

    provenance = {
        "candidate_export_path": str(source_path.resolve()),
        "candidate_export_sha256": source_sha,
        "candidate_estimator_spec_id": chart_spec["source_candidate_estimator_spec_id"],
        "candidate_contact_set_sha256": str(embedded_spec["contact_groups"]["contact_set_sha256"]),
        "chart_spec_id": str(chart_spec["chart_id"]),
        "chart_spec_sha256": _sha256_file(chart_path),
        "dynamics_estimator_id": str(dynamics_spec["estimator_id"]),
        "dynamics_spec_sha256": _sha256_file(dynamics_path),
        "candidate_window_count": int(len(center_time)),
        "sample_rate_hz": fs,
        "window_center_step_median_sec": float(np.median(np.diff(center_time))),
        "window_center_step_min_sec": float(np.min(np.diff(center_time))),
        "window_center_step_max_sec": float(np.max(np.diff(center_time))),
        "processing_batch_samples": int(batch_samples),
        "processing_batch_duration_sec": float(batch_samples / fs),
        "processing_batch_count": int(np.unique(batch_id).size),
        "raw_window_support_independence": "not_established",
        "session_local_calibration_is_transductive": True,
        "heldout_calibration_policy": "training_only",
        "condition_comparisons_performed": False,
        "jacobians_computed": False,
        "one_step_computed": False,
        "coordinate_replacement_decision": False,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "h5py": h5py.__version__,
        },
    }
    written = write_lfp_dynamics_h5(
        output_h5_path,
        groups=output_groups,
        chart_spec=chart_spec,
        dynamics_spec=dynamics_spec,
        provenance=provenance,
    )
    with h5py.File(written, "r") as check:
        if _text(check.attrs["schema"]) != DYNAMICS_SCHEMA:
            raise RuntimeError("LFP dynamics HDF5 schema round-trip check failed.")
        if len(check["groups"]) != len(group_ids):
            raise RuntimeError("LFP dynamics HDF5 group-count round-trip check failed.")
        if _text(check["provenance_json"][()]) == "":
            raise RuntimeError("LFP dynamics HDF5 provenance round-trip check failed.")
    return {
        "output_path": str(written.resolve()),
        "schema": DYNAMICS_SCHEMA,
        "group_count": len(group_ids),
        "window_count": int(len(center_time)),
        "processing_batch_samples": int(batch_samples),
        "processing_batch_count": int(np.unique(batch_id).size),
        "group_summaries": summaries,
        "hdf5_roundtrip_verified": True,
    }


__all__ = [
    "CHART_SCHEMA",
    "DYNAMICS_SCHEMA",
    "build_observed_chart",
    "derive_processing_batch_ids",
    "estimate_finite_lag_group",
    "run_lfp_finite_lag_dynamics",
    "write_lfp_dynamics_h5",
]
