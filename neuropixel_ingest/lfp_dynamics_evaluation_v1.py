"""Blocked held-out utility screen for the experimental LFP finite-lag drift.

The screen uses per-fold training-only chart calibration and holds out entire
envelope-computation batches with an adjacent-batch embargo. It is deliberately
separate from the session-transductive descriptive dynamics export.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from neuropixel_ingest.lfp_dynamics_v1 import (
    _decode_vector,
    _sha256_file,
    _text,
    derive_processing_batch_ids,
)


def _fit_fold_chart(
    *,
    centroid: np.ndarray,
    power: np.ndarray,
    envelope: np.ndarray,
    support: np.ndarray,
    training_windows: np.ndarray,
) -> dict[str, Any]:
    positive_power = np.isfinite(power) & (power > 0.0)
    raw = np.column_stack(
        (
            centroid,
            np.where(positive_power, np.log10(power / 1.0), np.nan),
            envelope,
        )
    )
    component_support = np.asarray(support, dtype=bool).copy()
    component_support[:, 1] &= positive_power
    raw_finite = np.isfinite(raw)
    calibration_mask = training_windows & component_support.all(axis=1) & raw_finite.all(axis=1)
    median = np.full(3, np.nan, dtype=np.float64)
    scale = np.full(3, np.nan, dtype=np.float64)
    if calibration_mask.any():
        calibration_values = raw[calibration_mask]
        median = np.median(calibration_values, axis=0)
        scale = 1.4826 * np.median(np.abs(calibration_values - median[None, :]), axis=0)
    scale_valid = np.isfinite(scale) & (scale > 0.0)
    values = np.full_like(raw, np.nan)
    if scale_valid.all():
        values = (raw - median[None, :]) / scale[None, :]
    complete = component_support.all(axis=1) & raw_finite.all(axis=1) & scale_valid.all()
    values[~complete] = np.nan
    return {
        "raw": raw,
        "values": values,
        "complete": complete,
        "calibration_window_count": int(calibration_mask.sum()),
        "median": median,
        "scale": scale,
        "scale_valid": scale_valid,
    }


def _eligible_transition_starts(
    *,
    complete: np.ndarray,
    time: np.ndarray,
    state: np.ndarray,
    segment: np.ndarray,
    batch: np.ndarray,
    stim_adjacent: np.ndarray,
    stim_onset: np.ndarray,
    induction: np.ndarray,
    state_label: str,
    lag: int,
    mode: str,
    heldout_batch: int,
    max_relative_tau_deviation: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    n = complete.size
    candidates: list[int] = []
    candidate_dt: list[float] = []
    counts = Counter()
    for source_idx in np.flatnonzero(state[: n - lag] == state_label):
        path = slice(source_idx, source_idx + lag + 1)
        if not np.all(complete[path]):
            counts["incomplete_chart_path"] += 1
            continue
        if not np.all(state[path] == state_label) or not state_label:
            counts["behavior_state_boundary_or_missing_state"] += 1
            continue
        if not np.all(segment[path] == segment[source_idx]):
            counts["timestamp_segment_boundary"] += 1
            continue
        if not np.all(batch[path] == batch[source_idx]):
            counts["processing_batch_boundary"] += 1
            continue
        if np.any(stim_adjacent[path] | stim_onset[path] | induction[path]):
            counts["stimulation_or_induction_excluded"] += 1
            continue
        dts = np.diff(time[path])
        if dts.size != lag or not np.isfinite(dts).all() or np.any(dts <= 0.0):
            counts["invalid_timestamp_step"] += 1
            continue
        if np.any(np.abs(dts - np.median(dts)) / np.median(dts) > max_relative_tau_deviation):
            counts["irregular_lag_path"] += 1
            continue
        if mode == "train" and abs(int(batch[source_idx]) - heldout_batch) < 2:
            counts["context_embargo"] += 1
            continue
        if mode == "test" and int(batch[source_idx]) != heldout_batch:
            continue
        candidates.append(source_idx)
        candidate_dt.append(float(time[source_idx + lag] - time[source_idx]))

    source_idx_arr = np.asarray(candidates, dtype=np.int32)
    dt_arr = np.asarray(candidate_dt, dtype=np.float64)
    if dt_arr.size and mode != "all":
        tau = float(np.median(dt_arr))
        keep = np.isfinite(dt_arr) & (dt_arr > 0.0) & (
            np.abs(dt_arr - tau) / tau <= max_relative_tau_deviation
        )
        counts["irregular_lag_vs_realized_median"] += int(np.count_nonzero(~keep))
        source_idx_arr = source_idx_arr[keep]
        dt_arr = dt_arr[keep]
    return source_idx_arr, dt_arr, dict(counts)


def _nearest_unique_batch_positions(
    *,
    distances: np.ndarray,
    source_indices: np.ndarray,
    source_batches: np.ndarray,
    k_max: int,
) -> np.ndarray:
    order = np.lexsort((source_indices, distances))
    ordered_batches = source_batches[order]
    _, first_positions = np.unique(ordered_batches, return_index=True)
    selected_positions = np.sort(first_positions)[:k_max]
    return order[selected_positions]


def _mse_per_row(error: np.ndarray) -> np.ndarray:
    return np.mean(np.square(error), axis=1)


def _summarize_records(records: list[dict[str, Any]], *, min_test_batches: int) -> dict[str, Any]:
    if not records:
        return {
            "paired_transition_count": 0,
            "scorable_heldout_batch_count": 0,
            "status": "not_evaluable_no_paired_predictions",
        }
    by_batch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_batch[int(record["heldout_batch_id"])].append(record)
    model_names = ("full_chart", "bandpower_only", "persistence")
    block_mse: dict[str, list[float]] = {name: [] for name in model_names}
    for batch_id in sorted(by_batch):
        batch_records = by_batch[batch_id]
        for model_name in model_names:
            block_mse[model_name].append(
                float(np.mean([record[f"{model_name}_mse"] for record in batch_records]))
            )
    summary: dict[str, Any] = {
        "status": "evaluable" if len(by_batch) >= min_test_batches else "not_evaluable_too_few_heldout_batches",
        "paired_transition_count": len(records),
        "scorable_heldout_batch_count": len(by_batch),
        "heldout_batch_ids": sorted(by_batch),
        "macro_mean_batch_mse": {
            model_name: float(np.mean(block_mse[model_name])) for model_name in model_names
        },
        "fraction_batches_full_chart_beats": {
            baseline: float(
                np.mean(
                    np.asarray(block_mse["full_chart"]) < np.asarray(block_mse[baseline])
                )
            )
            for baseline in ("bandpower_only", "persistence")
        },
        "per_coordinate_rmse": {
            model_name: np.sqrt(
                np.mean(
                    np.asarray([record[f"{model_name}_squared_error_by_coordinate"] for record in records]),
                    axis=0,
                )
            ).tolist()
            for model_name in model_names
        },
        "per_coordinate_mae": {
            model_name: np.mean(
                np.asarray([record[f"{model_name}_absolute_error_by_coordinate"] for record in records]),
                axis=0,
            ).tolist()
            for model_name in model_names
        },
        "median_full_chart_support_count": float(np.median([r["full_chart_support_count"] for r in records])),
        "median_full_chart_neighborhood_radius": float(np.median([r["full_chart_radius"] for r in records])),
    }
    summary["jacobian_followup_stratum_screen_pass"] = bool(
        summary["status"] == "evaluable"
        and all(
            summary["macro_mean_batch_mse"]["full_chart"] < summary["macro_mean_batch_mse"][baseline]
            and summary["fraction_batches_full_chart_beats"][baseline] >= 0.60
            for baseline in ("bandpower_only", "persistence")
        )
    )
    return summary


def evaluate_lfp_dynamics_blocked_v1(
    *,
    candidate_h5_path: str | Path,
    chart_spec_path: str | Path,
    dynamics_spec_path: str | Path,
    evaluation_plan_path: str | Path,
    output_json_path: str | Path,
) -> dict[str, Any]:
    """Evaluate held-out local-mean increments under the frozen blocked plan."""
    candidate_path = Path(candidate_h5_path)
    chart_spec_path = Path(chart_spec_path)
    dynamics_spec_path = Path(dynamics_spec_path)
    plan_path = Path(evaluation_plan_path)
    output_path = Path(output_json_path)
    chart_spec = json.loads(chart_spec_path.read_text(encoding="utf-8"))
    dynamics_spec = json.loads(dynamics_spec_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "frozen_before_lfp_dynamics_computation":
        raise ValueError("Blocked evaluation plan must be frozen before dynamics computation.")
    if plan.get("candidate_dynamics_values_examined") is not False:
        raise ValueError("Evaluation plan must state that candidate dynamics were not examined at freeze.")
    if plan.get("chart_spec_id") != chart_spec.get("chart_id"):
        raise ValueError("Blocked evaluation plan refers to a different chart spec.")
    if plan.get("dynamics_estimator_id") != dynamics_spec.get("estimator_id"):
        raise ValueError("Blocked evaluation plan refers to a different dynamics spec.")
    if chart_spec.get("status") != "frozen_before_lfp_dynamics_computation" or dynamics_spec.get("status") != "frozen_before_lfp_dynamics_computation":
        raise ValueError("Chart and dynamics specs must be frozen before evaluation.")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite LFP blocked evaluation report: {output_path}")

    lag_settings = dynamics_spec["lag_policy"]
    neighborhood = dynamics_spec["neighborhood"]
    max_rel_tau = float(lag_settings["max_relative_tau_deviation_from_realized_median"])
    lags = [int(v) for v in lag_settings["predeclared_sensitivity_lag_steps"]]
    k_max = int(neighborhood["k_unique_source_batches"])
    k_min = int(neighborhood["minimum_unique_source_batches"])
    min_train_transitions = int(neighborhood["minimum_eligible_transitions_per_behavior_state_and_group"])
    min_test_batches = int(plan["metrics"]["minimum_scorable_heldout_batches"])

    with h5py.File(candidate_path, "r") as handle:
        if _text(handle.attrs.get("schema", "")) != "mndm.experimental_lfp_mnps_3_4_candidates.v1":
            raise ValueError("Input is not the expected frozen LFP candidate export schema.")
        embedded_spec = json.loads(_text(handle["contracts/estimator_spec_json"][()]))
        if embedded_spec.get("spec_id") != chart_spec["source_candidate_estimator_spec_id"]:
            raise ValueError("Candidate export and chart spec do not match.")
        group_ids = _decode_vector(handle["group_metrics/group_id"][:]).tolist()
        if group_ids != chart_spec["trajectory_policy"]["group_ids"]:
            raise ValueError("Candidate depth-group order differs from the frozen chart.")
        source_samples = np.asarray(handle["windows/source_sample_start"][:], dtype=np.int64)
        start_time = np.asarray(handle["windows/source_time_start_sec"][:], dtype=np.float64)
        stop_time = np.asarray(handle["windows/source_time_stop_sec"][:], dtype=np.float64)
        time = (start_time + stop_time) / 2.0
        segment = _decode_vector(handle["windows/segment_id"][:])
        state = _decode_vector(handle["window_annotations/lfp_behavioral_state"][:])
        stim_adjacent = np.asarray(handle["window_annotations/lfp_stim_adjacent"][:], dtype=bool)
        stim_onset = np.asarray(handle["window_annotations/lfp_stim_contains_onset"][:], dtype=bool)
        induction = np.asarray(handle["window_annotations/lfp_induction_excluded"][:], dtype=bool)
        fs = float(embedded_spec["timebase"]["sample_rate_hz"])
        batch, batch_samples = derive_processing_batch_ids(source_samples, sample_rate_hz=fs)
        fold_ids = np.unique(batch)
        states = sorted({str(v) for v in state if str(v)})
        all_summaries: dict[str, Any] = {}
        all_failures: dict[str, Any] = {}

        for group_index, group_id in enumerate(group_ids):
            centroid = np.asarray(handle["group_metrics/centroid_hz_median"][:, group_index], dtype=np.float64)
            power = np.asarray(handle["group_metrics/bandpower_v2_median"][:, group_index], dtype=np.float64)
            envelope = np.asarray(handle["group_metrics/envelope_cv_median"][:, group_index], dtype=np.float64)
            support = np.column_stack(
                (
                    handle["group_metrics/centroid_hz_support_pass"][:, group_index],
                    handle["group_metrics/bandpower_v2_support_pass"][:, group_index],
                    handle["group_metrics/envelope_cv_support_pass"][:, group_index],
                )
            ).astype(bool)
            records: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
            failures: Counter[str] = Counter()
            transition_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
            for state_label in states:
                for lag in lags:
                    cached_indices, cached_dts, _ = _eligible_transition_starts(
                        complete=np.ones(len(time), dtype=bool),
                        time=time,
                        state=state,
                        segment=segment,
                        batch=batch,
                        stim_adjacent=stim_adjacent,
                        stim_onset=stim_onset,
                        induction=induction,
                        state_label=state_label,
                        lag=lag,
                        mode="all",
                        heldout_batch=-1,
                        max_relative_tau_deviation=max_rel_tau,
                    )
                    transition_cache[(state_label, lag)] = (cached_indices, cached_dts)
            for fold_batch in fold_ids:
                train_windows = np.abs(batch.astype(np.int64) - int(fold_batch)) >= 2
                fold_chart = _fit_fold_chart(
                    centroid=centroid,
                    power=power,
                    envelope=envelope,
                    support=support,
                    training_windows=train_windows,
                )
                if not fold_chart["scale_valid"].all():
                    failures["zero_or_nonfinite_training_scale"] += 1
                    continue
                x = fold_chart["values"]
                complete = fold_chart["complete"]
                for state_label in states:
                    for lag in lags:
                        all_indices, all_dts = transition_cache[(state_label, lag)]
                        train_keep = np.abs(batch[all_indices].astype(np.int64) - int(fold_batch)) >= 2
                        train_indices = all_indices[train_keep]
                        train_dts = all_dts[train_keep]
                        train_chart_complete = np.all(
                            np.column_stack([complete[train_indices + step] for step in range(lag + 1)]),
                            axis=1,
                        )
                        train_indices = train_indices[train_chart_complete]
                        train_dts = train_dts[train_chart_complete]
                        if train_dts.size:
                            provisional_tau = float(np.median(train_dts))
                            regular = np.abs(train_dts - provisional_tau) / provisional_tau <= max_rel_tau
                            train_indices = train_indices[regular]
                            train_dts = train_dts[regular]
                        if train_indices.size < min_train_transitions or train_dts.size == 0:
                            failures[f"{state_label}|lag_{lag}|insufficient_training_transitions"] += 1
                            continue
                        tau = float(np.median(train_dts))
                        test_batch_mask = batch[all_indices] == int(fold_batch)
                        test_indices = all_indices[test_batch_mask]
                        test_dts = all_dts[test_batch_mask]
                        test_chart_complete = np.all(
                            np.column_stack([complete[test_indices + step] for step in range(lag + 1)]),
                            axis=1,
                        )
                        test_indices = test_indices[test_chart_complete]
                        test_dts = test_dts[test_chart_complete]
                        if test_indices.size == 0:
                            failures[f"{state_label}|lag_{lag}|no_eligible_heldout_transition"] += 1
                            continue
                        test_keep = np.abs(test_dts - tau) / tau <= max_rel_tau
                        test_indices = test_indices[test_keep]
                        test_dts = test_dts[test_keep]
                        source_batches = batch[train_indices]
                        source_states = x[train_indices]
                        source_increments = x[train_indices + lag] - x[train_indices]
                        source_finite = np.isfinite(source_states).all(axis=1) & np.isfinite(source_increments).all(axis=1)
                        train_indices = train_indices[source_finite]
                        source_batches = source_batches[source_finite]
                        source_states = source_states[source_finite]
                        source_increments = source_increments[source_finite]
                        if train_indices.size < min_train_transitions:
                            failures[f"{state_label}|lag_{lag}|nonfinite_training_transitions"] += 1
                            continue

                        for query_idx, test_dt in zip(test_indices, test_dts):
                            allowed = segment[train_indices] == segment[query_idx]
                            local_batches = source_batches[allowed]
                            local_states = source_states[allowed]
                            local_increments = source_increments[allowed]
                            local_indices = train_indices[allowed]
                            if local_indices.size == 0:
                                failures[f"{state_label}|lag_{lag}|no_same_segment_training_source"] += 1
                                continue
                            full_distances = np.linalg.norm(local_states - x[query_idx][None, :], axis=1)
                            full_positions = _nearest_unique_batch_positions(
                                distances=full_distances,
                                source_indices=local_indices,
                                source_batches=local_batches,
                                k_max=k_max,
                            )
                            power_distances = np.abs(local_states[:, 1] - x[query_idx, 1])
                            power_positions = _nearest_unique_batch_positions(
                                distances=power_distances,
                                source_indices=local_indices,
                                source_batches=local_batches,
                                k_max=k_max,
                            )
                            if full_positions.size < k_min or power_positions.size < k_min:
                                failures[f"{state_label}|lag_{lag}|insufficient_local_source_batches"] += 1
                                continue
                            full_delta = local_increments[full_positions]
                            power_delta = local_increments[power_positions]
                            full_pred = (np.mean(full_delta, axis=0) / tau) * float(test_dt)
                            power_pred = (np.mean(power_delta, axis=0) / tau) * float(test_dt)
                            actual = x[query_idx + lag] - x[query_idx]
                            predictions = {
                                "full_chart": full_pred,
                                "bandpower_only": power_pred,
                                "persistence": np.zeros(3, dtype=np.float64),
                            }
                            record: dict[str, Any] = {
                                "heldout_batch_id": int(fold_batch),
                                "query_window_index": int(query_idx),
                                "full_chart_support_count": int(full_positions.size),
                                "full_chart_radius": float(np.max(full_distances[full_positions])),
                            }
                            for model_name, prediction in predictions.items():
                                error = prediction - actual
                                record[f"{model_name}_mse"] = float(np.mean(error**2))
                                record[f"{model_name}_squared_error_by_coordinate"] = (error**2).tolist()
                                record[f"{model_name}_absolute_error_by_coordinate"] = np.abs(error).tolist()
                            records[(state_label, lag)].append(record)

            group_summaries: dict[str, Any] = {}
            group_failures: dict[str, Any] = {}
            for state_label in states:
                for lag in lags:
                    key = f"{state_label}|lag_{lag}"
                    group_summaries[key] = _summarize_records(
                        records[(state_label, lag)], min_test_batches=min_test_batches
                    )
                    group_failures[key] = {
                        reason.split("|", 2)[-1]: count
                        for reason, count in failures.items()
                        if reason.startswith(f"{state_label}|lag_{lag}|")
                    }
            all_summaries[group_id] = group_summaries
            all_failures[group_id] = group_failures

    jacobian_screen: dict[str, Any] = {}
    per_group_passes: list[bool] = []
    for group_id, group_summaries in all_summaries.items():
        primary = [
            value
            for key, value in group_summaries.items()
            if key.endswith("|lag_1") and value.get("status") == "evaluable"
        ]
        group_pass = len(primary) >= 2 and all(
            bool(value.get("jacobian_followup_stratum_screen_pass")) for value in primary
        )
        per_group_passes.append(group_pass)
        jacobian_screen[group_id] = {
            "evaluable_primary_lag_state_strata": len(primary),
            "pass": group_pass,
        }
    jacobian_screen["overall_pass"] = bool(len(per_group_passes) == 2 and all(per_group_passes))

    report = {
        "schema": "mndm.experimental_lfp_blocked_dynamics_evaluation.v1",
        "evaluation_id": plan["evaluation_id"],
        "status": "completed",
        "candidate_export_path": str(candidate_path.resolve()),
        "candidate_export_sha256": _sha256_file(candidate_path),
        "candidate_export_spec_id": embedded_spec["spec_id"],
        "chart_spec_id": chart_spec["chart_id"],
        "chart_spec_sha256": _sha256_file(chart_spec_path),
        "dynamics_estimator_id": dynamics_spec["estimator_id"],
        "dynamics_spec_sha256": _sha256_file(dynamics_spec_path),
        "evaluation_plan_id": plan["evaluation_id"],
        "evaluation_plan_sha256": _sha256_file(plan_path),
        "candidate_window_count": int(len(time)),
        "processing_batch_count": int(len(fold_ids)),
        "processing_batch_samples": int(batch_samples),
        "processing_batch_duration_sec": float(batch_samples / fs),
        "session_local_full_data_scaling_used": False,
        "fold_calibration": "training batches only, excluding heldout batch and adjacent context-sharing batches",
        "envelope_features_recomputed": False,
        "raw_context_separation": "heldout and training compute blocks have disjoint recorded raw read intervals under the frozen exporter; longer neural temporal dependence is not ruled out",
        "directionality": "blocked interpolation; future training blocks may be used; not a causal forecast",
        "condition_comparisons_performed": False,
        "p_values_reported": False,
        "jacobians_computed": False,
        "jacobian_followup_screen": jacobian_screen,
        "group_state_lag_summaries": all_summaries,
        "unscored_fold_reason_counts": all_failures,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = ["evaluate_lfp_dynamics_blocked_v1"]
