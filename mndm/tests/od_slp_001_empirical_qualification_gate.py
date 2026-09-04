"""Run the audit-only OD-SLP-001 BOAS empirical qualification.

This runner reads paired BOAS source files, fits the already qualified 1-D
O2b adapter on the frozen DEV pid arm, and scores a frozen q-grid on HELD_OUT.
It never writes HDF5, creates an ingest overlay, or emits a production
orthogonal-dynamics artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from mndm.dynamical_families.committor import (
    estimate_committor_local_law_dense_grid_o2b,
)
from mndm.dynamical_families.sleep_committor_qualification import (
    COMPETING_OUTCOMES,
    FROZEN_PID_SPLIT,
    RESOLVED_OUTCOMES,
    assemble_adapter_arrays,
    assert_frozen_pid_split,
    binary_metrics,
    bootstrap_improvements,
    bootstrap_pairwise_metrics,
    build_n2_first_hit_segments,
    canonical_night_key,
    greedy_wrong_pid_map,
    permute_outcomes,
    resolved_scoring_rows,
    shift_reaction_coordinate,
)
from mndm.dynamical_families.sleep_first_hit_eligibility import (
    SleepFirstHitProtocol,
    audit_stage_intervals,
    json_safe,
)

try:
    from od_slp_000_first_hit_eligibility_gate import (  # type: ignore
        _channels_path,
        _clean_float,
        _event_geometry_matches,
        _event_path,
        _pair_acquisitions,
        _read_channels,
        _read_events,
        _read_headband_rc,
        _read_json,
        _load_pid_map,
        _sidecar_path,
    )
except ImportError:  # pragma: no cover - direct package import fallback
    from mndm.tests.od_slp_000_first_hit_eligibility_gate import (  # type: ignore
        _channels_path,
        _clean_float,
        _event_geometry_matches,
        _event_path,
        _pair_acquisitions,
        _read_channels,
        _read_events,
        _read_headband_rc,
        _read_json,
        _load_pid_map,
        _sidecar_path,
    )


GRID_RESOLUTION = 65
MIN_SUPPORT_PER_GRID = 64
MIN_TRANSITION_SEGMENTS = 20
MAX_DT_RELATIVE_DEVIATION = 0.05
BOOTSTRAP_REPLICATES = 2000
PRIMARY_TIME_SHIFT_INTERVALS = 120
SENSITIVITY_TIME_SHIFTS = (60, 180)
QUALIFICATION_RECORDS = ("OD-TQ2", "OD-TQ2b", "OD-TQ2b-R")


def _comparison_pass(result: Mapping[str, Any]) -> bool:
    return bool(
        result.get("status") == "computed"
        and float(result.get("point_relative_brier_improvement", -np.inf)) >= 0.05
        and float(result.get("point_log_loss_improvement", -np.inf)) >= 0.01
        and float(result.get("brier_lower_95", -np.inf)) > 0.0
        and float(result.get("log_loss_lower_95", -np.inf)) > 0.0
    )


def _clean_json_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Keep source metadata without serializing signal arrays."""
    return {
        "participant_id": record.get("participant_id"),
        "pid": record.get("pid"),
        "session": record.get("session"),
        "task": record.get("task"),
        "run": record.get("run"),
        "night_key": record.get("night_key"),
        "split": record.get("split"),
        "interval_count": record.get("interval_count"),
        "recording_duration_sec": record.get("recording_duration_sec"),
        "failure_reasons": list(record.get("failure_reasons", [])),
        "segment_count": len(record.get("segments", [])),
        "finite_segment_count": sum(
            bool(segment.get("external_rc_finite"))
            for segment in record.get("segments", [])
        ),
    }


def _load_night(
    pair: Mapping[str, Any],
    *,
    pid_map: Mapping[str, str | None],
    pid_to_split: Mapping[str, str],
    protocol: SleepFirstHitProtocol,
) -> dict[str, Any]:
    """Read and validate one paired source night."""
    psg_path = Path(pair["psg_path"])
    headband_path = Path(pair["headband_path"])
    record = {
        **dict(pair),
        "night_key": canonical_night_key(pair),
        "pid": pid_map.get(pair.get("participant_id")),
        "split": pid_to_split.get(str(pid_map.get(pair.get("participant_id")))),
        "failure_reasons": [],
    }
    reasons: list[str] = []
    psg_sidecar = _read_json(_sidecar_path(psg_path))
    headband_sidecar = _read_json(_sidecar_path(headband_path))
    psg_events = _read_events(_event_path(psg_path))
    headband_events = _read_events(_event_path(headband_path))
    psg_duration = _clean_float(psg_sidecar.get("RecordingDuration"))
    headband_duration = _clean_float(headband_sidecar.get("RecordingDuration"))
    psg_sfreq = _clean_float(psg_sidecar.get("SamplingFrequency"))
    headband_sfreq = _clean_float(headband_sidecar.get("SamplingFrequency"))
    record["recording_duration_sec"] = psg_duration
    record["interval_count"] = 0
    if record["pid"] is None:
        reasons.append("pid_missing")
    if record["split"] is None:
        reasons.append("pid_not_in_frozen_split")
    if not headband_path.exists():
        reasons.append("headband_missing")
    if psg_duration is None or headband_duration is None:
        reasons.append("recording_duration_missing")
    elif abs(psg_duration - headband_duration) > 1.0:
        reasons.append("alignment_invalid_or_unproven")
    if psg_sfreq is None or headband_sfreq is None:
        reasons.append("sampling_frequency_missing")
    if not _event_geometry_matches(psg_events, headband_events):
        reasons.append("alignment_invalid_or_unproven")
    channels = _read_channels(_channels_path(headband_path))
    names = (
        set(channels["name"].astype(str).str.strip())
        if "name" in channels.columns
        else set()
    )
    if not {"HB_1", "HB_2"}.issubset(names):
        reasons.append("headband_channels_incomplete")
    if not {"onset", "duration", protocol.stage_column}.issubset(psg_events.columns):
        reasons.append("missing_stage_hum")
        stage_audit: dict[str, Any] = {
            "dense_grid_available": False,
            "stage_hum_available": False,
            "segments": [],
            "failure_reasons": ["missing_stage_hum"],
        }
        reaction = np.empty(0, dtype=float)
    else:
        stage_audit = audit_stage_intervals(
            onsets_sec=pd.to_numeric(psg_events["onset"], errors="coerce").to_numpy(),
            durations_sec=pd.to_numeric(
                psg_events["duration"], errors="coerce"
            ).to_numpy(),
            stages=psg_events[protocol.stage_column].tolist(),
            recording_duration_sec=psg_duration,
            protocol=protocol,
        )
        reasons.extend(stage_audit["failure_reasons"])
        intervals = list(
            zip(
                stage_audit.get("interval_onsets_sec", []),
                stage_audit.get("interval_ends_sec", []),
            )
        )
        reaction = np.full(len(intervals), np.nan, dtype=float)
        rc_meta: dict[str, Any] = {"available": False}
        if not reasons and headband_path.exists():
            reaction, rc_meta = _read_headband_rc(
                headband_path,
                intervals=intervals,
                protocol=protocol,
            )
            if not rc_meta.get("available", False):
                reasons.append(
                    str(rc_meta.get("failure_reason", "external_rc_unavailable"))
                )
            elif headband_sfreq is not None and abs(
                float(rc_meta["sampling_frequency_hz"]) - headband_sfreq
            ) > 1e-6:
                reasons.append("sampling_frequency_mismatch")
        if not bool(stage_audit.get("dense_grid_available", False)):
            reasons.append("dense_grid_unavailable")
        if reaction.size and not np.all(np.isfinite(reaction)):
            reasons.append("external_rc_nonfinite")
    if reasons:
        record["failure_reasons"] = list(dict.fromkeys(reasons))
        record["segments"] = []
        record["reaction_coordinate"] = reaction
        return record
    segments = build_n2_first_hit_segments(
        stage_audit=stage_audit,
        reaction_coordinate=reaction,
        record=pair,
        pid=record["pid"],
        recording_duration_sec=psg_duration,
    )
    for segment in segments:
        segment["split"] = record["split"]
        segment["night_key"] = record["night_key"]
    record["segments"] = segments
    record["reaction_coordinate"] = reaction
    record["interval_count"] = int(reaction.size)
    return record


def _all_segments(records: Sequence[Mapping[str, Any]], split: str | None = None) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        if split is not None and record.get("split") != split:
            continue
        output.extend(
            dict(segment)
            for segment in record.get("segments", [])
            if segment.get("external_rc_finite")
        )
    return output


def _fit_grid(
    segments: Sequence[Mapping[str, Any]],
    *,
    lower: float,
    upper: float,
    diffusion_coefficient: float,
    min_support_per_grid: int = MIN_SUPPORT_PER_GRID,
) -> dict[str, Any]:
    arrays = assemble_adapter_arrays(segments)
    if arrays["state"].shape[0] < 30:
        return {
            "status": "not_testable",
            "failure_reason": "insufficient_adapter_rows",
            "n_rows": int(arrays["state"].shape[0]),
        }
    result = estimate_committor_local_law_dense_grid_o2b(
        arrays["state"],
        arrays["time"],
        arrays["reaction_coordinate"],
        arrays["regime_labels"],
        set_A=[3],
        set_B=[4],
        grid_min=float(lower),
        grid_max=float(upper),
        diffusion_coefficient=float(diffusion_coefficient),
        segment_id=arrays["segment_id"],
        coordinate_layer="external_headband_rc",
        coordinate_names=["headband_slow_fast_logratio"],
        reaction_coordinate_name="headband_slow_fast_logratio",
        grid_resolution=GRID_RESOLUTION,
        min_samples=30,
        min_support_per_grid=int(min_support_per_grid),
        min_transition_segments=MIN_TRANSITION_SEGMENTS,
        max_dt_relative_deviation=MAX_DT_RELATIVE_DEVIATION,
        min_valid_fraction=0.1,
    )
    return {
        "status": result.get("computation_status", "not_testable"),
        "failure_reason": result.get("failure_reason"),
        "summary": result.get("summary", {}),
        "provenance": result.get("provenance", {}),
        "series": result.get("series", {}),
        "n_rows": int(arrays["state"].shape[0]),
        "n_segments": int(len(arrays["included_segment_ids"])),
        "dropped_segment_count": int(len(arrays["dropped_segment_ids"])),
    }


def _fit_summary(fit_result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in fit_result.items()
        if key != "series"
    }


def _clone_segments_with_night_rc(
    records: Sequence[Mapping[str, Any]],
    *,
    reaction_by_night: Mapping[str, np.ndarray],
    split: str | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        if split is not None and record.get("split") != split:
            continue
        donor_rc = reaction_by_night.get(str(record["night_key"]))
        if donor_rc is None:
            continue
        donor_rc = np.asarray(donor_rc, dtype=float)
        for original in record.get("segments", []):
            segment = dict(original)
            start = int(segment["candidate_interval_index"])
            count = int(segment["n2_window_count"])
            values = donor_rc[start : start + count]
            if values.size != count:
                padded = np.full(count, np.nan, dtype=float)
                padded[: values.size] = values
                values = padded
            segment["reaction_coordinate"] = values
            segment["candidate_rc"] = (
                float(values[0]) if values.size else np.nan
            )
            segment["external_rc_finite"] = bool(
                values.size == count and np.all(np.isfinite(values))
            )
            output.append(segment)
    return output


def _score(
    segments: Sequence[Mapping[str, Any]],
    fit_result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    series = fit_result.get("series", {})
    return resolved_scoring_rows(
        segments,
        series.get("q_grid", []),
        series.get("query_grid", []),
    )


def _object1_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    lower: float,
    upper: float,
) -> dict[str, Any]:
    """Report the preregistered reliability and observed-RC diagnostics."""
    q_edges = np.linspace(0.0, 1.0, 11)
    q_bins: list[list[Mapping[str, Any]]] = [[] for _ in range(10)]
    for row in rows:
        prediction = float(row["prediction"])
        index = int(np.searchsorted(q_edges, prediction, side="right") - 1)
        q_bins[min(9, max(0, index))].append(row)
    reliability = []
    for index, values in enumerate(q_bins):
        reliability.append(
            {
                "lower": float(q_edges[index]),
                "upper": float(q_edges[index + 1]),
                "n": len(values),
                "mean_prediction": (
                    float(np.mean([float(row["prediction"]) for row in values]))
                    if values
                    else None
                ),
                "observed_rem_rate": (
                    float(np.mean([float(row["y"]) for row in values]))
                    if values
                    else None
                ),
            }
        )
    rc_edges = np.linspace(float(lower), float(upper), 11)
    rc_bins: list[list[Mapping[str, Any]]] = [[] for _ in range(10)]
    for row in rows:
        value = float(row["candidate_rc"])
        index = int(np.searchsorted(rc_edges, value, side="right") - 1)
        rc_bins[min(9, max(0, index))].append(row)
    observed_by_rc = [
        {
            "lower": float(rc_edges[index]),
            "upper": float(rc_edges[index + 1]),
            "n": len(values),
            "observed_rem_rate": (
                float(np.mean([float(row["y"]) for row in values]))
                if values
                else None
            ),
        }
        for index, values in enumerate(rc_bins)
    ]
    valid_rates = [
        item["observed_rem_rate"]
        for item in observed_by_rc
        if item["observed_rem_rate"] is not None
    ]
    monotonicity = (
        "nondecreasing"
        if len(valid_rates) > 1 and all(a <= b for a, b in zip(valid_rates, valid_rates[1:]))
        else "nonmonotone_or_insufficient"
    )
    return {
        "reliability_edges": q_edges.tolist(),
        "reliability": reliability,
        "observed_rem_rate_by_rc_bin": observed_by_rc,
        "monotonicity_diagnostic": monotonicity,
        "competing_or_censored_excluded_from_binary_n": 0,
    }


def _pids_for_records(records: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(record["pid"]) for record in records if record.get("pid") is not None}


def _floor_status(segments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pids = _pids_for_records(
        [
            {"pid": segment["pid"]}
            for segment in segments
            if segment.get("external_rc_finite")
        ]
    )
    outcomes = Counter(str(segment["outcome"]) for segment in segments)
    early = sum(segment.get("night_stratum") == "early_night" for segment in segments)
    late = sum(segment.get("night_stratum") == "late_night" for segment in segments)
    reasons = []
    if len(pids) < 10:
        reasons.append("held_out_pid_floor")
    if len(segments) < 20:
        reasons.append("held_out_segment_floor")
    if early < 5:
        reasons.append("held_out_early_floor")
    if late < 5:
        reasons.append("held_out_late_floor")
    if outcomes["first_hit_n3"] < 1:
        reasons.append("held_out_n3_floor")
    if outcomes["first_hit_rem"] < 1:
        reasons.append("held_out_rem_floor")
    return {
        "status": "PASS" if not reasons else "NOT_TESTABLE",
        "reasons": reasons,
        "eligible_pid_count": len(pids),
        "eligible_segment_count": len(segments),
        "early_segments": early,
        "late_segments": late,
        "outcome_counts": dict(outcomes),
    }


def _object2_competing_diagnostic(
    segments: Sequence[Mapping[str, Any]],
    *,
    lower: float,
    upper: float,
) -> dict[str, Any]:
    edges = np.linspace(float(lower), float(upper), 11)
    rows = [
        segment
        for segment in segments
        if segment.get("external_rc_finite")
        and np.isfinite(float(segment.get("candidate_rc", np.nan)))
        and lower <= float(segment["candidate_rc"]) <= upper
    ]
    bins: list[list[Mapping[str, Any]]] = [[] for _ in range(10)]
    for segment in rows:
        value = float(segment["candidate_rc"])
        index = int(np.searchsorted(edges, value, side="right") - 1)
        index = min(9, max(0, index))
        bins[index].append(segment)
    if any(not values for values in bins):
        return {
            "status": "NOT_TESTABLE",
            "reason": "competing_risk_empty_rc_bin",
            "bin_counts": [len(values) for values in bins],
        }
    probabilities = [
        float(
            np.mean(
                [
                    str(row["outcome"]) in COMPETING_OUTCOMES
                    for row in values
                ]
            )
        )
        for values in bins
    ]
    outcome_distribution = [
        dict(Counter(str(row["outcome"]) for row in values))
        for values in bins
    ]
    by_pid: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, values in enumerate(bins):
        for row in values:
            by_pid[str(row["pid"])].append((index, row))
    rng = np.random.default_rng(20260816)
    range_values: list[float] = []
    pids = sorted(by_pid)
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(pids, size=len(pids), replace=True)
        sampled_bins: list[list[Mapping[str, Any]]] = [[] for _ in range(10)]
        for pid in sampled:
            for index, row in by_pid[str(pid)]:
                sampled_bins[index].append(row)
        if any(not values for values in sampled_bins):
            return {
                "status": "NOT_TESTABLE",
                "reason": "competing_risk_bootstrap_empty_bin",
                "bin_counts": [len(values) for values in bins],
            }
        sampled_probabilities = [
            float(
                np.mean(
                    [
                        str(row["outcome"]) in COMPETING_OUTCOMES
                        for row in values
                    ]
                )
            )
            for values in sampled_bins
        ]
        range_values.append(
            float(max(sampled_probabilities) - min(sampled_probabilities))
        )
    return {
        "status": "computed",
        "bin_edges": edges.tolist(),
        "bin_counts": [len(values) for values in bins],
        "outcome_distribution_by_bin": outcome_distribution,
        "competing_probability_by_bin": probabilities,
        "point_range": float(max(probabilities) - min(probabilities)),
        "range_lower_95": float(np.percentile(range_values, 2.5)),
        "range_upper_95": float(np.percentile(range_values, 97.5)),
        "includes_right_censor_in_denominator": True,
        "right_censor_is_competing_exit": False,
        "method_limited_trigger": bool(
            max(probabilities) - min(probabilities) > 0.10
            and float(np.percentile(range_values, 2.5)) > 0.10
        ),
    }


def _stationarity_report(
    *,
    dev_segments: Sequence[Mapping[str, Any]],
    held_out_segments: Sequence[Mapping[str, Any]],
    lower: float,
    upper: float,
    diffusion_coefficient: float,
    base_rate: float,
    dev_fit: Mapping[str, Any],
    transfer_tolerance: float,
) -> dict[str, Any]:
    fits: dict[str, dict[str, Any]] = {
        "pooled": dict(dev_fit),
        "early": _fit_grid(
            [segment for segment in dev_segments if segment["night_stratum"] == "early_night"],
            lower=lower,
            upper=upper,
            diffusion_coefficient=diffusion_coefficient,
        ),
        "late": _fit_grid(
            [segment for segment in dev_segments if segment["night_stratum"] == "late_night"],
            lower=lower,
            upper=upper,
            diffusion_coefficient=diffusion_coefficient,
        ),
    }
    reports: dict[str, Any] = {}
    for fit_name, stratum in (
        ("pooled_to_early", "early_night"),
        ("pooled_to_late", "late_night"),
        ("early_to_early", "early_night"),
        ("late_to_late", "late_night"),
        ("early_to_late", "late_night"),
        ("late_to_early", "early_night"),
    ):
        fit_key = "pooled" if fit_name.startswith("pooled") else fit_name.split("_to_")[0]
        score_segments = [
            segment
            for segment in held_out_segments
            if segment["night_stratum"] == stratum
        ]
        fit_result = fits[fit_key]
        if fit_result.get("status") != "computed":
            reports[fit_name] = {
                "status": "NOT_TESTABLE",
                "reason": fit_result.get("failure_reason", "fit_not_computed"),
            }
            continue
        rows = _score(score_segments, fit_result)
        metrics = binary_metrics(rows)
        base_metrics = binary_metrics(
            [dict(row, prediction=base_rate) for row in rows]
        )
        reports[fit_name] = {
            "status": "computed" if rows else "NOT_TESTABLE",
            "metrics": metrics,
            "base_metrics": base_metrics,
            "n_rows": len(rows),
        }
    cross_pairs = (
        ("early_to_late", "late_to_late"),
        ("late_to_early", "early_to_early"),
    )
    transfer_pass = True
    for cross_name, same_name in cross_pairs:
        cross = reports.get(cross_name, {})
        same = reports.get(same_name, {})
        if cross.get("status") != "computed" or same.get("status") != "computed":
            transfer_pass = False
            continue
        same_brier = same["metrics"].get("brier")
        cross_brier = cross["metrics"].get("brier")
        base = cross["base_metrics"].get("brier")
        if (
            same_brier is None
            or cross_brier is None
            or base is None
            or float(cross_brier) >= float(base)
            or abs(float(cross_brier) - float(same_brier)) > transfer_tolerance
        ):
            transfer_pass = False
    return {
        "fits": {key: _fit_summary(value) for key, value in fits.items()},
        "evaluations": reports,
        "brier_transfer_tolerance": transfer_tolerance,
        "status": "PASS" if transfer_pass else "METHOD_LIMITED",
    }


def run_qualification(
    *,
    raw_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Run the source-level OD-SLP-001 qualification and write JSON."""
    protocol = SleepFirstHitProtocol()
    protocol_path = (
        Path(__file__).resolve().parents[2]
        / "project"
        / "orthagonal_axis"
        / "od_slp_001_preregistration.md"
    )
    participants = _load_pid_map(raw_root)
    split_check = assert_frozen_pid_split(
        [pid for pid in participants.values() if pid is not None]
    )
    pid_to_split = {
        str(pid): arm
        for arm, values in FROZEN_PID_SPLIT.items()
        for pid in values
    }
    pairs = _pair_acquisitions(raw_root)
    records = [
        _load_night(
            pair,
            pid_map=participants,
            pid_to_split=pid_to_split,
            protocol=protocol,
        )
        for pair in pairs
    ]
    source_failures = Counter(
        reason
        for record in records
        for reason in record.get("failure_reasons", [])
    )
    pid_arms = defaultdict(set)
    for record in records:
        if record.get("pid") is not None:
            pid_arms[str(record["pid"])].add(str(record.get("split")))
    pid_split_leakage = any(len(arms) > 1 for arms in pid_arms.values())
    dev_segments = _all_segments(records, "DEV")
    held_out_segments = _all_segments(records, "HELD_OUT")
    reserve_segments = _all_segments(records, "RESERVE")
    held_out_floor = _floor_status(held_out_segments)

    all_dev_rc = np.concatenate(
        [
            np.asarray(segment["reaction_coordinate"], dtype=float)
            for segment in dev_segments
            if segment.get("external_rc_finite")
        ]
    ) if dev_segments else np.empty(0, dtype=float)
    if all_dev_rc.size == 0 or pid_split_leakage or not split_check["matches_frozen_lists"]:
        dev_calibration: dict[str, Any] = {
            "status": "NOT_TESTABLE",
            "reason": "source_split_failure",
        }
        lower = upper = a_dev = np.nan
        dev_fit: dict[str, Any] = {"status": "not_testable", "series": {}}
    else:
        lower = float(np.percentile(all_dev_rc, 2.5))
        upper = float(np.percentile(all_dev_rc, 97.5))
        increment_variances = []
        for segment in dev_segments:
            rc = np.asarray(segment["reaction_coordinate"], dtype=float)
            time = np.asarray(segment["time"], dtype=float)
            if rc.size > 1:
                dt = np.diff(time)
                valid = np.isfinite(dt) & (dt > 0) & np.isfinite(np.diff(rc))
                increment_variances.extend(
                    ((np.diff(rc)[valid] ** 2) / dt[valid]).tolist()
                )
        a_dev = float(np.mean(increment_variances)) if increment_variances else np.nan
        dev_fit = _fit_grid(
            dev_segments,
            lower=lower,
            upper=upper,
            diffusion_coefficient=a_dev,
            min_support_per_grid=MIN_SUPPORT_PER_GRID,
        )
        support_sweep = {}
        for support in (16, 32, 64, 128):
            sweep_fit = _fit_grid(
                dev_segments,
                lower=lower,
                upper=upper,
                diffusion_coefficient=a_dev,
                min_support_per_grid=support,
            )
            support_sweep[str(support)] = _fit_summary(sweep_fit)
        dev_calibration = {
            "status": (
                "computed"
                if dev_fit.get("status") == "computed"
                else "NOT_TESTABLE"
            ),
            "lower_boundary": lower,
            "upper_boundary": upper,
            "a_DEV": a_dev,
            "support_floor": MIN_SUPPORT_PER_GRID,
            "support_sweep": support_sweep,
            "fit": _fit_summary(dev_fit),
            "base_rate": (
                float(
                    sum(segment["outcome"] == "first_hit_rem" for segment in dev_segments)
                    / max(
                        1,
                        sum(segment["outcome"] in RESOLVED_OUTCOMES for segment in dev_segments),
                    )
                )
                if any(segment["outcome"] in RESOLVED_OUTCOMES for segment in dev_segments)
                else None
            ),
        }
        dev_outcomes = Counter(
            str(segment["outcome"]) for segment in dev_segments
        )
        if (
            dev_outcomes["first_hit_n3"] < 1
            or dev_outcomes["first_hit_rem"] < 1
        ):
            dev_calibration["status"] = "NOT_TESTABLE"
            dev_calibration["reason"] = "dev_missing_resolved_outcome"

    decision_reasons: list[str] = []
    if not split_check["matches_frozen_lists"]:
        decision_reasons.append("frozen_pid_split_mismatch")
    if pid_split_leakage:
        decision_reasons.append("pid_split_leakage")
    if source_failures:
        decision_reasons.append("source_failure")
    if held_out_floor["status"] != "PASS":
        decision_reasons.extend(held_out_floor["reasons"])
    if dev_fit.get("status") != "computed":
        decision_reasons.append("dev_adapter_not_computed")
    if dev_calibration.get("reason") == "dev_missing_resolved_outcome":
        decision_reasons.append("dev_base_rate_not_testable")

    metrics: dict[str, Any] = {}
    nulls: dict[str, Any] = {}
    stationarity: dict[str, Any] = {"status": "NOT_TESTABLE"}
    object2: dict[str, Any] = {"status": "NOT_TESTABLE"}
    method_limited = False
    if not decision_reasons:
        base_rate = float(dev_calibration["base_rate"])
        true_rows = _score(held_out_segments, dev_fit)
        dev_rows = _score(dev_segments, dev_fit)
        dev_base_brier = binary_metrics(
            [dict(row, prediction=base_rate) for row in dev_rows]
        )["brier"]
        transfer_tolerance = (
            max(0.02, 0.25 * float(dev_base_brier))
            if dev_base_brier is not None
            else float("nan")
        )
        metrics["object1_diagnostics"] = _object1_diagnostics(
            true_rows,
            lower=lower,
            upper=upper,
        )
        metrics["binary_denominator"] = {
            "held_out_in_support_segments": len(held_out_segments),
            "resolved_n3_rem_segments": len(true_rows),
            "competing_or_censored_excluded": (
                len(held_out_segments) - len(true_rows)
            ),
        }
        base_compare = bootstrap_improvements(
            true_rows,
            reference_prediction=[base_rate] * len(true_rows),
            seed=20260816,
            replicates=BOOTSTRAP_REPLICATES,
        )
        metrics["true_vs_base"] = base_compare
        # Temporal nulls use frozen boundaries/a/support and refit only q_grid.
        for label, shift in (
            ("NC_time_primary", PRIMARY_TIME_SHIFT_INTERVALS),
            ("NC_time_1800s", SENSITIVITY_TIME_SHIFTS[0]),
            ("NC_time_5400s", SENSITIVITY_TIME_SHIFTS[1]),
        ):
            shifted_by_night = {
                str(record["night_key"]): shift_reaction_coordinate(
                    record["reaction_coordinate"], shift
                )
                for record in records
            }
            shifted_segments = _clone_segments_with_night_rc(
                records,
                reaction_by_night=shifted_by_night,
            )
            shifted_dev = [
                segment for segment in shifted_segments if segment.get("split") == "DEV"
            ]
            shifted_fit = _fit_grid(
                shifted_dev,
                lower=lower,
                upper=upper,
                diffusion_coefficient=a_dev,
            )
            if shifted_fit.get("status") != "computed":
                nulls[label] = {
                    "status": "accepted_negative",
                    "adapter_status": _fit_summary(shifted_fit),
                }
                continue
            shifted_rows = _score(
                [segment for segment in shifted_segments if segment.get("split") == "HELD_OUT"],
                shifted_fit,
            )
            true_for_null = true_rows
            nulls[label] = {
                "status": "computed",
                "comparison": bootstrap_pairwise_metrics(
                    true_for_null,
                    shifted_rows,
                    seed=20260816,
                    replicates=BOOTSTRAP_REPLICATES,
                ),
            }
        # Wrong-pid null.
        wrong_map: dict[str, str] = {}
        wrong_reason: str | None = None
        for arm in ("DEV", "HELD_OUT"):
            night_inputs = [
                {
                    "participant_id": record["participant_id"],
                    "session": record["session"],
                    "task": record["task"],
                    "run": record["run"],
                    "pid": record["pid"],
                    "interval_count": record["interval_count"],
                    "split": record["split"],
                    "night_key": record["night_key"],
                }
                for record in records
                if record.get("split") == arm
                and not record.get("failure_reasons")
            ]
            arm_map, arm_reason = greedy_wrong_pid_map(night_inputs)
            if arm_reason is not None:
                wrong_reason = f"{arm}:{arm_reason}"
                break
            wrong_map.update(arm_map)
        if wrong_reason is not None:
            nulls["NC_wrong_pid"] = {
                "status": "NOT_TESTABLE",
                "reason": wrong_reason,
            }
        else:
            by_key = {str(record["night_key"]): record for record in records}
            wrong_rc = {
                source: np.asarray(
                    by_key[donor]["reaction_coordinate"], dtype=float
                )
                for source, donor in wrong_map.items()
            }
            wrong_segments = _clone_segments_with_night_rc(
                records,
                reaction_by_night=wrong_rc,
            )
            wrong_fit = _fit_grid(
                [segment for segment in wrong_segments if segment.get("split") == "DEV"],
                lower=lower,
                upper=upper,
                diffusion_coefficient=a_dev,
            )
            if wrong_fit.get("status") != "computed":
                nulls["NC_wrong_pid"] = {
                    "status": "accepted_negative",
                    "adapter_status": _fit_summary(wrong_fit),
                }
            else:
                null_rows = _score(
                    [
                        segment
                        for segment in wrong_segments
                        if segment.get("split") == "HELD_OUT"
                    ],
                    wrong_fit,
                )
                nulls["NC_wrong_pid"] = {
                    "status": "computed",
                    "comparison": bootstrap_pairwise_metrics(
                        true_rows,
                        null_rows,
                        seed=20260816,
                        replicates=BOOTSTRAP_REPLICATES,
                    ),
                }
        # Label permutation changes only scoring outcomes.
        held_out_perm_map = permute_outcomes(held_out_segments)
        perm_segments = [
            dict(segment, outcome=held_out_perm_map[str(segment["segment_id"])])
            for segment in held_out_segments
        ]
        perm_rows = _score(perm_segments, dev_fit)
        nulls["NC_label"] = {
            "status": "computed" if perm_rows else "NOT_TESTABLE",
            "comparison": (
                bootstrap_pairwise_metrics(
                    true_rows,
                    perm_rows,
                    seed=20260817,
                    replicates=BOOTSTRAP_REPLICATES,
                )
                if perm_rows
                else None
            ),
        }
        stationarity = _stationarity_report(
            dev_segments=dev_segments,
            held_out_segments=held_out_segments,
            lower=lower,
            upper=upper,
            diffusion_coefficient=a_dev,
            base_rate=base_rate,
            dev_fit=dev_fit,
            transfer_tolerance=transfer_tolerance,
        )
        object2 = _object2_competing_diagnostic(
            held_out_segments,
            lower=lower,
            upper=upper,
        )
        true_pass = _comparison_pass(base_compare)
        null_pass = True
        for label in ("NC_time_primary", "NC_wrong_pid", "NC_label"):
            item = nulls.get(label, {})
            if item.get("status") == "computed":
                null_pass &= _comparison_pass(item.get("comparison", {}))
            elif item.get("status") == "NOT_TESTABLE":
                if label == "NC_wrong_pid":
                    decision_reasons.append("wrong_pid_null_not_testable")
                else:
                    null_pass = False
        if not true_pass:
            decision_reasons.append("held_out_true_vs_base_failed")
        if not null_pass:
            decision_reasons.append("held_out_null_specificity_failed")
        if stationarity.get("status") != "PASS":
            method_limited = True
        if object2.get("status") != "computed":
            decision_reasons.append("competing_risk_diagnostic_not_testable")
        elif object2.get("method_limited_trigger"):
            method_limited = True

    if decision_reasons:
        decision = (
            "NOT_TESTABLE"
            if any(
                reason in {
                    "frozen_pid_split_mismatch",
                    "pid_split_leakage",
                    "source_failure",
                    "dev_adapter_not_computed",
                    "dev_base_rate_not_testable",
                    "competing_risk_diagnostic_not_testable",
                    "wrong_pid_null_not_testable",
                    "held_out_pid_floor",
                    "held_out_segment_floor",
                    "held_out_early_floor",
                    "held_out_late_floor",
                    "held_out_n3_floor",
                    "held_out_rem_floor",
                }
                for reason in decision_reasons
            )
            else "FAIL"
        )
    elif method_limited:
        decision = "PARTIAL_PASS_METHOD_LIMITED"
    else:
        decision = "PASS"
    result = {
        "schema": "mndm.od_slp_001_empirical_qualification.v1",
        "dataset": "ds005555",
        "protocol_id": "OD-SLP-001",
        "protocol_path": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "raw_root": str(raw_root),
        "adapter_qualification_records": list(QUALIFICATION_RECORDS),
        "split": split_check,
        "inventory": {
            "recordings_total": len(records),
            "source_failure_counts": dict(source_failures),
            "split_record_counts": dict(Counter(record.get("split") for record in records)),
            "pid_split_leakage": pid_split_leakage,
            "dev_segment_count": len(dev_segments),
            "held_out_segment_count": len(held_out_segments),
            "reserve_segment_count": len(reserve_segments),
        },
        "held_out_floor": held_out_floor,
        "dev_calibration": dev_calibration,
        "metrics": metrics,
        "nulls": nulls,
        "stationarity": stationarity,
        "competing_risk": object2,
        "records": [_clean_json_record(record) for record in records],
        "decision": {
            "status": decision,
            "reasons": list(dict.fromkeys(decision_reasons)),
        },
        "fail_closed_assertions": {
            "production_overlay_written": False,
            "hdf5_written": False,
            "mnps_or_jacobian_used_as_reaction_coordinate": False,
            "stage_ai_used_as_ground_truth": False,
            "n2_to_exit_increment_used": False,
            "future_outcome_used_for_candidate_selection": False,
            "competing_or_censored_segments_dropped_from_census": False,
            "reserve_arm_opened": False,
            "held_out_refit_after_inspection": False,
        },
        "claim_boundary": (
            "Audit-only BOAS restricted-binary empirical qualification. It can "
            "only support P(REM before N3 | resolved N3/REM first hit, "
            "r_start) under the frozen pid split. It is not a competing-risk "
            "committor, biological claim, MNPS committor, or production overlay."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(json_safe(result), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_qualification(
        raw_root=args.raw_root,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "recordings_total": result["inventory"]["recordings_total"],
                "dev_segments": result["inventory"]["dev_segment_count"],
                "held_out_segments": result["inventory"]["held_out_segment_count"],
                "status": result["decision"]["status"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
