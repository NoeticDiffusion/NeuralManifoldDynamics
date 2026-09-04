"""Pure helpers for the audit-only OD-SLP-001 qualification.

This module contains no file I/O, MNE access, HDF5 writing, or production
overlay logic. It freezes the BOAS split/segment/null mechanics and provides
small scoring helpers for the empirical qualification runner.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

FROZEN_PID_SPLIT: dict[str, tuple[int, ...]] = {
    "DEV": (
        15, 77, 97, 43, 24, 108, 84, 75, 106, 20,
        87, 65, 99, 9, 10, 13, 102, 62, 4, 30,
        41, 32, 26, 1, 61, 36, 69, 86, 66, 90,
    ),
    "HELD_OUT": (
        73, 29, 68, 3, 23, 31, 78, 38, 45, 33,
        17, 91, 98, 100, 81, 72, 39, 40, 6, 48,
        14, 64, 34, 67, 46, 19, 18, 50, 60, 96,
        59, 37, 55, 92, 93, 35, 21, 22, 57, 89,
        5, 12, 53, 80, 8, 74, 54, 11, 82, 71,
    ),
    "RESERVE": (
        94, 52, 42, 51, 58, 63, 107, 76, 16, 56,
        88, 101, 83, 95, 44, 2, 85, 104, 79, 28,
    ),
}

OUTCOME_TO_LABEL = {"first_hit_n3": 3, "first_hit_rem": 4}
RESOLVED_OUTCOMES = frozenset(OUTCOME_TO_LABEL)
COMPETING_OUTCOMES = frozenset(
    {
        "competing_exit_wake",
        "competing_exit_n1",
        "competing_exit_disconnection",
        "competing_exit_artifact",
        "qc_or_gap_exit",
    }
)
ALL_OUTCOMES = RESOLVED_OUTCOMES | COMPETING_OUTCOMES | {"right_censored"}


def canonical_pid(value: Any) -> str:
    """Return the frozen decimal-string representation of a pid."""
    numeric = float(value)
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError("pid_must_be_a_finite_integer")
    return str(int(numeric))


def pid_digest(pid: Any) -> str:
    """Return the lowercase digest used to freeze the pid split."""
    value = canonical_pid(pid)
    return hashlib.sha256(
        f"OD-SLP-001|pid|{value}".encode("utf-8")
    ).hexdigest()


def recompute_pid_split(pids: Iterable[Any]) -> dict[str, tuple[int, ...]]:
    """Recompute the preregistered 30/50/20 split from source pids."""
    unique = sorted({int(canonical_pid(pid)) for pid in pids})
    ordered = sorted(unique, key=pid_digest)
    if len(ordered) != 100:
        raise ValueError("unexpected_pid_count")
    return {
        "DEV": tuple(ordered[:30]),
        "HELD_OUT": tuple(ordered[30:80]),
        "RESERVE": tuple(ordered[80:]),
    }


def assert_frozen_pid_split(pids: Iterable[Any]) -> dict[str, Any]:
    """Return a strict split verification payload."""
    try:
        recomputed = recompute_pid_split(pids)
        matches = recomputed == FROZEN_PID_SPLIT
        reason = None if matches else "frozen_pid_lists_mismatch"
    except (TypeError, ValueError) as exc:
        recomputed = {}
        matches = False
        reason = str(exc)
    return {
        "split_id": "OD-SLP-001-pid-split-v1",
        "matches_frozen_lists": bool(matches),
        "reason": reason,
        "recomputed": {
            key: list(value) for key, value in recomputed.items()
        },
        "frozen": {
            key: list(value) for key, value in FROZEN_PID_SPLIT.items()
        },
    }


def canonical_night_key(record: Mapping[str, Any]) -> str:
    """Return the frozen BIDS night key."""
    values = []
    for key in ("participant_id", "session", "task", "run"):
        value = record.get(key)
        text = "NA" if value is None or str(value).strip() == "" else str(value)
        values.append(text)
    return "|".join(values)


def canonical_segment_id(record: Mapping[str, Any], candidate_index: int) -> str:
    """Return a stable segment ID for true and null replays."""
    return f"{canonical_night_key(record)}|candidate-{int(candidate_index)}"


def build_n2_first_hit_segments(
    *,
    stage_audit: Mapping[str, Any],
    reaction_coordinate: Sequence[float],
    record: Mapping[str, Any],
    pid: Any,
    recording_duration_sec: float | None,
) -> list[dict[str, Any]]:
    """Build frozen N2-only segments from an OD-SLP-000 stage audit."""
    onsets = np.asarray(stage_audit.get("interval_onsets_sec", []), dtype=float)
    ends = np.asarray(stage_audit.get("interval_ends_sec", []), dtype=float)
    stages = np.asarray(stage_audit.get("interval_stages", []), dtype=np.int16)
    reaction = np.asarray(reaction_coordinate, dtype=float).reshape(-1)
    if not (onsets.size == ends.size == stages.size == reaction.size):
        raise ValueError("stage_rc_grid_shape_mismatch")
    mids = (onsets + ends) / 2.0
    output: list[dict[str, Any]] = []
    for candidate in stage_audit.get("segments", []):
        candidate_index = int(candidate["candidate_interval_index"])
        if candidate_index < 0 or candidate_index >= stages.size:
            continue
        n2_end = candidate_index
        while n2_end + 1 < stages.size and int(stages[n2_end + 1]) == 2:
            n2_end += 1
        rc = reaction[candidate_index : n2_end + 1].copy()
        times = mids[candidate_index : n2_end + 1].copy()
        labels = np.full(rc.size, 2, dtype=np.int16)
        outcome = str(candidate["outcome"])
        if outcome in OUTCOME_TO_LABEL and labels.size:
            labels[-1] = OUTCOME_TO_LABEL[outcome]
        finite = bool(
            rc.size > 0
            and np.all(np.isfinite(rc))
            and np.all(np.isfinite(times))
        )
        output.append(
            {
                "segment_id": canonical_segment_id(record, candidate_index),
                "pid": canonical_pid(pid),
                "participant_id": record.get("participant_id"),
                "night_key": canonical_night_key(record),
                "candidate_interval_index": candidate_index,
                "start_sec": float(candidate["start_sec"]),
                "recording_duration_sec": recording_duration_sec,
                "night_stratum": (
                    "early_night"
                    if recording_duration_sec is not None
                    and float(candidate["start_sec"]) < 0.5 * float(recording_duration_sec)
                    else "late_night"
                ),
                "outcome": outcome,
                "reaction_coordinate": rc,
                "time": times - times[0] if times.size else times,
                "regime_labels": labels,
                "external_rc_finite": finite,
                "n2_window_count": int(rc.size),
                "candidate_rc": float(rc[0]) if rc.size else np.nan,
            }
        )
    return output


def assemble_adapter_arrays(
    segments: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    """Concatenate N2-only segment rows without cross-segment increments."""
    states: list[np.ndarray] = []
    times: list[np.ndarray] = []
    reactions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    segment_ids: list[np.ndarray] = []
    included_ids: list[str] = []
    dropped_ids: list[str] = []
    for numeric_id, segment in enumerate(segments):
        rc = np.asarray(segment["reaction_coordinate"], dtype=float).reshape(-1)
        time = np.asarray(segment["time"], dtype=float).reshape(-1)
        label = np.asarray(segment["regime_labels"], dtype=np.int16).reshape(-1)
        if (
            rc.size == 0
            or rc.size != time.size
            or rc.size != label.size
            or not np.all(np.isfinite(rc))
            or not np.all(np.isfinite(time))
        ):
            dropped_ids.append(str(segment["segment_id"]))
            continue
        states.append(rc[:, None])
        reactions.append(rc)
        times.append(time)
        labels.append(label)
        segment_ids.append(np.full(rc.size, numeric_id, dtype=np.int32))
        included_ids.append(str(segment["segment_id"]))
    if not states:
        return {
            "state": np.empty((0, 1), dtype=float),
            "time": np.empty(0, dtype=float),
            "reaction_coordinate": np.empty(0, dtype=float),
            "regime_labels": np.empty(0, dtype=np.int16),
            "segment_id": np.empty(0, dtype=np.int32),
            "included_segment_ids": np.asarray([], dtype=object),
            "dropped_segment_ids": np.asarray(dropped_ids, dtype=object),
        }
    return {
        "state": np.concatenate(states, axis=0),
        "time": np.concatenate(times),
        "reaction_coordinate": np.concatenate(reactions),
        "regime_labels": np.concatenate(labels),
        "segment_id": np.concatenate(segment_ids),
        "included_segment_ids": np.asarray(included_ids, dtype=object),
        "dropped_segment_ids": np.asarray(dropped_ids, dtype=object),
    }


def shift_reaction_coordinate(
    reaction_coordinate: Sequence[float],
    shift_intervals: int,
) -> np.ndarray:
    """Apply the frozen within-night circular RC shift."""
    return np.roll(np.asarray(reaction_coordinate, dtype=float), int(shift_intervals))


def greedy_wrong_pid_map(
    nights: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], str | None]:
    """Build the frozen deterministic night-level wrong-pid assignment."""
    ordered = sorted(
        nights,
        key=lambda item: (
            pid_digest(item["pid"]),
            canonical_night_key(item),
        ),
    )
    unused = {canonical_night_key(item) for item in ordered}
    mapping: dict[str, str] = {}
    for source in ordered:
        source_key = canonical_night_key(source)
        candidates = [
            donor
            for donor in ordered
            if canonical_night_key(donor) in unused
            and canonical_night_key(donor) != source_key
            and canonical_pid(donor["pid"]) != canonical_pid(source["pid"])
            and abs(
                int(source["interval_count"]) - int(donor["interval_count"])
            )
            <= 2
        ]
        if not candidates:
            return {}, f"no_wrong_pid_donor:{source_key}"
        donor = candidates[0]
        donor_key = canonical_night_key(donor)
        mapping[source_key] = donor_key
        unused.remove(donor_key)
    return mapping, None


def permute_outcomes(
    segments: Sequence[Mapping[str, Any]],
    *,
    seed: int = 20260817,
) -> dict[str, str]:
    """Permute eight-way outcomes within one already-frozen split arm."""
    ordered = sorted(segments, key=lambda item: str(item["segment_id"]))
    outcomes = np.asarray([str(item["outcome"]) for item in ordered], dtype=object)
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(outcomes.size)
    if outcomes.size > 1 and np.array_equal(permutation, np.arange(outcomes.size)):
        permutation = np.roll(permutation, 1)
    return {
        str(item["segment_id"]): str(outcomes[permutation[index]])
        for index, item in enumerate(ordered)
    }


def resolved_scoring_rows(
    segments: Sequence[Mapping[str, Any]],
    q_grid: Sequence[float],
    query_grid: Sequence[float],
    *,
    split: str | None = None,
) -> list[dict[str, Any]]:
    """Score frozen q-grid values at segment starts for resolved outcomes."""
    grid = np.asarray(query_grid, dtype=float)
    q = np.asarray(q_grid, dtype=float)
    if grid.ndim != 1 or q.ndim != 1 or grid.size != q.size or grid.size < 2:
        raise ValueError("q_grid_shape_mismatch")
    rows: list[dict[str, Any]] = []
    for segment in segments:
        if split is not None and segment.get("split") != split:
            continue
        if segment.get("outcome") not in RESOLVED_OUTCOMES:
            continue
        start = float(segment["candidate_rc"])
        if not np.isfinite(start) or start < grid[0] or start > grid[-1]:
            continue
        rows.append(
            {
                "segment_id": str(segment["segment_id"]),
                "pid": str(segment["pid"]),
                "outcome": str(segment["outcome"]),
                "y": float(segment["outcome"] == "first_hit_rem"),
                "prediction": float(np.interp(start, grid, q)),
                "candidate_rc": start,
                "night_stratum": segment.get("night_stratum"),
            }
        )
    return rows


def binary_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    prediction_key: str = "prediction",
) -> dict[str, float | int | None]:
    """Return Brier and clipped log-loss for resolved rows."""
    if not rows:
        return {
            "n": 0,
            "brier": None,
            "log_loss": None,
            "rem_rate": None,
        }
    y = np.asarray([float(row["y"]) for row in rows], dtype=float)
    raw_prediction = np.asarray(
        [float(row[prediction_key]) for row in rows],
        dtype=float,
    )
    prediction = np.clip(raw_prediction, 1e-6, 1.0 - 1e-6)
    return {
        "n": int(y.size),
        "brier": float(np.mean((raw_prediction - y) ** 2)),
        "log_loss": float(
            -np.mean(y * np.log(prediction) + (1.0 - y) * np.log(1.0 - prediction))
        ),
        "rem_rate": float(np.mean(y)),
    }


def bootstrap_improvements(
    rows: Sequence[Mapping[str, Any]],
    *,
    reference_prediction: Sequence[float],
    seed: int = 20260816,
    replicates: int = 2000,
) -> dict[str, Any]:
    """Compare true predictions with a reference using pid bootstrap."""
    if not rows:
        return {"status": "not_testable", "reason": "no_resolved_rows"}
    ordered = list(rows)
    pids = sorted({str(row["pid"]) for row in ordered})
    by_pid: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in ordered:
        by_pid[str(row["pid"])].append(row)
    true_metrics = binary_metrics(ordered)
    ref_rows = [
        dict(row, prediction=float(prediction))
        for row, prediction in zip(ordered, reference_prediction)
    ]
    reference_by_segment = {
        str(row["segment_id"]): float(row["prediction"]) for row in ref_rows
    }
    ref_metrics = binary_metrics(ref_rows)
    if true_metrics["n"] == 0 or ref_metrics["n"] == 0:
        return {"status": "not_testable", "reason": "no_resolved_rows"}
    if float(ref_metrics["brier"]) <= 0:
        return {"status": "not_testable", "reason": "zero_reference_brier"}
    brier_values: list[float] = []
    logloss_values: list[float] = []
    rng = np.random.default_rng(int(seed))
    for _ in range(int(replicates)):
        sampled = rng.choice(pids, size=len(pids), replace=True)
        sample_rows = [row for pid in sampled for row in by_pid[pid]]
        if not sample_rows or len({row["y"] for row in sample_rows}) < 2:
            return {
                "status": "not_testable",
                "reason": "bootstrap_replicate_without_both_outcomes",
            }
        true_sample = binary_metrics(sample_rows)
        ref_sample_rows = [
            dict(
                row,
                prediction=reference_by_segment[str(row["segment_id"])],
            )
            for row in sample_rows
        ]
        ref_sample = binary_metrics(ref_sample_rows)
        brier_values.append(
            (float(ref_sample["brier"]) - float(true_sample["brier"]))
            / float(ref_sample["brier"])
        )
        logloss_values.append(
            float(ref_sample["log_loss"]) - float(true_sample["log_loss"])
        )
    brier_point = (
        float(ref_metrics["brier"]) - float(true_metrics["brier"])
    ) / float(ref_metrics["brier"])
    logloss_point = float(ref_metrics["log_loss"]) - float(true_metrics["log_loss"])
    return {
        "status": "computed",
        "point_relative_brier_improvement": brier_point,
        "point_log_loss_improvement": logloss_point,
        "brier_lower_95": float(np.percentile(brier_values, 2.5)),
        "brier_upper_95": float(np.percentile(brier_values, 97.5)),
        "log_loss_lower_95": float(np.percentile(logloss_values, 2.5)),
        "log_loss_upper_95": float(np.percentile(logloss_values, 97.5)),
        "true_metrics": true_metrics,
        "reference_metrics": ref_metrics,
        "n_pids": len(pids),
        "replicates": int(replicates),
    }


def bootstrap_pairwise_metrics(
    true_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 20260816,
    replicates: int = 2000,
) -> dict[str, Any]:
    """Compare two scored row sets with a pid-clustered bootstrap."""
    if not true_rows or not reference_rows:
        return {"status": "not_testable", "reason": "empty_comparison_rows"}
    true_by_pid: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    ref_by_pid: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in true_rows:
        true_by_pid[str(row["pid"])].append(row)
    for row in reference_rows:
        ref_by_pid[str(row["pid"])].append(row)
    pids = sorted(set(true_by_pid) | set(ref_by_pid))
    true_point = binary_metrics(true_rows)
    ref_point = binary_metrics(reference_rows)
    if float(ref_point["brier"] or 0.0) <= 0:
        return {"status": "not_testable", "reason": "zero_reference_brier"}
    brier_values: list[float] = []
    logloss_values: list[float] = []
    rng = np.random.default_rng(int(seed))
    for _ in range(int(replicates)):
        sampled = rng.choice(pids, size=len(pids), replace=True)
        true_sample = [
            row for pid in sampled for row in true_by_pid.get(pid, [])
        ]
        ref_sample = [
            row for pid in sampled for row in ref_by_pid.get(pid, [])
        ]
        if (
            not true_sample
            or not ref_sample
            or len({row["y"] for row in true_sample}) < 2
            or len({row["y"] for row in ref_sample}) < 2
        ):
            return {
                "status": "not_testable",
                "reason": "bootstrap_replicate_without_both_outcomes",
            }
        true_metrics = binary_metrics(true_sample)
        ref_metrics = binary_metrics(ref_sample)
        brier_values.append(
            (float(ref_metrics["brier"]) - float(true_metrics["brier"]))
            / float(ref_metrics["brier"])
        )
        logloss_values.append(
            float(ref_metrics["log_loss"]) - float(true_metrics["log_loss"])
        )
    return {
        "status": "computed",
        "point_relative_brier_improvement": (
            float(ref_point["brier"]) - float(true_point["brier"])
        )
        / float(ref_point["brier"]),
        "point_log_loss_improvement": (
            float(ref_point["log_loss"]) - float(true_point["log_loss"])
        ),
        "brier_lower_95": float(np.percentile(brier_values, 2.5)),
        "brier_upper_95": float(np.percentile(brier_values, 97.5)),
        "log_loss_lower_95": float(np.percentile(logloss_values, 2.5)),
        "log_loss_upper_95": float(np.percentile(logloss_values, 97.5)),
        "true_metrics": true_point,
        "reference_metrics": ref_point,
        "n_pids": len(pids),
        "replicates": int(replicates),
    }
