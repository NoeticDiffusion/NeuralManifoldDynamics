"""Bounded SL-013 equal-span, fixed-seven, and fast-dynamics diagnostics."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from mndm.mnps.v3_2.candidate_grid import CandidateWindow, reconstruct_condition_blocks
from mndm.mnps.v3_2.experiments import full_vector_challenge_001 as gate_b
from mndm.mnps.v3_2.experiments import full_vector_challenge_001_gate_c1 as gate_c1
from mndm.mnps.v3_2.experiments import window_support_001 as parent


PROTOCOL_ID = "SL-013-WINDOW-QUALITY-002-FAST-DYNAMICS-IDENTIFIABILITY-001"
BASELINE_MANIFEST_SHA256 = "d3cdcd72205279ae604d651c26a618d5323a4b613fa3f663af0c03173fa83365"
BASELINE_RUN_DEFAULT = Path(
    r"C:\Users\Robin\.codex\visualizations\2026\10\11\01a12bb4-ebf8-7840-a3c6-bbd60bb89c7a"
    r"\WINDOW-SUPPORT-001\run_20261015_v1"
)
PERTURBATION_RUN_DEFAULT = BASELINE_RUN_DEFAULT.parent / "perturbation_20261015_v1"
DEFAULT_PROTOCOL_PATH = (
    parent.REPO_ROOT / "project" / "new_mnps_v.3.2" / "protocols"
    / "SL-013-WINDOW-QUALITY-002-FAST-DYNAMICS-IDENTIFIABILITY-001.md"
)
DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\Robin\.codex\visualizations\2026\10\11\01a12bb4-ebf8-7840-a3c6-bbd60bb89c7a"
    r"\SL-013-BENCHMARK-001\run_20261015_v1"
)
PARTICIPANTS = ("sub-079", "sub-047", "sub-026", "sub-094")
SCALAR_FAMILIES = ("PCV", "TP", "NIE", "RD", "OI", "SC", "ED")
FAMILY_ORDER = (*SCALAR_FAMILIES, "AR")
COMMON7_ID = "SL-013.common7.PCV_TP_NIE_RD_OI_SC_ED.v1"
SAMPLE_RATE_HZ = 500.0
EQUAL_SPAN_OFFSETS_SECONDS = {4: (0, 4, 8, 12, 16), 8: (0, 4, 8, 12), 12: (0, 4, 8)}
ANNOTATION_PATTERN = re.compile(r"gap|discontinu|drop.?out|signal break|bad interval", re.I)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_class(row: Mapping[str, Any]) -> str:
    roles = set(row.get("window_grid_roles") or [])
    cadence = "cadence" in roles
    matched = "matched_20s_anchor_prefix" in roles
    if cadence and matched:
        return "cadence_and_matched"
    if cadence:
        return "cadence_only"
    if matched:
        return "matched_only"
    return "unclassified"


def physical_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("participant_id"), row.get("run_id"), row.get("condition"),
        row.get("state_block_id"), int(row.get("duration_seconds", -1)),
        int(row.get("support_start_sample", -1)), int(row.get("support_end_sample_exclusive", -1)),
    )


def row_value(row: Mapping[str, Any]) -> Any:
    return row.get("component_values") if row.get("family_id") == "AR" else row.get("value")


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool) and math.isfinite(float(value))


def parse_json_cell(value: Any) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def numeric_iqr(values: Sequence[float]) -> float | None:
    finite = np.asarray([float(value) for value in values if is_finite_number(value)], dtype=np.float64)
    if finite.size == 0:
        return None
    q25, q75 = np.quantile(finite, (0.25, 0.75), method="linear")
    return float(q75 - q25)


def equal_span_offsets(duration_seconds: int) -> tuple[int, ...]:
    if duration_seconds not in EQUAL_SPAN_OFFSETS_SECONDS:
        raise ValueError("duration_not_in_equal_span_design")
    return EQUAL_SPAN_OFFSETS_SECONDS[duration_seconds]


def intervals_cover_anchor(
    anchor_start: int, anchor_end: int, intervals: Iterable[tuple[int, int]]
) -> bool:
    ordered = sorted((int(start), int(end)) for start, end in intervals)
    cursor = int(anchor_start)
    for start, end in ordered:
        if start < anchor_start or end > anchor_end or end <= start:
            return False
        if start > cursor:
            return False
        cursor = max(cursor, end)
    return cursor >= anchor_end


def _design_diagnostics(values: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or not np.isfinite(matrix).all():
        return {"row_count": int(matrix.shape[0]) if matrix.ndim == 2 else 0,
                "column_count": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
                "singular_values": [], "rank": 0, "condition_number": None,
                "rank_tolerance": None}
    augmented = np.column_stack((matrix, np.ones(matrix.shape[0], dtype=np.float64)))
    singular = np.linalg.svd(augmented, compute_uv=False)
    tolerance = float(singular[0] * max(augmented.shape) * np.finfo(np.float64).eps)
    rank = int(np.count_nonzero(singular > tolerance))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0.0 else None
    return {"row_count": int(augmented.shape[0]), "column_count": int(augmented.shape[1]),
            "singular_values": [float(value) for value in singular], "rank": rank,
            "condition_number": condition, "rank_tolerance": tolerance}


def trajectory_design_diagnostics(
    left_states: Sequence[Sequence[float]],
) -> dict[str, Any]:
    values = np.asarray(left_states, dtype=np.float64)
    raw = _design_diagnostics(values)
    train_count = max(2, int(math.ceil(values.shape[0] / 2))) if values.ndim == 2 else 0
    standardized: dict[str, Any]
    means: list[float] | None = None
    scales: list[float] | None = None
    if values.ndim != 2 or values.shape[0] < train_count or not np.isfinite(values).all():
        standardized = {"available": False, "reason": "insufficient_finite_training_prefix",
                        **_design_diagnostics(np.empty((0, values.shape[1] if values.ndim == 2 else 0)))}
    else:
        prefix = values[:train_count]
        mean_array = np.mean(prefix, axis=0, dtype=np.float64)
        scale_array = np.std(prefix, axis=0, dtype=np.float64, ddof=0)
        if not np.isfinite(scale_array).all() or np.any(scale_array <= 0.0):
            standardized = {"available": False, "reason": "zero_or_nonfinite_training_prefix_scale",
                            **_design_diagnostics(np.empty((0, values.shape[1])))}
            means, scales = mean_array.tolist(), scale_array.tolist()
        else:
            standardized_values = (values - mean_array) / scale_array
            standardized = {"available": True,
                            **_design_diagnostics(standardized_values)}
            means, scales = mean_array.tolist(), scale_array.tolist()
    return {
        "raw": raw,
        "training_prefix_standardized": standardized,
        "training_prefix_state_count": train_count if values.ndim == 2 and values.shape[0] >= train_count else None,
        "training_prefix_mean": means,
        "training_prefix_population_sd": scales,
    }


def synthetic_recovery(seed: int = 13013, dimension: int = 8) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    transition = rng.normal(0.0, 0.08, size=(dimension, dimension))
    radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
    transition *= 0.78 / radius
    intercept = rng.normal(0.0, 0.03, size=dimension)
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for _ in range(dimension * 3):
        state = rng.normal(size=dimension)
        for _step in range(30):
            successor = transition @ state + intercept
            inputs.append(state.copy())
            targets.append(successor.copy())
            state = successor
    x = np.asarray(inputs, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    design = np.column_stack((x, np.ones(x.shape[0], dtype=np.float64)))
    coefficients, _residuals, _rank, _singular = np.linalg.lstsq(design, y, rcond=None)
    recovered_transition = coefficients[:-1, :].T
    recovered_intercept = coefficients[-1, :]
    full_rank = int(np.linalg.matrix_rank(design))
    transition_error = float(np.linalg.norm(recovered_transition - transition) / np.linalg.norm(transition))
    intercept_error = float(np.linalg.norm(recovered_intercept - intercept) / np.linalg.norm(intercept))

    deficient_x = x.copy()
    deficient_x[:, -1] = deficient_x[:, -2]
    deficient_design = np.column_stack((deficient_x, np.ones(deficient_x.shape[0], dtype=np.float64)))
    deficient_rank = int(np.linalg.matrix_rank(deficient_design))
    return {
        "seed": seed, "dimension": dimension,
        "system_type": "stable_affine_linear_discrete_time",
        "known_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(transition)))),
        "transitions": int(x.shape[0]), "design_rank": full_rank,
        "expected_full_rank": dimension + 1,
        "relative_transition_parameter_error": transition_error,
        "relative_intercept_error": intercept_error,
        "full_rank_recovery_pass": full_rank == dimension + 1 and transition_error < 1e-10 and intercept_error < 1e-10,
        "duplicate_coordinate_control_rank": deficient_rank,
        "duplicate_coordinate_control_rejected": deficient_rank < dimension + 1,
        "qualified": False,
    }


def _load_parent_rows(parent_run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = parent_run_dir / "manifest.json"
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != BASELINE_MANIFEST_SHA256:
        raise ValueError("parent_manifest_hash_does_not_match_frozen_protocol")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("qualified") is not False:
        raise ValueError("parent_run_must_remain_unqualified")
    if manifest.get("participants_processed") != list(PARTICIPANTS):
        raise ValueError("parent_participant_set_does_not_match_frozen_protocol")
    if not manifest.get("c1_20s_parity_pass"):
        raise ValueError("parent_c1_parity_must_pass")
    measurements_path = parent_run_dir / "window_measurements.jsonl.gz"
    expected_hash = manifest.get("output_hashes", {}).get("window_measurements.jsonl.gz")
    if not expected_hash or sha256_file(measurements_path) != expected_hash:
        raise ValueError("parent_measurements_hash_mismatch")
    rows: list[dict[str, Any]] = []
    with gzip.open(measurements_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != 8000 or len(rows) != int(manifest.get("all_window_rows", -1)):
        raise ValueError("parent_measurement_row_count_mismatch")
    return manifest, rows


def build_row_accounting(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_class_duration_family: dict[tuple[str, int, str], int] = {}
    unique: dict[tuple[Any, ...], str] = {}
    for row in rows:
        cls = record_class(row)
        key = (cls, int(row["duration_seconds"]), str(row["family_id"]))
        by_class_duration_family[key] = by_class_duration_family.get(key, 0) + 1
        window_key = physical_key(row)
        prior_class = unique.get(window_key)
        if prior_class is not None and prior_class != cls:
            raise ValueError("family_rows_disagree_on_parent_window_record_class")
        unique[window_key] = cls
    detail = [
        {"record_class": cls, "duration_seconds": duration, "family_id": family,
         "family_rows": count, "qualified": False}
        for (cls, duration, family), count in sorted(by_class_duration_family.items())
    ]
    totals: dict[str, int] = {}
    for cls in unique.values():
        totals[cls] = totals.get(cls, 0) + 1
    summary = [{"record_class": cls, "unique_windows": count, "family_rows": sum(
        row_count for (row_cls, _duration, _family), row_count in by_class_duration_family.items()
        if row_cls == cls
    ), "qualified": False} for cls, count in sorted(totals.items())]
    return detail, summary


def build_common7_and_ar(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("participant_id"), row.get("run_id"), row.get("condition"),
            row.get("state_block_id"), row.get("duration_seconds"),
            row.get("support_start_sample"), row.get("support_end_sample_exclusive"),
            row.get("window_id"),
        )
        grouped.setdefault(key, {})[str(row["family_id"])] = row
    common_rows: list[dict[str, Any]] = []
    ar_rows: list[dict[str, Any]] = []
    for key, families in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        first = next(iter(families.values()))
        common: dict[str, Any] = {
            "protocol_id": PROTOCOL_ID, "fixed_vector_realization_id": COMMON7_ID,
            "participant_id": key[0], "run_id": key[1], "condition": key[2],
            "state_block_id": key[3], "duration_seconds": key[4],
            "support_start_sample": key[5], "support_end_sample_exclusive": key[6],
            "window_id": key[7], "record_class": record_class(first),
            "sample_rate_hz": first.get("sample_rate_hz"), "channel_count": first.get("channel_count"),
            "channel_order_sha256": first.get("channel_order_sha256"),
            "channel_set_sha256": first.get("channel_set_sha256"),
            "reference_id": first.get("reference_id"),
            "reference_realization_sha256": first.get("reference_realization_sha256"),
            "reference_set_sha256": first.get("reference_set_sha256"),
            "coordinate_order": json.dumps(SCALAR_FAMILIES), "qualified": False,
        }
        for family in SCALAR_FAMILIES:
            row = families.get(family)
            common[f"{family}_value"] = row.get("value") if row else None
            common[f"{family}_valid"] = row.get("valid") if row else False
            common[f"{family}_status"] = row.get("status") if row else "missing_coordinate_row"
            common[f"{family}_reason"] = row.get("reason") if row else "missing_coordinate_row"
        common_rows.append(common)
        ar = families.get("AR")
        if ar:
            ar_rows.append({
                "protocol_id": PROTOCOL_ID, "participant_id": key[0], "run_id": key[1],
                "condition": key[2], "state_block_id": key[3], "duration_seconds": key[4],
                "support_start_sample": key[5], "support_end_sample_exclusive": key[6],
                "window_id": key[7], "record_class": record_class(ar),
                "duration_realization_id": ar.get("duration_realization_id"),
                "component_names": json.dumps(ar.get("component_names")),
                "component_values": json.dumps(ar.get("component_values")),
                "computed": ar.get("computed"), "valid": ar.get("valid"),
                "status": ar.get("status"), "reason": ar.get("reason"),
                "qualified": False,
            })
    return common_rows, ar_rows


def _percentile_summary(values: Sequence[float]) -> tuple[float | None, float | None, float | None]:
    finite = np.asarray([float(value) for value in values if is_finite_number(value)], dtype=np.float64)
    if finite.size == 0:
        return None, None, None
    q25, median, q75 = np.quantile(finite, (0.25, 0.5, 0.75), method="linear")
    return float(median), float(q75 - q25), float(q25)


def build_equal_span(
    *, rows: Sequence[Mapping[str, Any]],
    parent_run_dir: Path,
    selection_records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, str]], float]:
    by_interval: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        by_interval.setdefault(physical_key(row), {})[str(row["family_id"])] = row
    anchors: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        if int(row["duration_seconds"]) == 20 and "cadence" in row.get("window_grid_roles", []):
            anchors[physical_key(row)] = row
    if not anchors:
        raise ValueError("no_parent_20s_cadence_anchors_found")

    records_by_id = {str(record["participant_id"]): record for record in selection_records}
    missing: dict[str, list[tuple[tuple[Any, ...], parent.ScheduledWindow]]] = {}
    plans: list[dict[str, Any]] = []
    for anchor_key, anchor in anchors.items():
        participant_id, run_id, condition, block_id, _duration, start, end = anchor_key
        if int(end) - int(start) != int(round(20 * SAMPLE_RATE_HZ)):
            raise ValueError("anchor_not_20_seconds_on_frozen_500hz_grid")
        for duration, offsets in EQUAL_SPAN_OFFSETS_SECONDS.items():
            window_keys: list[tuple[Any, ...]] = []
            for index, offset in enumerate(offsets):
                sub_start = int(start + round(offset * SAMPLE_RATE_HZ))
                sub_end = int(sub_start + round(duration * SAMPLE_RATE_HZ))
                if sub_start < start or sub_end > end:
                    raise ValueError("equal_span_subwindow_exceeds_anchor")
                sub_key = (participant_id, run_id, condition, block_id, duration, sub_start, sub_end)
                window_keys.append(sub_key)
                if sub_key not in by_interval:
                    run_meta_record = records_by_id.get(str(participant_id))
                    if run_meta_record is None:
                        raise ValueError("missing_parent_selection_record_for_anchor")
                    window = CandidateWindow(
                        window_id=f"{anchor.get('window_id')}-eq{duration:02d}-p{index + 1:02d}",
                        block_id=str(block_id), condition=str(condition),
                        start_sample=sub_start, end_sample_exclusive=sub_end,
                    )
                    scheduled = parent.ScheduledWindow(
                        window=window, duration_seconds=duration,
                        grid_roles=("equal_20s_anchor_partition",),
                        matched_anchor_id=str(anchor.get("window_id")),
                        matched_anchor_start_sample=int(start),
                        matched_anchor_end_sample_exclusive=int(end),
                    )
                    missing.setdefault(str(participant_id), []).append((sub_key, scheduled))
            intervals = [(key[5], key[6]) for key in window_keys]
            plans.append({"anchor_key": anchor_key, "anchor": anchor, "duration": duration,
                          "window_keys": window_keys,
                          "geometric_coverage": intervals_cover_anchor(int(start), int(end), intervals)})

    source_hashes: dict[str, dict[str, str]] = {}
    compute_started = time.perf_counter()
    for participant_id, tasks in missing.items():
        record = records_by_id[participant_id]
        run_metadata = gate_c1._metadata_record_to_run(record)
        gate_b._verify_frozen_inputs(run_metadata)
        eeg = gate_b._read_json(Path(run_metadata.eeg_json_path))
        acquisition_reference = str(eeg.get("EEGReference", "unknown"))
        data, fdt_hash = gate_b._source_memmap(run_metadata)
        try:
            for sub_key, scheduled in tasks:
                computed_rows = parent.compute_window_rows(
                    run_metadata, scheduled, data, fdt_hash, acquisition_reference
                )
                by_interval[sub_key] = {
                    str(item["family_id"]): {**item, "_sl013_row_origin": "recomputed_equal_span_window"}
                    for item in computed_rows
                }
            source_hashes[participant_id] = {
                "set": run_metadata.set_sha256, "fdt": fdt_hash,
                "events": run_metadata.events_sha256, "channels": run_metadata.channels_sha256,
                "eeg_json": run_metadata.eeg_json_sha256,
            }
        finally:
            del data
    elapsed = time.perf_counter() - compute_started

    subwindow_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for plan in plans:
        anchor = plan["anchor"]
        anchor_key = plan["anchor_key"]
        anchor_families = by_interval[anchor_key]
        duration = int(plan["duration"])
        for family in SCALAR_FAMILIES:
            values: list[float] = []
            requested_rows = []
            for index, sub_key in enumerate(plan["window_keys"]):
                subrow = by_interval[sub_key].get(family)
                if subrow is None:
                    raise ValueError(f"equal_span_missing_family_row:{family}")
                value = subrow.get("value")
                valid_finite = bool(subrow.get("valid")) and is_finite_number(value)
                if valid_finite:
                    values.append(float(value))
                source = str(subrow.get("_sl013_row_origin", "parent_row"))
                record = {
                    "protocol_id": PROTOCOL_ID,
                    "participant_id": anchor_key[0], "run_id": anchor_key[1],
                    "condition": anchor_key[2], "state_block_id": anchor_key[3],
                    "anchor_window_id": anchor.get("window_id"),
                    "anchor_start_sample": anchor_key[5], "anchor_end_sample_exclusive": anchor_key[6],
                    "duration_seconds": duration, "subwindow_index": index + 1,
                    "subwindow_offset_seconds": EQUAL_SPAN_OFFSETS_SECONDS[duration][index],
                    "subwindow_id": subrow.get("window_id"),
                    "support_start_sample": sub_key[5], "support_end_sample_exclusive": sub_key[6],
                    "family_id": family, "value": value, "computed": subrow.get("computed"),
                    "valid": subrow.get("valid"), "finite": is_finite_number(value),
                    "status": subrow.get("status"), "reason": subrow.get("reason"),
                    "duration_realization_id": subrow.get("duration_realization_id"),
                    "estimator_id": subrow.get("estimator_id"),
                    "channel_set_sha256": subrow.get("channel_set_sha256"),
                    "reference_realization_sha256": subrow.get("reference_realization_sha256"),
                    "source_row_class": record_class(subrow) if source == "parent_row" else "equal_span_recomputed",
                    "row_source": source, "qualified": False,
                }
                requested_rows.append(record)
                subwindow_rows.append(record)
            median, iqr, _q25 = _percentile_summary(values)
            full_value = anchor_families[family].get("value")
            summary_rows.append({
                "protocol_id": PROTOCOL_ID,
                "participant_id": anchor_key[0], "run_id": anchor_key[1],
                "condition": anchor_key[2], "state_block_id": anchor_key[3],
                "anchor_window_id": anchor.get("window_id"),
                "anchor_start_sample": anchor_key[5], "anchor_end_sample_exclusive": anchor_key[6],
                "duration_seconds": duration, "family_id": family,
                "short_windows_requested": len(requested_rows),
                "short_windows_valid_finite": len(values),
                "all_short_windows_valid_finite": len(values) == len(requested_rows),
                "exact_geometric_anchor_coverage": plan["geometric_coverage"],
                "short_window_median": median, "short_window_iqr": iqr,
                "full_20s_anchor_value": full_value,
                "short_median_minus_20s": median - float(full_value)
                if median is not None and is_finite_number(full_value) else None,
                "qualified": False,
            })
    return subwindow_rows, summary_rows, source_hashes, elapsed


def _value_for_coordinate(row: Mapping[str, Any], family: str) -> float | None:
    value = row.get("value")
    return float(value) if bool(row.get("valid")) and is_finite_number(value) else None


def build_fast_dynamics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    intervals: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        if int(row.get("duration_seconds", -1)) != 4 or "cadence" not in row.get("window_grid_roles", []):
            continue
        key = (
            row.get("participant_id"), row.get("run_id"), row.get("condition"),
            row.get("state_block_id"), int(row["support_start_sample"]),
            int(row["support_end_sample_exclusive"]),
        )
        intervals.setdefault(key, {})[str(row["family_id"])] = row
    trajectories: dict[tuple[Any, ...], list[tuple[tuple[Any, ...], np.ndarray | None]]] = {}
    for key, family_rows in intervals.items():
        group = key[:4]
        coordinates: list[float] = []
        complete = True
        for family in SCALAR_FAMILIES:
            row = family_rows.get(family)
            value = _value_for_coordinate(row, family) if row else None
            if value is None:
                complete = False
                break
            coordinates.append(value)
        if complete:
            ar = family_rows.get("AR")
            ar_values = ar.get("component_values") if ar and bool(ar.get("valid")) else None
            if not isinstance(ar_values, list) or not ar_values or not is_finite_number(ar_values[0]):
                complete = False
            else:
                coordinates.append(float(ar_values[0]))
        state = np.asarray(coordinates, dtype=np.float64) if complete else None
        trajectories.setdefault(group, []).append((key, state))

    diagnostics: list[dict[str, Any]] = []
    for group, items in sorted(trajectories.items(), key=lambda item: tuple(str(value) for value in item[0])):
        items.sort(key=lambda item: item[0][4])
        effective_states = [(key, state) for key, state in items if state is not None]
        transitions: list[tuple[np.ndarray, np.ndarray]] = []
        for (left_key, left), (right_key, right) in zip(effective_states, effective_states[1:]):
            if left_key[5] == right_key[4]:
                transitions.append((left, right))
        left_values = np.asarray([left for left, _right in transitions], dtype=np.float64)
        design = trajectory_design_diagnostics(left_values if transitions else np.empty((0, 8)))
        diagnostics.append({
            "protocol_id": PROTOCOL_ID, "participant_id": group[0], "run_id": group[1],
            "condition": group[2], "state_block_id": group[3], "duration_seconds": 4,
            "coordinate_order": json.dumps((*SCALAR_FAMILIES, "AR_component_0")),
            "requested_cadence_windows": len(items), "complete_valid_finite_states": len(effective_states),
            "sample_grid_adjacent_transitions": len(transitions),
            "necessary_transition_count_for_9_parameter_affine_input_design": 9,
            "nominal_transition_floor_met": len(transitions) >= 9,
            "physical_continuity_confirmed": False,
            "physical_continuity_status": "unknown_from_available_source_annotations",
            "raw_design_rank": design["raw"]["rank"],
            "raw_design_columns": design["raw"]["column_count"],
            "raw_singular_values": json.dumps(design["raw"]["singular_values"]),
            "raw_condition_number": design["raw"]["condition_number"],
            "standardized_design_available": design["training_prefix_standardized"]["available"],
            "standardized_design_rank": design["training_prefix_standardized"]["rank"],
            "standardized_singular_values": json.dumps(design["training_prefix_standardized"]["singular_values"]),
            "standardized_condition_number": design["training_prefix_standardized"]["condition_number"],
            "scaler_prefix_state_count": design["training_prefix_state_count"],
            "scaler_prefix_mean": json.dumps(design["training_prefix_mean"]),
            "scaler_prefix_population_sd": json.dumps(design["training_prefix_population_sd"]),
            "interpretation": "retrospective design support diagnostic; no neural Jacobian or forecasting fit",
            "qualified": False,
        })
    return diagnostics


def source_continuity_audit(selection_records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    audits: list[dict[str, Any]] = []
    source_hashes: dict[str, dict[str, str]] = {}
    for record in selection_records:
        run = gate_c1._metadata_record_to_run(record)
        gate_b._verify_frozen_inputs(run)
        events = gate_b._events_for_run(Path(run.events_path))
        channels = gate_b._read_tsv(Path(run.channels_path))
        raw_blocks = reconstruct_condition_blocks(
            events, sample_rate_hz=run.sample_rate_hz,
            recording_sample_count=run.sample_count,
        )
        boundary_events = [event for event in events if (
            str(event.get("trial_type", "")).strip().casefold() == "boundary"
            or str(event.get("value", "")).strip().casefold() == "boundary"
        )]
        boundary_samples: list[int] = []
        for event in boundary_events:
            try:
                onset = float(event.get("onset", "nan"))
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(onset) and onset >= 0:
                boundary_samples.append(int(round(onset * run.sample_rate_hz)))
        interior_boundaries = sum(
            1 for sample in boundary_samples for block in raw_blocks
            if block.start_sample < sample < block.end_sample_exclusive
        )
        event_text_rows = [
            event for event in events
            if ANNOTATION_PATTERN.search(" ".join(str(value) for value in event.values()))
        ]
        status_present = any("status" in {str(k).casefold() for k in channel} for channel in channels)
        bad_channels = [
            channel for channel in channels
            if str(channel.get("status", "")).strip().casefold() == "bad"
        ]
        audits.append({
            "protocol_id": PROTOCOL_ID, "participant_id": run.participant_id, "run_id": run.run_id,
            "events_rows": len(events), "declared_boundary_events": len(boundary_events),
            "boundary_markers_strictly_inside_unsplit_condition_blocks": interior_boundaries,
            "event_rows_with_explicit_gap_or_discontinuity_text": len(event_text_rows),
            "channels_table_columns": json.dumps(sorted({str(key) for channel in channels for key in channel})),
            "channels_table_has_status_column": status_present,
            "declared_bad_channels": len(bad_channels),
            "source_annotation_audit_status": "no_explicit_gap_or_bad_interval_claim_found" if not event_text_rows else "explicit_gap_or_discontinuity_text_present_review_required",
            "physical_continuity_confirmed": False,
            "physical_continuity_limit": "available BIDS annotations cannot establish continuity of unmarked source samples",
            "qualified": False,
        })
        source_hashes[run.participant_id] = {
            "set": run.set_sha256, "events": run.events_sha256, "channels": run.channels_sha256,
            "eeg_json": run.eeg_json_sha256,
        }
    return audits, source_hashes


def normalized_perturbation_sensitivity(
    *, rows: Sequence[Mapping[str, Any]], perturbation_run_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_path = perturbation_run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("parent_manifest_sha256", "").casefold() != BASELINE_MANIFEST_SHA256:
        raise ValueError("perturbation_run_parent_hash_mismatch")
    effects_path = perturbation_run_dir / "perturbation_effects.csv"
    expected = manifest.get("output_hashes", {}).get("perturbation_effects.csv")
    if not expected or sha256_file(effects_path) != expected:
        raise ValueError("perturbation_effects_hash_mismatch")
    natural: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        if int(row.get("duration_seconds", -1)) != 4 or "cadence" not in row.get("window_grid_roles", []):
            continue
        if not bool(row.get("valid")):
            continue
        family = str(row.get("family_id"))
        value = row_value(row)
        if family == "AR":
            value = value[0] if isinstance(value, list) and value else None
        if is_finite_number(value):
            natural.setdefault((str(row["participant_id"]), str(row["condition"]), family), []).append(float(value))
    details: list[dict[str, Any]] = []
    with effects_path.open("r", encoding="utf-8", newline="") as stream:
        for effect in csv.DictReader(stream):
            if int(effect.get("duration_seconds", -1)) != 4:
                continue
            family = str(effect["family_id"])
            component_index: int | None = 0 if family == "AR" else None
            absolute = parse_json_cell(effect.get("absolute_delta"))
            if component_index is not None:
                absolute = absolute[component_index] if isinstance(absolute, list) and len(absolute) > component_index else None
            iqr = numeric_iqr(natural.get((str(effect["participant_id"]), str(effect["condition"]), family), []))
            ratio = float(absolute) / iqr if is_finite_number(absolute) and iqr is not None and iqr > 0 else None
            details.append({
                "protocol_id": PROTOCOL_ID, "participant_id": effect["participant_id"],
                "run_id": effect["run_id"], "condition": effect["condition"],
                "window_id": effect["window_id"], "duration_seconds": 4,
                "family_id": family, "component_index": component_index,
                "component_name": (
                    (parse_json_cell(effect.get("baseline_component_names")) or [])[component_index]
                    if component_index is not None and isinstance(parse_json_cell(effect.get("baseline_component_names")), list)
                    and len(parse_json_cell(effect.get("baseline_component_names"))) > component_index else None
                ),
                "noise_fraction_of_channel_rms": float(effect["noise_fraction_of_channel_rms"]),
                "absolute_change": absolute, "natural_variation_iqr": iqr,
                "absolute_change_over_natural_iqr": ratio,
                "qualified": False,
            })
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in details:
        key = (row["family_id"], row["component_index"], row["noise_fraction_of_channel_rms"])
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        ratios = [float(row["absolute_change_over_natural_iqr"]) for row in group
                  if is_finite_number(row["absolute_change_over_natural_iqr"])]
        deltas = [float(row["absolute_change"]) for row in group if is_finite_number(row["absolute_change"])]
        summaries.append({
            "protocol_id": PROTOCOL_ID, "family_id": key[0], "component_index": key[1],
            "noise_fraction_of_channel_rms": key[2], "windows_requested": len(group),
            "windows_with_nonzero_natural_iqr": len(ratios),
            "median_absolute_change": float(np.median(deltas)) if deltas else None,
            "median_absolute_change_over_natural_iqr": float(np.median(ratios)) if ratios else None,
            "qualified": False,
        })
    return details, summaries


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _render_report(
    *, accounting: Sequence[Mapping[str, Any]], equal_summary: Sequence[Mapping[str, Any]],
    dynamics: Sequence[Mapping[str, Any]], continuity: Sequence[Mapping[str, Any]],
    synthetic: Mapping[str, Any], gate_d_status: str,
) -> str:
    class_text = "; ".join(
        f"{row['record_class']}: {row['family_rows']} family rows, {row['unique_windows']} windows"
        for row in accounting
    )
    rank_pass = sum(bool(row["nominal_transition_floor_met"]) for row in dynamics)
    rank_full = sum(int(row["raw_design_rank"]) == 9 for row in dynamics)
    continuity_counts = ", ".join(
        f"{row['participant_id']}: {row['declared_boundary_events']} boundary marker(s), "
        f"{row['declared_bad_channels']} declared bad channel(s)"
        for row in continuity
    )
    return "\n".join([
        "# SL-013 bounded implementation report", "",
        "**Status:** descriptive four-recording technical investigation complete; outputs remain unqualified.", "",
        "## Scope and accounting", "",
        "The run used only the four technically selected recordings and verified the frozen WINDOW-SUPPORT-001 parent hash and its measurement hash. No clinical metadata was read or joined. The parent was left unchanged.",
        f"Parent class reconciliation: {class_text}.",
        f"The parent contains {sum(int(row['family_rows']) for row in accounting)} classified family rows; cadence and matched-only classes are both explicit in `record_class_accounting.csv`.", "",
        "## Equal-span and common-seven views", "",
        f"Gate A emitted {len(equal_summary)} anchor × duration × scalar-family comparisons. Geometric coverage is recorded per comparison; these short-window summaries and 20-second values are not mathematically equivalent in general.",
        f"Gate B emitted the fixed coordinate order `{', '.join(SCALAR_FAMILIES)}` and retained AR separately at its original 1/3/5/9-component duration-specific shape. This does not alter the registry or canonical chart.", "",
        "## Fast dynamics", "",
        f"The 4-second support audit includes {len(dynamics)} participant × condition × state-block trajectories; {rank_pass} met the necessary 9-transition count and {rank_full} had a raw affine input design of numerical rank 9.",
        "Rank and condition are support diagnostics only. No neural Jacobian was estimated. Perturbation changes are normalized by within-recording 4-second cadence IQR and have no threshold or reliability interpretation.",
        f"Source annotation audit: {continuity_counts}. Available BIDS annotations do not prove physical continuity in unmarked intervals.",
        f"Gate C: HOLD because physical continuity is unverified. Gate D: {gate_d_status}; no forecasting scores were generated.",
        f"The synthetic known-dynamics recovery check {'passed' if synthetic.get('full_rank_recovery_pass') and synthetic.get('duplicate_coordinate_control_rejected') else 'did not pass'}; this validates only the numerical diagnostic on synthetic data.", "",
        "## Limits", "",
        "Four recordings cannot establish measurement reliability, a preferred duration, useful biological dynamics, cross-dataset equivalence, clinical usefulness, or psychosis qualification. The short-window outputs remain experimental and all records have `qualified=false`.", "",
        "See `manifest.json` for input, code, software, output hashes and exact run status.", "",
    ])


def run(
    *, output_dir: Path = DEFAULT_OUTPUT_DIR,
    parent_run_dir: Path = BASELINE_RUN_DEFAULT,
    perturbation_run_dir: Path = PERTURBATION_RUN_DEFAULT,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing_to_overwrite_nonempty_output_dir:{output_dir}")
    if not protocol_path.is_file():
        raise FileNotFoundError("frozen_SL013_protocol_missing")
    parent_manifest, rows = _load_parent_rows(parent_run_dir)
    selection_manifest_path = parent_run_dir / "selection_manifest.json"
    selection_manifest = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    selection_records = selection_manifest.get("records")
    if not isinstance(selection_records, list) or [str(row.get("participant_id")) for row in selection_records] != list(PARTICIPANTS):
        raise ValueError("parent_selection_records_mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "protocol_frozen.md").write_bytes(protocol_path.read_bytes())
    start_time = time.perf_counter()

    accounting_detail, accounting_summary = build_row_accounting(rows)
    common7, ar_descriptors = build_common7_and_ar(rows)
    subwindows, equal_summary, equal_source_hashes, equal_compute_seconds = build_equal_span(
        rows=rows, parent_run_dir=parent_run_dir, selection_records=selection_records
    )
    dynamics = build_fast_dynamics(rows)
    continuity, continuity_source_hashes = source_continuity_audit(selection_records)
    perturb_details, perturb_summary = normalized_perturbation_sensitivity(
        rows=rows, perturbation_run_dir=perturbation_run_dir
    )
    synthetic = synthetic_recovery()
    continuity_confirmed = all(row.get("physical_continuity_confirmed") for row in continuity)
    gate_c_status = "PASS" if continuity_confirmed else "HOLD_SOURCE_PHYSICAL_CONTINUITY_UNVERIFIED"
    gate_d_status = "NOT_RUN_GATE_C_HOLD" if gate_c_status != "PASS" else "ELIGIBLE_FOR_SEPARATE_FORECAST_PROTOCOL"
    (output_dir / "gate_d_status.json").write_text(json.dumps({
        "protocol_id": PROTOCOL_ID, "gate_c_status": gate_c_status,
        "gate_d_status": gate_d_status, "forecast_scores_emitted": False,
        "reason": "physical continuity not established by available source annotations",
        "qualified": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_dir / "record_class_accounting.csv", accounting_detail)
    _write_csv(output_dir / "record_class_summary.csv", accounting_summary)
    _write_csv(output_dir / "common7_measurements.csv", common7)
    _write_csv(output_dir / "duration_ar_descriptors.csv", ar_descriptors)
    _write_csv(output_dir / "equal_span_subwindows.csv", subwindows)
    _write_csv(output_dir / "equal_span_summary.csv", equal_summary)
    _write_csv(output_dir / "fast_dynamics_design.csv", dynamics)
    _write_csv(output_dir / "source_continuity_audit.csv", continuity)
    _write_csv(output_dir / "normalized_perturbation_sensitivity.csv", perturb_details)
    _write_csv(output_dir / "normalized_perturbation_summary.csv", perturb_summary)
    _write_json(output_dir / "synthetic_recovery.json", synthetic)
    _write_json(output_dir / "parent_class_accounting.json", {
        "parent_manifest_sha256": BASELINE_MANIFEST_SHA256,
        "parent_measurement_sha256": parent_manifest["output_hashes"]["window_measurements.jsonl.gz"],
        "parent_total_family_rows": len(rows),
        "parent_cadence_family_rows": sum(int(row["family_rows"]) for row in accounting_detail
                                           if row["record_class"].startswith("cadence")),
        "parent_matched_only_family_rows": sum(int(row["family_rows"]) for row in accounting_detail
                                               if row["record_class"] == "matched_only"),
        "record_class_summary": accounting_summary,
        "qualified": False,
    })
    code_paths = {
        "runner": Path(__file__).resolve(),
        "parent_runner": Path(parent.__file__).resolve(),
        "gate_b_runner": Path(gate_b.__file__).resolve(),
        "gate_c1_runner": Path(gate_c1.__file__).resolve(),
        "candidate_grid": parent.REPO_ROOT / "mndm" / "src" / "mndm" / "mnps" / "v3_2" / "candidate_grid.py",
    }
    output_names = [
        "protocol_frozen.md", "record_class_accounting.csv", "record_class_summary.csv",
        "common7_measurements.csv", "duration_ar_descriptors.csv", "equal_span_subwindows.csv",
        "equal_span_summary.csv", "fast_dynamics_design.csv", "source_continuity_audit.csv",
        "normalized_perturbation_sensitivity.csv", "normalized_perturbation_summary.csv",
        "synthetic_recovery.json", "gate_d_status.json", "parent_class_accounting.json",
    ]
    manifest = {
        "protocol_id": PROTOCOL_ID, "status": "COMPLETED_BOUNDED_SL013_INVESTIGATION",
        "qualified": False, "clinical_metadata_read": False, "analyst_blinded": False,
        "participants_processed": list(PARTICIPANTS), "parent_manifest_sha256": BASELINE_MANIFEST_SHA256,
        "parent_measurement_sha256": parent_manifest["output_hashes"]["window_measurements.jsonl.gz"],
        "perturbation_manifest_sha256": sha256_file(perturbation_run_dir / "manifest.json"),
        "perturbation_effects_sha256": sha256_file(perturbation_run_dir / "perturbation_effects.csv"),
        "protocol_sha256": sha256_file(output_dir / "protocol_frozen.md"),
        "selection_manifest_sha256": sha256_file(selection_manifest_path),
        "parent_family_rows": len(rows), "record_class_accounting": accounting_summary,
        "cadence_and_matched_family_rows": sum(int(row["family_rows"]) for row in accounting_detail
                                                 if row["record_class"] == "cadence_and_matched"),
        "cadence_only_family_rows": sum(int(row["family_rows"]) for row in accounting_detail
                                         if row["record_class"] == "cadence_only"),
        "matched_only_family_rows": sum(int(row["family_rows"]) for row in accounting_detail
                                         if row["record_class"] == "matched_only"),
        "common7_rows": len(common7), "duration_ar_descriptor_rows": len(ar_descriptors),
        "equal_span_subwindow_rows": len(subwindows), "equal_span_comparison_rows": len(equal_summary),
        "equal_span_recomputed_windows_seconds": equal_compute_seconds,
        "fast_dynamics_trajectory_rows": len(dynamics),
        "gate_c_status": gate_c_status, "gate_d_status": gate_d_status,
        "synthetic_recovery": synthetic,
        "source_hashes": {**continuity_source_hashes, **equal_source_hashes},
        "implementation_code_sha256": {name: sha256_file(path) for name, path in code_paths.items()},
        "software": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "scipy": importlib.metadata.version("scipy")},
        "elapsed_seconds": time.perf_counter() - start_time,
        "output_hashes": {name: sha256_file(output_dir / name) for name in output_names},
    }
    (output_dir / "SL-013_REPORT.md").write_text(_render_report(
        accounting=accounting_summary, equal_summary=equal_summary, dynamics=dynamics,
        continuity=continuity, synthetic=synthetic, gate_d_status=gate_d_status,
    ), encoding="utf-8")
    output_names.append("SL-013_REPORT.md")
    manifest["output_hashes"] = {name: sha256_file(output_dir / name) for name in output_names}
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--parent-run-dir", type=Path, default=BASELINE_RUN_DEFAULT)
    parser.add_argument("--perturbation-run-dir", type=Path, default=PERTURBATION_RUN_DEFAULT)
    parser.add_argument("--protocol-path", type=Path, default=DEFAULT_PROTOCOL_PATH)
    args = parser.parse_args(argv)
    manifest = run(output_dir=args.output_dir, parent_run_dir=args.parent_run_dir,
                   perturbation_run_dir=args.perturbation_run_dir, protocol_path=args.protocol_path)
    print(json.dumps({
        "protocol_id": PROTOCOL_ID, "status": manifest["status"],
        "output_dir": str(args.output_dir.resolve()),
        "parent_family_rows": manifest["parent_family_rows"],
        "equal_span_comparison_rows": manifest["equal_span_comparison_rows"],
        "gate_c_status": manifest["gate_c_status"], "gate_d_status": manifest["gate_d_status"],
        "qualified": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
