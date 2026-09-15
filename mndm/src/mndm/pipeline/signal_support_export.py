"""Conservative export helpers for executed signal-support provenance.

This module records observed feature-row bounds and upstream execution records.
It deliberately leaves original raw sample mapping, effective filter support,
and unavailable reference-fit scope unknown.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import math
import re
from ..signal_support_provenance import _jsonable

import pandas as pd


def validate_support_record(record: Mapping[str, Any], expected_file: str | None = None) -> tuple[bool, str | None]:
    """Validate the immutable source identity before persistence."""
    if record.get("schema") != "mndm.signal_support_provenance.v1":
        return False, "support_schema_mismatch"
    source = record.get("source")
    if not isinstance(source, Mapping) or not source.get("path") or not source.get("exists"):
        return False, "support_source_identity_missing"
    if expected_file and Path(str(source.get("path"))).name != Path(expected_file).name:
        return False, "support_source_identity_mismatch"
    digest = str(source.get("sha256", ""))
    if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        return False, "support_source_hash_unverified"
    if source.get("hash_stable") is not True:
        return False, "support_source_hash_unstable"
    after = str(source.get("sha256_after", ""))
    if after.lower() != digest.lower():
        return False, "support_source_hash_unstable"
    return True, None


def build_signal_support_export(
    frame: pd.DataFrame,
    support_records: Mapping[str, Mapping[str, Any]] | list[Mapping[str, Any]],
    *,
    feature_baselines: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build additive, fail-closed support metadata for an exported feature frame."""
    by_file: dict[str, Mapping[str, Any]] = {}
    if isinstance(support_records, Mapping):
        by_file = {Path(str(k)).name: v for k, v in support_records.items() if isinstance(v, Mapping)}
    else:
        for record in support_records or []:
            if isinstance(record, Mapping) and record.get("raw_file"):
                by_file[Path(str(record["raw_file"])).name] = record

    epoch_rows: list[dict[str, Any]] = []
    quality_intervals: list[dict[str, Any]] = []
    for name, rec in by_file.items():
        if not validate_support_record(rec, name)[0]:
            continue
        intervals = rec.get("source_quality_intervals") if isinstance(rec, Mapping) else None
        if isinstance(intervals, Mapping):
            intervals = intervals.get("intervals", [])
        if isinstance(intervals, list):
            quality_intervals.extend(_jsonable(v) for v in intervals if isinstance(v, Mapping))
    for row_index, (_, row) in enumerate(frame.reset_index(drop=True).iterrows()):
        raw_file = Path(str(row.get("file", ""))).name if row.get("file") is not None else None
        rec = dict(by_file.get(raw_file, {}))
        valid_record, validation_reason = validate_support_record(rec, raw_file)
        if not valid_record:
            rec = {}
        t_start = row.get("t_start")
        t_end = row.get("t_end")
        overlaps = []
        offset = None
        crop = rec.get("crop", {})
        if rec.get("feature_time_clock", {}).get("reference") == "preprocessed_array_start":
            if crop.get("applied") is True:
                offset = crop.get("actual_tmin_sec")
            elif crop.get("reason") == "not_configured":
                offset = 0.0
        valid_bounds = all(isinstance(v, (int, float)) and math.isfinite(v) for v in (t_start, t_end)) and t_end > t_start
        mapped = valid_bounds and isinstance(offset, (int, float)) and math.isfinite(offset)
        if mapped:
            for interval in quality_intervals:
                try:
                    interval_file = Path(str(interval.get("raw_file", ""))).name
                    if interval_file != raw_file:
                        continue
                    if interval.get("clock") != "original_raw_seconds":
                        continue
                    lo, hi = float(interval["start_sec"]), float(interval["end_sec"])
                    if hi > float(t_start) + offset and lo < float(t_end) + offset:
                        overlaps.append({"interval": interval, "overlap_status": "overlaps_nominal_bounds"})
                except (KeyError, TypeError, ValueError):
                    continue
        epoch_rows.append({
            "row_index": int(row_index),
            "epoch_id": _jsonable(row.get("epoch_id")),
            "raw_file": raw_file,
            "nominal_time_start_sec": _jsonable(t_start),
            "nominal_time_end_sec": _jsonable(t_end),
            "original_sample_start": None,
            "original_sample_end": None,
            "sample_mapping_status": "unknown",
            "sample_mapping_reason": "original_raw_sample_mapping_not_recorded",
            "support_status": "unknown",
            "support_reason": rec.get("temporal_support", {}).get("reason", "support_not_recorded")
            if isinstance(rec.get("temporal_support"), Mapping) else "support_record_not_available",
            "source_identity": _jsonable(rec.get("source")),
            "source_clock": _jsonable({
                "input_sfreq_hz": rec.get("input_sfreq_hz"),
                "target_sfreq_hz": rec.get("target_sfreq_hz"),
            }),
            "source_quality_overlap": overlaps,
            "source_quality_overlap_status": "nominal_bounds_assessed" if mapped and rec.get("source_quality_intervals", {}).get("status") in {"observed", "none_observed"} else "unknown",
            "nominal_original_time_start_sec": float(t_start) + offset if mapped else None,
            "nominal_original_time_end_sec": float(t_end) + offset if mapped else None,
            "nominal_clock_mapping_status": "crop_offset_recorded" if mapped else "unknown",
            "upstream_operation_ids": [op.get("operation_id") for op in rec.get("operations", [])],
            "operation_scope": "recording_operations; not a certified per-feature dependency graph",
            "execution_validation_reason": validation_reason,
        })

    fit = {}
    for name, baseline in (feature_baselines or {}).items():
        if isinstance(baseline, Mapping):
            fit[str(name)] = {
                "center": _jsonable(baseline.get("standardization_center")),
                "scale": _jsonable(baseline.get("standardization_scale")),
                "fit_scope": _jsonable(baseline.get("fit_scope", "unknown")),
                "fit_population_ids": _jsonable(baseline.get("fit_population_ids")),
                "fit_population_hash": _jsonable(baseline.get("fit_population_hash")),
                "fit_population_count": _jsonable(baseline.get("fit_population_count")),
                "fit_input_row_count": _jsonable(baseline.get("fit_input_row_count")),
                "fit_population_encoding": _jsonable(baseline.get("fit_population_encoding")),
                "transformation_applied": _jsonable(baseline.get("transformation_applied")),
                "anchor_id": _jsonable(baseline.get("anchor_id")),
                "anchor_hash": _jsonable(baseline.get("anchor_hash")),
                "fit_status": "recorded" if baseline.get("fit_population_hash") else "unknown",
            }

    execution_records = []
    for raw_file, rec in sorted(by_file.items()):
        ok, reason = validate_support_record(rec, raw_file)
        execution_records.append({"raw_file": raw_file, "validation_status": "valid" if ok else "unknown", "validation_reason": reason, "record": _jsonable(rec)})
    return {
        "schema": "mndm.signal_support_export.v1",
        "status": "recorded" if epoch_rows else "unknown",
        "temporal_support_status": "unknown",
        "temporal_support_reason": "effective filter support is not certified by epoch bounds",
        "per_epoch_input_extent": epoch_rows,
        "execution_records": execution_records,
        "source_quality_intervals": quality_intervals,
        "source_quality_status": "recorded" if quality_intervals else "not_recorded",
        "reference_fit_population": {
            "status": "recorded" if any(v.get("fit_status") == "recorded" for v in fit.values()) else "unknown",
            "features": fit,
        },
    }
