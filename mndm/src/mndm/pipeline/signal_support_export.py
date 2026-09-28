"""Conservative export helpers for executed signal-support provenance.

This module records observed feature-row bounds and upstream execution records.
It persists explicit nominal source-grid mapping only when the feature extractor
provided actual array indices and a validated source clock. Effective filter
support is bounded only when a complete operation-specific dependency map is
provided; unavailable reference-fit scope remains unknown.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
import math
import re
from ..signal_support_provenance import _jsonable

import numpy as np
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
    associated = source.get("associated_signal_files")
    if associated is not None:
        if not isinstance(associated, list) or not associated:
            return False, "support_associated_signal_identity_missing"
        for item in associated:
            if not isinstance(item, Mapping) or item.get("exists") is not True:
                return False, "support_associated_signal_identity_missing"
            payload_hash = str(item.get("sha256", ""))
            if not re.fullmatch(r"[0-9a-fA-F]{64}", payload_hash):
                return False, "support_associated_signal_hash_unverified"
            if (
                item.get("hash_stable") is not True
                or str(item.get("sha256_after", "")).lower() != payload_hash.lower()
            ):
                return False, "support_associated_signal_hash_unstable"
    return True, None


def build_signal_support_export(
    frame: pd.DataFrame,
    support_records: Mapping[str, Mapping[str, Any]] | list[Mapping[str, Any]],
    *,
    feature_baselines: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build additive, fail-closed support metadata for an exported feature frame."""
    by_file: dict[str, Mapping[str, Any]] = {}
    candidates_by_file: dict[str, list[Mapping[str, Any]]] = {}
    raw_records: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(support_records, Mapping):
        for key, value in support_records.items():
            if not isinstance(value, Mapping):
                continue
            name = Path(str(key)).name
            candidates_by_file.setdefault(name, []).append(value)
            raw_records.append((name, value))
    else:
        for record in support_records or []:
            if not isinstance(record, Mapping):
                continue
            source = record.get("source", {})
            source_path = source.get("path") if isinstance(source, Mapping) else None
            raw_name = record.get("raw_file") or source_path
            if raw_name:
                name = Path(str(raw_name)).name
                candidates_by_file.setdefault(name, []).append(record)
                raw_records.append((name, record))
    ambiguous_files = {name for name, records in candidates_by_file.items() if len(records) > 1}
    by_file = {
        name: records[0]
        for name, records in candidates_by_file.items()
        if len(records) == 1
    }

    epoch_rows: list[dict[str, Any]] = []
    quality_intervals: list[dict[str, Any]] = []
    for name, rec in by_file.items():
        if not validate_support_record(rec, name)[0]:
            continue
        intervals = rec.get("source_quality_intervals") if isinstance(rec, Mapping) else None
        if isinstance(intervals, Mapping):
            intervals = intervals.get("intervals", [])
        if isinstance(intervals, list):
            for interval in intervals:
                if isinstance(interval, Mapping):
                    item = dict(_jsonable(interval))
                    item.setdefault("raw_file", name)
                    item.setdefault("source_file_sha256", rec.get("source", {}).get("sha256"))
                    quality_intervals.append(item)
    for row_index, (_, row) in enumerate(frame.reset_index(drop=True).iterrows()):
        raw_file = Path(str(row.get("file", ""))).name if row.get("file") is not None else None
        rec = dict(by_file.get(raw_file, {}))
        if raw_file in ambiguous_files:
            valid_record, validation_reason = False, "support_basename_ambiguous"
        elif rec.get("reason") == "duplicate_raw_basename_in_file_index":
            valid_record, validation_reason = False, "duplicate_raw_basename_in_file_index"
        else:
            valid_record, validation_reason = validate_support_record(rec, raw_file)
        if not valid_record:
            rec = {}
        t_start = row.get("t_start")
        t_end = row.get("t_end")
        overlaps = []
        offset = None
        crop = rec.get("crop", {})
        if not isinstance(crop, Mapping):
            crop = {}
        feature_clock = rec.get("feature_time_clock", {})
        if not isinstance(feature_clock, Mapping):
            feature_clock = {}
        row_mapping_status = str(row.get("source_sample_mapping_status", "unknown"))
        row_mapping_reason = str(row.get("source_sample_mapping_reason", ""))
        row_mapping_status_value = row.get("source_sample_mapping_status")
        row_mapping_reason_value = row.get("source_sample_mapping_reason")
        row_has_mapping_contract = (
            row_mapping_status_value is not None
            and not (isinstance(row_mapping_status_value, (float, np.floating)) and math.isnan(float(row_mapping_status_value)))
        ) or (
            row_mapping_reason_value is not None
            and str(row_mapping_reason_value).strip().lower() not in {"", "nan", "none"}
        )
        explicit_source_time = (
            row.get("nominal_source_time_start_sec"),
            row.get("nominal_source_time_end_sec"),
        )
        explicit_time_valid = all(
            isinstance(value, (int, float, np.number)) and math.isfinite(float(value))
            for value in explicit_source_time
        ) and float(explicit_source_time[1]) > float(explicit_source_time[0])
        explicit_mapping = bool(valid_record and row_mapping_status == "nominal_bounds_mapped" and explicit_time_valid)
        refuses_continuous_clock = row_mapping_reason == "flattened_eeglab_epochs_have_no_continuous_source_sample_map"
        if (
            not explicit_mapping
            and not row_has_mapping_contract
            and not refuses_continuous_clock
            and crop.get("source_sample_mapping_reason") != "flattened_eeglab_epochs_have_no_continuous_source_sample_map"
            and feature_clock.get("reference") == "preprocessed_array_start"
        ):
            if crop.get("applied") is True:
                offset = crop.get("actual_tmin_from_source_start_sec")
                if offset is None:
                    raw_clock = rec.get("raw_clock_before_crop", {})
                    source_first_time = raw_clock.get("first_time_sec") if isinstance(raw_clock, Mapping) else None
                    try:
                        offset = float(crop.get("actual_tmin_sec")) - float(source_first_time) if source_first_time is not None else crop.get("actual_tmin_sec")
                    except (TypeError, ValueError):
                        offset = None
            elif crop.get("reason") == "not_configured":
                offset = 0.0
        valid_bounds = all(isinstance(v, (int, float)) and math.isfinite(v) for v in (t_start, t_end)) and t_end > t_start
        mapped = explicit_mapping or (valid_bounds and isinstance(offset, (int, float)) and math.isfinite(offset))
        mapped_start = float(explicit_source_time[0]) if explicit_mapping else (float(t_start) + offset if mapped else None)
        mapped_end = float(explicit_source_time[1]) if explicit_mapping else (float(t_end) + offset if mapped else None)
        if mapped:
            for interval in quality_intervals:
                try:
                    interval_file = Path(str(interval.get("raw_file", ""))).name
                    if interval_file != raw_file:
                        continue
                    if interval.get("clock") != "original_raw_seconds":
                        continue
                    lo, hi = float(interval["start_sec"]), float(interval["end_sec"])
                    if hi > mapped_start and lo < mapped_end:
                        overlaps.append({"interval": interval, "overlap_status": "overlaps_nominal_bounds"})
                except (KeyError, TypeError, ValueError):
                    continue
        source = rec.get("source", {}) if isinstance(rec.get("source"), Mapping) else {}
        sample_start = row.get("nominal_source_sample_start") if explicit_mapping else None
        sample_end = row.get("nominal_source_sample_end_exclusive") if explicit_mapping else None
        try:
            sample_start = int(sample_start) if sample_start is not None and math.isfinite(float(sample_start)) else None
            sample_end = int(sample_end) if sample_end is not None and math.isfinite(float(sample_end)) else None
        except (TypeError, ValueError, OverflowError):
            sample_start, sample_end = None, None
        if sample_start is None or sample_end is None or sample_end <= sample_start:
            sample_start, sample_end = None, None
            exported_mapping_status = "unknown"
            exported_mapping_reason = row_mapping_reason or validation_reason or "source_sample_mapping_unavailable"
        else:
            exported_mapping_status = "nominal_bounds_mapped"
            exported_mapping_reason = row.get("source_sample_mapping_reason")
        recording_id = row.get("recording_id")
        recording_id_method = "feature_row"
        if recording_id is None or (isinstance(recording_id, float) and math.isnan(recording_id)):
            recording_id = Path(raw_file).stem if raw_file else None
            recording_id_method = "raw_file_stem_fallback"
        gap_spans = row.get("input_sample_gap_spans", [])
        if isinstance(gap_spans, str):
            try:
                parsed_gap_spans = json.loads(gap_spans)
                gap_spans = parsed_gap_spans if isinstance(parsed_gap_spans, list) else []
            except (TypeError, ValueError):
                gap_spans = []
        contributor_status = row.get("actual_sample_contributors_status", "unknown")
        effective_support = rec.get("effective_time_support", {})
        support_status = (
            "finite_causal_conservative_bound"
            if contributor_status == "finite_causal_conservative_bound"
            and isinstance(effective_support, Mapping)
            and effective_support.get("status") == "finite_causal"
            else "unknown"
        )
        epoch_rows.append({
            "row_index": int(row_index),
            "epoch_id": _jsonable(row.get("epoch_id")),
            "raw_file": raw_file,
            "recording_id": _jsonable(recording_id),
            "recording_id_method": recording_id_method,
            "source_file_sha256": source.get("sha256") if valid_record else None,
            "nominal_time_start_sec": _jsonable(t_start),
            "nominal_time_end_sec": _jsonable(t_end),
            "original_sample_start": sample_start,
            "original_sample_end": sample_end,
            "original_sample_interval_convention": "half_open_0_based_source_array_indices_within_nominal_window",
            "source_sample_rate_hz": _jsonable(row.get("source_sfreq_hz")),
            "source_first_samp": _jsonable(row.get("source_first_samp")),
            "source_sample_absolute_start": _jsonable(row.get("nominal_source_sample_absolute_start")) if explicit_mapping else None,
            "source_sample_absolute_end_exclusive": _jsonable(row.get("nominal_source_sample_absolute_end_exclusive")) if explicit_mapping else None,
            "sample_mapping_status": exported_mapping_status,
            "sample_mapping_reason": _jsonable(exported_mapping_reason),
            "sample_mapping_scope": "nominal_time_bounds_on_original_sample_grid; excludes filter_or_resampler_dependencies",
            "actual_sample_contributors_status": _jsonable(row.get("actual_sample_contributors_status", "unknown")),
            "actual_sample_contributors_reason": _jsonable(row.get("actual_sample_contributors_reason", "operation_dependency_map_not_recorded")),
            "actual_source_sample_start": _jsonable(row.get("actual_source_sample_start")),
            "actual_source_sample_end_exclusive": _jsonable(row.get("actual_source_sample_end_exclusive")),
            "actual_source_time_start_sec": _jsonable(row.get("actual_source_time_start_sec")),
            "actual_source_time_end_sec": _jsonable(row.get("actual_source_time_end_sec")),
            "actual_contributor_interval_convention": "half_open_0_based_source_array_indices; conservative_dependency_superset",
            "preprocessed_sample_start": _jsonable(row.get("preprocessed_sample_start")),
            "preprocessed_sample_end_exclusive": _jsonable(row.get("preprocessed_sample_end_exclusive")),
            "preprocessed_sfreq_hz": _jsonable(row.get("preprocessed_sfreq_hz")),
            "input_sample_finite_status": _jsonable(row.get("input_sample_finite_status", "not_assessed")),
            "input_sample_finite_fraction": _jsonable(row.get("input_sample_finite_fraction")),
            "input_sample_gap_count": _jsonable(row.get("input_sample_gap_count")),
            "input_sample_gap_spans": _jsonable(gap_spans),
            "input_sample_invalid_reason": _jsonable(row.get("input_sample_invalid_reason")),
            "support_status": support_status,
            "support_reason": _jsonable(row.get("actual_sample_contributors_reason")) if support_status != "unknown" else (
                rec.get("temporal_support", {}).get("reason", "support_not_recorded")
                if isinstance(rec.get("temporal_support"), Mapping) else "support_record_not_available"
            ),
            "source_identity": _jsonable(rec.get("source")),
            "source_signal_payload_files": _jsonable(source.get("associated_signal_files")) if valid_record else None,
            "source_clock": _jsonable({
                "input_sfreq_hz": rec.get("input_sfreq_hz"),
                "target_sfreq_hz": rec.get("target_sfreq_hz"),
                "raw_clock_before_crop": rec.get("raw_clock_before_crop"),
                "raw_clock_after_crop": rec.get("raw_clock_after_crop"),
                "recording_time_reference": rec.get("recording_time_reference"),
            }),
            "source_quality_overlap": overlaps,
            "source_quality_overlap_status": "nominal_bounds_assessed" if mapped and rec.get("source_quality_intervals", {}).get("status") in {"observed", "none_observed"} else "unknown",
            "nominal_original_time_start_sec": mapped_start,
            "nominal_original_time_end_sec": mapped_end,
            "nominal_clock_mapping_status": "preprocessed_sample_clock_mapped" if explicit_mapping else ("crop_offset_recorded" if mapped else "unknown"),
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
    for raw_file, rec in sorted(
        raw_records,
        key=lambda item: (
            item[0],
            str(item[1].get("source", {}).get("path", ""))
            if isinstance(item[1].get("source"), Mapping)
            else "",
        ),
    ):
        if raw_file in ambiguous_files:
            ok, reason = False, "support_basename_ambiguous"
        else:
            ok, reason = validate_support_record(rec, raw_file)
        execution_records.append({"raw_file": raw_file, "validation_status": "valid" if ok else "unknown", "validation_reason": reason, "record": _jsonable(rec)})
    finite_causal_support_complete = bool(epoch_rows) and all(
        row.get("support_status") == "finite_causal_conservative_bound"
        and row.get("actual_source_sample_start") is not None
        and row.get("actual_source_sample_end_exclusive") is not None
        for row in epoch_rows
    ) and bool(execution_records) and all(record.get("validation_status") == "valid" for record in execution_records)
    return {
        "schema": "mndm.signal_support_export.v1",
        "status": "recorded" if epoch_rows else "unknown",
        "temporal_support_status": "finite_causal_conservative_bound" if finite_causal_support_complete else "unknown",
        "temporal_support_reason": (
            "every epoch row has a validated conservative source-sample bound from recorded causal FIR operations"
            if finite_causal_support_complete
            else "effective filter support is not certified for every epoch row"
        ),
        "per_epoch_input_extent": epoch_rows,
        "execution_records": execution_records,
        "source_quality_intervals": quality_intervals,
        "source_quality_status": "recorded" if quality_intervals else "not_recorded",
        "reference_fit_population": {
            "status": "recorded" if any(v.get("fit_status") == "recorded" for v in fit.values()) else "unknown",
            "features": fit,
        },
    }
