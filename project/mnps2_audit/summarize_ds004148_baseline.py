"""Summarize frozen-anchor ds004148 EO/EC outputs under QC policy v1.1.

The script consumes already-produced H5 files and the separate, read-only
source QC windows table. It never rewrites or filters the H5 measurements.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path(
    r"M:/repos/NeuralManifoldDynamics/project/mnps2_audit/outputs/"
    r"20261116_ds004148_v3_4_frozen_ec_baseline"
)
PROCESSED = ROOT / "processed" / "ds004148"
QC_WINDOWS = ROOT / "qc_windows.csv"
QC_RECORDINGS = ROOT / "qc_recordings.csv"
SELECTION = ROOT / "cohort_selection_manifest.csv"
ANCHOR_ID = "ds005385_ses1_pretask_EyesClosed_20261111_v1"
ANCHOR_SHA256 = "79da8c208a6c9240e84cefee7abc9a26c3121d59c1c86bc3324d33172193d1ab"
MIN_WINDOWS = 30
MIN_UNIQUE_SECONDS = 120.0
TASKS = ("eyesclosed", "eyesopen")
TASK_DISPLAY = {"eyesclosed": "EC", "eyesopen": "EO"}
SUBJECT_RE = re.compile(r"^(sub-\d+)_ses-session1_(eyesclosed|eyesopen)\.h5$")

RAW_UNITS = {
    "delta_theta_level": "log10(microvolt^2)",
    "alpha_level": "log10(microvolt^2)",
    "beta_alpha_ratio": "dimensionless ratio",
    "gamma_level": "log10(microvolt^2)",
    "hjorth_mobility": "dimensionless per sample",
    "alpha_theta_ratio": "dimensionless ratio",
    "permutation_entropy": "dimensionless",
    "alpha_centroid_hz": "Hz",
    "alpha_envelope_cv": "dimensionless",
}
COORDINATE_SOURCE_FEATURES = {
    "delta_theta_level": ("eeg_delta", "eeg_theta"),
    "alpha_level": ("eeg_alpha",),
    "beta_alpha_ratio": ("eeg_beta_alpha",),
    "gamma_level": ("eeg_gamma",),
    "hjorth_mobility": ("eeg_hjorth_mobility",),
    "alpha_theta_ratio": ("eeg_alpha_theta",),
    "permutation_entropy": ("eeg_permutation_entropy",),
    "alpha_centroid_hz": ("mnps_v3_4_alpha_centroid_hz",),
    "alpha_envelope_cv": ("mnps_v3_4_alpha_envelope_cv",),
}
EXPECTED_COORDINATE_SOURCE_UNITS = {
    "delta_theta_level": ("microvolt^2", "microvolt^2"),
    "alpha_level": ("microvolt^2",),
    "beta_alpha_ratio": ("dimensionless_ratio",),
    "gamma_level": ("microvolt^2",),
    "hjorth_mobility": ("dimensionless_per_sample",),
    "alpha_theta_ratio": ("dimensionless_ratio",),
    "permutation_entropy": ("dimensionless",),
    "alpha_centroid_hz": ("Hz",),
    "alpha_envelope_cv": ("dimensionless",),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return [decode(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return decode(value.item())
    return value


def dataset(h5: h5py.File, *paths: str) -> np.ndarray:
    for path in paths:
        if path in h5:
            return np.asarray(h5[path][()])
    raise KeyError(f"No H5 dataset found among paths: {paths}")


def scalar_text(h5: h5py.File, path: str) -> str:
    return str(decode(h5[path][()]))


def scalar_float(h5: h5py.File, path: str) -> float:
    return float(h5[path][()])


def pair_compatibility(
    ec: dict[str, Any], eo: dict[str, Any], manifest_row: dict[str, str]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for field, label in (
        ("same_ordered_channel_labels", "metadata_channel_labels_or_order"),
        ("same_declared_reference", "metadata_reference"),
        ("same_sampling_frequency_metadata", "metadata_sampling_frequency"),
        ("same_declared_units", "metadata_units"),
    ):
        if manifest_row.get(field, "").lower() != "true":
            reasons.append(label)
    for record, label in ((ec, "EC"), (eo, "EO")):
        if record["pair_signature"]["feature_extraction_implementation_status"] != "single_consistent":
            reasons.append(
                f"{label}_feature_extraction_identity_"
                f"{record['pair_signature']['feature_extraction_implementation_status']}"
            )
    for key in sorted(set(ec["pair_signature"]) | set(eo["pair_signature"])):
        if ec["pair_signature"].get(key) != eo["pair_signature"].get(key):
            reasons.append(key)
    return not reasons, reasons


def pair_exclusion_reason(
    ec: dict[str, Any],
    eo: dict[str, Any],
    realization_ok: bool,
    realization_reasons: list[str],
    additional_reasons: list[str] | None = None,
) -> str:
    reasons: list[str] = []
    for task, row in (("EC", ec), ("EO", eo)):
        n_key = (
            "qc_eligible_supported_windows"
            if "qc_eligible_supported_windows" in row
            else (
                "qc_eligible_finite_windows"
                if "qc_eligible_finite_windows" in row
                else "qc_eligible_complete_9d_windows"
            )
        )
        if int(row[n_key]) < MIN_WINDOWS:
            reasons.append(f"{task}_eligible_windows_below_{MIN_WINDOWS}")
        if float(row["qc_eligible_unique_seconds"]) + 1e-9 < MIN_UNIQUE_SECONDS:
            reasons.append(f"{task}_unique_seconds_below_{MIN_UNIQUE_SECONDS:g}")
    if not realization_ok:
        reasons.extend(f"realization_mismatch:{reason}" for reason in realization_reasons)
    reasons.extend(additional_reasons or [])
    return ";".join(reasons)


def qc_index(rows: list[dict[str, str]]) -> dict[tuple[str, str, float, float], dict[str, str]]:
    result: dict[tuple[str, str, float, float], dict[str, str]] = {}
    for row in rows:
        key = (
            row["subject"],
            row["task"],
            round(float(row["window_start_sec"]), 6),
            round(float(row["window_end_sec"]), 6),
        )
        if key in result:
            raise ValueError(f"Duplicate QC window key: {key}")
        result[key] = row
    return result


def eligible(row: dict[str, str]) -> bool:
    # Fail closed when the input or external detector was not computable.
    # Amplitude/PTP/noise descriptors remain descriptive.
    try:
        source_finite = float(row["finite_fraction"]) == 1.0
        detector_valid = str(row["muscle_detector_valid"]).lower() == "true"
        exceedance_fraction = float(row["muscle_zscore_exceedance_fraction"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        source_finite
        and detector_valid
        and np.isfinite(exceedance_fraction)
        and exceedance_fraction == 0.0
    )


def exclusion_reason(row: dict[str, str]) -> str:
    try:
        if float(row["finite_fraction"]) != 1.0:
            return "source_signal_nonfinite"
        if str(row["muscle_detector_valid"]).lower() != "true":
            return "muscle_detector_unassessed"
        exceedance_fraction = float(row["muscle_zscore_exceedance_fraction"])
    except (KeyError, TypeError, ValueError):
        return "muscle_detector_unassessed"
    if not np.isfinite(exceedance_fraction):
        return "muscle_detector_unassessed"
    return "muscle_score_gt_4_sample_overlap" if exceedance_fraction > 0 else ""


def unique_interval_seconds(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    merged_start, merged_end = intervals[0]
    total = 0.0
    for start, end in intervals[1:]:
        if start <= merged_end + 1e-9:
            merged_end = max(merged_end, end)
        else:
            total += merged_end - merged_start
            merged_start, merged_end = start, end
    return total + merged_end - merged_start


def support_pass(n_windows: int, seconds: float) -> bool:
    return n_windows >= MIN_WINDOWS and seconds + 1e-9 >= MIN_UNIQUE_SECONDS


def stats(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "paired_n": 0,
            "mean": math.nan,
            "sd_sample": math.nan,
            "median": math.nan,
            "q1": math.nan,
            "q3": math.nan,
            "iqr": math.nan,
            "mad_unscaled": math.nan,
            "min": math.nan,
            "max": math.nan,
            "n_negative": 0,
            "n_zero": 0,
            "n_positive": 0,
        }
    median = float(np.median(array))
    q1, q3 = (float(x) for x in np.quantile(array, [0.25, 0.75]))
    return {
        "paired_n": int(array.size),
        "mean": float(np.mean(array)),
        "sd_sample": float(np.std(array, ddof=1)) if array.size > 1 else math.nan,
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "mad_unscaled": float(np.median(np.abs(array - median))),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "n_negative": int(np.sum(array < 0)),
        "n_zero": int(np.sum(array == 0)),
        "n_positive": int(np.sum(array > 0)),
    }


def read_h5(path: Path) -> dict[str, Any]:
    match = SUBJECT_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected H5 filename: {path.name}")
    subject, task = match.groups()
    with h5py.File(path, "r") as h5:
        anchor_id = scalar_text(h5, "measurements/v3_4/provenance/anchor_id")
        anchor_sha = scalar_text(h5, "measurements/v3_4/provenance/anchor_manifest_sha256")
        anchor_status = scalar_text(h5, "measurements/v3_4/provenance/anchor_application_status")
        if anchor_id != ANCHOR_ID or anchor_sha != ANCHOR_SHA256:
            raise ValueError(f"Frozen anchor mismatch in {path}: {anchor_id}, {anchor_sha}")
        if anchor_status != "computed_exploratory_transfer":
            raise ValueError(f"Unexpected anchor application status in {path}: {anchor_status}")

        names = [str(decode(x)) for x in dataset(
            h5, "measurements/v3_4/coordinates/names", "measurements/v3_4/coordinate_names"
        )]
        raw = np.asarray(dataset(h5, "measurements/v3_4/raw_coordinates/values"), dtype=np.float64)
        raw_support = np.asarray(dataset(h5, "measurements/v3_4/raw_coordinates/support"), dtype=bool)
        anchored = np.asarray(dataset(h5, "measurements/v3_4/coordinates/values"), dtype=np.float64)
        anchored_valid = np.asarray(dataset(h5, "measurements/v3_4/coordinate_validity_mask"), dtype=bool)
        complete_9d = np.asarray(dataset(h5, "measurements/v3_4/complete_9d_mask"), dtype=bool)
        starts = np.asarray(dataset(h5, "window_start_sec", "window_start"), dtype=np.float64)
        ends = np.asarray(dataset(h5, "window_end_sec", "window_end"), dtype=np.float64)
        feature_names = [str(decode(x)) for x in dataset(h5, "features_raw/names")]
        feature_values = np.asarray(dataset(h5, "features_raw/values"), dtype=np.float64)
        identity_path = "measurements/v3_4/provenance/target_measurement_identity"
        unit_group = h5[f"{identity_path}/feature_units"]
        feature_units = {name: str(decode(unit_group[name][()])) for name in unit_group}
        implementation_path = "measurements/v3_4/provenance/implementation_hashes_at_summarize"
        implementation_hashes = {
            name: str(decode(h5[f"{implementation_path}/{name}"][()]))
            for name in h5[implementation_path]
        }
        feature_impl_window_path = "measurements/v3_4/support/implementation_hashes_by_window_json"
        feature_impl_identity = None
        feature_impl_status = "missing_per_window_identity"
        feature_impl_invalid_count = 0
        feature_impl_window_count = 0
        if feature_impl_window_path in h5:
            per_window_values = dataset(h5, feature_impl_window_path)
            feature_impl_window_count = int(per_window_values.size)
            normalized_feature_impl: list[str] = []
            parsed_feature_impl: list[dict[str, str]] = []
            required_hashes = {
                "candidate_estimators_sha256",
                "coordinate_assembly_sha256",
                "feature_row_assembly_sha256",
                "preprocessing_pipeline_sha256",
            }
            for value in per_window_values:
                try:
                    item = json.loads(str(decode(value)))
                    valid = (
                        isinstance(item, dict)
                        and required_hashes.issubset(item)
                        and all(
                            isinstance(item[key], str)
                            and re.fullmatch(r"[0-9a-f]{64}", item[key]) is not None
                            for key in required_hashes
                        )
                    )
                except (TypeError, ValueError, json.JSONDecodeError):
                    valid = False
                    item = None
                if not valid:
                    feature_impl_invalid_count += 1
                else:
                    normalized_feature_impl.append(
                        json.dumps(item, sort_keys=True, separators=(",", ":"))
                    )
                    parsed_feature_impl.append(item)
            if feature_impl_window_count != len(starts):
                feature_impl_status = "per_window_identity_count_mismatch"
            elif feature_impl_invalid_count:
                feature_impl_status = "missing_or_invalid_per_window_identity"
            elif normalized_feature_impl and len(set(normalized_feature_impl)) == 1:
                feature_impl_status = "single_consistent"
                feature_impl_identity = parsed_feature_impl[0]
            elif normalized_feature_impl:
                feature_impl_status = "heterogeneous_per_window_identity"
            else:
                feature_impl_status = "empty_per_window_identity"
        identity = f"{identity_path}/"
        manifest_json = json.loads(str(decode(h5["manifest_json"][()])))
        pair_signature = {
            "dataset_id": scalar_text(h5, f"{identity}dataset_id"),
            "modality": scalar_text(h5, f"{identity}modality"),
            "channel_count": int(h5[f"{identity}channel_count"][()]),
            "channel_order_hashes": [
                str(decode(x)) for x in dataset(h5, f"{identity}channel_order_hashes")
            ],
            "channel_compatibility_set_hash": scalar_text(
                h5, f"{identity}mnps_v3_4_channel_compatibility_set_hash"
            ),
            "source_surface_hash": scalar_text(h5, f"{identity}mnps_v3_4_source_surface_hash"),
            "reference": scalar_text(h5, f"{identity}mnps_v3_4_reference"),
            "reference_set_hash": scalar_text(h5, f"{identity}mnps_v3_4_reference_set_hash"),
            "implementation_hashes": implementation_hashes,
            "feature_extraction_implementation_status": feature_impl_status,
            "feature_extraction_implementation": feature_impl_identity,
            "coordinate_names": names,
            "manifest_sampling_and_window_definition": manifest_json.get("mnps", {}),
            "time_grid_definition": {
                "window_sec_config": scalar_float(
                    h5, "provenance/geometry_contract/time_grid/window_sec_config"
                ),
                "dt_sec_config": scalar_float(
                    h5, "provenance/geometry_contract/time_grid/dt_sec_config"
                ),
                "window_len_matches_config": bool(
                    h5["provenance/geometry_contract/time_grid/window_len_matches_config"][()]
                ),
                "dt_matches_config_formula": bool(
                    h5["provenance/geometry_contract/time_grid/dt_matches_config_formula"][()]
                ),
                "status": scalar_text(h5, "provenance/geometry_contract/time_grid/status"),
            },
        }

    expected_names = list(RAW_UNITS)
    if names != expected_names:
        raise ValueError(f"Coordinate names/order mismatch in {path}: {names}")
    n = len(starts)
    expected_shape = (n, len(expected_names))
    for label, values in (("raw", raw), ("raw_support", raw_support), ("anchored", anchored), ("anchored_valid", anchored_valid)):
        if values.shape != expected_shape:
            raise ValueError(f"{label} shape {values.shape} != {expected_shape} in {path}")
    if ends.shape != starts.shape or complete_9d.shape != starts.shape:
        raise ValueError(f"Window/support shape mismatch in {path}")
    if feature_values.shape != (n, len(feature_names)):
        raise ValueError(f"Raw-feature shape {feature_values.shape} does not match names/time grid in {path}")
    window_keys = [(round(float(start), 6), round(float(end), 6)) for start, end in zip(starts, ends)]
    if len(window_keys) != len(set(window_keys)):
        raise ValueError(f"Duplicate H5 window bounds in {path}")
    return {
        "subject": subject,
        "task": task,
        "path": path,
        "names": names,
        "raw": raw,
        "raw_support": raw_support,
        "anchored": anchored,
        "anchored_valid": anchored_valid,
        "complete_9d": complete_9d,
        "feature_names": feature_names,
        "feature_values": feature_values,
        "feature_units": feature_units,
        "coordinate_source_units": {
            coordinate: tuple(
                feature_units.get(feature, "not_recorded_in_measurement_provenance")
                for feature in COORDINATE_SOURCE_FEATURES[coordinate]
            )
            for coordinate in names
        },
        "pair_signature": pair_signature,
        "feature_extraction_implementation_window_count": feature_impl_window_count,
        "feature_extraction_implementation_invalid_window_count": feature_impl_invalid_count,
        "starts": starts,
        "ends": ends,
    }


def main() -> None:
    for path in (QC_WINDOWS, QC_RECORDINGS, SELECTION):
        if not path.is_file():
            raise FileNotFoundError(path)
    qc_rows = read_csv(QC_WINDOWS)
    qc = qc_index(qc_rows)
    qc_recording_rows = read_csv(QC_RECORDINGS)
    qc_recordings: dict[tuple[str, str], dict[str, str]] = {}
    for row in qc_recording_rows:
        key = (row["subject"], row["task"])
        if key in qc_recordings:
            raise ValueError(f"Duplicate QC recording key: {key}")
        qc_recordings[key] = row
    selection_rows = read_csv(SELECTION)
    selection_by_subject = {row["subject"]: row for row in selection_rows}
    expected_subjects = {f"sub-{number:02d}" for number in range(1, 61)}
    expected_recording_keys = {
        (subject, task) for subject in expected_subjects for task in TASKS
    }
    if set(qc_recordings) != expected_recording_keys:
        missing = sorted(expected_recording_keys - set(qc_recordings))
        extra = sorted(set(qc_recordings) - expected_recording_keys)
        raise ValueError(f"Expected 120 QC recording rows; missing={missing}, extra={extra}")
    if set(selection_by_subject) != expected_subjects:
        raise ValueError("Selection manifest does not contain exactly sub-01 through sub-60")
    if any(row["pair_included_for_computation"].lower() != "true" for row in selection_rows):
        raise ValueError("Metadata gate failed: one or more expected EO/EC file pairs are incomplete")

    h5_paths = sorted(PROCESSED.rglob("*.h5"))
    records = [read_h5(path) for path in h5_paths]
    by_key = {(r["subject"], r["task"]): r for r in records}
    expected_keys = {(subject, task) for subject in expected_subjects for task in TASKS}
    if set(by_key) != expected_keys or len(records) != len(expected_keys):
        missing = sorted(expected_keys - set(by_key))
        extra = sorted(set(by_key) - expected_keys)
        raise ValueError(f"Expected 120 unique H5 files; missing={missing}, extra={extra}, total={len(records)}")
    h5_window_keys = {
        (record["subject"], record["task"], round(float(start), 6), round(float(end), 6))
        for record in records
        for start, end in zip(record["starts"], record["ends"])
    }
    if set(qc) != h5_window_keys:
        missing_qc = sorted(h5_window_keys - set(qc))[:10]
        extra_qc = sorted(set(qc) - h5_window_keys)[:10]
        raise ValueError(
            "QC/H5 windows do not correspond bidirectionally; "
            f"missing_qc={missing_qc}; extra_qc={extra_qc}"
        )

    detector_validation_rows: list[dict[str, Any]] = []
    detector_valid_by_recording: dict[tuple[str, str], bool] = {}
    recording_detector_fields = {
        "muscle_detector_valid",
        "muscle_score_finite_fraction",
        "muscle_detector_active_channel_count",
    }
    window_detector_fields = {
        "muscle_detector_valid",
        "muscle_score_finite_fraction",
    }
    if (
        not recording_detector_fields.issubset(qc_recording_rows[0])
        or not window_detector_fields.issubset(qc_rows[0])
    ):
        raise ValueError(
            "QC outputs do not contain the explicit v1.1 detector validity contract; "
            "rerun with the current QC policy"
        )
    for key, row in sorted(qc_recordings.items()):
        source_finite = float(row["finite_sample_fraction"]) == 1.0
        score_stats_finite = all(
            np.isfinite(float(row[name]))
            for name in ("muscle_score_p95", "muscle_score_max")
        )
        active_channels = int(row["muscle_detector_active_channel_count"])
        channel_count = int(row["channel_count_brainvision"])
        if active_channels < 0 or active_channels > channel_count:
            raise ValueError(
                f"Invalid active envelope channel count for {key}: "
                f"{active_channels} of {channel_count}"
            )
        explicit_valid = row["muscle_detector_valid"].lower() == "true"
        score_samples_finite = float(row["muscle_score_finite_fraction"]) == 1.0
        detector_valid = bool(
            source_finite
            and score_stats_finite
            and score_samples_finite
            and active_channels > 0
            and explicit_valid
        )
        detector_valid_by_recording[key] = detector_valid
        reasons = []
        if not source_finite:
            reasons.append("source_signal_nonfinite")
        if not score_stats_finite or not score_samples_finite:
            reasons.append("muscle_score_nonfinite_or_unassessed")
        if active_channels <= 0:
            reasons.append("no_active_envelope_channels")
        if not explicit_valid:
            reasons.append("detector_flagged_invalid_by_qc_runner")
        detector_validation_rows.append({
            "subject": key[0],
            "task": key[1],
            "source_signal_finite": source_finite,
            "muscle_score_summary_finite": score_stats_finite,
            "muscle_score_samples_finite": score_samples_finite,
            "active_envelope_channel_count": active_channels,
            "muscle_detector_valid": detector_valid,
            "validation_basis": "explicit recording-level runner validity plus finite source, finite score samples/summaries, and at least one active envelope channel",
            "invalidity_reasons": ";".join(reasons),
        })
    for qrow in qc_rows:
        key = (qrow["subject"], qrow["task"])
        valid = detector_valid_by_recording[key]
        valid = (
            valid
            and qrow["muscle_detector_valid"].lower() == "true"
            and float(qrow["muscle_score_finite_fraction"]) == 1.0
        )
        qrow["muscle_detector_valid"] = str(valid).lower()

    cohort_source_units: dict[str, dict[str, Any]] = {}
    for coordinate in RAW_UNITS:
        expected_units = EXPECTED_COORDINATE_SOURCE_UNITS[coordinate]
        observed_units = sorted({
            tuple(record["coordinate_source_units"][coordinate])
            for record in records
        })
        cohort_source_units[coordinate] = {
            "expected": expected_units,
            "observed": observed_units,
            "compatible": bool(observed_units) and all(
                units == expected_units for units in observed_units
            ),
        }

    pair_realization: dict[str, tuple[bool, list[str]]] = {}
    for subject in sorted(expected_subjects):
        pair_realization[subject] = pair_compatibility(
            by_key[(subject, "eyesclosed")],
            by_key[(subject, "eyesopen")],
            selection_by_subject[subject],
        )

    window_rows: list[dict[str, Any]] = []
    feature_window_rows: list[dict[str, Any]] = []
    participant_rows: list[dict[str, Any]] = []
    participant_feature_rows: list[dict[str, Any]] = []
    complete_support_rows: list[dict[str, Any]] = []
    for record in records:
        subject, task = record["subject"], record["task"]
        for index, (start, end) in enumerate(zip(record["starts"], record["ends"])):
            key = (subject, task, round(float(start), 6), round(float(end), 6))
            if key not in qc:
                raise ValueError(f"H5 window missing from QC table: {key}")
            q = qc[key]
            qc_eligible = eligible(q)
            for coordinate_index, coordinate in enumerate(record["names"]):
                raw_valid = bool(record["raw_support"][index, coordinate_index]) and np.isfinite(record["raw"][index, coordinate_index])
                anchored_valid = bool(record["anchored_valid"][index, coordinate_index]) and np.isfinite(record["anchored"][index, coordinate_index])
                window_rows.append({
                    "subject": subject,
                    "session": "ses-session1",
                    "task": task,
                    "window_index": index,
                    "window_start_sec": float(start),
                    "window_end_sec": float(end),
                    "coordinate": coordinate,
                    "qc_policy_eligible": qc_eligible,
                    "qc_exclusion_reason": exclusion_reason(q),
                    "source_finite_fraction": float(q["finite_fraction"]),
                    "muscle_detector_valid": q["muscle_detector_valid"],
                    "muscle_zscore_exceedance_fraction": float(q["muscle_zscore_exceedance_fraction"]),
                    "raw_coordinate": float(record["raw"][index, coordinate_index]) if raw_valid else math.nan,
                    "raw_supported": raw_valid,
                    "raw_unit": RAW_UNITS[coordinate],
                    "anchored_coordinate": float(record["anchored"][index, coordinate_index]) if anchored_valid else math.nan,
                    "anchored_valid": anchored_valid,
                    "anchored_unit": "dimensionless reference-relative score",
                    "complete_9d_valid": bool(record["complete_9d"][index]),
                })
            for feature_index, feature_name in enumerate(record["feature_names"]):
                feature_value = float(record["feature_values"][index, feature_index])
                feature_window_rows.append({
                    "subject": subject,
                    "session": "ses-session1",
                    "task": task,
                    "window_index": index,
                    "window_start_sec": float(start),
                    "window_end_sec": float(end),
                    "feature": feature_name,
                    "value": feature_value if np.isfinite(feature_value) else math.nan,
                    "supported_finite": bool(np.isfinite(feature_value)),
                    "unit": record["feature_units"].get(
                        feature_name, "not_recorded_in_measurement_provenance"
                    ),
                    "qc_policy_eligible": qc_eligible,
                    "qc_exclusion_reason": exclusion_reason(q),
                    "source_finite_fraction": float(q["finite_fraction"]),
                    "muscle_detector_valid": q["muscle_detector_valid"],
                    "muscle_zscore_exceedance_fraction": float(
                        q["muscle_zscore_exceedance_fraction"]
                    ),
                })

        for layer in ("raw", "anchored"):
            values = record[layer]
            valid_by_coord = record["raw_support"] if layer == "raw" else record["anchored_valid"]
            for coordinate_index, coordinate in enumerate(record["names"]):
                eligible_indices = []
                intervals = []
                supported_indices = []
                for index, (start, end) in enumerate(zip(record["starts"], record["ends"])):
                    qkey = (subject, task, round(float(start), 6), round(float(end), 6))
                    q = qc[qkey]
                    is_supported = bool(valid_by_coord[index, coordinate_index]) and np.isfinite(values[index, coordinate_index])
                    if is_supported:
                        supported_indices.append(index)
                    if eligible(q) and is_supported:
                        eligible_indices.append(index)
                        intervals.append((float(start), float(end)))
                value_list = [float(values[index, coordinate_index]) for index in eligible_indices]
                n_windows = len(value_list)
                unique_seconds = unique_interval_seconds(intervals)
                participant_rows.append({
                    "subject": subject,
                    "coordinate_layer": "pre_anchor" if layer == "raw" else "frozen_ec_anchored",
                    "coordinate": coordinate,
                    "unit": RAW_UNITS[coordinate] if layer == "raw" else "dimensionless reference-relative score",
                    "task": task,
                    "condition": TASK_DISPLAY[task],
                    "supported_finite_windows_before_external_qc": len(supported_indices),
                    "qc_eligible_supported_windows": n_windows,
                    "qc_excluded_supported_windows": len(supported_indices) - n_windows,
                    "qc_exclusion_rule": "source_finite_and_detector_valid_and_no_gt4_sample_overlap",
                    "qc_eligible_unique_seconds": unique_seconds,
                    "minimum_support_met": support_pass(n_windows, unique_seconds),
                    "median": float(np.median(value_list)) if value_list else math.nan,
                    "mean": float(np.mean(value_list)) if value_list else math.nan,
                    "q1": float(np.quantile(value_list, 0.25)) if value_list else math.nan,
                    "q3": float(np.quantile(value_list, 0.75)) if value_list else math.nan,
                })

        for feature_index, feature_name in enumerate(record["feature_names"]):
            eligible_indices = []
            intervals = []
            finite_indices = []
            for index, (start, end) in enumerate(zip(record["starts"], record["ends"])):
                qkey = (subject, task, round(float(start), 6), round(float(end), 6))
                value = record["feature_values"][index, feature_index]
                is_finite = bool(np.isfinite(value))
                if is_finite:
                    finite_indices.append(index)
                if eligible(qc[qkey]) and is_finite:
                    eligible_indices.append(index)
                    intervals.append((float(start), float(end)))
            value_list = [
                float(record["feature_values"][index, feature_index])
                for index in eligible_indices
            ]
            n_windows = len(value_list)
            unique_seconds = unique_interval_seconds(intervals)
            participant_feature_rows.append({
                "subject": subject,
                "feature": feature_name,
                "feature_present": True,
                "unit": record["feature_units"].get(
                    feature_name, "not_recorded_in_measurement_provenance"
                ),
                "task": task,
                "condition": TASK_DISPLAY[task],
                "finite_windows_before_external_qc": len(finite_indices),
                "qc_eligible_finite_windows": n_windows,
                "qc_excluded_finite_windows": len(finite_indices) - n_windows,
                "qc_exclusion_rule": "source_finite_and_detector_valid_and_no_gt4_sample_overlap",
                "qc_eligible_unique_seconds": unique_seconds,
                "minimum_support_met": support_pass(n_windows, unique_seconds),
                "median": float(np.median(value_list)) if value_list else math.nan,
                "mean": float(np.mean(value_list)) if value_list else math.nan,
                "q1": float(np.quantile(value_list, 0.25)) if value_list else math.nan,
                "q3": float(np.quantile(value_list, 0.75)) if value_list else math.nan,
            })

        complete_indices = []
        complete_intervals = []
        complete_supported_indices = []
        for index, (start, end) in enumerate(zip(record["starts"], record["ends"])):
            qkey = (subject, task, round(float(start), 6), round(float(end), 6))
            if bool(record["complete_9d"][index]):
                complete_supported_indices.append(index)
            if eligible(qc[qkey]) and bool(record["complete_9d"][index]):
                complete_indices.append(index)
                complete_intervals.append((float(start), float(end)))
        complete_support_rows.append({
            "subject": subject,
            "task": task,
            "condition": TASK_DISPLAY[task],
            "complete_9d_windows_before_external_qc": len(complete_supported_indices),
            "qc_eligible_complete_9d_windows": len(complete_indices),
            "qc_excluded_complete_9d_windows": len(complete_supported_indices) - len(complete_indices),
            "qc_exclusion_rule": "source_finite_and_detector_valid_and_no_gt4_sample_overlap",
            "qc_eligible_unique_seconds": unique_interval_seconds(complete_intervals),
            "minimum_support_met": support_pass(len(complete_indices), unique_interval_seconds(complete_intervals)),
        })

    participant_index = {
        (row["subject"], row["coordinate_layer"], row["coordinate"], row["task"]): row
        for row in participant_rows
    }
    paired_rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    for layer in ("raw", "anchored"):
        for coordinate in RAW_UNITS:
            deltas: list[float] = []
            expected_source_units = EXPECTED_COORDINATE_SOURCE_UNITS[coordinate]
            cohort_units_ok = cohort_source_units[coordinate]["compatible"]
            for subject in sorted(expected_subjects):
                realization_ok, realization_reasons = pair_realization[subject]
                ec_record = by_key[(subject, "eyesclosed")]
                eo_record = by_key[(subject, "eyesopen")]
                ec_source_units = ec_record["coordinate_source_units"][coordinate]
                eo_source_units = eo_record["coordinate_source_units"][coordinate]
                pair_units_match = (
                    ec_source_units == eo_source_units
                    and all(unit != "not_recorded_in_measurement_provenance" for unit in ec_source_units)
                    and ec_source_units == expected_source_units
                )
                pair_extra_reasons = []
                if ec_source_units != eo_source_units:
                    pair_extra_reasons.append("coordinate_source_feature_units_mismatch")
                elif any(unit == "not_recorded_in_measurement_provenance" for unit in ec_source_units):
                    pair_extra_reasons.append("coordinate_source_feature_units_not_recorded")
                elif ec_source_units != expected_source_units:
                    pair_extra_reasons.append("coordinate_source_feature_units_unexpected")
                ec = participant_index[(subject, "pre_anchor" if layer == "raw" else "frozen_ec_anchored", coordinate, "eyesclosed")]
                eo = participant_index[(subject, "pre_anchor" if layer == "raw" else "frozen_ec_anchored", coordinate, "eyesopen")]
                pair_valid = bool(
                    ec["minimum_support_met"]
                    and eo["minimum_support_met"]
                    and realization_ok
                    and pair_units_match
                )
                delta = float(eo["median"] - ec["median"]) if pair_valid else math.nan
                if pair_valid and cohort_units_ok:
                    deltas.append(delta)
                paired_rows.append({
                    "subject": subject,
                    "coordinate_layer": "pre_anchor" if layer == "raw" else "frozen_ec_anchored",
                    "coordinate": coordinate,
                    "unit": RAW_UNITS[coordinate] if layer == "raw" else "dimensionless reference-relative score",
                    "ec_median": ec["median"],
                    "ec_n_windows": ec["qc_eligible_supported_windows"],
                    "ec_unique_seconds": ec["qc_eligible_unique_seconds"],
                    "ec_minimum_support_met": ec["minimum_support_met"],
                    "eo_median": eo["median"],
                    "eo_n_windows": eo["qc_eligible_supported_windows"],
                    "eo_unique_seconds": eo["qc_eligible_unique_seconds"],
                    "eo_minimum_support_met": eo["minimum_support_met"],
                    "realization_compatible": realization_ok,
                    "realization_mismatch_reasons": ";".join(realization_reasons),
                    "coordinate_source_feature_units_ec": json.dumps(ec_source_units),
                    "coordinate_source_feature_units_eo": json.dumps(eo_source_units),
                    "coordinate_source_feature_units_expected": json.dumps(expected_source_units),
                    "coordinate_source_feature_units_compatible": pair_units_match,
                    "cohort_source_units_compatible": cohort_units_ok,
                    "paired_delta_eo_minus_ec": delta,
                    "pair_included": pair_valid,
                    "cohort_aggregation_included": pair_valid and cohort_units_ok,
                    "cohort_aggregation_exclusion_reason": (
                        "cohort_source_units_inconsistent" if pair_valid and not cohort_units_ok else ""
                    ),
                    "pair_exclusion_reason": pair_exclusion_reason(
                        ec, eo, realization_ok, realization_reasons, pair_extra_reasons
                    ),
                })
            cohort_rows.append({
                "coordinate_layer": "pre_anchor" if layer == "raw" else "frozen_ec_anchored",
                "coordinate": coordinate,
                "unit": RAW_UNITS[coordinate] if layer == "raw" else "dimensionless reference-relative score",
                "status": (
                    "available"
                    if deltas
                    else (
                        "unavailable_inconsistent_source_units"
                        if not cohort_units_ok
                        else "unavailable_no_eligible_paired_participants"
                    )
                ),
                "source_units_expected": json.dumps(expected_source_units),
                "source_unit_signatures_observed": json.dumps(
                    cohort_source_units[coordinate]["observed"]
                ),
                "source_units_cohort_compatible": cohort_units_ok,
                **stats(deltas),
                "pairing_policy": "participant median EO-EC; both conditions meet >=30 eligible windows and >=120 unique seconds; aggregate only when all source units match the v3.4 contract",
            })

    feature_participant_index = {
        (row["subject"], row["feature"], row["task"]): row
        for row in participant_feature_rows
    }
    feature_names: list[str] = []
    for record in records:
        for feature_name in record["feature_names"]:
            if feature_name not in feature_names:
                feature_names.append(feature_name)
    for subject in sorted(expected_subjects):
        for task in TASKS:
            record = by_key[(subject, task)]
            present = set(record["feature_names"])
            for feature_name in feature_names:
                key = (subject, feature_name, task)
                if key not in feature_participant_index:
                    unit = record["feature_units"].get(
                        feature_name, "not_recorded_in_measurement_provenance"
                    )
                    missing_row = {
                        "subject": subject,
                        "feature": feature_name,
                        "feature_present": feature_name in present,
                        "unit": unit,
                        "task": task,
                        "condition": TASK_DISPLAY[task],
                        "finite_windows_before_external_qc": 0,
                        "qc_eligible_finite_windows": 0,
                        "qc_excluded_finite_windows": 0,
                        "qc_exclusion_rule": "source_finite_and_detector_valid_and_no_gt4_sample_overlap",
                        "qc_eligible_unique_seconds": 0.0,
                        "minimum_support_met": False,
                        "median": math.nan,
                        "mean": math.nan,
                        "q1": math.nan,
                        "q3": math.nan,
                    }
                    participant_feature_rows.append(missing_row)
                    feature_participant_index[key] = missing_row
    paired_feature_rows: list[dict[str, Any]] = []
    cohort_feature_rows: list[dict[str, Any]] = []
    for feature_name in feature_names:
        deltas_by_unit: dict[str, list[float]] = {}
        for subject in sorted(expected_subjects):
            realization_ok, realization_reasons = pair_realization[subject]
            ec = feature_participant_index[(subject, feature_name, "eyesclosed")]
            eo = feature_participant_index[(subject, feature_name, "eyesopen")]
            unit_matches = ec["unit"] == eo["unit"]
            pair_extra_reasons = []
            if not ec.get("feature_present", False):
                pair_extra_reasons.append("EC_feature_missing_from_H5")
            if not eo.get("feature_present", False):
                pair_extra_reasons.append("EO_feature_missing_from_H5")
            if not unit_matches:
                pair_extra_reasons.append("feature_unit_mismatch")
            if ec["unit"] == "not_recorded_in_measurement_provenance":
                pair_extra_reasons.append("EC_feature_unit_not_recorded")
            if eo["unit"] == "not_recorded_in_measurement_provenance":
                pair_extra_reasons.append("EO_feature_unit_not_recorded")
            pair_valid = bool(
                ec["minimum_support_met"]
                and eo["minimum_support_met"]
                and realization_ok
                and unit_matches
                and ec["unit"] != "not_recorded_in_measurement_provenance"
                and ec.get("feature_present", False)
                and eo.get("feature_present", False)
            )
            delta = float(eo["median"] - ec["median"]) if pair_valid else math.nan
            if pair_valid:
                deltas_by_unit.setdefault(ec["unit"], []).append(delta)
            paired_feature_rows.append({
                "subject": subject,
                "feature": feature_name,
                "unit": ec["unit"] if unit_matches else "unit_mismatch",
                "ec_feature_present": ec.get("feature_present", False),
                "ec_median": ec["median"],
                "ec_n_windows": ec["qc_eligible_finite_windows"],
                "ec_unique_seconds": ec["qc_eligible_unique_seconds"],
                "ec_minimum_support_met": ec["minimum_support_met"],
                "eo_median": eo["median"],
                "eo_feature_present": eo.get("feature_present", False),
                "eo_n_windows": eo["qc_eligible_finite_windows"],
                "eo_unique_seconds": eo["qc_eligible_unique_seconds"],
                "eo_minimum_support_met": eo["minimum_support_met"],
                "realization_compatible": realization_ok,
                "realization_mismatch_reasons": ";".join(realization_reasons),
                "paired_delta_eo_minus_ec": delta,
                "pair_included": pair_valid,
                "pair_exclusion_reason": pair_exclusion_reason(
                    ec, eo, realization_ok, realization_reasons, pair_extra_reasons
                ),
            })
        units_for_feature = {
            row["unit"]
            for row in participant_feature_rows
            if row["feature"] == feature_name and row.get("feature_present", False)
        }
        for unit in sorted(units_for_feature):
            cohort_feature_rows.append({
                "feature": feature_name,
                "unit": unit,
                "status": (
                    "available"
                    if deltas_by_unit.get(unit, [])
                    else "unavailable_no_eligible_paired_participants"
                ),
                **stats(deltas_by_unit.get(unit, [])),
                "pairing_policy": "participant median EO-EC; both conditions meet >=30 eligible windows and >=120 unique seconds",
            })

    # Paired complete-9D support is a vector-completeness denominator, not a
    # new metric and not a substitute for coordinate-specific support.
    complete_index = {(row["subject"], row["task"]): row for row in complete_support_rows}
    complete_pairs = []
    for subject in sorted(expected_subjects):
        ec = complete_index[(subject, "eyesclosed")]
        eo = complete_index[(subject, "eyesopen")]
        realization_ok, realization_reasons = pair_realization[subject]
        ec_h5 = by_key[(subject, "eyesclosed")]
        eo_h5 = by_key[(subject, "eyesopen")]
        incompatible_coordinate_units = [
            coordinate
            for coordinate in RAW_UNITS
            if (
                ec_h5["coordinate_source_units"][coordinate]
                != eo_h5["coordinate_source_units"][coordinate]
                or ec_h5["coordinate_source_units"][coordinate]
                != EXPECTED_COORDINATE_SOURCE_UNITS[coordinate]
                or any(
                    unit == "not_recorded_in_measurement_provenance"
                    for unit in ec_h5["coordinate_source_units"][coordinate]
                )
            )
        ]
        all_coordinate_units_compatible = not incompatible_coordinate_units
        complete_pair_reason = pair_exclusion_reason(
            ec,
            eo,
            realization_ok,
            realization_reasons,
            [f"complete_9d_source_feature_units_incompatible:{','.join(incompatible_coordinate_units)}"]
            if incompatible_coordinate_units
            else [],
        )
        complete_pairs.append({
            "subject": subject,
            "ec_qc_eligible_complete_9d_windows": ec["qc_eligible_complete_9d_windows"],
            "ec_unique_seconds": ec["qc_eligible_unique_seconds"],
            "ec_minimum_support_met": ec["minimum_support_met"],
            "eo_qc_eligible_complete_9d_windows": eo["qc_eligible_complete_9d_windows"],
            "eo_unique_seconds": eo["qc_eligible_unique_seconds"],
            "eo_minimum_support_met": eo["minimum_support_met"],
            "realization_compatible": realization_ok,
            "realization_mismatch_reasons": ";".join(realization_reasons),
            "all_nine_coordinate_source_units_compatible": all_coordinate_units_compatible,
            "incompatible_coordinate_source_units": ";".join(incompatible_coordinate_units),
            "both_conditions_minimum_support_met": bool(
                ec["minimum_support_met"]
                and eo["minimum_support_met"]
                and realization_ok
                and all_coordinate_units_compatible
            ),
            "pair_exclusion_reason": complete_pair_reason,
        })

    has_anchored_pairs = any(
        row["coordinate_layer"] == "frozen_ec_anchored" and row["paired_n"] > 0
        for row in cohort_rows
    )

    write_csv(ROOT / "window_level_measurements_with_frozen_qc.csv", window_rows)
    write_csv(ROOT / "window_level_raw_features_with_frozen_qc.csv", feature_window_rows)
    write_csv(ROOT / "participant_condition_summaries.csv", participant_rows)
    write_csv(ROOT / "participant_raw_feature_summaries.csv", participant_feature_rows)
    write_csv(ROOT / "participant_paired_differences.csv", paired_rows)
    write_csv(ROOT / "participant_paired_raw_feature_differences.csv", paired_feature_rows)
    write_csv(ROOT / "participant_complete_9d_support.csv", complete_support_rows)
    write_csv(ROOT / "participant_complete_9d_pairs.csv", complete_pairs)
    write_csv(ROOT / "qc_detector_validation.csv", detector_validation_rows)

    summary = {
        "status": "complete" if has_anchored_pairs else "stopped_no_eligible_anchored_pairs",
        "policy_version": "DS004148_BASELINE_EC_QC_POLICY_V1_1",
        "anchor_id": ANCHOR_ID,
        "anchor_manifest_sha256": ANCHOR_SHA256,
        "subjects": len(expected_subjects),
        "h5_recordings": len(records),
        "qc_windows": len(qc_rows),
        "coordinate_names": list(RAW_UNITS),
        "raw_features": feature_names,
        "min_eligible_windows_per_condition_coordinate": MIN_WINDOWS,
        "min_unique_seconds_per_condition_coordinate": MIN_UNIQUE_SECONDS,
        "qc_eligible_definition": "window has fully finite source samples, a valid artifact-only muscle detector, and zero samples with score > 4",
        "qc_detector_validity": "requires finite source samples, finite muscle score summaries (and per-sample score support when exported), at least one non-flat channel, and a valid detector flag when exported; invalid detector windows are unassessed and ineligible",
        "qc_detector_validity_basis": "explicit runner validity and per-sample finite-score fields; legacy QC output is rejected",
        "coordinate_source_unit_contract": {
            coordinate: {
                "expected": value["expected"],
                "observed": value["observed"],
                "cohort_compatible": value["compatible"],
            }
            for coordinate, value in cohort_source_units.items()
        },
        "paired_summary_unit": "participant-level condition medians; EO minus EC; no window pooling or p-values",
        "cohort_by_coordinate": [
            {"layer": row["coordinate_layer"], "coordinate": row["coordinate"], "paired_n": row["paired_n"]}
            for row in cohort_rows
        ],
        "complete_9d_pairs_meeting_support": sum(row["both_conditions_minimum_support_met"] for row in complete_pairs),
        "subjects_with_realization_mismatch": sum(
            not pair_realization[subject][0] for subject in expected_subjects
        ),
        "cohort_summaries_written": has_anchored_pairs,
    }
    (ROOT / "baseline_summary_manifest.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if not has_anchored_pairs:
        print(json.dumps(summary, indent=2))
        raise SystemExit(
            "Policy stop: no anchored coordinate has even one eligible paired participant; "
            "participant-level condition/support/exclusion outputs were retained."
        )

    write_csv(ROOT / "cohort_paired_summary.csv", cohort_rows)
    write_csv(ROOT / "cohort_paired_raw_feature_summary.csv", cohort_feature_rows)
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
