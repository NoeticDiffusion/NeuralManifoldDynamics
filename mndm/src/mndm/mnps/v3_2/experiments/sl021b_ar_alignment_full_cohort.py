"""SL-021B full-cohort alignment characterization for ds004148.

Processes only the 48 participants outside the frozen SL-021 pilot, then
reports those participants separately and combines their session support with
the immutable pilot support. The AR kernel and SL-021 aggregation are reused
unchanged; all outputs remain exploratory and unqualified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import scipy
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[6]
SRC_ROOT = REPO_ROOT / "mndm" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mndm.mnps.v3_2 import ar as ar_kernel
from mndm.mnps.v3_2.experiments import sl020_ds004148 as sl020
from mndm.mnps.v3_2.experiments import sl021_ar_alignment_specificity as sl021


PROTOCOL_ID = "SL-021B-AR-ALIGNMENT-FULL-COHORT-001"
REMAINING_IDS = (
    "sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-08",
    "sub-09", "sub-10", "sub-11", "sub-12", "sub-14", "sub-15", "sub-17",
    "sub-18", "sub-20", "sub-22", "sub-24", "sub-25", "sub-26", "sub-27",
    "sub-28", "sub-30", "sub-31", "sub-33", "sub-34", "sub-35", "sub-36",
    "sub-37", "sub-38", "sub-40", "sub-41", "sub-43", "sub-44", "sub-45",
    "sub-46", "sub-47", "sub-48", "sub-49", "sub-50", "sub-51", "sub-52",
    "sub-53", "sub-54", "sub-56", "sub-57", "sub-59", "sub-60",
)
SESSIONS = sl021.SESSIONS
CONDITIONS = sl021.CONDITIONS
OFFSETS_SECONDS = sl021.OFFSETS_SECONDS
COMPONENTS = tuple(f"block_{i:03d}_to_{i + 1:03d}" for i in range(9))
TARGET_COMPONENTS = sl021.TARGET_COMPONENTS
WINDOWS_PER_OFFSET = sl021.WINDOWS_PER_OFFSET
WINDOW_SAMPLES = sl021.WINDOW_SAMPLES
FS_HZ = sl021.FS_HZ
BOOTSTRAPS = sl021.BOOTSTRAPS
SEED = sl021.SEED


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                if isinstance(value, (list, tuple, dict)) else value
                for key, value in row.items()
            })


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda index: (pvalues[index], index))
    adjusted = [1.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _bootstrap_median(values: np.ndarray, rng: np.random.Generator) -> tuple[float | None, float | None]:
    if values.size == 0:
        return None, None
    sample = values[rng.integers(0, values.size, size=(BOOTSTRAPS, values.size))]
    median = np.median(sample, axis=1)
    return float(np.quantile(median, 0.025)), float(np.quantile(median, 0.975))


def _session_medians(support_rows: list[dict[str, str]]) -> dict[tuple[str, str, str, int, str], float]:
    output: dict[tuple[str, str, str, int, str], float] = {}
    for row in support_rows:
        emitted = row.get("session_median_emitted", False)
        if not (emitted is True or str(emitted).casefold() == "true"):
            continue
        key = (row["participant_id"], row["session_id"], row["condition"],
               int(row["offset_seconds"]), row["component_id"])
        value = float(row["session_median"])
        if not math.isfinite(value):
            raise RuntimeError(f"nonfinite_emitted_session_median:{key}")
        output[key] = value
    return output


def _participant_contrasts(
    support_rows: list[dict[str, str]], participants: tuple[str, ...], label: str,
) -> list[dict[str, Any]]:
    medians = _session_medians(support_rows)
    rows: list[dict[str, Any]] = []
    for offset in OFFSETS_SECONDS:
        for component in COMPONENTS:
            for participant in participants:
                session_deltas = []
                for session in SESSIONS:
                    ec = medians.get((participant, session, "eyesclosed", offset, component))
                    music = medians.get((participant, session, "music", offset, component))
                    if ec is not None and music is not None:
                        session_deltas.append(music - ec)
                delta = float(np.mean(session_deltas)) if len(session_deltas) >= 2 else None
                rows.append({
                    "participant_set": label, "participant_id": participant,
                    "offset_seconds": offset, "component_id": component,
                    "paired_session_count": len(session_deltas),
                    "participant_music_minus_eyesclosed": delta,
                    "eligible_two_or_more_sessions": delta is not None,
                    "qualified": False,
                })
    return rows


def _summaries_and_tests(
    contrast_rows: list[dict[str, Any]], participants: tuple[str, ...], label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    group_values: dict[tuple[int, str], dict[str, float]] = {}
    for offset in OFFSETS_SECONDS:
        for component in COMPONENTS:
            group_values[(offset, component)] = {
                row["participant_id"]: float(row["participant_music_minus_eyesclosed"])
                for row in contrast_rows
                if row["offset_seconds"] == offset and row["component_id"] == component
                and row["participant_music_minus_eyesclosed"] is not None
            }
    rng = np.random.default_rng(SEED)
    summaries: list[dict[str, Any]] = []
    for component in COMPONENTS:
        offset_medians: list[float] = []
        positive_count = 0
        for offset in OFFSETS_SECONDS:
            group = group_values[(offset, component)]
            values = np.asarray([group[pid] for pid in participants if pid in group], dtype=float)
            low, high = _bootstrap_median(values, rng)
            median = float(np.median(values)) if values.size else None
            if median is not None:
                offset_medians.append(median)
                positive_count += int(median > 0.0)
            summaries.append({
                "participant_set": label, "offset_seconds": offset,
                "component_id": component, "participant_count": int(values.size),
                "participant_ids": [pid for pid in participants if pid in group],
                "median_participant_music_minus_eyesclosed": median,
                "bootstrap_ci95_low": low, "bootstrap_ci95_high": high,
                "qualified": False,
            })
        span = max(offset_medians) - min(offset_medians) if len(offset_medians) == 4 else None
        for row in summaries[-4:]:
            row["four_offset_median_range"] = span if component in TARGET_COMPONENTS else None
            row["positive_offset_median_count"] = positive_count if component in TARGET_COMPONENTS else None

    tests: list[dict[str, Any]] = []
    for offset in OFFSETS_SECONDS:
        for component in TARGET_COMPONENTS:
            group = group_values[(offset, component)]
            values = np.asarray([group[pid] for pid in participants if pid in group], dtype=float)
            statistic, pvalue, reason = None, 1.0, None
            if values.size < 2:
                reason = "fewer_than_2_eligible_participants"
            elif np.all(values == 0.0):
                statistic, pvalue = 0.0, 1.0
            else:
                try:
                    result = stats.wilcoxon(values, zero_method="wilcox", alternative="two-sided", method="auto")
                    statistic, pvalue = float(result.statistic), float(result.pvalue)
                except Exception as exc:
                    reason = f"wilcoxon_not_estimable:{type(exc).__name__}"
            tests.append({
                "participant_set": label, "offset_seconds": offset,
                "component_id": component, "participant_count": int(values.size),
                "wilcoxon_statistic": statistic, "p_raw_two_sided": pvalue,
                "not_estimable_reason": reason, "qualified": False,
            })
    if len(tests) != 8:
        raise RuntimeError(f"fixed_target_family_size_mismatch:{label}:{len(tests)}")
    for row, adjusted in zip(tests, holm([float(item["p_raw_two_sided"]) for item in tests]), strict=True):
        row["p_holm_8"] = adjusted
    return summaries, tests


def _verify_pilot(pilot_root: Path, pilot_manifest: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, str]]:
    expected_files = pilot_manifest.get("output_sha256", {})
    pilot_hashes: dict[str, str] = {}
    for relative, expected in expected_files.items():
        path = pilot_root / relative
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"SL021_pilot_output_hash_mismatch:{relative}")
        pilot_hashes[relative] = actual
    support = read_csv(pilot_root / "session_support.csv")
    if len(support) != 12 * 3 * 2 * 4 * 9:
        raise RuntimeError(f"SL021_pilot_support_rows_mismatch:{len(support)}")
    if any(row.get("computable_window_count") != "14" for row in support):
        raise RuntimeError("SL021_pilot_support_no_longer_complete")
    roster = set(row["participant_id"] for row in support)
    if roster != set(sl021.PILOT_IDS):
        raise RuntimeError("SL021_pilot_support_roster_mismatch")
    return support, pilot_hashes


def _compare_contrasts(recomputed: list[dict[str, Any]], pilot_root: Path) -> dict[str, Any]:
    previous_rows = read_csv(pilot_root / "participant_contrasts.csv")
    prior = {
        (row["participant_id"], int(row["offset_seconds"]), row["component_id"]): row
        for row in previous_rows
    }
    if len(recomputed) != 12 * 4 * 9 or len(prior) != len(recomputed):
        raise RuntimeError("SL021_pilot_contrast_reproduction_row_count_mismatch")
    maximum_difference = 0.0
    for row in recomputed:
        key = (row["participant_id"], row["offset_seconds"], row["component_id"])
        old = prior.get(key)
        if old is None or int(old["paired_session_count"]) != row["paired_session_count"]:
            raise RuntimeError(f"SL021_pilot_contrast_reproduction_key_mismatch:{key}")
        previous_delta = old["participant_music_minus_eyesclosed"]
        current_delta = row["participant_music_minus_eyesclosed"]
        if previous_delta == "" and current_delta is None:
            continue
        difference = abs(float(previous_delta) - float(current_delta))
        maximum_difference = max(maximum_difference, difference)
        if difference > 1e-15:
            raise RuntimeError(f"SL021_pilot_contrast_reproduction_value_mismatch:{key}:{difference}")
    return {"rows_compared": len(recomputed), "maximum_absolute_difference": maximum_difference,
            "absolute_tolerance": 1e-15, "status": "pass"}


def run(
    *, dataset_root: Path, r1_manifest_path: Path, cohort_manifest_path: Path,
    pilot_root: Path, protocol_path: Path, output_root: Path,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve(strict=True)
    r1_manifest_path = r1_manifest_path.resolve(strict=True)
    cohort_manifest_path = cohort_manifest_path.resolve(strict=True)
    pilot_root = pilot_root.resolve(strict=True)
    protocol_path = protocol_path.resolve(strict=True)
    output_root = output_root.resolve()
    if output_root.drive.upper() != "J:":
        raise ValueError("SL021B_output_root_must_be_on_J_drive")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing_nonempty_SL021B_output_root:{output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    protocol_sidecar = Path(str(protocol_path) + ".sha256")
    if not protocol_sidecar.is_file():
        raise FileNotFoundError(f"frozen_SL021B_protocol_hash_missing:{protocol_sidecar}")
    protocol_hash = sha256_file(protocol_path)
    if protocol_sidecar.read_text(encoding="ascii").split()[0].lower() != protocol_hash:
        raise RuntimeError("frozen_SL021B_protocol_hash_mismatch")
    r1 = read_json(r1_manifest_path)
    cohort = read_json(cohort_manifest_path)
    pilot_manifest_path = pilot_root / "run_manifest.json"
    pilot_manifest = read_json(pilot_manifest_path)
    pilot_support, pilot_output_hashes = _verify_pilot(pilot_root, pilot_manifest)
    if r1.get("audit_id") != sl020.AUDIT_ID or r1.get("source_root") != str(dataset_root):
        raise RuntimeError("R1_audit_identity_mismatch")
    if cohort.get("status") != "complete" or cohort.get("protocol_id") != sl020.PROTOCOL_ID:
        raise RuntimeError("SL020_cohort_manifest_incomplete_or_wrong_protocol")
    if cohort.get("frozen_identity", {}).get("mode") != "cohort":
        raise RuntimeError("SL020_cohort_manifest_mode_mismatch")
    if pilot_manifest.get("protocol_id") != sl021.PROTOCOL_ID:
        raise RuntimeError("SL021_pilot_protocol_mismatch")
    if tuple(cohort.get("frozen_identity", {}).get("selected_participants", ())) != tuple(f"sub-{i:02d}" for i in range(1, 61)):
        raise RuntimeError("SL020_full_cohort_roster_mismatch")
    current_sl020_hash = sha256_file(Path(sl020.__file__).resolve())
    current_sl021_hash = sha256_file(Path(sl021.__file__).resolve())
    current_ar_hash = sha256_file(Path(ar_kernel.__file__).resolve())
    frozen_code_hashes = cohort.get("frozen_identity", {}).get("code_hashes", {})
    if frozen_code_hashes.get("runner") != current_sl020_hash or frozen_code_hashes.get("ar") != current_ar_hash:
        raise RuntimeError("current_SL020_runner_or_AR_kernel_differs_from_frozen_cohort")
    if pilot_manifest.get("input_hashes", {}).get("sl020_runner_sha256") != current_sl020_hash:
        raise RuntimeError("current_SL020_runner_differs_from_SL021_pilot")
    if pilot_manifest.get("input_hashes", {}).get("ar_kernel_sha256") != current_ar_hash:
        raise RuntimeError("current_AR_kernel_differs_from_SL021_pilot")
    if pilot_manifest.get("input_hashes", {}).get("analysis_script_sha256") != current_sl021_hash:
        raise RuntimeError("current_SL021_runner_differs_from_completed_pilot")
    full_roster = set(cohort.get("frozen_identity", {}).get("selected_participants", ()))
    if (set(REMAINING_IDS) & set(sl021.PILOT_IDS) or len(REMAINING_IDS) != 48
            or set(REMAINING_IDS) | set(sl021.PILOT_IDS) != full_roster):
        raise RuntimeError("SL021B_remaining_roster_invalid")

    audit_records = r1.get("recordings", [])
    r1_by_key = {(row["participant_id"], row["session_id"], row["task_id"]): row for row in audit_records}
    if len(r1_by_key) != 900:
        raise RuntimeError(f"R1_recording_count_mismatch:{len(r1_by_key)}")
    cohort_sources: dict[tuple[str, str, str], str] = {}
    for participant in REMAINING_IDS:
        completed = cohort.get("completed_participants", {}).get(participant)
        if not completed:
            raise RuntimeError(f"SL020_cohort_missing_participant:{participant}")
        for source in completed.get("source_recordings", []):
            cohort_sources[(participant, source["session_id"], source["task_id"])] = source["payload_sha256"]

    started = datetime.now(timezone.utc).isoformat()
    window_rows: list[dict[str, Any]] = []
    remaining_support: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    planned_recordings = 0
    supported_recordings = 0
    for participant in REMAINING_IDS:
        print(f"SL021B processing {participant}", flush=True)
        for session in SESSIONS:
            for condition in CONDITIONS:
                key = (participant, session, condition)
                planned_recordings += 1
                audit_record = r1_by_key.get(key)
                if audit_record is None or key not in cohort_sources:
                    raise RuntimeError(f"frozen_source_record_missing:{key}")
                paths, header, metadata_hashes, labels, reorder = sl020._verify_source_metadata(dataset_root, audit_record)
                source = sl020._payload_source_record(dataset_root, audit_record, paths, header, labels, reorder)
                if source["payload_sha256"] != cohort_sources[key]:
                    raise RuntimeError(f"raw_source_hash_differs_from_SL020:{key}")
                source_reason = None
                if not source["payload_is_whole_frame"]:
                    source_reason = "payload_size_not_whole_sample_frame"
                elif not source["payload_matches_declared_support"]:
                    source_reason = "payload_size_differs_from_declared_duration"
                source_records.append({
                    "participant_id": participant, "session_id": session, "condition": condition,
                    "relative_payload_path": source["relative_payload_path"],
                    "payload_sha256": source["payload_sha256"],
                    "payload_size_bytes": source["payload_size_bytes"],
                    "payload_is_whole_frame": source["payload_is_whole_frame"],
                    "payload_matches_declared_support": source["payload_matches_declared_support"],
                    "metadata_hashes": metadata_hashes,
                    "channel_count": len(labels),
                    "channel_order_sha256": source["study_common_order_hash"],
                    "channel_set_sha256": source["study_common_set_hash"],
                    "declared_reference": source["declared_reference"],
                    "reference_set_hash": source["reference_set_hash"],
                    "metadata_qc": source["metadata_qc"], "unsupported_reason": source_reason,
                })
                local_values: dict[tuple[int, str], list[float | None]] = {
                    (offset, component): [] for offset in OFFSETS_SECONDS for component in COMPONENTS
                }
                if source_reason is None:
                    supported_recordings += 1
                    payload = np.memmap(
                        paths["eeg"], dtype="<f4", mode="r",
                        shape=(int(source["declared_samples"]), int(header["channel_count"])), order="C",
                    )
                    channel_indices = np.asarray([int(index) for index in reorder], dtype=np.int64)
                    for offset in OFFSETS_SECONDS:
                        for window_index in range(WINDOWS_PER_OFFSET):
                            start = offset * FS_HZ + window_index * WINDOW_SAMPLES
                            end = start + WINDOW_SAMPLES
                            if end > int(source["declared_samples"]):
                                raise RuntimeError(f"partial_window_forbidden:{key}:{offset}:{window_index}")
                            signals = np.asarray(payload[start:end, :][:, channel_indices].T, dtype=np.float64)
                            result = sl021._ar_for_window(signals, labels, start)
                            if tuple(result.component_names) != COMPONENTS:
                                raise RuntimeError("unexpected_AR_component_schema")
                            values = list(result.component_values)
                            statuses = list(result.component_status)
                            reasons = list(result.component_reasons)
                            for component, value in zip(COMPONENTS, values, strict=True):
                                local_values[(offset, component)].append(
                                    float(value) if value is not None and math.isfinite(float(value)) else None
                                )
                            window_rows.append({
                                "participant_id": participant, "session_id": session,
                                "condition": condition, "offset_seconds": offset,
                                "window_index": window_index,
                                "window_id": f"sl021b-20s-o{offset:02d}-{start:06d}-{end:06d}",
                                "start_sample": start, "end_sample_exclusive": end,
                                "start_seconds": start / FS_HZ, "end_seconds": end / FS_HZ,
                                "source_payload_sha256": source["payload_sha256"],
                                "realization_id": sl020.SENSOR_REALIZATION_ID,
                                "channel_labels": labels, "channel_count": len(labels),
                                "channel_order_sha256": source["study_common_order_hash"],
                                "channel_set_sha256": source["study_common_set_hash"],
                                "reference_id": sl020.SOURCE_REFERENCE_ID,
                                "reference_set_sha256": None, "support_status": "known_from_complete_brainvision_file",
                                "estimator_id": result.estimator_id,
                                "vector_construction_id": result.vector_construction_id,
                                "scaling_id": result.scaling_id,
                                "component_names": list(result.component_names),
                                "component_values": values, "component_status": statuses,
                                "component_reasons": reasons, "computed": result.computed,
                                "valid": result.valid, "qualified": False, "qc": list(result.qc),
                            })
                    del payload
                else:
                    names = list(COMPONENTS)
                    for offset in OFFSETS_SECONDS:
                        for window_index in range(WINDOWS_PER_OFFSET):
                            start = offset * FS_HZ + window_index * WINDOW_SAMPLES
                            end = start + WINDOW_SAMPLES
                            for component in COMPONENTS:
                                local_values[(offset, component)].append(None)
                            window_rows.append({
                                "participant_id": participant, "session_id": session,
                                "condition": condition, "offset_seconds": offset,
                                "window_index": window_index,
                                "window_id": f"sl021b-20s-o{offset:02d}-{start:06d}-{end:06d}",
                                "start_sample": start, "end_sample_exclusive": end,
                                "start_seconds": start / FS_HZ, "end_seconds": end / FS_HZ,
                                "source_payload_sha256": source["payload_sha256"],
                                "realization_id": sl020.SENSOR_REALIZATION_ID,
                                "channel_labels": labels, "channel_count": len(labels),
                                "channel_order_sha256": source["study_common_order_hash"],
                                "channel_set_sha256": source["study_common_set_hash"],
                                "reference_id": sl020.SOURCE_REFERENCE_ID,
                                "reference_set_sha256": None, "support_status": "unsupported_file_source",
                                "estimator_id": ar_kernel.AR_ACTIVITY_ESTIMATOR_ID,
                                "vector_construction_id": sl020.AR_VECTOR_IDS[20],
                                "scaling_id": sl020.AR_SCALING_ID,
                                "component_names": names, "component_values": [None] * 9,
                                "component_status": ["unsupported"] * 9,
                                "component_reasons": [source_reason] * 9,
                                "computed": False, "valid": False, "qualified": False,
                                "qc": list(source["metadata_qc"]),
                            })
                for (offset, component), values in local_values.items():
                    if len(values) != WINDOWS_PER_OFFSET:
                        raise RuntimeError(f"planned_window_count_mismatch:{key}:{offset}:{component}")
                    computable = [value for value in values if value is not None]
                    complete = len(computable) == WINDOWS_PER_OFFSET
                    remaining_support.append({
                        "participant_group": "remaining48", "participant_id": participant,
                        "session_id": session, "condition": condition,
                        "offset_seconds": offset, "component_id": component,
                        "requested_window_count": WINDOWS_PER_OFFSET,
                        "computable_window_count": len(computable),
                        "session_median_emitted": complete,
                        "session_median": float(np.median(computable)) if complete else None,
                        "reason": None if complete else (source_reason or "requires_all_14_component_windows"),
                        "qualified": False,
                    })

    expected_recordings = len(REMAINING_IDS) * len(SESSIONS) * len(CONDITIONS)
    expected_windows = expected_recordings * len(OFFSETS_SECONDS) * WINDOWS_PER_OFFSET
    if planned_recordings != expected_recordings or len(source_records) != expected_recordings:
        raise RuntimeError(f"remaining_recording_count_mismatch:{planned_recordings}:{expected_recordings}")
    if len(window_rows) != expected_windows:
        raise RuntimeError(f"remaining_window_count_mismatch:{len(window_rows)}:{expected_windows}")

    pilot_support_grouped = [{**row, "participant_group": "development_pilot12"} for row in pilot_support]
    pilot_contrasts = _participant_contrasts(pilot_support_grouped, tuple(sl021.PILOT_IDS), "development_pilot12")
    pilot_reproduction = _compare_contrasts(pilot_contrasts, pilot_root)
    remaining_contrasts = _participant_contrasts(remaining_support, REMAINING_IDS, "remaining48")
    combined_ids = tuple(f"sub-{i:02d}" for i in range(1, 61))
    combined_support = pilot_support_grouped + remaining_support
    combined_contrasts = _participant_contrasts(combined_support, combined_ids, "combined60")
    remaining_summaries, remaining_tests = _summaries_and_tests(remaining_contrasts, REMAINING_IDS, "remaining48")
    combined_summaries, combined_tests = _summaries_and_tests(combined_contrasts, combined_ids, "combined60")

    out_remaining = output_root / "remaining48"
    out_combined = output_root / "combined60"
    write_csv(out_remaining / "window_components.csv", window_rows)
    write_csv(out_remaining / "session_support.csv", remaining_support)
    write_csv(out_remaining / "participant_contrasts.csv", remaining_contrasts)
    write_csv(out_remaining / "offset_component_summaries.csv", remaining_summaries)
    write_csv(out_remaining / "primary_target_tests.csv", remaining_tests)
    write_csv(out_combined / "session_support.csv", combined_support)
    write_csv(out_combined / "participant_contrasts.csv", combined_contrasts)
    write_csv(out_combined / "offset_component_summaries.csv", combined_summaries)
    write_csv(out_combined / "primary_target_tests.csv", combined_tests)
    reproduction = {
        "recomputed_pilot_participant_contrast_rows": pilot_reproduction["rows_compared"],
        "maximum_absolute_difference": pilot_reproduction["maximum_absolute_difference"],
        "absolute_tolerance": pilot_reproduction["absolute_tolerance"],
        "status": pilot_reproduction["status"], "qualified": False,
    }
    (output_root / "pilot_reproduction_check.json").write_text(
        json.dumps(reproduction, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )

    output_relatives = [
        "remaining48/window_components.csv", "remaining48/session_support.csv",
        "remaining48/participant_contrasts.csv", "remaining48/offset_component_summaries.csv",
        "remaining48/primary_target_tests.csv", "combined60/session_support.csv",
        "combined60/participant_contrasts.csv", "combined60/offset_component_summaries.csv",
        "combined60/primary_target_tests.csv", "pilot_reproduction_check.json",
    ]
    output_hashes = {relative: sha256_file(output_root / relative) for relative in output_relatives}
    manifest = {
        "protocol_id": PROTOCOL_ID, "status": "complete",
        "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": "ds004148", "dataset_version": "doi:10.18112/openneuro.ds004148.v1.0.1",
        "remaining_participant_roster": list(REMAINING_IDS), "remaining_participant_count": len(REMAINING_IDS),
        "pilot_participant_roster": list(sl021.PILOT_IDS), "pilot_participant_count": len(sl021.PILOT_IDS),
        "combined_participant_count": len(combined_ids), "condition_ids": list(CONDITIONS),
        "sessions": list(SESSIONS), "planned_recording_count": planned_recordings,
        "supported_recording_count": supported_recordings,
        "unsupported_recordings": [item for item in source_records if item["unsupported_reason"]],
        "remaining_window_record_count": len(window_rows),
        "remaining_session_support_row_count": len(remaining_support),
        "combined_session_support_row_count": len(combined_support),
        "window_duration_seconds": 20, "offsets_seconds": list(OFFSETS_SECONDS),
        "windows_per_offset_per_recording": WINDOWS_PER_OFFSET,
        "ar_estimator_id": ar_kernel.AR_ACTIVITY_ESTIMATOR_ID,
        "ar_vector_construction_id": sl020.AR_VECTOR_IDS[20], "ar_scaling_id": sl020.AR_SCALING_ID,
        "sensor_realization_id": sl020.SENSOR_REALIZATION_ID,
        "channel_count": 61, "sampling_frequency_hz": FS_HZ,
        "declared_reference_id": sl020.SOURCE_REFERENCE_ID, "reference_set_sha256": None,
        "pilot_reproduction_check": reproduction,
        "input_hashes": {
            "protocol_sha256": protocol_hash,
            "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
            "sl020_runner_sha256": current_sl020_hash,
            "sl021_pilot_runner_sha256": current_sl021_hash,
            "ar_kernel_sha256": current_ar_hash,
            "r1_audit_sha256": sha256_file(r1_manifest_path),
            "sl020_cohort_manifest_sha256": sha256_file(cohort_manifest_path),
            "sl021_pilot_manifest_sha256": sha256_file(pilot_manifest_path),
            "sl021_pilot_output_hashes": pilot_output_hashes,
        },
        "remaining_source_recordings": source_records,
        "output_sha256": output_hashes,
        "bootstrap_resamples": BOOTSTRAPS, "bootstrap_seed": SEED,
        "pvalue_family_per_participant_set": "two target components by four frozen offsets; Holm across 8 two-sided Wilcoxon tests",
        "participant_aggregation": "median over 14 windows per session/condition; mean paired-session deltas if at least 2 complete paired sessions",
        "effect_stability_summary": "range and positive-offset count of four group median contrasts; descriptive only, no equivalence criterion",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "raw_samples_read": True, "clinical_metadata_read": False,
        "qualified": False, "interpretation": "internal_same_dataset_alignment_robustness_analysis",
        "output_root": str(output_root),
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "complete", "remaining_participants": len(REMAINING_IDS),
        "recordings": planned_recordings, "supported_recordings": supported_recordings,
        "window_records": len(window_rows), "pilot_reproduction": reproduction,
        "qualified": False, "output_root": str(output_root),
    }, indent=2), flush=True)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--r1-manifest", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    run(
        dataset_root=args.dataset_root, r1_manifest_path=args.r1_manifest,
        cohort_manifest_path=args.cohort_manifest, pilot_root=args.pilot_root,
        protocol_path=args.protocol, output_root=args.output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
