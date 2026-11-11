"""Bounded SL-021 AR window-phase experiment for ds004148.

The script reuses SL-020's source verification and AR estimator unchanged. It
reads only the frozen 12-participant pilot roster and writes to a new output
root supplied by the caller. Results remain exploratory and unqualified.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
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


PROTOCOL_ID = "SL-021-AR-ALIGNMENT-SPECIFICITY-PILOT12-001"
PILOT_IDS = (
    "sub-16", "sub-21", "sub-23", "sub-55", "sub-42", "sub-19",
    "sub-39", "sub-58", "sub-13", "sub-07", "sub-32", "sub-29",
)
SESSIONS = ("ses-session1", "ses-session2", "ses-session3")
CONDITIONS = ("eyesclosed", "music")
OFFSETS_SECONDS = (0, 5, 10, 15)
FS_HZ = 500
WINDOW_SECONDS = 20
WINDOW_SAMPLES = FS_HZ * WINDOW_SECONDS
WINDOWS_PER_OFFSET = 14
BOOTSTRAPS = 10_000
SEED = 20261021
TARGET_COMPONENTS = ("block_004_to_005", "block_007_to_008")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path}")
    return value


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda index: (pvalues[index], index))
    adjusted: list[float] = [1.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _ar_for_window(signals: np.ndarray, labels: list[str], start_sample: int):
    block_samples = 2 * FS_HZ
    if signals.shape != (len(labels), WINDOW_SAMPLES):
        raise ValueError("window_shape_or_channel_label_count_mismatch")
    block_vectors = np.asarray([
        np.sqrt(np.mean(signals[:, left:left + block_samples] ** 2, axis=1, dtype=np.float64))
        for left in range(0, signals.shape[1], block_samples)
    ], dtype=np.float64)
    start_seconds = start_sample / FS_HZ
    intervals = [
        (start_seconds + index * 2.0, start_seconds + (index + 1) * 2.0)
        for index in range(block_vectors.shape[0])
    ]
    return ar_kernel.compute_ar_activity_transition_vector(
        block_vectors, intervals,
        vector_construction_id=sl020.AR_VECTOR_IDS[WINDOW_SECONDS],
        scaling_id=sl020.AR_SCALING_ID,
        channel_labels=labels,
    )


def _reference_rows(pilot_root: Path, participant: str) -> dict[int, dict[str, Any]]:
    manifest = read_json(pilot_root / "run_manifest.json")
    info = manifest.get("completed_participants", {}).get(participant)
    if manifest.get("status") != "complete" or not info:
        raise RuntimeError("existing_SL020_pilot_reference_incomplete")
    shard = pilot_root / info["relative_path"]
    if not shard.is_file() or sha256_file(shard) != info.get("sha256"):
        raise RuntimeError("existing_SL020_pilot_reference_shard_hash_mismatch")
    out: dict[int, dict[str, Any]] = {}
    with gzip.open(shard, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if (row.get("record_type") == "measurement" and row.get("family_id") == "AR"
                    and row.get("duration_seconds") == WINDOW_SECONDS
                    and row.get("participant_id") == participant
                    and row.get("session_id") == "ses-session1"
                    and row.get("condition") == "eyesclosed"):
                start = int(row["start_sample"])
                if start % WINDOW_SAMPLES == 0:
                    out[start] = row
    if len(out) != 15:
        raise RuntimeError(f"existing_SL020_reference_expected_15_windows_got_{len(out)}")
    return out


def _bootstrap_median(values: np.ndarray, rng: np.random.Generator) -> tuple[float | None, float | None]:
    if values.size == 0:
        return None, None
    samples = values[rng.integers(0, values.size, size=(BOOTSTRAPS, values.size))]
    medians = np.median(samples, axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def _participant_contrasts(
    session_medians: dict[tuple[str, str, str, int, str], float],
) -> tuple[list[dict[str, Any]], dict[tuple[int, str], dict[str, float]]]:
    rows: list[dict[str, Any]] = []
    values_by_offset_component: dict[tuple[int, str], dict[str, float]] = {}
    components = tuple(f"block_{i:03d}_to_{i + 1:03d}" for i in range(9))
    for offset in OFFSETS_SECONDS:
        for component in components:
            group: dict[str, float] = {}
            for participant in PILOT_IDS:
                paired_session_deltas = []
                for session in SESSIONS:
                    ec = session_medians.get((participant, session, "eyesclosed", offset, component))
                    music = session_medians.get((participant, session, "music", offset, component))
                    if ec is not None and music is not None:
                        paired_session_deltas.append(music - ec)
                delta = float(np.mean(paired_session_deltas)) if len(paired_session_deltas) >= 2 else None
                if delta is not None:
                    group[participant] = delta
                rows.append({
                    "participant_id": participant,
                    "offset_seconds": offset,
                    "component_id": component,
                    "paired_session_count": len(paired_session_deltas),
                    "participant_music_minus_eyesclosed": delta,
                    "eligible_two_or_more_sessions": delta is not None,
                    "qualified": False,
                })
            values_by_offset_component[(offset, component)] = group
    return rows, values_by_offset_component


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
        raise ValueError("SL021_output_root_must_be_on_J_drive")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing_nonempty_SL021_output_root:{output_root}")
    if not output_root.exists():
        output_root.mkdir(parents=True)

    r1 = read_json(r1_manifest_path)
    cohort = read_json(cohort_manifest_path)
    pilot_manifest_path = pilot_root / "run_manifest.json"
    pilot_manifest = read_json(pilot_manifest_path)
    if r1.get("audit_id") != sl020.AUDIT_ID or r1.get("source_root") != str(dataset_root):
        raise RuntimeError("R1_audit_identity_mismatch")
    if cohort.get("status") != "complete" or pilot_manifest.get("status") != "complete":
        raise RuntimeError("SL020_source_run_not_complete")
    if cohort.get("frozen_identity", {}).get("mode") != "cohort":
        raise RuntimeError("SL020_cohort_manifest_mode_mismatch")
    if pilot_manifest.get("frozen_identity", {}).get("mode") != "pilot":
        raise RuntimeError("SL020_pilot_manifest_mode_mismatch")
    if tuple(pilot_manifest.get("frozen_identity", {}).get("selected_participants", ())) != PILOT_IDS:
        raise RuntimeError("SL020_pilot_roster_changed")
    if cohort.get("protocol_id") != sl020.PROTOCOL_ID or pilot_manifest.get("protocol_id") != sl020.PROTOCOL_ID:
        raise RuntimeError("SL020_source_protocol_id_mismatch")
    protocol_sidecar = Path(str(protocol_path) + ".sha256")
    if not protocol_sidecar.is_file():
        raise FileNotFoundError(f"frozen_SL021_protocol_hash_missing:{protocol_sidecar}")
    protocol_hash = sha256_file(protocol_path)
    expected_protocol_hash = protocol_sidecar.read_text(encoding="ascii").split()[0].lower()
    if protocol_hash != expected_protocol_hash:
        raise RuntimeError("frozen_SL021_protocol_hash_mismatch")
    current_runner_hash = sha256_file(Path(sl020.__file__).resolve())
    current_ar_hash = sha256_file(Path(ar_kernel.__file__).resolve())
    for label, manifest in (("cohort", cohort), ("pilot", pilot_manifest)):
        frozen_hashes = manifest.get("frozen_identity", {}).get("code_hashes", {})
        if frozen_hashes.get("runner") != current_runner_hash or frozen_hashes.get("ar") != current_ar_hash:
            raise RuntimeError(f"current_SL020_runner_or_AR_kernel_differs_from_frozen_{label}_run")

    audit_records = r1.get("recordings", [])
    r1_by_key = {(item["participant_id"], item["session_id"], item["task_id"]): item for item in audit_records}
    if len(r1_by_key) != 900:
        raise RuntimeError(f"R1_recording_count_mismatch:{len(r1_by_key)}")
    cohort_sources: dict[tuple[str, str, str], str] = {}
    for participant in PILOT_IDS:
        completed = cohort.get("completed_participants", {}).get(participant)
        if not completed:
            raise RuntimeError(f"SL020_cohort_missing_pilot_participant:{participant}")
        for source in completed.get("source_recordings", []):
            key = (participant, source["session_id"], source["task_id"])
            cohort_sources[key] = source["payload_sha256"]

    reference = _reference_rows(pilot_root, "sub-16")
    window_rows: list[dict[str, Any]] = []
    session_values: dict[tuple[str, str, str, int, str], list[float]] = {}
    source_records: list[dict[str, Any]] = []
    reproduction_differences: list[float] = []
    reproduction_slots_compared = 0
    reproduction_finite_values_compared = 0
    session_support_rows: list[dict[str, Any]] = []
    expected_recordings = len(PILOT_IDS) * len(SESSIONS) * len(CONDITIONS)
    started = datetime.now(timezone.utc).isoformat()

    for participant in PILOT_IDS:
        print(f"SL021 processing {participant}", flush=True)
        for session in SESSIONS:
            for condition in CONDITIONS:
                key = (participant, session, condition)
                audit_record = r1_by_key.get(key)
                if audit_record is None or key not in cohort_sources:
                    raise RuntimeError(f"frozen_source_record_missing:{key}")
                paths, header, metadata_hashes, labels, reorder = sl020._verify_source_metadata(dataset_root, audit_record)
                source = sl020._payload_source_record(dataset_root, audit_record, paths, header, labels, reorder)
                if source["payload_sha256"] != cohort_sources[key]:
                    raise RuntimeError(f"raw_source_hash_differs_from_SL020:{key}")
                source_records.append({
                    "participant_id": participant, "session_id": session, "condition": condition,
                    "relative_payload_path": source["relative_payload_path"],
                    "payload_sha256": source["payload_sha256"],
                    "metadata_hashes": metadata_hashes,
                    "channel_count": len(labels),
                    "channel_order_sha256": source["study_common_order_hash"],
                    "channel_set_sha256": source["study_common_set_hash"],
                    "declared_reference": source["declared_reference"],
                    "reference_set_hash": source["reference_set_hash"],
                    "metadata_qc": source["metadata_qc"],
                })
                payload = np.memmap(
                    paths["eeg"], dtype="<f4", mode="r",
                    shape=(int(source["declared_samples"]), int(header["channel_count"])), order="C",
                )
                channel_indices = np.asarray([int(index) for index in reorder], dtype=np.int64)
                local_component_values: dict[tuple[int, str], list[float | None]] = {}
                for offset in OFFSETS_SECONDS:
                    phase_start = offset * FS_HZ
                    for window_index in range(WINDOWS_PER_OFFSET):
                        start = phase_start + window_index * WINDOW_SAMPLES
                        end = start + WINDOW_SAMPLES
                        if end > int(source["declared_samples"]):
                            raise RuntimeError(f"partial_window_forbidden:{key}:{offset}:{window_index}")
                        signals = np.asarray(payload[start:end, :][:, channel_indices].T, dtype=np.float64)
                        result = _ar_for_window(signals, labels, start)
                        if len(result.component_names) != 9 or len(result.component_values) != 9:
                            raise RuntimeError("unexpected_AR_component_schema")
                        row = {
                            "participant_id": participant, "session_id": session, "condition": condition,
                            "offset_seconds": offset, "window_index": window_index,
                            "window_id": f"sl021-20s-o{offset:02d}-{start:06d}-{end:06d}",
                            "start_sample": start, "end_sample_exclusive": end,
                            "start_seconds": start / FS_HZ, "end_seconds": end / FS_HZ,
                            "source_payload_sha256": source["payload_sha256"],
                            "realization_id": sl020.SENSOR_REALIZATION_ID,
                            "channel_labels": labels, "channel_count": len(labels),
                            "channel_order_sha256": source["study_common_order_hash"],
                            "channel_set_sha256": source["study_common_set_hash"],
                            "reference_id": sl020.SOURCE_REFERENCE_ID,
                            "reference_set_sha256": None,
                            "estimator_id": result.estimator_id,
                            "vector_construction_id": result.vector_construction_id,
                            "scaling_id": result.scaling_id,
                            "component_names": list(result.component_names),
                            "component_values": list(result.component_values),
                            "component_status": list(result.component_status),
                            "component_reasons": list(result.component_reasons),
                            "computed": result.computed, "valid": result.valid,
                            "qualified": False, "qc": list(result.qc),
                        }
                        window_rows.append(row)
                        for component, value in zip(result.component_names, result.component_values, strict=True):
                            local_component_values.setdefault((offset, component), []).append(
                                float(value) if value is not None and math.isfinite(float(value)) else None
                            )
                        if participant == "sub-16" and session == "ses-session1" and condition == "eyesclosed" and offset == 0:
                            reference_row = reference.get(start)
                            if reference_row is None:
                                raise RuntimeError(f"SL020_reference_window_missing:{start}")
                            old_names = reference_row.get("component_names") or []
                            old_values = reference_row.get("component_values") or []
                            if list(result.component_names) != old_names:
                                raise RuntimeError("SL020_reference_component_names_changed")
                            for new, old in zip(result.component_values, old_values, strict=True):
                                reproduction_slots_compared += 1
                                if new is None or old is None:
                                    if new is not old:
                                        raise RuntimeError("SL020_reference_null_pattern_changed")
                                else:
                                    difference = abs(float(new) - float(old))
                                    reproduction_differences.append(difference)
                                    reproduction_finite_values_compared += 1
                                    if difference > 1e-15:
                                        raise RuntimeError(f"SL020_reference_value_mismatch:{difference}")
                # The historical 0-second SL-020 grid has a fifteenth endpoint
                # window; compute it for the predeclared single-recording audit.
                if participant == "sub-16" and session == "ses-session1" and condition == "eyesclosed":
                    start, end = 14 * WINDOW_SAMPLES, 15 * WINDOW_SAMPLES
                    signals = np.asarray(payload[start:end, :][:, channel_indices].T, dtype=np.float64)
                    result = _ar_for_window(signals, labels, start)
                    reference_row = reference.get(start)
                    if reference_row is None or list(result.component_names) != reference_row.get("component_names"):
                        raise RuntimeError("SL020_reference_endpoint_missing_or_schema_changed")
                    for new, old in zip(result.component_values, reference_row["component_values"], strict=True):
                        reproduction_slots_compared += 1
                        if new is None or old is None:
                            if new is not old:
                                raise RuntimeError("SL020_reference_endpoint_null_pattern_changed")
                        else:
                            difference = abs(float(new) - float(old))
                            reproduction_differences.append(difference)
                            reproduction_finite_values_compared += 1
                            if difference > 1e-15:
                                raise RuntimeError(f"SL020_reference_endpoint_mismatch:{difference}")
                del payload
                for (offset, component), values in local_component_values.items():
                    if len(values) != WINDOWS_PER_OFFSET:
                        raise RuntimeError(f"incomplete_AR_windows:{key}:{offset}:{component}:{len(values)}")
                    computable = [value for value in values if value is not None]
                    complete = len(computable) == WINDOWS_PER_OFFSET
                    session_support_rows.append({
                        "participant_id": participant, "session_id": session, "condition": condition,
                        "offset_seconds": offset, "component_id": component,
                        "requested_window_count": WINDOWS_PER_OFFSET,
                        "computable_window_count": len(computable),
                        "session_median_emitted": complete,
                        "session_median": float(np.median(computable)) if complete else None,
                        "reason": None if complete else "requires_all_14_component_windows",
                        "qualified": False,
                    })
                    if complete:
                        session_values[(participant, session, condition, offset, component)] = computable

    if len(source_records) != expected_recordings:
        raise RuntimeError(f"source_recording_count_mismatch:{len(source_records)}:{expected_recordings}")
    if reproduction_slots_compared != 135:
        raise RuntimeError(f"SL020_reproduction_slot_count_mismatch:{reproduction_slots_compared}")

    session_medians = {key: float(np.median(values)) for key, values in session_values.items()}
    participant_rows, participant_values = _participant_contrasts(session_medians)
    rng = np.random.default_rng(SEED)
    summary_rows: list[dict[str, Any]] = []
    for offset in OFFSETS_SECONDS:
        for component_index in range(9):
            component = f"block_{component_index:03d}_to_{component_index + 1:03d}"
            group = participant_values[(offset, component)]
            values = np.asarray([group[pid] for pid in PILOT_IDS if pid in group], dtype=float)
            low, high = _bootstrap_median(values, rng)
            summary_rows.append({
                "offset_seconds": offset, "component_id": component,
                "participant_count": int(values.size),
                "participant_ids": [pid for pid in PILOT_IDS if pid in group],
                "median_participant_music_minus_eyesclosed": float(np.median(values)) if values.size else None,
                "bootstrap_ci95_low": low, "bootstrap_ci95_high": high,
                "median_paired_sessions": float(np.median([
                    row["paired_session_count"] for row in participant_rows
                    if row["offset_seconds"] == offset and row["component_id"] == component
                    and row["eligible_two_or_more_sessions"]
                ])) if values.size else None,
                "descriptive_only_for_neighbor_components": component not in TARGET_COMPONENTS,
                "qualified": False,
            })

    target_rows: list[dict[str, Any]] = []
    for offset in OFFSETS_SECONDS:
        for component in TARGET_COMPONENTS:
            group = participant_values[(offset, component)]
            values = np.asarray([group[pid] for pid in PILOT_IDS if pid in group], dtype=float)
            statistic, pvalue, reason = None, 1.0, None
            if values.size < 2:
                reason = "fewer_than_2_eligible_participants"
            elif np.all(values == 0.0):
                statistic, pvalue = 0.0, 1.0
            else:
                try:
                    test = stats.wilcoxon(values, zero_method="wilcox", alternative="two-sided", method="auto")
                    statistic, pvalue = float(test.statistic), float(test.pvalue)
                except Exception as exc:
                    reason = f"wilcoxon_not_estimable:{type(exc).__name__}"
            target_rows.append({
                "offset_seconds": offset, "component_id": component,
                "participant_count": int(values.size), "wilcoxon_statistic": statistic,
                "p_raw_two_sided": pvalue, "not_estimable_reason": reason,
                "qualified": False,
            })
    if len(target_rows) != 8:
        raise RuntimeError(f"primary_target_family_size_mismatch:{len(target_rows)}")
    for row, adjusted in zip(target_rows, holm([float(item["p_raw_two_sided"]) for item in target_rows]), strict=True):
        row["p_holm_8"] = adjusted

    write_csv(output_root / "window_components.csv", window_rows)
    write_csv(output_root / "session_support.csv", session_support_rows)
    write_csv(output_root / "participant_contrasts.csv", participant_rows)
    write_csv(output_root / "offset_component_summaries.csv", summary_rows)
    write_csv(output_root / "primary_target_tests.csv", target_rows)
    reproduction = {
        "participant_id": "sub-16", "session_id": "ses-session1", "condition": "eyesclosed",
        "windows_compared": 15, "component_slots_compared": reproduction_slots_compared,
        "finite_values_compared": reproduction_finite_values_compared,
        "maximum_absolute_difference": max(reproduction_differences, default=0.0),
        "absolute_tolerance": 1e-15, "status": "pass", "qualified": False,
    }
    (output_root / "alignment_reproduction_check.json").write_text(
        json.dumps(reproduction, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )

    script_path = Path(__file__).resolve()
    input_hashes = {
        "protocol_sha256": protocol_hash,
        "analysis_script_sha256": sha256_file(script_path),
        "sl020_runner_sha256": sha256_file(Path(sl020.__file__).resolve()),
        "ar_kernel_sha256": sha256_file(Path(ar_kernel.__file__).resolve()),
        "r1_audit_sha256": sha256_file(r1_manifest_path),
        "sl020_cohort_manifest_sha256": sha256_file(cohort_manifest_path),
        "sl020_pilot_manifest_sha256": sha256_file(pilot_manifest_path),
        "sl020_sub16_pilot_shard_sha256": sha256_file(pilot_root / pilot_manifest["completed_participants"]["sub-16"]["relative_path"]),
    }
    output_files = [
        "window_components.csv", "session_support.csv", "participant_contrasts.csv", "offset_component_summaries.csv",
        "primary_target_tests.csv", "alignment_reproduction_check.json",
    ]
    manifest = {
        "protocol_id": PROTOCOL_ID, "status": "complete",
        "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": "ds004148", "dataset_version": "doi:10.18112/openneuro.ds004148.v1.0.1",
        "participant_roster": list(PILOT_IDS), "session_count_per_participant": len(SESSIONS),
        "condition_ids": list(CONDITIONS), "recording_count": len(source_records),
        "window_duration_seconds": WINDOW_SECONDS, "offsets_seconds": list(OFFSETS_SECONDS),
        "windows_per_offset_per_recording": WINDOWS_PER_OFFSET,
        "window_record_count": len(window_rows), "participant_contrast_count": len(participant_rows),
        "offset_component_summary_count": len(summary_rows), "primary_target_test_count": len(target_rows),
        "ar_estimator_id": ar_kernel.AR_ACTIVITY_ESTIMATOR_ID,
        "ar_vector_construction_id": sl020.AR_VECTOR_IDS[WINDOW_SECONDS],
        "ar_scaling_id": sl020.AR_SCALING_ID,
        "sensor_realization_id": sl020.SENSOR_REALIZATION_ID,
        "channel_count": 61, "sampling_frequency_hz": FS_HZ,
        "declared_reference_id": sl020.SOURCE_REFERENCE_ID,
        "reference_set_sha256": None,
        "source_recordings": source_records, "input_hashes": input_hashes,
        "output_sha256": {name: sha256_file(output_root / name) for name in output_files},
        "bootstrap_resamples": BOOTSTRAPS, "bootstrap_seed": SEED,
        "pvalue_family": "two target components by four frozen offsets; Holm across 8 two-sided Wilcoxon tests",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "raw_samples_read": True, "clinical_metadata_read": False,
        "qualified": False, "interpretation": "exploratory_same_dataset_alignment_robustness_pilot",
        "output_root": str(output_root),
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "complete", "participants": len(PILOT_IDS), "recordings": len(source_records),
        "windows": len(window_rows), "reproduction_check": reproduction,
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
