"""Outcome-blind first-day I-CARE v2.1 feasibility run for MNPS v3.2.

This runner inventories WFDB headers and payload paths only (never clinical
sidecars), freezes a deterministic patient sample, performs a three-minute
smoke test, and then computes all supported 20-second windows in the first
24 hours after ROSC for the frozen sample if the smoke test passes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence


os.environ.setdefault("MNE_USE_NUMBA", "false")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[6]
for _source_root in (REPO_ROOT / "core" / "src", REPO_ROOT / "mndm" / "src"):
    if str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))

DEFAULT_SOURCE_ROOT = Path(
    "E:/Science_Datasets/physionet/received/"
    "i-care_2_1_random100_longitudinal/training"
)
OUTPUT_ROOT = REPO_ROOT / "outputs" / "COMA-VECTOR-PREFLIGHT-002-R1"
PROTOCOL_PATH = REPO_ROOT / "docs" / "api" / "mnps_v3_2_coma_vector_preflight_002r1_protocol.rst"
BASE_CONFIG_PATH = REPO_ROOT / "mndm" / "config" / "sources" / "other" / "config_ingest_physionet_i-care_2_1.yaml"
MNPS_OVERLAY_PATH = REPO_ROOT / "mndm" / "config" / "sources" / "other" / "config_ingest_physionet_i-care_2_1_mnps_only.yaml"
COMMON_CONFIG_PATH = REPO_ROOT / "mndm" / "config" / "config_ingest_common_eeg.yaml"
EXPERIMENT_ID = "nmdm.mnps_v3_2.coma_vector_preflight_002r1.v1"
RESULT_SCHEMA_ID = "mndm.mnps_v3_2.coma_vector_preflight_002r1_window.v1"
MANIFEST_SCHEMA_ID = "mndm.mnps_v3_2.coma_vector_preflight_002r1_run_manifest.v1"
SELECTION_SEED = "COMA-VECTOR-PREFLIGHT-002|20261014|patient_id"
TARGET_END_SEC = 24 * 60 * 60
WINDOW_SECONDS = 20
SMOKE_WINDOW_COUNT = 9
CORE19 = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
    "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
)
RECORD_RE = re.compile(r"^(?P<subject>\d{4})_(?P<run>\d{3})_(?P<acq>\d{3})_EEG\.hea$", re.IGNORECASE)
CLOCK_RE = re.compile(r"^(?P<hours>\d{1,4}):(?P<minutes>\d{2}):(?P<seconds>\d{2}(?:\.\d+)?)\+?$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    import numpy as np

    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path}")
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list, tuple)) else _json_safe(value)
                for key, value in row.items()
            })


def _runtime_dependencies() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in ("numpy", "scipy", "mne", "wfdb", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _code_hashes() -> dict[str, str]:
    paths = {
        "runner": Path(__file__).resolve(),
        "preprocess": REPO_ROOT / "mndm" / "src" / "mndm" / "preprocess.py",
        "time_reference": REPO_ROOT / "mndm" / "src" / "mndm" / "pipeline" / "time_reference.py",
        "support_provenance": REPO_ROOT / "mndm" / "src" / "mndm" / "signal_support_provenance.py",
        "ed_kernel": REPO_ROOT / "mndm" / "src" / "mndm" / "mnps" / "v3_2" / "ed.py",
        "oi_kernel": REPO_ROOT / "mndm" / "src" / "mndm" / "mnps" / "v3_2" / "oi.py",
        "nie_kernel": REPO_ROOT / "mndm" / "src" / "mndm" / "mnps" / "v3_2" / "nie.py",
        "shared_kernel_helpers": REPO_ROOT / "mndm" / "src" / "mndm" / "mnps" / "v3_2" / "experiments" / "vector_pilot_001.py",
    }
    return {key: sha256_file(path) for key, path in paths.items()}


def _config_hashes() -> dict[str, str]:
    return {
        "base_i_care_config": sha256_file(BASE_CONFIG_PATH),
        "mnps_only_overlay": sha256_file(MNPS_OVERLAY_PATH),
        "common_eeg_config": sha256_file(COMMON_CONFIG_PATH),
    }


def _clock_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    match = CLOCK_RE.match(str(value).strip())
    if match is None:
        return None
    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    seconds = float(match.group("seconds"))
    if minutes >= 60 or seconds >= 60 or not math.isfinite(seconds):
        return None
    return float(hours * 3600 + minutes * 60 + seconds)


def _load_resolved_config() -> dict[str, Any]:
    from core import config_loader

    config = config_loader.load_config(MNPS_OVERLAY_PATH)
    if config.get("preprocess", {}).get("notch_hz") is not None:
        raise ValueError("frozen_i_care_config_notch_must_be_disabled")
    if config.get("preprocess", {}).get("eeg_bandpass") != [0.5, 40]:
        raise ValueError("frozen_i_care_config_bandpass_mismatch")
    if config.get("preprocess", {}).get("reref") != "average":
        raise ValueError("frozen_i_care_config_reference_mismatch")
    if config.get("preprocess", {}).get("wfdb", {}).get("scale_to_microvolts") is not False:
        raise ValueError("frozen_i_care_config_unit_scaling_mismatch")
    return config


def _scan_source_manifest(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"source_root_missing:{source_root}")

    records: list[dict[str, Any]] = []
    subject_dirs = sorted(path for path in source_root.iterdir() if path.is_dir())
    header_paths = sorted(source_root.rglob("*_EEG.hea"), key=lambda p: p.as_posix().lower())
    for header_index, header_path in enumerate(header_paths, start=1):
        match = RECORD_RE.match(header_path.name)
        if not match:
            continue
        relative_header = header_path.relative_to(source_root).as_posix()
        stat = header_path.stat()
        row: dict[str, Any] = {
            "subject_id": match.group("subject"),
            "run_id": match.group("run"),
            "acquisition_id": match.group("acq"),
            "header_relative_path": relative_header,
            "header_size_bytes": int(stat.st_size),
            "header_sha256": sha256_file(header_path),
            "payloads": [],
            "header_parse_status": "not_attempted",
            "time_interval_status": "not_attempted",
            "start_sec_from_rosc": None,
            "end_clock_sec_from_rosc": None,
            "end_exclusive_sec_from_rosc": None,
            "sample_rate_hz": None,
            "sample_count": None,
            "channel_labels_in_header_order": [],
            "channel_count": None,
            "qc": [],
        }
        try:
            header_lines = header_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            first_fields = header_lines[0].split() if header_lines else []
            if len(first_fields) < 4:
                raise ValueError("wfdb_record_line_incomplete")
            n_sig = int(first_fields[1])
            fs = float(first_fields[2].split("/", 1)[0])
            sample_count = int(first_fields[3])
            signal_lines = header_lines[1:1 + n_sig]
            if len(signal_lines) != n_sig:
                raise ValueError("wfdb_signal_line_count_mismatch")
            signal_fields = [line.split() for line in signal_lines]
            if any(len(fields) < 2 for fields in signal_fields):
                raise ValueError("wfdb_signal_line_incomplete")
            labels = [fields[-1] for fields in signal_fields]
            names = list(dict.fromkeys(fields[0] for fields in signal_fields))
            row.update({
                "header_parse_status": "ok",
                "sample_rate_hz": fs,
                "sample_count": sample_count,
                "channel_labels_in_header_order": labels,
                "channel_count": n_sig,
                "format_per_signal": [fields[1] for fields in signal_fields],
                "unit_per_signal": [
                    fields[2].split("/", 1)[1] if "/" in fields[2] else None
                    for fields in signal_fields
                ],
            })
            payloads: list[dict[str, Any]] = []
            for name in names:
                candidate = (header_path.parent / name).resolve()
                try:
                    candidate.relative_to(source_root)
                    path_ok = True
                except ValueError:
                    path_ok = False
                entry: dict[str, Any] = {
                    "relative_path": candidate.relative_to(source_root).as_posix() if path_ok else None,
                    "exists": bool(path_ok and candidate.is_file()),
                    "path_status": "within_source_root" if path_ok else "refused_outside_source_root",
                }
                if path_ok and candidate.is_file():
                    payload_stat = candidate.stat()
                    entry["size_bytes"] = int(payload_stat.st_size)
                payloads.append(entry)
            row["payloads"] = payloads
            start_raw = next((
                line.split(":", 1)[1].strip()
                for line in header_lines
                if re.match(r"^\s*#\s*start\s*time\s*:", line, re.IGNORECASE)
            ), None)
            end_raw = next((
                line.split(":", 1)[1].strip()
                for line in header_lines
                if re.match(r"^\s*#\s*end\s*time\s*:", line, re.IGNORECASE)
            ), None)
            start = _clock_seconds(start_raw)
            end_clock = _clock_seconds(end_raw)
            if start is not None and end_clock is not None:
                while end_clock < start:
                    end_clock += 24 * 60 * 60
            clocks = {
                "status": "ok" if start is not None and end_clock is not None else "missing_or_invalid_start_end",
                "run_start_clock_raw": start_raw,
                "run_end_clock_raw": end_raw,
                "run_start_clock_sec": start,
                "run_end_clock_sec": end_clock,
                "parser": "coma_preflight_wfdb_text_clock_v1",
            }
            row["clock_metadata"] = clocks
            if (
                isinstance(start, (int, float)) and math.isfinite(float(start))
                and isinstance(end_clock, (int, float)) and math.isfinite(float(end_clock))
                and fs > 0 and sample_count > 0
            ):
                end_exclusive = float(start) + sample_count / fs
                expected_last_sample = float(start) + (sample_count - 1) / fs
                delta = abs(float(end_clock) - expected_last_sample)
                timing_match = delta <= 1.01  # Header clocks are usually serialized to whole seconds.
                row.update({
                    "start_sec_from_rosc": float(start),
                    "end_clock_sec_from_rosc": float(end_clock),
                    "end_exclusive_sec_from_rosc": end_exclusive,
                    "header_end_last_sample_delta_sec": delta,
                    "time_interval_status": "verified_to_sample_grid" if timing_match else "header_end_sample_grid_mismatch",
                })
                if not timing_match:
                    row["qc"].append("header_end_time_does_not_match_sample_count_within_1_01s")
            else:
                row["time_interval_status"] = "unparseable_or_invalid_header_interval"
                row["qc"].append("missing_or_invalid_start_end_clock_or_sample_grid")
        except Exception as exc:
            row["header_parse_status"] = f"error:{type(exc).__name__}"
            row["time_interval_status"] = "unavailable"
            row["qc"].append(f"header_parse_failed:{type(exc).__name__}")
        records.append(row)
        if header_index % 500 == 0:
            print(f"header_inventory_progress={header_index}/{len(header_paths)}", flush=True)

    all_payload_inventory = []
    for payload_path in sorted(source_root.rglob("*.mat"), key=lambda p: p.as_posix().lower()):
        stat = payload_path.stat()
        all_payload_inventory.append({
            "relative_path": payload_path.relative_to(source_root).as_posix(),
            "size_bytes": int(stat.st_size),
        })

    subjects_with_headers = sorted({str(row["subject_id"]) for row in records})
    eligible: list[str] = []
    exclusions: list[dict[str, Any]] = []
    for subject_id in subjects_with_headers:
        subject_records = [row for row in records if str(row["subject_id"]) == subject_id]
        if any(
            row.get("header_parse_status") == "ok"
            and row.get("time_interval_status") in {"verified_to_sample_grid", "header_end_sample_grid_mismatch"}
            for row in subject_records
        ):
            eligible.append(subject_id)
        else:
            exclusions.append({"subject_id": subject_id, "reason": "no_parseable_header_interval_and_sample_grid"})
    for directory in subject_dirs:
        if directory.name.isdigit() and directory.name not in subjects_with_headers:
            exclusions.append({"subject_id": directory.name, "reason": "no_matching_EEG_header"})

    return {
        "schema_id": "mndm.mnps_v3_2.coma_vector_preflight_002_source_manifest.v1",
        "dataset_release": "PhysioNet I-CARE v2.1 training tree",
        "source_root": str(source_root),
        "sidecars_opened": False,
        "time_axis": "WFDB header clock relative to ROSC",
        "header_count": len(records),
        "payload_count": len(all_payload_inventory),
        "payload_total_bytes": sum(int(row["size_bytes"]) for row in all_payload_inventory),
        "subject_directory_count": len(subject_dirs),
        "eligible_subject_ids": eligible,
        "excluded_subjects": exclusions,
        "records": records,
        "payload_inventory": all_payload_inventory,
        "inventory_hash_policy": "all parsed EEG header bytes SHA-256; all MAT paths and sizes; selected MAT bytes SHA-256 before read",
    }


def _window_specs(record: Mapping[str, Any], limit: int | None = None) -> list[dict[str, Any]]:
    start = record.get("start_sec_from_rosc")
    end = record.get("end_exclusive_sec_from_rosc")
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return []
    start = float(start)
    end = float(end)
    if end <= 0 or start >= TARGET_END_SEC or end <= start:
        return []
    epsilon = 1e-9
    first_index = max(0, int(math.ceil((max(0.0, start) - epsilon) / WINDOW_SECONDS)))
    last_index = min(
        int(TARGET_END_SEC / WINDOW_SECONDS) - 1,
        int(math.floor((min(TARGET_END_SEC, end) - WINDOW_SECONDS + epsilon) / WINDOW_SECONDS)),
    )
    if last_index < first_index:
        return []
    result = [
        {
            "window_index_from_rosc": index,
            "window_id": f"t{index * WINDOW_SECONDS:05d}_{(index + 1) * WINDOW_SECONDS:05d}s",
            "window_start_sec_from_rosc": float(index * WINDOW_SECONDS),
            "window_end_sec_from_rosc_exclusive": float((index + 1) * WINDOW_SECONDS),
        }
        for index in range(first_index, last_index + 1)
    ]
    return result[:limit] if limit is not None else result


def _freeze_inputs(source_root: Path, *, selection_count: int = 16) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = OUTPUT_ROOT / "source_manifest.json"
    selection_path = OUTPUT_ROOT / "selection.json"
    if selection_path.exists():
        if not manifest_path.is_file() or not selection_path.is_file():
            raise FileExistsError("partial_freeze_exists_refusing_replacement")
        manifest = _read_json(manifest_path)
        selection = _read_json(selection_path)
        if sha256_json(manifest) != selection.get("source_manifest_sha256"):
            raise ValueError("frozen_source_manifest_hash_mismatch")
        if Path(str(manifest.get("source_root"))).resolve() != source_root.resolve():
            raise ValueError("frozen_source_root_mismatch")
        return manifest, selection

    if manifest_path.is_file():
        # A previously frozen, outcome-blind header inventory may seed a
        # versioned implementation repair without re-reading the large tree.
        manifest = _read_json(manifest_path)
        if Path(str(manifest.get("source_root"))).resolve() != source_root.resolve():
            raise ValueError("frozen_source_root_mismatch")
    else:
        manifest = _scan_source_manifest(source_root)
        _write_json(manifest_path, manifest)
    manifest_hash = sha256_json(manifest)
    if len(manifest["eligible_subject_ids"]) < 12:
        raise RuntimeError(f"fewer_than_12_eligible_patients:{len(manifest['eligible_subject_ids'])}")
    ordered = sorted(
        manifest["eligible_subject_ids"],
        key=lambda subject_id: hashlib.sha256(
            f"COMA-VECTOR-PREFLIGHT-002|20261014|{subject_id}".encode("utf-8")
        ).hexdigest(),
    )
    selected_ids = ordered[:selection_count]
    records_by_subject: dict[str, list[dict[str, Any]]] = {}
    for record in manifest["records"]:
        records_by_subject.setdefault(str(record["subject_id"]), []).append(record)
    smoke_subject: str | None = None
    smoke_record: dict[str, Any] | None = None
    for subject_id in selected_ids:
        subject_records = sorted(
            records_by_subject.get(subject_id, []),
            key=lambda row: (
                float(row["start_sec_from_rosc"]) if row.get("start_sec_from_rosc") is not None else math.inf,
                str(row.get("header_relative_path", "")).lower(),
            ),
        )
        for record in subject_records:
            if len(_window_specs(record, limit=SMOKE_WINDOW_COUNT)) < SMOKE_WINDOW_COUNT:
                continue
            if not any(payload.get("exists") is True for payload in record.get("payloads", [])):
                continue
            smoke_subject, smoke_record = subject_id, record
            break
        if smoke_subject is not None:
            break
    if smoke_subject is None or smoke_record is None:
        raise RuntimeError("no_selected_patient_has_nine_target_windows_in_a_paired_source_segment")
    selection = {
        "schema_id": "mndm.mnps_v3_2.coma_vector_preflight_002r1_selection.v1",
        "protocol_id": EXPERIMENT_ID,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "runner_and_kernel_sha256": _code_hashes(),
        "preprocessing_config_sha256": _config_hashes(),
        "source_manifest_sha256": manifest_hash,
        "selection_seed": SELECTION_SEED,
        "eligible_subject_count": len(ordered),
        "eligible_subject_order_by_sha256": ordered,
        "selected_subject_ids_in_draw_order": selected_ids,
        "smoke_subject_id": smoke_subject,
        "smoke_segment_relative_path": smoke_record["header_relative_path"],
        "smoke_window_count": SMOKE_WINDOW_COUNT,
        "source_root": str(source_root.resolve()),
        "outcome_fields_read": False,
        "selection_rationale": "first 16 eligible patient IDs in frozen SHA-256 draw; smoke fixture is first selected patient with nine target windows in a paired segment",
    }
    _write_json(selection_path, selection)
    return manifest, selection


def _verify_frozen_inputs(source_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = OUTPUT_ROOT / "source_manifest.json"
    selection_path = OUTPUT_ROOT / "selection.json"
    manifest = _read_json(manifest_path)
    selection = _read_json(selection_path)
    if sha256_json(manifest) != selection.get("source_manifest_sha256"):
        raise ValueError("frozen_source_manifest_hash_mismatch")
    if sha256_file(PROTOCOL_PATH) != selection.get("protocol_sha256"):
        raise ValueError("frozen_protocol_hash_mismatch")
    if _code_hashes() != selection.get("runner_and_kernel_sha256"):
        raise ValueError("frozen_runner_or_kernel_hash_mismatch")
    if _config_hashes() != selection.get("preprocessing_config_sha256"):
        raise ValueError("frozen_preprocessing_config_hash_mismatch")
    if Path(str(manifest.get("source_root"))).resolve() != source_root.resolve():
        raise ValueError("frozen_source_root_mismatch")
    return manifest, selection


def _hashes_from_preprocess(meta: Mapping[str, Any], header_path: Path) -> dict[str, Any]:
    provenance = meta.get("signal_support_provenance", {})
    source = provenance.get("source", {}) if isinstance(provenance, Mapping) else {}
    payload_rows = source.get("associated_signal_files", []) if isinstance(source, Mapping) else []
    return {
        "header_sha256_manifest": sha256_file(header_path),
        "header_sha256_preprocess": source.get("sha256") if isinstance(source, Mapping) else None,
        "header_sha256_after_read": source.get("sha256_after") if isinstance(source, Mapping) else None,
        "source_sha256_stable_after_read": source.get("hash_stable") if isinstance(source, Mapping) else None,
        "payloads": payload_rows,
    }


def _kernel_result(result: Any) -> dict[str, Any]:
    payload = _json_safe(result)
    if not isinstance(payload, dict):
        return {"value": None, "valid": False, "reason": "kernel_result_not_mapping"}
    value = payload.get("value")
    payload["value"] = value if isinstance(value, (int, float)) and math.isfinite(float(value)) else None
    payload["valid"] = bool(payload.get("valid", payload.get("computed", False)))
    return payload


def _compute_measurements(matrix: Any, labels: Sequence[str]) -> dict[str, Any]:
    import numpy as np
    from mndm.mnps.v3_2.ed import compute_ed_raw_sensor_covariance_participation_ratio
    from mndm.mnps.v3_2.nie import compute_nie_raw_weighted_global_efficiency
    from mndm.mnps.v3_2.oi import compute_oi_matrix
    from mndm.mnps.v3_2.experiments.vector_pilot_001 import (
        NIE_EDGE_ESTIMATOR_ID,
        NIE_EDGE_RULE_ID,
        construct_absolute_pearson_weights,
        labels_hashes,
    )

    signals = np.asarray(matrix, dtype=np.float64)
    ed = compute_ed_raw_sensor_covariance_participation_ratio(signals, channel_names=labels)
    oi = compute_oi_matrix(signals, channel_names=labels)
    weights, edge_audit = construct_absolute_pearson_weights(signals)
    if weights is None:
        nie_payload = {
            "value": None,
            "valid": False,
            "computed": False,
            "reason": str(edge_audit.get("reason") or "absolute_pearson_edge_construction_unsupported"),
            "edge_audit": edge_audit,
        }
    else:
        order_hash, _ = labels_hashes(labels)
        nie = compute_nie_raw_weighted_global_efficiency(
            weights,
            node_labels=labels,
            node_order_sha256=order_hash,
            edge_estimator_id=NIE_EDGE_ESTIMATOR_ID,
            edge_rule_id=NIE_EDGE_RULE_ID,
        )
        nie_payload = _kernel_result(nie)
        nie_payload["edge_audit"] = edge_audit
    return {"ED": _kernel_result(ed), "OI": _kernel_result(oi), "NIE": nie_payload}


def _realizations(signals: Any, labels: Sequence[str]) -> list[dict[str, Any]]:
    import numpy as np
    from mndm.mnps.v3_2.experiments.vector_pilot_001 import labels_hashes

    matrix = np.asarray(signals, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(labels):
        raise ValueError("preprocessed_EEG_shape_and_label_count_mismatch")
    native_order, native_set = labels_hashes(labels) if labels else (None, None)
    output = [{
        "realization_id": "native_eeg_channels.v1",
        "signals": matrix,
        "labels": list(labels),
        "channel_order_sha256": native_order,
        "channel_set_sha256": native_set,
        "header_order_preserved": True,
    }]
    if all(label in labels for label in CORE19):
        indices = [list(labels).index(label) for label in CORE19]
        core_matrix = matrix[indices, :]
        core_order, core_set = labels_hashes(CORE19)
        output.append({
            "realization_id": "core19_exact_average_referenced.v1",
            "signals": core_matrix,
            "labels": list(CORE19),
            "channel_order_sha256": core_order,
            "channel_set_sha256": core_set,
            "header_order_preserved": False,
        })
    return output


def _planned_crop(record: Mapping[str, Any]) -> dict[str, Any] | None:
    fs = float(record.get("sample_rate_hz") or 0.0)
    start = record.get("start_sec_from_rosc")
    end = record.get("end_exclusive_sec_from_rosc")
    n = int(record.get("sample_count") or 0)
    if fs <= 0 or n <= 0 or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return None
    start, end = float(start), float(end)
    overlap_start = max(0.0, start)
    overlap_end = min(float(TARGET_END_SEC), end)
    if overlap_end <= overlap_start:
        return None
    # Snap to complete source samples; MNE crop's inclusive tmax is set to the
    # final included source sample so no sample at/after 24 h enters the crop.
    first = max(0, int(math.ceil((overlap_start - start) * fs - 1e-8)))
    end_exclusive = min(n, int(math.floor((overlap_end - start) * fs + 1e-8)))
    if end_exclusive <= first:
        return None
    return {
        "source_sample_start": first,
        "source_sample_end_exclusive": end_exclusive,
        "start_sec_from_segment_start": first / fs,
        "stop_sec_inclusive_from_segment_start": (end_exclusive - 1) / fs,
        "start_sec_from_rosc": start + first / fs,
        "end_exclusive_sec_from_rosc": start + end_exclusive / fs,
        "source_sample_rate_hz": fs,
    }


def _unsupported_rows(
    record: Mapping[str, Any], windows: Sequence[Mapping[str, Any]], reason: str,
) -> list[dict[str, Any]]:
    labels = list(record.get("channel_labels_in_header_order", []))
    realization_specs: list[tuple[str, list[str], str | None, str | None]] = []
    from mndm.mnps.v3_2.experiments.vector_pilot_001 import labels_hashes
    order_hash, set_hash = labels_hashes(labels) if labels else (None, None)
    realization_specs.append(("native_eeg_channels.v1", labels, order_hash, set_hash))
    if all(label in labels for label in CORE19):
        core_order, core_set = labels_hashes(CORE19)
        realization_specs.append(("core19_exact_average_referenced.v1", list(CORE19), core_order, core_set))
    rows: list[dict[str, Any]] = []
    for window in windows:
        for realization_id, realization_labels, label_order_hash, label_set_hash in realization_specs:
            rows.append({
                "schema_id": RESULT_SCHEMA_ID,
                "subject_id": record.get("subject_id"),
                "segment_id": record.get("header_relative_path"),
                "window_id": window["window_id"],
                "window_index_from_rosc": window["window_index_from_rosc"],
                "window_start_sec_from_rosc": window["window_start_sec_from_rosc"],
                "window_end_sec_from_rosc_exclusive": window["window_end_sec_from_rosc_exclusive"],
                "realization_id": realization_id,
                "ordered_channel_labels": realization_labels,
                "channel_order_sha256": label_order_hash,
                "channel_set_sha256": label_set_hash,
                "channel_count": len(realization_labels),
                "support_status": "unsupported",
                "support_reason": reason,
                "effective_dependency_scope": None,
                "effective_dependency_start_sec_from_rosc": None,
                "effective_dependency_end_sec_from_rosc_exclusive": None,
                "metrics": {
                    family: {"value": None, "valid": False, "computed": False, "reason": reason}
                    for family in ("ED", "OI", "NIE")
                },
                "qualified": False,
            })
    return rows


def _process_segment(
    record: Mapping[str, Any],
    source_root: Path,
    resolved_config: Mapping[str, Any],
    *,
    window_limit: int | None = None,
) -> dict[str, Any]:
    import numpy as np
    from mndm.mnps.v3_2.experiments.vector_pilot_001 import labels_hashes, sha256_array
    from mndm.preprocess import preprocess_wfdb

    windows = _window_specs(record)
    if window_limit is not None:
        windows = windows[:window_limit]
    header_path = (source_root / str(record["header_relative_path"])).resolve()
    segment: dict[str, Any] = {
        "subject_id": record.get("subject_id"),
        "segment_id": record.get("header_relative_path"),
        "record_name": header_path.stem,
        "time_interval_status": record.get("time_interval_status"),
        "source_start_sec_from_rosc": record.get("start_sec_from_rosc"),
        "source_end_exclusive_sec_from_rosc": record.get("end_exclusive_sec_from_rosc"),
        "source_sample_rate_hz": record.get("sample_rate_hz"),
        "source_sample_count": record.get("sample_count"),
        "source_channel_labels_in_header_order": record.get("channel_labels_in_header_order", []),
        "window_count_planned": len(windows),
        "rows": [],
        "processing_status": "not_started",
    }
    if not windows:
        segment["processing_status"] = "no_full_target_windows"
        return segment
    if record.get("time_interval_status") != "verified_to_sample_grid":
        segment["processing_status"] = "unsupported"
        segment["reason"] = "header_time_interval_not_verified_to_sample_grid"
        segment["rows"] = _unsupported_rows(record, windows, segment["reason"])
        return segment
    payloads = record.get("payloads", [])
    existing = [
        (source_root / str(payload.get("relative_path"))).resolve()
        for payload in payloads
        if payload.get("exists") is True and payload.get("relative_path")
    ]
    if not existing or any(not path.is_file() for path in existing):
        segment["processing_status"] = "unsupported"
        segment["reason"] = "source_payload_missing"
        segment["rows"] = _unsupported_rows(record, windows, segment["reason"])
        return segment
    crop = _planned_crop(record)
    if crop is None:
        segment["processing_status"] = "unsupported"
        segment["reason"] = "no_source_sample_crop_inside_first_24h"
        segment["rows"] = _unsupported_rows(record, windows, segment["reason"])
        return segment

    local_config = dict(resolved_config)
    local_preprocess = dict(local_config.get("preprocess", {}))
    local_preprocess["crop"] = {
        "datasets": {
            "physionet_icare_2_1": {
                "start_sec": crop["start_sec_from_segment_start"],
                "stop_sec": crop["stop_sec_inclusive_from_segment_start"],
            }
        }
    }
    local_config["preprocess"] = local_preprocess
    started = time.perf_counter()
    processed = preprocess_wfdb(header_path, local_config)
    elapsed = time.perf_counter() - started
    values = np.asarray(processed.signals.get("eeg"), dtype=np.float64)
    labels = [str(value) for value in (processed.channels or {}).get("eeg", [])]
    sfreq = float(processed.sfreq)
    provenance = processed.meta.get("signal_support_provenance", {})
    provenance = provenance if isinstance(provenance, Mapping) else {}
    resolved_sfreq_raw = processed.meta.get("target_sfreq_resolved")
    try:
        resolved_sfreq = float(resolved_sfreq_raw)
    except (TypeError, ValueError, OverflowError):
        resolved_sfreq = math.nan
    resolved_sfreq_match = (
        math.isfinite(resolved_sfreq)
        and resolved_sfreq > 0
        and math.isclose(sfreq, resolved_sfreq, rel_tol=0.0, abs_tol=1e-9)
    )
    source_grid_ok = provenance.get("source_sample_grid_supported") is True
    source = provenance.get("source", {})
    source = source if isinstance(source, Mapping) else {}
    associated = source.get("associated_signal_files", [])
    hashes_stable = bool(associated) and all(
        isinstance(item, Mapping)
        and item.get("exists") is True
        and item.get("hash_status") == "captured_before_read"
        and bool(item.get("sha256"))
        and item.get("hash_stable") is True
        for item in associated
    )
    header_hash_stable = source.get("hash_stable") is True
    timing_match = record.get("time_interval_status") == "verified_to_sample_grid"
    crop_cfg_recorded = provenance.get("crop", {})
    crop_cfg_recorded = crop_cfg_recorded if isinstance(crop_cfg_recorded, Mapping) else {}
    crop_applied = crop_cfg_recorded.get("applied") is True
    expected_crop_start = int(crop["source_sample_start"])
    expected_crop_end = int(crop["source_sample_end_exclusive"])
    actual_crop_start = crop_cfg_recorded.get("actual_first_samp")
    actual_crop_end = crop_cfg_recorded.get("actual_last_samp_exclusive")
    crop_grid_match = (
        isinstance(actual_crop_start, int)
        and isinstance(actual_crop_end, int)
        and actual_crop_start == expected_crop_start
        and actual_crop_end == expected_crop_end
    )
    output_finite = values.ndim == 2 and values.size > 0 and bool(np.isfinite(values).all())
    crop_within_target = (
        float(crop["start_sec_from_rosc"]) >= -1e-9
        and float(crop["end_exclusive_sec_from_rosc"]) <= TARGET_END_SEC + 1e-9
    )
    support_complete = bool(
        source_grid_ok and hashes_stable and header_hash_stable and timing_match and crop_applied
        and crop_grid_match and output_finite and crop_within_target
        and resolved_sfreq_match
    )
    segment.update({
        "processing_status": "processed" if support_complete else "unsupported",
        "support_reason": None if support_complete else "source_or_preprocessing_support_audit_incomplete",
        "output_sfreq_hz": sfreq,
        "output_sample_count": int(values.shape[1]) if values.ndim == 2 else 0,
        "output_channel_labels": labels,
        "output_channel_count": len(labels),
        "realizations": [],
        "crop": {**crop, "applied": crop_applied, "actual_first_samp": actual_crop_start,
                 "actual_last_samp_exclusive": actual_crop_end, "crop_grid_matches_plan": crop_grid_match},
        "support_audit": {
            "source_sample_grid_supported": source_grid_ok,
            "selected_payload_hashes_present_before_read": hashes_stable,
            "header_hash_stable_after_read": header_hash_stable,
            "header_end_matches_sample_grid": timing_match,
            "crop_applied": crop_applied,
            "crop_grid_matches_plan": crop_grid_match,
            "resolved_target_sfreq_hz": resolved_sfreq if math.isfinite(resolved_sfreq) else None,
            "output_matches_resolved_target_sfreq": resolved_sfreq_match,
            "all_preprocessed_crop_samples_finite": output_finite,
            "whole_crop_dependency_within_first_day": crop_within_target,
            "effective_dependency_scope": "entire_preprocessed_source_crop_conservative_bound",
            "effective_dependency_start_sec_from_rosc": crop["start_sec_from_rosc"],
            "effective_dependency_end_sec_from_rosc_exclusive": crop["end_exclusive_sec_from_rosc"],
            "boundary_behavior": "MNE crop to exact source sample bounds before configured resampling/bandpass; configured whole-array resampling and filtering may use every sample in the crop; no samples outside this source segment crop are supplied",
            "filter_margin_seconds": "conservative_whole_crop_dependency",
        },
        "source_hash_audit": _hashes_from_preprocess(processed.meta, header_path),
        "preprocessing": {
            "dataset_id": processed.meta.get("dataset_id"),
            "source_format": processed.meta.get("source_format"),
            "source_sfreq_hz": processed.meta.get("original_sfreq"),
            "target_sfreq_hz": resolved_sfreq if math.isfinite(resolved_sfreq) else None,
            "sfreq_policy": processed.meta.get("sfreq_policy"),
            "reference_mode": processed.meta.get("eeg_reference_mode"),
            "reference_status": processed.meta.get("eeg_reference_status"),
            "reference_labels": labels,
            "reference_order_sha256": labels_hashes(labels)[0] if labels else None,
            "reference_set_sha256": labels_hashes(labels)[1] if labels else None,
            "source_units": processed.meta.get("wfdb_eeg_source_units"),
            "amplitude_unit": processed.meta.get("eeg_amplitude_unit"),
            "amplitude_unit_status": processed.meta.get("eeg_amplitude_unit_status"),
            "signal_support_provenance": provenance,
            "preprocess_timings": processed.meta.get("timings", {}),
            "wall_seconds": elapsed,
        },
    })

    realization_data = _realizations(values, labels) if labels else []
    segment["realizations"] = [
        {key: value for key, value in item.items() if key not in {"signals"}}
        for item in realization_data
    ]
    crop_start_abs = float(crop["start_sec_from_rosc"])
    for window in windows:
        window_start = float(window["window_start_sec_from_rosc"])
        window_end = float(window["window_end_sec_from_rosc_exclusive"])
        local_start_float = (window_start - crop_start_abs) * sfreq
        local_start = int(round(local_start_float))
        sample_count = int(round(WINDOW_SECONDS * sfreq))
        grid_aligned = math.isclose(local_start_float, local_start, rel_tol=0.0, abs_tol=1e-7)
        local_end = local_start + sample_count
        window_inside_crop = (
            grid_aligned and local_start >= 0 and local_end <= values.shape[1]
            and window_start >= crop_start_abs - 1e-9
            and window_end <= float(crop["end_exclusive_sec_from_rosc"]) + 1e-9
            and sample_count == int(round(WINDOW_SECONDS * sfreq))
        )
        row_support = support_complete and window_inside_crop
        for realization in realization_data:
            matrix_full = realization["signals"]
            labels_for_realization = list(realization["labels"])
            if row_support:
                matrix = np.asarray(matrix_full[:, local_start:local_end], dtype=np.float64)
                matrix_ok = matrix.shape == (len(labels_for_realization), sample_count) and bool(np.isfinite(matrix).all())
            else:
                matrix = np.empty((0, 0), dtype=np.float64)
                matrix_ok = False
            if row_support and matrix_ok:
                metrics = _compute_measurements(matrix, labels_for_realization)
                matrix_hash = sha256_array(matrix)
                support_status = "known_conservative_full_crop"
                row_reason = None
            else:
                reason = (
                    "window_sample_grid_not_aligned_or_outside_processed_crop"
                    if support_complete and not window_inside_crop
                    else "source_or_preprocessing_support_audit_incomplete"
                )
                metrics = {
                    family: {"value": None, "valid": False, "computed": False, "reason": reason}
                    for family in ("ED", "OI", "NIE")
                }
                matrix_hash = None
                support_status = "unsupported"
                row_reason = reason
            segment["rows"].append({
                "schema_id": RESULT_SCHEMA_ID,
                "experiment_id": EXPERIMENT_ID,
                "subject_id": record.get("subject_id"),
                "segment_id": record.get("header_relative_path"),
                "window_id": window["window_id"],
                "window_index_from_rosc": window["window_index_from_rosc"],
                "window_start_sec_from_rosc": window_start,
                "window_end_sec_from_rosc_exclusive": window_end,
                "window_duration_seconds": WINDOW_SECONDS,
                "realization_id": realization["realization_id"],
                "ordered_channel_labels": labels_for_realization,
                "channel_order_sha256": realization["channel_order_sha256"],
                "channel_set_sha256": realization["channel_set_sha256"],
                "channel_count": len(labels_for_realization),
                "reference_channel_labels": labels,
                "reference_set_sha256": segment["preprocessing"]["reference_set_sha256"],
                "sample_rate_hz": sfreq,
                "sample_count": sample_count if matrix_ok else 0,
                "support_status": support_status,
                "support_reason": row_reason,
                "effective_dependency_scope": "entire_preprocessed_source_crop_conservative_bound" if row_support else None,
                "effective_dependency_start_sec_from_rosc": crop_start_abs if row_support else None,
                "effective_dependency_end_sec_from_rosc_exclusive": crop["end_exclusive_sec_from_rosc"] if row_support else None,
                "source_sample_start": expected_crop_start if row_support else None,
                "source_sample_end_exclusive": expected_crop_end if row_support else None,
                "window_sample_start_in_crop": local_start if row_support else None,
                "window_sample_end_exclusive_in_crop": local_end if row_support else None,
                "window_signal_sha256": matrix_hash,
                "metrics": metrics,
                "qualified": False,
            })
    return segment


def _selected_records(manifest: Mapping[str, Any], selection: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    selected = set(str(value) for value in selection.get("selected_subject_ids_in_draw_order", []))
    output: dict[str, list[dict[str, Any]]] = {subject_id: [] for subject_id in selected}
    for record in manifest.get("records", []):
        subject_id = str(record.get("subject_id", ""))
        if subject_id in output:
            output[subject_id].append(dict(record))
    for subject_id in output:
        output[subject_id].sort(key=lambda row: (
            float(row["start_sec_from_rosc"]) if row.get("start_sec_from_rosc") is not None else math.inf,
            str(row.get("header_relative_path", "")).lower(),
        ))
    return output


def _collect_segment_files(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "segments").glob("*.json"), key=lambda path: path.name.lower())


def _aggregate_run(run_dir: Path, selection: Mapping[str, Any], *, mode: str, run_id: str) -> dict[str, Any]:
    segment_paths = _collect_segment_files(run_dir)
    segments = [_read_json(path) for path in segment_paths]
    rows = [row for segment in segments for row in segment.get("rows", [])]
    rows.sort(key=lambda row: (
        str(row.get("subject_id", "")),
        float(row.get("window_start_sec_from_rosc", math.inf)),
        str(row.get("segment_id", "")).lower(),
        str(row.get("realization_id", "")),
    ))
    _write_csv(run_dir / "window_results.csv", rows)

    participant_rows: list[dict[str, Any]] = []
    for subject_id in selection.get("selected_subject_ids_in_draw_order", []):
        subject_segments = [segment for segment in segments if str(segment.get("subject_id")) == str(subject_id)]
        subject_rows = [row for row in rows if str(row.get("subject_id")) == str(subject_id)]
        summary: dict[str, Any] = {
            "subject_id": str(subject_id),
            "source_segment_count_in_inventory": sum(
                1 for _ in subject_segments
            ),
            "window_realization_row_count": len(subject_rows),
            "supported_window_realization_rows": sum(row.get("support_status") != "unsupported" for row in subject_rows),
        }
        for family in ("ED", "OI", "NIE"):
            summary[f"{family}_valid_value_count"] = sum(
                bool(row.get("metrics", {}).get(family, {}).get("valid")) for row in subject_rows
            )
        participant_rows.append(summary)
    _write_csv(run_dir / "participant_summary.csv", participant_rows)

    metric_counts: dict[str, dict[str, int]] = {}
    for family in ("ED", "OI", "NIE"):
        metric_counts[family] = {
            "valid": sum(bool(row.get("metrics", {}).get(family, {}).get("valid")) for row in rows),
            "unsupported_or_invalid": sum(not bool(row.get("metrics", {}).get(family, {}).get("valid")) for row in rows),
        }
    outputs = {
        "window_results.csv": sha256_file(run_dir / "window_results.csv"),
        "participant_summary.csv": sha256_file(run_dir / "participant_summary.csv"),
        **{
            f"segments/{path.name}": sha256_file(path)
            for path in segment_paths
        },
    }
    manifest = {
        "schema_id": MANIFEST_SCHEMA_ID,
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "mode": mode,
        "status": "complete",
        "source_manifest_sha256": selection.get("source_manifest_sha256"),
        "selection_sha256": sha256_file(OUTPUT_ROOT / "selection.json"),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "code_sha256": _code_hashes(),
        "config_sha256": _config_hashes(),
        "runtime_dependencies": _runtime_dependencies(),
        "selected_subject_ids_in_draw_order": selection.get("selected_subject_ids_in_draw_order", []),
        "smoke_subject_id": selection.get("smoke_subject_id"),
        "target_interval_sec_from_rosc_half_open": [0, TARGET_END_SEC],
        "window_seconds": WINDOW_SECONDS,
        "window_alignment": "[20k,20(k+1)) seconds from ROSC",
        "segment_count_processed_or_retained": len(segments),
        "window_realization_row_count": len(rows),
        "supported_window_realization_rows": sum(row.get("support_status") != "unsupported" for row in rows),
        "metric_counts": metric_counts,
        "outcome_fields_read": False,
        "qualified": False,
        "output_file_sha256": outputs,
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def _run_smoke(
    manifest: Mapping[str, Any],
    selection: Mapping[str, Any],
    source_root: Path,
    resolved_config: Mapping[str, Any],
    run_id: str,
) -> tuple[Path, dict[str, Any]]:
    run_dir = OUTPUT_ROOT / "smoke" / run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing_to_overwrite_smoke_run:{run_dir}")
    run_dir.mkdir(parents=True)
    records = _selected_records(manifest, selection)
    smoke_subject = str(selection["smoke_subject_id"])
    smoke_path = str(selection["smoke_segment_relative_path"])
    record = next(
        row for row in records[smoke_subject]
        if row.get("header_relative_path") == smoke_path
    )
    segment = _process_segment(record, source_root, resolved_config, window_limit=SMOKE_WINDOW_COUNT)
    segment_name = f"{smoke_subject}_{Path(smoke_path).stem}.json"
    _write_json(run_dir / "segments" / segment_name, segment)
    _aggregate_run(run_dir, selection, mode="smoke_first_180s", run_id=run_id)
    rows = segment.get("rows", [])
    native_rows = [row for row in rows if row.get("realization_id") == "native_eeg_channels.v1"]
    valid_families = {
        family: any(bool(row.get("metrics", {}).get(family, {}).get("valid")) for row in native_rows)
        for family in ("ED", "OI", "NIE")
    }
    smoke_pass = bool(
        segment.get("processing_status") == "processed"
        and len(native_rows) == SMOKE_WINDOW_COUNT
        and all(valid_families.values())
        and segment.get("source_hash_audit", {}).get("source_sha256_stable_after_read") is True
    )
    smoke_summary = {
        "schema_id": "mndm.mnps_v3_2.coma_vector_preflight_002r1_smoke_summary.v1",
        "run_id": run_id,
        "status": "PASS" if smoke_pass else "FAIL",
        "smoke_subject_id": smoke_subject,
        "segment_id": smoke_path,
        "target_window_count": len(native_rows),
        "valid_native_metric_counts": {
            family: sum(bool(row.get("metrics", {}).get(family, {}).get("valid")) for row in native_rows)
            for family in ("ED", "OI", "NIE")
        },
        "valid_family_presence": valid_families,
        "source_hash_stable": segment.get("source_hash_audit", {}).get("source_sha256_stable_after_read"),
        "support_statuses": sorted({str(row.get("support_status")) for row in rows}),
        "reason": None if smoke_pass else segment.get("reason") or segment.get("support_reason") or "smoke_contract_not_met",
        "outcome_fields_read": False,
        "qualified": False,
    }
    _write_json(run_dir / "smoke_summary.json", smoke_summary)
    (OUTPUT_ROOT / "smoke_latest.json").write_text(
        json.dumps(smoke_summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    run_manifest = _read_json(run_dir / "run_manifest.json")
    run_manifest["status"] = "complete" if smoke_pass else "smoke_failed"
    run_manifest["engineering_status"] = smoke_summary["status"]
    run_manifest["smoke_summary_sha256"] = sha256_file(run_dir / "smoke_summary.json")
    _write_json(run_dir / "run_manifest.json", run_manifest)
    return run_dir, smoke_summary


def _run_cohort(
    manifest: Mapping[str, Any],
    selection: Mapping[str, Any],
    source_root: Path,
    resolved_config: Mapping[str, Any],
    run_id: str,
) -> tuple[Path, dict[str, Any]]:
    smoke_path = OUTPUT_ROOT / "smoke_latest.json"
    if not smoke_path.is_file():
        raise RuntimeError("cohort_refused_smoke_test_missing")
    smoke_summary = _read_json(smoke_path)
    if smoke_summary.get("status") != "PASS":
        raise RuntimeError("cohort_refused_smoke_test_not_passed")
    run_dir = OUTPUT_ROOT / "cohort" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    records = _selected_records(manifest, selection)
    segment_dir = run_dir / "segments"
    segment_dir.mkdir()
    for subject_id in selection.get("selected_subject_ids_in_draw_order", []):
        subject_records = records.get(str(subject_id), [])
        target_records = [row for row in subject_records if _window_specs(row)]
        for record in target_records:
            segment_name = f"{subject_id}_{Path(str(record['header_relative_path'])).stem}.json"
            path = segment_dir / segment_name
            if path.exists():
                raise FileExistsError(f"refusing_to_overwrite_segment_output:{path}")
            segment = _process_segment(record, source_root, resolved_config)
            _write_json(path, segment)
            print(
                f"segment_processed={record['header_relative_path']} "
                f"status={segment.get('processing_status')} windows={segment.get('window_count_planned')}",
                flush=True,
            )
        print(f"completed_subject={subject_id} target_segments={len(target_records)}", flush=True)
    run_manifest = _aggregate_run(run_dir, selection, mode="first_24h", run_id=run_id)
    return run_dir, run_manifest


def run_all(source_root: Path, *, selection_count: int = 16) -> tuple[Path, Path | None]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest, selection = _freeze_inputs(source_root, selection_count=selection_count)
    manifest, selection = _verify_frozen_inputs(source_root)
    if len(selection.get("selected_subject_ids_in_draw_order", [])) < 12:
        raise RuntimeError("fewer_than_12_selected_patients")
    resolved_config = _load_resolved_config()
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_coma_vector_preflight002r1")
    smoke_dir, smoke_summary = _run_smoke(manifest, selection, source_root, resolved_config, run_id + "_smoke")
    if smoke_summary.get("status") != "PASS":
        return smoke_dir, None
    cohort_dir, _ = _run_cohort(manifest, selection, source_root, resolved_config, run_id + "_first24h")
    return smoke_dir, cohort_dir


def run_smoke_only(source_root: Path, *, selection_count: int = 16) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest, selection = _freeze_inputs(source_root, selection_count=selection_count)
    manifest, selection = _verify_frozen_inputs(source_root)
    resolved_config = _load_resolved_config()
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_coma_vector_preflight002r1_smoke")
    run_dir, _ = _run_smoke(manifest, selection, source_root, resolved_config, run_id)
    return run_dir


def run_cohort_only(source_root: Path, *, run_id: str | None = None) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest, selection = _verify_frozen_inputs(source_root)
    resolved_config = _load_resolved_config()
    run_id = run_id or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_coma_vector_preflight002r1_first24h")
    run_dir, _ = _run_cohort(manifest, selection, source_root, resolved_config, run_id)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--mode", choices=("all", "smoke", "cohort"), default="all")
    parser.add_argument("--selection-count", type=int, default=16)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    if not 12 <= args.selection_count <= 20:
        raise SystemExit("--selection-count must be in [12, 20]")
    if args.mode == "all":
        smoke_dir, cohort_dir = run_all(args.source_root, selection_count=args.selection_count)
        print(f"smoke_output={smoke_dir}", flush=True)
        if cohort_dir is None:
            print("cohort_status=NOT_RUN_SMOKE_FAILED", flush=True)
            raise SystemExit(2)
        print(f"cohort_output={cohort_dir}", flush=True)
    elif args.mode == "smoke":
        print(f"smoke_output={run_smoke_only(args.source_root, selection_count=args.selection_count)}", flush=True)
    else:
        print(f"cohort_output={run_cohort_only(args.source_root, run_id=args.run_id)}", flush=True)


if __name__ == "__main__":
    main()
