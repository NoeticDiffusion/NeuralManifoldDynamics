"""Outcome-blind SL-014 psychosis dual-scale experiment adapter.

This is an isolated research runner for the SL-014 packet. It does not change
the production feature pipeline or read clinical outcome tables. The adapter
maps the two local BIDS BrainVision sources to explicit v3.2 matrix kernels,
retains source support and realization provenance, and writes only beneath an
operator-supplied output root.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import signal
from scipy.signal import hilbert

from mndm.mnps.v3_2.ar import compute_ar_activity_transition_vector
from mndm.mnps.v3_2.ed import compute_ed_raw_sensor_covariance_participation_ratio
from mndm.mnps.v3_2.nie import compute_nie_raw_weighted_global_efficiency
from mndm.mnps.v3_2.oi import compute_oi_matrix
from mndm.mnps.v3_2.pcv import compute_pcv_phase_order_parameter_sd
from mndm.mnps.v3_2.rd import compute_rd_centered_log_psd_dispersion
from mndm.mnps.v3_2.sc import compute_sc_lz78_exact_multiset_ratio
from mndm.mnps.v3_2.tp import (
    compute_tp_log_rms_envelope_persistence,
    compute_tp_raw_signal_persistence,
)


PACKET_ID = "SL014-MNPS32-PSYCHOSIS-DUALSCALE-20261015-v1"
DATA_ROOT = Path(r"K:\ExternalReceivedDatasets\openneuro")
DATASETS = ("ds003944", "ds003947")
EXPECTED_CHANNEL_COUNT = 64
EXPECTED_EEG_COUNT = 61
TARGET_FS_HZ = 1000.0
WINDOWS_SECONDS = (4, 20)
AR_BLOCK_SECONDS = 2
COMMON_SCALP_LABELS: tuple[str, ...] = ()  # resolved from BIDS metadata at P00
AVERAGE_REFERENCE_ID = "average_over_native_BIDS_EEG_61_including_M2.v1"
SOURCE_REFERENCE_ID = "recorded_TP9_as_declared_by_BIDS.v1"
PREPROCESS_ID = "native_source_or_resample_3to1_Kaiser5_FIR61_target1000Hz.v1"
PCV_PHASE_ID = "scipy.signal.hilbert_after_sosfiltfilt.v1"
PCV_TP_BAND_ID = "butterworth4_zero_phase_8_13hz_per_window.v1"
AR_VECTOR_ID = "per_channel_rms_contiguous_2s_blocks.v1"
AR_SCALING_ID = "raw_numeric_amplitude_rms.v1"
NIE_EDGE_ESTIMATOR_ID = "absolute_zero_lag_Pearson_r"
NIE_EDGE_RULE_ID = "no_threshold_exact_zero_diagonal.v1"
SC_REFERENCE_COUNT = 32
SC_MASTER_SEED = 310930
OI_ORDER = 5
OI_DELAY_SAMPLES = 1
TP_ENVELOPE_SECONDS = 1.0
TP_LAG_SECONDS = 1.0
SOURCE_REF_KIND = "source_reference"
AVERAGE_REF_KIND = "average_ref_sensitivity"
CORE_FAMILIES = ("PCV", "TP", "NIE", "RD", "OI", "SC", "ED", "AR")


class PreflightError(RuntimeError):
    pass


class BudgetStop(RuntimeError):
    pass


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hash_labels(labels: Sequence[str], *, ordered: bool = True) -> str:
    values = list(labels) if ordered else sorted(labels)
    return _hash_json(values)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _header_fields(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")

    def required(name: str) -> str:
        match = re.search(rf"(?m)^{re.escape(name)}=(.+)$", text)
        if match is None:
            raise PreflightError(f"brainvision_header_missing_{name}:{path.name}")
        return match.group(1).strip()

    data_name = required("DataFile")
    if Path(data_name).is_absolute():
        data_path = Path(data_name)
    else:
        data_path = path.parent / data_name
    try:
        sampling_interval_us = float(required("SamplingInterval"))
        sample_count = int(required("DataPoints"))
        channel_count = int(required("NumberOfChannels"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise PreflightError(f"brainvision_header_numeric_field_invalid:{path.name}") from exc
    data_format = required("DataFormat").upper()
    orientation = required("DataOrientation").upper()
    binary_format = required("BinaryFormat").upper()
    channels = []
    for line in text.splitlines():
        match = re.match(r"^Ch(\d+)=(.*)$", line)
        if match:
            channels.append((int(match.group(1)), match.group(2).split(",", 1)[0].strip()))
    channels.sort()
    if sampling_interval_us <= 0 or sample_count <= 0 or channel_count <= 0:
        raise PreflightError(f"brainvision_header_nonpositive_shape_or_rate:{path.name}")
    if data_format != "BINARY" or orientation != "VECTORIZED" or binary_format != "IEEE_FLOAT_32":
        raise PreflightError(f"unsupported_brainvision_storage_contract:{path.name}")
    if len(channels) != channel_count:
        raise PreflightError(f"brainvision_channel_header_count_mismatch:{path.name}")
    if not data_path.is_file():
        raise PreflightError(f"brainvision_payload_missing:{data_path}")
    expected_bytes = channel_count * sample_count * 4
    if data_path.stat().st_size != expected_bytes:
        raise PreflightError(f"brainvision_payload_size_mismatch:{path.name}")
    return {
        "header_path": str(path), "data_path": str(data_path),
        "sampling_interval_us": sampling_interval_us,
        "sample_rate_hz": 1_000_000.0 / sampling_interval_us,
        "sample_count": sample_count, "channel_count": channel_count,
        "data_format": data_format, "orientation": orientation,
        "binary_format": binary_format, "header_channel_names": [x[1] for x in channels],
        "payload_bytes": expected_bytes,
    }


def _subject_paths(dataset_root: Path, participant_id: str) -> dict[str, Path]:
    eeg_dir = dataset_root / participant_id / "eeg"
    headers = sorted(eeg_dir.glob("*.vhdr"))
    jsons = sorted(eeg_dir.glob("*_eeg.json"))
    channels = sorted(eeg_dir.glob("*_channels.tsv"))
    if not (len(headers) == len(jsons) == len(channels) == 1):
        raise PreflightError(f"expected_one_BrainVision_record_and_sidecars:{participant_id}")
    return {"vhdr": headers[0], "json": jsons[0], "channels": channels[0]}


def _subject_metadata(dataset_root: Path, participant_id: str) -> dict[str, Any]:
    paths = _subject_paths(dataset_root, participant_id)
    header = _header_fields(paths["vhdr"])
    sidecar = _read_json(paths["json"])
    rows = _read_tsv(paths["channels"])
    eeg_rows = [row for row in rows if str(row.get("type", "")).strip().upper() == "EEG"]
    sfreq_declared = float(sidecar["SamplingFrequency"])
    metadata_duration = float(sidecar["RecordingDuration"])
    duration_from_header = header["sample_count"] / header["sample_rate_hz"]
    if len(rows) != header["channel_count"]:
        raise PreflightError(f"bids_sidecar_row_count_mismatch:{participant_id}")
    if len(eeg_rows) != int(sidecar.get("EEGChannelCount", -1)):
        raise PreflightError(f"bids_eeg_channel_count_mismatch:{participant_id}")
    if abs(sfreq_declared - header["sample_rate_hz"]) > max(1e-3, header["sample_rate_hz"] * 1e-6):
        raise PreflightError(f"bids_sampling_frequency_header_mismatch:{participant_id}")
    if abs(metadata_duration - duration_from_header) > 0.01:
        raise PreflightError(f"bids_recording_duration_header_mismatch:{participant_id}")
    raw_data_path = Path(header["data_path"])
    return {
        "participant_id": participant_id,
        "recording_id": paths["vhdr"].stem,
        "dataset_id": dataset_root.name,
        **header,
        "sidecar_path": str(paths["json"]),
        "channels_path": str(paths["channels"]),
        "sidecar_sampling_frequency_hz": sfreq_declared,
        "sidecar_recording_duration_s": metadata_duration,
        "eeg_reference": str(sidecar.get("EEGReference", "unknown")),
        "instructions": str(sidecar.get("Instructions", "unspecified")),
        "task_description": str(sidecar.get("TaskDescription", "unspecified")),
        "recording_type": str(sidecar.get("RecordingType", "unknown")),
        "eeg_labels": [str(row.get("name", "")).strip() for row in eeg_rows],
        "eeg_channel_rows": [i for i, row in enumerate(rows)
                              if str(row.get("type", "")).strip().upper() == "EEG"],
        "channel_rows": rows,
        "metadata_qc": [
            "brainvision_header_names_generic_EEG001_order_mapped_by_BIDS_channels_tsv_row_order",
            "channel_resolution_blank_physical_amplitude_scale_not_independently_verified",
        ],
        "source_files": {
            "data": str(raw_data_path), "vhdr": str(paths["vhdr"]),
            "json": str(paths["json"]), "channels_tsv": str(paths["channels"]),
        },
    }


def _participant_ids(dataset_root: Path) -> list[str]:
    return sorted(path.name for path in dataset_root.glob("sub-*") if path.is_dir())


def _selection_digest(dataset_id: str, participant_id: str) -> str:
    return hashlib.sha256(f"SL014-SMOKE|{dataset_id}|{participant_id}".encode("utf-8")).hexdigest()


def _dataset_scan(dataset_root: Path) -> dict[str, Any]:
    ids = _participant_ids(dataset_root)
    if not ids:
        raise PreflightError(f"no_BIDS_subject_directories:{dataset_root}")
    records = [_subject_metadata(dataset_root, pid) for pid in ids]
    first_labels = tuple(label.upper() for label in records[0]["eeg_labels"])
    first_order = tuple((str(row.get("name", "")).upper(), str(row.get("type", "")).upper(),
                         str(row.get("units", "")).casefold()) for row in records[0]["channel_rows"])
    order_mismatches = []
    instruction_variants = set()
    reference_values = set()
    rates: dict[str, int] = {}
    duration_values: list[float] = []
    payload_bytes = 0
    for row in records:
        labels = tuple(label.upper() for label in row["eeg_labels"])
        order = tuple((str(item.get("name", "")).upper(), str(item.get("type", "")).upper(),
                       str(item.get("units", "")).casefold()) for item in row["channel_rows"])
        if labels != first_labels or order != first_order:
            order_mismatches.append(row["participant_id"])
        instruction_variants.add(row["instructions"])
        reference_values.add(row["eeg_reference"])
        rate_key = str(round(row["sample_rate_hz"], 3))
        rates[rate_key] = rates.get(rate_key, 0) + 1
        duration_values.append(row["sample_count"] / row["sample_rate_hz"])
        payload_bytes += int(row["payload_bytes"])
    if order_mismatches:
        raise PreflightError(f"within_dataset_channel_order_changed:{dataset_root.name}:{order_mismatches[:5]}")
    if len(instruction_variants) != 1 or len(reference_values) != 1:
        raise PreflightError(f"within_dataset_acquisition_metadata_varies:{dataset_root.name}")
    smoke = sorted(ids, key=lambda pid: (_selection_digest(dataset_root.name, pid), pid))[:4]
    return {
        "dataset_id": dataset_root.name,
        "dataset_root": str(dataset_root),
        "participant_ids": ids,
        "participant_count": len(ids),
        "source_roster_sha256": _hash_json(ids),
        "smoke_participant_ids": smoke,
        "smoke_selection_hashes": {pid: _selection_digest(dataset_root.name, pid) for pid in smoke},
        "source_rate_counts_hz_rounded_3dp": rates,
        "source_reference_values": sorted(reference_values),
        "instruction_variants": sorted(instruction_variants),
        "duration_min_s": min(duration_values),
        "duration_max_s": max(duration_values),
        "eeg_channel_count": len(first_labels),
        "eeg_labels": list(first_labels),
        "channel_rows": records[0]["channel_rows"],
        "records": records,
        "raw_payload_bytes": payload_bytes,
        "events_tsv_count": len(list(dataset_root.rglob("*_events.tsv"))),
        "electrodes_tsv_count": len(list(dataset_root.rglob("*_electrodes.tsv"))),
        "vmrk_count": len(list(dataset_root.rglob("*.vmrk"))),
    }


def _repo_facts(repo_root: Path) -> dict[str, Any]:
    import subprocess

    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=repo_root, capture_output=True,
                                text=True, check=True)
        return result.stdout.strip()

    dirty = git("status", "--short").splitlines()
    return {
        "root": str(repo_root.resolve()), "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"), "dirty_entry_count": len(dirty),
        "dirty_entries_sample": dirty[:50],
    }


def _analysis_repo_facts(repo_root: Path) -> dict[str, Any]:
    import subprocess

    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=repo_root, capture_output=True,
                                text=True, check=True)
        return result.stdout.strip()

    return {"root": str(repo_root.resolve()), "branch": git("branch", "--show-current"),
            "commit": git("rev-parse", "HEAD"),
            "dirty_entry_count": len(git("status", "--short").splitlines())}


def _source_hashes(repo_root: Path, analysis_root: Path) -> dict[str, str]:
    paths = [repo_root / "mndm/src/mndm/preprocess.py"]
    paths.extend(repo_root / f"mndm/src/mndm/mnps/v3_2/{name}.py" for name in (
        "pcv", "tp", "nie", "rd", "oi", "sc", "ed", "ar", "candidate_grid",
        "realizations", "support", "preprocessing"))
    paths.append(repo_root / "mndm/src/mndm/mnps/v3_2/experiments/full_vector_challenge_001.py")
    paths.extend(analysis_root / f"nmd-analysis/config/{name}" for name in (
        "ndt-analysis-ds003944.yaml", "ndt-analysis-ds003947.yaml"))
    paths.append(Path(__file__).resolve())
    return {str(path): _sha256_file(path) for path in paths if path.is_file()}


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in ("numpy", "scipy", "mne", "numba", "wfdb"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2,
                               sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(_jsonable(value), ensure_ascii=False)
                             if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def _write_gzip_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with gzip.open(path, "wt", encoding="utf-8", newline="\n", compresslevel=5) as stream:
        for row in rows:
            stream.write(json.dumps(_jsonable(row), ensure_ascii=False,
                                   sort_keys=True, allow_nan=False) + "\n")
            count += 1
    return count


def _common_sensor_set(scans: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    sets = [set(str(label).upper() for label in scans[dataset]["eeg_labels"])
            for dataset in DATASETS]
    # M2 is a mastoid/reference-area sensor, not a scalp channel; it remains in
    # the native average-reference set but is excluded from the shared scalp
    # measurement realization.
    return tuple(sorted((sets[0] & sets[1]) - {"M2"}))


def _freeze_projection(protocol: Mapping[str, Any]) -> str:
    projection = dict(protocol)
    projection.pop("frozen_protocol_sha256", None)
    return _hash_json(projection)


def preflight(
    *, output_root: Path, repo_root: Path, analysis_root: Path,
    data_root: Path = DATA_ROOT, packet_root: Path | None = None,
) -> Path:
    """Audit raw metadata, freeze the resolved protocol, and record P00."""
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise PreflightError(f"refusing_nonempty_output_root:{output_root}")
    usage = shutil.disk_usage(output_root.parent if output_root.parent.exists() else output_root.anchor)
    if usage.free < 5 * 1024**3:
        raise PreflightError(f"initial_free_space_below_5_GiB:{usage.free}")

    packet_root = packet_root or (
        repo_root / "project/new_mnps_v.3.2/science_lead/"
        "MNPS_v3_2_SL014_Psychosis_DualScale_20261015/"
        "MNPS_v3_2_SL014_Psychosis_DualScale_20261015"
    )
    packet_commands = {
        "packet_selfcheck": [sys.executable, "tools/packet_selfcheck.py"],
        "packet_unittest": [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
    }
    packet_results: dict[str, subprocess.CompletedProcess[str]] = {}
    for key, command in packet_commands.items():
        result = subprocess.run(command, cwd=packet_root, capture_output=True,
                                text=True, encoding="utf-8", errors="replace",
                                timeout=600, check=False)
        packet_results[key] = result
        if result.returncode != 0:
            raise PreflightError(
                f"{key}_failed_exit_{result.returncode}:\n{result.stdout}\n{result.stderr}"
            )

    output_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_root)

    scans = {dataset: _dataset_scan(data_root / dataset) for dataset in DATASETS}
    common_labels = _common_sensor_set(scans)
    if len(common_labels) < 2:
        raise PreflightError("shared_scalp_realization_has_fewer_than_two_channels")

    # Complete file hashes are recorded before any sample-level feature work.
    inventory: list[dict[str, Any]] = []
    total_raw = sum(int(scans[ds]["raw_payload_bytes"]) for ds in DATASETS)
    processed_bytes = 0
    for dataset in DATASETS:
        dataset_root = data_root / dataset
        for relpath, role in (("dataset_description.json", "dataset_description"),
                              ("README", "dataset_readme")):
            path = dataset_root / relpath
            if path.is_file():
                inventory.append({
                    "dataset_id": dataset, "participant_id": None,
                    "recording_id": None, "role": role,
                    "path": str(path), "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                })
        for record in scans[dataset]["records"]:
            for role, filename in record["source_files"].items():
                path = Path(filename)
                size = path.stat().st_size
                digest = _sha256_file(path)
                processed_bytes += size if role == "data" else 0
                inventory.append({
                    "dataset_id": dataset, "participant_id": record["participant_id"],
                    "recording_id": record["recording_id"], "role": role,
                    "path": str(path), "size_bytes": size, "sha256": digest,
                    "sample_rate_hz": record["sample_rate_hz"],
                    "sample_count": record["sample_count"],
                    "channel_count": record["channel_count"],
                    "eeg_channel_count": len(record["eeg_labels"]),
                    "recording_duration_s": record["sample_count"] / record["sample_rate_hz"],
                })
            if processed_bytes and processed_bytes % (2 * 1024**3) < int(record["payload_bytes"]):
                print(f"P00 hashed {processed_bytes / 1024**3:.2f}/{total_raw / 1024**3:.2f} GiB", flush=True)

    primary_scan = scans[DATASETS[0]]
    if any(scan["events_tsv_count"] for scan in scans.values()):
        event_inventory = "BIDS events TSV files present; see source inventory"
    else:
        event_inventory = "no BIDS events TSV or BrainVision marker files present"
    protocol: dict[str, Any] = {
        "packet_id": PACKET_ID,
        "status": "FROZEN_BEFORE_FEATURE_EXTRACTION",
        "current_repo": _repo_facts(repo_root),
        "analysis_repo": _analysis_repo_facts(analysis_root),
        "python_packages": _package_versions(),
        "source_hashes": _source_hashes(repo_root, analysis_root),
        "datasets": {
            dataset: {k: v for k, v in scan.items() if k not in ("records", "channel_rows")}
            for dataset, scan in scans.items()
        },
        "raw_source_rosters_sha256": {dataset: scans[dataset]["source_roster_sha256"] for dataset in DATASETS},
        "smoke_selection": {dataset: scans[dataset]["smoke_participant_ids"] for dataset in DATASETS},
        "smoke_selection_hashes": {dataset: scans[dataset]["smoke_selection_hashes"] for dataset in DATASETS},
        "channel_realization_policy": {
            "primary_id": "study_common_scalp_casefold_sorted_49.v1",
            "primary_labels_canonical_upper": list(common_labels),
            "primary_channel_count": len(common_labels),
            "excluded_from_shared_scalp": ["M2"],
            "source_channel_map": "BIDS channels.tsv row order mapped to generic BrainVision EEG001..EEG064; order mapping is a recorded assumption because header names are generic",
            "native_sets": "not pooled; not computed in this adapter run",
            "positions": "unavailable; SCG unsupported",
        },
        "primary_reference_id": SOURCE_REFERENCE_ID,
        "sensitivity_reference_id": AVERAGE_REFERENCE_ID,
        "reference_policy": {
            "primary": "BIDS EEGReference=TP9; use stored reference without additional transform",
            "sensitivity": "subtract instantaneous mean across all 61 BIDS-typed EEG channels including M2, then select the shared 49 scalp labels",
        },
        "preprocessing_id": PREPROCESS_ID,
        "sampling_contract": {
            "target_sample_rate_hz": TARGET_FS_HZ,
            "native_1000_hz": "unchanged",
            "native_3000_hz": "scipy.signal.resample_poly(up=1, down=3), explicit 61-tap Kaiser-5 FIR, constant edge padding",
            "effective_resampling_support": "30 native samples on each side (10 ms at 3 kHz); end padding is constant",
            "window_grid": "half-open integer samples on the 1000 Hz target grid",
            "source_rate_rounding": "preserve exact BrainVision SamplingInterval and BIDS SamplingFrequency in provenance; 3000.0003 Hz is nominally treated as 3000 Hz for integer 3:1 resampling (0.1 ppm metadata round-off)",
        },
        "filter_dependency_left_right_samples": "all samples within the current window for alpha PCV/TP because Butterworth SOS filtfilt and Hilbert are noncausal; raw families depend on their declared window plus the resampling FIR margin",
        "causal_prefix_invariance_pass": False,
        "eye_state_or_task_condition": "eyes open per identical BIDS Instructions and TaskDescription in each source; no events/markers",
        "source_continuity_operational_qc_contract": "one continuous BrainVision file and exact binary/header sizes; sample QC checks finite/flatline/step summaries; no sample-clock or gap annotation exists, therefore physical continuity remains unknown",
        "frequency_bands_and_sampling_contract": {
            "PCV_TP_alpha_hz": [8.0, 13.0],
            "RD_welch_hz": [1.0, 45.0],
            "line_notch": "none; 50 Hz is outside RD band and no notch is applied",
        },
        "family_estimator_ids_at_4_and_20_seconds": {
            "PCV": "PCV.sensor_phase_order_parameter_population_sd.v1",
            "TP": "TP.alpha_log_rms_envelope_pearson_persistence.v1",
            "NIE": "D1.sensor_weighted_global_efficiency_invweight.v0",
            "RD": "RD.pipeline_native_window_centered_log_psd_dispersion.v1",
            "OI": "OI.matrix_input_channel_median_normalized_PE5_delay1.v1",
            "SC": "SC.pipeline_native_window_lz78_exact_multiset_ratio.v1",
            "ED": "ED.sensor_covariance_participation_ratio.v1",
            "AR": "AR.sensor_activity_adjacent_pearson_change_vector.v1",
        },
        "family_parameters": {
            "PCV": {"phase": "Hilbert after order-4 zero-phase SOS Butterworth 8-13 Hz per window", "aggregation": "population SD ddof=0"},
            "TP": {"envelope": "trailing RMS of 1.0 s then log", "lag_seconds": 1.0, "epsilon_amplitude": 1e-12, "minimum_pair_count": 20},
            "NIE": {"edge": "absolute zero-lag Pearson r", "threshold": "none", "zero_diagonal": True},
            "RD": {"welch_segment_seconds": 4.0, "window": "Hann", "overlap": 0.5, "detrend": "constant", "density": True},
            "OI": {"order": OI_ORDER, "delay_samples": OI_DELAY_SAMPLES, "ties": "stable original index order"},
            "SC": {"reference_count": SC_REFERENCE_COUNT, "master_seed": SC_MASTER_SEED, "symbolization": "strictly greater than arithmetic channel median"},
            "ED": {"estimator": "raw sensor covariance participation ratio"},
            "AR": {"activity_vector": AR_VECTOR_ID, "scaling": AR_SCALING_ID, "block_seconds": AR_BLOCK_SECONDS, "components_4s": 1, "components_20s": 9},
        },
        "windows": {
            "primary_window_seconds": [4, 20],
            "20s_anchors": "nonoverlapping from sample 0; retain each complete span",
            "4s_children": "five nonoverlapping children exactly partition each 20s anchor",
            "extra_4s": "not included in this smoke; any later edge windows must use record_class=unmatched_4s",
            "record_class": "matched",
        },
        "p01": {
            "noise_fractions_of_channel_rms": [0.001, 0.01],
            "independent_seeds_per_window": 8,
            "sensitivity_window": "first 4s child of first complete 20s anchor for each smoke participant",
            "cadence_iqr": "OI baseline across all complete 4s cadence windows from the same source segment",
        },
        "p03_synthetic_noise_grid": {
            "process_sigma": [0.0, 0.1, 0.3], "observation_sigma": [0.0, 0.05, 0.1],
            "replicates_per_cell": 20, "seed_start": 2026101500,
            "train_states": 500, "test_successor_transitions": 199,
        },
        "source_outcome_metadata_sealed": True,
        "clinical_covariate_missingness_policy": {
            "cpz_at_scan_positive": "medicated", "zero": "unmedicated", "missing": "unknown",
            "age_sex": "retain missing explicitly; no silent imputation before P04",
        },
        "analysis_model_hierarchy": ["Q0", "Q1", "Q2", "Q3", "Q4", "Q5_conditional"],
        "output_root": str(output_root),
        "scientifically_qualified": False,
        "clinical_outcomes_read": False,
        "historical_manuscript_read_as_background": True,
        "historical_findings_are_not_calibration_targets": True,
        "packet_selfcheck_stdout_sha256": hashlib.sha256(
            packet_results["packet_selfcheck"].stdout.encode("utf-8")
        ).hexdigest(),
        "packet_unittest_stdout_sha256": hashlib.sha256(
            packet_results["packet_unittest"].stdout.encode("utf-8")
        ).hexdigest(),
        "packet_checksum_manifest_sha256": _sha256_file(packet_root / "SHA256SUMS.txt"),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["frozen_protocol_sha256"] = _freeze_projection(protocol)

    _write_csv(output_root / "input_inventory.csv", inventory)
    for key, result in packet_results.items():
        (output_root / f"{key}.stdout.txt").write_text(result.stdout, encoding="utf-8")
        (output_root / f"{key}.stderr.txt").write_text(result.stderr, encoding="utf-8")
    _write_json(output_root / "resolved_protocol.json", protocol)
    protocol_file_hash = _sha256_file(output_root / "resolved_protocol.json")
    (output_root / "resolved_protocol.sha256").write_text(protocol_file_hash + "  resolved_protocol.json\n", encoding="utf-8")
    manifest = {
        "packet_id": PACKET_ID, "status": "P00_METADATA_FROZEN",
        "resolved_protocol_projection_sha256": protocol["frozen_protocol_sha256"],
        "resolved_protocol_file_sha256": protocol_file_hash,
        "input_inventory_sha256": _sha256_file(output_root / "input_inventory.csv"),
        "input_files": len(inventory), "input_raw_payload_bytes": total_raw,
        "output_root": str(output_root), "clinical_outcomes_read": False,
    }
    _write_json(output_root / "manifest.json", manifest)

    source_qc = [
        f"# P00 source QC — {PACKET_ID}", "",
        f"- Output root: `{output_root}`",
        f"- Initial free space: {usage.free / 1024**3:.2f} GiB (minimum 5 GiB).",
        f"- Source binary payloads SHA-256 hashed: {len([r for r in inventory if r['role'] == 'data'])} files, {total_raw / 1024**3:.2f} GiB.",
        f"- Shared scalp realization: {len(common_labels)} labels; exact canonical order frozen in `resolved_protocol.json`.",
        f"- Clinical labels/outcomes: not opened. Historical manuscript: read as background, not used for selection or calibration.",
        "- Source sampling is mixed (1 kHz and approximately 3 kHz); all analyses will use the frozen 1 kHz grid with explicit 3:1 anti-alias resampling.",
        "- BrainVision headers use generic EEG channel names; BIDS `channels.tsv` row order supplies the channel mapping and is recorded as an assumption.",
        "- No events TSV, marker file, electrodes TSV, or coordinate system is present. Eye-open state is described in BIDS task instructions.",
        "- Header and payload sizes, source sample counts, BIDS sampling rates, and duration metadata passed the preflight consistency checks.",
        "- P00 source support is sufficient for descriptive window computation. Physical continuity is unknown and blocks P03 empirical forecasting.",
        "- Cross-cohort sensor realization is a candidate based on the 49-label BIDS intersection; channel mapping equivalence remains a documented limitation.",
        "",
    ]
    (output_root / "P00_SOURCE_QC.md").write_text("\n".join(source_qc), encoding="utf-8")
    _write_csv(output_root / "P00_INPUT_INVENTORY.csv", inventory)
    log = [
        f"# Execution log — {PACKET_ID}", "",
        f"- {datetime.now(timezone.utc).isoformat()} P00 metadata preflight completed.",
        f"- Resolved protocol projection SHA-256: `{protocol['frozen_protocol_sha256']}`.",
        f"- Resolved protocol file SHA-256: `{protocol_file_hash}`.",
        f"- Packet selfcheck exit={packet_results['packet_selfcheck'].returncode}; stdout SHA-256 `{protocol['packet_selfcheck_stdout_sha256']}`.",
        f"- Packet unittest exit={packet_results['packet_unittest'].returncode}; stdout SHA-256 `{protocol['packet_unittest_stdout_sha256']}`.",
        f"- Packet checksum-manifest SHA-256 `{protocol['packet_checksum_manifest_sha256']}`.",
        "- No clinical outcome file was opened. No signal samples were extracted during P00.",
        "",
    ]
    (output_root / "execution_log.md").write_text("\n".join(log), encoding="utf-8")
    _update_gate_files(output_root)
    return output_root


def _frozen_protocol(output_root: Path) -> dict[str, Any]:
    path = output_root / "resolved_protocol.json"
    if not path.is_file():
        raise PreflightError("resolved_protocol_missing_run_P00_first")
    protocol = _read_json(path)
    expected = str(protocol.get("frozen_protocol_sha256", ""))
    if expected != _freeze_projection(protocol):
        raise PreflightError("resolved_protocol_projection_hash_mismatch")
    digest_path = output_root / "resolved_protocol.sha256"
    if not digest_path.is_file() or digest_path.read_text(encoding="utf-8").split()[0] != _sha256_file(path):
        raise PreflightError("resolved_protocol_file_hash_mismatch")
    return protocol


def _inventory_hash_lookup(output_root: Path) -> dict[str, str]:
    with (output_root / "P00_INPUT_INVENTORY.csv").open("r", encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream)
        return {str(row["path"]): str(row["sha256"]) for row in rows}


def _frozen_common_labels(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(label) for label in protocol["channel_realization_policy"]["primary_labels_canonical_upper"])


def _verify_record_files(record: Mapping[str, Any], frozen_hashes: Mapping[str, str]) -> str:
    for path_text in record["source_files"].values():
        path = Path(str(path_text))
        expected = frozen_hashes.get(str(path))
        if expected is None:
            raise PreflightError(f"source_path_absent_from_frozen_inventory:{path}")
        if not path.is_file() or _sha256_file(path) != expected:
            raise PreflightError(f"frozen_source_hash_changed:{path}")
    return frozen_hashes[str(record["source_files"]["data"])]


def _load_resampled_eeg(record: Mapping[str, Any]) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    """Read the frozen BIDS EEG rows from a vectorized float32 payload."""
    data_path = Path(str(record["data_path"]))
    n_channels = int(record["channel_count"])
    n_samples = int(record["sample_count"])
    data = np.memmap(data_path, dtype="<f4", mode="r", shape=(n_channels, n_samples), order="C")
    eeg_rows = np.asarray(record["eeg_channel_rows"], dtype=np.int64)
    labels_source = tuple(str(label) for label in record["eeg_labels"])
    labels_upper = tuple(label.upper() for label in labels_source)
    if len(set(labels_upper)) != len(labels_upper):
        raise PreflightError(f"duplicate_casefolded_BIDS_EEG_label:{record['participant_id']}")
    native = np.asarray(data[eeg_rows, :], dtype=np.float64)
    fs_native = float(record["sample_rate_hz"])
    resample_record: dict[str, Any] = {
        "source_sample_rate_hz": fs_native,
        "target_sample_rate_hz": TARGET_FS_HZ,
        "source_sample_count": n_samples,
        "resampling": "identity",
        "fir_taps": 1,
        "source_dependency_margin_samples_each_side": 0,
        "source_dependency_margin_seconds_each_side": 0.0,
    }
    if abs(fs_native - TARGET_FS_HZ) <= 1e-3:
        output = native
    elif 2999.0 <= fs_native <= 3001.0:
        taps = signal.firwin(61, 1.0 / 3.0, window=("kaiser", 5.0))
        output = signal.resample_poly(native, up=1, down=3, axis=-1,
                                      window=taps, padtype="constant")
        resample_record.update({
            "resampling": "scipy.signal.resample_poly_1_to_3_kaiser5_FIR61_constant_pad.v1",
            "fir_taps": 61,
            "source_dependency_margin_samples_each_side": 30,
            "source_dependency_margin_seconds_each_side": 30.0 / fs_native,
            "tap_sha256": hashlib.sha256(np.asarray(taps, dtype="<f8").tobytes()).hexdigest(),
        })
    else:
        raise PreflightError(f"unsupported_source_sampling_rate:{fs_native}")
    output = np.asarray(output, dtype=np.float64)
    if output.ndim != 2 or output.shape[0] != len(labels_upper) or output.shape[1] < 20 * int(TARGET_FS_HZ):
        raise PreflightError(f"resampled_source_shape_invalid:{record['participant_id']}")
    resample_record["target_sample_count"] = int(output.shape[1])
    resample_record["nominal_duration_s"] = output.shape[1] / TARGET_FS_HZ
    return output, labels_upper, resample_record


def _realization_signals(
    native_eeg: np.ndarray, labels: Sequence[str], common_labels: Sequence[str]
) -> dict[str, tuple[np.ndarray, tuple[str, ...], tuple[str, ...], tuple[str, ...]]]:
    normalized_labels = tuple(str(label).upper() for label in labels)
    index = {label: i for i, label in enumerate(normalized_labels)}
    missing = [label for label in common_labels if label not in index]
    if missing:
        raise PreflightError(f"shared_channel_missing_in_participant:{missing}")
    selected = np.asarray([index[label] for label in common_labels], dtype=np.int64)
    source = native_eeg[selected, :]
    average = native_eeg - np.mean(native_eeg, axis=0, keepdims=True, dtype=np.float64)
    average = average[selected, :]
    return {
        SOURCE_REF_KIND: (source, tuple(common_labels), ("BIDS_EEGReference_TP9_declared",), ("TP9",)),
        AVERAGE_REF_KIND: (average, tuple(common_labels), ("reference_mean_over_all_61_native_EEG_rows_including_M2",), normalized_labels),
    }


def _window_hash(signals: np.ndarray, labels: Sequence[str], dataset: str,
                 participant: str, start: int, end: int, reference_id: str) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps({
        "dataset": dataset, "participant": participant, "labels": list(labels),
        "sample_rate_hz": TARGET_FS_HZ, "start": start, "end": end,
        "reference_id": reference_id, "shape": list(signals.shape),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(np.asarray(signals, dtype="<f8", order="C").tobytes(order="C"))
    return digest.hexdigest()


def _compute_nie(signals: np.ndarray, labels: Sequence[str]) -> Any:
    centered = signals - np.mean(signals, axis=1, keepdims=True, dtype=np.float64)
    norms = np.sqrt(np.sum(centered * centered, axis=1, dtype=np.float64))
    if np.any(norms <= 0.0) or not np.isfinite(norms).all():
        weights = np.full((signals.shape[0], signals.shape[0]), np.nan, dtype=np.float64)
    else:
        correlation = (centered @ centered.T) / np.outer(norms, norms)
        correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
        weights = np.abs(correlation)
        np.fill_diagonal(weights, 0.0)
    order_hash = _hash_labels(labels, ordered=True)
    return compute_nie_raw_weighted_global_efficiency(
        weights, labels, order_hash, NIE_EDGE_ESTIMATOR_ID, NIE_EDGE_RULE_ID
    )


def _compute_diagnostics(signals: np.ndarray, labels: Sequence[str]) -> dict[str, Any]:
    centered = signals - np.mean(signals, axis=1, keepdims=True, dtype=np.float64)
    norms = np.sqrt(np.sum(centered * centered, axis=1, dtype=np.float64))
    if np.any(norms <= 0.0) or not np.isfinite(norms).all():
        corr = np.full((signals.shape[0], signals.shape[0]), np.nan)
    else:
        corr = (centered @ centered.T) / np.outer(norms, norms)
        corr = np.clip((corr + corr.T) / 2.0, -1.0, 1.0)
    off_diag = ~np.eye(signals.shape[0], dtype=bool)
    mean_abs_corr = float(np.mean(np.abs(corr[off_diag]))) if np.isfinite(corr[off_diag]).all() else None
    acf = []
    for row in signals:
        left, right = row[:-1], row[1:]
        left = left - np.mean(left)
        right = right - np.mean(right)
        denom = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
        acf.append(float(np.dot(left, right) / denom) if denom > 0 and math.isfinite(denom) else None)
    freqs, psd = signal.welch(signals, fs=TARGET_FS_HZ, nperseg=min(4 * int(TARGET_FS_HZ), signals.shape[1]),
                               noverlap=min(2 * int(TARGET_FS_HZ), signals.shape[1] // 2),
                               axis=-1, window="hann", detrend="constant", scaling="density")
    bands = {"delta_1_4": (1.0, 4.0), "theta_4_8": (4.0, 8.0),
             "alpha_8_13": (8.0, 13.0), "beta_13_30": (13.0, 30.0),
             "gamma_30_45": (30.0, 45.0)}
    band_power: dict[str, float | None] = {}
    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        band_power[name] = float(np.mean(np.trapezoid(psd[:, mask], freqs[mask], axis=1))) if np.any(mask) else None
    positive_psd = np.maximum(psd, 0.0)
    row_sums = positive_psd.sum(axis=1)
    entropy_values = []
    for row, total in zip(positive_psd, row_sums, strict=True):
        if total <= 0.0:
            entropy_values.append(None)
            continue
        probs = row / total
        probs = probs[probs > 0]
        entropy_values.append(float(-np.sum(probs * np.log(probs)) / math.log(max(2, len(row)))))
    return {
        "mean_absolute_pearson_r": mean_abs_corr,
        "acf_lag1_mean": float(np.nanmean([x for x in acf if x is not None])) if any(x is not None for x in acf) else None,
        "acf_lag1_channel_values": acf,
        "mean_psd_band_power_source_units_squared": band_power,
        "normalized_spectral_entropy_mean": float(np.nanmean([x for x in entropy_values if x is not None])) if any(x is not None for x in entropy_values) else None,
        "spectral_entropy_channel_values": entropy_values,
        "spectral_support_hz": [1.0, 45.0],
        "channel_order_sha256": _hash_labels(labels, ordered=True),
    }


def _metric_rows(
    signals: np.ndarray, *, dataset_id: str, participant_id: str,
    recording_id: str, source_sha256: str, labels: Sequence[str],
    reference_id: str, reference_kind: str, start: int, end: int,
    duration_s: int, record_class: str, resample: Mapping[str, Any],
    metadata_qc: Sequence[str], source_sample_count: int | None = None,
    reference_set_labels: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected_samples = duration_s * int(TARGET_FS_HZ)
    if end - start != expected_samples or signals.shape != (len(labels), expected_samples):
        raise ValueError("frozen_window_sample_shape_mismatch")
    window_digest = _window_hash(signals, labels, dataset_id, participant_id,
                                 start, end, reference_id)
    source_margin = int(resample.get("source_dependency_margin_samples_each_side", 0))
    source_stride = 3 if str(resample.get("resampling", "")).startswith("scipy.signal.resample_poly") else 1
    source_start = max(0, start * source_stride - source_margin)
    source_end = end * source_stride + source_margin
    if source_sample_count is not None:
        source_end = min(source_end, int(source_sample_count))
    segment_id = f"{recording_id}:one_brainvision_file"
    base = {
        "packet_id": PACKET_ID,
        "dataset_id": dataset_id,
        "participant_id": participant_id,
        "recording_id": recording_id,
        "segment_id": segment_id,
        "record_class": record_class,
        "reference_id": reference_id,
        "reference_realization_kind": reference_kind,
        "preprocess_id": PREPROCESS_ID,
        "resampling": resample,
        "channel_labels": list(labels),
        "channel_order_sha256": _hash_labels(labels, ordered=True),
        "channel_set_sha256": _hash_labels(labels, ordered=False),
        "channel_count": len(labels),
        "reference_set_sha256": _hash_labels(reference_set_labels, ordered=True) if reference_set_labels else None,
        "source_sha256": source_sha256,
        "window_signal_sha256": window_digest,
        "window_start_sample": start,
        "window_end_sample_exclusive": end,
        "window_start_seconds": start / TARGET_FS_HZ,
        "window_end_seconds": end / TARGET_FS_HZ,
        "duration_s": duration_s,
        "sampling_rate_hz": TARGET_FS_HZ,
        "effective_source_start_sample": source_start,
        "effective_source_end_sample_exclusive": source_end,
        "effective_source_dependency_policy": "whole_window_due_zero_phase_alpha_filter_and_Hilbert; raw family source span expanded by resampling FIR margin",
        "physical_continuity_status": "unknown",
        "source_unit_declaration": "microV_from_channels_tsv; BrainVision float resolution blank, amplitude scale not independently verified",
        "source_qc": list(metadata_qc),
        "computed": True,
        "qualified": False,
        "clinical_outcomes_read": False,
    }
    results: dict[str, Any] = {}
    alpha_sos = signal.butter(4, (8.0, 13.0), btype="bandpass", fs=TARGET_FS_HZ, output="sos")
    try:
        alpha = signal.sosfiltfilt(alpha_sos, signals, axis=-1, padtype=None, padlen=0)
        analytic = hilbert(alpha, axis=-1)
        amplitude = np.abs(analytic)
        if not np.isfinite(amplitude).all():
            raise ValueError("non_finite_analytic_amplitude")
        if np.any(amplitude == 0.0):
            raise ValueError("zero_analytic_amplitude_phase_undefined")
        results["PCV"] = compute_pcv_phase_order_parameter_sd(
            np.angle(analytic), phase_estimator_id=PCV_PHASE_ID,
            band_id=PCV_TP_BAND_ID, channel_labels=labels,
        )
    except Exception as exc:
        results["PCV"] = {"computed": False, "valid": False, "qualified": False,
                           "status": "unsupported", "reason": f"alpha_phase_failed:{type(exc).__name__}:{exc}",
                           "value": None, "estimator_id": "PCV.sensor_phase_order_parameter_population_sd.v1"}
        alpha = None
    if alpha is not None:
        try:
            results["TP"] = compute_tp_log_rms_envelope_persistence(
                alpha, labels, sample_rate_hz=TARGET_FS_HZ,
                envelope_samples=int(TP_ENVELOPE_SECONDS * TARGET_FS_HZ),
                lag_samples=int(TP_LAG_SECONDS * TARGET_FS_HZ),
                epsilon_amplitude=1e-12, minimum_pair_count=20,
            )
        except Exception as exc:
            results["TP"] = {"computed": False, "valid": False, "qualified": False,
                              "status": "unsupported", "reason": f"tp_failed:{type(exc).__name__}:{exc}",
                              "value": None, "estimator_id": "TP.alpha_log_rms_envelope_pearson_persistence.v1"}
    else:
        results["TP"] = {"computed": False, "valid": False, "qualified": False,
                         "status": "unsupported", "reason": "alpha_filter_unavailable",
                         "value": None, "estimator_id": "TP.alpha_log_rms_envelope_pearson_persistence.v1"}
    try:
        results["NIE"] = _compute_nie(signals, labels)
    except Exception as exc:
        results["NIE"] = {"computed": False, "valid": False, "qualified": False,
                          "status": "unsupported", "reason": f"nie_failed:{type(exc).__name__}:{exc}",
                          "value": None, "estimator_id": "D1.sensor_weighted_global_efficiency_invweight.v0"}
    try:
        results["RD"] = compute_rd_centered_log_psd_dispersion(
            signals, labels, sample_rate_hz=TARGET_FS_HZ,
            welch_segment_seconds=4.0, frequency_min_hz=1.0, frequency_max_hz=45.0,
        )
    except Exception as exc:
        results["RD"] = {"computed": False, "valid": False, "qualified": False,
                          "status": "unsupported", "reason": f"rd_failed:{type(exc).__name__}:{exc}",
                          "value": None, "estimator_id": "RD.pipeline_native_window_centered_log_psd_dispersion.v1"}
    try:
        results["OI"] = compute_oi_matrix(signals, labels)
    except Exception as exc:
        results["OI"] = {"computed": False, "valid": False, "qualified": False,
                          "status": "unsupported", "reason": f"oi_failed:{type(exc).__name__}:{exc}",
                          "value": None, "estimator_id": "OI.matrix_input_channel_median_normalized_PE5_delay1.v1"}
    try:
        results["SC"] = compute_sc_lz78_exact_multiset_ratio(
            signals, labels,
            source_identity=f"{dataset_id}|{participant_id}|{recording_id}|{source_sha256}",
            segment_id=segment_id,
            start_seconds=start / TARGET_FS_HZ,
            end_seconds=end / TARGET_FS_HZ,
            reference_count=SC_REFERENCE_COUNT, master_seed=SC_MASTER_SEED,
        )
    except Exception as exc:
        results["SC"] = {"computed": False, "valid": False, "qualified": False,
                          "status": "unsupported", "reason": f"sc_failed:{type(exc).__name__}:{exc}",
                          "value": None, "estimator_id": "SC.pipeline_native_window_lz78_exact_multiset_ratio.v1"}
    try:
        results["ED"] = compute_ed_raw_sensor_covariance_participation_ratio(signals, labels)
    except Exception as exc:
        results["ED"] = {"computed": False, "valid": False, "qualified": False,
                          "status": "unsupported", "reason": f"ed_failed:{type(exc).__name__}:{exc}",
                          "value": None, "estimator_id": "ED.sensor_covariance_participation_ratio.v1"}
    try:
        block_samples = AR_BLOCK_SECONDS * int(TARGET_FS_HZ)
        if expected_samples % block_samples:
            raise ValueError("window_not_divisible_into_complete_2s_AR_blocks")
        block_vectors = np.asarray([
            np.sqrt(np.mean(signals[:, i:i + block_samples] ** 2, axis=1, dtype=np.float64))
            for i in range(0, expected_samples, block_samples)
        ], dtype=np.float64)
        start_sec = start / TARGET_FS_HZ
        intervals = [(start_sec + i * AR_BLOCK_SECONDS, start_sec + (i + 1) * AR_BLOCK_SECONDS)
                     for i in range(block_vectors.shape[0])]
        results["AR"] = compute_ar_activity_transition_vector(
            block_vectors, intervals, vector_construction_id=AR_VECTOR_ID,
            scaling_id=AR_SCALING_ID, channel_labels=labels,
        )
    except Exception as exc:
        results["AR"] = {"computed": False, "valid": False, "qualified": False,
                          "status": "unsupported", "reason": f"ar_failed:{type(exc).__name__}:{exc}",
                          "component_values": [], "component_reasons": [],
                          "estimator_id": "AR.sensor_activity_adjacent_pearson_change_vector.v1"}

    diagnostics = _compute_diagnostics(signals, labels)
    records: list[dict[str, Any]] = []
    normalized: list[dict[str, Any]] = []
    for family in CORE_FAMILIES:
        result = results[family]
        payload = _jsonable(result)
        if not isinstance(payload, Mapping):
            payload = {"value": None, "computed": False, "valid": False,
                       "qualified": False, "status": "unsupported", "reason": "invalid_kernel_result"}
        estimator_id = str(payload.get("estimator_id", "unknown"))
        value = payload.get("value")
        if family == "AR":
            value = None
        record = {
            **base, "family": family, "family_id": family,
            "estimator_id": estimator_id,
            "kernel_result": payload,
            "measurement_value": value,
            "computed": bool(payload.get("computed", False)),
            "valid": bool(payload.get("valid", False)),
            "qualified": False,
            "status": str(payload.get("status", "unsupported")),
            "reason": payload.get("reason"),
            "diagnostics": diagnostics,
        }
        records.append(record)
        if record["valid"]:
            status = "VALID"
        elif not record["computed"] or str(payload.get("status", "")).lower() == "unsupported":
            status = "UNSUPPORTED"
        else:
            status = "INVALID"
        # AR intentionally uses component_values, never a hidden scalar aggregation.
        normalized.append({
            "dataset_id": dataset_id, "participant_id": participant_id,
            "recording_id": recording_id, "segment_id": segment_id,
            "reference_id": reference_id, "preprocess_id": PREPROCESS_ID,
            "channel_set_hash": _hash_labels(labels, ordered=False),
            "channel_order_sha256": _hash_labels(labels, ordered=True),
            "source_sha256": source_sha256,
            "family": family, "estimator_id": estimator_id,
            "record_class": record_class,
            "window_start_sample": start, "window_end_sample": end,
            "sampling_rate_hz": TARGET_FS_HZ, "duration_s": duration_s,
            "computed": record["computed"], "valid": record["valid"],
            "qualified": False, "status": status,
            "reason": record["reason"],
            "value": _jsonable(value),
            "component_values": payload.get("component_values") if family == "AR" else None,
            "component_reasons": payload.get("component_reasons") if family == "AR" else None,
        })
    try:
        raw_tp = compute_tp_raw_signal_persistence(
            signals, labels, sample_rate_hz=TARGET_FS_HZ,
            lag_samples=int(TP_LAG_SECONDS * TARGET_FS_HZ), minimum_pair_count=20,
        )
        diagnostics["raw_tp_comparator"] = _jsonable(raw_tp)
    except Exception as exc:
        diagnostics["raw_tp_comparator"] = {"status": "unsupported", "reason": f"{type(exc).__name__}:{exc}"}
    nie_payload = _jsonable(results["NIE"])
    nie_value = nie_payload.get("value") if isinstance(nie_payload, Mapping) else None
    diagnostics["nie_minus_mean_absolute_pearson_r"] = (
        float(nie_value) - diagnostics["mean_absolute_pearson_r"]
        if nie_value is not None and diagnostics["mean_absolute_pearson_r"] is not None else None
    )
    for record in records:
        record["diagnostics"] = diagnostics
    return records, normalized


def _source_qc(record: Mapping[str, Any]) -> dict[str, Any]:
    """Stream numerical QC without interpreting clinical meaning or clipping."""
    n_channels = int(record["channel_count"])
    n_samples = int(record["sample_count"])
    source_fs = float(record["sample_rate_hz"])
    payload = np.memmap(Path(str(record["data_path"])), dtype="<f4", mode="r",
                        shape=(n_channels, n_samples), order="C")
    nonfinite = 0
    per_channel: list[dict[str, Any]] = []
    for channel_index in range(n_channels):
        values = np.asarray(payload[channel_index], dtype=np.float64)
        finite = np.isfinite(values)
        nonfinite += int(values.size - np.count_nonzero(finite))
        finite_values = values[finite]
        if finite_values.size:
            low = float(np.min(finite_values))
            high = float(np.max(finite_values))
            max_abs = float(max(abs(low), abs(high)))
        else:
            low = high = max_abs = None
        if values.size > 1 and finite.all():
            differences = np.abs(np.diff(values))
            change_boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
            run_bounds = np.concatenate(([0], change_boundaries, [values.size]))
            longest_run = int(np.diff(run_bounds).max(initial=0))
            max_step = float(np.max(differences))
            p999_step = float(np.percentile(differences, 99.9))
            exact_flatline_s = longest_run / source_fs
        else:
            max_step = p999_step = None
            exact_flatline_s = None
        per_channel.append({
            "source_row": channel_index + 1,
            "finite_sample_count": int(np.count_nonzero(finite)),
            "nonfinite_sample_count": int(values.size - np.count_nonzero(finite)),
            "minimum_finite_raw_value": low,
            "maximum_finite_raw_value": high,
            "maximum_absolute_finite_raw_value": max_abs,
            "maximum_adjacent_absolute_step": max_step,
            "p999_adjacent_absolute_step": p999_step,
            "longest_exact_flatline_seconds": exact_flatline_s,
        })
    del payload
    return {
        "participant_id": record["participant_id"],
        "dataset_id": record["dataset_id"],
        "sample_rate_hz": source_fs,
        "sample_count": n_samples,
        "payload_bytes_exactly_match_header": True,
        "nonfinite_sample_count": nonfinite,
        "sample_grid_monotone_by_index": True,
        "physical_continuity_status": "unknown_no_acquisition_clock_or_gap_markers",
        "clipping_status": "unknown_no_ADC_rail_or_resolution_metadata",
        "abrupt_step_policy": "reported_descriptively_no_outcome_tuned_rejection_threshold",
        "per_channel": per_channel,
        "status": "numerical_sample_scan_pass" if nonfinite == 0 else "numerical_sample_scan_hold",
    }


def _source_record_by_id(dataset_root: Path, participant_id: str) -> dict[str, Any]:
    return _subject_metadata(dataset_root, participant_id)


def _smoke_records(data_root: Path, protocol: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        dataset: [_source_record_by_id(data_root / dataset, pid)
                  for pid in protocol["smoke_selection"][dataset]]
        for dataset in DATASETS
    }


def _windows(n_samples: int) -> list[tuple[int, int, int, str]]:
    """Return matched 20 s anchors and their exact five 4 s children."""
    out: list[tuple[int, int, int, str]] = []
    anchor_samples = 20 * int(TARGET_FS_HZ)
    child_samples = 4 * int(TARGET_FS_HZ)
    anchor_index = 0
    for start in range(0, n_samples - anchor_samples + 1, anchor_samples):
        end = start + anchor_samples
        out.append((20, start, end, f"anchor20_{anchor_index:04d}"))
        for child in range(5):
            child_start = start + child * child_samples
            out.append((4, child_start, child_start + child_samples,
                        f"anchor20_{anchor_index:04d}_child4_{child:02d}"))
        anchor_index += 1
    return out


def _append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(_jsonable(row), ensure_ascii=False,
                                   sort_keys=True, allow_nan=False) + "\n")
            count += 1
    return count


def _disk_guard(output_root: Path) -> None:
    free = shutil.disk_usage(output_root).free
    if free < 2 * 1024**3:
        raise BudgetStop(f"disk_free_below_2_GiB:{free}")


def _elapsed_guard(started: float, budget_seconds: float, output_root: Path) -> None:
    _disk_guard(output_root)
    if time.perf_counter() - started > budget_seconds:
        raise BudgetStop("phase_wall_clock_budget_exceeded")


def _synthetic_controls() -> dict[str, Any]:
    rng = np.random.default_rng(140014)
    base = rng.normal(size=(8, 2000))
    base += 0.35 * base.mean(axis=0, keepdims=True)
    perm = rng.permutation(base.shape[1])
    oi = compute_oi_matrix(base, tuple(f"C{i}" for i in range(base.shape[0])))
    oi_permuted = compute_oi_matrix(base[:, perm], tuple(f"C{i}" for i in range(base.shape[0])))
    oi_gain = compute_oi_matrix(base * 2.75, tuple(f"C{i}" for i in range(base.shape[0])))
    ed = compute_ed_raw_sensor_covariance_participation_ratio(base, tuple(f"C{i}" for i in range(base.shape[0])))
    ed_permuted = compute_ed_raw_sensor_covariance_participation_ratio(base[:, perm], tuple(f"C{i}" for i in range(base.shape[0])))
    nie = _compute_nie(base, tuple(f"C{i}" for i in range(base.shape[0])))
    nie_permuted = _compute_nie(base[:, perm], tuple(f"C{i}" for i in range(base.shape[0])))
    constant = compute_oi_matrix(np.ones((3, 100)), ("A", "B", "C"))
    near_constant = compute_oi_matrix(
        1.0 + np.random.default_rng(140015).normal(scale=1e-14, size=(3, 100)),
        ("A", "B", "C"),
    )
    replay = compute_oi_matrix(base.copy(), tuple(f"C{i}" for i in range(base.shape[0])))
    out = {
        "time_permutation": {
            "OI_original": oi.value, "OI_permuted": oi_permuted.value,
            "ED_original": ed.value, "ED_permuted": ed_permuted.value,
            "NIE_original": nie.value, "NIE_permuted": nie_permuted.value,
            "ED_invariant_within_1e-12": bool(ed.valid and ed_permuted.valid and abs(float(ed.value) - float(ed_permuted.value)) <= 1e-12),
            "NIE_invariant_within_1e-12": bool(nie.valid and nie_permuted.valid and abs(float(nie.value) - float(nie_permuted.value)) <= 1e-12),
            "OI_changed": bool(oi.valid and oi_permuted.valid and not math.isclose(float(oi.value), float(oi_permuted.value), rel_tol=1e-8, abs_tol=1e-10)),
        },
        "common_positive_gain": {
            "OI_original": oi.value, "OI_gain": oi_gain.value,
            "exact_invariant": bool(oi.valid and oi_gain.valid and float(oi.value) == float(oi_gain.value)),
        },
        "constant_channel_control": {"computed": constant.computed, "valid": constant.valid, "value": constant.value, "reason": constant.reason},
        "near_constant_control": {"computed": near_constant.computed, "valid": near_constant.valid, "value": near_constant.value, "reason": near_constant.reason},
        "exact_replay": {"same_value": bool(oi.valid and replay.valid and float(oi.value) == float(replay.value))},
        "qualified": False,
    }
    out["passed"] = bool(
        out["time_permutation"]["ED_invariant_within_1e-12"]
        and out["time_permutation"]["NIE_invariant_within_1e-12"]
        and out["common_positive_gain"]["exact_invariant"]
        and out["exact_replay"]["same_value"]
        and constant.computed and near_constant.computed
    )
    return _jsonable(out)


def run_p01(*, output_root: Path, data_root: Path = DATA_ROOT) -> dict[str, Any]:
    protocol = _frozen_protocol(output_root)
    frozen_hashes = _inventory_hash_lookup(output_root)
    started = time.perf_counter()
    phase_root = output_root / "P01"
    phase_root.mkdir(exist_ok=False)
    smoke = _smoke_records(data_root, protocol)
    common_labels = _frozen_common_labels(protocol)
    reports: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for record in smoke[dataset]:
            _elapsed_guard(started, 1.5 * 3600, output_root)
            source_sha = _verify_record_files(record, frozen_hashes)
            qc = _source_qc(record)
            native, labels, resample = _load_resampled_eeg(record)
            realizations = _realization_signals(native, labels, common_labels)
            n_samples = native.shape[-1]
            four_samples = 4 * int(TARGET_FS_HZ)
            cadence_oi: dict[str, list[float]] = {SOURCE_REF_KIND: [], AVERAGE_REF_KIND: []}
            participant_rows: list[dict[str, Any]] = []
            for ref_kind, (ref_signals, ref_labels, ref_qc, reference_set_labels) in realizations.items():
                for start in range(0, n_samples - four_samples + 1, four_samples):
                    oi_result = compute_oi_matrix(ref_signals[:, start:start + four_samples], ref_labels)
                    if oi_result.valid and oi_result.value is not None:
                        cadence_oi[ref_kind].append(float(oi_result.value))
                    participant_rows.append({
                        "record_type": "baseline_oi_cadence",
                        "dataset_id": dataset, "participant_id": record["participant_id"],
                        "reference_id": SOURCE_REFERENCE_ID if ref_kind == SOURCE_REF_KIND else AVERAGE_REFERENCE_ID,
                        "reference_kind": ref_kind, "window_start_sample": start,
                        "window_end_sample_exclusive": start + four_samples,
                        "oi_value": oi_result.value, "oi_valid": oi_result.valid,
                        "oi_reason": oi_result.reason, "resampling": resample,
                    })
                values = cadence_oi[ref_kind]
                iqr = float(np.percentile(values, 75) - np.percentile(values, 25)) if len(values) >= 2 else None
                window_start, window_end = 0, four_samples
                baseline_signal = ref_signals[:, window_start:window_end]
                baseline_records, _ = _metric_rows(
                    baseline_signal, dataset_id=dataset, participant_id=record["participant_id"],
                    recording_id=record["recording_id"], source_sha256=source_sha,
                    labels=ref_labels,
                    reference_id=SOURCE_REFERENCE_ID if ref_kind == SOURCE_REF_KIND else AVERAGE_REFERENCE_ID,
                    reference_kind=ref_kind, start=window_start, end=window_end,
                    duration_s=4, record_class="matched", resample=resample,
                    metadata_qc=tuple(record["metadata_qc"]) + tuple(ref_qc),
                    source_sample_count=int(record["sample_count"]),
                    reference_set_labels=reference_set_labels,
                )
                base_oi = next((x["measurement_value"] for x in baseline_records if x["family"] == "OI"), None)
                if base_oi is None:
                    continue
                for fraction in (0.001, 0.01):
                    for replicate in range(8):
                        _elapsed_guard(started, 1.5 * 3600, output_root)
                        seed_text = "|".join((dataset, record["participant_id"], source_sha,
                                               str(window_start), float(fraction).hex(), str(replicate)))
                        seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "big")
                        rng = np.random.Generator(np.random.PCG64(seed))
                        rms = np.sqrt(np.mean(baseline_signal * baseline_signal, axis=1, dtype=np.float64))
                        noise = rng.standard_normal(size=baseline_signal.shape) * (fraction * rms[:, None])
                        perturbed = baseline_signal + noise
                        perturbed_records, _ = _metric_rows(
                            perturbed, dataset_id=dataset, participant_id=record["participant_id"],
                            recording_id=record["recording_id"], source_sha256=source_sha,
                            labels=ref_labels,
                            reference_id=SOURCE_REFERENCE_ID if ref_kind == SOURCE_REF_KIND else AVERAGE_REFERENCE_ID,
                            reference_kind=ref_kind, start=window_start, end=window_end,
                            duration_s=4, record_class="matched", resample=resample,
                            metadata_qc=tuple(record["metadata_qc"]) + tuple(ref_qc),
                            source_sample_count=int(record["sample_count"]),
                            reference_set_labels=reference_set_labels,
                        )
                        perturbed_oi = next((x["measurement_value"] for x in perturbed_records if x["family"] == "OI"), None)
                        replay = compute_oi_matrix(perturbed, ref_labels)
                        replay_equal = replay.valid and perturbed_oi is not None and float(replay.value) == float(perturbed_oi)
                        delta = abs(float(perturbed_oi) - float(base_oi)) if perturbed_oi is not None else None
                        standardized = delta / iqr if delta is not None and iqr is not None and iqr > 0 else None
                        for family_record in perturbed_records:
                            family_record.update({
                                "record_type": "oi_noise_sensitivity_family_recomputed",
                                "noise_fraction_of_channel_rms": fraction,
                                "replicate_index": replicate,
                                "seed_sha256": hashlib.sha256(seed_text.encode()).hexdigest(),
                                "seed_value": seed,
                                "baseline_oi": float(base_oi),
                                "perturbed_oi": _jsonable(perturbed_oi),
                                "absolute_oi_change": delta,
                                "oi_cadence_iqr": iqr,
                                "standardized_absolute_oi_change": standardized,
                                "standardization_reason": None if standardized is not None else ("zero_or_undefined_oi_cadence_iqr" if iqr is None or iqr == 0 else "oi_unavailable"),
                                "exact_oi_replay_equal": bool(replay_equal),
                            })
                            delta_rows.append({
                                "dataset_id": dataset, "participant_id": record["participant_id"],
                                "reference_kind": ref_kind, "family": family_record["family"],
                                "noise_fraction_of_channel_rms": fraction, "replicate_index": replicate,
                                "seed_sha256": hashlib.sha256(seed_text.encode()).hexdigest(),
                                "oi_absolute_change": delta if family_record["family"] == "OI" else None,
                                "oi_standardized_absolute_change": standardized if family_record["family"] == "OI" else None,
                                "oi_cadence_iqr": iqr,
                                "recomputed_family_valid": family_record["valid"],
                                "reason": family_record["reason"],
                                "exact_oi_replay_equal": bool(replay_equal),
                            })
                        participant_rows.extend(perturbed_records)
            participant_rows.append({"record_type": "source_qc", **qc})
            participant_rows.append({"record_type": "synthetic_controls", "result": _synthetic_controls()})
            _write_gzip_jsonl(phase_root / f"{dataset}_{record['participant_id']}.jsonl.gz", participant_rows)
            reports.append(qc)
            del native, realizations
            print(f"P01 completed {dataset}/{record['participant_id']}", flush=True)

    _write_csv(phase_root / "oi_sensitivity_summary.csv", delta_rows)
    synthetic = _synthetic_controls()
    elapsed = time.perf_counter() - started
    qc_pass = all(row["status"] == "numerical_sample_scan_pass" for row in reports)
    replay_pass = all(row["exact_oi_replay_equal"] for row in delta_rows if row.get("family") == "OI")
    summary = {
        "packet_id": PACKET_ID, "phase": "P01", "engineering_status": "PASS" if qc_pass and replay_pass and synthetic["passed"] else "HOLD",
        "scientific_status": "HOLD_PHYSICAL_CONTINUITY_UNKNOWN",
        "sample_qc_subjects": len(reports), "noise_family_rows": len(delta_rows),
        "oi_perturbation_windows": len([r for r in delta_rows if r.get("family") == "OI"]),
        "exact_replay_all": replay_pass, "synthetic_controls": synthetic,
        "physical_continuity": "unknown; no acquisition clock or explicit gaps/markers",
        "clipping": "unknown; no ADC rail metadata",
        "historical_SL013_ratio_1_227": "context only, not used as a threshold",
        "elapsed_seconds": elapsed, "qualified": False,
    }
    _write_json(phase_root / "P01_summary.json", summary)
    report = [
        "# P01 FAST4-QC-OI-001", "",
        f"- Engineering status: **{summary['engineering_status']}**.",
        f"- Scientific status: **{summary['scientific_status']}**.",
        f"- Source QC subjects: {summary['sample_qc_subjects']} (up to four per dataset).",
        f"- OI noise-response window: first four-second child of the first full 20-second anchor; {summary['oi_perturbation_windows']} OI perturbation records.",
        "- Per selected window: eight independent PCG64 seeds at 0.1% and 1% of channel RMS; every scalar/vector family was recomputed without changing its estimator.",
        "- OI cadence IQR uses every complete four-second cadence window in the same recording; standardized change is null with an explicit reason if the denominator is zero/undefined.",
        f"- Exact replay: {'PASS' if replay_pass else 'HOLD'}; synthetic invariance/tie controls: {'PASS' if synthetic['passed'] else 'HOLD'}.",
        "- Numerical checks found no physical acquisition clock or gap annotations; the recording grid is contiguous in the binary payload, which does not establish physical continuity.",
        "- Clipping is unknown because ADC rail metadata is absent. Amplitude steps are descriptive, with no outcome-tuned rejection threshold.",
        "- The prior SL-013 OI median normalized perturbation ratio 1.227 is context only; no failure threshold was derived from it.",
        "- No clinical outcomes were read; all values remain unqualified.", "",
    ]
    (phase_root / "P01_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    _append_execution_log(output_root, f"P01 completed with engineering={summary['engineering_status']}, scientific={summary['scientific_status']}; elapsed {elapsed:.1f} s.")
    _update_gate_files(output_root)
    return summary


def run_p02(*, output_root: Path, data_root: Path = DATA_ROOT,
            mode: str = "smoke") -> dict[str, Any]:
    protocol = _frozen_protocol(output_root)
    frozen_hashes = _inventory_hash_lookup(output_root)
    if mode not in ("smoke", "cohort"):
        raise ValueError("P02 mode must be smoke or cohort")
    started = time.perf_counter()
    budget = 3.0 * 3600 if mode == "cohort" else 3.0 * 3600
    phase_root = output_root / ("P02_smoke" if mode == "smoke" else "P02_cohort")
    phase_root.mkdir(exist_ok=False)
    normalized_path = phase_root / "normalized_measurements.jsonl"
    diagnostics_path = phase_root / "window_diagnostics.jsonl"
    family_status: dict[str, dict[str, int]] = {family: {"valid": 0, "unsupported": 0} for family in CORE_FAMILIES}
    participants_done: list[dict[str, Any]] = []
    common_labels = _frozen_common_labels(protocol)
    total_original = 0
    total_normalized = 0
    dataset_records: dict[str, list[dict[str, Any]]] = {}
    for dataset in DATASETS:
        if mode == "smoke":
            ids = list(protocol["smoke_selection"][dataset])
        else:
            ids = _participant_ids(data_root / dataset)
        dataset_records[dataset] = [_source_record_by_id(data_root / dataset, pid) for pid in ids]

    for dataset in DATASETS:
        for record in dataset_records[dataset]:
            _elapsed_guard(started, budget, output_root)
            participant_id = record["participant_id"]
            try:
                source_sha = _verify_record_files(record, frozen_hashes)
                native, labels, resample = _load_resampled_eeg(record)
                if len(labels) != EXPECTED_EEG_COUNT:
                    raise PreflightError("expected_61_BIDS_EEG_channels")
                realizations = _realization_signals(native, labels, common_labels)
                participant_rows: list[dict[str, Any]] = []
                normalized_rows: list[dict[str, Any]] = []
                diagnostics_rows: list[dict[str, Any]] = []
                for duration_s, start, end, window_id in _windows(native.shape[1]):
                    _elapsed_guard(started, budget, output_root)
                    window_signals_by_ref = {
                    kind: (source[:, start:end], ref_labels, ref_qc, reference_set_labels)
                        for kind, (source, ref_labels, ref_qc, reference_set_labels) in realizations.items()
                    }
                    for ref_kind, (signals, ref_labels, ref_qc, reference_set_labels) in window_signals_by_ref.items():
                        reference_id = SOURCE_REFERENCE_ID if ref_kind == SOURCE_REF_KIND else AVERAGE_REFERENCE_ID
                        records, norm = _metric_rows(
                            signals, dataset_id=dataset, participant_id=participant_id,
                            recording_id=record["recording_id"], source_sha256=source_sha,
                            labels=ref_labels, reference_id=reference_id,
                            reference_kind=ref_kind, start=start, end=end,
                            duration_s=duration_s, record_class="matched", resample=resample,
                            metadata_qc=tuple(record["metadata_qc"]) + tuple(ref_qc),
                            source_sample_count=int(record["sample_count"]),
                            reference_set_labels=reference_set_labels,
                        )
                        for row in records:
                            row["window_id"] = window_id
                            participant_rows.append(row)
                            family_status[row["family"]]["valid" if row["valid"] else "unsupported"] += 1
                        for row in norm:
                            row["window_id"] = window_id
                            normalized_rows.append(row)
                        diagnostic = records[0]["diagnostics"] if records else {}
                        diagnostics_rows.append({
                            "dataset_id": dataset, "participant_id": participant_id,
                            "recording_id": record["recording_id"], "window_id": window_id,
                            "record_class": "matched", "reference_id": reference_id,
                            "reference_realization_kind": ref_kind,
                            "window_start_sample": start, "window_end_sample_exclusive": end,
                            "duration_s": duration_s, "sampling_rate_hz": TARGET_FS_HZ,
                            "diagnostics": diagnostic,
                        })
                participant_out = phase_root / dataset
                participant_out.mkdir(exist_ok=True)
                part_path = participant_out / f"{participant_id}.jsonl.gz"
                count = _write_gzip_jsonl(part_path, participant_rows)
                total_original += count
                total_normalized += _append_jsonl(normalized_path, normalized_rows)
                _append_jsonl(diagnostics_path, diagnostics_rows)
                participants_done.append({
                    "dataset_id": dataset, "participant_id": participant_id,
                    "status": "processed", "window_count": len(_windows(native.shape[1])),
                    "record_count": count, "normalized_record_count": len(normalized_rows),
                    "source_sha256": source_sha,
                })
                del native, realizations
                print(f"P02 {mode} completed {dataset}/{participant_id}: {count} family records", flush=True)
            except BudgetStop:
                raise
            except Exception as exc:
                participants_done.append({
                    "dataset_id": dataset, "participant_id": participant_id,
                    "status": "failed", "reason": f"{type(exc).__name__}:{exc}",
                    "record_count": 0,
                })
                _write_gzip_jsonl(phase_root / f"{dataset}_{participant_id}_failure.jsonl.gz", [{
                    "dataset_id": dataset, "participant_id": participant_id,
                    "status": "unsupported", "reason": f"source_or_runner_failure:{type(exc).__name__}:{exc}",
                    "computed": False, "valid": False, "qualified": False,
                }])
                print(f"P02 {mode} HOLD {dataset}/{participant_id}: {type(exc).__name__}: {exc}", flush=True)

    audit_result: dict[str, Any] = {"status": "NOT_RUN", "returncode": None}
    if normalized_path.is_file():
        packet_root = Path(__file__).resolve().parents[6] / "project/new_mnps_v.3.2/science_lead/MNPS_v3_2_SL014_Psychosis_DualScale_20261015/MNPS_v3_2_SL014_Psychosis_DualScale_20261015"
        audit_script = packet_root / "tools/measurement_contract_audit.py"
        audit_report = phase_root / "measurement_contract_audit.json"
        result = __import__("subprocess").run(
            [sys.executable, str(audit_script), str(normalized_path), "--check-partitions", "--report", str(audit_report)],
            cwd=packet_root, capture_output=True, text=True,
        )
        audit_result = {
            "status": "PASS" if result.returncode == 0 else "HOLD",
            "returncode": result.returncode, "stdout": result.stdout,
            "stderr": result.stderr, "report": str(audit_report),
        }
    expected_count = sum(len(protocol["smoke_selection"][dataset]) for dataset in DATASETS) if mode == "smoke" else sum(len(_participant_ids(data_root / dataset)) for dataset in DATASETS)
    processed_count = sum(row["status"] == "processed" for row in participants_done)
    complete = processed_count == expected_count and audit_result["status"] == "PASS"
    elapsed = time.perf_counter() - started
    summary = {
        "packet_id": PACKET_ID, "phase": "P02", "mode": mode,
        "engineering_status": "PASS" if complete else "HOLD",
        "scientific_status": "HOLD_PHYSICAL_CONTINUITY_AND_CHANNEL_MAPPING_LIMITS",
        "expected_participant_count": expected_count,
        "processed_participant_count": processed_count,
        "failed_participant_count": expected_count - processed_count,
        "original_family_record_count": total_original,
        "normalized_family_record_count": total_normalized,
        "family_status_counts": family_status,
        "measurement_contract_audit": audit_result,
        "participants": participants_done,
        "elapsed_seconds": elapsed,
        "cohort_expansion_status": "NOT_RUN_AFTER_SMOKE" if mode == "smoke" else "COHORT_RUN_COMPLETED_OR_HELD",
        "qualified": False,
    }
    _write_json(phase_root / "P02_summary.json", summary)
    lines = [
        f"# P02 PSYCHOSIS-DUALSCALE-001 — {mode}", "",
        f"- Engineering status: **{summary['engineering_status']}**.",
        f"- Scientific status: **{summary['scientific_status']}**.",
        f"- Participants processed: {processed_count}/{expected_count}; measurement records: {total_original}; normalized rows: {total_normalized}.",
        f"- Contract/partition audit: **{audit_result['status']}**.",
        "- Realizations: fixed 49-channel scalp intersection, recorded TP9 reference primary, 61-channel native EEG mean-reference sensitivity; sensitivity reference outputs are not pooled or treated as the same realization.",
        "- FAST4 is `[PCV, TP, NIE, RD, OI, SC, ED, AR_pair_01]`; LONG20 retains seven scalar families plus the ordered AR9 vector. No AR aggregation is performed.",
        "- Missing/undefined values remain null with kernel reason; all measurements remain unqualified.",
        "- Cross-dataset label mapping relies on BIDS channel-row order because BrainVision header labels are generic; no electrodes/coordinate files are present.",
        "- Physical continuity remains unknown. These are descriptive window measurements, not prospective predictions.",
        "- No clinical outcomes were read.", "",
    ]
    (phase_root / "P02_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    if mode == "smoke":
        shutil.copyfile(phase_root / "P02_REPORT.md", output_root / "P02_REPORT.md")
    _append_execution_log(output_root, f"P02 {mode} completed with engineering={summary['engineering_status']}; participants {processed_count}/{expected_count}; elapsed {elapsed:.1f} s.")
    _update_gate_files(output_root)
    return summary


def _make_operator_truth(seed: int, dims: int = 8) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(dims, dims)))
    eigen = np.linspace(0.25, 0.82, dims)
    matrix = q @ np.diag(eigen) @ q.T
    offset = rng.normal(scale=0.08, size=dims)
    return matrix, offset


def _simulate_operator(seed: int, *, process_sigma: float, observation_sigma: float,
                       n_states: int = 700) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix, offset = _make_operator_truth(seed)
    process_rng = np.random.default_rng(seed + 1)
    latent = np.empty((n_states, matrix.shape[0]), dtype=np.float64)
    latent[0] = process_rng.normal(size=matrix.shape[0])
    for index in range(1, n_states):
        latent[index] = matrix @ latent[index - 1] + offset + process_rng.normal(
            scale=process_sigma, size=matrix.shape[0]
        )
    observation_rng = np.random.default_rng(seed + 0x5A17)
    observed = latent + observation_rng.normal(scale=observation_sigma, size=latent.shape)
    return observed, matrix, offset


def _fit_affine_ols(states: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
    if states.ndim != 2 or not np.isfinite(states).all() or states.shape[0] < states.shape[1] + 2:
        raise ValueError("insufficient_or_invalid_states")
    design = np.column_stack((states[:-1], np.ones(states.shape[0] - 1)))
    target = states[1:]
    rank = int(np.linalg.matrix_rank(design))
    if rank != design.shape[1]:
        raise ValueError(f"rank_deficient_design:{rank}/{design.shape[1]}")
    coefficient, *_ = np.linalg.lstsq(design, target, rcond=None)
    return coefficient[:-1].T, coefficient[-1], rank, float(np.linalg.cond(design))


def run_p03_synthetic(*, output_root: Path) -> dict[str, Any]:
    """Run Stage S only; empirical operator is gated by continuity/causality."""
    protocol = _frozen_protocol(output_root)
    phase_root = output_root / "P03"
    phase_root.mkdir(exist_ok=False)
    settings = protocol["p03_synthetic_noise_grid"]
    process_arms = [float(x) for x in settings["process_sigma"]]
    observation_arms = [float(x) for x in settings["observation_sigma"]]
    reps = int(settings["replicates_per_cell"])
    seed_start = int(settings["seed_start"])
    train_states = int(settings["train_states"])
    test_count = int(settings["test_successor_transitions"])
    rows: list[dict[str, Any]] = []
    seed_counter = 0
    for process_sigma in process_arms:
        for observation_sigma in observation_arms:
            for replicate in range(reps):
                seed = seed_start + seed_counter
                seed_counter += 1
                states, true_A, true_b = _simulate_operator(
                    seed, process_sigma=process_sigma,
                    observation_sigma=observation_sigma,
                    n_states=train_states + test_count + 1,
                )
                train = states[:train_states]
                matrix, offset, rank, condition = _fit_affine_ols(train)
                x_test = states[train_states:train_states + test_count]
                y_test = states[train_states + 1:train_states + test_count + 1]
                prediction = x_test @ matrix.T + offset
                persistence = x_test
                mean_successor = np.repeat(np.mean(train[1:], axis=0, keepdims=True), test_count, axis=0)
                rows.append({
                    "process_sigma": process_sigma,
                    "observation_sigma": observation_sigma,
                    "replicate_index": replicate,
                    "seed": seed,
                    "design_rank": rank,
                    "design_columns": train.shape[1] + 1,
                    "design_condition_number": condition,
                    "relative_A_parameter_error": float(np.linalg.norm(matrix - true_A) / np.linalg.norm(true_A)),
                    "absolute_b_parameter_error": float(np.linalg.norm(offset - true_b)),
                    "heldout_mse_full_ols": float(np.mean((y_test - prediction) ** 2)),
                    "heldout_mse_persistence": float(np.mean((y_test - persistence) ** 2)),
                    "heldout_mse_training_mean_successor": float(np.mean((y_test - mean_successor) ** 2)),
                    "heldout_transition_count": test_count,
                    "qualified": False,
                })

    exact_states, exact_A, exact_b = _simulate_operator(
        20261015, process_sigma=0.0, observation_sigma=0.0, n_states=350
    )
    exact_A_hat, exact_b_hat, exact_rank, exact_condition = _fit_affine_ols(exact_states)
    exact_recovery = {
        "design_rank": exact_rank,
        "design_columns": exact_states.shape[1] + 1,
        "design_condition_number": exact_condition,
        "relative_A_error": float(np.linalg.norm(exact_A_hat - exact_A) / np.linalg.norm(exact_A)),
        "absolute_b_error": float(np.linalg.norm(exact_b_hat - exact_b)),
    }
    rank_test, _, _ = _simulate_operator(
        20261017, process_sigma=0.3, observation_sigma=0.05,
        n_states=train_states,
    )
    duplicate = rank_test.copy()
    duplicate[:, -1] = duplicate[:, 0]
    try:
        _fit_affine_ols(duplicate[:train_states])
    except ValueError as exc:
        rank_negative = {"rejected": True, "reason": str(exc)}
    else:
        rank_negative = {"rejected": False, "reason": "rank_deficient_design_accepted"}

    # A mutation after the training prefix must not change a training-only fit.
    prefix_source, _, _ = _simulate_operator(
        20261018, process_sigma=0.15, observation_sigma=0.05,
        n_states=train_states + 200,
    )
    future_mutation = prefix_source.copy()
    future_mutation[train_states:] += 1000.0
    prefix_a, prefix_b, _, _ = _fit_affine_ols(prefix_source[:train_states])
    mutated_a, mutated_b, _, _ = _fit_affine_ols(future_mutation[:train_states])
    prefix_invariant = bool(np.array_equal(prefix_a, mutated_a) and np.array_equal(prefix_b, mutated_b))

    wrong_subject, _, _ = _simulate_operator(
        20261016, process_sigma=0.15, observation_sigma=0.05,
        n_states=train_states + test_count + 1,
    )
    own_fit, own_offset, _, _ = _fit_affine_ols(wrong_subject[:train_states])
    cross_input = states[train_states:train_states + test_count]
    cross_target = states[train_states + 1:train_states + test_count + 1]
    wrong_subject_control = {
        "control_subject_fit_source": "independent synthetic participant seed 20261016",
        "target_subject_seed": rows[-1]["seed"],
        "heldout_mse": float(np.mean((cross_target - (cross_input @ own_fit.T + own_offset)) ** 2)),
        "target_persistence_mse": float(np.mean((cross_target - cross_input) ** 2)),
    }
    output = {
        "packet_id": PACKET_ID,
        "phase": "P03 Stage S synthetic identification only",
        "noise_grid": {"process_sigma": process_arms, "observation_sigma": observation_arms,
                       "replicates_per_cell": reps, "seed_start": seed_start},
        "grid_rows": rows,
        "exact_noiseless_recovery": exact_recovery,
        "rank_deficient_coordinate_control": rank_negative,
        "training_prefix_future_mutation_invariant": prefix_invariant,
        "wrong_participant_operator_control": wrong_subject_control,
        "temporal_order_shuffle_control": "NOT_RUN; empirical operator blocked and synthetic grid does not establish neural trajectory validity",
        "empirical_operator_status": "NOT_RUN_PHYSICAL_CONTINUITY_UNKNOWN_AND_NONCAUSAL_FEATURES",
        "scientific_candidate_status": "NOT_TESTED",
        "qualified": False,
    }
    engineering_pass = (
        exact_recovery["relative_A_error"] < 1e-8
        and exact_recovery["absolute_b_error"] < 1e-8
        and rank_negative["rejected"]
        and prefix_invariant
        and len(rows) == len(process_arms) * len(observation_arms) * reps
        and all(math.isfinite(float(row["heldout_mse_full_ols"])) for row in rows)
    )
    output["engineering_status"] = "PASS" if engineering_pass else "HOLD"
    output["scientific_status"] = "NOT_TESTED_SYNTHETIC_ONLY"
    _write_json(phase_root / "P03_synthetic_grid.json", output)
    _write_csv(phase_root / "P03_synthetic_replicates.csv", rows)
    lines = [
        "# P03 FAST4-DISCRETE-OPERATOR-001", "",
        f"- Engineering status: **{output['engineering_status']}** (Stage S only).",
        "- Scientific status: **NOT_TESTED**; no empirical operator was estimated.",
        f"- Synthetic grid: {len(process_arms)} process-noise × {len(observation_arms)} observation-noise cells, {reps} seeded replicates per cell ({len(rows)} fits).",
        f"- Exact noiseless parameter recovery: relative A error {exact_recovery['relative_A_error']:.3g}; absolute b error {exact_recovery['absolute_b_error']:.3g}.",
        f"- Rank-deficient coordinate duplication rejected: {rank_negative['rejected']}; training prefix unchanged after future mutation: {prefix_invariant}.",
        "- The operator remains a finite four-second discrete transition model, not a neural Jacobian or continuous generator.",
        "- Empirical forecasting is `NOT_RUN`: physical continuity is unknown and alpha filtering/Hilbert features are noncausal.",
        "- All synthetic outputs are mathematical controls, not EEG evidence or scientific qualification.", "",
    ]
    (phase_root / "P03_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    _append_execution_log(output_root, f"P03 Stage S completed with engineering={output['engineering_status']}; empirical Stage E NOT_RUN.")
    _update_gate_files(output_root)
    return output


def _append_execution_log(output_root: Path, message: str) -> None:
    path = output_root / "execution_log.md"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f"- {datetime.now(timezone.utc).isoformat()} {message}\n")


def _update_gate_files(output_root: Path) -> None:
    summaries: dict[str, Any] = {}
    for key, path in (
        ("P01", output_root / "P01/P01_summary.json"),
        ("P02_smoke", output_root / "P02_smoke/P02_summary.json"),
        ("P02_cohort", output_root / "P02_cohort/P02_summary.json"),
        ("P03", output_root / "P03/P03_synthetic_grid.json"),
    ):
        if path.is_file():
            summaries[key] = _read_json(path)
    p04_status = "NOT_RUN_DEPENDENCIES_NOT_MET"
    if summaries.get("P02_cohort", {}).get("engineering_status") == "PASS":
        p04_status = "NOT_RUN_LABELS_REMAIN_SEALED"
    gate_lines = [
        f"# SL-014 gate status — {PACKET_ID}", "",
        "| Phase | Engineering | Scientific | Notes |", "|---|---|---|---|",
        "| P00 | PASS_WITH_LIMITATIONS | HOLD_FOR_PROSPECTIVE_CLAIMS | Data/timebase frozen; physical continuity unknown; BIDS row-order channel map is an assumption. |",
    ]
    if "P01" in summaries:
        gate_lines.append(f"| P01 | {summaries['P01'].get('engineering_status')} | {summaries['P01'].get('scientific_status')} | Smoke QC and perturbation profile; physical continuity unknown. |")
    else:
        gate_lines.append("| P01 | NOT_RUN | NOT_TESTED | Awaiting P00 freeze. |")
    for key, label in (("P02_smoke", "P02 smoke"), ("P02_cohort", "P02 full cohort")):
        if key in summaries:
            gate_lines.append(f"| {label} | {summaries[key].get('engineering_status')} | {summaries[key].get('scientific_status')} | {summaries[key].get('processed_participant_count')}/{summaries[key].get('expected_participant_count')} participants; clinical outcomes sealed. |")
        else:
            gate_lines.append(f"| {label} | NOT_RUN | NOT_TESTED | Depends on frozen P00 and prior smoke gate. |")
    if "P03" in summaries:
        gate_lines.append(f"| P03 Stage S | {summaries['P03'].get('engineering_status')} | {summaries['P03'].get('scientific_status')} | Synthetic only; empirical operator not run. |")
    else:
        gate_lines.append("| P03 | NOT_RUN | NOT_TESTED | Synthetic stage pending; empirical stage is continuity/causality gated. |")
    gate_lines.append(f"| P04 | NOT_RUN | {p04_status} | No clinical table has been opened; requires full P02 cohort and label-release gate. |")
    gate_lines.extend(["", "All measurements remain `qualified=false`. No biological or diagnostic claim is made.", ""])
    (output_root / "GATE_STATUS.md").write_text("\n".join(gate_lines), encoding="utf-8")

    claims = [
        {"claim_id": "C001", "claim": "The two BIDS sources contain source files with header-consistent sample counts and payload sizes.", "evidence": "P00_INPUT_INVENTORY.csv and P00_SOURCE_QC.md", "status": "ENGINEERING_SUPPORTED", "limit": "Does not prove physical continuity."},
        {"claim_id": "C002", "claim": "The current source supports descriptive 4 s and 20 s MNPS v3.2 computation on a frozen shared sensor realization.", "evidence": "P02 reports when available", "status": "NOT_ESTABLISHED" if "P02_smoke" not in summaries else summaries["P02_smoke"].get("engineering_status"), "limit": "Smoke is not cohort coverage, comparability, qualification, or biology."},
        {"claim_id": "C003", "claim": "Physical acquisition continuity is established for the 4 s transitions.", "evidence": "No physical clock or gap annotations were found.", "status": "UNKNOWN", "limit": "P03 empirical forecasting is blocked."},
        {"claim_id": "C004", "claim": "The empirical discrete operator forecasts beyond simple baselines.", "evidence": "P03 empirical stage not run.", "status": "NOT_TESTED", "limit": "Do not call A a neural Jacobian."},
        {"claim_id": "C005", "claim": "MNPS provides a clinically useful psychosis signal.", "evidence": "No current outcomes read and P04 not run.", "status": "NOT_TESTED", "limit": "Historical manuscript is background only."},
    ]
    _write_csv(output_root / "CLAIM_LEDGER.csv", claims)
    if not (output_root / "P04_REPORT.md").exists():
        (output_root / "P04_REPORT.md").write_text(
            "# P04 PSYCHOSIS-REPLICATION-INCREMENTAL-001\n\n"
            "- Status: **NOT_RUN**.\n"
            "- Reason: full-cohort P02 has not passed; clinical outcome tables remain unopened.\n"
            "- No group counts, clinical comparisons, or prediction results are reported.\n"
            "- Historical v2 manuscript was reviewed as background per packet instruction and is not a calibration target.\n"
            "- Every measurement remains `qualified=false`.\n",
            encoding="utf-8",
        )

    current_commit = _read_json(output_root / "resolved_protocol.json")["current_repo"]["commit"]
    protocol = _read_json(output_root / "resolved_protocol.json")
    decision = [
        f"# SL-014 decision report — {PACKET_ID}", "",
        f"- Execution date: {datetime.now(timezone.utc).date().isoformat()} (UTC).",
        f"- Output root: `{output_root.resolve()}`.",
        f"- NMD repository commit at freeze: `{current_commit}`.",
        f"- Frozen protocol projection SHA-256: `{protocol['frozen_protocol_sha256']}`.",
        f"- Frozen protocol file SHA-256: `{_sha256_file(output_root / 'resolved_protocol.json')}`.",
        "- Current clinical outcome files read: **no**. Historical manuscript read as background per packet prompt.",
        "- Source cohorts: ds003944 and ds003947; source files are locally present and hashed in `input_inventory.csv`.",
        "- Shared primary sensor surface: frozen 49-channel scalp intersection; TP9 source reference primary, average-reference sensitivity is separate.",
        "- Scientific qualification: **false** for every measurement and claim.", "",
        "## Gate outcomes", "",
        "See `GATE_STATUS.md` and the phase reports for engineering and scientific status.",
        "P04 remains not run until outcome-blind full-cohort P02 passes and a separate label-release phase is reached.",
        "P03 empirical forecasting remains not run while physical continuity is unknown and the frozen alpha features are noncausal.", "",
        "## Limitations and next falsifiable step", "",
        "The source has no sample-clock/gap annotations, no electrode coordinates, generic BrainVision channel header labels, and BIDS channel-row order is used to map those generic rows. Sampling rates vary by participant and are harmonized to a nominal 1 kHz with a fixed anti-alias FIR. These choices support an auditable exploratory computation but do not establish cross-cohort equivalence.",
        "Next step: inspect and validate the smoke outputs and measured runtime/disk estimate. Expand P02 only if the smoke contract/partition audit passes. Do not open clinical labels before full P02 and protocol freeze are confirmed.", "",
    ]
    (output_root / "DECISION_REPORT.md").write_text("\n".join(decision), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("p00", "p01", "p02-smoke", "p02-cohort", "p03-synthetic"), required=True)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="fresh operator-confirmed output root; no default is provided")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[6])
    parser.add_argument("--analysis-root", type=Path, default=Path(r"J:\repos\nmd-analysis"))
    args = parser.parse_args(argv)
    if args.phase == "p00":
        preflight(output_root=args.output_root, repo_root=args.repo_root,
                  analysis_root=args.analysis_root, data_root=args.data_root)
    elif args.phase == "p01":
        run_p01(output_root=args.output_root, data_root=args.data_root)
    elif args.phase == "p02-smoke":
        if not (args.output_root / "P01/P01_summary.json").is_file():
            raise PreflightError("P01_must_complete_before_P02_smoke")
        run_p02(output_root=args.output_root, data_root=args.data_root, mode="smoke")
    elif args.phase == "p02-cohort":
        smoke_summary = _read_json(args.output_root / "P02_smoke/P02_summary.json")
        if smoke_summary.get("engineering_status") != "PASS":
            raise PreflightError("P02_full_cohort_blocked_smoke_gate_not_passed")
        if shutil.disk_usage(args.output_root).free < 5 * 1024**3:
            raise PreflightError("P02_full_cohort_requires_5_GiB_free_at_start")
        run_p02(output_root=args.output_root, data_root=args.data_root, mode="cohort")
    else:
        run_p03_synthetic(output_root=args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

