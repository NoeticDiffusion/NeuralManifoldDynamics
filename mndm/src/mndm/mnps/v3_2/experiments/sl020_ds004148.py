"""SL-020 ds004148 BrainVision runner for MNPS v3.2.

This is an isolated experimental runner. It never reads participant-table rows,
uses no clinical outcomes, does not integrate into the active pipeline, and
writes only to an explicitly supplied output phase root. EEG payloads are
stream-hashed after protocol freeze, then mapped read-only as BrainVision
float32 multiplexed frames. Results are experimental and unqualified.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import signal
from scipy.integrate import trapezoid
from scipy.signal import hilbert

REPO_ROOT = Path(__file__).resolve().parents[6]
SRC_ROOT = REPO_ROOT / "mndm" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mndm.mnps.v3_2.ar import compute_ar_activity_transition_vector
from mndm.mnps.v3_2.ed import compute_ed_raw_sensor_covariance_participation_ratio
from mndm.mnps.v3_2.nie import compute_nie_raw_weighted_global_efficiency
from mndm.mnps.v3_2.oi import compute_oi_matrix
from mndm.mnps.v3_2.pcv import compute_pcv_phase_order_parameter_sd
from mndm.mnps.v3_2.rd import compute_rd_centered_log_psd_dispersion
from mndm.mnps.v3_2.registry import FULL_VECTOR_CHALLENGE_001_BY_FAMILY
from mndm.mnps.v3_2.sc import compute_sc_lz78_exact_multiset_ratio
from mndm.mnps.v3_2.tp import compute_tp_log_rms_envelope_persistence


PROTOCOL_ID = "SL-020-MNPS32-RETEST-001"
AUDIT_ID = "SL020-R1-METADATA-001"
DATASET_ID = "ds004148"
FS_HZ = 500.0
CHANNEL_COUNT = 61
RECORDING_SECONDS = 300
WINDOW_DURATIONS = (4, 20)
TASKS = ("eyesclosed", "eyesopen", "mathematic", "memory", "music")
SESSIONS = ("ses-session1", "ses-session2", "ses-session3")
FAMILIES = ("PCV", "TP", "AR", "NIE", "RD", "OI", "SC", "ED")
PILOT_IDS = (
    "sub-16", "sub-21", "sub-23", "sub-55", "sub-42", "sub-19",
    "sub-39", "sub-58", "sub-13", "sub-07", "sub-32", "sub-29",
)
SMOKE_RECORDING = ("sub-16", "ses-session1", "eyesclosed")
SOURCE_REFERENCE_ID = "ds004148.BIDS_EEGReference.FCz.v1"
SENSOR_REALIZATION_ID = "ds004148.study_common61.bids_casefold_alias.v1"
PCV_PHASE_ID = "scipy.signal.hilbert_after_sosfiltfilt.v1"
PCV_BAND_ID = "butterworth4_zero_phase_8_13hz_per_window.v1"
AR_SCALING_ID = "raw_numeric_amplitude_rms.v1"
AR_VECTOR_IDS = {
    4: "per_channel_rms_two_contiguous_2s_blocks.v1",
    20: "per_channel_rms_ten_contiguous_2s_blocks.v1",
}
NIE_EDGE_ESTIMATOR_ID = "pearson_zero_lag_sensor_pairs.v1"
NIE_EDGE_RULE_ID = "absolute_correlation_no_threshold_exact_zero_diagonal.v1"
SC_REFERENCE_COUNT = 32
SC_MASTER_SEED = 310930
MIN_FREE_BYTES = 10 * 1024**3

PROTOCOL_PATH = REPO_ROOT / "project" / "new_mnps_v.3.2" / "protocols" / f"{PROTOCOL_ID}.md"
PROTOCOL_HASH_PATH = Path(str(PROTOCOL_PATH) + ".sha256")
AMENDMENT_PATH = REPO_ROOT / "project" / "new_mnps_v.3.2" / "protocols" / "SL-020-PROTOCOL-AMENDMENT-001.md"
AMENDMENT_HASH_PATH = Path(str(AMENDMENT_PATH) + ".sha256")
R1_AUDIT_ID = "R1_COMPLETE"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json_atomic(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(_jsonable(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _read_frozen_hash(path: Path, target: Path) -> str:
    if not path.is_file() or not target.is_file():
        raise FileNotFoundError(f"frozen_protocol_or_hash_missing:{path}")
    expected = path.read_text(encoding="ascii").split()[0].lower()
    actual = _sha256_file(target)
    if actual != expected:
        raise RuntimeError(f"frozen_protocol_hash_mismatch:{target.name}")
    return actual


def _canonical_label_hash(labels: Sequence[str], *, ordered: bool) -> str:
    payload = list(labels) if ordered else sorted(labels)
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _parse_vhdr(path: Path) -> dict[str, Any]:
    sections: dict[str, dict[str, str]] = {"Common Infos": {}, "Binary Infos": {}, "Channel Infos": {}}
    section = ""
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        text = line.strip()
        if not text or text.startswith(";"):
            continue
        if text.startswith("[") and text.endswith("]"):
            section = text[1:-1]
        elif "=" in text and section in sections:
            key, value = text.split("=", 1)
            sections[section][key.strip()] = value.strip()
    common, binary, channel_info = sections["Common Infos"], sections["Binary Infos"], sections["Channel Infos"]
    labels = [
        value.split(",", 1)[0].strip()
        for key, value in sorted(channel_info.items(), key=lambda item: int(item[0][2:]) if item[0][2:].isdigit() else 10**9)
        if key.lower().startswith("ch") and key[2:].isdigit()
    ]
    interval = float(common["SamplingInterval"]) if common.get("SamplingInterval") else None
    return {
        "data_file": common.get("DataFile"),
        "marker_file": common.get("MarkerFile"),
        "data_format": common.get("DataFormat"),
        "data_orientation": common.get("DataOrientation"),
        "channel_count": int(common["NumberOfChannels"]) if common.get("NumberOfChannels") else None,
        "sampling_interval_us": interval,
        "sampling_frequency_hz": 1_000_000.0 / interval if interval else None,
        "binary_format": binary.get("BinaryFormat"),
        "channel_labels": labels,
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path}")
    return value


def _read_channels(path: Path) -> tuple[list[str], list[str], list[str], list[float | None]]:
    import csv
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    return (
        [str(row.get("name", "")).strip() for row in rows],
        [str(row.get("units", "")).strip() for row in rows],
        [str(row.get("type", "")).strip() for row in rows],
        [float(row["sampling_frequency"]) if row.get("sampling_frequency") not in (None, "", "n/a") else None for row in rows],
    )


def _metadata_paths(dataset_root: Path, record: Mapping[str, Any]) -> dict[str, Path]:
    eeg_dir = dataset_root / str(record["participant_id"]) / str(record["session_id"]) / "eeg"
    stem = str(record["bids_stem"])
    return {
        "vhdr": eeg_dir / f"{stem}_eeg.vhdr",
        "vmrk": eeg_dir / f"{stem}_eeg.vmrk",
        "eeg": eeg_dir / f"{stem}_eeg.eeg",
        "eeg_json": eeg_dir / f"{stem}_eeg.json",
        "channels_tsv": eeg_dir / f"{stem}_channels.tsv",
        "events_tsv": eeg_dir / f"{stem}_events.tsv",
        "electrodes_tsv": eeg_dir / f"{record['participant_id']}_{record['session_id']}_electrodes.tsv",
    }


def _verify_source_metadata(dataset_root: Path, record: Mapping[str, Any]) -> tuple[dict[str, Path], dict[str, Any], dict[str, str], list[str], list[str]]:
    paths = _metadata_paths(dataset_root, record)
    for name in ("vhdr", "vmrk", "eeg", "eeg_json", "channels_tsv", "events_tsv", "electrodes_tsv"):
        if not paths[name].is_file():
            raise FileNotFoundError(f"source_file_missing:{name}:{paths[name]}")
    source_hashes = {name: _sha256_file(paths[name]) for name in ("vhdr", "vmrk", "eeg_json", "channels_tsv", "events_tsv")}
    for name, actual in source_hashes.items():
        if actual != record["metadata_hashes"][name]:
            raise RuntimeError(f"R1_metadata_hash_changed:{record['participant_id']}:{record['session_id']}:{record['task_id']}:{name}")
    electrode_hash = _sha256_file(paths["electrodes_tsv"])
    if electrode_hash != record["electrodes_metadata"].get("sha256"):
        raise RuntimeError(f"R1_electrodes_hash_changed:{record['participant_id']}:{record['session_id']}")
    header = _parse_vhdr(paths["vhdr"])
    json_sidecar = _read_json(paths["eeg_json"])
    labels_tsv, units, channel_types, channel_rates = _read_channels(paths["channels_tsv"])
    if header["data_file"] != paths["eeg"].name or header["marker_file"] != paths["vmrk"].name:
        raise ValueError("brainvision_data_or_marker_pointer_mismatch")
    if header["data_format"] != "BINARY" or header["data_orientation"] != "MULTIPLEXED" or header["binary_format"] != "IEEE_FLOAT_32":
        raise ValueError("unsupported_brainvision_binary_contract")
    if header["channel_count"] != len(labels_tsv) or header["channel_labels"] != record["ordered_labels_brainvision_header"]:
        raise ValueError("BrainVision_header_changed_since_R1")
    if labels_tsv != record["ordered_labels_channels_tsv"]:
        raise ValueError("channels_tsv_order_changed_since_R1")
    if len(set(label.casefold() for label in labels_tsv)) != len(labels_tsv):
        raise ValueError("ambiguous_casefolded_channel_labels")
    if sorted(label.casefold() for label in header["channel_labels"]) != sorted(label.casefold() for label in labels_tsv):
        raise ValueError("source_header_and_bids_channel_sets_do_not_map")
    if any(unit != "uV" for unit in units):
        raise ValueError("unexpected_channel_units_in_BIDS_channels_tsv")
    if any(kind != "EEG" for kind in channel_types):
        raise ValueError("unexpected_non_EEG_channel_in_study_common_set")
    if any(rate is None or not math.isclose(rate, FS_HZ, rel_tol=0, abs_tol=1e-9) for rate in channel_rates):
        raise ValueError("channel_sidecar_sampling_frequency_mismatch")
    if not math.isclose(float(header["sampling_frequency_hz"]), FS_HZ, rel_tol=0, abs_tol=1e-9):
        raise ValueError("unexpected_header_sample_rate")
    if not math.isclose(float(json_sidecar.get("SamplingFrequency", 0)), FS_HZ, rel_tol=0, abs_tol=1e-9):
        raise ValueError("unexpected_json_sample_rate")
    if str(json_sidecar.get("EEGReference")) != "FCz":
        raise ValueError("declared_reference_changed_since_R1")
    if not math.isclose(float(json_sidecar.get("RecordingDuration", 0)), RECORDING_SECONDS, rel_tol=0, abs_tol=1e-9):
        raise ValueError("declared_recording_duration_changed_since_R1")
    if len(labels_tsv) != CHANNEL_COUNT:
        raise ValueError("unexpected_channel_count_from_actual_header_and_tsv")
    indices = {label.casefold(): index for index, label in enumerate(header["channel_labels"])}
    reorder = [indices[label.casefold()] for label in labels_tsv]
    return paths, header, source_hashes | {"electrodes_tsv": electrode_hash}, labels_tsv, [str(i) for i in reorder]


def _payload_source_record(dataset_root: Path, record: Mapping[str, Any], paths: Mapping[str, Path], header: Mapping[str, Any], labels: Sequence[str], reorder: Sequence[str]) -> dict[str, Any]:
    eeg_path = paths["eeg"]
    stat = eeg_path.stat()
    expected_samples = int(RECORDING_SECONDS * FS_HZ)
    frame_bytes = int(header["channel_count"]) * 4
    expected_bytes = expected_samples * frame_bytes
    byte_count = stat.st_size
    source_hash = _sha256_file(eeg_path)
    canonical_by_casefold = {label.casefold(): label for label in labels}
    header_to_canonical = [
        {"header_label": source_label, "bids_label": canonical_by_casefold[source_label.casefold()], "header_channel_index": index}
        for index, source_label in enumerate(header["channel_labels"])
        if source_label != canonical_by_casefold[source_label.casefold()]
    ]
    mtime_utc = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    if byte_count != int(record["raw_eeg_payload_size_bytes"]) or mtime_utc != record["raw_eeg_payload_mtime_utc"]:
        raise RuntimeError(f"raw_payload_stat_changed_since_R1:{record['participant_id']}:{record['session_id']}:{record['task_id']}")
    metadata_manifest = {
        "relative_payload_path": eeg_path.relative_to(dataset_root).as_posix(),
        "payload_size_bytes": byte_count,
        "payload_mtime_utc": mtime_utc,
        "payload_sha256": source_hash,
        "declared_reference": "FCz",
        "reference_set_hash": None,
        "reference_set_hash_reason": "reference set is not supplied; FCz is not a recorded channel",
        "source_header_labels": list(header["channel_labels"]),
        "study_common_labels": list(labels),
        "header_to_study_common_column_indices": [int(index) for index in reorder],
        "explicit_casefold_aliases": header_to_canonical,
        "source_label_order_hash": _canonical_label_hash(header["channel_labels"], ordered=True),
        "study_common_order_hash": _canonical_label_hash(labels, ordered=True),
        "study_common_set_hash": _canonical_label_hash(labels, ordered=False),
        "source_metadata_hashes": dict(record["metadata_hashes"]),
        "electrodes_tsv_sha256": record["electrodes_metadata"].get("sha256"),
        "metadata_qc": list(record["qc_flags"]),
        "json_channel_count_declared": record["channel_count_json_eeg"],
        "actual_header_and_tsv_channel_count": len(labels),
        "sampling_frequency_hz": FS_HZ,
        "units": "uV (BIDS channels.tsv)",
        "binary_format": header["binary_format"],
        "data_orientation": header["data_orientation"],
        "frame_bytes": frame_bytes,
        "declared_samples": expected_samples,
        "expected_payload_size_bytes": expected_bytes,
        "payload_is_whole_frame": byte_count % frame_bytes == 0,
        "payload_matches_declared_support": byte_count == expected_bytes,
        "payload_read_as_signal": False,
        "qualified": False,
    }
    return metadata_manifest


def _result_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        payload = dict(result)
    elif hasattr(result, "to_dict"):
        payload = result.to_dict()
    elif is_dataclass(result):
        payload = asdict(result)
    else:
        payload = vars(result)
    return _jsonable(payload)


def _unsupported(family: str, reason: str, *, estimator_id: str | None = None) -> dict[str, Any]:
    registry_item = FULL_VECTOR_CHALLENGE_001_BY_FAMILY[family]
    return {
        "family_id": family,
        "estimator_id": registry_item.primary_estimator_id,
        "kernel_estimator_id": estimator_id,
        "computed": False,
        "valid": False,
        "qualified": False,
        "status": "unsupported",
        "reason": reason,
        "value": None,
    }


def _nie(signals: np.ndarray, labels: Sequence[str]) -> Any:
    centered = signals - np.mean(signals, axis=1, keepdims=True, dtype=np.float64)
    norms = np.sqrt(np.sum(centered * centered, axis=1, dtype=np.float64))
    if np.any(norms <= 0.0) or not np.isfinite(norms).all():
        weights = np.full((signals.shape[0], signals.shape[0]), np.nan, dtype=np.float64)
    else:
        correlation = (centered @ centered.T) / np.outer(norms, norms)
        correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
        weights = np.abs(correlation)
        np.fill_diagonal(weights, 0.0)
    order_hash = _canonical_label_hash(labels, ordered=True)
    return compute_nie_raw_weighted_global_efficiency(
        weights, labels, order_hash, NIE_EDGE_ESTIMATOR_ID, NIE_EDGE_RULE_ID
    )


def _compute_families(signals: np.ndarray, labels: Sequence[str], *, participant: str, session: str, task: str,
                      source_sha256: str, start_sample: int, end_sample: int, duration: int) -> dict[str, dict[str, Any]]:
    fs = FS_HZ
    results: dict[str, Any] = {}
    try:
        alpha_sos = signal.butter(4, (8.0, 13.0), btype="bandpass", fs=fs, output="sos")
        alpha = signal.sosfiltfilt(alpha_sos, signals, axis=-1, padtype=None, padlen=0)
    except Exception as exc:
        alpha = None
        alpha_failure = f"alpha_filter_failed:{type(exc).__name__}:{exc}"
    if alpha is not None:
        try:
            analytic = hilbert(alpha, axis=-1)
            amplitude = np.abs(analytic)
            if not np.isfinite(amplitude).all():
                raise ValueError("non_finite_analytic_amplitude")
            if np.any(amplitude == 0.0):
                raise ValueError("zero_analytic_amplitude_phase_undefined")
            phase = np.angle(analytic)
            results["PCV"] = compute_pcv_phase_order_parameter_sd(
                phase, phase_estimator_id=PCV_PHASE_ID, band_id=PCV_BAND_ID, channel_labels=labels
            )
        except Exception as exc:
            results["PCV"] = _unsupported("PCV", f"alpha_phase_failed:{type(exc).__name__}:{exc}")
        try:
            results["TP"] = compute_tp_log_rms_envelope_persistence(
                alpha, labels, sample_rate_hz=fs, envelope_samples=500, lag_samples=500,
                epsilon_amplitude=1e-12, minimum_pair_count=20,
            )
        except Exception as exc:
            results["TP"] = _unsupported("TP", f"tp_failed:{type(exc).__name__}:{exc}")
    else:
        results["PCV"] = _unsupported("PCV", alpha_failure)
        results["TP"] = _unsupported("TP", alpha_failure)

    try:
        block_samples = int(2 * fs)
        if signals.shape[1] % block_samples:
            raise ValueError("window_not_divisible_into_complete_2s_AR_blocks")
        block_vectors = np.asarray([
            np.sqrt(np.mean(signals[:, left:left + block_samples] ** 2, axis=1, dtype=np.float64))
            for left in range(0, signals.shape[1], block_samples)
        ], dtype=np.float64)
        start_seconds = start_sample / fs
        intervals = [
            (start_seconds + index * 2.0, start_seconds + (index + 1) * 2.0)
            for index in range(block_vectors.shape[0])
        ]
        results["AR"] = compute_ar_activity_transition_vector(
            block_vectors, intervals, vector_construction_id=AR_VECTOR_IDS[duration],
            scaling_id=AR_SCALING_ID, channel_labels=labels,
        )
    except Exception as exc:
        results["AR"] = _unsupported("AR", f"ar_failed:{type(exc).__name__}:{exc}")
    try:
        results["NIE"] = _nie(signals, labels)
    except Exception as exc:
        results["NIE"] = _unsupported("NIE", f"nie_failed:{type(exc).__name__}:{exc}")
    try:
        results["RD"] = compute_rd_centered_log_psd_dispersion(
            signals, labels, sample_rate_hz=fs, welch_segment_seconds=4.0,
            frequency_min_hz=1.0, frequency_max_hz=45.0,
        )
    except Exception as exc:
        results["RD"] = _unsupported("RD", f"rd_failed:{type(exc).__name__}:{exc}")
    try:
        results["OI"] = compute_oi_matrix(signals, labels)
    except Exception as exc:
        results["OI"] = _unsupported("OI", f"oi_failed:{type(exc).__name__}:{exc}")
    try:
        segment_id = f"{participant}|{session}|{task}|{duration}s|{start_sample}:{end_sample}"
        results["SC"] = compute_sc_lz78_exact_multiset_ratio(
            signals, labels,
            source_identity=f"{DATASET_ID}|{participant}|{session}|{task}|{source_sha256}",
            segment_id=segment_id,
            start_seconds=start_sample / fs,
            end_seconds=end_sample / fs,
            reference_count=SC_REFERENCE_COUNT,
            master_seed=SC_MASTER_SEED,
        )
    except Exception as exc:
        results["SC"] = _unsupported("SC", f"sc_failed:{type(exc).__name__}:{exc}")
    try:
        results["ED"] = compute_ed_raw_sensor_covariance_participation_ratio(signals, labels)
    except Exception as exc:
        results["ED"] = _unsupported("ED", f"ed_failed:{type(exc).__name__}:{exc}")

    output: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        value = _result_dict(results[family])
        registry_item = FULL_VECTOR_CHALLENGE_001_BY_FAMILY[family]
        kernel_estimator_id = value.get("estimator_id", getattr(results[family], "estimator_id", None))
        value["family_id"] = family
        value["estimator_id"] = registry_item.primary_estimator_id
        value["kernel_estimator_id"] = kernel_estimator_id
        value["qualified"] = False
        value["realization_id"] = SENSOR_REALIZATION_ID
        output[family] = value
    return output


def _band_integral(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> np.ndarray:
    keep = (freqs >= low) & (freqs <= high)
    if int(np.count_nonzero(keep)) < 2:
        raise ValueError("baseline_band_has_fewer_than_two_frequency_bins")
    return trapezoid(psd[..., keep], freqs[keep], axis=-1)


def _baseline(signals: np.ndarray, fs: float) -> tuple[dict[str, float], dict[str, str]]:
    values: dict[str, float] = {}
    reasons: dict[str, str] = {}
    baseline_ids = {
        "log10_median_channel_rms_uV": "SL020.BASELINE.log10_median_channel_rms_uV.v1",
        "relative_delta_1_4hz": "SL020.BASELINE.relative_band_power_1_45_welch.v1",
        "relative_theta_4_8hz": "SL020.BASELINE.relative_band_power_1_45_welch.v1",
        "relative_alpha_8_13hz": "SL020.BASELINE.relative_band_power_1_45_welch.v1",
        "relative_beta_13_30hz": "SL020.BASELINE.relative_band_power_1_45_welch.v1",
        "relative_gamma_30_45hz": "SL020.BASELINE.relative_band_power_1_45_welch.v1",
        "normalized_spectral_entropy_1_45hz": "SL020.BASELINE.normalized_spectral_entropy_1_45_welch.v1",
        "median_channel_autocorrelation_lag_1s": "SL020.BASELINE.median_channel_pearson_autocorrelation_lag_1s.v1",
        "mean_absolute_zero_lag_channel_correlation": "SL020.BASELINE.mean_absolute_zero_lag_channel_correlation.v1",
    }
    try:
        if not np.isfinite(signals).all():
            raise ValueError("non_finite_required_channel_sample")
        rms = np.sqrt(np.mean(signals * signals, axis=1, dtype=np.float64))
        if np.any(~np.isfinite(rms)) or np.any(rms <= 0.0):
            raise ValueError("nonpositive_channel_rms")
        values["log10_median_channel_rms_uV"] = float(np.median(np.log10(rms)))
    except Exception as exc:
        reasons["log10_median_channel_rms_uV"] = f"baseline_rms_unsupported:{type(exc).__name__}:{exc}"
    try:
        freqs, psd = signal.welch(
            signals, fs=fs, window="hann", nperseg=2000, noverlap=1000,
            detrend="constant", scaling="density", axis=-1,
        )
        total = _band_integral(freqs, psd, 1.0, 45.0)
        if np.any(~np.isfinite(total)) or np.any(total <= 0.0) or np.any(~np.isfinite(psd)) or np.any(psd < 0.0):
            raise ValueError("invalid_total_1_45hz_psd_support")
        bands = {
            "relative_delta_1_4hz": (1.0, 4.0),
            "relative_theta_4_8hz": (4.0, 8.0),
            "relative_alpha_8_13hz": (8.0, 13.0),
            "relative_beta_13_30hz": (13.0, 30.0),
            "relative_gamma_30_45hz": (30.0, 45.0),
        }
        for name, (low, high) in bands.items():
            ratios = _band_integral(freqs, psd, low, high) / total
            if np.isfinite(ratios).all() and np.all(ratios >= 0.0):
                values[name] = float(np.median(ratios))
            else:
                reasons[name] = "invalid_relative_band_power"
        keep = (freqs >= 1.0) & (freqs <= 45.0)
        entropy_psd = psd[:, keep]
        sums = np.sum(entropy_psd, axis=1, dtype=np.float64)
        if entropy_psd.shape[1] < 2 or np.any(sums <= 0.0) or np.any(~np.isfinite(sums)):
            reasons["normalized_spectral_entropy_1_45hz"] = "invalid_spectral_entropy_support"
        else:
            probabilities = entropy_psd / sums[:, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                entropy = -np.sum(np.where(probabilities > 0.0, probabilities * np.log(probabilities), 0.0), axis=1) / math.log(entropy_psd.shape[1])
            if np.isfinite(entropy).all():
                values["normalized_spectral_entropy_1_45hz"] = float(np.median(entropy))
            else:
                reasons["normalized_spectral_entropy_1_45hz"] = "nonfinite_spectral_entropy"
    except Exception as exc:
        reason = f"welch_baseline_failed:{type(exc).__name__}:{exc}"
        for key in ("relative_delta_1_4hz", "relative_theta_4_8hz", "relative_alpha_8_13hz", "relative_beta_13_30hz", "relative_gamma_30_45hz", "normalized_spectral_entropy_1_45hz"):
            reasons.setdefault(key, reason)
    try:
        lag = int(fs)
        left, right = signals[:, :-lag], signals[:, lag:]
        left = left - np.mean(left, axis=1, keepdims=True, dtype=np.float64)
        right = right - np.mean(right, axis=1, keepdims=True, dtype=np.float64)
        denominator = np.sqrt(np.sum(left * left, axis=1, dtype=np.float64) * np.sum(right * right, axis=1, dtype=np.float64))
        if np.any(denominator <= 0.0) or not np.isfinite(denominator).all():
            raise ValueError("constant_or_nonfinite_channel_at_lag_1s")
        correlations = np.sum(left * right, axis=1, dtype=np.float64) / denominator
        if not np.isfinite(correlations).all():
            raise ValueError("nonfinite_lag_1s_autocorrelation")
        values["median_channel_autocorrelation_lag_1s"] = float(np.median(correlations))
    except Exception as exc:
        reasons["median_channel_autocorrelation_lag_1s"] = f"autocorrelation_unsupported:{type(exc).__name__}:{exc}"
    try:
        centered = signals - np.mean(signals, axis=1, keepdims=True, dtype=np.float64)
        norms = np.sqrt(np.sum(centered * centered, axis=1, dtype=np.float64))
        if np.any(norms <= 0.0) or not np.isfinite(norms).all():
            raise ValueError("constant_or_nonfinite_channel")
        corr = (centered @ centered.T) / np.outer(norms, norms)
        corr = np.clip((corr + corr.T) / 2.0, -1.0, 1.0)
        upper = corr[np.triu_indices(corr.shape[0], k=1)]
        if not np.isfinite(upper).all() or upper.size == 0:
            raise ValueError("invalid_interchannel_pairs")
        values["mean_absolute_zero_lag_channel_correlation"] = float(np.mean(np.abs(upper), dtype=np.float64))
    except Exception as exc:
        reasons["mean_absolute_zero_lag_channel_correlation"] = f"channel_correlation_unsupported:{type(exc).__name__}:{exc}"
    # The IDs travel with the values, without turning a missing metric into zero.
    values = {key: value for key, value in values.items() if math.isfinite(value)}
    reasons = {key: value for key, value in reasons.items() if key not in values}
    baseline_results = {
        key: {"estimator_id": baseline_ids[key], "value": value, "computed": True,
              "valid": True, "qualified": False, "status": "computed_exploratory", "reason": None}
        for key, value in values.items()
    }
    return baseline_results, reasons


def _window_rows(record: Mapping[str, Any], source: Mapping[str, Any], signals: np.ndarray, labels: Sequence[str],
                 duration: int, start: int, end: int) -> tuple[list[dict[str, Any]], dict[str, Any], Counter[str]]:
    window_id = f"{duration:02d}s-{start:06d}-{end:06d}"
    families = _compute_families(
        signals, labels, participant=str(record["participant_id"]), session=str(record["session_id"]),
        task=str(record["task_id"]), source_sha256=str(source["payload_sha256"]),
        start_sample=start, end_sample=end, duration=duration,
    )
    base = {
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "participant_id": record["participant_id"],
        "session_id": record["session_id"],
        "condition": record["task_id"],
        "recording_id": record["bids_stem"],
        "source_payload_sha256": source["payload_sha256"],
        "source_metadata_manifest_sha256": hashlib.sha256(json.dumps(source["source_metadata_hashes"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "realization_id": SENSOR_REALIZATION_ID,
        "channel_labels": list(labels),
        "channel_count": len(labels),
        "channel_order_sha256": source["study_common_order_hash"],
        "channel_set_sha256": source["study_common_set_hash"],
        "reference_id": SOURCE_REFERENCE_ID,
        "reference_set_sha256": None,
        "sampling_frequency_hz": FS_HZ,
        "units": "uV",
        "window_id": window_id,
        "duration_seconds": duration,
        "start_sample": start,
        "end_sample_exclusive": end,
        "start_seconds": start / FS_HZ,
        "end_seconds": end / FS_HZ,
        "support_status": "known_from_complete_brainvision_file",
        "dependency_start_sample": start,
        "dependency_end_sample_exclusive": end,
        "qualified": False,
        "clinical_metadata_read": False,
    }
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for family in FAMILIES:
        result = families[family]
        row = {"record_type": "measurement", **base, **result, "family_id": family}
        rows.append(_jsonable(row))
        counts[f"family:{family}:requested"] += 1
        counts[f"family:{family}:status:{result.get('status', 'unknown')}"] += 1
        counts[f"family:{family}:valid:{bool(result.get('valid'))}"] += 1
    baseline_values, baseline_reasons = _baseline(signals, FS_HZ)
    for name in (
        "log10_median_channel_rms_uV", "relative_delta_1_4hz", "relative_theta_4_8hz",
        "relative_alpha_8_13hz", "relative_beta_13_30hz", "relative_gamma_30_45hz",
        "normalized_spectral_entropy_1_45hz", "median_channel_autocorrelation_lag_1s",
        "mean_absolute_zero_lag_channel_correlation",
    ):
        result = baseline_values.get(name)
        if result is None:
            result = {"value": None, "computed": False, "valid": False, "qualified": False,
                      "status": "unsupported", "reason": baseline_reasons.get(name, "baseline_not_computed")}
        rows.append(_jsonable({"record_type": "qEEG_baseline", **base, "baseline_feature_id": name, **result}))
        counts[f"baseline:{name}:status:{result.get('status', 'unknown')}"] += 1
    return rows, {"window_id": window_id, "duration_seconds": duration, "start_sample": start, "end_sample_exclusive": end}, counts


def _write_gzip_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    if temp.exists():
        raise RuntimeError(f"partial_output_shard_exists:{temp}")
    with gzip.open(temp, "wt", encoding="utf-8", newline="\n", compresslevel=6) as stream:
        for row in rows:
            stream.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    # Verify gzip trailer/CRC without inspecting or summarizing feature values.
    with gzip.open(temp, "rt", encoding="utf-8") as stream:
        for _ in stream:
            pass
    digest = _sha256_file(temp)
    if path.exists():
        raise FileExistsError(f"refusing_to_overwrite_completed_shard:{path}")
    os.replace(temp, path)
    return digest


def _atomic_manifest(path: Path, manifest: dict[str, Any]) -> None:
    _write_json_atomic(path, manifest)


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _code_hashes() -> dict[str, str]:
    module_paths = {
        "runner": Path(__file__).resolve(),
        "contracts": SRC_ROOT / "mndm" / "mnps" / "v3_2" / "contracts.py",
        "ar": SRC_ROOT / "mndm" / "mnps" / "v3_2" / "ar.py",
        "ed": SRC_ROOT / "mndm" / "mnps" / "v3_2" / "ed.py",
        "nie": SRC_ROOT / "mndm" / "mnps" / "v3_2" / "nie.py",
        "oi": SRC_ROOT / "mndm" / "mnps" / "v3_2" / "oi.py",
        "pcv": SRC_ROOT / "mndm" / "mnps" / "v3_2" / "pcv.py",
        "rd": SRC_ROOT / "mndm" / "mnps" / "v3_2" / "rd.py",
        "registry": SRC_ROOT / "mndm" / "mnps" / "v3_2" / "registry.py",
        "sc": SRC_ROOT / "mndm" / "mnps" / "v3_2" / "sc.py",
        "tp": SRC_ROOT / "mndm" / "mnps" / "v3_2" / "tp.py",
    }
    return {name: _sha256_file(path) for name, path in module_paths.items()}


def _prepare_phase(output_root: Path, *, identity: Mapping[str, Any]) -> dict[str, Any]:
    output_root = output_root.resolve()
    if output_root.drive.upper() != "J:":
        raise ValueError("SL020_analysis_outputs_must_be_on_J_drive")
    if output_root.exists() and any(output_root.iterdir()):
        manifest_path = output_root / "run_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"nonempty_phase_root_without_manifest:{output_root}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_identity = existing.get("frozen_identity", {})
        if _jsonable(existing_identity) != _jsonable(identity):
            raise RuntimeError("resume_identity_mismatch_refusing_existing_phase_root")
        for participant, shard_info in existing.get("completed_participants", {}).items():
            shard = output_root / str(shard_info["relative_path"])
            if not shard.is_file() or _sha256_file(shard) != shard_info.get("sha256"):
                raise RuntimeError(f"completed_shard_checksum_mismatch:{participant}")
        if list(output_root.rglob("*.tmp")):
            raise RuntimeError("partial_shard_present_abort_and_use_new_phase_root")
        return existing
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "audit_id": AUDIT_ID,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_identity": _jsonable(identity),
        "completed_participants": {},
        "support_counts": {},
        "source_recording_manifest": [],
        "raw_signal_samples_read": True,
        "clinical_metadata_read": False,
        "feature_values_printed_to_stdout": False,
        "qualified": False,
    }
    _atomic_manifest(output_root / "run_manifest.json", manifest)
    return manifest


def _process_recording(dataset_root: Path, record: Mapping[str, Any], r1_record: Mapping[str, Any],
                       *, selected_for_smoke: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any], Counter[str]]:
    paths, header, metadata_hashes, labels, reorder_strings = _verify_source_metadata(dataset_root, r1_record)
    source = _payload_source_record(dataset_root, r1_record, paths, header, labels, reorder_strings)
    source["participant_id"] = record["participant_id"]
    source["session_id"] = record["session_id"]
    source["task_id"] = record["task_id"]
    source["recording_id"] = record["bids_stem"]
    source["source_metadata_hashes"] = metadata_hashes
    file_rows: list[dict[str, Any]] = [{"record_type": "recording_manifest", **source, "dataset_id": DATASET_ID,
                                       "protocol_id": PROTOCOL_ID, "sensor_realization_id": SENSOR_REALIZATION_ID,
                                       "reference_id": SOURCE_REFERENCE_ID, "reference_set_hash": None}]
    counts: Counter[str] = Counter()
    if not source["payload_matches_declared_support"] or not source["payload_is_whole_frame"]:
        reason = "payload_size_not_whole_sample_frame" if not source["payload_is_whole_frame"] else "payload_size_differs_from_declared_duration"
        for duration in WINDOW_DURATIONS:
            for family in FAMILIES:
                result = _unsupported(family, reason)
                file_rows.append({
                    "record_type": "measurement", "protocol_id": PROTOCOL_ID, "dataset_id": DATASET_ID,
                    "participant_id": record["participant_id"], "session_id": record["session_id"],
                    "condition": record["task_id"], "recording_id": record["bids_stem"],
                    "source_payload_sha256": source["payload_sha256"], "realization_id": SENSOR_REALIZATION_ID,
                    "reference_id": SOURCE_REFERENCE_ID, "reference_set_sha256": None,
                    "duration_seconds": duration, "window_id": None,
                    "window_count_requested": None, "support_status": "unsupported_file_source",
                    **result,
                })
                counts[f"family:{family}:requested"] += 1
                counts[f"family:{family}:status:unsupported_file_source"] += 1
                counts[f"family:{family}:valid:False"] += 1
        counts["recording:unsupported_source"] += 1
        return file_rows, source, counts

    sample_count = int(source["declared_samples"])
    if sample_count != int(RECORDING_SECONDS * FS_HZ):
        raise RuntimeError("declared_sample_count_changed_from_frozen_protocol")
    if source["payload_size_bytes"] != int(sample_count * CHANNEL_COUNT * 4):
        raise RuntimeError("payload_byte_count_changed_after_source_preflight")
    payload = np.memmap(paths["eeg"], dtype="<f4", mode="r", shape=(sample_count, CHANNEL_COUNT), order="C")
    channel_indices = np.asarray([int(index) for index in reorder_strings], dtype=np.int64)
    source["payload_read_as_signal"] = True
    source["sample_count"] = sample_count
    source["actual_source_start_sample"] = 0
    source["actual_source_end_sample_exclusive"] = sample_count
    file_rows[0].update(source)
    for duration in WINDOW_DURATIONS:
        window_samples = int(duration * FS_HZ)
        for start in range(0, sample_count - window_samples + 1, window_samples):
            end = start + window_samples
            signals = np.asarray(payload[start:end, :][:, channel_indices].T, dtype=np.float64)
            rows, _window, window_counts = _window_rows(record, source, signals, labels, duration, start, end)
            file_rows.extend(rows)
            counts.update(window_counts)
    del payload
    counts["recording:processed"] += 1
    counts["window:requested_4s"] += sample_count // int(4 * FS_HZ)
    counts["window:requested_20s"] += sample_count // int(20 * FS_HZ)
    return file_rows, source, counts


def run(mode: str, dataset_root: Path, r1_root: Path, output_root: Path) -> dict[str, Any]:
    dataset_root = dataset_root.resolve(strict=True)
    r1_root = r1_root.resolve(strict=True)
    output_root = output_root.resolve()
    r1_manifest_path = r1_root / "metadata_audit.json"
    if not r1_manifest_path.is_file():
        raise FileNotFoundError(f"R1_metadata_manifest_missing:{r1_manifest_path}")
    r1_sha = _sha256_file(r1_manifest_path)
    r1_manifest = json.loads(r1_manifest_path.read_text(encoding="utf-8"))
    if r1_manifest.get("audit_id") != AUDIT_ID or r1_manifest.get("source_root") != str(dataset_root):
        raise RuntimeError("R1_dataset_or_audit_identity_mismatch")
    protocol_sha = _read_frozen_hash(PROTOCOL_HASH_PATH, PROTOCOL_PATH)
    amendment_sha = _read_frozen_hash(AMENDMENT_HASH_PATH, AMENDMENT_PATH)
    code_hashes = _code_hashes()
    audit_records = r1_manifest.get("recordings", [])
    by_key = {(r["participant_id"], r["session_id"], r["task_id"]): r for r in audit_records}
    if len(by_key) != 900:
        raise RuntimeError(f"unexpected_R1_recording_count:{len(by_key)}")
    if mode == "smoke":
        selected_participants = [SMOKE_RECORDING[0]]
        selected_keys = [SMOKE_RECORDING]
    elif mode == "pilot":
        selected_participants = list(PILOT_IDS)
        selected_keys = [key for key in by_key if key[0] in set(PILOT_IDS)]
    elif mode == "cohort":
        selected_participants = sorted({key[0] for key in by_key})
        selected_keys = list(by_key)
    else:
        raise ValueError(f"unknown_mode:{mode}")
    selected_keys = sorted(selected_keys, key=lambda key: (selected_participants.index(key[0]), SESSIONS.index(key[1]), TASKS.index(key[2])))
    selected_records = [by_key[key] for key in selected_keys]
    if mode == "smoke" and len(selected_records) != 1:
        raise RuntimeError("smoke_roster_did_not_resolve_to_one_recording")
    if mode == "pilot" and len(selected_records) != 180:
        raise RuntimeError(f"pilot_recording_count_mismatch:{len(selected_records)}")

    identity = {
        "mode": mode,
        "dataset_id": DATASET_ID,
        "dataset_version": r1_manifest.get("dataset_description", {}).get("DatasetDOI"),
        "source_root": str(dataset_root),
        "r1_audit_sha256": r1_sha,
        "protocol_sha256": protocol_sha,
        "amendment_sha256": amendment_sha,
        "code_hashes": code_hashes,
        "selected_participants": selected_participants,
        "selected_recording_keys": [list(key) for key in selected_keys],
        "sensor_realization_id": SENSOR_REALIZATION_ID,
        "reference_id": SOURCE_REFERENCE_ID,
        "window_durations_seconds": list(WINDOW_DURATIONS),
        "reference_count_sc": SC_REFERENCE_COUNT,
        "master_seed_sc": SC_MASTER_SEED,
    }
    output_root.mkdir(parents=True, exist_ok=True) if not output_root.exists() else None
    manifest = _prepare_phase(output_root, identity=identity)
    support_counts = Counter(manifest.get("support_counts", {}))
    completed = manifest.get("completed_participants", {})
    r1_by_participant: dict[str, list[dict[str, Any]]] = {participant: [] for participant in selected_participants}
    record_by_participant: dict[str, list[dict[str, Any]]] = {participant: [] for participant in selected_participants}
    for key in selected_keys:
        r1_by_participant[key[0]].append(by_key[key])
    for record in selected_records:
        record_by_participant[str(record["participant_id"])].append(record)

    started = time.monotonic()
    for participant in selected_participants:
        if participant in completed:
            continue
        free = shutil.disk_usage(output_root).free
        if free < MIN_FREE_BYTES:
            raise RuntimeError(f"disk_guard_below_10_GiB:{free}")
        print(f"SL020 {mode}: processing {participant}", flush=True)
        participant_rows: list[dict[str, Any]] = []
        participant_sources: list[dict[str, Any]] = []
        participant_counts: Counter[str] = Counter()
        for record, r1_record in zip(record_by_participant[participant], r1_by_participant[participant], strict=True):
            print(f"SL020 {mode}: source {record['session_id']}/{record['task_id']}", flush=True)
            rows, source, counts = _process_recording(dataset_root, record, r1_record)
            participant_rows.extend(rows)
            participant_sources.append({
                "participant_id": participant, "session_id": record["session_id"], "task_id": record["task_id"],
                "relative_payload_path": source["relative_payload_path"], "payload_size_bytes": source["payload_size_bytes"],
                "payload_sha256": source["payload_sha256"], "payload_matches_declared_support": source["payload_matches_declared_support"],
                "metadata_hashes": source["source_metadata_hashes"],
            })
            participant_counts.update(counts)
        shard_path = output_root / "participants" / f"{participant}.jsonl.gz"
        shard_sha = _write_gzip_jsonl_atomic(shard_path, participant_rows)
        shard_bytes = shard_path.stat().st_size
        completed[participant] = {
            "relative_path": shard_path.relative_to(output_root).as_posix(),
            "sha256": shard_sha,
            "size_bytes": shard_bytes,
            "record_count": len(participant_rows),
            "recording_count": len(record_by_participant[participant]),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "support_counts": dict(sorted(participant_counts.items())),
            "source_recordings": participant_sources,
        }
        support_counts.update(participant_counts)
        manifest["completed_participants"] = completed
        manifest["support_counts"] = dict(sorted(support_counts.items()))
        manifest["source_recording_manifest"].extend(participant_sources)
        manifest["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["elapsed_seconds"] = round(time.monotonic() - started, 3)
        _atomic_manifest(output_root / "run_manifest.json", manifest)
        print(f"SL020 {mode}: completed {participant}; shard={shard_bytes:,} bytes; support/status counts only", flush=True)

    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["elapsed_seconds"] = round(time.monotonic() - started, 3)
    _atomic_manifest(output_root / "run_manifest.json", manifest)
    support_report = {
        "protocol_id": PROTOCOL_ID,
        "mode": mode,
        "status": manifest["status"],
        "selected_participant_count": len(selected_participants),
        "selected_recording_count": len(selected_records),
        "completed_participant_count": len(manifest["completed_participants"]),
        "support_counts": manifest["support_counts"],
        "elapsed_seconds": manifest["elapsed_seconds"],
        "output_bytes": sum(item["size_bytes"] for item in manifest["completed_participants"].values()),
        "free_bytes_after_run": shutil.disk_usage(output_root).free,
        "feature_values_inspected_by_gate": False,
        "qualified": False,
    }
    _write_json_atomic(output_root / "engineering_support_report.json", support_report)
    print(json.dumps(support_report, indent=2, ensure_ascii=False), flush=True)
    return support_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pilot", "cohort"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--r1-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.mode, args.dataset_root, args.r1_root, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
