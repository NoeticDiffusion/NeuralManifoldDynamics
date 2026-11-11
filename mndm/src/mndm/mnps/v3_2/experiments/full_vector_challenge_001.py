"""Outcome-blind Gate A/B runner for FULL-VECTOR-CHALLENGE-001.

This module reads the local BIDS ds003478 EEG files directly. It does not load
participants.tsv or any diagnosis/BDI field. Selection is frozen from technical
metadata before any signal window is read; the smoke and pilot use the same
selection manifest.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import io as scipy_io
from scipy import signal
from scipy.signal import hilbert

from mndm.mnps.v3_1.candidates import MNPS31WindowSupport
from mndm.mnps.v3_1.d2_spatial_gradient import compute_d2_spatial_coupling_gradient
from mndm.mnps.v3_1.eeg_metrics import _lz78_phrase_count_binary
from mndm.mnps.v3_2.ar import compute_ar_activity_transition_vector
from mndm.mnps.v3_2.candidate_grid import (
    CandidateWindow,
    enumerate_full_windows,
    reconstruct_condition_blocks,
)
from mndm.mnps.v3_2.ed import compute_ed_raw_sensor_covariance_participation_ratio
from mndm.mnps.v3_2.nie import compute_nie_raw_weighted_global_efficiency
from mndm.mnps.v3_2.oi import compute_oi_matrix
from mndm.mnps.v3_2.pcv import compute_pcv_phase_order_parameter_sd
from mndm.mnps.v3_2.rd import compute_rd_centered_log_psd_dispersion
from mndm.mnps.v3_2.registry import FULL_VECTOR_CHALLENGE_001_BY_FAMILY
from mndm.mnps.v3_2.sc import compute_sc_lz78_exact_multiset_ratio
from mndm.mnps.v3_2.tp import (
    compute_tp_log_rms_envelope_persistence,
    compute_tp_raw_signal_persistence,
)


PROTOCOL_ID = "FULL-VECTOR-CHALLENGE-001"
SELECTION_DATE = "20261014"
DEFAULT_DATA_ROOT = Path(r"K:\ExternalReceivedDatasets\openneuro\ds003478")
DEFAULT_SELECTION_PATH = Path(__file__).parents[5] / "config" / (
    f"mnps_v3_2_full_vector_challenge_001_selection_{SELECTION_DATE}_r2.json"
)
DEFAULT_OUTPUT_ROOT = Path.cwd() / "outputs" / PROTOCOL_ID / "GATE-B"
EXPECTED_FS_HZ = 500.0
WINDOW_SECONDS = 20.0
SCG_WINDOW_SECONDS = 60.0
EEG_EXCLUDED_LABELS = {"heog", "veog", "ekg", "ecg"}
PCV_PIPELINE_ESTIMATOR_ID = "PCV.pipeline_native_window_alpha_phase_order_parameter_sd.v1"
PCV_PHASE_ESTIMATOR_ID = "scipy.signal.hilbert_after_sosfiltfilt.v1"
PCV_BAND_ID = "butterworth4_zero_phase_8_13hz_per_window.v1"
TP_BAND_ID = "butterworth4_zero_phase_8_13hz_per_window.v1"
AR_VECTOR_CONSTRUCTION_ID = "per_channel_rms_ten_contiguous_2s_blocks.v1"
AR_SCALING_ID = "raw_numeric_amplitude_rms.v1"
NIE_EDGE_ESTIMATOR_ID = "pearson_zero_lag_sensor_pairs.v1"
NIE_EDGE_RULE_ID = "absolute_correlation_no_threshold_exact_zero_diagonal.v1"
SCG_SURFACE_ID = "ds003478_native_declared_64_eeg_bids_average_reference.v1"
SCG_PREPROCESSING_ID = "instantaneous_average_reference_exact_bids_eeg_surface_no_other_processing.v1"
SCG_REFERENCE_ID = "average_over_exact_ds003478_declared_64_eeg_surface.v1"
AVERAGE_REFERENCE_ID = "average_over_exact_native_declared_EEG_set.v1"
CORE_FAMILIES = (
    "PCV", "TP", "TP_RAW_COMPARATOR", "AR", "NIE", "RD", "OI", "SC", "ED",
)
CORE_REFERENCE_IDS = (
    "acquisition_reference_from_BIDS_sidecar", AVERAGE_REFERENCE_ID,
)


@dataclass(frozen=True)
class RunMetadata:
    participant_id: str
    run_id: str
    set_path: str
    fdt_path: str
    events_path: str
    channels_path: str
    eeg_json_path: str
    electrodes_path: str
    coordsystem_path: str
    electrodes_sha256: str | None
    coordsystem_sha256: str | None
    sample_rate_hz: float
    sample_count: int
    channel_count: int
    trial_count: int
    channel_labels: tuple[str, ...]
    eeg_channel_labels: tuple[str, ...]
    metadata_qc: tuple[str, ...]
    eligible_conditions: tuple[str, ...]
    first_windows: tuple[tuple[str, CandidateWindow], ...]
    scg_windows: tuple[tuple[str, CandidateWindow | None], ...]
    set_sha256: str
    events_sha256: str
    channels_sha256: str
    eeg_json_sha256: str

    def to_selection_record(self) -> dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "run_id": self.run_id,
            "set_path": self.set_path,
            "fdt_path": self.fdt_path,
            "events_path": self.events_path,
            "channels_path": self.channels_path,
            "eeg_json_path": self.eeg_json_path,
            "electrodes_path": self.electrodes_path,
            "coordsystem_path": self.coordsystem_path,
            "electrodes_sha256": self.electrodes_sha256,
            "coordsystem_sha256": self.coordsystem_sha256,
            "sample_rate_hz": self.sample_rate_hz,
            "sample_count": self.sample_count,
            "channel_count": self.channel_count,
            "trial_count": self.trial_count,
            "channel_labels": list(self.channel_labels),
            "eeg_channel_labels": list(self.eeg_channel_labels),
            "metadata_qc": list(self.metadata_qc),
            "eligible_conditions": list(self.eligible_conditions),
            "first_windows": {condition: window.to_dict() for condition, window in self.first_windows},
            "scg_windows": {
                condition: window.to_dict() if window is not None else None
                for condition, window in self.scg_windows
            },
            "metadata_sha256": {
                "set_header": self.set_sha256,
                "events": self.events_sha256,
                "channels": self.channels_sha256,
                "eeg_json": self.eeg_json_sha256,
                "electrodes": self.electrodes_sha256,
                "coordsystem": self.coordsystem_sha256,
            },
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_labels(labels: Sequence[str], *, ordered: bool) -> str:
    value: Sequence[str] = tuple(labels)
    if not ordered:
        value = sorted(labels)
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reference_provenance(
    labels: Sequence[str], reference_id: str, acquisition_reference: str
) -> tuple[str, str, str | None]:
    """Keep the input sensor realization separate from the applied reference."""
    source_reference_hash = hashlib.sha256(acquisition_reference.encode("utf-8")).hexdigest()
    if reference_id == "acquisition_reference_from_BIDS_sidecar":
        reference_set_hash = None  # The source text is not an asserted electrode set.
    else:
        reference_set_hash = _hash_labels(labels, ordered=False)
    record = {
        "reference_id": reference_id,
        "reference_set_sha256": reference_set_hash,
        "source_reference_metadata_sha256": source_reference_hash,
        "input_channel_order_sha256": _hash_labels(labels, ordered=True),
    }
    realization_hash = hashlib.sha256(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return realization_hash, source_reference_hash, reference_set_hash


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _mat_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(-1)[0]
    return value


def _mat_text(value: Any) -> str:
    value = _mat_scalar(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip("\x00 ")
    if isinstance(value, np.ndarray) and value.dtype.kind in {"U", "S"}:
        return "".join(str(item) for item in value.reshape(-1)).strip("\x00 ")
    return str(value).strip("\x00 ")


def _get_mat_field(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key)
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, np.void) and obj.dtype.names and key in obj.dtype.names:
        return obj[key]
    return None


def _set_header(set_path: Path) -> tuple[int, int, int, float, tuple[str, ...], str]:
    mat = scipy_io.loadmat(str(set_path), variable_names=["EEG"], simplify_cells=True)
    eeg = mat.get("EEG")
    if eeg is None:
        raise ValueError("EEGLAB_set_header_missing_EEG_struct")
    try:
        channels = int(float(_mat_scalar(_get_mat_field(eeg, "nbchan"))))
        samples = int(float(_mat_scalar(_get_mat_field(eeg, "pnts"))))
        trials = int(float(_mat_scalar(_get_mat_field(eeg, "trials"))))
        sample_rate = float(_mat_scalar(_get_mat_field(eeg, "srate")))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid_EEGLAB_set_header_dimensions") from exc
    raw_data_name = _mat_text(_get_mat_field(eeg, "data"))
    if not raw_data_name or "in set file" in raw_data_name.casefold():
        raise ValueError("external_EEGLAB_fdt_reference_missing")
    if Path(raw_data_name).name != raw_data_name or not raw_data_name.casefold().endswith(".fdt"):
        raise ValueError("unsafe_or_nonlocal_EEGLAB_fdt_reference")
    if Path(raw_data_name).name.casefold() != set_path.with_suffix(".fdt").name.casefold():
        raise ValueError("EEGLAB_fdt_reference_does_not_match_bids_sibling")
    raw_locs = _get_mat_field(eeg, "chanlocs")
    if isinstance(raw_locs, Mapping):
        locations = [raw_locs]
    elif isinstance(raw_locs, np.ndarray):
        locations = list(raw_locs.reshape(-1))
    elif isinstance(raw_locs, (list, tuple)):
        locations = list(raw_locs)
    else:
        locations = []
    labels = tuple(_mat_text(_get_mat_field(location, "labels")) for location in locations)
    if channels < 1 or samples < 1 or trials < 1 or not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("invalid_EEGLAB_set_header_support")
    if len(labels) != channels or any(not label for label in labels):
        raise ValueError("EEGLAB_channel_labels_missing_or_count_mismatch")
    return channels, samples, trials, sample_rate, labels, raw_data_name


def _run_id_from_path(set_path: Path) -> str:
    found = re.search(r"_run-([^_]+)_eeg\.set$", set_path.name, flags=re.IGNORECASE)
    return f"run-{found.group(1)}" if found else "run-unknown"


def _metadata_qc(
    channels_rows: Sequence[Mapping[str, str]],
    eeg_json: Mapping[str, Any],
    labels: Sequence[str],
) -> tuple[str, ...]:
    warnings: list[str] = []
    if len(channels_rows) != len(labels):
        warnings.append("channels_tsv_row_count_disagrees_with_set_header")
    tsv_labels = tuple(str(row.get("name", "")).strip() for row in channels_rows)
    if len(tsv_labels) == len(labels) and tuple(label.casefold() for label in tsv_labels) != tuple(label.casefold() for label in labels):
        warnings.append("channels_tsv_order_or_labels_disagree_with_set_header")
    if any(str(row.get("type", "")).strip().casefold() == "n/a" for row in channels_rows):
        warnings.append("channels_tsv_type_is_n_a")
    try:
        declared_eeg = int(eeg_json.get("EEGChannelCount"))
    except (TypeError, ValueError, OverflowError):
        declared_eeg = None
    if declared_eeg is not None and declared_eeg != sum(label.casefold() not in EEG_EXCLUDED_LABELS for label in labels):
        warnings.append("EEG_sidecar_count_disagrees_with_non_EOG_ECG_set_rows")
    try:
        declared_eog = int(eeg_json.get("EOGChannelCount"))
    except (TypeError, ValueError, OverflowError):
        declared_eog = None
    named_eog = sum(label.casefold() in {"heog", "veog"} for label in labels)
    if declared_eog is not None and declared_eog != named_eog:
        warnings.append("EOG_sidecar_count_disagrees_with_named_HEOG_VEOG")
    try:
        declared_ecg = int(eeg_json.get("ECGChannelCount"))
    except (TypeError, ValueError, OverflowError):
        declared_ecg = None
    named_ecg = sum(label.casefold() in {"ekg", "ecg"} for label in labels)
    if declared_ecg is not None and declared_ecg != named_ecg:
        warnings.append("ECG_sidecar_count_disagrees_with_named_EKG_ECG")
    if not any(label.casefold() == "heog" for label in labels):
        warnings.append("named_HEOG_channel_absent")
    if not any(label.casefold() == "veog" for label in labels):
        warnings.append("named_VEOG_channel_absent")
    if str(eeg_json.get("EEGReference", "")).strip() == "":
        warnings.append("acquisition_reference_metadata_missing")
    if any(str(row.get("units", "")).strip().casefold() in {"", "n/a"} for row in channels_rows):
        warnings.append("source_amplitude_units_n_a")
    return tuple(sorted(set(warnings)))


def _events_for_run(events_path: Path) -> list[dict[str, str]]:
    if not events_path.is_file():
        raise FileNotFoundError("events_tsv_missing")
    return _read_tsv(events_path)


def _candidate_from_set(set_path: Path, participant_id: str) -> RunMetadata:
    eeg_dir = set_path.parent
    fdt_path = set_path.with_suffix(".fdt")
    stem = set_path.name.removesuffix("_eeg.set")
    events_path = eeg_dir / f"{stem}_events.tsv"
    channels_path = eeg_dir / f"{stem}_channels.tsv"
    eeg_json_path = eeg_dir / f"{stem}_eeg.json"
    electrodes_path = eeg_dir / f"{stem}_electrodes.tsv"
    coordsystem_path = eeg_dir / f"{stem}_coordsystem.json"
    if not all(path.is_file() for path in (fdt_path, channels_path, eeg_json_path)):
        raise FileNotFoundError("required_BIDS_EEG_or_channel_metadata_missing")
    channel_count, samples, trials, sample_rate, labels, fdt_ref = _set_header(set_path)
    if fdt_ref.casefold() != fdt_path.name.casefold():
        raise ValueError("EEGLAB_fdt_basename_mismatch")
    expected_bytes = channel_count * samples * trials * 4
    if fdt_path.stat().st_size != expected_bytes:
        raise ValueError("external_fdt_byte_count_does_not_match_header")
    eeg_json = _read_json(eeg_json_path)
    channels_rows = _read_tsv(channels_path)
    qc = _metadata_qc(channels_rows, eeg_json, labels)
    eeg_labels = tuple(label for label in labels if label.casefold() not in EEG_EXCLUDED_LABELS)
    if abs(sample_rate - EXPECTED_FS_HZ) > 1e-9:
        raise ValueError(f"sample_rate_not_500_hz:{sample_rate}")
    events = _events_for_run(events_path)
    blocks = reconstruct_condition_blocks(
        events, sample_rate_hz=sample_rate, recording_sample_count=samples
    )
    windows = enumerate_full_windows(
        blocks, sample_rate_hz=sample_rate, window_seconds=WINDOW_SECONDS,
        window_prefix="fv20",
    )
    scg_windows = enumerate_full_windows(
        blocks, sample_rate_hz=sample_rate, window_seconds=SCG_WINDOW_SECONDS,
        window_prefix="scg60",
    )
    first_windows: list[tuple[str, CandidateWindow]] = []
    first_scg: list[tuple[str, CandidateWindow | None]] = []
    for condition in ("EO", "EC"):
        condition_windows = [window for window in windows if window.condition == condition]
        if condition_windows:
            first_windows.append((condition, condition_windows[0]))
        condition_scg = [window for window in scg_windows if window.condition == condition]
        first_scg.append((condition, condition_scg[0] if condition_scg else None))
    return RunMetadata(
        participant_id=participant_id,
        run_id=_run_id_from_path(set_path),
        set_path=str(set_path),
        fdt_path=str(fdt_path),
        events_path=str(events_path),
        channels_path=str(channels_path),
        eeg_json_path=str(eeg_json_path),
        electrodes_path=str(electrodes_path),
        coordsystem_path=str(coordsystem_path),
        electrodes_sha256=_sha256_file(electrodes_path) if electrodes_path.is_file() else None,
        coordsystem_sha256=_sha256_file(coordsystem_path) if coordsystem_path.is_file() else None,
        sample_rate_hz=sample_rate,
        sample_count=samples,
        channel_count=channel_count,
        trial_count=trials,
        channel_labels=labels,
        eeg_channel_labels=eeg_labels,
        metadata_qc=qc,
        eligible_conditions=tuple(condition for condition, _ in first_windows),
        first_windows=tuple(first_windows),
        scg_windows=tuple(first_scg),
        set_sha256=_sha256_file(set_path),
        events_sha256=_sha256_file(events_path),
        channels_sha256=_sha256_file(channels_path),
        eeg_json_sha256=_sha256_file(eeg_json_path),
    )


def _eligible_inventory(data_root: Path) -> tuple[list[RunMetadata], list[dict[str, str]]]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"ds003478_root_not_found:{data_root}")
    eligible: list[RunMetadata] = []
    rejected: list[dict[str, str]] = []
    for subject_dir in sorted(data_root.glob("sub-*")):
        participant_id = subject_dir.name
        eeg_dir = subject_dir / "eeg"
        if not eeg_dir.is_dir():
            rejected.append({"participant_id": participant_id, "reason": "eeg_directory_missing"})
            continue
        candidate_runs: list[RunMetadata] = []
        set_paths = sorted(eeg_dir.glob("*_task-Rest_run-*_eeg.set"))
        if not set_paths:
            set_paths = sorted(eeg_dir.glob("*_task-rest_run-*_eeg.set"))
        for set_path in set_paths:
            try:
                candidate_runs.append(_candidate_from_set(set_path, participant_id))
            except Exception as exc:
                rejected.append({
                    "participant_id": participant_id,
                    "run_id": _run_id_from_path(set_path),
                    "reason": f"{type(exc).__name__}:{str(exc)}",
                })
        matching = [run for run in candidate_runs if {"EO", "EC"}.issubset(set(run.eligible_conditions))]
        if matching:
            matching.sort(key=lambda run: (run.run_id, run.set_path))
            eligible.append(matching[0])
        else:
            rejected.append({"participant_id": participant_id, "reason": "no_single_run_with_full_20s_EO_and_EC"})
    return eligible, rejected


def _selection_rank(participant_id: str) -> str:
    text = f"{PROTOCOL_ID}|{SELECTION_DATE}|{participant_id}".encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def freeze_selection(data_root: Path, selection_path: Path, *, pilot_size: int = 12) -> dict[str, Any]:
    eligible, rejected = _eligible_inventory(data_root)
    if len(eligible) < pilot_size:
        raise RuntimeError(f"only_{len(eligible)}_eligible_participants_for_required_{pilot_size}")
    eligible.sort(key=lambda run: _selection_rank(run.participant_id))
    selected = eligible[:pilot_size]
    payload = {
        "protocol_id": PROTOCOL_ID,
        "selection_manifest_version": 2,
        "selection_date": SELECTION_DATE,
        "selection_rule": "SHA256(FULL-VECTOR-CHALLENGE-001|20261014|subject_id), ascending; technical metadata only",
        "data_root": str(data_root),
        "dataset_description_sha256": _sha256_file(data_root / "dataset_description.json"),
        "pilot_size": pilot_size,
        "eligible_count": len(eligible),
        "eligible_roster": [run.to_selection_record() for run in eligible],
        "rejected_inventory": rejected,
        "selected_participant_ids": [run.participant_id for run in selected],
        "selected_subject_rank_sha256": {run.participant_id: _selection_rank(run.participant_id) for run in selected},
        "outcome_blind": True,
        "clinical_metadata_read": False,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
    temporary = selection_path.with_suffix(selection_path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, selection_path)
    return payload


def load_or_freeze_selection(data_root: Path, selection_path: Path, *, pilot_size: int) -> dict[str, Any]:
    if not selection_path.is_file():
        frozen = freeze_selection(data_root, selection_path, pilot_size=pilot_size)
    else:
        with selection_path.open("r", encoding="utf-8") as stream:
            frozen = json.load(stream)
        if frozen.get("protocol_id") != PROTOCOL_ID or frozen.get("data_root") != str(data_root):
            raise RuntimeError("existing_selection_manifest_belongs_to_different_protocol_or_source")
        if frozen.get("pilot_size") != pilot_size:
            raise RuntimeError("existing_selection_manifest_has_different_pilot_size")
        if frozen.get("selection_manifest_version") != 2:
            raise RuntimeError("selection_manifest_must_be_refrozen_with_geometry_source_hashes")
        if not frozen.get("outcome_blind") or frozen.get("clinical_metadata_read"):
            raise RuntimeError("existing_selection_manifest_is_not_outcome_blind")
    current_dataset_description_hash = _sha256_file(data_root / "dataset_description.json")
    if frozen.get("dataset_description_sha256") != current_dataset_description_hash:
        raise RuntimeError("dataset_description_hash_changed_since_selection")
    return frozen


def _metadata_record_to_run(record: Mapping[str, Any]) -> RunMetadata:
    first = tuple(
        (condition, CandidateWindow(**{key: value for key, value in window.items() if key != "sample_count"}))
        for condition, window in record["first_windows"].items()
    )
    scg = tuple(
        (condition, CandidateWindow(**{key: value for key, value in window.items() if key != "sample_count"}) if window is not None else None)
        for condition, window in record["scg_windows"].items()
    )
    hashes = record["metadata_sha256"]
    return RunMetadata(
        participant_id=str(record["participant_id"]),
        run_id=str(record["run_id"]),
        set_path=str(record["set_path"]),
        fdt_path=str(record["fdt_path"]),
        events_path=str(record["events_path"]),
        channels_path=str(record["channels_path"]),
        eeg_json_path=str(record["eeg_json_path"]),
        electrodes_path=str(record["electrodes_path"]),
        coordsystem_path=str(record["coordsystem_path"]),
        electrodes_sha256=hashes["electrodes"],
        coordsystem_sha256=hashes["coordsystem"],
        sample_rate_hz=float(record["sample_rate_hz"]),
        sample_count=int(record["sample_count"]),
        channel_count=int(record["channel_count"]),
        trial_count=int(record["trial_count"]),
        channel_labels=tuple(str(label) for label in record["channel_labels"]),
        eeg_channel_labels=tuple(str(label) for label in record["eeg_channel_labels"]),
        metadata_qc=tuple(str(value) for value in record["metadata_qc"]),
        eligible_conditions=tuple(str(value) for value in record["eligible_conditions"]),
        first_windows=first,
        scg_windows=scg,
        set_sha256=str(hashes["set_header"]),
        events_sha256=str(hashes["events"]),
        channels_sha256=str(hashes["channels"]),
        eeg_json_sha256=str(hashes["eeg_json"]),
    )


def _verify_frozen_inputs(run: RunMetadata) -> None:
    """Reject source or sidecar changes after metadata-only sample selection."""
    frozen_inputs = (
        (Path(run.set_path), run.set_sha256, "set_header"),
        (Path(run.events_path), run.events_sha256, "events"),
        (Path(run.channels_path), run.channels_sha256, "channels"),
        (Path(run.eeg_json_path), run.eeg_json_sha256, "eeg_json"),
        (Path(run.electrodes_path), run.electrodes_sha256, "electrodes"),
        (Path(run.coordsystem_path), run.coordsystem_sha256, "coordsystem"),
    )
    for path, expected, name in frozen_inputs:
        if expected is None:
            if path.exists():
                raise ValueError(f"frozen_input_appeared_after_selection:{name}")
            continue
        if not path.is_file():
            raise FileNotFoundError(f"frozen_input_missing:{name}")
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen_input_hash_changed:{name}")


def _source_memmap(run: RunMetadata) -> tuple[np.memmap, str]:
    set_path = Path(run.set_path)
    fdt_path = Path(run.fdt_path)
    nbytes = run.channel_count * run.sample_count * run.trial_count * np.dtype("<f4").itemsize
    if fdt_path.parent.resolve() != set_path.parent.resolve() or fdt_path.name != set_path.with_suffix(".fdt").name:
        raise ValueError("source_fdt_must_be_bids_sibling_of_set_header")
    if fdt_path.stat().st_size != nbytes:
        raise ValueError("source_fdt_size_changed_since_selection")
    fdt_digest = _sha256_file(fdt_path)
    data = np.memmap(
        fdt_path, dtype="<f4", mode="r",
        shape=(run.channel_count, run.sample_count, run.trial_count), order="F",
    )
    return data, fdt_digest


def _window_data(
    data: np.memmap, run: RunMetadata, window: CandidateWindow
) -> tuple[np.ndarray, tuple[str, ...]]:
    raw = np.asarray(data[:, window.start_sample:window.end_sample_exclusive, 0], dtype=np.float64)
    keep = [index for index, label in enumerate(run.channel_labels) if label.casefold() not in EEG_EXCLUDED_LABELS]
    labels = tuple(run.channel_labels[index] for index in keep)
    eeg = raw[keep, :]
    if eeg.shape != (len(labels), window.sample_count):
        raise ValueError("selected_source_window_shape_mismatch")
    if not np.isfinite(eeg).all():
        raise ValueError("non_finite_source_window")
    return np.ascontiguousarray(eeg, dtype=np.float64), labels


def _unsupported_exception(family_id: str, exc: Exception) -> dict[str, Any]:
    primary_id = FULL_VECTOR_CHALLENGE_001_BY_FAMILY[family_id].primary_estimator_id
    return {
        "family_id": family_id,
        "estimator_id": primary_id,
        "computed": False,
        "valid": False,
        "qualified": False,
        "status": "unsupported",
        "reason": f"runner_exception:{type(exc).__name__}:{str(exc)}",
        "value": None,
    }


def _unsupported_reason(family_id: str, reason: str) -> dict[str, Any]:
    return {
        "family_id": family_id,
        "estimator_id": FULL_VECTOR_CHALLENGE_001_BY_FAMILY[family_id].primary_estimator_id,
        "computed": False,
        "valid": False,
        "qualified": False,
        "status": "unsupported",
        "reason": reason,
        "value": None,
    }


def _participant_failure_rows(
    run: RunMetadata, reason: str, acquisition_reference: str = "unknown"
) -> list[dict[str, Any]]:
    """Retain the frozen expected keys when source extraction fails."""
    rows: list[dict[str, Any]] = []
    for condition, window in run.first_windows:
        for reference_id in CORE_REFERENCE_IDS:
            realization_hash, source_reference_hash, reference_set_hash = _reference_provenance(
                run.eeg_channel_labels, reference_id, acquisition_reference
            )
            base = {
                "participant_id": run.participant_id, "run_id": run.run_id,
                "condition": condition, "window_id": window.window_id,
                "state_block_id": window.block_id,
                "support_start_sample": window.start_sample,
                "support_end_sample_exclusive": window.end_sample_exclusive,
                "sample_rate_hz": run.sample_rate_hz, "sample_count": window.sample_count,
                "start_seconds": window.start_sample / run.sample_rate_hz,
                "end_seconds": window.end_sample_exclusive / run.sample_rate_hz,
                "source_set_sha256": run.set_sha256, "source_fdt_sha256": None,
                "source_events_sha256": run.events_sha256,
                "source_channels_tsv_sha256": run.channels_sha256,
                "source_eeg_json_sha256": run.eeg_json_sha256,
                "source_electrodes_tsv_sha256": run.electrodes_sha256,
                "source_coordsystem_json_sha256": run.coordsystem_sha256,
                "window_signal_sha256": None, "reference_id": reference_id,
                "reference_realization_sha256": realization_hash,
                "reference_set_sha256": reference_set_hash,
                "source_reference_metadata_sha256": source_reference_hash,
                "source_acquisition_reference_metadata": acquisition_reference,
                "channel_labels": list(run.eeg_channel_labels),
                "channel_order_sha256": _hash_labels(run.eeg_channel_labels, ordered=True),
                "channel_set_sha256": _hash_labels(run.eeg_channel_labels, ordered=False),
                "channel_count": len(run.eeg_channel_labels),
                "metadata_qc": list(run.metadata_qc), "qualified": False,
                "support_audit": {
                    "window_selection_status": "frozen_from_metadata",
                    "window_extraction_status": "failed",
                    "dependency_start_sample": window.start_sample,
                    "dependency_end_sample_exclusive": window.end_sample_exclusive,
                    "physical_gap_status": "not_assessed_from_sample_index_source",
                    "bad_interval_status": "not_assessed",
                },
            }
            realization = "source_reference" if reference_id == CORE_REFERENCE_IDS[0] else "average_ref_sensitivity"
            for family_id in CORE_FAMILIES:
                if family_id == "TP_RAW_COMPARATOR":
                    estimator_id = "TP.raw_signal_pearson_persistence.v1"
                    formula_id = estimator_id
                    registry_family = "TP"
                else:
                    estimator_id = FULL_VECTOR_CHALLENGE_001_BY_FAMILY[family_id].primary_estimator_id
                    formula_id = None
                    registry_family = family_id
                row = {
                    **base, "family_id": family_id, "estimator_id": estimator_id,
                    "formula_kernel_estimator_id": formula_id,
                    "realization_id": realization, "computed": False,
                    "valid": False, "status": "unsupported", "value": None,
                    "reason": f"source_extraction_failed:{reason}",
                }
                rows.append(row)
    scg_windows = dict(run.scg_windows)
    for condition in ("EO", "EC"):
        window = scg_windows.get(condition)
        rows.append({
            "participant_id": run.participant_id, "run_id": run.run_id,
            "condition": condition, "window_id": window.window_id if window else None,
            "support_start_sample": window.start_sample if window else None,
            "support_end_sample_exclusive": window.end_sample_exclusive if window else None,
            "sample_rate_hz": run.sample_rate_hz,
            "sample_count": window.sample_count if window else None,
            "family_id": "SCG",
            "estimator_id": FULL_VECTOR_CHALLENGE_001_BY_FAMILY["SCG"].primary_estimator_id,
            "computed": False, "valid": False, "qualified": False,
            "status": "unsupported", "value": None,
            "reason": f"source_extraction_failed:{reason}",
            "support_audit": {"window_extraction_status": "failed"},
        })
    return rows


def _metric_record(family_id: str, kernel_result: Any, realization_id: str, kernel_estimator_id: str | None = None) -> dict[str, Any]:
    result_dict = _jsonable(kernel_result)
    if not isinstance(result_dict, dict):
        result_dict = {"raw_result": result_dict}
    result_dict["family_id"] = family_id
    family = FULL_VECTOR_CHALLENGE_001_BY_FAMILY.get(family_id)
    result_dict["estimator_id"] = (
        family.primary_estimator_id if family is not None
        else getattr(kernel_result, "estimator_id", "unresolved_estimator_id")
    )
    result_dict["formula_kernel_estimator_id"] = kernel_estimator_id or getattr(kernel_result, "estimator_id", None)
    result_dict["realization_id"] = realization_id
    result_dict["qualified"] = False
    return result_dict


def _window_hash(signals: np.ndarray, labels: Sequence[str], sample_rate_hz: float, reference_id: str) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps({
        "labels": list(labels), "sample_rate_hz": sample_rate_hz,
        "reference_id": reference_id, "shape": list(signals.shape),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(np.asarray(signals, dtype="<f8", order="C").tobytes(order="C"))
    return digest.hexdigest()


def _phase_from_alpha(alpha: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    analytic = hilbert(alpha, axis=-1)
    amplitude = np.abs(analytic)
    if not np.isfinite(amplitude).all():
        return None, "non_finite_analytic_amplitude"
    if np.any(amplitude == 0.0):
        return None, "zero_analytic_amplitude_phase_undefined"
    return np.angle(analytic), None


def _compute_nie(signals: np.ndarray, labels: Sequence[str]) -> Any:
    centered = signals - np.mean(signals, axis=1, keepdims=True, dtype=np.float64)
    norms = np.sqrt(np.sum(centered * centered, axis=1, dtype=np.float64))
    if np.any(norms <= 0.0):
        # Keep the matrix computable where Pearson correlations exist; the
        # low-level kernel will return an explicit unsupported reason.
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


def _base_record(
    run: RunMetadata,
    window: CandidateWindow,
    *,
    source_fdt_sha256: str,
    source_ref: str,
    acquisition_reference: str,
    reference_realization_hash: str,
    source_reference_metadata_hash: str,
    reference_set_hash: str | None,
    eeg_labels: Sequence[str],
    window_sha256: str,
) -> dict[str, Any]:
    return {
        "participant_id": run.participant_id,
        "run_id": run.run_id,
        "condition": window.condition,
        "window_id": window.window_id,
        "state_block_id": window.block_id,
        "support_start_sample": window.start_sample,
        "support_end_sample_exclusive": window.end_sample_exclusive,
        "sample_rate_hz": run.sample_rate_hz,
        "sample_count": window.sample_count,
        "start_seconds": window.start_sample / run.sample_rate_hz,
        "end_seconds": window.end_sample_exclusive / run.sample_rate_hz,
        "source_set_sha256": run.set_sha256,
        "source_fdt_sha256": source_fdt_sha256,
        "source_events_sha256": run.events_sha256,
        "source_channels_tsv_sha256": run.channels_sha256,
        "source_eeg_json_sha256": run.eeg_json_sha256,
        "source_electrodes_tsv_sha256": run.electrodes_sha256,
        "source_coordsystem_json_sha256": run.coordsystem_sha256,
        "window_signal_sha256": window_sha256,
        "reference_id": source_ref,
        "reference_realization_sha256": reference_realization_hash,
        "reference_set_sha256": reference_set_hash,
        "source_reference_metadata_sha256": source_reference_metadata_hash,
        "source_acquisition_reference_metadata": acquisition_reference,
        "channel_labels": list(eeg_labels),
        "channel_order_sha256": _hash_labels(eeg_labels, ordered=True),
        "channel_set_sha256": _hash_labels(eeg_labels, ordered=False),
        "channel_count": len(eeg_labels),
        "metadata_qc": list(run.metadata_qc),
        "source_amplitude_units": "n/a_raw_numeric_values",
        "support_audit": {
            "source_representation": "single_EEGLAB_FDT_sample_grid",
            "sample_grid_contiguous_in_payload": True,
            "physical_gap_status": "not_assessed_from_sample_index_source",
            "bad_interval_status": "not_assessed",
            "within_recording_bounds": (
                window.start_sample >= 0 and window.end_sample_exclusive <= run.sample_count
            ),
            "dependency_start_sample": window.start_sample,
            "dependency_end_sample_exclusive": window.end_sample_exclusive,
            "dependency_seconds": window.sample_count / run.sample_rate_hz,
            "dependency_policy": "full_frozen_window_no_channelwise_support_reduction",
        },
        "qualified": False,
    }


def _compute_window_family_records(
    run: RunMetadata,
    window: CandidateWindow,
    source_signals: np.ndarray,
    eeg_labels: tuple[str, ...],
    acquisition_reference: str,
    source_fdt_sha256: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    reference_variants = (
        ("source_reference", source_signals, "acquisition_reference_from_BIDS_sidecar"),
        ("average_ref_sensitivity", source_signals - np.mean(source_signals, axis=0, keepdims=True), AVERAGE_REFERENCE_ID),
    )
    for realization, signals, ref_id in reference_variants:
        ref_hash, source_reference_hash, reference_set_hash = _reference_provenance(
            eeg_labels, ref_id, acquisition_reference
        )
        window_sha = _window_hash(signals, eeg_labels, run.sample_rate_hz, ref_id)
        envelope: dict[str, Any] = _base_record(
            run, window, source_fdt_sha256=source_fdt_sha256,
            source_ref=ref_id, acquisition_reference=acquisition_reference,
            reference_realization_hash=ref_hash,
            source_reference_metadata_hash=source_reference_hash,
            reference_set_hash=reference_set_hash,
            eeg_labels=eeg_labels,
            window_sha256=window_sha,
        )
        try:
            sos = signal.butter(4, (8.0, 13.0), btype="bandpass", fs=run.sample_rate_hz, output="sos")
            alpha = signal.sosfiltfilt(sos, signals, axis=-1, padtype=None, padlen=0)
            analytic_phase, phase_reason = _phase_from_alpha(alpha)
            if phase_reason is not None:
                records.append({**envelope, **_unsupported_reason("PCV", phase_reason)})
            else:
                pcv = compute_pcv_phase_order_parameter_sd(
                    analytic_phase, phase_estimator_id=PCV_PHASE_ESTIMATOR_ID,
                    band_id=PCV_BAND_ID, channel_labels=eeg_labels,
                )
                records.append({**envelope, **_metric_record("PCV", pcv, realization)})
        except Exception as exc:
            records.append({**envelope, **_unsupported_exception("PCV", exc)})
        try:
            alpha = signal.sosfiltfilt(sos, signals, axis=-1, padtype=None, padlen=0)
            tp = compute_tp_log_rms_envelope_persistence(
                alpha, eeg_labels, sample_rate_hz=run.sample_rate_hz,
                envelope_samples=500, lag_samples=500, epsilon_amplitude=1e-12,
                minimum_pair_count=20,
            )
            records.append({**envelope, **_metric_record("TP", tp, realization)})
        except Exception as exc:
            records.append({**envelope, **_unsupported_exception("TP", exc)})
        try:
            raw_tp = compute_tp_raw_signal_persistence(
                signals, eeg_labels, sample_rate_hz=run.sample_rate_hz,
                lag_samples=500, minimum_pair_count=20,
            )
            records.append({**envelope, **_metric_record("TP_RAW_COMPARATOR", raw_tp, realization,
                                                         "TP.raw_signal_pearson_persistence.v1")})
        except Exception as exc:
            records.append({**envelope, **{
                **_unsupported_exception("TP", exc),
                "family_id": "TP_RAW_COMPARATOR",
                "estimator_id": "TP.raw_signal_pearson_persistence.v1",
            }})
        try:
            block_vectors = np.asarray([
                np.sqrt(np.mean(signals[:, index * 1000:(index + 1) * 1000] ** 2, axis=1, dtype=np.float64))
                for index in range(10)
            ], dtype=np.float64)
            start = window.start_sample / run.sample_rate_hz
            intervals = [(start + index * 2.0, start + (index + 1) * 2.0) for index in range(10)]
            ar = compute_ar_activity_transition_vector(
                block_vectors, intervals,
                vector_construction_id=AR_VECTOR_CONSTRUCTION_ID,
                scaling_id=AR_SCALING_ID, channel_labels=eeg_labels,
            )
            records.append({**envelope, **_metric_record("AR", ar, realization)})
        except Exception as exc:
            records.append({**envelope, **_unsupported_exception("AR", exc)})
        try:
            nie = _compute_nie(signals, eeg_labels)
            records.append({**envelope, **_metric_record("NIE", nie, realization)})
        except Exception as exc:
            records.append({**envelope, **_unsupported_exception("NIE", exc)})
        try:
            rd = compute_rd_centered_log_psd_dispersion(
                signals, eeg_labels, sample_rate_hz=run.sample_rate_hz,
                welch_segment_seconds=4.0, frequency_min_hz=1.0, frequency_max_hz=45.0,
            )
            records.append({**envelope, **_metric_record("RD", rd, realization)})
        except Exception as exc:
            records.append({**envelope, **_unsupported_exception("RD", exc)})
        try:
            oi = compute_oi_matrix(signals, eeg_labels)
            records.append({**envelope, **_metric_record("OI", oi, realization)})
        except Exception as exc:
            records.append({**envelope, **_unsupported_exception("OI", exc)})
        try:
            sc = compute_sc_lz78_exact_multiset_ratio(
                signals, eeg_labels,
                source_identity=f"ds003478|{run.participant_id}|{run.run_id}|{source_fdt_sha256}",
                segment_id=window.window_id,
                start_seconds=window.start_sample / run.sample_rate_hz,
                end_seconds=window.end_sample_exclusive / run.sample_rate_hz,
                reference_count=32, master_seed=310930,
            )
            records.append({**envelope, **_metric_record("SC", sc, realization)})
        except Exception as exc:
            records.append({**envelope, **_unsupported_exception("SC", exc)})
        try:
            ed = compute_ed_raw_sensor_covariance_participation_ratio(signals, eeg_labels)
            records.append({**envelope, **_metric_record("ED", ed, realization)})
        except Exception as exc:
            records.append({**envelope, **_unsupported_exception("ED", exc)})
    return records


def _positions_for_run(run: RunMetadata, labels: Sequence[str]) -> tuple[np.ndarray | None, dict[str, Any]]:
    if not Path(run.electrodes_path).is_file() or not Path(run.coordsystem_path).is_file():
        return None, {"geometry_status": "unsupported", "reason": "BIDS_electrodes_or_coordsystem_missing"}
    coordsystem = _read_json(Path(run.coordsystem_path))
    units = str(coordsystem.get("EEGCoordinateUnits", "")).strip().casefold()
    frame = str(coordsystem.get("EEGCoordinateSystem", "")).strip()
    description = str(coordsystem.get("EEGCoordinateSystemDescription", "")).strip()
    if units != "mm" or not frame:
        return None, {"geometry_status": "unsupported", "reason": "unsupported_or_missing_BIDS_geometry_frame_or_units",
                      "coordinate_units": units, "coordinate_frame": frame}
    rows = _read_tsv(Path(run.electrodes_path))
    by_label: dict[str, tuple[float, float, float]] = {}
    for row in rows:
        label = str(row.get("name", "")).strip()
        if not label:
            continue
        try:
            xyz = tuple(float(row[axis]) for axis in ("x", "y", "z"))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if not all(math.isfinite(value) for value in xyz):
            continue
        normalized = label.casefold()
        if normalized in by_label:
            return None, {"geometry_status": "unsupported", "reason": "duplicate_position_label",
                          "duplicate_label": label}
        by_label[normalized] = xyz
    missing = [label for label in labels if label.casefold() not in by_label]
    if missing:
        return None, {"geometry_status": "unsupported", "reason": "missing_or_nonfinite_BIDS_positions",
                      "missing_position_labels": missing, "coordinate_frame": frame,
                      "coordinate_description": description, "coordinate_units": units}
    positions = np.asarray([by_label[label.casefold()] for label in labels], dtype=np.float64) * 1e-3
    if positions.shape != (len(labels), 3) or not np.isfinite(positions).all() or np.any(np.linalg.norm(positions, axis=1) <= 0):
        return None, {"geometry_status": "unsupported", "reason": "invalid_position_vectors"}
    digest = hashlib.sha256(np.asarray(positions, dtype="<f8", order="C").tobytes(order="C")).hexdigest()
    return positions, {
        "geometry_status": "available", "coordinate_units_source": units,
        "coordinate_units_used": "m", "coordinate_frame": frame,
        "coordinate_description": description,
        "position_source": "bids", "positions_ordered_sha256": digest,
        "positions_channel_order": list(labels),
    }


def _compute_scg_record(
    run: RunMetadata,
    window: CandidateWindow | None,
    data: np.memmap,
    source_fdt_sha256: str,
    condition: str | None = None,
) -> dict[str, Any]:
    if window is None:
        return {
            "participant_id": run.participant_id, "run_id": run.run_id,
            "family_id": "SCG", "estimator_id": FULL_VECTOR_CHALLENGE_001_BY_FAMILY["SCG"].primary_estimator_id,
            "condition": condition, "support_seconds": 60.0, "value": None,
            "computed": False, "valid": False, "qualified": False,
            "status": "unsupported", "reason": "no_complete_60s_condition_window",
            "support_audit": {"window_selection_status": "no_complete_60s_condition_window"},
        }
    source_signals, labels = _window_data(data, run, window)
    metadata = _read_json(Path(run.eeg_json_path))
    acquisition_reference = str(metadata.get("EEGReference", "unknown"))
    avg_ref = source_signals - np.mean(source_signals, axis=0, keepdims=True, dtype=np.float64)
    positions, geometry = _positions_for_run(run, labels)
    reference_hash, source_reference_hash, reference_set_hash = _reference_provenance(
        labels, SCG_REFERENCE_ID, acquisition_reference
    )
    base = {
        "participant_id": run.participant_id, "run_id": run.run_id,
        "condition": window.condition, "window_id": window.window_id,
        "state_block_id": window.block_id,
        "support_start_sample": window.start_sample,
        "support_end_sample_exclusive": window.end_sample_exclusive,
        "start_seconds": window.start_sample / run.sample_rate_hz,
        "end_seconds": window.end_sample_exclusive / run.sample_rate_hz,
        "sample_rate_hz": run.sample_rate_hz, "sample_count": window.sample_count,
        "source_set_sha256": run.set_sha256, "source_fdt_sha256": source_fdt_sha256,
        "channel_labels": list(labels), "channel_count": len(labels),
        "channel_order_sha256": _hash_labels(labels, ordered=True),
        "channel_set_sha256": _hash_labels(labels, ordered=False),
        "reference_id": SCG_REFERENCE_ID,
        "reference_realization_sha256": reference_hash,
        "reference_set_sha256": reference_set_hash,
        "source_reference_metadata_sha256": source_reference_hash,
        "source_channels_tsv_sha256": run.channels_sha256,
        "source_eeg_json_sha256": run.eeg_json_sha256,
        "source_electrodes_tsv_sha256": run.electrodes_sha256,
        "source_coordsystem_json_sha256": run.coordsystem_sha256,
        "window_signal_sha256": _window_hash(avg_ref, labels, run.sample_rate_hz, SCG_REFERENCE_ID),
        "source_acquisition_reference_metadata": acquisition_reference,
        "support_audit": {
            "source_representation": "single_EEGLAB_FDT_sample_grid",
            "sample_grid_contiguous_in_payload": True,
            "physical_gap_status": "not_assessed_from_sample_index_source",
            "bad_interval_status": "not_assessed",
            "within_recording_bounds": (
                window.start_sample >= 0 and window.end_sample_exclusive <= run.sample_count
            ),
            "dependency_start_sample": window.start_sample,
            "dependency_end_sample_exclusive": window.end_sample_exclusive,
            "dependency_seconds": window.sample_count / run.sample_rate_hz,
            "dependency_policy": "full_frozen_60s_window",
        },
        "geometry": geometry, "metadata_qc": list(run.metadata_qc),
        "qualified": False,
    }
    if positions is None:
        return {**base, "family_id": "SCG", "estimator_id": FULL_VECTOR_CHALLENGE_001_BY_FAMILY["SCG"].primary_estimator_id,
                "computed": False, "valid": False, "status": "unsupported", "value": None,
                "reason": geometry.get("reason", "BIDS_geometry_unsupported")}
    support = MNPS31WindowSupport(
        window_start_sec=window.start_sample / run.sample_rate_hz,
        window_end_sec=window.end_sample_exclusive / run.sample_rate_hz,
        sample_rate_hz=run.sample_rate_hz, sample_count=window.sample_count,
        segment_id=f"{run.participant_id}|{run.run_id}|{window.window_id}",
        source_is_continuous=True, timestamps_uniform=True, gap_free=None,
        bad_interval_free=None, within_recording_bounds=True,
        effective_filter_support_valid=None,
    )
    try:
        measured = compute_d2_spatial_coupling_gradient(
            avg_ref, labels, positions, support,
            surface_id=SCG_SURFACE_ID,
            source_id=f"ds003478|{run.participant_id}|{run.run_id}",
            source_content_sha256=source_fdt_sha256,
            preprocessing_id=SCG_PREPROCESSING_ID,
            reference_mode="average_over_exact_surface",
            reference_id=SCG_REFERENCE_ID,
            position_source="bids", coordinate_frame=str(geometry["coordinate_frame"]),
            requested_surface_id=SCG_SURFACE_ID,
            requested_channel_names=labels,
            surface_qc={"warnings": list(run.metadata_qc), **geometry},
            allow_unknown_support_for_computation=False,
            signal_units="BIDS_units_n_a_raw_numeric",
            alpha_band_hz=(8.0, 13.0), segment_length_sec=2.0,
            segment_step_sec=2.0, min_segment_count=24,
        )
        metric = measured.scg
        result = metric.to_dict()
        return {**base, "family_id": "SCG", "estimator_id": FULL_VECTOR_CHALLENGE_001_BY_FAMILY["SCG"].primary_estimator_id,
                "formula_kernel_estimator_id": metric.estimator_id,
                "value": result.get("value"), "computed": result.get("computed"),
                "valid": result.get("valid"), "qualified": False,
                "status": result.get("status"), "reason": result.get("reason"),
                "measurement": result,
                "ldcb_sidecar": measured.ldcb_sidecar.to_dict()}
    except Exception as exc:
        return {**base, "family_id": "SCG", "estimator_id": FULL_VECTOR_CHALLENGE_001_BY_FAMILY["SCG"].primary_estimator_id,
                "computed": False, "valid": False, "status": "unsupported", "value": None,
                "reason": f"runner_exception:{type(exc).__name__}:{str(exc)}"}


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("family_id")), str(row.get("condition")), str(row.get("reference_id", "separate_scg"))), []).append(row)
    coverage: list[dict[str, Any]] = []
    for (family, condition, reference), items in sorted(grouped.items()):
        values: list[float] = []
        for item in items:
            value = item.get("value")
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
        coverage.append({
            "family_id": family, "condition": condition, "reference_id": reference,
            "row_count": len(items), "finite_value_count": len(values),
            "valid_value_count": sum(
                item.get("valid") is True
                and isinstance(item.get("value"), (int, float))
                and math.isfinite(float(item["value"]))
                for item in items
            ),
            "valid_record_count": sum(item.get("valid") is True for item in items),
            "participant_count_with_finite_value": len({str(item.get("participant_id")) for item in items if isinstance(item.get("value"), (int, float)) and math.isfinite(float(item["value"]))}),
            "participant_count_with_valid_value": len({str(item.get("participant_id")) for item in items if item.get("valid") is True and isinstance(item.get("value"), (int, float)) and math.isfinite(float(item["value"]))}),
            "finite_value_median": float(np.median(values)) if values else None,
            "statuses": dict(sorted({status: sum(item.get("status") == status for item in items) for status in {str(item.get("status")) for item in items}}.items())),
        })
    # Descriptive participant-paired contrasts only; windows are not treated as
    # independent participants, and no clinical values are accessed here.
    primary_lookup: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for row in rows:
        if row.get("reference_id") != "acquisition_reference_from_BIDS_sidecar":
            continue
        family = str(row.get("family_id"))
        if family == "AR":
            vector = row.get("component_values")
            if not isinstance(vector, list):
                continue
            values = [(f"component_{index:02d}", value) for index, value in enumerate(vector)]
        else:
            values = [("scalar", row.get("value"))]
        for component, value in values:
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                primary_lookup.setdefault((family, component), {}).setdefault(
                    str(row.get("participant_id")), {}
                )[str(row.get("condition"))] = float(value)
    paired: list[dict[str, Any]] = []
    for (family, component), participant_values in sorted(primary_lookup.items()):
        participant_deltas = [
            {"participant_id": participant_id, "delta_EC_minus_EO": values["EC"] - values["EO"]}
            for participant_id, values in sorted(participant_values.items())
            if "EC" in values and "EO" in values
        ]
        deltas = [item["delta_EC_minus_EO"] for item in participant_deltas]
        paired.append({
            "family_id": family, "component": component,
            "paired_participant_count": len(deltas),
            "median_EC_minus_EO": float(np.median(deltas)) if deltas else None,
            "participant_deltas": participant_deltas,
        })

    # Paired changes between the two reference realizations on identical source
    # samples, with participant identity retained for audit.
    reference_values: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in rows:
        family = str(row.get("family_id"))
        if family not in {"PCV", "TP", "NIE", "RD", "OI", "SC", "ED", "TP_RAW_COMPARATOR"}:
            continue
        value = row.get("value")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        key = (family, str(row.get("condition")), str(row.get("participant_id")))
        reference_values.setdefault(key, {})[str(row.get("reference_id"))] = float(value)
    reference_deltas: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for (family, condition, participant_id), values in sorted(reference_values.items()):
        source_value = values.get("acquisition_reference_from_BIDS_sidecar")
        average_value = values.get(AVERAGE_REFERENCE_ID)
        if source_value is not None and average_value is not None:
            reference_deltas.setdefault((family, condition), []).append({
                "participant_id": participant_id,
                "average_reference_minus_source_reference": average_value - source_value,
            })
    reference_summary = []
    for (family, condition), deltas in sorted(reference_deltas.items()):
        values = [row["average_reference_minus_source_reference"] for row in deltas]
        reference_summary.append({
            "family_id": family, "condition": condition,
            "paired_participant_count": len(values),
            "median_average_reference_minus_source_reference": float(np.median(values)),
            "participant_deltas": deltas,
        })

    # Participant-first descriptive redundancy: mean of the one frozen EO and
    # EC window values per participant and scalar family. AR and SCG stay
    # separate because they use vector and long-support outputs, respectively.
    family_participants: dict[str, dict[str, dict[str, float]]] = {}
    for (family, component), participants in primary_lookup.items():
        if component != "scalar" or family not in {"PCV", "TP", "NIE", "RD", "OI", "SC", "ED"}:
            continue
        family_participants[family] = participants
    family_means: dict[str, dict[str, float]] = {}
    for family, participants in family_participants.items():
        family_means[family] = {
            participant_id: (values["EO"] + values["EC"]) / 2.0
            for participant_id, values in participants.items()
            if "EO" in values and "EC" in values
        }
    family_correlations: list[dict[str, Any]] = []
    ordered_families = sorted(family_means)
    for left_index, left_family in enumerate(ordered_families):
        for right_family in ordered_families[left_index + 1:]:
            shared_ids = sorted(set(family_means[left_family]) & set(family_means[right_family]))
            left = np.asarray([family_means[left_family][subject] for subject in shared_ids], dtype=np.float64)
            right = np.asarray([family_means[right_family][subject] for subject in shared_ids], dtype=np.float64)
            if len(shared_ids) < 3 or np.std(left) == 0.0 or np.std(right) == 0.0:
                correlation, reason = None, "insufficient_participant_or_value_variation"
            else:
                correlation = float(np.corrcoef(left, right)[0, 1])
                reason = None if math.isfinite(correlation) else "non_finite_correlation"
                if reason is not None:
                    correlation = None
            family_correlations.append({
                "left_family": left_family, "right_family": right_family,
                "participant_count": len(shared_ids), "participant_ids": shared_ids,
                "pearson_r_of_EO_EC_participant_means": correlation,
                "reason": reason,
            })
    return {
        "outcome_blind": True,
        "clinical_metadata_read": False,
        "row_count": len(rows),
        "coverage": coverage,
        "paired_descriptive_EC_minus_EO": paired,
        "reference_sensitivity_deltas": reference_summary,
        "scalar_family_participant_correlations": family_correlations,
        "interpretation": "descriptive Gate B engineering output; no hypothesis test or scientific qualification",
    }


def run_numerical_controls() -> dict[str, Any]:
    """Run frozen implementation controls without accessing study outcomes."""
    checks: dict[str, dict[str, Any]] = {}
    try:
        from mndm.mnps.v3_2.sc import _lz78_count

        sequences = [
            np.asarray([0], dtype=np.uint8),
            np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8),
            np.zeros(64, dtype=np.uint8),
            np.ones(63, dtype=np.uint8),
        ]
        rng = np.random.Generator(np.random.PCG64(310930))
        sequences.extend(rng.integers(0, 2, size=length, dtype=np.uint8) for length in range(2, 130))
        mismatches = [
            index for index, symbols in enumerate(sequences)
            if int(_lz78_count(symbols)) != int(_lz78_phrase_count_binary(symbols))
        ]
        checks["sc_lz78_parser_parity"] = {
            "passed": not mismatches, "sequence_count": len(sequences),
            "mismatch_indices": mismatches,
        }
    except Exception as exc:
        checks["sc_lz78_parser_parity"] = {
            "passed": False, "reason": f"{type(exc).__name__}:{str(exc)}",
        }
    try:
        ties = compute_oi_matrix(np.ones((2, 64), dtype=np.float64), ("A", "B"))
        checks["oi_stable_ties_and_constant_rows"] = {
            "passed": ties.valid and ties.value == 0.0,
            "computed": ties.computed, "valid": ties.valid, "value": ties.value,
            "reason": ties.reason,
        }
    except Exception as exc:
        checks["oi_stable_ties_and_constant_rows"] = {"passed": False, "reason": f"{type(exc).__name__}:{str(exc)}"}
    try:
        phase, reason = _phase_from_alpha(np.zeros((2, 128), dtype=np.float64))
        checks["pcv_zero_analytic_amplitude_is_unsupported"] = {
            "passed": phase is None and reason == "zero_analytic_amplitude_phase_undefined",
            "phase_emitted": phase is not None, "reason": reason,
        }
    except Exception as exc:
        checks["pcv_zero_analytic_amplitude_is_unsupported"] = {
            "passed": False, "reason": f"{type(exc).__name__}:{str(exc)}",
        }
    try:
        blocks = np.tile(np.asarray([1.0, 2.0, 4.0]), (10, 1))
        intervals = [(index * 2.0, (index + 1) * 2.0) for index in range(10)]
        ar = compute_ar_activity_transition_vector(
            blocks, intervals, vector_construction_id=AR_VECTOR_CONSTRUCTION_ID,
            scaling_id=AR_SCALING_ID, channel_labels=("A", "B", "C"),
        )
        checks["ar_identical_adjacent_pattern"] = {
            "passed": ar.valid and len(ar.component_values) == 9 and all(abs(value or 0.0) <= 1e-14 for value in ar.component_values),
            "valid": ar.valid, "component_values": list(ar.component_values),
        }
    except Exception as exc:
        checks["ar_identical_adjacent_pattern"] = {"passed": False, "reason": f"{type(exc).__name__}:{str(exc)}"}
    try:
        rng = np.random.Generator(np.random.PCG64(771))
        sample = rng.normal(size=(5, 200))
        measured = compute_ed_raw_sensor_covariance_participation_ratio(sample, tuple(f"C{i}" for i in range(5)))
        covariance = np.cov(sample, rowvar=True, ddof=1)
        expected = float(np.trace(covariance) ** 2 / np.trace(covariance @ covariance))
        delta = abs(float(measured.value) - expected) if measured.value is not None else math.inf
        checks["ed_covariance_reconstruction"] = {
            "passed": measured.valid and delta <= 1e-11 * max(1.0, abs(expected)),
            "value": measured.value, "reference_value": expected, "absolute_delta": delta,
        }
    except Exception as exc:
        checks["ed_covariance_reconstruction"] = {"passed": False, "reason": f"{type(exc).__name__}:{str(exc)}"}
    try:
        labels = ("A", "B", "C", "D")
        weights = np.ones((4, 4), dtype=np.float64)
        np.fill_diagonal(weights, 0.0)
        graph = compute_nie_raw_weighted_global_efficiency(
            weights, labels, _hash_labels(labels, ordered=True),
            NIE_EDGE_ESTIMATOR_ID, NIE_EDGE_RULE_ID,
        )
        checks["nie_equal_weight_complete_graph"] = {
            "passed": graph.valid and abs(float(graph.value) - 1.0) <= 1e-14,
            "valid": graph.valid, "value": graph.value, "reason": graph.reason,
        }
    except Exception as exc:
        checks["nie_equal_weight_complete_graph"] = {"passed": False, "reason": f"{type(exc).__name__}:{str(exc)}"}
    try:
        unsupported = compute_tp_log_rms_envelope_persistence(
            np.zeros((2, 2000), dtype=np.float64), ("A", "B"),
            sample_rate_hz=EXPECTED_FS_HZ, envelope_samples=500, lag_samples=500,
        )
        checks["tp_constant_envelope_remains_null"] = {
            "passed": (not unsupported.valid) and unsupported.value is None,
            "status": unsupported.status, "value": unsupported.value,
            "reason": unsupported.reason,
        }
    except Exception as exc:
        checks["tp_constant_envelope_remains_null"] = {"passed": False, "reason": f"{type(exc).__name__}:{str(exc)}"}
    return {"passed": all(row.get("passed") is True for row in checks.values()), "checks": checks}


def _code_source_hashes() -> dict[str, str]:
    v32 = Path(__file__).parents[1]
    v31 = Path(__file__).parents[2] / "v3_1"
    sources = {
        "gate_b_runner": Path(__file__),
        "candidate_grid": v32 / "candidate_grid.py",
        "registry": v32 / "registry.py",
        "common_contracts": v32 / "contracts.py",
        "pcv": v32 / "pcv.py",
        "tp": v32 / "tp.py",
        "ar": v32 / "ar.py",
        "nie": v32 / "nie.py",
        "rd": v32 / "rd.py",
        "oi": v32 / "oi.py",
        "sc": v32 / "sc.py",
        "ed": v32 / "ed.py",
        "scg": v31 / "d2_spatial_gradient.py",
        "scg_support": v31 / "candidates.py",
        "lz78_reference_parser": v31 / "eeg_metrics.py",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"implementation_source_files_missing:{missing}")
    return {name: _sha256_file(path) for name, path in sources.items()}


def _optional_package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_gate_b(
    selection: Mapping[str, Any],
    *,
    mode: str,
    output_root: Path,
    selection_path: Path,
    numerical_controls: Mapping[str, Any],
    limit: int | None = None,
) -> Path:
    records = {str(item["participant_id"]): _metadata_record_to_run(item) for item in selection["eligible_roster"]}
    selected_ids = list(selection["selected_participant_ids"])
    if mode == "smoke":
        participants = selected_ids[:1]
    elif mode == "pilot":
        if limit is not None:
            raise ValueError("pilot_mode_does_not_allow_participant_limit")
        if int(selection.get("pilot_size", -1)) != 12 or len(selected_ids) != 12:
            raise ValueError("pilot_mode_requires_all_12_frozen_participants")
        participants = selected_ids
    else:
        raise ValueError("mode_must_be_smoke_or_pilot")
    if limit is not None:
        participants = participants[:max(1, int(limit))]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = output_root / f"{mode}_{stamp}"
    run_root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    participant_status: list[dict[str, Any]] = []
    started = time.perf_counter()
    source_digests: dict[str, str] = {}
    for participant_id in participants:
        run = records[participant_id]
        print(f"[{mode}] {participant_id} {run.run_id}: hashing and mapping selected source", flush=True)
        acquisition_reference = "unknown"
        try:
            _verify_frozen_inputs(run)
            eeg_metadata = _read_json(Path(run.eeg_json_path))
            acquisition_reference = str(eeg_metadata.get("EEGReference", "unknown"))
            data, source_fdt_sha256 = _source_memmap(run)
            source_digests[participant_id] = source_fdt_sha256
            run_rows: list[dict[str, Any]] = []
            windows = dict(run.first_windows)
            for condition in ("EO", "EC"):
                window = windows[condition]
                signal_matrix, labels = _window_data(data, run, window)
                if labels != run.eeg_channel_labels:
                    raise ValueError("frozen_EEG_label_order_changed_since_selection")
                print(f"[{mode}] {participant_id} {condition}: compute 8+TP comparator", flush=True)
                run_rows.extend(_compute_window_family_records(
                    run, window, signal_matrix, labels,
                    acquisition_reference, source_fdt_sha256,
                ))
            scg_windows = dict(run.scg_windows)
            for condition in ("EO", "EC"):
                scg_window = scg_windows[condition]
                try:
                    run_rows.append(_compute_scg_record(run, scg_window, data, source_fdt_sha256, condition))
                except Exception as exc:
                    run_rows.append({
                        "participant_id": participant_id, "run_id": run.run_id,
                        "condition": condition, "family_id": "SCG",
                        "estimator_id": FULL_VECTOR_CHALLENGE_001_BY_FAMILY["SCG"].primary_estimator_id,
                        "computed": False, "valid": False, "qualified": False,
                        "status": "unsupported", "reason": f"runner_exception:{type(exc).__name__}:{str(exc)}",
                        "value": None,
                    })
            rows.extend(run_rows)
            participant_status.append({"participant_id": participant_id, "run_id": run.run_id,
                                       "status": "processed", "row_count": len(run_rows),
                                       "source_fdt_sha256": source_fdt_sha256})
        except Exception as exc:
            rows.extend(_participant_failure_rows(run, f"{type(exc).__name__}:{str(exc)}", acquisition_reference))
            participant_status.append({"participant_id": participant_id, "run_id": run.run_id,
                                       "status": "failed", "reason": f"{type(exc).__name__}:{str(exc)}"})
            print(f"[{mode}] {participant_id} FAILED: {type(exc).__name__}: {exc}", flush=True)
        finally:
            if "data" in locals():
                del data
    rows_path = run_root / "measurements.jsonl.gz"
    with gzip.open(rows_path, "wt", encoding="utf-8", newline="\n", compresslevel=6) as stream:
        for row in rows:
            stream.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    summary = _summarize(rows)
    summary["numerical_controls"] = _jsonable(numerical_controls)
    summary["run_mode"] = mode
    summary["participant_status"] = participant_status
    summary["selected_participant_ids_processed"] = participants
    summary["selection_manifest_path"] = str(selection_path)
    summary["selection_manifest_sha256"] = _sha256_file(selection_path)
    summary["started_at_utc"] = datetime.fromtimestamp(time.time() - (time.perf_counter() - started), timezone.utc).isoformat()
    summary["elapsed_seconds"] = time.perf_counter() - started
    summary["measurement_file"] = rows_path.name
    summary["measurement_file_sha256"] = _sha256_file(rows_path)
    summary["source_fdt_sha256_by_participant"] = source_digests
    # Gate B is an engineering gate. Exact keys prevent a duplicate from
    # concealing a missing participant/condition/reference/family row.
    primary_coverage = {
        (item["family_id"], item["condition"]): item["valid_value_count"]
        for item in summary["coverage"]
        if item["reference_id"] == "acquisition_reference_from_BIDS_sidecar"
    }
    required_families = ("PCV", "TP", "NIE", "RD", "OI", "SC", "ED")
    missing_primary_support = [
        {"family_id": family, "condition": condition}
        for family in required_families for condition in ("EO", "EC")
        if primary_coverage.get((family, condition), 0) < 1
    ]
    ar_supported = [
        row for row in rows
        if row.get("family_id") == "AR"
        and row.get("reference_id") == "acquisition_reference_from_BIDS_sidecar"
        and isinstance(row.get("component_values"), list)
        and any(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in row["component_values"])
    ]
    processed_count = sum(item.get("status") == "processed" for item in participant_status)
    expected_subject_count = len(selected_ids)
    expected_core_rows = len(participants) * 2 * 2 * len(CORE_FAMILIES)
    core_rows = sum(row.get("family_id") in CORE_FAMILIES for row in rows)
    expected_keys = {
        (participant_id, condition, reference_id, family_id)
        for participant_id in participants
        for condition in ("EO", "EC")
        for reference_id in CORE_REFERENCE_IDS
        for family_id in CORE_FAMILIES
    }
    actual_key_counts: dict[tuple[str, str, str, str], int] = {}
    for row in rows:
        if row.get("family_id") not in CORE_FAMILIES:
            continue
        key = (
            str(row.get("participant_id")), str(row.get("condition")),
            str(row.get("reference_id")), str(row.get("family_id")),
        )
        actual_key_counts[key] = actual_key_counts.get(key, 0) + 1
    actual_keys = set(actual_key_counts)
    missing_keys = sorted(expected_keys - actual_keys)
    duplicate_keys = sorted(key for key, count in actual_key_counts.items() if count > 1)
    unexpected_keys = sorted(actual_keys - expected_keys)
    exact_key_audit_passed = not missing_keys and not duplicate_keys and not unexpected_keys
    expected_scg_keys = {
        (participant_id, condition)
        for participant_id in participants for condition in ("EO", "EC")
    }
    actual_scg_key_counts: dict[tuple[str, str], int] = {}
    for row in rows:
        if row.get("family_id") != "SCG":
            continue
        key = (str(row.get("participant_id")), str(row.get("condition")))
        actual_scg_key_counts[key] = actual_scg_key_counts.get(key, 0) + 1
    actual_scg_keys = set(actual_scg_key_counts)
    missing_scg_keys = sorted(expected_scg_keys - actual_scg_keys)
    duplicate_scg_keys = sorted(key for key, count in actual_scg_key_counts.items() if count > 1)
    unexpected_scg_keys = sorted(actual_scg_keys - expected_scg_keys)
    scg_row_audit_passed = not missing_scg_keys and not duplicate_scg_keys and not unexpected_scg_keys
    runner_exceptions = [
        {"participant_id": row.get("participant_id"), "family_id": row.get("family_id"), "condition": row.get("condition"), "reason": row.get("reason")}
        for row in rows if str(row.get("reason", "")).startswith("runner_exception:")
    ]
    numerical_controls_passed = bool(summary.get("numerical_controls", {}).get("passed"))
    smoke_passed = (
        processed_count == 1 and core_rows == expected_core_rows
        and exact_key_audit_passed and scg_row_audit_passed and not runner_exceptions
        and not missing_primary_support and len(ar_supported) > 0
        and numerical_controls_passed
    )
    full_pilot_requested = (
        mode == "pilot" and limit is None and len(participants) == 12
        and participants == selected_ids and len(selected_ids) == expected_subject_count == 12
    )
    pilot_passed = (
        full_pilot_requested and processed_count == expected_subject_count
        and core_rows == expected_core_rows and exact_key_audit_passed
        and scg_row_audit_passed
        and not runner_exceptions and not missing_primary_support
        and len(ar_supported) > 0 and numerical_controls_passed
    )
    if mode == "smoke":
        summary["smoke_test"] = {
            "status": "PASS" if smoke_passed else "FAIL",
            "participant_ids": participants,
            "core_rows": core_rows,
            "expected_core_rows": expected_core_rows,
            "exact_key_audit_passed": exact_key_audit_passed,
            "scg_row_audit_passed": scg_row_audit_passed,
            "runner_exception_count": len(runner_exceptions),
            "numerical_controls_passed": numerical_controls_passed,
            "interpretation": "single-participant smoke only; does not evaluate Gate B",
        }
        summary["paired_descriptive_EC_minus_EO"] = []
        summary["reference_sensitivity_deltas"] = []
        summary["scalar_family_participant_correlations"] = []
        summary["pilot_descriptive_summaries_status"] = "NOT_EVALUATED_SMOKE_SCOPE"
    summary["gate_b_engineering_decision"] = {
        "status": ("PASS" if pilot_passed else "FAIL") if mode == "pilot" else "NOT_EVALUATED",
        "criteria": [
            "pilot mode processes all 12 frozen selected participants; smoke is never a Gate B decision",
            "all 8 families plus raw TP comparator emitted for both conditions and both reference realizations",
            "every expected participant-condition-reference-family key appears exactly once",
            "one explicit optional SCG row is emitted per participant-condition; SCG values do not gate the core collection",
            "no runner exceptions; candidate-local unsupported results remain explicit",
            "each required scalar family has at least one finite primary-reference value in EO and EC",
            "AR emits at least one finite adjacent-block component on primary reference",
            "frozen numerical controls pass",
            "SCG remains optional and does not gate the core collection",
            "no condition contrast magnitude, sign, or p-value used for the decision",
        ],
        "processed_participant_count": processed_count,
        "expected_participant_count": expected_subject_count,
        "core_measurement_row_count": core_rows,
        "expected_core_measurement_row_count": expected_core_rows,
        "exact_expected_key_count": len(expected_keys),
        "exact_key_audit_passed": exact_key_audit_passed,
        "missing_expected_keys": [list(key) for key in missing_keys[:100]],
        "duplicate_keys": [list(key) for key in duplicate_keys[:100]],
        "unexpected_keys": [list(key) for key in unexpected_keys[:100]],
        "expected_scg_row_count": len(expected_scg_keys),
        "scg_row_count": sum(row.get("family_id") == "SCG" for row in rows),
        "scg_row_audit_passed": scg_row_audit_passed,
        "missing_expected_scg_keys": [list(key) for key in missing_scg_keys[:100]],
        "duplicate_scg_keys": [list(key) for key in duplicate_scg_keys[:100]],
        "unexpected_scg_keys": [list(key) for key in unexpected_scg_keys[:100]],
        "missing_primary_family_condition_support": missing_primary_support,
        "ar_rows_with_at_least_one_finite_component": len(ar_supported),
        "runner_exception_count": len(runner_exceptions),
        "numerical_controls_passed": numerical_controls_passed,
    }
    if mode == "pilot":
        summary["pilot_descriptive_summaries_status"] = "REPORTED_OUTCOME_BLIND_DESCRIPTIVE"
    with (run_root / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(summary), stream, indent=2, ensure_ascii=False, allow_nan=False)
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "mode": mode,
        "qualified": False,
        "outcome_blind": True,
        "clinical_metadata_read": False,
        "selection_manifest_sha256": _sha256_file(selection_path),
        "measurement_file_sha256": _sha256_file(rows_path),
        "summary_sha256": _sha256_file(run_root / "summary.json"),
        "runner_sha256": _sha256_file(Path(__file__)),
        "implementation_code_sha256": _code_source_hashes(),
        "protocol_sha256": _sha256_file(Path.cwd() / "project" / "new_mnps_v.3.2" / "protocols" / "FULL-VECTOR-CHALLENGE-001.md"),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "scipy_version": __import__("scipy").__version__,
        "numba_version": _optional_package_version("numba"),
        "rows_path": str(rows_path),
    }
    with (run_root / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return run_root


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pilot-size", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None,
                        help="development override; selection stays frozen and outcome-blind")
    args = parser.parse_args(argv)
    selection = load_or_freeze_selection(args.data_root, args.selection, pilot_size=args.pilot_size)
    print(f"Selection frozen: {args.selection} | selected={selection['selected_participant_ids']}", flush=True)
    numerical_controls = run_numerical_controls()
    print(f"Numerical controls: {'PASS' if numerical_controls['passed'] else 'FAIL'}", flush=True)
    if not numerical_controls["passed"]:
        raise RuntimeError("frozen_numerical_controls_failed; no source signal windows were processed")
    result = run_gate_b(selection, mode=args.mode, output_root=args.output_root,
                        selection_path=args.selection, numerical_controls=numerical_controls,
                        limit=args.limit)
    print(f"Gate B {args.mode} output: {result}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
