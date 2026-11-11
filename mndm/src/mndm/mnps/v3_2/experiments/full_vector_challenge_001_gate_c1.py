"""Outcome-blind full-cohort extraction for FULL-VECTOR-CHALLENGE-001 Gate C1."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import signal

from mndm.mnps.v3_2 import candidate_grid
from mndm.mnps.v3_2.candidate_grid import CandidateWindow
from mndm.mnps.v3_2.experiments import full_vector_challenge_001 as gate_b


PROTOCOL_ID = "FULL-VECTOR-CHALLENGE-001"
GATE_ID = "C1"
WINDOW_SECONDS = 20.0
BASELINE_FAMILIES = (
    "BASELINE_ALPHA_LOG10_POWER",
    "BASELINE_RELATIVE_DELTA",
    "BASELINE_RELATIVE_THETA",
    "BASELINE_RELATIVE_ALPHA",
    "BASELINE_RELATIVE_BETA",
    "BASELINE_RELATIVE_GAMMA",
    "BASELINE_SPECTRAL_ENTROPY_NORMALIZED",
    "BASELINE_ACF_LAG1",
    "BASELINE_MEAN_ABS_CHANNEL_CORRELATION",
)
BASELINE_ESTIMATORS = {
    "BASELINE_ALPHA_LOG10_POWER": "BASELINE.alpha_log10_channel_median_welch.v1",
    "BASELINE_RELATIVE_DELTA": "BASELINE.relative_delta_1_4hz_channel_median_welch.v1",
    "BASELINE_RELATIVE_THETA": "BASELINE.relative_theta_4_8hz_channel_median_welch.v1",
    "BASELINE_RELATIVE_ALPHA": "BASELINE.relative_alpha_8_13hz_channel_median_welch.v1",
    "BASELINE_RELATIVE_BETA": "BASELINE.relative_beta_13_30hz_channel_median_welch.v1",
    "BASELINE_RELATIVE_GAMMA": "BASELINE.relative_gamma_30_45hz_channel_median_welch.v1",
    "BASELINE_SPECTRAL_ENTROPY_NORMALIZED": "BASELINE.normalized_spectral_entropy_1_45hz_channel_median.v1",
    "BASELINE_ACF_LAG1": "BASELINE.channel_median_pearson_acf_lag1.v1",
    "BASELINE_MEAN_ABS_CHANNEL_CORRELATION": "BASELINE.mean_absolute_zero_lag_channel_correlation.v1",
}
BASELINE_BANDS = {
    "BASELINE_RELATIVE_DELTA": (1.0, 4.0),
    "BASELINE_RELATIVE_THETA": (4.0, 8.0),
    "BASELINE_RELATIVE_ALPHA": (8.0, 13.0),
    "BASELINE_RELATIVE_BETA": (13.0, 30.0),
    "BASELINE_RELATIVE_GAMMA": (30.0, 45.0),
}
CORE_FAMILIES = gate_b.CORE_FAMILIES
CORE_REFERENCE_IDS = gate_b.CORE_REFERENCE_IDS
PRIMARY_REFERENCE_ID = CORE_REFERENCE_IDS[0]
AR_COMPONENT_NAMES = tuple(f"block_{index:03d}_to_{index + 1:03d}" for index in range(9))

REPO_ROOT = Path(__file__).parents[6]
PROTOCOL_PATH = REPO_ROOT / "project" / "new_mnps_v.3.2" / "protocols" / (
    "FULL-VECTOR-CHALLENGE-001-GATE-C1.md"
)
BASE_PROTOCOL_PATH = REPO_ROOT / "project" / "new_mnps_v.3.2" / "protocols" / (
    "FULL-VECTOR-CHALLENGE-001.md"
)
DEFAULT_GATE_B_SELECTION_PATH = Path(__file__).parents[5] / "config" / (
    "mnps_v3_2_full_vector_challenge_001_selection_20261014_r2.json"
)
DEFAULT_SELECTION_PATH = Path(__file__).parents[5] / "config" / (
    "mnps_v3_2_full_vector_challenge_001_gate_c1_selection_20261014_r1.json"
)
DEFAULT_DATA_ROOT = gate_b.DEFAULT_DATA_ROOT
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / PROTOCOL_ID / "GATE-C1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if hasattr(gate_b, "_jsonable"):
        return gate_b._jsonable(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _canonical(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path}")
    return value


def _all_windows_for_run(run: gate_b.RunMetadata) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    events = gate_b._events_for_run(Path(run.events_path))
    blocks = candidate_grid.reconstruct_condition_blocks(
        events,
        sample_rate_hz=run.sample_rate_hz,
        recording_sample_count=run.sample_count,
    )
    windows = candidate_grid.enumerate_full_windows(
        blocks, sample_rate_hz=run.sample_rate_hz,
        window_seconds=WINDOW_SECONDS, window_prefix="fv20",
    )
    output = {"EO": [], "EC": []}
    for window in windows:
        output[window.condition].append(window.to_dict())
    scg_blocks = candidate_grid.enumerate_full_windows(
        blocks, sample_rate_hz=run.sample_rate_hz,
        window_seconds=gate_b.SCG_WINDOW_SECONDS, window_prefix="scg60",
    )
    return output, [window.to_dict() for window in scg_blocks]


def freeze_or_load_selection(
    data_root: Path,
    gate_b_selection_path: Path,
    selection_path: Path,
) -> dict[str, Any]:
    if not PROTOCOL_PATH.is_file() or not BASE_PROTOCOL_PATH.is_file():
        raise FileNotFoundError("frozen_gate_c1_or_base_protocol_missing")
    if not gate_b_selection_path.is_file():
        raise FileNotFoundError(f"gate_b_r2_selection_missing:{gate_b_selection_path}")
    gate_b_selection_sha = _sha256_file(gate_b_selection_path)
    protocol_sha = _sha256_file(PROTOCOL_PATH)
    base_protocol_sha = _sha256_file(BASE_PROTOCOL_PATH)
    dataset_description_sha = _sha256_file(data_root / "dataset_description.json")

    existing_selection = _load_json(selection_path) if selection_path.is_file() else None

    gate_b_selection = _load_json(gate_b_selection_path)
    if (gate_b_selection.get("protocol_id") != PROTOCOL_ID
            or gate_b_selection.get("data_root") != str(data_root)
            or gate_b_selection.get("selection_manifest_version") != 2
            or gate_b_selection.get("outcome_blind") is not True
            or gate_b_selection.get("clinical_metadata_read") is not False):
        raise RuntimeError("gate_b_r2_selection_is_not_compatible_or_outcome_blind")
    if gate_b_selection.get("dataset_description_sha256") != dataset_description_sha:
        raise RuntimeError("dataset_description_hash_changed_since_gate_b_selection")

    eligible, rejected = gate_b._eligible_inventory(data_root)
    eligible.sort(key=lambda run: gate_b._selection_rank(run.participant_id))
    if _canonical(rejected) != _canonical(gate_b_selection.get("rejected_inventory", [])):
        raise RuntimeError("technical_exclusion_inventory_changed_since_gate_b")
    expected_by_id = {
        str(item["participant_id"]): item
        for item in gate_b_selection["eligible_roster"]
    }
    if len(eligible) != len(expected_by_id):
        raise RuntimeError("technical_eligible_roster_count_changed_since_gate_b")
    frozen_records: list[dict[str, Any]] = []
    scg_geometry_audit: list[dict[str, Any]] = []
    for run in eligible:
        old_record = expected_by_id.get(run.participant_id)
        if old_record is None or _canonical(run.to_selection_record()) != _canonical(old_record):
            raise RuntimeError(f"technical_eligible_roster_changed_since_gate_b:{run.participant_id}")
        windows_by_condition, long_windows = _all_windows_for_run(run)
        if not windows_by_condition["EO"] or not windows_by_condition["EC"]:
            raise RuntimeError(f"frozen_eligible_run_lost_condition_windows:{run.participant_id}")
        for condition in ("EO", "EC"):
            if not any(item["window_id"] == old_record["first_windows"][condition]["window_id"]
                       for item in windows_by_condition[condition]):
                raise RuntimeError(f"gate_b_first_window_not_in_full_window_grid:{run.participant_id}:{condition}")
        record = dict(old_record)
        record["all_windows"] = windows_by_condition
        record["scg_60s_windows"] = long_windows
        frozen_records.append(record)
        _, geometry = gate_b._positions_for_run(run, run.eeg_channel_labels)
        scg_geometry_audit.append({
            "participant_id": run.participant_id,
            "run_id": run.run_id,
            "electrodes_sha256": run.electrodes_sha256,
            "coordsystem_sha256": run.coordsystem_sha256,
            **geometry,
        })

    selected_ids = [str(item["participant_id"]) for item in frozen_records]
    if selected_ids != [str(item["participant_id"]) for item in gate_b_selection["eligible_roster"]]:
        raise RuntimeError("eligible_roster_order_changed_since_gate_b")
    frozen_content = {
        "protocol_id": PROTOCOL_ID,
        "gate_id": GATE_ID,
        "selection_manifest_version": 1,
        "data_root": str(data_root),
        "dataset_description_sha256": dataset_description_sha,
        "gate_b_selection_path": str(gate_b_selection_path),
        "gate_b_selection_sha256": gate_b_selection_sha,
        "gate_c1_protocol_path": str(PROTOCOL_PATH),
        "gate_c1_protocol_sha256": protocol_sha,
        "base_protocol_sha256": base_protocol_sha,
        "eligibility_rule": (
            "all participants in the frozen Gate B R2 technical roster; earliest Rest run "
            "with a complete 20-second EO and EC state block; no clinical metadata"
        ),
        "window_rule": (
            "all complete nonoverlapping 20-second windows within frozen EO/EC state blocks; "
            "500 Hz; 10000 half-open samples per window"
        ),
        "eligible_count": len(frozen_records),
        "eligible_roster": frozen_records,
        "rejected_inventory": rejected,
        "scg_geometry_audit": scg_geometry_audit,
        "outcome_blind": True,
        "clinical_metadata_read": False,
        "clinical_paths_read": [],
    }
    if existing_selection is not None:
        for key, value in frozen_content.items():
            if _canonical(existing_selection.get(key)) != _canonical(value):
                raise RuntimeError(f"frozen_gate_c1_selection_content_changed:{key}")
        expected_hashes = {
            "gate_b_selection_sha256": gate_b_selection_sha,
            "gate_c1_protocol_sha256": protocol_sha,
            "base_protocol_sha256": base_protocol_sha,
            "dataset_description_sha256": dataset_description_sha,
        }
        for key, value in expected_hashes.items():
            if existing_selection.get(key) != value:
                raise RuntimeError(f"frozen_gate_c1_selection_mismatch:{key}")
        if existing_selection.get("eligible_count") != len(frozen_records):
            raise RuntimeError("frozen_gate_c1_eligible_count_changed")
        return existing_selection
    payload = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), **frozen_content}
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_jsonable(payload), indent=2, ensure_ascii=False,
                         allow_nan=False).encode("utf-8")
    temporary = selection_path.with_suffix(selection_path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, selection_path)
    return payload


def _metadata_record_to_run(record: Mapping[str, Any]) -> gate_b.RunMetadata:
    return gate_b._metadata_record_to_run(record)


def _window_from_record(record: Mapping[str, Any]) -> CandidateWindow:
    return CandidateWindow(**{
        key: value for key, value in record.items() if key != "sample_count"
    })


def _band_integral(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> np.ndarray:
    keep = (freqs >= low) & (freqs <= high)
    selected_freqs = freqs[keep]
    selected_psd = psd[..., keep]
    if selected_freqs.size < 2 or not np.isfinite(selected_psd).all():
        raise ValueError("baseline_band_support_or_psd_invalid")
    return np.trapz(selected_psd, selected_freqs, axis=-1)


def _baseline_values(signals: np.ndarray, fs: float) -> tuple[dict[str, float], dict[str, str]]:
    """Return supported baseline values and candidate-local failure reasons."""
    values: dict[str, float] = {}
    reasons: dict[str, str] = {}
    psd: np.ndarray | None = None
    freqs: np.ndarray | None = None
    try:
        freqs, psd = signal.welch(
            signals, fs=fs, window="hann", nperseg=2000, noverlap=1000,
            detrend="constant", scaling="density", axis=-1,
        )
        if not np.isfinite(psd).all() or np.any(psd < 0.0):
            raise ValueError("welch_psd_nonfinite_or_negative")
        total_1_45 = _band_integral(freqs, psd, 1.0, 45.0)
        if np.any(~np.isfinite(total_1_45)) or np.any(total_1_45 <= 0.0):
            raise ValueError("nonpositive_total_1_45hz_power")
        alpha_power = _band_integral(freqs, psd, 8.0, 13.0)
        if np.any(~np.isfinite(alpha_power)) or np.any(alpha_power <= 0.0):
            reasons["BASELINE_ALPHA_LOG10_POWER"] = "nonpositive_alpha_power"
        else:
            value = float(np.median(np.log10(alpha_power)))
            if math.isfinite(value):
                values["BASELINE_ALPHA_LOG10_POWER"] = value
            else:
                reasons["BASELINE_ALPHA_LOG10_POWER"] = "nonfinite_log10_alpha_power"
        for family_id, (low, high) in BASELINE_BANDS.items():
            band_power = _band_integral(freqs, psd, low, high)
            ratios = band_power / total_1_45
            if not np.isfinite(ratios).all() or np.any(ratios < 0.0):
                reasons[family_id] = "invalid_relative_band_power"
            else:
                value = float(np.median(ratios))
                if math.isfinite(value):
                    values[family_id] = value
                else:
                    reasons[family_id] = "nonfinite_relative_band_power"
        entropy_family = "BASELINE_SPECTRAL_ENTROPY_NORMALIZED"
        keep = (freqs >= 1.0) & (freqs <= 45.0)
        ent_psd = psd[:, keep]
        sums = np.sum(ent_psd, axis=1, dtype=np.float64)
        if ent_psd.shape[1] < 2 or np.any(~np.isfinite(sums)) or np.any(sums <= 0.0):
            reasons[entropy_family] = "invalid_spectral_entropy_psd_support"
        else:
            probabilities = ent_psd / sums[:, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                channel_entropy = -np.sum(
                    np.where(probabilities > 0.0,
                             probabilities * np.log(probabilities), 0.0),
                    axis=1, dtype=np.float64,
                ) / math.log(ent_psd.shape[1])
            if not np.isfinite(channel_entropy).all():
                reasons[entropy_family] = "nonfinite_spectral_entropy"
            else:
                value = float(np.median(channel_entropy))
                if math.isfinite(value):
                    values[entropy_family] = value
                else:
                    reasons[entropy_family] = "nonfinite_spectral_entropy"
    except Exception as exc:
        reason = f"welch_baseline_failure:{type(exc).__name__}:{str(exc)}"
        for family_id in ("BASELINE_ALPHA_LOG10_POWER", *BASELINE_BANDS,
                          "BASELINE_SPECTRAL_ENTROPY_NORMALIZED"):
            reasons.setdefault(family_id, reason)

    acf_family = "BASELINE_ACF_LAG1"
    try:
        left = signals[:, :-1]
        right = signals[:, 1:]
        left = left - np.mean(left, axis=1, keepdims=True, dtype=np.float64)
        right = right - np.mean(right, axis=1, keepdims=True, dtype=np.float64)
        left_norm = np.sqrt(np.sum(left * left, axis=1, dtype=np.float64))
        right_norm = np.sqrt(np.sum(right * right, axis=1, dtype=np.float64))
        if np.any(left_norm <= 0.0) or np.any(right_norm <= 0.0):
            raise ValueError("constant_required_channel")
        per_channel = np.sum(left * right, axis=1, dtype=np.float64) / (left_norm * right_norm)
        if not np.isfinite(per_channel).all():
            raise ValueError("nonfinite_acf_lag1")
        value = float(np.median(per_channel))
        if math.isfinite(value):
            values[acf_family] = value
        else:
            raise ValueError("nonfinite_acf_lag1_median")
    except Exception as exc:
        reasons[acf_family] = f"acf_lag1_unsupported:{type(exc).__name__}:{str(exc)}"

    corr_family = "BASELINE_MEAN_ABS_CHANNEL_CORRELATION"
    try:
        centered = signals - np.mean(signals, axis=1, keepdims=True, dtype=np.float64)
        norms = np.sqrt(np.sum(centered * centered, axis=1, dtype=np.float64))
        if np.any(norms <= 0.0):
            raise ValueError("constant_required_channel")
        correlations = (centered @ centered.T) / np.outer(norms, norms)
        correlations = (correlations + correlations.T) / 2.0
        if not np.isfinite(correlations).all():
            raise ValueError("nonfinite_channel_correlation")
        correlations = np.clip(correlations, -1.0, 1.0)
        upper = correlations[np.triu_indices(correlations.shape[0], k=1)]
        if upper.size == 0:
            raise ValueError("insufficient_channel_pairs")
        value = float(np.mean(np.abs(upper), dtype=np.float64))
        if math.isfinite(value):
            values[corr_family] = value
        else:
            raise ValueError("nonfinite_mean_abs_channel_correlation")
    except Exception as exc:
        reasons[corr_family] = (
            f"channel_correlation_unsupported:{type(exc).__name__}:{str(exc)}"
        )
    return values, reasons


def _baseline_records(
    run: gate_b.RunMetadata,
    window: CandidateWindow,
    signals: np.ndarray,
    labels: tuple[str, ...],
    acquisition_reference: str,
    source_fdt_sha256: str | None,
    *,
    failure_reason: str | None = None,
) -> list[dict[str, Any]]:
    reference_hash, source_ref_hash, reference_set_hash = gate_b._reference_provenance(
        labels, PRIMARY_REFERENCE_ID, acquisition_reference
    )
    window_hash = None if failure_reason is not None or signals is None else gate_b._window_hash(
        signals, labels, run.sample_rate_hz, PRIMARY_REFERENCE_ID
    )
    envelope = gate_b._base_record(
        run, window, source_fdt_sha256=source_fdt_sha256,
        source_ref=PRIMARY_REFERENCE_ID, acquisition_reference=acquisition_reference,
        reference_realization_hash=reference_hash,
        source_reference_metadata_hash=source_ref_hash,
        reference_set_hash=reference_set_hash, eeg_labels=labels,
        window_sha256=window_hash,
    )
    if failure_reason is not None:
        values: dict[str, float] = {}
        reasons = {family_id: f"source_window_unavailable:{failure_reason}"
                   for family_id in BASELINE_FAMILIES}
    else:
        values, reasons = _baseline_values(signals, run.sample_rate_hz)
    rows: list[dict[str, Any]] = []
    for family_id in BASELINE_FAMILIES:
        value = values.get(family_id)
        valid = value is not None and math.isfinite(float(value))
        rows.append({
            **envelope,
            "family_id": family_id,
            "estimator_id": BASELINE_ESTIMATORS[family_id],
            "computed": valid,
            "valid": valid,
            "qualified": False,
            "status": "valid" if valid else "unsupported",
            "value": float(value) if valid else None,
            "reason": None if valid else reasons.get(family_id, "baseline_value_unavailable"),
            "measurement_layer_role": "conventional_qeeg_comparator",
            "support_audit": {
                **envelope["support_audit"],
                "window_extraction_status": "failed" if failure_reason else "complete",
            },
        })
    return rows


def _core_failure_records(
    run: gate_b.RunMetadata,
    window: CandidateWindow,
    acquisition_reference: str,
    source_fdt_sha256: str | None,
    reason: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for reference_id in CORE_REFERENCE_IDS:
        ref_hash, source_ref_hash, ref_set_hash = gate_b._reference_provenance(
            run.eeg_channel_labels, reference_id, acquisition_reference
        )
        realization_id = (
            "source_reference" if reference_id == PRIMARY_REFERENCE_ID
            else "average_ref_sensitivity"
        )
        envelope = gate_b._base_record(
            run, window, source_fdt_sha256=source_fdt_sha256,
            source_ref=reference_id, acquisition_reference=acquisition_reference,
            reference_realization_hash=ref_hash,
            source_reference_metadata_hash=source_ref_hash,
            reference_set_hash=ref_set_hash, eeg_labels=run.eeg_channel_labels,
            window_sha256=None,
        )
        for family_id in CORE_FAMILIES:
            if family_id == "TP_RAW_COMPARATOR":
                estimator_id = "TP.raw_signal_pearson_persistence.v1"
                formula_kernel_id = estimator_id
            else:
                estimator_id = gate_b.FULL_VECTOR_CHALLENGE_001_BY_FAMILY[family_id].primary_estimator_id
                formula_kernel_id = None
            row = {
                **envelope,
                "family_id": family_id,
                "estimator_id": estimator_id,
                "formula_kernel_estimator_id": formula_kernel_id,
                "realization_id": realization_id,
                "computed": False,
                "valid": False,
                "qualified": False,
                "status": "unsupported",
                "value": None,
                "reason": f"source_window_unavailable:{reason}",
            }
            if family_id == "AR":
                row["component_values"] = [None] * 9
            rows.append(row)
    return rows


def _scg_status_records(
    run: gate_b.RunMetadata,
    selection: Mapping[str, Any],
    source_fdt_sha256: str | None,
) -> list[dict[str, Any]]:
    geometry = next((item for item in selection["scg_geometry_audit"]
                     if item["participant_id"] == run.participant_id), {})
    windows = dict(run.scg_windows)
    output: list[dict[str, Any]] = []
    for condition in ("EO", "EC"):
        window = windows.get(condition)
        geometry_available = geometry.get("geometry_status") == "available"
        output.append({
            "participant_id": run.participant_id,
            "run_id": run.run_id,
            "condition": condition,
            "window_id": window.window_id if window is not None else None,
            "support_start_sample": window.start_sample if window is not None else None,
            "support_end_sample_exclusive": window.end_sample_exclusive if window is not None else None,
            "sample_rate_hz": run.sample_rate_hz,
            "sample_count": window.sample_count if window is not None else None,
            "source_set_sha256": run.set_sha256,
            "source_fdt_sha256": source_fdt_sha256,
            "source_events_sha256": run.events_sha256,
            "source_electrodes_tsv_sha256": run.electrodes_sha256,
            "source_coordsystem_json_sha256": run.coordsystem_sha256,
            "family_id": "SCG",
            "estimator_id": gate_b.FULL_VECTOR_CHALLENGE_001_BY_FAMILY["SCG"].primary_estimator_id,
            "computed": False,
            "valid": False,
            "qualified": False,
            "status": "not_run_frozen_exclusion" if geometry_available else "unsupported",
            "value": None,
            "reason": (
                "science_lead_froze_SCG_outside_confirmatory_scope; no_value_computed"
                if geometry_available
                else str(geometry.get("reason", "required_BIDS_geometry_unavailable"))
            ),
            "geometry_audit": geometry,
            "support_audit": {
                "full_60s_windows_available_count": len([
                    item for item in next((row["scg_60s_windows"] for row in selection["eligible_roster"]
                                           if row["participant_id"] == run.participant_id), [])
                    if item["condition"] == condition
                ]),
                "geometry_status": geometry.get("geometry_status", "unknown"),
                "window_support_status": "not_computed_frozen_exclusion",
            },
        })
    return output


def _write_json_atomic(path: Path, value: Any) -> None:
    payload = json.dumps(_jsonable(value), indent=2, ensure_ascii=False,
                         sort_keys=True, allow_nan=False).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _participant_windows(record: Mapping[str, Any]) -> list[CandidateWindow]:
    output: list[CandidateWindow] = []
    for condition in ("EO", "EC"):
        output.extend(_window_from_record(item) for item in record["all_windows"][condition])
    return sorted(output, key=lambda item: (item.start_sample, item.condition, item.window_id))


def _expected_key_sets(selection: Mapping[str, Any], participant_ids: Sequence[str]) -> tuple[set[tuple[Any, ...]], set[tuple[Any, ...]], set[tuple[str, str]]]:
    core: set[tuple[Any, ...]] = set()
    baselines: set[tuple[Any, ...]] = set()
    scg: set[tuple[str, str]] = set()
    records = {str(item["participant_id"]): item for item in selection["eligible_roster"]}
    for participant_id in participant_ids:
        record = records[participant_id]
        for condition in ("EO", "EC"):
            for item in record["all_windows"][condition]:
                window_id = str(item["window_id"])
                for reference_id in CORE_REFERENCE_IDS:
                    for family_id in CORE_FAMILIES:
                        core.add((participant_id, condition, window_id, reference_id, family_id))
                for family_id in BASELINE_FAMILIES:
                    baselines.add((participant_id, condition, window_id, PRIMARY_REFERENCE_ID, family_id))
            scg.add((participant_id, condition))
    return core, baselines, scg


def _row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("participant_id")), str(row.get("condition")),
        str(row.get("window_id")), str(row.get("reference_id")),
        str(row.get("family_id")),
    )


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )


def _has_reason(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _audit_rows(rows: Iterable[Mapping[str, Any]], selection: Mapping[str, Any], participant_ids: Sequence[str]) -> dict[str, Any]:
    expected_core, expected_baselines, expected_scg = _expected_key_sets(selection, participant_ids)
    actual: dict[str, list[tuple[Any, ...]]] = {"core": [], "baseline": [], "scg": []}
    coverage: dict[tuple[str, str, str], dict[str, Any]] = {}
    component_coverage: dict[tuple[str, str, str], dict[str, Any]] = {}
    runner_exceptions: list[dict[str, Any]] = []
    unknown_family_ids: set[str] = set()
    component_audit_errors: list[dict[str, Any]] = []
    null_reason_audit = {
        "scalar_null_count": 0,
        "scalar_null_missing_reason_count": 0,
        "component_null_count": 0,
        "component_null_missing_reason_count": 0,
        "null_reason_counts": {},
    }
    all_rows_qualified_false = True
    for row in rows:
        if row.get("qualified") is not False:
            all_rows_qualified_false = False
        family_id = str(row.get("family_id"))
        if family_id in CORE_FAMILIES:
            group = "core"
            key = _row_key(row)
        elif family_id in BASELINE_FAMILIES:
            group = "baseline"
            key = _row_key(row)
        elif family_id == "SCG":
            group = "scg"
            key = (str(row.get("participant_id")), str(row.get("condition")))
        else:
            unknown_family_ids.add(family_id)
            continue
        actual[group].append(key)
        reference = str(row.get("reference_id", "separate_scg"))
        cov_key = (family_id, str(row.get("condition")), reference)
        stat = coverage.setdefault(cov_key, {
            "row_count": 0, "finite_value_count": 0, "valid_finite_value_count": 0,
            "valid_record_count": 0, "status_counts": {},
            "value_representation": "ordered_vector" if family_id == "AR" else "scalar",
        })
        stat["row_count"] += 1
        value = row.get("value")
        if family_id == "AR":
            component_values = row.get("component_values")
            finite = (
                isinstance(component_values, (list, tuple))
                and len(component_values) == len(AR_COMPONENT_NAMES)
                and all(_finite_number(component) for component in component_values)
            )
        else:
            finite = _finite_number(value)
        if finite:
            stat["finite_value_count"] += 1
        if row.get("valid") is True:
            stat["valid_record_count"] += 1
            if finite:
                stat["valid_finite_value_count"] += 1
        status = str(row.get("status", "missing_status"))
        stat["status_counts"][status] = stat["status_counts"].get(status, 0) + 1
        reason = str(row.get("reason", ""))
        if family_id == "AR":
            component_values = row.get("component_values")
            component_names = row.get("component_names")
            component_statuses = row.get("component_status")
            component_reasons = row.get("component_reasons")
            shapes = (component_values, component_names, component_statuses, component_reasons)
            if not all(isinstance(items, (list, tuple)) for items in shapes) or any(
                len(items) != len(AR_COMPONENT_NAMES) for items in shapes
            ):
                component_audit_errors.append({
                    "participant_id": row.get("participant_id"),
                    "condition": row.get("condition"),
                    "window_id": row.get("window_id"),
                    "reference_id": reference,
                    "reason": "component_vector_shape_or_length_mismatch",
                })
            else:
                if tuple(component_names) != AR_COMPONENT_NAMES:
                    component_audit_errors.append({
                        "participant_id": row.get("participant_id"),
                        "condition": row.get("condition"),
                        "window_id": row.get("window_id"),
                        "reference_id": reference,
                        "reason": "component_names_or_order_mismatch",
                        "component_names": list(component_names),
                    })
                for index, component_name in enumerate(AR_COMPONENT_NAMES):
                    component_value = component_values[index]
                    component_status = str(component_statuses[index] or "missing_status")
                    component_reason = component_reasons[index]
                    component_finite = _finite_number(component_value)
                    component_stat = component_coverage.setdefault(
                        (str(row.get("condition")), reference, component_name),
                        {
                            "family_id": "AR",
                            "component_name": component_name,
                            "row_count": 0,
                            "finite_value_count": 0,
                            "valid_finite_value_count": 0,
                            "valid_component_count": 0,
                            "null_count": 0,
                            "component_status_counts": {},
                            "null_reason_counts": {},
                        },
                    )
                    component_stat["row_count"] += 1
                    component_stat["valid_component_count"] += component_status == "computed_exploratory"
                    if component_finite:
                        component_stat["finite_value_count"] += 1
                        if row.get("valid") is True:
                            component_stat["valid_finite_value_count"] += 1
                    component_stat["component_status_counts"][component_status] = (
                        component_stat["component_status_counts"].get(component_status, 0) + 1
                    )
                    if component_value is None:
                        null_reason_audit["component_null_count"] += 1
                        component_stat["null_count"] += 1
                        if _has_reason(component_reason):
                            reasons = null_reason_audit["null_reason_counts"]
                            reason_key = f"AR:{component_reason}"
                            reasons[reason_key] = reasons.get(reason_key, 0) + 1
                            component_stat["null_reason_counts"][str(component_reason)] = (
                                component_stat["null_reason_counts"].get(str(component_reason), 0) + 1
                            )
                        else:
                            null_reason_audit["component_null_missing_reason_count"] += 1
                            component_audit_errors.append({
                                "participant_id": row.get("participant_id"),
                                "condition": row.get("condition"),
                                "window_id": row.get("window_id"),
                                "reference_id": reference,
                                "component_name": component_name,
                                "reason": "null_component_missing_candidate_local_reason",
                            })
                    elif not component_finite:
                        component_audit_errors.append({
                            "participant_id": row.get("participant_id"),
                            "condition": row.get("condition"),
                            "window_id": row.get("window_id"),
                            "reference_id": reference,
                            "component_name": component_name,
                            "reason": "component_value_not_finite_numeric",
                        })
                    if component_value is None and component_status != "unsupported":
                        component_audit_errors.append({
                            "participant_id": row.get("participant_id"),
                            "condition": row.get("condition"),
                            "window_id": row.get("window_id"),
                            "reference_id": reference,
                            "component_name": component_name,
                            "reason": "null_component_status_not_unsupported",
                        })
                    elif component_finite and component_status == "unsupported":
                        component_audit_errors.append({
                            "participant_id": row.get("participant_id"),
                            "condition": row.get("condition"),
                            "window_id": row.get("window_id"),
                            "reference_id": reference,
                            "component_name": component_name,
                            "reason": "finite_component_marked_unsupported",
                        })
        elif value is None:
            null_reason_audit["scalar_null_count"] += 1
            if _has_reason(row.get("reason")):
                reasons = null_reason_audit["null_reason_counts"]
                reason_key = f"{family_id}:{row['reason']}"
                reasons[reason_key] = reasons.get(reason_key, 0) + 1
            else:
                null_reason_audit["scalar_null_missing_reason_count"] += 1
                component_audit_errors.append({
                    "participant_id": row.get("participant_id"),
                    "condition": row.get("condition"),
                    "window_id": row.get("window_id"),
                    "family_id": family_id,
                    "reason": "null_scalar_missing_candidate_local_reason",
                })
        if reason.startswith("runner_exception:"):
            runner_exceptions.append({
                "participant_id": row.get("participant_id"), "condition": row.get("condition"),
                "window_id": row.get("window_id"), "family_id": family_id, "reason": reason,
            })
    audits = {}
    expected_map = {"core": expected_core, "baseline": expected_baselines, "scg": expected_scg}
    for group, expected in expected_map.items():
        key_counts: dict[tuple[Any, ...], int] = {}
        for key in actual[group]:
            key_counts[key] = key_counts.get(key, 0) + 1
        observed = set(key_counts)
        audits[group] = {
            "expected_key_count": len(expected),
            "actual_row_count": len(actual[group]),
            "unique_key_count": len(observed),
            "missing_keys": [list(key) for key in sorted(expected - observed)[:100]],
            "duplicate_keys": [list(key) for key, count in sorted(key_counts.items()) if count > 1][:100],
            "unexpected_keys": [list(key) for key in sorted(observed - expected)[:100]],
            "passed": observed == expected and all(count == 1 for count in key_counts.values()),
        }
    return {
        "exact_key_audits": audits,
        "exact_key_audits_passed": (
            all(item["passed"] for item in audits.values())
            and not unknown_family_ids
        ),
        "unknown_family_ids": sorted(unknown_family_ids),
        "coverage": [
            {"family_id": key[0], "condition": key[1], "reference_id": key[2], **value}
            for key, value in sorted(coverage.items())
        ],
        "component_coverage": [
            {"condition": key[0], "reference_id": key[1], **value}
            for key, value in sorted(component_coverage.items())
        ],
        "component_audit_passed": not component_audit_errors,
        "component_audit_error_count": len(component_audit_errors),
        "component_audit_errors": component_audit_errors[:100],
        "null_reason_audit": null_reason_audit,
        "null_reason_audit_passed": (
            null_reason_audit["scalar_null_missing_reason_count"] == 0
            and null_reason_audit["component_null_missing_reason_count"] == 0
        ),
        "runner_exception_count": len(runner_exceptions),
        "runner_exceptions": runner_exceptions[:100],
        "all_measurement_rows_qualified_false": all_rows_qualified_false,
    }


def _code_source_hashes() -> dict[str, str]:
    v32 = Path(__file__).parents[1]
    v31 = Path(__file__).parents[2] / "v3_1"
    sources = {
        "gate_c1_runner": Path(__file__),
        "gate_b_runner_and_estimators": Path(gate_b.__file__),
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
        "scg_support": v31 / "candidates.py",
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


def _prepare_run_directory(
    output_root: Path,
    mode: str,
    selection: Mapping[str, Any],
    selection_path: Path,
    resume_run_dir: Path | None,
) -> tuple[Path, dict[str, Any]]:
    code_hashes = _code_source_hashes()
    signature = {
        "mode": mode,
        "selection_manifest_sha256": _sha256_file(selection_path),
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "base_protocol_sha256": _sha256_file(BASE_PROTOCOL_PATH),
        "implementation_code_sha256": code_hashes,
    }
    if resume_run_dir is not None:
        run_root = resume_run_dir
        if not run_root.is_dir():
            raise FileNotFoundError(f"resume_run_directory_missing:{run_root}")
        state_path = run_root / "run_state.json"
        if not state_path.is_file():
            raise RuntimeError("resume_run_state_missing")
        state = _load_json(state_path)
        if state.get("signature") != signature:
            raise RuntimeError("resume_signature_mismatch_selection_protocol_or_code")
        return run_root, state
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = output_root / f"{mode}_{stamp}"
    (run_root / "participants").mkdir(parents=True, exist_ok=False)
    state = {"signature": signature, "participant_status": {}}
    _write_json_atomic(run_root / "run_state.json", state)
    return run_root, state


def _write_participant_shard(
    run_root: Path,
    participant_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    participant_dir = run_root / "participants"
    participant_dir.mkdir(parents=True, exist_ok=True)
    final_path = participant_dir / f"{participant_id}.jsonl.gz"
    temporary = final_path.with_suffix(final_path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="\n", compresslevel=6) as stream:
        for row in rows:
            stream.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True,
                                    allow_nan=False) + "\n")
    os.replace(temporary, final_path)
    return {
        "participant_id": participant_id,
        "status": "accounted",
        "row_count": len(rows),
        "shard_path": str(final_path),
        "shard_sha256": _sha256_file(final_path),
    }


def _rows_for_participant(
    run: gate_b.RunMetadata,
    record: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_fdt_sha256: str | None = None
    acquisition_reference = "unknown"
    data = None
    participant_failure: str | None = None
    try:
        gate_b._verify_frozen_inputs(run)
        eeg_metadata = gate_b._read_json(Path(run.eeg_json_path))
        acquisition_reference = str(eeg_metadata.get("EEGReference", "unknown"))
        data, source_fdt_sha256 = gate_b._source_memmap(run)
    except Exception as exc:
        participant_failure = f"{type(exc).__name__}:{str(exc)}"

    windows = _participant_windows(record)
    for index, window in enumerate(windows, start=1):
        print(
            f"[C1] {run.participant_id} {window.condition} {index}/{len(windows)} "
            f"{window.window_id}: extract 8 families + baselines",
            flush=True,
        )
        if participant_failure is not None or data is None:
            reason = participant_failure or "source_memmap_unavailable"
            rows.extend(_core_failure_records(run, window, acquisition_reference,
                                              source_fdt_sha256, reason))
            rows.extend(_baseline_records(
                run, window, np.empty((0, 0), dtype=np.float64), run.eeg_channel_labels,
                acquisition_reference, source_fdt_sha256, failure_reason=reason,
            ))
            continue
        try:
            source_signals, labels = gate_b._window_data(data, run, window)
            if labels != run.eeg_channel_labels:
                raise ValueError("frozen_EEG_label_order_changed_since_selection")
        except Exception as exc:
            reason = f"{type(exc).__name__}:{str(exc)}"
            rows.extend(_core_failure_records(run, window, acquisition_reference,
                                              source_fdt_sha256, reason))
            rows.extend(_baseline_records(
                run, window, np.empty((0, 0), dtype=np.float64), run.eeg_channel_labels,
                acquisition_reference, source_fdt_sha256, failure_reason=reason,
            ))
            continue
        try:
            rows.extend(gate_b._compute_window_family_records(
                run, window, source_signals, labels, acquisition_reference,
                source_fdt_sha256,
            ))
        except Exception as exc:
            reason = f"{type(exc).__name__}:{str(exc)}"
            rows.extend(_core_failure_records(run, window, acquisition_reference,
                                              source_fdt_sha256, reason))
        rows.extend(_baseline_records(
            run, window, source_signals, labels, acquisition_reference,
            source_fdt_sha256,
        ))
    rows.extend(_scg_status_records(run, selection, source_fdt_sha256))
    if data is not None:
        del data
    status = {
        "participant_id": run.participant_id,
        "run_id": run.run_id,
        "source_fdt_sha256": source_fdt_sha256,
        "processing_status": "source_failed" if participant_failure else "processed",
        "failure_reason": participant_failure,
        "window_count": len(windows),
        "condition_window_counts": {
            condition: len(record["all_windows"][condition]) for condition in ("EO", "EC")
        },
        "row_count": len(rows),
    }
    return rows, status


def _read_shard(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            rows.append(json.loads(line))
    return rows


def _iter_shards(run_root: Path, participant_ids: Sequence[str]) -> Iterable[dict[str, Any]]:
    for participant_id in participant_ids:
        path = run_root / "participants" / f"{participant_id}.jsonl.gz"
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                yield json.loads(line)


def _combine_shards(run_root: Path, participant_ids: Sequence[str]) -> Path:
    combined = run_root / "measurements.jsonl.gz"
    temporary = combined.with_suffix(combined.suffix + ".tmp")
    with temporary.open("wb") as out:
        for participant_id in participant_ids:
            shard = run_root / "participants" / f"{participant_id}.jsonl.gz"
            if not shard.is_file():
                raise FileNotFoundError(f"participant_shard_missing:{participant_id}")
            with shard.open("rb") as src:
                shutil.copyfileobj(src, out, length=1024 * 1024)
    os.replace(temporary, combined)
    return combined


def run_gate_c1(
    selection: Mapping[str, Any],
    *,
    selection_path: Path,
    output_root: Path,
    mode: str,
    resume_run_dir: Path | None = None,
) -> Path:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode_must_be_smoke_or_full")
    selected_ids = [str(item["participant_id"]) for item in selection["eligible_roster"]]
    if mode == "smoke":
        participant_ids = selected_ids[:1]
    else:
        participant_ids = selected_ids
    if mode == "full" and participant_ids != selected_ids:
        raise RuntimeError("full_mode_must_process_entire_frozen_roster")
    run_root, state = _prepare_run_directory(
        output_root, mode, selection, selection_path, resume_run_dir
    )
    records = {str(item["participant_id"]): item for item in selection["eligible_roster"]}
    status_by_id = dict(state.get("participant_status", {}))
    for position, participant_id in enumerate(participant_ids, start=1):
        existing = status_by_id.get(participant_id)
        if existing:
            shard = Path(existing.get("shard_path", ""))
            if (shard.is_file() and _sha256_file(shard) == existing.get("shard_sha256")):
                print(f"[C1] resume {position}/{len(participant_ids)} {participant_id}: verified shard", flush=True)
                continue
            raise RuntimeError(f"resume_shard_missing_or_hash_changed:{participant_id}")
        run = _metadata_record_to_run(records[participant_id])
        participant_started = time.perf_counter()
        rows, processing = _rows_for_participant(run, records[participant_id], selection)
        shard_status = _write_participant_shard(run_root, participant_id, rows)
        status_by_id[participant_id] = {**processing, **shard_status,
                                        "elapsed_seconds": time.perf_counter() - participant_started}
        state["participant_status"] = status_by_id
        _write_json_atomic(run_root / "run_state.json", state)
        print(
            f"[C1] completed {position}/{len(participant_ids)} {participant_id}: "
            f"{len(rows)} rows in {status_by_id[participant_id]['elapsed_seconds']:.1f}s",
            flush=True,
        )

    combined_path = _combine_shards(run_root, participant_ids)
    audit = _audit_rows(_iter_shards(run_root, participant_ids), selection, participant_ids)
    measurement_row_count = sum(int(status_by_id[participant_id].get("row_count", 0))
                                for participant_id in participant_ids)
    if mode == "smoke":
        audit["scope_status"] = "SMOKE_ONLY_NOT_A_GATE_C1_COHORT_RESULT"
    else:
        audit["scope_status"] = "FULL_COHORT_EXTRACTION"
    participant_status = [status_by_id[participant_id] for participant_id in participant_ids]
    summary = {
        "protocol_id": PROTOCOL_ID,
        "gate_id": GATE_ID,
        "mode": mode,
        "qualified": False,
        "outcome_blind": True,
        "clinical_metadata_read": False,
        "clinical_paths_read": [],
        "participant_count_expected": len(selected_ids),
        "participant_count_processed": len(participant_ids),
        "participant_status": participant_status,
        "window_count_expected": sum(
            len(records[participant_id]["all_windows"][condition])
            for participant_id in participant_ids for condition in ("EO", "EC")
        ),
        "measurement_row_count": measurement_row_count,
        "measurement_file": str(combined_path),
        "measurement_file_sha256": _sha256_file(combined_path),
        "selection_manifest_path": str(selection_path),
        "selection_manifest_sha256": _sha256_file(selection_path),
        "gate_b_selection_sha256": selection["gate_b_selection_sha256"],
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "base_protocol_sha256": _sha256_file(BASE_PROTOCOL_PATH),
        "dataset_description_sha256": selection["dataset_description_sha256"],
        "scg_geometry_status_counts": dict(sorted({
            str(item.get("geometry_status", "unknown")): sum(
                row.get("geometry_status") == item.get("geometry_status")
                for row in selection["scg_geometry_audit"]
            )
            for item in selection["scg_geometry_audit"]
        }.items())),
        "audit": audit,
        "c1_extraction_audit_passed": (
            mode == "full" and len(participant_ids) == len(selected_ids)
            and audit["exact_key_audits_passed"]
            and audit["all_measurement_rows_qualified_false"]
            and audit["component_audit_passed"]
            and audit["null_reason_audit_passed"]
            and not audit["runner_exception_count"]
            and all(item.get("status") == "accounted"
                    and item.get("processing_status") == "processed"
                    for item in participant_status)
        ),
        "smoke_check_passed": (
            mode == "smoke" and audit["exact_key_audits_passed"]
            and audit["all_measurement_rows_qualified_false"]
            and audit["component_audit_passed"]
            and audit["null_reason_audit_passed"]
            and not audit["runner_exception_count"]
            and all(item.get("status") == "accounted"
                    and item.get("processing_status") == "processed"
                    for item in participant_status)
        ),
        "gate_c1_status": "NOT_EVALUATED" if mode == "smoke" else (
            "PASS" if (
                len(participant_ids) == len(selected_ids)
                and audit["exact_key_audits_passed"]
                and audit["all_measurement_rows_qualified_false"]
                and audit["component_audit_passed"]
                and audit["null_reason_audit_passed"]
                and not audit["runner_exception_count"]
                and all(status_by_id[participant_id].get("processing_status") == "processed"
                        for participant_id in participant_ids)
            ) else "FAIL"
        ),
        "no_condition_contrasts_computed": True,
        "no_clinical_labels_opened": True,
        "participant_level_aggregates_computed": False,
        "participant_level_p_values_computed": False,
        "source_fdt_sha256_by_participant": {
            item["participant_id"]: item.get("source_fdt_sha256") for item in participant_status
        },
        "elapsed_seconds": sum(float(item.get("elapsed_seconds", 0.0)) for item in participant_status),
        "measurement_coverage": audit["coverage"],
        "measurement_component_coverage": audit["component_coverage"],
    }
    _write_json_atomic(run_root / "summary.json", summary)
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "gate_id": GATE_ID,
        "mode": mode,
        "qualified": False,
        "outcome_blind": True,
        "clinical_metadata_read": False,
        "selection_manifest_sha256": _sha256_file(selection_path),
        "measurement_file_sha256": _sha256_file(combined_path),
        "summary_sha256": _sha256_file(run_root / "summary.json"),
        "gate_c1_runner_sha256": _sha256_file(Path(__file__)),
        "gate_b_runner_sha256": _sha256_file(Path(gate_b.__file__)),
        "implementation_code_sha256": _code_source_hashes(),
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "base_protocol_sha256": _sha256_file(BASE_PROTOCOL_PATH),
        "gate_b_selection_sha256": selection["gate_b_selection_sha256"],
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "scipy_version": __import__("scipy").__version__,
        "numba_version": _optional_package_version("numba"),
        "gzip_member_shards": [
            {
                "participant_id": participant_id,
                "path": status_by_id[participant_id]["shard_path"],
                "sha256": status_by_id[participant_id]["shard_sha256"],
            }
            for participant_id in participant_ids
        ],
        "rows_path": str(combined_path),
    }
    _write_json_atomic(run_root / "manifest.json", manifest)
    return run_root


def refresh_existing_run_audit(
    run_root: Path,
    selection: Mapping[str, Any],
    selection_path: Path,
) -> Path:
    """Re-audit a completed C1 run without reprocessing signal data or rewriting shards."""
    state_path = run_root / "run_state.json"
    summary_path = run_root / "summary.json"
    manifest_path = run_root / "manifest.json"
    measurement_path = run_root / "measurements.jsonl.gz"
    state = _load_json(state_path)
    summary = _load_json(summary_path)
    manifest = _load_json(manifest_path)
    signature = state.get("signature", {})
    participant_ids = [str(item["participant_id"]) for item in selection["eligible_roster"]]
    expected_signature = {
        "mode": "full",
        "selection_manifest_sha256": _sha256_file(selection_path),
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "base_protocol_sha256": _sha256_file(BASE_PROTOCOL_PATH),
    }
    for key, expected in expected_signature.items():
        if signature.get(key) != expected or manifest.get(key) != expected:
            raise RuntimeError(f"audit_refresh_frozen_signature_mismatch:{key}")
    execution_code_hashes = signature.get("implementation_code_sha256")
    if not isinstance(execution_code_hashes, dict) or manifest.get("implementation_code_sha256") != execution_code_hashes:
        raise RuntimeError("audit_refresh_execution_code_hash_mismatch")
    if manifest.get("gate_c1_runner_sha256") != execution_code_hashes.get("gate_c1_runner"):
        raise RuntimeError("audit_refresh_runner_hash_mismatch")
    if summary.get("measurement_file_sha256") != manifest.get("measurement_file_sha256"):
        raise RuntimeError("audit_refresh_measurement_manifest_mismatch")
    if not measurement_path.is_file() or _sha256_file(measurement_path) != manifest.get("measurement_file_sha256"):
        raise RuntimeError("audit_refresh_measurement_hash_mismatch")

    status_by_id = state.get("participant_status", {})
    if set(status_by_id) != set(participant_ids):
        raise RuntimeError("audit_refresh_participant_roster_mismatch")
    for participant_id in participant_ids:
        item = status_by_id[participant_id]
        expected_shard = (run_root / "participants" / f"{participant_id}.jsonl.gz").resolve()
        shard = Path(item.get("shard_path", "")).resolve()
        if shard != expected_shard or not shard.is_file():
            raise RuntimeError(f"audit_refresh_shard_path_mismatch:{participant_id}")
        if _sha256_file(shard) != item.get("shard_sha256"):
            raise RuntimeError(f"audit_refresh_shard_hash_mismatch:{participant_id}")
        if item.get("status") != "accounted" or item.get("processing_status") != "processed":
            raise RuntimeError(f"audit_refresh_participant_not_processed:{participant_id}")

    audit = _audit_rows(_iter_shards(run_root, participant_ids), selection, participant_ids)
    audit["scope_status"] = "FULL_COHORT_EXTRACTION"
    measurement_row_count = sum(int(status_by_id[participant_id].get("row_count", 0))
                                for participant_id in participant_ids)
    if measurement_row_count != summary.get("measurement_row_count"):
        raise RuntimeError("audit_refresh_measurement_row_count_mismatch")
    summary_participants = {str(item.get("participant_id")): item
                            for item in summary.get("participant_status", [])}
    if set(summary_participants) != set(participant_ids):
        raise RuntimeError("audit_refresh_summary_roster_mismatch")
    for participant_id in participant_ids:
        if summary_participants[participant_id].get("shard_sha256") != status_by_id[participant_id].get("shard_sha256"):
            raise RuntimeError(f"audit_refresh_summary_shard_hash_mismatch:{participant_id}")

    passed = (
        audit["exact_key_audits_passed"]
        and audit["all_measurement_rows_qualified_false"]
        and audit["component_audit_passed"]
        and audit["null_reason_audit_passed"]
        and not audit["runner_exception_count"]
        and all(status_by_id[participant_id].get("processing_status") == "processed"
                for participant_id in participant_ids)
    )
    summary.update({
        "audit": audit,
        "c1_extraction_audit_passed": passed,
        "gate_c1_status": "PASS" if passed else "FAIL",
        "measurement_coverage": audit["coverage"],
        "measurement_component_coverage": audit["component_coverage"],
        "audit_refresh": {
            "method": "existing_verified_shards_only",
            "signal_reprocessing_performed": False,
            "audit_runner_sha256": _sha256_file(Path(__file__)),
        },
    })
    _write_json_atomic(summary_path, summary)
    manifest["summary_sha256"] = _sha256_file(summary_path)
    manifest["audit_refresh"] = {
        "method": "existing_verified_shards_only",
        "signal_reprocessing_performed": False,
        "audit_runner_sha256": _sha256_file(Path(__file__)),
    }
    _write_json_atomic(manifest_path, manifest)
    return run_root


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--gate-b-selection", type=Path, default=DEFAULT_GATE_B_SELECTION_PATH)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume-run-dir", type=Path, default=None,
                        help="resume an interrupted C1 extraction after verifying its frozen signature")
    parser.add_argument("--refresh-audit-run-dir", type=Path, default=None,
                        help="refresh audit metadata from an existing completed full C1 run without rewriting measurements")
    args = parser.parse_args(argv)
    if args.resume_run_dir is not None and args.refresh_audit_run_dir is not None:
        parser.error("--resume-run-dir and --refresh-audit-run-dir are mutually exclusive")
    selection = freeze_or_load_selection(args.data_root, args.gate_b_selection, args.selection)
    print(
        f"Gate C1 selection frozen: {args.selection} | "
        f"eligible={selection['eligible_count']} windows="
        f"{sum(len(r['all_windows']['EO']) + len(r['all_windows']['EC']) for r in selection['eligible_roster'])}",
        flush=True,
    )
    if args.refresh_audit_run_dir is not None:
        run_root = refresh_existing_run_audit(
            args.refresh_audit_run_dir, selection, args.selection
        )
        print(f"Gate C1 audit refresh output: {run_root}", flush=True)
        return 0
    run_root = run_gate_c1(
        selection, selection_path=args.selection, output_root=args.output_root,
        mode=args.mode, resume_run_dir=args.resume_run_dir,
    )
    print(f"Gate C1 {args.mode} output: {run_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
