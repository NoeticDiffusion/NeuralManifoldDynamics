"""Native fNIRS channel-window descriptors with explicit support provenance.

This module deliberately does not assign channel/window QC or qualification.
Its window statistics describe computed HbO/HbR samples only.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping
from urllib.parse import quote

import numpy as np
import pandas as pd


FEATURE_CONTRACT = "mndm.fnirs_features.v0.1"
_TIME_TOLERANCE_REL = 1e-6


class FnirsFeatureContractError(ValueError):
    """fNIRS feature config or support metadata is missing or inconsistent."""


def _feature_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    features = config.get("features", {}) if isinstance(config, Mapping) else {}
    fnirs = features.get("fnirs") if isinstance(features, Mapping) else None
    if not isinstance(fnirs, Mapping):
        raise FnirsFeatureContractError(
            "features.fnirs must explicitly configure window_length_s, "
            "window_step_s and min_valid_support_fraction; no EEG or global epoch defaults are used."
        )
    required = ("window_length_s", "window_step_s", "min_valid_support_fraction")
    missing = [key for key in required if key not in fnirs]
    if missing:
        raise FnirsFeatureContractError(f"features.fnirs is missing required setting(s): {', '.join(missing)}.")
    try:
        length_s = float(fnirs["window_length_s"])
        step_s = float(fnirs["window_step_s"])
        min_support = float(fnirs["min_valid_support_fraction"])
    except (TypeError, ValueError) as exc:
        raise FnirsFeatureContractError("fNIRS window settings must be numeric seconds/fractions.") from exc
    if not math.isfinite(length_s) or length_s <= 0:
        raise FnirsFeatureContractError("features.fnirs.window_length_s must be finite and greater than zero.")
    if not math.isfinite(step_s) or step_s <= 0:
        raise FnirsFeatureContractError("features.fnirs.window_step_s must be finite and greater than zero.")
    if not math.isfinite(min_support) or not 0 < min_support <= 1:
        raise FnirsFeatureContractError("features.fnirs.min_valid_support_fraction must be in (0, 1].")
    return {"window_length_s": length_s, "window_step_s": step_s, "min_valid_support_fraction": min_support}


def _payload_channels(payload: Mapping[str, Any], n_channels: int) -> tuple[list[str], list[str]]:
    raw_channels = payload.get("channels")
    if isinstance(raw_channels, Mapping):
        raw_channels = raw_channels.get("fnirs")
    if not isinstance(raw_channels, (list, tuple)) or len(raw_channels) != n_channels:
        raise FnirsFeatureContractError("fNIRS channel names must match the signal's channel dimension.")
    names = [str(name) for name in raw_channels]
    meta = payload.get("meta", {})
    fnirs_meta = meta.get("fnirs", {}) if isinstance(meta, Mapping) else {}
    raw_types = fnirs_meta.get("channel_types") if isinstance(fnirs_meta, Mapping) else None
    units = fnirs_meta.get("channel_units") if isinstance(fnirs_meta, Mapping) else None
    if units != "M (molar; MNE FIFF_UNIT_MOL)":
        raise FnirsFeatureContractError("fNIRS Hb channels must declare the verified MNE molar unit.")
    if not isinstance(raw_types, (list, tuple)) or len(raw_types) != n_channels:
        raise FnirsFeatureContractError("fNIRS channel_types must identify every HbO/HbR signal channel.")
    types = [str(value).lower() for value in raw_types]
    if any(value not in {"hbo", "hbr"} for value in types):
        raise FnirsFeatureContractError("G3 supports native HbO/HbR channels only.")
    if len(set(names)) != len(names):
        raise FnirsFeatureContractError("fNIRS channel names must be unique semantic identities.")
    return names, types


def _preprocessing_provenance(payload: Mapping[str, Any]) -> tuple[str, str, str]:
    meta = payload.get("meta", {})
    fnirs_meta = meta.get("fnirs", {}) if isinstance(meta, Mapping) else {}
    provenance = fnirs_meta.get("preprocessing_provenance") if isinstance(fnirs_meta, Mapping) else None
    if not isinstance(provenance, Mapping):
        raise FnirsFeatureContractError("fNIRS preprocessing_provenance is required for feature output.")
    source = provenance.get("source")
    if not isinstance(source, Mapping):
        raise FnirsFeatureContractError("fNIRS preprocessing provenance must identify the source recording.")
    source_digest = str(source.get("sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", source_digest):
        raise FnirsFeatureContractError("fNIRS source provenance must include a valid SHA-256 digest.")
    classification = str(provenance.get("input_classification", ""))
    operations = provenance.get("operations")
    if not classification or not isinstance(operations, (list, tuple)):
        raise FnirsFeatureContractError("fNIRS preprocessing provenance lacks input classification or executed operations.")
    if provenance.get("haemoglobin_units") != "M (molar; MNE FIFF_UNIT_MOL)":
        raise FnirsFeatureContractError("fNIRS preprocessing provenance lacks the verified MNE molar unit.")
    if not any(
        isinstance(operation, Mapping)
        and operation.get("name") == "beer_lambert_law"
        and operation.get("status") == "applied"
        for operation in operations
    ):
        raise FnirsFeatureContractError("fNIRS feature input must record an applied Beer-Lambert conversion.")
    if provenance.get("measurement_validity") != "not_assessed" or provenance.get("claim_status") != "no_biological_claim":
        raise FnirsFeatureContractError("fNIRS computed features may not upgrade G2 validity or biological claim status.")
    compact = {
        "schema": provenance.get("schema"),
        "source_sha256": source_digest,
        "input_classification": classification,
        "software": provenance.get("software"),
        "operations": operations,
        "haemoglobin_units": provenance.get("haemoglobin_units"),
        "measurement_validity": "not_assessed",
        "claim_status": "no_biological_claim",
    }
    return source_digest, classification, json.dumps(compact, sort_keys=True, separators=(",", ":"))


def _support_clock(payload: Mapping[str, Any], n_samples: int, sfreq: float) -> tuple[np.ndarray, np.ndarray, list[int]]:
    meta = payload.get("meta", {})
    support = meta.get("signal_support_provenance", {}) if isinstance(meta, Mapping) else {}
    if not isinstance(support, Mapping):
        raise FnirsFeatureContractError("signal_support_provenance is required for fNIRS windows.")
    if int(support.get("sample_count", -1)) != n_samples:
        raise FnirsFeatureContractError("fNIRS support sample_count does not match the signal array.")
    try:
        declared_sfreq = float(support["sampling_frequency_hz"])
        time_start = float(support["time_start_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FnirsFeatureContractError("fNIRS support must include sampling_frequency_hz and time_start_s.") from exc
    if not math.isfinite(declared_sfreq) or declared_sfreq <= 0 or not np.isclose(declared_sfreq, sfreq, rtol=1e-9, atol=1e-12):
        raise FnirsFeatureContractError("fNIRS support sampling frequency is invalid or disagrees with the payload.")
    if not math.isfinite(time_start):
        raise FnirsFeatureContractError("fNIRS time_start_s must be finite.")

    raw_times = support.get("sample_times_s")
    if raw_times is None:
        if support.get("time_gap_sample_indices"):
            raise FnirsFeatureContractError(
                "A gapped recording requires the original sample_times_s; a uniform sfreq clock cannot reconstruct it."
            )
        times = time_start + np.arange(n_samples, dtype=float) / sfreq
    else:
        times = np.asarray(raw_times, dtype=float)
    if times.shape != (n_samples,) or not np.isfinite(times).all() or (n_samples > 1 and np.any(np.diff(times) <= 0)):
        raise FnirsFeatureContractError("fNIRS sample_times_s must be finite, strictly increasing and match the signal.")
    if not np.isclose(times[0], time_start, rtol=1e-9, atol=1e-9):
        raise FnirsFeatureContractError("fNIRS first sample time disagrees with signal support time_start_s.")

    nominal_dt = 1.0 / sfreq
    tolerance = max(nominal_dt * _TIME_TOLERANCE_REL, 1e-12)
    segment_ids = np.zeros(n_samples, dtype=np.int64)
    if n_samples > 1:
        segment_ids[1:] = np.cumsum(np.diff(times) > nominal_dt + tolerance)
    gaps = [int(i) for i in np.flatnonzero(np.diff(times) > nominal_dt + tolerance) + 1]
    declared_gaps = support.get("time_gap_sample_indices", [])
    if declared_gaps and [int(i) for i in declared_gaps] != gaps:
        raise FnirsFeatureContractError("Declared fNIRS gap indices disagree with sample_times_s.")
    return times, segment_ids, gaps


def _channel_column(name: str, metric: str) -> str:
    key = quote(name, safe="")
    return f"fnirs_{metric}__{key}"


def compute_fnirs_features(signals: Mapping[str, Any], config: Mapping[str, Any]) -> pd.DataFrame:
    """Compute channel-level HbO/HbR mean and SD on an explicit seconds grid.

    The grid anchors to the first SNIRF sample, uses half-open windows, and does
    not reset at gaps. All candidate window IDs are retained. Partial recording
    windows and any clock-gap crossing are represented with missing feature
    values plus reason codes. Window support is temporal coverage; it is not
    signal QC. ``qc_ok_fnirs`` therefore remains unknown (nullable NA).
    Statistics are time-weighted over each sample's recorded support interval,
    clipped at the window boundaries.
    """
    signal_map = signals.get("signals", {})
    if not isinstance(signal_map, Mapping) or "fnirs" not in signal_map:
        return pd.DataFrame()
    data = np.asarray(signal_map["fnirs"], dtype=float)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise FnirsFeatureContractError("signals['fnirs'] must have non-empty (channel, sample) shape.")
    try:
        sfreq = float(signals["sfreq"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FnirsFeatureContractError("fNIRS sampling frequency is required.") from exc
    if not math.isfinite(sfreq) or sfreq <= 0:
        raise FnirsFeatureContractError("fNIRS sampling frequency must be finite and positive.")
    settings = _feature_config(config)
    channel_names, channel_types = _payload_channels(signals, data.shape[0])
    source_digest, input_classification, preprocessing_json = _preprocessing_provenance(signals)
    times, segment_ids, gap_indices = _support_clock(signals, data.shape[1], sfreq)
    if len(times) < 1:
        return pd.DataFrame()

    length_s = float(settings["window_length_s"])
    step_s = float(settings["window_step_s"])
    min_support = float(settings["min_valid_support_fraction"])
    dt = 1.0 / sfreq
    if length_s < dt - max(dt * _TIME_TOLERANCE_REL, 1e-12):
        raise FnirsFeatureContractError(
            "features.fnirs.window_length_s must span at least one nominal sample interval."
        )
    recording_start = float(times[0])
    recording_end_exclusive = float(times[-1] + dt)
    n_grid = max(0, int(math.ceil((recording_end_exclusive - recording_start) / step_s - 1e-12)))
    if n_grid == 0:
        return pd.DataFrame()

    channel_map = [
        {
            "column_key": quote(name, safe=""),
            "channel_name": name,
            "chromophore": chromophore,
            "unit": "M",
        }
        for name, chromophore in zip(channel_names, channel_types)
    ]
    mapping_json = json.dumps(channel_map, sort_keys=True, separators=(",", ":"))
    sample_interval_ends = np.empty_like(times)
    if len(times) > 1:
        sample_interval_ends[:-1] = np.minimum(times[:-1] + dt, times[1:])
    sample_interval_ends[-1] = times[-1] + dt

    records: list[dict[str, Any]] = []
    tolerance = max(dt * _TIME_TOLERANCE_REL, 1e-12)
    for grid_index in range(n_grid):
        t_start = recording_start + grid_index * step_s
        t_end = t_start + length_s
        interval_start = max(t_start, recording_start)
        interval_end = min(t_end, recording_end_exclusive)
        # Include the prior timestamp when its sample support interval reaches
        # into a window whose boundary falls between two sample instants.
        support_start = max(0, int(np.searchsorted(times, interval_start, side="right")) - 1)
        support_end = int(np.searchsorted(times, interval_end, side="left"))
        support_indices = np.arange(support_start, support_end, dtype=np.int64)
        sample_start = int(np.searchsorted(times, t_start, side="left"))
        sample_end = int(np.searchsorted(times, t_end, side="left"))
        sample_indices = np.arange(sample_start, sample_end, dtype=np.int64)
        if support_indices.size:
            clipped_starts = np.maximum(times[support_indices], t_start)
            clipped_ends = np.minimum(sample_interval_ends[support_indices], t_end)
            sample_support = np.maximum(0.0, clipped_ends - clipped_starts)
            observed_duration = float(sample_support.sum())
            observed_segment_ids = np.unique(segment_ids[support_indices][sample_support > 0])
            covered = sample_support > 0
            actual_start = float(np.min(clipped_starts[covered])) if np.any(covered) else pd.NA
            actual_end = float(np.max(clipped_ends[covered])) if np.any(covered) else pd.NA
        else:
            sample_support = np.empty(0, dtype=float)
            observed_duration = 0.0
            observed_segment_ids = np.empty(0, dtype=np.int64)
            actual_start = pd.NA
            actual_end = pd.NA
        nominal_duration = length_s
        clock_fraction = min(1.0, observed_duration / nominal_duration)
        partial = t_end > recording_end_exclusive + tolerance
        crosses_gap = clock_fraction < 1.0 - tolerance or len(observed_segment_ids) > 1
        if partial:
            window_reason = "partial_recording_window"
        elif crosses_gap:
            window_reason = "sample_clock_gap"
        else:
            window_reason = ""
        segment_id: int | None = (
            int(observed_segment_ids[0])
            if len(observed_segment_ids) == 1 and not crosses_gap
            else None
        )

        record: dict[str, Any] = {
            "epoch_id": int(grid_index),
            "t_start": float(t_start),
            "t_end": float(t_end),
            "fnirs_window_id": int(grid_index),
            "fnirs_window_start_s": float(t_start),
            "fnirs_window_end_s": float(t_end),
            "fnirs_actual_t_start_s": actual_start,
            "fnirs_actual_t_end_s": actual_end,
            "fnirs_window_length_s": float(length_s),
            "fnirs_window_step_s": float(step_s),
            "fnirs_window_grid_anchor_s": recording_start,
            "fnirs_clock_basis": "snirf_time_s",
            "fnirs_run_relative_t_start_s": float(t_start - recording_start),
            "fnirs_run_relative_t_end_s": float(t_end - recording_start),
            "fnirs_segment_id": segment_id if segment_id is not None else pd.NA,
            "fnirs_window_sample_count": int(np.count_nonzero(sample_support > 0)),
            "fnirs_window_timestamp_count": int(sample_indices.size),
            "fnirs_actual_covered_duration_s": observed_duration,
            "fnirs_clock_support_fraction": clock_fraction,
            "fnirs_window_status": "clock_supported" if not window_reason else "clock_rejected",
            "fnirs_rejection_reason": window_reason or pd.NA,
            "fnirs_min_valid_support_fraction": float(min_support),
            "fnirs_computation_status": "computed",
            "fnirs_measurement_validity": "not_assessed",
            "fnirs_qualification_status": "not_assessed",
            "fnirs_qc_status": "not_assessed",
            "fnirs_feature_contract": FEATURE_CONTRACT,
            "fnirs_source_sha256": source_digest,
            "fnirs_input_classification": input_classification,
            "fnirs_preprocessing_provenance_json": preprocessing_json,
            "fnirs_channel_map_json": mapping_json,
            "qc_ok_fnirs": pd.NA,
            "dataset_id": signals.get("dataset_id"),
        }
        for channel_index, (name, chromophore) in enumerate(zip(channel_names, channel_types)):
            finite_support = (
                np.isfinite(data[channel_index, support_indices]) if support_indices.size else np.empty(0, dtype=bool)
            )
            valid_duration = float(sample_support[finite_support].sum()) if sample_support.size else 0.0
            support_fraction = min(1.0, valid_duration / nominal_duration)
            if window_reason:
                reason = window_reason
            elif support_fraction < min_support - tolerance:
                reason = "insufficient_finite_sample_support"
            elif not np.any(finite_support & (sample_support > 0)):
                reason = "no_finite_sample_support_in_window"
            else:
                reason = ""
            supported = finite_support & (sample_support > 0)
            values = data[channel_index, support_indices][supported] if support_indices.size else np.empty(0, dtype=float)
            weights = sample_support[supported] if sample_support.size else np.empty(0, dtype=float)
            usable = not reason and values.size > 0 and float(weights.sum()) > 0
            if usable:
                weighted_mean = float(np.average(values, weights=weights))
                weighted_sd = float(np.sqrt(np.average((values - weighted_mean) ** 2, weights=weights)))
            else:
                weighted_mean = np.nan
                weighted_sd = np.nan
            finite_samples = np.isfinite(data[channel_index, sample_indices]) if sample_indices.size else np.empty(0, dtype=bool)
            record[_channel_column(name, "mean_M")] = weighted_mean
            record[_channel_column(name, "sd_M")] = weighted_sd
            record[_channel_column(name, "valid_sample_count")] = int(np.count_nonzero(supported))
            record[_channel_column(name, "valid_timestamp_count")] = int(finite_samples.sum()) if finite_samples.size else 0
            record[_channel_column(name, "valid_support_fraction")] = support_fraction
            record[_channel_column(name, "rejection_reason")] = reason or pd.NA
            # Keep chromophore explicit in a dedicated per-channel column.
            record[_channel_column(name, "chromophore")] = chromophore
        records.append(record)

    frame = pd.DataFrame.from_records(records)
    if "qc_ok_fnirs" in frame.columns:
        frame["qc_ok_fnirs"] = pd.array(frame["qc_ok_fnirs"], dtype="boolean")
    if "fnirs_segment_id" in frame.columns:
        frame["fnirs_segment_id"] = pd.array(frame["fnirs_segment_id"], dtype="Int64")
    frame.attrs["fnirs_feature_provenance"] = {
        "contract": FEATURE_CONTRACT,
        "window_length_s": length_s,
        "window_step_s": step_s,
        "minimum_valid_support_fraction": min_support,
        "window_anchor": "first_snirf_sample_time",
        "window_interval_convention": "half_open",
        "partial_window_policy": "record_rejected_missing_values",
        "gap_policy": "global_grid_no_reset_and_reject_crossing_windows",
        "filter_edge_exclusion_s": 0.0,
        "statistics": ["time_weighted_arithmetic_mean", "time_weighted_population_standard_deviation"],
        "sample_support_convention": "sample_constant_on_half_open_interval_until_min(next_time, sample_time_plus_1_over_sfreq)",
        "signal_units": "M",
        "qc_status": "not_assessed",
        "measurement_validity": "not_assessed",
        "qualification_status": "not_assessed",
        "sample_clock_gap_indices": gap_indices,
        "channel_map": channel_map,
    }
    return frame
