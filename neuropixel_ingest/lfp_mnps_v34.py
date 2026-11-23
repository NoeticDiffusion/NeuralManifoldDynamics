"""Opt-in contact-level LFP candidates for the MNPS 3.4 methods pilot.

This module emits auditable measurements only. It does not assign coordinates,
apply anchors, or qualify the measurements for a scientific contrast.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
import scipy
from scipy.signal import get_window, hilbert, sosfiltfilt, welch

from .anatomy import read_lfp_series_channels
from .lfp_state import (
    _running_speed_epoch_mean,
    annotate_lfp_features,
    read_000458_state_contract,
)


ESTIMATOR_SCHEMA = "mndm.experimental_lfp_mnps_3_4_candidate_spec.v1"
EXPORT_SCHEMA = "mndm.experimental_lfp_mnps_3_4_candidates.v1"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _float64_sha256(value: Any) -> str:
    array = np.asarray(value, dtype="<f8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def scan_explicit_timestamps(timestamps: np.ndarray, *, gap_factor: float = 1.5) -> dict[str, Any]:
    """Validate one continuous, monotone sample clock and summarize its rate."""
    values = np.asarray(timestamps, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("NWB LFP timestamps must be a one-dimensional vector with at least two samples.")
    if not np.isfinite(values).all():
        raise ValueError("NWB LFP timestamps contain non-finite values.")
    differences = np.diff(values)
    if np.any(differences <= 0):
        indices = np.flatnonzero(differences <= 0)[:10] + 1
        raise ValueError(f"NWB LFP timestamps are non-monotone at sample indices {indices.tolist()}.")
    median_interval = float(np.median(differences))
    if not np.isfinite(median_interval) or median_interval <= 0:
        raise ValueError("NWB LFP timestamps do not define a positive sample interval.")
    gap_indices = np.flatnonzero(differences > float(gap_factor) * median_interval) + 1
    if gap_indices.size:
        raise ValueError(
            f"NWB LFP timestamps contain {gap_indices.size} discontinuities; "
            f"first right-sample indices: {gap_indices[:10].tolist()}. "
            "Process segments separately before emitting this continuous-series export."
        )
    return {
        "sample_count": int(values.size),
        "sample_rate_hz": 1.0 / median_interval,
        "median_interval_sec": median_interval,
        "start_time_sec": float(values[0]),
        "stop_time_sec": float(values[-1]),
        "nonpositive_interval_count": 0,
        "gap_threshold_sec": float(gap_factor) * median_interval,
        "gap_count": 0,
        "segment_count": 1,
    }


def validate_frozen_estimator_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Load and verify the numerical arrays and sample counts in a frozen spec."""
    if spec.get("schema") != ESTIMATOR_SCHEMA:
        raise ValueError(f"Unsupported estimator schema: {spec.get('schema')!r}.")
    if spec.get("status") != "frozen_before_candidate_extraction":
        raise ValueError("Estimator settings must be frozen before candidate extraction.")
    if spec.get("candidate_values_examined") is not False:
        raise ValueError("The frozen estimator specification must record candidate_values_examined=false.")

    fs = float(spec["timebase"]["sample_rate_hz"])
    windows = spec["windows"]
    window_samples = int(windows["window_samples"])
    step_samples = int(windows["step_samples"])
    nperseg = int(spec["spectral_centroid"]["nperseg"])
    noverlap = int(spec["spectral_centroid"]["noverlap"])
    nfft = int(spec["spectral_centroid"]["nfft"])
    if window_samples != round(float(windows["length_sec"]) * fs):
        raise ValueError("Frozen window sample count does not match its duration and sample rate.")
    if step_samples != round(float(windows["step_sec"]) * fs):
        raise ValueError("Frozen step sample count does not match its duration and sample rate.")
    if nperseg > window_samples or noverlap >= nperseg or nfft < nperseg:
        raise ValueError("Invalid frozen Welch segment dimensions.")

    hann = get_window("hann", nperseg, fftbins=True).astype(np.float64)
    if _float64_sha256(hann) != spec["spectral_centroid"]["window_coefficients_sha256"]:
        raise ValueError("Periodic Hann coefficients do not match the frozen SHA-256.")
    frequencies = np.fft.rfftfreq(nfft, d=1.0 / fs)
    df = fs / nfft
    band = spec["frequency_band_hz"]
    band_mask = (frequencies >= float(band["low"]) - df / 2.0) & (
        frequencies <= float(band["high"]) + df / 2.0
    )
    if int(band_mask.sum()) != int(spec["spectral_centroid"]["selected_band_bin_count"]):
        raise ValueError("Frozen frequency-bin selection does not match the estimator specification.")

    sos = np.asarray(spec["envelope_cv"]["sos_coefficients"], dtype=np.float64)
    if _float64_sha256(sos) != spec["envelope_cv"]["sos_coefficients_sha256"]:
        raise ValueError("Butterworth SOS coefficients do not match the frozen SHA-256.")
    if sos.ndim != 2 or sos.shape[1] != 6:
        raise ValueError("Frozen SOS filter coefficients must have shape (sections, 6).")
    return {
        "sample_rate_hz": fs,
        "window_samples": window_samples,
        "step_samples": step_samples,
        "guard_samples": int(spec["envelope_cv"]["edge_guard_samples_each_side"]),
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "window": hann,
        "frequencies": frequencies,
        "band_mask": band_mask,
        "df": df,
        "sos": sos,
        "padlen": int(spec["envelope_cv"]["padlen_samples"]),
    }


def compute_spectral_window_metrics(
    signal_window: np.ndarray,
    *,
    fs: float,
    window: np.ndarray,
    nperseg: int,
    noverlap: int,
    nfft: int,
    frequencies: np.ndarray,
    band_mask: np.ndarray,
    df: float,
) -> dict[str, np.ndarray]:
    """Return per-contact centroid, band power, and missingness for one window."""
    data = np.asarray(signal_window, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2:
        raise ValueError("A spectral window must have shape (contacts, samples).")
    finite = np.isfinite(data).all(axis=1)
    safe_data = data.copy()
    safe_data[~finite] = 0.0
    _, psd = welch(
        safe_data,
        fs=float(fs),
        window=window,
        nperseg=int(nperseg),
        noverlap=int(noverlap),
        nfft=int(nfft),
        detrend="constant",
        return_onesided=True,
        scaling="density",
        average="mean",
        axis=-1,
    )
    band_psd = psd[:, band_mask]
    band_frequencies = frequencies[band_mask]
    power_sum = np.sum(band_psd, axis=1)
    bandpower = power_sum * float(df)
    centroid = np.full(data.shape[0], np.nan, dtype=np.float64)
    positive_power = finite & np.isfinite(power_sum) & (power_sum > 0.0)
    centroid[positive_power] = np.sum(
        band_psd[positive_power] * band_frequencies[None, :], axis=1
    ) / power_sum[positive_power]
    bandpower[~finite | ~np.isfinite(bandpower)] = np.nan
    centroid_reason = np.full(data.shape[0], "", dtype=object)
    bandpower_reason = np.full(data.shape[0], "", dtype=object)
    centroid_reason[~finite] = "nonfinite_input"
    bandpower_reason[~finite] = "nonfinite_input"
    centroid_reason[finite & ~positive_power] = "zero_band_power"
    return {
        "centroid_hz": centroid,
        "bandpower_v2": bandpower,
        "centroid_missing_reason": centroid_reason,
        "bandpower_missing_reason": bandpower_reason,
        "finite_input": finite,
    }


def compute_envelope_block(signal_block: np.ndarray, *, sos: np.ndarray, padlen: int) -> tuple[np.ndarray, np.ndarray]:
    """Filter and Hilbert-transform a continuous block with guard context."""
    data = np.asarray(signal_block, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("An envelope block must have shape (contacts, samples).")
    finite = np.isfinite(data).all(axis=1)
    safe_data = data.copy()
    safe_data[~np.isfinite(safe_data)] = 0.0
    filtered = sosfiltfilt(sos, safe_data, axis=-1, padtype="odd", padlen=int(padlen))
    envelope = np.abs(hilbert(filtered, axis=-1))
    return envelope, finite


def compute_envelope_cv_from_block(
    envelope: np.ndarray,
    *,
    local_window_start: int,
    window_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute population CV from the window interior of a guarded block."""
    values = np.asarray(envelope, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    stop = int(local_window_start) + int(window_samples)
    if local_window_start < 0 or stop > values.shape[-1]:
        raise ValueError("Envelope window lies outside its processed block.")
    central = values[:, int(local_window_start) : stop]
    means = np.mean(central, axis=1)
    cv = np.full(values.shape[0], np.nan, dtype=np.float64)
    valid = np.isfinite(central).all(axis=1) & np.isfinite(means) & (means > 0.0)
    cv[valid] = np.std(central[valid], axis=1, ddof=0) / means[valid]
    reasons = np.full(values.shape[0], "", dtype=object)
    reasons[~np.isfinite(central).all(axis=1)] = "nonfinite_input"
    reasons[np.isfinite(central).all(axis=1) & ~valid] = "zero_envelope_mean"
    return cv, reasons


def aggregate_contact_metrics(
    values: np.ndarray,
    *,
    contacts: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Median contact values within frozen groups and preserve support counts."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(contacts):
        raise ValueError("Contact metric matrix must have shape (windows, contacts).")
    index_by_channel = {int(contact["channel_index"]): idx for idx, contact in enumerate(contacts)}
    n_windows = matrix.shape[0]
    n_groups = len(groups)
    group_values = np.full((n_windows, n_groups), np.nan, dtype=np.float64)
    support_count = np.zeros((n_windows, n_groups), dtype=np.int32)
    support_fraction = np.zeros((n_windows, n_groups), dtype=np.float64)
    support_pass = np.zeros((n_windows, n_groups), dtype=bool)
    missing_reason = np.full((n_windows, n_groups), "insufficient_contact_support", dtype=object)
    for group_index, group in enumerate(groups):
        contact_positions = [index_by_channel[int(idx)] for idx in group["ordered_channel_indices"]]
        threshold = int(group["support_rule"]["required_computable_contacts"])
        denominator = int(group["support_rule"]["denominator"])
        group_contacts = matrix[:, contact_positions]
        finite = np.isfinite(group_contacts)
        counts = finite.sum(axis=1).astype(np.int32)
        fractions = counts / float(denominator)
        passed = counts >= threshold
        support_count[:, group_index] = counts
        support_fraction[:, group_index] = fractions
        support_pass[:, group_index] = passed
        for row in np.flatnonzero(passed):
            group_values[row, group_index] = float(np.median(group_contacts[row, finite[row]]))
            missing_reason[row, group_index] = ""
    return {
        "values": group_values,
        "support_count": support_count,
        "support_fraction": support_fraction,
        "support_pass": support_pass,
        "missing_reason": missing_reason,
    }


def _encode_text_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    flat = ["" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value) for value in array.ravel()]
    return np.asarray(flat, dtype=object).reshape(array.shape)


def _create_dataset(group: h5py.Group, name: str, values: Any) -> None:
    array = np.asarray(values)
    kwargs: dict[str, Any] = {}
    if array.ndim > 0 and array.size > 0:
        kwargs.update(compression="gzip", compression_opts=4, shuffle=True)
    if array.dtype.kind in {"U", "O"}:
        group.create_dataset(name, data=_encode_text_array(array), dtype=h5py.string_dtype("utf-8"), **kwargs)
    else:
        group.create_dataset(name, data=array, **kwargs)


def write_candidate_export_h5(
    output_path: str | Path,
    *,
    result: Mapping[str, Any],
    estimator_spec: Mapping[str, Any],
    contact_group_spec: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Path:
    """Write contact and group measurements with self-contained provenance."""
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite experimental LFP export: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(target, "w") as handle:
        handle.attrs["schema"] = EXPORT_SCHEMA
        handle.attrs["estimator_id"] = str(estimator_spec["spec_id"])
        handle.attrs["contact_set_sha256"] = str(contact_group_spec["contact_set_sha256"])
        handle.attrs["source_series_path"] = str(estimator_spec["input"]["series_path"])
        contracts = handle.create_group("contracts")
        _create_dataset(contracts, "estimator_spec_json", json.dumps(estimator_spec, ensure_ascii=False))
        _create_dataset(contracts, "contact_group_spec_json", json.dumps(contact_group_spec, ensure_ascii=False))
        _create_dataset(handle, "provenance_json", json.dumps(dict(provenance), ensure_ascii=False))

        contacts_group = handle.create_group("contacts")
        contacts = result["contacts"]
        for field in (
            "contact_id",
            "channel_index",
            "electrode_id",
            "probe_id",
            "location_raw",
            "depth_bin_index",
            "probe_vertical_position_um",
            "probe_horizontal_position_um",
            "is_data_valid",
            "reference_declaration",
        ):
            _create_dataset(contacts_group, field, [contact[field] for contact in contacts])
        _create_dataset(contacts_group, "group_id", [contact["group_id"] for contact in contacts])

        windows_group = handle.create_group("windows")
        for name, values in result["windows"].items():
            _create_dataset(windows_group, name, values)

        contact_metrics = handle.create_group("contact_metrics")
        for name, values in result["contact_metrics"].items():
            _create_dataset(contact_metrics, name, values)
        contact_missing = handle.create_group("contact_missing_reasons")
        for name, values in result["contact_missing_reasons"].items():
            _create_dataset(contact_missing, name, values)

        group_metrics = handle.create_group("group_metrics")
        _create_dataset(group_metrics, "group_id", [group["group_id"] for group in result["groups"]])
        for name, values in result["group_metrics"].items():
            _create_dataset(group_metrics, name, values)
        annotations = handle.create_group("window_annotations")
        for name in result["annotations"].columns:
            _create_dataset(annotations, name, result["annotations"][name].to_numpy())
        handle.flush()
    return target


def _verify_contact_group_spec(contact_spec: Mapping[str, Any], geometry: pd.DataFrame) -> None:
    contacts = list(contact_spec["contacts"])
    if _canonical_sha256(contacts) != contact_spec["contact_set_sha256"]:
        raise ValueError("Frozen contact-set hash does not match its contact records.")
    geometry_by_channel = geometry.set_index("channel_index", drop=False)
    for contact in contacts:
        channel_index = int(contact["channel_index"])
        if channel_index not in geometry_by_channel.index:
            raise ValueError(f"Frozen channel index {channel_index} is absent from the NWB series map.")
        row = geometry_by_channel.loc[channel_index]
        expected = (
            int(contact["electrode_id"]),
            str(contact["probe_id"]),
            str(contact["location_raw"]),
            float(contact["probe_vertical_position_um"]),
            float(contact["probe_horizontal_position_um"]),
            contact["is_data_valid"],
            contact["reference_declaration"],
        )
        actual_valid = None if pd.isna(row["is_data_valid"]) else bool(row["is_data_valid"])
        actual_reference = None if pd.isna(row["reference"]) else str(row["reference"])
        actual = (
            int(row["electrode_id"]),
            str(row["probe_id"]),
            str(row["location_raw"]),
            float(row["probe_vertical_position"]),
            float(row["probe_horizontal_position"]),
            actual_valid,
            actual_reference,
        )
        if actual != expected:
            raise ValueError(f"Frozen contact metadata differs from the NWB electrode table at channel {channel_index}.")
    for group in contact_spec["groups"]:
        members = [c for c in contacts if int(c["depth_bin_index"]) == int(group["dimensions"]["depth_bin_index"])]
        if _canonical_sha256(members) != group["contact_set_sha256"]:
            raise ValueError(f"Frozen group hash mismatch for {group['group_id']}.")
        if len(members) != int(group["intended_contact_count"]):
            raise ValueError(f"Frozen group denominator mismatch for {group['group_id']}.")


def _window_annotations(
    *,
    start_relative_sec: np.ndarray,
    stop_relative_sec: np.ndarray,
    source_path: Path,
    series_path: str,
) -> pd.DataFrame:
    features = pd.DataFrame({"t_start": start_relative_sec, "t_end": stop_relative_sec})
    contract = read_000458_state_contract(source_path, series_path)
    if contract.stim_events.empty:
        starts = start_relative_sec + contract.time_origin_sec
        ends = stop_relative_sec + contract.time_origin_sec
        midpoint = (starts + ends) / 2.0
        labels = np.full(len(midpoint), "", dtype=object)
        for row in contract.blocks.itertuples(index=False):
            labels[(midpoint >= row.start_sec) & (midpoint < row.stop_sec)] = str(row.state)
        features["lfp_epoch_midpoint_sec"] = midpoint
        features["lfp_behavioral_state"] = labels
        features["lfp_nearest_stim_distance_sec"] = np.nan
        features["lfp_stim_adjacent"] = False
        features["lfp_stim_contains_onset"] = False
        features["lfp_interstim_primary"] = labels != ""
        features["lfp_induction_excluded"] = False
    else:
        features = annotate_lfp_features(features, contract, stim_guard_sec=1.0, exclude_induction=True)
    features["lfp_running_speed_mean"] = _running_speed_epoch_mean(
        source_path,
        features["lfp_epoch_midpoint_sec"].to_numpy(dtype=np.float64),
    )
    return features


def run_lfp_mnps_v34_candidates(
    *,
    nwb_path: str | Path,
    estimator_spec_path: str | Path,
    contact_group_spec_path: str | Path,
    output_path: str | Path,
    progress: bool = True,
) -> dict[str, Any]:
    """Run one metadata-selected Probe F/MOs technical measurement export."""
    source_path = Path(nwb_path)
    estimator_path = Path(estimator_spec_path)
    contact_path = Path(contact_group_spec_path)
    target_path = Path(output_path)
    estimator_spec = json.loads(estimator_path.read_text(encoding="utf-8"))
    contact_spec = json.loads(contact_path.read_text(encoding="utf-8"))
    params = validate_frozen_estimator_spec(estimator_spec)
    if contact_spec.get("candidate_values_examined") is not False or contact_spec.get("metadata_only") is not True:
        raise ValueError("Contact group specification must be metadata-only and frozen before candidate extraction.")
    series_path = str(estimator_spec["input"]["series_path"])
    if contact_spec.get("series_path") != series_path:
        raise ValueError("Estimator and contact-group specifications refer to different NWB series.")
    geometry = read_lfp_series_channels(source_path, series_path)
    _verify_contact_group_spec(contact_spec, geometry)

    with h5py.File(source_path, "r") as handle:
        series = handle[series_path]
        data = series["data"]
        timestamp_data = series.get("timestamps")
        if timestamp_data is None:
            raise ValueError("Frozen LFP estimator requires explicit NWB timestamps.")
        timestamps = np.asarray(timestamp_data[:], dtype=np.float64)
        timebase = scan_explicit_timestamps(timestamps)
        source_fs = float(timebase["sample_rate_hz"])
        if not np.isclose(source_fs, params["sample_rate_hz"], rtol=1e-12, atol=1e-12):
            raise ValueError(
                f"Source sample rate {source_fs:.15g} Hz differs from frozen "
                f"{params['sample_rate_hz']:.15g} Hz. Freeze a new estimator version before extraction."
            )
        if data.shape[0] != int(estimator_spec["input"]["native_sample_count"]):
            raise ValueError("NWB sample count differs from the frozen estimator specification.")
        if data.shape[1] != int(estimator_spec["input"]["native_channel_count"]):
            raise ValueError("NWB channel count differs from the frozen estimator specification.")
        raw_unit = str(data.attrs.get("unit", ""))
        conversion = float(data.attrs.get("conversion", 1.0))
        offset = float(data.attrs.get("offset", 0.0))
        if raw_unit != str(estimator_spec["input"]["raw_unit"]):
            raise ValueError(f"NWB raw unit {raw_unit!r} differs from the frozen specification.")
        if not np.isclose(conversion, float(estimator_spec["input"]["conversion_to_volts"]), rtol=0, atol=0):
            raise ValueError("NWB conversion differs from the frozen estimator specification.")
        if not np.isclose(offset, float(estimator_spec["input"]["offset_volts"]), rtol=0, atol=0):
            raise ValueError("NWB offset differs from the frozen estimator specification.")

        groups = list(contact_spec["groups"])
        group_id_by_depth = {
            int(group["dimensions"]["depth_bin_index"]): str(group["group_id"]) for group in groups
        }
        contacts = [
            {
                **contact,
                "group_id": group_id_by_depth[int(contact["depth_bin_index"])],
            }
            for contact in contact_spec["contacts"]
        ]
        channel_indices = [int(contact["channel_index"]) for contact in contacts]
        sorted_channel_indices = sorted(channel_indices)
        source_n = int(data.shape[0])
        window_samples = params["window_samples"]
        step_samples = params["step_samples"]
        guard_samples = params["guard_samples"]
        starts = np.arange(0, source_n - window_samples + 1, step_samples, dtype=np.int64)
        if starts.size != int(estimator_spec["windows"]["candidate_window_count"]):
            raise ValueError("Computed complete-window count differs from the frozen specification.")
        stops = starts + window_samples
        window_start_times = timestamps[starts]
        window_stop_times = timestamps[stops]
        origin = float(timestamps[0])
        start_relative = window_start_times - origin
        stop_relative = window_stop_times - origin
        del timestamps

        n_windows = len(starts)
        n_contacts = len(contacts)
        contact_valid = np.asarray([contact["is_data_valid"] is True for contact in contacts], dtype=bool)
        contact_metrics = {
            "centroid_hz": np.full((n_windows, n_contacts), np.nan, dtype=np.float64),
            "bandpower_v2": np.full((n_windows, n_contacts), np.nan, dtype=np.float64),
            "envelope_cv": np.full((n_windows, n_contacts), np.nan, dtype=np.float64),
        }
        contact_missing = {
            "centroid_hz": np.full((n_windows, n_contacts), "", dtype=object),
            "bandpower_v2": np.full((n_windows, n_contacts), "", dtype=object),
            "envelope_cv": np.full((n_windows, n_contacts), "", dtype=object),
        }
        sorted_position = {channel: i for i, channel in enumerate(sorted_channel_indices)}
        contact_order = np.asarray([sorted_position[channel] for channel in channel_indices], dtype=np.int64)
        batch_samples = int(round(60.0 * source_fs))
        for batch_start in range(0, source_n, batch_samples):
            batch_stop = min(source_n, batch_start + batch_samples)
            window_rows = np.flatnonzero((starts >= batch_start) & (starts < batch_stop))
            if not window_rows.size:
                continue
            first_start = int(starts[window_rows[0]])
            last_start = int(starts[window_rows[-1]])
            read_start = max(0, first_start - guard_samples)
            read_stop = min(source_n, last_start + window_samples + guard_samples)
            raw_block = np.asarray(data[read_start:read_stop, sorted_channel_indices], dtype=np.float64)
            signal_block = raw_block.T
            signal_block *= conversion
            signal_block += offset
            signal_block = signal_block[contact_order, :]
            block_finite = np.isfinite(signal_block).all(axis=1)
            envelope_block, _ = compute_envelope_block(signal_block, sos=params["sos"], padlen=params["padlen"])

            for row in window_rows:
                local_start = int(starts[row]) - read_start
                local_stop = local_start + window_samples
                spectral = compute_spectral_window_metrics(
                    signal_block[:, local_start:local_stop],
                    fs=source_fs,
                    window=params["window"],
                    nperseg=params["nperseg"],
                    noverlap=params["noverlap"],
                    nfft=params["nfft"],
                    frequencies=params["frequencies"],
                    band_mask=params["band_mask"],
                    df=params["df"],
                )
                contact_metrics["centroid_hz"][row] = spectral["centroid_hz"]
                contact_metrics["bandpower_v2"][row] = spectral["bandpower_v2"]
                contact_missing["centroid_hz"][row] = spectral["centroid_missing_reason"]
                contact_missing["bandpower_v2"][row] = spectral["bandpower_missing_reason"]

                edge_ok = int(starts[row]) >= guard_samples and int(stops[row]) + guard_samples <= source_n
                if not edge_ok:
                    contact_missing["envelope_cv"][row, contact_valid] = "edge_guard_unavailable"
                else:
                    cv, cv_reason = compute_envelope_cv_from_block(
                        envelope_block,
                        local_window_start=local_start,
                        window_samples=window_samples,
                    )
                    contact_metrics["envelope_cv"][row] = cv
                    contact_missing["envelope_cv"][row] = cv_reason
                    contact_missing["envelope_cv"][row, ~block_finite] = "nonfinite_input"

                for metric_name in contact_metrics:
                    contact_metrics[metric_name][row, ~contact_valid] = np.nan
                    contact_missing[metric_name][row, ~contact_valid] = "metadata_invalid_channel"
            if progress:
                print(
                    f"LFP candidate windows {int(window_rows[-1]) + 1}/{n_windows} "
                    f"({read_start}:{read_stop} source samples)",
                    flush=True,
                )

    group_metrics: dict[str, np.ndarray] = {}
    for metric_name, matrix in contact_metrics.items():
        aggregate = aggregate_contact_metrics(matrix, contacts=contacts, groups=groups)
        group_metrics[f"{metric_name}_median"] = aggregate["values"]
        group_metrics[f"{metric_name}_support_count"] = aggregate["support_count"]
        group_metrics[f"{metric_name}_support_fraction"] = aggregate["support_fraction"]
        group_metrics[f"{metric_name}_support_pass"] = aggregate["support_pass"]
        group_metrics[f"{metric_name}_missing_reason"] = aggregate["missing_reason"]

    annotations = _window_annotations(
        start_relative_sec=start_relative,
        stop_relative_sec=stop_relative,
        source_path=source_path,
        series_path=series_path,
    )
    window_record = {
        "window_index": np.arange(n_windows, dtype=np.int64),
        "source_sample_start": starts,
        "source_sample_stop_exclusive": stops,
        "source_time_start_sec": window_start_times,
        "source_time_stop_sec": window_stop_times,
        "relative_start_sec": start_relative,
        "relative_stop_sec": stop_relative,
        "segment_id": np.full(n_windows, "segment-0000", dtype=object),
    }
    data_result = {
        "contacts": contacts,
        "groups": groups,
        "windows": window_record,
        "contact_metrics": contact_metrics,
        "contact_missing_reasons": contact_missing,
        "group_metrics": group_metrics,
        "annotations": annotations,
    }
    provenance = {
        "source_file": source_path.name,
        "source_file_path": str(source_path),
        "series_path": series_path,
        "raw_unit": raw_unit,
        "conversion_to_volts": conversion,
        "offset_volts": offset,
        "timebase": timebase,
        "candidate_window_count": n_windows,
        "contact_count": n_contacts,
        "group_count": len(groups),
        "candidate_values_examined_before_freeze": False,
        "condition_comparisons_performed": False,
        "spike_integration_performed": False,
        "coordinate_or_anchor_output": False,
        "reference_declaration": estimator_spec["reference"]["declared_description"],
        "physical_reference_realization": estimator_spec["reference"]["physical_realization"],
        "contact_set_sha256": contact_spec["contact_set_sha256"],
        "estimator_spec_id": estimator_spec["spec_id"],
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "h5py": h5py.__version__},
    }
    written = write_candidate_export_h5(
        target_path,
        result=data_result,
        estimator_spec=estimator_spec,
        contact_group_spec=contact_spec,
        provenance=provenance,
    )
    support_summary: dict[str, Any] = {}
    for metric_name in contact_metrics:
        support_pass = group_metrics[f"{metric_name}_support_pass"]
        support_summary[metric_name] = {
            str(groups[j]["group_id"]): int(support_pass[:, j].sum()) for j in range(len(groups))
        }
    reason_summary = {
        metric_name: {
            str(reason): int(np.count_nonzero(values == reason))
            for reason in np.unique(values)
            if str(reason)
        }
        for metric_name, values in contact_missing.items()
    }
    with h5py.File(written, "r") as check:
        if check.attrs["schema"] != EXPORT_SCHEMA:
            raise RuntimeError("HDF5 round-trip schema verification failed.")
        if check["windows/window_index"].shape[0] != n_windows:
            raise RuntimeError("HDF5 round-trip window-count verification failed.")
        if check.attrs["contact_set_sha256"] != contact_spec["contact_set_sha256"]:
            raise RuntimeError("HDF5 round-trip contact-hash verification failed.")
    return {
        "output_path": str(written.resolve()),
        "schema": EXPORT_SCHEMA,
        "window_count": n_windows,
        "contact_count": n_contacts,
        "group_count": len(groups),
        "sample_rate_hz": source_fs,
        "timebase_segment_count": timebase["segment_count"],
        "timebase_gap_count": timebase["gap_count"],
        "group_supported_window_counts": support_summary,
        "contact_missing_reason_counts": reason_summary,
        "hdf5_roundtrip_verified": True,
    }
