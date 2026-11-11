"""Audited ANPHY G2 loading for the frozen MNPS v3.1 M1 alpha protocol."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import importlib.metadata
import math
from pathlib import Path
import platform
from typing import Any, Mapping

import numpy as np
from scipy.signal import firwin, resample_poly

from .candidates import MNPS31WindowSupport
from .eeg_windows import _annotations, _intersects, _position_labels, _sha256_file
from .m1_phase import (
    M1_ALPHA_BAND_HZ,
    M1_ALPHA_CHANNEL_COUNT,
    M1_ALPHA_SAMPLE_RATE_HZ,
    M1_ALPHA_SOURCE_MARGIN_SAMPLES,
    M1_ALPHA_WINDOW_SAMPLES,
    M1_ALPHA_WINDOW_SECONDS,
    extract_m1_alpha_hilbert_phase,
)


M1_GRID_SCHEMA = "mndm.mnps_v3_1_m1_alpha_g2_overlay.v0"
M1_GRID_ID = "mnps_v3_1_anphy_epctl01_m1_g2_alpha_20260930_v2"
M1_NOMINAL_WINDOWS_SEC = tuple((float(start), float(start + 20)) for start in range(0, 300, 20))
M1_EXPECTED_SOURCE_PATH = Path("K:/ExternalReceivedDatasets/ANPHY/Subjects/EPCTL01/EPCTL01 - fixed.edf")
M1_EXPECTED_POSITIONS_PATH = Path("K:/ExternalReceivedDatasets/ANPHY/Co-registered average positions.pos")


@dataclass(frozen=True)
class AuditedM1AlphaWindow:
    """One frozen nominal interval, with its measurement and source audit."""

    nominal_window_index: int
    nominal_window_sec: tuple[float, float]
    status: str
    reason: str | None
    measurement: dict[str, Any]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class AuditedM1AlphaGrid:
    """All nominal windows in the existing ANPHY technical grid."""

    status: str
    reason: str | None
    windows: tuple[AuditedM1AlphaWindow, ...]
    provenance: dict[str, Any]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _blocked_measurement(reason: str, support: MNPS31WindowSupport | None = None) -> dict[str, Any]:
    return {
        "candidate_id": "M1",
        "estimator_id": "M1.sensor_phase_order_parameter_sd.v0",
        "value": None,
        "units": "1",
        "observation_space": "sensor_only_fixed_set",
        "computed": False,
        "valid": False,
        "qualified": False,
        "status": "unsupported",
        "reason": reason,
        "support": None if support is None else support.__dict__,
        "details": {},
        "coordinate_assignment": None,
        "claim_status": "no_biological_claim",
    }


def _runtime_versions() -> dict[str, str | None]:
    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "mne": version("mne"),
        "mndm": version("mndm"),
    }


def _blocked_window(index: int, bounds: tuple[float, float], reason: str) -> AuditedM1AlphaWindow:
    return AuditedM1AlphaWindow(
        nominal_window_index=index,
        nominal_window_sec=bounds,
        status="unsupported",
        reason=reason,
        measurement=_blocked_measurement(reason),
        provenance={
            "schema": "mndm.mnps_v3_1_m1_alpha_window_audit.v0",
            "audit_status": "unsupported",
            "reason": reason,
            "qualified": False,
            "coordinate_assignment": None,
            "claim_status": "no_biological_claim",
        },
    )


def _blocked_grid(reason: str, *, details: Mapping[str, Any] | None = None) -> AuditedM1AlphaGrid:
    return AuditedM1AlphaGrid(
        "completed",
        None,
        tuple(_blocked_window(i, bounds, reason) for i, bounds in enumerate(M1_NOMINAL_WINDOWS_SEC)),
        {
            "schema": "mndm.mnps_v3_1_m1_alpha_grid_audit.v0",
            "audit_status": "unsupported",
            "reason": reason,
            **dict(details or {}),
            "qualified": False,
            "coordinate_assignment": None,
        },
    )


def _overlap_annotations(records: list[dict[str, Any]], start: float, end: float):
    dependent = [record for record in records if _intersects(record, start, end)]
    bad = [
        record for record in dependent
        if str(record["description"]).strip().upper().startswith("BAD")
    ]
    boundaries = [
        record for record in dependent
        if any(token in str(record["description"]).upper() for token in ("ACQ_SKIP", "BOUNDARY", "DISCONTINU", "GAP"))
    ]
    return dependent, bad, boundaries


def _source_dependency(
    *,
    target_start: int,
    target_end_exclusive: int,
    source_sfreq: float,
    up: int,
    down: int,
    half_len: int,
) -> tuple[int, int]:
    first_target = target_start
    last_target = target_end_exclusive - 1
    first_source = math.ceil((first_target * down - half_len) / up)
    last_source = math.floor((last_target * down + half_len) / up)
    return int(first_source), int(last_source + 1)


def _load_overlay(overlay_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import yaml

    overlay = yaml.safe_load(overlay_path.read_text(encoding="utf-8"))
    if not isinstance(overlay, dict) or overlay.get("schema") != M1_GRID_SCHEMA:
        raise ValueError("m1_alpha_overlay_schema_unsupported")
    pilot = overlay.get("m1_alpha_g2")
    if not isinstance(pilot, dict):
        raise ValueError("m1_alpha_overlay_section_missing")
    if (
        pilot.get("dataset_id") != "ANPHY"
        or pilot.get("subject_id") != "EPCTL01"
        or Path(str(pilot.get("source_edf", ""))).resolve() != M1_EXPECTED_SOURCE_PATH.resolve()
        or Path(str(pilot.get("positions_file", ""))).resolve() != M1_EXPECTED_POSITIONS_PATH.resolve()
    ):
        raise ValueError("m1_alpha_source_identity_mismatch")
    if pilot.get("grid_id") != M1_GRID_ID:
        raise ValueError("m1_alpha_grid_id_mismatch")
    if pilot.get("decision_status") != "SELECT_FOR_G2_TECHNICAL_EVALUATION":
        raise ValueError("m1_alpha_decision_status_mismatch")
    if pilot.get("region_construction_class") != "sensor_only_fixed_set":
        raise ValueError("m1_alpha_observation_class_mismatch")
    if int(pilot.get("channel_count", -1)) != M1_ALPHA_CHANNEL_COUNT:
        raise ValueError("m1_alpha_fixed_channel_count_mismatch")
    if float(pilot.get("target_sfreq_hz", -1)) != M1_ALPHA_SAMPLE_RATE_HZ:
        raise ValueError("m1_alpha_sample_rate_mismatch")
    if tuple(float(value) for value in pilot.get("bandpass_hz", [])) != M1_ALPHA_BAND_HZ:
        raise ValueError("m1_alpha_band_mismatch")
    if float(pilot.get("window_seconds", -1)) != M1_ALPHA_WINDOW_SECONDS:
        raise ValueError("m1_alpha_window_mismatch")
    if pilot.get("rereference") != "source_reference_preserved" or pilot.get("csd") is not False:
        raise ValueError("m1_alpha_preprocessing_mismatch")
    if pilot.get("coordinates_used") is not False or pilot.get("condition_labels_used") is not False:
        raise ValueError("m1_alpha_forbidden_input_enabled")
    resampler = pilot.get("resampler")
    if not isinstance(resampler, Mapping) or (
        resampler.get("method") != "scipy.signal.resample_poly"
        or resampler.get("window") != "kaiser_beta_5.0"
        or resampler.get("tap_count") != "20_times_max_up_down_plus_1"
        or resampler.get("centering") != "symmetric_zero_phase_group_delay_removed"
        or resampler.get("edge_support") != "exact_finite_polyphase_dependency_in_source_samples"
    ):
        raise ValueError("m1_alpha_resampler_policy_mismatch")
    phase = pilot.get("phase_extraction")
    if not isinstance(phase, Mapping) or (
        phase.get("estimator_id") != "M1.sensor_alpha_8_13_hilbert_butterworth_sos_filtfilt.v2"
        or phase.get("filter") != "butterworth_bandpass_sos"
        or int(phase.get("order", -1)) != 4
        or phase.get("phase_mode") != "zero_phase_sosfiltfilt"
        or phase.get("padtype", "sentinel") is not None
        or int(phase.get("padlen", -1)) != 0
        or phase.get("analytic_signal") != "scipy.signal.hilbert"
        or phase.get("phase_output") != "principal_angle_minus_pi_to_pi"
        or phase.get("amplitude_threshold") is not None
        or phase.get("amplitude_validity_rule")
        != "reject_exactly_flat_input_channel_or_any_exact_zero_analytic_amplitude_sample"
        or float(phase.get("source_margin_cycles_at_low_edge", -1)) != 3.0
        or int(phase.get("source_margin_samples_each_side", -1)) != M1_ALPHA_SOURCE_MARGIN_SAMPLES
        or float(phase.get("source_margin_seconds_each_side", -1)) != M1_ALPHA_SOURCE_MARGIN_SAMPLES / M1_ALPHA_SAMPLE_RATE_HZ
        or phase.get("source_margin_basis") != "m1_g1_s1_three_cycle_synthetic_edge_support_probe"
    ):
        raise ValueError("m1_alpha_phase_policy_mismatch")
    support = pilot.get("source_support")
    if not isinstance(support, Mapping) or any(
        support.get(key) is not True
        for key in (
            "source_samples_must_cover_context_and_resampler_dependencies",
            "require_single_continuous_segment",
            "reject_missing_fixed_channel",
            "reject_source_bad_fixed_channel",
            "reject_gaps_or_boundaries_in_dependency",
            "reject_bad_annotations_in_dependency",
            "reject_nonfinite_inputs",
            "reject_exactly_flat_fixed_channel",
            "reject_zero_analytic_amplitude",
        )
    ) or support.get("fallback") != "none":
        raise ValueError("m1_alpha_support_policy_mismatch")
    windows = tuple(tuple(float(value) for value in bounds) for bounds in pilot.get("nominal_windows_sec", []))
    if windows != M1_NOMINAL_WINDOWS_SEC:
        raise ValueError("m1_alpha_nominal_grid_mismatch")
    repository_root = overlay_path.resolve().parents[2]
    decision_path = (repository_root / str(pilot.get("decision_record", ""))).resolve()
    if not decision_path.is_file() or _sha256_file(decision_path).lower() != str(
        pilot.get("decision_record_sha256", "")
    ).lower():
        raise ValueError("m1_alpha_decision_record_hash_mismatch")
    return overlay, pilot


def load_audited_edf_m1_alpha_grid(
    source_path: str | Path,
    positions_path: str | Path,
    overlay_path: str | Path,
) -> AuditedM1AlphaGrid:
    """Extract M1 phase on the existing 15-window ANPHY grid after audit.

    Each window uses 94 real resampled samples of source context on both sides,
    plus the exact finite polyphase-resampler dependency. Missing source labels,
    BAD/gap overlap, edge support, and non-finite data remain unsupported.
    """
    source, positions, config = (Path(value).resolve() for value in (source_path, positions_path, overlay_path))
    if not all(path.is_file() for path in (source, positions, config)):
        raise FileNotFoundError("M1 source EDF, positions file, and frozen overlay must exist")
    overlay, pilot = _load_overlay(config)
    if Path(str(pilot["source_edf"])).resolve() != source or Path(str(pilot["positions_file"])).resolve() != positions:
        raise ValueError("m1_alpha_overlay_input_paths_do_not_match")
    source_before = source.stat()
    source_hash = _sha256_file(source)
    position_hash = _sha256_file(positions)
    config_hash = _sha256_file(config)
    labels = _position_labels(positions, M1_ALPHA_CHANNEL_COUNT)
    labels_hash = hashlib.sha256("\n".join(label.casefold() for label in labels).encode("utf-8")).hexdigest()

    import mne
    from scipy import __version__ as scipy_version
    raw = mne.io.read_raw_edf(source, preload=False, verbose="ERROR")
    source_sfreq = float(raw.info["sfreq"])
    source_count = int(raw.n_times)
    source_duration = source_count / source_sfreq
    if not math.isfinite(source_sfreq) or source_sfreq <= M1_ALPHA_SAMPLE_RATE_HZ:
        return _blocked_grid("m1_source_sampling_rate_must_exceed_250_hz")
    if int(getattr(raw, "first_samp", 0)) != 0:
        return _blocked_grid("m1_nonzero_first_sample_unsupported")
    raw_lengths = getattr(raw, "_raw_lengths", None)
    if raw_lengths is not None and len(raw_lengths) != 1:
        reason = "m1_multi_segment_source_unsupported"
        return _blocked_grid(reason)
    try:
        annotations = _annotations(raw)
    except ValueError as exc:
        reason = f"source_annotations_unavailable:{exc}"
        return _blocked_grid(reason)

    normalized_names = [str(name).strip().casefold() for name in raw.ch_names]
    if len(normalized_names) != len(set(normalized_names)):
        return _blocked_grid("m1_duplicate_source_channel_names")
    raw_index = {name: index for index, name in enumerate(normalized_names)}
    if any(label.strip().casefold() not in raw_index for label in labels):
        missing = [label for label in labels if label.strip().casefold() not in raw_index]
        return _blocked_grid("fixed_channel_set_mismatch", details={"missing_labels": missing})
    eeg_picks = set(int(item) for item in mne.pick_types(raw.info, eeg=True, exclude=[]))
    picks = [raw_index[label.strip().casefold()] for label in labels]
    if any(index not in eeg_picks for index in picks):
        return _blocked_grid("m1_fixed_channel_set_contains_non_eeg_channel")
    bad_names = {str(name).strip().casefold() for name in raw.info.get("bads", [])}
    bad_fixed = [label for label in labels if label.strip().casefold() in bad_names]

    ratio = Fraction(M1_ALPHA_SAMPLE_RATE_HZ / source_sfreq).limit_denominator(100_000)
    up, down = int(ratio.numerator), int(ratio.denominator)
    if abs(source_sfreq * up / down - M1_ALPHA_SAMPLE_RATE_HZ) > 1e-10:
        return _blocked_grid("m1_sample_rate_ratio_not_representable")
    if up == down:
        half_len = 0
        resample_taps = np.asarray([1.0], dtype=np.float64)
    else:
        max_rate = max(up, down)
        half_len = 10 * max_rate
        resample_taps = firwin(2 * half_len + 1, 1.0 / max_rate, window=("kaiser", 5.0)).astype(np.float64)
    resample_hash = _sha256_bytes(np.asarray(resample_taps, dtype="<f8").tobytes())
    expected_output_samples = int(math.ceil(source_count * up / down))
    segment_id = f"edf:{source_hash[:16]}:segment0"
    windows: list[AuditedM1AlphaWindow] = []
    code_hashes = {
        "m1_phase": _sha256_file(Path(__file__).with_name("m1_phase.py")),
        "m1_windows": _sha256_file(Path(__file__)),
        "eeg_metrics": _sha256_file(Path(__file__).with_name("eeg_metrics.py")),
        "candidates": _sha256_file(Path(__file__).with_name("candidates.py")),
    }

    for index, bounds in enumerate(M1_NOMINAL_WINDOWS_SEC):
        start_sec, end_sec = bounds
        target_start = int(round(start_sec * M1_ALPHA_SAMPLE_RATE_HZ))
        target_end = int(round(end_sec * M1_ALPHA_SAMPLE_RATE_HZ))
        context_start = target_start - M1_ALPHA_SOURCE_MARGIN_SAMPLES
        context_end = target_end + M1_ALPHA_SOURCE_MARGIN_SAMPLES
        dependency_start, dependency_end = _source_dependency(
            target_start=context_start,
            target_end_exclusive=context_end,
            source_sfreq=source_sfreq,
            up=up,
            down=down,
            half_len=half_len,
        )
        support_start_sec = dependency_start / source_sfreq
        support_end_sec = dependency_end / source_sfreq
        annotations_in_support, bad_annotations, boundary_annotations = _overlap_annotations(
            annotations, support_start_sec, support_end_sec
        )
        source_bounds_valid = dependency_start >= 0 and dependency_end <= source_count
        sample_aligned = (
            math.isclose(start_sec * M1_ALPHA_SAMPLE_RATE_HZ, target_start, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(end_sec * M1_ALPHA_SAMPLE_RATE_HZ, target_end, rel_tol=0.0, abs_tol=1e-9)
        )
        support = MNPS31WindowSupport(
            window_start_sec=start_sec,
            window_end_sec=end_sec,
            sample_rate_hz=M1_ALPHA_SAMPLE_RATE_HZ,
            sample_count=M1_ALPHA_WINDOW_SAMPLES,
            segment_id=segment_id,
            source_is_continuous=not bool(boundary_annotations),
            timestamps_uniform=True,
            gap_free=not bool(boundary_annotations),
            bad_interval_free=not bool(bad_annotations) and not bool(bad_fixed),
            within_recording_bounds=source_bounds_valid and end_sec <= source_duration,
            effective_filter_support_valid=source_bounds_valid,
        )
        reason = None
        if bad_fixed:
            reason = "fixed_channel_set_contains_source_bad_channel"
        elif not sample_aligned:
            reason = "window_not_aligned_to_target_sample_grid"
        elif not source_bounds_valid or end_sec > source_duration:
            reason = "insufficient_filter_edge_support"
        elif boundary_annotations:
            reason = "segment_boundary_overlaps_effective_source_support"
        elif bad_annotations:
            reason = "bad_annotation_overlaps_effective_source_support"
        if reason is not None:
            measurement = _blocked_measurement(reason, support)
            audit_status = "unsupported"
            phase_details = None
        else:
            # Starting the crop at a multiple of ``down`` preserves the exact
            # global zero-phase polyphase output grid.
            source_crop_start = (dependency_start // down) * down
            source_crop_end = dependency_end
            local_target_start = context_start - (source_crop_start * up // down)
            needed_target_count = context_end - context_start
            source_data = np.asarray(
                raw.get_data(picks=picks, start=source_crop_start, stop=source_crop_end),
                dtype=np.float64,
            )
            if source_data.shape != (M1_ALPHA_CHANNEL_COUNT, source_crop_end - source_crop_start):
                reason = "source_sample_read_shape_mismatch"
                measurement = _blocked_measurement(reason, support)
                audit_status = "unsupported"
                phase_details = None
            elif not np.isfinite(source_data).all():
                reason = "non_finite_source_samples"
                measurement = _blocked_measurement(reason, support)
                audit_status = "unsupported"
                phase_details = None
            else:
                if up == down:
                    resampled = source_data
                else:
                    resampled = resample_poly(source_data, up, down, axis=-1, window=resample_taps, padtype="constant")
                context = resampled[:, local_target_start : local_target_start + needed_target_count]
                if context.shape != (M1_ALPHA_CHANNEL_COUNT, M1_ALPHA_WINDOW_SAMPLES + 2 * M1_ALPHA_SOURCE_MARGIN_SAMPLES):
                    reason = "insufficient_filter_edge_support"
                    measurement = _blocked_measurement(reason, support)
                    audit_status = "unsupported"
                    phase_details = None
                else:
                    phase_result = extract_m1_alpha_hilbert_phase(context, labels, labels)
                    reason = phase_result.reason
                    phase_details = phase_result.details if phase_result.reason is None else None
                    if reason is not None or phase_result.phase_radians is None:
                        measurement = _blocked_measurement(reason or "phase_estimator_failed", support)
                        measurement["details"] = phase_result.details
                        audit_status = "unsupported"
                    else:
                        from .eeg_metrics import compute_m1_sensor_phase_variability

                        measurement = compute_m1_sensor_phase_variability(
                            phase_result.phase_radians,
                            labels,
                            support,
                            phase_estimator_id=str(phase_result.details["phase_estimator_id"]),
                            band_id=str(phase_result.details["band_id"]),
                        ).to_dict()
                        measurement["observation_space"] = "sensor_only_fixed_set"
                        order_parameter = np.abs(np.mean(np.exp(1j * phase_result.phase_radians), axis=0))
                        measurement.setdefault("details", {})["validation_sidecars"] = {
                            "mean_kop_alpha_window": {
                                "value": float(np.mean(order_parameter)),
                                "estimator_id": "M1.validation.mean_alpha_sensor_kop.v1",
                                "role": "validation_sidecar_only_not_primary_estimator",
                                "units": "1",
                                "qualified": False,
                                "coordinate_assignment": None,
                            }
                        }
                        audit_status = "passed" if measurement.get("computed") is True else "unsupported"
                        if measurement.get("computed") is not True:
                            reason = str(measurement.get("reason") or "m1_metric_unsupported")

        provenance = {
            "schema": "mndm.mnps_v3_1_m1_alpha_window_audit.v0",
            "audit_status": audit_status,
            "qualified": False,
            "coordinate_assignment": None,
            "claim_status": "no_biological_claim",
            "nominal_window_index": index,
            "nominal_window_sec": [start_sec, end_sec],
            "source_dependency": {
                "source_sample_start": dependency_start,
                "source_sample_end_exclusive": dependency_end,
                "source_time_start_sec": support_start_sec,
                "source_time_end_sec": support_end_sec,
                "target_context_start_sample": context_start,
                "target_context_end_exclusive": context_end,
                "resampler_dependency_included": True,
            },
            "support_audit": {
                "source_is_continuous": support.source_is_continuous,
                "timestamps_uniform": support.timestamps_uniform,
                "gap_free": support.gap_free,
                "bad_interval_free": support.bad_interval_free,
                "within_recording_bounds": support.within_recording_bounds,
                "effective_filter_support_valid": support.effective_filter_support_valid,
                "dependent_annotations": annotations_in_support,
                "source_annotations": annotations,
            },
            "source": {
                "path": str(source),
                "size_bytes": int(source.stat().st_size),
                "sha256_before_read": source_hash,
                "mne_sample_rate_hz": source_sfreq,
                "mne_sample_count": source_count,
                "mne_duration_sec": source_duration,
                "segment_count": 1,
            },
            "channels": {
                "labels": list(labels),
                "count": len(labels),
                "order_sha256": labels_hash,
                "positions_path": str(positions),
                "positions_sha256": position_hash,
                "coordinates_used": False,
                "region_construction_class": "sensor_only_fixed_set",
                "source_reference": "preserved_as_recorded",
                "source_bad_labels": bad_fixed,
            },
            "preprocessing": {
                "pipeline_id": "m1_alpha_phase_preserving_polyphase_hilbert_support.v2",
                "resampler": {
                    "method": "scipy.signal.resample_poly_centered_symmetric_fir",
                    "up": up,
                    "down": down,
                    "fir_tap_count": int(resample_taps.size),
                    "fir_half_support_source_samples": half_len,
                    "fir_coefficients_sha256": resample_hash,
                    "window": "kaiser_beta_5.0",
                    "padding": "constant_only_outside_recording; dependent outputs rejected",
                },
                "phase_extraction": phase_details,
                "notch_hz": None,
                "rereference": "source_reference_preserved",
                "csd": False,
                "coordinates_used": False,
            },
            "configuration": {
                "overlay_path": str(config),
                "overlay_sha256": config_hash,
                "decision_record_path": str(pilot["decision_record"]),
                "decision_record_sha256": pilot["decision_record_sha256"],
            },
            "implementation_sha256": code_hashes,
            "runtime": {**_runtime_versions(), "scipy_reported_version": scipy_version},
            "reason": reason,
        }
        windows.append(AuditedM1AlphaWindow(index, bounds, audit_status, reason, measurement, provenance))

    source_after = source.stat()
    source_hash_after = _sha256_file(source)
    stable = (
        source_hash == source_hash_after
        and source_before.st_size == source_after.st_size
        and source_before.st_mtime_ns == source_after.st_mtime_ns
    )
    if not stable:
        windows = [
            _blocked_window(window.nominal_window_index, window.nominal_window_sec, "source_hash_unstable_during_read")
            for window in windows
        ]
    grid_provenance = {
        "schema": "mndm.mnps_v3_1_m1_alpha_grid_audit.v0",
        "grid_id": M1_GRID_ID,
        "audit_status": "completed" if stable else "unsupported",
        "source_hash_stable_after_read": stable,
        "source_sha256_before_read": source_hash,
        "source_sha256_after_read": source_hash_after,
        "positions_sha256": position_hash,
        "overlay_sha256": config_hash,
        "nominal_window_count": len(windows),
        "target_sfreq_hz": M1_ALPHA_SAMPLE_RATE_HZ,
        "band_hz": list(M1_ALPHA_BAND_HZ),
        "resampler_fir_half_support_source_samples": half_len,
        "resampler_output_count_expected": expected_output_samples,
        "runtime": _runtime_versions(),
        "qualified": False,
        "coordinate_assignment": None,
        "condition_labels_used": False,
        "condition_comparison": "not_performed",
    }
    return AuditedM1AlphaGrid("completed" if stable else "unsupported", None if stable else "source_hash_unstable_during_read", tuple(windows), grid_provenance)
