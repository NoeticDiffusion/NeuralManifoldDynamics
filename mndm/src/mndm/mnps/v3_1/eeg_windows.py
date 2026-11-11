"""Audited continuous-EEG window loading for v3.1 research sidecars.

This opt-in EDF path reads one source file, checks its MNE sample clock and
annotations, applies recorded finite-support transforms, and emits the actual
source-sample dependency bounds for one fixed window. It is intentionally
separate from the default MNDM feature pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import math
from pathlib import Path
import platform
from typing import Any, Mapping

import numpy as np

from .candidates import (
    E1_BANDPASS_HZ,
    E1_CANDIDATE_ID,
    E1_CHANNEL_COUNT,
    E1_DELAY_SAMPLES,
    E1_METHOD_VERSION,
    E1_ORDER,
    E1_SAMPLE_RATE_HZ,
    E1_WINDOW_SECONDS,
    MNPS31WindowSupport,
)


AUDIT_SCHEMA = "mndm.mnps_v3_1.eeg_window_audit.v0"
GRID_SCHEMA = "mndm.mnps_v3_1_eeg_grid.v0"
E1_G2_GRID_ID = "mnps_v3_1_anphy_epctl01_e1_g2_20260929"
E1_G2_CROP_SEC = (0.0, 300.001)
E1_G2_NOMINAL_WINDOWS_SEC = tuple(
    (float(start), float(start + E1_WINDOW_SECONDS)) for start in range(0, 300, int(E1_WINDOW_SECONDS))
)
_IDENTITY_KEYS = ("dataset_id", "dataset_version", "subject_id", "session_id", "task_id", "run_id")


@dataclass(frozen=True)
class AuditedEEGWindow:
    """One requested window and its independently derived support audit."""

    status: str
    reason: str | None
    signals: np.ndarray | None
    channel_names: tuple[str, ...]
    support: MNPS31WindowSupport | None
    provenance: dict[str, Any]


@dataclass(frozen=True)
class AuditedEEGGrid:
    """Every nominal window from one immutable grid manifest."""

    status: str
    reason: str | None
    windows: tuple[AuditedEEGWindow, ...]
    provenance: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdefABCDEF" for char in value)


def _position_labels(path: Path, expected_count: int) -> tuple[str, ...]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("position_file_empty")
    try:
        declared_count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError("position_file_count_invalid") from exc
    labels = tuple(
        parts[1]
        for line in lines[1:]
        if len(parts := line.split()) >= 5
    )
    keys = [label.strip().casefold() for label in labels]
    if declared_count != expected_count or len(labels) != expected_count:
        raise ValueError("position_file_channel_count_mismatch")
    if any(not label for label in keys) or len(set(keys)) != expected_count:
        raise ValueError("position_file_labels_not_unique")
    return labels


def _annotations(raw: Any) -> list[dict[str, Any]]:
    annotations = getattr(raw, "annotations", None)
    if annotations is None:
        raise ValueError("source_annotations_unavailable")
    first_time = float(getattr(raw, "first_time", 0.0) or 0.0)
    records: list[dict[str, Any]] = []
    arrays = (annotations.onset, annotations.duration, annotations.description)
    if len({len(values) for values in arrays}) != 1:
        raise ValueError("source_annotation_arrays_mismatch")
    for onset, duration, description in zip(*arrays):
        start_sec = float(onset) - first_time
        duration_sec = float(duration)
        if not math.isfinite(start_sec) or not math.isfinite(duration_sec) or duration_sec < 0:
            raise ValueError("source_annotation_bounds_invalid")
        records.append(
            {
                "start_sec": start_sec,
                "duration_sec": duration_sec,
                "end_sec": start_sec + duration_sec,
                "description": str(description),
            }
        )
    return records


def _intersects(record: Mapping[str, Any], start_sec: float, end_sec: float) -> bool:
    """Return whether an annotation overlaps the half-open interval ``[start_sec, end_sec)``."""
    start = float(record["start_sec"])
    end = float(record["end_sec"])
    if end == start:
        return start_sec <= start < end_sec
    return start < end_sec and end > start_sec


def _bandpass_is_frozen(bandpass_hz: tuple[float, float]) -> bool:
    return (
        len(bandpass_hz) == 2
        and math.isclose(float(bandpass_hz[0]), E1_BANDPASS_HZ[0], rel_tol=0.0, abs_tol=0.0)
        and math.isclose(float(bandpass_hz[1]), E1_BANDPASS_HZ[1], rel_tol=0.0, abs_tol=0.0)
    )


def _bounds_match(actual: tuple[float, float], expected: tuple[float, float]) -> bool:
    return math.isclose(actual[0], expected[0], rel_tol=0.0, abs_tol=1e-9) and math.isclose(
        actual[1], expected[1], rel_tol=0.0, abs_tol=1e-9
    )


def _rejected_window(
    reason: str,
    *,
    channel_names: tuple[str, ...] = (),
    provenance: dict[str, Any] | None = None,
) -> AuditedEEGWindow:
    payload = {
        "schema": AUDIT_SCHEMA,
        "audit_status": "unsupported",
        "qualified": False,
        "claim_status": "no_biological_claim",
        "reason": reason,
    }
    if provenance:
        payload.update(provenance)
        payload["reason"] = reason
        payload["qualified"] = False
        payload["audit_status"] = "unsupported"
    return AuditedEEGWindow(
        status="unsupported",
        reason=reason,
        signals=None,
        channel_names=channel_names,
        support=None,
        provenance=payload,
    )


def _dataset_identity(pilot: Mapping[str, Any]) -> dict[str, str]:
    identity = {key: pilot[key] for key in _IDENTITY_KEYS}
    if any(not isinstance(value, str) or not value.strip() for value in identity.values()):
        raise ValueError("pilot_dataset_identity_fields_must_be_explicit")
    return identity


def _preprocessing_policy_error(pilot: Mapping[str, Any]) -> str | None:
    if (
        pilot.get("resampler_id") != "mndm.causal_polyphase_kaiser_beta5.v0"
        or pilot.get("filter_phase") != "minimum"
        or pilot.get("filter_pad") != "constant"
        or pilot.get("notch_hz") is not None
        or pilot.get("rereference") != "average_over_fixed_channel_set"
        or pilot.get("csd") is not False
        or pilot.get("coordinates_read") is not False
        or pilot.get("estimator_id") != E1_METHOD_VERSION
        or int(pilot.get("embedding_order", -1)) != E1_ORDER
        or int(pilot.get("delay_samples", -1)) != E1_DELAY_SAMPLES
        or pilot.get("normalize") is not True
        or pilot.get("tie_policy") != "stable_mergesort"
        or pilot.get("candidate_id") != E1_CANDIDATE_ID
    ):
        return "pilot_overlay_preprocessing_policy_unsupported"
    return None


def _parsed_bandpass(value: Any) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError("e1_bandpass_must_be_1_45_hz")
    return tuple(float(item) for item in value)


def _e1_loader_contract_reason(*, channel_count: int, bandpass_hz: tuple[float, ...]) -> str | None:
    if channel_count != E1_CHANNEL_COUNT:
        return "e1_fixed_channel_set_must_contain_83_labels"
    if len(bandpass_hz) != 2 or not _bandpass_is_frozen((bandpass_hz[0], bandpass_hz[1])):
        return "e1_bandpass_must_be_1_45_hz"
    return None


def _frozen_g2_grid_reason(
    *,
    crop_sec: tuple[float, float],
    nominal_windows_sec: tuple[tuple[float, float], ...],
) -> str | None:
    if not _bounds_match(crop_sec, E1_G2_CROP_SEC):
        return "e1_g2_crop_mismatch"
    if len(nominal_windows_sec) != len(E1_G2_NOMINAL_WINDOWS_SEC):
        return "e1_g2_nominal_grid_mismatch"
    if any(
        not _bounds_match(actual, expected)
        for actual, expected in zip(nominal_windows_sec, E1_G2_NOMINAL_WINDOWS_SEC, strict=True)
    ):
        return "e1_g2_nominal_grid_mismatch"
    return None


def _code_hashes() -> dict[str, str]:
    return {
        "eeg_window_loader": _sha256_file(Path(__file__)),
        "eeg_metrics": _sha256_file(Path(__file__).with_name("eeg_metrics.py")),
        "candidate_protocols": _sha256_file(Path(__file__).with_name("candidates.py")),
        "preprocess_support_helpers": _sha256_file(Path(__file__).parents[2] / "preprocess.py"),
        "eeg_permutation_entropy": _sha256_file(Path(__file__).parents[2] / "features" / "eeg.py"),
    }


def _runtime_versions() -> dict[str, str | None]:
    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    from ...__about__ import __version__

    return {
        "python": platform.python_version(),
        "mndm": version("mndm") or __version__,
        "mne": version("mne"),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
    }


def _prepare_from_raw(
    raw: Any,
    *,
    source_path: Path,
    source_sha256: str,
    config_path: Path,
    config_sha256: str,
    positions_path: Path,
    positions_sha256: str,
    channel_names: tuple[str, ...],
    window_start_sec: float,
    window_end_sec: float,
    target_sfreq_hz: float,
    bandpass_hz: tuple[float, float],
    source_hash_stable: bool,
    code_hashes: Mapping[str, str],
) -> AuditedEEGWindow:
    """Apply the pilot's explicitly supported transforms to an MNE Raw."""
    def failed(reason: str, provenance: dict[str, Any] | None = None) -> AuditedEEGWindow:
        return AuditedEEGWindow(
            status="unsupported",
            reason=reason,
            signals=None,
            channel_names=channel_names,
            support=None,
            provenance=provenance or {},
        )

    if not source_hash_stable or not _valid_sha256(source_sha256):
        return failed("source_hash_missing_or_unstable")
    if not _valid_sha256(config_sha256) or not _valid_sha256(positions_sha256):
        return failed("configuration_or_channel_set_hash_invalid")
    if any(not _valid_sha256(value) for value in code_hashes.values()):
        return failed("implementation_hash_invalid")
    if not (source_path.is_file() and source_path.suffix.lower() == ".edf"):
        return failed("single_edf_source_required")
    if not (math.isfinite(window_start_sec) and math.isfinite(window_end_sec) and window_end_sec > window_start_sec):
        return failed("window_bounds_invalid")
    if not (math.isfinite(target_sfreq_hz) and target_sfreq_hz > 0):
        return failed("target_sampling_rate_invalid")
    if len(bandpass_hz) != 2 or not all(math.isfinite(value) for value in bandpass_hz):
        return failed("bandpass_invalid")
    if not _bandpass_is_frozen(bandpass_hz):
        return failed("e1_bandpass_must_be_1_45_hz")

    try:
        import mne
        from ...preprocess import _causal_polyphase_resample, _minimum_phase_eeg_filter_spec

        source_sfreq = float(raw.info["sfreq"])
        source_n_times = int(raw.n_times)
        source_first_samp = int(getattr(raw, "first_samp", 0))
        source_duration_sec = source_n_times / source_sfreq
        raw_lengths = getattr(raw, "_raw_lengths", None)
        segment_count = 1 if raw_lengths is None else len(raw_lengths)
        annotation_records = _annotations(raw)
    except Exception as exc:
        return failed(f"source_clock_or_annotations_unavailable:{type(exc).__name__}")

    if not math.isfinite(source_sfreq) or source_sfreq <= target_sfreq_hz:
        return failed("source_sampling_rate_must_exceed_target")
    if source_first_samp != 0:
        return failed("nonzero_source_first_sample_unsupported")
    if segment_count != 1:
        return failed("multi_segment_raw_unsupported")
    if window_end_sec > source_duration_sec:
        return failed("window_exceeds_recording_bounds")
    start_pos = window_start_sec * target_sfreq_hz
    end_pos = window_end_sec * target_sfreq_hz
    if not math.isclose(start_pos, round(start_pos), rel_tol=0.0, abs_tol=1e-8) or not math.isclose(
        end_pos, round(end_pos), rel_tol=0.0, abs_tol=1e-8
    ):
        return failed("window_not_aligned_to_target_sample_grid")
    window_start_sample = int(round(start_pos))
    window_end_sample = int(round(end_pos))
    sample_count = window_end_sample - window_start_sample
    if sample_count < 2:
        return failed("window_sample_count_insufficient")

    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    normalized_names = [str(name).strip().casefold() for name in raw.ch_names]
    if len(normalized_names) != len(set(normalized_names)):
        return failed("duplicate_source_channel_names")
    type_by_name = {
        normalized_names[int(index)]: str(raw.get_channel_types(picks=[int(index)])[0])
        for index in eeg_picks
    }
    raw_index = {name: index for index, name in enumerate(normalized_names)}
    keys = [name.strip().casefold() for name in channel_names]
    if any(key not in raw_index for key in keys):
        return failed("fixed_channel_set_mismatch")
    if any(type_by_name.get(key) != "eeg" for key in keys):
        return failed("fixed_channel_set_contains_non_eeg_channel")
    bad_channels = {str(name).strip().casefold() for name in raw.info.get("bads", [])}
    if any(key in bad_channels for key in keys):
        return failed("fixed_channel_set_contains_source_bad_channel")

    source_start_sample = 0
    source_stop_sample = min(source_n_times, int(math.ceil(window_end_sec * source_sfreq)))
    selected_picks = np.asarray([raw_index[key] for key in keys], dtype=np.int64)
    try:
        source_data = np.asarray(
            raw.get_data(picks=selected_picks, start=source_start_sample, stop=source_stop_sample),
            dtype=np.float64,
        )
        if source_data.shape != (len(channel_names), source_stop_sample - source_start_sample):
            return failed("source_sample_read_shape_mismatch")
        if not np.isfinite(source_data).all():
            return failed("non_finite_source_samples")
        info = mne.pick_info(raw.info, selected_picks.tolist(), copy=True)
        working_raw = mne.io.RawArray(source_data, info, first_samp=0, verbose="ERROR")
        resampled_raw, resampler = _causal_polyphase_resample(working_raw, target_sfreq_hz)
        filter_spec = _minimum_phase_eeg_filter_spec(
            resampled_raw,
            range(len(channel_names)),
            bandpass_hz,
        )
        resampled_raw.filter(
            l_freq=bandpass_hz[0],
            h_freq=bandpass_hz[1],
            picks=np.arange(len(channel_names)),
            method="fir",
            phase="minimum",
            fir_design="firwin",
            fir_window="hamming",
            filter_length="auto",
            pad="constant",
            verbose="ERROR",
        )
        processed = np.asarray(resampled_raw.get_data(), dtype=np.float64)
        # The overlay declares average rereferencing over this fixed EEG set.
        processed -= np.mean(processed, axis=0, keepdims=True)
    except Exception as exc:
        return failed(f"finite_support_preprocessing_failed:{type(exc).__name__}")

    resampler_taps = int(resampler["tap_count"])
    resampler_up = int(resampler["up"])
    resampler_down = int(resampler["down"])
    filter_taps = int(filter_spec["tap_count"])
    first_filtered_sample = window_start_sample - (filter_taps - 1)
    last_filtered_sample = window_end_sample - 1
    dependency_start_sample = math.ceil(
        (first_filtered_sample * resampler_down - (resampler_taps - 1)) / resampler_up
    )
    dependency_end_sample = math.floor(last_filtered_sample * resampler_down / resampler_up)
    dependency_start_sec = dependency_start_sample / source_sfreq
    dependency_end_sec = (dependency_end_sample + 1) / source_sfreq
    if (
        first_filtered_sample < 0
        or last_filtered_sample >= processed.shape[1]
        or dependency_start_sample < source_start_sample
        or dependency_end_sample >= source_n_times
        or dependency_end_sample >= source_stop_sample
    ):
        return failed(
            "insufficient_effective_filter_edge_support",
            {
                "effective_source_sample_start": int(dependency_start_sample),
                "effective_source_sample_end_exclusive": int(dependency_end_sample + 1),
                "resampler": resampler,
                "eeg_bandpass": filter_spec,
            },
        )

    dependent_annotations = [
        record
        for record in annotation_records
        if _intersects(record, dependency_start_sec, dependency_end_sec)
    ]
    bad_annotations = [
        record for record in dependent_annotations
        if str(record["description"]).strip().upper().startswith("BAD")
    ]
    boundary_annotations = [
        record for record in dependent_annotations
        if (
            "ACQ_SKIP" in str(record["description"]).upper()
            or "BOUNDARY" in str(record["description"]).upper()
            or "DISCONTINU" in str(record["description"]).upper()
        )
    ]

    support = MNPS31WindowSupport(
        window_start_sec=float(window_start_sec),
        window_end_sec=float(window_end_sec),
        sample_rate_hz=float(target_sfreq_hz),
        sample_count=int(sample_count),
        segment_id=f"edf:{source_sha256[:16]}:segment0",
        source_is_continuous=not boundary_annotations,
        timestamps_uniform=True,
        gap_free=not boundary_annotations,
        bad_interval_free=not bad_annotations,
        within_recording_bounds=True,
        effective_filter_support_valid=True,
    )
    source_dependency = {
        "source_sample_start": int(dependency_start_sample),
        "source_sample_end_exclusive": int(dependency_end_sample + 1),
        "source_time_start_sec": float(dependency_start_sec),
        "source_time_end_sec": float(dependency_end_sec),
        "filter_output_start_sample": int(first_filtered_sample),
        "filter_output_end_sample": int(last_filtered_sample),
        "resampler": resampler,
        "eeg_bandpass": filter_spec,
    }
    if bad_annotations:
        return AuditedEEGWindow(
            status="unsupported",
            reason="bad_annotation_overlaps_effective_source_support",
            signals=None,
            channel_names=channel_names,
            support=support,
            provenance={
                "qualified": False,
                "dependent_annotations": dependent_annotations,
                "source_annotations": annotation_records,
                "source_dependency": source_dependency,
            },
        )
    if boundary_annotations:
        return AuditedEEGWindow(
            status="unsupported",
            reason="segment_boundary_overlaps_effective_source_support",
            signals=None,
            channel_names=channel_names,
            support=support,
            provenance={
                "qualified": False,
                "dependent_annotations": dependent_annotations,
                "source_annotations": annotation_records,
                "source_dependency": source_dependency,
            },
        )

    window = np.array(processed[:, window_start_sample:window_end_sample], dtype=np.float64, copy=True)
    if window.shape != (len(channel_names), sample_count) or not np.isfinite(window).all():
        return failed("processed_window_shape_or_finite_check_failed")

    try:
        from . import candidates, eeg_metrics
        from ... import preprocess as preprocess_module
        from ...features import eeg as eeg_feature_module

        code_paths = {
            "eeg_window_loader": Path(__file__),
            "eeg_metrics": Path(eeg_metrics.__file__),
            "candidate_protocols": Path(candidates.__file__),
            "preprocess_support_helpers": Path(preprocess_module.__file__),
            "eeg_permutation_entropy": Path(eeg_feature_module.__file__),
        }
        actual_code_hashes = {name: _sha256_file(path) for name, path in code_paths.items()}
    except Exception:
        actual_code_hashes = dict(code_hashes)

    provenance: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "audit_status": "passed",
        "claim_status": "no_biological_claim",
        "qualified": False,
        "source": {
            "path": str(source_path.resolve()),
            "size_bytes": int(source_path.stat().st_size),
            "sha256_before_read": source_sha256.lower(),
            "sha256_stable_after_read": bool(source_hash_stable),
            "format": "single_edf_record",
            "mne_first_samp": source_first_samp,
            "mne_sample_rate_hz": source_sfreq,
            "mne_sample_count": source_n_times,
            "mne_duration_sec": source_duration_sec,
            "mne_segment_count": int(segment_count),
        },
        "window": {
            "requested_start_sec": float(window_start_sec),
            "requested_end_sec": float(window_end_sec),
            "output_clock_start_sec": float(window_start_sec),
            "output_clock_end_sec": float(window_end_sec),
            "output_sample_start": int(window_start_sample),
            "output_sample_end_exclusive": int(window_end_sample),
            "sample_count": int(sample_count),
            "sample_rate_hz": float(target_sfreq_hz),
            "time_semantics": "nominal output clock; uncompensated causal filter delay and exact source dependency bounds are recorded",
        },
        "support_audit": {
            "source_is_continuous": bool(support.source_is_continuous),
            "timestamps_uniform": True,
            "timestamp_basis": "single EDF sample grid defined by the EDF header and MNE Raw clock",
            "gap_free": bool(support.gap_free),
            "bad_interval_free": bool(support.bad_interval_free),
            "within_recording_bounds": True,
            "effective_filter_support_valid": True,
            "dependent_annotations": dependent_annotations,
            "source_annotations": annotation_records,
            "source_annotation_audit": "complete_mne_raw_annotations",
        },
        "source_dependency": source_dependency,
        "channels": {
            "labels": list(channel_names),
            "count": int(len(channel_names)),
            "order_sha256": hashlib.sha256(
                "\n".join(name.strip().casefold() for name in channel_names).encode("utf-8")
            ).hexdigest(),
            "positions_path": str(positions_path.resolve()),
            "positions_sha256": positions_sha256.lower(),
            "coordinates_used": False,
            "source_amplitude_unit": "V (MNE SI-scaled EDF data)",
        },
        "preprocessing": {
            "pipeline_id": "anphy_eeg_causal_fir_resample_bandpass_average_reference.v0",
            "target_sfreq_hz": float(target_sfreq_hz),
            "bandpass_hz": [float(bandpass_hz[0]), float(bandpass_hz[1])],
            "notch_hz": None,
            "reference": "average_over_fixed_channel_set",
            "csd": False,
            "phase_delay_compensation": "not_applied",
        },
        "configuration": {
            "overlay_path": str(config_path.resolve()),
            "overlay_sha256": config_sha256.lower(),
        },
        "implementation_sha256": actual_code_hashes,
        "runtime": _runtime_versions(),
    }
    return AuditedEEGWindow(
        status="ready",
        reason=None,
        signals=window,
        channel_names=channel_names,
        support=support,
        provenance=provenance,
    )


def load_audited_edf_eeg_window(
    source_path: str | Path,
    positions_path: str | Path,
    config_path: str | Path,
) -> AuditedEEGWindow:
    """Load one EDF window and require auditable source/config/code identity.

    The path reads a single EDF source, uses explicit finite causal resampling
    and a finite minimum-phase EEG FIR, and returns unsupported when effective
    source support touches a recording edge, BAD interval, or gap boundary.
    It never writes or changes an MNDM feature/HDF5 product.
    """
    source = Path(source_path).resolve()
    positions = Path(positions_path).resolve()
    config = Path(config_path).resolve()
    if not source.is_file() or not positions.is_file() or not config.is_file():
        raise FileNotFoundError("source EDF, positions file, and pilot overlay must exist")
    try:
        import yaml

        overlay = yaml.safe_load(config.read_text(encoding="utf-8"))
        pilot = overlay["eeg_pilot"]
        if overlay.get("schema") != "mndm.mnps_v3_1_eeg_pilot.v0":
            raise ValueError("pilot_overlay_schema_unsupported")
        configured_source = Path(str(pilot["source_edf"])).resolve()
        configured_positions = Path(str(pilot["positions_file"])).resolve()
        window_start_sec, window_end_sec = (float(value) for value in pilot["window_sec"])
        target_sfreq_hz = float(pilot["target_sfreq_hz"])
        bandpass_hz = _parsed_bandpass(pilot["eeg_bandpass_hz"])
        expected_channel_count = int(pilot["channel_count"])
        dataset_identity = _dataset_identity(pilot)
        if configured_source != source or configured_positions != positions:
            raise ValueError("pilot_overlay_input_paths_do_not_match")
        policy_error = _preprocessing_policy_error(pilot)
        if policy_error is not None or not math.isclose(
            window_end_sec - window_start_sec,
            E1_WINDOW_SECONDS,
            rel_tol=0.0,
            abs_tol=1e-9,
        ) or not math.isclose(target_sfreq_hz, E1_SAMPLE_RATE_HZ, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(policy_error or "pilot_overlay_preprocessing_policy_unsupported")
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid_audited_eeg_pilot_overlay:{exc}") from exc
    contract_reason = _e1_loader_contract_reason(
        channel_count=expected_channel_count,
        bandpass_hz=bandpass_hz,
    )
    if contract_reason is not None:
        return _rejected_window(contract_reason)
    source_stat_before = source.stat()
    source_hash_before = _sha256_file(source)
    positions_hash = _sha256_file(positions)
    config_hash = _sha256_file(config)
    try:
        labels = _position_labels(positions, expected_channel_count)
    except ValueError as exc:
        if str(exc) in {
            "position_file_channel_count_mismatch",
            "position_file_labels_not_unique",
            "position_file_empty",
            "position_file_count_invalid",
        }:
            return _rejected_window("e1_fixed_channel_set_must_contain_83_labels")
        raise
    if len(labels) != E1_CHANNEL_COUNT:
        return _rejected_window("e1_fixed_channel_set_must_contain_83_labels")
    try:
        import mne
    except ImportError as exc:
        raise RuntimeError("MNE is required to read an audited EDF EEG window") from exc
    raw = mne.io.read_raw_edf(source, preload=False, verbose="ERROR")
    result = _prepare_from_raw(
        raw,
        source_path=source,
        source_sha256=source_hash_before,
        config_path=config,
        config_sha256=config_hash,
        positions_path=positions,
        positions_sha256=positions_hash,
        channel_names=labels,
        window_start_sec=float(window_start_sec),
        window_end_sec=float(window_end_sec),
        target_sfreq_hz=float(target_sfreq_hz),
        bandpass_hz=(bandpass_hz[0], bandpass_hz[1]),
        source_hash_stable=True,
        code_hashes=_code_hashes(),
    )
    source_hash_after = _sha256_file(source)
    positions_hash_after = _sha256_file(positions)
    config_hash_after = _sha256_file(config)
    source_stat_after = source.stat()
    hash_stable = (
        source_hash_before == source_hash_after
        and source_stat_before.st_size == source_stat_after.st_size
        and source_stat_before.st_mtime_ns == source_stat_after.st_mtime_ns
        and positions_hash == positions_hash_after
        and config_hash == config_hash_after
    )
    result.provenance.setdefault("schema", AUDIT_SCHEMA)
    result.provenance.setdefault("audit_status", "passed" if result.status == "ready" else "unsupported")
    result.provenance.setdefault("reason", result.reason)
    result.provenance["source"] = {
        "path": str(source),
        "size_bytes": int(source_stat_after.st_size),
        "sha256_before_read": source_hash_before.lower(),
        "sha256_after_read": source_hash_after.lower(),
        "sha256_stable_after_read": bool(hash_stable),
        "format": "single_edf_record",
        **result.provenance.get("source", {}),
    }
    result.provenance["configuration"] = {
        "overlay_path": str(config),
        "overlay_sha256": config_hash.lower(),
        "overlay_sha256_after_read": config_hash_after.lower(),
        "overlay_hash_stable": config_hash == config_hash_after,
        **result.provenance.get("configuration", {}),
    }
    result.provenance["channels"] = {
        "positions_path": str(positions),
        "positions_sha256": positions_hash.lower(),
        "positions_sha256_after_read": positions_hash_after.lower(),
        "positions_hash_stable": positions_hash == positions_hash_after,
        "labels": list(labels),
        **result.provenance.get("channels", {}),
    }
    result.provenance.setdefault("runtime", _runtime_versions())
    result.provenance["recording_identity"] = {
        **dataset_identity,
        "metadata_status": str(pilot.get("metadata_status", "unresolved")),
        "metadata_note": str(pilot.get("metadata_note", "")),
    }
    if not hash_stable:
        result.provenance["audit_status"] = "unsupported"
        result.provenance["reason"] = "source_or_configuration_changed_during_read"
        return AuditedEEGWindow(
            status="unsupported",
            reason="source_or_configuration_changed_during_read",
            signals=None,
            channel_names=labels,
            support=None,
            provenance=result.provenance,
        )
    return result


def _pair(value: Any) -> tuple[float, float]:
    start, end = value
    return (float(start), float(end))


def _ensure_source_dependency(window: AuditedEEGWindow, source_sfreq: float) -> None:
    provenance = window.provenance
    if "source_dependency" in provenance or "effective_source_sample_start" not in provenance:
        return
    start = int(provenance["effective_source_sample_start"])
    end_exclusive = int(provenance["effective_source_sample_end_exclusive"])
    provenance["source_dependency"] = {
        "source_sample_start": start,
        "source_sample_end_exclusive": end_exclusive,
        "source_time_start_sec": start / source_sfreq,
        "source_time_end_sec": end_exclusive / source_sfreq,
        "resampler": provenance.get("resampler"),
        "eeg_bandpass": provenance.get("eeg_bandpass"),
    }


def _outside_crop(window: AuditedEEGWindow, crop_sec: tuple[float, float]) -> bool:
    dependency = window.provenance.get("source_dependency")
    if not isinstance(dependency, Mapping):
        return False
    start = float(dependency["source_time_start_sec"])
    end = float(dependency["source_time_end_sec"])
    return start < crop_sec[0] - 1e-9 or end > crop_sec[1] + 1e-9


def load_audited_edf_eeg_grid(
    source_path: str | Path,
    positions_path: str | Path,
    config_path: str | Path,
) -> AuditedEEGGrid:
    """Load every nominal window named by one immutable E1 G2 grid manifest.

    The manifest hash is the shared preprocessing and grid identity. Nominal
    windows are taken from that file and checked against the frozen
    ``[0, 20), ..., [280, 300)`` contract. A listed window is never shifted
    or replaced when its own support fails.
    """
    source = Path(source_path).resolve()
    positions = Path(positions_path).resolve()
    config = Path(config_path).resolve()
    if not source.is_file() or not positions.is_file() or not config.is_file():
        raise FileNotFoundError("source EDF, positions file, and grid manifest must exist")
    try:
        import yaml

        overlay = yaml.safe_load(config.read_text(encoding="utf-8"))
        if overlay.get("schema") != GRID_SCHEMA:
            raise ValueError("pilot_overlay_schema_unsupported")
        pilot = overlay["eeg_grid"]
        configured_source = Path(str(pilot["source_edf"])).resolve()
        configured_positions = Path(str(pilot["positions_file"])).resolve()
        target_sfreq_hz = float(pilot["target_sfreq_hz"])
        bandpass_hz = _parsed_bandpass(pilot["eeg_bandpass_hz"])
        expected_channel_count = int(pilot["channel_count"])
        crop_sec = _pair(pilot["crop_sec"])
        nominal_windows = tuple(_pair(item) for item in pilot["nominal_windows_sec"])
        dataset_identity = _dataset_identity(pilot)
        if configured_source != source or configured_positions != positions:
            raise ValueError("pilot_overlay_input_paths_do_not_match")
        if _preprocessing_policy_error(pilot) is not None or not math.isclose(
            target_sfreq_hz, E1_SAMPLE_RATE_HZ, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("pilot_overlay_preprocessing_policy_unsupported")
        if "window_sec" in pilot:
            raise ValueError("grid_manifest_must_not_carry_a_single_window")
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid_audited_eeg_grid_manifest:{exc}") from exc

    def blocked(reason: str) -> AuditedEEGGrid:
        return AuditedEEGGrid(
            status="unsupported",
            reason=reason,
            windows=(),
            provenance={
                "schema": GRID_SCHEMA,
                "audit_status": "unsupported",
                "qualified": False,
                "claim_status": "no_biological_claim",
                "reason": reason,
                "grid_id": E1_G2_GRID_ID,
                "configuration": {"overlay_path": str(config)},
            },
        )

    if str(pilot.get("grid_id", "")) != E1_G2_GRID_ID:
        return blocked("e1_g2_grid_id_mismatch")
    if str(pilot.get("decision_status", "")) != "SELECT_FOR_G2_TECHNICAL_EVALUATION":
        return blocked("e1_g2_decision_status_mismatch")
    contract_reason = _e1_loader_contract_reason(
        channel_count=expected_channel_count,
        bandpass_hz=bandpass_hz,
    )
    if contract_reason is not None:
        return blocked(contract_reason)
    grid_reason = _frozen_g2_grid_reason(crop_sec=crop_sec, nominal_windows_sec=nominal_windows)
    if grid_reason is not None:
        return blocked(grid_reason)

    source_stat_before = source.stat()
    source_hash_before = _sha256_file(source)
    positions_hash = _sha256_file(positions)
    config_hash = _sha256_file(config)
    try:
        labels = _position_labels(positions, expected_channel_count)
    except ValueError as exc:
        if str(exc) in {
            "position_file_channel_count_mismatch",
            "position_file_labels_not_unique",
            "position_file_empty",
            "position_file_count_invalid",
        }:
            return blocked("e1_fixed_channel_set_must_contain_83_labels")
        raise
    if len(labels) != E1_CHANNEL_COUNT:
        return blocked("e1_fixed_channel_set_must_contain_83_labels")
    try:
        import mne
    except ImportError as exc:
        raise RuntimeError("MNE is required to read an audited EDF EEG window") from exc
    raw = mne.io.read_raw_edf(source, preload=False, verbose="ERROR")
    source_sfreq = float(raw.info["sfreq"])
    code_hashes = _code_hashes()
    grid_identity = {
        "schema": GRID_SCHEMA,
        "grid_id": E1_G2_GRID_ID,
        "decision_status": "SELECT_FOR_G2_TECHNICAL_EVALUATION",
        "crop_sec": [E1_G2_CROP_SEC[0], E1_G2_CROP_SEC[1]],
        "nominal_window_count": len(E1_G2_NOMINAL_WINDOWS_SEC),
        "overlay_path": str(config),
        "overlay_sha256": config_hash.lower(),
        "output_root": str(pilot.get("output_root", "")),
        "qualified": False,
    }
    windows: list[AuditedEEGWindow] = []
    for index, (window_start_sec, window_end_sec) in enumerate(nominal_windows):
        result = _prepare_from_raw(
            raw,
            source_path=source,
            source_sha256=source_hash_before,
            config_path=config,
            config_sha256=config_hash,
            positions_path=positions,
            positions_sha256=positions_hash,
            channel_names=labels,
            window_start_sec=window_start_sec,
            window_end_sec=window_end_sec,
            target_sfreq_hz=target_sfreq_hz,
            bandpass_hz=(bandpass_hz[0], bandpass_hz[1]),
            source_hash_stable=True,
            code_hashes=code_hashes,
        )
        _ensure_source_dependency(result, source_sfreq)
        result.provenance["grid"] = {
            **grid_identity,
            "nominal_window_index": index,
            "nominal_window_sec": [window_start_sec, window_end_sec],
        }
        result.provenance.setdefault("window", {})
        result.provenance["window"]["requested_start_sec"] = window_start_sec
        result.provenance["window"]["requested_end_sec"] = window_end_sec
        result.provenance["qualified"] = False
        if result.status == "ready" and _outside_crop(result, crop_sec):
            result = _rejected_window(
                "dependency_outside_declared_crop",
                channel_names=result.channel_names,
                provenance=result.provenance,
            )
        windows.append(result)

    source_hash_after = _sha256_file(source)
    positions_hash_after = _sha256_file(positions)
    config_hash_after = _sha256_file(config)
    source_stat_after = source.stat()
    hash_stable = (
        source_hash_before == source_hash_after
        and source_stat_before.st_size == source_stat_after.st_size
        and source_stat_before.st_mtime_ns == source_stat_after.st_mtime_ns
        and positions_hash == positions_hash_after
        and config_hash == config_hash_after
    )
    finalized: list[AuditedEEGWindow] = []
    for result in windows:
        result.provenance["source"] = {
            "path": str(source),
            "size_bytes": int(source_stat_after.st_size),
            "sha256_before_read": source_hash_before.lower(),
            "sha256_after_read": source_hash_after.lower(),
            "sha256_stable_after_read": bool(hash_stable),
            "format": "single_edf_record",
            **result.provenance.get("source", {}),
        }
        result.provenance["configuration"] = {
            "overlay_path": str(config),
            "overlay_sha256": config_hash.lower(),
            "overlay_sha256_after_read": config_hash_after.lower(),
            "overlay_hash_stable": config_hash == config_hash_after,
            **result.provenance.get("configuration", {}),
        }
        result.provenance["channels"] = {
            "positions_path": str(positions),
            "positions_sha256": positions_hash.lower(),
            "positions_sha256_after_read": positions_hash_after.lower(),
            "positions_hash_stable": positions_hash == positions_hash_after,
            "labels": list(labels),
            "count": len(labels),
            **result.provenance.get("channels", {}),
        }
        result.provenance.setdefault("runtime", _runtime_versions())
        result.provenance["recording_identity"] = {
            **dataset_identity,
            "metadata_status": str(pilot.get("metadata_status", "unresolved")),
            "metadata_note": str(pilot.get("metadata_note", "")),
        }
        result.provenance["implementation_sha256"] = {
            **code_hashes,
            **result.provenance.get("implementation_sha256", {}),
        }
        if not hash_stable:
            finalized.append(
                _rejected_window(
                    "source_or_configuration_changed_during_read",
                    channel_names=labels,
                    provenance=result.provenance,
                )
            )
        else:
            finalized.append(result)
    return AuditedEEGGrid(
        status="completed" if hash_stable else "unsupported",
        reason=None if hash_stable else "source_or_configuration_changed_during_read",
        windows=tuple(finalized),
        provenance={
            "schema": GRID_SCHEMA,
            "audit_status": "completed" if hash_stable else "unsupported",
            "qualified": False,
            "claim_status": "no_biological_claim",
            "grid": grid_identity,
            "recording_identity": {
                **dataset_identity,
                "metadata_status": str(pilot.get("metadata_status", "unresolved")),
                "metadata_note": str(pilot.get("metadata_note", "")),
            },
            "source_sha256": source_hash_before.lower(),
            "source_sha256_stable_after_read": bool(hash_stable),
            "positions_sha256": positions_hash.lower(),
            "overlay_sha256": config_hash.lower(),
            "implementation_sha256": code_hashes,
            "runtime": _runtime_versions(),
            "nominal_windows_sec": [list(bounds) for bounds in nominal_windows],
        },
    )
