"""Opt-in MNE source reader for experimental MNPS surface/window grids.

The manifest pins a source recording, preprocessing, sensor surfaces, and
nominal windows. This path filters every EEG channel once without rereferencing;
the E3 surface runner then selects and references each surface independently.
Unsupported windows remain explicit rows. The default MNDM pipeline and frozen
83-channel pilot loaders are not changed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import numpy as np

from .candidates import MNPS31WindowSupport
from .eeg_windows import _annotations, _intersects
from .realizations import (
    CORE19_CHANNELS,
    MNPS31SensorSurface,
    MNPS31WindowSlice,
    compute_e3_surface_window_grid,
)


SURFACE_GRID_MANIFEST_SCHEMA = "mndm.mnps_v3_1_surface_grid_manifest.v0"
SURFACE_GRID_RUN_SCHEMA = "mndm.mnps_v3_1_surface_grid_run.v0"
_SUPPORTED_SOURCE_FORMATS = {"edf", "brainvision", "fif"}


@dataclass(frozen=True)
class MNPS31SurfaceGridRun:
    """Serializable E3 results and recording-level source provenance."""

    rows: tuple[dict[str, object], ...]
    provenance: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        computed = sum(bool(row["measurement"]["computed"]) for row in self.rows)  # type: ignore[index]
        return {
            "schema": SURFACE_GRID_RUN_SCHEMA,
            "status": "completed" if computed else "completed_no_supported_values",
            "row_count": len(self.rows),
            "computed_count": computed,
            "unsupported_count": len(self.rows) - computed,
            "qualified": False,
            "claim_status": "no_biological_claim",
            "provenance": self.provenance,
            "rows": list(self.rows),
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        "numpy": np.__version__,
        "scipy": version("scipy"),
    }


def _resolve_manifest_path(value: object, manifest_dir: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("manifest_path_must_be_nonempty_string")
    path = Path(value)
    return (path if path.is_absolute() else manifest_dir / path).resolve()


def _brainvision_header_files(header_path: Path) -> tuple[Path, Path]:
    """Resolve the data and marker files actually referenced by a VHDR."""
    text: str | None = None
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = header_path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError("brainvision_header_encoding_unsupported")
    references: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line or line.lstrip().startswith((";", "#")):
            continue
        key, value = line.split("=", 1)
        key = key.strip().casefold()
        if key in {"datafile", "markerfile"}:
            references[key] = value.strip().strip('"')
    if not references.get("datafile") or not references.get("markerfile"):
        raise ValueError("brainvision_header_data_or_marker_reference_missing")
    data_path = (header_path.parent / references["datafile"]).resolve()
    marker_path = (header_path.parent / references["markerfile"]).resolve()
    return data_path, marker_path


def _verify_mne_source_files(raw: Any, declared_paths: Sequence[Path]) -> None:
    """Ensure MNE did not follow an unpinned source or split-file reference."""
    filenames = getattr(raw, "filenames", None)
    if not filenames:
        raise ValueError("surface_grid_mne_source_file_inventory_unavailable")
    declared = {str(path.resolve()).casefold() for path in declared_paths}
    opened = {
        str(Path(filename).resolve()).casefold()
        for filename in filenames
        if filename is not None and str(filename).strip()
    }
    if not opened:
        raise ValueError("surface_grid_mne_source_file_inventory_unavailable")
    if not opened.issubset(declared):
        raise ValueError("surface_grid_mne_source_file_not_hashed")


def _implementation_hashes() -> dict[str, str]:
    """Hash every local implementation file that affects this result."""
    from . import candidates, eeg_metrics, eeg_windows, realizations
    from ... import preprocess

    code_paths = {
        "source_grid": Path(__file__),
        "candidates": Path(candidates.__file__),
        "eeg_metrics": Path(eeg_metrics.__file__),
        "eeg_windows": Path(eeg_windows.__file__),
        "realizations": Path(realizations.__file__),
        "preprocess": Path(preprocess.__file__),
    }
    try:
        return {key: _sha256_file(value) for key, value in code_paths.items()}
    except (OSError, TypeError) as exc:
        raise RuntimeError("surface_grid_implementation_hash_unavailable") from exc


def _read_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read an MNPS surface-grid manifest") from exc
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"surface_grid_manifest_unreadable:{type(exc).__name__}") from exc
    if not isinstance(raw, dict) or raw.get("schema") != SURFACE_GRID_MANIFEST_SCHEMA:
        raise ValueError("surface_grid_manifest_schema_unsupported")
    if not isinstance(raw.get("manifest_id"), str) or not raw["manifest_id"].strip():
        raise ValueError("surface_grid_manifest_id_missing")

    source = raw.get("source")
    preprocessing = raw.get("preprocessing")
    identity = raw.get("recording_identity")
    windows = raw.get("windows")
    surfaces = raw.get("surfaces")
    if not isinstance(source, dict) or not isinstance(preprocessing, dict):
        raise ValueError("surface_grid_source_or_preprocessing_missing")
    if not isinstance(identity, dict) or not identity:
        raise ValueError("surface_grid_recording_identity_missing")
    if any(not isinstance(value, str) or not value.strip() for value in identity.values()):
        raise ValueError("surface_grid_recording_identity_invalid")
    required_identity = {"dataset_id", "dataset_version", "subject_id", "session_id", "task_id", "run_id"}
    if not required_identity.issubset(identity):
        raise ValueError("surface_grid_recording_identity_incomplete")
    if not isinstance(windows, list) or not windows:
        raise ValueError("surface_grid_windows_must_be_nonempty_list")
    if not isinstance(surfaces, list) or not surfaces:
        raise ValueError("surface_grid_surfaces_must_be_nonempty_list")

    source_format = str(source.get("format", "")).casefold()
    if source_format not in _SUPPORTED_SOURCE_FORMATS:
        raise ValueError("surface_grid_source_format_unsupported")
    source_path = _resolve_manifest_path(source.get("path"), path.parent)
    associated_values = source.get("associated_files", [])
    if not isinstance(associated_values, list) or any(not isinstance(item, str) for item in associated_values):
        raise ValueError("surface_grid_associated_files_invalid")
    associated_paths = tuple(_resolve_manifest_path(item, path.parent) for item in associated_values)
    if source_path in associated_paths or len(set(associated_paths)) != len(associated_paths):
        raise ValueError("surface_grid_source_files_must_be_unique")
    expected_suffix = {"edf": ".edf", "brainvision": ".vhdr", "fif": ".fif"}[source_format]
    if source_format == "fif":
        suffix_matches = source_path.name.casefold().endswith((".fif", ".fif.gz"))
    else:
        suffix_matches = source_path.suffix.casefold() == expected_suffix
    if not suffix_matches:
        raise ValueError("surface_grid_source_path_format_mismatch")
    if source_format == "brainvision":
        suffixes = {path.suffix.casefold() for path in associated_paths}
        if ".eeg" not in suffixes or ".vmrk" not in suffixes:
            raise ValueError("brainvision_manifest_must_hash_eeg_and_marker_files")
        referenced_paths = _brainvision_header_files(source_path)
        declared = {str(path).casefold() for path in associated_paths}
        if any(str(referenced).casefold() not in declared for referenced in referenced_paths):
            raise ValueError("brainvision_manifest_sidecars_do_not_match_header")

    target_sfreq = float(preprocessing.get("target_sfreq_hz", float("nan")))
    bandpass = preprocessing.get("bandpass_hz")
    pipeline_id = preprocessing.get("pipeline_id")
    if not math.isfinite(target_sfreq) or target_sfreq <= 0:
        raise ValueError("surface_grid_target_sampling_rate_invalid")
    if (
        not isinstance(bandpass, (list, tuple))
        or len(bandpass) != 2
        or not all(math.isfinite(float(value)) for value in bandpass)
        or not 0 < float(bandpass[0]) < float(bandpass[1]) < target_sfreq / 2
    ):
        raise ValueError("surface_grid_bandpass_invalid")
    if not isinstance(pipeline_id, str) or not pipeline_id.strip():
        raise ValueError("surface_grid_preprocessing_id_missing")
    if preprocessing.get("resampler_id") != "mndm.causal_polyphase_kaiser_beta5.v0":
        raise ValueError("surface_grid_resampler_unsupported")
    if preprocessing.get("filter_id") != "mne.minimum_phase_firwin_hamming_constant.v0":
        raise ValueError("surface_grid_filter_unsupported")

    window_ids: set[str] = set()
    for item in windows:
        if not isinstance(item, dict):
            raise ValueError("surface_grid_window_entry_invalid")
        window_id = item.get("window_id")
        try:
            start_sec, end_sec = float(item["start_sec"]), float(item["end_sec"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("surface_grid_window_bounds_invalid") from exc
        if (
            not isinstance(window_id, str)
            or not window_id.strip()
            or window_id != window_id.strip()
            or window_id in window_ids
            or not math.isfinite(start_sec)
            or not math.isfinite(end_sec)
            or start_sec < 0
            or end_sec <= start_sec
        ):
            raise ValueError("surface_grid_window_bounds_or_id_invalid")
        window_ids.add(window_id)

    surface_ids: set[str] = set()
    for item in surfaces:
        if not isinstance(item, dict):
            raise ValueError("surface_grid_surface_entry_invalid")
        surface_id = item.get("surface_id")
        if (
            not isinstance(surface_id, str)
            or not surface_id.strip()
            or surface_id != surface_id.strip()
            or surface_id in surface_ids
        ):
            raise ValueError("surface_grid_surface_id_invalid_or_duplicate")
        surface_ids.add(surface_id)
        labels = item.get("channel_names")
        if labels != "all_eeg" and (
            not isinstance(labels, list)
            or not labels
            or any(not isinstance(label, str) for label in labels)
        ):
            raise ValueError("surface_grid_surface_labels_invalid")
        if labels != "all_eeg":
            label_keys = [label.strip().casefold() for label in labels]
            if any(not label for label in label_keys) or len(set(label_keys)) != len(label_keys):
                raise ValueError("surface_grid_surface_labels_must_be_nonempty_and_unique")
        kind = item.get("kind")
        if not isinstance(kind, str) or kind not in {"native", "core19", "study_common", "custom"}:
            raise ValueError("surface_grid_surface_kind_invalid")
        reference_mode = item.get("reference_mode")
        if not isinstance(reference_mode, str) or reference_mode not in {"recorded", "average_over_surface"}:
            raise ValueError("surface_grid_reference_mode_invalid")
        allow_partial = item.get("allow_partial_channels", False)
        if not isinstance(allow_partial, bool):
            raise ValueError("surface_grid_allow_partial_channels_must_be_boolean")
        reference_labels = item.get("reference_channel_names")
        if reference_labels is not None and (
            not isinstance(reference_labels, list)
            or not reference_labels
            or any(not isinstance(label, str) for label in reference_labels)
        ):
            raise ValueError("surface_grid_reference_channel_labels_invalid")
        if reference_labels is not None:
            reference_keys = [label.strip().casefold() for label in reference_labels]
            if any(not label for label in reference_keys) or len(set(reference_keys)) != len(reference_keys):
                raise ValueError("surface_grid_reference_channel_labels_must_be_nonempty_and_unique")
        reference_id = item.get("reference_id")
        if reference_id is not None and not isinstance(reference_id, str):
            raise ValueError("surface_grid_reference_id_must_be_string")
        if reference_mode == "average_over_surface" and (
            reference_id is not None or reference_labels is not None
        ):
            raise ValueError("surface_grid_average_reference_cannot_declare_recorded_reference")
        minimum_channels = item.get("minimum_channels", 2)
        if isinstance(minimum_channels, bool) or not isinstance(minimum_channels, int) or minimum_channels < 2:
            raise ValueError("surface_grid_minimum_channels_must_be_integer_at_least_two")
        if labels == "all_eeg":
            if kind != "native" or allow_partial is not True:
                raise ValueError("all_eeg_surface_must_be_partial_native")
        else:
            if minimum_channels > len(labels):
                raise ValueError("surface_grid_minimum_channels_exceeds_declared_labels")
            if kind in {"core19", "study_common"} and allow_partial:
                raise ValueError("fixed_surface_cannot_allow_partial_channels")
            if kind == "core19" and tuple(label.casefold() for label in labels) != tuple(
                label.casefold() for label in CORE19_CHANNELS
            ):
                raise ValueError("core19_surface_must_use_the_declared_core19_labels_in_order")

    resolved = {
        "source_path": source_path,
        "source_format": source_format,
        "associated_paths": associated_paths,
        "target_sfreq_hz": target_sfreq,
        "bandpass_hz": (float(bandpass[0]), float(bandpass[1])),
        "pipeline_id": pipeline_id,
    }
    return raw, resolved


def _open_mne_raw(source_path: Path, source_format: str) -> Any:
    try:
        import mne
    except ImportError as exc:
        raise RuntimeError("MNE is required to load an MNPS EEG source grid") from exc
    if source_format == "edf":
        return mne.io.read_raw_edf(source_path, preload=False, verbose="ERROR")
    if source_format == "brainvision":
        return mne.io.read_raw_brainvision(source_path, preload=False, verbose="ERROR")
    return mne.io.read_raw_fif(source_path, preload=False, verbose="ERROR")


def _make_surfaces(configs: Sequence[Mapping[str, Any]], eeg_labels: tuple[str, ...]) -> tuple[MNPS31SensorSurface, ...]:
    surfaces = []
    for item in configs:
        labels_value = item["channel_names"]
        if labels_value == "all_eeg":
            if item.get("kind") != "native" or item.get("allow_partial_channels") is not True:
                raise ValueError("all_eeg_surface_must_be_partial_native")
            labels = eeg_labels
        else:
            labels = tuple(str(label).strip() for label in labels_value)
        reference_names = item.get("reference_channel_names")
        surfaces.append(
            MNPS31SensorSurface(
                surface_id=item["surface_id"],
                kind=item["kind"],
                channel_names=labels,
                reference_mode=item["reference_mode"],
                reference_id=item.get("reference_id"),
                reference_channel_names=(None if reference_names is None else tuple(reference_names)),
                allow_partial_channels=item.get("allow_partial_channels", False),
                minimum_channels=item.get("minimum_channels", 2),
            )
        )
    return tuple(surfaces)


def _nominal_windows(
    configs: Sequence[Mapping[str, Any]],
    *,
    target_sfreq: float,
    source_sfreq: float,
    source_sample_count: int,
    processed_sample_count: int,
    filter_taps: int,
    resampler: Mapping[str, Any],
    annotations: Sequence[Mapping[str, Any]],
    source_sha256: str,
) -> tuple[MNPS31WindowSlice, ...]:
    rate_num = int(resampler["up"])
    rate_den = int(resampler["down"])
    resampler_taps = int(resampler["tap_count"])
    source_duration = source_sample_count / source_sfreq
    windows: list[MNPS31WindowSlice] = []
    for config in configs:
        start_sec, end_sec = float(config["start_sec"]), float(config["end_sec"])
        start_pos, end_pos = start_sec * target_sfreq, end_sec * target_sfreq
        if not math.isclose(start_pos, round(start_pos), rel_tol=0.0, abs_tol=1e-8) or not math.isclose(
            end_pos, round(end_pos), rel_tol=0.0, abs_tol=1e-8
        ):
            raise ValueError(f"window_not_aligned_to_output_clock:{config['window_id']}")
        start_sample, end_sample = int(round(start_pos)), int(round(end_pos))
        sample_count = end_sample - start_sample
        first_filtered_sample = start_sample - (filter_taps - 1)
        last_filtered_sample = end_sample - 1
        source_dependency_start = math.ceil(
            (first_filtered_sample * rate_den - (resampler_taps - 1)) / rate_num
        )
        source_dependency_end = math.floor(last_filtered_sample * rate_den / rate_num)
        dependency_start_sec = source_dependency_start / source_sfreq
        dependency_end_sec = (source_dependency_end + 1) / source_sfreq
        dependent_annotations = [
            dict(record)
            for record in annotations
            if _intersects(record, dependency_start_sec, dependency_end_sec)
        ]
        bad_annotations = [
            record for record in dependent_annotations
            if str(record["description"]).strip().upper().startswith("BAD")
        ]
        boundary_annotations = [
            record for record in dependent_annotations
            if any(
                token in str(record["description"]).upper()
                for token in ("ACQ_SKIP", "BOUNDARY", "DISCONTINU", "EDGE")
            )
        ]
        within_bounds = end_sec <= source_duration and end_sample <= processed_sample_count
        dependency_valid = (
            first_filtered_sample >= 0
            and last_filtered_sample < processed_sample_count
            and source_dependency_start >= 0
            and source_dependency_end < source_sample_count
        )
        support = MNPS31WindowSupport(
            window_start_sec=start_sec,
            window_end_sec=end_sec,
            sample_rate_hz=target_sfreq,
            sample_count=sample_count,
            segment_id=f"mne:{source_sha256[:16]}:segment0",
            source_is_continuous=not boundary_annotations,
            timestamps_uniform=True,
            gap_free=not boundary_annotations,
            bad_interval_free=not bad_annotations,
            within_recording_bounds=within_bounds,
            effective_filter_support_valid=dependency_valid,
        )
        windows.append(
            MNPS31WindowSlice(
                window_id=str(config["window_id"]),
                start_sample=start_sample,
                end_sample_exclusive=end_sample,
                support=support,
                provenance={
                    "requested_interval_semantics": "half_open",
                    "source_dependency": {
                        "source_sample_start": source_dependency_start,
                        "source_sample_end_exclusive": source_dependency_end + 1,
                        "source_time_start_sec": dependency_start_sec,
                        "source_time_end_sec": dependency_end_sec,
                        "filter_output_start_sample": first_filtered_sample,
                        "filter_output_end_sample": last_filtered_sample,
                    },
                    "dependent_annotations": dependent_annotations,
                    "annotation_support_sha256": _sha256_json(dependent_annotations),
                },
            )
        )
    return tuple(windows)


def run_e3_surface_grid_manifest(manifest_path: str | Path) -> MNPS31SurfaceGridRun:
    """Read one explicitly configured MNE source and compute E3 surface windows.

    Supported source formats are EDF, BrainVision, and FIF. EDF+D clocks and
    concatenated MNE Raw segments are rejected until their discontinuous sample
    maps can be processed without joining gaps. This function is opt-in and
    reads only the source files named by the manifest.
    """
    manifest_file = Path(manifest_path).resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError("surface_grid_manifest_not_found")
    manifest_stat_before = manifest_file.stat()
    manifest_sha256 = _sha256_file(manifest_file)
    manifest, config = _read_manifest(manifest_file)
    implementation_hashes = _implementation_hashes()
    source_path: Path = config["source_path"]
    associated_paths: tuple[Path, ...] = config["associated_paths"]
    source_paths = (source_path, *associated_paths)
    if any(not path.is_file() for path in source_paths):
        raise FileNotFoundError("surface_grid_source_or_associated_file_not_found")
    before = {str(path): {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for path in source_paths}

    raw = _open_mne_raw(source_path, config["source_format"])
    _verify_mne_source_files(raw, source_paths)
    if int(getattr(raw, "first_samp", 0)) != 0:
        raise ValueError("surface_grid_nonzero_first_sample_unsupported")
    raw_lengths = getattr(raw, "_raw_lengths", None)
    if raw_lengths is not None and len(raw_lengths) != 1:
        raise ValueError("surface_grid_concatenated_raw_segments_unsupported")
    if config["source_format"] == "edf":
        from ...preprocess import _verify_uniform_edf_sample_grid

        _verify_uniform_edf_sample_grid(raw, source_path)

    try:
        import mne
        from ...preprocess import _causal_polyphase_resample, _minimum_phase_eeg_filter_spec
    except ImportError as exc:
        raise RuntimeError("MNE and scipy are required for surface-grid preprocessing") from exc

    source_sfreq = float(raw.info["sfreq"])
    source_sample_count = int(raw.n_times)
    eeg_picks = np.asarray(mne.pick_types(raw.info, eeg=True, exclude=[]), dtype=np.int64)
    if eeg_picks.size == 0:
        raise ValueError("surface_grid_source_has_no_eeg_channels")
    all_source_labels = tuple(str(raw.ch_names[int(index)]).strip() for index in eeg_picks)
    if len({label.casefold() for label in all_source_labels}) != len(all_source_labels):
        raise ValueError("surface_grid_duplicate_eeg_labels")
    surfaces = _make_surfaces(manifest["surfaces"], all_source_labels)
    source_bad_labels = tuple(
        str(label).strip()
        for label in raw.info.get("bads", [])
        if str(label).strip().casefold() in {name.casefold() for name in all_source_labels}
    )
    raw_annotations = _annotations(raw)

    source_data = np.asarray(raw.get_data(picks=eeg_picks), dtype=np.float64)
    if source_data.shape != (len(all_source_labels), source_sample_count):
        raise ValueError("surface_grid_source_signal_shape_mismatch")
    finite_rows = np.isfinite(source_data).all(axis=1)
    nonfinite_labels = tuple(label for label, finite in zip(all_source_labels, finite_rows, strict=True) if not finite)
    kept_labels = tuple(label for label, finite in zip(all_source_labels, finite_rows, strict=True) if finite)
    if not kept_labels:
        raise ValueError("surface_grid_source_has_no_finite_eeg_channels")
    kept_picks = eeg_picks[finite_rows]
    work_data = source_data[finite_rows]
    work_info = mne.pick_info(raw.info, kept_picks.tolist(), copy=True)
    work_raw = mne.io.RawArray(work_data, work_info, first_samp=0, verbose="ERROR")

    target_sfreq = config["target_sfreq_hz"]
    bandpass_hz = config["bandpass_hz"]
    resampled_raw, resampler = _causal_polyphase_resample(work_raw, target_sfreq)
    filter_spec = _minimum_phase_eeg_filter_spec(resampled_raw, list(range(len(kept_labels))), bandpass_hz)
    resampled_raw.filter(
        l_freq=bandpass_hz[0],
        h_freq=bandpass_hz[1],
        picks=np.arange(len(kept_labels)),
        method="fir",
        phase="minimum",
        fir_design="firwin",
        fir_window="hamming",
        filter_length="auto",
        pad="constant",
        verbose="ERROR",
    )
    processed = np.asarray(resampled_raw.get_data(), dtype=np.float64)
    if processed.ndim != 2 or not np.isfinite(processed).all():
        raise ValueError("surface_grid_preprocessed_signal_nonfinite")

    after = {str(path): {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for path in source_paths}
    if before != after:
        raise RuntimeError("surface_grid_source_changed_during_read")
    source_sha256 = before[str(source_path)]["sha256"]
    windows = _nominal_windows(
        manifest["windows"],
        target_sfreq=target_sfreq,
        source_sfreq=source_sfreq,
        source_sample_count=source_sample_count,
        processed_sample_count=int(processed.shape[1]),
        filter_taps=int(filter_spec["tap_count"]),
        resampler=resampler,
        annotations=raw_annotations,
        source_sha256=source_sha256,
    )
    rows = compute_e3_surface_window_grid(
        processed,
        kept_labels,
        surfaces,
        windows,
        preprocessing_id=str(config["pipeline_id"]),
        bad_channel_names=tuple(
            label for label in source_bad_labels if label.casefold() in {name.casefold() for name in kept_labels}
        ),
    )

    manifest_sha256_after = _sha256_file(manifest_file)
    manifest_stat_after = manifest_file.stat()
    if (
        manifest_sha256 != manifest_sha256_after
        or manifest_stat_before.st_size != manifest_stat_after.st_size
        or manifest_stat_before.st_mtime_ns != manifest_stat_after.st_mtime_ns
    ):
        raise RuntimeError("surface_grid_manifest_changed_during_read")

    if implementation_hashes != _implementation_hashes():
        raise RuntimeError("surface_grid_implementation_changed_during_read")
    provenance: dict[str, object] = {
        "schema": SURFACE_GRID_RUN_SCHEMA,
        "manifest_id": manifest["manifest_id"],
        "manifest_path": str(manifest_file),
        "manifest_sha256": manifest_sha256,
        "recording_identity": dict(manifest["recording_identity"]),
        "source_files": before,
        "source_format": config["source_format"],
        "source_sample_rate_hz": source_sfreq,
        "source_sample_count": source_sample_count,
        "source_duration_sec": source_sample_count / source_sfreq,
        "all_eeg_channel_labels": list(all_source_labels),
        "processed_channel_labels": list(kept_labels),
        "source_bad_eeg_channel_labels": list(source_bad_labels),
        "source_nonfinite_eeg_channel_labels": list(nonfinite_labels),
        "non_eeg_channel_labels": [
            str(name) for index, name in enumerate(raw.ch_names) if index not in set(eeg_picks.tolist())
        ],
        "annotations_count": len(raw_annotations),
        "annotations_sha256": _sha256_json(raw_annotations),
        "resampler": resampler,
        "eeg_filter": filter_spec,
        "reference_applied_before_measurement": False,
        "target_sample_rate_hz": target_sfreq,
        "bandpass_hz": list(bandpass_hz),
        "pipeline_id": config["pipeline_id"],
        "implementation_sha256": implementation_hashes,
        "runtime": _runtime_versions(),
        "qualified": False,
        "claim_status": "no_biological_claim",
    }
    output_rows: list[dict[str, object]] = []
    for row in rows:
        output = dict(row)
        measurement = dict(row["measurement"])
        details = dict(measurement["details"])
        details["recording_identity"] = dict(manifest["recording_identity"])
        details["source_sha256"] = source_sha256
        details["associated_source_files_sha256"] = {
            path: info["sha256"] for path, info in before.items() if path != str(source_path)
        }
        details["manifest_sha256"] = manifest_sha256
        measurement["details"] = details
        output["measurement"] = measurement
        output_rows.append(output)
    return MNPS31SurfaceGridRun(rows=tuple(output_rows), provenance=provenance)


__all__ = [
    "SURFACE_GRID_MANIFEST_SCHEMA",
    "SURFACE_GRID_RUN_SCHEMA",
    "MNPS31SurfaceGridRun",
    "run_e3_surface_grid_manifest",
]
