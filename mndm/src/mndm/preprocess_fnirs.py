"""Bounded fNIRS optical preprocessing with explicit scientific status.

This module implements the frozen G2 reference path for one validated
continuous-wave fNIRS recording. It does not estimate task contrasts, filter
the canonical time series, construct regions, produce MNPS, or qualify a
measurement. MNE dependencies are optional and loaded only when processing is
requested.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np


MNE_VERSION = "1.6.1"
MNE_NIRS_VERSION = "0.7.3"
SHORT_CHANNEL_MAX_DIST_M = 0.010
SHORT_CHANNEL_BOUNDARY_ATOL_M = 1e-12
BEER_LAMBERT_PPF = 6.0
SCI_LOW_FREQUENCY_HZ = 0.7
SCI_HIGH_FREQUENCY_HZ = 1.5


class FnirsPreprocessingError(RuntimeError):
    """A fail-closed fNIRS preprocessing error with a stable reason code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class FnirsPreprocessingResult:
    """Computed optical-density and haemoglobin objects for one recording.

    ``computation_status`` reports execution only. ``measurement_validity``
    stays ``not_assessed`` because running these transforms does not establish
    signal quality, correction efficacy, or scientific qualification.
    """

    raw_od: Any
    raw_haemoglobin: Any
    scalp_coupling_index: Mapping[str, float]
    diagnostics: Mapping[str, Any]
    provenance: Mapping[str, Any]
    computation_status: str = "computed"
    measurement_validity: str = "not_assessed"
    claim_status: str = "no_biological_claim"


@dataclass(frozen=True)
class _MneFns:
    mne_version: str
    mne_nirs_version: str
    optical_density: Any
    scalp_coupling_index: Any
    tddr: Any
    short_channel_regression: Any
    beer_lambert_law: Any
    molar_unit_code: int


def _fail(code: str, message: str) -> None:
    raise FnirsPreprocessingError(code, message)


def _load_mne_functions() -> _MneFns:
    """Load the exact MNE API versions exercised by the G2 runtime smoke."""
    try:
        import mne
        import mne_nirs
        from mne.preprocessing.nirs import (
            beer_lambert_law,
            optical_density,
            scalp_coupling_index,
            temporal_derivative_distribution_repair,
        )
        from mne_nirs.signal_enhancement import short_channel_regression
    except Exception as exc:  # optional packages or transitive imports
        _fail(
            "preprocessing_dependency_unavailable",
            "MNE 1.6.1 and MNE-NIRS 0.7.3 with their fNIRS APIs are required "
            f"for this path ({type(exc).__name__}).",
        )
    mne_version = str(getattr(mne, "__version__", "unknown"))
    mne_nirs_version = str(getattr(mne_nirs, "__version__", "unknown"))
    if mne_version != MNE_VERSION or mne_nirs_version != MNE_NIRS_VERSION:
        _fail(
            "preprocessing_dependency_version_unsupported",
            f"Tested versions are MNE {MNE_VERSION} and MNE-NIRS "
            f"{MNE_NIRS_VERSION}; found MNE {mne_version} and MNE-NIRS "
            f"{mne_nirs_version}.",
        )
    return _MneFns(
        mne_version=mne_version,
        mne_nirs_version=mne_nirs_version,
        optical_density=optical_density,
        scalp_coupling_index=scalp_coupling_index,
        tddr=temporal_derivative_distribution_repair,
        short_channel_regression=short_channel_regression,
        beer_lambert_law=beer_lambert_law,
        molar_unit_code=int(mne.io.constants.FIFF.FIFF_UNIT_MOL),
    )


def _validation_parts(validation: Any) -> tuple[bool, str | None, Mapping[str, Any]]:
    """Read either the metadata report object or its JSON-ready dictionary."""
    if isinstance(validation, Mapping):
        valid = validation.get("valid")
        admissible = validation.get("admissible")
        classification = validation.get("classification")
        summary = validation.get("summary")
    else:
        valid = getattr(validation, "valid", None)
        admissible = getattr(validation, "admissible", None)
        classification = getattr(validation, "classification", None)
        summary = getattr(validation, "summary", None)
    if not isinstance(valid, bool) or not isinstance(admissible, bool) or not isinstance(summary, Mapping):
        _fail(
            "metadata_validation_report_invalid",
            "Pass the result from validate_fnirs_recording (or its to_dict() value).",
        )
    if not valid:
        _fail("metadata_not_valid", "The metadata validator did not mark this recording valid.")
    return admissible, str(classification) if classification is not None else None, summary


def _raw_clock(raw: Any, stage: str) -> dict[str, Any]:
    try:
        sfreq = float(raw.info["sfreq"])
        n_times = int(raw.n_times)
        first_samp = int(raw.first_samp)
        first_time = float(raw.first_time)
        times = np.asarray(raw.times, dtype=float)
        channel_names = tuple(str(name) for name in raw.ch_names)
    except Exception as exc:
        _fail("raw_clock_unavailable", f"Could not read the MNE sample clock at {stage} ({type(exc).__name__}).")
    if (
        not np.isfinite(sfreq)
        or sfreq <= 0
        or n_times < 2
        or not np.isfinite(first_time)
        or times.shape != (n_times,)
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
        or len(channel_names) != int(raw.info["nchan"])
    ):
        _fail("raw_clock_invalid", f"The MNE sample clock or channel layout is invalid at {stage}.")
    return {
        "sampling_frequency_hz": sfreq,
        "sample_count": n_times,
        "first_sample_index": first_samp,
        "first_time_s": first_time,
        "relative_time_start_s": float(times[0]),
        "relative_time_end_s": float(times[-1]),
        "channel_names": channel_names,
        "times": times,
    }


def _assert_clock_preserved(
    expected: Mapping[str, Any],
    raw: Any,
    stage: str,
    *,
    preserve_channel_names: bool = True,
) -> None:
    actual = _raw_clock(raw, stage)
    if (
        actual["sampling_frequency_hz"] != expected["sampling_frequency_hz"]
        or actual["sample_count"] != expected["sample_count"]
        or actual["first_sample_index"] != expected["first_sample_index"]
        or actual["first_time_s"] != expected["first_time_s"]
        or (preserve_channel_names and actual["channel_names"] != expected["channel_names"])
        or not np.array_equal(actual["times"], expected["times"])
    ):
        _fail("sample_clock_changed", f"The sample clock changed or channel layout changed unexpectedly during {stage}.")


def _raw_data(raw: Any, stage: str) -> np.ndarray:
    try:
        data = np.asarray(raw.get_data(), dtype=float)
    except Exception as exc:
        _fail("raw_data_unavailable", f"Could not read signal values at {stage} ({type(exc).__name__}).")
    if data.ndim != 2 or data.shape != (len(raw.ch_names), int(raw.n_times)):
        _fail("raw_data_shape_invalid", f"Signal data must have channel-by-sample shape at {stage}.")
    if not np.isfinite(data).all():
        _fail("raw_data_nonfinite", f"Signal data contain non-finite values at {stage}.")
    return data


def _channel_types(raw: Any, expected_type: str, stage: str) -> None:
    try:
        channel_types = tuple(raw.get_channel_types())
    except Exception as exc:
        _fail("channel_types_unavailable", f"Could not read fNIRS channel types at {stage} ({type(exc).__name__}).")
    if not channel_types or any(channel_type != expected_type for channel_type in channel_types):
        _fail(
            "channel_type_unsupported",
            f"All channels must be MNE type {expected_type!r} at {stage}; found {channel_types!r}.",
        )


def _check_report_against_raw(
    raw: Any,
    summary: Mapping[str, Any],
    classification: str,
) -> dict[str, Any]:
    clock = _raw_clock(raw, "input")
    declared = {
        "channel_count": len(raw.ch_names),
        "sample_count": int(raw.n_times),
        "sampling_frequency_hz": float(raw.info["sfreq"]),
    }
    for field, actual in declared.items():
        try:
            recorded = summary[field]
        except KeyError:
            _fail("metadata_summary_incomplete", f"Metadata validation summary lacks {field!r}.")
        if field == "sampling_frequency_hz":
            try:
                agrees = np.isclose(float(recorded), actual, rtol=1e-9, atol=1e-12)
            except (TypeError, ValueError):
                agrees = False
        else:
            agrees = isinstance(recorded, (int, np.integer)) and int(recorded) == actual
        if not agrees:
            _fail(
                "metadata_raw_mismatch",
                f"Metadata summary {field}={recorded!r} does not match the loaded recording value {actual!r}.",
            )
    gaps = summary.get("time_gap_sample_indices")
    if not isinstance(gaps, (list, tuple)):
        _fail("metadata_summary_incomplete", "Metadata summary must report time_gap_sample_indices.")
    if gaps:
        _fail("sample_clock_gaps_unsupported", f"The validated recording contains sample gaps at {list(gaps)!r}.")
    try:
        metadata_start = float(summary["time_start_s"])
        metadata_end = float(summary["time_end_s"])
        metadata_interval = float(summary["median_sample_interval_s"])
    except (KeyError, TypeError, ValueError):
        _fail("metadata_time_support_incomplete", "Metadata summary must include measured time start, end, and sample interval.")
    if not np.isfinite([metadata_start, metadata_end, metadata_interval]).all() or metadata_interval <= 0:
        _fail("metadata_time_support_invalid", "Metadata time support must be finite with a positive sample interval.")
    absolute_start = float(clock["first_time_s"] + clock["relative_time_start_s"])
    absolute_end = float(clock["first_time_s"] + clock["relative_time_end_s"])
    time_tolerance = max(metadata_interval * 1e-6, 1e-12)
    if not np.isclose(absolute_start, metadata_start, rtol=1e-9, atol=time_tolerance) or not np.isclose(
        absolute_end, metadata_end, rtol=1e-9, atol=time_tolerance
    ):
        _fail(
            "metadata_raw_time_mismatch",
            f"Raw time support {absolute_start:g}..{absolute_end:g}s disagrees with SNIRF metadata {metadata_start:g}..{metadata_end:g}s.",
        )
    if not np.isclose(metadata_interval, 1.0 / clock["sampling_frequency_hz"], rtol=1e-6, atol=1e-12):
        _fail("metadata_raw_sample_interval_mismatch", "Measured SNIRF sample interval disagrees with the loaded Raw clock.")
    if classification in {"cw_amplitude", "optical_density"}:
        wavelengths = summary.get("wavelengths_nm")
        if not isinstance(wavelengths, (list, tuple)) or len(wavelengths) != 2:
            _fail("metadata_wavelengths_unsupported", "The frozen path requires exactly two validated wavelengths.")
        try:
            wavelength_values = [float(value) for value in wavelengths]
        except (TypeError, ValueError):
            _fail("metadata_wavelengths_invalid", "Validated wavelengths must be finite positive numbers.")
        if not np.isfinite(wavelength_values).all() or any(value <= 0 for value in wavelength_values) or wavelength_values[0] == wavelength_values[1]:
            _fail("metadata_wavelengths_invalid", "Validated wavelengths must be distinct, finite, positive numbers.")
        try:
            raw_channel_wavelengths = np.asarray(
                [float(channel["loc"][9]) for channel in raw.info["chs"]],
                dtype=float,
            )
        except Exception as exc:
            _fail("metadata_raw_wavelength_unavailable", f"Could not read loaded Raw channel wavelengths ({type(exc).__name__}).")
        if raw_channel_wavelengths.shape != (declared["channel_count"],) or not np.isfinite(raw_channel_wavelengths).all():
            _fail("metadata_raw_wavelength_invalid", "Loaded Raw channel wavelengths must be finite and match the channel count.")
        try:
            pair_count = int(summary["measurement_pair_count"])
        except (KeyError, TypeError, ValueError):
            _fail("metadata_summary_incomplete", "Metadata summary must provide measurement_pair_count.")
        if pair_count < 1 or declared["channel_count"] != 2 * pair_count:
            _fail("metadata_channel_pairs_unsupported", "Each validated source-detector pair must contain exactly two wavelength channels.")
        try:
            short_pairs = int(summary["short_channel_pair_count"])
        except (KeyError, TypeError, ValueError):
            _fail("short_channel_metadata_missing", "Metadata summary must provide the geometric short-channel pair count.")
        if short_pairs < 1:
            _fail("short_channels_unavailable", "The frozen regression path requires at least one <10 mm channel pair.")
        channel_keys = summary.get("channel_semantic_keys")
        if not isinstance(channel_keys, (list, tuple)) or len(channel_keys) != declared["channel_count"]:
            _fail("metadata_channel_identity_incomplete", "Metadata summary must contain one ordered semantic key per signal channel.")
        expected_names: list[str] = []
        semantic_wavelengths: list[float] = []
        for key in channel_keys:
            if not isinstance(key, Mapping):
                _fail("metadata_channel_identity_invalid", "Every channel semantic key must identify source, detector, and wavelength.")
            try:
                source = str(key["source"])
                detector = str(key["detector"])
                wavelength = float(key["wavelength_or_chromophore"])
            except (KeyError, TypeError, ValueError):
                _fail("metadata_channel_identity_invalid", "Every channel semantic key must identify source, detector, and wavelength.")
            if not np.isfinite(wavelength):
                _fail("metadata_channel_identity_invalid", "Channel wavelengths must be finite.")
            if not any(np.isclose(wavelength, expected, rtol=0.0, atol=1e-6) for expected in wavelength_values):
                _fail("metadata_channel_wavelength_invalid", "Each channel wavelength must match one of the validated recording wavelengths.")
            expected_names.append(f"{source}_{detector} {format(wavelength, 'g')}")
            semantic_wavelengths.append(wavelength)
        if tuple(expected_names) != tuple(clock["channel_names"]):
            _fail(
                "metadata_raw_channel_identity_mismatch",
                "Loaded Raw channel identities/order do not match the validated SNIRF measurement-list order.",
            )
        if not np.allclose(raw_channel_wavelengths, semantic_wavelengths, rtol=0.0, atol=1e-6):
            _fail(
                "metadata_raw_wavelength_mismatch",
                "Loaded Raw channel wavelengths/order do not match the validated SNIRF measurement-list metadata.",
            )
    return clock


def _apply_stage(
    fn: Any,
    raw: Any,
    *,
    stage: str,
    expected_clock: Mapping[str, Any],
    kwargs: Mapping[str, Any] | None = None,
    preserve_channel_names: bool = True,
) -> Any:
    try:
        output = fn(raw, **dict(kwargs or {}))
    except Exception as exc:
        _fail(f"{stage}_failed", f"fNIRS operation {stage!r} failed ({type(exc).__name__}); no fallback was applied.")
    if output is None:
        _fail(f"{stage}_failed", f"fNIRS operation {stage!r} returned no result; no fallback was applied.")
    _assert_clock_preserved(expected_clock, output, stage, preserve_channel_names=preserve_channel_names)
    _raw_data(output, stage)
    return output


def _source_provenance(source_path: str | Path | None, source_sha256: str | None) -> dict[str, Any]:
    path_text: str | None = None
    digest = source_sha256
    if source_path is None:
        _fail(
            "source_identity_missing",
            "A verifiable source_path is required; a detached SHA-256 does not bind the Raw object to its source.",
        )
    if source_path is not None:
        path = Path(source_path)
        path_text = str(path)
        if not path.is_file():
            _fail("source_file_unavailable", f"Source file does not exist: {path}.")
        hasher = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
        except OSError as exc:
            _fail("source_hash_failed", f"Could not hash the source file ({type(exc).__name__}).")
        actual_digest = hasher.hexdigest()
        if digest is not None and str(digest).lower() != actual_digest:
            _fail("source_hash_mismatch", "The supplied source SHA-256 does not match the source file.")
        digest = actual_digest
    if digest is not None:
        digest = str(digest).lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            _fail("source_hash_invalid", "source_sha256 must be a 64-character hexadecimal SHA-256 digest.")
    return {"path": path_text, "sha256": digest}


def _assert_source_bound_to_raw(raw: Any, source: Mapping[str, Any]) -> None:
    """Ensure the supplied source path is the file MNE attached to this Raw."""
    expected = source.get("path")
    filenames = getattr(raw, "filenames", None)
    if not expected or not filenames:
        _fail("raw_source_identity_unavailable", "Loaded Raw must retain its source filename for provenance binding.")
    try:
        actual_paths = {Path(value).resolve() for value in filenames if value}
        expected_path = Path(str(expected)).resolve()
    except (OSError, TypeError, ValueError) as exc:
        _fail("raw_source_identity_invalid", f"Could not resolve Raw source identity ({type(exc).__name__}).")
    if expected_path not in actual_paths:
        _fail(
            "raw_source_identity_mismatch",
            "The supplied source_path does not match the filename attached to the loaded Raw object.",
        )


def _wavelength_stratified_short_channel_regression(
    raw: Any,
    regression_fn: Any,
    *,
    expected_clock: Mapping[str, Any],
    wavelengths_nm: tuple[float, ...],
) -> tuple[Any, list[dict[str, Any]]]:
    """Run the pinned nearest-channel regression within each wavelength.

    MNE-NIRS 0.7.3 selects one nearest OD *channel*. Passing all wavelengths
    together can make equal-geometry 760/850 channels tie, so both long
    wavelengths may use the same short wavelength. Subsetting by validated
    wavelength keeps the library's geometric regression while preserving the
    optical wavelength pairing. Boundary-distance channels are rejected
    because the pinned library classifies only distances strictly below or
    strictly above ``max_dist``.
    """
    try:
        channel_locations = np.asarray([channel["loc"] for channel in raw.info["chs"]], dtype=float)
        if channel_locations.shape != (len(raw.ch_names), 12):
            raise ValueError("channel location array has an unexpected shape")
        distances = np.linalg.norm(channel_locations[:, 3:6] - channel_locations[:, 6:9], axis=1)
    except Exception as exc:
        _fail("short_channel_geometry_unavailable", f"Could not resolve source-detector distances ({type(exc).__name__}).")
    if distances.shape != (len(raw.ch_names),) or not np.isfinite(distances).all():
        _fail("short_channel_geometry_invalid", "Source-detector distances must be finite and match the channel layout.")
    boundary = np.flatnonzero(np.isclose(
        distances,
        SHORT_CHANNEL_MAX_DIST_M,
        rtol=0.0,
        atol=SHORT_CHANNEL_BOUNDARY_ATOL_M,
    ))
    if len(boundary):
        names = [str(raw.ch_names[int(index)]) for index in boundary]
        _fail(
            "short_channel_distance_boundary_unsupported",
            "The pinned regression library leaves channels at the exact short/long cutoff untouched; "
            f"these channels are ambiguous at 10 mm: {names!r}.",
        )

    try:
        channel_wavelengths = np.asarray([float(channel["loc"][9]) for channel in raw.info["chs"]], dtype=float)
    except Exception as exc:
        _fail("short_channel_wavelength_unavailable", f"Could not read channel wavelengths ({type(exc).__name__}).")
    if channel_wavelengths.shape != (len(raw.ch_names),) or not np.isfinite(channel_wavelengths).all():
        _fail("short_channel_wavelength_invalid", "Every fNIRS channel must have a finite wavelength for regression pairing.")
    if len(wavelengths_nm) != 2 or not np.isfinite(wavelengths_nm).all() or wavelengths_nm[0] == wavelengths_nm[1]:
        _fail("short_channel_wavelength_contract_invalid", "The frozen regression requires exactly two distinct validated wavelengths.")

    try:
        corrected = raw.copy().load_data()
    except Exception as exc:
        _fail("short_channel_regression_copy_failed", f"Could not copy the optical-density data ({type(exc).__name__}).")
    regression_records: list[dict[str, Any]] = []
    for wavelength in wavelengths_nm:
        picks = np.flatnonzero(np.isclose(channel_wavelengths, wavelength, rtol=0.0, atol=1e-6))
        if not len(picks):
            _fail("short_channel_wavelength_missing", f"No channels were found at validated wavelength {wavelength:g} nm.")
        try:
            wavelength_raw = raw.copy().pick(picks)
        except Exception as exc:
            _fail("short_channel_wavelength_split_failed", f"Could not isolate {wavelength:g} nm channels ({type(exc).__name__}).")
        wavelength_distances = distances[picks]
        short_local = np.flatnonzero(wavelength_distances < SHORT_CHANNEL_MAX_DIST_M)
        long_local = np.flatnonzero(wavelength_distances > SHORT_CHANNEL_MAX_DIST_M)
        if not len(short_local):
            _fail("short_channel_regressor_unavailable", f"No short OD channel exists at {wavelength:g} nm.")
        if not len(long_local):
            _fail("long_channel_regression_target_unavailable", f"No long OD channel exists at {wavelength:g} nm.")

        wavelength_data = np.asarray(wavelength_raw.get_data(), dtype=float)
        if not np.isfinite(wavelength_data).all():
            _fail("short_channel_input_nonfinite", f"OD data at {wavelength:g} nm contain non-finite values.")
        for target_local in long_local:
            target_global = int(picks[int(target_local)])
            target_loc = channel_locations[target_global, :3]
            short_distances = np.asarray([
                np.linalg.norm(target_loc - channel_locations[int(picks[int(short_index)]), :3])
                for short_index in short_local
            ], dtype=float)
            if not np.isfinite(short_distances).all():
                _fail("short_channel_nearest_distance_invalid", "Nearest short-channel distances must be finite.")
            nearest_order = np.argsort(short_distances, kind="stable")
            nearest_distance = float(short_distances[int(nearest_order[0])])
            tied_nearest = np.flatnonzero(np.isclose(
                short_distances,
                nearest_distance,
                rtol=0.0,
                atol=SHORT_CHANNEL_BOUNDARY_ATOL_M,
            ))
            tie_count = int(len(tied_nearest))
            if tie_count > 1:
                tied_names = [str(raw.ch_names[int(picks[int(short_local[int(index)])])]) for index in tied_nearest]
                _fail(
                    "short_channel_nearest_regressor_ambiguous",
                    f"Multiple short channels are equally near {raw.ch_names[target_global]!r}: {tied_names!r}.",
                )
            nearest_local = int(short_local[int(nearest_order[0])])
            nearest_global = int(picks[nearest_local])
            regressor = wavelength_data[nearest_local]
            target = wavelength_data[int(target_local)]
            denominator = float(np.dot(regressor, regressor))
            if not np.isfinite(denominator) or denominator <= 0.0:
                _fail(
                    "short_channel_regressor_degenerate",
                    f"The selected short regressor {raw.ch_names[nearest_global]!r} has zero or invalid energy.",
                )
            coefficient = float(np.dot(regressor, target) / denominator)
            if not np.isfinite(coefficient):
                _fail("short_channel_regression_coefficient_invalid", f"The fitted coefficient for {raw.ch_names[target_global]!r} is non-finite.")
            regression_records.append({
                "target_channel": str(raw.ch_names[target_global]),
                "target_wavelength_nm": float(wavelength),
                "regressor_channel": str(raw.ch_names[nearest_global]),
                "regressor_wavelength_nm": float(channel_wavelengths[nearest_global]),
                "regressor_source_detector_distance_m": float(distances[nearest_global]),
                "target_to_regressor_midpoint_distance_m": nearest_distance,
                "nearest_regressor_tie_count": tie_count,
                "coefficient": coefficient,
                "fit_sample_count": int(raw.n_times),
            })

        wavelength_raw = _apply_stage(
            regression_fn,
            wavelength_raw,
            stage="short_channel_regression",
            expected_clock=expected_clock,
            preserve_channel_names=False,
            kwargs={"max_dist": SHORT_CHANNEL_MAX_DIST_M},
        )
        expected_subset_names = [str(raw.ch_names[int(index)]) for index in picks]
        if list(wavelength_raw.ch_names) != expected_subset_names:
            _fail(
                "short_channel_regression_channel_identity_mismatch",
                f"Short-channel regression at {wavelength:g} nm changed channel count or order.",
            )
        corrected._data[picks, :] = wavelength_raw.get_data()

    _assert_clock_preserved(expected_clock, corrected, "short_channel_regression_merge")
    _channel_types(corrected, "fnirs_od", "short_channel_regression")
    _raw_data(corrected, "short_channel_regression")
    return corrected, regression_records


def preprocess_fnirs(
    raw: Any,
    *,
    validation: Any,
    input_classification: str,
    source_path: str | Path | None = None,
    source_sha256: str | None = None,
) -> FnirsPreprocessingResult:
    """Apply the frozen G2 fNIRS path to one metadata-validated MNE Raw.

    Call ``validate_fnirs_recording`` first and pass its report. The report is
    cross-checked against the loaded MNE Raw's dimensions, clock, and
    classification. Supported input classifications are ``cw_amplitude``
    and ``optical_density``. No filters, windows, cutoffs, regions, or MNPS
    features are applied here.

    The required operations are intensity-to-OD when applicable; continuous
    scalp-coupling index on OD with the tested 0.7–1.5 Hz diagnostic band;
    TDDR on OD; nearest short-channel regression on OD with max distance
    10 mm, applied within each wavelength; and Beer-Lambert conversion using
    PPF 6.0. A source-detector distance exactly at 10 mm is rejected because
    the pinned MNE-NIRS selector leaves that boundary unclassified.
    """
    classification = str(input_classification).strip().lower()
    if classification not in {"cw_amplitude", "optical_density"}:
        _fail("input_classification_unsupported", f"Unsupported fNIRS input classification {classification!r}.")
    admissible, validated_classification, summary = _validation_parts(validation)
    if not admissible:
        _fail("metadata_not_admissible", "The metadata validator did not mark this recording admissible for the v0 path.")
    if validated_classification != classification:
        _fail(
            "metadata_classification_mismatch",
            f"Metadata classification {validated_classification!r} disagrees with requested {classification!r}.",
        )

    source = _source_provenance(source_path, source_sha256)
    _assert_source_bound_to_raw(raw, source)
    validated_digest = summary.get("snirf_sha256")
    if not isinstance(validated_digest, str) or len(validated_digest) != 64:
        _fail("metadata_source_identity_missing", "Metadata validation report lacks the SNIRF source SHA-256.")
    if source["sha256"] != validated_digest.lower():
        _fail(
            "metadata_source_identity_mismatch",
            "The SNIRF source bytes do not match the source validated by the metadata report.",
        )
    expected_clock = _check_report_against_raw(raw, summary, classification)
    input_type = "fnirs_cw_amplitude" if classification == "cw_amplitude" else "fnirs_od"
    _channel_types(raw, input_type, "input")
    input_data = _raw_data(raw, "input")
    if classification == "cw_amplitude" and np.any(input_data <= 0):
        _fail(
            "intensity_nonpositive",
            "Continuous-wave intensity must be strictly positive before optical-density conversion.",
        )

    fns = _load_mne_functions()
    try:
        working = raw.copy().load_data()
    except Exception as exc:
        _fail("raw_copy_failed", f"Could not create a preloaded working copy ({type(exc).__name__}).")

    operations: list[dict[str, Any]] = []
    if classification == "cw_amplitude":
        working = _apply_stage(
            fns.optical_density,
            working,
            stage="optical_density",
            expected_clock=expected_clock,
        )
        _channel_types(working, "fnirs_od", "optical_density")
        operations.append({"name": "intensity_to_optical_density", "status": "applied", "parameters": {}})
    else:
        _channel_types(working, "fnirs_od", "optical_density_input")
        operations.append({"name": "intensity_to_optical_density", "status": "not_applicable", "reason": "input_already_optical_density"})
    _raw_data(working, "optical_density")

    try:
        sci = np.asarray(
            fns.scalp_coupling_index(
                working,
                l_freq=SCI_LOW_FREQUENCY_HZ,
                h_freq=SCI_HIGH_FREQUENCY_HZ,
            ),
            dtype=float,
        )
    except Exception as exc:
        _fail("scalp_coupling_index_failed", f"Could not compute the continuous SCI diagnostic ({type(exc).__name__}).")
    if sci.shape != (len(working.ch_names),) or not np.isfinite(sci).all():
        _fail("scalp_coupling_index_invalid", "SCI must return one finite continuous value per input channel.")
    sci_by_channel = {str(name): float(value) for name, value in zip(working.ch_names, sci)}
    operations.append(
        {
            "name": "scalp_coupling_index",
            "status": "applied_diagnostic",
            "parameters": {
                "l_freq_hz": SCI_LOW_FREQUENCY_HZ,
                "h_freq_hz": SCI_HIGH_FREQUENCY_HZ,
                "acceptance_cutoff": None,
            },
        }
    )

    working = _apply_stage(
        fns.tddr,
        working,
        stage="tddr",
        expected_clock=expected_clock,
    )
    _channel_types(working, "fnirs_od", "tddr")
    operations.append({"name": "temporal_derivative_distribution_repair", "status": "applied", "parameters": {}})

    working, regression_records = _wavelength_stratified_short_channel_regression(
        working,
        fns.short_channel_regression,
        expected_clock=expected_clock,
        wavelengths_nm=tuple(float(value) for value in summary["wavelengths_nm"]),
    )
    _channel_types(working, "fnirs_od", "short_channel_regression")
    operations.append(
        {
            "name": "wavelength_stratified_nearest_short_channel_regression",
            "status": "applied",
            "parameters": {
                "max_dist_m": SHORT_CHANNEL_MAX_DIST_M,
                "regression_domain": "optical_density",
                "wavelength_matched": True,
                "channel_regressors": regression_records,
            },
        }
    )

    raw_haemoglobin = _apply_stage(
        fns.beer_lambert_law,
        working,
        stage="beer_lambert",
        expected_clock=expected_clock,
        kwargs={"ppf": BEER_LAMBERT_PPF},
        preserve_channel_names=False,
    )
    try:
        haemoglobin_types = tuple(raw_haemoglobin.get_channel_types())
    except Exception as exc:
        _fail("haemoglobin_types_unavailable", f"Could not verify Beer-Lambert output types ({type(exc).__name__}).")
    if not haemoglobin_types or any(channel_type not in {"hbo", "hbr"} for channel_type in haemoglobin_types):
        _fail("haemoglobin_output_unsupported", f"Beer-Lambert output must contain only HbO/HbR channels; found {haemoglobin_types!r}.")
    expected_hb_channels = 2 * int(summary["measurement_pair_count"])
    if len(haemoglobin_types) != expected_hb_channels or haemoglobin_types.count("hbo") != expected_hb_channels // 2 or haemoglobin_types.count("hbr") != expected_hb_channels // 2:
        _fail("haemoglobin_output_incomplete", "Beer-Lambert output must contain one HbO and one HbR channel per validated source-detector pair.")
    hb_by_pair: dict[str, set[str]] = {}
    for name in raw_haemoglobin.ch_names:
        try:
            pair_name, chromophore = str(name).rsplit(" ", 1)
        except ValueError:
            _fail("haemoglobin_channel_name_invalid", f"Cannot identify the source-detector pair in output channel {name!r}.")
        hb_by_pair.setdefault(pair_name, set()).add(chromophore.lower())
    if len(hb_by_pair) != int(summary["measurement_pair_count"]) or any(kinds != {"hbo", "hbr"} for kinds in hb_by_pair.values()):
        _fail("haemoglobin_pair_mapping_invalid", "Beer-Lambert output must map one HbO/HbR pair to each source-detector pair.")
    expected_pairs_raw = summary.get("measurement_pair_keys")
    if not isinstance(expected_pairs_raw, (list, tuple)):
        _fail("metadata_pair_identity_incomplete", "Metadata summary must provide the validated source-detector pair identities.")
    expected_pairs: set[str] = set()
    for pair in expected_pairs_raw:
        if not isinstance(pair, Mapping) or not pair.get("source") or not pair.get("detector"):
            _fail("metadata_pair_identity_invalid", "Each validated pair must identify a source and detector.")
        expected_pairs.add(f"{pair['source']}_{pair['detector']}")
    if set(hb_by_pair) != expected_pairs:
        _fail(
            "haemoglobin_pair_identity_mismatch",
            "Beer-Lambert output pair identities do not match the validated source-detector pairs.",
        )
    try:
        hb_units = tuple(int(channel["unit"]) for channel in raw_haemoglobin.info["chs"])
    except Exception as exc:
        _fail("haemoglobin_units_unavailable", f"Could not read Beer-Lambert output units ({type(exc).__name__}).")
    if len(hb_units) != len(haemoglobin_types) or any(unit != fns.molar_unit_code for unit in hb_units):
        _fail(
            "haemoglobin_units_unsupported",
            "Beer-Lambert output channels must use the MNE molar unit (FIFF_UNIT_MOL).",
        )
    operations.append(
        {
            "name": "beer_lambert_law",
            "status": "applied",
            "parameters": {"ppf": BEER_LAMBERT_PPF},
        }
    )

    provenance = {
        "schema": "mndm.fnirs_preprocessing_provenance.v1",
        "source": source,
        "input_classification": classification,
        "validation": {
            "valid": True,
            "admissible": True,
            "classification": validated_classification,
            "summary": dict(summary),
        },
        "software": {"mne": fns.mne_version, "mne_nirs": fns.mne_nirs_version},
        "operations": operations,
        "sample_clock": {
            key: value
            for key, value in expected_clock.items()
            if key != "times" and key != "channel_names"
        },
        "filter_policy": "no canonical time-series filter applied; SCI diagnostic uses its recorded 0.7-1.5 Hz band",
        "feature_scope": "channel-level HbO/HbR only; no windows, regions, or MNPS features",
        "haemoglobin_units": "M (molar; MNE FIFF_UNIT_MOL)",
        "claim_status": "no_biological_claim",
        "measurement_validity": "not_assessed",
    }
    diagnostics = {
        "scalp_coupling_index": dict(sci_by_channel),
        "scalp_coupling_index_cutoff": None,
        "short_channel_pair_count_from_metadata": int(summary["short_channel_pair_count"]),
        "signal_quality_assessment": "not_assessed",
        "correction_efficacy": "not_assessed",
    }
    return FnirsPreprocessingResult(
        raw_od=working,
        raw_haemoglobin=raw_haemoglobin,
        scalp_coupling_index=sci_by_channel,
        diagnostics=diagnostics,
        provenance=provenance,
    )
