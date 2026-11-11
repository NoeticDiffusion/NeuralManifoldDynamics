"""Execution provenance for signal preprocessing operations.

This module records what the existing preprocessing path actually attempted and
whether each operation succeeded.  It deliberately does not infer a finite
temporal support from a filter name or cutoff frequency.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import math
import numbers
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional


SCHEMA_VERSION = "mndm.signal_support_provenance.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except Exception:
            pass
    try:
        return _jsonable(value.item())  # numpy scalar
    except Exception:
        return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """Return a content hash for an auxiliary provenance source file."""
    return _sha256(Path(path))


def _version(package: str) -> Optional[str]:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def build_signal_support_provenance(
    file_path: Path,
    *,
    input_sfreq: Optional[float] = None,
    target_sfreq: Optional[float] = None,
    source_datatype: Optional[str] = None,
) -> dict[str, Any]:
    """Create a JSON-serialisable provenance record and hash ``file_path`` once."""
    path = Path(file_path)
    identity: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.exists():
        try:
            stat = path.stat()
            identity.update({"size_bytes": int(stat.st_size), "sha256": _sha256(path)})
        except Exception as exc:  # provenance remains explicit, never fabricated
            identity["hash_status"] = "failed"
            identity["hash_error"] = str(exc)
    else:
        identity["hash_status"] = "missing"

    # Read the installed distribution metadata without importing MNE.  Importing
    # MNE here can initialize numba and filesystem caches before preprocessing
    # has selected its runtime environment.
    mne_version = _version("mne")

    return {
        "schema": SCHEMA_VERSION,
        "status": "started",
        "source": identity,
        "source_datatype": source_datatype,
        "input_sfreq_hz": _jsonable(input_sfreq),
        "target_sfreq_hz": _jsonable(target_sfreq),
        "runtime": {
            "python": __import__("platform").python_version(),
            "mne": mne_version,
            "numpy": _version("numpy"),
            "scipy": _version("scipy"),
        },
        "operations": [],
        "temporal_support": {
            "status": "unknown",
            "reason": "effective filter/resampling support is not inferred by this layer",
            "epoch_bounds": "not_recorded_here",
        },
        "spatial_operations": [],
    }


def callable_signature(callable_obj: Any) -> Optional[str]:
    """Return the installed callable signature without invoking or changing it."""
    try:
        return str(inspect.signature(callable_obj))
    except Exception:
        return None


def refresh_source_hash(provenance: MutableMapping[str, Any]) -> None:
    """Re-hash the source at completion and record whether it stayed stable."""
    source = provenance.get("source", {})
    path = Path(str(source.get("path", "")))
    try:
        after = _sha256(path)
        source["sha256_after"] = after
        source["hash_stable"] = after == source.get("sha256")
    except Exception as exc:
        source["sha256_after"] = None
        source["hash_stable"] = False
        source["hash_after_error"] = str(exc)


def actual_crop_bounds(
    *,
    original_first_samp: int,
    actual_first_samp: int,
    actual_n_times: int,
    sfreq_hz: float,
    original_first_time_sec: float = 0.0,
) -> dict[str, Any]:
    """Derive clock bounds from the actual sample grid after a crop."""
    if sfreq_hz <= 0 or not all(isinstance(v, int) for v in (original_first_samp, actual_first_samp, actual_n_times)):
        raise ValueError("crop sample bounds require positive sfreq and integer samples")
    last_exclusive = actual_first_samp + actual_n_times
    return {
        "actual_first_samp": actual_first_samp,
        "actual_last_samp_exclusive": last_exclusive,
        "actual_tmin_sec": original_first_time_sec + (actual_first_samp - original_first_samp) / sfreq_hz,
        "actual_tmax_sec": original_first_time_sec + (last_exclusive - original_first_samp) / sfreq_hz,
    }


def map_preprocessed_window_to_source_samples(
    start_sample: int,
    end_sample_exclusive: int,
    *,
    preprocessed_sfreq_hz: float,
    provenance: Mapping[str, Any],
    allow_source_grid: bool = True,
    refusal_reason: str = "source_clock_mapping_not_supported",
) -> dict[str, Any]:
    """Map actual preprocessed-array bounds onto the original raw sample grid.

    Nominal bounds are always mapped when source clocks permit. A conservative
    effective contributor interval is added only when the explicit causal FIR
    pilot records every applied temporal dependency and the clocks agree.
    """
    unknown = {
        "source_sample_mapping_status": "unknown",
        "source_sample_mapping_reason": "source_clock_or_epoch_indices_unavailable",
        "nominal_source_sample_start": None,
        "nominal_source_sample_end_exclusive": None,
        "nominal_source_time_start_sec": None,
        "nominal_source_time_end_sec": None,
        "preprocessed_sample_start": None,
        "preprocessed_sample_end_exclusive": None,
        "actual_sample_contributors_status": "unknown",
        "actual_sample_contributors_reason": "operation_dependency_map_not_recorded",
        "actual_source_sample_start": None,
        "actual_source_sample_end_exclusive": None,
        "actual_source_time_start_sec": None,
        "actual_source_time_end_sec": None,
    }
    if not isinstance(start_sample, numbers.Integral) or not isinstance(end_sample_exclusive, numbers.Integral):
        return {**unknown, "source_sample_mapping_reason": "epoch_sample_bounds_not_integer"}
    start_sample = int(start_sample)
    end_sample_exclusive = int(end_sample_exclusive)
    if start_sample < 0 or end_sample_exclusive <= start_sample:
        return {**unknown, "source_sample_mapping_reason": "epoch_sample_bounds_invalid"}
    try:
        input_sfreq = float(preprocessed_sfreq_hz)
    except (TypeError, ValueError, OverflowError):
        input_sfreq = math.nan
    with_indices = {
        **unknown,
        "preprocessed_sample_start": start_sample,
        "preprocessed_sample_end_exclusive": end_sample_exclusive,
        "preprocessed_sfreq_hz": input_sfreq if math.isfinite(input_sfreq) else None,
    }
    if not allow_source_grid:
        return {**with_indices, "source_sample_mapping_reason": str(refusal_reason)}
    try:
        processed_hz = input_sfreq
        source_clock = provenance.get("raw_clock_before_crop", {})
        processed_clock = provenance.get("raw_clock_after_crop", {})
        source_hz = float(source_clock["sfreq_hz"])
        source_first_time = float(source_clock.get("first_time_sec", 0.0))
        source_first_samp = int(source_clock.get("first_samp", 0))
        source_n_times = int(source_clock["n_times"])
        processed_first_time = float(processed_clock["first_time_sec"])
        processed_n_times = int(processed_clock["n_times"])
        if (
            not all(math.isfinite(v) and v > 0 for v in (processed_hz, source_hz))
            or not all(math.isfinite(v) for v in (source_first_time, processed_first_time))
        ):
            raise ValueError("sample clock contains a non-finite or non-positive value")
        if end_sample_exclusive > processed_n_times:
            raise ValueError("epoch bounds exceed the preprocessed array")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return {**with_indices, "source_sample_mapping_reason": f"source_clock_invalid:{type(exc).__name__}"}

    # Epoch indices are produced by EEG extraction from the actual array. Use
    # the post-preprocessing first-sample clock, then map those times to the
    # original source grid. Do not reconstruct the start from t_start/t_end.
    absolute_time_start = processed_first_time + start_sample / processed_hz
    absolute_time_end = processed_first_time + end_sample_exclusive / processed_hz
    time_start = absolute_time_start - source_first_time
    time_end = absolute_time_end - source_first_time
    source_start_float = time_start * source_hz
    source_end_float = time_end * source_hz
    eps = max(1e-8, 1e-7 * source_hz)
    if (
        source_start_float < -eps
        or source_end_float > source_n_times + eps
        or source_end_float <= source_start_float
    ):
        return {
            **unknown,
            "source_sample_mapping_reason": "epoch_clock_bounds_outside_original_source",
            "preprocessed_sample_start": start_sample,
            "preprocessed_sample_end_exclusive": end_sample_exclusive,
            "preprocessed_sfreq_hz": processed_hz,
        }

    # Select source-grid timestamps in the half-open nominal interval. Values
    # within floating-point noise of an integer boundary are snapped first.
    start_pos = min(source_n_times, max(0, int(math.ceil(source_start_float - eps))))
    end_pos = min(source_n_times, max(0, int(math.ceil(source_end_float - eps))))
    if end_pos <= start_pos:
        return {
            **unknown,
            "source_sample_mapping_reason": "mapped_source_interval_empty",
            "preprocessed_sample_start": start_sample,
            "preprocessed_sample_end_exclusive": end_sample_exclusive,
            "preprocessed_sfreq_hz": processed_hz,
        }
    mapped = {
        "source_sample_mapping_status": "nominal_bounds_mapped",
        "source_sample_mapping_reason": "preprocessed_epoch_indices_mapped_to_original_clock",
        "nominal_source_sample_start": int(start_pos),
        "nominal_source_sample_end_exclusive": int(end_pos),
        "nominal_source_sample_absolute_start": int(source_first_samp + start_pos),
        "nominal_source_sample_absolute_end_exclusive": int(source_first_samp + end_pos),
        "source_first_samp": int(source_first_samp),
        "source_sfreq_hz": float(source_hz),
        "nominal_source_time_start_sec": float(time_start),
        "nominal_source_time_end_sec": float(time_end),
        "preprocessed_sample_start": int(start_sample),
        "preprocessed_sample_end_exclusive": int(end_sample_exclusive),
        "preprocessed_sfreq_hz": float(processed_hz),
        "actual_sample_contributors_status": "unknown",
        "actual_sample_contributors_reason": "filter_or_resampler_dependency_map_not_recorded",
        "actual_source_sample_start": None,
        "actual_source_sample_end_exclusive": None,
        "actual_source_time_start_sec": None,
        "actual_source_time_end_sec": None,
    }

    # Compute a finite causal bound only from an explicitly recorded identity
    # sample grid or causal polyphase FIR followed by the EEG FIR. Exact zero
    # taps may make this interval conservative but never too small.
    support = provenance.get("effective_time_support", {})
    if not isinstance(support, Mapping) or support.get("status") != "finite_causal":
        return mapped
    resampler = support.get("resampler", {})
    eeg_filter = support.get("eeg_bandpass", {})
    try:
        up = int(resampler["up"])
        down = int(resampler["down"])
        resampler_taps = int(resampler["tap_count"])
        filter_taps = int(eeg_filter["tap_count"])
        recorded_source_hz = float(resampler["source_sfreq_hz"])
        coefficient_hashes = (
            str(resampler["coefficient_sha256"]),
            str(eeg_filter["coefficient_sha256"]),
        )
        if min(up, down, resampler_taps, filter_taps) < 1:
            raise ValueError("nonpositive dependency parameters")
        resampler_method = resampler.get("method")
        identity_grid = resampler_method == "identity_sample_grid"
        if (
            (identity_grid and resampler.get("phase") != "identity")
            or (not identity_grid and resampler_method != "scipy.signal.upfirdn_causal_fir")
            or (not identity_grid and resampler.get("phase") != "causal_uncompensated")
            or eeg_filter.get("method") != "mne.filter.filter_data_fir"
            or eeg_filter.get("phase") != "minimum"
            or eeg_filter.get("pad") != "constant"
        ):
            raise ValueError("recorded temporal methods are not the supported causal FIR pair")
        if any(len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value) for value in coefficient_hashes):
            raise ValueError("causal FIR coefficient identity is missing or invalid")
        if int(resampler.get("output_sample_count", -1)) != processed_n_times:
            raise ValueError("recorded resampler sample count disagrees with processed clock")
        if source_first_samp != 0 or int(processed_clock.get("first_samp", 0)) != 0:
            raise ValueError("causal support mapper currently requires a zero-origin sample array")
        if identity_grid and (
            up != 1
            or down != 1
            or resampler_taps != 1
            or source_n_times != processed_n_times
            or abs(source_hz - processed_hz) > max(1e-9, processed_hz * 1e-10)
            or abs(processed_first_time - source_first_time) > max(1e-9, 1e-9 / source_hz)
        ):
            raise ValueError("recorded identity sample grid disagrees with source and processed clocks")
        if abs(source_hz * up / down - processed_hz) > max(1e-9, processed_hz * 1e-10):
            raise ValueError("recorded rational resampling ratio disagrees with sample clocks")
        if abs(recorded_source_hz - source_hz) > max(1e-9, source_hz * 1e-10):
            raise ValueError("recorded resampler source rate disagrees with source clock")
        if abs(processed_first_time - source_first_time) > max(1e-9, 1e-9 / source_hz):
            raise ValueError("causal support mapper does not support cropped or shifted clocks")
        temporal_ops = [
            op for op in provenance.get("operations", [])
            if isinstance(op, Mapping) and op.get("class", "temporal") == "temporal" and op.get("status") == "applied"
        ]
        if any(
            op.get("name") not in {"resample", "eeg_bandpass"}
            or op.get("temporal_support_status") != "finite_causal"
            for op in temporal_ops
        ):
            raise ValueError("an applied temporal operation lacks a finite causal dependency record")
        expected_temporal_ops = ["eeg_bandpass"] if identity_grid else ["resample", "eeg_bandpass"]
        if [op.get("name") for op in temporal_ops] != expected_temporal_ops:
            raise ValueError("recorded causal temporal operations do not match the supported sample-grid path")

        first_resampled = max(0, start_sample - (filter_taps - 1))
        last_resampled = end_sample_exclusive - 1
        low_numerator = first_resampled * down - (resampler_taps - 1)
        source_low = -((-low_numerator) // up)  # integer ceil division
        source_high = (last_resampled * down) // up
        source_low = min(source_n_times, max(0, source_low))
        source_high = min(source_n_times - 1, max(-1, source_high))
        if source_high < source_low:
            raise ValueError("computed finite source dependency interval is empty")
        actual_end = source_high + 1
        mapped.update({
            "source_sample_mapping_reason": "preprocessed_epoch_clock_mapped; finite causal contributors separately bounded",
            "actual_sample_contributors_status": "finite_causal_conservative_bound",
            "actual_sample_contributors_reason": "contiguous superset from recorded causal FIR tap lengths and rational upfirdn dependency; clipped at source boundaries",
            "actual_source_sample_start": int(source_low),
            "actual_source_sample_end_exclusive": int(actual_end),
            "actual_source_time_start_sec": float(source_low / source_hz),
            "actual_source_time_end_sec": float(actual_end / source_hz),
        })
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        mapped["actual_sample_contributors_reason"] = f"finite_causal_support_unavailable:{type(exc).__name__}"
    return mapped


def quality_interval(
    *,
    source_kind: str,
    start_sec: float,
    duration_sec: float,
    clock: str = "original_raw_seconds",
    **extra: Any,
) -> dict[str, Any]:
    """Build one source interval without turning invalid values into masks."""
    if not math.isfinite(float(start_sec)) or not math.isfinite(float(duration_sec)) or float(duration_sec) < 0:
        raise ValueError("quality interval requires finite nonnegative start/duration")
    return {
        "source_kind": str(source_kind),
        "start_sec": float(start_sec),
        "end_sec": float(start_sec) + float(duration_sec),
        "duration_sec": float(duration_sec),
        "clock": str(clock),
        **_jsonable(extra),
    }


def original_raw_annotation_start(onset_sec: float, raw_first_time_sec: float) -> float:
    """Map an MNE annotation onset to original-file-relative raw seconds."""
    if not math.isfinite(float(onset_sec)) or not math.isfinite(float(raw_first_time_sec)):
        raise ValueError("annotation onset and raw first time must be finite")
    return float(onset_sec) - float(raw_first_time_sec)


def start_operation(
    provenance: MutableMapping[str, Any],
    name: str,
    *,
    operation_class: str = "temporal",
    input_sfreq: Optional[float] = None,
    parameters: Optional[Mapping[str, Any]] = None,
    actual_kwargs: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Append a started operation and return the mutable record."""
    record: dict[str, Any] = {
        "operation_id": f"{len(provenance.get('operations', [])):03d}_{name}",
        "name": name,
        "class": operation_class,
        "status": "started",
        "started_at_utc": _utc_now(),
        "input_sfreq_hz": _jsonable(input_sfreq),
        "parameters": _jsonable(parameters or {}),
        "actual_kwargs": _jsonable(actual_kwargs or {}),
        "temporal_support_status": "unknown" if operation_class == "temporal" else "not_applicable",
        "support_reason": (
            "effective support not available from this execution hook"
            if operation_class == "temporal"
            else "spatial operation; temporal support is not applicable"
        ),
    }
    provenance.setdefault("operations", []).append(record)
    return record


def finish_operation(
    record: MutableMapping[str, Any],
    *,
    status: str,
    output_sfreq: Optional[float] = None,
    fallback: Optional[str] = None,
    error: Optional[BaseException | str] = None,
) -> None:
    """Complete an operation without hiding failure or fallback details."""
    if status not in {"applied", "skipped", "fallback", "failed"}:
        raise ValueError(f"invalid provenance operation status: {status}")
    record["status"] = status
    record["finished_at_utc"] = _utc_now()
    record["output_sfreq_hz"] = _jsonable(output_sfreq)
    if fallback is not None:
        record["fallback"] = str(fallback)
    if error is not None:
        record["error"] = str(error)
