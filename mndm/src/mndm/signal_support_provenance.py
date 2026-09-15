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
