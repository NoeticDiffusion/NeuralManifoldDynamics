"""FAR-003A native EEG synchronization and NMD time-base audit.

This gate is deliberately outcome-blind.  It reads native EEG samples only for
finite/clipping/rail QC around frozen FAR-002 stimulation events.  It does not
calculate features, displacements, responses, home/away regions, or
resilience outcomes.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from .far_perturbation_family_semantics import (
    OPTIONAL_PHYSICAL_V_FIELDS,
    V_FIELDS,
    _clean_text,
    _duration_class,
    _family_id,
    _v_key,
)
from .far_source_metadata_audit import (
    _finite_float,
    _nwb_file_records,
    _nwb_trial_table,
    _numeric_text,
    _truthy,
    _walk_files,
)


PROTOCOL_ID = "FAR-003A"
HORIZONS_SEC = (10.0, 5.0, 2.0, 1.0, 0.5)
ARTIFACT_PROBE_SEC = 0.005
MIN_SUBJECTS_PER_RHO = 2
NMD_CONFIG_RELATIVE = Path(
    "mndm/config/config_ingest_common_nwb_mouse_eeg.yaml"
)

SIGNAL_TIMEBASE_PASS = "SIGNAL_TIMEBASE_PASS"
RAW_SIGNAL_ONLY_PASS = "RAW_SIGNAL_ONLY_PASS"
NMD_TIMEBASE_METHOD_LIMITED = "NMD_TIMEBASE_METHOD_LIMITED"
STIM_ARTIFACT_NOT_TESTABLE = "STIM_ARTIFACT_NOT_TESTABLE"
EVENT_OVERLAP_NOT_TESTABLE = "EVENT_OVERLAP_NOT_TESTABLE"
INSUFFICIENT_COVERAGE = "INSUFFICIENT_COVERAGE"
SOURCE_SYNC_UNRESOLVED = "SOURCE_SYNC_UNRESOLVED"
STIM_ARTIFACT_UNRESOLVED = "STIM_ARTIFACT_UNRESOLVED"
STIM_ARTIFACT_FREE = "STIM_ARTIFACT_FREE"


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _session_id(path: Path) -> str:
    match = re.search(r"(ses-[^_]+)", path.name)
    return match.group(1) if match else path.stem


def _subject_id(path: Path) -> str:
    if path.parent.name.startswith("sub-"):
        return path.parent.name
    match = re.search(r"(sub-[^_]+)", path.name)
    return match.group(1) if match else path.parent.name


def _load_far002(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != "FAR-002":
        raise ValueError("far_002_protocol_id_mismatch")
    if payload.get("global_status") != "PASS":
        raise ValueError("far_002_global_status_not_pass")
    curve_rows = [
        row
        for row in payload.get("perturbation_family_ledger", [])
        if row.get("curve_status") == "CURVE_SEMANTICS_PASS"
    ]
    if len(curve_rows) != 10:
        raise ValueError(f"expected_ten_far_002_curve_rows:{len(curve_rows)}")
    return payload


def _make_v_values(table: dict[str, list[Any]], index: int) -> dict[str, str]:
    values = {
        field: _clean_text(table[field][index])
        for field in V_FIELDS
        if field in table
    }
    values["observed_stim_duration_class"] = _duration_class(
        (
            _finite_float(table["stop_time"][index])
            - _finite_float(table["start_time"][index])
        )
        if _finite_float(table["start_time"][index]) is not None
        and _finite_float(table["stop_time"][index]) is not None
        else None
    )
    for field in V_FIELDS:
        values.setdefault(field, "")
    for field in OPTIONAL_PHYSICAL_V_FIELDS:
        if field in table:
            values[field] = _clean_text(table[field][index])
    return values


def _candidate_key(table: dict[str, list[Any]], index: int) -> tuple[str, str]:
    values = _make_v_values(table, index)
    return _family_id(_v_key(values)), _clean_text(table["behavioral_epoch"][index]).lower()


def _scan_timestamps(timestamps: Any) -> dict[str, Any]:
    chunk_size = 1_000_000
    total = int(timestamps.shape[0])
    first = float(timestamps[0])
    last = float(timestamps[-1])
    finite = True
    monotonic = True
    gap_intervals: list[tuple[float, float]] = []
    previous: float | None = None
    expected_dt: float | None = None
    for start in range(0, total, chunk_size):
        values = np.asarray(
            timestamps[start : min(start + chunk_size, total)],
            dtype=float,
        )
        if values.size == 0:
            continue
        finite = finite and bool(np.isfinite(values).all())
        joined = (
            np.concatenate(([previous], values))
            if previous is not None
            else values
        )
        differences = np.diff(joined)
        positive = differences[np.isfinite(differences) & (differences > 0)]
        if expected_dt is None and positive.size:
            expected_dt = float(np.median(positive[: min(positive.size, 100_000)]))
        if differences.size:
            monotonic = monotonic and bool(np.all(differences > 0))
            if expected_dt is not None:
                tolerance = max(expected_dt * 0.01, 1.0e-6)
                bad = np.flatnonzero(np.abs(differences - expected_dt) > tolerance)
                for index in bad:
                    left = float(joined[index])
                    right = float(joined[index + 1])
                    if right - left > expected_dt + tolerance:
                        gap_intervals.append((left, right))
        previous = float(values[-1])
    return {
        "signal_start": first,
        "signal_end": last,
        "n_samples": total,
        "sample_interval_sec": expected_dt,
        "sample_rate_hz": 1.0 / expected_dt if expected_dt else None,
        "timestamp_finite": finite,
        "timestamp_monotonic": monotonic,
        "gap_intervals": gap_intervals,
        "gap_count": len(gap_intervals),
    }


def _find_gap(
    gaps: list[tuple[float, float]],
    start: float,
    end: float,
) -> bool:
    return any(gap_start < end and gap_end > start for gap_start, gap_end in gaps)


def _event_record(
    *,
    event: dict[str, Any],
    signal: dict[str, Any],
    data: Any,
    timestamps: Any,
    all_events: list[dict[str, Any]],
    native_recovery_available: bool,
    timestamp_values: np.ndarray | None = None,
) -> dict[str, Any]:
    start = signal["signal_start"]
    end = signal["signal_end"]
    dt = signal["sample_interval_sec"]
    stim_on = float(event["stim_on"])
    stim_off = float(event["stim_off"])
    timestamp_valid = bool(
        signal["timestamp_finite"]
        and signal["timestamp_monotonic"]
        and dt
        and start <= stim_on <= end
        and start <= stim_off <= end
        and stim_off >= stim_on
    )
    sample_index = None
    finite_probe = False
    rail_hits = 0
    if timestamp_valid:
        if timestamp_values is not None:
            sample_index = int(np.searchsorted(timestamp_values, stim_on))
            sample_index = max(0, min(sample_index, len(timestamp_values) - 1))
            if (
                sample_index > 0
                and abs(timestamp_values[sample_index - 1] - stim_on)
                < abs(timestamp_values[sample_index] - stim_on)
            ):
                sample_index -= 1
            timestamp_valid = timestamp_valid and bool(
                abs(float(timestamp_values[sample_index]) - stim_on)
                <= max(2.0 * dt, 0.001)
            )
        else:
            sample_index = int(round((stim_on - start) / dt))
        sample_index = max(0, min(sample_index, int(data.shape[0]) - 1))
        probe_end_time = min(stim_off + ARTIFACT_PROBE_SEC, end)
        if timestamp_values is not None:
            probe_end = int(np.searchsorted(timestamp_values, probe_end_time, side="right"))
        else:
            probe_end = int(round((probe_end_time - start) / dt)) + 1
        probe_end = max(sample_index + 1, min(probe_end, int(data.shape[0])))
        try:
            probe = np.asarray(data[sample_index:probe_end, :])
            finite_probe = bool(np.isfinite(probe).all())
            if np.issubdtype(data.dtype, np.integer):
                info = np.iinfo(data.dtype)
                rail_hits = int(np.count_nonzero((probe == info.min) | (probe == info.max)))
        except (OSError, ValueError, RuntimeError):
            finite_probe = False
    if not finite_probe or rail_hits:
        artifact_probe_status = STIM_ARTIFACT_UNRESOLVED
    else:
        artifact_probe_status = STIM_ARTIFACT_FREE
    artifact_recovery_status = (
        STIM_ARTIFACT_FREE
        if native_recovery_available and artifact_probe_status == STIM_ARTIFACT_FREE
        else STIM_ARTIFACT_UNRESOLVED
    )
    # ``artifact_status`` is the technical native probe result.  Recovery
    # semantics are reported separately: an unresolved native blanking rule
    # prevents biological interpretation, but does not erase a clean,
    # predeclared finite/rail QC result.
    artifact_status = artifact_probe_status

    previous_candidates = [
        candidate
        for candidate in all_events
        if candidate["event_id"] != event["event_id"]
        and candidate["stim_off"] <= stim_on
    ]
    next_candidates = [
        candidate
        for candidate in all_events
        if candidate["event_id"] != event["event_id"]
        and candidate["stim_on"] >= stim_on
    ]
    previous = (
        max(previous_candidates, key=lambda candidate: candidate["stim_off"])
        if previous_candidates
        else None
    )
    next_event = (
        min(next_candidates, key=lambda candidate: candidate["stim_on"])
        if next_candidates
        else None
    )
    previous_latency = (
        stim_on - float(previous["stim_off"]) if previous is not None else None
    )
    next_latency = (
        float(next_event["stim_on"]) - stim_off if next_event is not None else None
    )
    by_horizon: dict[str, bool] = {}
    gap_by_horizon: dict[str, bool] = {}
    overlap_by_horizon: dict[str, bool] = {}
    for horizon in HORIZONS_SEC:
        pre_available = stim_on - start >= horizon
        post_available = end - stim_off >= horizon
        overlap = (
            next_latency is not None and not next_latency > horizon
        ) or (
            previous_latency is not None and not previous_latency > horizon
        )
        gap_crossing = _find_gap(
            signal["gap_intervals"],
            stim_on - horizon,
            stim_off + horizon,
        )
        gap_by_horizon[str(horizon)] = gap_crossing
        overlap_by_horizon[str(horizon)] = overlap
        by_horizon[str(horizon)] = bool(
            timestamp_valid
            and pre_available
            and post_available
            and not overlap
            and not gap_crossing
            and artifact_status == STIM_ARTIFACT_FREE
        )
    return {
        "event_id": event["event_id"],
        "family_id": event["family_id"],
        "w": event["w"],
        "v": event["v"],
        "subject": event["subject"],
        "session": event["session"],
        "rho": event["rho"],
        "stim_on": stim_on,
        "stim_off": stim_off,
        "signal_start": start,
        "signal_end": end,
        "pre_available": bool(stim_on - start >= min(HORIZONS_SEC)),
        "post_available": bool(end - stim_off >= min(HORIZONS_SEC)),
        "gap_crossing": _find_gap(
            signal["gap_intervals"],
            stim_on - min(HORIZONS_SEC),
            stim_off + min(HORIZONS_SEC),
        ),
        "next_stim_latency": next_latency,
        "previous_stim_latency": previous_latency,
        "timestamp_valid": timestamp_valid,
        "sample_index": sample_index,
        "finite_probe": finite_probe,
        "rail_hits": rail_hits,
        "artifact_probe_status": artifact_probe_status,
        "artifact_recovery_status": artifact_recovery_status,
        "artifact_status": artifact_status,
        "raw_signal_eligible_by_horizon": by_horizon,
        "gap_crossing_by_horizon": gap_by_horizon,
        "overlap_by_horizon": overlap_by_horizon,
    }


def _load_nmd_timebase(repo_root: Path) -> dict[str, Any]:
    config_path = repo_root / NMD_CONFIG_RELATIVE
    text = config_path.read_text(encoding="utf-8") if config_path.is_file() else ""
    imported_text = ""
    import_match = re.search(r'^\s*-\s*"([^"]+)"', text, flags=re.MULTILINE)
    if import_match:
        imported_path = (config_path.parent / import_match.group(1)).resolve()
        if imported_path.is_file():
            imported_text = imported_path.read_text(encoding="utf-8")
            second_import = re.search(
                r'^\s*-\s*"([^"]+)"',
                imported_text,
                flags=re.MULTILINE,
            )
            if second_import:
                second_path = (imported_path.parent / second_import.group(1)).resolve()
                if second_path.is_file():
                    imported_text += "\n" + second_path.read_text(
                        encoding="utf-8"
                    )
    resolved_text = text + "\n" + imported_text

    def _number(pattern: str, default: float | None) -> float | None:
        match = re.search(pattern, resolved_text)
        return float(match.group(1)) if match else default

    artifact_candidates = [
        repo_root / "data/dandi/processed/dandi_000458",
        repo_root / "data/dandi/processed/000458",
        repo_root / "data/dandi/processed_lfp/dandi_000458",
        repo_root / "data/dandi/processed_lfp/000458",
    ]
    return {
        "status": "NMD_TIMEBASE_NOT_TESTABLE",
        "reason": "no_existing_serialized_nmd_artifact_or_time_grid",
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path) if config_path.is_file() else None,
        "configured_epoch_length_sec": _number(r"length_s:\s*([0-9.]+)", 4.0),
        "configured_epoch_step_sec": _number(r"step_s:\s*([0-9.]+)", 2.0),
        "configured_mnps_window_sec": _number(r"window_sec:\s*([0-9.]+)", 8.0),
        "configured_mnps_fs_out_hz": _number(r"fs_out:\s*([0-9.]+)", 4.0),
        "artifact_candidates": [
            {"path": str(path), "exists": path.exists()}
            for path in artifact_candidates
        ],
    }


def _identity_from_nmd_path(path: Path) -> tuple[str | None, str | None]:
    subject_match = re.search(r"(sub-[^_\\/]+)", str(path))
    session_match = re.search(r"(ses-[^_\\/]+)", str(path))
    return (
        subject_match.group(1) if subject_match else None,
        session_match.group(1) if session_match else None,
    )


def _load_nmd_windows(
    nmd_info: dict[str, Any],
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]]:
    """Read only existing NMD time-grid datasets, never coordinate payloads."""

    windows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    files: list[str] = []
    errors: list[dict[str, str]] = []
    try:
        import h5py
    except ImportError:
        return windows, {
            "serialized_artifact_count": 0,
            "serialized_artifacts": [],
            "time_grid_errors": [{"error": "h5py_required"}],
        }
    for candidate in nmd_info["artifact_candidates"]:
        root = Path(candidate["path"])
        if not root.is_dir():
            continue
        for path in sorted(
            item
            for item in root.rglob("*")
            if item.is_file() and item.suffix.lower() in {".h5", ".hdf5"}
        ):
            files.append(str(path))
            subject, session = _identity_from_nmd_path(path)
            if subject is None or session is None:
                errors.append(
                    {
                        "path": str(path),
                        "error": "subject_session_not_frozen_in_artifact_path",
                    }
                )
                continue
            try:
                with h5py.File(path, "r") as handle:
                    starts = None
                    ends = None

                    def visitor(name: str, item: Any) -> None:
                        nonlocal starts, ends
                        if not hasattr(item, "shape"):
                            return
                        basename = name.rsplit("/", 1)[-1].lower()
                        if basename == "window_start" and starts is None:
                            starts = np.asarray(item[:], dtype=float)
                        elif basename == "window_end" and ends is None:
                            ends = np.asarray(item[:], dtype=float)

                    handle.visititems(visitor)
                    if starts is None or ends is None:
                        errors.append(
                            {
                                "path": str(path),
                                "error": "window_start_or_window_end_missing",
                            }
                        )
                        continue
                    if starts.ndim != 1 or ends.ndim != 1 or len(starts) != len(ends):
                        errors.append(
                            {
                                "path": str(path),
                                "error": "window_grid_shape_mismatch",
                            }
                        )
                        continue
                    finite = np.isfinite(starts) & np.isfinite(ends)
                    ordered = np.r_[True, np.diff(starts) >= 0]
                    if not bool(np.all(finite & (ends >= starts) & ordered)):
                        errors.append(
                            {
                                "path": str(path),
                                "error": "window_grid_invalid",
                            }
                        )
                        continue
                    windows[(subject, session)].append(
                        {
                            "starts": starts,
                            "ends": ends,
                            "path": str(path),
                        }
                    )
            except (OSError, ValueError, RuntimeError) as error:
                errors.append(
                    {
                        "path": str(path),
                        "error": f"{type(error).__name__}:{error}",
                    }
                )
    return windows, {
        "serialized_artifact_count": len(files),
        "serialized_artifacts": files,
        "time_grid_errors": errors,
        "mapped_subject_session_count": len(windows),
    }


def _nmd_alignment(
    record: dict[str, Any],
    nmd_windows: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]],
) -> dict[str, Any]:
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    for window_set in nmd_windows.get(
        (record["subject"], record["session"]),
        [],
    ):
        starts.append(window_set["starts"])
        ends.append(window_set["ends"])
    if not starts:
        return {
            "aligned": False,
            "pre_windows": 0,
            "post_windows": 0,
        }
    all_starts = np.concatenate(starts)
    all_ends = np.concatenate(ends)
    pre_windows = int(np.count_nonzero(all_ends <= record["stim_on"]))
    post_windows = int(np.count_nonzero(all_starts >= record["stim_off"]))
    return {
        "aligned": bool(pre_windows > 0 and post_windows > 0),
        "pre_windows": pre_windows,
        "post_windows": post_windows,
    }


def _resolve_nmd_clock(
    nmd_windows: dict[tuple[str, str], list[dict[str, Any]]],
    signal_cache: dict[Path, dict[str, Any]],
) -> dict[str, Any]:
    native_ranges = {
        (_subject_id(path), _session_id(path)): (
            float(summary["signal_start"]),
            float(summary["signal_end"]),
        )
        for path, summary in signal_cache.items()
    }
    clock_audit: dict[str, Any] = {}
    aligned_windows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for identity, window_sets in nmd_windows.items():
        starts = np.concatenate([item["starts"] for item in window_sets])
        ends = np.concatenate([item["ends"] for item in window_sets])
        nmd_start = float(np.min(starts))
        nmd_end = float(np.max(ends))
        native = native_ranges.get(identity)
        if native is None:
            status = "SOURCE_SYNC_UNRESOLVED"
        else:
            native_start, native_end = native
            duration_match = abs(
                (nmd_end - nmd_start) - (native_end - native_start)
            ) <= 0.5
            absolute_match = (
                abs(nmd_start - native_start) <= 0.5
                and abs(nmd_end - native_end) <= 0.5
            )
            elapsed_match = abs(nmd_start) <= 0.5 and duration_match
            if absolute_match:
                status = "CLOCK_ABSOLUTE_ALIGNED"
                aligned_windows[identity].extend(window_sets)
            elif elapsed_match:
                status = "CLOCK_ELAPSED_UNRESOLVED"
            else:
                status = "CLOCK_UNRESOLVED"
        clock_audit[f"{identity[0]}/{identity[1]}"] = {
            "status": status,
            "nmd_start": nmd_start,
            "nmd_end": nmd_end,
            "native_start": native[0] if native else None,
            "native_end": native[1] if native else None,
        }
    return {
        "windows": aligned_windows,
        "audit": clock_audit,
    }


def _native_recovery_metadata(handle: Any) -> dict[str, Any]:
    """Inspect only acquisition/stimulus metadata for a native recovery rule."""

    matches: list[str] = []
    for root_name in ("/acquisition", "/stimulus", "/intervals"):
        root = handle.get(root_name)
        if root is None:
            continue
        for name, item in root.items():
            lowered = str(name).lower()
            if any(token in lowered for token in ("blank", "artifact", "recover")):
                matches.append(f"{root_name}/{name}")
            for attr_name in item.attrs:
                attr_lowered = str(attr_name).lower()
                if any(
                    token in attr_lowered
                    for token in ("blank", "artifact", "recover")
                ):
                    matches.append(f"{root_name}/{name}@{attr_name}")
    return {
        "found": bool(matches),
        "paths": sorted(set(matches)),
    }


def audit_far_003a(
    *,
    source_root: Path,
    far002_path: Path,
    protocol_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    far002 = _load_far002(far002_path)
    expected_source_root = str(far002.get("source_root", ""))
    expected_file_records = {
        str(record.get("relative_path")): int(record.get("size_bytes", -1))
        for record in far002.get("native_nwb_files", [])
    }
    frozen_point_rows = sum(
        row.get("curve_status") != "CURVE_SEMANTICS_PASS"
        for row in far002.get("perturbation_family_ledger", [])
    )
    frozen_rows = {
        (str(row["family_id"]), str(row["operating_state_w"])): row
        for row in far002["perturbation_family_ledger"]
        if row.get("curve_status") == "CURVE_SEMANTICS_PASS"
    }
    files = _walk_files(source_root, suffixes={".nwb"})
    if not files:
        return {
            "schema": "mndm.far_003a_signal_timebase.v1",
            "protocol_id": PROTOCOL_ID,
            "global_status": "BLOCKED",
            "global_reason": "no_native_nwb_files",
            "far_003b_authorized": False,
        }
    actual_file_records = {
        str(record.get("relative_path")): int(record.get("size_bytes", -1))
        for record in _nwb_file_records(files, source_root)
    }
    source_binding = {
        "expected_source_root": expected_source_root,
        "observed_source_root": str(source_root),
        "source_root_match": expected_source_root == str(source_root),
        "expected_file_count": len(expected_file_records),
        "observed_file_count": len(actual_file_records),
        "file_set_match": set(expected_file_records) == set(actual_file_records),
        "file_size_match": expected_file_records == actual_file_records,
    }
    if not (
        source_binding["source_root_match"]
        and source_binding["file_set_match"]
        and source_binding["file_size_match"]
    ):
        return {
            "schema": "mndm.far_003a_signal_timebase.v1",
            "protocol_id": PROTOCOL_ID,
            "global_status": "BLOCKED",
            "global_reason": "far_002_native_source_binding_mismatch",
            "far_003b_authorized": False,
            "source_root": str(source_root),
            "source_binding": source_binding,
            "far002_certificate": {
                "path": str(far002_path),
                "sha256": sha256_file(far002_path),
            },
        }

    columns = [
        "behavioral_epoch",
        "estim_current",
        "estim_target_depth",
        "estim_target_region",
        "is_valid",
        "start_time",
        "stimulus_description",
        "stimulus_type",
        "stop_time",
        *OPTIONAL_PHYSICAL_V_FIELDS,
    ]
    candidate_events: list[dict[str, Any]] = []
    all_events_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    file_errors: list[dict[str, Any]] = []
    for path in files:
        relative = str(path.relative_to(source_root)).replace("\\", "/")
        try:
            table, _descriptions, missing = _nwb_trial_table(path, columns)
            required = {
                "behavioral_epoch",
                "estim_current",
                "estim_target_depth",
                "estim_target_region",
                "is_valid",
                "start_time",
                "stimulus_description",
                "stimulus_type",
                "stop_time",
            }
            if required & set(missing):
                file_errors.append(
                    {
                        "relative_path": relative,
                        "missing_required_columns": sorted(required & set(missing)),
                    }
                )
                continue
            n_rows = int(table["_row_count"][0])
            for index in range(n_rows):
                stim_type = _clean_text(table["stimulus_type"][index]).lower()
                on = _finite_float(table["start_time"][index])
                off = _finite_float(table["stop_time"][index])
                if stim_type not in {"electrical", "visual"} or on is None or off is None:
                    continue
                all_events_by_file[relative].append(
                    {
                        "event_id": f"{relative}:trial-{index}",
                        "stim_on": on,
                        "stim_off": off,
                        "event_type": stim_type,
                    }
                )
                if stim_type != "electrical" or not _truthy(table["is_valid"][index]):
                    continue
                current = _numeric_text(table["estim_current"][index])
                if current is None:
                    continue
                v_values = _make_v_values(table, index)
                family_id = _family_id(_v_key(v_values))
                state = _clean_text(table["behavioral_epoch"][index]).lower()
                key = (family_id, state)
                if key not in frozen_rows:
                    continue
                frozen_row = frozen_rows[key]
                if json.dumps(
                    v_values,
                    sort_keys=True,
                    ensure_ascii=False,
                ) != json.dumps(
                    frozen_row["perturbation_identity_v"],
                    sort_keys=True,
                    ensure_ascii=False,
                ):
                    file_errors.append(
                        {
                            "relative_path": relative,
                            "trial_index": index,
                            "error": "frozen_v_binding_mismatch",
                        }
                    )
                    continue
                if not any(
                    np.isclose(
                        current,
                        float(level),
                        rtol=0.0,
                        atol=1.0e-9,
                    )
                    for level in frozen_row.get("rho_levels", [])
                ):
                    file_errors.append(
                        {
                            "relative_path": relative,
                            "trial_index": index,
                            "error": "frozen_rho_level_binding_mismatch",
                        }
                    )
                    continue
                candidate_events.append(
                    {
                        "event_id": f"{relative}:trial-{index}",
                        "family_id": family_id,
                        "w": state,
                        "v": v_values,
                        "subject": _subject_id(path),
                        "session": _session_id(path),
                        "rho": current,
                        "stim_on": on,
                        "stim_off": off,
                        "file_path": path,
                        "relative_path": relative,
                    }
                )
        except (OSError, ValueError, KeyError, TypeError) as error:
            file_errors.append(
                {
                    "relative_path": relative,
                    "error": f"{type(error).__name__}:{error}",
                }
            )

    signal_cache: dict[Path, dict[str, Any]] = {}
    event_records: list[dict[str, Any]] = []
    for path in files:
        events = [event for event in candidate_events if event["file_path"] == path]
        if not events:
            continue
        relative = str(path.relative_to(source_root)).replace("\\", "/")
        try:
            import h5py

            with h5py.File(path, "r") as handle:
                series = handle.get("/acquisition/ElectricalSeriesEEG")
                if series is None or "data" not in series or "timestamps" not in series:
                    file_errors.append(
                        {
                            "relative_path": relative,
                            "error": "missing_primary_eeg_series",
                        }
                    )
                    continue
                data = series["data"]
                timestamps = series["timestamps"]
                signal = _scan_timestamps(timestamps)
                signal.update(
                    {
                        "data_shape": [int(value) for value in data.shape],
                        "data_dtype": str(data.dtype),
                        "n_channels": (
                            int(data.shape[1]) if len(data.shape) > 1 else 1
                        ),
                        "path": "/acquisition/ElectricalSeriesEEG",
                    }
                )
                recovery_metadata = _native_recovery_metadata(handle)
                signal["native_recovery_metadata"] = recovery_metadata
                signal_cache[path] = signal
                timestamp_values = (
                    np.asarray(timestamps[:], dtype=float)
                    if signal["gap_count"]
                    else None
                )
                all_events = sorted(
                    all_events_by_file[relative],
                    key=lambda event: event["stim_on"],
                )
                for event in events:
                    event_records.append(
                        _event_record(
                            event=event,
                            signal=signal,
                            data=data,
                            timestamps=timestamps,
                            all_events=all_events,
                            native_recovery_available=bool(
                                recovery_metadata["found"]
                            ),
                            timestamp_values=timestamp_values,
                        )
                    )
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            file_errors.append(
                {
                    "relative_path": relative,
                    "error": f"{type(error).__name__}:{error}",
                }
            )

    nmd_timebase = _load_nmd_timebase(repo_root)
    nmd_windows, nmd_artifact_audit = _load_nmd_windows(nmd_timebase)
    nmd_timebase.update(nmd_artifact_audit)
    nmd_clock = _resolve_nmd_clock(nmd_windows, signal_cache)
    nmd_windows = nmd_clock["windows"]
    nmd_timebase["clock_audit"] = nmd_clock["audit"]
    if nmd_artifact_audit["serialized_artifact_count"]:
        nmd_timebase["status"] = (
            "NMD_TIMEBASE_METHOD_LIMITED"
            if not nmd_windows
            else "NMD_TIMEBASE_AUDITED"
        )
        nmd_timebase["reason"] = (
            "serialized_nmd_time_grids_found_but_family_alignment_requires_"
            "per_event_audit"
        )
    for record in event_records:
        record["nmd_alignment"] = _nmd_alignment(record, nmd_windows)

    by_family: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in event_records:
        by_family[(record["family_id"], record["w"])].append(record)
    family_ledger: list[dict[str, Any]] = []
    for key, far002_row in sorted(
        (
            key,
            frozen_rows[key],
        )
        for key in frozen_rows
    ):
        records = by_family.get(key, [])
        signal_sync = bool(records) and all(
            record["timestamp_valid"] for record in records
        )
        signal_continuity = bool(records) and all(
            not any(record["gap_crossing_by_horizon"].values())
            for record in records
        )
        artifact_counts: dict[str, int] = defaultdict(int)
        artifact_probe_counts: dict[str, int] = defaultdict(int)
        artifact_recovery_counts: dict[str, int] = defaultdict(int)
        for record in records:
            artifact_counts[record["artifact_status"]] += 1
            artifact_probe_counts[record["artifact_probe_status"]] += 1
            artifact_recovery_counts[record["artifact_recovery_status"]] += 1
        selected_horizon: float | None = None
        horizon_support: dict[str, Any] = {}
        for horizon in HORIZONS_SEC:
            eligible = [
                record
                for record in records
                if record["raw_signal_eligible_by_horizon"][str(horizon)]
            ]
            per_rho_subjects: dict[float, set[str]] = defaultdict(set)
            for record in eligible:
                per_rho_subjects[float(record["rho"])].add(record["subject"])
            surviving_rho = {
                rho: subjects
                for rho, subjects in per_rho_subjects.items()
                if len(subjects) >= MIN_SUBJECTS_PER_RHO
            }
            support_ok = len(surviving_rho) >= 3
            horizon_support[str(horizon)] = {
                "n_eligible_events": len(eligible),
                "n_subjects": len({record["subject"] for record in eligible}),
                "min_subjects_per_rho": (
                    min(map(len, surviving_rho.values()))
                    if surviving_rho
                    else 0
                ),
                "surviving_rho_levels": sorted(surviving_rho),
                "n_surviving_rho_levels": len(surviving_rho),
                "support_ok": support_ok,
            }
            if eligible and support_ok and selected_horizon is None:
                selected_horizon = horizon
        selected_records: list[dict[str, Any]] = []
        selected_surviving_rho: set[float] = set()
        if selected_horizon is not None:
            selected_summary = horizon_support[str(selected_horizon)]
            selected_surviving_rho = set(
                selected_summary["surviving_rho_levels"]
            )
            selected_records = [
                record
                for record in records
                if record["raw_signal_eligible_by_horizon"][
                    str(selected_horizon)
                ]
                and float(record["rho"]) in selected_surviving_rho
            ]
        nmd_resolution_ok = bool(
            selected_horizon is not None
            and nmd_timebase["configured_mnps_window_sec"] is not None
            and nmd_timebase["configured_mnps_window_sec"] <= selected_horizon
        )
        if artifact_counts.get(STIM_ARTIFACT_UNRESOLVED, 0):
            status = STIM_ARTIFACT_NOT_TESTABLE
            reason = "native_finite_or_rail_qc_unresolved"
        elif selected_horizon is None:
            status = INSUFFICIENT_COVERAGE
            reason = "no_frozen_horizon_meets_coverage_overlap_and_support"
        else:
            status = RAW_SIGNAL_ONLY_PASS
            reason = (
                "native_signal_pass; recovery_semantics_unresolved_and_existing_"
                "nmd_timebase_not_tested"
            )
        family_ledger.append(
            {
                "family_id": key[0],
                "w": key[1],
                "v": far002_row["perturbation_identity_v"],
                "n_rho_levels": far002_row["n_nonzero_levels"],
                "raw_signal_sync": signal_sync,
                "raw_signal_continuity": signal_continuity,
                "artifact_status": dict(artifact_counts),
                "artifact_probe_status": dict(artifact_probe_counts),
                "artifact_recovery_status": dict(artifact_recovery_counts),
                "min_clean_pre": selected_horizon,
                "min_clean_post": selected_horizon,
                "candidate_horizon": selected_horizon,
                "n_eligible_events": len(selected_records),
                "n_subjects": len(
                    {record["subject"] for record in selected_records}
                ),
                "n_surviving_rho_levels": (
                    horizon_support[str(selected_horizon)][
                        "n_surviving_rho_levels"
                    ]
                    if selected_horizon is not None
                    else 0
                ),
                "existing_nmd_alignment": bool(selected_records)
                and all(
                    record["nmd_alignment"]["aligned"]
                    for record in selected_records
                )
                and nmd_resolution_ok,
                "existing_nmd_resolution_ok": nmd_resolution_ok,
                "existing_nmd_pre_windows": sum(
                    record["nmd_alignment"]["pre_windows"]
                    for record in selected_records
                ),
                "existing_nmd_post_windows": sum(
                    record["nmd_alignment"]["post_windows"]
                    for record in selected_records
                ),
                "horizon_support": horizon_support,
                "status": status,
                "reason": reason,
            }
        )

    for row in family_ledger:
        if (
            row["status"] == RAW_SIGNAL_ONLY_PASS
            and row["existing_nmd_alignment"]
        ):
            row["status"] = SIGNAL_TIMEBASE_PASS
            row["reason"] = "native_signal_and_existing_nmd_timebase_pass"
    if any(row["status"] == SIGNAL_TIMEBASE_PASS for row in family_ledger):
        nmd_timebase["status"] = "NMD_TIMEBASE_PASS"
        nmd_timebase["reason"] = "pre_and_post_windows_align_for_a_full_family"

    by_v: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in family_ledger:
        by_v[
            json.dumps(row["v"], sort_keys=True, ensure_ascii=False)
        ].append(row)
    cross_state_matches: list[dict[str, Any]] = []
    for v_key, rows in by_v.items():
        for index, left in enumerate(rows):
            left_support = (
                left["horizon_support"][str(left["candidate_horizon"])][
                    "surviving_rho_levels"
                ]
                if left["candidate_horizon"] is not None
                else []
            )
            for right in rows[index + 1 :]:
                right_support = (
                    right["horizon_support"][str(right["candidate_horizon"])][
                        "surviving_rho_levels"
                    ]
                    if right["candidate_horizon"] is not None
                    else []
                )
                overlap = sorted(set(left_support) & set(right_support))
                if overlap and left["w"] != right["w"]:
                    cross_state_matches.append(
                        {
                            "v_key": v_key,
                            "left_family_id": left["family_id"],
                            "left_w": left["w"],
                            "right_family_id": right["family_id"],
                            "right_w": right["w"],
                            "overlapping_rho_levels": overlap,
                        }
                    )

    primary_candidates = [
        row
        for row in family_ledger
        if row["status"] in {RAW_SIGNAL_ONLY_PASS, SIGNAL_TIMEBASE_PASS}
        and row["n_surviving_rho_levels"] >= 3
    ]
    primary_candidates.sort(
        key=lambda row: (
            row["status"] != SIGNAL_TIMEBASE_PASS,
            -row["horizon_support"][str(row["candidate_horizon"])][
                "min_subjects_per_rho"
            ],
            -float(row["candidate_horizon"]),
            -row["n_surviving_rho_levels"],
            row["family_id"],
        )
    )
    raw_pass = any(
        row["status"] in {RAW_SIGNAL_ONLY_PASS, SIGNAL_TIMEBASE_PASS}
        for row in family_ledger
    )
    full_pass = any(row["status"] == SIGNAL_TIMEBASE_PASS for row in family_ledger)
    if full_pass:
        global_status = "PASS"
        global_reason = "full_signal_and_nmd_timebase_family_present"
    elif raw_pass:
        global_status = "METHOD_LIMITED"
        global_reason = "native_signal_pass_nmd_timebase_not_testable"
    elif family_ledger:
        global_status = "NOT_TESTABLE"
        global_reason = "no_native_signal_family_passed"
    else:
        global_status = "BLOCKED"
        global_reason = "no_frozen_candidate_events_found"

    return {
        "schema": "mndm.far_003a_signal_timebase.v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": sha256_file(protocol_path),
        "far002_certificate": {
            "path": str(far002_path),
            "sha256": sha256_file(far002_path),
        },
        "source_root": str(source_root),
        "source_binding": source_binding,
        "native_nwb_file_count": len(files),
        "native_nwb_files": _nwb_file_records(files, source_root),
        "frozen_curve_family_count": len(frozen_rows),
        "frozen_point_rows_excluded": frozen_point_rows,
        "primary_stream": "/acquisition/ElectricalSeriesEEG",
        "horizons_sec": list(HORIZONS_SEC),
        "artifact_probe_sec": ARTIFACT_PROBE_SEC,
        "artifact_policy": {
            "native_is_valid_required": True,
            "finite_required": True,
            "integer_dtype_rails_checked": True,
            "native_blanking_metadata_found": any(
                summary.get("native_recovery_metadata", {}).get("found", False)
                for summary in signal_cache.values()
            ),
            "blanking_policy": "strict_native_only",
        },
        "min_subjects_per_rho": MIN_SUBJECTS_PER_RHO,
        "event_records": event_records,
        "family_ledger": family_ledger,
        "cross_state_match_available": bool(cross_state_matches),
        "cross_state_matches": cross_state_matches,
        "primary_family_id": (
            primary_candidates[0]["family_id"] if primary_candidates else None
        ),
        "selection_hierarchy": [
            "CURVE_SEMANTICS_PASS",
            "native_signal_pass",
            "existing_nmd_timebase_pass",
            "at_least_three_surviving_rho_levels",
            "maximize_minimum_subjects_per_rho",
            "maximize_clean_post_horizon",
            "maximize_surviving_rho_levels",
            "family_id_tie_break",
        ],
        "signal_file_summaries": {
            str(path.relative_to(source_root)).replace("\\", "/"): summary
            for path, summary in signal_cache.items()
        },
        "nmd_timebase": nmd_timebase,
        "global_status": global_status,
        "global_reason": global_reason,
        "far_003b_authorized": full_pass,
        "file_errors": file_errors,
        "audit_scope": {
            "opened_native_paths": [
                "/intervals/trials",
                "/acquisition/ElectricalSeriesEEG/data (QC probe slices only)",
                "/acquisition/ElectricalSeriesEEG/timestamps",
            ],
            "not_opened": [
                "/units/spike_times",
                "waveforms",
                "/processing",
                "NMD outcomes",
            ],
        },
        "fail_closed_assertions": {
            "response_statistics_computed": False,
            "home_away_constructed": False,
            "rho_outcome_compared": False,
            "family_regrouped_after_signal_open": False,
            "nmd_windows_shortened": False,
            "held_out_opened": False,
            "artifact_blank_invented": False,
        },
        "claim_boundary": (
            "FAR-003A establishes only native signal synchronization, coverage, "
            "artifact-QC status, event overlap, and existing-NMD time-base "
            "eligibility. It does not establish home/away or resilience."
        ),
    }
