"""Source-agnostic QC for exported floating-point EEG.

This module implements only the frozen NMD-QC-FLOAT-RULES-0.3 contract.  It
does not know about FAR, stimulation, ds003670, biological labels, or native
ADC rails.
"""

from __future__ import annotations

import hashlib
import json
from math import ceil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROTOCOL_ID = "NMD-QC-FLOAT-001"
CONTRACT_ID = "NMD-QC-FLOAT"
CONTRACT_VERSION = "0.3"
RULE_MANIFEST_PATH = "NMD-QC-FLOAT-RULES-0.3.json"
RULE_MANIFEST_SHA256 = (
    "054394322bad951a0a7e245be657fe6842ba4f3fc5807787c2e1e47f5592a6fa"
)

TECHNICALLY_ADMISSIBLE = "TECHNICALLY_ADMISSIBLE"
TECHNICAL_INVALID = "TECHNICAL_INVALID"
TECHNICAL_STATUS_UNRESOLVED = "TECHNICAL_STATUS_UNRESOLVED"

_SUPPORTED_DTYPES = {np.dtype("float32"), np.dtype("float64")}
_RULE_IDS = ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8")


def _is_sequence(value: object) -> bool:
    return isinstance(value, (list, tuple, np.ndarray))


def _array_hash(value: np.ndarray | None) -> str | None:
    if value is None:
        return None
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _status_from_reasons(
    *,
    invalid: Sequence[str],
    unresolved: Sequence[str],
) -> str:
    if invalid:
        return TECHNICAL_INVALID
    if unresolved:
        return TECHNICAL_STATUS_UNRESOLVED
    return TECHNICALLY_ADMISSIBLE


def _new_rule_statuses() -> dict[str, dict[str, Any]]:
    return {
        rule_id: {
            "status": TECHNICALLY_ADMISSIBLE,
            "reasons": [],
        }
        for rule_id in _RULE_IDS
    }


def _mark_rule(
    rules: dict[str, dict[str, Any]],
    rule_id: str,
    status: str,
    reason: str,
) -> None:
    current = rules[rule_id]
    current["reasons"].append(reason)
    if status == TECHNICAL_INVALID:
        current["status"] = TECHNICAL_INVALID
    elif (
        status == TECHNICAL_STATUS_UNRESOLVED
        and current["status"] != TECHNICAL_INVALID
    ):
        current["status"] = TECHNICAL_STATUS_UNRESOLVED


def _max_run_length(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in mask.tolist():
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _r4_window_status(
    values: np.ndarray,
    window_length: int,
) -> str:
    """Return R4 status using an equivalent sliding-window implementation."""
    if values.size < window_length:
        return TECHNICAL_STATUS_UNRESOLVED

    finite = np.isfinite(values)
    run_start = 0
    while run_start < values.size:
        while run_start < values.size and not finite[run_start]:
            run_start += 1
        run_stop = run_start
        while run_stop < values.size and finite[run_stop]:
            run_stop += 1
        if run_stop - run_start >= window_length:
            counts: dict[Any, int] = {}
            for index in range(run_start, run_start + window_length):
                value = values[index].item()
                counts[value] = counts.get(value, 0) + 1
            if len(counts) <= 8:
                return TECHNICAL_INVALID
            for end in range(run_start + window_length, run_stop):
                outgoing = values[end - window_length].item()
                counts[outgoing] -= 1
                if counts[outgoing] == 0:
                    del counts[outgoing]
                incoming = values[end].item()
                counts[incoming] = counts.get(incoming, 0) + 1
                if len(counts) <= 8:
                    return TECHNICAL_INVALID
        run_start = run_stop
    return TECHNICALLY_ADMISSIBLE


def _ulp(value: np.generic) -> float:
    dtype = value.dtype if isinstance(value, np.generic) else np.asarray(value).dtype
    typed_value = np.asarray(value, dtype=dtype)[()]
    next_value = np.nextafter(
        typed_value,
        np.asarray(np.inf, dtype=dtype)[()],
    )
    return abs(float(next_value) - float(typed_value))


def _valid_positive_scalar(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result) or result <= 0:
        return None
    return result


def _timebase_audit(
    time: np.ndarray | None,
    sampling_frequency: object,
    expected_samples: int,
) -> tuple[dict[str, Any], float | None]:
    if time is None:
        return (
            {
                "status": TECHNICAL_STATUS_UNRESOLVED,
                "reason": "TIME_MISSING",
                "dt_ref": None,
            },
            _valid_positive_scalar(sampling_frequency),
        )
    if time.ndim != 1 or time.size != expected_samples or time.size < 2:
        return (
            {
                "status": TECHNICAL_STATUS_UNRESOLVED,
                "reason": (
                    "TIME_LENGTH_MISMATCH"
                    if time.ndim != 1 or time.size != expected_samples
                    else "TIME_SUPPORT_INSUFFICIENT"
                ),
                "dt_ref": None,
            },
            _valid_positive_scalar(sampling_frequency),
        )
    if time.dtype not in _SUPPORTED_DTYPES:
        return (
            {
                "status": TECHNICAL_STATUS_UNRESOLVED,
                "reason": "TIME_DTYPE_UNSUPPORTED",
                "dt_ref": None,
            },
            _valid_positive_scalar(sampling_frequency),
        )
    if not np.isfinite(time).all():
        return (
            {
                "status": TECHNICAL_INVALID,
                "reason": "TIME_NONFINITE",
                "dt_ref": None,
            },
            _valid_positive_scalar(sampling_frequency),
        )

    dt = time[1:].astype(np.float64) - time[:-1].astype(np.float64)
    supplied_fs = _valid_positive_scalar(sampling_frequency)
    if sampling_frequency is not None and supplied_fs is None:
        return (
            {
                "status": TECHNICAL_STATUS_UNRESOLVED,
                "reason": "SAMPLING_FREQUENCY_INVALID",
                "dt_ref": None,
            },
            None,
        )
    if supplied_fs is not None:
        dt_ref = 1.0 / supplied_fs
        effective_fs = supplied_fs
    else:
        if not np.isfinite(dt).all() or np.any(dt <= 0):
            return (
                {
                    "status": TECHNICAL_INVALID,
                    "reason": "TIME_NONMONOTONIC_WITHOUT_DECLARED_FS",
                    "dt_ref": None,
                },
                None,
            )
        dt_ref = float(np.median(dt))
        effective_fs = 1.0 / dt_ref

    representability_limit = dt_ref / 4.0
    ulp_values = [
        _ulp(value)
        for value in np.concatenate((time[:-1], time[1:]))
    ]
    if any(value > representability_limit for value in ulp_values):
        return (
            {
                "status": TECHNICAL_STATUS_UNRESOLVED,
                "reason": "TIME_REPRESENTATION_INSUFFICIENT",
                "dt_ref": dt_ref,
                "effective_sampling_frequency": effective_fs,
                "max_ulp": max(ulp_values),
                "ulp_limit": representability_limit,
            },
            effective_fs,
        )

    if np.any(dt <= 0):
        return (
            {
                "status": TECHNICAL_INVALID,
                "reason": "TIME_NONMONOTONIC",
                "dt_ref": dt_ref,
                "effective_sampling_frequency": effective_fs,
            },
            effective_fs,
        )

    tolerances = np.maximum(
        np.maximum(
            np.full(dt.shape, 1e-9, dtype=np.float64),
            np.full(dt.shape, 1e-6 * dt_ref, dtype=np.float64),
        ),
        np.asarray(
            [
                max(8.0 * _ulp(left), 8.0 * _ulp(right))
                for left, right in zip(time[:-1], time[1:])
            ],
            dtype=np.float64,
        ),
    )
    if np.any(np.abs(dt - dt_ref) > tolerances):
        return (
            {
                "status": TECHNICAL_INVALID,
                "reason": "TIME_INTERVAL_OUT_OF_TOLERANCE",
                "dt_ref": dt_ref,
                "effective_sampling_frequency": effective_fs,
                "max_interval_error": float(np.max(np.abs(dt - dt_ref))),
            },
            effective_fs,
        )
    return (
        {
            "status": TECHNICALLY_ADMISSIBLE,
            "reason": "TIMEBASE_PASS",
            "dt_ref": dt_ref,
            "effective_sampling_frequency": effective_fs,
            "max_ulp": max(ulp_values),
            "ulp_limit": representability_limit,
        },
        effective_fs,
    )


def _selection_audit(
    *,
    n_channels: int,
    required_channel_indices: object,
    required_channel_names: object,
    channel_labels: object,
    excluded_channel_indices: object,
    channel_selection_provenance: object,
) -> tuple[dict[str, Any], list[int]]:
    invalid: list[str] = []
    unresolved: list[str] = []
    if (required_channel_indices is None) == (required_channel_names is None):
        unresolved.append("EXACTLY_ONE_REQUIRED_CHANNEL_DECLARATION_REQUIRED")

    labels: list[str] | None = None
    if channel_labels is not None:
        if (
            not _is_sequence(channel_labels)
            or isinstance(channel_labels, (str, bytes))
            or len(channel_labels) != n_channels
        ):
            unresolved.append("CHANNEL_LABELS_LENGTH_MISMATCH")
        else:
            labels = [str(value) for value in channel_labels]
            if len(set(labels)) != len(labels):
                unresolved.append("CHANNEL_LABELS_NOT_UNIQUE")

    selected: list[int] = []
    if required_channel_indices is not None:
        if (
            not _is_sequence(required_channel_indices)
            or isinstance(required_channel_indices, (str, bytes))
            or len(required_channel_indices) == 0
        ):
            unresolved.append("REQUIRED_CHANNEL_INDICES_INVALID")
        else:
            for value in required_channel_indices:
                if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                    unresolved.append("REQUIRED_CHANNEL_INDICES_NOT_INTEGERS")
                    continue
                candidate = int(value)
                if candidate < 0 or candidate >= n_channels:
                    unresolved.append("REQUIRED_CHANNEL_INDICES_OUT_OF_RANGE")
                    continue
                selected.append(candidate)
            if len(set(selected)) != len(selected):
                unresolved.append("REQUIRED_CHANNEL_INDICES_NOT_UNIQUE")
    elif required_channel_names is not None:
        if labels is None:
            unresolved.append("CHANNEL_LABELS_REQUIRED_FOR_NAME_SELECTION")
        if (
            not _is_sequence(required_channel_names)
            or isinstance(required_channel_names, (str, bytes))
            or len(required_channel_names) == 0
        ):
            unresolved.append("REQUIRED_CHANNEL_NAMES_INVALID")
        elif (
            labels is not None
            and "CHANNEL_LABELS_NOT_UNIQUE" not in unresolved
        ):
            label_to_index = {label: index for index, label in enumerate(labels)}
            for name in required_channel_names:
                if str(name) not in label_to_index:
                    unresolved.append("REQUIRED_CHANNEL_NAME_NOT_FOUND")
                else:
                    selected.append(label_to_index[str(name)])
            if len(set(selected)) != len(selected):
                unresolved.append("REQUIRED_CHANNEL_NAMES_NOT_UNIQUE")

    excluded: list[int] = []
    if excluded_channel_indices is not None:
        if (
            not _is_sequence(excluded_channel_indices)
            or isinstance(excluded_channel_indices, (str, bytes))
        ):
            unresolved.append("EXCLUDED_CHANNEL_INDICES_INVALID")
        else:
            for value in excluded_channel_indices:
                if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                    unresolved.append("EXCLUDED_CHANNEL_INDICES_NOT_INTEGERS")
                    continue
                candidate = int(value)
                if candidate < 0 or candidate >= n_channels:
                    unresolved.append("EXCLUDED_CHANNEL_INDICES_OUT_OF_RANGE")
                    continue
                excluded.append(candidate)
            if len(set(excluded)) != len(excluded):
                unresolved.append("EXCLUDED_CHANNEL_INDICES_NOT_UNIQUE")
            if set(selected).intersection(excluded):
                unresolved.append("REQUIRED_AND_EXCLUDED_CHANNEL_OVERLAP")

    if channel_selection_provenance is None or (
        isinstance(channel_selection_provenance, str)
        and not channel_selection_provenance.strip()
    ):
        unresolved.append("CHANNEL_SELECTION_PROVENANCE_MISSING")
    elif (
        (
            _is_sequence(channel_selection_provenance)
            and len(channel_selection_provenance) == 0
        )
        or (
            isinstance(channel_selection_provenance, Mapping)
            and not channel_selection_provenance
        )
    ):
        unresolved.append("CHANNEL_SELECTION_PROVENANCE_EMPTY")

    status = _status_from_reasons(invalid=invalid, unresolved=unresolved)
    return (
        {
            "status": status,
            "invalid_reasons": invalid,
            "unresolved_reasons": unresolved,
            "selected_channel_indices": selected,
            "excluded_channel_indices": excluded,
            "channel_labels": labels,
        },
        selected,
    )


def _segment_audit(
    n_samples: int,
    segment_boundaries: object,
) -> tuple[dict[str, Any], list[tuple[int, int]]]:
    if segment_boundaries is None:
        segments = [(0, n_samples)] if n_samples > 0 else []
    else:
        segments = []
        if (
            not _is_sequence(segment_boundaries)
            or isinstance(segment_boundaries, (str, bytes))
            or len(segment_boundaries) == 0
        ):
            return (
                {
                    "status": TECHNICAL_STATUS_UNRESOLVED,
                    "reason": "SEGMENT_BOUNDARIES_INVALID",
                    "segments": [],
                    "uncovered_sample_count": n_samples,
                },
                [],
            )
        previous_start = -1
        previous_stop = -1
        for boundary in segment_boundaries:
            if (
                not _is_sequence(boundary)
                or isinstance(boundary, (str, bytes))
                or len(boundary) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, np.integer))
                    for value in boundary
                )
            ):
                return (
                    {
                        "status": TECHNICAL_STATUS_UNRESOLVED,
                        "reason": "SEGMENT_BOUNDARIES_INVALID",
                        "segments": [],
                        "uncovered_sample_count": n_samples,
                    },
                    [],
                )
            start, stop = (int(value) for value in boundary)
            if (
                start < 0
                or start >= stop
                or stop > n_samples
                or start < previous_start
                or start < previous_stop
            ):
                return (
                    {
                        "status": TECHNICAL_STATUS_UNRESOLVED,
                        "reason": "SEGMENT_BOUNDARIES_INVALID",
                        "segments": [],
                        "uncovered_sample_count": n_samples,
                    },
                    [],
                )
            segments.append((start, stop))
            previous_start, previous_stop = start, stop

    uncovered = 0
    cursor = 0
    for start, stop in segments:
        uncovered += max(0, start - cursor)
        cursor = max(cursor, stop)
    uncovered += max(0, n_samples - cursor)
    status = TECHNICALLY_ADMISSIBLE if segments else TECHNICAL_STATUS_UNRESOLVED
    return (
        {
            "status": status,
            "reason": (
                "SEGMENTS_PASS"
                if uncovered == 0
                else "SEGMENTS_VALID_WITH_UNCOVERED_SUPPORT"
            )
            if status == TECHNICALLY_ADMISSIBLE
            else "SEGMENT_BOUNDARIES_INVALID",
            "segments": [
                {"start": start, "stop": stop} for start, stop in segments
            ],
            "uncovered_sample_count": uncovered,
        },
        segments,
    )


def _audit_channel_segment(
    values: np.ndarray,
    *,
    effective_fs: float | None,
) -> dict[str, Any]:
    rules = _new_rule_statuses()
    invalid: list[str] = []
    unresolved: list[str] = []
    if effective_fs is None:
        for rule_id in ("R2", "R3", "R4", "R6", "R7"):
            _mark_rule(
                rules,
                rule_id,
                TECHNICAL_STATUS_UNRESOLVED,
                "EFFECTIVE_SAMPLING_FREQUENCY_UNAVAILABLE",
            )
        unresolved.append("EFFECTIVE_SAMPLING_FREQUENCY_UNAVAILABLE")
        finite = np.isfinite(values)
        if not finite.all():
            _mark_rule(rules, "R1", TECHNICAL_INVALID, "NONFINITE_SAMPLES")
            invalid.append("NONFINITE_SAMPLES")
        return {
            "status": _status_from_reasons(invalid=invalid, unresolved=unresolved),
            "rule_statuses": rules,
            "invalid_reasons": invalid,
            "unresolved_reasons": unresolved,
            "finite_sample_count": int(finite.sum()),
        }

    support_samples = int(ceil(effective_fs))
    plateau_samples = int(ceil(0.1 * effective_fs))
    finite = np.isfinite(values)
    finite_count = int(finite.sum())
    if not finite.all():
        _mark_rule(rules, "R1", TECHNICAL_INVALID, "NONFINITE_SAMPLES")
        invalid.append("NONFINITE_SAMPLES")

    if finite_count == 0:
        _mark_rule(rules, "R6", TECHNICAL_INVALID, "ZERO_FINITE_SUPPORT")
        _mark_rule(rules, "R7", TECHNICAL_INVALID, "NONFINITE_ONLY_SUPPORT")
        for rule_id in ("R2", "R3", "R4"):
            _mark_rule(
                rules,
                rule_id,
                TECHNICAL_STATUS_UNRESOLVED,
                "NO_FINITE_SUPPORT",
            )
        invalid.extend(("ZERO_FINITE_SUPPORT", "NONFINITE_ONLY_SUPPORT"))
        unresolved.append("NO_FINITE_SUPPORT")
    else:
        finite_values = values[finite]
        if np.all(finite_values == finite_values[0]):
            if finite_count >= support_samples:
                _mark_rule(
                    rules,
                    "R7",
                    TECHNICAL_INVALID,
                    "SINGLE_VALUED_SUPPORT",
                )
                invalid.append("SINGLE_VALUED_SUPPORT")
            else:
                _mark_rule(
                    rules,
                    "R7",
                    TECHNICAL_STATUS_UNRESOLVED,
                    "FINITE_SUPPORT_BELOW_ONE_SECOND",
                )
                unresolved.append("FINITE_SUPPORT_BELOW_ONE_SECOND")

        equal_mask = np.zeros(values.shape, dtype=bool)
        equal_mask[1:] = (
            finite[1:]
            & finite[:-1]
            & (values[1:] == values[:-1])
        )
        if _max_run_length(equal_mask) + 1 >= support_samples:
            _mark_rule(rules, "R2", TECHNICAL_INVALID, "FLATLINE_RUN")
            invalid.append("FLATLINE_RUN")
        elif finite_count < support_samples:
            _mark_rule(
                rules,
                "R2",
                TECHNICAL_STATUS_UNRESOLVED,
                "FINITE_SUPPORT_BELOW_ONE_SECOND",
            )

        if values.size >= plateau_samples:
            maxima = np.max(finite_values)
            minima = np.min(finite_values)
            max_mask = finite & (values == maxima)
            min_mask = finite & (values == minima)
            if (
                _max_run_length(max_mask) >= plateau_samples
                or _max_run_length(min_mask) >= plateau_samples
            ):
                _mark_rule(
                    rules,
                    "R3",
                    TECHNICAL_INVALID,
                    "EXACT_EXTREMUM_PLATEAU",
                )
                invalid.append("EXACT_EXTREMUM_PLATEAU")
        if values.size < support_samples:
            _mark_rule(
                rules,
                "R4",
                TECHNICAL_STATUS_UNRESOLVED,
                "SEGMENT_SHORTER_THAN_ONE_SECOND",
            )
            unresolved.append("SEGMENT_SHORTER_THAN_ONE_SECOND")
        else:
            r4_status = _r4_window_status(values, support_samples)
            if r4_status == TECHNICAL_INVALID:
                _mark_rule(
                    rules,
                    "R4",
                    TECHNICAL_INVALID,
                    "QUANTIZATION_COLLAPSE",
                )
                invalid.append("QUANTIZATION_COLLAPSE")
            elif r4_status == TECHNICAL_STATUS_UNRESOLVED:
                _mark_rule(
                    rules,
                    "R4",
                    TECHNICAL_STATUS_UNRESOLVED,
                    "NO_COMPLETE_FINITE_WINDOW",
                )
                unresolved.append("NO_COMPLETE_FINITE_WINDOW")

    if finite_count < support_samples and finite_count > 0:
        for rule_id in ("R2", "R3", "R4", "R6", "R7"):
            _mark_rule(
                rules,
                rule_id,
                TECHNICAL_STATUS_UNRESOLVED,
                "FINITE_SUPPORT_BELOW_ONE_SECOND",
            )
        unresolved.append("FINITE_SUPPORT_BELOW_ONE_SECOND")

    return {
        "status": _status_from_reasons(invalid=invalid, unresolved=unresolved),
        "rule_statuses": rules,
        "invalid_reasons": sorted(set(invalid)),
        "unresolved_reasons": sorted(set(unresolved)),
        "finite_sample_count": finite_count,
    }


def audit_exported_float(
    signal: object,
    time: object,
    *,
    required_channel_indices: object = None,
    required_channel_names: object = None,
    channel_labels: object = None,
    channel_selection_provenance: object = None,
    sampling_frequency: object = None,
    excluded_channel_indices: object = None,
    segment_boundaries: object = None,
    input_provenance: Mapping[str, Any] | None = None,
    implementation_identity: str = "mndm.nmd_qc_float",
    runtime_identity: object = None,
    fixture_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Audit one exported floating-point EEG object without biological labels."""

    try:
        signal_array = np.asarray(signal)
    except (TypeError, ValueError):
        signal_array = np.asarray([], dtype=np.float64)
    try:
        time_array = None if time is None else np.asarray(time)
    except (TypeError, ValueError):
        time_array = None

    signal_shape = list(signal_array.shape)
    n_channels = int(signal_array.shape[0]) if signal_array.ndim >= 1 else 0
    n_samples = (
        int(signal_array.shape[1])
        if signal_array.ndim == 2
        else 0
    )
    base: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "contract_id": CONTRACT_ID,
        "contract_version": CONTRACT_VERSION,
        "rule_manifest_path": RULE_MANIFEST_PATH,
        "rule_manifest_sha256": RULE_MANIFEST_SHA256,
        "input_shape": signal_shape,
        "input_dtype": str(signal_array.dtype),
        "time_dtype": None if time_array is None else str(time_array.dtype),
        "sampling_frequency": _json_safe(sampling_frequency),
        "input_provenance": _json_safe(input_provenance or {}),
        "implementation_identity": implementation_identity,
        "runtime_identity": _json_safe(runtime_identity or {}),
        "fixture_manifest_sha256": fixture_manifest_sha256,
        "input_hash": _array_hash(signal_array),
        "time_hash": _array_hash(time_array),
        "channel_flags": [],
        "segment_flags": [],
        "unresolved_reasons": [],
        "invalid_reasons": [],
        "timebase_status": {
            "status": TECHNICAL_STATUS_UNRESOLVED,
            "reason": "TIME_NOT_AUDITED",
        },
        "effective_sampling_frequency": None,
        "required_channel_indices": [],
        "required_channel_names": [],
        "excluded_channel_indices": [],
        "excluded_channel_names": [],
        "channel_selection_provenance": None,
        "segment_boundaries": [],
        "evaluated_support": {
            "segment_count": 0,
            "sample_count": 0,
        },
        "uncovered_support": {
            "sample_count": 0,
            "status": TECHNICAL_STATUS_UNRESOLVED,
        },
    }

    if signal_array.ndim != 2 or n_channels == 0 or n_samples == 0:
        base["recording_status"] = TECHNICAL_STATUS_UNRESOLVED
        base["unresolved_reasons"] = ["SIGNAL_SHAPE_OR_SUPPORT_INVALID"]
        base["determinism_hash"] = _canonical_hash(base)
        return base
    if signal_array.dtype not in _SUPPORTED_DTYPES:
        base["recording_status"] = TECHNICAL_STATUS_UNRESOLVED
        base["unresolved_reasons"] = ["SIGNAL_DTYPE_UNSUPPORTED"]
        base["determinism_hash"] = _canonical_hash(base)
        return base

    timebase, effective_fs = _timebase_audit(
        time_array,
        sampling_frequency,
        n_samples,
    )
    base["timebase_status"] = timebase
    base["effective_sampling_frequency"] = effective_fs

    selection, selected = _selection_audit(
        n_channels=n_channels,
        required_channel_indices=required_channel_indices,
        required_channel_names=required_channel_names,
        channel_labels=channel_labels,
        excluded_channel_indices=excluded_channel_indices,
        channel_selection_provenance=channel_selection_provenance,
    )
    base["required_channel_indices"] = selection["selected_channel_indices"]
    base["required_channel_names"] = (
        [
            selection["channel_labels"][index]
            for index in selected
        ]
        if selection["channel_labels"] is not None
        else []
    )
    base["excluded_channel_indices"] = selection["excluded_channel_indices"]
    base["excluded_channel_names"] = (
        [
            selection["channel_labels"][index]
            for index in selection["excluded_channel_indices"]
        ]
        if selection["channel_labels"] is not None
        else []
    )
    base["channel_selection_provenance"] = _json_safe(
        channel_selection_provenance
    )
    if not selected:
        invalid_reasons: list[str] = []
        unresolved_reasons = list(selection["unresolved_reasons"])
        if timebase["status"] == TECHNICAL_INVALID:
            invalid_reasons.append(f"R5:{timebase['reason']}")
        elif timebase["status"] == TECHNICAL_STATUS_UNRESOLVED:
            unresolved_reasons.append(f"R5:{timebase['reason']}")
        base["invalid_reasons"] = sorted(set(invalid_reasons))
        base["unresolved_reasons"] = sorted(set(unresolved_reasons))
        base["recording_status"] = _status_from_reasons(
            invalid=base["invalid_reasons"],
            unresolved=base["unresolved_reasons"],
        )
        base["determinism_hash"] = _canonical_hash(base)
        return base

    segment_audit, segments = _segment_audit(n_samples, segment_boundaries)
    base["segment_boundaries"] = segment_audit["segments"]
    base["uncovered_support"] = {
        "sample_count": segment_audit["uncovered_sample_count"],
        "status": (
            TECHNICAL_STATUS_UNRESOLVED
            if segment_audit["uncovered_sample_count"]
            else TECHNICALLY_ADMISSIBLE
        ),
    }
    if not segments or segment_audit["status"] != TECHNICALLY_ADMISSIBLE:
        invalid_reasons = []
        unresolved_reasons = list(selection["unresolved_reasons"])
        unresolved_reasons.append(segment_audit["reason"])
        base["uncovered_support"] = {
            "sample_count": n_samples,
            "status": TECHNICAL_STATUS_UNRESOLVED,
        }
        for channel_index in selected:
            if not np.isfinite(signal_array[channel_index]).all():
                invalid_reasons.append("R1:NONFINITE_SAMPLES")
        if timebase["status"] == TECHNICAL_INVALID:
            invalid_reasons.append(f"R5:{timebase['reason']}")
        elif timebase["status"] == TECHNICAL_STATUS_UNRESOLVED:
            unresolved_reasons.append(f"R5:{timebase['reason']}")
        base["invalid_reasons"] = sorted(set(invalid_reasons))
        base["unresolved_reasons"] = sorted(set(unresolved_reasons))
        base["recording_status"] = _status_from_reasons(
            invalid=base["invalid_reasons"],
            unresolved=base["unresolved_reasons"],
        )
        base["determinism_hash"] = _canonical_hash(base)
        return base

    channel_invalid: list[str] = []
    channel_unresolved: list[str] = []
    invalid_reasons: list[str] = []
    unresolved_reasons: list[str] = []
    for channel_index in selected:
        channel_segment_flags: list[dict[str, Any]] = []
        channel_invalid_reasons: list[str] = []
        channel_unresolved_reasons: list[str] = []
        for segment_index, (start, stop) in enumerate(segments):
            segment_result = _audit_channel_segment(
                signal_array[channel_index, start:stop],
                effective_fs=effective_fs,
            )
            segment_flag = {
                "channel_index": channel_index,
                "segment_index": segment_index,
                "start": start,
                "stop": stop,
                **segment_result,
            }
            channel_segment_flags.append(segment_flag)
            channel_invalid_reasons.extend(segment_result["invalid_reasons"])
            channel_unresolved_reasons.extend(
                segment_result["unresolved_reasons"]
            )
        channel_status = _status_from_reasons(
            invalid=channel_invalid_reasons,
            unresolved=channel_unresolved_reasons,
        )
        channel_flag = {
            "channel_index": channel_index,
            "channel_name": (
                selection["channel_labels"][channel_index]
                if selection["channel_labels"] is not None
                else None
            ),
            "status": channel_status,
            "segment_flags": channel_segment_flags,
            "invalid_reasons": sorted(set(channel_invalid_reasons)),
            "unresolved_reasons": sorted(set(channel_unresolved_reasons)),
        }
        base["channel_flags"].append(channel_flag)
        base["segment_flags"].extend(channel_segment_flags)
        if channel_status == TECHNICAL_INVALID:
            channel_invalid.append(f"CHANNEL_{channel_index}")
            invalid_reasons.extend(channel_invalid_reasons)
        elif channel_status == TECHNICAL_STATUS_UNRESOLVED:
            channel_unresolved.append(f"CHANNEL_{channel_index}")
            unresolved_reasons.extend(channel_unresolved_reasons)

    invalid_reasons.extend(channel_invalid)
    unresolved_reasons.extend(channel_unresolved)
    unresolved_reasons.extend(selection["unresolved_reasons"])
    if segment_audit["uncovered_sample_count"]:
        unresolved_reasons.append("UNCOVERED_SUPPORT")
    if timebase["status"] == TECHNICAL_INVALID:
        invalid_reasons.append(f"R5:{timebase['reason']}")
    elif timebase["status"] == TECHNICAL_STATUS_UNRESOLVED:
        unresolved_reasons.append(f"R5:{timebase['reason']}")
    base["invalid_reasons"] = sorted(set(invalid_reasons))
    base["unresolved_reasons"] = sorted(set(unresolved_reasons))
    base["recording_status"] = _status_from_reasons(
        invalid=base["invalid_reasons"],
        unresolved=base["unresolved_reasons"],
    )
    base["evaluated_support"] = {
        "segment_count": len(segments),
        "sample_count": sum(stop - start for start, stop in segments),
    }
    base["determinism_hash"] = _canonical_hash(base)
    return base
