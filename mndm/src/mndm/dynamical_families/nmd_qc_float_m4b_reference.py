"""Independent M4B WFDB/MAT-v4 reference path.

This module intentionally has no imports from ``wfdb``, the production
NMD-QC-FLOAT implementation, or production preprocessing.  It is a small
reference implementation for the separately frozen M4B comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import ceil
from pathlib import Path
import re
import struct
from typing import Any, Mapping

import numpy as np


TECHNICALLY_ADMISSIBLE = "TECHNICALLY_ADMISSIBLE"
TECHNICAL_INVALID = "TECHNICAL_INVALID"
TECHNICAL_STATUS_UNRESOLVED = "TECHNICAL_STATUS_UNRESOLVED"
RULE_IDS = ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8")
SENTINEL_FORMAT_16 = -32768

_GAIN_PATTERN = re.compile(
    r"(?P<gain>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(?:\((?P<baseline>[-+]?\d+(?:\.\d*)?)\))?/"
    r"(?P<units>\S+)"
)
_MAT4_DTYPES = {
    0: np.dtype("f8"),
    1: np.dtype("f4"),
    2: np.dtype("i4"),
    3: np.dtype("i2"),
    4: np.dtype("u2"),
    5: np.dtype("u1"),
}


class M4BReferenceError(RuntimeError):
    """Raised when the frozen M4B reference prerequisites are not met."""


@dataclass(frozen=True)
class WfdbSignalSpec:
    filename: str
    fmt: str
    base_format: str
    byte_offset: int | None
    gain: float
    baseline: float
    units: str
    channel_name: str


@dataclass(frozen=True)
class WfdbHeader:
    record_name: str
    channel_count: int
    sampling_frequency: float
    sample_count: int
    signals: tuple[WfdbSignalSpec, ...]
    start_time: str | None
    end_time: str | None

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(signal.channel_name for signal in self.signals)

    @property
    def gains(self) -> np.ndarray:
        return np.asarray([signal.gain for signal in self.signals], dtype=np.float64)

    @property
    def baselines(self) -> np.ndarray:
        return np.asarray(
            [signal.baseline for signal in self.signals],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class Mat4Variable:
    name: str
    value: np.ndarray
    mopt: int
    data_type_code: int
    class_type: int
    imagf: int
    matrix_offset: int
    data_offset: int
    byte_order: str


@dataclass(frozen=True)
class ReferenceSurface:
    native: np.ndarray
    missing_mask: np.ndarray
    surface: np.ndarray
    time: np.ndarray


def _status_from_reasons(
    *,
    invalid: list[str],
    unresolved: list[str],
) -> str:
    if invalid:
        return TECHNICAL_INVALID
    if unresolved:
        return TECHNICAL_STATUS_UNRESOLVED
    return TECHNICALLY_ADMISSIBLE


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


def _new_rule_statuses() -> dict[str, dict[str, Any]]:
    return {
        rule_id: {
            "status": TECHNICALLY_ADMISSIBLE,
            "reasons": [],
        }
        for rule_id in RULE_IDS
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


def canonical_array_hash(array: np.ndarray) -> str:
    """Hash explicit dtype, shape, little-endian bytes, and C-order bytes."""
    value = np.asarray(array)
    dtype = value.dtype
    canonical_dtype = dtype if dtype.itemsize == 1 else dtype.newbyteorder("<")
    contiguous = np.ascontiguousarray(value.astype(canonical_dtype, copy=False))
    descriptor = json.dumps(
        {
            "dtype": canonical_dtype.str,
            "order": "C",
            "shape": list(value.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(descriptor + b"\0" + contiguous.tobytes(order="C")).hexdigest()


def canonical_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_wfdb_signal_line(line: str) -> WfdbSignalSpec:
    tokens = line.split()
    if len(tokens) < 9:
        raise M4BReferenceError("WFDB signal line has fewer than nine fields")
    fmt = tokens[1]
    format_parts = fmt.split("+", 1)
    base_format = format_parts[0]
    byte_offset: int | None = None
    if len(format_parts) == 2:
        try:
            byte_offset = int(format_parts[1])
        except ValueError as exc:
            raise M4BReferenceError("WFDB byte offset is not an integer") from exc
    gain_match = _GAIN_PATTERN.fullmatch(tokens[2])
    if gain_match is None or gain_match.group("baseline") is None:
        raise M4BReferenceError("WFDB gain/baseline field is not fully specified")
    gain = float(gain_match.group("gain"))
    baseline = float(gain_match.group("baseline"))
    if not np.isfinite(gain) or gain == 0 or not np.isfinite(baseline):
        raise M4BReferenceError("WFDB gain/baseline is non-finite or invalid")
    return WfdbSignalSpec(
        filename=tokens[0],
        fmt=fmt,
        base_format=base_format,
        byte_offset=byte_offset,
        gain=gain,
        baseline=baseline,
        units=gain_match.group("units"),
        channel_name=tokens[-1],
    )


def read_wfdb_header(path: Path) -> WfdbHeader:
    """Read only the fixed WFDB text header."""
    text = path.read_text(encoding="utf-8")
    data_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not data_lines:
        raise M4BReferenceError("WFDB header is empty")
    record_tokens = data_lines[0].split()
    if len(record_tokens) < 4:
        raise M4BReferenceError("WFDB record line is incomplete")
    try:
        channel_count = int(record_tokens[1])
        sampling_frequency = float(record_tokens[2])
        sample_count = int(record_tokens[3])
    except ValueError as exc:
        raise M4BReferenceError("WFDB record fields are invalid") from exc
    signals = tuple(_parse_wfdb_signal_line(line) for line in data_lines[1:])
    if len(signals) != channel_count:
        raise M4BReferenceError("WFDB channel count does not match signal lines")
    comments = [line.strip() for line in text.splitlines() if line.lstrip().startswith("#")]
    start_time = next(
        (
            line.split(":", 1)[1].strip()
            for line in comments
            if line.lower().startswith("#start time:")
        ),
        None,
    )
    end_time = next(
        (
            line.split(":", 1)[1].strip()
            for line in comments
            if line.lower().startswith("#end time:")
        ),
        None,
    )
    return WfdbHeader(
        record_name=record_tokens[0],
        channel_count=channel_count,
        sampling_frequency=sampling_frequency,
        sample_count=sample_count,
        signals=signals,
        start_time=start_time,
        end_time=end_time,
    )


def _decode_mat4_header(raw_header: bytes) -> tuple[str, tuple[int, int, int, int, int]]:
    candidates: list[tuple[str, tuple[int, int, int, int, int]]] = []
    for byte_order in ("<", ">"):
        values = struct.unpack(f"{byte_order}5i", raw_header)
        mopt, rows, cols, imagf, name_length = values
        machine = mopt // 1000
        data_type = (mopt % 100) // 10
        class_type = mopt % 10
        if byte_order == "<" and machine != 0:
            continue
        if byte_order == ">" and machine != 1:
            continue
        if (
            rows < 0
            or cols < 0
            or name_length <= 0
            or name_length > 1_000_000
            or data_type not in _MAT4_DTYPES
            or class_type not in {0, 1, 2}
        ):
            continue
        candidates.append((byte_order, values))
    if len(candidates) != 1:
        raise M4BReferenceError("MAT v4 header byte order or MOPT is unresolved")
    return candidates[0]


def read_mat4_variable(path: Path, variable_name: str = "val") -> Mat4Variable:
    """Read the frozen first MAT v4 ``val`` matrix without a generic reader."""
    with path.open("rb") as handle:
        first_matrix = True
        while True:
            matrix_offset = handle.tell()
            raw_header = handle.read(20)
            if not raw_header:
                break
            if len(raw_header) != 20:
                raise M4BReferenceError("MAT v4 matrix header is truncated")
            byte_order, values = _decode_mat4_header(raw_header)
            mopt, rows, cols, imagf, name_length = values
            raw_name = handle.read(name_length)
            if len(raw_name) != name_length:
                raise M4BReferenceError("MAT v4 variable name is truncated")
            name = raw_name.rstrip(b"\x00").decode("ascii", errors="strict")
            data_type_code = (mopt % 100) // 10
            class_type = mopt % 10
            data_offset = handle.tell()
            if (
                not first_matrix
                or name != variable_name
                or name_length != len(variable_name) + 1
                or raw_name != f"{variable_name}\x00".encode("ascii")
                or matrix_offset != 0
                or data_offset != 20 + name_length
                or byte_order != "<"
                or mopt != 30
                or data_type_code != 3
                or class_type != 0
                or imagf != 0
            ):
                raise M4BReferenceError(
                    "MAT v4 file does not match the frozen WFDB2MAT layout"
                )
            dtype = _MAT4_DTYPES[data_type_code]
            data_bytes = rows * cols * dtype.itemsize
            raw_data = handle.read(data_bytes)
            if len(raw_data) != data_bytes:
                raise M4BReferenceError("MAT v4 matrix data is truncated")
            typed_dtype = dtype if dtype.itemsize == 1 else dtype.newbyteorder(byte_order)
            value = np.frombuffer(raw_data, dtype=typed_dtype).copy()
            value = value.reshape((rows, cols), order="F")
            if handle.read(1):
                raise M4BReferenceError("MAT v4 file contains trailing matrix data")
            return Mat4Variable(
                name=name,
                value=value,
                mopt=mopt,
                data_type_code=data_type_code,
                class_type=class_type,
                imagf=imagf,
                matrix_offset=matrix_offset,
                data_offset=data_offset,
                byte_order=byte_order,
            )
    raise M4BReferenceError(f"MAT v4 variable {variable_name!r} was not found")


def build_reference_surface(
    header: WfdbHeader,
    native: np.ndarray,
    *,
    sentinel: int = SENTINEL_FORMAT_16,
) -> ReferenceSurface:
    """Apply the frozen independent sentinel and gain/baseline transform."""
    if native.ndim != 2:
        raise M4BReferenceError("native matrix is not two-dimensional")
    expected_shape = (header.channel_count, header.sample_count)
    if native.shape != expected_shape:
        raise M4BReferenceError(
            f"native matrix shape {native.shape} != frozen {expected_shape}"
        )
    if native.dtype.kind != "i" or native.dtype.itemsize != 2:
        raise M4BReferenceError("native matrix is not signed 16-bit integer data")
    for signal in header.signals:
        if signal.base_format != "16" or signal.byte_offset != 24:
            raise M4BReferenceError("native WFDB format is not the frozen 16+24 layout")
        if signal.units != "nu":
            raise M4BReferenceError("native WFDB units are not the frozen nu units")
    if len(header.gains) != native.shape[0] or len(header.baselines) != native.shape[0]:
        raise M4BReferenceError("header calibration count does not match native data")
    missing_mask = native == sentinel
    physical = np.full(native.shape, np.nan, dtype=np.float64)
    valid = ~missing_mask
    digital = native.astype(np.float64, copy=False)
    for channel_index in range(native.shape[0]):
        channel_valid = valid[channel_index]
        physical[channel_index, channel_valid] = (
            digital[channel_index, channel_valid] - header.baselines[channel_index]
        ) / header.gains[channel_index]
    surface = physical.T.copy()
    time = np.linspace(
        0.0,
        (header.sample_count - 1) / header.sampling_frequency,
        header.sample_count,
        dtype=np.float64,
    )
    return ReferenceSurface(
        native=native,
        missing_mask=missing_mask,
        surface=surface,
        time=time,
    )


def validate_frozen_m4b_scope(
    header: WfdbHeader,
    variable: Mat4Variable,
    scope: Mapping[str, Any],
) -> None:
    """Reject any header/MAT layout that differs from the frozen scope."""
    reference = scope["reference"]
    if header.record_name != reference["record_id"]:
        raise M4BReferenceError("record name differs from frozen M4B scope")
    if header.channel_count != int(reference["channel_count"]):
        raise M4BReferenceError("channel count differs from frozen M4B scope")
    if header.sample_count != int(reference["sample_count"]):
        raise M4BReferenceError("sample count differs from frozen M4B scope")
    if header.sampling_frequency != float(reference["sampling_frequency_hz"]):
        raise M4BReferenceError("sampling frequency differs from frozen M4B scope")
    if list(header.channel_names) != list(reference["channel_order"]):
        raise M4BReferenceError("channel order differs from frozen M4B scope")
    if any(
        signal.fmt != reference["native_header_format"]
        or signal.base_format != reference["native_base_format"]
        or signal.byte_offset != 24
        or signal.units != reference["native_units"]
        for signal in header.signals
    ):
        raise M4BReferenceError("WFDB signal semantics differ from frozen scope")
    expected_shape = tuple(int(value) for value in reference["native_matrix_shape"])
    if variable.name != "val" or variable.value.shape != expected_shape:
        raise M4BReferenceError("MAT v4 variable shape/name differs from frozen scope")
    if (
        variable.mopt != 30
        or variable.data_type_code != 3
        or variable.class_type != 0
        or variable.imagf != 0
        or variable.matrix_offset != 0
        or variable.data_offset != 24
        or variable.byte_order != "<"
    ):
        raise M4BReferenceError("MAT v4 binary layout differs from frozen scope")


def _timebase_audit(
    time: np.ndarray,
    sampling_frequency: float,
    expected_samples: int,
) -> tuple[dict[str, Any], float | None]:
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
    if time.dtype not in {np.dtype("float32"), np.dtype("float64")}:
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
    effective_fs = _valid_positive_scalar(sampling_frequency)
    if effective_fs is None:
        return (
            {
                "status": TECHNICAL_STATUS_UNRESOLVED,
                "reason": "SAMPLING_FREQUENCY_INVALID",
                "dt_ref": None,
            },
            None,
        )
    dt_ref = 1.0 / effective_fs
    ulp_values = [
        _ulp(value)
        for value in np.concatenate((time[:-1], time[1:]))
    ]
    if any(value > dt_ref / 4.0 for value in ulp_values):
        return (
            {
                "status": TECHNICAL_STATUS_UNRESOLVED,
                "reason": "TIME_REPRESENTATION_INSUFFICIENT",
                "dt_ref": dt_ref,
                "effective_sampling_frequency": effective_fs,
                "max_ulp": max(ulp_values),
                "ulp_limit": dt_ref / 4.0,
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
            "ulp_limit": dt_ref / 4.0,
        },
        effective_fs,
    )


def _selection_audit(
    *,
    n_channels: int,
    required_channel_indices: list[int],
    channel_labels: list[str],
    channel_selection_provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], list[int]]:
    unresolved: list[str] = []
    selected = list(required_channel_indices)
    if len(selected) == 0 or len(set(selected)) != len(selected):
        unresolved.append("REQUIRED_CHANNEL_INDICES_NOT_UNIQUE")
    if any(index < 0 or index >= n_channels for index in selected):
        unresolved.append("REQUIRED_CHANNEL_INDICES_OUT_OF_RANGE")
    if len(channel_labels) != n_channels:
        unresolved.append("CHANNEL_LABELS_LENGTH_MISMATCH")
    if len(set(channel_labels)) != len(channel_labels):
        unresolved.append("CHANNEL_LABELS_NOT_UNIQUE")
    if (
        not isinstance(channel_selection_provenance, Mapping)
        or not channel_selection_provenance
    ):
        unresolved.append("CHANNEL_SELECTION_PROVENANCE_EMPTY")
    return (
        {
            "status": _status_from_reasons(invalid=[], unresolved=unresolved),
            "unresolved_reasons": unresolved,
            "selected_channel_indices": selected,
            "channel_labels": channel_labels,
        },
        selected if not unresolved else [],
    )


def _r4_window_status(values: np.ndarray, window_length: int) -> str:
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
            _mark_rule(rules, rule_id, TECHNICAL_STATUS_UNRESOLVED, "NO_FINITE_SUPPORT")
        invalid.extend(("ZERO_FINITE_SUPPORT", "NONFINITE_ONLY_SUPPORT"))
        unresolved.append("NO_FINITE_SUPPORT")
    else:
        finite_values = values[finite]
        if np.all(finite_values == finite_values[0]):
            if finite_count >= support_samples:
                _mark_rule(rules, "R7", TECHNICAL_INVALID, "SINGLE_VALUED_SUPPORT")
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
                _mark_rule(rules, "R4", TECHNICAL_INVALID, "QUANTIZATION_COLLAPSE")
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


def _segment_audit(
    n_samples: int,
    segment_boundaries: list[list[int]],
) -> tuple[dict[str, Any], list[tuple[int, int]]]:
    segments: list[tuple[int, int]] = []
    previous_start = -1
    previous_stop = -1
    for boundary in segment_boundaries:
        if (
            len(boundary) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in boundary)
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
        start, stop = boundary
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
    return (
        {
            "status": TECHNICALLY_ADMISSIBLE if segments else TECHNICAL_STATUS_UNRESOLVED,
            "reason": (
                "SEGMENTS_PASS"
                if uncovered == 0
                else "SEGMENTS_VALID_WITH_UNCOVERED_SUPPORT"
            ),
            "segments": [{"start": start, "stop": stop} for start, stop in segments],
            "uncovered_sample_count": uncovered,
        },
        segments,
    )


def evaluate_reference(
    signal: np.ndarray,
    time: np.ndarray,
    *,
    sampling_frequency: float,
    channel_labels: list[str],
    required_channel_indices: list[int],
    channel_selection_provenance: Mapping[str, Any],
    segment_boundaries: list[list[int]],
    expected_channel_count: int | None = None,
    expected_sample_count: int | None = None,
) -> dict[str, Any]:
    """Evaluate R1–R8 independently on a reference float surface."""
    expected_shape = (
        (expected_channel_count, expected_sample_count)
        if expected_channel_count is not None and expected_sample_count is not None
        else None
    )
    if (
        signal.ndim != 2
        or signal.dtype not in {np.dtype("float32"), np.dtype("float64")}
        or signal.shape[0] == 0
        or signal.shape[1] == 0
        or expected_shape is not None and signal.shape != expected_shape
    ):
        return {
            "recording_status": TECHNICAL_STATUS_UNRESOLVED,
            "invalid_reasons": [],
            "unresolved_reasons": [
                (
                    "SIGNAL_DTYPE_UNSUPPORTED"
                    if signal.ndim == 2
                    and signal.dtype not in {np.dtype("float32"), np.dtype("float64")}
                    else "SIGNAL_SHAPE_OR_SUPPORT_INVALID"
                    if expected_shape is not None and signal.shape != expected_shape
                    else "SIGNAL_SHAPE_OR_SUPPORT_INVALID"
                )
            ],
            "channel_flags": [],
            "segment_flags": [],
            "rule_flags": [],
            "evaluated_support": {"segment_count": 0, "sample_count": 0},
            "uncovered_support": {
                "sample_count": 0,
                "status": TECHNICAL_STATUS_UNRESOLVED,
            },
        }
    timebase, effective_fs = _timebase_audit(
        time,
        sampling_frequency,
        signal.shape[1],
    )
    selection, selected = _selection_audit(
        n_channels=signal.shape[0],
        required_channel_indices=required_channel_indices,
        channel_labels=channel_labels,
        channel_selection_provenance=channel_selection_provenance,
    )
    segment_audit, segments = _segment_audit(
        signal.shape[1],
        segment_boundaries,
    )
    if not selected or not segments:
        invalid_reasons: list[str] = []
        unresolved_reasons = list(selection["unresolved_reasons"])
        unresolved_reasons.append(segment_audit["reason"])
        if timebase["status"] == TECHNICAL_INVALID:
            invalid_reasons.append(f"R5:{timebase['reason']}")
        elif timebase["status"] == TECHNICAL_STATUS_UNRESOLVED:
            unresolved_reasons.append(f"R5:{timebase['reason']}")
        return {
            "recording_status": _status_from_reasons(
                invalid=invalid_reasons,
                unresolved=unresolved_reasons,
            ),
            "invalid_reasons": sorted(set(invalid_reasons)),
            "unresolved_reasons": sorted(set(unresolved_reasons)),
            "channel_flags": [],
            "segment_flags": [],
            "rule_flags": [],
            "evaluated_support": {"segment_count": 0, "sample_count": 0},
            "uncovered_support": {
                "sample_count": signal.shape[1],
                "status": TECHNICAL_STATUS_UNRESOLVED,
            },
        }

    channel_flags: list[dict[str, Any]] = []
    segment_flags: list[dict[str, Any]] = []
    rule_flags: list[dict[str, Any]] = []
    recording_invalid: list[str] = []
    recording_unresolved: list[str] = []
    for channel_index in selected:
        channel_segments: list[dict[str, Any]] = []
        channel_invalid: list[str] = []
        channel_unresolved: list[str] = []
        for segment_index, (start, stop) in enumerate(segments):
            result = _audit_channel_segment(
                signal[channel_index, start:stop],
                effective_fs=effective_fs,
            )
            segment_flag = {
                "channel_index": channel_index,
                "segment_index": segment_index,
                "start": start,
                "stop": stop,
                **result,
            }
            channel_segments.append(segment_flag)
            segment_flags.append(segment_flag)
            channel_invalid.extend(result["invalid_reasons"])
            channel_unresolved.extend(result["unresolved_reasons"])
            for rule_id, rule in result["rule_statuses"].items():
                if rule_id in {"R5", "R8"}:
                    continue
                rule_flags.append(
                    {
                        "channel_index": channel_index,
                        "segment_index": segment_index,
                        "rule_id": rule_id,
                        "status": rule["status"],
                        "trigger_identity": sorted(
                            {f"{rule_id}:{reason}" for reason in rule["reasons"]}
                        ),
                    }
                )
        channel_status = _status_from_reasons(
            invalid=channel_invalid,
            unresolved=channel_unresolved,
        )
        channel_flags.append(
            {
                "channel_index": channel_index,
                "channel_name": channel_labels[channel_index],
                "status": channel_status,
                "invalid_reason_ids": sorted(set(channel_invalid)),
                "unresolved_reason_ids": sorted(set(channel_unresolved)),
                "finite_sample_count": sum(
                    segment["finite_sample_count"] for segment in channel_segments
                ),
            }
        )
        if channel_status == TECHNICAL_INVALID:
            recording_invalid.append(f"CHANNEL_{channel_index}")
        elif channel_status == TECHNICAL_STATUS_UNRESOLVED:
            recording_unresolved.append(f"CHANNEL_{channel_index}")

    uncovered_count = int(segment_audit["uncovered_sample_count"])
    if uncovered_count:
        recording_unresolved.append("UNCOVERED_SUPPORT")
    if timebase["status"] == TECHNICAL_INVALID:
        recording_invalid.append(f"R5:{timebase['reason']}")
    elif timebase["status"] == TECHNICAL_STATUS_UNRESOLVED:
        recording_unresolved.append(f"R5:{timebase['reason']}")
    recording_unresolved.extend(selection["unresolved_reasons"])
    rule_flags.append(
        {
            "channel_index": -1,
            "segment_index": -1,
            "rule_id": "R5",
            "status": timebase["status"],
            "trigger_identity": [f"R5:{timebase['reason']}"],
        }
    )
    rule_flags.append(
        {
            "channel_index": -1,
            "segment_index": -1,
            "rule_id": "R8",
            "status": selection["status"],
            "trigger_identity": sorted(
                {f"R8:{reason}" for reason in selection["unresolved_reasons"]}
            ),
        }
    )
    return {
        "recording_status": _status_from_reasons(
            invalid=recording_invalid,
            unresolved=recording_unresolved,
        ),
        "invalid_reasons": sorted(set(recording_invalid)),
        "unresolved_reasons": sorted(set(recording_unresolved)),
        "channel_flags": channel_flags,
        "segment_flags": segment_flags,
        "rule_flags": rule_flags,
        "evaluated_support": {
            "segment_count": len(segments),
            "sample_count": sum(stop - start for start, stop in segments),
        },
        "uncovered_support": {
            "sample_count": uncovered_count,
            "status": (
                TECHNICAL_STATUS_UNRESOLVED
                if uncovered_count
                else TECHNICALLY_ADMISSIBLE
            ),
        },
    }


def reference_hashes(
    reference: ReferenceSurface,
    independent_qc: Mapping[str, Any],
) -> dict[str, str]:
    """Return provenance hashes with explicit array/JSON serialization."""
    return {
        "native_matrix_hash": canonical_array_hash(reference.native),
        "missing_mask_hash": canonical_array_hash(reference.missing_mask),
        "reference_surface_hash": canonical_array_hash(reference.surface),
        "reference_time_hash": canonical_array_hash(reference.time),
        "reference_qc_summary_hash": canonical_json_hash(independent_qc),
    }
