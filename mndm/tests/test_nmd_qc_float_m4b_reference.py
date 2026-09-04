from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.dynamical_families.nmd_qc_float import (  # noqa: E402
    TECHNICALLY_ADMISSIBLE,
    audit_exported_float,
)
from mndm.dynamical_families.nmd_qc_float_m4b_reference import (  # noqa: E402
    Mat4Variable,
    M4BReferenceError,
    build_reference_surface,
    evaluate_reference,
    canonical_array_hash,
    canonical_json_hash,
    read_mat4_variable,
    read_wfdb_header,
    validate_frozen_m4b_scope,
)
from mndm.dynamical_families.nmd_qc_float_m4b_compare import (  # noqa: E402
    canonicalize_production_qc,
    canonicalize_reference_qc,
    compare_canonical_qc,
)


def _write_mat4_int16(path: Path, value: np.ndarray) -> None:
    name = b"val\x00"
    header = struct.pack(
        "<5i",
        30,
        value.shape[0],
        value.shape[1],
        0,
        len(name),
    )
    path.write_bytes(header + name + value.astype("<i2").tobytes(order="F"))


def _write_header(path: Path, sample_count: int = 40) -> None:
    lines = [
        f"reference 2 20 {sample_count}",
        "reference.mat 16+24 18.9(14203)/nu 16 0 0 0 0 C3",
        "reference.mat 16+24 19.5(14020)/nu 16 0 0 0 0 C4",
        "#Start time: 19:00:00",
        "#End time: 19:00:01",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_minimal_mat4_reader_preserves_native_matrix_and_layout(tmp_path: Path) -> None:
    mat_path = tmp_path / "reference.mat"
    native = np.asarray(
        [
            [0, 1, -32768, 3],
            [10, 11, 12, 13],
        ],
        dtype=np.int16,
    )
    _write_mat4_int16(mat_path, native)
    variable = read_mat4_variable(mat_path)
    assert variable.name == "val"
    assert variable.value.dtype == np.dtype("<i2")
    assert variable.mopt == 30
    assert variable.byte_order == "<"
    assert variable.data_offset == 24
    np.testing.assert_array_equal(variable.value, native)


def test_reference_transform_maps_sentinel_before_calibration(tmp_path: Path) -> None:
    header_path = tmp_path / "reference.hea"
    _write_header(header_path, sample_count=4)
    header = read_wfdb_header(header_path)
    native = np.asarray(
        [
            [14203, -32768, 14222, 14241],
            [14020, 14039, 14059, 14078],
        ],
        dtype=np.int16,
    )
    reference = build_reference_surface(header, native)
    assert np.isnan(reference.surface[1, 0])
    assert reference.surface[0, 0] == 0.0
    assert reference.surface[2, 0] == (14222 - 14203) / 18.9
    assert reference.surface.dtype == np.dtype("float64")
    assert reference.surface.shape == (4, 2)


def test_independent_evaluator_matches_production_on_clean_synthetic_surface(
    tmp_path: Path,
) -> None:
    header_path = tmp_path / "reference.hea"
    _write_header(header_path)
    header = read_wfdb_header(header_path)
    native = np.vstack(
        [
            np.arange(40, dtype=np.int16),
            np.arange(40, dtype=np.int16) * 2,
        ]
    )
    reference = build_reference_surface(header, native)
    signal = reference.surface.T
    provenance = {
        "source": "synthetic-m4b-reference-fixture",
        "record_id": "reference",
    }
    independent = evaluate_reference(
        signal,
        reference.time,
        sampling_frequency=20.0,
        channel_labels=list(header.channel_names),
        required_channel_indices=[0, 1],
        channel_selection_provenance=provenance,
        segment_boundaries=[[0, 40]],
    )
    production = audit_exported_float(
        signal,
        reference.time,
        required_channel_indices=[0, 1],
        channel_labels=list(header.channel_names),
        channel_selection_provenance=provenance,
        sampling_frequency=20.0,
        segment_boundaries=[[0, 40]],
    )
    assert independent["recording_status"] == TECHNICALLY_ADMISSIBLE
    assert production["recording_status"] == TECHNICALLY_ADMISSIBLE
    assert all(
        flag["status"] == TECHNICALLY_ADMISSIBLE
        for flag in independent["rule_flags"]
    )
    comparison = compare_canonical_qc(
        canonicalize_reference_qc(independent),
        canonicalize_production_qc(
            production,
            expected_channel_indices=[0, 1],
            expected_channel_names=list(header.channel_names),
        ),
    )
    assert comparison["matches"] is True


def test_reference_reader_rejects_non_integer_native_matrix(tmp_path: Path) -> None:
    mat_path = tmp_path / "reference.mat"
    value = np.arange(8, dtype=np.float64).reshape(2, 4)
    name = b"val\x00"
    header = struct.pack("<5i", 0, 2, 4, 0, len(name))
    mat_path.write_bytes(header + name + value.astype("<f8").tobytes(order="F"))
    try:
        read_mat4_variable(mat_path)
    except M4BReferenceError:
        pass
    else:
        raise AssertionError("floating-point native matrix was accepted")


def test_frozen_scope_validation_accepts_the_frozen_layout(tmp_path: Path) -> None:
    header_path = tmp_path / "reference.hea"
    names = [
        "C3", "C4", "O1", "O2", "Cz", "F3", "F4", "F7", "F8", "Fz",
        "Fp1", "Fp2", "Fpz", "P3", "P4", "Pz", "T3", "T4", "T5", "T6",
    ]
    lines = [
        "0543_006_019_EEG 20 256 921600",
        *[
            f"reference.mat 16+24 18.9(14203)/nu 16 0 0 0 0 {name}"
            for name in names
        ],
    ]
    header_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    header = read_wfdb_header(header_path)
    variable = Mat4Variable(
        name="val",
        value=np.empty((20, 921600), dtype=np.dtype("<i2")),
        mopt=30,
        data_type_code=3,
        class_type=0,
        imagf=0,
        matrix_offset=0,
        data_offset=24,
        byte_order="<",
    )
    scope = {
        "reference": {
            "record_id": "0543_006_019_EEG",
            "channel_count": 20,
            "sample_count": 921600,
            "sampling_frequency_hz": 256.0,
            "channel_order": names,
            "native_header_format": "16+24",
            "native_base_format": "16",
            "native_units": "nu",
            "native_matrix_shape": [20, 921600],
        }
    }
    validate_frozen_m4b_scope(header, variable, scope)

    with pytest.raises(M4BReferenceError):
        validate_frozen_m4b_scope(
            header,
            replace(variable, data_offset=25),
            scope,
        )


def test_mat4_reader_rejects_non_frozen_layouts(tmp_path: Path) -> None:
    native = np.arange(8, dtype=np.int16).reshape(2, 4)
    big_endian = tmp_path / "big-endian.mat"
    big_endian.write_bytes(
        struct.pack(">5i", 1030, 2, 4, 0, 4)
        + b"val\x00"
        + native.astype(">i2").tobytes(order="F")
    )
    wrong_name = tmp_path / "wrong-name.mat"
    wrong_name.write_bytes(
        struct.pack("<5i", 30, 2, 4, 0, 4)
        + b"data"
        + native.astype("<i2").tobytes(order="F")
    )
    trailing = tmp_path / "trailing.mat"
    _write_mat4_int16(trailing, native)
    trailing.write_bytes(trailing.read_bytes() + b"\x00")
    truncated = tmp_path / "truncated.mat"
    truncated.write_bytes(b"\x1e\x00")
    for path in (big_endian, wrong_name, trailing, truncated):
        with pytest.raises(M4BReferenceError):
            read_mat4_variable(path)


def test_canonical_hashes_are_explicit_and_order_independent() -> None:
    little = np.asarray([1, 2, 3], dtype="<i2")
    big = np.asarray([1, 2, 3], dtype=">i2")
    matrix_c = np.arange(6, dtype=np.float64).reshape(2, 3, order="C")
    matrix_f = np.array(matrix_c, order="F")
    assert canonical_array_hash(little) == canonical_array_hash(big)
    assert canonical_array_hash(matrix_c) == canonical_array_hash(matrix_f)
    assert canonical_json_hash({"b": 2, "a": 1}) == canonical_json_hash(
        {"a": 1, "b": 2}
    )


def _evaluate_pair(
    signal: np.ndarray,
    time: np.ndarray | None = None,
    segment_boundaries: list[list[int]] | None = None,
    required_channel_indices: list[int] | None = None,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    sample_count = signal.shape[1]
    if time is None:
        time = np.linspace(0.0, (sample_count - 1) / 20.0, sample_count)
    if segment_boundaries is None:
        segment_boundaries = [[0, sample_count]]
    if required_channel_indices is None:
        required_channel_indices = [0, 1]
    labels = ["C3", "C4"]
    provenance = {"source": "synthetic-m4b-fixture"}
    independent = evaluate_reference(
        signal,
        time,
        sampling_frequency=20.0,
        channel_labels=labels,
        required_channel_indices=required_channel_indices,
        channel_selection_provenance=provenance,
        segment_boundaries=segment_boundaries,
        expected_channel_count=2,
        expected_sample_count=sample_count,
    )
    production = audit_exported_float(
        signal,
        time,
        required_channel_indices=required_channel_indices,
        channel_labels=labels,
        channel_selection_provenance=provenance,
        sampling_frequency=20.0,
        segment_boundaries=segment_boundaries,
    )
    comparison = compare_canonical_qc(
        canonicalize_reference_qc(independent),
        canonicalize_production_qc(
            production,
            expected_channel_indices=required_channel_indices,
            expected_channel_names=labels,
        ),
    )
    return independent, production, comparison


@pytest.mark.parametrize(
    ("defect", "expected_status"),
    [
        ("nonfinite", "TECHNICAL_INVALID"),
        ("flatline", "TECHNICAL_INVALID"),
        ("extremum_plateau", "TECHNICAL_INVALID"),
        ("quantization", "TECHNICAL_INVALID"),
        ("time_nonmonotonic", "TECHNICAL_INVALID"),
        ("uncovered", "TECHNICAL_STATUS_UNRESOLVED"),
    ],
)
def test_independent_evaluator_parity_on_static_expected_fixtures(
    defect: str,
    expected_status: str,
) -> None:
    signal = np.vstack(
        [
            np.arange(40, dtype=np.float64),
            np.arange(40, dtype=np.float64) * 2.0,
        ]
    )
    time = None
    segments = None
    if defect == "nonfinite":
        signal[0, 0] = np.nan
    elif defect == "flatline":
        signal[0, :20] = 1.0
    elif defect == "extremum_plateau":
        signal[0, :2] = 100.0
    elif defect == "quantization":
        signal[0] = np.tile(np.arange(8, dtype=np.float64), 5)
    elif defect == "time_nonmonotonic":
        time = np.linspace(0.0, 39.0 / 20.0, 40)
        time[10] = time[9]
    elif defect == "uncovered":
        segments = [[0, 20], [21, 40]]
    independent, production, comparison = _evaluate_pair(
        signal,
        time=time,
        segment_boundaries=segments,
    )
    assert independent["recording_status"] == expected_status
    assert production["recording_status"] == expected_status
    assert comparison["matches"] is True


def test_r8_out_of_range_selection_is_unresolved_in_both_paths() -> None:
    signal = np.vstack(
        [
            np.arange(40, dtype=np.float64),
            np.arange(40, dtype=np.float64) * 2.0,
        ]
    )
    independent, production, _ = _evaluate_pair(
        signal,
        required_channel_indices=[0, 2],
    )
    assert independent["recording_status"] == "TECHNICAL_STATUS_UNRESOLVED"
    assert production["recording_status"] == "TECHNICAL_STATUS_UNRESOLVED"
