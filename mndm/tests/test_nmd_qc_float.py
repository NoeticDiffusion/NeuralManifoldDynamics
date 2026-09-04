from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.dynamical_families.nmd_qc_float import (  # noqa: E402
    TECHNICALLY_ADMISSIBLE,
    TECHNICAL_INVALID,
    TECHNICAL_STATUS_UNRESOLVED,
    audit_exported_float,
)


def _fixture(
    *,
    n_samples: int = 500,
    sampling_frequency: float = 250.0,
    signal_dtype: np.dtype = np.dtype("float64"),
    time_dtype: np.dtype = np.dtype("float64"),
    time_origin: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    time = (
        np.arange(n_samples, dtype=time_dtype) / sampling_frequency
        + np.asarray(time_origin, dtype=time_dtype)
    )
    base = np.sin(2.0 * np.pi * 10.0 * time.astype(np.float64))
    second = 0.4 * np.cos(2.0 * np.pi * 6.0 * time.astype(np.float64))
    signal = np.vstack((base, second)).astype(signal_dtype)
    return signal, time


def _audit(signal: np.ndarray, time: np.ndarray, **kwargs: object) -> dict:
    return audit_exported_float(
        signal,
        time,
        required_channel_indices=[0, 1],
        channel_selection_provenance={"source": "synthetic"},
        sampling_frequency=kwargs.pop("sampling_frequency", 250.0),
        **kwargs,
    )


def test_clean_float_fixture_is_admissible() -> None:
    signal, time = _fixture()
    result = audit_exported_float(
        signal,
        time,
        required_channel_names=["C1", "C2"],
        channel_labels=["C1", "C2"],
        channel_selection_provenance={"source": "synthetic"},
        sampling_frequency=250.0,
    )
    assert result["recording_status"] == TECHNICALLY_ADMISSIBLE
    assert result["timebase_status"]["status"] == TECHNICALLY_ADMISSIBLE
    assert all(
        flag["status"] == TECHNICALLY_ADMISSIBLE
        for flag in result["channel_flags"]
    )


def test_nonfinite_and_flatline_are_invalid() -> None:
    signal, time = _fixture()
    signal[0, 0] = np.nan
    result = _audit(signal, time)
    assert result["recording_status"] == TECHNICAL_INVALID
    assert "NONFINITE_SAMPLES" in result["invalid_reasons"]

    signal, time = _fixture()
    signal[0, :250] = 0.25
    result = _audit(signal, time)
    assert result["recording_status"] == TECHNICAL_INVALID
    assert "FLATLINE_RUN" in result["invalid_reasons"]


def test_extremum_plateau_is_scored_below_one_second() -> None:
    signal, time = _fixture(n_samples=100)
    signal[0, :30] = 5.0
    signal[0, 30:60] = -5.0
    result = _audit(signal, time)
    channel = result["channel_flags"][0]
    assert result["recording_status"] == TECHNICAL_INVALID
    assert "EXACT_EXTREMUM_PLATEAU" in channel["invalid_reasons"]


def test_gross_quantization_is_invalid() -> None:
    signal, time = _fixture()
    levels = (np.arange(signal.shape[1]) % 8) / 8.0
    signal[:] = np.vstack((levels, levels))
    result = _audit(signal, time)
    assert result["recording_status"] == TECHNICAL_INVALID
    assert "QUANTIZATION_COLLAPSE" in result["invalid_reasons"]


def test_timestamp_representation_insufficiency_precedes_monotonicity() -> None:
    signal, time = _fixture(
        n_samples=4000,
        sampling_frequency=2000.0,
        signal_dtype=np.dtype("float32"),
        time_dtype=np.dtype("float32"),
        time_origin=100000.0,
    )
    result = _audit(
        signal,
        time,
        sampling_frequency=2000.0,
    )
    assert result["recording_status"] == TECHNICAL_STATUS_UNRESOLVED
    assert result["timebase_status"]["reason"] == (
        "TIME_REPRESENTATION_INSUFFICIENT"
    )


def test_adequate_duplicate_timestamp_is_invalid() -> None:
    signal, time = _fixture()
    time[250] = time[249]
    result = _audit(signal, time)
    assert result["recording_status"] == TECHNICAL_INVALID
    assert "R5:TIME_NONMONOTONIC" in result["invalid_reasons"]


def test_time_length_mismatch_and_invalid_fs_are_unresolved() -> None:
    signal, time = _fixture()
    result = _audit(signal, time[:-1])
    assert result["recording_status"] == TECHNICAL_STATUS_UNRESOLVED
    assert result["unresolved_reasons"] == ["R5:TIME_LENGTH_MISMATCH"]

    result = _audit(signal, time, sampling_frequency=np.nan)
    assert result["recording_status"] == TECHNICAL_STATUS_UNRESOLVED
    assert "R5:SAMPLING_FREQUENCY_INVALID" in result["unresolved_reasons"]


def test_names_require_labels_and_unsupported_dtype_is_unresolved() -> None:
    signal, time = _fixture()
    result = audit_exported_float(
        signal,
        time,
        required_channel_names=["C1", "C2"],
        channel_selection_provenance={"source": "synthetic"},
        sampling_frequency=250.0,
    )
    assert result["recording_status"] == TECHNICAL_STATUS_UNRESOLVED
    assert "CHANNEL_LABELS_REQUIRED_FOR_NAME_SELECTION" in result[
        "unresolved_reasons"
    ]

    result = audit_exported_float(
        signal.astype(np.int16),
        time,
        required_channel_indices=[0, 1],
        channel_selection_provenance={"source": "synthetic"},
        sampling_frequency=250.0,
    )
    assert result["recording_status"] == TECHNICAL_STATUS_UNRESOLVED
    assert result["unresolved_reasons"] == ["SIGNAL_DTYPE_UNSUPPORTED"]


def test_invalid_signal_outranks_missing_provenance() -> None:
    signal, time = _fixture()
    signal[0, 0] = np.nan
    result = audit_exported_float(
        signal,
        time,
        required_channel_indices=[0, 1],
        channel_selection_provenance=[],
        sampling_frequency=250.0,
    )
    assert result["recording_status"] == TECHNICAL_INVALID
    assert "NONFINITE_SAMPLES" in result["invalid_reasons"]
    assert "CHANNEL_SELECTION_PROVENANCE_EMPTY" in result["unresolved_reasons"]


def test_out_of_range_indices_with_labels_fail_closed_without_crashing() -> None:
    signal, time = _fixture()
    result = audit_exported_float(
        signal,
        time,
        required_channel_indices=[0, 99],
        channel_labels=["C1", "C2"],
        channel_selection_provenance={"source": "synthetic"},
        sampling_frequency=250.0,
    )
    assert result["recording_status"] == TECHNICAL_STATUS_UNRESOLVED
    assert result["required_channel_indices"] == [0]
    assert "REQUIRED_CHANNEL_INDICES_OUT_OF_RANGE" in result[
        "unresolved_reasons"
    ]


def test_segments_are_half_open_and_uncovered_support_is_unresolved() -> None:
    signal, time = _fixture(n_samples=600)
    result = _audit(
        signal,
        time,
        segment_boundaries=[(0, 250), (300, 550)],
    )
    assert result["recording_status"] == TECHNICAL_STATUS_UNRESOLVED
    assert result["uncovered_support"]["sample_count"] == 100
    assert result["unresolved_reasons"] == ["UNCOVERED_SUPPORT"]
    assert all(
        flag["status"] == TECHNICALLY_ADMISSIBLE
        for flag in result["channel_flags"]
    )

    result = _audit(
        signal,
        time,
        segment_boundaries=[(0, 250), (200, 400)],
    )
    assert result["recording_status"] == TECHNICAL_STATUS_UNRESOLVED
    assert result["unresolved_reasons"] == ["SEGMENT_BOUNDARIES_INVALID"]


def test_invalid_time_outranks_invalid_segment_boundaries() -> None:
    signal, time = _fixture()
    time[250] = time[249]
    result = _audit(
        signal,
        time,
        segment_boundaries=[(0, 250), (200, 400)],
    )
    assert result["recording_status"] == TECHNICAL_INVALID
    assert "R5:TIME_NONMONOTONIC" in result["invalid_reasons"]
    assert "SEGMENT_BOUNDARIES_INVALID" in result["unresolved_reasons"]


def test_result_is_sample_free_and_deterministic() -> None:
    signal, time = _fixture(signal_dtype=np.dtype("float32"))
    first = _audit(signal, time)
    second = _audit(signal, time)
    assert first == second
    assert "signal" not in first
    assert "time" not in first
    assert first["input_hash"]
    assert first["determinism_hash"]
