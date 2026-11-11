import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "project/new_mnps_v.3.2/science_lead/sl039_anphy_source_audit.py"
SPEC = importlib.util.spec_from_file_location("sl039_anphy_source_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def _put(field: bytearray, start: int, width: int, value: str) -> None:
    encoded = value.encode("ascii")
    assert len(encoded) <= width
    field[start : start + width] = encoded.ljust(width, b" ")


def _write_tiny_edf(path: Path, *, mode: str = "EDF+C", n_records: int = 1) -> None:
    fixed = bytearray(b" " * 256)
    _put(fixed, 0, 8, "0")
    _put(fixed, 168, 8, "01.01.24")
    _put(fixed, 176, 8, "00.00.00")
    _put(fixed, 184, 8, "512")
    _put(fixed, 192, 44, mode)
    _put(fixed, 236, 8, str(n_records))
    _put(fixed, 244, 8, "1")
    _put(fixed, 252, 4, "1")

    signal = bytearray(b" " * 256)
    widths = (16, 80, 8, 8, 8, 8, 8, 80, 8, 32)
    values = ("EEG Fp1", "transducer", "uV", "-100", "100", "-32768", "32767", "none", "4", "")
    offset = 0
    for width, value in zip(widths, values):
        _put(signal, offset, width, value)
        offset += width
    path.write_bytes(fixed + signal + b"\x00" * (2 * 4 * n_records))


def test_edf_header_checks_mode_record_extent_and_sample_grid(tmp_path):
    path = tmp_path / "tiny.edf"
    _write_tiny_edf(path, mode="EDF+C", n_records=2)

    header = audit.parse_edf_header(path)

    assert header["recording_mode"] == "EDF+C"
    assert header["data_record_count"] == 2
    assert header["nominal_duration_s"] == 2
    assert header["unique_non_annotation_sampling_frequencies_hz"] == [4]
    assert header["exact_declared_file_size"] is True
    assert header["header_sha256"]

    path.write_bytes(path.read_bytes()[:-1])
    assert audit.parse_edf_header(path)["exact_declared_file_size"] is False


def test_edf_header_rejects_short_or_malformed_header(tmp_path):
    path = tmp_path / "short.edf"
    path.write_bytes(b"short")
    with pytest.raises(ValueError, match="short EDF fixed header"):
        audit.parse_edf_header(path)


def test_edf_discontinuous_mode_is_not_promoted_to_continuous_support(tmp_path):
    subject = tmp_path / "EPCTL02"
    subject.mkdir()
    _write_tiny_edf(subject / "recording.edf", mode="EDF+D")
    (subject / "stages.txt").write_text("W\t0\t30\n", encoding="utf-8")

    result = audit.run_audit(tmp_path)

    assert result["records"][0]["nominal_recording_support"]["status"] == "declared_discontinuous_requires_record_onset_audit"


def test_txt_rows_keep_unmapped_codes_and_reject_bad_timing(tmp_path):
    path = tmp_path / "stages.txt"
    path.write_text("\ufeffW\t0\t30\nL\t30\t30\nN2\t60\tnan\nR 90 30\n", encoding="utf-8")

    events, errors = audit.parse_stage_text(path)

    assert [event["code"] for event in events] == ["W", "L", "R"]
    assert events[0]["stage_name_if_known"] == "Wake"
    assert events[1]["stage_name_if_known"] is None
    assert len(errors) == 1
    assert errors[0]["reason"] == "invalid_timing"


def test_timeline_reports_initial_gap_internal_gap_and_overlap():
    events = [
        {"event_index": 0, "line": 1, "onset_s": 30.0, "duration_s": 30.0, "end_s": 60.0},
        {"event_index": 1, "line": 2, "onset_s": 75.0, "duration_s": 30.0, "end_s": 105.0},
        {"event_index": 2, "line": 3, "onset_s": 100.0, "duration_s": 30.0, "end_s": 130.0},
    ]

    summary = audit.summarize_timeline(events)

    assert summary["first_annotation_onset_s"] == 30.0
    assert summary["gap_count"] == 2
    assert summary["overlap_count"] == 1
    assert summary["gaps"] == [
        {"start_s": 0.0, "end_s": 30.0, "before_event_index": 0},
        {"start_s": 60.0, "end_s": 75.0, "before_event_index": 1},
    ]


def test_centered_window_nominal_support_requires_annotation_extent_and_grid():
    event = {"onset_s": 0.0, "duration_s": 30.0, "end_s": 30.0}
    targets = audit.centered_target_support(event, recording_duration_s=24.0, sampling_frequency_hz=4.0)

    assert targets[0]["target_start_s"] == 5.0 and targets[0]["target_end_s"] == 25.0
    assert targets[0]["nominal_header_support"] is False
    assert targets[0]["reasons"] == ["outside_edf_header_extent"]
    assert targets[1]["source_start_sample"] == 44
    assert targets[1]["source_end_sample_exclusive"] == 76
    assert targets[1]["nominal_header_support"] is True


def test_audit_marks_semantic_and_signal_content_uncertainties(tmp_path):
    subject = tmp_path / "EPCTL01"
    subject.mkdir()
    _write_tiny_edf(subject / "recording.edf")
    (subject / "stages.txt").write_text("W\t0\t30\nL\t30\t30\n", encoding="utf-8")

    result = audit.run_audit(tmp_path)
    record = result["records"][0]

    assert "metadata" in result["scope"] and "source-support" in result["scope"]
    assert result["qualified_for_sleep_benchmark"] is False
    assert record["nominal_recording_support"]["status"] == "declared_continuous_records"
    assert record["annotation_text"]["unmapped_codes"] == ["L"]
    assert [event["code"] for event in record["nominal_recording_support"]["annotation_events_outside_edf_header_extent"]] == ["W", "L"]
    assert any("absolute EDF timestamp" in text for text in result["uncertainties"])
    assert any("signal contents" in text for text in result["uncertainties"])
    assert "signal payload was not read or hashed" in record["edf"]["source_hash_scope"]
    assert "qualification" in audit.render_summary(result).lower()
