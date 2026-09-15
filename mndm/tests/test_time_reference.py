"""Tests for time-reference extraction and alignment."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.pipeline.time_reference import (
    build_time_reference_for_run,
    parse_wfdb_clock_value,
    parse_wfdb_header_clocks,
)
from mndm.pipeline import time_reference as time_reference_module


def test_parse_wfdb_clock_value_supports_rollover_suffix():
    """Parse WFDB clocks with rollover suffixes."""
    assert parse_wfdb_clock_value("22:59:08") == pytest.approx(82748.0)
    assert parse_wfdb_clock_value("24:00:00+") == pytest.approx(86400.0)
    assert parse_wfdb_clock_value("00:10:00+") == pytest.approx(87000.0)


def test_parse_wfdb_clock_value_accepts_icare_elapsed_hours():
    """3+ digit hour fields are I-CARE elapsed-recording-hours, not malformed clocks.

    Regression guard for ingest_jacobian_fidelity_handover_2.md item 4: WFDB
    headers whose hour field tracks hours-since-ROSC (e.g. ``100:00:00``)
    must parse to elapsed seconds instead of being rejected as
    ``invalid_start_time``.
    """
    assert parse_wfdb_clock_value("100:00:00") == pytest.approx(360000.0)
    assert parse_wfdb_clock_value("101:00:00") == pytest.approx(363600.0)
    assert parse_wfdb_clock_value("206:00:00") == pytest.approx(741600.0)
    # Still rejects genuinely malformed input.
    assert parse_wfdb_clock_value("not_a_clock") is None
    assert parse_wfdb_clock_value("12:99:00") is None


def test_parse_wfdb_clock_value_rejects_pathological_hour_digit_counts():
    """A 6-digit-plus hour field must fail explicitly, not raise OverflowError.

    Regression for an independent-review finding on the elapsed-hours widen:
    the hour group must stay bounded (not truly unbounded ``\\d+``) so a
    corrupt/malformed header field with e.g. 20 digits cannot produce an
    arbitrary-precision int whose float() conversion raises OverflowError.
    """
    assert parse_wfdb_clock_value("1000000:00:00") is None  # 7 digits, over the cap
    assert parse_wfdb_clock_value("9" * 50 + ":00:00") is None  # pathological input
    # The cap boundary itself (6 digits) still parses normally.
    assert parse_wfdb_clock_value("999999:00:00") == pytest.approx(999999.0 * 3600.0)


def test_unwrap_subject_timeline_bounds_ambiguous_convention_mismatch():
    """A magnitude mismatch across rows must not silently inject an unbounded offset.

    Regression for an independent-review finding: if a >=100h elapsed-hours
    row precedes (by run/acq token order) a later row whose raw clock value
    is small, the day-rollover repair loop must not keep adding day offsets
    indefinitely to force ordering -- it must cap at
    ``_MAX_AUTO_DAY_ROLLOVERS`` and flag the row as ambiguous instead of
    inventing an implausible multi-day timeline.
    """
    entries = [
        {
            "path": "sub/run-001.hea",
            "run": "001",
            "acq": "001",
            "run_start_clock_sec": 363600.0,  # "101:00:00" elapsed-hours
            "run_end_clock_sec": 367200.0,
        },
        {
            "path": "sub/run-002.hea",
            "run": "002",
            "acq": "001",
            "run_start_clock_sec": 600.0,  # "00:10:00" -- looks like a later civil clock
            "run_end_clock_sec": 1200.0,
        },
    ]
    timeline, anchor, ambiguous = time_reference_module._unwrap_subject_timeline(entries)
    assert "sub/run-002.hea" in ambiguous
    assert ambiguous["sub/run-002.hea"] == "ambiguous_clock_convention_unresolved_ordering"
    # The offset is bounded, not unbounded: at most _MAX_AUTO_DAY_ROLLOVERS
    # day-rollovers (3 days = 259200s) were applied on top of the raw value.
    max_bound = 600.0 + time_reference_module._MAX_AUTO_DAY_ROLLOVERS * time_reference_module._SECONDS_PER_DAY
    assert timeline["sub/run-002.hea"][0] <= max_bound + 1e-6
    # First row is unaffected and not flagged ambiguous.
    assert "sub/run-001.hea" not in ambiguous
    assert timeline["sub/run-001.hea"][0] == pytest.approx(363600.0)


def test_unwrap_subject_timeline_normal_midnight_wrap_not_flagged_ambiguous():
    """A genuine, small civil-clock midnight wrap must still resolve cleanly."""
    entries = [
        {
            "path": "sub/run-001.hea",
            "run": "001",
            "acq": "001",
            "run_start_clock_sec": 82748.0,  # "22:59:08"
            "run_end_clock_sec": 82808.0,
        },
        {
            "path": "sub/run-002.hea",
            "run": "002",
            "acq": "001",
            "run_start_clock_sec": 600.0,  # "00:10:00" the next calendar day
            "run_end_clock_sec": 1200.0,
        },
    ]
    timeline, anchor, ambiguous = time_reference_module._unwrap_subject_timeline(entries)
    assert ambiguous == {}
    assert timeline["sub/run-002.hea"][0] == pytest.approx(600.0 + time_reference_module._SECONDS_PER_DAY)


def test_parse_wfdb_header_clocks_parses_start_end(tmp_path: Path):
    """Parse WFDB header start/end time lines."""
    hea = tmp_path / "rec.hea"
    hea.write_text(
        "\n".join(
            [
                "rec 1 200 1000",
                "#Start time: 22:59:08",
                "#End time: 23:10:08",
            ]
        ),
        encoding="utf-8",
    )
    parsed = parse_wfdb_header_clocks(hea)
    assert parsed["status"] == "ok"
    assert parsed["run_start_clock_sec"] == pytest.approx(82748.0)
    assert parsed["run_end_clock_sec"] == pytest.approx(83408.0)
    assert parsed["run_duration_sec"] == pytest.approx(660.0)


def test_parse_wfdb_header_clocks_reports_missing_end(tmp_path: Path):
    """Header parser reports missing end clock."""
    hea = tmp_path / "rec_missing_end.hea"
    hea.write_text(
        "\n".join(
            [
                "rec 1 200 1000",
                "#Start time: 08:00:00",
            ]
        ),
        encoding="utf-8",
    )
    parsed = parse_wfdb_header_clocks(hea)
    assert parsed["status"] == "ok_with_warnings"
    assert "missing_end_time" in parsed["parse_errors"]
    assert parsed["run_start_clock_sec"] == pytest.approx(28800.0)
    assert parsed["run_end_clock_sec"] is None


def test_parse_wfdb_header_clocks_accepts_icare_elapsed_hour_header(tmp_path: Path):
    """A 3-digit elapsed-hour header parses to status=ok, not invalid_start_time."""
    hea = tmp_path / "0332_081_100_EEG.hea"
    hea.write_text(
        "\n".join(
            [
                "0332_081_100_EEG 1 200 1000",
                "#Start time: 100:00:00",
                "#End time: 101:00:00",
            ]
        ),
        encoding="utf-8",
    )
    parsed = parse_wfdb_header_clocks(hea)
    assert parsed["status"] == "ok"
    assert parsed["parse_errors"] == []
    assert parsed["run_start_clock_sec"] == pytest.approx(360000.0)
    assert parsed["run_duration_sec"] == pytest.approx(3600.0)


def test_build_time_reference_for_run_aligns_windows_to_first_recording(tmp_path: Path):
    """Anchor run windows to subject first recording clock."""
    dataset_root = tmp_path / "icare"
    sub_dir = dataset_root / "0332"
    sub_dir.mkdir(parents=True)

    hea_first = sub_dir / "0332_001_001_EEG.hea"
    hea_second = sub_dir / "0332_002_001_EEG.hea"
    hea_first.write_text(
        "\n".join(
            [
                "0332_001_001_EEG 1 200 1000",
                "#Start time: 23:50:00",
                "#End time: 23:59:00",
            ]
        ),
        encoding="utf-8",
    )
    hea_second.write_text(
        "\n".join(
            [
                "0332_002_001_EEG 1 200 1000",
                "#Start time: 00:10:00+",
                "#End time: 00:20:00+",
            ]
        ),
        encoding="utf-8",
    )

    index_df = pd.DataFrame(
        [
            {
                "path": "0332/0332_001_001_EEG.hea",
                "subject": "0332",
                "run": "001",
                "acq": "001",
                "modality": "eeg",
            },
            {
                "path": "0332/0332_002_001_EEG.hea",
                "subject": "0332",
                "run": "002",
                "acq": "001",
                "modality": "eeg",
            },
        ]
    )
    sub_frame = pd.DataFrame(
        {
            "file": ["0332_002_001_EEG.hea", "0332_002_001_EEG.hea"],
            "t_start": [0.0, 2.0],
            "t_end": [2.0, 4.0],
        }
    )
    config = {
        "time_reference": {
            "enabled": True,
            "schema_version": "time_reference.v1",
            "parser": "wfdb_header",
            "anchor": "first_recording",
            "bins_hours": [0, 24, 48],
            "datasets": {
                "physionet_icare_2_1": {
                    "enabled": True,
                }
            },
        }
    }

    def _lookup(file_value: str) -> list[str]:
        if str(file_value).endswith("0332_002_001_EEG.hea"):
            return ["0332/0332_002_001_EEG.hea"]
        if str(file_value).endswith("0332_001_001_EEG.hea"):
            return ["0332/0332_001_001_EEG.hea"]
        return []

    out = build_time_reference_for_run(
        config=config,
        dataset_id="physionet_icare_2_1",
        dataset_root=dataset_root,
        index_df=index_df,
        lookup_rel_paths_by_file_value=_lookup,
        sub_id="sub-0332",
        run_id="002",
        acq_id="001",
        representative_file="0332_002_001_EEG.hea",
        sub_frame=sub_frame,
        window_start=np.array([0.0, 2.0], dtype=np.float32),
        window_end=np.array([2.0, 4.0], dtype=np.float32),
    )

    assert out["status"] == "ok"
    extension = out["extension"]
    assert isinstance(extension, dict)
    run_block = extension["run"]
    windows_block = extension["windows"]

    assert float(run_block["run_start_elapsed_sec"]) == pytest.approx(1200.0)
    np.testing.assert_allclose(
        windows_block["window_start_from_anchor_sec"],
        np.array([1200.0, 1202.0], dtype=np.float32),
        atol=1e-6,
    )
    assert np.all(windows_block["window_bin_id"] == 0)
    assert out["attrs"]["time_reference_status"] == "ok"
    assert out["manifest"]["status"] == "ok"


def test_build_time_reference_for_run_anchors_icare_elapsed_hour_run(tmp_path: Path):
    """A later run with a 3-digit elapsed-hour header still resolves an anchor offset.

    Regression guard for ingest_jacobian_fidelity_handover_2.md item 4: before
    the _CLOCK_RE widen, the second run's header ("100:00:00") failed to
    parse, so it was skipped by _unwrap_subject_timeline and
    window_start_from_anchor_sec stayed all-NaN for that run even though the
    subject's first recording (run-001, "22:59:08") did parse and could
    anchor it.
    """
    dataset_root = tmp_path / "icare"
    sub_dir = dataset_root / "0332"
    sub_dir.mkdir(parents=True)

    hea_first = sub_dir / "0332_022_001_EEG.hea"
    hea_late = sub_dir / "0332_081_100_EEG.hea"
    hea_first.write_text(
        "\n".join(
            [
                "0332_022_001_EEG 1 200 1000",
                "#Start time: 22:59:08",
                "#End time: 22:59:18",
            ]
        ),
        encoding="utf-8",
    )
    hea_late.write_text(
        "\n".join(
            [
                "0332_081_100_EEG 1 200 1000",
                "#Start time: 100:00:00",
                "#End time: 100:00:10",
            ]
        ),
        encoding="utf-8",
    )

    index_df = pd.DataFrame(
        [
            {
                "path": "0332/0332_022_001_EEG.hea",
                "subject": "0332",
                "run": "001",
                "acq": "022",
                "modality": "eeg",
            },
            {
                "path": "0332/0332_081_100_EEG.hea",
                "subject": "0332",
                "run": "100",
                "acq": "081",
                "modality": "eeg",
            },
        ]
    )
    sub_frame = pd.DataFrame(
        {
            "file": ["0332_081_100_EEG.hea", "0332_081_100_EEG.hea"],
            "t_start": [0.0, 2.0],
            "t_end": [2.0, 4.0],
        }
    )
    config = {
        "time_reference": {
            "enabled": True,
            "schema_version": "time_reference.v1",
            "parser": "wfdb_header",
            "anchor": "first_recording",
            "bins_hours": [0, 24, 48, 72],
            "datasets": {
                "physionet_icare_2_1": {
                    "enabled": True,
                }
            },
        }
    }

    def _lookup(file_value: str) -> list[str]:
        if str(file_value).endswith("0332_081_100_EEG.hea"):
            return ["0332/0332_081_100_EEG.hea"]
        if str(file_value).endswith("0332_022_001_EEG.hea"):
            return ["0332/0332_022_001_EEG.hea"]
        return []

    out = build_time_reference_for_run(
        config=config,
        dataset_id="physionet_icare_2_1",
        dataset_root=dataset_root,
        index_df=index_df,
        lookup_rel_paths_by_file_value=_lookup,
        sub_id="sub-0332",
        run_id="100",
        acq_id="081",
        representative_file="0332_081_100_EEG.hea",
        sub_frame=sub_frame,
        window_start=np.array([0.0, 2.0], dtype=np.float32),
        window_end=np.array([2.0, 4.0], dtype=np.float32),
    )

    assert out["status"] == "ok"
    windows_block = out["extension"]["windows"]
    # anchor (first recording) = 82748.0s ("22:59:08"); this run starts at
    # 360000.0s ("100:00:00") -> elapsed offset from anchor = 277252.0s.
    run_block = out["extension"]["run"]
    assert float(run_block["run_start_elapsed_sec"]) == pytest.approx(277252.0)
    np.testing.assert_allclose(
        windows_block["window_start_from_anchor_sec"],
        np.array([277252.0, 277254.0], dtype=np.float32),
        rtol=1e-5,
    )
