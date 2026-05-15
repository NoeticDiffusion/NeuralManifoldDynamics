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


def test_parse_wfdb_clock_value_supports_rollover_suffix():
    """Parse WFDB clocks with rollover suffixes."""
    assert parse_wfdb_clock_value("22:59:08") == pytest.approx(82748.0)
    assert parse_wfdb_clock_value("24:00:00+") == pytest.approx(86400.0)
    assert parse_wfdb_clock_value("00:10:00+") == pytest.approx(87000.0)


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
