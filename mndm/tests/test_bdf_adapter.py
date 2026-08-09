"""Regression tests for reusable non-BIDS BDF adapter utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _adapter_config(mapping: Path, bad_channels: Path) -> dict:
    return {
        "preprocess": {
            "bdf_adapter": {
                "enabled": True,
                "mapping_path": str(mapping),
                "mapping_key_column": "old_id",
                "mapping_subject_session_column": "subject_id",
                "mapping_subject_column": "subject",
                "mapping_session_column": "session",
                "subject_pad": 2,
                "session_pad": 2,
                "task": "rest",
                "channel_selection": {"mode": "first_n", "n_channels": 32},
                "canonical_channel_names": [f"C{i}" for i in range(32)],
                "bad_channels_path": str(bad_channels),
                "bad_channels_key_column": "subject_id",
                "bad_channels_column": "bad_channels",
                "interpolate_bads": False,
            }
        }
    }


def test_parse_bdf_entities_uses_explicit_mapping(tmp_path: Path):
    """Mapping tables override non-BIDS BDF filename inference."""
    from mndm.bdf_adapter import parse_bdf_entities

    mapping = tmp_path / "mapping.csv"
    mapping.write_text(
        "old_id,subject_id,subject,session\nTD1v2,sub-01_ses-02,1,2\n",
        encoding="utf-8",
    )
    bad = tmp_path / "bad.csv"
    bad.write_text("subject_id,bad_channels\nsub-01_ses-02,\"[C1]\"\n", encoding="utf-8")

    entities = parse_bdf_entities(tmp_path / "TD1v2.bdf", _adapter_config(mapping, bad), "test")

    assert entities == {
        "subject": "01",
        "session": "02",
        "task": "rest",
        "run": None,
        "acq": None,
        "mapping_id": "TD1v2",
    }


def test_bdf_adapter_selects_and_renames_active_bank(tmp_path: Path):
    """The adapter discards auxiliary/flat BDF banks before preprocessing."""
    mne = pytest.importorskip("mne")
    from mndm.bdf_adapter import apply_bdf_adapter_policy

    mapping = tmp_path / "mapping.csv"
    mapping.write_text(
        "old_id,subject_id,subject,session\nTD1v2,sub-01_ses-02,1,2\n",
        encoding="utf-8",
    )
    bad = tmp_path / "bad.csv"
    bad.write_text('subject_id,bad_channels\nsub-01_ses-02,"[C1,C4]"\n', encoding="utf-8")
    raw = mne.io.RawArray(
        np.random.default_rng(4).normal(size=(40, 256)),
        mne.create_info([f"A{i}" for i in range(40)], 256, "eeg"),
        verbose="ERROR",
    )

    meta = apply_bdf_adapter_policy(raw, tmp_path / "TD1v2.bdf", _adapter_config(mapping, bad), "test")

    assert meta["selected_channel_count"] == 32
    assert raw.ch_names == [f"C{i}" for i in range(32)]
    assert raw.info["bads"] == ["C1", "C4"]


def test_eeg_features_reject_windows_touching_bdf_bad_segments():
    """BAD_ masks reject windows in place without concatenating clean samples."""
    from mndm.features.eeg import compute_eeg_features

    sfreq = 50.0
    data = np.random.default_rng(3).normal(size=(3, int(20 * sfreq)))
    data[:, int(5 * sfreq) : int(6 * sfreq)] = np.nan
    signals = {
        "signals": {"eeg": data},
        "sfreq": sfreq,
        "channels": {"eeg": ["C1", "C2", "C3"]},
        "meta": {
            "bdf_adapter": {
                "bad_segments": {"enabled": True, "available": True},
            }
        },
    }
    config = {
        "epoching": {"length_s": 4.0, "step_s": 4.0},
        "features": {
            "eeg_psd": {"method": "welch"},
            "eeg_bands": {"theta": [4, 8]},
            "ratios": {},
        },
    }

    result = compute_eeg_features(signals, config)

    assert result["epoch_id"].tolist() == [0, 2, 3, 4]


def test_metadata_builder_exports_no_dates_or_dob(tmp_path: Path, monkeypatch):
    """Session tables retain age provenance but never timestamps or DOB."""
    from mndm import bdf_metadata

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for stem in ("TD1v1", "TD1v2"):
        (raw_dir / f"{stem}.bdf").write_bytes(b"placeholder")
    mapping = tmp_path / "mapping.csv"
    mapping.write_text(
        "old_id,subject_id\nTD1v1,sub-01_ses-01\nTD1v2,sub-01_ses-02\n",
        encoding="utf-8",
    )
    outcomes = tmp_path / "outcomes.xlsx"
    pd.DataFrame(
        {
            "Group": ["TD"],
            "Infant": ["TD1"],
            "Visit": [1],
            "Chronological Age (days)": [40],
            "Gender": ["Female"],
            "Bayley raw cognitive": [3],
        }
    ).to_excel(outcomes, index=False)
    dates = {
        "TD1v1.bdf": pd.Timestamp("2020-01-01"),
        "TD1v2.bdf": pd.Timestamp("2020-02-01"),
    }
    monkeypatch.setattr(bdf_metadata, "_read_meas_date", lambda path: dates[path.name])

    qc = bdf_metadata.build_bdf_metadata_tables(
        raw_dir=raw_dir,
        mapping_path=mapping,
        outcomes_path=outcomes,
        output_dir=tmp_path / "derived",
    )
    sessions = pd.read_csv(tmp_path / "derived" / "sessions.tsv", sep="\t")
    payload = (tmp_path / "derived" / "mapping_qc.json").read_text(encoding="utf-8").lower()

    assert sessions["age_days"].tolist() == [40, 71]
    assert sessions["age_source"].tolist() == [
        "spreadsheet_direct",
        "derived_from_first_visit_age_and_meas_date",
    ]
    assert not any("timestamp" in column.lower() and column != "recording_timestamp_available" for column in sessions.columns)
    assert "birthday" not in payload
    assert qc["privacy"]["exports_recording_timestamps"] is False
