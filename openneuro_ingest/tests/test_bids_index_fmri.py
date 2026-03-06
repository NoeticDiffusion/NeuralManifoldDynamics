"""Tests for BIDS index including fMRI files."""

from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def test_build_file_index_includes_fmri(tmp_path):
    """BIDS index should include both EEG and fMRI BOLD files."""
    from openneuro import bids_index

    root = tmp_path / "dsTEST"
    # EEG structure
    eeg_dir = root / "sub-001" / "ses-01" / "eeg"
    eeg_dir.mkdir(parents=True, exist_ok=True)
    eeg_file = eeg_dir / "sub-001_ses-01_task-rest_eeg.edf"
    eeg_file.write_bytes(b"EEG")
    (eeg_dir / "sub-001_ses-01_task-rest_eeg.json").write_text("{}", encoding="utf-8")
    (eeg_dir / "channels.tsv").write_text("name\ttype\nCz\tEEG\n", encoding="utf-8")
    (eeg_dir / "events.tsv").write_text("onset\tduration\n0\t1\n", encoding="utf-8")

    # fMRI structure (BOLD)
    func_dir = root / "sub-001" / "ses-01" / "func"
    func_dir.mkdir(parents=True, exist_ok=True)
    bold_name = "sub-001_ses-01_task-rest_run-1_bold.nii.gz"
    bold_file = func_dir / bold_name
    bold_file.write_bytes(b"NIFTI")
    (func_dir / "sub-001_ses-01_task-rest_run-1_bold.json").write_text("{}", encoding="utf-8")
    (func_dir / "sub-001_ses-01_task-rest_run-1_bold_events.tsv").write_text(
        "onset\tduration\n0\t10\n", encoding="utf-8"
    )

    df = bids_index.build_file_index(root)
    assert isinstance(df, pd.DataFrame)
    # Should contain exactly one EEG and one fMRI entry
    assert set(df["modality"]) == {"eeg", "fmri"}

    eeg_row = df[df["modality"] == "eeg"].iloc[0]
    fmri_row = df[df["modality"] == "fmri"].iloc[0]

    assert eeg_row["subject"] == "001"
    assert eeg_row["session"] == "01"
    assert eeg_row["task"] == "rest"
    assert eeg_row["eeg_json"].endswith("_eeg.json")
    assert eeg_row["channels_tsv"].endswith("channels.tsv")
    assert eeg_row["events_tsv"].endswith("events.tsv")
    assert fmri_row["subject"] == "001"
    assert fmri_row["session"] == "01"
    assert fmri_row["task"] == "rest"
    assert fmri_row["fmri_json"].endswith("_bold.json")
    assert fmri_row["fmri_events_tsv"].endswith("_bold_events.tsv")


