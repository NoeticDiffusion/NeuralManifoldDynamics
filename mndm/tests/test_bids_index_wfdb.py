"""WFDB-specific index coverage tests."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def test_build_file_index_includes_wfdb_header_with_paired_signal(tmp_path: Path):
    """WFDB .hea is indexed when matching .mat/.dat exists."""
    from mndm.bids_index import build_file_index

    dataset_root = tmp_path / "training"
    patient_dir = dataset_root / "0332"
    patient_dir.mkdir(parents=True)

    hea_path = patient_dir / "0332_001_022_EEG.hea"
    mat_path = patient_dir / "0332_001_022_EEG.mat"
    hea_path.write_text("dummy header", encoding="utf-8")
    mat_path.write_bytes(b"\x00\x01\x02\x03")

    index_df = build_file_index(dataset_root, config={"datasets": ["physionet_icare_2_1"]}, dataset_id="physionet_icare_2_1")

    assert len(index_df) == 1
    row = index_df.iloc[0]
    assert str(row["path"]).replace("\\", "/") == "0332/0332_001_022_EEG.hea"
    assert row["subject"] == "0332"
    assert row["modality"] == "eeg"


def test_build_file_index_skips_wfdb_header_without_paired_signal(tmp_path: Path):
    """WFDB .hea without .mat/.dat should be ignored to avoid guaranteed failures."""
    from mndm.bids_index import build_file_index

    dataset_root = tmp_path / "training"
    patient_dir = dataset_root / "0332"
    patient_dir.mkdir(parents=True)

    (patient_dir / "0332_001_022_EEG.hea").write_text("dummy header", encoding="utf-8")

    index_df = build_file_index(dataset_root, config={"datasets": ["physionet_icare_2_1"]}, dataset_id="physionet_icare_2_1")

    assert len(index_df) == 0
