"""Multimodal BIDS index coverage tests."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def test_build_file_index_detects_ds003838_multimodal_bundle_members(tmp_path: Path):
    """EEG/ECG/pupil/beh files with one BIDS key should share a bundle key."""
    from mndm.bids_index import build_file_index

    dataset_root = tmp_path / "ds003838"
    eeg_dir = dataset_root / "sub-001" / "eeg"
    ecg_dir = dataset_root / "sub-001" / "ecg"
    pupil_dir = dataset_root / "sub-001" / "pupil"
    beh_dir = dataset_root / "sub-001" / "beh"
    eeg_dir.mkdir(parents=True)
    ecg_dir.mkdir(parents=True)
    pupil_dir.mkdir(parents=True)
    beh_dir.mkdir(parents=True)

    (eeg_dir / "sub-001_task-memory_eeg.set").write_bytes(b"\x00")
    (ecg_dir / "sub-001_task-memory_ecg.set").write_bytes(b"\x00")
    (pupil_dir / "sub-001_task-memory_pupil.tsv").write_text("time\tpupil\n0.0\t1.0\n", encoding="utf-8")
    (pupil_dir / "sub-001_task-memory_events.tsv").write_text("onset\tduration\ttrial_type\n0.0\t1.0\tlisten\n", encoding="utf-8")
    (beh_dir / "sub-001_task-memory_beh.tsv").write_text("trial\tresponse\n1\t7\n", encoding="utf-8")

    index_df = build_file_index(dataset_root, config={"datasets": ["ds003838"]}, dataset_id="ds003838")

    assert len(index_df) == 4
    by_modality = {str(row["modality"]): row for _, row in index_df.iterrows()}
    assert set(by_modality.keys()) == {"eeg", "ecg", "pupil", "beh"}
    assert by_modality["eeg"]["datatype"] == "eeg"
    assert by_modality["ecg"]["datatype"] == "ecg"
    assert by_modality["pupil"]["datatype"] == "pupil"
    assert by_modality["beh"]["datatype"] == "beh"
    assert str(by_modality["pupil"]["path"]).endswith("sub-001_task-memory_pupil.tsv")
    bundle_keys = {str(row["bundle_key"]) for _, row in index_df.iterrows()}
    assert bundle_keys == {"001||memory||"}


def test_build_file_index_detects_meg_fif_with_meg_sidecars(tmp_path: Path):
    """MEG FIF recordings under /meg should be indexed as modality/datatype 'meg'."""
    from mndm.bids_index import build_file_index

    dataset_root = tmp_path / "ds003645"
    meg_dir = dataset_root / "sub-002" / "meg"
    meg_dir.mkdir(parents=True)

    (meg_dir / "sub-002_task-facerecognition_meg.fif").write_bytes(b"FIFF")
    (meg_dir / "sub-002_task-facerecognition_meg.json").write_text("{}", encoding="utf-8")
    (meg_dir / "sub-002_task-facerecognition_channels.tsv").write_text("name\ttype\nMEG0111\tMEGMAG\n", encoding="utf-8")
    (meg_dir / "sub-002_task-facerecognition_events.tsv").write_text(
        "onset\tduration\ttrial_type\n0\t1\tface\n",
        encoding="utf-8",
    )

    index_df = build_file_index(dataset_root, config={"datasets": ["ds003645"]}, dataset_id="ds003645")

    assert len(index_df) == 1
    row = index_df.iloc[0]
    assert row["modality"] == "meg"
    assert row["datatype"] == "meg"
    assert row["meg_json"].endswith("_meg.json")
    assert row["channels_tsv"].endswith("_channels.tsv")
    assert row["events_tsv"].endswith("_events.tsv")


def test_build_file_index_detects_ctf_ds_recordings(tmp_path: Path):
    """CTF MEG recordings (``*_meg.ds`` directories) are indexed once each.

    CTF stores each recording as a directory ending in ``.ds`` that itself
    contains internal sub-directories also ending in ``.ds`` (e.g. ``hz.ds``);
    only the outer recording directory should be indexed.
    """
    from mndm.bids_index import build_file_index

    dataset_root = tmp_path / "ds003568"
    meg_dir = dataset_root / "sub-24295" / "meg"
    ds_dir = meg_dir / "sub-24295_task-rest_run-1_meg.ds"
    ds_dir.mkdir(parents=True)
    (ds_dir / "sub-24295_task-rest_run-1_meg.meg4").write_bytes(b"\x00" * 1024)
    (ds_dir / "sub-24295_task-rest_run-1_meg.res4").write_bytes(b"\x00" * 512)
    (ds_dir / "hz.ds").mkdir()  # internal CTF sub-directory; must be skipped
    (ds_dir / "hz.ds" / "hz.meg4").write_bytes(b"\x00" * 256)
    (meg_dir / "sub-24295_task-rest_run-1_meg.json").write_text("{}", encoding="utf-8")
    (meg_dir / "sub-24295_task-rest_run-1_channels.tsv").write_text("name\ttype\nMLC11\tMEGMAG\n", encoding="utf-8")
    (meg_dir / "sub-24295_task-rest_run-1_events.tsv").write_text("onset\tduration\ttrial_type\n0\t1\trest\n", encoding="utf-8")

    index_df = build_file_index(dataset_root, config={"datasets": ["ds003568"]}, dataset_id="ds003568")

    assert len(index_df) == 1
    row = index_df.iloc[0]
    assert row["modality"] == "meg"
    assert row["datatype"] == "meg"
    assert row["subject"] == "24295"
    assert row["task"] == "rest"
    assert row["run"] == "1"
    assert str(row["path"]).endswith("sub-24295_task-rest_run-1_meg.ds")
    assert row["meg_json"].endswith("_meg.json")
    assert row["channels_tsv"].endswith("_channels.tsv")
    assert row["events_tsv"].endswith("_events.tsv")
    assert row["size"] == 1024 + 512 + 256  # all .ds dir contents, incl. hz.ds
