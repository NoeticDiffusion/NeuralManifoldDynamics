from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from nemar_ingest.client import DownloadError, NemarError
from nemar_ingest.contracts import ManifestEntry
from nemar_ingest.download import download_dataset, select_manifest_entries


def test_select_manifest_entries_filters_bids_entities_and_globs() -> None:
    entries = (
        ManifestEntry("dataset_description.json"),
        ManifestEntry("sub-01/eeg/sub-01_task-EyesClosed_eeg.set"),
        ManifestEntry("sub-02/eeg/sub-02_task-EyesClosed_eeg.set"),
        ManifestEntry("sub-01/eeg/sub-01_task-EyesOpen_eeg.set"),
    )
    selected = select_manifest_entries(
        entries,
        {"subjects": ["01"], "tasks": ["EyesClosed"], "datatypes": ["eeg"], "include": ["sub-01/**"]},
    )
    assert [entry.path for entry in selected] == [
        "dataset_description.json", "sub-01/eeg/sub-01_task-EyesClosed_eeg.set"
    ]


def test_root_metadata_is_retained_with_subject_and_datatype_filters() -> None:
    entries = (
        ManifestEntry("dataset_description.json"),
        ManifestEntry("participants.tsv"),
        ManifestEntry("sub-01/eeg/sub-01_task-rest_eeg.set"),
    )
    selected = select_manifest_entries(entries, {
        "subjects": ["sub-01"], "datatypes": ["eeg"], "include": ["sub-01/**"]
    })
    assert [entry.path for entry in selected] == [
        "dataset_description.json", "participants.tsv", "sub-01/eeg/sub-01_task-rest_eeg.set"
    ]


def test_empty_selection_is_fail_closed() -> None:
    with pytest.raises(NemarError, match="nemar_selection_empty"):
        select_manifest_entries((ManifestEntry("sub-01/eeg/sub-01_task-rest_eeg.set"),), {"include": ["sub-99/**"]})


class _FakeClient:
    def __init__(self, entries: tuple[ManifestEntry, ...]):
        self.entries = entries
        self.manifest_calls = 0

    def get_manifest_with_digest(self, dataset_id: str, version: str):
        self.manifest_calls += 1
        return "https://data.nemar.org/on005385/v1.0.0/manifest.json", self.entries, "a" * 64

    def download_entry(self, entry: ManifestEntry, root: Path, **kwargs):
        target = root / Path(*entry.path.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fixture")
        return target


def test_cli_process_failure_does_not_fallback_to_manifest(monkeypatch, tmp_path: Path) -> None:
    client = _FakeClient((ManifestEntry("participants.tsv", "https://data.nemar.org/x"),))

    def fail_cli(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["nemar", "dataset", "download"])

    monkeypatch.setattr("nemar_ingest.download._cli_download", fail_cli)
    with pytest.raises(subprocess.CalledProcessError):
        download_dataset(
            "on005385",
            {"dataset": {"version": "v1.0.0"}, "download": {"backend": "auto"}},
            tmp_path,
            client=client,
        )
    assert not (tmp_path / "on005385" / "acquisition_receipt.json").exists()


def test_manifest_download_writes_versioned_atomic_provenance_receipt(tmp_path: Path) -> None:
    entries = (
        ManifestEntry("dataset_description.json", "https://data.nemar.org/description", 7),
        ManifestEntry("sub-01/eeg/file.set", "https://data.nemar.org/file", 7),
    )
    result = download_dataset(
        "on005385",
        {
            "dataset": {"version": "v1.0.0"},
            "download": {"backend": "manifest"},
            "selection": {"include": ["sub-01/**"]},
        },
        tmp_path,
        client=_FakeClient(entries),
    )
    receipt_path = tmp_path / "on005385" / "acquisition_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert result.manifest_url == receipt["manifest_url"]
    assert receipt["dataset_id"] == "on005385"
    assert receipt["version"] == "v1.0.0"
    assert receipt["manifest_sha256"] == "a" * 64
    assert [item["path"] for item in receipt["selected_entries"]] == [
        "dataset_description.json", "sub-01/eeg/file.set"
    ]
    assert not list((tmp_path / "on005385").glob(".acquisition_receipt.*.tmp"))


def test_file_download_error_continues_and_records_the_failed_path(tmp_path: Path) -> None:
    class _FlakyClient(_FakeClient):
        def download_entry(self, entry: ManifestEntry, root: Path, **kwargs):
            if entry.path.endswith("channels.tsv"):
                raise DownloadError(
                    f"nemar_file_download_failed:{entry.path}:HTTP Error 500: Internal Server Error"
                )
            return super().download_entry(entry, root, **kwargs)

    entries = (
        ManifestEntry("dataset_description.json", "https://data.nemar.org/description", 7),
        ManifestEntry("sub-049/ses-1/eeg/sub-049_ses-1_task-EyesOpen_acq-post_channels.tsv", "https://data.nemar.org/channels", 12),
        ManifestEntry("sub-050/ses-1/eeg/sub-050_ses-1_task-EyesOpen_acq-post_eeg.edf", "https://data.nemar.org/edf", 20),
    )
    result = download_dataset(
        "on005385",
        {"dataset": {"version": "v1.0.0"}, "download": {"backend": "manifest", "continue_on_error": True}},
        tmp_path,
        client=_FlakyClient(entries),
    )
    root = tmp_path / "on005385"
    assert (root / "dataset_description.json").is_file()
    assert (root / "sub-050/ses-1/eeg/sub-050_ses-1_task-EyesOpen_acq-post_eeg.edf").is_file()
    assert not (root / "sub-049/ses-1/eeg/sub-049_ses-1_task-EyesOpen_acq-post_channels.tsv").exists()
    assert result.failed_files == ("sub-049/ses-1/eeg/sub-049_ses-1_task-EyesOpen_acq-post_channels.tsv",)
    receipt = json.loads((root / "acquisition_receipt.json").read_text(encoding="utf-8"))
    assert receipt["failed_entries"][0]["path"] == result.failed_files[0]
    assert "HTTP Error 500" in receipt["failed_entries"][0]["error"]


def test_continue_on_error_can_be_disabled(tmp_path: Path) -> None:
    class _FlakyClient(_FakeClient):
        def download_entry(self, entry: ManifestEntry, root: Path, **kwargs):
            raise DownloadError(f"nemar_file_download_failed:{entry.path}:HTTP Error 500: Internal Server Error")

    with pytest.raises(DownloadError, match="HTTP Error 500"):
        download_dataset(
            "on005385",
            {"dataset": {"version": "v1.0.0"}, "download": {"backend": "manifest", "continue_on_error": False}},
            tmp_path,
            client=_FlakyClient((ManifestEntry("participants.tsv", "https://data.nemar.org/x"),)),
        )
    assert not (tmp_path / "on005385" / "acquisition_receipt.json").exists()


def test_receipt_from_another_version_cannot_be_reused(tmp_path: Path) -> None:
    root = tmp_path / "on005385"
    root.mkdir(parents=True)
    (root / "acquisition_receipt.json").write_text(json.dumps({
        "dataset_id": "on005385", "version": "v0.9.0"
    }), encoding="utf-8")
    client = _FakeClient((ManifestEntry("participants.tsv", "https://data.nemar.org/x"),))
    with pytest.raises(NemarError, match="nemar_receipt_version_mismatch"):
        download_dataset(
            "on005385",
            {"dataset": {"version": "v1.0.0"}, "download": {"backend": "manifest"}},
            tmp_path,
            client=client,
        )
    assert client.manifest_calls == 0
