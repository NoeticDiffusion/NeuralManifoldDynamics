from __future__ import annotations

import json
from pathlib import Path

import pytest

from zenodo_ingest.client import DownloadError, ZenodoError
from zenodo_ingest.contracts import DownloadOptions, FileEntry, FileTransfer, ZenodoRecord
from zenodo_ingest.download import build_options, download_record, select_files


def _entry(key: str, size: int = 4) -> FileEntry:
    return FileEntry(key, f"https://zenodo.org/api/records/9/files/{key}/content", size, "md5:" + "ab" * 16, "md5", "ab" * 16)


def test_select_files_matches_basename_globs_with_spaces() -> None:
    entries = (
        _entry("Exp1_Raw Data.zip", 13),
        _entry("notes/readme.pdf", 2),
        _entry("GX_Demo_PreQuest_Scales.pdf", 5),
    )
    selected = select_files(entries, {"include": ["*.zip"], "exclude": ["notes/*"]})
    assert [item.key for item in selected] == ["Exp1_Raw Data.zip"]
    named = select_files(entries, {"files": ["readme.pdf"]})
    assert [item.key for item in named] == ["notes/readme.pdf"]


def test_empty_selection_is_fail_closed() -> None:
    with pytest.raises(ZenodoError, match="zenodo_selection_empty"):
        select_files((_entry("Exp1_Raw Data.zip"),), {"include": ["*.pdf"]})


def test_workers_above_eight_are_clamped() -> None:
    assert build_options({"workers": 30}).workers == 8


class _FakeClient:
    def __init__(self, record: ZenodoRecord, fail_keys: set[str] | None = None) -> None:
        self.record = record
        self.fail_keys = fail_keys or set()
        self.downloaded: list[str] = []

    def get_record(self, record_ref: str) -> ZenodoRecord:
        del record_ref
        return self.record

    def download_file(
        self,
        entry: FileEntry,
        destination_root: Path,
        options: DownloadOptions,
        *,
        trusted_digest: str | None = None,
    ) -> FileTransfer:
        del options, trusted_digest
        if entry.key in self.fail_keys:
            raise DownloadError(f"http_404:{entry.key}")
        target = destination_root / Path(*entry.key.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * (entry.size or 0))
        self.downloaded.append(entry.key)
        return FileTransfer(entry.key, str(target), entry.size, entry.checksum, "downloaded", True)


def _record() -> ZenodoRecord:
    return ZenodoRecord(
        record_id="9",
        files=(_entry("a.zip", 2), _entry("b.zip", 3)),
        title="Example",
        doi="10.5281/zenodo.9",
        concept_doi="10.5281/zenodo.8",
        api_url="https://zenodo.org/api/records/9",
    )


def test_download_writes_receipt_and_continues_after_a_file_error(tmp_path: Path) -> None:
    client = _FakeClient(_record(), fail_keys={"b.zip"})
    result = download_record(
        "9",
        {"download": {"workers": 1, "continue_on_error": True}, "selection": {}},
        tmp_path,
        client=client,
    )
    assert result.failed_files == ("b.zip",)
    assert client.downloaded == ["a.zip"]
    receipt = json.loads((tmp_path / "9" / "acquisition_receipt.json").read_text(encoding="utf-8"))
    assert receipt["record_id"] == "9"
    assert receipt["concept_doi"] == "10.5281/zenodo.8"
    assert [item["status"] for item in receipt["files"]] == ["downloaded", "error"]
    assert receipt["files"][0]["checksum_verified"] is True


def test_dry_run_does_not_create_files(tmp_path: Path) -> None:
    client = _FakeClient(_record())
    result = download_record("9", {"selection": {"include": ["a.zip"]}, "download": {}}, tmp_path, client=client, dry_run=True)
    assert [item.key for item in result.files] == ["a.zip"]
    assert result.dry_run is True
    assert client.downloaded == []
    assert not (tmp_path / "9").exists()


def test_receipt_mismatch_blocks_a_different_record(tmp_path: Path) -> None:
    root = tmp_path / "9"
    root.mkdir()
    (root / "acquisition_receipt.json").write_text(json.dumps({"record_id": "8", "files": []}), encoding="utf-8")
    with pytest.raises(ZenodoError, match="zenodo_receipt_record_mismatch"):
        download_record("9", {"selection": {}, "download": {"workers": 1}}, tmp_path, client=_FakeClient(_record()))
