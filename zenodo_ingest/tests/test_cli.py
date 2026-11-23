from __future__ import annotations

from pathlib import Path

from zenodo_ingest.cli import main
from zenodo_ingest.contracts import FileEntry, ZenodoRecord


def test_list_prints_selected_files_without_downloading(monkeypatch, capsys, tmp_path: Path) -> None:
    record = ZenodoRecord(
        record_id="4456079",
        title="Concurrent EEG",
        doi="10.5281/zenodo.4456079",
        api_url="https://zenodo.org/api/records/4456079",
        files=(
            FileEntry("Exp1_Raw Data.zip", "https://zenodo.org/files/a", 13368162568, "md5:" + "ab" * 16, "md5", "ab" * 16),
            FileEntry("GX_Demo_PreQuest_Scales.pdf", "https://zenodo.org/files/b", 514993, "md5:" + "cd" * 16, "md5", "cd" * 16),
        ),
    )

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def get_record(self, record_ref: str) -> ZenodoRecord:
            assert record_ref == "4456079"
            return record

        def download_file(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            raise AssertionError("list must not transfer bytes")

    monkeypatch.setattr("zenodo_ingest.download.ZenodoClient", FakeClient)
    config = tmp_path / "zenodo.yaml"
    config.write_text(
        "record: '4456079'\nselection:\n  include:\n    - '*.pdf'\n",
        encoding="utf-8",
    )
    assert main(["list", "--config", str(config)]) == 0
    output = capsys.readouterr().out
    assert "4456079  Concurrent EEG" in output
    assert "GX_Demo_PreQuest_Scales.pdf" in output
    assert "Exp1_Raw Data.zip" not in output
    assert "dry-run: nothing written" in output
