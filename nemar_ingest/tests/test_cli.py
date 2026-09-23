from __future__ import annotations

from pathlib import Path

from nemar_ingest import download


def test_cli_command_has_version_pin_and_selection_flags(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(download.shutil, "which", lambda name: "nemar.exe")
    monkeypatch.setattr(download.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd))
    download._cli_download(
        "on005385", tmp_path / "on005385", "v1.0.0",
        {"subjects": ["sub-01"], "tasks": ["EyesClosed"], "datatypes": ["eeg"]},
        {"cli_command": "nemar"},
    )
    assert calls == [["nemar.exe", "dataset", "download", "on005385", "--version", "v1.0.0", "--output", str(tmp_path / "on005385"), "--subjects", "sub-01", "--tasks", "EyesClosed", "--datatypes", "eeg"]]

