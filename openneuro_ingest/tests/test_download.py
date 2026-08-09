"""Tests for OpenNeuro download command isolation."""

from pathlib import Path

from openneuro import download


def test_uvx_download_drops_ingest_pythonpath(monkeypatch, tmp_path: Path):
    """The uvx child must not import the repository's local openneuro package."""
    monkeypatch.setenv("PYTHONPATH", r"H:\SourceRepo2\NeuralManifoldDynamics\openneuro_ingest\src")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)

    monkeypatch.setattr(
        download,
        "_resolve_openneuro_cli",
        lambda prefer_uvx=True: ["uvx", "openneuro-py@latest"],
    )
    monkeypatch.setattr(download.subprocess, "run", fake_run)

    download._perform_openneuro_download(
        "ds003710",
        tmp_path,
        [],
        [],
        use_uvx=True,
    )

    assert captured["command"][:2] == ["uvx", "openneuro-py@latest"]
    assert "PYTHONPATH" not in captured["env"]
