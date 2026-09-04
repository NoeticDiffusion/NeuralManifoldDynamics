"""Tests for OpenNeuro download command isolation."""

from pathlib import Path

import pytest

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
        ["**/*_events.tsv"],
        [],
        use_uvx=True,
    )

    assert captured["command"][:2] == ["uvx", "openneuro-py@latest"]
    assert "--include=**/*_events.tsv" in captured["command"]
    assert "PYTHONPATH" not in captured["env"]


def test_metadata_only_download_rejects_raw_patterns(tmp_path: Path):
    config = {
        "download": {
            "metadata_only_patterns": ["**/eeg/**"],
        }
    }

    try:
        download.download_metadata_only(["ds003670"], config, tmp_path)
    except ValueError as exc:
        assert "metadata_only_pattern_not_allowlisted" in str(exc)
    else:
        raise AssertionError("raw signal directory pattern was accepted")


def test_metadata_only_download_does_not_remove_missing_patterns(
    monkeypatch,
    tmp_path: Path,
):
    config = {
        "download": {
            "retries": 3,
            "metadata_only_patterns": ["**/*_events.tsv"],
        }
    }
    calls = []

    def fail_strictly(*args, **kwargs):
        calls.append(args)
        raise download.MissingIncludePatternError(
            ["**/*_events.tsv"],
            source="fixture",
        )

    monkeypatch.setattr(download, "_perform_openneuro_download", fail_strictly)

    try:
        download.download_metadata_only(["ds003670"], config, tmp_path)
    except download.MissingIncludePatternError:
        pass
    else:
        raise AssertionError("missing metadata pattern was not fail-closed")

    assert len(calls) == 1


def test_metadata_only_download_rejects_existing_payload(tmp_path: Path):
    dataset_path = tmp_path / "ds003670"
    dataset_path.mkdir()
    (dataset_path / "sub-01_task-stim_eeg.edf").write_bytes(b"payload")
    config = {
        "download": {
            "metadata_only_patterns": ["**/*_events.tsv"],
        }
    }

    try:
        download.download_metadata_only(["ds003670"], config, tmp_path)
    except RuntimeError as exc:
        assert "metadata_only_target_contains_payload" in str(exc)
    else:
        raise AssertionError("existing signal payload was accepted")


def test_metadata_only_optional_patterns_never_fall_back_to_unfiltered(
    monkeypatch,
    tmp_path: Path,
):
    config = {
        "download": {
            "retries": 1,
            "metadata_only_patterns": ["**/*_events.tsv"],
            "metadata_only_optional_patterns": ["**/*_events.json"],
        }
    }
    calls = []

    def emulate_optional_missing(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise download.MissingIncludePatternError(
                ["**/*_events.json"],
                source="fixture",
            )
        (args[1] / "dataset_description.json").write_text("{}", encoding="utf-8")
        (args[1] / "sub-01_task-stim_events.tsv").write_text(
            "onset\tduration\ttrial_type\n0\t1\tstim\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(download, "_perform_openneuro_download", emulate_optional_missing)

    result = download.download_metadata_only(["ds003670"], config, tmp_path)

    assert result["ds003670"].exists()
    assert len(calls) == 2
    assert calls[0][2] == ["**/*_events.tsv", "**/*_events.json"]
    assert calls[1][2] == ["**/*_events.tsv"]


def test_explicit_payload_download_keeps_exact_allowlist(
    monkeypatch,
    tmp_path: Path,
):
    calls = []

    def emulate_download(dataset_id, dataset_path, includes, excludes, **kwargs):
        calls.append(list(includes))
        for relative in includes:
            target = dataset_path / Path(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"payload")

    monkeypatch.setattr(download, "_perform_openneuro_download", emulate_download)
    result = download.download_explicit_allowlist(
        "ds003670",
        ["sub-001/ses-01/eeg/a_eeg.set"],
        tmp_path,
    )
    assert result == tmp_path / "ds003670"
    assert calls == [["sub-001/ses-01/eeg/a_eeg.set"]]


def test_explicit_payload_download_rejects_globs_and_unallowlisted_existing(
    tmp_path: Path,
):
    with pytest.raises(ValueError, match="explicit_payload_path_not_allowed"):
        download.download_explicit_allowlist(
            "ds003670",
            ["sub-*/**/*.set"],
            tmp_path,
        )
    dataset_path = tmp_path / "ds003670"
    dataset_path.mkdir()
    extra = dataset_path / "sub-001" / "unexpected.fdt"
    extra.parent.mkdir(parents=True)
    extra.write_bytes(b"payload")
    with pytest.raises(
        RuntimeError,
        match="target_contains_unallowlisted_payload",
    ):
        download.download_explicit_allowlist(
            "ds003670",
            ["sub-001/allowed.set"],
            tmp_path,
        )
