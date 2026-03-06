"""Tests for orchestrate summarize command."""

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("mne")

from mndm import orchestrate


def test_cmd_summarize_builds_context_and_delegates(monkeypatch):
    captured = {}

    class DummyContext:
        pass

    dummy_ctx = DummyContext()
    dummy_resolved = SimpleNamespace(
        raw={"paths": {}},
        paths=SimpleNamespace(received_dir=Path("/recv"), processed_dir=Path("/proc")),
        coverage=SimpleNamespace(min_seconds=0.0, min_epochs=0),
        weights={},
        normalize_override=None,
        ingest_meta={},
        mnps_cfg={},
        derivative_cfg={},
        extensions_cfg={},
    )

    def fake_from_mapping(cls, cfg, out_dir, cli_data_dir=None, mnps_overrides=None):
        captured["resolved"] = (cfg, out_dir, cli_data_dir, mnps_overrides)
        return dummy_resolved

    monkeypatch.setattr(orchestrate.ResolvedConfig, "from_mapping", classmethod(fake_from_mapping))

    def fake_from_resolved(cls, resolved):
        captured["from_resolved"] = resolved
        return dummy_ctx

    monkeypatch.setattr(orchestrate.SummarizeContext, "from_resolved", classmethod(fake_from_resolved))

    def fake_runner(ctx, dataset_ids, subject, h5_mode):
        captured["runner"] = (ctx, dataset_ids, subject, h5_mode)
        return 42

    monkeypatch.setattr(orchestrate, "_summarize_with_context", fake_runner)

    config = {"paths": {"received_dir": "/tmp/recv", "processed_dir": "/tmp/proc"}}
    result = orchestrate.cmd_summarize(
        config=config,
        dataset_ids=["ds001"],
        out_dir=Path("/custom/out"),
        subject="001",
        h5_mode="subject",
    )

    assert result == 42
    assert captured["resolved"][0] == config
    assert captured["resolved"][1] == Path("/custom/out")
    assert captured["resolved"][2] is None
    assert captured["resolved"][3] is None
    assert captured["from_resolved"] is dummy_resolved
    assert captured["runner"] == (dummy_ctx, ["ds001"], "001", "subject")

