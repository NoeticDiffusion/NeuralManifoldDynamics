"""Tests for orchestrate summarize command."""

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("mne")

from mndm import orchestrate


def test_cmd_summarize_builds_context_and_delegates(monkeypatch):
    """Test cmd summarize builds context and delegates."""
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
        """Handle fake from mapping."""
        captured["resolved"] = (cfg, out_dir, cli_data_dir, mnps_overrides)
        return dummy_resolved

    monkeypatch.setattr(orchestrate.ResolvedConfig, "from_mapping", classmethod(fake_from_mapping))

    def fake_from_resolved(cls, resolved):
        """Handle fake from resolved."""
        captured["from_resolved"] = resolved
        return dummy_ctx

    monkeypatch.setattr(orchestrate.SummarizeContext, "from_resolved", classmethod(fake_from_resolved))

    def fake_runner(ctx, dataset_ids, subject, h5_mode, n_jobs=1):
        """Handle fake runner."""
        captured["runner"] = (ctx, dataset_ids, subject, h5_mode, n_jobs)
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
    assert captured["runner"] == (dummy_ctx, ["ds001"], "001", "subject", 1)


def test_cmd_summarize_injects_anchor_fit_options(monkeypatch):
    """One-shot anchor-fit options should be added to the summarize config copy."""
    captured = {}

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
        captured["config"] = cfg
        return dummy_resolved

    monkeypatch.setattr(orchestrate.ResolvedConfig, "from_mapping", classmethod(fake_from_mapping))
    monkeypatch.setattr(
        orchestrate.SummarizeContext,
        "from_resolved",
        classmethod(lambda cls, resolved: object()),
    )
    monkeypatch.setattr(orchestrate, "_summarize_with_context", lambda *args, **kwargs: 0)

    rc = orchestrate.cmd_summarize(
        config={"paths": {"received_dir": "/tmp/recv", "processed_dir": "/tmp/proc"}},
        dataset_ids=["ds001"],
        out_dir=Path("/custom/out"),
        anchor_fit_options={"anchor_id": "unit-anchor", "scale_method": "iqr", "min_subjects": 2},
    )

    assert rc == 0
    assert captured["config"]["mnps_projection"]["anchor_auto_fit"]["enabled"] is True
    assert captured["config"]["mnps_projection"]["anchor_auto_fit"]["anchor_id"] == "unit-anchor"
    assert captured["config"]["mnps_projection"]["anchor_auto_fit"]["scale_method"] == "iqr"
    assert captured["config"]["mnps_projection"]["anchor_auto_fit"]["min_subjects"] == 2


def test_cmd_summarize_attaches_config_path_to_context(monkeypatch):
    """Config path should be forwarded so run manifest can copy source YAML."""
    captured = {}

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
    dummy_ctx = object()

    monkeypatch.setattr(
        orchestrate.ResolvedConfig,
        "from_mapping",
        classmethod(lambda cls, cfg, out_dir, cli_data_dir=None, mnps_overrides=None: dummy_resolved),
    )
    monkeypatch.setattr(
        orchestrate.SummarizeContext,
        "from_resolved",
        classmethod(lambda cls, resolved: dummy_ctx),
    )

    def fake_runner(ctx, dataset_ids, subject, h5_mode, n_jobs=1):
        """Capture proxied context."""
        captured["config_path"] = getattr(ctx, "config_path", None)
        return 0

    monkeypatch.setattr(orchestrate, "_summarize_with_context", fake_runner)

    rc = orchestrate.cmd_summarize(
        config={"paths": {"received_dir": "/tmp/recv", "processed_dir": "/tmp/proc"}},
        dataset_ids=["ds001"],
        out_dir=Path("/custom/out"),
        config_path=Path("/tmp/config_ingest.yaml"),
    )

    assert rc == 0
    assert captured["config_path"] == Path("/tmp/config_ingest.yaml")

