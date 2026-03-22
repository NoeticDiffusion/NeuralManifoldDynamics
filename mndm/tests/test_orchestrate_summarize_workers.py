"""Tests for summarize worker forwarding in orchestrate."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm import orchestrate


def test_cmd_summarize_forwards_n_jobs(monkeypatch):
    """Test cmd summarize forwards n jobs."""
    resolved = object()
    ctx = object()
    calls = {}

    monkeypatch.setattr(
        orchestrate.ResolvedConfig,
        "from_mapping",
        classmethod(lambda cls, config, out_dir, data_dir, mnps_overrides=None: resolved),
    )
    monkeypatch.setattr(
        orchestrate.SummarizeContext,
        "from_resolved",
        classmethod(lambda cls, resolved_cfg: ctx),
    )

    def _fake_summarize_with_context(ctx_arg, dataset_ids, subject=None, h5_mode="subject", n_jobs=1):
        """Internal helper: fake summarize with context."""
        calls["ctx"] = ctx_arg
        calls["dataset_ids"] = dataset_ids
        calls["subject"] = subject
        calls["h5_mode"] = h5_mode
        calls["n_jobs"] = n_jobs
        return 0

    monkeypatch.setattr(orchestrate, "_summarize_with_context", _fake_summarize_with_context)

    rc = orchestrate.cmd_summarize(
        config={"paths": {}},
        dataset_ids=["ds004511"],
        out_dir=None,
        data_dir=None,
        subject="sub-001",
        h5_mode="subject",
        n_jobs=12,
    )

    assert rc == 0
    assert calls["ctx"] is ctx
    assert calls["dataset_ids"] == ["ds004511"]
    assert calls["subject"] == "sub-001"
    assert calls["h5_mode"] == "subject"
    assert calls["n_jobs"] == 12
