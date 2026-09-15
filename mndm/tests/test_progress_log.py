from __future__ import annotations

import logging

from mndm.progress_log import (
    ProgressTracker,
    joblib_verbose,
    log_file_finished,
    verbose_logs_enabled,
)


def test_verbose_logs_default_off(monkeypatch):
    monkeypatch.delenv("MNDM_VERBOSE_LOGS", raising=False)
    assert verbose_logs_enabled({}) is False
    assert joblib_verbose(verbose_logs=False) == 0
    assert joblib_verbose(verbose_logs=True) == 10


def test_verbose_logs_env_and_config(monkeypatch):
    monkeypatch.setenv("MNDM_VERBOSE_LOGS", "1")
    assert verbose_logs_enabled({}) is True
    monkeypatch.setenv("MNDM_VERBOSE_LOGS", "0")
    assert verbose_logs_enabled({"run": {"verbose_logs": True}}) is False
    monkeypatch.delenv("MNDM_VERBOSE_LOGS", raising=False)
    assert verbose_logs_enabled({"run": {"verbose_logs": True}}) is True


def test_progress_tracker_emits_n_of_x(caplog):
    tracker = ProgressTracker(4)
    with caplog.at_level(logging.INFO, logger="mndm.progress"):
        tracker.started("a.edf")
        tracker.finished("a.edf")
        tracker.started("b.edf")
        tracker.finished("b.edf")
    messages = [rec.getMessage() for rec in caplog.records]
    assert "Started a.edf" in messages
    assert "Finished a.edf (1 of 4 finished, 25.0% done)" in messages
    assert "Finished b.edf (2 of 4 finished, 50.0% done)" in messages


def test_compact_mode_raises_noisy_loggers(monkeypatch):
    monkeypatch.delenv("MNDM_VERBOSE_LOGS", raising=False)
    from mndm.progress_log import configure_runtime_logging

    configure_runtime_logging(verbose=False)
    assert logging.getLogger("mndm.features").getEffectiveLevel() == logging.WARNING
    assert logging.getLogger("mne").getEffectiveLevel() == logging.ERROR
    configure_runtime_logging(verbose=True)

