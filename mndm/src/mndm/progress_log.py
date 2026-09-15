"""Compact ingest progress logging.

Default CLI output is one start line and one finish line per file, plus
``N of X finished, Y% done`` on the finish line. Stage timings, MNE FIR
dumps, ICA chatter, and per-sidecar write lines stay available behind
``--verbose-logs``.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Mapping, Optional

ENV_VERBOSE = "MNDM_VERBOSE_LOGS"
PROGRESS_LOGGER_NAME = "mndm.progress"

_COMPACT_WARNING_LOGGERS = (
    "core.io.h5_writer",
    "core.io.json_writer",
    "mne",
    "joblib",
    "mndm.preprocess",
    "mndm.features",
    "mndm.jacobian",
    "mndm.projection",
    "mndm.dynamics",
    "mndm.dynamical_families",
    "mndm.pipeline.event_alignment",
    "mndm.pipeline.event_annotations",
    "mndm.pipeline.event_locked_export",
    "mndm.pipeline.extensions_compute",
    "mndm.pipeline.regional_mnps",
    "mndm.pipeline.summary",
    "mndm.pipeline.summary_utils",
    "mndm.pipeline.summary_regional",
    "mndm.pipeline.summary_io",
    "mndm.pipeline.summary_qc",
)


def verbose_logs_enabled(config: Optional[Mapping[str, Any]] = None) -> bool:
    """Return True when the user asked for the old verbose ingest log."""
    env = str(os.environ.get(ENV_VERBOSE, "") or "").strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return True
    if env in {"0", "false", "no", "off"}:
        return False
    if isinstance(config, Mapping):
        run_cfg = config.get("run")
        if isinstance(run_cfg, Mapping) and bool(run_cfg.get("verbose_logs")):
            return True
    return False


def configure_runtime_logging(*, verbose: bool) -> None:
    """Apply compact or verbose logging for this process and child workers."""
    os.environ[ENV_VERBOSE] = "1" if verbose else "0"
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if verbose:
        for name in _COMPACT_WARNING_LOGGERS:
            logging.getLogger(name).setLevel(logging.NOTSET)
        try:
            import mne

            mne.set_log_level("INFO")
        except Exception:
            pass
        return
    for name in _COMPACT_WARNING_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger("mne").setLevel(logging.ERROR)
    try:
        import mne

        mne.set_log_level("ERROR")
    except Exception:
        pass


def joblib_verbose(*, verbose_logs: bool) -> int:
    """joblib 10 prints a progress dump per task; compact mode silences it."""
    return 10 if verbose_logs else 0


def log_file_started(label: str) -> None:
    """Log that a recording or summarize grouping has begun."""
    logging.getLogger(PROGRESS_LOGGER_NAME).info("Started %s", label)


def log_file_finished(label: str, done: int, total: int) -> None:
    """Log that a recording finished, with cohort progress."""
    pct = (100.0 * float(done) / float(total)) if total else 100.0
    logging.getLogger(PROGRESS_LOGGER_NAME).info(
        "Finished %s (%d of %d finished, %.1f%% done)",
        label,
        int(done),
        int(total),
        pct,
    )


class ProgressTracker:
    """Thread-safe N-of-X counter for feature and summarize loops."""

    def __init__(self, total: int):
        self.total = max(0, int(total))
        self._done = 0
        self._lock = threading.Lock()

    def started(self, label: str) -> None:
        log_file_started(label)

    def finished(self, label: str) -> int:
        with self._lock:
            self._done += 1
            done = self._done
        log_file_finished(label, done, self.total)
        return done
