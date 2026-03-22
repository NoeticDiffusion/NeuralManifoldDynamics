"""Utilities for parsing BIDS-style subject, session, task, run, and acq labels.

Regular expressions match BIDS-like entity boundaries in file paths so labels are
not captured from unrelated substrings inside longer tokens.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# Use BIDS-like entity boundaries to avoid matching inside longer tokens.
_SUBJECT_RE = re.compile(r"(?:^|[\\/_ -])(sub-[A-Za-z0-9]+)(?:$|[\\/_ .-])")
_SESSION_RE = re.compile(r"(?:^|[\\/_ -])(ses-[A-Za-z0-9]+)(?:$|[\\/_ .-])")
_TASK_RE = re.compile(r"(?:^|[\\/_ -])task-([A-Za-z0-9-]+)(?:$|[\\/_ .-])")
_RUN_RE = re.compile(r"(?:^|[\\/_ -])run-([A-Za-z0-9]+)(?:$|[\\/_ .-])")
_ACQ_RE = re.compile(r"(?:^|[\\/_ -])acq-([A-Za-z0-9-]+)(?:$|[\\/_ .-])")


def _find_tag(path_str: str, pattern: re.Pattern[str]) -> Optional[str]:
    """Return the first capture group matched by ``pattern`` in ``path_str``, or None."""
    match = pattern.search(path_str)
    if match:
        return match.group(1)
    return None


def parse_subject_session(path_str: str) -> Tuple[str, Optional[str]]:
    """Extract BIDS ``sub-`` and ``ses-`` labels from a filepath or filename.

    Args:
        path_str: Any string containing BIDS entities (path or filename).

    Returns:
        ``(subject, session)`` where ``subject`` defaults to ``sub-unknown`` if
        missing; ``session`` may be None.
    """
    s = str(path_str)
    subject = _find_tag(s, _SUBJECT_RE)
    session = _find_tag(s, _SESSION_RE)
    return subject or "sub-unknown", session


def parse_subject_session_task(path_str: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Extract BIDS subject, session, and task labels from a filepath or filename.

    Args:
        path_str: Any string containing BIDS entities.

    Returns:
        ``(subject, session, task)``. Session and task may be None. Task is
        returned without the ``task-`` prefix (for example ``audioawake``, not
        ``task-audioawake``).
    """
    s = str(path_str)
    subject = _find_tag(s, _SUBJECT_RE)
    session = _find_tag(s, _SESSION_RE)
    task = _find_tag(s, _TASK_RE)
    return subject or "sub-unknown", session, task


def parse_subject_session_task_run(path_str: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Extract BIDS subject, session, task, and run labels from a path or filename.

    Args:
        path_str: Any string containing BIDS entities.

    Returns:
        ``(subject, session, task, run)``. Session, task, or run may be None.
        Task omits the ``task-`` prefix. Run includes the ``run-`` prefix (for
        example ``run-01``).
    """
    s = str(path_str)
    subject = _find_tag(s, _SUBJECT_RE)
    session = _find_tag(s, _SESSION_RE)
    task = _find_tag(s, _TASK_RE)
    run_val = _find_tag(s, _RUN_RE)
    run = f"run-{run_val}" if run_val else None
    return subject or "sub-unknown", session, task, run


def parse_subject_session_task_run_acq(path_str: str) -> Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Extract BIDS subject, session, task, run, and acquisition labels.

    Args:
        path_str: Any string containing BIDS entities.

    Returns:
        ``(subject, session, task, run, acq)``. Optional fields may be None.
        Task omits the ``task-`` prefix. Run and acq include ``run-`` and
        ``acq-`` prefixes when present (for example ``run-01``, ``acq-psg``).
    """
    s = str(path_str)
    subject = _find_tag(s, _SUBJECT_RE)
    session = _find_tag(s, _SESSION_RE)
    task = _find_tag(s, _TASK_RE)
    run_val = _find_tag(s, _RUN_RE)
    acq_val = _find_tag(s, _ACQ_RE)
    run = f"run-{run_val}" if run_val else None
    acq = f"acq-{acq_val}" if acq_val else None
    return subject or "sub-unknown", session, task, run, acq

