"""
bids.py
Utilities for parsing BIDS-style subject/session identifiers.
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
    match = pattern.search(path_str)
    if match:
        return match.group(1)
    return None


def parse_subject_session(path_str: str) -> Tuple[str, Optional[str]]:
    """Extract BIDS subject/session labels from a filepath or filename."""
    s = str(path_str)
    subject = _find_tag(s, _SUBJECT_RE)
    session = _find_tag(s, _SESSION_RE)
    return subject or "sub-unknown", session


def parse_subject_session_task(path_str: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Extract BIDS subject/session/task labels from a filepath or filename.

    Returns
    -------
    tuple
        (subject, session, task) where session and task may be None.
        Task is returned WITHOUT the 'task-' prefix (e.g., 'audioawake' not 'task-audioawake').
    """
    s = str(path_str)
    subject = _find_tag(s, _SUBJECT_RE)
    session = _find_tag(s, _SESSION_RE)
    task = _find_tag(s, _TASK_RE)
    return subject or "sub-unknown", session, task


def parse_subject_session_task_run(path_str: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Extract BIDS subject/session/task/run labels from a filepath or filename.

    Returns
    -------
    tuple
        (subject, session, task, run) where session/task/run may be None.
        Task is returned WITHOUT the 'task-' prefix (e.g., 'audioawake').
        Run is returned WITH the 'run-' prefix (e.g., 'run-01') for clarity.
    """
    s = str(path_str)
    subject = _find_tag(s, _SUBJECT_RE)
    session = _find_tag(s, _SESSION_RE)
    task = _find_tag(s, _TASK_RE)
    run_val = _find_tag(s, _RUN_RE)
    run = f"run-{run_val}" if run_val else None
    return subject or "sub-unknown", session, task, run


def parse_subject_session_task_run_acq(path_str: str) -> Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Extract BIDS subject/session/task/run/acq labels from a filepath or filename.

    Returns
    -------
    tuple
        (subject, session, task, run, acq) where session/task/run/acq may be None.
        Task is returned WITHOUT the 'task-' prefix (e.g., 'Sleep').
        Run and acq are returned WITH their prefixes (e.g., 'run-01', 'acq-psg').
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

