"""Tests for BIDS parsing utilities."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from core.bids import parse_subject_session
from core.bids import parse_subject_session_task_run
from core.bids import parse_subject_session_task_run_acq


def test_parse_subject_and_session_nested_path():
    subject, session = parse_subject_session("sub-001/ses-02/eeg/sub-001_ses-02_task-rest_eeg.set")
    assert subject == "sub-001"
    assert session == "ses-02"


def test_parse_subject_without_session():
    subject, session = parse_subject_session("sub-control01_task-rest_eeg.set")
    assert subject == "sub-control01"
    assert session is None


def test_parse_subject_with_letters_and_session_in_parent():
    subject, session = parse_subject_session("data/sub-PatientA/ses-night/eeg/file.set")
    assert subject == "sub-PatientA"
    assert session == "ses-night"


def test_parse_returns_unknown_when_absent():
    subject, session = parse_subject_session("random/file_without_tags.set")
    assert subject == "sub-unknown"
    assert session is None


def test_parse_task_run_from_filename():
    subject, session, task, run = parse_subject_session_task_run("sub-001_ses-01_task-rest_run-1_bold.nii.gz")
    assert subject == "sub-001"
    assert session == "ses-01"
    assert task == "rest"
    assert run == "run-1"


def test_parse_task_run_none_when_absent():
    subject, session, task, run = parse_subject_session_task_run("sub-001_task-rest_bold.nii.gz")
    assert subject == "sub-001"
    assert session is None
    assert task == "rest"
    assert run is None


def test_parse_task_run_acq_from_filename():
    subject, session, task, run, acq = parse_subject_session_task_run_acq("sub-001_task-Sleep_acq-psg_eeg.edf")
    assert subject == "sub-001"
    assert session is None
    assert task == "Sleep"
    assert run is None
    assert acq == "acq-psg"


def test_parse_task_with_hyphenated_value():
    subject, session, task, run = parse_subject_session_task_run(
        "sub-001_ses-01_task-resting-state_run-02_bold.nii.gz"
    )
    assert subject == "sub-001"
    assert session == "ses-01"
    assert task == "resting-state"
    assert run == "run-02"


def test_parse_acq_with_hyphenated_value():
    subject, session, task, run, acq = parse_subject_session_task_run_acq(
        "sub-001_task-Sleep_acq-psg-night_run-1_eeg.edf"
    )
    assert subject == "sub-001"
    assert task == "Sleep"
    assert run == "run-1"
    assert acq == "acq-psg-night"


def test_parse_windows_path_separators():
    subject, session, task, run = parse_subject_session_task_run(
        r"C:\data\sub-010\ses-03\eeg\sub-010_ses-03_task-rest_run-1_eeg.vhdr"
    )
    assert subject == "sub-010"
    assert session == "ses-03"
    assert task == "rest"
    assert run == "run-1"


def test_parse_avoids_embedded_false_positive():
    subject, session = parse_subject_session("xsub-001y/file_without_entities.set")
    assert subject == "sub-unknown"
    assert session is None

