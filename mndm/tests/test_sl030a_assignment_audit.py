from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

_PATH = (Path(__file__).resolve().parents[2] / "project" / "new_mnps_v.3.2" /
         "science_lead" / "sl030a_assignment_audit_20261029.py")
_spec = spec_from_file_location("sl030a_assignment_audit", _PATH)
mod = module_from_spec(_spec)
_spec.loader.exec_module(mod)


def assignment():
    a = np.empty((16, 12), dtype=object)
    # Four groups; rotate condition by song so each participant has all four.
    for condition in range(4):
        for song in range(4):
            group = (condition - song) % 4
            for j in range(12):
                a[condition * 4 + song, j] = np.array([f"S{group * 12 + j + 1}"])
    return a


def test_assignment_metadata_cannot_certify_eeg_pairing():
    result = mod.audit_assignment(assignment())
    assert result["unique_participants"] == 48
    assert result["actual_repeat_eeg_pairing_verified"] is False
    assert result["qualified"] is False


def test_duplicate_participant_in_row_fails():
    a = assignment()
    a[0, 1] = a[0, 0]
    with pytest.raises(ValueError, match="duplicate"):
        mod.audit_assignment(a)


def test_bad_shape_fails():
    with pytest.raises(ValueError, match="Expected"):
        mod.audit_assignment(assignment()[:15])


def test_mismatched_group_design_fails():
    a = assignment()
    a[4], a[5] = a[5].copy(), a[4].copy()
    with pytest.raises(ValueError, match="one version"):
        mod.audit_assignment(a)
