"""Tests for config-driven file exclusion filters."""

from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def test_apply_exclude_file_filters_filters_path_and_file_columns():
    from mndm.file_filters import apply_exclude_file_filters

    frame = pd.DataFrame(
        {
            "path": [
                "sub-001/eeg/sub-001_task-Sleep_acq-headband_eeg.edf",
                "sub-001/eeg/sub-001_task-Sleep_acq-psg_eeg.edf",
            ],
            "file": [
                "sub-001_task-Sleep_acq-headband_eeg.edf",
                "sub-001_task-Sleep_acq-psg_eeg.edf",
            ],
        }
    )

    filtered, excluded, patterns = apply_exclude_file_filters(
        frame,
        config={"exclude-files": ["_acq-headband_"]},
        candidate_columns=("path", "file"),
    )

    assert excluded == 1
    assert patterns == ["_acq-headband_"]
    assert filtered["file"].tolist() == ["sub-001_task-Sleep_acq-psg_eeg.edf"]


def test_resolve_exclude_file_patterns_supports_underscore_alias():
    from mndm.file_filters import resolve_exclude_file_patterns

    patterns = resolve_exclude_file_patterns({"exclude_files": ["foo", "bar"]})
    assert patterns == ["foo", "bar"]
