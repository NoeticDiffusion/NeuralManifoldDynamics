from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import numpy as np
import pytest

path = Path(__file__).resolve().parents[2] / "project/new_mnps_v.3.2/science_lead/sl030a_raw_example_audit_20261029.py"
spec = spec_from_file_location("sl030a_raw", path)
mod = module_from_spec(spec)
spec.loader.exec_module(mod)


def markers(labels, times):
    a = np.empty((2, len(labels)), dtype=object)
    for i, (label, time) in enumerate(zip(labels, times)):
        a[0, i], a[1, i] = np.array([label]), np.array([[time]])
    return a


def test_trigger_parsing_and_click_offset():
    records = mod.extract_markers(markers(["DI21", "D128", "D128", "DIN6"], [1000, 2100, 62100, 70000]))
    trial = mod.summarize_trials(records, 20000, 125)[0]
    assert trial["stimulus"] == 21
    assert trial["audio_onset_from_first_click_ms"] == 1100
    assert trial["click_spacing_ms"] == [60000]
    assert trial["onset_inside_recording"]


def test_missing_click_remains_unsupported():
    trial = mod.summarize_trials([(21, 1000), (11, 2000), (128, 3000)], 1000, 125)[0]
    assert trial["audio_onset_from_first_click_ms"] is None
    assert not trial["onset_inside_recording"]


def test_unknown_code_and_nonmonotonic_times_rejected():
    with pytest.raises(ValueError, match="Unrecognized"):
        mod.extract_markers(markers(["bad"], [0]))
    with pytest.raises(ValueError, match="Non-monotonic"):
        mod.extract_markers(markers(["DI21", "D128"], [200, 100]))
