from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import numpy as np
import pytest

path = Path(__file__).resolve().parents[2] / "project/new_mnps_v.3.2/science_lead/sl030a_clean_support_qc_20261029.py"
spec = spec_from_file_location("sl030a_support", path)
mod = module_from_spec(spec)
spec.loader.exec_module(mod)


def test_distinguishes_constant_and_missing_samples():
    q = mod.participant_qc(np.array([[1., 1., 1.], [0., np.nan, 2.], [0., 1., 2.]]))
    assert q['nonfinite_values'] == 1
    assert q['constant_channel_row_indices_zero_based'] == [0]


def test_invalid_shape_fails():
    with pytest.raises(ValueError, match='channel-by-time'):
        mod.participant_qc(np.ones((2, 3, 4)))
