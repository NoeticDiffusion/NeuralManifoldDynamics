"""Checks for alignment, boundary sensitivity, and explicit unsupported support."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[2] / 'project/new_mnps_v.3.2/science_lead/sl037_synthetic_support.py'
SPEC = importlib.util.spec_from_file_location('sl037_synthetic_support', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize('fs', [256, 500])
def test_future_signal_does_not_enter_local_window_but_enters_context(fs):
    result = MODULE.experiment(fs)['future_burst_contamination']
    assert result['local']['max_fraction'] == 0
    assert result['context_4s']['max_fraction'] > 1e-5


def test_rejects_missing_target_support():
    with pytest.raises(ValueError, match='not supported'):
        MODULE.process(np.ones(5000), 500, 2, 18)


def test_clip_saturates_with_historical_coordinate_sign():
    z = np.array([6., 7., 12.])
    assert np.all(-np.clip(z, -6, 6)/2 == -3)
