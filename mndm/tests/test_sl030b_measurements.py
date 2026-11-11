from importlib.util import module_from_spec,spec_from_file_location
from pathlib import Path
import numpy as np
import pytest

path=Path(__file__).resolve().parents[2]/'project/new_mnps_v.3.2/science_lead/sl030b_measurements_20261029.py'
spec=spec_from_file_location('sl030b_measurements',path);mod=module_from_spec(spec);spec.loader.exec_module(mod)


def test_welch_alpha_power_known_sinusoid():
    t=np.arange(625)/125
    x=np.tile(np.sin(2*np.pi*10*t),(125,1))
    result=mod.alpha_proxy(x)
    assert result['valid']
    assert result['value']==pytest.approx(0.5,abs=1e-10)


def test_zero_is_value_but_missing_and_short_are_unsupported():
    assert mod.alpha_proxy(np.zeros((125,625)))['value']==0
    x=np.ones((125,625));x[2,9]=np.nan
    assert mod.alpha_proxy(x)['reason']=='nonfinite_window'
    assert mod.alpha_proxy(np.ones((125,624)))['reason']=='unexpected_window_shape'
