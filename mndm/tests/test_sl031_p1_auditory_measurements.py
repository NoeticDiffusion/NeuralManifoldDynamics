from importlib.util import spec_from_file_location,module_from_spec
from pathlib import Path
import numpy as np
import pytest
p=Path(__file__).resolve().parents[2]/'project/new_mnps_v.3.2/science_lead/sl031_p1_auditory_measurements_20261030.py'
s=spec_from_file_location('sl031_p1',p);m=module_from_spec(s);s.loader.exec_module(m)


def test_40hz_sinusoid_expected_energy():
    t=np.arange(2500)/250
    x=np.tile(np.sin(2*np.pi*40*t),(4,1))
    r=m.spectral_bands(x)
    assert r['valid'] and r['bands']['hz39_41']==pytest.approx(.5,abs=1e-10)
    assert r['bands']['hz36_38']<1e-20 and r['bands']['hz42_44']<1e-20


def test_zero_finite_and_nonfinite_and_short():
    x=np.zeros((4,2500));assert m.spectral_bands(x)['bands']['hz39_41']==0
    x[0,0]=np.nan;assert m.spectral_bands(x)['reason']=='nonfinite_input'
    assert m.spectral_bands(np.zeros((4,2499)))['reason']=='unexpected_shape'
