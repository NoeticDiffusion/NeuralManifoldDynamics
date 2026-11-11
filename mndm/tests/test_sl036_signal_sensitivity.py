import importlib.util
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'project/new_mnps_v.3.2/science_lead'))
spec=importlib.util.spec_from_file_location('sl036s',ROOT/'project/new_mnps_v.3.2/science_lead/sl036_signal_sensitivity.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_ed_known_rank_and_scale():
    x=np.array([[1,-1,1,-1],[1,1,-1,-1]],dtype=float)
    assert np.isclose(m.cov_ed(x),2.0)
    assert np.isclose(m.cov_ed(3*x),2.0)
    assert m.cov_ed(np.ones((2,4))) is None


def test_fractional_rms_segment():
    a=np.ones((2,8));b=np.ones((2,8));b[:,:2]+=0.25
    assert np.isclose(m.frac_diff(a,b,slice(0,2)),.25)
    assert m.frac_diff(a,b,slice(2,8))==0
    assert m.frac_diff(a*0,b) is None
