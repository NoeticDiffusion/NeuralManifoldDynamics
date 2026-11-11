import importlib.util
from pathlib import Path

def module():
    p=Path(__file__).resolve().parents[2]/'project/new_mnps_v.3.2/science_lead/sl029_ds004148_context_pilot_20261029.py'
    spec=importlib.util.spec_from_file_location('sl029_context_pilot',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_correlations_retain_missing_and_constant_status():
    m=module()
    assert m.number('n/a') is None and m.number('NaN') is None and m.number('0')==0
    assert m.corr(list(range(7)),list(range(7)))==(None,'fewer_than_8_complete_participants')
    assert m.corr(list(range(8)),[1]*8)==(None,'constant_difference')
    value,status=m.corr(list(range(8)),list(range(8)))
    assert status=='computed' and abs(value-1)<1e-12
