import importlib.util
from pathlib import Path
import pytest

def load():
    path=Path(__file__).resolve().parents[2]/'project/new_mnps_v.3.2/science_lead/sl028_ds007554_source_check_20261029.py'
    spec=importlib.util.spec_from_file_location('sl028_source_check',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def test_fixed_header_support_and_truncated_record(tmp_path):
    m=load();fixed=bytearray(b' '*256)
    def put(a,b,s):fixed[a:b]=str(s).encode().ljust(b-a,b' ')
    put(0,8,'0');put(184,192,'512');put(236,244,'1');put(244,252,'1');put(252,256,'1')
    signal=bytearray(b' '*256);signal[216:224]=b'2       '
    path=tmp_path/'tiny.edf';path.write_bytes(fixed+signal+b'\x00'*4)
    h=m.edf_header(path);assert h['samples_per_record_all_signals']==[2] and h['whole_declared_records']
    path.write_bytes(fixed+signal+b'\x00'*2)
    h=m.edf_header(path);assert not h['whole_declared_records'] and h['expected_bytes']==516
    path.write_bytes(b'bad')
    with pytest.raises(ValueError,match='short EDF'):m.edf_header(path)

def test_lag_keeps_declared_rate_boundary():
    m=load()
    assert m.lag(249.9997)['status']=='unsupported'
    a=m.lag(250.0009);assert a['samples']==1 and abs(a['realized_seconds']-0.004)<1e-7
