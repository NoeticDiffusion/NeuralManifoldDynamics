import importlib.util
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'project/new_mnps_v.3.2/science_lead'))
spec=importlib.util.spec_from_file_location('sl034cal',ROOT/'project/new_mnps_v.3.2/science_lead/sl034_ieeg_calibration_pilot.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def make_features(i):
    return {('eeg_highfreq_power_30_45' if name=='embodied_arousal_proxy' else name):float(i+1) for name in m.FEATURES}


def test_anchors_projection_and_clipping():
    a=m.make_anchors([{'features':make_features(i)} for i in range(4)])
    result=m.project(make_features(1),a)
    assert len(result['coordinates_9d'])==9 and all(v is not None for v in result['coordinates_9d'].values())
    assert result['coordinates_9d']['m_e']==-result['z_features']['eeg_alpha']
    assert all(result['clipped'][name] for name in m.FEATURES if name not in m.LOGS) is False
    assert m.project(make_features(100000),a)['clipped']['eeg_beta_alpha']


def test_constant_and_invalid_inputs_are_not_imputed():
    f=make_features(0);anchors=m.make_anchors([{'features':f} for _ in range(4)])
    assert all(a['reason']=='zero_mad' and a['scale'] is None for a in anchors.values())
    out=m.project(f,anchors)
    assert all(v is None for v in out['coordinates_9d'].values())
    f2=make_features(1); f2['eeg_delta']=0
    anchors=m.make_anchors([{'features':f2}]+[{'features':make_features(i)} for i in range(1,4)])
    assert anchors['eeg_delta']['reason']=='missing_calibration_feature'
    assert m.project(make_features(1),anchors)['coordinates_9d']['m_a'] is None
