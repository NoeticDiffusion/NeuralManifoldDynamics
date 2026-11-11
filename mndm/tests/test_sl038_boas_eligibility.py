import importlib.util
from pathlib import Path

PATH = Path(__file__).resolve().parents[2]/'project/new_mnps_v.3.2/science_lead/sl038_boas_eligibility.py'
spec = importlib.util.spec_from_file_location('sl038_boas_eligibility', PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_centered_half_open_window_and_sample_grid():
    candidates, failures = m.event_support([{'onset':'30','duration':'30','stage_hum':'2'}],256,20000)
    assert not failures
    assert [(r['target_start_s'],r['target_end_s']) for r in candidates] == [(35,55),(41,49)]
    assert candidates[0]['source_end_sample_exclusive'] == 55*256


def test_overlap_and_short_recording_do_not_emit_targets():
    events=[{'onset':'0','duration':'30','stage_hum':'0'}, {'onset':'29','duration':'30','stage_hum':'2'}]
    candidates, failures=m.event_support(events,256,100000)
    assert not candidates and failures['overlapping_annotation']==2
    candidates, failures=m.event_support(events[:1],256,100)
    assert not candidates and failures['20s_outside_support']==1 and failures['8s_outside_support']==1


def test_long_nested_annotation_is_detected():
    events=[{'onset':'0','duration':'100','stage_hum':'1'},
            {'onset':'10','duration':'1','stage_hum':'1'},
            {'onset':'30','duration':'30','stage_hum':'2'}]
    candidates, failures=m.event_support(events,256,100000)
    assert not candidates and failures['overlapping_annotation']==1
