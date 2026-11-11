import importlib.util
import json
from pathlib import Path

path = Path(__file__).resolve().parents[2] / 'project/new_mnps_v.3.2/science_lead/sl033_ds004100_event_audit.py'
spec = importlib.util.spec_from_file_location('sl033_event_audit', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_ordered_events_and_sample_mismatch(tmp_path):
    base = 'sub-HUP060_task-ictal_acq-seeg_run-01'
    sidecar = tmp_path / (base + '_ieeg.json')
    sidecar.write_text(json.dumps({'RecordingDuration': 100, 'SamplingFrequency': 250, 'iEEGReference': 'n/a'}))
    events = tmp_path / (base + '_events.tsv')
    events.write_text('onset\tduration\ttrial_type\tsample\n10\t0\tsz onset\t2500\n55\t0\tsz offset\t13750\n')
    out = module.audit(sidecar)
    assert out['ictal_interval_s'] == [10, 55]
    assert out['full_20s_windows_inside_interval'] == 2
    assert not out['edf_present']
    events.write_text('onset\tduration\ttrial_type\tsample\n10\t0\tsz onset\t2501\n55\t0\tsz offset\t13750\n')
    assert module.audit(sidecar)['event_qc'] == 'sample_onset_mismatch'


def test_missing_and_invalid_intervals(tmp_path):
    base = 'sub-X_task-ictal_acq-ecog_run-01'
    sidecar = tmp_path / (base + '_ieeg.json')
    sidecar.write_text(json.dumps({'RecordingDuration': 100, 'SamplingFrequency': 250}))
    assert module.audit(sidecar)['event_qc'] == 'events_file_absent'
    (tmp_path / (base + '_events.tsv')).write_text('onset\ttrial_type\tsample\n30\tsz onset\t7500\n20\tsz offset\t5000\n')
    assert module.audit(sidecar)['event_qc'] == 'invalid_interval'
