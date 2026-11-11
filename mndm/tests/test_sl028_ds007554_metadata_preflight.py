import importlib.util
import json
from pathlib import Path
import pytest

def test_metadata_inventory_preserves_missing_and_avoids_payload_reads(tmp_path, monkeypatch):
    repo=Path(__file__).resolve().parents[2]
    script=repo/'project/new_mnps_v.3.2/science_lead/sl028_ds007554_metadata_preflight.py'
    spec=importlib.util.spec_from_file_location('sl028_preflight',script)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    root=tmp_path/'source';eeg=root/'sub-001'/'ses-01'/'eeg';eeg.mkdir(parents=True)
    (root/'dataset_description.json').write_text(json.dumps({'DatasetDOI':'fixture'}))
    stem='sub-001_ses-01_task-mentalarithmetic'
    (eeg/f'{stem}_eeg.edf').write_bytes(b'fixture payload')
    (eeg/f'{stem}_eeg.json').write_text(json.dumps({'SamplingFrequency':250,'EEGReference':'ear'}))
    (eeg/f'{stem}_channels.tsv').write_text('name\ttype\tunits\nFz\tEEG\tuV\n')
    (eeg/f'{stem}_events.tsv').write_text('onset\tduration\ttrial_type\n25\t0.5\tstart\n26\t1\tstim\n')
    old=m.hash_file
    def guarded(path):
        assert path.suffix!='.edf'
        return old(path)
    monkeypatch.setattr(m,'hash_file',guarded)
    out=tmp_path/'out';result=m.run(root,out)
    assert result['requested_cells']==6 and result['structurally_complete_cells']==1 and result['structurally_complete_participants']==0
    assert result['snapshot_commit_verified'] is False and result['records'][0]['files']['payload']['sha256'] is None
    assert result['records'][0]['events']['start_marker_onsets_seconds']==[25.0]
    assert result['records'][0]['events']['task_end_status']=='not inferred from last event'
    assert json.loads((out/'manifest.json').read_text())['payloads_read'] is False
    with pytest.raises(FileExistsError):m.run(root,out)
    (eeg/f'{stem}_eeg.edf').unlink()
    tree=tmp_path/'tree.json'
    tree.write_text(json.dumps({'sha':'fixture','truncated':False,'tree':[{'path':f'sub-001/ses-01/eeg/{stem}_eeg.edf','mode':'120000'}]}))
    listed=m.run(root,tmp_path/'listed',tree)
    assert listed['structurally_complete_cells']==0 and listed['metadata_present_payload_listed_cells']==1
    assert listed['records'][0]['payload_tree_mode']=='120000' and listed['snapshot_commit_verified'] is False
