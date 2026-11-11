"""Regression: unsupported SC is a completed auditable result, not a crash."""
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def test_all_sc_windows_unsupported_completes_with_nulls(tmp_path, monkeypatch):
    repo=Path(__file__).resolve().parents[2]
    script=repo/'project/new_mnps_v.3.2/science_lead/sl025b3_precision_20261028_supportfix.py'
    spec=importlib.util.spec_from_file_location('sl025b3_supportfix',script)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    ar=tmp_path/'ar'; ar.mkdir(); native=tmp_path/'native'; native.mkdir(); (native/'participants').mkdir()
    out=tmp_path/'result'; payload=tmp_path/'payload'; header=tmp_path/'header'; channels=tmp_path/'channels'
    with payload.open('wb') as f:f.truncate(150000*61*4)
    header.write_text('header'); channels.write_text('channels')
    digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    payload_hash=digest(payload)
    labels=['Fz','Cz','Pz']+[f'Ch{i}' for i in range(58)]
    people=[f'sub-{i:02d}' for i in range(12)]
    monkeypatch.setattr(m,'AR',ar);monkeypatch.setattr(m,'NATIVE',native);monkeypatch.setattr(m,'OUT',out)
    monkeypatch.setattr(m.native,'source_paths',lambda *args:{'payload':payload,'header':header,'channels':channels})
    monkeypatch.setattr(m.native,'parse_brainvision_header',lambda p:(labels,{'NumberOfChannels':'61','SamplingInterval':'2000'}))
    monkeypatch.setattr(m.native,'load_channels',lambda p:(labels,[]))
    monkeypatch.setattr(m.np,'memmap',lambda *args,**kwargs:np.zeros((10000,61),dtype=np.float32))
    monkeypatch.setattr(m,'controls',lambda:None)
    ar_source=ar/'source';ar_source.write_text('fixture')
    def fake_ar(_):
        (out/'ar_fixture.txt').write_text('synthetic AR input, not analyzed')
        return ar_source
    monkeypatch.setattr(m,'run_ar',fake_ar)
    (ar/'run_manifest.json').write_text(json.dumps({'roster':{'participant_ids':people}}))
    inputs=[]
    for p in people[:4]:
        path=native/'participants'/f'{p}.jsonl.gz'
        with gzip.open(path,'wt',encoding='utf-8') as f:
            for s in m.SESSIONS:
                for c in m.CONDS:
                    rid=f'{p}_{s}_task-{c}'
                    record=dict(record_type='recording_manifest',participant_id=p,session_id=s,task_id=c,
                        payload_frame_support_valid=True,raw_payload_sha256=payload_hash,
                        source_header_labels=labels,ordered_measurement_labels=labels,
                        channel_order_sha256=m.native.digest_labels(labels),declared_reference_id='fixture',
                        reference_set_sha256='',metadata_sha256={'header':digest(header),'channels':digest(channels)})
                    f.write(json.dumps(record)+'\n')
                    for ch in m.CHANNELS:
                        record=dict(record_type='measurement',family_id='SC',participant_id=p,session_id=s,task_id=c,
                            recording_id=rid,window_id='20s-000000-010000',start_sample=0,end_sample_exclusive=10000,
                            source_payload_sha256=payload_hash,valid=False,reason='synthetic_unsupported',
                            details={'channel_label':ch,'channel_order_index':labels.index(ch)})
                        f.write(json.dumps(record)+'\n')
        inputs.append({'path':f'participants/{p}.jsonl.gz','sha256':digest(path)})
    (native/'run_manifest.json').write_text(json.dumps({'source_roster':{'frozen_participant_ids':people},'outputs':inputs}))
    m.main()
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['status']=='complete_exploratory' and manifest['qualified'] is False
    for s in manifest['sc_support_summaries']:
        assert s['requested_channel_windows']==72 and s['supported_channel_windows']==0 and s['unsupported_channel_windows']==72
        assert s['supported_session_pairs']==0 and s['median_seed_sd'] is None and s['rms_session_difference'] is None
    with (out/'sc_support.csv').open(newline='') as f: support=list(csv.DictReader(f))
    with (out/'sc_replicates.csv').open(newline='') as f: replicates=list(csv.DictReader(f))
    assert len(support)==72 and all(r['valid']=='False' and r['reason']=='prior_sc_unsupported:synthetic_unsupported' for r in support)
    assert len(replicates)==1152 and all(r['value']=='' and r['valid']=='False' for r in replicates)
    for name,expected in manifest['output_sha256'].items():assert digest(out/name)==expected
