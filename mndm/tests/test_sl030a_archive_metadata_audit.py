import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pytest
from scipy.io import savemat

path = Path(__file__).resolve().parents[2] / "project/new_mnps_v.3.2/science_lead/sl030a_archive_metadata_audit_20261029.py"
spec = spec_from_file_location("sl030a_archive", path)
mod = module_from_spec(spec)
spec.loader.exec_module(mod)


def build(folder):
    assignments=np.empty((16,12),dtype=object)
    for song in range(16):
        for j in range(12): assignments[song,j]=f'S{song*12+j+1}'
    savemat(folder/'stimAssignment.mat',{'stimAssignment':assignments})
    with ZipFile(folder/'CleanEEG_aggregatedByStimulus_orig.zip','w') as z:
        for song in range(21,25):
            for listen in 'ab':
                name=f'song{song}_{listen}_Imputed.mat'; p=folder/name
                ids=assignments[song-21:song-20]
                savemat(p,{f'data{song}_{listen}':np.zeros((125,20,12)), f'subs{song}_{listen}':ids, 'fs':125})
                z.write(p,'CleanEEG_aggregatedByStimulus_orig/'+name)
                p.unlink()


def test_extracts_eight_pairs_and_records_unknowns(tmp_path):
    build(tmp_path);out=tmp_path/'report.json';mod.main(tmp_path,out)
    result=json.loads(out.read_text())
    assert len(result['files'])==8
    assert all(x['same_ordered_ids'] and x['both_match_assignment_order'] for x in result['pairs'])
    assert not result['audio_alignment_verified']
    assert not result['qualified']


def test_does_not_overwrite_unmatched_existing_file(tmp_path):
    build(tmp_path)
    (tmp_path/'song21_a_Imputed.mat').write_bytes(b'unrelated data')
    with pytest.raises(ValueError,match='mismatches archive'):
        mod.main(tmp_path,tmp_path/'result.json')
    assert (tmp_path/'song21_a_Imputed.mat').read_bytes()==b'unrelated data'
