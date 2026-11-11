import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import numpy as np
import pytest
from scipy.io import savemat

path = Path(__file__).resolve().parents[2] / "project/new_mnps_v.3.2/science_lead/sl030a_clean_pair_audit_20261029.py"
spec = spec_from_file_location("sl030a_clean", path)
mod = module_from_spec(spec)
spec.loader.exec_module(mod)


def test_unequal_order_is_not_equal_pairing_but_same_people(tmp_path):
    ids = np.array([[f"S{i}" for i in range(1, 13)]], dtype=object)
    a,b,assignment,out = [tmp_path / x for x in ("a.mat","b.mat","assignment.mat","result.json")]
    savemat(a,{"data21_a": np.zeros((2, 10, 12)), "subs21_a": ids, "fs": 125})
    savemat(b,{"data21_b": np.zeros((2, 10, 12)), "subs21_b": ids[:, ::-1], "fs": 125})
    savemat(assignment,{"stimAssignment": ids})
    mod.main(a,b,assignment,out)
    result=json.loads(out.read_text())
    assert result['same_ids_both_listens']
    assert not result['id_order_equal']
    assert result['both_match_assignment_set']
    assert not result['actual_audio_sample_alignment_verified']
    assert not result['scientific_qualification']


def test_duplicate_participant_rejected(tmp_path):
    p=tmp_path/'a.mat'
    savemat(p,{"data21_a": np.zeros((2, 10, 2)), "subs21_a": np.array([["S1","S1"]],dtype=object), "fs":125})
    with pytest.raises(ValueError,match="Duplicate"):
        mod.inspect(p,21,'a')
