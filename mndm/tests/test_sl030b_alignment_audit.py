from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import numpy as np
import pytest

path=Path(__file__).resolve().parents[2]/'project/new_mnps_v.3.2/science_lead/sl030b_alignment_audit_20261029.py'
spec=spec_from_file_location('sl030b_alignment',path)
mod=module_from_spec(spec);spec.loader.exec_module(mod)


def test_known_shift_and_undefined_channel(monkeypatch):
    monkeypatch.setattr(mod,'INTERVALS',[(1,5)])
    rng=np.random.default_rng(123); x=rng.standard_normal((129,100))
    selected=x[np.r_[np.arange(124),128],:]
    clean=selected-selected.mean(axis=0,keepdims=True)
    row=mod.score_offsets(x,clean,0,5)[0]
    assert row['best_offset_samples']==0
    assert row['offset_scores'][4]['median_channel_r']==pytest.approx(1.0)
    clean[0,:]=1
    assert mod.score_offsets(x,clean,0,5)[0]['offset_scores'][4]['valid_channels']==124


def test_short_raw_is_unsupported(monkeypatch):
    monkeypatch.setattr(mod,'INTERVALS',[(0,3)])
    with pytest.raises(ValueError,match='raw_interval_not_supported'):
        mod.score_offsets(np.ones((129,6)),np.ones((125,15)),0,5)
