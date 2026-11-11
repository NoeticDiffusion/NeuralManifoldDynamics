from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import pytest

p=Path(__file__).resolve().parents[2]/'project/new_mnps_v.3.2/science_lead/sl031_p0_ds006036_audit_20261030.py'
spec=spec_from_file_location('sl031_p0',p);mod=module_from_spec(spec);spec.loader.exec_module(mod)


def test_marker_grouping_does_not_merge_long_gap():
    groups=mod.clusters([0.,.2,.4,10.4])
    assert [g['count'] for g in groups]==[3,1]
    assert groups[0]['span_s']==pytest.approx(.4)
    assert groups[1]['median_interval_s'] is None


def test_empty_and_nonmonotonic():
    assert mod.clusters([])==[]
    with pytest.raises(ValueError,match='nonmonotonic'):
        mod.clusters([1.,.5])
