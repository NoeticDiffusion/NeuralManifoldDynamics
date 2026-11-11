import importlib.util
from pathlib import Path
import numpy as np

path = Path(__file__).resolve().parents[2] / 'project/new_mnps_v.3.2/science_lead/sl034_signal_pilot.py'
spec = importlib.util.spec_from_file_location('sl034_signal_pilot', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_edf_reader_record_interleaving(tmp_path):
    fixed = bytearray(b' ' * 256)
    for a, b, value in [(184, 192, '768'), (236, 244, '2'), (244, 252, '1'), (252, 256, '2')]:
        fixed[a:b] = value.ljust(b-a).encode()
    fields = [('A', 'B', 16), ('', '', 80), ('', '', 8), ('', '', 8), ('', '', 8),
              ('', '', 8), ('', '', 8), ('', '', 80), ('4', '2', 8), ('', '', 32)]
    header = b''.join(str(v).ljust(width).encode() for left, right, width in fields for v in (left, right))
    target = tmp_path / 'sample.edf'
    target.write_bytes(bytes(fixed) + header + np.array([0,1,2,3,10,11,4,5,6,7,12,13], dtype='<i2').tobytes())
    labels, rates, values, n, duration = module.edf_excerpt(target, 0.5, 1.0)
    assert labels == ['A', 'B'] and rates == [4, 2] and n == 2 and duration == 1
    assert values[0].tolist() == [2, 3, 4, 5]
    assert values[1].tolist() == [11, 12]
