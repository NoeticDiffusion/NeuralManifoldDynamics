import numpy as np
from scipy.signal import resample_poly


def test_256_to_500_hz_sample_count_and_sinusoid_interior():
    source=np.sin(2*np.pi*10*np.arange(8*256)/256)
    target=resample_poly(source,125,64)
    expected=np.sin(2*np.pi*10*np.arange(8*500)/500)
    assert target.shape==(4000,)
    assert np.sqrt(np.mean((target[250:-250]-expected[250:-250])**2))<.002
