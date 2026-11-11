from fractions import Fraction
import math

import numpy as np
from scipy.signal import firwin, resample_poly

from mndm.mnps.v3_1.m1_windows import _source_dependency


def test_m1_resampler_dependency_bounds_cover_full_recording_polyphase_output():
    source_sfreq = 512.0
    target_sfreq = 250.0
    ratio = Fraction(target_sfreq / source_sfreq).limit_denominator(100_000)
    up, down = ratio.numerator, ratio.denominator
    half_len = 10 * max(up, down)
    taps = firwin(2 * half_len + 1, 1.0 / max(up, down), window=("kaiser", 5.0))
    source = np.random.default_rng(20260930).normal(size=60_000)
    full = resample_poly(source, up, down, window=taps, padtype="constant")

    target_start, target_end = 5_000, 10_188
    dependency_start, dependency_end = _source_dependency(
        target_start=target_start,
        target_end_exclusive=target_end,
        source_sfreq=source_sfreq,
        up=up,
        down=down,
        half_len=half_len,
    )
    crop_start = (dependency_start // down) * down
    cropped = resample_poly(
        source[crop_start:dependency_end],
        up,
        down,
        window=taps,
        padtype="constant",
    )
    local_start = target_start - crop_start * up // down
    local = cropped[local_start : local_start + target_end - target_start]

    assert dependency_start >= 0
    assert dependency_end <= source.size
    assert local.shape == (target_end - target_start,)
    assert np.allclose(local, full[target_start:target_end], rtol=1e-12, atol=1e-12)


def test_m1_source_dependency_marks_recording_edge_for_first_nominal_window():
    source_sfreq = 512.0
    ratio = Fraction(250.0 / source_sfreq).limit_denominator(100_000)
    up, down = ratio.numerator, ratio.denominator
    half_len = 10 * max(up, down)
    context_start = -94
    context_end = 5_000 + 94

    dependency_start, dependency_end = _source_dependency(
        target_start=context_start,
        target_end_exclusive=context_end,
        source_sfreq=source_sfreq,
        up=up,
        down=down,
        half_len=half_len,
    )

    assert dependency_start < 0
    assert dependency_end > math.ceil(20.0 * source_sfreq)
