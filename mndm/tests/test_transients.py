"""Tests for transient detection helper."""

from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def test_detect_transients_empty_bool_contract():
    """Test detect transients empty bool contract."""
    from mndm.transients import detect_transients

    out = detect_transients([], z_thresh=3.0, pad_epochs=1)
    assert out.dtype == bool
    assert out.shape == (0,)


def test_detect_transients_nan_safe_and_padding():
    """Test detect transients nan safe and padding."""
    from mndm.transients import detect_transients

    x = np.array([0.0, np.nan, 0.0, 0.1, 8.0, 0.0, 0.0], dtype=float)
    out = detect_transients(x, z_thresh=3.0, pad_epochs=1)
    assert out.shape == x.shape
    # NaN position itself should not be auto-flagged.
    assert not bool(out[1])
    # Spike + one-step padding on each side.
    assert out[3] and out[4] and out[5]


def test_detect_transients_accepts_iterables():
    """Test detect transients accepts iterables."""
    from mndm.transients import detect_transients

    seq = (v for v in [0.0, 0.1, 0.2, 5.0, 0.2, 0.1, 0.0])
    out = detect_transients(seq, z_thresh=3.0, pad_epochs=0)
    assert out.shape == (7,)
    assert out[3]
