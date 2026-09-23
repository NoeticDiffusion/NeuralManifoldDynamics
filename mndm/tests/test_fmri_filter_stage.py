"""P0.4: one canonical fMRI session bandpass in fmri_continuous."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("scipy")

from mndm.features.fmri_continuous import (
    FILTER_STAGE_CONTINUOUS,
    FILTER_STAGE_DEFERRED,
    _bandpass_signal,
    fmri_filter_attrs_from_frame,
    preprocess_filter_meta,
    process_session_signals,
)


def _roi(n_regions: int = 3, n_times: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_regions, n_times))


def test_process_session_signals_applies_one_bandpass():
    roi = _roi()
    sfreq = 1.0
    once = process_session_signals(roi, sfreq, {"bandpass": [0.01, 0.1]})
    expected = _bandpass_signal(roi, sfreq, 0.01, 0.1)
    np.testing.assert_allclose(once["filtered_ts"], expected)
    assert once["filter_applied"] is True
    assert once["filter_skipped"] is False
    assert once["filter_order"] == 4
    assert once["filter_stage"] == FILTER_STAGE_CONTINUOUS
    twice = _bandpass_signal(expected, sfreq, 0.01, 0.1)
    assert not np.allclose(once["filtered_ts"], twice)


def test_skip_bandpass_does_not_filter_again():
    roi = _roi()
    sfreq = 1.0
    first = process_session_signals(roi, sfreq, {"bandpass": [0.01, 0.1]})
    second = process_session_signals(
        first["filtered_ts"],
        sfreq,
        {"bandpass": [0.01, 0.1], "skip_bandpass": True, "filter_applied": True},
    )
    np.testing.assert_allclose(second["filtered_ts"], first["filtered_ts"])
    assert second["filter_skipped"] is True
    assert second["filter_applied"] is True


def test_dataset_bandpass_in_preprocess_fmri_does_not_change_filtered_ts():
    """After P0.4, overlay preprocess.fmri.datasets.*.bandpass is not a second filter."""
    from mndm.features.fmri import compute_fmri_features

    roi = _roi(n_regions=4, n_times=80)
    sfreq = 1.0
    signals = {"signals": {"fmri": roi}, "sfreq": sfreq}
    base = {
        "features": {"fmri": {"window_sec": 10.0, "step_sec": 10.0}},
        "preprocess": {"fmri_bandpass": [0.01, 0.1]},
    }
    with_overlay = {
        **base,
        "preprocess": {
            "fmri_bandpass": [0.01, 0.1],
            "fmri": {"datasets": {"dsTEST": {"bandpass": [0.01, 0.1]}}},
        },
    }
    out_a = compute_fmri_features(signals, base)
    out_b = compute_fmri_features(signals, with_overlay)
    np.testing.assert_allclose(out_a["fmri_signal_power"], out_b["fmri_signal_power"])
    assert set(out_a["fmri_filter_stage"]) == {FILTER_STAGE_CONTINUOUS}
    assert np.all(out_a["fmri_filter_applied"] == 1)


def test_already_filtered_meta_skips_continuous_bandpass():
    from mndm.features.fmri import compute_fmri_features

    roi = _roi(n_regions=4, n_times=80)
    sfreq = 1.0
    filtered = _bandpass_signal(roi, sfreq, 0.01, 0.1)
    signals = {
        "signals": {"fmri": filtered},
        "sfreq": sfreq,
        "meta": {
            "filter_applied": True,
            "filter_stage": "preprocess_fmri",
            "filter_bandpass": [0.01, 0.1],
        },
    }
    config = {
        "features": {"fmri": {"window_sec": 10.0, "step_sec": 10.0}},
        "preprocess": {"fmri_bandpass": [0.01, 0.1]},
    }
    out = compute_fmri_features(signals, config)
    assert set(out["fmri_filter_stage"].astype(str)) == {"preprocess_fmri"}
    assert np.allclose(out["fmri_filter_bandpass_low"], 0.01)
    assert np.allclose(out["fmri_filter_bandpass_high"], 0.1)
    # Power from already-filtered input must match a single continuous pass on raw.
    once = compute_fmri_features(
        {"signals": {"fmri": roi}, "sfreq": sfreq},
        config,
    )
    np.testing.assert_allclose(out["fmri_signal_power"], once["fmri_signal_power"], rtol=1e-5, atol=1e-6)


def test_preprocess_filter_meta_records_ignored_overlay_bandpass():
    meta = preprocess_filter_meta({"bandpass": [0.008, 0.09]})
    assert meta["filter_applied"] is False
    assert meta["filter_stage"] == FILTER_STAGE_DEFERRED
    assert meta["filter_ignored_preprocess_bandpass"] == [0.008, 0.09]
    empty = preprocess_filter_meta({})
    assert empty["filter_ignored_preprocess_bandpass"] is None


def test_maybe_apply_preprocess_bandpass_is_identity():
    from mndm.features.fmri_continuous import maybe_apply_preprocess_bandpass

    roi = _roi()
    out, meta = maybe_apply_preprocess_bandpass(roi, {"bandpass": [0.008, 0.09]})
    np.testing.assert_array_equal(out, roi)
    assert meta["filter_applied"] is False
    assert meta["filter_ignored_preprocess_bandpass"] == [0.008, 0.09]


def test_skip_path_records_upstream_bandpass_not_common_default():
    from mndm.features.fmri import compute_fmri_features

    roi = _roi(n_regions=4, n_times=80)
    out = compute_fmri_features(
        {
            "signals": {"fmri": roi},
            "sfreq": 1.0,
            "meta": {
                "filter_applied": True,
                "filter_stage": "preprocess_fmri",
                "filter_bandpass": [0.008, 0.09],
            },
        },
        {
            "features": {"fmri": {"window_sec": 10.0, "step_sec": 10.0}},
            "preprocess": {"fmri_bandpass": [0.01, 0.1]},
        },
    )
    assert np.allclose(out["fmri_filter_bandpass_low"], 0.008)
    assert np.allclose(out["fmri_filter_bandpass_high"], 0.09)


def test_fmri_filter_attrs_from_frame_and_eeg_empty():
    assert fmri_filter_attrs_from_frame(pd.DataFrame({"feat_a": [1.0]})) == {}
    frame = pd.DataFrame(
        {
            "fmri_filter_stage": ["fmri_continuous"] * 2,
            "fmri_filter_applied": [1, 1],
            "fmri_filter_skipped": [0, 0],
            "fmri_filter_order": [4, 4],
            "fmri_filter_bandpass_low": [0.01, 0.01],
            "fmri_filter_bandpass_high": [0.1, 0.1],
        }
    )
    attrs = fmri_filter_attrs_from_frame(frame)
    assert attrs["filter_stage"] == FILTER_STAGE_CONTINUOUS
    assert attrs["filter_applied"] == 1
    assert attrs["filter_skipped"] == 0
    assert attrs["filter_order"] == 4
    assert attrs["filter_bandpass_low"] == pytest.approx(0.01)
    assert attrs["filter_bandpass_high"] == pytest.approx(0.1)


def test_short_session_skips_bandpass_with_explicit_provenance():
    roi = _roi(n_regions=2, n_times=8)
    out = process_session_signals(roi, sfreq=0.5, config={"bandpass": [0.01, 0.1]})
    np.testing.assert_allclose(out["filtered_ts"], roi)
    assert out["filter_applied"] is False
    assert out["filter_skipped"] is True
    assert out["filter_order"] == 0
    assert out["filter_stage"] == "skipped_too_short"
