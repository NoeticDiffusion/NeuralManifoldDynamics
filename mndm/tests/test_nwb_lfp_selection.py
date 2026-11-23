"""Selection policy tests for dual EEG/LFP NWB recordings."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


class _Series:
    def __init__(self) -> None:
        self.data = object()
        self.electrodes = object()


class _NWB:
    def __init__(self) -> None:
        self.acquisition = {"ElectricalSeriesEEG": _Series(), "LFP": _Series()}
        self.processing = {}


def test_lfp_keyword_and_explicit_path_override_eeg_candidate() -> None:
    from mndm.preprocess import _find_nwb_electrical_series

    nwb = _NWB()
    path, _ = _find_nwb_electrical_series(nwb, {"prefer_series_keywords": ["lfp"]})
    assert path == "acquisition/LFP"

    path, _ = _find_nwb_electrical_series(
        nwb,
        {"series_path": "acquisition/LFP", "prefer_series_keywords": ["eeg"]},
    )
    assert path == "acquisition/LFP"


def test_overlap_trim_nwb_resampling_matches_continuous_reference() -> None:
    from types import SimpleNamespace

    import numpy as np
    from scipy.signal import resample_poly

    from mndm.preprocess import _nwb_chunked_downsample_overlap_trim

    source_sfreq = 2500.0
    target_sfreq = 500.0
    n_samples = 300_009
    times = np.arange(n_samples, dtype=np.float64) / source_sfreq
    data = np.column_stack(
        [
            (1.0 + 0.35 * np.sin(2 * np.pi * 0.2 * times + 0.31))
            * np.sin(2 * np.pi * 3.5 * times + 0.73),
            np.sin(2 * np.pi * 11.0 * times - 0.2),
        ]
    )
    series = SimpleNamespace(data=data)
    start_sample = 7
    stop_sample = 300_008

    actual = _nwb_chunked_downsample_overlap_trim(
        series,
        source_sfreq=source_sfreq,
        target_sfreq=target_sfreq,
        channel_indices=[0, 1],
        chunk_sec=60.0,
        start_sample=start_sample,
        stop_sample=stop_sample,
    )
    reference = resample_poly(
        data[start_sample:stop_sample, :].T,
        up=1,
        down=5,
        axis=1,
    )

    np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)


def test_experimental_nwb_timebase_preserves_origin_and_rejects_gaps() -> None:
    from types import SimpleNamespace

    import numpy as np
    import pytest

    from mndm.preprocess import _nwb_series_timebase_summary

    timestamps = 13.0 + np.arange(20, dtype=np.float64) / 100.0
    series = SimpleNamespace(data=np.zeros((20, 2)), timestamps=timestamps)
    summary = _nwb_series_timebase_summary(
        series,
        source_sfreq=100.0,
        start_sample=2,
        stop_sample=18,
    )
    assert summary["clock_source"] == "explicit_timestamps"
    assert summary["segment_count"] == 1
    assert summary["segments"][0]["segment_id"] == "segment-0000"
    assert summary["selected_time_start_sec"] == pytest.approx(timestamps[2])
    assert summary["selected_time_stop_sec"] == pytest.approx(timestamps[17])

    gap_timestamps = timestamps.copy()
    gap_timestamps[10:] += 0.02
    gap_series = SimpleNamespace(data=np.zeros((20, 2)), timestamps=gap_timestamps)
    with pytest.raises(ValueError, match="timestamp discontinuity"):
        _nwb_series_timebase_summary(gap_series, source_sfreq=100.0)
