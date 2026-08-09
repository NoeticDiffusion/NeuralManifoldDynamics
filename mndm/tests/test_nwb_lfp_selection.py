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
