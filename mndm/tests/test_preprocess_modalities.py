"""Focused modality-specific preprocess regressions."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_preprocess_file_ecg_notch_skips_empty_data_or_ica_path(tmp_path: Path):
    """ECG-first recordings should not fail notch filtering when only misc-like channels remain."""
    mne = pytest.importorskip("mne")

    from mndm.preprocess import preprocess_file

    sfreq = 100.0
    times = np.arange(0, 4.0, 1.0 / sfreq)
    data = np.vstack(
        [
            np.sin(2 * np.pi * 1.2 * times),
            0.5 * np.sin(2 * np.pi * 0.25 * times),
        ]
    )
    info = mne.create_info(ch_names=["Pulse", "Resp"], sfreq=sfreq, ch_types=["misc", "misc"])
    raw = mne.io.RawArray(data, info)

    ds_dir = tmp_path / "dsTEST" / "sub-001" / "ecg"
    ds_dir.mkdir(parents=True, exist_ok=True)
    file_path = ds_dir / "sub-001_task-rest_ecg.fif"
    raw.save(file_path, overwrite=True)

    result = preprocess_file(
        file_path,
        {
            "datasets": ["dsTEST"],
            "preprocess": {
                "sfreq": sfreq,
                "notch_hz": 25.0,
            },
        },
    )

    assert "ecg" in result.signals
    assert result.signals["ecg"].shape[1] == len(times)


def test_preprocess_file_meg_fif_extracts_meg_and_eeg_channels(tmp_path: Path):
    """Neuromag-style FIF files can expose both MEG and EEG channels to the pipeline."""
    mne = pytest.importorskip("mne")

    from mndm.preprocess import preprocess_file

    sfreq = 100.0
    times = np.arange(0, 4.0, 1.0 / sfreq)
    data = np.vstack(
        [
            np.sin(2 * np.pi * 8.0 * times),
            np.sin(2 * np.pi * 10.0 * times),
            np.sin(2 * np.pi * 12.0 * times),
            0.2 * np.sin(2 * np.pi * 1.0 * times),
            0.5 * np.sin(2 * np.pi * 1.5 * times),
        ]
    )
    info = mne.create_info(
        ch_names=["MEG0111", "MEG0112", "EEG001", "EOG001", "ECG001"],
        sfreq=sfreq,
        ch_types=["grad", "mag", "eeg", "eog", "ecg"],
    )
    raw = mne.io.RawArray(data, info)

    ds_dir = tmp_path / "ds003645" / "sub-002" / "meg"
    ds_dir.mkdir(parents=True, exist_ok=True)
    file_path = ds_dir / "sub-002_task-facerecognition_meg.fif"
    raw.save(file_path, overwrite=True)

    result = preprocess_file(
        file_path,
        {
            "datasets": ["ds003645"],
            "preprocess": {
                "sfreq": sfreq,
                "notch_hz": None,
                "eeg_bandpass": [1.0, 45.0],
                "meg_bandpass": [1.0, 45.0],
            },
        },
    )

    assert "meg" in result.signals
    assert "meg_mag" in result.signals
    assert "meg_grad" in result.signals
    assert "eeg" in result.signals
    assert result.signals["meg"].shape[0] == 2
    assert result.signals["meg_mag"].shape[0] == 1
    assert result.signals["meg_grad"].shape[0] == 1
    assert result.signals["eeg"].shape[0] == 1
