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


def test_preprocess_file_eda_channel_native_gsr_type(tmp_path: Path):
    """EDA/GSR channels typed with MNE's native "gsr" type survive pruning and are collected."""
    mne = pytest.importorskip("mne")

    from mndm.preprocess import preprocess_file

    sfreq = 100.0
    times = np.arange(0, 4.0, 1.0 / sfreq)
    data = np.vstack(
        [
            np.sin(2 * np.pi * 8.0 * times),   # EEG-like
            2.0 + 0.1 * np.sin(2 * np.pi * 0.05 * times),  # EDA-like (slow drift)
            0.01 * np.sin(2 * np.pi * 0.2 * times),        # unrelated misc channel
        ]
    )
    info = mne.create_info(
        ch_names=["Cz", "EDA", "Sync"],
        sfreq=sfreq,
        ch_types=["eeg", "gsr", "misc"],
    )
    raw = mne.io.RawArray(data, info)

    ds_dir = tmp_path / "dsEDA" / "sub-001" / "eeg"
    ds_dir.mkdir(parents=True, exist_ok=True)
    file_path = ds_dir / "sub-001_task-rest_eeg.fif"
    raw.save(file_path, overwrite=True)

    result = preprocess_file(
        file_path,
        {
            "datasets": ["dsEDA"],
            "preprocess": {
                "sfreq": sfreq,
                "notch_hz": None,
            },
        },
    )

    assert "eda" in result.signals
    assert result.signals["eda"].shape[1] == len(times)
    assert result.channels.get("eda") == ["EDA"]
    # The unrelated "Sync" misc channel must not be swept up into "eda".
    assert "Sync" not in result.channels.get("eda", [])


def test_preprocess_file_eda_channel_legacy_misc_name_fallback(tmp_path: Path):
    """A dataset that (legacy-style) types EDA as "misc" is still picked up by name.

    Regression guard: "misc"-typed channels are dropped by the pre-resample
    channel prune for plain EEG recordings unless matched by the name-based
    fallback, since MNE has no native EDA type in older configs.
    """
    mne = pytest.importorskip("mne")

    from mndm.preprocess import preprocess_file

    sfreq = 100.0
    times = np.arange(0, 4.0, 1.0 / sfreq)
    data = np.vstack(
        [
            np.sin(2 * np.pi * 8.0 * times),
            2.0 + 0.1 * np.sin(2 * np.pi * 0.05 * times),
        ]
    )
    info = mne.create_info(
        ch_names=["Cz", "GSR1"],
        sfreq=sfreq,
        ch_types=["eeg", "misc"],
    )
    raw = mne.io.RawArray(data, info)

    ds_dir = tmp_path / "dsEDALegacy" / "sub-001" / "eeg"
    ds_dir.mkdir(parents=True, exist_ok=True)
    file_path = ds_dir / "sub-001_task-rest_eeg.fif"
    raw.save(file_path, overwrite=True)

    result = preprocess_file(
        file_path,
        {
            "datasets": ["dsEDALegacy"],
            "preprocess": {
                "sfreq": sfreq,
                "notch_hz": None,
            },
        },
    )

    assert "eda" in result.signals
    assert result.channels.get("eda") == ["GSR1"]


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
