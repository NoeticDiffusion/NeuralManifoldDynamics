"""Regression tests for CSD-aware EEG preprocessing."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def test_preprocess_file_preserves_csd_as_eeg_modality(tmp_path, monkeypatch):
    """CSD-transformed scalp EEG should still be exported as modality 'eeg'."""
    mne = pytest.importorskip("mne")

    from mndm.preprocess import preprocess_file
    import mndm.preprocess as preprocess_mod

    sfreq = 100.0
    times = np.arange(0, 4.0, 1.0 / sfreq)
    ch_names = ["F3", "F4", "C3", "C4"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    data = np.vstack([np.sin(2 * np.pi * (8 + i) * times) for i in range(len(ch_names))])
    raw = mne.io.RawArray(data, info)

    raw_path = tmp_path / "test_raw_csd.fif"
    raw.save(raw_path, overwrite=True)

    ds_dir = tmp_path / "dsTEST"
    ds_dir.mkdir(exist_ok=True)
    file_path = ds_dir / raw_path.name
    raw_path.replace(file_path)

    csd_applied = {"value": False}

    def _fake_csd(in_raw, **_kwargs):
        csd_applied["value"] = True
        return in_raw

    original_pick_types = preprocess_mod.mne.pick_types

    def _pick_types_with_csd(info, **kwargs):
        if not csd_applied["value"]:
            return original_pick_types(info, **kwargs)
        if kwargs.get("csd", False):
            return np.arange(len(info["ch_names"]), dtype=int)
        if kwargs.get("eeg", False) and not kwargs.get("seeg", False) and not kwargs.get("ecog", False):
            return np.array([], dtype=int)
        return original_pick_types(info, **kwargs)

    monkeypatch.setattr(preprocess_mod, "compute_current_source_density", _fake_csd)
    monkeypatch.setattr(preprocess_mod.mne, "pick_types", _pick_types_with_csd)

    config = {
        "datasets": ["dsTEST"],
        "preprocess": {
            "sfreq": sfreq,
            "eeg_csd": {
                "enabled": True,
                "min_eeg_channels": 4,
                "lambda2": 1e-5,
                "stiffness": 4.0,
                "n_legendre_terms": 50,
                "on_error": "raise",
            },
        },
    }

    result = preprocess_file(file_path, config)
    assert "eeg" in result.signals
    assert result.signals["eeg"].shape[0] == 4
    assert result.channels["eeg"] == ch_names
