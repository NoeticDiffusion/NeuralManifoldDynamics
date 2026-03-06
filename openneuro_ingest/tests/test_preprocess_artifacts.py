"""Tests for artifact config resolution and bad-channel detection."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def test_resolve_artifact_config_merges_overrides():
    # Import guarded to avoid hard dependency on mne/scipy at import time
    import importlib
    import pytest

    try:
        preprocess = importlib.import_module("openneuro.preprocess")
    except ModuleNotFoundError:
        pytest.skip("openneuro.preprocess (mne dependency) not available")

    _resolve_artifact_config = getattr(preprocess, "_resolve_artifact_config")

    cfg = {
        "preprocess": {
            "artifacts": {
                "method": "none",
                "ica_n_components": 10,
                "datasets": {
                    "dsTEST": {
                        "method": "eog_reg",
                        "ica_n_components": 20,
                    }
                },
            }
        }
    }

    base = _resolve_artifact_config(cfg, None)
    ds = _resolve_artifact_config(cfg, "dsTEST")

    assert base["method"] == "none"
    assert base["ica_n_components"] == 10

    # Dataset override should replace base values where provided
    assert ds["method"] == "eog_reg"
    assert ds["ica_n_components"] == 20


def test_detect_bad_eeg_channels_on_synthetic():
    mne = pytest.importorskip("mne")

    from openneuro.preprocess import _detect_bad_eeg_channels

    sfreq = 100.0
    times = np.arange(0, 10.0, 1.0 / sfreq)

    # Build 8 channels: 6 clean sinusoids, 1 flat, 1 high-variance noise
    n_channels = 8
    data = np.zeros((n_channels, times.size), dtype=float)
    ch_names = []
    for i in range(6):
        data[i] = np.sin(2 * np.pi * 10 * times + 0.1 * i)
        ch_names.append(f"F{i+1}")
    # Flat channel
    data[6] = 0.0
    ch_names.append("Fflat")
    # High-variance channel
    rng = np.random.default_rng(123)
    data[7] = 10.0 * rng.standard_normal(times.size)
    ch_names.append("Fnoise")

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info)

    bad_cfg = {
        "var_low_factor": 1e-4,
        "var_high_factor": 25.0,
        "corr_thresh": 0.2,
        "max_bad_fraction": 0.5,
        "min_good_channels": 4,
    }

    out = _detect_bad_eeg_channels(raw, bad_cfg)
    bad_channels = set(out["bad_channels"])

    # Expect at least the flat and high-variance channels to be flagged
    assert "Fflat" in bad_channels
    assert "Fnoise" in bad_channels


def test_preprocess_file_artifact_method_meta(tmp_path):
    """preprocess_file should record the selected artifact method in meta."""
    mne = pytest.importorskip("mne")

    from openneuro.preprocess import PreprocessedSignals, preprocess_file

    sfreq = 100.0
    times = np.arange(0, 2.0, 1.0 / sfreq)

    # Simple 2‑EEG + 1‑EOG synthetic recording
    data = np.vstack(
        [
            np.sin(2 * np.pi * 10 * times),  # EEG 1
            np.sin(2 * np.pi * 12 * times),  # EEG 2
            np.sign(np.sin(2 * np.pi * 1 * times)),  # EOG
        ]
    )
    ch_names = ["Fz", "Cz", "VEOG"]
    ch_types = ["eeg", "eeg", "eog"]

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)

    # Save to a temporary FIF file so preprocess_file can read it
    raw_path = tmp_path / "test_raw.fif"
    raw.save(raw_path, overwrite=True)

    config = {
        "preprocess": {
            "sfreq": sfreq,
            "artifacts": {
                "method": "eog_reg",
                "datasets": {
                    "dsTEST": {"method": "eog_reg"},
                },
            },
        },
        # Ensure dataset id inference finds dsTEST in the path
        "datasets": ["dsTEST"],
    }

    # Place file under a path containing dsTEST so _infer_dataset_id picks it up
    ds_dir = tmp_path / "dsTEST"
    ds_dir.mkdir(exist_ok=True)
    file_path = ds_dir / raw_path.name
    raw_path.replace(file_path)

    result = preprocess_file(file_path, config)
    assert isinstance(result, PreprocessedSignals)
    artifact_meta = result.meta.get("artifact", {})
    assert artifact_meta.get("method") == "eog_reg"

