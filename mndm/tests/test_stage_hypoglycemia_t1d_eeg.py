"""Tests for the hypoglycemia T1D EEG (Zenodo 3465213) BIDS-lite staging script."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pyedflib = pytest.importorskip("pyedflib")

_MODULE_PATH = Path(__file__).resolve().parents[2] / "project" / "smoke_tests" / "stage_hypoglycemia_t1d_eeg.py"
_SPEC = importlib.util.spec_from_file_location("stage_hypoglycemia_t1d_eeg", _MODULE_PATH)
stage_mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = stage_mod
_SPEC.loader.exec_module(stage_mod)


def _synthetic_subject(seed: int, n_samples: int = 400) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "eu": {"EEG": rng.normal(loc=0.0, scale=10.0, size=(stage_mod.N_CHANNELS, n_samples))},
        "hy": {"EEG": rng.normal(loc=0.0, scale=15.0, size=(stage_mod.N_CHANNELS, n_samples))},
    }


@pytest.fixture
def synthetic_subjects():
    return [_synthetic_subject(seed=1), _synthetic_subject(seed=2)]


def test_channel_names_have_19_entries_with_confirmed_posterior_set():
    assert len(stage_mod.CHANNEL_NAMES) == stage_mod.N_CHANNELS
    assert stage_mod.CONFIRMED_CHANNELS <= set(stage_mod.CHANNEL_NAMES)
    assert stage_mod.CHANNEL_NAMES[12:19] == ["T5", "T6", "P3", "P4", "Pz", "O1", "O2"]


def test_validate_subject_arrays_rejects_wrong_channel_count():
    bad_subject = {
        "eu": {"EEG": np.zeros((5, 400))},
        "hy": {"EEG": np.zeros((stage_mod.N_CHANNELS, 400))},
    }
    with pytest.raises(ValueError, match="expected 19 channels"):
        stage_mod._validate_subject_arrays(1, bad_subject)


def test_validate_subject_arrays_rejects_too_short_recording():
    short_subject = {
        "eu": {"EEG": np.zeros((stage_mod.N_CHANNELS, 10))},
        "hy": {"EEG": np.zeros((stage_mod.N_CHANNELS, 400))},
    }
    with pytest.raises(ValueError, match="too short"):
        stage_mod._validate_subject_arrays(1, short_subject)


def test_write_condition_edf_roundtrips_data_and_channel_names(tmp_path):
    data_uv = np.random.default_rng(0).normal(scale=20.0, size=(stage_mod.N_CHANNELS, 400))
    edf_path = tmp_path / "sub-001_task-eu_eeg.edf"

    stage_mod.write_condition_edf(edf_path, data_uv)

    assert edf_path.exists()
    reader = pyedflib.EdfReader(str(edf_path))
    try:
        assert reader.signals_in_file == stage_mod.N_CHANNELS
        assert [reader.getSampleFrequency(i) for i in range(stage_mod.N_CHANNELS)] == [stage_mod.SFREQ_HZ] * stage_mod.N_CHANNELS
        assert reader.getSignalLabels() == list(stage_mod.CHANNEL_NAMES)
        recovered = np.stack([reader.readSignal(i) for i in range(stage_mod.N_CHANNELS)])
    finally:
        reader.close()

    # 16-bit digital quantization over a 1500 uV physical span => ~0.023 uV/bit.
    assert np.allclose(recovered, data_uv, atol=0.05)


def test_stage_dataset_writes_expected_bids_lite_layout(tmp_path, monkeypatch, synthetic_subjects):
    monkeypatch.setattr(stage_mod, "load_eegdata_struct", lambda mat_path: synthetic_subjects)
    dest_root = tmp_path / "staged"
    source_mat = tmp_path / "EEGdata.mat"  # never actually read; loader is monkeypatched

    manifest_df = stage_mod.stage_dataset(source_mat, dest_root)

    assert len(manifest_df) == len(synthetic_subjects) * 2
    assert set(manifest_df["condition"]) == {"eu", "hy"}
    assert set(manifest_df["participant_id"]) == {"sub-001", "sub-002"}

    for _, row in manifest_df.iterrows():
        edf_path = dest_root / row["edf_path"]
        assert edf_path.exists()
        assert edf_path.name == f"{row['participant_id']}_task-{row['condition']}_eeg.edf"

    participants = pd.read_csv(dest_root / "participants.tsv", sep="\t")
    assert sorted(participants["participant_id"]) == ["sub-001", "sub-002"]

    assert (dest_root / "dataset_description.json").exists()
    assert (dest_root / "hypoglycemia_t1d_manifest.tsv").exists()


def test_staged_filenames_are_discoverable_by_mndm_bids_index(tmp_path, monkeypatch, synthetic_subjects):
    """The staged layout must satisfy mndm.bids_index.build_file_index's BIDS-entity parsing."""
    sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
    from mndm import bids_index

    monkeypatch.setattr(stage_mod, "load_eegdata_struct", lambda mat_path: synthetic_subjects)
    dest_root = tmp_path / "staged"
    stage_mod.stage_dataset(tmp_path / "EEGdata.mat", dest_root)

    index_df = bids_index.build_file_index(dest_root, dataset_id=stage_mod.DATASET_ID)

    assert len(index_df) == len(synthetic_subjects) * 2
    assert set(index_df["subject"]) == {"001", "002"}
    assert set(index_df["task"]) == {"eu", "hy"}
    assert set(index_df["modality"]) == {"eeg"}
