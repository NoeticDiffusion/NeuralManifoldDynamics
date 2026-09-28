"""Regression tests for quality-support intersections after modality joins."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def _handlers():
    def eeg_features(signals, config):
        return pd.DataFrame({"epoch_id": [0, 2], "eeg_alpha": [1.0, 2.0]})

    def eog_features(signals, config):
        return pd.DataFrame({"epoch_id": [0, 1, 2], "eog_blink_rate": [0.1, 9.9, 0.2]})

    return {"eeg": eeg_features, "eog": eog_features}


def _preprocessed(policy):
    return SimpleNamespace(
        signals={"eeg": np.zeros((2, 48)), "eog": np.zeros((1, 48))},
        sfreq=4.0,
        channels={"eeg": ["C3", "C4"], "eog": ["EOG"]},
        meta={"file": "sub-01_task-sleep_eeg.edf", "quality_interval_policy": policy},
    )


def test_enabled_quality_policy_prevents_auxiliary_outer_join_resurrection(monkeypatch):
    from mndm import parallel

    monkeypatch.setattr(parallel, "_get_modality_handlers", _handlers)
    merged, _ = parallel._compute_features_from_preprocessed(
        _preprocessed({"enabled": True, "mask_bad_annotations": True}),
        {},
    )

    assert merged["epoch_id"].tolist() == [0, 2]
    assert merged["eog_blink_rate"].tolist() == [0.1, 0.2]
    assert merged.attrs["quality_interval_epoch_intersection"] == {
        "status": "applied",
        "required_modality": "eeg",
        "rows_before": 3,
        "rows_after": 2,
        "valid_eeg_epoch_count": 2,
        "excluded_epoch_ids": [1],
        "withheld_modalities": [],
        "reason": "masked_quality_support_must_not_be_reintroduced_by_outer_modality_join",
    }


def test_disabled_quality_policy_preserves_existing_outer_join(monkeypatch):
    from mndm import parallel

    monkeypatch.setattr(parallel, "_get_modality_handlers", _handlers)
    merged, _ = parallel._compute_features_from_preprocessed(
        _preprocessed({"enabled": False, "mask_bad_annotations": True}),
        {},
    )

    assert set(merged["epoch_id"].tolist()) == {0, 1, 2}
    assert len(merged) == 3
    assert "quality_interval_epoch_intersection" not in merged.attrs


def test_enabled_quality_policy_withholds_unreviewed_extractor_on_masked_signal(monkeypatch):
    from mndm import parallel

    calls = []

    def eeg_features(signals, config):
        return pd.DataFrame({"epoch_id": [0, 2], "eeg_alpha": [1.0, 2.0]})

    def ecg_features(signals, config):
        calls.append("ecg")
        return pd.DataFrame({"epoch_id": [0, 1, 2], "ecg_hr_bpm": [60.0, 70.0, 80.0]})

    def ppg_features(signals, config):
        calls.append("ppg")
        return pd.DataFrame({"epoch_id": [0, 2], "ppg_rate_bpm": [72.0, 73.0]})

    monkeypatch.setattr(
        parallel,
        "_get_modality_handlers",
        lambda: {"eeg": eeg_features, "ecg": ecg_features, "ppg": ppg_features},
    )
    preprocessed = _preprocessed({"enabled": True, "mask_bad_annotations": True})
    preprocessed.signals["ecg"] = np.zeros((1, 48))
    preprocessed.signals["ecg"][:, 16:32] = np.nan
    preprocessed.signals["ppg"] = np.zeros((1, 48))
    preprocessed.signals["ppg"][:, 16:32] = np.nan

    merged, _ = parallel._compute_features_from_preprocessed(preprocessed, {})

    assert calls == ["ppg"]
    assert merged["epoch_id"].tolist() == [0, 2]
    assert "ecg_hr_bpm" not in merged.columns
    assert merged["ppg_rate_bpm"].tolist() == [72.0, 73.0]
    assert merged.attrs["quality_interval_epoch_intersection"]["withheld_modalities"] == [
        {"modality": "ecg", "reason": "feature_extractor_not_gap_safe"}
    ]


def test_phase_anchor_is_withheld_when_an_optional_input_has_masked_support(monkeypatch):
    from mndm import parallel

    calls = []

    def eeg_features(signals, config):
        return pd.DataFrame({"epoch_id": [0, 1, 2], "eeg_alpha": [1.0, 1.5, 2.0]})

    def phase_anchor_features(signals, config):
        calls.append("phase_anchor")
        return pd.DataFrame({"epoch_id": [0, 1, 2], "phi_resp_mean": [0.0, 1.0, 2.0]})

    monkeypatch.setattr(
        parallel,
        "_get_modality_handlers",
        lambda: {"eeg": eeg_features, "phase_anchor": phase_anchor_features},
    )
    preprocessed = _preprocessed({"enabled": True, "mask_bad_annotations": True})
    preprocessed.signals["ecg"] = np.ones((1, 48))
    preprocessed.signals["ecg"][:, 10:12] = np.nan

    merged, _ = parallel._compute_features_from_preprocessed(preprocessed, {})

    assert calls == []
    assert "phi_resp_mean" not in merged.columns
    assert merged.attrs["quality_interval_epoch_intersection"]["withheld_modalities"] == [
        {"modality": "phase_anchor", "reason": "feature_extractor_not_gap_safe"}
    ]
