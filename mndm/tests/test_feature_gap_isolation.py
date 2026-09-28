"""Feature filters must not borrow samples across unavailable temporal support."""

from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def _config() -> dict:
    return {
        "datasets": ["dsX"],
        "epoching": {"length_s": 30.0, "step_s": 30.0},
        "features": {},
    }


def test_ppg_filter_and_threshold_are_independent_across_nan_gap():
    from mndm.features.ppg import compute_ppg_features

    sfreq = 50.0
    time = np.arange(90 * int(sfreq), dtype=float) / sfreq
    ppg = np.sin(2.0 * np.pi * 1.2 * time) + 0.05 * np.sin(2.0 * np.pi * 0.2 * time)
    ppg[30 * 50 : 60 * 50] = np.nan
    altered = ppg.copy()
    altered[60 * 50 :] = 20.0 * altered[60 * 50 :] + 700.0

    baseline = compute_ppg_features(
        {"signals": {"ppg": ppg[None, :]}, "sfreq": sfreq, "dataset_id": "dsX"},
        _config(),
    )
    changed = compute_ppg_features(
        {"signals": {"ppg": altered[None, :]}, "sfreq": sfreq, "dataset_id": "dsX"},
        _config(),
    )

    assert baseline["epoch_id"].tolist() == [0, 2]
    assert changed["epoch_id"].tolist() == [0, 2]
    columns = ["ppg_rate_bpm", "ppg_pulse_count", "ppg_amplitude_mean", "ppg_amplitude_std"]
    np.testing.assert_allclose(
        baseline.loc[baseline.epoch_id == 0, columns].to_numpy(dtype=float),
        changed.loc[changed.epoch_id == 0, columns].to_numpy(dtype=float),
        equal_nan=True,
    )
    operations = {item["name"]: item for item in baseline.attrs["signal_support_provenance"]["operations"]}
    assert operations["ppg_bandpass_filtfilt"]["actual_kwargs"]["finite_segment_count"] == 2
    assert operations["ppg_peak_detection"]["parameters"]["threshold_fit_scope"] == (
        "per_contiguous_finite_segment_median_and_MAD"
    )


def test_eog_filter_and_threshold_are_independent_across_nan_gap():
    from mndm.features.eog import compute_eog_features

    sfreq = 50.0
    n_samples = 90 * int(sfreq)
    time = np.arange(n_samples, dtype=float) / sfreq
    veog = 0.1 * np.sin(2.0 * np.pi * 1.2 * time)
    for peak in np.arange(1, 90, 3) * int(sfreq):
        if peak < n_samples:
            veog[peak : peak + 4] += 1.0
    heog = 0.05 * np.sin(2.0 * np.pi * 0.4 * time)
    eog = np.vstack([veog, heog])
    eog[:, 30 * 50 : 60 * 50] = np.nan
    altered = eog.copy()
    altered[:, 60 * 50 :] = 15.0 * altered[:, 60 * 50 :] + 90.0

    baseline = compute_eog_features(
        {"signals": {"eog": eog}, "sfreq": sfreq, "dataset_id": "dsX"}, _config()
    )
    changed = compute_eog_features(
        {"signals": {"eog": altered}, "sfreq": sfreq, "dataset_id": "dsX"}, _config()
    )

    assert baseline["epoch_id"].tolist() == [0, 2]
    assert changed["epoch_id"].tolist() == [0, 2]
    columns = [
        "eog_blink_rate",
        "eog_blink_count",
        "eog_blink_amplitude",
        "eog_heog_saccade_rate",
        "eog_heog_saccade_amplitude",
        "eog_artifact_fraction",
    ]
    np.testing.assert_allclose(
        baseline.loc[baseline.epoch_id == 0, columns].to_numpy(dtype=float),
        changed.loc[changed.epoch_id == 0, columns].to_numpy(dtype=float),
        equal_nan=True,
    )
    operations = {item["name"]: item for item in baseline.attrs["signal_support_provenance"]["operations"]}
    assert operations["eog_veog_bandpass_filtfilt"]["actual_kwargs"]["finite_segment_count"] == 2


def test_emg_omits_epochs_touching_nan_support():
    from mndm.features.emg import compute_emg_features

    sfreq = 50.0
    time = np.arange(90 * int(sfreq), dtype=float) / sfreq
    emg = np.sin(2.0 * np.pi * 20.0 * time)
    emg[30 * 50 : 60 * 50] = np.nan
    result = compute_emg_features(
        {"signals": {"emg": emg[None, :]}, "sfreq": sfreq, "dataset_id": "dsX"},
        _config(),
    )

    assert result["epoch_id"].tolist() == [0, 2]


def test_blink_duration_is_clipped_to_its_finite_segment():
    from mndm.features.eog import _blink_duration_ms

    first = np.asarray([0.0, 0.0, 8.0, 10.0, 8.0, np.nan, 0.0, 0.0, 0.0])
    altered_after_gap = first.copy()
    altered_after_gap[6:] = 1000.0

    duration_a = _blink_duration_ms(
        first, 3, 10.0, 0.5, segment_start=0, segment_stop=5
    )
    duration_b = _blink_duration_ms(
        altered_after_gap, 3, 10.0, 0.5, segment_start=0, segment_stop=5
    )

    assert duration_a == duration_b
