"""Tests for ECG/PPG/pupil anchor feature surfaces."""

from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def _basic_epoch_config() -> dict:
    return {
        "datasets": ["dsX"],
        "epoching": {"length_s": 4.0, "step_s": 4.0},
        "features": {},
    }


def test_compute_ecg_features_exports_hr_and_quality():
    """ECG extractor should emit heart-rate and QC columns."""
    from mndm.features.ecg import compute_ecg_features

    sfreq = 100.0
    n_samples = int(12 * sfreq)
    ecg = np.zeros(n_samples, dtype=np.float32)
    peak_idx = np.arange(50, n_samples, 100)
    ecg[peak_idx] = 5.0

    df = compute_ecg_features(
        {
            "signals": {"ecg": ecg[None, :]},
            "sfreq": sfreq,
            "dataset_id": "dsX",
        },
        _basic_epoch_config(),
    )

    assert not df.empty
    assert {"ecg_hr_bpm", "ecg_rr_mean", "ecg_rmssd", "ecg_quality_score", "qc_ok_ecg"}.issubset(df.columns)
    assert np.isfinite(df["ecg_hr_bpm"]).any()


def test_compute_ecg_features_exports_superwindow_hrv_columns_when_enabled():
    """ECG extractor should emit aligned HRV v0.1 superwindow features when enabled."""
    from mndm.features.ecg import compute_ecg_features

    sfreq = 100.0
    n_samples = int(24 * sfreq)
    ecg = np.zeros(n_samples, dtype=np.float32)
    peak_idx = np.arange(50, n_samples, 100)
    ecg[peak_idx] = 5.0

    cfg = _basic_epoch_config()
    cfg["features"] = {
        "ecg": {
            "hrv": {
                "enabled": True,
                "superwindow_s": 8.0,
                "window_mode": "centered",
                "min_nn_intervals": 4,
                "min_coverage_fraction": 0.5,
                "max_artifact_fraction": 0.25,
                "pnn50_threshold_ms": 50.0,
            }
        }
    }
    df = compute_ecg_features(
        {
            "signals": {"ecg": ecg[None, :]},
            "sfreq": sfreq,
            "dataset_id": "dsX",
        },
        cfg,
    )

    assert not df.empty
    assert {
        "ecg_hrv_hr_mean_bpm",
        "ecg_hrv_ibi_mean_ms",
        "ecg_hrv_sdnn_ms",
        "ecg_hrv_rmssd_ms",
        "ecg_hrv_pnn50",
        "ecg_hrv_nn_count",
        "ecg_hrv_artifact_fraction",
        "ecg_hrv_coverage_fraction",
        "ecg_hrv_quality_score",
        "qc_ok_ecg_hrv",
    }.issubset(df.columns)
    assert np.isfinite(df["ecg_hrv_hr_mean_bpm"]).any()
    assert (df["ecg_hrv_nn_count"] >= 0).all()


def test_compute_ppg_features_exports_rate_amplitude_and_quality():
    """PPG extractor should emit pulse-rate and amplitude columns."""
    from mndm.features.ppg import compute_ppg_features

    sfreq = 50.0
    t = np.arange(0, 12.0, 1.0 / sfreq)
    pulse = np.maximum(0.0, np.sin(2 * np.pi * 1.2 * t)) ** 2

    df = compute_ppg_features(
        {
            "signals": {"ppg": pulse[None, :].astype(np.float32)},
            "sfreq": sfreq,
            "dataset_id": "dsX",
        },
        _basic_epoch_config(),
    )

    assert not df.empty
    assert {"ppg_rate_bpm", "ppg_amplitude_mean", "ppg_quality_score", "qc_ok_ppg"}.issubset(df.columns)
    assert np.isfinite(df["ppg_amplitude_mean"]).any()


def test_preprocess_pupil_table_and_compute_pupil_features(tmp_path: Path):
    """Pupil TSVs should preprocess into signals and emit aligned features."""
    from mndm.features.pupil import compute_pupil_features
    from mndm.preprocess import preprocess_pupil_table

    sfreq = 10.0
    t = np.arange(0, 12.0, 1.0 / sfreq)
    pupil = 4.0 + 0.5 * np.sin(2 * np.pi * 0.2 * t)
    pupil[10:15] = np.nan

    table_path = tmp_path / "sub-001_task-memory_pupil.tsv"
    table_path.write_text(
        "time\tpupil_diameter\n"
        + "\n".join(f"{tt:.3f}\t{val}" for tt, val in zip(t, pupil)),
        encoding="utf-8",
    )

    pre = preprocess_pupil_table(table_path, {"datasets": ["dsX"]})
    assert "pupil" in pre.signals
    assert pre.signals["pupil"].shape[1] == len(t)

    df = compute_pupil_features(
        {
            "signals": pre.signals,
            "sfreq": pre.sfreq,
            "dataset_id": "dsX",
            "file_path": str(table_path),
        },
        _basic_epoch_config(),
    )

    assert not df.empty
    assert {"pupil_diameter_mean", "pupil_blink_fraction", "pupil_quality_score", "qc_ok_pupil"}.issubset(df.columns)
    assert ((df["pupil_quality_score"] >= 0.0) & (df["pupil_quality_score"] <= 1.0)).all()
