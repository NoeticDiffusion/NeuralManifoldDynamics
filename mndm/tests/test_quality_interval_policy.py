"""Regression tests for opt-in quality-interval support handling."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

mne = pytest.importorskip("mne")


def test_final_sleep_overlay_resolves_cautious_quality_and_segmented_jacobian_policy():
    from core.config_loader import load_config

    config_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "sources"
        / "openneuro"
        / "config_ingest_ds005555_continuous_c123_cautious_gap_isolated.yaml"
    )
    config = load_config(config_path)
    assert config["mnps"]["jacobian"]["forbid_cross_gap"] is True
    assert set(config["preprocess"]["quality_interval_policy"]["datasets"]["ds005555"]["source_kinds"]) == {
        "raw_joint_eeg_flatline",
        "source_stage_hum_code_8",
    }
    assert config["epoching"]["datasets"]["ds005555"]["sampling"]["enabled"] is False

    gap_safe_path = config_path.with_name(
        "config_ingest_ds005555_continuous_c123_cautious_gap_safe.yaml"
    )
    gap_safe = load_config(gap_safe_path)
    assert gap_safe["paths"]["processed_dir"].endswith(
        "ds005555_continuous_c123_cautious_gap_safe_20260925"
    )
    assert gap_safe["mnps"]["jacobian"]["forbid_cross_gap"] is True

    final_path = gap_safe_path.with_name(
        "config_ingest_ds005555_continuous_c123_cautious_gap_safe_final.yaml"
    )
    final = load_config(final_path)
    assert final["paths"]["processed_dir"].endswith(
        "ds005555_continuous_c123_cautious_gap_safe_final_20260925"
    )


def _raw(data: np.ndarray, sfreq: float = 16.0):
    info = mne.create_info([f"EEG{i}" for i in range(data.shape[0])], sfreq, ch_types="eeg")
    return mne.io.RawArray(data.copy(), info, verbose="ERROR")


def test_quality_interval_policy_only_marks_selected_merged_source_intervals():
    from mndm.preprocess import _apply_quality_interval_annotations

    raw = _raw(np.zeros((2, 1600)))
    intervals = [
        {"source_kind": "exclude", "clock": "original_raw_seconds", "start_sec": 20.0, "end_sec": 25.0},
        {"source_kind": "exclude", "clock": "original_raw_seconds", "start_sec": 25.0, "end_sec": 30.0},
        {"source_kind": "qc_only", "clock": "original_raw_seconds", "start_sec": 30.0, "end_sec": 35.0},
    ]

    record = _apply_quality_interval_annotations(
        raw,
        intervals,
        {
            "enabled": True,
            "source_kinds": ["exclude"],
            "annotation_description": "BAD_acq_skip",
            "mask_bad_annotations": True,
            "segment_resampling": True,
        },
    )

    selected = [
        (float(onset), float(duration), str(description))
        for onset, duration, description in zip(
            raw.annotations.onset, raw.annotations.duration, raw.annotations.description
        )
        if str(description) == "BAD_acq_skip"
    ]
    assert selected == [(20.0, 10.0, "BAD_acq_skip")]
    assert record["selected_source_interval_count"] == 2
    assert record["annotations_added"] == 1


def test_quality_interval_annotations_keep_source_clock_after_raw_crop():
    from mndm.preprocess import _apply_quality_interval_annotations

    raw = _raw(np.ones((1, 60 * 16)))
    raw.crop(tmin=10.0, tmax=59.9375, verbose="ERROR")
    record = _apply_quality_interval_annotations(
        raw,
        [{"source_kind": "gap", "start_sec": 20.0, "end_sec": 30.0}],
        {"enabled": True, "source_kinds": ["gap"], "annotation_description": "BAD_acq_skip"},
        source_origin_time_sec=0.0,
    )
    masked = raw.get_data(reject_by_annotation="NaN")
    assert record["annotations_added"] == 1
    assert np.isnan(masked[:, 10 * 16 : 20 * 16]).all()
    assert np.isfinite(masked[:, : 10 * 16]).all()
    assert np.isfinite(masked[:, 20 * 16 :]).all()


def test_segment_aware_resampling_does_not_mix_across_declared_gap():
    from mndm.preprocess import _apply_quality_interval_annotations, _split_raw_at_bad_annotation_boundaries

    rng = np.random.default_rng(31)
    original = rng.normal(size=(2, 60 * 16))
    changed_gap = original.copy()
    changed_gap[:, 20 * 16 : 30 * 16] = 1000.0
    policy = {
        "enabled": True,
        "source_kinds": ["gap"],
        "annotation_description": "BAD_acq_skip",
    }
    intervals = [{"source_kind": "gap", "start_sec": 20.0, "end_sec": 30.0}]
    outputs = []
    reports = []
    for data in (original, changed_gap):
        raw = _raw(data)
        _apply_quality_interval_annotations(raw, intervals, policy)
        raw, report = _split_raw_at_bad_annotation_boundaries(raw, target_sfreq=4.0)
        raw.resample(4.0, verbose="ERROR")
        outputs.append(raw.get_data())
        reports.append(report)

    assert reports[0]["status"] == "segmented"
    assert reports[0]["boundary_samples"] == [320, 480]
    assert reports[0]["segment_count_after"] == 3
    np.testing.assert_allclose(outputs[0][:, :80], outputs[1][:, :80])
    np.testing.assert_allclose(outputs[0][:, 120:], outputs[1][:, 120:])


def test_segment_aware_resampling_fails_closed_on_unaligned_gap_edge():
    from mndm.preprocess import _apply_quality_interval_annotations, _split_raw_at_bad_annotation_boundaries

    raw = _raw(np.random.default_rng(32).normal(size=(2, 60 * 16)))
    _apply_quality_interval_annotations(
        raw,
        [{"source_kind": "gap", "start_sec": 20.125, "end_sec": 30.0}],
        {"enabled": True, "source_kinds": ["gap"], "annotation_description": "BAD_acq_skip"},
    )
    with pytest.raises(ValueError, match="does not align to the target sample grid"):
        _split_raw_at_bad_annotation_boundaries(raw, target_sfreq=4.0)


def test_mne_zero_phase_filter_does_not_mix_samples_across_bad_gap():
    from mndm.preprocess import _apply_quality_interval_annotations, _filter_skip_by_annotation

    rng = np.random.default_rng(35)
    original = rng.normal(size=(2, 60 * 128))
    changed_gap = original.copy()
    changed_gap[:, 20 * 128 : 30 * 128] = 1000.0
    outputs = []
    for data in (original, changed_gap):
        raw = _raw(data, sfreq=128.0)
        policy = {
            "enabled": True,
            "source_kinds": ["gap"],
            "annotation_description": "BAD_custom_gap",
        }
        _apply_quality_interval_annotations(
            raw,
            [{"source_kind": "gap", "start_sec": 20.0, "end_sec": 30.0}],
            policy,
        )
        skip_by_annotation = _filter_skip_by_annotation(raw, policy)
        assert skip_by_annotation is not None and "BAD_custom_gap" in skip_by_annotation
        raw.notch_filter(freqs=50.0, skip_by_annotation=skip_by_annotation, verbose="ERROR")
        raw.filter(l_freq=1.0, h_freq=40.0, skip_by_annotation=skip_by_annotation, verbose="ERROR")
        outputs.append(raw.get_data())

    np.testing.assert_allclose(outputs[0][:, : 20 * 128], outputs[1][:, : 20 * 128], atol=1e-12)
    np.testing.assert_allclose(outputs[0][:, 30 * 128 :], outputs[1][:, 30 * 128 :], atol=1e-12)


def test_joint_flatline_audit_is_observational_and_returns_raw_clock_bounds():
    from mndm.preprocess import _joint_flatline_quality_intervals

    values = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 8.0],
            [4.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        ]
    )
    before = values.copy()
    intervals = _joint_flatline_quality_intervals(
        values,
        sfreq=2.0,
        first_time_sec=10.0,
        channel_names=["C3", "C4"],
        min_duration_sec=1.0,
        raw_file="synthetic.edf",
    )
    np.testing.assert_array_equal(values, before)
    assert len(intervals) == 1
    assert intervals[0]["source_kind"] == "raw_joint_eeg_flatline"
    assert intervals[0]["start_sec"] == 11.5
    assert intervals[0]["duration_sec"] == 2.0
    assert intervals[0]["channel_names"] == ["C3", "C4"]


def test_eeg_features_reject_whole_windows_touching_opt_in_bad_support():
    from mndm.features.eeg import compute_eeg_features

    rng = np.random.default_rng(33)
    data = rng.normal(size=(2, 50 * 16))
    data[:, 20 * 16 : 30 * 16] = np.nan
    signals = {
        "signals": {"eeg": data},
        "sfreq": 16.0,
        "meta": {
            "quality_interval_policy": {
                "enabled": True,
                "mask_bad_annotations": True,
                "status": "applied",
            }
        },
    }
    out = compute_eeg_features(
        signals,
        {
            "epoching": {"length_s": 10.0, "step_s": 10.0},
            "features": {"eeg_bands": {"alpha": [1.0, 4.0]}, "eeg_psd": {"method": "welch"}},
        },
    )

    assert out["epoch_id"].tolist() == [0, 1, 3, 4]
    assert out["input_sample_finite_status"].eq("all_finite").all()


def test_preprocess_file_applies_selected_source_gap_and_keeps_flatline_qc_separate(tmp_path: Path):
    import pandas as pd

    from mndm.preprocess import preprocess_file

    sfreq = 16.0
    rng = np.random.default_rng(34)
    data = rng.normal(size=(4, 60 * int(sfreq)))
    data[0, 19 * 16 : 24 * 16] = 0.1
    data[1, 19 * 16 : 24 * 16] = -0.2
    info = mne.create_info(
        ["EEG0", "EEG1", "EOG0", "PPG"],
        sfreq,
        ch_types=["eeg", "eeg", "eog", "bio"],
    )
    raw = mne.io.RawArray(data.copy(), info, verbose="ERROR")
    raw_path = tmp_path / "ds005555" / "sub-99" / "eeg" / "sub-99_task-Sleep_acq-psg_eeg.fif"
    raw_path.parent.mkdir(parents=True)
    raw.save(raw_path, overwrite=True, verbose="ERROR")
    events_path = raw_path.parent / "sub-99_task-Sleep_acq-psg_events.tsv"
    pd.DataFrame(
        {"onset": [0.0, 20.0, 30.0], "duration": [20.0, 10.0, 30.0], "stage_hum": [0, 8, 2]}
    ).to_csv(events_path, sep="\t", index=False)
    config = {
        "datasets": ["ds005555"],
        "preprocess": {
            "sfreq": 4.0,
            "notch_hz": None,
            "eeg_bandpass": None,
            "reref": "none",
            "eeg_csd": {"enabled": False},
            "quality_interval_policy": {
                "datasets": {
                    "ds005555": {
                        "enabled": True,
                        "source_kinds": ["source_stage_hum_code_8", "raw_joint_eeg_flatline"],
                        "annotation_description": "BAD_acq_skip",
                        "mask_bad_annotations": True,
                        "segment_resampling": True,
                    }
                }
            },
            "raw_flatline_qc": {"datasets": {"ds005555": {"enabled": True, "min_duration_sec": 1.0}}},
        },
    }

    result = preprocess_file(raw_path, config)

    eeg = result.signals["eeg"]
    eog = result.signals["eog"]
    ppg = result.signals["ppg"]
    policy = result.meta["quality_interval_policy"]
    flatline = result.meta["raw_joint_eeg_flatline_qc"]
    source_quality = result.meta["signal_support_provenance"]["source_quality_intervals"]
    assert policy["status"] == "applied"
    assert policy["resampling"]["status"] == "segmented"
    assert policy["annotations_added"] == 2
    assert policy["masking"]["masked_duration_sec"] == 11.0
    assert eeg.shape == (2, 240)
    assert np.isnan(eeg[:, 19 * 4 : 30 * 4]).all()
    assert np.isnan(eog[:, 19 * 4 : 30 * 4]).all()
    assert np.isnan(ppg[:, 19 * 4 : 30 * 4]).all()
    assert np.isfinite(eeg[:, : 19 * 4]).all() and np.isfinite(eeg[:, 30 * 4 :]).all()
    assert np.isfinite(eog[:, : 19 * 4]).all() and np.isfinite(eog[:, 30 * 4 :]).all()
    assert np.isfinite(ppg[:, : 19 * 4]).all() and np.isfinite(ppg[:, 30 * 4 :]).all()
    assert flatline["interval_count"] == 1
    assert flatline["intervals"][0]["start_sec"] == 19.0
    assert flatline["intervals"][0]["duration_sec"] == 5.0
    assert all(item["source_kind"] != "raw_joint_eeg_flatline" for item in source_quality["intervals"])
    assert source_quality["application_policy"]["annotations_added"] == 2
    assert source_quality["application_policy"]["resampling"]["status"] == "segmented"
    assert source_quality["application_policy"]["masking"]["masked_samples"] == 44


def test_preprocess_fails_closed_if_gap_aware_eeg_filter_fails(tmp_path: Path, monkeypatch):
    import pandas as pd

    from mndm import preprocess

    sfreq = 16.0
    raw_path = tmp_path / "ds005555" / "sub-99" / "eeg" / "sub-99_task-Sleep_acq-test_eeg.fif"
    raw_path.parent.mkdir(parents=True)
    _raw(np.random.default_rng(36).normal(size=(2, 60 * 16)), sfreq=sfreq).save(
        raw_path, overwrite=True, verbose="ERROR"
    )
    pd.DataFrame({"onset": [20.0], "duration": [10.0], "stage_hum": [8]}).to_csv(
        raw_path.parent / "sub-99_task-Sleep_acq-test_events.tsv", sep="\t", index=False
    )

    def fake_read_raw(*args, **kwargs):
        return _raw(np.random.default_rng(37).normal(size=(2, 60 * 16)), sfreq=sfreq)

    def fail_raw_filter(self, *args, **kwargs):
        raise RuntimeError("synthetic Raw.filter failure")

    monkeypatch.setattr(preprocess.mne.io, "read_raw", fake_read_raw)
    monkeypatch.setattr(preprocess.mne.io.BaseRaw, "filter", fail_raw_filter)
    config = {
        "datasets": ["ds005555"],
        "preprocess": {
            "sfreq": sfreq,
            "notch_hz": None,
            "eeg_bandpass": [1.0, 3.0],
            "reref": "none",
            "eeg_csd": {"enabled": False},
            "quality_interval_policy": {
                "datasets": {
                    "ds005555": {
                        "enabled": True,
                        "source_kinds": ["source_stage_hum_code_8"],
                        "annotation_description": "BAD_acq_skip",
                        "mask_bad_annotations": True,
                        "segment_resampling": True,
                    }
                }
            },
        },
    }

    with pytest.raises(RuntimeError, match="Refusing annotation-unaware EEG filter fallback"):
        preprocess.preprocess_file(raw_path, config)
