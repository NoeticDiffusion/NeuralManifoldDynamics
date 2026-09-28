"""Focused modality-specific preprocess regressions."""

from pathlib import Path
import sys
from types import SimpleNamespace

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


def test_artifact_meta_applied_reflects_actual_execution_not_configuration(tmp_path: Path):
    """meta["artifact"]["applied"] must be False when the method is configured but cannot run.

    Regression guard for ingest_jacobian_fidelity_handover_2.md item 9:
    "method" alone records what was configured, not whether it executed.
    EOG regression with EEG channels but no EOG channel must report
    applied=False (skipped for missing channels), while the same config with
    both EEG and EOG channels present must report applied=True.
    """
    mne = pytest.importorskip("mne")

    from mndm.preprocess import preprocess_file

    sfreq = 100.0
    times = np.arange(0, 4.0, 1.0 / sfreq)
    eeg_signal = np.sin(2 * np.pi * 8.0 * times)

    def _run(ch_names, ch_types, subdir):
        data = np.vstack([eeg_signal, 0.3 * np.sin(2 * np.pi * 2.0 * times)]) if len(ch_names) == 2 else eeg_signal[None, :]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data[: len(ch_names)], info)
        ds_dir = tmp_path / subdir / "sub-001" / "eeg"
        ds_dir.mkdir(parents=True, exist_ok=True)
        file_path = ds_dir / "sub-001_task-rest_eeg.fif"
        raw.save(file_path, overwrite=True)
        return preprocess_file(
            file_path,
            {
                "datasets": [subdir],
                "preprocess": {
                    "sfreq": sfreq,
                    "notch_hz": None,
                    "artifacts": {"method": "eog_reg"},
                },
            },
        )

    # EEG-only: EOG regression is configured but has no EOG channel to use.
    eeg_only = _run(["C3"], ["eeg"], "dsARTIFACT_SKIP")
    assert eeg_only.meta["artifact"]["method"] == "eog_reg"
    assert eeg_only.meta["artifact"]["applied"] is False

    # EEG + EOG: EOG regression is configured and can actually run.
    eeg_eog = _run(["C3", "EOG1"], ["eeg", "eog"], "dsARTIFACT_APPLIED")
    assert eeg_eog.meta["artifact"]["method"] == "eog_reg"
    assert eeg_eog.meta["artifact"]["applied"] is True


def test_ica_artifact_meta_applied_false_when_no_components_excluded(tmp_path: Path):
    """meta["artifact"]["applied"] must be False when ICA ran but excluded nothing.

    Regression guard for an independent-review finding on
    ingest_jacobian_fidelity_handover_2.md item 9: ``ica.apply(r)`` with an
    empty ``exclude`` list is a no-op -- the signal is byte-for-byte
    unmodified -- yet the pre-fix code returned True unconditionally after a
    successful fit-and-apply call sequence. A dataset with EEG channels but
    no EOG/ECG channel to detect artifact components against (e.g. I-CARE
    ICU EEG without a dedicated EOG lead) must not report a confirmed
    artifact-reduction outcome.
    """
    mne = pytest.importorskip("mne")
    pytest.importorskip("mne.preprocessing")

    from mndm.preprocess import preprocess_file

    sfreq = 100.0
    duration_s = 20.0
    times = np.arange(0, duration_s, 1.0 / sfreq)
    rng = np.random.default_rng(0)
    ch_names = ["C3", "C4", "Cz"]
    data = np.vstack(
        [
            np.sin(2 * np.pi * 8.0 * times) + 0.05 * rng.standard_normal(times.shape),
            np.sin(2 * np.pi * 10.0 * times) + 0.05 * rng.standard_normal(times.shape),
            np.sin(2 * np.pi * 12.0 * times) + 0.05 * rng.standard_normal(times.shape),
        ]
    )
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * 3)
    raw = mne.io.RawArray(data, info)
    ds_dir = tmp_path / "dsICA_NOOP" / "sub-001" / "eeg"
    ds_dir.mkdir(parents=True, exist_ok=True)
    file_path = ds_dir / "sub-001_task-rest_eeg.fif"
    raw.save(file_path, overwrite=True)

    result = preprocess_file(
        file_path,
        {
            "datasets": ["dsICA_NOOP"],
            "preprocess": {
                "sfreq": sfreq,
                "notch_hz": None,
                "artifacts": {"method": "ica", "ica_n_components": 2},
            },
        },
    )
    assert result.meta["artifact"]["method"] == "ica"
    # No EOG/ECG channel exists to detect components against, so exclude_idx
    # stays empty and ica.apply(r) is a no-op regardless of fit success.
    assert result.meta["artifact"]["applied"] is False


def test_preprocess_wfdb_forwards_resample_jobs_and_records_timing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """WFDB preprocessing honors the shared resample worker setting."""
    mne = pytest.importorskip("mne")

    import mndm.preprocess as preprocess

    times = np.arange(0, 10.0, 1.0 / 200.0)
    record = SimpleNamespace(
        p_signal=np.column_stack(
            [
                np.sin(2 * np.pi * 8.0 * times),
                np.sin(2 * np.pi * 12.0 * times),
            ]
        ),
        sig_name=["C3", "C4"],
        units=["uV", "uV"],
        fs=200.0,
        record_name="synthetic",
        file_name=["synthetic.mat", "synthetic.mat"],
    )
    header = SimpleNamespace(
        file_name=["synthetic.mat"],
        fs=200.0,
        sig_len=int(times.size),
        base_date=None,
        base_time=None,
    )
    monkeypatch.setattr(
        preprocess,
        "wfdb",
        SimpleNamespace(rdrecord=lambda _: record, rdheader=lambda _: header),
    )

    calls: list[int | str] = []
    original_resample = mne.io.BaseRaw.resample

    def spy_resample(self, *args, **kwargs):
        calls.append(kwargs.get("n_jobs"))
        return original_resample(self, *args, **kwargs)

    monkeypatch.setattr(mne.io.BaseRaw, "resample", spy_resample)
    hea_path = tmp_path / "dsWFDB" / "synthetic.hea"
    hea_path.parent.mkdir()
    hea_path.write_text(
        "synthetic 2 200 2000\n"
        "synthetic.mat 16 1 0 0 0 0 C3\n"
        "synthetic.mat 16 1 0 0 0 0 C4\n"
        "#Start time: 00:00:00\n#End time: 00:00:09\n",
        encoding="utf-8",
    )
    (hea_path.parent / "synthetic.mat").write_bytes(b"wfdb-payload")

    def run(n_jobs: int | None):
        resample_cfg = {} if n_jobs is None else {"n_jobs": n_jobs}
        return preprocess.preprocess_wfdb(
            hea_path,
            {
                "datasets": ["dsWFDB"],
                "preprocess": {
                    "sfreq": 100.0,
                    "sfreq_candidates": [100.0],
                    "resample": resample_cfg,
                    "notch_hz": None,
                    "eeg_bandpass": [1.0, 40.0],
                    "reref": "average",
                },
            },
        )

    default_result = run(None)
    parallel_result = run(4)

    assert calls == [1, 4]
    assert default_result.meta["timings"]["resample"] >= 0.0
    assert parallel_result.meta["timings"]["resample"] >= 0.0
    assert default_result.meta["timings"]["total"] >= default_result.meta["timings"]["resample"]
    assert np.array_equal(default_result.signals["eeg"], parallel_result.signals["eeg"])
    assert default_result.channels == parallel_result.channels
    assert default_result.sfreq == parallel_result.sfreq == 100.0
    assert default_result.meta["eeg_amplitude_unit_status"] == "verified"
    assert default_result.meta["eeg_amplitude_unit"] == "microvolt"
    assert default_result.meta["eeg_reference_status"] == "verified"
    assert default_result.meta["eeg_reference_mode"] == "average"
    from mndm.signal_support_provenance import map_preprocessed_window_to_source_samples

    support = parallel_result.meta["signal_support_provenance"]
    assert support["raw_clock_before_crop"]["n_times"] == int(times.size)
    assert support["raw_clock_after_crop"]["n_times"] == int(times.size / 2)
    assert support["source"]["hash_stable"] is True
    assert support["source"]["associated_signal_files_status"] == "verified"
    assert support["source"]["associated_signal_files"][0]["hash_stable"] is True
    assert support["recording_time_reference"]["status"] == "ok"
    mapped = map_preprocessed_window_to_source_samples(
        0,
        800,
        preprocessed_sfreq_hz=float(parallel_result.sfreq),
        provenance=support,
    )
    assert mapped["source_sample_mapping_status"] == "nominal_bounds_mapped"
    assert mapped["nominal_source_sample_start"] == 0
    assert mapped["nominal_source_sample_end_exclusive"] == 1600
    assert mapped["nominal_source_time_start_sec"] == 0.0
    assert mapped["nominal_source_time_end_sec"] == 8.0
    assert mapped["actual_sample_contributors_status"] == "unknown"

    # WFDB MultiRecord input is read as a concatenated Record by rdrecord's
    # default path. Without a segment-by-segment map, the resulting array
    # indices must not be represented as one source-file sample interval.
    multi_header = SimpleNamespace(
        seg_name=["segment_1", "segment_2"],
        fs=200.0,
        sig_len=int(times.size),
        base_date=None,
        base_time=None,
    )
    monkeypatch.setattr(
        preprocess,
        "wfdb",
        SimpleNamespace(rdrecord=lambda _: record, rdheader=lambda _: multi_header),
    )
    multi_result = run(1)
    assert multi_result.meta["signal_support_provenance"]["source_sample_grid_supported"] is False
    assert (
        multi_result.meta["signal_support_provenance"]["source_sample_grid_reason"]
        == "wfdb_multirecord_segment_mapping_not_supported"
    )
    from mndm.features.eeg import compute_eeg_features

    multi_features = compute_eeg_features(
        {
            "signals": multi_result.signals,
            "sfreq": multi_result.sfreq,
            "channels": multi_result.channels,
            "dataset_id": "dsWFDB",
            "file_path": str(hea_path),
            "meta": multi_result.meta,
        },
        {"epoching": {"length_s": 8.0, "step_s": 4.0}},
    )
    assert multi_features["source_sample_mapping_status"].tolist() == ["unknown"]
    assert multi_features["source_sample_mapping_reason"].tolist() == [
        "wfdb_multirecord_segment_mapping_not_supported"
    ]


def test_wfdb_causal_finite_pilot_records_and_maps_support(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """The explicit WFDB pilot path is causal and exports bounded source support."""
    mne = pytest.importorskip("mne")
    import mndm.preprocess as preprocess
    from mndm.signal_support_provenance import map_preprocessed_window_to_source_samples

    sfreq = 200.0
    times = np.arange(0, 30.0, 1.0 / sfreq)
    signal = np.column_stack([
        np.sin(2 * np.pi * 8.0 * times),
        np.sin(2 * np.pi * 12.0 * times),
    ])
    record = SimpleNamespace(
        p_signal=signal,
        sig_name=["C3", "C4"],
        units=["uV", "uV"],
        fs=sfreq,
        record_name="causal_synthetic",
        file_name=["causal_synthetic.mat"],
    )
    header = SimpleNamespace(
        file_name=["causal_synthetic.mat"],
        fs=sfreq,
        sig_len=int(times.size),
        n_sig=2,
        base_date=None,
        base_time=None,
    )
    monkeypatch.setattr(
        preprocess,
        "wfdb",
        SimpleNamespace(rdrecord=lambda _: record, rdheader=lambda _: header),
    )
    hea_path = tmp_path / "dsWFDB" / "causal_synthetic.hea"
    hea_path.parent.mkdir()
    hea_path.write_text(
        f"causal_synthetic 2 200 {times.size}\n"
        "causal_synthetic.mat 16 1 0 0 0 0 C3\n"
        "causal_synthetic.mat 16 1 0 0 0 0 C4\n"
        "#Start time: 00:00:00\n#End time: 00:00:29\n",
        encoding="utf-8",
    )
    (hea_path.parent / "causal_synthetic.mat").write_bytes(b"causal-wfdb-payload")

    pilot_config = {
        "datasets": ["dsWFDB"],
        "preprocess": {
            "temporal_support_mode": "causal_finite",
            "sfreq": 100.0,
            "sfreq_candidates": [100.0],
            "resample": {"method": "causal_polyphase"},
            "notch_hz": None,
            "eeg_bandpass": [0.5, 40.0],
            "eeg_bandpass_phase": "minimum",
            "reref": "average",
        },
    }
    result = preprocess.preprocess_wfdb(hea_path, pilot_config)

    support = result.meta["signal_support_provenance"]
    assert result.sfreq == 100.0
    assert result.signals["eeg"].shape == (2, 3000)
    assert support["effective_time_support"]["status"] == "finite_causal"
    assert support["effective_time_support"]["resampler"]["tap_count"] == 41
    assert support["effective_time_support"]["eeg_bandpass"]["phase"] == "minimum"
    mapped = map_preprocessed_window_to_source_samples(
        800,
        1600,
        preprocessed_sfreq_hz=result.sfreq,
        provenance=support,
    )
    assert mapped["actual_sample_contributors_status"] == "finite_causal_conservative_bound"
    assert mapped["actual_source_sample_start"] < mapped["nominal_source_sample_start"]
    assert mapped["actual_source_sample_end_exclusive"] <= mapped["nominal_source_sample_end_exclusive"]
    assert 0 <= mapped["actual_source_sample_start"] < mapped["actual_source_sample_end_exclusive"] <= times.size
    bad_support = dict(support)
    bad_support["effective_time_support"] = dict(support["effective_time_support"])
    bad_support["effective_time_support"]["eeg_bandpass"] = {
        **support["effective_time_support"]["eeg_bandpass"],
        "phase": "zero",
    }
    rejected = map_preprocessed_window_to_source_samples(
        800, 1600, preprocessed_sfreq_hz=result.sfreq, provenance=bad_support
    )
    assert rejected["actual_sample_contributors_status"] == "unknown"
    assert rejected["actual_sample_contributors_reason"] == "finite_causal_support_unavailable:ValueError"
    reordered_support = dict(support)
    reordered_support["operations"] = list(reversed(support["operations"]))
    reordered = map_preprocessed_window_to_source_samples(
        800, 1600, preprocessed_sfreq_hz=result.sfreq, provenance=reordered_support
    )
    assert reordered["actual_sample_contributors_status"] == "unknown"
    assert reordered["actual_sample_contributors_reason"] == "finite_causal_support_unavailable:ValueError"
    import pandas as pd
    from mndm.pipeline.signal_support_export import build_signal_support_export

    ledger = build_signal_support_export(
        pd.DataFrame([{"file": hea_path.name, "t_start": 8.0, "t_end": 16.0, **mapped}]),
        [support],
    )
    assert ledger["temporal_support_status"] == "finite_causal_conservative_bound"
    assert ledger["per_epoch_input_extent"][0]["actual_source_sample_start"] == mapped["actual_source_sample_start"]

    mismatched_header = SimpleNamespace(
        file_name=["causal_synthetic.mat"], fs=sfreq, sig_len=int(times.size) - 1,
        n_sig=2, base_date=None, base_time=None,
    )
    monkeypatch.setattr(
        preprocess,
        "wfdb",
        SimpleNamespace(rdrecord=lambda _: record, rdheader=lambda _: mismatched_header),
    )
    with pytest.raises(ValueError, match="header grid that differs from the loaded record array"):
        preprocess.preprocess_wfdb(hea_path, pilot_config)

    # Perturbing future raw input cannot alter earlier causal resampler outputs.
    probe_data = np.zeros((1, 2000), dtype=float)
    changed_data = probe_data.copy()
    changed_data[0, 800] = 1.0
    info = mne.create_info(["C3"], sfreq=sfreq, ch_types=["eeg"])
    base_raw, _ = preprocess._causal_polyphase_resample(mne.io.RawArray(probe_data, info, verbose=False), 100.0)
    changed_raw, _ = preprocess._causal_polyphase_resample(mne.io.RawArray(changed_data, info, verbose=False), 100.0)
    assert np.allclose(base_raw.get_data()[:, :400], changed_raw.get_data()[:, :400], atol=1e-13)

    # The exact MNE filter settings recorded by the pilot are also one-sided.
    filter_base = mne.filter.filter_data(
        probe_data[0], 100.0, 0.5, 40.0, method="fir", phase="minimum", pad="constant", verbose=False
    )
    filter_changed = np.zeros(2000, dtype=float)
    filter_changed[1200] = 1.0
    filter_changed = mne.filter.filter_data(
        filter_changed, 100.0, 0.5, 40.0, method="fir", phase="minimum", pad="constant", verbose=False
    )
    assert np.allclose(filter_base[:1200], filter_changed[:1200], atol=1e-12)

    gap_data = probe_data.copy()
    gap_data[0, 700] = np.nan
    with pytest.raises(ValueError, match="finite, nonempty continuous array"):
        preprocess._causal_polyphase_resample(mne.io.RawArray(gap_data, info, verbose=False), 100.0)
    annotated_raw = mne.io.RawArray(probe_data, info, verbose=False)
    annotated_raw.set_annotations(mne.Annotations(onset=[3.0], duration=[0.1], description=["BAD boundary"]))
    with pytest.raises(ValueError, match="annotated boundaries"):
        preprocess._causal_polyphase_resample(annotated_raw, 100.0)
