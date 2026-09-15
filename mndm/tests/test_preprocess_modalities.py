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
    )
    monkeypatch.setattr(preprocess, "wfdb", SimpleNamespace(rdrecord=lambda _: record))

    calls: list[int | str] = []
    original_resample = mne.io.BaseRaw.resample

    def spy_resample(self, *args, **kwargs):
        calls.append(kwargs.get("n_jobs"))
        return original_resample(self, *args, **kwargs)

    monkeypatch.setattr(mne.io.BaseRaw, "resample", spy_resample)
    hea_path = tmp_path / "dsWFDB" / "synthetic.hea"
    hea_path.parent.mkdir()

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
