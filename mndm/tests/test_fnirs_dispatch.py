from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _index_record(root: Path) -> dict[str, object]:
    return {
        "nirs_json_sidecars": "['dataset_nirs.json', 'sub-01/nirs/sub-01_task-rest_nirs.json']",
        "nirs_json_sidecar_ambiguous": "False",
        "nirs_channels_tsv": "sub-01/nirs/sub-01_task-rest_channels.tsv",
        "nirs_channels_tsv_ambiguous": "False",
        "nirs_optodes_tsv": "sub-01/nirs/sub-01_optodes.tsv",
        "nirs_optodes_tsv_ambiguous": "False",
        "nirs_events_tsv": "sub-01/nirs/sub-01_task-rest_events.tsv",
        "nirs_events_tsv_ambiguous": "False",
        "nirs_coordsystem_sidecars": "['dataset_coordsystem.json', 'sub-01/nirs/sub-01_coordsystem.json']",
        "nirs_coordsystem_sidecar_ambiguous": "False",
    }


def test_snirf_without_index_context_fails_before_generic_mne(tmp_path: Path, monkeypatch):
    from mndm import preprocess

    class _MustNotRead:
        class io:
            @staticmethod
            def read_raw(*args, **kwargs):
                raise AssertionError("SNIRF fell through to generic MNE loading")

    monkeypatch.setattr(preprocess, "mne", _MustNotRead())
    snirf = tmp_path / "sub-01_task-rest_nirs.snirf"
    snirf.touch()
    with pytest.raises(ValueError, match="requires its BIDS index record"):
        preprocess.preprocess_file(snirf, {})


def test_snirf_dispatch_passes_full_sidecar_chains_and_flags(tmp_path: Path, monkeypatch):
    from mndm import fnirs_metadata, preprocess

    root = tmp_path / "dataset"
    snirf = root / "sub-01" / "nirs" / "sub-01_task-rest_nirs.snirf"
    snirf.parent.mkdir(parents=True)
    snirf.touch()
    captured: dict[str, object] = {}

    class _Report:
        admissible = True

        @staticmethod
        def to_dict():
            return {"valid": True, "admissible": True}

    def fake_validate(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return _Report()

    monkeypatch.setattr(fnirs_metadata, "validate_fnirs_recording", fake_validate)
    report = preprocess._validate_fnirs_indexed_recording(
        snirf,
        _index_record(root),
        root,
    )
    assert report.admissible is True

    assert captured["nirs_json"] == [
        (root / "dataset_nirs.json").resolve(),
        (root / "sub-01/nirs/sub-01_task-rest_nirs.json").resolve(),
    ]
    assert captured["coordsystem_json"] == [
        (root / "dataset_coordsystem.json").resolve(),
        (root / "sub-01/nirs/sub-01_coordsystem.json").resolve(),
    ]
    assert captured["nirs_json_ambiguous"] is False
    assert captured["channels_tsv_ambiguous"] is False
    assert captured["channels_tsv"] == (root / "sub-01/nirs/sub-01_task-rest_channels.tsv").resolve()


def test_snirf_preprocess_file_returns_native_fnirs_payload_without_generic_path(
    tmp_path: Path, monkeypatch
):
    from types import SimpleNamespace

    from mndm import preprocess
    from mndm import preprocess_fnirs as fnirs_transform

    root = tmp_path / "dataset"
    snirf = root / "sub-01" / "nirs" / "sub-01_task-rest_nirs.snirf"
    snirf.parent.mkdir(parents=True)
    snirf.touch()
    report = SimpleNamespace(
        admissible=True,
        classification="cw_amplitude",
        summary={
            "time_start_s": 0.0,
            "time_end_s": 1.0,
            "median_sample_interval_s": 0.25,
            "time_gap_sample_indices": [],
            "wavelengths_nm": [760.0, 850.0],
        },
        to_dict=lambda: {"valid": True, "admissible": True},
    )

    class _Raw:
        ch_names = ["S1_D1 hbo", "S1_D1 hbr"]
        info = {"sfreq": 4.0}
        n_times = 5

        @staticmethod
        def get_channel_types():
            return ["hbo", "hbr"]

        @staticmethod
        def get_data():
            return np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]], dtype=float)

    captured: dict[str, object] = {}
    fake_result = SimpleNamespace(
        raw_haemoglobin=_Raw(),
        provenance={"haemoglobin_units": "M (molar; MNE FIFF_UNIT_MOL)"},
        diagnostics={"signal_quality_assessment": "not_assessed"},
        scalp_coupling_index={"S1_D1 760": 0.8},
        computation_status="computed",
        measurement_validity="not_assessed",
        claim_status="no_biological_claim",
    )
    monkeypatch.setattr(preprocess, "_validate_fnirs_indexed_recording", lambda *args: report)
    monkeypatch.setattr(
        preprocess,
        "mne",
        SimpleNamespace(
            io=SimpleNamespace(
                read_raw_snirf=lambda path, **kwargs: captured.update({"read_path": path}) or object()
            )
        ),
    )
    monkeypatch.setattr(
        fnirs_transform,
        "preprocess_fnirs",
        lambda raw, **kwargs: captured.update({"transform_args": kwargs}) or fake_result,
    )

    output = preprocess.preprocess_file(
        snirf,
        {},
        index_record=_index_record(root),
        dataset_root=root,
    )
    assert captured["read_path"] == snirf
    assert captured["transform_args"]["validation"] is report
    assert output.signals["fnirs"].shape == (2, 5)
    assert output.channels["fnirs"] == ["S1_D1 hbo", "S1_D1 hbr"]
    assert output.meta["fnirs"]["measurement_validity"] == "not_assessed"
    assert output.meta["fnirs"]["channel_units"] == "M (molar; MNE FIFF_UNIT_MOL)"


def test_snirf_dispatch_rejects_missing_ambiguity_columns(tmp_path: Path):
    from mndm.preprocess import preprocess_file

    root = tmp_path / "dataset"
    snirf = root / "sub-01" / "nirs" / "sub-01_task-rest_nirs.snirf"
    snirf.parent.mkdir(parents=True)
    snirf.touch()
    record = _index_record(root)
    record.pop("nirs_events_tsv_ambiguous")
    with pytest.raises(ValueError, match="missing required field"):
        preprocess_file(snirf, {}, index_record=record, dataset_root=root)


def test_snirf_source_must_resolve_inside_dataset_root(tmp_path: Path, monkeypatch):
    from mndm import fnirs_metadata, preprocess

    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "outside.snirf"
    outside.touch()

    class _Report:
        admissible = True

        @staticmethod
        def to_dict():
            return {"valid": True, "admissible": True}

    def should_not_validate(*args, **kwargs):
        raise AssertionError("escaped source must be rejected before validation")

    monkeypatch.setattr(fnirs_metadata, "validate_fnirs_recording", should_not_validate)
    with pytest.raises(ValueError, match="source file escapes"):
        preprocess.preprocess_file(
            outside,
            {},
            index_record=_index_record(root),
            dataset_root=root,
        )


def test_cached_index_refresh_is_targeted_to_old_schema_and_bids_nirs(tmp_path: Path):
    import pandas as pd

    from mndm.orchestrate import (
        _FNIRS_FILE_INDEX_REQUIRED_COLUMNS,
        _cached_index_needs_fnirs_refresh,
    )

    nirs_dir = tmp_path / "sub-01" / "ses-01" / "nirs"
    nirs_dir.mkdir(parents=True)
    old_index = pd.DataFrame(columns=["path", "subject", "modality"])
    assert _cached_index_needs_fnirs_refresh(old_index, tmp_path) is True

    current_index = pd.DataFrame(columns=sorted(_FNIRS_FILE_INDEX_REQUIRED_COLUMNS))
    assert _cached_index_needs_fnirs_refresh(current_index, tmp_path) is False

    no_nirs_root = tmp_path / "without-nirs"
    (no_nirs_root / "sub-01" / "eeg").mkdir(parents=True)
    assert _cached_index_needs_fnirs_refresh(old_index, no_nirs_root) is False


def test_pending_fnirs_does_not_fail_sibling_bundle_modality(tmp_path: Path, monkeypatch):
    import pandas as pd

    from mndm import parallel, preprocess

    root = tmp_path / "dataset"
    nirs_file = root / "sub-01" / "nirs" / "sub-01_task-rest_nirs.snirf"
    eeg_file = root / "sub-01" / "eeg" / "sub-01_task-rest_eeg.edf"
    nirs_file.parent.mkdir(parents=True)
    eeg_file.parent.mkdir(parents=True)
    nirs_file.touch()
    eeg_file.touch()

    def fake_preprocess(path, config, **kwargs):
        if path.suffix == ".snirf":
            raise __import__("mndm.preprocess_fnirs", fromlist=["FnirsPreprocessingError"]).FnirsPreprocessingError(
                "short_channel_regression_failed",
                "synthetic failure",
            )
        return preprocess.PreprocessedSignals(
            signals={"eeg": []},
            sfreq=1.0,
            channels=["Cz"],
            meta={"file": str(path)},
        )

    monkeypatch.setattr(preprocess, "preprocess_file", fake_preprocess)
    monkeypatch.setattr(
        parallel,
        "_compute_features_from_preprocessed",
        lambda *args, **kwargs: (
            pd.DataFrame({"epoch_id": [0], "eeg_test": [1.0]}),
            {},
        ),
    )
    result = parallel.process_file_bundle(
        [
            {"path": str(nirs_file.relative_to(root)), "modality": "fnirs"},
            {"path": str(eeg_file.relative_to(root)), "modality": "eeg"},
        ],
        root,
        {},
    )
    assert result.success is True
    assert result.meta["bundle_modalities"] == ["eeg"]
    assert result.meta["skipped_modalities"] == ["fnirs:preprocess_failed:FnirsPreprocessingError"]


def test_fnirs_only_bundle_preserves_preprocessing_metadata_without_features(tmp_path: Path, monkeypatch):
    import pandas as pd

    from mndm import parallel, preprocess

    root = tmp_path / "dataset"
    nirs_file = root / "sub-01" / "nirs" / "sub-01_task-rest_nirs.snirf"
    nirs_file.parent.mkdir(parents=True)
    nirs_file.touch()
    expected_meta = {
        "modality": "fnirs",
        "fnirs": {"computation_status": "computed", "measurement_validity": "not_assessed"},
        "signal_support_provenance": {"sample_count": 100},
    }

    monkeypatch.setattr(
        preprocess,
        "preprocess_file",
        lambda *args, **kwargs: preprocess.PreprocessedSignals(
            signals={"fnirs": [[0.0]]},
            sfreq=10.0,
            channels={"fnirs": ["S1_D1 hbo"]},
            meta=expected_meta,
        ),
    )
    monkeypatch.setattr(
        parallel,
        "_compute_features_from_preprocessed",
        lambda *args, **kwargs: (pd.DataFrame(), {}),
    )

    result = parallel.process_file_bundle(
        [{"path": str(nirs_file.relative_to(root)), "modality": "fnirs"}],
        root,
        {},
    )

    assert result.success is True
    assert result.features_df is None
    assert result.meta["fnirs_preprocessing"] == expected_meta
    assert result.meta["skipped_modalities"] == ["fnirs:no_features"]
    assert result.meta["reason"] == "no_bundle_features"


def test_fnirs_mixed_bundle_keeps_independent_clock_out_of_epoch_merge(tmp_path: Path, monkeypatch):
    import pandas as pd

    from mndm import parallel, preprocess

    root = tmp_path / "dataset"
    nirs_file = root / "sub-01" / "nirs" / "sub-01_task-rest_nirs.snirf"
    eeg_file = root / "sub-01" / "eeg" / "sub-01_task-rest_eeg.edf"
    nirs_file.parent.mkdir(parents=True)
    eeg_file.parent.mkdir(parents=True)
    nirs_file.touch()
    eeg_file.touch()
    nirs_meta = {"modality": "fnirs", "fnirs": {"computation_status": "computed"}}

    def fake_preprocess(path, config, **kwargs):
        key = "fnirs" if path.suffix == ".snirf" else "eeg"
        return preprocess.PreprocessedSignals(
            signals={key: np.ones((2, 10))},
            sfreq=1.0,
            channels={key: ["S1_D1 hbo", "S1_D1 hbr"]} if key == "fnirs" else ["Cz", "C3"],
            meta=nirs_meta if key == "fnirs" else {"file": str(path)},
        )

    monkeypatch.setattr(preprocess, "preprocess_file", fake_preprocess)
    monkeypatch.setattr(
        parallel,
        "_compute_features_from_preprocessed",
        lambda pre, *args, **kwargs: (
            pd.DataFrame({"epoch_id": [0], "eeg_test": [1.0]}), {}
        ) if "eeg" in pre.signals else pytest.fail("fNIRS must not enter an EEG bundle join"),
    )

    result = parallel.process_file_bundle(
        [
            {"path": str(nirs_file.relative_to(root)), "modality": "fnirs"},
            {"path": str(eeg_file.relative_to(root)), "modality": "eeg"},
        ],
        root,
        {},
    )

    assert result.success is True
    assert result.features_df is not None
    assert "eeg_test" in result.features_df
    assert result.meta["fnirs_preprocessing"] == nirs_meta
    assert "fnirs:bundle_timebase_unsupported" in result.meta["skipped_modalities"]


def test_multiple_fnirs_recordings_in_bundle_keep_separate_provenance(tmp_path: Path, monkeypatch):
    import pandas as pd

    from mndm import parallel, preprocess

    root = tmp_path / "dataset"
    nirs_paths = [
        root / "sub-01" / "nirs" / f"sub-01_task-rest_run-{run}_nirs.snirf"
        for run in (1, 2)
    ]
    for path in nirs_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def fake_preprocess(path, config, **kwargs):
        return preprocess.PreprocessedSignals(
            signals={"fnirs": np.ones((2, 10))},
            sfreq=1.0,
            channels={"fnirs": ["S1_D1 hbo", "S1_D1 hbr"]},
            meta={"run_file": path.name},
        )

    monkeypatch.setattr(preprocess, "preprocess_file", fake_preprocess)
    monkeypatch.setattr(
        parallel,
        "_compute_features_from_preprocessed",
        lambda *args, **kwargs: pytest.fail("multiple independent fNIRS runs must not enter local epoch_id merge"),
    )

    result = parallel.process_file_bundle(
        [{"path": str(path.relative_to(root)), "modality": "fnirs"} for path in nirs_paths],
        root,
        {},
    )

    assert result.success is True
    assert result.features_df is None
    assert set(result.meta["fnirs_preprocessing_by_file"]) == {
        str(path.relative_to(root)) for path in nirs_paths
    }
    assert result.meta["skipped_modalities"] == [
        "fnirs:multiple_sources_bundle_unsupported",
        "fnirs:multiple_sources_bundle_unsupported",
    ]
