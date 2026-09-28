from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from mndm.features.fnirs import compute_fnirs_features
from mndm.pipeline.fnirs_export_contract import FNIRS_H5_CONTRACT_VERSION
from mndm.pipeline.fnirs_native_export import (
    FnirsExportContractError,
    validate_fnirs_h5,
    write_fnirs_native_h5,
)

from test_features_fnirs import _config, _payload


def _features(times=None, *, length=3.0, step=3.0):
    if times is None:
        times = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    frame = compute_fnirs_features(
        _payload(np.asarray(times, dtype=float)),
        _config(length=length, step=step),
    )
    frame["file"] = "run.snirf"
    provenance = json.loads(frame["fnirs_preprocessing_provenance_json"].iloc[0])
    provenance.setdefault("schema", "mndm.fnirs_preprocessing_provenance.v1")
    if not any(op.get("name") == "intensity_to_optical_density" for op in provenance["operations"]):
        provenance["operations"].insert(
            0,
            {"name": "intensity_to_optical_density", "status": "applied", "parameters": {}},
        )
    for operation in provenance["operations"]:
        operation.setdefault("parameters", {})
        if operation.get("name") == "beer_lambert_law":
            operation["parameters"].setdefault("ppf", 6.0)
    frame["fnirs_preprocessing_provenance_json"] = json.dumps(provenance)
    return frame


def test_fnirs_native_h5_round_trip_keeps_gaps_qc_and_provenance(tmp_path):
    frame = _features()
    target = tmp_path / "run.h5"

    write_fnirs_native_h5(frame, target)
    validate_fnirs_h5(target)

    with h5py.File(target, "r") as h5:
        assert h5.attrs["export_contract_version"] == FNIRS_H5_CONTRACT_VERSION
        assert h5.attrs["has_fnirs"] == 1 and h5.attrs["has_eeg"] == 0
        assert h5.attrs["mnps_qualified"] == 0
        assert h5["fnirs/windows/window_id"][:].tolist() == frame["fnirs_window_id"].tolist()
        assert h5["fnirs/windows/qc_ok_fnirs"][:].tolist() == [-1] * len(frame)
        assert h5["fnirs/windows/is_qualified"][:].tolist() == [0] * len(frame)
        assert h5["fnirs/channels/mean_M"].shape == (len(frame), 2)
        assert np.isnan(h5["fnirs/channels/mean_M"][1]).all()
        assert h5["fnirs/windows/window_status"].asstr()[:].tolist() == frame["fnirs_window_status"].tolist()
        assert h5["row_source/row_source"].asstr()[:].tolist() == ["snirf_fnirs"] * len(frame)
        provenance = json.loads(h5["fnirs/provenance/preprocessing_provenance_json"].asstr()[()])
        assert provenance["source_sha256"] == "a" * 64
        assert "time" not in h5 and "mnps_3d" not in h5 and "coords_9d" not in h5


def test_fnirs_native_h5_preserves_windows_with_no_clock_support(tmp_path):
    frame = _features(np.array([0, 1, 12, 13], dtype=float), length=2.0, step=2.0)
    target = tmp_path / "gap.h5"

    write_fnirs_native_h5(frame, target)

    with h5py.File(target, "r") as h5:
        rejected = h5["fnirs/windows/window_status"].asstr()[:] == "clock_rejected"
        assert rejected.sum() == 5
        assert np.isnan(h5["fnirs/windows/actual_t_start_s"][:][rejected]).all()
        assert np.isnan(h5["fnirs/windows/actual_t_end_s"][:][rejected]).all()
        assert (h5["fnirs/windows/clock_support_fraction"][:][rejected] == 0).all()
        assert (h5["fnirs/windows/sample_count"][:][rejected] == 0).all()


def test_fnirs_native_h5_refuses_unknown_qc_when_qc_pass_is_required(tmp_path):
    with pytest.raises(FnirsExportContractError, match="unknown/failing QC is not pass"):
        write_fnirs_native_h5(_features(), tmp_path / "run.h5", require_qc_pass=True)


def test_fnirs_native_h5_preserves_explicit_qc_failure_without_qualifying(tmp_path):
    frame = _features()
    frame["qc_ok_fnirs"] = frame["qc_ok_fnirs"].astype("boolean")
    frame.loc[0, "qc_ok_fnirs"] = False
    frame.loc[0, "fnirs_qc_status"] = "assessed"
    target = tmp_path / "qc-fail.h5"

    write_fnirs_native_h5(frame, target)

    with h5py.File(target, "r") as h5:
        assert h5["fnirs/windows/qc_ok_fnirs"][0] == 0
        assert h5["fnirs/windows/is_qualified"][:].tolist() == [0] * len(frame)
    with pytest.raises(FnirsExportContractError, match="unknown/failing QC is not pass"):
        write_fnirs_native_h5(frame, tmp_path / "qc-required.h5", require_qc_pass=True)


def test_fnirs_native_h5_accepts_producer_shaped_already_od_provenance(tmp_path):
    frame = _features()
    provenance = json.loads(frame["fnirs_preprocessing_provenance_json"].iloc[0])
    provenance["input_classification"] = "optical_density"
    provenance["operations"] = [
        operation
        for operation in provenance["operations"]
        if operation.get("name") != "intensity_to_optical_density"
    ]
    provenance["operations"].insert(
        0,
        {
            "name": "intensity_to_optical_density",
            "status": "not_applicable",
            "reason": "input_already_optical_density",
        },
    )
    frame["fnirs_preprocessing_provenance_json"] = json.dumps(provenance)

    target = tmp_path / "already-od.h5"
    write_fnirs_native_h5(frame, target)
    validate_fnirs_h5(target)


def test_fnirs_native_h5_rejects_mixed_sources_and_multiple_recordings(tmp_path):
    frame = _features()
    frame.loc[1, "row_source"] = "nifti_bold"
    with pytest.raises(FnirsExportContractError, match="row_source"):
        write_fnirs_native_h5(frame, tmp_path / "mixed.h5")

    frame = _features()
    frame.loc[1, "file"] = "other.snirf"
    with pytest.raises(FnirsExportContractError, match="one non-empty value|Exactly one"):
        write_fnirs_native_h5(frame, tmp_path / "multiple.h5")

    frame = _features()
    frame["file"] = "unknown-source.edf"
    with pytest.raises(FnirsExportContractError, match=r"requires a \.snirf source"):
        write_fnirs_native_h5(frame, tmp_path / "unknown.h5")


def test_fnirs_native_h5_rejects_channel_key_that_does_not_encode_identity(tmp_path):
    frame = _features()
    channel_map = json.loads(frame["fnirs_channel_map_json"].iloc[0])
    original_key = channel_map[0]["column_key"]
    channel_map[0]["column_key"] = "different-but-unique-key"
    frame["fnirs_channel_map_json"] = json.dumps(channel_map)
    for prefix in (
        "fnirs_mean_M__", "fnirs_sd_M__", "fnirs_valid_support_fraction__",
        "fnirs_valid_sample_count__", "fnirs_valid_timestamp_count__", "fnirs_rejection_reason__",
    ):
        frame[prefix + "different-but-unique-key"] = frame[prefix + original_key]

    with pytest.raises(FnirsExportContractError, match="column_key must be the URL-quoted"):
        write_fnirs_native_h5(frame, tmp_path / "mislabeled.h5")


@pytest.mark.parametrize("column", ["fnirs_window_sample_count", "fnirs_window_timestamp_count"])
def test_fnirs_native_h5_rejects_negative_window_counts(tmp_path, column):
    frame = _features()
    frame.loc[0, column] = -1
    with pytest.raises(FnirsExportContractError, match="may not contain negative"):
        write_fnirs_native_h5(frame, tmp_path / "negative-count.h5")


def test_fnirs_native_h5_rejects_inconsistent_duration_and_support(tmp_path):
    frame = _features()
    frame.loc[0, "fnirs_actual_covered_duration_s"] = 0.5
    with pytest.raises(FnirsExportContractError, match="Covered duration must agree"):
        write_fnirs_native_h5(frame, tmp_path / "inconsistent-support.h5")


@pytest.mark.parametrize("dataset", ["fnirs/windows/sample_count", "fnirs/windows/timestamp_count"])
def test_fnirs_native_h5_validator_rejects_negative_stored_counts(tmp_path, dataset):
    target = tmp_path / "run.h5"
    write_fnirs_native_h5(_features(), target)
    with h5py.File(target, "r+") as h5:
        h5[dataset][0] = -1
    with pytest.raises(FnirsExportContractError, match="may not be negative"):
        validate_fnirs_h5(target)


def test_fnirs_native_h5_requires_overwrite_flag(tmp_path):
    target = tmp_path / "run.h5"
    write_fnirs_native_h5(_features(), target)
    with pytest.raises(FileExistsError):
        write_fnirs_native_h5(_features(), target)


def test_fnirs_native_h5_validator_rejects_shape_corruption(tmp_path):
    target = tmp_path / "run.h5"
    write_fnirs_native_h5(_features(), target)
    with h5py.File(target, "r+") as h5:
        values = h5["fnirs/channels/mean_M"][:]
        del h5["fnirs/channels/mean_M"]
        h5["fnirs/channels"].create_dataset("mean_M", data=values[:, :1])
    with pytest.raises(FnirsExportContractError, match=r"invalid \[window, channel\] shape"):
        validate_fnirs_h5(target)


@pytest.mark.parametrize(
    "dataset,value,message",
    [
        ("fnirs/windows/window_status", "unsupported", "unsupported value"),
        ("fnirs/windows/measurement_validity", "assessed", "cannot upgrade"),
    ],
)
def test_fnirs_native_h5_validator_rejects_invalid_status_combinations(tmp_path, dataset, value, message):
    target = tmp_path / "run.h5"
    write_fnirs_native_h5(_features(), target)
    with h5py.File(target, "r+") as h5:
        h5[dataset][0] = value
    with pytest.raises(FnirsExportContractError, match=message):
        validate_fnirs_h5(target)


def test_fnirs_native_h5_validator_rejects_nonfinite_window_time(tmp_path):
    target = tmp_path / "run.h5"
    write_fnirs_native_h5(_features(), target)
    with h5py.File(target, "r+") as h5:
        h5["fnirs/windows/t_start_s"][0] = np.nan
    with pytest.raises(FnirsExportContractError, match="must be finite"):
        validate_fnirs_h5(target)


@pytest.mark.parametrize(
    "dataset,index,value,message",
    [
        ("fnirs/channels/valid_support_fraction", (0, 0), np.inf, "Per-channel valid support"),
        ("fnirs/windows/window_id", 1, 0, "window_id values must be unique"),
        ("fnirs/provenance/source_sha256", (), "x", "64-character hexadecimal"),
    ],
)
def test_fnirs_native_h5_validator_rejects_corrupt_support_identity_and_digest(
    tmp_path, dataset, index, value, message
):
    target = tmp_path / "run.h5"
    write_fnirs_native_h5(_features(), target)
    with h5py.File(target, "r+") as h5:
        h5[dataset][index] = value
    with pytest.raises(FnirsExportContractError, match=message):
        validate_fnirs_h5(target)


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("units", "must declare MNE molar Hb units"),
        ("validity", "cannot upgrade measurement validity"),
        ("claim", "cannot upgrade measurement validity"),
        ("beer_lambert", "must record applied Beer-Lambert"),
        ("optical_density", "Optical-density provenance status"),
        ("null_parameters", "applicable parameters or reasons"),
        ("ppf", "positive finite PPF"),
        ("software", "record MNE and MNE-NIRS versions"),
    ],
)
def test_fnirs_native_h5_validator_rejects_invalid_preprocessing_provenance(tmp_path, mutation, message):
    target = tmp_path / "run.h5"
    write_fnirs_native_h5(_features(), target)
    with h5py.File(target, "r+") as h5:
        provenance_ds = h5["fnirs/provenance/preprocessing_provenance_json"]
        provenance = json.loads(provenance_ds.asstr()[()])
        if mutation == "units":
            provenance["haemoglobin_units"] = "unknown"
        elif mutation == "validity":
            provenance["measurement_validity"] = "valid"
        elif mutation == "claim":
            provenance["claim_status"] = "qualified"
        elif mutation == "beer_lambert":
            provenance["operations"] = [op for op in provenance["operations"] if op["name"] != "beer_lambert_law"]
        elif mutation == "optical_density":
            provenance["operations"] = [op for op in provenance["operations"] if op["name"] != "intensity_to_optical_density"]
        elif mutation == "null_parameters":
            next(op for op in provenance["operations"] if op["name"] == "intensity_to_optical_density")["parameters"] = None
        elif mutation == "ppf":
            next(op for op in provenance["operations"] if op["name"] == "beer_lambert_law")["parameters"].pop("ppf")
        elif mutation == "software":
            provenance["software"].pop("mne_nirs")
        del h5["fnirs/provenance/preprocessing_provenance_json"]
        h5["fnirs/provenance"].create_dataset(
            "preprocessing_provenance_json",
            data=json.dumps(provenance),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
    with pytest.raises(FnirsExportContractError, match=message):
        validate_fnirs_h5(target)


def test_fnirs_export_cli_does_not_require_pipeline_config_or_dataset(monkeypatch, tmp_path):
    from mndm import cli
    from mndm.pipeline import fnirs_native_export

    output = tmp_path / "out.h5"
    monkeypatch.setattr(cli.config_loader, "load_config", lambda _path: pytest.fail("config should not be loaded"))
    monkeypatch.setattr(fnirs_native_export, "export_fnirs_feature_table", lambda *a, **k: output)

    result = cli.main(["fnirs-export", "--features-table", str(tmp_path / "features.csv"), "--out", str(output)])

    assert result == 0


@pytest.mark.parametrize("command", ["summarize", "resummarize", "all"])
def test_fnirs_cli_keeps_mnps_summary_commands_blocked(monkeypatch, command):
    from mndm import cli

    monkeypatch.setattr(cli.config_loader, "load_config", lambda _path: {"modality": "fnirs"})

    result = cli.main([command, "--dataset", "ds005963"])

    assert result == 1
