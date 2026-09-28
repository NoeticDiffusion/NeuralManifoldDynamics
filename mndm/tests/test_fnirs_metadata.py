"""Synthetic BIDS/SNIRF metadata validation tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from mndm.fnirs_metadata import validate_fnirs_recording


def _write_tsv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_fixture(
    tmp_path: Path,
    *,
    representation: str = "cw",
    times: list[float] | None = None,
) -> dict[str, Path]:
    h5py = pytest.importorskip("h5py")
    import numpy as np

    if times is None:
        times = [0.0, 0.1, 0.2, 0.3, 0.4]
    if representation in {"cw", "od"}:
        od_label = "dOD" if representation == "od" else ""
        measurements = [
            (1, 1, 1, od_label),
            (1, 1, 2, od_label),
            (2, 2, 1, od_label),
            (2, 2, 2, od_label),
        ]
        classification = "cw_amplitude" if representation == "cw" else "optical_density"
        units = "V" if representation == "cw" else "unitless"
        channel_type = "NIRSCWAMPLITUDE" if representation == "cw" else "NIRSCWOPTICALDENSITY"
    elif representation == "hb":
        measurements = [
            (1, 1, None, "HbO"),
            (1, 1, None, "HbR"),
            (2, 2, None, "HbO"),
            (2, 2, None, "HbR"),
        ]
        classification = "haemoglobin"
        units = "M"
        channel_type = None
    else:
        raise ValueError(representation)

    snirf_path = tmp_path / "sub-01_task-test_nirs.snirf"
    with h5py.File(snirf_path, "w") as h5:
        nirs = h5.create_group("nirs")
        metadata = nirs.create_group("metaDataTags")
        metadata.create_dataset("LengthUnit", data=np.bytes_("m"))
        metadata.create_dataset("TimeUnit", data=np.bytes_("s"))
        data = nirs.create_group("data1")
        data.create_dataset("dataTimeSeries", data=np.ones((len(times), len(measurements))))
        data.create_dataset("time", data=np.asarray(times, dtype=float))
        probe = nirs.create_group("probe")
        probe.create_dataset("wavelengths", data=np.asarray([760.0, 850.0]))
        probe.create_dataset("sourceLabels", data=np.asarray([b"S1", b"S2"]))
        probe.create_dataset("detectorLabels", data=np.asarray([b"D1", b"D2"]))
        probe.create_dataset("sourcePos3D", data=np.asarray([[0.0, 0.0, 0.0], [0.03, 0.0, 0.0]]))
        probe.create_dataset("detectorPos3D", data=np.asarray([[0.005, 0.0, 0.0], [0.06, 0.0, 0.0]]))
        for index, (source, detector, wavelength_index, label) in enumerate(measurements, start=1):
            descriptor = data.create_group(f"measurementList{index}")
            descriptor.create_dataset("dataType", data=1 if representation == "cw" else 99999)
            descriptor.create_dataset("sourceIndex", data=source)
            descriptor.create_dataset("detectorIndex", data=detector)
            if wavelength_index is not None:
                descriptor.create_dataset("wavelengthIndex", data=wavelength_index)
            if label:
                descriptor.create_dataset("dataTypeLabel", data=np.bytes_(label))

    if representation in {"cw", "od"}:
        channel_rows = [
            {"name": f"S{source}-D{detector}", "type": channel_type, "source": f"S{source}", "detector": f"D{detector}", "wavelength_nominal": str(wavelength), "units": units}
            for source, detector in ((1, 1), (2, 2)) for wavelength in (760, 850)
        ]
    else:
        channel_rows = [
            {"name": f"S{source}-D{detector}", "type": "NIRSCWHBO" if chromophore == "hbo" else "NIRSCWHBR", "source": f"S{source}", "detector": f"D{detector}", "wavelength_nominal": "n/a", "units": units}
            for source, detector in ((1, 1), (2, 2)) for chromophore in ("hbo", "hbr")
        ]
    channels_path = tmp_path / "sub-01_task-test_channels.tsv"
    _write_tsv(channels_path, ["name", "type", "source", "detector", "wavelength_nominal", "units"], channel_rows)

    optodes_path = tmp_path / "sub-01_task-test_optodes.tsv"
    _write_tsv(optodes_path, ["name", "type", "x", "y", "z"], [
        {"name": "S1", "type": "source", "x": "0", "y": "0", "z": "0"},
        {"name": "S2", "type": "source", "x": "0.03", "y": "0", "z": "0"},
        {"name": "D1", "type": "detector", "x": "0.005", "y": "0", "z": "0"},
        {"name": "D2", "type": "detector", "x": "0.06", "y": "0", "z": "0"},
    ])
    nirs_json = tmp_path / "sub-01_task-test_nirs.json"
    nirs_json.write_text(json.dumps({"SamplingFrequency": 10.0}), encoding="utf-8")
    coordsystem_json = tmp_path / "sub-01_task-test_coordsystem.json"
    coordsystem_json.write_text(json.dumps({"NIRSCoordinateSystem": "CapTrak", "NIRSCoordinateUnits": "m"}), encoding="utf-8")
    events_path = tmp_path / "sub-01_task-test_events.tsv"
    _write_tsv(events_path, ["onset", "duration", "sample", "trial_type"], [
        {"onset": "0.1", "duration": "0.1", "sample": "1", "trial_type": "one"},
        {"onset": "0.3", "duration": "0.1", "sample": "3", "trial_type": "two"},
    ])
    return {
        "snirf_path": snirf_path,
        "nirs_json": nirs_json,
        "channels_tsv": channels_path,
        "optodes_tsv": optodes_path,
        "coordsystem_json": coordsystem_json,
        "events_tsv": events_path,
    }


def _validate(paths: dict[str, Path]):
    return validate_fnirs_recording(paths["snirf_path"], **{key: value for key, value in paths.items() if key != "snirf_path"})


def _codes(report) -> set[str]:
    return {issue.code for issue in report.issues}


def test_valid_cw_amplitude_checks_pairs_geometry_time_and_events(tmp_path: Path):
    report = _validate(_write_fixture(tmp_path))

    assert report.valid
    assert report.admissible
    assert report.classification == "cw_amplitude"
    assert report.summary["channel_count"] == 4
    assert report.summary["sample_count"] == 5
    assert report.summary["wavelengths_nm"] == [760.0, 850.0]
    assert report.summary["measurement_pair_count"] == 2
    assert report.summary["short_channel_pair_count"] == 1
    assert report.summary["short_channel_selector_m"] == 0.010
    assert report.summary["event_supported_count"] == 2
    assert report.summary["time_gap_sample_indices"] == []


def test_processed_optical_density_is_classified_and_admissible(tmp_path: Path):
    report = _validate(_write_fixture(tmp_path, representation="od"))

    assert report.valid
    assert report.admissible
    assert report.classification == "optical_density"


def test_processed_hbo_hbr_is_classified_but_unsupported_for_v0(tmp_path: Path):
    report = _validate(_write_fixture(tmp_path, representation="hb"))

    assert report.valid
    assert not report.admissible
    assert report.classification == "haemoglobin"
    assert "processed_hb_not_supported_v0" in _codes(report)


def test_nirs_json_parent_and_child_merge_complementary_metadata(tmp_path: Path):
    paths = _write_fixture(tmp_path)
    parent_dir = tmp_path / "dataset"
    parent_dir.mkdir()
    parent = parent_dir / "nirs.json"
    child = tmp_path / "sub-01_task-test_nirs.json"
    parent.write_text(json.dumps({"SamplingFrequency": 10.0, "PowerLineFrequency": 50.0}), encoding="utf-8")
    child.write_text(json.dumps({"TaskName": "test"}), encoding="utf-8")
    paths["nirs_json"] = [parent, child]

    report = _validate(paths)

    assert report.valid
    assert report.summary["sampling_frequency_hz"] == 10.0
    assert report.summary["nirs_json_sources"] == ["nirs.json", "sub-01_task-test_nirs.json"]


def test_coordsystem_json_inheritance_merges_and_applies_specific_override(tmp_path: Path):
    paths = _write_fixture(tmp_path)
    parent_dir = tmp_path / "dataset"
    parent_dir.mkdir()
    parent = parent_dir / "coordsystem.json"
    child = tmp_path / "sub-01_task-test_coordsystem.json"
    parent.write_text(json.dumps({"NIRSCoordinateSystem": "CapTrak", "NIRSCoordinateUnits": "cm"}), encoding="utf-8")
    child.write_text(json.dumps({"NIRSCoordinateUnits": "m"}), encoding="utf-8")
    paths["coordsystem_json"] = [parent, child]

    report = _validate(paths)

    assert report.valid
    assert report.summary["coordsystem_json_sources"] == ["coordsystem.json", "sub-01_task-test_coordsystem.json"]


def test_more_specific_nirs_json_value_overrides_parent(tmp_path: Path):
    paths = _write_fixture(tmp_path)
    parent_dir = tmp_path / "dataset"
    parent_dir.mkdir()
    parent = parent_dir / "nirs.json"
    child = tmp_path / "sub-01_task-test_nirs.json"
    parent.write_text(json.dumps({"SamplingFrequency": 5.0, "PowerLineFrequency": 50.0}), encoding="utf-8")
    child.write_text(json.dumps({"SamplingFrequency": 10.0}), encoding="utf-8")
    paths["nirs_json"] = [parent, child]

    report = _validate(paths)

    assert report.valid
    assert report.summary["sampling_frequency_hz"] == 10.0


@pytest.mark.parametrize(
    ("sidecar_field", "alternate_name", "issue_code", "alternate_json"),
    [
        ("nirs_json", "alternate_nirs.json", "nirs_json_ambiguous", {"SamplingFrequency": 10.0}),
        ("coordsystem_json", "alternate_coordsystem.json", "coordsystem_json_ambiguous", {"NIRSCoordinateSystem": "CapTrak", "NIRSCoordinateUnits": "m"}),
    ],
)
def test_json_inheritance_chain_rejects_distinct_same_directory_sidecars_without_flag(
    tmp_path: Path,
    sidecar_field: str,
    alternate_name: str,
    issue_code: str,
    alternate_json: dict[str, object],
):
    paths = _write_fixture(tmp_path)
    original = paths[sidecar_field]
    alternate = original.parent / alternate_name
    alternate.write_text(json.dumps(alternate_json), encoding="utf-8")
    paths[sidecar_field] = [original, alternate]  # type: ignore[assignment]

    report = _validate(paths)

    assert not report.valid
    assert not report.admissible
    assert issue_code in _codes(report)
    assert next(issue for issue in report.issues if issue.code == issue_code).severity == "error"
    assert "sidecar_unreadable" not in _codes(report)


def test_channel_type_unit_and_measurement_pair_mismatches_are_reason_coded(tmp_path: Path):
    paths = _write_fixture(tmp_path)
    rows = list(csv.DictReader(paths["channels_tsv"].open(encoding="utf-8"), delimiter="\t"))
    rows[0]["type"] = "NIRSDOD"
    rows[0]["units"] = "OD"
    _write_tsv(paths["channels_tsv"], ["name", "type", "source", "detector", "wavelength_nominal", "units"], rows)
    h5py = pytest.importorskip("h5py")
    with h5py.File(paths["snirf_path"], "r+") as h5:
        h5["nirs/data1/measurementList2/wavelengthIndex"][...] = 1

    report = _validate(paths)

    assert not report.valid
    assert {"channel_type_mismatch", "channel_unit_mismatch", "measurement_duplicate", "wavelength_pair_incomplete"} <= _codes(report)


def test_channels_tsv_rows_map_by_semantic_key_independent_of_row_order(tmp_path: Path):
    paths = _write_fixture(tmp_path)
    with paths["channels_tsv"].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    _write_tsv(paths["channels_tsv"], ["name", "type", "source", "detector", "wavelength_nominal", "units"], list(reversed(rows)))

    report = _validate(paths)

    assert report.valid
    assert report.admissible
    assert "channel_measurement_order_mismatch" not in _codes(report)
    assert "channel_measurement_mapping_mismatch" not in _codes(report)


def test_channel_semantic_key_must_match_snirf_descriptor(tmp_path: Path):
    paths = _write_fixture(tmp_path)
    with paths["channels_tsv"].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    rows[0]["source"] = "S99"
    _write_tsv(paths["channels_tsv"], ["name", "type", "source", "detector", "wavelength_nominal", "units"], rows)

    report = _validate(paths)

    assert not report.valid
    assert "channel_measurement_mapping_mismatch" in _codes(report)


@pytest.mark.parametrize(
    ("flag_name", "issue_code"),
    [
        ("nirs_json_ambiguous", "nirs_json_ambiguous"),
        ("coordsystem_json_ambiguous", "coordsystem_json_ambiguous"),
    ],
)
def test_ambiguous_inherited_json_resolution_is_an_error(tmp_path: Path, flag_name: str, issue_code: str):
    paths = _write_fixture(tmp_path)

    report = validate_fnirs_recording(
        paths["snirf_path"],
        nirs_json=None if flag_name == "nirs_json_ambiguous" else paths["nirs_json"],
        channels_tsv=paths["channels_tsv"],
        optodes_tsv=paths["optodes_tsv"],
        coordsystem_json=None if flag_name == "coordsystem_json_ambiguous" else paths["coordsystem_json"],
        events_tsv=paths["events_tsv"],
        **{flag_name: True},
    )

    assert not report.valid
    assert not report.admissible
    assert issue_code in _codes(report)
    assert next(issue for issue in report.issues if issue.code == issue_code).severity == "error"
    assert "nirs_json_path_missing" not in _codes(report)
    assert "coordsystem_json_path_missing" not in _codes(report)


@pytest.mark.parametrize(
    ("field", "flag_name", "issue_code"),
    [
        ("channels_tsv", "channels_tsv_ambiguous", "channels_tsv_ambiguous"),
        ("optodes_tsv", "optodes_tsv_ambiguous", "optodes_tsv_ambiguous"),
        ("events_tsv", "events_tsv_ambiguous", "events_tsv_ambiguous"),
    ],
)
def test_ambiguous_tsv_resolution_is_an_error_before_reading_missing_path(
    tmp_path: Path, field: str, flag_name: str, issue_code: str
):
    paths = _write_fixture(tmp_path)
    paths[field] = None  # type: ignore[assignment]

    report = validate_fnirs_recording(
        paths["snirf_path"],
        nirs_json=paths["nirs_json"],
        channels_tsv=paths["channels_tsv"],
        optodes_tsv=paths["optodes_tsv"],
        coordsystem_json=paths["coordsystem_json"],
        events_tsv=paths["events_tsv"],
        **{flag_name: True},
    )

    assert not report.valid
    assert not report.admissible
    assert issue_code in _codes(report)
    assert next(issue for issue in report.issues if issue.code == issue_code).severity == "error"
    assert f"{field}_path_missing" not in _codes(report)
    assert "sidecar_unreadable" not in _codes(report)


def test_bids_nirs_channel_semantic_columns_are_required(tmp_path: Path):
    paths = _write_fixture(tmp_path)
    with paths["channels_tsv"].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    _write_tsv(paths["channels_tsv"], ["name", "type", "units"], [
        {key: row[key] for key in ("name", "type", "units")} for row in rows
    ])

    report = _validate(paths)

    assert not report.valid
    assert {"channels_column_missing", "channel_measurement_mapping_mismatch"} & _codes(report)


def test_optode_coordinate_disagreement_is_rejected(tmp_path: Path):
    paths = _write_fixture(tmp_path)
    rows = list(csv.DictReader(paths["optodes_tsv"].open(encoding="utf-8"), delimiter="\t"))
    rows[2]["x"] = "0.010"
    _write_tsv(paths["optodes_tsv"], ["name", "type", "x", "y", "z"], rows)

    report = _validate(paths)

    assert not report.valid
    assert "optode_coordinates_mismatch" in _codes(report)


def test_time_gaps_and_events_outside_actual_support_are_reported(tmp_path: Path):
    paths = _write_fixture(tmp_path, times=[0.0, 0.1, 0.2, 0.5, 0.6])
    _write_tsv(paths["events_tsv"], ["onset", "duration", "sample", "trial_type"], [
        {"onset": "0.1", "duration": "0.1", "sample": "1", "trial_type": "supported"},
        {"onset": "0.5", "duration": "0.2", "sample": "5", "trial_type": "outside"},
    ])

    report = _validate(paths)

    assert not report.valid
    assert report.summary["time_gap_sample_indices"] == [3]
    assert "time_gaps_detected" in _codes(report)
    assert "event_duration_outside_time_support" in _codes(report)
    assert report.summary["event_supported_count"] == 1


def test_event_sample_must_match_onset_and_declared_sampling_frequency(tmp_path: Path):
    paths = _write_fixture(tmp_path)
    _write_tsv(paths["events_tsv"], ["onset", "duration", "sample", "trial_type"], [
        {"onset": "0.1", "duration": "0.1", "sample": "2", "trial_type": "mismatch"},
    ])

    report = _validate(paths)

    assert not report.valid
    assert "event_sample_onset_mismatch" in _codes(report)


def test_bids_events_are_run_relative_when_snirf_time_has_nonzero_origin(tmp_path: Path):
    paths = _write_fixture(tmp_path, times=[5.0, 5.1, 5.2, 5.3, 5.4])
    _write_tsv(paths["events_tsv"], ["onset", "duration", "sample", "trial_type"], [
        {"onset": "0", "duration": "0.1", "sample": "0", "trial_type": "run_start"},
        {"onset": "0.3", "duration": "0.1", "sample": "3", "trial_type": "later"},
    ])

    report = _validate(paths)

    assert report.valid
    assert report.summary["time_start_s"] == 5.0
    assert report.summary["time_end_s"] == 5.4
    assert report.summary["event_supported_count"] == 2
    assert "event_outside_time_support" not in _codes(report)
    assert "event_sample_onset_mismatch" not in _codes(report)


def test_negative_bids_onset_is_reported_unsupported_without_invalidating_recording(tmp_path: Path):
    paths = _write_fixture(tmp_path)
    _write_tsv(paths["events_tsv"], ["onset", "duration", "sample", "trial_type"], [
        {"onset": "-0.1", "duration": "0.05", "sample": "-1", "trial_type": "pre_acquisition"},
        {"onset": "0.1", "duration": "0.1", "sample": "1", "trial_type": "supported"},
    ])

    report = _validate(paths)

    assert report.valid
    assert report.admissible
    assert report.summary["event_supported_count"] == 1
    issue = next(issue for issue in report.issues if issue.code == "event_onset_outside_time_support")
    assert issue.severity == "unsupported"


def test_n_a_event_onset_is_unsupported_and_not_counted(tmp_path: Path):
    paths = _write_fixture(tmp_path)
    _write_tsv(paths["events_tsv"], ["onset", "duration", "sample", "trial_type"], [
        {"onset": "n/a", "duration": "n/a", "sample": "", "trial_type": "unlocated"},
        {"onset": "0.1", "duration": "0.1", "sample": "1", "trial_type": "supported"},
    ])

    report = _validate(paths)

    assert report.valid
    assert report.admissible
    assert report.summary["event_count"] == 2
    assert report.summary["event_supported_count"] == 1
    issue = next(issue for issue in report.issues if issue.code == "event_onset_unavailable")
    assert issue.severity == "unsupported"


def test_small_float_jitter_does_not_create_a_time_gap(tmp_path: Path):
    paths = _write_fixture(tmp_path, times=[0.0, 0.1, 0.2, 0.30000001, 0.4])

    report = _validate(paths)

    assert report.valid
    assert report.summary["time_gap_sample_indices"] == []
    assert "time_gaps_detected" not in _codes(report)


def test_time_vector_must_be_strictly_increasing(tmp_path: Path):
    paths = _write_fixture(tmp_path, times=[0.0, 0.1, 0.2, 0.2, 0.4])

    report = _validate(paths)

    assert not report.valid
    assert "time_vector_not_increasing" in _codes(report)


@pytest.mark.parametrize(
    ("layout_name", "reason_code"),
    [
        ("time", "snirf_time_layout_unsupported"),
        ("measurementList4", "snirf_measurement_layout_unsupported"),
    ],
)
def test_unimplemented_snirf_layouts_have_explicit_unsupported_reason(
    tmp_path: Path, layout_name: str, reason_code: str
):
    paths = _write_fixture(tmp_path)
    h5py = pytest.importorskip("h5py")
    with h5py.File(paths["snirf_path"], "r+") as h5:
        if layout_name == "time":
            h5.move("nirs/data1/time", "nirs/data1/timeValues")
        else:
            h5.move("nirs/data1/measurementList4", "nirs/data1/measurement4")

    report = _validate(paths)

    assert reason_code in _codes(report)
    assert "snirf_unreadable" not in _codes(report)
    assert next(issue for issue in report.issues if issue.code == reason_code).severity == "unsupported"
