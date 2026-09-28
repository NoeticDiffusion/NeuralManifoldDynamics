import json
from urllib.parse import quote

import numpy as np
import pandas as pd
import pytest

import mndm.pipeline.fnirs_regional as regional_module
from mndm.pipeline.fnirs_regional import (
    FnirsRegionalContractError,
    aggregate_fnirs_scalp_sectors,
    compute_channel_loss_sensitivity,
    derive_fnirs_scalp_sector_map,
    export_fnirs_scalp_sector_pilot,
)


@pytest.fixture
def regional_inputs():
    coordsystem = {
        "NIRSCoordinateSystem": "CapTrak",
        "NIRSCoordinateUnits": "m",
        "NIRSCoordinateSystemDescription": "X from LPA to RPA; Y through nasion; corresponds to RAS orientation.",
    }
    optodes = pd.DataFrame([
        {"name": "S1", "type": "source", "x": -0.040, "y": 0.040, "z": 0.080},
        {"name": "D1", "type": "detector", "x": -0.010, "y": 0.040, "z": 0.080},
        {"name": "S2", "type": "source", "x": -0.040, "y": -0.040, "z": 0.080},
        {"name": "D2", "type": "detector", "x": -0.010, "y": -0.040, "z": 0.080},
        {"name": "S3", "type": "source", "x": 0.010, "y": 0.040, "z": 0.080},
        {"name": "D3", "type": "detector", "x": 0.040, "y": 0.040, "z": 0.080},
        {"name": "S4", "type": "source", "x": 0.010, "y": -0.040, "z": 0.080},
        {"name": "D4", "type": "detector", "x": 0.040, "y": -0.040, "z": 0.080},
        {"name": "S5", "type": "source", "x": 0.040, "y": 0.020, "z": 0.080},
        {"name": "D5", "type": "detector", "x": 0.047, "y": 0.020, "z": 0.080},
        {"name": "S6", "type": "source", "x": -0.045, "y": 0.030, "z": 0.080},
        {"name": "D6", "type": "detector", "x": -0.015, "y": 0.030, "z": 0.080},
        {"name": "S7", "type": "source", "x": -0.045, "y": -0.030, "z": 0.080},
        {"name": "D7", "type": "detector", "x": -0.015, "y": -0.030, "z": 0.080},
        {"name": "S8", "type": "source", "x": 0.015, "y": 0.030, "z": 0.080},
        {"name": "D8", "type": "detector", "x": 0.045, "y": 0.030, "z": 0.080},
        {"name": "S9", "type": "source", "x": 0.015, "y": -0.030, "z": 0.080},
        {"name": "D9", "type": "detector", "x": 0.045, "y": -0.030, "z": 0.080},
    ])
    pairs = [("S1", "D1"), ("S2", "D2"), ("S3", "D3"), ("S4", "D4"), ("S5", "D5"),
             ("S6", "D6"), ("S7", "D7"), ("S8", "D8"), ("S9", "D9")]
    channels = pd.DataFrame([
        {"source": source, "detector": detector, "wavelength_nominal": 760.0}
        for source, detector in pairs
    ])
    channel_map = []
    for source, detector in pairs:
        for chromophore in ("hbo", "hbr"):
            name = f"{source}_{detector} {chromophore}"
            channel_map.append({
                "column_key": quote(name, safe=""),
                "channel_name": name,
                "chromophore": chromophore,
                "unit": "M",
            })
    region_map = derive_fnirs_scalp_sector_map(channels, optodes, coordsystem, channel_map)
    features = pd.DataFrame({
        "file": ["pilot.snirf", "pilot.snirf"],
        "epoch_id": [0, 1],
        "t_start": [0.0, 5.0],
        "t_end": [10.0, 15.0],
        "fnirs_window_id": [0, 1],
        "fnirs_window_start_s": [0.0, 5.0],
        "fnirs_window_end_s": [10.0, 15.0],
        "fnirs_actual_t_start_s": [0.0, 5.0],
        "fnirs_actual_t_end_s": [10.0, 15.0],
        "fnirs_actual_covered_duration_s": [10.0, 10.0],
        "fnirs_clock_support_fraction": [1.0, 1.0],
        "fnirs_min_valid_support_fraction": [1.0, 1.0],
        "fnirs_source_sha256": ["a" * 64, "a" * 64],
        "fnirs_channel_map_json": [json.dumps(channel_map, separators=(",", ":"))] * 2,
        "fnirs_measurement_validity": ["not_assessed"] * 2,
        "fnirs_qualification_status": ["not_assessed"] * 2,
        "fnirs_qc_status": ["not_assessed"] * 2,
        "qc_ok_fnirs": pd.array([pd.NA, pd.NA], dtype="boolean"),
    })
    for i, item in enumerate(channel_map):
        key = item["column_key"]
        features[f"fnirs_mean_M__{key}"] = [float(i + 1), float(i + 2)]
        features[f"fnirs_valid_support_fraction__{key}"] = [1.0, 1.0]
        features[f"fnirs_rejection_reason__{key}"] = [pd.NA, pd.NA]
    return coordsystem, optodes, channels, channel_map, region_map, features


def test_map_assigns_geometry_only_and_excludes_short_pairs(regional_inputs):
    _, _, _, _, region_map, _ = regional_inputs
    included = region_map[region_map["distance_class"] == "long_included"]
    excluded = region_map[region_map["distance_class"] == "short_excluded"]
    assert set(included["region_name"]) == {
        "left_anterior_scalp", "left_posterior_scalp",
        "right_anterior_scalp", "right_posterior_scalp",
    }
    assert len(included) == 16
    assert len(excluded) == 2
    assert set(included["region_kind"]) == {"geometric_scalp_sector"}
    assert excluded["region_name"].isna().all()


def test_aggregate_requires_full_fixed_channel_coverage_and_keeps_unknown_qc(regional_inputs):
    _, _, _, channel_map, region_map, features = regional_inputs
    for item in channel_map:
        features[f"fnirs_rejection_reason__{item['column_key']}"] = ""
    result = aggregate_fnirs_scalp_sectors(features, region_map)
    assert len(result) == 16
    assert set(result["region_status"]) == {"computed"}
    assert (result["channel_coverage_fraction"] == 1.0).all()
    assert result["qc_ok_fnirs"].isna().all()
    assert (result["is_qualified"] == 0).all()
    assert (result["region_kind"] == "geometric_scalp_sector").all()

    missing_channel = next(item for item in channel_map if item["channel_name"] == "S1_D1 hbo")
    features.loc[0, f"fnirs_mean_M__{missing_channel['column_key']}"] = np.nan
    result = aggregate_fnirs_scalp_sectors(features, region_map)
    row = result[(result["window_id"] == 0) & (result["region_name"] == "left_anterior_scalp") & (result["chromophore"] == "hbo")].iloc[0]
    assert row["region_status"] == "incomplete_channel_coverage"
    assert row["channel_count_valid"] == 1
    assert row["channel_coverage_fraction"] == 0.5
    assert np.isnan(row["region_mean_M"])


def test_support_roundoff_matches_g3_relative_tolerance(regional_inputs):
    _, _, _, channel_map, region_map, features = regional_inputs
    for item in channel_map:
        key = item["column_key"]
        features[f"fnirs_valid_support_fraction__{key}"] = 1.0 - 1e-14
        features[f"fnirs_rejection_reason__{key}"] = ""
    result = aggregate_fnirs_scalp_sectors(features, region_map)
    assert set(result["region_status"]) == {"computed"}


def test_map_fails_closed_on_non_captrak_or_ambiguous_channels(regional_inputs):
    coordsystem, optodes, channels, channel_map, _, _ = regional_inputs
    with pytest.raises(FnirsRegionalContractError, match="CapTrak"):
        derive_fnirs_scalp_sector_map(channels, optodes, {**coordsystem, "NIRSCoordinateSystem": "MNI152"}, channel_map)
    duplicate = pd.concat([channels, channels.iloc[[0]]], ignore_index=True)
    with pytest.raises(FnirsRegionalContractError, match="duplicate"):
        derive_fnirs_scalp_sector_map(duplicate, optodes, coordsystem, channel_map)
    with pytest.raises(FnirsRegionalContractError, match="freezes"):
        derive_fnirs_scalp_sector_map(channels, optodes, coordsystem, channel_map, short_distance_mm=12.0)
    reverse_x = {**coordsystem, "NIRSCoordinateSystemDescription": "CapTrak RAS, X goes from RPA to LPA and Y through nasion."}
    with pytest.raises(FnirsRegionalContractError, match="LPA-to-RPA"):
        derive_fnirs_scalp_sector_map(channels, optodes, reverse_x, channel_map)
    reverse_y = {**coordsystem, "NIRSCoordinateSystemDescription": "CapTrak RAS; LPA to RPA; posterior is positive Y; nasion landmark."}
    with pytest.raises(FnirsRegionalContractError, match="y-axis direction"):
        derive_fnirs_scalp_sector_map(channels, optodes, reverse_y, channel_map)
    duplicate_channel_map = channel_map + [dict(channel_map[0])]
    with pytest.raises(FnirsRegionalContractError, match="Duplicate G3 channel-map"):
        derive_fnirs_scalp_sector_map(channels, optodes, coordsystem, duplicate_channel_map)


def test_aggregation_rejects_tampered_spatial_assignment(regional_inputs):
    _, _, _, _, region_map, features = regional_inputs
    tampered = region_map.copy()
    index = tampered.index[tampered["distance_class"] == "long_included"][0]
    tampered.loc[index, "region_name"] = "motor_cortex"
    with pytest.raises(FnirsRegionalContractError, match="frozen geometry rule"):
        aggregate_fnirs_scalp_sectors(features, tampered)


def test_regional_aggregation_rejects_qc_or_qualification_upgrade(regional_inputs):
    _, _, _, _, region_map, features = regional_inputs
    features["qc_ok_fnirs"] = True
    with pytest.raises(FnirsRegionalContractError, match="QC"):
        aggregate_fnirs_scalp_sectors(features, region_map)
    features["qc_ok_fnirs"] = pd.NA
    features["fnirs_qualification_status"] = "qualified"
    with pytest.raises(FnirsRegionalContractError, match="qualification"):
        aggregate_fnirs_scalp_sectors(features, region_map)


def test_regional_aggregation_rejects_bad_source_identity_or_hash(regional_inputs):
    _, _, _, _, region_map, features = regional_inputs
    features["fnirs_source_sha256"] = "bad"
    with pytest.raises(FnirsRegionalContractError, match="checksum"):
        aggregate_fnirs_scalp_sectors(features, region_map)
    features["fnirs_source_sha256"] = "a" * 64
    features.loc[0, "file"] = pd.NA
    with pytest.raises(FnirsRegionalContractError, match="nonempty source identity"):
        aggregate_fnirs_scalp_sectors(features, region_map)


def test_regional_aggregation_rejects_support_outside_unit_interval(regional_inputs):
    _, _, _, channel_map, region_map, features = regional_inputs
    key = channel_map[0]["column_key"]
    features.loc[0, f"fnirs_valid_support_fraction__{key}"] = 2.0
    with pytest.raises(FnirsRegionalContractError, match=r"fraction must be in \[0, 1\]"):
        aggregate_fnirs_scalp_sectors(features, region_map)


def test_feature_table_rejects_duplicate_channel_map_entries(regional_inputs):
    _, _, _, channel_map, region_map, features = regional_inputs
    features["fnirs_channel_map_json"] = json.dumps(channel_map + [dict(channel_map[0])])
    with pytest.raises(FnirsRegionalContractError, match="duplicate channel names/column keys"):
        aggregate_fnirs_scalp_sectors(features, region_map)


def test_channel_loss_sensitivity_records_predefined_leave_one_pair_out(regional_inputs):
    _, _, _, _, region_map, features = regional_inputs
    result = compute_channel_loss_sensitivity(features, region_map)
    assert len(result) == 64
    assert result["omitted_pair"].nunique() == 8
    assert (result["windows_compared"] == 2).all()
    assert (result["max_abs_change_M"] >= 0).all()


def test_cli_export_writes_manifest_map_and_unqualified_regional_csv(tmp_path, regional_inputs):
    coordsystem, optodes, channels, channel_map, _, features = regional_inputs
    features_path = tmp_path / "features.csv"
    features.to_csv(features_path, index=False)
    channels_path = tmp_path / "channels.tsv"
    channels.to_csv(channels_path, sep="\t", index=False)
    optodes_path = tmp_path / "optodes.tsv"
    optodes.to_csv(optodes_path, sep="\t", index=False)
    coordsystem_path = tmp_path / "coordsystem.json"
    coordsystem_path.write_text(json.dumps(coordsystem), encoding="utf-8")
    out_dir = tmp_path / "out"
    manifest = export_fnirs_scalp_sector_pilot(
        features_path, channels_path, optodes_path, coordsystem_path, out_dir
    )
    assert manifest["counts"]["complete_regional_rows"] == 16
    assert manifest["claim_status"] == "no_biological_claim"
    assert (out_dir / "channel_region_map.json").is_file()
    regional = pd.read_csv(out_dir / "regional_features.csv")
    assert set(regional["region_kind"]) == {"geometric_scalp_sector"}
    assert set(regional["qualification_status"]) == {"not_assessed"}
    with pytest.raises(FileExistsError, match="already exists"):
        export_fnirs_scalp_sector_pilot(features_path, channels_path, optodes_path, coordsystem_path, out_dir)


def test_export_rolls_back_all_outputs_when_one_replace_fails(tmp_path, regional_inputs, monkeypatch):
    coordsystem, optodes, channels, _, _, features = regional_inputs
    features_path = tmp_path / "features.csv"
    features.to_csv(features_path, index=False)
    channels_path = tmp_path / "channels.tsv"
    channels.to_csv(channels_path, sep="\t", index=False)
    optodes_path = tmp_path / "optodes.tsv"
    optodes.to_csv(optodes_path, sep="\t", index=False)
    coordsystem_path = tmp_path / "coordsystem.json"
    coordsystem_path.write_text(json.dumps(coordsystem), encoding="utf-8")
    out_dir = tmp_path / "transactional-out"
    export_fnirs_scalp_sector_pilot(
        features_path, channels_path, optodes_path, coordsystem_path, out_dir
    )
    original_bytes = {path.name: path.read_bytes() for path in out_dir.iterdir()}
    original_replace = regional_module.os.replace

    def fail_once(source, destination):
        source_path = str(source)
        destination_path = str(destination)
        if destination_path.endswith("channel_loss_sensitivity.csv") and ".tmp" in source_path:
            raise OSError("simulated output replace failure")
        return original_replace(source, destination)

    monkeypatch.setattr(regional_module.os, "replace", fail_once)
    with pytest.raises(OSError, match="simulated output replace failure"):
        export_fnirs_scalp_sector_pilot(
            features_path, channels_path, optodes_path, coordsystem_path, out_dir, overwrite=True
        )
    assert {path.name: path.read_bytes() for path in out_dir.iterdir()} == original_bytes
