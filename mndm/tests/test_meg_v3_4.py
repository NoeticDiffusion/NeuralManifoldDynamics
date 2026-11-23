"""Contract tests for experimental MEG MNPS 3.4 measurements."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("scipy")


def _config(*, enabled: bool = True) -> dict:
    return {
        "epoching": {"length_s": 10.0, "step_s": 10.0},
        "features": {
            "meg_psd": {"method": "welch", "fmin": 0.0, "fmax": 45.0},
            "permutation_entropy": {"order": 5, "delay": 1, "normalize": True},
        },
        "mnps_v3_4_meg": {
            "enabled": enabled,
            "definition_version": "3.4-meg-exp.1",
            "variants": ["existing_coordinate_set", "removal_only", "alpha_replacement"],
            "sensor_family_policy": "explicit_mag_grad",
            "scaling": {"mode": "raw_only", "anchor_manifest": None},
            "estimators": {
                "sample_entropy": {"m": 2, "r_scale": 0.2, "max_n": 300},
                "alpha_centroid_hz": {
                    "frequency_range_hz": [7.0, 13.0],
                    "psd": {
                        "method": "welch", "segment_sec": 4.0,
                        "overlap_fraction": 0.5, "window": "hann",
                        "detrend": "constant", "scaling": "density",
                    },
                },
                "alpha_envelope_cv": {
                    "band_hz": [8.0, 12.0],
                    "bandpass": {"order": 4},
                    "edge_support_guard_sec": 10.0,
                },
            },
        },
    }


def _signals(*, constant: bool = False) -> dict:
    sfreq = 100.0
    time = np.arange(0.0, 80.0, 1.0 / sfreq)
    if constant:
        mag = np.ones((2, time.size), dtype=np.float64)
        grad = np.ones((2, time.size), dtype=np.float64) * 2.0
    else:
        mag = np.vstack([
            np.sin(2 * np.pi * 8.0 * time),
            np.sin(2 * np.pi * 12.0 * time),
        ])
        modulation = 1.0 + 0.45 * np.sin(2 * np.pi * 0.2 * time)
        grad = np.vstack([
            np.sin(2 * np.pi * 10.0 * time),
            modulation * np.sin(2 * np.pi * 10.0 * time + 0.3),
        ])
    return {
        "signals": {"meg_mag": mag, "meg_grad": grad},
        "channels": {
            "meg_mag": ["MEG0111", "MEG0121"],
            "meg_grad": ["MEG0112", "MEG0122"],
        },
        "sfreq": sfreq,
        "dataset_id": "ds003645",
    }


def test_meg_v3_4_variants_keep_legacy_values_and_known_alpha_centroid():
    from mndm.features.meg import compute_meg_features
    from mndm.mnps.v3_4.meg_candidates import MEG_V34_VARIANTS
    from mndm.mnps.v3_4.meg_export import build_meg_measurement_export

    signals = _signals()
    config = _config()
    with_experimental = compute_meg_features(signals, config)
    legacy_only = compute_meg_features(signals, _config(enabled=False))

    pd.testing.assert_frame_equal(
        with_experimental[legacy_only.columns], legacy_only, check_dtype=False
    )
    assert "mnps_v3_4_meg_mag_alpha_centroid_hz" in with_experimental
    assert "mnps_v3_4_meg_grad_alpha_envelope_cv" in with_experimental
    assert set(with_experimental["mnps_v3_4_meg_mag_alpha_centroid_hz_status"]) == {"computed"}
    assert np.allclose(
        with_experimental["mnps_v3_4_meg_mag_alpha_centroid_hz"].to_numpy(),
        10.0,
        atol=0.4,
    )
    assert (with_experimental["mnps_v3_4_meg_mag_alpha_centroid_hz_channel_count"] == 2).all()
    assert (with_experimental["mnps_v3_4_meg_grad_alpha_envelope_cv_channel_count"] > 0).any()
    assert json.loads(with_experimental.loc[0, "mnps_v3_4_meg_mag_channel_labels_json"]) == [
        "MEG0111", "MEG0121"
    ]

    export = build_meg_measurement_export(with_experimental, config)
    family = export["sensor_families"]["mag"]["variants"]
    assert family["existing_coordinate_set"]["raw_values"].shape[1] == 9
    assert family["removal_only"]["raw_values"].shape[1] == 7
    assert family["alpha_replacement"]["raw_values"].shape[1] == 9
    assert list(family["existing_coordinate_set"]["coordinate_names"]) == list(
        MEG_V34_VARIANTS["existing_coordinate_set"]
    )
    existing = family["existing_coordinate_set"]
    replacement = family["alpha_replacement"]
    for shared in MEG_V34_VARIANTS["removal_only"]:
        a = existing["coordinate_names"].index(shared)
        b = replacement["coordinate_names"].index(shared)
        assert np.array_equal(existing["raw_values"][:, a], replacement["raw_values"][:, b], equal_nan=True)


def test_meg_v3_4_failed_sample_entropy_is_missing_not_permutation_entropy():
    from mndm.features.meg import compute_meg_features

    frame = compute_meg_features(_signals(constant=True), _config())
    assert np.isfinite(frame.loc[0, "meg_mag_sample_entropy"])
    assert np.isnan(frame.loc[0, "mnps_v3_4_meg_mag_sample_entropy"])
    assert frame.loc[0, "mnps_v3_4_meg_mag_sample_entropy_reason"] == "zero_variance"
    assert frame.loc[0, "mnps_v3_4_meg_mag_permutation_entropy_status"] == "computed"


def test_meg_v3_4_unknown_sensor_type_does_not_become_grad():
    from mndm.features.meg import compute_meg_features

    time = np.arange(0.0, 40.0, 0.01)
    unknown = {
        "signals": {"meg": np.vstack([np.sin(2 * np.pi * 10.0 * time)])},
        "channels": {"meg": ["UNCLASSIFIED_SENSOR"]},
        "sfreq": 100.0,
        "dataset_id": "ds003645",
    }
    frame = compute_meg_features(unknown, _config())
    assert set(frame["mnps_v3_4_meg_sensor_family_status"]) == {"unknown_sensor_type"}
    assert not any(column.startswith("mnps_v3_4_meg_grad_") for column in frame.columns)


def test_meg_v3_4_sidecar_preserves_measurement_export(tmp_path):
    h5py = pytest.importorskip("h5py")
    from core.io.h5_writer import write_measurement_export_sidecar
    from mndm.features.meg import compute_meg_features
    from mndm.mnps.v3_4.meg_export import build_meg_measurement_export

    frame = compute_meg_features(_signals(), _config())
    export = build_meg_measurement_export(frame, _config())
    target = tmp_path / "meg_measurements.h5"
    write_measurement_export_sidecar(
        target,
        measurement_id="v3_4_meg",
        measurement_payload=export,
        recording_identity={"dataset_id": "ds003645", "subject_id": "sub-synthetic"},
        skip_reason="synthetic_contract_test",
    )
    with h5py.File(target, "r") as h5:
        root = h5["measurements/v3_4_meg"]
        assert root.attrs["_schema_version"] == "mndm.meg_mnps_v3_4.experimental.v1"
        mag = root["sensor_families/mag"]
        assert mag["variants/removal_only/raw_values"].shape[1] == 7
        assert mag["variants/alpha_replacement/raw_values"].shape[1] == 9
        assert mag["support/alpha_centroid_hz_estimator_id"].shape[0] == len(frame)
        assert bool(h5.attrs["measurement_only"])
