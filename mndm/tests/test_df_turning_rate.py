"""Realized successive-increment turning. Undefined is NaN, not zero."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))

from mndm.dynamical_families import (
    TURNING_SCHEMA_VERSION,
    WRITABLE_FAMILY_IDS,
    family_forbids,
    get_family,
)
from mndm.dynamical_families.turning import estimate_turning_rate
from mndm.dynamical_families.measurement_register import (
    MEASUREMENT_ID_CLOUD_VOLUME,
    MEASUREMENT_ID_TURNING_ANGLE,
    MEASUREMENT_ID_TURNING_RATE,
    QUALIFICATION_TURNING,
    RELATION_NEW_MEASURE,
    compatibility_entry,
    register_entry,
)
from mndm.pipeline.dynamical_families_export import build_dynamical_families_export
from mndm.pipeline.summary import _build_dynamical_families_export_for_layers
from mndm.schema import MNPSPayload


def _times(n: int = 160, dt: float = 0.25) -> np.ndarray:
    return np.arange(n, dtype=float) * dt


def _turning_config(*, enabled: bool = True, **overrides) -> dict:
    block = {
        "enabled": enabled,
        "min_samples": 40,
        "min_pairs": 20,
        "min_displacement": 1e-12,
    }
    block.update(overrides)
    return {"dynamical_families": {"enabled": True, "turning": block}}


def test_straight_line_turning_is_near_zero() -> None:
    time = _times()
    state = np.zeros((time.size, 3), dtype=np.float32)
    state[:, 0] = time.astype(np.float32)
    result = estimate_turning_rate(state, time, coordinate_layer="coords_3d_subject_anchored")
    assert result["computation_status"] == "computed"
    leaf = result[MEASUREMENT_ID_TURNING_RATE]
    assert int(leaf["interpretation_level"]) == 0
    assert leaf["qualification_status"] == QUALIFICATION_TURNING
    assert leaf["summary"]["undefined_not_zero"] is True
    assert float(leaf["summary"]["turning_rate_median"]) == pytest.approx(0.0, abs=1e-6)
    assert float(result[MEASUREMENT_ID_TURNING_ANGLE]["summary"]["turning_angle_median"]) == pytest.approx(
        0.0, abs=1e-6
    )


def test_uniform_circle_recovers_angular_rate() -> None:
    dt = 0.25
    omega = 0.5
    time = _times(dt=dt)
    state = np.zeros((time.size, 3), dtype=float)
    state[:, 0] = np.cos(omega * time)
    state[:, 1] = np.sin(omega * time)
    result = estimate_turning_rate(
        state.astype(np.float32), time, coordinate_layer="coords_3d_subject_anchored"
    )
    assert result["computation_status"] == "computed"
    median = float(result["summary"]["turning_rate_median"])
    assert median == pytest.approx(omega, rel=0.05, abs=0.02)


def test_tiny_displacement_is_undefined_not_zero() -> None:
    rng = np.random.default_rng(3)
    time = _times(n=80)
    state = (1e-16 * rng.normal(size=(time.size, 3))).astype(np.float32)
    result = estimate_turning_rate(
        state, time, coordinate_layer="coords_3d_subject_anchored", min_samples=40, min_pairs=20
    )
    assert result["computation_status"] == "insufficient_support"
    assert result["failure_reason"] == "turning_direction_undefined"
    assert int(result["summary"]["n_defined_turning"]) == 0
    assert 0.0 not in np.asarray(result.get("series", {}).get("turning_rate", []), dtype=float)


def test_nine_d_and_nonpositive_floor_are_not_testable() -> None:
    time = _times(n=80)
    state = np.zeros((time.size, 3), dtype=np.float32)
    state[:, 0] = time.astype(np.float32)
    nine = np.concatenate([state, state, state], axis=1)
    result = estimate_turning_rate(nine, time, coordinate_layer="coords_9d")
    assert result["failure_reason"] == "turning_subject_anchored_3d_only"
    bad = estimate_turning_rate(
        state, time, coordinate_layer="coords_3d_subject_anchored", min_displacement=0.0
    )
    assert bad["failure_reason"] == "turning_min_displacement_invalid"


def test_yaml_forbids_and_common_default_off() -> None:
    import yaml

    time = _times(n=80)
    state = np.zeros((time.size, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="cloud_volume_change_rate_level1"):
        build_dynamical_families_export(
            config={
                "dynamical_families": {
                    "enabled": True,
                    "turning": {"enabled": True, "cloud_volume_change_rate_level1": True},
                }
            },
            state=state,
            time=time,
            stage=None,
            segment_id=np.zeros(time.size, dtype=np.int32),
            coordinate_layer="coords_3d_subject_anchored",
            coordinate_names=["m", "d", "e"],
        )
    with pytest.raises(ValueError, match="zero_fill_undefined_direction"):
        build_dynamical_families_export(
            config={
                "dynamical_families": {
                    "enabled": True,
                    "turning": {"enabled": True, "zero_fill_undefined_direction": True},
                }
            },
            state=state,
            time=time,
            stage=None,
            segment_id=np.zeros(time.size, dtype=np.int32),
            coordinate_layer="coords_3d_subject_anchored",
            coordinate_names=["m", "d", "e"],
        )
    common = Path(__file__).resolve().parents[1] / "config" / "config_ingest_common_dynamical_families.yaml"
    with common.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    assert cfg["dynamical_families"]["turning"]["enabled"] is True
    sources = common.parent / "sources" / "other"
    from core.config_loader import load_config

    for name in (
        "config_ingest_physionet_i-care_2_1_dynamical_families.yaml",
        "config_ingest_physionet_i-care_2_1_part1_0_12h_dynamical_families.yaml",
        "config_ingest_physionet_i-care_2_1_next_140_0_12h_dynamical_families.yaml",
        "config_ingest_physionet_i-care_2_1_part1_0_12h_amplification_pilot.yaml",
        "config_ingest_physionet_i-care_2_1_next_140_0_12h_amplification_pilot.yaml",
    ):
        merged = load_config(sources / name)
        assert merged["dynamical_families"]["turning"]["enabled"] is True

    for name in (
        "config_ingest_physionet_i-care_2_1_part1_0_12h_history_turning_pilot.yaml",
        "config_ingest_physionet_i-care_2_1_next_140_0_12h_history_turning_pilot.yaml",
    ):
        with (sources / name).open(encoding="utf-8") as handle:
            overlay = yaml.safe_load(handle)
        turn = (overlay.get("dynamical_families") or {}).get("turning") or {}
        assert turn.get("enabled") is True
        merged = load_config(sources / name)
        families = merged["dynamical_families"]
        assert families["turning"]["enabled"] is True
        assert families["history"]["enabled"] is True
        assert families["one_step"]["enabled"] is True
        assert families["amplification"]["enabled"] is True


def test_registry_h5_and_cloud_volume_withheld(tmp_path: Path) -> None:
    assert "turning" in WRITABLE_FAMILY_IDS
    assert family_forbids("turning", "cloud_volume_change_rate_level1")
    assert family_forbids("turning", "operator_rotation_as_turning")
    assert get_family("turning")["schema"] == TURNING_SCHEMA_VERSION
    assert compatibility_entry(MEASUREMENT_ID_TURNING_RATE)["relation"] == RELATION_NEW_MEASURE
    assert compatibility_entry(MEASUREMENT_ID_TURNING_ANGLE)["relation"] == RELATION_NEW_MEASURE
    assert compatibility_entry(MEASUREMENT_ID_CLOUD_VOLUME)["relation"] == RELATION_NEW_MEASURE
    assert register_entry(MEASUREMENT_ID_CLOUD_VOLUME)["physical_path"] == (
        "/dynamical_families/amplification/v1/cloud_volume_change_rate_level1"
    )
    assert register_entry(MEASUREMENT_ID_CLOUD_VOLUME)["written_by_turning_family"] is False
    time = _times()
    state = np.zeros((time.size, 3), dtype=np.float32)
    state[:, 0] = time.astype(np.float32)
    export = build_dynamical_families_export(
        config=_turning_config(),
        state=state,
        time=time,
        stage=None,
        segment_id=np.zeros(time.size, dtype=np.int32),
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
    )
    from core.io.h5_writer import write_h5
    import h5py

    payload = MNPSPayload(
        time=time,
        x=state,
        x_dot=np.zeros_like(state),
        dynamical_families=export,
    )
    output = write_h5(tmp_path / "turning.h5", "turn", payload)
    with h5py.File(output, "r") as handle:
        root = "/dynamical_families/turning/v1"
        assert f"{root}/{MEASUREMENT_ID_TURNING_RATE}" in handle
        assert f"{root}/{MEASUREMENT_ID_TURNING_ANGLE}" in handle
        assert f"{root}/cloud_volume_change_rate_level1" not in handle
        grain = handle[f"{root}/grain"]
        assert grain["native"][()].decode() == "window"
        assert grain["repeated_measure"][()].decode() == "true"
        group = handle[f"{root}/{MEASUREMENT_ID_TURNING_RATE}"]
        assert group["qualification_status"][()].decode() == QUALIFICATION_TURNING
        assert int(group["interpretation_level"][()]) == 0


def test_missing_coordinate_layer_keeps_both_leaves_and_refuses_aliases() -> None:
    export = _build_dynamical_families_export_for_layers(
        config={
            "dynamical_families": {
                "enabled": True,
                "coordinate_layer": "subject_anchored",
                "turning": {"enabled": True},
            }
        },
        x_subject_anchored=None,
        x_cohort_anchored=None,
        time=_times(n=8),
        stage=None,
        segment_id=None,
    )
    turning = export["turning"]
    assert turning["computation_status"] == "not_testable"
    assert turning["failure_reason"] == "requested_coordinate_layer_not_available"
    assert MEASUREMENT_ID_TURNING_RATE in turning
    assert MEASUREMENT_ID_TURNING_ANGLE in turning
    assert turning[MEASUREMENT_ID_TURNING_RATE]["measurement_id"] == MEASUREMENT_ID_TURNING_RATE
    assert turning[MEASUREMENT_ID_TURNING_ANGLE]["measurement_id"] == MEASUREMENT_ID_TURNING_ANGLE
    with pytest.raises(ValueError, match="cloud_volume_change_rate_level1"):
        _build_dynamical_families_export_for_layers(
            config={
                "dynamical_families": {
                    "enabled": True,
                    "coordinate_layer": "subject_anchored",
                    "turning": {"enabled": True, "cloud_volume_change_rate_level1": True},
                }
            },
            x_subject_anchored=None,
            x_cohort_anchored=None,
            time=_times(n=8),
            stage=None,
            segment_id=None,
        )
