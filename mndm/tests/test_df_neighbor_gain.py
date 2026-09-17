"""Same-pair neighbor gain q90 (amplification L1). Not operator max-gain."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))

from mndm.dynamical_families import (
    AMPLIFICATION_SCHEMA_VERSION,
    WRITABLE_FAMILY_IDS,
    family_forbids,
    get_family,
)
from mndm.dynamical_families.amplification import estimate_neighbor_gain_q90
from mndm.dynamical_families.measurement_register import (
    MEASUREMENT_ID_CLOUD_VOLUME,
    MEASUREMENT_ID_HISTORY_PREDICTIVE_GAIN,
    MEASUREMENT_ID_NEIGHBOR_GAIN,
    MEASUREMENT_ID_NEIGHBOR_GAIN_RATE,
    MEASUREMENT_ID_NEIGHBOR_SEPARATION,
    MEASUREMENT_ID_OPERATOR_MAX_GAIN,
    QUALIFICATION_CLOUD_VOLUME,
    QUALIFICATION_NOT_ASSESSED,
    QUALIFICATION_SAME_PAIR_GAIN,
    QUALIFICATION_SAME_PAIR_GAIN_RATE,
    QUALIFICATION_SAME_PAIR_SEPARATION,
    RELATION_NEW_MEASURE,
    compatibility_entry,
    register_entry,
)
from mndm.pipeline.dynamical_families_export import build_dynamical_families_export
from mndm.pipeline.summary import _build_dynamical_families_export_for_layers
from mndm.schema import MNPSPayload


def _expanding(*, n: int = 240, dt: float = 0.25, scale: float = 1.15, seed: int = 3, start_scale: float = 1.0):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 3), dtype=float)
    x[0] = start_scale * rng.normal(size=3)
    noise = 0.02 * rng.normal(size=(n - 1, 3))
    for t in range(n - 1):
        x[t + 1] = scale * x[t] + noise[t]
    time = np.arange(n, dtype=float) * dt
    return x.astype(np.float32), time


def _amplification_config(*, enabled: bool = True, **overrides) -> dict:
    block = {
        "enabled": enabled,
        "neighborhood": {"k": 12},
        "min_samples": 40,
        "min_neighborhood_samples": 8,
        "q": 0.90,
        "distance_epsilon": 1e-12,
    }
    block.update(overrides)
    return {"dynamical_families": {"enabled": True, "amplification": block}}


def test_expanding_map_has_gain_above_one() -> None:
    state, time = _expanding(scale=1.2)
    result = estimate_neighbor_gain_q90(
        state,
        time,
        coordinate_layer="coords_3d_subject_anchored",
        neighborhood_k=12,
        min_samples=40,
        min_neighborhood_samples=8,
    )
    assert result["computation_status"] == "computed"
    leaf = result[MEASUREMENT_ID_NEIGHBOR_GAIN]
    assert leaf["computation_status"] == "computed"
    assert int(leaf["interpretation_level"]) == 1
    assert leaf["qualification_status"] == QUALIFICATION_SAME_PAIR_GAIN
    assert leaf["summary"]["not_operator_max_gain"] is True
    assert leaf["summary"]["not_resampled_at_target"] is True
    assert leaf["summary"]["aggregation"] == "per_source_q90_not_pooled_pair_q90"
    values = np.asarray(leaf["series"]["neighbor_gain_q90"], dtype=float)
    finite = values[np.isfinite(values)]
    assert finite.size > 20
    assert float(np.median(finite)) > 1.05
    sep = result[MEASUREMENT_ID_NEIGHBOR_SEPARATION]
    assert sep["computation_status"] == "computed"
    assert sep["qualification_status"] == QUALIFICATION_SAME_PAIR_SEPARATION
    assert sep["summary"]["not_spectral_abscissa"] is True
    sep_vals = np.asarray(sep["series"]["neighbor_separation_rate"], dtype=float)
    assert float(np.nanmedian(sep_vals)) > 0.0
    rate = result[MEASUREMENT_ID_NEIGHBOR_GAIN_RATE]
    assert rate["computation_status"] == "computed"
    assert rate["qualification_status"] == QUALIFICATION_SAME_PAIR_GAIN_RATE
    rate_vals = np.asarray(rate["series"]["neighbor_gain_rate_q90"], dtype=float)
    assert float(np.nanmedian(rate_vals)) > 0.0
    dt = float(leaf["summary"]["nominal_dt_sec"])
    mask = np.isfinite(values) & (values > 0.0) & np.isfinite(rate_vals)
    np.testing.assert_allclose(rate_vals[mask], np.log(values[mask]) / dt, rtol=1e-5, atol=1e-6)
    cloud = result[MEASUREMENT_ID_CLOUD_VOLUME]
    assert cloud["computation_status"] == "computed"
    assert cloud["qualification_status"] == QUALIFICATION_CLOUD_VOLUME
    assert cloud["summary"]["not_operator_volume"] is True
    assert cloud["summary"]["epsilon_logdet_regularized"] is True
    cloud_vals = np.asarray(cloud["series"]["cloud_volume_change_rate"], dtype=float)
    assert float(np.nanmedian(cloud_vals)) > 0.0


def test_contracting_early_window_has_gain_below_one() -> None:
    state, time = _expanding(scale=0.75, seed=5, start_scale=8.0, n=120)
    result = estimate_neighbor_gain_q90(
        state,
        time,
        coordinate_layer="coords_3d_subject_anchored",
        neighborhood_k=12,
        min_samples=40,
        min_neighborhood_samples=8,
    )
    values = np.asarray(
        result[MEASUREMENT_ID_NEIGHBOR_GAIN]["series"]["neighbor_gain_q90"], dtype=float
    )
    early = values[:20]
    finite = early[np.isfinite(early)]
    assert finite.size >= 8
    assert float(np.median(finite)) < 0.95


def test_rigid_translation_gain_is_near_one() -> None:
    n = 240
    rng = np.random.default_rng(9)
    state = np.stack(
        [0.5 * np.arange(n), np.zeros(n), np.zeros(n)], axis=1
    ).astype(np.float32)
    state += np.float32(1e-4) * rng.normal(size=state.shape).astype(np.float32)
    time = np.arange(n, dtype=float) * 0.25
    result = estimate_neighbor_gain_q90(
        state,
        time,
        coordinate_layer="coords_3d_subject_anchored",
        neighborhood_k=12,
        min_samples=40,
        min_neighborhood_samples=8,
    )
    assert result["computation_status"] == "computed"
    values = np.asarray(
        result[MEASUREMENT_ID_NEIGHBOR_GAIN]["series"]["neighbor_gain_q90"], dtype=float
    )
    finite = values[np.isfinite(values)]
    median = float(np.median(finite))
    assert 0.85 < median < 1.15
    sep_vals = np.asarray(
        result[MEASUREMENT_ID_NEIGHBOR_SEPARATION]["series"]["neighbor_separation_rate"],
        dtype=float,
    )
    assert abs(float(np.nanmedian(sep_vals))) < 0.4
    cloud_vals = np.asarray(
        result[MEASUREMENT_ID_CLOUD_VOLUME]["series"]["cloud_volume_change_rate"],
        dtype=float,
    )
    assert abs(float(np.nanmedian(cloud_vals))) < 0.4


def test_nine_d_and_other_q_are_not_testable() -> None:
    state, time = _expanding(n=80)
    nine = np.concatenate([state, state, state], axis=1)
    result = estimate_neighbor_gain_q90(nine, time, coordinate_layer="coords_9d")
    assert result["failure_reason"] == "amplification_subject_anchored_3d_only"
    assert MEASUREMENT_ID_NEIGHBOR_SEPARATION in result
    assert result[MEASUREMENT_ID_NEIGHBOR_SEPARATION]["qualification_status"] == QUALIFICATION_NOT_ASSESSED
    assert MEASUREMENT_ID_CLOUD_VOLUME in result
    other_q = estimate_neighbor_gain_q90(
        state, time, coordinate_layer="coords_3d_subject_anchored", q=0.5
    )
    assert other_q["failure_reason"] == "neighbor_gain_q_not_implemented"
    bad_eps = estimate_neighbor_gain_q90(
        state,
        time,
        coordinate_layer="coords_3d_subject_anchored",
        distance_epsilon=-1.0,
    )
    assert bad_eps["failure_reason"] == "neighbor_gain_distance_epsilon_invalid"


def test_yaml_forbids_and_common_default_off() -> None:
    import yaml

    state, time = _expanding(n=80)
    with pytest.raises(ValueError, match="resample_neighbors_at_target"):
        build_dynamical_families_export(
            config={
                "dynamical_families": {
                    "enabled": True,
                    "amplification": {"enabled": True, "resample_neighbors_at_target": True},
                }
            },
            state=state,
            time=time,
            stage=None,
            segment_id=np.zeros(time.size, dtype=np.int32),
            coordinate_layer="coords_3d_subject_anchored",
            coordinate_names=["m", "d", "e"],
        )
    with pytest.raises(ValueError, match="history_predictive_gain_level1"):
        build_dynamical_families_export(
            config={
                "dynamical_families": {
                    "enabled": True,
                    "amplification": {
                        "enabled": True,
                        "history_predictive_gain_level1": True,
                    },
                }
            },
            state=state,
            time=time,
            stage=None,
            segment_id=np.zeros(time.size, dtype=np.int32),
            coordinate_layer="coords_3d_subject_anchored",
            coordinate_names=["m", "d", "e"],
        )
    with pytest.raises(ValueError, match="cloud_volume_change_rate_level1"):
        build_dynamical_families_export(
            config={
                "dynamical_families": {
                    "enabled": True,
                    "amplification": {
                        "enabled": True,
                        "cloud_volume_change_rate_level1": True,
                    },
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
    assert cfg["dynamical_families"]["amplification"]["enabled"] is True
    assert float(cfg["dynamical_families"]["amplification"]["q"]) == pytest.approx(0.9)
    sources = common.parent / "sources" / "other"
    from core.config_loader import load_config

    for name in (
        "config_ingest_physionet_i-care_2_1_dynamical_families.yaml",
        "config_ingest_physionet_i-care_2_1_part1_0_12h_dynamical_families.yaml",
        "config_ingest_physionet_i-care_2_1_next_140_0_12h_dynamical_families.yaml",
    ):
        merged = load_config(sources / name)
        assert merged["dynamical_families"]["amplification"]["enabled"] is True
    for name in (
        "config_ingest_physionet_i-care_2_1_part1_0_12h_amplification_pilot.yaml",
        "config_ingest_physionet_i-care_2_1_next_140_0_12h_amplification_pilot.yaml",
    ):
        with (sources / name).open(encoding="utf-8") as handle:
            overlay = yaml.safe_load(handle)
        amp = (overlay.get("dynamical_families") or {}).get("amplification") or {}
        assert amp.get("enabled") is True


def test_registry_h5_and_withheld_history(tmp_path: Path) -> None:
    assert "amplification" in WRITABLE_FAMILY_IDS
    assert family_forbids("amplification", "resample_neighbors_at_target")
    assert family_forbids("amplification", "neighbor_gain_as_operator_max_gain")
    assert get_family("amplification")["schema"] == AMPLIFICATION_SCHEMA_VERSION
    assert register_entry(MEASUREMENT_ID_NEIGHBOR_GAIN)["written_by_one_step_family"] is False
    assert compatibility_entry(MEASUREMENT_ID_NEIGHBOR_GAIN)["relation"] == RELATION_NEW_MEASURE
    assert compatibility_entry(MEASUREMENT_ID_HISTORY_PREDICTIVE_GAIN)["relation"] == RELATION_NEW_MEASURE
    assert compatibility_entry(MEASUREMENT_ID_HISTORY_PREDICTIVE_GAIN)["physical_path"].startswith(
        "/dynamical_families/history/v1"
    )
    assert compatibility_entry(MEASUREMENT_ID_OPERATOR_MAX_GAIN)["relation"] == RELATION_NEW_MEASURE
    state, time = _expanding()
    export = build_dynamical_families_export(
        config=_amplification_config(),
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
        x=state.astype(np.float32),
        x_dot=np.zeros_like(state, dtype=np.float32),
        dynamical_families=export,
    )
    output = write_h5(tmp_path / "amplification.h5", "amp", payload)
    with h5py.File(output, "r") as handle:
        root = "/dynamical_families/amplification/v1"
        assert f"{root}/{MEASUREMENT_ID_NEIGHBOR_GAIN}" in handle
        assert f"{root}/{MEASUREMENT_ID_NEIGHBOR_SEPARATION}" in handle
        assert f"{root}/{MEASUREMENT_ID_NEIGHBOR_GAIN_RATE}" in handle
        assert f"{root}/{MEASUREMENT_ID_CLOUD_VOLUME}" in handle
        assert f"{root}/history_predictive_gain_level1" not in handle
        group = handle[f"{root}/{MEASUREMENT_ID_NEIGHBOR_GAIN}"]
        assert group["qualification_status"][()].decode() == QUALIFICATION_SAME_PAIR_GAIN
        assert int(group["interpretation_level"][()]) == 1
        sep = handle[f"{root}/{MEASUREMENT_ID_NEIGHBOR_SEPARATION}"]
        assert sep["qualification_status"][()].decode() == QUALIFICATION_SAME_PAIR_SEPARATION
        cloud = handle[f"{root}/{MEASUREMENT_ID_CLOUD_VOLUME}"]
        assert cloud["qualification_status"][()].decode() == QUALIFICATION_CLOUD_VOLUME
        grain = handle[f"{root}/grain"]
        assert grain["native"][()].decode() == "window"


def test_withheld_history_leaf_is_refused_at_h5_boundary() -> None:
    from mndm.schema import normalize_payload

    time = np.arange(4, dtype=float)
    state = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="history_predictive_gain_level1"):
        normalize_payload(
            MNPSPayload(
                time=time,
                x=state,
                x_dot=np.zeros_like(state),
                dynamical_families={
                    "amplification": {
                        "schema_version": AMPLIFICATION_SCHEMA_VERSION,
                        "computation_status": "computed",
                        "history_predictive_gain_level1": {
                            "computation_status": "computed"
                        },
                    }
                },
            )
        )
    with pytest.raises(ValueError, match="history_predictive_gain_level1"):
        normalize_payload(
            MNPSPayload(
                time=time,
                x=state,
                x_dot=np.zeros_like(state),
                dynamical_families={
                    "amplification": {
                        "schema_version": AMPLIFICATION_SCHEMA_VERSION,
                        "computation_status": "computed",
                        "nested": {
                            "history_predictive_gain_level1": {
                                "computation_status": "computed"
                            }
                        },
                    }
                },
            )
        )


def test_missing_coordinate_layer_keeps_all_leaves_and_refuses_aliases() -> None:
    export = _build_dynamical_families_export_for_layers(
        config={
            "dynamical_families": {
                "enabled": True,
                "coordinate_layer": "subject_anchored",
                "amplification": {"enabled": True},
            }
        },
        x_subject_anchored=None,
        x_cohort_anchored=None,
        time=np.arange(8, dtype=float) * 0.25,
        stage=None,
        segment_id=None,
    )
    amp = export["amplification"]
    assert amp["computation_status"] == "not_testable"
    assert MEASUREMENT_ID_NEIGHBOR_GAIN in amp
    assert MEASUREMENT_ID_NEIGHBOR_SEPARATION in amp
    assert MEASUREMENT_ID_NEIGHBOR_GAIN_RATE in amp
    assert MEASUREMENT_ID_CLOUD_VOLUME in amp
    with pytest.raises(ValueError, match="neighbor_separation_rate_level1"):
        _build_dynamical_families_export_for_layers(
            config={
                "dynamical_families": {
                    "enabled": True,
                    "coordinate_layer": "subject_anchored",
                    "amplification": {"enabled": True, "neighbor_separation_rate_level1": True},
                }
            },
            x_subject_anchored=None,
            x_cohort_anchored=None,
            time=np.arange(8, dtype=float) * 0.25,
            stage=None,
            segment_id=None,
        )
