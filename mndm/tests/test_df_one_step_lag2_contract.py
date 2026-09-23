"""Direct lag-2 affine map and lag-2 generator proxies. Not Φ₁² and not Itô."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import yaml
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))

from mndm.dynamical_families import family_forbids
from mndm.dynamical_families.measurement_register import (
    ALLOWED_ONE_STEP_DECLARED_LAGS,
    CLAIM_DISCRETE_ONE_STEP,
    CLAIM_FINITE_HORIZON_PROPAGATION,
    CLAIM_GENERATOR,
    DEFAULT_ONE_STEP_DECLARED_LAGS,
    MEASUREMENT_ID_AFFINE_MAP,
    MEASUREMENT_ID_AFFINE_MEAN_RATE,
    MEASUREMENT_ID_DIVERGENCE,
    MEASUREMENT_ID_GENERATOR_ROTATION,
    MEASUREMENT_ID_INNOVATION_COVARIANCE,
    MEASUREMENT_ID_ITERATED_ONE_STEP_MAP,
    MEASUREMENT_ID_ITO_DRIFT,
    MEASUREMENT_ID_NUMERICAL_ABSCISSA,
    MEASUREMENT_ID_OPERATOR_MAX_GAIN,
    MEASUREMENT_ID_OPERATOR_ROTATION_RATE,
    MEASUREMENT_ID_OPERATOR_VOLUME_GAIN,
    MEASUREMENT_ID_SPECTRAL_ABSCISSA,
    ONE_STEP_LAG2_INNOVATION,
    ONE_STEP_LAG2_MAP,
    ONE_STEP_LAG2_MEAN_RATE,
    ONE_STEP_LAG2_SPECTRAL,
    PHYSICAL_ONE_STEP_LAG2_ROOT,
    PHYSICAL_ONE_STEP_ROOT,
    QUALIFICATION_GENERATOR_PROXY_NOT_ITO,
    QUALIFICATION_ONE_STEP_FUNCTIONAL,
    QUALIFICATION_ONE_STEP_IDENTIFIED,
    QUALIFICATION_ONE_STEP_NOT_IDENTIFIED,
    QUALIFICATION_HORIZON_PROPAGATION_IDENTIFIED,
    QUALIFICATION_HORIZON_PROPAGATION_NOT_IDENTIFIED,
    RELATION_NEW_MEASURE,
    RELATION_SUPERSEDED,
    VARIANT_DECLARED_LAG_2,
    VARIANT_DECLARED_LAG_STEPS_1,
    VARIANT_DECLARED_LAG_STEPS_2,
    compatibility_entry,
    register_entry,
)
from mndm.dynamical_families.one_step_operator import (
    REASON_EMBARGO,
    REASON_INSUFFICIENT_LOCAL,
    REASON_UPSTREAM,
    _recording_affine_at_lag,
    estimate_affine_one_step_family,
)
from mndm.pipeline.dynamical_families_export import build_dynamical_families_export
from mndm.schema import MNPSPayload


def _ou_state(n: int = 120) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    x = np.cumsum(0.05 * rng.normal(size=(n, 3)), axis=0).astype(np.float32)
    time = np.arange(n, dtype=float) * 0.25
    return x, time


def _linear_map_state(
    n: int = 400, dt: float = 0.25, seed: int = 11
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    generator = np.array(
        [[-0.45, 0.20, 0.00], [-0.10, -0.55, 0.08], [0.00, -0.15, -0.35]],
        dtype=float,
    )
    phi = expm(generator * dt)
    x = np.zeros((n, 3), dtype=float)
    x[0] = rng.normal(size=3)
    noise = 0.04 * rng.normal(size=(n - 1, 3))
    for t in range(n - 1):
        x[t + 1] = phi @ x[t] + noise[t]
    time = np.arange(n, dtype=float) * dt
    return x.astype(np.float32), time, phi


def _finite_median(leaf: dict, series_name: str) -> float:
    series = np.asarray(leaf["series"][series_name], dtype=float)
    finite = np.isfinite(series)
    assert np.any(finite)
    return float(np.median(series[finite]))


def _finite_phi(leaf: dict) -> np.ndarray:
    series = np.asarray(leaf["series"]["phi_hat"], dtype=float)
    finite = np.all(np.isfinite(series.reshape(series.shape[0], -1)), axis=1)
    assert np.any(finite)
    return series[np.flatnonzero(finite)[0]]


def _switching_linear_state(
    n: int = 500, dt: float = 0.25, seed: int = 21
) -> tuple[np.ndarray, np.ndarray]:
    """Period-2 linear maps. Direct lag-2 is Φ_b Φ_a; composed lag-1 is not."""
    rng = np.random.default_rng(seed)
    phi_a = 0.40 * np.eye(3)
    phi_b = 0.95 * np.eye(3)
    x = np.zeros((n, 3), dtype=float)
    x[0] = rng.normal(size=3)
    noise = 0.015 * rng.normal(size=(n - 1, 3))
    for t in range(n - 1):
        phi = phi_a if t % 2 == 0 else phi_b
        x[t + 1] = phi @ x[t] + noise[t]
    time = np.arange(n, dtype=float) * dt
    return x.astype(np.float32), time


def test_lag2_register_paths_are_frozen_and_writable() -> None:
    assert ALLOWED_ONE_STEP_DECLARED_LAGS == (1, 2)
    assert DEFAULT_ONE_STEP_DECLARED_LAGS == (1,)
    lag1 = register_entry(MEASUREMENT_ID_AFFINE_MAP)
    assert lag1["declared_lag_steps"] == 1
    assert lag1["variant_id"] == VARIANT_DECLARED_LAG_STEPS_1
    assert lag1["physical_path"] == f"{PHYSICAL_ONE_STEP_ROOT}/{MEASUREMENT_ID_AFFINE_MAP}"
    assert lag1["written_by_one_step_family"] is True

    lag2 = register_entry(ONE_STEP_LAG2_MAP)
    assert lag2["measurement_id"] == MEASUREMENT_ID_AFFINE_MAP
    assert lag2["declared_lag_steps"] == 2
    assert lag2["variant_id"] == VARIANT_DECLARED_LAG_STEPS_2
    assert lag2["claim_class"] == CLAIM_DISCRETE_ONE_STEP
    assert lag2["interpretation_level"] == 2
    assert lag2["relation"] == RELATION_NEW_MEASURE
    assert lag2["written_by_one_step_family"] is True
    assert lag2["not_composed_one_step"] is True
    assert lag2["min_embargo_steps"] == 2
    assert lag2["embargo_semantics"] == "index_steps"
    assert lag2["raw_window_support_independence"] == "not_established"
    assert lag2["physical_path"] == (
        f"{PHYSICAL_ONE_STEP_LAG2_ROOT}/{MEASUREMENT_ID_AFFINE_MAP}"
    )
    assert lag2["physical_path"] != lag1["physical_path"]
    assert VARIANT_DECLARED_LAG_2 in lag2["physical_path"]

    rate = register_entry(ONE_STEP_LAG2_MEAN_RATE)
    innov = register_entry(ONE_STEP_LAG2_INNOVATION)
    assert rate["measurement_id"] == MEASUREMENT_ID_AFFINE_MEAN_RATE
    assert innov["measurement_id"] == MEASUREMENT_ID_INNOVATION_COVARIANCE
    assert rate["relation"] == RELATION_NEW_MEASURE
    assert innov["relation"] == RELATION_NEW_MEASURE
    assert innov["written_by_diffusion_family"] is False
    assert rate["min_embargo_steps"] == 2
    assert innov["min_embargo_steps"] == 2
    assert innov["embargo_semantics"] == "index_steps"
    assert innov["raw_window_support_independence"] == "not_established"


def test_phi_one_squared_is_not_the_lag2_identity() -> None:
    composed = register_entry(MEASUREMENT_ID_ITERATED_ONE_STEP_MAP)
    assert composed["interpretation_level"] == 4
    assert composed["claim_class"] == CLAIM_FINITE_HORIZON_PROPAGATION
    assert composed["physical_path"] == (
        f"{PHYSICAL_ONE_STEP_ROOT}/{MEASUREMENT_ID_ITERATED_ONE_STEP_MAP}"
    )
    assert composed["relation"] == RELATION_NEW_MEASURE
    assert composed["not_direct_lag2_map"] is True
    assert composed["written_by_one_step_family"] is True
    assert VARIANT_DECLARED_LAG_2 not in str(composed["physical_path"])

    lag2_l3 = register_entry(ONE_STEP_LAG2_SPECTRAL)
    assert lag2_l3["relation"] == RELATION_NEW_MEASURE
    assert lag2_l3["written_by_one_step_family"] is True
    assert lag2_l3["claim_class"] == CLAIM_GENERATOR
    assert lag2_l3["interpretation_level"] == 3
    assert lag2_l3["not_composed_one_step"] is True
    assert lag2_l3["physical_path"] == (
        f"{PHYSICAL_ONE_STEP_LAG2_ROOT}/{MEASUREMENT_ID_SPECTRAL_ABSCISSA}"
    )
    assert lag2_l3["default_qualification_status"] == QUALIFICATION_GENERATOR_PROXY_NOT_ITO
    assert compatibility_entry(ONE_STEP_LAG2_SPECTRAL)["relation"] == RELATION_NEW_MEASURE

    refused_name = compatibility_entry("affine_two_step_map_level2")
    assert refused_name["relation"] == RELATION_SUPERSEDED
    assert compatibility_entry(ONE_STEP_LAG2_MAP)["relation"] == RELATION_NEW_MEASURE
    assert compatibility_entry(MEASUREMENT_ID_ITERATED_ONE_STEP_MAP)["physical_path"] == (
        f"{PHYSICAL_ONE_STEP_ROOT}/{MEASUREMENT_ID_ITERATED_ONE_STEP_MAP}"
    )
    assert family_forbids("one_step", "phi_one_composed_as_lag2")
    assert family_forbids("one_step", "one_step_iteration_as_level2")
    assert family_forbids("one_step", "affine_two_step")
    assert family_forbids("one_step", "two_step")
    assert family_forbids("one_step", "affine_two_step_map_level2")
    assert family_forbids("one_step", "peak_gain_from_phi_powers")
    assert family_forbids("one_step", "operator_gain_as_spectral_abscissa")
    assert family_forbids("one_step", "epsilon_rescued_volume_gain")


def test_declared_lags_other_than_one_are_refused() -> None:
    state, time = _ou_state()
    segment = np.zeros(time.size, dtype=np.int32)
    layer = "coords_3d_subject_anchored"
    names = ["m", "d", "e"]
    with pytest.raises(ValueError, match="are not implemented"):
        build_dynamical_families_export(
            config={
                "dynamical_families": {
                    "enabled": True,
                    "one_step": {"enabled": True, "declared_lags": [3]},
                }
            },
            state=state,
            time=time,
            stage=None,
            segment_id=segment,
            coordinate_layer=layer,
            coordinate_names=names,
        )
    with pytest.raises(ValueError, match="phi_one_composed_as_lag2"):
        build_dynamical_families_export(
            config={
                "dynamical_families": {
                    "enabled": True,
                    "one_step": {"enabled": True, "phi_one_composed_as_lag2": True},
                }
            },
            state=state,
            time=time,
            stage=None,
            segment_id=segment,
            coordinate_layer=layer,
            coordinate_names=names,
        )
    with pytest.raises(ValueError, match="affine_two_step_map_level2"):
        build_dynamical_families_export(
            config={
                "dynamical_families": {
                    "enabled": True,
                    "one_step": {"enabled": True, "affine_two_step_map_level2": True},
                }
            },
            state=state,
            time=time,
            stage=None,
            segment_id=segment,
            coordinate_layer=layer,
            coordinate_names=names,
        )
    with pytest.raises(ValueError, match="declared_lags must be a list"):
        build_dynamical_families_export(
            config={
                "dynamical_families": {
                    "enabled": True,
                    "one_step": {"enabled": True, "declared_lags": [1.5]},
                }
            },
            state=state,
            time=time,
            stage=None,
            segment_id=segment,
            coordinate_layer=layer,
            coordinate_names=names,
        )


def test_estimator_still_writes_only_lag1_leaves() -> None:
    state, time = _ou_state(n=200)
    result = estimate_affine_one_step_family(
        state,
        time,
        coordinate_layer="coords_3d_subject_anchored",
        neighborhood_k=16,
        min_samples=30,
        min_neighborhood_samples=10,
        one_step_rel_mse_threshold=0.9,
    )
    assert VARIANT_DECLARED_LAG_2 not in result
    assert MEASUREMENT_ID_AFFINE_MAP in result
    assert MEASUREMENT_ID_ITERATED_ONE_STEP_MAP in result
    assert ONE_STEP_LAG2_MAP not in result

    export = build_dynamical_families_export(
        config={
            "dynamical_families": {
                "enabled": True,
                "one_step": {"enabled": True, "declared_lags": [1]},
            }
        },
        state=state,
        time=time,
        stage=None,
        segment_id=np.zeros(time.size, dtype=np.int32),
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
    )
    family = export["one_step"]
    assert VARIANT_DECLARED_LAG_2 not in family
    assert MEASUREMENT_ID_AFFINE_MAP in family
    assert MEASUREMENT_ID_ITERATED_ONE_STEP_MAP in family


def test_common_yaml_declared_lags_is_one() -> None:
    common = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "config_ingest_common_dynamical_families.yaml"
    )
    with common.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    one_step = cfg["dynamical_families"]["one_step"]
    assert one_step["enabled"] is True
    assert one_step["declared_lags"] == [1]
    assert one_step["one_step_rel_mse_threshold"] == 0.9
    for name in (
        "config_ingest_physionet_i-care_2_1_dynamical_families.yaml",
        "config_ingest_physionet_i-care_2_1_next_140_0_12h_dynamical_families.yaml",
    ):
        overlay = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "sources"
            / "other"
            / name
        )
        with overlay.open(encoding="utf-8") as handle:
            overlay_cfg = yaml.safe_load(handle)
        overlay_one_step = (overlay_cfg.get("dynamical_families") or {}).get("one_step") or {}
        assert overlay_one_step.get("enabled") is True
        assert overlay_one_step.get("declared_lags") == [1, 2]
        assert overlay_one_step.get("one_step_rel_mse_threshold") in (None, 0.9)


def test_direct_lag2_recovers_phi_squared_not_phi_one() -> None:
    state, time, phi_true = _linear_map_state()
    result = estimate_affine_one_step_family(
        state,
        time,
        coordinate_layer="coords_3d_subject_anchored",
        neighborhood_k=24,
        min_samples=40,
        min_neighborhood_samples=12,
        ridge_alpha=1e-6,
        declared_lags=(1, 2),
    )
    assert VARIANT_DECLARED_LAG_2 in result
    lag1_map = result[MEASUREMENT_ID_AFFINE_MAP]
    lag2_group = result[VARIANT_DECLARED_LAG_2]
    lag2_map = lag2_group[MEASUREMENT_ID_AFFINE_MAP]
    assert lag1_map["qualification_status"] == QUALIFICATION_ONE_STEP_IDENTIFIED
    assert lag2_map["qualification_status"] == QUALIFICATION_ONE_STEP_IDENTIFIED
    assert int(lag2_map["summary"]["transition_support_lag"]) == 2
    assert int(lag2_map["summary"]["declared_lag_steps"]) == 2
    assert lag2_map["summary"]["not_composed_one_step"] is True
    assert lag2_map["variant_id"] == VARIANT_DECLARED_LAG_STEPS_2
    _L3_SERIES = (
        (MEASUREMENT_ID_SPECTRAL_ABSCISSA, "spectral_abscissa"),
        (MEASUREMENT_ID_NUMERICAL_ABSCISSA, "numerical_abscissa"),
        (MEASUREMENT_ID_DIVERGENCE, "divergence"),
        (MEASUREMENT_ID_GENERATOR_ROTATION, "generator_rotation_norm"),
    )
    generator = np.array(
        [[-0.45, 0.20, 0.00], [-0.10, -0.55, 0.08], [0.00, -0.15, -0.35]],
        dtype=float,
    )
    true_spec = float(np.max(np.real(np.linalg.eigvals(generator))))
    true_num = float(np.linalg.eigvalsh(0.5 * (generator + generator.T))[-1])
    true_div = float(np.trace(generator))
    true_rot = float(np.linalg.norm(0.5 * (generator - generator.T)))
    truth = {
        "spectral_abscissa": true_spec,
        "numerical_abscissa": true_num,
        "divergence": true_div,
        "generator_rotation_norm": true_rot,
    }
    for measurement_id, series_name in _L3_SERIES:
        lag1_leaf = result[measurement_id]
        lag2_leaf = lag2_group[measurement_id]
        assert lag2_leaf["computation_status"] == "computed"
        assert lag2_leaf["qualification_status"] == QUALIFICATION_GENERATOR_PROXY_NOT_ITO
        assert int(lag2_leaf["summary"]["declared_lag_steps"]) == 2
        assert lag2_leaf["variant_id"] == VARIANT_DECLARED_LAG_STEPS_2
        assert lag2_leaf["summary"]["not_ito_drift"] is True
        assert lag2_leaf["summary"]["generator_conversion"] == "matrix_log_over_nominal_dt"
        val1 = _finite_median(lag1_leaf, series_name)
        val2 = _finite_median(lag2_leaf, series_name)
        assert abs(val1 - val2) < 0.15
        assert abs(val2 - truth[series_name]) < 0.20
    _L2_FUNCTIONALS = (
        MEASUREMENT_ID_OPERATOR_MAX_GAIN,
        MEASUREMENT_ID_OPERATOR_VOLUME_GAIN,
        MEASUREMENT_ID_OPERATOR_ROTATION_RATE,
    )
    for measurement_id in _L2_FUNCTIONALS:
        lag1_leaf = result[measurement_id]
        lag2_leaf = lag2_group[measurement_id]
        assert lag2_leaf["computation_status"] == "computed"
        assert lag2_leaf["qualification_status"] == QUALIFICATION_ONE_STEP_FUNCTIONAL
        assert int(lag2_leaf["interpretation_level"]) == 2
        assert int(lag2_leaf["summary"]["declared_lag_steps"]) == 2
        assert lag2_leaf["variant_id"] == VARIANT_DECLARED_LAG_STEPS_2
        assert lag1_leaf["qualification_status"] == QUALIFICATION_ONE_STEP_FUNCTIONAL
        assert int(lag1_leaf["summary"]["declared_lag_steps"]) == 1
    dt1 = float(lag1_map["summary"]["nominal_dt_sec"])
    dt2 = float(lag2_map["summary"]["nominal_dt_sec"])
    assert abs(dt2 - 2.0 * dt1) < 0.05
    phi1 = _finite_phi(lag1_map)
    phi2 = _finite_phi(lag2_map)
    assert not np.allclose(phi1, phi2, atol=0.05)
    assert np.linalg.norm(phi2 - (phi_true @ phi_true)) < 0.25
    assert np.linalg.norm(phi2 - (phi1 @ phi1)) < 0.25
    lag1_support = str(lag1_map["summary"]["transition_support_id"])
    lag2_support = str(lag2_map["summary"]["transition_support_id"])
    assert lag1_support != lag2_support
    export = build_dynamical_families_export(
        config={
            "dynamical_families": {
                "enabled": True,
                "one_step": {"enabled": True, "declared_lags": [1, 2], "ridge_alpha": 1e-6},
            }
        },
        state=state,
        time=time,
        stage=None,
        segment_id=np.zeros(time.size, dtype=np.int32),
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
    )
    family = export["one_step"]
    assert VARIANT_DECLARED_LAG_2 in family
    for measurement_id, _series_name in (
        (MEASUREMENT_ID_SPECTRAL_ABSCISSA, "spectral_abscissa"),
        (MEASUREMENT_ID_NUMERICAL_ABSCISSA, "numerical_abscissa"),
        (MEASUREMENT_ID_DIVERGENCE, "divergence"),
        (MEASUREMENT_ID_GENERATOR_ROTATION, "generator_rotation_norm"),
        (MEASUREMENT_ID_OPERATOR_MAX_GAIN, "operator_max_gain_rate"),
        (MEASUREMENT_ID_OPERATOR_VOLUME_GAIN, "operator_volume_gain_rate"),
        (MEASUREMENT_ID_OPERATOR_ROTATION_RATE, "operator_rotation_rate"),
    ):
        assert measurement_id in family
        assert measurement_id in family[VARIANT_DECLARED_LAG_2]
        nested = family[VARIANT_DECLARED_LAG_2][measurement_id]
        root = family[measurement_id]
        assert int(nested["summary"]["declared_lag_steps"]) == 2
        assert int(root["summary"]["declared_lag_steps"]) == 1
        assert MEASUREMENT_ID_ITO_DRIFT not in family
        assert MEASUREMENT_ID_ITO_DRIFT not in family[VARIANT_DECLARED_LAG_2]


def test_iid_lag2_is_not_identified() -> None:
    rng = np.random.default_rng(12)
    state = rng.normal(size=(240, 3)).astype(np.float32)
    time = np.arange(240, dtype=float) * 0.25
    result = estimate_affine_one_step_family(
        state,
        time,
        coordinate_layer="coords_3d_subject_anchored",
        min_samples=30,
        ridge_alpha=1e-4,
        declared_lags=(2,),
    )
    lag2_map = result[VARIANT_DECLARED_LAG_2][MEASUREMENT_ID_AFFINE_MAP]
    assert lag2_map["qualification_status"] == QUALIFICATION_ONE_STEP_NOT_IDENTIFIED
    assert MEASUREMENT_ID_SPECTRAL_ABSCISSA not in result
    assert MEASUREMENT_ID_ITO_DRIFT not in result
    assert MEASUREMENT_ID_ITO_DRIFT not in result[VARIANT_DECLARED_LAG_2]
    for measurement_id in (
        MEASUREMENT_ID_SPECTRAL_ABSCISSA,
        MEASUREMENT_ID_NUMERICAL_ABSCISSA,
        MEASUREMENT_ID_DIVERGENCE,
        MEASUREMENT_ID_GENERATOR_ROTATION,
    ):
        leaf = result[VARIANT_DECLARED_LAG_2][measurement_id]
        assert leaf["computation_status"] == "not_testable"
        assert leaf["failure_reason"] == REASON_UPSTREAM
        assert leaf["qualification_status"] == QUALIFICATION_GENERATOR_PROXY_NOT_ITO
        assert int(leaf["summary"]["declared_lag_steps"]) == 2
    for measurement_id in (
        MEASUREMENT_ID_OPERATOR_MAX_GAIN,
        MEASUREMENT_ID_OPERATOR_VOLUME_GAIN,
        MEASUREMENT_ID_OPERATOR_ROTATION_RATE,
    ):
        leaf = result[VARIANT_DECLARED_LAG_2][measurement_id]
        assert leaf["computation_status"] == "not_testable"
        assert leaf["failure_reason"] == REASON_UPSTREAM
        assert leaf["qualification_status"] == QUALIFICATION_ONE_STEP_FUNCTIONAL
        assert measurement_id not in result
        assert int(leaf["summary"]["declared_lag_steps"]) == 2


def test_two_short_segments_are_insufficient_local_support() -> None:
    state, time = _ou_state(n=32)
    segment = np.concatenate(
        [np.zeros(16, dtype=np.int32), np.ones(16, dtype=np.int32)]
    )
    result = estimate_affine_one_step_family(
        state,
        time,
        segment_id=segment,
        coordinate_layer="coords_3d_subject_anchored",
        min_samples=20,
        min_neighborhood_samples=10,
        declared_lags=(2,),
    )
    lag2_map = result[VARIANT_DECLARED_LAG_2][MEASUREMENT_ID_AFFINE_MAP]
    assert lag2_map["computation_status"] == "insufficient_support"
    assert lag2_map["failure_reason"] == REASON_INSUFFICIENT_LOCAL


def test_embargo_shorter_than_lag2_is_invalid() -> None:
    state, time = _ou_state(n=120)
    result = estimate_affine_one_step_family(
        state,
        time,
        coordinate_layer="coords_3d_subject_anchored",
        embargo_steps=1,
        declared_lags=(2,),
    )
    lag2_map = result[VARIANT_DECLARED_LAG_2][MEASUREMENT_ID_AFFINE_MAP]
    assert lag2_map["computation_status"] == "invalid"
    assert lag2_map["failure_reason"] == REASON_EMBARGO
    assert int(lag2_map["summary"]["embargo_steps"]) == 1
    assert lag2_map["summary"]["embargo_semantics"] == "index_steps"
    assert lag2_map["summary"]["raw_window_support_independence"] == "not_established"
    assert int(lag2_map["summary"]["min_embargo_steps"]) == 2
    assert int(lag2_map["summary"]["declared_lag_steps"]) == 2
    lag2_spectral = result[VARIANT_DECLARED_LAG_2][MEASUREMENT_ID_SPECTRAL_ABSCISSA]
    assert lag2_spectral["computation_status"] == "invalid"
    assert lag2_spectral["failure_reason"] == REASON_EMBARGO
    assert lag2_spectral["qualification_status"] == QUALIFICATION_GENERATOR_PROXY_NOT_ITO
    lag2_gain = result[VARIANT_DECLARED_LAG_2][MEASUREMENT_ID_OPERATOR_MAX_GAIN]
    assert lag2_gain["computation_status"] == "invalid"
    assert lag2_gain["failure_reason"] == REASON_EMBARGO
    assert lag2_gain["qualification_status"] == QUALIFICATION_ONE_STEP_FUNCTIONAL


def test_private_lag2_fitter_writes_level3_when_identified() -> None:
    state, time, _ = _linear_map_state()
    result = _recording_affine_at_lag(
        state,
        time,
        lag=2,
        write_level3=True,
        coordinate_layer="coords_3d_subject_anchored",
        neighborhood_k=24,
        min_samples=40,
        min_neighborhood_samples=12,
        ridge_alpha=1e-6,
    )
    spectral = result[MEASUREMENT_ID_SPECTRAL_ABSCISSA]
    assert spectral["computation_status"] == "computed"
    assert spectral["qualification_status"] == QUALIFICATION_GENERATOR_PROXY_NOT_ITO
    assert int(spectral["summary"]["declared_lag_steps"]) == 2
    assert spectral["variant_id"] == VARIANT_DECLARED_LAG_STEPS_2
    assert spectral["summary"]["not_ito_drift"] is True
    assert int(result[MEASUREMENT_ID_AFFINE_MAP]["summary"]["transition_support_lag"]) == 2
    assert result[MEASUREMENT_ID_AFFINE_MAP]["variant_id"] == VARIANT_DECLARED_LAG_STEPS_2
    assert int(result[MEASUREMENT_ID_AFFINE_MAP]["summary"]["declared_lag_steps"]) == 2


def test_direct_lag2_is_not_composed_one_step() -> None:
    state, time = _switching_linear_state()
    result = estimate_affine_one_step_family(
        state,
        time,
        coordinate_layer="coords_3d_subject_anchored",
        neighborhood_k=24,
        min_samples=40,
        min_neighborhood_samples=12,
        ridge_alpha=1e-6,
        declared_lags=(1, 2),
    )
    lag1_map = result[MEASUREMENT_ID_AFFINE_MAP]
    lag2_map = result[VARIANT_DECLARED_LAG_2][MEASUREMENT_ID_AFFINE_MAP]
    assert lag1_map["qualification_status"] == QUALIFICATION_ONE_STEP_IDENTIFIED
    assert lag2_map["qualification_status"] == QUALIFICATION_ONE_STEP_IDENTIFIED
    phi1 = _finite_phi(lag1_map)
    phi2 = _finite_phi(lag2_map)
    composed = phi1 @ phi1
    assert int(lag2_map["summary"]["transition_support_lag"]) == 2
    assert np.linalg.norm(phi2 - composed) > 0.08


def test_lag2_h5_nested_path_has_level3(tmp_path: Path) -> None:
    state, time, _ = _linear_map_state()
    export = build_dynamical_families_export(
        config={
            "dynamical_families": {
                "enabled": True,
                "one_step": {
                    "enabled": True,
                    "declared_lags": [1, 2],
                    "ridge_alpha": 1e-6,
                    "min_samples": 40,
                    "neighborhood": {"k": 24},
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
    from core.io.h5_writer import write_h5
    import h5py

    payload = MNPSPayload(
        time=time,
        x=state.astype(np.float32),
        x_dot=np.zeros_like(state, dtype=np.float32),
        dynamical_families=export,
    )
    output = write_h5(tmp_path / "one_step_lag2.h5", "one_step_lag2", payload)
    with h5py.File(output, "r") as handle:
        root = "/dynamical_families/one_step/v1"
        lag2 = f"{root}/{VARIANT_DECLARED_LAG_2}"
        assert f"{lag2}/{MEASUREMENT_ID_AFFINE_MAP}" in handle
        assert f"{lag2}/{MEASUREMENT_ID_AFFINE_MEAN_RATE}" in handle
        assert f"{lag2}/{MEASUREMENT_ID_INNOVATION_COVARIANCE}" in handle
        assert f"{lag2}/{MEASUREMENT_ID_SPECTRAL_ABSCISSA}" in handle
        assert f"{lag2}/{MEASUREMENT_ID_NUMERICAL_ABSCISSA}" in handle
        assert f"{lag2}/{MEASUREMENT_ID_DIVERGENCE}" in handle
        assert f"{lag2}/{MEASUREMENT_ID_GENERATOR_ROTATION}" in handle
        assert f"{lag2}/{MEASUREMENT_ID_OPERATOR_MAX_GAIN}" in handle
        assert f"{lag2}/{MEASUREMENT_ID_OPERATOR_VOLUME_GAIN}" in handle
        assert f"{lag2}/{MEASUREMENT_ID_OPERATOR_ROTATION_RATE}" in handle
        assert f"{root}/{MEASUREMENT_ID_SPECTRAL_ABSCISSA}" in handle
        assert f"{root}/{MEASUREMENT_ID_OPERATOR_MAX_GAIN}" in handle
        assert f"{root}/{MEASUREMENT_ID_ITO_DRIFT}" not in handle
        assert f"{lag2}/{MEASUREMENT_ID_ITO_DRIFT}" not in handle
        for measurement_id in (
            MEASUREMENT_ID_SPECTRAL_ABSCISSA,
            MEASUREMENT_ID_NUMERICAL_ABSCISSA,
            MEASUREMENT_ID_DIVERGENCE,
            MEASUREMENT_ID_GENERATOR_ROTATION,
        ):
            nested_l3 = handle[f"{lag2}/{measurement_id}"]
            root_l3 = handle[f"{root}/{measurement_id}"]
            assert nested_l3["variant_id"][()].decode() == VARIANT_DECLARED_LAG_STEPS_2
            assert int(nested_l3["summary"]["declared_lag_steps"][()]) == 2
            assert nested_l3["qualification_status"][()].decode() == QUALIFICATION_GENERATOR_PROXY_NOT_ITO
            assert int(root_l3["summary"]["declared_lag_steps"][()]) == 1
            assert root_l3["variant_id"][()].decode() != VARIANT_DECLARED_LAG_STEPS_2
        for measurement_id in (
            MEASUREMENT_ID_OPERATOR_MAX_GAIN,
            MEASUREMENT_ID_OPERATOR_VOLUME_GAIN,
            MEASUREMENT_ID_OPERATOR_ROTATION_RATE,
        ):
            nested_fn = handle[f"{lag2}/{measurement_id}"]
            root_fn = handle[f"{root}/{measurement_id}"]
            assert nested_fn["variant_id"][()].decode() == VARIANT_DECLARED_LAG_STEPS_2
            assert int(nested_fn["summary"]["declared_lag_steps"][()]) == 2
            assert nested_fn["qualification_status"][()].decode() == QUALIFICATION_ONE_STEP_FUNCTIONAL
            assert int(nested_fn["interpretation_level"][()]) == 2
            assert int(root_fn["summary"]["declared_lag_steps"][()]) == 1
        assert f"{root}/{MEASUREMENT_ID_ITERATED_ONE_STEP_MAP}" in handle
        assert f"{lag2}/{MEASUREMENT_ID_ITERATED_ONE_STEP_MAP}" not in handle
        group = handle[f"{lag2}/{MEASUREMENT_ID_AFFINE_MAP}"]
        assert group["measurement_id"][()].decode() == MEASUREMENT_ID_AFFINE_MAP
        assert group["variant_id"][()].decode() == VARIANT_DECLARED_LAG_STEPS_2
        assert int(group["summary"]["transition_support_lag"][()]) == 2
        assert int(group["summary"]["not_composed_one_step"][()]) == 1
        horizon = handle[f"{root}/{MEASUREMENT_ID_ITERATED_ONE_STEP_MAP}"]
        assert int(horizon["interpretation_level"][()]) == 4
        assert int(horizon["summary"]["not_direct_lag2_map"][()]) == 1


def _finite_composed(leaf: dict) -> np.ndarray:
    series = np.asarray(leaf["series"]["phi_composed"], dtype=float)
    finite = np.all(np.isfinite(series.reshape(series.shape[0], -1)), axis=1)
    assert np.any(finite)
    return series[np.flatnonzero(finite)[0]]


def test_linear_horizon_map_is_identified_and_not_lag2_path() -> None:
    state, time, phi_true = _linear_map_state()
    result = estimate_affine_one_step_family(
        state,
        time,
        coordinate_layer="coords_3d_subject_anchored",
        neighborhood_k=24,
        min_samples=40,
        min_neighborhood_samples=12,
        ridge_alpha=1e-6,
        declared_lags=(1, 2),
    )
    horizon = result[MEASUREMENT_ID_ITERATED_ONE_STEP_MAP]
    lag2_map = result[VARIANT_DECLARED_LAG_2][MEASUREMENT_ID_AFFINE_MAP]
    assert horizon["qualification_status"] == QUALIFICATION_HORIZON_PROPAGATION_IDENTIFIED
    assert int(horizon["interpretation_level"]) == 4
    assert horizon["claim_class"] == CLAIM_FINITE_HORIZON_PROPAGATION
    assert MEASUREMENT_ID_ITERATED_ONE_STEP_MAP not in result[VARIANT_DECLARED_LAG_2]
    phi_h = _finite_composed(horizon)
    assert np.linalg.norm(phi_h - (phi_true @ phi_true)) < 0.25
    assert float(horizon["summary"]["composed_phi_lag2_frobenius"]) < 0.25
    assert lag2_map["summary"]["not_composed_one_step"] is True


def test_horizon_map_is_not_direct_lag2_on_switching() -> None:
    state, time = _switching_linear_state()
    result = estimate_affine_one_step_family(
        state,
        time,
        coordinate_layer="coords_3d_subject_anchored",
        neighborhood_k=24,
        min_samples=40,
        min_neighborhood_samples=12,
        ridge_alpha=1e-6,
        declared_lags=(1, 2),
    )
    horizon = result[MEASUREMENT_ID_ITERATED_ONE_STEP_MAP]
    lag2_map = result[VARIANT_DECLARED_LAG_2][MEASUREMENT_ID_AFFINE_MAP]
    assert lag2_map["qualification_status"] == QUALIFICATION_ONE_STEP_IDENTIFIED
    assert horizon["summary"]["not_direct_lag2_map"] is True
    if horizon["qualification_status"] == QUALIFICATION_HORIZON_PROPAGATION_IDENTIFIED:
        phi_h = _finite_composed(horizon)
        phi2 = _finite_phi(lag2_map)
        assert np.linalg.norm(phi_h - phi2) > 0.08
    assert float(horizon["summary"]["composed_phi_lag2_frobenius"]) > 0.08


def test_lag2_only_does_not_write_horizon_map() -> None:
    state, time, _ = _linear_map_state()
    result = estimate_affine_one_step_family(
        state,
        time,
        coordinate_layer="coords_3d_subject_anchored",
        neighborhood_k=24,
        min_samples=40,
        min_neighborhood_samples=12,
        ridge_alpha=1e-6,
        declared_lags=(2,),
    )
    assert MEASUREMENT_ID_ITERATED_ONE_STEP_MAP not in result
    assert MEASUREMENT_ID_ITERATED_ONE_STEP_MAP not in result[VARIANT_DECLARED_LAG_2]
