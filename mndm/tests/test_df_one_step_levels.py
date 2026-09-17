"""Generic qualified one-step operator and generator proxies (not I-CARE)."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.linalg import expm, logm, polar

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))

from mndm.dynamical_families import (
    AFFINE_ONE_STEP_SCHEMA_VERSION,
    WRITABLE_FAMILY_IDS,
    family_forbids,
    get_family,
)
from mndm.dynamical_families.measurement_register import (
    MEASUREMENT_ID_AFFINE_MAP,
    MEASUREMENT_ID_AFFINE_MEAN_RATE,
    MEASUREMENT_ID_DIVERGENCE,
    MEASUREMENT_ID_GENERATOR_ROTATION,
    MEASUREMENT_ID_GENERATOR_SYM_ANISO,
    MEASUREMENT_ID_INNOVATION_COVARIANCE,
    MEASUREMENT_ID_ITO_DRIFT,
    MEASUREMENT_ID_ITERATED_ONE_STEP_MAP,
    MEASUREMENT_ID_NUMERICAL_ABSCISSA,
    MEASUREMENT_ID_OPERATOR_GAIN_ANISO,
    MEASUREMENT_ID_OPERATOR_MAX_GAIN,
    MEASUREMENT_ID_OPERATOR_ROTATION_RATE,
    MEASUREMENT_ID_OPERATOR_VOLUME_GAIN,
    MEASUREMENT_ID_REACTIVITY_GAP,
    MEASUREMENT_ID_SPECTRAL_ABSCISSA,
    QUALIFICATION_GENERATOR_PROXY_NOT_ITO,
    QUALIFICATION_HORIZON_PROPAGATION_IDENTIFIED,
    QUALIFICATION_ONE_STEP_FUNCTIONAL,
    QUALIFICATION_ONE_STEP_IDENTIFIED,
    QUALIFICATION_ONE_STEP_NOT_IDENTIFIED,
    compatibility_entry,
    register_entry,
)
from mndm.dynamical_families.one_step_operator import (
    REASON_POLAR_LOG,
    REASON_RANK,
    REASON_REFLECTION,
    _operator_functionals_from_phi,
    estimate_affine_one_step_family,
)
from mndm.dynamical_families import one_step_operator as one_step_operator_mod
from mndm.pipeline.dynamical_families_export import build_dynamical_families_export
from mndm.schema import MNPSPayload


def _ou_trajectory(n: int = 400, dt: float = 0.25, seed: int = 11) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return x.astype(np.float32), time, generator


def _one_step_config(*, enabled: bool = True) -> dict:
    return {
        "dynamical_families": {
            "enabled": True,
            "one_step": {
                "enabled": enabled,
                "neighborhood": {"k": 24},
                "min_samples": 40,
                "min_neighborhood_samples": 12,
                "ridge_alpha": 1e-6,
                "one_step_rel_mse_threshold": 0.9,
            },
        }
    }


def test_linear_map_is_identified_and_abscissa_recovers_generator() -> None:
    state, time, generator = _ou_trajectory()
    result = estimate_affine_one_step_family(
        state,
        time,
        coordinate_layer="coords_3d_subject_anchored",
        neighborhood_k=24,
        min_samples=40,
        min_neighborhood_samples=12,
        ridge_alpha=1e-6,
    )
    assert result["computation_status"] == "computed"
    assert result["qualification_status"] == QUALIFICATION_ONE_STEP_IDENTIFIED
    mapped = result[MEASUREMENT_ID_AFFINE_MAP]
    assert mapped["computation_status"] == "computed"
    assert mapped["qualification_status"] == QUALIFICATION_ONE_STEP_IDENTIFIED
    assert float(mapped["summary"]["rel_mse_baseline_median"]) < 0.4
    assert mapped["summary"]["operator_scope"] == "recording"
    assert mapped["summary"]["rel_mse_scoring"] == "blocked_holdout"
    assert int(mapped["summary"]["n_blocks"]) == 2
    assert result["grain"]["native"] == "recording"
    assert result["grain"]["repeated_measure"] == "false"
    spectral = result[MEASUREMENT_ID_SPECTRAL_ABSCISSA]
    assert spectral["computation_status"] == "computed"
    assert spectral["qualification_status"] == QUALIFICATION_GENERATOR_PROXY_NOT_ITO
    values = np.asarray(spectral["series"]["spectral_abscissa"], dtype=float)
    finite = values[np.isfinite(values)]
    assert finite.size > 20
    true_alpha = float(np.max(np.real(np.linalg.eigvals(generator))))
    assert abs(float(np.median(finite)) - true_alpha) < 0.2
    numerical = result[MEASUREMENT_ID_NUMERICAL_ABSCISSA]
    assert numerical["computation_status"] == "computed"
    assert MEASUREMENT_ID_ITO_DRIFT not in result
    assert MEASUREMENT_ID_REACTIVITY_GAP not in result
    assert MEASUREMENT_ID_OPERATOR_GAIN_ANISO not in result
    assert MEASUREMENT_ID_GENERATOR_SYM_ANISO not in result
    assert "ito_qualified" not in str(result.get("summary") or {})
    horizon = result[MEASUREMENT_ID_ITERATED_ONE_STEP_MAP]
    assert horizon["computation_status"] == "computed"
    assert horizon["qualification_status"] == QUALIFICATION_HORIZON_PROPAGATION_IDENTIFIED
    assert int(horizon["interpretation_level"]) == 4
    assert horizon["summary"]["not_direct_lag2_map"] is True
    assert int(horizon["summary"]["horizon_steps"]) == 2
    dt = float(mapped["summary"]["nominal_dt_sec"])
    phi_true = expm(generator * dt)
    smax = float(np.linalg.svd(phi_true, compute_uv=False)[0])
    true_gain = float(np.log(smax) / dt)
    _sign, logabs = np.linalg.slogdet(phi_true)
    true_volume = float(logabs / dt)
    rotation, _stretch = polar(phi_true, side="right")
    logged = np.real(np.asarray(logm(np.asarray(rotation, dtype=float)), dtype=complex))
    true_rotation = float(np.linalg.norm(logged) / (np.sqrt(2.0) * dt))
    gain = result[MEASUREMENT_ID_OPERATOR_MAX_GAIN]
    volume = result[MEASUREMENT_ID_OPERATOR_VOLUME_GAIN]
    rotation_rate = result[MEASUREMENT_ID_OPERATOR_ROTATION_RATE]
    for leaf in (gain, volume, rotation_rate):
        assert leaf["computation_status"] == "computed"
        assert leaf["qualification_status"] == QUALIFICATION_ONE_STEP_FUNCTIONAL
        assert int(leaf["interpretation_level"]) == 2
        assert leaf["summary"]["not_peak_gain_level4"] is True
        assert leaf["provenance"]["settings"]["operator_polar_side"] == "right"
    gain_vals = np.asarray(gain["series"]["operator_max_gain_rate"], dtype=float)
    vol_vals = np.asarray(volume["series"]["operator_volume_gain_rate"], dtype=float)
    rot_vals = np.asarray(rotation_rate["series"]["operator_rotation_rate"], dtype=float)
    assert abs(float(np.median(gain_vals[np.isfinite(gain_vals)])) - true_gain) < 0.2
    assert abs(float(np.median(vol_vals[np.isfinite(vol_vals)])) - true_volume) < 0.2
    assert abs(float(np.median(rot_vals[np.isfinite(rot_vals)])) - true_rotation) < 0.2
    assert abs(float(np.median(gain_vals[np.isfinite(gain_vals)])) - true_alpha) > 0.05
    assert gain["summary"]["not_spectral_abscissa"] is True
    assert rotation_rate["summary"]["not_generator_rotation"] is True


def test_white_noise_does_not_identify_operator_or_write_ito() -> None:
    rng = np.random.default_rng(12)
    state = rng.normal(size=(240, 3)).astype(np.float32)
    time = np.arange(240, dtype=float) * 0.25
    result = estimate_affine_one_step_family(
        state,
        time,
        coordinate_layer="coords_3d_subject_anchored",
        min_samples=30,
        ridge_alpha=1e-4,
    )
    assert result["computation_status"] == "insufficient_support"
    assert result["failure_reason"] == "one_step_fit_not_better_than_baseline"
    assert result["qualification_status"] == QUALIFICATION_ONE_STEP_NOT_IDENTIFIED
    assert result[MEASUREMENT_ID_SPECTRAL_ABSCISSA]["computation_status"] == "not_testable"
    assert result[MEASUREMENT_ID_SPECTRAL_ABSCISSA]["failure_reason"] == "upstream_one_step_not_identified"
    for measurement_id in (
        MEASUREMENT_ID_OPERATOR_MAX_GAIN,
        MEASUREMENT_ID_OPERATOR_VOLUME_GAIN,
        MEASUREMENT_ID_OPERATOR_ROTATION_RATE,
    ):
        leaf = result[measurement_id]
        assert leaf["computation_status"] == "not_testable"
        assert leaf["failure_reason"] == "upstream_one_step_not_identified"
        assert leaf["qualification_status"] == QUALIFICATION_ONE_STEP_FUNCTIONAL
    assert MEASUREMENT_ID_ITO_DRIFT not in result
    horizon = result[MEASUREMENT_ID_ITERATED_ONE_STEP_MAP]
    assert horizon["computation_status"] == "not_testable"
    assert horizon["failure_reason"] == "upstream_one_step_not_identified"
    spectral = np.asarray(
        result[MEASUREMENT_ID_SPECTRAL_ABSCISSA].get("series", {}).get("spectral_abscissa", []),
        dtype=float,
    )
    assert spectral.size == 0 or not np.any(np.isfinite(spectral))


def test_irregular_dt_is_not_testable() -> None:
    state, time, _ = _ou_trajectory(n=80)
    time = time.copy()
    time[40:] += 3.0
    result = estimate_affine_one_step_family(
        state, time, coordinate_layer="coords_3d_subject_anchored"
    )
    assert result["computation_status"] == "not_testable"
    assert result["failure_reason"] == "materially_irregular_increment_timestep"
    for measurement_id in (
        MEASUREMENT_ID_OPERATOR_MAX_GAIN,
        MEASUREMENT_ID_OPERATOR_VOLUME_GAIN,
        MEASUREMENT_ID_OPERATOR_ROTATION_RATE,
    ):
        leaf = result[measurement_id]
        assert leaf["computation_status"] == "not_testable"
        assert leaf["failure_reason"] == "materially_irregular_increment_timestep"
        assert leaf["qualification_status"] == QUALIFICATION_ONE_STEP_FUNCTIONAL


def test_nine_d_layer_is_not_testable() -> None:
    state, time, _ = _ou_trajectory(n=80)
    nine = np.concatenate([state, state, state], axis=1)
    result = estimate_affine_one_step_family(
        nine, time, coordinate_layer="coords_9d"
    )
    assert result["computation_status"] == "not_testable"
    assert result["failure_reason"] == "one_step_subject_anchored_3d_only"
    for measurement_id in (
        MEASUREMENT_ID_OPERATOR_MAX_GAIN,
        MEASUREMENT_ID_OPERATOR_VOLUME_GAIN,
        MEASUREMENT_ID_OPERATOR_ROTATION_RATE,
    ):
        leaf = result[measurement_id]
        assert leaf["computation_status"] == "not_testable"
        assert leaf["failure_reason"] == "one_step_subject_anchored_3d_only"
        assert leaf["qualification_status"] == QUALIFICATION_ONE_STEP_FUNCTIONAL
    three = estimate_affine_one_step_family(
        state, time, coordinate_layer="coords_9d"
    )
    assert three["failure_reason"] == "one_step_subject_anchored_3d_only"
    assert MEASUREMENT_ID_OPERATOR_MAX_GAIN in three


def test_yaml_forbidden_keys_and_common_default_off() -> None:
    import yaml

    state, time, _ = _ou_trajectory(n=80)
    with pytest.raises(ValueError, match="ito_drift_level3"):
        build_dynamical_families_export(
            config={
                "dynamical_families": {
                    "enabled": True,
                    "one_step": {"enabled": True, "ito_drift_level3": {"enabled": True}},
                }
            },
            state=state,
            time=time,
            stage=None,
            segment_id=np.zeros(time.size, dtype=np.int32),
            coordinate_layer="coords_3d_subject_anchored",
            coordinate_names=["m", "d", "e"],
        )
    with pytest.raises(ValueError, match="jacobian_expm_as_fitted_phi"):
        build_dynamical_families_export(
            config={
                "dynamical_families": {
                    "enabled": True,
                    "one_step": {"enabled": True, "jacobian_expm_as_fitted_phi": True},
                }
            },
            state=state,
            time=time,
            stage=None,
            segment_id=np.zeros(time.size, dtype=np.int32),
            coordinate_layer="coords_3d_subject_anchored",
            coordinate_names=["m", "d", "e"],
        )
    with pytest.raises(ValueError, match="peak_gain_from_phi_powers"):
        build_dynamical_families_export(
            config={
                "dynamical_families": {
                    "enabled": True,
                    "one_step": {"enabled": True, "peak_gain_from_phi_powers": True},
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
    assert cfg["dynamical_families"]["one_step"]["enabled"] is True
    icare = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "sources"
        / "other"
        / "config_ingest_physionet_i-care_2_1_dynamical_families.yaml"
    )
    with icare.open(encoding="utf-8") as handle:
        icare_cfg = yaml.safe_load(handle)
    one_step = (icare_cfg.get("dynamical_families") or {}).get("one_step") or {}
    assert one_step.get("enabled") is True
    next140 = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "sources"
        / "other"
        / "config_ingest_physionet_i-care_2_1_next_140_0_12h_dynamical_families.yaml"
    )
    with next140.open(encoding="utf-8") as handle:
        next_cfg = yaml.safe_load(handle)
    next_one_step = (next_cfg.get("dynamical_families") or {}).get("one_step") or {}
    assert next_one_step.get("enabled") is True


def test_registry_write_and_h5_nested_names(tmp_path: Path) -> None:
    assert "one_step" in WRITABLE_FAMILY_IDS
    assert family_forbids("one_step", "jacobian_expm_as_fitted_phi")
    assert family_forbids("one_step", "abscissa_from_unidentified_operator")
    assert family_forbids("one_step", "peak_gain_from_phi_powers")
    assert family_forbids("one_step", "operator_gain_as_spectral_abscissa")
    assert family_forbids("one_step", "epsilon_rescued_volume_gain")
    assert family_forbids("one_step", MEASUREMENT_ID_REACTIVITY_GAP)
    assert family_forbids("one_step", "abscissa_difference_as_reactivity_gap")
    assert family_forbids("one_step", "jacobian_reactivity_gap_as_level3")
    assert family_forbids("one_step", MEASUREMENT_ID_OPERATOR_GAIN_ANISO)
    assert family_forbids("one_step", MEASUREMENT_ID_GENERATOR_SYM_ANISO)
    assert get_family("one_step")["schema"] == AFFINE_ONE_STEP_SCHEMA_VERSION
    assert get_family("one_step")["namespace"] == "/dynamical_families/one_step/v1"
    state, time, _ = _ou_trajectory()
    export = build_dynamical_families_export(
        config=_one_step_config(),
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
    output = write_h5(tmp_path / "one_step.h5", "one_step", payload)
    with h5py.File(output, "r") as handle:
        root = "/dynamical_families/one_step/v1"
        assert f"{root}/{MEASUREMENT_ID_AFFINE_MAP}" in handle
        assert f"{root}/{MEASUREMENT_ID_AFFINE_MEAN_RATE}" in handle
        assert f"{root}/{MEASUREMENT_ID_INNOVATION_COVARIANCE}" in handle
        assert f"{root}/{MEASUREMENT_ID_SPECTRAL_ABSCISSA}" in handle
        assert f"{root}/{MEASUREMENT_ID_NUMERICAL_ABSCISSA}" in handle
        assert f"{root}/{MEASUREMENT_ID_DIVERGENCE}" in handle
        assert f"{root}/{MEASUREMENT_ID_GENERATOR_ROTATION}" in handle
        assert f"{root}/{MEASUREMENT_ID_OPERATOR_MAX_GAIN}" in handle
        assert f"{root}/{MEASUREMENT_ID_OPERATOR_VOLUME_GAIN}" in handle
        assert f"{root}/{MEASUREMENT_ID_OPERATOR_ROTATION_RATE}" in handle
        assert f"{root}/{MEASUREMENT_ID_ITERATED_ONE_STEP_MAP}" in handle
        assert f"{root}/ito_drift_level3" not in handle
        assert f"{root}/finite_time_peak_gain_level4" not in handle
        assert f"{root}/{MEASUREMENT_ID_REACTIVITY_GAP}" not in handle
        assert f"{root}/{MEASUREMENT_ID_OPERATOR_GAIN_ANISO}" not in handle
        assert f"{root}/{MEASUREMENT_ID_GENERATOR_SYM_ANISO}" not in handle
        grain = handle[f"{root}/grain"]
        assert grain["native"][()].decode() == "recording"
        assert grain["repeated_measure"][()].decode() == "false"
        group = handle[f"{root}/{MEASUREMENT_ID_AFFINE_MAP}"]
        assert group["measurement_id"][()].decode() == MEASUREMENT_ID_AFFINE_MAP
        assert int(group["interpretation_level"][()]) == 2
        assert group["computation_status"][()].decode() == "computed"
        spec = handle[f"{root}/{MEASUREMENT_ID_SPECTRAL_ABSCISSA}"]
        assert spec["qualification_status"][()].decode() == QUALIFICATION_GENERATOR_PROXY_NOT_ITO
        assert int(spec["interpretation_level"][()]) == 3
        gain = handle[f"{root}/{MEASUREMENT_ID_OPERATOR_MAX_GAIN}"]
        assert gain["qualification_status"][()].decode() == QUALIFICATION_ONE_STEP_FUNCTIONAL
        assert int(gain["interpretation_level"][()]) == 2


def test_operator_functionals_fail_closed_on_rank_and_reflection() -> None:
    dt = 0.25
    rank1 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)
    rank_out = _operator_functionals_from_phi(rank1, dt)
    assert rank_out["max_gain"][0] == pytest.approx(np.log(1.0) / dt)
    assert rank_out["volume_gain"] == (None, REASON_RANK)
    assert rank_out["rotation_rate"] == (None, REASON_RANK)
    reflection = np.diag([1.0, 1.0, -1.0])
    reflected = _operator_functionals_from_phi(reflection, dt)
    assert reflected["max_gain"][0] == pytest.approx(0.0)
    assert reflected["volume_gain"][0] == pytest.approx(0.0)
    assert reflected["rotation_rate"] == (None, REASON_REFLECTION)


def test_operator_functionals_fail_closed_on_nonreal_log_and_zero_det(monkeypatch) -> None:
    dt = 0.25
    identity = np.eye(3, dtype=float)
    original_logm = one_step_operator_mod.logm

    def _complex_logm(matrix):
        logged = np.asarray(original_logm(matrix), dtype=complex)
        return logged + 1.0j * np.eye(logged.shape[0])

    monkeypatch.setattr(one_step_operator_mod, "logm", _complex_logm)
    nonreal = _operator_functionals_from_phi(identity, dt)
    assert nonreal["max_gain"][0] == pytest.approx(0.0)
    assert nonreal["volume_gain"][0] == pytest.approx(0.0)
    assert nonreal["rotation_rate"] == (None, REASON_POLAR_LOG)
    monkeypatch.setattr(one_step_operator_mod, "logm", original_logm)

    def _zero_sign_slogdet(matrix):
        return 0, 0.0

    monkeypatch.setattr(np.linalg, "slogdet", _zero_sign_slogdet)
    zero_det = _operator_functionals_from_phi(identity, dt)
    assert zero_det["max_gain"][0] == pytest.approx(0.0)
    assert zero_det["volume_gain"] == (None, REASON_RANK)
    assert zero_det["rotation_rate"][0] is not None


def test_early_lag1_refusal_synthesizes_functional_leaves() -> None:
    state = np.zeros((8, 3), dtype=np.float32)
    time = np.arange(8, dtype=float) * 0.25
    result = estimate_affine_one_step_family(
        state, time, coordinate_layer="coords_3d_subject_anchored"
    )
    assert result["computation_status"] in {"insufficient_support", "invalid", "not_testable"}
    assert MEASUREMENT_ID_AFFINE_MAP not in result
    for measurement_id in (
        MEASUREMENT_ID_OPERATOR_MAX_GAIN,
        MEASUREMENT_ID_OPERATOR_VOLUME_GAIN,
        MEASUREMENT_ID_OPERATOR_ROTATION_RATE,
    ):
        leaf = result[measurement_id]
        assert leaf["computation_status"] == "not_testable"
        assert leaf["qualification_status"] == QUALIFICATION_ONE_STEP_FUNCTIONAL
        assert leaf["failure_reason"]
        series = leaf.get("series") or {}
        for values in series.values():
            arr = np.asarray(values, dtype=float)
            assert arr.size == 0 or not np.any(np.isfinite(arr))


def test_identified_rank_deficient_phi_fails_volume_and_rotation_leaves(monkeypatch) -> None:
    real_fit = one_step_operator_mod._fit_affine_map
    calls = {"n": 0}

    def _rank1_full_fit(*args, **kwargs):
        calls["n"] += 1
        fitted = real_fit(*args, **kwargs)
        if fitted is None:
            return None
        phi, intercept, x_ref = fitted
        if calls["n"] == 3:
            phi = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        return phi, intercept, x_ref

    monkeypatch.setattr(one_step_operator_mod, "_fit_affine_map", _rank1_full_fit)
    state, time, _ = _ou_trajectory()
    result = estimate_affine_one_step_family(
        state,
        time,
        coordinate_layer="coords_3d_subject_anchored",
        neighborhood_k=24,
        min_samples=40,
        min_neighborhood_samples=12,
        ridge_alpha=1e-6,
    )
    assert result[MEASUREMENT_ID_AFFINE_MAP]["qualification_status"] == QUALIFICATION_ONE_STEP_IDENTIFIED
    gain = result[MEASUREMENT_ID_OPERATOR_MAX_GAIN]
    volume = result[MEASUREMENT_ID_OPERATOR_VOLUME_GAIN]
    rotation_rate = result[MEASUREMENT_ID_OPERATOR_ROTATION_RATE]
    assert gain["computation_status"] == "computed"
    assert volume["computation_status"] == "insufficient_support"
    assert volume["failure_reason"] == REASON_RANK
    assert rotation_rate["computation_status"] == "insufficient_support"
    assert rotation_rate["failure_reason"] == REASON_RANK
    assert volume["qualification_status"] == QUALIFICATION_ONE_STEP_FUNCTIONAL
    assert calls["n"] >= 3


def test_identified_reflection_phi_fails_rotation_leaf(monkeypatch) -> None:
    real_fit = one_step_operator_mod._fit_affine_map
    calls = {"n": 0}

    def _reflection_full_fit(*args, **kwargs):
        calls["n"] += 1
        fitted = real_fit(*args, **kwargs)
        if fitted is None:
            return None
        phi, intercept, x_ref = fitted
        if calls["n"] == 3:
            phi = np.diag([1.0, 1.0, -1.0]).astype(np.float32)
        return phi, intercept, x_ref

    monkeypatch.setattr(one_step_operator_mod, "_fit_affine_map", _reflection_full_fit)
    state, time, _ = _ou_trajectory()
    result = estimate_affine_one_step_family(
        state,
        time,
        coordinate_layer="coords_3d_subject_anchored",
        neighborhood_k=24,
        min_samples=40,
        min_neighborhood_samples=12,
        ridge_alpha=1e-6,
    )
    assert result[MEASUREMENT_ID_AFFINE_MAP]["qualification_status"] == QUALIFICATION_ONE_STEP_IDENTIFIED
    volume = result[MEASUREMENT_ID_OPERATOR_VOLUME_GAIN]
    rotation_rate = result[MEASUREMENT_ID_OPERATOR_ROTATION_RATE]
    assert volume["computation_status"] == "computed"
    assert rotation_rate["computation_status"] == "insufficient_support"
    assert rotation_rate["failure_reason"] == REASON_REFLECTION
    assert calls["n"] >= 3


def test_innovation_is_not_withheld_and_not_diffusion() -> None:
    row = compatibility_entry(MEASUREMENT_ID_INNOVATION_COVARIANCE)
    entry = register_entry(MEASUREMENT_ID_INNOVATION_COVARIANCE)
    assert row["relation"] != "withheld_not_written"
    assert entry["written_by_one_step_family"] is True
    assert entry["written_by_diffusion_family"] is False
    assert entry["physical_path"].endswith(MEASUREMENT_ID_INNOVATION_COVARIANCE)
