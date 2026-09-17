"""SL-LEV-MES-003 §8 «slutligen»: bounded synthetic pilot of frozen identities.

Known mechanism, null, gaps, rank deficiency, insufficient support, and
denied Itô qualification leaving lower objects marked. No I-CARE cohort.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))

from mndm.dynamical_families.chart_drift import REASON_C2_CLOSED
from mndm.dynamical_families.diffusion_geometry import estimate_local_diffusion_geometry
from mndm.dynamical_families.finite_lag_drift import (
    REASON_INSUFFICIENT_LOCAL,
    REASON_LAG_INCONSISTENT,
    estimate_conditional_mean_rate_level1,
    estimate_lag_diagnostics,
    estimate_realized_velocity_level0,
)
from mndm.dynamical_families.measurement_register import (
    MEASUREMENT_ID_CONDITIONAL_COVARIANCE,
    MEASUREMENT_ID_CONDITIONAL_MEAN_RATE,
    MEASUREMENT_ID_INCREMENT_COVARIANCE,
    MEASUREMENT_ID_ITO_DIFFUSION,
    MEASUREMENT_ID_ITO_DRIFT,
    MEASUREMENT_ID_REALIZED_VELOCITY,
    QUALIFICATION_ITO_NOT_QUALIFIED,
    QUALIFICATION_NOT_ASSESSED,
    SUPPORT_SUFFICIENT,
    VARIANT_LAG_DIAGNOSTICS,
    VARIANT_POOLED,
)
from mndm.pipeline.dynamical_families_export import build_dynamical_families_export
from mndm.projection import estimate_derivatives

ROOT = Path(__file__).resolve().parents[2]
COMMON_FAMILIES = ROOT / "mndm" / "config" / "config_ingest_common_dynamical_families.yaml"
ICARE_FAMILIES = (
    ROOT
    / "mndm"
    / "config"
    / "sources"
    / "other"
    / "config_ingest_physionet_i-care_2_1_dynamical_families.yaml"
)

_B_TRUE = np.array([0.50, -0.25, 0.10], dtype=float)
_SIGMA = 0.05
_LEVEL1_MEAN_TOL = 0.15
_BROWNIAN_MEAN_TOL = 0.12


def _linear_sde(
    *,
    seed: int = 7,
    n: int = 1800,
    dt: float = 0.01,
    sigma: float = _SIGMA,
    b: np.ndarray = _B_TRUE,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=sigma * np.sqrt(dt), size=(n - 1, 3))
    state = np.zeros((n, 3), dtype=float)
    for i in range(n - 1):
        state[i + 1] = state[i] + b * dt + noise[i]
    time = np.arange(n, dtype=float) * dt
    return state, time


def _period2_oscillation(*, n: int = 800, dt: float = 0.01, amplitude: float = 5.0):
    state = np.zeros((n, 3), dtype=float)
    state[1::2] = amplitude
    time = np.arange(n, dtype=float) * dt
    return state, time


def _rank1_chart(*, seed: int = 21, n: int = 900, dt: float = 0.01, sigma: float = _SIGMA):
    rng = np.random.default_rng(seed)
    line = np.cumsum(rng.normal(scale=sigma * np.sqrt(dt), size=n))
    state = np.zeros((n, 3), dtype=float)
    state[:, 0] = line
    time = np.arange(n, dtype=float) * dt
    return state, time


def _drift_kwargs() -> dict:
    return dict(
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
        neighborhood_k=20,
        min_samples=30,
        min_neighborhood_samples=10,
        max_dt_relative_deviation=0.05,
        min_valid_fraction=0.1,
        weight_mode="inverse_distance",
    )


def _diffusion_kwargs() -> dict:
    return dict(
        neighborhood_k=20,
        min_samples=30,
        min_neighborhood_samples=10,
        max_dt_relative_deviation=0.05,
    )


def _pilot_export_config() -> dict:
    return {
        "dynamical_families": {
            "enabled": True,
            "coordinate_layer": "subject_anchored",
            "diffusion": {
                "enabled": True,
                "neighborhood": {"k": 20},
                "min_samples": 30,
                "min_neighborhood_samples": 10,
            },
            "drift": {
                "enabled": True,
                "neighborhood": {"k": 20},
                "min_samples": 30,
                "min_neighborhood_samples": 10,
                "weight_mode": "inverse_distance",
                "realized_velocity_level0": {"enabled": True},
                "conditional_mean_rate_level1": {
                    "pooled": {"enabled": True},
                    "blocked_crossfit": {"enabled": False},
                    "lag_diagnostics": {
                        "enabled": True,
                        "lags": [1, 2, 4],
                        "consistency_rel_tol": 0.5,
                    },
                },
            },
        }
    }


def test_pilot_known_linear_sde_recovers_b_and_keeps_a_hat_raw() -> None:
    state, time = _linear_sde()
    dt = float(time[1] - time[0])
    pooled = estimate_conditional_mean_rate_level1(state, time, **_drift_kwargs())
    realized = estimate_realized_velocity_level0(
        state,
        time,
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
        min_samples=30,
    )
    diffusion = estimate_local_diffusion_geometry(state, time, **_diffusion_kwargs())
    assert pooled["computation_status"] == "computed"
    assert realized["computation_status"] == "computed"
    assert diffusion["computation_status"] == "computed"
    valid = np.asarray(pooled["series"]["valid"]) == 1
    mean_b = np.mean(np.asarray(pooled["series"]["b_hat"], dtype=float)[valid], axis=0)
    assert np.all(np.sign(mean_b) == np.sign(_B_TRUE))
    assert np.all(np.abs(mean_b - _B_TRUE) < _LEVEL1_MEAN_TOL)
    assert pooled["qualification_status"] == QUALIFICATION_ITO_NOT_QUALIFIED
    assert realized["qualification_status"] == QUALIFICATION_NOT_ASSESSED
    dx_dt = np.asarray(realized["series"]["dx_dt"], dtype=float)
    realized_valid = np.asarray(realized["series"]["valid"]) == 1
    savgol = estimate_derivatives(state, dt=dt, method="sav_gol", window=7, polyorder=3)
    assert not np.allclose(dx_dt[realized_valid], savgol[realized_valid], atol=1e-5)
    assert diffusion["measurement_id"] == MEASUREMENT_ID_CONDITIONAL_COVARIANCE
    assert diffusion["summary"]["estimand"] == "centered_increment_covariance_over_nominal_dt"
    assert diffusion["summary"]["diffusion_convention"] == "a_not_D_over_2"
    assert diffusion["summary"]["a_semantics"] == "raw_increment_covariance"
    assert diffusion["summary"]["not_ito_diffusion_tensor"] is True
    assert diffusion["summary"]["A_bD_computation_status"] == "not_testable"
    assert np.all(np.isnan(diffusion["series"]["A_bD"]))
    d_total = float(np.nanmean(diffusion["series"]["D_total"]))
    centered_trace = 3.0 * (_SIGMA**2)
    uncentered_trace = centered_trace + dt * float(np.dot(_B_TRUE, _B_TRUE))
    assert abs(d_total - centered_trace) < abs(d_total - uncentered_trace)
    assert np.isclose(d_total, centered_trace, rtol=0.35)
    refused_c2 = estimate_local_diffusion_geometry(
        state,
        time,
        residualize_increments=True,
        **_diffusion_kwargs(),
    )
    assert refused_c2["computation_status"] == "invalid"
    assert refused_c2["failure_reason"] == REASON_C2_CLOSED
    assert refused_c2["series"] == {}
    c2_increment = refused_c2[MEASUREMENT_ID_INCREMENT_COVARIANCE]
    assert c2_increment["computation_status"] == "computed"


def test_pilot_brownian_null_is_near_zero_and_not_ito() -> None:
    state, time = _linear_sde(b=np.zeros(3), seed=11)
    pooled = estimate_conditional_mean_rate_level1(state, time, **_drift_kwargs())
    diagnostics = estimate_lag_diagnostics(state, time, **_drift_kwargs())
    valid = np.asarray(pooled["series"]["valid"]) == 1
    mean_b = np.mean(np.asarray(pooled["series"]["b_hat"], dtype=float)[valid], axis=0)
    assert np.all(np.abs(mean_b) < _BROWNIAN_MEAN_TOL)
    assert diagnostics["qualification_status"] == QUALIFICATION_ITO_NOT_QUALIFIED
    assert diagnostics["computation_status"] == "computed"
    diffusion = estimate_local_diffusion_geometry(state, time, **_diffusion_kwargs())
    assert diffusion["computation_status"] == "computed"
    assert diffusion["qualification_status"] == QUALIFICATION_ITO_NOT_QUALIFIED
    assert diffusion["summary"]["not_ito_diffusion_tensor"] is True


def test_pilot_gaps_are_nan_not_zero_for_mean_and_covariance() -> None:
    state, time = _linear_sde(n=500)
    state = state.copy()
    state[200] = np.nan
    pooled = estimate_conditional_mean_rate_level1(state, time, **_drift_kwargs())
    diffusion = estimate_local_diffusion_geometry(state, time, **_diffusion_kwargs())
    assert pooled["computation_status"] == "computed"
    assert diffusion["computation_status"] == "computed"
    assert int(pooled["series"]["valid"][200]) == 0
    assert np.all(np.isnan(pooled["series"]["b_hat"][200]))
    assert not np.any(pooled["series"]["b_hat"][200] == 0.0)
    assert int(diffusion["series"]["valid"][200]) == 0
    assert np.all(np.isnan(diffusion["series"]["a_hat"][200]))
    pooled_sources = set(np.asarray(pooled["series"]["source_idx"]).tolist())
    diffusion_sources = set(np.asarray(diffusion["series"]["source_idx"]).tolist())
    assert 198 in pooled_sources
    assert 199 not in pooled_sources
    assert 200 not in pooled_sources
    assert 198 in diffusion_sources
    assert 199 not in diffusion_sources
    assert 200 not in diffusion_sources


def test_pilot_rank_deficient_chart_floors_psd_and_is_not_ito_tensor() -> None:
    state, time = _rank1_chart()
    diffusion = estimate_local_diffusion_geometry(state, time, **_diffusion_kwargs())
    full_state, full_time = _linear_sde()
    full_rank = estimate_local_diffusion_geometry(full_state, full_time, **_diffusion_kwargs())
    assert diffusion["computation_status"] == "computed"
    assert diffusion["qualification_status"] == QUALIFICATION_ITO_NOT_QUALIFIED
    # Register identity stamp, not a rank-specific result.
    assert diffusion["summary"]["not_ito_diffusion_tensor"] is True
    assert full_rank["summary"]["not_ito_diffusion_tensor"] is True
    valid = np.asarray(diffusion["series"]["valid"]) == 1
    full_valid = np.asarray(full_rank["series"]["valid"]) == 1
    assert int(np.sum(valid)) > 10
    psd_floor = 1.0e-8
    raw_min = np.asarray(diffusion["series"]["raw_min_eigenvalue"], dtype=float)[valid]
    full_raw_min = np.asarray(full_rank["series"]["raw_min_eigenvalue"], dtype=float)[full_valid]
    assert float(np.median(raw_min)) < psd_floor
    assert float(np.median(full_raw_min)) > psd_floor
    assert int(np.sum(np.asarray(diffusion["series"]["psd_floor_applied"])[valid])) == int(np.sum(valid))
    floored_min = np.linalg.eigvalsh(np.asarray(diffusion["series"]["a_hat"], dtype=float)[valid])[:, 0]
    assert np.isclose(float(np.median(floored_min)), psd_floor, atol=1e-12)
    assert diffusion["summary"]["A_bD_computation_status"] == "not_testable"


def test_pilot_insufficient_support_does_not_zero_fill() -> None:
    state, time = _linear_sde(n=21)
    pooled = estimate_conditional_mean_rate_level1(
        state,
        time,
        min_samples=10,
        neighborhood_k=20,
        min_neighborhood_samples=10,
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
    )
    assert pooled["computation_status"] == "insufficient_support"
    assert pooled["failure_reason"] == REASON_INSUFFICIENT_LOCAL
    assert pooled["series"] == {}
    diffusion = estimate_local_diffusion_geometry(
        state[:8],
        time[:8],
        min_samples=10,
        neighborhood_k=20,
        min_neighborhood_samples=10,
    )
    assert diffusion["computation_status"] == "insufficient_support"
    assert diffusion["failure_reason"] == "insufficient_samples"
    assert diffusion["series"] == {}


def test_pilot_denied_ito_keeps_lower_objects_marked() -> None:
    state, time = _period2_oscillation()
    export = build_dynamical_families_export(
        config=_pilot_export_config(),
        state=state,
        time=time,
        stage=None,
        segment_id=np.zeros(time.size, dtype=np.int32),
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
    )
    realized = export["drift"][MEASUREMENT_ID_REALIZED_VELOCITY]
    pooled = export["drift"][MEASUREMENT_ID_CONDITIONAL_MEAN_RATE][VARIANT_POOLED]
    diagnostics = export["drift"][MEASUREMENT_ID_CONDITIONAL_MEAN_RATE][VARIANT_LAG_DIAGNOSTICS]
    diffusion = export["diffusion"]
    assert realized["computation_status"] == "computed"
    assert realized["qualification_status"] == QUALIFICATION_NOT_ASSESSED
    assert pooled["computation_status"] == "computed"
    assert pooled["support_status"] == SUPPORT_SUFFICIENT
    assert pooled["qualification_status"] == QUALIFICATION_ITO_NOT_QUALIFIED
    assert diagnostics["summary"]["multi_lag_consistency"] == REASON_LAG_INCONSISTENT
    assert diagnostics["qualification_status"] == QUALIFICATION_ITO_NOT_QUALIFIED
    assert diffusion["computation_status"] == "computed"
    assert diffusion["qualification_status"] == QUALIFICATION_ITO_NOT_QUALIFIED
    assert "a_hat" in diffusion["series"]
    assert MEASUREMENT_ID_ITO_DRIFT not in export["drift"]
    assert MEASUREMENT_ID_ITO_DRIFT not in export["diffusion"]
    assert MEASUREMENT_ID_ITO_DIFFUSION not in export["diffusion"]


def test_pilot_irregular_dt_is_not_testable_for_both_families() -> None:
    state = np.zeros((80, 3), dtype=float)
    state[:, 0] = np.arange(80, dtype=float)
    time = np.arange(80, dtype=float)
    time[40:] += 1.0
    pooled = estimate_conditional_mean_rate_level1(state, time, **_drift_kwargs())
    diffusion = estimate_local_diffusion_geometry(state, time, **_diffusion_kwargs())
    assert pooled["computation_status"] == "not_testable"
    assert pooled["failure_reason"] == "materially_irregular_increment_timestep"
    assert pooled["series"] == {}
    assert diffusion["computation_status"] == "not_testable"
    assert diffusion["failure_reason"] == "materially_irregular_increment_timestep"
    assert diffusion["series"] == {}
    increment = diffusion[MEASUREMENT_ID_INCREMENT_COVARIANCE]
    assert increment["computation_status"] == "computed"
    assert increment["summary"]["not_divided_by_dt"] is True


def test_pilot_reference_overlays_keep_new_families_on() -> None:
    common = yaml.safe_load(COMMON_FAMILIES.read_text(encoding="utf-8"))
    assert common["dynamical_families"]["enabled"] is True
    assert common["dynamical_families"]["drift"]["enabled"] is True
    assert common["dynamical_families"]["diffusion"]["enabled"] is True
    assert common["dynamical_families"]["one_step"]["enabled"] is True
    assert common["dynamical_families"]["amplification"]["enabled"] is True
    assert common["dynamical_families"]["history"]["enabled"] is True
    assert common["dynamical_families"]["turning"]["enabled"] is True
    from core.config_loader import load_config

    merged = load_config(ICARE_FAMILIES)
    families = merged["dynamical_families"]
    icare_one_step = families.get("one_step") or {}
    assert icare_one_step.get("enabled") is True
    assert families["diffusion"]["enabled"] is True
    assert families["amplification"]["enabled"] is True
    assert families["history"]["enabled"] is True
    assert families["turning"]["enabled"] is True
