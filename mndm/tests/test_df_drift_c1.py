"""DF-DRIFT-C1 M1–M2: alignment-only split, Gate C1-A, forbidden sources."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))

from mndm.dynamical_families.chart_drift import (
    FORBIDDEN_SOURCE_REASONS,
    REASON_C2_CLOSED,
    REASON_CROSSFIT_BEFORE_M3,
    REASON_NOT_SUPPLIED,
    REASON_PROTOCOL_CLOSED,
    resolve_chart_drift,
    resolve_ingest_chart_drift,
)
from mndm.dynamical_families.diffusion_geometry import estimate_local_diffusion_geometry
from mndm.pipeline.dynamical_families_export import build_dynamical_families_export

ROOT = Path(__file__).resolve().parents[2]
COMMON_FAMILIES = ROOT / "mndm" / "config" / "config_ingest_common_dynamical_families.yaml"


def _brownian_path(seed: int = 23, n: int = 800, dt: float = 0.01, sigma: float = 0.4):
    rng = np.random.default_rng(seed)
    increments = rng.normal(scale=sigma * np.sqrt(dt), size=(n - 1, 2))
    state = np.vstack([np.zeros((1, 2)), np.cumsum(increments, axis=0)])
    time = np.arange(n, dtype=float) * dt
    return state, time


def _estimator_kwargs(n: int) -> dict:
    return dict(
        neighborhood_k=n - 1,
        min_samples=100,
        min_neighborhood_samples=100,
    )


def test_gate_c1_a_alignment_drift_leaves_a_hat_byte_identical_to_no_drift() -> None:
    state, time = _brownian_path()
    n = state.shape[0]
    phase = np.linspace(0.0, 4.0 * np.pi, n)
    alignment_b = np.stack([np.sin(phase), np.cos(phase)], axis=1)
    kwargs = _estimator_kwargs(n)
    none = estimate_local_diffusion_geometry(state, time, **kwargs)
    c1 = estimate_local_diffusion_geometry(
        state,
        time,
        drift=alignment_b,
        residualize_increments=False,
        drift_source="truth_known_chart_b",
        **kwargs,
    )
    assert none["computation_status"] == c1["computation_status"] == "computed"
    for key in ("a_hat", "D_total", "d_diff", "c_diff", "valid"):
        assert np.array_equal(none["series"][key], c1["series"][key], equal_nan=True)
    assert none["summary"]["A_bD_computation_status"] == "not_testable"
    assert c1["summary"]["A_bD_computation_status"] == "computed"
    assert c1["summary"]["a_semantics"] == "raw_increment_covariance"
    assert c1["summary"]["ratio_semantics"] == "chart_velocity_to_increment_spread"
    assert c1["provenance"]["settings"]["drift_mode"] == "alignment_only"
    assert c1["provenance"]["settings"]["drift_residualization"] == "none"
    assert c1["provenance"]["settings"]["drift_source"] == "truth_known_chart_b"


def test_residualize_increments_is_invalid_until_c2_authorized() -> None:
    state, time = _brownian_path(seed=5, n=120)
    kwargs = dict(neighborhood_k=40, min_samples=30, min_neighborhood_samples=10)
    none = estimate_local_diffusion_geometry(
        state, time, residualize_increments=True, **kwargs
    )
    assert none["computation_status"] == "invalid"
    assert none["failure_reason"] == REASON_C2_CLOSED
    with_field = estimate_local_diffusion_geometry(
        state,
        time,
        drift=np.ones_like(state),
        residualize_increments=True,
        drift_source="truth_known_chart_b",
        **kwargs,
    )
    assert with_field["computation_status"] == "invalid"
    assert with_field["failure_reason"] == REASON_C2_CLOSED
    assert "a_hat" not in with_field.get("series", {})


def test_forbidden_chart_drift_sources_do_not_return_a_field() -> None:
    dummy = np.ones((12, 2), dtype=float)
    for token, reason in FORBIDDEN_SOURCE_REASONS.items():
        resolved = resolve_chart_drift(source=token, field=dummy, enabled=True)
        assert resolved.field is None
        assert resolved.failure_reason == reason
        assert resolved.residualize_increments is False


def test_estimator_refuses_forbidden_source_vector_and_keeps_raw_a_hat() -> None:
    state, time = _brownian_path()
    n = state.shape[0]
    kwargs = _estimator_kwargs(n)
    none = estimate_local_diffusion_geometry(state, time, **kwargs)
    fake_xdot = np.ones_like(state)
    refused = estimate_local_diffusion_geometry(
        state,
        time,
        drift=fake_xdot,
        residualize_increments=False,
        drift_source="mnps_xdot",
        **kwargs,
    )
    assert refused["computation_status"] == "computed"
    assert np.array_equal(none["series"]["a_hat"], refused["series"]["a_hat"], equal_nan=True)
    assert refused["summary"]["A_bD_computation_status"] == "not_testable"
    assert refused["summary"]["drift_alignment_failure_reason"] == "mnps_xdot_as_sde_drift"
    assert np.all(np.isnan(refused["series"]["A_bD"]))


def test_crossfit_and_c2_mode_are_closed_before_m3() -> None:
    dummy = np.ones((8, 2), dtype=float)
    crossfit = resolve_chart_drift(
        source="crossfit_local_chart_b", field=dummy, enabled=True
    )
    assert crossfit.field is None
    assert crossfit.failure_reason == REASON_CROSSFIT_BEFORE_M3
    protocol = resolve_chart_drift(
        source="protocol_imposed_chart_b", field=dummy, enabled=True
    )
    assert protocol.field is None
    assert protocol.failure_reason == REASON_PROTOCOL_CLOSED
    c2 = resolve_chart_drift(
        source="truth_known_chart_b",
        field=dummy,
        mode="residualize_increments",
        enabled=True,
    )
    assert c2.field is None
    assert c2.failure_reason == REASON_C2_CLOSED


def test_ingest_export_stays_not_supplied_even_if_yaml_requests_xdot() -> None:
    state, time = _brownian_path()
    export = build_dynamical_families_export(
        config={
            "dynamical_families": {
                "enabled": True,
                "diffusion": {
                    "enabled": True,
                    "neighborhood": {"k": 799},
                    "min_samples": 100,
                    "min_neighborhood_samples": 100,
                    "drift": {
                        "enabled": True,
                        "mode": "alignment_only",
                        "source": "mnps_xdot",
                    },
                },
            }
        },
        state=state,
        time=time,
        stage=None,
        segment_id=None,
        coordinate_layer="subject_anchored",
        coordinate_names=["m", "d"],
    )
    result = export["diffusion"]
    assert result["computation_status"] == "computed"
    assert result["summary"]["A_bD_computation_status"] == "not_testable"
    assert result["summary"]["drift_alignment_failure_reason"] == "mnps_xdot_as_sde_drift"
    assert result["provenance"]["settings"]["drift_residualization"] == "none"
    assert np.all(np.isnan(result["series"]["A_bD"]))


def test_default_ingest_resolution_is_not_supplied() -> None:
    resolved = resolve_ingest_chart_drift({"enabled": True, "neighborhood": {"k": 20}})
    assert resolved.field is None
    assert resolved.source == "not_supplied"
    assert resolved.failure_reason == REASON_NOT_SUPPLIED
    assert resolved.residualize_increments is False


def test_common_family_yaml_keeps_drift_disabled() -> None:
    text = COMMON_FAMILIES.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    drift = parsed["dynamical_families"]["diffusion"]["drift"]
    assert drift["enabled"] is False
    assert drift["mode"] == "alignment_only"
    assert drift["source"] == "not_supplied"
    assert parsed["dynamical_families"]["enabled"] is False
    assert parsed["dynamical_families"]["diffusion"]["enabled"] is False
