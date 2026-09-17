"""Withheld 001 identities stay closed until they have their own license."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))

from mndm.dynamical_families import HISTORY_SCHEMA_VERSION, family_forbids
from mndm.dynamical_families.committor import estimate_committor
from mndm.dynamical_families.history import estimate_history_predictive_gain
from mndm.dynamical_families.measurement_register import (
    MEASUREMENT_ID_DESTINATION_FIRST_HIT_L0,
    MEASUREMENT_ID_DESTINATION_HIT_L1,
    MEASUREMENT_ID_DESTINATION_RESOLVED,
    MEASUREMENT_ID_DESTINATION_UNRESOLVED,
    MEASUREMENT_ID_GENERATOR_COMMITTOR,
    MEASUREMENT_ID_SPONTANEOUS_RETURN,
    MEASUREMENT_ID_EXCURSION_RECOVERY,
    MEASUREMENT_ID_FAR_P50,
    MEASUREMENT_ID_FAR_P90,
    MEASUREMENT_ID_COND_FUTURE_SPREAD,
    MEASUREMENT_ID_R50_DISCRETE,
    MEASUREMENT_ID_DIFFUSION_COND,
    MEASUREMENT_ID_DIFFUSION_DEFF,
    MEASUREMENT_ID_DIFFUSION_ENTROPY,
    MEASUREMENT_ID_GENERATOR_SYM_ANISO,
    MEASUREMENT_ID_HISTORY_GENERATOR,
    MEASUREMENT_ID_HISTORY_OPERATOR,
    MEASUREMENT_ID_HISTORY_PROPAGATOR,
    MEASUREMENT_ID_ITO_DIFFUSION,
    MEASUREMENT_ID_ITO_DRIFT,
    MEASUREMENT_ID_ITERATED_ONE_STEP_MAP,
    MEASUREMENT_ID_NUMERICAL_ABSCISSA,
    MEASUREMENT_ID_OPERATOR_GAIN_ANISO,
    MEASUREMENT_ID_OPERATOR_MAX_GAIN,
    MEASUREMENT_ID_PEAK_GAIN,
    MEASUREMENT_ID_REACH_ANISO_L4,
    MEASUREMENT_ID_REACH_DEFF_L4,
    MEASUREMENT_ID_REACTIVITY_GAP,
    MEASUREMENT_ID_SPECTRAL_ABSCISSA,
    MEASUREMENT_ID_TRANSITION_HIT_L4,
    RELATION_DERIVED_SCALAR,
    RELATION_DIAGNOSTICS,
    RELATION_DOCUMENTED_IDENTITY,
    RELATION_EXISTING_DATASET,
    RELATION_SERIES_ALIAS,
    RELATION_WITHHELD,
    compatibility_entry,
    register_entry,
)
from mndm.dynamical_families.one_step_operator import estimate_affine_one_step_family
from mndm.inferential_grain import GRAIN_BY_FAMILY_ID, attach_grain
from mndm.pipeline.dynamical_families_export import build_dynamical_families_export
from mndm.schema import MNPSPayload, normalize_payload


def _ar_trajectory(*, n: int = 240, dt: float = 0.25, seed: int = 4):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 3), dtype=float)
    x[0] = rng.normal(size=3)
    x[1] = 0.35 * x[0] + 0.04 * rng.normal(size=3)
    for t in range(1, n - 1):
        x[t + 1] = 0.35 * x[t] + 0.55 * x[t - 1] + 0.04 * rng.normal(size=3)
    time = np.arange(n, dtype=float) * dt
    return x.astype(np.float32), time


def _export_kwargs(state, time):
    return dict(
        state=state,
        time=time,
        stage=None,
        segment_id=np.zeros(time.size, dtype=np.int32),
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
    )


def test_register_keeps_withheld_identities_unwritten() -> None:
    for name in (
        MEASUREMENT_ID_HISTORY_GENERATOR,
        MEASUREMENT_ID_HISTORY_PROPAGATOR,
        MEASUREMENT_ID_ITO_DRIFT,
        MEASUREMENT_ID_ITO_DIFFUSION,
        MEASUREMENT_ID_PEAK_GAIN,
        MEASUREMENT_ID_REACTIVITY_GAP,
        MEASUREMENT_ID_DIFFUSION_DEFF,
        MEASUREMENT_ID_DIFFUSION_COND,
        MEASUREMENT_ID_DIFFUSION_ENTROPY,
        MEASUREMENT_ID_OPERATOR_GAIN_ANISO,
        MEASUREMENT_ID_GENERATOR_SYM_ANISO,
        MEASUREMENT_ID_REACH_DEFF_L4,
        MEASUREMENT_ID_REACH_ANISO_L4,
        MEASUREMENT_ID_DESTINATION_FIRST_HIT_L0,
        MEASUREMENT_ID_DESTINATION_HIT_L1,
        MEASUREMENT_ID_DESTINATION_UNRESOLVED,
        MEASUREMENT_ID_TRANSITION_HIT_L4,
        MEASUREMENT_ID_SPONTANEOUS_RETURN,
        MEASUREMENT_ID_EXCURSION_RECOVERY,
        MEASUREMENT_ID_FAR_P50,
        MEASUREMENT_ID_FAR_P90,
        MEASUREMENT_ID_COND_FUTURE_SPREAD,
    ):
        entry = register_entry(name)
        row = compatibility_entry(name)
        assert entry["physical_path"] is None
        assert entry["relation"] == RELATION_WITHHELD
        assert row["physical_path"] is None
        assert row["relation"] == RELATION_WITHHELD
    ito_status = compatibility_entry("ito_qualified")
    assert ito_status["relation"] == RELATION_WITHHELD
    assert ito_status["measurement_id"] is None
    generator = register_entry(MEASUREMENT_ID_HISTORY_GENERATOR)
    assert generator["not_logm_of_m1"] is True
    assert "ito" in str(generator["identity_relation"])
    propagator = register_entry(MEASUREMENT_ID_HISTORY_PROPAGATOR)
    assert propagator["not_m1_iteration"] is True
    peak = register_entry(MEASUREMENT_ID_PEAK_GAIN)
    assert peak["written_by_one_step_family"] is False
    iterated = register_entry(MEASUREMENT_ID_ITERATED_ONE_STEP_MAP)
    assert iterated["physical_path"] is not None
    assert iterated["relation"] != RELATION_WITHHELD
    gap = register_entry(MEASUREMENT_ID_REACTIVITY_GAP)
    assert gap["written_by_one_step_family"] is False
    assert gap["not_jacobian_metrics"] is True
    assert register_entry(MEASUREMENT_ID_SPECTRAL_ABSCISSA)["physical_path"] is not None
    assert register_entry(MEASUREMENT_ID_NUMERICAL_ABSCISSA)["physical_path"] is not None
    assert compatibility_entry("reactivity_gap")["relation"] == RELATION_EXISTING_DATASET
    assert compatibility_entry("reactivity_gap")["measurement_id"] is None
    assert compatibility_entry("d_diff")["relation"] == RELATION_DERIVED_SCALAR
    assert compatibility_entry("d_diff")["measurement_id"] != MEASUREMENT_ID_DIFFUSION_DEFF
    assert compatibility_entry("diffusion_effective_dimension")["relation"] == RELATION_SERIES_ALIAS
    assert compatibility_entry("d_eff")["relation"] == RELATION_DERIVED_SCALAR
    assert compatibility_entry("d_eff")["measurement_id"] != MEASUREMENT_ID_REACH_DEFF_L4
    assert compatibility_entry("c_1_q")["relation"] == RELATION_DERIVED_SCALAR
    assert register_entry(MEASUREMENT_ID_OPERATOR_MAX_GAIN)["physical_path"] is not None
    assert register_entry(MEASUREMENT_ID_OPERATOR_GAIN_ANISO)["physical_path"] is None
    assert register_entry(MEASUREMENT_ID_DESTINATION_HIT_L1)["not_resolved_first_hit"] is True
    assert register_entry(MEASUREMENT_ID_DESTINATION_UNRESOLVED)["not_resolved_outcome_encoding"] is True
    assert register_entry(MEASUREMENT_ID_DESTINATION_RESOLVED)["physical_path"] is not None
    assert compatibility_entry("resolved_first_hit_outcome")["relation"] == RELATION_EXISTING_DATASET
    assert compatibility_entry("resolved_first_hit_outcome")["measurement_id"] is None
    assert compatibility_entry("n_resolved_first_hit_outcomes")["relation"] == RELATION_DIAGNOSTICS
    assert register_entry(MEASUREMENT_ID_SPONTANEOUS_RETURN)["not_far"] is True
    assert register_entry(MEASUREMENT_ID_FAR_P50)["not_existing_r50"] is True
    assert register_entry(MEASUREMENT_ID_COND_FUTURE_SPREAD)["not_controllability"] is True
    assert compatibility_entry("crossfit_local_chart_b")["relation"] == RELATION_WITHHELD
    assert compatibility_entry("r50_discrete_first_bin_at_or_below_half")["relation"] == RELATION_DIAGNOSTICS
    assert compatibility_entry("r50_discrete_first_bin_at_or_below_half")["measurement_id"] == MEASUREMENT_ID_R50_DISCRETE
    assert compatibility_entry("w_q")["relation"] == RELATION_DOCUMENTED_IDENTITY
    assert compatibility_entry("w_q")["measurement_id"] != MEASUREMENT_ID_COND_FUTURE_SPREAD


def test_yaml_and_registry_forbid_license_closed_names() -> None:
    assert family_forbids("history", MEASUREMENT_ID_HISTORY_GENERATOR)
    assert family_forbids("history", MEASUREMENT_ID_HISTORY_PROPAGATOR)
    assert family_forbids("history", "logm_of_history_m1")
    assert family_forbids("history", "m1_iteration_as_history_propagator")
    assert family_forbids("history", "history_m1_as_ito_drift")
    assert family_forbids("drift", MEASUREMENT_ID_ITO_DRIFT)
    assert family_forbids("drift", "ito_qualified")
    assert family_forbids("diffusion", MEASUREMENT_ID_ITO_DIFFUSION)
    assert family_forbids("diffusion", "ito_qualified")
    assert family_forbids("one_step", MEASUREMENT_ID_ITO_DRIFT)
    assert family_forbids("one_step", "ito_qualified")
    assert family_forbids("one_step", MEASUREMENT_ID_PEAK_GAIN)
    assert family_forbids("one_step", "peak_gain_from_phi_powers")
    assert family_forbids("one_step", MEASUREMENT_ID_REACTIVITY_GAP)
    assert family_forbids("one_step", "abscissa_difference_as_reactivity_gap")
    assert family_forbids("one_step", "jacobian_reactivity_gap_as_level3")
    assert family_forbids("one_step", MEASUREMENT_ID_OPERATOR_GAIN_ANISO)
    assert family_forbids("one_step", MEASUREMENT_ID_GENERATOR_SYM_ANISO)
    assert family_forbids("diffusion", MEASUREMENT_ID_DIFFUSION_DEFF)
    assert family_forbids("diffusion", MEASUREMENT_ID_DIFFUSION_COND)
    assert family_forbids("diffusion", MEASUREMENT_ID_DIFFUSION_ENTROPY)
    assert family_forbids("diffusion", "d_diff_as_diffusion_effective_dimension_level1")
    assert family_forbids("spread", MEASUREMENT_ID_REACH_DEFF_L4)
    assert family_forbids("spread", MEASUREMENT_ID_REACH_ANISO_L4)
    assert family_forbids("spread", "d_eff_as_reachability_effective_dimension_level4")
    assert family_forbids("destination", MEASUREMENT_ID_DESTINATION_HIT_L1)
    assert family_forbids("destination", MEASUREMENT_ID_DESTINATION_UNRESOLVED)
    assert family_forbids("destination", MEASUREMENT_ID_DESTINATION_FIRST_HIT_L0)
    assert family_forbids("destination", MEASUREMENT_ID_TRANSITION_HIT_L4)
    assert family_forbids("destination", "resolved_q_as_destination_hit_probability_level1")
    assert family_forbids("destination", "unresolved_outcomes_as_destination_unresolved_fraction_level1")
    assert family_forbids("destination", "include_unresolved_in_first_hit_q")
    assert family_forbids("destination", MEASUREMENT_ID_GENERATOR_COMMITTOR)
    assert family_forbids("destination", "committor")
    assert family_forbids("destination", "transition_model_first_hit_probability_level2")
    assert family_forbids("resilience", MEASUREMENT_ID_SPONTANEOUS_RETURN)
    assert family_forbids("resilience", MEASUREMENT_ID_FAR_P50)
    assert family_forbids("resilience", "amplitude_curve_as_spontaneous_return")
    assert family_forbids("resilience", "r50_as_far_threshold_p50")
    assert family_forbids("spread", MEASUREMENT_ID_COND_FUTURE_SPREAD)
    assert family_forbids("spread", "w_q_as_conditional_future_spread_level1")
    assert family_forbids("persistence", "state_return_probability_level0")
    assert family_forbids("hysteresis", "return_distance_level0")
    assert family_forbids("drift", "crossfit_local_chart_b")
    assert family_forbids("drift", "source_crossfit_as_independent_b")
    state, time = _ar_trajectory(n=40)
    kwargs = _export_kwargs(state, time)
    closed = (
        ("history", MEASUREMENT_ID_HISTORY_GENERATOR),
        ("history", MEASUREMENT_ID_HISTORY_PROPAGATOR),
        ("history", "logm_of_history_m1"),
        ("history", "m1_iteration_as_history_propagator"),
        ("drift", MEASUREMENT_ID_ITO_DRIFT),
        ("drift", "ito_qualified"),
        ("diffusion", MEASUREMENT_ID_ITO_DIFFUSION),
        ("diffusion", "ito_qualified"),
        ("one_step", MEASUREMENT_ID_PEAK_GAIN),
        ("one_step", "peak_gain_from_phi_powers"),
        ("one_step", MEASUREMENT_ID_ITO_DRIFT),
        ("one_step", "ito_qualified"),
        ("one_step", MEASUREMENT_ID_REACTIVITY_GAP),
        ("one_step", "abscissa_difference_as_reactivity_gap"),
        ("one_step", MEASUREMENT_ID_OPERATOR_GAIN_ANISO),
        ("one_step", MEASUREMENT_ID_GENERATOR_SYM_ANISO),
        ("diffusion", MEASUREMENT_ID_DIFFUSION_DEFF),
        ("diffusion", "d_diff_as_diffusion_effective_dimension_level1"),
        ("diffusion", MEASUREMENT_ID_DIFFUSION_COND),
        ("diffusion", MEASUREMENT_ID_DIFFUSION_ENTROPY),
        ("destination", MEASUREMENT_ID_DESTINATION_HIT_L1),
        ("destination", MEASUREMENT_ID_DESTINATION_UNRESOLVED),
        ("destination", MEASUREMENT_ID_DESTINATION_FIRST_HIT_L0),
        ("destination", MEASUREMENT_ID_TRANSITION_HIT_L4),
        ("destination", "resolved_q_as_destination_hit_probability_level1"),
        ("destination", "include_unresolved_in_first_hit_q"),
        ("resilience", MEASUREMENT_ID_SPONTANEOUS_RETURN),
        ("resilience", MEASUREMENT_ID_EXCURSION_RECOVERY),
        ("resilience", MEASUREMENT_ID_FAR_P50),
        ("resilience", MEASUREMENT_ID_FAR_P90),
        ("resilience", "amplitude_curve_as_spontaneous_return"),
        ("resilience", "r50_as_far_threshold_p50"),
        ("drift", "crossfit_local_chart_b"),
        ("drift", "source_crossfit_as_independent_b"),
    )
    for family, key in closed:
        with pytest.raises(ValueError, match=key):
            build_dynamical_families_export(
                config={
                    "dynamical_families": {
                        "enabled": True,
                        family: {"enabled": True, key: True},
                    }
                },
                **kwargs,
            )
        with pytest.raises(ValueError, match=key):
            build_dynamical_families_export(
                config={
                    "dynamical_families": {
                        "enabled": False,
                        family: {"enabled": False, key: True},
                    }
                },
                **kwargs,
            )
    for key in (
        "local_dynamics",
        "possible_futures",
        "perturbation",
        MEASUREMENT_ID_COND_FUTURE_SPREAD,
        "w_q_as_conditional_future_spread_level1",
        "state_return_probability_level0",
        "return_distance_level0",
    ):
        with pytest.raises(ValueError, match=key):
            build_dynamical_families_export(
                config={
                    "dynamical_families": {
                        "enabled": True,
                        key: {"enabled": True},
                    }
                },
                **kwargs,
            )


def test_h5_boundary_refuses_nested_withheld_leaves() -> None:
    time = np.arange(4, dtype=float)
    state = np.zeros((4, 3), dtype=np.float32)
    withheld = (
        ("history", MEASUREMENT_ID_HISTORY_GENERATOR),
        ("history", MEASUREMENT_ID_HISTORY_PROPAGATOR),
        ("drift", MEASUREMENT_ID_ITO_DRIFT),
        ("diffusion", MEASUREMENT_ID_ITO_DIFFUSION),
        ("one_step", MEASUREMENT_ID_PEAK_GAIN),
        ("one_step", "ito_qualified"),
        ("one_step", MEASUREMENT_ID_REACTIVITY_GAP),
        ("one_step", MEASUREMENT_ID_OPERATOR_GAIN_ANISO),
        ("one_step", MEASUREMENT_ID_GENERATOR_SYM_ANISO),
        ("diffusion", MEASUREMENT_ID_DIFFUSION_DEFF),
        ("diffusion", MEASUREMENT_ID_DIFFUSION_COND),
        ("diffusion", MEASUREMENT_ID_DIFFUSION_ENTROPY),
        ("one_step", MEASUREMENT_ID_REACH_DEFF_L4),
        ("one_step", MEASUREMENT_ID_REACH_ANISO_L4),
        ("destination", MEASUREMENT_ID_DESTINATION_HIT_L1),
        ("destination", MEASUREMENT_ID_DESTINATION_UNRESOLVED),
        ("destination", MEASUREMENT_ID_DESTINATION_FIRST_HIT_L0),
        ("destination", MEASUREMENT_ID_TRANSITION_HIT_L4),
        ("resilience", MEASUREMENT_ID_SPONTANEOUS_RETURN),
        ("resilience", MEASUREMENT_ID_EXCURSION_RECOVERY),
        ("resilience", MEASUREMENT_ID_FAR_P50),
        ("resilience", MEASUREMENT_ID_FAR_P90),
        ("drift", "crossfit_local_chart_b"),
    )
    for family, key in withheld:
        with pytest.raises(ValueError, match=key):
            normalize_payload(
                MNPSPayload(
                    time=time,
                    x=state,
                    x_dot=np.zeros_like(state),
                    dynamical_families={
                        family: {
                            "schema_version": HISTORY_SCHEMA_VERSION,
                            "computation_status": "computed",
                            key: {"computation_status": "computed"},
                        }
                    },
                )
            )
    with pytest.raises(ValueError, match="must be a mapping"):
        normalize_payload(
            MNPSPayload(
                time=time,
                x=state,
                x_dot=np.zeros_like(state),
                dynamical_families={
                    "destination": [
                        {MEASUREMENT_ID_DESTINATION_HIT_L1: {"computation_status": "computed"}}
                    ]
                },
            )
        )
    with pytest.raises(ValueError, match=MEASUREMENT_ID_DESTINATION_HIT_L1):
        normalize_payload(
            MNPSPayload(
                time=time,
                x=state,
                x_dot=np.zeros_like(state),
                transition_residuals={
                    "schema_version": "mndm.transition_residuals.v1",
                    MEASUREMENT_ID_DESTINATION_HIT_L1: {"computation_status": "computed"},
                },
            )
        )
    with pytest.raises(ValueError, match=MEASUREMENT_ID_DESTINATION_UNRESOLVED):
        normalize_payload(
            MNPSPayload(
                time=time,
                x=state,
                x_dot=np.zeros_like(state),
                residual_covariance_proxy={
                    "notes": [{MEASUREMENT_ID_DESTINATION_UNRESOLVED: {"computation_status": "computed"}}],
                },
            )
        )
    with pytest.raises(ValueError, match="reactivity_gap"):
        normalize_payload(
            MNPSPayload(
                time=time,
                x=state,
                x_dot=np.zeros_like(state),
                dynamical_families={
                    "one_step": {
                        "schema_version": HISTORY_SCHEMA_VERSION,
                        "computation_status": "computed",
                        "reactivity_gap": {"computation_status": "computed"},
                    }
                },
            )
        )
    with pytest.raises(ValueError, match=MEASUREMENT_ID_REACTIVITY_GAP):
        normalize_payload(
            MNPSPayload(
                time=time,
                x=state,
                x_dot=np.zeros_like(state),
                dynamical_families={
                    "one_step": {
                        "schema_version": HISTORY_SCHEMA_VERSION,
                        "computation_status": "computed",
                        "notes": [{MEASUREMENT_ID_REACTIVITY_GAP: {"computation_status": "computed"}}],
                    }
                },
            )
        )
    with pytest.raises(ValueError, match=MEASUREMENT_ID_REACH_DEFF_L4):
        normalize_payload(
            MNPSPayload(
                time=time,
                x=state,
                x_dot=np.zeros_like(state),
                stochastic_reachability={
                    "schema_version": "mndm.stochastic_reachability.v1",
                    MEASUREMENT_ID_REACH_DEFF_L4: {"computation_status": "computed"},
                },
            )
        )
    with pytest.raises(ValueError, match=MEASUREMENT_ID_COND_FUTURE_SPREAD):
        normalize_payload(
            MNPSPayload(
                time=time,
                x=state,
                x_dot=np.zeros_like(state),
                stochastic_reachability={
                    "schema_version": "mndm.stochastic_reachability.v1",
                    MEASUREMENT_ID_COND_FUTURE_SPREAD: {"computation_status": "computed"},
                },
            )
        )
    with pytest.raises(ValueError, match=MEASUREMENT_ID_SPONTANEOUS_RETURN):
        normalize_payload(
            MNPSPayload(
                time=time,
                x=state,
                x_dot=np.zeros_like(state),
                dynamical_families={
                    "resilience": {
                        "schema_version": "mndm.finite_amplitude_resilience.v1",
                        "computation_status": "computed",
                        "notes": [
                            {MEASUREMENT_ID_SPONTANEOUS_RETURN: {"computation_status": "computed"}}
                        ],
                    }
                },
            )
        )
    ok = normalize_payload(
        MNPSPayload(
            time=time,
            x=state,
            x_dot=np.zeros_like(state),
            jacobian_derived_metrics={"series": {"reactivity_gap": np.zeros(4)}},
            stochastic_reachability={
                "d_eff": np.array([1.0], dtype=np.float32),
                "w_q": np.eye(3, dtype=np.float32),
            },
            dynamical_families={
                "destination": attach_grain(
                    {
                        "schema_version": "mndm.committor.v1",
                        "computation_status": "computed",
                        "qualification_status": "not_assessed",
                        "series": {
                            "q_A_to_B": np.zeros(4, dtype=np.float32),
                            "resolved_first_hit_outcome": np.array(
                                [0.0, 1.0, np.nan, 1.0], dtype=np.float32
                            ),
                        },
                        "summary": {"n_resolved_first_hit_outcomes": 3},
                    },
                    **GRAIN_BY_FAMILY_ID["destination"],
                ),
                "resilience": attach_grain(
                    {
                        "schema_version": "mndm.finite_amplitude_resilience.v1",
                        "computation_status": "computed",
                        "qualification_status": "not_assessed",
                        "amplitude_curve": [{"return_fraction": 0.5}],
                        "summary": {"r50_discrete_first_bin_at_or_below_half": 2.0},
                    },
                    **GRAIN_BY_FAMILY_ID["resilience"],
                ),
            },
        )
    )
    assert "reactivity_gap" in ok.jacobian_derived_metrics["series"]
    assert "d_eff" in ok.stochastic_reachability
    assert "w_q" in ok.stochastic_reachability
    dest_series = ok.dynamical_families["destination"]["series"]
    assert "resolved_first_hit_outcome" in dest_series
    assert MEASUREMENT_ID_DESTINATION_HIT_L1 not in ok.dynamical_families["destination"]
    assert MEASUREMENT_ID_DESTINATION_UNRESOLVED not in ok.dynamical_families["destination"]
    resilience = ok.dynamical_families["resilience"]
    assert "amplitude_curve" in resilience
    assert "r50_discrete_first_bin_at_or_below_half" in resilience["summary"]
    assert MEASUREMENT_ID_SPONTANEOUS_RETURN not in resilience
    assert MEASUREMENT_ID_FAR_P50 not in resilience
    assert MEASUREMENT_ID_COND_FUTURE_SPREAD not in ok.stochastic_reachability


def test_estimators_do_not_write_closed_rungs() -> None:
    history_src = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "mndm"
        / "dynamical_families"
        / "history.py"
    ).read_text(encoding="utf-8")
    assert "from scipy.linalg import logm" not in history_src
    assert "logm(" not in history_src
    assert "matrix_power(" not in history_src
    one_step_src = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "mndm"
        / "dynamical_families"
        / "one_step_operator.py"
    ).read_text(encoding="utf-8")
    assert "matrix_power(" not in one_step_src
    assert MEASUREMENT_ID_PEAK_GAIN not in one_step_src
    assert MEASUREMENT_ID_REACTIVITY_GAP not in one_step_src
    assert MEASUREMENT_ID_OPERATOR_GAIN_ANISO not in one_step_src
    assert MEASUREMENT_ID_GENERATOR_SYM_ANISO not in one_step_src
    diffusion_src = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "mndm"
        / "dynamical_families"
        / "diffusion_geometry.py"
    ).read_text(encoding="utf-8")
    assert MEASUREMENT_ID_DIFFUSION_DEFF not in diffusion_src
    assert MEASUREMENT_ID_DIFFUSION_COND not in diffusion_src
    assert MEASUREMENT_ID_DIFFUSION_ENTROPY not in diffusion_src
    assert '"diffusion_effective_dimension"' in diffusion_src
    state, time = _ar_trajectory(n=480)
    history = estimate_history_predictive_gain(
        state, time, coordinate_layer="coords_3d_subject_anchored"
    )
    assert history["computation_status"] == "computed"
    assert MEASUREMENT_ID_HISTORY_OPERATOR in history
    assert MEASUREMENT_ID_HISTORY_GENERATOR not in history
    assert MEASUREMENT_ID_HISTORY_PROPAGATOR not in history
    phi = np.asarray(history[MEASUREMENT_ID_HISTORY_OPERATOR]["series"]["phi_hat"])
    assert phi.shape[1:] == (3, 6)
    one_step = estimate_affine_one_step_family(
        state, time, coordinate_layer="coords_3d_subject_anchored"
    )
    assert MEASUREMENT_ID_ITERATED_ONE_STEP_MAP in one_step
    assert MEASUREMENT_ID_PEAK_GAIN not in one_step
    assert MEASUREMENT_ID_ITO_DRIFT not in one_step
    assert MEASUREMENT_ID_REACTIVITY_GAP not in one_step
    assert MEASUREMENT_ID_OPERATOR_GAIN_ANISO not in one_step
    assert MEASUREMENT_ID_GENERATOR_SYM_ANISO not in one_step
    assert "ito_qualified" not in one_step
    assert MEASUREMENT_ID_SPECTRAL_ABSCISSA in one_step
    assert MEASUREMENT_ID_NUMERICAL_ABSCISSA in one_step
    assert str(one_step.get("computation_status") or "") != "ito_qualified"
    assert str(one_step.get("qualification_status") or "") != "ito_qualified"
    committor_src = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "mndm"
        / "dynamical_families"
        / "committor.py"
    ).read_text(encoding="utf-8")
    assert MEASUREMENT_ID_DESTINATION_HIT_L1 not in committor_src
    assert MEASUREMENT_ID_DESTINATION_UNRESOLVED not in committor_src
    assert MEASUREMENT_ID_DESTINATION_FIRST_HIT_L0 not in committor_src
    assert MEASUREMENT_ID_TRANSITION_HIT_L4 not in committor_src
    assert "resolved_first_hit_outcome" in committor_src
    resilience_src = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "mndm"
        / "dynamical_families"
        / "resilience.py"
    ).read_text(encoding="utf-8")
    assert MEASUREMENT_ID_SPONTANEOUS_RETURN not in resilience_src
    assert MEASUREMENT_ID_FAR_P50 not in resilience_src
    assert MEASUREMENT_ID_FAR_P90 not in resilience_src
    assert MEASUREMENT_ID_COND_FUTURE_SPREAD not in resilience_src
    assert "amplitude_curve" in resilience_src
    assert "r50_discrete_first_bin_at_or_below_half" in resilience_src
    reach_src = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "mndm"
        / "dynamics"
        / "stochastic_reachability.py"
    ).read_text(encoding="utf-8")
    assert MEASUREMENT_ID_COND_FUTURE_SPREAD not in reach_src
    assert '"w_q"' in reach_src
    n = 80
    per = 20
    extra = per
    state = np.concatenate(
        [np.tile(np.linspace(-1.0, 1.0, per), n // per), np.linspace(-0.2, 0.2, extra)]
    )[:, None]
    labels = np.full(state.shape[0], -1, dtype=np.int8)
    labels[: n: per] = 0
    labels[per - 1 : n : per] = 1
    segment_id = np.concatenate(
        [np.repeat(np.arange(n // per), per), np.full(extra, n // per, dtype=np.int32)]
    )
    time = np.arange(state.shape[0], dtype=float) * 0.01
    first_hit = estimate_committor(
        state,
        time,
        labels,
        set_A=[0],
        set_B=[1],
        segment_id=segment_id,
        neighborhood_k=30,
        min_support=20,
        min_transition_segments=2,
        min_valid_fraction=0.01,
    )
    assert first_hit["measurement_id"] == MEASUREMENT_ID_DESTINATION_RESOLVED
    assert MEASUREMENT_ID_DESTINATION_HIT_L1 not in first_hit
    assert MEASUREMENT_ID_DESTINATION_UNRESOLVED not in first_hit
    assert MEASUREMENT_ID_DESTINATION_FIRST_HIT_L0 not in first_hit
    assert "resolved_first_hit_outcome" in first_hit["series"]
    assert "n_resolved_first_hit_outcomes" in first_hit["summary"]
