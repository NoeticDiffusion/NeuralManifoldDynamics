"""SL-LEV-MES-002 reachability ladder under 003 corrections.

State-matched future spread is level1. Existing W_Q is discrete Lyapunov
predictive spread, not empirical future covariance or controllability.
"""

from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))

from core.io.h5_writer import write_h5
from mndm.dynamics.stochastic_reachability import (
    compute_stochastic_reachability,
    compute_stochastic_reachability_from_gate_e,
)
from mndm.dynamical_families import WRITABLE_FAMILY_IDS, family_forbids, get_family
from mndm.dynamical_families.measurement_register import (
    FORBIDDEN_SPREAD_CONFIG_KEYS,
    LOGICAL_POSSIBLE_FUTURES_SPREAD,
    MEASUREMENT_ID_COND_FUTURE_SPREAD,
    MEASUREMENT_ID_FINITE_TIME_REACH,
    MEASUREMENT_ID_GENERATOR_SPREAD,
    MEASUREMENT_ID_OBS_FUTURE_DEFF,
    MEASUREMENT_ID_OBS_FUTURE_SPREAD,
    MEASUREMENT_ID_REACH_ANISO_L4,
    MEASUREMENT_ID_REACH_DEFF_L4,
    MEASUREMENT_ID_TRANSITION_REACH_COV,
    PHYSICAL_WQ_ROOT,
    QUALIFICATION_NOT_ASSESSED,
    RELATION_DERIVED_SCALAR,
    RELATION_DOCUMENTED_IDENTITY,
    RELATION_EXISTING_DATASET,
    RELATION_LOGICAL,
    RELATION_SUPERSEDED,
    RELATION_WITHHELD,
    compatibility_entry,
    register_entry,
)
from mndm.pipeline.dynamical_families_export import build_dynamical_families_export
from mndm.schema import MNPSPayload

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


def _q_contract(covariance: np.ndarray) -> dict[str, object]:
    return {
        "computation_status": "computed",
        "q_time_semantics": "one_step_transition_covariance",
        "q_units": "state_squared",
        "conversion_model": "not_applicable",
        "q_dt_sec": 1.0,
        "covariance": covariance,
        "schema_version": "test.q.v1",
    }


def test_state_matched_spread_is_level1_not_level0() -> None:
    heading = compatibility_entry(MEASUREMENT_ID_OBS_FUTURE_SPREAD)
    assert heading["relation"] == RELATION_SUPERSEDED
    assert heading["measurement_id"] == MEASUREMENT_ID_COND_FUTURE_SPREAD
    assert register_entry(MEASUREMENT_ID_OBS_FUTURE_SPREAD)["relation"] == RELATION_SUPERSEDED
    withheld = register_entry(MEASUREMENT_ID_COND_FUTURE_SPREAD)
    assert withheld["relation"] == RELATION_WITHHELD
    assert withheld["physical_path"] is None
    assert withheld["interpretation_level"] == 1
    assert withheld["estimand"] == "state_matched_future_endpoint_covariance"
    assert "not_empirical_future_covariance" not in withheld
    deff = compatibility_entry(MEASUREMENT_ID_OBS_FUTURE_DEFF)
    assert deff["relation"] == RELATION_SUPERSEDED
    assert compatibility_entry("d_eff")["relation"] == RELATION_DERIVED_SCALAR
    assert compatibility_entry("d_eff")["measurement_id"] == MEASUREMENT_ID_FINITE_TIME_REACH
    assert compatibility_entry("c_1_q")["relation"] == RELATION_DERIVED_SCALAR
    deff_l4 = compatibility_entry(MEASUREMENT_ID_REACH_DEFF_L4)
    assert deff_l4["relation"] == RELATION_WITHHELD
    assert deff_l4["physical_path"] is None
    assert deff_l4["measurement_id"] != compatibility_entry("d_eff")["measurement_id"]
    aniso_l4 = compatibility_entry(MEASUREMENT_ID_REACH_ANISO_L4)
    assert aniso_l4["relation"] == RELATION_WITHHELD
    assert aniso_l4["physical_path"] is None


def test_w_q_identity_depends_on_propagator_steps() -> None:
    q = np.eye(2, dtype=np.float64)
    one = compute_stochastic_reachability(
        [np.eye(2)], q, q_contract=_q_contract(q), epsilon=1e-8, precision="float64"
    )
    assert one["computation_status"] == "computed"
    assert one["n_propagator_steps"] == 1
    assert one["measurement_id"] == MEASUREMENT_ID_TRANSITION_REACH_COV
    assert one["interpretation_level"] == 2
    assert one["claim_class"] == "discrete_one_step"
    assert one["qualification_status"] == QUALIFICATION_NOT_ASSESSED
    assert one["summary"]["not_controllability"] is True
    assert one["summary"]["not_empirical_future_covariance"] is True
    assert MEASUREMENT_ID_OBS_FUTURE_SPREAD not in one
    assert MEASUREMENT_ID_GENERATOR_SPREAD not in one

    many = compute_stochastic_reachability(
        [np.eye(2), np.diag([2.0, 1.0])],
        q,
        q_contract=_q_contract(q),
        epsilon=1e-8,
        precision="float64",
    )
    assert many["computation_status"] == "computed"
    assert many["n_propagator_steps"] == 2
    assert many["measurement_id"] == MEASUREMENT_ID_FINITE_TIME_REACH
    assert many["interpretation_level"] == 4
    assert many["claim_class"] == "finite_horizon_propagation"
    assert many["qualification_status"] == QUALIFICATION_NOT_ASSESSED
    assert many["summary"]["not_generator_spread"] is True


def test_unavailable_w_q_keeps_level4_identity() -> None:
    result = compute_stochastic_reachability_from_gate_e(None, None)
    assert result["computation_status"] == "unavailable"
    assert result["measurement_id"] == MEASUREMENT_ID_FINITE_TIME_REACH
    assert result["interpretation_level"] == 4
    assert "w_q" not in result
    invalid = compute_stochastic_reachability(
        [],
        np.eye(2),
        q_contract=_q_contract(np.eye(2)),
    )
    assert invalid["computation_status"] == "invalid"
    assert invalid["measurement_id"] == MEASUREMENT_ID_FINITE_TIME_REACH
    assert "w_q" not in invalid


def test_w_q_identity_is_written_to_hdf5(tmp_path: Path) -> None:
    q = np.eye(2, dtype=np.float64)
    one = compute_stochastic_reachability(
        [np.eye(2)], q, q_contract=_q_contract(q), epsilon=1e-8, precision="float64"
    )
    many = compute_stochastic_reachability(
        [np.eye(2), np.eye(2)],
        q,
        q_contract=_q_contract(q),
        epsilon=1e-8,
        precision="float64",
    )
    payload_one = MNPSPayload(
        time=np.asarray([0.0, 1.0]),
        x=np.zeros((2, 3)),
        x_dot=np.zeros((2, 3)),
        stochastic_reachability=one,
    )
    out_one = write_h5(tmp_path / "reach_one.h5", "reach_one", payload_one)
    with h5py.File(out_one, "r") as handle:
        primary = handle["stochastic_reachability/v1/primary"]
        assert primary["measurement_id"][()].decode() == MEASUREMENT_ID_TRANSITION_REACH_COV
        assert int(primary["interpretation_level"][()]) == 2
        assert "observed_future_spread_level0" not in primary
        assert "possible_futures" not in handle
    payload_many = MNPSPayload(
        time=np.asarray([0.0, 1.0]),
        x=np.zeros((2, 3)),
        x_dot=np.zeros((2, 3)),
        stochastic_reachability=many,
    )
    out_many = write_h5(tmp_path / "reach_many.h5", "reach_many", payload_many)
    with h5py.File(out_many, "r") as handle:
        primary = handle["stochastic_reachability/v1/primary"]
        assert primary["measurement_id"][()].decode() == MEASUREMENT_ID_FINITE_TIME_REACH
        assert int(primary["interpretation_level"][()]) == 4


def test_reachability_compatibility_and_forbids() -> None:
    wq = compatibility_entry(PHYSICAL_WQ_ROOT)
    assert wq["relation"] == RELATION_EXISTING_DATASET
    assert wq["measurement_id"] == MEASUREMENT_ID_FINITE_TIME_REACH
    assert compatibility_entry("w_q")["relation"] == RELATION_DOCUMENTED_IDENTITY
    assert compatibility_entry(MEASUREMENT_ID_GENERATOR_SPREAD)["relation"] == RELATION_WITHHELD
    assert compatibility_entry(LOGICAL_POSSIBLE_FUTURES_SPREAD)["relation"] == RELATION_LOGICAL
    spread = get_family("spread")
    assert spread["implementation"] == "gate_closed"
    assert spread["gate"] == "F"
    assert "spread" not in WRITABLE_FAMILY_IDS
    assert family_forbids("spread", "observed_future_spread_as_level0")
    assert family_forbids("spread", "w_q_as_empirical_future_covariance")
    assert family_forbids("spread", "w_q_as_controllability")
    assert family_forbids("spread", "w_q_as_occupancy")
    assert family_forbids("spread", "w_q_as_generator_spread")
    assert family_forbids("spread", "multi_step_w_q_as_level2")
    assert family_forbids("spread", "one_step_w_q_as_level4")
    assert family_forbids("spread", MEASUREMENT_ID_REACH_DEFF_L4)
    assert family_forbids("spread", MEASUREMENT_ID_REACH_ANISO_L4)
    assert family_forbids("spread", "d_eff_as_reachability_effective_dimension_level4")
    assert family_forbids("spread", MEASUREMENT_ID_COND_FUTURE_SPREAD)
    assert family_forbids("spread", "w_q_as_conditional_future_spread_level1")
    for key in FORBIDDEN_SPREAD_CONFIG_KEYS:
        assert family_forbids("spread", key)
    wq_notes = compatibility_entry("w_q")["notes"]
    assert "level2" in wq_notes


def test_forbidden_spread_yaml_keys_are_refused() -> None:
    state = np.zeros((20, 3), dtype=float)
    time = np.arange(20, dtype=float) * 0.01
    kwargs = dict(
        state=state,
        time=time,
        stage=None,
        segment_id=np.zeros(time.size, dtype=np.int32),
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
    )
    with pytest.raises(ValueError, match="gate_closed"):
        build_dynamical_families_export(
            config={"dynamical_families": {"enabled": True, "spread": {"enabled": True}}},
            **kwargs,
        )
    for key in FORBIDDEN_SPREAD_CONFIG_KEYS:
        config = {"dynamical_families": {"enabled": True, key: {"enabled": True}}}
        with pytest.raises(ValueError, match=key):
            build_dynamical_families_export(config=config, **kwargs)


def test_overlays_do_not_flip_wq_or_add_002_spread_names() -> None:
    common = yaml.safe_load(COMMON_FAMILIES.read_text(encoding="utf-8"))
    root = common["dynamical_families"]
    assert root["enabled"] is True
    assert "spread" not in root
    for key in FORBIDDEN_SPREAD_CONFIG_KEYS:
        assert key not in root
    local = common.get("local_dynamics") or {}
    reach = local.get("stochastic_reachability") or {}
    assert reach.get("enabled", False) is True
    icare = yaml.safe_load(ICARE_FAMILIES.read_text(encoding="utf-8"))
    icare_root = icare["dynamical_families"]
    assert icare_root.get("enabled", False) is True
    assert "spread" not in icare_root
    for key in FORBIDDEN_SPREAD_CONFIG_KEYS:
        assert key not in icare_root
    icare_reach = (icare.get("local_dynamics") or {}).get("stochastic_reachability") or {}
    assert icare_reach.get("enabled", False) is True
    assert icare_root.get("resilience", {}).get("enabled", False) is True
