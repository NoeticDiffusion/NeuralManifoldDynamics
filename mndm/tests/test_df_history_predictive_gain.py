"""Frozen M0/M1 history predictive gain. Not Markov restoration."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))

from mndm.dynamical_families import (
    HISTORY_SCHEMA_VERSION,
    WRITABLE_FAMILY_IDS,
    family_forbids,
    get_family,
)
from mndm.dynamical_families.history import estimate_history_predictive_gain
from mndm.dynamical_families.measurement_register import (
    MEASUREMENT_ID_HISTORY_GENERATOR,
    MEASUREMENT_ID_HISTORY_OPERATOR,
    MEASUREMENT_ID_HISTORY_PREDICTIVE_GAIN,
    MEASUREMENT_ID_HISTORY_PROPAGATOR,
    QUALIFICATION_HISTORY_GAIN,
    QUALIFICATION_HISTORY_OPERATOR_IDENTIFIED,
    QUALIFICATION_HISTORY_OPERATOR_NOT_IDENTIFIED,
    QUALIFICATION_NOT_ASSESSED,
    RELATION_NEW_MEASURE,
    RELATION_WITHHELD,
    compatibility_entry,
    register_entry,
)
from mndm.pipeline.dynamical_families_export import build_dynamical_families_export
from mndm.pipeline.summary import _build_dynamical_families_export_for_layers
from mndm.schema import MNPSPayload


def _ar_trajectory(
    *,
    n: int = 480,
    dt: float = 0.25,
    a0: float = 0.35,
    a1: float = 0.55,
    noise: float = 0.04,
    seed: int = 4,
):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 3), dtype=float)
    x[0] = rng.normal(size=3)
    x[1] = a0 * x[0] + noise * rng.normal(size=3)
    for t in range(1, n - 1):
        x[t + 1] = a0 * x[t] + a1 * x[t - 1] + noise * rng.normal(size=3)
    time = np.arange(n, dtype=float) * dt
    return x.astype(np.float32), time


def _history_config(*, enabled: bool = True, **overrides) -> dict:
    block = {
        "enabled": enabled,
        "min_samples": 40,
        "min_triples": 20,
        "ridge_alpha": 1e-4,
        "n_blocks": 2,
        "embargo_steps": 4,
    }
    block.update(overrides)
    return {"dynamical_families": {"enabled": True, "history": block}}


def test_ar2_history_reduces_oos_error() -> None:
    state, time = _ar_trajectory(a0=0.3, a1=0.55)
    result = estimate_history_predictive_gain(
        state, time, coordinate_layer="coords_3d_subject_anchored"
    )
    assert result["computation_status"] == "computed"
    leaf = result[MEASUREMENT_ID_HISTORY_PREDICTIVE_GAIN]
    assert int(leaf["interpretation_level"]) == 1
    assert leaf["qualification_status"] == QUALIFICATION_HISTORY_GAIN
    assert leaf["summary"]["not_markov_restoration"] is True
    gain = float(leaf["summary"]["history_predictive_gain"])
    mse0 = float(leaf["summary"]["mse_m0"])
    mse1 = float(leaf["summary"]["mse_m1"])
    assert mse1 < 0.75 * mse0
    assert gain == pytest.approx(mse0 - mse1)
    assert int(leaf["summary"]["n_valid_folds"]) == 2
    assert int(leaf["summary"]["n_history_triples"]) < int(leaf["summary"]["n_lag1_pairs"])
    source_idx = np.asarray(leaf["series"]["source_idx"], dtype=np.int32)
    assert source_idx.size == int(leaf["summary"]["n_history_triples"])
    lag1_id = str(leaf["summary"]["lag1_transition_support_id"])
    assert lag1_id != str(leaf["summary"]["transition_support_id"])
    operator = result[MEASUREMENT_ID_HISTORY_OPERATOR]
    assert int(operator["interpretation_level"]) == 2
    assert operator["computation_status"] == "computed"
    assert operator["qualification_status"] == QUALIFICATION_HISTORY_OPERATOR_IDENTIFIED
    assert operator["summary"]["one_step_identified"] is True
    assert operator["summary"]["history_m1_not_lag1_phi"] is True
    assert operator["summary"]["not_markov_restoration"] is True
    assert float(operator["summary"]["rel_mse_baseline_median"]) < 0.9
    assert list(operator["summary"]["phi_shape"]) == [3, 6]
    phi = np.asarray(operator["series"]["phi_hat"])
    assert phi.shape[1:] == (3, 6)
    finite_phi = np.isfinite(phi).all(axis=(1, 2))
    assert int(np.sum(finite_phi)) == int(operator["summary"]["n_fitted_windows"])
    assert MEASUREMENT_ID_HISTORY_GENERATOR not in result
    assert MEASUREMENT_ID_HISTORY_PROPAGATOR not in result


def test_markov_ar1_gain_is_small() -> None:
    state, time = _ar_trajectory(a0=0.7, a1=0.0, seed=8)
    result = estimate_history_predictive_gain(
        state, time, coordinate_layer="coords_3d_subject_anchored"
    )
    mse0 = float(result["summary"]["mse_m0"])
    gain = float(result["summary"]["history_predictive_gain"])
    assert abs(gain) < 0.25 * mse0
    operator = result[MEASUREMENT_ID_HISTORY_OPERATOR]
    assert operator["summary"]["history_m1_not_lag1_phi"] is True
    if operator["computation_status"] == "computed":
        assert operator["qualification_status"] == QUALIFICATION_HISTORY_OPERATOR_IDENTIFIED
        assert float(operator["summary"]["rel_mse_baseline_median"]) < 0.9
    else:
        assert operator["qualification_status"] == QUALIFICATION_HISTORY_OPERATOR_NOT_IDENTIFIED


def test_iid_does_not_identify_m1_operator() -> None:
    rng = np.random.default_rng(11)
    state = rng.normal(size=(480, 3)).astype(np.float32)
    time = np.arange(480, dtype=float) * 0.25
    result = estimate_history_predictive_gain(
        state, time, coordinate_layer="coords_3d_subject_anchored"
    )
    assert result["computation_status"] == "computed"
    operator = result[MEASUREMENT_ID_HISTORY_OPERATOR]
    assert operator["computation_status"] == "insufficient_support"
    assert operator["failure_reason"] == "history_m1_fit_not_better_than_baseline"
    assert operator["qualification_status"] == QUALIFICATION_HISTORY_OPERATOR_NOT_IDENTIFIED
    assert operator["summary"]["one_step_identified"] is False
    assert float(operator["summary"]["rel_mse_baseline_median"]) >= 0.9
    assert not np.any(np.isfinite(operator["series"]["phi_hat"]))


def test_nine_d_and_other_block_count_are_not_testable() -> None:
    state, time = _ar_trajectory(n=80)
    nine = np.concatenate([state, state, state], axis=1)
    result = estimate_history_predictive_gain(nine, time, coordinate_layer="coords_9d")
    assert result["failure_reason"] == "history_subject_anchored_3d_only"
    other = estimate_history_predictive_gain(
        state, time, coordinate_layer="coords_3d_subject_anchored", n_blocks=3
    )
    assert other["failure_reason"] == "history_n_blocks_not_implemented"
    embargo = estimate_history_predictive_gain(
        state, time, coordinate_layer="coords_3d_subject_anchored", embargo_steps=0
    )
    assert embargo["failure_reason"] == "history_embargo_steps_not_implemented"
    negative = estimate_history_predictive_gain(
        state, time, coordinate_layer="coords_3d_subject_anchored", embargo_steps=-1
    )
    assert negative["failure_reason"] == "history_embargo_steps_not_implemented"
    assert MEASUREMENT_ID_HISTORY_OPERATOR in result
    assert result[MEASUREMENT_ID_HISTORY_OPERATOR]["qualification_status"] == QUALIFICATION_NOT_ASSESSED
    assert result[MEASUREMENT_ID_HISTORY_OPERATOR]["summary"].get("not_one_step_operator") is not True
    assert result[MEASUREMENT_ID_HISTORY_OPERATOR]["summary"]["history_m1_not_lag1_phi"] is True
    assert MEASUREMENT_ID_HISTORY_OPERATOR in other
    threshold = estimate_history_predictive_gain(
        state, time, coordinate_layer="coords_3d_subject_anchored", one_step_rel_mse_threshold=0.5
    )
    assert threshold["failure_reason"] == "history_rel_mse_threshold_not_implemented"
    assert MEASUREMENT_ID_HISTORY_OPERATOR in threshold


def test_yaml_forbids_and_common_default_off() -> None:
    import yaml

    state, time = _ar_trajectory(n=80)
    with pytest.raises(ValueError, match="history_conditioned_operator_level2"):
        build_dynamical_families_export(
            config={
                "dynamical_families": {
                    "enabled": True,
                    "history": {"enabled": True, "history_conditioned_operator_level2": True},
                }
            },
            state=state,
            time=time,
            stage=None,
            segment_id=np.zeros(time.size, dtype=np.int32),
            coordinate_layer="coords_3d_subject_anchored",
            coordinate_names=["m", "d", "e"],
        )
    with pytest.raises(ValueError, match="history_as_markov_restoration"):
        build_dynamical_families_export(
            config={
                "dynamical_families": {
                    "enabled": True,
                    "history": {"enabled": True, "history_as_markov_restoration": True},
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
    assert cfg["dynamical_families"]["history"]["enabled"] is True
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
        assert merged["dynamical_families"]["history"]["enabled"] is True

    for name in (
        "config_ingest_physionet_i-care_2_1_part1_0_12h_history_turning_pilot.yaml",
        "config_ingest_physionet_i-care_2_1_next_140_0_12h_history_turning_pilot.yaml",
    ):
        with (sources / name).open(encoding="utf-8") as handle:
            overlay = yaml.safe_load(handle)
        hist = (overlay.get("dynamical_families") or {}).get("history") or {}
        assert hist.get("enabled") is True
        merged = load_config(sources / name)
        families = merged["dynamical_families"]
        assert families["history"]["enabled"] is True
        assert families["turning"]["enabled"] is True
        assert families["one_step"]["enabled"] is True
        assert families["amplification"]["enabled"] is True


def test_registry_h5_and_higher_rungs_withheld(tmp_path: Path) -> None:
    assert "history" in WRITABLE_FAMILY_IDS
    assert family_forbids("history", "history_as_markov_restoration")
    assert family_forbids("history", "one_step_phi_as_history_m0")
    assert get_family("history")["schema"] == HISTORY_SCHEMA_VERSION
    assert register_entry(MEASUREMENT_ID_HISTORY_PREDICTIVE_GAIN)["written_by_amplification_family"] is False
    assert compatibility_entry(MEASUREMENT_ID_HISTORY_PREDICTIVE_GAIN)["relation"] == RELATION_NEW_MEASURE
    assert compatibility_entry(MEASUREMENT_ID_HISTORY_OPERATOR)["relation"] == RELATION_NEW_MEASURE
    assert register_entry(MEASUREMENT_ID_HISTORY_OPERATOR)["physical_path"] is not None
    assert register_entry(MEASUREMENT_ID_HISTORY_OPERATOR)["history_m1_not_lag1_phi"] is True
    for name in (
        MEASUREMENT_ID_HISTORY_GENERATOR,
        MEASUREMENT_ID_HISTORY_PROPAGATOR,
    ):
        assert compatibility_entry(name)["relation"] == RELATION_WITHHELD
        assert register_entry(name)["physical_path"] is None
    state, time = _ar_trajectory()
    export = build_dynamical_families_export(
        config=_history_config(),
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
    output = write_h5(tmp_path / "history.h5", "hist", payload)
    with h5py.File(output, "r") as handle:
        root = "/dynamical_families/history/v1"
        assert f"{root}/{MEASUREMENT_ID_HISTORY_PREDICTIVE_GAIN}" in handle
        assert f"{root}/{MEASUREMENT_ID_HISTORY_OPERATOR}" in handle
        assert f"{root}/{MEASUREMENT_ID_HISTORY_GENERATOR}" not in handle
        assert f"{root}/{MEASUREMENT_ID_HISTORY_PROPAGATOR}" not in handle
        group = handle[f"{root}/{MEASUREMENT_ID_HISTORY_PREDICTIVE_GAIN}"]
        assert group["qualification_status"][()].decode() == QUALIFICATION_HISTORY_GAIN
        assert int(group["interpretation_level"][()]) == 1
        operator = handle[f"{root}/{MEASUREMENT_ID_HISTORY_OPERATOR}"]
        assert operator["qualification_status"][()].decode() == QUALIFICATION_HISTORY_OPERATOR_IDENTIFIED
        assert int(operator["interpretation_level"][()]) == 2
        phi = np.asarray(operator["series"]["phi_hat"])
        assert phi.ndim == 3 and phi.shape[1:] == (3, 6)
        grain = handle[f"{root}/grain"]
        assert grain["native"][()].decode() == "recording"
        assert grain["repeated_measure"][()].decode() == "false"
        assert "/dynamical_families/amplification/v1/history_predictive_gain_level1" not in handle


def test_gap_keeps_identical_m0_m1_triples() -> None:
    state, time = _ar_trajectory(n=160)
    state[40] = np.nan
    result = estimate_history_predictive_gain(
        state, time, coordinate_layer="coords_3d_subject_anchored", min_samples=40
    )
    assert result["computation_status"] == "computed"
    n_lag1 = int(result["summary"]["n_lag1_pairs"])
    n_triples = int(result["summary"]["n_history_triples"])
    assert n_triples < n_lag1
    assert int(result["summary"]["n_valid_folds"]) == 2
    triple_idx = np.asarray(result["series"]["source_idx"], dtype=np.int32)
    assert triple_idx.size == n_triples
    lag1_sources = set()
    finite = np.all(np.isfinite(state), axis=1)
    for t in range(int(state.shape[0]) - 1):
        if finite[t] and finite[t + 1]:
            lag1_sources.add(t)
    assert all(int(src) - 1 in lag1_sources for src in triple_idx.tolist())
    assert all(int(src) in lag1_sources for src in triple_idx.tolist())


def test_one_usable_fold_is_insufficient() -> None:
    state, time = _ar_trajectory(n=26)
    result = estimate_history_predictive_gain(
        state,
        time,
        coordinate_layer="coords_3d_subject_anchored",
        min_samples=10,
        min_triples=2,
    )
    assert result["computation_status"] == "insufficient_support"
    assert result["failure_reason"] == "history_both_folds_required"
    assert int(result["summary"]["n_valid_folds"]) != 2
    assert MEASUREMENT_ID_HISTORY_OPERATOR in result
    assert result[MEASUREMENT_ID_HISTORY_OPERATOR]["qualification_status"] == QUALIFICATION_NOT_ASSESSED


def test_missing_coordinate_layer_keeps_both_leaves_and_refuses_aliases() -> None:
    export = _build_dynamical_families_export_for_layers(
        config={
            "dynamical_families": {
                "enabled": True,
                "coordinate_layer": "subject_anchored",
                "history": {"enabled": True},
            }
        },
        x_subject_anchored=None,
        x_cohort_anchored=None,
        time=np.arange(8, dtype=float) * 0.25,
        stage=None,
        segment_id=None,
    )
    history = export["history"]
    assert history["computation_status"] == "not_testable"
    assert history["failure_reason"] == "requested_coordinate_layer_not_available"
    assert MEASUREMENT_ID_HISTORY_PREDICTIVE_GAIN in history
    assert MEASUREMENT_ID_HISTORY_OPERATOR in history
    assert MEASUREMENT_ID_HISTORY_GENERATOR not in history
    assert MEASUREMENT_ID_HISTORY_PROPAGATOR not in history
    assert history[MEASUREMENT_ID_HISTORY_OPERATOR]["summary"].get("not_one_step_operator") is not True
    assert history[MEASUREMENT_ID_HISTORY_OPERATOR]["summary"]["history_m1_not_lag1_phi"] is True
    with pytest.raises(ValueError, match="history_conditioned_operator_level2"):
        _build_dynamical_families_export_for_layers(
            config={
                "dynamical_families": {
                    "enabled": True,
                    "coordinate_layer": "subject_anchored",
                    "history": {"enabled": True, "history_conditioned_operator_level2": True},
                }
            },
            x_subject_anchored=None,
            x_cohort_anchored=None,
            time=np.arange(8, dtype=float) * 0.25,
            stage=None,
            segment_id=None,
        )
