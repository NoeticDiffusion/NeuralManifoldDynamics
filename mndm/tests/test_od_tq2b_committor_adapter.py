"""OD-TQ2b non-zero-drift bistable committor translation replay."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))

from core.io.h5_writer import write_h5
from mndm.dynamical_families.committor import (
    estimate_committor_local_law_dense_grid_o2b,
)
from mndm.dynamical_families.validity import increment_pairs
from mndm.pipeline.dynamical_families_export import build_dynamical_families_export
from mndm.schema import MNPSPayload

from od_tq2b_fixture import (
    DT,
    GRID_RESOLUTION,
    TRUTH_SYSTEMS,
    X_A,
    X_B,
    build_truth_trajectories,
    exact_committor,
    local_match_audit,
    monte_carlo_committor,
    truth_metrics,
)
from od_tq2b_dev_calibration import _with_support


def _config(
    *,
    diffusion_coefficient: float,
    qualification_id: str = "OD-TQ2b-dev-fixture",
    qualification_hash: str = "od-tq2b-dev-fixture-hash",
    grid_resolution: int = GRID_RESOLUTION,
    min_support_per_grid: int = 512,
    min_transition_segments: int = 20,
) -> dict:
    return {
        "dynamical_families": {
            "enabled": True,
            "destination": {
                "enabled": True,
                "regime_source": "explicit_first_hit_labels",
                "label_key": "stage",
                "reaction_coordinate": {
                    "source": "explicit_column",
                    "key": "q_coordinate",
                    "name": "q_coordinate",
                    "boundaries": [X_A, X_B],
                },
                "estimator": "local_law_dense_grid_o2b",
                "translation_qualification": {
                    "qualified": True,
                    "qualification_id": qualification_id,
                    "qualification_contract_hash": qualification_hash,
                },
                "grid_resolution": grid_resolution,
                "diffusion_coefficient": diffusion_coefficient,
                "set_A": [0],
                "set_B": [1],
                "min_samples": 100,
                "min_support_per_grid": min_support_per_grid,
                "min_transition_segments": min_transition_segments,
                "max_dt_relative_deviation": 0.05,
            },
        }
    }


def _estimate(system_index: int, *, support_per_grid: int = 512) -> tuple[dict, dict]:
    system = TRUTH_SYSTEMS[system_index]
    trajectory = build_truth_trajectories(
        system,
        seed=9300 + system_index,
        support_per_grid=support_per_grid,
        n_first_passage=64,
    )
    result = estimate_committor_local_law_dense_grid_o2b(
        trajectory["state"],
        trajectory["time"],
        trajectory["reaction_coordinate"],
        trajectory["regime_labels"],
        set_A=[0],
        set_B=[1],
        grid_min=X_A,
        grid_max=X_B,
        diffusion_coefficient=float(trajectory["diffusion_coefficient"]),
        segment_id=trajectory["segment_id"],
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
        reaction_coordinate_name="q_coordinate",
        grid_resolution=GRID_RESOLUTION,
        min_samples=100,
        min_support_per_grid=support_per_grid,
        min_transition_segments=20,
    )
    return trajectory, result


def test_od_tq2b_truth_systems_are_nonzero_drift_and_t1_local_match_is_exact() -> None:
    audit = local_match_audit()
    assert audit["drift_abs_diff"] < 1e-12
    assert audit["jacobian_abs_diff"] < 1e-8
    assert audit["diffusion_abs_diff"] < 1e-12

    assert any(abs(float(system.drift(0.25))) > 1e-6 for system in TRUTH_SYSTEMS)
    t0_truth = exact_committor(TRUTH_SYSTEMS[0], np.asarray([0.0]))[0]
    t1_truth = exact_committor(TRUTH_SYSTEMS[1], np.asarray([0.0]))[0]
    assert abs(float(t1_truth) - float(t0_truth)) > 0.05


def test_od_tq2b_recovers_nonzero_drift_bistable_committors() -> None:
    estimates: dict[str, np.ndarray] = {}
    for system_index, system in enumerate(TRUTH_SYSTEMS):
        trajectory, result = _estimate(system_index)
        assert result["computation_status"] == "computed"
        assert result["provenance"]["estimator"] == "local_law_dense_grid_o2b"
        assert result["provenance"]["settings"]["first_hit_label_semantics"] == (
            "one_terminal_A_or_B_label_per_independent_segment"
        )
        assert result["summary"]["n_segments_with_A_first_hit"] >= 1
        assert result["summary"]["n_segments_with_B_first_hit"] >= 1
        assert result["summary"]["n_independent_A_B_transition_segments"] >= 20
        assert result["series"]["grid_support_count"].shape == (GRID_RESOLUTION,)
        assert result["series"]["support_count"].shape == (
            trajectory["state"].shape[0],
        )

        metrics = truth_metrics(system, result["series"]["q_grid"])
        assert metrics["rmse"] < 0.35
        assert metrics["q_x0_estimate"] >= -1e-6
        assert metrics["q_x0_estimate"] <= 1.0 + 1e-6
        estimates[system.system_id] = np.asarray(result["series"]["q_grid"])

    t0_truth = float(exact_committor(TRUTH_SYSTEMS[0], np.asarray([0.0]))[0])
    t1_truth = float(exact_committor(TRUTH_SYSTEMS[1], np.asarray([0.0]))[0])
    delta_truth = t0_truth - t1_truth
    delta_estimate = (
        estimates[TRUTH_SYSTEMS[0].system_id][GRID_RESOLUTION // 2]
        - estimates[TRUTH_SYSTEMS[1].system_id][GRID_RESOLUTION // 2]
    )
    assert delta_truth * delta_estimate > 0.0
    assert abs(delta_estimate) >= 0.10 * abs(delta_truth)


def test_od_tq2b_truth_quadrature_agrees_with_independent_monte_carlo_reference() -> None:
    for system_index, system in enumerate(TRUTH_SYSTEMS[:2]):
        mc = monte_carlo_committor(
            system,
            seed=9500 + system_index,
            n_paths=128,
            max_steps=60_000,
        )
        q_truth = float(exact_committor(system, np.asarray([0.0]))[0])
        assert mc["n_resolved"] > 0
        assert abs(float(mc["q_mc"]) - q_truth) < 0.20


def test_od_tq2b_replays_adapter_payload_and_hdf5(tmp_path: Path) -> None:
    system = TRUTH_SYSTEMS[1]
    trajectory, _ = _estimate(1)
    config = _config(
        diffusion_coefficient=float(trajectory["diffusion_coefficient"]),
        qualification_id="OD-TQ2b-hdf5-fixture",
        qualification_hash="od-tq2b-hdf5-fixture-hash",
        min_support_per_grid=512,
    )
    export = build_dynamical_families_export(
        config=config,
        state=trajectory["state"],
        time=trajectory["time"],
        stage=trajectory["regime_labels"],
        segment_id=trajectory["segment_id"],
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
        reaction_coordinate=trajectory["reaction_coordinate"],
        reaction_coordinate_name="q_coordinate",
    )
    family = export["destination"]
    assert family["computation_status"] == "computed"
    assert family["provenance"]["validation_level"] == "mndm_translation_validated"

    payload = MNPSPayload(
        time=trajectory["time"],
        x=trajectory["state"].astype(np.float32),
        x_dot=np.zeros_like(trajectory["state"], dtype=np.float32),
        stage=trajectory["regime_labels"],
        dynamical_families=export,
    )
    output = write_h5(tmp_path / "od_tq2b.h5", "od_tq2b", payload)
    with h5py.File(output, "r") as handle:
        stored = handle["dynamical_families/destination/v1"]
        assert stored.attrs["_schema_version"] == "mndm.committor.v1"
        assert stored["computation_status"][()].decode() == "computed"
        assert stored["series/q_grid"].shape == (GRID_RESOLUTION,)
        assert stored["series/q_hat"].shape == (trajectory["state"].shape[0],)
        assert stored["series/grid_support_count"].shape == (GRID_RESOLUTION,)
        assert stored["series/support_count"].shape == (trajectory["state"].shape[0],)
        assert (
            stored["provenance/qualification_id"][()].decode()
            == "OD-TQ2b-hdf5-fixture"
        )
        assert (
            stored["provenance/qualification_contract_hash"][()].decode()
            == "od-tq2b-hdf5-fixture-hash"
        )
        assert (
            stored["provenance/settings/first_hit_label_semantics"][()].decode()
            == "one_terminal_A_or_B_label_per_independent_segment"
        )
        assert stored["provenance/settings/n_segments"][()] == trajectory["n_segments"]
        assert stored["summary/n_segments_with_A_first_hit"][()] >= 1
        assert stored["summary/n_segments_with_B_first_hit"][()] >= 1
        assert "jacobian/derived_metrics" not in handle


def test_od_tq2b_does_not_cross_segment_boundaries() -> None:
    trajectory, _ = _estimate(0, support_per_grid=512)
    segment_ids = np.asarray(trajectory["segment_id"])
    first_two_segments = np.flatnonzero(np.isin(segment_ids, [0, 1]))[:4]
    source_idx, _, _ = increment_pairs(
        trajectory["state"][first_two_segments],
        trajectory["time"][first_two_segments],
        segment_ids[first_two_segments],
        max_gap_sec=None,
    )
    # Each two-row probe contributes one increment. The row between probes is
    # a segment boundary and must never become a third increment.
    assert source_idx.tolist() == [0, 2]

    leaky_segment_ids = np.zeros(first_two_segments.size, dtype=np.int32)
    leaky = estimate_committor_local_law_dense_grid_o2b(
        trajectory["state"][first_two_segments],
        trajectory["time"][first_two_segments],
        trajectory["reaction_coordinate"][first_two_segments],
        trajectory["regime_labels"][first_two_segments],
        set_A=[0],
        set_B=[1],
        grid_min=X_A,
        grid_max=X_B,
        diffusion_coefficient=TRUTH_SYSTEMS[0].diffusion_coefficient,
        segment_id=leaky_segment_ids,
        min_samples=2,
        min_support_per_grid=2,
    )
    assert leaky["computation_status"] == "invalid"
    assert leaky["failure_reason"] == "non_monotone_time_within_segment"


def test_od_tq2b_support_downselection_is_per_grid_point() -> None:
    trajectory = build_truth_trajectories(
        TRUTH_SYSTEMS[0],
        seed=9300,
        support_per_grid=2_048,
        n_first_passage=20,
    )
    reduced = _with_support(trajectory, 512)
    local_segment_ids = np.asarray(reduced["segment_id"])
    local_segment_ids = local_segment_ids[
        local_segment_ids < 2_048 * GRID_RESOLUTION
    ]
    counts = np.bincount(
        local_segment_ids // 2_048,
        minlength=GRID_RESOLUTION,
    )
    assert counts.shape == (GRID_RESOLUTION,)
    assert np.all(counts == 2 * 512)
