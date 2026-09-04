"""Held-out OD-TQ2b gate runner.

This runner consumes the frozen DEV artifact. It never estimates or changes
thresholds and never writes a production qualification configuration.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))

from mndm.pipeline.dynamical_families_export import build_dynamical_families_export

from od_tq2b_dev_calibration import (
    DEV_SEEDS,
    HELD_OUT_SEEDS,
    HIGH_SUPPORT_PER_GRID,
    MIN_TRANSITION_SEGMENTS,
    _config,
    _hdf5_q_grid,
    _row,
    _with_support,
)
from od_tq2b_fixture import (
    GRID_RESOLUTION,
    TRUTH_SYSTEMS,
    build_truth_trajectories,
    truth_metrics,
)


QUALIFICATION_ID = "OD-TQ2b-held-out-gate"
QUALIFICATION_HASH = "od-tq2b-held-out-gate-hash"


def _dev_artifact(project_root: Path) -> dict:
    path = (
        project_root
        / "project"
        / "orthagonal_axis"
        / "results"
        / "od_tq2b_dev_calibration"
        / "dev_calibration.json"
    )
    if not path.exists():
        raise RuntimeError("Frozen DEV calibration artifact is missing")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("stage") != "OD-TQ2b-DEV":
        raise RuntimeError("Unexpected DEV artifact stage")
    if set(artifact.get("held_out_seeds_reserved", [])) != set(HELD_OUT_SEEDS):
        raise RuntimeError("Held-out seed reservation does not match preregistration")
    return artifact


def _permuted(trajectory: dict) -> dict:
    segment_id = np.asarray(trajectory["segment_id"])
    starts = np.concatenate(
        [
            np.asarray([0], dtype=np.int32),
            np.flatnonzero(segment_id[1:] != segment_id[:-1]).astype(np.int32) + 1,
        ]
    )
    ends = np.concatenate(
        [starts[1:], np.asarray([segment_id.size], dtype=np.int32)]
    )
    indices = np.concatenate(
        [
            np.arange(starts[index], ends[index])
            for index in range(starts.size - 1, -1, -1)
        ]
    )
    return {
        **trajectory,
        "state": np.asarray(trajectory["state"])[indices],
        "time": np.asarray(trajectory["time"])[indices],
        "reaction_coordinate": np.asarray(trajectory["reaction_coordinate"])[indices],
        "regime_labels": np.asarray(trajectory["regime_labels"])[indices],
        "segment_id": segment_id[indices],
    }


def _negative_controls(
    trajectory: dict,
    *,
    system_index: int,
    support_per_grid: int,
) -> dict[str, dict[str, object]]:
    def export_result(
        current: dict,
        *,
        grid_resolution: int = GRID_RESOLUTION,
        min_support: int = support_per_grid,
    ) -> dict:
        config = _config(
            system_index,
            min_support,
            grid_resolution=grid_resolution,
        )
        return build_dynamical_families_export(
            config=config,
            state=current["state"],
            time=current["time"],
            stage=current["regime_labels"],
            segment_id=current["segment_id"],
            coordinate_layer="coords_3d_subject_anchored",
            coordinate_names=["m", "d", "e"],
            reaction_coordinate=current["reaction_coordinate"],
            reaction_coordinate_name="q_coordinate",
        )["destination"]

    coarse = export_result(trajectory, grid_resolution=33)
    under_result = export_result(trajectory, min_support=10_000)

    shuffled = trajectory.copy()
    shuffled["reaction_coordinate"] = np.asarray(
        trajectory["reaction_coordinate"]
    ).copy()
    rng = np.random.default_rng(70_000 + system_index)
    rng.shuffle(shuffled["reaction_coordinate"])
    shuffled_result = export_result(shuffled)
    shuffled_metrics = (
        truth_metrics(TRUTH_SYSTEMS[system_index], shuffled_result["series"]["q_grid"])
        if shuffled_result["computation_status"] == "computed"
        else {}
    )

    missing_qualification_config = _config(system_index, support_per_grid)
    missing_qualification_config["dynamical_families"]["destination"][
        "translation_qualification"
    ] = {"qualified": True}
    missing_qualification = build_dynamical_families_export(
        config=missing_qualification_config,
        state=trajectory["state"],
        time=trajectory["time"],
        stage=trajectory["regime_labels"],
        segment_id=trajectory["segment_id"],
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
        reaction_coordinate=trajectory["reaction_coordinate"],
        reaction_coordinate_name="q_coordinate",
    )

    leaky = trajectory.copy()
    leaky["segment_id"] = np.asarray(trajectory["segment_id"]).copy()
    leaky["segment_id"][
        np.isin(leaky["segment_id"], [0, 1])
    ] = 0
    leakage_result = export_result(leaky)

    irregular_time = np.asarray(trajectory["time"]).copy()
    irregular_time[1] += 0.001
    irregular = trajectory.copy()
    irregular["time"] = irregular_time
    irregular_result = export_result(irregular)

    missing_grid = _with_support(trajectory, support_per_grid)
    missing_ids = np.arange(HIGH_SUPPORT_PER_GRID, dtype=np.int32)
    missing_grid["segment_id"] = np.asarray(missing_grid["segment_id"]).copy()
    segment_values = np.asarray(missing_grid["segment_id"])
    labels = np.asarray(missing_grid["regime_labels"])
    first_passage_start = HIGH_SUPPORT_PER_GRID * GRID_RESOLUTION
    # NC7 removes all local support at one grid point and keeps only terminal
    # labels from the independent first-passage segments. This preserves the
    # A/B transition-count gate while removing every increment that could
    # refill the deliberately gappy grid point.
    missing_mask = (
        (segment_values < first_passage_start)
        & np.isin(segment_values, missing_ids)
    ) | (
        (segment_values >= first_passage_start)
        & ~np.isin(labels, [0, 1])
    )
    missing_grid["state"] = np.asarray(missing_grid["state"])[~missing_mask]
    missing_grid["time"] = np.asarray(missing_grid["time"])[~missing_mask]
    missing_grid["reaction_coordinate"] = np.asarray(
        missing_grid["reaction_coordinate"]
    )[~missing_mask]
    missing_grid["regime_labels"] = np.asarray(
        missing_grid["regime_labels"]
    )[~missing_mask]
    missing_grid["segment_id"] = np.asarray(missing_grid["segment_id"])[~missing_mask]
    missing_result = export_result(missing_grid)

    return {
        "NC1_coarse_grid": {
            "status": coarse["computation_status"],
            "failure_reason": coarse["failure_reason"],
        },
        "NC2_insufficient_support": {
            "status": under_result["computation_status"],
            "failure_reason": under_result["failure_reason"],
        },
        "NC3_shuffled_reaction_coordinate": {
            "status": shuffled_result["computation_status"],
            "failure_reason": shuffled_result["failure_reason"],
            "rmse": shuffled_metrics.get("rmse"),
        },
        "NC5_missing_qualification_metadata": {
            "status": missing_qualification["destination"]["computation_status"],
            "failure_reason": missing_qualification["destination"]["failure_reason"],
        },
        "NC4_segment_boundary_leakage": {
            "status": leakage_result["computation_status"],
            "failure_reason": leakage_result["failure_reason"],
        },
        "NC6_irregular_timestep": {
            "status": irregular_result["computation_status"],
            "failure_reason": irregular_result["failure_reason"],
        },
        "NC7_missing_grid_support": {
            "status": missing_result["computation_status"],
            "failure_reason": missing_result["failure_reason"],
        },
    }


def run_held_out(project_root: Path) -> Path:
    artifact = _dev_artifact(project_root)
    thresholds = artifact["thresholds"]
    support_per_grid = int(artifact["protocol"]["qualified_support_per_grid"])
    if support_per_grid > HIGH_SUPPORT_PER_GRID:
        raise RuntimeError("Frozen support exceeds held-out fixture support")

    rows: list[dict] = []
    permutation_diffs: list[float] = []
    negative_controls: dict[str, dict[str, object]] = {}
    with tempfile.TemporaryDirectory(prefix="od_tq2b_heldout_") as temporary:
        temporary_dir = Path(temporary)
        for system_index, system in enumerate(TRUTH_SYSTEMS):
            for seed in HELD_OUT_SEEDS:
                trajectory = _with_support(
                    build_truth_trajectories(
                        system,
                        seed=seed,
                        support_per_grid=HIGH_SUPPORT_PER_GRID,
                        n_first_passage=256,
                    ),
                    support_per_grid,
                )
                rows.append(
                    _row(
                        system_index,
                        seed,
                        trajectory,
                        support_per_grid,
                        temporary_dir,
                        qualification_id=QUALIFICATION_ID,
                        qualification_hash=QUALIFICATION_HASH,
                    )
                )

            permutation_trajectory = _with_support(
                build_truth_trajectories(
                    system,
                    seed=HELD_OUT_SEEDS[0],
                    support_per_grid=HIGH_SUPPORT_PER_GRID,
                    n_first_passage=256,
                ),
                support_per_grid,
            )
            original_grid, _ = _hdf5_q_grid(
                system_index=system_index,
                seed=HELD_OUT_SEEDS[0],
                trajectory=permutation_trajectory,
                support_per_grid=support_per_grid,
                temporary_dir=temporary_dir,
                qualification_id=QUALIFICATION_ID,
                qualification_hash=QUALIFICATION_HASH,
            )
            permuted_grid, _ = _hdf5_q_grid(
                system_index=system_index,
                seed=HELD_OUT_SEEDS[0] + 100_000,
                trajectory=_permuted(permutation_trajectory),
                support_per_grid=support_per_grid,
                temporary_dir=temporary_dir,
                qualification_id=QUALIFICATION_ID,
                qualification_hash=QUALIFICATION_HASH,
            )
            permutation_diffs.append(
                float(np.max(np.abs(original_grid - permuted_grid)))
            )

            if system_index == 0:
                negative_controls[system.system_id] = _negative_controls(
                    permutation_trajectory,
                    system_index=system_index,
                    support_per_grid=support_per_grid,
                )

    computed = [row for row in rows if row["status"] == "computed"]
    a_pass = bool(computed) and all(
        float(row["rmse"]) <= float(thresholds["rmse_max"])
        and float(row["mae"]) <= float(thresholds["mae_max"])
        and float(row["max_abs_error"]) <= float(thresholds["max_abs_error_max"])
        and max(
            float(row["q_left_boundary_abs_error"]),
            float(row["q_right_boundary_abs_error"]),
        )
        <= float(thresholds["boundary_abs_error_max"])
        and max(0.0, -float(row["q_min"]), float(row["q_max"]) - 1.0)
        <= float(thresholds["q_range_violation_max"])
        and max(0.0, -float(row["min_grid_step"]))
        <= float(thresholds["monotonicity_negative_step_max"])
        and abs(float(row["mc_reference"]["q_mc"]) - float(row["q_x0_truth"]))
        <= float(thresholds["quadrature_mc_max_abs_discrepancy"])
        for row in computed
    ) and len(computed) == len(rows)

    by_system_seed = {
        (str(row["system_id"]), int(row["seed"])): row for row in rows
    }
    delta_truth = float(thresholds["delta_truth_t0_minus_t1"])
    b_checks = []
    for seed in HELD_OUT_SEEDS:
        t0 = by_system_seed[("T0_symmetric", seed)]
        t1 = by_system_seed[("T1_remote_barrier", seed)]
        delta_hat = float(t0["q_x0_estimate"]) - float(t1["q_x0_estimate"])
        b_checks.append(
            {
                "seed": seed,
                "delta_hat": delta_hat,
                "recovered_fraction": delta_hat / delta_truth,
                "pass": (
                    delta_hat > 0.0
                    and delta_hat / delta_truth
                    >= float(thresholds["remote_specificity_min_recovered_fraction"])
                    and abs(float(t0["q_x0_estimate"]) - float(t0["q_x0_truth"]))
                    <= float(thresholds["x0_individual_abs_error_max"])
                    and abs(float(t1["q_x0_estimate"]) - float(t1["q_x0_truth"]))
                    <= float(thresholds["x0_individual_abs_error_max"])
                ),
            }
        )
    b_pass = all(check["pass"] for check in b_checks)

    c_pass = all(
        controls["NC1_coarse_grid"]["status"] == "not_testable"
        and controls["NC1_coarse_grid"]["failure_reason"]
        == "o2b_grid_resolution_below_minimum"
        and controls["NC2_insufficient_support"]["status"]
        == "insufficient_support"
        and (
            controls["NC3_shuffled_reaction_coordinate"]["status"] != "computed"
            or float(controls["NC3_shuffled_reaction_coordinate"]["rmse"])
            > float(thresholds["rmse_max"])
        )
        and controls["NC5_missing_qualification_metadata"]["failure_reason"]
        == "committor_qualification_metadata_required"
        for controls in negative_controls.values()
    )
    d_pass = bool(permutation_diffs) and max(permutation_diffs) <= float(
        thresholds["segment_order_permutation_max_abs_difference"]
    ) and all(
        controls["NC4_segment_boundary_leakage"]["status"] == "invalid"
        and controls["NC4_segment_boundary_leakage"]["failure_reason"]
        == "non_monotone_time_within_segment"
        and controls["NC6_irregular_timestep"]["status"] == "not_testable"
        and controls["NC6_irregular_timestep"]["failure_reason"]
        == "materially_irregular_increment_timestep"
        and controls["NC7_missing_grid_support"]["status"]
        == "insufficient_support"
        for controls in negative_controls.values()
    )
    e_pass = len(computed) == len(rows) and all(
        bool(row["h5_replayed"]) and bool(row["h5_schema_checked"])
        for row in rows
    )

    gate_components = {
        "TQ2b-A_curve_recovery": a_pass,
        "TQ2b-B_remote_specificity": b_pass,
        "TQ2b-C_refusal_semantics": c_pass,
        "TQ2b-D_segmentation_time": d_pass,
        "TQ2b-E_hdf5_provenance": e_pass,
    }
    gate_pass = all(gate_components.values())
    result = {
        "stage": "OD-TQ2b-held-out",
        "preregistration": "project/orthagonal_axis/od_tq2b_preregistration.md",
        "dev_artifact": "project/orthagonal_axis/results/od_tq2b_dev_calibration/dev_calibration.json",
        "held_out_seeds": list(HELD_OUT_SEEDS),
        "dev_seeds_not_used": list(DEV_SEEDS),
        "qualification_id": QUALIFICATION_ID,
        "qualification_contract_hash": QUALIFICATION_HASH,
        "frozen_thresholds": thresholds,
        "gate_components": gate_components,
        "verdict": "PASS" if gate_pass else "FAIL",
        "rows": rows,
        "remote_specificity_checks": b_checks,
        "permutation_diffs": permutation_diffs,
        "negative_controls": negative_controls,
    }
    output_dir = (
        project_root
        / "project"
        / "orthagonal_axis"
        / "results"
        / "od_tq2b_heldout_gate"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "heldout_gate.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    print(run_held_out(root))
