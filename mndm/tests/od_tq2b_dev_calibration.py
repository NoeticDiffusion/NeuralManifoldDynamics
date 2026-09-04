"""DEV calibration runner for the preregistered OD-TQ2b replay.

This runner deliberately writes only a DEV artifact. It does not consume the
held-out seed pool and does not set a production qualification flag.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))

from core.io.h5_writer import write_h5
from mndm.dynamical_families.committor import (
    estimate_committor_local_law_dense_grid_o2b,
)
from mndm.pipeline.dynamical_families_export import build_dynamical_families_export
from mndm.schema import MNPSPayload

from od_tq2b_fixture import (
    DT,
    GRID_RESOLUTION,
    TRUTH_SYSTEMS,
    X_A,
    X_B,
    X0,
    build_truth_trajectories,
    exact_committor,
    local_match_audit,
    monte_carlo_committor,
    truth_metrics,
)


DEV_SEEDS = tuple(range(9300, 9320))
HELD_OUT_SEEDS = (1121, 1132, 1143, 1154, 1165, 1176, 1187, 1198, 1209, 1220)
SUPPORT_LEVELS = (512, 1024, 1536, 2048)
HIGH_SUPPORT_PER_GRID = max(SUPPORT_LEVELS)
N_FIRST_PASSAGE = 256
MIN_TRANSITION_SEGMENTS = 20
QUALIFICATION_ID = "OD-TQ2b-DEV-calibration"
QUALIFICATION_HASH = "od-tq2b-dev-calibration-hash"


def _config(
    system_index: int,
    support_per_grid: int,
    *,
    qualification_id: str = QUALIFICATION_ID,
    qualification_hash: str = QUALIFICATION_HASH,
    grid_resolution: int = GRID_RESOLUTION,
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
                "diffusion_coefficient": TRUTH_SYSTEMS[system_index].diffusion_coefficient,
                "set_A": [0],
                "set_B": [1],
                "min_samples": 100,
                "min_support_per_grid": support_per_grid,
                "min_transition_segments": MIN_TRANSITION_SEGMENTS,
                "max_dt_relative_deviation": 0.05,
            },
        }
    }


def _with_support(trajectory: dict, support_per_grid: int) -> dict:
    """Keep a deterministic prefix of each local probe grid plus all FP paths."""
    if support_per_grid > HIGH_SUPPORT_PER_GRID:
        raise ValueError("support level exceeds the generated high-support fixture")
    segment_id = np.asarray(trajectory["segment_id"])
    first_passage_start = int(HIGH_SUPPORT_PER_GRID * GRID_RESOLUTION)
    local_segment_id = np.mod(segment_id, HIGH_SUPPORT_PER_GRID)
    keep = (segment_id >= first_passage_start) | (
        (segment_id < first_passage_start)
        & (local_segment_id < int(support_per_grid))
    )
    return {
        **trajectory,
        "state": np.asarray(trajectory["state"])[keep],
        "time": np.asarray(trajectory["time"])[keep],
        "reaction_coordinate": np.asarray(trajectory["reaction_coordinate"])[keep],
        "regime_labels": np.asarray(trajectory["regime_labels"])[keep],
        "segment_id": segment_id[keep],
        "n_segments": int(np.unique(segment_id[keep]).size),
    }


def _permute_complete_segments(trajectory: dict) -> dict:
    """Reverse complete contiguous segments without changing their contents."""
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
    order = np.arange(starts.size - 1, -1, -1)
    indices = np.concatenate(
        [np.arange(starts[index], ends[index]) for index in order]
    )
    return {
        **trajectory,
        "state": np.asarray(trajectory["state"])[indices],
        "time": np.asarray(trajectory["time"])[indices],
        "reaction_coordinate": np.asarray(trajectory["reaction_coordinate"])[indices],
        "regime_labels": np.asarray(trajectory["regime_labels"])[indices],
        "segment_id": segment_id[indices],
    }


def _estimate(
    system_index: int,
    trajectory: dict,
    support_per_grid: int,
) -> dict:
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
        min_transition_segments=MIN_TRANSITION_SEGMENTS,
    )
    return result


def _hdf5_q_grid(
    *,
    system_index: int,
    seed: int,
    trajectory: dict,
    support_per_grid: int,
    temporary_dir: Path,
    qualification_id: str = QUALIFICATION_ID,
    qualification_hash: str = QUALIFICATION_HASH,
) -> tuple[np.ndarray, dict]:
    export = build_dynamical_families_export(
        config=_config(
            system_index,
            support_per_grid,
            qualification_id=qualification_id,
            qualification_hash=qualification_hash,
            grid_resolution=GRID_RESOLUTION,
        ),
        state=np.asarray(trajectory["state"]),
        time=np.asarray(trajectory["time"]),
        stage=np.asarray(trajectory["regime_labels"]),
        segment_id=np.asarray(trajectory["segment_id"]),
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
        reaction_coordinate=np.asarray(trajectory["reaction_coordinate"]),
        reaction_coordinate_name="q_coordinate",
    )
    family = export["destination"]
    if family["computation_status"] != "computed":
        return np.asarray([], dtype=float), family
    payload = MNPSPayload(
        time=np.asarray(trajectory["time"]),
        x=np.asarray(trajectory["state"], dtype=np.float32),
        x_dot=np.zeros_like(np.asarray(trajectory["state"], dtype=np.float32)),
        stage=np.asarray(trajectory["regime_labels"]),
        dynamical_families=export,
    )
    output = write_h5(
        temporary_dir / f"{system_index}_{seed}_{support_per_grid}.h5",
        f"od_tq2b_dev_{system_index}_{seed}_{support_per_grid}",
        payload,
    )
    with h5py.File(output, "r") as handle:
        stored = handle["dynamical_families/destination/v1"]
        reopened_status = stored["computation_status"][()].decode()
        if reopened_status != family["computation_status"]:
            raise RuntimeError("HDF5 computation status differs from adapter payload")
        if reopened_status == "computed":
            trajectory_size = int(np.asarray(trajectory["state"]).shape[0])
            if stored.attrs.get("_schema_version") != "mndm.committor.v1":
                raise RuntimeError("HDF5 committor schema version is incorrect")
            required_series = {
                "q_A_to_B": (trajectory_size,),
                "q_hat": (trajectory_size,),
                "q_grid": (GRID_RESOLUTION,),
                "query_grid": (GRID_RESOLUTION,),
                "reaction_coordinate": (trajectory_size,),
                "resolved_first_hit_outcome": (trajectory_size,),
                "support_count": (trajectory_size,),
                "grid_support_count": (GRID_RESOLUTION,),
                "valid": (trajectory_size,),
                "drift_estimate_grid": (GRID_RESOLUTION,),
            }
            for key, expected_shape in required_series.items():
                if stored[f"series/{key}"].shape != expected_shape:
                    raise RuntimeError(f"HDF5 series grain mismatch for {key}")
            if not np.allclose(
                stored["series/q_grid"][:],
                np.asarray(family["series"]["q_grid"]),
                rtol=0.0,
                atol=1e-7,
            ):
                raise RuntimeError("HDF5 q_grid differs from adapter payload")
            if not np.allclose(
                stored["series/q_hat"][:],
                np.asarray(family["series"]["q_hat"]),
                rtol=0.0,
                atol=1e-7,
            ):
                raise RuntimeError("HDF5 q_hat differs from adapter payload")
            valid_values = np.unique(stored["series/valid"][:])
            if not np.all(np.isin(valid_values, [0, 1])):
                raise RuntimeError("HDF5 validity field contains non-binary values")
            if (
                stored["provenance/qualification_id"][()].decode()
                != qualification_id
                or stored["provenance/qualification_contract_hash"][()].decode()
                != qualification_hash
                or stored["provenance/validation_level"][()].decode()
                != "mndm_translation_validated"
                or stored["provenance/estimator"][()].decode()
                != "local_law_dense_grid_o2b"
                or stored["provenance/settings/reaction_coordinate_name"][()].decode()
                != "q_coordinate"
                or stored["provenance/settings/first_hit_label_semantics"][()].decode()
                != "one_terminal_A_or_B_label_per_independent_segment"
                or not np.array_equal(
                    stored["provenance/settings/set_A"][:],
                    np.asarray([0]),
                )
                or not np.array_equal(
                    stored["provenance/settings/set_B"][:],
                    np.asarray([1]),
                )
                or stored["provenance/settings/segment_id_supplied"][()] != 1
                or stored["provenance/settings/gap_policy"][()].decode()
                != "segment_id_breaks_increment_pairs"
            ):
                raise RuntimeError("HDF5 qualification provenance is not preserved")
        return np.asarray(stored["series/q_grid"][:], dtype=float), family


def _row(
    system_index: int,
    seed: int,
    trajectory: dict,
    support_per_grid: int,
    temporary_dir: Path,
    qualification_id: str = QUALIFICATION_ID,
    qualification_hash: str = QUALIFICATION_HASH,
) -> dict:
    system = TRUTH_SYSTEMS[system_index]
    estimate_grid, result = _hdf5_q_grid(
        system_index=system_index,
        seed=seed,
        trajectory=trajectory,
        support_per_grid=support_per_grid,
        temporary_dir=temporary_dir,
        qualification_id=qualification_id,
        qualification_hash=qualification_hash,
    )
    row: dict[str, object] = {
        "system_id": system.system_id,
        "seed": int(seed),
        "status": result["computation_status"],
        "failure_reason": result["failure_reason"],
        "support_per_grid": support_per_grid,
        "n_first_passage": N_FIRST_PASSAGE,
        "n_segments": int(trajectory["n_segments"]),
        "h5_replayed": result["computation_status"] == "computed",
        "h5_schema_checked": result["computation_status"] == "computed",
    }
    if result["computation_status"] != "computed":
        return row
    metrics = truth_metrics(system, estimate_grid)
    truth_grid = exact_committor(
        system,
        np.linspace(X_A, X_B, GRID_RESOLUTION, dtype=float),
    )
    row.update(metrics)
    row["q_min"] = float(np.min(estimate_grid))
    row["q_max"] = float(np.max(estimate_grid))
    row["q_left_boundary_abs_error"] = abs(float(estimate_grid[0]))
    row["q_right_boundary_abs_error"] = abs(float(estimate_grid[-1]) - 1.0)
    row["min_grid_step"] = float(np.min(np.diff(estimate_grid)))
    row["max_grid_step_error"] = float(
        np.max(np.abs(np.diff(estimate_grid) - np.diff(truth_grid)))
    )
    row["mc_reference"] = monte_carlo_committor(
        system,
        start=X0,
        seed=seed + 50_000,
        n_paths=256,
        max_steps=60_000,
    )
    return row


def _thresholds(
    rows: list[dict],
    *,
    qualified_support_per_grid: int,
    permutation_diffs: list[float],
) -> dict[str, float | int]:
    computed = [row for row in rows if row.get("status") == "computed"]
    if len(computed) != len(rows):
        raise RuntimeError("DEV contains non-computed adapter results")
    rmse = [float(row["rmse"]) for row in computed]
    mae = [float(row["mae"]) for row in computed]
    max_abs = [float(row["max_abs_error"]) for row in computed]
    monotonicity = [max(0.0, -float(row["min_grid_step"])) for row in computed]
    range_violations = [
        max(0.0, -float(row["q_min"]), float(row["q_max"]) - 1.0)
        for row in computed
    ]
    boundary_errors = [
        max(
            float(row["q_left_boundary_abs_error"]),
            float(row["q_right_boundary_abs_error"]),
        )
        for row in computed
    ]
    x0_errors = [
        abs(float(row["q_x0_estimate"]) - float(row["q_x0_truth"]))
        for row in computed
    ]

    by_system = {
        str(system.system_id): [
            row for row in computed if row["system_id"] == system.system_id
        ]
        for system in TRUTH_SYSTEMS
    }
    t0_rows = {
        int(row["seed"]): row for row in by_system[TRUTH_SYSTEMS[0].system_id]
    }
    t1_rows = {
        int(row["seed"]): row for row in by_system[TRUTH_SYSTEMS[1].system_id]
    }
    delta_truth = float(
        exact_committor(TRUTH_SYSTEMS[0], np.asarray([X0]))[0]
        - exact_committor(TRUTH_SYSTEMS[1], np.asarray([X0]))[0]
    )
    paired_seeds = sorted(set(t0_rows) & set(t1_rows))
    recovered_fractions = [
        (
            float(t0_rows[seed]["q_x0_estimate"])
            - float(t1_rows[seed]["q_x0_estimate"])
        )
        / delta_truth
        for seed in paired_seeds
        if delta_truth != 0.0
    ]
    positive_fractions = [value for value in recovered_fractions if value > 0.0]
    if not positive_fractions:
        raise RuntimeError("DEV did not recover a positive T0/T1 specificity fraction")

    mc_discrepancies = [
        abs(float(row["mc_reference"]["q_mc"]) - float(row["q_x0_truth"]))
        + 1.96 * float(row["mc_reference"]["q_mc_se"])
        for row in computed
        if int(row["mc_reference"]["n_resolved"]) > 0
    ]
    if not mc_discrepancies:
        raise RuntimeError("DEV has no resolved Monte-Carlo truth references")
    return {
        "qualified_min_support_per_grid": qualified_support_per_grid,
        "rmse_max": 1.25 * max(rmse),
        "mae_max": 1.25 * max(mae),
        "max_abs_error_max": 1.25 * max(max_abs),
        "monotonicity_negative_step_max": max(1e-6, 1.25 * max(monotonicity)),
        "q_range_violation_max": max(1e-6, 1.25 * max(range_violations)),
        "boundary_abs_error_max": max(1e-6, 1.25 * max(boundary_errors)),
        "x0_individual_abs_error_max": 1.25 * max(x0_errors),
        "remote_specificity_min_recovered_fraction": 0.80 * min(positive_fractions),
        "remote_specificity_requires_positive_sign": True,
        "segment_order_permutation_max_abs_difference": max(
            1e-6,
            1.25 * max(permutation_diffs),
        ),
        "quadrature_mc_max_abs_discrepancy": max(mc_discrepancies),
        "delta_truth_t0_minus_t1": delta_truth,
        "n_computed_rows": len(computed),
    }


def run_dev(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = local_match_audit()
    if max(audit.values()) > 1e-8:
        raise RuntimeError(f"T0/T1 protected local match failed: {audit}")

    support_sweep: list[dict] = []
    for system_index in range(len(TRUTH_SYSTEMS)):
        for seed in DEV_SEEDS:
            max_trajectory = build_truth_trajectories(
                TRUTH_SYSTEMS[system_index],
                seed=seed,
                support_per_grid=HIGH_SUPPORT_PER_GRID,
                n_first_passage=N_FIRST_PASSAGE,
            )
            for support_per_grid in SUPPORT_LEVELS:
                trajectory = _with_support(max_trajectory, support_per_grid)
                result = _estimate(system_index, trajectory, support_per_grid)
                support_sweep.append(
                    {
                        "system_id": TRUTH_SYSTEMS[system_index].system_id,
                        "seed": int(seed),
                        "support_per_grid": support_per_grid,
                        "status": result["computation_status"],
                        "failure_reason": result["failure_reason"],
                    }
                )
            print(
                f"support sweep complete: {TRUTH_SYSTEMS[system_index].system_id} seed={seed}",
                flush=True,
            )

    eligible_levels = [
        support
        for support in SUPPORT_LEVELS
        if all(
            row["status"] == "computed"
            for row in support_sweep
            if row["support_per_grid"] == support
        )
    ]
    if not eligible_levels:
        raise RuntimeError("No support level was eligible for every DEV row")
    qualified_support = min(eligible_levels)

    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="od_tq2b_dev_") as temporary:
        temporary_dir = Path(temporary)
        for system_index in range(len(TRUTH_SYSTEMS)):
            for seed in DEV_SEEDS:
                max_trajectory = build_truth_trajectories(
                    TRUTH_SYSTEMS[system_index],
                    seed=seed,
                    support_per_grid=HIGH_SUPPORT_PER_GRID,
                    n_first_passage=N_FIRST_PASSAGE,
                )
                trajectory = _with_support(max_trajectory, qualified_support)
                rows.append(
                    _row(
                        system_index,
                        seed,
                        trajectory,
                        qualified_support,
                        temporary_dir,
                    )
                )
                print(
                    f"HDF5 scoring complete: {TRUTH_SYSTEMS[system_index].system_id} seed={seed}",
                    flush=True,
                )

        permutation_diffs: list[float] = []
        for system_index in range(len(TRUTH_SYSTEMS)):
            trajectory = _with_support(
                build_truth_trajectories(
                    TRUTH_SYSTEMS[system_index],
                    seed=DEV_SEEDS[0],
                    support_per_grid=HIGH_SUPPORT_PER_GRID,
                    n_first_passage=N_FIRST_PASSAGE,
                ),
                qualified_support,
            )
            permuted = _permute_complete_segments(trajectory)
            original_grid, _ = _hdf5_q_grid(
                system_index=system_index,
                seed=DEV_SEEDS[0],
                trajectory=trajectory,
                support_per_grid=qualified_support,
                temporary_dir=temporary_dir,
            )
            permuted_grid, _ = _hdf5_q_grid(
                system_index=system_index,
                seed=DEV_SEEDS[0] + 100_000,
                trajectory=permuted,
                support_per_grid=qualified_support,
                temporary_dir=temporary_dir,
            )
            permutation_diffs.append(
                float(np.max(np.abs(original_grid - permuted_grid)))
            )

    artifact = {
        "stage": "OD-TQ2b-DEV",
        "preregistration": "project/orthagonal_axis/od_tq2b_preregistration.md",
        "dev_seeds": list(DEV_SEEDS),
        "held_out_seeds_reserved": list(HELD_OUT_SEEDS),
        "protocol": {
            "dt_sec": DT,
            "grid_resolution": GRID_RESOLUTION,
            "support_levels": list(SUPPORT_LEVELS),
            "qualified_support_per_grid": qualified_support,
            "n_first_passage": N_FIRST_PASSAGE,
            "min_transition_segments": MIN_TRANSITION_SEGMENTS,
            "threshold_scored_surface": "reopened_hdf5_q_grid",
        },
        "local_match_audit": audit,
        "support_sweep": support_sweep,
        "thresholds": _thresholds(
            rows,
            qualified_support_per_grid=qualified_support,
            permutation_diffs=permutation_diffs,
        ),
        "permutation_diffs": permutation_diffs,
        "rows": rows,
    }
    output = output_dir / "dev_calibration.json"
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    result_path = run_dev(
        project_root / "project" / "orthagonal_axis" / "results" / "od_tq2b_dev_calibration"
    )
    print(result_path)
