"""Clean held-out confirmation runner for OD-TQ2b.

This is a new confirmation run, not a replacement for the consumed
OD-TQ2b held-out artifact.  It reuses the frozen fixture, adapter, HDF5
round-trip helpers, negative controls, support level, and DEV thresholds
without modifying any of those artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))

from od_tq2b_dev_calibration import (
    DEV_SEEDS,
    HELD_OUT_SEEDS,
    HIGH_SUPPORT_PER_GRID,
    MIN_TRANSITION_SEGMENTS,
    N_FIRST_PASSAGE,
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
from od_tq2b_heldout_gate import _negative_controls, _permuted


# Frozen before this run.  The 11-stride mirrors the consumed OD-TQ2b pool
# while remaining well separated from the historical O2/O2b and C2 pools.
CONFIRMATION_SEEDS = (
    14121,
    14132,
    14143,
    14154,
    14165,
    14176,
    14187,
    14198,
    14209,
    14220,
)

CONFIRMATION_STAGE = "OD-TQ2b-R"
QUALIFICATION_ID = "OD-TQ2b-R-confirmation"
# The contract hash remains the consumed OD-TQ2b held-out hash: this run
# confirms the same contract rather than defining a new one.
QUALIFICATION_HASH = "od-tq2b-held-out-gate-hash"

KNOWN_HISTORICAL_SEEDS = frozenset(
    {
        # Discarded and preregistered Stage O2 pools.
        *range(5000, 5020),
        *range(5100, 5120),
        *range(521, 621, 11),
        # OD-TQ2b DEV and consumed held-out pools.
        *range(9300, 9320),
        *HELD_OUT_SEEDS,
        # C2-S0 seeds recorded in the available cross-repository handover.
        12001,
        12002,
        12003,
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes(project_root: Path) -> dict[str, str]:
    paths = {
        "preregistration": project_root
        / "project"
        / "orthagonal_axis"
        / "od_tq2b_preregistration.md",
        "dev_artifact": project_root
        / "project"
        / "orthagonal_axis"
        / "results"
        / "od_tq2b_dev_calibration"
        / "dev_calibration.json",
        "dev_calibration_runner": project_root
        / "mndm"
        / "tests"
        / "od_tq2b_dev_calibration.py",
        "consumed_heldout_runner": project_root
        / "mndm"
        / "tests"
        / "od_tq2b_heldout_gate.py",
        "truth_fixture": project_root / "mndm" / "tests" / "od_tq2b_fixture.py",
        "confirmation_runner": project_root
        / "mndm"
        / "tests"
        / "od_tq2b_confirmation_gate.py",
        "committor_adapter": project_root
        / "mndm"
        / "src"
        / "mndm"
        / "dynamical_families"
        / "committor.py",
        "pipeline_export": project_root
        / "mndm"
        / "src"
        / "mndm"
        / "pipeline"
        / "dynamical_families_export.py",
        "hdf5_writer": project_root
        / "core"
        / "src"
        / "core"
        / "io"
        / "h5_writer.py",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"Cannot freeze missing source artifacts: {missing}")
    return {name: _sha256(path) for name, path in paths.items()}


def _git_revision(project_root: Path) -> dict[str, object]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Unable to record the repository revision") from error
    return {"head": revision, "working_tree_dirty": dirty}


def _load_frozen_dev(project_root: Path) -> dict:
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
        raise RuntimeError("Consumed held-out seed reservation was modified")
    if set(artifact.get("dev_seeds", [])) != set(DEV_SEEDS):
        raise RuntimeError("DEV seed pool in the frozen artifact was modified")
    if int(artifact["protocol"]["qualified_support_per_grid"]) > HIGH_SUPPORT_PER_GRID:
        raise RuntimeError("Frozen support exceeds the generated fixture support")
    if int(artifact["protocol"]["grid_resolution"]) != GRID_RESOLUTION:
        raise RuntimeError("Frozen grid resolution differs from the fixture")
    if int(artifact["protocol"]["n_first_passage"]) != N_FIRST_PASSAGE:
        raise RuntimeError("Frozen first-passage count differs from the fixture")
    if int(artifact["protocol"]["min_transition_segments"]) != MIN_TRANSITION_SEGMENTS:
        raise RuntimeError("Frozen transition-segment gate differs from the fixture")
    return artifact


def _assert_seed_pool_is_clean(artifact: dict) -> None:
    confirmation = set(CONFIRMATION_SEEDS)
    if len(confirmation) != 10:
        raise RuntimeError("OD-TQ2b-R requires exactly 10 unique seeds")
    if confirmation & KNOWN_HISTORICAL_SEEDS:
        raise RuntimeError("Confirmation seed overlaps a known historical pool")
    if confirmation & set(artifact.get("dev_seeds", [])):
        raise RuntimeError("Confirmation seed overlaps DEV")
    if confirmation & set(artifact.get("held_out_seeds_reserved", [])):
        raise RuntimeError("Confirmation seed overlaps consumed held-out data")
    if confirmation != set(CONFIRMATION_SEEDS):
        raise RuntimeError("Confirmation seed declaration is not deterministic")


def run_confirmation(project_root: Path) -> Path:
    artifact = _load_frozen_dev(project_root)
    _assert_seed_pool_is_clean(artifact)
    thresholds = artifact["thresholds"]
    support_per_grid = int(artifact["protocol"]["qualified_support_per_grid"])

    rows: list[dict] = []
    permutation_diffs: list[float] = []
    negative_controls: dict[str, dict[str, object]] = {}
    with tempfile.TemporaryDirectory(prefix="od_tq2b_confirmation_") as temporary:
        temporary_dir = Path(temporary)
        for system_index, system in enumerate(TRUTH_SYSTEMS):
            for seed in CONFIRMATION_SEEDS:
                trajectory = _with_support(
                    build_truth_trajectories(
                        system,
                        seed=seed,
                        support_per_grid=HIGH_SUPPORT_PER_GRID,
                        n_first_passage=N_FIRST_PASSAGE,
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
                    seed=CONFIRMATION_SEEDS[0],
                    support_per_grid=HIGH_SUPPORT_PER_GRID,
                    n_first_passage=N_FIRST_PASSAGE,
                ),
                support_per_grid,
            )
            original_grid, _ = _hdf5_q_grid(
                system_index=system_index,
                seed=CONFIRMATION_SEEDS[0],
                trajectory=permutation_trajectory,
                support_per_grid=support_per_grid,
                temporary_dir=temporary_dir,
                qualification_id=QUALIFICATION_ID,
                qualification_hash=QUALIFICATION_HASH,
            )
            permuted_grid, _ = _hdf5_q_grid(
                system_index=system_index,
                seed=CONFIRMATION_SEEDS[0] + 100_000,
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
    for seed in CONFIRMATION_SEEDS:
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
        "stage": CONFIRMATION_STAGE,
        "preregistration": "project/orthagonal_axis/od_tq2b_preregistration.md",
        "dev_artifact": "project/orthagonal_axis/results/od_tq2b_dev_calibration/dev_calibration.json",
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "consumed_held_out_seeds_not_used": list(HELD_OUT_SEEDS),
        "dev_seeds_not_used": list(DEV_SEEDS),
        "qualification_id": QUALIFICATION_ID,
        "qualification_contract_hash": QUALIFICATION_HASH,
        "frozen_thresholds": thresholds,
        "support_per_grid": support_per_grid,
        "n_rows": len(rows),
        "source_hashes": _source_hashes(project_root),
        "git_revision": _git_revision(project_root),
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
        / "od_tq2b_confirmation_gate"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "confirmation_gate.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    print(run_confirmation(root))
