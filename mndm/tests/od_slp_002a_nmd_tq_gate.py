"""Run the NMD-local OD-SLP-002A-NMD-TQ qualification.

This runner uses only truth-known synthetic systems. It writes a compact JSON
audit and never writes BOAS data, HDF5, q arrays, or production qualification
metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mndm.dynamical_families.committor_coarse_grid_qualification import (
    CANDIDATE_GRIDS,
    DEFAULT_DT,
    DEFAULT_N_FIRST_PASSAGE,
    DEFAULT_N_LOCAL_PER_GRID_POINT,
    DEFAULT_T_MAX_SEC,
    MAX_DT_RELATIVE_DEVIATION,
    MIN_TRANSITION_SEGMENTS,
    SUPPORT_FLOOR,
    TRUTH_A,
    TRUTH_B,
    TRUTH_SYSTEMS,
    build_truth_trajectories,
    estimate_audit_only_coarse_grid,
    score_truth_grid,
)

DEV_SEEDS = tuple(range(15000, 15020))
CONFIRMATION_SEEDS = (
    15201,
    15212,
    15223,
    15234,
    15245,
    15256,
    15267,
    15278,
    15289,
    15300,
)
DEV_THRESHOLD_MULTIPLIER = 1.25
BOUNDARY_TOLERANCE = 1e-6
NONDECREASING_Q_TOLERANCE = 1e-6
EVALUATION_ORDER = (65, 33, 17, 9)


def _protocol_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "project"
        / "orthagonal_axis"
        / "od_slp_002a_nmd_tq.md"
    )


def _module_fingerprint() -> str:
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "mndm"
        / "dynamical_families"
        / "committor_coarse_grid_qualification.py"
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_one(system_index: int, seed: int, grid_resolution: int) -> dict[str, Any]:
    system = TRUTH_SYSTEMS[system_index]
    fixture = build_truth_trajectories(
        system,
        seed=seed,
        grid_resolution=grid_resolution,
        n_local_per_grid_point=DEFAULT_N_LOCAL_PER_GRID_POINT,
        n_first_passage=DEFAULT_N_FIRST_PASSAGE,
        dt=DEFAULT_DT,
        t_max_sec=DEFAULT_T_MAX_SEC,
    )
    estimate = estimate_audit_only_coarse_grid(
        fixture["state"],
        fixture["time"],
        fixture["regime_labels"],
        set_A=[0],
        set_B=[1],
        grid_min=TRUTH_A,
        grid_max=TRUTH_B,
        diffusion_coefficient=system.diffusion_coefficient,
        segment_id=fixture["segment_id"],
        grid_resolution=grid_resolution,
        min_support_per_grid=SUPPORT_FLOOR,
        min_transition_segments=MIN_TRANSITION_SEGMENTS,
        max_dt_relative_deviation=MAX_DT_RELATIVE_DEVIATION,
    )
    metrics = score_truth_grid(estimate, system)
    summary = estimate.get("summary", {})
    return {
        "truth_system_id": system.system_id,
        "seed": int(seed),
        "grid_resolution": int(grid_resolution),
        "computation_status": estimate.get("computation_status"),
        "failure_reason": estimate.get("failure_reason"),
        "metrics": {
            key: (
                float(value)
                if isinstance(value, (float, np.floating))
                else bool(value)
                if isinstance(value, (bool, np.bool_))
                else value
            )
            for key, value in metrics.items()
        },
        "min_support_per_grid": summary.get("min_support_per_grid"),
        "n_transition_segments": summary.get("n_transition_segments"),
        "nominal_dt_sec": summary.get("nominal_dt_sec"),
        "max_dt_relative_deviation": summary.get("max_dt_relative_deviation"),
        "first_passage_summary": fixture["first_passage_summary"],
        "appended_absorbing_boundary_rows": fixture[
            "appended_absorbing_boundary_rows"
        ],
    }


def _thresholds(dev_records: list[Mapping[str, Any]]) -> dict[str, float] | None:
    metric_names = ("rmse_q", "mae_q", "e_max")
    if not dev_records or any(
        record.get("computation_status") != "computed"
        or record.get("metrics", {}).get("truth_valid") is not True
        for record in dev_records
    ):
        return None
    thresholds: dict[str, float] = {}
    for name in metric_names:
        values = [
            float(record["metrics"][name])
            for record in dev_records
            if np.isfinite(float(record["metrics"][name]))
        ]
        if len(values) != len(dev_records):
            return None
        thresholds[name] = max(
            1e-6,
            DEV_THRESHOLD_MULTIPLIER * max(values),
        )
    return thresholds


def _confirmation_pass(
    record: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if record.get("computation_status") != "computed":
        reasons.append(str(record.get("failure_reason", "adapter_not_computed")))
        return "NOT_TESTABLE", reasons
    metrics = record["metrics"]
    if metrics.get("truth_valid") is not True:
        reasons.append(str(metrics.get("truth_failure_reason", "truth_reference_invalid")))
        return "NOT_TESTABLE", reasons
    for name in ("rmse_q", "mae_q", "e_max"):
        if not np.isfinite(float(metrics[name])):
            reasons.append(f"{name}_nonfinite")
            continue
        if float(metrics[name]) > float(thresholds[name]):
            reasons.append(f"{name}_threshold_exceeded")
    if any(reason.endswith("_nonfinite") for reason in reasons):
        return "NOT_TESTABLE", reasons
    if float(metrics["endpoint_abs_error"]) > BOUNDARY_TOLERANCE:
        reasons.append("boundary_condition_error")
    if float(metrics["q_range_min"]) < -BOUNDARY_TOLERANCE:
        reasons.append("q_below_zero")
    if float(metrics["q_range_max"]) > 1.0 + BOUNDARY_TOLERANCE:
        reasons.append("q_above_one")
    if float(metrics["min_q_difference"]) < -NONDECREASING_Q_TOLERANCE:
        reasons.append("q_direction_not_nondecreasing")
    if int(record.get("appended_absorbing_boundary_rows", 0)) != 0:
        reasons.append("absorbing_boundary_row_appended")
    return ("FAIL" if reasons else "PASS"), reasons


def _compact_records(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Retain diagnostics without serializing q arrays or full trajectories."""
    computed = [
        record
        for record in records
        if record.get("computation_status") == "computed"
        and "rmse_q" in record.get("metrics", {})
    ]
    return {
        "n_records": len(records),
        "n_computed": len(computed),
        "n_not_testable": len(records) - len(computed),
        "failure_reasons": sorted(
            {
                str(record.get("failure_reason"))
                for record in records
                if record.get("computation_status") != "computed"
            }
        ),
        "min_support": (
            min(
                int(record["min_support_per_grid"])
                for record in records
                if record.get("min_support_per_grid") is not None
            )
            if any(record.get("min_support_per_grid") is not None for record in records)
            else None
        ),
    }


def _candidate_audit(grid_resolution: int) -> dict[str, Any]:
    dev_records = [
        _run_one(system_index, seed, grid_resolution)
        for system_index in range(len(TRUTH_SYSTEMS))
        for seed in DEV_SEEDS
    ]
    thresholds = _thresholds(dev_records)
    if thresholds is None:
        return {
            "grid_resolution": int(grid_resolution),
            "candidate_status": "NOT_TESTABLE",
            "reason": "dev_truth_or_adapter_not_testable",
            "dev_thresholds": None,
            "dev_summary": _compact_records(dev_records),
            "confirmation_summary": {
                "n_records": 0,
                "n_failures": 0,
                "n_not_testable": 0,
                "failures": [],
            },
            "confirmation_records": [],
        }
    if not CONFIRMATION_SEEDS:
        return {
            "grid_resolution": int(grid_resolution),
            "candidate_status": "NOT_TESTABLE",
            "reason": "empty_confirmation_seed_pool",
            "dev_thresholds": thresholds,
            "dev_summary": _compact_records(dev_records),
            "confirmation_summary": {
                "n_records": 0,
                "n_failures": 0,
                "n_not_testable": 0,
                "failures": [],
            },
            "confirmation_records": [],
        }
    confirmation_records = [
        _run_one(system_index, seed, grid_resolution)
        for system_index in range(len(TRUTH_SYSTEMS))
        for seed in CONFIRMATION_SEEDS
    ]
    confirmation_results = [
        _confirmation_pass(record, thresholds)
        for record in confirmation_records
    ]
    failures = [
        {
            "truth_system_id": record["truth_system_id"],
            "seed": record["seed"],
            "status": status,
            "reasons": reasons,
        }
        for record, (status, reasons) in zip(
            confirmation_records,
            confirmation_results,
        )
        if status != "PASS"
    ]
    n_failures = sum(status == "FAIL" for status, _ in confirmation_results)
    n_not_testable = sum(
        status == "NOT_TESTABLE" for status, _ in confirmation_results
    )
    status = (
        "FAIL"
        if n_failures
        else "NOT_TESTABLE"
        if n_not_testable
        else "PASS"
    )
    computed_confirmation = [
        record
        for record in confirmation_records
        if "rmse_q" in record.get("metrics", {})
    ]
    return {
        "grid_resolution": int(grid_resolution),
        "candidate_status": status,
        "reason": None
        if status == "PASS"
        else "confirmation_truth_mismatch"
        if status == "FAIL"
        else "confirmation_not_testable",
        "dev_thresholds": thresholds,
        "dev_summary": {
            "n_records": len(dev_records),
            "max_rmse_q": max(float(record["metrics"]["rmse_q"]) for record in dev_records),
            "max_mae_q": max(float(record["metrics"]["mae_q"]) for record in dev_records),
            "max_e_max": max(float(record["metrics"]["e_max"]) for record in dev_records),
            "min_support": min(
                int(record["min_support_per_grid"]) for record in dev_records
            ),
        },
        "dev_records": dev_records,
        "confirmation_summary": {
            "n_records": len(confirmation_records),
            "n_failures": n_failures,
            "n_not_testable": n_not_testable,
            "failures": failures,
            "max_rmse_q": max(
                (float(record["metrics"]["rmse_q"]) for record in computed_confirmation),
                default=None,
            ),
            "max_mae_q": max(
                (float(record["metrics"]["mae_q"]) for record in computed_confirmation),
                default=None,
            ),
            "max_e_max": max(
                (float(record["metrics"]["e_max"]) for record in computed_confirmation),
                default=None,
            ),
        },
        "diagnostics": _compact_records(confirmation_records),
    }


def _compact_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in candidate.items()
        if key not in {"dev_records", "confirmation_records"}
    }


def run_gate(output_path: Path) -> dict[str, Any]:
    """Run the fixed local TQ sweep and write a q-free qualification audit."""
    if output_path.exists():
        raise FileExistsError(f"refusing_to_overwrite:{output_path}")
    protocol_path = _protocol_path()
    parent_protocol_path = protocol_path.with_name("od_slp_002_preregistration.md")
    replay_first = _run_one(0, DEV_SEEDS[0], EVALUATION_ORDER[0])
    replay_second = _run_one(0, DEV_SEEDS[0], EVALUATION_ORDER[0])
    replay_json_first = json.dumps(replay_first, sort_keys=True, allow_nan=False)
    replay_json_second = json.dumps(replay_second, sort_keys=True, allow_nan=False)
    candidate_results = {
        str(grid): _candidate_audit(grid)
        for grid in EVALUATION_ORDER
    }
    statuses = [
        candidate_results[str(grid)]["candidate_status"]
        for grid in CANDIDATE_GRIDS
    ]
    if "PASS" in statuses:
        overall = "PASS"
    elif "FAIL" in statuses:
        overall = "FAIL"
    else:
        overall = "NOT_TESTABLE"
    result = {
        "schema": "mndm.od_slp_002a_nmd_tq.v1",
        "protocol_id": "OD-SLP-002A-NMD-TQ",
        "protocol_sha256": _sha256(protocol_path),
        "parent_protocol_id": "OD-SLP-002",
        "parent_protocol_sha256": _sha256(parent_protocol_path),
        "truth_systems": [
            {
                "system_id": system.system_id,
                "sigma": float(system.sigma),
                "diffusion_coefficient": system.diffusion_coefficient,
                "drift_kind": system.drift_kind,
                "mu": float(system.mu),
                "boundaries": [TRUTH_A, TRUTH_B],
            }
            for system in TRUTH_SYSTEMS
        ],
        "seed_pools": {
            "dev": list(DEV_SEEDS),
            "confirmation": list(CONFIRMATION_SEEDS),
        },
        "trajectory_contract": {
            "dt_sec": DEFAULT_DT,
            "t_max_sec": DEFAULT_T_MAX_SEC,
            "n_local_per_grid_point": DEFAULT_N_LOCAL_PER_GRID_POINT,
            "n_first_passage": DEFAULT_N_FIRST_PASSAGE,
            "set_A": [0],
            "set_B": [1],
            "support_floor": SUPPORT_FLOOR,
            "min_transition_segments": MIN_TRANSITION_SEGMENTS,
            "max_dt_relative_deviation": MAX_DT_RELATIVE_DEVIATION,
            "appended_absorbing_boundary_rows": 0,
            "terminal_carrier_is_final_in_bounds_row": True,
        },
        "threshold_contract": {
            "dev_multiplier": DEV_THRESHOLD_MULTIPLIER,
            "boundary_tolerance": BOUNDARY_TOLERANCE,
            "nondecreasing_q_tolerance": NONDECREASING_Q_TOLERANCE,
            "truth_quadrature_epsabs": 1e-12,
            "truth_quadrature_epsrel": 1e-12,
        },
        "candidate_grids": list(CANDIDATE_GRIDS),
        "evaluation_order": list(EVALUATION_ORDER),
        "adapter_fingerprint": _module_fingerprint(),
        "reproducibility": {
            "rng_recipe": [
                "seed",
                "truth_system_index",
                "candidate_grid",
                "stream_id",
                "grid_index_or_replicate",
            ],
            "local_probe_stream_id": 10,
            "first_passage_stream_id": 20,
            "numpy_version": np.__version__,
            "python_version": __import__("sys").version,
        },
        "determinism": {
            "representative_grid": EVALUATION_ORDER[0],
            "representative_seed": DEV_SEEDS[0],
            "representative_truth_system_id": TRUTH_SYSTEMS[0].system_id,
            "repeated_record_identical": replay_json_first == replay_json_second,
        },
        "candidates": {
            grid: _compact_candidate(candidate)
            for grid, candidate in candidate_results.items()
        },
        "decision": {
            "overall_status": overall,
            "candidate_statuses": {
                grid: candidate["candidate_status"]
                for grid, candidate in candidate_results.items()
            },
            "eligible_grids": [
                grid
                for grid in EVALUATION_ORDER
                if candidate_results[str(grid)]["candidate_status"] == "PASS"
            ],
        },
        "fail_closed_assertions": {
            "production_o2b_guard_modified": False,
            "production_hdf5_written": False,
            "boas_data_read": False,
            "held_out_or_reserve_data_read": False,
            "q_arrays_serialized": False,
            "absorbing_boundary_rows_appended": False,
            "terminal_exit_increment_created": False,
            "legacy_estimator_used": False,
            "candidate_selection_used_boas_outcomes": False,
        },
        "claim_boundary": (
            "Local truth-known adapter qualification only. A candidate PASS "
            "supports later q-free BOAS support auditing; it does not "
            "establish empirical BOAS validity or production overlay fitness."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_gate(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": result["decision"]["overall_status"],
                "candidate_statuses": result["decision"]["candidate_statuses"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
