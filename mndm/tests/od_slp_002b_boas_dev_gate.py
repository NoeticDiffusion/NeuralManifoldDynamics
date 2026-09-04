"""Run the q-free OD-SLP-002B BOAS DEV support and endpoint audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from mndm.dynamical_families.sleep_committor_dev_audit import (
    EVALUATION_ORDER,
    FROZEN_PID_SPLIT,
    endpoint_census,
    q_free_support_audit,
    select_supported_grid,
)
from mndm.dynamical_families.sleep_committor_qualification import (
    ALL_OUTCOMES,
    assert_frozen_pid_split,
    canonical_pid,
)
from mndm.dynamical_families.sleep_first_hit_eligibility import (
    SleepFirstHitProtocol,
    json_safe,
)

try:
    from od_slp_000_first_hit_eligibility_gate import (  # type: ignore
        _load_pid_map,
        _pair_acquisitions,
    )
    from od_slp_001_empirical_qualification_gate import (  # type: ignore
        _all_segments,
        _clean_json_record,
        _load_night,
    )
except ImportError:  # pragma: no cover - package import fallback
    from mndm.tests.od_slp_000_first_hit_eligibility_gate import (  # type: ignore
        _load_pid_map,
        _pair_acquisitions,
    )
    from mndm.tests.od_slp_001_empirical_qualification_gate import (  # type: ignore
        _all_segments,
        _clean_json_record,
        _load_night,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "project"
        / "orthagonal_axis"
        / "od_slp_002b_boas_dev.md"
    )


def _tq_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "project"
        / "orthagonal_axis"
        / "results"
        / "od_slp_002a_nmd_tq"
        / "qualification.json"
    )


def _calculate_dev_geometry(
    dev_segments: list[dict[str, Any]],
) -> tuple[float | None, float | None, float | None, str | None]:
    if not dev_segments:
        return None, None, None, "no_dev_segments"
    values = [
        np.asarray(segment["reaction_coordinate"], dtype=float)
        for segment in dev_segments
        if np.asarray(segment.get("reaction_coordinate", []), dtype=float).size
    ]
    if not values:
        return None, None, None, "no_finite_dev_reaction_coordinate"
    all_rc = np.concatenate(values)
    all_rc = all_rc[np.isfinite(all_rc)]
    if all_rc.size == 0:
        return None, None, None, "no_finite_dev_reaction_coordinate"
    lower = float(np.percentile(all_rc, 2.5))
    upper = float(np.percentile(all_rc, 97.5))
    increment_variances: list[float] = []
    for segment in dev_segments:
        reaction = np.asarray(segment["reaction_coordinate"], dtype=float)
        time = np.asarray(segment["time"], dtype=float)
        if reaction.size < 2 or reaction.size != time.size:
            continue
        dt = np.diff(time)
        dr = np.diff(reaction)
        valid = np.isfinite(dt) & (dt > 0) & np.isfinite(dr)
        increment_variances.extend(((dr[valid] ** 2) / dt[valid]).tolist())
    if not increment_variances:
        return lower, upper, None, "no_dev_reaction_increments"
    a_dev = float(np.mean(increment_variances))
    if not np.isfinite(a_dev) or not np.isfinite(lower) or not np.isfinite(upper):
        return None, None, None, "nonfinite_dev_geometry"
    if lower >= upper or a_dev <= 0:
        return None, None, None, "invalid_dev_geometry"
    return lower, upper, a_dev, None


def _empty_endpoint(reason: str) -> dict[str, Any]:
    return {
        "status": "ENDPOINT_NOT_TESTABLE",
        "failure_reason": reason,
        "outcome_counts": {key: 0 for key in sorted(ALL_OUTCOMES)},
        "n_dev_all_first_outcomes": 0,
        "p_AB_DEV": None,
        "endpoint_failure_reasons": [],
    }


def _support_records(
    *,
    dev_segments: list[dict[str, Any]],
    tq_status_by_grid: dict[int, str],
    lower: float | None,
    upper: float | None,
    geometry_failure: str | None,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    support_by_grid: dict[int, dict[str, Any]] = {}
    for grid in EVALUATION_ORDER:
        if tq_status_by_grid.get(grid) != "PASS":
            support_by_grid[grid] = {
                "grid_resolution": grid,
                "status": "EXCLUDED_TQ",
                "failure_reason": "local_tq_candidate_not_pass",
                "query_grid": [],
                "support_count": [],
            }
        elif geometry_failure or lower is None or upper is None:
            support_by_grid[grid] = {
                "grid_resolution": grid,
                "status": "SUPPORT_NOT_TESTABLE",
                "failure_reason": geometry_failure or "dev_geometry_unavailable",
                "query_grid": [],
                "support_count": [],
            }
        else:
            support_by_grid[grid] = q_free_support_audit(
                dev_segments,
                lower=lower,
                upper=upper,
                grid_resolution=grid,
                dev_pids=FROZEN_PID_SPLIT["DEV"],
            )
    selection = select_supported_grid(tq_status_by_grid, support_by_grid)
    return support_by_grid, selection


def run_audit(*, raw_root: Path, output_path: Path) -> dict[str, Any]:
    """Run DEV-only q-free support/census audit and write JSON."""
    if output_path.exists():
        raise FileExistsError(f"refusing_to_overwrite:{output_path}")
    protocol_path = _protocol_path()
    tq_path = _tq_path()
    parent_path = protocol_path.with_name("od_slp_001_preregistration.md")
    if not tq_path.exists():
        tq_result: dict[str, Any] = {}
        tq_failure = "od_slp_002a_result_missing"
    else:
        tq_result = json.loads(tq_path.read_text(encoding="utf-8"))
        tq_failure = None
    tq_status_by_grid = {
        int(grid): str(status)
        for grid, status in tq_result.get("decision", {}).get(
            "candidate_statuses", {}
        ).items()
    }
    protocol = SleepFirstHitProtocol()
    participants = _load_pid_map(raw_root)
    split_check = assert_frozen_pid_split(
        [pid for pid in participants.values() if pid is not None]
    )
    pid_to_split = {
        str(pid): arm
        for arm, values in FROZEN_PID_SPLIT.items()
        for pid in values
    }
    dev_pid_strings = {str(pid) for pid in FROZEN_PID_SPLIT["DEV"]}

    # Pair discovery is path/identity metadata only. The DEV filter happens
    # before _load_night, which is the first operation that reads EDF data.
    pairs = _pair_acquisitions(raw_root)
    dev_pairs: list[dict[str, Any]] = []
    non_dev_pairs_not_loaded = 0
    for pair in pairs:
        try:
            pid = canonical_pid(participants.get(pair.get("participant_id")))
        except (TypeError, ValueError, OverflowError):
            non_dev_pairs_not_loaded += 1
            continue
        if pid in dev_pid_strings:
            dev_pairs.append(pair)
        else:
            non_dev_pairs_not_loaded += 1

    records = [
        _load_night(
            pair,
            pid_map=participants,
            pid_to_split=pid_to_split,
            protocol=protocol,
        )
        for pair in dev_pairs
    ]
    source_failures = Counter(
        reason
        for record in records
        for reason in record.get("failure_reasons", [])
    )
    pid_split_leakage = any(
        record.get("split") != "DEV"
        for record in records
    )
    dev_segments = _all_segments(records, "DEV")
    lower, upper, a_dev, geometry_failure = _calculate_dev_geometry(dev_segments)

    if (
        tq_failure
        or not split_check.get("matches_frozen_lists", False)
        or pid_split_leakage
    ):
        support_by_grid = {
            grid: {
                "grid_resolution": grid,
                "status": "SUPPORT_NOT_TESTABLE",
                "failure_reason": (
                    tq_failure
                    or split_check.get("reason")
                    or "pid_split_leakage"
                ),
                "query_grid": [],
                "support_count": [],
            }
            for grid in EVALUATION_ORDER
        }
        selection = select_supported_grid(
            {grid: "NOT_TESTABLE" for grid in EVALUATION_ORDER},
            support_by_grid,
        )
    else:
        support_by_grid, selection = _support_records(
            dev_segments=dev_segments,
            tq_status_by_grid=tq_status_by_grid,
            lower=lower,
            upper=upper,
            geometry_failure=geometry_failure,
        )

    endpoint = (
        _empty_endpoint(
            tq_failure
            or split_check.get("reason")
            or geometry_failure
            or "source_or_split_failure"
        )
        if tq_failure
        or not split_check.get("matches_frozen_lists", False)
        or pid_split_leakage
        or not dev_segments
        else endpoint_census(
            dev_segments,
            dev_pids=FROZEN_PID_SPLIT["DEV"],
        )
    )
    support_status = (
        "SUPPORT_PASS"
        if selection.get("selected_grid") is not None
        else "SUPPORT_NOT_TESTABLE"
    )
    endpoint_status = endpoint.get("status", "ENDPOINT_NOT_TESTABLE")
    if support_status != "SUPPORT_PASS" or endpoint_status == "ENDPOINT_NOT_TESTABLE":
        combined_status = "NOT_TESTABLE"
    elif endpoint_status == "ENDPOINT_INADEQUATE":
        combined_status = "METHOD_LIMITED / ENDPOINT_INADEQUATE"
    else:
        combined_status = "PASS"

    result = {
        "schema": "mndm.od_slp_002b_boas_dev.v1",
        "protocol_id": "OD-SLP-002B",
        "protocol_sha256": _sha256(protocol_path),
        "parent_protocol_id": "OD-SLP-001",
        "parent_protocol_sha256": _sha256(parent_path),
        "od_slp_002a_nmd_tq_record_sha256": (
            _sha256(tq_path) if tq_path.exists() else None
        ),
        "dataset": "ds005555 / BOAS",
        "source_root": str(raw_root),
        "source_inventory": {
            "participants_pid_count": len(
                [pid for pid in participants.values() if pid is not None]
            ),
            "frozen_split_matches": bool(
                split_check.get("matches_frozen_lists", False)
            ),
            "pair_paths_discovered": len(pairs),
            "dev_pair_paths_loaded": len(dev_pairs),
            "dev_records": len(records),
            "dev_valid_records": sum(
                not record.get("failure_reasons") for record in records
            ),
            "non_dev_pair_paths_not_loaded": non_dev_pairs_not_loaded,
            "source_failures": dict(sorted(source_failures.items())),
        },
        "frozen_contract": {
            "pid_split_id": split_check.get("split_id"),
            "dev_pid_count": len(FROZEN_PID_SPLIT["DEV"]),
            "set_A": [3],
            "set_B": [4],
            "interior_stage": 2,
            "support_floor": 64,
            "candidate_grids": [9, 17, 33, 65],
            "evaluation_order": list(EVALUATION_ORDER),
            "lower_boundary": lower,
            "upper_boundary": upper,
            "a_DEV": a_dev,
            "geometry_failure": geometry_failure,
            "endpoint_floor_contract": {
                "n3_events": 20,
                "rem_events": 20,
                "n3_pids": 8,
                "rem_pids": 8,
                "both_outcome_pids": 5,
                "top3_share_max": 0.50,
            },
        },
        "candidate_tq_statuses": {
            str(grid): tq_status_by_grid.get(grid, "NOT_TESTABLE")
            for grid in EVALUATION_ORDER
        },
        "candidate_support": {
            str(grid): support_by_grid[grid]
            for grid in EVALUATION_ORDER
        },
        "selection": selection,
        "endpoint_census": endpoint,
        "support_status": support_status,
        "endpoint_status": endpoint_status,
        "combined_status": combined_status,
        "software_versions": {
            "python": sys.version,
            "numpy": np.__version__,
        },
        "records": [_clean_json_record(record) for record in records],
        "fail_closed_assertions": {
            "q_computed": False,
            "q_arrays_serialized": False,
            "hdf5_written": False,
            "production_overlay_written": False,
            "standard_measurement_written": False,
            "held_out_signal_or_outcome_read": False,
            "reserve_signal_or_outcome_read": False,
            "non_dev_edf_loaded": False,
            "endpoint_outcomes_used_for_grid_selection": False,
            "p_ab_dev_used_for_grid_selection": False,
            "legacy_fallback_used": False,
            "terminal_exit_increment_created": False,
            "competing_or_censored_segments_dropped_from_census": False,
        },
        "claim_boundary": (
            "Audit-only BOAS DEV source-support and endpoint census. A PASS "
            "would support preregistering a later restricted N3-versus-REM "
            "pid-held-out qualification. It does not establish a committor, "
            "biological sleep-transition claim, held-out calibration, or "
            "production overlay."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(json_safe(result), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(raw_root=args.raw_root, output_path=args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "pair_paths_discovered": result["source_inventory"][
                    "pair_paths_discovered"
                ],
                "dev_pair_paths_loaded": result["source_inventory"][
                    "dev_pair_paths_loaded"
                ],
                "selected_grid": result["selection"]["selected_grid"],
                "support_status": result["support_status"],
                "endpoint_status": result["endpoint_status"],
                "status": result["combined_status"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
