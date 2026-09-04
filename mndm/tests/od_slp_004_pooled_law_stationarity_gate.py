"""Run the DEV-only OD-SLP-004 pooled-law stationarity audit.

The runner fits exactly one pooled DEV G=17 audit-only q-grid, replays it
against the archived OD-SLP-003 pooled payload, and scores that fixed grid on
pooled/early/late DEV rows. It never opens HELD_OUT or RESERVE and never
writes a held-out qualification, HDF5, or production overlay.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from mndm.dynamical_families.sleep_committor_dev_manifest import (  # noqa: E402
    A_DEV,
    CANONICAL_SOURCE_ROOT,
    FROZEN_PID_SPLIT,
    GRID_RESOLUTION,
    LOWER_BOUNDARY,
    MAX_DT_RELATIVE_DEVIATION,
    MIN_TRANSITION_SEGMENTS,
    N3_DEV,
    P_REM_DEV,
    SUPPORT_FLOOR,
    UPPER_BOUNDARY,
    dev_base_rate_brier,
    calculate_transfer_tolerance,
    frozen_query_grid,
    sha256_file,
    split_digest,
    verify_frozen_geometry,
    verify_prerequisite_hashes,
    canonical_source_root_matches,
    fit_dev_grid,
)
from mndm.dynamical_families.sleep_committor_stationarity_audit import (  # noqa: E402
    EVALUATION_STRATA,
    _empty_score_payload,
    score_fixed_law_surface,
    stationarity_status,
    transfer_payload,
    compare_archived_pooled_fit,
)
from mndm.dynamical_families.sleep_committor_qualification import (  # noqa: E402
    FROZEN_PID_SPLIT,
    assert_frozen_pid_split,
    canonical_pid,
)
from mndm.dynamical_families.sleep_first_hit_eligibility import (  # noqa: E402
    SleepFirstHitProtocol,
    json_safe,
)

try:
    from od_slp_003_dev_manifest_gate import (  # type: ignore
        _all_segments,
        _clean_json_record,
        _load_night,
        _load_pid_map,
        _pair_acquisitions,
    )
except ImportError:  # pragma: no cover - package import fallback
    from mndm.tests.od_slp_001_empirical_qualification_gate import (  # type: ignore
        _all_segments,
        _clean_json_record,
        _load_night,
    )
    from mndm.tests.od_slp_000_first_hit_eligibility_gate import (  # type: ignore
        _load_pid_map,
        _pair_acquisitions,
    )


SCHEMA = "mndm.od_slp_004_dev_stationarity.v1"
PROTOCOL_ID = "OD-SLP-004"
PROTOCOL_PATH = (
    Path("project")
    / "orthagonal_axis"
    / "od_slp_004_pooled_law_stationarity_preregistration.md"
)
OD003_PROTOCOL_PATH = (
    Path("project")
    / "orthagonal_axis"
    / "od_slp_003_heldout_preregistration.md"
)
OD003_MANIFEST_PATH = (
    Path("project")
    / "orthagonal_axis"
    / "results"
    / "od_slp_003_heldout"
    / "dev_manifest.json"
)
OD003_PROTOCOL_SHA256 = (
    "da085e19eb86ea6caebd629f829c7353a0e7112fd6d1084bfde85c0ccb2c78b7"
)
OD003_MANIFEST_SHA256 = (
    "91f932c58095a4853d4db1b3d87721f76362e876387011d240a124233e6725ba"
)
OD004_PROTOCOL_SHA256 = (
    "5d3050a00e1e7b3c0c32ef6c28e084495c6e605769714a0e1c2925c8fe2ee8f5"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _verify_prerequisites(repo_root: Path) -> dict[str, Any]:
    base = verify_prerequisite_hashes(repo_root)
    records = dict(base.get("records", {}))
    for name, relative, expected in (
        ("od_slp_004_protocol", PROTOCOL_PATH, OD004_PROTOCOL_SHA256),
        ("od_slp_003_protocol", OD003_PROTOCOL_PATH, OD003_PROTOCOL_SHA256),
        ("od_slp_003_dev_manifest", OD003_MANIFEST_PATH, OD003_MANIFEST_SHA256),
    ):
        path = repo_root / relative
        actual = sha256_file(path) if path.exists() else None
        records[name] = {
            "path": str(relative),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "status": "MATCH" if actual == expected else "MISMATCH",
        }
    protocol_path = repo_root / PROTOCOL_PATH
    protocol_sha = sha256_file(protocol_path) if protocol_path.exists() else None
    status = (
        "MATCH"
        if base.get("status") == "MATCH"
        and all(record["status"] == "MATCH" for record in records.values())
        else "MISMATCH"
    )
    return {
        "status": status,
        "records": records,
        "protocol_sha256": protocol_sha,
    }


def _empty_inventory() -> dict[str, Any]:
    return {
        "participants_pid_count": 0,
        "pair_paths_discovered": 0,
        "dev_pair_paths_loaded": 0,
        "non_dev_pair_paths_not_loaded": 0,
        "dev_records": 0,
        "dev_valid_records": 0,
        "source_failures": {},
        "pid_split_leakage": False,
    }


def _empty_scores(reason: str) -> dict[str, dict[str, Any]]:
    return {
        stratum: _empty_score_payload(stratum, reason)
        for stratum in EVALUATION_STRATA
    }


def _write(payload: dict[str, Any], output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            json_safe(payload),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def _base_payload(
    *,
    raw_root: Path,
    prerequisites: dict[str, Any],
    root_ok: bool,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": prerequisites.get("protocol_sha256"),
        "prerequisite_hashes": {
            name: record.get("actual_sha256")
            for name, record in prerequisites.get("records", {}).items()
        },
        "prerequisite_hash_status": prerequisites.get("status", "MISMATCH"),
        "source_root": str(raw_root),
        "source_root_expected": CANONICAL_SOURCE_ROOT,
        "source_root_canonical": bool(root_ok),
        "dataset": "ds005555 / BOAS",
        "split": {
            "split_id": "OD-SLP-001-pid-split-v1",
            "digest": split_digest(),
            "matches_frozen_lists": False,
            "frozen": {
                arm: list(values) for arm, values in FROZEN_PID_SPLIT.items()
            },
        },
        "frozen_geometry": {
            "grid_resolution": GRID_RESOLUTION,
            "support_floor": SUPPORT_FLOOR,
            "lower": LOWER_BOUNDARY,
            "upper": UPPER_BOUNDARY,
            "a_DEV": A_DEV,
            "query_grid": frozen_query_grid(),
            "set_A": [3],
            "set_B": [4],
            "interior_stage": 2,
            "min_transition_segments": MIN_TRANSITION_SEGMENTS,
            "max_dt_relative_deviation": MAX_DT_RELATIVE_DEVIATION,
        },
        "frozen_base_rate": {
            "n3_DEV": N3_DEV,
            "rem_DEV": 206,
            "p_REM_DEV": P_REM_DEV,
            "q_0": P_REM_DEV,
        },
        "source_inventory": _empty_inventory(),
        "records": [],
        "archived_geometry_check": None,
        "pooled_fit": None,
        "archived_pooled_replay": None,
        "dev_base_rate_brier": None,
        "transfer": None,
        "score_surfaces": _empty_scores("audit_precondition_not_satisfied"),
        "evaluation_status": "NOT_TESTABLE",
        "stationarity_status": "NOT_TESTABLE",
        "combined_status": "NOT_TESTABLE",
        "single_pooled_fit_count": 0,
        "fit_request_log": [],
        "failure_reasons": [],
        "fail_closed_assertions": {
            "held_out_opened_once": False,
            "held_out_signal_or_outcome_read": False,
            "reserve_opened": False,
            "reserve_signal_or_outcome_read": False,
            "non_dev_edf_loaded": False,
            "pid_split_leakage": False,
            "single_pooled_fit_only": False,
            "early_fit_attempted": False,
            "late_fit_attempted": False,
            "query_grid_unchanged": False,
            "no_support_floor_relaxation": True,
            "no_grid_reselection": True,
            "no_q_extrapolation": True,
            "no_n2_to_exit_increment": False,
            "bootstrap_rng_constructed": False,
            "hdf5_written": False,
            "production_overlay_written": False,
            "production_o2b_guard_modified": False,
        },
        "software_versions": {
            "python": sys.version,
            "numpy": np.__version__,
        },
        "claim_boundary": (
            "DEV-only score-only stationarity audit of one pooled BOAS "
            "G=17 local law. It does not open HELD_OUT/RESERVE, perform "
            "held-out qualification, write HDF5, or establish a biological "
            "or time-homogeneous sleep-law claim."
        ),
    }


def run_audit(*, raw_root: Path, output_path: Path) -> dict[str, Any]:
    """Run the OD-SLP-004 DEV-only pooled-law stationarity audit."""
    if output_path.exists():
        raise FileExistsError(f"refusing_to_overwrite:{output_path}")
    repo_root = _repo_root()
    prerequisites = _verify_prerequisites(repo_root)
    root_ok = canonical_source_root_matches(raw_root)
    result = _base_payload(
        raw_root=raw_root,
        prerequisites=prerequisites,
        root_ok=root_ok,
    )
    if prerequisites.get("status") != "MATCH":
        result["failure_reasons"].append("prerequisite_hash_mismatch")
        return _write(result, output_path)
    if not root_ok:
        result["failure_reasons"].append("noncanonical_source_root")
        return _write(result, output_path)

    geometry_check = verify_frozen_geometry(repo_root)
    result["archived_geometry_check"] = geometry_check
    if geometry_check.get("status") != "MATCH":
        result["failure_reasons"].append("frozen_geometry_or_tq_mismatch")
        return _write(result, output_path)

    protocol = SleepFirstHitProtocol()
    try:
        participants = _load_pid_map(raw_root)
        participant_pids = [
            pid for pid in participants.values() if pid is not None
        ]
        split_check = assert_frozen_pid_split(participant_pids)
        result["split"].update(split_check)
        result["split"]["digest"] = split_digest()
        result["source_inventory"]["participants_pid_count"] = len(
            participant_pids
        )
        if not split_check.get("matches_frozen_lists", False):
            result["failure_reasons"].append(
                split_check.get("reason") or "frozen_pid_lists_mismatch"
            )
            return _write(result, output_path)

        pairs = _pair_acquisitions(raw_root)
        result["source_inventory"]["pair_paths_discovered"] = len(pairs)
        dev_pids = {str(int(pid)) for pid in FROZEN_PID_SPLIT["DEV"]}
        pid_to_split = {
            str(pid): arm
            for arm, values in FROZEN_PID_SPLIT.items()
            for pid in values
        }
        dev_pairs: list[dict[str, Any]] = []
        non_dev_not_loaded = 0
        for pair in pairs:
            try:
                pid = canonical_pid(
                    participants.get(pair.get("participant_id"))
                )
            except (TypeError, ValueError, OverflowError):
                non_dev_not_loaded += 1
                continue
            if pid in dev_pids:
                dev_pairs.append(pair)
            else:
                non_dev_not_loaded += 1
        result["source_inventory"]["dev_pair_paths_loaded"] = len(dev_pairs)
        result["source_inventory"][
            "non_dev_pair_paths_not_loaded"
        ] = non_dev_not_loaded
        result["fail_closed_assertions"]["non_dev_edf_loaded"] = bool(
            non_dev_not_loaded != len(pairs) - len(dev_pairs)
        )
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
        leakage = any(record.get("split") != "DEV" for record in records)
        result["source_inventory"].update(
            {
                "dev_records": len(records),
                "dev_valid_records": sum(
                    not record.get("failure_reasons") for record in records
                ),
                "source_failures": dict(sorted(source_failures.items())),
                "pid_split_leakage": bool(leakage),
            }
        )
        result["fail_closed_assertions"]["pid_split_leakage"] = bool(leakage)
        result["records"] = [
            _clean_json_record(record) for record in records
        ]
        dev_segments = _all_segments(records, "DEV")
    except (OSError, KeyError, TypeError, ValueError) as error:
        result["failure_reasons"].append(
            f"source_or_split_load_failed:{type(error).__name__}:{error}"
        )
        return _write(result, output_path)

    if source_failures or leakage or not dev_segments:
        result["failure_reasons"].append("dev_source_not_fit_eligible")
        return _write(result, output_path)

    # The only q-producing call in this runner. Keep the request log as the
    # source of truth for the fail-closed fit-attempt assertions.
    fit_request_log: list[str] = []

    def request_fit(stratum: str) -> dict[str, Any]:
        fit_request_log.append(stratum)
        return fit_dev_grid(dev_segments, stratum=stratum)

    pooled_fit = request_fit("pooled")
    result["fit_request_log"] = list(fit_request_log)
    result["single_pooled_fit_count"] = fit_request_log.count("pooled")
    result["pooled_fit"] = pooled_fit
    result["fail_closed_assertions"]["early_fit_attempted"] = (
        "early" in fit_request_log
    )
    result["fail_closed_assertions"]["late_fit_attempted"] = (
        "late" in fit_request_log
    )
    result["fail_closed_assertions"]["single_pooled_fit_only"] = (
        fit_request_log == ["pooled"]
        and not result["fail_closed_assertions"]["early_fit_attempted"]
        and not result["fail_closed_assertions"]["late_fit_attempted"]
    )
    if pooled_fit.get("status") != "computed":
        result["failure_reasons"].append("pooled_fit_not_testable")
        return _write(result, output_path)
    result["fail_closed_assertions"]["no_n2_to_exit_increment"] = (
        pooled_fit.get("appended_absorbing_boundary_rows") == 0
    )
    if not result["fail_closed_assertions"]["no_n2_to_exit_increment"]:
        result["failure_reasons"].append("absorbing_boundary_rows_present")
        return _write(result, output_path)

    try:
        archived_manifest = json.loads(
            (repo_root / OD003_MANIFEST_PATH).read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        result["failure_reasons"].append(
            f"archived_manifest_unreadable:{type(error).__name__}:{error}"
        )
        return _write(result, output_path)
    replay = compare_archived_pooled_fit(pooled_fit, archived_manifest)
    result["archived_pooled_replay"] = replay
    if replay.get("status") != "PASS":
        result["failure_reasons"].append("archived_pooled_replay_mismatch")
        return _write(result, output_path)
    result["fail_closed_assertions"]["query_grid_unchanged"] = True

    base_brier = dev_base_rate_brier(dev_segments, pooled_fit)
    result["dev_base_rate_brier"] = base_brier
    expected_brier = archived_manifest.get(
        "dev_base_rate_brier", {}
    ).get("brier")
    if (
        base_brier.get("status") != "computed"
        or expected_brier is None
        or not np.isclose(
            float(base_brier["brier"]),
            float(expected_brier),
            rtol=0.0,
            atol=1e-12,
        )
    ):
        result["failure_reasons"].append("dev_base_rate_brier_mismatch")
        return _write(result, output_path)
    expected_tolerance = archived_manifest.get("transfer_tolerance")
    actual_tolerance = calculate_transfer_tolerance(
        float(base_brier["brier"])
    )
    if (
        expected_tolerance is None
        or actual_tolerance is None
        or not np.isclose(
            float(actual_tolerance),
            float(expected_tolerance),
            rtol=0.0,
            atol=1e-12,
        )
    ):
        result["failure_reasons"].append("transfer_tolerance_mismatch")
        return _write(result, output_path)

    score_surfaces = {
        stratum: score_fixed_law_surface(
            dev_segments,
            q_grid=pooled_fit["q_grid"],
            query_grid=pooled_fit["query_grid"],
            stratum=stratum,
        )
        for stratum in EVALUATION_STRATA
    }
    result["score_surfaces"] = score_surfaces
    result["transfer"] = transfer_payload(
        dev_base_brier=float(base_brier["brier"]),
        early=score_surfaces["early"],
        late=score_surfaces["late"],
    )
    evaluation_ok = all(
        score_surfaces[stratum].get("status") == "computed"
        for stratum in EVALUATION_STRATA
    )
    result["evaluation_status"] = "PASS" if evaluation_ok else "NOT_TESTABLE"
    tolerance = result["transfer"].get("transfer_tolerance")
    statuses = stationarity_status(
        evaluation_status=result["evaluation_status"],
        early=score_surfaces["early"],
        late=score_surfaces["late"],
        transfer_tolerance=tolerance,
    )
    result.update(statuses)
    if result["evaluation_status"] != "PASS":
        result["failure_reasons"].append("evaluation_support_not_testable")
    return _write(result, output_path)


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
                "evaluation_status": result["evaluation_status"],
                "stationarity_status": result["stationarity_status"],
                "combined_status": result["combined_status"],
                "single_pooled_fit_count": result["single_pooled_fit_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
