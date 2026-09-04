"""Run the OD-SLP-003 BOAS DEV-only manifest gate.

This runner stops after fitting pooled, early, and late DEV G=17 audit-only
q-grids. It never loads HELD_OUT or RESERVE signal/outcome data and does not
implement held-out scoring, nulls, bootstrap evaluation, HDF5, or overlays.
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

from mndm.dynamical_families.sleep_committor_dev_audit import endpoint_census
from mndm.dynamical_families.sleep_committor_dev_manifest import (
    A_DEV,
    CANONICAL_SOURCE_ROOT,
    DEV_STRATA,
    GRID_RESOLUTION,
    LOWER_BOUNDARY,
    MAX_DT_RELATIVE_DEVIATION,
    MIN_TRANSITION_SEGMENTS,
    N3_DEV,
    OD_SLP_003_PROTOCOL_ID,
    OD_SLP_003_SCHEMA,
    P_REM_DEV,
    PREREQUISITE_HASHES,
    REM_DEV,
    SUPPORT_FLOOR,
    UPPER_BOUNDARY,
    calculate_transfer_tolerance,
    canonical_source_root_matches,
    dev_base_rate_brier,
    fit_dev_grid,
    frozen_query_grid,
    sha256_file,
    split_digest,
    verify_frozen_geometry,
    verify_prerequisite_hashes,
)
from mndm.dynamical_families.sleep_committor_qualification import (
    FROZEN_PID_SPLIT,
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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _not_testable_fit(stratum: str, reason: str) -> dict[str, Any]:
    return {
        "status": "not_testable",
        "failure_reason": reason,
        "stratum": stratum,
        "n_rows": 0,
        "n_segments": 0,
        "n_dropped_segments": 0,
        "n_increment_pairs": 0,
        "nominal_dt_sec": None,
        "max_dt_relative_deviation": None,
        "query_grid": [],
        "q_grid": [],
        "support_count": [],
        "n_transition_segments": None,
        "appended_absorbing_boundary_rows": None,
        "provenance": None,
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


def _frozen_geometry_payload() -> dict[str, Any]:
    return {
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
    }


def _future_scoring_contract() -> dict[str, Any]:
    return {
        "held_out_interpolation": "linear_np_interp",
        "held_out_refit": False,
        "held_out_support_floor_refit": False,
        "bootstrap_seed": 20260816,
        "bootstrap_replicates": 2000,
        "nc_label_seed": 20260817,
        "rng_constructed_in_this_gate": False,
    }


def _base_manifest(
    *,
    raw_root: Path,
    hashes: dict[str, Any],
    source_root_canonical: bool,
) -> dict[str, Any]:
    return {
        "schema": OD_SLP_003_SCHEMA,
        "protocol_id": OD_SLP_003_PROTOCOL_ID,
        "protocol_sha256": hashes.get("protocol_sha256"),
        "prerequisite_hashes": {
            name: record.get("actual_sha256")
            for name, record in hashes.get("records", {}).items()
        },
        "prerequisite_hash_status": hashes.get("status", "MISMATCH"),
        "source_root": str(raw_root),
        "source_root_expected": CANONICAL_SOURCE_ROOT,
        "source_root_canonical": bool(source_root_canonical),
        "dataset": "ds005555 / BOAS",
        "split": {
            "split_id": "OD-SLP-001-pid-split-v1",
            "digest": split_digest(),
            "matches_frozen_lists": False,
            "frozen": {
                arm: list(values) for arm, values in FROZEN_PID_SPLIT.items()
            },
        },
        "frozen_geometry": _frozen_geometry_payload(),
        "frozen_base_rate": {
            "n3_DEV": N3_DEV,
            "rem_DEV": REM_DEV,
            "p_REM_DEV": P_REM_DEV,
            "q_0": P_REM_DEV,
        },
        "source_inventory": _empty_inventory(),
        "endpoint_census_replay": None,
        "fits": {
            stratum: _not_testable_fit(
                stratum,
                "manifest_precondition_not_satisfied",
            )
            for stratum in DEV_STRATA
        },
        "dev_base_rate_brier": None,
        "transfer_tolerance": None,
        "dev_adapter_status": "NOT_TESTABLE",
        "held_out_eligibility_inventory": None,
        "held_out_scoring_deferred": True,
        "reserved_future_scoring_contract": _future_scoring_contract(),
        "adapter_fingerprint": None,
        "software_versions": {
            "python": sys.version,
            "numpy": np.__version__,
        },
        "failure_reasons": [],
        "records": [],
        "fail_closed_assertions": {
            "dev_manifest_frozen_before_held_out_scoring": True,
            "held_out_opened_once": False,
            "held_out_signal_or_outcome_read": False,
            "reserve_opened": False,
            "reserve_signal_or_outcome_read": False,
            "non_dev_edf_loaded": False,
            "pid_split_leakage": False,
            "no_held_out_recalibration": True,
            "no_outcome_driven_selection": True,
            "no_support_floor_relaxation": True,
            "no_q_extrapolation": True,
        "no_n2_to_exit_increment": False,
            "competing_or_censored_rows_dropped": False,
            "hdf5_written": False,
            "production_overlay_written": False,
            "production_o2b_guard_modified": False,
            "legacy_estimator_used": False,
            "geometry_regenerated": False,
            "p_ab_dev_used_as_comparator": False,
        },
        "claim_boundary": (
            "DEV-only BOAS manifest for the preregistered OD-SLP-003 "
            "held-out gate. It freezes three audit-only G=17 q-grids, "
            "support, provenance, and the empirical DEV base-rate Brier. "
            "It does not score HELD_OUT, open RESERVE, establish held-out "
            "calibration, or qualify a production measurement."
        ),
    }


def _write_manifest(payload: dict[str, Any], output_path: Path) -> dict[str, Any]:
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


def run_manifest(*, raw_root: Path, output_path: Path) -> dict[str, Any]:
    """Run only the DEV-manifest stage and refuse overwrite."""
    if output_path.exists():
        raise FileExistsError(f"refusing_to_overwrite:{output_path}")

    repo_root = _repo_root()
    hashes = verify_prerequisite_hashes(repo_root)
    source_root_canonical = canonical_source_root_matches(raw_root)
    result = _base_manifest(
        raw_root=raw_root,
        hashes=hashes,
        source_root_canonical=source_root_canonical,
    )
    result["adapter_fingerprint"] = sha256_file(
        repo_root
        / "mndm"
        / "src"
        / "mndm"
        / "dynamical_families"
        / "committor_coarse_grid_qualification.py"
    )
    if hashes.get("status") != "MATCH":
        result["failure_reasons"].append("prerequisite_hash_mismatch")
        return _write_manifest(result, output_path)
    if not source_root_canonical:
        result["failure_reasons"].append("noncanonical_source_root")
        return _write_manifest(result, output_path)

    geometry_check = verify_frozen_geometry(repo_root)
    result["archived_geometry_check"] = geometry_check
    if geometry_check.get("status") != "MATCH":
        result["failure_reasons"].append("frozen_geometry_or_tq_mismatch")
        return _write_manifest(result, output_path)

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
            return _write_manifest(result, output_path)

        pairs = _pair_acquisitions(raw_root)
        result["source_inventory"]["pair_paths_discovered"] = len(pairs)
        dev_pid_strings = {str(pid) for pid in FROZEN_PID_SPLIT["DEV"]}
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
            if pid in dev_pid_strings:
                dev_pairs.append(pair)
            else:
                non_dev_not_loaded += 1
        result["source_inventory"][
            "dev_pair_paths_loaded"
        ] = len(dev_pairs)
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
        pid_split_leakage = any(
            record.get("split") != "DEV" for record in records
        )
        result["source_inventory"].update(
            {
                "dev_records": len(records),
                "dev_valid_records": sum(
                    not record.get("failure_reasons") for record in records
                ),
                "source_failures": dict(sorted(source_failures.items())),
                "pid_split_leakage": bool(pid_split_leakage),
            }
        )
        result["fail_closed_assertions"]["pid_split_leakage"] = bool(
            pid_split_leakage
        )
        result["records"] = [_clean_json_record(record) for record in records]
        dev_segments = _all_segments(records, "DEV")
    except (OSError, KeyError, TypeError, ValueError) as error:
        result["failure_reasons"].append(
            f"source_or_split_load_failed:{type(error).__name__}:{error}"
        )
        return _write_manifest(result, output_path)

    endpoint = (
        endpoint_census(dev_segments, dev_pids=FROZEN_PID_SPLIT["DEV"])
        if dev_segments and not pid_split_leakage
        else {
            "status": "ENDPOINT_NOT_TESTABLE",
            "failure_reason": "no_valid_dev_segments_or_pid_leakage",
            "n3_count": 0,
            "rem_count": 0,
        }
    )
    endpoint["matches_frozen_305_206"] = bool(
        endpoint.get("n3_count") == N3_DEV
        and endpoint.get("rem_count") == REM_DEV
    )
    result["endpoint_census_replay"] = endpoint

    pre_fit_reasons: list[str] = []
    if source_failures:
        pre_fit_reasons.append("source_failures_present")
    if pid_split_leakage:
        pre_fit_reasons.append("pid_split_leakage")
    if geometry_check.get("status") != "MATCH":
        pre_fit_reasons.append("frozen_geometry_or_tq_mismatch")
    if not endpoint.get("matches_frozen_305_206", False):
        pre_fit_reasons.append("dev_endpoint_counts_mismatch")
    if not dev_segments:
        pre_fit_reasons.append("no_valid_dev_segments")
    if pre_fit_reasons:
        result["failure_reasons"].extend(pre_fit_reasons)
    else:
        result["fits"] = {
            stratum: fit_dev_grid(dev_segments, stratum=stratum)
            for stratum in DEV_STRATA
        }
    all_fits_computed = all(
        result["fits"][stratum].get("status") == "computed"
        for stratum in DEV_STRATA
    )
    no_absorbing_rows = all_fits_computed and all(
        result["fits"][stratum].get(
            "appended_absorbing_boundary_rows"
        ) == 0
        for stratum in DEV_STRATA
    )
    result["fail_closed_assertions"]["no_n2_to_exit_increment"] = (
        no_absorbing_rows
    )
    if all_fits_computed and not no_absorbing_rows:
        pre_fit_reasons.append("absorbing_boundary_rows_present")
        result["failure_reasons"].append("absorbing_boundary_rows_present")

    pooled_fit = result["fits"]["pooled"]
    if pooled_fit.get("status") == "computed" and not pre_fit_reasons:
        brier = dev_base_rate_brier(dev_segments, pooled_fit)
        result["dev_base_rate_brier"] = brier
        result["transfer_tolerance"] = calculate_transfer_tolerance(
            brier.get("brier")
        )
        if brier.get("status") != "computed":
            result["failure_reasons"].append(
                "dev_base_rate_brier_not_testable"
            )
    else:
        result["dev_base_rate_brier"] = {
            "status": "not_testable",
            "reason": "pooled_fit_precondition_not_satisfied",
            "n_rows": 0,
            "brier": None,
        }

    fit_statuses = [
        result["fits"][stratum].get("status") == "computed"
        for stratum in DEV_STRATA
    ]
    base_rate_ok = (
        isinstance(result["dev_base_rate_brier"], dict)
        and result["dev_base_rate_brier"].get("status") == "computed"
        and result.get("transfer_tolerance") is not None
    )
    result["dev_adapter_status"] = (
        "PASS"
        if all(fit_statuses) and base_rate_ok and not pre_fit_reasons
        else "NOT_TESTABLE"
    )
    if result["dev_adapter_status"] != "PASS":
        result["failure_reasons"].append("dev_adapter_not_testable")
    return _write_manifest(result, output_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_manifest(raw_root=args.raw_root, output_path=args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "dev_adapter_status": result["dev_adapter_status"],
                "source_root_canonical": result["source_root_canonical"],
                "fit_statuses": {
                    key: value.get("status")
                    for key, value in result["fits"].items()
                },
                "transfer_tolerance": result["transfer_tolerance"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
