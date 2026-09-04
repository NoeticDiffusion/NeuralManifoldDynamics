"""Pure contract helpers for the OD-SLP-003 BOAS DEV manifest.

This module deliberately stops before held-out scoring.  It freezes the DEV
G=17 audit-only q-grids, their support/provenance payloads, and the empirical
DEV base-rate Brier scalar used to derive the later stationarity tolerance.
It does not read source files, open EDFs, construct nulls, or use RNGs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .committor_coarse_grid_qualification import (
    estimate_audit_only_coarse_grid,
)
from .contracts import build_provenance
from .sleep_committor_qualification import (
    FROZEN_PID_SPLIT,
    RESOLVED_OUTCOMES,
    assemble_adapter_arrays,
    binary_metrics,
    resolved_scoring_rows,
)

OD_SLP_003_SCHEMA = "mndm.od_slp_003_dev_manifest.v1"
OD_SLP_003_PROTOCOL_ID = "OD-SLP-003"
OD_SLP_003_PROTOCOL_RELATIVE_PATH = (
    Path("project")
    / "orthagonal_axis"
    / "od_slp_003_heldout_preregistration.md"
)

GRID_RESOLUTION = 17
SUPPORT_FLOOR = 64
MIN_TRANSITION_SEGMENTS = 20
MAX_DT_RELATIVE_DEVIATION = 0.05
LOWER_BOUNDARY = 0.38829287332677653
UPPER_BOUNDARY = 4.212652852687446
A_DEV = 0.017567656341574884
N3_DEV = 305
REM_DEV = 206
P_REM_DEV = 0.40313111545988256
CANONICAL_SOURCE_ROOT = r"M:\datasets\received\openneuro\ds005555"
DEV_STRATA = ("pooled", "early", "late")

PREREQUISITE_HASHES: dict[str, tuple[Path, str]] = {
    "od_slp_001_protocol": (
        Path("project")
        / "orthagonal_axis"
        / "od_slp_001_preregistration.md",
        "f76c77f343ab2b94a29c04b70c6ca69b30f62f793043b4f0f8f329a9e6feb543",
    ),
    "od_slp_002b_protocol": (
        Path("project")
        / "orthagonal_axis"
        / "od_slp_002b_boas_dev.md",
        "cd48fd5ddc28e5fd0dabcec31c5b630e60c87ca3b29555da2af7d95ae36366df",
    ),
    "od_slp_002a_nmd_tq_result": (
        Path("project")
        / "orthagonal_axis"
        / "results"
        / "od_slp_002a_nmd_tq"
        / "qualification.json",
        "0647fbd389da255c73942e080f0407d2fe617ef91b0f14612e463d7f0ab53691",
    ),
    "od_slp_002b_boas_dev_audit": (
        Path("project")
        / "orthagonal_axis"
        / "results"
        / "od_slp_002b_boas_dev"
        / "boas_dev_audit.json",
        "9df48d503f0317bed3c039e5d23540095b97ab72e17bc5ef572b07be6eb1701d",
    ),
}


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 digest for one repository file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_prerequisite_hashes(repo_root: Path) -> dict[str, Any]:
    """Verify all hash-bound OD-SLP-003 inputs without opening source data."""
    records: dict[str, dict[str, Any]] = {}
    matches = True
    for name, (relative_path, expected) in PREREQUISITE_HASHES.items():
        path = repo_root / relative_path
        if not path.exists():
            actual = None
            reason = "missing_hash_bound_file"
            matches = False
        else:
            actual = sha256_file(path)
            reason = None if actual == expected else "hash_mismatch"
            matches = matches and actual == expected
        records[name] = {
            "path": str(relative_path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "status": "MATCH" if reason is None else "MISMATCH",
            "failure_reason": reason,
        }
    protocol_path = repo_root / OD_SLP_003_PROTOCOL_RELATIVE_PATH
    protocol_sha256 = sha256_file(protocol_path) if protocol_path.exists() else None
    return {
        "status": "MATCH" if matches else "MISMATCH",
        "records": records,
        "protocol_path": str(OD_SLP_003_PROTOCOL_RELATIVE_PATH),
        "protocol_sha256": protocol_sha256,
    }


def canonical_source_root_matches(raw_root: Path | str) -> bool:
    """Accept only the frozen M: source-root spelling modulo slash style."""
    value = str(raw_root).replace("/", "\\").rstrip("\\").casefold()
    expected = CANONICAL_SOURCE_ROOT.replace("/", "\\").rstrip("\\").casefold()
    return value == expected


def split_digest(
    split: Mapping[str, Sequence[int]] = FROZEN_PID_SPLIT,
) -> str:
    """Hash the frozen ordered split lists exactly as preregistered."""
    payload = "|".join(
        f"{arm}:{','.join(str(int(pid)) for pid in split[arm])}"
        for arm in ("DEV", "HELD_OUT", "RESERVE")
    )
    return hashlib.sha256(
        f"OD-SLP-001-pid-split-v1|{payload}".encode("utf-8")
    ).hexdigest()


def frozen_query_grid() -> np.ndarray:
    """Return the exact G=17 grid fixed by the BOAS geometry."""
    return np.linspace(
        LOWER_BOUNDARY,
        UPPER_BOUNDARY,
        GRID_RESOLUTION,
        dtype=float,
    )


def verify_frozen_geometry(repo_root: Path) -> dict[str, Any]:
    """Check archived 002B geometry and 002A G=17 status."""
    result: dict[str, Any] = {
        "status": "MATCH",
        "failure_reasons": [],
        "od_slp_002b": {},
        "od_slp_002a": {},
    }
    audit_path = repo_root / PREREQUISITE_HASHES[
        "od_slp_002b_boas_dev_audit"
    ][0]
    tq_path = repo_root / PREREQUISITE_HASHES["od_slp_002a_nmd_tq_result"][0]
    try:
        import json

        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        tq = json.loads(tq_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        result["status"] = "MISMATCH"
        result["failure_reasons"].append(f"archived_json_unreadable:{error}")
        return result

    frozen_contract = audit.get("frozen_contract", {})
    selected_grid = audit.get("selection", {}).get("selected_grid")
    endpoint = audit.get("endpoint_census", {})
    candidate_status = tq.get("decision", {}).get(
        "candidate_statuses", {}
    ).get("17")
    def _close(value: Any, expected: float) -> bool:
        try:
            return bool(np.isclose(float(value), expected))
        except (TypeError, ValueError):
            return False

    checks = {
        "selected_grid_is_17": selected_grid == GRID_RESOLUTION,
        "support_floor": frozen_contract.get("support_floor") == SUPPORT_FLOOR,
        "lower_boundary": _close(
            frozen_contract.get("lower_boundary"), LOWER_BOUNDARY
        ),
        "upper_boundary": _close(
            frozen_contract.get("upper_boundary"), UPPER_BOUNDARY
        ),
        "a_DEV": _close(frozen_contract.get("a_DEV"), A_DEV),
        "n3_count": endpoint.get("n3_count") == N3_DEV,
        "rem_count": endpoint.get("rem_count") == REM_DEV,
        "tq_candidate_17_pass": candidate_status == "PASS",
    }
    result["od_slp_002b"] = {
        "selected_grid": selected_grid,
        "endpoint_n3_count": endpoint.get("n3_count"),
        "endpoint_rem_count": endpoint.get("rem_count"),
        "checks": checks,
    }
    result["od_slp_002a"] = {
        "candidate_status_17": candidate_status,
    }
    result["failure_reasons"].extend(
        f"archived_contract_mismatch:{name}"
        for name, passed in checks.items()
        if not bool(passed)
    )
    result["status"] = (
        "MATCH" if not result["failure_reasons"] else "MISMATCH"
    )
    return result


def dev_segments_for_stratum(
    segments: Sequence[Mapping[str, Any]],
    stratum: str,
) -> list[Mapping[str, Any]]:
    """Return pooled, early, or late DEV segments without changing rows."""
    if stratum not in DEV_STRATA:
        raise ValueError(f"unknown_dev_stratum:{stratum}")
    if stratum == "pooled":
        return list(segments)
    return [
        segment
        for segment in segments
        if segment.get("night_stratum") == f"{stratum}_night"
    ]


def _increment_summary(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    segment_id = np.asarray(arrays["segment_id"])
    time = np.asarray(arrays["time"], dtype=float)
    if time.size < 2:
        return {
            "n_increment_pairs": 0,
            "nominal_dt_sec": None,
            "max_dt_relative_deviation": None,
        }
    same_segment = segment_id[1:] == segment_id[:-1]
    positive_dt = np.diff(time) > 0
    dts = np.diff(time)[same_segment & positive_dt]
    if dts.size == 0:
        return {
            "n_increment_pairs": 0,
            "nominal_dt_sec": None,
            "max_dt_relative_deviation": None,
        }
    nominal = float(np.median(dts))
    return {
        "n_increment_pairs": int(dts.size),
        "nominal_dt_sec": nominal,
        "max_dt_relative_deviation": float(
            np.max(np.abs(dts - nominal)) / nominal
        ),
    }


def _external_rc_provenance() -> dict[str, Any]:
    provenance = build_provenance(
        coordinate_layer="external_headband_rc",
        coordinate_names=["headband_slow_fast_logratio"],
        time_semantics="audit_only_within_segment_local_law",
        estimator="audit_only_local_law_dense_grid_o2b",
        settings={
            "grid_resolution": GRID_RESOLUTION,
            "grid_min": LOWER_BOUNDARY,
            "grid_max": UPPER_BOUNDARY,
            "diffusion_coefficient": A_DEV,
            "min_support_per_grid": SUPPORT_FLOOR,
            "min_transition_segments": MIN_TRANSITION_SEGMENTS,
            "max_dt_relative_deviation": MAX_DT_RELATIVE_DEVIATION,
            "set_A": [3],
            "set_B": [4],
            "interior_stage": 2,
            "audit_only": True,
            "production_guard_unchanged": True,
        },
    )
    provenance["validation_level"] = "mndm_translation_validated"
    provenance["data_qualification"] = "dev_manifest_only"
    return provenance


def fit_dev_grid(
    segments: Sequence[Mapping[str, Any]],
    *,
    stratum: str,
) -> dict[str, Any]:
    """Fit one frozen DEV audit grid and return a compact JSON-safe payload."""
    selected = dev_segments_for_stratum(segments, stratum)
    allowed_pids = {str(int(pid)) for pid in FROZEN_PID_SPLIT["DEV"]}
    invalid_pid = False
    for segment in selected:
        try:
            pid = str(int(float(segment.get("pid"))))
        except (TypeError, ValueError, OverflowError):
            invalid_pid = True
            break
        if pid not in allowed_pids:
            invalid_pid = True
            break
    if invalid_pid:
        return {
            "status": "not_testable",
            "failure_reason": "support_segment_not_in_dev_split",
            "stratum": stratum,
            "n_rows": 0,
            "n_segments": len(selected),
            "n_dropped_segments": 0,
            "n_increment_pairs": 0,
            "nominal_dt_sec": None,
            "max_dt_relative_deviation": None,
            "query_grid": [],
            "q_grid": [],
            "support_count": [],
            "n_transition_segments": None,
            "appended_absorbing_boundary_rows": 0,
            "provenance": None,
        }
    arrays = assemble_adapter_arrays(selected)
    increments = _increment_summary(arrays)
    payload: dict[str, Any] = {
        "status": "not_testable",
        "failure_reason": None,
        "stratum": stratum,
        "n_rows": int(arrays["state"].shape[0]),
        "n_segments": int(len(arrays["included_segment_ids"])),
        "n_dropped_segments": int(len(arrays["dropped_segment_ids"])),
        **increments,
        "query_grid": [],
        "q_grid": [],
        "support_count": [],
        "n_transition_segments": None,
        "appended_absorbing_boundary_rows": 0,
        "provenance": None,
    }
    if arrays["state"].shape[0] < 30:
        payload["failure_reason"] = "insufficient_adapter_rows"
        return payload
    result = estimate_audit_only_coarse_grid(
        arrays["state"],
        arrays["time"],
        arrays["regime_labels"],
        set_A=[3],
        set_B=[4],
        grid_min=LOWER_BOUNDARY,
        grid_max=UPPER_BOUNDARY,
        diffusion_coefficient=A_DEV,
        segment_id=arrays["segment_id"],
        grid_resolution=GRID_RESOLUTION,
        min_support_per_grid=SUPPORT_FLOOR,
        min_transition_segments=MIN_TRANSITION_SEGMENTS,
        max_dt_relative_deviation=MAX_DT_RELATIVE_DEVIATION,
    )
    payload["status"] = result.get("computation_status", "not_testable")
    payload["failure_reason"] = result.get("failure_reason")
    summary = result.get("summary", {})
    series = result.get("series", {})
    payload["query_grid"] = series.get(
        "query_grid", result.get("query_grid", [])
    )
    payload["q_grid"] = series.get("q_grid", result.get("q_grid", []))
    payload["support_count"] = series.get(
        "grid_support_count",
        result.get("grid_support_count", []),
    )
    payload["n_transition_segments"] = summary.get("n_transition_segments")
    absorbing_rows = summary.get("appended_absorbing_boundary_rows")
    payload["appended_absorbing_boundary_rows"] = (
        int(absorbing_rows) if absorbing_rows is not None else None
    )
    if payload["status"] == "computed":
        q_grid = np.asarray(payload["q_grid"], dtype=float)
        query_grid = np.asarray(payload["query_grid"], dtype=float)
        support = np.asarray(payload["support_count"], dtype=float)
        if (
            q_grid.shape != (GRID_RESOLUTION,)
            or query_grid.shape != (GRID_RESOLUTION,)
            or support.shape != (GRID_RESOLUTION,)
            or not np.all(np.isfinite(q_grid))
            or not np.all(np.isfinite(query_grid))
            or not np.all(np.isfinite(support))
            or np.any(support < SUPPORT_FLOOR)
            or not np.allclose(query_grid, frozen_query_grid(), rtol=0.0, atol=0.0)
        ):
            payload["status"] = "not_testable"
            payload["failure_reason"] = "invalid_finite_dev_fit_payload"
        elif payload["appended_absorbing_boundary_rows"] is None:
            payload["status"] = "not_testable"
            payload["failure_reason"] = (
                "absorbing_boundary_row_count_missing"
            )
        elif payload["appended_absorbing_boundary_rows"] != 0:
            payload["status"] = "not_testable"
            payload["failure_reason"] = "absorbing_boundary_rows_present"
        else:
            payload["provenance"] = _external_rc_provenance()
    if payload["status"] != "computed":
        payload["q_grid"] = []
        payload["provenance"] = None
    return payload


def dev_base_rate_brier(
    segments: Sequence[Mapping[str, Any]],
    pooled_fit: Mapping[str, Any],
) -> dict[str, Any]:
    """Calculate empirical pooled-DEV Brier for the frozen constant predictor."""
    if pooled_fit.get("status") != "computed":
        return {
            "status": "not_testable",
            "reason": "pooled_dev_fit_not_computed",
            "n_rows": 0,
            "brier": None,
        }
    allowed_pids = {str(int(pid)) for pid in FROZEN_PID_SPLIT["DEV"]}
    for segment in segments:
        try:
            pid = str(int(float(segment.get("pid"))))
        except (TypeError, ValueError, OverflowError):
            return {
                "status": "not_testable",
                "reason": "dev_base_rate_pid_invalid",
                "n_rows": 0,
                "brier": None,
            }
        if pid not in allowed_pids or segment.get("split") != "DEV":
            return {
                "status": "not_testable",
                "reason": "dev_base_rate_segment_not_in_dev_split",
                "n_rows": 0,
                "brier": None,
            }
    rows = resolved_scoring_rows(
        segments,
        pooled_fit["q_grid"],
        pooled_fit["query_grid"],
        split="DEV",
    )
    rows = [dict(row, prediction=P_REM_DEV) for row in rows]
    metrics = binary_metrics(rows)
    if metrics["brier"] is None or not np.isfinite(float(metrics["brier"])):
        return {
            "status": "not_testable",
            "reason": "dev_base_rate_brier_not_computable",
            "n_rows": int(metrics["n"]),
            "brier": None,
        }
    return {
        "status": "computed",
        "reason": None,
        "n_rows": int(metrics["n"]),
        "n3_rows": int(sum(row["outcome"] == "first_hit_n3" for row in rows)),
        "rem_rows": int(sum(row["outcome"] == "first_hit_rem" for row in rows)),
        "brier": float(metrics["brier"]),
        "log_loss": float(metrics["log_loss"]),
        "rem_rate": float(metrics["rem_rate"]),
        "prediction": P_REM_DEV,
        "resolved_outcomes": sorted(RESOLVED_OUTCOMES),
    }


def calculate_transfer_tolerance(dev_base_brier: float | None) -> float | None:
    """Freeze the inherited S2 tolerance from the empirical DEV Brier."""
    if dev_base_brier is None or not np.isfinite(float(dev_base_brier)):
        return None
    return float(max(0.02, 0.25 * float(dev_base_brier)))

