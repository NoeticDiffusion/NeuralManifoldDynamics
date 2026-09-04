"""FAR-000 inheritance and contract-lock gate.

This gate is deliberately below the empirical dataset boundary.  It replays
the existing OD-TQ3 translation tests, records the current local translation
surface, and audits the external O3 provenance without rerunning O3.

The gate is fail-closed: an O3 result/archive contradiction or seed-provenance
mismatch blocks FAR-001.  It never reads an empirical dataset, writes a
production HDF5 artifact, or interprets stimulation frequency as amplitude.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FAR_000_PROTOCOL_ID = "FAR-000"
FAR_000_SCHEMA = "mndm.far_000_inheritance.v1"
O3_HELD_OUT_SEEDS = [721, 732, 743, 754, 765, 776, 787, 798, 809, 820]
O3_AMENDED_DEV_SEEDS = list(range(7300, 7320))
REQUIRED_PROTOCOL_KEYS = (
    "perturbation_direction",
    "perturbation_time",
    "reference_attractor_or_state",
    "return_criterion",
    "escape_criterion",
    "observation_horizon",
    "non_return_is_escape",
)

LOCAL_LOCK_FILES = (
    "project/orthagonal_axis/od_tq3_preregistration.md",
    "mndm/tests/od_tq3_fixture.py",
    "mndm/tests/test_od_tq3_resilience_adapter.py",
    "mndm/src/mndm/dynamical_families/resilience.py",
    "mndm/src/mndm/dynamical_families/contracts.py",
    "mndm/src/mndm/pipeline/dynamical_families_export.py",
    "mndm/src/mndm/pipeline/summary.py",
    "core/src/core/io/h5_writer.py",
    "mndm/config/config_ingest_common_dynamical_families.yaml",
    "project/seed_registry.yaml",
)

O3_FILES = {
    "effect_summary": Path(
        "results/orthogonal_dynamics_stage_o3_gate/effect_summary.json"
    ),
    "preregistration": Path(
        "project/orthagonal_dynamics/017_stage_o3_preregistration.md"
    ),
    "program_archive": Path(
        "project/orthagonal_dynamics/015_o1_o3_program_archive.md"
    ),
    "dev_calibration": Path(
        "results/orthogonal_dynamics_stage_o3_dev_calibration/dev.json"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_fingerprints(repo_root: Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    missing: list[str] = []
    for relative in LOCAL_LOCK_FILES:
        path = repo_root / relative
        record: dict[str, Any] = {
            "path": relative,
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        records[relative] = record
        if not path.is_file():
            missing.append(relative)
    return {
        "status": "PASS" if not missing else "BLOCKED",
        "records": records,
        "missing": missing,
    }


def _default_config_semantics(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "mndm/config/config_ingest_common_dynamical_families.yaml"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    far_section = text.split("  resilience:", 1)
    far_text = far_section[1] if len(far_section) == 2 else ""
    checks = {
        "root_enabled_false": "  enabled: false" in text,
        "far_enabled_false": "    enabled: false" in far_text,
        "far_qualified_false": "      qualified: false" in far_text,
        "far_protocol_source_null": "    protocol_source: null" in far_text,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }


def _fixture_semantics(repo_root: Path) -> dict[str, Any]:
    tests_root = repo_root / "mndm" / "tests"
    if str(tests_root) not in sys.path:
        sys.path.insert(0, str(tests_root))
    fixture = importlib.import_module("od_tq3_fixture")
    payload = fixture.truth_outcomes()
    amplitudes = payload["amplitudes"]
    returned = payload["returned"]
    unique_amplitudes = [float(value) for value in sorted(set(amplitudes.tolist()))]
    return_fractions = [
        float(returned[amplitudes == value].mean())
        for value in unique_amplitudes
    ]
    expected_fractions = [1.0, 0.6, 0.3]
    checks = {
        "qualification_id": fixture.QUALIFICATION_ID,
        "legacy_declared_contract_hash": fixture.QUALIFICATION_HASH,
        "schema_version": "mndm.finite_amplitude_resilience.v1",
        "estimator": "observed_perturbation_outcome_summary",
        "min_trials_per_amplitude": 20,
        "amplitudes": unique_amplitudes,
        "return_fractions": return_fractions,
        "expected_return_fractions": expected_fractions,
        "r50_discrete": unique_amplitudes[
            next(
                index
                for index, value in enumerate(return_fractions)
                if value <= 0.5
            )
        ],
        "protocol": dict(fixture.PROTOCOL),
        "direction_semantics": fixture.PROTOCOL.get("perturbation_direction"),
        "non_return_is_escape": fixture.PROTOCOL.get("non_return_is_escape"),
        "required_protocol_keys": list(REQUIRED_PROTOCOL_KEYS),
        "protocol_keys_present": all(
            key in fixture.PROTOCOL for key in REQUIRED_PROTOCOL_KEYS
        ),
        "qualification_metadata_complete": bool(
            fixture.QUALIFICATION_ID and fixture.QUALIFICATION_HASH
        ),
    }
    checks["status"] = (
        "PASS"
        if checks["amplitudes"] == [0.0, 1.0, 2.0]
        and all(
            abs(actual - expected) <= 1e-12
            for actual, expected in zip(return_fractions, expected_fractions)
        )
        and checks["r50_discrete"] == 2.0
        and checks["non_return_is_escape"] is True
        and checks["protocol_keys_present"]
        and checks["qualification_metadata_complete"]
        else "FAIL"
    )
    return checks


def _run_translation_tests(repo_root: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "mndm/tests/test_od_tq3_resilience_adapter.py",
        "mndm/tests/test_od_v1_integration_audit.py",
        "-q",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "status": "BLOCKED",
            "returncode": None,
            "command": command,
            "error": f"{type(error).__name__}:{error}",
        }
    return {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "returncode": completed.returncode,
        "command": command,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, f"{type(error).__name__}:{error}"
    if not isinstance(value, dict):
        return None, "json_root_is_not_object"
    return value, None


def _markdown_section(text: str, heading: str) -> str:
    marker = f"## {heading}"
    start = text.find(marker)
    if start < 0:
        return ""
    remainder = text[start + len(marker) :]
    next_heading = remainder.find("\n## ")
    return remainder if next_heading < 0 else remainder[:next_heading]


def _audit_o3_inheritance(simulator_root: Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    missing: list[str] = []
    for name, relative in O3_FILES.items():
        path = simulator_root / relative
        record: dict[str, Any] = {
            "path": str(path),
            "relative_path": relative.as_posix(),
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        records[name] = record
        if not path.is_file():
            missing.append(name)

    reasons: list[str] = []
    effect: dict[str, Any] | None = None
    dev: dict[str, Any] | None = None
    parse_errors: dict[str, str] = {}
    effect_path = simulator_root / O3_FILES["effect_summary"]
    dev_path = simulator_root / O3_FILES["dev_calibration"]
    if effect_path.is_file():
        effect, error = _read_json(effect_path)
        if error is not None:
            parse_errors["effect_summary"] = error
    if dev_path.is_file():
        dev, error = _read_json(dev_path)
        if error is not None:
            parse_errors["dev_calibration"] = error
    if missing:
        reasons.extend(f"missing_o3_artifact:{name}" for name in missing)
    if parse_errors:
        reasons.extend(f"unreadable_o3_artifact:{name}" for name in parse_errors)

    effect_statuses = {
        "o3_a": (effect or {}).get("o3_a", {}).get("status"),
        "o3_b": (effect or {}).get("o3_b", {}).get("status"),
    }
    effect_pass = effect_statuses == {"o3_a": "pass", "o3_b": "pass"}
    if effect is not None and not effect_pass:
        reasons.append("o3_effect_summary_not_pass")

    prereg_path = simulator_root / O3_FILES["preregistration"]
    archive_path = simulator_root / O3_FILES["program_archive"]
    prereg_text = prereg_path.read_text(encoding="utf-8") if prereg_path.is_file() else ""
    archive_text = archive_path.read_text(encoding="utf-8") if archive_path.is_file() else ""
    o3_archive_section = _markdown_section(archive_text, "O3 status")
    prereg_amended = "7300..7319" in prereg_text
    archive_claims_blocked = "**Not started; blocked.**" in o3_archive_section
    archive_claims_pass = "**PASS" in o3_archive_section
    if prereg_path.is_file() and not prereg_amended:
        reasons.append("o3_preregistration_missing_amended_dev_pool")
    if archive_path.is_file() and not o3_archive_section:
        reasons.append("o3_archive_status_section_missing")
    if archive_path.is_file() and not archive_claims_blocked and not archive_claims_pass:
        reasons.append("o3_archive_status_unrecognized")
    if effect_pass and archive_claims_blocked:
        reasons.append("o3_archive_contradiction")
    if archive_claims_pass and not effect_pass:
        reasons.append("o3_effect_archive_status_mismatch")

    actual_dev_seeds = (dev or {}).get("dev_seeds")
    if actual_dev_seeds != O3_AMENDED_DEV_SEEDS:
        reasons.append("o3_dev_seed_provenance_mismatch")
    actual_held_out = (dev or {}).get("held_out_reserved")
    if actual_held_out != O3_HELD_OUT_SEEDS:
        reasons.append("o3_held_out_seed_provenance_mismatch")
    effect_held_out = (effect or {}).get("held_out_seeds")
    if effect_held_out != O3_HELD_OUT_SEEDS:
        reasons.append("o3_effect_held_out_seed_provenance_mismatch")

    effect_preregistration = (effect or {}).get("preregistration")
    effect_preregistration_matches = effect_preregistration == (
        O3_FILES["preregistration"].as_posix()
    )
    if effect is not None and not effect_preregistration_matches:
        reasons.append("o3_effect_preregistration_pointer_mismatch")

    return {
        "status": "PASS" if not reasons else "BLOCKED",
        "simulator_root": str(simulator_root),
        "records": records,
        "checks": {
            "effect_statuses": effect_statuses,
            "effect_summary_pass": effect_pass,
            "preregistration_has_amended_dev_pool": prereg_amended,
            "archive_claims_blocked": archive_claims_blocked,
            "archive_claims_pass": archive_claims_pass,
            "archive_o3_section_present": bool(o3_archive_section),
            "effect_preregistration_matches": effect_preregistration_matches,
            "effect_held_out_seeds": effect_held_out,
            "dev_seeds": actual_dev_seeds,
            "expected_dev_seeds": O3_AMENDED_DEV_SEEDS,
            "held_out_seeds": actual_held_out,
            "expected_held_out_seeds": O3_HELD_OUT_SEEDS,
        },
        "parse_errors": parse_errors,
        "reasons": reasons,
    }


def _default_simulator_root() -> Path:
    return Path(
        os.environ.get("NMD_NDT_SIMULATOR_ROOT", r"J:\repos\ndt-simulator")
    )


def run_gate(
    output_path: Path,
    *,
    repo_root: Path = ROOT,
    simulator_root: Path | None = None,
    test_runner: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run FAR-000 once and write a non-overwritable JSON certificate."""
    if output_path.exists():
        raise FileExistsError(f"refusing_to_overwrite:{output_path}")
    simulator_root = simulator_root or _default_simulator_root()
    local_fingerprints = _local_fingerprints(repo_root)
    semantics = _fixture_semantics(repo_root)
    default_config = _default_config_semantics(repo_root)
    translation_replay = (
        test_runner(repo_root)
        if test_runner is not None
        else _run_translation_tests(repo_root)
    )
    translation_status = (
        "PASS"
        if local_fingerprints["status"] == "PASS"
        and semantics["status"] == "PASS"
        and default_config["status"] == "PASS"
        and translation_replay["status"] == "PASS"
        else "FAIL"
    )
    o3 = _audit_o3_inheritance(simulator_root)
    reasons = list(o3["reasons"])
    if translation_status != "PASS":
        reasons.append("od_tq3_translation_replay_not_pass")
    overall_status = (
        "FAIL"
        if translation_status != "PASS"
        else "PASS"
        if o3["status"] == "PASS"
        else "BLOCKED"
    )
    result: dict[str, Any] = {
        "schema": FAR_000_SCHEMA,
        "protocol_id": FAR_000_PROTOCOL_ID,
        "status": overall_status,
        "translation_surface": {
            "status": translation_status,
            "local_fingerprints": local_fingerprints,
            "fixture_semantics": semantics,
            "default_config_semantics": default_config,
            "test_replay": translation_replay,
        },
        "o3_inheritance": o3,
        "decision": {
            "overall_status": overall_status,
            "reasons": reasons,
            "far_001_authorized": overall_status == "PASS",
            "next_gate": "FAR-001"
            if overall_status == "PASS"
            else "FAR-000 remediation",
        },
        "fail_closed_assertions": {
            "empirical_dataset_read": False,
            "far_001_source_scout_run": False,
            "frequency_treated_as_amplitude": False,
            "o3_rerun": False,
            "production_hdf5_written": False,
            "held_out_or_reserve_empirical_data_read": False,
            "spontaneous_excursion_used_as_perturbation": False,
        },
        "claim_boundary": (
            "FAR-000 locks or blocks inheritance of the existing OD-TQ3 "
            "translation surface and external simulator provenance. It does "
            "not qualify an empirical perturbation source, define home/away "
            "regions, estimate R(rho), or authorize production FAR output."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulator-root", type=Path, default=None)
    args = parser.parse_args(argv)
    result = run_gate(
        args.output,
        simulator_root=args.simulator_root,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": result["status"],
                "far_001_authorized": result["decision"]["far_001_authorized"],
                "reasons": result["decision"]["reasons"],
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
