"""Synthetic-only M2/M3 gate for NMD-QC-FLOAT-001 design-0.3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.dynamical_families.nmd_qc_float import (  # noqa: E402
    TECHNICALLY_ADMISSIBLE,
    TECHNICAL_INVALID,
    TECHNICAL_STATUS_UNRESOLVED,
    RULE_MANIFEST_PATH,
    RULE_MANIFEST_SHA256,
    audit_exported_float,
)


PROTOCOL_ID = "NMD-QC-FLOAT-001"
GATE_ID = "NMD-QC-FLOAT-001-M1-M2-M3-SYNTHETIC"
FIXTURE_GENERATOR = "od_nmd_qc_float_001_synthetic_gate.v1"

FIXTURE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "clean_float64",
        "class": "clean_complete",
        "expected_status": TECHNICALLY_ADMISSIBLE,
        "n_samples": 500,
        "sampling_frequency": 250.0,
        "signal_dtype": "float64",
        "time_dtype": "float64",
        "time_origin": 0.0,
    },
    {
        "id": "clean_float32_time_origin_1000",
        "class": "clean_complete",
        "expected_status": TECHNICALLY_ADMISSIBLE,
        "n_samples": 500,
        "sampling_frequency": 250.0,
        "signal_dtype": "float32",
        "time_dtype": "float32",
        "time_origin": 1000.0,
    },
    {
        "id": "nonfinite_sample",
        "class": "in_scope_defect",
        "expected_status": TECHNICAL_INVALID,
        "defect": "R1",
        "n_samples": 500,
        "sampling_frequency": 250.0,
        "signal_dtype": "float64",
        "time_dtype": "float64",
        "time_origin": 0.0,
    },
    {
        "id": "flatline_one_second",
        "class": "in_scope_defect",
        "expected_status": TECHNICAL_INVALID,
        "defect": "R2",
        "n_samples": 500,
        "sampling_frequency": 250.0,
        "signal_dtype": "float64",
        "time_dtype": "float64",
        "time_origin": 0.0,
    },
    {
        "id": "extremum_plateau_short_segment",
        "class": "in_scope_defect",
        "expected_status": TECHNICAL_INVALID,
        "defect": "R3",
        "n_samples": 100,
        "sampling_frequency": 250.0,
        "signal_dtype": "float64",
        "time_dtype": "float64",
        "time_origin": 0.0,
    },
    {
        "id": "quantization_eight_levels",
        "class": "in_scope_defect",
        "expected_status": TECHNICAL_INVALID,
        "defect": "R4",
        "n_samples": 500,
        "sampling_frequency": 250.0,
        "signal_dtype": "float64",
        "time_dtype": "float64",
        "time_origin": 0.0,
    },
    {
        "id": "duplicate_timestamp",
        "class": "in_scope_defect",
        "expected_status": TECHNICAL_INVALID,
        "defect": "R5",
        "n_samples": 500,
        "sampling_frequency": 250.0,
        "signal_dtype": "float64",
        "time_dtype": "float64",
        "time_origin": 0.0,
    },
    {
        "id": "large_origin_float32_time",
        "class": "support_limited",
        "expected_status": TECHNICAL_STATUS_UNRESOLVED,
        "defect": "R5_representation",
        "n_samples": 4000,
        "sampling_frequency": 2000.0,
        "signal_dtype": "float32",
        "time_dtype": "float32",
        "time_origin": 100000.0,
    },
    {
        "id": "caller_declared_uncovered_gap",
        "class": "support_limited",
        "expected_status": TECHNICAL_STATUS_UNRESOLVED,
        "defect": "coverage",
        "n_samples": 600,
        "sampling_frequency": 250.0,
        "signal_dtype": "float64",
        "time_dtype": "float64",
        "time_origin": 0.0,
        "segment_boundaries": [[0, 250], [300, 550]],
    },
    {
        "id": "missing_channel_labels",
        "class": "support_limited",
        "expected_status": TECHNICAL_STATUS_UNRESOLVED,
        "defect": "R8",
        "n_samples": 500,
        "sampling_frequency": 250.0,
        "signal_dtype": "float64",
        "time_dtype": "float64",
        "time_origin": 0.0,
    },
    {
        "id": "unsupported_signal_dtype",
        "class": "support_limited",
        "expected_status": TECHNICAL_STATUS_UNRESOLVED,
        "defect": "input_dtype",
        "n_samples": 500,
        "sampling_frequency": 250.0,
        "signal_dtype": "int16",
        "time_dtype": "float64",
        "time_origin": 0.0,
    },
    {
        "id": "malformed_segment_boundaries",
        "class": "support_limited",
        "expected_status": TECHNICAL_STATUS_UNRESOLVED,
        "defect": "segments",
        "n_samples": 500,
        "sampling_frequency": 250.0,
        "signal_dtype": "float64",
        "time_dtype": "float64",
        "time_origin": 0.0,
        "segment_boundaries": [[0, 250], [200, 400]],
    },
    {
        "id": "out_of_scope_single_sample_clip",
        "class": "out_of_scope_stress",
        "expected_status": TECHNICALLY_ADMISSIBLE,
        "defect": "clipping_without_R3",
        "n_samples": 500,
        "sampling_frequency": 250.0,
        "signal_dtype": "float64",
        "time_dtype": "float64",
        "time_origin": 0.0,
    },
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def fixture_manifest_sha256() -> str:
    return hashlib.sha256(_canonical_json(FIXTURE_SPECS)).hexdigest()


def _base_fixture(spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    n_samples = int(spec["n_samples"])
    fs = float(spec["sampling_frequency"])
    signal_dtype = np.dtype(spec["signal_dtype"])
    time_dtype = np.dtype(spec["time_dtype"])
    origin = float(spec["time_origin"])
    time = (
        np.arange(n_samples, dtype=time_dtype) / np.asarray(fs, dtype=time_dtype)
        + np.asarray(origin, dtype=time_dtype)
    )
    phase_time = np.arange(n_samples, dtype=np.float64) / fs
    rng = np.random.default_rng(90210 + n_samples)
    noise = rng.normal(0.0, 0.01, size=(2, n_samples))
    signal = np.vstack(
        (
            np.sin(2.0 * np.pi * 10.0 * phase_time),
            0.4 * np.cos(2.0 * np.pi * 6.0 * phase_time),
        )
    )
    return (signal + noise).astype(signal_dtype), time


def build_fixture(spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    signal, time = _base_fixture(spec)
    fixture_id = str(spec["id"])
    extra: dict[str, Any] = {}
    if fixture_id == "nonfinite_sample":
        signal[0, 0] = np.nan
    elif fixture_id == "flatline_one_second":
        signal[0, :250] = np.float64(0.25)
    elif fixture_id == "extremum_plateau_short_segment":
        signal[0, :30] = np.float64(5.0)
        signal[0, 30:60] = np.float64(-5.0)
    elif fixture_id == "quantization_eight_levels":
        levels = (np.arange(signal.shape[1]) % 8) / 8.0
        signal[:] = np.vstack((levels, levels))
    elif fixture_id == "duplicate_timestamp":
        time[250] = time[249]
    elif fixture_id == "out_of_scope_single_sample_clip":
        signal[0, 10] = np.float64(2.0)
        extra["out_of_scope_stress"] = "clipping_without_R3"
    return signal, time, extra


def _runtime_identity() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def run_single() -> dict[str, Any]:
    manifest_hash = fixture_manifest_sha256()
    case_results: list[dict[str, Any]] = []
    for spec in FIXTURE_SPECS:
        signal, time, extra = build_fixture(dict(spec))
        common: dict[str, Any] = {
            "sampling_frequency": spec["sampling_frequency"],
            "fixture_manifest_sha256": manifest_hash,
            "runtime_identity": _runtime_identity(),
            "input_provenance": {
                "generator": FIXTURE_GENERATOR,
                "fixture_id": spec["id"],
            },
        }
        if spec["id"] == "missing_channel_labels":
            common.update(
                required_channel_names=["C1", "C2"],
                channel_selection_provenance={"source": "synthetic"},
            )
        else:
            common.update(
                required_channel_indices=[0, 1],
                channel_selection_provenance={"source": "synthetic"},
            )
        if "segment_boundaries" in spec:
            common["segment_boundaries"] = spec["segment_boundaries"]
        result = audit_exported_float(signal, time, **common)
        case_results.append(
            {
                "fixture_id": spec["id"],
                "fixture_class": spec["class"],
                "defect": spec.get("defect"),
                "expected_status": spec["expected_status"],
                "observed_status": result["recording_status"],
                "exact_status_match": (
                    result["recording_status"] == spec["expected_status"]
                ),
                "invalid_reasons": result["invalid_reasons"],
                "unresolved_reasons": result["unresolved_reasons"],
                "timebase_status": result["timebase_status"],
                "input_hash": result["input_hash"],
                "determinism_hash": result["determinism_hash"],
                **extra,
            }
        )
    exact_status = all(case["exact_status_match"] for case in case_results)
    return {
        "protocol_id": PROTOCOL_ID,
        "gate_id": GATE_ID,
        "rule_manifest_path": RULE_MANIFEST_PATH,
        "rule_manifest_sha256": RULE_MANIFEST_SHA256,
        "fixture_generator": FIXTURE_GENERATOR,
        "fixture_manifest_sha256": manifest_hash,
        "m2_exact_status": exact_status,
        "cases": case_results,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_gate(output_path: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="nmd_qc_float_m3_") as directory:
        runs: list[dict[str, Any]] = []
        for index in range(3):
            run_path = Path(directory) / f"run_{index}.json"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--single-run",
                    "--output-json",
                    str(run_path),
                ],
                check=True,
            )
            runs.append(json.loads(run_path.read_text(encoding="utf-8")))

    m2_exact = all(bool(run["m2_exact_status"]) for run in runs)
    canonical_runs = [_canonical_json(run) for run in runs]
    m3_deterministic = len(set(canonical_runs)) == 1
    result = {
        "protocol_id": PROTOCOL_ID,
        "gate_id": GATE_ID,
        "scope": "synthetic_only",
        "real_data_accessed": False,
        "ds003670_accessed": False,
        "rule_manifest_path": RULE_MANIFEST_PATH,
        "rule_manifest_sha256": RULE_MANIFEST_SHA256,
        "fixture_manifest_sha256": runs[0]["fixture_manifest_sha256"],
        "m1_unit_implementation": "COMPLETE",
        "m2_exact_status": "PASS" if m2_exact else "FAIL",
        "m3_deterministic_reexecution": (
            "PASS" if m3_deterministic else "FAIL"
        ),
        "gate_status": "PASS" if m2_exact and m3_deterministic else "FAIL",
        "m3_run_hashes": [
            hashlib.sha256(payload).hexdigest() for payload in canonical_runs
        ],
        "case_count": len(runs[0]["cases"]),
        "cases": runs[0]["cases"],
        "closures": {
            "M4": "NOT_AUTHORIZED",
            "real_EEG": "NOT_AUTHORIZED",
            "ds003670": "FORBIDDEN",
            "FAR": "CLOSED",
            "MNPS": "CLOSED",
            "reconciliation": "CLOSED",
        },
    }
    _write_json(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-run", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.single_run:
        _write_json(args.output_json, run_single())
    else:
        result = run_gate(args.output_json)
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
