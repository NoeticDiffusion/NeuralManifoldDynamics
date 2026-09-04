"""Synthetic coverage-completion gate for NMD-QC-FLOAT-001 M2B."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import ceil
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.dynamical_families.nmd_qc_float import (  # noqa: E402
    RULE_MANIFEST_PATH,
    RULE_MANIFEST_SHA256,
    TECHNICALLY_ADMISSIBLE,
    TECHNICAL_INVALID,
    TECHNICAL_STATUS_UNRESOLVED,
    audit_exported_float,
)


PROTOCOL_ID = "NMD-QC-FLOAT-001"
GATE_ID = "NMD-QC-FLOAT-001-M2B-SYNTHETIC-COVERAGE"
FIXTURE_GENERATOR = "od_nmd_qc_float_001_m2b_synthetic_gate.v1"
CHANNEL_LABELS = [f"C{index + 1}" for index in range(32)]


def _case(
    case_id: str,
    family: str,
    expected_status: str,
    *,
    expected_rules: dict[str, str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "family": family,
        "expected_status": expected_status,
        "expected_rules": expected_rules or {},
        **kwargs,
    }


FIXTURE_SPECS: tuple[dict[str, Any], ...] = (
    _case(
        "r6_zero_finite",
        "R6",
        TECHNICAL_INVALID,
        expected_rules={"R6": TECHNICAL_INVALID, "R7": TECHNICAL_INVALID},
        defect="all_nonfinite",
        n_samples=250,
        fs=250.0,
        n_channels=8,
        signal_dtype="float64",
        time_dtype="float64",
    ),
    _case(
        "r6_one_sample_below_one_second",
        "R6",
        TECHNICAL_INVALID,
        expected_rules={"R6": TECHNICAL_STATUS_UNRESOLVED},
        defect="finite_support_249",
        n_samples=250,
        fs=250.0,
        n_channels=8,
        signal_dtype="float64",
        time_dtype="float64",
    ),
    _case(
        "r6_exactly_one_second",
        "R6",
        TECHNICALLY_ADMISSIBLE,
        expected_rules={"R6": TECHNICALLY_ADMISSIBLE},
        defect="none",
        n_samples=250,
        fs=250.0,
        n_channels=8,
        signal_dtype="float64",
        time_dtype="float64",
    ),
    _case(
        "r7_empty_nonfinite_only",
        "R7",
        TECHNICAL_INVALID,
        expected_rules={"R7": TECHNICAL_INVALID},
        defect="all_nonfinite",
        n_samples=250,
        fs=250.0,
        n_channels=8,
        signal_dtype="float64",
        time_dtype="float64",
    ),
    _case(
        "r7_single_valued_below_one_second",
        "R7",
        TECHNICAL_INVALID,
        expected_rules={"R7": TECHNICAL_STATUS_UNRESOLVED},
        defect="single_valued",
        n_samples=249,
        fs=250.0,
        n_channels=8,
        signal_dtype="float64",
        time_dtype="float64",
    ),
    _case(
        "r7_single_valued_exactly_one_second",
        "R7",
        TECHNICAL_INVALID,
        expected_rules={"R7": TECHNICAL_INVALID},
        defect="single_valued",
        n_samples=250,
        fs=250.0,
        n_channels=8,
        signal_dtype="float64",
        time_dtype="float64",
    ),
    _case(
        "r7_single_valued_above_one_second",
        "R7",
        TECHNICAL_INVALID,
        expected_rules={"R7": TECHNICAL_INVALID},
        defect="single_valued",
        n_samples=500,
        fs=250.0,
        n_channels=8,
        signal_dtype="float64",
        time_dtype="float64",
    ),
    _case(
        "r7_legitimate_near_zero_nonzero_range",
        "R7",
        TECHNICALLY_ADMISSIBLE,
        expected_rules={"R7": TECHNICALLY_ADMISSIBLE},
        defect="near_zero_nonzero",
        n_samples=500,
        fs=250.0,
        n_channels=8,
        signal_dtype="float64",
        time_dtype="float64",
    ),
    *tuple(
        _case(
            f"r2_run_{label}",
            "R2",
            TECHNICAL_INVALID,
            expected_rules={
                "R2": (
                    TECHNICAL_INVALID
                    if run_length >= 250
                    else TECHNICALLY_ADMISSIBLE
                ),
                "R4": TECHNICAL_INVALID,
            },
            defect="flatline_run",
            run_length=run_length,
            n_samples=2000,
            fs=250.0,
            n_channels=8,
            signal_dtype="float64",
            time_dtype="float64",
        )
        for label, run_length in (
            ("one_below", 249),
            ("exact", 250),
            ("one_above", 251),
        )
    ),
    *tuple(
        _case(
            f"r3_plateau_{label}",
            "R3",
            TECHNICALLY_ADMISSIBLE if plateau_length < 25 else TECHNICAL_INVALID,
            expected_rules={
                "R3": (
                    TECHNICAL_INVALID
                    if plateau_length >= 25
                    else TECHNICALLY_ADMISSIBLE
                )
            },
            defect="extremum_plateau",
            plateau_length=plateau_length,
            n_samples=500,
            fs=250.0,
            n_channels=8,
            signal_dtype="float64",
            time_dtype="float64",
        )
        for label, plateau_length in (
            ("one_below", 24),
            ("exact", 25),
            ("one_above", 26),
        )
    ),
    _case(
        "r4_eight_distinct_levels",
        "R4",
        TECHNICAL_INVALID,
        expected_rules={"R4": TECHNICAL_INVALID},
        defect="quantized",
        levels=8,
        n_samples=500,
        fs=250.0,
        n_channels=8,
        signal_dtype="float64",
        time_dtype="float64",
    ),
    _case(
        "r4_nine_distinct_levels",
        "R4",
        TECHNICALLY_ADMISSIBLE,
        expected_rules={"R4": TECHNICALLY_ADMISSIBLE},
        defect="quantized",
        levels=9,
        n_samples=500,
        fs=250.0,
        n_channels=8,
        signal_dtype="float64",
        time_dtype="float64",
    ),
    _case(
        "r4_incomplete_window",
        "R4",
        TECHNICAL_STATUS_UNRESOLVED,
        expected_rules={"R4": TECHNICAL_STATUS_UNRESOLVED},
        defect="quantized",
        levels=9,
        n_samples=249,
        fs=250.0,
        n_channels=8,
        signal_dtype="float64",
        time_dtype="float64",
    ),
    _case(
        "partial_channel_nonrequired_nan",
        "aggregation",
        TECHNICALLY_ADMISSIBLE,
        expected_rules={"R1": TECHNICALLY_ADMISSIBLE},
        defect="nonrequired_nan",
        n_samples=500,
        fs=250.0,
        n_channels=8,
        required_channel_indices=[0],
        signal_dtype="float64",
        time_dtype="float64",
    ),
    _case(
        "partial_channel_required_nan",
        "aggregation",
        TECHNICAL_INVALID,
        expected_rules={"R1": TECHNICAL_INVALID},
        expected_rule_channels={"R1": 1},
        defect="required_nan",
        n_samples=500,
        fs=250.0,
        n_channels=8,
        required_channel_indices=[0, 1],
        signal_dtype="float64",
        time_dtype="float64",
    ),
    _case(
        "invalid_over_unresolved",
        "aggregation",
        TECHNICAL_INVALID,
        defect="required_nan_missing_provenance",
        n_samples=500,
        fs=250.0,
        n_channels=8,
        required_channel_indices=[0, 1],
        signal_dtype="float64",
        time_dtype="float64",
    ),
    _case(
        "valid_supplied_segments",
        "segments",
        TECHNICALLY_ADMISSIBLE,
        defect="none",
        n_samples=500,
        fs=250.0,
        n_channels=8,
        segment_boundaries=[[0, 250], [250, 500]],
        signal_dtype="float64",
        time_dtype="float64",
    ),
    _case(
        "valid_segments_with_uncovered_support",
        "segments",
        TECHNICAL_STATUS_UNRESOLVED,
        defect="none",
        n_samples=500,
        fs=250.0,
        n_channels=8,
        segment_boundaries=[[0, 250], [300, 500]],
        signal_dtype="float64",
        time_dtype="float64",
    ),
)


_R5_EXPECTED_STATUS = {
    "float32": {
        "0": TECHNICALLY_ADMISSIBLE,
        "1000": TECHNICALLY_ADMISSIBLE,
        "100000": TECHNICAL_STATUS_UNRESOLVED,
        "1000000000": TECHNICAL_STATUS_UNRESOLVED,
        "10000000000000": TECHNICAL_STATUS_UNRESOLVED,
    },
    "float64": {
        "0": TECHNICALLY_ADMISSIBLE,
        "1000": TECHNICALLY_ADMISSIBLE,
        "100000": TECHNICALLY_ADMISSIBLE,
        "1000000000": TECHNICALLY_ADMISSIBLE,
        "10000000000000": TECHNICAL_STATUS_UNRESOLVED,
    },
}


def _r5_grid_specs() -> tuple[dict[str, Any], ...]:
    cases: list[dict[str, Any]] = []
    origins = (0.0, 1e3, 1e5, 1e9, 1e13)
    for time_dtype in ("float32", "float64"):
        for signal_dtype in ("float32", "float64"):
            for origin in origins:
                for fs in (250.0, 500.0, 1000.0, 2000.0):
                    origin_key = f"{origin:.0f}"
                    cases.append(
                        _case(
                            (
                                f"r5_grid_{signal_dtype}_{time_dtype}_"
                                f"origin_{origin:g}_fs_{fs:g}"
                            ),
                            "R5_grid",
                            _R5_EXPECTED_STATUS[time_dtype][origin_key],
                            defect="time_origin_grid",
                            n_samples=int(ceil(fs * 2.0)),
                            fs=fs,
                            n_channels=8,
                            required_channel_indices=list(range(8)),
                            signal_dtype=signal_dtype,
                            time_dtype=time_dtype,
                            time_origin=origin,
                        )
                    )
    return tuple(cases)


def _full_grid_specs() -> tuple[dict[str, Any], ...]:
    cases: list[dict[str, Any]] = []
    for fs in (250.0, 500.0, 1000.0, 2000.0):
        for duration in (10.0, 60.0):
            for n_channels in (8, 32):
                for signal_dtype in ("float32", "float64"):
                    for time_dtype in ("float32", "float64"):
                        cases.append(
                            _case(
                                (
                                    f"grid_fs_{fs:g}_duration_{duration:g}_"
                                    f"channels_{n_channels}_{signal_dtype}_"
                                    f"{time_dtype}"
                                ),
                                "grid",
                                TECHNICALLY_ADMISSIBLE,
                                defect="none",
                                n_samples=int(ceil(fs * duration)),
                                fs=fs,
                                n_channels=n_channels,
                                required_channel_indices=list(
                                    range(n_channels)
                                ),
                                signal_dtype=signal_dtype,
                                time_dtype=time_dtype,
                            )
                        )
    return tuple(cases)

FIXTURE_SPECS = FIXTURE_SPECS + _r5_grid_specs() + _full_grid_specs()


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
    fs = float(spec["fs"])
    signal_dtype = np.dtype(spec["signal_dtype"])
    time_dtype = np.dtype(spec["time_dtype"])
    origin = np.asarray(spec.get("time_origin", 0.0), dtype=time_dtype)
    time = (
        np.arange(n_samples, dtype=time_dtype) / np.asarray(fs, dtype=time_dtype)
        + origin
    )
    phase_time = np.arange(n_samples, dtype=np.float64) / fs
    seed = int.from_bytes(
        hashlib.sha256(str(spec["id"]).encode("utf-8")).digest()[:4],
        "little",
    )
    rng = np.random.default_rng(seed)
    base = np.vstack(
        (
            np.sin(2.0 * np.pi * 10.0 * phase_time),
            0.4 * np.cos(2.0 * np.pi * 6.0 * phase_time),
        )
    )
    n_channels = int(spec["n_channels"])
    signal = np.empty((n_channels, n_samples), dtype=np.float64)
    for channel_index in range(n_channels):
        component = base[channel_index % 2]
        signal[channel_index] = component + rng.normal(
            0.0,
            0.01 + 0.001 * channel_index,
            size=n_samples,
        )
    return signal.astype(signal_dtype), time


def build_fixture(
    spec: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    signal, time = _base_fixture(spec)
    defect = spec.get("defect")
    if defect == "all_nonfinite":
        signal[0, :] = np.nan
    elif defect == "finite_support_249":
        signal[0, 249:] = np.nan
    elif defect == "single_valued":
        signal[0, :] = np.asarray(0.25, dtype=signal.dtype)
    elif defect == "near_zero_nonzero":
        signal[0, :] = np.asarray(
            1e-12 + np.arange(signal.shape[1], dtype=np.float64) * 1e-15,
            dtype=signal.dtype,
        )
    elif defect == "flatline_run":
        signal[0, : int(spec["run_length"])] = np.asarray(
            0.25,
            dtype=signal.dtype,
        )
    elif defect == "extremum_plateau":
        signal[0, : int(spec["plateau_length"])] = np.asarray(
            5.0,
            dtype=signal.dtype,
        )
    elif defect == "quantized":
        levels = (
            np.arange(signal.shape[1], dtype=np.float64) % int(spec["levels"])
        ) / float(spec["levels"])
        signal[0, :] = levels.astype(signal.dtype)
    elif defect == "nonrequired_nan":
        signal[-1, :] = np.nan
    elif defect == "required_nan":
        signal[1, :] = np.nan
    elif defect == "required_nan_missing_provenance":
        signal[0, 0] = np.nan
    return signal, time, {}


def _reference_r4_status(values: np.ndarray, fs: float) -> str:
    window_length = int(ceil(fs))
    if values.size < window_length:
        return TECHNICAL_STATUS_UNRESOLVED
    complete_window_found = False
    for start in range(values.size - window_length + 1):
        window = values[start:start + window_length]
        if not np.isfinite(window).all():
            continue
        complete_window_found = True
        if np.unique(window).size <= 8:
            return TECHNICAL_INVALID
    return (
        TECHNICALLY_ADMISSIBLE
        if complete_window_found
        else TECHNICAL_STATUS_UNRESOLVED
    )


def _rule_status(
    result: dict[str, Any],
    rule_id: str,
    *,
    channel_index: int = 0,
    segment_index: int = 0,
) -> str:
    segment_flags = [
        flag
        for flag in result["segment_flags"]
        if flag["channel_index"] == channel_index
        and flag["segment_index"] == segment_index
    ]
    if not segment_flags:
        return "NOT_EVALUATED"
    return segment_flags[0]["rule_statuses"][rule_id]["status"]


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
        signal, time, _ = build_fixture(dict(spec))
        required_indices = spec.get(
            "required_channel_indices",
            list(range(int(spec["n_channels"]))),
        )
        common: dict[str, Any] = {
            "required_channel_indices": required_indices,
            "channel_selection_provenance": {
                "generator": FIXTURE_GENERATOR,
                "fixture_id": spec["id"],
            },
            "sampling_frequency": spec["fs"],
            "fixture_manifest_sha256": manifest_hash,
            "runtime_identity": _runtime_identity(),
            "input_provenance": {
                "generator": FIXTURE_GENERATOR,
                "fixture_id": spec["id"],
            },
        }
        if spec.get("segment_boundaries") is not None:
            common["segment_boundaries"] = spec["segment_boundaries"]
        if spec.get("defect") == "required_nan_missing_provenance":
            common["channel_selection_provenance"] = []
        result = audit_exported_float(signal, time, **common)
        expected_status = spec["expected_status"]
        exact_status = result["recording_status"] == expected_status
        rule_matches: dict[str, bool] = {}
        for rule_id, expected_rule in spec.get("expected_rules", {}).items():
            channel_index = spec.get("expected_rule_channels", {}).get(
                rule_id,
                0,
            )
            rule_matches[rule_id] = (
                _rule_status(
                    result,
                    rule_id,
                    channel_index=int(channel_index),
                )
                == expected_rule
            )
        if spec["family"] == "R5_grid":
            rule_matches["R5_timebase"] = (
                result["timebase_status"]["status"] == expected_status
            )
        r4_reference_match = True
        if (
            result["segment_flags"]
            and spec["family"] in {"R4", "segments"}
        ):
            for segment_flag in result["segment_flags"]:
                channel_index = int(segment_flag["channel_index"])
                segment_index = int(segment_flag["segment_index"])
                start = int(segment_flag["start"])
                stop = int(segment_flag["stop"])
                expected_r4 = _reference_r4_status(
                    signal[channel_index, start:stop],
                    float(spec["fs"]),
                )
                r4_reference_match &= (
                    segment_flag["rule_statuses"]["R4"]["status"]
                    == expected_r4
                )
        case_results.append(
            {
                "fixture_id": spec["id"],
                "family": spec["family"],
                "defect": spec.get("defect"),
                "expected_status": expected_status,
                "observed_status": result["recording_status"],
                "exact_status_match": exact_status,
                "expected_rule_matches": rule_matches,
                "all_expected_rules_match": all(rule_matches.values()),
                "r4_reference_match": r4_reference_match,
                "invalid_reasons": result["invalid_reasons"],
                "unresolved_reasons": result["unresolved_reasons"],
                "timebase_status": result["timebase_status"],
                "input_hash": result["input_hash"],
                "determinism_hash": result["determinism_hash"],
            }
        )
    ordinary_cases = [
        case
        for case, spec in zip(case_results, FIXTURE_SPECS)
        if spec["family"] != "R5_grid"
    ]
    exact_status = all(
        case["exact_status_match"] for case in case_results
    )
    exact_rules = all(
        case["all_expected_rules_match"] for case in case_results
    )
    r4_equivalence = all(
        case["r4_reference_match"] for case in case_results
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "gate_id": GATE_ID,
        "scope": "synthetic_only",
        "rule_manifest_path": RULE_MANIFEST_PATH,
        "rule_manifest_sha256": RULE_MANIFEST_SHA256,
        "fixture_generator": FIXTURE_GENERATOR,
        "fixture_manifest_sha256": manifest_hash,
        "case_count": len(case_results),
        "ordinary_case_count": len(ordinary_cases),
        "m2b_exact_recording_status": exact_status,
        "m2b_exact_target_rule_status": exact_rules,
        "m2b_r4_reference_equivalence": r4_equivalence,
        "cases": case_results,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_gate(output_path: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="nmd_qc_float_m2b_") as directory:
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

    exact_status = all(run["m2b_exact_recording_status"] for run in runs)
    exact_rules = all(run["m2b_exact_target_rule_status"] for run in runs)
    r4_equivalence = all(run["m2b_r4_reference_equivalence"] for run in runs)
    canonical_runs = [_canonical_json(run) for run in runs]
    deterministic = len(set(canonical_runs)) == 1
    result = {
        "protocol_id": PROTOCOL_ID,
        "gate_id": GATE_ID,
        "scope": "synthetic_only",
        "real_data_accessed": False,
        "ds003670_accessed": False,
        "rule_manifest_path": RULE_MANIFEST_PATH,
        "rule_manifest_sha256": RULE_MANIFEST_SHA256,
        "fixture_manifest_sha256": runs[0]["fixture_manifest_sha256"],
        "m2b_exact_recording_status": "PASS" if exact_status else "FAIL",
        "m2b_exact_target_rule_status": "PASS" if exact_rules else "FAIL",
        "m2b_r4_reference_equivalence": "PASS" if r4_equivalence else "FAIL",
        "m3_deterministic_reexecution": "PASS" if deterministic else "FAIL",
        "gate_status": (
            "PASS"
            if exact_status and exact_rules and r4_equivalence and deterministic
            else "FAIL"
        ),
        "case_count": runs[0]["case_count"],
        "ordinary_case_count": runs[0]["ordinary_case_count"],
        "m3_run_hashes": [
            hashlib.sha256(payload).hexdigest() for payload in canonical_runs
        ],
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
