"""One controlled live execution of NMD-QC-FLOAT-001 M4B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping
from urllib.request import Request, urlopen

import numpy as np

THIS_FILE = Path(__file__).resolve()
MNDM_ROOT = THIS_FILE.parents[1]
REPO_ROOT = MNDM_ROOT.parent
SRC_ROOT = MNDM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mndm.dynamical_families.nmd_qc_float import (  # noqa: E402
    RULE_MANIFEST_SHA256,
    audit_exported_float,
)
from mndm.dynamical_families.nmd_qc_float_m4b_compare import (  # noqa: E402
    canonicalize_production_qc,
    canonicalize_reference_qc,
    compare_canonical_qc,
)
from mndm.dynamical_families.nmd_qc_float_m4b_reference import (  # noqa: E402
    M4BReferenceError,
    build_reference_surface,
    canonical_array_hash,
    canonical_json_hash,
    evaluate_reference,
    read_mat4_variable,
    read_wfdb_header,
    validate_frozen_m4b_scope,
)


PROTOCOL_ID = "NMD-QC-FLOAT-001-M4B"
EXPECTED_SCOPE_SHA256 = (
    "dbcb847c809fa04483a6301740ceecf98ba230ebb3fa4363056b41758f030866"
)
EXPECTED_HEADER_SHA256 = (
    "055c2a3fe7e946c0a3e645d9d2fb70db7e3b47885165eadf5d80f4e6039c4f56"
)
EXPECTED_WFDB_VERSION = "4.0.0"
EXPECTED_M4A_AUDIT_HASH = (
    "e85c650d2daaa39cb6cb728e5f6ec4dc45349f94fbca9e6f6381ce3453fa3cd5"
)
EXPECTED_M4A_RESULT_SHA256 = (
    "0c1f90c55fa50aac6c5c4c5545add03b2357ba128151870dcd4244a70c0cdf92"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        url,
        headers={"User-Agent": "NMD-QC-FLOAT-001-M4B/0.1"},
    )
    with urlopen(request, timeout=120) as response, path.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _code_identity() -> dict[str, str]:
    reference_path = (
        SRC_ROOT
        / "mndm"
        / "dynamical_families"
        / "nmd_qc_float_m4b_reference.py"
    )
    compare_path = (
        SRC_ROOT
        / "mndm"
        / "dynamical_families"
        / "nmd_qc_float_m4b_compare.py"
    )
    production_path = (
        SRC_ROOT / "mndm" / "dynamical_families" / "nmd_qc_float.py"
    )
    runner_path = THIS_FILE
    return {
        "reference_parser_evaluator_sha256": _sha256_file(reference_path),
        "comparison_helpers_sha256": _sha256_file(compare_path),
        "production_qc_sha256": _sha256_file(production_path),
        "runner_sha256": _sha256_file(runner_path),
    }


def _representation_comparison(
    reference_surface: np.ndarray,
    test_surface: np.ndarray,
    reference_time: np.ndarray,
    test_time: np.ndarray,
    *,
    reference_channel_order: list[str],
    test_channel_order: list[str],
    reference_sampling_frequency: float,
    test_sampling_frequency: float,
    segment_boundaries: list[list[int]],
) -> dict[str, Any]:
    shape_match = reference_surface.shape == test_surface.shape
    channel_order_match = reference_channel_order == test_channel_order
    sampling_frequency_match = (
        reference_sampling_frequency == test_sampling_frequency
    )
    nan_mask_match = False
    finite_value_match = False
    max_abs_error: float | None = None
    if shape_match:
        reference_nan = np.isnan(reference_surface)
        test_nan = np.isnan(test_surface)
        nan_mask_match = bool(np.array_equal(reference_nan, test_nan))
        jointly_finite = np.isfinite(reference_surface) & np.isfinite(test_surface)
        if np.any(jointly_finite):
            differences = np.abs(
                reference_surface[jointly_finite] - test_surface[jointly_finite]
            )
            max_abs_error = float(np.max(differences))
            tolerance = 16.0 * np.finfo(np.float64).eps * np.maximum(
                1.0,
                np.abs(reference_surface[jointly_finite]),
            )
            finite_value_match = bool(np.all(differences <= tolerance))
        else:
            finite_value_match = True
    time_shape_match = reference_time.shape == test_time.shape
    time_match = False
    max_time_error: float | None = None
    if time_shape_match:
        time_difference = np.abs(reference_time - test_time)
        max_time_error = float(np.max(time_difference))
        time_tolerance = 16.0 * np.finfo(np.float64).eps * np.maximum(
            1.0,
            np.abs(reference_time),
        )
        time_match = bool(np.all(time_difference <= time_tolerance))
    segment_match = segment_boundaries == [[0, reference_surface.shape[0]]]
    return {
        "pass": all(
            (
                shape_match,
                channel_order_match,
                sampling_frequency_match,
                nan_mask_match,
                finite_value_match,
                time_match,
                segment_match,
            )
        ),
        "shape_match": shape_match,
        "reference_shape": list(reference_surface.shape),
        "test_shape": list(test_surface.shape),
        "channel_order_match": channel_order_match,
        "reference_channel_order": reference_channel_order,
        "test_channel_order": test_channel_order,
        "sampling_frequency_match": sampling_frequency_match,
        "reference_sampling_frequency": reference_sampling_frequency,
        "test_sampling_frequency": test_sampling_frequency,
        "nan_mask_match": nan_mask_match,
        "finite_value_match": finite_value_match,
        "max_abs_error": max_abs_error,
        "value_tolerance": (
            "relative_tolerance=0; "
            "absolute=16*eps(float64)*max(1,abs(reference_value))"
        ),
        "time_shape_match": time_shape_match,
        "time_match": time_match,
        "max_time_error": max_time_error,
        "time_tolerance": (
            "relative_tolerance=0; "
            "absolute=16*eps(float64)*max(1,abs(reference_time))"
        ),
        "segment_match": segment_match,
    }


def _load_scope(scope_path: Path) -> tuple[dict[str, Any], str]:
    scope_bytes = scope_path.read_bytes()
    scope_hash = hashlib.sha256(scope_bytes).hexdigest()
    if scope_hash != EXPECTED_SCOPE_SHA256:
        raise M4BReferenceError("scope manifest hash differs from preregistration")
    scope = json.loads(scope_bytes.decode("utf-8"))
    if scope["protocol_id"] != PROTOCOL_ID:
        raise M4BReferenceError("scope protocol ID is not M4B")
    if scope["reference"]["record_id"] != "0543_006_019_EEG":
        raise M4BReferenceError("scope record differs from M4B authorization")
    return scope, scope_hash


def run_gate(
    *,
    scope_path: Path,
    output_path: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    """Run M4B once; all live parameters come from the frozen scope."""
    if output_path.exists():
        return {
            "protocol_id": PROTOCOL_ID,
            "gate": "M4B",
            "status": "NOT_TESTABLE",
            "failure_reason": "one_shot_output_already_exists",
            "output_path": str(output_path),
        }
    result: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "gate": "M4B",
        "status": "NOT_TESTABLE",
        "scope_manifest_sha256": None,
        "m4a_audit_hash": EXPECTED_M4A_AUDIT_HASH,
        "m4a_result_file_sha256": EXPECTED_M4A_RESULT_SHA256,
        "wfdb_required_version": EXPECTED_WFDB_VERSION,
        "stage": "pre_signal_access",
        "failure_reason": None,
        "code_identity": _code_identity(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "hashes": {
            "native_matrix_hash": None,
            "reference_surface_hash": None,
            "test_surface_hash": None,
            "reference_time_hash": None,
            "test_time_hash": None,
            "reference_qc_summary_hash": None,
            "production_qc_summary_hash": None,
        },
    }
    try:
        scope, scope_hash = _load_scope(scope_path)
        result["scope_manifest_sha256"] = scope_hash
        reference = scope["reference"]
        test_path = scope["test_path"]
        if scope["m4a_audit_hash"] != EXPECTED_M4A_AUDIT_HASH:
            raise M4BReferenceError("M4A audit hash differs from frozen scope")
        if scope["m4a_result_file_sha256"] != EXPECTED_M4A_RESULT_SHA256:
            raise M4BReferenceError("M4A result hash differs from frozen scope")
        if (
            scope["nmd_qc_float_rule_manifest_sha256"]
            != "054394322bad951a0a7e245be657fe6842ba4f3fc5807787c2e1e47f5592a6fa"
        ):
            raise M4BReferenceError("rule-manifest hash differs from frozen scope")

        import wfdb

        result["runtime"]["wfdb"] = wfdb.__version__
        wfdb_source = Path(wfdb.__file__).resolve()
        result["runtime"]["wfdb_source"] = str(wfdb_source)
        result["runtime"]["wfdb_source_sha256"] = _sha256_file(wfdb_source)
        if wfdb.__version__ != EXPECTED_WFDB_VERSION:
            raise M4BReferenceError("required WFDB 4.0.0 is not installed")
        if RULE_MANIFEST_SHA256 != scope["nmd_qc_float_rule_manifest_sha256"]:
            raise M4BReferenceError("runtime rule-manifest hash differs from scope")
        required_channels = list(test_path["required_channel_indices"])
        channel_order = list(reference["channel_order"])
        segment_boundaries = [list(value) for value in test_path["segment_boundaries"]]
        if required_channels != list(range(int(reference["channel_count"]))):
            raise M4BReferenceError("channel scope is not all frozen channels")
        if len(channel_order) != int(reference["channel_count"]):
            raise M4BReferenceError("frozen channel scope is incomplete")
        if segment_boundaries != [[0, int(reference["sample_count"])]]:
            raise M4BReferenceError("segment scope is not the full frozen record")
        if not test_path["channel_selection_provenance"]["predeclared_before_signal_access"]:
            raise M4BReferenceError("selection provenance was not predeclared")

        cache_dir.mkdir(parents=True, exist_ok=True)
        record_stem = cache_dir / str(test_path["record_stem"])
        header_path = record_stem.with_suffix(".hea")
        signal_path = record_stem.with_suffix(".mat")
        _write_download(reference["header_url"], header_path)
        header_hash = _sha256_file(header_path)
        result["native_header_sha256"] = header_hash
        if header_hash != EXPECTED_HEADER_SHA256:
            raise M4BReferenceError("native header hash differs from M4A reference")

        _write_download(reference["signal_url"], signal_path)
        # This is intentionally the first operation on the downloaded payload.
        signal_file_hash = _sha256_file(signal_path)
        result["native_signal_file_sha256"] = signal_file_hash
        result["stage"] = "signal_accessed"

        header = read_wfdb_header(header_path)
        variable = read_mat4_variable(signal_path, "val")
        validate_frozen_m4b_scope(header, variable, scope)
        result["native_header_metadata"] = {
            "record_name": header.record_name,
            "start_time": header.start_time,
            "end_time": header.end_time,
            "channel_order": list(header.channel_names),
            "sampling_frequency_hz": header.sampling_frequency,
            "sample_count": header.sample_count,
        }
        reference_export = build_reference_surface(header, variable.value)
        result["native_matrix"] = {
            "dtype": str(variable.value.dtype),
            "shape": list(variable.value.shape),
            "mopt": variable.mopt,
            "data_type_code": variable.data_type_code,
            "data_offset": variable.data_offset,
            "byte_order": variable.byte_order,
            "sentinel": -32768,
        }
        result["hashes"].update(
            {
                "native_matrix_hash": canonical_array_hash(variable.value),
                "reference_surface_hash": canonical_array_hash(
                    reference_export.surface
                ),
                "reference_time_hash": canonical_array_hash(reference_export.time),
            }
        )

        record = wfdb.rdrecord(
            str(record_stem),
            physical=True,
            m2s=bool(test_path["m2s"]),
            return_res=int(test_path["return_res"]),
            force_channels=bool(test_path["force_channels"]),
        )
        test_surface = np.asarray(record.p_signal)
        if test_surface.ndim != 2:
            raise M4BReferenceError("WFDB physical export is not two-dimensional")
        test_time = np.linspace(
            0.0,
            (int(reference["sample_count"]) - 1)
            / float(reference["sampling_frequency_hz"]),
            int(reference["sample_count"]),
            dtype=np.float64,
        )
        result["hashes"].update(
            {
                "test_surface_hash": canonical_array_hash(test_surface),
                "test_time_hash": canonical_array_hash(test_time),
            }
        )
        representation = _representation_comparison(
            reference_export.surface,
            test_surface,
            reference_export.time,
            test_time,
            reference_channel_order=channel_order,
            test_channel_order=list(record.sig_name),
            reference_sampling_frequency=float(reference["sampling_frequency_hz"]),
            test_sampling_frequency=float(record.fs),
            segment_boundaries=segment_boundaries,
        )
        result["representation_comparison"] = representation
        if not representation["pass"]:
            result["status"] = "REPRESENTATION_TRANSFER_FAIL"
            result["failure_reason"] = "representation_comparison_mismatch"
            return result

        reference_signal = reference_export.surface.T
        test_signal = test_surface.T
        provenance = dict(test_path["channel_selection_provenance"])
        independent_qc = evaluate_reference(
            reference_signal,
            reference_export.time,
            sampling_frequency=float(reference["sampling_frequency_hz"]),
            channel_labels=channel_order,
            required_channel_indices=required_channels,
            channel_selection_provenance=provenance,
            segment_boundaries=segment_boundaries,
            expected_channel_count=int(reference["channel_count"]),
            expected_sample_count=int(reference["sample_count"]),
        )
        production_qc = audit_exported_float(
            test_signal,
            test_time,
            required_channel_indices=required_channels,
            channel_labels=channel_order,
            channel_selection_provenance=provenance,
            sampling_frequency=float(reference["sampling_frequency_hz"]),
            segment_boundaries=segment_boundaries,
            input_provenance={
                "record_id": reference["record_id"],
                "source": "wfdb.rdrecord",
                "exporter_version": wfdb.__version__,
            },
            implementation_identity="mndm.dynamical_families.nmd_qc_float",
            runtime_identity={
                "python": platform.python_version(),
                "numpy": np.__version__,
                "wfdb": wfdb.__version__,
            },
            fixture_manifest_sha256=scope_hash,
        )
        canonical_reference_qc = canonicalize_reference_qc(independent_qc)
        canonical_production_qc = canonicalize_production_qc(
            production_qc,
            expected_channel_indices=required_channels,
            expected_channel_names=channel_order,
        )
        qc_comparison = compare_canonical_qc(
            canonical_reference_qc,
            canonical_production_qc,
        )
        result["qc_status_comparison"] = qc_comparison
        result["qc_reference"] = canonical_reference_qc
        result["qc_production"] = canonical_production_qc
        result["hashes"] = {
            **result["hashes"],
            "reference_qc_summary_hash": canonical_json_hash(
                canonical_reference_qc
            ),
            "production_qc_summary_hash": canonical_json_hash(
                canonical_production_qc
            ),
        }
        if not qc_comparison["matches"]:
            result["status"] = "REPRESENTATION_TRANSFER_FAIL"
            result["failure_reason"] = "qc_status_comparison_mismatch"
            return result
        result["status"] = "REPRESENTATION_TRANSFER_PASS"
        result["failure_reason"] = None
        result["stage"] = "comparison_complete"
        return result
    except Exception as exc:
        result["status"] = "NOT_TESTABLE"
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        safe_result = _json_safe(result)
        try:
            output_path.write_text(
                json.dumps(safe_result, indent=2, sort_keys=True, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
        except (TypeError, ValueError, OverflowError) as exc:
            fallback = {
                "protocol_id": PROTOCOL_ID,
                "gate": "M4B",
                "status": "NOT_TESTABLE",
                "failure_reason": (
                    "result_serialization_failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }
            output_path.write_text(
                json.dumps(fallback, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        type=Path,
        default=(
            REPO_ROOT
            / "project"
            / "orthagonal_axis"
            / "orthagonal_dynamics"
            / "nmd-qc-float"
            / "NMD-QC-FLOAT-M4B-SCOPE-0.1.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "project"
            / "orthagonal_axis"
            / "orthagonal_dynamics"
            / "nmd-qc-float"
            / "results"
            / "nmd_qc_float_001_m4b_gate.json"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "project"
            / "orthagonal_axis"
            / "orthagonal_dynamics"
            / "nmd-qc-float"
            / "results"
            / "m4b_payload"
        ),
    )
    args = parser.parse_args()
    result = run_gate(
        scope_path=args.scope,
        output_path=args.output,
        cache_dir=args.cache_dir,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "failure_reason": result["failure_reason"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "REPRESENTATION_TRANSFER_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
