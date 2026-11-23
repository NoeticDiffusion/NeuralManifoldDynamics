"""Verify legacy Jacobian identity and the v3.4-only suppression smoke run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import h5py
import numpy as np


REPO = Path(r"M:/repos/NeuralManifoldDynamics")
OUTPUT_ROOT = REPO / "project/mnps2_audit/outputs/20261118_ds004148_v3_4_session_repeatability"
DEFAULT_REFERENCE_H5 = (
    OUTPUT_ROOT / "full_reextracted/processed/ds004148/neuralmanifolddynamics_ds004148_20261118_205216"
    / "sub-01_ses-session2_eyesclosed/sub-01_ses-session2_eyesclosed.h5"
)
DEFAULT_UNSUPPRESSED_RUN = DEFAULT_REFERENCE_H5.parent.parent
DEFAULT_SMOKE_RUN = (
    OUTPUT_ROOT / "legacy_jacobian_suppression_smoke/processed/ds004148/neuralmanifolddynamics_ds004148_20261119_093722"
)
DEFAULT_CONFIG = REPO / "mndm/config/v.3.4/config_ingest_ds004148_v34_legacy_jacobian_suppression_smoke.yml"
DEFAULT_OUTPUT = OUTPUT_ROOT / "legacy_jacobian_suppression_smoke/verification.json"
ANCHOR_ID = "ds005385_ses1_pretask_EyesClosed_20261111_v1"
ANCHOR_SHA256 = "79da8c208a6c9240e84cefee7abc9a26c3121d59c1c86bc3324d33172193d1ab"
LEGACY_PATHS = (
    "jacobian",
    "jacobian_9D",
    "jacobian_subject_anchored",
    "jacobian_9D_subject_anchored",
    "jacobian_cohort_anchored",
    "jacobian_9D_cohort_anchored",
)
V34_JACOBIAN_PATH = "measurements/v3_4/jacobian"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def verify_reference(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        manifest = json.loads(text(h5.attrs["manifest"]))
        expected_legacy_paths = {
            "primary_path": "/jacobian",
            "primary_v2_path": "/jacobian_9D",
        }
        if not manifest.get("jacobian_h5"):
            raise ValueError("Reference H5 lacks its Jacobian path identity manifest")
        for field, expected in expected_legacy_paths.items():
            if manifest["jacobian_h5"].get(field) != expected:
                raise ValueError(f"Reference H5 has unexpected legacy identity for {field}")
        if manifest.get("coords_9d", {}).get("export_prefix") != "v2_":
            raise ValueError("The 9D legacy output is not identified as the MNPS v2 export")
        if text(h5.attrs.get("v2_definition", "")) != "subcoords_9d_v2_0":
            raise ValueError("The 9D legacy output has an unexpected coordinate definition")
        if [text(v) for v in manifest.get("mnps_axes", {}).get("names", [])] != ["m", "d", "e"]:
            raise ValueError("The legacy primary output does not identify its [m, d, e] coordinate order")
        if V34_JACOBIAN_PATH in h5:
            raise ValueError("The reference unexpectedly includes the v3.4 Jacobian candidate")
        layer_identities = {}
        for path_name, expected_space in (
            ("jacobian_subject_anchored", "mnps_3d"),
            ("jacobian_9D_subject_anchored", "coords_9d"),
        ):
            if path_name not in h5:
                raise ValueError(f"Reference H5 is missing legacy layer {path_name}")
            group = h5[path_name]
            identity = {
                "schema_version": text(group.attrs.get("schema_version", "")),
                "coordinate_contract": text(group.attrs.get("coordinate_contract", "")),
                "coordinate_space": text(group.attrs.get("coordinate_space", "")),
            }
            if identity["schema_version"] != "mndm.jacobian_layer.v2.1" or identity["coordinate_space"] != expected_space:
                raise ValueError(f"Unexpected legacy layer identity at {path_name}: {identity}")
            layer_identities[path_name] = identity
        return {
            "file": path.name,
            "sha256": sha256(path),
            "legacy_primary": manifest["jacobian_h5"]["primary_path"],
            "legacy_9d": manifest["jacobian_h5"]["primary_v2_path"],
            "legacy_9d_coordinate_definition": text(h5.attrs["v2_definition"]),
            "legacy_layer_identities": layer_identities,
            "v3_4_candidate_path_present": False,
        }


def verify_smoke_run(run_dir: Path, unsuppressed_run_dir: Path) -> list[dict[str, Any]]:
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    paths = sorted(run_dir.rglob("*.h5"))
    expected_count = int(run_manifest.get("counts", {}).get("h5", -1))
    if expected_count != len(paths) or expected_count != 4:
        raise ValueError(f"Expected exactly four successful smoke H5 files; found {len(paths)} (manifest={expected_count})")
    rows = []
    for path in paths:
        with h5py.File(path, "r") as h5:
            forbidden = [name for name in (*LEGACY_PATHS, V34_JACOBIAN_PATH) if name in h5]
            if forbidden:
                raise ValueError(f"Suppressed Jacobian groups remain in {path.name}: {forbidden}")
            if "measurements/v3_4/coordinate_validity_mask" not in h5:
                raise ValueError(f"MNPS 3.4 coordinates were not exported in {path.name}")
            mask = np.asarray(h5["measurements/v3_4/coordinate_validity_mask"][()], dtype=bool)
            names = np.asarray(h5["measurements/v3_4/coordinate_names"][()])
            values_path = "measurements/v3_4/coordinates/values"
            if values_path not in h5:
                raise ValueError(f"MNPS 3.4 coordinate values were not exported in {path.name}")
            values = np.asarray(h5[values_path][()], dtype=np.float64)
            if mask.shape != (74, 9) or names.shape != (9,) or values.shape != mask.shape:
                raise ValueError(
                    f"Unexpected MNPS 3.4 dimensions in {path.name}: "
                    f"mask={mask.shape}, names={names.shape}, values={values.shape}"
                )
            if not np.isfinite(values[mask]).all():
                raise ValueError(
                    f"MNPS 3.4 validity mask marks non-finite values valid in {path.name}"
                )
            if not mask.any() or not np.isfinite(values).any():
                raise ValueError(f"No finite supported MNPS 3.4 coordinate values remain in {path.name}")
            unsuppressed_path = unsuppressed_run_dir / path.relative_to(run_dir)
            if not unsuppressed_path.is_file():
                raise FileNotFoundError(f"Matching unsuppressed reference H5 is missing: {unsuppressed_path}")
            with h5py.File(unsuppressed_path, "r") as reference_h5:
                reference_values = np.asarray(
                    reference_h5[values_path][()], dtype=np.float64
                )
                reference_mask = np.asarray(
                    reference_h5["measurements/v3_4/coordinate_validity_mask"][()], dtype=bool
                )
                reference_names = np.asarray(
                    reference_h5["measurements/v3_4/coordinate_names"][()]
                )
            if not np.array_equal(names, reference_names):
                raise ValueError(f"MNPS 3.4 coordinate order changed in {path.name}")
            if not np.array_equal(values, reference_values, equal_nan=True):
                raise ValueError(
                    f"Disabling Jacobians changed MNPS 3.4 coordinate values in {path.name}"
                )
            if not np.array_equal(mask, reference_mask):
                raise ValueError(
                    f"Disabling Jacobians changed MNPS 3.4 coordinate support in {path.name}"
                )
            anchor_id = text(h5["measurements/v3_4/provenance/anchor_id"][()])
            anchor_hash = text(h5["measurements/v3_4/provenance/anchor_manifest_sha256"][()])
            if anchor_id != ANCHOR_ID or anchor_hash != ANCHOR_SHA256:
                raise ValueError(f"Unexpected frozen anchor in {path.name}: {anchor_id} {anchor_hash}")
            file_manifest = json.loads(text(h5.attrs["manifest"]))
            jacobian_h5 = file_manifest.get("jacobian_h5", {})
            if jacobian_h5.get("primary_path") is not None or jacobian_h5.get("primary_v2_path") is not None or jacobian_h5.get("layer_paths"):
                raise ValueError(f"H5 manifest advertises Jacobian outputs in {path.name}: {jacobian_h5}")
            rows.append({
                "file": path.name,
                "sha256": sha256(path),
                "windows": int(mask.shape[0]),
                "coordinate_count": int(mask.shape[1]),
                "valid_coordinate_entries": int(mask.sum()),
                "coordinate_values_shape": list(values.shape),
                "finite_coordinate_values": int(np.isfinite(values).sum()),
                "coordinates_match_unsuppressed_reference": True,
                "unsuppressed_reference_sha256": sha256(unsuppressed_path),
                "anchor_id": anchor_id,
                "anchor_sha256": anchor_hash,
                "legacy_and_v3_4_jacobian_paths_present": [],
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-h5", type=Path, default=DEFAULT_REFERENCE_H5)
    parser.add_argument("--unsuppressed-run-dir", type=Path, default=DEFAULT_UNSUPPRESSED_RUN)
    parser.add_argument("--smoke-run-dir", type=Path, default=DEFAULT_SMOKE_RUN)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = {
        "schema": "mndm.ds004148_v34_jacobian_suppression_smoke.v2",
        "repository_base_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "mne": importlib.metadata.version("mne"),
            "MNE_USE_NUMBA": "false",
            "NUMBA_DISABLE_JIT": "1",
        },
        "implementation_sha256": {
            "h5_writer": sha256(REPO / "core/src/core/io/h5_writer.py"),
            "summary_pipeline": sha256(REPO / "mndm/src/mndm/pipeline/summary.py"),
            "eeg_feature_pipeline": sha256(REPO / "mndm/src/mndm/features/eeg.py"),
            "v3_4_candidates": sha256(REPO / "mndm/src/mndm/mnps/v3_4/candidates.py"),
            "v3_4_coordinates": sha256(REPO / "mndm/src/mndm/mnps/v3_4/coordinates.py"),
            "v3_4_external_anchor": sha256(REPO / "mndm/src/mndm/mnps/v3_4/external_anchor.py"),
            "verification_script": sha256(Path(__file__).resolve()),
        },
        "reference_legacy_identity_check": verify_reference(args.reference_h5),
        "suppression_smoke": {
            "run_dir": str(args.smoke_run_dir),
            "unsuppressed_reference_run_dir": str(args.unsuppressed_run_dir),
            "jacobian_paths_checked_absent": [f"/{path}" for path in (*LEGACY_PATHS, V34_JACOBIAN_PATH)],
            "config_path": str(args.config),
            "config_sha256": sha256(args.config),
            "inherited_feature_config_sha256": sha256(REPO / "mndm/config/v.3.4/config_ingest_ds004148_v34_session23_fullcohort_repeatability.yml"),
            "inherited_baseline_config_sha256": sha256(REPO / "mndm/config/v.3.4/config_ingest_ds004148_v34_baseline_ec_anchor.yml"),
            "common_eeg_config_sha256": sha256(REPO / "mndm/config/v.3.4/config_ingest_common_eeg_9d.yml"),
            "cached_features_sha256": sha256(args.smoke_run_dir.parent / "features.parquet"),
            "file_index_sha256": sha256(args.smoke_run_dir.parent / "file_index.csv"),
            "summarize_command": (
                "python -m mndm.cli summarize --config "
                "mndm/config/v.3.4/config_ingest_ds004148_v34_legacy_jacobian_suppression_smoke.yml "
                "--dataset ds004148 --subject sub-01 --n-jobs 1 --verbose-logs"
            ),
            "run_manifest_sha256": sha256(args.smoke_run_dir / "run_manifest.json"),
            "h5": verify_smoke_run(args.smoke_run_dir, args.unsuppressed_run_dir),
        },
        "result": "passed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "result": result["result"],
        "smoke_h5_count": len(result["suppression_smoke"]["h5"]),
        "legacy_reference_paths": ["/jacobian", "/jacobian_9D"],
        "jacobian_paths_checked_absent": result["suppression_smoke"]["jacobian_paths_checked_absent"],
        "legacy_paths_suppressed": True,
        "v3_4_coordinate_vectors_retained": all(
            row["coordinates_match_unsuppressed_reference"]
            for row in result["suppression_smoke"]["h5"]
        ),
        "verification_manifest": str(args.output),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
