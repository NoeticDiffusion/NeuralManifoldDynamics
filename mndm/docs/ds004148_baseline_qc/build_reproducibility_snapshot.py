"""Archive compact ds004148 QC provenance and checksum large local outputs.

The source code, locked configuration and policies remain at their canonical
repository paths. This script copies only small selection/run manifests and the
run log into this package, then records checksums for measurement tables and
the 120 H5 files without copying those large outputs into Git.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = Path(
    r"M:/repos/NeuralManifoldDynamics/project/mnps2_audit/outputs/"
    r"20261116_ds004148_v3_4_frozen_ec_baseline"
)
PROCESSED_ROOT = OUTPUT_ROOT / "processed" / "ds004148"
ARTIFACTS_DIR = PACKAGE_DIR / "artifacts"
REQUIREMENTS = REPO_ROOT / "requirements.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, label: str, base: Path | None = None) -> dict[str, Any]:
    try:
        shown_path = str(path.relative_to(base)) if base else str(path)
    except ValueError:
        shown_path = str(path)
    return {
        "label": label,
        "path": shown_path,
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    qc_manifest_path = OUTPUT_ROOT / "qc_assessment_manifest.json"
    summary_manifest_path = OUTPUT_ROOT / "baseline_summary_manifest.json"
    qc_manifest = json.loads(qc_manifest_path.read_text(encoding="utf-8"))
    summary_manifest = json.loads(summary_manifest_path.read_text(encoding="utf-8"))
    run_provenance = qc_manifest["run_provenance"]
    log_path = OUTPUT_ROOT / run_provenance["log_file"]

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    archive_names = (
        "cohort_selection_manifest.csv",
        "cohort_selection_manifest.json",
        "indexed_input_verification.json",
        "qc_assessment_manifest.json",
        "baseline_summary_manifest.json",
        "cohort_paired_summary.csv",
        "cohort_paired_raw_feature_summary.csv",
        log_path.name,
    )
    copied_artifacts: list[dict[str, Any]] = []
    for name in archive_names:
        source = OUTPUT_ROOT / name
        destination = ARTIFACTS_DIR / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)
        copied_artifacts.append(file_record(destination, label=f"archived:{name}", base=PACKAGE_DIR))

    source_paths = {
        "qc_runner": REPO_ROOT / "project/mnps2_audit/run_ds004148_v34_baseline_qc.py",
        "paired_summarizer": REPO_ROOT / "project/mnps2_audit/summarize_ds004148_baseline.py",
        "qc_policy": REPO_ROOT / "mndm/config/v.3.4/DS004148_BASELINE_EC_QC_POLICY_V1_1.md",
        "ingest_config": REPO_ROOT / "mndm/config/v.3.4/config_ingest_ds004148_v34_baseline_ec_anchor.yml",
        "frozen_ec_anchor": REPO_ROOT / "mndm/config/v.3.4/anchors/mnps_v3_4_ds005385_ses1_pretask_EyesClosed_20261111_v1.json",
        "requirements": REQUIREMENTS,
        "mndm_readme": REPO_ROOT / "mndm/README.md",
        "package_readme": PACKAGE_DIR / "README.md",
        "attrition_builder": PACKAGE_DIR / "build_attrition_report.py",
        "snapshot_builder": Path(__file__).resolve(),
        "attrition_csv": PACKAGE_DIR / "participant_qc_attrition.csv",
        "attrition_summary": PACKAGE_DIR / "qc_attrition_summary.md",
        "attrition_manifest": PACKAGE_DIR / "qc_attrition_summary.json",
    }
    missing_sources = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"Reproducibility source/artifact files missing: {missing_sources}")
    sources = [
        file_record(path, label=label, base=REPO_ROOT)
        for label, path in source_paths.items()
    ]
    anchor_record = next(item for item in sources if item["label"] == "frozen_ec_anchor")
    declared_anchor_sha256 = summary_manifest["anchor_manifest_sha256"]
    if anchor_record["sha256"] != declared_anchor_sha256:
        raise ValueError(
            "Frozen anchor SHA-256 differs from the completed summary manifest: "
            f"{anchor_record['sha256']} != {declared_anchor_sha256}"
        )

    output_files = sorted(
        path for path in OUTPUT_ROOT.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".csv", ".json", ".log"}
        and (path.suffix.lower() != ".log" or path.resolve() == log_path.resolve())
    )
    output_records = [
        file_record(path, label="run_output", base=OUTPUT_ROOT)
        for path in output_files
    ]

    h5_files = sorted(PROCESSED_ROOT.rglob("*.h5"))
    h5_records: list[dict[str, Any]] = []
    for index, path in enumerate(h5_files, start=1):
        h5_records.append(file_record(path, label="summary_input_h5", base=PROCESSED_ROOT))
        if index % 20 == 0 or index == len(h5_files):
            print(f"H5_CHECKSUM_PROGRESS files={index}/{len(h5_files)}", flush=True)

    actual_numpy = str(qc_manifest["runtime"]["numpy_version"])
    installed_distributions = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            installed_distributions.append({"name": name, "version": distribution.version})
    installed_distributions.sort(
        key=lambda item: (item["name"].casefold(), item["version"])
    )
    numpy_requirement = next(
        (line.strip() for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
         if line.strip().lower().startswith("numpy==")),
        None,
    )
    runtime = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "mne": qc_manifest["runtime"]["mne_version"],
        "numpy_loaded_by_qc_runner": actual_numpy,
        "numpy_distribution_metadata": distribution_version("numpy"),
        "scipy_distribution_metadata": distribution_version("scipy"),
        "h5py_distribution_metadata": distribution_version("h5py"),
        "installed_python_distributions": installed_distributions,
        "MNE_USE_NUMBA": qc_manifest["runtime"].get("mne_use_numba"),
        "NUMBA_DISABLE_JIT": qc_manifest["runtime"].get("numba_disable_jit"),
        "requirements_numpy_pin": numpy_requirement,
        "numpy_pin_matches_runtime": bool(
            numpy_requirement and numpy_requirement.split("==", 1)[-1] == actual_numpy
        ),
    }
    snapshot = {
        "schema": "mndm.ds004148_baseline_qc_reproducibility.v1",
        "run_id": run_provenance["run_id"],
        "run_started_utc": run_provenance["started_utc"],
        "run_results_ready_utc": run_provenance.get("results_ready_utc"),
        "run_elapsed_before_manifest_write_seconds": run_provenance.get(
            "elapsed_before_manifest_write_seconds"
        ),
        "dataset": "ds004148",
        "selection": {
            "subjects": summary_manifest["subjects"],
            "recordings": summary_manifest["h5_recordings"],
            "qc_windows": summary_manifest["qc_windows"],
        },
        "qc_status": summary_manifest["status"],
        "qc_policy_version": summary_manifest["policy_version"],
        "anchor_id": summary_manifest["anchor_id"],
        "anchor_manifest_sha256": summary_manifest["anchor_manifest_sha256"],
        "runtime": runtime,
        "environment_notes": [
            "The recorded NumPy runtime differs from the repository requirements pin; no parity with the pinned version is claimed.",
            "Raw .eeg binary hashes were not computed; the QC manifest records the hashed BIDS/header sidecars and states this limitation.",
        ],
        "output_root": str(OUTPUT_ROOT),
        "large_tables_and_h5_remain_at_output_root": True,
        "versioned_source_files": sources,
        "archived_small_artifacts": copied_artifacts,
        "output_file_checksums": output_records,
        "summary_input_h5_count": len(h5_records),
        "summary_input_h5_total_bytes": sum(item["size_bytes"] for item in h5_records),
        "summary_input_h5_checksums": h5_records,
    }
    inventory_path = PACKAGE_DIR / "runtime_and_artifact_inventory.json"
    inventory_path.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"SNAPSHOT_COMPLETE run_id={snapshot['run_id']} source_files={len(sources)} "
        f"output_files={len(output_records)} h5_files={len(h5_records)} "
        f"numpy_pin_matches_runtime={runtime['numpy_pin_matches_runtime']} "
        f"inventory={inventory_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
