"""Run the frozen source-level QC screen for the preselected 3-person retest pilot."""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(r"M:/repos/NeuralManifoldDynamics/project/mnps2_audit/outputs/20261118_ds004148_v3_4_session_repeatability")
OUTPUT_ROOT = ROOT / "pilot_cached" / "qc_sessions23"
SELECTION_PATH = ROOT / "selection" / "selection_manifest.json"
CONFIG_PATH = Path(r"M:/repos/NeuralManifoldDynamics/mndm/config/v.3.4/config_ingest_ds004148_v34_session_repeatability_cached_pilot.yml")
POLICY_PATH = Path(r"M:/repos/NeuralManifoldDynamics/mndm/config/v.3.4/DS004148_SESSION_REPEATABILITY_V1.md")
BASELINE_QC_MANIFEST = Path(r"M:/repos/NeuralManifoldDynamics/project/mnps2_audit/outputs/20261116_ds004148_v3_4_frozen_ec_baseline/qc_assessment_manifest.json")
QC_RUNNER_PATH = Path(r"M:/repos/NeuralManifoldDynamics/project/mnps2_audit/run_ds004148_v34_baseline_qc.py")
PROJECT_DIR = Path(r"M:/repos/NeuralManifoldDynamics/project/mnps2_audit")
SUBJECTS = ("sub-01", "sub-02", "sub-03")
SESSIONS = ("session2", "session3")
TASKS = ("eyesclosed", "eyesopen")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    os.environ.setdefault("MNE_USE_NUMBA", "false")
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    sys.path.insert(0, str(PROJECT_DIR))
    qc = importlib.import_module("run_ds004148_v34_baseline_qc")
    import mne
    import numpy as np
    from scipy.signal import hilbert, welch

    qc.mne = mne
    qc.np = np
    qc.hilbert = hilbert
    qc.welch = welch

    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    if not selection.get("metadata_gate_pass"):
        raise RuntimeError("Full metadata selection gate did not pass")
    chosen_pairs = [
        row for row in selection["recordings"]
        if row["subject"] in SUBJECTS and row["session"] in SESSIONS
    ]
    if len(chosen_pairs) != len(SUBJECTS) * len(SESSIONS) * len(TASKS):
        raise RuntimeError(f"Expected 12 preselected session-2/3 recordings, found {len(chosen_pairs)}")
    if any(row["metadata_status"] != "complete" or not row.get("metadata_valid") for row in chosen_pairs):
        raise RuntimeError("One or more pilot recordings failed the frozen metadata screen")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    started_monotonic = time.perf_counter()
    run_id = started.strftime("%Y%m%dT%H%M%S%fZ")
    log_path = OUTPUT_ROOT / f"qc_run_{run_id}.log"
    logger = qc.configure_logger(log_path)
    provenance = {
        "run_id": run_id,
        "started_utc": started.isoformat(),
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "shared_qc_implementation_path": str(QC_RUNNER_PATH),
        "shared_qc_implementation_sha256": sha256(QC_RUNNER_PATH),
        "config_path": str(CONFIG_PATH),
        "config_sha256": sha256(CONFIG_PATH),
        "policy_path": str(POLICY_PATH),
        "policy_sha256": sha256(POLICY_PATH),
        "selection_manifest_path": str(SELECTION_PATH),
        "selection_manifest_sha256": sha256(SELECTION_PATH),
        "log_file": log_path.name,
    }
    items = [(session, subject, task) for session in SESSIONS for subject in SUBJECTS for task in TASKS]
    progress = qc.RunProgress(logger, len(items), started_monotonic)
    recording_rows: list[dict] = []
    window_rows: list[dict] = []
    metadata_rows: list[dict] = []
    base_manifest = json.loads(BASELINE_QC_MANIFEST.read_text(encoding="utf-8"))
    try:
        progress.start()
        logger.info(
            "PILOT_QC_START run_id=%s recordings=%d subjects=%s sessions=%s tasks=%s MNE_USE_NUMBA=%s NUMBA_DISABLE_JIT=%s",
            run_id, len(items), ",".join(SUBJECTS), ",".join(SESSIONS), ",".join(TASKS),
            os.environ.get("MNE_USE_NUMBA"), os.environ.get("NUMBA_DISABLE_JIT"),
        )
        for index, (session, subject, task) in enumerate(items, start=1):
            qc.SESSION = session
            started_recording = time.perf_counter()
            progress.start_recording(index, f"{subject}:ses-{session}", task)
            recording, windows, metadata = qc.assess_one(subject, task, progress=progress)
            metadata["session"] = f"ses-{session}"
            recording_rows.append(recording)
            window_rows.extend(windows)
            metadata_rows.append(metadata)
            progress.recording_done(len(windows), time.perf_counter() - started_recording)

        qc.write_csv(OUTPUT_ROOT / "qc_recordings.csv", recording_rows)
        qc.write_csv(OUTPUT_ROOT / "qc_windows.csv", window_rows)
        completed = datetime.now(timezone.utc)
        manifest = {
            "schema": "mndm.ds004148_session_repeatability_pilot_qc.v1",
            "run_provenance": {
                **provenance,
                "completed_utc": completed.isoformat(),
                "elapsed_seconds": round(time.perf_counter() - started_monotonic, 3),
            },
            "selection": {
                "subjects": list(SUBJECTS),
                "sessions": list(SESSIONS),
                "tasks": list(TASKS),
                "basis": "Original metadata-frozen three-subject MNPS 3.4 pilot cohort; no QC, feature, or coordinate values used to select subjects.",
                "recordings_expected": len(items),
                "recordings_assessed": len(recording_rows),
            },
            "methods": base_manifest["methods"],
            "runtime": {
                "python": sys.version.split()[0],
                "mne_version": mne.__version__,
                "numpy_version": np.__version__,
                "mne_use_numba": os.environ.get("MNE_USE_NUMBA"),
                "numba_disable_jit": os.environ.get("NUMBA_DISABLE_JIT"),
            },
            "recording_metadata": metadata_rows,
            "recordings_csv_sha256": sha256(OUTPUT_ROOT / "qc_recordings.csv"),
            "windows_csv_sha256": sha256(OUTPUT_ROOT / "qc_windows.csv"),
            "raw_signal_hashes_computed": False,
            "artifact_qc_applied_to_feature_signal": False,
            "qualification": "Three-participant technical repeatability pilot only; no artifact-free or biological reliability claim.",
        }
        (OUTPUT_ROOT / "qc_assessment_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "PILOT_QC_COMPLETE run_id=%s recordings=%d windows=%d elapsed_sec=%.3f output=%s",
            run_id, len(recording_rows), len(window_rows), time.perf_counter() - started_monotonic, OUTPUT_ROOT,
        )
        print(f"recordings={len(recording_rows)} windows={len(window_rows)} output={OUTPUT_ROOT}")
    except BaseException:
        logger.exception("PILOT_QC_FAILED run_id=%s", run_id)
        raise


if __name__ == "__main__":
    main()
