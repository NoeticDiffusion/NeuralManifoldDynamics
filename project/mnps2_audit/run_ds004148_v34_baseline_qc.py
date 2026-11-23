"""Run the frozen source-level QC screen for the ds004148 baseline cohort.

This is read-only with respect to signal and MNPS outputs. It does not inspect
MNPS values, modify feature signals, or mark NMD artifact QC as applied.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from datetime import datetime, timezone

SOURCE_ROOT = Path(r"K:/ExternalReceivedDatasets/openneuro/received/ds004148")
OUTPUT_ROOT = Path(
    r"M:/repos/NeuralManifoldDynamics/project/mnps2_audit/outputs/"
    r"20261116_ds004148_v3_4_frozen_ec_baseline"
)
CONFIG_PATH = Path(
    r"M:/repos/NeuralManifoldDynamics/mndm/config/v.3.4/"
    r"config_ingest_ds004148_v34_baseline_ec_anchor.yml"
)
POLICY_PATH = Path(
    r"M:/repos/NeuralManifoldDynamics/mndm/config/v.3.4/"
    r"DS004148_BASELINE_EC_QC_POLICY_V1_1.md"
)
SUBJECTS = tuple(f"sub-{number:02d}" for number in range(1, 61))
TASKS = ("eyesclosed", "eyesopen")
SESSION = "session1"
WINDOW_SEC = 8.0
STEP_SEC = 4.0
HEARTBEAT_SECONDS = 60.0
MUSCLE_Z_THRESHOLD = 4.0  # MNE annotate_muscle_zscore default.
MUSCLE_FILTER_HZ = (110.0, 140.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def channel_label_hash(labels: list[str], *, ordered: bool) -> str:
    values = list(labels) if ordered else sorted(set(labels))
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class RunProgress:
    """Log recording/stage progress and periodic heartbeats for a run."""

    def __init__(self, logger: logging.Logger, total: int, started_monotonic: float):
        self.logger = logger
        self.total = total
        self.started_monotonic = started_monotonic
        self.completed = 0
        self.current_index = 0
        self.current_subject = "none"
        self.current_task = "none"
        self.current_stage = "startup"
        self.stage_started_monotonic = started_monotonic
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name="ds004148-qc-heartbeat",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def start_recording(self, index: int, subject: str, task: str) -> None:
        now = time.perf_counter()
        with self._lock:
            self.current_index = index
            self.current_subject = subject
            self.current_task = task
            self.current_stage = "recording_setup"
            self.stage_started_monotonic = now
        self.logger.info(
            "RECORDING_START index=%d/%d subject=%s task=%s completed=%d",
            index,
            self.total,
            subject,
            task,
            self.completed,
        )

    def stage_start(self, stage: str) -> float:
        now = time.perf_counter()
        with self._lock:
            self.current_stage = stage
            self.stage_started_monotonic = now
            subject = self.current_subject
            task = self.current_task
            index = self.current_index
        self.logger.info(
            "STAGE_START recording=%d/%d subject=%s task=%s stage=%s",
            index,
            self.total,
            subject,
            task,
            stage,
        )
        return now

    def stage_done(self, stage: str, started_monotonic: float) -> None:
        elapsed = time.perf_counter() - started_monotonic
        now = time.perf_counter()
        with self._lock:
            self.current_stage = "between_stages"
            self.stage_started_monotonic = now
            subject = self.current_subject
            task = self.current_task
            index = self.current_index
        self.logger.info(
            "STAGE_DONE recording=%d/%d subject=%s task=%s stage=%s elapsed_sec=%.3f",
            index,
            self.total,
            subject,
            task,
            stage,
            elapsed,
        )

    def recording_done(self, window_count: int, elapsed: float) -> None:
        with self._lock:
            self.completed += 1
            self.current_stage = "between_recordings"
            self.stage_started_monotonic = time.perf_counter()
            completed = self.completed
            index = self.current_index
            subject = self.current_subject
            task = self.current_task
        self.logger.info(
            "RECORDING_BUFFERED_IN_MEMORY index=%d/%d buffered=%d/%d subject=%s task=%s "
            "windows=%d persisted=false elapsed_sec=%.3f run_elapsed_sec=%.3f",
            index,
            self.total,
            completed,
            self.total,
            subject,
            task,
            window_count,
            elapsed,
            time.perf_counter() - self.started_monotonic,
        )

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(HEARTBEAT_SECONDS):
            with self._lock:
                now = time.perf_counter()
                completed = self.completed
                index = self.current_index
                subject = self.current_subject
                task = self.current_task
                stage = self.current_stage
                stage_started = self.stage_started_monotonic
            self.logger.info(
                "HEARTBEAT buffered=%d/%d current=%d/%d subject=%s task=%s "
                "stage=%s stage_elapsed_sec=%.1f run_elapsed_sec=%.1f",
                completed,
                self.total,
                index,
                self.total,
                subject,
                task,
                stage,
                now - stage_started,
                now - self.started_monotonic,
            )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


def configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"ds004148_baseline_qc.{log_path.stem}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)sZ %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime
    file_handler = logging.FileHandler(log_path, mode="x", encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    for handler in (file_handler, stream_handler):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def band_integral(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs <= high)
    if int(mask.sum()) < 2:
        return float("nan")
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(psd[mask], freqs[mask]))


def assess_one(
    subject: str,
    task: str,
    progress: RunProgress | None = None,
) -> tuple[dict, list[dict], dict]:
    stem = f"{subject}_ses-{SESSION}_task-{task}"
    stage_started = progress.stage_start("validate_source_metadata") if progress else None
    folder = SOURCE_ROOT / subject / f"ses-{SESSION}" / "eeg"
    vhdr = folder / f"{stem}_eeg.vhdr"
    eeg_binary = folder / f"{stem}_eeg.eeg"
    vmrk = folder / f"{stem}_eeg.vmrk"
    eeg_json_path = folder / f"{stem}_eeg.json"
    channels_path = folder / f"{stem}_channels.tsv"
    required = (vhdr, eeg_binary, vmrk, eeg_json_path, channels_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{stem}: missing BIDS components: {missing}")

    eeg_json = json.loads(eeg_json_path.read_text(encoding="utf-8"))
    with channels_path.open(encoding="utf-8-sig", newline="") as stream:
        channel_rows = list(csv.DictReader(stream, delimiter="\t"))
    labels = [str(row["name"]) for row in channel_rows]
    units = sorted({str(row["units"]).strip() for row in channel_rows})
    channel_types = sorted({str(row["type"]).strip() for row in channel_rows})
    sfreqs = sorted({float(row["sampling_frequency"]) for row in channel_rows})
    if units != ["uV"]:
        raise ValueError(f"{stem}: expected all BIDS channels to declare uV, found {units}")
    if sfreqs != [500.0] or eeg_json.get("SamplingFrequency") != 500:
        raise ValueError(f"{stem}: unexpected sampling frequency: channels={sfreqs}, json={eeg_json.get('SamplingFrequency')}")
    if progress:
        progress.stage_done("validate_source_metadata", stage_started)

    stage_started = progress.stage_start("load_raw_recording") if progress else None
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
    source_sfreq = float(raw.info["sfreq"])
    raw_uv = raw.get_data(picks="eeg") * 1e6  # MNE EEG data are SI volts; BIDS declares uV.
    brainvision_labels = list(raw.ch_names)
    if raw_uv.shape[0] != len(labels) or len(brainvision_labels) != len(labels):
        raise ValueError(
            f"{stem}: EEG/BIDS/BrainVision channel counts differ: "
            f"eeg_data={raw_uv.shape[0]}, bids={len(labels)}, brainvision={len(brainvision_labels)}"
        )
    channel_order_matches_bids = brainvision_labels == labels
    channel_order_casefold_matches_bids = (
        [label.casefold() for label in brainvision_labels]
        == [label.casefold() for label in labels]
    )
    if not channel_order_casefold_matches_bids:
        raise ValueError(
            f"{stem}: BrainVision and BIDS channel labels/order differ beyond capitalization"
        )
    channel_label_case_mismatches = [
        {"index": index, "bids": bids_label, "brainvision": brainvision_label}
        for index, (bids_label, brainvision_label) in enumerate(zip(labels, brainvision_labels))
        if bids_label != brainvision_label
    ]
    if channel_label_case_mismatches and progress:
        progress.logger.warning(
            "CHANNEL_LABEL_CASE_ONLY_MISMATCH subject=%s task=%s count=%d mismatches=%s",
            subject,
            task,
            len(channel_label_case_mismatches),
            json.dumps(channel_label_case_mismatches, ensure_ascii=False),
        )
    if not np.isclose(source_sfreq, 500.0):
        raise ValueError(f"{stem}: BrainVision sampling frequency differs from BIDS metadata: {source_sfreq}")
    if progress:
        progress.stage_done("load_raw_recording", stage_started)

    stage_started = progress.stage_start("raw_finite_flatline_amplitude_descriptors") if progress else None
    finite = np.isfinite(raw_uv)
    finite_fraction = float(finite.mean()) if finite.size else float("nan")
    exact_flatlines = int(sum(
        np.isfinite(channel).sum() > 1
        and float(np.nanmax(channel)) == float(np.nanmin(channel))
        for channel in raw_uv
    ))
    absolute = np.abs(raw_uv[finite])
    absolute_percentiles = (
        np.percentile(absolute, [50, 95, 99, 99.9, 99.99]).tolist()
        if absolute.size
        else [float("nan")] * 5
    )
    max_abs_uv = float(np.max(absolute)) if absolute.size else float("nan")
    if progress:
        progress.stage_done("raw_finite_flatline_amplitude_descriptors", stage_started)

    # Descriptive pre-notch 50 Hz power ratio. This is not a pass/fail threshold.
    stage_started = progress.stage_start("welch_line_noise_descriptor") if progress else None
    freqs, psd = welch(raw_uv, fs=source_sfreq, nperseg=2000, noverlap=1000, axis=1)
    median_psd = np.nanmedian(psd, axis=0)
    total_1_45 = band_integral(freqs, median_psd, 1.0, 45.0)
    line_49_51 = band_integral(freqs, median_psd, 49.0, 51.0)
    line_noise_ratio = line_49_51 / total_1_45 if total_1_45 > 0 else float("nan")
    if progress:
        progress.stage_done("welch_line_noise_descriptor", stage_started)

    # Artifact-only QC copy: average reference and notch 50 Hz and its first
    # two harmonics. Then replicate MNE's published muscle z-score recipe:
    # 110-140 Hz Hilbert envelope, channelwise z-scores, sum/sqrt(channels),
    # 4 Hz low-pass and a threshold of 4. MNE 1.6's wrapper currently calls
    # Raw.pick with keywords unsupported by that Raw API, so the documented
    # computation is applied through MNE's filter functions directly.
    stage_started = progress.stage_start("qc_copy_reference_and_notch") if progress else None
    raw_qc = raw.copy()
    raw_qc.set_eeg_reference("average", projection=False, verbose="ERROR")
    raw_qc.notch_filter([50.0, 100.0, 150.0], picks="eeg", verbose="ERROR")
    qc_data_uv = raw_qc.get_data(picks="eeg") * 1e6
    if progress:
        progress.stage_done("qc_copy_reference_and_notch", stage_started)

    stage_started = progress.stage_start("muscle_band_hilbert_and_score") if progress else None
    muscle_band = mne.filter.filter_data(
        qc_data_uv,
        sfreq=source_sfreq,
        l_freq=MUSCLE_FILTER_HZ[0],
        h_freq=MUSCLE_FILTER_HZ[1],
        verbose="ERROR",
    )
    muscle_envelope = np.abs(hilbert(muscle_band, axis=1))
    muscle_mean = np.mean(muscle_envelope, axis=1, keepdims=True)
    muscle_std = np.std(muscle_envelope, axis=1, keepdims=True)
    muscle_z = np.divide(
        muscle_envelope - muscle_mean,
        muscle_std,
        out=np.zeros_like(muscle_envelope),
        where=muscle_std > 0,
    )
    muscle_score_array = np.sum(muscle_z, axis=0) / np.sqrt(muscle_z.shape[0])
    muscle_score_array = mne.filter.filter_data(
        muscle_score_array,
        sfreq=source_sfreq,
        l_freq=None,
        h_freq=4.0,
        verbose="ERROR",
    )
    if progress:
        progress.stage_done("muscle_band_hilbert_and_score", stage_started)

    stage_started = progress.stage_start("detector_validity_and_windows") if progress else None
    muscle_score_finite = np.isfinite(muscle_score_array)
    muscle_detector_valid = bool(
        finite.all()
        and muscle_score_finite.all()
        and np.isfinite(muscle_std).all()
        and np.any(muscle_std > 0)
    )
    muscle_flag = muscle_score_finite & (muscle_score_array > MUSCLE_Z_THRESHOLD)
    muscle_flag_fraction = float(np.mean(muscle_flag))

    duration_sec = float(raw.times[-1] + 1.0 / source_sfreq)
    window_count = int(np.floor((duration_sec - WINDOW_SEC) / STEP_SEC)) + 1
    window_rows: list[dict] = []
    for window_index in range(window_count):
        start_sec = window_index * STEP_SEC
        stop_sec = start_sec + WINDOW_SEC
        start = int(round(start_sec * source_sfreq))
        stop = int(round(stop_sec * source_sfreq))
        window = raw_uv[:, start:stop]
        window_finite = np.isfinite(window)
        channel_ptp = np.ptp(np.where(window_finite, window, np.nan), axis=1)
        row = {
            "dataset_id": "ds004148",
            "subject": subject,
            "session": f"ses-{SESSION}",
            "task": task,
            "window_index": window_index,
            "window_start_sec": start_sec,
            "window_end_sec": stop_sec,
            "finite_fraction": float(window_finite.mean()) if window_finite.size else float("nan"),
            "flatline_channel_count": int(sum(
                np.isfinite(channel).sum() > 1
                and float(np.nanmax(channel)) == float(np.nanmin(channel))
                for channel in window
            )),
            "max_channel_peak_to_peak_uv": float(np.nanmax(channel_ptp)),
            "p99_channel_peak_to_peak_uv": float(np.nanpercentile(channel_ptp, 99)),
            "max_absolute_amplitude_uv": float(np.nanmax(np.abs(window))),
            "muscle_score_finite_fraction": float(np.mean(muscle_score_finite[start:stop])),
            "muscle_detector_valid": muscle_detector_valid,
            "muscle_zscore_exceedance_fraction": float(np.mean(
                muscle_flag[start:stop]
            )),
        }
        window_rows.append(row)
    if progress:
        progress.stage_done("detector_validity_and_windows", stage_started)

    stage_started = progress.stage_start("hash_sidecars_and_close_recording") if progress else None
    source_files = {
        path.name: {"sha256": sha256(path), "size_bytes": path.stat().st_size}
        for path in required
        if path.suffix.lower() != ".eeg"
    }
    recording = {
        "dataset_id": "ds004148",
        "subject": subject,
        "session": f"ses-{SESSION}",
        "task": task,
        "unit_from_bids_channels_tsv": units[0],
        "channel_types": channel_types,
        "channel_count_bids": len(labels),
        "channel_count_brainvision": len(brainvision_labels),
        "channel_labels_bids_json": json.dumps(labels, ensure_ascii=False),
        "channel_labels_brainvision_json": json.dumps(brainvision_labels, ensure_ascii=False),
        "channel_order_sha256_bids": channel_label_hash(labels, ordered=True),
        "channel_order_sha256_brainvision": channel_label_hash(brainvision_labels, ordered=True),
        "channel_set_sha256_bids": channel_label_hash(labels, ordered=False),
        "channel_set_sha256_brainvision": channel_label_hash(brainvision_labels, ordered=False),
        "channel_order_matches_bids": channel_order_matches_bids,
        "channel_order_casefold_matches_bids": channel_order_casefold_matches_bids,
        "channel_label_case_mismatches_json": json.dumps(
            channel_label_case_mismatches, ensure_ascii=False
        ),
        "sampling_frequency_hz": source_sfreq,
        "reference_from_bids_json": eeg_json.get("EEGReference"),
        "software_filters_from_bids_json": eeg_json.get("SoftwareFilters"),
        "recording_duration_sec": duration_sec,
        "finite_sample_fraction": finite_fraction,
        "exact_flatline_channel_count": exact_flatlines,
        "absolute_amplitude_uv_percentiles_50_95_99_99_9_99_99": absolute_percentiles,
        "maximum_absolute_amplitude_uv": max_abs_uv,
        "median_channel_psd_line_49_51_to_total_1_45_ratio": line_noise_ratio,
        "line_noise_status": "descriptive_pre_notch; no pass_fail_threshold",
        "muscle_zscore_threshold": MUSCLE_Z_THRESHOLD,
        "muscle_filter_hz": list(MUSCLE_FILTER_HZ),
        "muscle_detector_valid": muscle_detector_valid,
        "muscle_score_finite_fraction": float(np.mean(muscle_score_finite)),
        "muscle_detector_active_channel_count": int(np.count_nonzero(muscle_std[:, 0] > 0)),
        "muscle_zscore_exceedance_fraction": muscle_flag_fraction,
        "muscle_score_p95": float(np.nanpercentile(muscle_score_array, 95)) if muscle_score_array.size else float("nan"),
        "muscle_score_max": float(np.nanmax(muscle_score_array)) if muscle_score_array.size else float("nan"),
        "muscle_qc_not_applied_to_feature_signal": True,
        "window_count_at_8s_4s_grid": window_count,
        "component_hashes": source_files,
    }
    raw.close()
    raw_qc.close()
    if progress:
        progress.stage_done("hash_sidecars_and_close_recording", stage_started)
    return recording, window_rows, {
        "subject": subject,
        "task": task,
        "units": units,
        "channel_count": len(labels),
        "channel_labels_bids": labels,
        "channel_labels_brainvision": brainvision_labels,
        "channel_order_sha256_bids": channel_label_hash(labels, ordered=True),
        "channel_order_sha256_brainvision": channel_label_hash(brainvision_labels, ordered=True),
        "channel_set_sha256_bids": channel_label_hash(labels, ordered=False),
        "channel_set_sha256_brainvision": channel_label_hash(brainvision_labels, ordered=False),
        "channel_order_matches_bids": channel_order_matches_bids,
        "channel_order_casefold_matches_bids": channel_order_casefold_matches_bids,
        "channel_label_case_mismatches": channel_label_case_mismatches,
        "sampling_frequency_hz": source_sfreq,
        "reference": eeg_json.get("EEGReference"),
        "duration_sec": duration_sec,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    started_utc = datetime.now(timezone.utc)
    started_monotonic = time.perf_counter()
    run_id = started_utc.strftime("%Y%m%dT%H%M%S%fZ")
    log_path = OUTPUT_ROOT / f"qc_run_{run_id}.log"
    logger = configure_logger(log_path)
    logger.info(
        "RUN_BOOTSTRAP_START run_id=%s pid=%d started_utc=%s python=%s source=%s output=%s",
        run_id,
        os.getpid(),
        started_utc.isoformat(),
        sys.version.split()[0],
        SOURCE_ROOT,
        OUTPUT_ROOT,
    )
    try:
        provenance = {
            "run_id": run_id,
            "started_utc": started_utc.isoformat(),
            "process_id": os.getpid(),
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "config_path": str(CONFIG_PATH),
            "config_sha256": sha256(CONFIG_PATH),
            "policy_path": str(POLICY_PATH),
            "policy_sha256": sha256(POLICY_PATH),
            "log_file": log_path.name,
        }
    except BaseException:
        logger.exception("RUN_PREFLIGHT_FAILED run_id=%s", run_id)
        raise

    total_recordings = len(SUBJECTS) * len(TASKS)
    progress = RunProgress(logger, total_recordings, started_monotonic)
    recording_rows: list[dict] = []
    window_rows: list[dict] = []
    metadata_rows: list[dict] = []
    try:
        progress.start()
        stage_started = progress.stage_start("scientific_library_imports")
        # Match mndm.src.mndm.preprocess: Numba/JIT startup can stall on Windows.
        os.environ.setdefault("MNE_USE_NUMBA", "false")
        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
        logger.info(
            "IMPORT_RUNTIME_FLAGS MNE_USE_NUMBA=%s NUMBA_DISABLE_JIT=%s",
            os.environ.get("MNE_USE_NUMBA"),
            os.environ.get("NUMBA_DISABLE_JIT"),
        )
        global mne, np, hilbert, welch
        import mne
        import numpy as np
        from scipy.signal import hilbert, welch
        progress.stage_done("scientific_library_imports", stage_started)
        logger.info(
            "RUN_START run_id=%s pid=%d started_utc=%s recordings=%d subjects=%d tasks=%s "
            "window_sec=%.1f step_sec=%.1f source=%s output=%s python=%s mne=%s numpy=%s "
            "runner_sha256=%s config_sha256=%s policy_sha256=%s log=%s",
            run_id,
            os.getpid(),
            started_utc.isoformat(),
            total_recordings,
            len(SUBJECTS),
            ",".join(TASKS),
            WINDOW_SEC,
            STEP_SEC,
            SOURCE_ROOT,
            OUTPUT_ROOT,
            sys.version.split()[0],
            mne.__version__,
            np.__version__,
            provenance["runner_sha256"],
            provenance["config_sha256"],
            provenance["policy_sha256"],
            log_path,
        )
        for subject_index, subject in enumerate(SUBJECTS):
            for task_index, task in enumerate(TASKS):
                recording_index = subject_index * len(TASKS) + task_index + 1
                recording_started = time.perf_counter()
                progress.start_recording(recording_index, subject, task)
                recording, windows, metadata = assess_one(subject, task, progress=progress)
                recording_rows.append(recording)
                window_rows.extend(windows)
                metadata_rows.append(metadata)
                progress.recording_done(len(windows), time.perf_counter() - recording_started)

        stage_started = progress.stage_start("write_qc_recordings_csv")
        write_csv(OUTPUT_ROOT / "qc_recordings.csv", recording_rows)
        progress.stage_done("write_qc_recordings_csv", stage_started)

        stage_started = progress.stage_start("write_qc_windows_csv")
        write_csv(OUTPUT_ROOT / "qc_windows.csv", window_rows)
        progress.stage_done("write_qc_windows_csv", stage_started)

        results_ready_utc = datetime.now(timezone.utc)
        elapsed_before_manifest_write = time.perf_counter() - started_monotonic
        manifest = {
            "schema": "mndm.ds004148_pre_anchor_qc_assessment.v1_2",
            "run_provenance": {
                **provenance,
                "results_ready_utc": results_ready_utc.isoformat(),
                "elapsed_before_manifest_write_seconds": round(elapsed_before_manifest_write, 3),
            },
            "selection": {
                "subjects": list(SUBJECTS),
                "sessions": [SESSION],
                "tasks": list(TASKS),
                "selection_basis": "metadata-frozen all-pairs session-1 EO/EC cohort; no feature or coordinate values used",
            },
            "methods": {
                "technical_checks": ["BIDS units/channel order/sampling metadata", "finite values", "exact flatlines"],
                "channel_order_check": "exact BrainVision/BIDS label order is recorded; casefold-equivalent capitalization differences are warnings; different order/identity remains unsupported",
                "amplitude_and_peak_to_peak": "continuous descriptors only; no amplitude threshold or automatic amplitude exclusion",
                "line_noise": "median channel Welch PSD ratio 49-51 Hz / 1-45 Hz before the configured 50 Hz notch; descriptive only",
                "muscle_artifact": "pilot's MNE-inspired filter/Hilbert score on an average-reference, 50/100/150 Hz-notched QC-only copy; 110-140 Hz; threshold >4; any-overlap with 8 s feature window rejects that window for primary summaries; no min_length_good gap merging; nonfinite source/detector or no active envelope channels makes the detector unassessed and windows ineligible",
                "artifact_correction": "none; QC scores do not alter or reject feature signals",
                "adc_clipping": "not assessed because BrainVision IEEE float samples do not declare verified ADC rails",
                "eog_ecg": "not available in source BIDS metadata (0 EOG, 0 ECG channels)",
                "qc_flag_contract": "NMD qc_ok_eeg remains unassessed unless the ingest artifact-QC pipeline is actually applied; this separate QC report does not set it",
                "raw_signal_hashes": "not computed; BIDS/header sidecars are hashed",
            },
            "runtime": {
                "mne_version": mne.__version__,
                "numpy_version": np.__version__,
                "mne_use_numba": os.environ.get("MNE_USE_NUMBA"),
                "numba_disable_jit": os.environ.get("NUMBA_DISABLE_JIT"),
            },
            "recording_metadata": metadata_rows,
            "artifact_qc_applied_to_features": False,
            "qualification": "technical/QC description only; not evidence of artifact-free EEG or qualified external-anchor comparability",
        }
        stage_started = progress.stage_start("write_qc_manifest")
        (OUTPUT_ROOT / "qc_assessment_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
        progress.stage_done("write_qc_manifest", stage_started)
        logger.info(
            "RUN_COMPLETE run_id=%s completed=%d/%d recordings=%d windows=%d elapsed_sec=%.3f output=%s",
            run_id,
            progress.completed,
            total_recordings,
            len(recording_rows),
            len(window_rows),
            time.perf_counter() - started_monotonic,
            OUTPUT_ROOT,
        )
        print(f"recordings={len(recording_rows)} windows={len(window_rows)} output={OUTPUT_ROOT}")
    except BaseException:
        logger.exception(
            "RUN_ABORTED run_id=%s completed=%d/%d elapsed_sec=%.3f",
            run_id,
            progress.completed,
            total_recordings,
            time.perf_counter() - started_monotonic,
        )
        raise
    finally:
        progress.stop()


if __name__ == "__main__":
    main()
