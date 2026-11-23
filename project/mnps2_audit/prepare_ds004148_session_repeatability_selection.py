"""Freeze the ds004148 three-session EC/EO selection from BIDS metadata only."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SOURCE_ROOT = Path(r"K:/ExternalReceivedDatasets/openneuro/received/ds004148")
OUTPUT_ROOT = Path(
    r"M:/repos/NeuralManifoldDynamics/project/mnps2_audit/outputs/"
    r"20261118_ds004148_v3_4_session_repeatability"
)
SESSIONS = ("session1", "session2", "session3")
TASKS = ("eyesclosed", "eyesopen")
EXPECTED_SUBJECTS = tuple(f"sub-{number:02d}" for number in range(1, 61))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_recording(subject: str, session: str, task: str) -> dict[str, Any]:
    eeg_dir = SOURCE_ROOT / subject / f"ses-{session}" / "eeg"
    stem = f"{subject}_ses-{session}_task-{task}_eeg"
    paths = {
        "vhdr": eeg_dir / f"{stem}.vhdr",
        "eeg": eeg_dir / f"{stem}.eeg",
        "vmrk": eeg_dir / f"{stem}.vmrk",
        "json": eeg_dir / f"{stem}.json",
        "channels": eeg_dir / f"{subject}_ses-{session}_task-{task}_channels.tsv",
    }
    candidates = sorted(eeg_dir.glob(f"{subject}_ses-{session}_task-{task}*_eeg.vhdr")) if eeg_dir.is_dir() else []
    missing = [name for name, path in paths.items() if not path.is_file()]
    row: dict[str, Any] = {
        "subject": subject,
        "session": session,
        "task": task,
        "stem": stem,
        "relative_paths": {
            name: path.relative_to(SOURCE_ROOT).as_posix()
            for name, path in paths.items()
        },
        "candidate_brainvision_headers": [path.relative_to(SOURCE_ROOT).as_posix() for path in candidates],
        "missing_components": missing,
    }
    if missing:
        row["metadata_status"] = "incomplete_components"
        return row
    if candidates != [paths["vhdr"]]:
        row["metadata_status"] = "ambiguous_task_recordings"
        return row

    eeg_json = json.loads(paths["json"].read_text(encoding="utf-8"))
    with paths["channels"].open(encoding="utf-8-sig", newline="") as stream:
        channel_rows = list(csv.DictReader(stream, delimiter="\t"))
    labels = [str(item.get("name", "")).strip() for item in channel_rows]
    types = [str(item.get("type", "")).strip().lower() for item in channel_rows]
    units = sorted({str(item.get("units", "")).strip() for item in channel_rows})
    try:
        sfreqs = sorted({float(item.get("sampling_frequency", "")) for item in channel_rows})
    except ValueError:
        sfreqs = []
    row.update({
        "metadata_status": "complete",
        "channel_count": len(channel_rows),
        "ordered_labels": labels,
        "channel_types": sorted(set(types)),
        "units": units,
        "sampling_frequency_hz": sfreqs,
        "json_sampling_frequency_hz": eeg_json.get("SamplingFrequency"),
        "reference": eeg_json.get("EEGReference"),
        "software_filters": eeg_json.get("SoftwareFilters"),
        "sidecar_sha256": {
            name: sha256(paths[name]) for name in ("vhdr", "vmrk", "json", "channels")
        },
        "component_sizes_bytes": {name: path.stat().st_size for name, path in paths.items()},
    })
    row["metadata_valid"] = (
        units == ["uV"]
        and sfreqs == [500.0]
        and eeg_json.get("SamplingFrequency") == 500
        and sorted(set(types)) == ["eeg"]
        and bool(labels)
        and len(labels) == len(set(labels))
    )
    if not row["metadata_valid"]:
        row["metadata_status"] = "unexpected_units_frequency_or_channel_type_or_labels"
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(f"Missing BIDS dataset root: {SOURCE_ROOT}")
    actual_subjects = tuple(sorted(
        (path.name for path in SOURCE_ROOT.iterdir() if path.is_dir() and re.fullmatch(r"sub-\d+", path.name)),
        key=lambda value: int(value.split("-")[1]),
    ))
    missing_subjects = sorted(set(EXPECTED_SUBJECTS) - set(actual_subjects))
    unexpected_subjects = sorted(set(actual_subjects) - set(EXPECTED_SUBJECTS))
    recordings = [
        read_recording(subject, session, task)
        for subject in EXPECTED_SUBJECTS
        for session in SESSIONS
        for task in TASKS
    ]
    record_by_key = {(row["subject"], row["session"], row["task"]): row for row in recordings}
    pairs: list[dict[str, Any]] = []
    for subject in EXPECTED_SUBJECTS:
        for session in SESSIONS:
            ec = record_by_key[(subject, session, "eyesclosed")]
            eo = record_by_key[(subject, session, "eyesopen")]
            complete = ec["metadata_status"] == eo["metadata_status"] == "complete"
            pairs.append({
                "subject": subject,
                "session": session,
                "ec_file": ec["relative_paths"]["vhdr"],
                "eo_file": eo["relative_paths"]["vhdr"],
                "pair_included_for_computation": complete,
                "same_ordered_channel_labels": bool(complete and ec["ordered_labels"] == eo["ordered_labels"]),
                "same_declared_reference": bool(complete and ec["reference"] == eo["reference"]),
                "same_sampling_frequency_metadata": bool(complete and ec["sampling_frequency_hz"] == eo["sampling_frequency_hz"]),
                "same_declared_units": bool(complete and ec["units"] == eo["units"]),
                "ec_channel_count": ec.get("channel_count"),
                "eo_channel_count": eo.get("channel_count"),
                "ec_reference": ec.get("reference"),
                "eo_reference": eo.get("reference"),
                "exclusion_reason": ";".join(
                    f"{task}:{row['metadata_status']}:{','.join(row.get('missing_components', []))}"
                    for task, row in (("EC", ec), ("EO", eo))
                    if row["metadata_status"] != "complete"
                ) or None,
            })

    output = OUTPUT_ROOT / "selection"
    output.mkdir(parents=True, exist_ok=True)
    recordings_path = output / "recordings_metadata.csv"
    pairs_path = output / "participant_session_pairs.csv"
    write_csv(recordings_path, recordings)
    write_csv(pairs_path, pairs)
    description = json.loads((SOURCE_ROOT / "dataset_description.json").read_text(encoding="utf-8"))
    gate_pass = (
        not missing_subjects
        and not unexpected_subjects
        and all(row["metadata_status"] == "complete" and row.get("metadata_valid") for row in recordings)
        and all(row["pair_included_for_computation"] for row in pairs)
    )
    manifest = {
        "schema": "mndm.ds004148_v34_session_repeatability_selection.v1",
        "frozen_before_coordinate_inspection": True,
        "metadata_only": True,
        "dataset_description": {key: description.get(key) for key in ("Name", "BIDSVersion", "DatasetDOI", "DatasetType")},
        "subjects_expected": list(EXPECTED_SUBJECTS),
        "sessions": list(SESSIONS),
        "tasks": list(TASKS),
        "selection_basis": "All 60 participants and their session-1, session-2, and session-3 resting EC/EO pairs selected from BIDS metadata before examining signal, feature, or coordinate values.",
        "expected_recordings": len(EXPECTED_SUBJECTS) * len(SESSIONS) * len(TASKS),
        "metadata_complete_recordings": sum(row["metadata_status"] == "complete" and row.get("metadata_valid", False) for row in recordings),
        "complete_ec_eo_session_pairs": sum(row["pair_included_for_computation"] for row in pairs),
        "expected_ec_eo_session_pairs": len(pairs),
        "actual_subjects": list(actual_subjects),
        "missing_subjects": missing_subjects,
        "unexpected_subjects": unexpected_subjects,
        "metadata_gate_pass": gate_pass,
        "pair_surface_mismatch_count": sum(
            not all(row[field] for field in ("same_ordered_channel_labels", "same_declared_reference", "same_sampling_frequency_metadata", "same_declared_units"))
            for row in pairs if row["pair_included_for_computation"]
        ),
        "recordings_csv": recordings_path.name,
        "recordings_csv_sha256": sha256(recordings_path),
        "pairs_csv": pairs_path.name,
        "pairs_csv_sha256": sha256(pairs_path),
        "raw_eeg_binary_sha256_computed": False,
        "recordings": recordings,
    }
    manifest_path = output / "selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in (
        "expected_recordings", "metadata_complete_recordings", "complete_ec_eo_session_pairs",
        "expected_ec_eo_session_pairs", "pair_surface_mismatch_count", "metadata_gate_pass",
        "missing_subjects", "unexpected_subjects",
    )}, indent=2))
    if not gate_pass:
        raise SystemExit("Metadata gate failed; stop before features or QC.")


if __name__ == "__main__":
    main()
