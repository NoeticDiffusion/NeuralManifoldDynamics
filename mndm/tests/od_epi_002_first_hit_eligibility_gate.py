"""Run the metadata-only OD-EPI-002 first-hit eligibility recount.

This runner never calls a committor estimator. It joins the existing BIDS
events and feature table, audits window support, and optionally replays the
already-produced OD-EPI-001 HDF5 refusal surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path, PureWindowsPath
from typing import Any

import h5py
import numpy as np
import pandas as pd

from mndm.dynamical_families.io import resolve_family_group
from mndm.dynamical_families.first_hit_eligibility import (
    OD_EPI_002_PROTOCOL_ID,
    OD_EPI_002_SCHEMA_VERSION,
    FirstHitEligibilityProtocol,
    audit_first_hit_windows,
    explicit_reaction_coordinate_present,
    finite_quantiles,
    json_safe,
    pair_first_onset_with_next_offset,
    subject_split_leaks,
)
from mndm.pipeline.event_annotations import load_event_table_from_bids_events


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return value is None


def _relative_path(root: Path, value: Any) -> Path | None:
    if value is None or _is_missing(value):
        return None
    raw = str(value).replace("\\", "/").strip()
    if not raw:
        return None
    return root / Path(raw)


def _basename(value: Any) -> str:
    return PureWindowsPath(str(value).replace("/", "\\")).name


def _clean_text(value: Any) -> str | None:
    if value is None or _is_missing(value):
        return None
    text = str(value).strip()
    return text if text else None


def _clean_float(value: Any) -> float | None:
    if value is None or _is_missing(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _parse_entities(file_name: str) -> dict[str, str | None]:
    stem = Path(file_name).name
    entities: dict[str, str | None] = {
        "subject": None,
        "session": None,
        "task": None,
        "run": None,
        "acq": None,
    }
    for key in entities:
        match = re.search(rf"(?:^|_)({key})-([^_]+)", stem)
        if match:
            entities[key] = match.group(2)
    return entities


def _read_sidecar(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _audit_reference_hdf5(run_dir: Path | None) -> dict[str, Any]:
    """Check the already-produced HDF5 refusal surface without estimating q."""
    if run_dir is None or not run_dir.exists():
        return {
            "available": False,
            "h5_count": 0,
            "committor_values_computed": False,
            "status_counts": [],
        }
    status_counts: Counter[tuple[str, str, str | None]] = Counter()
    h5_paths = sorted(run_dir.rglob("*.h5"))
    q_series_present = 0
    for path in h5_paths:
        with h5py.File(path, "r") as handle:
            for family_id, report_name in (
                ("destination", "committor"),
                ("resilience", "finite_amplitude_resilience"),
            ):
                resolved = resolve_family_group(handle, family_id)
                group = resolved["group"]
                if group is None:
                    status_counts[(report_name, "missing", None)] += 1
                    continue
                status_value = group["computation_status"][()]
                status = (
                    status_value.decode("utf-8")
                    if isinstance(status_value, bytes)
                    else str(status_value)
                )
                reason = None
                if "failure_reason" in group:
                    reason_value = group["failure_reason"][()]
                    reason = (
                        reason_value.decode("utf-8")
                        if isinstance(reason_value, bytes)
                        else str(reason_value)
                    )
                status_counts[(report_name, status, reason)] += 1
                if "series/q_A_to_B" in group or "series/q_hat" in group:
                    q_series_present += 1
    return {
        "available": True,
        "run_dir": str(run_dir),
        "h5_count": len(h5_paths),
        "committor_values_computed": q_series_present > 0,
        "status_counts": [
            {
                "family": family,
                "computation_status": status,
                "failure_reason": reason,
                "count": count,
            }
            for (family, status, reason), count in sorted(status_counts.items())
        ],
    }


def run_audit(
    *,
    raw_root: Path,
    processed_root: Path,
    output_path: Path,
    reference_run_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the OD-EPI-002 metadata-only recount and write JSON."""
    protocol = FirstHitEligibilityProtocol()
    protocol_path = (
        Path(__file__).resolve().parents[2]
        / "project"
        / "orthagonal_axis"
        / "od_epi_002_preregistration.md"
    )
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    file_index_path = processed_root / "file_index.csv"
    feature_path = processed_root / "features.parquet"
    if not feature_path.exists():
        feature_path = processed_root / "features.csv"
    file_index = pd.read_csv(file_index_path)
    features = (
        pd.read_parquet(feature_path)
        if feature_path.suffix.lower() == ".parquet"
        else pd.read_csv(feature_path)
    )
    features["file_basename"] = features["file"].map(_basename)
    grouped_features = {
        str(name): frame.sort_values(["t_start", "epoch_id"], kind="stable")
        for name, frame in features.groupby("file_basename", sort=False)
    }

    records: list[dict[str, Any]] = []
    event_file_count = 0
    event_pair_count = 0
    task_counts = Counter(str(value) for value in file_index["task"].dropna())
    for row in file_index.to_dict(orient="records"):
        file_name = _basename(row.get("path", ""))
        entities = _parse_entities(file_name)
        task = _clean_text(row.get("task")) or entities.get("task") or ""
        feature_frame = grouped_features.get(file_name, pd.DataFrame())
        event_path = _relative_path(raw_root, row.get("events_tsv"))
        sidecar_path = _relative_path(raw_root, row.get("eeg_json"))
        sidecar = _read_sidecar(sidecar_path)
        recording_duration = sidecar.get("RecordingDuration")
        sampling_frequency = sidecar.get("SamplingFrequency")
        event_table = load_event_table_from_bids_events(
            event_path,
            event_types=[protocol.onset_event_type, protocol.offset_event_type],
            trial_type_column="trial_type",
        ) if event_path is not None else None
        if event_table is not None and event_table.n:
            event_file_count += 1
        pair = None
        if event_table is not None and event_table.n and event_table.event_type is not None:
            pair = pair_first_onset_with_next_offset(
                event_table.event_type,
                event_table.onset_sec,
                onset_event_type=protocol.onset_event_type,
                offset_event_type=protocol.offset_event_type,
            )
        if pair is not None:
            event_pair_count += 1
        onset_sec = pair[0] if pair else None
        offset_sec = pair[1] if pair else None
        if feature_frame.empty:
            t_start = np.empty(0, dtype=float)
            t_end = np.empty(0, dtype=float)
            epoch_id = np.empty(0, dtype=np.int64)
            qc_ok = np.empty(0, dtype=bool)
            qc_column_present = False
        else:
            t_start = pd.to_numeric(feature_frame["t_start"], errors="coerce").to_numpy()
            t_end = pd.to_numeric(feature_frame["t_end"], errors="coerce").to_numpy()
            epoch_id = pd.to_numeric(
                feature_frame["epoch_id"], errors="coerce"
            ).fillna(-1).to_numpy(dtype=np.int64)
            qc_column_present = "qc_ok_eeg" in feature_frame.columns
            qc_ok = (
                feature_frame["qc_ok_eeg"].fillna(False).astype(bool).to_numpy()
                if qc_column_present
                else np.ones(t_start.size, dtype=bool)
            )
        rc_present = explicit_reaction_coordinate_present(
            feature_frame.columns if not feature_frame.empty else [],
            protocol.reaction_coordinate_key,
        )
        audit = audit_first_hit_windows(
            onset_sec=onset_sec,
            offset_sec=offset_sec,
            t_start=t_start,
            t_end=t_end,
            epoch_id=epoch_id,
            qc_ok=qc_ok,
            reaction_coordinate_column_present=rc_present,
            task=task,
            protocol=protocol,
        )
        if not qc_column_present:
            audit["failure_reasons"].append("eeg_qc_column_absent")
            audit["is_continuous_first_hit_candidate"] = False
        audit.update(
            {
                "recording": file_name,
                "subject": _clean_text(row.get("subject")) or entities.get("subject"),
                "session": _clean_text(row.get("session")) or entities.get("session"),
                "run": _clean_text(row.get("run")) or entities.get("run"),
                "acq": _clean_text(row.get("acq")) or entities.get("acq"),
                "events_tsv": str(event_path) if event_path else None,
                "recording_duration_sec": _clean_float(recording_duration),
                "sampling_frequency_hz": _clean_float(sampling_frequency),
                "pre_onset_sidecar_sec": _clean_float(onset_sec),
                "n_feature_rows": int(len(feature_frame)),
            }
        )
        records.append(audit)

    ictal_records = [record for record in records if record["task"] == "ictal"]
    qualified_records = [
        record for record in ictal_records if record["is_continuous_first_hit_candidate"]
    ]
    complete_support_records = [
        record
        for record in ictal_records
        if (
            record["n_stable_A_windows_fully_contained"]
            >= protocol.min_stable_a_windows
            and record["n_B_core_windows_fully_contained"]
            >= protocol.min_b_core_windows
        )
    ]
    adapter_floor_met = (
        len(qualified_records) >= protocol.adapter_min_transition_segments
    )
    analysis_floor_met = (
        len(qualified_records) >= protocol.analysis_min_transition_segments
    )
    qualified_after_segment_floors = (
        qualified_records if adapter_floor_met and analysis_floor_met else []
    )
    subjects = sorted(
        {str(record["subject"]) for record in ictal_records if record["subject"]}
    )
    ictal_per_subject = Counter(str(record["subject"]) for record in ictal_records)
    subject_assignments = {
        record["recording"]: {"subject": record["subject"], "split": "eligibility"}
        for record in records
    }
    status_counts = Counter(
        reason
        for record in ictal_records
        for reason in record["failure_reasons"]
    )
    result = {
        "schema": OD_EPI_002_SCHEMA_VERSION,
        "dataset": "ds004100",
        "protocol_id": OD_EPI_002_PROTOCOL_ID,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_hash,
        "raw_root": str(raw_root),
        "processed_root": str(processed_root),
        "features_path": str(feature_path),
        "protocol": protocol.as_dict(),
        "recording_inventory": {
            "feature_rows": int(len(features)),
            "feature_files": int(features["file_basename"].nunique()),
            "file_index_rows": int(len(file_index)),
            "task_counts": dict(sorted(task_counts.items())),
            "subjects_total": int(file_index["subject"].nunique()),
            "subjects_with_ictal": int(
                file_index.loc[file_index["task"] == "ictal", "subject"].nunique()
            ),
            "subjects_with_interictal": int(
                file_index.loc[file_index["task"] == "interictal", "subject"].nunique()
            ),
            "events_tsv_files_with_selected_events": event_file_count,
            "ordered_onset_offset_pairs": event_pair_count,
            "ingest_structure": "separate_ictal_interictal_excerpts",
            "explicit_first_hit_A_B_protocol_present": True,
            "explicit_reaction_coordinate_column_present": False,
            "onset_offset_markers_inside_recorded_excerpts": True,
        },
        "eligibility_rollup": {
            "candidate_ictal_recordings_with_onset_offset_markers": int(
                sum(record["onset_sec"] is not None and record["offset_sec"] is not None for record in ictal_records)
            ),
            "n_with_onset_on_grid": int(
                sum(record["onset_aligned_to_grid"] for record in ictal_records)
            ),
            "n_with_stable_A_ge_min_windows": int(
                sum(
                    record["n_stable_A_windows_fully_contained"]
                    >= protocol.min_stable_a_windows
                    for record in ictal_records
                )
            ),
            "n_with_B_core_ge_min_windows": int(
                sum(
                    record["n_B_core_windows_fully_contained"]
                    >= protocol.min_b_core_windows
                    for record in ictal_records
                )
            ),
            "n_with_no_gap_break": int(
                sum(
                    not (
                        record["gap_or_qc_break_in_A"]
                        or record["gap_or_qc_break_in_B"]
                        or record["gap_or_qc_break_in_A_to_B"]
                    )
                    for record in complete_support_records
                )
            ),
            "n_protocol_candidate_first_hit_recordings": len(qualified_records),
            "adapter_transition_segment_floor_met": adapter_floor_met,
            "analysis_transition_segment_floor_met": analysis_floor_met,
            "qualified_continuous_first_hit_A_to_B_trajectories": len(
                qualified_after_segment_floors
            ),
            "n_subjects_with_ge_1_qualified_trajectory": len(
                {str(record["subject"]) for record in qualified_after_segment_floors}
            ),
            "ictal_recordings_per_subject": finite_quantiles(
                list(ictal_per_subject.values())
            ),
            "stable_A_window_count_quantiles": finite_quantiles(
                [
                    record["n_stable_A_windows_fully_contained"]
                    for record in ictal_records
                ]
            ),
            "onset_smear_window_count_quantiles": finite_quantiles(
                [record["n_onset_smear_windows"] for record in ictal_records]
            ),
            "failure_reason_counts": dict(status_counts),
        },
        "recordings": records,
        "reference_hdf5": _audit_reference_hdf5(reference_run_dir),
        "eligibility_decision": {
            "committor_onset": {
                "status": "NOT_TESTABLE",
                "qualified_continuous_first_hit_A_to_B_trajectories": len(
                    qualified_after_segment_floors
                ),
                "reason": [
                    "separate_ictal_interictal_excerpts",
                    "explicit_reaction_coordinate_column_absent",
                    "no_committor_values_computed",
                ],
            },
            "committor_recovery": {
                "status": "NOT_TESTABLE",
                "reason": "recovery_protocol_not_scored_in_OD-EPI-002_onset_gate",
            },
            "diffusion": {
                "status": "UNCHANGED_FROM_OD-EPI-001",
                "reason": "OD-EPI-002 does not modify or rerun diffusion.",
            },
            "resilience": {
                "status": "NOT_TESTABLE",
                "reason": "no_perturbation_protocol",
            },
        },
        "fail_closed_assertions": {
            "committor_values_computed": False,
            "qualified_overlay_used": False,
            "task_labels_mapped_to_A_B": False,
            "reaction_coordinate_inferred": False,
            "excerpts_pooled_across_files": False,
            "subject_split_leak": subject_split_leaks(subject_assignments),
        },
        "continuous_first_hit_A_to_B_trajectories_established": False,
        "claim_boundary": (
            "Metadata-only protocol eligibility audit. No committor values, "
            "biological contrasts, or A/B labels inferred from task membership."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(json_safe(result), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-run-dir", type=Path, default=None)
    args = parser.parse_args()
    result = run_audit(
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        output_path=args.output,
        reference_run_dir=args.reference_run_dir,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "file_index_rows": result["recording_inventory"]["file_index_rows"],
                "feature_rows": result["recording_inventory"]["feature_rows"],
                "qualified_trajectories": result["eligibility_rollup"][
                    "qualified_continuous_first_hit_A_to_B_trajectories"
                ],
                "committor_values_computed": result["fail_closed_assertions"][
                    "committor_values_computed"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
