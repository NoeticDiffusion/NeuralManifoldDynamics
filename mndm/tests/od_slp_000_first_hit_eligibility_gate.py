"""Run the source-level, audit-only OD-SLP-000 BOAS gate.

The runner pairs PSG and headband acquisitions directly from BIDS source
files. It does not use the PSG-only MNDM overlays, does not write HDF5, and
never imports or invokes a committor estimator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mndm.dynamical_families.sleep_first_hit_eligibility import (
    SleepFirstHitProtocol,
    audit_stage_intervals,
    headband_slow_fast_logratio,
    json_safe,
    pid_split_leaks,
)


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return value is None


def _clean_float(value: Any) -> float | None:
    if value is None or _is_missing(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _clean_pid(value: Any) -> str | None:
    numeric = _clean_float(value)
    if numeric is not None and numeric.is_integer():
        return str(int(numeric))
    if value is None or _is_missing(value):
        return None
    text = str(value).strip()
    return text if text else None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _parse_entities(path: Path) -> dict[str, str | None]:
    name = path.name
    output: dict[str, str | None] = {
        "participant_id": None,
        "session": None,
        "task": None,
        "run": None,
    }
    for key in output:
        token = "sub" if key == "participant_id" else key
        match = re.search(rf"(?:^|_)({token})-([^_]+)", name)
        if match:
            output[key] = (
                f"sub-{match.group(2)}"
                if key == "participant_id"
                else match.group(2)
            )
    return output


def _event_path(edf_path: Path) -> Path:
    return edf_path.with_name(edf_path.name.replace("_eeg.edf", "_events.tsv"))


def _sidecar_path(edf_path: Path) -> Path:
    return edf_path.with_name(edf_path.name.replace("_eeg.edf", "_eeg.json"))


def _channels_path(edf_path: Path) -> Path:
    return edf_path.with_name(edf_path.name.replace("_eeg.edf", "_channels.tsv"))


def _read_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t")
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


def _read_channels(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t")
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


def _event_geometry_matches(psg: pd.DataFrame, headband: pd.DataFrame) -> bool:
    required = {"onset", "duration"}
    if not required.issubset(psg.columns) or not required.issubset(headband.columns):
        return False
    left = psg[["onset", "duration"]].apply(pd.to_numeric, errors="coerce").to_numpy()
    right = headband[["onset", "duration"]].apply(pd.to_numeric, errors="coerce").to_numpy()
    return bool(
        left.shape == right.shape
        and np.all(np.isfinite(left))
        and np.all(np.isfinite(right))
        and np.allclose(left, right, atol=1e-6, rtol=0.0)
    )


def _load_pid_map(raw_root: Path) -> dict[str, str | None]:
    candidates = sorted(raw_root.rglob("participants.tsv"))
    if not candidates:
        return {}
    try:
        table = pd.read_csv(candidates[0], sep="\t")
    except (OSError, ValueError, pd.errors.ParserError):
        return {}
    if "participant_id" not in table.columns or "pid" not in table.columns:
        return {}
    return {
        str(row["participant_id"]).strip(): _clean_pid(row["pid"])
        for row in table.to_dict(orient="records")
    }


def _pair_acquisitions(raw_root: Path) -> list[dict[str, Any]]:
    psg_paths = sorted(raw_root.rglob("*_acq-psg_eeg.edf"))
    pairs: list[dict[str, Any]] = []
    for psg_path in psg_paths:
        headband_path = psg_path.with_name(
            psg_path.name.replace("_acq-psg_", "_acq-headband_")
        )
        entities = _parse_entities(psg_path)
        pairs.append(
            {
                "participant_id": entities["participant_id"],
                "session": entities["session"],
                "task": entities["task"],
                "run": entities["run"],
                "psg_path": psg_path,
                "headband_path": headband_path,
            }
        )
    return pairs


def _read_headband_rc(
    path: Path,
    *,
    intervals: list[tuple[float, float]],
    protocol: SleepFirstHitProtocol,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read one headband EDF and calculate the frozen RC offline."""
    try:
        import mne

        raw = mne.io.read_raw_edf(
            str(path),
            preload=True,
            verbose="ERROR",
        )
        picks = [raw.ch_names.index(channel) for channel in protocol.rc_channels]
        data = raw.get_data(picks=picks)
        sfreq = float(raw.info["sfreq"])
        values = headband_slow_fast_logratio(
            data,
            sampling_frequency_hz=sfreq,
            intervals=intervals,
            protocol=protocol,
        )
        del raw
        return values, {
            "available": True,
            "sampling_frequency_hz": sfreq,
            "channel_names": list(protocol.rc_channels),
        }
    except Exception as exc:
        return np.full(len(intervals), np.nan), {
            "available": False,
            "failure_reason": f"headband_rc_read_failed:{type(exc).__name__}",
        }


def _audit_pair(
    pair: dict[str, Any],
    *,
    pid_map: dict[str, str | None],
    protocol: SleepFirstHitProtocol,
) -> dict[str, Any]:
    psg_path = pair["psg_path"]
    headband_path = pair["headband_path"]
    psg_sidecar = _read_json(_sidecar_path(psg_path))
    headband_sidecar = _read_json(_sidecar_path(headband_path))
    psg_events = _read_events(_event_path(psg_path))
    headband_events = _read_events(_event_path(headband_path))
    psg_duration = _clean_float(psg_sidecar.get("RecordingDuration"))
    headband_duration = _clean_float(headband_sidecar.get("RecordingDuration"))
    psg_sampling_frequency = _clean_float(
        psg_sidecar.get("SamplingFrequency")
    )
    headband_sampling_frequency = _clean_float(
        headband_sidecar.get("SamplingFrequency")
    )
    pid = pid_map.get(pair["participant_id"])
    reasons: list[str] = []
    if pid is None:
        reasons.append("pid_missing")
    if not headband_path.exists():
        reasons.append("headband_missing")
    channels = _read_channels(_channels_path(headband_path))
    channel_names = (
        set(channels["name"].astype(str).str.strip())
        if "name" in channels.columns
        else set()
    )
    if not set(protocol.rc_channels).issubset(channel_names):
        reasons.append("headband_channels_incomplete")
    if protocol.stage_column not in psg_events.columns:
        reasons.append("missing_stage_hum")
    if psg_duration is None or headband_duration is None:
        reasons.append("recording_duration_missing")
    elif abs(psg_duration - headband_duration) > protocol.synchronization_tolerance_sec:
        reasons.append("alignment_invalid_or_unproven")
    if psg_sampling_frequency is None or headband_sampling_frequency is None:
        reasons.append("sampling_frequency_missing")
    if not _event_geometry_matches(psg_events, headband_events):
        reasons.append("alignment_invalid_or_unproven")

    stage_audit: dict[str, Any]
    rc_values = np.full(0, np.nan)
    rc_meta: dict[str, Any] = {"available": False}
    if {"onset", "duration", protocol.stage_column}.issubset(psg_events.columns):
        stage_audit = audit_stage_intervals(
            onsets_sec=pd.to_numeric(psg_events["onset"], errors="coerce").to_numpy(),
            durations_sec=pd.to_numeric(
                psg_events["duration"], errors="coerce"
            ).to_numpy(),
            stages=psg_events[protocol.stage_column].tolist(),
            recording_duration_sec=psg_duration,
            protocol=protocol,
        )
        reasons.extend(stage_audit["failure_reasons"])
        intervals = list(
            zip(
                stage_audit["interval_onsets_sec"],
                stage_audit["interval_ends_sec"],
            )
        )
        if not reasons or all(
            reason not in {"headband_missing", "headband_channels_incomplete"}
            for reason in reasons
        ):
            rc_values, rc_meta = _read_headband_rc(
                headband_path,
                intervals=intervals,
                protocol=protocol,
            )
            if not rc_meta.get("available", False):
                reasons.append(str(rc_meta.get("failure_reason", "external_rc_unavailable")))
            elif (
                headband_sampling_frequency is not None
                and rc_meta.get("sampling_frequency_hz") is not None
                and abs(
                    float(rc_meta["sampling_frequency_hz"])
                    - float(headband_sampling_frequency)
                )
                > 1e-6
            ):
                reasons.append("sampling_frequency_mismatch")
    else:
        stage_audit = {
            "dense_grid_available": False,
            "stage_hum_available": False,
            "n3_core_windows": 0,
            "rem_core_windows": 0,
            "n2_interior_windows": 0,
            "candidate_n2_blocks": 0,
            "segments": [],
            "failure_reasons": [],
        }

    segments = []
    for segment in stage_audit.get("segments", []):
        start = int(segment["candidate_interval_index"])
        end = int(segment["outcome_interval_index"])
        finite = (
            rc_values.size > end
            and bool(np.all(np.isfinite(rc_values[start : end + 1])))
        )
        item = dict(segment)
        item["external_rc_finite"] = finite
        item["night_stratum"] = (
            "early_night"
            if psg_duration is not None
            and float(segment["start_sec"]) < 0.5 * psg_duration
            else "late_night"
        )
        segments.append(item)
    if rc_values.size:
        finite_fraction = float(np.mean(np.isfinite(rc_values)))
        if finite_fraction < 1.0:
            reasons.append("external_rc_nonfinite")
    else:
        finite_fraction = 0.0
        reasons.append("external_rc_unavailable")

    stage_codes = np.asarray(
        stage_audit.get("interval_stages", []),
        dtype=np.int16,
    )
    rc_finite_mask = np.isfinite(rc_values)
    if rc_finite_mask.size != stage_codes.size:
        rc_finite_mask = np.zeros(stage_codes.size, dtype=bool)
    rc_support = {
        "A_n3": int(
            np.sum(rc_finite_mask & (stage_codes == protocol.n3_code))
        ),
        "B_rem": int(
            np.sum(rc_finite_mask & (stage_codes == protocol.rem_code))
        ),
        "interior_n2": int(
            np.sum(rc_finite_mask & (stage_codes == protocol.n2_code))
        ),
    }
    unique_reasons = list(dict.fromkeys(reasons))
    return {
        "participant_id": pair["participant_id"],
        "pid": pid,
        "session": pair["session"],
        "task": pair["task"],
        "run": pair["run"],
        "psg_path": str(psg_path),
        "headband_path": str(headband_path),
        "psg_duration_sec": psg_duration,
        "headband_duration_sec": headband_duration,
        "psg_sampling_frequency_hz": psg_sampling_frequency,
        "headband_sampling_frequency_hz": headband_sampling_frequency,
        "headband_psg_alignment_valid": bool(
            headband_path.exists()
            and set(protocol.rc_channels).issubset(channel_names)
            and "alignment_invalid_or_unproven" not in unique_reasons
            and "sampling_frequency_missing" not in unique_reasons
            and "sampling_frequency_mismatch" not in unique_reasons
        ),
        "external_rc_available": bool(rc_meta.get("available", False)),
        "external_rc_finite_fraction": finite_fraction,
        "external_rc_support": rc_support,
        "rc_metadata": rc_meta,
        "stage_audit": stage_audit,
        "segments": segments,
        "eligible_segments": int(
            sum(
                item["external_rc_finite"]
                for item in segments
            )
        ),
        "failure_reasons": unique_reasons,
    }


def _eligible_records_and_segments(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep all finite-RC outcomes, including competing exits and censoring."""
    eligible_records = [
        record
        for record in records
        if not record["failure_reasons"]
        and record["eligible_segments"] > 0
    ]
    segments = [
        segment
        for record in eligible_records
        for segment in record["segments"]
        if segment["external_rc_finite"]
    ]
    return eligible_records, segments


def run_audit(
    *,
    raw_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Run and serialize the source-level OD-SLP-000 audit."""
    protocol = SleepFirstHitProtocol()
    protocol_path = (
        Path(__file__).resolve().parents[2]
        / "project"
        / "orthagonal_axis"
        / "od_slp_000_preregistration.md"
    )
    pairs = _pair_acquisitions(raw_root)
    pid_map = _load_pid_map(raw_root)
    records = [
        _audit_pair(pair, pid_map=pid_map, protocol=protocol) for pair in pairs
    ]
    eligible_records, segments = _eligible_records_and_segments(records)
    pids = {str(record["pid"]) for record in eligible_records if record["pid"]}
    early_count = sum(
        segment["night_stratum"] == "early_night" for segment in segments
    )
    late_count = sum(
        segment["night_stratum"] == "late_night" for segment in segments
    )
    outcome_counts = Counter(
        segment["outcome"]
        for record in records
        for segment in record["segments"]
    )
    failure_counts = Counter(
        reason for record in records for reason in record["failure_reasons"]
    )
    split_assignments = {str(record["pid"]): "all_data" for record in records if record["pid"]}
    all_stage_audits = [record["stage_audit"] for record in records]
    all_rc_values = [
        record["external_rc_finite_fraction"]
        for record in records
        if record["external_rc_available"]
    ]
    alignment_count = sum(
        record["headband_psg_alignment_valid"] for record in records
    )
    stage_count = sum(
        record["stage_audit"].get("stage_hum_available", False)
        for record in records
    )
    dense_count = sum(
        record["stage_audit"].get("dense_grid_available", False)
        for record in records
    )
    eligibility_reasons: list[str] = []
    if not records:
        eligibility_reasons.append("no_psg_recordings")
    if len(segments) < protocol.minimum_eligible_segments:
        eligibility_reasons.append("insufficient_eligible_segments")
    if len(pids) < protocol.minimum_eligible_pids:
        eligibility_reasons.append("insufficient_eligible_pids")
    if early_count < protocol.minimum_stratum_segments:
        eligibility_reasons.append("insufficient_early_night_support")
    if late_count < protocol.minimum_stratum_segments:
        eligibility_reasons.append("insufficient_late_night_support")
    n3_total = sum(
        audit.get("n3_core_windows", 0) for audit in all_stage_audits
    )
    rem_total = sum(
        audit.get("rem_core_windows", 0) for audit in all_stage_audits
    )
    n2_total = sum(
        audit.get("n2_interior_windows", 0) for audit in all_stage_audits
    )
    if n3_total == 0:
        eligibility_reasons.append("n3_support_absent")
    if rem_total == 0:
        eligibility_reasons.append("rem_support_absent")
    if n2_total == 0:
        eligibility_reasons.append("n2_interior_support_absent")
    for reason in (
        "pid_missing",
        "missing_stage_hum",
        "alignment_invalid_or_unproven",
        "headband_missing",
        "headband_channels_incomplete",
        "stage_intervals_not_positive_duration",
        "stage_grid_gap",
        "stage_grid_overlap",
        "stage_grid_does_not_start_at_zero",
        "stage_grid_does_not_cover_recording",
        "stage_grid_exceeds_recording",
        "stage_hum_empty",
        "stage_hum_unmapped",
        "sampling_frequency_missing",
        "sampling_frequency_mismatch",
        "external_rc_unavailable",
        "external_rc_nonfinite",
    ):
        if failure_counts[reason]:
            eligibility_reasons.append(reason)
    if any(
        reason.startswith("headband_rc_read_failed:")
        for reason in failure_counts
    ):
        eligibility_reasons.append("headband_rc_read_failed")
    subject_split_leakage = pid_split_leaks(split_assignments)
    if subject_split_leakage:
        eligibility_reasons.append("subject_split_leakage")

    result = {
        "schema": "mndm.od_slp_000_first_hit_eligibility.v1",
        "dataset": "ds005555",
        "protocol_id": protocol.protocol_id,
        "protocol_path": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "raw_root": str(raw_root),
        "protocol": protocol.as_dict(),
        "inventory": {
            "recordings_total": len(records),
            "unique_pid_total": len(
                {str(record["pid"]) for record in records if record["pid"]}
            ),
            "psg_recordings": len(records),
            "headband_recordings": sum(
                "headband_missing" not in record["failure_reasons"]
                for record in records
            ),
            "paired_nights": sum(
                "headband_missing" not in record["failure_reasons"]
                for record in records
            ),
            "headband_psg_alignment_valid": alignment_count,
            "stage_hum_available": stage_count,
            "dense_grid_available": dense_count,
            "pid_map_available": bool(pid_map),
        },
        "support": {
            "n3_core_windows": sum(
                audit.get("n3_core_windows", 0) for audit in all_stage_audits
            ),
            "rem_core_windows": sum(
                audit.get("rem_core_windows", 0) for audit in all_stage_audits
            ),
            "n2_interior_windows": sum(
                audit.get("n2_interior_windows", 0) for audit in all_stage_audits
            ),
            "candidate_n2_blocks": sum(
                audit.get("candidate_n2_blocks", 0) for audit in all_stage_audits
            ),
            **dict(outcome_counts),
            "external_rc_available": sum(
                record["external_rc_available"] for record in records
            ),
            "external_rc_finite_fraction_mean": (
                float(np.mean(all_rc_values)) if all_rc_values else 0.0
            ),
            "external_rc_support_A": sum(
                record["external_rc_support"]["A_n3"] for record in records
            ),
            "external_rc_support_B": sum(
                record["external_rc_support"]["B_rem"] for record in records
            ),
            "external_rc_support_interior": sum(
                record["external_rc_support"]["interior_n2"]
                for record in records
            ),
            "eligible_segments_total": len(segments),
            "eligible_pid_total": len(pids),
            "early_night_support": early_count,
            "late_night_support": late_count,
            "failure_reason_counts": dict(failure_counts),
        },
        "records": records,
        "eligibility_decision": {
            "committor_eligibility": (
                "TESTABLE" if not eligibility_reasons else "NOT_TESTABLE"
            ),
            "reasons": list(dict.fromkeys(eligibility_reasons)),
        },
        "fail_closed_assertions": {
            "committor_values_computed": False,
            "qualified_overlay_used": False,
            "stage_ai_used_as_ground_truth": False,
            "rc_built_from_mnps_or_stage": False,
            "competing_exits_dropped_by_future_outcome": False,
            "subject_split_leakage": subject_split_leakage,
            "subject_split_evaluated": False,
            "committor_estimator_invoked": False,
        },
        "claim_boundary": (
            "Source-level BOAS eligibility audit only. No committor, drift, "
            "diffusion, HDF5 overlay, or biological transition claim was made."
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(raw_root=args.raw_root, output_path=args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "recordings_total": result["inventory"]["recordings_total"],
                "paired_nights": result["inventory"]["paired_nights"],
                "eligible_segments_total": result["support"][
                    "eligible_segments_total"
                ],
                "committor_eligibility": result["eligibility_decision"][
                    "committor_eligibility"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
